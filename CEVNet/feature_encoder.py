from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FeatureProjection, LayerNorm2d, SinePositionEncoding2d
from .protected_modules import PRS


class PyramidEmbeddingFusion(nn.Module):
    """Add position/level encodings and adapt pyramid features for decoding."""

    def __init__(
        self,
        channels: int,
        num_levels: int = 3,
    ) -> None:
        super().__init__()

        self.position_encoding = SinePositionEncoding2d(channels)
        self.level_embeddings = nn.Parameter(
            torch.zeros(num_levels, 1, channels, 1, 1)
        )
        nn.init.normal_(self.level_embeddings, mean=0.0, std=0.02)

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                channels * num_levels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(channels),
            nn.GELU(),
        )

    def forward(
        self,
        multi_scale_features: Sequence[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(multi_scale_features) != self.level_embeddings.shape[0]:
            raise ValueError(
                "Feature count must match level embedding count, "
                f"got {len(multi_scale_features)} and {self.level_embeddings.shape[0]}"
            )

        encoded_features = []
        aligned_features = []

        for level_index, feature in enumerate(multi_scale_features):
            position_encoding = self.position_encoding(feature)
            level_encoding = self.level_embeddings[level_index]
            encoded_feature = feature + position_encoding + level_encoding
            encoded_features.append(encoded_feature)

            if encoded_feature.shape[-2:] != target_size:
                aligned_feature = F.interpolate(
                    encoded_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                aligned_feature = encoded_feature
            aligned_features.append(aligned_feature)

        # Three B, C, H, W tensors -> B, 3*C, H, W -> B, C, H, W
        concatenated_feature = torch.cat(aligned_features, dim=1)
        fused_feature = self.output_projection(concatenated_feature)
        return fused_feature, tuple(encoded_features)


class FeatureEncoder(nn.Module):
    """Calibrate the deepest pyramid feature and fuse Stage 1-3 features.

    Inputs:
        Stage 1: [B, C1, H/4,  W/4]
        Stage 2: [B, C2, H/8,  W/8]
        Stage 3: [B, C3, H/16, W/16]
        Stage 4 embedding: [B, C_decoder, H/16, W/16]

    Outputs:
        decoder_feature: [B, C_decoder, H/16, W/16] embedding with pyramid residual.
        encoded_features: three encoded tensors with original pyramid resolutions.
        aux_outputs: PRS attention logits/probability maps for auxiliary supervision.
    """

    def __init__(
        self,
        stage_channels: Sequence[int] = (64, 128, 160),
        decoder_channels: int = 256,
        prs_cov_dims: Sequence[int | None] | None = None,
        use_prs: bool = True,
        prs_context_sensitivity_weight: float = 1.0,
        prs_intrinsic_saliency_weight: float = 0.0,
        prs_counterfactual_rank: int = 4,
        prs_stages: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        if len(stage_channels) != 3:
            raise ValueError(
                f"FeatureEncoder expects three pyramid stages, got {len(stage_channels)}"
            )

        if prs_cov_dims is None:
            prs_cov_dims = [None for _ in stage_channels]
        if len(prs_cov_dims) != len(stage_channels):
            raise ValueError(
                "prs_cov_dims must match stage_channels, "
                f"got {len(prs_cov_dims)} and {len(stage_channels)}"
            )
        if prs_stages is None:
            prs_stages = (1, 2, 3)
        invalid_stages = [
            int(stage_id)
            for stage_id in prs_stages
            if int(stage_id) < 1 or int(stage_id) > len(stage_channels)
        ]
        if invalid_stages:
            raise ValueError(
                "prs_stages must contain 1-based stage ids in "
                f"[1, {len(stage_channels)}], got {invalid_stages}"
            )

        self.use_prs = bool(use_prs)
        self.prs_stage_indices = tuple(
            sorted({int(stage_id) - 1 for stage_id in prs_stages})
        )
        self.prs_modules = nn.ModuleList()
        for stage_index, (in_channels, cov_dim) in enumerate(
            zip(stage_channels, prs_cov_dims)
        ):
            if self._should_apply_prs(stage_index):
                self.prs_modules.append(
                    PRS(
                        in_channels=in_channels,
                        cov_dim=cov_dim,
                        context_sensitivity_weight=prs_context_sensitivity_weight,
                        intrinsic_saliency_weight=prs_intrinsic_saliency_weight,
                        counterfactual_rank=prs_counterfactual_rank,
                    )
                )
            else:
                self.prs_modules.append(nn.Identity())
        self.stage_projections = nn.ModuleList(
            [
                FeatureProjection(
                    in_channels=in_channels,
                    out_channels=decoder_channels,
                )
                for in_channels in stage_channels
            ]
        )
        self.pyramid_fusion = PyramidEmbeddingFusion(
            channels=decoder_channels,
            num_levels=3,
        )

    def _should_apply_prs(self, stage_index: int) -> bool:
        return self.use_prs and stage_index in self.prs_stage_indices

    def forward(
        self,
        stage_features: Sequence[torch.Tensor],
        image_embedding: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Dict[str, Tuple[torch.Tensor, ...] | Tuple[int, ...]],
    ]:
        if len(stage_features) < 3:
            raise ValueError(
                f"Expected at least three backbone stage features, got {len(stage_features)}"
            )
        if image_embedding.dim() != 4:
            raise ValueError(
                "image_embedding must be [B, C, H, W], "
                f"got shape={tuple(image_embedding.shape)}"
            )

        projected_features = []
        attention_logits = []
        attention_probs = []
        replacement_gates = []

        for stage_index, (stage_feature, projection, prs) in enumerate(
            zip(
                stage_features[:3],
                self.stage_projections,
                self.prs_modules,
            )
        ):
            if self._should_apply_prs(stage_index):
                enhanced_feature, prs_aux = prs(stage_feature, return_aux=True)
                attention_logits.append(prs_aux["attention_logits"])
                attention_probs.append(prs_aux["attention_prob"])
                replacement_gates.append(prs_aux["replacement_gate"])
            else:
                enhanced_feature = prs(stage_feature)
            projected_feature = projection(enhanced_feature)
            projected_features.append(projected_feature)

        target_size = (
            int(image_embedding.shape[-2]),
            int(image_embedding.shape[-1]),
        )
        fused_feature, encoded_features = self.pyramid_fusion(
            multi_scale_features=projected_features,
            target_size=target_size,
        )

        # Directly inject encoded multi-scale features into the image embedding.
        decoder_feature = image_embedding + fused_feature
        # ``attention_logits`` follows the enabled PRS stage order, so the
        # auxiliary loss can supervise one or more foreground-preservation maps.
        aux_outputs = {
            "attention_logits": tuple(attention_logits),
            "all_attention_logits": tuple(attention_logits),
            "attention_probs": tuple(attention_probs),
            "replacement_gates": tuple(replacement_gates),
            "prs_stage_indices": self.prs_stage_indices,
            "prs_stage_ids": tuple(
                stage_index + 1 for stage_index in self.prs_stage_indices
            ),
        }
        return decoder_feature, encoded_features, aux_outputs
