from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BinaryMaskPredictionHead, ConvBNAct
from .protected_modules import (
    CompetitiveMaskRefinementBlock,
    CompetitiveMaskSeedBlock,
    CompetitiveSemanticGuidanceBlock,
)


class PromptProgressiveMaskDecoder(nn.Module):
    """Deep-to-shallow uncertainty-aware competitive mask decoder.

    The projected inputs are interpreted as F4, F3, F2 and F1. F1 guides
    F2/F3/F4 through competitive semantic guidance, F2 and F3 continue
    passing context to shallower levels, and mask features are refined from
    p1 to p4 through level context, history context, mask evidence and
    backbone-final prompt context. The mask evidence branch uses
    previous-side probability, entropy and local deviation to highlight
    ambiguous camouflaged regions without adding another PRS stack.
    """

    def __init__(
        self,
        channels: int,
        num_outputs: int = 4,
        hidden_channels: int = 192,
        output_size: Optional[int] = None,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        if num_outputs != 4:
            raise ValueError(f"Progressive decoder expects four outputs, got {num_outputs}")

        self.num_outputs = int(num_outputs)
        self.hidden_channels = int(hidden_channels)
        self.output_size = output_size

        self.level_projections = nn.ModuleList(
            [
                ConvBNAct(
                    in_channels=channels,
                    out_channels=self.hidden_channels,
                    kernel_size=1,
                    padding=0,
                )
                for _ in range(self.num_outputs)
            ]
        )
        self.level_embeddings = nn.Parameter(
            torch.zeros(self.num_outputs, 1, self.hidden_channels, 1, 1)
        )
        nn.init.normal_(self.level_embeddings, mean=0.0, std=0.02)

        self.backbone_prompt_projection = ConvBNAct(
            in_channels=channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            padding=0,
        )

        self.f2_guidance = CompetitiveSemanticGuidanceBlock(
            channels=self.hidden_channels,
            num_sources=1,
            num_heads=num_heads,
            residual_init=0.12,
        )
        self.f3_guidance = CompetitiveSemanticGuidanceBlock(
            channels=self.hidden_channels,
            num_sources=2,
            num_heads=num_heads,
            residual_init=0.10,
        )
        self.f4_guidance = CompetitiveSemanticGuidanceBlock(
            channels=self.hidden_channels,
            num_sources=3,
            num_heads=num_heads,
            residual_init=0.08,
        )

        self.seed_refinement = CompetitiveMaskSeedBlock(
            channels=self.hidden_channels,
            num_heads=num_heads,
            residual_init=0.10,
        )
        self.p2_refinement = CompetitiveMaskRefinementBlock(
            channels=self.hidden_channels,
            num_heads=num_heads,
            residual_init=0.12,
        )
        self.p3_refinement = CompetitiveMaskRefinementBlock(
            channels=self.hidden_channels,
            num_heads=num_heads,
            residual_init=0.10,
        )
        self.p4_refinement = CompetitiveMaskRefinementBlock(
            channels=self.hidden_channels,
            num_heads=num_heads,
            residual_init=0.08,
        )

        self.p1_head = BinaryMaskPredictionHead(
            channels=self.hidden_channels,
            use_refine=False,
        )
        self.p2_head = BinaryMaskPredictionHead(
            channels=self.hidden_channels,
            use_refine=False,
        )
        self.p3_head = BinaryMaskPredictionHead(
            channels=self.hidden_channels,
            use_refine=False,
        )
        self.p4_head = BinaryMaskPredictionHead(
            channels=self.hidden_channels,
            use_refine=True,
        )

    def _resolve_output_size(self, image_pe: torch.Tensor) -> Tuple[int, int]:
        if self.output_size is not None:
            output_size = int(self.output_size)
            if output_size <= 0:
                raise ValueError(f"output_size must be positive, got {output_size}")
            return output_size, output_size

        return (
            int(image_pe.shape[-2]) * 4,
            int(image_pe.shape[-1]) * 4,
        )

    def _project_level(
        self,
        feature: torch.Tensor,
        level_index: int,
    ) -> torch.Tensor:
        projected_feature = self.level_projections[level_index](feature)
        projected_feature = projected_feature + self.level_embeddings[level_index]
        return projected_feature

    def forward(
        self,
        multi_scale_features: Sequence[torch.Tensor],
        decoder_feature: torch.Tensor,
        image_pe: torch.Tensor,
        backbone_prompt_feature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        decoder_inputs = tuple(multi_scale_features) + (decoder_feature,)
        if len(decoder_inputs) != self.num_outputs:
            raise ValueError(
                "decoder input count must match four-side output count, "
                f"got {len(decoder_inputs)} and {self.num_outputs}"
            )

        output_size = self._resolve_output_size(image_pe)

        # F4/F3/F2 retain stage-1/2/3 resolutions; F1 is the fused deep embedding.
        # Each tensor is B, C_decoder, H_i, W_i.
        f4_input, f3_input, f2_input, f1_input = decoder_inputs
        f4_feature = self._project_level(
            feature=f4_input,
            level_index=0,
        )
        f3_feature = self._project_level(
            feature=f3_input,
            level_index=1,
        )
        f2_feature = self._project_level(
            feature=f2_input,
            level_index=2,
        )
        f1_feature = self._project_level(
            feature=f1_input,
            level_index=3,
        )

        if backbone_prompt_feature is None:
            backbone_prompt_feature = f1_input
        backbone_prompt_feature = backbone_prompt_feature.to(
            device=f1_feature.device,
            dtype=f1_feature.dtype,
        )
        prompt_feature = self.backbone_prompt_projection(backbone_prompt_feature)

        f2_fuse = self.f2_guidance(
            target_feature=f2_feature,
            source_features=(f1_feature,),
            prompt_feature=prompt_feature,
        )
        f3_fuse = self.f3_guidance(
            target_feature=f3_feature,
            source_features=(f2_fuse, f1_feature),
            prompt_feature=prompt_feature,
        )
        f4_fuse = self.f4_guidance(
            target_feature=f4_feature,
            source_features=(f3_fuse, f2_fuse, f1_feature),
            prompt_feature=prompt_feature,
        )

        # F1 -> p1. This is the coarsest prediction seed.
        p1_feature = self.seed_refinement(
            seed_feature=f1_feature,
            prompt_feature=prompt_feature,
        )
        p1_logit = self.p1_head(
            feature=p1_feature,
            output_size=output_size,
        )

        # p1 feature + F2_fuse + p1 mask evidence -> p2.
        p2_feature = self.p2_refinement(
            level_feature=f2_fuse,
            previous_feature=p1_feature,
            previous_logit=p1_logit,
            prompt_feature=prompt_feature,
        )
        p2_logit = self.p2_head(
            feature=p2_feature,
            output_size=output_size,
        )

        # p2 feature + F3_fuse + p2 mask evidence -> p3.
        p3_feature = self.p3_refinement(
            level_feature=f3_fuse,
            previous_feature=p2_feature,
            previous_logit=p2_logit,
            prompt_feature=prompt_feature,
        )
        p3_logit = self.p3_head(
            feature=p3_feature,
            output_size=output_size,
        )

        # p3 feature + F4_fuse + p3 mask evidence -> p4.
        p4_feature = self.p4_refinement(
            level_feature=f4_fuse,
            previous_feature=p3_feature,
            previous_logit=p3_logit,
            prompt_feature=prompt_feature,
        )
        p4_logit = self.p4_head(
            feature=p4_feature,
            output_size=output_size,
        )

        # Four raw logits, each B, 1, H_out, W_out; no sigmoid here.
        return p1_logit, p2_logit, p3_logit, p4_logit
