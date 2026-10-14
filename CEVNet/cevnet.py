from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoder import FeatureEncoder
from .layers import ConvBNAct
from .mask_decoder import PromptProgressiveMaskDecoder
from .protected_modules import PRS


FAIR_ENCODER_STAGE_CHANNELS: Tuple[int, int, int] = (72, 144, 288)


class BackboneStageChannelProjector(nn.Module):
    """Normalize backbone stage channels before the PRS FeatureEncoder.

    Different backbone families expose very different stage widths. Without this
    adapter, the downstream PRS and projection layers become much larger for
    backbones such as ResNet50, so a backbone ablation also changes the encoder
    capacity. This module keeps the encoder input widths fixed and makes the
    comparison closer to a pure backbone swap.
    """

    def __init__(
        self,
        input_stage_channels: Sequence[int],
        output_stage_channels: Sequence[int] = FAIR_ENCODER_STAGE_CHANNELS,
    ) -> None:
        super().__init__()

        if len(input_stage_channels) != len(output_stage_channels):
            raise ValueError(
                "input_stage_channels and output_stage_channels must have the "
                f"same length, got {len(input_stage_channels)} and "
                f"{len(output_stage_channels)}"
            )

        self.input_stage_channels = tuple(int(channels) for channels in input_stage_channels)
        self.output_stage_channels = tuple(int(channels) for channels in output_stage_channels)
        self.projectors = nn.ModuleList()
        for input_channels, output_channels in zip(
            self.input_stage_channels,
            self.output_stage_channels,
        ):
            if input_channels == output_channels:
                projector = nn.Identity()
            else:
                projector = ConvBNAct(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1,
                    padding=0,
                )
            self.projectors.append(projector)

    def forward(
        self,
        stage_features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        if len(stage_features) < len(self.projectors):
            raise ValueError(
                "BackboneStageChannelProjector expects at least "
                f"{len(self.projectors)} stage features, got {len(stage_features)}"
            )

        projected_features = []
        for stage_feature, projector in zip(stage_features, self.projectors):
            # B, C_i, H_i, W_i -> B, C_fair_i, H_i, W_i
            projected_feature = projector(stage_feature)
            projected_features.append(projected_feature)
        return tuple(projected_features)


class CEVNet(nn.Module):
    """Public architecture view of the active CEVNet path.

    This is a reduced constructor, not a checkpoint-compatible inference release.
    The caller provides a backbone adapter with forward_intermediates(image).
    Inputs must already be normalized and resized by the caller.
    PRS and competitive decoder operations are intentionally unavailable.
    """

    def __init__(
        self,
        backbone: nn.Module,
        stage_channels: Sequence[int],
        decoder_channels: int = 256,
        mask_output_size: int = 384,
        enable_stage4_prs: bool = False,
    ) -> None:
        super().__init__()

        if not callable(getattr(backbone, "forward_intermediates", None)):
            raise TypeError("backbone must expose forward_intermediates(image).")
        if len(stage_channels) != 4 or any(channels <= 0 for channels in stage_channels):
            raise ValueError("stage_channels must specify four positive channel counts.")
        if decoder_channels <= 0 or decoder_channels % 4 != 0:
            raise ValueError("decoder_channels must be positive and divisible by four.")
        if mask_output_size <= 0 or mask_output_size % 4 != 0:
            raise ValueError("mask_output_size must be positive and divisible by four.")

        self.backbone = backbone
        self.backbone_stage_projector = BackboneStageChannelProjector(
            input_stage_channels=stage_channels[:3],
        )
        self.stage4_prs = (
            PRS(in_channels=decoder_channels)
            if enable_stage4_prs
            else nn.Identity()
        )
        self.feature_encoder = FeatureEncoder(
            stage_channels=FAIR_ENCODER_STAGE_CHANNELS,
            decoder_channels=decoder_channels,
            use_prs=True,
        )
        self.multi_scale_mask_decoder = PromptProgressiveMaskDecoder(
            channels=decoder_channels,
            output_size=mask_output_size,
        )
        # 1, C_decoder, H_out/4, W_out/4; decoder reads only the spatial size.
        self.register_buffer(
            "image_pe",
            torch.zeros(1, decoder_channels, mask_output_size // 4, mask_output_size // 4),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> Dict[
        str,
        torch.Tensor | Tuple[torch.Tensor, ...]
        | Dict[str, Tuple[torch.Tensor, ...] | Tuple[int, ...]],
    ]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected normalized RGB/pseudo-color input B, 3, H, W.")

        # Four stage tensors: B, C_i, H_i, W_i.
        # The adapter projects stage 4 to B, C_decoder, H_3, W_3.
        stage_features, image_embedding = self.backbone.forward_intermediates(image)
        if len(stage_features) != 4:
            raise ValueError("The backbone adapter must return four feature stages.")
        image_embedding = self.stage4_prs(image_embedding)
        projected_features = self.backbone_stage_projector(stage_features)

        decoder_feature, multi_scale_features, aux_outputs = self.feature_encoder(
            stage_features=projected_features,
            image_embedding=image_embedding,
        )
        side_logits = self.multi_scale_mask_decoder(
            multi_scale_features=multi_scale_features,
            decoder_feature=decoder_feature,
            image_pe=self.image_pe,
            backbone_prompt_feature=image_embedding,
        )

        # Four B, 1, H_out, W_out logits -> B, 1, H_out, W_out summed logit.
        summed_logit = side_logits[0]
        for side_logit in side_logits[1:]:
            summed_logit = summed_logit + side_logit

        return {
            "pred_masks": summed_logit,
            "pred_masks_sum": summed_logit,
            "pred_masks_p4": side_logits[-1],
            "feature_logits": side_logits,
            "multi_scale_decoder_logits": side_logits,
            "aux_outputs": aux_outputs,
            "image_embeddings": image_embedding,
            "decoder_features": decoder_feature,
            "multi_scale_features": multi_scale_features,
            "stage_features": tuple(stage_features),
        }
