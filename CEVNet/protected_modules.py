from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRS(nn.Module):
    """Pseudo-response suppression: B, C, H, W -> B, C, H, W.
    return_aux=True also returns attention_logits, attention_prob and replacement_gate
    maps of shape B, 1, H, W, as consumed by FeatureEncoder.

    Interface only. The original trainable layers and computation are withheld.
    Calling forward raises NotImplementedError; no substitute output is returned.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        cov_dim: int | None = None,
        gamma: float = 1.0,
        delta: float = 1.0,
        eps: float = 1e-6,
        hidden_mediator: int = 16,
        context_sensitivity_weight: float = 1.0,
        intrinsic_saliency_weight: float = 0.0,
        counterfactual_rank: int = 4,
        use_bn: bool = True,
    ) -> None:
        super().__init__()

    def forward(
        self,
        feature: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError(
            "PRS: core implementation is not included in the public architecture release."
        )


class CompetitiveSemanticGuidanceBlock(nn.Module):
    """Cross-scale semantic guidance. Output has the target feature shape B, C, H, W.

    Interface only. The original trainable layers and computation are withheld.
    Calling forward raises NotImplementedError; no substitute output is returned.
    """

    def __init__(
        self,
        channels: int,
        num_sources: int,
        num_heads: int = 8,
        proxy_size: int = 12,
        residual_init: float = 0.1,
        reduction: int = 4,
    ) -> None:
        super().__init__()

    def forward(
        self,
        target_feature: torch.Tensor,
        source_features: Sequence[torch.Tensor],
        prompt_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CompetitiveSemanticGuidanceBlock: core implementation is not included in the public architecture release."
        )


class CompetitiveMaskSeedBlock(nn.Module):
    """Coarse mask feature construction. Output has the seed feature shape B, C, H, W.

    Interface only. The original trainable layers and computation are withheld.
    Calling forward raises NotImplementedError; no substitute output is returned.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        proxy_size: int = 12,
        residual_init: float = 0.1,
        reduction: int = 4,
    ) -> None:
        super().__init__()

    def forward(
        self,
        seed_feature: torch.Tensor,
        prompt_feature: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CompetitiveMaskSeedBlock: core implementation is not included in the public architecture release."
        )


class CompetitiveMaskRefinementBlock(nn.Module):
    """Progressive refinement with prior mask evidence. Output matches level_feature.

    Interface only. The original trainable layers and computation are withheld.
    Calling forward raises NotImplementedError; no substitute output is returned.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        proxy_size: int = 12,
        residual_init: float = 0.1,
        reduction: int = 4,
    ) -> None:
        super().__init__()

    def forward(
        self,
        level_feature: torch.Tensor,
        previous_feature: torch.Tensor,
        previous_logit: torch.Tensor,
        prompt_feature: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CompetitiveMaskRefinementBlock: core implementation is not included in the public architecture release."
        )
