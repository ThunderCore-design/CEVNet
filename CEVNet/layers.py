from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        channel_mean = feature.mean(dim=1, keepdim=True)
        channel_var = (feature - channel_mean).pow(2).mean(dim=1, keepdim=True)
        feature = (feature - channel_mean) / torch.sqrt(channel_var + self.eps)
        feature = self.weight[:, None, None] * feature
        feature = feature + self.bias[:, None, None]
        return feature


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        use_act: bool = True,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.conv(feature)
        feature = self.bn(feature)
        feature = self.act(feature)
        return feature


class FeatureProjection(nn.Module):
    """Project one backbone stage without changing its spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = LayerNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, stage_feature: torch.Tensor) -> torch.Tensor:
        projected_feature = self.proj(stage_feature)
        projected_feature = self.norm(projected_feature)
        projected_feature = self.act(projected_feature)
        return projected_feature


class SinePositionEncoding2d(nn.Module):
    """Deterministic 2D sine-cosine position encoding in BCHW layout."""

    def __init__(
        self,
        channels: int,
        temperature: float = 10000.0,
    ) -> None:
        super().__init__()

        if channels % 4 != 0:
            raise ValueError(
                f"Position encoding channels must be divisible by 4, got {channels}"
            )
        self.channels = channels
        self.temperature = float(temperature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError(
                f"Position encoding expected C={self.channels}, got C={channels}"
            )

        dtype = feature.dtype
        device = feature.device
        axis_channels = channels // 4
        frequency = torch.arange(
            axis_channels,
            device=device,
            dtype=torch.float32,
        )
        frequency = self.temperature ** (frequency / max(axis_channels - 1, 1))

        y_position = torch.arange(
            height,
            device=device,
            dtype=torch.float32,
        )
        x_position = torch.arange(
            width,
            device=device,
            dtype=torch.float32,
        )
        y_position = y_position / max(height - 1, 1)
        x_position = x_position / max(width - 1, 1)

        y_phase = y_position[:, None] / frequency[None, :]
        x_phase = x_position[:, None] / frequency[None, :]

        # H, C/2 and W, C/2
        y_encoding = torch.cat(
            [torch.sin(y_phase), torch.cos(y_phase)],
            dim=1,
        )
        x_encoding = torch.cat(
            [torch.sin(x_phase), torch.cos(x_phase)],
            dim=1,
        )

        # H, C/2 -> C/2, H, W; W, C/2 -> C/2, H, W.
        y_encoding = y_encoding.transpose(0, 1).unsqueeze(2).expand(-1, -1, width)
        x_encoding = x_encoding.transpose(0, 1).unsqueeze(1).expand(-1, height, -1)
        position_encoding = torch.cat(
            [y_encoding, x_encoding],
            dim=0,
        )
        # C, H, W -> 1, C, H, W -> B, C, H, W.
        position_encoding = position_encoding.unsqueeze(0)
        position_encoding = position_encoding.expand(batch_size, -1, -1, -1)
        return position_encoding.to(dtype=dtype)


class BinaryMaskPredictionHead(nn.Module):
    """Generate one binary side logit."""

    def __init__(
        self,
        channels: int,
        use_refine: bool,
    ) -> None:
        super().__init__()

        if use_refine:
            hidden_channels = max(channels // 2, 32)
            self.predictor = nn.Sequential(
                ConvBNAct(
                    in_channels=channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    hidden_channels,
                    1,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )
        else:
            self.predictor = nn.Conv2d(
                channels,
                1,
                kernel_size=3,
                padding=1,
                bias=True,
            )

    def forward(
        self,
        feature: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        # B, C, H, W -> B, 1, H, W
        logit = self.predictor(feature)
        if logit.shape[-2:] != output_size:
            logit = F.interpolate(
                logit,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return logit
