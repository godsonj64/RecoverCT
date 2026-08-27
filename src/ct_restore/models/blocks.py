from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class GatedDepthwiseBlock(nn.Module):
    """Local residual block with depthwise spatial mixing and a SwiGLU-style gate."""

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.spatial = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.expand = nn.Conv3d(channels, hidden * 2, kernel_size=1)
        self.project = nn.Conv3d(hidden, channels, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spatial(self.norm(x))
        value, gate = self.expand(y).chunk(2, dim=1)
        y = value * torch.nn.functional.silu(gate)
        return x + self.dropout(self.project(y))


class AxialGatedMixer(nn.Module):
    """Linear-cost global-ish context through large, factorized depthwise kernels."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.z = nn.Conv3d(
            channels, channels, (kernel_size, 1, 1), padding=(pad, 0, 0), groups=channels
        )
        self.y = nn.Conv3d(
            channels, channels, (1, kernel_size, 1), padding=(0, pad, 0), groups=channels
        )
        self.x = nn.Conv3d(
            channels, channels, (1, 1, kernel_size), padding=(0, 0, pad), groups=channels
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(4, channels // 8), 1),
            nn.SiLU(),
            nn.Conv3d(max(4, channels // 8), channels * 3, 1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        gates = self.gate(y).chunk(3, dim=1)
        y = self.z(y) * gates[0] + self.y(y) * gates[1] + self.x(y) * gates[2]
        return x + self.project(y)


class Stage(nn.Sequential):
    def __init__(self, channels: int, blocks: int, dropout: float = 0.0) -> None:
        super().__init__(*[GatedDepthwiseBlock(channels, dropout=dropout) for _ in range(blocks)])


class Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.GroupNorm(_groups(in_channels), in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )


class UpsampleFuse(nn.Module):
    """Resize-then-convolve upsampling.

    A strided ``ConvTranspose3d`` gives every sub-voxel position its own kernel, so a
    spatially constant input still leaves a period-2 lattice in the output. For a model
    whose output is quantitative HU that lattice is a systematic error, not cosmetic
    texture, so upsampling is done by interpolation followed by an ordinary convolution.
    Interpolating straight to the skip resolution also handles odd sizes without a
    separate correction step.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.fuse = nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        return self.fuse(torch.cat((self.project(x), skip), dim=1))
