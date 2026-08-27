from __future__ import annotations

import torch
from torch import nn

from ct_restore.models.blocks import AxialGatedMixer, Downsample, Stage, UpsampleFuse


class HybridRestoreNet(nn.Module):
    """Compact 3D U-Net with gated CNN detail paths and axial context mixing.

    Inputs are normalized CT, a suspected-artifact mask, and known-voxel confidence.
    Outputs are a bounded residual, corrected CT, log variance, and a refined mask logit.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 24,
        levels: int = 4,
        blocks_per_level: int = 2,
        dropout: float = 0.0,
        max_residual: float = 2.0,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        channels = [base_channels * (2**i) for i in range(levels)]
        self.max_residual = max_residual
        self.stem = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        self.encoders = nn.ModuleList([Stage(c, blocks_per_level, dropout) for c in channels])
        self.downsamples = nn.ModuleList(
            [Downsample(channels[i], channels[i + 1]) for i in range(levels - 1)]
        )
        self.context = nn.Sequential(
            AxialGatedMixer(channels[-1], 7),
            AxialGatedMixer(channels[-1], 11),
        )
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.upsamples.append(UpsampleFuse(channels[i], channels[i - 1], channels[i - 1]))
            self.decoders.append(Stage(channels[i - 1], blocks_per_level, dropout))
        self.head = nn.Conv3d(channels[0], 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5 or x.shape[1] < 1:
            raise ValueError("Expected input shape [B, C, D, H, W]")
        source = x[:, :1]
        y = self.stem(x)
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if index < len(self.downsamples):
                y = self.downsamples[index](y)
        y = self.context(y)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips[:-1]), strict=True
        ):
            y = decoder(upsample(y, skip))
        residual_raw, log_variance, artifact_logit = self.head(y).chunk(3, dim=1)
        residual = torch.tanh(residual_raw) * self.max_residual
        corrected = torch.clamp(source + residual, -1.0, 1.0)
        return {
            "corrected": corrected,
            "residual": residual,
            "log_variance": torch.clamp(log_variance, -8.0, 4.0),
            "artifact_logit": artifact_logit,
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
