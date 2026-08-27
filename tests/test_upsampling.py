"""Guards on decoder upsampling.

A strided transposed convolution gives each sub-voxel position its own kernel, so a
spatially constant input still leaves a period-2 lattice in the output. On a model that
emits quantitative HU that lattice is a systematic error, so it is asserted against.
"""

import torch
from torch import nn

from ct_restore.models import HybridRestoreNet
from ct_restore.models.blocks import UpsampleFuse

# Normalized [-1, 1] spans the default HU clamp of [-1024, 3071].
HU_PER_UNIT = (3071.0 - (-1024.0)) / 2.0


def _period_two_alternation(volume: torch.Tensor, crop: int = 16) -> float:
    """Largest mean |even - odd| neighbour gap over the three axes, border excluded."""
    inner = volume[crop:-crop, crop:-crop, crop:-crop]
    worst = 0.0
    for axis in range(3):
        moved = inner.movedim(axis, 0)
        half = moved.shape[0] // 2
        gap = (moved[::2][:half] - moved[1::2][:half]).abs().mean().item()
        worst = max(worst, gap)
    return worst


def test_constant_input_leaves_no_checkerboard_lattice() -> None:
    torch.manual_seed(0)
    model = HybridRestoreNet(base_channels=8, levels=3, blocks_per_level=1).eval()
    constant = torch.full((1, 3, 48, 80, 80), 0.3)
    with torch.inference_mode():
        output = model(constant)["corrected"][0, 0]
    alternation = _period_two_alternation(output)
    # ConvTranspose3d(kernel_size=2, stride=2) measures ~2.8e-2 here (~57 HU).
    assert alternation < 5.0e-3, f"period-2 lattice {alternation * HU_PER_UNIT:.1f} HU"


def test_upsampling_is_not_strided_transposed_convolution() -> None:
    block = UpsampleFuse(8, 4, 4)
    assert not any(isinstance(m, nn.ConvTranspose3d) for m in block.modules())


def test_upsample_fuse_matches_odd_skip_resolution() -> None:
    block = UpsampleFuse(6, 4, 4).eval()
    coarse = torch.randn(1, 6, 3, 5, 7)
    skip = torch.randn(1, 4, 5, 11, 13)
    with torch.inference_mode():
        assert block(coarse, skip).shape == (1, 4, 5, 11, 13)
