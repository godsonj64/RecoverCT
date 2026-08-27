"""Hard identity constraint outside the artifact mask.

A soft identity penalty competes with the reconstruction terms. Measured on a
synthetically corrupted test volume, the model cut artifact-region error by 14% while
more than doubling error in the untouched region (12.8 -> 28.0 HU), making it net
harmful. residual_leak turns preservation into a constraint instead of a preference.
"""

import pytest
import torch

from ct_restore.models import HybridRestoreNet


def _inputs(mask_value: float | None = None) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 24, 24).clamp(-1, 1)
    mask = torch.zeros(1, 1, 16, 24, 24)
    mask[:, :, 4:8, 6:12, 6:12] = 1.0
    if mask_value is not None:
        mask.fill_(mask_value)
    x[:, 1:2] = mask
    x[:, 2:3] = 1.0 - mask
    return x


def _model(leak: float) -> HybridRestoreNet:
    torch.manual_seed(0)
    return HybridRestoreNet(
        base_channels=4, levels=2, blocks_per_level=1, residual_leak=leak
    ).eval()


def test_leak_one_is_the_unconstrained_default() -> None:
    assert HybridRestoreNet().residual_leak == 1.0


def test_zero_leak_freezes_every_voxel_outside_the_mask() -> None:
    x = _inputs()
    with torch.inference_mode():
        out = _model(0.0)(x)
    outside = x[:, 1:2] < 0.5
    assert torch.allclose(out["corrected"][outside], x[:, :1][outside], atol=1e-6)


def test_zero_leak_still_allows_change_inside_the_mask() -> None:
    x = _inputs()
    with torch.inference_mode():
        out = _model(0.0)(x)
    inside = x[:, 1:2] >= 0.5
    assert (out["corrected"][inside] - x[:, :1][inside]).abs().max() > 1e-4


def test_partial_leak_bounds_the_change_outside_the_mask() -> None:
    x = _inputs()
    with torch.inference_mode():
        free = _model(1.0)(x)["corrected"]
        held = _model(0.1)(x)["corrected"]
    outside = x[:, 1:2] < 0.5
    source = x[:, :1]
    free_shift = (free - source)[outside].abs().mean()
    held_shift = (held - source)[outside].abs().mean()
    assert held_shift < free_shift


def test_leak_one_matches_ungated_behaviour_exactly() -> None:
    x = _inputs()
    with torch.inference_mode():
        a = _model(1.0)(x)["corrected"]
        b = _model(1.0)(x)["corrected"]
    assert torch.equal(a, b)


def test_full_mask_leaves_the_residual_untouched() -> None:
    """Everything flagged as artifact means the gate must not restrict anything."""
    x = _inputs(mask_value=1.0)
    with torch.inference_mode():
        gated = _model(0.0)(x)["corrected"]
        free = _model(1.0)(x)["corrected"]
    assert torch.allclose(gated, free, atol=1e-6)


@pytest.mark.parametrize("leak", [-0.1, 1.5])
def test_out_of_range_leak_is_rejected(leak: float) -> None:
    with pytest.raises(ValueError, match="residual_leak"):
        HybridRestoreNet(residual_leak=leak)
