import pytest
import torch

pytest.importorskip("nibabel")

from ct_restore.inference import sliding_window_predict  # noqa: E402
from ct_restore.models import HybridRestoreNet  # noqa: E402


def test_sliding_window_pads_channels_safely() -> None:
    model = HybridRestoreNet(base_channels=4, levels=2, blocks_per_level=1).eval()
    inputs = torch.zeros(1, 3, 9, 11, 13)
    inputs[:, 2] = 1.0
    corrected, uncertainty = sliding_window_predict(model, inputs, (16, 16, 16))
    assert corrected.shape == (1, 1, 9, 11, 13)
    assert uncertainty.shape == corrected.shape
    assert torch.isfinite(corrected).all()
    assert torch.isfinite(uncertainty).all()
