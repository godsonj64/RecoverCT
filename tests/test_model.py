import torch

from ct_restore.losses import RestorationLoss
from ct_restore.models import HybridRestoreNet


def test_model_shapes_and_gradients() -> None:
    model = HybridRestoreNet(base_channels=8, levels=3, blocks_per_level=1)
    inputs = torch.randn(1, 3, 16, 24, 24)
    target = torch.zeros(1, 1, 16, 24, 24)
    corrupted = inputs[:, :1].clamp(-1, 1)
    mask = (inputs[:, 1:2] > 0).float()
    outputs = model(inputs)
    assert set(outputs) == {"corrected", "residual", "log_variance", "artifact_logit"}
    assert all(value.shape == target.shape for value in outputs.values())
    loss, terms = RestorationLoss()(outputs, target, corrupted, mask)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_model_handles_odd_shape() -> None:
    model = HybridRestoreNet(base_channels=4, levels=3, blocks_per_level=1).eval()
    inputs = torch.zeros(1, 3, 17, 25, 27)
    with torch.inference_mode():
        output = model(inputs)["corrected"]
    assert output.shape == (1, 1, 17, 25, 27)
