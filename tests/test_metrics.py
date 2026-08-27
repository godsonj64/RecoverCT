import numpy as np

from ct_restore.metrics import image_metrics


def test_metrics_are_hu_stratified() -> None:
    target = np.array([[[-1000.0, 0.0, 500.0, 2500.0]]])
    prediction = target + 10.0
    mask = np.array([[[0, 1, 1, 0]]], dtype=bool)
    result = image_metrics(prediction, target, mask)
    assert result["mae_hu"] == 10.0
    assert result["artifact_mae_hu"] == 10.0
    assert result["known_region_mae_hu"] == 10.0
    assert result["air_bias_hu"] == 10.0
    assert result["soft_tissue_bias_hu"] == 10.0
    assert result["bone_bias_hu"] == 10.0
    assert result["dense_bone_bias_hu"] == 10.0
