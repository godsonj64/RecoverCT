import numpy as np

from ct_restore.artifacts import ArtifactSimulator


def test_analytic_artifact_is_deterministic() -> None:
    clean = np.zeros((12, 24, 24), dtype=np.float32)
    first = ArtifactSimulator(seed=7, physics_probability=0.0)(clean)
    second = ArtifactSimulator(seed=7, physics_probability=0.0)(clean)
    np.testing.assert_array_equal(first.corrupted_hu, second.corrupted_hu)
    np.testing.assert_array_equal(first.artifact_mask, second.artifact_mask)
    assert first.metal_mask.sum() > 0
    assert first.artifact_mask.sum() >= first.metal_mask.sum()
    assert first.metadata["mode"] == "analytic"


def test_radon_artifact_smoke() -> None:
    clean = np.zeros((12, 24, 24), dtype=np.float32)
    result = ArtifactSimulator(seed=11, angles=30, physics_probability=1.0)(clean)
    assert result.corrupted_hu.shape == clean.shape
    assert np.isfinite(result.corrupted_hu).all()
    assert result.metadata["mode"] == "slice_radon"
