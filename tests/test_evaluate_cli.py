"""`ct-restore evaluate` must refuse mismatched inputs rather than report nonsense."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nibabel")
import nibabel as nib  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from ct_restore import cli  # noqa: E402

runner = CliRunner()


def _save(path: Path, array: np.ndarray, affine: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(array, affine), path)
    return path


def test_matching_volumes_produce_metrics(tmp_path: Path) -> None:
    affine = np.eye(4)
    truth = np.random.default_rng(0).normal(0, 200, (12, 14, 16)).astype(np.float32)
    a = _save(tmp_path / "pred.nii.gz", truth + 5.0, affine)
    b = _save(tmp_path / "truth.nii.gz", truth, affine)
    result = runner.invoke(cli.app, ["evaluate", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "mae_hu" in result.output
    assert "psnr_db" in result.output


def test_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    affine = np.eye(4)
    a = _save(tmp_path / "pred.nii.gz", np.zeros((12, 14, 16), np.float32), affine)
    b = _save(tmp_path / "truth.nii.gz", np.zeros((12, 14, 8), np.float32), affine)
    result = runner.invoke(cli.app, ["evaluate", str(a), str(b)])
    assert result.exit_code != 0
    assert "does not match target shape" in result.output


def test_different_voxel_grids_are_rejected(tmp_path: Path) -> None:
    array = np.zeros((12, 14, 16), np.float32)
    a = _save(tmp_path / "pred.nii.gz", array, np.diag([1.0, 1.0, 1.0, 1.0]))
    b = _save(tmp_path / "truth.nii.gz", array, np.diag([2.0, 1.0, 1.0, 1.0]))
    result = runner.invoke(cli.app, ["evaluate", str(a), str(b)])
    assert result.exit_code != 0
    assert "different voxel grids" in result.output


def test_mask_on_a_different_grid_is_rejected(tmp_path: Path) -> None:
    affine = np.eye(4)
    array = np.zeros((12, 14, 16), np.float32)
    a = _save(tmp_path / "pred.nii.gz", array, affine)
    b = _save(tmp_path / "truth.nii.gz", array, affine)
    m = _save(tmp_path / "mask.nii.gz", array, np.diag([3.0, 1.0, 1.0, 1.0]))
    result = runner.invoke(
        cli.app, ["evaluate", str(a), str(b), "--artifact-mask", str(m)]
    )
    assert result.exit_code != 0
    assert "different voxel grid" in result.output
