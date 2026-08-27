"""Coverage for the metal-inpainting path.

The Colab smoke run used a clean abdominal collection whose maximum was 1363 HU, so
`estimate_artifact_mask` produced an all-zero mask and the inpainting branch never
executed. These tests drive that branch directly with synthetic metal.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("nibabel")
import nibabel as nib  # noqa: E402

from ct_restore.inference import estimate_artifact_mask, restore_nifti  # noqa: E402
from ct_restore.models import HybridRestoreNet  # noqa: E402

HU_MIN, HU_MAX = -1024.0, 3071.0


def _volume_with_metal(shape=(24, 48, 48), metal_hu=6000.0) -> np.ndarray:
    volume = np.full(shape, -1000.0, dtype=np.float32)
    volume[6:18, 12:36, 12:36] = 40.0  # soft tissue
    volume[10:13, 22:26, 22:26] = metal_hu  # implant
    return volume


def test_clean_volume_yields_empty_mask() -> None:
    """Reproduces the smoke-run condition: no voxel above threshold, nothing flagged."""
    clean = _volume_with_metal(metal_hu=1363.0)
    assert clean.max() < 2800.0
    assert estimate_artifact_mask(clean).sum() == 0.0


def test_metal_is_flagged_and_dilated_beyond_the_implant() -> None:
    volume = _volume_with_metal()
    mask = estimate_artifact_mask(volume)
    metal = volume > 2800.0
    assert mask.sum() > 0.0
    assert set(np.unique(mask)) <= {0.0, 1.0}
    # Streak artifacts extend well past the metal, so the mask must too.
    assert mask.sum() > metal.sum() * 10
    assert np.all(mask[metal] == 1.0)


def test_mask_spreads_to_neighbouring_slices() -> None:
    volume = _volume_with_metal()
    mask = estimate_artifact_mask(volume)
    metal_slices = np.flatnonzero((volume > 2800.0).any(axis=(1, 2)))
    flagged = np.flatnonzero(mask.any(axis=(1, 2)))
    assert flagged.min() < metal_slices.min()
    assert flagged.max() > metal_slices.max()


def _write_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model_cfg = {"base_channels": 4, "levels": 2, "blocks_per_level": 1}
    torch.save(
        {
            "model": HybridRestoreNet(**model_cfg).state_dict(),
            "config": {
                "model": model_cfg,
                "data": {"hu_min": HU_MIN, "hu_max": HU_MAX, "patch_size": [16, 32, 32]},
                "hardware": {"precision": "fp32"},
            },
        },
        path,
    )


def test_restore_nifti_runs_the_inpainting_branch(tmp_path: Path) -> None:
    volume = _volume_with_metal()
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    image_path = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), affine), image_path)
    checkpoint = tmp_path / "ckpt.pt"
    _write_checkpoint(checkpoint)

    restored, uncertainty = restore_nifti(
        image_path, checkpoint, tmp_path / "out.nii.gz", device="cpu"
    )

    assert restored.exists() and uncertainty.exists()
    out = np.asarray(nib.load(str(restored)).dataobj, dtype=np.float32)
    assert out.shape == volume.transpose(2, 1, 0).shape
    assert np.isfinite(out).all()
    assert HU_MIN - 1 <= out.min() and out.max() <= HU_MAX + 1

    unc = np.asarray(nib.load(str(uncertainty)).dataobj, dtype=np.float32)
    assert np.isfinite(unc).all() and (unc >= 0).all()

    provenance = json.loads((tmp_path / "out.provenance.json").read_text())
    assert provenance["artifact_mask_source"] == "heuristic_metal_threshold"
    assert provenance["research_use_only"] is True
    assert provenance["uncertainty_units"] == "approximate_HU_not_calibrated"


def test_provided_mask_overrides_the_heuristic(tmp_path: Path) -> None:
    volume = _volume_with_metal(metal_hu=1000.0)  # heuristic alone would find nothing
    affine = np.eye(4)
    image_path = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), affine), image_path)
    mask = np.zeros_like(volume)
    mask[8:16, 16:32, 16:32] = 1.0
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.transpose(2, 1, 0), affine), mask_path)
    checkpoint = tmp_path / "ckpt.pt"
    _write_checkpoint(checkpoint)

    restored, _ = restore_nifti(
        image_path, checkpoint, tmp_path / "out.nii.gz", mask_path=mask_path, device="cpu"
    )
    provenance = json.loads((tmp_path / "out.provenance.json").read_text())
    assert provenance["artifact_mask_source"] == "provided"
    assert restored.exists()


def test_mismatched_mask_shape_is_rejected(tmp_path: Path) -> None:
    volume = _volume_with_metal()
    affine = np.eye(4)
    image_path = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), affine), image_path)
    bad = np.zeros((10, 10, 10), dtype=np.float32)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(bad, affine), mask_path)
    checkpoint = tmp_path / "ckpt.pt"
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="shapes differ"):
        restore_nifti(
            image_path, checkpoint, tmp_path / "out.nii.gz", mask_path=mask_path, device="cpu"
        )


@pytest.mark.parametrize(
    ("output_name", "expected_uncertainty", "expected_provenance"),
    [
        ("out.nii.gz", "out_uncertainty.nii.gz", "out.provenance.json"),
        ("out.nii", "out_uncertainty.nii.gz", "out.provenance.json"),
    ],
)
def test_sidecar_paths_never_collide_with_the_restored_volume(
    tmp_path: Path, output_name: str, expected_uncertainty: str, expected_provenance: str
) -> None:
    """A ``.nii`` output once made the uncertainty map overwrite the restored CT."""
    volume = _volume_with_metal()
    affine = np.eye(4)
    image_path = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), affine), image_path)
    checkpoint = tmp_path / "ckpt.pt"
    _write_checkpoint(checkpoint)

    restored, uncertainty = restore_nifti(
        image_path, checkpoint, tmp_path / output_name, device="cpu"
    )
    assert restored != uncertainty
    assert uncertainty.name == expected_uncertainty
    assert (tmp_path / expected_provenance).exists()

    # If the sidecar collided, both reads would return the same array.
    out = np.asarray(nib.load(str(restored)).dataobj, dtype=np.float32)
    unc = np.asarray(nib.load(str(uncertainty)).dataobj, dtype=np.float32)
    assert not np.array_equal(out, unc), "restored volume was overwritten by the uncertainty map"
