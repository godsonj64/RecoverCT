import csv
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from ct_restore.data.dataset import CTVolumeDataset  # noqa: E402


def test_paired_manifest_uses_aligned_real_inputs(tmp_path: Path) -> None:
    affine = np.eye(4)
    paths = {}
    for name, value in (("input", 100.0), ("target", 0.0), ("mask", 1.0)):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((12, 12, 12), value, dtype=np.float32), affine), path)
        paths[name] = path
    manifest = tmp_path / "manifest.csv"
    fields = [
        "image_path",
        "input_path",
        "target_path",
        "artifact_mask_path",
        "patient_id",
        "split",
        "qc_pass",
        "manual_approved",
    ]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "",
                "input_path": paths["input"],
                "target_path": paths["target"],
                "artifact_mask_path": paths["mask"],
                "patient_id": "P001",
                "split": "train",
                "qc_pass": "true",
                "manual_approved": "true",
            }
        )
    sample = CTVolumeDataset(manifest, "train", patch_size=(8, 8, 8))[0]
    assert sample["input"].shape == (3, 8, 8, 8)
    assert sample["artifact_mask"].sum().item() == 8**3
    assert sample["target"].mean().item() < sample["corrupted"].mean().item()
