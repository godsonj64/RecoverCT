"""Sampling randomness.

``persistent_workers=True`` keeps a worker alive across epochs, and
``torch.initial_seed()`` is fixed for that worker's lifetime. Seeding a sample from
``(seed, index, initial_seed)`` alone therefore replayed the identical crop and the
identical simulated artifact every epoch.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("nibabel")
import nibabel as nib  # noqa: E402

from ct_restore.data.dataset import CTVolumeDataset  # noqa: E402

PATCH = (8, 16, 16)


def _manifest(tmp_path: Path, split: str = "train") -> Path:
    volume = np.random.default_rng(0).normal(0.0, 300.0, (24, 48, 48)).astype(np.float32)
    image = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), np.eye(4)), image)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "patient_id", "split", "qc_pass", "manual_approved"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(image),
                "patient_id": "P1",
                "split": split,
                "qc_pass": "true",
                "manual_approved": "true",
            }
        )
    return manifest


def _dataset(manifest: Path, **kwargs) -> CTVolumeDataset:
    return CTVolumeDataset(
        manifest, kwargs.pop("split", "train"), PATCH, artifact_probability=1.0, **kwargs
    )


def test_training_samples_change_across_epochs(tmp_path: Path) -> None:
    """The worker seed is held fixed, exactly as a persistent worker would."""
    dataset = _dataset(_manifest(tmp_path), seed=2026)
    torch.manual_seed(12345)
    first = dataset[0]["input"].numpy().copy()
    second = dataset[0]["input"].numpy().copy()
    assert not np.array_equal(first, second), "augmentation frozen across epochs"


def test_validation_samples_are_identical_across_epochs(tmp_path: Path) -> None:
    """Otherwise validation loss is augmentation noise and best.pt tracks it."""
    dataset = _dataset(_manifest(tmp_path, "val"), split="val", seed=2026, deterministic=True)
    torch.manual_seed(999)
    first = dataset[0]["input"].numpy().copy()
    torch.manual_seed(4242)  # a different worker seed must not matter
    second = dataset[0]["input"].numpy().copy()
    assert np.array_equal(first, second)


def test_training_stream_is_reproducible_for_a_given_seed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    def draw() -> list[np.ndarray]:
        dataset = _dataset(manifest, seed=2026)
        torch.manual_seed(777)
        return [dataset[0]["input"].numpy().copy() for _ in range(3)]

    for left, right in zip(draw(), draw(), strict=True):
        assert np.array_equal(left, right)


def test_different_seeds_give_different_streams(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    torch.manual_seed(777)
    a = _dataset(manifest, seed=1)[0]["input"].numpy()
    torch.manual_seed(777)
    b = _dataset(manifest, seed=2)[0]["input"].numpy()
    assert not np.array_equal(a, b)
