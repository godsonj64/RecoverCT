"""Resuming training must restore everything that decides checkpoint quality.

Resume previously reloaded only the model, optimizer, and scheduler. The averaged
weights were rebuilt from the live model, discarding the EMA history, and
``best_validation`` restarted at infinity so the first resumed epoch overwrote best.pt
no matter how bad it was.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("nibabel")
import nibabel as nib  # noqa: E402

from ct_restore.config import ExperimentConfig  # noqa: E402
from ct_restore.training import train  # noqa: E402


def _tiny_config(tmp_path: Path, epochs: int) -> ExperimentConfig:
    rng = np.random.default_rng(0)
    manifest = tmp_path / "manifest.csv"
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        volume = rng.normal(0.0, 300.0, (16, 32, 32)).astype(np.float32)
        path = tmp_path / f"v{index}.nii.gz"
        nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), np.eye(4)), path)
        rows.append(
            {
                "image_path": str(path),
                "patient_id": f"P{index}",
                "split": split,
                "qc_pass": "true",
                "manual_approved": "true",
            }
        )
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cfg = ExperimentConfig()
    cfg.data.manifest = str(manifest)
    cfg.data.patch_size = (8, 16, 16)
    cfg.data.num_workers = 0
    cfg.data.artifact_probability = 1.0
    cfg.model.base_channels = 4
    cfg.model.levels = 2
    cfg.model.blocks_per_level = 1
    cfg.train.epochs = epochs
    cfg.train.batch_size = 1
    cfg.train.grad_accumulation = 1
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.hardware.device = "cpu"
    cfg.hardware.auto_tune = False
    return cfg


def test_checkpoint_records_best_validation(tmp_path: Path) -> None:
    train(_tiny_config(tmp_path, epochs=1), allow_unreviewed=True)
    state = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu", weights_only=True)
    assert "best_validation" in state
    assert "ema_model" in state
    assert np.isfinite(state["best_validation"])


def test_resume_restores_the_averaged_weights(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path, epochs=1)
    train(cfg, allow_unreviewed=True)
    before = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu", weights_only=True)

    resumed = _tiny_config(tmp_path, epochs=2)
    resumed.train.resume = str(tmp_path / "ckpt" / "last.pt")
    train(resumed, allow_unreviewed=True)
    after = torch.load(tmp_path / "ckpt" / "last.pt", map_location="cpu", weights_only=True)

    # A discarded EMA would have been rebuilt from the live weights, so the shadow would
    # equal the raw model exactly at the moment of the restart.
    assert set(before["ema_model"]) == set(after["ema_model"])
    assert np.isfinite(after["best_validation"])


def test_resume_does_not_reset_best_validation(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path, epochs=1)
    train(cfg, allow_unreviewed=True)
    checkpoint_dir = tmp_path / "ckpt"

    # Pretend an excellent epoch already happened.
    state = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    state["best_validation"] = -1.0e6
    torch.save(state, checkpoint_dir / "last.pt")
    best_before = torch.load(
        checkpoint_dir / "best.pt", map_location="cpu", weights_only=True
    )["epoch"]

    resumed = _tiny_config(tmp_path, epochs=2)
    resumed.train.resume = str(checkpoint_dir / "last.pt")
    train(resumed, allow_unreviewed=True)

    best_after = torch.load(
        checkpoint_dir / "best.pt", map_location="cpu", weights_only=True
    )["epoch"]
    assert best_after == best_before, "a worse epoch overwrote best.pt after resume"
