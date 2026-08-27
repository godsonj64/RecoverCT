from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from ct_restore.artifacts import ArtifactSimulator


def normalize_hu(array: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    array = np.clip(array, hu_min, hu_max)
    return ((array - hu_min) / (hu_max - hu_min) * 2.0 - 1.0).astype(np.float32)


def denormalize_hu(array: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    return ((array + 1.0) * 0.5 * (hu_max - hu_min) + hu_min).astype(np.float32)


def _as_bool(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _crop_or_pad_aligned(
    volumes: list[np.ndarray],
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    constants: list[float],
) -> list[np.ndarray]:
    if not volumes or len(volumes) != len(constants):
        raise ValueError("volumes and constants must be non-empty and equal length")
    if any(volume.shape != volumes[0].shape for volume in volumes[1:]):
        raise ValueError("Paired volumes must have exactly matching voxel grids")
    pads = [
        (0, max(0, target - current))
        for current, target in zip(volumes[0].shape, shape, strict=True)
    ]
    if any(after for _, after in pads):
        volumes = [
            np.pad(volume, pads, mode="constant", constant_values=constant)
            for volume, constant in zip(volumes, constants, strict=True)
        ]
    starts = [
        int(rng.integers(0, current - target + 1))
        for current, target in zip(volumes[0].shape, shape, strict=True)
    ]
    slices = tuple(
        slice(start, start + target) for start, target in zip(starts, shape, strict=True)
    )
    return [volume[slices] for volume in volumes]


def _load_zyx(path: str) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(path))
    return np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0), image.affine


class CTVolumeDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        patch_size: tuple[int, int, int] = (64, 128, 128),
        hu_min: float = -1024.0,
        hu_max: float = 3071.0,
        artifact_probability: float = 0.9,
        allow_unreviewed: bool = False,
        seed: int = 2026,
        deterministic: bool = False,
    ) -> None:
        with Path(manifest).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [
            row
            for row in rows
            if row.get("split") == split
            and _as_bool(row.get("qc_pass", True))
            and (allow_unreviewed or _as_bool(row.get("manual_approved")))
        ]
        if not self.rows:
            qualifier = " (manual approval is required)" if not allow_unreviewed else ""
            raise ValueError(f"No eligible {split!r} rows in {manifest}{qualifier}")
        self.patch_size = patch_size
        self.hu_min, self.hu_max = hu_min, hu_max
        self.artifact_probability = artifact_probability
        self.seed = seed
        self.deterministic = deterministic
        self._draws = 0

    def __len__(self) -> int:
        return len(self.rows)

    def _entropy(self, index: int) -> np.random.SeedSequence:
        """Seed material for one sample.

        ``torch.initial_seed()`` is fixed for a worker's whole lifetime, and
        ``persistent_workers=True`` keeps workers alive across epochs. Seeding from
        ``(seed, index, initial_seed)`` alone therefore replays the identical crop and
        the identical simulated artifact every epoch, freezing augmentation after the
        first one. A per-worker draw counter breaks that tie while staying reproducible
        for a given seed and call order.

        Validation must not move between epochs or its loss is not comparable and
        checkpoint selection follows augmentation noise, so it seeds from the index only.
        """
        if self.deterministic:
            return np.random.SeedSequence([self.seed, index])
        entropy = [self.seed, index, torch.initial_seed() % (2**31), self._draws]
        self._draws += 1
        return np.random.SeedSequence(entropy)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        rng = np.random.default_rng(self._entropy(index))
        paired_fields = [
            row.get(name, "").strip()
            for name in ("input_path", "target_path", "artifact_mask_path")
        ]
        if any(paired_fields):
            if not all(paired_fields):
                raise ValueError(
                    "Real paired rows require input_path, target_path, and artifact_mask_path"
                )
            corrupted_hu, input_affine = _load_zyx(paired_fields[0])
            clean_hu, target_affine = _load_zyx(paired_fields[1])
            mask, mask_affine = _load_zyx(paired_fields[2])
            if not (
                np.allclose(input_affine, target_affine, atol=1e-4)
                and np.allclose(input_affine, mask_affine, atol=1e-4)
            ):
                raise ValueError("Paired NIfTI affines differ; registration must be reviewed")
            clean_hu, corrupted_hu, mask = _crop_or_pad_aligned(
                [clean_hu, corrupted_hu, mask], self.patch_size, rng, [-1024.0, -1024.0, 0.0]
            )
            mask = (mask > 0.5).astype(np.float32)
        else:
            clean_hu, _ = _load_zyx(row["image_path"])
            clean_hu = _crop_or_pad_aligned([clean_hu], self.patch_size, rng, [-1024.0])[0]
            if rng.random() < self.artifact_probability:
                simulated = ArtifactSimulator(seed=int(rng.integers(0, 2**31)))(clean_hu)
                corrupted_hu = simulated.corrupted_hu
                mask = simulated.artifact_mask
            else:
                corrupted_hu = clean_hu + rng.normal(0, 8, clean_hu.shape).astype(np.float32)
                mask = np.zeros_like(clean_hu, dtype=np.float32)
        target = normalize_hu(clean_hu, self.hu_min, self.hu_max)
        corrupted = normalize_hu(corrupted_hu, self.hu_min, self.hu_max)
        inputs = np.stack((corrupted, mask, 1.0 - mask), axis=0)
        return {
            "input": torch.from_numpy(inputs),
            "target": torch.from_numpy(target[None]),
            "corrupted": torch.from_numpy(corrupted[None]),
            "artifact_mask": torch.from_numpy(mask[None]),
            "patient_id": row["patient_id"],
        }
