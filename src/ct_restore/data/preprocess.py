from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class QCResult:
    image_path: str
    patient_id: str
    series_uid: str
    source_collection: str
    split: str
    slices: int
    spacing_z: float
    spacing_y: float
    spacing_x: float
    metal_fraction: float
    air_fraction: float
    qc_pass: bool
    manual_approved: bool
    qc_notes: str


def patient_split(patient_id: str, seed: int = 2026) -> str:
    value = int(hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "val"
    return "test"


def _resample(image: Any, spacing: tuple[float, float, float]) -> Any:
    import SimpleITK as sitk

    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        max(1, int(round(n * s / t)))
        for n, s, t in zip(old_size, old_spacing, spacing, strict=True)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        -1024.0,
        sitk.sitkFloat32,
    )


def preprocess_dicom_series(
    series_dir: str | Path,
    output_path: str | Path,
    target_spacing_zyx: tuple[float, float, float] = (1.5, 1.0, 1.0),
) -> tuple[Path, dict[str, float | int | str | bool]]:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required for DICOM preprocessing") from exc

    series_dir, output_path = Path(series_dir), Path(output_path)
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not ids:
        raise ValueError(f"No DICOM series found in {series_dir}")
    if len(ids) > 1:
        lengths = {uid: len(reader.GetGDCMSeriesFileNames(str(series_dir), uid)) for uid in ids}
        series_uid = max(lengths, key=lengths.get)
    else:
        series_uid = ids[0]
    files = reader.GetGDCMSeriesFileNames(str(series_dir), series_uid)
    try:
        import pydicom

        header = pydicom.dcmread(
            files[0],
            stop_before_pixels=True,
            specific_tags=["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
        )
        patient_id = str(header.get("PatientID", ""))
        dicom_series_uid = str(header.get("SeriesInstanceUID", series_uid))
    except Exception as exc:
        raise ValueError(f"Could not read required DICOM identifiers: {exc}") from exc
    if not patient_id:
        raise ValueError("DICOM PatientID is missing; patient-safe splitting is impossible")
    reader.SetFileNames(files)
    image = sitk.DICOMOrient(reader.Execute(), "LPS")
    original_spacing = image.GetSpacing()
    target_spacing_xyz = tuple(reversed(target_spacing_zyx))
    image = _resample(image, target_spacing_xyz)
    array = sitk.GetArrayViewFromImage(image)
    finite = bool(np.isfinite(array).all())
    metal_fraction = float(np.mean(array > 2800.0))
    air_fraction = float(np.mean(array < -1000.0))
    qc_pass = bool(
        finite
        and len(files) >= 80
        and original_spacing[2] <= 3.0
        and air_fraction < 0.85
        and metal_fraction < 5e-5
    )
    notes: list[str] = []
    if len(files) < 80:
        notes.append("too_few_slices")
    if original_spacing[2] > 3.0:
        notes.append("thick_slices")
    if metal_fraction >= 5e-5:
        notes.append("possible_metal")
    if air_fraction >= 0.85:
        notes.append("limited_anatomy_or_bad_rescale")
    if not finite:
        notes.append("non_finite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return output_path, {
        "patient_id": patient_id,
        "series_uid": dicom_series_uid,
        "slices": int(image.GetSize()[2]),
        "spacing_z": float(image.GetSpacing()[2]),
        "spacing_y": float(image.GetSpacing()[1]),
        "spacing_x": float(image.GetSpacing()[0]),
        "metal_fraction": metal_fraction,
        "air_fraction": air_fraction,
        "qc_pass": qc_pass,
        "qc_notes": ";".join(notes) if notes else "auto_qc_pass_manual_review_required",
    }


def preprocess_tree(
    input_root: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    collection: str,
    target_spacing_zyx: tuple[float, float, float] = (1.5, 1.0, 1.0),
) -> list[QCResult]:
    input_root, output_root = Path(input_root), Path(output_root)
    directories = [input_root, *input_root.rglob("*")]
    candidates = sorted(
        path
        for path in directories
        if path.is_dir()
        and any(child.is_file() and child.name != ".complete" for child in path.iterdir())
    )
    results: list[QCResult] = []
    for index, series_dir in enumerate(candidates):
        relative = str(series_dir.relative_to(input_root))
        suffix = hashlib.sha256(relative.encode()).hexdigest()[:12]
        output_path = output_root / f"{series_dir.name}_{suffix}.nii.gz"
        try:
            image_path, metadata = preprocess_dicom_series(
                series_dir, output_path, target_spacing_zyx
            )
        except (ValueError, RuntimeError) as exc:
            print(f"Skipping {series_dir}: {exc}")
            continue
        results.append(
            QCResult(
                image_path=str(image_path.resolve()),
                patient_id=str(metadata["patient_id"]),
                series_uid=str(metadata["series_uid"]),
                source_collection=collection,
                split=patient_split(str(metadata["patient_id"])),
                slices=int(metadata["slices"]),
                spacing_z=float(metadata["spacing_z"]),
                spacing_y=float(metadata["spacing_y"]),
                spacing_x=float(metadata["spacing_x"]),
                metal_fraction=float(metadata["metal_fraction"]),
                air_fraction=float(metadata["air_fraction"]),
                qc_pass=bool(metadata["qc_pass"]),
                manual_approved=False,
                qc_notes=str(metadata["qc_notes"]),
            )
        )
        print(f"[{index + 1}/{len(candidates)}] {series_dir.name}")
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QCResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    return results
