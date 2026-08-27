#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import yaml

from ct_restore.config import load_config
from ct_restore.data.preprocess import preprocess_tree
from ct_restore.data.tcia import (
    NBIA_BASE_URL,
    download_series,
    list_collections,
    query_ct_series,
    select_planning_candidates,
    write_series_manifest,
)
from ct_restore.inference import restore_nifti
from ct_restore.training import train


def _default_root() -> Path:
    configured = os.environ.get("CT_RESTORE_DATA_ROOT")
    if configured:
        return Path(configured)
    if os.environ.get("RUNPOD_POD_ID"):
        return Path("/workspace/ct_restore_data")
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        return Path("/content/ct_restore_data")
    return Path.cwd() / "data"


def _truth(value: str | bool | None) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def _smoke_manifest(source: Path, destination: Path) -> int:
    with source.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if _truth(row.get("qc_pass"))]
    if not rows:
        raise RuntimeError("No automatically eligible CT volumes remain after preprocessing QC")
    patients: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        patients.setdefault(row["patient_id"], []).append(row)
    patient_ids = sorted(patients)
    for row in rows:
        row["split"] = "train"
        row["qc_notes"] = (row.get("qc_notes", "") + ";smoke_only_split_override").lstrip(";")
    if len(patient_ids) > 1:
        for row in patients[patient_ids[-1]]:
            row["split"] = "val"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_config(template: Path, output: Path, manifest: Path, root: Path, epochs: int) -> Path:
    cfg = load_config(template)
    cfg.data.manifest = str(manifest)
    cfg.train.epochs = epochs
    cfg.train.checkpoint_dir = str(root / "checkpoints" / "notebook")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded TCIA download → preprocessing → smoke training → inference"
    )
    parser.add_argument("--data-root", type=Path, default=_default_root())
    parser.add_argument(
        "--collection",
        default="Pancreas-CT",
        help="Smoke-run collection. Must be served by the anonymous NBIA API; the "
        "head-and-neck planning collections need authenticated access.",
    )
    parser.add_argument("--limit", type=int, default=1, help="0 downloads the full collection")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "notebook.yaml",
    )
    parser.add_argument("--accept-data-terms", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_data_terms and not args.skip_download:
        raise SystemExit(
            "Downloading requires --accept-data-terms after reviewing the collection's "
            "current TCIA access, license, and citation requirements."
        )
    root = args.data_root.resolve()
    raw = root / "raw" / args.collection
    processed = root / "processed" / args.collection
    manifests = root / "manifests"
    series_manifest = manifests / f"{args.collection}_series.csv"
    qc_manifest = manifests / f"{args.collection}_qc.csv"
    train_manifest = manifests / f"{args.collection}_notebook_train.csv"
    if not args.skip_download:
        rows = query_ct_series(args.collection)
        if not rows:
            available = list_collections()
            if args.collection not in available:
                raise RuntimeError(
                    f"Collection {args.collection!r} is not served by the public NBIA API at "
                    f"{NBIA_BASE_URL}. It is either restricted (NBIA login / Data Retriever "
                    f"required) or renamed. {len(available)} collections are publicly listed; "
                    "run `ct_restore.data.tcia.list_collections()` to see them, then pass a "
                    "reachable one with --collection."
                )
            raise RuntimeError(
                f"Collection {args.collection!r} is public but returned no CT series. "
                "Its images may be restricted even though the collection is listed."
            )
        selected = select_planning_candidates(rows)
        if not selected:
            raise RuntimeError(
                f"{len(rows)} CT series returned for {args.collection!r}, but none passed the "
                "planning-CT filter (needs >=80 images and a non-localizer description). "
                "Relax select_planning_candidates() or choose a different collection."
            )
        write_series_manifest(selected, series_manifest)
        download_series(selected, raw, limit=args.limit, workers=args.workers)
    if not raw.exists():
        raise RuntimeError(f"Raw DICOM directory does not exist: {raw}")
    preprocess_tree(raw, processed, qc_manifest, args.collection)
    count = _smoke_manifest(qc_manifest, train_manifest)
    print(f"Prepared {count} automatically screened volume(s) for research smoke training")
    resolved_input = _write_config(
        args.template,
        root / "configs" / "notebook_input.yaml",
        train_manifest,
        root,
        args.epochs,
    )
    checkpoint = train(load_config(resolved_input), allow_unreviewed=True)
    print(f"Checkpoint: {checkpoint}")
    if not args.skip_inference:
        with train_manifest.open(newline="") as handle:
            first = next(csv.DictReader(handle))
        output = root / "outputs" / "notebook_restored.nii.gz"
        restored, uncertainty = restore_nifti(first["image_path"], checkpoint, output)
        print(f"Restored volume: {restored}")
        print(f"Uncertainty volume: {uncertainty}")


if __name__ == "__main__":
    main()
