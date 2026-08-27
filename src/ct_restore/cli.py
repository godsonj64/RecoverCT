from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer

from ct_restore.artifacts import ArtifactSimulator
from ct_restore.config import load_config
from ct_restore.data.preprocess import preprocess_tree
from ct_restore.data.tcia import (
    RECOMMENDED_COLLECTIONS,
    download_series,
    query_ct_series,
    select_planning_candidates,
    write_series_manifest,
)
from ct_restore.hardware import detect_hardware, resolve_config
from ct_restore.inference import restore_nifti
from ct_restore.metrics import image_metrics
from ct_restore.models import HybridRestoreNet
from ct_restore.training import train as run_training

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command("doctor")
def doctor(
    device: Annotated[str, typer.Option(help="auto, cpu, mps, or cuda")] = "auto",
) -> None:
    """Report runtime, CUDA capability, and safe high-throughput recommendations."""
    profile = detect_hardware(device)
    typer.echo(json.dumps(profile.to_dict(), indent=2))


@app.command("configure")
def configure(
    template: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Argument()],
    device: Annotated[str, typer.Option(help="auto, cpu, mps, or cuda")] = "auto",
) -> None:
    """Resolve an experiment template for the detected CPU/GPU."""
    import yaml

    cfg = load_config(template)
    profile = detect_hardware(device)
    cfg = resolve_config(cfg, profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    typer.echo(f"Wrote {output}")
    typer.echo(json.dumps(profile.to_dict(), indent=2))


@app.command("collections")
def collections() -> None:
    """Show reviewed TCIA collection choices and intended roles."""
    typer.echo(json.dumps(RECOMMENDED_COLLECTIONS, indent=2))


@app.command("tcia")
def tcia(
    collection: Annotated[str, typer.Option(help="TCIA collection name")] = "HNC-IMRT-70-33",
    output_dir: Annotated[Path, typer.Option(help="External raw-data directory")] = Path(
        "data/raw"
    ),
    manifest: Annotated[Path, typer.Option(help="Where to store selected series metadata")] = Path(
        "data/manifests/tcia_series.csv"
    ),
    download: Annotated[
        bool, typer.Option("--download", help="Actually download DICOM data")
    ] = False,
    limit: Annotated[int, typer.Option(help="Maximum series to download; 0 means all")] = 0,
    workers: Annotated[int, typer.Option(min=1, max=16)] = 4,
) -> None:
    """Query TCIA and optionally download filtered CT series."""
    rows = query_ct_series(collection)
    selected = select_planning_candidates(rows)
    write_series_manifest(selected, manifest)
    typer.echo(f"Selected {len(selected)}/{len(rows)} candidate CT series; metadata: {manifest}")
    if not download:
        typer.echo(
            "Dry run only. Review licenses, storage, and the manifest; pass --download to fetch."
        )
        return
    download_series(selected, output_dir / collection, limit=limit, workers=workers)


@app.command("preprocess")
def preprocess(
    input_root: Annotated[
        Path, typer.Argument(help="Root containing one DICOM series per directory")
    ],
    output_root: Annotated[Path, typer.Argument(help="Destination for resampled NIfTI volumes")],
    manifest: Annotated[Path, typer.Option()] = Path("data/manifests/head_neck.csv"),
    collection: Annotated[str, typer.Option()] = "HNC-IMRT-70-33",
    spacing_z: Annotated[float, typer.Option()] = 1.5,
    spacing_y: Annotated[float, typer.Option()] = 1.0,
    spacing_x: Annotated[float, typer.Option()] = 1.0,
) -> None:
    """Convert DICOM CT to calibrated, resampled NIfTI and create a QC manifest."""
    rows = preprocess_tree(
        input_root,
        output_root,
        manifest,
        collection,
        (spacing_z, spacing_y, spacing_x),
    )
    passed = sum(row.qc_pass for row in rows)
    typer.echo(f"Wrote {len(rows)} rows ({passed} auto-QC pass) to {manifest}")
    typer.echo("Set manual_approved=true only after visual and metadata review.")


@app.command("train")
def train(
    config: Annotated[Path, typer.Argument(exists=True, readable=True)],
    allow_unreviewed: Annotated[
        bool,
        typer.Option(help="Research smoke tests only; bypasses manual QC gate"),
    ] = False,
) -> None:
    """Run staged 3D restoration training."""
    cfg = load_config(config)
    output = run_training(cfg, allow_unreviewed=allow_unreviewed)
    typer.echo(f"Best checkpoint: {output}")


@app.command("infer")
def infer(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    checkpoint: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_path: Annotated[Path, typer.Argument()],
    mask: Annotated[Path | None, typer.Option(help="Reviewed artifact mask NIfTI")] = None,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Restore a NIfTI volume and save an uncertainty volume."""
    output, uncertainty = restore_nifti(input_path, checkpoint, output_path, mask, device)
    typer.echo(f"Research output: {output}")
    typer.echo(f"Uncertainty map: {uncertainty}")


@app.command("simulate")
def simulate(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Argument()],
    seed: Annotated[int, typer.Option()] = 2026,
) -> None:
    """Create one auditable synthetic corruption example from a clean NIfTI."""
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(str(input_path)))
    hu = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0)
    result = ArtifactSimulator(seed=seed)(hu)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in {
        "corrupted.nii.gz": result.corrupted_hu,
        "artifact_mask.nii.gz": result.artifact_mask,
        "metal_mask.nii.gz": result.metal_mask,
    }.items():
        nib.save(
            nib.Nifti1Image(array.transpose(2, 1, 0), image.affine, image.header), output_dir / name
        )
    (output_dir / "simulation.json").write_text(json.dumps(result.metadata, indent=2))
    typer.echo(str(output_dir))


@app.command("evaluate")
def evaluate(
    prediction: Annotated[Path, typer.Argument(exists=True, readable=True)],
    target: Annotated[Path, typer.Argument(exists=True, readable=True)],
    artifact_mask: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Compute HU-stratified image metrics for a paired case."""
    import nibabel as nib

    pred = np.asarray(nib.load(str(prediction)).dataobj, dtype=np.float32)
    truth = np.asarray(nib.load(str(target)).dataobj, dtype=np.float32)
    mask = None
    if artifact_mask:
        mask = np.asarray(nib.load(str(artifact_mask)).dataobj) > 0.5
    values = image_metrics(pred, truth, mask)
    payload = json.dumps(values, indent=2)
    typer.echo(payload)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)


@app.command("inspect-model")
def inspect_model(
    base_channels: Annotated[int, typer.Option(min=4)] = 24,
    patch_size: Annotated[int, typer.Option(min=16)] = 32,
) -> None:
    """Print parameter count and run a small shape smoke test."""
    model = HybridRestoreNet(base_channels=base_channels).eval()
    sample = torch.zeros(1, 3, patch_size, patch_size, patch_size)
    with torch.inference_mode():
        output = model(sample)
    typer.echo(
        json.dumps(
            {
                "parameters": model.parameter_count,
                "parameters_millions": round(model.parameter_count / 1e6, 3),
                "input_shape": list(sample.shape),
                "output_shapes": {key: list(value.shape) for key, value in output.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
