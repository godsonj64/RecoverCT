from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from ct_restore.data.dataset import denormalize_hu, normalize_hu
from ct_restore.hardware import autocast_dtype, detect_hardware
from ct_restore.models import HybridRestoreNet


def estimate_artifact_mask(volume_hu: np.ndarray) -> np.ndarray:
    metal = volume_hu > 2800.0
    mask = np.zeros_like(metal)
    for z in range(volume_hu.shape[0]):
        if metal[z].any():
            mask[z] = ndimage.binary_dilation(metal[z], iterations=18)
            mask[max(0, z - 2) : z + 3] |= mask[z]
    return ndimage.binary_dilation(mask, iterations=2).astype(np.float32)


def _gaussian_weight(shape: tuple[int, int, int]) -> torch.Tensor:
    axes = [torch.linspace(-1, 1, steps=size) for size in shape]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    return torch.exp(-4.0 * (zz.square() + yy.square() + xx.square())).clamp_min(1e-3)


@torch.inference_mode()
def sliding_window_predict(
    model: HybridRestoreNet,
    inputs: torch.Tensor,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 5 or inputs.shape[0] != 1:
        raise ValueError("sliding_window_predict expects [1,C,D,H,W]")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    original_shape = inputs.shape[2:]
    padding: list[int] = []
    for current, target in reversed(list(zip(original_shape, patch_size, strict=True))):
        padding.extend((0, max(0, target - current)))
    # Padding represents air for CT, no suspected artifact, and full known-data confidence.
    padded_channels = [
        torch.nn.functional.pad(inputs[:, :1], padding, value=-1.0),
        torch.nn.functional.pad(inputs[:, 1:2], padding, value=0.0),
        torch.nn.functional.pad(inputs[:, 2:3], padding, value=1.0),
    ]
    inputs = torch.cat(padded_channels, dim=1)
    spatial = inputs.shape[2:]
    strides = [max(1, int(size * (1 - overlap))) for size in patch_size]
    starts = []
    for total, size, stride in zip(spatial, patch_size, strides, strict=True):
        values = list(range(0, max(1, total - size + 1), stride))
        if not values or values[-1] != total - size:
            values.append(total - size)
        starts.append(values)
    weight = _gaussian_weight(patch_size).to(inputs.device)[None, None]
    corrected = torch.zeros((1, 1, *spatial), device=inputs.device)
    uncertainty = torch.zeros_like(corrected)
    denominator = torch.zeros_like(corrected)
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                patch = inputs[
                    :, :, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]
                ]
                output = model(patch)
                region = (
                    ...,
                    slice(z, z + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(x, x + patch_size[2]),
                )
                corrected[region] += output["corrected"] * weight
                uncertainty[region] += torch.exp(0.5 * output["log_variance"]) * weight
                denominator[region] += weight
    crop = (
        ...,
        slice(0, original_shape[0]),
        slice(0, original_shape[1]),
        slice(0, original_shape[2]),
    )
    return (corrected / denominator.clamp_min(1e-6))[crop], (
        uncertainty / denominator.clamp_min(1e-6)
    )[crop]


def restore_nifti(
    input_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    device: str = "auto",
) -> tuple[Path, Path]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_cfg = checkpoint.get("config", {})
    model_cfg = raw_cfg.get("model", {})
    data_cfg = raw_cfg.get("data", {})
    model = HybridRestoreNet(**model_cfg)
    if "ema_model" in checkpoint:
        weights = checkpoint["ema_model"]
    elif "model" in checkpoint:
        weights = checkpoint["model"]
    else:
        weights = checkpoint
    model.load_state_dict(weights)
    profile = detect_hardware(device)
    selected_device = torch.device(profile.device)
    hardware_cfg = raw_cfg.get("hardware", {})
    precision = hardware_cfg.get("precision", profile.precision)
    if precision == "auto":
        precision = profile.precision
    use_channels_last = bool(
        selected_device.type == "cuda" and hardware_cfg.get("channels_last_3d", True)
    )
    if selected_device.type == "cuda":
        allow_tf32 = bool(hardware_cfg.get("allow_tf32", True)) and profile.tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(hardware_cfg.get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision("high")
    model.to(selected_device).eval()
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    image = nib.as_closest_canonical(nib.load(str(input_path)))
    volume_xyz = np.asarray(image.dataobj, dtype=np.float32)
    volume_hu = volume_xyz.transpose(2, 1, 0)
    if mask_path:
        mask_image = nib.as_closest_canonical(nib.load(str(mask_path)))
        mask = np.asarray(mask_image.dataobj, dtype=np.float32).transpose(2, 1, 0)
        if mask.shape != volume_hu.shape:
            raise ValueError("Artifact mask and input volume shapes differ")
        mask = (mask > 0.5).astype(np.float32)
        mask_source = "provided"
    else:
        mask = estimate_artifact_mask(volume_hu)
        mask_source = "heuristic_metal_threshold"
    hu_min, hu_max = float(data_cfg.get("hu_min", -1024)), float(data_cfg.get("hu_max", 3071))
    normalized = normalize_hu(volume_hu, hu_min, hu_max)
    stacked = np.stack((normalized, mask, 1.0 - mask), axis=0)
    inputs = torch.from_numpy(stacked)[None].to(selected_device)
    if use_channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last_3d)
    patch_size = tuple(data_cfg.get("patch_size", (64, 128, 128)))
    with torch.autocast(
        device_type=selected_device.type,
        dtype=autocast_dtype(precision),
        enabled=selected_device.type == "cuda" and precision in {"fp16", "bf16"},
    ):
        corrected, uncertainty = sliding_window_predict(model, inputs, patch_size)
    corrected_hu = denormalize_hu(corrected[0, 0].cpu().numpy(), hu_min, hu_max)
    uncertainty_hu = uncertainty[0, 0].cpu().numpy() * (hu_max - hu_min) / 2.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(corrected_hu.transpose(2, 1, 0), image.affine, image.header), output_path
    )
    uncertainty_path = output_path.with_name(
        output_path.name.replace(".nii.gz", "_uncertainty.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(uncertainty_hu.transpose(2, 1, 0), image.affine, image.header),
        uncertainty_path,
    )
    output_path.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "research_use_only": True,
                "input": str(Path(input_path).resolve()),
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "artifact_mask_source": mask_source,
                "runtime": profile.runtime,
                "device": profile.device,
                "accelerator": profile.accelerator_name,
                "precision": precision,
                "hu_clamp": [hu_min, hu_max],
                "uncertainty_units": "approximate_HU_not_calibrated",
            },
            indent=2,
        )
    )
    return output_path, uncertainty_path
