from __future__ import annotations

import math

import numpy as np


def image_metrics(
    prediction_hu: np.ndarray,
    target_hu: np.ndarray,
    artifact_mask: np.ndarray | None = None,
) -> dict[str, float]:
    prediction_hu = np.asarray(prediction_hu, dtype=np.float64)
    target_hu = np.asarray(target_hu, dtype=np.float64)
    if prediction_hu.shape != target_hu.shape:
        raise ValueError("prediction and target must have equal shapes")
    error = prediction_hu - target_hu
    values = {
        "mae_hu": float(np.mean(np.abs(error))),
        "rmse_hu": float(np.sqrt(np.mean(error**2))),
        "bias_hu": float(np.mean(error)),
        "p95_abs_error_hu": float(np.percentile(np.abs(error), 95)),
    }
    dynamic_range = 4095.0
    values["psnr_db"] = float(20 * math.log10(dynamic_range / max(values["rmse_hu"], 1e-8)))
    if artifact_mask is not None:
        mask = np.asarray(artifact_mask).astype(bool)
        if mask.any():
            values["artifact_mae_hu"] = float(np.mean(np.abs(error[mask])))
            values["artifact_bias_hu"] = float(np.mean(error[mask]))
        if (~mask).any():
            values["known_region_mae_hu"] = float(np.mean(np.abs(error[~mask])))
    tissue_ranges = {
        "air": (-1024, -500),
        "soft_tissue": (-200, 200),
        "bone": (200, 2000),
        "dense_bone": (2000, 3071),
    }
    for name, (lower, upper) in tissue_ranges.items():
        region = (target_hu >= lower) & (target_hu < upper)
        if region.any():
            values[f"{name}_mae_hu"] = float(np.mean(np.abs(error[region])))
            values[f"{name}_bias_hu"] = float(np.mean(error[region]))
    return values
