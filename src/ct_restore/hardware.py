from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass

import torch

from ct_restore.config import ExperimentConfig


@dataclass(frozen=True)
class HardwareProfile:
    runtime: str
    device: str
    accelerator_name: str
    cuda_available: bool
    cuda_version: str | None
    compute_capability: tuple[int, int] | None
    total_vram_gb: float
    cpu_count: int
    precision: str
    amp: bool
    patch_size: tuple[int, int, int]
    base_channels: int
    batch_size: int
    grad_accumulation: int
    num_workers: int
    channels_last_3d: bool
    tf32: bool
    compile_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_runtime() -> str:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return "google_colab"
    if os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_GPU_COUNT"):
        return "runpod"
    return "local"


def _cuda_recommendation(vram: float) -> tuple[tuple[int, int, int], int, int, int]:
    if vram < 10:
        return (24, 64, 64), 12, 1, 8
    if vram < 18:
        return (32, 96, 96), 16, 1, 8
    if vram < 28:
        return (48, 112, 112), 20, 1, 4
    if vram < 52:
        return (64, 128, 128), 24, 1, 4
    return (80, 160, 160), 24, 1, 2


def detect_hardware(requested_device: str = "auto") -> HardwareProfile:
    if requested_device not in {"auto", "cpu", "mps", "cuda"} and not requested_device.startswith(
        "cuda:"
    ):
        raise ValueError(
            f"Unsupported device {requested_device!r}; use auto, cpu, mps, or cuda[:N]"
        )
    cpu_count = max(1, os.cpu_count() or 1)
    runtime = detect_runtime()
    cuda = torch.cuda.is_available() and requested_device not in {"cpu", "mps"}
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but torch.cuda.is_available() is false")
    if cuda:
        if requested_device.startswith("cuda:"):
            try:
                index = int(requested_device.split(":", maxsplit=1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device: {requested_device}") from exc
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {index} is outside 0..{torch.cuda.device_count() - 1}"
                )
            torch.cuda.set_device(index)
        else:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        vram = properties.total_memory / 1024**3
        capability = torch.cuda.get_device_capability(index)
        bf16 = capability[0] >= 8 and torch.cuda.is_bf16_supported()
        precision = "bf16" if bf16 else "fp16"
        patch, base, batch, accumulation = _cuda_recommendation(vram)
        workers = min(12, max(2, cpu_count // 2))
        return HardwareProfile(
            runtime=runtime,
            device=f"cuda:{index}",
            accelerator_name=properties.name,
            cuda_available=True,
            cuda_version=torch.version.cuda,
            compute_capability=capability,
            total_vram_gb=round(vram, 2),
            cpu_count=cpu_count,
            precision=precision,
            amp=True,
            patch_size=patch,
            base_channels=base,
            batch_size=batch,
            grad_accumulation=accumulation,
            num_workers=workers,
            channels_last_3d=True,
            tf32=capability[0] >= 8,
            compile_supported=capability[0] >= 7 and platform.system() == "Linux",
        )
    mps = (
        requested_device in {"auto", "mps"}
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if requested_device == "mps" and not mps:
        raise RuntimeError("MPS was explicitly requested but is unavailable")
    if mps:
        return HardwareProfile(
            runtime=runtime,
            device="mps",
            accelerator_name="Apple Metal Performance Shaders",
            cuda_available=False,
            cuda_version=None,
            compute_capability=None,
            total_vram_gb=0.0,
            cpu_count=cpu_count,
            precision="fp32",
            amp=False,
            patch_size=(24, 64, 64),
            base_channels=12,
            batch_size=1,
            grad_accumulation=8,
            num_workers=min(4, max(1, cpu_count // 2)),
            channels_last_3d=False,
            tf32=False,
            compile_supported=False,
        )
    return HardwareProfile(
        runtime=runtime,
        device="cpu",
        accelerator_name=platform.processor() or "CPU",
        cuda_available=False,
        cuda_version=None,
        compute_capability=None,
        total_vram_gb=0.0,
        cpu_count=cpu_count,
        precision="fp32",
        amp=False,
        patch_size=(16, 64, 64),
        base_channels=8,
        batch_size=1,
        grad_accumulation=8,
        num_workers=min(4, max(1, cpu_count // 2)),
        channels_last_3d=False,
        tf32=False,
        compile_supported=False,
    )


def resolve_config(cfg: ExperimentConfig, profile: HardwareProfile) -> ExperimentConfig:
    if cfg.hardware.auto_tune:
        cfg.data.patch_size = profile.patch_size
        cfg.model.base_channels = profile.base_channels
        cfg.train.batch_size = profile.batch_size
        cfg.train.grad_accumulation = profile.grad_accumulation
    if cfg.data.num_workers < 0:
        cfg.data.num_workers = profile.num_workers
    if cfg.hardware.precision == "auto":
        cfg.hardware.precision = profile.precision
    cfg.train.amp = cfg.hardware.precision in {"fp16", "bf16"} and profile.device.startswith("cuda")
    return cfg


def configure_torch(cfg: ExperimentConfig, profile: HardwareProfile) -> None:
    if profile.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = cfg.hardware.cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = cfg.hardware.allow_tf32 and profile.tf32
        torch.backends.cuda.matmul.allow_tf32 = cfg.hardware.allow_tf32 and profile.tf32
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(min(profile.cpu_count, 16))


def autocast_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32
