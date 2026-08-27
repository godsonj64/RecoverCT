from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str = "data/manifests/train.csv"
    patch_size: tuple[int, int, int] = (64, 128, 128)
    hu_min: float = -1024.0
    hu_max: float = 3071.0
    target_spacing: tuple[float, float, float] = (1.5, 1.0, 1.0)
    artifact_probability: float = 0.9
    num_workers: int = -1


@dataclass
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 24
    levels: int = 4
    blocks_per_level: int = 2
    dropout: float = 0.0
    # 1.0 leaves the residual unconstrained outside the artifact mask (original
    # behaviour); 0.0 freezes non-artifact voxels exactly.
    residual_leak: float = 1.0


@dataclass
class TrainConfig:
    stage: str = "head_neck_finetune"
    epochs: int = 200
    batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_accumulation: int = 4
    amp: bool = True
    ema_decay: float = 0.999
    seed: int = 2026
    checkpoint_dir: str = "checkpoints"
    resume: str | None = None
    pretrained: str | None = None
    validate_every: int = 1


@dataclass
class HardwareConfig:
    device: str = "auto"
    precision: str = "auto"
    auto_tune: bool = False
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    channels_last_3d: bool = True
    compile: bool = False
    prefetch_factor: int = 2


@dataclass
class LossConfig:
    artifact_l1: float = 5.0
    global_l1: float = 1.0
    gradient: float = 0.5
    ssim: float = 0.5
    identity: float = 2.0
    uncertainty: float = 0.1


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    allowed = set(obj.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys for {type(obj).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        if isinstance(value, list) and key in {"patch_size", "target_spacing"}:
            value = tuple(value)
        setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = ExperimentConfig()
    allowed_sections = {"data", "model", "train", "loss", "hardware"}
    unknown = set(raw) - allowed_sections
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    for section in allowed_sections:
        if section in raw:
            _merge_dataclass(getattr(cfg, section), raw[section])
    return cfg
