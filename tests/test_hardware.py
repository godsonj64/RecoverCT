import os

from ct_restore.config import ExperimentConfig
from ct_restore.hardware import (
    EFFECTIVE_BATCH,
    _cuda_recommendation,
    detect_hardware,
    detect_runtime,
    resolve_config,
    usable_cpu_count,
)


def test_cpu_profile_and_auto_tune(monkeypatch) -> None:
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    profile = detect_hardware("cpu")
    assert profile.device == "cpu"
    assert profile.precision == "fp32"
    assert profile.patch_size[0] >= 8
    cfg = ExperimentConfig()
    cfg.hardware.auto_tune = True
    cfg.data.num_workers = -1
    configured_width = cfg.model.base_channels
    resolved = resolve_config(cfg, profile)
    assert resolved.data.patch_size == profile.patch_size
    assert resolved.data.num_workers == profile.num_workers
    assert not resolved.train.amp
    # Auto-tuning must never alter the architecture: a model whose width depends on the
    # GPU produces checkpoints that cannot be loaded on another machine.
    assert resolved.model.base_channels == configured_width


def test_runtime_environment_detection(monkeypatch) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert detect_runtime() == "google_colab"
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "test")
    assert detect_runtime() == "runpod"


def test_auto_tune_holds_the_effective_batch_constant() -> None:
    """More VRAM previously meant a *smaller* effective batch (8 -> 2)."""
    for vram in (8, 16, 24, 40, 80, 141):
        _, batch = _cuda_recommendation(vram)
        accumulation = max(1, EFFECTIVE_BATCH // batch)
        assert batch * accumulation == EFFECTIVE_BATCH, f"{vram} GiB"


def test_auto_tune_patch_size_grows_monotonically_with_vram() -> None:
    previous = 0
    for vram in (8, 16, 24, 40, 80):
        patch, _ = _cuda_recommendation(vram)
        voxels = patch[0] * patch[1] * patch[2]
        assert voxels >= previous
        previous = voxels


def test_usable_cpu_count_respects_a_scheduler_allocation(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert usable_cpu_count() <= 2


def test_usable_cpu_count_is_positive_and_bounded(monkeypatch) -> None:
    for variable in ("SLURM_CPUS_PER_TASK", "NSLOTS", "OMP_NUM_THREADS"):
        monkeypatch.delenv(variable, raising=False)
    count = usable_cpu_count()
    assert count >= 1
    assert count <= (os.cpu_count() or 1)


def test_num_workers_is_clamped_to_available_cpus(monkeypatch) -> None:
    """The shipped configs request 8 workers; Colab runtimes have 2."""
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    profile = detect_hardware("cpu")
    cfg = ExperimentConfig()
    cfg.data.num_workers = profile.cpu_count + 6
    assert resolve_config(cfg, profile).data.num_workers == profile.cpu_count


def test_num_workers_below_the_limit_is_left_alone() -> None:
    profile = detect_hardware("cpu")
    cfg = ExperimentConfig()
    cfg.data.num_workers = 1
    assert resolve_config(cfg, profile).data.num_workers == 1
