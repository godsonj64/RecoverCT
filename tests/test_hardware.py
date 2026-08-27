from ct_restore.config import ExperimentConfig
from ct_restore.hardware import detect_hardware, detect_runtime, resolve_config


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
    resolved = resolve_config(cfg, profile)
    assert resolved.data.patch_size == profile.patch_size
    assert resolved.model.base_channels == profile.base_channels
    assert resolved.data.num_workers == profile.num_workers
    assert not resolved.train.amp


def test_runtime_environment_detection(monkeypatch) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert detect_runtime() == "google_colab"
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "test")
    assert detect_runtime() == "runpod"
