from pathlib import Path

import pytest

from ct_restore.config import load_config
from ct_restore.data.preprocess import patient_split


def test_patient_split_is_stable_and_bounded() -> None:
    values = [patient_split(f"patient-{index}") for index in range(200)]
    assert values == [patient_split(f"patient-{index}") for index in range(200)]
    assert set(values) == {"train", "val", "test"}


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  imaginary_option: true\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)
