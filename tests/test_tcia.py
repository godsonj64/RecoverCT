import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ct_restore.data import tcia


def test_select_planning_candidates() -> None:
    rows = [
        {"ImageCount": 120, "SliceThickness": 2.0, "SeriesDescription": "RT planning"},
        {"ImageCount": 2, "SliceThickness": 2.0, "SeriesDescription": "scout"},
        {"ImageCount": 120, "SliceThickness": 5.0, "SeriesDescription": "diagnostic"},
    ]
    assert tcia.select_planning_candidates(rows) == [rows[0]]


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(RuntimeError, match="Unsafe path"):
        tcia._safe_extract(archive, tmp_path / "destination")
    assert not (tmp_path / "escape.txt").exists()


def test_download_refuses_low_disk_before_network(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tcia.shutil, "disk_usage", lambda _: SimpleNamespace(free=2 * 1024**3))
    with pytest.raises(RuntimeError, match="Mount external storage"):
        tcia.download_series([{"SeriesInstanceUID": "1.2.3"}], tmp_path)
