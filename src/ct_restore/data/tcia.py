from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NBIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"

RECOMMENDED_COLLECTIONS = {
    "HNC-IMRT-70-33": {
        "role": "primary radiotherapy planning CT",
        "subjects": 211,
        "approx_size_gb": 23.27,
        "doi": "10.7937/ahqh-xc79",
    },
    "HEAD-NECK-PET-CT": {
        "role": "multi-institution external validation",
        "subjects": 298,
        "approx_size_gb": 72.46,
        "doi": "10.7937/K9/TCIA.2017.8oje5q00",
    },
}


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET",)),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "ct-restore/0.1 research-client"
    return session


def query_ct_series(collection: str) -> list[dict[str, Any]]:
    try:
        response = _session().get(
            NBIA_BASE_URL + "getSeries",
            params={"Collection": collection, "Modality": "CT", "format": "json"},
            timeout=(10, 60),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "TCIA NBIA query failed. Retry later or export a .TCIA manifest with the "
            f"official Data Retriever. Detail: {exc}"
        ) from exc
    if not response.content.strip():
        return []
    result = response.json()
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected TCIA response type: {type(result).__name__}")
    return result


def select_planning_candidates(
    rows: list[dict[str, Any]], minimum_images: int = 80, maximum_slice_thickness: float = 3.0
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        count = int(float(row.get("ImageCount") or row.get("NumberOfImages") or 0))
        thickness_raw = row.get("SliceThickness")
        thickness = float(thickness_raw) if thickness_raw not in (None, "") else None
        description = str(row.get("SeriesDescription", "")).lower()
        is_localizer = any(word in description for word in ("scout", "localizer", "topogram"))
        if count >= minimum_images and not is_localizer:
            if thickness is None or thickness <= maximum_slice_thickness:
                selected.append(row)
    return selected


def write_series_manifest(rows: list[dict[str, Any]], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str))
    return output


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        corrupt_member = handle.testzip()
        if corrupt_member:
            raise RuntimeError(f"CRC failure in TCIA archive member: {corrupt_member}")
        required = sum(member.file_size for member in handle.infolist())
        if required > shutil.disk_usage(destination).free * 0.9:
            raise RuntimeError("Insufficient disk space to safely extract TCIA series")
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in TCIA archive: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"Symlink rejected in TCIA archive: {member.filename}")
        handle.extractall(destination)
    (destination / ".complete").write_text("crc_checked_safe_zip_extraction\n")


def _download_one(uid: str, output_dir: Path) -> str:
    destination = output_dir / uid
    if (destination / ".complete").exists():
        return "already_complete"
    if destination.exists():
        raise RuntimeError(
            f"Incomplete destination exists for {uid}: {destination}. Inspect it before retrying."
        )
    part = output_dir / f".{uid}.zip.part"
    headers: dict[str, str] = {}
    mode = "wb"
    if part.exists() and part.stat().st_size:
        headers["Range"] = f"bytes={part.stat().st_size}-"
        mode = "ab"
    try:
        response = _session().get(
            NBIA_BASE_URL + "getImageWithMD5Hash",
            params={"SeriesInstanceUID": uid},
            headers=headers,
            timeout=(10, 120),
            stream=True,
        )
        response.raise_for_status()
        if mode == "ab" and response.status_code != 206:
            mode = "wb"
        with part.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if not zipfile.is_zipfile(part):
            raise RuntimeError(f"TCIA response for {uid} is not a valid ZIP archive")
        _safe_extract(part, destination)
        part.unlink()
        return "downloaded"
    except (requests.RequestException, OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Failed downloading series {uid}: {exc}") from exc


def download_series(
    rows: list[dict[str, Any]], output_dir: str | Path, limit: int = 0, workers: int = 4
) -> None:
    selected_rows = rows[:limit] if limit > 0 else rows
    uids = [str(row["SeriesInstanceUID"]) for row in selected_rows]
    if not uids:
        raise ValueError("No eligible CT series were selected")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / 1024**3
    if free_gb < 10:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free at {output_dir}; at least 10 GB is required even "
            "for a guarded sample download. Mount external storage."
        )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, uid, output_dir): uid for uid in uids}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                print(f"{uid}: {future.result()}")
            except RuntimeError as exc:
                failures.append(str(exc))
    if failures:
        raise RuntimeError("Some TCIA series failed:\n" + "\n".join(failures))
    write_series_manifest(selected_rows, output_dir / "download_metadata.csv")
