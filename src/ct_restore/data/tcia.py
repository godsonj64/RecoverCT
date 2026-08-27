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
NBIA_TOKEN_URL = "https://services.cancerimagingarchive.net/nbia-api/oauth/token"
NBIA_OAUTH_CLIENT_ID = "nbiaRestAPIClient"

RECOMMENDED_COLLECTIONS = {
    "HNC-IMRT-70-33": {
        "role": "primary radiotherapy planning CT",
        "subjects": 211,
        "approx_size_gb": 23.27,
        "doi": "10.7937/ahqh-xc79",
        "access": "restricted: returns no series over the anonymous NBIA API; "
        "requires an NBIA login or the official Data Retriever",
    },
    "HEAD-NECK-PET-CT": {
        "role": "multi-institution external validation",
        "subjects": 298,
        "approx_size_gb": 72.46,
        "doi": "10.7937/K9/TCIA.2017.8oje5q00",
        "access": "restricted: returns no series over the anonymous NBIA API; "
        "requires an NBIA login or the official Data Retriever",
    },
    "Pancreas-CT": {
        "role": "public smoke-test only; abdominal CT, not head-and-neck planning CT",
        "subjects": 82,
        "approx_size_gb": 7.2,
        "doi": "10.7937/K9/TCIA.2016.tNB1kqBU",
        "access": "public: served anonymously by the NBIA API",
    },
}


def _session(token: str | None = None) -> requests.Session:
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
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


def request_access_token(username: str, password: str) -> str:
    """Exchange NBIA credentials for an OAuth access token.

    Restricted collections -- the head-and-neck planning collections among them --
    return nothing anonymously and need this token. Credentials are sent once to
    ``NBIA_TOKEN_URL`` and never stored; pass them from an environment variable or an
    interactive prompt rather than a shell argument.
    """
    if not username or not password:
        raise ValueError("Both an NBIA username and password are required")
    try:
        response = requests.post(
            NBIA_TOKEN_URL,
            data={
                "username": username,
                "password": password,
                "client_id": NBIA_OAUTH_CLIENT_ID,
                "grant_type": "password",
            },
            timeout=(10, 60),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"NBIA token request failed: {exc}") from exc
    if response.status_code in {400, 401}:
        raise RuntimeError(f"NBIA rejected the credentials (HTTP {response.status_code})")
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("NBIA token response contained no access_token")
    return str(token)


def parse_tcia_manifest(path: str | Path) -> list[str]:
    """Series UIDs from a ``.tcia`` manifest exported by the TCIA Data Retriever.

    The file is a small key=value header followed by ``ListOfSeriesToDownload=`` and one
    UID per line. Using it avoids the API entirely for collections that require a login.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    uids: list[str] = []
    in_list = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.replace(" ", "").lower().startswith("listofseriestodownload="):
            in_list = True
            trailing = line.split("=", 1)[1].strip()
            if trailing:
                uids.append(trailing)
            continue
        if not in_list:
            continue
        if "=" in line and not line[0].isdigit():
            continue
        uids.append(line)
    invalid = [uid for uid in uids if not _looks_like_uid(uid)]
    if invalid:
        raise ValueError(f"Manifest contains {len(invalid)} malformed series UID(s): {invalid[:3]}")
    if not uids:
        raise ValueError(f"No series UIDs found in manifest: {path}")
    deduplicated = list(dict.fromkeys(uids))
    return deduplicated


def _looks_like_uid(value: str) -> bool:
    parts = value.split(".")
    return len(parts) >= 3 and all(part.isdigit() for part in parts) and len(value) <= 64


def list_collections(token: str | None = None) -> list[str]:
    """Collection names the NBIA API serves for the current credentials."""
    try:
        response = _session(token).get(
            NBIA_BASE_URL + "getCollectionValues",
            params={"format": "json"},
            timeout=(10, 60),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"TCIA collection listing failed: {exc}") from exc
    if not response.content.strip():
        return []
    return sorted(str(row["Collection"]) for row in response.json())


def query_ct_series(collection: str, token: str | None = None) -> list[dict[str, Any]]:
    try:
        response = _session(token).get(
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


def _download_one(uid: str, output_dir: Path, token: str | None = None) -> str:
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
        response = _session(token).get(
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
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    limit: int = 0,
    workers: int = 4,
    token: str | None = None,
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
        futures = {executor.submit(_download_one, uid, output_dir, token): uid for uid in uids}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                print(f"{uid}: {future.result()}")
            except RuntimeError as exc:
                failures.append(str(exc))
    if failures:
        raise RuntimeError("Some TCIA series failed:\n" + "\n".join(failures))
    write_series_manifest(selected_rows, output_dir / "download_metadata.csv")


def download_manifest_series(
    manifest_path: str | Path,
    output_dir: str | Path,
    limit: int = 0,
    workers: int = 4,
    token: str | None = None,
) -> list[str]:
    """Download every series named by a ``.tcia`` manifest.

    Returns the UIDs attempted. The manifest route is the supported way to obtain
    collections the anonymous API will not serve.
    """
    uids = parse_tcia_manifest(manifest_path)
    rows = [{"SeriesInstanceUID": uid} for uid in uids]
    download_series(rows, output_dir, limit=limit, workers=workers, token=token)
    return uids[:limit] if limit > 0 else uids
