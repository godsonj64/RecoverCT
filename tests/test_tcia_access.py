"""Authenticated and manifest-based retrieval.

The head-and-neck planning collections return nothing over the anonymous API, so the
supported routes are an OAuth token or a `.tcia` manifest from the Data Retriever.
Everything here runs offline; the live credential exchange cannot be tested without a
real NBIA account.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from ct_restore.data import tcia

MANIFEST = """downloadServerUrl=https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage
includeAnnotation=true
noOfrRetry=4
databasketId=manifest-1699999999999.tcia
manifestVersion=3.0
ListOfSeriesToDownload=
1.3.6.1.4.1.14519.5.2.1.7009.2403.111111111111111111111111
1.3.6.1.4.1.14519.5.2.1.7009.2403.222222222222222222222222
1.3.6.1.4.1.14519.5.2.1.7009.2403.222222222222222222222222
"""


def test_manifest_parses_uids_and_skips_the_header(tmp_path: Path) -> None:
    path = tmp_path / "manifest.tcia"
    path.write_text(MANIFEST)
    uids = tcia.parse_tcia_manifest(path)
    assert uids == [
        "1.3.6.1.4.1.14519.5.2.1.7009.2403.111111111111111111111111",
        "1.3.6.1.4.1.14519.5.2.1.7009.2403.222222222222222222222222",
    ]


def test_manifest_handles_a_uid_on_the_list_header_line(tmp_path: Path) -> None:
    path = tmp_path / "m.tcia"
    path.write_text("manifestVersion=3.0\nListOfSeriesToDownload=1.2.3.4.5\n")
    assert tcia.parse_tcia_manifest(path) == ["1.2.3.4.5"]


def test_manifest_rejects_malformed_uids(tmp_path: Path) -> None:
    path = tmp_path / "m.tcia"
    path.write_text("ListOfSeriesToDownload=\n1.2.3.4\nnot-a-uid\n")
    with pytest.raises(ValueError, match="malformed series UID"):
        tcia.parse_tcia_manifest(path)


def test_manifest_rejects_an_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "m.tcia"
    path.write_text("manifestVersion=3.0\nListOfSeriesToDownload=\n")
    with pytest.raises(ValueError, match="No series UIDs"):
        tcia.parse_tcia_manifest(path)


def test_session_attaches_bearer_token_only_when_given() -> None:
    assert "Authorization" not in tcia._session().headers
    assert tcia._session("abc123").headers["Authorization"] == "Bearer abc123"


def test_token_request_sends_the_password_grant(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"access_token": "tok", "token_type": "bearer"},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(tcia.requests, "post", fake_post)
    assert tcia.request_access_token("user", "secret") == "tok"
    assert captured["url"] == tcia.NBIA_TOKEN_URL
    assert captured["data"] == {
        "username": "user",
        "password": "secret",
        "client_id": tcia.NBIA_OAUTH_CLIENT_ID,
        "grant_type": "password",
    }


@pytest.mark.parametrize("status", [400, 401])
def test_token_request_reports_rejected_credentials(monkeypatch, status: int) -> None:
    monkeypatch.setattr(
        tcia.requests,
        "post",
        lambda *a, **k: SimpleNamespace(
            status_code=status, json=dict, raise_for_status=lambda: None
        ),
    )
    with pytest.raises(RuntimeError, match="rejected the credentials"):
        tcia.request_access_token("user", "wrong")


def test_token_request_requires_both_fields() -> None:
    with pytest.raises(ValueError, match="username and password"):
        tcia.request_access_token("user", "")


def test_token_request_wraps_network_failure(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(tcia.requests, "post", boom)
    with pytest.raises(RuntimeError, match="NBIA token request failed"):
        tcia.request_access_token("user", "secret")


def test_query_forwards_the_token(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_session(token=None):
        seen["token"] = token
        return SimpleNamespace(
            get=lambda *a, **k: SimpleNamespace(
                content=b"[]", raise_for_status=lambda: None, json=list
            )
        )

    monkeypatch.setattr(tcia, "_session", fake_session)
    tcia.query_ct_series("HNC-IMRT-70-33", token="tok")
    assert seen["token"] == "tok"


def test_manifest_download_forwards_rows_and_token(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "manifest.tcia"
    path.write_text(MANIFEST)
    seen: dict[str, object] = {}

    def fake_download(rows, output_dir, limit=0, workers=4, token=None):
        seen.update(rows=rows, output_dir=output_dir, limit=limit, token=token)

    monkeypatch.setattr(tcia, "download_series", fake_download)
    returned = tcia.download_manifest_series(path, tmp_path / "raw", limit=1, token="tok")

    assert seen["token"] == "tok"
    assert seen["limit"] == 1
    assert [row["SeriesInstanceUID"] for row in seen["rows"]] == [
        "1.3.6.1.4.1.14519.5.2.1.7009.2403.111111111111111111111111",
        "1.3.6.1.4.1.14519.5.2.1.7009.2403.222222222222222222222222",
    ]
    assert returned == ["1.3.6.1.4.1.14519.5.2.1.7009.2403.111111111111111111111111"]


# --- CLI wiring -------------------------------------------------------------------

MANIFEST_UIDS = [
    "1.3.6.1.4.1.14519.5.2.1.7009.2403.111111111111111111111111",
    "1.3.6.1.4.1.14519.5.2.1.7009.2403.222222222222222222222222",
]


def test_fetch_manifest_command_downloads_without_login(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from ct_restore import cli

    path = tmp_path / "manifest.tcia"
    path.write_text(MANIFEST)
    seen: dict[str, object] = {}

    def fake(manifest_path, output_dir, limit=0, workers=4, token=None):
        seen.update(manifest=Path(manifest_path), token=token, limit=limit)
        return MANIFEST_UIDS

    monkeypatch.setattr(cli, "download_manifest_series", fake)
    result = CliRunner().invoke(
        cli.app, ["fetch-manifest", str(path), "--output-dir", str(tmp_path / "raw")]
    )
    assert result.exit_code == 0, result.output
    assert seen["token"] is None
    assert "Requested 2 series" in result.output


def test_login_reads_credentials_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from ct_restore import cli

    path = tmp_path / "manifest.tcia"
    path.write_text(MANIFEST)
    monkeypatch.setenv("NBIA_USERNAME", "researcher")
    monkeypatch.setenv("NBIA_PASSWORD", "hunter2")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli, "request_access_token", lambda u, p: captured.update(user=u, password=p) or "tok"
    )
    monkeypatch.setattr(
        cli, "download_manifest_series", lambda *a, **k: captured.update(token=k["token"]) or []
    )
    result = CliRunner().invoke(cli.app, ["fetch-manifest", str(path), "--login"])

    assert result.exit_code == 0, result.output
    assert captured["user"] == "researcher"
    assert captured["token"] == "tok"
    assert "hunter2" not in result.output, "password must never be echoed"


def test_tcia_command_exits_cleanly_when_a_collection_is_restricted(monkeypatch) -> None:
    from typer.testing import CliRunner

    from ct_restore import cli

    monkeypatch.setattr(cli, "query_ct_series", lambda *a, **k: [])
    result = CliRunner().invoke(cli.app, ["tcia", "--collection", "HNC-IMRT-70-33"])
    assert result.exit_code == 1
    assert "fetch-manifest" in result.output
