from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "acquire_gpcrmd_fixture.py"
SPEC = importlib.util.spec_from_file_location("acquire_gpcrmd_fixture", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ACQUISITION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ACQUISITION
SPEC.loader.exec_module(ACQUISITION)

TARGET_ID = "gpcrmd-729-beta1-5f8u-cyanopindolol"
METADATA_URL = "https://www.gpcrmd.org/api/search_dyn/info/729"
REPORT_URL = "https://www.gpcrmd.org/dynadb/dynamics/id/729/"


class _Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(body)
        self._url = url
        self.headers = headers or {}

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class _Opener:
    def __init__(self, responses: dict[str, list[_Response]]) -> None:
        self.responses = responses
        self.opened: list[str] = []

    def open(self, request, timeout):
        url = request.full_url
        self.opened.append(url)
        try:
            return self.responses[url].pop(0)
        except (KeyError, IndexError) as exc:
            raise AssertionError(f"unexpected URL: {url}") from exc


class _ExplodingOpener:
    def open(self, request, timeout):
        raise AssertionError(f"network should not be used for verified reuse: {request.full_url}")


def _protocol_archive_bytes(*, member_name: str = "run/input.xsc") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        body = b"# XSC\n0 10 0 0 0 20 0 0 0 30\n"
        info = tarfile.TarInfo(member_name)
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
        config = b"temperature 310\n"
        info = tarfile.TarInfo("run/protocol.conf")
        info.size = len(config)
        archive.addfile(info, io.BytesIO(config))
    return buffer.getvalue()


def _write_bulk_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("dyn_729/15286_dyn_729.psf", "PSF fixture\n")
        archive.writestr("dyn_729/17686_dyn_729.pdb", "ATOM fixture\n")
        archive.writestr("dyn_729/15290_prm_729.prm", "PARAM fixture\n")
        archive.writestr("dyn_729/17687_oth_729.tar.gz", _protocol_archive_bytes())
        archive.writestr("dyn_729/15287_trj_729.xtc", b"optional trajectory")


def _metadata_response() -> _Response:
    body = json.dumps(
        [
            {
                "dyn_id": 729,
                "pdb_namechain": "5F8U",
                "mysoftware": "ACEMD3",
                "software_version": "GPUGRID",
                "forcefield": "CHARMM",
                "forcefield_version": "c36 Jul 2020",
            }
        ]
    ).encode()
    return _Response(body, url=METADATA_URL, headers={"Content-Type": "application/json"})


def _report_html() -> bytes:
    return (
        b'<html><body><a href="/dynadb/files/source/15286_dyn_729.psf">'
        b"<button>Topology file (ID: 15286)</button></a>"
        b'<a href="/dynadb/files/source/17686_dyn_729.pdb">'
        b"<button>Model file (ID: 17686)</button></a>"
        b'<a href="/dynadb/files/source/15290_prm_729.prm">'
        b"<button>Parameters file (ID: 15290)</button></a>"
        b'<a href="/dynadb/files/source/17687_oth_729.tar.gz">'
        b"<button>Others file (ID: 17687)</button></a></body></html>"
    )


def _file_response(url: str, body: bytes, filename: str) -> _Response:
    return _Response(
        body,
        url=url,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(body)),
            "Content-Type": "application/octet-stream",
        },
    )


def test_discover_report_links_resolves_required_file_ids():
    links = ACQUISITION.discover_report_links(
        _report_html().decode(),
        report_url=REPORT_URL,
        required_file_ids=[15286, 17686, 15290, 17687],
    )

    assert sorted(links) == [15286, 15290, 17686, 17687]
    assert links[15286].url.endswith("/15286_dyn_729.psf")
    assert links[17687].text == "Others file (ID: 17687)"


def test_local_source_archive_acquires_hashes_extracts_protocol_and_reuses(tmp_path: Path):
    source = tmp_path / "gpcrmd-729.zip"
    cache = tmp_path / "cache"
    manifest = tmp_path / "results" / "fixture-manifest.json"
    _write_bulk_archive(source)

    first = ACQUISITION.acquire_gpcrmd_fixture(
        target_id=TARGET_ID,
        cache=cache,
        manifest=manifest,
        source_archive=source,
        command_argv=["scripts/acquire_gpcrmd_fixture.py", "--source-archive", str(source)],
    )

    assert first["status"] == "complete"
    assert [item["file_id"] for item in first["files"]] == [15286, 15290, 17686, 17687]
    assert all(len(item["sha256"]) == 64 for item in first["files"])
    assert all(item["size_bytes"] > 0 for item in first["files"])
    protocol = next(item for item in first["files"] if item["file_id"] == 17687)
    extracted_files = [
        item for item in protocol["archive_members"] if item["kind"] == "file"
    ]
    assert {item["normalized_name"] for item in extracted_files} == {
        "run/input.xsc",
        "run/protocol.conf",
    }
    assert all(Path(item["path"]).is_file() for item in extracted_files)
    assert {item["role"] for item in first["archives"]} == {
        "gpcrmd_bulk_source",
        "protocol_start_files",
    }
    assert "uv run python scripts/acquire_gpcrmd_fixture.py" in first["command"][
        "reproduction_command"
    ]

    second = ACQUISITION.acquire_gpcrmd_fixture(
        target_id=TARGET_ID,
        cache=cache,
        manifest=manifest,
        opener=_ExplodingOpener(),
        command_argv=["scripts/acquire_gpcrmd_fixture.py", "--cache", str(cache)],
    )

    assert second["status"] == "complete"
    assert all(item["retrieval"] == "reused" for item in second["files"])
    protocol = next(item for item in second["files"] if item["file_id"] == 17687)
    assert all(item["retrieval"] == "reused" for item in protocol["archive_members"])
    assert json.loads(manifest.read_text()) == second


def test_existing_hash_mismatch_fails_closed_without_overwrite(tmp_path: Path):
    source = tmp_path / "gpcrmd-729.zip"
    cache = tmp_path / "cache"
    manifest = tmp_path / "fixture-manifest.json"
    _write_bulk_archive(source)
    first = ACQUISITION.acquire_gpcrmd_fixture(
        target_id=TARGET_ID,
        cache=cache,
        manifest=manifest,
        source_archive=source,
    )
    assert first["status"] == "complete"
    topology = cache / "15286_dyn_729.psf"
    topology.write_text("modified\n")

    second = ACQUISITION.acquire_gpcrmd_fixture(
        target_id=TARGET_ID,
        cache=cache,
        manifest=manifest,
        source_archive=source,
    )

    assert second["status"] == "blocked"
    assert second["blockers"][0]["code"] == "hash_mismatch"
    assert topology.read_text() == "modified\n"


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:\\escape"])
def test_archive_paths_fail_closed(tmp_path: Path, member: str):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, b"unsafe")

    with pytest.raises(ACQUISITION.AcquisitionFailure, match="archive") as exc_info:
        ACQUISITION.scan_archive(archive_path)

    assert exc_info.value.code == "unsafe_archive_path"
    assert not (tmp_path / "escape").exists()


def test_archive_escaping_link_fails_closed(tmp_path: Path):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("safe/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape"
        archive.addfile(info)

    with pytest.raises(ACQUISITION.AcquisitionFailure) as exc_info:
        ACQUISITION.scan_archive(archive_path)

    assert exc_info.value.code == "archive_link_rejected"
    assert exc_info.value.details["link_target"] == "../../escape"


def test_archive_normalized_target_collisions_fail_closed(tmp_path: Path):
    archive_path = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a/b", b"one")
        archive.writestr("a\\b", b"two")

    with pytest.raises(ACQUISITION.AcquisitionFailure) as exc_info:
        ACQUISITION.scan_archive(archive_path)

    assert exc_info.value.code == "duplicate_extraction_target"


def test_tar_root_directory_marker_is_ignored(tmp_path: Path):
    archive_path = tmp_path / "root-marker.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        body = b"source protocol\n"
        info = tarfile.TarInfo("./run/protocol.conf")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))

    kind, entries = ACQUISITION.scan_archive(archive_path)
    extracted = ACQUISITION.extract_archive_safely(
        archive_path,
        tmp_path / "extracted",
    )

    assert kind == "tar"
    assert [entry.normalized_name for entry in entries] == ["run/protocol.conf"]
    assert [item["normalized_name"] for item in extracted] == ["run/protocol.conf"]
    assert (tmp_path / "extracted" / "run" / "protocol.conf").read_bytes() == body


def test_login_requirement_is_normalized_without_persisting_cookie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "sessionid=do-not-persist-this-value"
    monkeypatch.setenv("GPCRMD_COOKIE", secret)
    login_url = "https://www.gpcrmd.org/accounts/login/?next=/dynadb/dynamics/id/729/"
    login_html = (
        b'<form action="/accounts/login/" method="post">'
        b"Please log in with your account to access downloading data."
        b"</form>"
    )
    opener = _Opener(
        {
            METADATA_URL: [_metadata_response()],
            REPORT_URL: [_Response(login_html, url=login_url)],
        }
    )
    manifest = tmp_path / "fixture-manifest.json"

    payload = ACQUISITION.acquire_gpcrmd_fixture(
        target_id=TARGET_ID,
        cache=tmp_path / "cache",
        manifest=manifest,
        opener=opener,
    )

    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["code"] == "source_access_login_required"
    assert payload["source"]["session_supplied"] is True
    assert payload["source"]["session_material_persisted"] is False
    assert secret not in manifest.read_text()


def test_incomplete_download_is_removed_and_recorded(tmp_path: Path):
    topology_url = "https://www.gpcrmd.org/dynadb/files/source/15286_dyn_729.psf"
    topology = _file_response(topology_url, b"short", "15286_dyn_729.psf")
    topology.headers["Content-Length"] = "100"
    opener = _Opener(
        {
            METADATA_URL: [_metadata_response()],
            REPORT_URL: [_Response(_report_html(), url=REPORT_URL)],
            topology_url: [topology],
        }
    )
    cache = tmp_path / "cache"

    payload = ACQUISITION.acquire_gpcrmd_fixture(
        target_id=TARGET_ID,
        cache=cache,
        manifest=tmp_path / "fixture-manifest.json",
        opener=opener,
    )

    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["code"] == "incomplete_download"
    assert not (cache / "15286_dyn_729.psf").exists()
    assert list(cache.glob("*.partial")) == []


def test_live_acquisition_paths_are_gitignored():
    paths = [
        "notebooks/ligand-receptor-motion/data/gpcrmd-cache/729/15286_dyn_729.psf",
        "results/gpcrmd-pme-runtime-closure/source/fixture-manifest.json",
    ]
    result = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert set(result.stdout.splitlines()) == set(paths)
