"""Acquire a GPCRmd source fixture into a caller-owned cache.

Network access intentionally lives in this script rather than the installed
``mlx_atomistic`` package. GPCRmd file downloads require an account; callers
may supply an already-authorized cookie through ``GPCRMD_COOKIE``. The cookie
value is used in memory only and is never written to the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from http.cookiejar import Cookie, CookieJar
from http.cookies import SimpleCookie
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

from mlx_atomistic.prep.gpcrmd import (
    GPCRMD_API_DOCS_URL,
    GPCRMD_DATA_DOWNLOAD_DOCS_URL,
    GPCRMD_DYNAMICS_METADATA_URL_TEMPLATE,
    GPCRMD_FILE_DOWNLOAD_REQUIRES_ACCOUNT,
    REQUIRED_MLX_IMPORT_FILE_ROLES,
    GPCRmdFile,
    select_gpcrmd_target,
)

MANIFEST_VERSION = 1
DEFAULT_COOKIE_ENV = "GPCRMD_COOKIE"
USER_AGENT = "mlx-atomistic-gpcrmd-acquisition/1"
REPORT_READ_LIMIT_BYTES = 20 * 1024 * 1024
METADATA_READ_LIMIT_BYTES = 2 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
FILE_ID_PATTERN = re.compile(r"\bID\s*:\s*(\d+)\b", flags=re.IGNORECASE)
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")


class AcquisitionFailure(RuntimeError):
    """A normalized fail-closed acquisition error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def blocker(self, *, command: str) -> dict[str, Any]:
        """Return the stable blocker record stored in the raw manifest."""

        return {
            "category": "artifact_source",
            "code": self.code,
            "status": "blocked",
            "prevents_bounded_pass": True,
            "command": command,
            "observed_result": str(self),
            "smallest_reproduction_context": dict(self.details),
            "affected_acceptance_criteria": ["fixture_acquisition_and_integrity"],
            "next_implementation_decision": _next_decision(self.code),
        }


@dataclass(frozen=True)
class ReportLink:
    """One file link parsed from an authenticated GPCRmd report page."""

    file_id: int
    url: str
    text: str


@dataclass(frozen=True)
class ArchiveEntry:
    """A validated archive member before extraction."""

    source_name: str
    normalized_name: str
    kind: str
    size_bytes: int


class _ReportLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._href: str | None = None
        self._text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a" or self._href is not None:
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = " ".join(" ".join(self._text).split())
        self.links.append((self._href, text))
        self._href = None
        self._text = []


def discover_report_links(
    html: str,
    *,
    report_url: str,
    required_file_ids: Sequence[int],
) -> dict[int, ReportLink]:
    """Resolve required file IDs from a GPCRmd simulation report page."""

    parser = _ReportLinkParser()
    parser.feed(html)
    required = set(int(file_id) for file_id in required_file_ids)
    resolved: dict[int, ReportLink] = {}
    duplicates: dict[int, list[str]] = {}
    report_host = urlparse(report_url).hostname

    for href, text in parser.links:
        match = FILE_ID_PATTERN.search(text)
        if match is None:
            continue
        file_id = int(match.group(1))
        if file_id not in required:
            continue
        url = urljoin(report_url, href)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname != report_host:
            raise AcquisitionFailure(
                "untrusted_source_url",
                f"GPCRmd report resolved file {file_id} to an untrusted URL",
                details={"file_id": file_id, "url": url, "report_url": report_url},
            )
        candidate = ReportLink(file_id=file_id, url=url, text=text)
        previous = resolved.get(file_id)
        if previous is not None and previous.url != candidate.url:
            duplicates.setdefault(file_id, [previous.url]).append(candidate.url)
            continue
        resolved[file_id] = candidate

    if duplicates:
        raise AcquisitionFailure(
            "duplicate_source_file_id",
            "GPCRmd report exposed multiple URLs for a required file ID",
            details={"duplicates": duplicates},
        )
    missing = sorted(required - resolved.keys())
    if missing:
        raise AcquisitionFailure(
            "source_layout_missing_file_ids",
            f"GPCRmd report did not expose required file IDs: {missing}",
            details={"missing_file_ids": missing, "report_url": report_url},
        )
    return resolved


def acquire_gpcrmd_fixture(
    *,
    target_id: str,
    cache: str | Path,
    manifest: str | Path,
    source_archive: str | Path | None = None,
    cookie_env: str = DEFAULT_COOKIE_ENV,
    opener: Any | None = None,
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Acquire or verify the required GPCRmd source files."""

    target = select_gpcrmd_target(target_id)
    required_specs = tuple(
        item for item in target.files if item.role in REQUIRED_MLX_IMPORT_FILE_ROLES
    )
    cache_path = Path(cache).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    command = _command_provenance(command_argv)
    previous: Mapping[str, Any] | None = None
    files: dict[int, dict[str, Any]] = {}
    archives: list[dict[str, Any]] = []
    source_metadata = None if previous is None else previous.get("source_metadata")
    session_supplied = bool(os.environ.get(cookie_env))

    payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "gpcrmd_fixture_acquisition",
        "status": "running",
        "target_id": target.target_id,
        "dynamics_id": target.dynamics_id,
        "cache": str(cache_path),
        "manifest": str(manifest_path),
        "required_file_ids": [item.file_id for item in required_specs],
        "required_roles": [item.role for item in required_specs],
        "source": {
            "report_url": target.source_url,
            "metadata_url": GPCRMD_DYNAMICS_METADATA_URL_TEMPLATE.format(
                dynamics_id=target.dynamics_id
            ),
            "download_docs_url": GPCRMD_DATA_DOWNLOAD_DOCS_URL,
            "api_docs_url": GPCRMD_API_DOCS_URL,
            "file_download_requires_account": GPCRMD_FILE_DOWNLOAD_REQUIRES_ACCOUNT,
            "session_cookie_env": cookie_env,
            "session_supplied": session_supplied,
            "session_material_persisted": False,
        },
        "source_metadata": source_metadata,
        "started_at": started_at,
        "completed_at": None,
        "command": command,
        "files": [],
        "archives": [],
        "blockers": [],
    }

    try:
        previous = _load_previous_manifest(manifest_path, target_id=target.target_id)
        payload["source_metadata"] = (
            None if previous is None else previous.get("source_metadata")
        )
        files.update(
            _verify_previous_files(
                previous,
                required_specs=required_specs,
                cache=cache_path,
            )
        )
        if len(files) == len(required_specs):
            archives = list(previous.get("archives", [])) if previous else []
            payload["status"] = "complete"
        elif source_archive is not None:
            source_path = Path(source_archive).expanduser().resolve()
            archive_record, extracted = _acquire_from_source_archive(
                source_path,
                cache=cache_path,
                required_specs=required_specs,
                already_present=files,
            )
            archives.append(archive_record)
            files.update(extracted)
        else:
            active_opener = opener or _build_source_opener(
                cookie_env=cookie_env,
                source_url=target.source_url,
            )
            source_metadata = _fetch_source_metadata(
                active_opener,
                dynamics_id=target.dynamics_id,
                expected_pdb_id=target.pdb_id,
            )
            payload["source_metadata"] = source_metadata
            report_html = _fetch_report_html(active_opener, target.source_url)
            links = discover_report_links(
                report_html,
                report_url=target.source_url,
                required_file_ids=[item.file_id for item in required_specs],
            )
            for spec in required_specs:
                if spec.file_id in files:
                    continue
                files[spec.file_id] = _download_required_file(
                    active_opener,
                    spec=spec,
                    source_url=links[spec.file_id].url,
                    cache=cache_path,
                )

        missing = sorted(
            spec.file_id for spec in required_specs if spec.file_id not in files
        )
        if missing:
            raise AcquisitionFailure(
                "incomplete_fixture",
                f"required GPCRmd file IDs remain missing: {missing}",
                details={"missing_file_ids": missing},
            )

        protocol_spec = next(item for item in required_specs if item.role == "protocol")
        protocol_record = files[protocol_spec.file_id]
        protocol_archive = Path(protocol_record["path"])
        protocol_extraction, protocol_archive_record = _ensure_protocol_extracted(
            protocol_archive,
            cache=cache_path,
            file_id=protocol_spec.file_id,
            previous_entry=_previous_file_entry(previous, protocol_spec.file_id),
        )
        protocol_record["archive_members"] = protocol_extraction
        protocol_record["archive_extraction"] = (
            "reused"
            if protocol_extraction
            and all(item.get("retrieval") == "reused" for item in protocol_extraction)
            else "extracted"
        )
        archives.append(protocol_archive_record)
        payload["status"] = "complete"
    except AcquisitionFailure as exc:
        payload["status"] = "blocked"
        payload["blockers"] = [exc.blocker(command=command["reproduction_command"])]
    except Exception as exc:  # pragma: no cover - last-resort normalization
        failure = AcquisitionFailure(
            "acquisition_internal_error",
            f"{type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__},
        )
        payload["status"] = "blocked"
        payload["blockers"] = [
            failure.blocker(command=command["reproduction_command"])
        ]

    payload["files"] = [files[file_id] for file_id in sorted(files)]
    payload["archives"] = _deduplicate_archive_records(archives)
    payload["completed_at"] = _utc_now()
    _write_json_atomic(manifest_path, payload)
    return payload


def scan_archive(path: str | Path) -> tuple[str, list[ArchiveEntry]]:
    """Validate archive paths and return a collision-free extraction plan."""

    archive_path = Path(path)
    if zipfile.is_zipfile(archive_path):
        kind = "zip"
        with zipfile.ZipFile(archive_path) as archive:
            entries = [
                entry
                for info in archive.infolist()
                if (entry := _zip_entry(info)) is not None
            ]
    elif tarfile.is_tarfile(archive_path):
        kind = "tar"
        with tarfile.open(archive_path, "r:*") as archive:
            entries = [
                entry
                for member in archive.getmembers()
                if (entry := _tar_entry(member)) is not None
            ]
    else:
        raise AcquisitionFailure(
            "unsupported_archive",
            f"source is not a supported zip or tar archive: {archive_path}",
            details={"path": str(archive_path)},
        )
    _validate_archive_collisions(entries)
    return kind, entries


def extract_archive_safely(
    archive_path: str | Path,
    destination: str | Path,
) -> list[dict[str, Any]]:
    """Extract a validated archive atomically below ``destination``."""

    source = Path(archive_path).resolve()
    target = Path(destination).resolve()
    kind, entries = scan_archive(source)
    if target.exists():
        raise AcquisitionFailure(
            "extraction_target_exists",
            f"archive extraction target already exists: {target}",
            details={"path": str(target)},
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    try:
        extracted = _extract_entries(
            source,
            archive_kind=kind,
            entries=entries,
            staging=staging,
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    for item in extracted:
        item["path"] = str(target / item.pop("relative_path"))
    return extracted


def _verify_previous_files(
    previous: Mapping[str, Any] | None,
    *,
    required_specs: Sequence[GPCRmdFile],
    cache: Path,
) -> dict[int, dict[str, Any]]:
    if previous is None:
        unverified = _existing_required_candidates(cache, required_specs)
        if unverified:
            raise AcquisitionFailure(
                "unverified_existing_file",
                "required GPCRmd files exist without an integrity manifest",
                details={"files": [str(path) for path in unverified]},
            )
        return {}

    verified: dict[int, dict[str, Any]] = {}
    for spec in required_specs:
        entry = _previous_file_entry(previous, spec.file_id)
        if entry is None:
            candidates = _existing_required_candidates(cache, [spec])
            if candidates:
                raise AcquisitionFailure(
                    "unverified_existing_file",
                    f"GPCRmd file {spec.file_id} exists without manifest integrity data",
                    details={"file_id": spec.file_id, "files": [str(p) for p in candidates]},
                )
            continue
        path = Path(str(entry.get("path", ""))).expanduser().resolve()
        if not _is_below(path, cache):
            raise AcquisitionFailure(
                "manifest_path_outside_cache",
                f"manifest path for file {spec.file_id} escapes the cache",
                details={"file_id": spec.file_id, "path": str(path), "cache": str(cache)},
            )
        if not path.is_file():
            continue
        expected_hash = str(entry.get("sha256", ""))
        expected_size = entry.get("size_bytes")
        actual_hash = _sha256(path)
        actual_size = path.stat().st_size
        if not expected_hash or expected_size is None:
            raise AcquisitionFailure(
                "manifest_integrity_missing",
                f"manifest lacks hash or size for GPCRmd file {spec.file_id}",
                details={"file_id": spec.file_id, "path": str(path)},
            )
        if actual_hash != expected_hash or actual_size != int(expected_size):
            raise AcquisitionFailure(
                "hash_mismatch",
                f"existing GPCRmd file {spec.file_id} does not match its manifest",
                details={
                    "file_id": spec.file_id,
                    "path": str(path),
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "expected_size_bytes": int(expected_size),
                    "actual_size_bytes": actual_size,
                },
            )
        reused = dict(entry)
        reused["retrieval"] = "reused"
        reused["verified_at"] = _utc_now()
        verified[spec.file_id] = reused
    return verified


def _acquire_from_source_archive(
    source_archive: Path,
    *,
    cache: Path,
    required_specs: Sequence[GPCRmdFile],
    already_present: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if not source_archive.is_file():
        raise AcquisitionFailure(
            "source_archive_missing",
            f"caller-provided GPCRmd source archive does not exist: {source_archive}",
            details={"path": str(source_archive)},
        )
    kind, entries = scan_archive(source_archive)
    candidates: dict[int, list[ArchiveEntry]] = {spec.file_id: [] for spec in required_specs}
    for entry in entries:
        if entry.kind != "file":
            continue
        basename = PurePosixPath(entry.normalized_name).name
        for spec in required_specs:
            if _contains_file_id(basename, spec.file_id):
                candidates[spec.file_id].append(entry)

    missing = [file_id for file_id, found in candidates.items() if not found]
    duplicates = {
        file_id: [item.normalized_name for item in found]
        for file_id, found in candidates.items()
        if len(found) > 1
    }
    if duplicates:
        raise AcquisitionFailure(
            "duplicate_source_file_id",
            "source archive contains multiple candidates for a required GPCRmd file ID",
            details={"duplicates": duplicates, "archive": str(source_archive)},
        )
    if missing:
        raise AcquisitionFailure(
            "source_archive_missing_file_ids",
            f"source archive is missing required GPCRmd file IDs: {missing}",
            details={"missing_file_ids": missing, "archive": str(source_archive)},
        )

    extracted: dict[int, dict[str, Any]] = {}
    for spec in required_specs:
        if spec.file_id in already_present:
            continue
        entry = candidates[spec.file_id][0]
        destination = cache / PurePosixPath(entry.normalized_name).name
        if destination.exists():
            raise AcquisitionFailure(
                "unverified_existing_file",
                f"archive target exists without matching manifest data: {destination}",
                details={"file_id": spec.file_id, "path": str(destination)},
            )
        digest, size = _extract_one_archive_entry(
            source_archive,
            archive_kind=kind,
            entry=entry,
            destination=destination,
        )
        extracted[spec.file_id] = _file_record(
            spec,
            source_url=source_archive.as_uri(),
            resolved_filename=destination.name,
            path=destination,
            size_bytes=size,
            sha256=digest,
            retrieval="extracted",
        )

    archive_record = {
        "role": "gpcrmd_bulk_source",
        "source_url": source_archive.as_uri(),
        "path": str(source_archive),
        "format": kind,
        "size_bytes": source_archive.stat().st_size,
        "sha256": _sha256(source_archive),
        "retrieved_at": _utc_now(),
        "members": [_archive_entry_payload(entry) for entry in entries],
    }
    return archive_record, extracted


def _fetch_source_metadata(
    opener: Any,
    *,
    dynamics_id: int,
    expected_pdb_id: str,
) -> dict[str, Any]:
    url = GPCRMD_DYNAMICS_METADATA_URL_TEMPLATE.format(dynamics_id=dynamics_id)
    response_url, headers, body = _read_url(
        opener,
        url,
        limit=METADATA_READ_LIMIT_BYTES,
        accept="application/json",
    )
    _raise_if_login_response(response_url, body, source_url=url)
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionFailure(
            "source_metadata_invalid",
            "GPCRmd dynamics metadata endpoint returned invalid JSON",
            details={"url": url, "content_type": headers.get("Content-Type")},
        ) from exc
    rows = decoded if isinstance(decoded, list) else [decoded]
    row = next(
        (
            item
            for item in rows
            if isinstance(item, Mapping) and int(item.get("dyn_id", -1)) == dynamics_id
        ),
        None,
    )
    if row is None:
        raise AcquisitionFailure(
            "source_identity_mismatch",
            f"GPCRmd metadata did not return dynamics {dynamics_id}",
            details={"url": url, "dynamics_id": dynamics_id},
        )
    observed_pdb = str(row.get("pdb_namechain", "")).split(".", maxsplit=1)[0]
    if observed_pdb.upper() != expected_pdb_id.upper():
        raise AcquisitionFailure(
            "source_identity_mismatch",
            "GPCRmd metadata PDB identity differs from the curated target",
            details={
                "url": url,
                "expected_pdb_id": expected_pdb_id,
                "observed_pdb_id": observed_pdb,
            },
        )
    return {
        "url": url,
        "retrieved_at": _utc_now(),
        "payload": dict(row),
    }


def _fetch_report_html(opener: Any, report_url: str) -> str:
    response_url, _headers, body = _read_url(
        opener,
        report_url,
        limit=REPORT_READ_LIMIT_BYTES,
        accept="text/html",
    )
    _raise_if_login_response(response_url, body, source_url=report_url)
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AcquisitionFailure(
            "source_report_invalid",
            "GPCRmd simulation report is not valid UTF-8 HTML",
            details={"url": report_url},
        ) from exc


def _download_required_file(
    opener: Any,
    *,
    spec: GPCRmdFile,
    source_url: str,
    cache: Path,
) -> dict[str, Any]:
    request = Request(
        source_url,
        headers={"Accept": "application/octet-stream,*/*", "User-Agent": USER_AGENT},
    )
    partial_path: Path | None = None
    try:
        response = opener.open(request, timeout=120)
        with response:
            response_url = str(response.geturl())
            headers = response.headers
            filename = _resolved_filename(
                headers,
                response_url=response_url,
                fallback=spec.filename_hint,
            )
            if not _contains_file_id(filename, spec.file_id):
                raise AcquisitionFailure(
                    "source_filename_mismatch",
                    f"resolved filename for GPCRmd file {spec.file_id} lacks its file ID",
                    details={
                        "file_id": spec.file_id,
                        "resolved_filename": filename,
                        "url": response_url,
                    },
                )
            destination = cache / filename
            if destination.exists():
                raise AcquisitionFailure(
                    "unverified_existing_file",
                    f"download target exists without matching manifest data: {destination}",
                    details={"file_id": spec.file_id, "path": str(destination)},
                )
            descriptor, partial_name = tempfile.mkstemp(
                prefix=f".{filename}.",
                suffix=".partial",
                dir=cache,
            )
            os.close(descriptor)
            partial_path = Path(partial_name)
            digest = hashlib.sha256()
            size = 0
            prefix = bytearray()
            with partial_path.open("wb") as output:
                while True:
                    chunk = response.read(COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    if len(prefix) < 64 * 1024:
                        prefix.extend(chunk[: 64 * 1024 - len(prefix)])
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            _raise_if_login_response(response_url, bytes(prefix), source_url=source_url)
            expected_size = _content_length(headers)
            if expected_size is not None and size != expected_size:
                raise AcquisitionFailure(
                    "incomplete_download",
                    f"GPCRmd file {spec.file_id} download length mismatch",
                    details={
                        "file_id": spec.file_id,
                        "url": response_url,
                        "expected_size_bytes": expected_size,
                        "actual_size_bytes": size,
                    },
                )
            if size <= 0:
                raise AcquisitionFailure(
                    "incomplete_download",
                    f"GPCRmd file {spec.file_id} download was empty",
                    details={"file_id": spec.file_id, "url": response_url},
                )
            partial_path.replace(destination)
            partial_path = None
            return _file_record(
                spec,
                source_url=response_url,
                resolved_filename=filename,
                path=destination,
                size_bytes=size,
                sha256=digest.hexdigest(),
                retrieval="downloaded",
            )
    except AcquisitionFailure:
        raise
    except HTTPError as exc:
        raise _http_failure(exc, source_url=source_url) from exc
    except URLError as exc:
        raise AcquisitionFailure(
            "source_unavailable",
            f"GPCRmd file {spec.file_id} could not be downloaded: {exc.reason}",
            details={"file_id": spec.file_id, "url": source_url},
        ) from exc
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


def _ensure_protocol_extracted(
    archive_path: Path,
    *,
    cache: Path,
    file_id: int,
    previous_entry: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind, entries = scan_archive(archive_path)
    extraction_root = cache / f"{file_id}_protocol"
    previous_members = (
        [] if previous_entry is None else list(previous_entry.get("archive_members", []))
    )
    if extraction_root.exists():
        verified = _verify_extracted_members(
            extraction_root,
            previous_members=previous_members,
        )
        archive_record = {
            "role": "protocol_start_files",
            "source_url": archive_path.as_uri(),
            "path": str(archive_path),
            "format": kind,
            "size_bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
            "retrieved_at": _utc_now(),
            "members": [_archive_entry_payload(entry) for entry in entries],
            "extraction_root": str(extraction_root),
            "extraction": "reused",
        }
        return verified, archive_record

    extracted = extract_archive_safely(archive_path, extraction_root)
    for item in extracted:
        item["retrieval"] = "extracted"
        item["retrieved_at"] = _utc_now()
    archive_record = {
        "role": "protocol_start_files",
        "source_url": archive_path.as_uri(),
        "path": str(archive_path),
        "format": kind,
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
        "retrieved_at": _utc_now(),
        "members": [_archive_entry_payload(entry) for entry in entries],
        "extraction_root": str(extraction_root),
        "extraction": "extracted",
    }
    return extracted, archive_record


def _verify_extracted_members(
    extraction_root: Path,
    *,
    previous_members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not previous_members:
        raise AcquisitionFailure(
            "unverified_extraction_target",
            "protocol extraction exists without member integrity metadata",
            details={"path": str(extraction_root)},
        )
    verified: list[dict[str, Any]] = []
    for member in previous_members:
        if member.get("kind") != "file":
            reused = dict(member)
            reused["retrieval"] = "reused"
            verified.append(reused)
            continue
        path = Path(str(member.get("path", ""))).resolve()
        if not _is_below(path, extraction_root) or not path.is_file():
            raise AcquisitionFailure(
                "extracted_member_missing",
                "protocol archive member is missing or escapes its extraction root",
                details={"path": str(path), "extraction_root": str(extraction_root)},
            )
        size = path.stat().st_size
        digest = _sha256(path)
        if size != int(member.get("size_bytes", -1)) or digest != member.get("sha256"):
            raise AcquisitionFailure(
                "hash_mismatch",
                "extracted protocol archive member does not match its manifest",
                details={"path": str(path)},
            )
        reused = dict(member)
        reused["retrieval"] = "reused"
        reused["verified_at"] = _utc_now()
        verified.append(reused)
    return verified


def _read_url(
    opener: Any,
    url: str,
    *,
    limit: int,
    accept: str,
) -> tuple[str, Mapping[str, str], bytes]:
    request = Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    try:
        response = opener.open(request, timeout=60)
        with response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise AcquisitionFailure(
                    "source_response_too_large",
                    f"GPCRmd response exceeded {limit} bytes",
                    details={"url": url, "limit_bytes": limit},
                )
            return str(response.geturl()), response.headers, body
    except AcquisitionFailure:
        raise
    except HTTPError as exc:
        raise _http_failure(exc, source_url=url) from exc
    except URLError as exc:
        raise AcquisitionFailure(
            "source_unavailable",
            f"GPCRmd source request failed: {exc.reason}",
            details={"url": url},
        ) from exc


def _http_failure(exc: HTTPError, *, source_url: str) -> AcquisitionFailure:
    if exc.code in {401, 403}:
        return AcquisitionFailure(
            "source_access_login_required",
            "GPCRmd rejected the file request because an authenticated account is required",
            details={
                "url": source_url,
                "http_status": exc.code,
                "account_guide": GPCRMD_API_DOCS_URL,
            },
        )
    return AcquisitionFailure(
        "source_http_error",
        f"GPCRmd source request returned HTTP {exc.code}",
        details={"url": source_url, "http_status": exc.code},
    )


def _raise_if_login_response(response_url: str, body: bytes, *, source_url: str) -> None:
    path = urlparse(response_url).path.rstrip("/")
    text = body[: 256 * 1024].decode("utf-8", errors="ignore").lower()
    login_page = path == "/accounts/login" or (
        "please log in with your account" in text
        and 'action="/accounts/login/' in text
    )
    if login_page:
        raise AcquisitionFailure(
            "source_access_login_required",
            "GPCRmd file access requires an authenticated GPCRmd account",
            details={
                "requested_url": source_url,
                "response_url": response_url,
                "account_guide": GPCRMD_API_DOCS_URL,
                "cookie_env": DEFAULT_COOKIE_ENV,
            },
        )


def _build_source_opener(*, cookie_env: str, source_url: str) -> Any:
    jar = CookieJar()
    raw_cookie = os.environ.get(cookie_env)
    if raw_cookie:
        parsed = SimpleCookie()
        try:
            parsed.load(raw_cookie)
        except Exception as exc:
            raise AcquisitionFailure(
                "invalid_session_cookie",
                f"{cookie_env} is not a valid HTTP Cookie header",
                details={"cookie_env": cookie_env},
            ) from exc
        host = urlparse(source_url).hostname
        if host is None or not parsed:
            raise AcquisitionFailure(
                "invalid_session_cookie",
                f"{cookie_env} did not contain any usable cookies",
                details={"cookie_env": cookie_env},
            )
        for morsel in parsed.values():
            jar.set_cookie(
                Cookie(
                    version=0,
                    name=morsel.key,
                    value=morsel.value,
                    port=None,
                    port_specified=False,
                    domain=host,
                    domain_specified=False,
                    domain_initial_dot=False,
                    path="/",
                    path_specified=True,
                    secure=True,
                    expires=None,
                    discard=True,
                    comment=None,
                    comment_url=None,
                    rest={"HttpOnly": None},
                    rfc2109=False,
                )
            )
    return build_opener(HTTPCookieProcessor(jar))


def _zip_entry(info: zipfile.ZipInfo) -> ArchiveEntry | None:
    if info.is_dir() and _is_archive_root_marker(info.filename):
        return None
    normalized = _normalize_archive_name(info.filename)
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise AcquisitionFailure(
            "archive_link_rejected",
            f"archive contains a link member: {info.filename}",
            details={"member": info.filename},
        )
    if info.is_dir():
        kind = "directory"
    elif file_type not in {0, stat.S_IFREG}:
        raise AcquisitionFailure(
            "archive_special_file_rejected",
            f"archive contains a special member: {info.filename}",
            details={"member": info.filename, "mode": mode},
        )
    else:
        kind = "file"
    return ArchiveEntry(
        source_name=info.filename,
        normalized_name=normalized,
        kind=kind,
        size_bytes=0 if kind == "directory" else int(info.file_size),
    )


def _tar_entry(member: tarfile.TarInfo) -> ArchiveEntry | None:
    if member.isdir() and _is_archive_root_marker(member.name):
        return None
    normalized = _normalize_archive_name(member.name)
    if member.issym() or member.islnk():
        raise AcquisitionFailure(
            "archive_link_rejected",
            f"archive contains a link member: {member.name}",
            details={"member": member.name, "link_target": member.linkname},
        )
    if member.isdir():
        kind = "directory"
    elif member.isreg():
        kind = "file"
    else:
        raise AcquisitionFailure(
            "archive_special_file_rejected",
            f"archive contains a special member: {member.name}",
            details={"member": member.name, "type": repr(member.type)},
        )
    return ArchiveEntry(
        source_name=member.name,
        normalized_name=normalized,
        kind=kind,
        size_bytes=0 if kind == "directory" else int(member.size),
    )


def _is_archive_root_marker(name: str) -> bool:
    if not name or "\x00" in name:
        return False
    portable = name.replace("\\", "/")
    if portable.startswith("/") or WINDOWS_ABSOLUTE_PATTERN.match(portable):
        return False
    return all(part in {"", "."} for part in portable.split("/"))


def _normalize_archive_name(name: str) -> str:
    if not name or "\x00" in name:
        raise AcquisitionFailure(
            "unsafe_archive_path",
            "archive contains an empty or NUL-containing member name",
            details={"member": name},
        )
    portable = name.replace("\\", "/")
    if portable.startswith("/") or WINDOWS_ABSOLUTE_PATTERN.match(portable):
        raise AcquisitionFailure(
            "unsafe_archive_path",
            f"archive contains an absolute member path: {name}",
            details={"member": name},
        )
    raw_parts = portable.split("/")
    if any(part == ".." for part in raw_parts):
        raise AcquisitionFailure(
            "unsafe_archive_path",
            f"archive member traverses outside its extraction root: {name}",
            details={"member": name},
        )
    parts = [part for part in raw_parts if part not in {"", "."}]
    if not parts:
        raise AcquisitionFailure(
            "unsafe_archive_path",
            f"archive member has no usable path: {name}",
            details={"member": name},
        )
    return str(PurePosixPath(*parts))


def _validate_archive_collisions(entries: Sequence[ArchiveEntry]) -> None:
    by_path: dict[str, ArchiveEntry] = {}
    files: set[str] = set()
    for entry in entries:
        key = entry.normalized_name.casefold()
        if key in by_path:
            raise AcquisitionFailure(
                "duplicate_extraction_target",
                "archive contains duplicate extraction targets",
                details={
                    "first": by_path[key].source_name,
                    "second": entry.source_name,
                    "target": entry.normalized_name,
                },
            )
        parts = PurePosixPath(entry.normalized_name).parts
        prefixes = {
            str(PurePosixPath(*parts[:index])).casefold()
            for index in range(1, len(parts))
        }
        parent_file = next((prefix for prefix in prefixes if prefix in files), None)
        if parent_file is not None:
            raise AcquisitionFailure(
                "duplicate_extraction_target",
                "archive places a member below a file target",
                details={"member": entry.source_name, "file_target": parent_file},
            )
        if entry.kind == "file":
            descendant = next(
                (
                    existing.normalized_name
                    for existing_key, existing in by_path.items()
                    if existing_key.startswith(key + "/")
                ),
                None,
            )
            if descendant is not None:
                raise AcquisitionFailure(
                    "duplicate_extraction_target",
                    "archive uses the same target as both a file and directory",
                    details={"file": entry.source_name, "descendant": descendant},
                )
            files.add(key)
        by_path[key] = entry


def _extract_entries(
    source: Path,
    *,
    archive_kind: str,
    entries: Sequence[ArchiveEntry],
    staging: Path,
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    if archive_kind == "zip":
        with zipfile.ZipFile(source) as archive:
            info_by_name = {info.filename: info for info in archive.infolist()}
            for entry in entries:
                extracted.append(
                    _extract_planned_entry(
                        entry,
                        staging=staging,
                        stream=(
                            None
                            if entry.kind == "directory"
                            else archive.open(info_by_name[entry.source_name], "r")
                        ),
                    )
                )
    else:
        with tarfile.open(source, "r:*") as archive:
            member_by_name = {member.name: member for member in archive.getmembers()}
            for entry in entries:
                stream = None
                if entry.kind == "file":
                    stream = archive.extractfile(member_by_name[entry.source_name])
                    if stream is None:
                        raise AcquisitionFailure(
                            "archive_member_unreadable",
                            f"archive member could not be read: {entry.source_name}",
                            details={"member": entry.source_name},
                        )
                extracted.append(
                    _extract_planned_entry(entry, staging=staging, stream=stream)
                )
    return extracted


def _extract_planned_entry(
    entry: ArchiveEntry,
    *,
    staging: Path,
    stream: BinaryIO | None,
) -> dict[str, Any]:
    relative = Path(*PurePosixPath(entry.normalized_name).parts)
    destination = staging / relative
    if entry.kind == "directory":
        destination.mkdir(parents=True, exist_ok=True)
        return {
            "name": entry.source_name,
            "normalized_name": entry.normalized_name,
            "relative_path": str(relative),
            "kind": "directory",
            "size_bytes": 0,
            "sha256": None,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    assert stream is not None
    with stream, destination.open("xb") as output:
        while True:
            chunk = stream.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != entry.size_bytes:
        raise AcquisitionFailure(
            "incomplete_archive_member",
            f"archive member size mismatch: {entry.source_name}",
            details={
                "member": entry.source_name,
                "expected_size_bytes": entry.size_bytes,
                "actual_size_bytes": size,
            },
        )
    return {
        "name": entry.source_name,
        "normalized_name": entry.normalized_name,
        "relative_path": str(relative),
        "kind": "file",
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _extract_one_archive_entry(
    source: Path,
    *,
    archive_kind: str,
    entry: ArchiveEntry,
    destination: Path,
) -> tuple[str, int]:
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    os.close(descriptor)
    partial = Path(partial_name)
    try:
        if archive_kind == "zip":
            with (
                zipfile.ZipFile(source) as archive,
                archive.open(entry.source_name, "r") as stream,
            ):
                digest, size = _copy_stream(stream, partial)
        else:
            with tarfile.open(source, "r:*") as archive:
                member = archive.getmember(entry.source_name)
                stream = archive.extractfile(member)
                if stream is None:
                    raise AcquisitionFailure(
                        "archive_member_unreadable",
                        f"archive member could not be read: {entry.source_name}",
                        details={"member": entry.source_name},
                    )
                with stream:
                    digest, size = _copy_stream(stream, partial)
        if size != entry.size_bytes:
            raise AcquisitionFailure(
                "incomplete_archive_member",
                f"archive member size mismatch: {entry.source_name}",
                details={
                    "member": entry.source_name,
                    "expected_size_bytes": entry.size_bytes,
                    "actual_size_bytes": size,
                },
            )
        partial.replace(destination)
        return digest, size
    finally:
        partial.unlink(missing_ok=True)


def _copy_stream(stream: BinaryIO, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while True:
            chunk = stream.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _resolved_filename(
    headers: Mapping[str, str],
    *,
    response_url: str,
    fallback: str | None,
) -> str:
    disposition = str(headers.get("Content-Disposition", ""))
    match = re.search(
        r"filename\*?=(?:UTF-8''|\")?([^\";]+)",
        disposition,
        flags=re.IGNORECASE,
    )
    candidate = unquote(match.group(1).strip()) if match else ""
    if not candidate:
        candidate = unquote(PurePosixPath(urlparse(response_url).path).name)
    if not candidate:
        candidate = fallback or ""
    if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
        raise AcquisitionFailure(
            "unsafe_resolved_filename",
            "GPCRmd response did not provide a safe resolved filename",
            details={"url": response_url, "resolved_filename": candidate},
        )
    return candidate


def _file_record(
    spec: GPCRmdFile,
    *,
    source_url: str,
    resolved_filename: str,
    path: Path,
    size_bytes: int,
    sha256: str,
    retrieval: str,
) -> dict[str, Any]:
    return {
        "file_id": spec.file_id,
        "role": spec.role,
        "label": spec.label,
        "format_hint": spec.format_hint,
        "filename_hint": spec.filename_hint,
        "source_url": source_url,
        "resolved_filename": resolved_filename,
        "path": str(path.resolve()),
        "size_bytes": int(size_bytes),
        "sha256": sha256,
        "retrieval": retrieval,
        "retrieved_at": _utc_now(),
        "archive_members": [],
    }


def _archive_entry_payload(entry: ArchiveEntry) -> dict[str, Any]:
    return {
        "name": entry.source_name,
        "normalized_name": entry.normalized_name,
        "kind": entry.kind,
        "size_bytes": entry.size_bytes,
    }


def _existing_required_candidates(
    cache: Path,
    specs: Sequence[GPCRmdFile],
) -> list[Path]:
    if not cache.exists():
        return []
    ids = {spec.file_id for spec in specs}
    return sorted(
        path
        for path in cache.rglob("*")
        if path.is_file()
        and not path.name.endswith(".partial")
        and any(_contains_file_id(path.name, file_id) for file_id in ids)
    )


def _contains_file_id(name: str, file_id: int) -> bool:
    return re.search(rf"(?<!\d){file_id}(?!\d)", name) is not None


def _previous_file_entry(
    previous: Mapping[str, Any] | None,
    file_id: int,
) -> Mapping[str, Any] | None:
    if previous is None:
        return None
    for item in previous.get("files", []):
        if isinstance(item, Mapping) and int(item.get("file_id", -1)) == file_id:
            return item
    return None


def _load_previous_manifest(
    path: Path,
    *,
    target_id: str,
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionFailure(
            "manifest_unreadable",
            f"existing acquisition manifest cannot be read: {path}",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise AcquisitionFailure(
            "manifest_unreadable",
            f"existing acquisition manifest is not a JSON object: {path}",
            details={"path": str(path)},
        )
    if payload.get("target_id") != target_id:
        raise AcquisitionFailure(
            "manifest_target_mismatch",
            "existing acquisition manifest belongs to a different target",
            details={
                "path": str(path),
                "expected_target_id": target_id,
                "observed_target_id": payload.get("target_id"),
            },
        )
    return payload


def _deduplicate_archive_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("role", "")), str(record.get("path", "")))
        deduplicated[key] = dict(record)
    return [deduplicated[key] for key in sorted(deduplicated)]


def _content_length(headers: Mapping[str, str]) -> int | None:
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _command_provenance(argv: Sequence[str] | None) -> dict[str, Any]:
    actual = list(sys.argv if argv is None else argv)
    reproduction = shlex.join(["uv", "run", "python", *actual])
    environment = {}
    if os.environ.get("UV_CACHE_DIR"):
        environment["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]
        reproduction = f"UV_CACHE_DIR={shlex.quote(os.environ['UV_CACHE_DIR'])} {reproduction}"
    return {
        "argv": actual,
        "cwd": str(Path.cwd().resolve()),
        "environment": environment,
        "reproduction_command": reproduction,
    }


def _next_decision(code: str) -> str:
    if code == "source_access_login_required":
        return (
            "run the same command with an explicitly authorized GPCRMD_COOKIE environment "
            "value, or download the official dynamics archive and pass --source-archive"
        )
    if code in {
        "hash_mismatch",
        "unverified_existing_file",
        "unverified_extraction_target",
    }:
        return "quarantine the conflicting cache path and reacquire from the official source"
    if code.startswith("unsafe_archive") or code in {
        "archive_link_rejected",
        "archive_special_file_rejected",
        "duplicate_extraction_target",
    }:
        return "do not extract the source; inspect or replace the official archive"
    return "resolve the artifact-source blocker before preparation or runtime execution"


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    os.close(descriptor)
    partial = Path(partial_name)
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="Official GPCRmd downloader archive supplied by the caller.",
    )
    parser.add_argument(
        "--cookie-env",
        default=DEFAULT_COOKIE_ENV,
        help="Environment variable containing an authorized Cookie header.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = acquire_gpcrmd_fixture(
        target_id=args.target_id,
        cache=args.cache,
        manifest=args.manifest,
        source_archive=args.source_archive,
        cookie_env=args.cookie_env,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
