"""Shared path-independent identity and atomic artifact publication primitives."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

GENERATION_SCHEMA = "mlx-atomistic.atomic-generation.v1"
GENERATION_MANIFEST = "artifact-manifest.json"
SOURCE_INVENTORY_SCHEMA = "mlx-atomistic.source-inventory.v1"


def _is_temporary_generation_name(name: str) -> bool:
    return name.startswith(".") and ".tmp-" in name


class ArtifactIntegrityError(ValueError):
    """Raised when a generated artifact fails strict integrity validation."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON value into deterministic, finite canonical bytes.

    Args:
        value: JSON-compatible value to encode.

    Returns:
        UTF-8 bytes with sorted keys and no insignificant whitespace.

    Raises:
        TypeError: If ``value`` is not JSON serializable.
        ValueError: If ``value`` contains NaN or infinity.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hexadecimal digest of ``payload``.

    Args:
        payload: Bytes to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one regular file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        ValueError: If ``path`` is not a regular non-symlink file.
    """

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        msg = f"artifact payload must be a regular non-symlink file: {source}"
        raise ValueError(msg)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confined_path(root: str | Path, relative_path: str, *, must_exist: bool = False) -> Path:
    """Resolve a portable relative path while confining it below ``root``.

    Args:
        root: Artifact or source root.
        relative_path: POSIX-style relative logical path.
        must_exist: Require the resolved path to exist. Defaults to ``False``.

    Returns:
        Resolved filesystem path below ``root``.

    Raises:
        ValueError: If the path is absolute, non-canonical, or escapes ``root``.
        FileNotFoundError: If ``must_exist`` is true and the path is absent.
    """

    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        msg = "artifact paths must be non-empty POSIX relative strings"
        raise ValueError(msg)
    logical = PurePosixPath(relative_path)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        msg = f"artifact path is not confined: {relative_path!r}"
        raise ValueError(msg)
    if logical.as_posix() != relative_path:
        msg = f"artifact path is not canonical: {relative_path!r}"
        raise ValueError(msg)
    resolved_root = Path(root).resolve()
    candidate = resolved_root.joinpath(*logical.parts).resolve(strict=False)
    if not candidate.is_relative_to(resolved_root):
        msg = f"artifact path escapes its root: {relative_path!r}"
        raise ValueError(msg)
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def resource_record(role: str, payload: bytes) -> dict[str, object]:
    """Build a logical path-free resource identity record.

    Args:
        role: Stable semantic resource role.
        payload: Exact canonical resource bytes.

    Returns:
        Role, byte size, and SHA-256 record.

    Raises:
        ValueError: If ``role`` is empty.
    """

    if not role:
        msg = "resource role must be non-empty"
        raise ValueError(msg)
    return {"role": role, "byte_size": len(payload), "sha256": sha256_bytes(payload)}


def source_inventory(
    repo_root: str | Path,
    *,
    logical_paths: Iterable[str] = (),
    recursive_roots: Iterable[str] = (),
) -> list[dict[str, object]]:
    """Build a sorted repo-relative inventory of source files.

    Args:
        repo_root: Repository root that owns all logical paths.
        logical_paths: Explicit repo-relative files to include.
        recursive_roots: Repo-relative directories whose ``*.py`` files are included.

    Returns:
        Sorted records containing logical path, byte size, and SHA-256.

    Raises:
        ValueError: If a path escapes the repository or is not a regular file.
    """

    root = Path(repo_root).resolve()
    selected: set[str] = set()
    for logical_path in logical_paths:
        selected.add(PurePosixPath(logical_path).as_posix())
    for logical_root in recursive_roots:
        directory = confined_path(root, PurePosixPath(logical_root).as_posix(), must_exist=True)
        if directory.is_symlink() or not directory.is_dir():
            msg = f"source inventory root must be a regular directory: {logical_root}"
            raise ValueError(msg)
        for path in directory.rglob("*.py"):
            if path.is_symlink() or not path.is_file():
                msg = f"source inventory entries must be regular files: {path}"
                raise ValueError(msg)
            selected.add(path.relative_to(root).as_posix())
    records: list[dict[str, object]] = []
    for logical_path in sorted(selected):
        path = confined_path(root, logical_path, must_exist=True)
        if path.is_symlink() or not path.is_file():
            msg = f"source inventory entry must be a regular file: {logical_path}"
            raise ValueError(msg)
        records.append(
            {
                "path": logical_path,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def inventory_fingerprint(scope: str, files: Iterable[Mapping[str, object]]) -> str:
    """Hash a named source or resource inventory.

    Args:
        scope: Stable inventory scope identifier.
        files: Canonically ordered inventory records.

    Returns:
        SHA-256 digest of the canonical inventory envelope.
    """

    envelope = {
        "schema_version": SOURCE_INVENTORY_SCHEMA,
        "scope": scope,
        "files": list(files),
    }
    return sha256_bytes(canonical_json_bytes(envelope))


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _raise_rename_error(error_number: int, source: Path, destination: Path) -> None:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source} -> {destination}",
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination is absent."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    system = platform.system()
    if system == "Darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        if renamex_np(source_bytes, destination_bytes, 0x00000004) != 0:
            _raise_rename_error(ctypes.get_errno(), source, destination)
        return
    if system == "Linux" and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        if renameat2(-100, source_bytes, -100, destination_bytes, 1) != 0:
            _raise_rename_error(ctypes.get_errno(), source, destination)
        return
    msg = "atomic no-replace directory rename is unavailable on this platform"
    raise OSError(errno.ENOTSUP, msg, destination)


def _payload_inventory(root: Path, *, synchronize: bool = False) -> list[dict[str, object]]:
    def walk_error(error: OSError) -> None:
        msg = f"artifact payload traversal failed: {error}"
        raise ArtifactIntegrityError(msg) from error

    records: list[dict[str, object]] = []
    for directory, names, files in os.walk(
        root,
        followlinks=False,
        onerror=walk_error,
    ):
        directory_path = Path(directory)
        for name in names:
            if (directory_path / name).is_symlink():
                msg = f"artifact directories may not be symlinks: {directory_path / name}"
                raise ArtifactIntegrityError(msg)
        for name in files:
            path = directory_path / name
            if path.name == GENERATION_MANIFEST and path.parent == root:
                continue
            if path.is_symlink() or not path.is_file():
                msg = f"artifact payload must be a regular file: {path}"
                raise ArtifactIntegrityError(msg)
            if path.stat().st_nlink != 1:
                msg = f"artifact payload may not be hard-linked: {path}"
                raise ArtifactIntegrityError(msg)
            logical_path = path.relative_to(root).as_posix()
            confined_path(root, logical_path, must_exist=True)
            if synchronize:
                _sync_file(path)
            records.append(
                {
                    "path": logical_path,
                    "role": "payload",
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sorted(records, key=lambda record: str(record["path"]))


@dataclass
class AtomicGeneration:
    """Build and exclusively publish one checksummed artifact generation.

    Args:
        destination: Previously absent final generation directory.
        artifact_kind: Stable artifact kind identifier.
        artifact_schema_version: Payload schema interpreted by the caller.
        identity: Path-independent identity fields recorded in the envelope.
        metadata: Additional JSON-safe non-identity metadata.
        fault_hook: Optional stage callback used by deterministic fault tests.
    """

    destination: Path
    artifact_kind: str
    artifact_schema_version: str
    identity: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    fault_hook: Callable[[str], None] | None = None
    _temporary: Path | None = field(default=None, init=False, repr=False)
    _published: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.destination = Path(self.destination).expanduser().resolve(strict=False)
        if not self.artifact_kind or not self.artifact_schema_version:
            msg = "artifact kind and payload schema must be non-empty"
            raise ValueError(msg)
        if _is_temporary_generation_name(self.destination.name):
            msg = "artifact destinations may not use the controlled temporary namespace"
            raise ValueError(msg)

    def __enter__(self) -> AtomicGeneration:
        """Create the same-parent temporary generation and return this builder."""

        parent = self.destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists() or self.destination.is_symlink():
            raise FileExistsError(self.destination)
        temporary = tempfile.mkdtemp(prefix=f".{self.destination.name}.tmp-", dir=parent)
        self._temporary = Path(temporary)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove an unpublished controlled temporary generation."""

        del exc_type, exc, traceback
        if not self._published and self._temporary is not None:
            shutil.rmtree(self._temporary, ignore_errors=True)
            self._temporary = None

    @property
    def root(self) -> Path:
        """Return the active temporary generation root."""

        if self._temporary is None:
            msg = "atomic generation has not been entered"
            raise RuntimeError(msg)
        return self._temporary

    def path(self, relative_path: str) -> Path:
        """Return a confined writable payload path in the temporary generation.

        Args:
            relative_path: Canonical POSIX relative payload path.

        Returns:
            Writable path whose parent directories have been created.
        """

        path = confined_path(self.root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        """Write, flush, and synchronize one payload.

        Args:
            relative_path: Canonical relative payload path.
            payload: Exact payload bytes.

        Returns:
            Temporary payload path.

        Raises:
            FileExistsError: If the payload path already exists.
        """

        path = self.path(relative_path)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def write_json(self, relative_path: str, payload: object) -> Path:
        """Write one canonical JSON payload.

        Args:
            relative_path: Canonical relative payload path.
            payload: JSON-compatible finite value.

        Returns:
            Temporary payload path.
        """

        return self.write_bytes(relative_path, canonical_json_bytes(payload) + b"\n")

    def publish(self) -> dict[str, object]:
        """Publish the completed generation with an exclusive atomic rename.

        Returns:
            The completed artifact manifest.

        Raises:
            FileExistsError: If the destination exists.
            ArtifactIntegrityError: If a payload is unsafe.
        """

        if self._published:
            msg = "atomic generation has already been published"
            raise RuntimeError(msg)
        temporary = self.root
        manifest_path = temporary / GENERATION_MANIFEST
        if manifest_path.exists() or manifest_path.is_symlink():
            msg = f"payloads may not reserve {GENERATION_MANIFEST}"
            raise ArtifactIntegrityError(msg)
        files = _payload_inventory(temporary, synchronize=True)
        if self.fault_hook is not None:
            self.fault_hook("after_payload_sync")
        unsigned: dict[str, object] = {
            "schema_version": GENERATION_SCHEMA,
            "artifact_kind": self.artifact_kind,
            "artifact_schema_version": self.artifact_schema_version,
            "complete": True,
            "identity": dict(self.identity),
            "metadata": dict(self.metadata),
            "files": files,
        }
        manifest = dict(unsigned)
        manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
        with manifest_path.open("xb") as handle:
            handle.write(canonical_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(temporary)
        if self.fault_hook is not None:
            self.fault_hook("after_manifest")

        if self.fault_hook is not None:
            self.fault_hook("before_rename")
        _rename_noreplace(temporary, self.destination)
        self._temporary = None
        self._published = True
        _sync_directory(self.destination.parent)
        if self.fault_hook is not None:
            self.fault_hook("after_rename")
        return manifest


def _generation_root(artifact: str | Path) -> Path:
    requested = Path(artifact).expanduser()
    if requested.is_symlink():
        msg = f"artifact root or payload may not be a symlink: {artifact}"
        raise ArtifactIntegrityError(msg)
    path = requested.resolve(strict=False)
    if path.name == GENERATION_MANIFEST or path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        manifest = candidate / GENERATION_MANIFEST
        if manifest.is_file() and not manifest.is_symlink():
            if _is_temporary_generation_name(candidate.name):
                msg = f"unpublished temporary generation is not inspectable: {candidate}"
                raise ArtifactIntegrityError(msg)
            return candidate
    msg = f"artifact generation manifest not found: {artifact}"
    raise ArtifactIntegrityError(msg)


def generation_root(artifact: str | Path) -> Path:
    """Return the completed generation root containing ``artifact``.

    Args:
        artifact: Generation directory, manifest, or nested payload path.

    Returns:
        Nearest ancestor containing the completion manifest.
    """

    return _generation_root(artifact)


def inspect_generation(artifact: str | Path) -> dict[str, object]:
    """Validate one completed generation without modifying it.

    Args:
        artifact: Generation directory, manifest, or direct payload path.

    Returns:
        Validated artifact manifest.

    Raises:
        ArtifactIntegrityError: If schema, confinement, inventory, or checksums fail.
    """

    root = _generation_root(artifact)
    manifest_path = root / GENERATION_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        msg = "artifact completion manifest is missing or unsafe"
        raise ArtifactIntegrityError(msg)
    if manifest_path.stat().st_nlink != 1:
        msg = "artifact completion manifest may not be hard-linked"
        raise ArtifactIntegrityError(msg)
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        msg = "artifact completion manifest is not valid JSON"
        raise ArtifactIntegrityError(msg) from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != GENERATION_SCHEMA:
        msg = "unsupported artifact generation schema"
        raise ArtifactIntegrityError(msg)
    if manifest.get("complete") is not True:
        msg = "artifact generation is incomplete"
        raise ArtifactIntegrityError(msg)
    declared_manifest_hash = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    observed_manifest_hash = sha256_bytes(canonical_json_bytes(unsigned))
    if declared_manifest_hash != observed_manifest_hash:
        msg = "artifact manifest checksum mismatch"
        raise ArtifactIntegrityError(msg)
    declared_files = manifest.get("files")
    if not isinstance(declared_files, list):
        msg = "artifact file inventory is missing"
        raise ArtifactIntegrityError(msg)
    observed_files = _payload_inventory(root)
    if declared_files != observed_files:
        msg = "artifact payload inventory or checksum mismatch"
        raise ArtifactIntegrityError(msg)
    return manifest


def read_generation_json(artifact: str | Path, relative_path: str) -> Any:
    """Read a JSON payload after validating its complete generation.

    Args:
        artifact: Generation directory, manifest, or direct payload path.
        relative_path: Confined relative JSON payload path.

    Returns:
        Parsed JSON value.
    """

    root = _generation_root(artifact)
    inspect_generation(root)
    payload_path = confined_path(root, relative_path, must_exist=True)
    if payload_path.is_symlink() or not payload_path.is_file():
        msg = f"artifact JSON payload is not a regular file: {relative_path}"
        raise ArtifactIntegrityError(msg)
    return json.loads(payload_path.read_bytes())
