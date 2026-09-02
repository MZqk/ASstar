#!/usr/bin/env python3
"""Embed, seal, and verify Starun's signed CPython 3.12 runtime payload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.native_pipeline_runtime import (  # noqa: E402
    NATIVE_BUILD_MANIFEST_NAME,
    NATIVE_BUILD_MANIFEST_SCHEMA,
    NATIVE_EXTENSION_SUFFIX,
    NATIVE_MODULES,
    NATIVE_RUNTIME_MANIFEST_NAME,
    NATIVE_RUNTIME_MANIFEST_SCHEMA,
    NativePipelineValidationError,
    canonical_payload_sha256,
    probe_native_imports,
    sha256_file,
    unsigned_macho_sha256,
    validate_native_runtime_payload,
)


SOURCE_BUILD_MANIFEST_NAME = "native-pipeline-manifest.json"
NATIVE_EMBED_RECEIPT_NAME = ".native-pipeline-embed-receipt.json"
NATIVE_EMBED_RECEIPT_SCHEMA = "starun.native-pipeline-embed-receipt.v1"

# These standalone build-payload blockers are resolved (or replaced with a
# more precise App/DMG-layer blocker) when the payload is embedded and sealed.
_APP_EMBEDDING_RESOLVED_BLOCKERS = frozenset(
    {
        "technical_poc_not_dmg_release",
        "developer_id_signature_missing",
        "developer_id_signature_not_verified",
        "notarization_missing",
        "legal_notice_bundle_missing",
        "app_embedding_requires_final_resign_and_rehash",
    }
)


def _absolute_path_without_symlinks(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    for candidate in reversed((absolute, *absolute.parents)):
        if candidate.is_symlink():
            raise NativePipelineValidationError(
                f"{label} contains a symbolic-link component: {candidate}"
            )
    return absolute


def _secure_copy_to_directory(source: Path, directory_fd: int, filename: str) -> None:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise NativePipelineValidationError(f"unsafe destination filename: {filename!r}")
    source_path = Path(source)
    if source_path.is_symlink() or not source_path.is_file():
        raise NativePipelineValidationError(
            f"copy source is missing or is a symlink: {source_path}"
        )
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source_path, read_flags)
    temporary_name = f".{filename}.tmp-{uuid.uuid4().hex}"
    destination_fd = -1
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise NativePipelineValidationError(
                f"copy source is not a regular file: {source_path}"
            )
        write_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        destination_fd = os.open(
            temporary_name,
            write_flags,
            stat.S_IMODE(source_stat.st_mode) or 0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(os.dup(source_fd), "rb") as source_handle, os.fdopen(
            os.dup(destination_fd), "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
            destination_handle.flush()
        os.fsync(destination_fd)
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _secure_write_to_directory(directory_fd: int, filename: str, data: bytes) -> None:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise NativePipelineValidationError(f"unsafe destination filename: {filename!r}")
    temporary_name = f".{filename}.tmp-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            handle.write(data)
            handle.flush()
        os.fsync(descriptor)
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _remove_destination_entry(directory_fd: int, filename: str) -> None:
    try:
        item_stat = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(item_stat.st_mode):
        raise NativePipelineValidationError(
            f"refusing to replace a destination directory: {filename}"
        )
    os.unlink(filename, dir_fd=directory_fd)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NativePipelineValidationError(f"manifest is missing or is a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativePipelineValidationError(f"manifest is unreadable: {path}: {error}") from error
    if not isinstance(value, dict):
        raise NativePipelineValidationError(f"manifest root must be an object: {path}")
    return value


def _verify_manifest_payload(manifest: Mapping[str, Any], *, schema: str) -> None:
    if manifest.get("schema") != schema:
        raise NativePipelineValidationError(
            f"unexpected build manifest schema: {manifest.get('schema')!r}"
        )
    claimed = str(manifest.get("manifest_payload_sha256") or "")
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    actual = canonical_payload_sha256(unsigned)
    if claimed != actual:
        raise NativePipelineValidationError(
            f"build manifest payload hash mismatch: want={claimed} got={actual}"
        )


def validate_build_payload(
    payload_dir: Path,
    *,
    source_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    payload = Path(payload_dir).expanduser()
    source = Path(source_dir).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    if payload.is_symlink() or not payload.is_dir():
        raise NativePipelineValidationError(
            f"native build payload is missing or is a symlink: {payload}"
        )
    payload = payload.resolve()
    manifest_path = payload / SOURCE_BUILD_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    _verify_manifest_payload(manifest, schema=NATIVE_BUILD_MANIFEST_SCHEMA)

    target = manifest.get("target")
    if not isinstance(target, Mapping) or (
        target.get("soabi") != "cpython-312-darwin"
        or target.get("extension_suffix") != NATIVE_EXTENSION_SUFFIX
        or target.get("arch") != "arm64"
        or not str(target.get("python_version") or "").startswith("3.12.")
    ):
        raise NativePipelineValidationError(f"native build target is incompatible: {target!r}")

    native_scope = manifest.get("native_scope")
    records = native_scope.get("modules") if isinstance(native_scope, Mapping) else None
    if not isinstance(records, list) or not all(isinstance(item, Mapping) for item in records):
        raise NativePipelineValidationError("native build module inventory is missing")
    names = tuple(str(item.get("import_name") or "") for item in records)
    if names != NATIVE_MODULES:
        raise NativePipelineValidationError(
            f"native build module inventory mismatch: want={NATIVE_MODULES!r} got={names!r}"
        )

    expected_files = {SOURCE_BUILD_MANIFEST_NAME}
    for record in records:
        name = str(record["import_name"])
        binary_name = f"{name}{NATIVE_EXTENSION_SUFFIX}"
        expected_binary_path = f"pipeline/{binary_name}"
        if record.get("binary_path") != expected_binary_path:
            raise NativePipelineValidationError(
                f"unexpected native build binary path for {name}: {record.get('binary_path')!r}"
            )
        binary = payload / binary_name
        if binary.is_symlink() or not binary.is_file():
            raise NativePipelineValidationError(
                f"native build binary is missing or is a symlink: {binary}"
            )
        if binary.stat().st_size != int(record.get("size_bytes") or -1):
            raise NativePipelineValidationError(f"native build binary size mismatch for {name}")
        if sha256_file(binary) != str(record.get("binary_sha256") or ""):
            raise NativePipelineValidationError(f"native build binary hash mismatch for {name}")
        source_path = source / f"{name}.py"
        if source_path.is_symlink() or not source_path.is_file():
            raise NativePipelineValidationError(f"native source is missing: {source_path}")
        if sha256_file(source_path) != str(record.get("source_set_sha256") or ""):
            raise NativePipelineValidationError(f"native source hash mismatch for {name}")
        expected_files.add(binary_name)

    actual_files = {
        path.name
        for path in payload.iterdir()
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise NativePipelineValidationError(
            "native build payload exact inventory mismatch: "
            f"want={sorted(expected_files)!r} got={sorted(actual_files)!r}"
        )

    source_record = manifest.get("source")
    build_inputs = source_record.get("build_inputs") if isinstance(source_record, Mapping) else None
    if not isinstance(build_inputs, list):
        raise NativePipelineValidationError("native build input inventory is missing")
    for item in build_inputs:
        if not isinstance(item, Mapping):
            raise NativePipelineValidationError("invalid native build input record")
        relative = Path(str(item.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise NativePipelineValidationError(f"unsafe native build input path: {relative}")
        path = project / relative
        if not path.is_file() or sha256_file(path) != str(item.get("sha256") or ""):
            raise NativePipelineValidationError(f"native build input hash mismatch: {relative}")
    return manifest


def embed_build_payload(
    payload_dir: Path,
    *,
    source_dir: Path,
    destination_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    manifest = validate_build_payload(
        payload_dir,
        source_dir=source_dir,
        project_root=project_root,
    )
    payload = Path(payload_dir).expanduser().resolve()
    destination = _absolute_path_without_symlinks(
        Path(destination_dir),
        label="native embed destination",
    )
    destination.mkdir(parents=True, exist_ok=True)
    destination = _absolute_path_without_symlinks(
        destination,
        label="native embed destination",
    )
    if not destination.is_dir():
        raise NativePipelineValidationError(
            f"native embed destination is not a directory: {destination}"
        )
    directory_fd = os.open(
        destination,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    destination_identity = os.fstat(directory_fd)
    records = manifest["native_scope"]["modules"]
    try:
        pycache_fd = -1
        try:
            pycache_stat = os.stat(
                "__pycache__",
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pycache_stat = None
        if pycache_stat is not None:
            if not stat.S_ISDIR(pycache_stat.st_mode):
                raise NativePipelineValidationError(
                    "native embed destination __pycache__ must be a real directory"
                )
            pycache_fd = os.open(
                "__pycache__",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        try:
            destination_names = set(os.listdir(directory_fd))
            for name in NATIVE_MODULES:
                if name in destination_names:
                    raise NativePipelineValidationError(
                        f"matching package fallback remains for {name}"
                    )
                _remove_destination_entry(directory_fd, f"{name}.py")
                _remove_destination_entry(directory_fd, f"{name}.pyc")
                if pycache_fd >= 0:
                    for cached_name in os.listdir(pycache_fd):
                        if cached_name.startswith(f"{name}.") and cached_name.endswith(
                            ".pyc"
                        ):
                            _remove_destination_entry(pycache_fd, cached_name)
                for old_name in tuple(os.listdir(directory_fd)):
                    if old_name.startswith(f"{name}.") and old_name.endswith(".so"):
                        _remove_destination_entry(directory_fd, old_name)
        finally:
            if pycache_fd >= 0:
                os.close(pycache_fd)

        _remove_destination_entry(directory_fd, NATIVE_RUNTIME_MANIFEST_NAME)
        _remove_destination_entry(directory_fd, NATIVE_EMBED_RECEIPT_NAME)
        _secure_copy_to_directory(
            payload / SOURCE_BUILD_MANIFEST_NAME,
            directory_fd,
            NATIVE_BUILD_MANIFEST_NAME,
        )
        for record in records:
            name = str(record["import_name"])
            filename = f"{name}{NATIVE_EXTENSION_SUFFIX}"
            _secure_copy_to_directory(payload / filename, directory_fd, filename)

        current_identity = os.stat(destination, follow_symlinks=False)
        if (
            current_identity.st_dev != destination_identity.st_dev
            or current_identity.st_ino != destination_identity.st_ino
        ):
            raise NativePipelineValidationError(
                "native embed destination identity changed during copy"
            )

        embedded_manifest_path = destination / NATIVE_BUILD_MANIFEST_NAME
        embedded_manifest = _read_json(embedded_manifest_path)
        _verify_manifest_payload(
            embedded_manifest,
            schema=NATIVE_BUILD_MANIFEST_SCHEMA,
        )
        if embedded_manifest != manifest:
            raise NativePipelineValidationError(
                "native build manifest changed between validation and embedding"
            )

        receipt_modules: list[dict[str, Any]] = []
        for record in records:
            name = str(record["import_name"])
            filename = f"{name}{NATIVE_EXTENSION_SUFFIX}"
            copied = destination / filename
            if sha256_file(copied) != str(record["binary_sha256"]):
                raise NativePipelineValidationError(
                    f"embedded native binary hash mismatch: {name}"
                )
            receipt_modules.append(
                {
                    "import_name": name,
                    "binary_filename": filename,
                    "build_binary_sha256": str(record["binary_sha256"]),
                    "unsigned_macho_sha256": unsigned_macho_sha256(copied),
                }
            )

        receipt: dict[str, Any] = {
            "schema": NATIVE_EMBED_RECEIPT_SCHEMA,
            "build_manifest": {
                "filename": NATIVE_BUILD_MANIFEST_NAME,
                "file_sha256": sha256_file(embedded_manifest_path),
                "manifest_payload_sha256": str(
                    manifest.get("manifest_payload_sha256") or ""
                ),
            },
            "modules": receipt_modules,
        }
        receipt["manifest_payload_sha256"] = canonical_payload_sha256(receipt)
        _secure_write_to_directory(
            directory_fd,
            NATIVE_EMBED_RECEIPT_NAME,
            (
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
    finally:
        os.close(directory_fd)

    result = dict(manifest)
    result["seal_receipt"] = str(destination / NATIVE_EMBED_RECEIPT_NAME)
    result["seal_receipt_sha256"] = receipt["manifest_payload_sha256"]
    return result


def _target_python_metadata(python: Path) -> dict[str, str]:
    code = (
        "import json,platform,sys,sysconfig;"
        "print(json.dumps({'python_version':platform.python_version(),"
        "'python_major_minor':f'{sys.version_info.major}.{sys.version_info.minor}',"
        "'arch':platform.machine(),'soabi':sysconfig.get_config_var('SOABI'),"
        "'extension_suffix':sysconfig.get_config_var('EXT_SUFFIX')}))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise NativePipelineValidationError(
            f"target Python metadata probe failed: {(completed.stdout + completed.stderr)[-1000:]}"
        )
    value = json.loads(completed.stdout)
    expected = {
        "python_major_minor": "3.12",
        "arch": "arm64",
        "soabi": "cpython-312-darwin",
        "extension_suffix": NATIVE_EXTENSION_SUFFIX,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise NativePipelineValidationError(f"target Python ABI mismatch: {value!r}")
    return {str(key): str(item) for key, item in value.items()}


def _codesign_metadata(
    binary: Path,
    *,
    signing_mode: str,
    expected_team_identifier: str | None,
) -> dict[str, Any]:
    verified = subprocess.run(
        ["/usr/bin/codesign", "--verify", "--strict", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode != 0:
        raise NativePipelineValidationError(
            f"native binary signature verification failed: {binary}: "
            f"{(verified.stdout + verified.stderr)[-1000:]}"
        )
    described = subprocess.run(
        ["/usr/bin/codesign", "-dvvv", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    if described.returncode != 0:
        raise NativePipelineValidationError(
            f"native binary signature metadata is unavailable: {binary}: "
            f"{(described.stdout + described.stderr)[-1000:]}"
        )
    detail = described.stdout + described.stderr
    fields: dict[str, str] = {}
    code_directory = ""
    for line in detail.splitlines():
        if line.startswith("CodeDirectory "):
            code_directory = line
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields.setdefault(key.strip(), value.strip())
    cdhash = fields.get("CDHash", "").lower()
    team = fields.get("TeamIdentifier", "")
    flags = fields.get("CodeDirectory v", "")
    signature = fields.get("Signature", "")
    if not cdhash:
        raise NativePipelineValidationError(f"native binary CDHash is unavailable: {binary}")
    if signing_mode == "developer_id":
        if not expected_team_identifier:
            raise NativePipelineValidationError(
                "Developer ID seal requires the expected TeamIdentifier"
            )
        if (
            not team
            or team == "not set"
            or team != expected_team_identifier
            or "runtime" not in code_directory
            or signature == "adhoc"
        ):
            raise NativePipelineValidationError(
                "Developer ID native signature does not match the expected "
                f"Team/runtime seal: {binary}"
            )
    else:
        if expected_team_identifier:
            raise NativePipelineValidationError(
                "ad-hoc seal must not declare an expected TeamIdentifier"
            )
        if (team and team != "not set") or signature != "adhoc":
            raise NativePipelineValidationError(
                f"native binary is not ad-hoc signed as declared: {binary}"
            )
    return {
        "verified": True,
        "cdhash": cdhash,
        "team_identifier": None if team in {"", "not set"} else team,
        "hardened_runtime": "runtime" in code_directory,
        "mode": signing_mode,
        "flags_detail": flags or None,
    }


def seal_runtime_payload(
    pipeline_dir: Path,
    *,
    target_python: Path,
    app_bundle_id: str,
    app_version: str,
    signing_mode: str,
    expected_receipt_sha256: str,
    expected_team_identifier: str | None = None,
) -> dict[str, Any]:
    root_input = Path(pipeline_dir).expanduser()
    if root_input.is_symlink() or not root_input.is_dir():
        raise NativePipelineValidationError(
            f"native pipeline seal root is missing or is a symlink: {root_input}"
        )
    root = root_input.resolve()
    if signing_mode not in {"ad_hoc", "developer_id"}:
        raise NativePipelineValidationError(f"unsupported signing mode: {signing_mode}")
    if signing_mode == "developer_id" and not expected_team_identifier:
        raise NativePipelineValidationError(
            "Developer ID seal requires --expected-team-identifier"
        )
    if signing_mode == "ad_hoc" and expected_team_identifier:
        raise NativePipelineValidationError(
            "ad-hoc seal must not declare an expected TeamIdentifier"
        )
    receipt_path = root / NATIVE_EMBED_RECEIPT_NAME
    receipt = _read_json(receipt_path)
    _verify_manifest_payload(receipt, schema=NATIVE_EMBED_RECEIPT_SCHEMA)
    if str(receipt.get("manifest_payload_sha256") or "") != expected_receipt_sha256:
        raise NativePipelineValidationError(
            "native embed receipt identity changed before sealing"
        )
    build_manifest_path = root / NATIVE_BUILD_MANIFEST_NAME
    build_manifest = _read_json(build_manifest_path)
    _verify_manifest_payload(build_manifest, schema=NATIVE_BUILD_MANIFEST_SCHEMA)
    receipt_build = receipt.get("build_manifest")
    if not isinstance(receipt_build, Mapping):
        raise NativePipelineValidationError("native embed receipt lacks build manifest identity")
    if (
        receipt_build.get("filename") != NATIVE_BUILD_MANIFEST_NAME
        or receipt_build.get("file_sha256") != sha256_file(build_manifest_path)
        or receipt_build.get("manifest_payload_sha256")
        != build_manifest.get("manifest_payload_sha256")
    ):
        raise NativePipelineValidationError(
            "embedded build manifest no longer matches the validated embed receipt"
        )
    build_records = build_manifest["native_scope"]["modules"]
    if tuple(str(item.get("import_name") or "") for item in build_records) != NATIVE_MODULES:
        raise NativePipelineValidationError("embedded build manifest module inventory mismatch")
    receipt_records = receipt.get("modules")
    if not isinstance(receipt_records, list) or not all(
        isinstance(item, Mapping) for item in receipt_records
    ):
        raise NativePipelineValidationError("native embed receipt module inventory is missing")
    if tuple(str(item.get("import_name") or "") for item in receipt_records) != NATIVE_MODULES:
        raise NativePipelineValidationError("native embed receipt module inventory mismatch")
    target = _target_python_metadata(Path(target_python))

    module_records = []
    for build_record, receipt_record in zip(build_records, receipt_records, strict=True):
        name = str(build_record["import_name"])
        filename = f"{name}{NATIVE_EXTENSION_SUFFIX}"
        if (
            receipt_record.get("binary_filename") != filename
            or receipt_record.get("build_binary_sha256")
            != build_record.get("binary_sha256")
        ):
            raise NativePipelineValidationError(
                f"native embed receipt provenance mismatch for {name}"
            )
        binary = root / filename
        if binary.is_symlink() or not binary.is_file():
            raise NativePipelineValidationError(f"embedded native binary is missing: {binary}")
        if (root / f"{name}.py").exists() or (root / f"{name}.pyc").exists():
            raise NativePipelineValidationError(f"matching source fallback remains for {name}")
        unsigned_sha = unsigned_macho_sha256(binary)
        if unsigned_sha != str(receipt_record.get("unsigned_macho_sha256") or ""):
            raise NativePipelineValidationError(
                f"signed native binary no longer descends from embedded build bytes: {name}"
            )
        module_records.append(
            {
                "import_name": name,
                "binary_filename": filename,
                "binary_sha256": sha256_file(binary),
                "size_bytes": binary.stat().st_size,
                "unsigned_macho_sha256": unsigned_sha,
                "source_set_sha256": build_record["source_set_sha256"],
                "codesign": _codesign_metadata(
                    binary,
                    signing_mode=signing_mode,
                    expected_team_identifier=expected_team_identifier,
                ),
            }
        )

    source_record = build_manifest.get("source")
    source_record = source_record if isinstance(source_record, Mapping) else {}
    blocking = ["app_notarization_missing", "dmg_notarization_missing"]
    if signing_mode == "ad_hoc":
        blocking.insert(0, "developer_id_signature_missing")
    blocking.extend(
        str(item)
        for item in build_manifest.get("blocking_reasons") or ()
        if str(item) not in _APP_EMBEDDING_RESOLVED_BLOCKERS
    )
    manifest: dict[str, Any] = {
        "schema": NATIVE_RUNTIME_MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "distribution_scope": (
            "internal_only" if signing_mode == "ad_hoc" else "developer_id_candidate"
        ),
        "app": {
            "bundle_id": app_bundle_id,
            "version": app_version,
        },
        "target": target,
        "source_build_manifest": {
            "filename": NATIVE_BUILD_MANIFEST_NAME,
            "sha256": sha256_file(build_manifest_path),
            "commit_sha": source_record.get("commit_sha"),
            "tag": source_record.get("tag"),
            "clean": source_record.get("clean"),
            "archive_sha256": source_record.get("archive_sha256"),
        },
        "modules": module_records,
        "signing": {
            "mode": signing_mode,
            "team_identifier": expected_team_identifier,
            "nested_modules_verified": True,
            "seal_order": "nested_modules_then_manifest_then_outer_app",
            "outer_app_signature_recorded_in_manifest": False,
            "notarized": False,
        },
        "release_eligible": False,
        "blocking_reasons": list(dict.fromkeys(blocking)),
    }
    manifest["manifest_payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    directory_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        directory_identity = os.fstat(directory_fd)
        current_identity = os.stat(root, follow_symlinks=False)
        if (
            current_identity.st_dev != directory_identity.st_dev
            or current_identity.st_ino != directory_identity.st_ino
        ):
            raise NativePipelineValidationError(
                "native pipeline seal root identity changed before manifest write"
            )

        receipt_fd = os.open(
            NATIVE_EMBED_RECEIPT_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            if not stat.S_ISREG(os.fstat(receipt_fd).st_mode):
                raise NativePipelineValidationError(
                    "native embed receipt is no longer a regular file"
                )
            with os.fdopen(os.dup(receipt_fd), "r", encoding="utf-8") as handle:
                current_receipt = json.load(handle)
        finally:
            os.close(receipt_fd)
        if not isinstance(current_receipt, dict):
            raise NativePipelineValidationError(
                "native embed receipt root must remain an object"
            )
        _verify_manifest_payload(
            current_receipt,
            schema=NATIVE_EMBED_RECEIPT_SCHEMA,
        )
        if (
            current_receipt != receipt
            or current_receipt.get("manifest_payload_sha256")
            != expected_receipt_sha256
        ):
            raise NativePipelineValidationError(
                "native embed receipt changed before seal completion"
            )

        _remove_destination_entry(directory_fd, NATIVE_RUNTIME_MANIFEST_NAME)
        os.unlink(NATIVE_EMBED_RECEIPT_NAME, dir_fd=directory_fd)
        _secure_write_to_directory(
            directory_fd,
            NATIVE_RUNTIME_MANIFEST_NAME,
            manifest_bytes,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativePipelineValidationError(
            f"unable to publish native runtime seal safely: {error}"
        ) from error
    finally:
        os.close(directory_fd)

    current_identity = os.stat(root, follow_symlinks=False)
    if (
        current_identity.st_dev != directory_identity.st_dev
        or current_identity.st_ino != directory_identity.st_ino
    ):
        raise NativePipelineValidationError(
            "native pipeline seal root identity changed after manifest write"
        )
    return validate_native_runtime_payload(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed")
    embed.add_argument("--payload-dir", type=Path, required=True)
    embed.add_argument("--source-dir", type=Path, required=True)
    embed.add_argument("--destination-dir", type=Path, required=True)
    embed.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--pipeline-dir", type=Path, required=True)
    seal.add_argument("--target-python", type=Path, required=True)
    seal.add_argument("--app-bundle-id", required=True)
    seal.add_argument("--app-version", required=True)
    seal.add_argument("--signing-mode", choices=("ad_hoc", "developer_id"), required=True)
    seal.add_argument("--expected-receipt-sha256", required=True)
    seal.add_argument("--expected-team-identifier")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--pipeline-dir", type=Path, required=True)
    verify.add_argument("--target-python", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "embed":
        value = embed_build_payload(
            args.payload_dir,
            source_dir=args.source_dir,
            destination_dir=args.destination_dir,
            project_root=args.project_root,
        )
        print(
            "native_build_payload_embedded="
            + str(value.get("manifest_payload_sha256") or "")
        )
        print("native_embed_receipt=" + str(value.get("seal_receipt_sha256") or ""))
        return 0
    if args.command == "seal":
        value = seal_runtime_payload(
            args.pipeline_dir,
            target_python=args.target_python,
            app_bundle_id=args.app_bundle_id,
            app_version=args.app_version,
            signing_mode=args.signing_mode,
            expected_receipt_sha256=args.expected_receipt_sha256,
            expected_team_identifier=args.expected_team_identifier,
        )
        print(
            "native_runtime_payload_sealed="
            + str(value.get("manifest_payload_sha256") or "")
        )
        return 0
    value = validate_native_runtime_payload(args.pipeline_dir)
    if args.target_python is not None:
        result = probe_native_imports(args.target_python, args.pipeline_dir)
        print("native_runtime_imports=" + ",".join(sorted(result["modules"])))
    print(
        "native_runtime_payload_verified="
        + str(value.get("manifest_payload_sha256") or "")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NativePipelineValidationError as error:
        print(f"[NATIVE][ERROR] {error}", file=sys.stderr)
        raise SystemExit(2) from error
