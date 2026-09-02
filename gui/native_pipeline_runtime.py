#!/usr/bin/env python3
"""Fail-closed runtime contract for Starun CPython 3.12 native modules."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


NATIVE_BUILD_MANIFEST_NAME = "native-pipeline-build-manifest.json"
NATIVE_RUNTIME_MANIFEST_NAME = "native-pipeline-runtime-manifest.json"
NATIVE_BUILD_MANIFEST_SCHEMA = "starun.native-pipeline-build.v1"
NATIVE_RUNTIME_MANIFEST_SCHEMA = "starun.native-pipeline-runtime.v1"
NATIVE_EXTENSION_SUFFIX = ".cpython-312-darwin.so"
NATIVE_MODULES = (
    "stage3_contract",
    "background_sampling",
    "stage4_auto_reference",
    "local_adjustments",
    "stage9_quality",
)


class NativePipelineValidationError(RuntimeError):
    """Raised when a declared native runtime payload is incomplete or altered."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _codesign_metadata(binary: Path) -> dict[str, Any]:
    """Return verified signature facts used by the sealed runtime manifest."""

    path = Path(binary)
    verified = subprocess.run(
        ["/usr/bin/codesign", "--verify", "--strict", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode != 0:
        detail = (verified.stdout + verified.stderr).strip()[-1000:]
        raise NativePipelineValidationError(
            f"native binary signature verification failed: {path}: {detail}"
        )
    described = subprocess.run(
        ["/usr/bin/codesign", "-dvvv", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if described.returncode != 0:
        detail = (described.stdout + described.stderr).strip()[-1000:]
        raise NativePipelineValidationError(
            f"native binary signature metadata is unavailable: {path}: {detail}"
        )
    detail = described.stdout + described.stderr
    fields: dict[str, str] = {}
    code_directory = ""
    for line in detail.splitlines():
        if line.startswith("CodeDirectory "):
            code_directory = line
        if "=" in line:
            key, value = line.split("=", 1)
            fields.setdefault(key.strip(), value.strip())
    cdhash = fields.get("CDHash", "").lower()
    if not re.fullmatch(r"[0-9a-f]{40,64}", cdhash):
        raise NativePipelineValidationError(
            f"native binary CDHash is unavailable or invalid: {path}"
        )
    team = fields.get("TeamIdentifier", "")
    signature = fields.get("Signature", "")
    return {
        "verified": True,
        "cdhash": cdhash,
        "team_identifier": None if team in {"", "not set"} else team,
        "hardened_runtime": "runtime" in code_directory,
        "mode": "ad_hoc" if signature == "adhoc" else "developer_id",
    }


def unsigned_macho_sha256(binary: Path) -> str:
    """Hash Mach-O bytes after removing only the replaceable code signature."""

    path = Path(binary)
    if path.is_symlink() or not path.is_file():
        raise NativePipelineValidationError(
            f"native binary is missing or is a symlink: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(path, flags)
    except OSError as error:
        raise NativePipelineValidationError(
            f"unable to open native binary without following links: {path}: {error}"
        ) from error
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise NativePipelineValidationError(
                f"native binary is not a regular file: {path}"
            )
        with tempfile.TemporaryDirectory(prefix="starun_native_unsigned_") as td:
            unsigned = Path(td) / path.name
            with os.fdopen(os.dup(source_fd), "rb") as source_handle, unsigned.open(
                "xb"
            ) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
            completed = subprocess.run(
                ["/usr/bin/codesign", "--remove-signature", str(unsigned)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = (completed.stdout + completed.stderr).strip()[-1000:]
                raise NativePipelineValidationError(
                    f"unable to remove native binary signature: {path}: {detail}"
                )
            return sha256_file(unsigned)
    finally:
        os.close(source_fd)


def _load_hashed_manifest(path: Path, *, schema: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise NativePipelineValidationError(
            f"native manifest is missing or is a symlink: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativePipelineValidationError(
            f"native manifest is unreadable: {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise NativePipelineValidationError("native manifest root must be an object")
    if payload.get("schema") != schema:
        raise NativePipelineValidationError(
            f"unexpected native manifest schema: {payload.get('schema')!r}"
        )
    claimed = str(payload.get("manifest_payload_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("manifest_payload_sha256", None)
    actual = canonical_payload_sha256(unsigned)
    if claimed != actual:
        raise NativePipelineValidationError(
            f"native manifest payload hash mismatch: want={claimed} got={actual}"
        )
    return payload


def _validate_module_names(records: object) -> list[Mapping[str, Any]]:
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise NativePipelineValidationError("native module inventory must be a list")
    names = tuple(str(record.get("import_name") or "") for record in records)
    if names != NATIVE_MODULES:
        raise NativePipelineValidationError(
            f"native module inventory mismatch: want={NATIVE_MODULES!r} got={names!r}"
        )
    return records


def _fallback_paths(root: Path, module: str) -> tuple[Path, ...]:
    paths = [
        root / module,
        root / f"{module}.py",
        root / f"{module}.pyc",
    ]
    pycache = root / "__pycache__"
    if pycache.is_dir() and not pycache.is_symlink():
        paths.extend(sorted(pycache.glob(f"{module}.*.pyc")))
    elif pycache.is_symlink():
        paths.append(pycache)
    return tuple(paths)


def validate_native_runtime_payload(
    pipeline_root: Path,
    *,
    expected_manifest_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate signed-byte hashes and the no-source-fallback runtime policy."""

    root = Path(pipeline_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise NativePipelineValidationError(
            f"native pipeline root is missing or is a symlink: {root}"
        )
    root = root.resolve()
    manifest = _load_hashed_manifest(
        root / NATIVE_RUNTIME_MANIFEST_NAME,
        schema=NATIVE_RUNTIME_MANIFEST_SCHEMA,
    )
    manifest_payload_sha256 = str(manifest.get("manifest_payload_sha256") or "")
    if (
        expected_manifest_payload_sha256 is not None
        and manifest_payload_sha256 != expected_manifest_payload_sha256
    ):
        raise NativePipelineValidationError(
            "native runtime manifest identity changed: "
            f"want={expected_manifest_payload_sha256} got={manifest_payload_sha256}"
        )
    signing = manifest.get("signing")
    if not isinstance(signing, Mapping):
        raise NativePipelineValidationError("native signing metadata is missing")
    signing_mode = str(signing.get("mode") or "")
    if signing_mode not in {"ad_hoc", "developer_id"}:
        raise NativePipelineValidationError(
            f"unsupported native signing mode: {signing_mode!r}"
        )
    expected_team = signing.get("team_identifier")
    if expected_team is not None:
        expected_team = str(expected_team)
    if signing_mode == "developer_id" and not expected_team:
        raise NativePipelineValidationError(
            "Developer ID runtime manifest lacks a TeamIdentifier"
        )
    if signing_mode == "ad_hoc" and expected_team:
        raise NativePipelineValidationError(
            "ad-hoc runtime manifest unexpectedly declares a TeamIdentifier"
        )
    target = manifest.get("target")
    if not isinstance(target, Mapping):
        raise NativePipelineValidationError("native target metadata is missing")
    expected_target = {
        "python_major_minor": "3.12",
        "soabi": "cpython-312-darwin",
        "arch": "arm64",
        "extension_suffix": NATIVE_EXTENSION_SUFFIX,
    }
    for key, expected in expected_target.items():
        if str(target.get(key) or "") != expected:
            raise NativePipelineValidationError(
                f"native target {key} mismatch: want={expected!r} "
                f"got={target.get(key)!r}"
            )

    build_record = manifest.get("source_build_manifest")
    if not isinstance(build_record, Mapping):
        raise NativePipelineValidationError("source build manifest record is missing")
    if build_record.get("filename") != NATIVE_BUILD_MANIFEST_NAME:
        raise NativePipelineValidationError("unexpected source build manifest filename")
    build_manifest = root / NATIVE_BUILD_MANIFEST_NAME
    if build_manifest.is_symlink() or not build_manifest.is_file():
        raise NativePipelineValidationError(
            f"source build manifest is missing or is a symlink: {build_manifest}"
        )
    expected_build_sha = str(build_record.get("sha256") or "")
    if sha256_file(build_manifest) != expected_build_sha:
        raise NativePipelineValidationError("source build manifest hash mismatch")

    records = _validate_module_names(manifest.get("modules"))
    expected_binaries: set[str] = set()
    for record in records:
        name = str(record["import_name"])
        filename = str(record.get("binary_filename") or "")
        expected_filename = f"{name}{NATIVE_EXTENSION_SUFFIX}"
        if filename != expected_filename or Path(filename).name != filename:
            raise NativePipelineValidationError(
                f"unsafe native binary filename for {name}: {filename!r}"
            )
        expected_binaries.add(filename)
        binary = root / filename
        if binary.is_symlink() or not binary.is_file():
            raise NativePipelineValidationError(
                f"native binary is missing or is a symlink: {binary}"
            )
        if binary.resolve().parent != root:
            raise NativePipelineValidationError(
                f"native binary escapes pipeline root: {binary}"
            )
        try:
            expected_size = int(record.get("size_bytes"))
        except (TypeError, ValueError) as error:
            raise NativePipelineValidationError(
                f"invalid native binary size for {name}"
            ) from error
        if binary.stat().st_size != expected_size:
            raise NativePipelineValidationError(
                f"native binary size mismatch for {name}"
            )
        if sha256_file(binary) != str(record.get("binary_sha256") or ""):
            raise NativePipelineValidationError(
                f"native binary hash mismatch for {name}"
            )
        codesign = record.get("codesign")
        if not isinstance(codesign, Mapping) or codesign.get("verified") is not True:
            raise NativePipelineValidationError(
                f"native binary signing record is missing for {name}"
            )
        actual_codesign = _codesign_metadata(binary)
        expected_codesign = {
            "cdhash": str(codesign.get("cdhash") or "").lower(),
            "team_identifier": codesign.get("team_identifier"),
            "hardened_runtime": codesign.get("hardened_runtime"),
            "mode": signing_mode,
        }
        for key, expected in expected_codesign.items():
            if actual_codesign.get(key) != expected:
                raise NativePipelineValidationError(
                    f"native binary signing {key} mismatch for {name}: "
                    f"want={expected!r} got={actual_codesign.get(key)!r}"
                )
        if actual_codesign["team_identifier"] != expected_team:
            raise NativePipelineValidationError(
                f"native binary TeamIdentifier does not match runtime seal for {name}"
            )
        expected_unsigned_sha = str(record.get("unsigned_macho_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_unsigned_sha):
            raise NativePipelineValidationError(
                f"native unsigned Mach-O hash is missing for {name}"
            )
        if unsigned_macho_sha256(binary) != expected_unsigned_sha:
            raise NativePipelineValidationError(
                f"native unsigned Mach-O lineage mismatch for {name}"
            )
        fallback = next(
            (
                path
                for path in _fallback_paths(root, name)
                if path.exists() or path.is_symlink()
            ),
            None,
        )
        if fallback is not None:
            raise NativePipelineValidationError(
                f"matching source/bytecode fallback is forbidden: {fallback}"
            )

    actual_binaries = {
        path.name
        for path in root.glob("*.so")
        if path.is_file() or path.is_symlink()
    }
    if actual_binaries != expected_binaries:
        raise NativePipelineValidationError(
            "native binary exact inventory mismatch: "
            f"want={sorted(expected_binaries)!r} got={sorted(actual_binaries)!r}"
        )
    return manifest


def inspect_native_pipeline(
    pipeline_root: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    """Return capability evidence for a formal App or development overlay."""

    root = Path(pipeline_root).expanduser()
    runtime_manifest = root / NATIVE_RUNTIME_MANIFEST_NAME
    if runtime_manifest.exists() or runtime_manifest.is_symlink():
        try:
            manifest = validate_native_runtime_payload(root)
        except NativePipelineValidationError as error:
            return {
                "status": "unavailable",
                "available": False,
                "required": bool(required),
                "mode": "native_invalid",
                "manifest": str(runtime_manifest),
                "modules": list(NATIVE_MODULES),
                "import_probe": {"status": "not_run", "available": None},
                "error": str(error),
            }
        return {
            "status": "available",
            "available": True,
            "required": bool(required),
            "mode": "native",
            "manifest": str(runtime_manifest),
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "modules": list(NATIVE_MODULES),
            "import_probe": {"status": "pending", "available": None},
            "error": None,
        }

    source_paths = [root / f"{name}.py" for name in NATIVE_MODULES]
    missing = [
        path.name
        for path in source_paths
        if path.is_symlink() or not path.is_file()
    ]
    available = not required and not missing
    return {
        "status": "source_mode" if available else "unavailable",
        "available": available,
        "required": bool(required),
        "mode": "source" if available else "native_missing",
        "manifest": str(runtime_manifest),
        "modules": list(NATIVE_MODULES),
        "missing_source_modules": missing,
        "import_probe": {"status": "not_applicable", "available": available},
        "error": (
            None
            if available
            else "formal App native runtime manifest is missing"
            if required
            else "development source modules are incomplete"
        ),
    }


def stage_native_runtime_payload(source_root: Path, destination_root: Path) -> dict[str, Any]:
    """Copy one already-validated payload to a per-run pyscript directory."""

    source = Path(source_root).expanduser()
    destination = Path(destination_root).expanduser()
    manifest = validate_native_runtime_payload(source)
    expected_manifest_hash = str(manifest["manifest_payload_sha256"])
    source = source.resolve()
    if destination.is_symlink():
        raise NativePipelineValidationError(
            f"native runtime destination must not be a symlink: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    filenames = [
        NATIVE_BUILD_MANIFEST_NAME,
        NATIVE_RUNTIME_MANIFEST_NAME,
        *[str(record["binary_filename"]) for record in manifest["modules"]],
    ]
    for filename in filenames:
        target = destination / filename
        if target.is_symlink():
            raise NativePipelineValidationError(
                f"refusing to replace a runtime symlink: {target}"
            )
        shutil.copy2(source / filename, target)
    return validate_native_runtime_payload(
        destination,
        expected_manifest_payload_sha256=expected_manifest_hash,
    )


def probe_native_imports(
    python_executable: Path,
    pipeline_root: Path,
    *,
    timeout_seconds: int = 60,
    expected_manifest_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Import every module with the actual isolated CPython 3.12 runtime."""

    root = Path(pipeline_root).expanduser()
    validate_native_runtime_payload(
        root,
        expected_manifest_payload_sha256=expected_manifest_payload_sha256,
    )
    root = root.resolve()
    python = Path(python_executable).expanduser()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise NativePipelineValidationError(
            f"native import probe Python is not executable: {python}"
        )
    probe_code = r'''
import importlib
import json
import platform
import sys
import sysconfig
from pathlib import Path

root = Path(sys.argv[1]).resolve()
modules = tuple(sys.argv[2:])
sys.path.insert(0, str(root))
loaded = {}
for name in modules:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if path.parent != root or path.suffix != ".so":
        raise RuntimeError(f"{name} did not load from native root: {path}")
    loaded[name] = path.name
print(json.dumps({
    "python_version": platform.python_version(),
    "arch": platform.machine(),
    "soabi": sysconfig.get_config_var("SOABI"),
    "modules": loaded,
}, sort_keys=True))
'''
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON") and not key.startswith("DYLD_")
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [str(python), "-I", "-B", "-c", probe_code, str(root), *NATIVE_MODULES],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise NativePipelineValidationError("native import probe timed out") from error
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()[-2000:]
        raise NativePipelineValidationError(
            f"native import probe failed ({completed.returncode}): {detail}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise NativePipelineValidationError(
            f"native import probe returned invalid JSON: {completed.stdout[-500:]}"
        ) from error
    if (
        not str(result.get("python_version") or "").startswith("3.12.")
        or result.get("arch") != "arm64"
        or result.get("soabi") != "cpython-312-darwin"
        or set((result.get("modules") or {}).keys()) != set(NATIVE_MODULES)
    ):
        raise NativePipelineValidationError(
            f"native import probe runtime mismatch: {result!r}"
        )
    return result
