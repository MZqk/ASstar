"""Durable processing-plan and result-manifest helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


_SENSITIVE_KEY_TOKENS = ("api_key", "token", "secret", "password", "credential")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(
    path: Path,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("SHA-256 calculation cancelled")
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except InterruptedError:
        raise
    except OSError:
        return None
    return digest.hexdigest()


def file_record(path: Path, *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    try:
        display_path = path.relative_to(base_dir) if base_dir else path
    except ValueError:
        display_path = path
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {
        "path": display_path.as_posix(),
        "size": int(size),
        "sha256": sha256_file(path),
    }


def collect_output_records(
    work_dir: Path,
    *,
    output_basenames: Iterable[str] = (),
    exported_after: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect durable outputs from this run without admitting stale artifacts."""
    basenames = {
        str(name).strip().lower()
        for name in output_basenames
        if str(name).strip()
    }
    durable_names = {
        "processing-plan.json",
        "result_linear.fit",
        "seestar_diagnostics.zip",
    }
    outputs: Dict[str, Dict[str, Any]] = {}
    for path in sorted(work_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name in durable_names:
            outputs[path.name] = file_record(path, base_dir=work_dir)
            continue

        if exported_after is None:
            selected = (
                lower_name.startswith("result_")
                or lower_name.startswith("seestar_diagnostics")
                or lower_name.startswith("processing-plan")
            )
        else:
            try:
                current_run_file = path.stat().st_mtime >= float(exported_after) - 1.0
            except (OSError, TypeError, ValueError):
                current_run_file = False
            selected = current_run_file and (
                lower_name.startswith("result_")
                or any(
                    lower_name.startswith(f"{basename}.")
                    or lower_name.startswith(f"{basename}_")
                    for basename in basenames
                )
            )
        if selected:
            outputs[path.name] = file_record(path, base_dir=work_dir)
    return outputs


def redact_sensitive(value: Any, *, key: str = "") -> Any:
    normalized_key = str(key).lower()
    if any(token in normalized_key for token in _SENSITIVE_KEY_TOKENS):
        return "<redacted>" if value else ""
    if isinstance(value, Mapping):
        return {
            str(child_key): redact_sensitive(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def verify_resume_provenance(
    *,
    work_dir: Path,
    input_path: Path,
    checkpoint_name: str,
) -> Dict[str, Any]:
    """Verify a resume checkpoint against the previous durable result manifest."""
    manifest_path = work_dir / "pipeline-result.json"
    manifest = load_json(manifest_path)
    actual_hash = sha256_file(input_path)
    result: Dict[str, Any] = {
        "verified": False,
        "state": "unknown",
        "checkpoint": checkpoint_name,
        "input_path": str(input_path),
        "manifest_path": str(manifest_path),
        "actual_sha256": actual_hash,
    }
    if manifest is None:
        result["detail"] = "pipeline-result.json is missing or invalid"
        return result
    if str(manifest.get("schema") or "") != "seestar.pipeline-result.v1":
        result["detail"] = "unsupported pipeline-result schema"
        return result
    expected_manifest_hash = str(manifest.get("manifest_hash") or "")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_hash", None)
    actual_manifest_hash = canonical_payload_hash(unsigned_manifest)
    result["manifest_hash"] = expected_manifest_hash or None
    if not expected_manifest_hash or actual_manifest_hash != expected_manifest_hash:
        result["detail"] = "pipeline-result manifest hash is missing or invalid"
        return result
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        result["detail"] = "manifest has no checkpoint provenance"
        return result
    expected = checkpoints.get(checkpoint_name)
    if not isinstance(expected, Mapping):
        result["detail"] = f"manifest has no {checkpoint_name} checkpoint"
        return result
    expected_hash = str(expected.get("sha256") or "")
    expected_state = str(expected.get("state") or "unknown").strip().lower()
    expected_path = Path(str(expected.get("path") or "")).name
    result["expected_sha256"] = expected_hash or None
    result["state"] = expected_state
    if expected_path and expected_path != input_path.name:
        result["detail"] = "checkpoint filename does not match manifest"
        return result
    if not actual_hash or not expected_hash:
        result["detail"] = "checkpoint hash is unavailable"
        return result
    if actual_hash != expected_hash:
        result["detail"] = "checkpoint SHA-256 does not match manifest"
        return result
    if expected_state != "linear":
        result["detail"] = f"manifest checkpoint state is {expected_state}, not linear"
        return result
    result["verified"] = True
    result["detail"] = "checkpoint SHA-256 and linear state match pipeline-result.json"
    result["plan_hash"] = manifest.get("plan_hash")
    return result
