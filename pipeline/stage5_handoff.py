"""Strict runtime lineage contract for the Stage 5 -> Stage 6 artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    from . import run_manifest
except ImportError:
    import run_manifest


STAGE5_HANDOFF_SCHEMA = "starun.stage5-stage6-handoff.v1"
STAGE5_INPUT_LINEAGE_SCHEMA = "starun.stage5-input-lineage.v1"
STAGE5_UPSTREAM_STEM = "stage4_color"
STAGE5_UPSTREAM_ARTIFACT = "stage4_color.fit"
STAGE5_INPUT_STEM = "stage5_input_linear"
STAGE5_INPUT_ARTIFACT = "stage5_input_linear.fit"
STAGE5_SOURCE_STEM = "stage5_linear"
STAGE5_SOURCE_ARTIFACT = "stage5_linear.fit"
STAGE5_HANDOFF_REPORT = "stage5_stage6_handoff.json"
CURRENT_RUN_ORIGIN = "current_run_stage5"
VERIFIED_RESUME_ORIGIN = "verified_stage5_resume"

REASON_SOURCE_UNAVAILABLE = "stage6_stage5_linear_unavailable"
REASON_LINEAGE_UNVERIFIED = "stage6_stage5_lineage_unverified"
REASON_INPUT_CHECKPOINT_FAILED = "stage6_input_checkpoint_failed"


class Stage5HandoffError(RuntimeError):
    """Raised when Stage 6 cannot prove its canonical Stage 5 lineage."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = str(reason_code)
        self.detail = str(detail)


def _run_id(pipeline: object) -> str:
    direct = str(getattr(pipeline, "_run_id", "") or "").strip()
    if direct:
        return direct
    manifest = getattr(pipeline, "_task_run_manifest_payload", None)
    if isinstance(manifest, Mapping):
        return str(manifest.get("run_id") or "").strip()
    return ""


def _artifact_path(pipeline: object) -> Optional[Path]:
    return _process_artifact_path(pipeline, STAGE5_SOURCE_ARTIFACT)


def _process_artifact_path(
    pipeline: object,
    artifact: str,
) -> Optional[Path]:
    process_dir = getattr(pipeline, "process_dir", None)
    if process_dir is None:
        return None
    return Path(process_dir) / artifact


def freeze_stage5_input_lineage(
    pipeline: object,
    *,
    upstream_loaded: bool,
    baseline_saved: bool,
) -> Dict[str, Any]:
    """Freeze the canonical Stage 4 input and saved Stage 5 baseline bytes."""

    upstream_path = _process_artifact_path(pipeline, STAGE5_UPSTREAM_ARTIFACT)
    baseline_path = _process_artifact_path(pipeline, STAGE5_INPUT_ARTIFACT)
    upstream_sha256 = (
        run_manifest.sha256_file(upstream_path)
        if upstream_path is not None and upstream_loaded
        else None
    )
    baseline_sha256 = (
        run_manifest.sha256_file(baseline_path)
        if baseline_path is not None and baseline_saved
        else None
    )
    load_verified = bool(upstream_loaded and upstream_sha256)
    save_verified = bool(baseline_saved and baseline_sha256)
    accepted = bool(load_verified and save_verified)
    if not upstream_loaded:
        detail = "canonical stage4_color load was not successful"
    elif not upstream_sha256:
        detail = "canonical stage4_color.fit is unavailable or cannot be hashed"
    elif not baseline_saved:
        detail = "stage5_input_linear save was not successful"
    elif not baseline_sha256:
        detail = "stage5_input_linear.fit is unavailable or cannot be hashed"
    else:
        detail = "canonical Stage 5 input baseline frozen"
    record: Dict[str, Any] = {
        "schema": STAGE5_INPUT_LINEAGE_SCHEMA,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "mode": "current_run_baseline",
        "upstream": {
            "source_stage": 4,
            "source_stem": STAGE5_UPSTREAM_STEM,
            "artifact": STAGE5_UPSTREAM_ARTIFACT,
            "load_verified": load_verified,
            "sha256": upstream_sha256,
        },
        "baseline": {
            "source_stage": 5,
            "source_stem": STAGE5_INPUT_STEM,
            "artifact": STAGE5_INPUT_ARTIFACT,
            "save_verified": save_verified,
            "sha256": baseline_sha256,
        },
        "reason_code": "accepted" if accepted else REASON_LINEAGE_UNVERIFIED,
        "detail": detail,
    }
    setattr(pipeline, "_stage5_input_lineage", record)
    return record


def _current_input_lineage_error(
    pipeline: object,
    lineage: Any,
) -> Optional[str]:
    if not isinstance(lineage, Mapping):
        return "Stage 5 input lineage is missing"
    upstream = lineage.get("upstream")
    baseline = lineage.get("baseline")
    if (
        str(lineage.get("schema") or "") != STAGE5_INPUT_LINEAGE_SCHEMA
        or str(lineage.get("status") or "") != "accepted"
        or lineage.get("accepted") is not True
        or str(lineage.get("mode") or "") != "current_run_baseline"
        or not isinstance(upstream, Mapping)
        or not isinstance(baseline, Mapping)
        or upstream.get("source_stage") != 4
        or str(upstream.get("source_stem") or "") != STAGE5_UPSTREAM_STEM
        or str(upstream.get("artifact") or "") != STAGE5_UPSTREAM_ARTIFACT
        or upstream.get("load_verified") is not True
        or baseline.get("source_stage") != 5
        or str(baseline.get("source_stem") or "") != STAGE5_INPUT_STEM
        or str(baseline.get("artifact") or "") != STAGE5_INPUT_ARTIFACT
        or baseline.get("save_verified") is not True
    ):
        return "Stage 5 input lineage contract is not accepted"

    for label, artifact, expected in (
        (
            "canonical stage4_color.fit",
            STAGE5_UPSTREAM_ARTIFACT,
            upstream.get("sha256"),
        ),
        (
            "stage5_input_linear.fit",
            STAGE5_INPUT_ARTIFACT,
            baseline.get("sha256"),
        ),
    ):
        path = _process_artifact_path(pipeline, artifact)
        actual_sha256 = (
            run_manifest.sha256_file(path) if path is not None else None
        )
        expected_sha256 = str(expected or "").strip()
        if (
            not actual_sha256
            or not expected_sha256
            or actual_sha256 != expected_sha256
        ):
            return f"{label} changed or disappeared after input freeze"
    return None


def _resume_input_lineage() -> Dict[str, Any]:
    return {
        "schema": STAGE5_INPUT_LINEAGE_SCHEMA,
        "status": "verified_resume",
        "accepted": True,
        "mode": "formal_stage5_checkpoint",
        "upstream": None,
        "baseline": None,
        "reason_code": "verified_stage5_resume",
        "detail": "input lineage inherited from verified formal Stage 5 checkpoint",
    }


def public_handoff(record: Any) -> Dict[str, Any]:
    """Return the serializable, path-free audit view of a handoff."""

    if not isinstance(record, Mapping):
        return {}
    return {
        str(key): value
        for key, value in record.items()
        if str(key) != "_artifact_path"
    }


def _persist_handoff(pipeline: object, record: Dict[str, Any]) -> Dict[str, Any]:
    process_dir = getattr(pipeline, "process_dir", None)
    if process_dir is None:
        record.update(
            status="rejected",
            accepted=False,
            reason_code=REASON_LINEAGE_UNVERIFIED,
            detail="Stage 5 handoff report directory is unavailable",
        )
        setattr(pipeline, "_stage5_linear_handoff", record)
        return record
    report_path = Path(process_dir) / STAGE5_HANDOFF_REPORT
    try:
        run_manifest.atomic_write_json(report_path, public_handoff(record))
    except (OSError, TypeError, ValueError) as error:
        record.update(
            status="rejected",
            accepted=False,
            reason_code=REASON_LINEAGE_UNVERIFIED,
            detail=f"Stage 5 handoff report write failed: {error}",
        )
    setattr(pipeline, "_stage5_linear_handoff", record)
    return record


def freeze_stage5_handoff(
    pipeline: object,
    *,
    origin: str,
    stage_status: str,
    deconvolution_integrity_ok: bool,
    denoise_integrity_ok: bool,
    input_lineage: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze one canonical Stage 5 artifact for same-run Stage 6 use."""

    normalized_origin = str(origin or "").strip()
    normalized_status = str(stage_status or "").strip().lower()
    record: Dict[str, Any] = {
        "schema": STAGE5_HANDOFF_SCHEMA,
        "status": "rejected",
        "accepted": False,
        "source_stage": 5,
        "source_stem": STAGE5_SOURCE_STEM,
        "artifact": STAGE5_SOURCE_ARTIFACT,
        "state": "linear",
        "origin": normalized_origin,
        "run_id": _run_id(pipeline) or None,
        "stage_status": normalized_status,
        "input_integrity_ok": False,
        "input_lineage": (
            dict(input_lineage)
            if isinstance(input_lineage, Mapping)
            else {}
        ),
        "deconvolution_integrity_ok": bool(deconvolution_integrity_ok),
        "denoise_integrity_ok": bool(denoise_integrity_ok),
        "integrity_ok": False,
        "sha256": None,
        "reason_code": REASON_LINEAGE_UNVERIFIED,
        "resume": {
            "provenance_verified": False,
            "checkpoint": None,
            "run_manifest_hash": None,
            "config_fingerprint": None,
            "semantic_context_status": None,
        },
    }
    setattr(pipeline, "_stage5_linear_handoff", record)

    if normalized_origin not in {CURRENT_RUN_ORIGIN, VERIFIED_RESUME_ORIGIN}:
        record["detail"] = "unsupported Stage 5 handoff origin"
        return _persist_handoff(pipeline, record)

    artifact_path = _artifact_path(pipeline)
    if artifact_path is None or not artifact_path.is_file():
        record.update(
            reason_code=REASON_SOURCE_UNAVAILABLE,
            detail="canonical stage5_linear.fit is unavailable",
        )
        return _persist_handoff(pipeline, record)
    artifact_sha256 = run_manifest.sha256_file(artifact_path)
    if not artifact_sha256:
        record.update(
            reason_code=REASON_SOURCE_UNAVAILABLE,
            detail="canonical stage5_linear.fit cannot be hashed",
        )
        return _persist_handoff(pipeline, record)
    record["sha256"] = artifact_sha256

    if normalized_origin == CURRENT_RUN_ORIGIN:
        input_error = _current_input_lineage_error(pipeline, input_lineage)
        if input_error:
            record["detail"] = input_error
            return _persist_handoff(pipeline, record)
        record["input_integrity_ok"] = True
        record["integrity_ok"] = bool(
            deconvolution_integrity_ok and denoise_integrity_ok
        )
        if normalized_status not in {"ok", "degraded"}:
            record["detail"] = "current-run Stage 5 status is not deliverable"
            return _persist_handoff(pipeline, record)
        if not bool(
            deconvolution_integrity_ok and denoise_integrity_ok
        ):
            record["detail"] = "current-run Stage 5 transaction integrity failed"
            return _persist_handoff(pipeline, record)
    else:
        record["input_lineage"] = _resume_input_lineage()
        resume = dict(provenance or {})
        record["resume"] = {
            "provenance_verified": resume.get("verified") is True,
            "checkpoint": resume.get("checkpoint"),
            "run_manifest_hash": resume.get("run_manifest_hash"),
            "config_fingerprint": resume.get("config_fingerprint"),
            "semantic_context_status": resume.get("semantic_context_status"),
        }
        if not (
            resume.get("verified") is True
            and str(resume.get("checkpoint") or "") == "stage5"
            and str(resume.get("state") or "").lower() == "linear"
            and str(resume.get("semantic_context_status") or "") == "verified"
            and str(resume.get("run_manifest_hash") or "").strip()
            and str(resume.get("config_fingerprint") or "").strip()
        ):
            record["detail"] = "Stage 5 resume provenance is incomplete"
            return _persist_handoff(pipeline, record)
        expected_sha256 = str(resume.get("actual_sha256") or "").strip()
        if not expected_sha256 or expected_sha256 != artifact_sha256:
            record["detail"] = "materialized Stage 5 resume SHA-256 mismatch"
            return _persist_handoff(pipeline, record)
        if not bool(
            deconvolution_integrity_ok and denoise_integrity_ok
        ):
            record["detail"] = "verified Stage 5 resume integrity was not asserted"
            return _persist_handoff(pipeline, record)
        record["input_integrity_ok"] = True
        record["integrity_ok"] = True

    record.update(
        status="accepted",
        accepted=True,
        reason_code="accepted",
        detail="canonical Stage 5 lineage frozen for Stage 6",
    )
    return _persist_handoff(pipeline, record)


def verify_stage5_handoff(pipeline: object) -> Dict[str, Any]:
    """Verify the frozen handoff and current canonical bytes fail-closed."""

    record = getattr(pipeline, "_stage5_linear_handoff", None)
    if not isinstance(record, Mapping):
        raise Stage5HandoffError(
            REASON_LINEAGE_UNVERIFIED,
            "Stage 5 -> Stage 6 handoff is missing",
        )
    origin = str(record.get("origin") or "")
    if (
        str(record.get("schema") or "") != STAGE5_HANDOFF_SCHEMA
        or record.get("accepted") is not True
        or str(record.get("status") or "") != "accepted"
        or int(record.get("source_stage", 0) or 0) != 5
        or str(record.get("source_stem") or "") != STAGE5_SOURCE_STEM
        or str(record.get("artifact") or "") != STAGE5_SOURCE_ARTIFACT
        or str(record.get("state") or "") != "linear"
        or origin not in {CURRENT_RUN_ORIGIN, VERIFIED_RESUME_ORIGIN}
        or record.get("input_integrity_ok") is not True
        or record.get("deconvolution_integrity_ok") is not True
        or record.get("denoise_integrity_ok") is not True
        or record.get("integrity_ok") is not True
    ):
        raise Stage5HandoffError(
            REASON_LINEAGE_UNVERIFIED,
            "Stage 5 -> Stage 6 handoff contract is not accepted",
        )
    if str(record.get("run_id") or "") != _run_id(pipeline):
        raise Stage5HandoffError(
            REASON_LINEAGE_UNVERIFIED,
            "Stage 5 -> Stage 6 handoff run_id mismatch",
        )

    if origin == CURRENT_RUN_ORIGIN:
        input_error = _current_input_lineage_error(
            pipeline,
            record.get("input_lineage"),
        )
        if input_error:
            raise Stage5HandoffError(
                REASON_LINEAGE_UNVERIFIED,
                input_error,
            )
    else:
        resume_input_lineage = record.get("input_lineage")
        if not isinstance(resume_input_lineage, Mapping) or not (
            str(resume_input_lineage.get("schema") or "")
            == STAGE5_INPUT_LINEAGE_SCHEMA
            and str(resume_input_lineage.get("status") or "")
            == "verified_resume"
            and resume_input_lineage.get("accepted") is True
            and str(resume_input_lineage.get("mode") or "")
            == "formal_stage5_checkpoint"
        ):
            raise Stage5HandoffError(
                REASON_LINEAGE_UNVERIFIED,
                "verified Stage 5 resume input-lineage marker is invalid",
            )

    artifact_path = _artifact_path(pipeline)
    if artifact_path is None or not artifact_path.is_file():
        raise Stage5HandoffError(
            REASON_SOURCE_UNAVAILABLE,
            "canonical stage5_linear.fit is unavailable",
        )
    actual_sha256 = run_manifest.sha256_file(artifact_path)
    expected_sha256 = str(record.get("sha256") or "")
    if not actual_sha256 or not expected_sha256 or actual_sha256 != expected_sha256:
        raise Stage5HandoffError(
            REASON_LINEAGE_UNVERIFIED,
            "canonical stage5_linear.fit changed after handoff freeze",
        )

    process_dir = getattr(pipeline, "process_dir", None)
    persisted = (
        run_manifest.load_json(Path(process_dir) / STAGE5_HANDOFF_REPORT)
        if process_dir is not None
        else None
    )
    if persisted != public_handoff(record):
        raise Stage5HandoffError(
            REASON_LINEAGE_UNVERIFIED,
            "persisted Stage 5 -> Stage 6 handoff does not match runtime lineage",
        )

    if origin == VERIFIED_RESUME_ORIGIN:
        provenance = getattr(pipeline, "_trusted_input_provenance", None)
        resume_record = record.get("resume")
        if not isinstance(provenance, Mapping) or not (
            provenance.get("verified") is True
            and str(provenance.get("checkpoint") or "") == "stage5"
            and str(provenance.get("state") or "").lower() == "linear"
            and str(provenance.get("semantic_context_status") or "")
            == "verified"
            and str(provenance.get("run_manifest_hash") or "").strip()
            and str(provenance.get("config_fingerprint") or "").strip()
            and str(provenance.get("actual_sha256") or "") == actual_sha256
            and isinstance(resume_record, Mapping)
            and resume_record.get("provenance_verified") is True
            and str(resume_record.get("checkpoint") or "") == "stage5"
            and str(resume_record.get("semantic_context_status") or "")
            == "verified"
            and str(resume_record.get("run_manifest_hash") or "")
            == str(provenance.get("run_manifest_hash") or "")
            and str(resume_record.get("config_fingerprint") or "")
            == str(provenance.get("config_fingerprint") or "")
        ):
            raise Stage5HandoffError(
                REASON_LINEAGE_UNVERIFIED,
                "verified Stage 5 resume provenance no longer matches the artifact",
            )
    return public_handoff(record)


__all__ = [
    "CURRENT_RUN_ORIGIN",
    "REASON_INPUT_CHECKPOINT_FAILED",
    "REASON_LINEAGE_UNVERIFIED",
    "REASON_SOURCE_UNAVAILABLE",
    "STAGE5_HANDOFF_SCHEMA",
    "STAGE5_HANDOFF_REPORT",
    "STAGE5_INPUT_ARTIFACT",
    "STAGE5_INPUT_LINEAGE_SCHEMA",
    "STAGE5_INPUT_STEM",
    "STAGE5_SOURCE_ARTIFACT",
    "STAGE5_SOURCE_STEM",
    "STAGE5_UPSTREAM_ARTIFACT",
    "STAGE5_UPSTREAM_STEM",
    "Stage5HandoffError",
    "VERIFIED_RESUME_ORIGIN",
    "freeze_stage5_input_lineage",
    "freeze_stage5_handoff",
    "public_handoff",
    "verify_stage5_handoff",
]
