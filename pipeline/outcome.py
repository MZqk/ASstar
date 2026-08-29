"""Canonical Starun stage and run outcome semantics.

The pipeline writes v2 payloads. Historical v1 and v3 payloads are normalized
only after their existing manifest hash has been verified by the caller; v3 is
read-only legacy evidence and can never regain formal-delivery eligibility.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


PIPELINE_RESULT_SCHEMA_V1 = "starun.pipeline-result.v1"
PIPELINE_RESULT_SCHEMA_V2 = "starun.pipeline-result.v2"
PIPELINE_RESULT_SCHEMA_V3 = "starun.pipeline-result.v3"
PIPELINE_STAGE_DETAIL_SCHEMA_V1 = "starun.pipeline-stage-detail.v1"
PIPELINE_STAGE_DETAIL_SCHEMA_V2 = "starun.pipeline-stage-detail.v2"
RUN_STATE_SCHEMA_V1 = "starun.run-state.v1"
RUN_STATE_SCHEMA_V2 = "starun.run-state.v2"

SUPPORTED_PIPELINE_RESULT_SCHEMAS = frozenset(
    {
        PIPELINE_RESULT_SCHEMA_V1,
        PIPELINE_RESULT_SCHEMA_V2,
        PIPELINE_RESULT_SCHEMA_V3,
    }
)
SUPPORTED_RUN_STATE_SCHEMAS = frozenset(
    {RUN_STATE_SCHEMA_V1, RUN_STATE_SCHEMA_V2}
)

STAGE_STATUSES = frozenset({"ok", "degraded", "failed", "skipped"})
STAGE_EXECUTIONS = frozenset({"completed", "safe_passthrough", "skipped"})
ISSUE_SEVERITIES = frozenset({"warning", "error", "fatal"})
FINAL_STATUSES = frozenset(
    {"success", "partial_success", "review_required", "failed"}
)


def normalize_review_requirement(
    value: Mapping[str, Any],
    *,
    legacy_inferred: bool = False,
) -> dict[str, Any]:
    """Return one JSON-safe, stage-attributed review requirement."""

    try:
        stage = int(value.get("stage"))
    except (TypeError, ValueError) as error:
        raise ValueError("review requirement has no valid stage") from error
    if stage not in range(1, 11):
        raise ValueError(f"review requirement stage is out of range: {stage}")
    code = str(value.get("code") or "").strip()
    if not code:
        raise ValueError("review requirement has no reason code")
    details = value.get("details")
    if not isinstance(details, Mapping):
        details = {}
    normalized = {
        "stage": stage,
        "code": code,
        "details": deepcopy(dict(details)),
    }
    if legacy_inferred or bool(value.get("legacy_inferred", False)):
        normalized["legacy_inferred"] = True
    return normalized


def normalize_issue(
    value: Mapping[str, Any],
    *,
    default_stage: int | None = None,
) -> dict[str, Any]:
    """Return one canonical issue record."""

    raw_stage = value.get("stage", default_stage)
    try:
        stage = int(raw_stage) if raw_stage is not None else 0
    except (TypeError, ValueError):
        stage = 0
    if stage not in range(1, 11):
        stage = int(default_stage or 0)
    severity = str(value.get("severity") or "warning").strip().lower()
    if severity not in ISSUE_SEVERITIES:
        raise ValueError(f"unsupported issue severity: {severity!r}")
    code = str(value.get("code") or "unspecified_issue").strip()
    component = str(value.get("component") or "stage").strip() or "stage"
    message = str(value.get("message") or code).strip()
    return {
        "stage": stage,
        "component": component,
        "severity": severity,
        "code": code,
        "recovered": bool(value.get("recovered", False)),
        "message": message,
    }


def deduplicate_review_requirements(
    values: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered: dict[tuple[int, str], dict[str, Any]] = {}
    for value in values:
        normalized = normalize_review_requirement(value)
        ordered[(normalized["stage"], normalized["code"])] = normalized
    return list(ordered.values())


def summarize_outcome(
    steps: Sequence[Mapping[str, Any]],
    review_requirements: Sequence[Mapping[str, Any]],
    *,
    failure_reason: Any = None,
    extra_issues: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Derive the only supported run-level status truth table."""

    normalized_reviews = deduplicate_review_requirements(review_requirements)
    issues: list[dict[str, Any]] = []
    had_degradations = False
    had_fallbacks = False
    failed_stages = 0
    for raw_step in steps:
        try:
            stage = int(raw_step.get("stage") or 0)
        except (TypeError, ValueError):
            stage = 0
        status = str(raw_step.get("status") or "").strip().lower()
        fallback_used = bool(raw_step.get("fallback_used", False))
        if fallback_used and status == "ok":
            status = "degraded"
        if status == "degraded":
            had_degradations = True
        if fallback_used:
            had_fallbacks = True
        if status == "failed":
            failed_stages += 1
        raw_issues = raw_step.get("issues")
        if isinstance(raw_issues, Sequence) and not isinstance(
            raw_issues, (str, bytes)
        ):
            for raw_issue in raw_issues:
                if isinstance(raw_issue, Mapping):
                    issues.append(normalize_issue(raw_issue, default_stage=stage))
    for raw_issue in extra_issues:
        if isinstance(raw_issue, Mapping):
            issues.append(normalize_issue(raw_issue))

    fatal_from_reason = bool(str(failure_reason or "").strip())
    had_fatal_errors = bool(
        fatal_from_reason
        or failed_stages
        or any(issue["severity"] == "fatal" for issue in issues)
    )
    had_errors = bool(
        had_fatal_errors
        or any(issue["severity"] in {"error", "fatal"} for issue in issues)
    )
    review_required = bool(normalized_reviews)
    recovered_errors = sum(
        1
        for issue in issues
        if issue["severity"] == "error" and issue["recovered"]
    )
    if had_fatal_errors:
        status = "failed"
    elif review_required:
        status = "review_required"
    elif had_degradations or had_fallbacks or recovered_errors:
        status = "partial_success"
    else:
        status = "success"
    return {
        "status": status,
        "had_errors": had_errors,
        "had_fatal_errors": had_fatal_errors,
        "had_degradations": had_degradations,
        "had_fallbacks": had_fallbacks,
        "review_required": review_required,
        "issues": issues,
        "errors": [
            issue["message"]
            for issue in issues
            if issue["severity"] in {"error", "fatal"}
        ],
        "outcome_counts": {
            "failed_stages": failed_stages,
            "degraded_stages": sum(
                1
                for step in steps
                if str(step.get("status") or "").strip().lower() == "degraded"
                or bool(step.get("fallback_used", False))
            ),
            "fallback_stages": sum(
                1 for step in steps if bool(step.get("fallback_used", False))
            ),
            "review_requirements": len(normalized_reviews),
            "issues": len(issues),
            "recovered_errors": recovered_errors,
        },
    }


def _legacy_review_requirements(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    review = payload.get("review_requirements")
    if isinstance(review, Mapping):
        mapping = (
            ("stage2_view_review_required", 2, "stage2_view_review_required"),
            (
                "stage3_background_review_required",
                3,
                "stage3_background_review_required",
            ),
            ("stage9_psf_review_required", 9, "stage9_psf_review_required"),
            (
                "stage9_review_candidate_selected",
                9,
                "stage9_review_candidate_selected",
            ),
        )
        for field, stage, code in mapping:
            if bool(review.get(field, False)):
                requirements.append(
                    normalize_review_requirement(
                        {"stage": stage, "code": code, "details": {}},
                        legacy_inferred=True,
                    )
                )
    color = payload.get("color_calibration")
    if isinstance(color, Mapping) and bool(color.get("requires_review", False)):
        requirements.append(
            normalize_review_requirement(
                {
                    "stage": 4,
                    "code": "legacy_color_calibration_review_required",
                    "details": {},
                },
                legacy_inferred=True,
            )
        )
    return deduplicate_review_requirements(requirements)


def normalize_pipeline_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a verified v1/v2/v3 result into the v2 in-memory contract."""

    source_schema = str(payload.get("schema") or "")
    if source_schema not in SUPPORTED_PIPELINE_RESULT_SCHEMAS:
        raise ValueError(f"unsupported pipeline result schema: {source_schema!r}")
    normalized = deepcopy(dict(payload))
    raw_steps = normalized.get("actual_steps")
    steps: list[dict[str, Any]] = []
    if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, (str, bytes)):
        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, Mapping):
                continue
            step = deepcopy(dict(raw_step))
            status = str(step.get("status") or "skipped").strip().lower()
            if bool(step.get("fallback_used", False)) and status == "ok":
                status = "degraded"
            if status not in STAGE_STATUSES:
                status = "failed"
            step["status"] = status
            step.setdefault("stage", index)
            step.setdefault("review_reasons", [])
            step.setdefault("issues", [])
            steps.append(step)
    raw_reviews = normalized.get("review_requirements")
    if source_schema in {
        PIPELINE_RESULT_SCHEMA_V2,
        PIPELINE_RESULT_SCHEMA_V3,
    } and isinstance(raw_reviews, list):
        reviews = deduplicate_review_requirements(
            value for value in raw_reviews if isinstance(value, Mapping)
        )
    else:
        reviews = _legacy_review_requirements(normalized)
    summary = summarize_outcome(
        steps,
        reviews,
        failure_reason=normalized.get("failure_reason"),
    )
    normalized.update(summary)
    normalized["schema"] = PIPELINE_RESULT_SCHEMA_V2
    normalized["source_schema"] = source_schema
    normalized["legacy_inferred"] = source_schema != PIPELINE_RESULT_SCHEMA_V2
    normalized["actual_steps"] = steps
    normalized["review_requirements"] = reviews
    if source_schema == PIPELINE_RESULT_SCHEMA_V3:
        raw_delivery_gates = normalized.get("delivery_gates")
        delivery_gates = (
            deepcopy(dict(raw_delivery_gates))
            if isinstance(raw_delivery_gates, Mapping)
            else {}
        )
        delivery_gates.update(
            legacy_delivery_contract=True,
            formal_delivery_accepted=False,
        )
        normalized["delivery_gates"] = delivery_gates
        normalized["delivery_eligible"] = False
        normalized["legacy_schema_read_only"] = True
    return normalized


def normalize_run_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a verified historical run-state payload to v2 semantics."""

    source_schema = str(payload.get("schema") or "")
    if source_schema not in SUPPORTED_RUN_STATE_SCHEMAS:
        raise ValueError(f"unsupported run-state schema: {source_schema!r}")
    normalized = deepcopy(dict(payload))
    raw_status = str(normalized.get("status") or "").strip().lower()
    status = {
        "completed": "success",
        "completedwithwarning": "partial_success",
        "failed": "failed",
        "stopped": "stopped",
    }.get(raw_status, raw_status)
    raw_issues = normalized.get("issues")
    issues = [
        normalize_issue(issue)
        for issue in raw_issues or []
        if isinstance(issue, Mapping)
    ]
    had_fatal_errors = bool(
        normalized.get("had_fatal_errors", False)
        or status == "failed"
        or any(issue["severity"] == "fatal" for issue in issues)
    )
    had_errors = bool(
        normalized.get("had_errors", False)
        or had_fatal_errors
        or any(issue["severity"] in {"error", "fatal"} for issue in issues)
    )
    if source_schema == RUN_STATE_SCHEMA_V1 and not issues:
        for message in normalized.get("errors") or []:
            issues.append(
                normalize_issue(
                    {
                        "stage": 0,
                        "component": "legacy_worker",
                        "severity": "fatal" if had_fatal_errors else "error",
                        "code": "legacy_run_error",
                        "recovered": not had_fatal_errors,
                        "message": str(message),
                    }
                )
            )
    had_errors = bool(
        had_errors
        or any(issue["severity"] in {"error", "fatal"} for issue in issues)
    )
    normalized.update(
        {
            "schema": RUN_STATE_SCHEMA_V2,
            "source_schema": source_schema,
            "legacy_inferred": source_schema == RUN_STATE_SCHEMA_V1,
            "status": status,
            "had_errors": had_errors,
            "had_fatal_errors": had_fatal_errors,
            "had_degradations": bool(
                normalized.get("had_degradations", False)
            ),
            "had_fallbacks": bool(normalized.get("had_fallbacks", False)),
            "review_required": bool(
                normalized.get("review_required", False)
                or status == "review_required"
            ),
            "issues": issues,
            "errors": [
                issue["message"]
                for issue in issues
                if issue["severity"] in {"error", "fatal"}
            ],
        }
    )
    return normalized


__all__ = [
    "FINAL_STATUSES",
    "PIPELINE_RESULT_SCHEMA_V1",
    "PIPELINE_RESULT_SCHEMA_V2",
    "PIPELINE_RESULT_SCHEMA_V3",
    "PIPELINE_STAGE_DETAIL_SCHEMA_V1",
    "PIPELINE_STAGE_DETAIL_SCHEMA_V2",
    "RUN_STATE_SCHEMA_V1",
    "RUN_STATE_SCHEMA_V2",
    "SUPPORTED_PIPELINE_RESULT_SCHEMAS",
    "SUPPORTED_RUN_STATE_SCHEMAS",
    "deduplicate_review_requirements",
    "normalize_issue",
    "normalize_pipeline_result",
    "normalize_run_state",
    "normalize_review_requirement",
    "summarize_outcome",
]
