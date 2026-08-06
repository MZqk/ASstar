"""Shared, hash-verified task-plan routing primitives.

The GUI and pipeline will consume the same serialized plan.  This P0 module
owns the stable route semantics; input discovery and execution integration are
added in later phases without duplicating the decision tree.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Mapping, Optional

try:
    from . import run_manifest
    from .stage_contracts import (
        FORMAL_RESUME_STAGES,
        PIPELINE_CONTRACT_SCHEMA,
        PIPELINE_CONTRACT_VERSION,
        formal_resume_contracts,
        pipeline_contract_manifest,
        product_stage_contracts,
        stage_contract,
    )
except ImportError:
    import run_manifest
    from stage_contracts import (
        FORMAL_RESUME_STAGES,
        PIPELINE_CONTRACT_SCHEMA,
        PIPELINE_CONTRACT_VERSION,
        formal_resume_contracts,
        pipeline_contract_manifest,
        product_stage_contracts,
        stage_contract,
    )


PROCESSING_PLAN_SCHEMA = "seestar.processing-plan.v2"


class InputTrust(str, Enum):
    """User-facing provenance strength for an analyzed input."""

    VERIFIED = "verified"
    RECOGNIZED = "recognized"
    REVIEW_REQUIRED = "review_required"


class StagePlanAction(str, Enum):
    """Stable product actions rendered by the GUI and enforced by pipeline."""

    VERIFIED = "verified"
    EXECUTE = "execute"
    INPUT_STATE_GUARD = "input_state_guard"
    REVIEW_EXPORT = "review_export"


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _normalize_input_state(value: Any) -> str:
    normalized = _enum_value(value)
    if normalized not in {"linear", "nonlinear", "unknown"}:
        raise ValueError(f"unsupported input state: {value!r}")
    return normalized


def _normalize_input_trust(value: Any) -> InputTrust:
    normalized = _enum_value(value)
    try:
        return InputTrust(normalized)
    except ValueError as error:
        raise ValueError(f"unsupported input trust: {value!r}") from error


def _normalize_resume_stage(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        stage_number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid resume stage: {value!r}") from error
    if stage_number not in FORMAL_RESUME_STAGES:
        raise ValueError(
            f"Stage {stage_number} is not a formal resume boundary; "
            f"expected one of {FORMAL_RESUME_STAGES}"
        )
    return stage_number


def _normalize_checkpoint_fingerprints(
    value: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint_fingerprints must be a mapping")

    allowed_keys = {
        f"stage{contract.number}": contract
        for contract in formal_resume_contracts()
    }
    unexpected_keys = set(value) - set(allowed_keys)
    if unexpected_keys:
        raise ValueError(
            "checkpoint fingerprints contain non-resumable stages: "
            + ", ".join(sorted(str(key) for key in unexpected_keys))
        )

    normalized: Dict[str, Dict[str, Any]] = {}
    for key, raw_record in value.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"checkpoint fingerprint {key!r} must be a mapping")
        contract = allowed_keys[str(key)]
        record = dict(raw_record)
        if record.get("stage") != contract.number:
            raise ValueError(f"checkpoint fingerprint {key!r} has wrong stage")
        if str(record.get("artifact") or "") != contract.primary_artifact:
            raise ValueError(f"checkpoint fingerprint {key!r} has wrong artifact")
        if str(record.get("contract_version") or "") != PIPELINE_CONTRACT_VERSION:
            raise ValueError(
                f"checkpoint fingerprint {key!r} has incompatible contract"
            )
        if not str(record.get("fingerprint") or "").strip():
            raise ValueError(f"checkpoint fingerprint {key!r} has no fingerprint")
        normalized[str(key)] = record
    return normalized


def build_stage_steps(
    *,
    input_state: Any,
    input_trust: Any,
    resume_after_stage: Any = None,
) -> list[Dict[str, Any]]:
    """Build the exact Stage 1-10 product route for an analyzed input."""

    state = _normalize_input_state(input_state)
    trust = _normalize_input_trust(input_trust)
    resume_stage = _normalize_resume_stage(resume_after_stage)
    if resume_stage is not None:
        if state != "linear":
            raise ValueError("only a verified linear input can resume a checkpoint")
        if trust is not InputTrust.VERIFIED:
            raise ValueError("resume requires verified task provenance")

    steps: list[Dict[str, Any]] = []
    for contract in product_stage_contracts():
        if resume_stage is not None and contract.number <= resume_stage:
            action = StagePlanAction.VERIFIED
            reason_code = "verified_checkpoint_chain"
        elif state == "linear":
            action = StagePlanAction.EXECUTE
            reason_code = "planned_processing"
        elif contract.number in (1, 2):
            action = StagePlanAction.EXECUTE
            reason_code = "safe_input_preparation"
        elif contract.number == 10:
            action = StagePlanAction.REVIEW_EXPORT
            reason_code = "review_only_input_state"
        else:
            action = StagePlanAction.INPUT_STATE_GUARD
            reason_code = "input_state_not_linear"
        steps.append(
            {
                "stage": contract.number,
                "key": contract.key,
                "title": contract.title,
                "display_label": contract.display_label,
                "phase": contract.phase.value,
                "primary_artifact": contract.primary_artifact,
                "formal_resume_checkpoint": contract.formal_resume_checkpoint,
                "action": action.value,
                "reason_code": reason_code,
            }
        )
    return steps


def build_resume_fingerprints(
    *,
    input_fingerprint: str,
    stage_config: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build cumulative compatibility hashes for Stage 1/2/5 checkpoints.

    ``stage_config`` must contain only result-affecting, already-redacted
    configuration for its stage.  Runtime/retry/UI-only settings should not be
    included by callers.
    """

    normalized_input = str(input_fingerprint or "").strip()
    if not normalized_input:
        raise ValueError("input_fingerprint is required")
    normalized_config: Dict[int, Dict[str, Any]] = {}
    for raw_stage, payload in dict(stage_config or {}).items():
        try:
            stage_number = int(raw_stage)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid stage config key: {raw_stage!r}") from error
        if stage_number < 1 or stage_number > 5:
            raise ValueError(
                "resume fingerprints only accept Stage 1-5 configuration"
            )
        if not isinstance(payload, Mapping):
            raise ValueError(f"Stage {stage_number} config must be a mapping")
        normalized_config[stage_number] = dict(payload)

    cumulative: Dict[str, Any] = {
        "contract_schema": PIPELINE_CONTRACT_SCHEMA,
        "contract_version": PIPELINE_CONTRACT_VERSION,
        "input_fingerprint": normalized_input,
        "stages": {},
    }
    fingerprints: Dict[str, Dict[str, Any]] = {}
    for stage_number in range(1, 6):
        cumulative["stages"][str(stage_number)] = normalized_config.get(
            stage_number,
            {},
        )
        if stage_number not in FORMAL_RESUME_STAGES:
            continue
        contract = stage_contract(stage_number)
        fingerprints[f"stage{stage_number}"] = {
            "stage": stage_number,
            "artifact": contract.primary_artifact,
            "contract_version": PIPELINE_CONTRACT_VERSION,
            "fingerprint": run_manifest.canonical_payload_hash(cumulative),
        }
    return fingerprints


def latest_compatible_resume_stage(
    saved: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
) -> Optional[int]:
    """Return the latest formal boundary with an identical contract hash."""

    for stage_number in reversed(FORMAL_RESUME_STAGES):
        key = f"stage{stage_number}"
        previous_record = saved.get(key)
        current_record = current.get(key)
        if not isinstance(previous_record, Mapping) or not isinstance(
            current_record,
            Mapping,
        ):
            continue
        previous_hash = str(previous_record.get("fingerprint") or "")
        current_hash = str(current_record.get("fingerprint") or "")
        if previous_hash and previous_hash == current_hash:
            return stage_number
    return None


def build_processing_plan(
    *,
    run_id: str,
    generated_at: str,
    input_record: Mapping[str, Any],
    input_state: Any,
    input_trust: Any,
    resume_after_stage: Any = None,
    checkpoint_fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
    output: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete, deterministic, hash-verified shared task plan."""

    normalized_run_id = str(run_id or "").strip()
    normalized_generated_at = str(generated_at or "").strip()
    if not normalized_run_id:
        raise ValueError("run_id is required")
    if not normalized_generated_at:
        raise ValueError("generated_at is required")
    if not isinstance(input_record, Mapping):
        raise ValueError("input_record must be a mapping")

    state = _normalize_input_state(input_state)
    trust = _normalize_input_trust(input_trust)
    resume_stage = _normalize_resume_stage(resume_after_stage)
    steps = build_stage_steps(
        input_state=state,
        input_trust=trust,
        resume_after_stage=resume_stage,
    )
    normalized_input = dict(input_record)
    if resume_stage is not None and not str(
        normalized_input.get("fingerprint") or ""
    ).strip():
        raise ValueError("resume plans require an input fingerprint")

    normalized_fingerprints = _normalize_checkpoint_fingerprints(
        checkpoint_fingerprints
    )
    if resume_stage is not None:
        resume_key = f"stage{resume_stage}"
        if resume_key not in normalized_fingerprints:
            raise ValueError(
                f"resume plan requires a verified {resume_key} fingerprint"
            )

    payload: Dict[str, Any] = {
        "schema": PROCESSING_PLAN_SCHEMA,
        "run_id": normalized_run_id,
        "generated_at": normalized_generated_at,
        "pipeline_contract": pipeline_contract_manifest(),
        "input": normalized_input,
        "route": {
            "input_state": state,
            "input_trust": trust.value,
            "resume_after_stage": resume_stage,
            "review_only": state != "linear",
        },
        "planned_steps": steps,
        "checkpoint_fingerprints": normalized_fingerprints,
        "output": dict(output or {}),
        "metadata": dict(metadata or {}),
    }
    payload["plan_hash"] = run_manifest.canonical_payload_hash(payload)
    return payload


def verify_processing_plan(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify plan integrity, contract version, and route determinism."""

    result: Dict[str, Any] = {
        "verified": False,
        "detail": "processing plan is invalid",
    }
    if not isinstance(payload, Mapping):
        result["detail"] = "processing plan is not an object"
        return result
    plan = dict(payload)
    if str(plan.get("schema") or "") != PROCESSING_PLAN_SCHEMA:
        result["detail"] = "unsupported processing plan schema"
        return result
    contract = plan.get("pipeline_contract")
    if not isinstance(contract, Mapping):
        result["detail"] = "processing plan has no pipeline contract"
        return result
    if str(contract.get("schema") or "") != PIPELINE_CONTRACT_SCHEMA or str(
        contract.get("version") or ""
    ) != PIPELINE_CONTRACT_VERSION:
        result["detail"] = "processing plan uses an incompatible pipeline contract"
        return result
    if dict(contract) != pipeline_contract_manifest():
        result["detail"] = "processing plan pipeline contract was modified"
        return result

    expected_hash = str(plan.get("plan_hash") or "")
    unsigned_plan = dict(plan)
    unsigned_plan.pop("plan_hash", None)
    actual_hash = run_manifest.canonical_payload_hash(unsigned_plan)
    result["plan_hash"] = expected_hash or None
    if not expected_hash or actual_hash != expected_hash:
        result["detail"] = "processing plan hash is missing or invalid"
        return result

    route = plan.get("route")
    if not isinstance(route, Mapping):
        result["detail"] = "processing plan has no route"
        return result
    try:
        expected_steps = build_stage_steps(
            input_state=route.get("input_state"),
            input_trust=route.get("input_trust"),
            resume_after_stage=route.get("resume_after_stage"),
        )
    except ValueError as error:
        result["detail"] = f"processing plan route is invalid: {error}"
        return result
    if plan.get("planned_steps") != expected_steps:
        result["detail"] = "processing plan steps do not match its frozen route"
        return result
    if bool(route.get("review_only")) != (
        str(route.get("input_state") or "") != "linear"
    ):
        result["detail"] = "processing plan review route is inconsistent"
        return result
    try:
        fingerprints = _normalize_checkpoint_fingerprints(
            plan.get("checkpoint_fingerprints")
        )
    except ValueError as error:
        result["detail"] = f"processing plan checkpoint data is invalid: {error}"
        return result
    if route.get("resume_after_stage") is not None:
        input_record = plan.get("input")
        if not isinstance(input_record, Mapping) or not str(
            input_record.get("fingerprint") or ""
        ).strip():
            result["detail"] = "resume plan input fingerprint is missing"
            return result
        if f"stage{int(route['resume_after_stage'])}" not in fingerprints:
            result["detail"] = "resume plan checkpoint fingerprint is missing"
            return result

    result["verified"] = True
    result["detail"] = "processing plan hash and stage route are valid"
    result["resume_after_stage"] = route.get("resume_after_stage")
    return result


__all__ = [
    "InputTrust",
    "PROCESSING_PLAN_SCHEMA",
    "StagePlanAction",
    "build_processing_plan",
    "build_resume_fingerprints",
    "build_stage_steps",
    "latest_compatible_resume_stage",
    "verify_processing_plan",
]
