"""Canonical integrity binding for the Stage 8 to Stage 9 handoff."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Mapping


STAGE8_HANDOFF_INTEGRITY_SCHEMA = "starun.stage8-handoff-integrity.v1"
STAGE8_HANDOFF_DIGEST_METHOD = "sha256"
STAGE8_HANDOFF_CANONICALIZATION = "utf8-json-sort-keys-compact-v1"

_INTEGRITY_FIELD = "handoff_integrity"
_ROUTE_EVIDENCE_FIELD = "route_evidence_summary"
# ``final_quality`` remains a diagnostic label.  Stage 9 authorization must not
# depend on a mutable status string; every actual safety and identity field is
# bound below.
_UNBOUND_DIAGNOSTIC_FIELDS = frozenset({"final_quality"})


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _route_evidence_summary(handoff: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the safety evidence whose meaning is selected by the route."""

    route = str(handoff.get("processing_route") or "")
    common = {
        "spatial_background_lineage": copy.deepcopy(
            handoff.get("spatial_background_lineage")
        ),
        "final_cumulative_quality_verified": handoff.get(
            "final_cumulative_quality_verified"
        ),
    }
    if route == "safe_passthrough_color_only":
        route_evidence = {
            "safe_passthrough_color_only": copy.deepcopy(
                handoff.get("safe_passthrough_color_only")
            ),
            "color_gate": copy.deepcopy(handoff.get("color_gate")),
            "star_halo_guard": copy.deepcopy(handoff.get("star_halo_guard")),
            "final_cumulative_quality": copy.deepcopy(
                handoff.get("final_cumulative_quality")
            ),
        }
    elif route == "structure_enhanced":
        route_evidence = {
            "final_cumulative_quality": copy.deepcopy(
                handoff.get("final_cumulative_quality")
            ),
            "starless_finish": copy.deepcopy(handoff.get("starless_finish")),
            "subject_chroma": copy.deepcopy(handoff.get("subject_chroma")),
            "color_gate": copy.deepcopy(handoff.get("color_gate")),
            "star_halo_guard": copy.deepcopy(handoff.get("star_halo_guard")),
            "saturation_execution": copy.deepcopy(
                handoff.get("saturation_execution")
            ),
        }
    elif route == "star_preserve_secondary_nebulosity":
        route_evidence = {
            "star_preserve_secondary_nebulosity": copy.deepcopy(
                handoff.get("star_preserve_secondary_nebulosity")
            ),
        }
    else:
        route_evidence = {
            "reason_code": handoff.get("reason_code"),
            "reasons": copy.deepcopy(handoff.get("reasons")),
        }
    return {
        "processing_route": route,
        "common": common,
        "route_evidence": route_evidence,
    }


def canonical_stage8_handoff_payload(
    handoff: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the canonical v3 safety payload, excluding only diagnostics."""

    bound_record = {
        str(key): copy.deepcopy(value)
        for key, value in handoff.items()
        if key not in {
            _INTEGRITY_FIELD,
            _ROUTE_EVIDENCE_FIELD,
            *_UNBOUND_DIAGNOSTIC_FIELDS,
        }
    }
    return {
        "handoff_schema": str(handoff.get("schema") or ""),
        "bound_record": bound_record,
        "route_evidence_summary": _route_evidence_summary(handoff),
    }


def verify_stage8_handoff_integrity(
    handoff: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recompute every v3 handoff binding; missing or altered data fails closed."""

    issues = []
    integrity = handoff.get(_INTEGRITY_FIELD)
    if not isinstance(integrity, Mapping):
        integrity = {}
        issues.append("stage8_handoff_integrity_missing")
    if integrity.get("schema") != STAGE8_HANDOFF_INTEGRITY_SCHEMA:
        issues.append("stage8_handoff_integrity_schema_invalid")
    if integrity.get("digest_method") != STAGE8_HANDOFF_DIGEST_METHOD:
        issues.append("stage8_handoff_integrity_digest_method_invalid")
    if integrity.get("canonicalization") != STAGE8_HANDOFF_CANONICALIZATION:
        issues.append("stage8_handoff_integrity_canonicalization_invalid")
    if integrity.get("writer_self_verified") is not True:
        issues.append("stage8_handoff_writer_self_verification_missing")

    expected_route_summary = _route_evidence_summary(handoff)
    stored_route_summary = handoff.get(_ROUTE_EVIDENCE_FIELD)
    if not isinstance(stored_route_summary, Mapping):
        issues.append("stage8_handoff_route_evidence_summary_missing")
    elif dict(stored_route_summary) != expected_route_summary:
        issues.append("stage8_handoff_route_evidence_summary_mismatch")

    expected_route_digest = _canonical_sha256(expected_route_summary)
    stored_route_digest = str(integrity.get("route_evidence_sha256") or "")
    if not stored_route_digest:
        issues.append("stage8_handoff_route_evidence_digest_missing")
    elif stored_route_digest != expected_route_digest:
        issues.append("stage8_handoff_route_evidence_digest_mismatch")

    expected_digest = _canonical_sha256(
        canonical_stage8_handoff_payload(handoff)
    )
    stored_digest = str(integrity.get("canonical_sha256") or "")
    if not stored_digest:
        issues.append("stage8_handoff_canonical_digest_missing")
    elif stored_digest != expected_digest:
        issues.append("stage8_handoff_canonical_digest_mismatch")

    issues = list(dict.fromkeys(issues))
    return {
        "schema": "starun.stage8-handoff-integrity-verification.v1",
        "status": "verified" if not issues else "rejected",
        "accepted": not issues,
        "issues": issues,
        "processing_route": str(handoff.get("processing_route") or "") or None,
        "stored_canonical_sha256": stored_digest or None,
        "recomputed_canonical_sha256": expected_digest,
        "stored_route_evidence_sha256": stored_route_digest or None,
        "recomputed_route_evidence_sha256": expected_route_digest,
    }


def seal_stage8_handoff(handoff: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind a handoff and self-verify it before the writer persists the record."""

    sealed = copy.deepcopy(dict(handoff))
    sealed.pop(_INTEGRITY_FIELD, None)
    sealed[_ROUTE_EVIDENCE_FIELD] = _route_evidence_summary(sealed)
    route_digest = _canonical_sha256(sealed[_ROUTE_EVIDENCE_FIELD])
    canonical_digest = _canonical_sha256(
        canonical_stage8_handoff_payload(sealed)
    )
    sealed[_INTEGRITY_FIELD] = {
        "schema": STAGE8_HANDOFF_INTEGRITY_SCHEMA,
        "digest_method": STAGE8_HANDOFF_DIGEST_METHOD,
        "canonicalization": STAGE8_HANDOFF_CANONICALIZATION,
        "canonical_sha256": canonical_digest,
        "route_evidence_sha256": route_digest,
        "writer_self_verified": True,
    }
    verification = verify_stage8_handoff_integrity(sealed)
    if verification.get("accepted") is not True:
        raise ValueError(
            "stage8_handoff_writer_self_verification_failed: "
            + ", ".join(str(item) for item in verification.get("issues") or [])
        )
    return sealed
