"""Evidence-backed physical channel semantics for the processing pipeline."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


BROADBAND_RGB_OSC = "broadband_rgb_osc"
NARROWBAND_COMPOSITE = "narrowband_composite"
MONO = "mono"
NONLINEAR_COLOR = "nonlinear_color"
UNKNOWN = "unknown"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
NARROWBAND_KEYWORDS = frozenset(
    {
        "narrowband",
        "narrow-band",
        "dualband",
        "dual-band",
        "dual narrow",
        "dual-narrow",
        "duo narrow",
        "duo-narrow",
        "l-extreme",
        "l-enhance",
        "l-ultimate",
        "ha+oiii",
        "ha oiii",
        "ha-oiii",
        "ha_oiii",
        "haoiii",
        "h-alpha",
        "oiii",
        "o-iii",
        "triband",
        "tri-band",
        "hoo",
        "sho",
        "sii",
        "s-ii",
        "hubble palette",
    }
)


def channel_shape_dict(shape: Any) -> Dict[str, int]:
    """Normalize Siril's ``(channels, height, width)`` shape contract."""
    if not shape:
        return {}
    try:
        channels, height, width = shape
        return {
            "channels": int(channels),
            "height": int(height),
            "width": int(width),
        }
    except (TypeError, ValueError):
        return {}


def filter_hint_suggests_narrowband(filter_hint: str) -> bool:
    normalized = str(filter_hint or "").strip().lower()
    return bool(
        normalized
        and any(keyword in normalized for keyword in NARROWBAND_KEYWORDS)
    )


def combined_filter_hint(
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    explicit_hint: str = "",
    target_profile: Optional[Mapping[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    values = [
        explicit_hint,
        metadata.get("FILTER", ""),
        metadata.get("FILTER1", ""),
        metadata.get("FILTER2", ""),
    ]
    if target_profile:
        values.extend(
            (
                target_profile.get("filter", ""),
                target_profile.get("filter_name", ""),
            )
        )
    return " ".join(
        str(value or "").strip().lower()
        for value in values
        if str(value or "").strip()
    )


def _resolve_linearity(
    metadata: Mapping[str, Any],
    *,
    input_state: str,
    checkpoint_linear: bool,
) -> Dict[str, Any]:
    normalized_state = str(input_state or "").strip().lower()
    if normalized_state == "linear":
        return {
            "status": "linear",
            "confidence": 1.0,
            "reason": "InputProfile",
        }
    if normalized_state == "nonlinear":
        return {
            "status": "nonlinear",
            "confidence": 1.0,
            "reason": "InputProfile",
        }

    evidence = " ".join(
        str(metadata.get(key, "") or "").strip().lower()
        for key in ("LINEAR", "NONLINEA", "STRETCH", "HISTORY", "PROCSTEP")
    )
    if any(
        token in evidence
        for token in ("nonlinear", "non-linear", "stretched", "histogram")
    ):
        return {
            "status": "nonlinear",
            "confidence": 0.95,
            "reason": "FITS metadata",
        }
    explicit_linear = metadata.get("LINEAR")
    if explicit_linear is True or str(explicit_linear).strip().lower() in _TRUE_VALUES:
        return {
            "status": "linear",
            "confidence": 0.98,
            "reason": "FITS LINEAR keyword",
        }
    if checkpoint_linear:
        return {
            "status": "linear",
            "confidence": 0.96,
            "reason": "verified linear checkpoint contract",
        }
    return {
        "status": "unknown",
        "confidence": 0.0,
        "reason": "linear state not confirmed",
    }


def classify_channel_semantics(
    *,
    channels: int,
    metadata: Optional[Mapping[str, Any]] = None,
    input_state: str = "unknown",
    checkpoint_linear: bool = False,
    explicit_filter_hint: str = "",
    target_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify channels without inferring physical meaning from RGB count alone."""
    metadata = metadata or {}
    linearity = _resolve_linearity(
        metadata,
        input_state=input_state,
        checkpoint_linear=checkpoint_linear,
    )
    filter_hint = combined_filter_hint(
        metadata,
        explicit_hint=explicit_filter_hint,
        target_profile=target_profile,
    )
    narrowband = filter_hint_suggests_narrowband(filter_hint)
    channel_count = max(0, int(channels or 0))

    if channel_count == 1:
        kind = MONO
        confidence = 1.0
        action = "skip_color_calibration"
    elif linearity["status"] == "nonlinear":
        kind = NONLINEAR_COLOR
        confidence = float(linearity["confidence"])
        action = "preserve_input"
    elif channel_count < 3 or linearity["status"] != "linear":
        kind = UNKNOWN
        confidence = 0.0
        action = "preserve_input_review"
    elif narrowband:
        kind = NARROWBAND_COMPOSITE
        confidence = 0.95
        action = "skip_pcc_local_star_only"
    else:
        kind = BROADBAND_RGB_OSC
        confidence = 0.90
        action = "single_pcc"

    evidence = [
        f"channels={channel_count or 'unknown'}",
        f"linearity={linearity['status']}:{linearity['reason']}",
    ]
    if filter_hint:
        evidence.append(f"filter_hint={filter_hint}")

    return {
        "kind": kind,
        "confidence": confidence,
        "action": action,
        "channels": channel_count,
        "linearity": linearity,
        "filter_hint": filter_hint or None,
        "narrowband_detected": bool(narrowband),
        "evidence": evidence,
    }
