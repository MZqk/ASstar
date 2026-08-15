"""Evidence-backed physical channel semantics for the processing pipeline."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from narrowband_normalization import (
    resolve_dual_narrowband_mapping,
    select_filter_header_evidence,
)


BROADBAND_RGB_OSC = "broadband_rgb_osc"
NARROWBAND_COMPOSITE = "narrowband_composite"
MONO = "mono"
NONLINEAR_COLOR = "nonlinear_color"
UNKNOWN = "unknown"

CHANNEL_SEMANTICS_SCHEMA = "starun.channel-semantics.v2"
TRANSFER_LINEAR = "linear"
TRANSFER_NONLINEAR = "nonlinear"
TRANSFER_UNKNOWN = "unknown"
LAYOUT_MONO = "mono"
LAYOUT_RGB = "rgb"
LAYOUT_MULTICHANNEL = "multichannel"
LAYOUT_UNKNOWN = "unknown"
SPECTRAL_BROADBAND = "broadband"
SPECTRAL_DUALBAND_HA_OIII = "dualband_ha_oiii"
SPECTRAL_NARROWBAND_OTHER = "narrowband_other_or_composite"
SPECTRAL_UNKNOWN = "unknown"
COMPOSITION_SINGLE_CHANNEL = "single_channel"
COMPOSITION_RGB_OR_OSC = "rgb_or_osc"
COMPOSITION_NARROWBAND_RGB = "narrowband_rgb_composite"
COMPOSITION_UNKNOWN = "unknown"

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
    selection = select_filter_header_evidence(metadata)
    selected = selection.get("selected_filter_headers") or []
    if selected:
        values = [item.get("normalized_filter", "") for item in selected]
    else:
        values = [explicit_hint]
    if target_profile and not selected:
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


def _spectral_axis(
    *,
    filter_hint: str,
    narrowband: bool,
    device_filter_match: Optional[Mapping[str, Any]],
    mapping_supplied: bool = False,
    mapping_confirmed: bool = False,
) -> Dict[str, Any]:
    """Preserve spectral identity independently from transfer/layout state."""
    normalized = str(filter_hint or "").strip().lower()
    dualband_ha_oiii = bool(mapping_confirmed)
    if not mapping_supplied:
        dualband_ha_oiii = bool(
            mapping_confirmed
            or device_filter_match
            or any(
                token in normalized
                for token in (
                    "dualband",
                    "dual-band",
                    "duo narrow",
                    "duo-narrow",
                    "ha+oiii",
                    "ha oiii",
                    "ha-oiii",
                    "ha_oiii",
                    "haoiii",
                    "l-extreme",
                    "l-enhance",
                    "l-ultimate",
                )
            )
        )
    if narrowband and dualband_ha_oiii:
        kind = SPECTRAL_DUALBAND_HA_OIII
        confidence = 0.95
    elif narrowband:
        kind = SPECTRAL_NARROWBAND_OTHER
        confidence = 0.75
    elif normalized:
        kind = SPECTRAL_BROADBAND
        confidence = 0.80
    else:
        kind = SPECTRAL_UNKNOWN
        confidence = 0.0
    return {
        "kind": kind,
        "confidence": confidence,
        "narrowband_detected": bool(narrowband),
        "filter_hint": normalized or None,
    }


def _layout_axis(channel_count: int) -> Dict[str, Any]:
    if channel_count == 1:
        kind = LAYOUT_MONO
        supported = True
    elif channel_count == 3:
        kind = LAYOUT_RGB
        supported = True
    elif channel_count > 3:
        kind = LAYOUT_MULTICHANNEL
        supported = False
    else:
        kind = LAYOUT_UNKNOWN
        supported = False
    return {
        "kind": kind,
        "channels": channel_count,
        "supported": supported,
    }


def _composition_axis(
    *,
    channel_count: int,
    narrowband: bool,
) -> Dict[str, Any]:
    if channel_count == 1:
        kind = COMPOSITION_SINGLE_CHANNEL
        supported = True
    elif channel_count == 3 and narrowband:
        kind = COMPOSITION_NARROWBAND_RGB
        supported = True
    elif channel_count == 3:
        kind = COMPOSITION_RGB_OR_OSC
        supported = True
    else:
        kind = COMPOSITION_UNKNOWN
        supported = False
    return {
        "kind": kind,
        "supported": supported,
        "lrgb_composed": False,
        "rgb_ha_composed": False,
        "independent_channel_roles_confirmed": False,
    }


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
    narrowband_mapping: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify channels without inferring physical meaning from RGB count alone."""
    metadata = metadata or {}
    linearity = _resolve_linearity(
        metadata,
        input_state=input_state,
        checkpoint_linear=checkpoint_linear,
    )
    mapping_supplied = isinstance(narrowband_mapping, Mapping)
    if isinstance(narrowband_mapping, dict):
        mapping_value = narrowband_mapping
    elif mapping_supplied:
        mapping_value = dict(narrowband_mapping or {})
    else:
        mapping_value = resolve_dual_narrowband_mapping(
            metadata,
            filter_hint=explicit_filter_hint,
        )
    if mapping_supplied:
        mapping_detail = mapping_value.get("evidence_detail") or {}
        selected_headers = (
            mapping_detail.get("selected_filter_headers")
            if isinstance(mapping_detail, Mapping)
            else []
        )
        filter_hint = " ".join(
            str(item.get("normalized_filter") or "").strip().lower()
            for item in (selected_headers or [])
            if isinstance(item, Mapping)
            and str(item.get("normalized_filter") or "").strip()
        )
    else:
        filter_hint = combined_filter_hint(
            metadata,
            explicit_hint=explicit_filter_hint,
            target_profile=target_profile,
        )
    mapping_confirmed = bool(
        mapping_value.get("mapping") == "osc_hoo_rgb"
        and mapping_value.get("ha_channel") == "R"
        and list(mapping_value.get("oiii_channels") or []) == ["G", "B"]
    )
    mapping_detail = mapping_value.get("evidence_detail") or {}
    if (
        mapping_value.get("evidence") == "verified_device_profile"
        and isinstance(mapping_detail, Mapping)
    ):
        device_filter_match = {
            "profile_id": mapping_detail.get("device_profile_id"),
            "instrument": mapping_detail.get("instrument"),
            "header_key": mapping_detail.get("header_key"),
            "filter_value": mapping_detail.get("filter_value"),
            "normalized_filter": mapping_detail.get("normalized_filter"),
        }
    else:
        device_filter_match = None
    narrowband = bool(
        mapping_confirmed
        or filter_hint_suggests_narrowband(filter_hint)
        or device_filter_match
    )
    channel_count = max(0, int(channels or 0))

    layout_axis = _layout_axis(channel_count)
    spectral_axis = _spectral_axis(
        filter_hint=filter_hint,
        narrowband=narrowband,
        device_filter_match=device_filter_match,
        mapping_supplied=mapping_supplied,
        mapping_confirmed=mapping_confirmed,
    )
    composition_axis = _composition_axis(
        channel_count=channel_count,
        narrowband=narrowband,
    )
    transfer_axis = {
        "kind": str(linearity["status"]),
        "confidence": float(linearity["confidence"]),
        "reason": str(linearity["reason"]),
    }
    unsupported_reasons = []
    if channel_count > 3:
        unsupported_reasons.append(
            "multichannel_layout_requires_explicit_channel_roles"
        )
    elif channel_count in {0, 2}:
        unsupported_reasons.append("unsupported_or_unknown_channel_layout")
    if linearity["status"] == TRANSFER_UNKNOWN:
        unsupported_reasons.append("transfer_state_unconfirmed")

    if channel_count == 1:
        kind = MONO
        confidence = 1.0
        action = "skip_color_calibration"
    elif channel_count > 3:
        # Do not silently treat RGBA, LRGB, RGB+Ha, or arbitrary cubes as RGB.
        kind = UNKNOWN
        confidence = 0.0
        action = "preserve_input_review"
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
        action = "spcc_narrowband_then_degraded_pcc_with_isolated_hoo"
    else:
        kind = BROADBAND_RGB_OSC
        confidence = 0.90
        action = "spcc_then_pcc"

    evidence = [
        f"channels={channel_count or 'unknown'}",
        f"linearity={linearity['status']}:{linearity['reason']}",
    ]
    if filter_hint:
        evidence.append(f"filter_hint={filter_hint}")
    if device_filter_match:
        evidence.append(
            "device_filter="
            f"{device_filter_match.get('profile_id')}:"
            f"{device_filter_match.get('header_key')}:"
            f"{device_filter_match.get('normalized_filter')}"
        )

    evidence.extend(
        (
            f"layout={layout_axis['kind']}",
            f"spectral={spectral_axis['kind']}",
            f"composition={composition_axis['kind']}",
        )
    )

    color_operations_authorized = bool(
        kind in {BROADBAND_RGB_OSC, NARROWBAND_COMPOSITE}
        and linearity["status"] == TRANSFER_LINEAR
        and layout_axis["kind"] == LAYOUT_RGB
        and composition_axis["supported"]
    )

    return {
        "schema": CHANNEL_SEMANTICS_SCHEMA,
        "kind": kind,
        "confidence": confidence,
        "action": action,
        "channels": channel_count,
        "linearity": linearity,
        "filter_hint": filter_hint or None,
        "narrowband_detected": bool(narrowband),
        "device_filter_match": device_filter_match,
        "narrowband_mapping": mapping_value if mapping_supplied else None,
        "axes": {
            "transfer": transfer_axis,
            "layout": layout_axis,
            "spectral": spectral_axis,
            "composition": composition_axis,
        },
        "color_operations_authorized": color_operations_authorized,
        "unsupported_reasons": unsupported_reasons,
        "evidence": evidence,
    }
