"""Deterministic Ha/OIII channel mapping for confirmed dual-narrowband data."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional

from device_geometry import (
    resolve_smart_device_narrowband_filter,
    smart_device_wide_path_reason,
)


NARROWBAND_CHANNEL_MAPPING_SCHEMA = "starun.narrowband-channel-mapping.v1"

_PRIMARY_FILTER_HEADER_KEY = "FILTER"
_SUPPLEMENTAL_FILTER_HEADER_KEYS = (
    "FILTER1",
    "FILTER2",
    "INSFLNAM",
    "FILTERNAME",
    "FILTNAME",
    "FILTNAM",
)
_FILTER_HEADER_KEYS = (
    _PRIMARY_FILTER_HEADER_KEY,
    *_SUPPLEMENTAL_FILTER_HEADER_KEYS,
)
_VERIFIED_FILTER_PROFILES = (
    ("zwo_duo_band", ("zwo duo band", "zwo dual band")),
    (
        "optolong_l_extreme",
        ("optolong l extreme", "l extreme", "lextreme"),
    ),
    (
        "optolong_l_ultimate",
        ("optolong l ultimate", "l ultimate", "lultimate"),
    ),
    ("optolong_l_para", ("optolong l para", "l para", "lpara")),
    (
        "idas_nbz",
        ("idas nbz", "idas nbz ii", "nbz", "nbz ii", "nbzii"),
    ),
    ("svbony_sv220", ("svbony sv220", "sv220")),
)
_GENERIC_DUALBAND_PHRASES = (
    "dual band",
    "dualband",
    "duo band",
    "duoband",
)
_BROADBAND_FILTER_PHRASES = (
    "no filter",
    "no lp",
    "without lp",
    "lp off",
    "ircut",
    "ir cut",
    "uv ir",
    "uv ir cut",
    "uv ir block",
    "astro",
    "vis",
    "clear",
    "broadband",
    "astro filter",
)
_AUTHORITATIVE_LP_VALUES = frozenset(("lp", "lp starless"))
_MULTIBAND_FILTER_PHRASES = (
    "tri band",
    "triband",
    "quad band",
    "quadband",
    "multi band",
    "multiband",
)


def _unpack_metadata_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    return value


def _metadata_index(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        str(key).strip().upper(): _unpack_metadata_value(value)
        for key, value in (metadata or {}).items()
    }


def _normalize_filter_text(value: Any) -> str:
    text = str(_unpack_metadata_value(value) or "").casefold()
    text = (
        text.replace("hα", "ha")
        .replace("oⅲ", "oiii")
        .replace("oxygen iii", "oiii")
        .replace("hydrogen alpha", "ha")
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _filter_header_values(
    metadata: Optional[Mapping[str, Any]],
) -> list[Dict[str, str]]:
    indexed = _metadata_index(metadata)
    values: list[Dict[str, str]] = []
    for key in _FILTER_HEADER_KEYS:
        raw = indexed.get(key)
        normalized = _normalize_filter_text(raw)
        if normalized:
            values.append(
                {
                    "header_key": key,
                    "filter_value": str(raw or "").strip(),
                    "normalized_filter": normalized,
                }
            )
    return values


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _contains_ha(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(
        re.search(r"(?:^| )ha(?: |$)", text)
        or "halpha" in compact
        or "6563" in compact
        or "65628" in compact
    )


def _contains_oiii(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(
        "oiii" in compact
        or re.search(r"(?:^| )o3(?: |$)", text)
        or "5007" in compact
        or "50070" in compact
    )


def _contains_sii(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(
        "sii" in compact
        or re.search(r"(?:^| )s2(?: |$)", text)
        or "6724" in compact
    )


def _filter_value_category(text: str) -> str:
    """Classify one normalized filter field without consulting other fields."""
    if not text:
        return "unknown"
    if _contains_sii(text) or _contains_phrase(text, _MULTIBAND_FILTER_PHRASES):
        return "conflicting_lines"
    if _contains_phrase(text, _BROADBAND_FILTER_PHRASES):
        return "broadband"
    has_ha = _contains_ha(text)
    has_oiii = _contains_oiii(text)
    if (has_ha and has_oiii) or _contains_phrase(text, ("hoo",)):
        return "explicit_hoo"
    for _profile_id, aliases in _VERIFIED_FILTER_PROFILES:
        if _contains_phrase(text, aliases):
            return "verified_filter"
    if has_ha or has_oiii:
        return "single_line"
    if text in _AUTHORITATIVE_LP_VALUES:
        return "authoritative_lp"
    if _contains_phrase(text, _GENERIC_DUALBAND_PHRASES):
        return "generic_dualband"
    return "unknown"


def select_filter_header_evidence(
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select the one authoritative FILTER view used by all Stage 4 consumers."""
    header_values = _filter_header_values(metadata)
    primary = next(
        (
            item
            for item in header_values
            if item["header_key"] == _PRIMARY_FILTER_HEADER_KEY
        ),
        None,
    )
    supplemental = [
        item
        for item in header_values
        if item["header_key"] in _SUPPLEMENTAL_FILTER_HEADER_KEYS
    ]
    primary_category = _filter_value_category(
        str((primary or {}).get("normalized_filter") or "")
    )

    if primary and primary_category != "unknown":
        selected = [dict(primary)]
        ignored = [
            {**dict(item), "ignored_reason": "authoritative_filter_selected"}
            for item in supplemental
        ]
        selection_source = "authoritative_filter"
    else:
        selected = [
            dict(item)
            for item in supplemental
            if _filter_value_category(item["normalized_filter"]) != "unknown"
        ]
        ignored = []
        if primary:
            ignored.append(
                {**dict(primary), "ignored_reason": "unrecognized_primary_filter"}
            )
        ignored.extend(
            {
                **dict(item),
                "ignored_reason": "unrecognized_supplemental_filter",
            }
            for item in supplemental
            if _filter_value_category(item["normalized_filter"]) == "unknown"
        )
        selection_source = "supplemental_fallback" if selected else "none"

    categories = [
        _filter_value_category(item["normalized_filter"]) for item in selected
    ]
    return {
        "filter_field_policy": "filter_authoritative_v1",
        "selection_source": selection_source,
        "primary_filter_header": dict(primary) if primary else None,
        "primary_filter_category": primary_category,
        "supplemental_filter_headers": [dict(item) for item in supplemental],
        "selected_filter_headers": selected,
        "selected_filter_categories": categories,
        "ignored_filter_headers": ignored,
        "filter_headers": [dict(item) for item in header_values],
    }


def _metadata_for_selected_filter_headers(
    metadata: Optional[Mapping[str, Any]],
    selected: list[Dict[str, str]],
) -> Dict[str, Any]:
    """Build a device-profile view that cannot inspect ignored filter fields."""
    effective = {
        key: value
        for key, value in (metadata or {}).items()
        if str(key).strip().upper() not in _FILTER_HEADER_KEYS
    }
    if selected:
        # Device profiles only need one exact vendor alias. Remapping the
        # frozen selection to FILTER isolates profile matching from ignored
        # secondary headers.
        effective[_PRIMARY_FILTER_HEADER_KEY] = selected[0]["filter_value"]
    return effective


def _mapping_contract(
    *,
    confirmed: bool,
    confidence: float,
    evidence: str,
    evidence_detail: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema": NARROWBAND_CHANNEL_MAPPING_SCHEMA,
        "mapping": "osc_hoo_rgb" if confirmed else "unknown",
        "ha_channel": "R" if confirmed else None,
        "oiii_channels": ["G", "B"] if confirmed else [],
        "confidence": float(confidence if confirmed else 0.0),
        "evidence": evidence,
        "evidence_detail": dict(evidence_detail or {}),
        # Retained for readers of the pre-contract diagnostic shape.
        "reason": evidence,
    }


def unknown_narrowband_channel_mapping(
    evidence: str = "ha_oiii_mapping_unconfirmed",
    *,
    evidence_detail: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a complete fail-closed mapping contract."""
    return _mapping_contract(
        confirmed=False,
        confidence=0.0,
        evidence=str(evidence or "ha_oiii_mapping_unconfirmed"),
        evidence_detail=evidence_detail,
    )


def resolve_dual_narrowband_mapping(
    metadata: Optional[Mapping[str, Any]],
    *,
    filter_hint: str = "",
) -> Dict[str, Any]:
    """Resolve one auditable Ha/OIII OSC mapping from filter evidence."""
    selection = select_filter_header_evidence(metadata)
    selected_headers = list(selection["selected_filter_headers"])
    header_text = " ".join(
        item["normalized_filter"] for item in selected_headers
    )
    explicit_hint = _normalize_filter_text(filter_hint)
    detail: Dict[str, Any] = dict(selection)

    wide_path_reason = smart_device_wide_path_reason(metadata)
    if wide_path_reason:
        detail["wide_path_reason"] = wide_path_reason
        return unknown_narrowband_channel_mapping(
            "wide_path_not_supported",
            evidence_detail=detail,
        )

    categories = list(selection["selected_filter_categories"])
    narrow_categories = {
        "conflicting_lines",
        "explicit_hoo",
        "verified_filter",
        "single_line",
        "authoritative_lp",
        "generic_dualband",
    }
    if "conflicting_lines" in categories:
        detail["conflict"] = "non_hoo_or_multiband_line_identity"
        return unknown_narrowband_channel_mapping(
            "conflicting_filter_lines",
            evidence_detail=detail,
        )

    if "broadband" in categories and any(
        category in narrow_categories for category in categories
    ):
        detail["conflict"] = "broadband_and_narrowband_supplemental_fields"
        return unknown_narrowband_channel_mapping(
            "conflicting_filter_fields",
            evidence_detail=detail,
        )

    if "broadband" in categories:
        return unknown_narrowband_channel_mapping(
            "explicit_broadband_filter",
            evidence_detail=detail,
        )

    has_ha = _contains_ha(header_text)
    has_oiii = _contains_oiii(header_text)
    has_sii = _contains_sii(header_text)

    device_match = resolve_smart_device_narrowband_filter(
        _metadata_for_selected_filter_headers(metadata, selected_headers)
    )
    if device_match:
        selected_header = selected_headers[0] if selected_headers else {}
        detail.update(
            {
                "device_profile_id": device_match.get("profile_id"),
                "instrument": device_match.get("instrument"),
                "header_key": selected_header.get("header_key"),
                "filter_value": selected_header.get("filter_value"),
                "normalized_filter": selected_header.get("normalized_filter"),
            }
        )
        return _mapping_contract(
            confirmed=True,
            confidence=0.97,
            evidence="verified_device_profile",
            evidence_detail=detail,
        )

    if has_ha and has_oiii:
        return _mapping_contract(
            confirmed=True,
            confidence=0.95,
            evidence="explicit_filter_lines",
            evidence_detail=detail,
        )
    if _contains_phrase(header_text, ("hoo",)):
        return _mapping_contract(
            confirmed=True,
            confidence=0.95,
            evidence="explicit_filter_lines",
            evidence_detail=detail,
        )

    for profile_id, aliases in _VERIFIED_FILTER_PROFILES:
        if _contains_phrase(header_text, aliases):
            detail["filter_profile_id"] = profile_id
            return _mapping_contract(
                confirmed=True,
                confidence=0.93,
                evidence="verified_filter_profile",
                evidence_detail=detail,
            )

    if has_ha or has_oiii:
        detail["unresolved_lines"] = {
            "ha": has_ha,
            "oiii": has_oiii,
            "sii": has_sii,
        }
        return unknown_narrowband_channel_mapping(
            "single_or_unresolved_narrowband",
            evidence_detail=detail,
        )

    if "authoritative_lp" in categories:
        return _mapping_contract(
            confirmed=True,
            confidence=0.86,
            evidence="authoritative_filter_field_hint",
            evidence_detail=detail,
        )

    if "generic_dualband" in categories:
        return _mapping_contract(
            confirmed=True,
            confidence=0.86,
            evidence="generic_dualband_hint",
            evidence_detail=detail,
        )

    # The user hint is a gap-filler, not permission to contradict a clear
    # broadband or non-HOO FILTER value.
    hint_has_ha_oiii = _contains_ha(explicit_hint) and _contains_oiii(explicit_hint)
    hint_has_dual = _contains_phrase(explicit_hint, _GENERIC_DUALBAND_PHRASES)
    if explicit_hint and (hint_has_ha_oiii or hint_has_dual):
        detail["explicit_filter_hint"] = explicit_hint
        return _mapping_contract(
            confirmed=True,
            confidence=0.99,
            evidence="explicit_user_hint",
            evidence_detail=detail,
        )

    return unknown_narrowband_channel_mapping(
        "ha_oiii_mapping_unconfirmed",
        evidence_detail=detail,
    )


def validate_narrowband_channel_mapping(
    mapping: Optional[Mapping[str, Any]],
    *,
    confidence_min: float = 0.85,
) -> Dict[str, Any]:
    """Validate the frozen contract before a pixel operation consumes it."""
    value = mapping if isinstance(mapping, Mapping) else {}
    issues: list[str] = []
    if str(value.get("schema") or "") != NARROWBAND_CHANNEL_MAPPING_SCHEMA:
        issues.append("channel_mapping_schema_invalid")
    if str(value.get("mapping") or "") != "osc_hoo_rgb":
        issues.append("channel_mapping_kind_invalid")
    if str(value.get("ha_channel") or "") != "R":
        issues.append("ha_channel_not_red")
    oiii_channels = value.get("oiii_channels")
    if not isinstance(oiii_channels, (list, tuple)) or list(oiii_channels) != [
        "G",
        "B",
    ]:
        issues.append("oiii_channels_not_green_blue")
    try:
        confidence = float(value.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
        issues.append("channel_mapping_confidence_nonfinite")
    minimum = max(0.70, min(0.99, float(confidence_min)))
    if confidence < minimum:
        issues.append("ha_oiii_mapping_unconfirmed")
    return {
        "valid": not issues,
        "issues": issues,
        "mapping_confidence": confidence,
        "mapping_confidence_min": minimum,
    }


__all__ = [
    "NARROWBAND_CHANNEL_MAPPING_SCHEMA",
    "resolve_dual_narrowband_mapping",
    "select_filter_header_evidence",
    "unknown_narrowband_channel_mapping",
    "validate_narrowband_channel_mapping",
]
