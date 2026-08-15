"""Deterministic Ha/OIII normalization for confirmed dual-narrowband RGB data."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional

import numpy as np

from device_geometry import (
    resolve_smart_device_narrowband_filter,
    smart_device_wide_path_reason,
)


NARROWBAND_NORMALIZATION_SCHEMA = "starun.narrowband-normalization.v1"
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


def _as_rgb_float(image: Any) -> tuple[np.ndarray, np.ndarray, str]:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("dual-narrowband normalization requires a 3-channel image")
    if source.shape[0] == 3:
        rgb_source = source
        layout = "chw"
    elif source.shape[-1] == 3:
        rgb_source = np.transpose(source, (2, 0, 1))
        layout = "hwc"
    else:
        raise ValueError(f"expected RGB input, got shape={source.shape}")
    rgb = rgb_source.astype(np.float32, copy=True)
    if np.issubdtype(source.dtype, np.integer):
        rgb /= max(1.0, float(np.iinfo(source.dtype).max))
    if not np.all(np.isfinite(rgb)):
        raise ValueError("nonfinite input pixels")
    return source, rgb, layout


def _restore(source: np.ndarray, rgb: np.ndarray, layout: str) -> np.ndarray:
    output = rgb if layout == "chw" else np.transpose(rgb, (1, 2, 0))
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        return np.clip(output * maximum, 0.0, maximum).astype(source.dtype)
    return output.astype(np.float32, copy=False)


def _blur3(plane: np.ndarray) -> np.ndarray:
    padded = np.pad(plane, ((1, 1), (1, 1)), mode="reflect")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _mad_sigma(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size < 16:
        return 0.0
    median = float(np.median(data))
    return float(1.4826 * np.median(np.abs(data - median)))


def _normalized_chroma(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    samples = rgb[:, mask]
    total = np.maximum(np.sum(samples, axis=0), 1e-6)
    return samples / total[None, :]


def normalize_dual_narrowband_candidate(
    image: Any,
    *,
    mapping: Mapping[str, Any],
    mapping_confidence_min: float = 0.85,
    strength: float = 0.55,
    gain_limit: float = 1.08,
    line_ratio_drift_max: float = 0.12,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Return a guarded HOO normalization candidate without mutating input."""
    validation = validate_narrowband_channel_mapping(
        mapping,
        confidence_min=mapping_confidence_min,
    )
    if not validation["valid"]:
        raise ValueError(
            "dual-narrowband channel mapping contract is invalid: "
            + ",".join(validation["issues"])
        )
    source, rgb, layout = _as_rgb_float(image)
    original = rgb.copy()
    luma = (
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    ).astype(np.float32)
    finite_values = luma[np.isfinite(luma)]
    if finite_values.size < 256:
        raise ValueError("insufficient image pixels")
    q005, q40, q60, q98 = (
        float(value)
        for value in np.quantile(finite_values, (0.005, 0.40, 0.60, 0.98))
    )
    gradient = np.abs(luma - _blur3(luma))
    gradient_limit = float(np.quantile(gradient, 0.70))
    background = (
        (luma >= q005)
        & (luma <= q40)
        & (gradient <= gradient_limit)
    )
    signal = (luma >= q60) & (luma <= q98)
    if np.count_nonzero(background) < 128 or np.count_nonzero(signal) < 128:
        raise ValueError("insufficient background or signal samples")

    backgrounds = np.median(rgb[:, background], axis=1)
    spans = np.array(
        [
            max(
                float(np.quantile(rgb[channel, signal], 0.90))
                - float(backgrounds[channel]),
                1e-6,
            )
            for channel in range(3)
        ],
        dtype=np.float64,
    )
    safe_strength = max(0.10, min(0.85, float(strength)))
    safe_gain_limit = max(1.01, min(1.15, float(gain_limit)))
    oiii_span = math.sqrt(float(spans[1] * spans[2]))
    gb_balance = math.sqrt(float(spans[2] / spans[1]))
    ha_oiii_balance = math.sqrt(oiii_span / float(spans[0]))
    raw_gains = np.array(
        [
            ha_oiii_balance ** (safe_strength * 0.35),
            gb_balance ** safe_strength,
            (1.0 / gb_balance) ** safe_strength,
        ],
        dtype=np.float64,
    )
    gains = np.clip(
        raw_gains,
        1.0 / safe_gain_limit,
        safe_gain_limit,
    ).astype(np.float32)

    target_background = float(np.dot(backgrounds, [0.2126, 0.7152, 0.0722]))
    normalized = (
        (rgb - backgrounds[:, None, None]) * gains[:, None, None]
        + (
            backgrounds[:, None, None] * (1.0 - safe_strength * 0.65)
            + target_background * (safe_strength * 0.65)
        )
    )
    normalized_luma = (
        0.2126 * normalized[0]
        + 0.7152 * normalized[1]
        + 0.0722 * normalized[2]
    )
    normalized *= (
        luma / np.maximum(normalized_luma, 1e-6)
    )[None, :, :]

    star_detail = np.maximum(luma - _blur3(luma), 0.0)
    star_threshold = max(
        float(np.median(star_detail)) + 8.0 * _mad_sigma(
            star_detail[background]
        ),
        float(np.quantile(star_detail, 0.992)),
    )
    star_peak = float(np.quantile(star_detail, 0.9995))
    star_mask = np.clip(
        (star_detail - star_threshold)
        / max(star_peak - star_threshold, 1e-6),
        0.0,
        1.0,
    )
    star_mask = np.clip(_blur3(star_mask), 0.0, 1.0)
    normalized = (
        normalized * (1.0 - 0.90 * star_mask[None, :, :])
        + original * (0.90 * star_mask[None, :, :])
    )
    candidate = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    before_bg_chroma = _normalized_chroma(original, background)
    after_bg_chroma = _normalized_chroma(candidate, background)
    neutral = np.full((3, before_bg_chroma.shape[1]), 1.0 / 3.0)
    bg_delta_before = float(np.median(np.max(np.abs(before_bg_chroma - neutral), axis=0)))
    bg_delta_after = float(np.median(np.max(np.abs(after_bg_chroma - neutral), axis=0)))

    candidate_backgrounds = np.median(candidate[:, background], axis=1)
    candidate_spans = np.array(
        [
            max(
                float(np.quantile(candidate[channel, signal], 0.90))
                - float(candidate_backgrounds[channel]),
                1e-6,
            )
            for channel in range(3)
        ]
    )
    ratio_before = float(spans[0] / math.sqrt(spans[1] * spans[2]))
    ratio_after = float(
        candidate_spans[0]
        / math.sqrt(candidate_spans[1] * candidate_spans[2])
    )
    line_ratio_drift = abs(ratio_after / max(ratio_before, 1e-6) - 1.0)
    before_star_chroma = _normalized_chroma(original, star_mask >= 0.35)
    after_star_chroma = _normalized_chroma(candidate, star_mask >= 0.35)
    star_chroma_drift = (
        float(np.median(np.max(np.abs(after_star_chroma - before_star_chroma), axis=0)))
        if before_star_chroma.shape[1] >= 16
        else 0.0
    )
    before_clip = float(np.mean((original <= 0.0) | (original >= 1.0)))
    after_clip = float(np.mean((candidate <= 0.0) | (candidate >= 1.0)))
    candidate_luma = (
        0.2126 * candidate[0] + 0.7152 * candidate[1] + 0.0722 * candidate[2]
    )
    luma_drift = float(
        np.quantile(np.abs(candidate_luma - luma), 0.95)
    )
    metrics = {
        "background_color_delta_before": bg_delta_before,
        "background_color_delta_after": bg_delta_after,
        "background_color_improvement": bg_delta_before - bg_delta_after,
        "ha_oiii_ratio_before": ratio_before,
        "ha_oiii_ratio_after": ratio_after,
        "ha_oiii_ratio_drift": line_ratio_drift,
        "star_chroma_drift": star_chroma_drift,
        "luminance_drift_p95": luma_drift,
        "clip_growth": after_clip - before_clip,
        "star_mask_coverage": float(np.mean(star_mask > 0.05)),
    }
    limits = {
        "ha_oiii_ratio_drift_max": max(
            0.04,
            min(0.20, float(line_ratio_drift_max)),
        ),
        "star_chroma_drift_max": 0.10,
        "luminance_drift_p95_max": 0.015,
        "clip_growth_max": 0.001,
        "background_color_worsening_max": 0.005,
        "star_mask_coverage_max": 0.12,
    }
    issues: list[str] = []
    if not np.all(np.isfinite(candidate)):
        issues.append("nonfinite_output")
    if line_ratio_drift > limits["ha_oiii_ratio_drift_max"]:
        issues.append("ha_oiii_ratio_drift")
    if star_chroma_drift > limits["star_chroma_drift_max"]:
        issues.append("star_chroma_drift")
    if luma_drift > limits["luminance_drift_p95_max"]:
        issues.append("luminance_drift")
    if metrics["clip_growth"] > limits["clip_growth_max"]:
        issues.append("clip_growth")
    if (
        bg_delta_after - bg_delta_before
        > limits["background_color_worsening_max"]
    ):
        issues.append("background_color_worsened")
    if metrics["star_mask_coverage"] > limits["star_mask_coverage_max"]:
        issues.append("star_mask_coverage")
    changed = float(np.mean(np.max(np.abs(candidate - original), axis=0) > 1e-4))
    if changed <= 1e-4:
        issues.append("no_effect")
    accepted = not issues
    report = {
        "schema": NARROWBAND_NORMALIZATION_SCHEMA,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "mapping": mapping,
        "algorithm": "luminance_preserving_guarded_hoo_normalization",
        "strength": safe_strength,
        "gains": {
            "red_ha": float(gains[0]),
            "green_oiii": float(gains[1]),
            "blue_oiii": float(gains[2]),
        },
        "metrics": metrics,
        "limits": limits,
        "issues": issues,
        "changed_pixel_ratio": changed,
        "transaction": {
            "baseline": "stage4_pre_nbn.fit",
            "candidate": "stage4_nbn_candidate.fit",
        },
    }
    return _restore(source, candidate, layout), report


__all__ = [
    "NARROWBAND_CHANNEL_MAPPING_SCHEMA",
    "normalize_dual_narrowband_candidate",
    "resolve_dual_narrowband_mapping",
    "select_filter_header_evidence",
    "unknown_narrowband_channel_mapping",
    "validate_narrowband_channel_mapping",
]
