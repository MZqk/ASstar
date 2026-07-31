"""Report-only device and optical geometry resolution.

The resolver deliberately does not alter plate-solving arguments.  It records
the evidence and a conservative recommendation so a later release can enable
automatic geometry only after real-image validation.
"""
from __future__ import annotations

import math
import os
import re
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Sequence


GEOMETRY_SCHEMA = "seestar.device-geometry.v1"
DEFAULT_FOCAL_LENGTH_MM = 160.0
DEFAULT_PIXEL_SIZE_UM = 2.9
AUTO_APPLY_CONFIDENCE_MIN = 0.85
CONFLICT_RELATIVE_TOLERANCE = 0.05

_FOCAL_KEYS = (
    "FOCALLEN",
    "FOCALLENGTH",
    "FOCAL",
    "EFL",
)
_PIXEL_X_KEYS = (
    "XPIXSZ",
    "XPIXSIZE",
    "PIXSIZE1",
)
_PIXEL_Y_KEYS = (
    "YPIXSZ",
    "YPIXSIZE",
    "PIXSIZE2",
)
_PIXEL_GENERIC_KEYS = (
    "PIXSIZE",
    "PIXELSIZ",
    "PIXELSIZE",
)
_BIN_X_KEYS = ("XBINNING", "XBIN", "BINX")
_BIN_Y_KEYS = ("YBINNING", "YBIN", "BINY")
_IDENTITY_KEYS = (
    "INSTRUME",
    "INSTRUMENT",
    "CAMERA",
    "DETECTOR",
    "SENSOR",
    "TELESCOP",
    "TELESCOPE",
)

_KNOWN_PROFILES = (
    {
        "id": "seestar_s30_pro_imx585",
        "match_all": ("s30", "pro"),
        "instrument": "Seestar S30 Pro",
        "sensor": "Sony IMX585",
        "focal_length_mm": 160.0,
        "pixel_size_um": 2.9,
        "confidence": 0.88,
    },
)


def _unpack_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("value", "raw", "text"):
            if key in value:
                return value[key]
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    return value


def _metadata_index(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key).strip().upper(): _unpack_metadata_value(value)
        for key, value in metadata.items()
        if not str(key).startswith("_")
    }


def _finite_float(value: Any) -> Optional[float]:
    value = _unpack_metadata_value(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            str(value or ""),
        )
        if not match:
            return None
        try:
            number = float(match.group(0))
        except ValueError:
            return None
    if not math.isfinite(number):
        return None
    return number


def _positive_float(value: Any) -> Optional[float]:
    number = _finite_float(value)
    return number if number is not None and number > 0.0 else None


def _first_value(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[Optional[float], Optional[str]]:
    for key in keys:
        value = _positive_float(metadata.get(key))
        if value is not None:
            return value, key
    return None, None


def _candidate(
    field: str,
    value: float,
    *,
    source: str,
    confidence: float,
    priority: int,
    evidence: str,
) -> Dict[str, Any]:
    return {
        "field": field,
        "value": float(value),
        "source": source,
        "confidence": float(confidence),
        "priority": int(priority),
        "evidence": evidence,
    }


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-9)


def _shape_value(image_shape: Optional[Mapping[str, Any]], name: str) -> int:
    if not image_shape:
        return 0
    try:
        return max(0, int(image_shape.get(name, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _matching_profile(identity_text: str) -> Optional[Dict[str, Any]]:
    lowered = identity_text.casefold()
    for profile in _KNOWN_PROFILES:
        if all(token in lowered for token in profile["match_all"]):
            return dict(profile)
    return None


def _select(candidates: list[Dict[str, Any]], field: str) -> Dict[str, Any]:
    matches = [item for item in candidates if item["field"] == field]
    return dict(max(matches, key=lambda item: item["priority"]))


def build_device_geometry_report(
    metadata: Optional[Mapping[str, Any]],
    *,
    image_shape: Optional[Mapping[str, Any]] = None,
    crop_report: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve geometry evidence without applying it to the running pipeline."""
    indexed = _metadata_index(metadata or {})
    environment = environ if environ is not None else os.environ
    candidates: list[Dict[str, Any]] = []
    conflicts: list[Dict[str, Any]] = []

    focal, focal_key = _first_value(indexed, _FOCAL_KEYS)
    if focal is not None and focal_key:
        candidates.append(
            _candidate(
                "focal_length_mm",
                focal,
                source="fits_header",
                confidence=0.96,
                priority=100,
                evidence=focal_key,
            )
        )

    pixel_x, pixel_x_key = _first_value(indexed, _PIXEL_X_KEYS)
    pixel_y, pixel_y_key = _first_value(indexed, _PIXEL_Y_KEYS)
    pixel_generic, pixel_generic_key = _first_value(indexed, _PIXEL_GENERIC_KEYS)
    pixel_values = [
        value for value in (pixel_x, pixel_y, pixel_generic) if value is not None
    ]
    if pixel_values:
        if (
            pixel_x is not None
            and pixel_y is not None
            and _relative_difference(pixel_x, pixel_y)
            > CONFLICT_RELATIVE_TOLERANCE
        ):
            conflicts.append(
                {
                    "field": "pixel_size_um",
                    "reason": "x_y_pixel_size_mismatch",
                    "values": {
                        str(pixel_x_key): pixel_x,
                        str(pixel_y_key): pixel_y,
                    },
                }
            )
        pixel_value = float(sum(pixel_values) / len(pixel_values))
        pixel_evidence = ",".join(
            key
            for key in (pixel_x_key, pixel_y_key, pixel_generic_key)
            if key
        )
        candidates.append(
            _candidate(
                "pixel_size_um",
                pixel_value,
                source="fits_header",
                confidence=0.95,
                priority=100,
                evidence=pixel_evidence,
            )
        )

    identity_values = [
        str(indexed[key]).strip()
        for key in _IDENTITY_KEYS
        if key in indexed and str(indexed[key]).strip()
    ]
    identity_text = " | ".join(identity_values)
    profile = _matching_profile(identity_text)
    if profile:
        candidates.extend(
            (
                _candidate(
                    "focal_length_mm",
                    profile["focal_length_mm"],
                    source="known_device_profile",
                    confidence=profile["confidence"],
                    priority=80,
                    evidence=profile["id"],
                ),
                _candidate(
                    "pixel_size_um",
                    profile["pixel_size_um"],
                    source="known_device_profile",
                    confidence=profile["confidence"],
                    priority=80,
                    evidence=profile["id"],
                ),
            )
        )

    env_focal = _positive_float(environment.get("SEESTAR_STAGE4_PLATESOLVE_FOCAL"))
    env_pixel = _positive_float(environment.get("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE"))
    if env_focal is not None:
        candidates.append(
            _candidate(
                "focal_length_mm",
                env_focal,
                source="environment_override",
                confidence=0.60,
                priority=60,
                evidence="SEESTAR_STAGE4_PLATESOLVE_FOCAL",
            )
        )
    if env_pixel is not None:
        candidates.append(
            _candidate(
                "pixel_size_um",
                env_pixel,
                source="environment_override",
                confidence=0.60,
                priority=60,
                evidence="SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE",
            )
        )

    candidates.extend(
        (
            _candidate(
                "focal_length_mm",
                DEFAULT_FOCAL_LENGTH_MM,
                source="legacy_default",
                confidence=0.25,
                priority=10,
                evidence="current Stage4 fallback",
            ),
            _candidate(
                "pixel_size_um",
                DEFAULT_PIXEL_SIZE_UM,
                source="legacy_default",
                confidence=0.25,
                priority=10,
                evidence="current Stage4 fallback",
            ),
        )
    )

    selected_focal = _select(candidates, "focal_length_mm")
    selected_pixel = _select(candidates, "pixel_size_um")
    for selected, field in (
        (selected_focal, "focal_length_mm"),
        (selected_pixel, "pixel_size_um"),
    ):
        for other in candidates:
            if (
                other["field"] == field
                and other["source"] in {"fits_header", "known_device_profile"}
                and selected["source"] in {"fits_header", "known_device_profile"}
                and other["source"] != selected["source"]
                and _relative_difference(selected["value"], other["value"])
                > CONFLICT_RELATIVE_TOLERANCE
            ):
                conflicts.append(
                    {
                        "field": field,
                        "reason": "header_profile_mismatch",
                        "selected": selected,
                        "other": other,
                    }
                )

    bin_x, bin_x_key = _first_value(indexed, _BIN_X_KEYS)
    bin_y, bin_y_key = _first_value(indexed, _BIN_Y_KEYS)
    binning_x = max(1, int(round(bin_x or 1.0)))
    binning_y = max(1, int(round(bin_y or bin_x or 1.0)))
    if binning_x != binning_y:
        conflicts.append(
            {
                "field": "binning",
                "reason": "asymmetric_binning",
                "values": {
                    str(bin_x_key or "X"): binning_x,
                    str(bin_y_key or "Y"): binning_y,
                },
            }
        )

    effective_pixel_x = selected_pixel["value"] * binning_x
    effective_pixel_y = selected_pixel["value"] * binning_y
    plate_scale_x = 206.265 * effective_pixel_x / selected_focal["value"]
    plate_scale_y = 206.265 * effective_pixel_y / selected_focal["value"]
    width = _shape_value(image_shape, "width")
    height = _shape_value(image_shape, "height")
    confidence = min(
        float(selected_focal["confidence"]),
        float(selected_pixel["confidence"]),
    )
    would_auto_apply = bool(
        confidence >= AUTO_APPLY_CONFIDENCE_MIN and not conflicts
    )

    return {
        "schema": GEOMETRY_SCHEMA,
        "mode": "report_only",
        "applied": False,
        "identity": {
            "metadata_text": identity_text or None,
            "matched_profile": profile,
        },
        "selected": {
            "focal_length_mm": selected_focal,
            "pixel_size_um": selected_pixel,
            "binning": {
                "x": binning_x,
                "y": binning_y,
                "source": "fits_header" if bin_x or bin_y else "implicit_1x1",
            },
            "effective_pixel_size_um": {
                "x": effective_pixel_x,
                "y": effective_pixel_y,
            },
            "predicted_plate_scale_arcsec_per_pixel": {
                "x": plate_scale_x,
                "y": plate_scale_y,
            },
            "predicted_field_of_view_degrees": {
                "width": plate_scale_x * width / 3600.0 if width else None,
                "height": plate_scale_y * height / 3600.0 if height else None,
            },
        },
        "current_runtime": {
            "focal_length_mm": env_focal or DEFAULT_FOCAL_LENGTH_MM,
            "pixel_size_um": env_pixel or DEFAULT_PIXEL_SIZE_UM,
            "source": (
                "environment_override"
                if env_focal is not None or env_pixel is not None
                else "legacy_default"
            ),
            "unchanged_by_report": True,
        },
        "decision": {
            "confidence": confidence,
            "minimum_confidence_for_future_auto_apply": AUTO_APPLY_CONFIDENCE_MIN,
            "would_auto_apply": would_auto_apply,
            "reason": (
                "high_confidence_no_conflicts"
                if would_auto_apply
                else "conflicting_geometry_evidence"
                if conflicts
                else "insufficient_confidence"
            ),
        },
        "image_shape": dict(image_shape or {}),
        "stage2_crop": dict(crop_report or {}),
        "conflicts": conflicts,
        "candidates": candidates,
    }


def activate_device_geometry_report(
    report: Mapping[str, Any],
    *,
    enabled: bool,
    confidence_min: float = AUTO_APPLY_CONFIDENCE_MIN,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Select runtime geometry while preserving explicit manual overrides."""
    result = deepcopy(dict(report))
    environment = environ if environ is not None else os.environ
    explicit_override = bool(
        str(environment.get("SEESTAR_STAGE4_PLATESOLVE_FOCAL", "") or "").strip()
        or str(
            environment.get("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE", "") or ""
        ).strip()
    )
    decision = result.get("decision") or {}
    selected = result.get("selected") or {}
    confidence = float(decision.get("confidence", 0.0) or 0.0)
    conflicts = list(result.get("conflicts") or [])
    current = result.get("current_runtime") or {}

    applied = bool(
        enabled
        and not explicit_override
        and not conflicts
        and confidence >= max(0.0, min(1.0, float(confidence_min)))
    )
    if applied:
        focal = float(
            (selected.get("focal_length_mm") or {}).get(
                "value",
                DEFAULT_FOCAL_LENGTH_MM,
            )
        )
        effective_pixel = selected.get("effective_pixel_size_um") or {}
        pixel = float(
            effective_pixel.get(
                "x",
                (selected.get("pixel_size_um") or {}).get(
                    "value",
                    DEFAULT_PIXEL_SIZE_UM,
                ),
            )
        )
        source = "auto_geometry"
        reason = "high_confidence_no_conflicts"
    else:
        focal = float(current.get("focal_length_mm", DEFAULT_FOCAL_LENGTH_MM))
        pixel = float(current.get("pixel_size_um", DEFAULT_PIXEL_SIZE_UM))
        source = str(current.get("source") or "legacy_default")
        reason = (
            "disabled"
            if not enabled
            else "explicit_environment_override"
            if explicit_override
            else "conflicting_geometry_evidence"
            if conflicts
            else "insufficient_confidence"
        )

    result["mode"] = "active_guarded"
    result["applied"] = applied
    result["activation"] = {
        "enabled": bool(enabled),
        "applied": applied,
        "source": source,
        "reason": reason,
        "confidence": confidence,
        "confidence_min": max(0.0, min(1.0, float(confidence_min))),
        "explicit_environment_override": explicit_override,
        "runtime_geometry": {
            "focal_length_mm": focal,
            "pixel_size_um": pixel,
        },
        "validation": {
            "status": "pending" if applied else "not_required",
        },
    }
    return result


def _wcs_pixel_scale_arcsec(metadata: Mapping[str, Any]) -> Optional[float]:
    indexed = _metadata_index(metadata)
    direct, _direct_key = _first_value(
        indexed,
        ("SECPIX", "PIXSCALE", "PIXSCAL1"),
    )
    if direct is not None:
        return direct

    cd11 = _finite_float(indexed.get("CD1_1"))
    cd12 = _finite_float(indexed.get("CD1_2"))
    cd21 = _finite_float(indexed.get("CD2_1"))
    cd22 = _finite_float(indexed.get("CD2_2"))
    scales: list[float] = []
    if cd11 is not None or cd21 is not None:
        scales.append(math.hypot(cd11 or 0.0, cd21 or 0.0) * 3600.0)
    if cd12 is not None or cd22 is not None:
        scales.append(math.hypot(cd12 or 0.0, cd22 or 0.0) * 3600.0)
    if scales:
        return float(sum(scales) / len(scales))

    cdelt1 = _finite_float(indexed.get("CDELT1"))
    cdelt2 = _finite_float(indexed.get("CDELT2"))
    scales = [
        abs(value) * 3600.0
        for value in (cdelt1, cdelt2)
        if value is not None and value != 0.0
    ]
    return float(sum(scales) / len(scales)) if scales else None


def validate_active_geometry(
    report: Mapping[str, Any],
    solved_metadata: Optional[Mapping[str, Any]],
    *,
    residual_max: float = CONFLICT_RELATIVE_TOLERANCE,
) -> Dict[str, Any]:
    """Validate activated geometry against solved WCS pixel scale."""
    result = deepcopy(dict(report))
    activation = dict(result.get("activation") or {})
    if not bool(activation.get("applied")):
        activation["validation"] = {"status": "not_required", "accepted": True}
        result["activation"] = activation
        return result

    actual = _wcs_pixel_scale_arcsec(solved_metadata or {})
    predicted_values = (
        (result.get("selected") or {}).get(
            "predicted_plate_scale_arcsec_per_pixel",
            {},
        )
        or {}
    )
    predicted_candidates = [
        _positive_float(predicted_values.get(axis)) for axis in ("x", "y")
    ]
    predicted_candidates = [
        value for value in predicted_candidates if value is not None
    ]
    predicted = (
        float(sum(predicted_candidates) / len(predicted_candidates))
        if predicted_candidates
        else None
    )
    limit = max(0.01, min(0.25, float(residual_max)))
    if actual is None or predicted is None:
        validation = {
            "status": "unavailable",
            "accepted": True,
            "reason": "solved_wcs_pixel_scale_unavailable",
            "predicted_arcsec_per_pixel": predicted,
            "actual_arcsec_per_pixel": actual,
            "residual_max": limit,
        }
    else:
        residual = _relative_difference(predicted, actual)
        accepted = residual <= limit
        validation = {
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "reason": (
                "wcs_scale_matches_geometry"
                if accepted
                else "wcs_scale_residual_exceeds_limit"
            ),
            "predicted_arcsec_per_pixel": predicted,
            "actual_arcsec_per_pixel": actual,
            "relative_residual": residual,
            "residual_max": limit,
        }
    activation["validation"] = validation
    result["activation"] = activation
    return result


__all__ = [
    "activate_device_geometry_report",
    "build_device_geometry_report",
    "validate_active_geometry",
]
