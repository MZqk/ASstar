"""Serializable observer-only rendition used by review-only outputs."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


SCHEMA = "starun.display-rendition-contract.v2"
LEGACY_SCHEMA = "starun.display-rendition-contract.v1"
LEGACY_CONTRACT_NAME = "linked_review_bright_v1"
CONTRACT_NAME = "linked_review_visibility_v2"
PRESERVE_CONTRACT_NAME = "preserve_accepted_nonlinear_v1"
BLACK_PERCENTILE = 0.002
WHITE_PERCENTILE = 0.995
TARGET_MEDIAN = 0.18
GAMMA_MIN = 0.20
GAMMA_MAX = 1.00
LEGACY_LUMINANCE_GAMMA = 0.5
REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


def _as_rgb_chw(image: Any) -> Tuple[np.ndarray, str]:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("linked Review rendition requires an RGB image")
    if source.shape[0] == 3:
        rgb = source
        layout = "chw"
    elif source.shape[-1] == 3:
        rgb = np.moveaxis(source, -1, 0)
        layout = "hwc"
    else:
        raise ValueError(f"expected RGB input, got shape={source.shape}")
    original_dtype = source.dtype
    rgb = np.asarray(rgb, dtype=np.float64)
    if np.issubdtype(original_dtype, np.integer):
        rgb = rgb / max(float(np.iinfo(original_dtype).max), 1.0)
    if rgb.size == 0 or not np.all(np.isfinite(rgb)):
        raise ValueError("linked Review rendition received invalid pixels")
    return rgb, layout


def _restore_layout(rgb: np.ndarray, layout: str) -> np.ndarray:
    result = np.asarray(rgb, dtype=np.float32)
    return np.moveaxis(result, 0, -1) if layout == "hwc" else result


def build_linked_review_contract(
    image: Any,
    *,
    reason: str,
    source_stem: str,
    input_visibility: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze one bounded linked mapping for an underexposed Review source."""
    rgb, _layout = _as_rgb_chw(image)
    positive = np.clip(rgb, 0.0, None)
    luminance = np.tensordot(REC709, positive, axes=(0, 0))
    black = float(np.quantile(luminance, BLACK_PERCENTILE))
    white = float(np.quantile(luminance, WHITE_PERCENTILE))
    span = white - black
    if (
        not np.isfinite(black)
        or not np.isfinite(white)
        or not np.isfinite(span)
        or span <= 1e-8
    ):
        raise ValueError("linked Review black/white span is unavailable")
    normalized_luminance = np.clip((luminance - black) / span, 0.0, 1.0)
    normalized_median = float(np.median(normalized_luminance))
    if not 1e-8 < normalized_median < 1.0:
        raise ValueError("linked Review normalized median is unavailable")
    gamma = float(
        np.clip(
            np.log(TARGET_MEDIAN) / np.log(normalized_median),
            GAMMA_MIN,
            GAMMA_MAX,
        )
    )
    return {
        "schema": SCHEMA,
        "status": "ready",
        "applicable": True,
        "observer_only": True,
        "mode": "linked_visibility_v2",
        "name": CONTRACT_NAME,
        "reason": str(reason),
        "source_stem": str(source_stem),
        "input_exposure_state": str(
            (input_visibility or {}).get("exposure_state") or "underexposed"
        ),
        "derivative_pixels_changed": True,
        "source_shape_chw": [int(value) for value in rgb.shape],
        "source_visibility": dict(input_visibility or {}),
        "luminance": {
            "space": "Rec.709",
            "weights": [float(value) for value in REC709],
            "black_percentile": BLACK_PERCENTILE,
            "black_point": black,
            "white_percentile": WHITE_PERCENTILE,
            "white_point": white,
            "span": span,
            "normalized_median": normalized_median,
            "target_median": TARGET_MEDIAN,
            "mapping": "pow(clip((Y-black)/(white-black),0,1),gamma)",
            "gamma": gamma,
            "actual_gamma": gamma,
            "gamma_bounds": [GAMMA_MIN, GAMMA_MAX],
        },
        "rgb_mapping": {
            "gain": "mapped_luminance / source_luminance",
            "linked_channels": True,
            "gamut_policy": "shared_per_pixel_scale",
            "preserves_rgb_direction": True,
            "source_pixels_changed": False,
            "derivative_pixels_changed": True,
        },
        "applies_to": [
            "stage7_review_preview",
            "stage9_review_preview",
            "stage10_review_preview",
            "review_bundle_after",
            "result_review.png",
            "managed_display_srgb.png",
        ],
        "excluded_from": ["fit", "tiff", "normal_delivery"],
    }


def build_preserve_review_contract(
    image: Any,
    *,
    reason: str,
    source_stem: str,
    input_visibility: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze an identity rendition for an already-visible nonlinear source."""
    rgb, _layout = _as_rgb_chw(image)
    return {
        "schema": SCHEMA,
        "status": "ready",
        "applicable": True,
        "observer_only": True,
        "mode": "preserve",
        "name": PRESERVE_CONTRACT_NAME,
        "reason": str(reason),
        "source_stem": str(source_stem),
        "input_exposure_state": str(
            (input_visibility or {}).get("exposure_state") or "acceptable"
        ),
        "derivative_pixels_changed": False,
        "source_shape_chw": [int(value) for value in rgb.shape],
        "source_visibility": dict(input_visibility or {}),
        "rgb_mapping": {
            "gain": "identity",
            "linked_channels": True,
            "gamut_policy": "clip_display_domain_only",
            "preserves_rgb_direction": True,
            "source_pixels_changed": False,
            "derivative_pixels_changed": False,
        },
        "applies_to": [
            "stage7_review_preview",
            "stage9_review_preview",
            "stage10_review_preview",
            "review_bundle_after",
            "result_review.png",
            "managed_display_srgb.png",
        ],
        "excluded_from": ["fit", "tiff", "normal_delivery"],
    }


def unavailable_contract(
    *,
    reason: str,
    error: str,
    input_exposure_state: str = "unmappable",
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "unavailable",
        "applicable": True,
        "observer_only": True,
        "mode": "unavailable",
        "name": CONTRACT_NAME,
        "reason": str(reason),
        "input_exposure_state": str(input_exposure_state or "unmappable"),
        "derivative_pixels_changed": False,
        "error": str(error),
    }


def build_review_contract(
    image: Any,
    *,
    reason: str,
    source_stem: str,
    input_visibility: Dict[str, Any],
) -> Dict[str, Any]:
    """Select identity or one linked mapping from a frozen visibility audit."""
    visibility = dict(input_visibility or {})
    exposure_state = str(visibility.get("exposure_state") or "unmappable")
    if bool(visibility.get("passed", False)) and exposure_state == "acceptable":
        return build_preserve_review_contract(
            image,
            reason=reason,
            source_stem=source_stem,
            input_visibility=visibility,
        )
    if exposure_state == "underexposed":
        return build_linked_review_contract(
            image,
            reason=reason,
            source_stem=source_stem,
            input_visibility=visibility,
        )
    return unavailable_contract(
        reason=reason,
        input_exposure_state=exposure_state,
        error=f"review source is not safely mappable: {exposure_state}",
    )


def _validate_legacy_linked_contract(contract: Dict[str, Any]) -> bool:
    luminance = contract.get("luminance")
    rgb_mapping = contract.get("rgb_mapping")
    try:
        return bool(
            contract.get("schema") == LEGACY_SCHEMA
            and contract.get("status") == "ready"
            and contract.get("applicable") is True
            and contract.get("name") == LEGACY_CONTRACT_NAME
            and isinstance(luminance, dict)
            and abs(float(luminance.get("white_percentile")) - WHITE_PERCENTILE)
            <= 1e-12
            and abs(float(luminance.get("gamma")) - LEGACY_LUMINANCE_GAMMA)
            <= 1e-12
            and float(luminance.get("white_point")) > 1e-8
            and isinstance(rgb_mapping, dict)
            and rgb_mapping.get("linked_channels") is True
            and rgb_mapping.get("gamut_policy") == "shared_per_pixel_scale"
        )
    except (TypeError, ValueError):
        return False


def validate_review_contract(contract: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(contract, dict):
        return False
    if _validate_legacy_linked_contract(contract):
        return True
    luminance = contract.get("luminance")
    rgb_mapping = contract.get("rgb_mapping")
    try:
        common = bool(
            contract.get("schema") == SCHEMA
            and contract.get("status") == "ready"
            and contract.get("applicable") is True
            and isinstance(rgb_mapping, dict)
            and rgb_mapping.get("linked_channels") is True
        )
        if not common:
            return False
        if contract.get("mode") == "preserve":
            return bool(
                contract.get("name") == PRESERVE_CONTRACT_NAME
                and rgb_mapping.get("gain") == "identity"
                and rgb_mapping.get("derivative_pixels_changed") is False
            )
        return bool(
            contract.get("mode") == "linked_visibility_v2"
            and contract.get("name") == CONTRACT_NAME
            and isinstance(luminance, dict)
            and abs(float(luminance.get("black_percentile")) - BLACK_PERCENTILE)
            <= 1e-12
            and abs(float(luminance.get("white_percentile")) - WHITE_PERCENTILE)
            <= 1e-12
            and abs(float(luminance.get("target_median")) - TARGET_MEDIAN)
            <= 1e-12
            and GAMMA_MIN <= float(luminance.get("gamma")) <= GAMMA_MAX
            and float(luminance.get("white_point"))
            > float(luminance.get("black_point"))
            and rgb_mapping.get("gamut_policy") == "shared_per_pixel_scale"
        )
    except (TypeError, ValueError):
        return False


def validate_linked_review_contract(contract: Optional[Dict[str, Any]]) -> bool:
    """Backward-compatible validator for linked (non-identity) contracts."""
    return bool(
        validate_review_contract(contract)
        and isinstance(contract, dict)
        and contract.get("name") in {CONTRACT_NAME, LEGACY_CONTRACT_NAME}
    )


def apply_review_contract(
    image: Any,
    contract: Dict[str, Any],
) -> np.ndarray:
    """Replay a frozen contract without changing the source pixel buffer."""
    if not validate_review_contract(contract):
        raise ValueError("invalid Review rendition contract")
    rgb, layout = _as_rgb_chw(image)
    positive = np.clip(rgb, 0.0, None)
    if contract.get("mode") == "preserve":
        return _restore_layout(np.clip(positive, 0.0, 1.0), layout)
    luminance = np.tensordot(REC709, positive, axes=(0, 0))
    if contract.get("schema") == LEGACY_SCHEMA:
        white = float(contract["luminance"]["white_point"])
        mapped_luminance = np.sqrt(np.clip(luminance / white, 0.0, 1.0))
    else:
        black = float(contract["luminance"]["black_point"])
        span = float(contract["luminance"]["span"])
        gamma = float(contract["luminance"]["gamma"])
        normalized = np.clip((luminance - black) / span, 0.0, 1.0)
        mapped_luminance = np.power(normalized, gamma)
    gain = np.divide(
        mapped_luminance,
        luminance,
        out=np.zeros_like(mapped_luminance),
        where=luminance > 1e-12,
    )
    rendered = positive * gain[np.newaxis, ...]
    peak = np.max(rendered, axis=0)
    gamut_scale = np.maximum(peak, 1.0)
    rendered = rendered / gamut_scale[np.newaxis, ...]
    if not np.all(np.isfinite(rendered)):
        raise ValueError("linked Review rendition produced nonfinite pixels")
    return _restore_layout(np.clip(rendered, 0.0, 1.0), layout)


def apply_linked_review_contract(
    image: Any,
    contract: Dict[str, Any],
) -> np.ndarray:
    """Backward-compatible entry point used by existing preview consumers."""
    return apply_review_contract(image, contract)


__all__ = [
    "CONTRACT_NAME",
    "PRESERVE_CONTRACT_NAME",
    "SCHEMA",
    "apply_linked_review_contract",
    "apply_review_contract",
    "build_linked_review_contract",
    "build_preserve_review_contract",
    "build_review_contract",
    "unavailable_contract",
    "validate_linked_review_contract",
    "validate_review_contract",
]
