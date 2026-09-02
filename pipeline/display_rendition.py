"""Serializable observer-only rendition used by review-only outputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from .image_metrics import _box_blur_gray
    from .stage8_color_rendition import (
        apply_subject_chroma_rendition,
        assess_subject_chroma_candidate,
        subject_saturation_distribution,
    )
except ImportError:
    from image_metrics import _box_blur_gray
    from stage8_color_rendition import (
        apply_subject_chroma_rendition,
        assess_subject_chroma_candidate,
        subject_saturation_distribution,
    )


SCHEMA = "starun.display-rendition-contract.v2"
V3_SCHEMA = "starun.display-rendition-contract.v3"
LEGACY_SCHEMA = "starun.display-rendition-contract.v1"
LEGACY_CONTRACT_NAME = "linked_review_bright_v1"
CONTRACT_NAME = "linked_review_visibility_v2"
PRESERVE_CONTRACT_NAME = "preserve_accepted_nonlinear_v1"
V3_CONTRACT_NAME = "tone_then_subject_chroma_v1"
MASK_SCHEMA = "starun.display-rendition-masks.v1"
MASK_COORDINATE_DOMAIN = "siril_bottom_up"
PIXEL_DOMAIN_BOTTOM_UP = "siril_bottom_up"
PIXEL_DOMAIN_TOP_DOWN = "display_top_down"
MASK_ARRAY_NAMES = (
    "subject_mask",
    "background_mask",
    "core_mask",
    "star_mask",
    "star_halo_guard_mask",
    "shared_valid_mask",
)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_summary(value: np.ndarray) -> Dict[str, Any]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
        "nonzero_count": int(np.count_nonzero(array)),
    }


def _safe_relative_artifact(root: Path, relative_path: str) -> Path:
    relative = Path(str(relative_path or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("display rendition mask path must be relative and contained")
    root_resolved = root.resolve(strict=True)
    candidate = root / relative
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.exists() and parent.is_symlink():
            raise ValueError("display rendition mask path contains a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("display rendition mask path escapes artifact root") from error
    return candidate


def persist_display_rendition_masks(
    artifact_root: str | Path,
    masks: Dict[str, Any],
    *,
    relative_path: str = "process/display_rendition_masks.npz",
) -> Dict[str, Any]:
    """Freeze the display masks and return a replay-verifiable artifact binding."""

    root = Path(artifact_root)
    target = _safe_relative_artifact(root, relative_path)
    if target.exists() and target.is_symlink():
        raise ValueError("display rendition mask artifact may not be a symlink")
    arrays: Dict[str, np.ndarray] = {}
    spatial_shape: Optional[tuple[int, int]] = None
    for name in MASK_ARRAY_NAMES:
        source_name = name
        value = masks.get(source_name) if isinstance(masks, dict) else None
        if value is None:
            raise ValueError(f"required display rendition mask is missing: {name}")
        array = np.asarray(value)
        if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid display rendition mask: {name}")
        if spatial_shape is None:
            spatial_shape = tuple(int(item) for item in array.shape)
        if tuple(array.shape) != spatial_shape:
            raise ValueError(f"display rendition mask shape mismatch: {name}")
        if name == "shared_valid_mask":
            arrays[name] = (array > 0).astype(np.uint8)
        else:
            arrays[name] = np.clip(array, 0.0, 1.0).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return {
        "schema": MASK_SCHEMA,
        "status": "ready",
        "relative_path": Path(relative_path).as_posix(),
        "file_sha256": _sha256_file(target),
        "coordinate_domain": MASK_COORDINATE_DOMAIN,
        "shape": [int(item) for item in spatial_shape or ()],
        "arrays": {
            name: _array_summary(array) for name, array in arrays.items()
        },
    }


def load_display_rendition_masks(
    artifact_root: str | Path,
    record: Dict[str, Any],
    *,
    pixel_coordinate_domain: str,
) -> Dict[str, np.ndarray]:
    """Load a frozen mask artifact after path, file, array and shape checks."""

    if not isinstance(record, dict) or record.get("schema") != MASK_SCHEMA:
        raise ValueError("invalid display rendition mask record")
    if record.get("status") != "ready":
        raise ValueError("display rendition mask record is not ready")
    root = Path(artifact_root)
    target = _safe_relative_artifact(root, str(record.get("relative_path") or ""))
    if target.is_symlink() or not target.is_file():
        raise ValueError("display rendition mask artifact is unavailable or symlinked")
    if _sha256_file(target) != str(record.get("file_sha256") or ""):
        raise ValueError("display rendition mask artifact SHA mismatch")
    expected_shape = tuple(int(item) for item in (record.get("shape") or ()))
    expected_arrays = record.get("arrays")
    if len(expected_shape) != 2 or not isinstance(expected_arrays, dict):
        raise ValueError("display rendition mask manifest shape is invalid")
    arrays: Dict[str, np.ndarray] = {}
    with np.load(target, allow_pickle=False) as payload:
        if set(payload.files) != set(MASK_ARRAY_NAMES):
            raise ValueError("display rendition mask artifact members changed")
        for name in MASK_ARRAY_NAMES:
            value = np.asarray(payload[name])
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"display rendition mask shape mismatch: {name}")
            expected = expected_arrays.get(name)
            if not isinstance(expected, dict) or _array_summary(value) != expected:
                raise ValueError(f"display rendition mask digest mismatch: {name}")
            arrays[name] = np.array(value, copy=True)
    domain = str(pixel_coordinate_domain or "")
    if domain == PIXEL_DOMAIN_TOP_DOWN:
        arrays = {name: np.flip(value, axis=0).copy() for name, value in arrays.items()}
    elif domain != PIXEL_DOMAIN_BOTTOM_UP:
        raise ValueError("v3 display rendition requires an explicit pixel coordinate domain")
    return arrays


def assess_weak_subject_chroma_source(
    image: Any,
    masks: Dict[str, Any],
) -> Dict[str, Any]:
    """Require spatially coherent low-frequency subject colour before recovery."""

    limits = {
        "support_min": 2048,
        "grid_cells_min": 4,
        "quadrants_min": 3,
        "scale_correlation_min": 0.90,
        "subject_background_energy_ratio_min": 1.50,
        "aggregate_snr_min": 5.0,
    }
    report: Dict[str, Any] = {
        "schema": "starun.weak-subject-chroma-evidence.v1",
        "status": "rejected",
        "accepted": False,
        "limits": limits,
        "issues": [],
    }
    try:
        rgb, _layout = _as_rgb_chw(image)
        height, width = rgb.shape[1:]
        shape = (height, width)

        def mask(name: str) -> np.ndarray:
            value = np.asarray(masks.get(name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid weak-color mask: {name}")
            return np.clip(value, 0.0, 1.0)

        subject = mask("subject_mask") > 0.05
        background = mask("background_mask") >= 0.80
        valid = mask("shared_valid_mask") > 0.5
        core = mask("core_mask") >= 0.50
        stars = mask("star_mask") >= 0.05
        halo = mask("star_halo_guard_mask") >= 0.05
        protected = core | stars | halo | ~valid
        subject &= ~protected
        background &= ~(stars | halo) & valid
        support_count = int(np.count_nonzero(subject))
        background_count = int(np.count_nonzero(background))
        if support_count < limits["support_min"]:
            report["issues"].append("subject_support_insufficient")
        if background_count < 256:
            report["issues"].append("background_support_insufficient")

        cell_hits = 0
        quadrant_hits = 0
        for row in range(4):
            for column in range(4):
                block = subject[
                    row * height // 4 : (row + 1) * height // 4,
                    column * width // 4 : (column + 1) * width // 4,
                ]
                cell_hits += int(np.count_nonzero(block) >= 64)
        for row in range(2):
            for column in range(2):
                block = subject[
                    row * height // 2 : (row + 1) * height // 2,
                    column * width // 2 : (column + 1) * width // 2,
                ]
                quadrant_hits += int(np.count_nonzero(block) >= 128)
        if cell_hits < limits["grid_cells_min"]:
            report["issues"].append("grid_coverage_insufficient")
        if quadrant_hits < limits["quadrants_min"]:
            report["issues"].append("quadrant_coverage_insufficient")

        rec709 = REC709[:, None, None]
        scale_vectors: Dict[str, np.ndarray] = {}
        scale_reports: Dict[str, Any] = {}
        for passes in (2, 4, 6):
            low = np.asarray(rgb, dtype=np.float64)
            for _ in range(passes):
                low = np.stack([_box_blur_gray(channel) for channel in low], axis=0)
            luminance = np.sum(low * rec709, axis=0)
            opponent = np.stack((low[0] - luminance, low[2] - luminance), axis=0)
            background_center = np.median(opponent[:, background], axis=1)
            opponent -= background_center[:, None, None]
            subject_vectors = opponent[:, subject].T
            background_vectors = opponent[:, background].T
            subject_energy = float(np.sqrt(np.mean(subject_vectors**2)))
            background_energy = float(np.sqrt(np.mean(background_vectors**2)))
            ratio = subject_energy / max(background_energy, 1e-12)
            mean_vector = np.mean(subject_vectors, axis=0)
            aggregate_snr = float(
                np.linalg.norm(mean_vector)
                / max(background_energy / math.sqrt(max(support_count, 1)), 1e-12)
            )
            scale_vectors[str(passes)] = subject_vectors.reshape(-1)
            scale_reports[str(passes)] = {
                "subject_energy": subject_energy,
                "background_energy": background_energy,
                "subject_background_energy_ratio": ratio,
                "aggregate_color_direction_snr": aggregate_snr,
            }
            if ratio < limits["subject_background_energy_ratio_min"]:
                report["issues"].append(f"scale_{passes}_energy_ratio_low")
            if aggregate_snr < limits["aggregate_snr_min"]:
                report["issues"].append(f"scale_{passes}_aggregate_snr_low")
        correlations: Dict[str, float] = {}
        for left, right in ((2, 4), (4, 6), (2, 6)):
            a = scale_vectors[str(left)]
            b = scale_vectors[str(right)]
            correlation = float(np.corrcoef(a, b)[0, 1])
            correlations[f"{left}x{right}"] = correlation
            if not math.isfinite(correlation) or correlation < limits["scale_correlation_min"]:
                report["issues"].append(f"scale_{left}x{right}_correlation_low")
        report.update(
            support={
                "subject_pixels": support_count,
                "background_pixels": background_count,
                "grid_cells": cell_hits,
                "quadrants": quadrant_hits,
            },
            scales=scale_reports,
            correlations=correlations,
        )
        report["issues"] = list(dict.fromkeys(report["issues"]))
        report["accepted"] = not report["issues"]
        report["status"] = "accepted" if report["accepted"] else "rejected"
        return report
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["issues"] = [str(error)]
        return report


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


def build_subject_chroma_review_contract(
    image: Any,
    *,
    tone_contract: Dict[str, Any],
    mask_artifact: Dict[str, Any],
    chroma_evidence: Dict[str, Any],
    effective_saturation_budget: float,
    artifact_root: str | Path,
    pixel_coordinate_domain: str,
) -> Dict[str, Any]:
    """Wrap one v2 tone contract in the bounded observer-only chroma contract."""

    if not validate_review_contract(tone_contract) or tone_contract.get("schema") == V3_SCHEMA:
        raise ValueError("v3 chroma rendition requires a valid v1/v2 tone contract")
    if not isinstance(chroma_evidence, dict) or chroma_evidence.get("accepted") is not True:
        raise ValueError("weak subject chroma evidence is not accepted")
    budget = float(np.clip(float(effective_saturation_budget), 0.0, 0.65))
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("observer chroma budget is zero")
    masks = load_display_rendition_masks(
        artifact_root,
        mask_artifact,
        pixel_coordinate_domain=pixel_coordinate_domain,
    )
    tone_pixels = apply_review_contract(image, tone_contract)
    tone_rgb, _tone_layout = _as_rgb_chw(tone_pixels)
    tone_peak = float(np.max(tone_rgb))
    tone_headroom_scale = (
        min(1.0, 0.995 / tone_peak)
        if math.isfinite(tone_peak) and tone_peak > 0.0
        else 1.0
    )
    tone_pixels = np.asarray(tone_pixels, dtype=np.float32) * tone_headroom_scale
    before = subject_saturation_distribution(tone_pixels, masks)
    if before.get("status") != "available":
        raise ValueError("subject saturation distribution is unavailable")
    budget_scale = budget / 0.65
    maximum_factor = min(4.0, 1.0 + 3.0 * budget_scale)
    p50_delta_max = 0.10 * budget_scale
    p50_limit = min(0.12, float(before["p50"]) + p50_delta_max)
    p95_limit = 0.30

    selected: Optional[tuple[float, np.ndarray, Dict[str, Any], Dict[str, Any]]] = None
    low = 1.0
    high = maximum_factor
    for _ in range(24):
        factor = high if selected is None and low == 1.0 else (low + high) / 2.0
        candidate, metadata = apply_subject_chroma_rendition(
            tone_pixels,
            masks,
            factor=factor,
            output_headroom=0.995,
            expand_faint_signal=False,
        )
        quality = assess_subject_chroma_candidate(metadata)
        after = subject_saturation_distribution(candidate, masks)
        accepted = bool(
            quality.get("accepted", False)
            and after.get("status") == "available"
            and float(after.get("p50", math.inf)) <= p50_limit + 1e-8
            and float(after.get("p95", math.inf)) <= p95_limit + 1e-8
        )
        if accepted:
            selected = (factor, candidate, metadata, after)
            low = factor
        else:
            high = factor
        if high - low <= 1e-5:
            break
    if selected is None or selected[0] <= 1.0 + 1e-6:
        raise ValueError("bounded subject chroma candidate did not pass hard gates")
    factor, _candidate, metadata, after = selected
    return {
        "schema": V3_SCHEMA,
        "status": "ready",
        "applicable": True,
        "observer_only": True,
        "mode": V3_CONTRACT_NAME,
        "name": V3_CONTRACT_NAME,
        "reason": str(tone_contract.get("reason") or "review_only"),
        "source_stem": str(tone_contract.get("source_stem") or ""),
        "source_shape_chw": list(tone_contract.get("source_shape_chw") or []),
        "derivative_pixels_changed": True,
        "pixel_coordinate_domain_required": True,
        "tone_contract": dict(tone_contract),
        "mask_artifact": dict(mask_artifact),
        "chroma_evidence": dict(chroma_evidence),
        "subject_chroma": {
            "method": "frozen_low_frequency_rgb_minus_y",
            "factor": float(factor),
            "maximum_factor": float(maximum_factor),
            "effective_saturation_budget": float(budget),
            "budget_scale": float(budget_scale),
            "p50_delta_max": float(p50_delta_max),
            "subject_saturation_before": before,
            "subject_saturation_after": after,
            "subject_saturation_limits": {"p50_max": 0.12, "p95_max": 0.30},
            "build_metadata": metadata,
            "quality_gate": assess_subject_chroma_candidate(metadata),
            "output_headroom": 0.995,
            "tone_headroom_scale": float(tone_headroom_scale),
            "new_hue_generated": False,
            "luminance_error_max": 1e-6,
            "newly_clipped_ratio_max": 0.0,
        },
        "applies_to": [
            "stage7_review_preview",
            "stage9_review_preview",
            "stage10_review_preview",
            "review_bundle_after",
            "result_review.png",
            "managed_display_srgb.png",
        ],
        "excluded_from": ["fit", "tiff", "scientific_output", "siril_buffer"],
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
    subject_chroma_plan: Optional[Dict[str, Any]] = None,
    artifact_root: Optional[str | Path] = None,
    pixel_coordinate_domain: str = PIXEL_DOMAIN_BOTTOM_UP,
) -> Dict[str, Any]:
    """Select identity or one linked mapping from a frozen visibility audit."""
    visibility = dict(input_visibility or {})
    exposure_state = str(visibility.get("exposure_state") or "unmappable")
    # Contract selection only decides how already-derived review pixels are
    # displayed.  Formal star-catalog or target gates may keep ``passed``
    # false even when exposure and scene content are safely visible; those
    # gates still control delivery, but must not invalidate an identity
    # observer mapping.
    if exposure_state == "acceptable":
        tone_contract = build_preserve_review_contract(
            image,
            reason=reason,
            source_stem=source_stem,
            input_visibility=visibility,
        )
    elif exposure_state == "underexposed":
        tone_contract = build_linked_review_contract(
            image,
            reason=reason,
            source_stem=source_stem,
            input_visibility=visibility,
        )
    else:
        return unavailable_contract(
            reason=reason,
            input_exposure_state=exposure_state,
            error=f"review source is not safely mappable: {exposure_state}",
        )
    if not isinstance(subject_chroma_plan, dict) or not subject_chroma_plan.get(
        "accepted", False
    ):
        return tone_contract
    if artifact_root is None:
        return tone_contract
    try:
        return build_subject_chroma_review_contract(
            image,
            tone_contract=tone_contract,
            mask_artifact=dict(subject_chroma_plan.get("mask_artifact") or {}),
            chroma_evidence=dict(subject_chroma_plan.get("evidence") or {}),
            effective_saturation_budget=float(
                subject_chroma_plan.get("effective_saturation_budget", 0.0) or 0.0
            ),
            artifact_root=artifact_root,
            pixel_coordinate_domain=pixel_coordinate_domain,
        )
    except (OSError, RuntimeError, TypeError, ValueError, FloatingPointError):
        return tone_contract


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
    if contract.get("schema") == V3_SCHEMA:
        tone_contract = contract.get("tone_contract")
        mask_artifact = contract.get("mask_artifact")
        evidence = contract.get("chroma_evidence")
        chroma = contract.get("subject_chroma")
        try:
            return bool(
                contract.get("status") == "ready"
                and contract.get("applicable") is True
                and contract.get("observer_only") is True
                and contract.get("mode") == V3_CONTRACT_NAME
                and contract.get("name") == V3_CONTRACT_NAME
                and isinstance(tone_contract, dict)
                and tone_contract.get("schema") in {SCHEMA, LEGACY_SCHEMA}
                and validate_review_contract(tone_contract)
                and isinstance(mask_artifact, dict)
                and mask_artifact.get("schema") == MASK_SCHEMA
                and mask_artifact.get("status") == "ready"
                and isinstance(evidence, dict)
                and evidence.get("accepted") is True
                and isinstance(chroma, dict)
                and chroma.get("method") == "frozen_low_frequency_rgb_minus_y"
                and 1.0 < float(chroma.get("factor")) <= 4.0
                and 0.0 < float(chroma.get("effective_saturation_budget")) <= 0.65
                and abs(float(chroma.get("output_headroom")) - 0.995) <= 1e-12
                and 0.0 < float(chroma.get("tone_headroom_scale", 1.0)) <= 1.0
            )
        except (TypeError, ValueError):
            return False
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
    *,
    artifact_root: Optional[str | Path] = None,
    pixel_coordinate_domain: Optional[str] = None,
) -> np.ndarray:
    """Replay a frozen contract without changing the source pixel buffer."""
    if not validate_review_contract(contract):
        raise ValueError("invalid Review rendition contract")
    rgb, layout = _as_rgb_chw(image)
    if contract.get("schema") == V3_SCHEMA:
        if artifact_root is None or pixel_coordinate_domain is None:
            raise ValueError(
                "v3 Review rendition requires artifact_root and pixel coordinate domain"
            )
        expected_shape = tuple(
            int(value) for value in (contract.get("source_shape_chw") or ())
        )
        if expected_shape and tuple(rgb.shape) != expected_shape:
            raise ValueError("v3 Review rendition source shape mismatch")
        tone_pixels = apply_review_contract(image, contract["tone_contract"])
        masks = load_display_rendition_masks(
            artifact_root,
            dict(contract["mask_artifact"]),
            pixel_coordinate_domain=pixel_coordinate_domain,
        )
        chroma = dict(contract["subject_chroma"])
        tone_pixels = np.asarray(tone_pixels, dtype=np.float32) * float(
            chroma.get("tone_headroom_scale", 1.0)
        )
        rendered, metadata = apply_subject_chroma_rendition(
            tone_pixels,
            masks,
            factor=float(chroma["factor"]),
            output_headroom=float(chroma["output_headroom"]),
            expand_faint_signal=False,
        )
        quality = assess_subject_chroma_candidate(metadata)
        distribution = subject_saturation_distribution(rendered, masks)
        limits = dict(chroma.get("subject_saturation_limits") or {})
        after = dict(chroma.get("subject_saturation_after") or {})
        build_metadata = dict(chroma.get("build_metadata") or {})
        replay_matches = bool(
            distribution.get("status") == "available"
            and abs(float(distribution.get("p50")) - float(after.get("p50"))) <= 2e-6
            and abs(float(distribution.get("p95")) - float(after.get("p95"))) <= 2e-6
            and abs(
                float(metadata.get("subject_saturation_median_before"))
                - float(build_metadata.get("subject_saturation_median_before"))
            )
            <= 2e-6
        )
        if not (
            quality.get("accepted", False)
            and replay_matches
            and float(distribution.get("p50", math.inf))
            <= float(limits.get("p50_max", 0.12)) + 1e-8
            and float(distribution.get("p95", math.inf))
            <= float(limits.get("p95_max", 0.30)) + 1e-8
        ):
            raise ValueError("v3 Review rendition replay failed its frozen hard gates")
        return _restore_layout(rendered, layout)
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
    *,
    artifact_root: Optional[str | Path] = None,
    pixel_coordinate_domain: Optional[str] = None,
) -> np.ndarray:
    """Backward-compatible entry point used by existing preview consumers."""
    return apply_review_contract(
        image,
        contract,
        artifact_root=artifact_root,
        pixel_coordinate_domain=pixel_coordinate_domain,
    )


__all__ = [
    "CONTRACT_NAME",
    "MASK_COORDINATE_DOMAIN",
    "MASK_SCHEMA",
    "PIXEL_DOMAIN_BOTTOM_UP",
    "PIXEL_DOMAIN_TOP_DOWN",
    "PRESERVE_CONTRACT_NAME",
    "SCHEMA",
    "V3_CONTRACT_NAME",
    "V3_SCHEMA",
    "apply_linked_review_contract",
    "apply_review_contract",
    "assess_weak_subject_chroma_source",
    "build_linked_review_contract",
    "build_preserve_review_contract",
    "build_review_contract",
    "build_subject_chroma_review_contract",
    "load_display_rendition_masks",
    "persist_display_rendition_masks",
    "unavailable_contract",
    "validate_linked_review_contract",
    "validate_review_contract",
]
