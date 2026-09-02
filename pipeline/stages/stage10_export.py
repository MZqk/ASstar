"""Stage 10 final denoise and export."""
import hashlib
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sirilpy.exceptions import CommandError, SirilError

from channel_semantics import BROADBAND_RGB_OSC
from color_quality_metrics import build_color_quality_report, resolve_color_contract
import display_rendition
from local_adjustments import dilate_mask, feather_mask
from managed_output import (
    audit_display_visibility,
    export_managed_outputs,
    read_managed_display_png,
    write_managed_display_png,
)
from models import PipelineStage
from output_color import build_output_color_manifest
from pipeline_safety import (
    clamp_saturation_boost,
    color_safety_limits,
    should_skip_final_denoise,
)
import presentation_quality
from save_utils import export_final_outputs
import spatial_background_lineage
import stage8_pixels
import stage9_quality


_STAGE10_CHROMA_FOCUS_MIN = 0.34
_STAGE10_SEPARATE_MIN = 0.70
_STAGE10_FULL_BG_STD_MIN = 0.018
_STAGE10_FULL_MOTTLING_MIN = 0.45
_STAGE10_DENOISE_STRENGTH = 0.28
_STAGE10_STAR_PROTECTION_COVERAGE_MAX = 0.35
_STAGE10_QUALITY_CHROMA_REPAIR_STRENGTH = 0.35
_STAGE10_QUALITY_TEXTURE_REPAIR_STRENGTH = 0.20
_STAGE10_QUALITY_RISK_IMPROVEMENT_MIN = 0.15
_STAGE10_QUALITY_SIGNAL_CORRELATION_MIN = 0.995
_STAGE10_QUALITY_SIGNAL_FLUX_RATIO_MIN = 0.98
_STAGE10_QUALITY_SIGNAL_FLUX_RATIO_MAX = 1.02
_STAGE10_QUALITY_CORE_CLIP_GROWTH_MAX = 0.001


def _stage10_mask_sha256(mask: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(mask, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(
        str(tuple(int(value) for value in canonical.shape)).encode("ascii")
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _stage10_freeze_authenticated_background_masks(
    pipeline,
) -> Dict[str, Any]:
    """Bind the Stage 10 noise gate to pre-candidate Stage 3/6 support."""

    report: Dict[str, Any] = {
        "schema": "starun.stage10-authenticated-background-support.v1",
        "status": "rejected",
        "accepted": False,
        "candidate_independent": True,
        "method": "stage3_sky_intersect_stage6_frozen_masks",
        "issues": [],
        "masks": {},
    }
    try:
        lineage = spatial_background_lineage.load_lineage(
            getattr(pipeline, "process_dir", None)
        )
        if lineage.get("accepted") is not True:
            raise ValueError(
                "authenticated Stage 3 spatial background lineage is unavailable"
            )
        support = np.asarray(lineage.get("support_mask"), dtype=np.float32)
        if support.ndim != 2 or not np.all(np.isfinite(support)):
            raise ValueError("authenticated Stage 3 sky support is invalid")
        support = np.clip(support, 0.0, 1.0)

        frozen = getattr(pipeline, "_stage7_frozen_rendition_masks", None)
        if not isinstance(frozen, dict) or not frozen:
            raise ValueError("Stage 6 frozen rendition masks are unavailable")
        stage6_background = np.asarray(
            frozen.get("background_mask"),
            dtype=np.float32,
        )
        if (
            stage6_background.shape != support.shape
            or not np.all(np.isfinite(stage6_background))
        ):
            raise ValueError("Stage 6 frozen background mask is invalid")

        masks: Dict[str, np.ndarray] = {
            "background_mask": np.clip(
                support * np.clip(stage6_background, 0.0, 1.0),
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
        }
        for name in (
            "core_mask",
            "nebula_mask",
            "faint_nebula_mask",
            "galaxy_signal_mask",
            "limited_core_exclusion_mask",
            "star_halo_guard_mask",
            "star_mask",
            "subject_mask",
            "shared_valid_mask",
            "original_saturation_map",
            "saturation_map",
        ):
            raw = frozen.get(name)
            if raw is None:
                continue
            value = np.asarray(raw, dtype=np.float32)
            if value.shape != support.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Stage 6 frozen {name} is invalid")
            masks[name] = np.clip(value, 0.0, 1.0).astype(
                np.float32,
                copy=True,
            )

        exclusive = stage8_pixels._stage8_exclusive_background_weight(
            masks,
            masks["background_mask"],
        )
        coverage = float(np.mean(exclusive > 0.50))
        if not math.isfinite(coverage) or coverage <= 0.01:
            raise ValueError(
                "authenticated frozen background coverage is insufficient: "
                f"{coverage:.6f}<=0.010000"
            )

        mask_records = {
            name: {
                "shape": [int(value) for value in mask.shape],
                "sha256": _stage10_mask_sha256(mask),
            }
            for name, mask in sorted(masks.items())
        }
        report.update(
            status="accepted",
            accepted=True,
            issues=[],
            coverage_gt_0_50=coverage,
            stage3_support_sha256=lineage.get("support_sha256"),
            stage3_lineage_chain_digest=lineage.get("chain_digest"),
            masks=mask_records,
        )
        pipeline._stage10_quality_frozen_background_masks = masks
        pipeline._stage10_quality_frozen_background_sampling = dict(report)
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        report["issues"] = [str(error)]
        pipeline._stage10_quality_frozen_background_masks = None
        pipeline._stage10_quality_frozen_background_sampling = dict(report)
    return report


def _metric_value(metrics: Dict[str, Any], name: str) -> float:
    try:
        value = float(metrics.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _has_complete_noise_metrics(metrics: Dict[str, Any]) -> bool:
    for name in (
        "chroma_noise_score",
        "bg_std",
        "background_mottling_score",
    ):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value) or value < 0.0:
            return False
    return True


def _config_value(cfg: Any, name: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(getattr(cfg, name, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(lower, min(upper, value))


def _stage10_stage9_contract_state(
    pipeline,
    *,
    final_source: str,
    final_loaded: bool,
) -> Dict[str, Any]:
    """Resolve the explicit Stage9 delivery contract for the loaded source."""

    required_fields = (
        "_stage9_stars_required",
        "_stage9_stars_applied",
        "_stage9_output_contains_stars",
        "_stage9_remix_formally_accepted",
        "_stage9_star_delivery_contract_accepted",
        "_stage9_final_source",
    )
    missing_fields = [
        name for name in required_fields if not hasattr(pipeline, name)
    ]
    declared_source = str(
        getattr(pipeline, "_stage9_final_source", "") or ""
    )
    source_matched = bool(
        final_loaded
        and declared_source
        and declared_source == str(final_source or "")
    )
    known = bool(not missing_fields and source_matched)
    stars_required = bool(
        getattr(pipeline, "_stage9_stars_required", False)
    )
    stars_applied = bool(
        getattr(pipeline, "_stage9_stars_applied", False)
    )
    output_contains_stars = bool(
        getattr(pipeline, "_stage9_output_contains_stars", False)
    )
    remix_formally_accepted = bool(
        getattr(pipeline, "_stage9_remix_formally_accepted", False)
    )
    delivery_contract_accepted = bool(
        getattr(
            pipeline,
            "_stage9_star_delivery_contract_accepted",
            False,
        )
    )
    formal = bool(
        known
        and remix_formally_accepted
        and delivery_contract_accepted
        and output_contains_stars
        and (not stars_required or stars_applied)
    )
    return {
        "schema": "starun.stage10-stage9-contract-state.v1",
        "known": known,
        "formal": formal,
        "missing_fields": missing_fields,
        "declared_source": declared_source or None,
        "loaded_source": str(final_source or "") or None,
        "source_matched": source_matched,
        "remix_formally_accepted": remix_formally_accepted,
        "delivery_contract_accepted": delivery_contract_accepted,
        "stars_required": stars_required,
        "stars_applied": stars_applied,
        "output_contains_stars": output_contains_stars,
    }


def _stage10_stage9_local_color_saturation_guard(
    pipeline,
    saturation: float,
) -> Tuple[float, Dict[str, Any]]:
    """Reduce positive saturation when Stage 9 reports localized color risk."""
    requested = float(saturation)
    selected = getattr(pipeline, "_stage9_selected_remix_quality", None)
    metrics = (
        selected.get("metrics") or {}
        if isinstance(selected, dict)
        else {}
    )
    local_status = str(metrics.get("local_quality_status", "") or "")
    stage9_contract_known = hasattr(pipeline, "_stage9_stars_applied")
    stars_required = bool(getattr(pipeline, "_stage9_stars_required", False))
    stars_applied = bool(getattr(pipeline, "_stage9_stars_applied", False))
    reason = "no_local_color_risk"
    risk_score = 0.0
    if stage9_contract_known and stars_required and not stars_applied:
        risk_score = 1.0
        reason = "stage9_required_stars_not_applied"
    elif isinstance(selected, dict) and bool(selected.get("accepted", False)):
        if local_status == "ok":
            risk_score = max(
                0.0,
                min(
                    1.0,
                    _metric_value(metrics, "local_color_risk_score"),
                ),
            )
            if risk_score > 0.0:
                reason = "stage9_local_color_risk"
        elif stage9_contract_known and stars_required:
            risk_score = 1.0
            reason = "stage9_local_color_metrics_unavailable"
    strength = _config_value(
        getattr(pipeline, "cfg", None),
        "stage10_stage9_local_color_risk_strength",
        1.0,
        0.0,
        1.0,
    )
    factor = max(0.0, 1.0 - risk_score * strength)
    guarded = requested * factor if requested > 0.0 else requested
    return guarded, {
        "requested_saturation": requested,
        "effective_saturation": guarded,
        "local_quality_status": local_status or "not_available",
        "local_color_risk_score": risk_score,
        "risk_strength": strength,
        "saturation_factor": factor,
        "applied": bool(requested > 0.0 and guarded < requested - 1e-8),
        "reason": reason,
    }


def _select_stage10_denoise_plan(
    metrics: Optional[Dict[str, Any]],
    *,
    color_input: bool,
    cfg: Any = None,
) -> Dict[str, Any]:
    """Select full/chroma/separate/skip from input background metrics."""
    measured = dict(metrics or {})
    metrics_available = _has_complete_noise_metrics(measured)
    chroma = _metric_value(measured, "chroma_noise_score")
    bg_std = _metric_value(measured, "bg_std")
    mottling = _metric_value(measured, "background_mottling_score")
    chroma_focus_min = _config_value(
        cfg,
        "stage10_chroma_focus_score_min",
        _STAGE10_CHROMA_FOCUS_MIN,
        0.10,
        0.80,
    )
    separate_min = max(
        chroma_focus_min,
        _config_value(
            cfg,
            "stage10_separate_chroma_score_min",
            _STAGE10_SEPARATE_MIN,
            0.35,
            1.50,
        ),
    )
    full_bg_std_min = _config_value(
        cfg,
        "stage10_full_bg_std_min",
        _STAGE10_FULL_BG_STD_MIN,
        0.001,
        0.10,
    )
    full_mottling_min = _config_value(
        cfg,
        "stage10_full_mottling_score_min",
        _STAGE10_FULL_MOTTLING_MIN,
        0.10,
        1.00,
    )

    if not color_input:
        selected_mode = "full"
        reason = "non-color input"
    elif not metrics_available:
        selected_mode = "full"
        reason = "input metrics unavailable; balanced fallback"
    elif chroma >= separate_min:
        selected_mode = "separate"
        reason = "severe channel-specific background color noise"
    elif chroma >= chroma_focus_min:
        if bg_std >= full_bg_std_min or mottling >= full_mottling_min:
            selected_mode = "full"
            reason = "material luminance/background noise accompanies color noise"
        else:
            selected_mode = "chroma"
            reason = "color noise dominates a comparatively stable luminance background"
    elif bg_std >= full_bg_std_min or mottling >= full_mottling_min:
        selected_mode = "full"
        reason = "material luminance/background noise requires full denoise"
    else:
        selected_mode = "skip"
        reason = "all measured background noise is below final-denoise thresholds"

    # Bundled CosmicClarity supports full/luminance/separate. Chroma-only is
    # produced by running full and restoring the original luminance afterwards.
    # The low-noise skip path never invokes CosmicClarity.
    cosmic_clarity_mode = (
        "none"
        if selected_mode == "skip"
        else "full" if selected_mode == "chroma" else selected_mode
    )
    return {
        "selected_mode": selected_mode,
        "cosmic_clarity_mode": cosmic_clarity_mode,
        "reason": reason,
        "color_input": bool(color_input),
        "input_metrics_available": metrics_available,
        "input_metrics": {
            "chroma_noise_score": chroma,
            "bg_std": bg_std,
            "background_mottling_score": mottling,
        },
        "thresholds": {
            "chroma_focus_min": chroma_focus_min,
            "separate_min": separate_min,
            "full_bg_std_min": full_bg_std_min,
            "full_mottling_min": full_mottling_min,
        },
    }


def _is_color_image(image: np.ndarray) -> bool:
    arr = np.asarray(image)
    return bool(
        arr.ndim == 3
        and (arr.shape[0] in (3, 4) or arr.shape[-1] in (3, 4))
    )


def _rgb_float(image: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected RGB image, got shape={arr.shape}")
    if arr.shape[0] >= 3 and arr.shape[-1] not in (1, 3, 4):
        rgb = arr[:3]
        layout = "chw"
    elif arr.shape[-1] >= 3:
        rgb = np.transpose(arr[..., :3], (2, 0, 1))
        layout = "hwc"
    elif arr.shape[0] >= 3:
        rgb = arr[:3]
        layout = "chw"
    else:
        raise ValueError(f"expected RGB image, got shape={arr.shape}")

    rgb = np.nan_to_num(
        np.asarray(rgb, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
    else:
        peak = float(np.max(np.abs(rgb))) if rgb.size else 0.0
        if peak <= 1.5:
            scale = 1.0
        elif peak <= 255.0 * 1.05:
            scale = 255.0
        elif peak <= 65535.0 * 1.05:
            scale = 65535.0
        else:
            scale = max(peak, 1.0)
    return np.clip(rgb / max(scale, 1e-12), 0.0, 1.0), layout


def _restore_rgb_like(source: np.ndarray, rgb: np.ndarray, layout: str) -> np.ndarray:
    source_arr = np.asarray(source)
    restored = np.asarray(rgb, dtype=np.float32)
    if layout == "hwc":
        restored = np.transpose(restored, (1, 2, 0))
    if np.issubdtype(source_arr.dtype, np.integer):
        scale = float(np.iinfo(source_arr.dtype).max)
        return np.rint(np.clip(restored, 0.0, 1.0) * scale).astype(
            source_arr.dtype,
            copy=False,
        )
    peak = float(np.max(np.abs(source_arr))) if source_arr.size else 0.0
    if peak <= 1.5:
        scale = 1.0
    elif peak <= 255.0 * 1.05:
        scale = 255.0
    elif peak <= 65535.0 * 1.05:
        scale = 65535.0
    else:
        scale = max(peak, 1.0)
    return (np.clip(restored, 0.0, 1.0) * scale).astype(
        source_arr.dtype,
        copy=False,
    )


def _chroma_only_denoised_image(
    original: np.ndarray,
    denoised: np.ndarray,
) -> np.ndarray:
    """Keep denoised chroma while restoring the pre-denoise luminance."""
    if np.asarray(original).shape != np.asarray(denoised).shape:
        raise ValueError(
            "chroma-only merge shape mismatch: "
            f"original={np.asarray(original).shape}, denoised={np.asarray(denoised).shape}"
        )
    original_rgb, original_layout = _rgb_float(original)
    denoised_rgb, denoised_layout = _rgb_float(denoised)
    if original_layout != denoised_layout:
        raise ValueError(
            f"chroma-only merge layout mismatch: {original_layout}!={denoised_layout}"
        )
    original_luma = (
        0.2126 * original_rgb[0]
        + 0.7152 * original_rgb[1]
        + 0.0722 * original_rgb[2]
    )
    denoised_luma = (
        0.2126 * denoised_rgb[0]
        + 0.7152 * denoised_rgb[1]
        + 0.0722 * denoised_rgb[2]
    )
    chroma_only = np.clip(
        denoised_rgb + (original_luma - denoised_luma)[None, :, :],
        0.0,
        1.0,
    )
    return _restore_rgb_like(np.asarray(denoised), chroma_only, denoised_layout)


def _stage10_denoise_input(pipeline) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("empty Stage10 denoise input")
        image = np.asarray(image_data)
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        pipeline.log.warn(f"Stage10 denoise input unavailable: {error}")
        return None, _select_stage10_denoise_plan(
            {},
            color_input=True,
            cfg=getattr(pipeline, "cfg", None),
        )

    try:
        metric_reader = getattr(pipeline, "_background_quality_metrics", None)
        metrics = metric_reader(image) if callable(metric_reader) else {}
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        pipeline.log.warn(f"Stage10 denoise input metrics unavailable: {error}")
        metrics = {}

    plan = _select_stage10_denoise_plan(
        metrics if isinstance(metrics, dict) else {},
        color_input=_is_color_image(image),
        cfg=getattr(pipeline, "cfg", None),
    )
    # Every active denoise mode is transactional: the frozen source is needed
    # both for star-protected recomposition and for fail-closed rollback. The
    # low-noise skip path deliberately avoids retaining a full-resolution copy.
    snapshot = (
        np.array(image, copy=True)
        if plan["selected_mode"] != "skip"
        else None
    )
    return snapshot, plan


def _stage10_spatial_shape(image: np.ndarray) -> Tuple[int, int]:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim != 3:
        raise ValueError(f"unsupported Stage10 image shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        return int(arr.shape[1]), int(arr.shape[2])
    if arr.shape[-1] in (1, 3, 4):
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.shape[0] in (1, 3, 4):
        return int(arr.shape[1]), int(arr.shape[2])
    raise ValueError(f"cannot determine Stage10 image layout from shape={arr.shape}")


def _stage10_expanded_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    spatial = np.asarray(mask)
    if source.ndim == 2 and source.shape == spatial.shape:
        return spatial
    if source.ndim == 3 and source.shape[1:] == spatial.shape:
        return spatial[np.newaxis, ...]
    if source.ndim == 3 and source.shape[:2] == spatial.shape:
        return spatial[..., np.newaxis]
    raise ValueError(
        "Stage10 star-protection shape mismatch: "
        f"image={source.shape}, mask={spatial.shape}"
    )


def _build_stage10_star_protection_mask(
    pipeline,
    original: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Build a bounded mask only from Stage 9's validated star catalog."""
    report: Dict[str, Any] = {
        "schema": "starun.stage10-star-protection.v1",
        "status": "unavailable",
        "source": "stage9_validated_star_reference_catalog",
        "reason": "unknown",
    }
    if original is None:
        report["reason"] = "frozen_stage10_input_unavailable"
        return None, report
    if not hasattr(pipeline, "_stage9_stars_applied"):
        report["reason"] = "stage9_star_contract_unavailable"
        return None, report
    if not bool(getattr(pipeline, "_stage9_stars_applied", False)):
        report["reason"] = "stage9_stars_not_applied"
        return None, report

    catalog = getattr(pipeline, "_stage9_star_reference_catalog", None)
    if not isinstance(catalog, dict) or str(catalog.get("status", "")) != "ok":
        report["reason"] = (
            str(catalog.get("reason") or "stage9_star_reference_catalog_unavailable")
            if isinstance(catalog, dict)
            else "stage9_star_reference_catalog_unavailable"
        )
        return None, report

    try:
        _weak, _bright, union = stage9_quality.build_star_overlay_masks(
            catalog,
            strict=False,
        )
        hard_core = np.asarray(union, dtype=bool)
        if hard_core.shape != _stage10_spatial_shape(np.asarray(original)):
            raise ValueError(
                "validated Stage9 catalog is not aligned with Stage10 input: "
                f"catalog={hard_core.shape}, input={_stage10_spatial_shape(np.asarray(original))}"
            )
        if not np.any(hard_core):
            raise ValueError("validated Stage9 star support is empty")

        outer = dilate_mask(hard_core, iterations=1)
        feathered = feather_mask(outer, radius=2)
        protection = np.maximum(
            hard_core.astype(np.float32),
            np.asarray(feathered, dtype=np.float32),
        )
        if not np.all(np.isfinite(protection)):
            raise ValueError("Stage10 star-protection mask contains non-finite values")
        protection = np.clip(protection, 0.0, 1.0)
        hard_coverage = float(np.mean(hard_core))
        protected_coverage = float(np.mean(protection > 0.01))
        weighted_coverage = float(np.mean(protection))
        coverage_max = _config_value(
            getattr(pipeline, "cfg", None),
            "stage10_star_protection_coverage_max",
            _STAGE10_STAR_PROTECTION_COVERAGE_MAX,
            0.05,
            0.60,
        )
        if protected_coverage > coverage_max:
            raise ValueError(
                "Stage10 star-protection coverage exceeds safety limit: "
                f"{protected_coverage:.6f}>{coverage_max:.6f}"
            )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        report["reason"] = str(error)
        return None, report

    report.update(
        {
            "status": "ready",
            "reason": "validated_stage9_catalog",
            "hard_coverage": hard_coverage,
            "protected_coverage": protected_coverage,
            "weighted_coverage": weighted_coverage,
            "coverage_max": coverage_max,
            "hard_core_bit_exact": True,
            "feather_radius": 2,
            "dilation_iterations": 1,
        }
    )
    return protection, report


def _star_protected_denoised_image(
    original: np.ndarray,
    denoised: np.ndarray,
    protection_mask: np.ndarray,
) -> np.ndarray:
    """Restore protected star pixels and feather their boundary into the result."""
    source = np.asarray(original)
    candidate = np.asarray(denoised)
    if source.shape != candidate.shape:
        raise ValueError(
            "Stage10 protected merge shape mismatch: "
            f"original={source.shape}, denoised={candidate.shape}"
        )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Stage10 protected merge received non-finite pixels")

    mask = np.clip(np.asarray(protection_mask, dtype=np.float32), 0.0, 1.0)
    if not np.all(np.isfinite(mask)):
        raise ValueError("Stage10 protected merge mask contains non-finite values")
    expanded = _stage10_expanded_mask(source, mask)
    source_float = source.astype(np.float64, copy=False)
    candidate_float = candidate.astype(np.float64, copy=False)
    merged_float = candidate_float * (1.0 - expanded) + source_float * expanded

    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        merged = np.rint(np.clip(merged_float, info.min, info.max)).astype(source.dtype)
    else:
        merged = merged_float.astype(source.dtype, copy=False)
    merged = np.where(expanded <= 0.0, candidate, merged)
    merged = np.where(expanded >= 1.0 - 1e-7, source, merged)
    return merged.astype(source.dtype, copy=False)


def _set_stage10_pixels(pipeline, pixels: np.ndarray, *, label: str) -> None:
    setter = getattr(pipeline, "_set_current_image_pixeldata", None)
    if callable(setter):
        setter(pixels, label=label)
        return
    lock_factory = getattr(pipeline.siril, "image_lock", None)
    if callable(lock_factory):
        with lock_factory():
            pipeline.siril.set_image_pixeldata(pixels)
        return
    pipeline.siril.set_image_pixeldata(pixels)


def _apply_stage10_star_protected_result(
    pipeline,
    original: Optional[np.ndarray],
    protection_mask: Optional[np.ndarray],
) -> Tuple[bool, str]:
    if original is None or protection_mask is None:
        return False, "frozen input or validated star-protection mask unavailable"
    try:
        denoised = pipeline.siril.get_image_pixeldata(preview=False)
        if denoised is None:
            raise RuntimeError("empty Stage10 denoised image")
        merged = _star_protected_denoised_image(
            original,
            np.asarray(denoised),
            protection_mask,
        )
        _set_stage10_pixels(
            pipeline,
            merged,
            label="Stage10 star-protected denoise merge",
        )
        return True, "validated Stage9 star support restored with feathered boundary"
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        return False, str(error)


def _rollback_stage10_denoise(
    pipeline,
    original: Optional[np.ndarray],
) -> Tuple[bool, str]:
    if original is None:
        return False, "frozen Stage10 input unavailable"
    try:
        _set_stage10_pixels(
            pipeline,
            original,
            label="Stage10 denoise safety rollback",
        )
        return True, "frozen pre-denoise input restored"
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        return False, str(error)


def _apply_stage10_chroma_only_result(
    pipeline,
    original: Optional[np.ndarray],
) -> Tuple[bool, str]:
    if original is None:
        return False, "original Stage10 input unavailable"
    try:
        denoised = pipeline.siril.get_image_pixeldata(preview=False)
        if denoised is None:
            raise RuntimeError("empty denoised image")
        merged = _chroma_only_denoised_image(original, np.asarray(denoised))
        setter = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(setter):
            setter(merged, label="Stage10 chroma-only merge")
        else:
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    pipeline.siril.set_image_pixeldata(merged)
            else:
                pipeline.siril.set_image_pixeldata(merged)
        return True, "original luminance restored after full-model chroma denoise"
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        return False, str(error)


def _run_stage10_scunet_fallback(
    pipeline,
    step_key: str,
    strength: float,
) -> Optional[str]:
    """Avoid another long AI wait after the primary denoiser already timed out."""
    primary_error = str(
        getattr(pipeline, "_last_plugin_script_error", None) or ""
    ).strip()
    if "timeout" in primary_error.lower() or "timed out" in primary_error.lower():
        reason = (
            "SCUNet skipped after primary denoiser timeout to avoid a second "
            "long-running fallback"
        )
        pipeline._last_scunet_fallback_error = reason
        pipeline.log.warn(f"{step_key}: {reason}")
        return None
    return pipeline._run_siril_scunet_denoise_fallback(step_key, strength)


def _stage10_current_pixels(pipeline) -> Optional[np.ndarray]:
    try:
        image = pipeline.siril.get_image_pixeldata(preview=False)
        if image is None:
            return None
        return np.array(image, copy=True)
    except (
        AttributeError,
        CommandError,
        SirilError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def _stage10_quality_noise_risk(report: Dict[str, Any]) -> float:
    metrics = report.get("metrics") or {}
    limits = metrics.get("noise_gate_limits") or {}
    if not isinstance(metrics, dict) or not isinstance(limits, dict):
        return float("inf")

    pairs = (
        ("background_chroma_noise_score", "chroma_advisory_max"),
        ("background_mottling_score", "mottling_advisory_max"),
        ("starless_artifact_score", "artifact_advisory_max"),
        (
            "local_texture_residual_outlier_score",
            "texture_outlier_score_hard_max",
        ),
        (
            "local_texture_affected_patch_ratio",
            "texture_affected_ratio_hard_max",
        ),
    )
    risk = 0.0
    measured = False
    for metric_name, limit_name in pairs:
        try:
            value = float(metrics[metric_name])
            limit = float(limits[limit_name])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or not math.isfinite(limit) or limit <= 0.0:
            continue
        measured = True
        risk += max(0.0, value / limit - 1.0)
    return risk if measured else float("inf")


def _stage10_quality_repairable(report: Dict[str, Any]) -> Tuple[bool, str]:
    if str(report.get("severity") or "") != "hard_reject":
        return False, "final_quality_is_not_hard_reject"
    hard_issues = report.get("hard_issues")
    if not isinstance(hard_issues, list) or not hard_issues:
        return False, "structured_hard_issues_unavailable"
    recoverable_prefixes = (
        "background_chroma_noise_extreme",
        "background_mottling_extreme",
        "starless_artifact_extreme",
        "local_texture_residual_extreme",
        "background_noise_combined_growth",
    )
    unsupported = [
        str(issue)
        for issue in hard_issues
        if not str(issue).startswith(recoverable_prefixes)
    ]
    if unsupported:
        return False, "non_repairable_hard_issues:" + ",".join(unsupported[:3])
    return True, "recoverable_noise_only"


def _stage10_quality_repair_candidate(
    pipeline,
    original: np.ndarray,
    report: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    source = np.asarray(original)
    rgb, layout = _rgb_float(source)
    frozen_masks = getattr(
        pipeline,
        "_stage10_quality_frozen_background_masks",
        None,
    )
    if not isinstance(frozen_masks, dict) or not frozen_masks:
        raise ValueError("frozen signal-excluded background masks unavailable")
    background = np.asarray(
        frozen_masks["background_mask"],
        dtype=np.float32,
    )
    background_weight = stage8_pixels._stage8_exclusive_background_weight(
        frozen_masks,
        background,
    )
    if background_weight.shape != rgb.shape[1:]:
        raise ValueError(
            "quality repair background mask shape mismatch: "
            f"mask={background_weight.shape}, image={rgb.shape[1:]}"
        )

    star_protection, star_report = _build_stage10_star_protection_mask(
        pipeline,
        source,
    )
    if star_protection is None:
        raise ValueError(
            "validated Stage9 star protection unavailable: "
            f"{star_report.get('reason', 'unknown')}"
        )
    repair_weight = np.clip(
        background_weight * (1.0 - np.asarray(star_protection, dtype=np.float32)),
        0.0,
        1.0,
    )
    if float(np.mean(repair_weight > 0.05)) <= 0.01:
        raise ValueError("quality repair background coverage is too small")

    metrics = report.get("metrics") or {}
    limits = metrics.get("noise_gate_limits") or {}
    chroma = _metric_value(metrics, "background_chroma_noise_score")
    mottling = _metric_value(metrics, "background_mottling_score")
    artifact = _metric_value(metrics, "starless_artifact_score")
    patch_variance = _metric_value(metrics, "local_patch_variance")
    patch_limit = max(_metric_value(metrics, "local_patch_variance_max"), 1e-9)
    texture_outlier = _metric_value(
        metrics,
        "local_texture_residual_outlier_score",
    )
    texture_affected = _metric_value(
        metrics,
        "local_texture_affected_patch_ratio",
    )
    chroma_repair = chroma > _metric_value(limits, "chroma_advisory_max")
    texture_repair = bool(
        patch_variance > patch_limit
        or mottling > _metric_value(limits, "mottling_advisory_max")
        or artifact > _metric_value(limits, "artifact_advisory_max")
        or (
            texture_outlier
            > _metric_value(limits, "texture_outlier_score_hard_max")
            and texture_affected
            > _metric_value(limits, "texture_affected_ratio_hard_max")
        )
    )
    if not chroma_repair and not texture_repair:
        raise ValueError("hard noise report has no repairable active metric")

    gray = (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ).astype(np.float32)
    candidate_rgb = np.array(rgb, copy=True)
    if chroma_repair:
        red_green = rgb[0] - rgb[1]
        blue_green = rgb[2] - rgb[1]
        repaired_red_green = red_green - (
            red_green - stage8_pixels._box_blur_gray(red_green)
        ) * (_STAGE10_QUALITY_CHROMA_REPAIR_STRENGTH * repair_weight)
        repaired_blue_green = blue_green - (
            blue_green - stage8_pixels._box_blur_gray(blue_green)
        ) * (_STAGE10_QUALITY_CHROMA_REPAIR_STRENGTH * repair_weight)
        repaired_green = (
            gray
            - 0.2126 * repaired_red_green
            - 0.0722 * repaired_blue_green
        )
        candidate_rgb = np.stack(
            (
                repaired_green + repaired_red_green,
                repaired_green,
                repaired_green + repaired_blue_green,
            ),
            axis=0,
        ).astype(np.float32)

    texture_affected_ratio = 0.0
    if texture_repair:
        blur = stage8_pixels._box_blur_gray(gray)
        residual = gray - blur
        patch = 16
        h, w = gray.shape
        tile_records: List[Tuple[int, int, float]] = []
        for y in range(0, max(h - patch + 1, 1), patch):
            for x in range(0, max(w - patch + 1, 1), patch):
                tile_weight = repair_weight[y:y + patch, x:x + patch]
                if not tile_weight.size or float(np.mean(tile_weight)) <= 0.65:
                    continue
                tile_residual = residual[y:y + patch, x:x + patch]
                total = max(float(np.sum(tile_weight)), 1e-6)
                mean = float(np.sum(tile_residual * tile_weight) / total)
                rms = float(
                    np.sqrt(
                        np.sum(((tile_residual - mean) ** 2) * tile_weight)
                        / total
                    )
                )
                tile_records.append((y, x, rms))
        texture_weight = np.zeros_like(gray, dtype=np.float32)
        if tile_records:
            tile_rms = np.asarray(
                [record[2] for record in tile_records],
                dtype=np.float64,
            )
            tile_median = float(np.median(tile_rms))
            tile_mad = float(np.median(np.abs(tile_rms - tile_median)))
            tile_scale = max(1.4826 * tile_mad, tile_median * 0.15, 1e-6)
            affected = 0
            for y, x, rms in tile_records:
                if (rms - tile_median) / tile_scale <= 4.0:
                    continue
                affected += 1
                texture_weight[y:y + patch, x:x + patch] = repair_weight[
                    y:y + patch,
                    x:x + patch,
                ]
            texture_affected_ratio = affected / max(len(tile_records), 1)
        if artifact > _metric_value(limits, "artifact_advisory_max"):
            texture_weight = np.maximum(texture_weight, repair_weight)
        repaired_luma = gray - residual * (
            _STAGE10_QUALITY_TEXTURE_REPAIR_STRENGTH * texture_weight
        )
        candidate_luma = (
            0.2126 * candidate_rgb[0]
            + 0.7152 * candidate_rgb[1]
            + 0.0722 * candidate_rgb[2]
        )
        candidate_rgb += (repaired_luma - candidate_luma)[None, :, :]

    bg_std = max(_metric_value(metrics, "bg_std"), 0.0)
    delta_limit = max(0.002, min(0.02, 3.0 * bg_std))
    delta = np.clip(candidate_rgb - rgb, -delta_limit, delta_limit)
    candidate_rgb = np.clip(rgb + delta, 0.0, 1.0)
    candidate = _restore_rgb_like(source, candidate_rgb, layout)
    candidate = _star_protected_denoised_image(
        source,
        candidate,
        np.asarray(star_protection, dtype=np.float32),
    )
    if candidate.shape != source.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("quality repair candidate is invalid")
    return candidate, {
        "mode": "bounded_signal_excluded_noise_repair",
        "chroma_repair": bool(chroma_repair),
        "texture_repair": bool(texture_repair),
        "chroma_strength": _STAGE10_QUALITY_CHROMA_REPAIR_STRENGTH,
        "texture_strength": _STAGE10_QUALITY_TEXTURE_REPAIR_STRENGTH,
        "delta_limit": delta_limit,
        "background_coverage": float(np.mean(repair_weight > 0.05)),
        "texture_affected_patch_ratio": texture_affected_ratio,
        "star_protection": star_report,
    }


def _stage10_quality_repair_structure_metrics(
    pipeline,
    original: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, float]:
    original_rgb, _layout = _rgb_float(original)
    candidate_rgb, _candidate_layout = _rgb_float(candidate)
    original_luma = (
        0.2126 * original_rgb[0]
        + 0.7152 * original_rgb[1]
        + 0.0722 * original_rgb[2]
    )
    candidate_luma = (
        0.2126 * candidate_rgb[0]
        + 0.7152 * candidate_rgb[1]
        + 0.0722 * candidate_rgb[2]
    )
    masks = getattr(pipeline, "_stage10_quality_frozen_background_masks", None)
    if not isinstance(masks, dict) or not masks:
        raise ValueError("frozen background masks unavailable for structure check")
    background = np.asarray(masks["background_mask"], dtype=np.float32)
    exclusive = stage8_pixels._stage8_exclusive_background_weight(
        masks,
        background,
    )
    signal_weight = np.clip(1.0 - exclusive, 0.0, 1.0)
    signal = signal_weight > 0.25
    if int(np.sum(signal)) < 64:
        raise ValueError("insufficient protected signal pixels for structure check")
    before_values = original_luma[signal].astype(np.float64)
    after_values = candidate_luma[signal].astype(np.float64)
    before_std = float(np.std(before_values))
    after_std = float(np.std(after_values))
    if before_std <= 1e-9 or after_std <= 1e-9:
        correlation = 1.0 if np.allclose(before_values, after_values, atol=1e-6) else 0.0
    else:
        correlation = float(np.corrcoef(before_values, after_values)[0, 1])
    before_flux = float(np.sum(before_values * signal_weight[signal]))
    after_flux = float(np.sum(after_values * signal_weight[signal]))
    flux_ratio = after_flux / max(before_flux, 1e-9)
    original_clip = float(np.mean(original_luma >= 0.985))
    candidate_clip = float(np.mean(candidate_luma >= 0.985))
    return {
        "signal_luminance_correlation": correlation,
        "signal_flux_ratio": flux_ratio,
        "core_clip_growth": candidate_clip - original_clip,
    }


def _attempt_stage10_quality_repair(
    pipeline,
    initial_report: Dict[str, Any],
    *,
    source_trusted: bool,
) -> Dict[str, Any]:
    repairable, reason = _stage10_quality_repairable(initial_report)
    repair: Dict[str, Any] = {
        "attempted": False,
        "status": "not_requested",
        "reason": reason,
    }
    if not source_trusted:
        repair.update(status="skipped", reason="stage10_source_not_trusted")
        initial_report["repair"] = repair
        pipeline._stage10_quality_repair_report = dict(repair)
        return initial_report
    if not repairable:
        initial_report["repair"] = repair
        pipeline._stage10_quality_repair_report = dict(repair)
        return initial_report
    frozen_masks = getattr(
        pipeline,
        "_stage10_quality_frozen_background_masks",
        None,
    )
    if not isinstance(frozen_masks, dict) or not frozen_masks:
        repair.update(status="skipped", reason="trusted_background_masks_unavailable")
        initial_report["repair"] = repair
        pipeline._stage10_quality_repair_report = dict(repair)
        return initial_report

    repair.update(attempted=True, status="attempting", reason=reason)
    pipeline._stage10_quality_repair_report = dict(repair)
    original = _stage10_current_pixels(pipeline)
    if original is None:
        repair.update(status="failed", reason="stage10_final_pixels_unavailable")
        initial_report["repair"] = repair
        pipeline._stage10_quality_repair_report = dict(repair)
        return initial_report
    if not pipeline._save_stage_output("stage10_pre_quality_repair"):
        repair.update(status="failed", reason="pre_repair_checkpoint_save_failed")
        initial_report["repair"] = repair
        pipeline._stage10_quality_repair_report = dict(repair)
        return initial_report

    try:
        candidate, candidate_metadata = _stage10_quality_repair_candidate(
            pipeline,
            original,
            initial_report,
        )
        repair["candidate"] = candidate_metadata
        _set_stage10_pixels(
            pipeline,
            candidate,
            label="Stage10 bounded final-quality repair",
        )
        if not pipeline._save_stage_output("stage10_quality_repair_candidate"):
            raise RuntimeError("quality repair candidate checkpoint save failed")
        reporter = getattr(pipeline, "_final_quality_report", None)
        if not callable(reporter):
            raise RuntimeError("final quality reporter unavailable for repair")
        candidate_report = reporter("stage10_quality_repair_candidate")
        if not isinstance(candidate_report, dict):
            raise TypeError("quality repair candidate report must be a mapping")
        structure = _stage10_quality_repair_structure_metrics(
            pipeline,
            original,
            candidate,
        )
        initial_risk = _stage10_quality_noise_risk(initial_report)
        candidate_risk = _stage10_quality_noise_risk(candidate_report)
        risk_improvement = (
            (initial_risk - candidate_risk) / max(initial_risk, 1e-9)
            if math.isfinite(initial_risk) and math.isfinite(candidate_risk)
            else float("-inf")
        )
        repair.update(
            candidate_severity=candidate_report.get("severity"),
            candidate_hard_issues=list(candidate_report.get("hard_issues") or []),
            initial_noise_risk=initial_risk,
            candidate_noise_risk=candidate_risk,
            risk_improvement=risk_improvement,
            structure=structure,
        )
        accepted = bool(
            not candidate_report.get("hard_issues")
            and str(candidate_report.get("severity") or "") != "hard_reject"
            and risk_improvement >= _STAGE10_QUALITY_RISK_IMPROVEMENT_MIN
            and structure["signal_luminance_correlation"]
            >= _STAGE10_QUALITY_SIGNAL_CORRELATION_MIN
            and _STAGE10_QUALITY_SIGNAL_FLUX_RATIO_MIN
            <= structure["signal_flux_ratio"]
            <= _STAGE10_QUALITY_SIGNAL_FLUX_RATIO_MAX
            and structure["core_clip_growth"]
            <= _STAGE10_QUALITY_CORE_CLIP_GROWTH_MAX
        )
        if not accepted:
            raise RuntimeError("quality repair candidate did not satisfy acceptance gates")
        if not pipeline._save_stage_output("stage10_final"):
            raise RuntimeError("accepted quality repair final checkpoint save failed")
        repair.update(status="accepted", reason="all_acceptance_gates_passed")
        pipeline._stage10_quality_repair_report = dict(repair)
        final_report = reporter("stage10_final")
        if not isinstance(final_report, dict):
            raise TypeError("accepted quality repair report must be a mapping")
        final_report["repair"] = dict(repair)
        return final_report
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as error:
        rollback_ok = False
        rollback_error = ""
        try:
            pipeline.cmd_with_check("load", "stage10_pre_quality_repair")
            rollback_ok = bool(pipeline._save_stage_output("stage10_final"))
        except (AttributeError, CommandError, RuntimeError, SirilError) as rollback_exc:
            rollback_error = str(rollback_exc)
        repair.update(
            status="rolled_back" if rollback_ok else "rollback_failed",
            reason=str(error),
            rollback_ok=rollback_ok,
            rollback_error=rollback_error or None,
        )
        pipeline._stage10_quality_repair_report = dict(repair)
        initial_report["repair"] = dict(repair)
        if not rollback_ok:
            hard_issues = list(initial_report.get("hard_issues") or [])
            hard_issues.append("stage10_quality_repair_rollback_failed")
            initial_report["hard_issues"] = list(dict.fromkeys(hard_issues))
            initial_report["issues"] = list(initial_report["hard_issues"])
            initial_report["severity"] = "hard_reject"
            initial_report["status"] = "needs_conservative_rerun"
            initial_report["final_quality"] = "poor"
            initial_report["needs_conservative_rerun"] = True
        return initial_report


def _write_stage10_color_rebalance_report(
    pipeline,
    *,
    pre_denoise: Optional[np.ndarray],
    pre_rebalance: Optional[np.ndarray],
    final_pixels: Optional[np.ndarray],
    requested_saturation: float,
    effective_saturation: float,
    applied_saturation: float,
    blocked_reason: str = "",
) -> Dict[str, Any]:
    """Record denoise and post-denoise color changes without gating delivery."""
    channel_profile = getattr(pipeline, "channel_profile", {}) or {}
    if not isinstance(channel_profile, dict) or not channel_profile:
        channel_profile = {
            "kind": str(
                getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
            )
        }
    contract = resolve_color_contract(
        channel_profile=channel_profile,
        color_report=getattr(pipeline, "color_calibration_report", {}) or {},
        palette_report=getattr(pipeline, "_stage8_palette_report", {}) or {},
    )
    unavailable = {
        "schema": "starun.color-quality-report.v1",
        "stage": "stage10",
        "status": "unavailable",
        "mode": "report_only",
        "used_for_gate": False,
        "contract": contract,
        "issues": ["Stage10 color measurement pixels are unavailable"],
    }
    baseline = pre_denoise if pre_denoise is not None else pre_rebalance
    if baseline is None or pre_rebalance is None or final_pixels is None:
        denoise_delta = dict(unavailable)
        rebalance_delta = dict(unavailable)
        end_to_end = dict(unavailable)
    else:
        denoise_delta = build_color_quality_report(
            baseline,
            pre_rebalance,
            stage="stage10",
            baseline_name="stage10_pre_denoise_memory",
            candidate_name="stage10_post_denoise_memory",
            contract=contract,
            operation="final_denoise_color_delta",
        )
        rebalance_delta = build_color_quality_report(
            pre_rebalance,
            final_pixels,
            stage="stage10",
            baseline_name="stage10_post_denoise_memory",
            candidate_name="stage10_final_memory",
            contract=contract,
            requested_saturation=requested_saturation,
            effective_saturation=effective_saturation,
            applied_saturation=applied_saturation,
            operation="post_denoise_budgeted_saturation",
        )
        end_to_end = build_color_quality_report(
            baseline,
            final_pixels,
            stage="stage10",
            baseline_name="stage10_pre_denoise_memory",
            candidate_name="stage10_final_memory",
            contract=contract,
            requested_saturation=requested_saturation,
            effective_saturation=effective_saturation,
            applied_saturation=applied_saturation,
            operation="denoise_and_final_rebalance_end_to_end",
        )

    ledger = list(getattr(pipeline, "_color_adjustment_ledger", []) or [])
    for delta in (denoise_delta, rebalance_delta):
        entry = delta.get("ledger_entry")
        if isinstance(entry, dict):
            ledger.append(dict(entry))
    pipeline._color_adjustment_ledger = ledger
    report = {
        "schema": "starun.stage10-color-rebalance.v1",
        "status": (
            "reported"
            if all(
                item.get("status") == "reported"
                for item in (denoise_delta, rebalance_delta, end_to_end)
            )
            else "unavailable"
        ),
        "mode": "report_only",
        "used_for_gate": False,
        "operation_order": [
            "final_denoise_or_safe_skip",
            "star_protected_merge_or_rollback",
            "budgeted_saturation_rebalance",
            "stage10_checkpoint_and_export",
        ],
        "decision": {
            "requested_saturation": round(float(requested_saturation), 7),
            "effective_saturation": round(float(effective_saturation), 7),
            "applied_saturation": round(float(applied_saturation), 7),
            "execution_blocked_reason": str(blocked_reason or "") or None,
            "automatic_chroma_loss_threshold_used": False,
            "reason": (
                "thresholds_require_validation-corpus calibration; current run "
                "uses only the existing Stage4 budget and Stage9 local-risk guard"
            ),
        },
        "contract": contract,
        "denoise_delta": denoise_delta,
        "post_denoise_rebalance_delta": rebalance_delta,
        "end_to_end_delta": end_to_end,
        "cross_stage_ledger": list(ledger),
    }
    pipeline._stage10_color_rebalance_report = dict(report)
    writer = getattr(pipeline, "_write_stage_json", None)
    if callable(writer):
        writer("stage10_color_rebalance_report.json", report)
    return report


def _stage10_config_choice(
    cfg: Any,
    field: str,
    default: str,
    allowed: Tuple[str, ...],
) -> str:
    value = str(getattr(cfg, field, default) or default).strip().lower()
    return value if value in allowed else default


def _stage10_record_policy_event(
    pipeline,
    *,
    action: str,
    event: str,
    reason: str,
    source: str,
) -> None:
    recorder = getattr(pipeline, "_record_stage_policy_event", None)
    if callable(recorder):
        recorder(10, event=event, reason=reason, source=source)
    writer = getattr(pipeline, "_write_stage_json", None)
    if callable(writer):
        writer(
            "stage10_failure_policy.json",
            {
                "schema": "starun.stage10-failure-policy.v1",
                "stage": 10,
                "failure_action": action,
                "event": event,
                "reason": reason,
                "source": source,
            },
        )


def _stage10_terminal_failure(
    pipeline,
    stage_label: str,
    messages: List[str],
    *,
    reason_code: str,
    reason: str,
    action: str,
    strict_stop: bool,
) -> None:
    """Write Stage 10 diagnostics/result and withhold every final export."""

    pipeline._final_output_review_only = True
    _stage10_record_policy_event(
        pipeline,
        action=action,
        event="strict_stop" if strict_stop else "output_withheld",
        reason=reason,
        source="stage10",
    )
    messages.append(f"Stage10 output withheld: {reason}")
    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(
        stage_label,
        "failed",
        elapsed,
        "；".join(messages),
        reason_code=reason_code,
        details={
            "failure_action": action,
            "strict_stop": bool(strict_stop),
            "final_export_generated": False,
        },
    )
    if strict_stop:
        raise RuntimeError(f"Stage 10 用户严格停止：{reason}")


def _stage10_star_visibility_reference(
    pipeline,
    final_pixels: np.ndarray,
    *,
    restore_stem: str,
    display_contract: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Resolve a dimension-checked frozen catalog for final star audits."""
    final_shape = _stage10_spatial_shape(final_pixels)
    primary = getattr(pipeline, "_stage9_star_reference_catalog", None)
    primary_is_source_confirmed = bool(
        isinstance(primary, dict)
        and primary.get("status") == "ok"
        and primary.get("source_matched") is True
        and primary.get("reference_degraded") is not True
    )
    if primary_is_source_confirmed:
        y = np.asarray(
            primary.get("_source_peak_y", primary.get("_peak_y", ())),
            dtype=np.int32,
        )
        x = np.asarray(
            primary.get("_source_peak_x", primary.get("_peak_x", ())),
            dtype=np.int32,
        )
        contrast = np.asarray(
            primary.get("_reference_local_contrast", ()),
            dtype=np.float32,
        )
        if (
            y.size > 0
            and x.size == y.size
            and contrast.size == y.size
            and np.all((y >= 0) & (y < final_shape[0]))
            and np.all((x >= 0) & (x < final_shape[1]))
        ):
            return primary, {
                "schema": "starun.stage10-star-catalog-resolution.v2",
                "status": "ready",
                "source": "stage9_frozen_star_reference",
                "star_count": int(y.size),
                "spatial_shape": list(final_shape),
                "audit_domain": "stage9_authenticated_display_domain",
            }

    stage5_report = getattr(pipeline, "_stage5_star_reference_report", None)
    resolution: Dict[str, Any] = {
        "schema": "starun.stage10-star-catalog-resolution.v2",
        "status": "unavailable",
        "source": "stage5_frozen_star_reference",
        "reason": "validated Stage9 catalog and signed Stage5 fallback are unavailable",
        "spatial_shape": list(final_shape),
    }
    if not (
        isinstance(stage5_report, dict)
        and stage5_report.get("schema") == "starun.stage5-star-reference.v1"
        and stage5_report.get("status") == "available"
        and stage5_report.get("fixed_before_deconvolution") is True
        and stage5_report.get("source_checkpoint") == "stage5_input_linear.fit"
    ):
        return None, resolution
    stars = [
        star
        for star in list(stage5_report.get("stars") or [])
        if isinstance(star, dict)
        and bool(star.get("geometry_valid", False))
        and star.get("x") is not None
        and star.get("y") is not None
        and star.get("fwhm_geometry") is not None
    ]
    try:
        source_path = pipeline.process_dir / "stage5_input_linear.fit"
        if not source_path.is_file():
            raise RuntimeError("signed Stage5 source checkpoint is missing")
        pipeline.cmd_with_check("load", "stage5_input_linear")
        source_pixels = pipeline.siril.get_image_pixeldata(preview=False)
        if source_pixels is None:
            raise RuntimeError("signed Stage5 source pixels are unavailable")
        source_shape = _stage10_spatial_shape(np.asarray(source_pixels))
        if source_shape != final_shape:
            raise RuntimeError(
                "Stage5/final spatial shape mismatch: "
                f"{source_shape}!={final_shape}"
            )
        source_reference_pixels = np.asarray(source_pixels)
        audit_domain = "native_stage5_linear_domain"
        display_contract_name = None
        selected_with_stars_review = bool(
            str(
                getattr(pipeline, "_stage7_candidate_domain", "") or ""
            )
            == "with_stars"
            and bool(
                str(
                    getattr(pipeline, "_stage7_stretch_selected", "") or ""
                )
            )
            and str(
                getattr(pipeline, "_stage7_review_source", "") or ""
            )
            == "stage7_review_with_stars"
        )
        if selected_with_stars_review:
            # The catalog geometry remains the immutable Stage 5 reference,
            # while visibility must be measured in the hard-gated nonlinear
            # with-stars domain that Stage 8/9/10 only pass through.
            source_reference_pixels = np.asarray(final_pixels)
            audit_domain = "selected_with_stars_review_domain"
        if display_contract is not None:
            if not display_rendition.validate_review_contract(display_contract):
                raise RuntimeError("review display contract is invalid")
            reference_contract = (
                dict(display_contract.get("tone_contract") or {})
                if display_contract.get("schema") == display_rendition.V3_SCHEMA
                else display_contract
            )
            source_reference_pixels = display_rendition.apply_review_contract(
                source_reference_pixels,
                reference_contract,
            )
            audit_domain = "shared_review_display_contract"
            display_contract_name = str(display_contract.get("name") or "") or None
        filtered = [
            star
            for star in stars
            if 0 <= int(round(float(star["y"]))) < final_shape[0]
            and 0 <= int(round(float(star["x"]))) < final_shape[1]
        ]
        if len(filtered) < 4:
            raise RuntimeError("Stage5 frozen catalog has insufficient valid stars")
        amplitudes = np.asarray(
            [
                float(star.get("amplitude"))
                if star.get("amplitude") is not None
                else float("-inf")
                for star in filtered
            ],
            dtype=np.float32,
        )
        finite_amplitudes = amplitudes[np.isfinite(amplitudes)]
        if finite_amplitudes.size < 4:
            raise RuntimeError("Stage5 frozen catalog lacks amplitude classification")
        amplitude_split = float(np.median(finite_amplitudes))
        fwhm = np.asarray(
            [float(star["fwhm_geometry"]) for star in filtered],
            dtype=np.float32,
        )
        y = np.asarray(
            [int(round(float(star["y"]))) for star in filtered],
            dtype=np.int32,
        )
        x = np.asarray(
            [int(round(float(star["x"]))) for star in filtered],
            dtype=np.int32,
        )
        catalog: Dict[str, Any] = {
            "status": "ok",
            "source_matched": True,
            "component_count": int(y.size),
            "source_reference": {
                "schema": "starun.stage5-star-reference.v1",
                "status": "available",
                "source_checkpoint": "stage5_input_linear.fit",
                "fixed_before_deconvolution": True,
            },
            "stage9_spatial_scale": {
                "status": "ready",
                "source": "stage5_fwhm_geometry",
                "fwhm_median_px": float(np.median(fwhm)),
                "anchor_fwhm_px": 4.0,
                "radius_scale": float(np.median(fwhm) / 4.0),
                "area_scale": float((np.median(fwhm) / 4.0) ** 2),
            },
            "_peak_y": y,
            "_peak_x": x,
            "_source_peak_y": y.copy(),
            "_source_peak_x": x.copy(),
            "_weak_flags": np.asarray(
                np.isfinite(amplitudes) & (amplitudes <= amplitude_split),
                dtype=bool,
            ),
            "_stage9_spatial_fwhm_px": fwhm,
        }
        catalog = stage9_quality.enrich_star_reference_with_display_psf(
            catalog,
            source_reference_pixels,
            pipeline.cfg,
        )
        contrast = np.asarray(
            catalog.get("_reference_local_contrast", ()),
            dtype=np.float32,
        )
        if contrast.size != y.size:
            raise RuntimeError("Stage5 visibility reference enrichment failed")
        contrast_min = min(
            max(
                float(
                    getattr(
                        pipeline.cfg,
                        "stage9_catalog_star_visibility_contrast_min",
                        0.002,
                    )
                ),
                0.0005,
            ),
            0.02,
        )
        source_visible = np.isfinite(contrast) & (contrast >= contrast_min)
        if int(np.count_nonzero(source_visible)) >= 4:
            visible_split = float(np.median(contrast[source_visible]))
            visibility_weak = source_visible & (contrast <= visible_split)
            catalog["_weak_flags"] = visibility_weak
            catalog.update(
                stage5_visibility_classification=(
                    "source_visible_local_contrast_median"
                ),
                stage5_visibility_contrast_split=visible_split,
                stage5_visibility_weak_count=int(
                    np.count_nonzero(visibility_weak)
                ),
                stage5_visibility_bright_count=int(
                    np.count_nonzero(source_visible & ~visibility_weak)
                ),
            )
        resolution.update(
            status="ready",
            source="stage5_frozen_star_reference",
            star_count=int(y.size),
            source_checkpoint="stage5_input_linear.fit",
            source_shape=list(source_shape),
            audit_domain=audit_domain,
            display_contract=display_contract_name,
            catalog_geometry_source="stage5_input_linear.fit",
            visibility_reference_source=(
                "selected_with_stars_review_candidate"
                if selected_with_stars_review
                else "stage5_input_linear.fit"
            ),
        )
        resolution.pop("reason", None)
        return catalog, resolution
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        resolution["reason"] = str(error)
        return None, resolution
    finally:
        try:
            pipeline.cmd_with_check("load", restore_stem)
        except (CommandError, SirilError):
            pass


def run_stage10_export(pipeline) -> None:
    """
    阶段 10: 最终降噪与导出
    - SCUNet / CosmicClarity 最终降噪（若可用）
    - 降噪后按剩余预算做最终色彩微调
    - 导出 TIFF/PNG/FITS
    """
    stage_label = PipelineStage.EXPORT.label
    pipeline.log.stage_start(stage_label)
    pipeline._clear_stage_reviews(10)
    status = "ok"
    messages: List[str] = []
    pipeline._stage10_color_rebalance_report = {}
    pipeline._stage10_quality_repair_report = {
        "attempted": False,
        "status": "not_requested",
    }
    pipeline._stage10_quality_frozen_background_masks = None
    pipeline._stage10_quality_frozen_background_sampling = None
    pipeline._stage10_quality_baseline_stem = ""
    stage10_background_support = (
        _stage10_freeze_authenticated_background_masks(pipeline)
    )
    pipeline._write_stage_json(
        "stage10_background_support.json",
        stage10_background_support,
    )
    messages.append(
        "authenticated_background_support="
        f"{stage10_background_support.get('status', 'rejected')}"
    )
    if stage10_background_support.get("accepted") is not True:
        pipeline._require_review(
            10,
            "stage10_authenticated_background_support_unavailable",
            details={
                "issues": list(stage10_background_support.get("issues") or [])
            },
        )
    pipeline._presentation_quality_report = presentation_quality.unavailable_presentation_report(
        "stage10_not_evaluated"
    )
    pipeline._presentation_quality_accepted = False
    pipeline._scientific_quality_accepted = False
    processing_mode = _stage10_config_choice(
        pipeline.cfg,
        "stage10_processing_mode",
        "auto",
        ("auto", "preserve"),
    )
    failure_action = _stage10_config_choice(
        pipeline.cfg,
        "stage10_failure_action",
        "auto_fallback",
        ("auto_fallback", "preserve_review", "stop"),
    )
    denoise_backend_policy = _stage10_config_choice(
        pipeline.cfg,
        "stage10_denoise_backend_policy",
        "auto_chain",
        ("auto_chain", "cosmic_only", "scunet_only"),
    )
    preserve_mode = processing_mode == "preserve"
    bright_core_fallback = dict(
        getattr(pipeline, "_bright_core_with_stars_fallback", {}) or {}
    )
    preserve_pixels = preserve_mode
    final_denoise_enabled = bool(
        getattr(pipeline.cfg, "stage10_final_denoise_enabled", True)
    )
    final_saturation_enabled = bool(
        getattr(pipeline.cfg, "stage10_final_saturation_enabled", True)
    )
    quality_repair_enabled = bool(
        getattr(pipeline.cfg, "stage10_quality_repair_enabled", True)
    )
    preserve_review_triggered = False
    failure_policy_reason = ""
    messages.append(
        "Stage10 policy "
        f"mode={processing_mode}; denoise_backend={denoise_backend_policy}; "
        f"failure_action={failure_action}"
    )
    stage9_contract_known = hasattr(pipeline, "_stage9_stars_applied")
    stage9_stars_required = bool(
        getattr(pipeline, "_stage9_stars_required", False)
    )
    stage9_stars_applied = bool(
        getattr(pipeline, "_stage9_stars_applied", False)
    )
    stage9_output_contains_stars = bool(
        getattr(
            pipeline,
            "_stage9_output_contains_stars",
            stage9_stars_applied,
        )
    )
    active_target_type = ""
    target_type_getter = getattr(pipeline, "_active_target_type", None)
    if callable(target_type_getter):
        try:
            active_target_type = str(target_type_getter() or "")
        except (AttributeError, RuntimeError, SirilError, TypeError, ValueError):
            active_target_type = ""
    stage9_missing_required_stars = bool(
        stage9_contract_known
        and stage9_stars_required
        and not stage9_stars_applied
    )
    stage9_starmask_stretch_failed = bool(
        getattr(pipeline, "_stage9_starmask_stretch_failed", False)
    )
    stage9_review_reasons = set(pipeline._stage_review_reasons(9))
    stage9_psf_review_required = bool(
        "stage9_psf_subgroup_evidence_insufficient" in stage9_review_reasons
    )
    stage9_review_candidate_selected = bool(
        "best_failed_candidate_review" in stage9_review_reasons
    )
    stage9_minimal_remix_fallback = bool(
        "stage8_starmask_review_fallback" in stage9_review_reasons
    )
    stage9_remix_formally_accepted = bool(
        getattr(pipeline, "_stage9_remix_formally_accepted", False)
    )
    stage4_core_color = dict(
        (getattr(pipeline, "color_calibration_report", {}) or {}).get(
            "bright_core_color_integrity"
        )
        or {}
    )
    if bool(stage4_core_color.get("applicable", False)) and str(
        stage4_core_color.get("status") or ""
    ) not in {"ok", "repaired"}:
        pipeline._require_review(
            4,
            "stage4_bright_core_color_integrity_unresolved",
        )
    stage4_color_review_required = bool(pipeline._stage_review_reasons(4))
    stage2_view_review_required = bool(pipeline._stage_review_reasons(2))
    stage3_background_review_required = bool(pipeline._stage_review_reasons(3))
    stage7_review_reasons = set(pipeline._stage_review_reasons(7))
    stage7_background_color_review_required = bool(
        getattr(
            pipeline,
            "_stage7_background_color_review_required",
            False,
        )
        or "uncalibrated_background_color_review_required"
        in stage7_review_reasons
    )
    stage7_forced_delivery = bool(
        getattr(pipeline, "_stage7_stretch_forced_delivery", False)
    )
    stage7_background_color_blocks_normal = bool(
        stage7_background_color_review_required
    )
    stage6_starmask_borderline_review_required = bool(
        "starmask_cleanup_borderline" in pipeline._stage_review_reasons(6)
    )
    stage6_quality_hard_failed_retained = bool(
        getattr(
            pipeline,
            "_stage6_quality_hard_failed_retained",
            False,
        )
    )
    forced_review_only = bool(
        getattr(pipeline.cfg, "force_review_only_output", False)
    )
    if stage9_missing_required_stars:
        pipeline._require_review(9, "stage9_required_stars_not_applied")
    if stage9_starmask_stretch_failed:
        pipeline._require_review(9, "stage9_starmask_stretch_failed")
    if stage9_contract_known and not stage9_remix_formally_accepted:
        pipeline._require_review(9, "stage9_remix_not_formally_accepted")
    if stage6_quality_hard_failed_retained:
        pipeline._require_review(6, "stage6_quality_hard_failed_retained")
    if stage7_forced_delivery:
        pipeline._require_review(
            7,
            "stage7_forced_quality_delivery_review_only",
        )
    if forced_review_only:
        pipeline._require_review(10, "forced_review_only_output")
    review_only_output = bool(pipeline._review_requirements_payload())
    final_source_review_reason = ""
    final_quality_gate_status = "pending"
    final_quality_gate_error = ""
    final_quality_reason_code = ""
    pipeline._final_output_review_only = False
    if stage9_missing_required_stars:
        messages.append(
            "stage9_stars_applied=false while stars_required=true; "
            "normal delivery is not allowed"
        )
    if stage9_starmask_stretch_failed:
        messages.append(
            "stage9_starmask_stretch_failed=true; normal delivery is not allowed"
        )
    if stage9_review_candidate_selected:
        messages.append(
            "stage9_review_candidate_selected=true while "
            "stage9_remix_formally_accepted="
            f"{str(stage9_remix_formally_accepted).lower()}; stars are present "
            "but the remix exceeded a formal quality gate, so normal delivery "
            "is not allowed"
        )
    elif stage9_psf_review_required:
        messages.append(
            "stage9_psf_review_required=true; PSF subgroup evidence is partial "
            "and normal delivery is not allowed"
        )
    elif stage9_review_reasons:
        messages.append(
            "stage9 review requirements disable normal delivery: "
            + ",".join(sorted(stage9_review_reasons))
        )
    if stage4_color_review_required:
        messages.append(
            "stage4_color_review_required=true; normal delivery is not allowed"
        )
    if stage2_view_review_required:
        messages.append(
            "stage2_view_review_required=true; normal delivery is not allowed"
        )
    if stage3_background_review_required:
        messages.append(
            "stage3_background_review_required=true; normal delivery is not allowed"
        )
    if stage7_background_color_review_required:
        if stage7_forced_delivery:
            messages.append(
                "stage7_uncalibrated_background_color_review_required=true; "
                "technically-safe forced delivery keeps normal names and "
                "partial_success; global white balance remains prohibited"
            )
        else:
            messages.append(
                "stage7_uncalibrated_background_color_review_required=true; "
                "normal delivery is not allowed; global white balance remains prohibited"
            )
    if stage6_starmask_borderline_review_required:
        messages.append(
            "stage6_starmask_diffuse_residual_borderline=true; "
            "normal delivery is not allowed"
        )
    if stage6_quality_hard_failed_retained:
        messages.append(
            "stage6_quality_hard_failed_retained=true; "
            "normal delivery is not allowed"
        )
    if forced_review_only:
        messages.append(
            "force_review_only_output=true; normal delivery names are disabled"
        )
    if (
        stage9_contract_known
        and stage9_stars_required
        and not stage9_output_contains_stars
    ):
        _stage10_record_policy_event(
            pipeline,
            action=failure_action,
            event="required_stars_output_withheld",
            reason="verified final source does not contain required stars",
            source="required_stars_contract",
        )
        messages.append(
            "required-stars final source is unavailable; no Starless-only "
            "candidate will be published as a final/review output"
        )
        pipeline._final_output_review_only = True
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            "failed",
            elapsed,
            "；".join(messages),
            reason_code="required_stars_output_withheld",
            details={
                "stars_required": True,
                "stars_applied": stage9_stars_applied,
                "output_contains_stars": False,
                "stage9_output_withheld": bool(
                    getattr(pipeline, "_stage9_output_withheld", False)
                ),
            },
        )
        if failure_action == "stop":
            raise RuntimeError(
                "Stage 10 用户严格停止：required-stars output withheld"
            )
        return

    # 按优先级加载最终图像
    final_file = "stage9_remixed"
    final_loaded = False
    preferred_final_source = str(
        getattr(pipeline, "_stage9_final_source", None) or final_file
    )
    verified_preserve_source = bool(
        stage9_contract_known and stage9_output_contains_stars
    )
    final_candidates = (
        [preferred_final_source]
        if preserve_pixels and verified_preserve_source
        else []
        if preserve_pixels
        else [
            preferred_final_source,
            final_file,
            "input_state_passthrough",
            "stage8_enhanced",
            pipeline.stretched_name or "stage7_stretched",
            "stage7_stretched",
        ]
    )
    for candidate in dict.fromkeys(
        str(item) for item in final_candidates if item
    ):
        candidate_path = pipeline.process_dir / f"{candidate}.fit"
        if not candidate_path.exists():
            messages.append(f"final_candidate_missing={candidate}")
            continue
        try:
            pipeline.cmd_with_check("load", candidate)
            final_file = candidate
            final_loaded = True
            break
        except (CommandError, SirilError):
            messages.append(f"final_candidate_load_failed={candidate}")
            continue
    if preserve_pixels and not final_loaded:
        reason = (
            "verified Stage9 with-stars source unavailable for pixel-preserve mode"
        )
        _stage10_terminal_failure(
            pipeline,
            stage_label,
            messages,
            reason_code="preserve_source_unavailable",
            reason=reason,
            action=failure_action,
            strict_stop=failure_action == "stop",
        )
        return
    if not final_loaded:
        status = "degraded"
        messages.append("最终候选图加载失败，沿用当前 Siril 图像")
    input_source_fallback_used = bool(
        final_loaded and final_file != preferred_final_source
    )
    if input_source_fallback_used:
        final_source_review_reason = "final_source_recovery_review_required"
        review_only_output = True
        messages.append(
            "final source recovered from a non-preferred checkpoint "
            f"({preferred_final_source}->{final_file}); fail-closed review-only output"
        )
        pipeline.log.warn(
            "Stage10 首选最终源不可用，已从较早检查点恢复；"
            "本轮只允许导出 result_review*"
        )
    elif not final_loaded:
        final_source_review_reason = "final_source_unavailable_review_required"
        review_only_output = True
        messages.append(
            "final source lineage unavailable; current Siril image retained "
            "only for fail-closed review output"
        )
        pipeline.log.warn(
            "Stage10 无法确认最终图像来源；当前 Siril 图像只允许"
            "导出为 result_review*"
        )
    if final_loaded:
        pipeline.log.info(f"使用最终图像: {final_file}")
        pipeline._stage10_quality_baseline_stem = final_file
    else:
        pipeline.log.warn("使用来源未确认的 Siril 当前图像生成复核产物")

    stage9_contract_state = _stage10_stage9_contract_state(
        pipeline,
        final_source=final_file,
        final_loaded=final_loaded,
    )
    stage9_contract_known = bool(stage9_contract_state["known"])
    if not stage9_contract_known:
        pipeline._require_review(9, "stage9_formal_contract_unavailable")
        review_only_output = True
        messages.append(
            "Stage9 formal delivery contract is unavailable for the loaded "
            "source; terminal denoise is disabled and only review output is "
            "allowed"
        )
    elif not bool(stage9_contract_state["formal"]):
        pipeline._require_review(9, "stage9_delivery_contract_not_formal")
        review_only_output = True
        messages.append(
            "Stage9 delivery contract is not formal for the loaded source; "
            "terminal denoise is disabled and only review output is allowed"
        )
    stage9_review_reasons = set(pipeline._stage_review_reasons(9))
    review_only_output = bool(
        review_only_output or pipeline._review_requirements_payload()
    )

    def handle_decisive_failure(
        reason: str,
        *,
        source: str,
        resave_checkpoint: bool = False,
    ) -> str:
        """Apply Stage 10 policy without ever accepting a failed candidate."""

        nonlocal review_only_output
        nonlocal status
        nonlocal final_file
        nonlocal final_loaded
        nonlocal final_source_review_reason
        nonlocal preserve_review_triggered
        nonlocal failure_policy_reason
        _stage10_record_policy_event(
            pipeline,
            action=failure_action,
            event="decisive_failure",
            reason=reason,
            source=source,
        )
        if failure_action == "auto_fallback":
            return "auto_fallback"
        failure_policy_reason = reason
        if failure_action == "stop":
            _stage10_terminal_failure(
                pipeline,
                stage_label,
                messages,
                reason_code="strict_stop",
                reason=reason,
                action=failure_action,
                strict_stop=True,
            )
        source_path = pipeline.process_dir / f"{preferred_final_source}.fit"
        if not (
            stage9_contract_known
            and stage9_output_contains_stars
            and source_path.is_file()
        ):
            _stage10_terminal_failure(
                pipeline,
                stage_label,
                messages,
                reason_code="preserve_review_source_unavailable",
                reason=(
                    f"{reason}; verified Stage9 with-stars rollback source unavailable"
                ),
                action=failure_action,
                strict_stop=False,
            )
            return "terminal"
        try:
            pipeline.cmd_with_check("load", preferred_final_source)
            if resave_checkpoint and not pipeline._save_stage_output(
                "stage10_final"
            ):
                raise RuntimeError("restored stage10_final checkpoint save failed")
        except (CommandError, RuntimeError, SirilError) as error:
            _stage10_terminal_failure(
                pipeline,
                stage_label,
                messages,
                reason_code="preserve_review_rollback_failed",
                reason=f"{reason}; Stage9 rollback failed: {error}",
                action=failure_action,
                strict_stop=False,
            )
            return "terminal"
        final_file = preferred_final_source
        final_loaded = True
        pipeline._stage10_quality_baseline_stem = preferred_final_source
        preserve_review_triggered = True
        review_only_output = True
        status = "degraded" if status == "ok" else status
        final_source_review_reason = "stage10_preserve_review"
        messages.append(
            "Stage10 preserve_review restored verified Stage9 with-stars source: "
            f"{reason}"
        )
        return "preserved"

    # Compute the budget now; execute the color command only after denoise.
    pipeline.log.info("计算降噪后最终色彩补偿预算...")
    color_limits = color_safety_limits(
        getattr(pipeline, "pipeline_policy", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    channel_semantics = str(
        getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
    )
    channel_color_adjustments_allowed = channel_semantics == BROADBAND_RGB_OSC
    requested_final_saturation = (
        0.0
        if (
            preserve_mode
            or not final_saturation_enabled
            or review_only_output
            or not stage9_contract_known
            or not bool(stage9_contract_state["formal"])
            or bool(
                getattr(pipeline, "_skip_stage10_color_adjustments", False)
            )
            or stage9_review_candidate_selected
            or stage9_minimal_remix_fallback
            or not channel_color_adjustments_allowed
        )
        else _config_value(
            pipeline.cfg,
            "final_saturation",
            0.15,
            0.0,
            0.25,
        )
    )
    if preserve_mode:
        messages.append("Stage10 preserve mode skipped final saturation")
    elif not final_saturation_enabled:
        messages.append("Stage10 final saturation disabled by task configuration")
    elif (
        review_only_output
        or not stage9_contract_known
        or not bool(stage9_contract_state["formal"])
    ):
        messages.append(
            "Stage10 final saturation skipped because the loaded source or "
            "run is review-only"
        )
    elif bool(getattr(pipeline, "_skip_stage10_color_adjustments", False)):
        messages.append(
            "Stage10 color adjustment skipped by input-state review guard"
        )
    elif stage9_review_candidate_selected:
        messages.append(
            "Stage10 color adjustment skipped because the Stage9 remix is a "
            "bounded review candidate rather than a formally accepted candidate"
        )
    elif stage9_minimal_remix_fallback:
        messages.append(
            "Stage10 color adjustment skipped because the Stage9 remix uses "
            "the minimal Stage8 + starmask review fallback"
        )
    elif not channel_color_adjustments_allowed:
        messages.append(
            "Stage10 global color adjustment skipped by channel semantics "
            f"({channel_semantics})"
        )
    color_budget_saturation = clamp_saturation_boost(
        requested_final_saturation,
        already_applied=float(getattr(pipeline, "_saturation_boost_applied", 0.0)),
        limits=color_limits,
    )
    if color_budget_saturation != requested_final_saturation:
        messages.append(
            "Stage4 color policy capped Stage10 saturation "
            f"{requested_final_saturation:.3f}->{color_budget_saturation:.3f} "
            f"(budget={color_limits['max_saturation_boost']:.3f})"
        )
    effective_final_saturation, stage9_color_guard = (
        _stage10_stage9_local_color_saturation_guard(
            pipeline,
            color_budget_saturation,
        )
    )
    pipeline._stage10_saturation_guard = dict(stage9_color_guard)
    if stage9_color_guard["applied"]:
        messages.append(
            "Stage9 local color risk capped Stage10 saturation "
            f"{color_budget_saturation:.3f}->{effective_final_saturation:.3f} "
            f"(risk={stage9_color_guard['local_color_risk_score']:.3f}, "
            f"factor={stage9_color_guard['saturation_factor']:.3f}, "
            f"reason={stage9_color_guard['reason']})"
        )
    stage_json_writer = getattr(pipeline, "_write_stage_json", None)
    if callable(stage_json_writer):
        try:
            stage_json_writer(
                "stage10_saturation_guard.json",
                stage9_color_guard,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            pipeline.log.warn(f"Stage10 saturation guard report failed: {error}")
    # The actual saturation command is deliberately deferred until the final
    # denoiser and the validated star-protection merge have completed.
    denoise_input_pixels, denoise_plan = _stage10_denoise_input(pipeline)
    selected_denoise_mode = str(denoise_plan["selected_mode"])
    cosmic_clarity_mode = str(denoise_plan["cosmic_clarity_mode"])
    final_denoise_strength = _config_value(
        getattr(pipeline, "cfg", None),
        "stage10_final_denoise_strength",
        _STAGE10_DENOISE_STRENGTH,
        0.05,
        0.50,
    )
    final_denoise_strength_text = f"{final_denoise_strength:.6g}"
    pipeline._cosmic_clarity_native_denoise_mode_override = cosmic_clarity_mode
    pipeline._cosmic_clarity_native_denoise_strength_override = (
        final_denoise_strength_text
    )
    denoise_plan["requested_strength"] = final_denoise_strength
    input_metrics = denoise_plan["input_metrics"]
    pipeline.log.info(
        "[Stage10] denoise mode selection "
        f"selected={selected_denoise_mode}, cosmic_clarity={cosmic_clarity_mode}, "
        f"strength={final_denoise_strength_text}, "
        f"chroma={float(input_metrics['chroma_noise_score']):.3f}, "
        f"bg_std={float(input_metrics['bg_std']):.5f}, "
        f"mottling={float(input_metrics['background_mottling_score']):.3f}"
    )
    messages.append(
        "Stage10 denoise mode "
        f"selected={selected_denoise_mode}, cosmic_clarity={cosmic_clarity_mode}, "
        f"reason={denoise_plan['reason']}"
    )

    final_denoise_used = None
    final_scunet_used = None
    denoise_primary = "CosmicClarity Denoise in-process script"
    denoise_primary_status = "skipped"
    denoise_effective = "none"
    denoise_effective_status = "skipped"
    denoise_fallback_used = False
    denoise_fallback_reason = ""
    duplicate_denoise_skip = bool(
        should_skip_final_denoise(
            stage5_denoise_applied=bool(
                getattr(pipeline, "_stage5_denoise_applied", False)
            ),
            stage8_final_quality=str(
                getattr(pipeline, "_stage8_final_quality", "unknown")
            ),
            stage8_fallback_used=bool(
                getattr(pipeline, "_stage8_fallback_used", False)
            ),
        )
    )
    review_only_denoise_skip = bool(review_only_output)
    low_noise_denoise_skip = selected_denoise_mode == "skip"
    preserve_mode_denoise_skip = bool(preserve_mode)
    configured_denoise_skip = bool(not final_denoise_enabled)
    skip_before_star_protection = (
        review_only_denoise_skip
        or duplicate_denoise_skip
        or low_noise_denoise_skip
        or preserve_mode_denoise_skip
        or configured_denoise_skip
    )
    star_protection_mask: Optional[np.ndarray] = None
    star_protection_report: Dict[str, Any] = {
        "schema": "starun.stage10-star-protection.v1",
        "status": "not_required",
        "reason": "denoise_skipped_by_existing_guard",
        "applied": False,
    }
    if not skip_before_star_protection:
        star_protection_mask, star_protection_report = (
            _build_stage10_star_protection_mask(
                pipeline,
                denoise_input_pixels,
            )
        )
        star_protection_report["applied"] = False
    star_protection_denoise_skip = bool(
        not skip_before_star_protection and star_protection_mask is None
    )
    skip_final_denoise = (
        skip_before_star_protection or star_protection_denoise_skip
    )
    denoise_plan["star_protection"] = star_protection_report
    denoise_plan["processing_mode"] = processing_mode
    denoise_plan["backend_policy"] = denoise_backend_policy
    denoise_plan["final_denoise_enabled"] = final_denoise_enabled
    denoise_plan["stage9_contract"] = dict(stage9_contract_state)
    if denoise_backend_policy == "scunet_only":
        final_denoise_script = None
        final_denoise_executable_args = None
    else:
        final_denoise_script = pipeline._find_plugin_script(
            ("processing/CosmicClarity_Denoise.py",)
        )
        final_denoise_executable_args = pipeline._classic_cosmic_clarity_args(
            "sirilcc_denoise.conf",
            "CosmicClarity Denoise",
        )
    if skip_final_denoise:
        if review_only_denoise_skip:
            denoise_primary = "review-only fast export guard"
            denoise_effective = "review source retained"
        elif duplicate_denoise_skip:
            denoise_primary = "duplicate-denoise guard"
            denoise_effective = "stage5 denoise retained"
        elif low_noise_denoise_skip:
            denoise_primary = "low-noise metric guard"
            denoise_effective = "low-noise input retained"
        elif preserve_mode_denoise_skip:
            denoise_primary = "preserve-mode pixel guard"
            denoise_effective = "verified Stage9 source retained"
        elif configured_denoise_skip:
            denoise_primary = "task denoise switch"
            denoise_effective = "Stage9 source retained"
        else:
            denoise_primary = "star-protection fail-closed guard"
            denoise_effective = "pre-denoise source retained"
        denoise_primary_status = "skipped"
        denoise_effective_status = "skipped_safe"
    elif denoise_backend_policy == "scunet_only":
        denoise_primary = "Siril-SCUNet Denoise"
        final_scunet_used = pipeline._run_siril_scunet_denoise_fallback(
            "最终降噪",
            final_denoise_strength,
        )
        if final_scunet_used:
            denoise_primary_status = "success"
            denoise_effective = final_scunet_used
            denoise_effective_status = "success"
        else:
            denoise_primary_status = "failed"
            denoise_effective_status = "failed"
    elif final_denoise_script is not None and final_denoise_executable_args is not None:
        cli_args: List[str] = [
            "-denoising_mode",
            cosmic_clarity_mode,
            "-denoise_strength",
            final_denoise_strength_text,
            "-use_gpu",
            *final_denoise_executable_args,
        ]

        final_denoise_used = pipeline._run_plugin_script_by_path(
            "最终降噪",
            "CosmicClarity Denoise",
            final_denoise_script,
            args=tuple(cli_args),
        )
        if final_denoise_used:
            denoise_primary_status = "success"
            denoise_effective = final_denoise_used
            denoise_effective_status = "success"
        else:
            denoise_primary_status = "failed"
            script_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or final_denoise_script.name
            )
            cli_denoise_used = pipeline._run_plugin_script_cli_subprocess(
                "最终降噪",
                "CosmicClarity Denoise",
                final_denoise_script,
                args=tuple(cli_args),
                timeout_sec=pipeline._final_denoise_cli_timeout_sec(),
            )
            if cli_denoise_used:
                final_denoise_used = cli_denoise_used
                denoise_effective = cli_denoise_used
                denoise_effective_status = "success"
                denoise_fallback_used = True
                denoise_fallback_reason = "inprocess_to_cli"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Denoise in-process",
                        script_error,
                        cli_denoise_used,
                        True,
                    )
                )
            else:
                cli_error = (
                    getattr(pipeline, "_last_plugin_script_error", None)
                    or final_denoise_script.name
                )
                native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
                    "最终降噪回退"
                )
                if native_denoise_used:
                    final_denoise_used = native_denoise_used
                    denoise_effective = native_denoise_used
                    denoise_effective_status = "success"
                    denoise_fallback_used = True
                    denoise_fallback_reason = "cli_to_native"
                    messages.append(
                        pipeline._fallback_summary(
                            "CosmicClarity Denoise CLI subprocess",
                            cli_error,
                            native_denoise_used,
                            True,
                        )
                    )
                else:
                    native_error = (
                        getattr(pipeline, "_last_plugin_script_error", None)
                        or "CosmicClarity_Native.py unavailable"
                    )
                    final_scunet_used = (
                        _run_stage10_scunet_fallback(
                            pipeline,
                            "最终降噪回退",
                            final_denoise_strength,
                        )
                        if denoise_backend_policy == "auto_chain"
                        else None
                    )
                    if final_scunet_used:
                        denoise_effective = final_scunet_used
                        denoise_effective_status = "success"
                        denoise_fallback_used = True
                        denoise_fallback_reason = "native_to_scunet"
                        messages.append(
                            pipeline._fallback_summary(
                                "CosmicClarity Native Denoise",
                                native_error,
                                final_scunet_used,
                                True,
                            )
                        )
                    else:
                        scunet_reason = getattr(
                            pipeline,
                            "_last_scunet_fallback_error",
                            None,
                        )
                        messages.append(
                            pipeline._fallback_summary(
                                "CosmicClarity Native Denoise",
                                native_error,
                                "Siril-SCUNet Denoise",
                                False,
                            )
                        )
                        if scunet_reason:
                            messages.append(
                                f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}"
                            )
                        else:
                            messages.append("Siril-SCUNet Denoise 回退不可用")
    elif final_denoise_script is not None:
        denoise_primary = "CosmicClarity Native Denoise cli-subprocess"
        pipeline.log.info(
            "CosmicClarity Denoise classic 路径未启用，使用 Native Denoise"
        )
        native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
            "最终降噪"
        )
        if native_denoise_used:
            final_denoise_used = native_denoise_used
            denoise_effective = native_denoise_used
            denoise_primary_status = "success"
            denoise_effective_status = "success"
            messages.append("CosmicClarity classic 路径未启用，已选择 Native Denoise")
        else:
            denoise_primary_status = "failed"
            native_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or "CosmicClarity_Native.py unavailable"
            )
            final_scunet_used = (
                _run_stage10_scunet_fallback(
                    pipeline,
                    "最终降噪回退",
                    final_denoise_strength,
                )
                if denoise_backend_policy == "auto_chain"
                else None
            )
            if final_scunet_used:
                denoise_effective = final_scunet_used
                denoise_effective_status = "success"
                denoise_fallback_used = True
                denoise_fallback_reason = "native_to_scunet"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        final_scunet_used,
                        True,
                    )
                )
            else:
                scunet_reason = getattr(pipeline, "_last_scunet_fallback_error", None)
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        "Siril-SCUNet Denoise",
                        False,
                    )
                )
                if scunet_reason:
                    messages.append(f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}")
    else:
        denoise_primary_status = "missing"
        native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
            "最终降噪回退"
        )
        if native_denoise_used:
            final_denoise_used = native_denoise_used
            denoise_effective = native_denoise_used
            denoise_effective_status = "success"
            denoise_fallback_used = True
            denoise_fallback_reason = "script_missing_to_native"
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity_Denoise.py",
                    "script missing",
                    native_denoise_used,
                    True,
                )
            )
        else:
            native_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or "CosmicClarity_Native.py unavailable"
            )
            final_scunet_used = (
                _run_stage10_scunet_fallback(
                    pipeline,
                    "最终降噪回退",
                    final_denoise_strength,
                )
                if denoise_backend_policy == "auto_chain"
                else None
            )
            if final_scunet_used:
                denoise_effective = final_scunet_used
                denoise_effective_status = "success"
                denoise_fallback_used = True
                denoise_fallback_reason = "script_missing_to_scunet"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        final_scunet_used,
                        True,
                    )
                )
            else:
                scunet_reason = getattr(pipeline, "_last_scunet_fallback_error", None)
                if scunet_reason:
                    messages.append(
                        pipeline._fallback_summary(
                            "CosmicClarity Native Denoise",
                            native_error,
                            "Siril-SCUNet Denoise",
                            False,
                        )
                    )
                    messages.append(f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}")

    if review_only_denoise_skip:
        pipeline.log.info(
            "Stage10 review-only output skips expensive final denoise"
        )
        messages.append(
            "Stage10 review-only fast export skipped final denoise"
        )
    elif duplicate_denoise_skip:
        pipeline.log.info("Stage5 降噪已通过后续质量门，跳过 Stage10 重复降噪")
        messages.append(
            "Stage10 duplicate-denoise guard skipped final denoise "
            f"(stage8_quality={getattr(pipeline, '_stage8_final_quality', 'unknown')})"
        )
    elif low_noise_denoise_skip:
        pipeline.log.info(
            "Stage10 输入噪声指标均低于门限，跳过昂贵的最终降噪"
        )
        messages.append(
            "Stage10 low-noise guard skipped final denoise "
            f"(chroma={float(input_metrics['chroma_noise_score']):.3f}, "
            f"bg_std={float(input_metrics['bg_std']):.5f}, "
            "mottling="
            f"{float(input_metrics['background_mottling_score']):.3f})"
        )
    elif preserve_mode_denoise_skip:
        pipeline.log.info(
            "Stage10 preserve 模式保留已验证 Stage9 含星源，跳过末端降噪"
        )
        messages.append("Stage10 preserve mode retained Stage9 pixels")
    elif configured_denoise_skip:
        pipeline.log.info("Stage10 末端降噪已由任务配置关闭")
        messages.append("Stage10 final denoise disabled by task configuration")
    elif star_protection_denoise_skip:
        protection_reason = str(
            star_protection_report.get("reason")
            or "validated Stage9 star mask unavailable"
        )
        pipeline.log.warn(
            "Stage10 星点保护 mask 不可用，安全跳过末端降噪: "
            f"{protection_reason}"
        )
        messages.append(
            "Stage10 star-protection guard skipped final denoise; "
            f"reason={protection_reason}"
        )
    elif final_denoise_used:
        pipeline.log.info("已执行最终降噪（作为最后一步处理）")
        messages.append(f"最终降噪使用 {final_denoise_used}")
    elif final_scunet_used:
        pipeline.log.info("已执行 Siril-SCUNet 最终降噪（代码回退）")
        messages.append(f"最终降噪使用 {final_scunet_used}")
    elif (
        denoise_backend_policy == "auto_chain"
        and getattr(pipeline.cfg, "aberration_api_enabled", False)
    ):
        pipeline.log.warn("最终降噪脚本不可用，尝试 Aberration API 作为回退")
        local_model = pipeline._resolve_local_aberration_model()
        aberration_used = pipeline._run_aberration_api("最终降噪", model_path=local_model)
        if aberration_used:
            denoise_effective = aberration_used
            denoise_effective_status = "success"
            denoise_fallback_used = True
            denoise_fallback_reason = "denoiser_chain_to_aberration"
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity/SCUNet final denoise",
                    "previous denoise candidates unavailable",
                    aberration_used,
                    True,
                )
            )
        else:
            pipeline.log.warn("最终降噪脚本与 Aberration API 均不可用，跳过最终降噪")
            if pipeline._last_aberration_api_error:
                messages.append(
                    "Aberration API 不可用: "
                    f"{pipeline._short_text(pipeline._last_aberration_api_error, 160)}"
                )
            messages.append("最终降噪脚本与 Aberration API 均不可用")
            denoise_effective_status = "failed"
            status = "degraded" if status == "ok" else status
    else:
        pipeline.log.info("最终降噪脚本不可用，且 Aberration API 默认关闭，跳过最终降噪")
        messages.append("最终降噪未执行（script/scunet unavailable, Aberration API disabled）")
        status = "degraded" if status == "ok" else status

    effective_denoise_mode = "skipped"
    if denoise_effective_status == "success":
        if selected_denoise_mode == "chroma":
            chroma_applied, chroma_note = _apply_stage10_chroma_only_result(
                pipeline,
                denoise_input_pixels,
            )
            if chroma_applied:
                effective_denoise_mode = "chroma"
                messages.append(f"Stage10 chroma-only merge: {chroma_note}")
            else:
                effective_denoise_mode = "full_fallback"
                denoise_fallback_used = True
                denoise_fallback_reason = (
                    denoise_fallback_reason or "chroma_merge_to_full"
                )
                pipeline.log.warn(
                    "Stage10 chroma-only merge unavailable; retaining full denoise: "
                    f"{chroma_note}"
                )
                messages.append(
                    "Stage10 chroma-only merge failed; retained full denoise: "
                    f"{chroma_note}"
                )
        elif selected_denoise_mode == "separate" and final_denoise_used:
            effective_denoise_mode = "separate"
        elif selected_denoise_mode == "separate":
            effective_denoise_mode = "full_fallback"
            messages.append(
                "Stage10 separate mode unavailable on the effective fallback denoiser; "
                "retained fallback output"
            )
        else:
            effective_denoise_mode = "full"

        protection_applied, protection_note = (
            _apply_stage10_star_protected_result(
                pipeline,
                denoise_input_pixels,
                star_protection_mask,
            )
        )
        star_protection_report["applied"] = bool(protection_applied)
        star_protection_report["apply_note"] = protection_note
        if protection_applied:
            messages.append(f"Stage10 star-protected denoise merge: {protection_note}")
        else:
            rollback_ok, rollback_note = _rollback_stage10_denoise(
                pipeline,
                denoise_input_pixels,
            )
            star_protection_report["rollback_status"] = (
                "success" if rollback_ok else "failed"
            )
            star_protection_report["rollback_note"] = rollback_note
            denoise_fallback_used = True
            denoise_fallback_reason = "star_protection_merge_rollback"
            if rollback_ok:
                denoise_effective = "pre-denoise source retained"
                denoise_effective_status = "rolled_back_safe"
                effective_denoise_mode = "skipped"
                pipeline.log.warn(
                    "Stage10 星点保护回混失败，已恢复降噪前输入: "
                    f"{protection_note}"
                )
                messages.append(
                    "Stage10 star-protection merge failed; frozen input restored: "
                    f"{protection_note}"
                )
            else:
                denoise_effective_status = "protection_failed"
                effective_denoise_mode = "unsafe_review"
                review_only_output = True
                status = "degraded" if status == "ok" else status
                pipeline.log.error(
                    "Stage10 星点保护回混与回滚均失败，仅允许 review-only 导出: "
                    f"merge={protection_note}; rollback={rollback_note}"
                )
                messages.append(
                    "Stage10 star-protection merge and rollback failed; "
                    "forced review-only output"
                )

    if (
        not skip_final_denoise
        and denoise_effective_status
        not in {"success", "rolled_back_safe", "protection_failed"}
    ):
        denoise_effective_status = "failed"

    denoise_plan.update(
        {
            "effective_mode": effective_denoise_mode,
            "effective_component": denoise_effective,
            "effective_status": denoise_effective_status,
            "skipped_by_duplicate_guard": bool(duplicate_denoise_skip),
            "skipped_by_review_only": bool(review_only_denoise_skip),
            "skipped_by_low_noise_guard": bool(low_noise_denoise_skip),
            "skipped_by_preserve_mode": bool(preserve_mode_denoise_skip),
            "skipped_by_task_switch": bool(configured_denoise_skip),
            "skipped_by_star_protection_guard": bool(
                star_protection_denoise_skip
            ),
            "fallback_used": bool(denoise_fallback_used),
            "fallback_reason": denoise_fallback_reason or None,
        }
    )
    stage_json_writer = getattr(pipeline, "_write_stage_json", None)
    if callable(stage_json_writer):
        try:
            stage_json_writer("stage10_denoise_plan.json", denoise_plan)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            pipeline.log.warn(f"Stage10 denoise plan report failed: {error}")
            messages.append("stage10_denoise_plan.json 写入失败")

    if denoise_effective_status in {"failed", "protection_failed"}:
        policy_outcome = handle_decisive_failure(
            "all final denoise candidates failed"
            if denoise_effective_status == "failed"
            else "star-protection merge and rollback failed",
            source="final_denoise",
        )
        if policy_outcome == "terminal":
            return
        if policy_outcome == "preserved":
            denoise_effective = "verified Stage9 source restored"
            denoise_effective_status = "rolled_back_safe"
            effective_denoise_mode = "skipped"
            denoise_fallback_used = True
            denoise_fallback_reason = "preserve_review"
            denoise_plan.update(
                {
                    "effective_mode": effective_denoise_mode,
                    "effective_component": denoise_effective,
                    "effective_status": denoise_effective_status,
                    "fallback_used": True,
                    "fallback_reason": "preserve_review",
                }
            )
            if callable(stage_json_writer):
                stage_json_writer("stage10_denoise_plan.json", denoise_plan)

    messages.append(
        f"final_denoise_primary={denoise_primary}; "
        f"primary_status={denoise_primary_status}; "
        f"final_denoise_effective={denoise_effective}; "
        f"effective_status={denoise_effective_status}; "
        f"selected_mode={selected_denoise_mode}; "
        f"effective_mode={effective_denoise_mode}"
    )

    pre_rebalance_pixels = _stage10_current_pixels(pipeline)
    stage10_applied_saturation = 0.0
    stage10_saturation_failed = False
    stage10_color_rebalance_blocked_reason = (
        "review_only_output"
        if review_only_output
        else "preserve_review"
        if preserve_review_triggered
        else "denoise_safety_failure"
        if denoise_effective_status == "protection_failed"
        else ""
    )
    if (
        abs(effective_final_saturation) > 1e-8
        and not stage10_color_rebalance_blocked_reason
        and stage9_contract_known
        and bool(stage9_contract_state["formal"])
    ):
        try:
            pipeline.log.info("最终降噪后执行预算内色彩补偿...")
            pipeline.cmd_with_check(
                "satu",
                f"{effective_final_saturation:.6f}",
                str(
                    int(
                        round(
                            _config_value(
                                pipeline.cfg,
                                "final_bg_factor",
                                1.0,
                                0.0,
                                10.0,
                            )
                        )
                    )
                ),
            )
            stage10_applied_saturation = float(effective_final_saturation)
            pipeline._saturation_boost_applied = float(
                getattr(pipeline, "_saturation_boost_applied", 0.0)
            ) + max(0.0, stage10_applied_saturation)
            messages.append(
                "Stage10 applied final saturation after denoise and "
                "star-protected merge"
            )
        except (CommandError, SirilError) as e:
            stage10_saturation_failed = True
            pipeline.log.warn(f"最终饱和度调整跳过: {e}")
            status = "degraded"
            messages.append(f"最终饱和度调整失败: {e}")
    elif stage10_color_rebalance_blocked_reason:
        messages.append(
            "Stage10 saturation skipped after denoise protection/rollback failure"
        )
    elif stage9_color_guard["applied"]:
        messages.append(
            "Stage10 saturation skipped by Stage9 local color risk guard"
        )
    elif requested_final_saturation == 0.0:
        messages.append("Stage10 saturation skipped by color-operation policy")
    else:
        messages.append("Stage10 saturation skipped: Stage4 color budget exhausted")

    final_color_pixels = _stage10_current_pixels(pipeline)
    try:
        stage10_color_report = _write_stage10_color_rebalance_report(
            pipeline,
            pre_denoise=denoise_input_pixels,
            pre_rebalance=pre_rebalance_pixels,
            final_pixels=final_color_pixels,
            requested_saturation=requested_final_saturation,
            effective_saturation=effective_final_saturation,
            applied_saturation=stage10_applied_saturation,
            blocked_reason=stage10_color_rebalance_blocked_reason,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        stage10_color_report = {
            "schema": "starun.stage10-color-rebalance.v1",
            "status": "unavailable",
            "mode": "report_only",
            "used_for_gate": False,
            "issues": [str(error)],
        }
        pipeline._stage10_color_rebalance_report = dict(stage10_color_report)
        pipeline.log.warn(f"Stage10 color rebalance report failed: {error}")
        messages.append("stage10_color_rebalance_report.json 写入失败")

    if stage10_saturation_failed:
        policy_outcome = handle_decisive_failure(
            "final saturation command failed",
            source="final_saturation",
        )
        if policy_outcome == "terminal":
            return

    final_quality: Dict[str, Any] = {}
    stage_saved = pipeline._save_stage_output("stage10_final")
    if not stage_saved and status == "ok":
        status = "degraded"
        messages.append("stage10 输出保存失败")
    if not stage_saved:
        final_quality_gate_status = "checkpoint_unavailable"
        final_quality_gate_error = "stage10_final checkpoint save failed"
        final_quality_reason_code = "final_quality_checkpoint_unavailable"
        review_only_output = True
        messages.append(
            "final quality gate unavailable because stage10_final.fit was not saved; "
            "fail-closed review-only output"
        )
        pipeline.log.warn(
            "Stage10 规范检查点保存失败，无法完成最终质量门；"
            "本轮只允许导出 result_review*"
        )
        policy_outcome = handle_decisive_failure(
            "stage10_final checkpoint save failed",
            source="checkpoint",
            resave_checkpoint=True,
        )
        if policy_outcome == "terminal":
            return
    elif stage_saved:
        diff_note = pipeline._stage_diff_note("stage10_final", final_file)
        if diff_note:
            messages.append(diff_note)
        if hasattr(pipeline, "_create_stage_review_bundle"):
            review = pipeline._create_stage_review_bundle(
                "stage10_export",
                final_file,
                "stage10_final",
                context={
                    "final_denoise_skipped": skip_final_denoise,
                    "color_policy_limits": color_limits,
                    "effective_final_saturation": effective_final_saturation,
                    "channel_semantics": channel_semantics,
                    "target_type": active_target_type or "unknown",
                    "stage9_local_color_saturation_guard": stage9_color_guard,
                    "denoise_plan": denoise_plan,
                    "color_rebalance_report": stage10_color_report,
                },
            )
            if review.get("report_path"):
                messages.append(f"review_bundle={review['report_path']}")
        feature_note = pipeline._feature_summary_note("最终导出前特征")
        if feature_note:
            messages.append(feature_note)
        final_quality_reporter = getattr(pipeline, "_final_quality_report", None)
        if callable(final_quality_reporter):
            try:
                final_quality = final_quality_reporter("stage10_final")
                if not isinstance(final_quality, dict):
                    raise TypeError("final quality report must be a mapping")
                quality_repair_source_trusted = bool(
                    not review_only_output
                    and final_loaded
                    and not input_source_fallback_used
                    and final_file == preferred_final_source
                    and stage9_contract_known
                    and (not stage9_stars_required or stage9_stars_applied)
                    and not bool(
                        getattr(pipeline, "_stage8_fallback_used", False)
                    )
                    and not bool(
                        getattr(pipeline, "_stage9_bypassed_bad_starless", False)
                    )
                    and not stage9_missing_required_stars
                    and not bool(
                        getattr(
                            pipeline,
                            "_stage9_starmask_preparation_failed",
                            False,
                        )
                    )
                    and not stage9_starmask_stretch_failed
                )
                if (
                    quality_repair_enabled
                    and not preserve_pixels
                    and not preserve_review_triggered
                ):
                    final_quality = _attempt_stage10_quality_repair(
                        pipeline,
                        final_quality,
                        source_trusted=quality_repair_source_trusted,
                    )
                else:
                    repair_reason = (
                        "preserve_mode"
                        if preserve_mode
                        else "preserve_review"
                        if preserve_review_triggered
                        else "disabled_by_task_configuration"
                    )
                    pipeline._stage10_quality_repair_report = {
                        "attempted": False,
                        "status": "disabled",
                        "reason": repair_reason,
                    }
                    final_quality["repair"] = dict(
                        pipeline._stage10_quality_repair_report
                    )
                repair_record = final_quality.get("repair") or {}
                if isinstance(repair_record, dict) and repair_record.get("attempted"):
                    messages.append(
                        "stage10_quality_repair="
                        f"{repair_record.get('status', 'unknown')}"
                    )
                    if (
                        repair_record.get("status") == "accepted"
                        and hasattr(pipeline, "_create_stage_review_bundle")
                    ):
                        repair_review = pipeline._create_stage_review_bundle(
                            "stage10_quality_repair",
                            "stage10_pre_quality_repair",
                            "stage10_final",
                            context={
                                "repair": repair_record,
                                "target_type": active_target_type or "unknown",
                            },
                        )
                        if repair_review.get("report_path"):
                            messages.append(
                                f"quality_repair_review_bundle={repair_review['report_path']}"
                            )
                final_quality_value = str(
                    final_quality.get("final_quality", "") or ""
                ).strip().lower()
                final_quality_status = str(
                    final_quality.get("status", "") or ""
                ).strip().lower()
                needs_conservative_rerun = final_quality.get(
                    "needs_conservative_rerun"
                )
                issues = final_quality.get("issues")
                if not isinstance(needs_conservative_rerun, bool):
                    raise TypeError(
                        "final quality report needs_conservative_rerun must be bool"
                    )
                if not isinstance(issues, list):
                    raise TypeError("final quality report issues must be a list")
                quality_requires_review = bool(
                    needs_conservative_rerun
                    or final_quality_value != "ok"
                    or final_quality_status != "ok"
                    or issues
                )
                final_quality_gate_status = (
                    "review_required" if quality_requires_review else "ok"
                )
                pipeline._write_stage_json("final_quality_report.json", final_quality)
                pipeline.log.info(
                    "[Stage10] final_quality="
                    f"{final_quality.get('final_quality')} "
                    f"status={final_quality.get('status')} "
                    f"needs_conservative_rerun={str(bool(final_quality.get('needs_conservative_rerun'))).lower()}"
                )
                messages.append(
                    "final_quality="
                    f"{final_quality.get('final_quality')} "
                    f"status={final_quality.get('status')}"
                )
                if quality_requires_review:
                    review_only_output = True
                    final_quality_reason_code = "final_quality_requires_review"
                    status = "degraded" if status == "ok" else status
                    if issues:
                        issue_text = ", ".join(str(x) for x in issues[:2])
                        pipeline.log.warn(f"[Stage10] final_quality_issues={issue_text}")
                        messages.append("final_quality_issues=" + issue_text)
                    else:
                        messages.append(
                            "final quality report contract is not fully ok; "
                            "fail-closed review-only output"
                        )
                    policy_outcome = handle_decisive_failure(
                        "final quality gate rejected Stage10 candidate",
                        source="final_quality_gate",
                        resave_checkpoint=True,
                    )
                    if policy_outcome == "terminal":
                        return
            except (
                AttributeError,
                CommandError,
                OSError,
                RuntimeError,
                SirilError,
                TypeError,
                ValueError,
            ) as e:
                final_quality_gate_status = "unavailable"
                final_quality_gate_error = str(e)
                final_quality_reason_code = "final_quality_gate_unavailable"
                review_only_output = True
                pipeline.log.warn(f"final quality gate unavailable: {e}")
                messages.append(
                    "final quality gate unavailable; fail-closed review-only output: "
                    f"{e}"
                )
                status = "degraded" if status == "ok" else status
                policy_outcome = handle_decisive_failure(
                    "final quality gate unavailable",
                    source="final_quality_gate",
                    resave_checkpoint=True,
                )
                if policy_outcome == "terminal":
                    return
        else:
            final_quality_gate_status = "unavailable"
            final_quality_gate_error = "final quality reporter unavailable"
            final_quality_reason_code = "final_quality_gate_unavailable"
            review_only_output = True
            messages.append(
                "final quality reporter unavailable; fail-closed review-only output"
            )
            pipeline.log.warn(
                "Stage10 最终质量门实现不可用；本轮只允许导出 "
                "result_review*"
            )
            policy_outcome = handle_decisive_failure(
                "final quality reporter unavailable",
                source="final_quality_gate",
                resave_checkpoint=True,
            )
            if policy_outcome == "terminal":
                return

    pipeline._scientific_quality_accepted = bool(
        final_quality_gate_status == "ok"
        and str(final_quality.get("status") or "").strip().lower() == "ok"
        and str(final_quality.get("final_quality") or "").strip().lower()
        == "ok"
        and final_quality.get("needs_conservative_rerun") is False
        and not list(final_quality.get("issues") or [])
    )
    presentation_reason_code = ""
    if stage_saved and pipeline._scientific_quality_accepted:
        try:
            final_presentation_pixels = pipeline._read_image_by_stem(
                "stage10_final"
            )
            stage7_presentation_pixels = pipeline._read_image_by_stem(
                "stage7_presentation_reference"
            )
            if final_presentation_pixels is None:
                raise RuntimeError("stage10_final pixels unavailable")
            if stage7_presentation_pixels is None:
                raise RuntimeError(
                    "stage7_presentation_reference pixels unavailable"
                )
            reference_verification = (
                presentation_quality.verify_stage7_presentation_reference(
                    getattr(
                        pipeline,
                        "_stage7_presentation_reference_report",
                        {},
                    ),
                    stage7_presentation_pixels,
                    Path(pipeline.process_dir)
                    / "stage7_presentation_reference.fit",
                )
            )
            profile_name = ""
            profile_getter = getattr(
                pipeline,
                "_stage7_target_stretch_profile",
                None,
            )
            if callable(profile_getter):
                profile_record = profile_getter()
                if isinstance(profile_record, dict):
                    profile_name = str(profile_record.get("name") or "")
            stars_application_mode = str(
                getattr(pipeline, "_stage9_stars_application_mode", "") or ""
            )
            stars_not_required_verified = bool(
                not stage9_stars_required
                and stage9_remix_formally_accepted
                and stars_application_mode == "stars_not_required"
                and not pipeline._stage_review_reasons(9)
            )
            presentation_report = (
                presentation_quality.build_presentation_quality_report(
                    stage7_presentation_pixels,
                    final_presentation_pixels,
                    getattr(
                        pipeline,
                        "_stage7_frozen_rendition_masks",
                        None,
                    ),
                    pipeline.cfg,
                    target_type=active_target_type,
                    profile_name=profile_name,
                    stage9_quality=getattr(
                        pipeline,
                        "_stage9_selected_remix_quality",
                        None,
                    ),
                    stars_required=stage9_stars_required,
                    stars_not_required_verified=(
                        stars_not_required_verified
                    ),
                    scientific_report=final_quality,
                )
            )
            presentation_report["reference_verification"] = (
                reference_verification
            )
            pipeline.cmd_with_check("load", "stage10_final")
        except (
            AttributeError,
            CommandError,
            OSError,
            RuntimeError,
            SirilError,
            TypeError,
            ValueError,
        ) as error:
            presentation_report = (
                presentation_quality.unavailable_presentation_report(
                    str(error)
                )
            )
            try:
                pipeline.cmd_with_check("load", "stage10_final")
            except (CommandError, SirilError):
                pass
    else:
        presentation_report = presentation_quality.unavailable_presentation_report(
            "scientific_quality_gate_not_accepted"
        )
    pipeline._presentation_quality_report = dict(presentation_report)
    pipeline._presentation_quality_accepted = bool(
        presentation_report.get("accepted", False)
    )
    pipeline._write_stage_json(
        "presentation_quality_report.json",
        presentation_report,
    )
    messages.append(
        "presentation_quality="
        f"{presentation_report.get('status', 'unavailable')}"
    )
    if not pipeline._presentation_quality_accepted:
        presentation_reason_code = "presentation_quality_requires_review"
        review_only_output = True
        status = "degraded" if status == "ok" else status
        pipeline._require_review(10, presentation_reason_code)
        pipeline.log.warn(
            "Stage10 表现门未通过；本轮只允许导出 result_review*"
        )

    managed_export_enabled = bool(
        getattr(pipeline.cfg, "stage10_managed_output_enabled", True)
    )
    managed_pixels: Optional[np.ndarray] = None
    managed_export_report: Optional[Dict[str, Any]] = None
    stage10_star_reference: Optional[Dict[str, Any]] = None
    stage10_star_catalog_resolution: Dict[str, Any] = {
        "schema": "starun.stage10-star-catalog-resolution.v2",
        "status": "not_required" if not stage9_stars_required else "not_run",
    }
    stage10_pre_export_visibility: Optional[Dict[str, Any]] = None
    review_display_contract: Optional[Dict[str, Any]] = None
    if managed_export_enabled or review_only_output:
        try:
            if stage_saved:
                pipeline.cmd_with_check("load", "stage10_final")
            current_pixels = pipeline.siril.get_image_pixeldata(preview=False)
            if current_pixels is None:
                raise RuntimeError("Stage10 managed-export pixels unavailable")
            managed_pixels = np.array(current_pixels, copy=True)
        except (
            AttributeError,
            CommandError,
            RuntimeError,
            SirilError,
            TypeError,
            ValueError,
        ) as error:
            managed_export_report = {
                "schema": "starun.managed-output.v2",
                "status": "partial",
                "ready": False,
                "mode": "independent_managed_derivatives",
                "issues": [f"source_pixels_unavailable: {error}"],
            }
            messages.append(
                "managed output source unavailable; independent derivatives skipped"
            )

    if stage9_stars_required:
        if managed_pixels is None:
            _stage10_terminal_failure(
                pipeline,
                stage_label,
                messages,
                reason_code="required_stars_catalog_visibility_failed",
                reason=(
                    "required-stars catalog audit cannot read the final pixel buffer"
                ),
                action=failure_action,
                strict_stop=failure_action == "stop",
            )
            return
        star_audit_pixels = managed_pixels
        star_audit_domain = "native_final_pixel_domain"
        review_catalog_audit = bool(
            review_only_output
            and stage9_contract_known
            and stage9_output_contains_stars
            and final_loaded
            and final_file == preferred_final_source
        )
        if review_catalog_audit:
            try:
                # Verify a real frozen catalog before allowing a very flat
                # linear review source to receive an observer-only mapping.
                # Generic compact peaks are deliberately not consulted here.
                preliminary_reference, preliminary_resolution = (
                    _stage10_star_visibility_reference(
                        pipeline,
                        managed_pixels,
                        restore_stem=(
                            "stage10_final" if stage_saved else final_file
                        ),
                    )
                )
                stage10_star_reference = preliminary_reference
                stage10_star_catalog_resolution = preliminary_resolution
                if (
                    preliminary_reference is None
                    or preliminary_resolution.get("status") != "ready"
                ):
                    raise RuntimeError(
                        "trusted with-stars review catalog is unavailable: "
                        + str(
                            preliminary_resolution.get("reason")
                            or "catalog lineage could not be verified"
                        )
                    )
                preliminary_star_count = int(
                    preliminary_resolution.get("star_count") or 0
                )
                if preliminary_star_count < 16:
                    raise RuntimeError(
                        "trusted with-stars review catalog has fewer than "
                        f"16 stars: {preliminary_star_count}"
                    )
                review_input_visibility = audit_display_visibility(
                    managed_pixels,
                    target_type=active_target_type,
                    stars_required=False,
                    pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
                    star_visibility_config=pipeline.cfg,
                )
                review_display_contract = display_rendition.build_review_contract(
                    managed_pixels,
                    reason=(
                        final_source_review_reason
                        or "stage10_review_only_catalog_audit"
                    ),
                    source_stem="stage10_final",
                    input_visibility=review_input_visibility,
                    subject_chroma_plan=dict(
                        getattr(pipeline, "_review_subject_chroma_plan", {}) or {}
                    ),
                    artifact_root=pipeline.work_dir,
                    pixel_coordinate_domain=(
                        display_rendition.PIXEL_DOMAIN_BOTTOM_UP
                    ),
                )
                if (
                    not display_rendition.validate_review_contract(
                        review_display_contract
                    )
                    and review_input_visibility.get("exposure_state")
                    == "unmappable"
                    and bool(
                        (review_input_visibility.get("metrics") or {}).get(
                            "underexposed"
                        )
                    )
                    and not bool(
                        (review_input_visibility.get("metrics") or {}).get(
                            "overexposed"
                        )
                    )
                ):
                    # A linear galaxy can be too compressed to satisfy the
                    # pre-map extended-subject proxy even though its signed
                    # catalog and with-stars lineage are valid.  Freeze the
                    # bounded linked mapping, then require the mapped galaxy
                    # and all catalog visibility gates to pass below.
                    review_display_contract = (
                        display_rendition.build_linked_review_contract(
                            managed_pixels,
                            reason=(
                                final_source_review_reason
                                or "stage10_review_only_catalog_audit"
                            ),
                            source_stem="stage10_final",
                            input_visibility=review_input_visibility,
                        )
                    )
                    subject_chroma_plan = dict(
                        getattr(pipeline, "_review_subject_chroma_plan", {}) or {}
                    )
                    if subject_chroma_plan.get("accepted", False):
                        try:
                            review_display_contract = (
                                display_rendition.build_subject_chroma_review_contract(
                                    managed_pixels,
                                    tone_contract=review_display_contract,
                                    mask_artifact=dict(
                                        subject_chroma_plan.get("mask_artifact") or {}
                                    ),
                                    chroma_evidence=dict(
                                        subject_chroma_plan.get("evidence") or {}
                                    ),
                                    effective_saturation_budget=float(
                                        subject_chroma_plan.get(
                                            "effective_saturation_budget",
                                            0.0,
                                        )
                                        or 0.0
                                    ),
                                    artifact_root=pipeline.work_dir,
                                    pixel_coordinate_domain=(
                                        display_rendition.PIXEL_DOMAIN_BOTTOM_UP
                                    ),
                                )
                            )
                        except (OSError, RuntimeError, TypeError, ValueError):
                            pass
                    review_display_contract["selection_evidence"] = {
                        "mode": "trusted_catalog_underexposed_linear_review",
                        "catalog_source": preliminary_resolution.get(
                            "source"
                        ),
                        "catalog_star_count": preliminary_star_count,
                        "native_subject_proxy": "unmappable",
                        "post_mapping_subject_gate_required": True,
                        "compact_peaks_used": False,
                    }
                if not display_rendition.validate_review_contract(
                    review_display_contract
                ):
                    raise RuntimeError("review display contract validation failed")
                star_audit_pixels = display_rendition.apply_review_contract(
                    managed_pixels,
                    review_display_contract,
                    artifact_root=pipeline.work_dir,
                    pixel_coordinate_domain=(
                        display_rendition.PIXEL_DOMAIN_BOTTOM_UP
                    ),
                )
                star_audit_domain = "shared_review_display_contract"
            except (RuntimeError, TypeError, ValueError) as error:
                diagnostic_audit = audit_display_visibility(
                    managed_pixels,
                    target_type=active_target_type,
                    stars_required=True,
                    star_reference=None,
                    pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
                    star_visibility_config=pipeline.cfg,
                )
                pipeline._write_stage_json(
                    "stage10_pre_export_visibility.json",
                    {
                        "schema": "starun.stage10-pre-export-visibility.v2",
                        "audit_domain": "native_final_pixel_diagnostic_only",
                        "delivery_scope": "review_only",
                        "display_contract": {
                            "status": "unavailable",
                            "error": str(error),
                        },
                        "catalog_resolution": (
                            stage10_star_catalog_resolution
                        ),
                        "audit": diagnostic_audit,
                    },
                )
                _stage10_terminal_failure(
                    pipeline,
                    stage_label,
                    messages,
                    reason_code="required_stars_catalog_visibility_failed",
                    reason=f"review-only star audit mapping unavailable: {error}",
                    action=failure_action,
                    strict_stop=failure_action == "stop",
                )
                return
        stage10_star_reference, stage10_star_catalog_resolution = (
            _stage10_star_visibility_reference(
                pipeline,
                star_audit_pixels,
                restore_stem="stage10_final" if stage_saved else final_file,
                display_contract=(
                    review_display_contract if review_catalog_audit else None
                ),
            )
        )
        stage10_pre_export_visibility = audit_display_visibility(
            star_audit_pixels,
            target_type=active_target_type,
            stars_required=True,
            star_reference=stage10_star_reference,
            pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
            star_visibility_config=pipeline.cfg,
        )
        pipeline._write_stage_json(
            "stage10_pre_export_visibility.json",
            {
                "schema": "starun.stage10-pre-export-visibility.v2",
                "audit_domain": star_audit_domain,
                "delivery_scope": (
                    "review_only" if review_catalog_audit else "formal"
                ),
                "display_contract": (
                    {
                        "schema": review_display_contract.get("schema"),
                        "name": review_display_contract.get("name"),
                        "mode": review_display_contract.get("mode"),
                        "observer_only": review_display_contract.get(
                            "observer_only"
                        ),
                    }
                    if review_display_contract is not None
                    else None
                ),
                "catalog_resolution": stage10_star_catalog_resolution,
                "audit": stage10_pre_export_visibility,
            },
        )
        star_check = (
            (stage10_pre_export_visibility.get("checks") or {}).get(
                "star_visibility"
            )
            or {}
        )
        if star_check.get("passed") is not True:
            catalog_reason = str(
                (
                    star_check.get("catalog_visibility")
                    or {}
                ).get("reason")
                or (
                    star_check.get("catalog_visibility")
                    or {}
                ).get("reason_code")
                or "source-catalog star visibility thresholds failed"
            )
            _stage10_terminal_failure(
                pipeline,
                stage_label,
                messages,
                reason_code="required_stars_catalog_visibility_failed",
                reason=(
                    "required stars failed the pre-export source-catalog audit: "
                    + catalog_reason
                ),
                action=failure_action,
                strict_stop=failure_action == "stop",
            )
            return

    if review_only_output:
        pipeline._review_display_route = True
        frozen_contract = dict(
            getattr(pipeline, "_display_rendition_contract", {}) or {}
        )
        contract_reason = str(
            frozen_contract.get("reason") or "stage10_review_only_output"
        )
        if review_display_contract is not None:
            frozen_contract = dict(review_display_contract)
        elif managed_pixels is not None:
            try:
                input_visibility = audit_display_visibility(
                    managed_pixels,
                    target_type=active_target_type,
                    stars_required=stage9_stars_required,
                    star_reference=stage10_star_reference,
                    pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
                    star_visibility_config=pipeline.cfg,
                )
                frozen_contract = display_rendition.build_review_contract(
                    managed_pixels,
                    reason=contract_reason,
                    source_stem="stage10_final",
                    input_visibility=input_visibility,
                    subject_chroma_plan=dict(
                        getattr(pipeline, "_review_subject_chroma_plan", {}) or {}
                    ),
                    artifact_root=pipeline.work_dir,
                    pixel_coordinate_domain=(
                        display_rendition.PIXEL_DOMAIN_BOTTOM_UP
                    ),
                )
            except (RuntimeError, TypeError, ValueError) as error:
                frozen_contract = display_rendition.unavailable_contract(
                    reason=contract_reason,
                    error=str(error),
                )
        else:
            frozen_contract = display_rendition.unavailable_contract(
                reason=contract_reason,
                error="review display source pixels unavailable",
            )
        pipeline._display_rendition_contract = dict(frozen_contract)
        pipeline._write_stage_json(
            "display_rendition_contract.json",
            frozen_contract,
        )
        if display_rendition.validate_review_contract(frozen_contract):
            review_display_contract = frozen_contract
        else:
            messages.append(
                "review display contract unavailable; PNG publication will fail closed"
            )

    # 切换回原工作目录导出
    pipeline.cmd_with_check("cd", f'"{pipeline.work_dir}"')

    if review_only_output:
        review_suffix = "_linear" if pipeline._stage1_input_mode == "linear_resume" else ""
        base_filename = f"result_review{review_suffix}"
        fallback_base = base_filename
        fallback_fit_base = f"{base_filename}_final"
        pipeline.main_output_basename_template = base_filename
        pipeline.main_output_fit_basename_template = fallback_fit_base
        pipeline._final_output_review_only = True
        messages.append(
            "review_only_output=true; normal result_processed/result_final names withheld"
        )
        pipeline.log.warn(
            "Stage10 安全门要求复核；本轮仅导出 result_review* 复核产物，"
            "不写入普通 result_processed/result_final 名称"
        )
    else:
        base_filename = pipeline._result_output_basename()
        fallback_base = "result_processed"
        fallback_fit_base = "result_final"
        if pipeline._stage1_input_mode == "linear_resume":
            fallback_base = "result_processed_linear"
            fallback_fit_base = "result_final_linear"

    fit_filename = getattr(
        pipeline,
        "main_output_fit_basename_template",
        base_filename + "_final",
    )
    export_started_at = time.time()
    pipeline._final_export_started_at = export_started_at
    pipeline._final_output_basenames = tuple(
        dict.fromkeys(
            (
                base_filename,
                fit_filename,
                fallback_base,
                fallback_fit_base,
            )
        )
    )
    export_report: Dict[str, Any] = {}
    status, messages = export_final_outputs(
        pipeline.cmd_with_check,
        pipeline.log,
        base_filename=base_filename,
        fit_filename=fit_filename,
        fallback_base=fallback_base,
        fallback_fit_base=fallback_fit_base,
        output_format=getattr(pipeline.cfg, "output_format", "all"),
        png_preview_stretch=(
            not review_only_output
            and not bool(getattr(pipeline, "_stage7_stretch_accepted", False))
        ),
        status=status,
        messages=messages,
        export_report=export_report,
    )
    export_report["color_rebalance_report"] = {
        "file": "stage10_color_rebalance_report.json",
        "status": stage10_color_report.get("status", "unavailable"),
        "used_for_gate": False,
        "applied_saturation": stage10_applied_saturation,
    }
    pipeline._write_stage_json("stage10_export_report.json", export_report)
    if managed_export_enabled and managed_pixels is not None:
        scientific_names = {
            str(fit_filename or ""),
            str(fallback_fit_base or ""),
        }
        scientific_paths = [
            pipeline.work_dir / f"{name}.{extension}"
            for name in scientific_names
            if name
            for extension in ("fit", "fits")
        ]
        for extension in ("fit", "fits"):
            for candidate in pipeline.work_dir.glob(f"*.{extension}"):
                try:
                    if candidate.stat().st_mtime >= export_started_at - 1.0:
                        scientific_paths.append(candidate)
                except OSError:
                    continue
        scientific_paths = list(dict.fromkeys(scientific_paths))
        try:
            managed_export_report = export_managed_outputs(
                managed_pixels,
                work_dir=pipeline.work_dir,
                base_filename=base_filename,
                output_format=getattr(pipeline.cfg, "output_format", "all"),
                scientific_paths=scientific_paths,
                target_type=active_target_type,
                stars_required=stage9_stars_required,
                star_reference=stage10_star_reference,
                star_visibility_config=pipeline.cfg,
                display_contract=(
                    review_display_contract if review_only_output else None
                ),
            )
            pipeline._write_stage_json(
                "managed_output_report.json",
                managed_export_report,
            )
            messages.append(
                "managed_output="
                f"{managed_export_report.get('status', 'unknown')} "
                f"artifacts={len(managed_export_report.get('artifacts') or [])} "
                "scientific_unchanged="
                f"{str(bool((managed_export_report.get('scientific_archive') or {}).get('unchanged', False))).lower()} "
                "display_visibility="
                f"{str((managed_export_report.get('display_visibility') or {}).get('status', 'not_requested'))}"
            )
            if not bool(managed_export_report.get("ready", False)):
                messages.append(
                    "managed output incomplete; primary Siril/FITS exports remain valid"
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            managed_export_report = {
                "schema": "starun.managed-output.v2",
                "status": "partial",
                "ready": False,
                "mode": "independent_managed_derivatives",
                "issues": [str(error)],
            }
            pipeline._write_stage_json(
                "managed_output_report.json",
                managed_export_report,
            )
            pipeline.log.warn(f"Stage10 managed export unavailable: {error}")
            messages.append("managed output export failed")
    elif not managed_export_enabled:
        messages.append("managed output disabled by configuration")

    review_png_report: Dict[str, Any] = {
        "applicable": bool(review_only_output),
        "status": "not_requested",
        "contract": (
            dict(getattr(pipeline, "_display_rendition_contract", {}) or {})
            if review_only_output
            else None
        ),
    }
    png_export = (export_report.get("outputs") or {}).get("png")
    if review_only_output and isinstance(png_export, dict):
        selected_png_name = str(png_export.get("selected") or "")
        selected_png_path = (
            pipeline.work_dir / selected_png_name
            if selected_png_name
            else pipeline.work_dir / f"{base_filename}.png"
        )
        review_png_published = False
        review_png_error = ""
        if review_display_contract is None:
            review_png_error = "required Review contract unavailable"
        elif managed_export_enabled:
            display_artifact = next(
                (
                    artifact
                    for artifact in (
                        (managed_export_report or {}).get("artifacts") or []
                    )
                    if artifact.get("role") == "display"
                    and artifact.get("status") == "written"
                    and Path(str(artifact.get("path") or "")).is_file()
                ),
                None,
            )
            if display_artifact is None:
                review_png_error = (
                    "managed Review PNG missing or visibility audit failed"
                )
            else:
                try:
                    managed_display_path = Path(str(display_artifact["path"]))
                    shutil.copyfile(managed_display_path, selected_png_path)
                    review_png_published = True
                    review_png_report.update(
                        managed_display=str(managed_display_path),
                        primary=str(selected_png_path),
                        pixel_identity="byte_identical_copy",
                        visibility=display_artifact.get("visibility"),
                    )
                except OSError as error:
                    review_png_error = str(error)
        elif managed_pixels is not None:
            try:
                review_pixels = display_rendition.apply_review_contract(
                    managed_pixels,
                    review_display_contract,
                    artifact_root=pipeline.work_dir,
                    pixel_coordinate_domain=(
                        display_rendition.PIXEL_DOMAIN_BOTTOM_UP
                    ),
                )
                visibility = audit_display_visibility(
                    review_pixels,
                    target_type=active_target_type,
                    stars_required=stage9_stars_required,
                    star_reference=stage10_star_reference,
                    pixel_coordinate_domain="siril_pixel_buffer_bottom_up",
                    star_visibility_config=pipeline.cfg,
                )
                if not bool(visibility.get("passed", False)):
                    raise ValueError(
                        "review PNG visibility audit failed: "
                        + ",".join(visibility.get("failed_checks") or [])
                    )
                write_managed_display_png(selected_png_path, review_pixels)
                decoded_visibility = audit_display_visibility(
                    read_managed_display_png(selected_png_path),
                    target_type=active_target_type,
                    stars_required=stage9_stars_required,
                    star_reference=stage10_star_reference,
                    pixel_coordinate_domain="display_array_top_down",
                    star_visibility_config=pipeline.cfg,
                )
                if not bool(decoded_visibility.get("passed", False)):
                    raise ValueError(
                        "decoded review PNG visibility audit failed: "
                        + ",".join(
                            decoded_visibility.get("failed_checks") or []
                        )
                    )
                review_png_published = True
                review_png_report.update(
                    primary=str(selected_png_path),
                    visibility=decoded_visibility,
                    pre_encode_visibility=visibility,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                review_png_error = str(error)
        else:
            review_png_error = "review display source pixels unavailable"

        if review_png_published:
            review_png_report["status"] = "written"
            if str(png_export.get("status") or "") not in {
                "primary",
                "fallback",
            }:
                png_export["status"] = "managed_review"
            png_export["selected"] = selected_png_path.name
            png_export["review_display_status"] = "written"
            png_export["display_rendition_contract"] = (
                "display_rendition_contract.json"
            )
            messages.append(
                "review PNG uses the frozen audited Review rendition"
            )
        else:
            selected_png_path.unlink(missing_ok=True)
            review_png_report.update(
                status="rejected_not_published",
                error=review_png_error,
            )
            png_export.update(
                status="rejected_not_published",
                selected=None,
                review_display_status="failed_closed",
                review_display_error=review_png_error,
            )
            export_report["overall_status"] = "partial"
            status = "degraded" if status == "ok" else status
            messages.append(
                "review PNG withheld because the rendition contract or "
                "visibility audit failed"
            )
    normal_png_catalog_audit: Dict[str, Any] = {
        "applicable": bool(
            not review_only_output
            and stage9_stars_required
            and isinstance(png_export, dict)
        ),
        "status": "not_requested",
    }
    if (
        not review_only_output
        and stage9_stars_required
        and isinstance(png_export, dict)
    ):
        selected_png_name = str(png_export.get("selected") or "")
        selected_png_path = (
            pipeline.work_dir / selected_png_name
            if selected_png_name
            else pipeline.work_dir / f"{base_filename}.png"
        )
        try:
            managed_display = next(
                (
                    Path(str(artifact.get("path") or ""))
                    for artifact in (
                        (managed_export_report or {}).get("artifacts") or []
                    )
                    if artifact.get("role") == "display"
                    and artifact.get("status") == "written"
                    and Path(str(artifact.get("path") or "")).is_file()
                ),
                None,
            )
            if managed_display is not None:
                shutil.copyfile(managed_display, selected_png_path)
            elif managed_pixels is not None and not managed_export_enabled:
                write_managed_display_png(selected_png_path, managed_pixels)
            else:
                raise RuntimeError(
                    "catalog-audited managed display PNG is unavailable"
                )
            decoded_visibility = audit_display_visibility(
                read_managed_display_png(selected_png_path),
                target_type=active_target_type,
                stars_required=True,
                star_reference=stage10_star_reference,
                pixel_coordinate_domain="display_array_top_down",
                star_visibility_config=pipeline.cfg,
            )
            star_check = (
                (decoded_visibility.get("checks") or {}).get(
                    "star_visibility"
                )
                or {}
            )
            if star_check.get("passed") is not True:
                raise ValueError(
                    "decoded final PNG catalog-star visibility failed"
                )
            normal_png_catalog_audit.update(
                status="passed",
                path=str(selected_png_path),
                visibility=decoded_visibility,
                source=(
                    str(managed_display)
                    if managed_display is not None
                    else "stage10_final_pixels"
                ),
            )
            png_export["selected"] = selected_png_path.name
            png_export["catalog_visibility_status"] = "passed"
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            selected_png_path.unlink(missing_ok=True)
            normal_png_catalog_audit.update(
                status="rejected_not_published",
                error=str(error),
            )
            png_export.update(
                status="rejected_not_published",
                selected=None,
                catalog_visibility_status="failed_closed",
                catalog_visibility_error=str(error),
            )
            export_report["overall_status"] = "partial"
            status = "degraded" if status == "ok" else status
            messages.append(
                "final PNG withheld because decoded source-catalog star "
                "visibility failed"
            )
    export_report["review_display"] = review_png_report
    export_report["normal_png_catalog_audit"] = normal_png_catalog_audit
    pipeline._write_stage_json("stage10_export_report.json", export_report)

    try:
        output_color_manifest = build_output_color_manifest(
            work_dir=pipeline.work_dir,
            base_filename=base_filename,
            fit_filename=fit_filename,
            fallback_base=fallback_base,
            fallback_fit_base=fallback_fit_base,
            output_format=getattr(pipeline.cfg, "output_format", "all"),
            channel_semantics=channel_semantics,
            review_only=review_only_output,
            exported_after=export_started_at,
            managed_export_report=managed_export_report,
            source_color_contract=stage10_color_report.get("contract"),
            display_rendition_contract=(
                dict(getattr(pipeline, "_display_rendition_contract", {}) or {})
                if review_only_output
                else None
            ),
            export_report=export_report,
        )
        pipeline._output_color_manifest = output_color_manifest
        pipeline._write_stage_json(
            "output_color_manifest.json",
            output_color_manifest,
        )
        color_summary = output_color_manifest["summary"]
        messages.append(
            "output_color_audit="
            f"{output_color_manifest.get('mode', 'unknown')} "
            f"artifacts={int(color_summary['artifact_count'])} "
            "managed_export_ready="
            f"{str(bool(color_summary.get('managed_export_ready', False))).lower()}"
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        pipeline._output_color_manifest = {
            "schema": "starun.output-color-manifest.v1",
            "mode": "report_only",
            "rewrote_outputs": False,
            "status": "unavailable",
            "error": str(error),
        }
        pipeline.log.warn(f"Stage10 output color audit unavailable: {error}")
        messages.append("output color audit unavailable; exported files unchanged")

    if denoise_effective_status == "success":
        denoise_component_status = "applied"
        denoise_reason_code = denoise_fallback_reason or "accepted"
    elif denoise_effective_status == "rolled_back_safe":
        denoise_component_status = "skipped"
        denoise_reason_code = "star_protection_merge_rollback"
    elif denoise_effective_status == "protection_failed":
        denoise_component_status = "failed"
        denoise_reason_code = "star_protection_rollback_failed"
    elif review_only_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "review_only_output"
    elif duplicate_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "duplicate_denoise_guard"
    elif low_noise_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "auto_low_noise"
    elif preserve_mode_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "preserve_mode"
    elif configured_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "disabled_by_task_configuration"
    elif star_protection_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "star_protection_unavailable"
    else:
        denoise_component_status = "failed"
        denoise_reason_code = "all_final_denoisers_failed"
    denoise_component = {
        "status": denoise_component_status,
        "method": denoise_effective,
        "primary": denoise_primary,
        "primary_status": denoise_primary_status,
        "selected_mode": selected_denoise_mode,
        "effective_mode": effective_denoise_mode,
        "requested_strength": final_denoise_strength,
        "star_protection": star_protection_report,
        "reason_code": denoise_reason_code,
        "fallback_used": bool(denoise_fallback_used),
        "input": final_file if final_loaded else None,
        "output": "stage10_final" if stage_saved else None,
    }
    input_source_component = {
        "status": "applied" if final_loaded else "failed",
        "method": final_file if final_loaded else None,
        "reason_code": (
            "final_source_recovery"
            if input_source_fallback_used
            else "accepted"
            if final_loaded
            else "unavailable"
        ),
        "fallback_used": input_source_fallback_used,
    }
    export_outputs = export_report.get("outputs") or {}
    export_failed = any(
        isinstance(item, dict) and item.get("status") == "failed"
        for item in export_outputs.values()
    )
    export_fallback_used = bool(export_report.get("fallback_used", False))
    export_component = {
        "status": (
            "failed"
            if export_failed
            else "applied"
            if export_outputs
            else "skipped"
        ),
        "method": {
            key: value.get("selected")
            for key, value in export_outputs.items()
            if isinstance(value, dict) and value.get("selected")
        },
        "reason_code": (
            "final_export_failed"
            if export_failed
            else "final_export_fallback"
            if export_fallback_used
            else "accepted"
            if export_outputs
            else "not_requested"
        ),
        "fallback_used": export_fallback_used,
    }
    color_component = {
        "status": (
            "failed"
            if stage10_saturation_failed
            else "applied"
            if abs(stage10_applied_saturation) > 1e-8
            else "skipped"
        ),
        "method": "post_denoise_budgeted_saturation",
        "reason_code": (
            "command_failed"
            if stage10_saturation_failed
            else stage10_color_rebalance_blocked_reason
            if stage10_color_rebalance_blocked_reason
            else "accepted"
            if abs(stage10_applied_saturation) > 1e-8
            else "stage9_local_color_risk"
            if stage9_color_guard["applied"]
            else "color_operation_policy"
            if requested_final_saturation == 0.0
            else "stage4_budget_exhausted"
        ),
        "requested_saturation": requested_final_saturation,
        "effective_saturation": effective_final_saturation,
        "applied_saturation": stage10_applied_saturation,
        "report_status": stage10_color_report.get("status", "unavailable"),
        "report": "stage10_color_rebalance_report.json",
        "fallback_used": False,
    }
    stage_denoise_fallback_used = bool(
        denoise_fallback_used and denoise_effective_status == "success"
    )
    stage_fallback_used = bool(
        input_source_fallback_used
        or stage_denoise_fallback_used
        or export_fallback_used
    )
    stage_reason_code = (
        "stage2_view_review_required"
        if stage2_view_review_required
        else "stage3_background_review_required"
        if stage3_background_review_required
        else "stage4_color_review_required"
        if stage4_color_review_required
        else "uncalibrated_background_color_review_required"
        if stage7_background_color_blocks_normal
        else "starmask_diffuse_residual_borderline"
        if stage6_starmask_borderline_review_required
        else "stage6_quality_hard_failed_retained"
        if stage6_quality_hard_failed_retained
        else "stage7_forced_quality_delivery_review_only"
        if stage7_forced_delivery
        else final_source_review_reason
        if final_source_review_reason
        else "stage10_failure_policy"
        if failure_policy_reason
        else final_quality_reason_code
        if final_quality_reason_code
        else presentation_reason_code
        if presentation_reason_code
        else denoise_fallback_reason
        if stage_denoise_fallback_used
        else "final_source_recovery"
        if input_source_fallback_used
        else "final_export_fallback"
        if export_fallback_used
        else ""
    )

    if preserve_review_triggered or failure_policy_reason:
        pipeline._require_review(
            10,
            failure_policy_reason or "stage10_failure_policy_preserve_review",
        )
    if final_source_review_reason:
        pipeline._require_review(10, final_source_review_reason)
    if final_quality_reason_code:
        pipeline._require_review(10, final_quality_reason_code)
    if review_only_output and status == "ok":
        status = "degraded"

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(
        stage_label,
        status,
        elapsed,
        "；".join(messages),
        fallback_used=stage_fallback_used,
        reason_code=stage_reason_code,
        details={
            "processing_mode": processing_mode,
            "bright_core_with_stars_fallback": bright_core_fallback,
            "failure_action": failure_action,
            "denoise_backend_policy": denoise_backend_policy,
            "preserve_review_triggered": preserve_review_triggered,
            "failure_policy_reason": failure_policy_reason or None,
            "review_only_output": bool(review_only_output),
            "stage2_view_review_required": stage2_view_review_required,
            "stage3_background_review_required": (
                stage3_background_review_required
            ),
            "stage4_color_review_required": stage4_color_review_required,
            "stage7_background_color_review_required": (
                stage7_background_color_review_required
            ),
            "stage7_forced_delivery": stage7_forced_delivery,
            "stage7_forced_delivery_reasons": list(
                getattr(
                    pipeline,
                    "_stage7_forced_delivery_reasons",
                    [],
                )
                or []
            ),
            "stage6_starmask_borderline_review_required": (
                stage6_starmask_borderline_review_required
            ),
            "stage6_quality_hard_failed_retained": (
                stage6_quality_hard_failed_retained
            ),
            "stage7_background_color_review_gate": dict(
                getattr(
                    pipeline,
                    "_stage7_background_color_review_gate",
                    {},
                )
                or {}
            ),
            "final_source": final_file if final_loaded else None,
            "preferred_final_source": preferred_final_source,
            "final_source_review_required": bool(final_source_review_reason),
            "retained_unverified_current_image": not final_loaded,
            "final_quality_gate_status": final_quality_gate_status,
            "final_quality_gate_error": final_quality_gate_error or None,
            "scientific_quality_accepted": bool(
                pipeline._scientific_quality_accepted
            ),
            "presentation_quality_status": str(
                presentation_report.get("status") or "unavailable"
            ),
            "presentation_quality_accepted": bool(
                pipeline._presentation_quality_accepted
            ),
            "presentation_quality_report": "presentation_quality_report.json",
            "final_quality_severity": final_quality.get("severity"),
            "final_quality_warning_count": len(
                final_quality.get("warnings") or []
            ),
            "stage10_quality_repair": dict(
                final_quality.get("repair")
                or getattr(pipeline, "_stage10_quality_repair_report", {})
                or {}
            ),
            "managed_output_ready": bool(
                managed_export_report
                and managed_export_report.get("ready", False)
            ),
            "stage10_color_rebalance_report_status": stage10_color_report.get(
                "status", "unavailable"
            ),
            "color_adjustment_ledger_entries": len(
                getattr(pipeline, "_color_adjustment_ledger", []) or []
            ),
            "display_visibility_status": str(
                (
                    (managed_export_report or {}).get("display_visibility")
                    or {}
                ).get("status", "not_requested")
            ),
        },
        components={
            "input_source": input_source_component,
            "denoise": denoise_component,
            "color_rebalance": color_component,
            "export": export_component,
        },
        review_reasons=pipeline._stage_review_reasons(10),
    )
