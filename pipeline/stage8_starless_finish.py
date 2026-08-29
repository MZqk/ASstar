"""Deterministic pixel contracts for the Stage 8 Starless finish chain."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np


STAGE8_STARLESS_FINISH_SCHEMA = "starun.stage8-starless-finish.v1"
FITS_DATA_SHA256_METHOD = "fits_logical_data_v1"
DECODED_PIXEL_SHA256_METHOD = "canonical_native_float32_chw_v1"


def pixel_sha256(pixels: np.ndarray) -> str:
    """Return a stable identity for the exact in-memory pixel representation."""

    array = np.ascontiguousarray(np.asarray(pixels))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_decoded_pixel_sha256(pixels: np.ndarray) -> str:
    """Hash decoded pixels in the canonical Siril CHW float32 domain."""

    array = np.ascontiguousarray(np.asarray(pixels, dtype=np.float32))
    if array.ndim == 2:
        pass
    elif array.ndim == 3 and array.shape[0] in {1, 3}:
        pass
    else:
        raise ValueError(
            "decoded Stage8 pixels must be mono or CHW RGB, "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("decoded Stage8 pixels contain non-finite values")
    return pixel_sha256(array)


def decoded_science_image_hdus(hdul: Any) -> list[Any]:
    """Return only decoded mono/RGB science images, excluding metadata images."""

    science_hdus = []
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is None:
            continue
        if str(getattr(hdu, "name", "") or "").strip().lower() == "iccprofile":
            continue
        shape = tuple(int(value) for value in np.shape(data))
        if len(shape) == 2 or (len(shape) == 3 and shape[0] in {1, 3}):
            science_hdus.append(hdu)
    return science_hdus


def persisted_fits_decoded_pixel_sha256(path: Path) -> str:
    """Decode a FITS artifact and hash its canonical Siril pixel domain."""

    from astropy.io import fits

    with fits.open(
        Path(path),
        memmap=False,
        do_not_scale_image_data=False,
    ) as hdul:
        image_hdus = decoded_science_image_hdus(hdul)
        if len(image_hdus) != 1:
            raise ValueError(
                "formal Stage8 FITS must contain exactly one decoded image HDU"
            )
        pixels = np.asarray(image_hdus[0].data)
    return canonical_decoded_pixel_sha256(pixels)


def _as_chw(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array[np.newaxis, ...], True
    if array.ndim != 3 or array.shape[0] not in {1, 3}:
        raise ValueError(f"unsupported Starless image shape: {array.shape}")
    return array, False


def _restore_shape(image: np.ndarray, squeezed: bool) -> np.ndarray:
    return image[0] if squeezed else image


def linked_luminance(image: np.ndarray) -> np.ndarray:
    array, _ = _as_chw(image)
    if array.shape[0] == 1:
        return np.asarray(array[0], dtype=np.float32)
    return np.asarray(
        0.2126 * array[0] + 0.7152 * array[1] + 0.0722 * array[2],
        dtype=np.float32,
    )


def _validated_mask(mask: np.ndarray, shape: Tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(mask, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid {name}: shape={array.shape}, expected={shape}")
    return np.clip(array, 0.0, 1.0)


def _finish_masks(
    masks: Mapping[str, Any],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_layers = []
    for name in (
        "galaxy_signal_mask",
        "nebula_mask",
        "faint_nebula_mask",
        "subject_mask",
    ):
        value = masks.get(name)
        if value is None:
            continue
        subject_layers.append(_validated_mask(value, shape, name))
    if subject_layers:
        subject = np.maximum.reduce(subject_layers)
    else:
        background_value = masks.get("background_mask")
        if background_value is None:
            raise ValueError("Stage8 finish subject mask is unavailable")
        subject = 1.0 - _validated_mask(
            background_value,
            shape,
            "background_mask",
        )
    background = _validated_mask(
        masks.get("background_mask", 1.0 - subject),
        shape,
        "background_mask",
    )
    core_value = masks.get("core_hard_mask")
    if core_value is None:
        core_value = masks.get("core_mask", np.zeros(shape, dtype=np.float32))
    core = _validated_mask(core_value, shape, "core_hard_mask")
    weight = np.clip(subject * (1.0 - background) * (1.0 - core), 0.0, 1.0)
    return weight.astype(np.float32), background, core


def project_linked_luminance_candidate(
    baseline: np.ndarray,
    plugin_output: np.ndarray,
    masks: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep only a linked-luminance plugin delta inside the Stage8 subject mask."""

    base, squeezed = _as_chw(baseline)
    plugin, _ = _as_chw(plugin_output)
    if plugin.shape != base.shape:
        raise ValueError(
            f"plugin output shape changed: {plugin.shape} != {base.shape}"
        )
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(plugin)):
        raise ValueError("non-finite pixels in Stage8 finish input")
    shape = (int(base.shape[-2]), int(base.shape[-1]))
    weight, background, core = _finish_masks(masks, shape)
    source_luma = linked_luminance(base)
    plugin_luma = linked_luminance(plugin)
    gain = np.ones_like(source_luma, dtype=np.float32)
    supported = source_luma > 1e-5
    gain[supported] = plugin_luma[supported] / source_luma[supported]
    gain = np.clip(gain, 0.50, 1.50)
    linked = np.clip(base * gain[np.newaxis, :, :], 0.0, 1.0)
    candidate = base + (linked - base) * weight[np.newaxis, :, :]
    restore = (weight <= 1e-6) | (background >= 0.50) | (core >= 0.50)
    candidate[:, restore] = base[:, restore]
    candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)
    return _restore_shape(candidate, squeezed), {
        "projection": "linked_rec709_luminance_subject_mask",
        "subject_support_pixels": int(np.count_nonzero(weight > 0.05)),
        "background_restore_pixels": int(np.count_nonzero(background >= 0.50)),
        "core_restore_pixels": int(np.count_nonzero(core >= 0.50)),
        "gain_min": float(np.min(gain)),
        "gain_max": float(np.max(gain)),
    }


def _saturation(image: np.ndarray) -> np.ndarray:
    array, _ = _as_chw(image)
    if array.shape[0] == 1:
        return np.zeros(array.shape[-2:], dtype=np.float32)
    maximum = np.max(array, axis=0)
    minimum = np.min(array, axis=0)
    return np.asarray((maximum - minimum) / np.maximum(maximum, 1e-6), dtype=np.float32)


def project_luminance_locked_color_candidate(
    baseline: np.ndarray,
    plugin_output: np.ndarray,
    masks: Mapping[str, Any],
    *,
    effective_saturation_budget: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep only a bounded, luminance-locked Vectra delta inside the subject."""

    base, squeezed = _as_chw(baseline)
    plugin, _ = _as_chw(plugin_output)
    if base.shape[0] != 3 or plugin.shape != base.shape:
        raise ValueError("Vectra requires shape-compatible RGB pixels")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(plugin)):
        raise ValueError("non-finite pixels in Vectra input")
    shape = (int(base.shape[-2]), int(base.shape[-1]))
    weight, background, core = _finish_masks(masks, shape)
    base_luma = linked_luminance(base)
    plugin_luma = linked_luminance(plugin)
    chroma = plugin - plugin_luma[np.newaxis, :, :]
    locked = np.clip(base_luma[np.newaxis, :, :] + chroma, 0.0, 1.0)
    try:
        budget = float(effective_saturation_budget)
    except (TypeError, ValueError):
        budget = 0.0
    if not np.isfinite(budget):
        budget = 0.0
    blend = float(np.clip(budget / 0.40, 0.0, 1.0))
    candidate = base + (locked - base) * (weight * blend)[np.newaxis, :, :]
    restore = (weight <= 1e-6) | (background >= 0.50) | (core >= 0.50)
    candidate[:, restore] = base[:, restore]
    candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)
    support = weight > 0.05
    before_sat = _saturation(base)
    after_sat = _saturation(candidate)
    subject_gain = (
        float(np.median(after_sat[support]) - np.median(before_sat[support]))
        if np.count_nonzero(support) >= 1
        else 0.0
    )
    return _restore_shape(candidate, squeezed), {
        "projection": "rec709_luminance_locked_subject_color",
        "budget": budget,
        "blend": blend,
        "subject_support_pixels": int(np.count_nonzero(support)),
        "subject_saturation_gain": subject_gain,
        "background_restore_pixels": int(np.count_nonzero(background >= 0.50)),
        "core_restore_pixels": int(np.count_nonzero(core >= 0.50)),
    }


def _box_blur_3x3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((1, 1), (1, 1)), mode="reflect")
    return sum(
        padded[row : row + values.shape[0], col : col + values.shape[1]]
        for row in range(3)
        for col in range(3)
    ) / 9.0


def assess_finish_candidate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    masks: Mapping[str, Any],
    *,
    mode: str,
    highlight_clip_ratio_max: float,
    texture_growth_max: float,
    effective_saturation_budget: float = 0.0,
) -> Dict[str, Any]:
    """Assess a projected Stage8 finish candidate without trusting plugin state."""

    report: Dict[str, Any] = {
        "status": "rejected",
        "accepted": False,
        "mode": str(mode),
        "issues": [],
        "metrics": {},
    }
    try:
        base, _ = _as_chw(baseline)
        output, _ = _as_chw(candidate)
        if output.shape != base.shape:
            raise ValueError(f"shape_changed:{output.shape}!={base.shape}")
        if not np.all(np.isfinite(output)):
            raise ValueError("candidate_non_finite")
        shape = (int(base.shape[-2]), int(base.shape[-1]))
        weight, background, core = _finish_masks(masks, shape)
        support = weight > 0.05
        if np.count_nonzero(support) < 64:
            report["issues"].append("subject_support_insufficient")
        delta = np.max(np.abs(output - base), axis=0)
        outside = (weight <= 1e-6) | (background >= 0.50)
        outside_max = float(np.max(delta[outside])) if np.any(outside) else 0.0
        core_max = float(np.max(delta[core >= 0.50])) if np.any(core >= 0.50) else 0.0
        effect_p95 = float(np.percentile(delta[support], 95.0)) if np.any(support) else 0.0
        clip_ratio = float(np.mean(output >= 0.9995))
        base_clip_ratio = float(np.mean(base >= 0.9995))
        clip_growth = max(0.0, clip_ratio - base_clip_ratio)
        base_luma = linked_luminance(base)
        output_luma = linked_luminance(output)
        luma_delta = np.abs(output_luma - base_luma)
        luma_p95 = float(np.percentile(luma_delta[support], 95.0)) if np.any(support) else 0.0
        base_detail = np.abs(base_luma - _box_blur_3x3(base_luma))
        output_detail = np.abs(output_luma - _box_blur_3x3(output_luma))
        base_texture = float(np.median(base_detail[support])) if np.any(support) else 0.0
        output_texture = float(np.median(output_detail[support])) if np.any(support) else 0.0
        texture_growth = output_texture / max(base_texture, 1e-6)
        metrics = {
            "subject_support_pixels": int(np.count_nonzero(support)),
            "outside_mask_max_abs_delta": outside_max,
            "core_max_abs_delta": core_max,
            "subject_effect_p95": effect_p95,
            "subject_luma_delta_p95": luma_p95,
            "highlight_clip_ratio": clip_ratio,
            "highlight_clip_growth": clip_growth,
            "texture_growth": texture_growth,
        }
        if outside_max > 1e-6:
            report["issues"].append("outside_mask_changed")
        if core_max > 1e-6:
            report["issues"].append("bright_core_changed")
        if effect_p95 <= 1e-6:
            report["issues"].append("candidate_no_effect")
        if clip_ratio > float(highlight_clip_ratio_max) or clip_growth > 0.002:
            report["issues"].append("highlight_clipping_growth")
        if str(mode) == "structure":
            if texture_growth > float(texture_growth_max):
                report["issues"].append("texture_growth_exceeded")
            if luma_p95 > 0.12:
                report["issues"].append("subject_luminance_drift_exceeded")
        elif str(mode) == "color":
            before_sat = _saturation(base)
            after_sat = _saturation(output)
            sat_gain = (
                float(np.median(after_sat[support]) - np.median(before_sat[support]))
                if np.any(support)
                else 0.0
            )
            metrics["subject_saturation_gain"] = sat_gain
            if luma_p95 > 0.005:
                report["issues"].append("luminance_lock_drift")
            if sat_gain <= 1e-4:
                report["issues"].append("subject_chroma_gain_missing")
            if sat_gain > max(0.0, float(effective_saturation_budget)) + 0.01:
                report["issues"].append("color_budget_exceeded")
        report["metrics"] = metrics
        report["accepted"] = not report["issues"]
        report["status"] = "accepted" if report["accepted"] else "rejected"
    except (IndexError, TypeError, ValueError, FloatingPointError) as error:
        report["issues"] = [str(error)]
    return report


__all__ = [
    "STAGE8_STARLESS_FINISH_SCHEMA",
    "assess_finish_candidate",
    "linked_luminance",
    "pixel_sha256",
    "project_linked_luminance_candidate",
    "project_luminance_locked_color_candidate",
]
