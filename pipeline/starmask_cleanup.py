"""Color-preserving multi-scale cleanup for linear star layers."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from image_metrics import _box_blur_gray, _to_rgb_float_fullres


DIFFUSE_RETRY_MAX_STRENGTH = 0.50
DIFFUSE_RETRY_TARGET_MARGIN = 0.95


def _bounded(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def _image_scale(image: np.ndarray) -> float:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    finite = np.asarray(arr, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    peak = float(np.max(np.abs(finite)))
    if peak <= 1.5:
        return 1.0
    if peak <= 255.0 * 1.05:
        return 255.0
    if peak <= 65535.0 * 1.05:
        return 65535.0
    return max(peak, 1.0)


def _weight_like(image: np.ndarray, weight: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return weight
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1:] == weight.shape:
        return weight[None, :, :]
    if arr.ndim == 3 and arr.shape[-1] <= 4 and arr.shape[:2] == weight.shape:
        return weight[:, :, None]
    raise ValueError(f"unsupported starmask layout: shape={arr.shape}")


def _restore_dtype(cleaned_norm: np.ndarray, original: np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(original)
    restored = np.clip(cleaned_norm, 0.0, 1.0) * scale
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return np.rint(restored).clip(info.min, info.max).astype(arr.dtype, copy=False)
    return restored.astype(np.float32, copy=False)


def clean_starmask_pixels(
    starmask: np.ndarray,
    cfg: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove diffuse residuals while preserving compact stars and RGB ratios."""
    original = np.asarray(starmask)
    if original.size == 0:
        raise ValueError("empty starmask")
    scale = _image_scale(original)
    stars_norm = np.nan_to_num(
        original.astype(np.float32, copy=False) / max(scale, 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    stars_norm = np.clip(stars_norm, 0.0, 1.0)
    rgb = _to_rgb_float_fullres(stars_norm)
    gray = (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ).astype(np.float32)
    if float(np.max(gray)) <= 1e-8:
        raise ValueError("empty starmask signal")

    floor_percentile = _bounded(
        getattr(cfg, "stage7_starmask_background_floor_percentile", 55.0),
        55.0,
        20.0,
        80.0,
    )
    sample_limit = float(np.percentile(gray, floor_percentile))
    background_sample = gray[gray <= sample_limit]
    if background_sample.size < 32:
        background_sample = gray.reshape(-1)
    background = float(np.median(background_sample))
    mad = float(np.median(np.abs(background_sample - background)))
    noise_sigma = max(1.4826 * mad, 1e-6)
    detection_sigma = _bounded(
        getattr(cfg, "stage7_starmask_cleanup_noise_sigma", 2.5),
        2.5,
        1.0,
        6.0,
    )

    broad = gray.copy()
    for _ in range(5):
        broad = _box_blur_gray(broad)
    local_detail = np.clip(gray - broad, 0.0, None)
    local_noise = noise_sigma + np.maximum(broad - background, 0.0) * 0.06
    compact_snr = local_detail / np.maximum(local_noise, 1e-6)
    compact_core = np.clip((compact_snr - 0.8) / 2.7, 0.0, 1.0)

    compact_support = compact_core.copy()
    for _ in range(3):
        compact_support = _box_blur_gray(compact_support)
    compact_protection = np.maximum(
        compact_core,
        np.clip(compact_support * 1.8, 0.0, 1.0),
    )

    signal = np.clip(gray - background, 0.0, None)
    snr_weight = np.clip(signal / max(detection_sigma * noise_sigma, 1e-6), 0.0, 1.0)
    keep_weight = np.maximum(snr_weight, compact_protection)

    diffuse_strength = _bounded(
        getattr(cfg, "stage7_starmask_nebula_suppression", 0.75),
        0.75,
        0.0,
        0.95,
    )
    halo_strength = _bounded(
        getattr(cfg, "stage7_starmask_halo_blur_strength", 0.35),
        0.35,
        0.0,
        0.80,
    )
    diffuse_component = np.clip(broad - background, 0.0, None)
    unprotected = 1.0 - compact_protection
    cleaned_luma = np.clip(
        signal * keep_weight
        - diffuse_component * unprotected * diffuse_strength,
        0.0,
        None,
    )

    broad_fraction = np.clip(
        diffuse_component / np.maximum(gray, 1e-6),
        0.0,
        1.0,
    )
    cleaned_luma *= 1.0 - broad_fraction * unprotected * halo_strength

    compact_floor = _bounded(
        getattr(cfg, "stage7_starmask_small_star_scale", 0.88),
        0.88,
        0.50,
        1.0,
    )
    scalar_weight = np.clip(cleaned_luma / np.maximum(gray, 1e-7), 0.0, 1.0)
    compact_min_weight = compact_floor * np.clip(
        compact_protection / 0.18,
        0.0,
        1.0,
    )
    scalar_weight = np.maximum(
        scalar_weight,
        compact_min_weight,
    )
    cleaned_norm = stars_norm * _weight_like(stars_norm, scalar_weight)

    compact_mask = compact_protection > 0.18
    faint_compact_mask = compact_mask & (compact_protection < 0.55)
    diffuse_mask = (compact_protection < 0.05) & (gray > background + noise_sigma)
    before_signal = float(np.sum(stars_norm))
    compact_before = float(np.sum(gray[compact_mask])) if np.any(compact_mask) else 0.0
    faint_compact_before = (
        float(np.sum(gray[faint_compact_mask])) if np.any(faint_compact_mask) else 0.0
    )
    diffuse_before = float(np.sum(gray[diffuse_mask])) if np.any(diffuse_mask) else 0.0
    min_compact_retention = _bounded(
        getattr(cfg, "stage7_starmask_compact_retention_min", 0.82),
        0.82,
        0.60,
        0.98,
    )
    max_diffuse_residual_ratio = _bounded(
        getattr(cfg, "stage7_starmask_diffuse_residual_ratio_max", 0.08),
        0.08,
        0.01,
        0.50,
    )

    cleaned_rgb = _to_rgb_float_fullres(cleaned_norm)
    cleaned_gray = (
        0.2126 * cleaned_rgb[0]
        + 0.7152 * cleaned_rgb[1]
        + 0.0722 * cleaned_rgb[2]
    ).astype(np.float32)
    initial_compact_after = (
        float(np.sum(cleaned_gray[compact_mask])) if np.any(compact_mask) else 0.0
    )
    initial_faint_compact_after = (
        float(np.sum(cleaned_gray[faint_compact_mask]))
        if np.any(faint_compact_mask)
        else 0.0
    )
    initial_diffuse_after = (
        float(np.sum(cleaned_gray[diffuse_mask])) if np.any(diffuse_mask) else 0.0
    )
    initial_compact_retention = (
        initial_compact_after / max(compact_before, 1e-8)
        if compact_before > 0.0
        else 1.0
    )
    initial_faint_compact_retention = (
        initial_faint_compact_after / max(faint_compact_before, 1e-8)
        if faint_compact_before > 0.0
        else 1.0
    )
    initial_diffuse_residual_ratio = (
        initial_diffuse_after / max(diffuse_before, 1e-8)
        if diffuse_before > 0.0
        else 0.0
    )

    diffuse_retry = {
        "attempted": False,
        "applied": False,
        "strength": 0.0,
        "max_strength": DIFFUSE_RETRY_MAX_STRENGTH,
        "target_ratio": max_diffuse_residual_ratio * DIFFUSE_RETRY_TARGET_MARGIN,
        "ratio_before": initial_diffuse_residual_ratio,
        "ratio_after": initial_diffuse_residual_ratio,
    }
    retention_safe = bool(
        initial_compact_retention >= min_compact_retention
        and initial_faint_compact_retention >= compact_floor - 0.01
    )
    if (
        initial_diffuse_residual_ratio > max_diffuse_residual_ratio
        and diffuse_before > 0.0
        and retention_safe
        and np.any(diffuse_mask)
    ):
        diffuse_retry["attempted"] = True
        retry_profile = np.zeros_like(gray, dtype=np.float32)
        retry_profile[diffuse_mask] = np.sqrt(
            np.clip(
                (0.05 - compact_protection[diffuse_mask]) / 0.05,
                0.0,
                1.0,
            )
        )
        removable_signal = float(
            np.sum(cleaned_gray[diffuse_mask] * retry_profile[diffuse_mask])
        )
        target_signal = (
            diffuse_before
            * max_diffuse_residual_ratio
            * DIFFUSE_RETRY_TARGET_MARGIN
        )
        retry_strength = min(
            DIFFUSE_RETRY_MAX_STRENGTH,
            max(
                0.0,
                (initial_diffuse_after - target_signal)
                / max(removable_signal, 1e-12),
            ),
        )
        if retry_strength > 1e-6:
            retry_weight = np.clip(
                1.0 - retry_profile * retry_strength,
                0.0,
                1.0,
            )
            cleaned_norm *= _weight_like(cleaned_norm, retry_weight)
            cleaned_rgb = _to_rgb_float_fullres(cleaned_norm)
            cleaned_gray = (
                0.2126 * cleaned_rgb[0]
                + 0.7152 * cleaned_rgb[1]
                + 0.0722 * cleaned_rgb[2]
            ).astype(np.float32)
            diffuse_retry["applied"] = True
            diffuse_retry["strength"] = retry_strength

    after_signal = float(np.sum(cleaned_norm))
    compact_after = (
        float(np.sum(cleaned_gray[compact_mask])) if np.any(compact_mask) else 0.0
    )
    faint_compact_after = (
        float(np.sum(cleaned_gray[faint_compact_mask]))
        if np.any(faint_compact_mask)
        else 0.0
    )
    diffuse_after = (
        float(np.sum(cleaned_gray[diffuse_mask])) if np.any(diffuse_mask) else 0.0
    )
    signal_ratio = after_signal / max(before_signal, 1e-8)
    compact_retention = (
        compact_after / max(compact_before, 1e-8) if compact_before > 0.0 else 1.0
    )
    faint_compact_retention = (
        faint_compact_after / max(faint_compact_before, 1e-8)
        if faint_compact_before > 0.0
        else 1.0
    )
    diffuse_residual_ratio = (
        diffuse_after / max(diffuse_before, 1e-8) if diffuse_before > 0.0 else 0.0
    )
    diffuse_retry["ratio_after"] = diffuse_residual_ratio

    issues = []
    if not np.all(np.isfinite(cleaned_norm)):
        issues.append("non-finite cleaned starmask pixels")
    if compact_retention < min_compact_retention:
        issues.append(
            f"compact_retention {compact_retention:.3f}<{min_compact_retention:.3f}"
        )
    if faint_compact_retention < compact_floor - 0.01:
        issues.append(
            "faint_compact_retention "
            f"{faint_compact_retention:.3f}<{compact_floor - 0.01:.3f}"
        )
    diffuse_hard_gate_failed = bool(
        diffuse_residual_ratio > max_diffuse_residual_ratio
    )
    if diffuse_hard_gate_failed:
        issues.append(
            "diffuse_residual_ratio "
            f"{diffuse_residual_ratio:.3f}>{max_diffuse_residual_ratio:.3f}"
        )

    metrics = {
        "background_median_before": background,
        "background_median_after": float(np.median(cleaned_gray[gray <= sample_limit])),
        "noise_sigma": noise_sigma,
        "signal_before": before_signal,
        "signal_after": after_signal,
        "signal_ratio": signal_ratio,
        "compact_pixels": int(np.count_nonzero(compact_mask)),
        "compact_retention": compact_retention,
        "faint_compact_pixels": int(np.count_nonzero(faint_compact_mask)),
        "faint_compact_retention": faint_compact_retention,
        "diffuse_pixels": int(np.count_nonzero(diffuse_mask)),
        "diffuse_residual_ratio": diffuse_residual_ratio,
        "diffuse_retry": diffuse_retry,
        "changed_pixel_ratio": float(np.mean(np.abs(cleaned_gray - gray) > 0.001)),
        "limits": {
            "min_compact_retention": min_compact_retention,
            "max_diffuse_residual_ratio": max_diffuse_residual_ratio,
        },
        "issues": issues,
        "diffuse_hard_gate_failed": diffuse_hard_gate_failed,
        "accepted": not issues,
    }
    return _restore_dtype(cleaned_norm, original, scale), metrics
