"""Deterministic, report-only multiscale noise measurements."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np


NOISE_MODEL_SCHEMA = "starun.multiscale-noise-model.v1"
DEFAULT_MAX_SIDE = 1024
DEFAULT_SCALES = (1, 2, 4, 8, 16)


def _as_chw_view(image: Any) -> np.ndarray:
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError("empty image")
    while source.ndim > 3:
        source = source[0]
    if source.ndim == 2:
        source = source[None, :, :]
    elif source.ndim == 3:
        first_is_channel = source.shape[0] in (1, 2, 3, 4)
        last_is_channel = source.shape[-1] in (1, 2, 3, 4)
        if first_is_channel:
            source = source[:3]
        elif last_is_channel:
            source = np.transpose(source[..., :3], (2, 0, 1))
        else:
            raise ValueError(f"unsupported image shape: {source.shape}")
    else:
        raise ValueError(f"unsupported image ndim: {source.ndim}")
    return source


def _bounded_sample(chw: np.ndarray, max_side: int) -> tuple[np.ndarray, int]:
    height, width = int(chw.shape[1]), int(chw.shape[2])
    step = max(1, int(math.ceil(max(height, width) / max(64, int(max_side)))))
    sampled = chw[:, ::step, ::step]
    original_dtype = sampled.dtype
    array = sampled.astype(np.float32, copy=True)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        array /= max(1.0, float(info.max))
    for channel in range(array.shape[0]):
        plane = array[channel]
        finite = np.isfinite(plane)
        replacement = float(np.median(plane[finite])) if np.any(finite) else 0.0
        array[channel] = np.where(finite, plane, replacement)
    return array, step


def _box_blur(plane: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = radius * 2 + 1
    padded = np.pad(plane, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
    ).cumsum(axis=0, dtype=np.float32).cumsum(axis=1, dtype=np.float32)
    total = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    )
    return (total / float(kernel * kernel)).astype(np.float32)


def _mad_sigma(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size < 16:
        return 0.0
    median = float(np.median(data))
    return float(1.4826 * np.median(np.abs(data - median)))


def _luminance(chw: np.ndarray) -> np.ndarray:
    if chw.shape[0] >= 3:
        return (
            0.2126 * chw[0] + 0.7152 * chw[1] + 0.0722 * chw[2]
        ).astype(np.float32)
    return chw[0].astype(np.float32, copy=False)


def _background_mask(luma: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    finite = np.isfinite(luma)
    values = luma[finite]
    if values.size < 64:
        raise ValueError("insufficient finite pixels")
    low = float(np.quantile(values, 0.005))
    high = float(np.quantile(values, 0.45))
    gradient = np.zeros_like(luma, dtype=np.float32)
    gradient[:, 1:] += np.abs(luma[:, 1:] - luma[:, :-1])
    gradient[1:, :] += np.abs(luma[1:, :] - luma[:-1, :])
    gradient_limit = float(np.quantile(gradient[finite], 0.65))
    mask = finite & (luma >= low) & (luma <= high) & (gradient <= gradient_limit)
    minimum = max(128, int(luma.size * 0.01))
    fallback = False
    if int(np.count_nonzero(mask)) < minimum:
        fallback = True
        high = float(np.quantile(values, 0.55))
        mask = finite & (luma >= low) & (luma <= high)
    if int(np.count_nonzero(mask)) < 32:
        raise ValueError("insufficient background samples")
    return mask, {
        "method": "low_signal_low_gradient",
        "fallback_used": fallback,
        "sample_count": int(np.count_nonzero(mask)),
        "coverage": float(np.mean(mask)),
        "luminance_low": low,
        "luminance_high": high,
        "gradient_limit": gradient_limit,
    }


def _scale_radii(shape: tuple[int, int], scales: Iterable[int]) -> list[int]:
    minimum_dimension = min(shape)
    radii = sorted(
        {
            max(1, int(radius))
            for radius in scales
            if int(radius) > 0 and int(radius) * 2 + 1 < minimum_dimension
        }
    )
    return radii or [1]


def _channel_covariance(chw: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    if chw.shape[0] < 2:
        return [[float(_mad_sigma(chw[0][mask]) ** 2)]]
    samples = np.stack([channel[mask] for channel in chw], axis=0)
    covariance = np.cov(samples, rowvar=True)
    covariance = np.atleast_2d(covariance)
    return [
        [float(value) if math.isfinite(float(value)) else 0.0 for value in row]
        for row in covariance
    ]


def _signal_noise_curve(
    luma: np.ndarray,
    residual: np.ndarray,
) -> list[Dict[str, Any]]:
    finite = np.isfinite(luma) & np.isfinite(residual)
    values = luma[finite]
    if values.size < 64:
        return []
    edges = np.quantile(values, np.linspace(0.0, 1.0, 6))
    curve: list[Dict[str, Any]] = []
    for index in range(5):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == 4:
            selected = finite & (luma >= lower) & (luma <= upper)
        else:
            selected = finite & (luma >= lower) & (luma < upper)
        curve.append(
            {
                "quantile_bin": index,
                "signal_min": lower,
                "signal_max": upper,
                "sample_count": int(np.count_nonzero(selected)),
                "noise_sigma": _mad_sigma(residual[selected]),
            }
        )
    return curve


def build_noise_model_report(
    image: Any,
    *,
    source_checkpoint: str,
    channel_semantics: str = "unknown",
    max_side: int = DEFAULT_MAX_SIDE,
    scales: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Measure noise without mutating the supplied image or selecting a denoiser."""
    chw = _as_chw_view(image)
    input_shape = [int(value) for value in chw.shape]
    sampled, sample_step = _bounded_sample(chw, max_side=max_side)
    luma = _luminance(sampled)
    mask, background = _background_mask(luma)
    radii = _scale_radii(luma.shape, scales or DEFAULT_SCALES)

    current_luma = luma.copy()
    current_channels = sampled.copy()
    scale_reports: list[Dict[str, Any]] = []
    first_luma_residual: Optional[np.ndarray] = None
    for radius in radii:
        smoothed_luma = _box_blur(current_luma, radius)
        luma_detail = current_luma - smoothed_luma
        smoothed_channels = np.stack(
            [_box_blur(channel, radius) for channel in current_channels],
            axis=0,
        )
        channel_detail = current_channels - smoothed_channels
        if first_luma_residual is None:
            first_luma_residual = luma_detail
        scale_reports.append(
            {
                "radius_pixels_in_sample": radius,
                "equivalent_radius_input_pixels": radius * sample_step,
                "luma_sigma": _mad_sigma(luma_detail[mask]),
                "channel_sigma": [
                    _mad_sigma(channel_detail[index][mask])
                    for index in range(channel_detail.shape[0])
                ],
                "background_detail_energy": float(
                    np.mean(np.square(luma_detail[mask], dtype=np.float64))
                ),
            }
        )
        current_luma = smoothed_luma
        current_channels = smoothed_channels

    channel_sigma = [
        _mad_sigma(sampled[index][mask]) for index in range(sampled.shape[0])
    ]
    luma_sigma = _mad_sigma(luma[mask])
    chroma_sigma: Dict[str, float] = {}
    if sampled.shape[0] >= 3:
        chroma_sigma = {
            "r_minus_g": _mad_sigma((sampled[0] - sampled[1])[mask]),
            "b_minus_g": _mad_sigma((sampled[2] - sampled[1])[mask]),
        }
    strongest_chroma = max(chroma_sigma.values(), default=0.0)
    advisory_mode = (
        "chroma_first"
        if strongest_chroma > max(luma_sigma * 1.25, 1e-8)
        else "luma_chroma_balanced"
        if sampled.shape[0] >= 3
        else "luminance"
    )

    return {
        "schema": NOISE_MODEL_SCHEMA,
        "mode": "report_only",
        "applied_to_pixels": False,
        "consumed_by_denoiser": False,
        "source_checkpoint": source_checkpoint,
        "channel_semantics": str(channel_semantics or "unknown"),
        "input": {
            "shape_chw": input_shape,
            "sampled_shape_chw": [int(value) for value in sampled.shape],
            "sample_step": sample_step,
            "max_side": int(max_side),
        },
        "background": background,
        "aggregate": {
            "luma_sigma": luma_sigma,
            "channel_sigma": channel_sigma,
            "chroma_sigma": chroma_sigma,
            "channel_covariance": _channel_covariance(sampled, mask),
        },
        "scales": scale_reports,
        "signal_noise_curve": _signal_noise_curve(
            luma,
            first_luma_residual
            if first_luma_residual is not None
            else luma - _box_blur(luma, 1),
        ),
        "future_advisory": {
            "mode": advisory_mode,
            "active": False,
            "reason": "Batch A records measurements only",
        },
    }


def _full_float_chw(image: Any) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(image)
    view = _as_chw_view(source)
    array, step = _bounded_sample(
        view,
        max_side=max(int(view.shape[1]), int(view.shape[2])),
    )
    if step != 1:
        raise RuntimeError("unexpected full-resolution sampling")
    return source, array


def _restore_like(source: np.ndarray, chw: np.ndarray) -> np.ndarray:
    restored: np.ndarray
    if source.ndim == 2:
        restored = chw[0]
    elif source.ndim == 3 and source.shape[0] in (1, 2, 3, 4):
        restored = chw[: source.shape[0]]
    elif source.ndim == 3 and source.shape[-1] in (1, 2, 3, 4):
        restored = np.transpose(chw[: source.shape[-1]], (1, 2, 0))
    else:
        raise ValueError(f"unsupported source image shape: {source.shape}")
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        return np.clip(restored * maximum, 0.0, maximum).astype(source.dtype)
    return restored.astype(np.float32, copy=False)


def _soft_threshold_multiscale(
    component: np.ndarray,
    background_mask: np.ndarray,
    *,
    radii: Iterable[int],
    threshold_multiplier: float,
    strength: float,
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    current = np.asarray(component, dtype=np.float32).copy()
    reports: list[Dict[str, Any]] = []
    for radius in _scale_radii(component.shape, radii):
        smooth = _box_blur(current, radius)
        detail = current - smooth
        sigma = _mad_sigma(detail[background_mask])
        threshold = sigma * max(0.0, threshold_multiplier) * strength
        magnitude = np.abs(detail)
        retained = np.maximum(magnitude - threshold, 0.0)
        shrunk = np.sign(detail) * retained
        current = smooth + shrunk
        reports.append(
            {
                "radius_pixels": radius,
                "sigma": sigma,
                "threshold": threshold,
                "retained_detail_ratio": float(
                    np.mean(retained[background_mask])
                    / max(float(np.mean(magnitude[background_mask])), 1e-9)
                ),
            }
        )
    return current.astype(np.float32), reports


def _quality_metrics(
    before: np.ndarray,
    after: np.ndarray,
) -> Dict[str, Any]:
    before_luma = _luminance(before)
    after_luma = _luminance(after)
    finite = np.isfinite(before_luma) & np.isfinite(after_luma)
    values = before_luma[finite]
    if values.size < 128:
        raise ValueError("insufficient finite pixels for denoise quality gate")
    q005, q45, q70, q97, q99 = (
        float(value)
        for value in np.quantile(values, (0.005, 0.45, 0.70, 0.97, 0.99))
    )
    background = finite & (before_luma >= q005) & (before_luma <= q45)
    signal = finite & (before_luma >= q70) & (before_luma <= q97)
    bright = finite & (before_luma >= q99)
    if np.count_nonzero(background) < 64:
        background = finite & (before_luma <= q70)
    if np.count_nonzero(signal) < 64:
        signal = finite

    before_detail = before_luma - _box_blur(before_luma, 1)
    after_detail = after_luma - _box_blur(after_luma, 1)
    before_noise = _mad_sigma(before_detail[background])
    after_noise = _mad_sigma(after_detail[background])
    noise_reduction = 1.0 - after_noise / max(before_noise, 1e-9)

    before_structure = _box_blur(before_luma, 2) - _box_blur(before_luma, 8)
    after_structure = _box_blur(after_luma, 2) - _box_blur(after_luma, 8)
    before_signal_energy = float(np.mean(np.abs(before_structure[signal])))
    after_signal_energy = float(np.mean(np.abs(after_structure[signal])))
    detail_retention = after_signal_energy / max(before_signal_energy, 1e-9)
    before_bright_ratio = float(np.mean(bright))
    after_bright_ratio = float(np.mean(after_luma >= q99))
    bright_spread_growth = (
        after_bright_ratio / max(before_bright_ratio, 1e-9) - 1.0
    )
    before_clip = float(np.mean((before <= 0.0) | (before >= 1.0)))
    after_clip = float(np.mean((after <= 0.0) | (after >= 1.0)))

    chroma_before = 0.0
    chroma_after = 0.0
    if before.shape[0] >= 3:
        before_chroma = (
            (before[0] - before[1]) + (before[2] - before[1])
        ) * 0.5
        after_chroma = (
            (after[0] - after[1]) + (after[2] - after[1])
        ) * 0.5
        chroma_before = _mad_sigma(
            (before_chroma - _box_blur(before_chroma, 1))[background]
        )
        chroma_after = _mad_sigma(
            (after_chroma - _box_blur(after_chroma, 1))[background]
        )

    return {
        "finite": bool(np.all(np.isfinite(after))),
        "background_luma_sigma_before": before_noise,
        "background_luma_sigma_after": after_noise,
        "background_noise_reduction": noise_reduction,
        "background_chroma_sigma_before": chroma_before,
        "background_chroma_sigma_after": chroma_after,
        "signal_detail_retention": detail_retention,
        "background_median_drift": abs(
            float(np.median(after_luma[background]))
            - float(np.median(before_luma[background]))
        ),
        "bright_spread_growth": bright_spread_growth,
        "clip_ratio_before": before_clip,
        "clip_ratio_after": after_clip,
        "clip_growth": after_clip - before_clip,
        "background_sample_count": int(np.count_nonzero(background)),
        "signal_sample_count": int(np.count_nonzero(signal)),
    }


def assess_denoise_candidate(
    before_image: Any,
    after_image: Any,
    *,
    detail_retention_min: float = 0.82,
    noise_reduction_min: float = 0.05,
    chroma_noise_growth_max: float = 1.05,
) -> Dict[str, Any]:
    """使用同一组指标验收任意 Stage 5 线性降噪候选。"""
    _before_source, before = _full_float_chw(before_image)
    _after_source, after = _full_float_chw(after_image)
    if before.shape != after.shape:
        raise ValueError(
            "denoise candidate shape changed: "
            f"before={before.shape}, after={after.shape}"
        )

    metrics = _quality_metrics(before, after)
    metrics["finite"] = bool(
        np.all(np.isfinite(np.asarray(after_image)))
    )
    chroma_growth: Optional[float] = None
    if before.shape[0] >= 3:
        chroma_before = float(metrics["background_chroma_sigma_before"])
        chroma_after = float(metrics["background_chroma_sigma_after"])
        if chroma_before <= 1e-9:
            chroma_growth = 1.0 if chroma_after <= 1e-9 else 1.0e9
        else:
            chroma_growth = chroma_after / chroma_before
    metrics["background_chroma_noise_growth"] = chroma_growth

    limits = {
        "detail_retention_min": max(
            0.70,
            min(0.98, float(detail_retention_min)),
        ),
        "noise_reduction_min": max(
            0.0,
            min(0.50, float(noise_reduction_min)),
        ),
        "chroma_noise_growth_max": max(
            1.0,
            min(1.50, float(chroma_noise_growth_max)),
        ),
        "clip_growth_max": 0.001,
        "background_median_drift_max": 0.003,
        "bright_spread_growth_max": 0.03,
    }
    issues: list[str] = []
    if not metrics["finite"]:
        issues.append("nonfinite_output")
    if metrics["clip_growth"] > limits["clip_growth_max"]:
        issues.append("clip_growth")
    if metrics["background_median_drift"] > limits["background_median_drift_max"]:
        issues.append("background_median_drift")
    if metrics["signal_detail_retention"] < limits["detail_retention_min"]:
        issues.append("signal_detail_retention")
    if metrics["bright_spread_growth"] > limits["bright_spread_growth_max"]:
        issues.append("bright_spread_growth")
    if (
        chroma_growth is not None
        and chroma_growth > limits["chroma_noise_growth_max"]
    ):
        issues.append("background_chroma_noise_growth")

    low_noise_input = metrics["background_luma_sigma_before"] <= 1e-5
    if (
        not low_noise_input
        and metrics["background_noise_reduction"] < limits["noise_reduction_min"]
    ):
        issues.append("insufficient_noise_reduction")
    accepted = not issues and not low_noise_input
    status = (
        "accepted"
        if accepted
        else "skipped_low_noise"
        if low_noise_input
        else "rejected"
    )
    return {
        "schema": "starun.denoise-quality-gate.v1",
        "status": status,
        "accepted": accepted,
        "metrics": metrics,
        "limits": limits,
        "issues": issues,
    }


def multiscale_denoise_candidate(
    image: Any,
    *,
    strength: float = 0.72,
    radii: Iterable[int] = (1, 2, 4),
    luma_threshold_multiplier: float = 1.35,
    chroma_threshold_multiplier: float = 1.90,
    detail_retention_min: float = 0.82,
    noise_reduction_min: float = 0.05,
    chroma_noise_growth_max: float = 1.05,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Build and gate a deterministic linear denoise candidate."""
    source, before = _full_float_chw(image)
    luma = _luminance(before)
    finite_luma = luma[np.isfinite(luma)]
    if finite_luma.size < 128:
        raise ValueError("insufficient image pixels")
    q005, q45, q95 = (
        float(value)
        for value in np.quantile(finite_luma, (0.005, 0.45, 0.95))
    )
    background = (
        np.isfinite(luma)
        & (luma >= q005)
        & (luma <= q45)
    )
    if np.count_nonzero(background) < 64:
        raise ValueError("insufficient background pixels")
    safe_strength = max(0.10, min(1.0, float(strength)))
    signal_weight = np.clip(
        (luma - q45) / max(q95 - q45, 1e-6),
        0.0,
        1.0,
    )
    blend = safe_strength * (1.0 - 0.78 * signal_weight)

    filtered_luma, luma_scales = _soft_threshold_multiscale(
        luma,
        background,
        radii=radii,
        threshold_multiplier=luma_threshold_multiplier,
        strength=safe_strength,
    )
    component_reports: Dict[str, Any] = {"luma": luma_scales}
    if before.shape[0] >= 3:
        red_green = before[0] - before[1]
        blue_green = before[2] - before[1]
        filtered_rg, rg_scales = _soft_threshold_multiscale(
            red_green,
            background,
            radii=radii,
            threshold_multiplier=chroma_threshold_multiplier,
            strength=safe_strength,
        )
        filtered_bg, bg_scales = _soft_threshold_multiscale(
            blue_green,
            background,
            radii=radii,
            threshold_multiplier=chroma_threshold_multiplier,
            strength=safe_strength,
        )
        green = filtered_luma - 0.2126 * filtered_rg - 0.0722 * filtered_bg
        filtered = np.stack(
            (green + filtered_rg, green, green + filtered_bg),
            axis=0,
        ).astype(np.float32)
        if before.shape[0] > 3:
            filtered = np.concatenate((filtered, before[3:]), axis=0)
        component_reports.update(
            {
                "red_minus_green": rg_scales,
                "blue_minus_green": bg_scales,
            }
        )
    else:
        filtered = filtered_luma[None, :, :]

    candidate = before * (1.0 - blend[None, :, :]) + filtered * blend[None, :, :]
    candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)
    gate = assess_denoise_candidate(
        before,
        candidate,
        detail_retention_min=detail_retention_min,
        noise_reduction_min=noise_reduction_min,
        chroma_noise_growth_max=chroma_noise_growth_max,
    )
    report = {
        "schema": "starun.multiscale-denoise-candidate.v1",
        "status": gate["status"],
        "accepted": gate["accepted"],
        "algorithm": "luma_opponent_chroma_multiscale_soft_threshold",
        "strength": safe_strength,
        "radii": [int(value) for value in radii],
        "component_scales": component_reports,
        "metrics": gate["metrics"],
        "limits": gate["limits"],
        "issues": gate["issues"],
        "quality_gate_schema": gate["schema"],
        "transaction": {
            "baseline": "stage5_pre_denoise.fit",
            "candidate": "stage5_multiscale_candidate.fit",
            "rollback_required_on_rejection": True,
        },
    }
    return _restore_like(source, candidate), report


__all__ = [
    "assess_denoise_candidate",
    "build_noise_model_report",
    "multiscale_denoise_candidate",
]
