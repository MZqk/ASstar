"""Deterministic, report-only multiscale noise measurements."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


NOISE_MODEL_SCHEMA = "starun.multiscale-noise-model.v2"
NOISE_MODEL_VALIDATION_SCHEMA = "starun.multiscale-noise-model-validation.v1"
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


def noise_model_pixel_sha256(image: Any) -> str:
    """Return a canonical full-resolution CHW float32 pixel digest."""

    source = _as_chw_view(image)
    original_dtype = source.dtype
    canonical = np.ascontiguousarray(source.astype("<f4", copy=True))
    if np.issubdtype(original_dtype, np.integer):
        canonical /= max(1.0, float(np.iinfo(original_dtype).max))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def noise_model_mask_sha256(mask: Any) -> str:
    """Bind a frozen two-dimensional support mask including its shape."""

    canonical = np.asarray(mask, dtype=bool)
    if canonical.ndim != 2 or canonical.size == 0:
        raise ValueError("noise model background mask must be a nonempty 2-D array")
    packed = np.packbits(
        np.ascontiguousarray(canonical).reshape(-1),
        bitorder="little",
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in canonical.shape)).encode("ascii"))
    digest.update(packed.tobytes(order="C"))
    return digest.hexdigest()


def _model_digest_payload(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Select immutable measurement fields covered by the model digest."""

    return {
        "schema": report.get("schema"),
        "source_checkpoint": report.get("source_checkpoint"),
        "channel_semantics": report.get("channel_semantics"),
        "input": report.get("input"),
        "background": report.get("background"),
        "aggregate": report.get("aggregate"),
        "scales": report.get("scales"),
    }


def noise_model_digest_sha256(report: Mapping[str, Any]) -> str:
    """Return the canonical digest for immutable model measurements."""

    encoded = json.dumps(
        _model_digest_payload(report),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_noise_model_report(
    report: Any,
    *,
    image: Any,
    background_mask: Any,
    source_checkpoint: str,
) -> Dict[str, Any]:
    """Validate all bindings and every sigma consumed by the denoiser."""

    issues: list[str] = []
    payload = dict(report) if isinstance(report, Mapping) else {}
    expected_shape = [int(value) for value in _as_chw_view(image).shape]
    mask = np.asarray(background_mask, dtype=bool)
    spatial_shape = tuple(expected_shape[-2:])
    if mask.shape != spatial_shape:
        issues.append("background_mask_shape_mismatch")
    elif int(np.count_nonzero(mask)) < 64:
        issues.append("background_mask_support_insufficient")

    if payload.get("schema") != NOISE_MODEL_SCHEMA:
        issues.append("noise_model_schema_mismatch")
    if str(payload.get("source_checkpoint") or "") != str(source_checkpoint):
        issues.append("noise_model_source_checkpoint_mismatch")

    input_binding = payload.get("input")
    if not isinstance(input_binding, Mapping):
        issues.append("noise_model_input_binding_missing")
    else:
        if list(input_binding.get("shape_chw") or []) != expected_shape:
            issues.append("noise_model_input_shape_mismatch")
        expected_pixel_sha = str(input_binding.get("pixel_sha256") or "")
        if (
            len(expected_pixel_sha) != 64
            or expected_pixel_sha != noise_model_pixel_sha256(image)
        ):
            issues.append("noise_model_input_pixel_sha256_mismatch")

    background_binding = payload.get("background")
    if not isinstance(background_binding, Mapping):
        issues.append("noise_model_background_binding_missing")
    elif mask.shape == spatial_shape:
        expected_mask_sha = str(background_binding.get("mask_sha256") or "")
        if (
            len(expected_mask_sha) != 64
            or expected_mask_sha != noise_model_mask_sha256(mask)
        ):
            issues.append("noise_model_background_mask_sha256_mismatch")

    scales = payload.get("scales")
    scale_count = len(scales) if isinstance(scales, list) else 0
    if not isinstance(scales, list) or not scales:
        issues.append("noise_model_scales_missing")
    else:
        channel_count = int(expected_shape[0])
        seen_equivalent_radii: set[int] = set()
        for index, scale in enumerate(scales):
            if not isinstance(scale, Mapping):
                issues.append(f"noise_model_scale_{index}_invalid")
                continue
            try:
                sample_radius = int(scale.get("radius_pixels_in_sample"))
                equivalent_radius = int(
                    scale.get("equivalent_radius_input_pixels")
                )
                luma_sigma = float(scale.get("luma_sigma"))
            except (TypeError, ValueError):
                issues.append(f"noise_model_scale_{index}_invalid")
                continue
            if sample_radius <= 0 or equivalent_radius <= 0:
                issues.append(f"noise_model_scale_{index}_radius_invalid")
            if equivalent_radius in seen_equivalent_radii:
                issues.append(f"noise_model_scale_{index}_radius_duplicate")
            seen_equivalent_radii.add(equivalent_radius)
            if not math.isfinite(luma_sigma) or luma_sigma <= 0.0:
                issues.append(f"noise_model_scale_{index}_luma_sigma_invalid")
            channel_sigma = scale.get("channel_sigma")
            channel_sigma_valid = bool(
                isinstance(channel_sigma, list)
                and len(channel_sigma) >= channel_count
            )
            if channel_sigma_valid:
                try:
                    channel_sigma_valid = all(
                        math.isfinite(float(value)) and float(value) > 0.0
                        for value in channel_sigma[:channel_count]
                    )
                except (TypeError, ValueError):
                    channel_sigma_valid = False
            if not channel_sigma_valid:
                issues.append(f"noise_model_scale_{index}_channel_sigma_invalid")
            if channel_count >= 3:
                opponent = scale.get("opponent_sigma")
                if not isinstance(opponent, Mapping):
                    issues.append(
                        f"noise_model_scale_{index}_opponent_sigma_missing"
                    )
                else:
                    for key in ("r_minus_g", "b_minus_g"):
                        try:
                            sigma = float(opponent.get(key))
                        except (TypeError, ValueError):
                            sigma = float("nan")
                        if not math.isfinite(sigma) or sigma <= 0.0:
                            issues.append(
                                f"noise_model_scale_{index}_{key}_sigma_invalid"
                            )

    expected_digest = str(payload.get("model_digest_sha256") or "")
    try:
        actual_digest = noise_model_digest_sha256(payload)
    except (TypeError, ValueError):
        actual_digest = ""
    if (
        len(expected_digest) != 64
        or not actual_digest
        or expected_digest != actual_digest
    ):
        issues.append("noise_model_digest_sha256_mismatch")

    unique_issues = list(dict.fromkeys(issues))
    return {
        "schema": NOISE_MODEL_VALIDATION_SCHEMA,
        "status": "accepted" if not unique_issues else "rejected",
        "accepted": not unique_issues,
        "issues": unique_issues,
        "source_checkpoint": source_checkpoint,
        "input_pixel_sha256": (
            (payload.get("input") or {}).get("pixel_sha256")
            if isinstance(payload.get("input"), Mapping)
            else None
        ),
        "background_mask_sha256": (
            (payload.get("background") or {}).get("mask_sha256")
            if isinstance(payload.get("background"), Mapping)
            else None
        ),
        "model_digest_sha256": expected_digest or None,
        "scale_count": scale_count,
    }


def build_noise_model_report(
    image: Any,
    *,
    source_checkpoint: str,
    channel_semantics: str = "unknown",
    max_side: int = DEFAULT_MAX_SIDE,
    scales: Optional[Iterable[int]] = None,
    background_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Measure noise without mutating the supplied image or selecting a denoiser."""
    chw = _as_chw_view(image)
    input_shape = [int(value) for value in chw.shape]
    sampled, sample_step = _bounded_sample(chw, max_side=max_side)
    luma = _luminance(sampled)
    if background_mask is None:
        mask, background = _background_mask(luma)
        background["source"] = "candidate_quantile_fallback"
        background["mask_sha256"] = None
        background["full_resolution_sample_count"] = None
    else:
        full_mask = np.asarray(background_mask, dtype=bool)
        if full_mask.shape != tuple(input_shape[-2:]):
            raise ValueError(
                "noise model background mask shape changed: "
                f"mask={full_mask.shape}, image={tuple(input_shape[-2:])}"
            )
        if int(np.count_nonzero(full_mask)) < 64:
            raise ValueError("noise model background mask has insufficient support")
        mask = full_mask[::sample_step, ::sample_step].copy()
        mask &= np.isfinite(luma)
        if int(np.count_nonzero(mask)) < 32:
            raise ValueError(
                "noise model sampled background mask has insufficient support"
            )
        background = {
            "method": "frozen_stage3_spatial_background_lineage",
            "source": "stage3_spatial_background_lineage",
            "fallback_used": False,
            "sample_count": int(np.count_nonzero(mask)),
            "coverage": float(np.mean(mask)),
            "full_resolution_sample_count": int(np.count_nonzero(full_mask)),
            "mask_sha256": noise_model_mask_sha256(full_mask),
        }
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
        opponent_sigma: Dict[str, float] = {}
        if channel_detail.shape[0] >= 3:
            opponent_sigma = {
                "r_minus_g": _mad_sigma(
                    (channel_detail[0] - channel_detail[1])[mask]
                ),
                "b_minus_g": _mad_sigma(
                    (channel_detail[2] - channel_detail[1])[mask]
                ),
            }
        scale_reports.append(
            {
                "radius_pixels_in_sample": radius,
                "equivalent_radius_input_pixels": radius * sample_step,
                "luma_sigma": _mad_sigma(luma_detail[mask]),
                "channel_sigma": [
                    _mad_sigma(channel_detail[index][mask])
                    for index in range(channel_detail.shape[0])
                ],
                "opponent_sigma": opponent_sigma,
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

    report = {
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
            "pixel_sha256": noise_model_pixel_sha256(image),
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
    report["model_digest_sha256"] = noise_model_digest_sha256(report)
    return report


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
    frozen_sigmas: Iterable[float],
    threshold_multiplier: float,
    strength: float,
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    current = np.asarray(component, dtype=np.float32).copy()
    reports: list[Dict[str, Any]] = []
    resolved_radii = _scale_radii(component.shape, radii)
    resolved_sigmas = [float(value) for value in frozen_sigmas]
    if len(resolved_sigmas) != len(resolved_radii):
        raise ValueError("frozen noise model scale count does not match denoiser radii")
    for radius, sigma in zip(resolved_radii, resolved_sigmas):
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("frozen noise model sigma must be finite and positive")
        smooth = _box_blur(current, radius)
        detail = current - smooth
        threshold = sigma * max(0.0, threshold_multiplier) * strength
        magnitude = np.abs(detail)
        retained = np.maximum(magnitude - threshold, 0.0)
        shrunk = np.sign(detail) * retained
        current = smooth + shrunk
        reports.append(
            {
                "radius_pixels": radius,
                "sigma": sigma,
                "sigma_source": "frozen_multiscale_noise_model",
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
    *,
    background_mask: Optional[np.ndarray] = None,
    signal_mask: Optional[np.ndarray] = None,
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
    background_source = "quantile_fallback"
    signal_source = "quantile_fallback"
    background = finite & (before_luma >= q005) & (before_luma <= q45)
    signal = finite & (before_luma >= q70) & (before_luma <= q97)
    if background_mask is not None:
        frozen_background = np.asarray(background_mask, dtype=bool)
        if frozen_background.shape != before_luma.shape:
            raise ValueError(
                "frozen background mask shape changed: "
                f"mask={frozen_background.shape}, image={before_luma.shape}"
            )
        frozen_background &= finite
        if int(np.count_nonzero(frozen_background)) < 64:
            raise ValueError("frozen background mask has insufficient support")
        background = frozen_background
        background_source = "stage3_spatial_background_lineage"
    if signal_mask is not None:
        frozen_signal = np.asarray(signal_mask, dtype=bool)
        if frozen_signal.shape != before_luma.shape:
            raise ValueError(
                "frozen signal mask shape changed: "
                f"mask={frozen_signal.shape}, image={before_luma.shape}"
            )
        frozen_signal &= finite
        if int(np.count_nonzero(frozen_signal)) >= 64:
            signal = frozen_signal
            signal_source = "stage5_frozen_target_structure"
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
        "background_mask_source": background_source,
        "signal_mask_source": signal_source,
    }


def assess_denoise_candidate(
    before_image: Any,
    after_image: Any,
    *,
    detail_retention_min: float = 0.90,
    noise_reduction_min: float = 0.12,
    chroma_noise_growth_max: float = 1.05,
    background_mask: Optional[np.ndarray] = None,
    signal_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """使用同一组指标验收任意 Stage 5 线性降噪候选。"""
    _before_source, before = _full_float_chw(before_image)
    _after_source, after = _full_float_chw(after_image)
    if before.shape != after.shape:
        raise ValueError(
            "denoise candidate shape changed: "
            f"before={before.shape}, after={after.shape}"
        )

    metrics = _quality_metrics(
        before,
        after,
        background_mask=background_mask,
        signal_mask=signal_mask,
    )
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


def _frozen_component_sigmas(
    report: Mapping[str, Any],
    *,
    component: str,
    radii: Iterable[int],
    image_shape: tuple[int, int],
) -> tuple[list[float], list[Dict[str, int]]]:
    """Map requested full-resolution radii to nearest frozen model scales."""

    scale_records = list(report.get("scales") or [])
    requested_radii = _scale_radii(image_shape, radii)
    sigmas: list[float] = []
    mappings: list[Dict[str, int]] = []
    for radius in requested_radii:
        selected = min(
            scale_records,
            key=lambda record: (
                abs(
                    int(record["equivalent_radius_input_pixels"])
                    - int(radius)
                ),
                int(record["equivalent_radius_input_pixels"]),
            ),
        )
        if component == "luma":
            sigma = float(selected["luma_sigma"])
        elif component == "red_minus_green":
            sigma = float(selected["opponent_sigma"]["r_minus_g"])
        elif component == "blue_minus_green":
            sigma = float(selected["opponent_sigma"]["b_minus_g"])
        else:
            raise ValueError(f"unsupported frozen noise component: {component}")
        sigmas.append(sigma)
        mappings.append(
            {
                "denoiser_radius_pixels": int(radius),
                "model_equivalent_radius_input_pixels": int(
                    selected["equivalent_radius_input_pixels"]
                ),
            }
        )
    return sigmas, mappings


def multiscale_denoise_candidate(
    image: Any,
    *,
    strength: float = 0.72,
    radii: Iterable[int] = (1, 2, 4),
    luma_threshold_multiplier: float = 1.35,
    chroma_threshold_multiplier: float = 1.90,
    detail_retention_min: float = 0.90,
    noise_reduction_min: float = 0.12,
    chroma_noise_growth_max: float = 1.05,
    background_mask: Optional[np.ndarray] = None,
    signal_mask: Optional[np.ndarray] = None,
    noise_model_report: Optional[Dict[str, Any]] = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Build and gate a deterministic linear denoise candidate."""
    source, before = _full_float_chw(image)
    luma = _luminance(before)
    candidate_radii = tuple(_scale_radii(luma.shape, radii))
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
    background_source = "quantile_fallback"
    if background_mask is not None:
        frozen_background = np.asarray(background_mask, dtype=bool)
        if frozen_background.shape != luma.shape:
            raise ValueError(
                "frozen background mask shape changed: "
                f"mask={frozen_background.shape}, image={luma.shape}"
            )
        frozen_background &= np.isfinite(luma)
        if int(np.count_nonzero(frozen_background)) < 64:
            raise ValueError("frozen background mask has insufficient support")
        background = frozen_background
        background_source = "stage3_spatial_background_lineage"
    if np.count_nonzero(background) < 64:
        raise ValueError("insufficient background pixels")

    low_noise_probe = assess_denoise_candidate(
        before,
        before,
        detail_retention_min=detail_retention_min,
        noise_reduction_min=noise_reduction_min,
        chroma_noise_growth_max=chroma_noise_growth_max,
        background_mask=(background if background_mask is not None else None),
        signal_mask=signal_mask,
    )
    if low_noise_probe.get("status") == "skipped_low_noise":
        return _restore_like(source, before), {
            "schema": "starun.multiscale-denoise-candidate.v1",
            "status": "skipped_low_noise",
            "accepted": False,
            "algorithm": "luma_opponent_chroma_multiscale_soft_threshold",
            "strength": max(0.10, min(1.0, float(strength))),
            "radii": list(candidate_radii),
            "component_scales": {},
            "metrics": low_noise_probe["metrics"],
            "limits": low_noise_probe["limits"],
            "issues": low_noise_probe["issues"],
            "quality_gate_schema": low_noise_probe["schema"],
            "frozen_context": {
                "background_mask_source": background_source,
                "background_sample_count": int(np.count_nonzero(background)),
                "noise_model_verified": False,
                "noise_model_consumed": False,
                "reason": "verified_low_noise_skip",
            },
            "transaction": {
                "baseline": "stage5_pre_denoise.fit",
                "candidate": None,
                "rollback_required_on_rejection": False,
            },
        }

    frozen_noise_model = dict(noise_model_report or {})
    noise_model_validation = validate_noise_model_report(
        frozen_noise_model,
        image=before,
        background_mask=background,
        source_checkpoint="stage5_pre_denoise.fit",
    )
    if noise_model_validation.get("accepted") is not True:
        raise ValueError(
            "frozen noise model rejected: "
            + ",".join(noise_model_validation.get("issues") or ["unknown"])
        )

    safe_strength = max(0.10, min(1.0, float(strength)))
    signal_weight = np.clip(
        (luma - q45) / max(q95 - q45, 1e-6),
        0.0,
        1.0,
    )
    signal_source = "luminance_quantile"
    frozen_signal: Optional[np.ndarray] = None
    if signal_mask is not None:
        candidate_signal = np.asarray(signal_mask, dtype=bool)
        if candidate_signal.shape != luma.shape:
            raise ValueError(
                "frozen signal mask shape changed: "
                f"mask={candidate_signal.shape}, image={luma.shape}"
            )
        candidate_signal &= np.isfinite(luma)
        if int(np.count_nonzero(candidate_signal)) >= 64:
            frozen_signal = candidate_signal
            feathered_signal = _box_blur(candidate_signal.astype(np.float32), 3)
            signal_weight = np.maximum(signal_weight, feathered_signal)
            signal_source = "stage5_frozen_target_structure"
    blend = safe_strength * (1.0 - 0.92 * signal_weight)

    luma_sigmas, luma_mappings = _frozen_component_sigmas(
        frozen_noise_model,
        component="luma",
        radii=candidate_radii,
        image_shape=luma.shape,
    )

    filtered_luma, luma_scales = _soft_threshold_multiscale(
        luma,
        background,
        radii=candidate_radii,
        frozen_sigmas=luma_sigmas,
        threshold_multiplier=luma_threshold_multiplier,
        strength=safe_strength,
    )
    for scale_report, mapping in zip(luma_scales, luma_mappings):
        scale_report.update(mapping)
    component_reports: Dict[str, Any] = {"luma": luma_scales}
    if before.shape[0] >= 3:
        red_green = before[0] - before[1]
        blue_green = before[2] - before[1]
        red_green_sigmas, red_green_mappings = _frozen_component_sigmas(
            frozen_noise_model,
            component="red_minus_green",
            radii=candidate_radii,
            image_shape=luma.shape,
        )
        filtered_rg, rg_scales = _soft_threshold_multiscale(
            red_green,
            background,
            radii=candidate_radii,
            frozen_sigmas=red_green_sigmas,
            threshold_multiplier=chroma_threshold_multiplier,
            strength=safe_strength,
        )
        for scale_report, mapping in zip(rg_scales, red_green_mappings):
            scale_report.update(mapping)
        blue_green_sigmas, blue_green_mappings = _frozen_component_sigmas(
            frozen_noise_model,
            component="blue_minus_green",
            radii=candidate_radii,
            image_shape=luma.shape,
        )
        filtered_bg, bg_scales = _soft_threshold_multiscale(
            blue_green,
            background,
            radii=candidate_radii,
            frozen_sigmas=blue_green_sigmas,
            threshold_multiplier=chroma_threshold_multiplier,
            strength=safe_strength,
        )
        for scale_report, mapping in zip(bg_scales, blue_green_mappings):
            scale_report.update(mapping)
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
        background_mask=(background if background_mask is not None else None),
        signal_mask=frozen_signal,
    )
    report = {
        "schema": "starun.multiscale-denoise-candidate.v1",
        "status": gate["status"],
        "accepted": gate["accepted"],
        "algorithm": "luma_opponent_chroma_multiscale_soft_threshold",
        "strength": safe_strength,
        "radii": list(candidate_radii),
        "component_scales": component_reports,
        "metrics": gate["metrics"],
        "limits": gate["limits"],
        "issues": gate["issues"],
        "quality_gate_schema": gate["schema"],
        "frozen_context": {
            "background_mask_source": background_source,
            "background_sample_count": int(np.count_nonzero(background)),
            "signal_mask_source": signal_source,
            "signal_sample_count": int(
                np.count_nonzero(frozen_signal)
                if frozen_signal is not None
                else 0
            ),
            "noise_model_schema": frozen_noise_model.get("schema"),
            "noise_model_source_checkpoint": frozen_noise_model.get(
                "source_checkpoint"
            ),
            "noise_model_scale_count": len(
                frozen_noise_model.get("scales") or []
            ),
            "noise_model_verified": True,
            "noise_model_consumed": True,
            "noise_model_input_pixel_sha256": (
                frozen_noise_model.get("input") or {}
            ).get("pixel_sha256"),
            "noise_model_background_mask_sha256": (
                frozen_noise_model.get("background") or {}
            ).get("mask_sha256"),
            "noise_model_digest_sha256": frozen_noise_model.get(
                "model_digest_sha256"
            ),
            "noise_model_validation": noise_model_validation,
        },
        "transaction": {
            "baseline": "stage5_pre_denoise.fit",
            "candidate": "stage5_multiscale_candidate.fit",
            "rollback_required_on_rejection": True,
        },
    }
    return _restore_like(source, candidate), report


__all__ = [
    "NOISE_MODEL_SCHEMA",
    "assess_denoise_candidate",
    "build_noise_model_report",
    "multiscale_denoise_candidate",
    "noise_model_digest_sha256",
    "noise_model_mask_sha256",
    "noise_model_pixel_sha256",
    "validate_noise_model_report",
]
