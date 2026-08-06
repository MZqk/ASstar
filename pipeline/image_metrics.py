"""Image feature and quality metrics for the Seestar pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import ImageFeatures, QualityMetrics


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _dark_clip_threshold(bg_median: float, bg_mad: float) -> float:
    """Detect true clipped black pixels without flagging normal low linear sky."""
    return max(
        1e-6,
        min(
            0.002,
            bg_median * 0.45,
            bg_median - 8.0 * bg_mad,
        ),
    )


def _low_linear_signal_floor(
    bg_median: float,
    bg_mad: float,
    *,
    default_floor: float,
    mad_multiplier: float,
    median_multiplier: float,
    minimum: float,
) -> float:
    return min(
        default_floor,
        max(
            bg_mad * mad_multiplier,
            bg_median * median_multiplier,
            minimum,
        ),
    )


def _to_rgb_float_image(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("empty image data")

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[-1] not in (1, 3):
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] == 3 and arr.shape[-1] not in (1, 3):
            arr = arr[:3, :, :]
        elif arr.shape[-1] == 1:
            arr = np.repeat(np.transpose(arr, (2, 0, 1)), 3, axis=0)
        elif arr.shape[-1] >= 3:
            arr = np.transpose(arr[..., :3], (2, 0, 1))
        elif arr.shape[0] >= 3:
            arr = arr[:3, :, :]
        else:
            raise ValueError(f"unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"unsupported image ndim: {arr.ndim}")

    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)

    max_value = float(np.max(arr)) if arr.size else 0.0
    if max_value > 0.0:
        if max_value > 2.0:
            arr = arr / max_value
        else:
            arr = np.clip(arr, 0.0, 1.0)

    h, w = arr.shape[1], arr.shape[2]
    longest = max(h, w)
    if longest > max_side:
        step = int(np.ceil(longest / float(max_side)))
        arr = arr[:, ::step, ::step]
    return arr


def _to_rgb_float_fullres(image: np.ndarray) -> np.ndarray:
    # Use a very large max_side to keep original resolution.
    return _to_rgb_float_image(image, max_side=2_147_483_647)


def _component_areas(
    binary_mask: np.ndarray,
    *,
    min_area: int = 1,
    max_area: Optional[int] = None,
    max_components: int = 12000,
) -> List[int]:
    mask = np.asarray(binary_mask).astype(bool, copy=False)
    if mask.ndim != 2:
        return []

    coords = np.argwhere(mask)
    fg_count = int(coords.shape[0])
    if fg_count == 0:
        return []

    # 超大前景区域时直接近似，避免在 Python 层做百万级 flood fill。
    if fg_count > 450000:
        if fg_count >= min_area and (max_area is None or fg_count <= max_area):
            return [fg_count]
        return []

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    areas: List[int] = []

    for y, x in coords:
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = 1
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            ny = cy - 1
            if ny >= 0 and mask[ny, cx] and not visited[ny, cx]:
                visited[ny, cx] = 1
                stack.append((ny, cx))
            ny = cy + 1
            if ny < h and mask[ny, cx] and not visited[ny, cx]:
                visited[ny, cx] = 1
                stack.append((ny, cx))
            nx = cx - 1
            if nx >= 0 and mask[cy, nx] and not visited[cy, nx]:
                visited[cy, nx] = 1
                stack.append((cy, nx))
            nx = cx + 1
            if nx < w and mask[cy, nx] and not visited[cy, nx]:
                visited[cy, nx] = 1
                stack.append((cy, nx))

        if area >= min_area and (max_area is None or area <= max_area):
            areas.append(area)
            if len(areas) >= max_components:
                break
    return areas


def measure_image_features(image: np.ndarray) -> ImageFeatures:
    """
    测量自动调参需要的关键图像特征。
    任何异常都回退为保守默认值，保证流程可继续。
    """
    defaults = ImageFeatures()
    feat = ImageFeatures()
    try:
        rgb = _to_rgb_float_image(image)
        r, g, b = rgb[0], rgb[1], rgb[2]
        gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        image_area = max(1, int(gray.size))

        bg_threshold = float(np.quantile(gray, 0.22))
        bg_mask = gray <= bg_threshold
        if int(np.count_nonzero(bg_mask)) < 64:
            bg_mask = gray <= float(np.quantile(gray, 0.30))
        bg_values = gray[bg_mask] if np.any(bg_mask) else gray.reshape(-1)
        feat.bg_median = float(np.median(bg_values))
        feat.bg_std = float(np.std(bg_values))
        bg_mad = float(np.median(np.abs(bg_values - feat.bg_median)))

        signal_threshold = max(
            float(np.quantile(gray, 0.55)),
            feat.bg_median + 1.2 * feat.bg_std
        )
        signal_mask = gray > signal_threshold
        if int(np.count_nonzero(signal_mask)) < 128:
            signal_mask = gray > float(np.quantile(gray, 0.50))
        eps = 1e-6
        if np.any(signal_mask):
            g_signal = g[signal_mask] + eps
            feat.red_dominance = float(
                np.median((r[signal_mask] + eps) / g_signal)
            )
            feat.blue_dominance = float(
                np.median((b[signal_mask] + eps) / g_signal)
            )

        object_floor = _low_linear_signal_floor(
            feat.bg_median,
            bg_mad,
            default_floor=0.020,
            mad_multiplier=6.0,
            median_multiplier=0.08,
            minimum=0.00015,
        )
        object_threshold = max(
            float(np.quantile(gray, 0.70)),
            feat.bg_median + max(1.8 * feat.bg_std, object_floor)
        )
        object_mask = gray > object_threshold
        object_pixels = int(np.count_nonzero(object_mask))
        feat.object_area_ratio = object_pixels / float(image_area)

        object_areas = _component_areas(object_mask, min_area=1, max_components=6000)
        diffuse_area_limit = max(24, int(image_area * 0.0015))
        if object_pixels > 0 and object_areas:
            diffuse_pixels = sum(a for a in object_areas if a >= diffuse_area_limit)
            feat.diffuse_ratio = diffuse_pixels / float(object_pixels)

        core_threshold = max(
            float(np.quantile(gray, 0.992)),
            feat.bg_median + max(6.0 * feat.bg_std, 0.08)
        )
        if object_pixels > 0:
            core_pixels = int(np.count_nonzero(object_mask & (gray > core_threshold)))
            feat.core_brightness_ratio = core_pixels / float(object_pixels)
        else:
            core_pixels = int(np.count_nonzero(gray > core_threshold))
            feat.core_brightness_ratio = core_pixels / float(image_area)

        star_threshold = max(
            float(np.quantile(gray, 0.985)),
            feat.bg_median + max(4.0 * feat.bg_std, 0.05)
        )
        star_mask = gray > star_threshold
        max_star_area = max(4, int(image_area * 0.0006))
        star_areas = _component_areas(
            star_mask,
            min_area=1,
            max_area=max_star_area,
            max_components=18000,
        )
        feat.star_density = len(star_areas) / float(image_area)
        if star_areas:
            median_area = float(np.median(star_areas))
            feat.median_star_size = 2.0 * np.sqrt(median_area / np.pi)

        edge_w = max(2, int(min(gray.shape) * 0.05))
        top = gray[:edge_w, :].reshape(-1)
        bottom = gray[-edge_w:, :].reshape(-1)
        left = gray[:, :edge_w].reshape(-1)
        right = gray[:, -edge_w:].reshape(-1)
        edge_values = np.concatenate([top, bottom, left, right], axis=0)
        global_dark_threshold = _dark_clip_threshold(feat.bg_median, bg_mad)
        feat.global_dark_ratio = float(np.mean(gray <= global_dark_threshold))
        edge_black_threshold = global_dark_threshold
        if edge_values.size:
            feat.edge_black_ratio = float(np.mean(edge_values <= edge_black_threshold))
    except (TypeError, ValueError, IndexError, FloatingPointError):
        feat = defaults

    # 统一清洗和限幅，保证返回值总是有效。
    feat.bg_median = _clamp_float(feat.bg_median, 0.0, 1.0)
    feat.bg_std = _clamp_float(feat.bg_std, 0.0, 1.0)
    feat.red_dominance = _clamp_float(feat.red_dominance, 0.2, 4.0)
    feat.blue_dominance = _clamp_float(feat.blue_dominance, 0.2, 4.0)
    feat.star_density = _clamp_float(feat.star_density, 0.0, 0.2)
    feat.median_star_size = _clamp_float(feat.median_star_size, 0.2, 64.0)
    feat.object_area_ratio = _clamp_float(feat.object_area_ratio, 0.0, 1.0)
    feat.diffuse_ratio = _clamp_float(feat.diffuse_ratio, 0.0, 1.0)
    feat.core_brightness_ratio = _clamp_float(feat.core_brightness_ratio, 0.0, 1.0)
    feat.edge_black_ratio = _clamp_float(feat.edge_black_ratio, 0.0, 1.0)
    feat.global_dark_ratio = _clamp_float(feat.global_dark_ratio, 0.0, 1.0)

    for key, value in asdict(feat).items():
        if not np.isfinite(value):
            setattr(feat, key, getattr(defaults, key))
    return feat


def measure_stage3_signal_preservation(
    before_image: np.ndarray,
    after_image: np.ndarray,
    *,
    sky_points: Optional[List[Tuple[float, float]]] = None,
    sky_patch_radius: int = 12,
) -> Dict[str, Any]:
    """Measure source fidelity without treating a sky-level shift as flux loss.

    Stage 3 is expected to change the additive background level.  The legacy
    raw-mean diagnostic is therefore retained for compatibility, but target
    flux and morphology are measured relative to the *same sky coordinates*
    before and after correction.
    """
    result: Dict[str, Any] = {
        "available": False,
        "star_retention_ratio": None,
        "before_star_count": 0,
        "after_star_count": 0,
        "nebula_mean_change_ratio": None,
        "before_nebula_mean": None,
        "after_nebula_mean": None,
        "nebula_pixel_count": 0,
        "before_sky_median": None,
        "after_sky_median": None,
        "before_sky_rms": None,
        "after_sky_rms": None,
        "before_target_flux": None,
        "after_target_flux": None,
        "target_flux_retention_ratio": None,
        "target_flux_change_significance": None,
        "target_morphology_correlation": None,
        "target_centroid_shift_pixels": None,
        "target_centroid_shift_fraction": None,
        "target_change_residual_rms": None,
        "target_change_residual_significance": None,
        "target_sky_reference": "fixed_before_sky_coordinates",
        "heldout_sky_model": None,
        "fidelity_method": (
            "fixed_before_sky_coordinates_and_background_referenced_target_mask"
        ),
        "notes": [],
    }
    try:
        before_rgb = _to_rgb_float_image(before_image)
        after_rgb = _to_rgb_float_image(after_image)
        if before_rgb.shape != after_rgb.shape:
            result["notes"].append(
                f"shape mismatch: before={before_rgb.shape}, after={after_rgb.shape}"
            )
            return result

        before_gray = (
            0.2126 * before_rgb[0] + 0.7152 * before_rgb[1] + 0.0722 * before_rgb[2]
        ).astype(np.float32)
        after_gray = (
            0.2126 * after_rgb[0] + 0.7152 * after_rgb[1] + 0.0722 * after_rgb[2]
        ).astype(np.float32)
        image_area = max(1, int(before_gray.size))

        source = np.asarray(before_image)
        if source.ndim == 2:
            source_height, source_width = source.shape
        elif source.ndim == 3 and source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
            source_height, source_width = source.shape[1], source.shape[2]
        elif source.ndim == 3:
            source_height, source_width = source.shape[0], source.shape[1]
        else:
            source_height, source_width = before_gray.shape

        bg_threshold = float(np.quantile(before_gray, 0.22))
        bg_mask = before_gray <= bg_threshold
        if int(np.count_nonzero(bg_mask)) < 64:
            bg_mask = before_gray <= float(np.quantile(before_gray, 0.30))
        bg_values = before_gray[bg_mask] if np.any(bg_mask) else before_gray.reshape(-1)
        after_bg_values = (
            after_gray[bg_mask]
            if np.any(bg_mask)
            else after_gray.reshape(-1)
        )
        bg_median = float(np.median(bg_values))
        after_bg_median = float(np.median(after_bg_values))
        bg_std = float(np.std(bg_values))
        bg_mad = float(np.median(np.abs(bg_values - bg_median)))
        after_bg_mad = float(
            np.median(np.abs(after_bg_values - after_bg_median))
        )
        before_sky_rms = 1.4826 * bg_mad
        after_sky_rms = 1.4826 * after_bg_mad
        result["before_sky_median"] = bg_median
        result["after_sky_median"] = after_bg_median
        result["before_sky_rms"] = before_sky_rms
        result["after_sky_rms"] = after_sky_rms

        star_threshold = max(
            float(np.quantile(before_gray, 0.985)),
            bg_median + max(4.0 * bg_std, 0.05),
        )
        max_star_area = max(4, int(image_area * 0.0008))
        before_star_mask = before_gray > star_threshold
        # Keep the detection contrast fixed relative to each image's measured
        # sky pedestal.  A legitimate additive subtraction must not turn into
        # an apparent loss of stars merely because the absolute level moved.
        star_contrast = max(star_threshold - bg_median, 0.0)
        after_star_mask = after_gray > (after_bg_median + star_contrast)
        before_star_areas = _component_areas(
            before_star_mask,
            min_area=1,
            max_area=max_star_area,
            max_components=18000,
        )
        after_star_areas = _component_areas(
            after_star_mask,
            min_area=1,
            max_area=max_star_area,
            max_components=18000,
        )
        before_star_count = len(before_star_areas)
        after_star_count = len(after_star_areas)
        result["before_star_count"] = before_star_count
        result["after_star_count"] = after_star_count
        if before_star_count >= 8:
            result["star_retention_ratio"] = after_star_count / float(before_star_count)
        else:
            result["notes"].append("star retention skipped: too few stars")

        object_floor = _low_linear_signal_floor(
            bg_median,
            bg_mad,
            default_floor=0.020,
            mad_multiplier=6.0,
            median_multiplier=0.08,
            minimum=0.00015,
        )
        object_threshold = max(
            float(np.quantile(before_gray, 0.70)),
            bg_median + max(1.8 * bg_std, object_floor),
        )
        object_mask = before_gray > object_threshold
        object_areas = _component_areas(object_mask, min_area=1, max_components=6000)
        diffuse_area_limit = max(24, int(image_area * 0.0015))
        diffuse_mask = np.zeros_like(object_mask, dtype=bool)
        if object_areas:
            # Rebuild an approximate diffuse mask by excluding compact star-like highlights.
            diffuse_mask = object_mask & ~before_star_mask
            if int(np.count_nonzero(diffuse_mask)) < diffuse_area_limit:
                diffuse_mask = np.zeros_like(object_mask, dtype=bool)
        nebula_count = int(np.count_nonzero(diffuse_mask))
        result["nebula_pixel_count"] = nebula_count
        if nebula_count >= diffuse_area_limit:
            before_mean = float(np.mean(before_gray[diffuse_mask]))
            after_mean = float(np.mean(after_gray[diffuse_mask]))
            result["before_nebula_mean"] = before_mean
            result["after_nebula_mean"] = after_mean
            result["nebula_mean_change_ratio"] = abs(after_mean - before_mean) / max(
                before_mean,
                1e-6,
            )

            before_signal = before_gray[diffuse_mask].astype(np.float64) - bg_median
            after_signal = after_gray[diffuse_mask].astype(np.float64) - after_bg_median
            flux_systematic_rms = float(np.hypot(before_sky_rms, after_sky_rms))

            def fit_heldout_sky_plane(gray: np.ndarray) -> Optional[Dict[str, Any]]:
                if not sky_points or len(sky_points) < 6:
                    return None
                analysis_height, analysis_width = gray.shape
                radius_scale = min(
                    analysis_width / max(float(source_width), 1.0),
                    analysis_height / max(float(source_height), 1.0),
                )
                radius = max(2, int(round(float(sky_patch_radius) * radius_scale)))
                records: List[Tuple[float, float, float, float]] = []
                for raw_x, raw_y in sky_points:
                    x = int(round(
                        float(raw_x)
                        * max(analysis_width - 1, 1)
                        / max(source_width - 1, 1)
                    ))
                    top_y = max(source_height - 1, 1) - float(raw_y)
                    y = int(round(
                        top_y
                        * max(analysis_height - 1, 1)
                        / max(source_height - 1, 1)
                    ))
                    if (
                        x - radius < 0
                        or x + radius >= analysis_width
                        or y - radius < 0
                        or y + radius >= analysis_height
                    ):
                        continue
                    patch = gray[
                        y - radius : y + radius + 1,
                        x - radius : x + radius + 1,
                    ].astype(np.float64)
                    if not np.all(np.isfinite(patch)):
                        continue
                    median = float(np.median(patch))
                    mad = 1.4826 * float(np.median(np.abs(patch - median)))
                    records.append(
                        (
                            x / max(analysis_width - 1, 1),
                            y / max(analysis_height - 1, 1),
                            median,
                            mad,
                        )
                    )
                if len(records) < 6:
                    return None
                design = np.asarray(
                    [[1.0, record[0], record[1]] for record in records],
                    dtype=np.float64,
                )
                values = np.asarray([record[2] for record in records], dtype=np.float64)
                try:
                    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                residual = values - design @ coefficients
                residual_center = float(np.median(residual))
                residual_rms = 1.4826 * float(
                    np.median(np.abs(residual - residual_center))
                )
                return {
                    "coefficients": coefficients,
                    "sample_count": len(records),
                    "residual_rms": residual_rms,
                    "patch_rms": float(np.median([record[3] for record in records])),
                }

            before_plane = fit_heldout_sky_plane(before_gray)
            after_plane = fit_heldout_sky_plane(after_gray)
            if before_plane is not None and after_plane is not None:
                target_y, target_x = np.nonzero(diffuse_mask)
                target_design = np.column_stack(
                    (
                        np.ones(target_x.size, dtype=np.float64),
                        target_x / max(before_gray.shape[1] - 1, 1),
                        target_y / max(before_gray.shape[0] - 1, 1),
                    )
                )
                before_signal = (
                    before_gray[diffuse_mask].astype(np.float64)
                    - target_design @ before_plane["coefficients"]
                )
                after_signal = (
                    after_gray[diffuse_mask].astype(np.float64)
                    - target_design @ after_plane["coefficients"]
                )
                flux_systematic_rms = float(
                    np.hypot(
                        max(
                            float(before_plane["residual_rms"]),
                            float(before_plane["patch_rms"]),
                        ),
                        max(
                            float(after_plane["residual_rms"]),
                            float(after_plane["patch_rms"]),
                        ),
                    )
                )
                result["target_sky_reference"] = "heldout_sky_plane_degree_1"
                result["fidelity_method"] = (
                    "heldout_sky_plane_and_fixed_before_target_mask"
                )
                result["heldout_sky_model"] = {
                    "sample_count": int(before_plane["sample_count"]),
                    "before_coefficients": [
                        float(value) for value in before_plane["coefficients"]
                    ],
                    "after_coefficients": [
                        float(value) for value in after_plane["coefficients"]
                    ],
                    "before_residual_rms": float(before_plane["residual_rms"]),
                    "after_residual_rms": float(after_plane["residual_rms"]),
                    "before_patch_rms": float(before_plane["patch_rms"]),
                    "after_patch_rms": float(after_plane["patch_rms"]),
                }

            before_flux = float(np.sum(before_signal))
            after_flux = float(np.sum(after_signal))
            result["before_target_flux"] = before_flux
            result["after_target_flux"] = after_flux
            if before_flux > 1e-12:
                result["target_flux_retention_ratio"] = after_flux / before_flux

            # Background-model uncertainty is spatially correlated, so use a
            # conservative N * RMS uncertainty rather than sqrt(N) pixel noise.
            flux_uncertainty = float(nebula_count) * flux_systematic_rms
            if flux_uncertainty > 1e-12:
                result["target_flux_change_significance"] = (
                    after_flux - before_flux
                ) / flux_uncertainty

            before_centered = before_signal - float(np.mean(before_signal))
            after_centered = after_signal - float(np.mean(after_signal))
            correlation_denominator = float(
                np.linalg.norm(before_centered) * np.linalg.norm(after_centered)
            )
            if correlation_denominator > 1e-12:
                result["target_morphology_correlation"] = float(
                    np.dot(before_centered, after_centered)
                    / correlation_denominator
                )

            target_change = after_signal - before_signal
            target_change_center = float(np.median(target_change))
            target_change_rms = 1.4826 * float(
                np.median(np.abs(target_change - target_change_center))
            )
            result["target_change_residual_rms"] = target_change_rms
            if flux_systematic_rms > 1e-12:
                result["target_change_residual_significance"] = (
                    target_change_rms / flux_systematic_rms
                )

            target_y, target_x = np.nonzero(diffuse_mask)
            before_weights = np.clip(before_signal, 0.0, None)
            after_weights = np.clip(after_signal, 0.0, None)
            before_weight_sum = float(np.sum(before_weights))
            after_weight_sum = float(np.sum(after_weights))
            if before_weight_sum > 1e-12 and after_weight_sum > 1e-12:
                before_cx = float(np.dot(target_x, before_weights) / before_weight_sum)
                before_cy = float(np.dot(target_y, before_weights) / before_weight_sum)
                after_cx = float(np.dot(target_x, after_weights) / after_weight_sum)
                after_cy = float(np.dot(target_y, after_weights) / after_weight_sum)
                centroid_shift = float(
                    np.hypot(after_cx - before_cx, after_cy - before_cy)
                )
                result["target_centroid_shift_pixels"] = centroid_shift
                result["target_centroid_shift_fraction"] = centroid_shift / max(
                    float(np.hypot(*before_gray.shape)),
                    1.0,
                )
        else:
            result["notes"].append("nebula retention skipped: diffuse mask too small")

        result["available"] = (
            result["star_retention_ratio"] is not None
            or result["target_flux_retention_ratio"] is not None
            or result["target_morphology_correlation"] is not None
        )
    except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
        result["notes"].append(f"preservation metrics failed: {exc}")
    return result


def _box_blur_gray(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected gray image, got shape={arr.shape}")
    h, w = arr.shape
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="reflect")
    acc = np.zeros_like(arr, dtype=np.float32)
    for y in range(3):
        for x in range(3):
            acc += padded[y:y + h, x:x + w]
    return acc / 9.0


def measure_quality_metrics(image: np.ndarray) -> QualityMetrics:
    """Measure conservative quality metrics for stage-level AI gates."""
    defaults = QualityMetrics()
    metrics = QualityMetrics()
    try:
        rgb = _to_rgb_float_image(image)
        r, g, b = rgb[0], rgb[1], rgb[2]
        gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        image_area = max(1, int(gray.size))
        eps = 1e-6

        bg_threshold = float(np.quantile(gray, 0.22))
        bg_mask = gray <= bg_threshold
        if int(np.count_nonzero(bg_mask)) < 64:
            bg_mask = gray <= float(np.quantile(gray, 0.30))
        bg_values = gray[bg_mask] if np.any(bg_mask) else gray.reshape(-1)
        metrics.bg_median = float(np.median(bg_values))

        metrics.black_pixel_ratio = float(np.mean(gray <= 0.010))
        metrics.highlight_clip_ratio = float(
            np.mean((gray >= 0.985) | (np.max(rgb, axis=0) >= 0.995))
        )

        star_threshold = max(
            float(np.quantile(gray, 0.985)),
            metrics.bg_median + max(4.0 * float(np.std(bg_values)), 0.05),
        )
        star_mask = gray > star_threshold
        max_star_area = max(4, int(image_area * 0.0008))
        star_areas = _component_areas(
            star_mask,
            min_area=1,
            max_area=max_star_area,
            max_components=18000,
        )
        metrics.star_density = len(star_areas) / float(image_area)
        if star_areas:
            median_area = float(np.median(star_areas))
            metrics.median_star_size = 2.0 * np.sqrt(median_area / np.pi)
            metrics.star_coverage_ratio = sum(star_areas) / float(image_area)
            total_signal = float(np.sum(np.clip(gray - metrics.bg_median, 0.0, None)))
            star_signal = float(np.sum(np.clip(gray[star_mask] - metrics.bg_median, 0.0, None)))
            metrics.star_energy_ratio = star_signal / max(total_signal, eps)

        maxc = np.max(rgb, axis=0)
        minc = np.min(rgb, axis=0)
        saturation = (maxc - minc) / np.maximum(maxc, eps)
        signal_mask = gray > max(float(np.quantile(gray, 0.50)), metrics.bg_median + 0.02)
        sat_values = saturation[signal_mask] if np.any(signal_mask) else saturation.reshape(-1)
        metrics.saturation_median = float(np.median(sat_values))
        metrics.saturation_p95 = float(np.quantile(sat_values, 0.95))

        blurred = _box_blur_gray(gray)
        signal_weight = signal_mask.astype(np.float32)
        if float(np.sum(signal_weight)) > 0:
            metrics.microcontrast = float(
                np.sum(np.abs(gray - blurred) * signal_weight)
                / max(float(np.sum(signal_weight)), eps)
            )
        else:
            metrics.microcontrast = float(np.mean(np.abs(gray - blurred)))

        signal_values = signal_mask
        if np.any(signal_values):
            red_dom = float(np.median((r[signal_values] + eps) / (g[signal_values] + eps)))
            blue_dom = float(np.median((b[signal_values] + eps) / (g[signal_values] + eps)))
            metrics.blue_excess = max(0.0, blue_dom - max(1.08, red_dom + 0.12))
    except (TypeError, ValueError, IndexError, FloatingPointError):
        metrics = defaults

    metrics.bg_median = _clamp_float(metrics.bg_median, 0.0, 1.0)
    metrics.black_pixel_ratio = _clamp_float(metrics.black_pixel_ratio, 0.0, 1.0)
    metrics.highlight_clip_ratio = _clamp_float(metrics.highlight_clip_ratio, 0.0, 1.0)
    metrics.star_density = _clamp_float(metrics.star_density, 0.0, 0.2)
    metrics.median_star_size = _clamp_float(metrics.median_star_size, 0.0, 64.0)
    metrics.star_coverage_ratio = _clamp_float(metrics.star_coverage_ratio, 0.0, 1.0)
    metrics.star_energy_ratio = _clamp_float(metrics.star_energy_ratio, 0.0, 1.0)
    metrics.saturation_median = _clamp_float(metrics.saturation_median, 0.0, 1.0)
    metrics.saturation_p95 = _clamp_float(metrics.saturation_p95, 0.0, 1.0)
    metrics.microcontrast = _clamp_float(metrics.microcontrast, 0.0, 1.0)
    metrics.blue_excess = _clamp_float(metrics.blue_excess, 0.0, 4.0)

    for key, value in asdict(metrics).items():
        if not np.isfinite(value):
            setattr(metrics, key, getattr(defaults, key))
    return metrics


def format_feature_summary(feat: ImageFeatures) -> str:
    return (
        "bg_median={:.4f}, bg_std={:.4f}, red_dom={:.3f}, blue_dom={:.3f}, "
        "star_density={:.5f}, median_star_size={:.3f}, object_area={:.3f}, "
        "diffuse={:.3f}, core={:.3f}, edge_black={:.3f}"
    ).format(
        feat.bg_median,
        feat.bg_std,
        feat.red_dominance,
        feat.blue_dominance,
        feat.star_density,
        feat.median_star_size,
        feat.object_area_ratio,
        feat.diffuse_ratio,
        feat.core_brightness_ratio,
        feat.edge_black_ratio,
    )
