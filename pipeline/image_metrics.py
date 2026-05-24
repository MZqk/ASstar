"""Image feature and quality metrics for the Seestar pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

import numpy as np

from models import ImageFeatures, QualityMetrics


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


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

        object_threshold = max(
            float(np.quantile(gray, 0.70)),
            feat.bg_median + max(1.8 * feat.bg_std, 0.02)
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
        global_dark_threshold = max(0.0015, min(0.010, feat.bg_median * 0.45))
        feat.global_dark_ratio = float(np.mean(gray <= global_dark_threshold))
        edge_black_threshold = global_dark_threshold
        if edge_values.size:
            feat.edge_black_ratio = float(np.mean(edge_values <= edge_black_threshold))
    except Exception:
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
    except Exception:
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
