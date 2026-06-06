"""Image feature analysis helpers for adaptive deep-sky processing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from save_utils import write_png_rgb16
except Exception:  # pragma: no cover - import fallback for unusual loaders
    write_png_rgb16 = None  # type: ignore[assignment]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return lower
    if not np.isfinite(numeric):
        return lower
    return max(lower, min(upper, numeric))


def _to_rgb_float(image: Any, max_side: int = 1024) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            if arr.shape[0] == 1:
                arr = np.broadcast_to(arr, (3, arr.shape[1], arr.shape[2])).copy()
        elif arr.shape[-1] in (1, 3):
            arr = np.moveaxis(arr, -1, 0)
            if arr.shape[0] == 1:
                arr = np.broadcast_to(arr, (3, arr.shape[1], arr.shape[2])).copy()
        else:
            arr = arr[:3]
    else:
        raise ValueError(f"unsupported image shape: {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        raise ValueError("empty image")
    max_val = float(np.max(arr))
    min_val = float(np.min(arr))
    if max_val > 1.5:
        scale = 65535.0 if max_val > 4096.0 else max_val
        arr = arr / max(scale, 1.0)
    elif min_val < 0.0:
        arr = arr - min_val
        arr = arr / max(float(np.max(arr)), 1e-6)
    arr = np.clip(arr, 0.0, 1.0)

    h, w = arr.shape[1], arr.shape[2]
    largest = max(h, w)
    if largest > max_side:
        step = int(np.ceil(largest / max_side))
        arr = arr[:, ::step, ::step]
    return arr.astype(np.float32, copy=False)


def _box_blur(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    radius = max(1, int(radius))
    padded = np.pad(gray, ((radius, radius), (radius, radius)), mode="reflect")
    out = np.zeros_like(gray, dtype=np.float32)
    count = 0
    for y in range(radius * 2 + 1):
        for x in range(radius * 2 + 1):
            out += padded[y:y + gray.shape[0], x:x + gray.shape[1]]
            count += 1
    return out / max(count, 1)


def _edge_mask(shape: tuple[int, int], width_ratio: float = 0.08) -> np.ndarray:
    h, w = shape
    band = max(2, int(min(h, w) * width_ratio))
    mask = np.zeros((h, w), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return mask


def _dark_clip_threshold(bg_median: float, bg_mad: float) -> float:
    """Return a threshold for true clipped pixels, not merely low linear sky."""
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


@dataclass
class AdaptiveImageFeatures:
    bg_median: float = 0.0
    bg_std: float = 0.0
    bg_mad: float = 0.0
    edge_black_ratio: float = 0.0
    gradient_score: float = 0.0
    dirty_background_score: float = 0.0
    chroma_noise_score: float = 0.0
    object_area_ratio: float = 0.0
    bright_core_score: float = 0.0
    core_peak_ratio: float = 0.0
    core_clip_ratio: float = 0.0
    nebulosity_area_ratio: float = 0.0
    faint_structure_score: float = 0.0
    symmetry_score: float = 0.0
    elongation_score: float = 0.0
    compactness_score: float = 0.0
    star_count: int = 0
    star_density: float = 0.0
    bright_star_count: int = 0
    star_bloat_score: float = 0.0
    halo_risk_score: float = 0.0
    dense_star_field_score: float = 0.0
    red_dominance: float = 1.0
    blue_dominance: float = 1.0
    green_cast: float = 1.0
    color_balance_score: float = 1.0
    blue_excess_score: float = 0.0
    low_snr_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_image(image: Any, *, max_side: int = 1024) -> AdaptiveImageFeatures:
    rgb = _to_rgb_float(image, max_side=max_side)
    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)

    bg_cut = float(np.quantile(gray, 0.35))
    bg_pixels = gray[gray <= bg_cut]
    if bg_pixels.size < 16:
        bg_pixels = gray.reshape(-1)
    bg_median = float(np.median(bg_pixels))
    bg_std = float(np.std(bg_pixels))
    bg_mad = float(np.median(np.abs(bg_pixels - bg_median)))
    edge = _edge_mask(gray.shape)
    edge_values = gray[edge]
    edge_black_ratio = float(np.mean(edge_values <= _dark_clip_threshold(bg_median, bg_mad)))

    top = float(np.median(gray[: max(1, gray.shape[0] // 8), :]))
    bottom = float(np.median(gray[-max(1, gray.shape[0] // 8):, :]))
    left = float(np.median(gray[:, : max(1, gray.shape[1] // 8)]))
    right = float(np.median(gray[:, -max(1, gray.shape[1] // 8):]))
    gradient_score = _clamp((max(top, bottom, left, right) - min(top, bottom, left, right)) / max(bg_std * 6.0, 0.01))

    smooth = _box_blur(gray, radius=2)
    texture = np.abs(gray - smooth)
    chroma = np.std(np.stack([r - gray, g - gray, b - gray], axis=0), axis=0)
    chroma_noise_score = _clamp(float(np.median(chroma[gray <= bg_cut])) / max(bg_std * 2.0, 0.01))
    dirty_background_score = _clamp(
        0.40 * _clamp(bg_std / max(bg_median, 0.015))
        + 0.35 * gradient_score
        + 0.25 * chroma_noise_score
    )

    object_floor = _low_linear_signal_floor(
        bg_median,
        bg_mad,
        default_floor=0.015,
        mad_multiplier=6.0,
        median_multiplier=0.08,
        minimum=0.00015,
    )
    object_threshold = max(
        float(np.quantile(gray, 0.72)),
        bg_median + max(bg_std * 2.0, object_floor),
    )
    object_mask = gray > object_threshold
    object_area_ratio = float(np.mean(object_mask))
    bright_threshold = max(float(np.quantile(gray, 0.985)), bg_median + max(bg_std * 6.0, 0.08))
    bright_mask = gray > bright_threshold
    bright_area = float(np.mean(bright_mask))
    core_peak_ratio = _clamp((float(np.max(gray)) - bg_median) / max(1.0 - bg_median, 1e-4))
    core_clip_ratio = float(np.mean(gray >= 0.985))
    bright_core_score = _clamp(core_peak_ratio * 0.7 + min(bright_area * 30.0, 1.0) * 0.3)

    faint_floor = _low_linear_signal_floor(
        bg_median,
        bg_mad,
        default_floor=0.008,
        mad_multiplier=3.0,
        median_multiplier=0.05,
        minimum=0.00008,
    )
    faint_threshold = max(
        float(np.quantile(gray, 0.58)),
        bg_median + max(bg_std * 1.1, faint_floor),
    )
    faint_mask = (gray > faint_threshold) & (~bright_mask)
    nebulosity_area_ratio = float(np.mean(faint_mask))
    faint_structure_score = _clamp(nebulosity_area_ratio * 2.2 + float(np.median(texture[faint_mask])) / max(bg_std * 3.0, 0.01) if np.any(faint_mask) else 0.0)

    ys, xs = np.nonzero(object_mask)
    if xs.size:
        x_span = max(float(xs.max() - xs.min() + 1), 1.0)
        y_span = max(float(ys.max() - ys.min() + 1), 1.0)
        elongation_score = _clamp(abs(x_span - y_span) / max(x_span, y_span))
        compactness_score = _clamp(object_area_ratio / max((x_span * y_span) / gray.size, 1e-6))
        left_mass = float(np.sum(object_mask[:, : gray.shape[1] // 2]))
        right_mass = float(np.sum(object_mask[:, gray.shape[1] // 2:]))
        symmetry_score = 1.0 - _clamp(abs(left_mass - right_mass) / max(left_mass + right_mass, 1.0))
    else:
        elongation_score = 0.0
        compactness_score = 0.0
        symmetry_score = 0.0

    local = gray - _box_blur(gray, radius=1)
    star_threshold = max(float(np.quantile(local, 0.992)), float(np.std(local)) * 3.0)
    star_mask = local > star_threshold
    star_count = int(np.count_nonzero(star_mask))
    star_density = float(star_count / max(gray.size, 1))
    bright_star_count = int(np.count_nonzero(bright_mask))
    halo_risk_score = _clamp(float(np.mean(_box_blur(bright_mask.astype(np.float32), radius=3) > 0.010)) * 12.0)
    star_bloat_score = _clamp(float(np.mean(_box_blur(star_mask.astype(np.float32), radius=1) > 0.15)) * 9.0)
    dense_star_field_score = _clamp(star_density / 0.018)

    eps = 1e-6
    signal = gray > faint_threshold
    if int(np.count_nonzero(signal)) < 32:
        signal = np.ones_like(gray, dtype=bool)
    r_med = float(np.median(r[signal])) + eps
    g_med = float(np.median(g[signal])) + eps
    b_med = float(np.median(b[signal])) + eps
    red_dominance = r_med / g_med
    blue_dominance = b_med / g_med
    green_cast = g_med / max((r_med + b_med) * 0.5, eps)
    color_balance_score = 1.0 - _clamp((abs(red_dominance - 1.0) + abs(blue_dominance - 1.0) + abs(green_cast - 1.0)) / 2.5)
    blue_excess_score = _clamp(blue_dominance - max(1.10, red_dominance + 0.10), 0.0, 1.0)
    low_snr_score = _clamp(bg_std / max(object_threshold - bg_median, 0.01))

    return AdaptiveImageFeatures(
        bg_median=bg_median,
        bg_std=bg_std,
        bg_mad=bg_mad,
        edge_black_ratio=edge_black_ratio,
        gradient_score=gradient_score,
        dirty_background_score=dirty_background_score,
        chroma_noise_score=chroma_noise_score,
        object_area_ratio=object_area_ratio,
        bright_core_score=bright_core_score,
        core_peak_ratio=core_peak_ratio,
        core_clip_ratio=core_clip_ratio,
        nebulosity_area_ratio=nebulosity_area_ratio,
        faint_structure_score=faint_structure_score,
        symmetry_score=symmetry_score,
        elongation_score=elongation_score,
        compactness_score=compactness_score,
        star_count=star_count,
        star_density=star_density,
        bright_star_count=bright_star_count,
        star_bloat_score=star_bloat_score,
        halo_risk_score=halo_risk_score,
        dense_star_field_score=dense_star_field_score,
        red_dominance=red_dominance,
        blue_dominance=blue_dominance,
        green_cast=green_cast,
        color_balance_score=color_balance_score,
        blue_excess_score=blue_excess_score,
        low_snr_score=low_snr_score,
    )


def feature_flags(features: AdaptiveImageFeatures) -> Dict[str, bool]:
    return {
        "bright_core": features.bright_core_score > 0.45 or features.core_peak_ratio > 0.70,
        "large_nebulosity": features.nebulosity_area_ratio > 0.18,
        "faint_outer_cloud": features.faint_structure_score > 0.30,
        "dense_star_field": features.dense_star_field_score > 0.35,
        "reflection_blue": features.blue_dominance > 1.12,
        "emission_red": features.red_dominance > 1.08,
        "small_galaxy": features.object_area_ratio < 0.12 and features.elongation_score > 0.20,
        "star_cluster_dominant": features.dense_star_field_score > 0.55 and features.nebulosity_area_ratio < 0.20,
    }


def risk_levels(features: AdaptiveImageFeatures) -> Dict[str, str]:
    def level(value: float, medium: float, high: float) -> str:
        if value >= high:
            return "high"
        if value >= medium:
            return "medium"
        return "low"

    return {
        "core_blowout": level(max(features.bright_core_score, features.core_clip_ratio * 20.0), 0.40, 0.70),
        "dirty_background": level(features.dirty_background_score, 0.28, 0.45),
        "blue_excess": level(features.blue_excess_score, 0.08, 0.18),
        "star_removal_residue": level(max(features.halo_risk_score, features.star_bloat_score), 0.35, 0.62),
        "overstretch": level(max(features.low_snr_score, features.dirty_background_score), 0.35, 0.55),
        "halo_risk": level(features.halo_risk_score, 0.30, 0.60),
    }


def write_safe_preview(image: Any, path: Path) -> bool:
    if write_png_rgb16 is None:
        return False
    try:
        rgb = _to_rgb_float(image, max_side=1600)
        gray = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
        lo = float(np.quantile(gray, 0.01))
        hi = float(np.quantile(gray, 0.995))
        stretched = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        stretched = np.sqrt(stretched)
        # FITS pixel data is bottom-up; flip vertically for correct PNG orientation.
        stretched = np.flip(stretched, axis=1)
        write_png_rgb16(path, stretched)
        return True
    except Exception:
        return False
