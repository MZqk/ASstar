"""Deterministic Ha/OIII normalization for confirmed dual-narrowband RGB data."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional

import numpy as np


NARROWBAND_NORMALIZATION_SCHEMA = "seestar.narrowband-normalization.v1"

_HA_TOKENS = ("ha", "h-alpha", "halpha", "hα")
_OIII_TOKENS = ("oiii", "o-iii", "o3", "oⅲ")
_DUAL_BAND_TOKENS = (
    "dualband",
    "dual-band",
    "dual band",
    "duo-band",
    "l-extreme",
    "l-enhance",
    "l-ultimate",
    "hoo",
)


def _metadata_text(
    metadata: Optional[Mapping[str, Any]],
    filter_hint: str,
) -> str:
    values = [filter_hint]
    for key in ("FILTER", "FILTER1", "FILTER2", "INSFLNAM", "OBJECT"):
        value = (metadata or {}).get(key)
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        if value:
            values.append(str(value))
    return " ".join(values).casefold()


def classify_dual_narrowband_mapping(
    metadata: Optional[Mapping[str, Any]],
    *,
    filter_hint: str = "",
) -> Dict[str, Any]:
    text = _metadata_text(metadata, filter_hint)
    normalized = re.sub(r"\s+", " ", text).strip()
    has_ha = any(token in normalized for token in _HA_TOKENS)
    has_oiii = any(token in normalized for token in _OIII_TOKENS)
    has_dual = any(token in normalized for token in _DUAL_BAND_TOKENS)
    if has_ha and has_oiii:
        confidence = 0.97
        reason = "explicit_ha_oiii"
    elif has_dual and ("hoo" in normalized or "l-" in normalized):
        confidence = 0.90
        reason = "known_dualband_ha_oiii_hint"
    elif has_dual:
        confidence = 0.76
        reason = "generic_dualband_without_line_identity"
    else:
        confidence = 0.0
        reason = "ha_oiii_mapping_unconfirmed"
    return {
        "mapping": "osc_hoo_rgb" if confidence > 0.0 else "unknown",
        "ha_channel": "R" if confidence > 0.0 else None,
        "oiii_channels": ["G", "B"] if confidence > 0.0 else [],
        "confidence": confidence,
        "reason": reason,
        "evidence_text": normalized or None,
    }


def _as_rgb_float(image: Any) -> tuple[np.ndarray, np.ndarray, str]:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("dual-narrowband normalization requires a 3-channel image")
    if source.shape[0] == 3:
        rgb_source = source
        layout = "chw"
    elif source.shape[-1] == 3:
        rgb_source = np.transpose(source, (2, 0, 1))
        layout = "hwc"
    else:
        raise ValueError(f"expected RGB input, got shape={source.shape}")
    rgb = rgb_source.astype(np.float32, copy=True)
    if np.issubdtype(source.dtype, np.integer):
        rgb /= max(1.0, float(np.iinfo(source.dtype).max))
    if not np.all(np.isfinite(rgb)):
        raise ValueError("nonfinite input pixels")
    return source, rgb, layout


def _restore(source: np.ndarray, rgb: np.ndarray, layout: str) -> np.ndarray:
    output = rgb if layout == "chw" else np.transpose(rgb, (1, 2, 0))
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        return np.clip(output * maximum, 0.0, maximum).astype(source.dtype)
    return output.astype(np.float32, copy=False)


def _blur3(plane: np.ndarray) -> np.ndarray:
    padded = np.pad(plane, ((1, 1), (1, 1)), mode="reflect")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _mad_sigma(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size < 16:
        return 0.0
    median = float(np.median(data))
    return float(1.4826 * np.median(np.abs(data - median)))


def _normalized_chroma(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    samples = rgb[:, mask]
    total = np.maximum(np.sum(samples, axis=0), 1e-6)
    return samples / total[None, :]


def normalize_dual_narrowband_candidate(
    image: Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    filter_hint: str = "",
    mapping_confidence_min: float = 0.85,
    strength: float = 0.55,
    gain_limit: float = 1.08,
    line_ratio_drift_max: float = 0.12,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Return a guarded HOO normalization candidate without mutating input."""
    mapping = classify_dual_narrowband_mapping(
        metadata,
        filter_hint=filter_hint,
    )
    confidence_min = max(0.70, min(0.99, float(mapping_confidence_min)))
    if float(mapping["confidence"]) < confidence_min:
        raise ValueError(
            "dual-narrowband channel mapping confidence is insufficient"
        )
    source, rgb, layout = _as_rgb_float(image)
    original = rgb.copy()
    luma = (
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    ).astype(np.float32)
    finite_values = luma[np.isfinite(luma)]
    if finite_values.size < 256:
        raise ValueError("insufficient image pixels")
    q005, q40, q60, q98 = (
        float(value)
        for value in np.quantile(finite_values, (0.005, 0.40, 0.60, 0.98))
    )
    gradient = np.abs(luma - _blur3(luma))
    gradient_limit = float(np.quantile(gradient, 0.70))
    background = (
        (luma >= q005)
        & (luma <= q40)
        & (gradient <= gradient_limit)
    )
    signal = (luma >= q60) & (luma <= q98)
    if np.count_nonzero(background) < 128 or np.count_nonzero(signal) < 128:
        raise ValueError("insufficient background or signal samples")

    backgrounds = np.median(rgb[:, background], axis=1)
    spans = np.array(
        [
            max(
                float(np.quantile(rgb[channel, signal], 0.90))
                - float(backgrounds[channel]),
                1e-6,
            )
            for channel in range(3)
        ],
        dtype=np.float64,
    )
    safe_strength = max(0.10, min(0.85, float(strength)))
    safe_gain_limit = max(1.01, min(1.15, float(gain_limit)))
    oiii_span = math.sqrt(float(spans[1] * spans[2]))
    gb_balance = math.sqrt(float(spans[2] / spans[1]))
    ha_oiii_balance = math.sqrt(oiii_span / float(spans[0]))
    raw_gains = np.array(
        [
            ha_oiii_balance ** (safe_strength * 0.35),
            gb_balance ** safe_strength,
            (1.0 / gb_balance) ** safe_strength,
        ],
        dtype=np.float64,
    )
    gains = np.clip(
        raw_gains,
        1.0 / safe_gain_limit,
        safe_gain_limit,
    ).astype(np.float32)

    target_background = float(np.dot(backgrounds, [0.2126, 0.7152, 0.0722]))
    normalized = (
        (rgb - backgrounds[:, None, None]) * gains[:, None, None]
        + (
            backgrounds[:, None, None] * (1.0 - safe_strength * 0.65)
            + target_background * (safe_strength * 0.65)
        )
    )
    normalized_luma = (
        0.2126 * normalized[0]
        + 0.7152 * normalized[1]
        + 0.0722 * normalized[2]
    )
    normalized *= (
        luma / np.maximum(normalized_luma, 1e-6)
    )[None, :, :]

    star_detail = np.maximum(luma - _blur3(luma), 0.0)
    star_threshold = max(
        float(np.median(star_detail)) + 8.0 * _mad_sigma(
            star_detail[background]
        ),
        float(np.quantile(star_detail, 0.992)),
    )
    star_peak = float(np.quantile(star_detail, 0.9995))
    star_mask = np.clip(
        (star_detail - star_threshold)
        / max(star_peak - star_threshold, 1e-6),
        0.0,
        1.0,
    )
    star_mask = np.clip(_blur3(star_mask), 0.0, 1.0)
    normalized = (
        normalized * (1.0 - 0.90 * star_mask[None, :, :])
        + original * (0.90 * star_mask[None, :, :])
    )
    candidate = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    before_bg_chroma = _normalized_chroma(original, background)
    after_bg_chroma = _normalized_chroma(candidate, background)
    neutral = np.full((3, before_bg_chroma.shape[1]), 1.0 / 3.0)
    bg_delta_before = float(np.median(np.max(np.abs(before_bg_chroma - neutral), axis=0)))
    bg_delta_after = float(np.median(np.max(np.abs(after_bg_chroma - neutral), axis=0)))

    candidate_backgrounds = np.median(candidate[:, background], axis=1)
    candidate_spans = np.array(
        [
            max(
                float(np.quantile(candidate[channel, signal], 0.90))
                - float(candidate_backgrounds[channel]),
                1e-6,
            )
            for channel in range(3)
        ]
    )
    ratio_before = float(spans[0] / math.sqrt(spans[1] * spans[2]))
    ratio_after = float(
        candidate_spans[0]
        / math.sqrt(candidate_spans[1] * candidate_spans[2])
    )
    line_ratio_drift = abs(ratio_after / max(ratio_before, 1e-6) - 1.0)
    before_star_chroma = _normalized_chroma(original, star_mask >= 0.35)
    after_star_chroma = _normalized_chroma(candidate, star_mask >= 0.35)
    star_chroma_drift = (
        float(np.median(np.max(np.abs(after_star_chroma - before_star_chroma), axis=0)))
        if before_star_chroma.shape[1] >= 16
        else 0.0
    )
    before_clip = float(np.mean((original <= 0.0) | (original >= 1.0)))
    after_clip = float(np.mean((candidate <= 0.0) | (candidate >= 1.0)))
    candidate_luma = (
        0.2126 * candidate[0] + 0.7152 * candidate[1] + 0.0722 * candidate[2]
    )
    luma_drift = float(
        np.quantile(np.abs(candidate_luma - luma), 0.95)
    )
    metrics = {
        "background_color_delta_before": bg_delta_before,
        "background_color_delta_after": bg_delta_after,
        "background_color_improvement": bg_delta_before - bg_delta_after,
        "ha_oiii_ratio_before": ratio_before,
        "ha_oiii_ratio_after": ratio_after,
        "ha_oiii_ratio_drift": line_ratio_drift,
        "star_chroma_drift": star_chroma_drift,
        "luminance_drift_p95": luma_drift,
        "clip_growth": after_clip - before_clip,
        "star_mask_coverage": float(np.mean(star_mask > 0.05)),
    }
    limits = {
        "ha_oiii_ratio_drift_max": max(
            0.04,
            min(0.20, float(line_ratio_drift_max)),
        ),
        "star_chroma_drift_max": 0.10,
        "luminance_drift_p95_max": 0.015,
        "clip_growth_max": 0.001,
        "background_color_worsening_max": 0.005,
        "star_mask_coverage_max": 0.12,
    }
    issues: list[str] = []
    if not np.all(np.isfinite(candidate)):
        issues.append("nonfinite_output")
    if line_ratio_drift > limits["ha_oiii_ratio_drift_max"]:
        issues.append("ha_oiii_ratio_drift")
    if star_chroma_drift > limits["star_chroma_drift_max"]:
        issues.append("star_chroma_drift")
    if luma_drift > limits["luminance_drift_p95_max"]:
        issues.append("luminance_drift")
    if metrics["clip_growth"] > limits["clip_growth_max"]:
        issues.append("clip_growth")
    if (
        bg_delta_after - bg_delta_before
        > limits["background_color_worsening_max"]
    ):
        issues.append("background_color_worsened")
    if metrics["star_mask_coverage"] > limits["star_mask_coverage_max"]:
        issues.append("star_mask_coverage")
    changed = float(np.mean(np.max(np.abs(candidate - original), axis=0) > 1e-4))
    if changed <= 1e-4:
        issues.append("no_effect")
    accepted = not issues
    report = {
        "schema": NARROWBAND_NORMALIZATION_SCHEMA,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "mapping": mapping,
        "algorithm": "luminance_preserving_guarded_hoo_normalization",
        "strength": safe_strength,
        "gains": {
            "red_ha": float(gains[0]),
            "green_oiii": float(gains[1]),
            "blue_oiii": float(gains[2]),
        },
        "metrics": metrics,
        "limits": limits,
        "issues": issues,
        "changed_pixel_ratio": changed,
        "transaction": {
            "baseline": "stage4_pre_nbn.fit",
            "candidate": "stage4_nbn_candidate.fit",
        },
    }
    return _restore(source, candidate, layout), report


__all__ = [
    "classify_dual_narrowband_mapping",
    "normalize_dual_narrowband_candidate",
]
