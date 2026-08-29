# Copyright (C) 2025 Carlo Mollicone - AstroBOH
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic Stage 8 false-color palettes for confirmed Ha/OIII OSC data.

The channel formulas and HSO/SHO/OSH/OHS/HOS/HOO permutations are adapted
from Carlo Mollicone's ``Hubble_Palette_from_Dual-Band_OSC.py`` (AstroBOH),
GPL-3.0-or-later.  This module intentionally contains no GUI, Siril commands,
runtime dependency installation, custom formulas, or temporary-file handling.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


DUALBAND_PALETTE_SCHEMA = "starun.stage8-dualband-palette.v2"
DUALBAND_PALETTE_SOURCE = {
    "script": "Hubble_Palette_from_Dual-Band_OSC.py",
    "upstream_version": "2.0.4",
    "formula_mode": "regular",
    "preset": "Classic",
    "author": "Carlo Mollicone (AstroBOH)",
    "license": "GPL-3.0-or-later",
}

PALETTE_CHANNELS: Dict[str, tuple[str, str, str]] = {
    "HSO": ("H", "S", "O"),
    "SHO": ("S", "H", "O"),
    "OSH": ("O", "S", "H"),
    "OHS": ("O", "H", "S"),
    "HOS": ("H", "O", "S"),
    "HOO": ("H", "O", "O"),
}

_FILAMENT_TARGET_NAMES = (
    "veil",
    "cygnus loop",
    "ngc 6960",
    "ngc 6992",
)
_FILAMENT_TARGET_TYPES = frozenset(
    {
        "supernova_remnant",
        "filamentary_supernova_remnant",
        "filamentary_emission_nebula",
    }
)
_OIII_COMPACT_TARGET_TYPES = frozenset(
    {
        "planetary_nebula",
        "planetary",
        "compact_oiii_nebula",
        "oiii_dominant_nebula",
    }
)
_DARK_HA_TARGET_TYPES = frozenset(
    {
        "dark_nebula_low_contrast",
        "dark_nebula",
        "ha_backlit_dark_nebula",
    }
)
_BRIGHT_CORE_TARGET_TYPES = frozenset(
    {
        "bright_emission_reflection_nebula",
        "bright_emission_nebula",
    }
)
_HA_WIDEFIELD_TARGET_TYPES = frozenset(
    {
        "emission_nebula_widefield",
        "emission_nebula",
        "widefield_emission_nebula",
    }
)


def _finite_clamp(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    if not np.isfinite(numeric):
        numeric = float(default)
    return max(float(lower), min(float(upper), numeric))


def select_primary_target_palette(
    primary_target: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select exactly one palette from the frozen primary target.

    This is deliberately a first-choice lookup rather than candidate ranking.
    Unknown confirmed Ha/OIII targets use HOO, the least synthetic mapping.
    """
    primary = primary_target if isinstance(primary_target, Mapping) else {}
    target_name = str(primary.get("name") or "").strip()
    target_type = str(primary.get("type") or "generic_low_snr_safe").strip().lower()
    confidence = _finite_clamp(
        primary.get("confidence", 0.0) or 0.0,
        0.0,
        1.0,
        0.0,
    )
    normalized_name = target_name.casefold()

    if target_type in _FILAMENT_TARGET_TYPES or any(
        token in normalized_name for token in _FILAMENT_TARGET_NAMES
    ):
        palette = "OSH"
        category = "filamentary_ha_oiii_separation"
        reason = "high-confidence filamentary target uses OSH first choice"
    elif target_type in _OIII_COMPACT_TARGET_TYPES:
        palette = "OHS"
        category = "compact_oiii_dominant"
        reason = "compact OIII-dominant target uses OHS first choice"
    elif target_type in _DARK_HA_TARGET_TYPES:
        palette = "HSO"
        category = "dark_nebula_ha_backdrop"
        reason = "dark nebula or Ha backdrop uses HSO first choice"
    elif target_type in _BRIGHT_CORE_TARGET_TYPES:
        palette = "HOS"
        category = "bright_core_mixed_emission"
        reason = "bright-core mixed emission target uses HOS first choice"
    elif target_type in _HA_WIDEFIELD_TARGET_TYPES:
        palette = "SHO"
        category = "ha_dominant_widefield_emission"
        reason = "Ha-dominant widefield emission target uses SHO first choice"
    else:
        palette = "HOO"
        category = "confirmed_ha_oiii_unknown_target"
        reason = "unknown confirmed Ha/OIII target uses HOO safe first choice"

    return {
        "palette": palette,
        "category": category,
        "reason": reason,
        "selection_mode": "frozen_primary_target_first_choice",
        "selected_rank": 1,
        "candidate_count": 1,
        "target": {
            "name": target_name or None,
            "type": target_type,
            "confidence": confidence,
            "method": str(primary.get("method") or "unknown"),
            "frozen": bool(primary.get("frozen", False)),
        },
    }


def resolve_palette_selection(
    primary_target: Optional[Mapping[str, Any]],
    requested_palette: Any = "auto",
) -> Dict[str, Any]:
    """Resolve one automatic or explicit palette from the frozen target.

    The returned mapping is suitable for freezing in ``processing-plan.json``.
    Explicit requests override only the artistic palette permutation; they do
    not alter the Stage 4 Ha/OIII channel contract.
    """
    automatic = select_primary_target_palette(primary_target)
    requested = str(requested_palette or "auto").strip()
    if requested.casefold() == "auto":
        return {
            **automatic,
            "requested_palette": "auto",
            "automatic_palette": automatic["palette"],
            "selection_mode": "automatic_target_mapping",
            "manual_override": False,
        }

    palette = requested.upper()
    if palette not in PALETTE_CHANNELS:
        raise ValueError(f"unsupported dual-band palette: {requested!r}")
    return {
        **automatic,
        "palette": palette,
        "requested_palette": palette,
        "automatic_palette": automatic["palette"],
        "automatic_reason": automatic["reason"],
        "reason": f"user explicitly selected {palette}",
        "selection_mode": "explicit_user_palette",
        "manual_override": True,
    }


def stage8_palette_eligibility(
    *,
    enabled: bool,
    channel_semantics: str,
    mapping: Mapping[str, Any],
    mapping_confidence_min: float,
    processing_policy: str,
    stage8_quality: str,
    stage7_accepted: bool,
    external_override: bool,
) -> Dict[str, Any]:
    """Return an auditable fail-closed decision for the Stage 8 palette branch."""
    issues = []
    if not bool(enabled):
        issues.append("optional_color_transform_disabled")
    if str(channel_semantics or "") != "narrowband_composite":
        issues.append("channel_semantics_not_narrowband_composite")
    if str(mapping.get("schema") or "") != "starun.narrowband-channel-mapping.v1":
        issues.append("channel_mapping_schema_invalid")
    if str(mapping.get("mapping") or "") != "osc_hoo_rgb":
        issues.append("channel_mapping_kind_invalid")
    if str(mapping.get("ha_channel") or "") != "R":
        issues.append("ha_channel_not_red")
    oiii_channels = mapping.get("oiii_channels")
    if not isinstance(oiii_channels, (list, tuple)) or list(oiii_channels) != [
        "G",
        "B",
    ]:
        issues.append("oiii_channels_not_green_blue")
    confidence_min = _finite_clamp(
        mapping_confidence_min,
        0.70,
        0.99,
        0.85,
    )
    mapping_confidence = _finite_clamp(
        mapping.get("confidence", 0.0) or 0.0,
        0.0,
        1.0,
        0.0,
    )
    if mapping_confidence < confidence_min:
        issues.append("ha_oiii_mapping_unconfirmed")
    if str(processing_policy or "").strip().lower() != "full":
        issues.append("stage8_policy_not_full")
    if str(stage8_quality or "").strip().lower() != "ok":
        issues.append("stage8_structural_quality_not_ok")
    if not bool(stage7_accepted):
        issues.append("stage7_starless_not_accepted")
    if bool(external_override):
        issues.append("external_starless_mapping_provenance_unverified")
    return {
        "eligible": not issues,
        "issues": issues,
        "mapping_confidence": mapping_confidence,
        "mapping_confidence_min": confidence_min,
    }


def _as_rgb_float(image: Any) -> tuple[np.ndarray, np.ndarray, str]:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("dual-band palette requires a 3-channel image")
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
        raise ValueError("dual-band palette input contains nonfinite pixels")
    if float(np.min(rgb)) < -0.05 or float(np.max(rgb)) > 1.05:
        raise ValueError("dual-band palette expects normalized RGB pixels")
    return source, np.clip(rgb, 0.0, 1.0), layout


def _restore(source: np.ndarray, rgb: np.ndarray, layout: str) -> np.ndarray:
    restored = rgb if layout == "chw" else np.transpose(rgb, (1, 2, 0))
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        return np.clip(restored * maximum, 0.0, maximum).astype(source.dtype)
    return restored.astype(np.float32, copy=False)


def derive_classic_dualband_channels(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Derive H, O, and synthetic S using the fixed regular/Classic preset."""
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected RGB CHW input, got shape={image.shape}")
    red, green, blue = image
    hydrogen = np.clip(red, 0.0, 1.0)
    oxygen = np.clip((green + blue) * 0.50, 0.0, 1.0)
    synthetic_sii = np.clip((hydrogen + oxygen) * 0.50, 0.0, 1.0)
    return {"H": hydrogen, "O": oxygen, "S": synthetic_sii}


def compose_palette(
    channels: Mapping[str, np.ndarray],
    palette: str,
) -> np.ndarray:
    palette_id = str(palette or "").strip().upper()
    mapping = PALETTE_CHANNELS.get(palette_id)
    if mapping is None:
        raise ValueError(f"unsupported dual-band palette: {palette}")
    return np.stack([np.asarray(channels[name], dtype=np.float32) for name in mapping])


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ).astype(np.float32)


def _saturation_proxy(rgb: np.ndarray) -> np.ndarray:
    maximum = np.max(rgb, axis=0)
    minimum = np.min(rgb, axis=0)
    return (maximum - minimum) / np.maximum(maximum, 1e-6)


def _luminance_preserving_gamut_map(
    mapped: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    target_luma = _luminance(base)
    mapped_luma = _luminance(mapped)
    valid = mapped_luma > 1e-6
    scaled = base.copy()
    scaled[:, valid] = (
        mapped[:, valid]
        * (target_luma[valid] / mapped_luma[valid])[None, :]
    )
    chroma = scaled - target_luma[None, :, :]
    gamut_scale = np.ones_like(target_luma, dtype=np.float32)
    for channel in range(3):
        component = chroma[channel]
        positive = component > 1e-7
        negative = component < -1e-7
        channel_scale = np.ones_like(target_luma, dtype=np.float32)
        channel_scale[positive] = (
            (1.0 - target_luma[positive]) / component[positive]
        )
        channel_scale[negative] = (
            target_luma[negative] / -component[negative]
        )
        gamut_scale = np.minimum(gamut_scale, channel_scale)
    gamut_scale = np.clip(gamut_scale * 0.995, 0.0, 1.0)
    result = target_luma[None, :, :] + chroma * gamut_scale[None, :, :]
    result[:, ~valid] = base[:, ~valid]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _luminance_preserving_chroma_scale(
    rgb: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale palette chroma without changing luminance or leaving gamut."""
    image = np.asarray(rgb, dtype=np.float32)
    target_luma = _luminance(image)
    chroma = image - target_luma[None, :, :]
    requested = _finite_clamp(scale, 1.0, 1.40, 1.0)
    gamut_scale = np.full_like(target_luma, requested, dtype=np.float32)
    for channel in range(3):
        component = chroma[channel]
        positive = component > 1e-7
        negative = component < -1e-7
        channel_scale = np.full_like(target_luma, requested, dtype=np.float32)
        channel_scale[positive] = np.minimum(
            requested,
            (1.0 - target_luma[positive]) / component[positive],
        )
        channel_scale[negative] = np.minimum(
            requested,
            target_luma[negative] / -component[negative],
        )
        gamut_scale = np.minimum(gamut_scale, channel_scale)
    gamut_scale = np.clip(gamut_scale * 0.995, 0.0, requested)
    result = target_luma[None, :, :] + chroma * gamut_scale[None, :, :]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def evaluate_palette_quality_metrics(
    metrics: Mapping[str, Any],
    *,
    luma_drift_p95_max: float,
    clip_growth_max: float,
    warning_tolerance: float = 0.50,
) -> Dict[str, Any]:
    """Classify palette quality with a warning-only 50% tolerance band.

    Luminance drift and clip growth may exceed their nominal limits by the
    configured fraction without rejecting the artistic candidate. Scope
    invariants remain hard failures: a palette must not alter the protected
    background or pixels outside the subject mask.
    """
    luma_limit = _finite_clamp(luma_drift_p95_max, 0.001, 0.03, 0.005)
    clip_limit = _finite_clamp(clip_growth_max, 0.0, 0.02, 0.002)
    tolerance = _finite_clamp(warning_tolerance, 0.0, 1.0, 0.50)
    limits = {
        "luminance_drift_p95_max": luma_limit,
        "luminance_drift_p95_hard_max": luma_limit * (1.0 + tolerance),
        "clip_growth_max": clip_limit,
        "clip_growth_hard_max": clip_limit * (1.0 + tolerance),
        "outside_subject_max_abs_change_max": 1e-6,
        "background_change_p95_max": 1e-6,
        "quality_warning_tolerance": tolerance,
    }
    issues: list[str] = []
    warnings: list[str] = []

    luma_drift = float(metrics.get("luminance_drift_p95", 0.0) or 0.0)
    clip_growth = float(metrics.get("clip_growth", 0.0) or 0.0)
    outside_change = float(
        metrics.get("outside_subject_max_abs_change", 0.0) or 0.0
    )
    background_change = float(metrics.get("background_change_p95", 0.0) or 0.0)
    if not all(
        np.isfinite(value)
        for value in (
            luma_drift,
            clip_growth,
            outside_change,
            background_change,
        )
    ):
        issues.append("nonfinite_quality_metrics")
    if luma_drift > limits["luminance_drift_p95_hard_max"]:
        issues.append("luminance_drift")
    elif luma_drift > limits["luminance_drift_p95_max"]:
        warnings.append("luminance_drift_within_warning_tolerance")
    if clip_growth > limits["clip_growth_hard_max"]:
        issues.append("clip_growth")
    elif clip_growth > limits["clip_growth_max"]:
        warnings.append("clip_growth_within_warning_tolerance")
    if outside_change > limits["outside_subject_max_abs_change_max"]:
        issues.append("outside_subject_change")
    if background_change > limits["background_change_p95_max"]:
        issues.append("background_change")
    return {
        "accepted": not issues,
        "status": (
            "rejected"
            if issues
            else "accepted_with_warning"
            if warnings
            else "accepted"
        ),
        "limits": limits,
        "issues": issues,
        "warnings": warnings,
    }


def build_dualband_palette_candidate(
    image: Any,
    *,
    palette: str,
    core_mask: np.ndarray,
    nebula_mask: np.ndarray,
    faint_nebula_mask: np.ndarray,
    background_mask: np.ndarray,
    star_mask: Optional[np.ndarray] = None,
    star_halo_guard_mask: Optional[np.ndarray] = None,
    strength: float = 0.85,
    luma_drift_p95_max: float = 0.005,
    clip_growth_max: float = 0.002,
    quality_warning_tolerance: float = 0.50,
    subject_chroma_separation_gain_min: float = 1.0e-4,
    subject_saturation_input_ratio_min: float = 0.50,
    subject_saturation_absolute_min: float = 0.08,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Build one masked, luminance-preserving palette candidate."""
    source, base, layout = _as_rgb_float(image)
    height, width = base.shape[1:]

    def checked_mask(name: str, value: np.ndarray) -> np.ndarray:
        mask = np.asarray(value, dtype=np.float32)
        if mask.shape != (height, width):
            raise ValueError(f"{name} shape mismatch: {mask.shape} != {(height, width)}")
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"{name} contains nonfinite values")
        return np.clip(mask, 0.0, 1.0)

    core = checked_mask("core_mask", core_mask)
    nebula = checked_mask("nebula_mask", nebula_mask)
    faint = checked_mask("faint_nebula_mask", faint_nebula_mask)
    background = checked_mask("background_mask", background_mask)
    stars = (
        checked_mask("star_mask", star_mask)
        if star_mask is not None
        else np.zeros((height, width), dtype=np.float32)
    )
    halo = (
        checked_mask("star_halo_guard_mask", star_halo_guard_mask)
        if star_halo_guard_mask is not None
        else np.zeros((height, width), dtype=np.float32)
    )
    protected = np.maximum.reduce((core, stars, halo))
    subject = np.maximum(nebula, faint)
    subject = np.clip(
        subject * (1.0 - background) * (1.0 - protected),
        0.0,
        1.0,
    )
    subject[background >= 0.80] = 0.0
    safe_strength = _finite_clamp(strength, 0.10, 1.0, 0.85)
    weight = subject * safe_strength
    weight[weight <= 0.01] = 0.0
    subject_support = weight > 0.0
    background_support = background >= 0.80
    if int(np.count_nonzero(subject_support)) < 64:
        raise ValueError("dual-band palette subject mask has insufficient support")

    # These are delivery minima, not tuning knobs.  Configuration may make a
    # palette stricter, but it cannot relax the physical chroma contract.
    effective_separation_gain_min = max(
        float(subject_chroma_separation_gain_min),
        1.0e-4,
    )
    effective_saturation_ratio_min = max(
        _finite_clamp(
            subject_saturation_input_ratio_min,
            0.0,
            1.0,
            0.50,
        ),
        0.50,
    )
    effective_saturation_absolute_min = max(
        _finite_clamp(
            subject_saturation_absolute_min,
            0.0,
            1.0,
            0.08,
        ),
        0.08,
    )

    channels = derive_classic_dualband_channels(base)
    raw_palette = compose_palette(channels, palette)
    safe_palette = _luminance_preserving_gamut_map(raw_palette, base)
    base_luma = _luminance(base)
    base_saturation = _saturation_proxy(base)
    subject_saturation_before = base_saturation[subject_support]
    if np.count_nonzero(background_support) >= 64:
        background_saturation_before = base_saturation[background_support]
        background_p50_before = float(
            np.quantile(background_saturation_before, 0.50)
        )
    else:
        background_p50_before = 0.0
    subject_p50_before = float(
        np.quantile(subject_saturation_before, 0.50)
    )
    saturation_floor = min(
        effective_saturation_ratio_min * subject_p50_before,
        effective_saturation_absolute_min,
    )

    # A Classic channel permutation can reduce median chroma after its
    # luminance-preserving gamut map.  Close that loop only inside the frozen
    # subject support: evaluate a bounded ladder and keep the weakest palette
    # chroma scale that clears the unchanged delivery gates.  This is not a
    # global saturation operation; ``weight`` still makes every protected or
    # out-of-mask pixel exactly equal to the physical Stage 7 parent.
    chroma_closure_attempts: list[Dict[str, Any]] = []
    selected_chroma_scale = 1.40
    candidate = base.copy()
    for chroma_scale in (1.0, 1.12, 1.25, 1.40):
        scaled_palette = _luminance_preserving_chroma_scale(
            safe_palette,
            chroma_scale,
        )
        trial = base + (scaled_palette - base) * weight[None, :, :]
        trial = np.clip(trial, 0.0, 1.0).astype(np.float32)
        trial_saturation = _saturation_proxy(trial)
        trial_subject_p50 = float(
            np.quantile(trial_saturation[subject_support], 0.50)
        )
        trial_background_p50 = (
            float(np.quantile(trial_saturation[background_support], 0.50))
            if np.count_nonzero(background_support) >= 64
            else 0.0
        )
        trial_separation_gain = (
            (trial_subject_p50 - trial_background_p50)
            - (subject_p50_before - background_p50_before)
        )
        clears_chroma_contract = bool(
            trial_separation_gain > effective_separation_gain_min
            and trial_subject_p50 + 1e-12 >= saturation_floor
        )
        chroma_closure_attempts.append(
            {
                "scale": float(chroma_scale),
                "subject_saturation_p50_after": trial_subject_p50,
                "subject_background_chroma_separation_gain": (
                    trial_separation_gain
                ),
                "clears_chroma_contract": clears_chroma_contract,
            }
        )
        candidate = trial
        selected_chroma_scale = float(chroma_scale)
        if clears_chroma_contract:
            break

    candidate_luma = _luminance(candidate)
    luma_drift_p95 = float(np.quantile(np.abs(candidate_luma - base_luma), 0.95))
    before_clip = float(np.mean((base <= 0.0) | (base >= 1.0)))
    after_clip = float(np.mean((candidate <= 0.0) | (candidate >= 1.0)))
    delta = np.max(np.abs(candidate - base), axis=0)
    outside_support = ~subject_support
    outside_delta_max = (
        float(np.max(delta[outside_support])) if np.any(outside_support) else 0.0
    )
    background_delta_p95 = (
        float(np.quantile(delta[background_support], 0.95))
        if np.count_nonzero(background_support) >= 64
        else 0.0
    )
    subject_delta_p95 = float(np.quantile(delta[subject_support], 0.95))
    candidate_saturation_map = _saturation_proxy(candidate)
    subject_saturation = candidate_saturation_map[subject_support]
    if np.count_nonzero(background_support) >= 64:
        background_saturation_after = candidate_saturation_map[
            background_support
        ]
        background_p50_after = float(
            np.quantile(background_saturation_after, 0.50)
        )
    else:
        background_p50_after = 0.0
    subject_p50_after = float(np.quantile(subject_saturation, 0.50))
    separation_before = subject_p50_before - background_p50_before
    separation_after = subject_p50_after - background_p50_after
    separation_gain = separation_after - separation_before
    metrics = {
        "subject_mask_coverage": float(np.mean(subject_support)),
        "background_mask_coverage": float(np.mean(background_support)),
        "luminance_drift_p95": luma_drift_p95,
        "clip_growth": after_clip - before_clip,
        "outside_subject_max_abs_change": outside_delta_max,
        "background_change_p95": background_delta_p95,
        "subject_change_p95": subject_delta_p95,
        "subject_saturation_p95": float(np.quantile(subject_saturation, 0.95)),
        "subject_saturation_p50_before": subject_p50_before,
        "subject_saturation_p50_after": subject_p50_after,
        "subject_saturation_p50_gain": (
            subject_p50_after - subject_p50_before
        ),
        "background_saturation_p50_before": background_p50_before,
        "background_saturation_p50_after": background_p50_after,
        "subject_background_chroma_separation_before": separation_before,
        "subject_background_chroma_separation_after": separation_after,
        "subject_background_chroma_separation_gain": (
            separation_gain
        ),
        "subject_saturation_p50_floor": saturation_floor,
        "subject_saturation_p50_floor_passed": bool(
            subject_p50_after + 1e-12 >= saturation_floor
        ),
    }
    quality = evaluate_palette_quality_metrics(
        metrics,
        luma_drift_p95_max=luma_drift_p95_max,
        clip_growth_max=clip_growth_max,
        warning_tolerance=quality_warning_tolerance,
    )
    limits = dict(quality["limits"])
    issues = list(quality["issues"])
    warnings = list(quality["warnings"])
    if separation_gain <= effective_separation_gain_min:
        issues.append("subject_background_chroma_separation_gain_unmet")
    if subject_p50_after + 1e-12 < saturation_floor:
        issues.append("subject_saturation_floor_unmet")
    if not np.all(np.isfinite(candidate)):
        issues.append("nonfinite_candidate")

    palette_id = str(palette).strip().upper()
    accepted = not issues
    status = (
        "rejected"
        if issues
        else "accepted_with_warning"
        if warnings
        else "accepted"
    )
    report = {
        "schema": DUALBAND_PALETTE_SCHEMA,
        "algorithm_source": dict(DUALBAND_PALETTE_SOURCE),
        "status": status,
        "accepted": accepted,
        "role": "artistic_false_color",
        "palette": palette_id,
        "synthetic_sii": palette_id != "HOO",
        "formula": {
            "mode": "regular",
            "preset": "Classic",
            "H": "R",
            "O": "(G+B)*0.5",
            "S_proxy": "(H+O)*0.5",
        },
        "subject_chroma_closure": {
            "schema": "starun.stage8-palette-subject-chroma-closure.v1",
            "scope": "frozen_subject_mask_only",
            "selection": "weakest_passing_scale",
            "scale_max": 1.40,
            "selected_scale": selected_chroma_scale,
            "attempts": chroma_closure_attempts,
        },
        "mask_contract": {
            "subject_formula": "max(nebula,faint)*(1-background)*(1-max(core,star,halo))",
            "guard_sources": ["core_mask", "star_mask", "star_halo_guard_mask"],
            "protected_coverage": float(np.mean(protected > 0.0)),
        },
        "mapping": {
            "R": PALETTE_CHANNELS[palette_id][0],
            "G": PALETTE_CHANNELS[palette_id][1],
            "B": PALETTE_CHANNELS[palette_id][2],
        },
        "strength": safe_strength,
        "metrics": metrics,
        "limits": limits,
        "issues": issues,
        "warnings": warnings,
    }
    report["limits"].update(
        {
            "subject_background_chroma_separation_gain_min_exclusive": float(
                effective_separation_gain_min
            ),
            "subject_saturation_input_ratio_min": float(
                effective_saturation_ratio_min
            ),
            "subject_saturation_absolute_min": float(
                effective_saturation_absolute_min
            ),
        }
    )
    return _restore(source, candidate, layout), report


__all__ = [
    "DUALBAND_PALETTE_SCHEMA",
    "DUALBAND_PALETTE_SOURCE",
    "PALETTE_CHANNELS",
    "build_dualband_palette_candidate",
    "compose_palette",
    "derive_classic_dualband_channels",
    "evaluate_palette_quality_metrics",
    "resolve_palette_selection",
    "select_primary_target_palette",
    "stage8_palette_eligibility",
]
