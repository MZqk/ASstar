"""Target profiling for adaptive deep-sky processing."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from image_feature_analyzer import AdaptiveImageFeatures, feature_flags, risk_levels
from policy_selector import DEFAULT_POLICY_NAME, policy_for_profile


CATALOG_PATH = Path(__file__).resolve().parent / "configs" / "target_catalog" / "popular_dso.json"

BUILTIN_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "M42",
        "aliases": ["M 42", "Orion Nebula", "NGC 1976", "NGC1976"],
        "type": "bright_emission_reflection_nebula",
        "ra_deg": 83.822,
        "dec_deg": -5.391,
        "size_arcmin": [85, 60],
        "features": ["bright_core", "large_nebulosity", "reflection_blue", "emission_red", "dense_stars"],
        "default_policy": "bright_nebula_hdr_conservative",
    },
    {
        "name": "M31",
        "aliases": ["Andromeda Galaxy", "NGC 224"],
        "type": "large_galaxy",
        "ra_deg": 10.685,
        "dec_deg": 41.269,
        "size_arcmin": [190, 60],
        "features": ["bright_core", "large_halo", "elongated"],
        "default_policy": "large_galaxy_core_protect",
    },
    {
        "name": "M45",
        "aliases": ["Pleiades", "Seven Sisters"],
        "type": "reflection_nebula_cluster",
        "ra_deg": 56.75,
        "dec_deg": 24.117,
        "size_arcmin": [110, 110],
        "features": ["reflection_blue", "bright_stars", "halo_risk"],
        "default_policy": "reflection_nebula_halo_protect",
    },
    {
        "name": "Horsehead Nebula",
        "aliases": ["Barnard 33", "B33", "IC 434", "IC434", "Flame Nebula", "NGC 2024"],
        "type": "dark_nebula_low_contrast",
        "ra_deg": 85.245,
        "dec_deg": -2.459,
        "size_arcmin": [60, 30],
        "features": ["dark_nebula", "emission_red", "low_contrast"],
        "default_policy": "dark_nebula_low_contrast",
    },
]


TYPE_TO_POLICY = {
    "bright_emission_reflection_nebula": "bright_nebula_hdr_conservative",
    "large_galaxy": "large_galaxy_core_protect",
    "small_galaxy": "large_galaxy_core_protect",
    "reflection_nebula_cluster": "reflection_nebula_halo_protect",
    "emission_nebula_widefield": "emission_nebula_widefield",
    "dark_nebula_low_contrast": "dark_nebula_low_contrast",
    "globular_cluster": "globular_cluster_star_preserve",
    "open_cluster": "open_cluster_color_preserve",
    "widefield_milkyway": "generic_low_snr_safe",
    "generic_low_snr_safe": DEFAULT_POLICY_NAME,
}


def load_catalog(path: Path = CATALOG_PATH) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]
            known = {_norm(str(item.get("name", ""))) for item in items}
            for fallback in BUILTIN_CATALOG:
                if _norm(str(fallback.get("name", ""))) not in known:
                    items.append(fallback)
            return items
    except (OSError, UnicodeError, ValueError, TypeError):
        pass
    return list(BUILTIN_CATALOG)


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _metadata_text(metadata: Optional[Dict[str, Any]], context: str) -> str:
    parts = [context]
    if metadata:
        for key in ("OBJECT", "OBJCT", "TARGET", "target", "object", "AUTO_TARGET_TYPE"):
            value = metadata.get(key)
            if value:
                parts.append(str(value))
    return " ".join(parts)


def _catalog_name_match(catalog: Iterable[Dict[str, Any]], text: str) -> Optional[Tuple[Dict[str, Any], float]]:
    normalized = _norm(text)
    raw_lower = text.lower()
    if not normalized:
        return None
    best: Optional[Tuple[Dict[str, Any], float]] = None
    for item in catalog:
        names = [str(item.get("name", "")), *[str(alias) for alias in item.get("aliases", [])]]
        for name in names:
            candidate = _norm(name)
            if not candidate:
                continue
            score = 0.0
            name_lower = name.strip().lower()
            if re.fullmatch(r"m\d+", name_lower):
                # Messier IDs are short and often appear in folder names such as M42_test.
                if re.search(rf"(?<![a-z0-9]){re.escape(name_lower)}(?!\d)", raw_lower):
                    score = 0.93
            elif name_lower and name_lower in raw_lower:
                score = 0.90
            if candidate == normalized:
                score = max(score, 0.95)
            elif candidate in normalized:
                score = max(score, min(0.88, 0.55 + len(candidate) / max(len(normalized), 1)))
            if best is None or score > best[1]:
                best = (item, score)
    return best if best and best[1] > 0.0 else None


def _parse_angle(value: Any, *, is_ra: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if is_ra and 0.0 <= numeric <= 24.0:
            return numeric * 15.0
        return numeric
    text = str(value).strip().strip("'\"")
    if not text:
        return None
    try:
        numeric = float(text)
        if is_ra and 0.0 <= numeric <= 24.0:
            return numeric * 15.0
        return numeric
    except ValueError:
        pass
    parts = re.split(r"[:\s]+", text.replace("h", ":").replace("m", ":").replace("s", ""))
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return None
    sign = -1.0 if parts[0].startswith("-") else 1.0
    first = abs(float(parts[0]))
    second = float(parts[1])
    third = float(parts[2]) if len(parts) > 2 else 0.0
    degrees = first + second / 60.0 + third / 3600.0
    if is_ra:
        return degrees * 15.0
    return sign * degrees


def _metadata_coordinates(metadata: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not metadata:
        return None, None
    ra_value = None
    ra_key = ""
    for key in ("OBJCTRA", "OBJRA", "RA", "CRVAL1", "PLTSOLRA", "CENTER_RA"):
        if key in metadata:
            ra_value = metadata.get(key)
            ra_key = key
            break
    dec_value = None
    for key in ("OBJCTDEC", "OBJDEC", "DEC", "CRVAL2", "PLTSOLDEC", "CENTER_DEC"):
        if key in metadata:
            dec_value = metadata.get(key)
            break
    ra_is_hours = ra_key not in {"CRVAL1", "CENTER_RA"} if ra_value is not None else False
    return _parse_angle(ra_value, is_ra=ra_is_hours), _parse_angle(dec_value, is_ra=False)


def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    import math

    r1 = math.radians(ra1)
    d1 = math.radians(dec1)
    r2 = math.radians(ra2)
    d2 = math.radians(dec2)
    cos_d = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_d))))


def _catalog_coordinate_match(
    catalog: Iterable[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], float, float]]:
    ra_deg, dec_deg = _metadata_coordinates(metadata)
    if ra_deg is None or dec_deg is None:
        return None
    best: Optional[Tuple[Dict[str, Any], float, float]] = None
    for item in catalog:
        item_ra = item.get("ra_deg")
        item_dec = item.get("dec_deg")
        if item_ra is None or item_dec is None:
            continue
        distance = _angular_distance_deg(float(ra_deg), float(dec_deg), float(item_ra), float(item_dec))
        size = item.get("size_arcmin") or [60.0, 60.0]
        try:
            radius = max(float(size[0]), float(size[1])) / 120.0
        except (TypeError, ValueError, IndexError):
            radius = 0.5
        # Seestar/FITS center coordinates often point at the framed field center,
        # not the catalog object centroid. Large nebula entries therefore need a
        # field-aware tolerance so IC 434/Horsehead-like frames are not demoted to
        # visual-only galaxy guesses.
        tolerance = max(1.2, radius * 1.35 + 0.35)
        if distance <= tolerance:
            score = max(0.78, min(0.98, 0.98 - distance / max(tolerance, 1e-6) * 0.18))
            if best is None or score > best[1]:
                best = (item, score, distance)
    return best


def _visual_type(features: AdaptiveImageFeatures, flags: Dict[str, bool]) -> Tuple[str, float, str]:
    if features.low_snr_score > 0.75 or features.dirty_background_score > 0.72:
        return "generic_low_snr_safe", 0.56, "low_snr_visual_features"
    if flags["star_cluster_dominant"]:
        if features.compactness_score > 0.25 and features.bright_core_score > 0.35:
            return "globular_cluster", 0.70, "visual_features"
        return "open_cluster", 0.64, "visual_features"
    if flags["bright_core"] and flags["large_nebulosity"] and flags["reflection_blue"] and flags["emission_red"]:
        return "bright_emission_reflection_nebula", 0.74, "visual_features"
    if flags["reflection_blue"] and features.halo_risk_score > 0.35:
        return "reflection_nebula_cluster", 0.68, "visual_features"
    if flags["emission_red"] and flags["large_nebulosity"]:
        return "emission_nebula_widefield", 0.66, "visual_features"
    if features.elongation_score > 0.25 and features.bright_core_score > 0.30 and features.nebulosity_area_ratio < 0.28:
        if features.object_area_ratio < 0.11:
            return "small_galaxy", 0.62, "visual_features"
        return "large_galaxy", 0.66, "visual_features"
    if features.faint_structure_score > 0.30 and features.color_balance_score > 0.70 and features.bright_core_score < 0.35:
        return "dark_nebula_low_contrast", 0.58, "visual_features"
    return "generic_low_snr_safe", 0.50, "visual_features_uncertain"


def _auto_hint_type(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    hint = str(metadata.get("AUTO_TARGET_TYPE") or metadata.get("auto_target_type") or "").lower()
    if "dark" in hint and "nebula" in hint:
        return "dark_nebula_low_contrast"
    if "bright_emission" in hint or "emission" in hint:
        return "bright_emission_reflection_nebula"
    if "reflection" in hint:
        return "reflection_nebula_cluster"
    if "nebula" in hint:
        return "bright_emission_reflection_nebula"
    if "galaxy" in hint:
        return "large_galaxy"
    if "cluster" in hint:
        return "open_cluster"
    return None


def build_target_profile(
    features: AdaptiveImageFeatures,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    context_text: str = "",
    catalog: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    catalog_items = catalog if catalog is not None else load_catalog()
    flags = feature_flags(features)
    visual_type, visual_confidence, visual_method = _visual_type(features, flags)
    auto_hint = _auto_hint_type(metadata)
    if auto_hint and visual_type == "small_galaxy" and visual_confidence <= 0.64:
        visual_type = auto_hint
        visual_confidence = 0.66
        visual_method = "auto_target_hint_plus_visual_features"
    coord_match = _catalog_coordinate_match(catalog_items, metadata)
    name_match = _catalog_name_match(catalog_items, _metadata_text(metadata, context_text))
    match = None
    match_method = "catalog_name_match"
    match_distance: Optional[float] = None
    if coord_match:
        item, score, distance = coord_match
        match = (item, score)
        match_method = "catalog_coordinate_match"
        match_distance = distance
    elif name_match:
        match = name_match

    target_name: Optional[str] = None
    target_confidence = visual_confidence
    target_type = visual_type
    method = visual_method
    diagnostics: List[str] = []
    warnings: List[str] = []
    if auto_hint:
        diagnostics.append(f"auto_target_hint={auto_hint}")
        if (
            visual_type
            in {
                "small_galaxy",
                "generic_low_snr_safe",
                "dark_nebula_low_contrast",
                "emission_nebula_widefield",
            }
            and visual_confidence <= 0.66
        ):
            visual_type = auto_hint
            visual_confidence = max(visual_confidence, 0.66)
            visual_method = "auto_target_hint_plus_visual_features"

    if match:
        item, catalog_score = match
        catalog_type = str(item.get("type") or visual_type)
        feature_overlap = 0
        for feature in item.get("features", []):
            key = str(feature)
            aliases = {
                "dense_stars": "dense_star_field",
                "dark_nebula": "faint_outer_cloud",
                "low_contrast": "faint_outer_cloud",
            }
            if flags.get(aliases.get(key, key), False):
                feature_overlap += 1
        if item.get("features"):
            catalog_score = min(0.98, catalog_score + 0.08 * feature_overlap / max(len(item["features"]), 1))
        if catalog_type != visual_type:
            diagnostics.append(
                f"catalog_visual_type_resolution: catalog={catalog_type}, visual={visual_type}"
            )
        if coord_match or catalog_score >= visual_confidence:
            target_name = str(item.get("name") or "")
            target_type = catalog_type
            target_confidence = catalog_score
            method = (
                f"{match_method}_plus_visual_features"
                if feature_overlap
                else match_method
            )
            if match_distance is not None:
                diagnostics.append(f"catalog_coordinate_distance_deg={match_distance:.4f}")

    if (
        auto_hint
        and auto_hint != target_type
        and not target_name
        and target_type
        in {
            "generic_low_snr_safe",
            "dark_nebula_low_contrast",
            "emission_nebula_widefield",
        }
        and target_confidence <= 0.72
    ):
        diagnostics.append(f"auto_target_hint_override: {target_type}->{auto_hint}")
        target_type = auto_hint
        target_confidence = max(target_confidence, 0.66)
        method = "auto_target_hint_plus_visual_features"

    if target_confidence < 0.55:
        target_name = None
        target_type = visual_type if visual_type != "generic_low_snr_safe" else "generic_low_snr_safe"

    policy_name = TYPE_TO_POLICY.get(target_type, DEFAULT_POLICY_NAME)
    profile = {
        "target_name_guess": target_name,
        "target_confidence": round(float(target_confidence), 4),
        "target_type": target_type,
        "pipeline": policy_name,
        "classification_method": method,
        "features": flags,
        "risks": risk_levels(features),
        "image_stats": {
            "bg_median": features.bg_median,
            "bg_std": features.bg_std,
            "bg_mad": features.bg_mad,
            "dynamic_range": features.core_peak_ratio,
            "saturation_ratio": features.core_clip_ratio,
            "edge_black": features.edge_black_ratio,
            "gradient_score": features.gradient_score,
            "dirty_background_score": features.dirty_background_score,
        },
        "object_stats": {
            "object_area_ratio": features.object_area_ratio,
            "core_peak_ratio": features.core_peak_ratio,
            "core_area_ratio": features.bright_core_score,
            "nebulosity_area_ratio": features.nebulosity_area_ratio,
            "symmetry_score": features.symmetry_score,
            "elongation_score": features.elongation_score,
            "compactness_score": features.compactness_score,
        },
        "star_stats": {
            "star_count": features.star_count,
            "star_density": features.star_density,
            "bright_star_count": features.bright_star_count,
            "halo_risk": features.halo_risk_score,
            "star_bloat_score": features.star_bloat_score,
        },
        "color_stats": {
            "red_dominance": features.red_dominance,
            "blue_dominance": features.blue_dominance,
            "green_cast": features.green_cast,
            "chroma_noise": features.chroma_noise_score,
            "color_balance_score": features.color_balance_score,
        },
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
    profile["policy"] = policy_for_profile(profile)
    return profile
