"""Stage 4 plate solving and color calibration."""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from models import PipelineStage
from sirilpy.exceptions import CommandError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_STAGE4_FOCAL_LENGTH_MM = 160.0
DEFAULT_STAGE4_PIXEL_SIZE_UM = 2.9
DEFAULT_STAGE4_INSTRUMENT = "Seestar S30 Pro"
DEFAULT_OSC_SENSOR = "Sony IMX585"
PCC_CATALOG = "gaia"
PCC_CHECKPOINT_STEM = "stage4_pre_pcc"
PCC_CANDIDATE_STEM = "stage4_pcc_candidate"
LOCAL_ASTROMETRIC_FILENAME = "siril_cat_healpix8_astro.dat"
MIN_LOCAL_CATALOG_FILE_BYTES = 1024
EMISSION_NEBULA_TARGET_TYPES = frozenset(
    {
        "emission_nebula",
        "emission_nebula_widefield",
        "bright_emission_reflection_nebula",
    }
)
DUAL_NARROWBAND_KEYWORDS = frozenset(
    {
        "narrowband",
        "narrow-band",
        "dualband",
        "dual-band",
        "dual narrow",
        "dual-narrow",
        "duo narrow",
        "duo-narrow",
        "l-extreme",
        "l-enhance",
        "l-ultimate",
        "ha+oiii",
        "ha oiii",
        "ha-oiii",
        "ha_oiii",
        "haoiii",
        "h-alpha",
        "oiii",
        "o-iii",
        "triband",
        "tri-band",
        "hoo",
        "sho",
        "sii",
        "s-ii",
        "hubble palette",
    }
)


def _stage4_network_enabled() -> bool:
    return (
        os.getenv("SEESTAR_NETWORK_MODE", "1").strip().lower()
        in ENV_TRUE_VALUES
    )


def _stage4_local_astrometric_catalog_path(pipeline) -> Path:
    configured = (
        getattr(pipeline, "local_gaia_astro_catalog", None)
        or os.getenv("SEESTAR_GAIA_ASTRO_CATALOG", "")
    )
    if configured:
        return Path(configured).expanduser()
    return (
        Path.home()
        / ".local"
        / "share"
        / "siril"
        / LOCAL_ASTROMETRIC_FILENAME
    )


def _stage4_valid_catalog_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_LOCAL_CATALOG_FILE_BYTES
    except OSError:
        return False


def _stage4_local_astrometric_catalog_status(pipeline) -> Dict[str, Any]:
    path = _stage4_local_astrometric_catalog_path(pipeline)
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    return {
        "path": str(path),
        "available": _stage4_valid_catalog_file(path),
        "size_bytes": int(size),
        "minimum_size_bytes": MIN_LOCAL_CATALOG_FILE_BYTES,
    }


def _stage4_active_target_type(pipeline) -> str:
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        target_type = str(profile.get("target_type") or "").strip().lower()
        if target_type:
            return target_type
    if hasattr(pipeline, "_active_target_type"):
        return str(pipeline._active_target_type() or "").strip().lower()
    return ""


def _stage4_filter_hint_suggests_narrowband(filter_hint: str) -> bool:
    if not filter_hint:
        return False
    return any(
        keyword in filter_hint
        for keyword in DUAL_NARROWBAND_KEYWORDS
    )


def _stage4_focal_length() -> float:
    return float(os.getenv("SEESTAR_STAGE4_PLATESOLVE_FOCAL", "160.0"))


def _stage4_pixel_size() -> float:
    return float(os.getenv("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE", "2.90"))


def _stage4_platesolve_order() -> str:
    return str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_ORDER", "3") or "").strip()


def _stage4_platesolve_geometry_args() -> Tuple[str, ...]:
    focal = str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_FOCAL", "160") or "160").strip()
    pixelsize = str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE", "2.90") or "2.90").strip()
    # Plate solving adds WCS metadata; it must not rewrite the user's image orientation.
    return ("-noflip", f"-focal={focal}", f"-pixelsize={pixelsize}")


def _stage4_platesolve_args() -> Tuple[str, ...]:
    args = _stage4_platesolve_geometry_args()
    order = _stage4_platesolve_order()
    if order:
        args += (f"-order={order}",)
    return args


def _stage4_header_metadata(pipeline, *metadata_candidates: Any) -> Dict[str, Any]:
    if not hasattr(pipeline, "_read_fits_header_metadata"):
        return {}
    candidates = metadata_candidates or (
        "stage3_bgremoved",
        getattr(pipeline, "source_file", None),
    )
    try:
        metadata = pipeline._read_fits_header_metadata(*candidates)
    except TypeError:
        metadata = pipeline._read_fits_header_metadata("stage3_bgremoved")
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.debug(f"Stage4 FITS header metadata unavailable: {e}")
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _stage4_platesolve_catalogs(pipeline=None) -> Tuple[str, ...]:
    raw = os.getenv("SEESTAR_STAGE4_PLATESOLVE_CATALOGS", "gaia")
    catalogs = tuple(
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    )
    catalogs = catalogs or ("gaia",)
    if pipeline is not None and not _stage4_network_enabled() and "localgaia" not in catalogs:
        catalogs = ("localgaia",) + catalogs
    return catalogs


def _stage4_coordinate_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f"{float(value):.8f}".rstrip("0").rstrip(".")
    text = str(value).strip().strip("'\"")
    if not text:
        return None
    try:
        return f"{float(text):.8f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return re.sub(r"\s+", ":", text)


def _stage4_header_center_coordinates(metadata: Dict[str, Any]) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    for ra_key, dec_key in (
        ("RA", "DEC"),
        ("OBJCTRA", "OBJCTDEC"),
        ("CRVAL1", "CRVAL2"),
    ):
        ra = _stage4_coordinate_text(metadata.get(ra_key))
        dec = _stage4_coordinate_text(metadata.get(dec_key))
        if ra and dec:
            return f"{ra},{dec}"
    return None


def _stage4_header_platesolve_args(metadata: Dict[str, Any]) -> Tuple[str, ...]:
    center = _stage4_header_center_coordinates(metadata)
    if not center:
        return ()
    args = (center,) + _stage4_platesolve_geometry_args()
    radius = str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_HEADER_RADIUS", "") or "").strip()
    if radius:
        args += (f"-radius={radius}",)
    return args


def _stage4_platesolve_variants(
    pipeline,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Tuple[str, ...]]]:
    base = _stage4_platesolve_geometry_args()
    order = _stage4_platesolve_order()
    order_args = (f"-order={order}",) if order else ()
    variants: List[Tuple[str, Tuple[str, ...]]] = []
    for catalog in _stage4_platesolve_catalogs(pipeline):
        variants.append((f"catalog:{catalog}", base + (f"-catalog={catalog}",) + order_args))
    header_base = _stage4_header_platesolve_args(metadata or {})
    if header_base:
        for catalog in _stage4_platesolve_catalogs(pipeline):
            variants.append((f"header:catalog:{catalog}", header_base + (f"-catalog={catalog}",) + order_args))
    return variants


def _stage4_catalog_skip_reason(pipeline, catalog: str) -> Optional[str]:
    if catalog == "localgaia":
        astro_status = _stage4_local_astrometric_catalog_status(pipeline)
        if not astro_status["available"]:
            return (
                "local Gaia astrometric catalog unavailable: "
                f"{astro_status['path']} ({astro_status['size_bytes']} bytes)"
            )
        return None
    if not _stage4_network_enabled():
        return f"online catalog disabled by SEESTAR_NETWORK_MODE=0: {catalog}"
    return None


def _stage4_run_platesolve(pipeline, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[Dict[str, str]]]:
    attempts: List[Dict[str, str]] = []
    original_retries = getattr(pipeline.cfg, "max_retries", None)
    if original_retries is not None:
        pipeline.cfg.max_retries = 0
    try:
        for label, args in _stage4_platesolve_variants(pipeline, metadata):
            command = "platesolve " + " ".join(args)
            catalog = label.rsplit("catalog:", 1)[-1]
            skip_reason = _stage4_catalog_skip_reason(pipeline, catalog)
            if skip_reason:
                attempts.append(
                    {
                        "label": label,
                        "command": command,
                        "status": "skipped",
                        "error": skip_reason,
                    }
                )
                pipeline.log.warn(f"图像解析候选跳过 ({label}): {skip_reason}")
                continue
            pipeline.log.info(f"执行图像解析: {command}")
            try:
                pipeline.cmd_with_check("platesolve", *args)
                attempts.append({"label": label, "command": command, "status": "ok"})
                return True, command, attempts
            except (CommandError, SirilError) as e:
                error_text = str(e)
                attempts.append(
                    {
                        "label": label,
                        "command": command,
                        "status": "failed",
                        "error": error_text,
                    }
                )
                pipeline.log.warn(f"图像解析候选失败 ({label}): {e}")
    finally:
        if original_retries is not None:
            pipeline.cfg.max_retries = original_retries

    last_error = attempts[-1].get("error", "unknown error") if attempts else "no attempts"
    return False, f"platesolve failed: {last_error}", attempts


def _stage4_platesolve_diagnostics(
    attempts: List[Dict[str, str]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    failed_attempts = [attempt for attempt in attempts if attempt.get("status") != "ok"]
    combined_error = " ".join(str(attempt.get("error") or "") for attempt in failed_attempts).lower()
    labels = [str(attempt.get("label") or "") for attempt in attempts]

    if not attempts:
        failure_kind = "not_attempted"
    elif any(attempt.get("status") == "ok" for attempt in attempts):
        failure_kind = "none"
    elif any(token in combined_error for token in ("503", "server", "catalogue", "catalog", "vizier", "unreachable")):
        failure_kind = "catalog_service_or_cache_unavailable"
    elif "generic error" in combined_error or "mock failure" in combined_error:
        failure_kind = "siril_generic_failure"
    else:
        failure_kind = "star_match_or_catalog_failure"

    header_center = _stage4_header_center_coordinates(metadata)
    has_coordinates = bool(header_center)
    next_actions: List[str] = []
    if failure_kind != "none":
        if "catalog:localgaia" in labels:
            next_actions.append("install_or_verify_local_gaia_catalog")
        next_actions.append("verify_network_catalog_access_or_keep_offline_brightstars_enabled")
        if not has_coordinates:
            next_actions.append("ensure_fits_header_has_target_coordinates")
        next_actions.append("check_run_log_for_siril_catalog_error_details")

    return {
        "failure_kind": failure_kind,
        "attempt_count": len(attempts),
        "catalogs_attempted": labels,
        "has_header_coordinates": bool(has_coordinates),
        "header_center_coordinates": header_center,
        "object": metadata.get("OBJECT"),
        "filter": metadata.get("FILTER"),
        "instrument": metadata.get("INSTRUME"),
        "telescope": metadata.get("TELESCOP"),
        "next_actions": next_actions,
    }


def _stage4_run_pcc(
    pipeline,
    *,
    phase: str,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Run exactly one online Gaia PCC behind a killable process boundary."""
    timeout_sec = int(getattr(pipeline.cfg, "stage4_pcc_timeout_sec", 30) or 30)
    timeout_sec = max(5, min(timeout_sec, 120))
    command = f"pcc -catalog={PCC_CATALOG}"
    attempt: Dict[str, Any] = {
        "label": f"catalog:{PCC_CATALOG}",
        "phase": phase,
        "command": command,
        "status": "failed",
        "timeout_sec": timeout_sec,
        "attempt": 1,
        "max_attempts": 1,
    }

    if not _stage4_network_enabled():
        attempt.update(status="skipped", error="network mode disabled")
        return False, "PCC skipped: network mode disabled", [attempt]

    test_runner = getattr(pipeline, "_run_stage4_pcc_once", None)
    if callable(test_runner):
        try:
            ok, detail = test_runner(timeout_sec=timeout_sec, catalog=PCC_CATALOG)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            ok, detail = False, str(error)
        attempt["status"] = "ok" if ok else "failed"
        if not ok:
            attempt["error"] = str(detail)
        return bool(ok), str(detail), [attempt]

    cli = _stage4_resolve_siril_cli()
    process_dir = Path(getattr(pipeline, "process_dir", "") or "")
    if cli is None or not process_dir.is_dir():
        reason = (
            "independent siril-cli unavailable"
            if cli is None
            else f"process directory unavailable: {process_dir}"
        )
        attempt["error"] = reason
        return False, f"PCC {phase} failed: {reason}", [attempt]

    script_path = process_dir / ".stage4_pcc_once.ssf"
    candidate_path = process_dir / f"{PCC_CANDIDATE_STEM}.fit"
    try:
        candidate_path.unlink(missing_ok=True)
        escaped_dir = str(process_dir).replace('"', '\\"')
        script_path.write_text(
            "\n".join(
                (
                    "requires 1.4.0",
                    f'cd "{escaped_dir}"',
                    f"load {PCC_CHECKPOINT_STEM}",
                    command,
                    f"save {PCC_CANDIDATE_STEM}",
                    "close",
                )
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        attempt["error"] = f"unable to prepare PCC script: {error}"
        return False, f"PCC {phase} failed: {attempt['error']}", [attempt]

    cli_command = [str(cli), "-d", str(process_dir)]
    ini_path = os.getenv("SEESTAR_SIRIL_CONFIG", "").strip()
    if ini_path:
        cli_command.extend(("-i", ini_path))
    cli_command.extend(("-s", str(script_path)))
    attempt["runner"] = "independent_siril_cli"
    attempt["cli"] = str(cli)
    pipeline.log.info(
        f"PCC 单次在线 Gaia 校色开始（timeout={timeout_sec}s）"
    )
    try:
        completed = subprocess.run(
            cli_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        attempt.update(status="timeout", error=f"timeout after {timeout_sec}s")
        pipeline.log.warn(f"PCC 单次尝试超时（{timeout_sec}s），不重试")
        return False, f"PCC {phase} timed out after {timeout_sec}s", [attempt]
    except (OSError, subprocess.SubprocessError) as error:
        attempt["error"] = str(error)
        return False, f"PCC {phase} failed: {error}", [attempt]
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    output_lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if output_lines:
        attempt["output_tail"] = " | ".join(output_lines[-8:])
    if completed.returncode != 0:
        attempt["error"] = f"siril-cli exit={completed.returncode}"
        return False, f"PCC {phase} failed: {attempt['error']}", [attempt]
    if not candidate_path.is_file() or candidate_path.stat().st_size <= 0:
        attempt["error"] = "PCC candidate output missing"
        return False, f"PCC {phase} failed: {attempt['error']}", [attempt]

    attempt["status"] = "ok"
    attempt["output"] = candidate_path.name
    return True, command, [attempt]


def _stage4_resolve_siril_cli() -> Optional[Path]:
    configured = os.getenv("SEESTAR_SIRIL_CLI", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate
    resolved = shutil.which("siril-cli")
    return Path(resolved) if resolved else None


def _stage4_filter_semantics_text(pipeline, metadata: Dict[str, Any]) -> str:
    profile = getattr(pipeline, "target_profile", None)
    values: List[Any] = [
        os.getenv("SEESTAR_STAGE4_FILTER_HINT", ""),
        metadata.get("FILTER", ""),
        metadata.get("FILTER1", ""),
        metadata.get("FILTER2", ""),
    ]
    if isinstance(profile, dict):
        values.extend((profile.get("filter", ""), profile.get("filter_name", "")))
    return " ".join(str(value or "").strip().lower() for value in values)


def _stage4_linearity(metadata: Dict[str, Any], *, checkpoint_loaded: bool) -> Dict[str, Any]:
    evidence = " ".join(
        str(metadata.get(key, "") or "").strip().lower()
        for key in ("LINEAR", "NONLINEA", "STRETCH", "HISTORY", "PROCSTEP")
    )
    if any(token in evidence for token in ("nonlinear", "non-linear", "stretched", "histogram")):
        return {"status": "nonlinear", "confidence": 0.95, "reason": "FITS metadata"}
    explicit_linear = metadata.get("LINEAR")
    if explicit_linear is True or str(explicit_linear).strip().lower() in ENV_TRUE_VALUES:
        return {"status": "linear", "confidence": 0.98, "reason": "FITS LINEAR keyword"}
    if checkpoint_loaded:
        return {
            "status": "linear",
            "confidence": 0.96,
            "reason": "Stage 3/Stage 4 linear checkpoint contract",
        }
    return {"status": "unknown", "confidence": 0.0, "reason": "linear checkpoint not confirmed"}


def _stage4_channel_policy(
    pipeline,
    metadata: Dict[str, Any],
    *,
    checkpoint_loaded: bool,
) -> Dict[str, Any]:
    try:
        shape = _stage4_shape_dict(pipeline.siril.get_image_shape())
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError):
        shape = {}
    channels = int(shape.get("channels", 0) or 0)
    linearity = _stage4_linearity(metadata, checkpoint_loaded=checkpoint_loaded)
    filter_text = _stage4_filter_semantics_text(pipeline, metadata)
    narrowband = _stage4_filter_hint_suggests_narrowband(filter_text)

    if channels == 1:
        kind = "mono"
        confidence = 1.0
        action = "skip_color_calibration"
    elif linearity["status"] == "nonlinear":
        kind = "nonlinear_color"
        confidence = float(linearity["confidence"])
        action = "preserve_input"
    elif channels < 3 or linearity["status"] != "linear":
        kind = "unknown"
        confidence = 0.0
        action = "preserve_input_review"
    elif narrowband:
        kind = "narrowband_composite"
        confidence = 0.95
        action = "skip_pcc_local_star_only"
    else:
        kind = "broadband_rgb_osc"
        confidence = 0.90
        action = "single_pcc"

    return {
        "kind": kind,
        "confidence": confidence,
        "action": action,
        "channels": channels,
        "shape": shape,
        "linearity": linearity,
        "filter_hint": filter_text or None,
        "narrowband_detected": bool(narrowband),
    }


def _stage4_color_statistics(chw: np.ndarray) -> Dict[str, Any]:
    rgb = np.asarray(chw[:3], dtype=np.float32)
    finite = np.isfinite(rgb)
    finite_ratio = float(np.mean(finite)) if finite.size else 0.0
    clean = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    lum = _stage4_luminance(clean)
    valid_lum = lum[np.isfinite(lum)]
    if valid_lum.size < 64:
        return {"finite_ratio": finite_ratio, "valid": False}
    bg_limit = float(np.quantile(valid_lum, 0.35))
    bg_mask = lum <= bg_limit
    channel_medians = np.array([float(np.median(channel)) for channel in clean])
    bg_medians = np.array([float(np.median(channel[bg_mask])) for channel in clean])
    bg_mean = max(float(np.mean(np.abs(bg_medians))), 1e-6)
    dynamic = float(np.quantile(valid_lum, 0.995) - np.quantile(valid_lum, 0.05))
    peak = float(np.max(clean))
    clip_level = 0.999 if peak <= 1.5 else 65534.0
    clip_ratio = float(np.mean(np.max(clean, axis=0) >= clip_level))
    return {
        "valid": True,
        "finite_ratio": finite_ratio,
        "channel_medians": [float(value) for value in channel_medians],
        "background_medians": [float(value) for value in bg_medians],
        "background_channel_spread": float((np.max(bg_medians) - np.min(bg_medians)) / bg_mean),
        "dynamic_range": dynamic,
        "highlight_clip_ratio": clip_ratio,
    }


def _stage4_pcc_quality_gate(
    before: np.ndarray,
    after: np.ndarray,
    pipeline,
) -> Tuple[bool, Dict[str, Any]]:
    before_stats = _stage4_color_statistics(before)
    after_stats = _stage4_color_statistics(after)
    target_type = _stage4_active_target_type(pipeline)
    target_name = ""
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        target_name = str(profile.get("target_name_guess") or "")
    emission_target = target_type in EMISSION_NEBULA_TARGET_TYPES
    galaxy_target = "galaxy" in target_type
    bg_spread_max = 0.45 if emission_target else (0.16 if galaxy_target else 0.28)
    gain_ratio_max = float(
        getattr(pipeline.cfg, "stage4_pcc_channel_gain_ratio_max", 1.80) or 1.80
    )
    gain_ratio_max = max(1.10, min(gain_ratio_max, 3.0))
    clip_growth_max = float(
        getattr(pipeline.cfg, "stage4_pcc_clip_growth_max", 0.005) or 0.005
    )
    clip_growth_max = max(0.0, min(clip_growth_max, 0.05))
    reasons: List[str] = []

    if before.shape != after.shape:
        reasons.append("shape_changed")
    if not before_stats.get("valid") or not after_stats.get("valid"):
        reasons.append("invalid_statistics")
    if float(after_stats.get("finite_ratio", 0.0)) < 0.999999:
        reasons.append("non_finite_pixels")

    before_medians = np.asarray(before_stats.get("channel_medians", []), dtype=np.float64)
    after_medians = np.asarray(after_stats.get("channel_medians", []), dtype=np.float64)
    gains: List[float] = []
    gain_ratio = float("inf")
    if before_medians.size == 3 and after_medians.size == 3 and float(np.min(np.abs(before_medians))) > 1e-8:
        gain_values = np.abs(after_medians / before_medians)
        gains = [float(value) for value in gain_values]
        if float(np.min(gain_values)) > 1e-8:
            gain_ratio = float(np.max(gain_values) / np.min(gain_values))
            if gain_ratio > gain_ratio_max:
                reasons.append("channel_gain_ratio_exceeded")
    else:
        reasons.append("channel_gain_unavailable")

    bg_spread = float(after_stats.get("background_channel_spread", float("inf")))
    if bg_spread > bg_spread_max:
        reasons.append("background_chroma_exceeded")
    clip_growth = float(after_stats.get("highlight_clip_ratio", 0.0)) - float(
        before_stats.get("highlight_clip_ratio", 0.0)
    )
    if clip_growth > clip_growth_max:
        reasons.append("highlight_clip_growth_exceeded")
    before_dynamic = float(before_stats.get("dynamic_range", 0.0))
    after_dynamic = float(after_stats.get("dynamic_range", 0.0))
    dynamic_ratio = after_dynamic / max(before_dynamic, 1e-8)
    if before_dynamic > 1e-6 and (dynamic_ratio < 0.50 or dynamic_ratio > 2.50):
        reasons.append("dynamic_range_shift_exceeded")

    enabled = bool(getattr(pipeline.cfg, "stage4_pcc_quality_gate_enabled", True))
    accepted = (not enabled) or not reasons
    report = {
        "enabled": enabled,
        "accepted": bool(accepted),
        "target_type": target_type or "unknown",
        "target_name": target_name or None,
        "target_aware_profile": (
            "emission_nebula_red_dominance_allowed"
            if emission_target
            else ("galaxy_strict_background" if galaxy_target else "general_natural_color")
        ),
        "thresholds": {
            "background_channel_spread_max": bg_spread_max,
            "channel_gain_ratio_max": gain_ratio_max,
            "highlight_clip_growth_max": clip_growth_max,
        },
        "measurements": {
            "channel_gains": gains,
            "channel_gain_ratio": gain_ratio,
            "background_channel_spread": bg_spread,
            "highlight_clip_growth": clip_growth,
            "dynamic_range_ratio": dynamic_ratio,
        },
        "before": before_stats,
        "after": after_stats,
        "rejection_reasons": reasons,
    }
    return bool(accepted), report


def _stage4_shape_dict(shape) -> Dict[str, int]:
    if not shape:
        return {}
    channels, height, width = shape
    return {"channels": int(channels), "height": int(height), "width": int(width)}


def _stage4_pixel_scale_arcsec_per_px() -> float:
    return 206.265 * _stage4_pixel_size() / _stage4_focal_length()


def _stage4_image_geometry(pipeline) -> Dict[str, Any]:
    try:
        shape = _stage4_shape_dict(pipeline.siril.get_image_shape())
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError):
        shape = {}
    crop_report = getattr(pipeline, "stage2_crop_report", None)
    if not isinstance(crop_report, dict):
        crop_report = {}
    pixel_scale = _stage4_pixel_scale_arcsec_per_px()
    width = int(shape.get("width", 0) or 0)
    height = int(shape.get("height", 0) or 0)
    return {
        "instrument": DEFAULT_STAGE4_INSTRUMENT,
        "sensor": DEFAULT_OSC_SENSOR,
        "focal_length_mm": _stage4_focal_length(),
        "pixel_size_um": _stage4_pixel_size(),
        "pixel_scale_arcsec_per_px": pixel_scale,
        "current_shape": shape,
        "cropped_fov_deg": {
            "width": width * pixel_scale / 3600.0 if width else None,
            "height": height * pixel_scale / 3600.0 if height else None,
        },
        "stage2_crop": {
            "mode": crop_report.get("mode"),
            "original_shape": crop_report.get("original_shape"),
            "final_shape": crop_report.get("final_shape") or crop_report.get("current_shape"),
            "total_crop": crop_report.get("total_crop"),
            "crop_count": len(crop_report.get("crops", [])) if isinstance(crop_report.get("crops"), list) else 0,
        },
        "notes": [
            "Stage4 platesolve uses the Stage2/Stage3 cropped frame dimensions.",
            "Focal length and pixel size remain the Seestar S30 Pro tele optical parameters after crop.",
        ],
    }


def _stage4_image_as_chw(image_data: Any) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    arr = np.asarray(image_data)
    if arr.size == 0:
        raise ValueError("image buffer is empty")

    original_dtype = arr.dtype
    if arr.ndim == 2:
        work = arr[np.newaxis, :, :].astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(chw[0], original_dtype)

        return work, restore

    if arr.ndim != 3:
        raise ValueError(f"unsupported image ndim: {arr.ndim}")

    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        work = arr.astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(chw, original_dtype)

        return work, restore

    if arr.shape[-1] in (1, 3, 4):
        work = np.moveaxis(arr, -1, 0).astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(np.moveaxis(chw, 0, -1), original_dtype)

        return work, restore

    if arr.shape[0] >= 3:
        work = arr.astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(chw, original_dtype)

        return work, restore

    raise ValueError(f"unsupported image shape: {arr.shape}")


def _stage4_cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype, copy=False)
    return values.astype(dtype, copy=False)


def _stage4_luminance(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] < 3:
        return rgb[0]
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)


def _stage4_sigma_clip_star_colors(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_iter: int = 3,
    sigma: float = 2.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    clipped = np.asarray(mask, dtype=bool).copy()
    report: Dict[str, Any] = {
        "enabled": True,
        "iterations": 0,
        "input_pixels": int(np.count_nonzero(clipped)),
        "output_pixels": int(np.count_nonzero(clipped)),
    }
    if report["input_pixels"] < 8 or rgb.shape[0] < 3:
        return clipped, report

    eps = 1e-6
    for iteration in range(max(1, int(max_iter))):
        r = rgb[0][clipped]
        g = rgb[1][clipped]
        b = rgb[2][clipped]
        if r.size < 8 or float(np.min(g)) <= 0.0:
            break
        ratios = np.stack(
            [
                np.log((r + eps) / (g + eps)),
                np.log((b + eps) / (g + eps)),
            ],
            axis=0,
        )
        keep_values = np.ones(r.shape, dtype=bool)
        for values in ratios:
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-4)
            keep_values &= np.abs(values - center) <= sigma * scale
        next_clipped = np.zeros_like(clipped)
        coords = np.flatnonzero(clipped)
        next_clipped.reshape(-1)[coords[keep_values]] = True
        if int(np.count_nonzero(next_clipped)) == int(np.count_nonzero(clipped)):
            report["iterations"] = iteration + 1
            clipped = next_clipped
            break
        clipped = next_clipped
        report["iterations"] = iteration + 1

    report["output_pixels"] = int(np.count_nonzero(clipped))
    return clipped, report


def _stage4_soft_star_mask(star_mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, min(int(radius), 4))
    height, width = star_mask.shape
    source = star_mask.astype(np.float32)
    soft = source.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = math.hypot(dx, dy)
            if distance == 0.0 or distance > radius + 0.25:
                continue
            weight = max(0.12, 1.0 - distance / (radius + 0.75))
            shifted = np.zeros_like(source)
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_y0 = src_y0 + dy
            dst_y1 = src_y1 + dy
            dst_x0 = src_x0 + dx
            dst_x1 = src_x1 + dx
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = source[
                src_y0:src_y1, src_x0:src_x1
            ]
            soft = np.maximum(soft, shifted * weight)
    return np.clip(soft, 0.0, 1.0)


def _stage4_star_white_balance(chw: np.ndarray, pipeline) -> Tuple[np.ndarray, Dict[str, Any]]:
    output = np.clip(chw.astype(np.float32, copy=True), 0.0, None)
    if output.shape[0] < 3:
        return output, {"applied": False, "reason": "mono image", "white_reference_pixels": 0}

    rgb = output[:3]
    lum = _stage4_luminance(rgb)
    finite_mask = np.isfinite(lum)
    if int(np.count_nonzero(finite_mask)) < 256:
        return output, {"applied": False, "reason": "not enough finite pixels", "white_reference_pixels": 0}

    valid_lum = lum[finite_mask]
    max_channel = np.max(rgb, axis=0)
    min_channel = np.min(rgb, axis=0)
    chroma = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    saturation_limit = float(np.quantile(max_channel[finite_mask], 0.999))

    min_pixels = int(getattr(pipeline.cfg, "stage4_local_star_wb_min_pixels", 32) or 32)
    min_pixels = max(16, min(min_pixels, 4096))
    star_mask = np.zeros_like(finite_mask, dtype=bool)
    sample_policy = "medium-bright unsaturated low-chroma stars"
    for quantile, sigma_floor, chroma_limit in (
        (0.985, 3.0, 0.35),
        (0.975, 2.5, 0.45),
        (0.965, 2.0, 0.55),
    ):
        median_lum = float(np.median(valid_lum))
        std_lum = float(np.std(valid_lum))
        low = max(float(np.quantile(valid_lum, quantile)), median_lum + sigma_floor * std_lum)
        high = float(np.quantile(valid_lum, 0.9995))
        if high <= low:
            high = float(np.max(valid_lum))
        candidate_mask = (
            finite_mask
            & (lum >= low)
            & (lum <= high)
            & (max_channel <= saturation_limit)
            & (chroma <= chroma_limit)
        )
        if int(np.count_nonzero(candidate_mask)) >= min_pixels:
            star_mask = candidate_mask
            sample_policy = (
                "iterative sigma-clipped stars "
                f"(lum_q={quantile:.3f}, sigma_floor={sigma_floor:.1f}, "
                f"chroma<={chroma_limit:.2f})"
            )
            break
        if int(np.count_nonzero(candidate_mask)) > int(np.count_nonzero(star_mask)):
            star_mask = candidate_mask
            sample_policy = (
                "relaxed candidate stars "
                f"(lum_q={quantile:.3f}, sigma_floor={sigma_floor:.1f}, "
                f"chroma<={chroma_limit:.2f})"
            )

    sigma_report: Dict[str, Any] = {"enabled": False}
    if int(np.count_nonzero(star_mask)) >= max(8, min_pixels // 2):
        clipped_mask, sigma_report = _stage4_sigma_clip_star_colors(rgb, star_mask)
        clipped_pixels = int(np.count_nonzero(clipped_mask))
        if clipped_pixels >= max(16, min_pixels // 2):
            star_mask = clipped_mask

    star_pixels = int(np.count_nonzero(star_mask))
    if star_pixels < min_pixels:
        return output, {
            "applied": False,
            "reason": "insufficient unsaturated low-chroma star samples",
            "white_reference_pixels": star_pixels,
            "required_pixels": min_pixels,
            "sample_policy": sample_policy,
            "sigma_clip": sigma_report,
        }

    medians = np.array([float(np.median(channel[star_mask])) for channel in rgb], dtype=np.float32)
    if float(np.min(medians)) <= 0.0:
        return output, {
            "applied": False,
            "reason": "invalid star channel medians",
            "white_reference_pixels": star_pixels,
            "channel_medians_before": [float(v) for v in medians],
        }

    target = float(np.median(medians))
    gain_limit = float(getattr(pipeline.cfg, "stage4_local_star_wb_gain_limit", 1.20) or 1.20)
    gain_limit = max(1.01, min(gain_limit, 1.50))
    gains = np.clip(target / medians, 1.0 / gain_limit, gain_limit)
    mask_radius = int(getattr(pipeline.cfg, "stage4_local_star_mask_radius", 2) or 2)
    soft_mask = _stage4_soft_star_mask(star_mask, mask_radius)
    mask_coverage = float(np.mean(soft_mask > 0.01))
    coverage_max = float(
        getattr(pipeline.cfg, "stage4_local_star_mask_coverage_max", 0.12) or 0.12
    )
    coverage_max = max(0.01, min(coverage_max, 0.30))
    if mask_coverage > coverage_max:
        return output, {
            "applied": False,
            "reason": "star soft mask coverage exceeds safety limit",
            "white_reference_pixels": star_pixels,
            "mask_coverage": mask_coverage,
            "mask_coverage_max": coverage_max,
        }
    local_gains = 1.0 + (gains[:, np.newaxis, np.newaxis] - 1.0) * soft_mask[np.newaxis]
    output[:3] = np.clip(rgb * local_gains, 0.0, None)
    return output, {
        "applied": True,
        "white_reference_pixels": star_pixels,
        "channel_medians_before": [float(v) for v in medians],
        "target_median": target,
        "gains": [float(v) for v in gains],
        "gain_limit": gain_limit,
        "application": "star_soft_mask_only",
        "mask_radius": mask_radius,
        "mask_coverage": mask_coverage,
        "mask_coverage_max": coverage_max,
        "sample_policy": sample_policy,
        "sigma_clip": sigma_report,
    }


def _stage4_write_image_pixels(pipeline, pixels: np.ndarray) -> None:
    lock_factory = getattr(pipeline.siril, "image_lock", None)
    if callable(lock_factory):
        with lock_factory():
            pipeline.siril.set_image_pixeldata(pixels)
        return
    pipeline.siril.set_image_pixeldata(pixels)


def _stage4_local_color_fallback(
    pipeline,
    *,
    narrowband: bool = False,
) -> Tuple[bool, str, str, float, Dict[str, Any], str]:
    image_data = pipeline.siril.get_image_pixeldata(preview=False)
    chw, restore = _stage4_image_as_chw(image_data)
    wb_enabled = bool(getattr(pipeline.cfg, "stage4_local_star_wb_enabled", True))
    star_report: Dict[str, Any] = {"applied": False, "reason": "disabled", "white_reference_pixels": 0}
    balanced = chw
    if wb_enabled:
        balanced, star_report = _stage4_star_white_balance(chw, pipeline)
        if star_report.get("applied"):
            _stage4_write_image_pixels(pipeline, restore(balanced))

    report = {
        "narrowband": bool(narrowband),
        "global_white_balance": {"applied": False, "prohibited": True},
        "input_color_preserved_when_not_applied": True,
        "star_white_balance": star_report,
    }
    if star_report.get("applied"):
        message = "local star-soft-mask color restoration applied"
        if narrowband:
            message += " (narrowband policy; global white balance prohibited)"
        return True, "LOCAL_STAR_COLOR_RESTORE", "local_star_mask_fallback", 0.55, report, message

    message = "star samples insufficient; input color preserved"
    return False, "PRESERVE_INPUT", "insufficient_star_samples", 0.30, report, message


def run_stage4_color_calibration(pipeline) -> None:
    """Stage 4: plate solve, one guarded Gaia PCC, then local star-only fallback."""
    stage_label = PipelineStage.COLOR_CALIBRATION.label
    pipeline.log.stage_start(stage_label)
    status = "ok"
    hard_degraded = False
    requires_review = False
    color_method = "PRESERVE_INPUT"
    color_warning = ""
    color_confidence = 0.0
    messages: List[str] = []
    local_fallback_report: Optional[Dict[str, Any]] = None
    pcc_attempts: List[Dict[str, Any]] = []
    pcc_quality_report: Dict[str, Any] = {"enabled": True, "accepted": False, "status": "not_run"}
    rollback_report: Dict[str, Any] = {"required": False, "restored": False}
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}
    stage4_input_stem = "stage3_bgremoved"
    checkpoint_loaded = False

    try:
        pipeline.cmd_with_check("load", stage4_input_stem)
        checkpoint_loaded = True
        messages.append(f"input={stage4_input_stem}")
    except (CommandError, SirilError) as e:
        status = "degraded"
        hard_degraded = True
        requires_review = True
        messages.append(f"linear checkpoint load failed; preserving current image: {e}")
        pipeline.log.warn(f"Stage4 线性检查点载入失败，禁止 PCC: {e}")

    stage4_metadata = _stage4_header_metadata(pipeline)
    setattr(pipeline, "_stage4_header_metadata", stage4_metadata)
    stage4_geometry = _stage4_image_geometry(pipeline)
    crop_total = (stage4_geometry.get("stage2_crop") or {}).get("total_crop") or {}
    if crop_total:
        messages.append(
            "stage4 geometry uses stage2 crop "
            f"L/T/R/B={crop_total.get('left')}/{crop_total.get('top')}/"
            f"{crop_total.get('right')}/{crop_total.get('bottom')}"
        )
    pipeline.log.info(
        "Stage4 instrument geometry: "
        f"{DEFAULT_STAGE4_INSTRUMENT} tele sensor={DEFAULT_OSC_SENSOR}, "
        f"focal={_stage4_focal_length():g}mm, "
        f"pixelsize={_stage4_pixel_size():g}um, "
        f"shape={stage4_geometry.get('current_shape')}"
    )

    pipeline.platesolve_ok = False
    platesolve_attempted = True
    platesolve_command = "platesolve " + " ".join(_stage4_platesolve_args())
    platesolve_attempts: List[Dict[str, str]] = []
    if bool(getattr(pipeline.cfg, "stage4_platesolve_enabled", True)):
        pipeline.platesolve_ok, platesolve_result, platesolve_attempts = _stage4_run_platesolve(
            pipeline,
            stage4_metadata,
        )
        if pipeline.platesolve_ok:
            platesolve_command = platesolve_result
            messages.append(f"{platesolve_command} ok")
        else:
            status = "degraded"
            messages.append(platesolve_result)
            pipeline.log.warn(f"图像解析失败: {platesolve_result}")
    else:
        platesolve_attempted = False
        status = "degraded"
        messages.append("platesolve disabled by config; stage4_psolved will mirror input")
        pipeline.log.warn("Stage4 platesolve disabled by config")

    ps_saved = pipeline._save_stage_output("stage4_psolved")
    if not ps_saved:
        status = "degraded"
        hard_degraded = True
        messages.append("stage4_psolved 保存失败")

    if hasattr(pipeline, "_run_target_profile_preflight"):
        profile_msg = pipeline._run_target_profile_preflight(
            source="Stage4 preflight",
            metadata_candidates=("stage4_psolved", "stage3_bgremoved", getattr(pipeline, "source_file", None)),
        )
        if profile_msg:
            messages.append(profile_msg)
            policy = getattr(pipeline, "pipeline_policy", {}) or {}
            stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}
    refreshed_metadata = _stage4_header_metadata(
        pipeline,
        "stage4_psolved",
        "stage3_bgremoved",
        getattr(pipeline, "source_file", None),
    )
    if refreshed_metadata:
        stage4_metadata = refreshed_metadata
        setattr(pipeline, "_stage4_header_metadata", stage4_metadata)

    channel_policy = _stage4_channel_policy(
        pipeline,
        stage4_metadata,
        checkpoint_loaded=checkpoint_loaded,
    )
    target_aware_color = _stage4_active_target_type(pipeline) in EMISSION_NEBULA_TARGET_TYPES
    pre_pcc_saved = pipeline._save_stage_output(PCC_CHECKPOINT_STEM)
    if not pre_pcc_saved:
        status = "degraded"
        hard_degraded = True
        requires_review = True
        messages.append("immutable pre_pcc checkpoint save failed; PCC prohibited")

    before_chw: Optional[np.ndarray] = None
    try:
        before_pixels = pipeline.siril.get_image_pixeldata(preview=False)
        before_chw, _restore_before = _stage4_image_as_chw(before_pixels)
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        hard_degraded = True
        requires_review = True
        status = "degraded"
        messages.append(f"pre_pcc pixels unavailable: {error}")

    policy_status = "not_applicable"
    pcc_allowed = bool(
        channel_policy["action"] == "single_pcc"
        and pipeline.platesolve_ok
        and pre_pcc_saved
        and before_chw is not None
        and _stage4_network_enabled()
    )

    if channel_policy["kind"] == "mono":
        policy_status = "skipped_by_policy"
        color_method = "SKIPPED_MONO"
        color_confidence = 1.0
        messages.append("mono input: color calibration skipped by policy")
    elif channel_policy["kind"] == "nonlinear_color":
        policy_status = "skipped_by_policy"
        color_method = "PRESERVE_INPUT"
        color_confidence = 0.80
        messages.append("nonlinear input: input colors preserved; PCC prohibited")
    elif channel_policy["kind"] == "unknown":
        policy_status = "preserved_unknown"
        color_method = "PRESERVE_INPUT"
        color_warning = "linear_or_channel_semantics_unknown"
        color_confidence = 0.20
        requires_review = True
        status = "degraded"
        messages.append("linear/channel semantics unknown: input colors preserved")
    elif channel_policy["kind"] == "narrowband_composite":
        policy_status = "skipped_by_policy"
        messages.append("high-confidence narrowband composite: PCC and global white balance skipped")
        try:
            (
                local_applied,
                color_method,
                color_warning,
                color_confidence,
                local_fallback_report,
                fallback_message,
            ) = _stage4_local_color_fallback(pipeline, narrowband=True)
            messages.append(fallback_message)
            if not local_applied:
                color_method = "PRESERVE_INPUT"
                color_warning = "narrowband_preserved_no_star_samples"
                color_confidence = 0.80
        except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
            color_method = "PRESERVE_INPUT"
            color_warning = "narrowband_local_star_restore_failed"
            color_confidence = 0.75
            messages.append(f"narrowband local star restoration unavailable; colors preserved: {error}")
    else:
        if not pcc_allowed:
            reason = (
                "plate solve unavailable"
                if not pipeline.platesolve_ok
                else ("network mode disabled" if not _stage4_network_enabled() else "pre_pcc unavailable")
            )
            messages.append(f"PCC skipped: {reason}")
        else:
            pcc_ok, pcc_result, pcc_attempts = _stage4_run_pcc(
                pipeline,
                phase="linear_broadband",
            )
            if pcc_ok:
                try:
                    pipeline.cmd_with_check("load", PCC_CANDIDATE_STEM)
                    candidate_pixels = pipeline.siril.get_image_pixeldata(preview=False)
                    candidate_chw, _restore_candidate = _stage4_image_as_chw(candidate_pixels)
                    accepted, pcc_quality_report = _stage4_pcc_quality_gate(
                        before_chw,
                        candidate_chw,
                        pipeline,
                    )
                    if accepted:
                        color_method = "PCC"
                        color_confidence = 0.78
                        policy_status = "accepted"
                        messages.append(f"{pcc_result} accepted by target-aware quality gate")
                        pipeline.log.info("PCC 单次 Gaia 校色通过目标感知质量门")
                    else:
                        color_warning = "pcc_quality_gate_rejected"
                        messages.append(
                            "PCC candidate rejected: "
                            + ",".join(pcc_quality_report.get("rejection_reasons", []))
                        )
                except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
                    pcc_quality_report = {
                        "enabled": True,
                        "accepted": False,
                        "status": "candidate_load_or_measure_failed",
                        "error": str(error),
                    }
                    color_warning = "pcc_candidate_unavailable"
                    messages.append(f"PCC candidate unavailable: {error}")
            else:
                color_warning = "pcc_single_attempt_failed"
                messages.append(pcc_result)

        if color_method != "PCC":
            rollback_report["required"] = bool(pcc_attempts)
            try:
                pipeline.cmd_with_check("load", PCC_CHECKPOINT_STEM)
                rollback_report.update(restored=True, checkpoint=f"{PCC_CHECKPOINT_STEM}.fit")
                messages.append("restored immutable pre_pcc linear checkpoint")
            except (CommandError, SirilError) as error:
                rollback_report.update(restored=False, error=str(error))
                hard_degraded = True
                messages.append(f"pre_pcc restore failed: {error}")

            if rollback_report.get("restored") or not pcc_attempts:
                try:
                    (
                        local_applied,
                        color_method,
                        fallback_warning,
                        color_confidence,
                        local_fallback_report,
                        fallback_message,
                    ) = _stage4_local_color_fallback(pipeline, narrowband=False)
                    color_warning = color_warning or fallback_warning
                    messages.append(fallback_message)
                    if not local_applied:
                        color_method = "PRESERVE_INPUT"
                except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
                    color_method = "PRESERVE_INPUT"
                    color_warning = color_warning or "local_star_fallback_failed"
                    color_confidence = 0.20
                    messages.append(f"local star-only fallback failed; input preserved: {error}")
            requires_review = True
            status = "degraded"
            policy_status = "fallback_review_required"

    if (
        color_warning
        and stage4_policy.get("reduce_saturation_if_solution_imprecise", False)
    ):
        messages.append("color policy limits later saturation/color gains due to imprecise solution")
    if color_method == "LOCAL_STAR_COLOR_RESTORE" and platesolve_attempted and not pipeline.platesolve_ok:
        messages.append("platesolve 失败，已使用恒星软遮罩局部色彩回退")

    color_saved = pipeline._save_stage_output("stage4_color")
    if not color_saved:
        status = "degraded"
        hard_degraded = True
        messages.append("stage4_color 输出保存失败")
    elif not hard_degraded and not requires_review:
        status = "ok"
    if color_saved and hasattr(pipeline, "_create_stage_review_bundle"):
        review = pipeline._create_stage_review_bundle(
            "stage4_color_calibration",
            stage4_input_stem,
            "stage4_color",
            context={
                "method": color_method,
                "color_confidence": color_confidence,
                "warning": color_warning or None,
            },
        )
        if review.get("report_path"):
            messages.append(f"review_bundle={review['report_path']}")

    pipeline.color_calibration_report = {
        "stage": "stage4_color",
        "input": stage4_input_stem,
        "platesolve": {
            "attempted": platesolve_attempted,
            "ok": bool(pipeline.platesolve_ok),
            "command": platesolve_command,
            "attempts": platesolve_attempts,
            "diagnostics": _stage4_platesolve_diagnostics(platesolve_attempts, stage4_metadata),
            "output": "stage4_psolved.fit",
            "instrument_geometry": stage4_geometry,
            "input_metadata": stage4_metadata,
        },
        "method": color_method,
        "target_aware_color_mapping": bool(target_aware_color),
        "channel_policy": channel_policy,
        "photometric_calibration_allowed": bool(pcc_allowed),
        "requires_review": bool(requires_review),
        "global_white_balance": {"applied": False, "prohibited": True},
        "pcc": {
            "catalog": PCC_CATALOG,
            "max_attempts": 1,
            "timeout_sec": int(getattr(pipeline.cfg, "stage4_pcc_timeout_sec", 30) or 30),
            "policy_status": policy_status,
            "attempts": pcc_attempts,
            "network_enabled": _stage4_network_enabled(),
            "quality_gate": pcc_quality_report,
            "rollback": rollback_report,
        },
        "local_fallback": local_fallback_report,
        "status": (
            "review_required"
            if requires_review
            else ("success_with_warning" if color_warning else "success")
        ),
        "warning": color_warning or None,
        "color_confidence": color_confidence,
        "policy": (
            pipeline._active_policy_name()
            if hasattr(pipeline, "_active_policy_name")
            else str(policy.get("policy_name", "generic_low_snr_safe"))
        ),
        "policy_adjustments": {
            "reduce_saturation_boost": bool(
                stage4_policy.get("reduce_saturation_if_solution_imprecise", False)
                and color_warning
            ),
            "blue_gain_limit": stage4_policy.get("blue_gain_limit"),
            "red_gain_limit": stage4_policy.get("red_gain_limit"),
            "max_allowed_saturation_boost": stage4_policy.get("max_allowed_saturation_boost"),
        },
        "outputs": {
            "psolved": "stage4_psolved.fit",
            "pre_pcc": f"{PCC_CHECKPOINT_STEM}.fit",
            "color": "stage4_color.fit",
            "compat_color": "stage4_colorbalanced.fit",
        },
        "messages": messages,
    }
    pipeline._write_stage_json("color_calibration_report.json", pipeline.color_calibration_report)

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(stage_label, status, elapsed, "；".join(messages))
