"""Stage 4 plate solving and color calibration."""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from sirilpy.exceptions import CommandError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_SPCC_WHITE_REF = "Average Spiral Galaxy"
NEBULA_SPCC_WHITE_REF = "Star, type G2(v)"
DEFAULT_STAGE4_PLATESOLVE_ARGS = ("-focal=160", "-pixelsize=2.90", "-order=3")
DEFAULT_STAGE4_FOCAL_LENGTH_MM = 160.0
DEFAULT_STAGE4_PIXEL_SIZE_UM = 2.9
DEFAULT_STAGE4_INSTRUMENT = "Seestar S30 Pro"
DEFAULT_OSC_SENSOR = "Sony IMX585"
DEFAULT_OSC_FILTER_LP = "ZWO Seestar LP"
DEFAULT_OSC_FILTER_NO_FILTER = "No filter"
DEFAULT_SPCC_LIMITMAG = "10.5"
SPCC_RUNTIME_CPU = 1
SPCC_NARROWBAND_ARGS = (
    "-narrowband",
    "-rwl=656.28",
    "-rbw=20",
    "-gwl=500.70",
    "-gbw=30",
    "-bwl=500.70",
    "-bbw=30",
)
DEFAULT_PCC_CATALOGS = ("localgaia", "gaia", "nomad", "apass")
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
    }
)
S30_PRO_LP_FILTER_KEYWORDS = frozenset(
    {
        "lp",
        "light pollution",
        "light-pollution",
    }
)
NO_LP_FILTER_KEYWORDS = frozenset(
    {
        "no filter",
        "nofilter",
        "no_filter",
        "no-filter",
        "no lp",
        "no-lp",
        "without lp",
        "lp off",
        "clear",
        "none",
    }
)


def _stage4_active_target_type(pipeline) -> str:
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        target_type = str(profile.get("target_type") or "").strip().lower()
        if target_type:
            return target_type
    if hasattr(pipeline, "_active_target_type"):
        return str(pipeline._active_target_type() or "").strip().lower()
    return ""


def _stage4_target_aware_color_mapping(pipeline) -> bool:
    target_type = _stage4_active_target_type(pipeline)
    if target_type in EMISSION_NEBULA_TARGET_TYPES:
        return True
    if bool(getattr(pipeline.cfg, "stage4_spcc_builtin_dualband_filter_enabled", False)):
        return True

    candidates = [
        getattr(pipeline.cfg, "stage4_spcc_osc_filter", ""),
        getattr(pipeline.cfg, "stage4_spcc_osc_sensor", ""),
        getattr(pipeline, "source_file", ""),
    ]
    metadata = getattr(pipeline, "_stage4_header_metadata", None)
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("FILTER", ""),
                metadata.get("INSTRUME", ""),
                metadata.get("TELESCOP", ""),
            ]
        )
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        candidates.extend(
            [
                profile.get("filter", ""),
                profile.get("filter_name", ""),
                profile.get("instrument", ""),
                profile.get("target_type", ""),
            ]
        )

    text = " ".join(str(item or "").lower() for item in candidates)
    return _stage4_filter_hint_suggests_narrowband(text) or any(
        keyword in text for keyword in DUAL_NARROWBAND_KEYWORDS
    )


def _stage4_effective_spcc_white_ref(pipeline) -> Tuple[str, str]:
    configured = str(
        getattr(pipeline.cfg, "stage4_spcc_white_ref", DEFAULT_SPCC_WHITE_REF)
        or DEFAULT_SPCC_WHITE_REF
    )
    if configured != DEFAULT_SPCC_WHITE_REF:
        return configured, "explicit_config"

    if (
        bool(getattr(pipeline.cfg, "stage4_spcc_adaptive_white_ref_enabled", False))
        and _stage4_target_aware_color_mapping(pipeline)
    ):
        target_type = _stage4_active_target_type(pipeline) or "dual_narrowband"
        nebula_ref = str(
            getattr(pipeline.cfg, "stage4_spcc_nebula_white_ref", NEBULA_SPCC_WHITE_REF)
            or NEBULA_SPCC_WHITE_REF
        )
        return nebula_ref, f"target_profile:{target_type}"

    return configured, "default"


def _stage4_spcc_args(pipeline, *, whiteref: Optional[str] = None) -> Tuple[Tuple[str, ...], List[str]]:
    mode = str(getattr(pipeline.cfg, "stage4_spcc_sensor_mode", "osc") or "osc").lower()
    whiteref = str(whiteref or DEFAULT_SPCC_WHITE_REF)
    messages: List[str] = []

    if mode in {"mono", "mono_lrgb", "lrgb"}:
        pipeline.cmd_with_check("spcc_list", "monosensor")
        pipeline.cmd_with_check("spcc_list", "redfilter")
        pipeline.cmd_with_check("spcc_list", "greenfilter")
        pipeline.cmd_with_check("spcc_list", "bluefilter")
        sensor = str(getattr(pipeline.cfg, "stage4_spcc_mono_sensor", "") or "")
        r_filter = str(getattr(pipeline.cfg, "stage4_spcc_r_filter", "") or "")
        g_filter = str(getattr(pipeline.cfg, "stage4_spcc_g_filter", "") or "")
        b_filter = str(getattr(pipeline.cfg, "stage4_spcc_b_filter", "") or "")
        if not all((sensor, r_filter, g_filter, b_filter)):
            messages.append("Mono/LRGB SPCC config incomplete")
        return (
            (
                _stage4_siril_named_arg("monosensor", sensor),
                _stage4_siril_named_arg("rfilter", r_filter),
                _stage4_siril_named_arg("gfilter", g_filter),
                _stage4_siril_named_arg("bfilter", b_filter),
                _stage4_siril_named_arg("whiteref", whiteref),
            )
            + _stage4_spcc_limitmag_args(pipeline)
            + _stage4_spcc_bgtol_args(pipeline),
            messages,
        )

    pipeline.cmd_with_check("spcc_list", "oscsensor")
    sensor = str(getattr(pipeline.cfg, "stage4_spcc_osc_sensor", DEFAULT_OSC_SENSOR) or DEFAULT_OSC_SENSOR)
    osc_mode = _stage4_effective_osc_spcc_mode(pipeline)
    if bool(osc_mode["narrowband"]):
        messages.append(f"OSC SPCC mode=narrowband ({osc_mode['reason']})")
        return (
            (
                _stage4_siril_named_arg("oscsensor", sensor),
                _stage4_siril_named_arg("whiteref", whiteref),
            )
            + SPCC_NARROWBAND_ARGS
            + _stage4_spcc_limitmag_args(pipeline)
            + _stage4_spcc_bgtol_args(pipeline),
            messages,
        )

    pipeline.cmd_with_check("spcc_list", "oscfilter")
    osc_filter = str(osc_mode["osc_filter"])
    messages.append(f"OSC SPCC filter={osc_filter} ({osc_mode['reason']})")
    return (
        (
            _stage4_siril_named_arg("oscsensor", sensor),
            _stage4_siril_named_arg("oscfilter", osc_filter),
            _stage4_siril_named_arg("whiteref", whiteref),
        )
        + _stage4_spcc_limitmag_args(pipeline)
        + _stage4_spcc_bgtol_args(pipeline),
        messages,
    )


def _stage4_effective_osc_spcc_mode(pipeline) -> Dict[str, Any]:
    configured = str(getattr(pipeline.cfg, "stage4_spcc_osc_filter", "") or "").strip()
    if configured:
        configured_hint = configured.lower()
        if _stage4_filter_hint_suggests_narrowband(configured_hint):
            return {
                "narrowband": True,
                "osc_filter": "",
                "reason": "explicit_narrowband_config",
                "narrowband_args": list(SPCC_NARROWBAND_ARGS),
            }
        if _stage4_filter_hint_suggests_no_lp(configured_hint):
            return {
                "narrowband": False,
                "osc_filter": DEFAULT_OSC_FILTER_NO_FILTER,
                "reason": "explicit_no_lp_config",
                "narrowband_args": [],
            }
        if _stage4_filter_hint_suggests_lp(configured_hint):
            return {
                "narrowband": False,
                "osc_filter": DEFAULT_OSC_FILTER_LP,
                "reason": "explicit_lp_config",
                "narrowband_args": [],
            }
        return {
            "narrowband": False,
            "osc_filter": configured,
            "reason": "explicit_config",
            "narrowband_args": [],
        }

    filter_hint = _stage4_filter_hint_text(pipeline)
    if _stage4_filter_hint_suggests_narrowband(filter_hint):
        return {
            "narrowband": True,
            "osc_filter": "",
            "reason": "narrowband_filter_hint",
            "narrowband_args": list(SPCC_NARROWBAND_ARGS),
        }
    if _stage4_filter_hint_suggests_no_lp(filter_hint):
        return {
            "narrowband": False,
            "osc_filter": DEFAULT_OSC_FILTER_NO_FILTER,
            "reason": "no_lp_filter_hint",
            "narrowband_args": [],
        }
    if _stage4_filter_hint_suggests_lp(filter_hint):
        return {
            "narrowband": False,
            "osc_filter": DEFAULT_OSC_FILTER_LP,
            "reason": "lp_filter_hint",
            "narrowband_args": [],
        }
    if bool(getattr(pipeline.cfg, "stage4_spcc_builtin_dualband_filter_enabled", False)):
        return {
            "narrowband": False,
            "osc_filter": DEFAULT_OSC_FILTER_LP,
            "reason": "builtin_lp_filter",
            "narrowband_args": [],
        }
    return {
        "narrowband": False,
        "osc_filter": DEFAULT_OSC_FILTER_LP,
        "reason": "default_lp_filter",
        "narrowband_args": [],
    }


def _stage4_filter_hint_suggests_no_lp(filter_hint: str) -> bool:
    if not filter_hint:
        return False
    return any(keyword in filter_hint for keyword in NO_LP_FILTER_KEYWORDS)


def _stage4_filter_hint_suggests_lp(filter_hint: str) -> bool:
    if not filter_hint:
        return False
    if re.search(r"(^|[^a-z0-9])lp([^a-z0-9]|$)", filter_hint):
        return True
    return any(keyword in filter_hint for keyword in S30_PRO_LP_FILTER_KEYWORDS)


def _stage4_filter_hint_suggests_narrowband(filter_hint: str) -> bool:
    if not filter_hint:
        return False
    return any(
        keyword in filter_hint
        for keyword in DUAL_NARROWBAND_KEYWORDS
    )


def _stage4_filter_hint_text(pipeline) -> str:
    candidates: List[Any] = [getattr(pipeline, "source_file", "")]
    metadata = getattr(pipeline, "_stage4_header_metadata", None)
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("FILTER", ""),
                metadata.get("INSTRUME", ""),
                metadata.get("TELESCOP", ""),
            ]
        )
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        candidates.extend(
            [
                profile.get("filter", ""),
                profile.get("filter_name", ""),
                profile.get("instrument", ""),
            ]
        )
    return " ".join(str(item or "").strip().lower() for item in candidates if str(item or "").strip())


def _stage4_siril_named_arg(name: str, value: str) -> str:
    arg = f"-{name}={str(value or '').strip()}"
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _stage4_debug_command(command: str, args: Tuple[str, ...]) -> str:
    parts = [command]
    parts.extend(str(arg) for arg in args)
    return " ".join(parts)


def _stage4_spcc_bgtol_args(pipeline) -> Tuple[str, ...]:
    bgtol = str(getattr(pipeline.cfg, "stage4_spcc_bgtol", "-2.8,2.0") or "").strip()
    normalized = bgtol.replace(" ", "").replace("+", "")
    if not bgtol or normalized in {"-2.8,2.0", "-2.80,2.00"}:
        return ()
    return (f"-bgtol={bgtol}",)


def _stage4_spcc_limitmag_args(pipeline) -> Tuple[str, ...]:
    limitmag = str(getattr(pipeline.cfg, "stage4_spcc_limitmag", DEFAULT_SPCC_LIMITMAG) or "").strip()
    if not limitmag:
        return ()
    try:
        value = float(limitmag)
    except (TypeError, ValueError):
        pipeline.log.warn(f"忽略无效 SPCC limit magnitude: {limitmag}")
        return ()
    if value <= 0:
        return ()
    return (f"-limitmag={value:g}",)


def _stage4_spcc_restore_cpu(pipeline) -> int:
    raw = str(getattr(pipeline.cfg, "stage4_spcc_restore_cpu", "") or "").strip()
    if not raw:
        raw = str(getattr(pipeline.cfg, "stage4_spcc_restore_maxprocs", "") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pipeline.log.warn(f"忽略无效 SPCC setcpu 恢复值: {raw}")
    return max(1, int(os.cpu_count() or 1))


def _stage4_run_spcc_with_cpu_guard(
    pipeline,
    spcc_args: Tuple[str, ...],
    messages: List[str],
) -> bool:
    restore_cpu = _stage4_spcc_restore_cpu(pipeline)
    messages.append(
        f"SPCC CPU guard: setcpu {SPCC_RUNTIME_CPU} -> restore {restore_cpu}"
    )
    pipeline.cmd_with_check("setcpu", str(SPCC_RUNTIME_CPU), quiet=True)
    try:
        pipeline.cmd_with_check("spcc", *spcc_args)
        return True
    finally:
        try:
            pipeline.cmd_with_check("setcpu", str(restore_cpu), quiet=True)
        except (CommandError, SirilError) as e:
            messages.append(f"SPCC setcpu restore failed: {e}")
            pipeline.log.warn(f"SPCC 后恢复 Siril 线程限制失败: {e}")


def _stage4_pcc_catalogs() -> Tuple[str, ...]:
    raw = os.getenv("SEESTAR_STAGE4_PCC_CATALOGS", ",".join(DEFAULT_PCC_CATALOGS))
    catalogs = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    return catalogs or DEFAULT_PCC_CATALOGS


def _stage4_pcc_variants() -> List[Tuple[str, Tuple[str, ...]]]:
    # Siril's PCC default bgtol is already -2.8,+2.0. Keeping it implicit avoids
    # Siril 1.4.x rejecting explicit negative tuple forms in some CLI paths.
    return [
        (f"catalog:{catalog}", (f"-catalog={catalog}",))
        for catalog in _stage4_pcc_catalogs()
    ]


def _stage4_focal_length() -> float:
    return float(os.getenv("SEESTAR_STAGE4_PLATESOLVE_FOCAL", "160.0"))


def _stage4_pixel_size() -> float:
    return float(os.getenv("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE", "2.90"))


def _stage4_platesolve_order() -> str:
    return str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_ORDER", "3") or "").strip()


def _stage4_platesolve_geometry_args() -> Tuple[str, ...]:
    focal = str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_FOCAL", "160") or "160").strip()
    pixelsize = str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE", "2.90") or "2.90").strip()
    return (f"-focal={focal}", f"-pixelsize={pixelsize}")


def _stage4_platesolve_args() -> Tuple[str, ...]:
    args = _stage4_platesolve_geometry_args()
    order = _stage4_platesolve_order()
    if order:
        args += (f"-order={order}",)
    return args


def _stage4_header_metadata(pipeline) -> Dict[str, Any]:
    if not hasattr(pipeline, "_read_fits_header_metadata"):
        return {}
    candidates = ("stage3_bgremoved", getattr(pipeline, "source_file", None))
    try:
        metadata = pipeline._read_fits_header_metadata(*candidates)
    except TypeError:
        metadata = pipeline._read_fits_header_metadata("stage3_bgremoved")
    except Exception as e:
        pipeline.log.debug(f"Stage4 FITS header metadata unavailable: {e}")
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _stage4_platesolve_catalogs() -> Tuple[str, ...]:
    raw = os.getenv("SEESTAR_STAGE4_PLATESOLVE_CATALOGS", "gaia")
    catalogs = tuple(
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    )
    return catalogs or ("gaia",)


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


def _stage4_platesolve_variants(metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Tuple[str, ...]]]:
    base = _stage4_platesolve_geometry_args()
    order = _stage4_platesolve_order()
    order_args = (f"-order={order}",) if order else ()
    variants: List[Tuple[str, Tuple[str, ...]]] = []
    for catalog in _stage4_platesolve_catalogs():
        variants.append((f"catalog:{catalog}", base + (f"-catalog={catalog}",) + order_args))
    header_base = _stage4_header_platesolve_args(metadata or {})
    if header_base:
        for catalog in _stage4_platesolve_catalogs():
            variants.append((f"header:catalog:{catalog}", header_base + (f"-catalog={catalog}",) + order_args))
    return variants


def _stage4_run_platesolve(pipeline, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[Dict[str, str]]]:
    attempts: List[Dict[str, str]] = []
    original_retries = getattr(pipeline.cfg, "max_retries", None)
    if original_retries is not None:
        pipeline.cfg.max_retries = 0
    try:
        for label, args in _stage4_platesolve_variants(metadata):
            command = "platesolve " + " ".join(args)
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
) -> Tuple[bool, str, List[Dict[str, str]]]:
    attempts: List[Dict[str, str]] = []
    for label, args in _stage4_pcc_variants():
        command = "pcc " + " ".join(args)
        pipeline.log.debug(f"PCC fallback command ({phase}): {command}")
        try:
            pipeline.cmd_with_check("pcc", *args)
            attempts.append(
                {
                    "label": label,
                    "phase": phase,
                    "command": command,
                    "status": "ok",
                }
            )
            return True, command, attempts
        except (CommandError, SirilError) as e:
            attempts.append(
                {
                    "label": label,
                    "phase": phase,
                    "command": command,
                    "status": "failed",
                    "error": str(e),
                }
            )
            pipeline.log.warn(f"PCC 候选失败 ({phase}, {label}): {e}")
    last_error = attempts[-1].get("error", "unknown error") if attempts else "no attempts"
    return False, f"PCC {phase} failed: {last_error}", attempts


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
    except Exception:
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


def _stage4_profile_metric(pipeline, key: str) -> float:
    profile = getattr(pipeline, "target_profile", None)
    if not isinstance(profile, dict):
        return 0.0
    for section_name in ("object_stats", "image_stats", "color_stats", "star_stats"):
        section = profile.get(section_name)
        if isinstance(section, dict) and key in section:
            try:
                return float(section.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    if key in profile:
        try:
            return float(profile.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _stage4_background_sampling_window(pipeline=None) -> Dict[str, Any]:
    object_area = _stage4_profile_metric(pipeline, "object_area_ratio") if pipeline is not None else 0.0
    nebulosity_area = _stage4_profile_metric(pipeline, "nebulosity_area_ratio") if pipeline is not None else 0.0
    target_aware = False
    if pipeline is not None:
        try:
            target_aware = _stage4_target_aware_color_mapping(pipeline)
        except AttributeError:
            target_aware = _stage4_active_target_type(pipeline) in EMISSION_NEBULA_TARGET_TYPES
    effective_area = max(object_area, nebulosity_area if target_aware else 0.0)
    lo_q = 0.05
    if effective_area >= 0.35:
        hi_q = 0.25
        mode = "large_target_q5_q25"
    elif effective_area >= 0.20:
        t = min(1.0, max(0.0, (effective_area - 0.20) / 0.15))
        hi_q = 0.35 - 0.10 * t
        mode = "target_area_adaptive"
    elif effective_area >= 0.10:
        t = min(1.0, max(0.0, (effective_area - 0.10) / 0.10))
        hi_q = 0.45 - 0.10 * t
        mode = "target_area_adaptive"
    else:
        hi_q = 0.45
        mode = "default_q5_q45"
    return {
        "mode": mode,
        "lower_quantile": lo_q,
        "upper_quantile": float(hi_q),
        "object_area_ratio": float(object_area),
        "nebulosity_area_ratio": float(nebulosity_area),
        "effective_area_ratio": float(effective_area),
        "target_aware": bool(target_aware),
    }


def _stage4_background_neutralize(chw: np.ndarray, pipeline=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    output = np.nan_to_num(chw.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    if output.shape[0] < 3:
        return np.clip(output, 0.0, None), {"applied": False, "reason": "mono image"}

    rgb = output[:3]
    lum = _stage4_luminance(rgb)
    finite_mask = np.isfinite(lum)
    if int(np.count_nonzero(finite_mask)) < 64:
        raise ValueError("not enough finite pixels for background neutralization")

    sampling = _stage4_background_sampling_window(pipeline)
    valid_lum = lum[finite_mask]
    lo_q = float(sampling["lower_quantile"])
    hi_q = float(sampling["upper_quantile"])
    lo = float(np.quantile(valid_lum, lo_q))
    hi = float(np.quantile(valid_lum, hi_q))
    bg_mask = finite_mask & (lum >= lo) & (lum <= hi)
    if int(np.count_nonzero(bg_mask)) < 256:
        if float(sampling.get("effective_area_ratio", 0.0) or 0.0) >= 0.35:
            fallback_hi_q = min(0.35, max(hi_q + 0.10, 0.30))
        else:
            fallback_hi_q = min(0.60, max(hi_q + 0.15, 0.45))
        hi = float(np.quantile(valid_lum, fallback_hi_q))
        bg_mask = finite_mask & (lum >= lo) & (lum <= hi)
        sampling["fallback_upper_quantile"] = float(fallback_hi_q)
    if int(np.count_nonzero(bg_mask)) < 64:
        bg_mask = finite_mask
        sampling["fallback"] = "all_finite_pixels"

    medians = np.array([float(np.median(channel[bg_mask])) for channel in rgb], dtype=np.float32)
    target = float(np.median(medians))
    offsets = target - medians
    output[:3] = np.clip(rgb + offsets[:, np.newaxis, np.newaxis], 0.0, None)
    return output, {
        "applied": True,
        "sample_pixels": int(np.count_nonzero(bg_mask)),
        "sampling_window": sampling,
        "channel_medians_before": [float(v) for v in medians],
        "target_median": target,
        "offsets": [float(v) for v in offsets],
    }


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
    gain_limit = float(getattr(pipeline.cfg, "stage4_local_star_wb_gain_limit", 1.25) or 1.25)
    gain_limit = max(1.01, min(gain_limit, 1.50))
    gains = np.clip(target / medians, 1.0 / gain_limit, gain_limit)
    output[:3] = np.clip(rgb * gains[:, np.newaxis, np.newaxis], 0.0, None)
    return output, {
        "applied": True,
        "white_reference_pixels": star_pixels,
        "channel_medians_before": [float(v) for v in medians],
        "target_median": target,
        "gains": [float(v) for v in gains],
        "gain_limit": gain_limit,
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


def _stage4_local_color_fallback(pipeline, *, target_aware: bool = False) -> Tuple[bool, str, str, float, Dict[str, Any], str]:
    image_data = pipeline.siril.get_image_pixeldata(preview=False)
    chw, restore = _stage4_image_as_chw(image_data)
    neutralized, bg_report = _stage4_background_neutralize(chw, pipeline)

    wb_enabled = bool(getattr(pipeline.cfg, "stage4_local_star_wb_enabled", True))
    star_report: Dict[str, Any] = {"applied": False, "reason": "disabled", "white_reference_pixels": 0}
    balanced = neutralized
    target_aware_star_wb_enabled = bool(
        getattr(pipeline.cfg, "stage4_local_star_wb_target_aware_enabled", False)
    )
    if wb_enabled and target_aware and not target_aware_star_wb_enabled:
        star_report = {
            "applied": False,
            "reason": "target-aware emission/dualband color preservation",
            "white_reference_pixels": 0,
            "skipped": True,
        }
    elif wb_enabled:
        balanced, star_report = _stage4_star_white_balance(neutralized, pipeline)

    restored = restore(balanced)
    _stage4_write_image_pixels(pipeline, restored)

    report = {
        "target_aware": bool(target_aware),
        "background_neutralization": bg_report,
        "star_white_balance": star_report,
    }
    if star_report.get("applied"):
        message = "local background neutralization + star white balance fallback ok"
        if target_aware:
            message += " (target-aware color mapping)"
        return True, "LOCAL_STAR_WB", "local_star_white_balance_fallback", 0.58, report, message

    if target_aware and star_report.get("skipped"):
        message = (
            "local background neutralization fallback ok; "
            "target-aware star white balance skipped to preserve emission colors"
        )
        return (
            True,
            "BACKGROUND_NEUTRALIZATION",
            "target_aware_background_neutralization_only",
            0.42,
            report,
            message,
        )

    message = "local background neutralization fallback ok; star samples insufficient"
    if target_aware:
        message += " (target-aware color mapping)"
    return True, "BACKGROUND_NEUTRALIZATION", "background_neutralization_only", 0.35, report, message


def run_stage4_color_calibration(pipeline) -> None:
    """
    阶段 4: 图像解析 + 色彩校准
    - 显式从 stage3_bgremoved 进入
    - platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3 后保存 stage4_psolved
    - SPCC 指定 Sony IMX585 OSC 参数，按 LP / No filter / narrowband 三类参数分支执行，失败后 PCC，最后本地背景中性化/星点白平衡回退
    """
    stage_label = "阶段 4: 图像解析 + 色彩校准"
    pipeline.log.stage_start(stage_label)
    status = "ok"
    hard_degraded = False
    color_ok = False
    color_method = "none"
    color_warning = ""
    color_confidence = 0.0
    messages: List[str] = []
    spcc_white_ref = DEFAULT_SPCC_WHITE_REF
    spcc_white_ref_reason = "default"
    spcc_white_ref_fallback = None
    local_fallback_report: Optional[Dict[str, Any]] = None
    pcc_attempts: List[Dict[str, str]] = []
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}

    try:
        pipeline.cmd_with_check("load", "stage3_bgremoved")
        messages.append("input=stage3_bgremoved")
    except (CommandError, SirilError) as e:
        status = "degraded"
        hard_degraded = True
        messages.append(f"stage3_bgremoved load failed; using current image: {e}")
        pipeline.log.warn(f"stage4 load stage3_bgremoved failed, using current image: {e}")

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

    spcc_runtime_allowed = bool(getattr(pipeline.cfg, "spcc_enabled", True))
    target_aware_color = _stage4_target_aware_color_mapping(pipeline)
    spcc_white_ref, spcc_white_ref_reason = _stage4_effective_spcc_white_ref(pipeline)
    header_center_coordinates = _stage4_header_center_coordinates(stage4_metadata)
    pcc_header_fallback_allowed = bool(
        platesolve_attempted
        and not pipeline.platesolve_ok
        and header_center_coordinates
        and bool(getattr(pipeline.cfg, "stage4_pcc_header_fallback_enabled", True))
    )
    allow_light_spcc = (
        os.getenv("SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS", "1").strip().lower()
        in ENV_TRUE_VALUES
    )
    if (
        spcc_runtime_allowed
        and getattr(pipeline, "_stage1_input_mode", "") == "light_preprocess"
        and not allow_light_spcc
    ):
        spcc_runtime_allowed = False
        messages.append("SPCC skipped on Light_ preprocess mode to avoid siril-cli crash risk")

    photometric_calibration_allowed = bool(pipeline.platesolve_ok)
    if not photometric_calibration_allowed:
        spcc_runtime_allowed = False
        if pcc_header_fallback_allowed:
            messages.append(
                "SPCC skipped: plate solve unavailable; trying PCC header-coordinate fallback"
            )
        else:
            messages.append("SPCC/PCC skipped: plate solve unavailable")
        pipeline.log.warn("Stage4 photometric calibration skipped because image is not plate-solved")

    pipeline.log.debug(
        "Stage4 runtime: "
        f"platesolve_enabled={bool(getattr(pipeline.cfg, 'stage4_platesolve_enabled', True))}, "
        f"platesolve_ok={bool(pipeline.platesolve_ok)}, "
        f"spcc_enabled={bool(getattr(pipeline.cfg, 'spcc_enabled', True))}, "
        f"spcc_runtime_allowed={bool(spcc_runtime_allowed)}, "
        f"photometric_calibration_allowed={bool(photometric_calibration_allowed)}, "
        f"allow_light_spcc={bool(allow_light_spcc)}, "
        f"input_mode={getattr(pipeline, '_stage1_input_mode', '') or 'unknown'}, "
        f"target_aware_color={bool(target_aware_color)}"
    )

    if spcc_runtime_allowed:
        try:
            spcc_args, spcc_messages = _stage4_spcc_args(pipeline, whiteref=spcc_white_ref)
            messages.extend(spcc_messages)
            messages.append(
                f"SPCC whiteref={spcc_white_ref} ({spcc_white_ref_reason})"
            )
            pipeline.log.debug(
                "SPCC selected white reference: "
                f"{spcc_white_ref} (reason={spcc_white_ref_reason})"
            )
            pipeline.log.debug(
                "SPCC command: "
                + _stage4_debug_command("spcc", spcc_args)
            )
            _stage4_run_spcc_with_cpu_guard(pipeline, spcc_args, messages)
            color_ok = True
            color_method = "SPCC"
            color_confidence = 0.90
            messages.append("SPCC ok")
            pipeline.log.info("分光光度色彩校准成功 (SPCC)")
        except (CommandError, SirilError) as e:
            primary_error = e
            if (
                spcc_white_ref != DEFAULT_SPCC_WHITE_REF
                and spcc_white_ref_reason != "explicit_config"
                and not target_aware_color
            ):
                try:
                    spcc_white_ref_fallback = DEFAULT_SPCC_WHITE_REF
                    messages.append(
                        "SPCC adaptive whiteref failed; retrying "
                        f"{DEFAULT_SPCC_WHITE_REF}: {e}"
                    )
                    fallback_args, fallback_messages = _stage4_spcc_args(
                        pipeline,
                        whiteref=DEFAULT_SPCC_WHITE_REF,
                    )
                    messages.extend(fallback_messages)
                    pipeline.log.debug(
                        "SPCC fallback command: "
                        + _stage4_debug_command("spcc", fallback_args)
                    )
                    _stage4_run_spcc_with_cpu_guard(
                        pipeline,
                        fallback_args,
                        messages,
                    )
                    color_ok = True
                    color_method = "SPCC"
                    color_confidence = 0.86
                    messages.append("SPCC ok with default whiteref fallback")
                    pipeline.log.info(
                        "分光光度色彩校准成功 (SPCC, default whiteref fallback)"
                    )
                except (CommandError, SirilError) as fallback_error:
                    messages.append(
                        f"SPCC failed: {primary_error}; default whiteref retry failed: {fallback_error}"
                    )
                    pipeline.log.warn(
                        f"SPCC 失败: {primary_error}; default whiteref retry failed: {fallback_error}"
                    )
            elif target_aware_color and spcc_white_ref != DEFAULT_SPCC_WHITE_REF:
                messages.append(
                    f"SPCC failed: {e}; ordinary galaxy white reference fallback disabled for target-aware color mapping"
                )
                pipeline.log.warn(
                    f"SPCC 失败: {e}; target-aware 目标禁止回退到普通星系白参考"
                )
            else:
                messages.append(f"SPCC failed: {e}")
                pipeline.log.warn(f"SPCC 失败: {e}")
    elif photometric_calibration_allowed:
        messages.append("SPCC disabled; using PCC fallback")
        pipeline.log.warn("SPCC 已禁用，尝试 PCC")

    if not color_ok and photometric_calibration_allowed:
        pcc_ok, pcc_result, pcc_attempts = _stage4_run_pcc(
            pipeline,
            phase="plate_solved",
        )
        if pcc_ok:
            color_ok = True
            color_method = "PCC"
            color_confidence = 0.72
            messages.append(f"{pcc_result} ok (default bgtol)")
            pipeline.log.info("PCC 色彩校准成功")
        else:
            messages.append(pcc_result)
            pipeline.log.warn(f"PCC 失败: {pcc_result}")
    elif not color_ok and pcc_header_fallback_allowed:
        pcc_ok, pcc_result, pcc_attempts = _stage4_run_pcc(
            pipeline,
            phase="header_metadata",
        )
        if pcc_ok:
            color_ok = True
            color_method = "PCC_HEADER"
            color_warning = "pcc_header_coordinate_fallback"
            color_confidence = 0.64
            messages.append(f"{pcc_result} ok using FITS header coordinates")
            pipeline.log.info("PCC header-coordinate 色彩校准成功")
        else:
            messages.append(pcc_result)
            pipeline.log.warn(f"PCC header-coordinate fallback 失败: {pcc_result}")

    if not color_ok:
        try:
            (
                color_ok,
                color_method,
                color_warning,
                color_confidence,
                local_fallback_report,
                fallback_message,
            ) = _stage4_local_color_fallback(pipeline, target_aware=target_aware_color)
            messages.append(fallback_message)
            pipeline.log.info(fallback_message)
            pipeline.log.debug(f"local color fallback report: {local_fallback_report}")
        except Exception as e:
            color_ok = False
            status = "degraded"
            messages.append(f"local color fallback failed: {e}")
            pipeline.log.warn(f"本地色彩回退失败: {e}")

    if not color_ok:
        status = "degraded"
        color_method = "none"
        color_warning = color_warning or "color_calibration_failed"
    elif color_method == "BACKGROUND_NEUTRALIZATION":
        status = "degraded"
    elif not hard_degraded:
        status = "ok"

    if (
        color_warning
        and stage4_policy.get("reduce_saturation_if_solution_imprecise", False)
    ):
        messages.append("color policy limits later saturation/color gains due to imprecise solution")
    if color_method in {"LOCAL_STAR_WB", "BACKGROUND_NEUTRALIZATION"} and platesolve_attempted and not pipeline.platesolve_ok:
        messages.append("platesolve 失败，已使用本地背景中性化/星点白平衡回退")

    color_saved = pipeline._save_stage_output("stage4_color")
    if not color_saved:
        status = "degraded"
        hard_degraded = True
        messages.append("stage4_color 输出保存失败")
    elif color_ok and color_method != "BACKGROUND_NEUTRALIZATION" and not hard_degraded:
        status = "ok"

    spcc_osc_mode_report = _stage4_effective_osc_spcc_mode(pipeline)
    pipeline.color_calibration_report = {
        "stage": "stage4_color",
        "input": "stage3_bgremoved",
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
        "photometric_calibration_allowed": bool(photometric_calibration_allowed),
        "pcc": {
            "catalogs": list(_stage4_pcc_catalogs()),
            "attempts": pcc_attempts,
            "header_coordinate_fallback_allowed": bool(pcc_header_fallback_allowed),
            "header_center_coordinates": header_center_coordinates,
        },
        "spcc": {
            "sensor_mode": str(getattr(pipeline.cfg, "stage4_spcc_sensor_mode", "osc") or "osc"),
            "osc_sensor": str(getattr(pipeline.cfg, "stage4_spcc_osc_sensor", DEFAULT_OSC_SENSOR) or DEFAULT_OSC_SENSOR),
            "osc_filter": spcc_osc_mode_report["osc_filter"],
            "osc_filter_reason": spcc_osc_mode_report["reason"],
            "narrowband": bool(spcc_osc_mode_report["narrowband"]),
            "narrowband_args": spcc_osc_mode_report["narrowband_args"],
            "limitmag": str(getattr(pipeline.cfg, "stage4_spcc_limitmag", DEFAULT_SPCC_LIMITMAG) or ""),
            "bgtol": str(getattr(pipeline.cfg, "stage4_spcc_bgtol", "-2.8,2.0") or ""),
            "cpu_guard": {
                "runtime": SPCC_RUNTIME_CPU,
                "restore": _stage4_spcc_restore_cpu(pipeline),
            },
        },
        "spcc_white_reference": {
            "requested": spcc_white_ref,
            "reason": spcc_white_ref_reason,
            "fallback": spcc_white_ref_fallback,
            "ordinary_galaxy_fallback_allowed": not bool(target_aware_color),
        },
        "local_fallback": local_fallback_report,
        "status": (
            "success_with_warning"
            if color_ok and color_warning
            else ("success" if color_ok else "degraded")
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
            "color": "stage4_color.fit",
            "compat_color": "stage4_colorbalanced.fit",
        },
        "messages": messages,
    }
    pipeline._write_stage_json("color_calibration_report.json", pipeline.color_calibration_report)

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(stage_label, status, elapsed, "；".join(messages))
