"""Stage 4 plate solving and color calibration."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from models import PipelineStage
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
SPCC_CATALOG = "localgaia"
SPCC_RUNTIME_CPU = 1
SPCC_IMPRECISE_CONFIDENCE = 0.45
SPCC_IMPRECISE_PCC_RECOVERY_CONFIDENCE = 0.62
SPCC_IMPRECISE_LOG_MARKERS = (
    "the photometric color calibration seems to have found an imprecise solution",
    "测光法色彩校准似乎不能精确校准",
)
SPCC_DATABASE_ENV = "SEESTAR_SPCC_DATABASE_DIR"
SPCC_DATABASE_RELATIVE_PATH = Path(
    "Library/Application Support/org.siril.Siril/siril-spcc-database"
)
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
LOCAL_SPCC_DIRNAME = "siril_cat1_healpix8_xpsamp"
LOCAL_SPCC_FILE_PATTERN = "siril_cat1_healpix8_xpsamp_*.dat"
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


def _stage4_network_enabled() -> bool:
    return (
        os.getenv("SEESTAR_NETWORK_MODE", "1").strip().lower()
        in ENV_TRUE_VALUES
    )


def _stage4_local_spcc_catalog_dir(pipeline) -> Path:
    configured = (
        getattr(pipeline, "local_gaia_photo_catalog", None)
        or os.getenv("SEESTAR_GAIA_PHOTO_CATALOG", "")
    )
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "siril" / LOCAL_SPCC_DIRNAME


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


def _stage4_spcc_database_dir(pipeline) -> Path:
    configured = (
        getattr(pipeline, "spcc_database_dir", None)
        or os.getenv(SPCC_DATABASE_ENV, "")
    )
    if configured:
        return Path(configured).expanduser()
    return Path.home() / SPCC_DATABASE_RELATIVE_PATH


def _stage4_spcc_valid_response_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    wavelength = entry.get("wavelength")
    values = entry.get("values")
    if not isinstance(wavelength, dict) or not isinstance(values, dict):
        return False
    wavelength_values = wavelength.get("value")
    response_values = values.get("value")
    if (
        not isinstance(wavelength_values, list)
        or not isinstance(response_values, list)
        or not wavelength_values
        or len(wavelength_values) != len(response_values)
    ):
        return False
    numeric_values = wavelength_values + response_values
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in numeric_values
    )


def _stage4_spcc_required_metadata(
    pipeline,
    whiteref: str,
) -> List[Dict[str, Any]]:
    mode = str(
        getattr(pipeline.cfg, "stage4_spcc_sensor_mode", "osc") or "osc"
    ).lower()
    requirements: List[Dict[str, Any]] = []
    if mode in {"mono", "mono_lrgb", "lrgb"}:
        requirements.append(
            {
                "kind": "mono_sensor",
                "value": str(
                    getattr(pipeline.cfg, "stage4_spcc_mono_sensor", "") or ""
                ),
            }
        )
        requirements.extend(
            {
                "kind": "mono_filter",
                "channel": channel,
                "required_channels": [channel],
                "value": str(getattr(pipeline.cfg, attr, "") or ""),
            }
            for channel, attr in (
                ("RED", "stage4_spcc_r_filter"),
                ("GREEN", "stage4_spcc_g_filter"),
                ("BLUE", "stage4_spcc_b_filter"),
            )
        )
    else:
        requirements.append(
            {
                "kind": "osc_sensor",
                "value": str(
                    getattr(
                        pipeline.cfg,
                        "stage4_spcc_osc_sensor",
                        DEFAULT_OSC_SENSOR,
                    )
                    or DEFAULT_OSC_SENSOR
                ),
                "required_channels": ["RED", "GREEN", "BLUE"],
            }
        )
        osc_mode = _stage4_effective_osc_spcc_mode(pipeline)
        if not bool(osc_mode["narrowband"]):
            requirements.append(
                {
                    "kind": "osc_filter",
                    "value": str(osc_mode["osc_filter"]),
                }
            )
    requirements.append({"kind": "white_reference", "value": str(whiteref)})
    return requirements


def _stage4_spcc_metadata_not_checked(pipeline) -> Dict[str, Any]:
    return {
        "path": str(_stage4_spcc_database_dir(pipeline)),
        "checked": False,
        "available": False,
        "reason": "not_checked",
        "requirements": [],
        "missing": [],
        "valid_json_files": [],
        "invalid_json_files": [],
        "empty_json_files": [],
        "invalid_entry_files": [],
    }


def _stage4_spcc_metadata_status(
    pipeline,
    *,
    whiteref: str,
) -> Dict[str, Any]:
    root = _stage4_spcc_database_dir(pipeline)
    result = _stage4_spcc_metadata_not_checked(pipeline)
    result["checked"] = True
    if not root.is_dir():
        result["reason"] = "database_directory_missing"
        return result

    category_types = {
        "osc_sensors": "OSC_SENSOR",
        "mono_sensors": "MONO_SENSOR",
        "osc_filters": "OSC_FILTER",
        "mono_filters": "MONO_FILTER",
        "wb_refs": "WB_REF",
    }
    index: Dict[str, Dict[str, Dict[str, set[str]]]] = {
        "osc_sensor": {},
        "mono_sensor": {},
        "osc_filter": {},
        "mono_filter": {},
        "white_reference": {},
    }
    json_paths: List[Tuple[Path, str]] = []
    for directory_name, expected_type in category_types.items():
        directory = root / directory_name
        if directory.is_dir():
            json_paths.extend(
                (path, expected_type) for path in sorted(directory.rglob("*.json"))
            )

    if not json_paths:
        result["reason"] = "no_metadata_json_files"
        return result

    for path, expected_type in json_paths:
        relative_path = str(path.relative_to(root))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            result["invalid_json_files"].append(
                {"path": relative_path, "error": str(error)}
            )
            continue
        if not isinstance(payload, list):
            result["invalid_json_files"].append(
                {"path": relative_path, "error": "top_level_not_array"}
            )
            continue
        if not payload:
            result["empty_json_files"].append(relative_path)
            continue

        file_errors: List[str] = []
        for position, entry in enumerate(payload):
            if not _stage4_spcc_valid_response_entry(entry):
                file_errors.append(f"entry[{position}]: invalid_or_empty_response_arrays")
                continue
            entry_type = str(entry.get("type") or "").upper()
            if entry_type != expected_type:
                file_errors.append(
                    f"entry[{position}]: expected_type={expected_type}, actual_type={entry_type or 'missing'}"
                )
                continue

            if entry_type == "OSC_SENSOR":
                key = str(entry.get("model") or "")
                kind = "osc_sensor"
            elif entry_type == "MONO_SENSOR":
                key = str(entry.get("name") or entry.get("model") or "")
                kind = "mono_sensor"
            elif entry_type == "OSC_FILTER":
                key = str(entry.get("name") or "")
                kind = "osc_filter"
            elif entry_type == "MONO_FILTER":
                key = str(entry.get("name") or "")
                kind = "mono_filter"
            else:
                key = str(entry.get("name") or entry.get("model") or "")
                kind = "white_reference"
            if not key:
                file_errors.append(f"entry[{position}]: missing_list_name")
                continue

            matched = index[kind].setdefault(
                key,
                {"files": set(), "channels": set()},
            )
            matched["files"].add(relative_path)
            channel = str(entry.get("channel") or "").upper()
            if channel:
                matched["channels"].add(channel)

        if file_errors:
            result["invalid_entry_files"].append(
                {"path": relative_path, "errors": file_errors}
            )
        else:
            result["valid_json_files"].append(relative_path)

    requirements = _stage4_spcc_required_metadata(pipeline, whiteref)
    missing: List[str] = []
    requirement_results: List[Dict[str, Any]] = []
    for requirement in requirements:
        kind = str(requirement["kind"])
        value = str(requirement.get("value") or "")
        matched = index[kind].get(value)
        required_channels = set(requirement.get("required_channels") or [])
        actual_channels = set(matched["channels"]) if matched else set()
        found = bool(value and matched and required_channels.issubset(actual_channels))
        requirement_result = dict(requirement)
        requirement_result.update(
            {
                "found": found,
                "files": sorted(matched["files"]) if matched else [],
                "channels": sorted(actual_channels),
            }
        )
        requirement_results.append(requirement_result)
        if not found:
            label = f"{kind}={value or '<empty>'}"
            if required_channels and matched:
                label += ":missing_channels=" + ",".join(
                    sorted(required_channels - actual_channels)
                )
            missing.append(label)

    result["requirements"] = requirement_results
    result["missing"] = missing
    result["available_counts"] = {
        kind: len(entries) for kind, entries in index.items()
    }
    if result["invalid_json_files"]:
        result["reason"] = "invalid_metadata_json"
    elif result["empty_json_files"]:
        result["reason"] = "empty_metadata_json_array"
    elif result["invalid_entry_files"]:
        result["reason"] = "invalid_metadata_response_arrays"
    elif missing:
        result["reason"] = "required_metadata_missing"
    elif not result["valid_json_files"]:
        result["reason"] = "no_valid_metadata_json"
    else:
        result["available"] = True
        result["reason"] = "ok"
    return result


def _stage4_spcc_args(pipeline, *, whiteref: Optional[str] = None) -> Tuple[Tuple[str, ...], List[str]]:
    mode = str(getattr(pipeline.cfg, "stage4_spcc_sensor_mode", "osc") or "osc").lower()
    whiteref = str(whiteref or DEFAULT_SPCC_WHITE_REF)
    messages: List[str] = []

    if mode in {"mono", "mono_lrgb", "lrgb"}:
        pipeline.cmd_with_check("spcc_list", "whiteref")
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
                f"-catalog={SPCC_CATALOG}",
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

    pipeline.cmd_with_check("spcc_list", "whiteref")
    pipeline.cmd_with_check("spcc_list", "oscsensor")
    sensor = str(getattr(pipeline.cfg, "stage4_spcc_osc_sensor", DEFAULT_OSC_SENSOR) or DEFAULT_OSC_SENSOR)
    osc_mode = _stage4_effective_osc_spcc_mode(pipeline)
    if bool(osc_mode["narrowband"]):
        messages.append(f"OSC SPCC mode=narrowband ({osc_mode['reason']})")
        return (
            (
                f"-catalog={SPCC_CATALOG}",
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
            f"-catalog={SPCC_CATALOG}",
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


def _stage4_read_siril_log(pipeline) -> Optional[str]:
    getter = getattr(pipeline.siril, "get_siril_log", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except (
        AttributeError,
        CommandError,
        SirilError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        pipeline.log.debug(f"Stage4 Siril log snapshot unavailable: {error}")
        return None
    return str(value) if value is not None else None


def _stage4_spcc_solution_quality(
    before_log: Optional[str],
    after_log: Optional[str],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "status": "not_checked",
        "imprecise": False,
        "warning_code": None,
        "matched_messages": [],
        "log_delta_chars": 0,
        "confidence": None,
        "recommended_action": None,
        "fallback_triggered": False,
        "fallback_source": None,
        "fallback_source_restored": None,
    }
    if before_log is None or after_log is None:
        report["reason"] = "Siril log snapshots unavailable"
        return report

    log_continuity = "prefix"
    if after_log.startswith(before_log):
        log_delta = after_log[len(before_log):]
    else:
        # Siril may trim the oldest visible log lines. Recover only a proven
        # suffix/prefix overlap so an old SPCC warning cannot become a false hit.
        before_lines = before_log.splitlines(keepends=True)[-512:]
        after_lines = after_log.splitlines(keepends=True)
        overlap = 0
        for count in range(min(len(before_lines), len(after_lines)), 0, -1):
            if before_lines[-count:] == after_lines[:count]:
                overlap = count
                break
        if overlap <= 0:
            report["reason"] = "Siril log continuity unavailable"
            return report
        log_continuity = "trimmed_prefix_overlap"
        log_delta = "".join(after_lines[overlap:])
    report["log_continuity"] = log_continuity
    report["log_delta_chars"] = len(log_delta)
    matched = [
        line.strip()
        for line in log_delta.splitlines()
        if any(marker in line.lower() for marker in SPCC_IMPRECISE_LOG_MARKERS)
    ]
    if matched:
        report.update(
            {
                "status": "imprecise",
                "imprecise": True,
                "warning_code": "spcc_imprecise_solution",
                "matched_messages": matched,
                "confidence": SPCC_IMPRECISE_CONFIDENCE,
                "recommended_action": (
                    "correct_gradient_then_retry_spcc_or_use_backup_color_calibration"
                ),
            }
        )
        return report

    report.update(
        {
            "status": "accepted",
            "reason": "no imprecise-solution warning in SPCC log delta",
        }
    )
    return report


def _stage4_finalize_spcc_solution(
    pipeline,
    solution_quality: Dict[str, Any],
    messages: List[str],
    *,
    success_confidence: float,
) -> Tuple[bool, str, float, str, bool]:
    """Accept SPCC or restore the plate-solved source for backup calibration."""
    if not bool(solution_quality.get("imprecise", False)):
        solution_quality["confidence"] = float(success_confidence)
        return True, "SPCC", float(success_confidence), "", False

    solution_quality["fallback_triggered"] = True
    solution_quality["fallback_source"] = "stage4_psolved"
    messages.append(
        "SPCC imprecise solution detected from Siril log; confidence reduced "
        f"to {SPCC_IMPRECISE_CONFIDENCE:.2f}; restoring stage4_psolved for PCC fallback"
    )
    pipeline.log.warn(
        "SPCC 返回不精确解；降低置信度并回载 stage4_psolved 进入 PCC 备用校色"
    )
    try:
        pipeline.cmd_with_check("load", "stage4_psolved")
        solution_quality["fallback_source_restored"] = True
        return (
            False,
            "none",
            SPCC_IMPRECISE_CONFIDENCE,
            "spcc_imprecise_solution",
            False,
        )
    except (CommandError, SirilError) as error:
        solution_quality["fallback_source_restored"] = False
        solution_quality["fallback_error"] = str(error)
        messages.append(
            "SPCC imprecise fallback source restore failed; retaining provisional "
            f"SPCC result: {error}"
        )
        pipeline.log.warn(f"SPCC 不精确解回滚失败，保留低置信度结果: {error}")
        return (
            True,
            "SPCC_IMPRECISE",
            SPCC_IMPRECISE_CONFIDENCE,
            "spcc_imprecise_solution_restore_failed",
            True,
        )


def _stage4_run_spcc_with_cpu_guard(
    pipeline,
    spcc_args: Tuple[str, ...],
    messages: List[str],
) -> Dict[str, Any]:
    restore_cpu = _stage4_spcc_restore_cpu(pipeline)
    messages.append(
        f"SPCC CPU guard: setcpu {SPCC_RUNTIME_CPU} -> restore {restore_cpu}"
    )
    before_log = _stage4_read_siril_log(pipeline)
    pipeline.cmd_with_check("setcpu", str(SPCC_RUNTIME_CPU), quiet=True)
    try:
        pipeline.cmd_with_check("spcc", *spcc_args)
        after_log = _stage4_read_siril_log(pipeline)
        return _stage4_spcc_solution_quality(before_log, after_log)
    finally:
        if bool(getattr(pipeline, "_siril_process_terminated", False)):
            messages.append(
                "SPCC CPU restore skipped: Siril native process terminated"
            )
        else:
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


def _stage4_angle_degrees(value: Any, *, right_ascension: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip().strip("'\"")
    if not text:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        pass

    parts = [item for item in re.split(r"[:\s]+", text) if item]
    if len(parts) < 2:
        return None
    try:
        first = float(parts[0])
        minute = abs(float(parts[1]))
        second = abs(float(parts[2])) if len(parts) > 2 else 0.0
    except ValueError:
        return None
    sign = -1.0 if first < 0 or text.startswith("-") else 1.0
    degrees = abs(first) + minute / 60.0 + second / 3600.0
    if right_ascension:
        degrees *= 15.0
    return sign * degrees


def _stage4_center_degrees(metadata: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not isinstance(metadata, dict):
        return None
    for ra_key, dec_key, ra_is_hours in (
        ("CRVAL1", "CRVAL2", False),
        ("RA", "DEC", True),
        ("OBJCTRA", "OBJCTDEC", True),
    ):
        ra = _stage4_angle_degrees(
            metadata.get(ra_key),
            right_ascension=ra_is_hours and isinstance(metadata.get(ra_key), str),
        )
        dec = _stage4_angle_degrees(
            metadata.get(dec_key),
            right_ascension=False,
        )
        if ra is None or dec is None or not -90.0 <= dec <= 90.0:
            continue
        return ra % 360.0, dec
    return None


def _stage4_healpix_level1_pixel(ra_deg: float, dec_deg: float) -> int:
    """Return nested HEALPix nside=2 (level-1) pixel without extra runtime deps."""
    nside = 2
    z = math.sin(math.radians(dec_deg))
    za = abs(z)
    tt = (math.radians(ra_deg) % (2.0 * math.pi)) / (0.5 * math.pi)

    if za <= 2.0 / 3.0:
        temp1 = nside * (0.5 + tt)
        temp2 = nside * (z * 0.75)
        jp = int(temp1 - temp2)
        jm = int(temp1 + temp2)
        ifp = jp // nside
        ifm = jm // nside
        if ifp == ifm:
            face = (ifp % 4) + 4
        elif ifp < ifm:
            face = ifp % 4
        else:
            face = (ifm % 4) + 8
        ix = jm % nside
        iy = nside - (jp % nside) - 1
    else:
        ntt = int(tt)
        tp = tt - ntt
        tmp = nside * math.sqrt(3.0 * (1.0 - za))
        jp = min(nside - 1, int(tp * tmp))
        jm = min(nside - 1, int((1.0 - tp) * tmp))
        if z >= 0.0:
            face = ntt
            ix = nside - jm - 1
            iy = nside - jp - 1
        else:
            face = ntt + 8
            ix = jp
            iy = jm

    pixel_in_face = (ix & 1) | ((iy & 1) << 1)
    return face * nside * nside + pixel_in_face


def _stage4_destination_point(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    bearing_deg: float,
) -> Tuple[float, float]:
    lon1 = math.radians(ra_deg)
    lat1 = math.radians(dec_deg)
    distance = math.radians(radius_deg)
    bearing = math.radians(bearing_deg)
    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_distance = math.sin(distance)
    cos_distance = math.cos(distance)
    lat2 = math.asin(
        sin_lat1 * cos_distance
        + cos_lat1 * sin_distance * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * sin_distance * cos_lat1,
        cos_distance - sin_lat1 * math.sin(lat2),
    )
    return math.degrees(lon2) % 360.0, math.degrees(lat2)


def _stage4_required_spcc_pixels(
    metadata: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Tuple[Optional[Tuple[float, float]], List[int], float]:
    center = _stage4_center_degrees(metadata)
    if center is None:
        return None, [], 0.0
    fov = geometry.get("cropped_fov_deg", {}) if isinstance(geometry, dict) else {}
    width = float((fov or {}).get("width") or 0.0)
    height = float((fov or {}).get("height") or 0.0)
    radius = math.hypot(width, height) if width > 0.0 and height > 0.0 else 5.0
    radius = max(0.1, min(30.0, radius))

    pixels = {_stage4_healpix_level1_pixel(*center)}
    for bearing in range(0, 360, 5):
        pixels.add(
            _stage4_healpix_level1_pixel(
                *_stage4_destination_point(*center, radius, float(bearing))
            )
        )
    return center, sorted(pixels), radius


def _stage4_local_spcc_catalog_status(
    pipeline,
    metadata: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    catalog_dir = _stage4_local_spcc_catalog_dir(pipeline)
    valid_chunks: Dict[int, Dict[str, Any]] = {}
    if catalog_dir.is_dir():
        try:
            candidates = catalog_dir.glob(LOCAL_SPCC_FILE_PATTERN)
            for path in candidates:
                match = re.fullmatch(
                    r"siril_cat1_healpix8_xpsamp_(\d+)\.dat",
                    path.name,
                )
                if match is None or not _stage4_valid_catalog_file(path):
                    continue
                valid_chunks[int(match.group(1))] = {
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                }
        except OSError:
            valid_chunks = {}

    center, required_pixels, query_radius = _stage4_required_spcc_pixels(
        metadata,
        geometry,
    )
    available_pixels = sorted(valid_chunks)
    missing_pixels = [pixel for pixel in required_pixels if pixel not in valid_chunks]
    coverage_known = center is not None and bool(required_pixels)
    if not valid_chunks:
        reason = "no_valid_catalog_chunks"
    elif not coverage_known:
        reason = "target_coordinates_unavailable"
    elif missing_pixels:
        reason = "target_healpix_not_installed"
    else:
        reason = "ok"
    return {
        "path": str(catalog_dir),
        "available": reason == "ok",
        "reason": reason,
        "minimum_size_bytes": MIN_LOCAL_CATALOG_FILE_BYTES,
        "center_degrees": list(center) if center is not None else None,
        "query_radius_deg": query_radius,
        "coverage_known": coverage_known,
        "required_pixels": required_pixels,
        "available_pixels": available_pixels,
        "missing_pixels": missing_pixels,
        "chunks": valid_chunks,
    }


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
) -> Tuple[bool, str, List[Dict[str, str]]]:
    attempts: List[Dict[str, str]] = []
    for label, args in _stage4_pcc_variants():
        command = "pcc " + " ".join(args)
        catalog = label.split(":", 1)[-1]
        skip_reason = _stage4_catalog_skip_reason(pipeline, catalog)
        if skip_reason:
            attempts.append(
                {
                    "label": label,
                    "phase": phase,
                    "command": command,
                    "status": "skipped",
                    "error": skip_reason,
                }
            )
            pipeline.log.warn(f"PCC 候选跳过 ({phase}, {label}): {skip_reason}")
            continue
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
    - platesolve -noflip -focal=160 -pixelsize=2.90 -catalog=gaia -order=3 后保存 stage4_psolved
    - SPCC 固定使用本地 Gaia DR3 xp_sampled 星表，并指定 Sony IMX585 OSC 参数；本地 SPCC 失败后 PCC，最后本地背景中性化/星点白平衡回退
    - SPCC 原生崩溃重启时可直接载入已有 stage4_psolved，跳过重复 platesolve
    """
    stage_label = PipelineStage.COLOR_CALIBRATION.label
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
    spcc_solution_quality: Dict[str, Any] = {
        "status": "not_attempted",
        "imprecise": False,
        "warning_code": None,
        "matched_messages": [],
        "confidence": None,
        "fallback_triggered": False,
    }
    spcc_solution_attempts: List[Dict[str, Any]] = []
    local_fallback_report: Optional[Dict[str, Any]] = None
    pcc_attempts: List[Dict[str, str]] = []
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}
    resume_from_psolved = (
        getattr(pipeline, "_stage1_input_mode", "") == "stage4_psolved_resume"
    )
    stage4_input_stem = "stage4_psolved" if resume_from_psolved else "stage3_bgremoved"

    try:
        pipeline.cmd_with_check("load", stage4_input_stem)
        messages.append(f"input={stage4_input_stem}")
    except (CommandError, SirilError) as e:
        if resume_from_psolved:
            pipeline.log.error(
                f"stage4 crash-resume input {stage4_input_stem} load failed: {e}"
            )
            raise
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

    pipeline.platesolve_ok = bool(resume_from_psolved)
    platesolve_attempted = not resume_from_psolved
    platesolve_command = (
        "resume existing stage4_psolved.fit"
        if resume_from_psolved
        else "platesolve " + " ".join(_stage4_platesolve_args())
    )
    platesolve_attempts: List[Dict[str, str]] = []
    if resume_from_psolved:
        messages.append(
            "platesolve reused from existing stage4_psolved.fit after SPCC native crash"
        )
        pipeline.log.info(
            "Stage4 crash-resume: reusing WCS from existing stage4_psolved.fit"
        )
    elif bool(getattr(pipeline.cfg, "stage4_platesolve_enabled", True)):
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

    ps_saved = True if resume_from_psolved else pipeline._save_stage_output("stage4_psolved")
    if not ps_saved:
        status = "degraded"
        hard_degraded = True
        messages.append("stage4_psolved 保存失败")

    catalog_metadata = _stage4_header_metadata(
        pipeline,
        "stage4_psolved",
        "stage3_bgremoved",
        getattr(pipeline, "source_file", None),
    )
    if not catalog_metadata:
        catalog_metadata = stage4_metadata
    spcc_catalog_status = _stage4_local_spcc_catalog_status(
        pipeline,
        catalog_metadata,
        stage4_geometry,
    )

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
    spcc_metadata_status = _stage4_spcc_metadata_not_checked(pipeline)
    spcc_fallback_metadata_status: Optional[Dict[str, Any]] = None
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

    if spcc_runtime_allowed and not spcc_catalog_status["available"]:
        spcc_runtime_allowed = False
        catalog_reason = str(spcc_catalog_status["reason"])
        required_pixels = spcc_catalog_status.get("required_pixels") or []
        available_pixels = spcc_catalog_status.get("available_pixels") or []
        skip_message = (
            "SPCC skipped before Siril call: local Gaia DR3 xp_sampled catalog "
            f"{catalog_reason}; path={spcc_catalog_status['path']}; "
            f"required_pixels={required_pixels}; available_pixels={available_pixels}"
        )
        messages.append(skip_message)
        pipeline.log.warn(skip_message)

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

    if spcc_runtime_allowed:
        spcc_metadata_status = _stage4_spcc_metadata_status(
            pipeline,
            whiteref=spcc_white_ref,
        )
        if not spcc_metadata_status["available"]:
            spcc_runtime_allowed = False
            skip_message = (
                "SPCC skipped before Siril call: metadata database preflight "
                f"{spcc_metadata_status['reason']}; "
                f"path={spcc_metadata_status['path']}; "
                f"missing={spcc_metadata_status['missing']}; "
                f"invalid_json={spcc_metadata_status['invalid_json_files']}; "
                f"empty_json={spcc_metadata_status['empty_json_files']}; "
                f"invalid_entries={spcc_metadata_status['invalid_entry_files']}"
            )
            messages.append(skip_message)
            pipeline.log.warn(skip_message)

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
            spcc_solution_quality = _stage4_run_spcc_with_cpu_guard(
                pipeline,
                spcc_args,
                messages,
            )
            spcc_solution_quality["white_reference"] = spcc_white_ref
            (
                color_ok,
                color_method,
                color_confidence,
                color_warning,
                spcc_restore_hard_degraded,
            ) = _stage4_finalize_spcc_solution(
                pipeline,
                spcc_solution_quality,
                messages,
                success_confidence=0.90,
            )
            spcc_solution_attempts.append(dict(spcc_solution_quality))
            hard_degraded = hard_degraded or spcc_restore_hard_degraded
            if spcc_restore_hard_degraded:
                status = "degraded"
            if color_ok and color_method == "SPCC":
                messages.append("SPCC ok")
                pipeline.log.info("分光光度色彩校准成功 (SPCC)")
        except (CommandError, SirilError) as e:
            primary_error = e
            if (
                spcc_white_ref != DEFAULT_SPCC_WHITE_REF
                and spcc_white_ref_reason != "explicit_config"
                and not target_aware_color
            ):
                spcc_fallback_metadata_status = _stage4_spcc_metadata_status(
                    pipeline,
                    whiteref=DEFAULT_SPCC_WHITE_REF,
                )
                if not spcc_fallback_metadata_status["available"]:
                    fallback_skip_message = (
                        "SPCC default whiteref retry skipped before Siril call: "
                        "metadata database preflight "
                        f"{spcc_fallback_metadata_status['reason']}; "
                        f"missing={spcc_fallback_metadata_status['missing']}"
                    )
                    messages.append(fallback_skip_message)
                    pipeline.log.warn(fallback_skip_message)
                else:
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
                        spcc_solution_quality = _stage4_run_spcc_with_cpu_guard(
                            pipeline,
                            fallback_args,
                            messages,
                        )
                        spcc_solution_quality["white_reference"] = (
                            DEFAULT_SPCC_WHITE_REF
                        )
                        (
                            color_ok,
                            color_method,
                            color_confidence,
                            color_warning,
                            spcc_restore_hard_degraded,
                        ) = _stage4_finalize_spcc_solution(
                            pipeline,
                            spcc_solution_quality,
                            messages,
                            success_confidence=0.86,
                        )
                        spcc_solution_attempts.append(dict(spcc_solution_quality))
                        hard_degraded = hard_degraded or spcc_restore_hard_degraded
                        if spcc_restore_hard_degraded:
                            status = "degraded"
                        if color_ok and color_method == "SPCC":
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
        recovering_imprecise_spcc = color_warning == "spcc_imprecise_solution"
        pcc_ok, pcc_result, pcc_attempts = _stage4_run_pcc(
            pipeline,
            phase=(
                "spcc_imprecise_recovery"
                if recovering_imprecise_spcc
                else "plate_solved"
            ),
        )
        if pcc_ok:
            color_ok = True
            color_method = "PCC"
            if recovering_imprecise_spcc:
                color_warning = "spcc_imprecise_solution_pcc_fallback"
                color_confidence = SPCC_IMPRECISE_PCC_RECOVERY_CONFIDENCE
                messages.append(
                    f"{pcc_result} ok after imprecise SPCC recovery "
                    f"(confidence={color_confidence:.2f})"
                )
            else:
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
        imprecise_spcc_warning = (
            color_warning if color_warning.startswith("spcc_imprecise_solution") else ""
        )
        try:
            (
                color_ok,
                color_method,
                fallback_warning,
                color_confidence,
                local_fallback_report,
                fallback_message,
            ) = _stage4_local_color_fallback(pipeline, target_aware=target_aware_color)
            if imprecise_spcc_warning:
                color_warning = (
                    f"{imprecise_spcc_warning}_{fallback_warning or 'local_fallback'}"
                )
                color_confidence = min(float(color_confidence), 0.55)
            else:
                color_warning = fallback_warning
            messages.append(fallback_message)
            pipeline.log.info(fallback_message)
            pipeline.log.debug(f"local color fallback report: {local_fallback_report}")
        except (
            AttributeError,
            CommandError,
            SirilError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
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

    spcc_osc_mode_report = _stage4_effective_osc_spcc_mode(pipeline)
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
        "photometric_calibration_allowed": bool(photometric_calibration_allowed),
        "pcc": {
            "catalogs": list(_stage4_pcc_catalogs()),
            "attempts": pcc_attempts,
            "local_astrometric_catalog": _stage4_local_astrometric_catalog_status(
                pipeline
            ),
            "network_enabled": _stage4_network_enabled(),
            "header_coordinate_fallback_allowed": bool(pcc_header_fallback_allowed),
            "header_center_coordinates": header_center_coordinates,
        },
        "spcc": {
            "catalog": SPCC_CATALOG,
            "solution_quality": spcc_solution_quality,
            "solution_attempts": spcc_solution_attempts,
            "local_catalog": spcc_catalog_status,
            "metadata_database": spcc_metadata_status,
            "fallback_metadata_database": spcc_fallback_metadata_status,
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
