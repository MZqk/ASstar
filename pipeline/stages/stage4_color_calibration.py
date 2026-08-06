"""Stage 4 plate solving and color calibration."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from channel_semantics import (
    channel_shape_dict,
    classify_channel_semantics,
    filter_hint_suggests_narrowband,
)
from device_geometry import (
    activate_device_geometry_report,
    build_device_geometry_report,
    validate_active_geometry,
)
from models import PipelineStage
from narrowband_normalization import (
    classify_dual_narrowband_mapping,
    normalize_dual_narrowband_candidate,
)
from sirilpy.exceptions import CommandError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_STAGE4_FOCAL_LENGTH_MM = 160.0
DEFAULT_STAGE4_PIXEL_SIZE_UM = 2.9
DEFAULT_STAGE4_INSTRUMENT = "Seestar S30 Pro"
DEFAULT_OSC_SENSOR = "Sony IMX585"
DEFAULT_OSC_FILTER_LP = "ZWO Seestar LP"
DEFAULT_OSC_FILTER_NO_FILTER = "No filter"
DEFAULT_SPCC_WHITE_REF = "Average Spiral Galaxy"
DEFAULT_SPCC_LIMITMAG = "10.5"
SPCC_SEED_MARKER_NAME = ".seestar-superimpose-spcc-seed"
SPCC_METADATA_FILES = (
    ("osc_sensor", DEFAULT_OSC_SENSOR, "osc_sensors/Sony_IMX585.json", True),
    (
        "osc_filter",
        DEFAULT_OSC_FILTER_LP,
        "osc_filters/ZWO_Seestar_LP.json",
        True,
    ),
    (
        "osc_filter",
        DEFAULT_OSC_FILTER_NO_FILTER,
        "osc_filters/No_filter.json",
        True,
    ),
    (
        "white_reference",
        DEFAULT_SPCC_WHITE_REF,
        "wb_refs/Average_spiral_galaxy.json",
        True,
    ),
    ("white_reference", "Star, type G2(v)", "wb_refs/Star_g2v.json", False),
)
PCC_CATALOG = "gaia"
PCC_LOCAL_CATALOG = "localgaia"
SPCC_CATALOG = "gaia"
SPCC_LOCAL_CATALOG = "localgaia"
PCC_TIMEOUT_DEFAULT_SEC = 180
PCC_TIMEOUT_MIN_SEC = 5
PCC_TIMEOUT_MAX_SEC = 180
PCC_CHECKPOINT_STEM = "stage4_pre_pcc"
PCC_CANDIDATE_STEM = "stage4_pcc_candidate"
SPCC_CANDIDATE_STEM = "stage4_spcc_candidate"
PHYSICAL_COLOR_STEM = "stage4_physical_color"
HOO_ARTISTIC_STEM = "stage4_hoo_artistic"
LOCAL_SPCC_DIRNAME = "siril_cat1_healpix8_xpsamp"
LOCAL_SPCC_FILE_PATTERN = "siril_cat1_healpix8_xpsamp_*.dat"
LOCAL_ASTROMETRIC_FILENAME = "siril_cat_healpix8_astro.dat"
MIN_LOCAL_CATALOG_FILE_BYTES = 1024
SPCC_IMPRECISE_LOG_MARKERS = (
    "the photometric color calibration seems to have found an imprecise solution",
    "测光法色彩校准似乎不能精确校准",
)
NO_FILTER_KEYWORDS = frozenset(
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
        os.getenv("SEESTAR_NETWORK_MODE", "0").strip().lower()
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


def _stage4_local_spcc_catalog_dir(pipeline) -> Path:
    configured = (
        getattr(pipeline, "local_gaia_photo_catalog", None)
        or os.getenv("SEESTAR_GAIA_PHOTO_CATALOG", "")
    )
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "siril" / LOCAL_SPCC_DIRNAME


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


def _stage4_local_spcc_catalog_status(pipeline) -> Dict[str, Any]:
    catalog_dir = _stage4_local_spcc_catalog_dir(pipeline)
    try:
        candidates = sorted(catalog_dir.glob(LOCAL_SPCC_FILE_PATTERN))
    except OSError:
        candidates = []
    valid_files = [path for path in candidates if _stage4_valid_catalog_file(path)]
    try:
        total_bytes = sum(path.stat().st_size for path in valid_files)
    except OSError:
        total_bytes = 0
    return {
        "path": str(catalog_dir),
        "available": bool(valid_files),
        "valid_chunk_count": len(valid_files),
        "valid_chunks": [path.name for path in valid_files],
        "size_bytes": int(total_bytes),
        "minimum_file_bytes": MIN_LOCAL_CATALOG_FILE_BYTES,
        "catalog": SPCC_LOCAL_CATALOG,
        "photometric_product": "Gaia DR3 xp_sampled",
    }


def _stage4_pcc_catalog_status(pipeline) -> Dict[str, Any]:
    """The Siril Gaia astrometric extract also carries Teff for offline PCC."""
    status = _stage4_local_astrometric_catalog_status(pipeline)
    return {
        **status,
        "catalog": PCC_LOCAL_CATALOG,
        "photometric_field": "Gaia DR3 Teff",
        "supports_offline_pcc": bool(status["available"]),
    }


def _stage4_preferred_pcc_catalog(pipeline) -> Optional[str]:
    if _stage4_pcc_catalog_status(pipeline)["supports_offline_pcc"]:
        return PCC_LOCAL_CATALOG
    if _stage4_network_enabled():
        return PCC_CATALOG
    return None


def _stage4_preferred_spcc_catalog(pipeline) -> Optional[str]:
    if _stage4_local_spcc_catalog_status(pipeline)["available"]:
        return SPCC_LOCAL_CATALOG
    if _stage4_network_enabled():
        return SPCC_CATALOG
    return None


def _stage4_spcc_runtime_enabled(pipeline) -> bool:
    configured = bool(
        getattr(
            pipeline.cfg,
            "stage4_spcc_enabled",
            getattr(pipeline.cfg, "spcc_enabled", True),
        )
    )
    raw = os.getenv("SEESTAR_SPCC_ENABLE")
    if raw is None:
        return configured
    normalized = raw.strip().lower()
    if normalized in ENV_TRUE_VALUES:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return configured


def _stage4_file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _stage4_spcc_source_commit(root: Path) -> Tuple[Optional[str], Optional[str]]:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest.get("source") if isinstance(manifest, dict) else None
        commit = source.get("commit") if isinstance(source, dict) else None
        if isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            return commit.lower(), "manifest.json"
    except (OSError, TypeError, ValueError):
        pass

    for path, pattern in (
        (root / SPCC_SEED_MARKER_NAME, r"(?m)^source_commit=([0-9a-fA-F]{40})$"),
        (root / "VERSION.txt", r"(?m)^Pinned commit:\s*([0-9a-fA-F]{40})$"),
    ):
        try:
            match = re.search(pattern, path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if match:
            return match.group(1).lower(), path.name
    return None, None


def _stage4_spcc_database_status(pipeline) -> Dict[str, Any]:
    configured = (
        getattr(pipeline, "spcc_database_dir", None)
        or os.getenv("SEESTAR_SPCC_DATABASE_DIR", "")
    )
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home()
        / "Library/Application Support/org.siril.Siril/siril-spcc-database"
    )
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    for kind, label, relative, required in SPCC_METADATA_FILES:
        path = root / relative
        try:
            size_bytes = path.stat().st_size if path.is_file() else 0
        except OSError:
            size_bytes = 0
        valid = size_bytes > 0
        sha256 = _stage4_file_sha256(path) if valid else None
        records.append(
            {
                "kind": kind,
                "label": label,
                "path": relative,
                "required": bool(required),
                "available": bool(valid),
                "size_bytes": int(size_bytes),
                "sha256": sha256,
            }
        )
        if required and not valid:
            missing.append(relative)
    source_commit, source_commit_source = _stage4_spcc_source_commit(root)
    return {
        "path": str(root),
        "available": not missing,
        "source_commit": source_commit,
        "source_commit_source": source_commit_source,
        "files": records,
        "required_files": [
            record["path"] for record in records if record["required"]
        ],
        "missing_files": missing,
        "preflight_only": True,
    }


def _stage4_selected_spcc_metadata(
    database_status: Dict[str, Any],
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    requested = {
        "osc_sensor": parameters.get("sensor"),
        "white_reference": parameters.get("white_reference"),
    }
    if not bool(parameters.get("narrowband")):
        requested["osc_filter"] = parameters.get("osc_filter")

    available_records = {
        (str(record.get("kind")), str(record.get("label", "")).casefold()): record
        for record in database_status.get("files", [])
        if isinstance(record, dict)
    }
    selected: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    for kind, raw_label in requested.items():
        label = str(raw_label or "").strip()
        record = available_records.get((kind, label.casefold()))
        if record is None:
            unresolved.append(f"{kind}={label or '<unset>'}")
        else:
            selected.append(dict(record))
    return {
        "source_commit": database_status.get("source_commit"),
        "source_commit_source": database_status.get("source_commit_source"),
        "files": selected,
        "unresolved": unresolved,
    }


def _stage4_active_target_type(pipeline) -> str:
    if hasattr(pipeline, "_active_target_type"):
        target_type = str(
            pipeline._active_target_type() or ""
        ).strip().lower()
        if target_type:
            return target_type
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        target_type = str(profile.get("target_type") or "").strip().lower()
        if target_type:
            return target_type
    return ""


def _stage4_filter_hint_suggests_narrowband(filter_hint: str) -> bool:
    return filter_hint_suggests_narrowband(filter_hint)


def _stage4_focal_length() -> float:
    return float(os.getenv("SEESTAR_STAGE4_PLATESOLVE_FOCAL", "160.0"))


def _stage4_pixel_size() -> float:
    return float(os.getenv("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE", "2.90"))


def _stage4_platesolve_order() -> str:
    return str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_ORDER", "3") or "").strip()


def _stage4_platesolve_geometry_args(pipeline=None) -> Tuple[str, ...]:
    active = (
        getattr(pipeline, "_stage4_active_geometry", None)
        if pipeline is not None
        else None
    )
    if isinstance(active, dict):
        focal = f"{float(active.get('focal_length_mm', 160.0)):.8g}"
        pixelsize = f"{float(active.get('pixel_size_um', 2.90)):.8g}"
    else:
        focal = str(
            os.getenv("SEESTAR_STAGE4_PLATESOLVE_FOCAL", "160") or "160"
        ).strip()
        pixelsize = str(
            os.getenv("SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE", "2.90") or "2.90"
        ).strip()
    # Plate solving adds WCS metadata; it must not rewrite the user's image orientation.
    return ("-noflip", f"-focal={focal}", f"-pixelsize={pixelsize}")


def _stage4_platesolve_args(pipeline=None) -> Tuple[str, ...]:
    args = _stage4_platesolve_geometry_args(pipeline)
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


def _stage4_header_platesolve_args(
    metadata: Dict[str, Any],
    pipeline=None,
) -> Tuple[str, ...]:
    center = _stage4_header_center_coordinates(metadata)
    if not center:
        return ()
    args = (center,) + _stage4_platesolve_geometry_args(pipeline)
    radius = str(os.getenv("SEESTAR_STAGE4_PLATESOLVE_HEADER_RADIUS", "") or "").strip()
    if radius:
        args += (f"-radius={radius}",)
    return args


def _stage4_platesolve_variants(
    pipeline,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Tuple[str, ...]]]:
    base = _stage4_platesolve_geometry_args(pipeline)
    order = _stage4_platesolve_order()
    order_args = (f"-order={order}",) if order else ()
    variants: List[Tuple[str, Tuple[str, ...]]] = []
    for catalog in _stage4_platesolve_catalogs(pipeline):
        variants.append((f"catalog:{catalog}", base + (f"-catalog={catalog}",) + order_args))
    header_base = _stage4_header_platesolve_args(metadata or {}, pipeline)
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
    catalog: str = PCC_CATALOG,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Run exactly one local or online Gaia PCC behind a killable boundary."""
    timeout_sec = int(
        getattr(pipeline.cfg, "stage4_pcc_timeout_sec", PCC_TIMEOUT_DEFAULT_SEC)
        or PCC_TIMEOUT_DEFAULT_SEC
    )
    timeout_sec = max(PCC_TIMEOUT_MIN_SEC, min(timeout_sec, PCC_TIMEOUT_MAX_SEC))
    catalog = (
        PCC_LOCAL_CATALOG
        if str(catalog).strip().lower() == PCC_LOCAL_CATALOG
        else PCC_CATALOG
    )
    command = f"pcc -catalog={catalog}"
    attempt: Dict[str, Any] = {
        "label": f"catalog:{catalog}",
        "phase": phase,
        "command": command,
        "status": "failed",
        "timeout_sec": timeout_sec,
        "attempt": 1,
        "max_attempts": 1,
    }

    if catalog == PCC_LOCAL_CATALOG:
        catalog_status = _stage4_pcc_catalog_status(pipeline)
        attempt["catalog_status"] = catalog_status
        attempt["offline"] = True
        if not catalog_status["supports_offline_pcc"]:
            attempt.update(status="skipped", error="local Gaia catalogue unavailable")
            return False, "PCC skipped: local Gaia catalogue unavailable", [attempt]
    elif not _stage4_network_enabled():
        attempt.update(status="skipped", error="network mode disabled")
        return False, "PCC skipped: network mode disabled", [attempt]
    else:
        attempt["offline"] = False

    test_runner = getattr(pipeline, "_run_stage4_pcc_once", None)
    if callable(test_runner):
        try:
            ok, detail = test_runner(timeout_sec=timeout_sec, catalog=catalog)
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
        "PCC 单次 "
        + ("离线" if catalog == PCC_LOCAL_CATALOG else "在线")
        + f" Gaia 校色开始（timeout={timeout_sec}s）"
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


def _stage4_spcc_output_is_imprecise(output: str) -> bool:
    normalized = str(output or "").casefold()
    return any(marker.casefold() in normalized for marker in SPCC_IMPRECISE_LOG_MARKERS)


def _stage4_run_spcc(
    pipeline,
    *,
    phase: str,
    catalog: str,
    args: Tuple[str, ...],
    narrowband: bool,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Run one SPCC candidate behind the same killable boundary as PCC."""
    timeout_sec = int(
        getattr(
            pipeline.cfg,
            "stage4_spcc_timeout_sec",
            getattr(pipeline.cfg, "stage4_pcc_timeout_sec", PCC_TIMEOUT_DEFAULT_SEC),
        )
        or PCC_TIMEOUT_DEFAULT_SEC
    )
    timeout_sec = max(PCC_TIMEOUT_MIN_SEC, min(timeout_sec, PCC_TIMEOUT_MAX_SEC))
    catalog = (
        SPCC_LOCAL_CATALOG
        if str(catalog).strip().lower() == SPCC_LOCAL_CATALOG
        else SPCC_CATALOG
    )
    command = "spcc " + " ".join(args)
    attempt: Dict[str, Any] = {
        "label": f"catalog:{catalog}",
        "phase": phase,
        "command": command,
        "status": "failed",
        "timeout_sec": timeout_sec,
        "attempt": 1,
        "max_attempts": 1,
        "narrowband": bool(narrowband),
    }

    if catalog == SPCC_LOCAL_CATALOG:
        catalog_status = _stage4_local_spcc_catalog_status(pipeline)
        attempt["catalog_status"] = catalog_status
        attempt["offline"] = True
        if not catalog_status["available"]:
            attempt.update(
                status="skipped",
                error="local Gaia DR3 xp_sampled catalogue unavailable",
            )
            return (
                False,
                "SPCC skipped: local Gaia DR3 xp_sampled catalogue unavailable",
                [attempt],
            )
    elif not _stage4_network_enabled():
        attempt.update(status="skipped", error="network mode disabled")
        return False, "SPCC skipped: network mode disabled", [attempt]
    else:
        attempt["offline"] = False

    test_runner = getattr(pipeline, "_run_stage4_spcc_once", None)
    if callable(test_runner):
        try:
            ok, detail = test_runner(
                timeout_sec=timeout_sec,
                catalog=catalog,
                args=tuple(args),
                narrowband=bool(narrowband),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            ok, detail = False, str(error)
        if ok and _stage4_spcc_output_is_imprecise(str(detail)):
            attempt["precision_warning"] = "spcc_imprecise_solution"
            attempt["precision_warning_policy"] = (
                "defer_to_target_aware_pixel_quality_gate"
            )
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
        return False, f"SPCC {phase} failed: {reason}", [attempt]

    script_path = process_dir / ".stage4_spcc_once.ssf"
    candidate_path = process_dir / f"{SPCC_CANDIDATE_STEM}.fit"
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
                    f"save {SPCC_CANDIDATE_STEM}",
                    "close",
                )
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        attempt["error"] = f"unable to prepare SPCC script: {error}"
        return False, f"SPCC {phase} failed: {attempt['error']}", [attempt]

    cli_command = [str(cli), "-d", str(process_dir)]
    ini_path = os.getenv("SEESTAR_SIRIL_CONFIG", "").strip()
    if ini_path:
        cli_command.extend(("-i", ini_path))
    cli_command.extend(("-s", str(script_path)))
    attempt["runner"] = "independent_siril_cli"
    attempt["cli"] = str(cli)
    pipeline.log.info(
        "SPCC 单次 "
        + ("离线" if catalog == SPCC_LOCAL_CATALOG else "在线")
        + f" Gaia DR3 校色开始（timeout={timeout_sec}s）"
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
        pipeline.log.warn(f"SPCC 单次尝试超时（{timeout_sec}s），转 PCC 回退")
        return False, f"SPCC {phase} timed out after {timeout_sec}s", [attempt]
    except (OSError, subprocess.SubprocessError) as error:
        attempt["error"] = str(error)
        return False, f"SPCC {phase} failed: {error}", [attempt]
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    output = completed.stdout or ""
    output_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if output_lines:
        attempt["output_tail"] = " | ".join(output_lines[-8:])
    if completed.returncode != 0:
        attempt["error"] = f"siril-cli exit={completed.returncode}"
        return False, f"SPCC {phase} failed: {attempt['error']}", [attempt]
    if _stage4_spcc_output_is_imprecise(output):
        # Siril emits this advisory when either fitted colour-ratio deviation
        # exceeds 0.1.  It is useful evidence, but not sufficient on its own to
        # reject a candidate: the caller still measures channel gains,
        # background chroma, clipping, dynamic range, stellar temperature and
        # target colour drift against the unchanged pre-colour checkpoint.
        attempt["precision_warning"] = "spcc_imprecise_solution"
        attempt["precision_warning_policy"] = (
            "defer_to_target_aware_pixel_quality_gate"
        )
    if not candidate_path.is_file() or candidate_path.stat().st_size <= 0:
        attempt["error"] = "SPCC candidate output missing"
        return False, f"SPCC {phase} failed: {attempt['error']}", [attempt]

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


def _stage4_siril_named_arg(name: str, value: str) -> str:
    escaped = str(value or "").replace('"', '\\"')
    return f'"-{name}={escaped}"'


def _stage4_effective_spcc_filter(pipeline, metadata: Dict[str, Any]) -> Tuple[str, str]:
    configured = str(
        getattr(pipeline.cfg, "stage4_spcc_osc_filter", "") or ""
    ).strip()
    if configured:
        return configured, "explicit_config"
    hint = _stage4_filter_semantics_text(pipeline, metadata)
    if any(keyword in hint for keyword in NO_FILTER_KEYWORDS):
        return DEFAULT_OSC_FILTER_NO_FILTER, "fits_or_user_no_filter_hint"
    return DEFAULT_OSC_FILTER_LP, "seestar_lp_default"


def _stage4_spcc_args(
    pipeline,
    metadata: Dict[str, Any],
    channel_policy: Dict[str, Any],
    *,
    catalog: str,
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    sensor = str(
        getattr(pipeline.cfg, "stage4_spcc_osc_sensor", DEFAULT_OSC_SENSOR)
        or DEFAULT_OSC_SENSOR
    ).strip()
    white_ref = str(
        getattr(
            pipeline.cfg,
            "stage4_spcc_white_ref",
            DEFAULT_SPCC_WHITE_REF,
        )
        or DEFAULT_SPCC_WHITE_REF
    ).strip()
    limitmag = str(
        getattr(
            pipeline.cfg,
            "stage4_spcc_limit_magnitude",
            DEFAULT_SPCC_LIMITMAG,
        )
        or DEFAULT_SPCC_LIMITMAG
    ).strip()
    common = (
        f"-catalog={catalog}",
        _stage4_siril_named_arg("oscsensor", sensor),
        _stage4_siril_named_arg("whiteref", white_ref),
    )
    narrowband = channel_policy.get("kind") == "narrowband_composite"
    if narrowband:
        mapping = classify_dual_narrowband_mapping(
            metadata,
            filter_hint=str(channel_policy.get("filter_hint") or ""),
        )
        minimum = float(
            getattr(pipeline.cfg, "stage4_nbn_mapping_confidence_min", 0.85)
            or 0.85
        )
        if float(mapping.get("confidence", 0.0)) < max(0.70, min(minimum, 0.99)):
            raise ValueError("dual-narrowband wavelengths are not confirmed")

        def numeric(attr: str, default: float, lower: float, upper: float) -> float:
            raw = float(getattr(pipeline.cfg, attr, default) or default)
            if not math.isfinite(raw):
                raw = default
            return max(lower, min(raw, upper))

        r_wl = numeric("stage4_spcc_narrowband_r_wavelength_nm", 656.28, 600.0, 700.0)
        r_bw = numeric("stage4_spcc_narrowband_r_bandwidth_nm", 20.0, 1.0, 100.0)
        g_wl = numeric("stage4_spcc_narrowband_g_wavelength_nm", 500.70, 450.0, 550.0)
        g_bw = numeric("stage4_spcc_narrowband_g_bandwidth_nm", 30.0, 1.0, 100.0)
        b_wl = numeric("stage4_spcc_narrowband_b_wavelength_nm", 500.70, 450.0, 550.0)
        b_bw = numeric("stage4_spcc_narrowband_b_bandwidth_nm", 30.0, 1.0, 100.0)
        args = common + (
            "-narrowband",
            f"-rwl={r_wl:g}",
            f"-rbw={r_bw:g}",
            f"-gwl={g_wl:g}",
            f"-gbw={g_bw:g}",
            f"-bwl={b_wl:g}",
            f"-bbw={b_bw:g}",
            f"-limitmag={limitmag}",
        )
        return args, {
            "sensor": sensor,
            "white_reference": white_ref,
            "limit_magnitude": limitmag,
            "narrowband": True,
            "mapping": mapping,
            "wavelengths_nm": {"r": r_wl, "g": g_wl, "b": b_wl},
            "bandwidths_nm": {"r": r_bw, "g": g_bw, "b": b_bw},
        }

    osc_filter, filter_reason = _stage4_effective_spcc_filter(pipeline, metadata)
    args = common + (
        _stage4_siril_named_arg("oscfilter", osc_filter),
        f"-limitmag={limitmag}",
    )
    return args, {
        "sensor": sensor,
        "white_reference": white_ref,
        "limit_magnitude": limitmag,
        "narrowband": False,
        "osc_filter": osc_filter,
        "osc_filter_reason": filter_reason,
    }


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
    input_profile = getattr(pipeline, "input_profile", {}) or {}
    input_state = (
        str(input_profile.get("state") or "unknown")
        if isinstance(input_profile, dict)
        else "unknown"
    )
    policy = classify_channel_semantics(
        channels=int(shape.get("channels", 0) or 0),
        metadata=metadata,
        input_state=input_state,
        checkpoint_linear=checkpoint_loaded,
        explicit_filter_hint=os.getenv("SEESTAR_STAGE4_FILTER_HINT", ""),
        target_profile=(
            pipeline.target_profile
            if isinstance(getattr(pipeline, "target_profile", None), dict)
            else None
        ),
    )
    policy["shape"] = shape
    return policy


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


def _stage4_rgb_chromaticity(values: np.ndarray) -> Optional[List[float]]:
    rgb = np.asarray(values, dtype=np.float64)
    if rgb.size != 3 or not np.all(np.isfinite(rgb)):
        return None
    total = float(np.sum(np.clip(rgb, 0.0, None)))
    if total <= 1e-8:
        return None
    return [float(value) for value in np.clip(rgb, 0.0, None) / total]


def _stage4_star_temperature_distribution(chw: np.ndarray) -> Dict[str, Any]:
    """Estimate a CCT distribution proxy from unsaturated bright star pixels."""
    rgb = np.clip(np.asarray(chw[:3], dtype=np.float64), 0.0, None)
    if rgb.shape[0] < 3 or rgb.size < 192:
        return {"valid": False, "reason": "insufficient_rgb_pixels", "sample_count": 0}
    lum = _stage4_luminance(rgb.astype(np.float32))
    finite = np.isfinite(lum) & np.all(np.isfinite(rgb), axis=0)
    if int(np.count_nonzero(finite)) < 64:
        return {"valid": False, "reason": "insufficient_finite_pixels", "sample_count": 0}
    valid_lum = lum[finite]
    low = float(np.quantile(valid_lum, 0.97))
    high = float(np.quantile(valid_lum, 0.9995))
    if high <= low:
        return {"valid": False, "reason": "no_star_dynamic_range", "sample_count": 0}
    channel_max = np.max(rgb, axis=0)
    channel_min = np.min(rgb, axis=0)
    chroma = (channel_max - channel_min) / np.maximum(channel_max, 1e-8)
    mask = finite & (lum >= low) & (lum <= high) & (chroma <= 0.85)
    sample_count = int(np.count_nonzero(mask))
    if sample_count < 24:
        return {
            "valid": False,
            "reason": "insufficient_unsaturated_star_samples",
            "sample_count": sample_count,
        }

    samples = rgb[:, mask]
    scale = np.maximum(np.max(samples, axis=0), 1e-8)
    r, g, b = samples / scale
    x_xyz = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y_xyz = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z_xyz = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xyz_sum = x_xyz + y_xyz + z_xyz
    valid_xyz = xyz_sum > 1e-8
    x = x_xyz[valid_xyz] / xyz_sum[valid_xyz]
    y = y_xyz[valid_xyz] / xyz_sum[valid_xyz]
    denominator = 0.1858 - y
    valid_xy = np.abs(denominator) > 1e-5
    n = (x[valid_xy] - 0.3320) / denominator[valid_xy]
    cct = -449.0 * n**3 + 3525.0 * n**2 - 6823.3 * n + 5520.33
    cct = cct[np.isfinite(cct) & (cct >= 1500.0) & (cct <= 30000.0)]
    if cct.size < 16:
        return {
            "valid": False,
            "reason": "temperature_proxy_unavailable",
            "sample_count": sample_count,
            "temperature_samples": int(cct.size),
        }
    return {
        "valid": True,
        "method": "linear_rgb_mccamy_cct_proxy",
        "sample_count": sample_count,
        "temperature_samples": int(cct.size),
        "median_kelvin": float(np.median(cct)),
        "p10_kelvin": float(np.quantile(cct, 0.10)),
        "p90_kelvin": float(np.quantile(cct, 0.90)),
        "warm_fraction": float(np.mean(cct < 4000.0)),
        "cool_fraction": float(np.mean(cct > 7500.0)),
    }


def _stage4_target_chromaticity(
    chw: np.ndarray,
    reference_luminance: np.ndarray,
) -> Optional[List[float]]:
    finite_lum = reference_luminance[np.isfinite(reference_luminance)]
    if finite_lum.size < 64:
        return None
    low = float(np.quantile(finite_lum, 0.60))
    high = float(np.quantile(finite_lum, 0.96))
    mask = (
        np.isfinite(reference_luminance)
        & (reference_luminance >= low)
        & (reference_luminance <= high)
    )
    if int(np.count_nonzero(mask)) < 32:
        mask = np.isfinite(reference_luminance)
    medians = np.array(
        [float(np.median(np.asarray(channel)[mask])) for channel in chw[:3]],
        dtype=np.float64,
    )
    return _stage4_rgb_chromaticity(medians)


def _stage4_post_calibration_color_checks(
    before: np.ndarray,
    after: np.ndarray,
    *,
    before_stats: Dict[str, Any],
    after_stats: Dict[str, Any],
    emission_target: bool,
    pipeline,
) -> Tuple[List[str], Dict[str, Any]]:
    before_star = _stage4_star_temperature_distribution(before)
    after_star = _stage4_star_temperature_distribution(after)
    star_ratio_min = float(
        getattr(pipeline.cfg, "stage4_pcc_star_temperature_ratio_min", 0.45)
        or 0.45
    )
    star_ratio_max = float(
        getattr(pipeline.cfg, "stage4_pcc_star_temperature_ratio_max", 2.20)
        or 2.20
    )
    star_ratio_min = max(0.20, min(star_ratio_min, 0.95))
    star_ratio_max = max(1.05, min(star_ratio_max, 5.0))
    star_ratio: Optional[float] = None
    reasons: List[str] = []
    if before_star.get("valid") and after_star.get("valid"):
        star_ratio = float(after_star["median_kelvin"]) / max(
            float(before_star["median_kelvin"]),
            1e-8,
        )
        if not star_ratio_min <= star_ratio <= star_ratio_max:
            reasons.append("star_temperature_distribution_shift_exceeded")

    before_bg = _stage4_rgb_chromaticity(
        np.asarray(before_stats.get("background_medians", []), dtype=np.float64)
    )
    after_bg = _stage4_rgb_chromaticity(
        np.asarray(after_stats.get("background_medians", []), dtype=np.float64)
    )
    background_delta: Optional[float] = None
    background_delta_max = float(
        getattr(pipeline.cfg, "stage4_pcc_background_color_delta_max", 0.22)
        or 0.22
    )
    background_delta_max = max(0.05, min(background_delta_max, 0.60))
    before_spread = float(before_stats.get("background_channel_spread", float("inf")))
    after_spread = float(after_stats.get("background_channel_spread", float("inf")))
    if before_bg is not None and after_bg is not None:
        background_delta = float(
            np.linalg.norm(np.asarray(after_bg) - np.asarray(before_bg))
        )
        if (
            background_delta > background_delta_max
            and after_spread > before_spread * 1.05
        ):
            reasons.append("background_color_difference_exceeded")

    before_lum = _stage4_luminance(np.asarray(before[:3], dtype=np.float32))
    before_target = _stage4_target_chromaticity(before, before_lum)
    after_target = _stage4_target_chromaticity(after, before_lum)
    target_drift: Optional[float] = None
    target_drift_max = float(
        getattr(
            pipeline.cfg,
            (
                "stage4_pcc_emission_target_color_drift_max"
                if emission_target
                else "stage4_pcc_target_color_drift_max"
            ),
            0.45 if emission_target else 0.28,
        )
        or (0.45 if emission_target else 0.28)
    )
    target_drift_max = max(0.05, min(target_drift_max, 0.75))
    if before_target is not None and after_target is not None:
        target_drift = float(
            np.linalg.norm(np.asarray(after_target) - np.asarray(before_target))
        )
        if target_drift > target_drift_max:
            reasons.append("target_color_drift_exceeded")

    return reasons, {
        "star_color_temperature_distribution": {
            "before": before_star,
            "after": after_star,
            "median_temperature_ratio": star_ratio,
            "accepted_range": [star_ratio_min, star_ratio_max],
            "status": (
                "passed"
                if star_ratio is not None
                and star_ratio_min <= star_ratio <= star_ratio_max
                else ("not_measurable" if star_ratio is None else "rejected")
            ),
        },
        "background_color_difference": {
            "before_chromaticity": before_bg,
            "after_chromaticity": after_bg,
            "chromaticity_delta": background_delta,
            "maximum_worsening_delta": background_delta_max,
            "before_channel_spread": before_spread,
            "after_channel_spread": after_spread,
            "status": (
                "not_measurable"
                if background_delta is None
                else (
                    "rejected"
                    if "background_color_difference_exceeded" in reasons
                    else "passed"
                )
            ),
        },
        "target_color_drift": {
            "before_chromaticity": before_target,
            "after_chromaticity": after_target,
            "chromaticity_delta": target_drift,
            "maximum_delta": target_drift_max,
            "emission_target_profile": bool(emission_target),
            "status": (
                "not_measurable"
                if target_drift is None
                else (
                    "rejected"
                    if "target_color_drift_exceeded" in reasons
                    else "passed"
                )
            ),
        },
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
    emission_balance_gain_ratio_max = float(
        getattr(
            pipeline.cfg,
            "stage4_pcc_emission_balance_gain_ratio_max",
            4.0,
        )
        or 4.0
    )
    emission_balance_gain_ratio_max = max(
        gain_ratio_max,
        min(emission_balance_gain_ratio_max, 5.0),
    )
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
    before_bg_spread = float(
        before_stats.get("background_channel_spread", float("inf"))
    )
    bg_spread_improvement_ratio = (
        bg_spread / max(before_bg_spread, 1e-8)
        if np.isfinite(before_bg_spread)
        else float("inf")
    )
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

    color_check_reasons, post_calibration_checks = (
        _stage4_post_calibration_color_checks(
            before,
            after,
            before_stats=before_stats,
            after_stats=after_stats,
            emission_target=emission_target,
            pipeline=pipeline,
        )
    )
    reasons.extend(color_check_reasons)

    target_aware_exemptions: List[str] = []
    emission_balance_verified = bool(
        emission_target
        and "channel_gain_ratio_exceeded" in reasons
        and gain_ratio <= emission_balance_gain_ratio_max
        and before_bg_spread >= 0.60
        and bg_spread <= min(bg_spread_max, 0.12)
        and bg_spread_improvement_ratio <= 0.25
        and clip_growth <= clip_growth_max
        and (
            before_dynamic <= 1e-6
            or 0.50 <= dynamic_ratio <= 2.50
        )
        and before.shape == after.shape
        and bool(before_stats.get("valid"))
        and bool(after_stats.get("valid"))
        and float(after_stats.get("finite_ratio", 0.0)) >= 0.999999
    )
    if emission_balance_verified:
        reasons.remove("channel_gain_ratio_exceeded")
        target_aware_exemptions.append(
            "large_gain_accepted_after_verified_background_balance"
        )

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
            "emission_balance_gain_ratio_max": emission_balance_gain_ratio_max,
            "highlight_clip_growth_max": clip_growth_max,
        },
        "measurements": {
            "channel_gains": gains,
            "channel_gain_ratio": gain_ratio,
            "background_channel_spread_before": before_bg_spread,
            "background_channel_spread": bg_spread,
            "background_channel_spread_improvement_ratio": (
                bg_spread_improvement_ratio
            ),
            "highlight_clip_growth": clip_growth,
            "dynamic_range_ratio": dynamic_ratio,
        },
        "post_calibration_checks": post_calibration_checks,
        "target_aware_exemptions": target_aware_exemptions,
        "before": before_stats,
        "after": after_stats,
        "rejection_reasons": reasons,
    }
    return bool(accepted), report


def _stage4_shape_dict(shape) -> Dict[str, int]:
    return channel_shape_dict(shape)


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


def _stage4_run_narrowband_normalization(
    pipeline,
    metadata: Dict[str, Any],
) -> tuple[bool, Dict[str, Any], str]:
    report: Dict[str, Any] = {
        "schema": "seestar.narrowband-normalization.v1",
        "status": "not_run",
        "accepted": False,
    }
    if not bool(
        getattr(pipeline.cfg, "stage4_narrowband_normalization_enabled", True)
    ):
        report.update(status="disabled", issues=["disabled_by_configuration"])
        return False, report, "narrowband normalization disabled"
    filter_hint = _stage4_filter_semantics_text(pipeline, metadata)
    mapping_confidence_min = float(
        getattr(
            pipeline.cfg,
            "stage4_nbn_mapping_confidence_min",
            0.85,
        )
    )
    mapping = classify_dual_narrowband_mapping(
        metadata,
        filter_hint=filter_hint,
    )
    if float(mapping.get("confidence", 0.0)) < mapping_confidence_min:
        report.update(
            status="skipped_unconfirmed_mapping",
            mapping=mapping,
            issues=["channel_mapping_unconfirmed"],
        )
        return (
            False,
            report,
            "narrowband normalization skipped: Ha/OIII mapping unconfirmed",
        )
    baseline_saved = pipeline._save_stage_output("stage4_pre_nbn")
    if not baseline_saved:
        report.update(
            status="prohibited",
            issues=["immutable_baseline_save_failed"],
        )
        return (
            False,
            report,
            "narrowband normalization prohibited: immutable baseline save failed",
        )
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        candidate, report = normalize_dual_narrowband_candidate(
            image_data,
            metadata=metadata,
            filter_hint=filter_hint,
            mapping_confidence_min=mapping_confidence_min,
            strength=float(
                getattr(pipeline.cfg, "stage4_nbn_strength", 0.55)
            ),
            gain_limit=float(
                getattr(pipeline.cfg, "stage4_nbn_gain_limit", 1.08)
            ),
            line_ratio_drift_max=float(
                getattr(
                    pipeline.cfg,
                    "stage4_nbn_line_ratio_drift_max",
                    0.12,
                )
            ),
        )
        report["transaction"]["baseline_saved"] = True
        if not bool(report.get("accepted")):
            return (
                False,
                report,
                "narrowband normalization candidate rejected: "
                + ",".join(report.get("issues") or []),
            )
        _stage4_write_image_pixels(pipeline, candidate)
        if not pipeline._save_stage_output("stage4_nbn_candidate"):
            raise RuntimeError("stage4_nbn_candidate save failed")
        report["transaction"].update(
            candidate_saved=True,
            rollback_performed=False,
        )
        metrics = report.get("metrics") or {}
        return (
            True,
            report,
            "narrowband normalization accepted "
            f"(background_improvement={float(metrics.get('background_color_improvement', 0.0)):.4f}, "
            f"line_ratio_drift={float(metrics.get('ha_oiii_ratio_drift', 0.0)):.3f})",
        )
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report.update(
            status="failed",
            accepted=False,
            error=str(error),
        )
        try:
            pipeline.cmd_with_check("load", "stage4_pre_nbn")
            report.setdefault("transaction", {}).update(
                rollback_performed=True,
            )
        except (CommandError, SirilError) as rollback_error:
            report.setdefault("transaction", {}).update(
                rollback_performed=False,
                rollback_error=str(rollback_error),
            )
        return (
            False,
            report,
            f"narrowband normalization unavailable; baseline restored: {error}",
        )


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
    """Stage 4: plate solve, SPCC-first physical color, then bounded fallbacks."""
    stage_label = PipelineStage.COLOR_CALIBRATION.label
    pipeline.log.stage_start(stage_label)
    status = "ok"
    hard_degraded = False
    requires_review = False
    pipeline._stage4_color_review_required = False
    color_method = "PRESERVE_INPUT"
    color_warning = ""
    color_confidence = 0.0
    messages: List[str] = []
    local_fallback_report: Optional[Dict[str, Any]] = None
    narrowband_normalization_report: Dict[str, Any] = {
        "schema": "seestar.narrowband-normalization.v1",
        "status": "not_applicable",
        "accepted": False,
    }
    pcc_attempts: List[Dict[str, Any]] = []
    pcc_quality_report: Dict[str, Any] = {"enabled": True, "accepted": False, "status": "not_run"}
    rollback_report: Dict[str, Any] = {"required": False, "restored": False}
    spcc_attempts: List[Dict[str, Any]] = []
    spcc_quality_report: Dict[str, Any] = {
        "enabled": True,
        "accepted": False,
        "status": "not_run",
    }
    spcc_parameters: Dict[str, Any] = {}
    spcc_rollback_report: Dict[str, Any] = {
        "required": False,
        "restored": False,
    }
    pcc_fallback_used = False
    artistic_hoo_report: Dict[str, Any] = {
        "role": "artistic_derivative",
        "status": "not_applicable",
        "feeds_main_pipeline": False,
        "output": None,
    }
    physical_saved = False
    artistic_saved = False
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
        pipeline.log.warn(f"Stage4 线性检查点载入失败，禁止 SPCC/PCC: {e}")

    stage4_metadata = _stage4_header_metadata(pipeline)
    setattr(pipeline, "_stage4_header_metadata", stage4_metadata)
    stage4_geometry = _stage4_image_geometry(pipeline)
    device_geometry_report: Dict[str, Any]
    try:
        device_geometry_report = activate_device_geometry_report(
            build_device_geometry_report(
                stage4_metadata,
                image_shape=stage4_geometry.get("current_shape") or {},
                crop_report=stage4_geometry.get("stage2_crop") or {},
            ),
            enabled=bool(
                getattr(pipeline.cfg, "stage4_auto_geometry_enabled", True)
            ),
            confidence_min=float(
                getattr(
                    pipeline.cfg,
                    "stage4_auto_geometry_confidence_min",
                    0.85,
                )
            ),
        )
        pipeline._stage4_active_geometry = dict(
            device_geometry_report["activation"]["runtime_geometry"]
        )
        pipeline._device_geometry_report = device_geometry_report
        pipeline._write_stage_json(
            "device_geometry_report.json",
            device_geometry_report,
        )
        geometry_activation = device_geometry_report["activation"]
        messages.append(
            "device_geometry=active_guarded "
            f"confidence={float(geometry_activation['confidence']):.2f} "
            f"applied={str(bool(geometry_activation['applied'])).lower()} "
            f"source={geometry_activation['source']}"
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        device_geometry_report = {
            "schema": "seestar.device-geometry.v1",
            "mode": "report_only",
            "applied": False,
            "status": "unavailable",
            "error": str(error),
        }
        pipeline._device_geometry_report = device_geometry_report
        pipeline._stage4_active_geometry = {
            "focal_length_mm": _stage4_focal_length(),
            "pixel_size_um": _stage4_pixel_size(),
        }
        pipeline.log.warn(f"Stage4 device geometry report unavailable: {error}")
        messages.append("device_geometry report unavailable; runtime geometry unchanged")
    crop_total = (stage4_geometry.get("stage2_crop") or {}).get("total_crop") or {}
    if crop_total:
        messages.append(
            "stage4 geometry uses stage2 crop "
            f"L/T/R/B={crop_total.get('left')}/{crop_total.get('top')}/"
            f"{crop_total.get('right')}/{crop_total.get('bottom')}"
        )
    active_geometry = getattr(pipeline, "_stage4_active_geometry", {}) or {}
    pipeline.log.info(
        "Stage4 instrument geometry: "
        f"{DEFAULT_STAGE4_INSTRUMENT} tele sensor={DEFAULT_OSC_SENSOR}, "
        f"focal={float(active_geometry.get('focal_length_mm', _stage4_focal_length())):g}mm, "
        f"pixelsize={float(active_geometry.get('pixel_size_um', _stage4_pixel_size())):g}um, "
        f"shape={stage4_geometry.get('current_shape')}"
    )

    pipeline.platesolve_ok = False
    platesolve_attempted = True
    platesolve_command = "platesolve " + " ".join(
        _stage4_platesolve_args(pipeline)
    )
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
    elif pipeline.platesolve_ok:
        solved_metadata = _stage4_header_metadata(
            pipeline,
            "stage4_psolved",
        )
        device_geometry_report = validate_active_geometry(
            device_geometry_report,
            solved_metadata,
            residual_max=float(
                getattr(
                    pipeline.cfg,
                    "stage4_auto_geometry_scale_residual_max",
                    0.05,
                )
            ),
        )
        pipeline._device_geometry_report = device_geometry_report
        pipeline._write_stage_json(
            "device_geometry_report.json",
            device_geometry_report,
        )
        geometry_validation = (
            device_geometry_report.get("activation", {}).get("validation", {})
        )
        messages.append(
            "device_geometry_validation="
            f"{geometry_validation.get('status', 'unavailable')}"
        )
        if geometry_validation.get("accepted") is False:
            pipeline.log.warn(
                "Stage4 自动几何与解算 WCS 比例冲突，回滚到未解析输入并禁止 SPCC/PCC"
            )
            try:
                pipeline.cmd_with_check("load", stage4_input_stem)
                ps_saved = pipeline._save_stage_output("stage4_psolved")
                pipeline.platesolve_ok = False
                status = "degraded"
                requires_review = True
                messages.append(
                    "auto geometry WCS validation rejected; restored stage3 input"
                )
            except (CommandError, SirilError) as error:
                pipeline.platesolve_ok = False
                status = "degraded"
                hard_degraded = True
                requires_review = True
                messages.append(
                    f"auto geometry validation rollback failed: {error}"
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
    target_profile = getattr(pipeline, "target_profile", {}) or {}
    profile_fallback_used = bool(
        isinstance(target_profile, dict)
        and str(
            target_profile.get("classification_method") or ""
        ).strip().lower()
        == "fallback"
    )
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
    pipeline.channel_profile = dict(channel_policy)
    pipeline._channel_semantics = str(channel_policy["kind"])
    input_classification = {
        "kind": pipeline._channel_semantics,
        "label": {
            "broadband_rgb_osc": "宽带 RGB/OSC",
            "narrowband_composite": "双窄带/窄带合成",
            "mono": "单色",
            "nonlinear_color": "非线性色彩图",
            "unknown": "未知",
        }.get(pipeline._channel_semantics, pipeline._channel_semantics),
        "calibration_route": {
            "broadband_rgb_osc": "spcc_primary_then_pcc_exception_fallback",
            "narrowband_composite": "spcc_narrowband_physical_plus_isolated_hoo_artistic",
            "mono": "skip_all_color_calibration",
            "nonlinear_color": "preserve_input",
            "unknown": "preserve_input_review",
        }.get(pipeline._channel_semantics, "preserve_input_review"),
    }
    messages.append(
        "channel_semantics="
        f"{pipeline._channel_semantics} confidence={float(channel_policy['confidence']):.2f}"
    )
    target_aware_color = _stage4_active_target_type(pipeline) in EMISSION_NEBULA_TARGET_TYPES
    pre_pcc_saved = pipeline._save_stage_output(PCC_CHECKPOINT_STEM)
    if not pre_pcc_saved:
        status = "degraded"
        hard_degraded = True
        requires_review = True
        messages.append(
            "immutable pre-color checkpoint save failed; SPCC/PCC prohibited"
        )

    before_chw: Optional[np.ndarray] = None
    try:
        before_pixels = pipeline.siril.get_image_pixeldata(preview=False)
        before_chw, _restore_before = _stage4_image_as_chw(before_pixels)
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        hard_degraded = True
        requires_review = True
        status = "degraded"
        messages.append(f"pre-color pixels unavailable: {error}")

    policy_status = "not_applicable"
    local_spcc_catalog = _stage4_local_spcc_catalog_status(pipeline)
    selected_spcc_catalog = _stage4_preferred_spcc_catalog(pipeline)
    spcc_database = _stage4_spcc_database_status(pipeline)
    local_pcc_catalog = _stage4_pcc_catalog_status(pipeline)
    selected_pcc_catalog = _stage4_preferred_pcc_catalog(pipeline)
    physical_color_input = channel_policy["kind"] in {
        "broadband_rgb_osc",
        "narrowband_composite",
    }
    spcc_allowed = bool(
        physical_color_input
        and _stage4_spcc_runtime_enabled(pipeline)
        and pipeline.platesolve_ok
        and pre_pcc_saved
        and before_chw is not None
        and selected_spcc_catalog is not None
    )
    pcc_allowed = bool(
        channel_policy["kind"] == "broadband_rgb_osc"
        and pipeline.platesolve_ok
        and pre_pcc_saved
        and before_chw is not None
        and selected_pcc_catalog is not None
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
        messages.append("nonlinear input: input colors preserved; SPCC/PCC prohibited")
    elif channel_policy["kind"] == "unknown":
        policy_status = "preserved_unknown"
        color_method = "PRESERVE_INPUT"
        color_warning = "linear_or_channel_semantics_unknown"
        color_confidence = 0.20
        requires_review = True
        status = "degraded"
        messages.append("linear/channel semantics unknown: input colors preserved")
    else:
        narrowband_physical = channel_policy["kind"] == "narrowband_composite"
        accepted_spcc_methods = {
            "SPCC",
            "SPCC_LOCAL_GAIA",
            "SPCC_NARROWBAND",
            "SPCC_NARROWBAND_LOCAL_GAIA",
        }
        if not spcc_allowed:
            if not _stage4_spcc_runtime_enabled(pipeline):
                spcc_skip_reason = "disabled by config/runtime preflight"
            elif not pipeline.platesolve_ok:
                spcc_skip_reason = "plate solve unavailable"
            elif selected_spcc_catalog is None:
                spcc_skip_reason = (
                    "local Gaia DR3 xp_sampled catalogue unavailable and network mode disabled"
                )
            else:
                spcc_skip_reason = "immutable pre-color source unavailable"
            color_warning = "spcc_not_available"
            messages.append(f"SPCC skipped: {spcc_skip_reason}")
        else:
            try:
                spcc_args, spcc_parameters = _stage4_spcc_args(
                    pipeline,
                    stage4_metadata,
                    channel_policy,
                    catalog=str(selected_spcc_catalog),
                )
                spcc_database["selected"] = _stage4_selected_spcc_metadata(
                    spcc_database,
                    spcc_parameters,
                )
            except (TypeError, ValueError) as error:
                color_warning = "spcc_narrowband_metadata_unconfirmed"
                messages.append(f"SPCC preflight rejected: {error}")
            else:
                spcc_ok, spcc_result, spcc_attempts = _stage4_run_spcc(
                    pipeline,
                    phase=(
                        "linear_dual_narrowband_physical"
                        if narrowband_physical
                        else "linear_broadband"
                    ),
                    catalog=str(selected_spcc_catalog),
                    args=spcc_args,
                    narrowband=narrowband_physical,
                )
                if spcc_ok:
                    try:
                        pipeline.cmd_with_check("load", SPCC_CANDIDATE_STEM)
                        candidate_pixels = pipeline.siril.get_image_pixeldata(
                            preview=False
                        )
                        candidate_chw, _restore_candidate = _stage4_image_as_chw(
                            candidate_pixels
                        )
                        accepted, spcc_quality_report = _stage4_pcc_quality_gate(
                            before_chw,
                            candidate_chw,
                            pipeline,
                        )
                        spcc_quality_report["calibration"] = "SPCC"
                        spcc_quality_report["physical_color"] = True
                        precision_warnings = [
                            str(attempt.get("precision_warning"))
                            for attempt in spcc_attempts
                            if attempt.get("precision_warning")
                        ]
                        spcc_quality_report["siril_precision_warning"] = {
                            "present": bool(precision_warnings),
                            "codes": precision_warnings,
                            "policy": (
                                "accepted_only_if_target_aware_pixel_quality_gate_passes"
                                if precision_warnings
                                else "not_applicable"
                            ),
                        }
                        if accepted:
                            if narrowband_physical:
                                color_method = (
                                    "SPCC_NARROWBAND_LOCAL_GAIA"
                                    if selected_spcc_catalog == SPCC_LOCAL_CATALOG
                                    else "SPCC_NARROWBAND"
                                )
                            else:
                                color_method = (
                                    "SPCC_LOCAL_GAIA"
                                    if selected_spcc_catalog == SPCC_LOCAL_CATALOG
                                    else "SPCC"
                                )
                            color_confidence = (
                                0.92
                                if selected_spcc_catalog == SPCC_LOCAL_CATALOG
                                else 0.88
                            )
                            color_warning = ""
                            policy_status = "accepted"
                            messages.append(
                                f"{spcc_result} accepted by physical-color quality gate"
                            )
                            pipeline.log.info(
                                "SPCC Gaia DR3 校色通过目标感知质量门"
                            )
                        else:
                            color_warning = "spcc_quality_gate_rejected"
                            messages.append(
                                "SPCC candidate rejected: "
                                + ",".join(
                                    spcc_quality_report.get(
                                        "rejection_reasons",
                                        [],
                                    )
                                )
                            )
                    except (
                        AttributeError,
                        CommandError,
                        SirilError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        spcc_quality_report = {
                            "enabled": True,
                            "accepted": False,
                            "status": "candidate_load_or_measure_failed",
                            "error": str(error),
                        }
                        color_warning = "spcc_candidate_unavailable"
                        messages.append(f"SPCC candidate unavailable: {error}")
                else:
                    rejected_reason = next(
                        (
                            str(attempt.get("rejection_reason") or "")
                            for attempt in spcc_attempts
                            if attempt.get("rejection_reason")
                        ),
                        "",
                    )
                    color_warning = rejected_reason or "spcc_single_attempt_failed"
                    messages.append(spcc_result)

        baseline_restored = True
        if color_method not in accepted_spcc_methods:
            spcc_rollback_report["required"] = bool(spcc_attempts)
            try:
                pipeline.cmd_with_check("load", PCC_CHECKPOINT_STEM)
                spcc_rollback_report.update(
                    restored=True,
                    checkpoint=f"{PCC_CHECKPOINT_STEM}.fit",
                )
                messages.append(
                    "restored immutable pre-color linear checkpoint after SPCC"
                )
            except (CommandError, SirilError) as error:
                spcc_rollback_report.update(restored=False, error=str(error))
                baseline_restored = False
                hard_degraded = True
                messages.append(f"pre-color restore after SPCC failed: {error}")

            if narrowband_physical:
                color_method = "PRESERVE_INPUT"
                color_confidence = 0.55 if baseline_restored else 0.15
                color_warning = color_warning or "narrowband_physical_spcc_failed"
                requires_review = True
                status = "degraded"
                policy_status = "physical_calibration_review_required"
                messages.append(
                    "dual-narrowband physical SPCC unavailable; broadband PCC is prohibited"
                )
            else:
                if not pcc_allowed:
                    pcc_skip_reason = (
                        "plate solve unavailable"
                        if not pipeline.platesolve_ok
                        else (
                            "local Gaia astrometric catalogue unavailable and network mode disabled"
                            if selected_pcc_catalog is None
                            else "immutable pre-color source unavailable"
                        )
                    )
                    messages.append(f"PCC exception fallback skipped: {pcc_skip_reason}")
                elif baseline_restored:
                    pcc_ok, pcc_result, pcc_attempts = _stage4_run_pcc(
                        pipeline,
                        phase="spcc_exception_fallback",
                        catalog=str(selected_pcc_catalog),
                    )
                    if pcc_ok:
                        try:
                            pipeline.cmd_with_check("load", PCC_CANDIDATE_STEM)
                            candidate_pixels = pipeline.siril.get_image_pixeldata(
                                preview=False
                            )
                            candidate_chw, _restore_candidate = _stage4_image_as_chw(
                                candidate_pixels
                            )
                            accepted, pcc_quality_report = _stage4_pcc_quality_gate(
                                before_chw,
                                candidate_chw,
                                pipeline,
                            )
                            pcc_quality_report["calibration"] = "PCC"
                            if accepted:
                                color_method = (
                                    "PCC_LOCAL_GAIA"
                                    if selected_pcc_catalog == PCC_LOCAL_CATALOG
                                    else "PCC"
                                )
                                color_confidence = (
                                    0.82
                                    if selected_pcc_catalog == PCC_LOCAL_CATALOG
                                    else 0.78
                                )
                                color_warning = "spcc_exception_pcc_fallback"
                                policy_status = "accepted_exception_fallback"
                                pcc_fallback_used = True
                                messages.append(
                                    f"{pcc_result} accepted after SPCC exception"
                                )
                                pipeline.log.info(
                                    "SPCC 异常后 PCC 校色通过目标感知质量门"
                                )
                            else:
                                color_warning = "pcc_quality_gate_rejected"
                                messages.append(
                                    "PCC fallback candidate rejected: "
                                    + ",".join(
                                        pcc_quality_report.get(
                                            "rejection_reasons",
                                            [],
                                        )
                                    )
                                )
                        except (
                            AttributeError,
                            CommandError,
                            SirilError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as error:
                            pcc_quality_report = {
                                "enabled": True,
                                "accepted": False,
                                "status": "candidate_load_or_measure_failed",
                                "error": str(error),
                            }
                            color_warning = "pcc_candidate_unavailable"
                            messages.append(f"PCC fallback candidate unavailable: {error}")
                    else:
                        color_warning = "pcc_single_attempt_failed"
                        messages.append(pcc_result)

                if color_method not in {"PCC", "PCC_LOCAL_GAIA"}:
                    rollback_report["required"] = bool(pcc_attempts)
                    try:
                        pipeline.cmd_with_check("load", PCC_CHECKPOINT_STEM)
                        rollback_report.update(
                            restored=True,
                            checkpoint=f"{PCC_CHECKPOINT_STEM}.fit",
                        )
                        messages.append(
                            "restored immutable pre-color checkpoint after PCC fallback"
                        )
                    except (CommandError, SirilError) as error:
                        rollback_report.update(restored=False, error=str(error))
                        hard_degraded = True
                        messages.append(f"pre-color restore after PCC failed: {error}")

                    if rollback_report.get("restored") or not pcc_attempts:
                        try:
                            (
                                local_applied,
                                color_method,
                                fallback_warning,
                                color_confidence,
                                local_fallback_report,
                                fallback_message,
                            ) = _stage4_local_color_fallback(
                                pipeline,
                                narrowband=False,
                            )
                            color_warning = color_warning or fallback_warning
                            messages.append(fallback_message)
                            if not local_applied:
                                color_method = "PRESERVE_INPUT"
                        except (
                            AttributeError,
                            CommandError,
                            SirilError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as error:
                            color_method = "PRESERVE_INPUT"
                            color_warning = color_warning or "local_star_fallback_failed"
                            color_confidence = 0.20
                            messages.append(
                                "local star-only fallback failed; input preserved: "
                                f"{error}"
                            )
                    requires_review = True
                    status = "degraded"
                    policy_status = "fallback_review_required"

        if narrowband_physical:
            physical_saved = pipeline._save_stage_output(PHYSICAL_COLOR_STEM)
            if not physical_saved:
                hard_degraded = True
                requires_review = True
                status = "degraded"
                messages.append("dual-narrowband physical-color checkpoint save failed")
            else:
                (
                    narrowband_normalized,
                    narrowband_normalization_report,
                    narrowband_message,
                ) = _stage4_run_narrowband_normalization(
                    pipeline,
                    stage4_metadata,
                )
                narrowband_normalization_report = dict(
                    narrowband_normalization_report
                )
                narrowband_normalization_report.update(
                    {
                        "role": "artistic_derivative",
                        "physical_parent": f"{PHYSICAL_COLOR_STEM}.fit",
                        "feeds_main_pipeline": False,
                    }
                )
                messages.append(narrowband_message)
                if narrowband_normalized:
                    artistic_saved = pipeline._save_stage_output(HOO_ARTISTIC_STEM)
                    narrowband_normalization_report["artistic_output"] = (
                        f"{HOO_ARTISTIC_STEM}.fit" if artistic_saved else None
                    )
                    if artistic_saved:
                        artistic_hoo_report = {
                            **narrowband_normalization_report,
                            "status": "accepted",
                            "output": f"{HOO_ARTISTIC_STEM}.fit",
                        }
                        messages.append(
                            "isolated HOO artistic derivative saved; main pipeline remains physical"
                        )
                    else:
                        artistic_hoo_report = {
                            **narrowband_normalization_report,
                            "status": "accepted_output_save_failed",
                            "output": None,
                        }
                        messages.append("isolated HOO artistic derivative save failed")
                else:
                    artistic_hoo_report = {
                        **narrowband_normalization_report,
                        "output": None,
                    }
                pipeline._stage4_narrowband_normalization_report = (
                    narrowband_normalization_report
                )
                pipeline._write_stage_json(
                    "stage4_narrowband_normalization.json",
                    artistic_hoo_report,
                )
                try:
                    pipeline.cmd_with_check("load", PHYSICAL_COLOR_STEM)
                    messages.append(
                        "restored physical dual-narrowband branch after HOO derivative"
                    )
                except (CommandError, SirilError) as error:
                    hard_degraded = True
                    requires_review = True
                    status = "degraded"
                    messages.append(
                        "physical branch restore after HOO derivative failed: "
                        f"{error}"
                    )

    color_risk_warning = bool(
        color_warning and color_warning != "spcc_exception_pcc_fallback"
    )
    if (
        color_risk_warning
        and stage4_policy.get("reduce_saturation_if_solution_imprecise", False)
    ):
        messages.append("color policy limits later saturation/color gains due to imprecise solution")
    if color_method == "LOCAL_STAR_COLOR_RESTORE" and platesolve_attempted and not pipeline.platesolve_ok:
        messages.append("platesolve 失败，已使用恒星软遮罩局部色彩回退")

    color_saved = pipeline._save_stage_output("stage4_color")
    if not color_saved:
        status = "degraded"
        hard_degraded = True
        requires_review = True
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

    local_star_applied = bool(
        isinstance(local_fallback_report, dict)
        and bool(
            (
                local_fallback_report.get("star_white_balance") or {}
            ).get("applied")
        )
    )
    broadband_local_fallback_used = bool(
        channel_policy.get("kind") == "broadband_rgb_osc"
        and policy_status == "fallback_review_required"
        and color_method == "LOCAL_STAR_COLOR_RESTORE"
        and local_star_applied
    )
    stage_fallback_used = bool(
        profile_fallback_used
        or pcc_fallback_used
        or broadband_local_fallback_used
    )
    platesolve_diagnostics = _stage4_platesolve_diagnostics(
        platesolve_attempts,
        stage4_metadata,
    )
    color_applied_methods = {
        "SPCC",
        "SPCC_LOCAL_GAIA",
        "SPCC_NARROWBAND",
        "SPCC_NARROWBAND_LOCAL_GAIA",
        "PCC",
        "PCC_LOCAL_GAIA",
        "LOCAL_STAR_COLOR_RESTORE",
    }
    intentional_color_skip = channel_policy.get("kind") in {
        "mono",
        "nonlinear_color",
    }
    components = {
        "target_profile": {
            "status": "applied",
            "method": target_profile.get("classification_method"),
            "reason_code": (
                "target_profiler_fallback"
                if profile_fallback_used
                else "accepted"
            ),
            "fallback_used": profile_fallback_used,
        },
        "platesolve": {
            "status": (
                "applied"
                if pipeline.platesolve_ok
                else "skipped"
                if not platesolve_attempted
                else "failed"
            ),
            "method": platesolve_command if pipeline.platesolve_ok else None,
            "reason_code": (
                "accepted"
                if pipeline.platesolve_ok
                else "user_disabled"
                if not platesolve_attempted
                else str(platesolve_diagnostics.get("failure_kind") or "failed")
            ),
            "fallback_used": False,
        },
        "color_calibration": {
            "status": (
                "applied"
                if color_method in color_applied_methods
                else "skipped"
                if intentional_color_skip
                else "failed"
                if requires_review
                else "skipped"
            ),
            "method": color_method,
            "reason_code": color_warning or policy_status,
            "input": stage4_input_stem,
            "output": "stage4_color" if color_saved else None,
            "fallback_used": bool(
                pcc_fallback_used or broadband_local_fallback_used
            ),
        },
        "artistic_hoo": {
            "status": (
                "applied"
                if artistic_hoo_report.get("status") == "accepted"
                else "failed"
                if artistic_hoo_report.get("status")
                == "accepted_output_save_failed"
                else "skipped"
            ),
            "method": "HOO_DETERMINISTIC_MAPPING" if physical_color_input else None,
            "reason_code": artistic_hoo_report.get("status", "not_applicable"),
            "input": artistic_hoo_report.get("physical_parent"),
            "output": artistic_hoo_report.get("output"),
            "fallback_used": False,
            "feeds_main_pipeline": False,
        },
    }
    pipeline._stage4_color_review_required = bool(requires_review)
    pipeline.color_calibration_report = {
        "stage": "stage4_color",
        "input": stage4_input_stem,
        "platesolve": {
            "attempted": platesolve_attempted,
            "ok": bool(pipeline.platesolve_ok),
            "command": platesolve_command,
            "attempts": platesolve_attempts,
            "diagnostics": platesolve_diagnostics,
            "output": "stage4_psolved.fit",
            "instrument_geometry": stage4_geometry,
            "device_geometry_report": device_geometry_report,
            "input_metadata": stage4_metadata,
        },
        "method": color_method,
        "target_aware_color_mapping": bool(target_aware_color),
        "channel_policy": channel_policy,
        "input_classification": input_classification,
        "physical_calibration_allowed": bool(spcc_allowed or pcc_allowed),
        "photometric_calibration_allowed": bool(spcc_allowed or pcc_allowed),
        "requires_review": bool(requires_review),
        "global_white_balance": {"applied": False, "prohibited": True},
        "spcc": {
            "enabled": _stage4_spcc_runtime_enabled(pipeline),
            "role": "primary_physical_calibration",
            "catalog": selected_spcc_catalog,
            "catalog_policy": "local_gaia_xp_sampled_preferred_then_online_gaia",
            "local_catalog": local_spcc_catalog,
            "metadata_database": spcc_database,
            "max_attempts": 1,
            "timeout_sec": (
                int(spcc_attempts[0].get("timeout_sec", PCC_TIMEOUT_DEFAULT_SEC))
                if spcc_attempts
                else int(
                    getattr(
                        pipeline.cfg,
                        "stage4_spcc_timeout_sec",
                        PCC_TIMEOUT_DEFAULT_SEC,
                    )
                )
            ),
            "parameters": spcc_parameters,
            "attempts": spcc_attempts,
            "network_enabled": _stage4_network_enabled(),
            "quality_gate": spcc_quality_report,
            "rollback": spcc_rollback_report,
        },
        "pcc": {
            "role": "exception_fallback_broadband_only",
            "used": bool(pcc_fallback_used),
            "catalog": selected_pcc_catalog,
            "catalog_policy": "local_gaia_preferred_then_online_gaia",
            "local_catalog": local_pcc_catalog,
            "max_attempts": 1,
            "timeout_sec": (
                int(pcc_attempts[0].get("timeout_sec", PCC_TIMEOUT_DEFAULT_SEC))
                if pcc_attempts
                else PCC_TIMEOUT_DEFAULT_SEC
            ),
            "policy_status": policy_status,
            "attempts": pcc_attempts,
            "network_enabled": _stage4_network_enabled(),
            "quality_gate": pcc_quality_report,
            "rollback": rollback_report,
        },
        "local_fallback": local_fallback_report,
        "physical_color": {
            "method": color_method,
            "accepted": color_method in color_applied_methods,
            "output": (
                f"{PHYSICAL_COLOR_STEM}.fit"
                if physical_saved
                else "stage4_color.fit" if color_saved else None
            ),
            "feeds_main_pipeline": True,
        },
        "artistic_hoo": artistic_hoo_report,
        "narrowband_normalization": artistic_hoo_report,
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
                and color_risk_warning
            ),
            "blue_gain_limit": stage4_policy.get("blue_gain_limit"),
            "red_gain_limit": stage4_policy.get("red_gain_limit"),
            "max_allowed_saturation_boost": stage4_policy.get("max_allowed_saturation_boost"),
        },
        "outputs": {
            "psolved": "stage4_psolved.fit",
            "pre_color": f"{PCC_CHECKPOINT_STEM}.fit",
            "pre_pcc": f"{PCC_CHECKPOINT_STEM}.fit",
            "spcc_candidate": (
                f"{SPCC_CANDIDATE_STEM}.fit"
                if spcc_quality_report.get("status") != "not_run"
                else None
            ),
            "pcc_candidate": (
                f"{PCC_CANDIDATE_STEM}.fit"
                if pcc_quality_report.get("status") != "not_run"
                else None
            ),
            "physical_color": (
                f"{PHYSICAL_COLOR_STEM}.fit" if physical_saved else None
            ),
            "artistic_hoo": artistic_hoo_report.get("output"),
            "color": "stage4_color.fit",
            "legacy_read_aliases": ["stage4_colorbalanced.fit"],
        },
        "messages": messages,
        "fallback_used": stage_fallback_used,
        "components": components,
    }
    pipeline._write_stage_json("color_calibration_report.json", pipeline.color_calibration_report)

    elapsed = pipeline.log.stage_end(stage_label)
    reason_code = (
        "stage4_input_checkpoint_unavailable"
        if not checkpoint_loaded
        else "stage4_output_save_failed"
        if not color_saved
        else "stage4_hard_degraded"
        if hard_degraded
        else "color_calibration_review_required"
        if requires_review
        else "target_profiler_fallback"
        if profile_fallback_used
        else "broadband_local_star_fallback"
        if broadband_local_fallback_used
        else "spcc_exception_pcc_fallback"
        if pcc_fallback_used
        else "color_not_applicable_mono"
        if color_method == "SKIPPED_MONO"
        else "color_preserved_by_policy"
        if color_method == "PRESERVE_INPUT"
        else ""
    )
    pipeline._record_stage(
        stage_label,
        status,
        elapsed,
        "；".join(messages),
        fallback_used=stage_fallback_used,
        reason_code=reason_code,
        components=components,
    )
