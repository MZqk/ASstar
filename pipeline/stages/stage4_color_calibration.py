"""Stage 4 plate solving and color calibration."""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

import bright_core_color
import stage4_evidence
import stage7_quality
from channel_semantics import (
    channel_shape_dict,
    classify_channel_semantics,
)
from device_geometry import (
    activate_device_geometry_report,
    build_device_geometry_report,
    resolve_device_report_identity,
    resolve_smart_device_profile,
    resolve_spcc_sensor_from_metadata,
    smart_device_wide_path_reason,
    validate_active_geometry,
)
from models import PipelineStage
from narrowband_normalization import (
    normalize_dual_narrowband_candidate,
    resolve_dual_narrowband_mapping,
    select_filter_header_evidence,
    validate_narrowband_channel_mapping,
)
from stage4_auto_reference import (
    AUTO_REFERENCE_SCHEMA,
    BACKGROUND_METHOD as AUTO_BACKGROUND_METHOD,
    WHITE_REGION_METHOD as AUTO_WHITE_REGION_METHOD,
    evaluate_auto_local_reference,
)
from sirilpy.exceptions import CommandError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_OSC_SENSOR = "Sony IMX585"
DEFAULT_OSC_FILTER_LP = "ZWO Seestar LP"
DEFAULT_OSC_FILTER_NO_FILTER = "No filter"
DEFAULT_OSC_FILTER_UV_IR = "UV/IR Block"
DWARF_MINI_ASTRO_FILTER = "Dwarf Mini Astro"
DEFAULT_SPCC_WHITE_REF = "Average Spiral Galaxy"
DEFAULT_SPCC_LIMITMAG = "10.5"
SPCC_SEED_MARKER_NAME = ".starun-spcc-seed"
SPCC_METADATA_FILES = (
    ("osc_sensor", DEFAULT_OSC_SENSOR, "osc_sensors/Sony_IMX585.json", True),
    ("osc_sensor", "Sony IMX415", "osc_sensors/Sony_IMX415.json", False),
    ("osc_sensor", "Sony IMX462", "osc_sensors/Sony_IMX462.json", False),
    ("osc_sensor", "Sony IMX662", "osc_sensors/Sony_IMX662.json", False),
    ("osc_sensor", "Sony IMX678", "osc_sensors/Sony_IMX678.json", False),
    (
        "osc_sensor",
        "ZWO Seestar S30",
        "osc_sensors/ZWO_Seestar_S30.json",
        False,
    ),
    (
        "osc_sensor",
        "ZWO Seestar S50",
        "osc_sensors/ZWO_Seestar_S50.json",
        False,
    ),
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
        "osc_filter",
        DEFAULT_OSC_FILTER_UV_IR,
        "osc_filters/UV-IR-Block.json",
        False,
    ),
    (
        "osc_filter",
        DWARF_MINI_ASTRO_FILTER,
        "osc_filters/DWARFLAB_Dwarf_Mini_Astro.json",
        False,
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
SPCC_TIMEOUT_DEFAULT_SEC = 300
SPCC_TIMEOUT_MIN_SEC = 5
SPCC_TIMEOUT_MAX_SEC = 300
SPCC_ONLINE_UNVERIFIED_TIMEOUT_DEFAULT_SEC = 300
SPCC_ONLINE_UNVERIFIED_TIMEOUT_MIN_SEC = 30
SPCC_ONLINE_UNVERIFIED_TIMEOUT_MAX_SEC = 300
SPCC_ONLINE_CIRCUIT_ENV = "STARUN_STAGE4_SPCC_ONLINE_CIRCUIT_OPEN"
SPCC_OPERATIONAL_CACHE_ENV = "STARUN_STAGE4_SPCC_OPERATIONAL_CACHE_STATUS"
SPCC_OPERATIONAL_CACHE_KEY_ENV = "STARUN_STAGE4_SPCC_OPERATIONAL_CACHE_KEY"
PCC_CHECKPOINT_STEM = "stage4_pre_pcc"
PCC_CANDIDATE_STEM = "stage4_pcc_candidate"
SPCC_CANDIDATE_STEM = "stage4_spcc_candidate"
PHYSICAL_COLOR_STEM = "stage4_physical_color"
HOO_ARTISTIC_STEM = "stage4_hoo_artistic"
LOCAL_SPCC_DIRNAME = "siril_cat1_healpix8_xpsamp"
LOCAL_SPCC_FILE_PATTERN = "siril_cat1_healpix8_xpsamp_*.dat"
LOCAL_SPCC_FILE_PREFIX = "siril_cat1_healpix8_xpsamp_"
LOCAL_SPCC_EXPECTED_CHUNKS = 48
LOCAL_ASTROMETRIC_FILENAME = "siril_cat_healpix8_astro.dat"
LOCAL_ASTROMETRIC_EXPECTED_SIZE_BYTES = 1_521_132_640
MIN_LOCAL_CATALOG_FILE_BYTES = 1024
SPCC_IMPRECISE_LOG_MARKERS = (
    "the photometric color calibration seems to have found an imprecise solution",
    "测光法色彩校准似乎不能精确校准",
)
SPCC_CAPABILITY_SCHEMA = "starun.stage4-spcc-capabilities.v1"
RUNTIME_CAPABILITIES_SCHEMA = "starun.runtime-capabilities.v1"
RUNTIME_COLOR_DECISION_SCHEMA = "starun.stage4-color-capability-decision.v2"
LEGACY_RUNTIME_COLOR_DECISION_SCHEMA = "starun.stage4-color-capability-decision.v1"
RUNTIME_CAPABILITIES_ENV = "STARUN_RUNTIME_CAPABILITIES_MANIFEST"
AUTO_REFERENCE_REPORT_NAME = "stage4_auto_local_reference.json"
SPCC_CAPABILITY_TIMEOUT_SEC = 15
SPCC_LIST_HEADERS = {
    "oscsensor": "OSC Sensors",
    "oscfilter": "OSC Filters",
    "whiteref": "White References",
}
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
UV_IR_FILTER_KEYWORDS = frozenset(
    {
        "uv/ir",
        "uv-ir",
        "uv ir",
        "uvir",
        "ir cut",
        "ir-cut",
        "ircut",
        "astro filter",
        "astro-filter",
        "broadband",
    }
)
LIGHT_POLLUTION_FILTER_KEYWORDS = frozenset(
    {
        "light pollution",
        "anti-light pollution",
        "lp filter",
        "lp-filter",
        "seestar lp",
    }
)
EMISSION_NEBULA_TARGET_TYPES = frozenset(
    {
        "emission_nebula",
        "emission_nebula_widefield",
        "bright_emission_reflection_nebula",
    }
)
def _stage4_network_enabled() -> bool:
    raw = os.getenv("STARUN_NETWORK_MODE")
    if raw is None:
        return True
    normalized = raw.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in ENV_TRUE_VALUES:
        return True
    return True


def _stage4_local_astrometric_catalog_path(pipeline) -> Path:
    configured = (
        getattr(pipeline, "local_gaia_astro_catalog", None)
        or os.getenv("STARUN_GAIA_ASTRO_CATALOG", "")
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
        or os.getenv("STARUN_GAIA_PHOTO_CATALOG", "")
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
        "available": int(size) == LOCAL_ASTROMETRIC_EXPECTED_SIZE_BYTES,
        "size_bytes": int(size),
        "expected_size_bytes": LOCAL_ASTROMETRIC_EXPECTED_SIZE_BYTES,
    }


def _stage4_local_spcc_catalog_status(pipeline) -> Dict[str, Any]:
    catalog_dir = _stage4_local_spcc_catalog_dir(pipeline)
    expected_names = {
        f"{LOCAL_SPCC_FILE_PREFIX}{index}.dat"
        for index in range(LOCAL_SPCC_EXPECTED_CHUNKS)
    }
    try:
        candidates = sorted(catalog_dir.glob(LOCAL_SPCC_FILE_PATTERN))
    except OSError:
        candidates = []
    candidates_by_name = {path.name: path for path in candidates}
    valid_files = [
        candidates_by_name[name]
        for name in sorted(expected_names)
        if name in candidates_by_name
        and _stage4_valid_catalog_file(candidates_by_name[name])
    ]
    valid_names = {path.name for path in valid_files}
    missing_names = sorted(expected_names - set(candidates_by_name))
    invalid_names = sorted(
        name
        for name in expected_names & set(candidates_by_name)
        if name not in valid_names
    )
    try:
        total_bytes = sum(path.stat().st_size for path in valid_files)
    except OSError:
        total_bytes = 0
    return {
        "path": str(catalog_dir),
        "available": not missing_names and not invalid_names,
        "expected_chunk_count": LOCAL_SPCC_EXPECTED_CHUNKS,
        "valid_chunk_count": len(valid_files),
        "valid_chunks": [path.name for path in valid_files],
        "missing_chunks": missing_names,
        "invalid_chunks": invalid_names,
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
    if _stage4_network_enabled():
        return PCC_CATALOG
    if _stage4_pcc_catalog_status(pipeline)["supports_offline_pcc"]:
        return PCC_LOCAL_CATALOG
    return None


def _stage4_preferred_spcc_catalog(pipeline) -> Optional[str]:
    if _stage4_network_enabled():
        return SPCC_CATALOG
    if _stage4_local_spcc_catalog_status(pipeline)["available"]:
        return SPCC_LOCAL_CATALOG
    return None


def _stage4_offline_fallback_mode(pipeline) -> str:
    mode = str(
        getattr(
            pipeline.cfg,
            "stage4_offline_fallback_mode",
            "auto_local_reference",
        )
        or "auto_local_reference"
    ).strip().lower()
    return mode if mode in {"auto_local_reference", "preserve"} else "auto_local_reference"


def _stage4_derived_runtime_decision(pipeline) -> Dict[str, Any]:
    mode = _stage4_offline_fallback_mode(pipeline)
    if _stage4_network_enabled():
        return {
            "schema": RUNTIME_COLOR_DECISION_SCHEMA,
            "status": "online_unverified",
            "route": "current_network_attempt",
            "attempt_policy": "attempt_then_fallback",
            "preflight_advisory_only": True,
            "offline_fallback_mode": mode,
            "astrometric_source": PCC_CATALOG,
            "xp_source": SPCC_CATALOG,
            "spcc_readiness": "online_unverified",
            "spcc_endpoint_evidence": "manifest_unavailable_online_attempt",
            "physical_color_available": None,
            "spcc_available": None,
            "pcc_available": None,
            "auto_local_reference_available": True,
            "preserve_input_available": mode == "preserve",
            "commands": {"platesolve": True, "spcc": True, "pcc": True},
            "skip_photometric_commands": [],
            "requires_review": False,
            "reason_codes": ["online_capability_manifest_unavailable"],
            "trusted": False,
            "source": "derived_online_unverified",
        }

    astro_available = bool(
        _stage4_local_astrometric_catalog_status(pipeline).get("available")
    )
    xp_available = bool(_stage4_local_spcc_catalog_status(pipeline).get("available"))
    if astro_available and xp_available:
        route = "physical_spcc_then_pcc"
        skip: List[str] = []
    elif astro_available:
        route = "physical_pcc_only"
        skip = ["spcc"]
    else:
        route = "auto_local_reference" if mode == "auto_local_reference" else "preserve_input"
        skip = ["platesolve", "spcc", "pcc"]
    return {
        "schema": RUNTIME_COLOR_DECISION_SCHEMA,
        "status": "ready" if astro_available and xp_available else "degraded_allowed",
        "route": route,
        "attempt_policy": "attempt_then_fallback",
        "preflight_advisory_only": True,
        "offline_fallback_mode": mode,
        "astrometric_source": PCC_LOCAL_CATALOG if astro_available else None,
        "xp_source": SPCC_LOCAL_CATALOG if xp_available else None,
        "physical_color_available": astro_available,
        "spcc_available": astro_available and xp_available,
        "spcc_readiness": (
            "local_verified" if astro_available and xp_available else "unavailable"
        ),
        "spcc_endpoint_evidence": (
            "complete_local_catalog" if astro_available and xp_available else None
        ),
        "pcc_available": astro_available,
        "auto_local_reference_available": not astro_available and mode == "auto_local_reference",
        "preserve_input_available": not astro_available and mode == "preserve",
        "commands": {
            "platesolve": astro_available,
            "spcc": astro_available and xp_available,
            "pcc": astro_available,
        },
        "skip_photometric_commands": skip,
        "requires_review": not astro_available,
        "reason_codes": (
            []
            if astro_available and xp_available
            else ["gaia_xp_unavailable_spcc_skipped"]
            if astro_available
            else ["gaia_astrometry_unavailable_physical_color_skipped"]
        ),
        "trusted": True,
        "source": "derived_explicit_offline_configuration",
    }


def _stage4_runtime_color_decision(pipeline) -> Dict[str, Any]:
    configured = str(os.getenv(RUNTIME_CAPABILITIES_ENV, "") or "").strip()
    if not configured:
        return _stage4_derived_runtime_decision(pipeline)
    path = Path(configured).expanduser()
    try:
        resolved = path.resolve()
        work_dir = getattr(pipeline, "work_dir", None)
        if work_dir is not None and resolved.parent != Path(work_dir).resolve():
            raise ValueError("runtime capability manifest is outside the active work directory")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != RUNTIME_CAPABILITIES_SCHEMA:
            raise ValueError("runtime capability manifest schema mismatch")
        if str(payload.get("status") or "") not in {"ready", "degraded_allowed"}:
            raise ValueError("runtime capability manifest is not in a runnable state")
        expected_run_id = str(getattr(pipeline, "_run_id", "") or "")
        actual_run_id = str(payload.get("run_id") or "")
        if not expected_run_id or not actual_run_id:
            raise ValueError("runtime capability manifest run_id is unavailable")
        if actual_run_id != expected_run_id:
            raise ValueError("runtime capability manifest run_id mismatch")
        decisions = payload.get("decisions")
        decision = (
            decisions.get("stage4_color_calibration")
            if isinstance(decisions, dict)
            else None
        )
        if not isinstance(decision, dict):
            raise ValueError("Stage 4 capability decision is missing")
        decision_schema = str(decision.get("schema") or "")
        if decision_schema not in {
            RUNTIME_COLOR_DECISION_SCHEMA,
            LEGACY_RUNTIME_COLOR_DECISION_SCHEMA,
        }:
            raise ValueError("Stage 4 capability decision schema mismatch")
        if str(decision.get("status") or "") not in {"ready", "degraded_allowed"}:
            raise ValueError("Stage 4 capability decision is not final")
        route = str(decision.get("route") or "")
        if route not in {
            "physical_spcc_then_pcc",
            "physical_pcc_only",
            "auto_local_reference",
            "preserve_input",
        }:
            raise ValueError("Stage 4 capability route is invalid")
        commands = decision.get("commands")
        if not isinstance(commands, dict) or not {
            "platesolve",
            "spcc",
            "pcc",
        }.issubset(commands):
            raise ValueError("Stage 4 command decision is incomplete")
        if any(
            not isinstance(commands[name], bool)
            for name in ("platesolve", "spcc", "pcc")
        ):
            raise ValueError("Stage 4 command decision must use booleans")
        result = dict(decision)
        # v1 manifests encoded speculative availability as command bans.  In
        # online mode normalize them to the v2 attempt-first policy, unless a
        # real SPCC timeout cache has explicitly disabled SPCC for this runtime.
        operational_cache = result.get("spcc_operational_cache")
        explicit_local_sources = bool(
            str(result.get("astrometric_source") or "") == PCC_LOCAL_CATALOG
            or str(result.get("xp_source") or "") == SPCC_LOCAL_CATALOG
        )
        if _stage4_network_enabled() and not explicit_local_sources:
            normalized_commands = {
                "platesolve": True,
                "spcc": not isinstance(operational_cache, Mapping),
                "pcc": True,
            }
            result.update(
                schema=RUNTIME_COLOR_DECISION_SCHEMA,
                route=(
                    "physical_pcc_only"
                    if isinstance(operational_cache, Mapping)
                    else "physical_spcc_then_pcc"
                ),
                attempt_policy="attempt_then_fallback",
                preflight_advisory_only=True,
                astrometric_source=PCC_CATALOG,
                xp_source=SPCC_CATALOG,
                commands=normalized_commands,
            )
            commands = normalized_commands
        readiness = str(result.get("spcc_readiness") or "").strip()
        if readiness not in {
            "local_verified",
            "online_unverified",
            "unavailable",
        }:
            readiness = (
                "local_verified"
                if str(result.get("xp_source") or "") == SPCC_LOCAL_CATALOG
                and bool(commands.get("spcc"))
                else "online_unverified"
                if bool(commands.get("spcc"))
                else "unavailable"
            )
        result["spcc_readiness"] = readiness
        result.update(
            trusted=True,
            source="runtime_capabilities_manifest",
            manifest_path=str(resolved),
            manifest_status=payload.get("status"),
        )
        return result
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        fallback = _stage4_derived_runtime_decision(pipeline)
        fallback.update(
            manifest_path=str(path),
            manifest_error=str(error),
        )
        return fallback


def _stage4_runtime_manifest_for_evidence(
    pipeline,
    runtime_decision: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    raw_path = str(runtime_decision.get("manifest_path") or "").strip()
    if not raw_path:
        return None
    try:
        path = Path(raw_path).expanduser().resolve()
        work_dir = Path(getattr(pipeline, "work_dir", "") or "").resolve()
        if path.parent != work_dir:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != RUNTIME_CAPABILITIES_SCHEMA
        ):
            return None
        return payload
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _stage4_solver_catalog_evidence(pipeline) -> List[Dict[str, Any]]:
    catalogs: List[Dict[str, Any]] = []
    for order, catalog in enumerate(_stage4_platesolve_catalogs(pipeline)):
        if catalog == PCC_LOCAL_CATALOG:
            local = _stage4_local_astrometric_catalog_status(pipeline)
            catalogs.append(
                {
                    "id": catalog,
                    "kind": "local_catalog",
                    "order": order,
                    "available": bool(local.get("available")),
                    "evidence_level": "exact_file_size",
                    "size_bytes": local.get("size_bytes"),
                    "expected_size_bytes": local.get("expected_size_bytes"),
                    "skip_reason": (
                        None
                        if local.get("available")
                        else "local_gaia_astrometric_catalog_unavailable"
                    ),
                    "selected": False,
                    "attempted": False,
                    "result": "not_run",
                }
            )
        else:
            network_enabled = _stage4_network_enabled()
            catalogs.append(
                {
                    "id": catalog,
                    "kind": "online_catalog",
                    "order": order,
                    "available": bool(network_enabled),
                    "evidence_level": "runtime_network_policy",
                    "skip_reason": (
                        None if network_enabled else "network_mode_disabled"
                    ),
                    "selected": False,
                    "attempted": False,
                    "result": "not_run",
                }
            )
    return catalogs


def _stage4_write_observer_evidence(
    pipeline,
    *,
    solver_evidence: Dict[str, Any],
    header_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    process_dir = Path(getattr(pipeline, "process_dir", "") or "")
    stage_json = getattr(pipeline, "stage_json", None)
    if isinstance(stage_json, dict):
        # Unit-test/runtime adapters may intentionally virtualize stage JSON.
        # Keep evidence on that same surface rather than bypassing the adapter.
        records: Dict[str, Any] = {}
        for key, filename, payload in (
            (
                "solver_capabilities",
                stage4_evidence.SOLVER_CAPABILITIES_NAME,
                solver_evidence,
            ),
            (
                "filter_header",
                stage4_evidence.FILTER_HEADER_EVIDENCE_NAME,
                header_evidence,
            ),
        ):
            stage4_evidence.validate_evidence_payload(payload)
            pipeline._write_stage_json(filename, payload)
            records[key] = {
                "schema": payload.get("schema"),
                "status": payload.get("status"),
                "path": filename,
                "sha256": stage4_evidence.canonical_sha256(payload),
                "hash_scope": "virtualized_stage_json_payload",
            }
        return records
    return {
        "solver_capabilities": stage4_evidence.write_evidence_artifact(
            process_dir,
            stage4_evidence.SOLVER_CAPABILITIES_NAME,
            solver_evidence,
            log=pipeline.log,
        ),
        "filter_header": stage4_evidence.write_evidence_artifact(
            process_dir,
            stage4_evidence.FILTER_HEADER_EVIDENCE_NAME,
            header_evidence,
            log=pipeline.log,
        ),
    }


def _stage4_spcc_runtime_enabled(pipeline) -> bool:
    configured = bool(
        getattr(
            pipeline.cfg,
            "stage4_spcc_enabled",
            getattr(pipeline.cfg, "spcc_enabled", True),
        )
    )
    raw = os.getenv("STARUN_SPCC_ENABLE")
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
        or os.getenv("STARUN_SPCC_DATABASE_DIR", "")
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


def _stage4_focal_length() -> float:
    return float(os.getenv("STARUN_STAGE4_PLATESOLVE_FOCAL", "160.0"))


def _stage4_pixel_size() -> float:
    return float(os.getenv("STARUN_STAGE4_PLATESOLVE_PIXELSIZE", "2.90"))


def _stage4_platesolve_order() -> str:
    return str(os.getenv("STARUN_STAGE4_PLATESOLVE_ORDER", "3") or "").strip()


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
            os.getenv("STARUN_STAGE4_PLATESOLVE_FOCAL", "160") or "160"
        ).strip()
        pixelsize = str(
            os.getenv("STARUN_STAGE4_PLATESOLVE_PIXELSIZE", "2.90") or "2.90"
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
    raw = os.getenv("STARUN_STAGE4_PLATESOLVE_CATALOGS", "gaia")
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
    radius = str(os.getenv("STARUN_STAGE4_PLATESOLVE_HEADER_RADIUS", "") or "").strip()
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
    header_base = (
        _stage4_header_platesolve_args(metadata or {}, pipeline)
        if bool(
            getattr(
                pipeline.cfg,
                "stage4_header_guided_platesolve_enabled",
                True,
            )
        )
        else ()
    )
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
        return f"online catalog disabled by STARUN_NETWORK_MODE=0: {catalog}"
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
    ini_path = os.getenv("STARUN_SIRIL_CONFIG", "").strip()
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


def _stage4_strip_siril_log_prefix(line: str) -> str:
    value = str(line or "").replace("\x00", "").strip()
    value = re.sub(r"^log:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^(?:\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{6,}):\s*",
        "",
        value,
    )
    return value.strip()


def _stage4_parse_spcc_list_output(output: str, kind: str) -> List[str]:
    """Parse one isolated ``spcc_list`` response without trusting prior logs."""
    expected_header = SPCC_LIST_HEADERS.get(str(kind).strip().lower())
    if not expected_header:
        raise ValueError(f"unsupported SPCC list kind: {kind}")
    collecting = False
    command_seen = False
    runtime_header_skipped = False
    values: List[str] = []
    seen: set[str] = set()
    for raw_line in str(output or "").splitlines():
        line = _stage4_strip_siril_log_prefix(raw_line)
        normalized = re.sub(r"\s+", " ", line).strip()
        folded = normalized.casefold()
        if folded.startswith("running command"):
            if "spcc_list" in folded:
                command_seen = True
                collecting = False
                runtime_header_skipped = False
            elif collecting:
                break
            continue
        if folded == expected_header.casefold():
            collecting = True
            runtime_header_skipped = True
            continue
        if command_seen and not runtime_header_skipped and normalized:
            # Each isolated spcc_list response prints one localized category
            # heading before the stable database labels.  Skipping that first
            # payload line keeps the parser independent of Siril's UI locale.
            runtime_header_skipped = True
            collecting = True
            continue
        if not collecting:
            continue
        if (
            folded.startswith("script execution finished")
            or folded.startswith("total execution time")
            or folded == "closing pipes"
        ):
            break
        if not normalized:
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            values.append(normalized)
    return values


def _stage4_run_spcc_list_once(pipeline, kind: str) -> Dict[str, Any]:
    """Query one runtime SPCC metadata category in a bounded Siril process."""
    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind not in SPCC_LIST_HEADERS:
        return {
            "status": "unavailable",
            "kind": normalized_kind,
            "error": f"unsupported SPCC list kind: {kind}",
        }

    test_runner = getattr(pipeline, "_run_stage4_spcc_list_once", None)
    if callable(test_runner):
        try:
            ok, detail = test_runner(
                kind=normalized_kind,
                timeout_sec=SPCC_CAPABILITY_TIMEOUT_SEC,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            ok, detail = False, str(error)
        output = str(detail or "")
        if not ok:
            return {
                "status": "unavailable",
                "kind": normalized_kind,
                "runner": "test_hook",
                "error": output or "SPCC runtime list probe failed",
            }
        values = _stage4_parse_spcc_list_output(output, normalized_kind)
        return {
            "status": "ok" if values else "unavailable",
            "kind": normalized_kind,
            "runner": "test_hook",
            "values": values,
            "error": None if values else "SPCC runtime list output was unparseable",
        }

    cli = _stage4_resolve_siril_cli()
    process_dir = Path(getattr(pipeline, "process_dir", "") or "")
    if cli is None or not process_dir.is_dir():
        return {
            "status": "unavailable",
            "kind": normalized_kind,
            "runner": "independent_siril_cli",
            "error": (
                "independent siril-cli unavailable"
                if cli is None
                else f"process directory unavailable: {process_dir}"
            ),
        }

    script_path = process_dir / f".stage4_spcc_list_{normalized_kind}.ssf"
    try:
        escaped_dir = str(process_dir).replace('"', '\\"')
        script_path.write_text(
            "\n".join(
                (
                    "requires 1.4.0",
                    f'cd "{escaped_dir}"',
                    f"spcc_list {normalized_kind}",
                    "close",
                )
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        return {
            "status": "unavailable",
            "kind": normalized_kind,
            "runner": "independent_siril_cli",
            "error": f"unable to prepare SPCC list probe: {error}",
        }

    cli_command = [str(cli), "-d", str(process_dir)]
    ini_path = os.getenv("STARUN_SIRIL_CONFIG", "").strip()
    if ini_path:
        cli_command.extend(("-i", ini_path))
    cli_command.extend(("-s", str(script_path)))
    try:
        completed = subprocess.run(
            cli_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=SPCC_CAPABILITY_TIMEOUT_SEC,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "unavailable",
            "kind": normalized_kind,
            "runner": "independent_siril_cli",
            "timeout_sec": SPCC_CAPABILITY_TIMEOUT_SEC,
            "error": "SPCC runtime list probe timed out",
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "status": "unavailable",
            "kind": normalized_kind,
            "runner": "independent_siril_cli",
            "error": str(error),
        }
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    output = completed.stdout or ""
    if completed.returncode != 0:
        output_lines = [line.strip() for line in output.splitlines() if line.strip()]
        return {
            "status": "unavailable",
            "kind": normalized_kind,
            "runner": "independent_siril_cli",
            "exit_code": int(completed.returncode),
            "output_tail": " | ".join(output_lines[-6:]),
            "error": f"siril-cli exit={completed.returncode}",
        }
    values = _stage4_parse_spcc_list_output(output, normalized_kind)
    version_match = re.search(r"Welcome to siril\s+([^\s]+)", output, re.IGNORECASE)
    return {
        "status": "ok" if values else "unavailable",
        "kind": normalized_kind,
        "runner": "independent_siril_cli",
        "siril_version": version_match.group(1) if version_match else None,
        "values": values,
        "error": None if values else "SPCC runtime list output was unparseable",
    }


def _stage4_spcc_runtime_capabilities(
    pipeline,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify requested SPCC metadata against Siril's effective runtime list."""
    requirements = [
        ("oscsensor", "sensor", "osc_sensor"),
        ("whiteref", "white_reference", "white_reference"),
    ]
    if not bool(parameters.get("narrowband")):
        requirements.append(("oscfilter", "osc_filter", "osc_filter"))

    categories: Dict[str, Any] = {}
    blocking_missing: List[str] = []
    unknown: List[str] = []
    for kind, parameter_key, report_key in requirements:
        requested = str(parameters.get(parameter_key) or "").strip()
        probe = _stage4_run_spcc_list_once(pipeline, kind)
        values = [str(value).strip() for value in probe.get("values", []) if str(value).strip()]
        normalized_values = {
            re.sub(r"\s+", " ", value).strip().casefold(): value
            for value in values
        }
        requested_key = re.sub(r"\s+", " ", requested).strip().casefold()
        matched = normalized_values.get(requested_key)
        if probe.get("status") == "ok":
            found: Optional[bool] = bool(matched)
            if not found:
                blocking_missing.append(f"{report_key}={requested or '<unset>'}")
        else:
            found = None
            unknown.append(report_key)
        close_keys = difflib.get_close_matches(
            requested_key,
            list(normalized_values),
            n=3,
            cutoff=0.55,
        )
        categories[report_key] = {
            **probe,
            "requested": requested or None,
            "found": found,
            "matched_label": matched,
            "available_count": len(values),
            "available_values": values,
            "suggestions": [normalized_values[key] for key in close_keys],
        }

    if blocking_missing:
        status = "rejected"
        decision = "reject"
        reason = "runtime_metadata_missing"
    elif unknown:
        status = "unverified" if len(unknown) == len(requirements) else "partially_verified"
        decision = "allow_unverified"
        reason = "runtime_probe_unavailable"
    else:
        status = "verified"
        decision = "allow"
        reason = "all_requested_metadata_available"
    return {
        "schema": SPCC_CAPABILITY_SCHEMA,
        "status": status,
        "decision": decision,
        "reason": reason,
        "policy": "block_only_confirmed_missing",
        "narrowband": bool(parameters.get("narrowband")),
        "blocking_missing": blocking_missing,
        "unverified_requirements": unknown,
        "categories": categories,
    }


class _Stage4SpccRuntimeMetadataMissing(ValueError):
    pass


class _Stage4SpccDeviceMetadataMissing(ValueError):
    pass


def _stage4_run_spcc(
    pipeline,
    *,
    phase: str,
    catalog: str,
    args: Tuple[str, ...],
    narrowband: bool,
    runtime_decision: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Run one SPCC candidate behind the same killable boundary as PCC."""
    configured_timeout_sec = int(
        getattr(
            pipeline.cfg,
            "stage4_spcc_timeout_sec",
            SPCC_TIMEOUT_DEFAULT_SEC,
        )
        or SPCC_TIMEOUT_DEFAULT_SEC
    )
    configured_timeout_sec = max(
        SPCC_TIMEOUT_MIN_SEC,
        min(configured_timeout_sec, SPCC_TIMEOUT_MAX_SEC),
    )
    online_unverified_timeout_sec = int(
        getattr(
            pipeline.cfg,
            "stage4_spcc_online_unverified_timeout_sec",
            SPCC_ONLINE_UNVERIFIED_TIMEOUT_DEFAULT_SEC,
        )
        or SPCC_ONLINE_UNVERIFIED_TIMEOUT_DEFAULT_SEC
    )
    online_unverified_timeout_sec = max(
        SPCC_ONLINE_UNVERIFIED_TIMEOUT_MIN_SEC,
        min(
            online_unverified_timeout_sec,
            SPCC_ONLINE_UNVERIFIED_TIMEOUT_MAX_SEC,
        ),
    )
    catalog = (
        SPCC_LOCAL_CATALOG
        if str(catalog).strip().lower() == SPCC_LOCAL_CATALOG
        else SPCC_CATALOG
    )
    decision = runtime_decision if isinstance(runtime_decision, Mapping) else {}
    declared_readiness = str(decision.get("spcc_readiness") or "").strip()
    if catalog == SPCC_LOCAL_CATALOG:
        spcc_readiness = "local_verified"
        timeout_sec = configured_timeout_sec
        timeout_policy = "configured_localgaia"
    else:
        spcc_readiness = (
            declared_readiness
            if declared_readiness == "online_unverified"
            else "online_unverified"
        )
        timeout_sec = min(
            configured_timeout_sec,
            online_unverified_timeout_sec,
        )
        timeout_policy = "online_unverified_cap"
    command = "spcc " + " ".join(args)
    attempt: Dict[str, Any] = {
        "label": f"catalog:{catalog}",
        "phase": phase,
        "command": command,
        "status": "failed",
        "timeout_sec": timeout_sec,
        "configured_timeout_sec": configured_timeout_sec,
        "online_unverified_timeout_sec": online_unverified_timeout_sec,
        "timeout_policy": timeout_policy,
        "online_unverified_cap_applied": (
            timeout_policy == "online_unverified_cap"
        ),
        "spcc_readiness": spcc_readiness,
        "attempt": 1,
        "max_attempts": 1,
        "narrowband": bool(narrowband),
    }

    if (
        catalog == SPCC_CATALOG
        and str(os.getenv(SPCC_OPERATIONAL_CACHE_ENV, "")).strip().lower()
        == "operational_timeout_cached"
    ):
        cache_key = str(os.getenv(SPCC_OPERATIONAL_CACHE_KEY_ENV, "") or "").strip()
        attempt.update(
            status="skipped",
            error="online SPCC operational timeout is cached for this app session",
            reason_code="operational_timeout_cached",
            operational_cache={
                "status": "operational_timeout_cached",
                "scope": "application_session",
                "cache_key": cache_key or None,
            },
        )
        return (
            False,
            "SPCC skipped: operational timeout cached for this app session",
            [attempt],
        )

    if (
        catalog == SPCC_CATALOG
        and str(os.getenv(SPCC_ONLINE_CIRCUIT_ENV, "")).strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        attempt.update(
            status="skipped",
            error="batch online SPCC timeout circuit is open",
            reason_code="batch_online_timeout_circuit_open",
            circuit_open=True,
        )
        return (
            False,
            "SPCC skipped: batch online timeout circuit is open",
            [attempt],
        )

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
                "advisory_only_reduce_downstream_saturation_budget"
            )
        detail_text = str(detail)
        timed_out = bool(
            not ok
            and re.search(r"(?:^|\b)(?:timeout|timed out)(?:\b|$)", detail_text, re.I)
        )
        attempt["status"] = "ok" if ok else "timeout" if timed_out else "failed"
        if not ok:
            attempt["error"] = detail_text
        return bool(ok), detail_text, [attempt]

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
    ini_path = os.getenv("STARUN_SIRIL_CONFIG", "").strip()
    if ini_path:
        cli_command.extend(("-i", ini_path))
    cli_command.extend(("-s", str(script_path)))
    attempt["runner"] = "independent_siril_cli"
    attempt["cli"] = str(cli)
    pipeline.log.info(
        "SPCC 单次 "
        + ("离线" if catalog == SPCC_LOCAL_CATALOG else "在线")
        + f" Gaia DR3 校色开始（timeout={timeout_sec}s, policy={timeout_policy}）"
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
            "advisory_only_reduce_downstream_saturation_budget"
        )
    if not candidate_path.is_file() or candidate_path.stat().st_size <= 0:
        attempt["error"] = "SPCC candidate output missing"
        return False, f"SPCC {phase} failed: {attempt['error']}", [attempt]

    attempt["status"] = "ok"
    attempt["output"] = candidate_path.name
    return True, command, [attempt]


def _stage4_resolve_siril_cli() -> Optional[Path]:
    configured = os.getenv("STARUN_SIRIL_CLI", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate
    resolved = shutil.which("siril-cli")
    return Path(resolved) if resolved else None


def _stage4_filter_semantics_text(
    pipeline,
    metadata: Dict[str, Any],
    narrowband_mapping: Optional[Mapping[str, Any]] = None,
) -> str:
    profile = getattr(pipeline, "target_profile", None)
    selected: List[Mapping[str, Any]] = []
    mapping_supplied = isinstance(narrowband_mapping, Mapping)
    if mapping_supplied:
        detail = narrowband_mapping.get("evidence_detail") or {}
        frozen_selected = (
            detail.get("selected_filter_headers")
            if isinstance(detail, Mapping)
            else []
        )
        selected = [
            item for item in (frozen_selected or []) if isinstance(item, Mapping)
        ]
    else:
        selection = select_filter_header_evidence(metadata)
        selected = [
            item
            for item in (selection.get("selected_filter_headers") or [])
            if isinstance(item, Mapping)
        ]

    if selected:
        values: List[Any] = [
            item.get("normalized_filter", "") for item in selected
        ]
    else:
        values = [_stage4_explicit_filter_hint(pipeline)]
    if isinstance(profile, dict) and not selected and not mapping_supplied:
        values.extend((profile.get("filter", ""), profile.get("filter_name", "")))
    return " ".join(str(value or "").strip().lower() for value in values)


def _stage4_explicit_filter_hint(pipeline) -> str:
    raw = str(getattr(pipeline.cfg, "stage4_filter_hint", "auto") or "").strip()
    if raw.lower() == "auto":
        raw = str(os.getenv("STARUN_STAGE4_FILTER_HINT", "") or "").strip()
    return {
        "auto": "",
        "no_filter": "broadband no filter",
        "seestar_lp": "broadband Seestar LP",
        "dual_narrowband": "dualband Ha OIII",
    }.get(raw.lower(), raw)


def _stage4_siril_named_arg(name: str, value: str) -> str:
    escaped = str(value or "").replace('"', '\\"')
    return f'"-{name}={escaped}"'


def _stage4_effective_spcc_sensor(
    pipeline,
    metadata: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    configured_raw = getattr(pipeline.cfg, "stage4_spcc_osc_sensor", None)
    configured = str(configured_raw or "").strip()
    if configured:
        source = (
            "environment_override"
            if str(os.getenv("STARUN_STAGE4_SPCC_OSC_SENSOR", "") or "").strip()
            else "explicit_config"
        )
        return configured, source
    if profile and str(profile.get("spcc_sensor") or "").strip():
        return str(profile["spcc_sensor"]).strip(), "smart_device_profile"
    wide_path_reason = smart_device_wide_path_reason(metadata)
    if wide_path_reason:
        raise _Stage4SpccDeviceMetadataMissing(
            "smart-telescope wide camera has no bundled SPCC response: "
            f"{wide_path_reason}"
        )
    sensor, source = resolve_spcc_sensor_from_metadata(metadata)
    if sensor:
        return sensor, str(source or "fits_header")
    raise _Stage4SpccDeviceMetadataMissing(
        "SPCC sensor is unresolved; provide a supported FITS device identity "
        "or STARUN_STAGE4_SPCC_OSC_SENSOR"
    )


def _stage4_effective_spcc_filter(
    pipeline,
    metadata: Dict[str, Any],
    narrowband_mapping: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str]:
    configured_raw = getattr(pipeline.cfg, "stage4_spcc_osc_filter", None)
    configured = str(configured_raw or "").strip()
    if configured:
        return configured, "explicit_config"
    profile = resolve_smart_device_profile(metadata)
    hint = _stage4_filter_semantics_text(
        pipeline,
        metadata,
        narrowband_mapping=narrowband_mapping,
    )
    normalized_hint = re.sub(r"\s+", " ", hint).strip()
    if any(
        keyword in normalized_hint
        for keyword in LIGHT_POLLUTION_FILTER_KEYWORDS
    ) or re.search(r"(?:^|\s)lp(?:$|\s)", normalized_hint):
        selected = (
            str(profile.get("spcc_light_pollution_filter") or "").strip()
            if profile
            else ""
        )
        if profile and not selected:
            raise _Stage4SpccDeviceMetadataMissing(
                f"SPCC light-pollution filter is unsupported for "
                f"{profile.get('instrument')}"
            )
        return selected or DEFAULT_OSC_FILTER_LP, "fits_or_user_lp_hint"
    if any(keyword in hint for keyword in NO_FILTER_KEYWORDS):
        selected = (
            str(profile.get("spcc_clear_filter") or "").strip()
            if profile
            else ""
        )
        return (
            selected or DEFAULT_OSC_FILTER_NO_FILTER,
            "fits_or_user_no_filter_hint",
        )
    if any(keyword in hint for keyword in UV_IR_FILTER_KEYWORDS):
        selected = (
            str(profile.get("spcc_clear_filter") or "").strip()
            if profile
            else ""
        )
        return selected or DEFAULT_OSC_FILTER_UV_IR, "fits_or_user_uv_ir_hint"
    if profile:
        selected = str(profile.get("spcc_default_filter") or "").strip()
        if selected and (
            not normalized_hint
            or normalized_hint in {"astro", "astro mode", "automatic", "auto"}
        ):
            return selected, "smart_device_profile_default"
        if normalized_hint:
            raise _Stage4SpccDeviceMetadataMissing(
                f"SPCC filter hint is unsupported for {profile.get('instrument')}: "
                f"{normalized_hint}"
            )
    raise _Stage4SpccDeviceMetadataMissing(
        "SPCC filter is unresolved; provide FILTER metadata or "
        "STARUN_STAGE4_SPCC_OSC_FILTER"
    )


def _stage4_spcc_args(
    pipeline,
    metadata: Dict[str, Any],
    channel_policy: Dict[str, Any],
    *,
    catalog: str,
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    profile = resolve_smart_device_profile(metadata)
    sensor, sensor_source = _stage4_effective_spcc_sensor(
        pipeline,
        metadata,
        profile,
    )
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
    mapping = channel_policy.get("narrowband_mapping") or getattr(
        pipeline,
        "narrowband_channel_mapping",
        {},
    )
    frozen_mapping = (
        mapping
        if isinstance(mapping, Mapping) and str(mapping.get("schema") or "")
        else None
    )
    narrowband = channel_policy.get("kind") == "narrowband_composite"
    if narrowband:
        minimum = float(
            getattr(pipeline.cfg, "stage4_nbn_mapping_confidence_min", 0.85)
            or 0.85
        )
        mapping_validation = validate_narrowband_channel_mapping(
            mapping,
            confidence_min=minimum,
        )
        if not mapping_validation["valid"]:
            raise ValueError(
                "dual-narrowband wavelengths are not confirmed: "
                + ",".join(mapping_validation["issues"])
            )
        profile_dualband = (
            profile.get("spcc_dualband")
            if isinstance(profile, dict)
            and isinstance(profile.get("spcc_dualband"), dict)
            else {}
        )

        parameter_sources: Dict[str, str] = {}

        def numeric(
            attr: str,
            env_key: str,
            profile_key: str,
            default: float,
            lower: float,
            upper: float,
        ) -> float:
            configured = float(getattr(pipeline.cfg, attr, default) or default)
            profile_value = profile_dualband.get(profile_key)
            use_profile = bool(
                profile_value is not None
                and os.getenv(env_key) is None
                and math.isclose(configured, default, rel_tol=0.0, abs_tol=1e-9)
            )
            raw = float(profile_value if use_profile else configured)
            if not math.isfinite(raw):
                raw = default
            parameter_sources[profile_key] = (
                "smart_device_profile" if use_profile else "config"
            )
            return max(lower, min(raw, upper))

        r_wl = numeric(
            "stage4_spcc_narrowband_r_wavelength_nm",
            "STARUN_STAGE4_SPCC_NB_R_WAVELENGTH_NM",
            "r_wavelength_nm",
            656.28,
            600.0,
            700.0,
        )
        r_bw = numeric(
            "stage4_spcc_narrowband_r_bandwidth_nm",
            "STARUN_STAGE4_SPCC_NB_R_BANDWIDTH_NM",
            "r_bandwidth_nm",
            20.0,
            1.0,
            100.0,
        )
        g_wl = numeric(
            "stage4_spcc_narrowband_g_wavelength_nm",
            "STARUN_STAGE4_SPCC_NB_G_WAVELENGTH_NM",
            "g_wavelength_nm",
            500.70,
            450.0,
            550.0,
        )
        g_bw = numeric(
            "stage4_spcc_narrowband_g_bandwidth_nm",
            "STARUN_STAGE4_SPCC_NB_G_BANDWIDTH_NM",
            "g_bandwidth_nm",
            30.0,
            1.0,
            100.0,
        )
        b_wl = numeric(
            "stage4_spcc_narrowband_b_wavelength_nm",
            "STARUN_STAGE4_SPCC_NB_B_WAVELENGTH_NM",
            "b_wavelength_nm",
            500.70,
            450.0,
            550.0,
        )
        b_bw = numeric(
            "stage4_spcc_narrowband_b_bandwidth_nm",
            "STARUN_STAGE4_SPCC_NB_B_BANDWIDTH_NM",
            "b_bandwidth_nm",
            30.0,
            1.0,
            100.0,
        )
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
            "sensor_source": sensor_source,
            "device_profile_id": profile.get("id") if profile else None,
            "white_reference": white_ref,
            "limit_magnitude": limitmag,
            "narrowband": True,
            "mapping": mapping,
            "wavelengths_nm": {"r": r_wl, "g": g_wl, "b": b_wl},
            "bandwidths_nm": {"r": r_bw, "g": g_bw, "b": b_bw},
            "parameter_sources": parameter_sources,
        }

    osc_filter, filter_reason = _stage4_effective_spcc_filter(
        pipeline,
        metadata,
        narrowband_mapping=frozen_mapping,
    )
    args = common + (
        _stage4_siril_named_arg("oscfilter", osc_filter),
        f"-limitmag={limitmag}",
    )
    return args, {
        "sensor": sensor,
        "sensor_source": sensor_source,
        "device_profile_id": profile.get("id") if profile else None,
        "white_reference": white_ref,
        "limit_magnitude": limitmag,
        "narrowband": False,
        "osc_filter": osc_filter,
        "osc_filter_reason": filter_reason,
    }


def _stage4_channel_policy(
    pipeline,
    metadata: Dict[str, Any],
    *,
    checkpoint_loaded: bool,
    narrowband_mapping: Optional[Dict[str, Any]] = None,
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
        explicit_filter_hint=_stage4_explicit_filter_hint(pipeline),
        target_profile=(
            pipeline.target_profile
            if isinstance(getattr(pipeline, "target_profile", None), dict)
            else None
        ),
        narrowband_mapping=narrowband_mapping,
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
            0.45 if emission_target else 0.40,
        )
        or (0.45 if emission_target else 0.40)
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
    technical_accepted, technical_integrity = (
        _stage4_candidate_technical_integrity(before, after)
    )
    if not technical_accepted:
        return False, {
            "enabled": bool(
                getattr(pipeline.cfg, "stage4_pcc_quality_gate_enabled", True)
            ),
            "deprecated_no_routing_effect": True,
            "routing_effect": "technical_failure",
            "accepted": False,
            "status": "technical_integrity_rejected",
            "technical_integrity": technical_integrity,
            "rejection_reasons": list(
                technical_integrity.get("rejection_reasons") or []
            ),
        }
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
        getattr(pipeline.cfg, "stage4_pcc_channel_gain_ratio_max", 10.0) or 10.0
    )
    gain_ratio_max = max(1.10, min(gain_ratio_max, 10.0))
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
        "deprecated_no_routing_effect": True,
        "routing_effect": "advisory_only",
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


def _stage4_candidate_technical_integrity(
    before: np.ndarray,
    after: np.ndarray,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate only the invariants that make a Siril candidate usable."""
    reasons: List[str] = []
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    if before_array.shape != after_array.shape:
        reasons.append("shape_changed")
    if after_array.size == 0:
        reasons.append("empty_candidate")
    if not np.all(np.isfinite(after_array)):
        reasons.append("non_finite_pixels")
    return not reasons, {
        "accepted": not reasons,
        "routing_effect": "technical_integrity",
        "before_shape": [int(value) for value in before_array.shape],
        "after_shape": [int(value) for value in after_array.shape],
        "finite": bool(np.all(np.isfinite(after_array))),
        "rejection_reasons": reasons,
    }


def _stage4_verified_physical_pcc_global_rebalance(
    pipeline,
    pcc_quality_report: Dict[str, Any],
    core_integrity_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Distinguish a verified global PCC solution from a local core colour defect."""
    profile = getattr(pipeline, "target_profile", None)
    profile = profile if isinstance(profile, dict) else {}
    target_type = str(profile.get("target_type") or "").strip().lower()
    composite_targets = profile.get("composite_targets")
    if not isinstance(composite_targets, list):
        composite_targets = []
    same_type_targets = [
        item
        for item in composite_targets
        if isinstance(item, dict)
        and str(item.get("type") or "").strip().lower() == target_type
    ]
    labels = {
        str(value).strip().lower()
        for value in (profile.get("secondary_labels") or [])
        if str(value).strip()
    }
    triggers = {
        str(value).strip()
        for value in (core_integrity_report.get("trigger_reasons") or [])
        if str(value).strip()
    }
    measurements = pcc_quality_report.get("measurements") or {}
    core_measurements = core_integrity_report.get("measurements") or {}
    fixed_limits = core_integrity_report.get("fixed_limits") or {}
    component_limits = fixed_limits.get("largest_component_ratio") or {}
    local_component_limit = float(component_limits.get("accepted", 0.005) or 0.005)
    local_component_ratio = float(
        core_measurements.get("largest_component_ratio_of_roi", float("inf"))
    )
    broad_platform_ratio = float(
        core_measurements.get("broad_platform_ratio_of_roi", 0.0) or 0.0
    )
    before_spread = float(
        measurements.get("background_channel_spread_before", float("inf"))
    )
    after_spread = float(
        measurements.get("background_channel_spread", float("inf"))
    )
    spread_ratio = float(
        measurements.get("background_channel_spread_improvement_ratio", float("inf"))
    )
    clip_growth = float(measurements.get("highlight_clip_growth", float("inf")))
    dynamic_ratio = float(measurements.get("dynamic_range_ratio", 0.0) or 0.0)
    post_checks = pcc_quality_report.get("post_calibration_checks") or {}
    measurable_post_checks = [
        str(value.get("status") or "")
        for value in post_checks.values()
        if isinstance(value, dict)
    ]
    post_checks_passed = bool(measurable_post_checks) and all(
        status in {"passed", "not_measurable"}
        for status in measurable_post_checks
    )
    finite_after = float(
        (pcc_quality_report.get("after") or {}).get("finite_ratio", 0.0) or 0.0
    )

    checks = {
        "physical_pcc_technical_integrity_passed": bool(
            pcc_quality_report.get(
                "technical_accepted",
                (pcc_quality_report.get("technical_integrity") or {}).get(
                    "accepted",
                    False,
                ),
            )
        ),
        "composite_emission_reflection_scene": bool(
            target_type == "bright_emission_reflection_nebula"
            and len(same_type_targets) >= 2
            and {"emission_red", "reflection_blue"}.issubset(labels)
        ),
        "broad_platform_only_trigger": triggers == {"broad_core_chroma_platform"},
        "local_component_within_hard_limit": bool(
            local_component_ratio <= local_component_limit
        ),
        "global_support": broad_platform_ratio >= 0.50,
        "background_balance_verified": bool(
            before_spread >= 0.60
            and after_spread <= 0.12
            and spread_ratio <= 0.10
        ),
        "highlight_clipping_not_increased": clip_growth <= 0.0,
        "dynamic_range_retained": 0.50 <= dynamic_ratio <= 1.50,
        "finite_candidate": finite_after >= 0.999999,
        "post_calibration_checks_passed": post_checks_passed,
    }
    accepted = all(checks.values())
    return {
        "applicable": bool(
            target_type == "bright_emission_reflection_nebula"
            and len(same_type_targets) >= 2
        ),
        "accepted": bool(accepted),
        "reason_code": (
            "verified_physical_pcc_global_rebalance"
            if accepted
            else "physical_pcc_global_rebalance_not_verified"
        ),
        "checks": checks,
        "measurements": {
            "same_type_composite_target_count": len(same_type_targets),
            "local_component_ratio_of_roi": local_component_ratio,
            "local_component_ratio_limit": local_component_limit,
            "broad_platform_ratio_of_roi": broad_platform_ratio,
            "background_channel_spread_before": before_spread,
            "background_channel_spread": after_spread,
            "background_channel_spread_improvement_ratio": spread_ratio,
            "highlight_clip_growth": clip_growth,
            "dynamic_range_ratio": dynamic_ratio,
        },
    }


def _stage4_narrowband_pcc_signal_preservation(
    before: np.ndarray,
    after: np.ndarray,
    *,
    channel_gain_ratio: float,
) -> Dict[str, Any]:
    """Measure advisory Ha/OIII signal preservation for degraded narrowband PCC."""
    before_rgb = np.asarray(before, dtype=np.float32)
    after_rgb = np.asarray(after, dtype=np.float32)
    limits = {
        "source_signal_saturation_min": 0.02,
        "signal_saturation_retention_min": 0.50,
        "ha_oiii_ratio_drift_max": 0.40,
        "channel_gain_ratio_max": 2.25,
    }
    issues: List[str] = []
    if (
        before_rgb.shape != after_rgb.shape
        or before_rgb.ndim != 3
        or before_rgb.shape[0] != 3
    ):
        return {
            "status": "rejected",
            "accepted": False,
            "applicable": True,
            "limits": limits,
            "metrics": {},
            "issues": ["shape_or_channel_mismatch"],
        }

    height, width = before_rgb.shape[1:]
    stride = max(1, int(math.ceil(max(height, width) / 640.0)))
    before_sample = before_rgb[:, ::stride, ::stride]
    after_sample = after_rgb[:, ::stride, ::stride]
    finite = np.all(np.isfinite(before_sample), axis=0) & np.all(
        np.isfinite(after_sample), axis=0
    )
    luma = (
        0.2126 * before_sample[0]
        + 0.7152 * before_sample[1]
        + 0.0722 * before_sample[2]
    )
    finite_values = luma[finite]
    if finite_values.size < 256:
        return {
            "status": "unavailable",
            "accepted": False,
            "applicable": True,
            "limits": limits,
            "metrics": {"sample_stride": stride},
            "issues": ["insufficient_finite_samples"],
        }
    q40, q60, q98 = (
        float(value) for value in np.quantile(finite_values, (0.40, 0.60, 0.98))
    )
    background = finite & (luma <= q40)
    signal = finite & (luma >= q60) & (luma <= q98)
    if np.count_nonzero(background) < 64 or np.count_nonzero(signal) < 64:
        return {
            "status": "unavailable",
            "accepted": False,
            "applicable": True,
            "limits": limits,
            "metrics": {
                "sample_stride": stride,
                "background_samples": int(np.count_nonzero(background)),
                "signal_samples": int(np.count_nonzero(signal)),
            },
            "issues": ["insufficient_signal_or_background_samples"],
        }

    def signal_metrics(rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
        backgrounds = np.median(rgb[:, background], axis=1)
        spans = np.asarray(
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
        samples = np.maximum(rgb[:, signal] - backgrounds[:, None], 0.0)
        maximum = np.max(samples, axis=0)
        saturation = (maximum - np.min(samples, axis=0)) / np.maximum(
            maximum,
            1e-6,
        )
        ratio = float(spans[0] / math.sqrt(float(spans[1] * spans[2])))
        return spans, float(np.median(saturation)), ratio

    before_spans, before_saturation, ratio_before = signal_metrics(before_sample)
    after_spans, after_saturation, ratio_after = signal_metrics(after_sample)
    saturation_retention = after_saturation / max(before_saturation, 1e-6)
    ratio_drift = abs(ratio_after / max(ratio_before, 1e-6) - 1.0)
    applicable = before_saturation >= limits["source_signal_saturation_min"]
    if applicable and saturation_retention < limits["signal_saturation_retention_min"]:
        issues.append("narrowband_signal_saturation_collapsed")
    if applicable and ratio_drift > limits["ha_oiii_ratio_drift_max"]:
        issues.append("ha_oiii_source_ratio_drift")
    if channel_gain_ratio > limits["channel_gain_ratio_max"]:
        issues.append("narrowband_channel_gain_ratio_exceeded")
    accepted = not issues
    return {
        "status": (
            "accepted"
            if accepted and applicable
            else "not_applicable_low_source_chroma"
            if accepted
            else "rejected"
        ),
        "accepted": accepted,
        "applicable": applicable,
        "policy": "preserve_source_line_separation_before_artistic_mapping",
        "limits": limits,
        "metrics": {
            "sample_stride": stride,
            "background_samples": int(np.count_nonzero(background)),
            "signal_samples": int(np.count_nonzero(signal)),
            "source_channel_spans": [float(value) for value in before_spans],
            "candidate_channel_spans": [float(value) for value in after_spans],
            "source_signal_saturation": before_saturation,
            "candidate_signal_saturation": after_saturation,
            "signal_saturation_retention": saturation_retention,
            "ha_oiii_ratio_before": ratio_before,
            "ha_oiii_ratio_after": ratio_after,
            "ha_oiii_ratio_drift": ratio_drift,
            "channel_gain_ratio": float(channel_gain_ratio),
        },
        "issues": issues,
    }


def _stage4_shape_dict(shape) -> Dict[str, int]:
    return channel_shape_dict(shape)


def _stage4_pixel_scale_arcsec_per_px() -> float:
    return 206.265 * _stage4_pixel_size() / _stage4_focal_length()


def _stage4_image_geometry(
    pipeline,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    identity = resolve_device_report_identity(metadata or {})
    return {
        "instrument": identity["instrument"],
        "instrument_source": identity["instrument_source"],
        "sensor": identity["sensor"],
        "sensor_source": identity["sensor_source"],
        "identity_source": identity["source"],
        "device_profile_id": identity["device_profile_id"],
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
            "Focal length and pixel size remain the resolved tele optical parameters after crop.",
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


def _stage4_write_image_pixels(pipeline, pixels: np.ndarray) -> None:
    lock_factory = getattr(pipeline.siril, "image_lock", None)
    if callable(lock_factory):
        with lock_factory():
            pipeline.siril.set_image_pixeldata(pixels)
        return
    pipeline.siril.set_image_pixeldata(pixels)


def _stage4_empty_auto_reference_report(
    *,
    status: str = "not_run",
    reason: str = "not_evaluated",
) -> Dict[str, Any]:
    return {
        "schema": AUTO_REFERENCE_SCHEMA,
        "status": status,
        "eligibility": {
            "eligible": False,
            "reason": reason,
        },
        "sampling": {},
        "reference_regions": {
            "background": None,
            "white": None,
        },
        "candidates": {},
        "selection": {
            "method": "PRESERVE_INPUT",
            "applied": False,
            "physical_color": False,
            "requires_review": True,
        },
        "transaction": {
            "pixels_written": False,
            "checkpoint": f"{PCC_CHECKPOINT_STEM}.fit",
            "rollback_performed": False,
        },
        "physical_color": {"accepted": False},
        "degraded_color_correction": {"applied": False},
        "requires_review": True,
    }


def _stage4_restore_exact_pre_color(
    pipeline,
    expected_pixels: Optional[np.ndarray],
) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "checkpoint": f"{PCC_CHECKPOINT_STEM}.fit",
        "restored": False,
        "verified_exact": False,
        "source": None,
    }
    try:
        pipeline.cmd_with_check("load", PCC_CHECKPOINT_STEM)
        actual = np.asarray(pipeline.siril.get_image_pixeldata(preview=False))
        exact = bool(
            expected_pixels is None
            or (
                actual.shape == expected_pixels.shape
                and np.array_equal(actual, expected_pixels)
            )
        )
        if exact:
            report.update(
                restored=True,
                verified_exact=expected_pixels is not None,
                source=f"{PCC_CHECKPOINT_STEM}.fit",
            )
            return True, report
        report["checkpoint_mismatch"] = True
    except (
        AttributeError,
        CommandError,
        KeyError,
        OSError,
        SirilError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report["checkpoint_error"] = str(error)

    if expected_pixels is None:
        return False, report
    try:
        _stage4_write_image_pixels(pipeline, np.asarray(expected_pixels).copy())
        actual = np.asarray(pipeline.siril.get_image_pixeldata(preview=False))
        exact = bool(
            actual.shape == expected_pixels.shape
            and np.array_equal(actual, expected_pixels)
        )
        report.update(
            restored=exact,
            verified_exact=exact,
            source="in_memory_pre_color" if exact else None,
        )
        if not exact:
            report["in_memory_error"] = "post-write pixels differ from immutable pre-color"
        return exact, report
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        report["in_memory_error"] = str(error)
        return False, report


def _stage4_apply_auto_reference_candidate(
    pipeline,
    candidate: np.ndarray,
    *,
    expected_pre_color: Optional[np.ndarray],
    report: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    transaction = report.setdefault("transaction", {})
    transaction.update(
        checkpoint=f"{PCC_CHECKPOINT_STEM}.fit",
        pixels_written=False,
        candidate_saved=False,
        rollback_performed=False,
    )
    try:
        _stage4_write_image_pixels(pipeline, candidate)
        actual = np.asarray(pipeline.siril.get_image_pixeldata(preview=False))
        expected_candidate = np.asarray(candidate)
        if actual.shape != expected_candidate.shape:
            raise RuntimeError("auto-reference post-write shape changed")
        if not np.all(np.isfinite(actual)):
            raise RuntimeError("auto-reference post-write contains non-finite pixels")
        if not np.array_equal(actual, expected_candidate):
            raise RuntimeError("auto-reference post-write pixels differ from candidate")
        transaction["pixels_written"] = True
        transaction["verified_exact"] = True
        if not pipeline._save_stage_output("stage4_auto_reference_candidate"):
            raise RuntimeError("stage4_auto_reference_candidate save failed")
        transaction["candidate_saved"] = True
        post_save = np.asarray(
            pipeline.siril.get_image_pixeldata(preview=False)
        )
        if (
            post_save.shape != expected_candidate.shape
            or not np.all(np.isfinite(post_save))
            or not np.array_equal(post_save, expected_candidate)
        ):
            raise RuntimeError(
                "auto-reference pixels changed while saving candidate"
            )
        transaction["post_save_verified_exact"] = True
        transaction["output"] = "stage4_auto_reference_candidate.fit"
        return True, report
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        transaction["apply_error"] = str(error)
        restored, rollback = _stage4_restore_exact_pre_color(
            pipeline,
            expected_pre_color,
        )
        transaction["rollback_performed"] = restored
        transaction["rollback"] = rollback
        report["status"] = "apply_failed_rolled_back" if restored else "apply_failed_rollback_failed"
        report["selection"].update(method="PRESERVE_INPUT", applied=False)
        report["degraded_color_correction"].update(applied=False, method=None)
        return False, report


def _stage4_run_narrowband_normalization(
    pipeline,
    mapping: Dict[str, Any],
) -> tuple[bool, Dict[str, Any], str]:
    report: Dict[str, Any] = {
        "schema": "starun.narrowband-normalization.v1",
        "status": "not_run",
        "accepted": False,
    }
    if not bool(
        getattr(pipeline.cfg, "stage4_narrowband_normalization_enabled", True)
    ):
        report.update(status="disabled", issues=["disabled_by_configuration"])
        return False, report, "narrowband normalization disabled"
    mapping_confidence_min = float(
        getattr(
            pipeline.cfg,
            "stage4_nbn_mapping_confidence_min",
            0.85,
        )
    )
    mapping_validation = validate_narrowband_channel_mapping(
        mapping,
        confidence_min=mapping_confidence_min,
    )
    if not mapping_validation["valid"]:
        report.update(
            status="skipped_unconfirmed_mapping",
            mapping=mapping,
            issues=list(mapping_validation["issues"]),
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
            mapping=mapping,
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


def run_stage4_color_calibration(pipeline) -> None:
    """Stage 4: plate solve, SPCC-first physical color, then bounded fallbacks."""
    stage_label = PipelineStage.COLOR_CALIBRATION.label
    pipeline.log.stage_start(stage_label)
    pipeline._clear_stage_reviews(4)
    status = "ok"
    hard_degraded = False
    requires_review = False
    pipeline._stage4_color_review_required = False
    color_method = "PRESERVE_INPUT"
    degraded_narrowband_pcc_methods = {
        "PCC_NARROWBAND_DEGRADED",
        "PCC_NARROWBAND_DEGRADED_LOCAL_GAIA",
    }
    accepted_spcc_methods = {
        "SPCC",
        "SPCC_LOCAL_GAIA",
        "SPCC_NARROWBAND",
        "SPCC_NARROWBAND_LOCAL_GAIA",
    }
    accepted_pcc_methods = {
        "PCC",
        "PCC_LOCAL_GAIA",
        *degraded_narrowband_pcc_methods,
    }
    color_warning = ""
    color_confidence = 0.0
    messages: List[str] = []
    local_fallback_report: Optional[Dict[str, Any]] = None
    auto_reference_report: Dict[str, Any] = _stage4_empty_auto_reference_report()
    auto_reference_applied = False
    narrowband_normalization_report: Dict[str, Any] = {
        "schema": "starun.narrowband-normalization.v1",
        "status": "not_applicable",
        "accepted": False,
    }
    pcc_attempts: List[Dict[str, Any]] = []
    pcc_quality_report: Dict[str, Any] = {
        "enabled": True,
        "accepted": False,
        "technical_accepted": False,
        "diagnostic_quality_accepted": False,
        "legacy_quality_accepted": False,
        "status": "not_run",
    }
    rollback_report: Dict[str, Any] = {"required": False, "restored": False}
    spcc_attempts: List[Dict[str, Any]] = []
    spcc_quality_report: Dict[str, Any] = {
        "enabled": True,
        "accepted": False,
        "technical_accepted": False,
        "diagnostic_quality_accepted": False,
        "legacy_quality_accepted": False,
        "status": "not_run",
    }
    bright_core_color_integrity: Dict[str, Any] = {
        "schema": bright_core_color.SCHEMA,
        "applicable": False,
        "status": "not_evaluated",
        "accepted": True,
        "repaired": False,
        "final_action": "not_evaluated",
        "trigger_reasons": [],
    }
    pcc_bright_core_color_integrity: Dict[str, Any] = {
        "schema": bright_core_color.SCHEMA,
        "applicable": False,
        "status": "not_evaluated",
        "accepted": True,
        "final_action": "not_evaluated",
        "trigger_reasons": [],
    }
    spcc_parameters: Dict[str, Any] = {}
    spcc_runtime_capabilities: Dict[str, Any] = {
        "schema": SPCC_CAPABILITY_SCHEMA,
        "status": "not_run",
        "decision": "not_evaluated",
        "reason": "spcc_preflight_not_reached",
        "policy": "block_only_confirmed_missing",
        "blocking_missing": [],
        "unverified_requirements": [],
        "categories": {},
    }
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
    main_output_blocked = False
    physical_main_restore_report: Dict[str, Any] = {
        "required": False,
        "restored": False,
        "source": None,
        "fallback_used": False,
    }
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}
    stage4_input_stem = "stage3_bgremoved"
    checkpoint_loaded = False
    runtime_color_decision: Dict[str, Any] = {}
    stage4_solver_evidence: Dict[str, Any] = {}
    stage4_header_evidence: Dict[str, Any] = {}
    stage4_evidence_artifacts: Dict[str, Any] = {
        "solver_capabilities": {
            "schema": stage4_evidence.SOLVER_CAPABILITIES_SCHEMA,
            "status": "unavailable",
            "path": None,
            "sha256": None,
        },
        "filter_header": {
            "schema": stage4_evidence.FILTER_HEADER_EVIDENCE_SCHEMA,
            "status": "unavailable",
            "path": None,
            "sha256": None,
        },
    }

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
    channel_mapping = resolve_dual_narrowband_mapping(
        stage4_metadata,
        filter_hint=_stage4_explicit_filter_hint(pipeline),
    )
    pipeline.narrowband_channel_mapping = channel_mapping
    channel_policy = _stage4_channel_policy(
        pipeline,
        stage4_metadata,
        checkpoint_loaded=checkpoint_loaded,
        narrowband_mapping=channel_mapping,
    )
    pipeline.channel_profile = dict(channel_policy)
    pipeline._channel_semantics = str(channel_policy["kind"])
    pipeline._write_stage_json(
        "stage4_channel_mapping.json",
        channel_mapping,
    )
    messages.append(
        "channel_mapping="
        f"{channel_mapping.get('mapping')} "
        f"confidence={float(channel_mapping.get('confidence', 0.0)):.2f} "
        f"evidence={channel_mapping.get('evidence')}"
    )
    messages.append(
        "channel_semantics="
        f"{pipeline._channel_semantics} "
        f"confidence={float(channel_policy['confidence']):.2f}"
    )
    runtime_color_decision = _stage4_runtime_color_decision(pipeline)
    messages.append(
        "stage4_runtime_route="
        f"{runtime_color_decision.get('route', 'unknown')} "
        f"source={runtime_color_decision.get('source', 'unknown')}"
    )
    stage4_geometry = _stage4_image_geometry(pipeline, stage4_metadata)
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
            "schema": "starun.device-geometry.v1",
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
        f"{stage4_geometry.get('instrument', 'unknown')} "
        f"sensor={stage4_geometry.get('sensor', 'unknown')}, "
        f"identity_source={stage4_geometry.get('identity_source', 'unknown')}, "
        f"focal={float(active_geometry.get('focal_length_mm', _stage4_focal_length())):g}mm, "
        f"pixelsize={float(active_geometry.get('pixel_size_um', _stage4_pixel_size())):g}um, "
        f"shape={stage4_geometry.get('current_shape')}"
    )

    try:
        runtime_manifest_evidence = _stage4_runtime_manifest_for_evidence(
            pipeline,
            runtime_color_decision,
        )
        stage4_solver_evidence = stage4_evidence.build_solver_capabilities(
            runtime_decision=runtime_color_decision,
            runtime_manifest=runtime_manifest_evidence,
            configured=bool(
                getattr(pipeline.cfg, "stage4_platesolve_enabled", True)
            ),
            catalogs=_stage4_solver_catalog_evidence(pipeline),
            processing_mode=str(
                getattr(pipeline.cfg, "stage4_processing_mode", "auto")
            ),
        )
        stage4_header_evidence = stage4_evidence.build_filter_header_evidence(
            process_dir=Path(pipeline.process_dir),
            source_file=getattr(pipeline, "source_file", None),
            task_run_manifest=getattr(
                pipeline,
                "_task_run_manifest_payload",
                None,
            ),
            stage4_metadata=stage4_metadata,
            filter_selection=select_filter_header_evidence(stage4_metadata),
            channel_mapping=channel_mapping,
            explicit_filter_hint=_stage4_explicit_filter_hint(pipeline),
            device_geometry_report=device_geometry_report,
            header_guided_enabled=bool(
                getattr(
                    pipeline.cfg,
                    "stage4_header_guided_platesolve_enabled",
                    True,
                )
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        pipeline.log.debug(f"Stage4 observer evidence unavailable: {error}")
        stage4_solver_evidence = {
            "schema": stage4_evidence.SOLVER_CAPABILITIES_SCHEMA,
            "status": "unavailable",
            "evidence_only": True,
            "reason": "evidence_builder_failed",
        }
        stage4_header_evidence = {
            "schema": stage4_evidence.FILTER_HEADER_EVIDENCE_SCHEMA,
            "status": "unavailable",
            "evidence_only": True,
            "reason": "evidence_builder_failed",
            "limitations": list(stage4_evidence.HEADER_LIMITATIONS),
        }

    if str(getattr(pipeline.cfg, "stage4_processing_mode", "auto")) == "preserve":
        pipeline.platesolve_ok = False
        ps_saved = pipeline._save_stage_output("stage4_psolved")
        color_saved = pipeline._save_stage_output("stage4_color")
        try:
            stage4_evidence.capture_solver_candidate(
                stage4_header_evidence,
                Path(pipeline.process_dir) / "stage4_psolved.fit",
                platesolve_ok=False,
                output_saved=bool(ps_saved),
            )
            stage4_header_evidence = (
                stage4_evidence.finalize_filter_header_evidence(
                    stage4_header_evidence,
                    final_path=Path(pipeline.process_dir)
                    / "stage4_psolved.fit",
                    final_output_saved=bool(ps_saved),
                    processing_mode="preserve",
                    platesolve_attempted=False,
                    platesolve_ok=False,
                    device_geometry_report=device_geometry_report,
                )
            )
            stage4_solver_evidence = (
                stage4_evidence.finalize_solver_capabilities(
                    stage4_solver_evidence,
                    attempts=(),
                    platesolve_attempted=False,
                    platesolve_ok=False,
                    skip_reason="user_preserve",
                )
            )
            stage4_evidence_artifacts = _stage4_write_observer_evidence(
                pipeline,
                solver_evidence=stage4_solver_evidence,
                header_evidence=stage4_header_evidence,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            pipeline.log.debug(
                f"Stage4 preserve observer evidence unavailable: {error}"
            )
        requires_review = not checkpoint_loaded
        auto_reference_report = _stage4_empty_auto_reference_report(
            status="not_applicable",
            reason="stage4_processing_mode_preserve",
        )
        auto_reference_report["runtime_capability_decision"] = runtime_color_decision
        pipeline._write_stage_json(
            AUTO_REFERENCE_REPORT_NAME,
            auto_reference_report,
        )
        pipeline._stage4_color_review_required = requires_review
        pipeline.color_calibration_report = {
            "stage": "stage4_color",
            "status": "ok" if color_saved else "degraded",
            "execution": "safe_passthrough",
            "reason_code": "user_preserve",
            "input": stage4_input_stem,
            "output": "stage4_color" if color_saved else None,
            "method": "PRESERVE_INPUT",
            "bright_core_color_integrity": bright_core_color_integrity,
            "requires_review": requires_review,
            "runtime_capability_decision": runtime_color_decision,
            "solver_capabilities": stage4_evidence_artifacts[
                "solver_capabilities"
            ],
            "filter_header_evidence": stage4_evidence_artifacts[
                "filter_header"
            ],
            "auto_local_reference": auto_reference_report,
            "physical_color": {"accepted": False, "output": None},
            "degraded_color_correction": {
                "applied": False,
                "method": None,
                "physical_color": False,
                "requires_review": requires_review,
            },
            "channel_mapping": channel_mapping,
            "channel_policy": channel_policy,
            "platesolve": {
                "attempted": False,
                "ok": False,
                "reason_code": "user_preserve",
                "output": "stage4_psolved.fit" if ps_saved else None,
                "instrument_geometry": stage4_geometry,
                "device_geometry_report": device_geometry_report,
            },
            "components": {
                "platesolve": {
                    "status": "skipped",
                    "reason_code": "user_preserve",
                    "fallback_used": False,
                },
                "color_calibration": {
                    "status": "skipped" if color_saved else "failed",
                    "method": "PRESERVE_INPUT",
                    "reason_code": "user_preserve",
                    "fallback_used": False,
                },
            },
            "outputs": {
                "auto_local_reference": AUTO_REFERENCE_REPORT_NAME,
                "solver_capabilities": stage4_evidence_artifacts[
                    "solver_capabilities"
                ].get("path"),
                "filter_header_evidence": stage4_evidence_artifacts[
                    "filter_header"
                ].get("path"),
                "psolved": "stage4_psolved.fit" if ps_saved else None,
                "color": "stage4_color.fit" if color_saved else None,
            },
        }
        pipeline._write_stage_json(
            "color_calibration_report.json",
            pipeline.color_calibration_report,
        )
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            "ok" if color_saved else "degraded",
            elapsed,
            "用户选择保留输入颜色；已跳过 Plate Solve 与 SPCC/PCC",
            execution="safe_passthrough",
            reason_code="user_preserve",
            details={
                "output": "stage4_color" if color_saved else None,
                "requires_review": requires_review,
            },
            components=pipeline.color_calibration_report["components"],
        )
        return

    pipeline.platesolve_ok = False
    platesolve_attempted = True
    platesolve_skip_reason = ""
    platesolve_command = "platesolve " + " ".join(
        _stage4_platesolve_args(pipeline)
    )
    platesolve_attempts: List[Dict[str, str]] = []
    runtime_commands = runtime_color_decision.get("commands")
    runtime_commands = runtime_commands if isinstance(runtime_commands, dict) else {}
    runtime_platesolve_allowed = bool(runtime_commands.get("platesolve", True))
    if bool(getattr(pipeline.cfg, "stage4_platesolve_enabled", True)) and runtime_platesolve_allowed:
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
        if not runtime_platesolve_allowed:
            platesolve_skip_reason = "runtime_gaia_astrometry_unavailable"
            messages.append(
                "runtime capability decision skipped platesolve; "
                "stage4_psolved will mirror input"
            )
            pipeline.log.warn(
                "Stage4 已确认 Gaia 定位来源不可用，跳过 Plate Solve/SPCC/PCC"
            )
        else:
            platesolve_skip_reason = "user_disabled"
            messages.append("platesolve disabled by config; stage4_psolved will mirror input")
            pipeline.log.warn("Stage4 platesolve disabled by config")

    ps_saved = pipeline._save_stage_output("stage4_psolved")
    try:
        stage4_evidence.capture_solver_candidate(
            stage4_header_evidence,
            Path(pipeline.process_dir) / "stage4_psolved.fit",
            platesolve_ok=bool(pipeline.platesolve_ok),
            output_saved=bool(ps_saved),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        pipeline.log.debug(
            f"Stage4 solver-candidate header evidence unavailable: {error}"
        )
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
            "narrowband_composite": (
                "spcc_narrowband_physical_then_degraded_pcc_plus_"
                "isolated_hoo_artistic"
            ),
            "mono": "skip_all_color_calibration",
            "nonlinear_color": "preserve_input",
            "unknown": "preserve_input_review",
        }.get(pipeline._channel_semantics, "preserve_input_review"),
    }
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
    before_pixels_native: Optional[np.ndarray] = None
    restore_before_pixels: Optional[Callable[[np.ndarray], np.ndarray]] = None
    try:
        before_pixels = pipeline.siril.get_image_pixeldata(preview=False)
        before_pixels_native = np.asarray(before_pixels).copy()
        before_chw, restore_before_pixels = _stage4_image_as_chw(before_pixels)
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
    runtime_spcc_allowed = bool(runtime_commands.get("spcc", True))
    runtime_pcc_allowed = bool(runtime_commands.get("pcc", True))
    if not runtime_spcc_allowed:
        selected_spcc_catalog = None
    elif runtime_color_decision.get("xp_source") == SPCC_LOCAL_CATALOG:
        selected_spcc_catalog = SPCC_LOCAL_CATALOG
    elif runtime_color_decision.get("xp_source") == SPCC_CATALOG:
        selected_spcc_catalog = SPCC_CATALOG
    if not runtime_pcc_allowed:
        selected_pcc_catalog = None
    elif runtime_color_decision.get("astrometric_source") == PCC_LOCAL_CATALOG:
        selected_pcc_catalog = PCC_LOCAL_CATALOG
    elif runtime_color_decision.get("astrometric_source") == PCC_CATALOG:
        selected_pcc_catalog = PCC_CATALOG
    physical_color_input = channel_policy["kind"] in {
        "broadband_rgb_osc",
        "narrowband_composite",
    }
    failure_action = str(
        getattr(pipeline.cfg, "stage4_failure_action", "auto_fallback")
    )
    pcc_fallback_enabled = bool(
        getattr(pipeline.cfg, "stage4_pcc_fallback_enabled", True)
    )
    narrowband_degraded_pcc_enabled = bool(
        getattr(
            pipeline.cfg,
            "stage4_narrowband_degraded_pcc_enabled",
            True,
        )
    )
    spcc_allowed = bool(
        physical_color_input
        and _stage4_spcc_runtime_enabled(pipeline)
        and pipeline.platesolve_ok
        and pre_pcc_saved
        and before_chw is not None
        and runtime_spcc_allowed
        and selected_spcc_catalog is not None
    )
    pcc_allowed = bool(
        physical_color_input
        and pcc_fallback_enabled
        and (
            channel_policy["kind"] != "narrowband_composite"
            or narrowband_degraded_pcc_enabled
        )
        and failure_action == "auto_fallback"
        and pipeline.platesolve_ok
        and pre_pcc_saved
        and before_chw is not None
        and runtime_pcc_allowed
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
        if not spcc_allowed:
            operational_cache = runtime_color_decision.get(
                "spcc_operational_cache"
            )
            if not _stage4_spcc_runtime_enabled(pipeline):
                spcc_skip_reason = "disabled by config/runtime preflight"
            elif isinstance(operational_cache, Mapping):
                spcc_skip_reason = "operational_timeout_cached"
                spcc_attempts = [
                    {
                        "label": "catalog:gaia",
                        "phase": (
                            "linear_dual_narrowband_physical"
                            if narrowband_physical
                            else "linear_broadband"
                        ),
                        "status": "skipped",
                        "reason_code": "operational_timeout_cached",
                        "spcc_readiness": "online_unverified",
                        "operational_cache": dict(operational_cache),
                    }
                ]
            elif not runtime_spcc_allowed:
                spcc_skip_reason = "runtime capability decision marked Gaia XP/SPCC unavailable"
            elif not pipeline.platesolve_ok:
                spcc_skip_reason = "plate solve unavailable"
            elif selected_spcc_catalog is None:
                spcc_skip_reason = (
                    "explicit offline mode requires a valid local Gaia DR3 "
                    "xp_sampled catalogue"
                )
            else:
                spcc_skip_reason = "immutable pre-color source unavailable"
            spcc_runtime_capabilities.update(
                status=(
                    "cached_unavailable"
                    if isinstance(operational_cache, Mapping)
                    else "not_run"
                ),
                decision=(
                    "skip"
                    if isinstance(operational_cache, Mapping)
                    else "not_evaluated"
                ),
                reason=spcc_skip_reason,
                operational_cache=(
                    dict(operational_cache)
                    if isinstance(operational_cache, Mapping)
                    else None
                ),
            )
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
                spcc_runtime_capabilities = _stage4_spcc_runtime_capabilities(
                    pipeline,
                    spcc_parameters,
                )
                spcc_database["runtime_capabilities"] = spcc_runtime_capabilities
                if spcc_runtime_capabilities.get("decision") == "reject":
                    spcc_runtime_capabilities["routing_effect"] = "advisory_only"
                    messages.append(
                        "SPCC runtime metadata probe reported missing values; "
                        "real Siril command will determine fallback"
                    )
            except _Stage4SpccDeviceMetadataMissing as error:
                color_warning = "spcc_command_preparation_failed"
                spcc_attempts = [{
                    "label": "command_preparation",
                    "phase": "spcc",
                    "status": "failed",
                    "reason_code": "command_preparation_failed",
                    "error": str(error),
                }]
                spcc_quality_report.update(
                    status="command_preparation_failed",
                    rejection_reasons=["spcc_device_metadata_unresolved"],
                    routing_effect="technical_failure",
                )
                spcc_database["device_resolution"] = {
                    "status": "rejected",
                    "reason": str(error),
                }
                messages.append(f"SPCC command preparation failed: {error}")
            except _Stage4SpccRuntimeMetadataMissing as error:
                color_warning = "spcc_command_preparation_failed"
                spcc_attempts = [{
                    "label": "command_preparation",
                    "phase": "spcc",
                    "status": "failed",
                    "reason_code": "command_preparation_failed",
                    "error": str(error),
                }]
                spcc_quality_report.update(
                    status="command_preparation_failed",
                    rejection_reasons=["spcc_runtime_metadata_missing"],
                    runtime_capabilities=spcc_runtime_capabilities,
                    routing_effect="technical_failure",
                )
                messages.append(f"SPCC command preparation failed: {error}")
            except (TypeError, ValueError) as error:
                color_warning = "spcc_command_preparation_failed"
                spcc_attempts = [{
                    "label": "command_preparation",
                    "phase": "spcc",
                    "status": "failed",
                    "reason_code": "command_preparation_failed",
                    "error": str(error),
                }]
                spcc_quality_report.update(
                    status="command_preparation_failed",
                    rejection_reasons=["spcc_command_preparation_failed"],
                    routing_effect="technical_failure",
                    error=str(error),
                )
                messages.append(f"SPCC command preparation failed: {error}")
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
                    runtime_decision=runtime_color_decision,
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
                        legacy_quality_accepted, spcc_quality_report = _stage4_pcc_quality_gate(
                            before_chw,
                            candidate_chw,
                            pipeline,
                        )
                        technical_accepted, technical_integrity = (
                            _stage4_candidate_technical_integrity(
                                before_chw,
                                candidate_chw,
                            )
                        )
                        spcc_quality_report.update(
                            accepted=bool(technical_accepted),
                            technical_accepted=bool(technical_accepted),
                            diagnostic_quality_accepted=bool(
                                legacy_quality_accepted
                            ),
                            legacy_quality_accepted=bool(legacy_quality_accepted),
                            routing_effect=(
                                "advisory_only"
                                if technical_accepted
                                else "technical_failure"
                            ),
                            technical_integrity=technical_integrity,
                        )
                        accepted = bool(technical_accepted)
                        if technical_accepted:
                            (
                                bright_core_color_integrity,
                                _unused_spcc_core_context,
                            ) = bright_core_color.assess_spcc_bright_core_color(
                                (
                                    before_pixels_native
                                    if before_pixels_native is not None
                                    else before_chw
                                ),
                                candidate_pixels,
                                target_type=_stage4_active_target_type(pipeline),
                                target_profile=getattr(
                                    pipeline,
                                    "target_profile",
                                    None,
                                ),
                            )
                        else:
                            bright_core_color_integrity = {
                                "schema": bright_core_color.SCHEMA,
                                "applicable": False,
                                "status": "not_run_technical_failure",
                                "accepted": False,
                                "trigger_reasons": list(
                                    technical_integrity.get("rejection_reasons") or []
                                ),
                            }
                        bright_core_color_integrity.update(
                            routing_effect=(
                                "advisory_only"
                                if technical_accepted
                                else "technical_failure"
                            ),
                            candidate_mutated=False,
                            final_action=(
                                "accept_successful_siril_candidate"
                                if technical_accepted
                                else "reject_technically_invalid_candidate"
                            ),
                        )
                        pipeline._stage4_bright_core_color_integrity = dict(
                            bright_core_color_integrity
                        )
                        spcc_quality_report["calibration"] = "SPCC"
                        spcc_quality_report["physical_color"] = True
                        spcc_quality_report[
                            "bright_core_color_integrity"
                        ] = bright_core_color_integrity
                        precision_warnings = [
                            str(attempt.get("precision_warning"))
                            for attempt in spcc_attempts
                            if attempt.get("precision_warning")
                        ]
                        spcc_quality_report["siril_precision_warning"] = {
                            "present": bool(precision_warnings),
                            "codes": precision_warnings,
                            "policy": (
                                "advisory_only_after_successful_command"
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
                                f"{spcc_result} accepted after technical integrity validation"
                            )
                            pipeline.log.info(
                                "SPCC Gaia DR3 校色命令成功，候选通过技术完整性检查"
                            )
                        else:
                            color_warning = "spcc_technical_integrity_rejected"
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
            baseline_restored, exact_spcc_restore = (
                _stage4_restore_exact_pre_color(
                    pipeline,
                    before_pixels_native,
                )
            )
            spcc_rollback_report.update(
                restored=baseline_restored,
                checkpoint=f"{PCC_CHECKPOINT_STEM}.fit",
                exact_restore=exact_spcc_restore,
            )
            if baseline_restored:
                messages.append(
                    "restored immutable pre-color linear checkpoint after SPCC"
                )
            else:
                hard_degraded = True
                main_output_blocked = True
                status = "failed"
                messages.append(
                    "pre-color restore after SPCC failed exact verification; "
                    "PCC and heuristic fallbacks prohibited"
                )

            if not pcc_allowed:
                pcc_skip_reason = (
                    "disabled by stage4_pcc_fallback_enabled"
                    if not pcc_fallback_enabled
                    else "narrowband degraded PCC disabled"
                    if narrowband_physical
                    and not narrowband_degraded_pcc_enabled
                    else f"blocked by failure policy {failure_action}"
                    if failure_action != "auto_fallback"
                    else "runtime capability decision marked Gaia astrometry/PCC unavailable"
                    if not runtime_pcc_allowed
                    else "plate solve unavailable"
                    if not pipeline.platesolve_ok
                    else (
                        "explicit offline mode requires a valid local Gaia "
                        "astrometric catalogue"
                        if selected_pcc_catalog is None
                        else "immutable pre-color source unavailable"
                    )
                )
                messages.append(f"PCC exception fallback skipped: {pcc_skip_reason}")
            elif baseline_restored:
                pcc_ok, pcc_result, pcc_attempts = _stage4_run_pcc(
                    pipeline,
                    phase=(
                        "dual_narrowband_spcc_degraded_fallback"
                        if narrowband_physical
                        else "spcc_exception_fallback"
                    ),
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
                        legacy_quality_accepted, pcc_quality_report = _stage4_pcc_quality_gate(
                            before_chw,
                            candidate_chw,
                            pipeline,
                        )
                        technical_accepted, technical_integrity = (
                            _stage4_candidate_technical_integrity(
                                before_chw,
                                candidate_chw,
                            )
                        )
                        pcc_quality_report.update(
                            accepted=bool(technical_accepted),
                            technical_accepted=bool(technical_accepted),
                            diagnostic_quality_accepted=bool(
                                legacy_quality_accepted
                            ),
                            legacy_quality_accepted=bool(legacy_quality_accepted),
                            routing_effect=(
                                "advisory_only"
                                if technical_accepted
                                else "technical_failure"
                            ),
                            technical_integrity=technical_integrity,
                        )
                        accepted = bool(technical_accepted)
                        if narrowband_physical and technical_accepted:
                            signal_preservation = (
                                _stage4_narrowband_pcc_signal_preservation(
                                    before_chw,
                                    candidate_chw,
                                    channel_gain_ratio=float(
                                        (
                                            pcc_quality_report.get(
                                                "measurements",
                                                {},
                                            )
                                            or {}
                                        ).get("channel_gain_ratio", float("inf"))
                                    ),
                                )
                            )
                            pcc_quality_report[
                                "source_signal_preservation"
                            ] = signal_preservation
                            signal_preservation["routing_effect"] = "advisory_only"
                        if technical_accepted:
                            (
                                pcc_bright_core_color_integrity,
                                _unused_pcc_core_context,
                            ) = bright_core_color.assess_spcc_bright_core_color(
                                (
                                    before_pixels_native
                                    if before_pixels_native is not None
                                    else before_chw
                                ),
                                candidate_pixels,
                                target_type=_stage4_active_target_type(pipeline),
                                target_profile=getattr(
                                    pipeline,
                                    "target_profile",
                                    None,
                                ),
                            )
                        else:
                            pcc_bright_core_color_integrity = {
                                "schema": bright_core_color.SCHEMA,
                                "applicable": False,
                                "status": "not_run_technical_failure",
                                "accepted": False,
                                "trigger_reasons": list(
                                    technical_integrity.get("rejection_reasons") or []
                                ),
                            }
                        pcc_bright_core_color_integrity.update(
                            candidate_method=(
                                "PCC_LOCAL_GAIA"
                                if selected_pcc_catalog == PCC_LOCAL_CATALOG
                                else "PCC"
                            ),
                            assessment_role="physical_fallback_validation",
                        )
                        physical_pcc_global_rebalance = (
                            _stage4_verified_physical_pcc_global_rebalance(
                                pipeline,
                                pcc_quality_report,
                                pcc_bright_core_color_integrity,
                            )
                            if not narrowband_physical
                            else {
                                "applicable": False,
                                "accepted": False,
                                "reason_code": "narrowband_pcc_is_not_physical_color",
                            }
                        )
                        pcc_bright_core_color_integrity[
                            "physical_pcc_global_rebalance"
                        ] = physical_pcc_global_rebalance
                        if bool(physical_pcc_global_rebalance.get("accepted", False)):
                            pcc_bright_core_color_integrity.update(
                                accepted=True,
                                status="ok",
                                final_action=(
                                    "accept_verified_physical_pcc_global_rebalance"
                                ),
                                diagnostic_trigger_reasons=list(
                                    pcc_bright_core_color_integrity.get(
                                        "trigger_reasons"
                                    )
                                    or []
                                ),
                                trigger_reasons=[],
                                resolved_by="verified_physical_pcc_global_rebalance",
                            )
                        pcc_quality_report["bright_core_color_integrity"] = (
                            pcc_bright_core_color_integrity
                        )
                        pcc_core_applicable = bool(
                            pcc_bright_core_color_integrity.get(
                                "applicable",
                                False,
                            )
                        )
                        if pcc_core_applicable:
                            bright_core_color_integrity[
                                "pcc_fallback_assessment"
                            ] = pcc_bright_core_color_integrity
                        pcc_bright_core_color_integrity.update(
                            routing_effect=(
                                "advisory_only"
                                if technical_accepted
                                else "technical_failure"
                            ),
                            candidate_mutated=False,
                            final_action=(
                                "accept_successful_siril_candidate"
                                if technical_accepted
                                else "reject_technically_invalid_candidate"
                            ),
                        )
                        pcc_quality_report.update(
                            calibration=(
                                "PCC_NARROWBAND_DEGRADED"
                                if narrowband_physical
                                else "PCC"
                            ),
                            physical_color=not narrowband_physical,
                            degraded_color_correction=bool(narrowband_physical),
                        )
                        if accepted:
                            if narrowband_physical:
                                color_method = (
                                    "PCC_NARROWBAND_DEGRADED_LOCAL_GAIA"
                                    if selected_pcc_catalog == PCC_LOCAL_CATALOG
                                    else "PCC_NARROWBAND_DEGRADED"
                                )
                                color_confidence = (
                                    0.48
                                    if selected_pcc_catalog == PCC_LOCAL_CATALOG
                                    else 0.44
                                )
                                color_warning = (
                                    "narrowband_spcc_failed_pcc_degraded"
                                )
                                policy_status = (
                                    "accepted_degraded_review_required"
                                )
                                requires_review = True
                                status = "degraded"
                                messages.append(
                                    f"{pcc_result} accepted after technical integrity validation as degraded dual-"
                                    "narrowband color correction; not physical color"
                                )
                                pipeline.log.warn(
                                    "双窄带 SPCC 不可用，PCC 仅作为降级基础颜色矫正；"
                                    "结果不是物理校色并强制复核"
                                )
                            else:
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
                                messages.append(
                                    f"{pcc_result} accepted after SPCC technical failure"
                                )
                                pipeline.log.info(
                                    "SPCC 技术失败后 PCC 命令成功，候选通过技术完整性检查"
                                )
                            pcc_fallback_used = True
                        else:
                            color_warning = "pcc_technical_integrity_rejected"
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
                            "physical_color": False if narrowband_physical else None,
                            "degraded_color_correction": bool(narrowband_physical),
                        }
                        color_warning = "pcc_candidate_unavailable"
                        messages.append(f"PCC fallback candidate unavailable: {error}")
                else:
                    color_warning = "pcc_single_attempt_failed"
                    messages.append(pcc_result)

            if color_method not in accepted_pcc_methods:
                rollback_report["required"] = bool(pcc_attempts)
                if main_output_blocked:
                    exact_restored = False
                    exact_restore_report = {
                        "restored": False,
                        "verified_exact": False,
                        "source": None,
                        "reason": "prior_exact_pre_color_restore_failed",
                    }
                else:
                    exact_restored, exact_restore_report = (
                        _stage4_restore_exact_pre_color(
                            pipeline,
                            before_pixels_native,
                        )
                    )
                rollback_report.update(
                    restored=exact_restored,
                    checkpoint=f"{PCC_CHECKPOINT_STEM}.fit",
                    exact_restore=exact_restore_report,
                )
                if exact_restored:
                    messages.append(
                        "restored immutable pre-color checkpoint after PCC fallback"
                    )
                else:
                    hard_degraded = True
                    main_output_blocked = True
                    status = "failed"
                    messages.append(
                        "pre-color restore after PCC failed exact verification; "
                        "heuristic fallbacks prohibited"
                    )

                if narrowband_physical:
                    color_method = "PRESERVE_INPUT"
                    color_confidence = 0.55 if rollback_report.get("restored") else 0.15
                    color_warning = color_warning or "narrowband_pcc_degraded_failed"
                    messages.append(
                        "dual-narrowband degraded PCC unavailable; preserved pre-color input"
                    )
                elif failure_action == "auto_fallback" and not main_output_blocked:
                    offline_fallback_mode = _stage4_offline_fallback_mode(pipeline)
                    rollback_report["auto_reference_baseline"] = exact_restore_report
                    if not pre_pcc_saved:
                        hard_degraded = True
                        auto_reference_report = _stage4_empty_auto_reference_report(
                            status="unavailable",
                            reason="immutable_pre_color_checkpoint_unavailable",
                        )
                        auto_reference_report["runtime_capability_decision"] = (
                            runtime_color_decision
                        )
                        auto_reference_report["transaction"].update(
                            rollback_performed=True,
                            rollback=exact_restore_report,
                        )
                        color_method = "PRESERVE_INPUT"
                        color_warning = "immutable_pre_color_checkpoint_unavailable"
                        color_confidence = 0.30
                        messages.append(
                            "auto/local color fallbacks prohibited: immutable "
                            "stage4_pre_pcc checkpoint was not saved; input preserved"
                        )
                    elif offline_fallback_mode == "preserve":
                        auto_reference_report = _stage4_empty_auto_reference_report(
                            status="preserved_by_configuration",
                            reason="offline_fallback_mode_preserve",
                        )
                        auto_reference_report["runtime_capability_decision"] = (
                            runtime_color_decision
                        )
                        auto_reference_report["transaction"].update(
                            rollback_performed=True,
                            rollback=exact_restore_report,
                        )
                        color_method = "PRESERVE_INPUT"
                        color_warning = "offline_fallback_preserve_input"
                        color_confidence = 0.55
                        messages.append(
                            "Gaia unavailable or physical calibration rejected; "
                            "offline preserve mode kept immutable pre-color pixels"
                        )
                    else:
                        try:
                            auto_started = time.monotonic()
                            pipeline.log.info(
                                "自动局部参考评估开始："
                                f"shape={tuple(np.asarray(before_pixels_native).shape)}"
                            )
                            auto_candidate, auto_reference_report = (
                                evaluate_auto_local_reference(
                                    before_pixels_native,
                                    config=pipeline.cfg,
                                    channel_kind=channel_policy["kind"],
                                    linear=True,
                                )
                            )
                            auto_reference_report["runtime_capability_decision"] = (
                                runtime_color_decision
                            )
                            auto_stars = (
                                auto_reference_report.get("sampling", {})
                                .get("stars", {})
                                .get("selection", {})
                            )
                            auto_a = auto_reference_report.get("candidates", {}).get(
                                AUTO_BACKGROUND_METHOD,
                                {},
                            )
                            auto_b = auto_reference_report.get("candidates", {}).get(
                                AUTO_WHITE_REGION_METHOD,
                                {},
                            )
                            pipeline.log.info(
                                "自动局部参考评估完成："
                                f"status={auto_reference_report.get('status')} "
                                f"A={bool(auto_a.get('accepted'))} "
                                f"B={bool(auto_b.get('would_accept'))} "
                                f"stars={int(auto_stars.get('valid_object_count') or 0)}/"
                                f"{int(auto_stars.get('analysis_object_count') or 0)} "
                                f"elapsed={time.monotonic() - auto_started:.2f}s"
                            )
                            if auto_candidate is not None:
                                (
                                    auto_reference_applied,
                                    auto_reference_report,
                                ) = _stage4_apply_auto_reference_candidate(
                                    pipeline,
                                    auto_candidate,
                                    expected_pre_color=before_pixels_native,
                                    report=auto_reference_report,
                                )
                            if auto_reference_applied:
                                color_method = str(
                                    auto_reference_report.get("selection", {}).get(
                                        "method"
                                    )
                                    or AUTO_BACKGROUND_METHOD
                                )
                                color_confidence = (
                                    0.50
                                    if color_method == AUTO_WHITE_REGION_METHOD
                                    else 0.45
                                )
                                color_warning = (
                                    "auto_white_region_reference_non_physical"
                                    if color_method == AUTO_WHITE_REGION_METHOD
                                    else "auto_background_neutralization_non_physical"
                                )
                                messages.append(
                                    f"{color_method} accepted by independent auto-reference "
                                    "quality gate; non-physical color requires review"
                                )
                            else:
                                restored_after_rejection, rejection_restore = (
                                    _stage4_restore_exact_pre_color(
                                        pipeline,
                                        before_pixels_native,
                                    )
                                )
                                auto_reference_report.setdefault(
                                    "transaction", {}
                                ).update(
                                    rollback_performed=restored_after_rejection,
                                    rollback=rejection_restore,
                                )
                                if not restored_after_rejection:
                                    raise RuntimeError(
                                        "auto-reference rejection rollback was not exact"
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
                            auto_reference_applied = False
                            auto_reference_report = (
                                auto_reference_report
                                if auto_reference_report.get("schema")
                                == AUTO_REFERENCE_SCHEMA
                                else _stage4_empty_auto_reference_report()
                            )
                            auto_reference_report["status"] = "failed"
                            auto_reference_report["error"] = str(error)
                            auto_reference_report["runtime_capability_decision"] = (
                                runtime_color_decision
                            )
                            restored_after_error, error_restore = (
                                _stage4_restore_exact_pre_color(
                                    pipeline,
                                    before_pixels_native,
                                )
                            )
                            auto_reference_report.setdefault(
                                "transaction", {}
                            ).update(
                                rollback_performed=restored_after_error,
                                rollback=error_restore,
                            )
                            if not restored_after_error:
                                hard_degraded = True
                                main_output_blocked = True
                                status = "failed"
                            messages.append(
                                "auto local reference failed; immutable pre-color "
                                f"restored={restored_after_error}: {error}"
                            )
                            pipeline.log.warn(
                                "自动局部参考评估失败："
                                f"restored={restored_after_error} error={error}"
                            )

                        if not auto_reference_applied and not main_output_blocked:
                            color_method = "PRESERVE_INPUT"
                            color_confidence = 0.30
                            color_warning = (
                                color_warning or "auto_reference_candidate_rejected"
                            )
                            local_fallback_report = {
                                "status": "retired",
                                "reason": "replaced_by_auto_rectangular_reference_v2",
                                "applied": False,
                            }
                            messages.append(
                                "automatic rectangular reference was not accepted; "
                                "immutable pre-color input preserved without legacy "
                                "local star-mask fallback"
                            )
                else:
                    color_method = "PRESERVE_INPUT"
                    color_confidence = (
                        0.55 if rollback_report.get("restored") else 0.15
                    )
                    color_warning = (
                        color_warning
                        or f"stage4_failure_policy_{failure_action}"
                    )
                    messages.append(
                        "color fallback search stopped by failure policy; "
                        "pre-color input preserved"
                    )
                requires_review = True
                if not main_output_blocked:
                    status = "degraded"
                policy_status = "fallback_review_required"

        if narrowband_physical:
            narrowband_parent_is_physical = color_method in accepted_spcc_methods
            narrowband_parent_is_degraded_pcc = (
                color_method in degraded_narrowband_pcc_methods
            )
            if narrowband_parent_is_physical:
                narrowband_main_parent_stem = PHYSICAL_COLOR_STEM
                narrowband_main_parent_role = "physical_spcc"
                physical_saved = pipeline._save_stage_output(PHYSICAL_COLOR_STEM)
                narrowband_main_parent_available = physical_saved
            elif narrowband_parent_is_degraded_pcc:
                narrowband_main_parent_stem = PCC_CANDIDATE_STEM
                narrowband_main_parent_role = "degraded_pcc_not_physical"
                narrowband_main_parent_available = True
            else:
                narrowband_main_parent_stem = PCC_CHECKPOINT_STEM
                narrowband_main_parent_role = "preserved_pre_color"
                narrowband_main_parent_available = bool(pre_pcc_saved)

            if not narrowband_main_parent_available:
                hard_degraded = True
                requires_review = True
                status = "degraded"
                messages.append(
                    "dual-narrowband main color checkpoint unavailable; "
                    "isolated HOO derivative skipped"
                )
            else:
                (
                    narrowband_normalized,
                    narrowband_normalization_report,
                    narrowband_message,
                ) = _stage4_run_narrowband_normalization(
                    pipeline,
                    channel_mapping,
                )
                narrowband_normalization_report = dict(
                    narrowband_normalization_report
                )
                narrowband_normalization_report.update(
                    {
                        "role": "artistic_derivative",
                        "source_parent": f"{narrowband_main_parent_stem}.fit",
                        "physical_parent": (
                            f"{PHYSICAL_COLOR_STEM}.fit"
                            if narrowband_parent_is_physical
                            else None
                        ),
                        "parent_calibration_role": narrowband_main_parent_role,
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
                            "isolated HOO artistic derivative saved; main pipeline "
                            f"restores {narrowband_main_parent_role} parent"
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
                physical_main_restore_report["required"] = True
                try:
                    pipeline.cmd_with_check("load", narrowband_main_parent_stem)
                    physical_main_restore_report.update(
                        restored=True,
                        source=f"{narrowband_main_parent_stem}.fit",
                        source_role=narrowband_main_parent_role,
                    )
                    messages.append(
                        "restored dual-narrowband main branch after HOO derivative: "
                        f"{narrowband_main_parent_role}"
                    )
                except (CommandError, SirilError) as error:
                    hard_degraded = True
                    requires_review = True
                    status = "degraded"
                    physical_main_restore_report["physical_restore_error"] = str(
                        error
                    )
                    messages.append(
                        "dual-narrowband main branch restore after HOO derivative failed: "
                        f"{error}"
                    )
                    try:
                        pipeline.cmd_with_check("load", PCC_CHECKPOINT_STEM)
                        physical_main_restore_report.update(
                            restored=True,
                            source=f"{PCC_CHECKPOINT_STEM}.fit",
                            fallback_used=True,
                        )
                        color_method = "PRESERVE_INPUT"
                        color_warning = "physical_branch_restore_fallback_pre_color"
                        color_confidence = min(color_confidence, 0.35)
                        policy_status = "physical_restore_fallback_review_required"
                        messages.append(
                            "calibrated branch unavailable; restored immutable pre-color "
                            "checkpoint before main output"
                        )
                    except (CommandError, SirilError) as fallback_error:
                        physical_main_restore_report["pre_color_restore_error"] = str(
                            fallback_error
                        )
                        if before_chw is not None and restore_before_pixels is not None:
                            try:
                                _stage4_write_image_pixels(
                                    pipeline,
                                    restore_before_pixels(before_chw.copy()),
                                )
                                physical_main_restore_report.update(
                                    restored=True,
                                    source="in_memory_pre_color",
                                    fallback_used=True,
                                )
                                color_method = "PRESERVE_INPUT"
                                color_warning = (
                                    "physical_branch_restore_fallback_in_memory_pre_color"
                                )
                                color_confidence = min(color_confidence, 0.25)
                                policy_status = (
                                    "physical_restore_fallback_review_required"
                                )
                                messages.append(
                                    "physical and file checkpoints unavailable; restored "
                                    "in-memory pre-color pixels before main output"
                                )
                            except (
                                AttributeError,
                                CommandError,
                                SirilError,
                                RuntimeError,
                                TypeError,
                                ValueError,
                            ) as memory_error:
                                physical_main_restore_report[
                                    "in_memory_restore_error"
                                ] = str(memory_error)
                        if not physical_main_restore_report["restored"]:
                            main_output_blocked = True
                            status = "failed"
                            messages.append(
                                "stage4_color prohibited: no verified calibrated or "
                                "pre-color source could be restored"
                            )

    strict_core_evidence = stage7_quality.strict_bright_core_target_evidence(
        _stage4_active_target_type(pipeline),
        getattr(pipeline, "target_profile", None),
    )
    if bool(strict_core_evidence.get("strict", False)):
        if color_method in accepted_spcc_methods | accepted_pcc_methods:
            bright_core_color_integrity.update(
                diagnostic_status=str(
                    bright_core_color_integrity.get("status") or "not_evaluated"
                ),
                diagnostic_accepted=bool(
                    bright_core_color_integrity.get("accepted", False)
                ),
                routing_effect="advisory_only",
                status="ok",
                accepted=True,
                repaired=False,
                final_action="accept_successful_siril_candidate",
                resolved_by=color_method,
            )
        elif not bool(bright_core_color_integrity.get("applicable", False)):
            if before_chw is not None:
                bright_core_color_integrity, _unused_context = (
                    bright_core_color.assess_spcc_bright_core_color(
                        (
                            before_pixels_native
                            if before_pixels_native is not None
                            else before_chw
                        ),
                        (
                            before_pixels_native
                            if before_pixels_native is not None
                            else before_chw
                        ),
                        target_type=_stage4_active_target_type(pipeline),
                        target_profile=getattr(pipeline, "target_profile", None),
                    )
                )
                bright_core_color_integrity.update(
                    evaluated_spcc_candidate=False,
                    final_action="no_spcc_candidate_retained",
                )
            else:
                bright_core_color_integrity.update(
                    applicable=True,
                    strict_target_evidence=strict_core_evidence,
                    status="hard_failed",
                    accepted=False,
                    final_action="unresolved",
                    trigger_reasons=["pre_color_reference_unavailable"],
                )
        elif bright_core_color_integrity.get("status") == "advisory":
            bright_core_color_integrity.update(
                assessment_status="advisory",
                status="ok",
                accepted=True,
                final_action="accept_spcc_with_advisory",
            )
        elif str(bright_core_color_integrity.get("status")) != "ok":
            bright_core_color_integrity.update(
                status="hard_failed",
                accepted=False,
                final_action="unresolved",
            )
    pipeline._stage4_bright_core_color_integrity = dict(
        bright_core_color_integrity
    )

    physical_calibration_methods = {
        "SPCC",
        "SPCC_LOCAL_GAIA",
        "SPCC_NARROWBAND",
        "SPCC_NARROWBAND_LOCAL_GAIA",
        "PCC",
        "PCC_LOCAL_GAIA",
    }
    if (
        auto_reference_report.get("status") == "not_run"
        and color_method in physical_calibration_methods
    ):
        auto_reference_report = _stage4_empty_auto_reference_report(
            status="shadow_skipped_physical_color_accepted",
            reason="physical_color_accepted_shadow_disabled",
        )
        auto_reference_report["shadow_comparison"] = {
            "enabled": False,
            "reason": "physical_color_accepted_shadow_disabled",
            "would_select": None,
            "physical_method_preserved": color_method,
            "pixels_written": False,
        }
        auto_reference_report["selection"].update(
            applied=False,
            shadow_candidate_method=None,
            physical_method_preserved=color_method,
        )
        auto_reference_report["degraded_color_correction"].update(
            applied=False,
            method=None,
        )
    elif auto_reference_report.get("status") == "not_run":
        auto_reference_report = _stage4_empty_auto_reference_report(
            status="not_applicable",
            reason=(
                "unsupported_channel_semantics"
                if channel_policy.get("kind") != "broadband_rgb_osc"
                else "physical_or_fallback_route_not_eligible"
            ),
        )
    color_risk_warning = bool(
        color_warning and color_warning != "spcc_exception_pcc_fallback"
    )
    spcc_precision_warning = any(
        str(attempt.get("status") or "") == "ok"
        and str(attempt.get("precision_warning") or "")
        == "spcc_imprecise_solution"
        for attempt in spcc_attempts
        if isinstance(attempt, dict)
    )
    reduce_saturation_boost = bool(
        stage4_policy.get("reduce_saturation_if_solution_imprecise", False)
        and (color_risk_warning or spcc_precision_warning)
    )
    if reduce_saturation_boost:
        messages.append("color policy limits later saturation/color gains due to imprecise solution")
    color_saved = False
    if not main_output_blocked:
        color_saved = pipeline._save_stage_output("stage4_color")
    if not color_saved:
        if status != "failed":
            status = "degraded"
        hard_degraded = True
        requires_review = True
        messages.append(
            "stage4_color 输出被禁止"
            if main_output_blocked
            else "stage4_color 输出保存失败"
        )
    auto_reference_report["runtime_capability_decision"] = runtime_color_decision
    auto_reference_report.setdefault("transaction", {})[
        "main_output_saved"
    ] = bool(color_saved)
    pipeline._write_stage_json(
        AUTO_REFERENCE_REPORT_NAME,
        auto_reference_report,
    )
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

    policy_decisive_failure = bool(
        physical_color_input
        and failure_action != "auto_fallback"
        and color_method not in accepted_spcc_methods
        and (
            bool(spcc_attempts)
            or bool(spcc_allowed)
            or str(color_warning or "").startswith("spcc_")
        )
    )
    if policy_decisive_failure:
        requires_review = True
        pipeline._stage4_color_review_required = True
        if failure_action == "stop":
            status = "failed"
        else:
            status = "degraded"
        if hasattr(pipeline, "_record_stage_policy_event"):
            pipeline._record_stage_policy_event(
                4,
                event="fallback_search_stopped",
                reason=str(color_warning or "physical color candidate failed"),
                source="physical_color_gate",
            )
    stage_fallback_used = bool(
        profile_fallback_used
        or pcc_fallback_used
        or auto_reference_applied
        or physical_main_restore_report.get("fallback_used")
    )
    platesolve_diagnostics = _stage4_platesolve_diagnostics(
        platesolve_attempts,
        stage4_metadata,
    )
    auto_reference_methods = {
        AUTO_BACKGROUND_METHOD,
        AUTO_WHITE_REGION_METHOD,
    }
    color_applied_methods = {
        *physical_calibration_methods,
        *degraded_narrowband_pcc_methods,
        *auto_reference_methods,
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
                else platesolve_skip_reason or "user_disabled"
                if not platesolve_attempted
                else str(platesolve_diagnostics.get("failure_kind") or "failed")
            ),
            "fallback_used": False,
        },
        "color_calibration": {
            "status": (
                "failed"
                if not color_saved
                else "applied"
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
                pcc_fallback_used
                or auto_reference_applied
                or physical_main_restore_report.get("fallback_used")
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
            "method": (
                "HOO_DETERMINISTIC_MAPPING"
                if channel_policy.get("kind") == "narrowband_composite"
                else None
            ),
            "reason_code": artistic_hoo_report.get("status", "not_applicable"),
            "input": artistic_hoo_report.get("source_parent")
            or artistic_hoo_report.get("physical_parent"),
            "output": artistic_hoo_report.get("output"),
            "fallback_used": False,
            "feeds_main_pipeline": False,
        },
    }
    try:
        geometry_validation = (
            device_geometry_report.get("activation", {}).get("validation", {})
            if isinstance(device_geometry_report, Mapping)
            else {}
        )
        evidence_skip_reason = platesolve_skip_reason
        if (
            not pipeline.platesolve_ok
            and any(
                attempt.get("status") == "ok"
                for attempt in platesolve_attempts
            )
            and geometry_validation.get("accepted") is False
        ):
            evidence_skip_reason = "wcs_geometry_validation_rejected"
        stage4_header_evidence = (
            stage4_evidence.finalize_filter_header_evidence(
                stage4_header_evidence,
                final_path=Path(pipeline.process_dir) / "stage4_psolved.fit",
                final_output_saved=bool(ps_saved),
                processing_mode=str(
                    getattr(pipeline.cfg, "stage4_processing_mode", "auto")
                ),
                platesolve_attempted=bool(platesolve_attempted),
                platesolve_ok=bool(pipeline.platesolve_ok),
                device_geometry_report=device_geometry_report,
                spcc_parameters=spcc_parameters,
            )
        )
        stage4_solver_evidence = stage4_evidence.finalize_solver_capabilities(
            stage4_solver_evidence,
            attempts=platesolve_attempts,
            platesolve_attempted=bool(platesolve_attempted),
            platesolve_ok=bool(pipeline.platesolve_ok),
            skip_reason=evidence_skip_reason,
        )
        stage4_evidence_artifacts = _stage4_write_observer_evidence(
            pipeline,
            solver_evidence=stage4_solver_evidence,
            header_evidence=stage4_header_evidence,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        pipeline.log.debug(f"Stage4 final observer evidence unavailable: {error}")
    pipeline._stage4_color_review_required = bool(requires_review)
    pipeline.color_calibration_report = {
        "stage": "stage4_color",
        "failure_action": failure_action,
        "failure_policy_triggered": policy_decisive_failure,
        "input": stage4_input_stem,
        "runtime_capability_decision": runtime_color_decision,
        "solver_capabilities": stage4_evidence_artifacts[
            "solver_capabilities"
        ],
        "filter_header_evidence": stage4_evidence_artifacts[
            "filter_header"
        ],
        "platesolve": {
            "attempted": platesolve_attempted,
            "ok": bool(pipeline.platesolve_ok),
            "command": platesolve_command,
            "attempts": platesolve_attempts,
            "diagnostics": platesolve_diagnostics,
            "output": "stage4_psolved.fit" if ps_saved else None,
            "instrument_geometry": stage4_geometry,
            "device_geometry_report": device_geometry_report,
            "input_metadata": stage4_metadata,
        },
        "method": color_method,
        "bright_core_color_integrity": bright_core_color_integrity,
        "channel_mapping": channel_mapping,
        "target_aware_color_mapping": bool(target_aware_color),
        "channel_policy": channel_policy,
        "input_classification": input_classification,
        "physical_calibration_allowed": bool(
            spcc_allowed
            or (
                pcc_allowed
                and channel_policy.get("kind") == "broadband_rgb_osc"
            )
        ),
        "photometric_calibration_allowed": bool(
            spcc_allowed
            or (
                pcc_allowed
                and channel_policy.get("kind") == "broadband_rgb_osc"
            )
        ),
        "degraded_color_correction_allowed": bool(
            (
                pcc_allowed
                and channel_policy.get("kind") == "narrowband_composite"
            )
            or (
                channel_policy.get("kind") == "broadband_rgb_osc"
                and failure_action == "auto_fallback"
                and _stage4_offline_fallback_mode(pipeline)
                == "auto_local_reference"
            )
        ),
        "requires_review": bool(requires_review),
        "global_white_balance": {
            "applied": color_method == AUTO_WHITE_REGION_METHOD,
            "prohibited": color_method != AUTO_WHITE_REGION_METHOD,
            "reference": (
                "single_rectangular_white_reference"
                if color_method == AUTO_WHITE_REGION_METHOD
                else None
            ),
            "physical_color": False,
        },
        "spcc": {
            "enabled": _stage4_spcc_runtime_enabled(pipeline),
            "role": "primary_physical_calibration",
            "catalog": selected_spcc_catalog,
            "catalog_policy": "online_gaia_default_explicit_offline_localgaia",
            "local_catalog": local_spcc_catalog,
            "metadata_database": spcc_database,
            "max_attempts": 1,
            "timeout_sec": (
                int(spcc_attempts[0].get("timeout_sec", SPCC_TIMEOUT_DEFAULT_SEC))
                if spcc_attempts
                else int(
                    getattr(
                        pipeline.cfg,
                        "stage4_spcc_timeout_sec",
                        SPCC_TIMEOUT_DEFAULT_SEC,
                    )
                )
            ),
            "parameters": spcc_parameters,
            "runtime_capabilities": spcc_runtime_capabilities,
            "attempts": spcc_attempts,
            "network_enabled": _stage4_network_enabled(),
            "quality_gate": spcc_quality_report,
            "rollback": spcc_rollback_report,
        },
        "pcc": {
            "role": (
                "degraded_narrowband_color_correction_not_physical"
                if channel_policy.get("kind") == "narrowband_composite"
                else "exception_fallback_broadband_only"
            ),
            "used": bool(pcc_fallback_used),
            "degraded": bool(
                color_method in degraded_narrowband_pcc_methods
            ),
            "physical_color": bool(
                color_method in {"PCC", "PCC_LOCAL_GAIA"}
            ),
            "requires_review": bool(
                color_method in degraded_narrowband_pcc_methods
            ),
            "limitation": (
                "dual-narrowband stellar fluxes do not represent broadband "
                "continuum; PCC is only a basic color correction"
                if channel_policy.get("kind") == "narrowband_composite"
                else None
            ),
            "catalog": selected_pcc_catalog,
            "catalog_policy": "online_gaia_default_explicit_offline_localgaia",
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
        "auto_local_reference": auto_reference_report,
        "local_fallback": local_fallback_report,
        "physical_color": {
            "method": color_method,
            "accepted": color_method in physical_calibration_methods,
            "output": (
                f"{PHYSICAL_COLOR_STEM}.fit"
                if physical_saved
                else (
                    "stage4_color.fit"
                    if color_saved
                    and color_method in physical_calibration_methods
                    else None
                )
            ),
            "feeds_main_pipeline": bool(
                color_saved and color_method in physical_calibration_methods
            ),
            "main_pipeline_restore": physical_main_restore_report,
        },
        "degraded_color_correction": {
            "applied": color_method
            in (
                degraded_narrowband_pcc_methods
                | auto_reference_methods
            ),
            "method": (
                color_method
                if color_method
                in (
                    degraded_narrowband_pcc_methods
                    | auto_reference_methods
                )
                else None
            ),
            "physical_color": False,
            "requires_review": color_method
            in (
                degraded_narrowband_pcc_methods
                | auto_reference_methods
            ),
            "output": (
                "stage4_color.fit"
                if color_saved
                and color_method
                in (
                    degraded_narrowband_pcc_methods
                    | auto_reference_methods
                )
                else None
            ),
        },
        "artistic_hoo": artistic_hoo_report,
        "narrowband_normalization": artistic_hoo_report,
        "status": (
            "failed"
            if main_output_blocked
            else "review_required"
            if requires_review
            else ("success_with_warning" if color_warning else "success")
        ),
        "main_output_blocked": bool(main_output_blocked),
        "warning": color_warning or None,
        "color_confidence": color_confidence,
        "policy": (
            pipeline._active_policy_name()
            if hasattr(pipeline, "_active_policy_name")
            else str(policy.get("policy_name", "generic_low_snr_safe"))
        ),
        "policy_adjustments": {
            "reduce_saturation_boost": reduce_saturation_boost,
            "blue_gain_limit": stage4_policy.get("blue_gain_limit"),
            "red_gain_limit": stage4_policy.get("red_gain_limit"),
            "max_allowed_saturation_boost": stage4_policy.get("max_allowed_saturation_boost"),
        },
        "outputs": {
            "channel_mapping": "stage4_channel_mapping.json",
            "solver_capabilities": stage4_evidence_artifacts[
                "solver_capabilities"
            ].get("path"),
            "filter_header_evidence": stage4_evidence_artifacts[
                "filter_header"
            ].get("path"),
            "psolved": "stage4_psolved.fit" if ps_saved else None,
            "pre_color": f"{PCC_CHECKPOINT_STEM}.fit" if pre_pcc_saved else None,
            "pre_pcc": f"{PCC_CHECKPOINT_STEM}.fit" if pre_pcc_saved else None,
            "spcc_candidate": (
                f"{SPCC_CANDIDATE_STEM}.fit"
                if any(
                    attempt.get("status") == "ok"
                    for attempt in spcc_attempts
                )
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
            "spcc_capabilities": "stage4_spcc_capabilities.json",
            "auto_local_reference": AUTO_REFERENCE_REPORT_NAME,
            "auto_reference_candidate": (
                "stage4_auto_reference_candidate.fit"
                if auto_reference_applied
                else None
            ),
            "color": "stage4_color.fit" if color_saved else None,
        },
        "messages": messages,
        "fallback_used": stage_fallback_used,
        "components": components,
    }
    pipeline._write_stage_json(
        "stage4_spcc_capabilities.json",
        spcc_runtime_capabilities,
    )
    pipeline._write_stage_json("color_calibration_report.json", pipeline.color_calibration_report)

    elapsed = pipeline.log.stage_end(stage_label)
    reason_code = (
        "failure_policy_stop"
        if policy_decisive_failure and failure_action == "stop"
        else "failure_policy_preserve_review"
        if policy_decisive_failure
        else "stage4_main_output_blocked"
        if main_output_blocked
        else "stage4_input_checkpoint_unavailable"
        if not checkpoint_loaded
        else "stage4_output_save_failed"
        if not color_saved
        else "stage4_hard_degraded"
        if hard_degraded
        else "narrowband_pcc_degraded_fallback"
        if color_method in degraded_narrowband_pcc_methods
        else "auto_white_region_reference_non_physical"
        if color_method == AUTO_WHITE_REGION_METHOD
        else "auto_background_neutralization_non_physical"
        if color_method == AUTO_BACKGROUND_METHOD
        else "color_calibration_review_required"
        if requires_review
        else "stage4_platesolve_disabled"
        if not platesolve_attempted
        else "stage4_platesolve_failed"
        if not pipeline.platesolve_ok
        else "target_profiler_fallback"
        if profile_fallback_used
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
        review_reasons=(
            [reason_code or "color_calibration_review_required"]
            if requires_review
            else []
        ),
    )
    if main_output_blocked:
        raise RuntimeError(
            "Stage4 stopped: immutable pre-color baseline could not be restored "
            "after the physical color candidate"
        )
