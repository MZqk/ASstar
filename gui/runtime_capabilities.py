#!/usr/bin/env python3
"""Runtime capability inspection for one Starun task run.

The inspector is deliberately path-driven: callers pass the resource root that
belongs to the running application (or an explicit development overlay) and the
isolated runtime HOME.  It never searches a build output such as
``release/Starun.app``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence


RUNTIME_CAPABILITIES_SCHEMA = "starun.runtime-capabilities.v1"
RUNTIME_CAPABILITIES_NAME = "runtime-capabilities.json"
RUNTIME_CAPABILITIES_ENV = "STARUN_RUNTIME_CAPABILITIES_MANIFEST"
RUN_STATE_SCHEMA = "starun.run-state.v1"
RUN_STATE_NAME = "run-state.json"

GAIA_ASTRO_FILENAME = "siril_cat_healpix8_astro.dat"
GAIA_ASTRO_EXPECTED_SIZE_BYTES = 1_521_132_640
GAIA_XP_DIRNAME = "siril_cat1_healpix8_xpsamp"
GAIA_XP_FILE_PREFIX = "siril_cat1_healpix8_xpsamp_"
GAIA_XP_EXPECTED_CHUNKS = 48
GAIA_XP_MIN_CHUNK_BYTES = 1024

GAIA_ASTRO_ENDPOINT_ENV = "STARUN_PREFLIGHT_GAIA_ASTRO_ENDPOINTS"
GAIA_XP_ENDPOINT_ENV = "STARUN_PREFLIGHT_GAIA_XP_ENDPOINTS"
DEFAULT_GAIA_ASTRO_ENDPOINTS = (
    "https://tapvizier.u-strasbg.fr/TAPVizieR/tap/availability",
    "https://gea.esac.esa.int/tap-server/tap/availability",
)
DEFAULT_GAIA_XP_ENDPOINTS = (
    "https://zenodo.org/records/17988559/files/siril_cat1_healpix8_xpsamp_0.dat",
)

PIPELINE_REQUIRED_PATHS = (
    "stages/__init__.py",
    "stages/stage1_preparation.py",
    "stages/stage2_view_correction.py",
    "stages/stage3_background_extraction.py",
    "stages/stage4_color_calibration.py",
    "stage4_auto_reference.py",
    "stages/stage5_linear_denoise.py",
    "stages/stage7_stretching.py",
    "stages/stage6_star_separation.py",
    "stages/stage8_nebula_enhancement.py",
    "stages/stage9_star_remixing.py",
    "stages/stage10_export.py",
    "configs/policies",
    "configs/rules",
    "configs/target_catalog",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def runtime_catalog_paths(runtime_home: Path) -> tuple[Path, Path]:
    root = Path(runtime_home).expanduser() / ".local" / "share" / "siril"
    return root / GAIA_ASTRO_FILENAME, root / GAIA_XP_DIRNAME


def _safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.is_file() else 0
    except OSError:
        return 0


def _is_readable_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def resource_origin(resources_root: Path) -> dict[str, object]:
    resources = Path(resources_root).expanduser().resolve()
    bundle_root: Path | None = None
    if resources.name == "Resources" and resources.parent.name == "Contents":
        candidate = resources.parent.parent
        if candidate.suffix.lower() == ".app":
            bundle_root = candidate
    return {
        "kind": "app_bundle" if bundle_root is not None else "development_overlay",
        "app_bundle_root": str(bundle_root) if bundle_root is not None else None,
        "resources_root": str(resources),
    }


def _inspect_siril(
    candidates: Sequence[Path],
    *,
    resources_root: Path,
    enforce_resource_boundary: bool,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for raw_path in candidates:
        path = Path(raw_path).expanduser()
        is_file = path.is_file()
        executable = bool(is_file and os.access(path, os.X_OK))
        within_resources = _is_within(path, resources_root)
        usable = bool(
            is_file
            and executable
            and (within_resources or not enforce_resource_boundary)
        )
        records.append(
            {
                "path": str(path),
                "is_file": is_file,
                "executable": executable,
                "within_resources_root": within_resources,
                "usable": usable,
            }
        )
    selected = next((item for item in records if item["usable"]), None)
    return {
        "status": "available" if selected is not None else "unavailable",
        "available": selected is not None,
        "required": True,
        "selected_path": selected["path"] if selected is not None else None,
        "candidates": records,
        "launch_probe": {
            "status": "pending" if selected is not None else "not_run",
            "launchable": None,
        },
    }


def _inspect_template(
    path: Path,
    *,
    resources_root: Path,
    enforce_resource_boundary: bool,
) -> dict[str, object]:
    path = Path(path).expanduser()
    readable = _is_readable_file(path)
    has_core_section = False
    read_error = ""
    if readable:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            has_core_section = any(
                line.strip().lower() == "[core]" for line in text.splitlines()
            )
        except OSError as error:
            readable = False
            read_error = str(error)
    within_resources = _is_within(path, resources_root)
    available = bool(
        path.is_file()
        and readable
        and has_core_section
        and (within_resources or not enforce_resource_boundary)
    )
    return {
        "status": "available" if available else "unavailable",
        "available": available,
        "required": True,
        "path": str(path),
        "is_file": path.is_file(),
        "readable": readable,
        "has_core_section": has_core_section,
        "within_resources_root": within_resources,
        "error": read_error or None,
    }


def _inspect_pipeline(
    path: Path,
    *,
    resources_root: Path,
    enforce_resource_boundary: bool,
) -> dict[str, object]:
    path = Path(path).expanduser()
    readable = _is_readable_file(path)
    syntax_valid = False
    syntax_error = ""
    if readable:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            compile(source, str(path), "exec")
            syntax_valid = True
        except (OSError, SyntaxError, ValueError) as error:
            syntax_error = str(error)
    pipeline_root = path.parent
    missing_paths = [
        relative
        for relative in PIPELINE_REQUIRED_PATHS
        if not (pipeline_root / relative).exists()
    ]
    within_resources = _is_within(path, resources_root)
    available = bool(
        path.is_file()
        and readable
        and syntax_valid
        and not missing_paths
        and (within_resources or not enforce_resource_boundary)
    )
    return {
        "status": "available" if available else "unavailable",
        "available": available,
        "required": True,
        "path": str(path),
        "root": str(pipeline_root),
        "is_file": path.is_file(),
        "readable": readable,
        "syntax_valid": syntax_valid,
        "missing_required_paths": missing_paths,
        "within_resources_root": within_resources,
        "error": syntax_error or None,
    }


def _inspect_gaia_astro(runtime_home: Path, *, required: bool) -> dict[str, object]:
    path, _xp_root = runtime_catalog_paths(runtime_home)
    size = _safe_file_size(path)
    available = size == GAIA_ASTRO_EXPECTED_SIZE_BYTES
    return {
        "status": "available" if available else "unavailable",
        "available": available,
        "required": bool(required),
        "path": str(path),
        "size_bytes": size,
        "expected_size_bytes": GAIA_ASTRO_EXPECTED_SIZE_BYTES,
        "runtime_scoped": _is_within(path, runtime_home),
    }


def _inspect_gaia_xp(runtime_home: Path, *, required: bool) -> dict[str, object]:
    _astro_path, root = runtime_catalog_paths(runtime_home)
    expected_names = {
        f"{GAIA_XP_FILE_PREFIX}{index}.dat"
        for index in range(GAIA_XP_EXPECTED_CHUNKS)
    }
    valid_names: set[str] = set()
    invalid_names: list[str] = []
    total_bytes = 0
    try:
        candidates = tuple(root.glob(f"{GAIA_XP_FILE_PREFIX}*.dat"))
    except OSError:
        candidates = ()
    for path in candidates:
        size = _safe_file_size(path)
        if path.name in expected_names and size >= GAIA_XP_MIN_CHUNK_BYTES:
            valid_names.add(path.name)
            total_bytes += size
        elif path.name in expected_names:
            invalid_names.append(path.name)
    missing_names = sorted(expected_names - valid_names)
    available = not missing_names and not invalid_names
    return {
        "status": "available" if available else "unavailable",
        "available": available,
        "required": bool(required),
        "path": str(root),
        "expected_chunk_count": GAIA_XP_EXPECTED_CHUNKS,
        "valid_chunk_count": len(valid_names),
        "missing_chunks": missing_names,
        "invalid_chunks": sorted(invalid_names),
        "minimum_chunk_bytes": GAIA_XP_MIN_CHUNK_BYTES,
        "size_bytes": total_bytes,
        "runtime_scoped": _is_within(root, runtime_home),
    }


def _endpoint_values(
    environ: Mapping[str, str],
    name: str,
    defaults: Sequence[str],
) -> tuple[str, ...]:
    raw = str(environ.get(name) or "").strip()
    if not raw:
        return tuple(defaults)
    normalized = raw.replace(",", " ").replace(";", " ")
    return tuple(value for value in normalized.split() if value)


def configured_network_endpoints(
    environ: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    source = os.environ if environ is None else environ
    return {
        "gaia_astro": _endpoint_values(
            source,
            GAIA_ASTRO_ENDPOINT_ENV,
            DEFAULT_GAIA_ASTRO_ENDPOINTS,
        ),
        "gaia_xp": _endpoint_values(
            source,
            GAIA_XP_ENDPOINT_ENV,
            DEFAULT_GAIA_XP_ENDPOINTS,
        ),
    }


def _network_capability(
    *,
    enabled: bool,
    gaia_astro_available: bool,
    gaia_xp_available: bool,
    endpoints: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    status = "pending" if enabled else "disabled"
    groups: dict[str, object] = {}
    for name, local_available in (
        ("gaia_astro", gaia_astro_available),
        ("gaia_xp", gaia_xp_available),
    ):
        groups[name] = {
            "status": status,
            "available": None if enabled else False,
            "required": bool(enabled and not local_available),
            "local_fallback_available": bool(local_available),
            "configured_endpoints": list(endpoints.get(name, ())),
            "probes": [],
            "evidence_level": (
                "endpoint_reachability_only"
                if name == "gaia_xp"
                else "service_availability_endpoint"
            ),
            "service_verified": False,
        }
    return {
        "status": status,
        "enabled": bool(enabled),
        "probe_completed": not enabled,
        "groups": groups,
    }


def build_runtime_capabilities(
    *,
    resources_root: Path,
    runtime_home: Path,
    siril_candidates: Sequence[Path],
    config_template: Path,
    pipeline_path: Path,
    siril_plugin_dir: Path,
    network_enabled: bool,
    stage4_offline_fallback_mode: str = "auto_local_reference",
    stage4_spcc_online_unverified_timeout_sec: int = 90,
    run_id: str | None = None,
    endpoints: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    resources = Path(resources_root).expanduser().resolve()
    runtime = Path(runtime_home).expanduser().resolve()
    origin = resource_origin(resources)
    enforce_boundary = origin["kind"] == "app_bundle"
    fallback_mode = str(stage4_offline_fallback_mode or "").strip().lower()
    if fallback_mode not in {"auto_local_reference", "preserve"}:
        fallback_mode = "auto_local_reference"
    try:
        online_unverified_timeout_sec = int(
            stage4_spcc_online_unverified_timeout_sec
        )
    except (TypeError, ValueError):
        online_unverified_timeout_sec = 90
    online_unverified_timeout_sec = max(
        30,
        min(online_unverified_timeout_sec, 180),
    )
    astro = _inspect_gaia_astro(runtime, required=False)
    xp = _inspect_gaia_xp(runtime, required=False)
    configured = configured_network_endpoints() if endpoints is None else endpoints
    manifest: dict[str, object] = {
        "schema": RUNTIME_CAPABILITIES_SCHEMA,
        "generated_at": utc_now(),
        "updated_at": utc_now(),
        "run_id": run_id,
        "resource_origin": origin,
        "runtime_home": str(runtime),
        "configuration": {
            "stage4_offline_fallback_mode": fallback_mode,
            "stage4_spcc_online_unverified_timeout_sec": (
                online_unverified_timeout_sec
            ),
        },
        "resource_paths": {
            "config_template": str(Path(config_template).expanduser()),
            "pipeline": str(Path(pipeline_path).expanduser()),
            "siril_plugins": str(Path(siril_plugin_dir).expanduser()),
        },
        "capabilities": {
            "siril": _inspect_siril(
                siril_candidates,
                resources_root=resources,
                enforce_resource_boundary=enforce_boundary,
            ),
            "config_template": _inspect_template(
                config_template,
                resources_root=resources,
                enforce_resource_boundary=enforce_boundary,
            ),
            "pipeline": _inspect_pipeline(
                pipeline_path,
                resources_root=resources,
                enforce_resource_boundary=enforce_boundary,
            ),
            "gaia_astro": astro,
            "gaia_xp": xp,
            "network_endpoints": _network_capability(
                enabled=network_enabled,
                gaia_astro_available=bool(astro["available"]),
                gaia_xp_available=bool(xp["available"]),
                endpoints=configured,
            ),
        },
        "status": "pending",
        "blocking_errors": [],
        "degraded_reasons": [],
        "decisions": {},
    }
    refresh_blocking_errors(manifest)
    return manifest


def _capabilities(manifest: Mapping[str, object]) -> Mapping[str, object]:
    value = manifest.get("capabilities")
    return value if isinstance(value, Mapping) else {}


def _stage4_color_decision(
    manifest: Mapping[str, object],
    capabilities: Mapping[str, object],
) -> dict[str, object]:
    configuration = manifest.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    fallback_mode = str(
        configuration.get("stage4_offline_fallback_mode")
        or "auto_local_reference"
    ).strip().lower()
    if fallback_mode not in {"auto_local_reference", "preserve"}:
        fallback_mode = "auto_local_reference"
    try:
        online_unverified_timeout_sec = int(
            configuration.get("stage4_spcc_online_unverified_timeout_sec")
            or 90
        )
    except (TypeError, ValueError):
        online_unverified_timeout_sec = 90
    online_unverified_timeout_sec = max(
        30,
        min(online_unverified_timeout_sec, 180),
    )

    network = capabilities.get("network_endpoints")
    network = network if isinstance(network, Mapping) else {}
    groups = network.get("groups")
    groups = groups if isinstance(groups, Mapping) else {}
    astro = capabilities.get("gaia_astro")
    astro = astro if isinstance(astro, Mapping) else {}
    xp = capabilities.get("gaia_xp")
    xp = xp if isinstance(xp, Mapping) else {}
    astro_group = groups.get("gaia_astro")
    astro_group = astro_group if isinstance(astro_group, Mapping) else {}
    xp_group = groups.get("gaia_xp")
    xp_group = xp_group if isinstance(xp_group, Mapping) else {}

    network_enabled = bool(network.get("enabled"))
    probe_completed = bool(network.get("probe_completed"))
    local_astro = bool(astro.get("available"))
    local_xp = bool(xp.get("available"))
    remote_astro = bool(
        network_enabled and probe_completed and astro_group.get("available")
    )
    remote_xp = bool(
        network_enabled and probe_completed and xp_group.get("available")
    )
    astro_available = local_astro or remote_astro
    xp_available = local_xp or remote_xp
    astrometric_source = (
        "gaia" if remote_astro else "localgaia" if local_astro else None
    )
    xp_source = "gaia" if remote_xp else "localgaia" if local_xp else None
    spcc_readiness = (
        "online_unverified"
        if astro_available and xp_source == "gaia"
        else "local_verified"
        if astro_available and xp_source == "localgaia"
        else "unavailable"
    )

    reasons: list[str] = []
    if astro_available and xp_available:
        route = "physical_spcc_then_pcc"
        status = "ready"
        skip_commands: list[str] = []
        requires_review = False
        if spcc_readiness == "online_unverified":
            reasons.append("gaia_xp_endpoint_reachable_spcc_unverified")
    elif astro_available:
        route = "physical_pcc_only"
        status = "degraded_allowed"
        skip_commands = ["spcc"]
        requires_review = False
        reasons.append("gaia_xp_unavailable_spcc_skipped")
    else:
        route = (
            "auto_local_reference"
            if fallback_mode == "auto_local_reference"
            else "preserve_input"
        )
        status = "degraded_allowed"
        skip_commands = ["platesolve", "spcc", "pcc"]
        requires_review = True
        reasons.append("gaia_astrometry_unavailable_physical_color_skipped")
        if xp_available:
            reasons.append("gaia_xp_unusable_without_astrometric_solution")

    if network_enabled and not probe_completed and not (
        local_astro and local_xp
    ):
        decision_state = "pending_network"
    else:
        decision_state = status
    return {
        "schema": "starun.stage4-color-capability-decision.v1",
        "status": decision_state,
        "route": route,
        "offline_fallback_mode": fallback_mode,
        "astrometric_source": astrometric_source,
        "xp_source": xp_source,
        "physical_color_available": bool(astro_available),
        "spcc_available": bool(astro_available and xp_available),
        "spcc_readiness": spcc_readiness,
        "spcc_endpoint_evidence": (
            "endpoint_reachability_only"
            if spcc_readiness == "online_unverified"
            else "complete_local_catalog"
            if spcc_readiness == "local_verified"
            else None
        ),
        "spcc_operational_verified": spcc_readiness == "local_verified",
        "spcc_online_unverified_timeout_sec": online_unverified_timeout_sec,
        "pcc_available": bool(astro_available),
        "auto_local_reference_available": bool(
            not astro_available and fallback_mode == "auto_local_reference"
        ),
        "preserve_input_available": bool(
            not astro_available and fallback_mode == "preserve"
        ),
        "commands": {
            "platesolve": bool(astro_available),
            "spcc": bool(astro_available and xp_available),
            "pcc": bool(astro_available),
        },
        "skip_photometric_commands": skip_commands,
        "requires_review": requires_review,
        "reason_codes": reasons,
    }


def refresh_blocking_errors(manifest: MutableMapping[str, object]) -> list[str]:
    capabilities = _capabilities(manifest)
    errors: list[str] = []

    siril = capabilities.get("siril")
    if not isinstance(siril, Mapping) or not siril.get("available"):
        errors.append("实际 App 资源中未找到可执行的 Siril CLI。")
    elif isinstance(siril.get("launch_probe"), Mapping):
        launch_probe = siril["launch_probe"]
        if launch_probe.get("status") == "failed":
            errors.append(
                "Siril CLI 启动探测失败："
                + str(launch_probe.get("detail") or siril.get("selected_path") or "未知原因")
            )

    template = capabilities.get("config_template")
    if not isinstance(template, Mapping) or not template.get("available"):
        path = template.get("path") if isinstance(template, Mapping) else "<未知>"
        errors.append(f"配置模板缺失、不可读或格式无效：{path}")

    pipeline = capabilities.get("pipeline")
    if not isinstance(pipeline, Mapping) or not pipeline.get("available"):
        path = pipeline.get("path") if isinstance(pipeline, Mapping) else "<未知>"
        detail = ""
        if isinstance(pipeline, Mapping):
            missing = pipeline.get("missing_required_paths")
            if isinstance(missing, Sequence) and missing:
                detail = "；缺少 " + "、".join(str(item) for item in missing)
            elif pipeline.get("error"):
                detail = "；" + str(pipeline["error"])
        errors.append(f"流水线资源缺失、不可读或语法无效：{path}{detail}")

    network = capabilities.get("network_endpoints")
    decision = _stage4_color_decision(manifest, capabilities)
    decisions = manifest.get("decisions")
    if not isinstance(decisions, MutableMapping):
        decisions = {}
        manifest["decisions"] = decisions
    decisions["stage4_color_calibration"] = decision

    manifest["blocking_errors"] = errors
    network_pending = bool(
        isinstance(network, Mapping)
        and network.get("enabled")
        and not network.get("probe_completed")
    )
    decision_status = str(decision.get("status") or "")
    manifest["degraded_reasons"] = list(decision.get("reason_codes") or [])
    manifest["status"] = (
        "blocked"
        if errors
        else "pending_network"
        if network_pending
        else "degraded_allowed"
        if decision_status == "degraded_allowed"
        else "ready"
    )
    manifest["updated_at"] = utc_now()
    return errors


def probe_network_endpoint(
    url: str,
    *,
    timeout_seconds: float = 4.0,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> dict[str, object]:
    started = time.monotonic()
    normalized = str(url or "").strip()
    if not normalized.lower().startswith(("https://", "http://")):
        return {
            "url": normalized,
            "reachable": False,
            "status": "invalid",
            "http_status": None,
            "elapsed_ms": 0,
            "detail": "endpoint must use http or https",
        }
    request = urllib.request.Request(
        normalized,
        headers={
            "User-Agent": "Starun/1.0 RuntimePreflight",
            "Range": "bytes=0-0",
        },
        method="GET",
    )
    response = None
    try:
        response = opener(request, timeout=max(0.1, float(timeout_seconds)))
        code = int(getattr(response, "status", 200) or 200)
        reachable = 200 <= code < 400
        read = getattr(response, "read", None)
        if reachable and callable(read):
            read(1)
        return {
            "url": normalized,
            "reachable": reachable,
            "status": "reachable" if reachable else "unavailable",
            "http_status": code,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "detail": None if reachable else f"HTTP {code}",
        }
    except urllib.error.HTTPError as error:
        code = int(error.code)
        reachable = 200 <= code < 400
        return {
            "url": normalized,
            "reachable": reachable,
            "status": "reachable" if reachable else "unavailable",
            "http_status": code,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "detail": None if reachable else str(error),
        }
    except (OSError, ValueError, urllib.error.URLError) as error:
        return {
            "url": normalized,
            "reachable": False,
            "status": "unreachable",
            "http_status": None,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "detail": str(error),
        }
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def probe_network_capabilities(
    manifest: MutableMapping[str, object],
    *,
    timeout_seconds: float = 4.0,
    opener: Callable[..., object] = urllib.request.urlopen,
    check_cancelled: Callable[[], None] | None = None,
) -> MutableMapping[str, object]:
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, MutableMapping):
        return manifest
    network = capabilities.get("network_endpoints")
    if not isinstance(network, MutableMapping) or not network.get("enabled"):
        refresh_blocking_errors(manifest)
        return manifest
    groups = network.get("groups")
    if not isinstance(groups, MutableMapping):
        refresh_blocking_errors(manifest)
        return manifest

    all_available = True
    for _name, raw_group in groups.items():
        if check_cancelled is not None:
            check_cancelled()
        if not isinstance(raw_group, MutableMapping):
            all_available = False
            continue
        endpoints = raw_group.get("configured_endpoints")
        endpoint_values = endpoints if isinstance(endpoints, Sequence) else ()
        probes = []
        for endpoint in endpoint_values:
            if check_cancelled is not None:
                check_cancelled()
            probes.append(
                probe_network_endpoint(
                    str(endpoint),
                    timeout_seconds=timeout_seconds,
                    opener=opener,
                )
            )
        available = any(bool(probe.get("reachable")) for probe in probes)
        raw_group["probes"] = probes
        raw_group["available"] = available
        raw_group["status"] = "available" if available else "unavailable"
        all_available = all_available and available
    network["probe_completed"] = True
    network["status"] = "available" if all_available else "degraded"
    refresh_blocking_errors(manifest)
    return manifest


def update_siril_launch_probe(
    manifest: MutableMapping[str, object],
    *,
    launchable: bool,
    version: str = "",
    detail: str = "",
) -> None:
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, MutableMapping):
        return
    siril = capabilities.get("siril")
    if not isinstance(siril, MutableMapping):
        return
    siril["launch_probe"] = {
        "status": "available" if launchable else "failed",
        "launchable": bool(launchable),
        "version": str(version or "") or None,
        "detail": str(detail or "") or None,
        "checked_at": utc_now(),
    }
    refresh_blocking_errors(manifest)


def capability_summary_lines(manifest: Mapping[str, object]) -> list[str]:
    capabilities = _capabilities(manifest)
    labels = {
        "siril": "Siril",
        "config_template": "配置模板",
        "pipeline": "流水线",
        "gaia_astro": "Gaia astro",
        "gaia_xp": "Gaia XP",
        "network_endpoints": "网络端点",
    }
    lines = [f"运行能力清单（{manifest.get('status', 'unknown')}）："]
    for key in (
        "siril",
        "config_template",
        "pipeline",
        "gaia_astro",
        "gaia_xp",
        "network_endpoints",
    ):
        value = capabilities.get(key)
        status = value.get("status") if isinstance(value, Mapping) else "unavailable"
        path = ""
        if isinstance(value, Mapping):
            path = str(value.get("selected_path") or value.get("path") or "")
        suffix = f" · {path}" if path else ""
        lines.append(f"  - {labels[key]}: {status}{suffix}")
    decisions = manifest.get("decisions")
    decisions = decisions if isinstance(decisions, Mapping) else {}
    stage4 = decisions.get("stage4_color_calibration")
    if isinstance(stage4, Mapping):
        lines.append(
            "  - Stage 4 色彩路线: "
            f"{stage4.get('route', 'unknown')}"
            + (" · 非物理色彩，需复核" if stage4.get("requires_review") else "")
        )
        if stage4.get("spcc_readiness") == "online_unverified":
            lines.append(
                "  - SPCC 在线能力: 未验证 · 单次预算 "
                f"{int(stage4.get('spcc_online_unverified_timeout_sec') or 90)} 秒"
            )
    return lines


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "DEFAULT_GAIA_ASTRO_ENDPOINTS",
    "DEFAULT_GAIA_XP_ENDPOINTS",
    "GAIA_ASTRO_ENDPOINT_ENV",
    "GAIA_ASTRO_EXPECTED_SIZE_BYTES",
    "GAIA_XP_ENDPOINT_ENV",
    "GAIA_XP_EXPECTED_CHUNKS",
    "GAIA_XP_MIN_CHUNK_BYTES",
    "RUNTIME_CAPABILITIES_NAME",
    "RUNTIME_CAPABILITIES_ENV",
    "RUNTIME_CAPABILITIES_SCHEMA",
    "RUN_STATE_NAME",
    "RUN_STATE_SCHEMA",
    "atomic_write_json",
    "build_runtime_capabilities",
    "capability_summary_lines",
    "configured_network_endpoints",
    "probe_network_capabilities",
    "probe_network_endpoint",
    "refresh_blocking_errors",
    "resource_origin",
    "runtime_catalog_paths",
    "update_siril_launch_probe",
    "utc_now",
]
