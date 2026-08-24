"""Observer-only Stage 4 solver and FITS header evidence.

The helpers in this module never load an image into Siril and never select a
processing branch.  They only describe files, already-frozen decisions, and
the plate-solving attempts executed by Stage 4.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


SOLVER_CAPABILITIES_SCHEMA = "starun.stage4-solver-capabilities.v1"
FILTER_HEADER_EVIDENCE_SCHEMA = "starun.stage4-filter-header-evidence.v1"
SOLVER_CAPABILITIES_NAME = "stage4_solver_capabilities.json"
FILTER_HEADER_EVIDENCE_NAME = "stage4_filter_header_evidence.json"

SOLVER_BACKEND_IDS = (
    "siril_platesolve",
    "astap",
    "ansvr",
    "astrometry_net",
)
FILTER_KEYS = (
    "FILTER",
    "FILTER1",
    "FILTER2",
    "INSFLNAM",
    "FILTERNAME",
    "FILTNAME",
    "FILTNAM",
)
COORDINATE_PAIRS = (
    ("RA", "DEC"),
    ("OBJCTRA", "OBJCTDEC"),
    ("CRVAL1", "CRVAL2"),
)
GEOMETRY_KEYS = (
    "FOCALLEN",
    "FOCALLENGTH",
    "FOCAL",
    "EFL",
    "XPIXSZ",
    "XPIXSIZE",
    "PIXSIZE1",
    "YPIXSZ",
    "YPIXSIZE",
    "PIXSIZE2",
    "PIXSIZE",
    "PIXELSIZ",
    "PIXELSIZE",
    "XBINNING",
    "XBIN",
    "BINX",
    "YBINNING",
    "YBIN",
    "BINY",
    "INSTRUME",
    "INSTRUMENT",
    "CAMERA",
    "DETECTOR",
    "SENSOR",
    "TELESCOP",
    "TELESCOPE",
    "CREATOR",
    "ORIGIN",
    "LENS",
    "OPTIC",
)
WCS_KEYS = (
    "SECPIX",
    "PIXSCALE",
    "PIXSCAL1",
    "CD1_1",
    "CD2_2",
    "CDELT1",
    "CDELT2",
)
_DIMENSION_KEYS = ("NAXIS1", "NAXIS2", "NAXIS3")
HEADER_EVIDENCE_KEYS = frozenset(
    FILTER_KEYS + tuple(key for pair in COORDINATE_PAIRS for key in pair)
    + GEOMETRY_KEYS + WCS_KEYS + _DIMENSION_KEYS
)
HEADER_LIMITATIONS = (
    "not_new_decision_authority",
    "not_original_xisf_header_parse",
    "postsolve_wcs_is_solver_derived",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe(scalar())
        except (TypeError, ValueError):
            pass
    return str(value)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_solver_capabilities(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SOLVER_CAPABILITIES_SCHEMA:
        raise ValueError("Stage 4 solver capability schema mismatch")
    if payload.get("status") not in {"available", "partial", "unavailable"}:
        raise ValueError("Stage 4 solver capability status is invalid")
    backends = payload.get("backends")
    if not isinstance(backends, list) or [
        backend.get("id") if isinstance(backend, Mapping) else None
        for backend in backends
    ] != list(SOLVER_BACKEND_IDS):
        raise ValueError("Stage 4 solver backend inventory is invalid")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Stage 4 solver attempts are invalid")
    if payload.get("attempts_sha256") != canonical_sha256(attempts):
        raise ValueError("Stage 4 solver attempts SHA-256 mismatch")
    limitations = {str(value) for value in payload.get("limitations") or []}
    if "not_new_routing_authority" not in limitations:
        raise ValueError("Stage 4 solver evidence authority boundary is missing")


def validate_filter_header_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != FILTER_HEADER_EVIDENCE_SCHEMA:
        raise ValueError("Stage 4 Filter/Header evidence schema mismatch")
    if payload.get("status") not in {"available", "partial", "unavailable"}:
        raise ValueError("Stage 4 Filter/Header evidence status is invalid")
    limitations = {str(value) for value in payload.get("limitations") or []}
    if not set(HEADER_LIMITATIONS).issubset(limitations):
        raise ValueError("Stage 4 Filter/Header evidence limitations are incomplete")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, Mapping):
        raise ValueError("Stage 4 Filter/Header snapshots are unavailable")
    for role, snapshot in snapshots.items():
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"Stage 4 Header snapshot is invalid: {role}")
        status = snapshot.get("status")
        if status not in {"available", "unavailable", "not_applicable"}:
            raise ValueError(f"Stage 4 Header snapshot status is invalid: {role}")
        cards = snapshot.get("cards")
        if not isinstance(cards, list):
            raise ValueError(f"Stage 4 Header cards are invalid: {role}")
        if snapshot.get("cards_sha256") != canonical_sha256(cards):
            raise ValueError(f"Stage 4 Header card SHA-256 mismatch: {role}")
        if status == "available":
            for key in ("sha256", "header_sha256", "cards_sha256"):
                if not _is_sha256(snapshot.get(key)):
                    raise ValueError(
                        f"Stage 4 Header snapshot {key} is invalid: {role}"
                    )
        elif snapshot.get("sha256") is not None and not _is_sha256(
            snapshot.get("sha256")
        ):
            raise ValueError(f"Stage 4 source SHA-256 is invalid: {role}")


def validate_evidence_payload(payload: Mapping[str, Any]) -> None:
    schema = payload.get("schema")
    if schema == SOLVER_CAPABILITIES_SCHEMA:
        validate_solver_capabilities(payload)
        return
    if schema == FILTER_HEADER_EVIDENCE_SCHEMA:
        validate_filter_header_evidence(payload)
        return
    raise ValueError("Stage 4 evidence schema is unsupported")


def file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _split_fits_value(raw: str) -> str:
    quoted = False
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "'":
            if quoted and index + 1 < len(raw) and raw[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif char == "/" and not quoted:
            return raw[:index].strip()
        index += 1
    return raw.strip()


def _parse_fits_value(raw: str) -> Any:
    value = _split_fits_value(raw)
    if value.startswith("'"):
        content = value[1:]
        if content.endswith("'"):
            content = content[:-1]
        return content.replace("''", "'").rstrip()
    if value == "T":
        return True
    if value == "F":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value.replace("D", "E").replace("d", "e"))
        except ValueError:
            return value


def _unavailable_snapshot(
    role: str,
    *,
    relation: str,
    reason: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "role": role,
        "status": "unavailable",
        "relation": relation,
        "filename": filename,
        "reason": reason,
        "cards": [],
        "cards_sha256": canonical_sha256([]),
    }


def not_applicable_snapshot(
    role: str,
    *,
    relation: str,
    reason: str,
    filename: Optional[str] = None,
    sha256: Optional[str] = None,
    size_bytes: Optional[int] = None,
    sha256_source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "role": role,
        "status": "not_applicable",
        "relation": relation,
        "filename": filename,
        "sha256": sha256,
        "sha256_source": sha256_source,
        "size_bytes": size_bytes,
        "reason": reason,
        "cards": [],
        "cards_sha256": canonical_sha256([]),
    }


def read_fits_header_snapshot(
    path: Path,
    *,
    role: str,
    relation: str,
) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() not in {".fit", ".fits", ".fts"}:
        return _unavailable_snapshot(
            role,
            relation=relation,
            reason="unsupported_header_container",
            filename=path.name,
        )
    if not path.is_file():
        return _unavailable_snapshot(
            role,
            relation=relation,
            reason="file_missing",
            filename=path.name,
        )

    header = bytearray()
    raw_cards: list[tuple[int, str, str]] = []
    end_found = False
    try:
        with path.open("rb") as handle:
            card_index = 0
            while True:
                block = handle.read(2880)
                if not block:
                    break
                header.extend(block)
                for offset in range(0, len(block), 80):
                    card_bytes = block[offset : offset + 80]
                    if len(card_bytes) < 80:
                        break
                    card = card_bytes.decode("ascii", errors="replace")
                    key = card[:8].strip().upper()
                    if key == "END":
                        end_found = True
                        break
                    if key in HEADER_EVIDENCE_KEYS and card[8:10] == "= ":
                        raw_cards.append((card_index, key, card[10:80]))
                    card_index += 1
                if end_found:
                    break
    except OSError as error:
        return _unavailable_snapshot(
            role,
            relation=relation,
            reason=f"header_read_failed:{error}",
            filename=path.name,
        )
    if not end_found:
        return _unavailable_snapshot(
            role,
            relation=relation,
            reason="fits_header_end_missing",
            filename=path.name,
        )

    occurrences: Dict[str, int] = {}
    cards: list[Dict[str, Any]] = []
    dimensions: Dict[str, int] = {}
    for card_index, key, raw_value in raw_cards:
        occurrence = occurrences.get(key, 0) + 1
        occurrences[key] = occurrence
        value = _parse_fits_value(raw_value)
        if key in _DIMENSION_KEYS and isinstance(value, int):
            dimensions[key.lower()] = value
            continue
        cards.append(
            {
                "card_index": card_index,
                "key": key,
                "occurrence": occurrence,
                "value": _json_safe(value),
                "value_type": type(value).__name__,
                "raw_value": _split_fits_value(raw_value),
            }
        )

    conflicts = []
    for key in sorted({card["key"] for card in cards}):
        values = [card["value"] for card in cards if card["key"] == key]
        normalized = {canonical_sha256(value) for value in values}
        if len(normalized) > 1:
            conflicts.append(
                {
                    "key": key,
                    "reason": "duplicate_values_disagree",
                    "values": values,
                }
            )
    try:
        size_bytes = int(path.stat().st_size)
    except OSError:
        size_bytes = None
    return {
        "role": role,
        "status": "available",
        "relation": relation,
        "filename": path.name,
        "size_bytes": size_bytes,
        "sha256": file_sha256(path),
        "sha256_source": "observed_file_bytes",
        "header_sha256": hashlib.sha256(bytes(header)).hexdigest(),
        "dimensions": dimensions,
        "cards": cards,
        "cards_sha256": canonical_sha256(cards),
        "conflicts": conflicts,
    }


def _source_summary(
    task_run_manifest: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    source = (
        task_run_manifest.get("source")
        if isinstance(task_run_manifest, Mapping)
        else None
    )
    if not isinstance(source, Mapping):
        return {
            "status": "unavailable",
            "kind": "unknown",
            "reason": "verified_task_source_unavailable",
        }
    files = source.get("files")
    files = files if isinstance(files, list) else []
    return {
        "status": "available" if files else "unavailable",
        "kind": str(source.get("kind") or "unknown"),
        "read_only": source.get("read_only") is True,
        "source_fingerprint": source.get("fingerprint"),
        "file_count": int(source.get("file_count") or len(files)),
        "total_bytes": source.get("total_bytes"),
        "manifest_hash": task_run_manifest.get("manifest_hash"),
    }


def _original_source_snapshot(
    *,
    task_run_manifest: Optional[Mapping[str, Any]],
    source_file: Optional[Path],
) -> Dict[str, Any]:
    source = (
        task_run_manifest.get("source")
        if isinstance(task_run_manifest, Mapping)
        else None
    )
    kind = str(source.get("kind") or "") if isinstance(source, Mapping) else ""
    records = source.get("files") if isinstance(source, Mapping) else None
    records = records if isinstance(records, list) else []
    if kind == "light_directory":
        return not_applicable_snapshot(
            "original_source",
            relation="registered_light_group_identity",
            reason="per_frame_headers_not_read",
        )

    record = records[0] if len(records) == 1 and isinstance(records[0], Mapping) else None
    path_value = record.get("path") if isinstance(record, Mapping) else source_file
    path = Path(str(path_value)).expanduser() if path_value else None
    expected_sha256 = str(record.get("sha256") or "") if isinstance(record, Mapping) else None
    expected_size = record.get("size") if isinstance(record, Mapping) else None
    if path is None:
        return _unavailable_snapshot(
            "original_source",
            relation="direct_source_identity",
            reason="source_path_unavailable",
        )
    if path.suffix.lower() == ".xisf":
        source_sha256 = expected_sha256 or file_sha256(path)
        if not source_sha256:
            return _unavailable_snapshot(
                "original_source",
                relation="original_xisf_identity_only",
                reason="original_xisf_sha256_unavailable",
                filename=path.name,
            )
        return not_applicable_snapshot(
            "original_source",
            relation="original_xisf_identity_only",
            reason="original_xisf_header_not_parsed",
            filename=path.name,
            sha256=source_sha256,
            size_bytes=int(expected_size) if expected_size is not None else None,
            sha256_source=(
                "verified_task_run_manifest" if expected_sha256 else "observed_file_bytes"
            ),
        )
    snapshot = read_fits_header_snapshot(
        path,
        role="original_source",
        relation="direct_source_fits_header",
    )
    if expected_sha256:
        snapshot["manifest_sha256"] = expected_sha256
        snapshot["manifest_size_bytes"] = expected_size
        snapshot["manifest_matches_observation"] = bool(
            snapshot.get("sha256") == expected_sha256
        )
    return snapshot


def _metadata_coordinate_evidence(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    indexed = {str(key).upper(): value for key, value in metadata.items()}
    candidates = []
    selected = None
    for ra_key, dec_key in COORDINATE_PAIRS:
        ra = indexed.get(ra_key)
        dec = indexed.get(dec_key)
        complete = bool(str(ra or "").strip() and str(dec or "").strip())
        candidate = {
            "ra_key": ra_key,
            "dec_key": dec_key,
            "ra": _json_safe(ra),
            "dec": _json_safe(dec),
            "complete": complete,
        }
        candidates.append(candidate)
        if selected is None and complete:
            selected = dict(candidate)
    return {
        "priority": [f"{ra}/{dec}" for ra, dec in COORDINATE_PAIRS],
        "candidates": candidates,
        "selected": selected,
        "consumer": "header_guided_platesolve",
        "blind_variants_remain_first": True,
    }


def _compact_geometry(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": report.get("schema"),
        "status": report.get("status", "available"),
        "identity": copy.deepcopy(report.get("identity")),
        "selected": copy.deepcopy(report.get("selected")),
        "conflicts": copy.deepcopy(report.get("conflicts") or []),
        "activation": copy.deepcopy(report.get("activation")),
        "consumer": "stage4_device_geometry",
    }


def build_filter_header_evidence(
    *,
    process_dir: Path,
    source_file: Optional[Path],
    task_run_manifest: Optional[Mapping[str, Any]],
    stage4_metadata: Mapping[str, Any],
    filter_selection: Mapping[str, Any],
    channel_mapping: Mapping[str, Any],
    explicit_filter_hint: str,
    device_geometry_report: Mapping[str, Any],
    header_guided_enabled: bool,
) -> Dict[str, Any]:
    process_dir = Path(process_dir)
    original = _original_source_snapshot(
        task_run_manifest=task_run_manifest,
        source_file=source_file,
    )
    xisf_source = original.get("relation") == "original_xisf_identity_only"
    stage1_relation = (
        "siril_converted_fits_observation"
        if xisf_source
        else "stage1_task_local_fits"
    )
    evidence: Dict[str, Any] = {
        "schema": FILTER_HEADER_EVIDENCE_SCHEMA,
        "status": "unavailable",
        "evidence_only": True,
        "captured_before_solve": True,
        "finalized_after_solve": False,
        "source": _source_summary(task_run_manifest),
        "snapshots": {
            "original_source": original,
            "stage1_prepared": read_fits_header_snapshot(
                process_dir / "stage1_prepared.fit",
                role="stage1_prepared",
                relation=stage1_relation,
            ),
            "stage4_input": read_fits_header_snapshot(
                process_dir / "stage3_bgremoved.fit",
                role="stage4_input",
                relation="pre_solve_linear_input",
            ),
            "solver_candidate": not_applicable_snapshot(
                "solver_candidate",
                relation="solver_output_before_wcs_validation",
                reason="platesolve_not_finalized",
            ),
            "final_psolved": not_applicable_snapshot(
                "final_psolved",
                relation="stage4_persisted_solver_output",
                reason="stage4_psolved_not_finalized",
            ),
        },
        "filter_support": {
            "selection": copy.deepcopy(_json_safe(filter_selection)),
            "frozen_channel_mapping": {
                key: _json_safe(channel_mapping.get(key))
                for key in (
                    "schema",
                    "mapping",
                    "confidence",
                    "evidence",
                    "reason",
                )
            },
            "explicit_filter_hint": {
                "value": str(explicit_filter_hint or ""),
                "source": "explicit_config_or_environment"
                if str(explicit_filter_hint or "").strip()
                else "not_set",
            },
            "consumers": [
                {
                    "id": "stage4_channel_mapping",
                    "used": True,
                    "decision": channel_mapping.get("mapping"),
                },
                {
                    "id": "stage4_spcc_metadata",
                    "used": None,
                    "status": "pending",
                },
                {
                    "id": "stage4_device_profile",
                    "used": bool(
                        channel_mapping.get("evidence")
                        == "verified_device_profile"
                    ),
                    "status": "recorded_existing_profile_evidence",
                },
            ],
            "decision_snapshot": "pre_solve_frozen",
        },
        "coordinate_support": {
            **_metadata_coordinate_evidence(stage4_metadata),
            "enabled": bool(header_guided_enabled),
        },
        "device_geometry_support": _compact_geometry(device_geometry_report),
        "spcc_metadata_consumer": {
            "status": "not_evaluated",
            "parameters": None,
        },
        "header_diff": {
            "solver_candidate": None,
            "final_psolved": None,
        },
        "limitations": list(HEADER_LIMITATIONS),
    }
    _refresh_header_status(evidence)
    return evidence


def capture_solver_candidate(
    evidence: Dict[str, Any],
    path: Path,
    *,
    platesolve_ok: bool,
    output_saved: bool,
) -> None:
    snapshots = evidence.setdefault("snapshots", {})
    if not platesolve_ok:
        snapshots["solver_candidate"] = not_applicable_snapshot(
            "solver_candidate",
            relation="solver_output_before_wcs_validation",
            reason="platesolve_not_accepted",
        )
    elif not output_saved:
        snapshots["solver_candidate"] = _unavailable_snapshot(
            "solver_candidate",
            relation="solver_output_before_wcs_validation",
            reason="solver_candidate_save_failed",
            filename=Path(path).name,
        )
    else:
        snapshot = read_fits_header_snapshot(
            path,
            role="solver_candidate",
            relation="solver_output_before_wcs_validation",
        )
        snapshot["header_origin"] = "solver_derived"
        snapshots["solver_candidate"] = snapshot
    snapshots_map = evidence.get("snapshots") or {}
    evidence.setdefault("header_diff", {})["solver_candidate"] = _header_diff(
        snapshots_map.get("stage4_input"),
        snapshots_map.get("solver_candidate"),
    )
    _refresh_header_status(evidence)


def finalize_filter_header_evidence(
    evidence: Dict[str, Any],
    *,
    final_path: Path,
    final_output_saved: bool,
    processing_mode: str,
    platesolve_attempted: bool,
    platesolve_ok: bool,
    device_geometry_report: Mapping[str, Any],
    spcc_parameters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result = evidence
    snapshots = result.setdefault("snapshots", {})
    if final_output_saved:
        candidate = snapshots.get("solver_candidate")
        relation = (
            "preserve_input_mirror"
            if processing_mode == "preserve"
            else "accepted_solver_output"
            if platesolve_ok
            else "solver_output_validation_rejected"
            if isinstance(candidate, Mapping)
            and candidate.get("status") == "available"
            else "unsolved_input_mirror"
        )
        final_snapshot = read_fits_header_snapshot(
            final_path,
            role="final_psolved",
            relation=relation,
        )
        if platesolve_ok:
            final_snapshot["header_origin"] = "solver_derived"
        elif (
            isinstance(candidate, Mapping)
            and candidate.get("status") == "available"
        ):
            same_candidate = bool(
                final_snapshot.get("status") == "available"
                and final_snapshot.get("sha256") == candidate.get("sha256")
            )
            final_snapshot["candidate_retained"] = same_candidate
            final_snapshot["rollback_observed"] = not same_candidate
    else:
        final_snapshot = _unavailable_snapshot(
            "final_psolved",
            relation="stage4_persisted_solver_output",
            reason="stage4_psolved_save_failed",
            filename=Path(final_path).name,
        )
    snapshots["final_psolved"] = final_snapshot
    result.setdefault("header_diff", {})["final_psolved"] = _header_diff(
        snapshots.get("stage4_input"),
        final_snapshot,
    )
    result["device_geometry_support"] = _compact_geometry(device_geometry_report)
    result["solve_execution"] = {
        "processing_mode": processing_mode,
        "attempted": bool(platesolve_attempted),
        "accepted": bool(platesolve_ok),
    }
    result["spcc_metadata_consumer"] = {
        "status": "available" if spcc_parameters else "not_applicable",
        "parameters": _json_safe(dict(spcc_parameters)) if spcc_parameters else None,
    }
    filter_support = result.get("filter_support")
    consumers = (
        filter_support.get("consumers")
        if isinstance(filter_support, Mapping)
        else None
    )
    if isinstance(consumers, list):
        for consumer in consumers:
            if (
                isinstance(consumer, dict)
                and consumer.get("id") == "stage4_spcc_metadata"
            ):
                consumer.update(
                    used=bool(spcc_parameters),
                    status=("used" if spcc_parameters else "not_applicable"),
                )
    result["finalized_after_solve"] = True
    _refresh_header_status(result)
    return result


def _last_values(snapshot: Any) -> Dict[str, Any]:
    if not isinstance(snapshot, Mapping) or snapshot.get("status") != "available":
        return {}
    result: Dict[str, Any] = {}
    for card in snapshot.get("cards") or []:
        if isinstance(card, Mapping):
            result[str(card.get("key") or "")] = card.get("value")
    return result


def _header_diff(before: Any, after: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    if before.get("status") != "available" or after.get("status") != "available":
        return None
    left = _last_values(before)
    right = _last_values(after)
    added = []
    changed = []
    removed = []
    for key in sorted(right.keys() - left.keys()):
        added.append({"key": key, "value": right[key], "origin": "solver_derived"})
    for key in sorted(left.keys() & right.keys()):
        if canonical_sha256(left[key]) != canonical_sha256(right[key]):
            changed.append(
                {
                    "key": key,
                    "before": left[key],
                    "after": right[key],
                    "origin": "solver_derived",
                }
            )
    for key in sorted(left.keys() - right.keys()):
        removed.append({"key": key, "value": left[key]})
    payload = {"added": added, "changed": changed, "removed": removed}
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _refresh_header_status(evidence: Dict[str, Any]) -> None:
    snapshots = evidence.get("snapshots") or {}
    input_snapshot = snapshots.get("stage4_input") or {}
    if input_snapshot.get("status") != "available":
        evidence["status"] = "unavailable"
        return
    relevant = [
        snapshot
        for snapshot in snapshots.values()
        if isinstance(snapshot, Mapping)
        and snapshot.get("status") != "not_applicable"
    ]
    evidence["status"] = (
        "partial"
        if any(
            snapshot.get("status") != "available"
            or snapshot.get("manifest_matches_observation") is False
            for snapshot in relevant
        )
        else "available"
    )


def build_solver_capabilities(
    *,
    runtime_decision: Mapping[str, Any],
    runtime_manifest: Optional[Mapping[str, Any]],
    configured: bool,
    catalogs: Sequence[Mapping[str, Any]],
    processing_mode: str,
) -> Dict[str, Any]:
    runtime_caps = (
        runtime_manifest.get("capabilities")
        if isinstance(runtime_manifest, Mapping)
        else None
    )
    runtime_caps = runtime_caps if isinstance(runtime_caps, Mapping) else {}
    siril = runtime_caps.get("siril")
    siril = siril if isinstance(siril, Mapping) else {}
    launch_probe = siril.get("launch_probe")
    launch_probe = launch_probe if isinstance(launch_probe, Mapping) else {}
    commands = runtime_decision.get("commands")
    commands = commands if isinstance(commands, Mapping) else {}
    runtime_available = bool(
        siril.get("available", True)
        and str(runtime_decision.get("status") or "")
        in {"ready", "degraded_allowed"}
    )
    packaged = None
    selected_path = siril.get("selected_path")
    for candidate in siril.get("candidates") or []:
        if (
            isinstance(candidate, Mapping)
            and candidate.get("path") == selected_path
        ):
            packaged = bool(candidate.get("within_resources_root"))
            break
    manifest_backends = runtime_caps.get("stage4_plate_solver_backends")
    if (
        isinstance(manifest_backends, list)
        and [
            backend.get("id") if isinstance(backend, Mapping) else None
            for backend in manifest_backends
        ]
        == list(SOLVER_BACKEND_IDS)
    ):
        backends = [_json_safe(dict(backend)) for backend in manifest_backends]
        backend_inventory_source = "runtime_capabilities_manifest"
    else:
        backends = [
            {
                "id": "siril_platesolve",
                "implementation_status": "integrated",
                "runtime_status": (
                    "available" if runtime_available else "unavailable"
                ),
                "packaged": packaged,
                "configured": bool(configured),
                "eligible": bool(configured and commands.get("platesolve", True)),
                "selected": False,
                "attempted": False,
                "result": "not_run",
                "reason_codes": [],
                "executor": "active_siril_session",
            }
        ]
        for backend_id in SOLVER_BACKEND_IDS[1:]:
            backends.append(
                {
                    "id": backend_id,
                    "implementation_status": "not_integrated",
                    "runtime_status": "not_probed",
                    "packaged": False,
                    "configured": False,
                    "eligible": False,
                    "selected": False,
                    "attempted": False,
                    "result": "not_run",
                    "reason_codes": ["backend_not_integrated"],
                }
            )
        backend_inventory_source = "stage4_compatibility_builder"
    backends[0].update(
        runtime_status="available" if runtime_available else "unavailable",
        packaged=packaged,
        configured=bool(configured),
        eligible=bool(configured and commands.get("platesolve", True)),
        selected=False,
        attempted=False,
        result="not_run",
        reason_codes=[],
        executor="active_siril_session",
        version=launch_probe.get("version"),
        runtime_evidence_source=(
            "runtime_capabilities_manifest"
            if runtime_manifest is not None
            else "active_pipeline_session"
        ),
    )
    return {
        "schema": SOLVER_CAPABILITIES_SCHEMA,
        "status": "available" if runtime_available else "partial",
        "evidence_only": True,
        "captured_before_solve": True,
        "finalized_after_solve": False,
        "processing_mode": processing_mode,
        "runtime_manifest": {
            "status": runtime_manifest.get("status") if runtime_manifest else None,
            "schema": runtime_manifest.get("schema") if runtime_manifest else None,
            "payload_sha256": (
                canonical_sha256(runtime_manifest) if runtime_manifest else None
            ),
            "source": runtime_decision.get("source"),
        },
        "backend_inventory_source": backend_inventory_source,
        "backends": backends,
        "catalog_sources": [_json_safe(dict(item)) for item in catalogs],
        "selection": {
            "backend": None,
            "catalog": None,
            "reason": "user_preserve" if processing_mode == "preserve" else "pending",
        },
        "attempts": [],
        "attempts_sha256": canonical_sha256([]),
        "limitations": [
            "inventory_does_not_integrate_external_backends",
            "catalog_source_is_not_solver_backend",
            "not_new_routing_authority",
        ],
    }


def finalize_solver_capabilities(
    evidence: Dict[str, Any],
    *,
    attempts: Sequence[Mapping[str, Any]],
    platesolve_attempted: bool,
    platesolve_ok: bool,
    skip_reason: str,
) -> Dict[str, Any]:
    normalized_attempts = [_json_safe(dict(attempt)) for attempt in attempts]
    result = evidence
    result["attempts"] = normalized_attempts
    result["attempts_sha256"] = canonical_sha256(normalized_attempts)
    successful_attempt_catalog = None
    for attempt in normalized_attempts:
        if attempt.get("status") == "ok":
            label = str(attempt.get("label") or "")
            successful_attempt_catalog = label.rsplit("catalog:", 1)[-1] or None
            break
    selected_catalog = successful_attempt_catalog if platesolve_ok else None
    siril = next(
        (
            backend
            for backend in result.get("backends", [])
            if backend.get("id") == "siril_platesolve"
        ),
        None,
    )
    if siril is None:
        result["finalized_after_solve"] = True
        return result
    siril["attempted"] = bool(platesolve_attempted and normalized_attempts)
    siril["selected"] = bool(platesolve_ok)
    siril["result"] = (
        "accepted"
        if platesolve_ok
        else "failed"
        if platesolve_attempted and normalized_attempts
        else "not_run"
    )
    if skip_reason:
        siril["reason_codes"] = [skip_reason]
    elif platesolve_attempted and not platesolve_ok:
        siril["reason_codes"] = ["platesolve_failed"]
    result["selection"] = {
        "backend": "siril_platesolve" if platesolve_ok else None,
        "catalog": selected_catalog,
        "successful_attempt_catalog": successful_attempt_catalog,
        "reason": (
            "platesolve_accepted"
            if platesolve_ok
            else (
                skip_reason
                or ("platesolve_failed" if platesolve_attempted else "not_attempted")
            )
        ),
    }
    for catalog in result.get("catalog_sources", []):
        catalog["selected"] = bool(
            selected_catalog and catalog.get("id") == selected_catalog
        )
        catalog_attempts = [
            attempt
            for attempt in normalized_attempts
            if str(attempt.get("label") or "").endswith(
                f"catalog:{catalog.get('id')}"
            )
        ]
        catalog["attempted"] = any(
            attempt.get("status") != "skipped" for attempt in catalog_attempts
        )
        catalog["result"] = (
            "accepted"
            if catalog.get("selected")
            else "failed"
            if any(attempt.get("status") == "failed" for attempt in catalog_attempts)
            else "skipped"
            if catalog_attempts
            else "not_run"
        )
    result["finalized_after_solve"] = True
    return result


def write_evidence_artifact(
    process_dir: Path,
    filename: str,
    payload: Mapping[str, Any],
    *,
    log: Any = None,
) -> Dict[str, Any]:
    path = Path(process_dir) / filename
    try:
        validate_evidence_payload(payload)
        text = json.dumps(
            _json_safe(dict(payload)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        path.write_text(text, encoding="utf-8")
        digest = file_sha256(path)
        return {
            "schema": payload.get("schema"),
            "status": payload.get("status"),
            "path": filename,
            "sha256": digest,
            "hash_scope": "artifact_file_bytes",
        }
    except (OSError, TypeError, ValueError) as error:
        debug = getattr(log, "debug", None)
        if callable(debug):
            debug(f"Stage4 evidence unavailable ({filename}): {error}")
        return {
            "schema": payload.get("schema"),
            "status": "unavailable",
            "path": None,
            "sha256": None,
            "reason": "artifact_write_failed",
        }


__all__ = [
    "FILTER_HEADER_EVIDENCE_NAME",
    "FILTER_HEADER_EVIDENCE_SCHEMA",
    "SOLVER_BACKEND_IDS",
    "SOLVER_CAPABILITIES_NAME",
    "SOLVER_CAPABILITIES_SCHEMA",
    "build_filter_header_evidence",
    "build_solver_capabilities",
    "canonical_sha256",
    "capture_solver_candidate",
    "file_sha256",
    "finalize_filter_header_evidence",
    "finalize_solver_capabilities",
    "not_applicable_snapshot",
    "read_fits_header_snapshot",
    "validate_evidence_payload",
    "validate_filter_header_evidence",
    "validate_solver_capabilities",
    "write_evidence_artifact",
]
