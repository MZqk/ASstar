"""Decoded-pixel identity gates for Stage 10 formal delivery artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from astropy.io import fits

import managed_output
import display_rendition

from stage8_starless_finish import (
    DECODED_PIXEL_SHA256_METHOD,
    canonical_decoded_pixel_sha256,
    decoded_science_image_hdus,
    persisted_fits_decoded_pixel_sha256,
    pixel_sha256,
)


SCHEMA = "starun.final-artifact-identity.v1"
MANAGED_SOURCE_METHOD = DECODED_PIXEL_SHA256_METHOD
MANAGED_DERIVATIVE_METHOD = "canonical_uint16_top_down_float32_chw_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _requested_formats(value: Any) -> set[str]:
    requested = {
        item.strip().lower()
        for item in str(value or "all").split(",")
        if item.strip()
    }
    if not requested or "all" in requested:
        return {"fit", "tif", "png"}
    if "tiff" in requested:
        requested.add("tif")
    return requested


def _decoded_stage10_pixels(path: Path) -> np.ndarray:
    with fits.open(
        Path(path),
        memmap=False,
        do_not_scale_image_data=False,
    ) as hdul:
        image_hdus = decoded_science_image_hdus(hdul)
        if len(image_hdus) != 1:
            raise ValueError(
                "formal Stage10 FITS must contain exactly one decoded image HDU"
            )
        pixels = np.ascontiguousarray(
            np.asarray(image_hdus[0].data, dtype=np.float32)
        )
    canonical_decoded_pixel_sha256(pixels)
    return pixels


def _normalized_basenames(values: Any) -> set[str]:
    return {
        str(value).strip().lower()
        for value in values or ()
        if str(value).strip()
        and Path(str(value).strip()).name == str(value).strip()
    }


def _managed_artifact_verification(
    work_dir: Path,
    outputs: Mapping[str, Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    expected_source_pixels: Optional[np.ndarray],
    expected_source_pixel_sha256: str,
    expected_derivative_pixel_sha256: str,
    formal_basenames: set[str],
) -> Dict[str, Any]:
    role = str(artifact.get("role") or "")
    name = str(artifact.get("name") or "")
    chain = artifact.get("pixel_chain")
    record = outputs.get(name)
    path = work_dir / name if name else Path()
    issues = []
    if role not in {"display", "editable"}:
        issues.append("managed_artifact_role_invalid")
    expected_suffix = ".png" if role == "display" else ".tif"
    expected_names = {
        (
            f"{base}_display_srgb.png"
            if role == "display"
            else f"{base}_edit_srgb.tif"
        )
        for base in formal_basenames
    }
    if (
        not name
        or Path(name).name != name
        or Path(name).suffix.lower() != expected_suffix
        or name.lower() not in expected_names
    ):
        issues.append("managed_artifact_name_invalid")
    if artifact.get("status") != "written":
        issues.append("managed_artifact_not_written")
    if not name or not path.is_file() or not isinstance(record, Mapping):
        issues.append("managed_artifact_output_missing")
    if path.is_symlink():
        issues.append("managed_artifact_symlink_forbidden")
    actual_file_sha = _sha256(path) if path.is_file() else None
    if not actual_file_sha or actual_file_sha != str(record.get("sha256") or ""):
        issues.append("managed_artifact_manifest_sha_mismatch")
    if actual_file_sha != str(artifact.get("sha256") or ""):
        issues.append("managed_artifact_report_sha_mismatch")
    if not isinstance(chain, Mapping):
        issues.append("managed_artifact_pixel_chain_missing")
        chain = {}
    if chain.get("accepted") is not True:
        issues.append("managed_artifact_pixel_chain_rejected")
    if chain.get("schema") != "starun.managed-output-pixel-chain.v1":
        issues.append("managed_artifact_pixel_chain_schema_invalid")
    if chain.get("source_pixel_sha256") != expected_source_pixel_sha256:
        issues.append("managed_artifact_source_pixel_sha_mismatch")
    if chain.get("source_pixel_sha256_method") != MANAGED_SOURCE_METHOD:
        issues.append("managed_artifact_source_pixel_method_invalid")
    if chain.get("decoded_pixel_sha256_method") != MANAGED_DERIVATIVE_METHOD:
        issues.append("managed_artifact_decoded_pixel_method_invalid")
    reported_expected_sha = str(chain.get("expected_pixel_sha256") or "")
    reported_decoded_sha = str(chain.get("decoded_pixel_sha256") or "")
    artifact_expected_derivative_sha = expected_derivative_pixel_sha256
    display_transform = artifact.get("display_transform")
    if (
        role == "display"
        and isinstance(display_transform, Mapping)
        and display_transform.get("derivative_pixels_changed") is True
    ):
        try:
            if expected_source_pixels is None:
                raise ValueError("Stage10 source pixels are unavailable")
            if not display_rendition.validate_review_contract(
                dict(display_transform)
            ):
                raise ValueError("display rendition contract is invalid")
            if (
                display_transform.get("observer_only") is not True
                or (
                    display_transform.get("rgb_mapping") or {}
                ).get("source_pixels_changed") is not False
                or str(display_transform.get("source_stem") or "")
                != "stage10_final"
            ):
                raise ValueError("display rendition source binding is invalid")
            source_display = np.flip(
                np.asarray(expected_source_pixels, dtype=np.float32),
                axis=1,
            )
            rendered_display = display_rendition.apply_review_contract(
                source_display,
                dict(display_transform),
            )
            expected_display = (
                managed_output.canonical_managed_derivative_pixels(
                    np.flip(rendered_display, axis=1)
                )
            )
            artifact_expected_derivative_sha = pixel_sha256(
                np.ascontiguousarray(expected_display, dtype=np.float32)
            )
        except (TypeError, ValueError, FloatingPointError) as error:
            issues.append(f"managed_display_transform_invalid:{error}")
    actual_derivative_sha = ""
    try:
        if role == "display":
            decoded = managed_output.read_managed_display_png(path)
        elif role == "editable":
            decoded = managed_output.read_managed_edit_tiff(path)
        else:
            raise ValueError("unsupported managed artifact role")
        actual_derivative_sha = pixel_sha256(
            np.ascontiguousarray(decoded, dtype=np.float32)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        issues.append(f"managed_artifact_decode_failed:{error}")
    if (
        len(artifact_expected_derivative_sha) != 64
        or reported_expected_sha != artifact_expected_derivative_sha
        or reported_decoded_sha != artifact_expected_derivative_sha
        or actual_derivative_sha != artifact_expected_derivative_sha
    ):
        issues.append("managed_artifact_decoded_pixel_sha_mismatch")
    return {
        "role": role,
        "name": name,
        "accepted": not issues,
        "sha256": actual_file_sha,
        "decoded_pixel_sha256": actual_derivative_sha or None,
        "issues": issues,
    }


def verify_formal_artifacts(
    *,
    work_dir: Path,
    process_dir: Optional[Path],
    outputs: Mapping[str, Mapping[str, Any]],
    output_format: Any,
    formal_basenames: Any = (),
) -> Dict[str, Any]:
    """Verify root outputs against persisted Stage 10 decoded pixels."""

    root = Path(work_dir)
    process = Path(process_dir) if process_dir is not None else root / "process"
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "rejected",
        "accepted": False,
        "source": {},
        "scientific": [],
        "managed": [],
        "formal_outputs": [],
        "issues": [],
    }
    issues: list[str] = []
    formal_outputs: list[str] = []
    source_path = process / "stage10_final.fit"
    source_pixel_sha = ""
    expected_derivative_pixel_sha = ""
    source_pixels: Optional[np.ndarray] = None
    allowed_basenames = _normalized_basenames(formal_basenames)
    managed_report = _read_json(process / "managed_output_report.json")
    report["formal_basenames"] = sorted(allowed_basenames)
    if not allowed_basenames:
        issues.append("formal_output_basenames_unavailable")
    source_error = ""
    try:
        if not source_path.is_file():
            raise ValueError("stage10_final.fit is unavailable")
        if source_path.is_symlink():
            raise ValueError("stage10_final.fit must not be a symlink")
        source_pixels = _decoded_stage10_pixels(source_path)
        source_pixel_sha = canonical_decoded_pixel_sha256(source_pixels)
        expected_derivative_pixel_sha = pixel_sha256(
            managed_output.canonical_managed_derivative_pixels(source_pixels)
        )
        report["source"] = {
            "artifact": source_path.name,
            "sha256": _sha256(source_path),
            "pixel_sha256": source_pixel_sha,
            "pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
            "managed_derivative_pixel_sha256": (
                expected_derivative_pixel_sha
            ),
            "managed_derivative_pixel_sha256_method": (
                MANAGED_DERIVATIVE_METHOD
            ),
        }
    except (OSError, TypeError, ValueError) as error:
        source_error = str(error)

    if source_pixels is None:
        source_record = managed_report.get("source_pixels")
        compacted_candidates = []
        compacted_issues = []
        if (
            managed_report.get("schema") != "starun.managed-output.v2"
            or managed_report.get("ready") is not True
            or bool(managed_report.get("issues") or [])
            or not isinstance(source_record, Mapping)
            or source_record.get("checkpoint") != "stage10_final.fit"
            or source_record.get("pixel_sha256_method")
            != MANAGED_SOURCE_METHOD
            or len(str(source_record.get("pixel_sha256") or "")) != 64
        ):
            compacted_issues.append(
                "managed Stage10 source anchor is unavailable or invalid"
            )
        else:
            expected_compacted_sha = str(
                source_record.get("pixel_sha256") or ""
            )
            for name, record in outputs.items():
                candidate_name = str(name)
                candidate_path = root / candidate_name
                if (
                    Path(candidate_name).suffix.lower()
                    not in {".fit", ".fits", ".fts"}
                    or "review" in candidate_name.lower()
                    or Path(candidate_name).stem.lower()
                    not in allowed_basenames
                    or not isinstance(record, Mapping)
                    or Path(candidate_name).name != candidate_name
                ):
                    continue
                try:
                    if not candidate_path.is_file():
                        raise ValueError("formal scientific FITS is unavailable")
                    if candidate_path.is_symlink():
                        raise ValueError("formal scientific FITS is a symlink")
                    candidate_file_sha = _sha256(candidate_path)
                    if candidate_file_sha != str(record.get("sha256") or ""):
                        raise ValueError(
                            "formal scientific FITS manifest SHA mismatch"
                        )
                    candidate_pixels = _decoded_stage10_pixels(candidate_path)
                    candidate_pixel_sha = canonical_decoded_pixel_sha256(
                        candidate_pixels
                    )
                    if candidate_pixel_sha != expected_compacted_sha:
                        raise ValueError(
                            "formal scientific FITS decoded pixels do not match "
                            "the frozen Stage10 source anchor"
                        )
                    compacted_candidates.append(
                        (
                            candidate_name,
                            candidate_file_sha,
                            candidate_pixels,
                            candidate_pixel_sha,
                        )
                    )
                except (OSError, TypeError, ValueError) as candidate_error:
                    compacted_issues.append(
                        f"{candidate_name}:{candidate_error}"
                    )
        if len(compacted_candidates) == 1:
            (
                compacted_name,
                compacted_file_sha,
                source_pixels,
                source_pixel_sha,
            ) = compacted_candidates[0]
            expected_derivative_pixel_sha = pixel_sha256(
                managed_output.canonical_managed_derivative_pixels(
                    source_pixels
                )
            )
            report["source"] = {
                "artifact": compacted_name,
                "checkpoint": "stage10_final.fit",
                "role": "compacted_scientific_archive_anchor",
                "compacted_source": True,
                "sha256": compacted_file_sha,
                "pixel_sha256": source_pixel_sha,
                "pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
                "managed_derivative_pixel_sha256": (
                    expected_derivative_pixel_sha
                ),
                "managed_derivative_pixel_sha256_method": (
                    MANAGED_DERIVATIVE_METHOD
                ),
            }
        else:
            detail = source_error or "stage10_final.fit is unavailable"
            if compacted_candidates:
                compacted_issues.append(
                    "multiple compacted scientific anchors matched"
                )
            if compacted_issues:
                detail += "; " + "; ".join(compacted_issues)
            issues.append(f"stage10_source_identity_unavailable:{detail}")

    scientific_records = []
    for name, record in outputs.items():
        suffix = Path(str(name)).suffix.lower()
        if suffix not in {".fit", ".fits", ".fts"}:
            continue
        if "review" in str(name).lower() or not isinstance(record, Mapping):
            continue
        if Path(str(name)).stem.lower() not in allowed_basenames:
            continue
        path = root / str(name)
        item: Dict[str, Any] = {
            "name": str(name),
            "accepted": False,
            "issues": [],
        }
        try:
            if path.is_symlink():
                raise ValueError("formal FITS must not be a symlink")
            actual_file_sha = _sha256(path)
            decoded_sha = persisted_fits_decoded_pixel_sha256(path)
            item.update(
                sha256=actual_file_sha,
                decoded_pixel_sha256=decoded_sha,
                decoded_pixel_sha256_method=DECODED_PIXEL_SHA256_METHOD,
            )
            if actual_file_sha != str(record.get("sha256") or ""):
                item["issues"].append("output_manifest_sha_mismatch")
            if not source_pixel_sha or decoded_sha != source_pixel_sha:
                item["issues"].append("stage10_decoded_pixel_sha_mismatch")
        except (OSError, TypeError, ValueError) as error:
            item["issues"].append(f"formal_fits_decode_failed:{error}")
        item["accepted"] = not item["issues"]
        if item["accepted"]:
            formal_outputs.append(str(name))
        else:
            issues.append("scientific_fit_unverified:" + str(name))
        scientific_records.append(item)
    report["scientific"] = scientific_records
    if not any(item.get("accepted") is True for item in scientific_records):
        issues.append("no_stage10_pixel_bound_scientific_fits")

    requested = _requested_formats(output_format)
    managed_required_roles = {
        role
        for extension, role in (("png", "display"), ("tif", "editable"))
        if extension in requested
    }
    managed_records = []
    if managed_required_roles:
        source_record = managed_report.get("source_pixels")
        if (
            managed_report.get("schema") != "starun.managed-output.v2"
            or managed_report.get("ready") is not True
            or bool(managed_report.get("issues") or [])
            or not isinstance(source_record, Mapping)
            or source_record.get("pixel_sha256") != source_pixel_sha
            or source_record.get("pixel_sha256_method")
            != MANAGED_SOURCE_METHOD
        ):
            issues.append("managed_output_source_identity_unverified")
        role_counts = {role: 0 for role in managed_required_roles}
        for artifact in managed_report.get("artifacts") or []:
            if not isinstance(artifact, Mapping):
                continue
            if str(artifact.get("role") or "") not in managed_required_roles:
                continue
            role_counts[str(artifact.get("role") or "")] += 1
            verification = _managed_artifact_verification(
                root,
                outputs,
                artifact,
                expected_source_pixels=source_pixels,
                expected_source_pixel_sha256=source_pixel_sha,
                expected_derivative_pixel_sha256=(
                    expected_derivative_pixel_sha
                ),
                formal_basenames=allowed_basenames,
            )
            managed_records.append(verification)
            if verification["accepted"]:
                formal_outputs.append(str(verification["name"]))
            else:
                issues.append(
                    "managed_artifact_unverified:"
                    + str(verification.get("role") or "unknown")
                )
        duplicate_roles = sorted(
            role for role, count in role_counts.items() if count != 1
        )
        if duplicate_roles:
            issues.append(
                "managed_output_role_cardinality_invalid:"
                + ",".join(duplicate_roles)
            )
        accepted_roles = {
            str(item.get("role") or "")
            for item in managed_records
            if item.get("accepted") is True
        }
        missing_roles = sorted(managed_required_roles - accepted_roles)
        if missing_roles:
            issues.append(
                "managed_output_roles_unverified:" + ",".join(missing_roles)
            )
    report["managed"] = managed_records
    report["requested_formats"] = sorted(requested)
    report["required_managed_roles"] = sorted(managed_required_roles)
    report["formal_outputs"] = sorted(set(formal_outputs))
    report["issues"] = issues
    report["accepted"] = not issues
    report["status"] = "accepted" if report["accepted"] else "rejected"
    return report


__all__ = [
    "MANAGED_DERIVATIVE_METHOD",
    "MANAGED_SOURCE_METHOD",
    "SCHEMA",
    "verify_formal_artifacts",
]
