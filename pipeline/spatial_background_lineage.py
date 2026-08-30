"""SHA-bound Stage 3 sky support and downstream spatial-gradient gates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


LINEAGE_SCHEMA = "starun.stage3-spatial-background-lineage.v2"
LEGACY_LINEAGE_SCHEMAS = {
    "starun.stage3-spatial-background-lineage.v1",
}
LINEAGE_CHAIN_SCHEMA = "starun.stage3-spatial-background-lineage-chain.v1"
STAGE7_SCHEMA = "starun.stage7-spatial-chroma-lineage.v1"
STAGE7_REFERENCE_SCHEMA = "starun.stage7-spatial-background-reference.v1"
STAGE7_REFERENCE_NAME = "stage7_spatial_background_reference.json"
STAGE7_QUALITY_NAME = "stage7_stretch_quality.json"
STAGE7_REFERENCE_REVIEW_ONLY = (
    "stage7_spatial_background_reference_unavailable_due_to_review_only"
)
STAGE7_REFERENCE_MISSING = "stage7_spatial_background_reference_missing"
STAGE7_REFERENCE_READ_FAILED = (
    "stage7_spatial_background_reference_read_failed"
)
STAGE7_REFERENCE_INVALID = "stage7_spatial_background_reference_invalid"
FINAL_SCHEMA = "starun.final-spatial-background-gradient.v2"
SIGNIFICANCE_SIGMA = 3.0
GROWTH_MAX = 1.25
NORMALIZED_MATERIAL_FLOOR = 1.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(image: Any) -> str:
    values = np.ascontiguousarray(np.asarray(image, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in values.shape)).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lineage_chain_digest(payload: Dict[str, Any]) -> str:
    """Digest every field required to treat a Stage 3 lineage as formal."""

    chain = {
        "schema": payload.get("schema"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "accepted": payload.get("accepted"),
        "review_required": payload.get("review_required"),
        "processing_route": payload.get("processing_route"),
        "image_shape": payload.get("image_shape"),
        "channel_layout": payload.get("channel_layout"),
        "patch_radius": payload.get("patch_radius"),
        "fit_points": payload.get("fit_points"),
        "validation_points": payload.get("validation_points"),
        "support_artifact": payload.get("support_artifact"),
        "support_kind": payload.get("support_kind"),
        "support_pixel_count": payload.get("support_pixel_count"),
        "support_coverage": payload.get("support_coverage"),
        "sample_patch_support_pixel_count": payload.get(
            "sample_patch_support_pixel_count"
        ),
        "sample_patch_min_support_pixel_count": payload.get(
            "sample_patch_min_support_pixel_count"
        ),
        "support_sha256": payload.get("support_sha256"),
        "stage3_input_sha256": payload.get("stage3_input_sha256"),
        "stage3_input_pixel_sha256": payload.get(
            "stage3_input_pixel_sha256"
        ),
        "stage3_output_sha256": payload.get("stage3_output_sha256"),
        "stage3_output_pixel_sha256": payload.get(
            "stage3_output_pixel_sha256"
        ),
        "reference_metrics": payload.get("reference_metrics"),
        "reference_plane": payload.get("reference_plane"),
        "projection_schema": payload.get("projection_schema"),
        "projection_reason_code": payload.get("projection_reason_code"),
        "selected_components": payload.get("selected_components"),
        "unresolved_components": payload.get("unresolved_components"),
    }
    return _json_sha256(chain)


def build_sample_patch_support(
    image_shape: Sequence[int],
    points: Sequence[Tuple[float, float]],
    support: np.ndarray,
    *,
    patch_radius: int,
) -> Tuple[np.ndarray, list[int]]:
    """Rebuild the point-sampling subset inside a full frozen sky mask."""

    if len(image_shape) != 2:
        raise ValueError("Stage 3 spatial lineage image shape is invalid")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0 or int(patch_radius) <= 0:
        raise ValueError("Stage 3 spatial sample geometry is invalid")
    frozen = np.asarray(support, dtype=bool)
    if frozen.shape != (height, width):
        raise ValueError("Stage 3 frozen support mask shape mismatch")
    sample_support = np.zeros((height, width), dtype=bool)
    point_support_counts: list[int] = []
    for raw_x, raw_y in points:
        x_value = float(raw_x)
        y_value = float(raw_y)
        if (
            not math.isfinite(x_value)
            or not math.isfinite(y_value)
            or x_value < 0.0
            or x_value > float(width - 1)
            or y_value < 0.0
            or y_value > float(height - 1)
        ):
            raise ValueError("Stage 3 spatial sample point is out of bounds")
        x = int(round(x_value))
        y = int(round(height - 1 - y_value))
        x0 = max(0, x - patch_radius)
        x1 = min(width, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(height, y + patch_radius + 1)
        patch = frozen[y0:y1, x0:x1]
        count = int(np.count_nonzero(patch))
        if count <= 0:
            raise ValueError("frozen spatial background patch has no support")
        point_support_counts.append(count)
        sample_support[y0:y1, x0:x1] |= patch
    return sample_support, point_support_counts


def seal_lineage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy sealed with the formal Stage 3 lineage chain digest."""

    sealed = dict(payload)
    sealed["chain_digest"] = {
        "schema": LINEAGE_CHAIN_SCHEMA,
        "algorithm": "sha256",
        "sha256": lineage_chain_digest(sealed),
    }
    return sealed


def _validated_points(payload: Any, *, name: str) -> list[list[float]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Stage 3 {name} samples are unavailable")
    points: list[list[float]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Stage 3 {name} samples are invalid")
        x = float(item[0])
        y = float(item[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Stage 3 {name} samples contain non-finite values")
        points.append([x, y])
    return points


def _rgb_chw(image: Any) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim != 3:
        raise ValueError("spatial background gate requires an RGB image")
    if source.shape[0] in (3, 4):
        rgb = source[:3]
    elif source.shape[-1] in (3, 4):
        rgb = np.moveaxis(source[..., :3], -1, 0)
    else:
        raise ValueError(f"unsupported spatial background RGB shape: {source.shape}")
    values = np.asarray(rgb, dtype=np.float64)
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("spatial background image contains non-finite pixels")
    return values


def _patch_medians(
    plane: np.ndarray,
    points: Sequence[Tuple[float, float]],
    support: np.ndarray,
    patch_radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = plane.shape
    rows = []
    values = []
    for raw_x, raw_y in points:
        x = int(round(float(raw_x)))
        y = int(round(height - 1 - float(raw_y)))
        x0 = max(0, x - patch_radius)
        x1 = min(width, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(height, y + patch_radius + 1)
        patch = np.asarray(plane[y0:y1, x0:x1], dtype=np.float64)
        patch_support = support[y0:y1, x0:x1]
        finite = patch[patch_support & np.isfinite(patch)]
        if finite.size == 0:
            raise ValueError("frozen spatial background patch has no support")
        rows.append(
            [
                1.0,
                float(raw_x) / max(width - 1, 1),
                float(y) / max(height - 1, 1),
            ]
        )
        values.append(float(np.median(finite)))
    return np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _plane_metrics(design: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design,
        values,
        rcond=None,
    )
    fitted = design @ coefficients
    residual = values - fitted
    dof = max(int(design.shape[0]) - 3, 1)
    sigma2 = float(np.sum(np.square(residual)) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    slope_variance = max(float(np.trace(covariance[1:3, 1:3])), 1e-30)
    slope = np.asarray(coefficients[1:3], dtype=np.float64)
    span = float(abs(float(slope[0])) + abs(float(slope[1])))
    significance = float(np.linalg.norm(slope) / math.sqrt(slope_variance))
    return {
        "coefficients": [float(value) for value in coefficients],
        "rank": int(rank),
        "condition_number": float(np.linalg.cond(design)),
        "singular_values": [float(value) for value in singular_values],
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "slope_span": span,
        "slope_significance_sigma": significance,
        "material": bool(span > NORMALIZED_MATERIAL_FLOOR),
    }


def measure_spatial_background_planes(
    image: Any,
    support: np.ndarray,
    points: Sequence[Tuple[float, float]],
    *,
    patch_radius: int,
) -> Dict[str, Any]:
    rgb = _rgb_chw(image)
    height, width = rgb.shape[1:]
    support_mask = np.asarray(support, dtype=bool)
    if support_mask.shape != (height, width):
        raise ValueError("frozen spatial support and image shapes differ")
    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    intensity = np.maximum(rgb[0] + rgb[1] + rgb[2], 0.015)
    planes = {
        "luma": luma / max(float(np.median(luma[support_mask])), 0.015),
        "R-G": (rgb[0] - rgb[1]) / intensity,
        "B-G": (rgb[2] - rgb[1]) / intensity,
    }
    metrics: Dict[str, Any] = {}
    for name, plane in planes.items():
        design, medians = _patch_medians(
            plane,
            points,
            support_mask,
            patch_radius,
        )
        metrics[name] = {
            **_plane_metrics(design, medians),
            "patch_count": int(len(medians)),
            "patch_median": float(np.median(medians)),
        }
    return metrics


def load_lineage(process_dir: Optional[Path]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": LINEAGE_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "issues": [],
    }
    if process_dir is None:
        report["issues"] = ["process directory is unavailable"]
        return report
    root = Path(process_dir)
    json_path = root / "stage3_spatial_background_lineage.json"
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        schema = str(payload.get("schema") or "")
        if schema in LEGACY_LINEAGE_SCHEMAS:
            legacy = dict(payload)
            legacy.update(
                status="legacy_nonformal",
                accepted=False,
                formal_eligible=False,
                legacy_delivery_contract=True,
                issues=["legacy Stage 3 spatial lineage is readable but non-formal"],
            )
            return legacy
        if schema != LINEAGE_SCHEMA:
            raise ValueError("Stage 3 spatial lineage schema mismatch")
        required = {
            "run_id",
            "review_required",
            "processing_route",
            "image_shape",
            "channel_layout",
            "patch_radius",
            "fit_points",
            "validation_points",
            "support_artifact",
            "support_kind",
            "support_pixel_count",
            "support_coverage",
            "sample_patch_support_pixel_count",
            "sample_patch_min_support_pixel_count",
            "support_sha256",
            "stage3_input_sha256",
            "stage3_input_pixel_sha256",
            "stage3_output_sha256",
            "stage3_output_pixel_sha256",
            "reference_metrics",
            "reference_plane",
            "chain_digest",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                "Stage 3 formal spatial lineage fields are missing: "
                + ", ".join(missing)
            )
        if (
            payload.get("status") != "accepted"
            or payload.get("accepted") is not True
            or payload.get("review_required") is not False
        ):
            raise ValueError("Stage 3 spatial lineage is not formally accepted")
        route = str(payload.get("processing_route") or "")
        if route not in {"background_correction", "verified_noop"}:
            raise ValueError("Stage 3 spatial lineage processing route is invalid")
        if not str(payload.get("run_id") or "").strip():
            raise ValueError("Stage 3 spatial lineage run_id is unavailable")
        image_shape = payload.get("image_shape")
        if (
            not isinstance(image_shape, list)
            or len(image_shape) != 2
            or any(int(value) <= 0 for value in image_shape)
        ):
            raise ValueError("Stage 3 spatial lineage image shape is invalid")
        fit_points = _validated_points(payload.get("fit_points"), name="fit")
        validation_points = _validated_points(
            payload.get("validation_points"),
            name="validation",
        )
        patch_radius = int(payload.get("patch_radius"))
        if patch_radius <= 0:
            raise ValueError("Stage 3 spatial lineage patch radius is invalid")
        chain_digest = dict(payload.get("chain_digest") or {})
        if (
            chain_digest.get("schema") != LINEAGE_CHAIN_SCHEMA
            or chain_digest.get("algorithm") != "sha256"
            or str(chain_digest.get("sha256") or "")
            != lineage_chain_digest(payload)
        ):
            raise ValueError("Stage 3 spatial lineage chain digest mismatch")
        artifact_name = str(
            payload.get("support_artifact") or ""
        )
        artifact = root / artifact_name
        expected_sha = str(payload.get("support_sha256") or "")
        if not artifact.is_file() or not expected_sha:
            raise ValueError("Stage 3 spatial support artifact is unavailable")
        actual_sha = _sha256(artifact)
        if actual_sha != expected_sha:
            raise ValueError("Stage 3 spatial support SHA mismatch")
        with fits.open(artifact, memmap=False, do_not_scale_image_data=False) as hdul:
            support = np.asarray(hdul[0].data, dtype=bool)
        if support.ndim != 2 or int(np.count_nonzero(support)) < 64:
            raise ValueError("Stage 3 spatial support artifact is invalid")
        if list(support.shape) != [int(value) for value in image_shape]:
            raise ValueError("Stage 3 spatial support/image shape mismatch")
        if int(np.count_nonzero(support)) != int(payload["support_pixel_count"]):
            raise ValueError("Stage 3 spatial support pixel count mismatch")
        if payload.get("support_kind") != "candidate_independent_full_sky_mask":
            raise ValueError("Stage 3 spatial support kind is not formal")
        support_coverage = float(payload.get("support_coverage"))
        actual_coverage = float(np.count_nonzero(support)) / float(support.size)
        if (
            not math.isfinite(support_coverage)
            or support_coverage <= 0.0
            or support_coverage > 1.0
            or not math.isclose(
                support_coverage,
                actual_coverage,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("Stage 3 spatial support coverage mismatch")
        sample_support, point_support_counts = build_sample_patch_support(
            support.shape,
            [*fit_points, *validation_points],
            support,
            patch_radius=patch_radius,
        )
        if int(np.count_nonzero(sample_support)) != int(
            payload.get("sample_patch_support_pixel_count")
        ):
            raise ValueError("Stage 3 spatial sample support count mismatch")
        if min(point_support_counts) != int(
            payload.get("sample_patch_min_support_pixel_count")
        ):
            raise ValueError("Stage 3 spatial sample minimum support mismatch")
        canonical = root / "stage3_bgremoved.fit"
        expected_output_sha = str(payload.get("stage3_output_sha256") or "")
        if not canonical.is_file() or not expected_output_sha:
            raise ValueError("Stage 3 canonical output lineage is unavailable")
        if _sha256(canonical) != expected_output_sha:
            raise ValueError("Stage 3 canonical output SHA mismatch")
        baseline = root / "stage3_bg_input.fit"
        expected_input_sha = str(payload.get("stage3_input_sha256") or "")
        if not baseline.is_file() or not expected_input_sha:
            raise ValueError("Stage 3 canonical input lineage is unavailable")
        if _sha256(baseline) != expected_input_sha:
            raise ValueError("Stage 3 canonical input SHA mismatch")
        expected_output_pixel_sha = str(
            payload.get("stage3_output_pixel_sha256") or ""
        )
        if not expected_output_pixel_sha:
            raise ValueError("Stage 3 canonical output pixel lineage is unavailable")
        with fits.open(
            canonical,
            memmap=False,
            do_not_scale_image_data=False,
        ) as hdul:
            output_pixels = np.asarray(hdul[0].data)
        if _array_sha256(output_pixels) != expected_output_pixel_sha:
            raise ValueError("Stage 3 canonical output pixel SHA mismatch")
        expected_input_pixel_sha = str(
            payload.get("stage3_input_pixel_sha256") or ""
        )
        if not expected_input_pixel_sha:
            raise ValueError("Stage 3 canonical input pixel lineage is unavailable")
        with fits.open(
            baseline,
            memmap=False,
            do_not_scale_image_data=False,
        ) as hdul:
            input_pixels = np.asarray(hdul[0].data)
        if _array_sha256(input_pixels) != expected_input_pixel_sha:
            raise ValueError("Stage 3 canonical input pixel SHA mismatch")
        if route == "verified_noop" and (
            expected_input_pixel_sha != expected_output_pixel_sha
            or input_pixels.shape != output_pixels.shape
            or input_pixels.dtype != output_pixels.dtype
            or not np.array_equal(input_pixels, output_pixels, equal_nan=True)
        ):
            raise ValueError("verified Stage 3 no-op pixel identity mismatch")
        reference_plane = dict(payload.get("reference_plane") or {})
        expected_reference_sha = str(reference_plane.get("sha256") or "")
        if not expected_reference_sha:
            raise ValueError("Stage 3 reference plane digest is unavailable")
        if _json_sha256(reference_plane.get("components") or {}) != (
            expected_reference_sha
        ):
            raise ValueError("Stage 3 reference plane SHA mismatch")
        reference_metrics = dict(payload.get("reference_metrics") or {})
        if any(name not in reference_metrics for name in ("luma", "R-G", "B-G")):
            raise ValueError("Stage 3 reference plane metrics are incomplete")
        payload = dict(payload)
        payload["support_mask"] = support
        payload["formal_eligible"] = True
        payload["fit_points"] = fit_points
        payload["validation_points"] = validation_points
        return payload
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        report["issues"] = [str(error)]
        return report


def assess_stage7_spatial_chroma(
    process_dir: Optional[Path],
    actual_candidate: Any,
    expected_candidate: Any,
    *,
    transform_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": STAGE7_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "issues": ["stage7_spatial_chroma_lineage_unverified"],
        "growth_max": GROWTH_MAX,
        "components": {},
    }
    lineage = load_lineage(process_dir)
    if not bool(lineage.get("accepted", False)):
        report["lineage"] = {
            key: value
            for key, value in lineage.items()
            if key != "support_mask"
        }
        return report
    if expected_candidate is None:
        report["issues"] = ["stage7_spatial_chroma_transform_replay_unavailable"]
        return report
    try:
        points = [
            tuple(item)
            for item in [
                *(lineage.get("fit_points") or []),
                *(lineage.get("validation_points") or []),
            ]
        ]
        expected = measure_spatial_background_planes(
            expected_candidate,
            lineage["support_mask"],
            points,
            patch_radius=int(lineage.get("patch_radius") or 12),
        )
        actual = measure_spatial_background_planes(
            actual_candidate,
            lineage["support_mask"],
            points,
            patch_radius=int(lineage.get("patch_radius") or 12),
        )
        issues = []
        for name in ("R-G", "B-G"):
            expected_span = float(expected[name]["slope_span"])
            actual_span = float(actual[name]["slope_span"])
            ratio = actual_span / max(expected_span, NORMALIZED_MATERIAL_FLOOR)
            newly_significant = bool(
                actual[name]["material"]
                and float(actual[name]["slope_significance_sigma"])
                >= SIGNIFICANCE_SIGMA
                and (
                    not expected[name]["material"]
                    or float(expected[name]["slope_significance_sigma"])
                    < SIGNIFICANCE_SIGMA
                )
            )
            exceeded = bool(ratio > GROWTH_MAX and actual[name]["material"])
            if newly_significant or exceeded:
                issues.append(f"stage7_spatial_chroma_amplification:{name}")
            report["components"][name] = {
                "expected": expected[name],
                "actual": actual[name],
                "slope_growth": ratio,
                "newly_significant": newly_significant,
                "growth_exceeded": exceeded,
                "accepted": not (newly_significant or exceeded),
            }
        report.update(
            status="rejected" if issues else "ok",
            accepted=not issues,
            issues=issues,
            support_sha256=lineage.get("support_sha256"),
            display_reference={
                "schema": STAGE7_REFERENCE_SCHEMA,
                "status": "ready",
                "support_sha256": lineage.get("support_sha256"),
                "expected_metrics": expected,
                "expected_pixel_sha256": _array_sha256(expected_candidate),
                "candidate_pixel_sha256": _array_sha256(actual_candidate),
                "transform_identity": dict(transform_identity or {}),
            },
        )
        return report
    except (IndexError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        report["issues"] = [str(error)]
        return report


def build_stage7_display_reference(
    process_dir: Optional[Path],
    selected_attempt: Optional[Dict[str, Any]],
    matched_domain_transfer: Optional[Dict[str, Any]],
    *,
    stars_required: bool = True,
) -> Dict[str, Any]:
    """Bind the selected Stage 7 candidate to its theoretical display reference."""

    report: Dict[str, Any] = {
        "schema": STAGE7_REFERENCE_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "issues": [],
    }
    try:
        lineage = load_lineage(process_dir)
        if not bool(lineage.get("accepted", False)):
            raise ValueError("Stage3 spatial background lineage is unavailable")
        if not isinstance(selected_attempt, dict):
            raise ValueError("formal Stage7 selected attempt is unavailable")
        spatial = dict(selected_attempt.get("spatial_chroma_quality") or {})
        display_reference = dict(spatial.get("display_reference") or {})
        if (
            not bool(spatial.get("accepted", False))
            or display_reference.get("schema") != STAGE7_REFERENCE_SCHEMA
            or display_reference.get("status") != "ready"
        ):
            raise ValueError("Stage7 theoretical display reference is unavailable")
        artifact = str(selected_attempt.get("file") or "")
        if not artifact:
            raise ValueError("Stage7 selected candidate artifact is unavailable")
        transfer = dict(matched_domain_transfer or {})
        if transfer.get("status") != "active":
            if stars_required:
                raise ValueError("Stage7 matched-domain transfer is unavailable")
            transfer = {
                "schema": "starun.stage7-direct-presentation-reference.v1",
                "status": "not_required",
                "method": "frozen_stage7_presentation_reference",
                "reason_code": "stars_not_required",
                "selected_candidate_id": selected_attempt.get("name"),
            }
        support_sha256 = str(display_reference.get("support_sha256") or "")
        if support_sha256 != str(lineage.get("support_sha256") or ""):
            raise ValueError("Stage7/Stage3 support SHA mismatch")
        transform_identity = dict(
            display_reference.get("transform_identity") or {}
        )
        transform_digest = _json_sha256(
            {
                "transform_identity": transform_identity,
                "matched_domain_transfer": transfer,
            }
        )
        report.update(
            status="accepted",
            accepted=True,
            support_sha256=support_sha256,
            stage3_output_sha256=lineage.get("stage3_output_sha256"),
            stage7_candidate={
                "name": selected_attempt.get("name"),
                "artifact": artifact,
                "pixel_sha256": display_reference.get(
                    "candidate_pixel_sha256"
                ),
            },
            expected_pixel_sha256=display_reference.get(
                "expected_pixel_sha256"
            ),
            expected_metrics=dict(
                display_reference.get("expected_metrics") or {}
            ),
            transform_identity=transform_identity,
            matched_domain_transfer=transfer,
            transform_digest_sha256=transform_digest,
        )
        return report
    except (TypeError, ValueError) as error:
        report["issues"] = [str(error)]
        return report


def load_stage7_display_reference(process_dir: Optional[Path]) -> Dict[str, Any]:
    """Load and verify the Stage 7 display-domain spatial reference."""

    report: Dict[str, Any] = {
        "schema": STAGE7_REFERENCE_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "reason_code": None,
        "issues": [],
    }
    if process_dir is None:
        report["issues"] = [STAGE7_REFERENCE_READ_FAILED]
        return report
    root = Path(process_dir)
    reference_path = root / STAGE7_REFERENCE_NAME
    if not reference_path.is_file():
        reason_code = STAGE7_REFERENCE_MISSING
        quality_path = root / STAGE7_QUALITY_NAME
        if quality_path.is_file():
            try:
                quality = json.loads(quality_path.read_text(encoding="utf-8"))
                if (
                    quality.get("formal_accepted") is False
                    and str(quality.get("delivery_class") or "")
                    == "review_only"
                    and str(quality.get("status") or "")
                    in {"review_only", "failed"}
                ):
                    reason_code = STAGE7_REFERENCE_REVIEW_ONLY
            except (OSError, TypeError, json.JSONDecodeError):
                reason_code = STAGE7_REFERENCE_MISSING
        report["reason_code"] = reason_code
        report["issues"] = [reason_code]
        return report
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
        if payload.get("schema") != STAGE7_REFERENCE_SCHEMA:
            raise ValueError("Stage7 spatial reference schema mismatch")
        if payload.get("status") != "accepted" or payload.get("accepted") is not True:
            raise ValueError("Stage7 spatial reference is not accepted")
        lineage = load_lineage(root)
        if not bool(lineage.get("accepted", False)):
            raise ValueError("Stage3 spatial background lineage is unavailable")
        if str(payload.get("support_sha256") or "") != str(
            lineage.get("support_sha256") or ""
        ):
            raise ValueError("Stage7 spatial reference support SHA mismatch")
        candidate = dict(payload.get("stage7_candidate") or {})
        artifact_name = str(candidate.get("artifact") or "")
        expected_pixel_sha = str(candidate.get("pixel_sha256") or "")
        artifact = root / artifact_name
        if not artifact_name or not artifact.is_file() or not expected_pixel_sha:
            raise ValueError("Stage7 selected candidate lineage is unavailable")
        with fits.open(artifact, memmap=False, do_not_scale_image_data=False) as hdul:
            candidate_pixels = np.asarray(hdul[0].data)
        if _array_sha256(candidate_pixels) != expected_pixel_sha:
            raise ValueError("Stage7 selected candidate pixel SHA mismatch")
        transform_identity = dict(payload.get("transform_identity") or {})
        transfer = dict(payload.get("matched_domain_transfer") or {})
        expected_digest = _json_sha256(
            {
                "transform_identity": transform_identity,
                "matched_domain_transfer": transfer,
            }
        )
        if expected_digest != str(payload.get("transform_digest_sha256") or ""):
            raise ValueError("Stage7 spatial reference transform digest mismatch")
        metrics = dict(payload.get("expected_metrics") or {})
        if any(name not in metrics for name in ("luma", "R-G", "B-G")):
            raise ValueError("Stage7 display-domain reference metrics are incomplete")
        payload = dict(payload)
        payload["status"] = "accepted"
        payload["accepted"] = True
        payload["reason_code"] = None
        return payload
    except (OSError, TypeError, json.JSONDecodeError):
        report["reason_code"] = STAGE7_REFERENCE_READ_FAILED
        report["issues"] = [STAGE7_REFERENCE_READ_FAILED]
        return report
    except ValueError as error:
        report["reason_code"] = STAGE7_REFERENCE_INVALID
        report["issues"] = [str(error)]
        return report


def assess_final_spatial_background(
    process_dir: Optional[Path],
    final_image: Any,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "status": "unavailable",
        "accepted": False,
        "reason_code": "final_spatial_background_gradient_unresolved",
        "issues": ["final_spatial_background_lineage_unverified"],
        "components": {},
    }
    lineage = load_lineage(process_dir)
    if not bool(lineage.get("accepted", False)):
        report["lineage"] = {
            key: value
            for key, value in lineage.items()
            if key != "support_mask"
        }
        return report
    try:
        points = [
            tuple(item)
            for item in [
                *(lineage.get("fit_points") or []),
                *(lineage.get("validation_points") or []),
            ]
        ]
        metrics = measure_spatial_background_planes(
            final_image,
            lineage["support_mask"],
            points,
            patch_radius=int(lineage.get("patch_radius") or 12),
        )
        stage7_reference = load_stage7_display_reference(process_dir)
        if not bool(stage7_reference.get("accepted", False)):
            report["issues"] = list(stage7_reference.get("issues") or []) or [
                "Stage7 display-domain spatial reference is unavailable"
            ]
            report["stage7_display_reference"] = stage7_reference
            return report
        reference = dict(stage7_reference.get("expected_metrics") or {})
        issues = []
        for name in ("luma", "R-G", "B-G"):
            component = metrics[name]
            reference_component = dict(reference.get(name) or {})
            reference_span = float(
                reference_component.get("slope_span")
                or NORMALIZED_MATERIAL_FLOOR
            )
            growth = float(component["slope_span"]) / max(
                reference_span,
                NORMALIZED_MATERIAL_FLOOR,
            )
            significant = bool(
                component["material"]
                and float(component["slope_significance_sigma"])
                >= SIGNIFICANCE_SIGMA
            )
            reference_significant = bool(
                reference_component.get("material")
                and float(
                    reference_component.get("slope_significance_sigma") or 0.0
                )
                >= SIGNIFICANCE_SIGMA
            )
            newly_significant = bool(significant and not reference_significant)
            growth_exceeded = bool(
                component["material"] and growth > GROWTH_MAX
            )
            if newly_significant or growth_exceeded:
                issues.append(f"final_spatial_background_gradient_unresolved:{name}")
            report["components"][name] = {
                "metrics": component,
                "display_domain_reference": reference_component or None,
                "stage3_linear_reference": dict(
                    (lineage.get("reference_metrics") or {}).get(name) or {}
                ) or None,
                "slope_growth": growth,
                "significant": significant,
                "reference_significant": reference_significant,
                "newly_significant": newly_significant,
                "growth_exceeded": growth_exceeded,
                "accepted": not (newly_significant or growth_exceeded),
            }
        report.update(
            status="rejected" if issues else "ok",
            accepted=not issues,
            issues=issues,
            support_sha256=lineage.get("support_sha256"),
            stage7_display_reference={
                "schema": stage7_reference.get("schema"),
                "status": stage7_reference.get("status"),
                "support_sha256": stage7_reference.get("support_sha256"),
                "stage7_candidate": stage7_reference.get("stage7_candidate"),
                "transform_digest_sha256": stage7_reference.get(
                    "transform_digest_sha256"
                ),
            },
        )
        return report
    except (IndexError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        report["issues"] = [str(error)]
        return report


__all__ = [
    "FINAL_SCHEMA",
    "LEGACY_LINEAGE_SCHEMAS",
    "LINEAGE_CHAIN_SCHEMA",
    "LINEAGE_SCHEMA",
    "STAGE7_SCHEMA",
    "STAGE7_REFERENCE_NAME",
    "STAGE7_REFERENCE_SCHEMA",
    "assess_final_spatial_background",
    "assess_stage7_spatial_chroma",
    "build_stage7_display_reference",
    "load_lineage",
    "load_stage7_display_reference",
    "lineage_chain_digest",
    "measure_spatial_background_planes",
    "seal_lineage",
]
