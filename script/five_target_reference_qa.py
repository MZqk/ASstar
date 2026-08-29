#!/usr/bin/env python3
"""Independent, display-domain QA against external reference images.

This tool is deliberately outside the production pipeline.  Reference pixels
are used only for registration, measurement, and report/contact-sheet output;
they are never returned to a Starun run or used to synthesize production
pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    maximum_filter,
)


MANIFEST_SCHEMA = "starun.external-reference-qa-manifest.v1"
REPORT_SCHEMA = "starun.external-reference-qa.v1"
ANALYSIS_MAX_SIDE = 1600
MIN_REGISTRATION_MATCHES = 8
MIN_REGISTRATION_INLIERS = 8
MIN_REGISTRATION_INLIER_RATIO = 0.30
MAX_REGISTRATION_REPROJECTION_P95 = 5.0
MIN_REGISTRATION_OVERLAP = 0.45
MIN_MATCHED_STARS = 12

LOCKED_LIMITS = {
    "subject_saturation_gap_shrinkage_min": 0.50,
    "subject_saturation_ratio_min": 0.50,
    "subject_saturation_ratio_max": 1.25,
    "star_fwhm_ratio_min": 0.85,
    "star_fwhm_ratio_max": 1.15,
    "background_hf_gap_shrinkage_min": 0.50,
    "background_hf_ratio_max": 2.00,
    "background_lf_ratio_max": 1.50,
    "detail_gap_shrinkage_min": 0.30,
    "contrast_gap_shrinkage_min": 0.30,
    "structure_correlation_drop_max": 0.02,
    "dimension_gap_regression_max": 0.10,
    "dimensions_improved_min": 4,
}

EXPECTED_TARGET_PROFILES = {
    "M31": "large_galaxy",
    "NGC6888": "emission_nebula",
    "NGC7000": "emission_nebula",
    "NGC6910": "star_preserve",
    "M8": "bright_emission_reflection_nebula",
}

VISUAL_REVIEW_CHECKS = (
    "no_new_seam",
    "no_new_halo_or_dark_rim",
    "no_black_level_crush",
    "no_core_overexposure",
    "no_fluorescent_or_implausible_color",
    "full_resolution_reviewed",
)


class QAError(RuntimeError):
    """A fail-closed external-reference QA error."""


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise QAError(
            "OpenCV with SIFT is required for external-reference registration; "
            "left-edge cropping or unregistered resize fallback is prohibited"
        ) from exc
    if not hasattr(cv2, "SIFT_create"):
        raise QAError("OpenCV is present but SIFT_create is unavailable")
    return cv2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QAError(f"manifest field {field!r} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise QAError(f"manifest {field} is not a readable file: {path}")
    return path


def _resolve_directory(value: Any, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QAError(f"manifest field {field!r} must be a non-empty directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_dir():
        raise QAError(f"manifest {field} is not a readable directory: {path}")
    return path


def _read_json_object(path: Path, purpose: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QAError(f"cannot read {purpose} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QAError(f"{purpose} must be a JSON object: {path}")
    return payload


def _verify_run_artifact(
    run_root: Path,
    artifact: Path,
    *,
    require_formal_delivery: bool,
) -> dict[str, Any]:
    try:
        relative_artifact = artifact.relative_to(run_root)
    except ValueError as exc:
        raise QAError(f"artifact is outside declared run root: {artifact}") from exc
    if len(relative_artifact.parts) != 1:
        raise QAError(
            "QA display artifact must be a root-level run output recorded by "
            "pipeline-result.json"
        )
    result_path = run_root / "pipeline-result.json"
    result = _read_json_object(result_path, "pipeline result")
    if result.get("schema") != "starun.pipeline-result.v2":
        raise QAError(f"unsupported pipeline result schema in {result_path}")
    outputs = result.get("outputs")
    record = outputs.get(artifact.name) if isinstance(outputs, dict) else None
    if not isinstance(record, dict):
        raise QAError(f"artifact is absent from pipeline-result outputs: {artifact.name}")
    expected_sha = str(record.get("sha256") or "").lower()
    actual_sha = _sha256(artifact)
    if len(expected_sha) != 64 or expected_sha != actual_sha:
        raise QAError(f"pipeline-result artifact SHA mismatch: {artifact.name}")

    gates = result.get("delivery_gates")
    formal_outputs: list[Any] = []
    if isinstance(gates, dict):
        artifact_gate = gates.get("artifacts")
        if isinstance(artifact_gate, dict) and isinstance(
            artifact_gate.get("formal_outputs"), list
        ):
            formal_outputs = artifact_gate["formal_outputs"]
    if require_formal_delivery:
        accepted = bool(
            result.get("status") in {"success", "partial_success"}
            and result.get("delivery_eligible") is True
            and result.get("review_required") is False
            and not list(result.get("review_requirements") or ())
            and isinstance(gates, dict)
            and gates.get("schema") == "starun.final-delivery-gates.v1"
            and gates.get("legacy_delivery_contract") is False
            and gates.get("formal_delivery_accepted") is True
            and all(
                isinstance(gates.get(name), dict)
                and gates[name].get("accepted") is True
                for name in ("scientific", "presentation", "artifacts", "review")
            )
            and artifact.name in formal_outputs
        )
        if not accepted:
            raise QAError(
                "optimized artifact is not an identity-bound, dual-gate formal "
                f"delivery: {artifact.name}"
            )
    return {
        "run_root": str(run_root),
        "pipeline_result": str(result_path),
        "pipeline_result_sha256": _sha256(result_path),
        "artifact": artifact.name,
        "artifact_sha256": actual_sha,
        "formal_delivery_required": require_formal_delivery,
        "formal_delivery_verified": bool(require_formal_delivery),
    }


def _visual_review_payload(raw: Any, optimized_sha256: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise QAError("visual_artifact_review must be an object")
    bound_sha = str(raw.get("optimized_sha256") or "").strip().lower()
    if bound_sha != optimized_sha256:
        raise QAError("visual review optimized SHA does not match the artifact")
    checks = {name: raw.get(name) is True for name in VISUAL_REVIEW_CHECKS}
    return {
        "optimized_sha256": bound_sha,
        "checks": checks,
        "passed": all(checks.values()),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QAError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QAError("manifest must be a JSON object")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise QAError(
            f"manifest schema must be {MANIFEST_SCHEMA!r}, got "
            f"{payload.get('schema')!r}"
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise QAError("manifest.entries must be a non-empty list")
    entries: list[dict[str, Any]] = []
    targets: set[str] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise QAError(f"manifest entry {index} must be an object")
        target = str(raw.get("target") or "").strip()
        if not target:
            raise QAError(f"manifest entry {index} has no target")
        if target.casefold() in targets:
            raise QAError(f"duplicate manifest target: {target}")
        targets.add(target.casefold())
        expected_profile = EXPECTED_TARGET_PROFILES.get(target)
        if expected_profile is None:
            raise QAError(f"unexpected five-target QA target: {target}")
        profile = str(raw.get("profile") or "").strip().lower()
        if profile != expected_profile:
            raise QAError(
                f"target {target} requires profile {expected_profile!r}, got "
                f"{profile!r}"
            )
        baseline = _resolve_path(
            raw.get("baseline"), manifest_path.parent, "baseline"
        )
        optimized = _resolve_path(
            raw.get("optimized"), manifest_path.parent, "optimized"
        )
        reference = _resolve_path(
            raw.get("reference"), manifest_path.parent, "reference"
        )
        role_paths = (baseline, optimized, reference)
        if len(set(role_paths)) != len(role_paths):
            raise QAError(f"target {target} reuses the same path for multiple roles")
        role_sha = {role: _sha256(role) for role in role_paths}
        if role_sha[optimized] == role_sha[reference]:
            raise QAError(f"target {target} optimized pixels equal the reference")
        if role_sha[baseline] == role_sha[reference]:
            raise QAError(f"target {target} baseline pixels equal the reference")
        baseline_run_root = _resolve_directory(
            raw.get("baseline_run_root"),
            manifest_path.parent,
            "baseline_run_root",
        )
        optimized_run_root = _resolve_directory(
            raw.get("optimized_run_root"),
            manifest_path.parent,
            "optimized_run_root",
        )
        baseline_provenance = _verify_run_artifact(
            baseline_run_root,
            baseline,
            require_formal_delivery=False,
        )
        optimized_provenance = _verify_run_artifact(
            optimized_run_root,
            optimized,
            require_formal_delivery=True,
        )
        visual_review = _visual_review_payload(
            raw.get("visual_artifact_review"),
            role_sha[optimized],
        )
        entry = dict(raw)
        entry.update(
            target=target,
            baseline=baseline,
            optimized=optimized,
            reference=reference,
            profile=profile,
            baseline_run_root=baseline_run_root,
            optimized_run_root=optimized_run_root,
            baseline_provenance=baseline_provenance,
            optimized_provenance=optimized_provenance,
            input_sha256={
                "baseline": role_sha[baseline],
                "optimized": role_sha[optimized],
                "reference": role_sha[reference],
            },
            visual_artifact_review=visual_review,
            visual_artifact_review_passed=visual_review["passed"],
        )
        entries.append(entry)
    expected_targets = set(EXPECTED_TARGET_PROFILES)
    observed_targets = {entry["target"] for entry in entries}
    if observed_targets != expected_targets:
        missing = sorted(expected_targets - observed_targets)
        unexpected = sorted(observed_targets - expected_targets)
        raise QAError(
            "five-target coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": manifest_path,
        "payload": payload,
        "entries": entries,
    }


def _read_rgb(path: Path, cv2: Any) -> tuple[np.ndarray, list[int]]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise QAError(f"OpenCV cannot decode image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise QAError(f"unsupported image shape {image.shape}: {path}")
    original_shape = [int(value) for value in image.shape]
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
    else:
        scale = float(np.nanmax(image))
        if scale <= 1.0:
            scale = 1.0
    values = np.asarray(image, dtype=np.float32) / max(scale, 1e-12)
    if values.shape[2] == 4:
        alpha = np.clip(values[:, :, 3:4], 0.0, 1.0)
        values = values[:, :, :3] * alpha
    else:
        values = values[:, :, :3]
    values = values[:, :, ::-1]
    if not np.all(np.isfinite(values)):
        raise QAError(f"image contains non-finite pixels: {path}")
    return np.clip(values, 0.0, 1.0), original_shape


def _resize_max_side(image: np.ndarray, cv2: Any) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, ANALYSIS_MAX_SIDE / max(height, width))
    if scale >= 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return np.asarray(resized, dtype=np.float32), float(scale)


def _luminance(image: np.ndarray) -> np.ndarray:
    return np.asarray(
        image[:, :, 0] * 0.2126
        + image[:, :, 1] * 0.7152
        + image[:, :, 2] * 0.0722,
        dtype=np.float32,
    )


def _registration_gray(image: np.ndarray, cv2: Any) -> np.ndarray:
    gray = np.clip(_luminance(image) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(gray)


def _register_to_reference(
    source: np.ndarray,
    reference: np.ndarray,
    cv2: Any,
    *,
    source_shape: Sequence[int],
    reference_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_small, source_scale = _resize_max_side(source, cv2)
    reference_small, reference_scale = _resize_max_side(reference, cv2)
    source_gray = _registration_gray(source_small, cv2)
    reference_gray = _registration_gray(reference_small, cv2)
    sift = cv2.SIFT_create(
        nfeatures=16000,
        contrastThreshold=0.008,
        edgeThreshold=12,
        sigma=1.2,
    )
    source_keypoints, source_desc = sift.detectAndCompute(source_gray, None)
    reference_keypoints, reference_desc = sift.detectAndCompute(
        reference_gray, None
    )
    if source_desc is None or reference_desc is None:
        raise QAError("SIFT descriptors unavailable")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward_pairs = matcher.knnMatch(source_desc, reference_desc, k=2)
    reverse_pairs = matcher.knnMatch(reference_desc, source_desc, k=2)
    forward_good = [
        match
        for match, other in forward_pairs
        if match.distance < 0.70 * other.distance
    ]
    reverse_good = [
        match
        for match, other in reverse_pairs
        if match.distance < 0.70 * other.distance
    ]
    reverse_lookup = {
        int(match.queryIdx): int(match.trainIdx) for match in reverse_good
    }
    mutual_good = [
        match
        for match in forward_good
        if reverse_lookup.get(int(match.trainIdx)) == int(match.queryIdx)
    ]
    # Dense stellar fields contain many locally similar descriptors.  Mutual
    # Lowe-ratio matches reject one-way aliases before RANSAC, increasing the
    # evidence quality without weakening any inlier, reprojection, or overlap
    # gate.  Sparse fields retain the established forward matcher when mutual
    # support is insufficient to estimate a transform.
    if len(mutual_good) >= MIN_REGISTRATION_MATCHES:
        good = mutual_good
        match_policy = "mutual_bidirectional_ratio"
    else:
        good = forward_good
        match_policy = "forward_ratio_sparse_fallback"
    if len(good) < MIN_REGISTRATION_MATCHES:
        raise QAError(
            f"registration has {len(good)} ratio-test matches; "
            f"need at least {MIN_REGISTRATION_MATCHES}"
        )
    source_points = np.float32(
        [source_keypoints[item.queryIdx].pt for item in good]
    )
    reference_points = np.float32(
        [reference_keypoints[item.trainIdx].pt for item in good]
    )
    transform, raw_inliers = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        3.0,
        maxIters=10000,
        confidence=0.999,
    )
    if transform is None or raw_inliers is None:
        raise QAError("SIFT/RANSAC could not estimate a homography")
    inliers = raw_inliers.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / max(len(good), 1)
    if inlier_count < MIN_REGISTRATION_INLIERS:
        raise QAError(
            f"registration has {inlier_count} inliers; "
            f"need at least {MIN_REGISTRATION_INLIERS}"
        )
    if inlier_ratio < MIN_REGISTRATION_INLIER_RATIO:
        raise QAError(
            f"registration inlier ratio {inlier_ratio:.3f}<"
            f"{MIN_REGISTRATION_INLIER_RATIO:.3f}"
        )
    projected = cv2.perspectiveTransform(
        source_points[inliers].reshape(-1, 1, 2), transform
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - reference_points[inliers], axis=1)
    error_p95 = float(np.quantile(errors, 0.95))
    if error_p95 > MAX_REGISTRATION_REPROJECTION_P95:
        raise QAError(
            f"registration reprojection p95 {error_p95:.3f}>"
            f"{MAX_REGISTRATION_REPROJECTION_P95:.3f} px"
        )
    height, width = reference_small.shape[:2]
    registered = cv2.warpPerspective(
        source_small,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpPerspective(
        np.ones(source_small.shape[:2], dtype=np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    valid = binary_erosion(valid, iterations=8)
    overlap = float(np.mean(valid))
    if overlap < MIN_REGISTRATION_OVERLAP:
        raise QAError(
            f"registration overlap {overlap:.3f}<"
            f"{MIN_REGISTRATION_OVERLAP:.3f}"
        )
    report = {
        "method": "opencv_sift_homography_ransac",
        "match_policy": match_policy,
        "fallback_used": match_policy != "mutual_bidirectional_ratio",
        "source_original_shape": [int(value) for value in source_shape],
        "reference_original_shape": [int(value) for value in reference_shape],
        "source_analysis_shape": [int(value) for value in source_small.shape],
        "reference_analysis_shape": [int(value) for value in reference_small.shape],
        "source_analysis_scale": source_scale,
        "reference_analysis_scale": reference_scale,
        "source_keypoints": len(source_keypoints),
        "reference_keypoints": len(reference_keypoints),
        "forward_ratio_test_matches": len(forward_good),
        "reverse_ratio_test_matches": len(reverse_good),
        "mutual_ratio_test_matches": len(mutual_good),
        "ratio_test_matches": len(good),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "reprojection_error_px": {
            "p50": float(np.median(errors)),
            "p95": error_p95,
        },
        "overlap_ratio": overlap,
        "source_analysis_to_reference_analysis_homography": [
            [float(value) for value in row] for row in transform
        ],
    }
    return np.clip(registered, 0.0, 1.0), valid, report


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise QAError("robust statistic has no finite samples")
    center = float(np.median(finite))
    return 1.4826 * float(np.median(np.abs(finite - center)))


def _saturation(image: np.ndarray) -> np.ndarray:
    smooth = np.stack(
        [gaussian_filter(image[:, :, channel], 1.2) for channel in range(3)],
        axis=2,
    )
    maximum = np.max(smooth, axis=2)
    minimum = np.min(smooth, axis=2)
    return (maximum - minimum) / np.maximum(maximum, 1e-6)


def _reference_masks(
    reference: np.ndarray,
    valid: np.ndarray,
    profile: str,
) -> dict[str, np.ndarray]:
    gray = _luminance(reference)
    low = gaussian_filter(gray, 10.0)
    high = gaussian_filter(gray, 0.7) - gaussian_filter(gray, 4.0)
    pool = high[valid]
    high_center = float(np.median(pool))
    high_sigma = max(_robust_sigma(pool), 1e-5)
    star_threshold = max(
        float(np.quantile(pool, 0.993)), high_center + 5.0 * high_sigma, 0.004
    )
    star_peaks = (
        (high == maximum_filter(high, size=7))
        & (high > star_threshold)
        & valid
        & (gray < 0.995)
    )
    stars = binary_dilation(star_peaks, iterations=5)
    grad_y, grad_x = np.gradient(low)
    gradient = np.hypot(grad_x, grad_y)
    nonstar = valid & ~binary_dilation(stars, iterations=2)
    if int(np.count_nonzero(nonstar)) < 1024:
        raise QAError("reference leaves too little non-star support")
    background = (
        nonstar
        & (low <= np.quantile(low[nonstar], 0.42))
        & (gradient <= np.quantile(gradient[nonstar], 0.62))
    )
    saturation = gaussian_filter(_saturation(reference), 2.0)
    if profile in {"star_preserve", "open_cluster", "star_cluster", "cluster"}:
        subject = binary_dilation(star_peaks, iterations=7) & valid
    else:
        low_signal = low - float(np.median(low[background]))
        score = low_signal + 0.20 * saturation
        subject = (
            nonstar
            & (score >= np.quantile(score[nonstar], 0.60))
            & (low >= np.quantile(low[nonstar], 0.42))
        )
    background &= ~binary_dilation(subject, iterations=3)
    minimum = max(256, int(valid.size * 0.001))
    if int(np.count_nonzero(background)) < minimum:
        raise QAError("reference background mask has insufficient support")
    if int(np.count_nonzero(subject)) < minimum:
        raise QAError("reference subject mask has insufficient support")
    return {
        "valid": valid,
        "background": background,
        "subject": subject,
        "stars": stars,
        "star_peaks": star_peaks,
    }


def _fit_plane_residual(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if yy.size < 256:
        raise QAError("low-frequency plane has insufficient support")
    stride = max(1, yy.size // 100000)
    yy = yy[::stride]
    xx = xx[::stride]
    values = image[yy, xx].astype(np.float64)
    design = np.column_stack(
        (
            np.ones(xx.size),
            xx / max(image.shape[1] - 1, 1),
            yy / max(image.shape[0] - 1, 1),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    all_y, all_x = np.indices(image.shape)
    plane = (
        coefficients[0]
        + coefficients[1] * all_x / max(image.shape[1] - 1, 1)
        + coefficients[2] * all_y / max(image.shape[0] - 1, 1)
    )
    return np.asarray(image - plane, dtype=np.float32)


def _image_metrics(
    image: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    valid = masks["valid"]
    background = masks["background"]
    subject = masks["subject"]
    gray = _luminance(image)
    saturation = _saturation(image)
    high_frequency = gray - gaussian_filter(gray, 1.2)
    low_frequency = gaussian_filter(gray, 12.0)
    low_detrended = _fit_plane_residual(low_frequency, background)
    low_residual = low_frequency - float(np.median(low_frequency[background]))
    detail_band = gaussian_filter(gray, 1.0) - gaussian_filter(gray, 4.0)
    contrast = float(np.median(gray[subject]) - np.median(gray[background]))
    if contrast <= 1e-8:
        raise QAError("subject/background contrast is not measurable")
    return {
        "subject_saturation_p50": float(np.median(saturation[subject])),
        "subject_saturation_p95": float(np.quantile(saturation[subject], 0.95)),
        "background_high_frequency_sigma": _robust_sigma(
            high_frequency[background]
        ),
        "background_low_frequency_residual_sigma": _robust_sigma(
            low_residual[background]
        ),
        "background_low_frequency_detrended_sigma": _robust_sigma(
            low_detrended[background]
        ),
        "detail_band_robust_sigma": _robust_sigma(detail_band[subject]),
        "subject_background_contrast": contrast,
        "black_pixel_ratio": float(np.mean(gray[valid] <= 1.0 / 255.0)),
        "highlight_pixel_ratio": float(
            np.mean(np.max(image, axis=2)[valid] >= 0.995)
        ),
        "extreme_saturation_ratio": float(
            np.mean((saturation[valid] >= 0.95) & (gray[valid] >= 0.08))
        ),
        "_detail_band": detail_band,
    }


def _structure_correlation(
    candidate_band: np.ndarray,
    reference_band: np.ndarray,
    subject: np.ndarray,
) -> float:
    candidate = np.asarray(candidate_band[subject], dtype=np.float64)
    reference = np.asarray(reference_band[subject], dtype=np.float64)
    candidate -= float(np.mean(candidate))
    reference -= float(np.mean(reference))
    denominator = float(np.linalg.norm(candidate) * np.linalg.norm(reference))
    if denominator <= 1e-12:
        raise QAError("structure correlation is not measurable")
    return float(np.dot(candidate, reference) / denominator)


def _measure_star(gray: np.ndarray, y: int, x: int) -> dict[str, float] | None:
    radius = 8
    if (
        y - radius < 0
        or x - radius < 0
        or y + radius >= gray.shape[0]
        or x + radius >= gray.shape[1]
    ):
        return None
    search = gray[y - 2 : y + 3, x - 2 : x + 3]
    dy, dx = np.unravel_index(int(np.argmax(search)), search.shape)
    center_y = y + int(dy) - 2
    center_x = x + int(dx) - 2
    if (
        center_y - radius < 0
        or center_x - radius < 0
        or center_y + radius >= gray.shape[0]
        or center_x + radius >= gray.shape[1]
    ):
        return None
    patch = gray[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ].astype(np.float64)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    radial = np.hypot(xx, yy)
    background = float(np.median(patch[(radial >= 6.0) & (radial <= 8.0)]))
    weights = np.clip(patch - background, 0.0, None)
    weights[radial > 5.5] = 0.0
    flux = float(np.sum(weights))
    if flux <= 1e-8 or float(np.max(patch)) >= 0.995:
        return None
    centroid_x = float(np.sum(weights * xx) / flux)
    centroid_y = float(np.sum(weights * yy) / flux)
    if math.hypot(centroid_x, centroid_y) > 2.0:
        return None
    variance_x = float(np.sum(weights * (xx - centroid_x) ** 2) / flux)
    variance_y = float(np.sum(weights * (yy - centroid_y) ** 2) / flux)
    fwhm = 2.35482 * math.sqrt(max((variance_x + variance_y) / 2.0, 0.0))
    if not 0.7 <= fwhm <= 12.0:
        return None
    return {"fwhm_px": fwhm, "peak": float(np.max(patch) - background)}


def _matched_star_metrics(
    baseline: np.ndarray,
    optimized: np.ndarray,
    reference: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    baseline_gray = _luminance(baseline)
    optimized_gray = _luminance(optimized)
    reference_gray = _luminance(reference)
    high = gaussian_filter(reference_gray, 0.7) - gaussian_filter(
        reference_gray, 4.0
    )
    pool = high[masks["valid"]]
    threshold = max(
        float(np.quantile(pool, 0.990)),
        float(np.median(pool)) + 5.0 * max(_robust_sigma(pool), 1e-5),
        0.004,
    )
    peaks = (
        (high == maximum_filter(high, size=9))
        & (high > threshold)
        & masks["valid"]
        & (reference_gray < 0.985)
    )
    ys, xs = np.nonzero(peaks)
    scores = high[ys, xs]
    order = np.argsort(scores)[::-1][:3000]
    triples: list[tuple[float, float, float]] = []
    for index in order:
        y = int(ys[index])
        x = int(xs[index])
        reference_star = _measure_star(reference_gray, y, x)
        baseline_star = _measure_star(baseline_gray, y, x)
        optimized_star = _measure_star(optimized_gray, y, x)
        if not reference_star or not baseline_star or not optimized_star:
            continue
        triples.append(
            (
                baseline_star["fwhm_px"],
                optimized_star["fwhm_px"],
                reference_star["fwhm_px"],
            )
        )
        if len(triples) >= 800:
            break
    if len(triples) < MIN_MATCHED_STARS:
        raise QAError(
            f"only {len(triples)} common unsaturated stars are measurable; "
            f"need at least {MIN_MATCHED_STARS}"
        )
    values = np.asarray(triples, dtype=np.float64)
    baseline_ratios = values[:, 0] / values[:, 2]
    optimized_ratios = values[:, 1] / values[:, 2]
    return {
        "common_star_count": len(triples),
        "reference_fwhm_px_p50": float(np.median(values[:, 2])),
        "baseline_fwhm_px_p50": float(np.median(values[:, 0])),
        "optimized_fwhm_px_p50": float(np.median(values[:, 1])),
        "baseline_reference_ratio_p50": float(np.median(baseline_ratios)),
        "optimized_reference_ratio_p50": float(np.median(optimized_ratios)),
        "baseline_reference_ratio_p10": float(np.quantile(baseline_ratios, 0.10)),
        "baseline_reference_ratio_p90": float(np.quantile(baseline_ratios, 0.90)),
        "optimized_reference_ratio_p10": float(np.quantile(optimized_ratios, 0.10)),
        "optimized_reference_ratio_p90": float(np.quantile(optimized_ratios, 0.90)),
    }


def _ratio(value: float, reference: float, name: str) -> float:
    if not math.isfinite(value) or not math.isfinite(reference) or reference <= 1e-10:
        raise QAError(f"{name} reference denominator is not measurable")
    return float(value / reference)


def _gap_shrinkage(baseline_gap: float, optimized_gap: float) -> float | None:
    if baseline_gap <= 1e-8:
        return 0.0 if optimized_gap <= 0.01 else None
    return float((baseline_gap - optimized_gap) / baseline_gap)


def _nonregressed(baseline_gap: float, optimized_gap: float) -> bool:
    relative_limit = baseline_gap * (
        1.0 + LOCKED_LIMITS["dimension_gap_regression_max"]
    )
    return bool(optimized_gap <= relative_limit + 1e-12)


def _comparison(
    baseline_value: float,
    optimized_value: float,
    reference_value: float,
    *,
    name: str,
) -> dict[str, Any]:
    baseline_ratio = _ratio(baseline_value, reference_value, name)
    optimized_ratio = _ratio(optimized_value, reference_value, name)
    baseline_gap = abs(baseline_ratio - 1.0)
    optimized_gap = abs(optimized_ratio - 1.0)
    return {
        "baseline": baseline_value,
        "optimized": optimized_value,
        "reference": reference_value,
        "baseline_reference_ratio": baseline_ratio,
        "optimized_reference_ratio": optimized_ratio,
        "baseline_gap": baseline_gap,
        "optimized_gap": optimized_gap,
        "gap_shrinkage": _gap_shrinkage(baseline_gap, optimized_gap),
        "not_degraded_over_10_percent": _nonregressed(
            baseline_gap, optimized_gap
        ),
    }


def _artifact_guard(
    baseline_metrics: Mapping[str, Any],
    optimized_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for key, absolute_tolerance in (
        ("black_pixel_ratio", 0.005),
        ("highlight_pixel_ratio", 0.002),
        ("extreme_saturation_ratio", 0.010),
    ):
        baseline = float(baseline_metrics[key])
        optimized = float(optimized_metrics[key])
        limit = max(baseline * 1.10, baseline + absolute_tolerance)
        gates[key] = {
            "baseline": baseline,
            "optimized": optimized,
            "limit": limit,
            "accepted": optimized <= limit + 1e-12,
        }
    accepted = all(gate["accepted"] for gate in gates.values())
    return {
        "accepted": accepted,
        "automated_gates": gates,
        "human_only_checks": [
            "new seam or tiling boundary",
            "new stellar halo or dark rim",
            "core overexposure morphology",
            "fluorescent or implausible local colour",
        ],
    }


def _acceptance_status(
    quantitative_accepted: bool,
    visual_artifact_review_passed: bool,
) -> tuple[bool, str]:
    accepted = bool(
        quantitative_accepted and visual_artifact_review_passed
    )
    if accepted:
        return True, "accepted"
    if quantitative_accepted:
        return False, "review_required"
    return False, "rejected"


def _dimension_reports(
    baseline: Mapping[str, Any],
    optimized: Mapping[str, Any],
    reference: Mapping[str, Any],
    stars: Mapping[str, Any],
    baseline_structure: float,
    optimized_structure: float,
) -> dict[str, Any]:
    color = _comparison(
        float(baseline["subject_saturation_p50"]),
        float(optimized["subject_saturation_p50"]),
        float(reference["subject_saturation_p50"]),
        name="subject_saturation_p50",
    )
    color["absolute_target_passed"] = bool(
        LOCKED_LIMITS["subject_saturation_ratio_min"]
        <= color["optimized_reference_ratio"]
        <= LOCKED_LIMITS["subject_saturation_ratio_max"]
    )
    color["improved"] = bool(
        color["absolute_target_passed"]
        and color["gap_shrinkage"] is not None
        and color["gap_shrinkage"]
        >= LOCKED_LIMITS["subject_saturation_gap_shrinkage_min"]
    )

    baseline_star_ratio = float(stars["baseline_reference_ratio_p50"])
    optimized_star_ratio = float(stars["optimized_reference_ratio_p50"])
    baseline_star_gap = abs(baseline_star_ratio - 1.0)
    optimized_star_gap = abs(optimized_star_ratio - 1.0)
    star = {
        "baseline_reference_ratio": baseline_star_ratio,
        "optimized_reference_ratio": optimized_star_ratio,
        "baseline_gap": baseline_star_gap,
        "optimized_gap": optimized_star_gap,
        "gap_shrinkage": _gap_shrinkage(
            baseline_star_gap, optimized_star_gap
        ),
        "absolute_target_passed": bool(
            LOCKED_LIMITS["star_fwhm_ratio_min"]
            <= optimized_star_ratio
            <= LOCKED_LIMITS["star_fwhm_ratio_max"]
        ),
        "not_degraded_over_10_percent": _nonregressed(
            baseline_star_gap, optimized_star_gap
        ),
    }
    star["improved"] = bool(
        star["absolute_target_passed"]
        and star["not_degraded_over_10_percent"]
        and optimized_star_gap <= baseline_star_gap + 1e-12
    )

    high_frequency = _comparison(
        float(baseline["background_high_frequency_sigma"]),
        float(optimized["background_high_frequency_sigma"]),
        float(reference["background_high_frequency_sigma"]),
        name="background_high_frequency_sigma",
    )
    low_frequency = _comparison(
        float(baseline["background_low_frequency_residual_sigma"]),
        float(optimized["background_low_frequency_residual_sigma"]),
        float(reference["background_low_frequency_residual_sigma"]),
        name="background_low_frequency_residual_sigma",
    )
    high_frequency["absolute_or_shrinkage_passed"] = bool(
        (
            high_frequency["gap_shrinkage"] is not None
            and high_frequency["gap_shrinkage"]
            >= LOCKED_LIMITS["background_hf_gap_shrinkage_min"]
        )
        or high_frequency["optimized_reference_ratio"]
        <= LOCKED_LIMITS["background_hf_ratio_max"]
    )
    low_frequency["absolute_target_passed"] = bool(
        low_frequency["optimized_reference_ratio"]
        <= LOCKED_LIMITS["background_lf_ratio_max"]
    )
    noise = {
        "high_frequency": high_frequency,
        "low_frequency": low_frequency,
        "not_degraded_over_10_percent": bool(
            high_frequency["not_degraded_over_10_percent"]
            and low_frequency["not_degraded_over_10_percent"]
        ),
    }
    noise["improved"] = bool(
        high_frequency["absolute_or_shrinkage_passed"]
        and low_frequency["absolute_target_passed"]
        and noise["not_degraded_over_10_percent"]
    )

    detail = _comparison(
        float(baseline["detail_band_robust_sigma"]),
        float(optimized["detail_band_robust_sigma"]),
        float(reference["detail_band_robust_sigma"]),
        name="detail_band_robust_sigma",
    )
    detail.update(
        baseline_structure_correlation=baseline_structure,
        optimized_structure_correlation=optimized_structure,
        structure_correlation_change=(
            optimized_structure - baseline_structure
        ),
        structure_guard_passed=(
            optimized_structure
            >= baseline_structure
            - LOCKED_LIMITS["structure_correlation_drop_max"]
        ),
    )
    detail["improved"] = bool(
        detail["gap_shrinkage"] is not None
        and detail["gap_shrinkage"]
        >= LOCKED_LIMITS["detail_gap_shrinkage_min"]
        and detail["structure_guard_passed"]
    )

    contrast = _comparison(
        float(baseline["subject_background_contrast"]),
        float(optimized["subject_background_contrast"]),
        float(reference["subject_background_contrast"]),
        name="subject_background_contrast",
    )
    contrast["improved"] = bool(
        contrast["gap_shrinkage"] is not None
        and contrast["gap_shrinkage"]
        >= LOCKED_LIMITS["contrast_gap_shrinkage_min"]
    )
    return {
        "color": color,
        "detail": detail,
        "stars": star,
        "noise": noise,
        "contrast": contrast,
    }


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not str(key).startswith("_")
    }


def evaluate_entry(entry: Mapping[str, Any], cv2: Any) -> dict[str, Any]:
    baseline_raw, baseline_shape = _read_rgb(Path(entry["baseline"]), cv2)
    optimized_raw, optimized_shape = _read_rgb(Path(entry["optimized"]), cv2)
    reference_raw, reference_shape = _read_rgb(Path(entry["reference"]), cv2)
    baseline, baseline_valid, baseline_registration = _register_to_reference(
        baseline_raw,
        reference_raw,
        cv2,
        source_shape=baseline_shape,
        reference_shape=reference_shape,
    )
    optimized, optimized_valid, optimized_registration = _register_to_reference(
        optimized_raw,
        reference_raw,
        cv2,
        source_shape=optimized_shape,
        reference_shape=reference_shape,
    )
    reference, _ = _resize_max_side(reference_raw, cv2)
    if baseline.shape != optimized.shape or baseline.shape != reference.shape:
        raise QAError("registered baseline/optimized/reference shapes differ")
    common_valid = binary_erosion(
        baseline_valid & optimized_valid,
        iterations=4,
    )
    if float(np.mean(common_valid)) < MIN_REGISTRATION_OVERLAP:
        raise QAError("common registered overlap is insufficient")
    masks = _reference_masks(reference, common_valid, str(entry["profile"]))
    baseline_metrics = _image_metrics(baseline, masks)
    optimized_metrics = _image_metrics(optimized, masks)
    reference_metrics = _image_metrics(reference, masks)
    baseline_structure = _structure_correlation(
        baseline_metrics["_detail_band"],
        reference_metrics["_detail_band"],
        masks["subject"],
    )
    optimized_structure = _structure_correlation(
        optimized_metrics["_detail_band"],
        reference_metrics["_detail_band"],
        masks["subject"],
    )
    stars = _matched_star_metrics(
        baseline, optimized, reference, masks
    )
    dimensions = _dimension_reports(
        baseline_metrics,
        optimized_metrics,
        reference_metrics,
        stars,
        baseline_structure,
        optimized_structure,
    )
    improved_count = sum(
        dimension.get("improved") is True
        for dimension in dimensions.values()
    )
    no_dimension_regressed = all(
        dimension.get("not_degraded_over_10_percent") is True
        for dimension in dimensions.values()
    )
    structure_guard = bool(dimensions["detail"]["structure_guard_passed"])
    artifacts = _artifact_guard(baseline_metrics, optimized_metrics)
    quantitative_accepted = bool(
        improved_count >= LOCKED_LIMITS["dimensions_improved_min"]
        and no_dimension_regressed
        and structure_guard
        and artifacts["accepted"]
    )
    visual_review = entry.get("visual_artifact_review")
    visual_passed = bool(
        isinstance(visual_review, Mapping)
        and visual_review.get("passed") is True
    )
    accepted, status = _acceptance_status(
        quantitative_accepted,
        visual_passed,
    )
    return {
        "target": entry["target"],
        "profile": entry["profile"],
        "status": status,
        "accepted": accepted,
        "quantitative_accepted": quantitative_accepted,
        "visual_artifact_review_passed": visual_passed,
        "visual_artifact_review_required": not visual_passed,
        "visual_artifact_review": dict(visual_review or {}),
        "improved_dimensions": [
            name for name, value in dimensions.items() if value["improved"]
        ],
        "improved_dimension_count": improved_count,
        "no_dimension_regressed_over_10_percent": no_dimension_regressed,
        "structure_correlation_guard_passed": structure_guard,
        "inputs": {
            "baseline": {
                "path": str(entry["baseline"]),
                "sha256": _sha256(Path(entry["baseline"])),
            },
            "optimized": {
                "path": str(entry["optimized"]),
                "sha256": _sha256(Path(entry["optimized"])),
            },
            "reference": {
                "path": str(entry["reference"]),
                "sha256": _sha256(Path(entry["reference"])),
            },
        },
        "production_provenance": {
            "baseline": dict(entry["baseline_provenance"]),
            "optimized": dict(entry["optimized_provenance"]),
        },
        "registration": {
            "baseline_to_reference": baseline_registration,
            "optimized_to_reference": optimized_registration,
            "common_overlap_ratio": float(np.mean(common_valid)),
        },
        "mask_support": {
            name: int(np.count_nonzero(mask))
            for name, mask in masks.items()
            if name != "star_peaks"
        },
        "measurements": {
            "baseline": _public_metrics(baseline_metrics),
            "optimized": _public_metrics(optimized_metrics),
            "reference": _public_metrics(reference_metrics),
            "matched_stars": stars,
        },
        "dimensions": dimensions,
        "artifact_guard": artifacts,
    }


def _fit_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (8, 8, 8))
    panel.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return panel


def write_contact_sheet(
    entries: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    panel_size = (460, 320)
    margin = 24
    gap = 18
    title_height = 54
    row_label_height = 54
    row_height = row_label_height + panel_size[1]
    width = margin * 2 + panel_size[0] * 3 + gap * 2
    height = title_height + margin + len(entries) * (row_height + gap)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 16),
        "External-reference QA — BASELINE / OPTIMIZED / REFERENCE",
        fill=(245, 245, 245),
    )
    result_by_target = {str(item.get("target")): item for item in results}
    y = title_height
    for entry in entries:
        result = result_by_target.get(str(entry["target"]), {})
        state = str(result.get("status") or "failed").upper()
        count = result.get("improved_dimension_count", 0)
        draw.text(
            (margin, y + 10),
            f"{entry['target']}  |  {state}  |  improved dimensions {count}/5",
            fill=(255, 255, 255),
        )
        y += row_label_height
        for column, key in enumerate(("baseline", "optimized", "reference")):
            x = margin + column * (panel_size[0] + gap)
            panel = _fit_panel(Path(entry[key]), panel_size)
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + panel_size[0] - 1, y + panel_size[1] - 1),
                outline=(90, 90, 90),
            )
            draw.text((x + 8, y + 8), key.upper(), fill=(255, 225, 180))
        y += panel_size[1] + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=95, subsampling=0)


def _format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.3f}" if math.isfinite(number) else "n/a"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Five-target external-reference QA",
        "",
        "External reference images were used only for offline QA. Production feedback is disabled.",
        "",
        f"- Status: `{report['status']}`",
        f"- External reference used: `{str(report['external_reference_used']).lower()}`",
        f"- Production feedback: `{str(report['production_feedback']).lower()}`",
        f"- Production pixels written: `{str(report['production_pixels_written']).lower()}`",
        f"- Five-target coverage: `{str(report['coverage_gate']['accepted']).lower()}`",
        "",
        "| Target | Status | Improved | Saturation ratio | FWHM ratio | HF noise ratio | LF residual ratio | Structure corr change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        if result.get("status") == "failed":
            lines.append(
                f"| {result.get('target', 'unknown')} | failed | 0/5 | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        dimensions = result["dimensions"]
        lines.append(
            "| {target} | {status} | {count}/5 | {sat} | {fwhm} | {hf} | {lf} | {corr} |".format(
                target=result["target"],
                status=result["status"],
                count=result["improved_dimension_count"],
                sat=_format_float(
                    dimensions["color"]["optimized_reference_ratio"]
                ),
                fwhm=_format_float(
                    dimensions["stars"]["optimized_reference_ratio"]
                ),
                hf=_format_float(
                    dimensions["noise"]["high_frequency"][
                        "optimized_reference_ratio"
                    ]
                ),
                lf=_format_float(
                    dimensions["noise"]["low_frequency"][
                        "optimized_reference_ratio"
                    ]
                ),
                corr=_format_float(
                    dimensions["detail"]["structure_correlation_change"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Acceptance boundary",
            "",
            "Quantitative acceptance requires at least four of five dimensions to improve, no dimension gap to regress by more than 10%, and structure correlation not to fall by more than 0.02. Seam, halo, core morphology, and implausible local colour remain a human contact-sheet gate.",
            "",
            f"Contact sheet: `{report['artifacts']['contact_sheet']}`",
            f"Contact sheet SHA-256: `{report['artifacts']['contact_sheet_sha256']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_qa(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    destination = output_dir.expanduser().resolve()
    input_paths = {
        Path(entry[key]).resolve()
        for entry in manifest["entries"]
        for key in ("baseline", "optimized", "reference")
    }
    protected_roots = {
        Path(entry["baseline_run_root"]).resolve()
        for entry in manifest["entries"]
    } | {
        Path(entry["optimized_run_root"]).resolve()
        for entry in manifest["entries"]
    } | {
        Path(entry["reference"]).resolve().parent
        for entry in manifest["entries"]
    }
    for protected_root in protected_roots:
        try:
            destination.relative_to(protected_root)
        except ValueError:
            continue
        raise QAError(
            "report output directory must be outside every production run and "
            "reference input tree"
        )
    for input_path in input_paths:
        if destination == input_path:
            raise QAError("report output directory conflicts with an input file")
    destination.mkdir(parents=True, exist_ok=True)
    cv2 = _load_cv2()
    results: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        try:
            before = dict(entry["input_sha256"])
            result = evaluate_entry(entry, cv2)
            after = {
                key: _sha256(Path(entry[key]))
                for key in ("baseline", "optimized", "reference")
            }
            if after != before:
                raise QAError(
                    f"QA input pixels changed during evaluation for {entry['target']}"
                )
            result["input_immutability"] = {
                "verified": True,
                "before_sha256": before,
                "after_sha256": after,
            }
            results.append(result)
        except Exception as exc:  # keep a complete five-target failure report
            results.append(
                {
                    "target": entry["target"],
                    "status": "failed",
                    "accepted": False,
                    "quantitative_accepted": False,
                    "visual_artifact_review_passed": bool(
                        entry["visual_artifact_review_passed"]
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                    "inputs": {
                        key: {"path": str(entry[key]), "sha256": _sha256(Path(entry[key]))}
                        for key in ("baseline", "optimized", "reference")
                    },
                }
            )
    observed_targets = {str(item.get("target")) for item in results}
    coverage_accepted = observed_targets == set(EXPECTED_TARGET_PROFILES)
    any_failed = any(item["status"] == "failed" for item in results)
    all_accepted = bool(results) and coverage_accepted and all(
        item["accepted"] for item in results
    )
    any_rejected = any(item["status"] == "rejected" for item in results)
    status = (
        "failed"
        if any_failed
        else "accepted"
        if all_accepted
        else "rejected"
        if any_rejected
        else "review_required"
    )
    json_path = destination / "five_target_reference_qa.json"
    markdown_path = destination / "five_target_reference_qa.md"
    contact_sheet_path = destination / "five_target_reference_qa.jpg"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "accepted": all_accepted,
        "external_reference_used": True,
        "production_feedback": False,
        "production_pixels_written": False,
        "measurement_domain": "registered_display_rgb_0_1",
        "manifest": {
            "path": str(manifest["path"]),
            "sha256": _sha256(Path(manifest["path"])),
            "schema": MANIFEST_SCHEMA,
        },
        "coverage_gate": {
            "accepted": coverage_accepted,
            "expected_targets": sorted(EXPECTED_TARGET_PROFILES),
            "observed_targets": sorted(observed_targets),
            "expected_profiles": dict(EXPECTED_TARGET_PROFILES),
        },
        "path_isolation_gate": {
            "accepted": True,
            "output_directory": str(destination),
            "protected_roots": sorted(str(root) for root in protected_roots),
        },
        "locked_limits": dict(LOCKED_LIMITS),
        "results": results,
        "summary": {
            "target_count": len(results),
            "accepted_count": sum(item["accepted"] for item in results),
            "quantitative_accepted_count": sum(
                item.get("quantitative_accepted") is True for item in results
            ),
            "failed_count": sum(item["status"] == "failed" for item in results),
            "review_required_count": sum(
                item["status"] == "review_required" for item in results
            ),
            "rejected_count": sum(item["status"] == "rejected" for item in results),
        },
        "artifacts": {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "contact_sheet": str(contact_sheet_path),
        },
        "human_review_boundary": {
            "required_unless_all_bound_checks_passed": True,
            "required_checks": list(VISUAL_REVIEW_CHECKS),
            "optimized_sha256_by_target": {
                str(entry["target"]): str(entry["input_sha256"]["optimized"])
                for entry in manifest["entries"]
            },
            "full_resolution_review_required": True,
        },
    }
    write_contact_sheet(manifest["entries"], results, contact_sheet_path)
    report["artifacts"]["contact_sheet_sha256"] = _sha256(contact_sheet_path)
    write_markdown(report, markdown_path)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimized display images with external references"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_qa(args.manifest, args.output_dir)
    except QAError as exc:
        print(f"five-target reference QA failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    if report["status"] == "failed":
        return 2
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
