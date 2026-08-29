"""Stage 6 to Stage 8 star-associated halo protection contract."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from astropy.io import fits

from image_metrics import _box_blur_gray, _to_rgb_float_fullres
from local_adjustments import dilate_mask, feather_mask


SCHEMA = "starun.stage6-star-halo-guard.v1"
REPORT_NAME = "stage6_star_halo_guard.json"
ARTIFACT_NAME = "stage6_star_halo_guard.fit"


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _read_primary(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"FITS primary image is empty: {path.name}")
        return np.asarray(data)


def _components8(mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    support = np.asarray(mask, dtype=bool)
    height, width = support.shape
    visited = np.zeros_like(support, dtype=bool)
    components: List[Tuple[np.ndarray, np.ndarray]] = []
    for y, x in np.argwhere(support):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        points: List[Tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and support[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(points) >= 2:
            yy, xx = zip(*points)
            components.append(
                (
                    np.asarray(yy, dtype=np.int32),
                    np.asarray(xx, dtype=np.int32),
                )
            )
    return components


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def build_star_halo_guard(
    starless: np.ndarray,
    starmask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a soft guard and component-local residual audit."""

    starless_rgb = _to_rgb_float_fullres(starless)
    starmask_rgb = _to_rgb_float_fullres(starmask)
    if starless_rgb.shape != starmask_rgb.shape:
        raise ValueError("starless/starmask shape mismatch")
    luma = (
        0.2126 * starless_rgb[0]
        + 0.7152 * starless_rgb[1]
        + 0.0722 * starless_rgb[2]
    ).astype(np.float32)
    mask_luma = (
        0.2126 * starmask_rgb[0]
        + 0.7152 * starmask_rgb[1]
        + 0.0722 * starmask_rgb[2]
    ).astype(np.float32)
    floor = float(np.quantile(mask_luma, 0.55))
    signal = np.clip(mask_luma - floor, 0.0, None)
    positive = signal[signal > 0.0]
    empty = np.zeros_like(luma, dtype=np.float32)
    if positive.size < 8:
        return empty, {
            "status": "not_applicable",
            "reason_code": "starmask_compact_support_unavailable",
            "component_count": 0,
            "hard_anomaly_count": 0,
            "coverage": 0.0,
            "components": [],
        }
    scale = max(float(np.quantile(positive, 0.995)), 1e-7)
    normalized = np.clip(signal / scale, 0.0, 1.0)
    threshold = max(0.12, float(np.quantile(normalized[normalized > 0.0], 0.90)))
    components = _components8(normalized >= threshold)
    texture = np.abs(luma - _box_blur_gray(luma))
    chroma = (np.max(starless_rgb, axis=0) - np.min(starless_rgb, axis=0)).astype(
        np.float32
    )
    guard = np.zeros_like(luma, dtype=np.float32)
    records: List[Dict[str, Any]] = []
    hard_count = 0
    for index, (ys, xs) in enumerate(components[:512], start=1):
        area = int(ys.size)
        if area < 2:
            continue
        radius = max(1.0, math.sqrt(area / math.pi))
        inner_expand = max(1, int(math.ceil(0.5 * radius)))
        outer_expand = min(24, max(5, int(math.ceil(2.0 * radius))))
        control_expand = max(3, inner_expand)
        margin = outer_expand + control_expand + 3
        y0 = max(0, int(np.min(ys)) - margin)
        y1 = min(luma.shape[0], int(np.max(ys)) + margin + 1)
        x0 = max(0, int(np.min(xs)) - margin)
        x1 = min(luma.shape[1], int(np.max(xs)) + margin + 1)
        component = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        component[ys - y0, xs - x0] = 1.0
        core = dilate_mask(component, iterations=inner_expand)
        outer = dilate_mask(component, iterations=outer_expand)
        control_outer = dilate_mask(outer, iterations=control_expand)
        ring = (outer > 0.5) & (core <= 0.5)
        control = (control_outer > 0.5) & (outer <= 0.5)
        if int(np.count_nonzero(ring)) < 8 or int(np.count_nonzero(control)) < 8:
            continue
        local_texture = texture[y0:y1, x0:x1]
        local_chroma = chroma[y0:y1, x0:x1]
        control_texture = local_texture[control]
        control_chroma = local_chroma[control]
        texture_sigma = max(_robust_sigma(control_texture), 1e-6)
        chroma_sigma = max(_robust_sigma(control_chroma), 1e-6)
        texture_delta = float(
            np.median(local_texture[ring]) - np.median(control_texture)
        )
        chroma_delta = float(
            np.median(local_chroma[ring]) - np.median(control_chroma)
        )
        texture_z = texture_delta / texture_sigma
        chroma_z = chroma_delta / chroma_sigma
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        local_yy, local_xx = np.indices(component.shape, dtype=np.float32)
        angles = np.arctan2(
            local_yy + float(y0) - cy,
            local_xx + float(x0) - cx,
        )
        coherent = 0
        populated = 0
        for sector in range(8):
            lower = -math.pi + sector * math.pi / 4.0
            upper = lower + math.pi / 4.0
            sector_mask = ring & (angles >= lower) & (angles < upper)
            if int(np.count_nonzero(sector_mask)) < 2:
                continue
            populated += 1
            if (
                float(np.median(local_texture[sector_mask]))
                > float(np.median(control_texture)) + 3.0 * texture_sigma
                or float(np.median(local_chroma[sector_mask]))
                > float(np.median(control_chroma)) + 3.0 * chroma_sigma
            ):
                coherent += 1
        radial_coherence = coherent / max(populated, 1)
        hard_anomaly = bool(
            max(texture_z, chroma_z) > 3.0
            and radial_coherence >= 0.50
        )
        hard_count += int(hard_anomaly)
        guard[y0:y1, x0:x1] = np.maximum(
            guard[y0:y1, x0:x1],
            feather_mask(outer, radius=2),
        )
        records.append(
            {
                "id": index,
                "area": area,
                "centroid": {"x": cx, "y": cy},
                "equivalent_radius": radius,
                "ring_pixels": int(np.count_nonzero(ring)),
                "control_pixels": int(np.count_nonzero(control)),
                "texture_delta": texture_delta,
                "texture_z": texture_z,
                "chroma_delta": chroma_delta,
                "chroma_z": chroma_z,
                "radial_coherence": radial_coherence,
                "hard_anomaly": hard_anomaly,
            }
        )
    guard = np.clip(guard, 0.0, 1.0).astype(np.float32)
    return guard, {
        "status": "hard_failed" if hard_count else "ok",
        "reason_code": (
            "stage6_local_star_halo_residual"
            if hard_count
            else "stage6_star_halo_guard_ready"
        ),
        "component_count": len(records),
        "hard_anomaly_count": hard_count,
        "coverage": float(np.mean(guard > 0.05)),
        "threshold": threshold,
        "components": records,
    }


def persist_stage6_guard(pipeline, *, starless_path: Path, starmask_path: Path) -> Dict[str, Any]:
    process_dir = Path(pipeline.process_dir)
    artifact = process_dir / ARTIFACT_NAME
    report_path = process_dir / REPORT_NAME
    try:
        starless = _read_primary(starless_path)
        starmask = _read_primary(starmask_path)
        guard, metrics = build_star_halo_guard(starless, starmask)
        fits.PrimaryHDU(guard).writeto(artifact, overwrite=True, checksum=True)
        artifact_sha = _sha256(artifact)
        if not artifact_sha:
            raise OSError("halo guard SHA256 unavailable")
        report: Dict[str, Any] = {
            "schema": SCHEMA,
            "run_id": str(getattr(pipeline, "_run_id", "") or ""),
            "status": metrics.get("status"),
            "reason_code": metrics.get("reason_code"),
            "source": {
                "starless": starless_path.name,
                "starless_sha256": _sha256(starless_path),
                "starmask": starmask_path.name,
                "starmask_sha256": _sha256(starmask_path),
            },
            "artifact": ARTIFACT_NAME,
            "artifact_sha256": artifact_sha,
            "shape": [int(value) for value in guard.shape],
            "metrics": metrics,
        }
    except (OSError, TypeError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "run_id": str(getattr(pipeline, "_run_id", "") or ""),
            "status": "failed",
            "reason_code": "stage6_star_halo_guard_failed",
            "error": str(error),
        }
        try:
            artifact.unlink(missing_ok=True)
        except OSError:
            pass
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def verify_stage6_guard(pipeline, shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    process_dir = Path(pipeline.process_dir)
    report_path = process_dir / REPORT_NAME
    artifact = process_dir / ARTIFACT_NAME
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != SCHEMA:
            raise ValueError("halo guard schema mismatch")
        expected_run = str(getattr(pipeline, "_run_id", "") or "")
        if expected_run and str(report.get("run_id") or "") != expected_run:
            raise ValueError("halo guard run_id mismatch")
        if report.get("artifact") != ARTIFACT_NAME:
            raise ValueError("halo guard artifact name mismatch")
        artifact_sha = _sha256(artifact)
        if artifact_sha != report.get("artifact_sha256"):
            raise ValueError("halo guard SHA256 mismatch")
        expected = (
            (getattr(pipeline, "_stage8_handoff", {}) or {}).get(
                "star_halo_guard"
            )
            or {}
        )
        if expected:
            if str(expected.get("report") or REPORT_NAME) != REPORT_NAME:
                raise ValueError("halo guard handoff report name mismatch")
            if str(expected.get("artifact") or ARTIFACT_NAME) != ARTIFACT_NAME:
                raise ValueError("halo guard handoff artifact name mismatch")
            expected_sha = str(expected.get("artifact_sha256") or "")
            if expected_sha and expected_sha != artifact_sha:
                raise ValueError("halo guard handoff SHA256 mismatch")
        source = report.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("halo guard source lineage is missing")
        starless_name = str(source.get("starless") or "")
        starmask_name = str(source.get("starmask") or "")
        if not starless_name or Path(starless_name).name != starless_name:
            raise ValueError("halo guard starless source name is invalid")
        if not starmask_name or Path(starmask_name).name != starmask_name:
            raise ValueError("halo guard starmask source name is invalid")
        starless_path = process_dir / starless_name
        if _sha256(starless_path) != source.get("starless_sha256"):
            raise ValueError("halo guard starless source SHA256 mismatch")
        starmask_candidates = [process_dir / starmask_name]
        active_starmask = getattr(pipeline, "starmask_file", None)
        if active_starmask is not None:
            active_path = Path(active_starmask)
            if active_path.name == starmask_name:
                starmask_candidates.insert(0, active_path)
        if not any(
            _sha256(path) == source.get("starmask_sha256")
            for path in starmask_candidates
        ):
            raise ValueError("halo guard starmask source SHA256 mismatch")
        guard = np.asarray(_read_primary(artifact), dtype=np.float32)
        guard = np.squeeze(guard)
        if guard.shape != tuple(shape) or not np.all(np.isfinite(guard)):
            raise ValueError("halo guard shape or finite-value check failed")
        return np.clip(guard, 0.0, 1.0), report
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        return None, {
            "schema": SCHEMA,
            "status": "rejected",
            "reason_code": "stage6_star_halo_guard_lineage_unverified",
            "error": str(error),
        }


def apply_guard_to_masks(pipeline, masks: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(masks)
    gray = np.asarray(result.get("gray"))
    if gray.ndim != 2:
        return result
    process_dir = getattr(pipeline, "process_dir", None)
    expected = (
        (getattr(pipeline, "_stage8_handoff", {}) or {}).get(
            "star_halo_guard"
        )
        or {}
    )
    if process_dir is None:
        if str(expected.get("status") or "") in {"ok", "hard_failed"}:
            raise ValueError("stage6 star-halo guard workspace is unavailable")
        pipeline._stage8_star_halo_guard_verified = False
        return result
    guard, report = verify_stage6_guard(pipeline, tuple(gray.shape))
    pipeline._stage8_star_halo_guard_report = report
    if guard is None:
        pipeline._stage8_star_halo_guard_verified = False
        if str(expected.get("status") or "") in {"ok", "hard_failed"}:
            pipeline._stage8_halo_guard_lineage_rejected = True
            raise ValueError("stage6 star-halo guard lineage verification failed")
        return result
    pipeline._stage8_star_halo_guard_verified = True
    protection = np.clip(1.0 - guard, 0.0, 1.0)
    for name in (
        "nebula_mask",
        "faint_nebula_mask",
        "subject_mask",
        "galaxy_signal_mask",
    ):
        value = result.get(name)
        if value is not None and np.asarray(value).shape == guard.shape:
            result[name] = np.clip(
                np.asarray(value, dtype=np.float32) * protection,
                0.0,
                1.0,
            )
    result["star_halo_guard_mask"] = guard
    coverage = dict(result.get("coverage") or {})
    coverage["star_halo_guard"] = float(np.mean(guard > 0.05))
    result["coverage"] = coverage
    return result


def assess_candidate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    guard: Optional[np.ndarray],
    *,
    mode: str,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema": "starun.stage8-star-halo-local-gate.v1",
        "status": "not_applicable",
        "accepted": True,
        "mode": mode,
        "issues": [],
    }
    if guard is None:
        return report
    base_rgb = _to_rgb_float_fullres(baseline)
    cand_rgb = _to_rgb_float_fullres(candidate)
    if base_rgb.shape != cand_rgb.shape or guard.shape != base_rgb.shape[1:]:
        report.update(status="unavailable", accepted=False, issues=["shape_mismatch"])
        return report
    support = np.asarray(guard) > 0.05
    if int(np.count_nonzero(support)) < 8:
        return report
    base_luma = 0.2126 * base_rgb[0] + 0.7152 * base_rgb[1] + 0.0722 * base_rgb[2]
    cand_luma = 0.2126 * cand_rgb[0] + 0.7152 * cand_rgb[1] + 0.0722 * cand_rgb[2]
    base_texture = np.abs(base_luma - _box_blur_gray(base_luma))
    cand_texture = np.abs(cand_luma - _box_blur_gray(cand_luma))
    texture_delta = float(np.median(cand_texture[support] - base_texture[support]))
    texture_growth = float(
        np.median(cand_texture[support])
        / max(float(np.median(base_texture[support])), 1e-6)
    )
    base_chroma = np.max(base_rgb, axis=0) - np.min(base_rgb, axis=0)
    cand_chroma = np.max(cand_rgb, axis=0) - np.min(cand_rgb, axis=0)
    chroma_delta_p95 = float(np.quantile(cand_chroma[support] - base_chroma[support], 0.95))
    issues: List[str] = []
    if mode == "luminance" and texture_growth > 1.05 and texture_delta > 0.00075:
        issues.append("star_halo_texture_growth")
    if mode == "color" and chroma_delta_p95 > 0.006:
        issues.append("star_halo_chroma_growth")
    report.update(
        status="rejected" if issues else "ok",
        accepted=not issues,
        issues=issues,
        support_count=int(np.count_nonzero(support)),
        texture_growth=texture_growth,
        texture_delta=texture_delta,
        chroma_delta_p95=chroma_delta_p95,
    )
    return report
