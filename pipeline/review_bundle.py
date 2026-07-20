"""Create compact visual evidence bundles for human or multimodal review."""
from __future__ import annotations

import base64
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from image_metrics import _to_rgb_float_image, measure_image_features, measure_quality_metrics
from save_utils import write_png_rgb16
from sirilpy.exceptions import CommandError, DataError, SirilError


def _safe_preview(rgb: np.ndarray) -> np.ndarray:
    """Create a display preview without clipping shadow percentiles."""
    gray = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    high = max(float(np.quantile(gray, 0.995)), 1e-6)
    preview = np.sqrt(np.clip(rgb / high, 0.0, 1.0))
    return np.flip(preview, axis=1)


def _aligned_previews(before: Any, after: Any) -> tuple[np.ndarray, np.ndarray]:
    before_rgb = _to_rgb_float_image(np.asarray(before), max_side=1600)
    after_rgb = _to_rgb_float_image(np.asarray(after), max_side=1600)
    height = min(before_rgb.shape[1], after_rgb.shape[1])
    width = min(before_rgb.shape[2], after_rgb.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError("review images have no overlapping pixels")
    return before_rgb[:, :height, :width], after_rgb[:, :height, :width]


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            delta[key] = round(float(after_value) - float(before_value), 6)
    return delta


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def image_path_to_data_url(path: Path) -> str:
    """Encode an existing PNG/JPEG preview as an OpenAI-compatible data URL."""
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_data_to_data_url(image_data: Any) -> str:
    """Build a bounded, display-stretched PNG data URL from a Siril pixel array."""
    rgb = _to_rgb_float_image(np.asarray(image_data), max_side=960)
    preview = _safe_preview(rgb)
    # Reuse the deterministic in-project PNG writer without adding Pillow.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="seestar-ai-preview-") as tmpdir:
        path = Path(tmpdir) / "preview.png"
        write_png_rgb16(path, preview)
        return image_path_to_data_url(path)


def _candidate_identity(candidate: Mapping[str, Any], index: int) -> str:
    for key in ("id", "name", "label", "attempt", "stem", "file"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"candidate_{index + 1}"


def _candidate_records(
    candidates: Optional[list[Mapping[str, Any]]],
    selected_candidate: Optional[str],
    *,
    default_after: str,
) -> list[Dict[str, Any]]:
    raw_candidates = list(candidates or [])
    if not raw_candidates:
        raw_candidates = [{"name": default_after, "status": "selected"}]
    selected_text = str(selected_candidate or "").strip()
    records: list[Dict[str, Any]] = []
    for index, item in enumerate(raw_candidates):
        record = dict(item)
        identity = _candidate_identity(record, index)
        is_selected = bool(record.get("selected")) or (
            bool(selected_text)
            and selected_text
            in {
                identity,
                str(record.get("name") or ""),
                str(record.get("label") or ""),
                str(record.get("attempt") or ""),
                str(record.get("stem") or ""),
                str(record.get("file") or ""),
            }
        )
        if not selected_text and len(raw_candidates) == 1:
            is_selected = True
        algorithm_status = str(record.get("status") or "unknown").lower()
        record.update(
            {
                "id": identity,
                "selection_status": "selected" if is_selected else "not_selected",
                "visual_acceptance_status": (
                    "unavailable"
                    if algorithm_status in {"failed", "command_failed", "rejected"}
                    else ("not_requested" if is_selected else "not_selected")
                ),
            }
        )
        records.append(record)
    if not any(item["selection_status"] == "selected" for item in records):
        records[-1]["selection_status"] = "selected"
        records[-1]["visual_acceptance_status"] = "not_requested"
    return records


def create_image_review_bundle(
    before_data: Any,
    after_data: Any,
    *,
    output_dir: Path,
    stage_key: str,
    source: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
    candidates: Optional[list[Mapping[str, Any]]] = None,
    selected_candidate: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a review bundle from two already-loaded pixel arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    before_rgb, after_rgb = _aligned_previews(before_data, after_data)
    difference = after_rgb - before_rgb
    absolute = np.abs(difference)
    scale = max(float(np.quantile(absolute, 0.995)), 1e-6)
    absolute_preview = np.flip(np.clip(absolute / scale, 0.0, 1.0), axis=1)
    luminance_delta = (
        0.2126 * difference[0]
        + 0.7152 * difference[1]
        + 0.0722 * difference[2]
    )
    signed = np.zeros_like(after_rgb, dtype=np.float32)
    signed[0] = np.clip(luminance_delta / scale, 0.0, 1.0)
    signed[2] = np.clip(-luminance_delta / scale, 0.0, 1.0)
    signed = np.flip(signed, axis=1)

    paths = {
        "before_preview": output_dir / "before.png",
        "after_preview": output_dir / "after.png",
        "absolute_difference": output_dir / "difference.png",
        "signed_luminance_difference": output_dir / "signed_difference.png",
    }
    write_png_rgb16(paths["before_preview"], _safe_preview(before_rgb))
    write_png_rgb16(paths["after_preview"], _safe_preview(after_rgb))
    write_png_rgb16(paths["absolute_difference"], absolute_preview)
    write_png_rgb16(paths["signed_luminance_difference"], signed)

    before_features = asdict(measure_image_features(np.asarray(before_data)))
    after_features = asdict(measure_image_features(np.asarray(after_data)))
    before_quality = asdict(measure_quality_metrics(np.asarray(before_data)))
    after_quality = asdict(measure_quality_metrics(np.asarray(after_data)))
    payload = {
        "schema_version": 2,
        "stage": stage_key,
        "status": "ready",
        "source": dict(source),
        "previews": {key: str(path) for key, path in paths.items()},
        "metrics": {
            "before": {"features": before_features, "quality": before_quality},
            "after": {"features": after_features, "quality": after_quality},
            "delta": {
                "features": _metric_delta(before_features, after_features),
                "quality": _metric_delta(before_quality, after_quality),
            },
        },
        "context": dict(context or {}),
        "candidates": _candidate_records(
            candidates,
            selected_candidate,
            default_after=str(source.get("after_stem") or source.get("after_path") or stage_key),
        ),
        "visual_review": {
            "status": "not_requested",
            "advisor_mode": "not_requested",
            "acceptance_blocking": False,
            "checklist": [
                "no invented or removed astronomical structure",
                "no black holes, seams, or over-smoothed background",
                "no star halos, dark rims, color fringing, or abnormal growth",
                "no clipped highlight core or excessive saturation",
            ],
        },
    }
    report_path = output_dir / "review.json"
    _write_json_atomic(report_path, payload)
    payload["report_path"] = str(report_path)
    return payload


def apply_visual_acceptance(
    payload: Dict[str, Any],
    result: Optional[Mapping[str, Any]],
    *,
    advisor_mode: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a non-blocking visual verdict and mirror it onto the selected candidate."""
    visual_review = dict(payload.get("visual_review") or {})
    normalized_advisor_mode = str(advisor_mode or "not_requested").strip().lower()
    visual_review["advisor_mode"] = normalized_advisor_mode
    visual_review["acceptance_blocking"] = False
    if result:
        verdict = str(result.get("verdict") or "review_required").strip().lower()
        status_map = {
            "accept": "accepted",
            "accepted": "accepted",
            "review": "review_required",
            "review_required": "review_required",
            "reject": "rejected",
            "rejected": "rejected",
        }
        status = status_map.get(verdict, "review_required")
        visual_review.update(
            {
                "status": status,
                "confidence": result.get("confidence"),
                "summary": result.get("summary"),
                "issues": list(result.get("issues") or []),
                "recommended_parameter_ranges": dict(
                    result.get("recommended_parameter_ranges") or {}
                ),
            }
        )
    elif error:
        status = "unavailable"
        visual_review["status"] = status
        visual_review["advisor_error"] = error
    elif normalized_advisor_mode == "multimodal":
        status = "unavailable"
        visual_review["status"] = status
        visual_review["advisor_note"] = "visual advisor returned no verdict"
    else:
        status = "not_requested"
        visual_review["status"] = status
    payload["visual_review"] = visual_review
    for candidate in payload.get("candidates", []):
        if candidate.get("selection_status") == "selected":
            candidate["visual_acceptance_status"] = status

    report_path = Path(str(payload.get("report_path") or ""))
    if report_path.name:
        disk_payload = dict(payload)
        disk_payload.pop("report_path", None)
        _write_json_atomic(report_path, disk_payload)
    return payload


def create_stage_review_bundle(
    pipeline: Any,
    *,
    stage_key: str,
    before_stem: str,
    after_stem: str,
    context: Optional[Mapping[str, Any]] = None,
    candidates: Optional[list[Mapping[str, Any]]] = None,
    selected_candidate: Optional[str] = None,
) -> Dict[str, Any]:
    """Load two Siril artifacts and write previews, diffs, metrics, and context."""
    if not bool(getattr(pipeline.cfg, "review_bundle_enabled", True)):
        return {"status": "disabled", "stage": stage_key}
    if not pipeline.process_dir:
        return {"status": "unavailable", "stage": stage_key, "reason": "process_dir missing"}

    output_dir = Path(pipeline.process_dir) / "review_bundles" / stage_key
    output_dir.mkdir(parents=True, exist_ok=True)
    before_path = Path(pipeline.process_dir) / f"{before_stem}.fit"
    after_path = Path(pipeline.process_dir) / f"{after_stem}.fit"
    if not before_path.exists() or not after_path.exists():
        return {
            "status": "unavailable",
            "stage": stage_key,
            "reason": "source artifact missing",
            "before": str(before_path),
            "after": str(after_path),
        }

    try:
        pipeline.cmd_with_check("load", before_stem)
        before_data = pipeline.siril.get_image_pixeldata(preview=False)
        pipeline.cmd_with_check("load", after_stem)
        after_data = pipeline.siril.get_image_pixeldata(preview=False)
        if before_data is None or after_data is None:
            raise RuntimeError("Siril returned an empty review image")

        return create_image_review_bundle(
            before_data,
            after_data,
            output_dir=output_dir,
            stage_key=stage_key,
            source={
                "before_stem": before_stem,
                "after_stem": after_stem,
                "before_path": str(before_path),
                "after_path": str(after_path),
            },
            context=context,
            candidates=candidates,
            selected_candidate=selected_candidate,
        )
    finally:
        try:
            pipeline.cmd_with_check("load", after_stem)
        except (
            CommandError,
            DataError,
            SirilError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            pipeline.log.warn(f"review bundle restore failed ({after_stem}): {error}")
