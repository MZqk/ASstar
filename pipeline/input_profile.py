"""Evidence-driven linear/nonlinear/unknown input classification."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from models import InputProfile, InputState


_NONLINEAR_TOKENS = (
    "nonlinear",
    "non-linear",
    "stretched",
    "autostretch",
    "asinh",
    "generalized hyperbolic",
    "ghs stretch",
    "histogram transform",
    "midtone transfer",
)
_LINEAR_TOKENS = (
    "linear image",
    "linear master",
    "unstretched",
    "not stretched",
)
_ACQUISITION_KEYS = frozenset(
    {
        "BAYERPAT",
        "STACKCNT",
        "EXPTIME",
        "GAIN",
        "OFFSET",
        "CCD-TEMP",
        "SENSOR",
        "INSTRUME",
        "TELESCOP",
    }
)


def _truth_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(int(value))
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on", "linear"}:
        return True
    if normalized in {"0", "false", "no", "off", "nonlinear", "non-linear"}:
        return False
    return None


def read_fits_processing_text(path: Optional[Path]) -> str:
    """Read bounded FITS header cards, including HISTORY/COMMENT provenance."""
    if path is None or not path.is_file():
        return ""
    parts: List[str] = []
    try:
        with path.open("rb") as handle:
            card_count = 0
            while card_count < 512:
                block = handle.read(2880)
                if not block:
                    break
                for offset in range(0, len(block), 80):
                    card = block[offset : offset + 80].decode(
                        "ascii",
                        errors="ignore",
                    )
                    card_count += 1
                    key = card[:8].strip()
                    if key == "END":
                        return " ".join(parts).lower()
                    if key in {"HISTORY", "COMMENT"}:
                        value = card[8:].strip()
                    elif card[8:10] == "= ":
                        value = card[10:80].split("/", 1)[0].strip()
                    else:
                        value = ""
                    if value:
                        parts.append(f"{key} {value}")
                    if card_count >= 512:
                        break
    except OSError:
        return ""
    return " ".join(parts).lower()


def _normalized_luminance(image_data: Any) -> Optional[np.ndarray]:
    if image_data is None:
        return None
    try:
        source = np.asarray(image_data)
        if source.size == 0:
            return None
        arr = source.astype(np.float32, copy=False)
        if arr.ndim == 3:
            if arr.shape[0] in (3, 4):
                arr = np.mean(arr[:3], axis=0)
            elif arr.shape[-1] in (3, 4):
                arr = np.mean(arr[..., :3], axis=-1)
            else:
                arr = np.mean(arr, axis=0)
        elif arr.ndim > 3:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        if np.issubdtype(source.dtype, np.integer):
            scale = float(np.iinfo(source.dtype).max)
        else:
            peak = float(np.max(np.abs(finite)))
            if peak <= 1.5:
                scale = 1.0
            elif peak <= 255.0 * 1.05:
                scale = 255.0
            elif peak <= 65535.0 * 1.05:
                scale = 65535.0
            else:
                scale = max(peak, 1.0)
        normalized = np.clip(finite / max(scale, 1e-12), 0.0, 1.0)
        max_samples = 250_000
        if normalized.size > max_samples:
            stride = int(math.ceil(normalized.size / max_samples))
            normalized = normalized[::stride]
        return normalized
    except (TypeError, ValueError, OverflowError):
        return None


def _pixel_metrics(image_data: Any) -> Dict[str, Any]:
    luminance = _normalized_luminance(image_data)
    if luminance is None or luminance.size == 0:
        return {"available": False}
    p01, p50, p99 = np.percentile(luminance, (1.0, 50.0, 99.0))
    return {
        "available": True,
        "sample_count": int(luminance.size),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "black_ratio": float(np.mean(luminance <= 0.0005)),
        "highlight_ratio": float(np.mean(luminance >= 0.995)),
    }


def infer_input_profile(
    *,
    input_mode: str,
    source_path: Optional[Path],
    metadata: Optional[Mapping[str, Any]] = None,
    image_data: Any = None,
    trusted_provenance: Optional[Mapping[str, Any]] = None,
) -> InputProfile:
    """Resolve input state without trusting checkpoint filenames."""
    normalized_mode = str(input_mode or "unknown").strip().lower()
    evidence: List[Dict[str, Any]] = []
    conflicts: List[str] = []
    metadata_dict = {
        str(key).upper(): value
        for key, value in dict(metadata or {}).items()
        if not str(key).startswith("_")
    }
    processing_text = read_fits_processing_text(source_path)
    metrics = _pixel_metrics(image_data)

    if normalized_mode == "light_preprocess":
        evidence.append(
            {
                "kind": "runtime_provenance",
                "state": InputState.LINEAR.value,
                "confidence": 1.0,
                "detail": "calibrate/register/stack output generated in this run",
            }
        )
        return InputProfile(
            state=InputState.LINEAR,
            confidence=1.0,
            source="current_run_light_stack",
            input_mode=normalized_mode,
            evidence=evidence,
            pixel_metrics=metrics,
        )

    provenance = dict(trusted_provenance or {})
    if bool(provenance.get("verified")):
        provenance_state = str(provenance.get("state") or "").strip().lower()
        if provenance_state in {state.value for state in InputState}:
            evidence.append(
                {
                    "kind": "verified_manifest",
                    "state": provenance_state,
                    "confidence": 0.99,
                    "detail": str(
                        provenance.get("detail")
                        or "input hash matches processing manifest"
                    ),
                }
            )
            resolved = InputState(provenance_state)
            return InputProfile(
                state=resolved,
                confidence=0.99,
                source="verified_manifest",
                input_mode=normalized_mode,
                evidence=evidence,
                pixel_metrics=metrics,
            )

    explicit_linear: List[str] = []
    explicit_nonlinear: List[str] = []
    linear_flag = _truth_value(metadata_dict.get("LINEAR"))
    if linear_flag is True:
        explicit_linear.append("FITS LINEAR=true")
    elif linear_flag is False:
        explicit_nonlinear.append("FITS LINEAR=false")
    for key in ("STRETCHED", "NONLINEAR"):
        flag = _truth_value(metadata_dict.get(key))
        if flag is True:
            explicit_nonlinear.append(f"FITS {key}=true")
        elif flag is False and key == "STRETCHED":
            explicit_linear.append("FITS STRETCHED=false")

    stretch_value = str(metadata_dict.get("STRETCH", "") or "").strip().lower()
    if stretch_value:
        if stretch_value in {"none", "false", "0", "linear"}:
            explicit_linear.append(f"FITS STRETCH={stretch_value}")
        else:
            explicit_nonlinear.append(f"FITS STRETCH={stretch_value}")

    if any(token in processing_text for token in _NONLINEAR_TOKENS):
        explicit_nonlinear.append("FITS processing history contains stretch operation")
    if any(token in processing_text for token in _LINEAR_TOKENS):
        explicit_linear.append("FITS processing history declares linear/unstretched")

    acquisition_keys = sorted(_ACQUISITION_KEYS.intersection(metadata_dict))
    acquisition_linear = bool(
        "BAYERPAT" in acquisition_keys
        or "STACKCNT" in acquisition_keys
        or len(acquisition_keys) >= 3
    )
    nonlinear_pixels = bool(
        metrics.get("available")
        and (
            (
                float(metrics.get("p50", 0.0)) >= 0.075
                and float(metrics.get("p99", 0.0)) >= 0.30
            )
            or (
                float(metrics.get("p50", 0.0)) >= 0.035
                and float(metrics.get("black_ratio", 0.0)) >= 0.025
            )
        )
    )
    linear_pixels = bool(
        metrics.get("available")
        and float(metrics.get("p50", 1.0)) <= 0.030
        and float(metrics.get("p99", 1.0)) <= 0.55
        and float(metrics.get("highlight_ratio", 1.0)) <= 0.002
    )

    for detail in explicit_linear:
        evidence.append(
            {
                "kind": "fits_header",
                "state": InputState.LINEAR.value,
                "confidence": 0.96,
                "detail": detail,
            }
        )
    for detail in explicit_nonlinear:
        evidence.append(
            {
                "kind": "fits_header",
                "state": InputState.NONLINEAR.value,
                "confidence": 0.96,
                "detail": detail,
            }
        )
    if acquisition_linear:
        evidence.append(
            {
                "kind": "acquisition_metadata",
                "state": InputState.LINEAR.value,
                "confidence": 0.88,
                "detail": "raw/stack acquisition keys: " + ", ".join(acquisition_keys),
            }
        )
    if nonlinear_pixels:
        evidence.append(
            {
                "kind": "pixel_distribution",
                "state": InputState.NONLINEAR.value,
                "confidence": 0.80,
                "detail": "bright or clipped background distribution is consistent with a stretch",
            }
        )
    elif linear_pixels:
        evidence.append(
            {
                "kind": "pixel_distribution",
                "state": InputState.LINEAR.value,
                "confidence": 0.68,
                "detail": "low median/highlight distribution is consistent with linear data",
            }
        )

    if explicit_linear and explicit_nonlinear:
        conflicts.append("FITS header/history contains both linear and nonlinear claims")
    if explicit_linear and nonlinear_pixels:
        conflicts.append("explicit linear metadata conflicts with nonlinear pixel distribution")
    if explicit_nonlinear and acquisition_linear:
        conflicts.append("stretch history conflicts with retained raw acquisition metadata")
    if acquisition_linear and nonlinear_pixels and not explicit_linear:
        conflicts.append("acquisition metadata conflicts with nonlinear pixel distribution")

    if conflicts:
        return InputProfile(
            state=InputState.UNKNOWN,
            confidence=0.20,
            source="conflicting_evidence",
            input_mode=normalized_mode,
            evidence=evidence,
            conflicts=conflicts,
            pixel_metrics=metrics,
        )
    if explicit_nonlinear:
        return InputProfile(
            state=InputState.NONLINEAR,
            confidence=0.96,
            source="fits_processing_metadata",
            input_mode=normalized_mode,
            evidence=evidence,
            pixel_metrics=metrics,
        )
    if explicit_linear:
        return InputProfile(
            state=InputState.LINEAR,
            confidence=0.96,
            source="fits_processing_metadata",
            input_mode=normalized_mode,
            evidence=evidence,
            pixel_metrics=metrics,
        )
    if acquisition_linear:
        return InputProfile(
            state=InputState.LINEAR,
            confidence=0.88,
            source="acquisition_metadata",
            input_mode=normalized_mode,
            evidence=evidence,
            pixel_metrics=metrics,
        )
    if nonlinear_pixels:
        return InputProfile(
            state=InputState.NONLINEAR,
            confidence=0.80,
            source="pixel_distribution",
            input_mode=normalized_mode,
            evidence=evidence,
            pixel_metrics=metrics,
        )
    return InputProfile(
        state=InputState.UNKNOWN,
        confidence=0.35 if linear_pixels else 0.10,
        source="insufficient_evidence",
        input_mode=normalized_mode,
        evidence=evidence,
        pixel_metrics=metrics,
    )
