"""Canonical pixel-domain contract for Stage 7 target stretching."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


STAGE7_PIXEL_DOMAIN_SCHEMA = "starun.stage7-pixel-domain.v1"
STAGE7_FLOAT_DOMAIN_TOLERANCE = 2e-6


class Stage7PixelDomainError(ValueError):
    """Raised when Stage 7 pixels cannot be interpreted safely as 0..1."""


def canonicalize_stage7_pixels_01(
    image_data: Any,
    *,
    float_tolerance: float = STAGE7_FLOAT_DOMAIN_TOLERANCE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a float32 0..1 view plus auditable source-domain metadata.

    Unsigned integer pixels use their dtype full scale, never the observed
    image maximum. Floating-point pixels must already be in the 0..1 domain;
    only tiny numerical excursions are clipped. Signed, boolean, complex and
    object arrays are rejected because their scale/offset is ambiguous at this
    API boundary.
    """

    source = np.asarray(image_data)
    if source.size == 0:
        raise Stage7PixelDomainError("Stage7 image buffer is empty")

    source_dtype = source.dtype
    tolerance = float(float_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise Stage7PixelDomainError("Stage7 float-domain tolerance is invalid")

    source_min = float(np.min(source))
    source_max = float(np.max(source))
    normalization_applied = False
    normalization_scale = 1.0

    if np.issubdtype(source_dtype, np.unsignedinteger):
        normalization_scale = float(np.iinfo(source_dtype).max)
        if normalization_scale <= 0.0:
            raise Stage7PixelDomainError(
                f"Stage7 unsigned dtype has no usable full scale: {source_dtype}"
            )
        canonical = source.astype(np.float32) / normalization_scale
        normalization_applied = True
        source_kind = "unsigned_integer"
    elif np.issubdtype(source_dtype, np.floating):
        if not np.all(np.isfinite(source)):
            raise Stage7PixelDomainError(
                f"Stage7 floating pixels contain NaN or Inf: dtype={source_dtype}"
            )
        if source_min < -tolerance or source_max > 1.0 + tolerance:
            raise Stage7PixelDomainError(
                "Stage7 floating pixels are outside the declared 0..1 domain: "
                f"dtype={source_dtype}, min={source_min:.9g}, max={source_max:.9g}"
            )
        canonical = source.astype(np.float32, copy=False)
        source_kind = "floating_point"
    else:
        raise Stage7PixelDomainError(
            "Stage7 pixels require an unsigned integer or floating dtype: "
            f"got {source_dtype}"
        )

    canonical = np.clip(canonical, 0.0, 1.0).astype(np.float32, copy=False)
    if not np.all(np.isfinite(canonical)):
        raise Stage7PixelDomainError("Stage7 canonical pixels contain NaN or Inf")

    provenance: Dict[str, Any] = {
        "schema": STAGE7_PIXEL_DOMAIN_SCHEMA,
        "source_dtype": str(source_dtype),
        "source_kind": source_kind,
        "source_shape": [int(value) for value in source.shape],
        "source_min": source_min,
        "source_max": source_max,
        "normalization_applied": normalization_applied,
        "normalization_scale": normalization_scale,
        "canonical_dtype": "float32",
        "canonical_domain": "0..1",
        "canonical_min": float(np.min(canonical)),
        "canonical_max": float(np.max(canonical)),
        "float_tolerance": tolerance,
    }
    return canonical, provenance
