"""Service mixins for StarunPostProcessor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import plistlib
import re
import shutil
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

import cosmic_clarity
import plugin_runner
import sasp_runner
import scunet_denoise
import syqon_starless
import stage_contracts
import stage3_contract
import stage7_quality
import stage7_repair
import stage8_pixels
import outcome
from channel_semantics import channel_shape_dict, classify_channel_semantics
from dualband_palette import PALETTE_CHANNELS, resolve_palette_selection
from input_profile import infer_input_profile
import run_manifest
import task_plan
import task_workspace
from processing_parameters import (
    SPECS_BY_STAGE,
    apply_processing_parameters_to_config,
    default_processing_parameters,
    gate_profile_requires_review,
    normalize_processing_parameters,
    processing_gate_profile_audit,
)
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    format_feature_summary,
    measure_image_features,
    measure_quality_metrics,
)
from models import (
    ImageFeatures,
    InputProfile,
    PipelineStage,
    QualityMetrics,
    StageResult,
    TargetType,
)
from save_utils import save_stage_output, write_stage_json

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except ImportError:
    CommandError = RuntimeError
    DataError = RuntimeError
    SirilError = RuntimeError

try:
    from image_feature_analyzer import analyze_image as analyze_adaptive_image
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from target_profiler import build_target_profile
except (ImportError, RuntimeError):
    analyze_adaptive_image = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage7_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_target_profile = None

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "STARUN_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "STARUN_INPUT_MODE"
PROJECT_DEFAULT_ENV_RESOURCE_REL = Path("resources") / "default.env"
PROJECT_ENV_ALLOWED_KEYS = frozenset(
    {
        "STARUN_DEBUG_MODE",
        "STARUN_INPUT_MODE",
        "STARUN_OUTPUT_FORMAT",
        "STARUN_NETWORK_MODE",
        "STARUN_WORKFLOW_PLUGIN_PROBE",
        "STARUN_SPCC_ENABLE",
        "STARUN_GAIA_PHOTO_CATALOG",
        "STARUN_SPCC_DATABASE_DIR",
        "STARUN_STAGE4_PLATESOLVE_ENABLE",
        "STARUN_STAGE4_PLATESOLVE_FOCAL",
        "STARUN_STAGE4_PLATESOLVE_PIXELSIZE",
        "STARUN_STAGE4_PLATESOLVE_ORDER",
        "STARUN_STAGE4_PLATESOLVE_CATALOGS",
        "STARUN_STAGE4_PLATESOLVE_HEADER_RADIUS",
        "STARUN_STAGE4_AUTO_GEOMETRY_ENABLE",
        "STARUN_STAGE4_AUTO_GEOMETRY_CONFIDENCE_MIN",
        "STARUN_STAGE4_AUTO_GEOMETRY_SCALE_RESIDUAL_MAX",
        "STARUN_STAGE4_SPCC_TIMEOUT_SEC",
        "STARUN_STAGE4_SPCC_ONLINE_UNVERIFIED_TIMEOUT_SEC",
        "STARUN_STAGE4_SPCC_ONLINE_CIRCUIT_OPEN",
        "STARUN_STAGE4_SPCC_OSC_SENSOR",
        "STARUN_STAGE4_SPCC_OSC_FILTER",
        "STARUN_STAGE4_SPCC_WHITE_REF",
        "STARUN_STAGE4_SPCC_LIMITMAG",
        "STARUN_STAGE4_SPCC_NB_R_WAVELENGTH_NM",
        "STARUN_STAGE4_SPCC_NB_R_BANDWIDTH_NM",
        "STARUN_STAGE4_SPCC_NB_G_WAVELENGTH_NM",
        "STARUN_STAGE4_SPCC_NB_G_BANDWIDTH_NM",
        "STARUN_STAGE4_SPCC_NB_B_WAVELENGTH_NM",
        "STARUN_STAGE4_SPCC_NB_B_BANDWIDTH_NM",
        "STARUN_STAGE4_NBN_ENABLE",
        "STARUN_STAGE4_NBN_MAPPING_CONFIDENCE_MIN",
        "STARUN_STAGE4_NBN_STRENGTH",
        "STARUN_STAGE4_NBN_GAIN_LIMIT",
        "STARUN_STAGE4_NBN_LINE_RATIO_DRIFT_MAX",
        "STARUN_GAIA_ASTRO_CATALOG",
        "STARUN_STAGE4_FILTER_HINT",
        "STARUN_STAGE4_OFFLINE_FALLBACK_MODE",
        "STARUN_STAGE4_AUTO_REFERENCE_GLOBAL_WHITE_ENABLE",
        "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_SAMPLE_TARGET",
        "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_SAMPLE_MIN",
        "STARUN_STAGE4_AUTO_REFERENCE_HOLDOUT_RATIO",
        "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_ERROR_MIN",
        "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_IMPROVEMENT_MIN",
        "STARUN_STAGE4_AUTO_REFERENCE_STAR_MIN_OBJECTS",
        "STARUN_STAGE4_AUTO_REFERENCE_STAR_RATIO_MAD_MAX",
        "STARUN_STAGE4_AUTO_REFERENCE_STAR_SATURATION_RATIO_MAX",
        "STARUN_STAGE4_AUTO_REFERENCE_GAIN_LIMIT",
        "STARUN_STAGE4_AUTO_REFERENCE_STAR_IMPROVEMENT_MIN",
        "STARUN_STAGE4_AUTO_REFERENCE_HIGHLIGHT_CLIP_GROWTH_MAX",
        "STARUN_STAGE4_AUTO_REFERENCE_BLACK_CLIP_GROWTH_MAX",
        "STARUN_STAGE4_AUTO_REFERENCE_GRADIENT_GROWTH_MAX",
        "STARUN_STAGE4_AUTO_REFERENCE_TEXTURE_GROWTH_MAX",
        "STARUN_STAGE4_AUTO_REFERENCE_TARGET_CHROMA_DRIFT_MAX",
        "STARUN_STAGE4_PCC_TIMEOUT_SEC",
        "STARUN_STAGE4_LOCAL_STAR_WB_ENABLE",
        "STARUN_STAGE4_LOCAL_STAR_WB_MIN_PIXELS",
        "STARUN_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT",
        "STARUN_STAGE4_LOCAL_STAR_MASK_RADIUS",
        "STARUN_STAGE4_LOCAL_STAR_MASK_COVERAGE_MAX",
        "STARUN_ABERRATION_API_ENABLE",
        "STARUN_ABERRATION_PROVIDER",
        "STARUN_OPTIONAL_COLOR_TRANSFORM",
        "STARUN_STAGE8_DUALBAND_PALETTE_ENABLE",
        "STARUN_STAGE8_DUALBAND_PALETTE_QUALITY_WARNING_TOLERANCE",
        "STARUN_DENOISE_ENABLE",
        "STARUN_DENOISE_FORCE",
        "STARUN_DENOISE_MOD",
        "STARUN_STAGE5_MULTISCALE_DENOISE_ENABLE",
        "STARUN_STAGE5_MULTISCALE_DENOISE_STRENGTH",
        "STARUN_STAGE5_MULTISCALE_DETAIL_RETENTION_MIN",
        "STARUN_STAGE5_MULTISCALE_NOISE_REDUCTION_MIN",
        "STARUN_STAGE5_DENOISE_CHROMA_NOISE_GROWTH_MAX",
        "STARUN_STAGE5_DECONV_ENABLE",
        "STARUN_STAGE5_GRAXPERT_DECONV_ENABLE",
        "STARUN_STAGE5_RL_MAXSTARS",
        "STARUN_STAGE5_RL_PSF_KS",
        "STARUN_STAGE5_RL_ITERS",
        "STARUN_STAGE5_RL_ALPHA",
        "STARUN_STAGE5_RL_GDSTEP",
        "STARUN_STAGE5_RL_STOP",
        "STARUN_STAGE5_GRAXPERT_DECONV_STRENGTH",
        "STARUN_GRAXPERT_OBJECT_MODEL_PATH",
        "STARUN_GRAXPERT_GPU",
        "STARUN_STAGE7_QUALITY_RETRY_MAX",
        "STARUN_STAGE7_LARGE_GALAXY_HALO_RESIDUE_SCORE_MAX",
        "STARUN_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX",
        "STARUN_STAGE7_GALAXY_ROI_HALO_GATE_ENABLE",
        "STARUN_STAGE7_GALAXY_CORE_PRESERVATION_RATIO_MIN",
        "STARUN_STAGE7_GALAXY_CORE_CONTRAST_RATIO_MIN",
        "STARUN_STAGE7_STARLESS_REPAIR_STRENGTH",
        "STARUN_STAGE7_STARLESS_HALO_REPAIR_STRENGTH",
        "STARUN_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH",
        "STARUN_STAGE7_STRETCH_CHROMA_NOISE_SCORE_MAX",
        "STARUN_STAGE7_STRETCH_BACKGROUND_MOTTLING_SCORE_MAX",
        "STARUN_STAGE7_STRETCH_CHROMA_LOAD_GROWTH_MAX",
        "STARUN_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_MAX",
        "STARUN_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_TOLERANCE",
        "STARUN_STAGE7_UNCALIBRATED_BACKGROUND_CHROMA_LOAD_REVIEW_MAX",
        "STARUN_STAGE7_CHROMA_RESCUE_ENABLE",
        "STARUN_STAGE7_PREVIEW_TARGET_P50_MIN_RATIO",
        "STARUN_STAGE7_PREVIEW_TARGET_P50_MAX_RATIO",
        "STARUN_STAGE7_STRETCH_FEEDBACK_RETRY_MAX",
        "STARUN_STAGE7_STARLESS_STRUCTURE_GATE_ENABLE",
        "STARUN_STAGE7_STARLESS_MASKED_RANK_DRIFT_P95_MAX",
        "STARUN_STAGE7_STARLESS_HALO_DETAIL_GROWTH_RATIO_MAX",
        "STARUN_STAGE7_STARLESS_HALO_DETAIL_DELTA_MIN",
        "STARUN_STAGE7_QUANTILE_FALLBACK_ENABLE",
        "STARUN_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE",
        "STARUN_STAGE7_STARLESS_REPAIR_CHROMA_REDUCTION_MIN",
        "STARUN_STAGE7_STARLESS_REPAIR_CHROMA_DELTA_MIN",
        "STARUN_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX",
        "STARUN_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR",
        "STARUN_STAGE8_LOCAL_ADJUSTMENT_ENGINE_ENABLE",
        "STARUN_STAGE8_LOCAL_CURVE_OPACITY",
        "STARUN_STAGE8_LIMITED_SATURATION_MAX",
        "STARUN_STAGE8_LIMITED_CORE_EXCLUSION_EXPAND",
        "STARUN_STAGE8_LIMITED_HALO_TEXTURE_GROWTH_MAX",
        "STARUN_STAGE8_LIMITED_HALO_TEXTURE_DELTA_MAX",
        "STARUN_STAGE8_DUALBAND_PALETTE_STRENGTH",
        "STARUN_STAGE8_DUALBAND_PALETTE_LUMA_DRIFT_MAX",
        "STARUN_STAGE8_DUALBAND_PALETTE_CLIP_GROWTH_MAX",
        "STARUN_STAGE9_STARMASK_STRETCH_ENABLE",
        "STARUN_STAGE9_STARMASK_ADAPTIVE_STRETCH_ENABLE",
        "STARUN_STAGE9_COMPACT_STARMASK_ENABLE",
        "STARUN_STAGE9_STARMASK_PRE_STRETCH_COMPACT_ENABLE",
        "STARUN_STAGE9_STAR_COLOR_REPAIR_ENABLE",
        "STARUN_STAGE9_STAR_COLOR_REPAIR_STRENGTH",
        "STARUN_STAGE9_STAR_COLOR_SUPPORT_RATIO_MAX",
        "STARUN_STAGE9_STAR_COLOR_IMPROVEMENT_MIN",
        "STARUN_STAGE9_STAR_COLOR_POST_CHROMA_ERROR_MAX",
        "STARUN_STAGE9_STAR_COLOR_POST_VALIDATION_ENABLE",
        "STARUN_STAGE9_SOURCE_STAR_DETAIL_PERCENTILE",
        "STARUN_STAGE9_SOURCE_COMPONENT_DENSITY_MAX",
        "STARUN_STAGE9_SOURCE_SINGLE_PIXEL_RATIO_MAX",
        "STARUN_STAGE9_STARMASK_ASINH_STRETCH",
        "STARUN_STAGE9_STARMASK_ASINH_OFFSET",
        "STARUN_STAGE9_STARMASK_ASINH_STRETCH_MAX",
        "STARUN_STAGE9_STARMASK_FAINT_TARGET",
        "STARUN_STAGE9_STARMASK_MID_TARGET",
        "STARUN_STAGE9_STARMASK_BRIGHT_TARGET",
        "STARUN_STAGE9_STARMASK_PEAK_TARGET",
        "STARUN_STAGE9_STARMASK_CHROMA_REGULARIZATION_ENABLE",
        "STARUN_STAGE9_STARMASK_FAINT_CHROMA_MAX",
        "STARUN_STAGE9_STARMASK_BRIGHT_CHROMA_MAX",
        "STARUN_STAGE9_STARMASK_PREDICTED_CHANGE_RATIO_MAX",
        "STARUN_STAGE9_STAR_REFERENCE_SIGMA",
        "STARUN_STAGE9_COMPACT_WEAK_STAR_RETENTION_MIN",
        "STARUN_STAGE9_MIXED_STAR_PEAK_RATIO_MIN",
        "STARUN_STAGE9_MIXED_STAR_WEAK_COUNT_MIN",
        "STARUN_STAGE9_MIXED_STAR_BRIGHT_COUNT_MIN",
        "STARUN_STAGE7_TARGET_LOCAL_METRICS_ENABLE",
        "STARUN_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX",
        "STARUN_STAGE7_LOCAL_FAINT_SNR_MIN",
        "STARUN_STAGE7_LOCAL_DARK_SEPARATION_MIN",
        "STARUN_STAGE9_QUALITY_GATE_ENABLE",
        "STARUN_STAGE9_HIGHLIGHT_CLIP_RATIO_MAX",
        "STARUN_STAGE9_HIGHLIGHT_CLIP_GROWTH_MAX",
        "STARUN_STAGE9_BRIGHT_PIXEL_GROWTH_MAX",
        "STARUN_STAGE9_BACKGROUND_LIFT_MAX",
        "STARUN_STAGE9_BACKGROUND_MOTTLING_GROWTH_MAX",
        "STARUN_STAGE9_MOTTLING_EXEMPTION_CHANGED_PIXEL_RATIO_MAX",
        "STARUN_STAGE9_CHANGED_PIXEL_RATIO_MAX",
        "STARUN_STAGE9_DARKENING_RATIO_MAX",
        "STARUN_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN",
        "STARUN_STAGE9_STAR_RECOVERY_RATIO_MIN",
        "STARUN_STAGE9_WEAK_STAR_SCREEN_INTENSITY_MIN",
        "STARUN_STAGE9_STAR_SUPPORT_RATIO_MAX",
        "STARUN_STAGE9_UNMATCHED_CHANGED_RATIO_MAX",
        "STARUN_STAGE9_CHROMATIC_ADDITION_PEAK_MIN",
        "STARUN_STAGE9_CHROMATIC_ADDITION_SATURATION_MIN",
        "STARUN_STAGE9_CHROMATIC_ADDITION_RATIO_MAX",
        "STARUN_STAGE9_STAR_POSITIVE_DELTA_WINDOW_RECOVERY_RATIO_MIN",
        "STARUN_STAGE9_STAR_WING_RECOVERY_RATIO_MIN",
        "STARUN_STAGE9_RESIDUAL_DARK_HOLE_RATIO_MAX",
        "STARUN_STAGE9_HOLLOW_STRUCTURE_DELTA_MIN",
        "STARUN_STAGE9_NEW_HOLLOW_STRUCTURE_AREA_MAX",
        "STARUN_FORCE_REVIEW_ONLY_OUTPUT",
        "STARUN_STAGE10_MANAGED_OUTPUT_ENABLE",
        "STARUN_STAGE10_FINAL_DENOISE_STRENGTH",
        "STARUN_STAGE10_STAR_PROTECTION_COVERAGE_MAX",
        "STARUN_COSMIC_CLASSIC_ENABLE",
        "STARUN_COSMIC_CLARITY_EXECUTABLE",
        "STARUN_COSMIC_CLASSIC_GPU",
        "STARUN_COSMIC_NATIVE_GPU",
        "STARUN_SYQON_TIMEOUT_SEC",
        "STARUN_SYQON_MODEL_DIR",
        "STARUN_SIRILPY_TIMEOUT_SEC",
        "STARUN_SIRIL_PLUGIN_DIR",
        "SIRIL_PYTHON_CLI",
        "STARUN_SIRIL_PYTHON_CLI",
    }
)
INPUT_MODE_AUTO = "auto"
INPUT_MODE_STAGE1_PREPARED_RESUME = "stage1_prepared_resume"
INPUT_MODE_LINEAR_RESUME = "stage5_linear_resume"
INPUT_MODE_STAGE2_CORRECTED_RESUME = "stage2_corrected_resume"
ENV_TASK_RUN_MANIFEST_KEY = "STARUN_TASK_RUN_MANIFEST"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)


def _safe_output_token(value: Any, *, fallback: str = "") -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return token or fallback


_UNRESOLVED_METADATA_TOKEN = re.compile(
    r"\$[^$]*\$|%(?:[-+0-9.# ]*)[sdf]",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_METADATA_VALUES = frozenset(
    {"unknown", "null", "none", "n/a", "na", "unset", "undefined"}
)


def _resolved_metadata_text(value: Any) -> str:
    text = str(value or "").strip()
    if (
        not text
        or text.lower() in _PLACEHOLDER_METADATA_VALUES
        or _UNRESOLVED_METADATA_TOKEN.search(text)
    ):
        return ""
    return text


def _validated_output_metadata(
    metadata: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    validated: Dict[str, Any] = {}
    invalid: List[str] = []
    object_text = _resolved_metadata_text(metadata.get("OBJECT"))
    if object_text:
        validated["OBJECT"] = object_text
    else:
        invalid.append("OBJECT")
    for key, integer_only in (("STACKCNT", True), ("EXPTIME", False)):
        try:
            numeric = float(metadata.get(key))
        except (TypeError, ValueError):
            numeric = float("nan")
        if math.isfinite(numeric) and numeric > 0.0:
            validated[key] = int(numeric) if integer_only else numeric
        else:
            invalid.append(key)
    date_text = _resolved_metadata_text(metadata.get("DATE-OBS"))
    date_digits = re.sub(r"[^0-9]+", "", date_text)
    if len(date_digits) >= 8:
        validated["DATE-OBS"] = date_text
    else:
        invalid.append("DATE-OBS")
    return validated, invalid


def _partial_metadata_output_basename(
    metadata: Dict[str, Any],
    *,
    linear_resume: bool,
    identity_fallback: str = "",
) -> str:
    """Build a useful literal filename when Siril's full template cannot resolve."""
    object_token = _safe_output_token(
        metadata.get("OBJECT") or identity_fallback
    )
    date_digits = re.sub(r"[^0-9]+", "", str(metadata.get("DATE-OBS") or ""))
    if not object_token:
        return ""

    parts = [object_token]
    try:
        stack_count = int(float(metadata.get("STACKCNT")))
    except (TypeError, ValueError):
        stack_count = 0
    if stack_count > 0:
        parts.append(f"{stack_count}x")

    try:
        exposure = float(metadata.get("EXPTIME"))
    except (TypeError, ValueError):
        exposure = 0.0
    if math.isfinite(exposure) and exposure > 0.0:
        exposure_token = f"{exposure:g}".replace(".", "p")
        parts.append(f"{exposure_token}sec")

    if len(date_digits) >= 8:
        date_token = date_digits[:8]
        if len(date_digits) >= 14:
            date_token += f"_{date_digits[8:14]}"
        parts.append(date_token)
    parts.append("processed")
    base = "_".join(parts)
    return f"{base}_linear" if linear_resume else base


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


_FITS_BLOCK_BYTES = 2880
_FITS_CARD_BYTES = 80
_FITS_LAYOUT_KEYS = frozenset(
    {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "PCOUNT",
        "GCOUNT",
        "GROUPS",
        "BSCALE",
        "BZERO",
        "BLANK",
        "THEAP",
        "TFIELDS",
        "ZIMAGE",
        "ZBITPIX",
        "ZNAXIS",
        "ZCMPTYPE",
        "ZSCALE",
        "ZZERO",
    }
)
_FITS_INDEXED_LAYOUT_KEY = re.compile(
    r"^(?:NAXIS|ZNAXIS|TFORM|TDIM|TSCAL|TZERO|TNULL|ZNAME|ZVAL)\d+$"
)


def _fits_card_value_text(card: str) -> Optional[str]:
    if len(card) < _FITS_CARD_BYTES or card[8:10] != "= ":
        return None
    value = card[10:_FITS_CARD_BYTES].lstrip()
    if not value.startswith("'"):
        return value.split("/", 1)[0].strip()

    chars: List[str] = []
    index = 1
    while index < len(value):
        if value[index] == "'":
            if index + 1 < len(value) and value[index + 1] == "'":
                chars.append("'")
                index += 2
                continue
            return "".join(chars).strip()
        chars.append(value[index])
        index += 1
    return "".join(chars).strip()


def _read_fits_stage_fingerprint(path: Path) -> Dict[str, Any]:
    """Hash FITS data/layout separately from mutable header cards."""
    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError("empty FITS file")

    data_digest = hashlib.sha256()
    layout_digest = hashlib.sha256()
    header_digest = hashlib.sha256()
    container_digest = hashlib.sha256()
    hdu_count = 0
    logical_data_bytes = 0

    with path.open("rb") as handle:
        while handle.tell() < file_size:
            header_blocks: List[bytes] = []
            header_cards: List[str] = []
            end_found = False
            for _ in range(4096):
                block = handle.read(_FITS_BLOCK_BYTES)
                if len(block) != _FITS_BLOCK_BYTES:
                    raise ValueError("truncated FITS header block")
                header_blocks.append(block)
                container_digest.update(block)
                for offset in range(0, _FITS_BLOCK_BYTES, _FITS_CARD_BYTES):
                    raw_card = block[offset : offset + _FITS_CARD_BYTES]
                    card = raw_card.decode("ascii", errors="replace")
                    header_cards.append(card)
                    if card[:8].strip().upper() == "END":
                        end_found = True
                        break
                if end_found:
                    break
            if not end_found:
                raise ValueError("FITS header END card not found")

            first_keyword = header_cards[0][:8].strip().upper()
            expected_keyword = "SIMPLE" if hdu_count == 0 else "XTENSION"
            if first_keyword != expected_keyword:
                raise ValueError(
                    f"invalid FITS HDU start: expected {expected_keyword}, "
                    f"got {first_keyword or 'blank'}"
                )

            header_bytes = b"".join(header_blocks)
            header_digest.update(hdu_count.to_bytes(4, "big"))
            header_digest.update(len(header_bytes).to_bytes(8, "big"))
            header_digest.update(header_bytes)

            values: Dict[str, str] = {}
            layout: Dict[str, str] = {}
            for card in header_cards:
                keyword = card[:8].strip().upper()
                if keyword == "END":
                    break
                value = _fits_card_value_text(card)
                if value is None or not keyword:
                    continue
                values[keyword] = value
                if (
                    keyword in _FITS_LAYOUT_KEYS
                    or _FITS_INDEXED_LAYOUT_KEY.fullmatch(keyword)
                ):
                    layout[keyword] = value

            try:
                bitpix = int(values["BITPIX"])
                naxis = int(values["NAXIS"])
                axes = [int(values[f"NAXIS{index}"]) for index in range(1, naxis + 1)]
                pcount = int(values.get("PCOUNT", "0"))
                gcount = int(values.get("GCOUNT", "1"))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid FITS data layout: {error}") from error
            if bitpix not in {8, 16, 32, 64, -32, -64}:
                raise ValueError(f"unsupported FITS BITPIX={bitpix}")
            if naxis < 0 or any(axis < 0 for axis in axes) or pcount < 0 or gcount < 1:
                raise ValueError("negative or invalid FITS data dimensions")

            random_groups = values.get("GROUPS", "F").strip().upper() == "T"
            if random_groups and axes and axes[0] == 0:
                axis_elements = math.prod(axes[1:])
            else:
                axis_elements = math.prod(axes) if axes else 0
            data_size = (abs(bitpix) // 8) * gcount * (pcount + axis_elements)
            if data_size < 0 or handle.tell() + data_size > file_size:
                raise ValueError("FITS data section exceeds file size")

            layout_payload = json.dumps(
                {
                    "hdu": hdu_count,
                    "data_size": data_size,
                    "layout": layout,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            layout_digest.update(len(layout_payload).to_bytes(8, "big"))
            layout_digest.update(layout_payload)
            data_digest.update(hdu_count.to_bytes(4, "big"))
            data_digest.update(data_size.to_bytes(8, "big"))

            remaining = data_size
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("truncated FITS data section")
                data_digest.update(chunk)
                container_digest.update(chunk)
                remaining -= len(chunk)
            logical_data_bytes += data_size

            padding_size = (-data_size) % _FITS_BLOCK_BYTES
            if padding_size:
                padding = handle.read(padding_size)
                if len(padding) != padding_size:
                    raise ValueError("truncated FITS data padding")
                container_digest.update(padding)
            hdu_count += 1

    if hdu_count == 0:
        raise ValueError("FITS contains no HDU")
    return {
        "data_sha256": data_digest.hexdigest(),
        "layout_sha256": layout_digest.hexdigest(),
        "header_sha256": header_digest.hexdigest(),
        "container_sha256": container_digest.hexdigest(),
        "hdu_count": hdu_count,
        "logical_data_bytes": logical_data_bytes,
    }


class ProcessorRuntimeMixin:
    def _ensure_review_registry(self) -> Dict[Tuple[int, str], Dict[str, Any]]:
        registry = getattr(self, "_review_requirements", None)
        if not isinstance(registry, dict):
            registry = {}
            self._review_requirements = registry
        return registry

    def _require_review(
        self,
        stage: int,
        code: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        requirement = outcome.normalize_review_requirement(
            {"stage": stage, "code": code, "details": dict(details or {})}
        )
        self._ensure_review_registry()[
            (requirement["stage"], requirement["code"])
        ] = requirement
        for result in reversed(getattr(self, "results", []) or []):
            match = re.match(
                r"^阶段\s+(\d+)\s*:",
                str(getattr(result, "name", "")).strip(),
            )
            if match and int(match.group(1)) == int(requirement["stage"]):
                reasons = getattr(result, "review_reasons", None)
                if isinstance(reasons, list) and requirement["code"] not in reasons:
                    reasons.append(str(requirement["code"]))
                break
        return copy.deepcopy(requirement)

    def _clear_stage_reviews(self, stage: int) -> None:
        stage_number = int(stage)
        registry = self._ensure_review_registry()
        self._review_requirements = {
            key: value for key, value in registry.items() if key[0] != stage_number
        }

    def _stage_review_reasons(self, stage: int) -> List[str]:
        stage_number = int(stage)
        return [
            str(value["code"])
            for key, value in self._ensure_review_registry().items()
            if key[0] == stage_number
        ]

    def _review_requirements_payload(
        self,
        *,
        through_stage: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        limit = int(through_stage) if through_stage is not None else 10
        return [
            copy.deepcopy(value)
            for key, value in sorted(self._ensure_review_registry().items())
            if key[0] <= limit
        ]

    def _stage_failure_action(self, stage_number: int) -> str:
        value = str(
            getattr(
                self.cfg,
                f"stage{int(stage_number)}_failure_action",
                "auto_fallback",
            )
            or "auto_fallback"
        ).strip().lower()
        if value not in {"auto_fallback", "preserve_review", "stop"}:
            return "auto_fallback"
        return value

    def _record_stage_policy_event(
        self,
        stage_number: int,
        *,
        event: str,
        reason: str,
        source: str = "runtime",
    ) -> Dict[str, Any]:
        record = {
            "stage": int(stage_number),
            "failure_action": self._stage_failure_action(stage_number),
            "event": str(event),
            "reason": str(reason),
            "source": str(source),
        }
        events = getattr(self, "_stage_policy_events", None)
        if not isinstance(events, list):
            events = []
            self._stage_policy_events = events
        events.append(record)
        self.log.info(
            "[StagePolicy] "
            f"stage={stage_number} action={record['failure_action']} "
            f"event={record['event']} reason={record['reason']}"
        )
        return record

    def _handle_stage_decisive_failure(
        self,
        stage_number: int,
        reason: str,
        *,
        source: str = "quality_gate",
    ) -> str:
        """Apply the user-selected fail-closed policy after diagnostics exist."""

        action = self._stage_failure_action(stage_number)
        self._record_stage_policy_event(
            stage_number,
            event="decisive_failure",
            reason=reason,
            source=source,
        )
        if action == "preserve_review":
            self._require_review(
                int(stage_number),
                "failure_policy_preserve_review",
                {"reason": str(reason), "source": str(source)},
            )
            if int(stage_number) == 2:
                self._stage2_view_review_required = True
            elif int(stage_number) == 3:
                self._background_review_required = True
            return action
        if action == "stop":
            raise RuntimeError(
                f"Stage {stage_number} 用户严格停止：{reason}"
            )
        return action

    def _enforce_stage_failure_action(self, stage_number: int) -> None:
        """Terminate after a stage has written its failed result and diagnostics."""

        prefix = f"阶段 {int(stage_number)}:"
        result = next(
            (
                item
                for item in reversed(getattr(self, "results", []))
                if str(getattr(item, "name", "")).startswith(prefix)
            ),
            None,
        )
        if result is None or str(getattr(result, "status", "")) != "failed":
            return
        self._handle_stage_decisive_failure(
            stage_number,
            str(getattr(result, "reason_code", "") or getattr(result, "message", "") or "stage_failed"),
            source="stage_result",
        )

    def _processing_software_identity(self) -> Dict[str, Any]:
        """Return a packaged/dev build identity with a Stage 3 source hash."""
        module_path = Path(__file__).resolve()
        app_version = os.getenv("STARUN_APP_VERSION", "").strip() or "dev"
        app_build: Optional[str] = None
        app_identity_source = "environment" if app_version != "dev" else "development"
        info_candidates = [
            module_path.parents[2] / "Info.plist",
            module_path.parents[1] / "Info.plist",
        ]
        for info_path in info_candidates:
            if not info_path.is_file():
                continue
            try:
                with info_path.open("rb") as handle:
                    info = plistlib.load(handle)
                app_version = str(
                    info.get("CFBundleShortVersionString") or app_version
                )
                app_build = str(info.get("CFBundleVersion") or "") or None
                app_identity_source = "Info.plist"
                break
            except (OSError, TypeError, ValueError):
                continue

        source_files = (
            module_path.parent / "stage3_contract.py",
            module_path.parent / "background_sampling.py",
            module_path.parent / "stages" / "stage3_background_extraction.py",
        )
        source_records: Dict[str, Optional[str]] = {}
        aggregate = hashlib.sha256()
        aggregate_count = 0
        for path in source_files:
            relative_name = path.relative_to(module_path.parent).as_posix()
            digest = run_manifest.sha256_file(path)
            source_records[relative_name] = digest
            if digest:
                aggregate.update(relative_name.encode("utf-8"))
                aggregate.update(b"\0")
                aggregate.update(digest.encode("ascii"))
                aggregate.update(b"\0")
                aggregate_count += 1

        stage3_identity = stage3_contract.stage3_static_contract_manifest()
        stage3_identity.update(
            {
                "source_sha256": (
                    aggregate.hexdigest() if aggregate_count else None
                ),
                "source_files": source_records,
            }
        )
        return {
            "schema": "starun.software-identity.v1",
            "app": {
                "version": app_version,
                "build": app_build,
                "identity_source": app_identity_source,
            },
            "pipeline_contract": {
                "schema": stage_contracts.PIPELINE_CONTRACT_SCHEMA,
                "version": stage_contracts.PIPELINE_CONTRACT_VERSION,
            },
            "stage_algorithms": {
                "stage3_background": stage3_identity,
            },
        }


    def _project_env_candidates(self) -> List[Path]:
        module_project_root = Path(__file__).resolve().parents[1]
        return [
            module_project_root / PROJECT_DEFAULT_ENV_RESOURCE_REL,
        ]


    def _parse_project_env_file(self, path: Path) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return parsed

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key not in PROJECT_ENV_ALLOWED_KEYS:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            elif value and value[0] in {"'", '"'}:
                continue
            else:
                hash_with_space = value.find(" #")
                if hash_with_space >= 0:
                    value = value[:hash_with_space].rstrip()
            parsed[key] = value
        return parsed


    def _load_project_env_defaults(self) -> None:
        merged: Dict[str, str] = {}
        for path in self._project_env_candidates():
            if not path.is_file():
                continue
            merged.update(self._parse_project_env_file(path))
        if not hasattr(self, "_project_env_explicit_keys"):
            self._project_env_explicit_keys = frozenset(
                key for key in merged if key in os.environ
            )
        for key, value in merged.items():
            os.environ.setdefault(key, value)


    def _result_output_basename(self) -> str:
        linear_resume = self._stage1_input_mode == "linear_resume"
        fallback_base = "result_processed_linear" if linear_resume else "result_processed"
        fallback_fit_base = "result_final_linear" if linear_resume else "result_final"
        metadata = self._read_fits_header_metadata(
            "stage10_final",
            "stage9_remixed",
            "stage2_corrected",
            getattr(self, "source_file", None),
        )
        validated_metadata, invalid_keys = _validated_output_metadata(metadata)
        identity_fallback = ""
        profile = getattr(self, "target_profile", {}) or {}
        if isinstance(profile, dict):
            primary = profile.get("primary_target")
            primary = primary if isinstance(primary, dict) else {}
            try:
                primary_confidence = float(
                    primary.get("confidence", profile.get("target_confidence", 0.0))
                    or 0.0
                )
            except (TypeError, ValueError):
                primary_confidence = 0.0
            if primary_confidence >= 0.90:
                identity_fallback = _resolved_metadata_text(primary.get("name"))
            if not identity_fallback:
                try:
                    profile_confidence = float(
                        profile.get("target_confidence", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    profile_confidence = 0.0
                if profile_confidence >= 0.90:
                    identity_fallback = _resolved_metadata_text(
                        profile.get("target_name_guess")
                    )
        if not identity_fallback:
            source_file = getattr(self, "source_file", None)
            if source_file:
                identity_fallback = _resolved_metadata_text(Path(source_file).stem)

        if invalid_keys:
            partial_base = _partial_metadata_output_basename(
                validated_metadata,
                linear_resume=linear_resume,
                identity_fallback=identity_fallback,
            )
            if partial_base:
                base_filename = partial_base
                fit_base_filename = partial_base + "_final"
                self.log.warn(
                    "输出命名所需 FITS 头不完整，使用已有目标元数据生成安全名称，"
                    "避免未解析占位符和通用结果名覆盖: "
                    + ", ".join(invalid_keys)
                )
            else:
                base_filename = fallback_base
                fit_base_filename = fallback_fit_base
                self.log.warn(
                    "输出命名所需 FITS 头缺失，使用安全回退名，避免输出 "
                    "$OBJECT/$STACKCNT 等未解析占位符: "
                    + ", ".join(invalid_keys)
                )
        else:
            base_filename = RESULT_BASENAME_TEMPLATE
            if linear_resume:
                base_filename += "_linear"
            fit_base_filename = base_filename + "_final"
        self.main_output_basename_template = base_filename
        self.main_output_fit_basename_template = fit_base_filename
        return base_filename


    def _parse_env_bool(self, raw_value: str, fallback: bool) -> bool:
        value = raw_value.strip().lower()
        if value in ENV_TRUE_VALUES:
            return True
        if value in ENV_FALSE_VALUES:
            return False
        return fallback


    def _sync_logger_level(self):
        self.log.min_level = self.log._LEVELS.get(
            "DEBUG" if self.cfg.debug_mode else "INFO",
            1,
        )


    def _format_debug_quality_metrics_line(
        self,
        stem: str,
        metrics: QualityMetrics,
        features: ImageFeatures,
    ) -> str:
        return (
            "[STAGE_QUALITY_METRICS] "
            "schema=starun.stage_quality.v1 "
            f"stem={stem} "
            f"bg_median={metrics.bg_median:.6f} "
            f"black_pixel_ratio={metrics.black_pixel_ratio:.6f} "
            f"highlight_clip_ratio={metrics.highlight_clip_ratio:.6f} "
            f"star_density={metrics.star_density:.8f} "
            f"median_star_size={metrics.median_star_size:.6f} "
            f"star_coverage_ratio={metrics.star_coverage_ratio:.6f} "
            f"star_energy_ratio={metrics.star_energy_ratio:.6f} "
            f"saturation_median={metrics.saturation_median:.6f} "
            f"saturation_p95={metrics.saturation_p95:.6f} "
            f"microcontrast={metrics.microcontrast:.6f} "
            f"blue_excess={metrics.blue_excess:.6f} "
            f"edge_black_ratio={features.edge_black_ratio:.6f} "
            f"global_dark_ratio={features.global_dark_ratio:.6f} "
            f"object_area_ratio={features.object_area_ratio:.6f} "
            f"diffuse_ratio={features.diffuse_ratio:.6f} "
            f"core_brightness_ratio={features.core_brightness_ratio:.6f}"
        )


    def _write_debug_quality_metrics(self, stem: str) -> None:
        if not self.cfg.debug_mode:
            return

        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            metrics = measure_quality_metrics(image_data)
            features = measure_image_features(image_data)
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.warn(f"阶段质量指标采集失败 ({stem}): {e}")
            return

        self._debug_quality_metric_index = (
            int(getattr(self, "_debug_quality_metric_index", 0)) + 1
        )
        payload = {
            "schema": "starun.stage_quality.v1",
            "sequence": self._debug_quality_metric_index,
            "stem": stem,
            "file": f"{stem}.fit",
            "metrics": asdict(metrics),
            "features": asdict(features),
        }

        self.log.info(self._format_debug_quality_metrics_line(stem, metrics, features))
        if not self.process_dir:
            return

        try:
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            (self.process_dir / f"{stem}_quality_metrics.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            with (self.process_dir / "stage_quality_metrics.jsonl").open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(text + "\n")
        except OSError as e:
            self.log.warn(f"写入阶段质量指标失败 ({stem}): {e}")


    def _save_stage_output(self, stem: str) -> bool:
        saved = save_stage_output(self.cmd_with_check, self.log, stem)
        if saved:
            self._write_debug_quality_metrics(stem)
        return saved


    def _sha256_file(self, path: Path) -> Optional[str]:
        if not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None


    def _fits_stage_fingerprint(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists() or not path.is_file():
            return None
        try:
            return _read_fits_stage_fingerprint(path)
        except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
            self.log.debug(f"FITS 阶段指纹跳过 ({path.name}): {error}")
            return None


    def _stage_diff_note(self, current_stem: str, previous_stem: str) -> Optional[str]:
        if not self.process_dir:
            return None
        current_path = self.process_dir / f"{current_stem}.fit"
        previous_path = self.process_dir / f"{previous_stem}.fit"
        if not current_path.exists() or not previous_path.exists():
            return None

        current_fingerprint = self._fits_stage_fingerprint(current_path)
        previous_fingerprint = self._fits_stage_fingerprint(previous_path)
        if not current_fingerprint or not previous_fingerprint:
            current_hash = self._sha256_file(current_path)
            previous_hash = self._sha256_file(previous_path)
            if not current_hash or not previous_hash:
                return None
            if current_hash == previous_hash:
                return (
                    f"阶段对比: {current_stem}.fit 与 {previous_stem}.fit "
                    "FITS 容器完全一致；像素内容指纹不可用 "
                    f"(container_sha256={current_hash[:12]})"
                )
            return (
                f"阶段对比: {current_stem}.fit 与 {previous_stem}.fit "
                "像素内容无法判定；完整文件 SHA-256 有变化，"
                "不作为像素变化依据 "
                f"({previous_hash[:8]} -> {current_hash[:8]})"
            )

        data_same = bool(
            current_fingerprint["data_sha256"]
            == previous_fingerprint["data_sha256"]
        )
        layout_same = bool(
            current_fingerprint["layout_sha256"]
            == previous_fingerprint["layout_sha256"]
        )
        header_same = bool(
            current_fingerprint["header_sha256"]
            == previous_fingerprint["header_sha256"]
        )
        container_same = bool(
            current_fingerprint["container_sha256"]
            == previous_fingerprint["container_sha256"]
        )
        data_transition = (
            f"{previous_fingerprint['data_sha256'][:8]} -> "
            f"{current_fingerprint['data_sha256'][:8]}"
        )
        layout_transition = (
            f"{previous_fingerprint['layout_sha256'][:8]} -> "
            f"{current_fingerprint['layout_sha256'][:8]}"
        )
        header_transition = (
            f"{previous_fingerprint['header_sha256'][:8]} -> "
            f"{current_fingerprint['header_sha256'][:8]}"
        )

        if data_same and layout_same:
            content_note = (
                "像素内容一致 "
                f"(data_sha256={current_fingerprint['data_sha256'][:12]})"
            )
        elif not data_same:
            content_note = f"像素内容有变化 (data_sha256={data_transition})"
            if not layout_same:
                content_note += f"，数据布局/缩放语义也有变化 ({layout_transition})"
        else:
            content_note = (
                "像素数据区一致，但数据布局/缩放语义有变化 "
                f"({layout_transition})"
            )

        if header_same:
            header_note = (
                "FITS header 一致 "
                f"(header_sha256={current_fingerprint['header_sha256'][:12]})"
            )
            if not container_same and data_same and layout_same:
                header_note += "；FITS 容器填充有变化，不计为像素内容变化"
        else:
            header_note = (
                "FITS header 有变化，单独报告且不作为像素变化依据 "
                f"({header_transition})"
            )
        return (
            f"阶段对比: {current_stem}.fit 与 {previous_stem}.fit "
            f"{content_note}；{header_note}"
        )


    def _feature_summary_note(self, label: str) -> Optional[str]:
        feat = self._measure_current_features()
        if feat is None:
            return None
        return f"{label}: {format_feature_summary(feat)}"


    def _apply_runtime_env_overrides(self):
        input_mode_raw = os.getenv(ENV_INPUT_MODE_KEY)
        if input_mode_raw is not None:
            normalized = input_mode_raw.strip().lower()
            if normalized in {
                INPUT_MODE_AUTO,
                INPUT_MODE_STAGE1_PREPARED_RESUME,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }:
                self.input_mode = normalized
            else:
                self.log.warn(
                    f"{ENV_INPUT_MODE_KEY} has invalid value; keeping current setting"
                )

        debug_raw = os.getenv(ENV_DEBUG_MODE_KEY)
        if debug_raw is not None:
            parsed = self._parse_env_bool(debug_raw, self.cfg.debug_mode)
            self.cfg.debug_mode = parsed
            if debug_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    f"{ENV_DEBUG_MODE_KEY} has invalid value; keeping current setting"
                )
            self._sync_logger_level()

        output_format_raw = os.getenv("STARUN_OUTPUT_FORMAT")
        if output_format_raw is not None:
            normalized_output = output_format_raw.strip().lower()
            allowed_formats = {"all", "tif", "tiff", "png", "fit", "fits"}
            requested = {
                item.strip()
                for item in normalized_output.split(",")
                if item.strip()
            }
            if requested and requested.issubset(allowed_formats):
                self.cfg.output_format = normalized_output
            else:
                self.log.warn(
                    f"Invalid STARUN_OUTPUT_FORMAT={output_format_raw!r}; using current value"
                )

        filter_hint_raw = os.getenv("STARUN_STAGE4_FILTER_HINT")
        if filter_hint_raw is not None:
            self.cfg.stage4_filter_hint = filter_hint_raw.strip() or "auto"

        stage2_center_protect_raw = os.getenv(
            "STARUN_STAGE2_CENTER_PROTECT_AREA_RATIO"
        )
        if stage2_center_protect_raw is not None:
            try:
                self.cfg.stage2_center_protect_area_ratio = _clamp_float(
                    float(stage2_center_protect_raw.strip()),
                    0.50,
                    0.95,
                )
            except (TypeError, ValueError):
                self.log.warn(
                    "STARUN_STAGE2_CENTER_PROTECT_AREA_RATIO has invalid value; "
                    "keeping current setting"
                )

        plugin_probe_raw = os.getenv("STARUN_WORKFLOW_PLUGIN_PROBE")
        if plugin_probe_raw is not None:
            parsed = self._parse_env_bool(
                plugin_probe_raw,
                self.cfg.workflow_plugin_probe_enabled,
            )
            self.cfg.workflow_plugin_probe_enabled = parsed
            if plugin_probe_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_WORKFLOW_PLUGIN_PROBE has invalid value; "
                    "keeping current setting"
                )

        stage4_platesolve_raw = os.getenv("STARUN_STAGE4_PLATESOLVE_ENABLE")
        if stage4_platesolve_raw is not None:
            parsed = self._parse_env_bool(
                stage4_platesolve_raw,
                getattr(self.cfg, "stage4_platesolve_enabled", False),
            )
            self.cfg.stage4_platesolve_enabled = parsed
            if stage4_platesolve_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_STAGE4_PLATESOLVE_ENABLE has invalid value; "
                    "keeping current setting"
                )

        stage4_auto_geometry_raw = os.getenv(
            "STARUN_STAGE4_AUTO_GEOMETRY_ENABLE"
        )
        if stage4_auto_geometry_raw is not None:
            parsed = self._parse_env_bool(
                stage4_auto_geometry_raw,
                getattr(self.cfg, "stage4_auto_geometry_enabled", True),
            )
            self.cfg.stage4_auto_geometry_enabled = parsed
            if stage4_auto_geometry_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "STARUN_STAGE4_AUTO_GEOMETRY_ENABLE has invalid value; "
                    "keeping current setting"
                )

        stage4_nbn_raw = os.getenv("STARUN_STAGE4_NBN_ENABLE")
        if stage4_nbn_raw is not None:
            parsed = self._parse_env_bool(
                stage4_nbn_raw,
                getattr(
                    self.cfg,
                    "stage4_narrowband_normalization_enabled",
                    True,
                ),
            )
            self.cfg.stage4_narrowband_normalization_enabled = parsed
            if stage4_nbn_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "STARUN_STAGE4_NBN_ENABLE has invalid value; "
                    "keeping current setting"
                )

        stage4_spcc_raw = os.getenv("STARUN_SPCC_ENABLE")
        if stage4_spcc_raw is not None:
            parsed = self._parse_env_bool(
                stage4_spcc_raw,
                getattr(self.cfg, "stage4_spcc_enabled", True),
            )
            self.cfg.stage4_spcc_enabled = parsed
            if stage4_spcc_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "STARUN_SPCC_ENABLE has invalid value; keeping current setting"
                )

        for env_key, attr_name in (
            ("STARUN_STAGE4_SPCC_OSC_SENSOR", "stage4_spcc_osc_sensor"),
            ("STARUN_STAGE4_SPCC_OSC_FILTER", "stage4_spcc_osc_filter"),
            ("STARUN_STAGE4_SPCC_WHITE_REF", "stage4_spcc_white_ref"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is not None:
                setattr(self.cfg, attr_name, raw_value.strip())

        offline_fallback_raw = os.getenv("STARUN_STAGE4_OFFLINE_FALLBACK_MODE")
        if offline_fallback_raw is not None:
            normalized_fallback = offline_fallback_raw.strip().lower()
            if normalized_fallback in {"auto_local_reference", "preserve"}:
                self.cfg.stage4_offline_fallback_mode = normalized_fallback
            else:
                self.log.warn(
                    "STARUN_STAGE4_OFFLINE_FALLBACK_MODE has invalid value; "
                    "keeping current setting"
                )

        auto_reference_white_raw = os.getenv(
            "STARUN_STAGE4_AUTO_REFERENCE_GLOBAL_WHITE_ENABLE"
        )
        if auto_reference_white_raw is not None:
            parsed = self._parse_env_bool(
                auto_reference_white_raw,
                getattr(
                    self.cfg,
                    "stage4_auto_reference_global_white_enabled",
                    False,
                ),
            )
            self.cfg.stage4_auto_reference_global_white_enabled = parsed
            if auto_reference_white_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "STARUN_STAGE4_AUTO_REFERENCE_GLOBAL_WHITE_ENABLE has "
                    "invalid value; keeping current setting"
                )

        local_star_wb_raw = os.getenv("STARUN_STAGE4_LOCAL_STAR_WB_ENABLE")
        if local_star_wb_raw is not None:
            parsed = self._parse_env_bool(
                local_star_wb_raw,
                getattr(self.cfg, "stage4_local_star_wb_enabled", True),
            )
            self.cfg.stage4_local_star_wb_enabled = parsed
            if local_star_wb_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_STAGE4_LOCAL_STAR_WB_ENABLE has invalid value; "
                    "keeping current setting"
                )

        for env_key, attr_name, caster in (
            (
                "STARUN_STAGE4_AUTO_GEOMETRY_CONFIDENCE_MIN",
                "stage4_auto_geometry_confidence_min",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_GEOMETRY_SCALE_RESIDUAL_MAX",
                "stage4_auto_geometry_scale_residual_max",
                float,
            ),
            (
                "STARUN_STAGE4_NBN_MAPPING_CONFIDENCE_MIN",
                "stage4_nbn_mapping_confidence_min",
                float,
            ),
            ("STARUN_STAGE4_NBN_STRENGTH", "stage4_nbn_strength", float),
            ("STARUN_STAGE4_NBN_GAIN_LIMIT", "stage4_nbn_gain_limit", float),
            (
                "STARUN_STAGE4_NBN_LINE_RATIO_DRIFT_MAX",
                "stage4_nbn_line_ratio_drift_max",
                float,
            ),
            ("STARUN_STAGE4_SPCC_TIMEOUT_SEC", "stage4_spcc_timeout_sec", int),
            (
                "STARUN_STAGE4_SPCC_ONLINE_UNVERIFIED_TIMEOUT_SEC",
                "stage4_spcc_online_unverified_timeout_sec",
                int,
            ),
            ("STARUN_STAGE4_SPCC_LIMITMAG", "stage4_spcc_limit_magnitude", float),
            (
                "STARUN_STAGE4_SPCC_NB_R_WAVELENGTH_NM",
                "stage4_spcc_narrowband_r_wavelength_nm",
                float,
            ),
            (
                "STARUN_STAGE4_SPCC_NB_R_BANDWIDTH_NM",
                "stage4_spcc_narrowband_r_bandwidth_nm",
                float,
            ),
            (
                "STARUN_STAGE4_SPCC_NB_G_WAVELENGTH_NM",
                "stage4_spcc_narrowband_g_wavelength_nm",
                float,
            ),
            (
                "STARUN_STAGE4_SPCC_NB_G_BANDWIDTH_NM",
                "stage4_spcc_narrowband_g_bandwidth_nm",
                float,
            ),
            (
                "STARUN_STAGE4_SPCC_NB_B_WAVELENGTH_NM",
                "stage4_spcc_narrowband_b_wavelength_nm",
                float,
            ),
            (
                "STARUN_STAGE4_SPCC_NB_B_BANDWIDTH_NM",
                "stage4_spcc_narrowband_b_bandwidth_nm",
                float,
            ),
            ("STARUN_STAGE4_PCC_TIMEOUT_SEC", "stage4_pcc_timeout_sec", int),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_SAMPLE_TARGET",
                "stage4_auto_reference_background_sample_target",
                int,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_SAMPLE_MIN",
                "stage4_auto_reference_background_sample_min",
                int,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_HOLDOUT_RATIO",
                "stage4_auto_reference_holdout_ratio",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_ERROR_MIN",
                "stage4_auto_reference_background_error_min",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_IMPROVEMENT_MIN",
                "stage4_auto_reference_background_improvement_min",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_STAR_MIN_OBJECTS",
                "stage4_auto_reference_star_min_objects",
                int,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_STAR_RATIO_MAD_MAX",
                "stage4_auto_reference_star_ratio_mad_max",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_STAR_SATURATION_RATIO_MAX",
                "stage4_auto_reference_star_saturation_ratio_max",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_GAIN_LIMIT",
                "stage4_auto_reference_gain_limit",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_STAR_IMPROVEMENT_MIN",
                "stage4_auto_reference_star_improvement_min",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_HIGHLIGHT_CLIP_GROWTH_MAX",
                "stage4_auto_reference_highlight_clip_growth_max",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_BLACK_CLIP_GROWTH_MAX",
                "stage4_auto_reference_black_clip_growth_max",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_GRADIENT_GROWTH_MAX",
                "stage4_auto_reference_gradient_growth_max",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_TEXTURE_GROWTH_MAX",
                "stage4_auto_reference_texture_growth_max",
                float,
            ),
            (
                "STARUN_STAGE4_AUTO_REFERENCE_TARGET_CHROMA_DRIFT_MAX",
                "stage4_auto_reference_target_chroma_drift_max",
                float,
            ),
            ("STARUN_STAGE4_LOCAL_STAR_WB_MIN_PIXELS", "stage4_local_star_wb_min_pixels", int),
            ("STARUN_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT", "stage4_local_star_wb_gain_limit", float),
            ("STARUN_STAGE4_LOCAL_STAR_MASK_RADIUS", "stage4_local_star_mask_radius", int),
            ("STARUN_STAGE4_LOCAL_STAR_MASK_COVERAGE_MAX", "stage4_local_star_mask_coverage_max", float),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, caster(raw_value.strip()))
            except (TypeError, ValueError):
                self.log.warn(f"{env_key} has invalid value; keeping current setting")

        optional_color_raw = os.getenv("STARUN_OPTIONAL_COLOR_TRANSFORM")
        if optional_color_raw is not None:
            parsed = self._parse_env_bool(
                optional_color_raw,
                self.cfg.optional_color_transform_enabled,
            )
            self.cfg.optional_color_transform_enabled = parsed
            if optional_color_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_OPTIONAL_COLOR_TRANSFORM has invalid value; keeping current setting"
                )

        dualband_palette_raw = os.getenv("STARUN_STAGE8_DUALBAND_PALETTE_ENABLE")
        if dualband_palette_raw is not None:
            parsed = self._parse_env_bool(
                dualband_palette_raw,
                getattr(self.cfg, "stage8_dualband_palette_enabled", True),
            )
            self.cfg.stage8_dualband_palette_enabled = parsed
            if dualband_palette_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "STARUN_STAGE8_DUALBAND_PALETTE_ENABLE has invalid value; "
                    "keeping current setting"
                )

        aberration_api_raw = os.getenv("STARUN_ABERRATION_API_ENABLE")
        if aberration_api_raw is not None:
            parsed = self._parse_env_bool(
                aberration_api_raw,
                getattr(self.cfg, "aberration_api_enabled", False),
            )
            self.cfg.aberration_api_enabled = parsed
            if aberration_api_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_ABERRATION_API_ENABLE has invalid value; keeping current setting"
                )

        denoise_enable_raw = os.getenv("STARUN_DENOISE_ENABLE")
        if denoise_enable_raw is not None:
            parsed = self._parse_env_bool(
                denoise_enable_raw,
                self.cfg.denoise_enabled,
            )
            self.cfg.denoise_enabled = parsed
            if denoise_enable_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_DENOISE_ENABLE has invalid value; keeping current setting"
                )

        multiscale_denoise_raw = os.getenv(
            "STARUN_STAGE5_MULTISCALE_DENOISE_ENABLE"
        )
        if multiscale_denoise_raw is not None:
            parsed = self._parse_env_bool(
                multiscale_denoise_raw,
                getattr(self.cfg, "stage5_multiscale_denoise_enabled", True),
            )
            self.cfg.stage5_multiscale_denoise_enabled = parsed
            if multiscale_denoise_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "STARUN_STAGE5_MULTISCALE_DENOISE_ENABLE has invalid value; "
                    "keeping current setting"
                )

        denoise_force_raw = os.getenv("STARUN_DENOISE_FORCE")
        if denoise_force_raw is not None:
            parsed = self._parse_env_bool(
                denoise_force_raw,
                self.cfg.denoise_enabled,
            )
            self._force_denoise_enabled = parsed
            if denoise_force_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_DENOISE_FORCE has invalid value; keeping current setting"
                )

        stage5_deconv_raw = os.getenv("STARUN_STAGE5_DECONV_ENABLE")
        if stage5_deconv_raw is not None:
            parsed = self._parse_env_bool(
                stage5_deconv_raw,
                getattr(self.cfg, "stage5_deconvolution_enabled", True),
            )
            self.cfg.stage5_deconvolution_enabled = parsed
            if stage5_deconv_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "STARUN_STAGE5_DECONV_ENABLE has invalid value; keeping current setting"
                )

        stage5_graxpert_deconv_raw = os.getenv(
            "STARUN_STAGE5_GRAXPERT_DECONV_ENABLE"
        )
        if stage5_graxpert_deconv_raw is not None:
            parsed = self._parse_env_bool(
                stage5_graxpert_deconv_raw,
                getattr(self.cfg, "stage5_graxpert_deconvolution_enabled", True),
            )
            self.cfg.stage5_graxpert_deconvolution_enabled = parsed
            if stage5_graxpert_deconv_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "STARUN_STAGE5_GRAXPERT_DECONV_ENABLE has invalid value; "
                    "keeping current setting"
                )

        denoise_mod_raw = os.getenv("STARUN_DENOISE_MOD")
        if denoise_mod_raw is not None:
            try:
                self.cfg.denoise_mod = float(denoise_mod_raw.strip())
            except ValueError:
                self.log.warn(
                    "Invalid STARUN_DENOISE_MOD="
                    f"{denoise_mod_raw!r}; using current value"
                )

        for env_key, attr_name, caster in (
            (
                "STARUN_STAGE5_MULTISCALE_DENOISE_STRENGTH",
                "stage5_multiscale_denoise_strength",
                float,
            ),
            (
                "STARUN_STAGE5_MULTISCALE_DETAIL_RETENTION_MIN",
                "stage5_multiscale_detail_retention_min",
                float,
            ),
            (
                "STARUN_STAGE5_MULTISCALE_NOISE_REDUCTION_MIN",
                "stage5_multiscale_noise_reduction_min",
                float,
            ),
            (
                "STARUN_STAGE5_DENOISE_CHROMA_NOISE_GROWTH_MAX",
                "stage5_denoise_chroma_noise_growth_max",
                float,
            ),
            ("STARUN_STAGE5_RL_MAXSTARS", "stage5_rl_maxstars", int),
            ("STARUN_STAGE5_RL_PSF_KS", "stage5_rl_psf_kernel_size", int),
            ("STARUN_STAGE5_RL_ITERS", "stage5_rl_iters", int),
            ("STARUN_STAGE5_RL_ALPHA", "stage5_rl_alpha", float),
            ("STARUN_STAGE5_RL_GDSTEP", "stage5_rl_gdstep", float),
            ("STARUN_STAGE5_RL_STOP", "stage5_rl_stop", float),
            ("STARUN_STAGE5_GRAXPERT_DECONV_STRENGTH", "stage5_graxpert_deconv_strength", float),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, caster(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        stage7_retry_raw = os.getenv("STARUN_STAGE7_QUALITY_RETRY_MAX")
        if stage7_retry_raw is not None:
            try:
                self.cfg.stage7_quality_retry_max = int(stage7_retry_raw.strip())
            except ValueError:
                self.log.warn(
                    "Invalid STARUN_STAGE7_QUALITY_RETRY_MAX="
                    f"{stage7_retry_raw!r}; using current value"
                )

        for env_key, attr_name in (
            ("STARUN_STAGE7_LARGE_GALAXY_HALO_RESIDUE_SCORE_MAX", "stage7_large_galaxy_halo_residue_score_max"),
            ("STARUN_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX", "stage7_bright_nebula_halo_residue_score_max"),
            ("STARUN_STAGE7_GALAXY_CORE_PRESERVATION_RATIO_MIN", "stage7_galaxy_core_preservation_ratio_min"),
            ("STARUN_STAGE7_GALAXY_CORE_CONTRAST_RATIO_MIN", "stage7_galaxy_core_contrast_ratio_min"),
            ("STARUN_STAGE7_STARLESS_REPAIR_STRENGTH", "stage7_starless_repair_strength"),
            ("STARUN_STAGE7_STARLESS_HALO_REPAIR_STRENGTH", "stage7_starless_halo_repair_strength"),
            ("STARUN_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH", "stage7_starless_chroma_denoise_strength"),
            ("STARUN_STAGE7_STRETCH_CHROMA_NOISE_SCORE_MAX", "stage7_stretch_chroma_noise_score_max"),
            ("STARUN_STAGE7_STRETCH_BACKGROUND_MOTTLING_SCORE_MAX", "stage7_stretch_background_mottling_score_max"),
            ("STARUN_STAGE7_STRETCH_CHROMA_LOAD_GROWTH_MAX", "stage7_stretch_chroma_load_growth_max"),
            ("STARUN_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_MAX", "stage7_stretch_chroma_load_low_absolute_max"),
            ("STARUN_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_TOLERANCE", "stage7_stretch_chroma_load_low_absolute_tolerance"),
            ("STARUN_STAGE7_UNCALIBRATED_BACKGROUND_CHROMA_LOAD_REVIEW_MAX", "stage7_uncalibrated_background_chroma_load_review_max"),
            ("STARUN_STAGE7_PREVIEW_TARGET_P50_MIN_RATIO", "stage7_preview_target_p50_min_ratio"),
            ("STARUN_STAGE7_PREVIEW_TARGET_P50_MAX_RATIO", "stage7_preview_target_p50_max_ratio"),
            ("STARUN_STAGE7_BRIGHT_NEBULA_STAR_MASK_EXPAND", "stage7_bright_nebula_star_mask_expand"),
            ("STARUN_STAGE7_BRIGHT_NEBULA_STAR_FAINT_SUPPRESSION", "stage7_bright_nebula_star_faint_suppression"),
            ("STARUN_STAGE7_BRIGHT_NEBULA_STAR_DETAIL_SUPPRESSION", "stage7_bright_nebula_star_detail_suppression"),
            ("STARUN_STAGE7_STARLESS_MASKED_RANK_DRIFT_P95_MAX", "stage7_starless_masked_rank_drift_p95_max"),
            ("STARUN_STAGE7_STARLESS_HALO_DETAIL_GROWTH_RATIO_MAX", "stage7_starless_halo_detail_growth_ratio_max"),
            ("STARUN_STAGE7_STARLESS_HALO_DETAIL_DELTA_MIN", "stage7_starless_halo_detail_delta_min"),
            ("STARUN_STAGE7_STARLESS_PEAK_BACKGROUND_RATIO_MIN", "stage7_starless_peak_background_ratio_min"),
            ("STARUN_STAGE7_STARLESS_REPAIR_CHROMA_REDUCTION_MIN", "stage7_starless_repair_chroma_reduction_min"),
            ("STARUN_STAGE7_STARLESS_REPAIR_CHROMA_DELTA_MIN", "stage7_starless_repair_chroma_delta_min"),
            ("STARUN_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX", "stage7_starmask_diffuse_residual_ratio_max"),
            ("STARUN_STAGE8_LOCAL_CURVE_OPACITY", "stage8_local_curve_opacity"),
            ("STARUN_STAGE8_LIMITED_SATURATION_MAX", "stage8_limited_saturation_max"),
            ("STARUN_STAGE8_LIMITED_HALO_TEXTURE_GROWTH_MAX", "stage8_limited_halo_texture_growth_max"),
            ("STARUN_STAGE8_LIMITED_HALO_TEXTURE_DELTA_MAX", "stage8_limited_halo_texture_delta_max"),
            ("STARUN_STAGE8_DUALBAND_PALETTE_STRENGTH", "stage8_dualband_palette_strength"),
            ("STARUN_STAGE8_DUALBAND_PALETTE_LUMA_DRIFT_MAX", "stage8_dualband_palette_luma_drift_max"),
            ("STARUN_STAGE8_DUALBAND_PALETTE_CLIP_GROWTH_MAX", "stage8_dualband_palette_clip_growth_max"),
            ("STARUN_STAGE8_DUALBAND_PALETTE_QUALITY_WARNING_TOLERANCE", "stage8_dualband_palette_quality_warning_tolerance"),
            ("STARUN_STAGE9_SASP_STAR_STRETCH_AMOUNT", "stage9_sasp_star_stretch_amount"),
            ("STARUN_STAGE9_NB_TO_RGB_STARS_RATIO", "stage9_nb_to_rgb_stars_ratio"),
            ("STARUN_STAGE9_STARMASK_ASINH_STRETCH", "stage9_starmask_asinh_stretch"),
            ("STARUN_STAGE9_STARMASK_ASINH_OFFSET", "stage9_starmask_asinh_offset"),
            ("STARUN_STAGE9_STARMASK_ASINH_STRETCH_MAX", "stage9_starmask_asinh_stretch_max"),
            ("STARUN_STAGE9_STARMASK_FAINT_TARGET", "stage9_starmask_faint_target"),
            ("STARUN_STAGE9_STARMASK_MID_TARGET", "stage9_starmask_mid_target"),
            ("STARUN_STAGE9_STARMASK_BRIGHT_TARGET", "stage9_starmask_bright_target"),
            ("STARUN_STAGE9_STARMASK_PEAK_TARGET", "stage9_starmask_peak_target"),
            ("STARUN_STAGE9_STARMASK_FAINT_CHROMA_MAX", "stage9_starmask_faint_chroma_max"),
            ("STARUN_STAGE9_STARMASK_BRIGHT_CHROMA_MAX", "stage9_starmask_bright_chroma_max"),
            ("STARUN_STAGE9_STARMASK_PREDICTED_CHANGE_RATIO_MAX", "stage9_starmask_predicted_change_ratio_max"),
            ("STARUN_STAGE9_STAR_COLOR_REPAIR_STRENGTH", "stage9_star_color_repair_strength"),
            ("STARUN_STAGE9_STAR_COLOR_SUPPORT_RATIO_MAX", "stage9_star_color_support_ratio_max"),
            ("STARUN_STAGE9_STAR_COLOR_IMPROVEMENT_MIN", "stage9_star_color_improvement_min"),
            ("STARUN_STAGE9_STAR_COLOR_POST_CHROMA_ERROR_MAX", "stage9_star_color_post_chroma_error_max"),
            ("STARUN_STAGE9_STAR_REFERENCE_SIGMA", "stage9_star_reference_sigma"),
            ("STARUN_STAGE9_COMPACT_WEAK_STAR_RETENTION_MIN", "stage9_compact_weak_star_retention_min"),
            ("STARUN_STAGE9_MIXED_STAR_PEAK_RATIO_MIN", "stage9_mixed_star_peak_ratio_min"),
            ("STARUN_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX", "stage7_local_core_clip_ratio_max"),
            ("STARUN_STAGE7_LOCAL_FAINT_SNR_MIN", "stage7_local_faint_snr_min"),
            ("STARUN_STAGE7_LOCAL_DARK_SEPARATION_MIN", "stage7_local_dark_separation_min"),
            ("STARUN_STAGE9_HIGHLIGHT_CLIP_RATIO_MAX", "stage9_highlight_clip_ratio_max"),
            ("STARUN_STAGE9_HIGHLIGHT_CLIP_GROWTH_MAX", "stage9_highlight_clip_growth_max"),
            ("STARUN_STAGE9_BRIGHT_PIXEL_GROWTH_MAX", "stage9_bright_pixel_growth_max"),
            ("STARUN_STAGE9_BACKGROUND_LIFT_MAX", "stage9_background_lift_max"),
            ("STARUN_STAGE9_BACKGROUND_MOTTLING_GROWTH_MAX", "stage9_background_mottling_growth_max"),
            ("STARUN_STAGE9_MOTTLING_EXEMPTION_CHANGED_PIXEL_RATIO_MAX", "stage9_mottling_exemption_changed_pixel_ratio_max"),
            ("STARUN_STAGE9_CHANGED_PIXEL_RATIO_MAX", "stage9_changed_pixel_ratio_max"),
            ("STARUN_STAGE9_DARKENING_RATIO_MAX", "stage9_darkening_ratio_max"),
            ("STARUN_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN", "stage9_weak_star_recovery_ratio_min"),
            ("STARUN_STAGE9_STAR_RECOVERY_RATIO_MIN", "stage9_star_recovery_ratio_min"),
            ("STARUN_STAGE9_SOURCE_STAR_DETAIL_PERCENTILE", "stage9_source_star_detail_percentile"),
            ("STARUN_STAGE9_SOURCE_COMPONENT_DENSITY_MAX", "stage9_source_component_density_max"),
            ("STARUN_STAGE9_SOURCE_SINGLE_PIXEL_RATIO_MAX", "stage9_source_single_pixel_ratio_max"),
            ("STARUN_STAGE9_WEAK_STAR_SCREEN_INTENSITY_MIN", "stage9_weak_star_screen_intensity_min"),
            ("STARUN_STAGE9_STAR_SUPPORT_RATIO_MAX", "stage9_star_support_ratio_max"),
            ("STARUN_STAGE9_UNMATCHED_CHANGED_RATIO_MAX", "stage9_unmatched_changed_ratio_max"),
            ("STARUN_STAGE9_CHROMATIC_ADDITION_PEAK_MIN", "stage9_chromatic_addition_peak_min"),
            ("STARUN_STAGE9_CHROMATIC_ADDITION_SATURATION_MIN", "stage9_chromatic_addition_saturation_min"),
            ("STARUN_STAGE9_CHROMATIC_ADDITION_RATIO_MAX", "stage9_chromatic_addition_ratio_max"),
            ("STARUN_STAGE9_STAR_POSITIVE_DELTA_WINDOW_RECOVERY_RATIO_MIN", "stage9_star_positive_delta_window_recovery_ratio_min"),
            ("STARUN_STAGE9_STAR_WING_RECOVERY_RATIO_MIN", "stage9_star_wing_recovery_ratio_min"),
            ("STARUN_STAGE9_RESIDUAL_DARK_HOLE_RATIO_MAX", "stage9_residual_dark_hole_ratio_max"),
            ("STARUN_STAGE9_HOLLOW_STRUCTURE_DELTA_MIN", "stage9_hollow_structure_delta_min"),
            ("STARUN_STAGE9_NEW_HOLLOW_STRUCTURE_AREA_MAX", "stage9_new_hollow_structure_area_max"),
            ("STARUN_STAGE9_LOCAL_COMPONENT_PEAK_MIN", "stage9_local_component_peak_min"),
            ("STARUN_STAGE9_LOCAL_COMPONENT_AREA_MAX", "stage9_local_component_area_max"),
            ("STARUN_STAGE9_LOCAL_COMPONENT_ASPECT_RATIO_MAX", "stage9_local_component_aspect_ratio_max"),
            ("STARUN_STAGE9_LOCAL_COMPONENT_FILL_RATIO_MIN", "stage9_local_component_fill_ratio_min"),
            ("STARUN_STAGE9_LOCAL_SINGLE_PIXEL_RATIO_MAX", "stage9_local_single_pixel_ratio_max"),
            ("STARUN_STAGE9_LOCAL_CYAN_BLUE_PEAK_MIN", "stage9_local_cyan_blue_peak_min"),
            ("STARUN_STAGE9_LOCAL_CYAN_BLUE_SATURATION_MIN", "stage9_local_cyan_blue_saturation_min"),
            ("STARUN_STAGE9_LOCAL_CYAN_BLUE_COMPONENT_AREA_MAX", "stage9_local_cyan_blue_component_area_max"),
            ("STARUN_STAGE9_CORE_PERCENTILE", "stage9_core_percentile"),
            ("STARUN_STAGE9_CORE_COLOR_JUMP_MIN", "stage9_core_color_jump_min"),
            ("STARUN_STAGE9_CORE_COLOR_JUMP_COMPONENT_AREA_MAX", "stage9_core_color_jump_component_area_max"),
            ("STARUN_STAGE10_CHROMA_FOCUS_SCORE_MIN", "stage10_chroma_focus_score_min"),
            ("STARUN_STAGE10_SEPARATE_CHROMA_SCORE_MIN", "stage10_separate_chroma_score_min"),
            ("STARUN_STAGE10_FULL_BG_STD_MIN", "stage10_full_bg_std_min"),
            ("STARUN_STAGE10_FULL_MOTTLING_SCORE_MIN", "stage10_full_mottling_score_min"),
            ("STARUN_STAGE10_FINAL_DENOISE_STRENGTH", "stage10_final_denoise_strength"),
            ("STARUN_STAGE10_STAR_PROTECTION_COVERAGE_MAX", "stage10_star_protection_coverage_max"),
            ("STARUN_STAGE10_LARGE_GALAXY_LOCAL_PATCH_VARIANCE_MAX", "stage10_large_galaxy_local_patch_variance_max"),
            ("STARUN_STAGE10_STAGE9_LOCAL_COLOR_RISK_STRENGTH", "stage10_stage9_local_color_risk_strength"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, float(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        for env_key, attr_name in (
            ("STARUN_STAGE7_STRETCH_FEEDBACK_RETRY_MAX", "stage7_stretch_feedback_retry_max"),
            ("STARUN_STAGE8_LIMITED_CORE_EXCLUSION_EXPAND", "stage8_limited_core_exclusion_expand"),
            ("STARUN_STAGE9_MIXED_STAR_WEAK_COUNT_MIN", "stage9_mixed_star_weak_count_min"),
            ("STARUN_STAGE9_MIXED_STAR_BRIGHT_COUNT_MIN", "stage9_mixed_star_bright_count_min"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, int(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        for env_key, attr_name in (
            ("STARUN_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE", "stage7_starless_pixel_repair_enabled"),
            ("STARUN_STAGE7_GALAXY_ROI_HALO_GATE_ENABLE", "stage7_galaxy_roi_halo_gate_enabled"),
            ("STARUN_STAGE7_CHROMA_RESCUE_ENABLE", "stage7_chroma_rescue_enabled"),
            ("STARUN_STAGE7_STARLESS_STRUCTURE_GATE_ENABLE", "stage7_starless_structure_gate_enabled"),
            ("STARUN_STAGE7_QUANTILE_FALLBACK_ENABLE", "stage7_quantile_fallback_enabled"),
            ("STARUN_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR", "stage8_force_conservative_after_stage7_repair"),
            ("STARUN_STAGE8_LOCAL_ADJUSTMENT_ENGINE_ENABLE", "stage8_local_adjustment_engine_enabled"),
            ("STARUN_STAGE9_SASP_STAR_STRETCH_ENABLE", "stage9_sasp_star_stretch_enabled"),
            ("STARUN_STAGE9_NB_TO_RGB_STARS_ENABLE", "stage9_nb_to_rgb_stars_enabled"),
            ("STARUN_STAGE9_STARMASK_STRETCH_ENABLE", "stage9_starmask_stretch_enabled"),
            ("STARUN_STAGE9_STARMASK_ADAPTIVE_STRETCH_ENABLE", "stage9_starmask_adaptive_stretch_enabled"),
            ("STARUN_STAGE9_COMPACT_STARMASK_ENABLE", "stage9_compact_starmask_enabled"),
            ("STARUN_STAGE9_STARMASK_PRE_STRETCH_COMPACT_ENABLE", "stage9_starmask_pre_stretch_compact_enabled"),
            ("STARUN_STAGE9_STAR_COLOR_REPAIR_ENABLE", "stage9_star_color_repair_enabled"),
            ("STARUN_STAGE9_STAR_COLOR_POST_VALIDATION_ENABLE", "stage9_star_color_post_validation_enabled"),
            ("STARUN_STAGE9_STARMASK_CHROMA_REGULARIZATION_ENABLE", "stage9_starmask_chroma_regularization_enabled"),
            ("STARUN_STAGE9_QUALITY_GATE_ENABLE", "stage9_quality_gate_enabled"),
            ("STARUN_STAGE10_MANAGED_OUTPUT_ENABLE", "stage10_managed_output_enabled"),
            ("STARUN_FORCE_REVIEW_ONLY_OUTPUT", "force_review_only_output"),
            ("STARUN_STAGE7_TARGET_LOCAL_METRICS_ENABLE", "stage7_target_local_metrics_enabled"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            parsed = self._parse_env_bool(raw_value, getattr(self.cfg, attr_name))
            setattr(self.cfg, attr_name, parsed)
            if raw_value.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(f"{env_key} has invalid value; keeping current setting")

        legacy_compact_env_key = "STARUN_STAGE9_COMPACT_STARMASK_ENABLE"
        pre_stretch_compact_env_key = (
            "STARUN_STAGE9_STARMASK_PRE_STRETCH_COMPACT_ENABLE"
        )
        explicit_env_keys = getattr(
            self,
            "_project_env_explicit_keys",
            frozenset(
                key
                for key in (
                    legacy_compact_env_key,
                    pre_stretch_compact_env_key,
                )
                if os.getenv(key) is not None
            ),
        )
        legacy_compact_raw = os.getenv(legacy_compact_env_key)
        if (
            legacy_compact_env_key in explicit_env_keys
            and pre_stretch_compact_env_key not in explicit_env_keys
            and legacy_compact_raw is not None
            and legacy_compact_raw.strip().lower()
            in (ENV_TRUE_VALUES | ENV_FALSE_VALUES)
        ):
            self.cfg.stage9_starmask_pre_stretch_compact_enabled = (
                self._parse_env_bool(
                    legacy_compact_raw,
                    self.cfg.stage9_starmask_pre_stretch_compact_enabled,
                )
            )
            self.log.info(
                "Legacy STARUN_STAGE9_COMPACT_STARMASK_ENABLE mirrored to "
                "STARUN_STAGE9_STARMASK_PRE_STRETCH_COMPACT_ENABLE"
            )

        old_stage7_retry = self.cfg.stage7_quality_retry_max
        old_stage7_large_galaxy_halo = self.cfg.stage7_large_galaxy_halo_residue_score_max
        old_stage7_bright_halo = self.cfg.stage7_bright_nebula_halo_residue_score_max
        old_stage7_repair = self.cfg.stage7_starless_repair_strength
        old_stage7_halo = self.cfg.stage7_starless_halo_repair_strength
        old_stage7_chroma = self.cfg.stage7_starless_chroma_denoise_strength
        old_stage9_starmask_stretch = self.cfg.stage9_starmask_asinh_stretch
        old_stage9_starmask_offset = self.cfg.stage9_starmask_asinh_offset
        self.cfg.stage4_spcc_timeout_sec = _clamp_int(
            getattr(self.cfg, "stage4_spcc_timeout_sec", 300),
            5,
            300,
        )
        self.cfg.stage4_spcc_online_unverified_timeout_sec = _clamp_int(
            getattr(
                self.cfg,
                "stage4_spcc_online_unverified_timeout_sec",
                90,
            ),
            30,
            180,
        )
        self.cfg.stage4_spcc_limit_magnitude = _clamp_float(
            getattr(self.cfg, "stage4_spcc_limit_magnitude", 10.5),
            1.0,
            25.0,
        )
        for attr_name, lower, upper in (
            ("stage4_spcc_narrowband_r_wavelength_nm", 300.0, 900.0),
            ("stage4_spcc_narrowband_g_wavelength_nm", 300.0, 900.0),
            ("stage4_spcc_narrowband_b_wavelength_nm", 300.0, 900.0),
            ("stage4_spcc_narrowband_r_bandwidth_nm", 1.0, 100.0),
            ("stage4_spcc_narrowband_g_bandwidth_nm", 1.0, 100.0),
            ("stage4_spcc_narrowband_b_bandwidth_nm", 1.0, 100.0),
        ):
            original = float(getattr(self.cfg, attr_name))
            effective = _clamp_float(original, lower, upper)
            setattr(self.cfg, attr_name, effective)
            if effective != original:
                self.log.warn(
                    f"{attr_name} outside safe range [{lower:g}, {upper:g}]; "
                    f"clamped {original:g} -> {effective:g}"
                )
        self.cfg.stage4_auto_reference_background_sample_target = _clamp_int(
            getattr(
                self.cfg,
                "stage4_auto_reference_background_sample_target",
                40,
            ),
            16,
            64,
        )
        self.cfg.stage4_auto_reference_background_sample_min = min(
            self.cfg.stage4_auto_reference_background_sample_target,
            _clamp_int(
                getattr(
                    self.cfg,
                    "stage4_auto_reference_background_sample_min",
                    16,
                ),
                16,
                40,
            ),
        )
        self.cfg.stage4_auto_reference_star_min_objects = _clamp_int(
            getattr(self.cfg, "stage4_auto_reference_star_min_objects", 16),
            16,
            256,
        )
        for attr_name, lower, upper in (
            ("stage4_auto_reference_holdout_ratio", 0.20, 0.40),
            ("stage4_auto_reference_background_error_min", 0.0, 0.25),
            ("stage4_auto_reference_background_improvement_min", 0.01, 0.90),
            ("stage4_auto_reference_star_ratio_mad_max", 0.01, 0.50),
            ("stage4_auto_reference_star_saturation_ratio_max", 0.0, 0.50),
            ("stage4_auto_reference_gain_limit", 1.01, 1.20),
            ("stage4_auto_reference_star_improvement_min", 0.01, 0.90),
            ("stage4_auto_reference_highlight_clip_growth_max", 0.0, 0.05),
            ("stage4_auto_reference_black_clip_growth_max", 0.0, 0.05),
            ("stage4_auto_reference_gradient_growth_max", 1.0, 2.0),
            ("stage4_auto_reference_texture_growth_max", 1.0, 2.0),
            ("stage4_auto_reference_target_chroma_drift_max", 0.01, 0.50),
        ):
            setattr(
                self.cfg,
                attr_name,
                _clamp_float(getattr(self.cfg, attr_name), lower, upper),
            )
        self.cfg.stage7_quality_retry_max = _clamp_int(
            self.cfg.stage7_quality_retry_max, 0, 3
        )
        self.cfg.stage7_stretch_feedback_retry_max = _clamp_int(
            self.cfg.stage7_stretch_feedback_retry_max, 0, 1
        )
        self.cfg.stage7_bright_nebula_star_mask_expand = _clamp_int(
            self.cfg.stage7_bright_nebula_star_mask_expand, 1, 8
        )
        self.cfg.stage8_limited_core_exclusion_expand = _clamp_int(
            self.cfg.stage8_limited_core_exclusion_expand, 2, 16
        )
        self.cfg.stage7_bright_nebula_halo_residue_score_max = _clamp_float(
            self.cfg.stage7_bright_nebula_halo_residue_score_max,
            self.cfg.stage7_halo_residue_score_max,
            1.20,
        )
        self.cfg.stage7_large_galaxy_halo_residue_score_max = _clamp_float(
            self.cfg.stage7_large_galaxy_halo_residue_score_max,
            self.cfg.stage7_halo_residue_score_max,
            1.0,
        )
        self.cfg.stage7_starless_repair_strength = _clamp_float(
            self.cfg.stage7_starless_repair_strength,
            0.0,
            0.85,
        )
        self.cfg.stage7_starless_halo_repair_strength = _clamp_float(
            self.cfg.stage7_starless_halo_repair_strength,
            0.0,
            0.90,
        )
        self.cfg.stage7_starless_chroma_denoise_strength = _clamp_float(
            self.cfg.stage7_starless_chroma_denoise_strength,
            0.0,
            0.90,
        )
        self.cfg.stage9_starmask_asinh_stretch = _clamp_float(
            self.cfg.stage9_starmask_asinh_stretch,
            1.10,
            3.00,
        )
        self.cfg.stage9_sasp_star_stretch_amount = _clamp_float(
            getattr(self.cfg, "stage9_sasp_star_stretch_amount", 3.0),
            0.50,
            5.00,
        )
        self.cfg.stage9_nb_to_rgb_stars_ratio = _clamp_float(
            getattr(self.cfg, "stage9_nb_to_rgb_stars_ratio", 0.30),
            0.0,
            1.0,
        )
        self.cfg.stage9_starmask_asinh_offset = _clamp_float(
            self.cfg.stage9_starmask_asinh_offset,
            0.0005,
            0.0060,
        )
        for attr_name, lower, upper in (
            ("stage7_9_quality_advisory_multiplier", 1.0, 2.0),
            ("stage7_stretch_chroma_noise_score_max", 0.10, 0.80),
            ("stage7_stretch_background_mottling_score_max", 0.10, 1.00),
            ("stage7_stretch_chroma_load_growth_max", 1.00, 3.00),
            ("stage7_stretch_chroma_load_low_absolute_max", 0.01, 0.15),
            ("stage7_stretch_chroma_load_low_absolute_tolerance", 0.0, 0.01),
            ("stage7_display90_reference_chroma_load_ratio_max", 1.00, 1.20),
            ("stage7_display90_reference_chroma_load_absolute_max", 0.15, 0.50),
            ("stage7_uncalibrated_background_chroma_load_review_max", 0.04, 0.50),
            ("stage7_preview_target_p50_min_ratio", 0.25, 0.90),
            ("stage7_preview_target_p50_max_ratio", 1.00, 3.00),
            ("stage7_mtf_reference_blackpoint_sigma", 0.50, 8.00),
            ("stage7_mtf_reference_p50_relative_error_max", 0.01, 0.25),
            ("stage7_mtf_reference_p50_absolute_error_max", 0.0001, 0.03),
            ("stage7_bright_nebula_star_faint_suppression", 0.0, 1.0),
            ("stage7_bright_nebula_star_detail_suppression", 0.0, 0.60),
            ("stage7_starless_masked_rank_drift_p95_max", 0.02, 0.50),
            ("stage7_starless_halo_detail_growth_ratio_max", 1.05, 4.00),
            ("stage7_starless_halo_detail_delta_min", 0.001, 0.10),
            ("stage7_starless_peak_background_ratio_min", 1.5, 12.0),
            ("stage7_galaxy_core_preservation_ratio_min", 0.30, 0.95),
            ("stage7_galaxy_core_contrast_ratio_min", 0.30, 0.95),
            ("stage7_starless_repair_chroma_reduction_min", 0.05, 0.80),
            ("stage7_starless_repair_chroma_delta_min", 0.00001, 0.05000),
            ("stage7_starmask_diffuse_residual_ratio_max", 0.01, 0.50),
            ("stage8_dualband_palette_strength", 0.10, 1.00),
            ("stage8_dualband_palette_luma_drift_max", 0.001, 0.030),
            ("stage8_dualband_palette_clip_growth_max", 0.0, 0.020),
            ("stage8_dualband_palette_quality_warning_tolerance", 0.0, 1.0),
            ("stage9_psf_review_fwhm_ratio_max", 1.10, 1.65),
            ("stage9_highlight_clip_ratio_max", 0.001, 0.10),
            ("stage9_highlight_clip_growth_max", 0.0, 0.05),
            ("stage9_bright_pixel_growth_max", 0.0, 0.10),
            ("stage9_background_lift_max", 0.0, 0.05),
            ("stage9_background_mottling_growth_max", 1.0, 3.0),
            ("stage9_changed_pixel_ratio_max", 0.05, 0.80),
            ("stage9_starmask_predicted_change_ratio_max", 0.05, 0.60),
            ("stage9_darkening_ratio_max", 0.0, 0.05),
            ("stage9_star_reference_sigma", 3.0, 8.0),
            ("stage9_compact_weak_star_retention_min", 0.50, 0.98),
            ("stage9_mixed_star_peak_ratio_min", 2.0, 20.0),
            ("stage9_weak_star_recovery_ratio_min", 0.40, 0.95),
            ("stage9_star_recovery_ratio_min", 0.40, 0.98),
            ("stage9_source_star_detail_percentile", 97.0, 99.5),
            ("stage9_source_component_density_max", 500.0, 10000.0),
            ("stage9_source_single_pixel_ratio_max", 0.10, 0.90),
            ("stage9_starmask_faint_target", 0.08, 0.40),
            ("stage9_starmask_mid_target", 0.30, 0.70),
            ("stage9_starmask_bright_target", 0.50, 0.88),
            ("stage9_starmask_peak_target", 0.75, 0.95),
            ("stage9_starmask_faint_chroma_max", 0.10, 0.80),
            ("stage9_starmask_bright_chroma_max", 0.10, 0.90),
            ("stage9_weak_star_screen_intensity_min", 0.10, 1.05),
            ("stage9_star_support_ratio_max", 0.03, 0.20),
            ("stage9_unmatched_changed_ratio_max", 0.0, 0.05),
            ("stage9_chromatic_addition_peak_min", 0.002, 0.25),
            ("stage9_chromatic_addition_saturation_min", 0.30, 0.95),
            ("stage9_chromatic_addition_ratio_max", 0.0, 0.05),
            ("stage9_star_positive_delta_window_recovery_ratio_min", 0.40, 0.98),
            ("stage9_star_wing_recovery_ratio_min", 0.30, 0.95),
            ("stage9_residual_dark_hole_ratio_max", 0.0, 0.50),
            ("stage9_hollow_structure_delta_min", 0.01, 0.25),
            ("stage9_new_hollow_structure_area_max", 4.0, 4096.0),
            ("stage9_local_component_peak_min", 0.002, 0.10),
            ("stage9_local_component_area_max", 16.0, 4096.0),
            ("stage9_local_component_aspect_ratio_max", 1.2, 10.0),
            ("stage9_local_component_fill_ratio_min", 0.02, 0.80),
            ("stage9_local_single_pixel_ratio_max", 0.0, 0.90),
            ("stage9_local_cyan_blue_peak_min", 0.002, 0.10),
            ("stage9_local_cyan_blue_saturation_min", 0.20, 0.95),
            ("stage9_local_cyan_blue_component_area_max", 4.0, 2048.0),
            ("stage9_core_percentile", 70.0, 99.0),
            ("stage9_core_color_jump_min", 0.03, 0.50),
            ("stage9_core_color_jump_component_area_max", 4.0, 2048.0),
            ("stage10_chroma_focus_score_min", 0.10, 0.80),
            ("stage10_separate_chroma_score_min", 0.35, 1.50),
            ("stage10_full_bg_std_min", 0.001, 0.10),
            ("stage10_full_mottling_score_min", 0.10, 1.00),
            ("stage10_final_denoise_strength", 0.05, 0.50),
            ("stage10_star_protection_coverage_max", 0.05, 0.60),
            ("stage10_large_galaxy_local_patch_variance_max", 0.00022, 0.00100),
            ("stage10_stage9_local_color_risk_strength", 0.0, 1.0),
        ):
            old_value = float(getattr(self.cfg, attr_name))
            new_value = _clamp_float(old_value, lower, upper)
            setattr(self.cfg, attr_name, new_value)
            if old_value != new_value:
                self.log.warn(
                    f"{attr_name} clamped: {old_value} -> {new_value}"
                )
        faint_star_target = float(self.cfg.stage9_starmask_faint_target)
        mid_star_target = max(
            float(self.cfg.stage9_starmask_mid_target),
            faint_star_target + 0.03,
        )
        bright_star_target = max(
            float(self.cfg.stage9_starmask_bright_target),
            mid_star_target + 0.03,
        )
        peak_star_target = max(
            float(self.cfg.stage9_starmask_peak_target),
            bright_star_target + 0.03,
        )
        (
            self.cfg.stage9_starmask_faint_target,
            self.cfg.stage9_starmask_mid_target,
            self.cfg.stage9_starmask_bright_target,
            self.cfg.stage9_starmask_peak_target,
        ) = (
            faint_star_target,
            mid_star_target,
            bright_star_target,
            peak_star_target,
        )
        self.cfg.stage9_starmask_bright_chroma_max = max(
            float(self.cfg.stage9_starmask_bright_chroma_max),
            float(self.cfg.stage9_starmask_faint_chroma_max),
        )
        for attr_name, lower, upper in (
            ("stage9_mixed_star_weak_count_min", 4, 1000),
            ("stage9_mixed_star_bright_count_min", 1, 100),
        ):
            old_value = int(getattr(self.cfg, attr_name))
            new_value = _clamp_int(old_value, lower, upper)
            setattr(self.cfg, attr_name, new_value)
            if old_value != new_value:
                self.log.warn(
                    f"{attr_name} clamped: {old_value} -> {new_value}"
                )
        if old_stage7_retry != self.cfg.stage7_quality_retry_max:
            self.log.warn(
                "Stage7 quality retry max clamped: "
                f"{old_stage7_retry} -> {self.cfg.stage7_quality_retry_max}"
            )
        for label, old_value, new_value in (
            ("Stage7 large-galaxy halo threshold", old_stage7_large_galaxy_halo, self.cfg.stage7_large_galaxy_halo_residue_score_max),
            ("Stage7 bright-nebula halo threshold", old_stage7_bright_halo, self.cfg.stage7_bright_nebula_halo_residue_score_max),
            ("Stage7 starless repair strength", old_stage7_repair, self.cfg.stage7_starless_repair_strength),
            ("Stage7 starless halo repair strength", old_stage7_halo, self.cfg.stage7_starless_halo_repair_strength),
            ("Stage7 starless chroma denoise strength", old_stage7_chroma, self.cfg.stage7_starless_chroma_denoise_strength),
            ("Stage9 starmask asinh stretch", old_stage9_starmask_stretch, self.cfg.stage9_starmask_asinh_stretch),
            ("Stage9 starmask asinh offset", old_stage9_starmask_offset, self.cfg.stage9_starmask_asinh_offset),
        ):
            if old_value != new_value:
                self.log.warn(f"{label} clamped: {old_value} -> {new_value}")

    def _safe_unlink(self, path: Path):
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            self.log.debug(f"Unable to remove temp file {path.name}: {e}")


    def _measure_current_features(self) -> Optional[ImageFeatures]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            return measure_image_features(image_data)
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.warn(f"[AutoTune] Failed to measure image features: {e}")
            return None


    def _measure_current_quality(self) -> Optional[QualityMetrics]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            return measure_quality_metrics(image_data)
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.warn(f"[AutoTune] Failed to measure image quality metrics: {e}")
            return None


    def _read_image_by_stem(self, stem: str) -> Optional[np.ndarray]:
        try:
            self.cmd_with_check("load", stem)
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                return None
            return np.asarray(image_data)
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.warn(f"读取图像失败 ({stem}): {e}")
            return None


    def _set_current_image_pixeldata(self, image_data: np.ndarray, *, label: str) -> None:
        lock_factory = getattr(self.siril, "image_lock", None)
        if callable(lock_factory):
            with lock_factory():
                self.siril.set_image_pixeldata(image_data)
            return
        self.log.warn(f"{label}: image_lock unavailable, writing pixels without thread lock")
        self.siril.set_image_pixeldata(image_data)


    def _read_fits_header_metadata(self, *candidates: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        paths: List[Path] = []
        for candidate in candidates:
            if not candidate:
                continue
            path = candidate if isinstance(candidate, Path) else Path(str(candidate))
            if path.suffix.lower() not in {".fit", ".fits"}:
                path = path.with_suffix(".fit")
            if not path.is_absolute() and self.process_dir:
                path = self.process_dir / path
            paths.append(path)
        for path in paths:
            try:
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    while True:
                        block = handle.read(2880)
                        if not block:
                            break
                        for offset in range(0, len(block), 80):
                            card = block[offset:offset + 80].decode("ascii", errors="ignore")
                            key = card[:8].strip()
                            if key == "END":
                                metadata["_header_source"] = str(path)
                                return metadata
                            if not key or card[8:10] != "= ":
                                continue
                            raw = card[10:80].split("/", 1)[0].strip()
                            if raw.startswith("'") and "'" in raw[1:]:
                                value: Any = raw[1:raw.find("'", 1)]
                            else:
                                try:
                                    value = int(raw)
                                except ValueError:
                                    try:
                                        value = float(raw)
                                    except ValueError:
                                        value = raw
                            metadata[key] = value
            except OSError as e:
                self.log.debug(f"Unable to read FITS header {path}: {e}")
        return metadata


    def _resolve_input_profile(self) -> InputProfile:
        """Resolve transfer-function state before any linear-only stage runs."""
        source_candidates = [
            getattr(self, "source_file", None),
            getattr(self, "linear_intermediate_path", None),
        ]
        if self.process_dir:
            source_candidates.extend(
                [
                    self.process_dir / "stage2_corrected.fit",
                    self.process_dir / "stage1_prepared.fit",
                    self.process_dir / "working.fit",
                ]
            )
        source_path = next(
            (
                Path(candidate)
                for candidate in source_candidates
                if candidate and Path(candidate).is_file()
            ),
            None,
        )
        metadata = self._read_fits_header_metadata(*source_candidates)
        image_data = None
        try:
            getter = getattr(self.siril, "get_image_pixeldata", None)
            if callable(getter):
                image_data = getter(preview=False)
        except (
            AttributeError,
            CommandError,
            DataError,
            SirilError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            self.log.warn(f"输入状态像素采样失败，将依赖其他证据: {error}")

        profile = infer_input_profile(
            input_mode=str(getattr(self, "_stage1_input_mode", "unknown")),
            source_path=source_path,
            metadata=metadata,
            image_data=image_data,
            trusted_provenance=getattr(
                self,
                "_trusted_input_provenance",
                None,
            ),
        )
        self.input_profile = profile.to_dict()
        self._write_stage_json("input_profile.json", self.input_profile)
        self.log.info(
            "[InputProfile] "
            f"state={profile.state.value} confidence={profile.confidence:.2f} "
            f"source={profile.source} "
            f"linear_safe={str(profile.safe_for_linear_steps).lower()}"
        )
        for conflict in profile.conflicts:
            self.log.warn(f"[InputProfile] conflict: {conflict}")
        if profile.requires_review:
            self.log.warn(
                "[InputProfile] 线性状态不可信；将跳过线性/去星链并仅生成复核输出"
            )
        return profile


    def _load_trusted_input_provenance_for_resume(self) -> Dict[str, Any]:
        """Resolve resume trust before the process directory is rebuilt."""
        self._resume_semantic_context = None
        self._resume_semantic_context_status = "not_applicable"
        result: Dict[str, Any] = {
            "verified": False,
            "state": "unknown",
            "detail": "current input mode is not a resume checkpoint",
        }
        if not self.work_dir:
            self._trusted_input_provenance = result
            return result

        task_manifest_value = str(
            os.getenv(ENV_TASK_RUN_MANIFEST_KEY, "") or ""
        ).strip()
        if task_manifest_value:
            manifest_path = Path(task_manifest_value).expanduser().resolve()
            expected_stage = {
                INPUT_MODE_STAGE1_PREPARED_RESUME: 1,
                INPUT_MODE_STAGE2_CORRECTED_RESUME: 2,
                INPUT_MODE_LINEAR_RESUME: 5,
            }.get(self.input_mode)
            task_result: Dict[str, Any] = {
                "verified": False,
                "state": "unknown",
                "manifest_path": str(manifest_path),
                "detail": "task-run resume record is invalid",
            }
            payload = run_manifest.load_json(manifest_path)
            if manifest_path.parent != self.work_dir.resolve():
                task_result["detail"] = "task-run manifest is outside current run"
            elif payload is None or str(payload.get("schema") or "") != (
                "starun.task-run.v1"
            ):
                task_result["detail"] = "task-run manifest is missing or unsupported"
            else:
                expected_hash = str(payload.get("manifest_hash") or "")
                unsigned = dict(payload)
                unsigned.pop("manifest_hash", None)
                manifest_valid = bool(
                    expected_hash
                    and expected_hash == run_manifest.canonical_payload_hash(unsigned)
                )
                contract = payload.get("pipeline_contract")
                contract_valid = bool(
                    isinstance(contract, Mapping)
                    and str(contract.get("schema") or "")
                    == stage_contracts.PIPELINE_CONTRACT_SCHEMA
                    and str(contract.get("version") or "")
                    == stage_contracts.PIPELINE_CONTRACT_VERSION
                )
                resume = payload.get("resume")
                if not manifest_valid:
                    task_result["detail"] = "task-run manifest hash is invalid"
                elif not contract_valid:
                    task_result["detail"] = "task-run pipeline contract is incompatible"
                elif expected_stage is None or not isinstance(resume, Mapping):
                    task_result["detail"] = "task-run has no resume record for this mode"
                else:
                    try:
                        resume_stage = int(resume.get("stage"))
                    except (TypeError, ValueError):
                        resume_stage = 0
                    contract_record = (
                        stage_contracts.stage_contract(resume_stage)
                        if resume_stage in stage_contracts.FORMAL_RESUME_STAGES
                        else None
                    )
                    checkpoint_path = Path(
                        str(resume.get("path") or "")
                    ).expanduser().resolve()
                    task_root = Path(
                        str(payload.get("task_directory") or "")
                    ).expanduser().resolve()
                    try:
                        checkpoint_path.relative_to(task_root / "checkpoints")
                        path_inside_task = True
                    except ValueError:
                        path_inside_task = False
                    actual_sha256 = run_manifest.sha256_file(checkpoint_path)
                    expected_sha256 = str(resume.get("sha256") or "")
                    semantic_context = None
                    semantic_error = None
                    if resume_stage in {2, 5}:
                        try:
                            semantic_context = (
                                task_workspace._normalize_resume_semantic_context(
                                    resume.get("semantic_context"),
                                    stage_number=resume_stage,
                                )
                            )
                        except task_workspace.WorkspaceError as error:
                            semantic_error = str(error)
                        if semantic_context is None and semantic_error is None:
                            semantic_error = (
                                f"Stage {resume_stage} semantic context is missing"
                            )
                    if resume_stage != expected_stage or contract_record is None:
                        task_result["detail"] = "task-run resume stage does not match mode"
                    elif str(resume.get("artifact") or "") != (
                        contract_record.primary_artifact
                    ) or checkpoint_path.name != contract_record.primary_artifact:
                        task_result["detail"] = "task-run resume artifact violates contract"
                    elif not path_inside_task:
                        task_result["detail"] = "task-run checkpoint is outside task"
                    elif str(resume.get("state") or "").lower() != "linear":
                        task_result["detail"] = "task-run checkpoint state is not linear"
                    elif not actual_sha256 or actual_sha256 != expected_sha256:
                        task_result["detail"] = "task-run checkpoint SHA-256 mismatch"
                    elif semantic_error is not None:
                        task_result["detail"] = (
                            f"task-run Stage {resume_stage} semantic context is invalid: "
                            + semantic_error
                        )
                    else:
                        self._task_run_manifest_payload = copy.deepcopy(dict(payload))
                        self._task_run_manifest_path = manifest_path
                        task_result.update(
                            {
                                "verified": True,
                                "state": "linear",
                                "checkpoint": f"stage{resume_stage}",
                                "input_path": str(checkpoint_path),
                                "actual_sha256": actual_sha256,
                                "run_manifest_hash": resume.get(
                                    "run_manifest_hash"
                                ),
                                "detail": (
                                    "task-run manifest, stage contract, and "
                                    "checkpoint SHA-256 match"
                                ),
                            }
                        )
                        self._task_resume_checkpoint_path = checkpoint_path
                        if resume_stage in {2, 5}:
                            self._resume_semantic_context = copy.deepcopy(
                                semantic_context
                            )
                            self._resume_semantic_context_status = "verified"
                            task_result["semantic_context_status"] = "verified"
                            task_result["semantic_context"] = copy.deepcopy(
                                semantic_context
                            )
            self._trusted_input_provenance = task_result
            if task_result.get("verified"):
                self.log.info(
                    "[InputProfile] verified task resume provenance: "
                    f"{task_result.get('detail')}"
                )
            else:
                self.log.warn(
                    "[InputProfile] task resume provenance not trusted: "
                    f"{task_result.get('detail')}"
                )
            if expected_stage is not None and not task_result.get("verified"):
                raise RuntimeError(
                    "正式断点续跑校验失败："
                    + str(task_result.get("detail") or "unknown error")
                )
            return task_result
        if self.input_mode in {
            INPUT_MODE_STAGE1_PREPARED_RESUME,
            INPUT_MODE_STAGE2_CORRECTED_RESUME,
            INPUT_MODE_LINEAR_RESUME,
        }:
            raise RuntimeError(
                "续跑仅接受已验签 task-run manifest 中的 Stage 1/2/5 正式断点"
            )
        self._trusted_input_provenance = result
        return result


    def _apply_trusted_resume_semantics(self) -> bool:
        """Restore Stage 1-5 meaning after fresh image profiling on resume."""
        expected_stage = {
            INPUT_MODE_STAGE2_CORRECTED_RESUME: 2,
            INPUT_MODE_LINEAR_RESUME: 5,
        }.get(self.input_mode)
        if expected_stage is None:
            return False
        context = getattr(self, "_resume_semantic_context", None)
        if (
            isinstance(context, Mapping)
            and str(context.get("schema") or "")
            == task_workspace.RESUME_SEMANTIC_SCHEMA
            and int(context.get("checkpoint_stage", 0) or 0) == expected_stage
        ):
            review_requirements = context.get("review_requirements") or []
            if not isinstance(review_requirements, list):
                raise RuntimeError(
                    f"已验签 Stage {expected_stage} 复核语义无效"
                )
            for requirement in review_requirements:
                if not isinstance(requirement, Mapping):
                    continue
                self._require_review(
                    int(requirement.get("stage", 0) or 0),
                    str(requirement.get("code") or ""),
                    requirement.get("details")
                    if isinstance(requirement.get("details"), Mapping)
                    else {},
                )
            self._stage2_view_review_required = bool(
                self._stage_review_reasons(2)
            )
            self._background_review_required = bool(
                self._stage_review_reasons(3)
            )
            self._stage4_color_review_required = bool(
                self._stage_review_reasons(4)
            )

            if expected_stage == 2:
                stage2_crop = context.get("stage2_crop")
                if not isinstance(stage2_crop, Mapping):
                    raise RuntimeError("已验签 Stage 2 语义缺少裁切契约")
                restored_crop = copy.deepcopy(dict(stage2_crop))
                restored_crop.update(
                    {
                        "mode": "verified_stage2_checkpoint_resume",
                        "original_shape": copy.deepcopy(
                            dict(stage2_crop.get("original_dimensions") or {})
                        ),
                        "current_shape": copy.deepcopy(
                            dict(stage2_crop.get("final_dimensions") or {})
                        ),
                        "final_shape": copy.deepcopy(
                            dict(stage2_crop.get("final_dimensions") or {})
                        ),
                        "total_crop": copy.deepcopy(
                            dict(stage2_crop.get("cumulative_crop") or {})
                        ),
                        "requires_review": bool(self._stage_review_reasons(2)),
                        "field_rotation": {
                            "actual_passes": int(
                                stage2_crop.get("field_rotation_passes", 0) or 0
                            ),
                            "max_passes": int(
                                stage2_crop.get("field_rotation_max_passes", 2)
                                or 2
                            ),
                            "residual": copy.deepcopy(
                                dict(
                                    stage2_crop.get("final_residual_detection")
                                    or {}
                                )
                            ),
                        },
                    }
                )
                self.stage2_crop_report = restored_crop
                self._resume_semantic_context_status = "restored"
                self.log.info(
                    "[ResumeSemantics] restored Stage 2 crop contract: "
                    f"passes={stage2_crop.get('field_rotation_passes', 0)}, "
                    f"reviews={len(self._stage_review_reasons(2))}"
                )
                return True

            channel_semantics = str(
                context.get("channel_semantics") or "unknown"
            )
            channel_profile = context.get("channel_profile") or {}
            mapping_context = context.get("narrowband_channel_mapping")
            if not isinstance(mapping_context, Mapping) and isinstance(
                channel_profile,
                Mapping,
            ):
                mapping_context = channel_profile.get("narrowband_mapping")
            target_profile = context.get("target_profile") or {}
            pipeline_policy = context.get("pipeline_policy") or {}
            color_report = context.get("color_calibration_report") or {}
            stage5_star_reference_report = context.get(
                "stage5_star_reference_report"
            ) or {}
            if not all(
                isinstance(value, Mapping)
                for value in (
                    channel_profile,
                    target_profile,
                    pipeline_policy,
                    color_report,
                )
            ):
                raise RuntimeError(
                    "已验签 Stage 5 语义契约字段不完整"
                )
            else:
                self._channel_semantics = channel_semantics
                self.channel_profile = copy.deepcopy(dict(channel_profile))
                mapping_missing = not isinstance(mapping_context, Mapping) or not bool(
                    mapping_context
                )
                if mapping_missing:
                    raise RuntimeError("已验签 Stage 5 语义缺少通道映射契约")
                restored_mapping = copy.deepcopy(dict(mapping_context))
                self.narrowband_channel_mapping = restored_mapping
                self.channel_profile["narrowband_mapping"] = copy.deepcopy(
                    restored_mapping
                )
                self.target_profile = copy.deepcopy(dict(target_profile))
                self.pipeline_policy = copy.deepcopy(dict(pipeline_policy))
                self.color_calibration_report = copy.deepcopy(dict(color_report))
                self._stage5_star_reference_report = copy.deepcopy(
                    dict(stage5_star_reference_report)
                    if isinstance(stage5_star_reference_report, Mapping)
                    else {}
                )
                self.color_calibration_report["channel_mapping"] = copy.deepcopy(
                    restored_mapping
                )
                self._resume_semantic_context_status = "restored"
                physical = self.color_calibration_report.get("physical_color") or {}
                physical_accepted = bool(
                    isinstance(physical, Mapping)
                    and physical.get("accepted", False)
                )
                self.log.info(
                    "[ResumeSemantics] restored Stage 5 upstream contract: "
                    f"channel={channel_semantics}, "
                    f"color_method={self.color_calibration_report.get('method')}, "
                    f"channel_mapping={restored_mapping.get('mapping')}, "
                    f"physical_color_accepted={str(physical_accepted).lower()}"
                )
                return True

        raise RuntimeError(
            f"Stage {expected_stage} 续跑缺少已验签语义契约"
        )


    def _processing_plan_input_path(self) -> Optional[Path]:
        for candidate in (
            getattr(self, "source_file", None),
            getattr(self, "linear_intermediate_path", None),
            self.process_dir / "working.fit" if self.process_dir else None,
        ):
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    def _load_task_processing_parameters(self) -> None:
        """Load and verify the frozen GUI parameter payload, if present."""
        self._task_run_manifest_payload = None
        self._task_run_manifest_path = None
        self._task_processing_parameters = default_processing_parameters()
        self._task_processing_parameter_request = copy.deepcopy(
            self._task_processing_parameters
        )
        self._task_processing_parameter_adjustments = []
        self._task_manual_override_fields = ()
        self._task_gate_profile_audit = processing_gate_profile_audit(
            self._task_processing_parameters
        )
        configured = str(os.getenv(ENV_TASK_RUN_MANIFEST_KEY, "") or "").strip()
        if not configured:
            return
        manifest_path = Path(configured).expanduser().resolve()
        try:
            payload = run_manifest.load_json(manifest_path)
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"任务运行清单无法读取：{error}") from error
        if not isinstance(payload, Mapping):
            raise RuntimeError("任务运行清单缺失或不是有效 JSON 映射")
        if payload.get("schema") != "starun.task-run.v1":
            raise RuntimeError("任务运行清单 schema 不受支持")
        claimed_hash = str(payload.get("manifest_hash") or "")
        unsigned = dict(payload)
        unsigned.pop("manifest_hash", None)
        if not claimed_hash or claimed_hash != run_manifest.canonical_payload_hash(
            unsigned
        ):
            raise RuntimeError("任务运行清单签名校验失败")
        self._task_run_manifest_payload = copy.deepcopy(dict(payload))
        self._task_run_manifest_path = manifest_path
        raw_parameters = payload.get("processing_parameters")
        if raw_parameters is None:
            raise RuntimeError("任务运行清单缺少 v4 处理参数")
        self._task_processing_parameter_request = copy.deepcopy(raw_parameters)
        try:
            normalized, runtime_adjustments = normalize_processing_parameters(
                raw_parameters,
                validate_paths=True,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"任务处理参数无效：{error}") from error
        frozen_adjustments = payload.get("processing_parameter_adjustments", [])
        if not isinstance(frozen_adjustments, list):
            raise RuntimeError("任务处理参数调整记录格式无效")
        self._task_processing_parameters = normalized
        self._task_gate_profile_audit = processing_gate_profile_audit(normalized)
        frozen_gate_profile = payload.get("processing_gate_profile")
        if frozen_gate_profile is not None:
            if not isinstance(frozen_gate_profile, Mapping) or (
                run_manifest.canonical_payload_hash(dict(frozen_gate_profile))
                != run_manifest.canonical_payload_hash(
                    self._task_gate_profile_audit
                )
            ):
                raise RuntimeError("任务门禁档位审计与签名参数不一致")
        self._task_processing_parameter_adjustments = [
            *[dict(item) for item in frozen_adjustments if isinstance(item, Mapping)],
            *runtime_adjustments,
        ]
        general = normalized["general"]
        self.cfg.output_format = ",".join(general["output_formats"])
        self.cfg.auto_tune_enabled = bool(general["auto_tune_enabled"])
        self.cfg.max_retries = int(general["max_retries"])
        self.cfg.retry_delay = float(general["retry_delay"])
        self.cfg.review_bundle_enabled = bool(general["review_bundle_enabled"])
        self.cfg.stage10_managed_output_enabled = bool(
            general["managed_output_enabled"]
        )
        self.cfg.checkpoint_mode = bool(general["checkpoint_mode"])
        self.cfg.force_review_only_output = bool(
            general["review_only"]
            or gate_profile_requires_review(normalized["gate_profile"])
        )
        accelerated = "0" if general["compute_mode"] == "cpu" else "1"
        for env_key in (
            "STARUN_COSMIC_NATIVE_GPU",
            "STARUN_COSMIC_CLASSIC_GPU",
            "STARUN_GRAXPERT_GPU",
        ):
            os.environ[env_key] = accelerated
        for record in self._task_processing_parameter_adjustments:
            self.log.warn(
                "[ProcessingParameters] safe clamp "
                f"{record.get('field')}: {record.get('requested')} -> "
                f"{record.get('effective')} "
                f"({record.get('reason', 'safe_range')})"
            )
        self.log.info(
            "[ProcessingParameters] verified frozen task parameters "
            f"manifest={manifest_path.name}; "
            f"general=output:{self.cfg.output_format},"
            f"review_only:{self.cfg.force_review_only_output},"
            f"compute:{general['compute_mode']},"
            f"auto_tune:{self.cfg.auto_tune_enabled},"
            f"retries:{self.cfg.max_retries},"
            f"retry_delay:{self.cfg.retry_delay:g},"
            f"review_bundle:{self.cfg.review_bundle_enabled},"
            f"managed_output:{self.cfg.stage10_managed_output_enabled},"
            f"checkpoint_mode:{self.cfg.checkpoint_mode}"
        )
        self.log.info(
            "[GateProfile] verified "
            f"profile={self._task_gate_profile_audit['profile']} "
            f"multiplier={self._task_gate_profile_audit['multiplier']:g} "
            f"managed={self._task_gate_profile_audit['managed_field_count']} "
            f"forced_review={self._task_gate_profile_audit['forced_review_only']}"
        )

    def _apply_task_processing_parameter_overrides(self) -> None:
        """Apply the signed static gate profile, then expert task overrides."""
        payload = getattr(self, "_task_processing_parameters", None)
        if not isinstance(payload, Mapping):
            return
        normalized, adjustments, manual_fields = (
            apply_processing_parameters_to_config(self.cfg, payload)
        )
        self._task_processing_parameters = normalized
        self._task_manual_override_fields = manual_fields
        self._task_gate_profile_audit = processing_gate_profile_audit(normalized)
        for record in self._task_gate_profile_audit["fields"]:
            if record.get("physical_clamped"):
                self.log.warn(
                    "[GateProfile] physical clamp "
                    f"{record.get('field')}: "
                    f"{record.get('profile_requested')} -> "
                    f"{record.get('profile_effective')}"
                )
        if adjustments:
            self._task_processing_parameter_adjustments.extend(adjustments)
        if manual_fields:
            self.log.info(
                "[ProcessingParameters] applied after auto tune: "
                + ", ".join(manual_fields)
            )
        for record in adjustments:
            self.log.warn(
                "[ProcessingParameters] safe clamp "
                f"{record.get('field')}: {record.get('requested')} -> "
                f"{record.get('effective')}"
            )

    def _resolve_channel_profile(self, profile: InputProfile) -> Dict[str, Any]:
        """Resolve physical channel meaning before the processing plan is frozen."""
        resume_context = getattr(self, "_resume_semantic_context", None)
        if (
            self.input_mode == INPUT_MODE_LINEAR_RESUME
            and getattr(self, "_resume_semantic_context_status", "") == "restored"
            and isinstance(resume_context, Mapping)
        ):
            frozen_profile = resume_context.get("channel_profile") or {}
            frozen_kind = str(
                resume_context.get("channel_semantics") or "unknown"
            )
            if isinstance(frozen_profile, Mapping) and frozen_kind:
                channel_profile = copy.deepcopy(dict(frozen_profile))
                channel_profile["kind"] = frozen_kind
                channel_profile["source"] = "signed_stage5_resume_semantics"
                frozen_mapping = resume_context.get("narrowband_channel_mapping")
                if not isinstance(frozen_mapping, Mapping) or not frozen_mapping:
                    raise RuntimeError(
                        "已验签 Stage 5 语义缺少通道映射契约"
                    )
                self.narrowband_channel_mapping = copy.deepcopy(
                    dict(frozen_mapping)
                )
                channel_profile["narrowband_mapping"] = copy.deepcopy(
                    self.narrowband_channel_mapping
                )
                self.channel_profile = channel_profile
                self._channel_semantics = frozen_kind
                self.log.info(
                    "[ChannelProfile] restored signed Stage 5 semantics "
                    f"kind={frozen_kind}"
                )
                return channel_profile
        try:
            shape = channel_shape_dict(self.siril.get_image_shape())
        except (
            AttributeError,
            CommandError,
            SirilError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            shape = {}
        metadata = self._read_fits_header_metadata(
            "stage2_corrected",
            "stage1_prepared",
            "working",
            getattr(self, "source_file", None),
        )
        channel_profile = classify_channel_semantics(
            channels=int(shape.get("channels", 0) or 0),
            metadata=metadata,
            input_state=profile.state.value,
            explicit_filter_hint=(
                ""
                if str(getattr(self.cfg, "stage4_filter_hint", "auto")).strip().lower()
                == "auto"
                else str(getattr(self.cfg, "stage4_filter_hint", ""))
            ),
            target_profile=(
                self.target_profile
                if isinstance(getattr(self, "target_profile", None), dict)
                else None
            ),
        )
        channel_profile["shape"] = shape
        self.channel_profile = channel_profile
        self._channel_semantics = str(channel_profile["kind"])
        self.log.info(
            "[ChannelProfile] "
            f"kind={self._channel_semantics} "
            f"confidence={float(channel_profile['confidence']):.2f} "
            f"action={channel_profile['action']}"
        )
        return channel_profile


    def _write_processing_plan(self, profile: InputProfile) -> bool:
        """Freeze this run's resolved route before post-processing transforms."""
        if not self.work_dir:
            return False
        task_manifest = getattr(self, "_task_run_manifest_payload", None)
        task_manifest = (
            dict(task_manifest) if isinstance(task_manifest, Mapping) else {}
        )
        self._run_id = str(task_manifest.get("run_id") or uuid.uuid4())
        input_path = self._processing_plan_input_path()
        input_record: Dict[str, Any] = {
            "path": str(input_path) if input_path else None,
            "input_mode": self.input_mode,
            "stage1_input_mode": getattr(self, "_stage1_input_mode", "unknown"),
            "profile": profile.to_dict(),
        }
        if input_path:
            input_record.update(run_manifest.file_record(input_path))
        source_fingerprint = str(task_manifest.get("source_fingerprint") or "")
        input_record["fingerprint"] = source_fingerprint or str(
            input_record.get("sha256") or ""
        )
        freeze_primary = getattr(self, "_freeze_primary_target", None)
        if callable(freeze_primary):
            freeze_primary()
        channel_profile = self._resolve_channel_profile(profile)
        target_profile = dict(getattr(self, "target_profile", {}) or {})
        try:
            palette_selection = resolve_palette_selection(
                target_profile.get("primary_target"),
                getattr(
                    self.cfg,
                    "stage8_dualband_palette_selection",
                    "auto",
                ),
            )
        except ValueError as error:
            self._stage8_palette_selection = {}
            self.log.warn(f"Stage 8 哈勃色方案无效，处理计划拒绝冻结: {error}")
            return False
        self._stage8_palette_selection = copy.deepcopy(palette_selection)
        resolved_policy = dict(getattr(self, "pipeline_policy", {}) or {})
        stage3_policy = dict(resolved_policy.get("stage3_background") or {})
        stretch_policy = dict(resolved_policy.get("stage7_stretch") or {})
        configured_remix_levels = getattr(
            self.cfg,
            "stage9_fallback_intensity_levels",
            (),
        )
        if not isinstance(configured_remix_levels, (list, tuple)):
            configured_remix_levels = ()
        remix_levels: List[float] = []
        for raw_level in configured_remix_levels:
            try:
                remix_levels.append(float(raw_level))
            except (TypeError, ValueError):
                continue
        frozen_processing_parameters = dict(
            getattr(
                self,
                "_task_processing_parameters",
                default_processing_parameters(),
            )
        )
        gate_profile_audit = copy.deepcopy(
            getattr(
                self,
                "_task_gate_profile_audit",
                processing_gate_profile_audit(frozen_processing_parameters),
            )
        )
        gate_profile_records_by_stage: Dict[int, list[Mapping[str, Any]]] = {}
        for record in gate_profile_audit.get("fields", []):
            if not isinstance(record, Mapping):
                continue
            try:
                record_stage = int(record.get("stage"))
            except (TypeError, ValueError):
                continue
            gate_profile_records_by_stage.setdefault(record_stage, []).append(record)
        processing_parameter_states: Dict[str, Any] = {}
        frozen_stages = frozen_processing_parameters.get("stages", {})
        for stage, specs in SPECS_BY_STAGE.items():
            entry = (
                frozen_stages.get(str(stage), {})
                if isinstance(frozen_stages, Mapping)
                else {}
            )
            entry = entry if isinstance(entry, Mapping) else {}
            overrides = entry.get("overrides", {})
            overrides = overrides if isinstance(overrides, Mapping) else {}
            mode = str(entry.get("mode", "auto") or "auto")
            mode_spec = next((spec for spec in specs if spec.stage_mode), None)
            visible_fields = tuple(
                spec.field
                for spec in specs
                if not spec.stage_mode
            )
            processing_parameter_states[str(stage)] = {
                "mode_field": mode_spec.field if mode_spec is not None else None,
                "mode": mode,
                "mode_state": "automatic" if mode == "auto" else "custom",
                "custom_fields": sorted(str(field) for field in overrides),
                "gate_profile_fields": sorted(
                    str(record.get("field"))
                    for record in gate_profile_records_by_stage.get(stage, [])
                    if record.get("source") == "gate_profile"
                ),
                "expert_gate_override_fields": sorted(
                    str(record.get("field"))
                    for record in gate_profile_records_by_stage.get(stage, [])
                    if record.get("source") == "expert_override"
                ),
                "automatic_fields": sorted(
                    field for field in visible_fields if field not in overrides
                ),
            }
        resume = task_manifest.get("resume")
        resume_stage = None
        if isinstance(resume, Mapping):
            try:
                resume_stage = int(resume.get("stage"))
            except (TypeError, ValueError):
                return False
        input_trust = (
            task_plan.InputTrust.VERIFIED
            if resume_stage is not None
            else (
                task_plan.InputTrust.RECOGNIZED
                if profile.safe_for_linear_steps
                else task_plan.InputTrust.REVIEW_REQUIRED
            )
        )
        metadata: Dict[str, Any] = {
            "software": self._processing_software_identity(),
            "task_run_manifest_hash": task_manifest.get("manifest_hash"),
            "input_profile": profile.to_dict(),
            "channel_semantics": str(
                getattr(self, "_channel_semantics", "unknown") or "unknown"
            ),
            "channel_profile": dict(channel_profile),
            "narrowband_channel_mapping": dict(
                getattr(self, "narrowband_channel_mapping", {}) or {}
            ),
            "target_profile": run_manifest.redact_sensitive(target_profile),
            "pipeline_policy": run_manifest.redact_sensitive(resolved_policy),
            "target": {
                "primary": target_profile.get("target_type"),
                "primary_record": target_profile.get("primary_target", {}),
                "secondary": target_profile.get("secondary_labels", []),
                "confidence": target_profile.get("target_confidence"),
                "policy": resolved_policy.get("policy_name"),
                "star_separation_basis": "primary_target_only",
            },
            "candidate_contracts": {
                "stage3_background": {
                    "model_priority": list(stage3_policy.get("model_priority") or []),
                    "runtime_capability_probe": True,
                    "sample_source": "custom_safe_points",
                    "sample_safety_gates": [
                        "low_signal",
                        "low_texture",
                        "star_rejection",
                        "spatial_coverage",
                    ],
                    "subsky_requires_existing_samples": True,
                    "pattern_noise_routes": [
                        "low_frequency_gradient",
                        "pattern_noise_deferred",
                        "mixed_gradient_and_pattern_noise",
                    ],
                    "pattern_noise_branches": [
                        "banding_review",
                        "walking_noise_review",
                        "directional_pattern_review",
                    ],
                    "plugins_are_fallbacks": True,
                },
                "stage7_stretch": {
                    "allowed_modes": list(stretch_policy.get("candidate_mode") or []),
                    "fallback_candidate": stretch_policy.get("fallback_candidate"),
                    "parameters_owned_by": "code",
                    "selector": "deterministic_quality_rank",
                    "selection_after_hard_gates": True,
                },
                "stage6_star_separation": {
                    "candidate_ids": ["syqon_standard"],
                    "parameters_owned_by": "code",
                    "selector": "fixed_offline_profile",
                    "runtime_profile": "zenith_baseline",
                },
                "stage8_enhancement": {
                    "candidate_ids": [
                        "preserve",
                        "conservative",
                        "balanced",
                        "detail_preserving",
                    ],
                    "parameters_owned_by": "code",
                    "selector": "deterministic_local_quality",
                    "dualband_palette": {
                        "enabled": bool(
                            getattr(
                                self.cfg,
                                "stage8_dualband_palette_enabled",
                                True,
                            )
                        ),
                        "role": "artistic_false_color",
                        "selection": palette_selection,
                        "available_palette_ids": list(PALETTE_CHANNELS),
                        "runtime_requirements": [
                            "confirmed_ha_oiii_mapping",
                            "stage7_starless_accepted",
                            "stage8_full_policy",
                            "stage8_structural_quality_ok",
                            "no_unverified_external_starless",
                            "degraded_pcc_parent_allowed",
                        ],
                    },
                },
                "stage9_star_remix": {
                    "primary_id": "primary",
                    "fallback_intensity_levels": remix_levels,
                    "ids_generated_after_quality_scaling": True,
                },
            },
            "capabilities": {
                "offline_first": True,
            },
            "processing_parameters": run_manifest.redact_sensitive(
                frozen_processing_parameters
            ),
            "processing_gate_profile": run_manifest.redact_sensitive(
                gate_profile_audit
            ),
            "processing_parameter_request": run_manifest.redact_sensitive(
                copy.deepcopy(
                    getattr(
                        self,
                        "_task_processing_parameter_request",
                        frozen_processing_parameters,
                    )
                )
            ),
            "processing_parameter_states": processing_parameter_states,
            "manual_override_fields": list(
                getattr(self, "_task_manual_override_fields", ())
            ),
            "parameter_adjustments": list(
                getattr(self, "_task_processing_parameter_adjustments", [])
            ),
            "config": run_manifest.redact_sensitive(asdict(self.cfg)),
            "stage_policies": {
                str(number): {
                    "failure_action": self._stage_failure_action(number)
                }
                for number in range(2, 11)
            },
        }
        try:
            plan = task_plan.build_processing_plan(
                run_id=self._run_id,
                generated_at=run_manifest.utc_timestamp(),
                input_record=input_record,
                input_state=profile.state.value,
                input_trust=input_trust,
                resume_after_stage=resume_stage,
                checkpoint_fingerprints=(
                    task_manifest.get("checkpoint_fingerprints")
                    if task_manifest
                    else None
                ),
                output={
                    "configured_formats": str(
                        getattr(self.cfg, "output_format", "") or ""
                    ),
                    "review_only": bool(
                        profile.requires_review
                        or getattr(self.cfg, "force_review_only_output", False)
                    ),
                },
                metadata=metadata,
            )
        except (TypeError, ValueError) as error:
            self.log.warn(f"processing-plan.v2 构建失败: {error}")
            return False
        verification = task_plan.verify_processing_plan(plan)
        if not verification.get("verified"):
            self.log.warn(
                "processing-plan.v2 校验失败: "
                + str(verification.get("detail") or "unknown error")
            )
            return False
        self._processing_plan = plan
        self._processing_plan_hash = str(plan["plan_hash"])

        try:
            run_manifest.atomic_write_json(
                self.work_dir / "processing-plan.json",
                plan,
            )
            self.log.info(
                "[ProcessingPlan] frozen "
                f"run_id={self._run_id} hash={self._processing_plan_hash[:12]}"
            )
            return True
        except (OSError, TypeError, ValueError) as error:
            self.log.warn(f"processing-plan.json 写入失败: {error}")
            return False


    @staticmethod
    def _result_stage_number(result: StageResult) -> int:
        match = re.match(r"^阶段\s+(\d+)\s*:", str(result.name).strip())
        return int(match.group(1)) if match else 0

    def _actual_steps_payload(self) -> List[Dict[str, Any]]:
        return [
            {
                "stage": self._result_stage_number(result),
                "name": result.name,
                "status": result.status,
                "display_status": result.display_status,
                "execution": result.execution,
                "fallback_used": result.fallback_used,
                "upstream_passthrough": result.upstream_passthrough,
                "reason_code": result.reason_code or None,
                "review_required": bool(result.review_reasons),
                "review_reasons": list(result.review_reasons),
                "issues": list(result.issues),
                "duration_seconds": float(result.duration),
                "message": result.message,
                "details": result.details,
                "components": result.components,
            }
            for result in self.results
        ]

    def _pipeline_outcome(self, failure_reason: Optional[str] = None) -> Dict[str, Any]:
        return outcome.summarize_outcome(
            self._actual_steps_payload(),
            self._review_requirements_payload(),
            failure_reason=failure_reason,
        )

    def _pipeline_result_status(self, failure_reason: Optional[str] = None) -> str:
        return str(self._pipeline_outcome(failure_reason)["status"])


    def _write_pipeline_result_manifest(
        self,
        *,
        failure_reason: Optional[str] = None,
    ) -> bool:
        """Persist actual steps, output hashes, fallbacks, and global status."""
        if not self.work_dir:
            return False
        plan = run_manifest.load_json(self.work_dir / "processing-plan.json")
        plan_verification = task_plan.verify_processing_plan(plan or {})
        if not plan_verification.get("verified"):
            self.log.warn(
                "pipeline-result.json 拒绝发布：processing plan 校验失败："
                + str(plan_verification.get("detail") or "unknown error")
            )
            return False
        plan_hash = str(plan.get("plan_hash") or "")
        if not plan_hash or plan_hash != str(
            getattr(self, "_processing_plan_hash", "") or ""
        ):
            self.log.warn("pipeline-result.json 拒绝发布：计划哈希引用不一致")
            return False
        if str(plan.get("run_id") or "") != str(getattr(self, "_run_id", "") or ""):
            self.log.warn("pipeline-result.json 拒绝发布：run_id 与处理计划不一致")
            return False
        outputs = run_manifest.collect_output_records(
            self.work_dir,
            output_basenames=getattr(self, "_final_output_basenames", ()),
            exported_after=getattr(self, "_final_export_started_at", None),
        )

        actual_steps = self._actual_steps_payload()
        review_requirements = self._review_requirements_payload()
        outcome_summary = outcome.summarize_outcome(
            actual_steps,
            review_requirements,
            failure_reason=failure_reason,
        )
        status = str(outcome_summary["status"])
        manifest: Dict[str, Any] = {
            "schema": outcome.PIPELINE_RESULT_SCHEMA_V2,
            "run_id": getattr(self, "_run_id", None),
            "generated_at": run_manifest.utc_timestamp(),
            "status": status,
            "failure_reason": failure_reason,
            "plan_hash": plan_hash,
            "input_profile": dict(getattr(self, "input_profile", {}) or {}),
            "channel_semantics": str(
                getattr(self, "_channel_semantics", "unknown") or "unknown"
            ),
            "narrowband_channel_mapping": dict(
                getattr(self, "narrowband_channel_mapping", {}) or {}
            ),
            "target_profile": dict(getattr(self, "target_profile", {}) or {}),
            "review_requirements": review_requirements,
            "color_calibration": {
                "status": str(
                    (getattr(self, "color_calibration_report", {}) or {}).get(
                        "status",
                        "not_run",
                    )
                ),
                "method": (getattr(self, "color_calibration_report", {}) or {}).get(
                    "method"
                ),
                "requires_review": bool(
                    self._stage_review_reasons(4)
                    or self._stage_review_reasons(7)
                ),
                "stage7_forced_delivery": bool(
                    getattr(
                        self,
                        "_stage7_stretch_forced_delivery",
                        False,
                    )
                ),
                "stage7_forced_delivery_reasons": list(
                    getattr(
                        self,
                        "_stage7_forced_delivery_reasons",
                        [],
                    )
                    or []
                ),
                "background_color_review_gate": dict(
                    getattr(
                        self,
                        "_stage7_background_color_review_gate",
                        {},
                    )
                    or {}
                ),
                "physical_color": dict(
                    (getattr(self, "color_calibration_report", {}) or {}).get(
                        "physical_color",
                        {},
                    )
                    or {}
                ),
                "degraded_color_correction": dict(
                    (getattr(self, "color_calibration_report", {}) or {}).get(
                        "degraded_color_correction",
                        {},
                    )
                    or {}
                ),
                "auto_local_reference": dict(
                    (getattr(self, "color_calibration_report", {}) or {}).get(
                        "auto_local_reference",
                        {},
                    )
                    or {}
                ),
                "artistic_hoo": dict(
                    (getattr(self, "color_calibration_report", {}) or {}).get(
                        "artistic_hoo",
                        {},
                    )
                    or {}
                ),
            },
            "star_separation": {
                "state": getattr(self, "_star_separation_state", "pending"),
                "stars_required": bool(
                    getattr(self, "_stage9_stars_required", False)
                ),
                "stars_applied": bool(
                    getattr(self, "_stage9_stars_applied", False)
                ),
                "output_contains_stars": bool(
                    getattr(self, "_stage9_output_contains_stars", False)
                ),
                "output_withheld": bool(
                    getattr(self, "_stage9_output_withheld", False)
                ),
                "starmask_borderline_review_required": bool(
                    getattr(
                        self,
                        "_stage6_starmask_borderline_review_required",
                        False,
                    )
                ),
                "psf_review_required": bool(
                    getattr(self, "_stage9_psf_review_required", False)
                ),
                "remix_formally_accepted": bool(
                    getattr(self, "_stage9_remix_formally_accepted", False)
                ),
                "delivery_contract_accepted": bool(
                    getattr(
                        self,
                        "_stage9_star_delivery_contract_accepted",
                        False,
                    )
                ),
                "with_stars_hdr_fallback": copy.deepcopy(
                    getattr(self, "_bright_core_with_stars_fallback", {}) or {}
                ),
                "review_candidate_selected": bool(
                    getattr(self, "_stage9_review_candidate_selected", False)
                ),
                "final_source": str(
                    getattr(self, "_stage9_final_source", "") or ""
                ),
                "application_mode": getattr(
                    self,
                    "_stage9_stars_application_mode",
                    "pending",
                ),
            },
            "actual_steps": actual_steps,
            "stage_policy_events": list(
                getattr(self, "_stage_policy_events", []) or []
            ),
            "outputs": outputs,
            "retention": dict(
                getattr(self, "_checkpoint_retention_report", {}) or {}
            ),
        }
        manifest.update(
            {
                key: outcome_summary[key]
                for key in (
                    "had_errors",
                    "had_fatal_errors",
                    "had_degradations",
                    "had_fallbacks",
                    "review_required",
                    "issues",
                    "errors",
                    "outcome_counts",
                )
            }
        )
        manifest["manifest_hash"] = run_manifest.canonical_payload_hash(manifest)
        self._pipeline_result_manifest = manifest
        self._pipeline_result_global_status = status

        destinations = [self.work_dir / "pipeline-result.json"]
        if self.process_dir:
            destinations.append(self.process_dir / "pipeline-result.json")
        try:
            for destination in destinations:
                run_manifest.atomic_write_json(destination, manifest)
            self.log.info(
                "[PIPELINE_RESULT] "
                f"status={status} manifest={self.work_dir / 'pipeline-result.json'}"
            )
            return True
        except (OSError, TypeError, ValueError) as error:
            self.log.warn(f"pipeline-result.json 写入失败: {error}")
            return False


    def _write_stage_json(self, filename: str, payload: Dict[str, Any]) -> None:
        write_stage_json(self.process_dir, self.log, filename, payload)
