"""Service mixins for SeestarPostProcessor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ai_advisory
import cosmic_clarity
import plugin_runner
import sasp_runner
import scunet_denoise
import syqon_starless
import stage7_quality
import stage7_repair
import stage8_pixels
from channel_semantics import channel_shape_dict, classify_channel_semantics
from input_profile import infer_input_profile
import run_manifest
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
    Stage6StretchStrategy,
    StageResult,
    TargetType,
)
from save_utils import save_stage_output, write_ai_raw_response, write_stage_json

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except ImportError:
    CommandError = RuntimeError
    DataError = RuntimeError
    SirilError = RuntimeError

try:
    from image_feature_analyzer import analyze_image as analyze_adaptive_image
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from stretch_candidate_evaluator import (
        build_candidate_spec,
        candidate_modes,
        choose_best as choose_best_stretch_candidate,
        score_candidate as score_stretch_candidate,
    )
    from target_profiler import build_target_profile
except (ImportError, RuntimeError):
    analyze_adaptive_image = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage6_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_candidate_spec = None
    candidate_modes = None
    choose_best_stretch_candidate = None
    score_stretch_candidate = None
    build_target_profile = None

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "SEESTAR_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "SEESTAR_INPUT_MODE"
PROJECT_DEFAULT_ENV_RESOURCE_REL = Path("resources") / "default.env"
PROJECT_ENV_RESOURCE_REL = Path("resources") / "ai.env"
PROJECT_ENV_OVERRIDE_NAME = ".seestar_ai.env"
PROJECT_ENV_ALLOWED_KEYS = frozenset(
    {
        "SEESTAR_DEBUG_MODE",
        "SEESTAR_INPUT_MODE",
        "SEESTAR_OUTPUT_FORMAT",
        "SEESTAR_NETWORK_MODE",
        "SEESTAR_WORKFLOW_PLUGIN_PROBE",
        "SEESTAR_STAGE4_PLATESOLVE_ENABLE",
        "SEESTAR_STAGE4_PLATESOLVE_FOCAL",
        "SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE",
        "SEESTAR_STAGE4_PLATESOLVE_ORDER",
        "SEESTAR_STAGE4_PLATESOLVE_CATALOGS",
        "SEESTAR_STAGE4_PLATESOLVE_HEADER_RADIUS",
        "SEESTAR_STAGE4_AUTO_GEOMETRY_ENABLE",
        "SEESTAR_STAGE4_AUTO_GEOMETRY_CONFIDENCE_MIN",
        "SEESTAR_STAGE4_AUTO_GEOMETRY_SCALE_RESIDUAL_MAX",
        "SEESTAR_STAGE4_NBN_ENABLE",
        "SEESTAR_STAGE4_NBN_MAPPING_CONFIDENCE_MIN",
        "SEESTAR_STAGE4_NBN_STRENGTH",
        "SEESTAR_STAGE4_NBN_GAIN_LIMIT",
        "SEESTAR_STAGE4_NBN_LINE_RATIO_DRIFT_MAX",
        "SEESTAR_GAIA_ASTRO_CATALOG",
        "SEESTAR_STAGE4_FILTER_HINT",
        "SEESTAR_STAGE4_PCC_TIMEOUT_SEC",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT",
        "SEESTAR_STAGE4_LOCAL_STAR_MASK_RADIUS",
        "SEESTAR_STAGE4_LOCAL_STAR_MASK_COVERAGE_MAX",
        "SEESTAR_ABERRATION_API_ENABLE",
        "SEESTAR_ABERRATION_PROVIDER",
        "SEESTAR_OPTIONAL_COLOR_TRANSFORM",
        "SEESTAR_DENOISE_ENABLE",
        "SEESTAR_DENOISE_FORCE",
        "SEESTAR_STAGE5_MULTISCALE_DENOISE_ENABLE",
        "SEESTAR_STAGE5_MULTISCALE_DENOISE_STRENGTH",
        "SEESTAR_STAGE5_MULTISCALE_DETAIL_RETENTION_MIN",
        "SEESTAR_STAGE5_MULTISCALE_NOISE_REDUCTION_MIN",
        "SEESTAR_STAGE5_BUILTIN_DENOISE_MOD",
        "SEESTAR_STAGE5_DECONV_ENABLE",
        "SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE",
        "SEESTAR_STAGE5_RL_MAXSTARS",
        "SEESTAR_STAGE5_RL_PSF_KS",
        "SEESTAR_STAGE5_RL_ITERS",
        "SEESTAR_STAGE5_RL_ALPHA",
        "SEESTAR_STAGE5_RL_GDSTEP",
        "SEESTAR_STAGE5_RL_STOP",
        "SEESTAR_STAGE5_GRAXPERT_DECONV_STRENGTH",
        "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH",
        "SEESTAR_GRAXPERT_GPU",
        "SEESTAR_AI_ENABLED",
        "SEESTAR_AI_ENDPOINT",
        "SEESTAR_AI_MODEL",
        "SEESTAR_AI_API_KEY",
        "SEESTAR_AI_TIMEOUT_SEC",
        "SEESTAR_AI_STRENGTH",
        "SEESTAR_AI_PROMPT",
        "SEESTAR_AI_ADVISOR_MODE",
        "SEESTAR_AI_STAGE6_ENABLE",
        "SEESTAR_AI_STAGE7_ENABLE",
        "SEESTAR_AI_STAGE8_ENABLE",
        "SEESTAR_AI_ARTISTIC_DERIVATIVE_ENABLED",
        "SEESTAR_AI_ARTISTIC_ENDPOINT",
        "SEESTAR_AI_ARTISTIC_MODEL",
        "SEESTAR_AI_ARTISTIC_API_KEY",
        "SEESTAR_AI_ARTISTIC_PROMPT",
        "SEESTAR_AI_ARTISTIC_TIMEOUT_SEC",
        "SEESTAR_STAGE7_QUALITY_RETRY_MAX",
        "SEESTAR_STAGE7_SKIP_UNREADY_STARLESS",
        "SEESTAR_STAR_SEPARATION_MODE",
        "SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH",
        "SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX",
        "SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH",
        "SEESTAR_STAGE7_PREVIEW_TARGET_P50_MIN_RATIO",
        "SEESTAR_STAGE7_PREVIEW_TARGET_P50_MAX_RATIO",
        "SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE",
        "SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_REDUCTION_MIN",
        "SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_DELTA_MIN",
        "SEESTAR_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX",
        "SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR",
        "SEESTAR_STAGE8_LOCAL_ADJUSTMENT_ENGINE_ENABLE",
        "SEESTAR_STAGE8_LOCAL_CURVE_OPACITY",
        "SEESTAR_STAGE8_LIMITED_SATURATION_MAX",
        "SEESTAR_STAGE8_LIMITED_HALO_TEXTURE_GROWTH_MAX",
        "SEESTAR_STAGE8_LIMITED_HALO_TEXTURE_DELTA_MAX",
        "SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE",
        "SEESTAR_STAGE9_STARMASK_ADAPTIVE_STRETCH_ENABLE",
        "SEESTAR_STAGE9_STAR_COLOR_REPAIR_ENABLE",
        "SEESTAR_STAGE9_STAR_COLOR_REPAIR_STRENGTH",
        "SEESTAR_STAGE9_STAR_COLOR_SUPPORT_RATIO_MAX",
        "SEESTAR_STAGE9_STAR_COLOR_IMPROVEMENT_MIN",
        "SEESTAR_STAGE9_STAR_COLOR_POST_CHROMA_ERROR_MAX",
        "SEESTAR_STAGE9_SOURCE_STAR_DETAIL_PERCENTILE",
        "SEESTAR_STAGE9_SOURCE_COMPONENT_DENSITY_MAX",
        "SEESTAR_STAGE9_SOURCE_SINGLE_PIXEL_RATIO_MAX",
        "SEESTAR_STAGE9_STARMASK_ASINH_STRETCH",
        "SEESTAR_STAGE9_STARMASK_ASINH_OFFSET",
        "SEESTAR_STAGE9_STARMASK_ASINH_STRETCH_MAX",
        "SEESTAR_STAGE9_STARMASK_FAINT_TARGET",
        "SEESTAR_STAGE9_STARMASK_MID_TARGET",
        "SEESTAR_STAGE9_STARMASK_BRIGHT_TARGET",
        "SEESTAR_STAGE9_STARMASK_PEAK_TARGET",
        "SEESTAR_STAGE9_STARMASK_CHROMA_REGULARIZATION_ENABLE",
        "SEESTAR_STAGE9_STARMASK_FAINT_CHROMA_MAX",
        "SEESTAR_STAGE9_STARMASK_BRIGHT_CHROMA_MAX",
        "SEESTAR_STAGE9_STARMASK_PREDICTED_CHANGE_RATIO_MAX",
        "SEESTAR_STAGE9_STAR_REFERENCE_SIGMA",
        "SEESTAR_STAGE9_COMPACT_WEAK_STAR_RETENTION_MIN",
        "SEESTAR_STAGE9_MIXED_STAR_PEAK_RATIO_MIN",
        "SEESTAR_STAGE9_MIXED_STAR_WEAK_COUNT_MIN",
        "SEESTAR_STAGE9_MIXED_STAR_BRIGHT_COUNT_MIN",
        "SEESTAR_STAGE7_TARGET_LOCAL_METRICS_ENABLE",
        "SEESTAR_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX",
        "SEESTAR_STAGE7_LOCAL_FAINT_SNR_MIN",
        "SEESTAR_STAGE7_LOCAL_DARK_SEPARATION_MIN",
        "SEESTAR_STAGE9_QUALITY_GATE_ENABLE",
        "SEESTAR_STAGE9_HIGHLIGHT_CLIP_RATIO_MAX",
        "SEESTAR_STAGE9_HIGHLIGHT_CLIP_GROWTH_MAX",
        "SEESTAR_STAGE9_BRIGHT_PIXEL_GROWTH_MAX",
        "SEESTAR_STAGE9_BACKGROUND_LIFT_MAX",
        "SEESTAR_STAGE9_BACKGROUND_MOTTLING_GROWTH_MAX",
        "SEESTAR_STAGE9_MOTTLING_EXEMPTION_CHANGED_PIXEL_RATIO_MAX",
        "SEESTAR_STAGE9_CHANGED_PIXEL_RATIO_MAX",
        "SEESTAR_STAGE9_DARKENING_RATIO_MAX",
        "SEESTAR_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN",
        "SEESTAR_STAGE9_STAR_RECOVERY_RATIO_MIN",
        "SEESTAR_STAGE9_WEAK_STAR_SCREEN_INTENSITY_MIN",
        "SEESTAR_STAGE9_STAR_SUPPORT_RATIO_MAX",
        "SEESTAR_STAGE9_UNMATCHED_CHANGED_RATIO_MAX",
        "SEESTAR_STAGE9_CHROMATIC_ADDITION_PEAK_MIN",
        "SEESTAR_STAGE9_CHROMATIC_ADDITION_SATURATION_MIN",
        "SEESTAR_STAGE9_CHROMATIC_ADDITION_RATIO_MAX",
        "SEESTAR_STAGE9_STAR_APERTURE_RECOVERY_RATIO_MIN",
        "SEESTAR_STAGE9_STAR_WING_RECOVERY_RATIO_MIN",
        "SEESTAR_STAGE9_RESIDUAL_DARK_HOLE_RATIO_MAX",
        "SEESTAR_STAGE9_HOLLOW_STRUCTURE_DELTA_MIN",
        "SEESTAR_STAGE9_NEW_HOLLOW_STRUCTURE_AREA_MAX",
        "SEESTAR_FORCE_REVIEW_ONLY_OUTPUT",
        "SEESTAR_STAGE10_MANAGED_OUTPUT_ENABLE",
        "SEESTAR_COSMIC_CLASSIC_ENABLE",
        "SEESTAR_COSMIC_CLARITY_EXECUTABLE",
        "SEESTAR_COSMIC_CLASSIC_GPU",
        "SEESTAR_COSMIC_NATIVE_GPU",
        "SEESTAR_SYQON_GPU",
        "SEESTAR_SYQON_TIMEOUT_SEC",
        "SEESTAR_SIRILPY_TIMEOUT_SEC",
        "SEESTAR_SIRIL_PLUGIN_DIR",
        "SIRIL_PYTHON_CLI",
        "SEESTAR_SIRIL_PYTHON_CLI",
    }
)
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "result_linear_resume"
INPUT_MODE_STAGE2_CORRECTED_RESUME = "stage2_corrected_resume"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)


def _safe_output_token(value: Any, *, fallback: str = "") -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return token or fallback


def _partial_metadata_output_basename(
    metadata: Dict[str, Any],
    *,
    linear_resume: bool,
) -> str:
    """Build a useful literal filename when Siril's full template cannot resolve."""
    object_token = _safe_output_token(metadata.get("OBJECT"))
    date_digits = re.sub(r"[^0-9]+", "", str(metadata.get("DATE-OBS") or ""))
    if not object_token or len(date_digits) < 8:
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

    date_token = date_digits[:8]
    if len(date_digits) >= 14:
        date_token += f"_{date_digits[8:14]}"
    parts.extend((date_token, "processed"))
    base = "_".join(parts)
    return f"{base}_linear" if linear_resume else base


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))

class ProcessorRuntimeMixin:
    def _project_env_candidates(self) -> List[Path]:
        module_project_root = Path(__file__).resolve().parents[1]
        return [
            module_project_root / PROJECT_DEFAULT_ENV_RESOURCE_REL,
            module_project_root / PROJECT_ENV_RESOURCE_REL,
            Path.cwd() / PROJECT_ENV_OVERRIDE_NAME,
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
        required_keys = ("OBJECT", "STACKCNT", "EXPTIME", "DATE-OBS")
        missing_keys = [
            key for key in required_keys if not str(metadata.get(key, "")).strip()
        ]
        if missing_keys:
            partial_base = _partial_metadata_output_basename(
                metadata,
                linear_resume=linear_resume,
            )
            if partial_base:
                base_filename = partial_base
                fit_base_filename = partial_base + "_final"
                self.log.warn(
                    "输出命名所需 FITS 头不完整，使用已有目标元数据生成安全名称，"
                    "避免未解析占位符和通用结果名覆盖: "
                    + ", ".join(missing_keys)
                )
            else:
                base_filename = fallback_base
                fit_base_filename = fallback_fit_base
                self.log.warn(
                    "输出命名所需 FITS 头缺失，使用安全回退名，避免输出 "
                    "$OBJECT/$STACKCNT 等未解析占位符: "
                    + ", ".join(missing_keys)
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
            "schema=seestar.stage_quality.v1 "
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
            "schema": "seestar.stage_quality.v1",
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


    def _stage_diff_note(self, current_stem: str, previous_stem: str) -> Optional[str]:
        if not self.process_dir:
            return None
        current_path = self.process_dir / f"{current_stem}.fit"
        previous_path = self.process_dir / f"{previous_stem}.fit"
        if not current_path.exists() or not previous_path.exists():
            return None

        current_hash = self._sha256_file(current_path)
        previous_hash = self._sha256_file(previous_path)
        if not current_hash or not previous_hash:
            return None

        if current_hash == previous_hash:
            return (
                f"阶段对比: {current_stem}.fit 与 {previous_stem}.fit 内容一致 "
                f"(sha256={current_hash[:12]})"
            )
        return (
            f"阶段对比: {current_stem}.fit 与 {previous_stem}.fit 内容有变化 "
            f"({previous_hash[:8]} -> {current_hash[:8]})"
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

        output_format_raw = os.getenv("SEESTAR_OUTPUT_FORMAT")
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
                    f"Invalid SEESTAR_OUTPUT_FORMAT={output_format_raw!r}; using current value"
                )

        plugin_probe_raw = os.getenv("SEESTAR_WORKFLOW_PLUGIN_PROBE")
        if plugin_probe_raw is not None:
            parsed = self._parse_env_bool(
                plugin_probe_raw,
                self.cfg.workflow_plugin_probe_enabled,
            )
            self.cfg.workflow_plugin_probe_enabled = parsed
            if plugin_probe_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_WORKFLOW_PLUGIN_PROBE has invalid value; "
                    "keeping current setting"
                )

        stage4_platesolve_raw = os.getenv("SEESTAR_STAGE4_PLATESOLVE_ENABLE")
        if stage4_platesolve_raw is not None:
            parsed = self._parse_env_bool(
                stage4_platesolve_raw,
                getattr(self.cfg, "stage4_platesolve_enabled", False),
            )
            self.cfg.stage4_platesolve_enabled = parsed
            if stage4_platesolve_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE4_PLATESOLVE_ENABLE has invalid value; "
                    "keeping current setting"
                )

        stage4_auto_geometry_raw = os.getenv(
            "SEESTAR_STAGE4_AUTO_GEOMETRY_ENABLE"
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
                    "SEESTAR_STAGE4_AUTO_GEOMETRY_ENABLE has invalid value; "
                    "keeping current setting"
                )

        stage4_nbn_raw = os.getenv("SEESTAR_STAGE4_NBN_ENABLE")
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
                    "SEESTAR_STAGE4_NBN_ENABLE has invalid value; "
                    "keeping current setting"
                )

        local_star_wb_raw = os.getenv("SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE")
        if local_star_wb_raw is not None:
            parsed = self._parse_env_bool(
                local_star_wb_raw,
                getattr(self.cfg, "stage4_local_star_wb_enabled", True),
            )
            self.cfg.stage4_local_star_wb_enabled = parsed
            if local_star_wb_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE has invalid value; "
                    "keeping current setting"
                )

        for env_key, attr_name, caster in (
            (
                "SEESTAR_STAGE4_AUTO_GEOMETRY_CONFIDENCE_MIN",
                "stage4_auto_geometry_confidence_min",
                float,
            ),
            (
                "SEESTAR_STAGE4_AUTO_GEOMETRY_SCALE_RESIDUAL_MAX",
                "stage4_auto_geometry_scale_residual_max",
                float,
            ),
            (
                "SEESTAR_STAGE4_NBN_MAPPING_CONFIDENCE_MIN",
                "stage4_nbn_mapping_confidence_min",
                float,
            ),
            ("SEESTAR_STAGE4_NBN_STRENGTH", "stage4_nbn_strength", float),
            ("SEESTAR_STAGE4_NBN_GAIN_LIMIT", "stage4_nbn_gain_limit", float),
            (
                "SEESTAR_STAGE4_NBN_LINE_RATIO_DRIFT_MAX",
                "stage4_nbn_line_ratio_drift_max",
                float,
            ),
            ("SEESTAR_STAGE4_PCC_TIMEOUT_SEC", "stage4_pcc_timeout_sec", int),
            ("SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS", "stage4_local_star_wb_min_pixels", int),
            ("SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT", "stage4_local_star_wb_gain_limit", float),
            ("SEESTAR_STAGE4_LOCAL_STAR_MASK_RADIUS", "stage4_local_star_mask_radius", int),
            ("SEESTAR_STAGE4_LOCAL_STAR_MASK_COVERAGE_MAX", "stage4_local_star_mask_coverage_max", float),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, caster(raw_value.strip()))
            except (TypeError, ValueError):
                self.log.warn(f"{env_key} has invalid value; keeping current setting")

        optional_color_raw = os.getenv("SEESTAR_OPTIONAL_COLOR_TRANSFORM")
        if optional_color_raw is not None:
            parsed = self._parse_env_bool(
                optional_color_raw,
                self.cfg.optional_color_transform_enabled,
            )
            self.cfg.optional_color_transform_enabled = parsed
            if optional_color_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_OPTIONAL_COLOR_TRANSFORM has invalid value; keeping current setting"
                )

        aberration_api_raw = os.getenv("SEESTAR_ABERRATION_API_ENABLE")
        if aberration_api_raw is not None:
            parsed = self._parse_env_bool(
                aberration_api_raw,
                getattr(self.cfg, "aberration_api_enabled", False),
            )
            self.cfg.aberration_api_enabled = parsed
            if aberration_api_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_ABERRATION_API_ENABLE has invalid value; keeping current setting"
                )

        denoise_enable_raw = os.getenv("SEESTAR_DENOISE_ENABLE")
        if denoise_enable_raw is not None:
            parsed = self._parse_env_bool(
                denoise_enable_raw,
                self.cfg.denoise_enabled,
            )
            self.cfg.denoise_enabled = parsed
            if denoise_enable_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_DENOISE_ENABLE has invalid value; keeping current setting"
                )

        multiscale_denoise_raw = os.getenv(
            "SEESTAR_STAGE5_MULTISCALE_DENOISE_ENABLE"
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
                    "SEESTAR_STAGE5_MULTISCALE_DENOISE_ENABLE has invalid value; "
                    "keeping current setting"
                )

        denoise_force_raw = os.getenv("SEESTAR_DENOISE_FORCE")
        if denoise_force_raw is not None:
            parsed = self._parse_env_bool(
                denoise_force_raw,
                self.cfg.denoise_enabled,
            )
            self._force_denoise_enabled = parsed
            if denoise_force_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_DENOISE_FORCE has invalid value; keeping current setting"
                )

        stage5_deconv_raw = os.getenv("SEESTAR_STAGE5_DECONV_ENABLE")
        if stage5_deconv_raw is not None:
            parsed = self._parse_env_bool(
                stage5_deconv_raw,
                getattr(self.cfg, "stage5_deconvolution_enabled", True),
            )
            self.cfg.stage5_deconvolution_enabled = parsed
            if stage5_deconv_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE5_DECONV_ENABLE has invalid value; keeping current setting"
                )

        stage5_graxpert_deconv_raw = os.getenv(
            "SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE"
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
                    "SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE has invalid value; "
                    "keeping current setting"
                )

        for env_key, attr_name, caster in (
            (
                "SEESTAR_STAGE5_MULTISCALE_DENOISE_STRENGTH",
                "stage5_multiscale_denoise_strength",
                float,
            ),
            (
                "SEESTAR_STAGE5_MULTISCALE_DETAIL_RETENTION_MIN",
                "stage5_multiscale_detail_retention_min",
                float,
            ),
            (
                "SEESTAR_STAGE5_MULTISCALE_NOISE_REDUCTION_MIN",
                "stage5_multiscale_noise_reduction_min",
                float,
            ),
            ("SEESTAR_STAGE5_BUILTIN_DENOISE_MOD", "stage5_builtin_denoise_mod", float),
            ("SEESTAR_STAGE5_RL_MAXSTARS", "stage5_rl_maxstars", int),
            ("SEESTAR_STAGE5_RL_PSF_KS", "stage5_rl_psf_kernel_size", int),
            ("SEESTAR_STAGE5_RL_ITERS", "stage5_rl_iters", int),
            ("SEESTAR_STAGE5_RL_ALPHA", "stage5_rl_alpha", float),
            ("SEESTAR_STAGE5_RL_GDSTEP", "stage5_rl_gdstep", float),
            ("SEESTAR_STAGE5_RL_STOP", "stage5_rl_stop", float),
            ("SEESTAR_STAGE5_GRAXPERT_DECONV_STRENGTH", "stage5_graxpert_deconv_strength", float),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, caster(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        enabled_raw = os.getenv("SEESTAR_AI_ENABLED")
        if enabled_raw is not None:
            parsed = self._parse_env_bool(enabled_raw, self.cfg.ai_post_enabled)
            self.cfg.ai_post_enabled = parsed
            if enabled_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_AI_ENABLED has invalid value; keeping current setting"
                )

        for env_key, attr_name in (
            ("SEESTAR_AI_STAGE6_ENABLE", "ai_stage6_enabled"),
            ("SEESTAR_AI_STAGE7_ENABLE", "ai_stage7_enabled"),
            ("SEESTAR_AI_STAGE8_ENABLE", "ai_stage8_enabled"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            parsed = self._parse_env_bool(raw_value, getattr(self.cfg, attr_name))
            setattr(self.cfg, attr_name, parsed)
            if raw_value.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(f"{env_key} has invalid value; keeping current setting")

        endpoint = os.getenv("SEESTAR_AI_ENDPOINT")
        if endpoint is not None:
            self.cfg.ai_endpoint = endpoint.strip()

        model = os.getenv("SEESTAR_AI_MODEL")
        if model is not None:
            self.cfg.ai_model = model.strip()

        api_key = os.getenv("SEESTAR_AI_API_KEY")
        if api_key is not None:
            self.cfg.ai_api_key = api_key.strip()

        prompt = os.getenv("SEESTAR_AI_PROMPT")
        if prompt is not None:
            self.cfg.ai_prompt = prompt.strip()

        advisor_mode = os.getenv("SEESTAR_AI_ADVISOR_MODE")
        if advisor_mode is not None:
            normalized_mode = advisor_mode.strip().lower()
            if normalized_mode in {"text", "multimodal"}:
                self.cfg.ai_advisor_mode = normalized_mode
            else:
                self.log.warn(
                    "Invalid SEESTAR_AI_ADVISOR_MODE="
                    f"{advisor_mode!r}; expected text or multimodal"
                )

        artistic_enabled_raw = os.getenv("SEESTAR_AI_ARTISTIC_DERIVATIVE_ENABLED")
        if artistic_enabled_raw is not None:
            parsed = self._parse_env_bool(
                artistic_enabled_raw,
                self.cfg.ai_artistic_derivative_enabled,
            )
            self.cfg.ai_artistic_derivative_enabled = parsed
            if artistic_enabled_raw.strip().lower() not in (
                ENV_TRUE_VALUES | ENV_FALSE_VALUES
            ):
                self.log.warn(
                    "SEESTAR_AI_ARTISTIC_DERIVATIVE_ENABLED has invalid value; "
                    "keeping current setting"
                )

        for env_key, attr_name in (
            ("SEESTAR_AI_ARTISTIC_ENDPOINT", "ai_artistic_endpoint"),
            ("SEESTAR_AI_ARTISTIC_MODEL", "ai_artistic_model"),
            ("SEESTAR_AI_ARTISTIC_API_KEY", "ai_artistic_api_key"),
            ("SEESTAR_AI_ARTISTIC_PROMPT", "ai_artistic_prompt"),
        ):
            value = os.getenv(env_key)
            if value is not None:
                setattr(self.cfg, attr_name, value.strip())

        artistic_timeout_raw = os.getenv("SEESTAR_AI_ARTISTIC_TIMEOUT_SEC")
        if artistic_timeout_raw is not None:
            try:
                self.cfg.ai_artistic_timeout_sec = int(artistic_timeout_raw.strip())
            except ValueError:
                self.log.warn(
                    "Invalid SEESTAR_AI_ARTISTIC_TIMEOUT_SEC="
                    f"{artistic_timeout_raw!r}; using current value"
                )

        timeout_raw = os.getenv("SEESTAR_AI_TIMEOUT_SEC")
        if timeout_raw is not None:
            try:
                self.cfg.ai_timeout_sec = int(timeout_raw.strip())
            except ValueError:
                self.log.warn(
                    f"Invalid SEESTAR_AI_TIMEOUT_SEC={timeout_raw!r}; using current value"
                )

        strength_raw = os.getenv("SEESTAR_AI_STRENGTH")
        if strength_raw is not None:
            try:
                self.cfg.ai_strength = float(strength_raw.strip())
            except ValueError:
                self.log.warn(
                    f"Invalid SEESTAR_AI_STRENGTH={strength_raw!r}; using current value"
                )

        stage7_retry_raw = os.getenv("SEESTAR_STAGE7_QUALITY_RETRY_MAX")
        if stage7_retry_raw is not None:
            try:
                self.cfg.stage7_quality_retry_max = int(stage7_retry_raw.strip())
            except ValueError:
                self.log.warn(
                    "Invalid SEESTAR_STAGE7_QUALITY_RETRY_MAX="
                    f"{stage7_retry_raw!r}; using current value"
                )

        stage7_skip_raw = os.getenv("SEESTAR_STAGE7_SKIP_UNREADY_STARLESS")
        if stage7_skip_raw is not None:
            parsed = self._parse_env_bool(
                stage7_skip_raw,
                self.cfg.stage7_skip_unready_starless,
            )
            self.cfg.stage7_skip_unready_starless = parsed
            if stage7_skip_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE7_SKIP_UNREADY_STARLESS has invalid value; keeping current setting"
                )

        for retired_key in (
            "SEESTAR_STAR_SEPARATION_MODE",
            "SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH",
            "SEESTAR_MILD_PRESTRETCH_STRENGTH",
        ):
            if os.getenv(retired_key) is not None:
                self.log.warn(
                    f"{retired_key} is retired and ignored; "
                    "Stage 6 always uses the linear input"
                )

        for env_key, attr_name in (
            ("SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH", "stage7_soft_starless_asinh_stretch"),
            ("SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX", "stage7_bright_nebula_halo_residue_score_max"),
            ("SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH", "stage7_starless_repair_strength"),
            ("SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH", "stage7_starless_halo_repair_strength"),
            ("SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH", "stage7_starless_chroma_denoise_strength"),
            ("SEESTAR_STAGE7_STRETCH_CHROMA_NOISE_SCORE_MAX", "stage7_stretch_chroma_noise_score_max"),
            ("SEESTAR_STAGE7_STRETCH_BACKGROUND_MOTTLING_SCORE_MAX", "stage7_stretch_background_mottling_score_max"),
            ("SEESTAR_STAGE7_STRETCH_CHROMA_LOAD_GROWTH_MAX", "stage7_stretch_chroma_load_growth_max"),
            ("SEESTAR_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_MAX", "stage7_stretch_chroma_load_low_absolute_max"),
            ("SEESTAR_STAGE7_PREVIEW_TARGET_P50_MIN_RATIO", "stage7_preview_target_p50_min_ratio"),
            ("SEESTAR_STAGE7_PREVIEW_TARGET_P50_MAX_RATIO", "stage7_preview_target_p50_max_ratio"),
            ("SEESTAR_STAGE7_STARLESS_PEAK_BACKGROUND_RATIO_MIN", "stage7_starless_peak_background_ratio_min"),
            ("SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_REDUCTION_MIN", "stage7_starless_repair_chroma_reduction_min"),
            ("SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_DELTA_MIN", "stage7_starless_repair_chroma_delta_min"),
            ("SEESTAR_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX", "stage7_starmask_diffuse_residual_ratio_max"),
            ("SEESTAR_STAGE8_LOCAL_CURVE_OPACITY", "stage8_local_curve_opacity"),
            ("SEESTAR_STAGE8_LIMITED_SATURATION_MAX", "stage8_limited_saturation_max"),
            ("SEESTAR_STAGE8_LIMITED_HALO_TEXTURE_GROWTH_MAX", "stage8_limited_halo_texture_growth_max"),
            ("SEESTAR_STAGE8_LIMITED_HALO_TEXTURE_DELTA_MAX", "stage8_limited_halo_texture_delta_max"),
            ("SEESTAR_STAGE9_STARMASK_ASINH_STRETCH", "stage9_starmask_asinh_stretch"),
            ("SEESTAR_STAGE9_STARMASK_ASINH_OFFSET", "stage9_starmask_asinh_offset"),
            ("SEESTAR_STAGE9_STARMASK_ASINH_STRETCH_MAX", "stage9_starmask_asinh_stretch_max"),
            ("SEESTAR_STAGE9_STARMASK_FAINT_TARGET", "stage9_starmask_faint_target"),
            ("SEESTAR_STAGE9_STARMASK_MID_TARGET", "stage9_starmask_mid_target"),
            ("SEESTAR_STAGE9_STARMASK_BRIGHT_TARGET", "stage9_starmask_bright_target"),
            ("SEESTAR_STAGE9_STARMASK_PEAK_TARGET", "stage9_starmask_peak_target"),
            ("SEESTAR_STAGE9_STARMASK_FAINT_CHROMA_MAX", "stage9_starmask_faint_chroma_max"),
            ("SEESTAR_STAGE9_STARMASK_BRIGHT_CHROMA_MAX", "stage9_starmask_bright_chroma_max"),
            ("SEESTAR_STAGE9_STARMASK_PREDICTED_CHANGE_RATIO_MAX", "stage9_starmask_predicted_change_ratio_max"),
            ("SEESTAR_STAGE9_STAR_COLOR_REPAIR_STRENGTH", "stage9_star_color_repair_strength"),
            ("SEESTAR_STAGE9_STAR_COLOR_SUPPORT_RATIO_MAX", "stage9_star_color_support_ratio_max"),
            ("SEESTAR_STAGE9_STAR_COLOR_IMPROVEMENT_MIN", "stage9_star_color_improvement_min"),
            ("SEESTAR_STAGE9_STAR_COLOR_POST_CHROMA_ERROR_MAX", "stage9_star_color_post_chroma_error_max"),
            ("SEESTAR_STAGE9_STAR_REFERENCE_SIGMA", "stage9_star_reference_sigma"),
            ("SEESTAR_STAGE9_COMPACT_WEAK_STAR_RETENTION_MIN", "stage9_compact_weak_star_retention_min"),
            ("SEESTAR_STAGE9_MIXED_STAR_PEAK_RATIO_MIN", "stage9_mixed_star_peak_ratio_min"),
            ("SEESTAR_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX", "stage7_local_core_clip_ratio_max"),
            ("SEESTAR_STAGE7_LOCAL_FAINT_SNR_MIN", "stage7_local_faint_snr_min"),
            ("SEESTAR_STAGE7_LOCAL_DARK_SEPARATION_MIN", "stage7_local_dark_separation_min"),
            ("SEESTAR_STAGE9_HIGHLIGHT_CLIP_RATIO_MAX", "stage9_highlight_clip_ratio_max"),
            ("SEESTAR_STAGE9_HIGHLIGHT_CLIP_GROWTH_MAX", "stage9_highlight_clip_growth_max"),
            ("SEESTAR_STAGE9_BRIGHT_PIXEL_GROWTH_MAX", "stage9_bright_pixel_growth_max"),
            ("SEESTAR_STAGE9_BACKGROUND_LIFT_MAX", "stage9_background_lift_max"),
            ("SEESTAR_STAGE9_BACKGROUND_MOTTLING_GROWTH_MAX", "stage9_background_mottling_growth_max"),
            ("SEESTAR_STAGE9_MOTTLING_EXEMPTION_CHANGED_PIXEL_RATIO_MAX", "stage9_mottling_exemption_changed_pixel_ratio_max"),
            ("SEESTAR_STAGE9_CHANGED_PIXEL_RATIO_MAX", "stage9_changed_pixel_ratio_max"),
            ("SEESTAR_STAGE9_DARKENING_RATIO_MAX", "stage9_darkening_ratio_max"),
            ("SEESTAR_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN", "stage9_weak_star_recovery_ratio_min"),
            ("SEESTAR_STAGE9_STAR_RECOVERY_RATIO_MIN", "stage9_star_recovery_ratio_min"),
            ("SEESTAR_STAGE9_SOURCE_STAR_DETAIL_PERCENTILE", "stage9_source_star_detail_percentile"),
            ("SEESTAR_STAGE9_SOURCE_COMPONENT_DENSITY_MAX", "stage9_source_component_density_max"),
            ("SEESTAR_STAGE9_SOURCE_SINGLE_PIXEL_RATIO_MAX", "stage9_source_single_pixel_ratio_max"),
            ("SEESTAR_STAGE9_WEAK_STAR_SCREEN_INTENSITY_MIN", "stage9_weak_star_screen_intensity_min"),
            ("SEESTAR_STAGE9_STAR_SUPPORT_RATIO_MAX", "stage9_star_support_ratio_max"),
            ("SEESTAR_STAGE9_UNMATCHED_CHANGED_RATIO_MAX", "stage9_unmatched_changed_ratio_max"),
            ("SEESTAR_STAGE9_CHROMATIC_ADDITION_PEAK_MIN", "stage9_chromatic_addition_peak_min"),
            ("SEESTAR_STAGE9_CHROMATIC_ADDITION_SATURATION_MIN", "stage9_chromatic_addition_saturation_min"),
            ("SEESTAR_STAGE9_CHROMATIC_ADDITION_RATIO_MAX", "stage9_chromatic_addition_ratio_max"),
            ("SEESTAR_STAGE9_STAR_APERTURE_RECOVERY_RATIO_MIN", "stage9_star_aperture_recovery_ratio_min"),
            ("SEESTAR_STAGE9_STAR_WING_RECOVERY_RATIO_MIN", "stage9_star_wing_recovery_ratio_min"),
            ("SEESTAR_STAGE9_RESIDUAL_DARK_HOLE_RATIO_MAX", "stage9_residual_dark_hole_ratio_max"),
            ("SEESTAR_STAGE9_HOLLOW_STRUCTURE_DELTA_MIN", "stage9_hollow_structure_delta_min"),
            ("SEESTAR_STAGE9_NEW_HOLLOW_STRUCTURE_AREA_MAX", "stage9_new_hollow_structure_area_max"),
            ("SEESTAR_STAGE9_LOCAL_COMPONENT_PEAK_MIN", "stage9_local_component_peak_min"),
            ("SEESTAR_STAGE9_LOCAL_COMPONENT_AREA_MAX", "stage9_local_component_area_max"),
            ("SEESTAR_STAGE9_LOCAL_COMPONENT_ASPECT_RATIO_MAX", "stage9_local_component_aspect_ratio_max"),
            ("SEESTAR_STAGE9_LOCAL_COMPONENT_FILL_RATIO_MIN", "stage9_local_component_fill_ratio_min"),
            ("SEESTAR_STAGE9_LOCAL_SINGLE_PIXEL_RATIO_MAX", "stage9_local_single_pixel_ratio_max"),
            ("SEESTAR_STAGE9_LOCAL_CYAN_BLUE_PEAK_MIN", "stage9_local_cyan_blue_peak_min"),
            ("SEESTAR_STAGE9_LOCAL_CYAN_BLUE_SATURATION_MIN", "stage9_local_cyan_blue_saturation_min"),
            ("SEESTAR_STAGE9_LOCAL_CYAN_BLUE_COMPONENT_AREA_MAX", "stage9_local_cyan_blue_component_area_max"),
            ("SEESTAR_STAGE9_CORE_PERCENTILE", "stage9_core_percentile"),
            ("SEESTAR_STAGE9_CORE_COLOR_JUMP_MIN", "stage9_core_color_jump_min"),
            ("SEESTAR_STAGE9_CORE_COLOR_JUMP_COMPONENT_AREA_MAX", "stage9_core_color_jump_component_area_max"),
            ("SEESTAR_STAGE10_CHROMA_FOCUS_SCORE_MIN", "stage10_chroma_focus_score_min"),
            ("SEESTAR_STAGE10_SEPARATE_CHROMA_SCORE_MIN", "stage10_separate_chroma_score_min"),
            ("SEESTAR_STAGE10_FULL_BG_STD_MIN", "stage10_full_bg_std_min"),
            ("SEESTAR_STAGE10_FULL_MOTTLING_SCORE_MIN", "stage10_full_mottling_score_min"),
            ("SEESTAR_STAGE10_STAGE9_LOCAL_COLOR_RISK_STRENGTH", "stage10_stage9_local_color_risk_strength"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, float(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        for env_key, attr_name in (
            ("SEESTAR_STAGE9_MIXED_STAR_WEAK_COUNT_MIN", "stage9_mixed_star_weak_count_min"),
            ("SEESTAR_STAGE9_MIXED_STAR_BRIGHT_COUNT_MIN", "stage9_mixed_star_bright_count_min"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, int(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        for env_key, attr_name in (
            ("SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE", "stage7_starless_pixel_repair_enabled"),
            ("SEESTAR_STAGE7_CHROMA_RESCUE_ENABLE", "stage7_chroma_rescue_enabled"),
            ("SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR", "stage8_force_conservative_after_stage7_repair"),
            ("SEESTAR_STAGE8_LOCAL_ADJUSTMENT_ENGINE_ENABLE", "stage8_local_adjustment_engine_enabled"),
            ("SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE", "stage9_starmask_stretch_enabled"),
            ("SEESTAR_STAGE9_STARMASK_ADAPTIVE_STRETCH_ENABLE", "stage9_starmask_adaptive_stretch_enabled"),
            ("SEESTAR_STAGE9_COMPACT_STARMASK_ENABLE", "stage9_compact_starmask_enabled"),
            ("SEESTAR_STAGE9_STAR_COLOR_REPAIR_ENABLE", "stage9_star_color_repair_enabled"),
            ("SEESTAR_STAGE9_STARMASK_CHROMA_REGULARIZATION_ENABLE", "stage9_starmask_chroma_regularization_enabled"),
            ("SEESTAR_STAGE9_QUALITY_GATE_ENABLE", "stage9_quality_gate_enabled"),
            ("SEESTAR_STAGE10_MANAGED_OUTPUT_ENABLE", "stage10_managed_output_enabled"),
            ("SEESTAR_FORCE_REVIEW_ONLY_OUTPUT", "force_review_only_output"),
            ("SEESTAR_STAGE7_TARGET_LOCAL_METRICS_ENABLE", "stage7_target_local_metrics_enabled"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            parsed = self._parse_env_bool(raw_value, getattr(self.cfg, attr_name))
            setattr(self.cfg, attr_name, parsed)
            if raw_value.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(f"{env_key} has invalid value; keeping current setting")

        old_timeout = self.cfg.ai_timeout_sec
        old_strength = self.cfg.ai_strength
        old_stage7_retry = self.cfg.stage7_quality_retry_max
        old_stage7_soft = self.cfg.stage7_soft_starless_asinh_stretch
        old_stage7_bright_halo = self.cfg.stage7_bright_nebula_halo_residue_score_max
        old_stage7_repair = self.cfg.stage7_starless_repair_strength
        old_stage7_halo = self.cfg.stage7_starless_halo_repair_strength
        old_stage7_chroma = self.cfg.stage7_starless_chroma_denoise_strength
        old_stage9_starmask_stretch = self.cfg.stage9_starmask_asinh_stretch
        old_stage9_starmask_offset = self.cfg.stage9_starmask_asinh_offset
        self.cfg.ai_timeout_sec = _clamp_int(self.cfg.ai_timeout_sec, 15, 300)
        self.cfg.ai_strength = _clamp_float(self.cfg.ai_strength, 0.05, 0.25)
        self.cfg.stage7_quality_retry_max = _clamp_int(
            self.cfg.stage7_quality_retry_max, 0, 3
        )
        self.cfg.stage7_soft_starless_asinh_stretch = _clamp_float(
            self.cfg.stage7_soft_starless_asinh_stretch,
            1.05,
            self.cfg.stage7_ultra_conservative_asinh_stretch,
        )
        self.cfg.stage7_bright_nebula_halo_residue_score_max = _clamp_float(
            self.cfg.stage7_bright_nebula_halo_residue_score_max,
            self.cfg.stage7_halo_residue_score_max,
            1.20,
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
        self.cfg.stage9_starmask_asinh_offset = _clamp_float(
            self.cfg.stage9_starmask_asinh_offset,
            0.0005,
            0.0060,
        )
        for attr_name, lower, upper in (
            ("stage7_stretch_chroma_noise_score_max", 0.10, 0.80),
            ("stage7_stretch_background_mottling_score_max", 0.10, 1.00),
            ("stage7_stretch_chroma_load_growth_max", 1.00, 3.00),
            ("stage7_stretch_chroma_load_low_absolute_max", 0.01, 0.15),
            ("stage7_preview_target_p50_min_ratio", 0.25, 0.90),
            ("stage7_preview_target_p50_max_ratio", 1.00, 3.00),
            ("stage7_starless_peak_background_ratio_min", 1.5, 12.0),
            ("stage7_starless_repair_chroma_reduction_min", 0.05, 0.80),
            ("stage7_starless_repair_chroma_delta_min", 0.00001, 0.05000),
            ("stage7_starmask_diffuse_residual_ratio_max", 0.01, 0.50),
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
            ("stage9_star_aperture_recovery_ratio_min", 0.40, 0.98),
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
        if old_timeout != self.cfg.ai_timeout_sec:
            self.log.warn(
                f"AI timeout clamped: {old_timeout} -> {self.cfg.ai_timeout_sec}"
            )
        if old_strength != self.cfg.ai_strength:
            self.log.warn(
                f"AI strength clamped: {old_strength} -> {self.cfg.ai_strength}"
            )
        if old_stage7_retry != self.cfg.stage7_quality_retry_max:
            self.log.warn(
                "Stage7 quality retry max clamped: "
                f"{old_stage7_retry} -> {self.cfg.stage7_quality_retry_max}"
            )
        for label, old_value, new_value in (
            ("Stage7 soft starless asinh stretch", old_stage7_soft, self.cfg.stage7_soft_starless_asinh_stretch),
            ("Stage7 bright-nebula halo threshold", old_stage7_bright_halo, self.cfg.stage7_bright_nebula_halo_residue_score_max),
            ("Stage7 starless repair strength", old_stage7_repair, self.cfg.stage7_starless_repair_strength),
            ("Stage7 starless halo repair strength", old_stage7_halo, self.cfg.stage7_starless_halo_repair_strength),
            ("Stage7 starless chroma denoise strength", old_stage7_chroma, self.cfg.stage7_starless_chroma_denoise_strength),
            ("Stage9 starmask asinh stretch", old_stage9_starmask_stretch, self.cfg.stage9_starmask_asinh_stretch),
            ("Stage9 starmask asinh offset", old_stage9_starmask_offset, self.cfg.stage9_starmask_asinh_offset),
        ):
            if old_value != new_value:
                self.log.warn(f"{label} clamped: {old_value} -> {new_value}")


    def _apply_ai_env_overrides(self):
        """Compatibility wrapper for isolated Stage11 runners."""
        self._apply_runtime_env_overrides()


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
            self.log.warn(f"[AI] Failed to measure image features: {e}")
            return None


    def _measure_current_quality(self) -> Optional[QualityMetrics]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            return measure_quality_metrics(image_data)
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.warn(f"[AI] Failed to measure image quality metrics: {e}")
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
        result: Dict[str, Any] = {
            "verified": False,
            "state": "unknown",
            "detail": "current input mode is not a resume checkpoint",
        }
        if not self.work_dir:
            self._trusted_input_provenance = result
            return result

        input_path: Optional[Path] = None
        checkpoint_name = ""
        if self.input_mode == INPUT_MODE_LINEAR_RESUME:
            input_path = self.work_dir / "result_linear.fit"
            checkpoint_name = "result_linear"
        elif self.input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            checkpoint_name = "stage2_corrected"
            candidates = [
                self.work_dir / "stage2_corrected.fit",
                self.work_dir / "process" / "stage2_corrected.fit",
            ]
            input_path = next((path for path in candidates if path.is_file()), None)

        if input_path is not None and input_path.is_file() and checkpoint_name:
            result = run_manifest.verify_resume_provenance(
                work_dir=self.work_dir,
                input_path=input_path,
                checkpoint_name=checkpoint_name,
            )
        elif checkpoint_name:
            result = {
                "verified": False,
                "state": "unknown",
                "checkpoint": checkpoint_name,
                "detail": "resume checkpoint file is missing",
            }

        self._trusted_input_provenance = result
        if result.get("verified"):
            self.log.info(
                "[InputProfile] verified resume provenance: "
                f"{result.get('detail')}"
            )
        elif checkpoint_name:
            self.log.warn(
                "[InputProfile] resume provenance not trusted: "
                f"{result.get('detail')}"
            )
        return result


    def _processing_plan_input_path(self) -> Optional[Path]:
        for candidate in (
            getattr(self, "source_file", None),
            getattr(self, "linear_intermediate_path", None),
            self.process_dir / "working.fit" if self.process_dir else None,
        ):
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    def _resolve_channel_profile(self, profile: InputProfile) -> Dict[str, Any]:
        """Resolve physical channel meaning before the processing plan is frozen."""
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
            explicit_filter_hint=os.getenv("SEESTAR_STAGE4_FILTER_HINT", ""),
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


    def _planned_stage_actions(self, profile: InputProfile) -> List[Dict[str, Any]]:
        safe = profile.safe_for_linear_steps
        if self.input_mode == INPUT_MODE_LINEAR_RESUME:
            early_actions = {
                1: "load_verified_checkpoint" if safe else "load_untrusted_checkpoint",
                2: "skip_resume",
                3: "skip_resume" if safe else "skip_input_state_guard",
                4: "skip_resume" if safe else "skip_input_state_guard",
                5: "skip_resume" if safe else "skip_input_state_guard",
            }
        elif self.input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            early_actions = {
                1: "skip_resume",
                2: "load_verified_checkpoint" if safe else "load_untrusted_checkpoint",
                3: "apply" if safe else "skip_input_state_guard",
                4: "apply" if safe else "skip_input_state_guard",
                5: "apply" if safe else "skip_input_state_guard",
            }
        else:
            early_actions = {
                1: "completed_before_plan",
                2: "completed_before_plan",
                3: (
                    "conditional_background_decision"
                    if safe
                    else "skip_input_state_guard"
                ),
                4: "apply" if safe else "skip_input_state_guard",
                5: "apply" if safe else "skip_input_state_guard",
            }
        late_actions = {
            6: "apply" if safe else "skip_input_state_guard",
            7: "apply" if safe else "skip_input_state_guard",
            8: "apply" if safe else "skip_input_state_guard",
            9: "apply" if safe else "skip_input_state_guard",
            10: "apply" if safe else "review_export_only",
            11: (
                "optional"
                if (
                    safe
                    and ai_advisory.network_mode_enabled()
                    and bool(getattr(self.cfg, "ai_post_enabled", False))
                )
                else "skip"
            ),
        }
        actions = {**early_actions, **late_actions}
        stages_by_number = {
            index: stage
            for index, stage in enumerate(PipelineStage, start=1)
        }
        return [
            {
                "stage": number,
                "label": stages_by_number[number].label,
                "action": actions[number],
            }
            for number in range(1, 12)
        ]


    def _write_processing_plan(self, profile: InputProfile) -> bool:
        """Freeze this run's resolved route before post-processing transforms."""
        if not self.work_dir:
            return False
        self._run_id = str(getattr(self, "_run_id", "") or uuid.uuid4())
        input_path = self._processing_plan_input_path()
        input_record: Dict[str, Any] = {
            "path": str(input_path) if input_path else None,
            "input_mode": self.input_mode,
            "stage1_input_mode": getattr(self, "_stage1_input_mode", "unknown"),
            "profile": profile.to_dict(),
        }
        if input_path:
            input_record.update(run_manifest.file_record(input_path))
        freeze_primary = getattr(self, "_freeze_primary_target", None)
        if callable(freeze_primary):
            freeze_primary()
        channel_profile = self._resolve_channel_profile(profile)
        target_profile = dict(getattr(self, "target_profile", {}) or {})
        resolved_policy = dict(getattr(self, "pipeline_policy", {}) or {})
        stage3_policy = dict(resolved_policy.get("stage3_background") or {})
        stretch_policy = dict(resolved_policy.get("stage6_stretch") or {})
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
        plan: Dict[str, Any] = {
            "schema": "seestar.processing-plan.v1",
            "run_id": self._run_id,
            "generated_at": run_manifest.utc_timestamp(),
            "input": input_record,
            "channel_semantics": str(
                getattr(self, "_channel_semantics", "unknown") or "unknown"
            ),
            "channel_profile": dict(channel_profile),
            "target": {
                "primary": target_profile.get("target_type"),
                "primary_record": target_profile.get("primary_target", {}),
                "secondary": target_profile.get("secondary_labels", []),
                "confidence": target_profile.get("target_confidence"),
                "policy": resolved_policy.get("policy_name"),
                "star_separation_basis": "primary_target_only",
            },
            "planned_steps": self._planned_stage_actions(profile),
            "candidate_contracts": {
                "stage3_background": {
                    "model_priority": list(stage3_policy.get("model_priority") or []),
                    "runtime_capability_probe": True,
                },
                "stage7_stretch": {
                    "allowed_modes": list(stretch_policy.get("candidate_mode") or []),
                    "fallback_candidate": stretch_policy.get("fallback_candidate"),
                    "model_output_fields": ["selected_candidate_id"],
                    "parameters_owned_by": "code",
                    "selection_after_hard_gates": True,
                },
                "stage6_star_separation": {
                    "candidate_ids": [
                        "syqon_standard",
                        "syqon_large_context",
                        "syqon_axiom_standard_if_available",
                    ],
                    "model_output_fields": ["selected_candidate_id"],
                    "parameters_owned_by": "code",
                },
                "stage8_enhancement": {
                    "candidate_ids": [
                        "preserve",
                        "conservative",
                        "balanced",
                        "detail_preserving",
                    ],
                    "model_output_fields": ["selected_candidate_id"],
                    "parameters_owned_by": "code",
                },
                "stage11_optional_derivative": {
                    "candidate_ids": [
                        "preserve",
                        "conservative",
                        "balanced",
                        "detail_safe",
                    ],
                    "model_output_fields": ["selected_candidate_id"],
                    "parameters_owned_by": "code",
                },
                "stage9_star_remix": {
                    "primary_id": "primary",
                    "fallback_intensity_levels": remix_levels,
                    "ids_generated_after_quality_scaling": True,
                },
            },
            "capabilities": {
                "offline_first": True,
                "network_requested": ai_advisory.network_mode_enabled(),
                "ai_advisory_requested": bool(
                    getattr(self.cfg, "ai_post_enabled", False)
                ),
                "ai_advisory_enabled": bool(
                    ai_advisory.network_mode_enabled()
                    and getattr(self.cfg, "ai_post_enabled", False)
                ),
            },
            "config": run_manifest.redact_sensitive(asdict(self.cfg)),
        }
        plan["plan_hash"] = run_manifest.canonical_payload_hash(plan)
        self._processing_plan = plan
        self._processing_plan_hash = str(plan["plan_hash"])

        destinations = [self.work_dir / "processing-plan.json"]
        if self.process_dir:
            destinations.append(self.process_dir / "processing-plan.json")
        try:
            for destination in destinations:
                run_manifest.atomic_write_json(destination, plan)
            self.log.info(
                "[ProcessingPlan] frozen "
                f"run_id={self._run_id} hash={self._processing_plan_hash[:12]}"
            )
            return True
        except (OSError, TypeError, ValueError) as error:
            self.log.warn(f"processing-plan.json 写入失败: {error}")
            return False


    def _pipeline_result_status(self, failure_reason: Optional[str] = None) -> str:
        if failure_reason or any(result.status == "failed" for result in self.results):
            return "failed"
        review_required = bool(
            getattr(self, "_input_state_review_route", False)
            or getattr(self, "_final_output_review_only", False)
            or getattr(self, "_background_review_required", False)
            or (
                bool(getattr(self, "_stage9_stars_required", False))
                and not bool(getattr(self, "_stage9_stars_applied", False))
            )
        )
        if review_required:
            return "review_required"
        if any(
            result.status == "degraded"
            or result.fallback_used
            for result in self.results
        ):
            return "partial_success"
        return "success"


    def _result_checkpoint_record(
        self,
        path: Path,
        *,
        state: str,
    ) -> Optional[Dict[str, Any]]:
        if not path.is_file():
            return None
        record = run_manifest.file_record(path, base_dir=self.work_dir)
        record["state"] = state
        return record


    def _write_pipeline_result_manifest(
        self,
        *,
        failure_reason: Optional[str] = None,
    ) -> bool:
        """Persist actual steps, output hashes, fallbacks, and global status."""
        if not self.work_dir:
            return False
        input_state = str((getattr(self, "input_profile", {}) or {}).get("state") or "unknown")
        checkpoint_state = "linear" if input_state == "linear" else "unknown"
        checkpoints: Dict[str, Any] = {}
        result_linear = self.work_dir / "result_linear.fit"
        result_linear_record = self._result_checkpoint_record(
            result_linear,
            state=checkpoint_state,
        )
        if result_linear_record:
            checkpoints["result_linear"] = result_linear_record

        stage2_candidates = [
            self.work_dir / "stage2_corrected.fit",
            self.process_dir / "stage2_corrected.fit" if self.process_dir else None,
        ]
        stage2_path = next(
            (
                path
                for path in stage2_candidates
                if path is not None and path.is_file()
            ),
            None,
        )
        if stage2_path is not None:
            stage2_record = self._result_checkpoint_record(
                stage2_path,
                state=checkpoint_state,
            )
            if stage2_record:
                checkpoints["stage2_corrected"] = stage2_record

        outputs = run_manifest.collect_output_records(
            self.work_dir,
            output_basenames=getattr(self, "_final_output_basenames", ()),
            exported_after=getattr(self, "_final_export_started_at", None),
        )

        status = self._pipeline_result_status(failure_reason)
        manifest: Dict[str, Any] = {
            "schema": "seestar.pipeline-result.v1",
            "run_id": getattr(self, "_run_id", None),
            "generated_at": run_manifest.utc_timestamp(),
            "status": status,
            "failure_reason": failure_reason,
            "plan_hash": getattr(self, "_processing_plan_hash", None),
            "input_profile": dict(getattr(self, "input_profile", {}) or {}),
            "channel_semantics": str(
                getattr(self, "_channel_semantics", "unknown") or "unknown"
            ),
            "target_profile": dict(getattr(self, "target_profile", {}) or {}),
            "star_separation": {
                "state": getattr(self, "_star_separation_state", "pending"),
                "stars_required": bool(
                    getattr(self, "_stage9_stars_required", False)
                ),
                "stars_applied": bool(
                    getattr(self, "_stage9_stars_applied", False)
                ),
                "application_mode": getattr(
                    self,
                    "_stage9_stars_application_mode",
                    "pending",
                ),
            },
            "actual_steps": [
                {
                    "name": result.name,
                    "status": result.status,
                    "display_status": result.display_status,
                    "execution": result.execution,
                    "fallback_used": result.fallback_used,
                    "upstream_passthrough": result.upstream_passthrough,
                    "reason_code": result.reason_code or None,
                    "duration_seconds": float(result.duration),
                    "message": result.message,
                    "details": result.details,
                    "components": result.components,
                }
                for result in self.results
            ],
            "checkpoints": checkpoints,
            "outputs": outputs,
        }
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


    def _write_ai_raw_response(
        self,
        stage_name: str,
        *,
        endpoint_url: str,
        temperature: float,
        json_mode: bool,
        response_obj: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
        error_text: Optional[str] = None,
    ) -> None:
        self._ai_raw_response_counter = write_ai_raw_response(
            self.process_dir,
            self.log,
            self._ai_raw_response_counter,
            self._short_text,
            stage_name,
            endpoint_url=endpoint_url,
            temperature=temperature,
            json_mode=json_mode,
            response_obj=response_obj,
            content=content,
            error_text=error_text,
        )
