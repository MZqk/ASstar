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
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    format_feature_summary,
    measure_image_features,
    measure_quality_metrics,
)
from models import ImageFeatures, QualityMetrics, Stage6StretchStrategy, StageResult, TargetType
from save_utils import save_stage_output, write_ai_raw_response, write_stage_json

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except Exception:
    CommandError = Exception
    DataError = Exception
    SirilError = Exception

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
except Exception:
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
        "SEESTAR_WORKFLOW_PLUGIN_PROBE",
        "SEESTAR_SPCC_ENABLE",
        "SEESTAR_STAGE4_PLATESOLVE_ENABLE",
        "SEESTAR_STAGE4_PLATESOLVE_FOCAL",
        "SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE",
        "SEESTAR_STAGE4_PLATESOLVE_ORDER",
        "SEESTAR_STAGE4_PLATESOLVE_CATALOGS",
        "SEESTAR_STAGE4_PLATESOLVE_HEADER_RADIUS",
        "SEESTAR_STAGE4_SPCC_SENSOR_MODE",
        "SEESTAR_STAGE4_SPCC_OSC_SENSOR",
        "SEESTAR_STAGE4_SPCC_OSC_FILTER",
        "SEESTAR_STAGE4_SPCC_BUILTIN_DUALBAND_FILTER",
        "SEESTAR_STAGE4_SPCC_MONO_SENSOR",
        "SEESTAR_STAGE4_SPCC_R_FILTER",
        "SEESTAR_STAGE4_SPCC_G_FILTER",
        "SEESTAR_STAGE4_SPCC_B_FILTER",
        "SEESTAR_STAGE4_SPCC_WHITE_REF",
        "SEESTAR_STAGE4_SPCC_ADAPTIVE_WHITE_REF",
        "SEESTAR_STAGE4_SPCC_NEBULA_WHITE_REF",
        "SEESTAR_STAGE4_SPCC_BGTOL",
        "SEESTAR_STAGE4_SPCC_LIMITMAG",
        "SEESTAR_STAGE4_SPCC_RESTORE_CPU",
        "SEESTAR_STAGE4_SPCC_RESTORE_MAXPROCS",
        "SEESTAR_STAGE4_PCC_CATALOGS",
        "SEESTAR_STAGE4_PCC_HEADER_FALLBACK_ENABLE",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_TARGET_AWARE_ENABLE",
        "SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS",
        "SEESTAR_ABERRATION_API_ENABLE",
        "SEESTAR_ABERRATION_PROVIDER",
        "SEESTAR_OPTIONAL_COLOR_TRANSFORM",
        "SEESTAR_DENOISE_ENABLE",
        "SEESTAR_DENOISE_FORCE",
        "SEESTAR_STAGE5_BUILTIN_DENOISE_MOD",
        "SEESTAR_STAGE5_DECONV_ENABLE",
        "SEESTAR_STAGE5_RL_MAXSTARS",
        "SEESTAR_STAGE5_RL_PSF_KS",
        "SEESTAR_STAGE5_RL_ITERS",
        "SEESTAR_STAGE5_RL_ALPHA",
        "SEESTAR_STAGE5_RL_GDSTEP",
        "SEESTAR_STAGE5_RL_STOP",
        "SEESTAR_STAGE5_GRAXPERT_DECONV_STRENGTH",
        "SEESTAR_AI_ENABLED",
        "SEESTAR_AI_ENDPOINT",
        "SEESTAR_AI_MODEL",
        "SEESTAR_AI_API_KEY",
        "SEESTAR_AI_TIMEOUT_SEC",
        "SEESTAR_AI_STRENGTH",
        "SEESTAR_AI_PROMPT",
        "SEESTAR_AI_STAGE6_ENABLE",
        "SEESTAR_AI_STAGE7_ENABLE",
        "SEESTAR_AI_STAGE8_ENABLE",
        "SEESTAR_STAGE7_QUALITY_RETRY_MAX",
        "SEESTAR_STAGE7_SKIP_UNREADY_STARLESS",
        "SEESTAR_STAR_SEPARATION_MODE",
        "SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH",
        "SEESTAR_MILD_PRESTRETCH_STRENGTH",
        "SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH",
        "SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX",
        "SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE",
        "SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR",
        "SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE",
        "SEESTAR_STAGE9_STARMASK_ASINH_STRETCH",
        "SEESTAR_STAGE9_STARMASK_ASINH_OFFSET",
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
        base_filename = RESULT_BASENAME_TEMPLATE
        if self._stage1_input_mode == "linear_resume":
            base_filename += "_linear"
        self.main_output_basename_template = base_filename
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
        except Exception as e:
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

        spcc_enable_raw = os.getenv("SEESTAR_SPCC_ENABLE")
        if spcc_enable_raw is not None:
            parsed = self._parse_env_bool(
                spcc_enable_raw,
                self.cfg.spcc_enabled,
            )
            self.cfg.spcc_enabled = parsed
            if spcc_enable_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_SPCC_ENABLE has invalid value; keeping current setting"
                )

        spcc_adaptive_white_ref_raw = os.getenv("SEESTAR_STAGE4_SPCC_ADAPTIVE_WHITE_REF")
        if spcc_adaptive_white_ref_raw is not None:
            parsed = self._parse_env_bool(
                spcc_adaptive_white_ref_raw,
                self.cfg.stage4_spcc_adaptive_white_ref_enabled,
            )
            self.cfg.stage4_spcc_adaptive_white_ref_enabled = parsed
            if spcc_adaptive_white_ref_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE4_SPCC_ADAPTIVE_WHITE_REF has invalid value; "
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

        builtin_dualband_raw = os.getenv("SEESTAR_STAGE4_SPCC_BUILTIN_DUALBAND_FILTER")
        if builtin_dualband_raw is not None:
            parsed = self._parse_env_bool(
                builtin_dualband_raw,
                getattr(self.cfg, "stage4_spcc_builtin_dualband_filter_enabled", False),
            )
            self.cfg.stage4_spcc_builtin_dualband_filter_enabled = parsed
            if builtin_dualband_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE4_SPCC_BUILTIN_DUALBAND_FILTER has invalid value; "
                    "keeping current setting"
                )

        pcc_header_fallback_raw = os.getenv("SEESTAR_STAGE4_PCC_HEADER_FALLBACK_ENABLE")
        if pcc_header_fallback_raw is not None:
            parsed = self._parse_env_bool(
                pcc_header_fallback_raw,
                getattr(self.cfg, "stage4_pcc_header_fallback_enabled", True),
            )
            self.cfg.stage4_pcc_header_fallback_enabled = parsed
            if pcc_header_fallback_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE4_PCC_HEADER_FALLBACK_ENABLE has invalid value; "
                    "keeping current setting"
                )

        for env_key, attr_name in (
            ("SEESTAR_STAGE4_SPCC_SENSOR_MODE", "stage4_spcc_sensor_mode"),
            ("SEESTAR_STAGE4_SPCC_OSC_SENSOR", "stage4_spcc_osc_sensor"),
            ("SEESTAR_STAGE4_SPCC_OSC_FILTER", "stage4_spcc_osc_filter"),
            ("SEESTAR_STAGE4_SPCC_MONO_SENSOR", "stage4_spcc_mono_sensor"),
            ("SEESTAR_STAGE4_SPCC_R_FILTER", "stage4_spcc_r_filter"),
            ("SEESTAR_STAGE4_SPCC_G_FILTER", "stage4_spcc_g_filter"),
            ("SEESTAR_STAGE4_SPCC_B_FILTER", "stage4_spcc_b_filter"),
            ("SEESTAR_STAGE4_SPCC_WHITE_REF", "stage4_spcc_white_ref"),
            ("SEESTAR_STAGE4_SPCC_NEBULA_WHITE_REF", "stage4_spcc_nebula_white_ref"),
            ("SEESTAR_STAGE4_SPCC_BGTOL", "stage4_spcc_bgtol"),
            ("SEESTAR_STAGE4_SPCC_LIMITMAG", "stage4_spcc_limitmag"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is not None:
                setattr(self.cfg, attr_name, raw_value.strip())

        spcc_restore_cpu_raw = (
            os.getenv("SEESTAR_STAGE4_SPCC_RESTORE_CPU")
            or os.getenv("SEESTAR_STAGE4_SPCC_RESTORE_MAXPROCS")
        )
        if spcc_restore_cpu_raw is not None:
            try:
                parsed = int(spcc_restore_cpu_raw.strip())
                if parsed < 0:
                    raise ValueError
                self.cfg.stage4_spcc_restore_cpu = parsed
            except (TypeError, ValueError):
                self.log.warn(
                    "SEESTAR_STAGE4_SPCC_RESTORE_CPU has invalid value; "
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

        local_star_wb_target_aware_raw = os.getenv("SEESTAR_STAGE4_LOCAL_STAR_WB_TARGET_AWARE_ENABLE")
        if local_star_wb_target_aware_raw is not None:
            parsed = self._parse_env_bool(
                local_star_wb_target_aware_raw,
                getattr(self.cfg, "stage4_local_star_wb_target_aware_enabled", False),
            )
            self.cfg.stage4_local_star_wb_target_aware_enabled = parsed
            if local_star_wb_target_aware_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE4_LOCAL_STAR_WB_TARGET_AWARE_ENABLE has invalid value; "
                    "keeping current setting"
                )

        for env_key, attr_name, caster in (
            ("SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS", "stage4_local_star_wb_min_pixels", int),
            ("SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT", "stage4_local_star_wb_gain_limit", float),
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

        for env_key, attr_name, caster in (
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

        star_mode_raw = os.getenv("SEESTAR_STAR_SEPARATION_MODE")
        if star_mode_raw is not None:
            normalized_mode = star_mode_raw.strip().lower()
            if normalized_mode in {
                "linear_star_separation",
                "mild_prestretch_star_separation",
            }:
                self.cfg.star_separation_mode = normalized_mode
            else:
                self.log.warn(
                    "Invalid SEESTAR_STAR_SEPARATION_MODE="
                    f"{star_mode_raw!r}; using current value"
                )

        star_fallback_raw = os.getenv(
            "SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH"
        )
        if star_fallback_raw is not None:
            parsed = self._parse_env_bool(
                star_fallback_raw,
                self.cfg.star_separation_fallback_to_mild_prestretch,
            )
            self.cfg.star_separation_fallback_to_mild_prestretch = parsed
            if star_fallback_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH has invalid value; "
                    "keeping current setting"
                )

        mild_prestretch_raw = os.getenv("SEESTAR_MILD_PRESTRETCH_STRENGTH")
        if mild_prestretch_raw is not None:
            try:
                self.cfg.mild_prestretch_strength = float(mild_prestretch_raw.strip())
            except ValueError:
                self.log.warn(
                    "Invalid SEESTAR_MILD_PRESTRETCH_STRENGTH="
                    f"{mild_prestretch_raw!r}; using current value"
                )

        for env_key, attr_name in (
            ("SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH", "stage7_soft_starless_asinh_stretch"),
            ("SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX", "stage7_bright_nebula_halo_residue_score_max"),
            ("SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH", "stage7_starless_repair_strength"),
            ("SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH", "stage7_starless_halo_repair_strength"),
            ("SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH", "stage7_starless_chroma_denoise_strength"),
            ("SEESTAR_STAGE9_STARMASK_ASINH_STRETCH", "stage9_starmask_asinh_stretch"),
            ("SEESTAR_STAGE9_STARMASK_ASINH_OFFSET", "stage9_starmask_asinh_offset"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, float(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        for env_key, attr_name in (
            ("SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE", "stage7_starless_pixel_repair_enabled"),
            ("SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR", "stage8_force_conservative_after_stage7_repair"),
            ("SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE", "stage9_starmask_stretch_enabled"),
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
        old_mild_prestretch = self.cfg.mild_prestretch_strength
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
        self.cfg.mild_prestretch_strength = _clamp_float(
            self.cfg.mild_prestretch_strength,
            1.05,
            1.80,
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
        if old_mild_prestretch != self.cfg.mild_prestretch_strength:
            self.log.warn(
                "Mild prestretch strength clamped: "
                f"{old_mild_prestretch} -> {self.cfg.mild_prestretch_strength}"
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
        except Exception as e:
            self.log.warn(f"[AI] Failed to measure image features: {e}")
            return None


    def _measure_current_quality(self) -> Optional[QualityMetrics]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            return measure_quality_metrics(image_data)
        except Exception as e:
            self.log.warn(f"[AI] Failed to measure image quality metrics: {e}")
            return None


    def _read_image_by_stem(self, stem: str) -> Optional[np.ndarray]:
        try:
            self.cmd_with_check("load", stem)
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                return None
            return np.asarray(image_data)
        except Exception as e:
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
