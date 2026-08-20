"""
Starun Post-processing Pipeline for Pre-stacked Images
Optimized for Siril 1.4+
处理由 Seestar 望远镜机内叠加好的单帧图像

详细流程说明见同目录文档: starun_workflow.md

处理顺序遵循 starless-first 天文后期链路:
  线性阶段: 背景提取 → 色彩校准 → 降噪 → 星点分离
  非线性阶段: Starless 主体拉伸 → 星云增强 → 星点层拉伸与回混 → 导出
"""
import json
import importlib
import hashlib
import errno
import math
import os
import platform
import sys
import types
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple
import copy
import re
import shutil
import subprocess
import textwrap
import threading
import traceback
import time
import zipfile

import cosmic_clarity
import plugin_runner
import review_bundle
import run_manifest
import scunet_denoise
import syqon_starless
import sasp_runner
import stage7_quality
import stage7_repair
import stage8_pixels
import outcome
from logging_utils import PipelineLogger
from stage_support import (
    PluginServiceMixin,
    SaspServiceMixin,
    Stage7ServiceMixin,
    Stage8ServiceMixin,
    StageSupportMixin,
)
from stage6_services import Stage6ServiceMixin
from target_runtime import TargetRuntimeMixin
from processor_runtime import ProcessorRuntimeMixin

from models import (
    AutoTuneResult,
    ImageFeatures,
    InputProfile,
    InputState,
    PipelineConfig,
    PipelineStage,
    QualityMetrics,
    StarSeparationState,
    StageResult,
    TargetType,
)
from image_metrics import (
    _box_blur_gray,
    _to_rgb_float_fullres,
    _to_rgb_float_image,
    format_feature_summary,
    measure_image_features,
    measure_quality_metrics,
)

from save_utils import (
    save_stage_output,
    write_stage_json,
)
from ui_preview import write_display_preview

try:
    import numpy as np
except ImportError as e:
    print(f"错误: 无法导入 numpy 模块: {e}")
    print("请确保 Siril Python 环境包含 numpy")
    sys.exit(1)

try:
    import sirilpy as s
    from sirilpy.exceptions import (
        SirilError, SirilConnectionError, CommandError, DataError
    )
except ImportError as e:
    print(f"错误: 无法导入 sirilpy 模块: {e}")
    print("请确保在 Siril 的脚本编辑器中运行此脚本，或使用 siril -s 命令执行")
    sys.exit(1)

try:
    from sirilpy.enums import CommandStatus
    RETRYABLE_STATUSES = frozenset({
        CommandStatus.CMD_GENERIC_ERROR,
        CommandStatus.CMD_THREAD_RUNNING,
    })
except (ImportError, AttributeError):
    RETRYABLE_STATUSES = frozenset()

from stages.stage1_preparation import run_stage1_preparation
from stages.stage2_view_correction import run_stage2_view_correction
from stages.stage3_background_extraction import run_stage3_background_extraction
from stages.stage4_color_calibration import run_stage4_color_calibration
from stages.stage5_linear_denoise import run_stage5_linear_denoise
from stages.stage7_stretching import run_stage7_stretching
from stages.stage6_star_separation import run_stage6_star_separation
from stages.stage8_nebula_enhancement import run_stage8_nebula_enhancement
from stages.stage9_star_remixing import run_stage9_star_remixing
from stages.stage10_export import run_stage10_export
import task_workspace
import task_plan

try:
    from image_feature_analyzer import (
        analyze_image as analyze_adaptive_image,
        write_safe_preview,
    )
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from target_profiler import build_target_profile
except (ImportError, RuntimeError) as adaptive_import_exc:
    analyze_adaptive_image = None
    write_safe_preview = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage7_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_target_profile = None
    ADAPTIVE_IMPORT_ERROR = adaptive_import_exc
else:
    ADAPTIVE_IMPORT_ERROR = None


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "STARUN_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "STARUN_INPUT_MODE"
ENV_SYQON_TIMEOUT_KEY = "STARUN_SYQON_TIMEOUT_SEC"
ENV_COSMIC_NATIVE_GPU_KEY = "STARUN_COSMIC_NATIVE_GPU"
ENV_COSMIC_CLASSIC_GPU_KEY = "STARUN_COSMIC_CLASSIC_GPU"
ENV_COSMIC_CLASSIC_ENABLE_KEY = "STARUN_COSMIC_CLASSIC_ENABLE"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_STAGE1_PREPARED_RESUME = "stage1_prepared_resume"
INPUT_MODE_LINEAR_RESUME = "stage5_linear_resume"
INPUT_MODE_STAGE2_CORRECTED_RESUME = "stage2_corrected_resume"
STAGE1_PREPARED_INPUT_NAME = "stage1_prepared.fit"
STAGE5_LINEAR_INPUT_NAME = "stage5_linear.fit"
STAGE2_CORRECTED_INPUT_NAME = "stage2_corrected.fit"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)

SIRIL_NATIVE_PROCESS_TERMINATED_MARKER = "[SIRIL_NATIVE_PROCESS_TERMINATED]"
_SIRIL_NATIVE_DEATH_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "EPIPE", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "ENOTCONN", None),
        getattr(errno, "ESHUTDOWN", None),
    )
    if value is not None
)
_SIRIL_NATIVE_DEATH_HINTS = (
    "broken pipe",
    "connection closed",
    "connection was closed",
    "connection reset",
    "connection aborted",
    "connection lost",
    "socket closed",
    "socket is closed",
    "pipe is closed",
    "end of file",
    "unexpected eof",
    "server disconnected",
    "process exited",
    "process has exited",
    "siril has exited",
    "siril process terminated",
)


class SirilNativeProcessTerminated(BaseException):
    """Fatal control flow: the connected Siril native process is no longer usable."""

    def __init__(self, command: str, cause: BaseException):
        self.command = command
        self.cause = cause
        super().__init__(
            f"Siril native process terminated during '{command}': "
            f"{type(cause).__name__}: {cause}"
        )


def _is_siril_native_process_termination(error: BaseException) -> bool:
    if isinstance(
        error,
        (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, EOFError),
    ):
        return True
    if type(error).__name__ == "SirilConnectionError":
        return True
    error_number = getattr(error, "errno", None)
    if error_number in _SIRIL_NATIVE_DEATH_ERRNOS:
        return True
    lowered = str(error).strip().lower()
    return any(hint in lowered for hint in _SIRIL_NATIVE_DEATH_HINTS)


class _FatalSirilInterfaceProxy:
    """Turn native connection death from any sirilpy API into fatal control flow."""

    def __init__(self, owner, interface):
        self._owner = owner
        self._interface = interface

    def __getattr__(self, name: str):
        attribute = getattr(self._interface, name)
        if not callable(attribute):
            return attribute

        def guarded(*args, **kwargs):
            owner = self._owner
            if owner._siril_process_terminated:
                fatal = owner._siril_process_termination_error
                if fatal is None:
                    fatal = SirilNativeProcessTerminated(
                        name,
                        RuntimeError("Siril connection was already closed"),
                    )
                    owner._siril_process_termination_error = fatal
                raise fatal
            try:
                return attribute(*args, **kwargs)
            except Exception as error:
                initial_connect = name == "connect" and not owner._siril_ever_connected
                if (
                    not initial_connect
                    and _is_siril_native_process_termination(error)
                ):
                    owner._raise_siril_native_process_terminated(name, error)
                raise

        return guarded


# ============================================================
# 配置与基础设施
# ============================================================

CLAMP_RULES: list[tuple[str, type, float, float]] = [
    ("max_retries", int, 0, 3),
    ("retry_delay", float, 0.0, 10.0),
    ("stage1_register_fail_ratio_max", float, 0.0, 0.50),
    ("stage2_base_crop_margin", float, 0.0, 0.06),
    ("stage2_edge_black_target", float, 0.03, 0.18),
    ("stage2_adaptive_edge_crop_max_passes", int, 0, 6),
    ("stage2_adaptive_edge_crop_max_extra", float, 0.005, 0.060),
    ("stage2_guard_band_pixels", int, 0, 8),
    ("stage2_center_protect_area_ratio", float, 0.50, 0.95),
    ("bg_samples", int, 12, 32),
    ("bg_tolerance", float, 0.6, 1.8),
    ("bg_smooth", float, 0.2, 1.2),
    ("stage3_compound_min_sample_count", int, 12, 64),
    ("stage3_compound_fit_min_count", int, 8, 56),
    ("stage3_compound_validation_min_count", int, 4, 20),
    ("stage3_compound_validation_ratio", float, 0.15, 0.35),
    ("stage3_compound_score_abs_improvement_min", float, 0.03, 0.15),
    ("stage3_compound_score_rel_improvement_min", float, 0.10, 0.40),
    ("stage3_compound_validation_improvement_min", float, 0.10, 0.40),
    ("stage3_compound_zero_point_abs_max", float, 0.002, 0.010),
    ("stage3_compound_zero_point_rel_max", float, 0.05, 0.15),
    ("stage5_multiscale_denoise_strength", float, 0.10, 1.00),
    ("stage5_multiscale_detail_retention_min", float, 0.70, 0.98),
    ("stage5_multiscale_noise_reduction_min", float, 0.00, 0.50),
    ("stage5_denoise_chroma_noise_growth_max", float, 1.00, 1.50),
    ("stage5_rl_maxstars", int, 20, 1000),
    ("stage5_rl_psf_kernel_size", int, 9, 99),
    ("stage5_rl_iters", int, 1, 40),
    ("stage5_rl_alpha", float, 100.0, 10000.0),
    ("stage5_rl_gdstep", float, 0.00001, 0.01),
    ("stage5_rl_stop", float, 0.0001, 0.05),
    ("stage5_graxpert_deconv_strength", float, 0.20, 0.40),
    ("stage5_graxpert_guard_retry_strength", float, 0.20, 0.30),
    ("stage7_display90_strength", float, 0.50, 0.95),
    ("stage7_display90_reference_chroma_load_ratio_max", float, 1.00, 1.20),
    ("stage7_display90_reference_chroma_load_absolute_max", float, 0.15, 0.50),
    ("asinh_stretch", float, 1.6, 3.6),
    ("asinh_offset", float, 0.0005, 0.006),
    ("ghs_shadowsclip", float, -3.6, -1.8),
    ("ghs_stretchamount", float, 1.0, 2.8),
    ("nebula_saturation", float, 0.0, 0.65),
    ("stage8_core_protection_strength", float, 0.50, 1.00),
    ("stage8_background_denoise_strength", float, 0.0, 0.25),
    ("stage8_faint_nebula_boost_max", float, 0.0, 0.18),
    ("stage8_nebula_contrast_max", float, 0.0, 0.20),
    ("stage8_masked_unsharp_amount_max", float, 0.0, 0.25),
    ("stage8_blue_precontrol_strength", float, 0.0, 1.00),
    ("stage8_bg_std_growth_max", float, 1.00, 1.50),
    ("stage8_texture_artifact_growth_max", float, 1.00, 2.20),
    ("stage8_limited_saturation_max", float, 0.0, 0.10),
    ("stage8_limited_core_exclusion_expand", int, 2, 16),
    ("stage8_limited_halo_texture_growth_max", float, 1.0, 1.50),
    ("stage8_limited_halo_texture_delta_max", float, 0.00001, 0.01000),
    ("stage8_dualband_palette_strength", float, 0.10, 1.00),
    ("stage8_dualband_palette_luma_drift_max", float, 0.001, 0.030),
    ("stage8_dualband_palette_clip_growth_max", float, 0.0, 0.020),
    ("stage8_dualband_palette_quality_warning_tolerance", float, 0.0, 1.0),
    ("star_intensity", float, 0.8, 1.05),
    ("stage9_fallback_intensity_cap", float, 0.75, 1.05),
    ("final_saturation", float, 0.0, 0.25),
    ("stage7_bg_median_min", float, 0.005, 0.080),
    ("stage7_black_pixel_ratio_max", float, 0.10, 0.70),
    ("stage7_highlight_clip_ratio_max", float, 0.001, 0.050),
    ("stage7_star_growth_ratio_max", float, 1.05, 1.80),
    ("stage7_bright_nebula_star_growth_ratio_max", float, 1.05, 1.80),
    ("stage7_bright_nebula_star_mask_expand", int, 1, 8),
    ("stage7_bright_nebula_star_faint_suppression", float, 0.0, 1.0),
    ("stage7_bright_nebula_star_detail_suppression", float, 0.0, 0.60),
    ("stage7_stretch_feedback_retry_max", int, 0, 1),
    ("stage7_starless_masked_rank_drift_p95_max", float, 0.02, 0.50),
    ("stage7_starless_halo_detail_growth_ratio_max", float, 1.05, 4.00),
    ("stage7_starless_halo_detail_delta_min", float, 0.001, 0.10),
    ("stage7_stretch_chroma_noise_score_max", float, 0.10, 0.80),
    ("stage7_stretch_background_mottling_score_max", float, 0.10, 1.00),
    ("stage7_stretch_chroma_load_growth_max", float, 1.00, 3.00),
    ("stage7_stretch_chroma_load_low_absolute_max", float, 0.01, 0.15),
    ("stage7_stretch_chroma_load_low_absolute_tolerance", float, 0.0, 0.01),
    ("stage7_uncalibrated_background_chroma_load_review_max", float, 0.04, 0.50),
    ("stage7_9_quality_advisory_multiplier", float, 1.0, 2.0),
    ("stage7_quality_retry_max", int, 0, 3),
    ("stage6_syqon_regional_texture_ratio_max", float, 1.20, 4.00),
    ("stage6_syqon_regional_texture_sigma_min", float, 3.0, 10.0),
    ("stage6_syqon_regional_affected_ratio_max", float, 0.05, 0.50),
    ("stage7_edge_black_warn", float, 0.04, 0.30),
    ("stage7_bg_median_high", float, 0.08, 0.35),
    ("stage7_bg_std_high", float, 0.020, 0.120),
    ("stage7_bg_noise_ratio_high", float, 0.20, 1.50),
    ("stage7_residual_star_score_max", float, 0.10, 1.20),
    ("stage7_halo_residue_score_max", float, 0.05, 1.00),
    ("stage7_large_galaxy_halo_residue_score_max", float, 0.05, 1.00),
    ("stage7_galaxy_roi_star_clip_percentile", float, 98.0, 99.9),
    ("stage7_galaxy_roi_peak_floor_ratio", float, 0.005, 0.10),
    ("stage7_galaxy_roi_min_extent_ratio", float, 0.004, 0.03),
    ("stage7_galaxy_core_preservation_ratio_min", float, 0.30, 0.95),
    ("stage7_galaxy_core_contrast_ratio_min", float, 0.30, 0.95),
    ("stage7_black_hole_score_max", float, 0.01, 0.35),
    ("stage7_starmask_contamination_max", float, 0.05, 0.80),
    ("stage7_starless_noise_gain_max", float, 1.00, 2.50),
    ("stage7_starmask_coverage_min_ratio", float, 0.05, 0.90),
    ("stage7_starmask_width_ratio_max", float, 1.10, 3.00),
    ("stage7_starless_dynamic_range_min_ratio", float, 0.20, 0.90),
    ("stage7_starless_peak_signal_min", float, 0.0015, 0.0300),
    ("stage7_starless_peak_background_ratio_min", float, 1.5, 12.0),
    ("stage7_starmask_background_floor_percentile", float, 20.0, 80.0),
    ("stage7_starmask_halo_blur_strength", float, 0.0, 0.80),
    ("stage7_starmask_small_star_scale", float, 0.50, 1.00),
    ("stage7_starmask_nebula_suppression", float, 0.0, 0.95),
    ("stage7_starmask_cleanup_noise_sigma", float, 1.0, 6.0),
    ("stage7_starmask_compact_retention_min", float, 0.60, 0.98),
    ("stage7_starmask_diffuse_residual_ratio_max", float, 0.01, 0.50),
    ("stage7_starmask_diffuse_uncertainty_abs", float, 0.0, 0.01),
    (
        "stage7_starmask_diffuse_borderline_star_intensity_scale",
        float,
        0.35,
        1.00,
    ),
    ("stage7_starless_repair_strength", float, 0.0, 0.85),
    ("stage7_starless_halo_repair_strength", float, 0.0, 0.90),
    ("stage7_starless_chroma_denoise_strength", float, 0.0, 0.90),
    ("stage7_starless_repair_max_score_growth", float, 0.0, 0.20),
    ("stage7_starless_repair_chroma_reduction_min", float, 0.05, 0.80),
    ("stage7_starless_repair_chroma_delta_min", float, 0.00001, 0.05000),
    ("stage8_mask_signal_coverage_min", float, 0.001, 0.050),
    ("stage8_blue_excess_max", float, 0.02, 0.30),
    ("stage8_saturation_growth_ratio_max", float, 1.05, 2.50),
    ("stage8_microcontrast_growth_ratio_max", float, 1.05, 2.80),
    ("stage8_highlight_clip_ratio_max", float, 0.001, 0.060),
    ("stage9_unscreen_denominator_floor", float, 0.02, 0.25),
    ("stage9_unscreen_reliable_support_min", float, 0.50, 0.98),
    ("stage9_unscreen_peak_max", float, 0.75, 0.98),
    ("stage9_unscreen_roundtrip_relative_improvement_min", float, 0.0, 0.50),
    ("stage9_unscreen_roundtrip_absolute_improvement_min", float, 0.0, 0.05),
    ("stage9_unscreen_chroma_regression_max", float, 0.0, 0.10),
    ("stage9_unscreen_recovery_regression_max", float, 0.0, 0.10),
    ("stage9_unscreen_wing_regression_max", float, 0.0, 0.15),
    ("stage9_unscreen_fwhm_regression_max", float, 0.0, 0.25),
    ("stage9_psf_fwhm_ratio_min", float, 0.50, 1.00),
    ("stage9_psf_fwhm_ratio_max", float, 1.00, 1.50),
    ("stage9_psf_fwhm_ratio_uncertainty_floor", float, 0.0, 0.01),
    ("stage9_psf_fwhm_ratio_uncertainty_max", float, 0.002, 0.05),
    ("stage9_psf_review_fwhm_ratio_max", float, 1.10, 1.65),
    ("stage9_psf_recovery_target_min", float, 0.50, 1.00),
    ("stage9_psf_recovery_target_max", float, 1.00, 1.50),
    ("stage9_psf_selective_wing_target_ratio", float, 0.93, 1.10),
    ("stage9_psf_selective_wing_strength_max", float, 0.90, 1.25),
    ("stage9_psf_min_sample_count", int, 4, 256),
    ("stage9_psf_support_radius_max", int, 2, 12),
    ("stage9_psf_support_retry_pixels", int, 0, 2),
    ("stage9_stage5_bright_star_fwhm_min", float, 4.0, 20.0),
    ("stage9_stage5_bright_star_support_radius_max", int, 6, 16),
    ("stage9_stage5_bright_star_match_radius", float, 1.0, 8.0),
    ("stage9_starmask_output_adequacy_min", float, 0.25, 0.90),
    ("stage9_catalog_star_visibility_contrast_min", float, 0.0005, 0.02),
    ("stage9_bright_star_visibility_ratio_min", float, 0.50, 1.00),
    ("stage9_highlight_clip_ratio_max", float, 0.001, 0.10),
    ("stage9_highlight_clip_growth_max", float, 0.0, 0.05),
    ("stage9_bright_pixel_growth_max", float, 0.0, 0.10),
    ("stage9_background_lift_max", float, 0.0, 0.05),
    ("stage9_background_mottling_growth_max", float, 1.0, 3.0),
    ("stage9_mottling_exemption_changed_pixel_ratio_max", float, 0.02, 0.35),
    ("stage9_changed_pixel_ratio_max", float, 0.05, 0.80),
    ("stage9_starmask_predicted_change_ratio_max", float, 0.05, 0.60),
    ("stage9_darkening_ratio_max", float, 0.0, 0.05),
    ("stage9_local_component_peak_min", float, 0.002, 0.10),
    ("stage9_local_component_area_max", int, 16, 4096),
    ("stage9_local_component_aspect_ratio_max", float, 1.2, 10.0),
    ("stage9_local_component_fill_ratio_min", float, 0.02, 0.80),
    ("stage9_local_single_pixel_ratio_max", float, 0.0, 0.90),
    ("stage9_local_cyan_blue_peak_min", float, 0.002, 0.10),
    ("stage9_local_cyan_blue_saturation_min", float, 0.20, 0.95),
    ("stage9_local_cyan_blue_component_area_max", int, 4, 2048),
    ("stage9_core_percentile", float, 70.0, 99.0),
    ("stage9_core_color_jump_min", float, 0.03, 0.50),
    ("stage9_core_color_jump_component_area_max", int, 4, 2048),
    ("stage10_chroma_focus_score_min", float, 0.10, 0.80),
    ("stage10_separate_chroma_score_min", float, 0.35, 1.50),
    ("stage10_full_bg_std_min", float, 0.001, 0.10),
    ("stage10_full_mottling_score_min", float, 0.10, 1.00),
    ("stage10_final_denoise_strength", float, 0.05, 0.50),
    ("stage10_star_protection_coverage_max", float, 0.05, 0.60),
    ("stage10_large_galaxy_local_patch_variance_max", float, 0.00022, 0.00100),
    ("stage10_stage9_local_color_risk_strength", float, 0.0, 1.0),
]

DENOISE_MOD_MIN = 0.20
DENOISE_SAFETY_MIN = 0.20
DENOISE_SAFETY_MAX = 0.55
RL_PSF_KERNEL_ODD_INCREMENT = 1
STAGE7_EDGE_BLACK_HIGH_MAX = 0.60
STAGE7_BRIGHT_NEBULA_HALO_SCORE_MAX = 1.20
DYNAMIC_CLAMP_FIELDS = (
    "denoise_mod",
    "stage7_edge_black_high",
    "stage7_large_galaxy_halo_residue_score_max",
    "stage7_bright_nebula_halo_residue_score_max",
)
AUTO_CLAMP_FIELDS = tuple(
    dict.fromkeys(
        [name for name, _value_type, _lower, _upper in CLAMP_RULES]
        + list(DYNAMIC_CLAMP_FIELDS)
    )
)

TARGET_KEYWORDS: Dict[TargetType, Tuple[str, ...]] = {
    TargetType.EMISSION_NEBULA: (
        "m42", "orion", "rosette", "ngc2237", "ngc2244", "lagoon",
        "trifid", "north america", "north_america", "northamerica", "ngc7000",
        "heart", "soul", "ic434", "ic 434", "horsehead", "barnard33", "b33",
        "flame", "ngc2024",
    ),
    TargetType.REFLECTION_NEBULA: (
        "m45", "pleiades", "iris",
    ),
    TargetType.GALAXY: (
        "m31", "andromeda", "m33", "triangulum", "ngc253", "bode", "cigar",
    ),
    TargetType.CLUSTER: (
        "m13", "m44", "double cluster", "double_cluster", "doublecluster",
        "omega cen", "omega_cen", "omegacen",
    ),
    TargetType.PLANETARY: (
        "jupiter", "saturn", "mars", "venus", "moon", "luna", "sun", "solar",
    ),
    TargetType.PLANETARY_NEBULA: (
        "ring", "ring nebula", "ring_nebula", "ringnebula", "dumbbell",
        "m27", "m57", "helix",
    ),
    TargetType.WIDEFIELD: (
        "widefield", "wide field", "milkyway", "milky way",
    ),
}


def _normalize_search_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-\s]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_keyword(normalized_text: str, keyword: str) -> bool:
    kw = _normalize_search_text(keyword)
    if not kw:
        return False
    if f" {kw} " in f" {normalized_text} ":
        return True
    compact_kw = kw.replace(" ", "")
    if any(ch.isdigit() for ch in compact_kw):
        return compact_kw in normalized_text.replace(" ", "")
    return False


def _path_context_text(input_path: Optional[Path]) -> str:
    if input_path is None:
        return ""
    tokens: List[str] = [input_path.stem]
    for parent in list(input_path.parents)[:2]:
        if parent and parent.name:
            tokens.append(parent.name)
    return _normalize_search_text(" ".join(tokens))


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return int(max(lower, min(upper, int(round(value)))))


def _infer_target_type_from_features(feat: ImageFeatures) -> TargetType:
    # 行星类：高频纹理主导、黑背景很少，优先保守处理
    if (
        feat.edge_black_ratio < 0.04
        and feat.bg_median > 0.25
        and feat.object_area_ratio > 0.70
        and feat.star_density < 0.003
    ):
        return TargetType.PLANETARY

    # 弥散结构 + 红/蓝通道占优
    if (
        feat.diffuse_ratio > 0.50
        and feat.object_area_ratio > 0.18
        and feat.red_dominance > 1.12
    ):
        return TargetType.EMISSION_NEBULA
    if (
        feat.diffuse_ratio > 0.50
        and feat.object_area_ratio > 0.18
        and feat.blue_dominance > 1.12
    ):
        return TargetType.REFLECTION_NEBULA

    # 星系：主体集中 + 亮核
    if (
        0.08 <= feat.object_area_ratio <= 0.45
        and feat.core_brightness_ratio > 0.05
        and feat.diffuse_ratio > 0.30
        and feat.star_density < 0.004
    ):
        return TargetType.GALAXY

    # 行星状星云：主体较小但有明显壳层/核心亮部
    if (
        feat.object_area_ratio < 0.20
        and feat.core_brightness_ratio > 0.08
        and feat.diffuse_ratio < 0.55
        and (feat.red_dominance > 1.10 or feat.blue_dominance > 1.10)
    ):
        return TargetType.PLANETARY_NEBULA

    # 星团 / 广域
    if feat.star_density > 0.0045 and feat.diffuse_ratio < 0.22:
        if feat.object_area_ratio < 0.22:
            return TargetType.CLUSTER
        return TargetType.WIDEFIELD
    if feat.star_density > 0.0030 and feat.diffuse_ratio < 0.30:
        return TargetType.WIDEFIELD

    return TargetType.UNKNOWN


def detect_target_type(
    input_path: Optional[Path],
    image_data: Optional[np.ndarray] = None
) -> TargetType:
    """
    先按文件名/路径关键词识别；无法识别时再用图像特征保守推断。
    """
    normalized_context = _path_context_text(input_path)
    for target_type in (
        TargetType.PLANETARY,
        TargetType.PLANETARY_NEBULA,
        TargetType.GALAXY,
        TargetType.CLUSTER,
        TargetType.EMISSION_NEBULA,
        TargetType.REFLECTION_NEBULA,
        TargetType.WIDEFIELD,
    ):
        for keyword in TARGET_KEYWORDS.get(target_type, ()):
            if _contains_keyword(normalized_context, keyword):
                return target_type

    if image_data is None:
        return TargetType.UNKNOWN

    try:
        features = measure_image_features(image_data)
        return _infer_target_type_from_features(features)
    except (TypeError, ValueError, RuntimeError, FloatingPointError):
        return TargetType.UNKNOWN


def clamp_config(cfg: PipelineConfig) -> PipelineConfig:
    """自动调参的统一安全限幅，防止参数进入危险区间。"""
    tuned = copy.deepcopy(cfg)
    for name, value_type, lower, upper in CLAMP_RULES:
        clamp = _clamp_int if value_type is int else _clamp_float
        setattr(tuned, name, clamp(getattr(tuned, name), lower, upper))

    denoise_upper = max(
        DENOISE_SAFETY_MIN,
        min(DENOISE_SAFETY_MAX, float(tuned.denoise_safety_max)),
    )
    tuned.denoise_mod = _clamp_float(
        tuned.denoise_mod,
        DENOISE_MOD_MIN,
        denoise_upper,
    )
    if tuned.stage5_rl_psf_kernel_size % 2 == 0:
        tuned.stage5_rl_psf_kernel_size += RL_PSF_KERNEL_ODD_INCREMENT
    tuned.stage7_edge_black_high = _clamp_float(
        tuned.stage7_edge_black_high,
        tuned.stage7_edge_black_warn,
        STAGE7_EDGE_BLACK_HIGH_MAX,
    )
    tuned.stage7_bright_nebula_halo_residue_score_max = _clamp_float(
        tuned.stage7_bright_nebula_halo_residue_score_max,
        tuned.stage7_halo_residue_score_max,
        STAGE7_BRIGHT_NEBULA_HALO_SCORE_MAX,
    )
    tuned.stage7_large_galaxy_halo_residue_score_max = _clamp_float(
        tuned.stage7_large_galaxy_halo_residue_score_max,
        tuned.stage7_halo_residue_score_max,
        1.0,
    )
    return tuned


def auto_tune_config(
    cfg: PipelineConfig,
    target_type: TargetType,
    feat: ImageFeatures
) -> Tuple[PipelineConfig, AutoTuneResult]:
    """
    自动调参顺序:
      默认配置副本 -> 特征归一化 -> 连续公式生成 -> 统一限幅
    """
    tuned = copy.deepcopy(cfg)
    changed: List[Tuple[str, Any, Any, str]] = []
    notes: List[str] = []

    def set_param(name: str, value: Any, reason: str) -> None:
        old_value = getattr(tuned, name)
        if isinstance(old_value, bool):
            new_value = bool(value)
        elif isinstance(old_value, int) and not isinstance(old_value, bool):
            new_value = int(round(float(value)))
        else:
            new_value = float(value)
        if old_value != new_value:
            setattr(tuned, name, new_value)
            changed.append((name, old_value, new_value, reason))

    noise_score = _clamp_float((feat.bg_std - 0.015) / 0.070, 0.0, 1.0)
    star_score = _clamp_float(feat.star_density / 0.010, 0.0, 1.0)
    diffuse_score = _clamp_float(feat.diffuse_ratio, 0.0, 1.0)
    core_score = _clamp_float((feat.core_brightness_ratio - 0.01) / 0.14, 0.0, 1.0)
    edge_score = _clamp_float((feat.edge_black_ratio - 0.05) / 0.50, 0.0, 1.0)
    red_score = _clamp_float((feat.red_dominance - 1.0) / 0.5, 0.0, 1.0)
    blue_score = _clamp_float((feat.blue_dominance - 1.0) / 0.5, 0.0, 1.0)
    object_score = _clamp_float(feat.object_area_ratio, 0.0, 1.0)

    notes.append(
        "Target type is diagnostic only; parameters are generated from feature formulas."
    )
    if target_type == TargetType.UNKNOWN:
        notes.append("Target type detection is UNKNOWN; feature-formula auto tune still applies.")

    set_param(
        "stage2_base_crop_margin",
        0.01 + 0.03 * edge_score,
        "feature_formula:stage2_base_crop_margin"
    )
    set_param(
        "bg_samples",
        round(30 - 10 * object_score - 6 * star_score + 4 * noise_score),
        "feature_formula:bg_samples"
    )
    set_param(
        "bg_tolerance",
        0.75 + 0.5 * (1.0 - diffuse_score) - 0.15 * star_score,
        "feature_formula:bg_tolerance"
    )
    set_param(
        "bg_smooth",
        0.45 + 0.35 * object_score + 0.20 * noise_score,
        "feature_formula:bg_smooth"
    )
    set_param(
        "denoise_mod",
        0.24 + 0.22 * noise_score - 0.06 * core_score,
        "feature_formula:denoise_mod"
    )
    set_param(
        "asinh_stretch",
        2.05 + 0.9 * diffuse_score - 0.55 * noise_score - 0.45 * core_score,
        "feature_formula:asinh_stretch"
    )
    set_param(
        "asinh_offset",
        0.0009 + 0.0011 * core_score + 0.0006 * red_score,
        "feature_formula:asinh_offset"
    )
    set_param(
        "ghs_shadowsclip",
        -3.2 + 0.8 * core_score + 0.3 * noise_score,
        "feature_formula:ghs_shadowsclip"
    )
    set_param(
        "ghs_stretchamount",
        2.3 - 0.55 * core_score - 0.35 * noise_score,
        "feature_formula:ghs_stretchamount"
    )
    set_param(
        "nebula_saturation",
        0.16 + 0.27 * diffuse_score - 0.12 * star_score
        - 0.18 * red_score + 0.08 * blue_score,
        "feature_formula:nebula_saturation"
    )
    star_intensity = 1.08 - 0.17 * star_score - 0.08 * diffuse_score + 0.05 * core_score
    set_param(
        "star_intensity",
        star_intensity,
        "feature_formula:star_intensity"
    )
    set_param(
        "stage9_fallback_intensity_cap",
        star_intensity - 0.06,
        "feature_formula:stage9_fallback_intensity_cap"
    )
    set_param(
        "final_saturation",
        0.07 + 0.12 * diffuse_score - 0.07 * star_score
        - 0.10 * red_score + 0.05 * blue_score,
        "feature_formula:final_saturation"
    )

    low_signal_emission = (
        target_type == TargetType.EMISSION_NEBULA
        and feat.object_area_ratio < 0.02
        and feat.diffuse_ratio < 0.08
    )
    if low_signal_emission:
        notes.append(
            "low-signal emission nebula tuning: lift nebula/final saturation and cap star remix"
        )
        set_param(
            "asinh_stretch",
            max(float(tuned.asinh_stretch), 1.85),
            "low_signal_emission_nebula:asinh_floor",
        )
        set_param(
            "asinh_offset",
            max(float(tuned.asinh_offset), 0.0018),
            "low_signal_emission_nebula:offset_floor",
        )
        set_param(
            "nebula_saturation",
            max(float(tuned.nebula_saturation), 0.30),
            "low_signal_emission_nebula:nebula_saturation_floor",
        )
        set_param(
            "final_saturation",
            max(float(tuned.final_saturation), 0.12),
            "low_signal_emission_nebula:final_saturation_floor",
        )
        set_param(
            "star_intensity",
            min(float(tuned.star_intensity), 0.95),
            "low_signal_emission_nebula:star_intensity_cap",
        )
        set_param(
            "stage9_fallback_intensity_cap",
            min(float(tuned.stage9_fallback_intensity_cap), 0.90),
            "low_signal_emission_nebula:stage9_fallback_intensity_cap",
        )
        set_param(
            "stage8_mask_signal_coverage_min",
            min(float(tuned.stage8_mask_signal_coverage_min), 0.001),
            "low_signal_emission_nebula:stage8_mask_floor",
        )

    before_clamp = copy.deepcopy(tuned)
    tuned = clamp_config(tuned)
    for name in AUTO_CLAMP_FIELDS:
        old_value = getattr(before_clamp, name)
        new_value = getattr(tuned, name)
        if old_value != new_value:
            changed.append((name, old_value, new_value, "safety_clamp"))

    result = AutoTuneResult(
        target_type=target_type,
        features=feat,
        changed_params=changed,
        notes=notes,
    )
    return tuned, result


def format_config_summary(cfg: PipelineConfig) -> str:
    return (
        "stage2_base_crop_margin={:.3f}, stage2_center_protect={:.2f}, "
        "bg_samples={}, bg_tol={:.2f}, bg_smooth={:.2f}, "
        "denoise={}({:.2f}), asinh=({:.2f},{:.4f}), ghs=({:.2f},{:.2f}), "
        "nebula_sat={:.2f}, star_input_domain=linear, external_prestretch=false, "
        "star_intensity={:.2f}, final_sat={:.2f}"
    ).format(
        cfg.stage2_base_crop_margin,
        cfg.stage2_center_protect_area_ratio,
        cfg.bg_samples,
        cfg.bg_tolerance,
        cfg.bg_smooth,
        cfg.denoise_enabled,
        cfg.denoise_mod,
        cfg.asinh_stretch,
        cfg.asinh_offset,
        cfg.ghs_shadowsclip,
        cfg.ghs_stretchamount,
        cfg.nebula_saturation,
        cfg.star_intensity,
        cfg.final_saturation,
    )


# ============================================================
# 主处理类
# ============================================================

class StarunPostProcessor(
    PluginServiceMixin,
    Stage7ServiceMixin,
    SaspServiceMixin,
    Stage8ServiceMixin,
    Stage6ServiceMixin,
    TargetRuntimeMixin,
    ProcessorRuntimeMixin,
    StageSupportMixin,
):
    """
    Starun 后期处理流水线
    专为望远镜机内叠加好的 .fit 文件设计
    """

    # 非幂等命令 - 不应自动重试
    _NON_IDEMPOTENT = frozenset({
        'stack', 'calibrate', 'register', 'seqapplyreg', 'link',
        'save', 'savetif', 'savepng', 'savejpg', 'pm', 'pcc', 'pyscript',
    })

    # 可重试的 CommandStatus 代码
    _RETRYABLE_STATUS = RETRYABLE_STATUSES
    _RETRYABLE_ERROR_HINTS = (
        'generic error', 'thread running', 'connection', 'timeout', 'busy',
    )
    _SCRIPT_PREREQUISITE_MODULES = {
        # These scripts call ensure_installed() and will emit traceback noise in
        # offline runtime when dependencies are unavailable.
        "GraXpert-AI.py": ("onnx", "onnxruntime", "appdirs", "cv2", "PyQt6"),
        "AutoBGE.py": ("cv2", "scipy", "PyQt6"),
        "CosmicClarity_Sharpen.py": (
            "PyQt6", "tiffile", "lz4", "zstandard", "exifread", "cv2"
        ),
        "CosmicClarity_Denoise.py": (
            "PyQt6", "tiffile", "lz4", "zstandard", "exifread", "cv2"
        ),
        "Starless.py": ("PyQt6", "PySide6", "astropy", "scipy"),
    }

    def __init__(self, config=None):
        self._load_project_env_defaults()
        if config is None:
            self.cfg = PipelineConfig()
        else:
            self.cfg = copy.deepcopy(config)
        self.initial_cfg = copy.deepcopy(self.cfg)
        self.base_cfg = copy.deepcopy(self.initial_cfg)
        self.log = PipelineLogger('DEBUG' if self.cfg.debug_mode else 'INFO')
        self.siril = _FatalSirilInterfaceProxy(self, s.SirilInterface())
        self.results = []
        self._review_requirements: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.work_dir = None
        self.process_dir = None
        self.source_file = None
        self.starless_file = None
        self.starmask_file = None
        self.stretched_name = None
        self.linear_intermediate_path = None
        self.auto_tune_result = None
        self.platesolve_ok = False
        self.workflow_command_used: Dict[str, str] = {}
        self.sasp_starless_exchange: Optional[Path] = None
        self.sasp_starmask_exchange: Optional[Path] = None
        self.main_output_basename_template = RESULT_BASENAME_TEMPLATE
        self._sasp_aberration_module = None
        self._sasp_aberration_module_error: Optional[str] = None
        self._last_scunet_fallback_error: Optional[str] = None
        self._last_aberration_api_error: Optional[str] = None
        self._force_denoise_enabled: Optional[bool] = None
        self._stage1_registration_stats: Optional[Dict[str, float]] = None
        self._last_plugin_script_error: Optional[str] = None
        self._last_syqon_exchange_report: Dict[str, Any] = {}
        self._sasp_stage8_module = None
        self._sasp_stage8_module_error: Optional[str] = None
        self._last_sasp_stage8_error: Optional[str] = None
        self._sasp_star_stretch_module = None
        self._sasp_star_stretch_module_error: Optional[str] = None
        self._last_sasp_star_stretch_error: Optional[str] = None
        self._last_nb_to_rgb_stars_error: Optional[str] = None
        self._stage9_sasp_star_stretch_report: Dict[str, Any] = {}
        self._stage9_nb_to_rgb_stars_report: Dict[str, Any] = {}
        self._last_stage8_masked_diagnostics: Dict[str, Any] = {}
        self._stage8_final_source: str = "stage8_enhanced"
        self._stage8_fallback_used: bool = False
        self._stage8_final_quality: str = "unknown"
        self._stage8_conservative_mode: bool = False
        self._stage8_handoff: Dict[str, Any] = {}
        self._stage8_artistic_palette_applied: bool = False
        self._stage8_palette_report: Dict[str, Any] = {}
        self._stage8_saturation_execution: Dict[str, Any] = {}
        self._stage8_color_quality_report: Dict[str, Any] = {}
        self._stage10_color_rebalance_report: Dict[str, Any] = {}
        self._stage7_selected_quality: Optional[Dict[str, Any]] = None
        self._stage7_closed_form_mtf_reference: Optional[Dict[str, Any]] = None
        self._stage7_matched_domain_transfer: Optional[Dict[str, Any]] = None
        self._stage7_residual_star_score: float = 0.0
        self._stage7_starless_skipped: bool = False
        self._star_separation_state: str = StarSeparationState.PENDING.value
        self._stage6_passthrough_source: Optional[str] = None
        self._stage6_starmask_borderline_review_required: bool = False
        self._bright_core_with_stars_fallback: Dict[str, Any] = {
            "schema": "starun.bright-core-with-stars-fallback.v1",
            "eligible": False,
            "accepted": False,
            "status": "not_evaluated",
        }
        self._review_display_route: bool = False
        self._display_rendition_contract: Dict[str, Any] = {}
        self._stage9_star_intensity_scale: float = 1.0
        self._stage9_star_intensity_reason: str = ""
        self._stage9_bypassed_bad_starless: bool = False
        self._stage9_stars_required: bool = True
        self._stage9_stars_applied: bool = False
        self._stage9_stars_application_mode: str = "pending"
        self._stage9_output_contains_stars: bool = False
        self._stage9_output_withheld: bool = False
        self._stage9_psf_review_required: bool = False
        self._stage9_remix_formally_accepted: bool = False
        self._stage9_star_delivery_contract_accepted: bool = False
        self._stage9_review_candidate_selected: bool = False
        self._stage9_fallback_used: bool = False
        self._stage9_fallback_reason: Optional[str] = None
        self._stage9_final_source: str = ""
        self._star_preserve_target_bypass: bool = False
        self.input_profile: Dict[str, Any] = {}
        self._trusted_input_provenance: Optional[Dict[str, Any]] = None
        self._resume_semantic_context: Optional[Dict[str, Any]] = None
        self._resume_semantic_context_status: str = "not_applicable"
        self._input_state_review_route: bool = False
        self._skip_stage10_color_adjustments: bool = False
        self._stage2_view_review_required: bool = False
        self._background_review_required: bool = False
        self._stage4_color_review_required: bool = False
        self._stage7_background_color_review_required: bool = False
        self._stage7_background_color_review_gate: Dict[str, Any] = {}
        self._stage7_stretch_forced_delivery: bool = False
        self._stage7_forced_delivery_reasons: List[str] = []
        self._channel_semantics: str = "unknown"
        self.channel_profile: Dict[str, Any] = {}
        self.narrowband_channel_mapping: Dict[str, Any] = {}
        self._run_id: Optional[str] = None
        self._processing_plan: Dict[str, Any] = {}
        self._processing_plan_hash: Optional[str] = None
        self._stage8_palette_selection: Dict[str, Any] = {}
        self._pipeline_result_manifest: Dict[str, Any] = {}
        self._pipeline_result_global_status: Optional[str] = None
        self._checkpoint_retention_report: Dict[str, Any] = {}
        self._formal_checkpoint_publish_failures: List[Dict[str, Any]] = []
        self._stage5_denoise_applied: bool = False
        self._saturation_boost_applied: float = 0.0
        self._color_adjustment_ledger: List[Dict[str, Any]] = []
        self._debug_quality_metric_index: int = 0
        self.target_profile: Dict[str, Any] = {}
        self.pipeline_policy: Dict[str, Any] = copy.deepcopy(DEFAULT_POLICY)
        self._target_primary_frozen: bool = False
        self._frozen_primary_target: Dict[str, Any] = {}
        self.color_calibration_report: Dict[str, Any] = {}
        self.input_mode: str = INPUT_MODE_AUTO
        self._stage1_input_mode: str = "unknown"
        self._siril_process_terminated: bool = False
        self._siril_process_termination_error: Optional[
            SirilNativeProcessTerminated
        ] = None
        self._siril_ever_connected: bool = False
        plugin_dir_raw = os.getenv("STARUN_SIRIL_PLUGIN_DIR", "").strip()
        self.siril_plugin_dir = (
            Path(plugin_dir_raw).expanduser()
            if plugin_dir_raw else None
        )

    # --------------------------------------------------
    # 核心辅助方法
    # --------------------------------------------------

    def _is_transient_error(self, error):
        status_code = getattr(error, 'status_code', None)
        if status_code in self._RETRYABLE_STATUS:
            return True
        error_text = str(error).lower()
        return any(token in error_text for token in self._RETRYABLE_ERROR_HINTS)

    def _command_error_debug_detail(self, error):
        details = [f"type={type(error).__name__}"]
        for attr in ("status_code", "status", "code", "errno"):
            value = getattr(error, attr, None)
            if value is not None:
                details.append(f"{attr}={value}")
        text = str(error).strip()
        if text:
            details.append(f"message={self._short_text(text)}")
        return ", ".join(details)

    def _raise_siril_native_process_terminated(self, command: str, error):
        fatal = SirilNativeProcessTerminated(command, error)
        self._siril_process_terminated = True
        self._siril_process_termination_error = fatal
        if hasattr(self.log, "set_sink"):
            self.log.set_sink(None)
        self.log.error(
            f"{SIRIL_NATIVE_PROCESS_TERMINATED_MARKER} "
            f"{fatal}; aborting pipeline without fallback or save"
        )
        raise fatal from error

    def cmd_with_check(self, *args, quiet=False):
        """执行 Siril 命令，带基于状态码与错误文本的智能重试"""
        cmd_str = ' '.join(map(str, args))
        cmd_name = str(args[0]).lower() if args else ''
        max_attempts = self.cfg.max_retries + 1

        if self._siril_process_terminated:
            fatal = self._siril_process_termination_error
            if fatal is None:
                fatal = SirilNativeProcessTerminated(
                    cmd_str,
                    RuntimeError("Siril connection was already closed"),
                )
                self._siril_process_termination_error = fatal
            raise fatal

        for attempt in range(1, max_attempts + 1):
            started = time.time()
            attempt_note = (
                f" attempt={attempt}/{max_attempts}" if max_attempts > 1 else ""
            )
            self.log.debug(f"  → Siril command{attempt_note}: {cmd_str}")
            try:
                self.siril.cmd(*args)
                elapsed = time.time() - started
                self.log.debug(f"  ✓ Siril command ok ({elapsed:.2f}s): {cmd_str}")
                return True
            except CommandError as e:
                elapsed = time.time() - started
                if _is_siril_native_process_termination(e):
                    self._raise_siril_native_process_terminated(cmd_str, e)
                self.log.debug(
                    "  ✗ Siril command error "
                    f"({elapsed:.2f}s): {cmd_str} "
                    f"[{self._command_error_debug_detail(e)}]"
                )
                can_retry = (
                    attempt < max_attempts
                    and cmd_name not in self._NON_IDEMPOTENT
                    and self._is_transient_error(e)
                )
                if can_retry:
                    delay = self.cfg.retry_delay * attempt
                    if quiet:
                        self.log.debug(
                            f"重试 ({attempt}/{self.cfg.max_retries}): {cmd_str}")
                    else:
                        self.log.warn(
                            f"重试 ({attempt}/{self.cfg.max_retries}): {cmd_str}")
                    time.sleep(delay)
                    continue
                if quiet:
                    self.log.debug(f"命令失败: {cmd_str} ({e})")
                else:
                    self.log.error(f"命令失败: {cmd_str}\n    错误: {e}")
                raise
            except (DataError, SirilError) as e:
                elapsed = time.time() - started
                if _is_siril_native_process_termination(e):
                    self._raise_siril_native_process_terminated(cmd_str, e)
                self.log.debug(
                    "  ✗ Siril command failed "
                    f"({elapsed:.2f}s): {cmd_str} "
                    f"[{self._command_error_debug_detail(e)}]"
                )
                if quiet:
                    self.log.debug(f"失败: {cmd_str} ({e})")
                else:
                    self.log.error(f"失败: {cmd_str}\n    错误: {e}")
                raise
            except Exception as e:
                if _is_siril_native_process_termination(e):
                    self._raise_siril_native_process_terminated(cmd_str, e)
                raise

    def _try_cmd(self, *args):
        """尝试执行命令，失败仅返回 False，用于插件能力探测。"""
        try:
            self.cmd_with_check(*args, quiet=True)
            return True
        except (CommandError, SirilError, DataError):
            return False

    def _short_text(self, value, max_len: int = 240) -> str:
        text = str(value).strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _find_external_fit(self, candidate_names):
        """在 work_dir 和 process_dir 中查找外部回写的 FITS 产物。"""
        roots = [p for p in (self.work_dir, self.process_dir) if p]
        for root in roots:
            for name in candidate_names:
                path = root / name
                if path.exists() and path.is_file():
                    return path
        return None

    def _import_external_fit(self, source: Path, target_stem: str):
        """将外部 FITS 复制进 process 目录并载入。"""
        if not self.process_dir:
            return None
        suffix = source.suffix.lower()
        if suffix not in (".fit", ".fits"):
            suffix = ".fit"
        target = self.process_dir / f"{target_stem}{suffix}"
        shutil.copy2(source, target)
        self.cmd_with_check("load", target.stem)
        return target

    def _export_sasp_exchange_files(self):
        """导出给 SetiAstroSuitePro 的交换文件。"""
        if not self.work_dir:
            return
        if self.starless_file and self.starless_file.exists():
            self.sasp_starless_exchange = self.work_dir / "sasp_starless_input.fit"
            shutil.copy2(self.starless_file, self.sasp_starless_exchange)
            self.log.info(f"已导出 Starless 交换文件: {self.sasp_starless_exchange.name}")
        if self.starmask_file and self.starmask_file.exists():
            self.sasp_starmask_exchange = self.work_dir / "sasp_starmask_input.fit"
            shutil.copy2(self.starmask_file, self.sasp_starmask_exchange)
            self.log.info(f"已导出 Starmask 交换文件: {self.sasp_starmask_exchange.name}")

    def _export_linear_intermediate(self):
        export_name = "result_linear"

        try:
            self.cmd_with_check("cd", f'"{self.work_dir}"')
            self.cmd_with_check("save", export_name)
            self.linear_intermediate_path = self.work_dir / f"{export_name}.fit"
            self.log.info(
                f"已导出拉伸前中间文件: {self.linear_intermediate_path.name}")
            return True
        except (CommandError, SirilError) as e:
            self.linear_intermediate_path = None
            self.log.warn(f"导出拉伸前中间文件失败: {e}")
            return False
        finally:
            if self.process_dir:
                self.cmd_with_check("cd", f'"{self.process_dir}"')

    def _require_review(
        self,
        stage: int,
        code: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register one review requirement under its actual owning stage."""

        requirement = outcome.normalize_review_requirement(
            {
                "stage": int(stage),
                "code": str(code or "").strip(),
                "details": dict(details or {}),
            }
        )
        key = (int(requirement["stage"]), str(requirement["code"]))
        registry = getattr(self, "_review_requirements", None)
        if not isinstance(registry, dict):
            registry = {}
            self._review_requirements = registry
        registry[key] = requirement
        # Ownership may only become known after the stage detail was emitted
        # (for example input-state routing and final cross-stage gates).
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
        registry = getattr(self, "_review_requirements", None)
        if not isinstance(registry, dict):
            registry = {}
        self._review_requirements = {
            key: value
            for key, value in registry.items()
            if key[0] != stage_number
        }

    def _stage_review_reasons(self, stage: int) -> List[str]:
        stage_number = int(stage)
        registry = getattr(self, "_review_requirements", None)
        if not isinstance(registry, dict):
            return []
        return [
            str(value["code"])
            for key, value in registry.items()
            if key[0] == stage_number
        ]

    def _review_requirements_payload(
        self,
        *,
        through_stage: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        limit = int(through_stage) if through_stage is not None else 10
        registry = getattr(self, "_review_requirements", None)
        if not isinstance(registry, dict):
            return []
        return [
            copy.deepcopy(value)
            for key, value in sorted(registry.items())
            if key[0] <= limit
        ]

    def _record_stage(
        self,
        name,
        status,
        duration=0.0,
        message='',
        *,
        execution=None,
        fallback_used=False,
        upstream_passthrough=False,
        reason_code='',
        details=None,
        components=None,
        review_reasons=None,
        issues=None,
    ):
        normalized_status = str(status).strip().lower()
        if bool(fallback_used) and normalized_status == "ok":
            normalized_status = "degraded"
        if normalized_status not in outcome.STAGE_STATUSES:
            raise ValueError(f"unsupported stage status: {status!r}")
        normalized_execution = str(
            execution
            or ("skipped" if normalized_status == "skipped" else "completed")
        ).strip().lower()
        if normalized_execution not in outcome.STAGE_EXECUTIONS:
            raise ValueError(f"unsupported stage execution: {execution!r}")
        stage_match = re.match(r"^阶段\s+(\d+)\s*:\s*(.*)$", str(name).strip())
        stage_number = int(stage_match.group(1)) if stage_match else 0
        normalized_review_reasons: List[str] = (
            list(self._stage_review_reasons(stage_number))
            if stage_number in range(1, 11)
            else []
        )
        for raw_reason in list(review_reasons or []):
            if isinstance(raw_reason, Mapping):
                code = str(raw_reason.get("code") or "").strip()
                reason_details = raw_reason.get("details")
            else:
                code = str(raw_reason or "").strip()
                reason_details = None
            if not code:
                continue
            if stage_number <= 0:
                raise ValueError("review reasons require a formal Stage 1-10 label")
            self._require_review(
                stage_number,
                code,
                reason_details if isinstance(reason_details, Mapping) else None,
            )
            if code not in normalized_review_reasons:
                normalized_review_reasons.append(code)
        normalized_components = dict(components or {})
        normalized_issues = [
            outcome.normalize_issue(raw_issue, default_stage=stage_number)
            for raw_issue in list(issues or [])
            if isinstance(raw_issue, Mapping)
        ]
        execution_failure_components = {
            "platesolve",
            "deconvolution",
            "denoise",
            "export",
            "input_source",
        }
        for component_name, raw_component in normalized_components.items():
            if not isinstance(raw_component, Mapping) or str(
                raw_component.get("status") or ""
            ).strip().lower() != "failed":
                continue
            component_code = str(
                raw_component.get("reason_code") or "component_failed"
            ).strip()
            lowered_code = component_code.lower()
            is_execution_error = bool(
                str(component_name) in execution_failure_components
                or any(
                    marker in lowered_code
                    for marker in (
                        "failed",
                        "error",
                        "exception",
                        "execution",
                    )
                )
            )
            if not is_execution_error or any(
                issue["component"] == str(component_name)
                and issue["code"] == component_code
                for issue in normalized_issues
            ):
                continue
            normalized_issues.append(
                outcome.normalize_issue(
                    {
                        "component": str(component_name),
                        "severity": (
                            "fatal" if normalized_status == "failed" else "error"
                        ),
                        "code": component_code,
                        "recovered": normalized_status != "failed",
                        "message": str(
                            raw_component.get("message")
                            or raw_component.get("error")
                            or f"{component_name}: {component_code}"
                        ),
                    },
                    default_stage=stage_number,
                )
            )
        if normalized_status == "failed" and not any(
            issue["severity"] == "fatal" for issue in normalized_issues
        ):
            normalized_issues.append(
                outcome.normalize_issue(
                    {
                        "component": "stage",
                        "severity": "fatal",
                        "code": str(reason_code or "stage_failed"),
                        "recovered": False,
                        "message": str(message or reason_code or "stage failed"),
                    },
                    default_stage=stage_number,
                )
            )
        result = StageResult(
            name,
            normalized_status,
            duration,
            message,
            execution=normalized_execution,
            fallback_used=bool(fallback_used),
            upstream_passthrough=bool(upstream_passthrough),
            reason_code=str(reason_code or ""),
            details=dict(details or {}),
            components=normalized_components,
            review_reasons=normalized_review_reasons,
            issues=normalized_issues,
        )
        self.results.append(result)
        if stage_match:
            stage_title = stage_match.group(2).strip()
            try:
                duration_seconds = max(0.0, float(duration))
            except (TypeError, ValueError):
                duration_seconds = 0.0
            event_status = {
                "ok_skipped_optional": "skipped",
                # Legacy GUI versions do not know the neutral passthrough state.
                "ok_safe_passthrough": "ok",
            }.get(result.display_status, normalized_status)
            self.log.info(
                "[PIPELINE_STAGE_RESULT] "
                f"stage={stage_number} "
                f"status={event_status} "
                f"duration={duration_seconds:.1f} "
                f"title={stage_title}"
            )
            stage_detail = {
                "schema": outcome.PIPELINE_STAGE_DETAIL_SCHEMA_V2,
                "stage": stage_number,
                "title": stage_title,
                "status": result.status,
                "display_status": result.display_status,
                "execution": result.execution,
                "fallback_used": result.fallback_used,
                "upstream_passthrough": result.upstream_passthrough,
                "reason_code": result.reason_code or None,
                "review_required": bool(result.review_reasons),
                "review_reasons": list(result.review_reasons),
                "issues": list(result.issues),
                "duration_seconds": duration_seconds,
                "details": result.details,
                "components": result.components,
            }
            self.log.info(
                "[PIPELINE_STAGE_DETAIL] "
                + json.dumps(
                    stage_detail,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            try:
                self._publish_stage_preview(
                    stage_number,
                    stage_title,
                    result.status,
                )
            except Exception as exc:
                # Preview is an observer-only UI feature. No unexpected decode,
                # filesystem, or runtime error may change a scientific stage result.
                reason = self._short_text(exc, 180)
                self.log.warn(
                    f"Stage {stage_number} 预览观察链路异常，继续主流程："
                    f"{reason}"
                )
                try:
                    self._emit_preview_event(
                        stage_number,
                        stage_title,
                        "unavailable",
                        reason,
                    )
                except Exception:
                    pass
            if (
                stage_number in (2, 5)
                and normalized_status in {"ok", "degraded"}
                and str((getattr(self, "input_profile", {}) or {}).get("state") or "")
                == "linear"
            ):
                self._publish_task_formal_checkpoint(stage_number)

    def _publish_task_formal_checkpoint(self, stage_number: int) -> None:
        """Promote accepted Stage 1/2/5 data before normal process cleanup."""

        manifest_value = str(os.getenv("STARUN_TASK_RUN_MANIFEST", "") or "").strip()
        if not manifest_value or not self.process_dir:
            return
        try:
            contract = task_workspace.stage_contract(stage_number)
            artifact = self.process_dir / contract.primary_artifact
            record = task_workspace.publish_formal_checkpoint(
                run_manifest_path=Path(manifest_value),
                stage_number=stage_number,
                artifact_path=artifact,
                semantic_context=self._formal_checkpoint_semantic_context(
                    stage_number
                ),
            )
            self.log.info(
                "[TaskCheckpoint] published "
                f"stage={stage_number} sha256={str(record.get('sha256') or '')[:12]}"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._formal_checkpoint_publish_failures.append(
                {"stage": stage_number, "error": str(error)}
            )
            self.log.warn(
                f"[TaskCheckpoint] Stage {stage_number} 发布失败；"
                f"本阶段不能跨运行续跑：{error}"
            )

    def _formal_checkpoint_semantic_context(
        self,
        stage_number: int,
    ) -> Optional[Dict[str, Any]]:
        """Freeze upstream meaning that pixel hashes alone cannot recover."""
        if stage_number == 2:
            crop_report = copy.deepcopy(
                getattr(self, "stage2_crop_report", {}) or {}
            )
            original = crop_report.get("original_shape") or {}
            final = (
                crop_report.get("final_shape")
                or crop_report.get("current_shape")
                or {}
            )
            cumulative_crop = crop_report.get("total_crop") or {}
            field_rotation = crop_report.get("field_rotation") or {}
            passes = field_rotation.get("passes") or []
            if isinstance(passes, list) and passes:
                actual_passes = sum(
                    1
                    for item in passes
                    if isinstance(item, Mapping)
                    and isinstance(item.get("applied_crop"), Mapping)
                    and not bool(item.get("rolled_back", False))
                )
            else:
                actual_passes = int(
                    field_rotation.get("actual_passes", 0)
                    if isinstance(field_rotation, Mapping)
                    else 0
                )
            residual = field_rotation.get("residual")
            if not isinstance(residual, Mapping):
                residual = (
                    field_rotation
                    if isinstance(field_rotation, Mapping)
                    else {"accepted": False, "reason": "not_run"}
                )
            original_area = int(original.get("width", 0) or 0) * int(
                original.get("height", 0) or 0
            )
            final_area = int(final.get("width", 0) or 0) * int(
                final.get("height", 0) or 0
            )
            return {
                "schema": task_workspace.RESUME_SEMANTIC_SCHEMA,
                "checkpoint_stage": 2,
                "review_requirements": self._review_requirements_payload(
                    through_stage=2
                ),
                "stage2_crop": {
                    "original_dimensions": copy.deepcopy(dict(original)),
                    "final_dimensions": copy.deepcopy(dict(final)),
                    "cumulative_crop": copy.deepcopy(dict(cumulative_crop)),
                    "retained_area_ratio": (
                        float(final_area / original_area)
                        if original_area > 0
                        else None
                    ),
                    "field_rotation_passes": min(
                        2,
                        actual_passes,
                    ),
                    "field_rotation_max_passes": min(
                        2,
                        max(
                            1,
                            int(
                                field_rotation.get("max_passes", 2)
                                if isinstance(field_rotation, Mapping)
                                else 2
                            ),
                        ),
                    ),
                    "final_residual_detection": copy.deepcopy(dict(residual)),
                },
            }
        if stage_number != 5:
            return None
        return {
            "schema": task_workspace.RESUME_SEMANTIC_SCHEMA,
            "checkpoint_stage": 5,
            "review_requirements": self._review_requirements_payload(
                through_stage=5
            ),
            "channel_semantics": str(
                getattr(self, "_channel_semantics", "unknown") or "unknown"
            ),
            "channel_profile": copy.deepcopy(
                getattr(self, "channel_profile", {}) or {}
            ),
            "narrowband_channel_mapping": copy.deepcopy(
                getattr(self, "narrowband_channel_mapping", {}) or {}
            ),
            "target_profile": copy.deepcopy(
                getattr(self, "target_profile", {}) or {}
            ),
            "pipeline_policy": copy.deepcopy(
                getattr(self, "pipeline_policy", {}) or {}
            ),
            "color_calibration_report": copy.deepcopy(
                getattr(self, "color_calibration_report", {}) or {}
            ),
            "stage5_star_reference_report": copy.deepcopy(
                getattr(self, "_stage5_star_reference_report", {}) or {}
            ),
            "stage5_deconvolution_acceptance": copy.deepcopy(
                getattr(self, "_stage5_deconvolution_acceptance", {}) or {}
            ),
        }

    def _stage_preview_candidates(self, stage: int) -> List[Path]:
        """Return accepted stage artifacts in strict preference order."""
        if not self.process_dir:
            return []
        stems = {
            1: ("stage1_prepared",),
            2: ("stage2_corrected",),
            3: ("stage3_bgremoved",),
            4: ("stage4_color", "stage4_psolved"),
            5: ("stage5_linear",),
            6: ("stage6_starless", "stage6_passthrough"),
            7: ("stage7_stretched", "stage7_review_with_stars"),
            8: (
                "stage8_enhanced",
                "stage8_review_with_stars",
            ),
            9: ("stage9_remixed",),
            10: ("stage10_final",),
        }.get(stage, ())
        return [self.process_dir / f"{stem}.fit" for stem in stems]

    def _emit_preview_event(
        self,
        stage: int,
        title: str,
        status: str,
        payload: str,
    ) -> None:
        self.log.info(
            "[PIPELINE_PREVIEW] "
            + json.dumps(
                {
                    "stage": int(stage),
                    "title": str(title),
                    "status": str(status),
                    "payload": str(payload),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def _publish_stage_preview(self, stage: int, title: str, status: str) -> None:
        """Publish the accepted stage image without changing pipeline quality state."""
        if status not in {"ok", "degraded"}:
            return
        candidates = self._stage_preview_candidates(stage)
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            reason = "accepted stage artifact is unavailable"
            self.log.warn(f"Stage {stage} 预览不可用：{reason}")
            self._emit_preview_event(stage, title, "unavailable", reason)
            return

        try:
            self.cmd_with_check("cd", f'"{source.parent}"')
            self.cmd_with_check("load", source.stem)
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                raise RuntimeError("accepted stage image buffer is empty")
            preview_path = self.process_dir / "ui_preview" / "latest.png"
            write_display_preview(
                image_data,
                preview_path,
                apply_stretch=stage <= 6,
                display_contract=(
                    dict(self._display_rendition_contract)
                    if stage >= 7
                    and bool(self._review_display_route)
                    and isinstance(self._display_rendition_contract, dict)
                    else None
                ),
            )
            self._emit_preview_event(stage, title, "ready", str(preview_path))
        except (
            CommandError,
            DataError,
            SirilError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            reason = self._short_text(exc, 180)
            self.log.warn(f"Stage {stage} 预览生成失败，继续主流程：{reason}")
            self._emit_preview_event(stage, title, "unavailable", reason)
        finally:
            if self.process_dir:
                try:
                    self.cmd_with_check("cd", f'"{self.process_dir}"')
                except (CommandError, DataError, SirilError):
                    pass

    def _record_skipped_stage(self, name: str, message: str) -> None:
        self.log.info(f"{name} 已跳过: {message}")
        self._record_stage(name, "skipped", 0.0, message)

    def _prepare_process_dir(self) -> None:
        if not self.work_dir:
            raise SirilError("工作目录尚未初始化")

        for old_process in self.work_dir.glob("process_*"):
            if old_process.is_dir():
                try:
                    shutil.rmtree(old_process)
                    self.log.info(f"已清理旧归档目录: {old_process.name}")
                except OSError as e:
                    self.log.warn(f"清理旧归档目录失败: {old_process.name} ({e})")

        self.process_dir = self.work_dir / "process"
        if self.process_dir.exists():
            try:
                shutil.rmtree(self.process_dir)
            except OSError as e:
                self.log.warn(f"清理旧 process 目录失败: {e}")
                raise
        self.process_dir.mkdir(exist_ok=True)
        self.log.info(f"处理目录: {self.process_dir}")

    def _prepare_linear_resume_input(self) -> None:
        stage_label = PipelineStage.PREPARATION.label
        self.log.stage_start(stage_label)
        self._prepare_process_dir()

        trusted_path = getattr(self, "_task_resume_checkpoint_path", None)
        linear_path = Path(trusted_path) if trusted_path else None
        if (
            linear_path is None
            or not linear_path.is_file()
            or linear_path.name != STAGE5_LINEAR_INPUT_NAME
        ):
            raise SirilError(
                "Stage 5 续跑缺少 task-run manifest 中已验签的规范断点"
            )

        self.source_file = linear_path
        self.linear_intermediate_path = linear_path
        self._stage1_input_mode = "linear_resume"
        self._stage1_registration_stats = None

        working_file = self.process_dir / "working.fit"
        shutil.copy2(linear_path, working_file)
        self.log.info(f"线性续跑输入: {linear_path.name}")
        self.log.info("已复制线性输入到处理目录")

        self.cmd_with_check("cd", f'"{self.process_dir}"')
        self.cmd_with_check("load", "working")

        stage_saved = self._save_stage_output("stage1_prepared")
        stage_status = "ok" if stage_saved else "degraded"
        message = f"loaded verified Stage 5 checkpoint {linear_path.name}"
        if not stage_saved:
            message += "；stage1 输出保存失败"

        elapsed = self.log.stage_end(stage_label)
        self._record_stage(stage_label, stage_status, elapsed, message)

    def _prepare_stage1_prepared_resume_input(self) -> None:
        stage_label = PipelineStage.PREPARATION.label
        self.log.stage_start(stage_label)
        trusted_path = getattr(self, "_task_resume_checkpoint_path", None)
        source_path = Path(trusted_path) if trusted_path else None
        if (
            source_path is None
            or not source_path.is_file()
            or source_path.name != STAGE1_PREPARED_INPUT_NAME
        ):
            raise SirilError(
                "Stage 1 续跑缺少 task-run manifest 中已验签的规范断点"
            )

        self._prepare_process_dir()
        prepared_file = self.process_dir / STAGE1_PREPARED_INPUT_NAME
        working_file = self.process_dir / "working.fit"
        shutil.copy2(source_path, prepared_file)
        shutil.copy2(source_path, working_file)
        self.source_file = source_path
        self.linear_intermediate_path = None
        self._stage1_input_mode = "stage1_prepared_resume"
        self._stage1_registration_stats = None
        self.cmd_with_check("cd", f'"{self.process_dir}"')
        self.cmd_with_check("load", "working")
        elapsed = self.log.stage_end(stage_label)
        self._record_stage(
            stage_label,
            "ok",
            elapsed,
            f"loaded verified Stage 1 checkpoint {source_path.name}",
        )

    def _prepare_stage2_corrected_resume_input(self) -> None:
        self._record_skipped_stage(
            PipelineStage.PREPARATION.label,
            "skipped by stage2 corrected resume mode",
        )
        stage_label = PipelineStage.VIEW_CORRECTION.label
        self.log.stage_start(stage_label)

        trusted_path = getattr(self, "_task_resume_checkpoint_path", None)
        source_path = Path(trusted_path) if trusted_path else None
        if (
            source_path is None
            or not source_path.is_file()
            or source_path.name != STAGE2_CORRECTED_INPUT_NAME
        ):
            raise SirilError(
                "Stage 2 续跑缺少 task-run manifest 中已验签的规范断点"
            )

        self._prepare_process_dir()
        self.source_file = source_path
        self.linear_intermediate_path = None
        self._stage1_input_mode = "stage2_corrected_resume"
        self._stage1_registration_stats = None

        corrected_file = self.process_dir / STAGE2_CORRECTED_INPUT_NAME
        working_file = self.process_dir / "working.fit"
        shutil.copy2(source_path, corrected_file)
        shutil.copy2(source_path, working_file)
        self.log.info(f"叠加后处理输入: {source_path}")
        self.log.info("已复制 stage2_corrected 输入到处理目录")

        self.cmd_with_check("cd", f'"{self.process_dir}"')
        self.cmd_with_check("load", "stage2_corrected")

        stage_saved = self._save_stage_output("stage2_corrected")
        stage_status = "ok" if stage_saved else "degraded"
        message = f"loaded verified {STAGE2_CORRECTED_INPUT_NAME}; continue from stage3"
        if not stage_saved:
            message += "；stage2 输出保存失败"

        elapsed = self.log.stage_end(stage_label)
        self._record_stage(stage_label, stage_status, elapsed, message)

    def _activate_input_state_review_route(
        self,
        profile: InputProfile,
    ) -> None:
        """Keep an unsafe/unknown input intact and prepare Stage 10 review export."""
        self._input_state_review_route = True
        self._require_review(
            1,
            "input_state_review_required",
            {
                "state": str(profile.state.value),
                "confidence": float(profile.confidence),
            },
        )
        self._skip_stage10_color_adjustments = True
        self.cfg.force_review_only_output = True
        self._star_separation_state = StarSeparationState.REJECTED.value
        self._stage7_starless_skipped = True
        self._stage8_fallback_used = True
        self._stage8_final_quality = "input_state_passthrough"
        self._stage9_stars_required = False
        self._stage9_stars_applied = False
        self._stage9_remix_formally_accepted = False
        self._stage9_review_candidate_selected = False
        self._stage9_stars_application_mode = "not_required_input_state_passthrough"
        self.starless_file = None
        self.starmask_file = None

        source_stem = None
        if self.process_dir:
            for candidate in (
                "stage2_corrected",
                "stage1_prepared",
                "working",
            ):
                if (self.process_dir / f"{candidate}.fit").is_file():
                    source_stem = candidate
                    break
        if source_stem:
            try:
                self.cmd_with_check("load", source_stem)
                if self._save_stage_output("input_state_passthrough"):
                    source_stem = "input_state_passthrough"
            except (CommandError, SirilError) as error:
                self.log.warn(
                    "输入状态复核 checkpoint 保存失败，沿用当前图像: "
                    f"{error}"
                )
        else:
            source_stem = "input_state_passthrough"

        self.stretched_name = source_stem
        self._stage8_final_source = source_stem
        self._stage9_final_source = source_stem
        self.log.warn(
            "[InputProfile] conservative review route active: "
            f"state={profile.state.value}, source={source_stem}"
        )

    def _record_input_state_skipped_stages(
        self,
        profile: InputProfile,
        stages: Tuple[PipelineStage, ...],
    ) -> None:
        reason = (
            f"skipped because input state={profile.state.value} "
            f"confidence={profile.confidence:.2f}; review-only route"
        )
        for stage in stages:
            self._record_skipped_stage(stage.label, reason)

    def _apply_forced_runtime_switches(self):
        if self._force_denoise_enabled is not None:
            if self.cfg.denoise_enabled != self._force_denoise_enabled:
                self.log.info(
                    "[AUTO] Apply forced denoise_enabled="
                    f"{self._force_denoise_enabled} (STARUN_DENOISE_FORCE)"
                )
            self.cfg.denoise_enabled = self._force_denoise_enabled

    def _auto_tune_for_current_input(self):
        """
        自动识别并生成本次任务专用配置:
        默认配置副本 -> 特征归一化/连续公式 -> 安全限幅
        失败时回退到默认配置副本，不中断主流程。
        """
        self.cfg = copy.deepcopy(self.base_cfg)
        self.auto_tune_result = None

        if not self.cfg.auto_tune_enabled:
            self.log.info("[AUTO] Auto tune disabled, using default config")
            self._apply_forced_runtime_switches()
            self._apply_task_processing_parameter_overrides()
            return

        source_hint = self.source_file if isinstance(self.source_file, Path) else None

        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
        except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
            self.log.warn(f"[AUTO] Failed to load image pixel data: {e}")
            self.log.warn("[AUTO] Fallback to default config")
            self.cfg = copy.deepcopy(self.base_cfg)
            self._apply_forced_runtime_switches()
            self._apply_task_processing_parameter_overrides()
            return

        try:
            features = measure_image_features(image_data)
            target_type = detect_target_type(source_hint, image_data=image_data)
            tuned_cfg, result = auto_tune_config(self.cfg, target_type, features)

            self.cfg = tuned_cfg
            self.auto_tune_result = result
            self._apply_forced_runtime_switches()
            self._apply_task_processing_parameter_overrides()

            self.log.info(f"[AUTO] Detected target type: {result.target_type.name}")
            self.log.info(f"[AUTO] Features: {format_feature_summary(result.features)}")
            self.log.info("[AUTO] Param generation model: feature_formula")
            if result.changed_params:
                for name, old_value, new_value, reason in result.changed_params:
                    self.log.info(
                        f"[AUTO] Param change: {name} {old_value} -> {new_value} ({reason})"
                    )
            else:
                self.log.info("[AUTO] Param change: none")

            if result.notes:
                for note in result.notes:
                    if note.startswith(
                        (
                            "Target type is diagnostic only",
                            "Target type detection is UNKNOWN",
                        )
                    ):
                        self.log.info(f"[AUTO] {note}")
                    else:
                        self.log.warn(f"[AUTO] {note}")

            self.log.info(f"[AUTO] Final tuned config: {format_config_summary(self.cfg)}")

            if self.cfg.auto_tune_debug or self.cfg.debug_mode:
                self.log.debug(
                    f"[AUTO] Source hint: {source_hint if source_hint else 'N/A'}"
                )
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            FloatingPointError,
        ) as e:
            self.log.warn(f"[AUTO] Auto tuning failed: {e}")
            self.log.warn("[AUTO] Fallback to default config")
            if self.cfg.debug_mode:
                self.log.debug(traceback.format_exc())
            self.cfg = copy.deepcopy(self.base_cfg)
            self.auto_tune_result = None
            self._apply_forced_runtime_switches()
            self._apply_task_processing_parameter_overrides()

    def connect(self):
        self.log.info("正在连接 Siril...")
        try:
            self.siril.connect()
            self._siril_ever_connected = True
            self.work_dir = Path(self.siril.get_siril_wd())
            self.log.set_file_path(self.work_dir / "starun_pipeline_python.log")
            self.log.set_sink(self.siril.log)
            self.log.info(f"已连接，工作目录: {self.work_dir}")
        except SirilConnectionError as e:
            self.log.error(f"连接失败: {e}")
            self.log.error("请确保 Siril 正在运行")
            raise

    # ========================================
    # 阶段 1: 前期准备
    # ========================================
    def stage1_preparation(self):
        run_stage1_preparation(self)

    def stage2_view_correction(self):
        result = run_stage2_view_correction(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(2)
        return result

    def _run_target_profile_preflight(
        self,
        *,
        source: str,
        metadata_candidates: Optional[Tuple[Any, ...]] = None,
        preview_name: str = "",
    ) -> str:
        """Run target profile detection without adding a separate pipeline stage."""
        messages: List[str] = []
        self.log.info(f"[{source}] Target profiler preflight started")
        try:
            if build_target_profile is None or analyze_adaptive_image is None:
                raise RuntimeError(f"adaptive modules unavailable: {ADAPTIVE_IMPORT_ERROR}")

            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                raise RuntimeError("image buffer is empty")
            features_obj = analyze_adaptive_image(image_data)
            context = self._target_profile_context_text()
            metadata = self._read_fits_header_metadata(
                *(metadata_candidates or ("stage2_corrected", self.source_file))
            )
            auto_hint = self._auto_target_hint()
            if auto_hint:
                metadata["AUTO_TARGET_TYPE"] = auto_hint
            profile = build_target_profile(
                features_obj,
                metadata=metadata,
                context_text=context,
            )
            policy = profile.pop("policy", None)
            policy = self._sync_runtime_policy_from_profile(
                profile,
                source=source,
                policy_candidate=policy if isinstance(policy, dict) else None,
            )
            if profile.get("identity_status") == "conflict":
                self._require_review(
                    3,
                    "target_identity_conflict",
                    {
                        "source": source,
                        "identity_evidence": copy.deepcopy(
                            profile.get("identity_evidence") or {}
                        ),
                    },
                )
            self.log.info(
                f"[{source}] Classification: "
                f"{profile.get('target_type')} confidence={float(profile.get('target_confidence', 0.0)):.2f}"
            )
            if metadata:
                self.log.info(
                    f"[{source}] FITS metadata source: "
                    f"{metadata.get('_header_source', 'unknown')}"
                )
            for diagnostic in profile.get("diagnostics", []) or []:
                self.log.info(f"[{source}] {diagnostic}")
            for warning in profile.get("warnings", []) or []:
                self.log.warn(f"[{source}] {warning}")
            self.log.info(
                f"[{source}] Selected policy: {self._active_policy_name()}"
            )
            self._write_stage_json("target_profile.json", profile)
            self._write_stage_json("pipeline_policy.json", policy)
            if write_safe_preview is not None and self.process_dir and preview_name:
                preview_path = self.process_dir / preview_name
                if write_safe_preview(image_data, preview_path):
                    messages.append(f"{preview_name} generated")
            messages.append(
                f"target_type={profile.get('target_type')}; policy={self._active_policy_name()}"
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
            reason = self._short_text(e, 180)
            self.log.warn(f"[{source}] Target profiler failed, using generic policy: {reason}")
            profile = self._fallback_target_profile(reason)
            policy = profile.pop("policy", copy.deepcopy(DEFAULT_POLICY))
            self.target_profile = profile
            self.pipeline_policy = policy if isinstance(policy, dict) else copy.deepcopy(DEFAULT_POLICY)
            self._write_stage_json("target_profile.json", self.target_profile)
            self._write_stage_json("pipeline_policy.json", self.pipeline_policy)
            messages.append(f"target profiler fallback: {reason}")
        return "；".join(messages)

    # ========================================
    # 阶段 3: 背景提取
    # ========================================
    def stage3_background_extraction(self):
        result = run_stage3_background_extraction(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(3)
        return result

    # ========================================
    # 阶段 4: 色彩校准
    # ========================================
    def stage4_color_calibration(self):
        result = run_stage4_color_calibration(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(4)
        return result

    def stage5_linear_denoise(self):
        result = run_stage5_linear_denoise(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(5)
        return result

    def stage7_stretching(self):
        result = run_stage7_stretching(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(7)
        return result

    # ========================================
    # 阶段 6: 星点分离（starless-first，先于主体拉伸执行）
    # ========================================
    def stage6_star_separation(self):
        result = run_stage6_star_separation(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(6)
        return result

    def stage8_nebula_enhancement(self):
        result = run_stage8_nebula_enhancement(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(8)
        return result

    def stage9_star_remixing(self):
        result = run_stage9_star_remixing(self)
        enforce = getattr(self, "_enforce_stage_failure_action", None)
        if callable(enforce):
            enforce(9)
        return result

    def stage10_export(self):
        run_stage10_export(self)

    # ========================================
    # 清理
    # ========================================
    def _delivery_problem_stage_numbers(self, pipeline_status: str) -> set[int]:
        """Return stages whose compact visual evidence remains useful."""
        problem_stages: set[int] = set()
        for result in self.results:
            if not (
                result.status in {"degraded", "failed"}
                or bool(result.fallback_used)
            ):
                continue
            stage_match = re.match(r"^阶段\s+(\d+)\s*:", str(result.name).strip())
            if stage_match:
                problem_stages.add(int(stage_match.group(1)))

        problem_stages.update(
            int(requirement["stage"])
            for requirement in self._review_requirements_payload()
        )

        if pipeline_status == "success":
            return set()
        if pipeline_status == "failed" and not problem_stages and self.process_dir:
            review_root = self.process_dir / "review_bundles"
            available = []
            if review_root.is_dir():
                for stage_dir in review_root.iterdir():
                    match = re.match(r"^stage(\d+)(?:_|$)", stage_dir.name)
                    if stage_dir.is_dir() and match:
                        available.append(int(match.group(1)))
            if available:
                problem_stages.add(max(available))
        return problem_stages

    def _cleanup_sasp_exchange_files(self) -> int:
        """Remove exact app-owned SASP exchange copies after their last consumer."""
        if not self.work_dir:
            return 0
        removed = 0
        for name in ("sasp_starless_input.fit", "sasp_starmask_input.fit"):
            path = self.work_dir / name
            if not path.is_file():
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as error:
                self.log.warn(f"SASP 交换文件清理失败: {name} ({error})")
        self.sasp_starless_exchange = None
        self.sasp_starmask_exchange = None
        return removed

    def _cleanup_non_delivery_process_images(
        self,
        *,
        kept_review_images: List[str],
    ) -> int:
        """Keep only the live GUI preview and selected compact review evidence."""
        if not self.process_dir:
            return 0
        kept = {Path(path).resolve() for path in kept_review_images}
        kept.add((self.process_dir / "ui_preview" / "latest.png").resolve())
        removed = 0
        for path in self.process_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name == ".DS_Store":
                should_remove = True
            else:
                should_remove = path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                if should_remove and path.resolve() in kept:
                    should_remove = False
            if not should_remove:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as error:
                self.log.warn(f"诊断图片清理失败: {path.name} ({error})")
        return removed

    def _archive_diagnostics(self) -> Optional[Path]:
        """归档 JSON/日志等轻量诊断；可视证据在 process 中分层保留。"""
        if not self.process_dir or not self.process_dir.exists():
            return None

        work_dir = self.work_dir or self.process_dir.parent
        archive_path = work_dir / "starun_diagnostics.zip"
        temporary_path = archive_path.with_suffix(".zip.tmp")
        diagnostic_suffixes = {
            ".json", ".jsonl", ".log", ".txt", ".csv",
        }
        candidates = [
            path
            for path in self.process_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in diagnostic_suffixes
        ]
        log_path = getattr(self.log, "_file_path", None)
        if isinstance(log_path, Path) and log_path.is_file():
            candidates.append(log_path)

        unique_candidates = sorted(set(candidates), key=lambda path: str(path))
        if not unique_candidates:
            return None

        archived_names: List[str] = []
        try:
            temporary_path.unlink(missing_ok=True)
            with zipfile.ZipFile(
                temporary_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                for path in unique_candidates:
                    try:
                        relative = path.relative_to(work_dir)
                    except ValueError:
                        relative = Path("logs") / path.name
                    archive_name = relative.as_posix()
                    archive.write(path, archive_name)
                    archived_names.append(archive_name)
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "schema": "starun.diagnostics.v1",
                            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "files": archived_names,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            temporary_path.replace(archive_path)
            self.log.info(
                f"诊断归档已生成: {archive_path.name} ({len(archived_names)} 个文件)"
            )
            return archive_path
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as e:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.log.warn(f"诊断归档生成失败，继续清理: {e}")
            return None

    def _checkpoint_compaction_preflight(self) -> tuple[bool, str]:
        """Verify completion, delivery, plan, and formal checkpoints."""
        if not self.work_dir:
            return False, "work directory is unavailable"
        plan = run_manifest.load_json(self.work_dir / "processing-plan.json")
        verification = task_plan.verify_processing_plan(plan or {})
        if not verification.get("verified"):
            return False, "processing plan verification failed: " + str(
                verification.get("detail") or "unknown error"
            )
        result = run_manifest.load_json(self.work_dir / "pipeline-result.json")
        if not isinstance(result, Mapping):
            return False, "pipeline result is missing or invalid"
        claimed_hash = str(result.get("manifest_hash") or "")
        unsigned = dict(result)
        unsigned.pop("manifest_hash", None)
        if not claimed_hash or claimed_hash != run_manifest.canonical_payload_hash(
            unsigned
        ):
            return False, "pipeline result hash is invalid"
        if str(result.get("plan_hash") or "") != str(plan.get("plan_hash") or ""):
            return False, "pipeline result does not reference the verified plan"
        if str(result.get("run_id") or "") != str(plan.get("run_id") or ""):
            return False, "pipeline result run_id does not match the verified plan"
        if str(result.get("status") or "") == "failed":
            return False, "pipeline status is failed"
        stage10 = next(
            (
                item
                for item in reversed(self.results)
                if item.name == PipelineStage.EXPORT.label
            ),
            None,
        )
        if stage10 is None or stage10.status == "failed":
            return False, "Stage 10 did not complete successfully"
        outputs = result.get("outputs")
        if not isinstance(outputs, Mapping) or not outputs:
            return False, "no final delivery was recorded"
        verified_outputs = 0
        for record in outputs.values():
            if not isinstance(record, Mapping):
                continue
            relative_path = Path(str(record.get("path") or ""))
            if relative_path.is_absolute():
                continue
            output_path = (self.work_dir / relative_path).resolve()
            try:
                output_path.relative_to(self.work_dir.resolve())
            except ValueError:
                continue
            if str(record.get("sha256") or "") == str(
                run_manifest.sha256_file(output_path) or ""
            ):
                verified_outputs += 1
        if verified_outputs == 0:
            return False, "no final delivery passed SHA-256 verification"
        failures = list(
            getattr(self, "_formal_checkpoint_publish_failures", []) or []
        )
        if failures:
            return False, "formal checkpoint publication failed"

        task_payload = getattr(self, "_task_run_manifest_payload", None)
        if isinstance(task_payload, Mapping):
            manifest_path = getattr(self, "_task_run_manifest_path", None)
            persisted = (
                run_manifest.load_json(manifest_path)
                if isinstance(manifest_path, Path)
                else None
            )
            if not isinstance(persisted, Mapping):
                return False, "task-run manifest is unavailable"
            persisted_hash = str(persisted.get("manifest_hash") or "")
            persisted_unsigned = dict(persisted)
            persisted_unsigned.pop("manifest_hash", None)
            if not persisted_hash or persisted_hash != (
                run_manifest.canonical_payload_hash(persisted_unsigned)
            ):
                return False, "task-run manifest hash is invalid"
            input_state = str(
                (getattr(self, "input_profile", {}) or {}).get("state")
                or "unknown"
            )
            if input_state == "linear":
                task_root = Path(
                    str(task_payload.get("task_directory") or "")
                )
                inspection = task_workspace.inspect_task_workspace(
                    task_root,
                    current_resume_fingerprints=task_payload.get(
                        "checkpoint_fingerprints"
                    ),
                )
                if not inspection.get("verified") or int(
                    inspection.get("resume_after_stage") or 0
                ) != 5:
                    return False, "task-level Stage 5 checkpoint is not verified"
        return True, f"verified {verified_outputs} final delivery file(s)"

    def _preserve_checkpoint_evidence(self, reason: str) -> None:
        """Record why checkpoint compaction was skipped without deleting data."""
        if not bool(getattr(self.cfg, "checkpoint_mode", False)):
            return
        self._checkpoint_retention_report = {
            "requested": True,
            "applied": False,
            "status": "preserved",
            "reason": str(reason or "pipeline did not complete"),
            "debug_mode": bool(self.cfg.debug_mode),
            "task_managed": isinstance(
                getattr(self, "_task_run_manifest_payload", None),
                Mapping,
            ),
        }
        if not self.work_dir:
            return
        try:
            run_manifest.atomic_write_json(
                self.work_dir / "checkpoint-retention.json",
                self._checkpoint_retention_report,
            )
        except (OSError, TypeError, ValueError) as error:
            self.log.warn(f"断点现场保留报告写入失败: {error}")

    def cleanup(self):
        """清理处理目录中的中间文件"""
        self.log.stage_start("清理中间文件")
        checkpoint_mode = bool(getattr(self.cfg, "checkpoint_mode", False))
        task_managed = isinstance(
            getattr(self, "_task_run_manifest_payload", None),
            Mapping,
        )
        if checkpoint_mode:
            preflight_ok, preflight_detail = (
                self._checkpoint_compaction_preflight()
            )
            if not preflight_ok:
                self._preserve_checkpoint_evidence(preflight_detail)
                self.log.warn(
                    "断点保留模式未收敛，完整保留现场：" + preflight_detail
                )
                self.log.stage_end("清理")
                return

        deleted_count = self._cleanup_sasp_exchange_files()

        if not self.process_dir or not self.process_dir.exists():
            if deleted_count:
                self.log.info(f"已清理 {deleted_count} 个 SASP 交换文件")
            elapsed = self.log.stage_end("清理")
            return

        deleted_count += self._cleanup_lightsrc_intermediates()
        pipeline_status = self._pipeline_result_status()
        problem_stage_numbers = self._delivery_problem_stage_numbers(
            pipeline_status
        )
        review_retention = review_bundle.prune_review_bundles(
            self.process_dir / "review_bundles",
            pipeline_status=pipeline_status,
            problem_stage_numbers=problem_stage_numbers,
            preserve_full=bool(self.cfg.debug_mode and not checkpoint_mode),
        )
        deleted_count += len(review_retention.get("removed_images") or [])

        if self.cfg.debug_mode and not checkpoint_mode:
            self._archive_diagnostics()
            if deleted_count:
                self.log.info(
                    f"调试模式: 已清理 {deleted_count} 个预处理/交换文件"
                )
            self.log.info("调试模式: 保留 stage* 等中间文件")
            elapsed = self.log.stage_end("清理")
            return

        deleted_count += self._cleanup_non_delivery_process_images(
            kept_review_images=list(
                review_retention.get("kept_compact_images") or []
            ),
        )
        self._archive_diagnostics()

        standalone_checkpoints = {
            "stage1_prepared.fit",
            "stage2_corrected.fit",
            "stage5_linear.fit",
        }
        cleanup_patterns = (
            ("*.fit", "*.fits", "*.fts", "*.xisf", "*.seq")
            if checkpoint_mode
            else ("*.fit", "*.fits", "*.seq", "*.log", "*.csv", "*.lst")
        )
        kept_checkpoints: List[str] = []
        for pattern in cleanup_patterns:
            for f in self.process_dir.rglob(pattern):
                if (
                    checkpoint_mode
                    and not task_managed
                    and f.parent.resolve() == self.process_dir.resolve()
                    and f.name in standalone_checkpoints
                ):
                    kept_checkpoints.append(f.name)
                    continue
                try:
                    f.unlink()
                    deleted_count += 1
                except OSError as e:
                    self.log.warn(f"清理失败: {f.name} ({e})")

        try:
            remaining = list(self.process_dir.iterdir())
            if not remaining:
                self.process_dir.rmdir()
                self.log.info("已删除空的 process 目录")
            else:
                self.log.info(f"已清理 {deleted_count} 个中间文件")
                self.log.info(
                    f"保留文件: {[f.name for f in remaining]}")
        except OSError as e:
            self.log.warn(f"清理收尾失败: {e}")

        if checkpoint_mode:
            self._checkpoint_retention_report = {
                "requested": True,
                "applied": True,
                "status": "compacted",
                "reason": preflight_detail,
                "debug_mode": bool(self.cfg.debug_mode),
                "task_managed": task_managed,
                "deleted_files": deleted_count,
                "kept_process_checkpoints": sorted(set(kept_checkpoints)),
                "task_checkpoint_stages": [1, 2, 5] if task_managed else [],
            }
            if self.work_dir:
                run_manifest.atomic_write_json(
                    self.work_dir / "checkpoint-retention.json",
                    self._checkpoint_retention_report,
                )

        self.log.stage_end("清理")

    # ========================================
    # 主流程
    # ========================================
    def run(self):
        """执行完整的后期处理流程"""
        try:
            start_time = time.time()

            self._load_project_env_defaults()
            self.cfg = copy.deepcopy(self.initial_cfg)
            self._force_denoise_enabled = None
            self.input_mode = INPUT_MODE_AUTO
            self._siril_process_terminated = False
            self._siril_process_termination_error = None
            self._siril_ever_connected = False
            self._apply_runtime_env_overrides()
            self._load_task_processing_parameters()
            self.base_cfg = copy.deepcopy(self.cfg)

            self.connect()

            try:
                self.cmd_with_check("requires", "1.4.0")
            except (CommandError, SirilError) as e:
                self.log.error(f"版本检查失败: {e}")
                self.log.error("此脚本需要 Siril 1.4.0 或更高版本")
                raise RuntimeError("Siril version check failed") from e

            self.log.info("")
            self.log.info("#" * 50)
            self.log.info("# Starun 后期处理流水线")
            self.log.info("# 适用于望远镜机内叠加好的图像")
            self.log.info("#" * 50)
            if self.siril_plugin_dir:
                self.log.info(f"插件目录: {self.siril_plugin_dir}")
            self.log.info(f"输入模式: {self.input_mode}")
            self.platesolve_ok = False
            self._stage8_final_source = "stage8_enhanced"
            self._stage8_fallback_used = False
            self._stage8_final_quality = "unknown"
            self._stage8_conservative_mode = False
            self._stage8_artistic_palette_applied = False
            self._stage8_palette_report = {}
            self._stage8_saturation_execution = {}
            self._stage8_color_quality_report = {}
            self._stage10_color_rebalance_report = {}
            self._stage7_selected_quality = None
            self._stage7_closed_form_mtf_reference = None
            self._stage7_matched_domain_transfer = None
            self._stage7_residual_star_score = 0.0
            self._stage7_starless_skipped = False
            self._star_separation_state = StarSeparationState.PENDING.value
            self._stage6_passthrough_source = None
            self._stage6_starmask_borderline_review_required = False
            self._bright_core_with_stars_fallback = {
                "schema": "starun.bright-core-with-stars-fallback.v1",
                "eligible": False,
                "accepted": False,
                "status": "not_evaluated",
            }
            self._review_display_route = False
            self._display_rendition_contract = {}
            self._stage7_before_stage6 = True
            self._stage7_starless_first_source = ""
            self._stage9_star_intensity_scale = 1.0
            self._stage9_star_intensity_reason = ""
            self._stage9_bypassed_bad_starless = False
            self._stage9_stars_required = True
            self._stage9_stars_applied = False
            self._stage9_stars_application_mode = "pending"
            self._stage9_output_contains_stars = False
            self._stage9_output_withheld = False
            self._stage9_psf_review_required = False
            self._stage9_remix_formally_accepted = False
            self._stage9_star_delivery_contract_accepted = False
            self._stage9_review_candidate_selected = False
            self._stage9_final_source = ""
            self._last_syqon_exchange_report = {}
            self._star_preserve_target_bypass = False
            self._stage5_denoise_applied = False
            self._saturation_boost_applied = 0.0
            self._color_adjustment_ledger = []
            self.results = []
            self._review_requirements = {}
            self._stage_policy_events = []
            self.input_profile = {}
            self._trusted_input_provenance = None
            self._resume_semantic_context = None
            self._resume_semantic_context_status = "not_applicable"
            self._task_resume_checkpoint_path = None
            self._input_state_review_route = False
            self._skip_stage10_color_adjustments = False
            self._stage2_view_review_required = False
            self._background_review_required = False
            self._stage4_color_review_required = False
            self._stage7_background_color_review_required = False
            self._stage7_background_color_review_gate = {}
            self._stage7_stretch_forced_delivery = False
            self._stage7_forced_delivery_reasons = []
            self._channel_semantics = "unknown"
            self.channel_profile = {}
            self.narrowband_channel_mapping = {}
            self._run_id = None
            self._processing_plan = {}
            self._processing_plan_hash = None
            self._stage8_palette_selection = {}
            self._pipeline_result_manifest = {}
            self._pipeline_result_global_status = None
            self._checkpoint_retention_report = {}
            self._formal_checkpoint_publish_failures = []
            self._final_export_started_at = None
            self._final_output_basenames = ()
            self.target_profile = {}
            self.pipeline_policy = copy.deepcopy(DEFAULT_POLICY)
            self._target_primary_frozen = False
            self._frozen_primary_target = {}

            self._load_trusted_input_provenance_for_resume()

            if self.input_mode == INPUT_MODE_STAGE1_PREPARED_RESUME:
                self._prepare_stage1_prepared_resume_input()
                input_profile = self._resolve_input_profile()
                if input_profile.safe_for_linear_steps:
                    self._publish_task_formal_checkpoint(1)
                self._auto_tune_for_current_input()
                self.stage2_view_correction()
                self._run_target_profile_preflight(
                    source="Stage1 resume processing-plan preflight",
                    metadata_candidates=("stage2_corrected", self.source_file),
                    preview_name="stage3_target_preview.png",
                )
                self._freeze_primary_target()
                if not self._write_processing_plan(input_profile):
                    raise RuntimeError("processing-plan.json could not be frozen")
                if input_profile.safe_for_linear_steps:
                    self.stage3_background_extraction()
                    self.stage4_color_calibration()
                    self.stage5_linear_denoise()
                else:
                    self._record_input_state_skipped_stages(
                        input_profile,
                        (
                            PipelineStage.BACKGROUND_EXTRACTION,
                            PipelineStage.COLOR_CALIBRATION,
                            PipelineStage.LINEAR_DENOISE,
                        ),
                    )
            elif self.input_mode == INPUT_MODE_LINEAR_RESUME:
                self._prepare_linear_resume_input()
                input_profile = self._resolve_input_profile()
                self._auto_tune_for_current_input()
                self._run_target_profile_preflight(
                    source="Linear resume processing-plan preflight",
                    metadata_candidates=(STAGE5_LINEAR_INPUT_NAME, self.source_file),
                    preview_name="stage3_target_preview.png",
                )
                self._apply_trusted_resume_semantics()
                self._freeze_primary_target()
                self._record_skipped_stage(
                    PipelineStage.VIEW_CORRECTION.label,
                    "skipped by linear resume mode",
                )
                if input_profile.safe_for_linear_steps:
                    if not self._write_processing_plan(input_profile):
                        raise RuntimeError("processing-plan.json could not be frozen")
                    self._record_skipped_stage(
                        PipelineStage.BACKGROUND_EXTRACTION.label,
                        "skipped by verified linear resume mode",
                    )
                    self._record_skipped_stage(
                        PipelineStage.COLOR_CALIBRATION.label,
                        "skipped by verified linear resume mode",
                    )
                    self._record_skipped_stage(
                        PipelineStage.LINEAR_DENOISE.label,
                        "skipped by verified linear resume mode",
                    )
                else:
                    if not self._write_processing_plan(input_profile):
                        raise RuntimeError("processing-plan.json could not be frozen")
                    self._record_input_state_skipped_stages(
                        input_profile,
                        (
                            PipelineStage.BACKGROUND_EXTRACTION,
                            PipelineStage.COLOR_CALIBRATION,
                            PipelineStage.LINEAR_DENOISE,
                        ),
                    )
            elif self.input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
                self._apply_trusted_resume_semantics()
                self._prepare_stage2_corrected_resume_input()
                input_profile = self._resolve_input_profile()
                self._auto_tune_for_current_input()
                self._run_target_profile_preflight(
                    source="Stage2 resume processing-plan preflight",
                    metadata_candidates=("stage2_corrected", self.source_file),
                    preview_name="stage3_target_preview.png",
                )
                self._freeze_primary_target()
                if not self._write_processing_plan(input_profile):
                    raise RuntimeError("processing-plan.json could not be frozen")
                if input_profile.safe_for_linear_steps:
                    self.stage3_background_extraction()
                    self.stage4_color_calibration()
                    self.stage5_linear_denoise()
                else:
                    self._record_input_state_skipped_stages(
                        input_profile,
                        (
                            PipelineStage.BACKGROUND_EXTRACTION,
                            PipelineStage.COLOR_CALIBRATION,
                            PipelineStage.LINEAR_DENOISE,
                        ),
                    )
            else:
                # 线性阶段 (1-5)
                self.stage1_preparation()
                input_profile = self._resolve_input_profile()
                if input_profile.safe_for_linear_steps:
                    self._publish_task_formal_checkpoint(1)
                self._auto_tune_for_current_input()
                self.stage2_view_correction()
                self._run_target_profile_preflight(
                    source="Processing-plan preflight",
                    metadata_candidates=("stage2_corrected", self.source_file),
                    preview_name="stage3_target_preview.png",
                )
                self._freeze_primary_target()
                if not self._write_processing_plan(input_profile):
                    raise RuntimeError("processing-plan.json could not be frozen")
                if input_profile.safe_for_linear_steps:
                    self.stage3_background_extraction()
                    self.stage4_color_calibration()
                    self.stage5_linear_denoise()
                else:
                    self._record_input_state_skipped_stages(
                        input_profile,
                        (
                            PipelineStage.BACKGROUND_EXTRACTION,
                            PipelineStage.COLOR_CALIBRATION,
                            PipelineStage.LINEAR_DENOISE,
                        ),
                    )

            if input_profile.safe_for_linear_steps:
                self.stage6_star_separation()
                self.stage7_stretching()
            else:
                self._activate_input_state_review_route(input_profile)
                self._record_input_state_skipped_stages(
                    input_profile,
                    (
                        PipelineStage.STAR_SEPARATION,
                        PipelineStage.STRETCHING,
                    ),
                )
            if input_profile.safe_for_linear_steps:
                self.stage8_nebula_enhancement()
                self.stage9_star_remixing()
            else:
                self._record_input_state_skipped_stages(
                    input_profile,
                    (
                        PipelineStage.NEBULA_ENHANCEMENT,
                        PipelineStage.STAR_REMIXING,
                    ),
                )
            self.stage10_export()
            if bool(getattr(self.cfg, "checkpoint_mode", False)):
                self._checkpoint_retention_report = {
                    "requested": True,
                    "applied": False,
                    "status": "pending_verification",
                    "debug_mode": bool(self.cfg.debug_mode),
                }
            if not self._write_pipeline_result_manifest():
                raise RuntimeError("pipeline-result.json could not be written")
            self.cleanup()
            if not self._write_pipeline_result_manifest():
                raise RuntimeError(
                    "pipeline-result.json could not be updated after retention"
                )

            duration = (time.time() - start_time) / 60

            # 结果汇总
            self.log.info("")
            self.log.info("=" * 60)
            self.log.info("处理结果汇总")
            self.log.info("=" * 60)
            status_icons = {
                'ok': '✓', 'degraded': '⚠', 'failed': '✗', 'skipped': '—',
                'ok_with_fallback': '⚠', 'ok_safe_passthrough': '↪',
                'ok_skipped_optional': '—',
            }
            self.log.info(
                f"  {'阶段':<28} {'状态':<18} {'耗时':>8}")
            self.log.info("-" * 60)
            for r in self.results:
                display_status = r.display_status
                icon = status_icons.get(display_status, status_icons.get(r.status, '?'))
                self.log.info(
                    f"  {icon} {r.name:<26} {display_status:<18} {r.duration:>6.1f}s")
                if r.message:
                    for line in self._summary_message_lines(r.message):
                        self.log.info(f"      {line}")
            self.log.info("-" * 60)
            self.log.info(f"总耗时: {duration:.2f} 分钟")

            failed = [r for r in self.results if r.status == 'failed']
            degraded = [
                r
                for r in self.results
                if r.status == 'degraded' or r.fallback_used
            ]
            if failed:
                self.log.warn(f"{len(failed)} 个阶段失败")
            if degraded:
                self.log.warn(f"{len(degraded)} 个阶段降级或回退完成")
            if not failed and not degraded:
                self.log.info("所有阶段成功完成")
            self.log.info(
                "[PIPELINE_RUN_SUMMARY] "
                f"failed={len(failed)} degraded={len(degraded)}"
            )

            self.log.info(f"输出目录: {self.work_dir}")
            self.log.info("生成文件:")
            output_manifest = getattr(self, "_output_color_manifest", {}) or {}
            actual_outputs = [
                item
                for item in (output_manifest.get("artifacts") or [])
                if bool(item.get("exists", False))
                and Path(str(item.get("path") or "")).is_file()
            ]
            logged_paths = set()
            for artifact in actual_outputs:
                path = Path(str(artifact["path"]))
                if path in logged_paths:
                    continue
                logged_paths.add(path)
                self.log.info(f"  - {path.name} ({artifact.get('format', 'file')})")
            if (
                self.linear_intermediate_path
                and Path(self.linear_intermediate_path).is_file()
                and Path(self.linear_intermediate_path) not in logged_paths
            ):
                self.log.info(
                    f"  - {Path(self.linear_intermediate_path).name} "
                    "(拉伸前线性中间文件)"
                )
            if not actual_outputs and not (
                self.linear_intermediate_path
                and Path(self.linear_intermediate_path).is_file()
            ):
                self.log.warn("  - 本轮没有成功发布可列出的输出文件")
        except SirilNativeProcessTerminated as e:
            self._preserve_checkpoint_evidence(
                f"Siril native process terminated: {e}"
            )
            self.log.error(
                "Siril 原生进程或连接已终止；当前流水线立即中止，"
                "不会继续降级、保存或执行后续阶段。"
            )
            raise
        except KeyboardInterrupt:
            self.log.warn("用户中断操作")
            self._preserve_checkpoint_evidence("user interrupted")
            try:
                self._write_pipeline_result_manifest(failure_reason="user interrupted")
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as manifest_error:
                self.log.warn(f"中断结果清单写入失败: {manifest_error}")
            raise
        except Exception as e:
            self.log.error(f"程序中断: {e}")
            self.log.error(traceback.format_exc())
            self._preserve_checkpoint_evidence(str(e))
            try:
                self._write_pipeline_result_manifest(failure_reason=str(e))
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as manifest_error:
                self.log.warn(f"失败结果清单写入失败: {manifest_error}")
            raise
        finally:
            if not self._siril_process_terminated and self._siril_ever_connected:
                try:
                    self.siril.disconnect()
                except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                    self.log.warn(f"Siril 断开连接失败: {e}")


if __name__ == "__main__":
    StarunPostProcessor().run()
