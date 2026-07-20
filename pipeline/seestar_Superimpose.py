"""
SeeStar Post-processing Pipeline for Pre-stacked Images
Optimized for Siril 1.4+
处理由 Seestar 望远镜机内叠加好的单帧图像

详细流程说明见同目录文档: seestar_Superimpose_workflow.md

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
from typing import Any, Dict, List, Optional, Tuple
import copy
import re
import shutil
import subprocess
import tempfile
import textwrap
import threading
import traceback
import time
import zipfile

import ai_advisory
import cosmic_clarity
import plugin_runner
import scunet_denoise
import syqon_starless
import sasp_runner
import stage7_quality
import stage7_repair
import stage8_pixels
from logging_utils import PipelineLogger
from processor_services import (
    AiPostServiceMixin,
    PluginServiceMixin,
    SaspServiceMixin,
    Stage7ServiceMixin,
    Stage8ServiceMixin,
    Stage6ServiceMixin,
    TargetRuntimeMixin,
    ProcessorRuntimeMixin,
    StageSupportMixin,
)

from models import (
    AutoTuneResult,
    ImageFeatures,
    PipelineCheckpoint,
    PipelineConfig,
    PipelineStage,
    QualityMetrics,
    Stage6StretchStrategy,
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
    write_ai_raw_response,
    write_png_rgb16,
    write_stage_json,
)
from ui_preview import write_raw_preview

try:
    from stage11_ai_postprocess import run_stage11_ai_postprocess
    STAGE11_IMPORT_ERROR = None
except (ImportError, RuntimeError) as stage11_import_exc:
    run_stage11_ai_postprocess = None
    STAGE11_IMPORT_ERROR = stage11_import_exc

try:
    from ai_artistic_derivative import run_ai_artistic_derivative
    AI_ARTISTIC_IMPORT_ERROR = None
except (ImportError, RuntimeError) as artistic_import_exc:
    run_ai_artistic_derivative = None
    AI_ARTISTIC_IMPORT_ERROR = artistic_import_exc

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
from stages.stage6_stretching import run_stage6_stretching
from stages.stage7_star_separation import run_stage7_star_separation
from stages.stage8_nebula_enhancement import run_stage8_nebula_enhancement
from stages.stage9_star_remixing import run_stage9_star_remixing
from stages.stage10_export import run_stage10_export

try:
    from image_feature_analyzer import (
        analyze_image as analyze_adaptive_image,
        write_safe_preview,
    )
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from quality_gate import evaluate_pre_starless_gate
    from stretch_candidate_evaluator import (
        build_candidate_spec,
        candidate_modes,
        choose_best as choose_best_stretch_candidate,
        score_candidate as score_stretch_candidate,
    )
    from target_profiler import build_target_profile
except (ImportError, RuntimeError) as adaptive_import_exc:
    analyze_adaptive_image = None
    write_safe_preview = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage6_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    evaluate_pre_starless_gate = None
    build_candidate_spec = None
    candidate_modes = None
    choose_best_stretch_candidate = None
    score_stretch_candidate = None
    build_target_profile = None
    ADAPTIVE_IMPORT_ERROR = adaptive_import_exc
else:
    ADAPTIVE_IMPORT_ERROR = None


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "SEESTAR_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "SEESTAR_INPUT_MODE"
ENV_SYQON_GPU_KEY = "SEESTAR_SYQON_GPU"
ENV_SYQON_TIMEOUT_KEY = "SEESTAR_SYQON_TIMEOUT_SEC"
ENV_COSMIC_NATIVE_GPU_KEY = "SEESTAR_COSMIC_NATIVE_GPU"
ENV_COSMIC_CLASSIC_GPU_KEY = "SEESTAR_COSMIC_CLASSIC_GPU"
ENV_COSMIC_CLASSIC_ENABLE_KEY = "SEESTAR_COSMIC_CLASSIC_ENABLE"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "result_linear_resume"
INPUT_MODE_STAGE2_CORRECTED_RESUME = "stage2_corrected_resume"
INPUT_MODE_STAGE4_PSOLVED_RESUME = "stage4_psolved_resume"
LINEAR_RESUME_INPUT_NAME = "result_linear.fit"
STAGE2_CORRECTED_INPUT_NAME = "stage2_corrected.fit"
STAGE4_PSOLVED_INPUT_NAME = "stage4_psolved.fit"
DEFAULT_AI_PROMPT = (
    "Conservative deep-sky astrophotography enhancement only. "
    "Preserve astronomical realism, faint structures, and natural star colors. "
    "Do not invent objects, do not oversaturate, do not clip black background, "
    "do not increase star size, and avoid halos or artificial sharpening artifacts."
)
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

AUTO_CLAMP_FIELDS = (
    "crop_margin",
    "stage2_edge_black_target",
    "stage2_adaptive_edge_crop_max_passes",
    "stage2_adaptive_edge_crop_max_extra",
    "stage2_guard_band_pixels",
    "stage2_color_artifact_max_crop",
    "bg_samples",
    "bg_tolerance",
    "bg_smooth",
    "bg_std_worsen_ratio_max",
    "bg_median_drop_ratio_min",
    "bg_object_preserve_ratio_min",
    "bg_edge_black_rise_max",
    "bg_star_preserve_ratio_min",
    "bg_nebula_mean_change_max",
    "denoise_mod",
    "stage5_builtin_denoise_mod",
    "stage5_rl_maxstars",
    "stage5_rl_psf_kernel_size",
    "stage5_rl_iters",
    "stage5_rl_alpha",
    "stage5_rl_gdstep",
    "stage5_rl_stop",
    "stage5_graxpert_deconv_strength",
    "asinh_stretch",
    "asinh_offset",
    "ghs_shadowsclip",
    "ghs_stretchamount",
    "nebula_saturation",
    "stage8_core_protection_strength",
    "stage8_background_denoise_strength",
    "stage8_faint_nebula_boost_max",
    "stage8_nebula_contrast_max",
    "stage8_masked_unsharp_amount_max",
    "stage8_blue_precontrol_strength",
    "stage8_bg_std_growth_max",
    "stage8_texture_artifact_growth_max",
    "star_intensity",
    "star_fallback_intensity",
    "final_saturation",
    "ai_timeout_sec",
    "ai_strength",
    "ai_artistic_timeout_sec",
    "ai_bg_median_delta_max",
    "ai_color_ratio_delta_max",
    "ai_core_growth_ratio_max",
    "ai_star_growth_ratio_max",
    "stage6_bg_median_min",
    "stage6_black_pixel_ratio_max",
    "stage6_highlight_clip_ratio_max",
    "stage6_star_growth_ratio_max",
    "stage7_stretch_chroma_noise_score_max",
    "stage7_stretch_background_mottling_score_max",
    "stage7_stretch_chroma_load_growth_max",
    "stage7_stretch_chroma_load_low_absolute_max",
    "stage7_quality_retry_max",
    "stage7_edge_black_warn",
    "stage7_edge_black_high",
    "stage7_bg_median_high",
    "stage7_bg_std_high",
    "stage7_bg_noise_ratio_high",
    "stage7_residual_star_score_max",
    "stage7_halo_residue_score_max",
    "stage7_bright_nebula_halo_residue_score_max",
    "stage7_black_hole_score_max",
    "stage7_starmask_contamination_max",
    "stage7_starless_noise_gain_max",
    "stage7_starmask_coverage_min_ratio",
    "stage7_starmask_width_ratio_max",
    "stage7_starless_dynamic_range_min_ratio",
    "stage7_starless_peak_signal_min",
    "stage7_starless_peak_background_ratio_min",
    "stage7_starmask_background_floor_percentile",
    "stage7_starmask_halo_blur_strength",
    "stage7_starmask_small_star_scale",
    "stage7_starmask_nebula_suppression",
    "stage7_starmask_cleanup_noise_sigma",
    "stage7_starmask_compact_retention_min",
    "stage7_starmask_diffuse_residual_ratio_max",
    "mild_prestretch_strength",
    "stage7_conservative_asinh_stretch",
    "stage7_ultra_conservative_asinh_stretch",
    "stage7_soft_starless_asinh_stretch",
    "stage7_conservative_asinh_offset",
    "stage7_starless_repair_strength",
    "stage7_starless_halo_repair_strength",
    "stage7_starless_chroma_denoise_strength",
    "stage7_starless_repair_max_score_growth",
    "stage7_starless_repair_chroma_reduction_min",
    "stage7_starless_repair_chroma_delta_min",
    "stage8_mask_signal_coverage_min",
    "stage8_blue_excess_max",
    "stage8_saturation_growth_ratio_max",
    "stage8_microcontrast_growth_ratio_max",
    "stage8_highlight_clip_ratio_max",
    "stage9_highlight_clip_ratio_max",
    "stage9_highlight_clip_growth_max",
    "stage9_bright_pixel_growth_max",
    "stage9_background_lift_max",
    "stage9_background_mottling_growth_max",
    "stage9_mottling_exemption_changed_pixel_ratio_max",
    "stage9_changed_pixel_ratio_max",
    "stage9_starmask_predicted_change_ratio_max",
    "stage9_darkening_ratio_max",
    "stage9_local_component_peak_min",
    "stage9_local_component_area_max",
    "stage9_local_component_aspect_ratio_max",
    "stage9_local_component_fill_ratio_min",
    "stage9_local_single_pixel_ratio_max",
    "stage9_local_cyan_blue_peak_min",
    "stage9_local_cyan_blue_saturation_min",
    "stage9_local_cyan_blue_component_area_max",
    "stage9_core_percentile",
    "stage9_core_color_jump_min",
    "stage9_core_color_jump_component_area_max",
    "stage10_chroma_focus_score_min",
    "stage10_separate_chroma_score_min",
    "stage10_full_bg_std_min",
    "stage10_full_mottling_score_min",
    "stage10_stage9_local_color_risk_strength",
)

CLAMP_RULES: list[tuple[str, type, float, float]] = [
    ("crop_margin", float, 0.0, 0.06),
    ("stage2_edge_black_target", float, 0.03, 0.18),
    ("stage2_adaptive_edge_crop_max_passes", int, 0, 6),
    ("stage2_adaptive_edge_crop_max_extra", float, 0.005, 0.060),
    ("stage2_guard_band_pixels", int, 0, 8),
    ("stage2_color_artifact_max_crop", float, 0.05, 0.25),
    ("bg_samples", int, 12, 32),
    ("bg_tolerance", float, 0.6, 1.8),
    ("bg_smooth", float, 0.2, 1.2),
    ("bg_std_worsen_ratio_max", float, 1.0, 1.4),
    ("bg_median_drop_ratio_min", float, 0.05, 0.60),
    ("bg_object_preserve_ratio_min", float, 0.20, 0.90),
    ("bg_edge_black_rise_max", float, 0.10, 0.60),
    ("bg_star_preserve_ratio_min", float, 0.50, 1.00),
    ("bg_nebula_mean_change_max", float, 0.02, 0.35),
    ("stage5_builtin_denoise_mod", float, 0.20, 0.55),
    ("stage5_rl_maxstars", int, 20, 1000),
    ("stage5_rl_psf_kernel_size", int, 9, 99),
    ("stage5_rl_iters", int, 1, 40),
    ("stage5_rl_alpha", float, 100.0, 10000.0),
    ("stage5_rl_gdstep", float, 0.00001, 0.01),
    ("stage5_rl_stop", float, 0.0001, 0.05),
    ("stage5_graxpert_deconv_strength", float, 0.20, 0.40),
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
    ("star_intensity", float, 0.8, 1.05),
    ("star_fallback_intensity", float, 0.75, 1.05),
    ("final_saturation", float, 0.05, 0.25),
    ("ai_timeout_sec", int, 15, 300),
    ("ai_strength", float, 0.05, 0.25),
    ("ai_artistic_timeout_sec", int, 30, 600),
    ("ai_bg_median_delta_max", float, 0.01, 0.06),
    ("ai_color_ratio_delta_max", float, 0.08, 0.35),
    ("ai_core_growth_ratio_max", float, 1.05, 1.80),
    ("ai_star_growth_ratio_max", float, 1.05, 1.80),
    ("stage6_bg_median_min", float, 0.005, 0.080),
    ("stage6_black_pixel_ratio_max", float, 0.10, 0.70),
    ("stage6_highlight_clip_ratio_max", float, 0.001, 0.050),
    ("stage6_star_growth_ratio_max", float, 1.05, 1.80),
    ("stage7_stretch_chroma_noise_score_max", float, 0.10, 0.80),
    ("stage7_stretch_background_mottling_score_max", float, 0.10, 1.00),
    ("stage7_stretch_chroma_load_growth_max", float, 1.00, 3.00),
    ("stage7_stretch_chroma_load_low_absolute_max", float, 0.01, 0.15),
    ("stage7_quality_retry_max", int, 0, 3),
    ("stage7_edge_black_warn", float, 0.04, 0.30),
    ("stage7_bg_median_high", float, 0.08, 0.35),
    ("stage7_bg_std_high", float, 0.020, 0.120),
    ("stage7_bg_noise_ratio_high", float, 0.20, 1.50),
    ("stage7_residual_star_score_max", float, 0.10, 1.20),
    ("stage7_halo_residue_score_max", float, 0.05, 1.00),
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
    ("mild_prestretch_strength", float, 1.05, 1.80),
    ("stage7_conservative_asinh_stretch", float, 1.60, 2.60),
    ("stage7_conservative_asinh_offset", float, 0.0005, 0.0060),
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
    ("stage10_stage9_local_color_risk_strength", float, 0.0, 1.0),
]

DENOISE_MOD_MIN = 0.20
DENOISE_SAFETY_MIN = 0.20
DENOISE_SAFETY_MAX = 0.55
RL_PSF_KERNEL_ODD_INCREMENT = 1
STAGE7_EDGE_BLACK_HIGH_MAX = 0.60
STAGE7_BRIGHT_NEBULA_HALO_SCORE_MAX = 1.20
STAGE7_ULTRA_CONSERVATIVE_ASINH_MIN = 1.20
STAGE7_SOFT_STARLESS_ASINH_MIN = 1.05
DYNAMIC_CLAMP_FIELDS = (
    "denoise_mod",
    "stage7_edge_black_high",
    "stage7_bright_nebula_halo_residue_score_max",
    "stage7_ultra_conservative_asinh_stretch",
    "stage7_soft_starless_asinh_stretch",
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
    tuned.stage7_ultra_conservative_asinh_stretch = _clamp_float(
        tuned.stage7_ultra_conservative_asinh_stretch,
        STAGE7_ULTRA_CONSERVATIVE_ASINH_MIN,
        tuned.stage7_conservative_asinh_stretch,
    )
    tuned.stage7_soft_starless_asinh_stretch = _clamp_float(
        tuned.stage7_soft_starless_asinh_stretch,
        STAGE7_SOFT_STARLESS_ASINH_MIN,
        tuned.stage7_ultra_conservative_asinh_stretch,
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
        "crop_margin",
        0.01 + 0.03 * edge_score,
        "feature_formula:crop_margin"
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
        "denoise_enabled",
        noise_score > 0.35,
        "feature_formula:denoise_enabled"
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
        "star_fallback_intensity",
        star_intensity - 0.06,
        "feature_formula:star_fallback_intensity"
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
            "star_fallback_intensity",
            min(float(tuned.star_fallback_intensity), 0.90),
            "low_signal_emission_nebula:star_fallback_intensity_cap",
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
        "crop_margin={:.3f}, bg_samples={}, bg_tol={:.2f}, bg_smooth={:.2f}, "
        "denoise={}({:.2f}), asinh=({:.2f},{:.4f}), ghs=({:.2f},{:.2f}), "
        "nebula_sat={:.2f}, star_mode={}, star_input_domain=linear, "
        "star_intensity={:.2f}, final_sat={:.2f}, "
        "ai_enabled={}, ai_strength={:.2f}"
    ).format(
        cfg.crop_margin,
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
        cfg.star_separation_mode,
        cfg.star_intensity,
        cfg.final_saturation,
        cfg.ai_post_enabled,
        cfg.ai_strength,
    )


# ============================================================
# 主处理类
# ============================================================

class SeestarPostProcessor(
    PluginServiceMixin,
    Stage7ServiceMixin,
    SaspServiceMixin,
    Stage8ServiceMixin,
    AiPostServiceMixin,
    Stage6ServiceMixin,
    TargetRuntimeMixin,
    ProcessorRuntimeMixin,
    StageSupportMixin,
):
    """
    Seestar 后期处理流水线
    专为望远镜机内叠加好的 .fit 文件设计
    """

    # 非幂等命令 - 不应自动重试
    _NON_IDEMPOTENT = frozenset({
        'stack', 'calibrate', 'register', 'seqapplyreg', 'link',
        'save', 'savetif', 'savepng', 'savejpg', 'pm',
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
        "SyQon-Starless.py": ("PyQt6", "PySide6", "astropy", "scipy"),
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
        self.work_dir = None
        self.process_dir = None
        self.source_file = None
        self.starless_file = None
        self.starmask_file = None
        self.stretched_name = None
        self.linear_intermediate_path = None
        self.auto_tune_result = None
        self.platesolve_ok = False
        self.ai_outputs_generated = False
        self.ai_artistic_output_generated = False
        self.ai_artistic_output_path: Optional[Path] = None
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
        self._ai_plan_parse_fallback: bool = False
        self._ai_plan_parse_fallback_reason: Optional[str] = None
        self._ai_stage_circuit_breaker: Dict[str, str] = {}
        self._ai_raw_response_counter: int = 0
        self._last_plugin_script_error: Optional[str] = None
        self._sasp_stage8_module = None
        self._sasp_stage8_module_error: Optional[str] = None
        self._last_sasp_stage8_error: Optional[str] = None
        self._last_stage8_masked_diagnostics: Dict[str, Any] = {}
        self._stage8_final_source: str = "starless_enhanced"
        self._stage8_fallback_used: bool = False
        self._stage8_final_quality: str = "unknown"
        self._stage8_conservative_mode: bool = False
        self._stage7_selected_quality: Optional[Dict[str, Any]] = None
        self._stage7_residual_star_score: float = 0.0
        self._stage7_starless_skipped: bool = False
        self._stage9_star_intensity_scale: float = 1.0
        self._stage9_star_intensity_reason: str = ""
        self._stage9_bypassed_bad_starless: bool = False
        self._stage9_stars_required: bool = True
        self._stage9_stars_applied: bool = False
        self._stage9_stars_application_mode: str = "pending"
        self._stage9_final_source: str = ""
        self._star_preserve_target_bypass: bool = False
        self._stage5_denoise_applied: bool = False
        self._saturation_boost_applied: float = 0.0
        self._debug_quality_metric_index: int = 0
        self.target_profile: Dict[str, Any] = {}
        self.pipeline_policy: Dict[str, Any] = copy.deepcopy(DEFAULT_POLICY)
        self.pre_starless_gate_report: Dict[str, Any] = {}
        self.color_calibration_report: Dict[str, Any] = {}
        self.input_mode: str = INPUT_MODE_AUTO
        self._stage1_input_mode: str = "unknown"
        self._siril_process_terminated: bool = False
        self._siril_process_termination_error: Optional[
            SirilNativeProcessTerminated
        ] = None
        self._siril_ever_connected: bool = False
        plugin_dir_raw = os.getenv("SEESTAR_SIRIL_PLUGIN_DIR", "").strip()
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

    def _record_stage(self, name, status, duration=0.0, message=''):
        result = StageResult(name, status, duration, message)
        self.results.append(result)
        stage_match = re.match(r"^阶段\s+(\d+)\s*:\s*(.*)$", str(name).strip())
        if stage_match:
            stage_number = int(stage_match.group(1))
            stage_title = stage_match.group(2).strip()
            try:
                duration_seconds = max(0.0, float(duration))
            except (TypeError, ValueError):
                duration_seconds = 0.0
            event_status = {
                "ok_with_fallback": "degraded",
                "ok_skipped_optional": "skipped",
            }.get(result.display_status, str(status).strip().lower())
            self.log.info(
                "[PIPELINE_STAGE_RESULT] "
                f"stage={stage_number} "
                f"status={event_status} "
                f"duration={duration_seconds:.1f} "
                f"title={stage_title}"
            )
            try:
                self._publish_stage_preview(
                    stage_number,
                    stage_title,
                    str(status).strip().lower(),
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

    def _stage_preview_candidates(self, stage: int) -> List[Path]:
        """Return accepted stage artifacts in strict preference order."""
        if stage == 11:
            if not bool(getattr(self, "ai_outputs_generated", False)):
                return []
            return [self.work_dir / "result_final_ai.fit"] if self.work_dir else []
        if not self.process_dir:
            return []
        stems = {
            1: ("stage1_prepared",),
            2: ("stage2_corrected",),
            3: ("stage3_bgremoved",),
            4: ("stage4_color", "stage4_colorbalanced", "stage4_psolved"),
            5: ("stage5_linear", "stage5_denoised"),
            6: ("stage6_starless", "stage7_starless"),
            7: ("stage7_stretched",),
            8: ("stage8_enhanced", "starless_enhanced"),
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
            write_raw_preview(image_data, preview_path)
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

        linear_path = self.work_dir / LINEAR_RESUME_INPUT_NAME
        if not linear_path.is_file():
            raise SirilError(
                f"未找到线性续跑输入文件，请检查工作目录: {linear_path}"
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
        message = f"loaded existing {LINEAR_RESUME_INPUT_NAME}"
        if not stage_saved:
            message += "；stage1 输出保存失败"

        elapsed = self.log.stage_end(stage_label)
        self._record_stage(stage_label, stage_status, elapsed, message)

    def _stage2_corrected_resume_candidates(self) -> List[Path]:
        candidates = [
            self.work_dir / STAGE2_CORRECTED_INPUT_NAME,
            self.work_dir / "process" / STAGE2_CORRECTED_INPUT_NAME,
        ]
        seen = set()
        unique: List[Path] = []
        for path in candidates:
            resolved = path.resolve() if path.exists() else path
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(path)
        return unique

    def _prepare_stage2_corrected_resume_input(self) -> None:
        self._record_skipped_stage(
            PipelineStage.PREPARATION.label,
            "skipped by stage2 corrected resume mode",
        )
        stage_label = PipelineStage.VIEW_CORRECTION.label
        self.log.stage_start(stage_label)

        source_path = next(
            (path for path in self._stage2_corrected_resume_candidates() if path.is_file()),
            None,
        )
        if source_path is None:
            searched = ", ".join(str(path) for path in self._stage2_corrected_resume_candidates())
            raise SirilError(
                f"未找到叠加后处理输入文件 {STAGE2_CORRECTED_INPUT_NAME}，已检查: {searched}"
            )

        temp_source: Optional[Path] = None
        source_for_copy = source_path
        process_dir_candidate = self.work_dir / "process"
        try:
            if process_dir_candidate in source_path.parents:
                fd, temp_name = tempfile.mkstemp(
                    prefix="seestar_stage2_corrected_",
                    suffix=".fit",
                    dir=str(self.work_dir),
                )
                os.close(fd)
                temp_source = Path(temp_name)
                shutil.copy2(source_path, temp_source)
                source_for_copy = temp_source

            self._prepare_process_dir()
            self.source_file = source_path
            self.linear_intermediate_path = None
            self._stage1_input_mode = "stage2_corrected_resume"
            self._stage1_registration_stats = None

            corrected_file = self.process_dir / STAGE2_CORRECTED_INPUT_NAME
            working_file = self.process_dir / "working.fit"
            shutil.copy2(source_for_copy, corrected_file)
            shutil.copy2(source_for_copy, working_file)
            self.log.info(f"叠加后处理输入: {source_path}")
            self.log.info("已复制 stage2_corrected 输入到处理目录")

            self.cmd_with_check("cd", f'"{self.process_dir}"')
            self.cmd_with_check("load", "stage2_corrected")

            stage_saved = self._save_stage_output("stage2_corrected")
            stage_status = "ok" if stage_saved else "degraded"
            message = f"loaded existing {STAGE2_CORRECTED_INPUT_NAME}; continue from stage3"
            if not stage_saved:
                message += "；stage2 输出保存失败"

            elapsed = self.log.stage_end(stage_label)
            self._record_stage(stage_label, stage_status, elapsed, message)
        finally:
            if temp_source is not None:
                try:
                    temp_source.unlink()
                except OSError as e:
                    self.log.debug(f"Unable to remove temp stage2 input {temp_source.name}: {e}")

    def _stage4_psolved_resume_candidates(self) -> List[Path]:
        candidates = [
            self.work_dir / "process" / STAGE4_PSOLVED_INPUT_NAME,
            self.work_dir / STAGE4_PSOLVED_INPUT_NAME,
        ]
        seen = set()
        unique: List[Path] = []
        for path in candidates:
            resolved = path.resolve() if path.exists() else path
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(path)
        return unique

    def _prepare_stage4_psolved_resume_input(self) -> None:
        source_path = next(
            (
                path
                for path in self._stage4_psolved_resume_candidates()
                if path.is_file() and path.stat().st_size > 0
            ),
            None,
        )
        if source_path is None:
            searched = ", ".join(
                str(path) for path in self._stage4_psolved_resume_candidates()
            )
            raise SirilError(
                f"未找到 SPCC 崩溃恢复输入 {STAGE4_PSOLVED_INPUT_NAME}，已检查: {searched}"
            )

        temp_source: Optional[Path] = None
        source_for_copy = source_path
        process_dir_candidate = self.work_dir / "process"
        try:
            if process_dir_candidate in source_path.parents:
                fd, temp_name = tempfile.mkstemp(
                    prefix="seestar_stage4_psolved_",
                    suffix=".fit",
                    dir=str(self.work_dir),
                )
                os.close(fd)
                temp_source = Path(temp_name)
                shutil.copy2(source_path, temp_source)
                source_for_copy = temp_source

            self._prepare_process_dir()
            psolved_file = self.process_dir / STAGE4_PSOLVED_INPUT_NAME
            shutil.copy2(source_for_copy, psolved_file)
            self.source_file = psolved_file
            self.linear_intermediate_path = None
            self._stage1_input_mode = INPUT_MODE_STAGE4_PSOLVED_RESUME
            self._stage1_registration_stats = None
            self.platesolve_ok = True

            self.log.info(
                f"SPCC 崩溃恢复输入: {source_path}; 从 Stage 4 校色检查点继续"
            )
            self.cmd_with_check("cd", f'"{self.process_dir}"')
            self.cmd_with_check("load", "stage4_psolved")
        finally:
            if temp_source is not None:
                try:
                    temp_source.unlink()
                except OSError as e:
                    self.log.debug(
                        "Unable to remove temp stage4 psolved input "
                        f"{temp_source.name}: {e}"
                    )

    def _apply_forced_runtime_switches(self):
        if self._force_denoise_enabled is not None:
            if self.cfg.denoise_enabled != self._force_denoise_enabled:
                self.log.info(
                    "[AUTO] Apply forced denoise_enabled="
                    f"{self._force_denoise_enabled} (SEESTAR_DENOISE_FORCE)"
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
            return

        source_hint = self.source_file if isinstance(self.source_file, Path) else None

        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
        except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
            self.log.warn(f"[AUTO] Failed to load image pixel data: {e}")
            self.log.warn("[AUTO] Fallback to default config")
            self.cfg = copy.deepcopy(self.base_cfg)
            self._apply_forced_runtime_switches()
            return

        try:
            features = measure_image_features(image_data)
            target_type = detect_target_type(source_hint, image_data=image_data)
            tuned_cfg, result = auto_tune_config(self.cfg, target_type, features)

            self.cfg = tuned_cfg
            self.auto_tune_result = result
            self._apply_forced_runtime_switches()

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

    def connect(self):
        self.log.info("正在连接 Siril...")
        try:
            self.siril.connect()
            self._siril_ever_connected = True
            self.work_dir = Path(self.siril.get_siril_wd())
            self.log.set_file_path(self.work_dir / "seestar_pipeline_python.log")
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
        run_stage2_view_correction(self)

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

    def target_profile_preflight(self):
        """Run target profiling as an internal Stage 3/4 preflight."""
        message = self._run_target_profile_preflight(
            source=PipelineCheckpoint.TARGET_PROFILE_PREFLIGHT.label,
            preview_name="stage3_target_preview.png",
        )
        self.log.info(
            "Target profiler 属于 Stage 3/4 内部 preflight，不写入独立阶段"
            + (f"；{message}" if message else "")
        )

    def stage2_5_target_profiler(self):
        """Deprecated compatibility alias for target_profile_preflight()."""
        self.log.warn(
            "stage2_5_target_profiler() 是历史兼容别名；"
            "Target profiler 已合并到 Stage 3/4 preflight"
        )
        return self.target_profile_preflight()

    # ========================================
    # 阶段 3: 背景提取
    # ========================================
    def stage3_background_extraction(self):
        run_stage3_background_extraction(self)

    # ========================================
    # 阶段 4: 色彩校准
    # ========================================
    def stage4_color_calibration(self):
        return run_stage4_color_calibration(self)

    def stage5_linear_denoise(self):
        return run_stage5_linear_denoise(self)

    def stage7_stretching(self):
        return run_stage6_stretching(self)

    def stage6_stretching(self):
        """Deprecated compatibility alias for stage7_stretching()."""
        return self.stage7_stretching()

    def pre_starless_compatibility_gate(self):
        """Run the legacy pre-starless checkpoint outside the formal stages."""
        stage_label = PipelineCheckpoint.PRE_STARLESS_COMPATIBILITY_GATE.label
        self.log.stage_start(stage_label)
        status = "ok"
        messages: List[str] = []
        report: Dict[str, Any]
        try:
            if evaluate_pre_starless_gate is None:
                raise RuntimeError(f"quality gate module unavailable: {ADAPTIVE_IMPORT_ERROR}")
            source_stem = self.stretched_name or "stage7_stretched"
            metrics = self._adaptive_features_by_stem(source_stem)
            quality_metrics = self._measure_current_quality()
            if quality_metrics is not None:
                metrics.update(
                    {
                        "black_pixel_ratio": float(quality_metrics.black_pixel_ratio),
                        "highlight_clip_ratio": float(quality_metrics.highlight_clip_ratio),
                        "median_star_size": float(quality_metrics.median_star_size),
                    }
                )
            current_features = self._measure_current_features()
            if current_features is not None:
                metrics["edge_black_ratio"] = max(
                    float(metrics.get("edge_black_ratio", 0.0) or 0.0),
                    float(current_features.edge_black_ratio),
                )
                metrics.setdefault("legacy_bg_median", float(current_features.bg_median))
                metrics.setdefault("legacy_bg_std", float(current_features.bg_std))
            report = evaluate_pre_starless_gate(
                metrics,
                getattr(self, "target_profile", {}) or {},
                getattr(self, "pipeline_policy", {}) or {},
            )
            recommended = str(report.get("recommended_starless_input") or source_stem)
            if recommended.endswith(".fit"):
                recommended = recommended[:-4]
            if recommended != source_stem:
                records = self._stage7_build_conservative_starless_inputs()
                report["conservative_inputs"] = records
                report["fallback_created"] = any(
                    item.get("stem") == recommended and item.get("status") == "ok"
                    for item in records
                )
                if self.process_dir and not (self.process_dir / f"{recommended}.fit").exists():
                    fallback = "stage7_conservative_asinh"
                    if (self.process_dir / f"{fallback}.fit").exists():
                        recommended = fallback
                    else:
                        recommended = source_stem
                    report["recommended_starless_input"] = recommended
                    report.setdefault("reason", []).append("recommended fallback missing; using available source")
            self.pre_starless_gate_report = report
            self._write_stage_json("pre_starless_gate_report.json", report)
            self.log.info(
                "[PreStarlessCompatibilityGate] ready_for_starless="
                f"{bool(report.get('ready_for_starless'))} recommended={report.get('recommended_starless_input')}"
            )
            messages.append(
                f"recommended_starless_input={report.get('recommended_starless_input')}"
            )
            if report.get("reason"):
                messages.append("; ".join(str(item) for item in report.get("reason", [])[:3]))
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            status = "degraded"
            reason = self._short_text(e, 180)
            report = {
                "stage": "stage6_5_pre_starless_gate",
                "ready_for_starless": True,
                "reason": [reason],
                "recommended_starless_input": self.stretched_name or "stage7_stretched",
                "fallback_created": False,
            }
            self.pre_starless_gate_report = report
            self._write_stage_json("pre_starless_gate_report.json", report)
            self.log.warn(
                "[PreStarlessCompatibilityGate] quality gate failed, "
                f"using stage7_stretched: {reason}"
            )
            messages.append(f"quality gate fallback: {reason}")

        elapsed = self.log.stage_end(stage_label)
        self._record_stage(
            stage_label,
            status,
            elapsed,
            "；".join(messages),
        )

    def stage6_5_pre_starless_gate(self):
        """Deprecated compatibility alias for pre_starless_compatibility_gate()."""
        self.log.warn(
            "stage6_5_pre_starless_gate() 是历史兼容别名；"
            "该门控不是正式 Stage 6.5"
        )
        return self.pre_starless_compatibility_gate()

    # ========================================
    # 阶段 6: 星点分离（starless-first，先于主体拉伸执行）
    # ========================================
    def stage6_star_separation(self):
        return run_stage7_star_separation(self)

    def stage7_star_separation(self):
        """Deprecated compatibility alias for stage6_star_separation()."""
        return self.stage6_star_separation()

    def stage8_nebula_enhancement(self):
        return run_stage8_nebula_enhancement(self)

    def stage9_star_remixing(self):
        return run_stage9_star_remixing(self)

    def stage10_export(self):
        run_stage10_export(self)

    # ========================================
    # Stage 11: Optional AI postprocess
    # ========================================
    def stage11_ai_postprocess(self):
        if run_stage11_ai_postprocess is None:
            stage_label = PipelineStage.AI_POSTPROCESS.label
            self.log.stage_start(stage_label)
            status = "degraded"
            message = f"stage11 module import failed: {STAGE11_IMPORT_ERROR}"
            self.log.warn(message)
            elapsed = self.log.stage_end(stage_label)
            self._record_stage(stage_label, status, elapsed, message)
            return

        run_stage11_ai_postprocess(
            self,
            write_png_rgb16_func=write_png_rgb16,
        )

    def ai_artistic_derivative_experiment(self):
        """Run the non-stage artistic branch without changing Stage 1-11 results."""
        self.ai_artistic_output_generated = False
        self.ai_artistic_output_path = None
        if not self.cfg.ai_artistic_derivative_enabled:
            return
        if run_ai_artistic_derivative is None:
            self.log.warn(
                "[AI-Artistic] isolated module unavailable; experiment skipped: "
                f"{AI_ARTISTIC_IMPORT_ERROR}"
            )
            return
        run_ai_artistic_derivative(
            self,
            write_png_rgb16_func=write_png_rgb16,
        )

    # ========================================
    # 清理
    # ========================================
    def _archive_diagnostics(self) -> Optional[Path]:
        """在清理中间文件前归档轻量诊断产物；归档失败不影响主任务。"""
        if not self.process_dir or not self.process_dir.exists():
            return None

        work_dir = self.work_dir or self.process_dir.parent
        archive_path = work_dir / "seestar_diagnostics.zip"
        temporary_path = archive_path.with_suffix(".zip.tmp")
        diagnostic_suffixes = {
            ".json", ".jsonl", ".log", ".txt", ".csv", ".png",
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
                            "schema": "seestar.diagnostics.v1",
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

    def cleanup(self):
        """清理处理目录中的中间文件"""
        self.log.stage_start("清理中间文件")

        if not self.process_dir or not self.process_dir.exists():
            elapsed = self.log.stage_end("清理")
            return

        self._archive_diagnostics()
        deleted_count = self._cleanup_lightsrc_intermediates()

        if self.cfg.debug_mode:
            if deleted_count:
                self.log.info(f"调试模式: 已清理 {deleted_count} 个 lightsrc 中间文件")
            self.log.info("调试模式: 保留 stage* 等中间文件")
            elapsed = self.log.stage_end("清理")
            return

        # 保留关键文件
        keep_files = {
            'starless.fit',
            'starmask.fit',
            'starmask_raw.fit',
            'starmask_clean.fit',
            'starmask_external_raw.fit',
            'starmask_stretched.fit',
        }
        if self.stretched_name:
            keep_files.add(f"{self.stretched_name}_starmask.fit")

        # 清理所有中间文件类型
        cleanup_patterns = [
            '*.fit', '*.fits', '*.seq', '*.log', '*.csv', '*.lst',
        ]
        for pattern in cleanup_patterns:
            for f in self.process_dir.glob(pattern):
                if f.name not in keep_files:
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
            self.log.info("# Seestar 后期处理流水线")
            self.log.info("# 适用于望远镜机内叠加好的图像")
            self.log.info("#" * 50)
            if self.siril_plugin_dir:
                self.log.info(f"插件目录: {self.siril_plugin_dir}")
            self.log.info(f"输入模式: {self.input_mode}")
            self.platesolve_ok = False
            self.ai_outputs_generated = False
            self.ai_artistic_output_generated = False
            self.ai_artistic_output_path = None
            self._stage8_final_source = "starless_enhanced"
            self._stage8_fallback_used = False
            self._stage8_final_quality = "unknown"
            self._stage8_conservative_mode = False
            self._stage7_selected_quality = None
            self._stage7_residual_star_score = 0.0
            self._stage7_starless_skipped = False
            self._stage7_before_stage6 = True
            self._stage7_starless_first_source = ""
            self._stage9_star_intensity_scale = 1.0
            self._stage9_star_intensity_reason = ""
            self._stage9_bypassed_bad_starless = False
            self._stage9_stars_required = True
            self._stage9_stars_applied = False
            self._stage9_stars_application_mode = "pending"
            self._stage9_final_source = ""
            self._star_preserve_target_bypass = False
            self._stage5_denoise_applied = False
            self._saturation_boost_applied = 0.0

            if self.cfg.ai_post_enabled:
                self.log.info(
                    "[AI] Enabled for stage6-8 parameter optimization and optional stage11: "
                    f"endpoint={self.cfg.ai_endpoint}, model={self.cfg.ai_model}, "
                    f"timeout={self.cfg.ai_timeout_sec}s, strength={self.cfg.ai_strength}"
                )
            if self.cfg.ai_artistic_derivative_enabled:
                self.log.info(
                    "[AI-Artistic] isolated experiment enabled "
                    f"model={self.cfg.ai_artistic_model or 'unset'}, "
                    f"timeout={self.cfg.ai_artistic_timeout_sec}s; "
                    "output will not re-enter Siril"
                )

            if self.input_mode == INPUT_MODE_LINEAR_RESUME:
                self._prepare_linear_resume_input()
                self._auto_tune_for_current_input()
                self._record_skipped_stage(
                    PipelineStage.VIEW_CORRECTION.label,
                    "skipped by linear resume mode",
                )
                self._run_target_profile_preflight(
                    source="Linear resume preflight",
                    metadata_candidates=(LINEAR_RESUME_INPUT_NAME, self.source_file),
                    preview_name="stage3_target_preview.png",
                )
                self._record_skipped_stage(
                    PipelineStage.BACKGROUND_EXTRACTION.label,
                    "skipped by linear resume mode",
                )
                self._record_skipped_stage(
                    PipelineStage.COLOR_CALIBRATION.label,
                    "skipped by linear resume mode",
                )
                self._record_skipped_stage(
                    PipelineStage.LINEAR_DENOISE.label,
                    "skipped by linear resume mode",
                )
            elif self.input_mode == INPUT_MODE_STAGE4_PSOLVED_RESUME:
                self._record_skipped_stage(
                    PipelineStage.PREPARATION.label,
                    "skipped by stage4 psolved crash-resume mode",
                )
                self._record_skipped_stage(
                    PipelineStage.VIEW_CORRECTION.label,
                    "skipped by stage4 psolved crash-resume mode",
                )
                self._record_skipped_stage(
                    PipelineStage.BACKGROUND_EXTRACTION.label,
                    "skipped by stage4 psolved crash-resume mode",
                )
                self._prepare_stage4_psolved_resume_input()
                self._auto_tune_for_current_input()
                self.stage4_color_calibration()
                self.stage5_linear_denoise()
            elif self.input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
                self._prepare_stage2_corrected_resume_input()
                self._auto_tune_for_current_input()
                self.stage3_background_extraction()
                self.stage4_color_calibration()
                self.stage5_linear_denoise()
            else:
                # 线性阶段 (1-5)
                self.stage1_preparation()
                self._auto_tune_for_current_input()
                self.stage2_view_correction()
                self.stage3_background_extraction()
                self.stage4_color_calibration()
                self.stage5_linear_denoise()
            self.stage6_star_separation()
            self.stage7_stretching()
            self._record_skipped_stage(
                PipelineCheckpoint.PRE_STARLESS_COMPATIBILITY_GATE.label,
                "not a formal stage; starless-first mode completes Stage 6 "
                "before Stage 7",
            )
            self.stage8_nebula_enhancement()
            self.stage9_star_remixing()
            self.stage10_export()
            self.stage11_ai_postprocess()
            if bool(getattr(self.cfg, "ai_artistic_derivative_enabled", False)):
                self.ai_artistic_derivative_experiment()

            self.cleanup()

            duration = (time.time() - start_time) / 60

            # 结果汇总
            self.log.info("")
            self.log.info("=" * 60)
            self.log.info("处理结果汇总")
            self.log.info("=" * 60)
            status_icons = {
                'ok': '✓', 'degraded': '⚠', 'failed': '✗', 'skipped': '—',
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
                if r.status == 'degraded' or r.display_status == 'ok_with_fallback'
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
            base_name = (self.main_output_basename_template or RESULT_BASENAME_TEMPLATE)
            fit_base_name = getattr(
                self,
                "main_output_fit_basename_template",
                base_name + "_final",
            )
            self.log.info(f"  - {base_name}.tif  (16-bit Astro-TIFF)")
            self.log.info(f"  - {base_name}.png  (Preview PNG)")
            self.log.info(f"  - {fit_base_name}.fit (FITS archive)")
            fallback_line = "result_processed.tif / result_processed.png / result_final.fit"
            if bool(getattr(self, "_final_output_review_only", False)):
                fallback_line = (
                    f"{base_name}.tif / {base_name}.png / {fit_base_name}.fit "
                    "(review-only)"
                )
            elif self._stage1_input_mode == "linear_resume":
                fallback_line = (
                    "result_processed_linear.tif / result_processed_linear.png / "
                    "result_final_linear.fit"
                )
            self.log.info(f"  - fallback: {fallback_line}")
            if self.linear_intermediate_path:
                self.log.info("  - result_linear.fit (拉伸前线性中间文件)")
            if self.ai_outputs_generated:
                self.log.info("  - result_processed_ai.tif (16-bit Astro-TIFF, AI)")
                self.log.info("  - result_processed_ai.png (Preview PNG, AI)")
                self.log.info("  - result_final_ai.fit (AI 后期 FITS 存档)")
            if self.ai_artistic_output_generated and self.ai_artistic_output_path:
                self.log.info(
                    "  - "
                    f"{self.ai_artistic_output_path} "
                    "(AI artistic derivative; non-scientific isolated output)"
                )

        except SirilNativeProcessTerminated as e:
            self.log.error(
                "Siril 原生进程或连接已终止；当前流水线立即中止，"
                "不会继续降级、保存或执行后续阶段。"
            )
            raise
        except KeyboardInterrupt:
            self.log.warn("用户中断操作")
            raise
        except Exception as e:
            self.log.error(f"程序中断: {e}")
            self.log.error(traceback.format_exc())
            raise
        finally:
            if not self._siril_process_terminated and self._siril_ever_connected:
                try:
                    self.siril.disconnect()
                except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                    self.log.warn(f"Siril 断开连接失败: {e}")


if __name__ == "__main__":
    SeestarPostProcessor().run()
