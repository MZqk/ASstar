"""
SeeStar Post-processing Pipeline for Pre-stacked Images
Optimized for Siril 1.4+
处理由 Seestar 望远镜机内叠加好的单帧图像

详细流程说明见同目录文档: seestar_Superimpose_workflow.md

处理顺序遵循天文后期最佳实践:
  线性阶段 (拉伸前): 背景提取 → 色彩校准 → 降噪
  非线性阶段 (拉伸后): 星点分离 → 星云增强 → 星点混合 → 导出
"""
import json
import importlib
import hashlib
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
    PipelineConfig,
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

try:
    from stage11_ai_postprocess import run_stage11_ai_postprocess
    STAGE11_IMPORT_ERROR = None
except Exception as stage11_import_exc:
    run_stage11_ai_postprocess = None
    STAGE11_IMPORT_ERROR = stage11_import_exc

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
except Exception:
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
except Exception as adaptive_import_exc:
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
LINEAR_RESUME_INPUT_NAME = "result_linear.fit"
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


# ============================================================
# 配置与基础设施
# ============================================================

AUTO_CLAMP_FIELDS = (
    "crop_margin",
    "stage2_edge_black_target",
    "stage2_adaptive_edge_crop_max_passes",
    "stage2_adaptive_edge_crop_max_extra",
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
    "ai_bg_median_delta_max",
    "ai_color_ratio_delta_max",
    "ai_core_growth_ratio_max",
    "ai_star_growth_ratio_max",
    "stage6_bg_median_min",
    "stage6_black_pixel_ratio_max",
    "stage6_highlight_clip_ratio_max",
    "stage6_star_growth_ratio_max",
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
    "stage7_starmask_background_floor_percentile",
    "stage7_starmask_halo_blur_strength",
    "stage7_starmask_small_star_scale",
    "stage7_starmask_nebula_suppression",
    "mild_prestretch_strength",
    "stage7_conservative_asinh_stretch",
    "stage7_ultra_conservative_asinh_stretch",
    "stage7_soft_starless_asinh_stretch",
    "stage7_conservative_asinh_offset",
    "stage7_starless_repair_strength",
    "stage7_starless_halo_repair_strength",
    "stage7_starless_chroma_denoise_strength",
    "stage7_starless_repair_max_score_growth",
    "stage8_mask_signal_coverage_min",
    "stage8_blue_excess_max",
    "stage8_saturation_growth_ratio_max",
    "stage8_microcontrast_growth_ratio_max",
    "stage8_highlight_clip_ratio_max",
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
    except Exception:
        return TargetType.UNKNOWN


def clamp_config(cfg: PipelineConfig) -> PipelineConfig:
    """自动调参的统一安全限幅，防止参数进入危险区间。"""
    tuned = copy.deepcopy(cfg)
    tuned.crop_margin = _clamp_float(tuned.crop_margin, 0.0, 0.06)
    tuned.stage2_edge_black_target = _clamp_float(
        tuned.stage2_edge_black_target, 0.03, 0.18
    )
    tuned.stage2_adaptive_edge_crop_max_passes = _clamp_int(
        tuned.stage2_adaptive_edge_crop_max_passes, 0, 6
    )
    tuned.stage2_adaptive_edge_crop_max_extra = _clamp_float(
        tuned.stage2_adaptive_edge_crop_max_extra, 0.005, 0.060
    )
    tuned.bg_samples = _clamp_int(tuned.bg_samples, 12, 32)
    tuned.bg_tolerance = _clamp_float(tuned.bg_tolerance, 0.6, 1.8)
    tuned.bg_smooth = _clamp_float(tuned.bg_smooth, 0.2, 1.2)
    tuned.bg_std_worsen_ratio_max = _clamp_float(
        tuned.bg_std_worsen_ratio_max, 1.0, 1.4
    )
    tuned.bg_median_drop_ratio_min = _clamp_float(
        tuned.bg_median_drop_ratio_min, 0.05, 0.60
    )
    tuned.bg_object_preserve_ratio_min = _clamp_float(
        tuned.bg_object_preserve_ratio_min, 0.20, 0.90
    )
    tuned.bg_edge_black_rise_max = _clamp_float(
        tuned.bg_edge_black_rise_max, 0.10, 0.60
    )
    tuned.bg_star_preserve_ratio_min = _clamp_float(
        tuned.bg_star_preserve_ratio_min, 0.50, 1.00
    )
    tuned.bg_nebula_mean_change_max = _clamp_float(
        tuned.bg_nebula_mean_change_max, 0.02, 0.35
    )

    denoise_upper = max(0.2, min(0.55, float(tuned.denoise_safety_max)))
    tuned.denoise_mod = _clamp_float(tuned.denoise_mod, 0.2, denoise_upper)
    tuned.stage5_builtin_denoise_mod = _clamp_float(
        tuned.stage5_builtin_denoise_mod, 0.20, 0.55
    )
    tuned.stage5_rl_maxstars = _clamp_int(tuned.stage5_rl_maxstars, 20, 1000)
    tuned.stage5_rl_psf_kernel_size = _clamp_int(
        tuned.stage5_rl_psf_kernel_size, 9, 99
    )
    if tuned.stage5_rl_psf_kernel_size % 2 == 0:
        tuned.stage5_rl_psf_kernel_size += 1
    tuned.stage5_rl_iters = _clamp_int(tuned.stage5_rl_iters, 1, 40)
    tuned.stage5_rl_alpha = _clamp_float(tuned.stage5_rl_alpha, 100.0, 10000.0)
    tuned.stage5_rl_gdstep = _clamp_float(tuned.stage5_rl_gdstep, 0.00001, 0.01)
    tuned.stage5_rl_stop = _clamp_float(tuned.stage5_rl_stop, 0.0001, 0.05)
    tuned.stage5_graxpert_deconv_strength = _clamp_float(
        tuned.stage5_graxpert_deconv_strength, 0.20, 0.40
    )

    tuned.asinh_stretch = _clamp_float(tuned.asinh_stretch, 1.6, 3.6)
    tuned.asinh_offset = _clamp_float(tuned.asinh_offset, 0.0005, 0.006)
    tuned.ghs_shadowsclip = _clamp_float(tuned.ghs_shadowsclip, -3.6, -1.8)
    tuned.ghs_stretchamount = _clamp_float(tuned.ghs_stretchamount, 1.0, 2.8)
    tuned.nebula_saturation = _clamp_float(tuned.nebula_saturation, 0.0, 0.65)
    tuned.stage8_core_protection_strength = _clamp_float(
        tuned.stage8_core_protection_strength, 0.50, 1.00
    )
    tuned.stage8_background_denoise_strength = _clamp_float(
        tuned.stage8_background_denoise_strength, 0.0, 0.25
    )
    tuned.stage8_faint_nebula_boost_max = _clamp_float(
        tuned.stage8_faint_nebula_boost_max, 0.0, 0.18
    )
    tuned.stage8_nebula_contrast_max = _clamp_float(
        tuned.stage8_nebula_contrast_max, 0.0, 0.20
    )
    tuned.stage8_masked_unsharp_amount_max = _clamp_float(
        tuned.stage8_masked_unsharp_amount_max, 0.0, 0.25
    )
    tuned.stage8_blue_precontrol_strength = _clamp_float(
        tuned.stage8_blue_precontrol_strength, 0.0, 1.00
    )
    tuned.stage8_bg_std_growth_max = _clamp_float(
        tuned.stage8_bg_std_growth_max, 1.00, 1.50
    )
    tuned.stage8_texture_artifact_growth_max = _clamp_float(
        tuned.stage8_texture_artifact_growth_max, 1.00, 2.20
    )
    tuned.star_intensity = _clamp_float(tuned.star_intensity, 0.8, 1.05)
    tuned.star_fallback_intensity = _clamp_float(
        tuned.star_fallback_intensity, 0.75, 1.05
    )
    tuned.final_saturation = _clamp_float(tuned.final_saturation, 0.05, 0.25)
    tuned.ai_timeout_sec = _clamp_int(tuned.ai_timeout_sec, 15, 300)
    tuned.ai_strength = _clamp_float(tuned.ai_strength, 0.05, 0.25)
    tuned.ai_bg_median_delta_max = _clamp_float(
        tuned.ai_bg_median_delta_max, 0.01, 0.06
    )
    tuned.ai_color_ratio_delta_max = _clamp_float(
        tuned.ai_color_ratio_delta_max, 0.08, 0.35
    )
    tuned.ai_core_growth_ratio_max = _clamp_float(
        tuned.ai_core_growth_ratio_max, 1.05, 1.80
    )
    tuned.ai_star_growth_ratio_max = _clamp_float(
        tuned.ai_star_growth_ratio_max, 1.05, 1.80
    )
    tuned.stage6_bg_median_min = _clamp_float(tuned.stage6_bg_median_min, 0.005, 0.080)
    tuned.stage6_black_pixel_ratio_max = _clamp_float(
        tuned.stage6_black_pixel_ratio_max, 0.10, 0.70
    )
    tuned.stage6_highlight_clip_ratio_max = _clamp_float(
        tuned.stage6_highlight_clip_ratio_max, 0.001, 0.050
    )
    tuned.stage6_star_growth_ratio_max = _clamp_float(
        tuned.stage6_star_growth_ratio_max, 1.05, 1.80
    )
    tuned.stage7_quality_retry_max = _clamp_int(tuned.stage7_quality_retry_max, 0, 3)
    tuned.stage7_edge_black_warn = _clamp_float(
        tuned.stage7_edge_black_warn, 0.04, 0.30
    )
    tuned.stage7_edge_black_high = _clamp_float(
        tuned.stage7_edge_black_high,
        tuned.stage7_edge_black_warn,
        0.60,
    )
    tuned.stage7_bg_median_high = _clamp_float(
        tuned.stage7_bg_median_high, 0.08, 0.35
    )
    tuned.stage7_bg_std_high = _clamp_float(
        tuned.stage7_bg_std_high, 0.020, 0.120
    )
    tuned.stage7_bg_noise_ratio_high = _clamp_float(
        tuned.stage7_bg_noise_ratio_high, 0.20, 1.50
    )
    tuned.stage7_residual_star_score_max = _clamp_float(
        tuned.stage7_residual_star_score_max, 0.10, 1.20
    )
    tuned.stage7_halo_residue_score_max = _clamp_float(
        tuned.stage7_halo_residue_score_max, 0.05, 1.00
    )
    tuned.stage7_bright_nebula_halo_residue_score_max = _clamp_float(
        tuned.stage7_bright_nebula_halo_residue_score_max,
        tuned.stage7_halo_residue_score_max,
        1.20,
    )
    tuned.stage7_black_hole_score_max = _clamp_float(
        tuned.stage7_black_hole_score_max, 0.01, 0.35
    )
    tuned.stage7_starmask_contamination_max = _clamp_float(
        tuned.stage7_starmask_contamination_max, 0.05, 0.80
    )
    tuned.stage7_starless_noise_gain_max = _clamp_float(
        tuned.stage7_starless_noise_gain_max, 1.00, 2.50
    )
    tuned.stage7_starmask_coverage_min_ratio = _clamp_float(
        tuned.stage7_starmask_coverage_min_ratio, 0.05, 0.90
    )
    tuned.stage7_starmask_width_ratio_max = _clamp_float(
        tuned.stage7_starmask_width_ratio_max, 1.10, 3.00
    )
    tuned.stage7_starmask_background_floor_percentile = _clamp_float(
        tuned.stage7_starmask_background_floor_percentile, 20.0, 80.0
    )
    tuned.stage7_starmask_halo_blur_strength = _clamp_float(
        tuned.stage7_starmask_halo_blur_strength, 0.0, 0.80
    )
    tuned.stage7_starmask_small_star_scale = _clamp_float(
        tuned.stage7_starmask_small_star_scale, 0.50, 1.00
    )
    tuned.stage7_starmask_nebula_suppression = _clamp_float(
        tuned.stage7_starmask_nebula_suppression, 0.0, 0.95
    )
    tuned.mild_prestretch_strength = _clamp_float(
        tuned.mild_prestretch_strength, 1.05, 1.80
    )
    tuned.stage7_conservative_asinh_stretch = _clamp_float(
        tuned.stage7_conservative_asinh_stretch, 1.60, 2.60
    )
    tuned.stage7_ultra_conservative_asinh_stretch = _clamp_float(
        tuned.stage7_ultra_conservative_asinh_stretch, 1.20, tuned.stage7_conservative_asinh_stretch
    )
    tuned.stage7_soft_starless_asinh_stretch = _clamp_float(
        tuned.stage7_soft_starless_asinh_stretch, 1.05, tuned.stage7_ultra_conservative_asinh_stretch
    )
    tuned.stage7_conservative_asinh_offset = _clamp_float(
        tuned.stage7_conservative_asinh_offset, 0.0005, 0.0060
    )
    tuned.stage7_starless_repair_strength = _clamp_float(
        tuned.stage7_starless_repair_strength, 0.0, 0.85
    )
    tuned.stage7_starless_halo_repair_strength = _clamp_float(
        tuned.stage7_starless_halo_repair_strength, 0.0, 0.90
    )
    tuned.stage7_starless_chroma_denoise_strength = _clamp_float(
        tuned.stage7_starless_chroma_denoise_strength, 0.0, 0.90
    )
    tuned.stage7_starless_repair_max_score_growth = _clamp_float(
        tuned.stage7_starless_repair_max_score_growth, 0.0, 0.20
    )
    tuned.stage8_mask_signal_coverage_min = _clamp_float(
        tuned.stage8_mask_signal_coverage_min, 0.001, 0.050
    )
    tuned.stage8_blue_excess_max = _clamp_float(tuned.stage8_blue_excess_max, 0.02, 0.30)
    tuned.stage8_saturation_growth_ratio_max = _clamp_float(
        tuned.stage8_saturation_growth_ratio_max, 1.05, 2.50
    )
    tuned.stage8_microcontrast_growth_ratio_max = _clamp_float(
        tuned.stage8_microcontrast_growth_ratio_max, 1.05, 2.80
    )
    tuned.stage8_highlight_clip_ratio_max = _clamp_float(
        tuned.stage8_highlight_clip_ratio_max, 0.001, 0.060
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
        "nebula_sat={:.2f}, star_mode={}, mild_prestretch={:.2f}, "
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
        cfg.mild_prestretch_strength,
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
        self.siril = s.SirilInterface()
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
        self._stage9_final_source: str = ""
        self.target_profile: Dict[str, Any] = {}
        self.pipeline_policy: Dict[str, Any] = copy.deepcopy(DEFAULT_POLICY)
        self.pre_starless_gate_report: Dict[str, Any] = {}
        self.color_calibration_report: Dict[str, Any] = {}
        self.input_mode: str = INPUT_MODE_AUTO
        self._stage1_input_mode: str = "unknown"
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

    def cmd_with_check(self, *args, quiet=False):
        """执行 Siril 命令，带基于状态码与错误文本的智能重试"""
        cmd_str = ' '.join(map(str, args))
        cmd_name = str(args[0]).lower() if args else ''
        max_attempts = self.cfg.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            try:
                self.siril.cmd(*args)
                self.log.debug(f"  ✓ {cmd_str}")
                return True
            except CommandError as e:
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
                if quiet:
                    self.log.debug(f"失败: {cmd_str} ({e})")
                else:
                    self.log.error(f"失败: {cmd_str}\n    错误: {e}")
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
        self.results.append(StageResult(name, status, duration, message))

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
        self.log.stage_start("阶段 1: 前期准备")
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

        elapsed = self.log.stage_end("阶段 1: 前期准备")
        self._record_stage("阶段 1: 前期准备", stage_status, elapsed, message)

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
        except Exception as e:
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
                    self.log.warn(f"[AUTO] {note}")

            self.log.info(f"[AUTO] Final tuned config: {format_config_summary(self.cfg)}")

            if self.cfg.auto_tune_debug or self.cfg.debug_mode:
                self.log.debug(
                    f"[AUTO] Source hint: {source_hint if source_hint else 'N/A'}"
                )
        except Exception as e:
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
            self.work_dir = Path(self.siril.get_siril_wd())
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
        except Exception as e:
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

    def stage2_5_target_profiler(self):
        """Compatibility wrapper: target profiling is now a Stage3/4 preflight."""
        message = self._run_target_profile_preflight(
            source="Stage2.5 compatibility preflight",
            preview_name="stage3_target_preview.png",
        )
        self.log.info(
            "阶段 2.5 已合并到 Stage3/4 preflight，不再写入独立阶段"
            + (f"；{message}" if message else "")
        )

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

    def stage6_stretching(self):
        return run_stage6_stretching(self)

    def stage6_5_pre_starless_gate(self):
        """Compatibility gate: choose the safest input for star separation."""
        self.log.stage_start("阶段 6.5: 去星前质量门控")
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
                "[Stage6.5] ready_for_starless="
                f"{bool(report.get('ready_for_starless'))} recommended={report.get('recommended_starless_input')}"
            )
            messages.append(
                f"recommended_starless_input={report.get('recommended_starless_input')}"
            )
            if report.get("reason"):
                messages.append("; ".join(str(item) for item in report.get("reason", [])[:3]))
        except Exception as e:
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
            self.log.warn(f"[Stage6.5] quality gate failed, using stage7_stretched: {reason}")
            messages.append(f"quality gate fallback: {reason}")

        elapsed = self.log.stage_end("阶段 6.5: 去星前质量门控")
        self._record_stage(
            "阶段 6.5: 去星前质量门控",
            status,
            elapsed,
            "；".join(messages),
        )

    # ========================================
    # 阶段 6: 星点分离（starless-first，先于主体拉伸执行）
    # ========================================
    def stage7_star_separation(self):
        return run_stage7_star_separation(self)

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
            self.log.stage_start("阶段 11: AI 后期美化")
            status = "degraded"
            message = f"stage11 module import failed: {STAGE11_IMPORT_ERROR}"
            self.log.warn(message)
            elapsed = self.log.stage_end("阶段 11: AI 后期美化")
            self._record_stage("阶段 11: AI 后期美化", status, elapsed, message)
            return

        run_stage11_ai_postprocess(
            self,
            write_png_rgb16_func=write_png_rgb16,
        )

    # ========================================
    # 清理
    # ========================================
    def cleanup(self):
        """清理处理目录中的中间文件"""
        self.log.stage_start("清理中间文件")

        if not self.process_dir or not self.process_dir.exists():
            elapsed = self.log.stage_end("清理")
            return

        deleted_count = self._cleanup_lightsrc_intermediates()

        if self.cfg.debug_mode:
            if deleted_count:
                self.log.info(f"调试模式: 已清理 {deleted_count} 个 lightsrc 中间文件")
            self.log.info("调试模式: 保留 stage* 等中间文件")
            elapsed = self.log.stage_end("清理")
            return

        # 保留关键文件
        keep_files = {'starless.fit', 'starmask.fit'}
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
            self._apply_runtime_env_overrides()
            self.base_cfg = copy.deepcopy(self.cfg)

            self.connect()

            try:
                self.siril.cmd("requires", "1.4.0")
            except (CommandError, SirilError) as e:
                self.log.error(f"版本检查失败: {e}")
                self.log.error("此脚本需要 Siril 1.4.0 或更高版本")
                return

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
            self._stage9_final_source = ""

            if self.cfg.ai_post_enabled:
                self.log.info(
                    "[AI] Enabled for stage6-8 parameter optimization and optional stage11: "
                    f"endpoint={self.cfg.ai_endpoint}, model={self.cfg.ai_model}, "
                    f"timeout={self.cfg.ai_timeout_sec}s, strength={self.cfg.ai_strength}"
                )

            if self.input_mode == INPUT_MODE_LINEAR_RESUME:
                self._prepare_linear_resume_input()
                self._auto_tune_for_current_input()
                self._record_skipped_stage(
                    "阶段 2: 裁切",
                    "skipped by linear resume mode",
                )
                self._run_target_profile_preflight(
                    source="Linear resume preflight",
                    metadata_candidates=(LINEAR_RESUME_INPUT_NAME, self.source_file),
                    preview_name="stage3_target_preview.png",
                )
                self._record_skipped_stage(
                    "阶段 3: 背景提取",
                    "skipped by linear resume mode",
                )
                self._record_skipped_stage(
                    "阶段 4: 图像解析 + 色彩校准",
                    "skipped by linear resume mode",
                )
                self._record_skipped_stage(
                    "阶段 5: 线性反卷积 / 轻降噪",
                    "skipped by linear resume mode",
                )
            else:
                # 线性阶段 (1-5)
                self.stage1_preparation()
                self._auto_tune_for_current_input()
                self.stage2_view_correction()
                self.stage3_background_extraction()
                self.stage4_color_calibration()
                self.stage5_linear_denoise()
            self.stage7_star_separation()
            self.stage6_stretching()
            self._record_skipped_stage(
                "阶段 7.5: 去星前质量门控（兼容跳过）",
                "starless-first mode: Stage6 already separated stars before Stage7 stretch",
            )
            self.stage8_nebula_enhancement()
            self.stage9_star_remixing()
            self.stage10_export()
            self.stage11_ai_postprocess()

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
            degraded = [r for r in self.results if r.status == 'degraded']
            if failed:
                self.log.warn(f"{len(failed)} 个阶段失败")
            if degraded:
                self.log.warn(f"{len(degraded)} 个阶段降级完成")
            if not failed and not degraded:
                self.log.info("所有阶段成功完成")

            self.log.info(f"输出目录: {self.work_dir}")
            self.log.info("生成文件:")
            base_name = (self.main_output_basename_template or RESULT_BASENAME_TEMPLATE)
            self.log.info(f"  - {base_name}.tif  (16-bit Astro-TIFF)")
            self.log.info(f"  - {base_name}.png  (Preview PNG)")
            self.log.info(f"  - {base_name}_final.fit (FITS archive)")
            fallback_line = "result_processed.tif / result_processed.png / result_final.fit"
            if self._stage1_input_mode == "linear_resume":
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

        except KeyboardInterrupt:
            self.log.warn("用户中断操作")
        except Exception as e:
            self.log.error(f"程序中断: {e}")
            traceback.print_exc()
        finally:
            try:
                self.siril.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    SeestarPostProcessor().run()
