"""Shared Stage 1-10 processing-parameter contract.

The GUI, signed task manifest, resume fingerprints and runtime all use this
module so that defaults, ranges and parameter ownership cannot silently drift.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

try:
    from .models import PipelineConfig
except ImportError:  # Direct execution from the pipeline directory.
    from models import PipelineConfig  # type: ignore[no-redef]


PROCESSING_PARAMETERS_SCHEMA = "starun.processing-parameters.v5"
LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4 = "starun.processing-parameters.v4"
SUPPORTED_PROCESSING_PARAMETERS_SCHEMAS = frozenset(
    {PROCESSING_PARAMETERS_SCHEMA, LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4}
)
GATE_PROFILE_DEFAULT = "default"
GATE_PROFILE_RELAXED = "relaxed"
GATE_PROFILE_UNLIMITED = "unlimited"
GATE_PROFILE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("默认档位", GATE_PROFILE_DEFAULT),
    ("放松模式", GATE_PROFILE_RELAXED),
    ("无限模式", GATE_PROFILE_UNLIMITED),
)
GATE_PROFILE_MULTIPLIERS = {
    GATE_PROFILE_DEFAULT: 1.0,
    GATE_PROFILE_RELAXED: 3.0,
    GATE_PROFILE_UNLIMITED: 10.0,
}
GATE_PROFILE_LABELS = {value: label for label, value in GATE_PROFILE_CHOICES}

PROFILE_SCALE_NONE = "none"
PROFILE_SCALE_UPPER = "upper"
PROFILE_SCALE_LOWER = "lower"
PROFILE_SCALE_UPPER_FROM_ONE = "upper_from_one"
GENERAL_DEFAULTS: Dict[str, Any] = {
    "output_formats": ["tif", "png", "fit"],
    "review_only": False,
    "compute_mode": "auto",
    "auto_tune_enabled": True,
    "max_retries": 2,
    "retry_delay": 1.0,
    "review_bundle_enabled": True,
    "managed_output_enabled": True,
    "checkpoint_mode": False,
}
GENERAL_KEYS = frozenset(GENERAL_DEFAULTS)
STAGE_TITLES = {
    1: "配准与叠加",
    2: "自动裁剪",
    3: "背景提取",
    4: "色彩校准",
    5: "降噪与反卷积",
    6: "恒星分离",
    7: "主体拉伸",
    8: "星云增强",
    9: "恒星重组",
    10: "最终处理与导出",
}
FAILURE_ACTION_STAGES = tuple(range(2, 10))


@dataclass(frozen=True)
class ParameterSpec:
    field: str
    stage: int
    label: str
    level: str
    kind: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    decimals: int = 2
    choices: Tuple[Tuple[str, Any], ...] = ()
    suffix: str = ""
    help: str = ""
    stage_mode: bool = False
    odd: bool = False
    section: str = "algorithm"
    depends_on: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()
    strictness: str = "neutral"
    profile_scaling: str = PROFILE_SCALE_NONE
    profile_minimum: float | int | None = None
    profile_maximum: float | int | None = None

    @property
    def default(self) -> Any:
        return getattr(PipelineConfig(), self.field)


def _choice(*items: Tuple[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(items)


def _gate(
    field: str,
    stage: int,
    label: str,
    kind: str,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    step: float | int | None = None,
    decimals: int = 2,
    *,
    suffix: str = "",
    help: str = "",
    section: str = "quality_gate",
    strictness: str = "",
    depends_on: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (),
) -> ParameterSpec:
    """Build a task-scoped expert quality/process-gate parameter."""

    return ParameterSpec(
        field,
        stage,
        f"门禁 · {label}",
        "expert",
        kind,
        minimum,
        maximum,
        step,
        decimals,
        suffix=suffix,
        help=(
            help
            or "高级过程门禁；收紧可能触发回退或复核，放宽可能接受更多候选。"
        ),
        section=section,
        depends_on=depends_on,
        strictness=(
            strictness
            or (
                "higher_is_stricter"
                if field.endswith(("_min", "_enabled"))
                else "lower_is_stricter"
                if field.endswith(("_max", "_ratio", "_growth"))
                else "contextual"
            )
        ),
    )


def _spec(
    field: str,
    stage: int,
    label: str,
    kind: str,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    step: float | int | None = None,
    decimals: int = 2,
    *,
    level: str = "expert",
    choices: Tuple[Tuple[str, Any], ...] = (),
    suffix: str = "",
    help: str = "",
    stage_mode: bool = False,
    odd: bool = False,
    section: str = "algorithm",
    depends_on: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (),
    strictness: str = "neutral",
) -> ParameterSpec:
    """Build a parameter in the current processing-parameters contract."""

    return ParameterSpec(
        field,
        stage,
        label,
        level,
        kind,
        minimum,
        maximum,
        step,
        decimals,
        choices=choices,
        suffix=suffix,
        help=help,
        stage_mode=stage_mode,
        odd=odd,
        section=section,
        depends_on=depends_on,
        strictness=strictness,
    )


def _failure_action_spec(stage: int) -> ParameterSpec:
    return _spec(
        f"stage{stage}_failure_action",
        stage,
        "失败处置",
        "choice",
        choices=_choice(
            ("自动安全回退", "auto_fallback"),
            ("恢复输入并复核", "preserve_review"),
            ("写诊断后终止任务", "stop"),
        ),
        section="fallback",
        strictness="stop_is_stricter",
        help=(
            "决定性失败时的处置；任何模式都不允许忽略数值、裁切或结构技术门。"
        ),
    )


PROCESSING_PARAMETER_SPECS: Tuple[ParameterSpec, ...] = (
    _spec(
        "stage1_register_fail_ratio_max",
        1,
        "配准失败率降级线",
        "float",
        0.0,
        0.50,
        0.01,
        2,
        section="quality_gate",
        strictness="lower_is_stricter",
        help="仅控制现有配准失败率的 degraded 判定，不跳过输入校验。",
    ),
    ParameterSpec(
        "stage2_processing_mode", 2, "处理方式", "recommended", "choice",
        choices=_choice(("自动裁剪", "auto"), ("保留原图", "preserve")),
        stage_mode=True,
        help="自动裁切叠加黑边；保留原图仍会生成 Stage 2 规范产物。",
    ),
    ParameterSpec(
        "stage2_center_protect_area_ratio", 2, "中心保护区域", "expert", "float",
        0.50, 0.95, 0.01, 2, suffix="×",
        help="限制自动裁切最多可侵入的中心区域，数值越高越保守。",
    ),
    ParameterSpec(
        "stage3_processing_mode", 3, "处理方式", "recommended", "choice",
        choices=_choice(("自动提取", "auto"), ("保留背景", "preserve")),
        stage_mode=True,
        help="自动评估并提取梯度；保留背景会安全直通并保留诊断。",
    ),
    ParameterSpec("bg_samples", 3, "背景采样数", "expert", "int", 12, 32, 1, help="背景模型采样密度。"),
    ParameterSpec("bg_tolerance", 3, "背景容差", "expert", "float", 0.60, 1.80, 0.05, 2, help="背景样点容差。"),
    ParameterSpec("bg_smooth", 3, "背景平滑度", "expert", "float", 0.20, 1.20, 0.05, 2, help="背景模型平滑度，过高可能削弱大尺度弱信号。"),
    ParameterSpec(
        "stage4_processing_mode", 4, "处理方式", "recommended", "choice",
        choices=_choice(("自动校准", "auto"), ("保留颜色", "preserve")),
        stage_mode=True,
        help="自动执行解析与物理校色；保留颜色会安全直通。",
    ),
    ParameterSpec(
        "stage4_offline_fallback_mode", 4, "无 Gaia 时", "recommended", "choice",
        choices=_choice(
            ("自动局部参考（非物理色彩，需复核）", "auto_local_reference"),
            ("保留输入颜色（需复核）", "preserve"),
        ),
        help="Gaia 定位星表不可用时允许任务继续；两种路线均明确标记需复核。",
    ),
    ParameterSpec(
        "stage4_filter_hint", 4, "拍摄滤镜", "recommended", "choice",
        choices=_choice(
            ("无滤镜", "no_filter"),
            ("Seestar LP", "seestar_lp"),
            ("双窄带 Ha/OIII", "dual_narrowband"),
        ),
        help="FITS FILTER 缺失时补充通道语义；自动状态下由元数据识别。",
    ),
    ParameterSpec("stage4_platesolve_enabled", 4, "Plate Solve", "expert", "bool", help="是否在校色前执行图像解析。"),
    ParameterSpec("stage4_auto_geometry_enabled", 4, "自动几何推导", "expert", "bool", help="允许从可信设备画像推导解析几何。"),
    ParameterSpec("stage4_spcc_enabled", 4, "优先 SPCC", "expert", "bool", help="关闭后不尝试 SPCC，宽带按安全回退策略处理。"),
    ParameterSpec(
        "stage4_auto_reference_global_white_enabled", 4,
        "应用恒星集合伪白参考", "expert", "bool",
        help="关闭时仍生成影子评估，但只允许自动背景中和修改像素。",
    ),
    ParameterSpec("stage4_spcc_timeout_sec", 4, "SPCC 超时", "expert", "int", 5, 300, 5, suffix=" 秒"),
    ParameterSpec(
        "stage4_spcc_online_unverified_timeout_sec", 4,
        "在线未验证 SPCC 超时", "expert", "int", 30, 180, 15,
        suffix=" 秒",
        help="仅有在线 XP 端点可达证据时的单次预算；localgaia 使用普通 SPCC 超时。",
    ),
    ParameterSpec("stage4_pcc_timeout_sec", 4, "PCC 超时", "expert", "int", 5, 180, 5, suffix=" 秒"),
    ParameterSpec(
        "stage4_spcc_osc_sensor", 4, "OSC 传感器", "expert", "choice",
        choices=_choice(
            ("Sony IMX415", "Sony IMX415"), ("Sony IMX462", "Sony IMX462"),
            ("Sony IMX585", "Sony IMX585"), ("Sony IMX662", "Sony IMX662"),
            ("Sony IMX678", "Sony IMX678"), ("ZWO Seestar S30", "ZWO Seestar S30"),
            ("ZWO Seestar S50", "ZWO Seestar S50"),
        ),
        help="仅允许应用内 SPCC 数据库中已校验的响应。",
    ),
    ParameterSpec(
        "stage4_spcc_osc_filter", 4, "OSC 滤镜响应", "expert", "choice",
        choices=_choice(
            ("UV/IR Block", "UV/IR Block"), ("Dwarf Mini Astro", "Dwarf Mini Astro"),
            ("ZWO Seestar LP", "ZWO Seestar LP"), ("No filter", "No filter"),
        ),
        help="仅允许应用内 SPCC 数据库中已校验的滤镜响应。",
    ),
    ParameterSpec(
        "stage4_spcc_white_ref", 4, "SPCC 白参考", "expert", "choice",
        choices=_choice(
            ("Average Spiral Galaxy", "Average Spiral Galaxy"),
            ("Star, type G2(v)", "Star, type G2(v)"),
        ),
    ),
    ParameterSpec("stage4_spcc_limit_magnitude", 4, "SPCC 极限星等", "expert", "float", 1.0, 25.0, 0.5, 1),
    ParameterSpec("stage4_spcc_narrowband_r_wavelength_nm", 4, "R/Ha 中心波长", "expert", "float", 300.0, 900.0, 0.1, 2, suffix=" nm"),
    ParameterSpec("stage4_spcc_narrowband_r_bandwidth_nm", 4, "R/Ha 带宽", "expert", "float", 1.0, 100.0, 0.5, 1, suffix=" nm"),
    ParameterSpec("stage4_spcc_narrowband_g_wavelength_nm", 4, "G/OIII 中心波长", "expert", "float", 300.0, 900.0, 0.1, 2, suffix=" nm"),
    ParameterSpec("stage4_spcc_narrowband_g_bandwidth_nm", 4, "G/OIII 带宽", "expert", "float", 1.0, 100.0, 0.5, 1, suffix=" nm"),
    ParameterSpec("stage4_spcc_narrowband_b_wavelength_nm", 4, "B/OIII 中心波长", "expert", "float", 300.0, 900.0, 0.1, 2, suffix=" nm"),
    ParameterSpec("stage4_spcc_narrowband_b_bandwidth_nm", 4, "B/OIII 带宽", "expert", "float", 1.0, 100.0, 0.5, 1, suffix=" nm"),
    ParameterSpec("stage4_narrowband_normalization_enabled", 4, "窄带归一化", "expert", "bool"),
    ParameterSpec("stage4_nbn_strength", 4, "窄带归一化强度", "expert", "float", 0.0, 1.0, 0.05, 2),
    ParameterSpec("stage4_local_star_wb_enabled", 4, "本地恒星白平衡", "expert", "bool"),
    ParameterSpec("stage4_local_star_wb_gain_limit", 4, "白平衡增益上限", "expert", "float", 1.01, 1.50, 0.01, 2, suffix="×"),
    ParameterSpec("denoise_enabled", 5, "线性降噪", "recommended", "bool", help="自动状态下允许 Stage 5 基于冻结基线低噪声门和候选质量门自行决定是否实际降噪。", depends_on=(("stage5_processing_mode", ("auto",)),)),
    ParameterSpec("denoise_mod", 5, "降噪强度", "recommended", "float", 0.20, 0.55, 0.05, 2, depends_on=(("stage5_processing_mode", ("auto",)), ("denoise_enabled", (True,)))),
    ParameterSpec(
        "stage5_deconvolution_mode", 5, "反卷积方式", "recommended", "choice",
        choices=_choice(
            ("GraXpert → Siril RL", "graxpert_rl"),
            ("仅 Siril RL", "rl"),
            ("关闭反卷积", "off"),
        ),
        help="自动状态下使用 GraXpert→RL 的安全回退链。",
    ),
    ParameterSpec("stage5_graxpert_deconv_strength", 5, "GraXpert 强度", "recommended", "float", 0.20, 0.40, 0.01, 2),
    ParameterSpec("stage5_graxpert_guard_retry_strength", 5, "GraXpert 保护重试强度", "expert", "float", 0.20, 0.30, 0.01, 2),
    ParameterSpec("stage5_multiscale_denoise_enabled", 5, "多尺度降噪", "expert", "bool"),
    ParameterSpec("stage5_multiscale_denoise_strength", 5, "多尺度降噪强度", "expert", "float", 0.10, 1.00, 0.05, 2),
    ParameterSpec("stage5_rl_iters", 5, "RL 迭代次数", "expert", "int", 1, 40, 1),
    ParameterSpec("stage5_rl_maxstars", 5, "PSF 最大星数", "expert", "int", 20, 1000, 20),
    ParameterSpec("stage5_rl_psf_kernel_size", 5, "PSF 核尺寸", "expert", "int", 9, 99, 2, odd=True),
    ParameterSpec("stage5_rl_alpha", 5, "RL Alpha", "expert", "float", 100.0, 10000.0, 100.0, 0),
    ParameterSpec("stage5_rl_gdstep", 5, "RL GD Step", "expert", "float", 0.00001, 0.01, 0.00005, 5),
    ParameterSpec("stage5_rl_stop", 5, "RL 停止阈值", "expert", "float", 0.0001, 0.05, 0.0005, 4),
    ParameterSpec("graxpert_object_model_path", 5, "GraXpert 模型文件", "expert", "path", help="必须是可读的本地 ONNX 文件。"),
    ParameterSpec(
        "stage6_processing_mode", 6, "处理方式", "recommended", "choice",
        choices=_choice(("自动分离", "auto"), ("保留含星图", "preserve")),
        stage_mode=True,
        help="保留含星图会建立明确的下游旁路状态，不会误当作 Starless。",
    ),
    ParameterSpec("stage7_quality_retry_max", 6, "去星质量重试", "recommended", "int", 0, 3, 1),
    ParameterSpec("stage7_starless_pixel_repair_enabled", 6, "无星图坏点修复", "recommended", "bool"),
    ParameterSpec("stage7_starless_repair_strength", 6, "残留修复强度", "expert", "float", 0.0, 0.85, 0.05, 2),
    ParameterSpec("stage7_starless_halo_repair_strength", 6, "光晕修复强度", "expert", "float", 0.0, 0.90, 0.05, 2),
    ParameterSpec("stage7_starless_chroma_denoise_strength", 6, "色噪修复强度", "expert", "float", 0.0, 0.90, 0.05, 2),
    ParameterSpec("stage7_galaxy_roi_star_clip_percentile", 6, "星系 ROI 亮星截尾分位", "expert", "float", 98.0, 99.9, 0.1, 1, suffix="%"),
    ParameterSpec("stage7_galaxy_roi_peak_floor_ratio", 6, "星系 ROI Q99 信号底比例", "expert", "float", 0.005, 0.10, 0.005, 3),
    ParameterSpec("stage7_galaxy_roi_min_extent_ratio", 6, "星系 ROI 最小尺度比例", "expert", "float", 0.004, 0.03, 0.001, 3),
    ParameterSpec("stage6_star_preserve_target_bypass_enabled", 6, "目标不适合时保留含星图", "expert", "bool"),
    ParameterSpec(
        "stage7_processing_mode", 7, "参数方式", "recommended", "choice",
        choices=_choice(("自动标定", "auto"), ("显式参数", "manual")),
        stage_mode=True,
        help="显式参数会重建亮度验收契约；复杂 GHS 候选改用独立安全门。",
    ),
    ParameterSpec("asinh_stretch", 7, "Asinh 拉伸强度", "recommended", "float", 1.60, 3.60, 0.05, 2),
    ParameterSpec("asinh_offset", 7, "Asinh Offset", "expert", "float", 0.0005, 0.0060, 0.0001, 4),
    ParameterSpec("ghs_shadowsclip", 7, "GHS Shadows Clip", "expert", "float", -3.60, -1.80, 0.05, 2),
    ParameterSpec("ghs_stretchamount", 7, "GHS Stretch Amount", "expert", "float", 1.00, 2.80, 0.05, 2),
    ParameterSpec(
        "stage7_iterative_masked_mtf_enabled",
        7,
        "条件启用 Iterative Masked MTF",
        "expert",
        "bool",
        help="仅在 target_bypass 的严格星团目标中替换 cand_a。",
        depends_on=(("stage7_processing_mode", ("auto",)),),
    ),
    ParameterSpec(
        "stage7_iterative_masked_mtf_iterations",
        7,
        "Masked MTF 迭代轮数",
        "expert",
        "int",
        8,
        32,
        1,
        0,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_iterative_masked_mtf_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_dual_stage_mtf_ghs_enabled",
        7,
        "条件启用 Dual-stage MTF+GHS",
        "expert",
        "bool",
        help="仅在可信冻结 ROI 的弱信号 Starless 大/小星系中替换 cand_a。",
        depends_on=(("stage7_processing_mode", ("auto",)),),
    ),
    ParameterSpec(
        "stage7_dual_stage_weak_snr_max",
        7,
        "Dual-stage 代理 SNR 上限",
        "expert",
        "float",
        3.5,
        12.0,
        0.1,
        1,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_dual_stage_mtf_ghs_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_dual_stage_subject_p90_min",
        7,
        "Dual-stage 主体 P90 下限",
        "expert",
        "float",
        0.10,
        0.35,
        0.01,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_dual_stage_mtf_ghs_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_dual_stage_subject_p90_max",
        7,
        "Dual-stage 主体 P90 上限",
        "expert",
        "float",
        0.15,
        0.50,
        0.01,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_dual_stage_mtf_ghs_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_dual_stage_ghs_b",
        7,
        "Dual-stage GHS B",
        "expert",
        "float",
        2.0,
        8.0,
        0.25,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_dual_stage_mtf_ghs_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_dual_stage_ghs_d_min",
        7,
        "Dual-stage GHS 搜索下界",
        "expert",
        "float",
        0.10,
        8.0,
        0.10,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_dual_stage_mtf_ghs_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_dual_stage_ghs_d_max",
        7,
        "Dual-stage GHS 搜索上界",
        "expert",
        "float",
        0.50,
        16.0,
        0.25,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_dual_stage_mtf_ghs_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_dual_stage_ghs_search_steps",
        7,
        "Dual-stage GHS 搜索采样数",
        "expert",
        "int",
        9,
        97,
        2,
        0,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            ("stage7_dual_stage_mtf_ghs_enabled", (True,)),
        ),
    ),
    ParameterSpec(
        "stage7_conditional_lut_max_derivative",
        7,
        "条件拉伸 LUT 最大导数",
        "expert",
        "float",
        250.0,
        20000.0,
        250.0,
        0,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            (
                "stage7_target_aware_stretch_enabled",
                (True,),
            ),
        ),
    ),
    ParameterSpec("stage8_masked_enhancement_enabled", 8, "分区增强", "recommended", "bool", depends_on=(("stage8_processing_mode", ("auto", "limited")),)),
    ParameterSpec("nebula_saturation", 8, "星云饱和度", "recommended", "float", 0.0, 0.65, 0.05, 2, depends_on=(("stage8_processing_mode", ("auto", "limited")), ("stage8_nebula_saturation_enabled", (True,)))),
    ParameterSpec("stage8_background_denoise_strength", 8, "背景降噪", "recommended", "float", 0.0, 0.25, 0.01, 2, depends_on=(("stage8_processing_mode", ("auto", "limited", "background_only")), ("stage8_background_denoise_enabled", (True,)))),
    ParameterSpec("stage8_faint_nebula_boost_max", 8, "暗弱星云提升", "recommended", "float", 0.0, 0.18, 0.01, 2, depends_on=(("stage8_processing_mode", ("auto", "limited")), ("stage8_faint_nebula_boost_enabled", (True,)))),
    ParameterSpec("stage8_nebula_contrast_max", 8, "星云对比度", "recommended", "float", 0.0, 0.20, 0.01, 2, depends_on=(("stage8_processing_mode", ("auto", "limited")), ("stage8_nebula_contrast_enabled", (True,)))),
    ParameterSpec("stage8_masked_unsharp_amount_max", 8, "蒙版锐化", "recommended", "float", 0.0, 0.25, 0.01, 2, depends_on=(("stage8_processing_mode", ("auto", "limited")), ("stage8_masked_unsharp_enabled", (True,)))),
    ParameterSpec("stage8_dualband_palette_enabled", 8, "双窄带哈勃色", "recommended", "bool", depends_on=(("stage8_processing_mode", ("auto",)),)),
    ParameterSpec(
        "stage8_dualband_palette_selection",
        8,
        "哈勃色方案",
        "recommended",
        "choice",
        choices=_choice(
            ("自动（按目标类型）", "auto"),
            ("HSO", "HSO"),
            ("SHO", "SHO"),
            ("OSH", "OSH"),
            ("OHS", "OHS"),
            ("HOS", "HOS"),
            ("HOO", "HOO"),
        ),
        help="自动模式按冻结目标类型选择；手动模式只执行指定的常规 Classic 色盘。",
        depends_on=(
            ("stage8_processing_mode", ("auto",)),
            ("stage8_dualband_palette_enabled", (True,)),
        ),
    ),
    ParameterSpec("stage8_dualband_palette_strength", 8, "艺术调色强度", "recommended", "float", 0.10, 1.00, 0.05, 2, depends_on=(("stage8_processing_mode", ("auto",)), ("stage8_dualband_palette_enabled", (True,)))),
    ParameterSpec("optional_color_transform_enabled", 8, "可选调色插件", "expert", "bool"),
    ParameterSpec("nebula_bg_factor", 8, "背景饱和抑制", "expert", "int", 0, 10, 1),
    ParameterSpec("stage8_local_adjustment_engine_enabled", 8, "本地调整引擎", "expert", "bool"),
    ParameterSpec("stage8_local_curve_opacity", 8, "本地曲线透明度", "expert", "float", 0.0, 1.0, 0.05, 2),
    ParameterSpec("stage8_core_protection_strength", 8, "核心保护", "expert", "float", 0.50, 1.00, 0.05, 2),
    ParameterSpec("stage8_blue_precontrol_strength", 8, "蓝色预控制", "expert", "float", 0.0, 1.0, 0.05, 2),
    ParameterSpec("star_intensity", 9, "恒星强度", "recommended", "float", 0.80, 1.05, 0.05, 2, depends_on=(("stage9_processing_mode", ("auto",)),)),
    ParameterSpec("stage9_unscreen_candidate_enabled", 9, "同域 Unscreen 候选", "recommended", "bool", depends_on=(("stage9_processing_mode", ("auto",)),)),
    ParameterSpec("stage9_targeted_recovery_enabled", 9, "定向星点恢复", "expert", "bool", help="按正式失败类型启用分组软补翼或局部色差收缩；不会放宽任何质量门。", depends_on=(("stage9_processing_mode", ("auto",)),)),
    ParameterSpec("stage9_targeted_recovery_retry_max", 9, "定向恢复尝试上限", "expert", "int", 0, 4, 1, suffix=" 次", depends_on=(("stage9_processing_mode", ("auto",)), ("stage9_targeted_recovery_enabled", (True,)))),
    ParameterSpec("stage9_stage5_bright_star_completion_enabled", 9, "冻结大亮星补全", "recommended", "bool", depends_on=(("stage9_processing_mode", ("auto",)),)),
    ParameterSpec("stage9_star_color_repair_enabled", 9, "恒星颜色修复", "recommended", "bool", depends_on=(("stage9_processing_mode", ("auto",)),)),
    ParameterSpec("stage9_star_color_repair_strength", 9, "颜色修复强度", "recommended", "float", 0.0, 1.0, 0.05, 2, depends_on=(("stage9_processing_mode", ("auto",)), ("stage9_star_color_repair_enabled", (True,)))),
    ParameterSpec("stage9_sasp_star_stretch_enabled", 9, "SASP 星点拉伸", "recommended", "bool", help="直接使用随包 SASP 无头适配器，不依赖 Siril 插件命令探测。", depends_on=(("stage9_processing_mode", ("auto",)),)),
    ParameterSpec("stage9_nb_to_rgb_stars_enabled", 9, "NB→RGB 星点", "recommended", "bool", help="只在 Stage 4 已确认 R→Ha、G/B→OIII 时运行。", depends_on=(("stage9_processing_mode", ("auto",)),)),
    ParameterSpec("stage9_sasp_star_stretch_amount", 9, "SASP 星点拉伸强度", "expert", "float", 0.50, 5.00, 0.25, 2, depends_on=(("stage9_processing_mode", ("auto",)), ("stage9_sasp_star_stretch_enabled", (True,)))),
    ParameterSpec("stage9_nb_to_rgb_stars_ratio", 9, "NB→RGB Ha:OIII 比", "expert", "float", 0.0, 1.0, 0.05, 2, depends_on=(("stage9_processing_mode", ("auto",)), ("stage9_nb_to_rgb_stars_enabled", (True,)))),
    ParameterSpec("stage9_starmask_stretch_enabled", 9, "星掩膜拉伸", "expert", "bool"),
    ParameterSpec("stage9_starmask_adaptive_stretch_enabled", 9, "自适应星掩膜拉伸", "expert", "bool"),
    ParameterSpec(
        "stage9_compact_starmask_enabled",
        9,
        "紧凑支持候选",
        "expert",
        "bool",
        help="控制 normal/strict 支持竞争和更紧支持恢复；不控制拉伸前像素缩星。",
    ),
    ParameterSpec(
        "stage9_starmask_pre_stretch_compact_enabled",
        9,
        "拉伸前缩星",
        "expert",
        "bool",
        help="仅对内置星掩膜拉伸生效；开启后先按当前 normal/strict 支持裁剪星层。SASP 已拉伸星层不受影响。",
        depends_on=(("stage9_starmask_stretch_enabled", (True,)),),
    ),
    ParameterSpec("stage9_starmask_asinh_stretch", 9, "固定 Asinh 强度", "expert", "float", 1.10, 3.00, 0.10, 2),
    ParameterSpec("stage9_starmask_asinh_offset", 9, "固定 Asinh Offset", "expert", "float", 0.0005, 0.0060, 0.0001, 4),
    ParameterSpec("stage9_weak_star_recovery_ratio_min", 9, "弱星恢复门槛", "expert", "float", 0.40, 0.95, 0.05, 2),
    _spec(
        "stage9_psf_recovery_target_min",
        9,
        "PSF 恢复软目标下限",
        "float",
        0.50,
        1.00,
        0.01,
        2,
        section="quality_gate",
        depends_on=(("stage9_psf_size_gate_enabled", (True,)),),
        strictness="higher_is_stricter",
        help="通过 FWHM 硬门后的星翼恢复软目标；不会放宽硬门下限。",
    ),
    _spec(
        "stage9_psf_recovery_target_max",
        9,
        "PSF 恢复软目标上限",
        "float",
        1.00,
        1.50,
        0.01,
        2,
        section="quality_gate",
        depends_on=(("stage9_psf_size_gate_enabled", (True,)),),
        strictness="lower_is_stricter",
        help="限制统一补翼的软目标上限；仍受 FWHM 硬门上限约束。",
    ),
    _spec(
        "stage10_processing_mode",
        10,
        "处理方式",
        "choice",
        level="recommended",
        choices=_choice(("自动末端处理", "auto"), ("保留 Stage 9 含星源", "preserve")),
        stage_mode=True,
        section="execution",
        help="保留模式跳过像素修改，但仍执行最终质量门与导出。",
    ),
    _spec(
        "stage10_final_denoise_enabled",
        10,
        "末端降噪",
        "bool",
        level="recommended",
        section="execution",
        depends_on=(("stage10_processing_mode", ("auto",)),),
    ),
    _spec(
        "stage10_final_saturation_enabled",
        10,
        "最终饱和度",
        "bool",
        level="recommended",
        section="execution",
        depends_on=(("stage10_processing_mode", ("auto",)),),
    ),
    _spec(
        "final_saturation",
        10,
        "最终饱和度强度",
        "float",
        0.0,
        0.25,
        0.01,
        2,
        level="recommended",
        depends_on=(
            ("stage10_processing_mode", ("auto",)),
            ("stage10_final_saturation_enabled", (True,)),
        ),
    ),
    _spec(
        "stage10_final_denoise_strength",
        10,
        "末端降噪强度",
        "float",
        0.05,
        0.50,
        0.01,
        2,
        level="recommended",
        depends_on=(
            ("stage10_processing_mode", ("auto",)),
            ("stage10_final_denoise_enabled", (True,)),
        ),
    ),
    _spec(
        "stage10_denoise_backend_policy",
        10,
        "末端降噪后端",
        "choice",
        choices=_choice(
            ("自动安全链", "auto_chain"),
            ("仅 CosmicClarity", "cosmic_only"),
            ("仅 SCUNet", "scunet_only"),
        ),
        section="execution",
        depends_on=(
            ("stage10_processing_mode", ("auto",)),
            ("stage10_final_denoise_enabled", (True,)),
        ),
    ),
    _spec(
        "final_bg_factor",
        10,
        "最终饱和度背景抑制",
        "int",
        0,
        10,
        1,
        0,
        depends_on=(
            ("stage10_processing_mode", ("auto",)),
            ("stage10_final_saturation_enabled", (True,)),
        ),
    ),
    _spec("stage10_chroma_focus_score_min", 10, "色度优先评分下限", "float", 0.10, 0.80, 0.01, 2, section="quality_gate", strictness="lower_is_stricter"),
    _spec("stage10_separate_chroma_score_min", 10, "分通道降噪评分下限", "float", 0.35, 1.50, 0.01, 2, section="quality_gate", strictness="lower_is_stricter"),
    _spec("stage10_full_bg_std_min", 10, "全通道降噪背景噪声下限", "float", 0.001, 0.10, 0.001, 3, section="quality_gate", strictness="lower_is_stricter"),
    _spec("stage10_full_mottling_score_min", 10, "全通道降噪斑驳评分下限", "float", 0.10, 1.00, 0.01, 2, section="quality_gate", strictness="lower_is_stricter"),
    _spec(
        "stage10_star_protection_coverage_max",
        10,
        "星点保护覆盖上限",
        "float",
        0.05,
        0.60,
        0.01,
        2,
        section="quality_gate",
        depends_on=(("stage10_final_denoise_enabled", (True,)),),
        strictness="lower_is_stricter",
    ),
    _spec("stage10_large_galaxy_local_patch_variance_max", 10, "大型星系局部方差上限", "float", 0.00022, 0.00100, 0.00001, 5, section="quality_gate", strictness="lower_is_stricter"),
    _spec(
        "stage10_stage9_local_color_risk_strength",
        10,
        "Stage 9 局部颜色风险抑制",
        "float",
        0.0,
        1.0,
        0.05,
        2,
        section="quality_gate",
        depends_on=(("stage10_final_saturation_enabled", (True,)),),
    ),
    _spec(
        "stage10_quality_repair_enabled",
        10,
        "最终质量修复",
        "bool",
        section="fallback",
        depends_on=(("stage10_processing_mode", ("auto",)),),
    ),
    _spec(
        "aberration_api_enabled",
        10,
        "实验性 Aberration 回退",
        "bool",
        section="fallback",
        depends_on=(
            ("stage10_processing_mode", ("auto",)),
            ("stage10_final_denoise_enabled", (True,)),
            ("stage10_denoise_backend_policy", ("auto_chain",)),
        ),
        help="仅作为自动后端链末档实验回退；默认关闭。",
    ),
    _spec(
        "stage10_failure_action",
        10,
        "失败处置",
        "choice",
        choices=_choice(
            ("自动安全回退", "auto_fallback"),
            ("恢复 Stage 9 并复核", "preserve_review"),
            ("写诊断后终止任务", "stop"),
        ),
        section="fallback",
        strictness="stop_is_stricter",
        help="决定性失败时的处置；不可绕过最终含星与质量不变量。",
    ),
)


PROCESSING_PARAMETER_SPECS += (
    *(_failure_action_spec(stage) for stage in FAILURE_ACTION_STAGES),

    # Stage 2: first-party crop execution and active edge gates.
    _spec("stage2_base_crop_enabled", 2, "基础裁边", "bool", section="execution"),
    _spec(
        "stage2_base_crop_margin", 2, "基础裁边比例", "float",
        0.0, 0.06, 0.005, 3, suffix="×", section="algorithm",
        depends_on=(("stage2_base_crop_enabled", (True,)),),
    ),
    _spec(
        "stage2_field_rotation_detection_enabled", 2, "场旋覆盖裁切", "bool",
        level="recommended", section="execution",
    ),
    _spec(
        "stage2_field_rotation_max_passes", 2, "场旋自动裁切轮次", "int",
        1, 2, 1, 0, level="expert", section="algorithm",
        depends_on=(("stage2_field_rotation_detection_enabled", (True,)),),
        help="总计最多两次；第二次裁切后必须重新验证残余场旋。",
    ),
    _spec(
        "stage2_field_rotation_noise_ratio_min", 2, "场旋亮度噪声倍率", "float",
        1.15, 3.0, 0.05, 2, suffix="×", section="process_gate",
        depends_on=(("stage2_field_rotation_detection_enabled", (True,)),),
        strictness="higher_is_stricter",
    ),
    _spec(
        "stage2_field_rotation_chroma_ratio_min", 2, "场旋色噪倍率", "float",
        1.10, 3.0, 0.05, 2, suffix="×", section="process_gate",
        depends_on=(("stage2_field_rotation_detection_enabled", (True,)),),
        strictness="higher_is_stricter",
    ),
    _spec(
        "stage2_color_edge_cleanup_enabled", 2, "边缘色偏清理", "bool",
        section="execution",
    ),
    _spec(
        "stage2_level_artifact_window", 2, "水平伪影窗口", "int",
        9, 201, 2, 0, odd=True, suffix=" px", section="process_gate",
        strictness="contextual",
    ),
    _spec(
        "stage2_edge_black_improvement_min", 2, "最小黑边改善", "float",
        0.0, 0.05, 0.001, 3, section="process_gate",
        strictness="higher_is_stricter",
    ),
    _spec(
        "stage2_preserve_review_edge_black_max", 2, "保留模式复核线", "float",
        0.05, 0.50, 0.01, 2, section="quality_gate",
        strictness="lower_is_stricter",
    ),
    _spec(
        "stage2_edge_cast_absolute_max", 2, "边缘色偏绝对线", "float",
        0.002, 0.05, 0.001, 3, section="process_gate",
        depends_on=(("stage2_color_edge_cleanup_enabled", (True,)),),
        strictness="lower_is_stricter",
    ),
    _spec(
        "stage2_edge_cast_detect_ratio_max", 2, "色偏检测相对倍率", "float",
        1.2, 5.0, 0.1, 1, suffix="×", section="process_gate",
        depends_on=(("stage2_color_edge_cleanup_enabled", (True,)),),
        strictness="lower_is_stricter",
    ),
    _spec(
        "stage2_edge_cast_cleanup_ratio_max", 2, "色偏清理相对倍率", "float",
        1.2, 5.0, 0.1, 1, suffix="×", section="quality_gate",
        depends_on=(("stage2_color_edge_cleanup_enabled", (True,)),),
        strictness="lower_is_stricter",
    ),
    _spec(
        "stage2_color_cleanup_strip_ratio", 2, "色偏清理带宽", "float",
        0.005, 0.05, 0.001, 3, suffix="×", section="algorithm",
        depends_on=(("stage2_color_edge_cleanup_enabled", (True,)),),
    ),

    # Stage 3: candidate source and search controls.
    _spec(
        "stage3_backend_policy", 3, "背景提取后端", "choice",
        choices=_choice(
            ("安全自动链", "auto_chain"),
            ("仅 GraXpert", "graxpert_only"),
            ("仅内置 Polynomial/RBF", "builtin_only"),
        ),
        section="execution",
    ),
    _spec(
        "stage3_gate_profile", 3, "背景门禁策略", "choice",
        choices=_choice(
            ("输出优先", "output_first"),
            ("平衡", "balanced"),
            ("严格保真", "strict"),
        ),
        section="quality_gate",
        help="软告警候选仍可出图；仅所选档位定义的过度异常触发硬回滚。",
    ),
    _spec(
        "stage3_plugin_fallback_enabled", 3, "插件回退", "bool",
        section="execution",
        depends_on=(("stage3_backend_policy", ("auto_chain",)),),
    ),
    _spec(
        "stage3_compound_candidate_enabled", 3, "复合候选", "bool",
        section="execution",
        depends_on=(("stage3_backend_policy", ("auto_chain", "builtin_only")),),
    ),
    _spec(
        "stage3_candidate_attempt_limit", 3, "候选尝试上限", "int",
        0, 9, 1, 0, section="fallback",
        help="0 表示不限制；只截断已启用候选，不绕过验收门。",
    ),

    # Stage 4: physical-color fallback and trusted-header solve variants.
    _spec("stage4_pcc_fallback_enabled", 4, "PCC 回退", "bool", section="fallback"),
    _spec(
        "stage4_narrowband_degraded_pcc_enabled", 4, "窄带降级 PCC", "bool",
        section="fallback",
        depends_on=(("stage4_pcc_fallback_enabled", (True,)),),
    ),
    _spec(
        "stage4_header_guided_platesolve_enabled", 4, "Header 引导解析", "bool",
        section="execution",
        depends_on=(("stage4_platesolve_enabled", (True,)),),
    ),

    # Stage 5: safe passthrough, denoise backend and deconvolution rollback gates.
    _spec(
        "stage5_processing_mode", 5, "处理方式", "choice",
        level="recommended", stage_mode=True,
        choices=_choice(("自动处理", "auto"), ("保留线性结果", "preserve")),
        help="保留模式复制 Stage 4 线性结果并生成完整诊断。",
    ),
    _spec(
        "stage5_denoise_backend_policy", 5, "降噪后端", "choice",
        choices=_choice(
            ("自动链", "auto_chain"),
            ("仅多尺度", "multiscale_only"),
            ("仅 Siril", "siril_only"),
            ("仅 CosmicClarity", "cosmic_clarity_only"),
        ),
        section="execution",
    ),
    _spec("stage5_low_noise_auto_skip_enabled", 5, "低噪声自动跳过", "bool", section="execution"),
    _spec(
        "stage5_deconv_bg_std_growth_max", 5, "反卷积背景噪声增长", "float",
        1.0, 2.0, 0.01, 2, suffix="×", section="quality_gate",
        strictness="lower_is_stricter",
    ),
    _spec(
        "stage5_deconv_chroma_growth_max", 5, "反卷积色噪增长", "float",
        1.0, 2.0, 0.01, 2, suffix="×", section="quality_gate",
        strictness="lower_is_stricter",
    ),
    _spec(
        "stage5_deconv_chroma_ratio_growth_max", 5, "色噪/背景比增长", "float",
        1.0, 3.0, 0.01, 2, suffix="×", section="quality_gate",
        strictness="lower_is_stricter",
    ),
    _spec(
        "stage5_deconv_dirty_delta_max", 5, "脏背景增量", "float",
        0.0, 0.25, 0.01, 2, section="quality_gate",
        strictness="lower_is_stricter",
    ),

    # Stage 6: backend selection and existing effective starmask controls.
    _spec(
        "stage6_starless_backend_policy", 6, "去星后端", "choice",
        choices=_choice(
            ("SyQon → SASP", "auto_chain"),
            ("仅 SyQon", "syqon_only"),
            ("仅 SASP", "sasp_only"),
        ),
        section="execution",
    ),
    _spec("stage7_starmask_clean_enabled", 6, "星掩膜清理", "bool", section="execution"),
    _spec(
        "stage7_starmask_halo_blur_strength", 6, "光晕衰减", "float",
        0.0, 0.80, 0.05, 2, section="algorithm",
        depends_on=(("stage7_starmask_clean_enabled", (True,)),),
    ),
    _spec(
        "stage7_starmask_small_star_scale", 6, "弱星保留", "float",
        0.50, 1.0, 0.01, 2, section="algorithm",
        depends_on=(("stage7_starmask_clean_enabled", (True,)),),
    ),
    _spec(
        "stage7_starmask_nebula_suppression", 6, "星云污染抑制", "float",
        0.0, 0.95, 0.05, 2, section="algorithm",
        depends_on=(("stage7_starmask_clean_enabled", (True,)),),
    ),
    _spec("stage7_conservative_repair_enabled", 6, "保守参数重试", "bool", section="fallback"),
    _spec(
        "stage6_syqon_seam_retry_enabled",
        6,
        "分块伪影安全重试",
        "bool",
        section="fallback",
        help="分块像素门硬失败时，仅以 CPU/FP32、128 px 重叠重试一次。",
    ),
    _spec(
        "stage7_quality_advisory_multiplier", 6, "去星软告警倍率", "float",
        1.0, 4.0, 0.1, 1, suffix="×", section="process_gate",
        strictness="lower_is_stricter",
    ),

    # Stage 7: candidate routing and bounded chroma rescue.
    _spec(
        "stage7_rendition_intent", 7, "成片呈现", "choice",
        level="recommended",
        choices=_choice(
            ("鲜艳且安全", "vivid_safe"),
            ("自然平衡", "balanced"),
            ("保守呈现", "conservative"),
        ),
        section="execution",
        help="调整候选生成与选优目标，不放宽裁切、结构或数值完整性门禁。",
    ),
    _spec(
        "stage7_forced_delivery_enabled", 7, "画质失败保留复核候选", "bool",
        level="recommended",
        section="fallback",
        depends_on=(("stage7_failure_action", ("auto_fallback",)),),
        help="只保留技术完整的最佳失败候选供诊断；最终仅允许含星 review-only 输出。",
    ),
    _spec(
        "stage7_candidate_policy", 7, "拉伸候选", "choice",
        choices=_choice(
            ("自动 A/B/Display90", "auto_display90"),
            ("自动双候选", "auto_dual"),
            ("仅候选 A", "candidate_a_only"),
            ("仅候选 B", "candidate_b_only"),
            ("仅 Display90", "display90_only"),
        ),
        section="execution",
    ),
    _spec(
        "stage7_display90_strength",
        7,
        "Display90 曲线强度",
        "float",
        0.50,
        0.95,
        0.01,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            (
                "stage7_candidate_policy",
                ("auto_display90", "display90_only"),
            ),
        ),
        help="以共享 RGB 单调 LUT 保留 Stage 6 GUI linked 显示曲线的比例。",
    ),
    _gate(
        "stage7_display90_reference_chroma_load_ratio_max",
        7,
        "Display90 GUI 色度比值",
        "float",
        1.00,
        1.20,
        0.01,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            (
                "stage7_candidate_policy",
                ("auto_display90", "display90_only"),
            ),
        ),
        help=(
            "仅在窄带 SPCC_NARROWBAND 已接受时，限制 Display90 候选背景"
            "色度负载相对真实 GUI linked D 参考的最大比值。"
        ),
    ),
    _gate(
        "stage7_display90_reference_chroma_load_absolute_max",
        7,
        "Display90 GUI 绝对色度",
        "float",
        0.15,
        0.50,
        0.01,
        2,
        depends_on=(
            ("stage7_processing_mode", ("auto",)),
            (
                "stage7_candidate_policy",
                ("auto_display90", "display90_only"),
            ),
        ),
        help=(
            "GUI D 参考匹配不能越过的 Display90 候选背景绝对色度负载"
            "硬上限；不使用共享 advisory 倍率放宽。"
        ),
    ),
    _spec("stage7_bright_nebula_star_mask_expand", 7, "亮星区扩张", "int", 1, 8, 1, 0, suffix=" px"),
    _spec("stage7_bright_nebula_star_faint_suppression", 7, "弱信号抑制", "float", 0.0, 1.0, 0.05, 2),
    _spec("stage7_bright_nebula_star_detail_suppression", 7, "细节抑制", "float", 0.0, 0.60, 0.01, 2),
    _spec(
        "stage7_vivid_subject_chroma_enabled", 7, "安全主体增色", "bool",
        section="algorithm",
        depends_on=(("stage7_rendition_intent", ("vivid_safe",)),),
        help="仅增强冻结主体区域的低频色度，并逐像素限制到可用 RGB 余量。",
    ),
    _spec(
        "stage7_chroma_rescue_max_attempts", 7, "色度救援次数", "int",
        0, 3, 1, 0, section="fallback",
    ),

    # Stage 8: user upper-bound mode and independently switchable substeps.
    _spec(
        "stage8_processing_mode", 8, "处理方式", "choice",
        level="recommended", stage_mode=True,
        choices=_choice(
            ("自动", "auto"),
            ("受限增强", "limited"),
            ("仅背景处理", "background_only"),
            ("保留输入", "preserve"),
        ),
        help="只作为增强上限；上游安全门仍可继续降级。",
    ),
    _spec("stage8_nebula_saturation_enabled", 8, "启用星云饱和度", "bool", section="execution", depends_on=(("stage8_processing_mode", ("auto", "limited")),)),
    _spec("stage8_background_denoise_enabled", 8, "启用背景降噪", "bool", section="execution", depends_on=(("stage8_processing_mode", ("auto", "limited", "background_only")),)),
    _spec("stage8_faint_nebula_boost_enabled", 8, "启用暗弱星云提升", "bool", section="execution", depends_on=(("stage8_processing_mode", ("auto", "limited")),)),
    _spec("stage8_nebula_contrast_enabled", 8, "启用星云对比度", "bool", section="execution", depends_on=(("stage8_processing_mode", ("auto", "limited")),)),
    _spec("stage8_masked_unsharp_enabled", 8, "启用蒙版锐化", "bool", section="execution", depends_on=(("stage8_processing_mode", ("auto", "limited")),)),
    _spec(
        "stage8_quality_retry_max", 8, "保守重跑次数", "int",
        0, 1, 1, 0, section="fallback",
    ),

    # Stage 9: verified-with-stars passthrough and bounded fallback ladder.
    _spec(
        "stage9_processing_mode", 9, "处理方式", "choice",
        level="recommended", stage_mode=True,
        choices=_choice(
            ("自动重组", "auto"),
            ("保留可信含星源", "preserve_with_stars"),
        ),
        help="保留模式仍强制复核；没有可信含星源时 fail-closed。",
    ),
    _spec("stage9_fallback_intensity_cap", 9, "回退强度上限", "float", 0.40, 1.05, 0.05, 2, section="fallback"),
    _spec("stage9_fallback_retry_max", 9, "回星回退次数", "int", 0, 3, 1, 0, section="fallback"),
    _spec(
        "stage9_fallback_intensity_floor", 9, "回星强度下限", "float",
        0.40, 0.75, 0.05, 2, section="fallback",
        depends_on=(("stage9_fallback_retry_max", (1, 2, 3)),),
    ),
)


# These are deliberately separate from the primary processing controls above:
# they remain hidden behind the global expert switch, but otherwise use the
# exact same task snapshot, validation, audit and resume-fingerprint contract.
PROCESSING_GATE_PARAMETER_SPECS: Tuple[ParameterSpec, ...] = (
    # Stage 2: crop detection and center/edge protection.
    _gate("stage2_edge_black_target", 2, "残余黑边目标", "float", 0.03, 0.18, 0.005, 3, suffix="×"),
    _gate("stage2_adaptive_edge_crop_max_passes", 2, "自适应裁边最大轮次", "int", 0, 6, 1),
    _gate("stage2_adaptive_edge_crop_max_extra", 2, "单轮额外裁边上限", "float", 0.005, 0.060, 0.005, 3, suffix="×"),
    _gate("stage2_guard_band_pixels", 2, "边缘护带宽度", "int", 0, 8, 1, suffix=" px"),

    # Stage 3: audited samples, compound-candidate validation and routing.
    _gate("bg_quality_gate_enabled", 3, "背景质量总门", "bool"),
    _gate("stage3_conditional_decision_enabled", 3, "先诊断后决策", "bool"),
    _gate("stage3_deterministic_auto_apply_enabled", 3, "高置信自动应用", "bool"),
    _gate("stage3_apply_confidence_min", 3, "应用置信度下限", "float", 0.50, 0.99, 0.01, 2),
    _gate("stage3_safe_sample_target_count", 3, "安全样点目标数", "int", 16, 64, 1),
    _gate("stage3_safe_sample_min_count", 3, "安全样点下限", "int", 12, 48, 1),
    _gate("stage3_safe_sample_patch_radius", 3, "样点审计半径", "int", 4, 24, 1, suffix=" px"),
    _gate("stage3_safe_sample_brightness_quantile_max", 3, "样点亮度分位上限", "float", 0.50, 0.85, 0.01, 2),
    _gate("stage3_safe_sample_texture_quantile_max", 3, "样点纹理分位上限", "float", 0.25, 0.75, 0.01, 2),
    _gate("stage3_compound_min_sample_count", 3, "复合候选样点下限", "int", 12, 64, 1),
    _gate("stage3_compound_fit_min_count", 3, "复合拟合样点下限", "int", 8, 56, 1),
    _gate("stage3_compound_validation_min_count", 3, "冻结验证样点下限", "int", 4, 20, 1),
    _gate("stage3_compound_validation_ratio", 3, "冻结验证样点比例", "float", 0.15, 0.35, 0.01, 2),
    _gate("stage3_compound_score_abs_improvement_min", 3, "复合评分绝对改善", "float", 0.03, 0.15, 0.01, 2),
    _gate("stage3_compound_score_rel_improvement_min", 3, "复合评分相对改善", "float", 0.10, 0.40, 0.01, 2),
    _gate("stage3_compound_validation_improvement_min", 3, "验证残差改善下限", "float", 0.10, 0.40, 0.01, 2),
    _gate("stage3_compound_zero_point_abs_max", 3, "天空零点绝对漂移", "float", 0.002, 0.010, 0.001, 3),
    _gate("stage3_compound_zero_point_rel_max", 3, "天空零点相对漂移", "float", 0.05, 0.15, 0.01, 2),
    _gate("stage3_pattern_routing_enabled", 3, "方向噪声分流", "bool"),
    _gate("stage3_pattern_score_min", 3, "方向噪声评分下限", "float", 0.25, 0.90, 0.01, 2),
    _gate("stage3_walking_noise_score_min", 3, "Walking Noise 评分下限", "float", 0.25, 0.90, 0.01, 2),
    _gate("stage3_pattern_score_growth_max", 3, "方向噪声增长上限", "float", 0.02, 0.40, 0.01, 2),

    # Stage 4: geometry, narrowband normalization and color-candidate rollback.
    _gate("stage4_auto_reference_background_sample_target", 4, "自动参考背景样点目标", "int", 16, 64, 1),
    _gate("stage4_auto_reference_background_sample_min", 4, "自动参考背景样点下限", "int", 16, 40, 1),
    _gate("stage4_auto_reference_holdout_ratio", 4, "自动参考留出比例", "float", 0.20, 0.40, 0.01, 2),
    _gate("stage4_auto_reference_background_error_min", 4, "背景色差触发下限", "float", 0.0, 0.25, 0.001, 3),
    _gate("stage4_auto_reference_background_improvement_min", 4, "留出背景改善下限", "float", 0.01, 0.90, 0.01, 2),
    _gate("stage4_auto_reference_star_min_objects", 4, "恒星集合对象下限", "int", 16, 256, 1),
    _gate("stage4_auto_reference_star_ratio_mad_max", 4, "恒星色比离散上限", "float", 0.01, 0.50, 0.01, 2),
    _gate("stage4_auto_reference_star_saturation_ratio_max", 4, "恒星饱和比例上限", "float", 0.0, 0.50, 0.01, 2),
    _gate("stage4_auto_reference_gain_limit", 4, "伪白参考增益上限", "float", 1.01, 1.20, 0.01, 2, suffix="×"),
    _gate("stage4_auto_reference_star_improvement_min", 4, "留出恒星改善下限", "float", 0.01, 0.90, 0.01, 2),
    _gate("stage4_auto_reference_highlight_clip_growth_max", 4, "自动参考高光增长", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage4_auto_reference_black_clip_growth_max", 4, "自动参考黑位增长", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage4_auto_reference_gradient_growth_max", 4, "自动参考梯度增长", "float", 1.0, 2.0, 0.01, 2, suffix="×"),
    _gate("stage4_auto_reference_texture_growth_max", 4, "自动参考纹理增长", "float", 1.0, 2.0, 0.01, 2, suffix="×"),
    _gate("stage4_auto_reference_target_chroma_drift_max", 4, "自动参考主体色漂移", "float", 0.01, 0.50, 0.01, 2),
    _gate("stage4_auto_geometry_confidence_min", 4, "自动几何置信度", "float", 0.0, 1.0, 0.01, 2),
    _gate("stage4_auto_geometry_scale_residual_max", 4, "WCS 比例残差", "float", 0.01, 0.25, 0.01, 2),
    _gate("stage4_nbn_mapping_confidence_min", 4, "窄带映射置信度", "float", 0.70, 0.99, 0.01, 2),
    _gate("stage4_nbn_gain_limit", 4, "窄带通道增益上限", "float", 1.01, 1.15, 0.01, 2, suffix="×"),
    _gate("stage4_nbn_line_ratio_drift_max", 4, "Ha/OIII 比例漂移", "float", 0.04, 0.20, 0.01, 2),
    _gate("stage4_pcc_quality_gate_enabled", 4, "SPCC/PCC 质量门", "bool"),
    _gate("stage4_pcc_channel_gain_ratio_max", 4, "校色通道增益跨度", "float", 1.10, 10.0, 0.10, 2, suffix="×"),
    _gate("stage4_pcc_emission_balance_gain_ratio_max", 4, "发射星云增益跨度", "float", 1.10, 5.0, 0.10, 2, suffix="×"),
    _gate("stage4_pcc_clip_growth_max", 4, "校色高光增长", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage4_pcc_star_temperature_ratio_min", 4, "恒星色温比例下限", "float", 0.20, 0.95, 0.01, 2, suffix="×"),
    _gate("stage4_pcc_star_temperature_ratio_max", 4, "恒星色温比例上限", "float", 1.05, 5.0, 0.05, 2, suffix="×"),
    _gate("stage4_pcc_background_color_delta_max", 4, "背景色差上限", "float", 0.05, 0.60, 0.01, 2),
    _gate("stage4_pcc_target_color_drift_max", 4, "主体色度漂移", "float", 0.05, 0.75, 0.01, 2),
    _gate("stage4_pcc_emission_target_color_drift_max", 4, "发射主体色度漂移", "float", 0.05, 0.75, 0.01, 2),
    _gate("stage4_local_star_wb_min_pixels", 4, "本地白平衡样本下限", "int", 16, 4096, 16),
    _gate("stage4_local_star_mask_radius", 4, "恒星软遮罩半径", "int", 1, 4, 1, suffix=" px"),
    _gate("stage4_local_star_mask_coverage_max", 4, "恒星软遮罩覆盖上限", "float", 0.01, 0.30, 0.01, 2),

    # Stage 5: denoise strength cap and shared candidate acceptance gates.
    _gate("denoise_safety_max", 5, "降噪强度安全上限", "float", 0.20, 0.55, 0.01, 2),
    _gate("stage5_multiscale_detail_retention_min", 5, "主体细节保留下限", "float", 0.70, 0.98, 0.01, 2),
    _gate("stage5_multiscale_noise_reduction_min", 5, "背景噪声下降下限", "float", 0.0, 0.50, 0.01, 2),
    _gate("stage5_denoise_chroma_noise_growth_max", 5, "背景色噪增长上限", "float", 1.0, 1.50, 0.01, 2, suffix="×"),

    # Stage 6: starless/starmask diagnostics and repair acceptance.
    _gate("stage7_edge_black_warn", 6, "黑边风险提示", "float", 0.04, 0.30, 0.01, 2),
    _gate("stage7_edge_black_high", 6, "黑边高风险", "float", 0.04, 0.60, 0.01, 2),
    _gate("stage7_bg_median_high", 6, "背景中值高风险", "float", 0.08, 0.35, 0.01, 2),
    _gate("stage7_bg_std_high", 6, "背景噪声高风险", "float", 0.020, 0.120, 0.005, 3),
    _gate("stage7_bg_noise_ratio_high", 6, "噪声背景比高风险", "float", 0.20, 1.50, 0.05, 2),
    _gate("stage7_residual_star_score_max", 6, "残星评分上限", "float", 0.10, 1.20, 0.05, 2),
    _gate("stage7_halo_residue_score_max", 6, "星晕残留上限", "float", 0.05, 1.00, 0.05, 2),
    _gate("stage7_large_galaxy_halo_residue_score_max", 6, "大星系星晕上限", "float", 0.05, 1.00, 0.05, 2),
    _gate("stage7_bright_nebula_halo_residue_score_max", 6, "亮星云星晕上限", "float", 0.05, 1.20, 0.05, 2),
    _gate("stage7_galaxy_roi_halo_gate_enabled", 6, "星系 ROI 星晕门", "bool"),
    _gate("stage7_galaxy_core_preservation_ratio_min", 6, "星系核心保留下限", "float", 0.30, 0.95, 0.01, 2),
    _gate("stage7_galaxy_core_contrast_ratio_min", 6, "星系核心对比下限", "float", 0.30, 0.95, 0.01, 2),
    _gate("stage7_black_hole_score_max", 6, "暗坑评分上限", "float", 0.01, 0.35, 0.01, 2),
    _gate("stage7_starmask_contamination_max", 6, "星掩膜污染上限", "float", 0.05, 0.80, 0.01, 2),
    _gate("stage7_starless_noise_gain_max", 6, "无星图噪声增益", "float", 1.0, 2.50, 0.05, 2, suffix="×"),
    _gate("stage6_syqon_regional_texture_ratio_max", 6, "分块纹理比上限", "float", 1.20, 4.00, 0.05, 2, suffix="×"),
    _gate("stage6_syqon_regional_texture_sigma_min", 6, "分块异常显著性", "float", 3.0, 10.0, 0.5, 1, suffix=" σ"),
    _gate("stage6_syqon_regional_affected_ratio_max", 6, "分块异常覆盖上限", "float", 0.05, 0.50, 0.01, 2),
    _gate("stage7_starmask_coverage_min_ratio", 6, "星掩膜覆盖下限", "float", 0.05, 0.90, 0.01, 2),
    _gate("stage7_starmask_width_ratio_max", 6, "星掩膜宽度上限", "float", 1.10, 3.00, 0.05, 2, suffix="×"),
    _gate("stage7_starless_dynamic_range_min_ratio", 6, "无星图动态范围下限", "float", 0.20, 0.90, 0.01, 2),
    _gate("stage7_starless_peak_signal_min", 6, "无星图峰值信号下限", "float", 0.0015, 0.0300, 0.0005, 4),
    _gate("stage7_starless_peak_background_ratio_min", 6, "峰值背景比下限", "float", 1.5, 12.0, 0.5, 1, suffix="×"),
    _gate("stage7_starmask_background_floor_percentile", 6, "星掩膜背景分位", "float", 20.0, 80.0, 1.0, 1, suffix="%"),
    _gate("stage7_starmask_cleanup_noise_sigma", 6, "星掩膜噪声检测", "float", 1.0, 6.0, 0.1, 1, suffix=" σ"),
    _gate("stage7_starmask_compact_retention_min", 6, "紧致星点保留下限", "float", 0.60, 0.98, 0.01, 2),
    _gate("stage7_starmask_diffuse_residual_ratio_max", 6, "弥散残留上限", "float", 0.01, 0.50, 0.01, 2),
    _gate("stage7_starmask_diffuse_uncertainty_abs", 6, "弥散残留不确定带", "float", 0.0, 0.01, 0.0001, 4),
    _gate("stage7_starmask_diffuse_borderline_star_intensity_scale", 6, "边界状态回星上限", "float", 0.35, 1.00, 0.05, 2),
    _gate("stage7_starless_repair_max_score_growth", 6, "修复评分增长上限", "float", 0.0, 0.20, 0.01, 2),
    _gate("stage7_starless_repair_chroma_reduction_min", 6, "修复色噪下降下限", "float", 0.05, 0.80, 0.01, 2),
    _gate("stage7_starless_repair_chroma_delta_min", 6, "修复色噪绝对下降", "float", 0.00001, 0.05000, 0.0001, 5),

    # Stage 7: brightness contract, structure preservation and color/background gates.
    _gate(
        "stage7_9_quality_advisory_multiplier",
        7,
        "Stage 7-9 数值告警倍率",
        "float",
        1.0,
        2.0,
        0.05,
        2,
        suffix="×",
        help=(
            "Stage 7-9 可恢复数值门的共享告警倍率；结构性错误不受影响。"
        ),
    ),
    _gate("stage7_bg_median_min", 7, "背景中值下限", "float", 0.005, 0.080, 0.001, 3),
    _gate("stage7_black_pixel_ratio_max", 7, "近黑像素上限", "float", 0.10, 0.70, 0.01, 2),
    _gate("stage7_highlight_clip_ratio_max", 7, "高光裁剪上限", "float", 0.001, 0.050, 0.001, 3),
    _gate("stage7_star_growth_ratio_max", 7, "星状结构增长上限", "float", 1.05, 1.80, 0.05, 2, suffix="×"),
    _gate("stage7_bright_nebula_star_growth_ratio_max", 7, "亮星云星状增长", "float", 1.05, 1.80, 0.05, 2, suffix="×"),
    _gate("stage7_preview_calibration_enabled", 7, "预览亮度标定", "bool"),
    _gate("stage7_target_aware_stretch_enabled", 7, "目标感知拉伸", "bool"),
    _gate("stage7_preview_cand_a_p50_ratio", 7, "候选 A 预览 P50 比例", "float", 0.10, 0.90, 0.01, 2),
    _gate("stage7_preview_cand_b_p50_ratio", 7, "候选 B 预览 P50 比例", "float", 0.10, 0.90, 0.01, 2),
    _gate("stage7_preview_asinh_p50_max", 7, "Asinh 预览 P50 上限", "float", 0.12, 0.35, 0.01, 2),
    _gate("stage7_preview_asinh_stretch_max", 7, "预览反解 Asinh 上限", "float", 10.0, 1000.0, 10.0, 0),
    _gate("stage7_starless_linked_mtf_p50_min", 7, "Linked MTF P50 下限", "float", 0.05, 0.30, 0.01, 2),
    _gate("stage7_starless_linked_mtf_diffuse_p50_min", 7, "弥散目标 P50 下限", "float", 0.05, 0.30, 0.01, 2),
    _gate("stage7_starless_linked_mtf_preview_p50_ratio", 7, "Linked MTF 预览比例", "float", 0.40, 0.90, 0.01, 2),
    _gate("stage7_starless_linked_mtf_p50_max", 7, "Linked MTF P50 上限", "float", 0.15, 0.35, 0.01, 2),
    _gate("stage7_starless_linked_mtf_shadow_noise_sigma", 7, "Linked MTF 阴影距离", "float", 2.0, 6.0, 0.1, 1, suffix=" σ"),
    _gate("stage7_mtf_reference_blackpoint_sigma", 7, "MTF 参考黑点距离", "float", 0.50, 8.00, 0.10, 2, suffix=" σ"),
    _gate("stage7_mtf_reference_p50_relative_error_max", 7, "MTF P50 相对误差", "float", 0.01, 0.25, 0.01, 2),
    _gate("stage7_mtf_reference_p50_absolute_error_max", 7, "MTF P50 绝对误差", "float", 0.0001, 0.0300, 0.0005, 4),
    _gate("stage7_preview_target_p50_min_ratio", 7, "目标 P50 达成下限", "float", 0.25, 0.90, 0.01, 2),
    _gate("stage7_preview_target_p50_hard_min_ratio", 7, "目标 P50 硬拒绝线", "float", 0.25, 0.90, 0.01, 2),
    _gate("stage7_preview_target_p50_max_ratio", 7, "目标 P50 达成上限", "float", 1.00, 3.00, 0.05, 2),
    _gate("stage7_diffuse_visibility_score_min", 7, "主体绝对可见度", "float", 0.0, 0.20, 0.005, 3),
    _gate("stage7_preview_visibility_retention_min", 7, "预览可见度保留", "float", 0.20, 1.00, 0.01, 2),
    _gate("stage7_stretch_feedback_retry_max", 7, "亮度闭环重试", "int", 0, 1, 1),
    _gate("stage7_starless_structure_gate_enabled", 7, "无星图结构门", "bool"),
    _gate("stage7_starless_masked_rank_drift_p95_max", 7, "局部秩漂移 P95", "float", 0.02, 0.50, 0.01, 2),
    _gate("stage7_starless_halo_detail_growth_ratio_max", 7, "星周细节增长", "float", 1.05, 4.00, 0.05, 2, suffix="×"),
    _gate("stage7_starless_halo_detail_delta_min", 7, "星周细节绝对增量", "float", 0.001, 0.10, 0.001, 3),
    _gate("stage7_quantile_fallback_enabled", 7, "分位数安全回退", "bool"),
    _gate("stage7_target_local_metrics_enabled", 7, "目标局部门禁", "bool"),
    _gate("stage7_local_core_clip_ratio_max", 7, "局部核心裁剪上限", "float", 0.01, 0.30, 0.01, 2),
    _gate("stage7_local_faint_snr_min", 7, "局部暗弱信号 SNR", "float", 0.0, 2.0, 0.05, 2),
    _gate("stage7_local_dark_separation_min", 7, "局部暗云分离下限", "float", 0.0, 0.020, 0.001, 3),
    _gate("stage7_stretch_chroma_noise_score_max", 7, "拉伸色噪评分", "float", 0.10, 0.80, 0.01, 2),
    _gate("stage7_stretch_background_mottling_score_max", 7, "背景斑驳评分", "float", 0.10, 1.00, 0.01, 2),
    _gate("stage7_stretch_chroma_load_growth_max", 7, "综合色偏增长", "float", 1.00, 3.00, 0.05, 2, suffix="×"),
    _gate("stage7_stretch_chroma_load_low_absolute_max", 7, "低绝对色偏豁免", "float", 0.01, 0.15, 0.005, 3),
    _gate("stage7_stretch_chroma_load_low_absolute_tolerance", 7, "低色偏数值容差", "float", 0.0, 0.01, 0.0001, 4),
    _gate("stage7_stretch_chroma_load_signal_excluded_max", 7, "排除主体后色偏", "float", 0.01, 0.15, 0.005, 3),
    _gate("stage7_uncalibrated_background_chroma_load_review_max", 7, "未校色背景复核线", "float", 0.04, 0.50, 0.01, 2),
    _gate("stage7_chroma_rescue_enabled", 7, "背景色噪救援", "bool"),
    _gate("stage7_chroma_rescue_max_strength", 7, "背景色度救援上限", "float", 0.10, 0.90, 0.05, 2),
    _gate("stage7_transform_new_hard_clip_ratio_warn", 7, "新增硬裁切告警", "float", 0.0, 0.005, 0.0001, 4),
    _gate("stage7_transform_new_hard_clip_ratio_max", 7, "新增硬裁切上限", "float", 0.0001, 0.010, 0.0001, 4),
    _gate("stage7_transform_unexpected_zero_ratio_max", 7, "异常新增纯黑上限", "float", 0.0001, 0.020, 0.0001, 4),
    _gate("stage7_color_vector_p95_advisory_max", 7, "宽带色度漂移告警", "float", 0.01, 0.20, 0.01, 2),
    _gate("stage7_color_vector_p95_hard_max", 7, "宽带色度漂移上限", "float", 0.02, 0.30, 0.01, 2),
    _gate("stage7_narrowband_color_vector_p95_advisory_max", 7, "窄带色度漂移告警", "float", 0.02, 0.30, 0.01, 2),
    _gate("stage7_narrowband_color_vector_p95_hard_max", 7, "窄带色度漂移上限", "float", 0.04, 0.40, 0.01, 2),

    # Stage 8: enhancement-mask sufficiency and artifact rollback.
    _gate("stage8_mask_signal_coverage_min", 8, "信号蒙版覆盖下限", "float", 0.001, 0.050, 0.001, 3),
    _gate("stage8_blue_excess_max", 8, "蓝色过量上限", "float", 0.02, 0.30, 0.01, 2),
    _gate("stage8_saturation_growth_ratio_max", 8, "饱和度增长上限", "float", 1.05, 2.50, 0.05, 2, suffix="×"),
    _gate("stage8_microcontrast_growth_ratio_max", 8, "微对比增长上限", "float", 1.05, 2.80, 0.05, 2, suffix="×"),
    _gate("stage8_highlight_clip_ratio_max", 8, "高光裁剪上限", "float", 0.001, 0.060, 0.001, 3),
    _gate("stage8_bg_std_growth_max", 8, "背景噪声增长", "float", 1.00, 1.50, 0.01, 2, suffix="×"),
    _gate("stage8_texture_artifact_growth_max", 8, "纹理伪影增长", "float", 1.00, 2.20, 0.05, 2, suffix="×"),
    _gate("stage8_limited_saturation_max", 8, "受限候选饱和上限", "float", 0.0, 0.10, 0.01, 2),
    _gate("stage8_limited_core_exclusion_expand", 8, "受限候选核心扩张", "int", 2, 16, 1, suffix=" px"),
    _gate("stage8_limited_halo_texture_growth_max", 8, "受限星晕纹理增长", "float", 1.0, 1.50, 0.01, 2, suffix="×"),
    _gate("stage8_limited_halo_texture_delta_max", 8, "受限星晕绝对增量", "float", 0.00001, 0.01000, 0.0001, 5),
    _gate("stage8_dualband_palette_luma_drift_max", 8, "伪色亮度漂移", "float", 0.001, 0.030, 0.001, 3),
    _gate("stage8_dualband_palette_clip_growth_max", 8, "伪色裁剪增长", "float", 0.0, 0.020, 0.001, 3),
    _gate("stage8_dualband_palette_quality_warning_tolerance", 8, "伪色质量提醒宽限", "float", 0.0, 1.0, 0.05, 2, suffix="×"),

    # Stage 9: catalog, star-color, remix and local-structure acceptance gates.
    _gate("stage9_unscreen_denominator_floor", 9, "Unscreen 分母下限", "float", 0.02, 0.25, 0.01, 2),
    _gate("stage9_unscreen_reliable_support_min", 9, "Unscreen 可靠覆盖下限", "float", 0.50, 0.98, 0.01, 2),
    _gate("stage9_unscreen_peak_max", 9, "Unscreen 星点峰值上限", "float", 0.75, 0.98, 0.01, 2),
    _gate("stage9_unscreen_roundtrip_relative_improvement_min", 9, "Unscreen 闭环相对改善", "float", 0.0, 0.50, 0.01, 2),
    _gate("stage9_unscreen_roundtrip_absolute_improvement_min", 9, "Unscreen 闭环绝对改善", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage9_unscreen_chroma_regression_max", 9, "Unscreen 色度回退上限", "float", 0.0, 0.10, 0.005, 3),
    _gate("stage9_unscreen_recovery_regression_max", 9, "Unscreen 恢复率回退上限", "float", 0.0, 0.10, 0.005, 3),
    _gate("stage9_unscreen_wing_regression_max", 9, "Unscreen 星翼回退上限", "float", 0.0, 0.15, 0.005, 3),
    _gate("stage9_unscreen_fwhm_regression_max", 9, "Unscreen 星径回退上限", "float", 0.0, 0.25, 0.01, 2),
    _gate("stage9_psf_size_gate_enabled", 9, "同星 FWHM 正式门", "bool"),
    _gate("stage9_psf_fwhm_ratio_min", 9, "同星 FWHM 比例下限", "float", 0.50, 1.00, 0.01, 2),
    _gate("stage9_psf_fwhm_ratio_max", 9, "同星 FWHM 比例上限", "float", 1.00, 1.50, 0.01, 2),
    _gate("stage9_psf_fwhm_ratio_uncertainty_floor", 9, "FWHM 95% 容差下限", "float", 0.0, 0.01, 0.001, 3),
    _gate("stage9_psf_fwhm_ratio_uncertainty_max", 9, "FWHM 95% 容差上限", "float", 0.002, 0.05, 0.001, 3),
    _gate("stage9_psf_review_fwhm_ratio_max", 9, "复核候选 FWHM 上限", "float", 1.10, 1.65, 0.01, 2),
    _gate("stage9_psf_selective_wing_enabled", 9, "逐星低端补翼", "bool"),
    _gate("stage9_psf_selective_wing_target_ratio", 9, "逐星补翼目标比例", "float", 0.93, 1.10, 0.01, 2),
    _gate("stage9_psf_selective_wing_strength_max", 9, "逐星半高线下补翼幅度", "float", 0.90, 1.25, 0.01, 2, suffix="×"),
    _gate("stage9_source_autostretch_wing_reference_enabled", 9, "同源自动拉伸外翼参考", "bool"),
    _gate("stage9_source_autostretch_wing_floor_fraction", 9, "可见外翼峰值下限", "float", 0.03, 0.15, 0.01, 2),
    _gate("stage9_source_autostretch_wing_target_ratio", 9, "5%/10% 可见外翼目标外径比", "float", 0.90, 1.10, 0.01, 2),
    _gate("stage9_source_autostretch_wing_radius_max", 9, "可见外翼半径上限", "int", 6, 16, 1, suffix=" px"),
    _gate("stage9_psf_min_sample_count", 9, "FWHM 最少样本", "int", 4, 256, 1),
    _gate("stage9_psf_support_radius_max", 9, "FWHM 星翼支持半径", "int", 2, 12, 1, suffix=" px"),
    _gate("stage9_psf_support_retry_pixels", 9, "偏小星径补翼像素", "int", 0, 2, 1, suffix=" px"),
    _gate("stage9_stage5_bright_star_fwhm_min", 9, "非饱和大亮星 FWHM 下限", "float", 4.0, 20.0, 0.5, 1, suffix=" px"),
    _gate("stage9_stage5_bright_star_support_radius_max", 9, "大亮星支持半径上限", "int", 6, 16, 1, suffix=" px"),
    _gate("stage9_stage5_bright_star_match_radius", 9, "Stage5/9 星表判重半径", "float", 1.0, 8.0, 0.5, 1, suffix=" px"),
    _gate("stage9_star_color_support_ratio_max", 9, "星色修复覆盖上限", "float", 0.02, 0.25, 0.01, 2),
    _gate("stage9_star_color_improvement_min", 9, "星色色差改善下限", "float", 0.0, 0.10, 0.005, 3),
    _gate("stage9_star_color_post_chroma_error_max", 9, "星色后验误差上限", "float", 0.12, 0.35, 0.01, 2),
    _gate("stage9_star_color_post_validation_enabled", 9, "星色后验安全门", "bool"),
    _gate("stage9_source_star_detail_percentile", 9, "源星点细节分位", "float", 97.0, 99.5, 0.1, 1, suffix="%"),
    _gate("stage9_source_component_density_max", 9, "源组件密度上限", "float", 500.0, 10000.0, 100.0, 0),
    _gate("stage9_source_single_pixel_ratio_max", 9, "源单像素组件上限", "float", 0.10, 0.90, 0.01, 2),
    _gate("stage9_star_reference_sigma", 9, "星点参考检测", "float", 3.0, 8.0, 0.1, 1, suffix=" σ"),
    _gate("stage9_compact_weak_star_retention_min", 9, "紧致弱星保留下限", "float", 0.50, 0.98, 0.01, 2),
    _gate("stage9_mixed_star_peak_ratio_min", 9, "混合星场峰值倍率", "float", 2.0, 20.0, 0.5, 1, suffix="×"),
    _gate("stage9_mixed_star_weak_count_min", 9, "混合星场弱星下限", "int", 4, 1000, 1),
    _gate("stage9_mixed_star_bright_count_min", 9, "混合星场亮星下限", "int", 1, 100, 1),
    _gate("stage9_starmask_asinh_stretch_max", 9, "自适应 Asinh 上限", "float", 10.0, 1000.0, 10.0, 0),
    _gate("stage9_starmask_faint_target", 9, "弱星目标亮度", "float", 0.08, 0.40, 0.01, 2),
    _gate("stage9_starmask_mid_target", 9, "中亮星目标亮度", "float", 0.30, 0.70, 0.01, 2),
    _gate("stage9_starmask_bright_target", 9, "亮星目标亮度", "float", 0.50, 0.88, 0.01, 2),
    _gate("stage9_starmask_peak_target", 9, "极亮星目标上限", "float", 0.75, 0.95, 0.01, 2),
    _gate("stage9_starmask_output_adequacy_min", 9, "星层输出充分度下限", "float", 0.25, 0.90, 0.01, 2),
    _gate("stage9_starmask_chroma_regularization_enabled", 9, "星掩膜色度正则", "bool"),
    _gate("stage9_starmask_faint_chroma_max", 9, "弱星色度跨度", "float", 0.10, 0.80, 0.01, 2),
    _gate("stage9_starmask_bright_chroma_max", 9, "亮星色度跨度", "float", 0.10, 0.90, 0.01, 2),
    _gate("stage9_starmask_predicted_change_ratio_max", 9, "预测变化覆盖上限", "float", 0.05, 0.60, 0.01, 2),
    _gate("stage9_quality_gate_enabled", 9, "回星质量总门", "bool"),
    _gate("stage9_highlight_clip_ratio_max", 9, "高光裁剪占比", "float", 0.001, 0.10, 0.001, 3),
    _gate("stage9_highlight_clip_growth_max", 9, "高光裁剪增长", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage9_bright_pixel_growth_max", 9, "亮像素增长", "float", 0.0, 0.10, 0.001, 3),
    _gate("stage9_background_lift_max", 9, "背景抬升上限", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage9_background_mottling_growth_max", 9, "背景斑驳增长", "float", 1.0, 3.0, 0.05, 2, suffix="×"),
    _gate("stage9_mottling_exemption_changed_pixel_ratio_max", 9, "斑驳豁免变化覆盖", "float", 0.02, 0.35, 0.01, 2),
    _gate("stage9_changed_pixel_ratio_max", 9, "显著变化像素上限", "float", 0.05, 0.80, 0.01, 2),
    _gate("stage9_darkening_ratio_max", 9, "异常变暗像素上限", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage9_star_recovery_ratio_min", 9, "全部星点恢复下限", "float", 0.40, 0.98, 0.01, 2),
    _gate("stage9_catalog_star_visibility_contrast_min", 9, "目录星可见对比度下限", "float", 0.0005, 0.02, 0.0005, 4),
    _gate("stage9_bright_star_visibility_ratio_min", 9, "亮星可见率下限", "float", 0.50, 1.00, 0.01, 2),
    _gate("stage9_weak_star_screen_intensity_min", 9, "弱星 Screen 强度下限", "float", 0.10, 1.05, 0.05, 2),
    _gate("stage9_star_support_ratio_max", 9, "实际回星覆盖上限", "float", 0.03, 0.20, 0.01, 2),
    _gate("stage9_unmatched_changed_ratio_max", 9, "支持层外变化上限", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage9_chromatic_addition_peak_min", 9, "伪色新增峰值下限", "float", 0.002, 0.25, 0.001, 3),
    _gate("stage9_chromatic_addition_saturation_min", 9, "伪色新增饱和下限", "float", 0.30, 0.95, 0.01, 2),
    _gate("stage9_chromatic_addition_ratio_max", 9, "伪色新增占比上限", "float", 0.0, 0.05, 0.001, 3),
    _gate("stage9_local_component_peak_min", 9, "局部组件峰值下限", "float", 0.002, 0.10, 0.001, 3),
    _gate("stage9_local_component_area_max", 9, "局部组件面积上限", "int", 16, 4096, 16, suffix=" px"),
    _gate("stage9_local_component_aspect_ratio_max", 9, "局部组件长宽比", "float", 1.2, 10.0, 0.1, 1, suffix="×"),
    _gate("stage9_local_component_fill_ratio_min", 9, "局部组件填充下限", "float", 0.02, 0.80, 0.01, 2),
    _gate("stage9_local_single_pixel_ratio_max", 9, "局部单像素组件上限", "float", 0.0, 0.90, 0.01, 2),
    _gate("stage9_local_cyan_blue_peak_min", 9, "青蓝团块峰值下限", "float", 0.002, 0.10, 0.001, 3),
    _gate("stage9_local_cyan_blue_saturation_min", 9, "青蓝团块饱和下限", "float", 0.20, 0.95, 0.01, 2),
    _gate("stage9_local_cyan_blue_component_area_max", 9, "青蓝团块面积上限", "int", 4, 2048, 4, suffix=" px"),
    _gate("stage9_core_percentile", 9, "星云核心定义分位", "float", 70.0, 99.0, 0.5, 1, suffix="%"),
    _gate("stage9_core_color_jump_min", 9, "核心颜色突变下限", "float", 0.03, 0.50, 0.01, 2),
    _gate("stage9_core_color_jump_component_area_max", 9, "核心突变面积上限", "int", 4, 2048, 4, suffix=" px"),
    _gate("stage9_star_positive_delta_window_recovery_ratio_min", 9, "7×7 正增量窗口恢复下限", "float", 0.40, 0.98, 0.01, 2),
    _gate("stage9_star_wing_recovery_ratio_min", 9, "星翼恢复下限", "float", 0.30, 0.95, 0.01, 2),
    _gate("stage9_residual_dark_hole_ratio_max", 9, "残余暗坑占比上限", "float", 0.0, 0.50, 0.01, 2),
    _gate("stage9_hollow_structure_delta_min", 9, "空心结构变化下限", "float", 0.01, 0.25, 0.01, 2),
    _gate("stage9_new_hollow_structure_area_max", 9, "新增空心面积上限", "int", 4, 4096, 4, suffix=" px"),
)

# A gate joins the task-wide profile only after its relaxation direction and
# physical domain have been reviewed.  Counts, retries, detector tuning,
# algorithm targets, booleans and structural invariants intentionally remain
# absent.  This explicit allow-list avoids inferring safety behavior from field
# suffixes such as ``_min_ratio``.
_GATE_PROFILE_RULES: Dict[
    str,
    Tuple[str, float | int | None, float | int | None],
] = {
    # Stage 2: residual edge acceptance only; crop mechanics remain unchanged.
    "stage2_edge_black_target": (PROFILE_SCALE_UPPER, 0.0, 1.0),

    # Stage 3: candidate acceptance after the immutable sample audit.
    "stage3_apply_confidence_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage3_compound_score_abs_improvement_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage3_compound_score_rel_improvement_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage3_compound_validation_improvement_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage3_compound_zero_point_abs_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage3_compound_zero_point_rel_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage3_pattern_score_growth_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),

    # Stage 4: geometry/color candidate acceptance; mapping identity and local
    # sample construction stay on their static safe behavior.
    "stage4_auto_geometry_confidence_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage4_auto_geometry_scale_residual_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage4_nbn_line_ratio_drift_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage4_pcc_channel_gain_ratio_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage4_pcc_emission_balance_gain_ratio_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage4_pcc_clip_growth_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage4_pcc_star_temperature_ratio_min": (PROFILE_SCALE_LOWER, 0.0, None),
    "stage4_pcc_star_temperature_ratio_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage4_pcc_background_color_delta_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage4_pcc_target_color_drift_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage4_pcc_emission_target_color_drift_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage4_local_star_mask_coverage_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),

    # Stage 5: measured detail/noise acceptance; denoise strength stays fixed.
    "stage5_multiscale_detail_retention_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage5_multiscale_noise_reduction_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage5_denoise_chroma_noise_growth_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),

    # Stage 6: accepted starless/starmask quality.  Cleanup targets and repair
    # strengths are algorithm settings and therefore do not join the profile.
    "stage7_edge_black_warn": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage7_edge_black_high": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage7_bg_median_high": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage7_bg_std_high": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage7_bg_noise_ratio_high": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage7_residual_star_score_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage7_halo_residue_score_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage7_large_galaxy_halo_residue_score_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage7_bright_nebula_halo_residue_score_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage7_galaxy_core_preservation_ratio_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_galaxy_core_contrast_ratio_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_black_hole_score_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage7_starmask_contamination_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage7_starless_noise_gain_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage7_starmask_coverage_min_ratio": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_starmask_width_ratio_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage7_starless_dynamic_range_min_ratio": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_starless_peak_signal_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_starless_peak_background_ratio_min": (PROFILE_SCALE_LOWER, 0.0, None),
    "stage7_starmask_compact_retention_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_starmask_diffuse_residual_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage7_starless_repair_max_score_growth": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage7_starless_repair_chroma_reduction_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_starless_repair_chroma_delta_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),

    # Stage 7: only presentation-attainment gates follow the global profile.
    # Clipping, structure, LUT conformance and colour-safety limits stay fixed;
    # vivid_safe changes candidate search rather than weakening those limits.
    "stage7_bg_median_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_preview_target_p50_min_ratio": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_preview_target_p50_hard_min_ratio": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_preview_target_p50_max_ratio": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage7_diffuse_visibility_score_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_preview_visibility_retention_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage7_local_faint_snr_min": (PROFILE_SCALE_LOWER, 0.0, None),
    "stage7_local_dark_separation_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),

    # Stage 8: post-enhancement quality acceptance; limited-mode strength caps
    # and mask construction remain static algorithm settings.
    "stage8_mask_signal_coverage_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage8_blue_excess_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage8_saturation_growth_ratio_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage8_microcontrast_growth_ratio_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage8_highlight_clip_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage8_bg_std_growth_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage8_texture_artifact_growth_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage8_limited_halo_texture_growth_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage8_limited_halo_texture_delta_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage8_dualband_palette_luma_drift_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage8_dualband_palette_clip_growth_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),

    # Stage 9: final accepted-with-stars quality.  Artifact detector seeds,
    # mask targets and morphology descriptors stay at static defaults.
    "stage9_star_color_support_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_star_color_improvement_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage9_star_color_post_chroma_error_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_source_component_density_max": (PROFILE_SCALE_UPPER, 0.0, None),
    "stage9_source_single_pixel_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_compact_weak_star_retention_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage9_psf_fwhm_ratio_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage9_psf_fwhm_ratio_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage9_unscreen_fwhm_regression_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_mixed_star_peak_ratio_min": (PROFILE_SCALE_LOWER, 0.0, None),
    "stage9_mixed_star_weak_count_min": (PROFILE_SCALE_LOWER, 1, None),
    "stage9_mixed_star_bright_count_min": (PROFILE_SCALE_LOWER, 1, None),
    "stage9_highlight_clip_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_highlight_clip_growth_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_bright_pixel_growth_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_background_lift_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_background_mottling_growth_max": (PROFILE_SCALE_UPPER_FROM_ONE, 1.0, None),
    "stage9_mottling_exemption_changed_pixel_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_changed_pixel_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_darkening_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_star_recovery_ratio_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage9_weak_star_screen_intensity_min": (PROFILE_SCALE_LOWER, 0.0, None),
    "stage9_star_support_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_unmatched_changed_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_chromatic_addition_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
    "stage9_star_positive_delta_window_recovery_ratio_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage9_star_wing_recovery_ratio_min": (PROFILE_SCALE_LOWER, 0.0, 1.0),
    "stage9_residual_dark_hole_ratio_max": (PROFILE_SCALE_UPPER, 0.0, 1.0),
}

# Every remaining production gate is explicitly reviewed out of profile
# scaling.  Keeping this separate from the active rule map makes a newly added
# gate fail registry initialization until its profile behavior is decided.
_GATE_PROFILE_EXCLUDED_FIELDS = frozenset(
    {
        "stage2_adaptive_edge_crop_max_passes",
        "stage2_adaptive_edge_crop_max_extra",
        "stage2_guard_band_pixels",
        "bg_quality_gate_enabled",
        "stage3_conditional_decision_enabled",
        "stage3_deterministic_auto_apply_enabled",
        "stage3_safe_sample_target_count",
        "stage3_safe_sample_min_count",
        "stage3_safe_sample_patch_radius",
        "stage3_safe_sample_brightness_quantile_max",
        "stage3_safe_sample_texture_quantile_max",
        "stage3_compound_min_sample_count",
        "stage3_compound_fit_min_count",
        "stage3_compound_validation_min_count",
        "stage3_compound_validation_ratio",
        "stage3_pattern_routing_enabled",
        "stage3_pattern_score_min",
        "stage3_walking_noise_score_min",
        "stage4_auto_reference_background_sample_target",
        "stage4_auto_reference_background_sample_min",
        "stage4_auto_reference_holdout_ratio",
        "stage4_auto_reference_background_error_min",
        "stage4_auto_reference_background_improvement_min",
        "stage4_auto_reference_star_min_objects",
        "stage4_auto_reference_star_ratio_mad_max",
        "stage4_auto_reference_star_saturation_ratio_max",
        "stage4_auto_reference_gain_limit",
        "stage4_auto_reference_star_improvement_min",
        "stage4_auto_reference_highlight_clip_growth_max",
        "stage4_auto_reference_black_clip_growth_max",
        "stage4_auto_reference_gradient_growth_max",
        "stage4_auto_reference_texture_growth_max",
        "stage4_auto_reference_target_chroma_drift_max",
        "stage4_nbn_mapping_confidence_min",
        "stage4_nbn_gain_limit",
        "stage4_pcc_quality_gate_enabled",
        "stage4_local_star_wb_min_pixels",
        "stage4_local_star_mask_radius",
        "denoise_safety_max",
        "stage7_galaxy_roi_halo_gate_enabled",
        "stage6_syqon_regional_texture_ratio_max",
        "stage6_syqon_regional_texture_sigma_min",
        "stage6_syqon_regional_affected_ratio_max",
        "stage7_starmask_background_floor_percentile",
        "stage7_starmask_cleanup_noise_sigma",
        "stage7_starmask_diffuse_uncertainty_abs",
        "stage7_starmask_diffuse_borderline_star_intensity_scale",
        "stage7_9_quality_advisory_multiplier",
        "stage7_preview_calibration_enabled",
        "stage7_target_aware_stretch_enabled",
        "stage7_preview_cand_a_p50_ratio",
        "stage7_preview_cand_b_p50_ratio",
        "stage7_preview_asinh_p50_max",
        "stage7_preview_asinh_stretch_max",
        "stage7_black_pixel_ratio_max",
        "stage7_highlight_clip_ratio_max",
        "stage7_star_growth_ratio_max",
        "stage7_bright_nebula_star_growth_ratio_max",
        "stage7_starless_linked_mtf_p50_min",
        "stage7_starless_linked_mtf_diffuse_p50_min",
        "stage7_starless_linked_mtf_preview_p50_ratio",
        "stage7_starless_linked_mtf_p50_max",
        "stage7_starless_linked_mtf_shadow_noise_sigma",
        "stage7_mtf_reference_blackpoint_sigma",
        "stage7_mtf_reference_p50_relative_error_max",
        "stage7_mtf_reference_p50_absolute_error_max",
        "stage7_stretch_feedback_retry_max",
        "stage7_starless_structure_gate_enabled",
        "stage7_starless_masked_rank_drift_p95_max",
        "stage7_starless_halo_detail_growth_ratio_max",
        "stage7_starless_halo_detail_delta_min",
        "stage7_quantile_fallback_enabled",
        "stage7_target_local_metrics_enabled",
        "stage7_local_core_clip_ratio_max",
        "stage7_stretch_chroma_noise_score_max",
        "stage7_stretch_background_mottling_score_max",
        "stage7_stretch_chroma_load_growth_max",
        "stage7_stretch_chroma_load_low_absolute_max",
        "stage7_stretch_chroma_load_low_absolute_tolerance",
        "stage7_stretch_chroma_load_signal_excluded_max",
        "stage7_uncalibrated_background_chroma_load_review_max",
        "stage7_chroma_rescue_enabled",
        "stage7_chroma_rescue_max_strength",
        "stage7_transform_new_hard_clip_ratio_warn",
        "stage7_transform_new_hard_clip_ratio_max",
        "stage7_transform_unexpected_zero_ratio_max",
        "stage7_color_vector_p95_advisory_max",
        "stage7_color_vector_p95_hard_max",
        "stage7_narrowband_color_vector_p95_advisory_max",
        "stage7_narrowband_color_vector_p95_hard_max",
        "stage8_limited_saturation_max",
        "stage8_limited_core_exclusion_expand",
        "stage8_dualband_palette_quality_warning_tolerance",
        "stage9_unscreen_denominator_floor",
        "stage9_unscreen_reliable_support_min",
        "stage9_unscreen_peak_max",
        "stage9_unscreen_roundtrip_relative_improvement_min",
        "stage9_unscreen_roundtrip_absolute_improvement_min",
        "stage9_unscreen_chroma_regression_max",
        "stage9_unscreen_recovery_regression_max",
        "stage9_unscreen_wing_regression_max",
        "stage9_psf_size_gate_enabled",
        "stage9_psf_fwhm_ratio_uncertainty_floor",
        "stage9_psf_fwhm_ratio_uncertainty_max",
        "stage9_psf_review_fwhm_ratio_max",
        "stage9_psf_selective_wing_enabled",
        "stage9_psf_selective_wing_target_ratio",
        "stage9_psf_selective_wing_strength_max",
        "stage9_source_autostretch_wing_reference_enabled",
        "stage9_source_autostretch_wing_floor_fraction",
        "stage9_source_autostretch_wing_target_ratio",
        "stage9_source_autostretch_wing_radius_max",
        "stage9_psf_min_sample_count",
        "stage9_psf_support_radius_max",
        "stage9_psf_support_retry_pixels",
        "stage9_stage5_bright_star_fwhm_min",
        "stage9_stage5_bright_star_support_radius_max",
        "stage9_stage5_bright_star_match_radius",
        "stage9_star_color_post_validation_enabled",
        "stage9_source_star_detail_percentile",
        "stage9_star_reference_sigma",
        "stage9_starmask_asinh_stretch_max",
        "stage9_starmask_faint_target",
        "stage9_starmask_mid_target",
        "stage9_starmask_bright_target",
        "stage9_starmask_peak_target",
        "stage9_starmask_output_adequacy_min",
        "stage9_starmask_chroma_regularization_enabled",
        "stage9_starmask_faint_chroma_max",
        "stage9_starmask_bright_chroma_max",
        "stage9_starmask_predicted_change_ratio_max",
        "stage9_quality_gate_enabled",
        "stage9_chromatic_addition_peak_min",
        "stage9_chromatic_addition_saturation_min",
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
        "stage9_hollow_structure_delta_min",
        "stage9_new_hollow_structure_area_max",
        "stage9_catalog_star_visibility_contrast_min",
        "stage9_bright_star_visibility_ratio_min",
    }
)

_gate_specs_by_field = {spec.field: spec for spec in PROCESSING_GATE_PARAMETER_SPECS}
_unknown_profile_fields = set(_GATE_PROFILE_RULES) - set(_gate_specs_by_field)
if _unknown_profile_fields:
    raise RuntimeError(
        "门禁档位注册包含未知字段：" + ", ".join(sorted(_unknown_profile_fields))
    )
_unknown_excluded_fields = _GATE_PROFILE_EXCLUDED_FIELDS - set(
    _gate_specs_by_field
)
if _unknown_excluded_fields:
    raise RuntimeError(
        "门禁档位排除表包含未知字段："
        + ", ".join(sorted(_unknown_excluded_fields))
    )
_profile_rule_overlap = set(_GATE_PROFILE_RULES) & _GATE_PROFILE_EXCLUDED_FIELDS
if _profile_rule_overlap:
    raise RuntimeError(
        "门禁档位字段同时登记为缩放和排除："
        + ", ".join(sorted(_profile_rule_overlap))
    )
_production_gate_fields = {
    spec.field for spec in PROCESSING_GATE_PARAMETER_SPECS
}
_unclassified_profile_fields = _production_gate_fields - (
    set(_GATE_PROFILE_RULES) | _GATE_PROFILE_EXCLUDED_FIELDS
)
if _unclassified_profile_fields:
    raise RuntimeError(
        "门禁档位字段尚未显式分类："
        + ", ".join(sorted(_unclassified_profile_fields))
    )
for _field, (_scaling, _physical_minimum, _physical_maximum) in (
    _GATE_PROFILE_RULES.items()
):
    _spec = _gate_specs_by_field[_field]
    if _spec.kind not in {"int", "float"}:
        raise RuntimeError(f"门禁档位字段必须是数值：{_field}")
    if _scaling not in {
        PROFILE_SCALE_UPPER,
        PROFILE_SCALE_LOWER,
        PROFILE_SCALE_UPPER_FROM_ONE,
    }:
        raise RuntimeError(f"门禁档位缩放方向无效：{_field}={_scaling}")
    if (
        _physical_minimum is not None
        and _physical_maximum is not None
        and _physical_minimum > _physical_maximum
    ):
        raise RuntimeError(f"门禁档位物理域无效：{_field}")
    if _scaling == PROFILE_SCALE_UPPER_FROM_ONE and float(_spec.default) < 1.0:
        raise RuntimeError(f"以 1 为基线的门禁默认值小于 1：{_field}")

PROCESSING_GATE_PARAMETER_SPECS = tuple(
    replace(
        spec,
        profile_scaling=_GATE_PROFILE_RULES[spec.field][0],
        profile_minimum=_GATE_PROFILE_RULES[spec.field][1],
        profile_maximum=_GATE_PROFILE_RULES[spec.field][2],
    )
    if spec.field in _GATE_PROFILE_RULES
    else spec
    for spec in PROCESSING_GATE_PARAMETER_SPECS
)
GATE_PROFILE_PARAMETER_SPECS: Tuple[ParameterSpec, ...] = tuple(
    spec
    for spec in PROCESSING_GATE_PARAMETER_SPECS
    if spec.profile_scaling != PROFILE_SCALE_NONE
)

PROCESSING_PARAMETER_SPECS += PROCESSING_GATE_PARAMETER_SPECS

_registered_fields = tuple(spec.field for spec in PROCESSING_PARAMETER_SPECS)
if len(_registered_fields) != len(set(_registered_fields)):
    duplicates = sorted(
        field for field in set(_registered_fields) if _registered_fields.count(field) > 1
    )
    raise RuntimeError("处理参数注册表包含重复字段：" + ", ".join(duplicates))

SPECS_BY_FIELD = {spec.field: spec for spec in PROCESSING_PARAMETER_SPECS}
SPECS_BY_STAGE = {
    stage: tuple(spec for spec in PROCESSING_PARAMETER_SPECS if spec.stage == stage)
    for stage in STAGE_TITLES
}
MODE_SPEC_BY_STAGE = {
    spec.stage: spec for spec in PROCESSING_PARAMETER_SPECS if spec.stage_mode
}
STAGE7_MANUAL_PARAMETER_FIELDS = frozenset(
    {"asinh_stretch", "asinh_offset", "ghs_shadowsclip", "ghs_stretchamount"}
)


def default_processing_parameters(
    *, general: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    merged_general = dict(GENERAL_DEFAULTS)
    if general:
        merged_general.update(dict(general))
    normalized_general = _normalize_general(merged_general)
    return {
        "schema": PROCESSING_PARAMETERS_SCHEMA,
        "gate_profile": GATE_PROFILE_DEFAULT,
        "general": normalized_general,
        "stages": {
            str(stage): {
                "mode": (
                    str(MODE_SPEC_BY_STAGE[stage].default)
                    if stage in MODE_SPEC_BY_STAGE
                    else "auto"
                ),
                "overrides": {},
            }
            for stage in STAGE_TITLES
        },
    }


def _normalize_general(
    raw: Mapping[str, Any],
    *,
    adjustments: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    unexpected = set(raw) - GENERAL_KEYS
    if unexpected:
        raise ValueError(
            "通用处理参数包含未知字段：" + ", ".join(sorted(unexpected))
        )
    formats_raw = raw.get("output_formats", GENERAL_DEFAULTS["output_formats"])
    if isinstance(formats_raw, str):
        formats_raw = [part.strip() for part in formats_raw.split(",")]
    if not isinstance(formats_raw, Sequence):
        raise ValueError("output_formats 必须是格式列表")
    formats: list[str] = []
    for value in formats_raw:
        token = str(value).strip().lower()
        if token not in {"tif", "png", "fit"}:
            raise ValueError(f"不支持的输出格式：{value!r}")
        if token not in formats:
            formats.append(token)
    if not formats:
        raise ValueError("至少选择一种输出格式")
    review_only = raw.get("review_only", False)
    if not isinstance(review_only, bool):
        raise ValueError("review_only 必须是布尔值")
    compute_mode = str(raw.get("compute_mode", "auto") or "").strip().lower()
    if compute_mode not in {"auto", "cpu"}:
        raise ValueError("compute_mode 只允许 auto 或 cpu")
    auto_tune_enabled = raw.get("auto_tune_enabled", True)
    review_bundle_enabled = raw.get("review_bundle_enabled", True)
    managed_output_enabled = raw.get("managed_output_enabled", True)
    checkpoint_mode = raw.get("checkpoint_mode", False)
    for field, value in (
        ("auto_tune_enabled", auto_tune_enabled),
        ("review_bundle_enabled", review_bundle_enabled),
        ("managed_output_enabled", managed_output_enabled),
        ("checkpoint_mode", checkpoint_mode),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{field} 必须是布尔值")

    max_retries_raw = raw.get("max_retries", 2)
    if isinstance(max_retries_raw, bool):
        raise ValueError("max_retries 必须是整数")
    try:
        max_retries_requested = int(max_retries_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_retries 必须是整数") from exc
    if isinstance(max_retries_raw, float) and not max_retries_raw.is_integer():
        raise ValueError("max_retries 必须是整数")
    max_retries = min(3, max(0, max_retries_requested))

    retry_delay_raw = raw.get("retry_delay", 1.0)
    if isinstance(retry_delay_raw, bool):
        raise ValueError("retry_delay 必须是数值")
    try:
        retry_delay_requested = float(retry_delay_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("retry_delay 必须是数值") from exc
    if not math.isfinite(retry_delay_requested):
        raise ValueError("retry_delay 必须是有限数值")
    retry_delay = min(10.0, max(0.0, retry_delay_requested))
    if adjustments is not None:
        for field, requested, effective in (
            ("max_retries", max_retries_requested, max_retries),
            ("retry_delay", retry_delay_requested, retry_delay),
        ):
            if requested != effective:
                adjustments.append(
                    {
                        "stage": 0,
                        "field": field,
                        "requested": requested,
                        "effective": effective,
                        "reason": "safe_clamp",
                    }
                )
    return {
        "output_formats": formats,
        "review_only": review_only,
        "compute_mode": compute_mode,
        "auto_tune_enabled": auto_tune_enabled,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
        "review_bundle_enabled": review_bundle_enabled,
        "managed_output_enabled": managed_output_enabled,
        "checkpoint_mode": checkpoint_mode,
    }


def _coerce_value(
    spec: ParameterSpec,
    raw: Any,
    *,
    validate_paths: bool,
) -> Tuple[Any, list[Dict[str, Any]]]:
    adjustments: list[Dict[str, Any]] = []
    if spec.kind == "bool":
        if isinstance(raw, bool):
            value: Any = raw
        elif isinstance(raw, str) and raw.strip().lower() in {"1", "true", "yes", "on"}:
            value = True
        elif isinstance(raw, str) and raw.strip().lower() in {"0", "false", "no", "off"}:
            value = False
        else:
            raise ValueError(f"{spec.field} 必须是布尔值")
    elif spec.kind == "int":
        if isinstance(raw, bool):
            raise ValueError(f"{spec.field} 必须是整数")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{spec.field} 必须是整数") from error
        if not numeric.is_integer():
            raise ValueError(f"{spec.field} 必须是整数")
        value = int(numeric)
    elif spec.kind == "float":
        if isinstance(raw, bool):
            raise ValueError(f"{spec.field} 必须是数值")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{spec.field} 必须是数值") from error
        if not math.isfinite(value):
            raise ValueError(f"{spec.field} 必须是有限数值")
    elif spec.kind == "choice":
        allowed = {choice_value for _label, choice_value in spec.choices}
        value = str(raw or "").strip()
        if spec.field == "stage8_dualband_palette_selection":
            value = "auto" if value.casefold() == "auto" else value.upper()
        if value not in allowed:
            raise ValueError(f"{spec.field} 的选项不受支持：{value!r}")
    elif spec.kind == "path":
        value = str(raw or "").strip()
        if value and validate_paths:
            path = Path(value).expanduser()
            if not path.is_file() or not os.access(path, os.R_OK):
                raise ValueError(f"{spec.label}不是可读的本地文件：{path}")
            value = str(path.resolve())
    else:
        raise ValueError(f"不支持的参数类型：{spec.kind}")

    original = value
    if spec.minimum is not None and value < spec.minimum:
        value = type(value)(spec.minimum)
    if spec.maximum is not None and value > spec.maximum:
        value = type(value)(spec.maximum)
    if spec.odd and int(value) % 2 == 0:
        candidate = int(value) + 1
        if spec.maximum is not None and candidate > int(spec.maximum):
            candidate = int(value) - 1
        value = candidate
    if value != original:
        adjustments.append(
            {
                "field": spec.field,
                "requested": original,
                "effective": value,
                "reason": "safe_clamp" if not spec.odd else "safe_clamp_or_odd",
            }
        )
    return value, adjustments


def normalize_processing_parameters(
    raw: Mapping[str, Any] | None,
    *,
    validate_paths: bool = False,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Validate a frozen processing payload and return a canonical copy."""

    if raw is None:
        return default_processing_parameters(), []
    if not isinstance(raw, Mapping):
        raise ValueError("处理参数必须是映射")
    unexpected_top = set(raw) - {"schema", "gate_profile", "general", "stages"}
    if unexpected_top:
        raise ValueError("处理参数包含未知顶层字段：" + ", ".join(sorted(unexpected_top)))
    raw_schema = str(raw.get("schema") or "")
    if raw_schema not in SUPPORTED_PROCESSING_PARAMETERS_SCHEMAS:
        raise ValueError(
            "不支持的处理参数 schema："
            f"{raw.get('schema')!r}；仅接受 starun.processing-parameters.v4/v5"
        )
    gate_profile = str(
        raw.get("gate_profile", GATE_PROFILE_DEFAULT) or GATE_PROFILE_DEFAULT
    ).strip().lower()
    if gate_profile not in GATE_PROFILE_MULTIPLIERS:
        raise ValueError(f"不支持的门禁档位：{gate_profile!r}")
    general_raw = raw.get("general", {})
    if not isinstance(general_raw, Mapping):
        raise ValueError("general 必须是映射")
    unexpected_general = set(general_raw) - GENERAL_KEYS
    if unexpected_general:
        raise ValueError(
            "通用处理参数包含未知字段：" + ", ".join(sorted(unexpected_general))
        )
    adjustments: list[Dict[str, Any]] = []
    general = dict(GENERAL_DEFAULTS)
    general.update(dict(general_raw))
    normalized = default_processing_parameters()
    normalized["general"] = _normalize_general(general, adjustments=adjustments)
    normalized["gate_profile"] = gate_profile
    stages_raw = raw.get("stages", {})
    if not isinstance(stages_raw, Mapping):
        raise ValueError("stages 必须是映射")
    allowed_stage_keys = set(normalized["stages"])
    unexpected_stages = set(str(key) for key in stages_raw) - allowed_stage_keys
    if unexpected_stages:
        raise ValueError("处理参数包含未知阶段：" + ", ".join(sorted(unexpected_stages)))
    for stage in STAGE_TITLES:
        raw_entry = stages_raw.get(str(stage), stages_raw.get(stage, {}))
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Stage {stage} 参数必须是映射")
        unexpected_entry = set(raw_entry) - {"mode", "overrides"}
        if unexpected_entry:
            raise ValueError(
                f"Stage {stage} 包含未知字段：" + ", ".join(sorted(unexpected_entry))
            )
        mode = str(raw_entry.get("mode", "auto") or "auto").strip().lower()
        mode_spec = MODE_SPEC_BY_STAGE.get(stage)
        allowed_modes = (
            {str(value) for _label, value in mode_spec.choices}
            if mode_spec is not None
            else {"auto"}
        )
        if mode not in allowed_modes:
            raise ValueError(f"Stage {stage} 不支持处理方式 {mode!r}")
        overrides_source = raw_entry.get("overrides", {})
        if not isinstance(overrides_source, Mapping):
            raise ValueError(f"Stage {stage} overrides 必须是映射")
        overrides_raw = dict(overrides_source)
        allowed_fields = {
            spec.field
            for spec in SPECS_BY_STAGE[stage]
            if not spec.stage_mode
        }
        unexpected_fields = set(overrides_raw) - allowed_fields
        if unexpected_fields:
            raise ValueError(
                f"Stage {stage} 包含未知参数：" + ", ".join(sorted(unexpected_fields))
            )
        clean_overrides: Dict[str, Any] = {}
        for field, value in overrides_raw.items():
            clean_value, field_adjustments = _coerce_value(
                SPECS_BY_FIELD[field], value, validate_paths=validate_paths
            )
            clean_overrides[field] = clean_value
            for record in field_adjustments:
                adjustments.append({"stage": stage, **record})
        if (
            raw_schema == LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4
            and stage == 9
            and "stage9_compact_starmask_enabled" in clean_overrides
            and "stage9_starmask_pre_stretch_compact_enabled"
            not in clean_overrides
        ):
            legacy_value = bool(
                clean_overrides["stage9_compact_starmask_enabled"]
            )
            clean_overrides[
                "stage9_starmask_pre_stretch_compact_enabled"
            ] = legacy_value
            adjustments.append(
                {
                    "stage": 9,
                    "field": "stage9_starmask_pre_stretch_compact_enabled",
                    "requested": None,
                    "effective": legacy_value,
                    "reason": "v4_explicit_compact_starmask_migration",
                }
            )
        if (
            stage == 7
            and STAGE7_MANUAL_PARAMETER_FIELDS.intersection(clean_overrides)
            and mode == "auto"
        ):
            adjustments.append(
                {
                    "stage": 7,
                    "field": "stage7_processing_mode",
                    "requested": "auto",
                    "effective": "manual",
                    "reason": "manual_overrides_require_manual_mode",
                }
            )
            mode = "manual"
        if (
            stage == 7
            and mode == "manual"
            and clean_overrides.get(
                "stage7_candidate_policy",
                PipelineConfig().stage7_candidate_policy,
            )
            == "display90_only"
        ):
            raise ValueError(
                "Stage 7 manual 参数方式不能与 display90_only 候选策略同时使用"
            )
        normalized["stages"][str(stage)] = {
            "mode": mode,
            "overrides": clean_overrides,
        }
    return normalized, adjustments


def gate_profile_requires_review(profile: str) -> bool:
    """Return whether a task profile must be exported as review-only."""

    return str(profile or "").strip().lower() == GATE_PROFILE_UNLIMITED


def _scaled_gate_profile_value(
    spec: ParameterSpec,
    profile: str,
) -> Tuple[Any, Any, bool]:
    """Return (requested, physical-domain value, was_clamped)."""

    if spec.profile_scaling == PROFILE_SCALE_NONE:
        return spec.default, spec.default, False
    multiplier = float(GATE_PROFILE_MULTIPLIERS[profile])
    baseline = spec.default
    if spec.profile_scaling == PROFILE_SCALE_UPPER:
        raw_value = float(baseline) * multiplier
        requested = math.ceil(raw_value) if spec.kind == "int" else raw_value
    elif spec.profile_scaling == PROFILE_SCALE_LOWER:
        raw_value = float(baseline) / multiplier
        requested = math.floor(raw_value) if spec.kind == "int" else raw_value
    elif spec.profile_scaling == PROFILE_SCALE_UPPER_FROM_ONE:
        raw_value = 1.0 + (float(baseline) - 1.0) * multiplier
        requested = math.ceil(raw_value) if spec.kind == "int" else raw_value
    else:  # Guard explicit registry metadata even if constructed externally.
        raise ValueError(
            f"不支持的门禁档位缩放方向：{spec.field}={spec.profile_scaling}"
        )

    effective = requested
    if spec.profile_minimum is not None:
        effective = max(effective, spec.profile_minimum)
    if spec.profile_maximum is not None:
        effective = min(effective, spec.profile_maximum)
    if spec.kind == "int":
        effective = int(effective)
    else:
        effective = float(effective)
    return requested, effective, effective != requested


def _gate_profile_audit_from_normalized(
    normalized: Mapping[str, Any],
) -> Dict[str, Any]:
    profile = str(normalized.get("gate_profile") or GATE_PROFILE_DEFAULT)
    multiplier = float(GATE_PROFILE_MULTIPLIERS[profile])
    stages = normalized["stages"]
    fields: list[Dict[str, Any]] = []
    for spec in GATE_PROFILE_PARAMETER_SPECS:
        requested, profile_value, was_clamped = _scaled_gate_profile_value(
            spec,
            profile,
        )
        overrides = stages[str(spec.stage)]["overrides"]
        has_expert_override = spec.field in overrides
        effective_value = (
            overrides[spec.field] if has_expert_override else profile_value
        )
        fields.append(
            {
                "stage": spec.stage,
                "field": spec.field,
                "label": spec.label,
                "scaling": spec.profile_scaling,
                "static_baseline": spec.default,
                "profile_requested": requested,
                "profile_effective": profile_value,
                "physical_minimum": spec.profile_minimum,
                "physical_maximum": spec.profile_maximum,
                "physical_clamped": was_clamped,
                "source": "expert_override" if has_expert_override else "gate_profile",
                "effective": effective_value,
            }
        )
    return {
        "profile": profile,
        "label": GATE_PROFILE_LABELS[profile],
        "multiplier": multiplier,
        "basis": "pipeline_config_static_defaults",
        "forced_review_only": gate_profile_requires_review(profile),
        "managed_field_count": len(fields),
        "fields": fields,
    }


def processing_gate_profile_audit(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve the signed task-wide gate profile and per-field provenance."""

    normalized, _adjustments = normalize_processing_parameters(payload)
    return _gate_profile_audit_from_normalized(normalized)


def processing_parameter_custom_fields(
    payload: Mapping[str, Any], *, stages: Iterable[int] | None = None
) -> Dict[str, Any]:
    normalized, _adjustments = normalize_processing_parameters(payload)
    selected = set(stages or STAGE_TITLES)
    result: Dict[str, Any] = {}
    for stage in sorted(selected):
        entry = normalized["stages"][str(stage)]
        if stage in MODE_SPEC_BY_STAGE and entry["mode"] != "auto":
            result[MODE_SPEC_BY_STAGE[stage].field] = entry["mode"]
        result.update(entry["overrides"])
    return result


def apply_processing_parameters_to_config(
    cfg: PipelineConfig,
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], list[Dict[str, Any]], Tuple[str, ...]]:
    """Apply the static gate profile, then explicit task values after auto tune."""

    normalized, adjustments = normalize_processing_parameters(payload)
    manual_fields: list[str] = []
    profile_audit = _gate_profile_audit_from_normalized(normalized)
    for record in profile_audit["fields"]:
        # Every managed gate is reset from the immutable PipelineConfig default
        # before the selected task profile is applied.  Explicit expert values
        # are written in the stage loop below and therefore remain highest
        # priority.
        setattr(cfg, record["field"], record["profile_effective"])
    for stage in STAGE_TITLES:
        entry = normalized["stages"][str(stage)]
        mode_spec = MODE_SPEC_BY_STAGE.get(stage)
        if mode_spec is not None:
            setattr(cfg, mode_spec.field, entry["mode"])
            if entry["mode"] != "auto":
                manual_fields.append(mode_spec.field)
        for field, value in entry["overrides"].items():
            setattr(cfg, field, value)
            manual_fields.append(field)

    deconvolution_mode = getattr(cfg, "stage5_deconvolution_mode", "auto")
    if deconvolution_mode == "graxpert_rl":
        cfg.stage5_deconvolution_enabled = True
        cfg.stage5_graxpert_deconvolution_enabled = True
    elif deconvolution_mode == "rl":
        cfg.stage5_deconvolution_enabled = True
        cfg.stage5_graxpert_deconvolution_enabled = False
    elif deconvolution_mode == "off":
        cfg.stage5_deconvolution_enabled = False
        cfg.stage5_graxpert_deconvolution_enabled = False
    return normalized, adjustments, tuple(manual_fields)


def effective_parameter_value(payload: Mapping[str, Any], field: str) -> Any:
    spec = SPECS_BY_FIELD[field]
    normalized, _adjustments = normalize_processing_parameters(payload)
    entry = normalized["stages"][str(spec.stage)]
    if spec.stage_mode:
        return entry["mode"]
    if field in entry["overrides"]:
        return entry["overrides"][field]
    if spec.profile_scaling != PROFILE_SCALE_NONE:
        _requested, effective, _was_clamped = _scaled_gate_profile_value(
            spec,
            normalized["gate_profile"],
        )
        return effective
    return spec.default


def reset_stage_parameters(
    payload: Mapping[str, Any], stages: Iterable[int] | None = None
) -> Dict[str, Any]:
    normalized, _adjustments = normalize_processing_parameters(payload)
    selected_stages = tuple(stages) if stages is not None else tuple(STAGE_TITLES)
    for stage in selected_stages:
        normalized["stages"][str(stage)] = {"mode": "auto", "overrides": {}}
    if stages is None:
        normalized["gate_profile"] = GATE_PROFILE_DEFAULT
    return normalized


__all__ = [
    "GATE_PROFILE_CHOICES",
    "GATE_PROFILE_DEFAULT",
    "GATE_PROFILE_LABELS",
    "GATE_PROFILE_MULTIPLIERS",
    "GATE_PROFILE_PARAMETER_SPECS",
    "GATE_PROFILE_RELAXED",
    "GATE_PROFILE_UNLIMITED",
    "GENERAL_DEFAULTS",
    "MODE_SPEC_BY_STAGE",
    "PROCESSING_GATE_PARAMETER_SPECS",
    "PROCESSING_PARAMETERS_SCHEMA",
    "LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4",
    "SUPPORTED_PROCESSING_PARAMETERS_SCHEMAS",
    "PROCESSING_PARAMETER_SPECS",
    "ParameterSpec",
    "SPECS_BY_FIELD",
    "SPECS_BY_STAGE",
    "STAGE_TITLES",
    "apply_processing_parameters_to_config",
    "default_processing_parameters",
    "effective_parameter_value",
    "gate_profile_requires_review",
    "normalize_processing_parameters",
    "processing_gate_profile_audit",
    "processing_parameter_custom_fields",
    "reset_stage_parameters",
]
