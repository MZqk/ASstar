"""Shared data models for the Seestar post-processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class PipelineStage(str, Enum):
    """Stable identities and display labels for the formal pipeline stages."""

    PREPARATION = "阶段 1: 前期准备"
    VIEW_CORRECTION = "阶段 2: 裁切"
    BACKGROUND_EXTRACTION = "阶段 3: 背景提取"
    COLOR_CALIBRATION = "阶段 4: 图像解析 + 色彩校准"
    LINEAR_DENOISE = "阶段 5: 线性反卷积 / 轻降噪"
    STAR_SEPARATION = "阶段 6: 去星与星点层准备"
    STRETCHING = "阶段 7: 主体拉伸"
    NEBULA_ENHANCEMENT = "阶段 8: Starless 深加工"
    STAR_REMIXING = "阶段 9: 星点处理与合成"
    EXPORT = "阶段 10: 最终降噪与导出"
    AI_POSTPROCESS = "阶段 11: AI 后期美化"

    @property
    def label(self) -> str:
        return self.value


class PipelineCheckpoint(str, Enum):
    """Named checkpoints that do not consume a formal stage number."""

    TARGET_PROFILE_PREFLIGHT = "Stage 3/4 target profile preflight"
    PRE_STARLESS_COMPATIBILITY_GATE = "Stage 7 兼容检查点: 去星前质量门控"

    @property
    def label(self) -> str:
        return self.value


@dataclass
class PipelineConfig:
    """处理参数配置 - 所有可调参数集中管理"""
    # 重试控制
    max_retries: int = 2            # 命令失败后最大重试次数（总尝试次数 = 1 + max_retries）
    retry_delay: float = 1.0        # 重试基础等待秒数（会按尝试次数递增）
    stage1_register_fail_ratio_max: float = 0.10  # Mark stage1 degraded when register fail ratio exceeds threshold

    # 阶段 2: 裁切
    crop_margin: float = 0.02       # 每边裁切比例（0.02 = 宽高各裁 2%）
    stage2_edge_black_target: float = 0.03  # 阶段2自动黑边裁切目标，后续阶段不再临时裁黑边
    stage2_adaptive_edge_crop_max_passes: int = 3  # 阶段2自适应黑边裁切最大迭代次数
    stage2_adaptive_edge_crop_max_extra: float = 0.035  # 阶段2单次自适应额外裁切上限
    stage2_guard_band_pixels: int = 3  # 阶段2最终裁切后护带检查的像素宽度，清除残余暗边/色偏
    stage2_color_artifact_max_crop: float = 0.15  # 阶段2彩色伪影裁切单边最大比例上限

    # 阶段 3: 背景提取 (RBF)
    bg_samples: int = 20            # 背景采样点数量；越大拟合更细，但更慢且更易过拟合
    bg_tolerance: float = 1.0       # 背景采样容差；增大可容忍更多亮度波动
    bg_smooth: float = 0.5          # 背景模型平滑度；增大更平滑，过大会吞掉大尺度弱信号
    bg_quality_gate_enabled: bool = True  # Enable stage3 post-check to avoid over-subtraction
    bg_std_worsen_ratio_max: float = 1.10  # Reject when bg noise rises above this multiplier
    bg_median_drop_ratio_min: float = 0.20  # Reject when bg median drops below this ratio
    bg_object_preserve_ratio_min: float = 0.40  # Reject when object coverage drops too much
    bg_edge_black_rise_max: float = 0.35  # Reject when edge black clipping rises too much
    bg_star_preserve_ratio_min: float = 0.90  # 阶段3背景提取后星点数量保留率下限
    bg_nebula_mean_change_max: float = 0.10  # 阶段3星云/弥散信号均值相对变化上限

    # 阶段 5: 线性整理 / 反卷积 / 降噪
    denoise_enabled: bool = False   # 是否启用线性阶段降噪
    denoise_mod: float = 0.35       # 降噪强度参数（0~1），越大降噪越强
    denoise_safety_max: float = 0.55  # 降噪强度安全上限，防止细节被抹平
    stage5_builtin_denoise_mod: float = 0.50  # Stage5 Siril 内置线性降噪强度，默认 denoise -mod=0.50 -indep
    stage5_deconvolution_enabled: bool = True  # Stage5 是否在线性降噪前执行 Siril RL 反卷积
    stage5_rl_maxstars: int = 200  # RL 反卷积 PSF 找星数量上限
    stage5_rl_psf_kernel_size: int = 33  # RL 反卷积 makepsf kernel size
    stage5_rl_iters: int = 8  # RL 反卷积迭代次数，过高易放大噪声和星环
    stage5_rl_alpha: float = 3000.0  # RL TV 正则 alpha，越高越保守
    stage5_rl_gdstep: float = 0.0005  # RL 梯度下降步长
    stage5_rl_stop: float = 0.001  # RL 提前停止阈值
    stage5_graxpert_deconv_strength: float = 0.30  # GraXpert 反卷积建议强度占位；默认不自动调用外部工具
    optional_color_transform_enabled: bool = False  # 是否启用可选转色（Alchemy/Hubble）
    workflow_plugin_probe_enabled: bool = False  # Probe broad workflow plugin commands only when explicitly enabled; stage8 has a narrow safe SASP probe
    spcc_enabled: bool = True  # Prefer SPCC first by default; can be disabled via SEESTAR_SPCC_ENABLE=0
    stage4_platesolve_enabled: bool = True  # 阶段4默认执行 platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3
    stage4_spcc_sensor_mode: str = "osc"  # SPCC 传感器模式：osc 或 mono_lrgb
    stage4_spcc_osc_sensor: str = "Sony IMX585"  # OSC/SPCC oscsensor 名称；Seestar S30 Pro 使用 Sony IMX585
    stage4_spcc_osc_filter: str = ""  # OSC/SPCC oscfilter 显式覆盖；为空时按内置双带光害滤镜开关自动选择
    stage4_spcc_builtin_dualband_filter_enabled: bool = False  # 是否开启 Seestar 内置双带光害滤镜；决定默认 SPCC oscfilter
    stage4_spcc_mono_sensor: str = ""  # Mono/LRGB SPCC monosensor 名称；为空时回退 OSC 配置
    stage4_spcc_r_filter: str = ""  # Mono/LRGB SPCC R filter 名称
    stage4_spcc_g_filter: str = ""  # Mono/LRGB SPCC G filter 名称
    stage4_spcc_b_filter: str = ""  # Mono/LRGB SPCC B filter 名称
    stage4_spcc_white_ref: str = "Average Spiral Galaxy"  # SPCC white reference
    stage4_spcc_adaptive_white_ref_enabled: bool = False  # 默认按 Siril SPCC 方案使用 Average Spiral Galaxy；需要时可显式开启星云白参考切换
    stage4_spcc_nebula_white_ref: str = "Star, type G2(v)"  # 发射/双窄带星云默认 SPCC 星型白参考；可由 env 覆盖
    stage4_spcc_bgtol: str = "-2.8,2.0"  # SPCC 背景容差参数；默认值不显式传入，PCC 使用 Siril 默认 bgtol
    stage4_spcc_limitmag: str = "10.5"  # SPCC Gaia 查询极限星等；限制测光星数以降低 Siril 1.4.x SPCC 崩溃风险
    stage4_spcc_restore_cpu: int = 0  # SPCC 前 setcpu 1；结束后恢复到该值，0 表示自动使用 CPU 数
    stage4_pcc_header_fallback_enabled: bool = True  # platesolve 失败但 FITS 有 RA/DEC 时，允许尝试 PCC header 坐标回退
    stage4_local_star_wb_enabled: bool = True  # SPCC/PCC 都失败时，允许基于未饱和星点做保守白平衡
    stage4_local_star_wb_min_pixels: int = 32  # 本地星点白平衡所需的最小白参考像素数
    stage4_local_star_wb_gain_limit: float = 1.25  # 本地星点白平衡单通道增益限制，避免替代光度校准
    stage4_local_star_wb_target_aware_enabled: bool = False  # 发射/双窄带目标默认跳过本地星点白平衡以保护 Hα/OIII 色彩
    aberration_api_enabled: bool = False  # Disabled by default: API path may fail in siril-cli thread ownership context

    # 阶段 6: 拉伸
    asinh_stretch: float = 3.0      # Asinh 拉伸强度（越大整体越亮）
    asinh_offset: float = 0.001     # Asinh 偏移，影响暗部起拉位置
    ghs_shadowsclip: float = -2.8   # GHS 阴影裁剪，控制黑场压暗程度（回退方案用）
    ghs_stretchamount: float = 2.0  # GHS 拉伸量（回退方案用）

    # 阶段 8: 星云饱和度
    nebula_saturation: float = 0.4  # 去星图饱和度增强幅度
    nebula_bg_factor: int = 1       # 饱和度算法背景抑制系数（Siril `satu` 第二参数）
    stage8_masked_enhancement_enabled: bool = True  # 阶段8默认使用 Starless soft mask 分区增强，保护核心与背景
    stage8_core_protection_strength: float = 0.92  # 阶段8亮核心回混原图强度，越高越保护核心不过曝
    stage8_background_denoise_strength: float = 0.14  # 阶段8背景区域轻量降噪强度
    stage8_faint_nebula_boost_max: float = 0.08  # 阶段8外围暗云气最大提亮强度
    stage8_nebula_contrast_max: float = 0.10  # 阶段8星云主体局部对比最大强度
    stage8_masked_unsharp_amount_max: float = 0.12  # 阶段8非核心星云区域轻锐化上限
    stage8_blue_precontrol_strength: float = 0.55  # 阶段8增强前信号区蓝偏预抑制强度，减少后置 ccm 补救
    stage8_bg_std_growth_max: float = 1.08  # 阶段8背景噪声增长上限
    stage8_texture_artifact_growth_max: float = 1.25  # 阶段8纹理伪影评分增长上限

    # 阶段 9: 星点混合
    star_intensity: float = 1.0     # 传统回混时星点层主强度
    star_fallback_intensity: float = 0.95  # 主强度失败时的回退星点强度
    stage9_starmask_stretch_enabled: bool = True  # 阶段9像素回混前默认把线性 starmask 独立 Asinh 拉伸到非线性域
    stage9_starmask_asinh_stretch: float = 2.0  # 阶段9 starmask 温和 Asinh 拉伸强度，保护星色并避免星核过曝
    stage9_starmask_asinh_offset: float = 0.001  # 阶段9 starmask Asinh 偏移
    remix_safe_blend: bool = True   # 兼容旧配置；阶段 9 当前固定使用上一阶段 starless + starmask
    remix_nebula_weight: float = 0.18  # 兼容旧自动调参字段；阶段 9 不再用于 stage6/starless 安全混合

    # 阶段 10: 最终饱和度
    final_saturation: float = 0.15  # 导出前最终全图饱和度微调量
    final_bg_factor: int = 1        # 最终饱和度背景抑制系数（Siril `satu` 第二参数）

    # Stage 11: Optional AI postprocess
    ai_post_enabled: bool = False   # Enable optional AI postprocess stage
    ai_endpoint: str = ""           # OpenAI-compatible image edit endpoint
    ai_model: str = ""              # User-provided model name
    ai_api_key: str = ""            # User-provided API key
    ai_timeout_sec: int = 90        # API request timeout in seconds
    ai_strength: float = 0.12       # Conservative blend ratio for AI result
    ai_prompt: str = ""             # Custom prompt; empty uses default conservative prompt
    ai_stage6_enabled: bool = True  # AI 总开关开启且凭据齐全时，允许阶段6使用 AI 拉伸顾问
    ai_stage7_enabled: bool = True  # AI 总开关开启且凭据齐全时，允许阶段7使用 AI SyQon 参数优化
    ai_stage8_enabled: bool = True  # AI 总开关开启且凭据齐全时，允许阶段8使用 AI Starless 参数优化

    # Stage 11: Quality gate thresholds
    ai_bg_median_delta_max: float = 0.03    # Allowed background median drift
    ai_color_ratio_delta_max: float = 0.25  # Allowed color-ratio drift (R/G, B/G)
    ai_core_growth_ratio_max: float = 1.35  # Allowed bright-core growth ratio
    ai_star_growth_ratio_max: float = 1.25  # Allowed median star-size growth ratio

    # Stage 6-8: AI parameter optimization diagnostic thresholds
    stage6_bg_median_min: float = 0.020  # 阶段6验收：背景中值不能低于此值，避免黑场压死
    stage6_black_pixel_ratio_max: float = 0.35  # 阶段6验收：近黑像素占比上限
    stage6_highlight_clip_ratio_max: float = 0.010  # 阶段6验收：高亮裁剪占比上限
    stage6_star_growth_ratio_max: float = 1.25  # 阶段6验收：星点中位尺寸增长上限
    stage7_quality_retry_max: int = 2  # 阶段7质量差时最多追加的 SyQon/SASP 补救尝试次数
    stage7_edge_black_warn: float = 0.10  # 阶段7去星前黑边风险提示阈值
    stage7_edge_black_high: float = 0.18  # 阶段7去星前高风险黑边阈值
    stage7_bg_median_high: float = 0.16  # 阶段7去星前高背景中值阈值
    stage7_bg_std_high: float = 0.055  # 阶段7去星前高背景噪声阈值
    stage7_bg_noise_ratio_high: float = 0.55  # 阶段7去星前背景噪声/背景中值高风险阈值
    stage7_residual_star_score_max: float = 0.45  # 阶段7验收：starless 残星评分上限
    stage7_halo_residue_score_max: float = 0.35  # 阶段7验收：亮星 halo 残留评分上限
    stage7_bright_nebula_halo_residue_score_max: float = 0.60  # M42/亮核心星云允许真实星云光晕保留，使用更高 halo 验收上限
    stage7_black_hole_score_max: float = 0.08  # 阶段7验收：去星暗坑/暗环评分上限
    stage7_starmask_contamination_max: float = 0.25  # 阶段7验收：星点层星云污染评分上限
    stage7_starless_noise_gain_max: float = 1.25  # 阶段7验收：starless 背景噪声增益上限
    stage7_starmask_coverage_min_ratio: float = 0.35  # 阶段7验收：starmask 覆盖相对 stage6 星点覆盖的下限
    stage7_starmask_width_ratio_max: float = 1.80  # 阶段7验收：starmask 中位宽度相对 stage6 星点宽度上限
    stage7_starless_dynamic_range_min_ratio: float = 0.55  # 阶段7验收：starless 相对输入的有效动态范围下限，防止去星输出塌缩
    stage7_starless_peak_signal_min: float = 0.006  # 阶段7验收：低动态范围 starless 的最低峰值信号，低于此值触发保守重试
    stage7_starmask_clean_enabled: bool = True  # 阶段7默认清理星点层，避免背景/星云残差直接回混
    stage7_starmask_background_floor_percentile: float = 55.0  # 阶段7星点层低亮背景残差软阈值百分位
    stage7_starmask_halo_blur_strength: float = 0.35  # 阶段7亮星 halo 区域轻微平滑强度
    stage7_starmask_small_star_scale: float = 0.88  # 阶段7小/弱星点层保守降强比例
    stage7_starmask_nebula_suppression: float = 0.75  # 阶段7星云主体区域星点层降权强度
    stage7_conservative_repair_enabled: bool = True  # 阶段7质量差时允许用较轻拉伸输入重跑去星
    stage7_skip_unready_starless: bool = True  # Stage6.5 判定不适合去星时，默认跳过去星并进入 review export
    star_separation_mode: str = "linear_star_separation"  # 星点分离输入模式：linear_star_separation 或 mild_prestretch_star_separation
    star_separation_fallback_to_mild_prestretch: bool = True  # 线性去星失败时是否自动改用轻微预拉伸输入重试
    mild_prestretch_strength: float = 1.35  # 轻微预拉伸去星强度；仅用于去星输入，不替代后续主体拉伸
    stage7_conservative_asinh_stretch: float = 2.00  # 阶段7保守去星输入的轻拉伸强度
    stage7_ultra_conservative_asinh_stretch: float = 1.65  # 阶段7更保守去星输入的极轻拉伸强度
    stage7_soft_starless_asinh_stretch: float = 1.35  # 阶段7质量差时追加更低强度去星输入，减少黑洞和彩色残渣
    stage7_conservative_asinh_offset: float = 0.0025  # 阶段7保守去星输入的 Asinh 偏移
    stage7_starless_pixel_repair_enabled: bool = True  # 阶段7质量差时对 starless 做残星/halo/背景彩噪局部修复
    stage7_starless_repair_strength: float = 0.68  # 阶段7残星小尺度修复强度
    stage7_starless_halo_repair_strength: float = 0.70  # 阶段7亮星 halo 平滑修补强度
    stage7_starless_chroma_denoise_strength: float = 0.55  # 阶段7背景 chroma noise reduction 强度
    stage7_starless_repair_max_score_growth: float = 0.00  # 阶段7像素修复不得让综合质量评分变差
    stage8_force_conservative_after_stage7_repair: bool = True  # Stage7 修复/不安全时 Stage8 禁止细节增强
    stage8_mask_signal_coverage_min: float = 0.002  # Stage8 增强前信号 mask 覆盖下限，小视场 M42 等紧凑目标保守放行
    stage8_blue_excess_max: float = 0.08  # 阶段8验收：蓝色相对红/绿通道的过量上限
    stage8_saturation_growth_ratio_max: float = 1.45  # 阶段8验收：饱和度增长上限
    stage8_microcontrast_growth_ratio_max: float = 1.60  # 阶段8验收：微对比增长上限
    stage8_highlight_clip_ratio_max: float = 0.012  # 阶段8验收：高亮裁剪占比上限

    # I/O 控制
    output_format: str = "all"  # 最终导出格式：all 或逗号分隔 tif/png/fit
    checkpoint_mode: bool = False   # True: 仅保存关键检查点
    debug_mode: bool = False        # True: 保留 stage* 中间文件（lightsrc 序列仍会清理）
    auto_tune_enabled: bool = True  # True: 启用自动识别目标并安全调参
    auto_tune_debug: bool = False   # True: 输出更详细自动调参日志；False 时跟随 debug_mode

    # 输入过滤 - 排除中间产物前缀和后缀
    exclude_prefixes: tuple = (      # 文件名此前缀开头时视为中间产物并跳过
        'light_', 'pp_', 'r_', 'stage', 'starless', 'starmask',
        'working', 'result', 'sasp_',
    )
    exclude_suffixes: tuple = (      # 文件名（不含扩展名）以这些后缀结尾时跳过
        '_processed', '_final', '_enhanced', '_remixed',
    )
    exclude_substrings: tuple = (    # 文件名（不含扩展名）包含这些片段时跳过
        'starless', 'starmask',
    )


class TargetType(Enum):
    """自动识别目标类型"""
    EMISSION_NEBULA = "emission_nebula"
    REFLECTION_NEBULA = "reflection_nebula"
    GALAXY = "galaxy"
    CLUSTER = "cluster"
    PLANETARY_NEBULA = "planetary_nebula"
    WIDEFIELD = "widefield"
    PLANETARY = "planetary"
    UNKNOWN = "unknown"


@dataclass
class ImageFeatures:
    """自动调参用图像特征"""
    bg_median: float = 0.12
    bg_std: float = 0.03
    red_dominance: float = 1.0
    blue_dominance: float = 1.0
    star_density: float = 0.002
    median_star_size: float = 1.8
    object_area_ratio: float = 0.25
    diffuse_ratio: float = 0.25
    core_brightness_ratio: float = 0.03
    edge_black_ratio: float = 0.10
    global_dark_ratio: float = 0.0


@dataclass
class QualityMetrics:
    """阶段 6-8 AI 参数优化/诊断用的轻量像素指标"""
    bg_median: float = 0.0
    black_pixel_ratio: float = 0.0
    highlight_clip_ratio: float = 0.0
    star_density: float = 0.0
    median_star_size: float = 0.0
    star_coverage_ratio: float = 0.0
    star_energy_ratio: float = 0.0
    saturation_median: float = 0.0
    saturation_p95: float = 0.0
    microcontrast: float = 0.0
    blue_excess: float = 0.0


@dataclass
class AutoTuneResult:
    """记录自动识别与调参详情"""
    target_type: TargetType = TargetType.UNKNOWN
    features: ImageFeatures = field(default_factory=ImageFeatures)
    changed_params: List[Tuple[str, Any, Any, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class Stage6StretchStrategy:
    """阶段 6 拉伸策略。"""
    name: str
    summary: str
    candidates: List[Dict[str, Any]]
    use_curves: bool = False
    curves_label: str = ""
    protection_note: str = ""


@dataclass
class StageResult:
    """单个阶段的执行结果"""
    name: str
    status: str = 'pending'     # ok / degraded / failed / skipped
    duration: float = 0.0
    message: str = ''

    @property
    def display_status(self) -> str:
        """Return a summary-only status without changing pipeline control flow."""
        if self.status != "ok":
            return self.status
        message = self.message.lower()
        fallback_markers = (
            "fallback:",
            "fallback_status=success",
            "fallback_used=true",
            "fallback used=true",
            "using fallback",
            "fallback applied",
            "explicit fallback",
            "已回退",
            "使用回退",
        )
        if any(token in message for token in fallback_markers):
            return "ok_with_fallback"
        if any(token in message for token in ("skipped", "跳过", "disabled")):
            return "ok_skipped_optional"
        return self.status
