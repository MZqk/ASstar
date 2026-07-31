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


class StarSeparationState(str, Enum):
    """Semantic result of Stage 6; passthrough images are never starless."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    TARGET_BYPASS = "target_bypass"
    REJECTED = "rejected"
    TOOL_FAILED = "tool_failed"


class InputState(str, Enum):
    """Resolved transfer-function state of the current input image."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    UNKNOWN = "unknown"


@dataclass
class InputProfile:
    """Evidence-backed input state used to guard destructive linear stages."""

    state: InputState
    confidence: float
    source: str
    input_mode: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    pixel_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def safe_for_linear_steps(self) -> bool:
        return self.state is InputState.LINEAR and not self.conflicts

    @property
    def requires_review(self) -> bool:
        return not self.safe_for_linear_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "seestar.input-profile.v1",
            "state": self.state.value,
            "confidence": max(0.0, min(1.0, float(self.confidence))),
            "source": self.source,
            "input_mode": self.input_mode,
            "safe_for_linear_steps": self.safe_for_linear_steps,
            "requires_review": self.requires_review,
            "evidence": list(self.evidence),
            "conflicts": list(self.conflicts),
            "pixel_metrics": dict(self.pixel_metrics),
        }


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
    stage3_conditional_decision_enabled: bool = True  # 阶段3先诊断再决定 apply/skip/review，禁止无证据直接扣背景
    stage3_deterministic_auto_apply_enabled: bool = True  # 无视觉顾问时仅对高置信梯度使用离线确定性 apply
    stage3_apply_confidence_min: float = 0.75  # 外部/视觉背景建议获准执行所需最低置信度
    stage3_gradient_skip_max: float = 0.045  # 低于该内部工程门禁且背景干净时跳过 DBE
    stage3_dirty_skip_max: float = 0.16  # 与低梯度同时满足时判定无需 DBE
    stage3_gradient_apply_min: float = 0.08  # 确定性自动 apply 所需最低方向梯度评分
    stage3_dirty_apply_min: float = 0.18  # 确定性自动 apply 所需最低背景污染评分
    stage3_diffuse_auto_apply_enabled: bool = False  # 弥散星云/大尺度信号默认禁止自动 DBE，转人工复核

    # 阶段 5: 线性整理 / 反卷积 / 降噪
    denoise_enabled: bool = False   # 是否启用线性阶段降噪
    denoise_mod: float = 0.35       # 降噪强度参数（0~1），越大降噪越强
    denoise_safety_max: float = 0.55  # 降噪强度安全上限，防止细节被抹平
    stage5_multiscale_denoise_enabled: bool = True  # 启用噪声模型驱动的亮度/对立色度多尺度确定性候选
    stage5_multiscale_denoise_strength: float = 0.72  # 多尺度软阈值与回混强度，受质量门限制
    stage5_multiscale_detail_retention_min: float = 0.82  # 主体高频细节最低保留比例
    stage5_multiscale_noise_reduction_min: float = 0.05  # 非低噪输入所需最低背景噪声下降比例
    stage5_builtin_denoise_mod: float = 0.50  # Stage5 Siril 内置线性降噪强度，默认 denoise -mod=0.50 -indep
    stage5_deconvolution_enabled: bool = True  # Stage5 是否在线性降噪前执行 GraXpert/RL 反卷积
    stage5_graxpert_deconvolution_enabled: bool = True  # Stage5 是否优先尝试本地 GraXpert 对象反卷积；关闭后直接使用 RL
    stage5_rl_maxstars: int = 200  # RL 反卷积 PSF 找星数量上限
    stage5_rl_psf_kernel_size: int = 33  # RL 反卷积 makepsf kernel size
    stage5_rl_iters: int = 8  # RL 反卷积迭代次数，过高易放大噪声和星环
    stage5_rl_alpha: float = 3000.0  # RL TV 正则 alpha，越高越保守
    stage5_rl_gdstep: float = 0.0005  # RL 梯度下降步长
    stage5_rl_stop: float = 0.001  # RL 提前停止阈值
    stage5_graxpert_deconv_strength: float = 0.30  # 本地 Object Deconvolution 模型可用时的 GraXpert 强度
    optional_color_transform_enabled: bool = False  # 是否启用可选转色（Alchemy/Hubble）
    workflow_plugin_probe_enabled: bool = False  # Probe broad workflow plugin commands only when explicitly enabled; stage8 has a narrow safe SASP probe
    stage4_platesolve_enabled: bool = True  # 阶段4默认执行 platesolve -noflip，解算时保留原始图像方向
    stage4_auto_geometry_enabled: bool = True  # 高置信 FITS/设备几何可用于 platesolve；显式环境覆盖始终优先
    stage4_auto_geometry_confidence_min: float = 0.85  # 自动使用设备几何所需最低置信度
    stage4_auto_geometry_scale_residual_max: float = 0.05  # 解算 WCS 像素比例相对预测值最大偏差，超限回滚并禁止 PCC
    stage4_narrowband_normalization_enabled: bool = True  # 仅对已确认 Ha/OIII 通道映射的双窄带 RGB 启用确定性归一化
    stage4_nbn_mapping_confidence_min: float = 0.85  # 低于此置信度时保留输入，不猜测窄带通道含义
    stage4_nbn_strength: float = 0.55  # HOO 通道跨度与背景中和的保守混合强度
    stage4_nbn_gain_limit: float = 1.08  # 单通道最大归一化增益/衰减范围
    stage4_nbn_line_ratio_drift_max: float = 0.12  # Ha/OIII 信号比例最大相对漂移
    stage4_pcc_timeout_sec: int = 180  # 本地优先/在线回退的 Gaia PCC 只尝试一次；独立 Siril 子进程达到该秒数即终止
    stage4_pcc_quality_gate_enabled: bool = True  # PCC 候选必须通过目标感知色彩质量门，否则回到 pre_pcc
    stage4_pcc_channel_gain_ratio_max: float = 1.80  # PCC 三通道相对增益最大跨度，限制异常色偏
    stage4_pcc_emission_balance_gain_ratio_max: float = 4.0  # 发射星云仅在背景色差显著改善且其他门均安全时允许的大增益跨度
    stage4_pcc_clip_growth_max: float = 0.005  # PCC 相对 pre_pcc 允许新增的高光裁剪比例
    stage4_pcc_star_temperature_ratio_min: float = 0.45  # 校色后可测恒星综合色温中位数相对输入的最低比例
    stage4_pcc_star_temperature_ratio_max: float = 2.20  # 校色后可测恒星综合色温中位数相对输入的最高比例
    stage4_pcc_background_color_delta_max: float = 0.22  # 背景变得更失衡时允许的最大归一化 RGB 色差
    stage4_pcc_target_color_drift_max: float = 0.28  # 普通主体归一化 RGB 色度允许的最大漂移
    stage4_pcc_emission_target_color_drift_max: float = 0.45  # 发射星云保留真实线发射主色时允许的更宽主体色度漂移
    stage4_local_star_wb_enabled: bool = True  # PCC 失败/拒绝时，仅在恒星软遮罩内做保守色彩恢复
    stage4_local_star_wb_min_pixels: int = 32  # 本地星点白平衡所需的最小白参考像素数
    stage4_local_star_wb_gain_limit: float = 1.20  # 恒星软遮罩内单通道增益限制，避免替代光度校准
    stage4_local_star_mask_radius: int = 2  # 恒星样本向星翼扩展的软遮罩半径，限制校色只影响局部
    stage4_local_star_mask_coverage_max: float = 0.12  # 恒星软遮罩最大覆盖率，超限时保留输入颜色
    aberration_api_enabled: bool = False  # Disabled by default: API path may fail in siril-cli thread ownership context

    # 阶段 7: Starless 主体拉伸
    asinh_stretch: float = 3.0      # Asinh 拉伸强度（越大整体越亮）
    asinh_offset: float = 0.001     # Asinh 偏移，影响暗部起拉位置
    ghs_shadowsclip: float = -2.8   # GHS 阴影裁剪，控制黑场压暗程度（回退方案用）
    ghs_stretchamount: float = 2.0  # GHS 拉伸量（回退方案用）

    # 阶段 8: 星云饱和度
    nebula_saturation: float = 0.4  # 去星图饱和度增强幅度
    nebula_bg_factor: int = 1       # 饱和度算法背景抑制系数（Siril `satu` 第二参数）
    stage8_masked_enhancement_enabled: bool = True  # 阶段8默认使用 Starless soft mask 分区增强，保护核心与背景
    stage8_local_adjustment_engine_enabled: bool = True  # 使用版本化本地曲线/蒙版配方，候选未过门则保留配方前图像
    stage8_local_curve_opacity: float = 0.30  # 本地微曲线最大混合比例，Stage8 仍受背景与核心质量门限制
    stage8_core_protection_strength: float = 0.92  # 阶段8亮核心回混原图强度，越高越保护核心不过曝
    stage8_background_denoise_strength: float = 0.14  # 阶段8背景区域轻量降噪强度
    stage8_faint_nebula_boost_max: float = 0.08  # 阶段8外围暗云气最大提亮强度
    stage8_nebula_contrast_max: float = 0.10  # 阶段8星云主体局部对比最大强度
    stage8_masked_unsharp_amount_max: float = 0.12  # 阶段8非核心星云区域轻锐化上限
    stage8_blue_precontrol_strength: float = 0.55  # 阶段8增强前信号区蓝偏预抑制强度，减少后置 ccm 补救
    stage8_bg_std_growth_max: float = 1.08  # 阶段8背景噪声增长上限
    stage8_texture_artifact_growth_max: float = 1.25  # 阶段8纹理伪影评分增长上限
    stage8_limited_saturation_max: float = 0.05  # 亮星云 halo 中风险区受限候选的饱和度硬上限
    stage8_limited_halo_texture_growth_max: float = 1.05  # 受限候选星周环带纹理相对增长上限
    stage8_limited_halo_texture_delta_max: float = 0.00075  # 环带绝对增长低于此值时豁免比例误报

    # 阶段 9: 星点混合
    star_intensity: float = 1.0     # 传统回混时星点层主强度
    star_fallback_intensity: float = 0.95  # 兼容字段；作为 Stage 9 首档回退强度的上限
    stage9_fallback_intensity_levels: Tuple[float, ...] = (0.75, 0.55, 0.40)  # 主候选拒绝后逐档降低回星强度
    stage9_starmask_stretch_enabled: bool = True  # 阶段9像素回混前默认把线性 starmask 独立 Asinh 拉伸到非线性域
    stage9_starmask_adaptive_stretch_enabled: bool = True  # 根据星点层有效信号分布反解 Asinh 强度，而不是固定套用单一参数
    stage9_compact_starmask_enabled: bool = True  # Asinh 前仅保留连通紧致星核和窄星翼；覆盖异常时允许重建更严格支持层
    stage9_star_color_repair_enabled: bool = True  # 用不可变线性含星参考修复星层核心/星翼色度，候选失败即保留原层
    stage9_star_color_repair_strength: float = 0.72  # 参考色度回混强度；亮星核心自动降低以保留真实星色和通量
    stage9_star_color_support_ratio_max: float = 0.12  # 星色修复允许影响的最大画面覆盖率
    stage9_star_color_improvement_min: float = 0.01  # 候选所需最小中位色度误差改善
    stage9_star_color_post_chroma_error_max: float = 0.22  # 拉伸及回混前最终星层相对参考的中位色度误差上限
    stage9_source_star_detail_percentile: float = 98.0  # 从原始含星图提取独立星核目录的局部细节百分位
    stage9_source_component_density_max: float = 2500.0  # 独立星表紧致组件密度上限；伴随单像素噪点证据时自适应收紧或 fail-closed
    stage9_source_single_pixel_ratio_max: float = 0.20  # 独立星表单像素组件比例上限，单项超限即自适应收紧或 fail-closed
    stage9_star_reference_sigma: float = 5.0  # 原始 starmask 星点目录检测阈值，使用背景加该倍数噪声标准差
    stage9_compact_weak_star_retention_min: float = 0.80  # compact 支持层必须保留的弱星组件数量比例下限
    stage9_mixed_star_peak_ratio_min: float = 4.0  # 亮星组峰值中位数相对弱星组的倍率达到该值时启用多锚点曲线
    stage9_mixed_star_weak_count_min: int = 20  # 启用混合星场多锚点曲线所需的最少弱星组件数
    stage9_mixed_star_bright_count_min: int = 3  # 启用混合星场多锚点曲线所需的最少亮星组件数
    stage9_starmask_asinh_stretch: float = 2.0  # 阶段9 starmask 温和 Asinh 拉伸强度，保护星色并避免星核过曝
    stage9_starmask_asinh_offset: float = 0.001  # 阶段9 starmask Asinh 偏移
    stage9_starmask_asinh_stretch_max: float = 1000.0  # 自适应星点拉伸反解上限；亮星目标约束优先，统计不可用时仍回退固定强度
    stage9_starmask_faint_target: float = 0.26  # 多锚点曲线弱星中位目标亮度
    stage9_starmask_mid_target: float = 0.50  # 多锚点曲线中亮星目标亮度
    stage9_starmask_bright_target: float = 0.75  # 多锚点曲线亮星目标亮度
    stage9_starmask_peak_target: float = 0.90  # 多锚点曲线极亮星上限，保留高光和星色余量
    stage9_starmask_chroma_regularization_enabled: bool = True  # 多锚点拉伸时用邻域星色约束微弱星翼，避免把单像素通道噪声放大成蓝紫色块
    stage9_starmask_faint_chroma_max: float = 0.35  # 微弱星翼允许的最大通道跨度比例，优先抑制低信号伪色
    stage9_starmask_bright_chroma_max: float = 0.60  # 亮星核心允许的最大通道跨度比例，保留真实星色同时限制极端色边
    stage9_starmask_predicted_change_ratio_max: float = 0.30  # 拉伸求解时预测的显著变化覆盖上限，给正式门控预留余量
    stage9_quality_gate_enabled: bool = True  # 阶段9保存前验收星点回混，高风险候选回滚并降强度重试
    stage9_highlight_clip_ratio_max: float = 0.015  # 阶段9回混后高光裁剪占比上限
    stage9_highlight_clip_growth_max: float = 0.006  # 阶段9相对 Stage8 的高光裁剪增长上限
    stage9_bright_pixel_growth_max: float = 0.025  # 阶段9亮像素覆盖增长上限，限制星点膨胀/光晕
    stage9_background_lift_max: float = 0.010  # 阶段9暗背景中位抬升上限，限制 starmask 污染
    stage9_background_mottling_growth_max: float = 1.35  # 阶段9相对 Stage8 的低频背景斑驳增长倍数上限
    stage9_mottling_exemption_changed_pixel_ratio_max: float = 0.12  # 仅局部变化时才允许低绝对斑驳豁免，防止大面积星点/背景污染漏过门控
    stage9_changed_pixel_ratio_max: float = 0.35  # 阶段9显著变化像素占比上限
    stage9_darkening_ratio_max: float = 0.005  # 阶段9异常变暗像素占比上限
    stage9_weak_star_recovery_ratio_min: float = 0.70  # 候选相对 Stage8 至少恢复的弱星组件数量比例
    stage9_star_recovery_ratio_min: float = 0.75  # 候选相对 Stage8 至少恢复的全部星点组件数量比例
    stage9_weak_star_screen_intensity_min: float = 0.40  # 弱星 Screen 强度下限；恢复率门控负责防止 fallback 过度压低弱星
    stage9_star_support_ratio_max: float = 0.12  # 独立星表生成的实际回混支持层最大覆盖
    stage9_unmatched_changed_ratio_max: float = 0.01  # 星点支持层之外允许发生显著变化的最大比例
    stage9_chromatic_addition_peak_min: float = 0.02  # 局部伪色门控只统计高于该新增峰值的回星像素
    stage9_chromatic_addition_saturation_min: float = 0.70  # 新增星点通道跨度达到该比例时视为极端局部色彩
    stage9_chromatic_addition_ratio_max: float = 0.003  # 极端局部色彩新增像素占全图的最大比例，超限回滚候选
    stage9_local_component_peak_min: float = 0.01  # 局部连通块门控使用的最低新增峰值
    stage9_local_component_area_max: int = 256  # 单个新增连通块最大面积，限制非星形大片结构
    stage9_local_component_aspect_ratio_max: float = 3.0  # 新增连通块最长/最短边上限
    stage9_local_component_fill_ratio_min: float = 0.15  # 新增连通块最低包围盒填充率，排除细长/碎裂结构
    stage9_local_single_pixel_ratio_max: float = 0.20  # Stage 9 新增连通块中单像素组件比例上限
    stage9_local_cyan_blue_peak_min: float = 0.01  # 局部青蓝色团块最低新增峰值
    stage9_local_cyan_blue_saturation_min: float = 0.50  # 局部青蓝色团块最低新增色彩跨度
    stage9_local_cyan_blue_component_area_max: int = 64  # 单个局部青蓝色团块最大面积
    stage9_core_percentile: float = 90.0  # 以 Stage 8 底图信号区该亮度百分位定义星云核心
    stage9_core_color_jump_min: float = 0.10  # 核心区域归一化 RGB 单通道最大突变量
    stage9_core_color_jump_component_area_max: int = 64  # 核心颜色突变连通块最大面积
    stage9_star_aperture_recovery_ratio_min: float = 0.75  # 7x7 星点孔径通量最低恢复数量比例
    stage9_star_wing_recovery_ratio_min: float = 0.65  # 参考星翼最低恢复数量比例
    stage9_residual_dark_hole_ratio_max: float = 0.15  # 回星后星点邻域残余暗坑像素比例上限
    stage9_hollow_structure_delta_min: float = 0.05  # 检测新增环状/空心结构的星层最小亮度变化
    stage9_new_hollow_structure_area_max: int = 64  # 新增闭合环内部允许的最大空心面积，超限候选回滚
    remix_safe_blend: bool = True   # 兼容旧配置；阶段 9 当前固定使用上一阶段 starless 与 starmask 的 Screen 回混
    remix_nebula_weight: float = 0.18  # 兼容旧自动调参字段；阶段 9 不再用于 stage6/starless 安全混合

    # 阶段 10: 最终饱和度
    final_saturation: float = 0.15  # 导出前最终全图饱和度微调量
    final_bg_factor: int = 1        # 最终饱和度背景抑制系数（Siril `satu` 第二参数）
    stage10_chroma_focus_score_min: float = 0.34  # Stage10 综合色噪达到该值时优先保护亮度、只处理色度
    stage10_separate_chroma_score_min: float = 0.70  # Stage10 严重通道色噪启用逐通道降噪的门限
    stage10_full_bg_std_min: float = 0.018  # Stage10 色噪伴随亮度背景噪声时改用 full 的 bg_std 门限
    stage10_full_mottling_score_min: float = 0.45  # Stage10 色噪伴随背景斑驳时改用 full 的斑驳门限
    stage10_stage9_local_color_risk_strength: float = 1.0  # 按 Stage9 局部青蓝/核心突变风险比例压低正向最终饱和度
    stage10_managed_output_enabled: bool = True  # 独立生成带 sRGB/ICC 的 16-bit PNG/TIFF；永不重写 FITS 科学存档
    force_review_only_output: bool = False  # 显式启用时 Stage10 仅写 result_review*，不写正式结果名

    # Stage 11: Optional AI postprocess
    review_bundle_enabled: bool = True  # 为关键阶段生成 before/after/diff/metrics 视觉复核包
    ai_post_enabled: bool = False   # 可选 AI 顾问/Stage11 副本总开关；联网仍需 SEESTAR_NETWORK_MODE=1
    ai_endpoint: str = ""           # OpenAI-compatible chat endpoint
    ai_model: str = ""              # User-provided model name
    ai_api_key: str = ""            # User-provided API key
    ai_timeout_sec: int = 90        # API request timeout in seconds
    ai_strength: float = 0.12       # Conservative blend ratio for AI result
    ai_prompt: str = ""             # Custom prompt; empty uses default conservative prompt
    ai_advisor_mode: str = "text"  # AI 顾问模式：text 或 multimodal；模型只返回代码候选 ID
    ai_stage6_enabled: bool = True  # 联网与总开关开启后，允许 AI 从通过硬门控的拉伸候选 ID 中选择
    ai_stage7_enabled: bool = True  # 联网与总开关开启后，允许 AI 从代码生成的 SyQon 候选 ID 中选择
    ai_stage8_enabled: bool = True  # 联网与总开关开启后，允许 AI 从代码生成的 Starless 增强候选 ID 中选择

    # Stage 11 后：完全隔离的 AI 艺术衍生实验（不属于正式阶段）
    ai_artistic_derivative_enabled: bool = False  # 显式开启后才上传 Stage10 预览并生成独立艺术衍生图
    ai_artistic_endpoint: str = ""  # 独立 OpenAI-compatible /images/edits endpoint，不复用 advisor endpoint
    ai_artistic_model: str = ""  # 独立图像编辑/生成模型名
    ai_artistic_api_key: str = ""  # 独立 API 密钥，不复用 advisor 密钥
    ai_artistic_prompt: str = ""  # 艺术衍生提示词；为空使用带非科学声明的默认提示词
    ai_artistic_timeout_sec: int = 180  # 艺术衍生请求超时，安全限幅 30–600 秒

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
    stage7_bright_nebula_star_growth_ratio_max: float = 1.50  # Stage 7 亮核心星云 starless 拉伸的星状结构增长上限
    stage7_preview_calibration_enabled: bool = True  # 阶段7用 linked preview 的 P50/P99 标定正式 Asinh 候选
    stage7_target_aware_stretch_enabled: bool = True  # 阶段7按 target profile/policy 收紧核心保护或增强弱信号，但保持固定双候选
    stage7_preview_cand_a_p50_ratio: float = 0.35  # cand_a 目标背景相对 preview P50 的保守比例
    stage7_preview_cand_b_p50_ratio: float = 0.25  # cand_b 后接 GHS，Asinh 目标比例更低
    stage7_preview_asinh_stretch_max: float = 1000.0  # preview 反解 Asinh 强度安全上限（Siril 常用范围上限）
    stage7_preview_target_p50_min_ratio: float = 0.55  # 实际 P50 至少达到标定目标的比例，避免上限饱和后仍接受过暗候选
    stage7_preview_target_p50_max_ratio: float = 1.50  # 实际 P50 不得超过标定目标的比例，避免过亮候选进入正式交付
    stage7_target_local_metrics_enabled: bool = True  # Stage 7 用线性源构建核心/弱结构/暗云局部区域并参与候选门控
    stage7_local_core_clip_ratio_max: float = 0.12  # 亮核局部区域允许的裁剪像素比例上限
    stage7_local_faint_snr_min: float = 0.25  # 目标弱结构相对局部背景的最低可分离信噪比
    stage7_local_dark_separation_min: float = 0.001  # 暗云周围亮云与暗结构的最低局部亮度分离
    stage7_stretch_chroma_noise_score_max: float = 0.34  # Stage 7 正式拉伸候选的背景绝对色噪上限
    stage7_stretch_background_mottling_score_max: float = 0.45  # Stage 7 正式拉伸候选的背景斑驳上限
    stage7_stretch_chroma_load_growth_max: float = 1.35  # Stage 7 拉伸后综合色偏差相对背景亮度的最大放大倍数
    stage7_stretch_chroma_load_low_absolute_max: float = 0.05  # 绝对 chroma load 低于此值时豁免极低基线导致的相对增长失真
    stage7_chroma_rescue_enabled: bool = True  # 双候选仅因背景色噪被拒绝时，允许生成背景限定的保亮度色度抑制救援候选
    stage7_chroma_rescue_strength_levels: Tuple[float, ...] = (0.35, 0.55, 0.65)  # 救援按低到高三档抑制背景色度；运行时强制限幅到 0.10–0.75
    stage7_quality_retry_max: int = 2  # Stage 6 去星质量差时最多追加的同源 SyQon 参数重试次数
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
    stage7_starmask_coverage_min_ratio: float = 0.35  # Stage 6 验收：starmask 覆盖相对输入星点覆盖的下限
    stage7_starmask_width_ratio_max: float = 1.80  # Stage 6 验收：starmask 中位宽度相对输入星点宽度上限
    stage7_starless_dynamic_range_min_ratio: float = 0.55  # Stage 6 验收：starless 相对输入的有效动态范围下限，防止去星输出塌缩
    stage7_starless_peak_signal_min: float = 0.006  # Stage 6 验收：低动态范围 starless 的最低峰值信号，低于此值触发参数重试
    stage7_starless_peak_background_ratio_min: float = 4.0  # 极低背景下峰值至少高于背景此倍数，避免固定峰值线误报塌缩
    stage7_starmask_clean_enabled: bool = True  # Stage 6 默认清理星点层，避免背景/星云残差直接回混
    stage7_starmask_background_floor_percentile: float = 55.0  # Stage 6 星点层背景噪声样本百分位
    stage7_starmask_halo_blur_strength: float = 0.35  # Stage 6 非紧致宽 halo/低频残差衰减强度（不再直接模糊星点）
    stage7_starmask_small_star_scale: float = 0.88  # Stage 6 紧致弱星最低保留比例，不再对小星统一降强
    stage7_starmask_nebula_suppression: float = 0.75  # Stage 6 星点层自身低频弥散污染扣除强度
    stage7_starmask_cleanup_noise_sigma: float = 2.5  # Stage 6 多尺度清理的背景噪声检测倍数
    stage7_starmask_compact_retention_min: float = 0.82  # 清理后紧致星核/星翼最低保留比例，低于则回滚
    stage7_starmask_diffuse_residual_ratio_max: float = 0.08  # 清理后弥散残留能量比例硬上限，超限时禁用该星点层
    stage7_conservative_repair_enabled: bool = True  # Stage 6 去星质量差时允许在同一线性输入上调整 SyQon 参数重试
    stage7_skip_unready_starless: bool = True  # Stage6.5 判定不适合去星时，默认跳过去星并进入 review export
    stage7_conservative_asinh_stretch: float = 2.00  # 旧去星前兼容门禁的诊断候选参数，不进入正式 Stage 6
    stage7_ultra_conservative_asinh_stretch: float = 1.65  # 旧去星前兼容门禁的诊断候选参数，不进入正式 Stage 6
    stage7_soft_starless_asinh_stretch: float = 1.35  # 旧去星前兼容参数；当前 Stage 6 不使用预拉伸重试
    stage7_conservative_asinh_offset: float = 0.0025  # 旧去星前兼容门禁的 Asinh 偏移
    stage7_starless_pixel_repair_enabled: bool = True  # 阶段6质量不合格或亮星云 halo advisory 时，对 starless 做残星/halo/背景彩噪局部修复
    stage7_starless_repair_strength: float = 0.68  # 阶段7残星小尺度修复强度
    stage7_starless_halo_repair_strength: float = 0.70  # 阶段7亮星 halo 平滑修补强度
    stage7_starless_chroma_denoise_strength: float = 0.55  # 阶段7背景 chroma noise reduction 强度
    stage6_star_preserve_target_bypass_enabled: bool = True  # 星团/M45 类主体默认绕过去星与 Starless 专用增强
    stage7_starless_repair_max_score_growth: float = 0.00  # 阶段7像素修复不得让综合质量评分变差
    stage7_starless_repair_chroma_reduction_min: float = 0.20  # 背景综合色噪至少下降此比例时允许走专用验收路径
    stage7_starless_repair_chroma_delta_min: float = 0.0005  # 背景综合色噪专用验收所需的最小绝对下降量
    stage8_force_conservative_after_stage7_repair: bool = True  # Stage6 修复被接受后强制 Stage8 使用受限候选，不进入完整增强
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
    """拉伸策略；类名为历史兼容命名，当前用于阶段 7。"""
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
    execution: str = "completed"  # completed / safe_passthrough / skipped
    fallback_used: bool = False  # 仅表示本阶段实际采用回退路径
    upstream_passthrough: bool = False  # 上游安全旁路，不等同本阶段回退
    reason_code: str = ''
    details: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def display_status(self) -> str:
        """Return a structured summary status without inspecting human text."""
        if self.status != "ok":
            return self.status
        if self.fallback_used:
            return "ok_with_fallback"
        if self.execution == "safe_passthrough":
            return "ok_safe_passthrough"
        if self.execution == "skipped":
            return "ok_skipped_optional"
        return self.status
