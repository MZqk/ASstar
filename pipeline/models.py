"""Shared data models for the Starun post-processing pipeline."""

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

    @property
    def label(self) -> str:
        return self.value


class StarSeparationState(str, Enum):
    """Semantic result of Stage 6; passthrough images are never starless."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REVIEW_REQUIRED = "review_required"
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
            "schema": "starun.input-profile.v1",
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
    stage2_processing_mode: str = "auto"  # 阶段2处理方式：auto 自动裁切；preserve 安全保留原图
    stage2_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage2_base_crop_enabled: bool = False  # 显式启用后先做受中心保护约束的对称基础裁边
    stage2_base_crop_margin: float = 0.02  # 每边基础裁切比例；仅在 stage2_base_crop_enabled 时生效
    stage2_field_rotation_detection_enabled: bool = True  # 检测经纬仪场旋形成的边缘连通低覆盖噪声区
    stage2_field_rotation_max_passes: int = 2  # 场旋覆盖自动裁切总轮次；最多两轮，第二轮必须复检
    stage2_field_rotation_noise_ratio_min: float = 1.35  # 场旋低覆盖区相对中心的亮度噪声硬门槛
    stage2_field_rotation_chroma_ratio_min: float = 1.20  # 场旋低覆盖区相对中心的色噪硬门槛
    stage2_color_edge_cleanup_enabled: bool = True  # 自动裁切后允许执行保尺寸的边缘色偏清理
    stage2_level_artifact_window: int = 81  # 水平伪影检测窗口，必须为奇数
    stage2_edge_black_improvement_min: float = 0.003  # 自适应裁边每轮所需最小黑边改善
    stage2_preserve_review_edge_black_max: float = 0.30  # 保留模式超过此残余黑边比例时标记复核
    stage2_edge_cast_absolute_max: float = 0.010  # 边缘色偏检测/清理的绝对下限
    stage2_edge_cast_detect_ratio_max: float = 2.8  # 边缘相对中心色偏检测倍率
    stage2_edge_cast_cleanup_ratio_max: float = 2.6  # 色边清理候选相对中心色偏验收倍率
    stage2_color_cleanup_strip_ratio: float = 0.025  # 色偏清理候选的边缘带宽比例
    stage2_edge_black_target: float = 0.03  # 阶段2自动黑边裁切目标，后续阶段不再临时裁黑边
    stage2_adaptive_edge_crop_max_passes: int = 3  # 阶段2自适应黑边裁切最大迭代次数
    stage2_adaptive_edge_crop_max_extra: float = 0.035  # 阶段2单次自适应额外裁切上限
    stage2_guard_band_pixels: int = 3  # 阶段2最终裁切后护带检查的像素宽度，清除残余暗边/色偏
    stage2_center_protect_area_ratio: float = 0.70  # 阶段2必须保留的原图中心保护区面积比例，防止自动检测过裁

    # 阶段 3: 背景提取 (RBF)
    stage3_processing_mode: str = "auto"  # 阶段3处理方式：auto 自动评估背景；preserve 安全保留背景
    stage3_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage3_backend_policy: str = "auto_chain"  # auto_chain/graxpert_only/builtin_only
    stage3_gate_profile: str = "output_first"  # output_first/balanced/strict
    stage3_plugin_fallback_enabled: bool = True  # 自动链中允许 GraXpert 等插件候选作为回退
    stage3_compound_candidate_enabled: bool = True  # 允许 Polynomial→残差 RBF 复合候选
    stage3_candidate_attempt_limit: int = 0  # 已启用候选最大尝试数；0 表示不限制
    bg_samples: int = 20            # subsky 兼容采样密度参数；实际样点坐标由 Stage3 自定义安全样点固定
    bg_tolerance: float = 1.0       # subsky 兼容容差参数；不得绕过 -existing 自动重建样点
    bg_smooth: float = 0.5          # 背景模型平滑度；增大更平滑，过大会吞掉大尺度弱信号
    bg_quality_gate_enabled: bool = True  # Stage3 过程验证总开关；生产门禁由留出天空与目标保真证据执行
    stage3_conditional_decision_enabled: bool = True  # 阶段3先诊断再决定 apply/skip/review，禁止无证据直接扣背景
    stage3_deterministic_auto_apply_enabled: bool = True  # 无视觉顾问时仅对高置信梯度使用离线确定性 apply
    stage3_apply_confidence_min: float = 0.75  # 外部/视觉背景建议获准执行所需最低置信度
    stage3_gradient_skip_max: float = 0.045  # 旧无 InputProfile 调用方兼容；生产决策不读取该阈值
    stage3_dirty_skip_max: float = 0.16  # 旧无 InputProfile 调用方兼容；生产决策不读取该阈值
    stage3_gradient_apply_min: float = 0.08  # 旧兼容/诊断路由参数；生产授权使用过程证据
    stage3_dirty_apply_min: float = 0.18  # 旧兼容/诊断路由参数；生产授权使用过程证据
    stage3_diffuse_auto_apply_enabled: bool = False  # 旧决策兼容字段；生产场景由真实天空与复杂度限制裁决
    stage3_safe_sample_target_count: int = 40  # 自定义安全背景样点目标数；按低信号、低纹理与空间覆盖筛选
    stage3_safe_sample_min_count: int = 12  # 低于该数量或覆盖不足时禁止 subsky -existing，避免稀疏误拟合
    stage3_safe_sample_patch_radius: int = 12  # 样点安全评估半径；用于排除恒星、星云纹理和裁剪暗边
    stage3_safe_sample_brightness_quantile_max: float = 0.70  # 相对低频背景模型允许的亮度残差分位上限
    stage3_safe_sample_texture_quantile_max: float = 0.55  # 候选局部纹理分位上限，越低越保守
    stage3_compound_min_sample_count: int = 12  # Polynomial→残差RBF复合候选所需的已审计安全样点下限
    stage3_compound_fit_min_count: int = 8  # 复合候选确定性拟合集最小样点数
    stage3_compound_validation_min_count: int = 4  # 复合候选冻结验证集最小样点数
    stage3_compound_validation_ratio: float = 0.25  # 已审计样点中冻结且永不参与拟合的比例
    stage3_compound_score_abs_improvement_min: float = 0.03  # 相对最佳安全单阶段候选所需绝对评分改善
    stage3_compound_score_rel_improvement_min: float = 0.10  # 相对最佳安全单阶段候选所需相对评分改善
    stage3_compound_validation_improvement_min: float = 0.10  # 冻结验证集低频残差跨度所需改善比例
    stage3_compound_zero_point_abs_max: float = 0.01  # 第二次RBF允许的验证集天空零点绝对漂移
    stage3_compound_zero_point_rel_max: float = 0.15  # 第二次RBF允许的验证集天空零点相对漂移
    stage3_pattern_routing_enabled: bool = True  # 将条纹/walking noise 与低频天空梯度分流，禁止误用 DBE 修复
    stage3_pattern_score_min: float = 0.55  # 条纹/方向性噪声进入独立分流的最低综合评分
    stage3_walking_noise_score_min: float = 0.50  # 对角 walking-noise 进入独立分流的最低评分
    stage3_pattern_score_growth_max: float = 0.12  # 背景候选允许新增的方向性噪声评分上限

    # 阶段 5: 线性整理 / 反卷积 / 降噪
    stage5_processing_mode: str = "auto"  # auto 正常处理；preserve 复制 Stage4 线性结果并保留诊断
    stage5_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage5_denoise_backend_policy: str = "auto_chain"  # auto_chain/multiscale_only/siril_only/cosmic_clarity_only
    stage5_low_noise_auto_skip_enabled: bool = True  # 低噪声输入自动跳过额外降噪候选
    denoise_enabled: bool = True  # 自动模式允许 Stage 5 自行通过低噪声/质量门决定是否实际降噪
    denoise_mod: float = 0.35       # 降噪强度参数（0~1），越大降噪越强
    denoise_safety_max: float = 0.55  # 降噪强度安全上限，防止细节被抹平
    stage5_multiscale_denoise_enabled: bool = True  # 启用噪声模型驱动的亮度/对立色度多尺度确定性候选
    stage5_multiscale_denoise_strength: float = 0.72  # 多尺度软阈值与回混强度，受质量门限制
    stage5_multiscale_detail_retention_min: float = 0.82  # 主体高频细节最低保留比例
    stage5_multiscale_noise_reduction_min: float = 0.05  # 非低噪输入所需最低背景噪声下降比例
    stage5_denoise_chroma_noise_growth_max: float = 1.05  # 所有 Stage5 降噪候选允许的背景色噪增长倍数
    stage5_deconvolution_mode: str = "auto"  # 阶段5显式模式：auto/graxpert_rl/rl/off；auto 仍由安全回退链决定
    stage5_deconvolution_enabled: bool = True  # Stage5 是否在线性降噪前执行 GraXpert/RL 反卷积
    stage5_graxpert_deconvolution_enabled: bool = True  # Stage5 是否优先尝试本地 GraXpert 对象反卷积；关闭后直接使用 RL
    stage5_rl_maxstars: int = 200  # RL 反卷积 PSF 找星数量上限
    stage5_rl_psf_kernel_size: int = 33  # RL 反卷积 makepsf kernel size
    stage5_rl_iters: int = 8  # RL 反卷积迭代次数，过高易放大噪声和星环
    stage5_rl_alpha: float = 3000.0  # RL TV 正则 alpha，越高越保守
    stage5_rl_gdstep: float = 0.0005  # RL 梯度下降步长
    stage5_rl_stop: float = 0.001  # RL 提前停止阈值
    stage5_graxpert_deconv_strength: float = 0.30  # 本地 Object Deconvolution 模型可用时的 GraXpert 强度
    stage5_graxpert_guard_retry_strength: float = 0.25  # GraXpert 局部星点门失败后唯一一次降强度重试值
    stage5_deconv_bg_std_growth_max: float = 1.38  # 反卷积背景标准差最大增长倍率
    stage5_deconv_chroma_growth_max: float = 1.15  # 反卷积背景色噪最大增长倍率
    stage5_deconv_chroma_ratio_growth_max: float = 1.35  # 色噪/背景比最大增长倍率
    stage5_deconv_dirty_delta_max: float = 0.06  # 脏背景评分最大绝对增量
    graxpert_object_model_path: str = ""  # 当前任务显式 GraXpert Object Deconvolution ONNX 文件；空值走离线自动发现
    optional_color_transform_enabled: bool = False  # 显式授权宽带/Stage 9 可选调色插件；双窄带哈勃色由 Stage 8 独立开关控制
    workflow_plugin_probe_enabled: bool = False  # Probe broad workflow plugin commands only when explicitly enabled; stage8 has a narrow safe SASP probe
    stage4_processing_mode: str = "auto"  # 阶段4处理方式：auto 解析并校色；preserve 安全保留输入颜色
    stage4_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage4_offline_fallback_mode: str = "auto_local_reference"  # Gaia 不可用时：自动非物理局部参考，或 preserve 原样保色继续
    stage4_auto_reference_global_white_enabled: bool = False  # 首发只影子评估恒星集合伪白参考；显式开启后仍受独立质量门约束
    stage4_auto_reference_background_sample_target: int = 40  # Stage4 独立安全背景样点目标数
    stage4_auto_reference_background_sample_min: int = 16  # 自动背景候选所需最少安全样点
    stage4_auto_reference_holdout_ratio: float = 0.25  # 背景与星体证据冻结留出比例
    stage4_auto_reference_background_error_min: float = 0.01  # 低于该背景 RGB 绝对色差时禁止启发式修正
    stage4_auto_reference_background_improvement_min: float = 0.10  # 留出背景色差所需相对改善
    stage4_auto_reference_star_min_objects: int = 16  # 恒星集合伪白参考所需独立连通星体下限
    stage4_auto_reference_star_ratio_mad_max: float = 0.12  # 星群 log 色比最大稳健离散度
    stage4_auto_reference_star_saturation_ratio_max: float = 0.10  # 单个星体允许的饱和像素比例
    stage4_auto_reference_gain_limit: float = 1.10  # 伪白参考全局单通道增益上下限
    stage4_auto_reference_star_improvement_min: float = 0.10  # 留出星群中性色误差所需相对改善
    stage4_auto_reference_highlight_clip_growth_max: float = 0.002  # 自动参考候选允许的高光裁剪增长
    stage4_auto_reference_black_clip_growth_max: float = 0.002  # 自动参考候选允许的黑位裁剪增长
    stage4_auto_reference_gradient_growth_max: float = 1.05  # 留出背景梯度允许增长倍率
    stage4_auto_reference_texture_growth_max: float = 1.10  # 留出背景纹理允许增长倍率
    stage4_auto_reference_target_chroma_drift_max: float = 0.08  # 固定主体掩膜 P95 色度漂移上限
    stage4_pcc_fallback_enabled: bool = True  # SPCC 失败后允许 PCC 安全回退
    stage4_narrowband_degraded_pcc_enabled: bool = True  # 窄带 SPCC 失败后允许降级 PCC 并强制复核
    stage4_header_guided_platesolve_enabled: bool = True  # 允许可信 FITS Header 中心坐标增加解析候选
    stage4_filter_hint: str = "auto"  # FITS FILTER 缺失时的显式滤镜提示：auto/no_filter/seestar_lp/dual_narrowband
    stage4_platesolve_enabled: bool = True  # 阶段4默认执行 platesolve -noflip，解算时保留原始图像方向
    stage4_auto_geometry_enabled: bool = True  # 高置信 FITS/设备几何可用于 platesolve；显式环境覆盖始终优先
    stage4_auto_geometry_confidence_min: float = 0.85  # 自动使用设备几何所需最低置信度
    stage4_auto_geometry_scale_residual_max: float = 0.05  # 解算 WCS 像素比例相对预测值最大偏差，超限回滚并禁止 SPCC/PCC
    stage4_spcc_enabled: bool = True  # Stage4 物理校色首选 SPCC；异常时宽带回退 PCC，双窄带只允许降级 PCC 基础校色并强制复核
    stage4_spcc_timeout_sec: int = 300  # SPCC 单次独立 Siril 子进程超时；不在同一进程中重试
    stage4_spcc_online_unverified_timeout_sec: int = 90  # 仅有在线端点可达证据时的 SPCC 有界预算；本地 localgaia 不受此上限影响
    stage4_spcc_osc_sensor: str = ""  # 为空时按 FITS 设备画像自动选择；无法确认则禁止误套传感器响应
    stage4_spcc_osc_filter: str = ""  # 为空时按设备画像和 FITS 滤镜提示选择已校验响应
    stage4_spcc_white_ref: str = "Average Spiral Galaxy"  # SPCC 物理白参考
    stage4_spcc_limit_magnitude: float = 10.5  # 限制 SPCC 测光星数与查询成本
    stage4_spcc_narrowband_r_wavelength_nm: float = 656.28  # 双窄带物理校色 R/Ha 中心波长
    stage4_spcc_narrowband_r_bandwidth_nm: float = 20.0  # 双窄带物理校色 R/Ha 带宽
    stage4_spcc_narrowband_g_wavelength_nm: float = 500.70  # 双窄带物理校色 G/OIII 中心波长
    stage4_spcc_narrowband_g_bandwidth_nm: float = 30.0  # 双窄带物理校色 G/OIII 带宽
    stage4_spcc_narrowband_b_wavelength_nm: float = 500.70  # HOO 的 B 与 G 使用同一 OIII 中心波长
    stage4_spcc_narrowband_b_bandwidth_nm: float = 30.0  # HOO 的 B 与 G 使用同一 OIII 带宽
    stage4_narrowband_normalization_enabled: bool = True  # 仅生成隔离的 HOO 艺术派生图，不作为后续物理主链输入
    stage4_nbn_mapping_confidence_min: float = 0.85  # 低于此置信度时保留输入，不猜测窄带通道含义
    stage4_nbn_strength: float = 0.55  # HOO 通道跨度与背景中和的保守混合强度
    stage4_nbn_gain_limit: float = 1.08  # 单通道最大归一化增益/衰减范围
    stage4_nbn_line_ratio_drift_max: float = 0.12  # Ha/OIII 信号比例最大相对漂移
    stage4_pcc_timeout_sec: int = 180  # SPCC 异常后的 Gaia PCC 只尝试一次；双窄带结果仅作降级基础校色，达到该秒数即终止
    stage4_pcc_quality_gate_enabled: bool = True  # SPCC/PCC 物理校色候选必须通过目标感知质量门，否则回到 pre_pcc
    stage4_pcc_channel_gain_ratio_max: float = 10.0  # SPCC/PCC 三通道相对增益最大跨度，超限仍回滚候选
    stage4_pcc_emission_balance_gain_ratio_max: float = 4.0  # 发射星云仅在背景色差显著改善且其他门均安全时允许的大增益跨度
    stage4_pcc_clip_growth_max: float = 0.005  # PCC 相对 pre_pcc 允许新增的高光裁剪比例
    stage4_pcc_star_temperature_ratio_min: float = 0.45  # 校色后可测恒星综合色温中位数相对输入的最低比例
    stage4_pcc_star_temperature_ratio_max: float = 2.20  # 校色后可测恒星综合色温中位数相对输入的最高比例
    stage4_pcc_background_color_delta_max: float = 0.22  # 背景变得更失衡时允许的最大归一化 RGB 色差
    stage4_pcc_target_color_drift_max: float = 0.40  # 普通主体归一化 RGB 色度允许的最大漂移
    stage4_pcc_emission_target_color_drift_max: float = 0.45  # 发射星云保留真实线发射主色时允许的更宽主体色度漂移
    stage4_local_star_wb_enabled: bool = True  # SPCC/PCC 均失败或拒绝时，仅在恒星软遮罩内做保守色彩恢复
    stage4_local_star_wb_min_pixels: int = 32  # 本地星点白平衡所需的最小白参考像素数
    stage4_local_star_wb_gain_limit: float = 1.20  # 恒星软遮罩内单通道增益限制，避免替代光度校准
    stage4_local_star_mask_radius: int = 2  # 恒星样本向星翼扩展的软遮罩半径，限制校色只影响局部
    stage4_local_star_mask_coverage_max: float = 0.12  # 恒星软遮罩最大覆盖率，超限时保留输入颜色
    aberration_api_enabled: bool = False  # Disabled by default: API path may fail in siril-cli thread ownership context

    # 阶段 7: Starless 主体拉伸
    stage7_processing_mode: str = "auto"  # auto 使用预览标定；manual 使用签名参数并重建对应亮度契约
    stage7_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage7_rendition_intent: str = "vivid_safe"  # vivid_safe 默认鲜艳安全出图；balanced/conservative 保留更自然或更保守的呈现
    stage7_forced_delivery_enabled: bool = True  # 自动回退耗尽后，保留技术完整的最佳失败候选作为复核诊断，不参与正式交付
    stage7_candidate_policy: str = "auto_display90"  # auto_display90/auto_dual/candidate_a_only/candidate_b_only/display90_only
    stage7_display90_strength: float = 0.90  # GUI linked 显示曲线正式候选的保留强度，运行时钳制 0.50–0.95
    stage7_display90_reference_chroma_load_ratio_max: float = 1.05  # 已认证窄带 Display90 相对真实 GUI D 背景色度负载的最大比值
    stage7_display90_reference_chroma_load_absolute_max: float = 0.30  # GUI D 参考匹配豁免仍允许的候选背景绝对色度负载硬上限
    stage7_vivid_subject_chroma_enabled: bool = True  # 仅对冻结主体 ROI 的低频色度做保亮度、保余量增强
    stage7_chroma_rescue_max_attempts: int = 3  # 截断固定色度救援安全阶梯；0 表示不尝试
    asinh_stretch: float = 3.0      # Asinh 拉伸强度（越大整体越亮）
    asinh_offset: float = 0.001     # Asinh 偏移，影响暗部起拉位置
    ghs_shadowsclip: float = -2.8   # GHS 阴影裁剪，控制黑场压暗程度（回退方案用）
    ghs_stretchamount: float = 2.0  # GHS 拉伸量（回退方案用）
    stage7_iterative_masked_mtf_enabled: bool = True  # 保星目标旁路下按严格目标条件以迭代蒙版 MTF 替换 cand_a
    stage7_iterative_masked_mtf_iterations: int = 16  # 迭代蒙版 MTF 轮数，运行时安全限幅 8–32
    stage7_dual_stage_mtf_ghs_enabled: bool = True  # 弱信号 Starless 星系在可信 ROI 下以 MTF+GHS 替换 cand_a
    stage7_dual_stage_weak_snr_max: float = 8.0  # Dual-stage 仅用于低于该冻结 ROI 代理 SNR 的星系
    stage7_dual_stage_subject_p90_min: float = 0.20  # 低 SNR 星系主体 P90 的最低目标
    stage7_dual_stage_subject_p90_max: float = 0.26  # 接近 SNR 上限时星系主体 P90 的最高目标
    stage7_dual_stage_ghs_b: float = 5.0  # Dual-stage GHS 局部强度参数，运行时限幅 2–8
    stage7_dual_stage_ghs_d_min: float = 0.5  # Dual-stage GHS 确定性搜索下界（ln(D+1) 语义）
    stage7_dual_stage_ghs_d_max: float = 12.0  # Dual-stage GHS 确定性搜索上界（ln(D+1) 语义）
    stage7_dual_stage_ghs_search_steps: int = 47  # Dual-stage GHS 固定网格采样数，运行时限幅 9–97
    stage7_conditional_lut_max_derivative: float = 5000.0  # 新增共享 LUT 的最大离散导数硬上限

    # 阶段 8: 星云饱和度
    stage8_processing_mode: str = "auto"  # auto/limited/background_only/preserve，仅作为上限
    stage8_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage8_nebula_saturation_enabled: bool = True  # 独立控制星云饱和度子步骤
    stage8_background_denoise_enabled: bool = True  # 独立控制背景降噪子步骤
    stage8_faint_nebula_boost_enabled: bool = True  # 独立控制暗弱星云提升子步骤
    stage8_nebula_contrast_enabled: bool = True  # 独立控制主体对比子步骤
    stage8_masked_unsharp_enabled: bool = True  # 独立控制蒙版锐化子步骤
    stage8_quality_retry_max: int = 1  # 质量拒绝后的保守重跑次数；安全范围 0–1
    nebula_saturation: float = 0.4  # 去星图饱和度增强幅度
    nebula_bg_factor: int = 1       # 饱和度算法背景抑制系数（Siril `satu` 第二参数）
    stage8_masked_enhancement_enabled: bool = True  # 阶段8默认使用 Starless soft mask 分区增强，保护核心与背景
    stage8_local_adjustment_engine_enabled: bool = True  # 使用版本化本地曲线/蒙版配方，候选未过门则保留配方前图像
    stage8_local_curve_opacity: float = 0.30  # 本地微曲线最大混合比例，Stage8 仍受背景与核心质量门限制
    stage8_core_protection_strength: float = 0.92  # 阶段8亮核心回混原图强度，越高越保护核心不过曝
    stage8_background_denoise_strength: float = 0.0  # 默认不重复 Stage5/7/10 降噪；仅保留显式兼容覆盖
    stage8_faint_nebula_boost_max: float = 0.08  # 阶段8外围暗云气最大提亮强度
    stage8_nebula_contrast_max: float = 0.10  # 阶段8星云主体局部对比最大强度
    stage8_masked_unsharp_amount_max: float = 0.12  # 阶段8非核心星云区域轻锐化上限
    stage8_blue_precontrol_strength: float = 0.55  # 阶段8增强前信号区蓝偏预抑制强度，减少后置 ccm 补救
    stage8_bg_std_growth_max: float = 1.08  # 阶段8背景噪声增长上限
    stage8_texture_artifact_growth_max: float = 1.25  # 阶段8纹理伪影评分增长上限
    stage8_limited_saturation_max: float = 0.05  # 亮星云 halo 中风险区受限候选的饱和度硬上限
    stage8_limited_core_exclusion_expand: int = 8  # 受限候选在亮核硬掩膜外追加扩张像素，曲线/饱和度/弱信号提升均不得进入
    stage8_limited_halo_texture_growth_max: float = 1.05  # 受限候选星周环带纹理相对增长上限
    stage8_limited_halo_texture_delta_max: float = 0.00075  # 环带绝对增长低于此值时豁免比例误报
    stage8_dualband_palette_enabled: bool = True  # 确认 Ha/OIII 的双窄带默认在 Stage 8 执行目标首选哈勃伪色
    stage8_dualband_palette_selection: str = "auto"  # auto 按冻结目标选择；也可显式指定 HSO/SHO/OSH/OHS/HOS/HOO
    stage8_dualband_palette_strength: float = 0.85  # 确认 Ha/OIII Starless 主体的目标首选调色混合强度
    stage8_dualband_palette_luma_drift_max: float = 0.005  # 双窄带伪色候选相对调色前亮度 P95 最大漂移
    stage8_dualband_palette_clip_growth_max: float = 0.002  # 双窄带伪色候选允许的新增通道裁剪占比
    stage8_dualband_palette_quality_warning_tolerance: float = 0.50  # 亮度/裁剪超出基础门限的此比例内仍接受，仅记录质量提醒

    # 阶段 9: 星点混合
    stage9_processing_mode: str = "auto"  # auto 或 preserve_with_stars（仅可信含星源）
    stage9_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    star_intensity: float = 1.05  # 星点层静态主强度上限；自动调参和上游安全缩放可继续降低
    stage9_fallback_intensity_cap: float = 0.95  # Stage 9 回退强度阶梯的安全上限
    stage9_fallback_retry_max: int = 3  # 安全回星强度阶梯最多尝试档数
    stage9_fallback_intensity_floor: float = 0.40  # 安全回星强度阶梯最低允许值
    stage9_fallback_intensity_levels: Tuple[float, ...] = (0.75, 0.55, 0.40)  # 主候选拒绝后逐档降低回星强度
    stage9_targeted_recovery_enabled: bool = True  # 按失败类型启用分组软 PSF 与局部色差定向恢复，不改变正式质量门
    stage9_targeted_recovery_retry_max: int = 3  # 单个主候选最多执行的有界定向恢复次数
    stage9_unscreen_candidate_enabled: bool = True  # 默认让同域 Unscreen 星点幅度与线性差分星层竞争，所有既有画质门保持生效
    stage9_unscreen_denominator_floor: float = 0.08  # Unscreen 的 1-B 各通道安全下限，低余量像素回退可信星层
    stage9_unscreen_reliable_support_min: float = 0.80  # 紧致星点支持层内可靠 Unscreen 像素覆盖下限
    stage9_unscreen_peak_max: float = 0.95  # Unscreen 仅向上恢复星点幅度时允许的峰值上限
    stage9_unscreen_roundtrip_relative_improvement_min: float = 0.10  # 相对基线所需的同域闭环 RGB MAE 改善下限
    stage9_unscreen_roundtrip_absolute_improvement_min: float = 0.005  # 相对基线所需的同域闭环 RGB MAE 绝对改善下限
    stage9_unscreen_chroma_regression_max: float = 0.02  # Unscreen 相对基线允许的星色色度中位误差回退上限
    stage9_unscreen_recovery_regression_max: float = 0.02  # Unscreen 相对基线允许的弱星/全星/孔径恢复率下降上限
    stage9_unscreen_wing_regression_max: float = 0.03  # Unscreen 相对基线允许的星翼恢复率下降上限
    stage9_unscreen_fwhm_regression_max: float = 0.05  # Unscreen 星径偏离 1.0 的程度相对基线允许回退上限
    stage9_psf_size_gate_enabled: bool = True  # 用 matched-domain 同星 FWHM 正式验收 Screen/Unscreen 候选
    stage9_psf_fwhm_ratio_min: float = 0.93  # 正式回星后同星 FWHM 相对源图的下限
    stage9_psf_fwhm_ratio_max: float = 1.10  # 正式回星后同星 FWHM 相对源图的上限
    stage9_psf_fwhm_ratio_uncertainty_floor: float = 0.002  # FWHM 比例 95% 测量容差的最小有效值
    stage9_psf_fwhm_ratio_uncertainty_max: float = 0.020  # FWHM 比例临界豁免允许的最大 95% 测量容差
    stage9_psf_review_fwhm_ratio_max: float = 1.65  # 正式候选全拒后，含星复核候选允许的独立 FWHM 上限；不改变正式门
    stage9_psf_recovery_target_min: float = 0.97  # 通过硬门后仍低于此软目标时逐级补回真实星翼
    stage9_psf_recovery_target_max: float = 1.05  # 任一可测星组高于此软上限时禁止统一补翼
    stage9_psf_selective_wing_enabled: bool = True  # 统一补翼触顶后，仅对仍偏小的同星样本补回源确认外翼
    stage9_psf_selective_wing_target_ratio: float = 1.08  # 低于此同星 FWHM 比例的未饱和星才进入选择性补翼
    stage9_psf_selective_wing_strength_max: float = 1.15  # 选择性补翼相对 raw Unscreen 的最大幅度；新增外翼在实际 Screen 域钳于局部半高线以下
    stage9_source_autostretch_wing_reference_enabled: bool = True  # 用同源 linked autostretch 只校准肉眼可见低亮外翼，不替换 Stage7 与 FWHM 硬门
    stage9_source_autostretch_wing_floor_fraction: float = 0.05  # 外翼形态参考的局部峰值下限；低于此值不补，避免背景噪声被当作星翼
    stage9_source_autostretch_wing_target_ratio: float = 1.03  # 5%/10% 低亮轮廓任一外径低于同源自动拉伸的此比例时补翼；逐像素输出不超过源参考
    stage9_source_autostretch_wing_radius_max: int = 10  # 普通星同源可见外翼的独立半径上限；半高核仍由 <=6 px 原支持契约约束
    stage9_psf_min_sample_count: int = 16  # FWHM 正式闭环所需的最少孤立同星样本数
    stage9_psf_support_radius_max: int = 6  # 源 FWHM 驱动星翼支持的最大半径
    stage9_psf_support_retry_pixels: int = 2  # 星径偏小时允许补回的额外真实星翼像素
    stage9_stage5_bright_star_completion_enabled: bool = True  # 用 Stage5 冻结 Siril 星表补全被普通紧致/FWHM 目录排除的大亮或饱和星
    stage9_stage5_bright_star_fwhm_min: float = 8.0  # 非饱和补全星所需的 Stage5 几何 FWHM 下限；饱和星不受此下限限制
    stage9_stage5_bright_star_support_radius_max: int = 12  # Stage5 大亮/饱和星的独立源几何支持半径上限
    stage9_stage5_bright_star_match_radius: float = 3.0  # Stage5 星表与现有 Stage9 目录判重半径
    stage9_sasp_star_stretch_enabled: bool = True  # Stage9 直接调用随包 SASP Star Stretch，无需 Siril 命令探测
    stage9_sasp_star_stretch_amount: float = 3.0  # SASP Star Stretch 无头安全强度（0.50–5.00）
    stage9_nb_to_rgb_stars_enabled: bool = True  # 仅对已冻结且确认的 Ha/OIII 星层自动调用 NB to RGB Stars
    stage9_nb_to_rgb_stars_ratio: float = 0.30  # SASP NB to RGB Stars 的 Ha:OIII 混合比
    stage9_starmask_stretch_enabled: bool = True  # 阶段9像素回混前默认把线性 starmask 独立 Asinh 拉伸到非线性域
    stage9_starmask_adaptive_stretch_enabled: bool = True  # 根据星点层有效信号分布反解 Asinh 强度，而不是固定套用单一参数
    stage9_compact_starmask_enabled: bool = True  # 启用 normal/strict 紧凑支持候选与更严格支持恢复；不再控制拉伸前像素缩星
    stage9_starmask_pre_stretch_compact_enabled: bool = False  # 内置星掩膜拉伸前是否按当前 normal/strict 支持缩星；GUI 专家参数默认关闭
    stage9_star_color_repair_enabled: bool = True  # 用不可变线性含星参考修复星层核心/星翼色度，候选失败即保留原层
    stage9_star_color_repair_strength: float = 0.72  # 参考色度回混强度；亮星核心自动降低以保留真实星色和通量
    stage9_star_color_support_ratio_max: float = 0.12  # 星色修复允许影响的最大画面覆盖率
    stage9_star_color_improvement_min: float = 0.01  # 候选所需最小中位色度误差改善
    stage9_star_color_post_chroma_error_max: float = 0.22  # 拉伸及回混前最终星层相对参考的中位色度误差上限
    stage9_star_color_post_validation_enabled: bool = True  # 独立控制最终星层色度安全门；关闭普通质量门时仍默认强制执行
    stage9_source_star_detail_percentile: float = 98.0  # 从原始含星图提取独立星核目录的局部细节百分位
    stage9_source_component_density_max: float = 2500.0  # 独立星表紧致组件密度上限；伴随单像素噪点证据时自适应收紧或 fail-closed
    stage9_source_single_pixel_ratio_max: float = 0.20  # 独立星表单像素组件比例上限，单项超限即自适应收紧或 fail-closed
    stage9_star_reference_sigma: float = 5.0  # 原始 starmask 星点目录检测阈值，使用背景加该倍数噪声标准差
    stage9_compact_weak_star_retention_min: float = 0.80  # compact 支持层必须保留的弱星组件数量比例下限
    stage9_mixed_star_peak_ratio_min: float = 4.0  # 亮星组峰值中位数相对弱星组的倍率达到该值时启用多锚点曲线
    stage9_mixed_star_weak_count_min: int = 20  # 启用混合星场多锚点曲线所需的最少弱星组件数
    stage9_mixed_star_bright_count_min: int = 3  # 启用混合星场多锚点曲线所需的最少亮星组件数
    stage9_starmask_asinh_stretch: float = 2.0  # 自适应关闭时的初始提案；仍由四锚点输出与变化覆盖硬约束钳制
    stage9_starmask_asinh_offset: float = 0.001  # 阶段9 starmask Asinh 偏移
    stage9_starmask_asinh_stretch_max: float = 1000.0  # 四锚点输出约束的反解上限；统计不可用时 Stage 9 在回混前 fail-closed
    stage9_starmask_faint_target: float = 0.26  # 多锚点曲线弱星中位目标亮度
    stage9_starmask_mid_target: float = 0.50  # 多锚点曲线中亮星目标亮度
    stage9_starmask_bright_target: float = 0.75  # 多锚点曲线亮星目标亮度
    stage9_starmask_peak_target: float = 0.90  # 多锚点曲线极亮星上限，保留高光和星色余量
    stage9_starmask_output_adequacy_min: float = 0.50  # 实测星层四锚点相对冻结目标的最低比例；不足时插件不得进入正式候选
    stage9_starmask_chroma_regularization_enabled: bool = True  # 多锚点拉伸时用邻域星色约束微弱星翼，避免把单像素通道噪声放大成蓝紫色块
    stage9_starmask_faint_chroma_max: float = 0.35  # 微弱星翼允许的最大通道跨度比例，优先抑制低信号伪色
    stage9_starmask_bright_chroma_max: float = 0.60  # 亮星核心允许的最大通道跨度比例，保留真实星色同时限制极端色边
    stage9_starmask_predicted_change_ratio_max: float = 0.30  # 拉伸求解时预测的显著变化覆盖上限，给正式门控预留余量
    stage9_quality_gate_enabled: bool = True  # 阶段9保存前验收星点回混，高风险候选回滚并降强度重试
    stage9_highlight_clip_ratio_max: float = 0.015  # 阶段9回混后高光裁剪占比上限
    stage9_highlight_clip_growth_max: float = 0.006  # 阶段9相对本轮不可变 remix base 的高光裁剪增长上限
    stage9_bright_pixel_growth_max: float = 0.025  # 阶段9亮像素覆盖增长上限，限制星点膨胀/光晕
    stage9_background_lift_max: float = 0.010  # 阶段9暗背景中位抬升上限，限制 starmask 污染
    stage9_background_mottling_growth_max: float = 1.35  # 阶段9相对本轮不可变 remix base 的低频背景斑驳增长倍数上限
    stage9_mottling_exemption_changed_pixel_ratio_max: float = 0.12  # 仅局部变化时才允许低绝对斑驳豁免，防止大面积星点/背景污染漏过门控
    stage9_changed_pixel_ratio_max: float = 0.35  # 阶段9显著变化像素占比上限
    stage9_darkening_ratio_max: float = 0.005  # 阶段9异常变暗像素占比上限
    stage9_weak_star_recovery_ratio_min: float = 0.70  # 候选相对 remix base 至少恢复的弱星组件数量比例
    stage9_star_recovery_ratio_min: float = 0.75  # 候选相对 remix base 至少恢复的全部星点组件数量比例
    stage9_catalog_star_visibility_contrast_min: float = 0.002  # matched-display 源目录星及候选星的最低局部亮度对比度
    stage9_bright_star_visibility_ratio_min: float = 0.90  # 候选中源目录亮星必须保持可见的最低数量比例
    stage9_weak_star_screen_intensity_min: float = 0.55  # 弱星 Screen 强度下限；亮星仍可按 fallback 梯级降至 0.40
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
    stage9_star_positive_delta_window_recovery_ratio_min: float = 0.75  # 7x7 星点正增量窗口最低恢复数量比例
    stage9_star_wing_recovery_ratio_min: float = 0.65  # 参考星翼最低恢复数量比例
    stage9_residual_dark_hole_ratio_max: float = 0.15  # 回星后星点邻域残余暗坑像素比例上限
    stage9_hollow_structure_delta_min: float = 0.05  # 检测新增环状/空心结构的星层最小亮度变化
    stage9_new_hollow_structure_area_max: int = 64  # 新增闭合环内部允许的最大空心面积，超限候选回滚
    # 阶段 10: 最终饱和度
    stage10_processing_mode: str = "auto"  # auto 正常末端处理；preserve 保留已验证的 Stage9 含星源
    stage10_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage10_final_denoise_enabled: bool = True  # 允许末端降噪；安全跳过条件始终优先
    stage10_final_saturation_enabled: bool = True  # 允许最终饱和度微调
    stage10_denoise_backend_policy: str = "auto_chain"  # auto_chain/cosmic_only/scunet_only
    stage10_quality_repair_enabled: bool = True  # 最终质量门失败时允许有界修复候选
    final_saturation: float = 0.15  # 导出前最终全图饱和度微调量
    final_bg_factor: int = 1        # 最终饱和度背景抑制系数（Siril `satu` 第二参数）
    stage10_chroma_focus_score_min: float = 0.34  # Stage10 综合色噪达到该值时优先保护亮度、只处理色度
    stage10_separate_chroma_score_min: float = 0.70  # Stage10 严重通道色噪启用逐通道降噪的门限
    stage10_full_bg_std_min: float = 0.018  # Stage10 色噪伴随亮度背景噪声时改用 full 的 bg_std 门限
    stage10_full_mottling_score_min: float = 0.45  # Stage10 色噪伴随背景斑驳时改用 full 的斑驳门限
    stage10_final_denoise_strength: float = 0.28  # 末端降噪统一强度；适用于 CosmicClarity/SCUNet 回退链并受安全钳制
    stage10_star_protection_coverage_max: float = 0.35  # Stage9 星表生成的羽化保护 mask 最大覆盖率，超限时安全跳过末端降噪
    stage10_large_galaxy_local_patch_variance_max: float = 0.00032  # 大型星系最终回星后允许的局部 patch 方差上限；其他目标仍使用 0.00022
    stage10_stage9_local_color_risk_strength: float = 1.0  # 按 Stage9 局部青蓝/核心突变风险比例压低正向最终饱和度
    stage10_managed_output_enabled: bool = True  # 独立生成带 sRGB/ICC 的 16-bit PNG/TIFF；永不重写 FITS 科学存档
    force_review_only_output: bool = False  # 显式启用时 Stage10 仅写 result_review*，不写正式结果名

    # Visual review evidence
    review_bundle_enabled: bool = True  # 为关键阶段生成 before/after/diff/metrics 视觉复核包

    # Stage 6-8: local deterministic diagnostic thresholds
    stage7_bg_median_min: float = 0.020  # 阶段7验收：背景中值不能低于此值，避免黑场压死
    stage7_black_pixel_ratio_max: float = 0.35  # 阶段7验收：近黑像素占比上限
    stage7_highlight_clip_ratio_max: float = 0.010  # 阶段7验收：高亮裁剪占比上限
    stage7_star_growth_ratio_max: float = 1.25  # 阶段7验收：星点中位尺寸增长上限
    stage7_bright_nebula_star_growth_ratio_max: float = 1.50  # Stage 7 亮核心星云 starless 拉伸的星状结构增长上限
    stage7_bright_nebula_star_mask_expand: int = 4  # cand_a 使用 Stage6 星掩膜向外扩张的柔性保护半径
    stage7_bright_nebula_star_faint_suppression: float = 0.85  # cand_a 星区局部弱信号/饱和抬升抑制强度
    stage7_bright_nebula_star_detail_suppression: float = 0.18  # cand_a 扩张星区正向局部细节压制强度，控制残余星点/halo 膨胀
    stage7_preview_calibration_enabled: bool = True  # 阶段7用 linked preview 的 P50/P99 标定正式 Asinh 候选
    stage7_target_aware_stretch_enabled: bool = True  # 阶段7按 target profile/policy 收紧核心保护或增强弱信号，不改变已配置的主候选池
    stage7_preview_cand_a_p50_ratio: float = 0.85  # cand_a 普通/宽场目标背景约为 linked preview P50 的 85%
    stage7_preview_cand_b_p50_ratio: float = 0.85  # cand_b 与 cand_a 使用同一显示标尺，再由亮核/星团/星系画像收紧
    stage7_preview_asinh_p50_max: float = 0.26  # Asinh 标定目标 P50 上限，与 linked MTF 显示上限一致
    stage7_preview_asinh_stretch_max: float = 1000.0  # preview 反解 Asinh 强度安全上限（Siril 常用范围上限）
    stage7_starless_linked_mtf_p50_min: float = 0.15  # Starless 重组候选的最低可视背景，避免状态合格但正式图近黑
    stage7_starless_linked_mtf_diffuse_p50_min: float = 0.17  # 宽场/暗星云保留更高的弱信号可见度下限
    stage7_starless_linked_mtf_preview_p50_ratio: float = 0.85  # 可视背景相对 linked preview P50 的基础目标比例
    stage7_starless_linked_mtf_p50_max: float = 0.26  # 可视背景上限，避免为提亮弱信号而过度拉伸
    stage7_starless_linked_mtf_shadow_noise_sigma: float = 3.0  # shadow 放在实测最低值/低分位以下的噪声安全距离
    stage7_mtf_reference_blackpoint_sigma: float = 5.0  # Statistical Stretch 名义 lower-half MAD 倍数（高斯等效约0.5917倍），仅作 reference-only 对照
    stage7_mtf_reference_p50_relative_error_max: float = 0.05  # 闭式 linked MTF 预测/实测 P50 最大相对偏差
    stage7_mtf_reference_p50_absolute_error_max: float = 0.005  # 闭式 linked MTF 预测/实测 P50 最大绝对偏差
    stage7_preview_target_p50_min_ratio: float = 0.90  # 实际 P50 达到标定目标 90% 才无告警
    stage7_preview_target_p50_hard_min_ratio: float = 0.80  # 实际 P50 低于标定目标 80% 时硬拒绝，避免近黑候选交付
    stage7_preview_target_p50_max_ratio: float = 1.50  # 实际 P50 不得超过标定目标的比例，避免过亮候选进入正式交付
    stage7_diffuse_visibility_score_min: float = 0.08  # 所有正式候选的绝对主体可见度下限
    stage7_preview_visibility_retention_min: float = 0.60  # 候选至少保留 linked preview 60% 的可见度；共享 1.5× 告警带对应 40% 硬线
    stage7_stretch_chroma_load_signal_excluded_max: float = 0.06  # 已从冻结背景排除显现星云后允许的低绝对色度负载；真实色噪/斑驳硬门不变
    stage7_stretch_feedback_retry_max: int = 1  # Stage 7 完整变换后实测 P50 偏离目标时，从同一线性基线重跑一次
    stage7_starless_structure_gate_enabled: bool = True  # Starless 拉伸使用星点掩膜局部秩结构门禁，避免通用阈值把星云纹理误判为星点膨胀
    stage7_starless_masked_rank_drift_p95_max: float = 0.18  # 原星点邻域亮度秩结构 P95 漂移上限；全局单调拉伸基本不改变该指标
    stage7_starless_halo_detail_growth_ratio_max: float = 1.60  # 原星点邻域秩域高频细节允许的增长倍数
    stage7_starless_halo_detail_delta_min: float = 0.010  # 细节增长同时超过此绝对量才触发比例硬门禁，避免低基线比值失真
    stage7_quantile_fallback_enabled: bool = True  # 双候选及亮度闭环仍失败时，允许用 preview P50/P99 标定的确定性单调分位数曲线兜底
    stage7_target_local_metrics_enabled: bool = True  # Stage 7 用线性源构建核心/弱结构/暗云局部区域并参与候选门控
    stage7_local_core_clip_ratio_max: float = 0.12  # 亮核局部区域允许的裁剪像素比例上限
    stage7_local_faint_snr_min: float = 0.25  # 目标弱结构相对局部背景的最低可分离信噪比
    stage7_local_dark_separation_min: float = 0.001  # 暗云周围亮云与暗结构的最低局部亮度分离
    stage7_stretch_chroma_noise_score_max: float = 0.34  # Stage 7 正式拉伸候选的背景绝对色噪上限
    stage7_stretch_background_mottling_score_max: float = 0.45  # Stage 7 正式拉伸候选的背景斑驳上限
    stage7_stretch_chroma_load_growth_max: float = 1.37  # Stage 7 拉伸后综合色偏差相对背景亮度的最大放大倍数
    stage7_stretch_chroma_load_low_absolute_max: float = 0.05  # 绝对 chroma load 低于此值时豁免极低基线导致的相对增长失真
    stage7_stretch_chroma_load_low_absolute_tolerance: float = 0.0005  # 极低背景绝对 chroma load 门的数值抖动容差；仅与低绝对豁免组合使用
    stage7_uncalibrated_background_chroma_load_review_max: float = 0.12  # 物理校色不可用时，排除主体/银河信号后的背景绝对色偏超过此值只允许复核交付
    stage7_chroma_rescue_enabled: bool = True  # 双候选仅因背景色噪被拒绝时，允许生成背景限定的保亮度色度抑制救援候选
    stage7_chroma_rescue_strength_levels: Tuple[float, ...] = (0.10, 0.20, 0.35)  # 救援按弱到中三档抑制背景色度；避免临界超限触发强去色
    stage7_chroma_rescue_max_strength: float = 0.90  # 自适应背景色度救援的绝对上限，主体/核心由冻结掩膜保护
    stage7_transform_new_hard_clip_ratio_warn: float = 0.0001  # 新增硬高光裁切超过此比例进入告警
    stage7_transform_new_hard_clip_ratio_max: float = 0.0005  # 新增硬高光裁切不可越过的技术门
    stage7_transform_unexpected_zero_ratio_max: float = 0.001  # 扣除声明黑点后仍新增纯黑像素的技术上限
    stage7_color_vector_p95_advisory_max: float = 0.04  # 宽带主体 RGB 色度方向 P95 告警线
    stage7_color_vector_p95_hard_max: float = 0.08  # 宽带主体 RGB 色度方向 P95 画质拒绝线
    stage7_narrowband_color_vector_p95_advisory_max: float = 0.10  # 窄带伪色/线发射允许的色度方向告警线
    stage7_narrowband_color_vector_p95_hard_max: float = 0.20  # 窄带色度方向画质拒绝线
    stage6_processing_mode: str = "auto"  # 阶段6处理方式：auto 去星；preserve 明确保留含星线性图并旁路 Starless 分支
    stage6_failure_action: str = "auto_fallback"  # 决定性失败：auto_fallback/preserve_review/stop
    stage6_starless_backend_policy: str = "auto_chain"  # auto_chain/syqon_only/sasp_only
    stage6_syqon_regional_texture_ratio_max: float = 1.80  # SyQon 分块区域纹理 P90/P10 硬上限
    stage6_syqon_regional_texture_sigma_min: float = 5.0  # SyQon 分块异常的最小绝对显著性
    stage6_syqon_regional_affected_ratio_max: float = 0.15  # SyQon 相连异常区域覆盖硬上限
    stage6_syqon_seam_retry_enabled: bool = True  # 分块像素门失败后仅允许一次 CPU/高重叠重试
    stage7_quality_retry_max: int = 2  # Stage 6 去星质量差时最多追加的同源 SyQon 参数重试次数
    stage7_quality_advisory_multiplier: float = 2.0  # Stage 6 数值门禁在接纳线至 2 倍异常度之间仅告警并继续；超过后才硬拒绝
    stage7_9_quality_advisory_multiplier: float = 1.5  # Stage 7-9 可恢复数值门禁在接纳线至 1.5 倍异常度之间仅告警并继续；结构性错误仍硬拒绝
    stage7_edge_black_warn: float = 0.10  # 阶段7去星前黑边风险提示阈值
    stage7_edge_black_high: float = 0.18  # 阶段7去星前高风险黑边阈值
    stage7_bg_median_high: float = 0.16  # 阶段7去星前高背景中值阈值
    stage7_bg_std_high: float = 0.055  # 阶段7去星前高背景噪声阈值
    stage7_bg_noise_ratio_high: float = 0.55  # 阶段7去星前背景噪声/背景中值高风险阈值
    stage7_residual_star_score_max: float = 0.45  # 阶段7验收：starless 残星评分上限
    stage7_halo_residue_score_max: float = 0.35  # 阶段7验收：亮星 halo 残留评分上限
    stage7_large_galaxy_halo_residue_score_max: float = 0.48  # M31/M81 等大星系的 halo 验收上限，避免全局统计混入盘面结构
    stage7_bright_nebula_halo_residue_score_max: float = 0.60  # M42/亮核心星云允许真实星云光晕保留，使用更高 halo 验收上限
    stage7_galaxy_roi_halo_gate_enabled: bool = True  # 星系目标用同一局部拉伸比较 original/starless 盘区，并从通用星点 halo 门排除核球/盘面
    stage7_galaxy_roi_star_clip_percentile: float = 99.5  # 星系低频定位前的亮星 winsorize 百分位
    stage7_galaxy_roi_peak_floor_ratio: float = 0.02  # 星系 ROI 信号底相对局部 Q99 的最低比例
    stage7_galaxy_roi_min_extent_ratio: float = 0.008  # 星系未钳制协方差尺度相对短边的下限
    stage7_galaxy_core_preservation_ratio_min: float = 0.72  # 星系亮核去星后相对同拉伸原图的最低亮度保留比例
    stage7_galaxy_core_contrast_ratio_min: float = 0.60  # 星系亮核相对核周环带的最低对比保留比例
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
    stage7_starmask_diffuse_residual_ratio_max: float = 0.08  # 清理后弥散残留能量比例验收线
    stage7_starmask_diffuse_uncertainty_abs: float = 0.0005  # 弥散残留软告警带的绝对不确定度
    stage7_starmask_diffuse_borderline_star_intensity_scale: float = 0.70  # 弥散残留处于软告警带时的 Stage9 星点强度上限
    stage7_conservative_repair_enabled: bool = True  # Stage 6 去星质量差时允许在同一线性输入上调整 SyQon 参数重试
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
class StageResult:
    """单个阶段的执行结果"""
    name: str
    status: str = 'skipped'     # ok / degraded / failed / skipped
    duration: float = 0.0
    message: str = ''
    execution: str = "completed"  # completed / safe_passthrough / skipped
    fallback_used: bool = False  # 仅表示本阶段实际采用回退路径
    upstream_passthrough: bool = False  # 上游安全旁路，不等同本阶段回退
    reason_code: str = ''
    details: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    review_reasons: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = str(self.status).strip().lower()
        if self.fallback_used and self.status == "ok":
            self.status = "degraded"
        if self.status not in {"ok", "degraded", "failed", "skipped"}:
            raise ValueError(f"unsupported stage status: {self.status!r}")
        self.execution = str(self.execution).strip().lower()
        if self.status == "skipped":
            self.execution = "skipped"
        if self.execution not in {"completed", "safe_passthrough", "skipped"}:
            raise ValueError(f"unsupported stage execution: {self.execution!r}")

    @property
    def display_status(self) -> str:
        """Return a structured summary status without inspecting human text."""
        if self.status != "ok":
            return self.status
        if self.execution == "safe_passthrough":
            return "ok_safe_passthrough"
        if self.execution == "skipped":
            return "ok_skipped_optional"
        return self.status
