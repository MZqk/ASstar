# Stage 8（Starless 深加工 / 星云增强）三维度审计报告

> 审计对象：`/Users/mz/dev/aiseestart` —— Seestar 望远镜离线深空后期 pipeline（Python + Siril 1.4.0）
> 审计范围：Stage 8 = starless-first 链路的非线性深加工阶段，负责分区提升星云主体与外围弱信号、保护背景与亮核
> 审计性质：**纯只读研究，未修改任何项目代码**
> 代码基准：提交 `HEAD`（截至 2026-08-05）
> 审计方法：静态精读 Stage 8 主流程 `stage8_nebula_enhancement.py`（1156 行）与像素实现 `stage8_pixels.py`（1942 行）核心函数，跨模块溯源 `models.py` / `seestar_Superimpose.py` / `task_workspace.py` / `stage_contracts.py` / `ai_advisory.py` / `stage_support.py` / `processor_runtime.py`，并联网比对 PixInsight / StarXTerminator / Siril / Legault / Croman / Adam Block / Warren Keller 等权威资料。

---

## 摘要与总评分

| 维度 | 评分（0-10） | 一句话结论 |
|------|-------------|-----------|
| **D1 逻辑专业合理性** | **8.5** | 四级路由 + 输入门禁 + 质量门回滚 + 蓝色守卫的保守架构合理扎实；仅有少数耦合/过度保守细节 |
| **D2 门禁来源与归档** | **8.0** | 门禁阈值集中、来源清晰、可追；唯 Stage 8 产物未纳入 `checkpoint-manifest` 签名断点（仅 1/2/5 阶段） |
| **D3 算法行业标准符合度** | **8.0** | 分区 mask + 局部曲线 + 弱信号提升 + 过度增强护栏，与 PixInsight/StarXTerminator/Legault 等主流实践高度一致；SASP 外部插件性质未验证 |

**最严重的三个问题（TOP3 参见末节）：**

1. **归档缺口（D2/P1）**：Stage 8 输出 `stage8_enhanced.fit` / `starless_enhanced.fit` 不被 `checkpoint-manifest.json` 覆盖（仅 Stage 1/2/5 为正式断点，`stage_contracts.py:18`）。断点完整性依赖于磁盘已有 `.fit`，而非签名校验断点。
2. **limited 模式在缺 starmask 时被整体跳过（D1/D2/P2）**：`stage8_input_enhancement_guard` 在 `requested_policy=="limited"` 且 `starmask_file` 缺失时会把整段增强判为 `skip`（`stage8_pixels.py:852-855`），但弱信号增强本身并不依赖 starmask，仅质量门 halotexture 报告需要，判定偏保守。
3. **对比度与饱和度耦合（D1/P2）**：`contrast_strength = min(cfg.stage8_nebula_contrast_max, 0.28*saturation)`（`stage8_pixels.py:1093`），饱和度=0 时局部对比恒为 0，对窄带/单色或禁色目标会抑制合理的结构提升。

---

## D1 逻辑专业合理性分析

### D1.1 路由与状态机：四态设计严谨 ✅

主入口 `run_stage8_nebula_enhancement`（`stage8_nebula_enhancement.py:98`）按上游 `StarSeparationState` 与 handoff 策略分流：

- **`REJECTED` / `TOOL_FAILED`**（`stage8_nebula_enhancement.py:131-198`）：**只保存含星复核点 `stage8_review_with_stars`**，设置 `passthrough=True, restricted_downstream=True`，**不运行任何 Starless-only 增强**。这与任务书路由约束完全一致。
- **`_star_preserve_target_bypass` + Stage7 接受**（`stage8_nebula_enhancement.py:225-310`）：目标保星旁路，直接拷贝 `stage7_stretched` 为 `starless_enhanced`/`stage8_enhanced`，`passthrough=True`。
- **正常路径**：加载 Stage7 拉伸输出（或回退 `starless`/`stage6_starless`，`stage8_nebula_enhancement.py:62-89`），进入输入门禁 → 增强 → 质量评估 → 回滚/校正链路。
- **输入门禁 skip**（如 halo/噪声越界）：保守跳过，输出 `stage8_input_starless`（`stage8_nebula_enhancement.py:392-470`）。

结论：状态机符合「starless-first 链路」的 starless 可用性门控哲学，未出现越界运行 Starless 增强的情形。**专业合理。**

### D1.2 分区 mask：与业界「starless + masked local enhancement」同构 ✅

`stage8_generate_starless_masks`（`stage8_pixels.py:109-231`）基于 luma 分位生成 6 类 mask：

- `background` / `nebula`（P68≈中亮主体）/ `faint_nebula`（P40≈外围弱信号）/ `core`（P99.2 + 通道峰≥0.985）/ `limited_core_exclusion`（核心扩张保护）/ `core_mask`
- 关键公式：`core_threshold = max(quantile(gray,0.992), bg_median + max(5.5*bg_std, 0.07))`（`stage8_pixels.py:147-150`），并对核心做 dilate+feather 羽化；`nebula_mask` 与 `faint_nebula_mask` 均乘以 `(1 - 0.90*core_mask)` 实现核心排斥（`stage8_pixels.py:171-189`）。
- `low_signal_thresholds`（`stage8_pixels.py:79-106`）按 `bg_median<0.08 && p99<0.120` 自动收紧噪声地板，避免弱信号误提。

这与 PixInsight 工作流中「先去星 → 用 mask 隔离主体/暗尘/背景，分别处理」完全同构（见 D3：`https://pixinsight.com/tutorials/NGC7023-HDR`、`https://skyandtelescope.org/astronomy-resources/astrophotography-tips/star-power-stellar-images-with-pixinsight/`）。mask 的「核心排斥 + 羽化」也对应 PixInsight 中 MT/MLT 软化 mask 的惯例（`http://cosgrovescosmos.com/tips-n-techniques/masks-asa-auperpower-in-pi`）。**专业合理。**

### D1.3 像素增强算子：局部、加权、可回退 ✅

`apply_stage8_masked_pixel_enhancement`（`stage8_pixels.py:991-1399`）的算子链：

1. **背景去噪**：`result = (1-w)*result + w*blurred`，`w = clamp(denoise_strength*(1.35*bg+0.30*faint+0.10),0,0.42)*core_guard`（`stage8_pixels.py:1136-1141`）。仅作用于背景/弱信号，核心受 `core_guard` 保护。
2. **蓝色预控（blue pre-control）**：在 `color_sample_weight` 区域估计 `blue_dom/red_dom`，若 `blue_excess>target` 则对蓝通道乘 `blue_pre_gain∈[0.82,1.0]` 并同步下调饱和度（`stage8_pixels.py:1109-1134`）。这是「消除蓝偏」的局部手段，对应 scopetrader 规则 3（饱和度不得超信号，`https://scopetrader.com/astrophoto-processing:-when-you've-gone-too-far/`）。
3. **弱信号提升（faint boost）**：`lift = faint_boost * faint_effect * background_guard * core_guard * (1-gray)`（`stage8_pixels.py:1147-1152`），加法提升外围暗结构。对应业界「reveal faint detail」（`https://astrosaver.com/enhancing-nebula-details-post-processing`）。
4. **局部对比（contrast）**：`result += (result-center)*contrast_weight`，`center` 取信号区加权均值（`stage8_pixels.py:1159-1170`）。属多尺度中性对比的近似。
5. **unsharp 细节**：box-blur 残差按 `nebula_effect * core_guard * background_guard` 加权（`stage8_pixels.py:1172-1181`）。
6. **饱和度**：`result += (result-gray)*sat_weight`，`sat_weight=clamp(saturation*(0.78*nebula+0.18*faint)*core_guard*background_guard,0,0.35)`（`stage8_pixels.py:1183-1192`）—— 饱和度被严格限制在 ≤0.35 的像素权重内，防止整幅爆色。
7. **SASP 插件回混**：`plugin_rgb` 经 `nebula_effect/faint_effect` mask 以 ≤0.38 权重混合（`stage8_pixels.py:1194-1209`），满足「SASP 输出必须经 Starless soft mask 回混」的顶层约束。

所有算子均为**加法/乘法、mask 加权、数值钳制**，无不可逆非线性破坏，且最终 `result = clamp(result,0,1)`（`stage8_pixels.py:1397`）。**专业合理。**

### D1.4 核心与背景保护：还原式保护 ✅

非 limited 模式下，背景与核心被「还原到输入 base」：

```
background_restore = clamp(background_restore_strength * background, 0, 1)   # strength≈0.985~1.0
result = result*(1-background_restore) + denoised_base*background_restore
core_restore = clamp(core_protection * core, 0, 1)                         # core_protection 默认 0.92
result = result*(1-core_restore) + base*core_restore
```
（`stage8_pixels.py:1232-1242`）

即背景区保留去噪后的输入、核心区还原 92% 原始像素，杜绝背景被拉亮 / 核心被压平。这与 scopetrader 规则 1（背景不得纯黑、不得被破坏）和规则 2（不得产生光环/振铃）一致。**专业合理。**

### D1.5 limited 模式不变量：强约束 ✅

limited 模式（高 halo 风险 / 亮发射反射星云）下：

- `saturation≤0.05`、`unsharp=0`、`bg_factor=0`、背景降噪/全局调色/蓝色全局校正全部禁用（`stage8_nebula_enhancement.py:505-512`、`stage8_pixels.py:1066-1071`）。
- **作用域不变量强制**：`result[:, weak_signal<=0.0] = base[:, weak_signal<=0.0]`，并对扩张硬核区还原（`stage8_pixels.py:1341-1350`）。注释明确指出这是「不变量，而非质量门期望」——像素在弱信号 mask 外与扩张硬核内必须**逐位等于 Stage8 输入**。这是极高水准的防御式设计。

### D1.6 质量门与回滚：双层护栏 ✅

- `stage8_quality_assessment`（`stage8_pixels.py:1657-1852`）计算 9 项增长比/绝对值门禁（blue_excess、saturation_growth、microcontrast_growth、highlight_clip、bg_std_growth、background_brightening、core_clip_growth、texture_artifact_growth、limited_halo_texture）。
- 未过门时：`_apply_stage8_color_correction_from_quality`（ccm 降蓝，`stage8_pixels.py:1449-1475`）→ 至多循环校正 → 仍不过则 `_stage8_conservative_rerun`（`stage8_pixels.py:1852-1913`，饱和度折半至 ≤0.18）→ 仍不过则 `_rollback_stage8_to_input`（`stage8_pixels.py:466-477`）。
- 蓝色守卫自身带回滚：若 `ccm` 后 blue_excess 未改善则还原（`stage8_nebula_enhancement.py:719-757`）。

这套「诊断 → 校正 → 保守重跑 → 回滚」链路与 AGENTS.md「画质风险参数必须有上限或回退值」原则契合。**专业合理。**

### D1.7 局部配方引擎与 object_mask_only 模式：分层防护 ✅

`apply_stage8_masked_pixel_enhancement` 在像素算子之后，还调用 `apply_local_adjustment_recipe`（`stage8_pixels.py:1283-1339`）叠加一层**版本化本地配方**，覆盖三类操作：

- `curve`（`faint_nebula` / `limited_weak_signal` mask，`(0,0)-(0.18,0.19)-(0.5,0.51)-(1,1)` 近线性曲线，`opacity≤0.60`）——温和提升暗区。
- `saturation`（`nebula` mask，`amount≤min(0.05, sat*0.12)`，`opacity 0.50`）——局部补饱和。
- `local_contrast`（`nebula` mask，`amount≤min(0.03, contrast*0.25)`，`radius=2`，`opacity 0.50`）——局部微对比。

该配方引擎（`local_adjustments.py`）自带**安全限制**（`local_adjustments.py:337-342`）：`clip_growth_max=0.002`、`background_median_drift_max=0.006`、`core_p99_drift_max=0.025`、`outside_mask_changed_ratio_max=0.001`。即「候选未过配方安全限制则保留配方前图像」（`local_adjustments.py` 的 `accepted` 判定）——这是**第二层像素级护栏**，与质量门（图像级）形成纵深防御。设计优秀。

`object_mask_only` 模式（`stage8_pixels.py:1028-1045`）：当目标为 `bright_emission_reflection_nebula` 或 `high_halo_risk` 时启用，将 `background_guard = 1.0 - 1.0*background`（彻底屏蔽背景）、下调饱和/对比/unsharp、低覆盖时进一步 `mask_quality_scale` 衰减。该分支对应亮发射星云「大动态范围、易 halo」的处理难点（`https://stargazerslounge.com/topic/441804-orions-core-with-pixinsight-hdrmt-mas-ghs/`），与 HDRMT「仅针对亮区、Lightness Mask 保护暗背景」理念同构。

### D1.8 值得注意的逻辑耦合（次要问题）

- **对比度依赖饱和度**（`stage8_pixels.py:1093`）：`contrast_strength = min(cfg.stage8_nebula_contrast_max, 0.28*saturation)`。当 `saturation=0`（窄带 OSC 或 `broadband_color_allowed=False` 禁色）时局部对比恒为 0，弱信号结构提升也被间接抑制。设计上虽保守，但将「色彩强度」与「结构对比」不当耦合，对灰度/窄带目标可能欠增强。
- **faint_boost / contrast 也按饱和度缩放**：`faint_boost = min(boost_max, 0.20*saturation)`（`stage8_pixels.py:1092`）。同理，禁色目标下弱信号提升也归零。建议将「亮度域」增强（对比/弱信号）与「色度域」增强（饱和度）解耦。
- **limited 模式缺 starmask 整体跳过**（见 D2 与 TOP3 #2）：防御过度，可能误杀可进行的弱信号增强。

### D1.9 高背景噪声守卫与跨阶段饱和度预算

两处额外的防御性设计值得记录：

- **高背景噪声守卫**：当 `masks["bg_std"] > cfg.stage7_bg_std_high` 时，像素增强内 `unsharp_amount *= 0.40`、`saturation *= 0.72`（`stage8_pixels.py:1062-1064`）。即背景噪声本底高时自动「收力」，避免放大噪点——直接呼应 scopetrader 规则 4（塑料/重复纹理 = 降噪过度或提噪）。
- **跨阶段饱和度预算钳制**：`effective_stage8_saturation = clamp_saturation_boost(requested_saturation, already_applied, color_limits)`（`stage8_nebula_enhancement.py:567-572`，来自 `pipeline_safety.py` 的 `color_safety_limits`）。Stage4 已施加的饱和被计入预算，Stage8 不得超过 `max_saturation_boost`（`stage8_nebula_enhancement.py:601-605` 有裁切 log）。这是**跨阶段一致性护栏**，防止 Stage4→Stage8 累积过饱和，符合「饱和度不得超信号」原则。
- **soft-mask 羽化**：所有分区 mask 经 `stage8_soften_mask`（多次 box-blur 软化，`stage8_pixels.py:72-77`）与 `feather_mask` 羽化（`stage8_pixels.py:160-175`）。羽化避免了 mask 硬边在增强后产生可见分界/振铃，对应 PixInsight 中以 MT/MLT 软化 mask 的惯例（`https://pixinsight.com/tutorials/NGC7023-HDR` 图 17）。

### D1.10 AI 参与的边界与安全（与 D2 门禁呼应）

`ai_advisory.py` 定义了 4 个代码拥有的候选预设 `stage8_processing_candidate_presets`（`ai_advisory.py:1150-1351`）：`preservation` / `conservative` / `balanced` / `detail_preserving`，饱和度钳制区间 `0.0–0.65`。AI 经 `request_stage8_processing_plan` **仅返回 `selected_candidate_id`**（`ai_advisory.py` 的 `normalize_stage8_ai_quality` 输出 `verdict/blue_guard/recommended_action`，而非参数）。即：

- **参数所有权归代码**：AI 不生成/不返回任何像素参数，杜绝「AI 编造细节」风险（与 scopetrader 规则 / Croman 真信号论一致，`https://scopetrader.com/astrophotography-fails:-the-bad-habits-ruining-your-shots/` 中「Letting AI Models Edit Your Image」条款）。
- **AI 仅做候选选择 + 质量复核**：质量门 `stage8_quality_assessment` 另调 `_request_stage8_quality_ai` 输出 `verdict/oversaturated/blue_bias/microcontrast_overdone`，作为质量门 `issues` 的补充（`stage8_pixels.py:1853` 附近）。AI 即便判「overdone」，最终执行仍由代码门禁（9 项阈值）决定，AI 无否决权之外的写权限。
- **离线优先**：`processing-plan.json.capabilities.offline_first=True`（`processor_runtime.py:~1970`），AI 仅 `network_requested` 时参与，离线场景下回退到代码默认 `balanced`/`nebula_saturation`。

该边界划分是本项目在「AI 辅助 vs 真实性优先」之间最稳健的设计点之一，符合 AGENTS.md 与 Croman/Legault 的过度处理警示。

### D1.12 综合判定

D1 维度逻辑自洽、保守、可回退，分区策略与护栏均符合天文深空增强的专业范式。扣除次要耦合与过度保守点后评 **8.5**。

---

## D2 门禁来源与归档

### D2.1 门禁清单与来源（逐条溯源）

| 门禁 / 阈值 | 默认值 | 钳制范围（CLAMP_RULES） | 来源位置 | 驱动来源 |
|------------|-------|------------------------|---------|---------|
| `stage8_mask_signal_coverage_min` | 0.002 | — | `models.py:418` | 输入门禁 mask 覆盖下限（`stage8_pixels.py:920-926`） |
| `stage7_residual_star_score_max` | （Stage7 配置） | — | `models.py`（Stage7 段） | 残星分数越界则 skip（`stage8_pixels.py:895-899`） |
| `stage7_halo_residue_score_max` | 0.35 | — | `models.py` | halo 基础阈值（`stage8_pixels.py:901`） |
| `_stage7_effective_halo_threshold()` | 0.600 | — | Stage7 方法 | halo 接受上限，>则 skip（`stage8_pixels.py:912-916`） |
| `stage7_starless_noise_gain_max` | （Stage7） | — | `models.py` | 噪声增益越界则 skip（`stage8_pixels.py:917-920`） |
| `stage8_blue_excess_max` | 0.08 | 0.02–0.30 | `models.py:419` / `seestar_Superimpose.py:557` | 质量门蓝偏阈值（`stage8_pixels.py:1724`） |
| `stage8_saturation_growth_ratio_max` | 1.45 | 1.05–2.50 | `models.py:420` / `seestar_Superimpose.py:558` | 饱和度增长比（`stage8_pixels.py:1731-1736`） |
| `stage8_microcontrast_growth_ratio_max` | 1.60 | （见 CLAMP） | `models.py:421` | 微对比增长比（`stage8_pixels.py:1738-1744`） |
| `stage8_highlight_clip_ratio_max` | 0.012 | 0.001–0.060 | `models.py:422` / `seestar_Superimpose.py:560` | 高光裁切（`stage8_pixels.py:1746-1749`） |
| `stage8_bg_std_growth_max` | 1.08 | 1.00–1.50 | `models.py:417`-系 / `seestar_Superimpose.py:493` | 背景噪声增长（`stage8_pixels.py:1751-1766`、`_stage8_bg_noise_growth_issue:443`） |
| `stage8_texture_artifact_growth_max` | 1.25 | 1.00–2.20 | `models.py` / `seestar_Superimpose.py:494` | 纹理伪影增长（`stage8_pixels.py:1786-1793`） |
| `core_clip_growth` | 0.0100 | （质量门硬编码） | `stage8_pixels.py:1783-1784` | 核心裁切增长绝对门 |
| `background_brightening` | 0.0200 | （质量门硬编码） | `stage8_pixels.py:1781-1782` | 背景提亮绝对门 |
| `stage8_limited_saturation_max` | 0.05 | — | `models.py:423` / `processor_runtime.py:1071` | limited 模式饱和度上限（`stage8_pixels.py:1068`） |
| `stage8_limited_core_exclusion_expand` | 8 | 2–16 | `models.py:424` / `processor_runtime.py:1213` | limited 核心扩张迭代（`stage8_pixels.py:160-175`） |
| `stage8_limited_halo_texture_growth_max` | 1.05 | — | `models.py:425` / `processor_runtime.py:1072` | limited halotexture 增长门（`stage8_pixels.py:1499-1506`） |
| `stage8_limited_halo_texture_delta_max` | 0.00075 | — | `models.py:426` / `processor_runtime.py:1073` | limited halotexture 绝对差门（`stage8_pixels.py:1504`） |
| `stage8_core_protection_strength` | 0.92 | 0.50–1.00 | `models.py:215` / `seestar_Superimpose.py:487` | 核心还原强度 |
| `stage8_faint_nebula_boost_max` | 0.08 | 0.0–0.18 | `models.py:216` / `seestar_Superimpose.py:489` | 弱信号提升上限 |
| `stage8_nebula_contrast_max` | 0.10 | 0.0–0.20 | `models.py:217` / `seestar_Superimpose.py:490` | 局部对比上限 |
| `stage8_masked_unsharp_amount_max` | 0.12 | 0.0–0.25 | `models.py:218` / `seestar_Superimpose.py:491` | unsharp 上限 |
| `stage8_background_denoise_strength` | 0.14 | — | `models.py:219` | 背景去噪强度 |
| `stage8_blue_precontrol_strength` | 0.55 | — | `models.py:220` | 蓝偏预控强度 |

**溯源结论**：所有门禁阈值均集中在 `PipelineConfig`（`models.py`）+ `CLAMP_RULES`（`seestar_Superimpose.py:448-594`）+ `AUTO_CLAMP_FIELDS`/`DYNAMIC_CLAMP_FIELDS` 统一管理，符合 AGENTS.md「可调参数集中在 PipelineConfig」「画质风险参数必须有上限」。**来源清晰、可追、可复现。**

### D2.2 门禁触发路径

- **第一道（输入门禁）** `stage8_input_enhancement_guard`（`stage8_pixels.py:802-989`）：依据 Stage6/7 handoff 的 `processing_policy`（full/limited/skip）、Stage7 质量状态、残星/halo/噪声增益、mask 覆盖率，决定 `skip_enhancement` 与最终 `processing_policy`。若 `reasons` 非空则整段跳过（`stage8_nebula_enhancement.py:392-470`）。
- **第二道（质量门）** `stage8_quality_assessment`（`stage8_pixels.py:1657-1852`）：增强后对候选 vs 基线做 9 项比对，输出 `status`（ok/poor）。
- **第三道（limited 专属）** `stage8_limited_halo_texture_report`（`stage8_pixels.py:1499-`）：基于 starmask 推导的星晕环带纹理增长/绝对差门。
- **第四道（AI 复核）** `_request_stage8_quality_ai`（`stage8_pixels.py:1853` 附近调用）：AI 仅输出 `verdict`/`oversaturated`/`blue_bias`/`microcontrast_overdone`（见 `ai_advisory.py` 的 `normalize_stage8_ai_quality`），**AI 不返回参数，仅确认问题**——边界清晰、安全。

### D2.3 归档路径核验

| 归档产物 | 内容 | Stage 8 覆盖情况 | 位置 |
|---------|------|----------------|------|
| `processing-plan.json` | `candidate_contracts.stage8_enhancement`（4 个候选 id：`preserve/conservative/balanced/detail_preserving`）、`parameters_owned_by: "code"`、 sanitized `config`、`plan_hash` | ✅ Stage 8 候选契约与配置入档 | `processor_runtime.py:1942-1963`、`1981` |
| `pipeline-result.json` | `actual_steps`（各 Stage `StageResult` details，含 `stage8_handoff`）、`checkpoints`、`outputs`、`manifest_hash` | ✅ Stage 8 实际步骤与 handoff 入档 | `processor_runtime.py:2060-2176`、`2157` |
| `stage8_enhancement_report.json` | 路由/门禁/质量/回滚详情 | ✅ 专属于 Stage 8 | `stage8_nebula_enhancement.py:178-196` 等多处 `_write_stage_json` |
| `stage8_quality.json` | 初始/最终质量评估 | ✅ | `stage8_nebula_enhancement.py:430-440` |
| `checkpoint-manifest.json` | **仅** `stage{1,2,5}` 正式断点（签名校验） | ❌ **Stage 8 不在白名单** | `stage_contracts.py:18`、`task_workspace.py:862-865` |

**关键缺口**：`FORMAL_RESUME_STAGES = (1, 2, 5)`（`stage_contracts.py:18`），`build_checkpoint_manifest` 仅接受这些阶段（`task_workspace.py:862`：`allowed = {f"stage{stage}" for stage in FORMAL_RESUME_STAGES}`），且 `publish_checkpoint` 显式 `raise WorkspaceError("只能发布 Stage 1、2、5 正式断点")`（`task_workspace.py:462-463`）。

**影响评估**：Stage 8 产物 `stage8_enhanced.fit` / `starless_enhanced.fit` / `stage8_limited_candidate.fit` 仅作为磁盘 `.fit` 存在，其**完整 provenance 在 `pipeline-result.json`（含 `stage8_handoff` details 与 `manifest_hash`）中可追溯**，但**缺少与 Stage 1/2/5 同等的签名断点保护**。若发生断点续跑（如从 Stage 9 续跑），Stage 8 的正确性依赖磁盘文件未被污染/移动，而无独立完整性校验。

> 注：Stage 8 非「正式断点」在功能上可接受（下游 Stage 9 直接读 `starless_enhanced`），但作为审计项，应至少在 `pipeline-result.json` 之外为 Stage 8 关键产物补充内容哈希/校验和，或将其纳入一个「soft checkpoint」清单，以闭合 provenance 完整性。详见改进建议 P1。

### D2.4 门禁链路追踪（一次典型 full 路径）

以「Stage7 接受 + full 策略 + 无高 halo 风险」为例，门禁逐层生效：

1. `run_stage8_nebula_enhancement` 加载 `stage7_stretched`（`stage8_nebula_enhancement.py:62-89`）。
2. `stage8_input_enhancement_guard`：读 handoff `processing_policy=full`；检查 Stage7 `status==ok`、残星/halo/噪声增益均在阈值内、`mask_signal_coverage≥0.002` → `reasons=[]` → `skip_enhancement=False`，`processing_policy` 保持 `full`（`stage8_pixels.py:802-989`）。
3. 若 `ai_stage8_enabled`：请求 AI 选候选 id（`stage8_nebula_enhancement.py:497` → `ai_advisory.py` 的 `request_stage8_processing_plan`），AI **仅返回 `selected_candidate_id`**，参数由代码从 `stage8_processing_candidate_presets` 取（`ai_advisory.py:1150-1351`）。
4. 尝试 SASP API（`stage8_nebula_enhancement.py:622`）；不可用则 `_apply_stage8_builtin_enhancement`（`stage8_nebula_enhancement.py:656` → `stage8_pixels.py:1399`）。
5. 保存 `starless_enhanced` / `stage8_enhanced`；若 `limited_mode` 另存 `stage8_limited_candidate`（`stage8_nebula_enhancement.py:704-712`）。
6. `stage8_quality_assessment` 9 项比对（`stage8_pixels.py:1657-1852`）；不过则蓝色校正 → 保守重跑 → 回滚。
7. 所有结果写入 `stage8_enhancement_report.json` 与 `pipeline-result.json`（含 `stage8_handoff` details）。

该链路在 every 分支都有 `handoff` 回写与 `_record_stage`（`seestar_Superimpose.py:1318`），可追溯性良好。

### D2.6 产物文件生命周期与清理保留

Stage 8 生成的 `.fit` 在 pipeline 的清理/保留逻辑中被显式保护：

- **检查点保留**：`seestar_Superimpose.py:1466` 附近将 `stage8_enhanced` / `starless_enhanced` / `stage8_review_with_stars` 列入保留检查点，清理阶段不会被回收。
- **limited 候选保留**：`seestar_Superimpose.py:2250-2280` 的 `keep_files` 含 `stage8_limited_candidate.fit`，确保「事务式回滚」所需的基线候选不被误删。
- **回滚目标存在性**：`_rollback_stage8_to_input`（`stage8_pixels.py:466-477`）与保守重跑（`:1852-1913`）均依赖 `stage8_input_starless.fit` / `starless_enhanced.fit` 在磁盘存在。由于这些文件在保留列表中，回滚链在正常运行下可靠。

**残留风险**：若用户手动移动/删除了 `stage8_input_starless.fit`，回滚将失败且仅有 `status='degraded'` 提示（`stage8_nebula_enhancement.py:928-962`），无更上游的二次恢复。这正是 D2.3 所述「缺乏签名断点」的连锁后果——建议的 P0-1（内容哈希）可覆盖此场景。

### D2.7 综合判定

门禁来源集中、阈值可追、三级门 + AI 复核 + 回滚链路完整；归档主链路（`processing-plan` / `pipeline-result` / 专报）完备，但 `checkpoint-manifest` 签名断点未含 Stage 8。评 **8.0**。

---

## D3 算法行业标准符合度（含联网取证）

### D3.1 分区 mask + 局部增强 ≈ PixInsight 业界标准 ✅

- PixInsight 官方 NGC7023 HDR 教程明确：**先去星 → 用修改后的 star mask（MT/ATWT/HT/CT 软化）做 LHE 局部增强 → 用 CurvesTransformation 在 S 通道补回因局部增强丢失的饱和度 → 用 luminance mask 做 S 形曲线**（`https://pixinsight.com/tutorials/NGC7023-HDR`）。本 pipeline 的「分区 mask（core/nebula/faint/background）+ 局部曲线 + 饱和度加权 + 亮度 mask 对照」与该范式一一对应。
- StarXTerminator / StarNet 去星后将星与星云分离处理、再重合成，是当代主流（Russ Croman StarXTerminator、`https://skyandtelescope.org/astronomy-resources/astrophotography-tips/star-power-stellar-images-with-pixinsight/`；Siril 1.4 原生 StarNet 集成：`https://siril.org/tutorials/synthetic-stars/`、`https://www.fas37.org/wp/wp-content/uploads/2025/10/Siril-Workflow.pdf`）。本 pipeline「starless-first」即此范式。
- Warren Keller 窄带发射星云 PixInsight 大师课同样采用：StarXTerminator 去星 → GAME 星云 mask → LHE 局部对比 → MMT 锐化 → DarkStructureEnhance → 重屏星（`https://telescope.live/tutorials-old/master-classes`）。DarkStructureEnhance 的「暗尘 mask」理念与本项目 `faint_nebula_mask` 同源。

### D3.2 弱信号提升 ≈ 「reveal faint detail / 渐进拉伸」 ✅

- 业界强调「一系列温和拉伸而非一次激进变换」以揭示暗结构（`https://astrosaver.com/enhancing-nebula-details-post-processing`；Adam Block Stretch Academy：`http://astronomy.robpettengill.org/blog230614.html`）。本项目 `faint_boost` 加法提升（受 `faint_nebula_mask` 与 `(1-gray)` 约束）即温和提升外围弱信号，且 `low_signal_thresholds` 自动收紧噪声地板，避免过度提噪。

### D3.3 过度增强护栏 ≈ scopetrader「5 条过度处理红线」✅

scopetrader 总结的过度处理判定（`https://scopetrader.com/astrophoto-processing:-when-you've-gone-too-far/`）：
1. 背景纯黑 = 裁切 → 本项目保护背景（`background_restore`，`stage8_pixels.py:1232`）+ `background_brightening>0.020` 门（`stage8_pixels.py:1781`）。
2. 星/结构出现光环、暗环、脆边 = 锐化过度 → 本项目 `highlight_clip`、`core_clip_growth`、unsharp 受 `core_guard/background_guard` 限制，且 `stage8_high_risk` 时禁 unsharp（`stage8_nebula_enhancement.py:697-701`）。
3. 颜色全拉满 = 超信号饱和度 → 本项目 `saturation` 像素权重 ≤0.35、全局 `saturation_growth_ratio_max=1.45`、`blue_excess` 门。
4. 噪声变塑料/重复纹理 = 降噪过度 → 本项目 `bg_std_growth_max=1.08`、`texture_artifact_growth_max=1.25` 门。
5. 仅缩略图好看 = 掩盖问题 → 本项目质量门在全分辨率像素域逐区比对，非缩略。

### D3.4 「真实性优先 / 不制造假细节」≈ Legault & Croman 共识 ✅

- Thierry Legault：「最好的处理者是知道何时停止的人；过度平滑/锐化/放大会在噪声与湍流上制造不存在的细节」（`https://test.universetoday.com/articles/how-to-avoid-bad-astrophotography-advice-from-thierry-legault/`）。
- Russell Croman：「AI/过度降噪会抹掉真实细节、产生塑料感；真实信号只能靠更多曝光获得，处理不能提升 SNR」（`https://ssr.app.astrobin.com/forum/topic/113150/russell-croman-astrophotography-noisexterminator/the-need-for-real-signal-thoughts-on-true-image-quality`）。

本项目与之吻合的要点：
- `AGENTS.md` 原则「真实性优先」「保守默认」；
- 所有增强为**加法/乘法加权**（不凭空生成结构），与「不发明细节」一致；
- `stage8_conservative_rerun` 在质量门不过时**主动退让**而非强行增强；
- 背景去噪权重上限 0.42、纹理伪影增长门 1.25，均抑制「塑料感」。

### D3.5 HDRMT / LHE 的「核心保护」理念一致 ✅

- HDR Multiscale Transform 用 `Lightness Mask` 保护暗背景、用 `To Intensity / Preserve Hue` 保核心色彩（`https://www.chaoticnebula.com/pixinsight-hdr-multiscale-transform/`、`https://stargazerslounge.com/topic/441804-orions-core-with-pixinsight-hdrmt-mas-ghs/`）。本项目 `core_mask` 还原 + `core_clip_growth` 门，正是核心保护的非线性等价实现。
- 但需注意：HDRMT/LHE 是**多尺度小波**方法；本项目内置增强是**单尺度 unsharp + 局部曲线 + 加法提升**，尺度表达能力弱于 PixInsight 的 ATWT/MMT。这一差距由外部 **SASP WaveScale/DSE** 插件补足（见 D3.6）。

### D3.6 外部 SASP 插件：性质未验证 ⚠️

代码通过 `_run_sasp_stage8_api`（`stage8_nebula_enhancement.py:622`）调用 `sasp_starless.fit` / `starless_sasp.fit` 等外部产物，并在不可用时间退内置链（`stage8_nebula_enhancement.py:622-640`）。命名暗示「WaveScale Dark Enhancer / 多尺度暗结构增强」，与 PixInsight DarkStructureEnhance / Wavelet 思路相近（`http://cosgrovescosmos.com/tips-n-techniques/masks-asa-auperpower-in-pi`）。

> **未验证**：仓库内未包含 SASP 插件本体或文档，无法确认其为哪一第三方工具（是否为 Siril 的 SASP 脚本/插件、或其多尺度算法细节）。该不确定性已在 `sasp_runner.py:641-730` 的 `boost_upper/gamma_upper/n_scales/decay_rate` 参数中可见其「多尺度 boost」意图，但**算法标准符合度无法独立验证**，需以插件官方文档为准。建议补充 SASP 来源与版本归档（见改进 P2）。
>
> 补充说明：即便 SASP 不可用，Stage 8 仍有完整的**内置像素增强回退链**（`stage8_nebula_enhancement.py:622-640` → `apply_stage8_builtin_enhancement` → `stage8_pixels.py:1399`），因此 SASP 的「未验证」不影响 Stage 8 的可用性，仅影响「多尺度暗结构增强」这一子能力的标准符合度评级。该回退设计本身是健壮的（offline_first）。

### D3.7 Siril 1.4 原生能力边界（上下文）

Siril 1.4 提供 StarNet 去星、Histogram/Asinh 拉伸、Star Recomposition、Curves（`https://siril.org/tutorials/synthetic-stars/`、`https://www.fas37.org/wp/wp-content/uploads/2025/10/Siril-Workflow.pdf`）。其内置无 LHE/多尺度 HDRMT 等价物，因此本项目**内置像素增强（unsharp/局部曲线/加法提升）属 Siril 能力范围内的合理替代**，而真正的多尺度增强依赖 SASP 外部插件。这符合「offline_first、能力不足时回退」的设计（`processor_runtime.py` `capabilities.offline_first=True`）。

### D3.8 符合度对照矩阵

| 实践项 | 业界标准做法 | 本项目实现 | 符合度 |
|-------|------------|-----------|-------|
| 去星后分离处理 | StarXTerminator/StarNet + 重合成（`skyandtelescope.org` 文） | starless-first 链路，Stage8 在 starless 上增强 | ✅ |
| 分区 mask（核心/主体/弱信号/背景） | PixInsight GAME/Range mask + MT 软化（`cosgrovescosmos.com`） | `stage8_generate_starless_masks` 6 类 + feather | ✅ |
| 局部对比增强 | PixInsight LHE/ATWT/MMT（`pixinsight.com/tutorials/NGC7023-HDR`） | unsharp + 局部曲线 + 加法对比（单尺度） | ⚠️ 尺度弱于多尺度 |
| 弱信号提升 | 渐进拉伸揭示暗结构（`astrosaver.com`） | `faint_boost` 加法 + 噪声地板自适应 | ✅ |
| 核心保护 | HDRMT Lightness Mask + Preserve Hue（`chaoticnebula.com`） | `core_mask` 还原 92% + `core_clip_growth` 门 | ✅ |
| 饱和度控制 | LHE 后 S 通道补饱和（`pixinsight.com` 文） | 加权饱和 ≤0.35 + `saturation_growth` 门 | ✅ |
| 蓝偏校正 | 去绿/色彩平衡（`eathealthy365.com` 指南） | `blue_pre_gain` + `ccm` 蓝色守卫（含回滚） | ✅ |
| 过度处理防护 | scopetrader 5 红线（`scopetrader.com` 文） | 9 项质量门 + 保守重跑 + 回滚 | ✅ |
| 多尺度暗结构增强 | DarkStructureEnhance（`cosgrovescosmos.com`） | 依赖外部 SASP 插件（未验证） | ⚠️ 未验证 |
| 真实性/不造假 | Legault「知止」、Croman「真信号」 | 加法/乘法加权、无生成式算子 | ✅ |

### D3.9 综合判定

算法选择与护栏与 PixInsight/StarXTerminator/Legault/Croman/Adam Block 等主流实践高度一致；唯一的符合度盲点是 SASP 外部插件算法未验证。评 **8.0**。

---

## 关键发现 TOP3

1. **归档缺口（D2/P1）**：`stage8_enhanced.fit` / `starless_enhanced.fit` 不被 `checkpoint-manifest.json` 签名断点覆盖（`stage_contracts.py:18`、`task_workspace.py:862`）。Stage 8 provenance 仅靠 `pipeline-result.json` 的 `stage8_handoff` details 与 `manifest_hash` 追溯，缺乏与 Stage 1/2/5 同等的完整性校验。建议为 Stage 8 关键产物补充内容哈希或纳入 soft-checkpoint 清单。
2. **limited 模式缺 starmask 整体跳过（D1/D2/P2）**：`stage8_input_enhancement_guard` 在 `requested_policy=="limited"` 且 `starmask_file` 缺失时，将整段增强判 `skip`（`stage8_pixels.py:852-855`）。但弱信号增强本身依赖 `stage8_generate_starless_masks` 而非 starmask，仅 `stage8_limited_halo_texture_report` 需要 starmask。判定偏保守，可能误杀可进行的弱信号提升。建议将「starmask 缺失」降级为「halotexture 门不可用警告」而非整体 skip。
3. **亮度域与色度域增强耦合（D1/P2）**：`contrast_strength` 与 `faint_boost` 均按 `saturation` 缩放（`stage8_pixels.py:1092-1093`）。当 `saturation=0`（窄带/单色/禁色目标）时局部对比与弱信号提升恒为 0，结构提升被间接抑制。建议将亮度域增强（对比/弱信号/去噪）与色度域增强（饱和度）解耦，使前者在 `saturation=0` 时仍可生效。

---

## 改进建议（可执行，指明文件位置）

### P0（必须，影响正确性/安全性）

> 本次审计为只读研究，未修改任何代码。以下为建议，由维护者评估后实施。

- **P0-1 闭合 Stage 8 断点完整性**：在 `pipeline/_build_pipeline_result`（`processor_runtime.py:2060-2176`）之外，为 `stage8_enhanced.fit` / `starless_enhanced.fit` 计算内容哈希（sha256）并写入 `pipeline-result.json.outputs`；或扩展 `task_workspace.build_checkpoint_manifest`（`task_workspace.py:853`）接受一个「soft stage」集合，将 Stage 8 纳入非强制但受校验的断点清单。参考 `stage_contracts.py:18` 的 `FORMAL_RESUME_STAGES` 模式。
  - 依据：D2.3 缺口；断点续跑依赖未校验磁盘文件。

### P1（重要，影响质量/鲁棒性）

- **P1-1 解耦亮度域与色度域增强**：修改 `apply_stage8_masked_pixel_enhancement`（`stage8_pixels.py:1092-1093`），将 `contrast_strength` 与 `faint_boost` 的基准从 `saturation` 改为独立的亮度域上限（如 `stage8_nebula_contrast_max`、`stage8_faint_nebula_boost_max` 直接作为上限，不乘 `saturation`）。可保留「禁色目标不调色」但允许「禁色目标仍做结构/弱信号提升」。
  - 依据：D1.7、TOP3 #3。
- **P1-2 limited 模式 starmask 缺失降级**：修改 `stage8_input_enhancement_guard`（`stage8_pixels.py:852-855`），将 `stage8_limited_starmask_unavailable` 由 `reasons`（触发 skip）改为 `advisories`（仅提示 halotexture 门不可用，质量门仍可运行并记录 `limited_halo_texture_gate_unavailable` 问题，`stage8_pixels.py:1701-1706` 已具备该分支）。
  - 依据：D2.2、TOP3 #2。

### P2（建议，提升符合度/可维护性）

- **P2-1 归档 SASP 插件来源与版本**：在 `processing-plan.json.capabilities`（`processor_runtime.py:~1970`）或 `sasp_runner.py:641-730` 增加 SASP 插件名称/版本/算法说明字段，闭合 D3.6 的「未验证」项。
- **P2-2 增强尺度表达**：内置增强为单尺度（unsharp + 局部曲线）。若长期不依赖 SASP，可考虑在 `apply_stage8_masked_pixel_enhancement` 增加多尺度（小波/高斯金字塔）对比分支，以更接近 PixInsight LHE/MMT 的多尺度表达能力（参考 `https://pixinsight.com/tutorials/NGC7023-HDR`）。属性能增强，非缺陷。
- **P2-3 高背景噪声守卫的可观测性**：`stage8_pixels.py:1062-1064` 的 `high-bg-noise guard reduced saturation/detail` 已打 message，建议在 `stage8_enhancement_report.json` 的 `advisories` 显式记录，便于事后追溯为何某帧增强偏弱。
- **P2-4 为对象掩码增强增加多尺度选项**：当前 `object_mask_only` 模式仅下调单尺度 unsharp（`stage8_pixels.py:1028-1045`）。可参考 PixInsight MMT（`https://pixinsight.com/tutorials/NGC7023-HDR`）在 `apply_stage8_masked_pixel_enhancement` 内叠加 2–3 层高斯金字塔对比，兼顾亮发射星云的尺度和 halo 安全。属性能增强，非缺陷。
- **P2-5 limited 模式 `weak_signal` 尾截断阈值文档化**：`weak_signal = clip((weak_signal-0.05)/0.95,0,1)`（`stage8_pixels.py:1019-1021`）中的 `0.05` 硬阈值未在 `PipelineConfig` 暴露，建议提升为可配置项（如 `stage8_limited_weak_signal_floor`），便于不同目标调参，同时保持 `limited_core_exclusion` 不变量的强制（`:1341-1350`）。

### 审计方法与局限

- **方法**：静态代码精读（主流程 `stage8_nebula_enhancement.py` 全 1156 行、像素实现 `stage8_pixels.py` 关键函数 1942 行中 12 个核心函数）、跨模块溯源（`models.py` / `seestar_Superimpose.py` CLAMP / `task_workspace.py` / `stage_contracts.py` / `ai_advisory.py` / `stage_support.py` / `processor_runtime.py`）、联网检索 PixInsight/StarXTerminator/Siril/Legault/Croman/Adam Block/Warren Keller 等权威资料比对。
- **未运行**：本次为只读审计，未执行 pipeline，故门禁命中率、limited 模式实际触发频次、SASP 插件可用性均为静态溯源结论（见「未验证项」）。
- **命名错位提示**：全仓存在 Stage 编号与文件名错位（Stage6 去星在 `stages/stage7_star_separation.py`、Stage7 拉伸在 `stages/stage6_stretching.py`），但 Stage8 文件名与编号一致，不影响本报告结论；该错位已在 `outline.yaml` code_map_note 标注，建议后续统一重命名以降低维护风险。
- **评分口径**：0–10，越高越优；D1 看逻辑自洽/保守/可回退，D2 看门禁来源清晰与归档完整，D3 看与业界标准的算法同构度；不确定性均显式标注「未验证」。

---

## 证据索引

### 代码证据（file:line）

| 主题 | 位置 |
|------|------|
| 主入口与四态路由 | `pipeline/stages/stage8_nebula_enhancement.py:98` / `:131-198`(REJECTED) / `:225-310`(bypass) / `:340-470`(输入门禁 skip) |
| 输入源回退链 | `pipeline/stages/stage8_nebula_enhancement.py:62-89` |
| 输入门禁 `stage8_input_enhancement_guard` | `pipeline/stage8_pixels.py:802-989`（limited starmask 跳过 `:852-855`） |
| 分区 mask 生成 | `pipeline/stage8_pixels.py:109-231`（core `:147-150`、nebula/faint `:171-189`） |
| 低信号阈值自适应 | `pipeline/stage8_pixels.py:79-106` |
| 像素增强主函数 | `pipeline/stage8_pixels.py:991-1399`（去噪 `:1136-1141`、蓝预控 `:1109-1134`、弱信号 `:1147-1152`、对比 `:1159-1170`、unsharp `:1172-1181`、饱和 `:1183-1192`、SASP 回混 `:1194-1209`、背景/核心还原 `:1232-1242`、limited 不变量 `:1341-1350`） |
| 对比/弱信号与饱和度耦合 | `pipeline/stage8_pixels.py:1092-1093` |
| 质量评估 `stage8_quality_assessment` | `pipeline/stage8_pixels.py:1657-1852`（blue `:1724`、sat `:1731`、micro `:1738`、clip `:1746`、bg `:1751`、bg_bright `:1781`、core_clip `:1783`、texture `:1786`） |
| 蓝色守卫回滚 | `pipeline/stages/stage8_nebula_enhancement.py:719-757` |
| 保守重跑 / 回滚 | `pipeline/stage8_pixels.py:1852-1913`(rerun) / `:466-477`(rollback) |
| blue_guard ccm | `pipeline/stage_support.py:800-813` |
| 门禁阈值（config） | `pipeline/models.py:214-230`(增强参数) / `:417-426`(门禁参数) |
| 钳制规则 | `pipeline/seestar_Superimpose.py:448-594`（stage8 段 `:487-494`、`:557-560`） |
| AI 候选契约（参数归代码） | `pipeline/processor_runtime.py:1942-1963` |
| 归档（plan/result hash） | `pipeline/processor_runtime.py:1981`(plan_hash) / `:2157`(manifest_hash) / `:2060-2176`(pipeline-result) |
| 断点白名单缺口 | `pipeline/stage_contracts.py:18` / `pipeline/task_workspace.py:862`(allowed) / `:462-463`(publish 限制) |
| SASP 外部调用 | `pipeline/stages/stage8_nebula_enhancement.py:622` / `pipeline/sasp_runner.py:641-730` |
| 全仓命名错位提醒 | `deep-sky-stage-audit-research/outline.yaml` code_map_note（Stage6=去星在 `stages/stage7_star_separation.py`，Stage7=拉伸在 `stages/stage6_stretching.py`） |

### 行业标准证据（URL）

| 主题 | URL |
|------|-----|
| PixInsight LHE + mask + 饱和度补偿 + deringing（官方） | `https://pixinsight.com/tutorials/NGC7023-HDR` |
| Star mask / starless 分离处理与重合成 | `https://skyandtelescope.org/astronomy-resources/astrophotography-tips/star-power-stellar-images-with-pixinsight/` |
| HDRMT Lightness Mask / 核心保护 | `https://www.chaoticnebula.com/pixinsight-hdr-multiscale-transform/` |
| HDRMT 核心保留色彩实践 | `https://stargazerslounge.com/topic/441804-orions-core-with-pixinsight-hdrmt-mas-ghs/` |
| Dark Structure Mask / 暗尘 mask 理念 | `http://cosgrovescosmos.com/tips-n-techniques/masks-asa-auperpower-in-pi` |
| 过度处理 5 条红线 | `https://scopetrader.com/astrophoto-processing:-when-you've-gone-too-far/` |
| Legault：过度处理制造假细节、「知止」 | `https://test.universetoday.com/articles/how-to-avoid-bad-astrophotography-advice-from-thierry-legault/` |
| Croman：真实信号不可被处理替代、AI 降噪塑料感 | `https://ssr.app.astrobin.com/forum/topic/113150/russell-croman-astrophotography-noisexterminator/the-need-for-real-signal-thoughts-on-true-image-quality` |
| Warren Keller 窄带星云 PixInsight 大师课（LHE/MMT/DarkStructureEnhance/星重屏） | `https://telescope.live/tutorials-old/master-classes` |
| Adam Block Soft Light 局部对比 + object mask | `https://www.astronomy.com/magazine/adam-block/2016/02/soft-light` |
| Adam Block Stretch Academy（星/星云分离拉伸） | `http://astronomy.robpettengill.org/blog230614.html` |
| 弱信号「温和渐进拉伸」揭示暗结构 | `https://astrosaver.com/enhancing-nebula-details-post-processing` |
| Siril 1.4 StarNet 去星 / Star Recomposition 原生能力 | `https://siril.org/tutorials/synthetic-stars/` / `https://www.fas37.org/wp/wp-content/uploads/2025/10/Siril-Workflow.pdf` |

### 未验证项

- **SASP 插件算法与版本**：仓库未含插件本体/文档，`sasp_runner.py:641-730` 仅暴露 `boost_upper/gamma_upper/n_scales/decay_rate` 等多尺度参数，无法独立验证其算法标准符合度（D3.6）。
- **limited 模式实际触发频率**：`requested_policy=="limited"` 由 Stage6/7 handoff 驱动，本次只读审计未运行真实数据，无法统计其命中率与跳过率（D2.2 仅做静态溯源）。

---

*审计完成。本报告所有代码结论均带 `文件:行号`，行业结论均带 URL；无法验证处已显式标注「未验证」。未对仓库做任何修改。*

---

## 给维护者的优先级速查

| 优先级 | 项 | 动作 | 关键文件 |
|-------|---|------|---------|
| P0 | Stage 8 断点完整性缺口 | 为 `stage8_enhanced`/`starless_enhanced` 补内容哈希或纳入 soft-checkpoint | `processor_runtime.py:2060-2176`、`task_workspace.py:853` |
| P1 | 亮度/色度增强耦合 | `contrast_strength`/`faint_boost` 不再乘 `saturation` | `stage8_pixels.py:1092-1093` |
| P1 | limited 缺 starmask 整体跳过 | 将 starmask 缺失由 `reasons` 降级为 `advisories` | `stage8_pixels.py:852-855` |
| P2 | SASP 插件未验证 | 归档插件名称/版本/算法说明 | `sasp_runner.py:641-730`、`processor_runtime.py:~1970` |
| P2 | 单尺度增强尺度弱 | 增加多尺度对比分支 | `stage8_pixels.py:991-1399` |
| P2 | weak_signal 尾阈值硬编码 | 提升为可配置 `stage8_limited_weak_signal_floor` | `stage8_pixels.py:1019-1021` |
| P2 | 全仓 Stage 编号/文件名错位 | 统一重命名降低维护风险 | `outline.yaml` code_map_note |

> 整体结论：Stage 8 在工程实现上属于**高成熟度、保守优先、可回退、可追**的深空增强模块，三维度均达 8.0+。主要改进空间集中在「断点签名完整性（P0）」与「两类过度保守/耦合的细节（P1）」，无功能性缺陷或标准违背。
