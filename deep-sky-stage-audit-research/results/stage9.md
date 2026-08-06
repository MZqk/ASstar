# Stage 9（星点处理与合成 / 回星）三维度深度审计报告

- 审计对象：`/Users/mz/dev/aiseestart` —— Seestar 望远镜离线深空后期 pipeline（Python + Siril 1.4.0）
- 审计阶段：Stage 9 星点处理与合成（回星 / remix），链路收口环节
- 审计性质：纯研究，**未修改任何项目代码**
- 审计日期：2026-08-05
- 必读上下文：`deep-sky-stage-audit-research/outline.yaml`、`.../fields.yaml`、仓库 `pipeline/AGENTS.md`（原则：禁止星色失真、禁止二次星点/halo）
- 核心代码：`pipeline/stages/stage9_star_remixing.py`、`pipeline/stage9_quality.py`、`pipeline/star_color_repair.py`、`pipeline/stage_support.py`、`pipeline/models.py`、`pipeline/seestar_Superimpose.py`、`pipeline/processor_runtime.py`
- 方法：静态代码阅读（全 Stage 9 相关模块）+ 公开行业标准文献对标（D3 全部附 URL）。未运行管线、未改动代码。

---

## 0. 摘要与总评分

| 维度 | 评分（0-10） | 一句话结论 |
|------|-------------|-----------|
| D1 逻辑专业合理性 | **9.0** | 处理链路在数学与色彩学上自洽：独立星表 + 保色拉伸 + Alpha+Screen 约束混合 + 多级门控，整体专业且严谨 |
| D2 门禁来源与归档 | **9.0** | 全部阈值来自 `PipelineConfig` 并经 `_bounded` 钳制，无 env 覆盖、无越界硬编码；回退档位/应用模式/拒绝原因结构化归档 |
| D3 算法行业标准符合度 | **8.0** | Screen 混合与保色星层思路与 RC Croman / PixInsight 社区共识一致；缺「unscreen 提取」与显式几何星缩小（MT/erode），为部分符合 |

**总评（加权 0.35 / 0.35 / 0.30）：8.67 / 10。** Stage 9 实现质量高，符合 starless-first 管线的工程与美学目标。最严重问题见 §5 TOP3-1：**弱星 Screen 强度下限（0.40）与弱星恢复率门控存在耦合风险**——当强星触发 `intensity_scale` 大幅降强度时，弱星可能被同步压低至不可见，且 3 档固定步长重试的收敛性有限、非自适应。

---

## 1. Stage 9 处理链路全景（时序与文件映射）

为建立 D1/D2 的上下文，先给出 Stage 9 端到端步骤与代码落点：

1. **上游交接**：`_stage9_upstream_handoff`（stage9_star_remixing.py L20）确定 Stage 5 线性含星图、Stage 6/7 starmask、Stage 8 starless 底图；判断是否 `upstream_restricted`（受限源）。
2. **本地回退判定**：`_stage9_local_fallback`（L41）预备回滚到无星底图。
3. **强度缩放**：`remix_scale = clamp(_stage9_star_intensity_scale, 0.45, 1.0)`（L1197-1201）；受限源额外 cap `0.95/max(star_intensity,1e-6)`（L1204）。
4. **星表建立**：`build_star_reference_catalog`（stage9_quality.py L636）+ `_build_source_matched_star_catalog`（L813），产出独立星表与匹配星目录。
5. **星点层拉伸**：`_color_preserving_asinh`（L1296）或 `_color_preserving_multi_anchor_curve`（L1372）；自适应反解 `calibrate_starmask_asinh`（L1694）；色度正则化（L259-261）。
6. **星色修复**：`repair_star_layer_colors`（star_color_repair.py L90），用不可变线性含星参考，失败保留原层。
7. **混合合成**：`screen_blend`（stage9_quality.py L454-517），Alpha+Screen，星层为上层、starless 为底层。
8. **post-remix 门控**：`assess_remix`（L2123），十余类阈值经 `_bounded` 钳制（L2544-2635），`upper_limit_names` 循环检查（L2655-2659）。
9. **降强度重试**：主候选被拒 → 依次 `(0.75, 0.55, 0.40)`（models.py L235）重试，每档重新过门控。
10. **最终降级**：全档拒绝 → 回滚 Stage 8 starless，`stars_applied=false`（processor_runtime.py L2124-2137）。
11. **归档**：processing-plan.json（档位）、pipeline-result.json（应用模式）、stage9_remix_quality.json（拒绝原因）、checkpoint-manifest.json（状态）。

该链路体现了「**先建立可信星表 → 保色拉伸 → 约束混合 → 严格验收 → 安全降级**」的闭环，工程范式专业。

---

## 2. D1：逻辑专业合理性

### 2.1 星点层（starmask）建立与独立拉伸

**独立星表建立**（`stage9_quality.py`）：
- `build_star_reference_catalog`（L636）：以 `stage9_star_reference_sigma=5.0`（models.py L247）为背景 + 5σ 噪声标准差，从原始 starmask 提取**紧致组件**目录；组件密度受 `stage9_source_component_density_max=2500.0`（L245）约束，单像素噪点比例受 `stage9_source_single_pixel_ratio_max=0.20`（L246）约束，超限即**自适应收紧或 fail-closed**。
- `_build_source_matched_star_catalog`（L813）：在原始含星图（stage5_linear）用 `stage9_source_star_detail_percentile=98.0`（L244）做局部高频细节提取，再在原始 starmask 的 5×5 邻域匹配；密度/单像素污染检测 fail-closed。

**星点层拉伸（保色，核心正确性）**：
- `_asinh_map`（L1278）+ `_color_preserving_asinh`（L1296）：对每像素**峰值（peak）**做映射，增益 `gain = mapped_peak / peak`（L1306-1311），再以同一标量增益乘到 RGB 三通道（`source * gain[...]`，L1313-1317）。因为增益是**逐像素标量**，RGB 通道比例不变 → 保色成立。
- `_color_preserving_multi_anchor_curve`（L1372）：峰值比满足 `stage9_mixed_star_peak_ratio_min=4.0`（L249）且弱/亮星组件数满足 L250-251 时，启用**多锚点单调对数曲线**（`_monotonic_anchor_map` L1323，锚点严格单调递增校验 L1332-1342，输出钳位 L1351）。同样用逐像素峰值增益保色。
- 自适应反解：`calibrate_starmask_asinh`（L1694）根据星点层有效信号分布反解 asinh 强度（受 `stage9_starmask_asinh_stretch_max=1000.0` 上限 L254），统计不可用时回退固定 `stage9_starmask_asinh_stretch=2.0`（L252）。
- **偏差提示（非阻断）**：L1278-1293 的 asinh 实为自定义曲线 `(sample-offset)*arcsinh(sample*stretch)/(sample*arcsinh(stretch))`，含 `(sample-offset)/sample` 低信号衰减因子，**并非**标准 asinh 拉伸 `arcsinh(a·x)/arcsinh(a)`。属可接受的非标准实现（单调、有界、保色），但文档未标注差异（见 §7 P2-8）。

**结论**：星点层独立拉伸 + 保色增益的设计专业、正确，避免「先拉伸后回星导致星色被底图污染」这一行业经典错误（见 D3）。**D1 合理**。

### 2.2 Alpha + Screen 混合数学正确性（含数值示例）

`screen_blend`（`stage9_quality.py` L454-517）实现：

```
star_term = clip(stars_norm * intensity_map, 0, 1)            # L489-494
screened  = 1.0 - (1.0 - base_norm) * (1.0 - star_term)        # L495  标准 Screen 公式
mixed     = base_norm * (1 - alpha) + screened * alpha         # L509  Alpha 约束层
```

**验证（取 alpha=1, intensity=1）**：设 `base=0.4, stars=0.3`
- `star_term = 0.3`，`screened = 1-(1-0.4)*(1-0.3) = 1-0.6*0.7 = 1-0.42 = 0.58`
- 标准 Screen：`base+stars-base*stars = 0.4+0.3-0.12 = 0.58` ✅ 一致
- 标准 Screen 公式 `~((~starless)*(~stars))`：`1-(1-0.4)*(1-0.3) = 0.58` ✅ 与 RC Croman 社区公式完全等价（D3 佐证）

**关键性质**：对任意 `base,stars∈[0,1]`，`screened ≤ 1`，**永不溢出**——这正是 Screen 优于加法（`base+stars` 易 >1 截断导致星胀）的数学根源（D3 URL 21794 / astrobin 178601）。

**`intensity_map` 空间分区**（L470-488）：可按 `weak_mask` 提升弱星强度（取 `max(intensity, weak_intensity)`，L477-481），按 `bright_mask` 维持亮星强度（L482-488）。

**`alpha_mask` 约束层**（L499-509）：仅在星支持层内把 Screen 结果混入底图，层外保持 Stage8 starless 不变——这正是 AGENTS.md「星点作为上层、Starless 作为底层」且「受支持层约束」的工程实现。

**数学正确性**：Screen 项与 Alpha 组合均无误，且 `clip` 保证无溢出（L489-494、L504-508）。**D1 正确**。

注意：`stage9_star_remixing.py` L1206-1210 显式 `composer_used = None` 并说明「bypass StarComposer，使用显式 starmask-top/starless-bottom Alpha+Screen 合成」。属刻意选择（需约束层），合理。

### 2.3 二次星点（伪星/重影）成因与防护

**行业成因**（D3）：回星时若星层含底图已存在的非星结构、或拉伸过度放大大噪声，会在层外引入「二次星点」。

**本项目防护（纵深防御）**：
1. 前置坏蒙版检测：密度 + 单像素污染 fail-closed（§2.1）。
2. 强度自适应降级：`_stage9_star_intensity_scale`（seestar_Superimpose.py L1097）由 Stage7 残星/halo/污染触发降强度（钳制 0.35-0.95，见 `stage7_update_star_remix_from_quality` L679 区域），写入 `_stage9_star_intensity_reason`。
3. 变化覆盖门控：`stage9_changed_pixel_ratio_max=0.35`（L270）、`stage9_unmatched_changed_ratio_max=0.01`（L276）限制星支持层之外的显著变化——直接拦截层外二次星点。
4. 局部连通块门控（L280-284）：新增连通块面积/长宽比/填充率/单像素比例限制，排除细长/碎裂伪结构。
5. 青蓝色团块门控（L285-287）：拦截低信号被放大成的蓝紫伪色块。

**结论**：二次星点防护是**五道闸**（前置 + 强度 + 覆盖 + 形态 + 颜色），设计专业。**D1 合理**。

### 2.4 Halo（星芒/光晕）抑制

- 紧致支持层：Stage 9 仅保留连通紧致星核 + 窄星翼（`stage9_compact_starmask_enabled`，L238）；覆盖异常时允许重建更严格支持层（`stage9_compact_weak_star_retention_min=0.80` 保弱星 L248）。
- 亮像素增长门控 `stage9_bright_pixel_growth_max=0.025`（L266）、高光裁剪增长 `stage9_highlight_clip_growth_max=0.006`（L265）限制星点膨胀/光晕。
- 色度正则化 `stage9_starmask_chroma_regularization_enabled`（L259）：多锚点拉伸时以邻域星色约束微弱星翼，避免把单像素通道噪声放大成蓝紫色块（L260-261 通道跨度上限）。

**注意（部分符合）**：halo 抑制依赖上游 starmask 本身不含大 halo。本管线去星在 Stage 6（`stages/stage7_star_separation.py`，注意命名错位），若其去星残留大 halo，Stage 9 的 compact 规则只能缓解、不能完全消除。**D1 基本合理，强耦合上游质量**。

### 2.5 星色保真

- `_color_preserving_asinh` / `_color_preserving_multi_anchor_curve`：逐像素标量增益保色（§2.1）。
- `repair_star_layer_colors`（`star_color_repair.py` L90）：用**不可变线性含星参考**做 `reference_chroma_luminance_preserving_core_wing_repair`；默认 `strength=0.72`（L94），`support_coverage_max=0.12`（L95），`chroma_improvement_min=0.01`（L96）；覆盖超 `max(0.02, min(0.25, support_coverage_max))` 即 fail-closed（L125-126）；候选失败时保留原层（AGENTS.md 原则：禁止星色失真）。
- 色度误差上限门控 `stage9_star_color_post_chroma_error_max=0.22`（L243）：拉伸及回混前最终星层相对参考的中位色度误差上限。
- 核心颜色突变门控（L288-290）：核心区归一化 RGB 单通道最大突变量 `stage9_core_color_jump_min=0.10`，连通块面积 ≤ 64。

**结论**：星色保真采用「参考修复 + 增益保色 + 误差门控 + 突变拦截」四重机制，符合行业 RGB 星色保护最佳实践（D3）。**D1 合理**。

### 2.6 坏蒙版污染检测

- 密度 `stage9_source_component_density_max=2500.0`（L245）、单像素比例 `stage9_source_single_pixel_ratio_max=0.20`（L246）：配合 fail-closed（检测失败即拒收星表，回退安全源）。
- `_stage9_bad_starless_reason`（stage_support.py L1835）：判断 Stage 7/8 是否坏；`_stage9_review_safe_source`（L1922）选择安全回退源（stage7 review source 优先）。
- 背景抬升门控 `stage9_background_lift_max=0.010`（L267）、斑驳增长 `stage9_background_mottling_growth_max=1.35`（L268）+ 豁免条件 `stage9_mottling_exemption_changed_pixel_ratio_max=0.12`（L269）。

**结论**：坏蒙版与污染检测 fail-closed 且带结构化原因，专业。**D1 合理**。

### 2.7 降强度重试的收敛性

- 重试档位：`stage9_fallback_intensity_levels=(0.75, 0.55, 0.40)`（models.py L235）；主候选拒绝后逐档降低回星强度。
- 主强度公式（`stage9_star_remixing.py` L1334）：`intensity = _clamp_float(star_intensity * remix_scale, 0.10, 1.05)`，其中 `remix_scale = clamp(_stage9_star_intensity_scale, 0.45, 1.0)`（L1197-1201），受限源额外 cap `0.95/max(star_intensity,1e-6)`（L1204）。
- 弱星 Screen 强度下限 `stage9_weak_star_screen_intensity_min=0.40`（L274；stage_support.py L1339-1351 钳制）。

**收敛性评估**：3 档固定步长 + 主强度与 `remix_scale` 乘积，属于**确定性但有界**的离散搜索，**不具备自适应收敛保证**。若主候选因「星色误差」或「高光裁剪」被拒，降强度可能仍未达标（例如强度降到 0.40 时星色误差未必改善，反而弱星消失）。但每次重试都过 post-remix 门控，不达标即继续或回滚，**不会产出坏图**——这是正确工程权衡（fail-safe）。

**结论**：收敛性有限但安全。**D1 合理但非最优**（见 §6 P1-5）。

### 2.8 回滚到无星底图作为最终降级

- 当所有回退档位均被 post-remix 门控拒绝，`_stage9_local_fallback`（stage9_star_remixing.py L41）与 `_apply_previous_stage_star_remix`（stage_support.py L1301）回滚到 Stage 8 starless。
- 归档中 `stars_applied=false`（processor_runtime.py L2124-2137）显式记录，确保下游 Stage 10 知道没有星点。

**评估**：回滚到无星底图是**激进但最终安全**的降级（牺牲星点完整度换取无伪影成片），符合 AGENTS.md「禁止二次星点/halo」优先于「保留全部星点」的取向。**D1 合理**。

### 2.9 数值示例：多锚点曲线保色验证

以 `_color_preserving_multi_anchor_curve`（L1372）说明保色如何成立。设某蓝色星像素 RGB = (0.10, 0.15, 0.40)（蓝星，B 通道主导），其峰值 `peak = max = 0.40`。映射后（假设锚点目标使 `mapped_peak = 0.80`）：
- `gain = 0.80 / 0.40 = 2.0`（逐像素标量）
- 输出 RGB = (0.10×2.0, 0.15×2.0, 0.40×2.0) = (0.20, 0.30, 0.80)
- 通道比 0.10:0.15:0.40 = 0.20:0.30:0.80 完全不变 ✅ 蓝星色相保留

若改用分通道独立拉伸（行业禁止做法），B 通道可能被单独压低，破坏 0.10:0.15:0.40 比例 → 星色失真。Stage 9 的标量增益从机制上杜绝此问题。注意：`gain` 是空间数组（每像素一个值），故不同亮度星点获不同增益，但**单像素内三通道共用同一增益**，保色与局部自适应拉伸同时成立。

---

## 3. D2：门禁来源与归档

### 3.1 门禁穷举（file:line + 阈值来源 + 钳制）

Stage 9 全部门控阈值定义在 `models.py` L233-295（`stage9_*` 系列），运行时经 `seestar_Superimpose.py` 的 `CLAMP_FIELDS`（L421-446 列出全部 stage9 字段）+ `CLAMP_RULES`（L448-640）钳制。**未发现任何 env 覆盖或越界硬编码**。

| 门控类别 | 代表参数（models.py 行号） | 默认 | 钳制来源 | 是否钳制 |
|---------|--------------------------|------|---------|---------|
| 高光裁剪 | `stage9_highlight_clip_ratio_max` L264 / `_growth_max` L265 | 0.015 / 0.006 | CLAMP_FIELDS L421-422 | 是 |
| 亮像素/膨胀 | `stage9_bright_pixel_growth_max` L266 | 0.025 | CLAMP_FIELDS L423 | 是 |
| 背景抬升 | `stage9_background_lift_max` L267 | 0.010 | CLAMP_FIELDS L424 | 是 |
| 斑驳增长 | `stage9_background_mottling_growth_max` L268 / 豁免 L269 | 1.35 / 0.12 | CLAMP_FIELDS L425-426 | 是 |
| 变化覆盖 | `stage9_changed_pixel_ratio_max` L270 / `unmatched` L276 | 0.35 / 0.01 | CLAMP_FIELDS L427 / L428 | 是 |
| 变暗 | `stage9_darkening_ratio_max` L271 | 0.005 | CLAMP_FIELDS L429 | 是 |
| 星恢复率 | `weak_star_recovery` L272 / `star_recovery` L273 / `aperture` L291 / `wing` L292 | 0.70/0.75/0.75/0.65 | assess_remix limits | 是（_bounded） |
| 弱星强度 | `stage9_weak_star_screen_intensity_min` L274 | 0.40 | stage_support.py L1339-1351 | 是（软下限） |
| 星表/支持层 | `star_support_ratio_max` L275 / `density_max` L245 / `single_pixel` L246 | 0.12/2500/0.20 | 内部 fail-closed | 是 |
| 局部连通块 | L280-284 | 见 L280-284 | assess_remix limits（L2544-2635） | 是（_bounded） |
| 青蓝团块 | L285-287 | 见 L285-287 | 同上 | 是 |
| 核心颜色突变 | L288-290 | 见 L288-290 | 同上 | 是 |
| 伪色新增 | L277-279 | 见 L277-279 | 同上 | 是 |
| 暗坑/空心结构 | L293-295 | 见 L293-295 | 同上 | 是 |
| 星色误差 | `post_chroma_error_max` L243 / `improvement_min` L242 | 0.22 / 0.01 | 内部 clamp | 部分（见 §3.3） |
| 主强度 | `star_intensity` L233 | 1.0 | CLAMP_RULES L499 `(0.8,1.05)` | 是 |
| 首档回退强度 | `star_fallback_intensity` L234 | 0.95 | CLAMP_RULES L500 `(0.75,1.05)` | 是 |
| 重试档位 | `fallback_intensity_levels` L235 | (0.75,0.55,0.40) | 元组常量 | 是（离散） |
| 强度缩放 | `remix_scale` | 由 Stage7 写 | stage9_star_remixing.py L1197-1201 `[0.45,1.0]` | 是 |

**主强度钳制**：`intensity = _clamp_float(star_intensity * remix_scale, 0.10, 1.05)`（stage9_star_remixing.py L1334）；`star_intensity` 本身由特征公式 `1.08 - 0.17*star_score - 0.08*diffuse_score + 0.05*core_score`（seestar_Superimpose.py L892）计算并钳制 `(0.8, 1.05)`（L499）。

**assess_remix（stage9_quality.py L2123）**：所有阈值经 `_bounded(value, default, lower, upper)` 钳制（L2544-2635 limits 字典），`upper_limit_names` 循环检查（L2655-2659）。未发现越界。

### 3.2 阈值来源分类

- **A 类（PipelineConfig 默认 + CLAMP_RULES 钳制）**：绝大多数 `stage9_*` 参数，来源透明、可调。✅
- **B 类（特征公式推导）**：`star_intensity`（L892）由图像特征（star_score/diffuse_score/core_score）推导，仍经 CLAMP_RULES 钳制。✅
- **C 类（内部自适应反解）**：starmask asinh 自适应强度（L1694）、`remix_scale` 由 Stage7 诊断写入。⚠️ 来源在代码内可见，但未单独在配置项暴露——可审计性稍弱（见 §6 P2-7）。
- **D 类（硬编码常数，合理）**：如 `_asinh_map` 的 `1e-12` 防除零、色度 `1e-6` 阈值——属数值稳定性常数，非门控阈值。✅

### 3.3 钳制（clamp）情况汇总

- 全部 `stage9_*` 字段在 `CLAMP_FIELDS`（L421-446）+ `CLAMP_RULES`（L448-640）中列出并钳制。
- 主强度 `star_intensity ∈ [0.8, 1.05]`、`star_fallback_intensity ∈ [0.75, 1.05]`（L499-500）。
- `remix_scale ∈ [0.45, 1.0]`（stage9_star_remixing.py L1197-1201）。
- `intensity ∈ [0.10, 1.05]`（L1334）。
- `weak_star_screen_intensity_min = 0.40` 为软下限（取 max），非钳制区间。
- **未发现**未钳制的外部输入（env/CLI/文件）影响任何门控阈值——攻击面封闭。

### 3.4 结构化归档（关键：混合强度 / 重试次数 / 是否回滚）

| 产物 | 字段 | 含义 | 位置 |
|------|------|------|------|
| processing-plan.json | `candidate_contracts.stage9_star_remix.fallback_intensity_levels` | 降级强度档位（0.75/0.55/0.40） | processor_runtime.py L1962-1966 |
| pipeline-result.json | `star_separation.stars_required` / `stars_applied` / `application_mode` | 是否要求星点、是否实际回星、应用模式 | processor_runtime.py L2124-2137 |
| stage9_remix_quality.json | 拒绝原因 / 重试轮次 / 最终档位 | post-remix 门控拒绝原因结构化记录 | seestar_Superimpose_workflow.md L735-736 |
| checkpoint-manifest.json | Stage9 状态 | 阶段完成/跳过/回滚 | run_manifest.py |
| 日志 | `_stage9_star_intensity_reason`、`messages.append(...)` | 人类可读降级原因（如「stage7 residual stars」） | stage9_star_remixing.py L1336-1345 |

**回滚归档**：当回滚到无星底图，`stars_applied=false` 被显式写入（L2124-2137），下游 Stage 10 据此跳过星相关处理。**结构化、可追溯**。

**归档缺口（非阻断）**：
- 重试每档的中间指标（如第 2 档为何被拒的具体超标项数值）主要落在 `stage9_remix_quality.json`，但**跨档对比所需的完整 trace** 是否全量落盘未逐字段验证（见 §7 未验证项）。
- 部分 runtime 原因（如 `upstream_restricted` 提示）仅入日志，未进 pipeline-result.json 的确定性字段。

**结论**：D2 优秀——阈值来源透明、钳制完备、回退/强度/回滚均结构化归档。**评分 9.0**。

---

## 4. D3：算法行业标准符合度

> 以下结论均附可访问 URL。对标判定：符合 / 部分符合 / 偏离 / 自创。

### 4.1 Screen 混合作为回星标准（符合）

- **RC Croman（StarXTerminator 作者）**官方用法笔记：非线性阶段回星应使用 Screen 混合（PixelMath `~((~starless)*(~stars))` 或 `combine(Stars, Starless, op_screen())`），且「for best results when recombining the (perhaps separately processed) starless and stars images, do it after stretching both, and use screen blending in PixelMath」。
  - URL：https://www.rc-astro.com?p=1781/ （StarXTerminator Usage Notes）
  - URL：https://www.pixinsight.com/forum/index.php?threads/unscreening-and-re-screening-recombining-stars-with-starless-images.18602/ （RC Croman「Unscreening and re-screening」原帖，给出 `~((~original)/(~starless))` 提取 + 重新 Screen 的完整方法）
- **PixInsight 社区共识**：`combine(Stars_image, Starless_image, op_screen())` 是 Adam Block 等推崇的 staple；`starless+stars` 简单相加易过曝/星胀。
  - URL：https://pixinsight.com/forum/index.php?threads/adding-stars.17837/
  - URL：https://pixinsight.com/forum/index.php?threads/benefit-of-starless-stars-compared-to-starless-stars.21794/ （screen 永不超出 1.0，避免高光过曝；`stars+starless` 在值 >1 时被截断导致星胀）
  - URL：https://ssr.app.astrobin.com/forum/topic/178601/processing/adding-stars-to-starless-image-causes-them-to-bloat-what-am-i-doing-wrong （star bloat 来自加法，screen 防止）

**对标**：Stage 9 `screen_blend`（L454-517）数学与社区 Screen 公式一致，**符合**。差异：Stage 9 额外叠加 Alpha 约束层与 intensity 缩放（自创增强，见 §4.5），属合理工程扩展。

### 4.2 星层单独拉伸 + 保色（符合）

- 行业做法：星层与 starless 层**分别拉伸**后再回星（RC Croman 用法笔记；Siril Star Recomposition 提供独立「Background Stretch」与「Star Stretch」滑块）。
  - URL：https://www.rc-astro.com?p=1781/
  - URL：https://bridgecameraastroimaging.blogspot.com/2024/03/using-starnet-in-siril.html （Siril Star Recomposition：背景与星点拉伸参数独立可调）
- 保色原则：RGB 星色保护要求对星层做**通道一致的标量增益**，而非分通道独立拉伸（否则星色失真）。
  - URL：https://astrosource.net/resources/rgb-stars （RGB 星重组：对 starred 与 starless 必须施加相同调整以保色，否则产生伪影）
  - URL：https://astrobackyard.com/rgb-stars-narrowband-images/ （以 Screen 叠加 RGB 星，保护自然星色）

**对标**：Stage 9 `_color_preserving_asinh` / `_color_preserving_multi_anchor_curve` 用逐像素标量增益保色（§2.1），与行业保色原则一致，**符合**。

### 4.3 星色修复用不可变参考（符合）

- 行业做法：从原始含星参考提取星色，避免底图污染导致的色偏（RC Croman unscreen 法的核心动机：减法星层会被底图颜色「污染」，unscreen 才能恢复真色）。
  - URL：https://www.pixinsight.com/forum/index.php?threads/unscreening-and-re-screening-recombining-stars-with-starless-images.18602/ （减法星层「mis-colored because we subtracted the nebula color that was behind them」）

**对标**：Stage 9 `repair_star_layer_colors`（star_color_repair.py L90）用**不可变线性含星参考**做 chroma 修复 + 误差门控，直接对应 unscreen 的「恢复真色」目标，**符合**。

### 4.4 星缩小 / 星尺寸控制（部分符合 / 偏离）

- 行业主流星缩小手段：**形态学变换（MorphologicalTransformation）**、**最小滤波（minimum/erosion filter）**、BlurXTerminator / StarShrink、多尺度中值变换（MSMT/MLT）。
  - URL：https://deepskycolors.com/tools-tutorials/star-size-reduction-via-morphological-transformations （MT 缩小星尺寸，仅 dim 而非伪造数据）
  - URL：http://astroscape.nl/fototext/Star_erosion.html （minimum filter 侵蚀星足印）
  - URL：https://www.astrobin.com/forum/c/astrophotography/deep-sky-processing-techniques/how-can-i-remove-just-the-small-stars/ （MT / StarShrink / MSMT 缩小星）
  - URL：https://starfyi.com/es/guide/processing-astro-images （StarNet++ 降 opacity 重组；PixInsight StarReduction 用形态学变换）

**对标**：Stage 9 **没有显式星缩小算子**（无 erosion / MT / StarShrink）。它通过 `intensity` 缩放与 `remix_scale` 降强度来「减弱」星点，并通过亮星多锚点目标亮度 `stage9_starmask_bright_target=0.75`（L257）、`peak_target=0.90`（L258）限制亮星峰值——属**亮度/强度抑制**而非**几何尺寸缩小**。

**判定**：部分符合。对「星过亮」有效；对「星几何胀大（bloat）」无直接几何修正。若上游 starmask 本身星尺寸偏大，Stage 9 无法收缩其足印（仅抑亮度）。行业 trap「star bloat from addition」已被 Screen 规避，但几何 bloat 仍需上游控制。建议见 §6 P1-3。

### 4.5 Alpha+Screen 约束层 + 自适应强度（自创 / 合理扩展）

- 行业回星通常为「全图 Screen」或「masked Screen」。Stage 9 引入：
  1. `alpha_mask` 约束层（L499-509）：仅在星支持层内混 Screen，层外保持 starless——比社区 mask 更严格（mask 通常决定哪里加星，本实现还约束混合比例）。
  2. `intensity_map` 空间可变（弱星/亮星分区，L470-488）。
  3. `remix_scale` 由 Stage7 诊断驱动（L1197-1201）。
- 这些属**自创增强**，数学正确且目标明确（防二次星点 + 保弱星），无行业反例表明其有害。**判定：自创（合理）**。

### 4.6 行业已知陷阱的规避核查

| 行业陷阱 | 来源 URL | Stage 9 是否规避 |
|---------|---------|----------------|
| 简单加法回星导致过曝/星胀 | 18602 / 21794 / astrobin 178601 | ✅ Screen（L495）规避 |
| 减法星层被底图污染致星色失真 | 18602 | ✅ unscreen 思路 + reference 修复（L90）规避 |
| 拉伸后分通道独立拉伸破坏星色 | astrosource.net/rgb-stars | ✅ 标量增益保色（L1313-1317）规避 |
| 非线性重组星色漂移 | 21794（Mike Cranfield 注） | ⚠️ 部分：Stage 9 在像素空间重组，依赖 starless 已是非线性；伪线性做法未采用 |
| 大 halo 残留 | astrobackyard StarNet++ | ⚠️ 依赖上游去星质量（§2.4） |
| 弱星在降强度时被误杀 | （审计推断） | ⚠️ 见 §5 TOP3-1 |

### 4.7 D3 综合判定

- Screen 混合：**符合**社区标准。
- 星层独立保色拉伸：**符合**。
- 参考色修复：**符合** unscreen 精神。
- 星色保护流程：**符合** RGB 星重组最佳实践。
- 星几何缩小：**部分符合**（仅亮度抑制，无几何收缩）。
- unscreen 提取法：本项目用 starmask 分离而非 unscreen 反解——属等效替代（已有独立星表），**可接受**。

**D3 评分 8.0**（扣分给「缺几何星缩小」与「非伪线性重组」）。

### 4.8 与 Siril 原生 Star Recomposition 的差异对标

Siril（本项目底层引擎）自带 Star Recomposition 对话框，提供独立的 Background Stretch 与 Star Stretch 滑块，本质也是「星层与底图分别拉伸后重组」。
- URL：https://bridgecameraastroimaging.blogspot.com/2024/03/using-starnet-in-siril.html

**差异**：Siril 原生重组依赖用户在 GUI 中手工拉滑块，且混合模式相对固定；Stage 9 将其**全自动化 + 保色增益 + 约束 Alpha + 多级验收**，并以 `stage9_*` 配置项驱动。Stage 9 刻意 bypass StarComposer（`composer_used=None`，L1206-1210）正是因为需要约束层与验收闭环，原生重组不具备这些。判定：**自创自动化封装，符合 Siril 重组的设计意图且超出其能力**。

### 4.9 与 StarNet++ 官方回星建议的对标

StarNet++ 官方文档建议「stretch 后分离、分别处理、Screen 或 PixelMath 重组」，并强调不要对星层做 STF 自动拉伸以免破坏信号。
- URL：http://skyatnightmagazine.com/astrophotography/astrophoto-tips/pixinsight-enhance-galaxy-brightness-without-affecting-stars
- URL：https://www.cloudynights.com/topic/721758-experimenting-with-starnet-workflows/ （线性星提取：unstretch → StarNet++ → 再 subtract）

**对标**：Stage 9 在非线性域做保色拉伸（而非 STF 自动拉伸），避免 StarNet++ 警告的「自动拉伸破坏星信号」问题；回星用 Screen（与官方建议一致）。但 Stage 9 的星层来自本管线去星（非 StarNet++），属等效替代。**判定：符合**。

### 4.10 结论置信度分级

| 结论 | 置信度 | 依据 |
|------|--------|------|
| Screen 数学正确 | 高 | 公式逐行核对 + 数值示例验证（§2.2） |
| 保色增益成立 | 高 | 标量增益机制 + 数值示例（§2.9） |
| 阈值全钳制无越界 | 高 | CLAMP_FIELDS/RULES 逐行核对（§3.1-3.3） |
| 弱星耦合风险 | 中高 | 静态推导，未运行管线实测（见未验证项） |
| 几何 bloat 缺口 | 高 | 代码中确无 erosion/MT 算子（§4.4） |
| 上游 halo 残留影响 | 中 | 依赖 Stage 6 去星质量，未实测（§2.4） |

---

## 5. 残留风险矩阵（D1×D2×D3 交叉）

| 风险 | D1 影响 | D2 可见性 | D3 对标 | 严重度 |
|------|--------|----------|--------|--------|
| 弱星被强星降强度连带压低（TOP3-1） | 弱星消失/回滚 | 有 reason 字段但未区分强弱星 | 偏离「保弱星」隐含预期 | 高 |
| 上游 starmask halo 残留 | halo 复发 | 背景/膨胀门控部分拦截 | 行业 trap 未完全规避 | 中 |
| 重试非自适应（TOP3-3） | 计算浪费/易回滚 | 档位已归档 | 无行业反例但非最优 | 中 |
| 几何 bloat 无收缩（TOP3-2） | 星足印偏大 | 无对应门控 | 部分符合 | 中 |
| 非标准 asinh 曲线 | 低（有界单调） | 文档未标注 | 偏离标准形式 | 低 |

---

## 6. 关键发现 TOP3

### TOP3-1（最严重）：弱星强度下限与弱星恢复率门控的耦合风险
`stage9_weak_star_screen_intensity_min=0.40`（models.py L274）是弱星 Screen 软下限；但当强星触发 `_stage9_star_intensity_scale` 大幅降强度（如 Stage7 残星/halo），主 `intensity` 可能降到 0.45×0.40 量级，弱星实际强度同步被压低至接近消失。此时 `stage9_weak_star_recovery_ratio_min=0.70`（L272）门控可能因弱星消失而拒绝候选，但 3 档固定重试（0.75/0.55/0.40）无针对「仅弱星不足」的自适应补偿——最终可能回滚到无星底图，**牺牲全部弱星**。这是「防二次星点」与「保弱星」目标间未完全解耦的风险。

### TOP3-2：星点层无几何尺寸控制（仅亮度抑制）
Stage 9 通过 intensity / remix_scale / 多锚点目标亮度抑制星亮度，但无 erosion / morphological 缩小算子（§4.4）。若上游 starmask 星足印偏大，Stage 9 无法收缩几何尺寸，只能靠 compact 支持层缓解。行业主流用 MT/StarShrink 做几何缩小。

### TOP3-3：主强度与重试档位的收敛性非自适应
3 档固定步长 + `remix_scale` 乘积（§2.7）是离散有界搜索，不保证在「星色误差」类拒绝下收敛；每次重试都过门控（fail-safe），但可能浪费计算并更易触发回滚。CLAMP_RULES 完备，但搜索策略偏保守。

---

## 7. 改进建议

### P0（必须，影响成片正确性）
1. **解耦弱星与强星强度**（对应 TOP3-1）：在 `screen_blend` 的 `weak_mask` 分支（L477-481）已支持弱星独立强度，但主 `intensity` 降强度时会连带压低弱星。建议当 `remix_scale` 降低时，**保持弱星 Screen 强度不低于 `stage9_weak_star_screen_intensity_min=0.40` 的独立档**，并让弱星恢复率门控（L272）只评估弱星通道，避免强星问题拖垮弱星。
2. **重试档位与拒绝原因联动**：当前 3 档固定（L235），建议按 `stage9_remix_quality.json` 记录的首 reject 原因（星色 vs 高光 vs 覆盖）选择重试策略——星色类拒绝不应靠降强度解决（降强度不改善色误差），应转交 `repair_star_layer_colors` 或回滚。

### P1（重要，提升质量/稳健性）
3. **增加几何星缩小算子**（对应 TOP3-2）：在星点层拉伸前对 starmask 做轻量 morphological erosion / 最小滤波（参考 deepskycolors.com 与 astroscape.nl 方法），收缩星足印，减少 halo 与 bloat 残留，降低对上游去星质量的依赖。
4. **采用伪线性重组路径**（对应 D3 §4.6）：对非线性 starless + 非线性 stars 的 Screen 重组，可参考 Mike Cranfield「reverse-stretch → recombine → re-stretch」保留星色（URL：21794），或显式在校准域内做重组再拉伸。
5. **重试自适应步长**：将固定 `(0.75,0.55,0.40)` 改为基于首候选超标幅度的二分/线性插值搜索，减少回滚概率。

### P2（建议，可维护性/可审计性）
6. **归档增强**：将每档重试的完整超标项数值（不仅是最终原因）写入 `stage9_remix_quality.json`，便于跨档对比（§3.4 缺口）。
7. **C 类阈值外显**：`remix_scale` 推导（L1197-1201）与 asinh 自适应反解（L1694）虽在代码内可见，建议在 processing-plan.json 记录其最终取值与触发原因，提升可审计性。
8. **文档标注非标准 asinh**：`_asinh_map`（L1278）曲线与标准 asinh 不同，建议在 `seestar_Superimpose_workflow.md` 注明，避免维护者误用。
9. **命名错位提示**：Stage 6 去星在 `stages/stage7_star_separation.py`、Stage 7 拉伸在 `stages/stage6_stretching.py`（Stage 9 文件名正常）。虽不影响 Stage 9 审计，但建议仓库级重命名以降低误读（非本次修改范围）。

---

## 8. 证据索引

### 代码证据（文件:行号）
- 主强度公式与钳制：`pipeline/stages/stage9_star_remixing.py` L1334（`_clamp_float(star_intensity*remix_scale, 0.10, 1.05)`）、L1197-1201（remix_scale 钳制）、L1204（受限源 cap）、L1206-1210（bypass StarComposer）、L41（`_stage9_local_fallback`）、L20（`_stage9_upstream_handoff`）
- Alpha+Screen 混合数学：`pipeline/stage9_quality.py` L454-517（screen_blend；L489-494 star_term clip、L495 screened、L509 mixed）
- 保色拉伸：`pipeline/stage9_quality.py` L1278-1293（`_asinh_map`）、L1296-1320（`_color_preserving_asinh`）、L1323-1351（`_monotonic_anchor_map`）、L1372-1405（`_color_preserving_multi_anchor_curve`）、L1694（`calibrate_starmask_asinh`）
- 星表建立：`pipeline/stage9_quality.py` L636（`build_star_reference_catalog`）、L813（`_build_source_matched_star_catalog`）、L2123（`assess_remix`）、L2544-2635（limits 字典）、L2655-2659（upper_limit_names）
- 星色修复：`pipeline/star_color_repair.py` L90-140（`repair_star_layer_colors`、strength=0.72、coverage fail-closed L125-126）
- 强度/质量支撑：`pipeline/stage_support.py` L1301（`_apply_previous_stage_star_remix`）、L1339-1351（弱星强度钳制）、L1389（`_stage9_assess_current_remix`）、L1835（`_stage9_bad_starless_reason`）、L1922（`_stage9_review_safe_source`）
- 配置默认值：`pipeline/models.py` L233-295（全部 `stage9_*` 默认与注释）
- 钳制规则：`pipeline/seestar_Superimpose.py` L355-356（CLAMP_FIELDS 含 star_intensity）、L421-446（CLAMP_FIELDS stage9 字段）、L448-640（CLAMP_RULES）、L499-500（star_intensity/fallback 钳制）、L892（特征公式）、L1097-1098（`_stage9_star_intensity_scale`）
- Stage7 触发降强度：`pipeline/stage7_quality.py` L679 区域（`stage7_update_star_remix_from_quality` 写 `_stage9_star_intensity_scale` / `_stage9_star_intensity_reason`）
- 归档：`pipeline/processor_runtime.py` L1870-1998（`_write_processing_plan`，L1962-1966 fallback 档位）、L2038-2174（`_write_pipeline_result_manifest`，L2124-2137 `stars_required/stars_applied/application_mode`）、`pipeline/run_manifest.py`（读写）

### 行业证据（URL）
- RC Croman StarXTerminator 用法 + Screen 回星：https://www.rc-astro.com?p=1781/
- RC Croman「Unscreening and re-screening」原文（~((~original)/(~starless))）：https://www.pixinsight.com/forum/index.php?threads/unscreening-and-re-screening-recombining-stars-with-starless-images.18602/
- PixInsight 社区 Screen 回星（op_screen / Adam Block）：https://pixinsight.com/forum/index.php?threads/adding-stars.17837/
- Screen vs Add 优劣（防过曝/星胀）：https://pixinsight.com/forum/index.php?threads/benefit-of-starless-stars-compared-to-starless-stars.21794/
- 加法导致 star bloat：https://ssr.app.astrobin.com/forum/topic/178601/processing/adding-stars-to-starless-image-causes-them-to-bloat-what-am-i-doing-wrong
- StarNet++ 工作流 / 线性星提取：https://www.cloudynights.com/topic/721758-experimenting-with-starnet-workflows/
- Siril Star Recomposition（背景/星独立拉伸）：https://bridgecameraastroimaging.blogspot.com/2024/03/using-starnet-in-siril.html
- RGB 星重组保色（相同调整原则）：https://astrosource.net/resources/rgb-stars
- RGB 星以 Screen 叠加（窄带）：https://astrobackyard.com/rgb-stars-narrowband-images/
- 形态学星缩小（MT）：https://deepskycolors.com/tools-tutorials/star-size-reduction-via-morphological-transformations
- 最小滤波侵蚀星足印：http://astroscape.nl/fototext/Star_erosion.html
- MSMT/StarShrink 缩小星：https://www.astrobin.com/forum/c/astrophotography/deep-sky-processing-techniques/how-can-i-remove-just-the-small-stars/
- StarNet++ 降 opacity 重组 / StarReduction：https://starfyi.com/es/guide/processing-astro-images

### 未验证项（声明，严禁编造）
- 「跨档重试完整超标项数值是否全量落盘」：仅确认 `stage9_remix_quality.json` 记录最终拒绝原因（workflow.md L735-736），未逐字段核对中间档 trace 是否持久化。
- 「回滚到无星底图时 Stage 10 是否完全跳过星相关处理」：确认 `stars_applied=false` 写入（processor_runtime.py L2124-2137），但 Stage 10 消费该字段的分支未读取验证。
- 「上游 starmask 实际 halo 残留量级」：依赖 Stage 6 去星质量，本次未运行管线生成真实 starmask 做实测（纯静态审计）。
- 「特征公式 star_score/diffuse_score/core_score 的具体取值分布」：见 seestar_Superimpose.py L892，但其输入特征的计算实现未逐行追踪。
- 「`_bounded` 函数对全部 assess_remix 阈值的实际钳制边界」：确认 limits 字典（L2544-2635）结构，但未逐一运行验证每个 upper/lower 数值边界。

---

*报告结束。本审计未修改任何项目代码，所有结论基于静态代码阅读与公开行业标准文献。总篇幅约 480 行。*
