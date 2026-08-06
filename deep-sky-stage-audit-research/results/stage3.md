# Stage 3 背景提取 / 梯度去除 —— 三维度深度审计报告

| 项 | 内容 |
| --- | --- |
| 审计对象 | `/Users/mz/dev/aiseestart` Stage 3（背景提取 / 梯度去除） |
| 主实现 | `pipeline/stages/stage3_background_extraction.py`（3314 行） |
| 核心依赖 | `pipeline/background_sampling.py`（1781 行）、`pipeline/image_metrics.py`、`pipeline/stage_support.py`、`pipeline/policy_selector.py`、`pipeline/models.py`、`pipeline/seestar_Superimpose.py` |
| 引擎 | Siril 1.4.0（`subsky` Poly/RBF），外部回退 GraXpert / ADBE / DBE / AutoDBE / NOX |
| 审计日期 | 2026-08-05 |
| 审计方式 | 纯静态代码审计 + 联网行业对标；**未运行任何代码、未修改任何项目文件** |
| 说明 | 本仓命名存在错位（Stage6 去星在 `stage7_star_separation.py`、Stage7 拉伸在 `stage6_stretching.py`），但 Stage 3 无此问题，文件名与阶段号一致 |

---

## 一、摘要与总评分

| 维度 | 评分 | 一句话理由 |
| --- | --- | --- |
| **D1 逻辑专业合理性** | **8.5 / 10** | 三态决策由「线性域 + source-mask 真实天空 + 冻结留出集 + 3σ 显著性」构成的过程证据驱动，复合拟合有事务回滚与目标保真门，专业性显著高于常规开源 pipeline；扣分在于「是否足够干净（`sufficient`）」与候选排序仍完全由旧经验分数决定，与硬门的统计学基准不同源。 |
| **D2 门禁来源与归档** | **7.5 / 10** | 归档极为完整（`background_quality_report.json` 落盘全量 process evidence + 每候选门禁快照 + 回滚事件；`processing-plan.json` 冻结完整 cfg；`pipeline_policy.json` 冻结完整 policy），**「仅凭归档复现 preserve vs apply」可以做到**；扣分在于存在一批决定分流/授权的硬编码字面量既不在 cfg 也不在 policy，且归档无代码版本指纹。 |
| **D3 算法行业标准符合度** | **8.0 / 10** | 「样点法 + 低阶模型优先 + 减法校正 + 乘性梯度改走 flat 复核」完全符合 Siril / PixInsight / photutils 主流共识；「Polynomial→残差 RBF 串联」在 PixInsight `GradientCorrection` 的 *Simplified Model → 主模型* 结构中有明确先例；扣分在于未导出合成背景模型（行业通行能力）、未支持 division 校正路径、RBF 平滑参数未与证据强度联动。 |

**总评（一句话）**：Stage 3 已经从「阈值打分制」进化为「过程证据 + 统计显著性 + 事务回滚」的可审计架构，工程完成度在开源/离线 pipeline 中属上乘；当前最大的结构性缝隙是**硬门已升级为 3σ 证据制、但排序与充分性判定仍停留在旧经验分数制**，两套标准共存导致「安全」与「够好」的判据不同源。

---

## 二、D1 逻辑专业合理性

### 2.1 阶段目标与上下游契约

- **目标**（`pipeline/stages/stage3_background_extraction.py:1334-1343` docstring）：在线性域建模并减除低频天空梯度；条纹 / walking noise 分流；每候选质量门控；失败逐级回退。
- **上游契约**：输入为 Stage2 输出 `stage2_corrected`（`stage3_background_extraction.py:1352` 用作 profile preflight 的元数据候选）；进入时立刻冻结不可变基线 `stage3_bg_input`（`:1365-1375`）。
- **下游契约**：输出固定命名 `stage3_bgremoved`（`:1729`、`:3021`），即使 preserve / review_required 分支也必须保存该 stem（`:1729`），保证下游 Stage4 色彩定标的输入名恒定 —— **这是正确设计**，避免「跳过阶段导致下游找不到文件」的常见坑。
- **副产物**：`pipeline._stage3_background_decision`（`:1720`）、`pipeline._stage3_pattern_noise_report`（`:1667`、`:3016`）、`pipeline._stage3_safe_sample_report`（`:1686`）、`pipeline._background_review_required`（`:1732`、`:2858`、`:2986`、`:3012`）。

### 2.2 实际处理链（按代码执行顺序）

| # | 步骤 | 位置 |
| --- | --- | --- |
| 1 | target profile preflight | `stage3_background_extraction.py:1349-1354` |
| 2 | 读取 policy 与 `stage3_background` 策略块 | `:1356-1363` |
| 3 | **冻结不可变基线** `save stage3_bg_input` | `:1365-1375` |
| 4 | 定义 `restore_baseline()` 事务回滚闭包 | `:1377-1402` |
| 5 | before 特征 / 像素 / adaptive 指标采样 | `:1404-1414` |
| 6 | 方向性噪声分析 `analyze_directional_pattern_noise` | `:1419-1446` |
| 7 | 安全样点构建 `build_safe_background_samples` | `:1502` |
| 8 | 确定性拆分拟合/留出集 `split_background_sample_points` | `:1552` |
| 9 | 留出基线量测 `measure_background_validation` | `:1560-1602` |
| 10 | **过程证据判定** `assess_background_process` | `:1619-1639` |
| 11 | 梯度 vs 方向噪声分流 `select_background_route` | `:1640-1666` |
| 12 | 复合拟合准入硬闸 `_stage3_compound_target_guard` | `:1671-1676` |
| 13 | **三态决策** `_stage3_background_decision` | `:1695-1721` |
| 14 | 非 apply → 回滚 + 保存 + 写归档 + 记录阶段并**返回** | `:1727-1848` |
| 15 | apply → 内置候选评估 → 复合候选 → 外部回退 → 选优 → 写归档 | `:2915-3314` |

### 2.3 域假设：线性域执行

- **判据来源**：`assess_background_process`（`pipeline/background_sampling.py:1284-1287`）读取 `input_profile.state == "linear"` 且 `safe_for_linear_steps is not False`，否则写入硬阻断 `input_linearity_not_confirmed`（`:1330-1331`）。
- **上游供给**：`pipeline.input_profile` 由 `processor_runtime.py:1489-1541` 的 `_resolve_input_profile()` → `infer_input_profile()` 产生并落盘 `input_profile.json`。
- **专业判断**：**正确**。减法背景校正必须在线性域进行（拉伸后减法会破坏通量比例关系），Siril 官方文档也把 subtraction 定义为「校正加性效应（光污染/月光）」，隐含线性域前提（<https://siril.readthedocs.io/en/stable/processing/background.html>）。
- **⚠ 风险**：`self.input_profile` 在 `seestar_Superimpose.py:1107` 初始化为 `{}`，在 resume 分支 `:2355` 也被重置为 `{}`。若 Stage 3 在 `input_profile` 仍为空 dict 的状态下执行，`state` 解析为 `"unknown"` → `linear_confirmed=False` → 硬阻断 → **永远 review_required，永不 apply**。这是 fail-closed（安全方向），但会表现为「背景提取静默失能」。resume 路径在 `:2397`/`:2425` 会调用 `_resolve_input_profile()` 重建，故常规路径应无问题 —— **未验证：未运行代码，无法确认所有 resume 分支都在 Stage3 之前完成重建**。

### 2.4 三态决策（preserve / review_required / apply）证据链

生产判定逻辑集中在 `_stage3_background_decision`（`stage3_background_extraction.py:1077-1122`），完全由 `process_report` 驱动：

| 判定 | 触发条件 | 位置 |
| --- | --- | --- |
| `review_required` | `hard_block_reasons` 非空 | `:1085-1093` |
| `apply` | `should_evaluate == True`（等价于「无硬阻断 且 mechanism == additive_low_frequency_gradient」） | `:1094-1104` |
| `preserve` | `mechanism == no_measurable_low_frequency_gradient` | `:1105-1115` |
| `review_required`（兜底） | 其余 mechanism（乘性渐晕 / 方向噪声 / 天空受限） | `:1116-1122` |

`mechanism` 的产生（`background_sampling.py:1313-1327`）为互斥五分支：

1. `target_signal_or_sky_limited` —— 安全样点或留出集未就绪，或可用天空网格 < 8 格（`:1292`）；
2. `directional_pattern_or_walking_noise` —— 检出方向性图案且无显著空间变化（`:1316`）；
3. `multiplicative_vignetting_or_flat_error` —— 径向模型显著优于平面模型（`:1319`）；
4. `additive_low_frequency_gradient` —— 留出天空 patch 中值的 P10–P90 跨度超过 3σ 中值标准误（`:1322`）；
5. `no_measurable_low_frequency_gradient` —— 兜底（`:1326`）。

**空间显著性的统计构造**（`background_sampling.py:1237-1244`）：
```
median_standard_error = 1.2533 * patch_MAD / sqrt(patch_pixels)   # 中值的渐近标准误
significance_limit    = max(3.0 * median_standard_error, 1e-12)
spatial_variation_supported = (P90 - P10) > significance_limit
```
**专业评价**：`1.2533 = sqrt(pi/2)`，是正态分布下中值相对均值的标准误放大因子，用法正确；以 MAD 而非 std 估计尺度对星点残留稳健。以「留出天空 patch 中值跨度 vs 采样不确定度」而非「经验梯度分数」作为「是否存在梯度」的判据，**这是本 Stage 最专业的一处设计**，直接对应 PixInsight 官方对 DBE 残差可信性的质疑（Vicent Peris：「我们无法分辨残差是真实结构还是梯度残留」，<https://pixinsight.com/tutorials/multiscale-gradient-correction>）。

**加性 vs 乘性的判别**（`background_sampling.py:1215-1251`）：对留出样点同时拟合一阶平面 `[1, x, y]` 与径向模型 `[1, r²]`，用 BIC 比较，要求 `radial.bic + 6.0 < plane.bic` 且径向边缘落差为负且超 3σ，才判定为乘性渐晕并**拒绝减法、改走 master-flat 复核**。这与 Siril 文档「division 用于校正乘性现象（渐晕），但这类问题应当用 master-flat 修正」的表述完全一致（<https://siril.readthedocs.io/en/stable/processing/background.html>）。ΔBIC ≥ 6 对应 Kass–Raftery「strong evidence」量级，取值合理。

### 2.5 安全样点选取

`build_safe_background_samples`（`background_sampling.py:414-746`）：

- **source_mask / coverage_mask**：`_build_source_and_coverage_masks`（`:101-210`）用迭代二次拟合 + 连贯结构生长构造源掩膜，并输出 `usable_sky_fraction` / `usable_sky_grid_cells`（`:205`）。这直接对应 photutils `Background2D` 的 `mask`（源/坏像元）与 `coverage_mask`（无覆盖区）双掩膜设计（<https://photutils.readthedocs.io/en/latest/user_guide/background.html>），概念对齐度高。
- **逐 patch 拒绝**：亮度分位（默认 0.70）、纹理分位（默认 0.55）、星点超出限、削波检查，拒绝计数落盘 `rejection_counts`（`:738`）。
- **空间覆盖硬性要求**（`:698-704`）：样点数 ≥ `min_count`、覆盖 ≥ 3 个象限、≥ 8 个网格单元、x 跨度 ≥ 0.55、y 跨度 ≥ 0.55；不满足则 `status="insufficient_safe_coverage"` 且 `points=[]`（`:739`）—— **fail closed，不降级放行**。
- **最小间距**：`max(patch_radius*2, 0.035*min(H,W))`（`:664`），避免样点扎堆导致 RBF 局部过拟合。
- **安装审计**（`stage3_background_extraction.py:712-878`）：调用 `set_image_bgsamples(recalculate=True)` 后**回读**实际样点，逐一与请求集合做 0.75 px 匹配；出现「Siril 返回了审计集合之外的样点」直接 `RuntimeError`（`:787-790`），保留数少于 `stage3_safe_sample_min_count` 也拒绝（`:803-807`）。
  **专业评价**：这是对 Siril `subsky -existing` 语义的正确加固。Siril 官方明确 `-existing` 之外会「自动重新生成背景样点」（<https://siril.readthedocs.io/en/stable/processing/background.html>），而本实现进一步防御了「Siril 内部重算后偷偷改动样点集」的情形。**行业内少见的严谨度。**

### 2.6 Polynomial→残差 RBF 复合候选：是否过拟合？

**结构**（`stage3_background_extraction.py:2226-2852`）：先 `subsky 1 -existing`（一阶多项式，`:2353`），再复用「已通过硬门的最佳 RBF 候选命令」对残差建模（`:2354`）。

**反过拟合防线（共 8 道，全部 fail-closed）**：

| # | 防线 | 位置 |
| --- | --- | --- |
| 1 | 目标类型硬闸：弥散/暗云气/低对比/方向噪声一律禁用复合 | `:569-600`（`_stage3_compound_target_guard`） |
| 2 | 单阶段内置候选**已足够**则完全不触发复合 | `:2926-2933` |
| 3 | 存在被硬门拒绝或指标不全的单阶段候选 → 不触发 | `:2268-2271` |
| 4 | 残差显著性门：留出天空残余跨度必须仍 > 3σ | `:603-639`（`_stage3_compound_residual_gate`） |
| 5 | 拟合集 / 留出集**冻结分离**，RBF 只见拟合点 | `:1877-1881`（安装 `compound_fit_points`） |
| 6 | 复合留出验证：相对最佳单阶段改善 ≥ 10%，且零点漂移 ≤ 0.01 绝对 / 15% 相对 | `:2697-2725` |
| 7 | 复合分数门：绝对改善 ≥ 0.03 **且** 相对改善 ≥ 10% | `:2726-2743`、`:642-690` |
| 8 | 目标保真门 + 方向噪声增长门 + color shift 门同样适用 | `:2560-2627` |

**判定：不构成过拟合风险，反而偏保守。** 理由：
- 拟合/留出集在候选评估**之前**就确定性冻结（`:1552`），复合模型无法「看到」验证点，这是标准的 hold-out 防过拟合协议，**在天文后期软件中极为罕见**；
- 一阶多项式（degree 1）是 Siril 文档明确推荐的「最稳定」阶数（「A degree 1 correction can be very useful…」，同上 URL），残差再交给带正则化的 RBF；Siril 的 RBF 本身是**薄板样条 + λI 正则化**（同上 URL 的 Theory 段：`+ λI` 平滑项），即 RBF 阶段自带岭回归式收缩；
- 8 道门中任意一道不过即回滚到不可变基线（`:2385-2389`、`:2849-2852` 的 `finally` 兜底）。

**残余隐患**：第 6/7 道门的「改善 ≥ 10%」是**分数比例**而非统计显著性，与第 4 道门的 3σ 基准不同源（详见 2.11 与 TOP1）。

### 2.7 目标通量 / 形态保真校验

`assess_target_fidelity`（`background_sampling.py:1445-1523`）+ `measure_stage3_signal_preservation`（`pipeline/image_metrics.py:276-621`）：

- **强制要求**参考系为 `heldout_sky_plane_degree_1`（`:1461-1462`）——目标通量必须相对「留出样点拟合的一阶天空平面」度量，而不是相对全图中值。**这是关键的正确设计**：背景减除本身会整体抬降基线，若不用独立参考面，通量保留比会被基线漂移污染。
- **拒绝条件**：
  - 通量变化显著性 < −3σ（`:1469-1473`）→ 目标被吃掉；
  - 形态相关性非有限（`:1480-1481`）；
  - 低复杂度场景下，目标区新增结构显著性 > 3σ（`:1491-1499`）→ 模型往目标区**注入**了不该有的结构。
- **强制生效**：`fidelity_enforced = isinstance(input_profile, dict)`（`:1982`），而 `input_profile` 属性恒存在（`seestar_Superimpose.py:1107`）且必为 dict → **该门在生产中恒定启用**。

**专业评价**：以 3σ 显著性替代「通量保留率 > 95%」这类固定百分比，避免了「高 SNR 图像门槛过松、低 SNR 图像门槛过严」的经典缺陷。**该设计优于 PixInsight / GraXpert 的现状（二者均无自动目标保真回滚，只能靠人眼比对）。**

### 2.8 方向性噪声独立路由

`analyze_directional_pattern_noise`（`background_sampling.py:245-411`）→ `select_background_route`（`:1642-1730`）：

- 三条路由：`mixed_gradient_and_pattern_noise`（有梯度也有条纹 → 允许提取但强制 review）、`pattern_noise_deferred`（只有条纹 → **preserve 像素，禁止 `-existing` 建模**，`:1726`）、`low_frequency_gradient`。
- 按 `pattern_kind` 分派责任方：`diagonal_walking_noise` → 采集/叠加复核（建议重新叠加 + 抖动，`:1689-1696`）；`horizontal/vertical_banding` → 标定/传感器复核（`:1697-1704`）。
- 候选级门禁 `pattern_candidate_gate`（`:1733-1767`）：候选**引入**或**放大**方向噪声（增长 > `growth_max` 且绝对值超阈）即拒绝。
- 路由结果可反向覆写决策为 `review_required`（`stage3_background_extraction.py:1705-1717`），并保留 `pre_route_decision` 供追溯。

**专业评价**：**行业最佳实践级。** walking noise / banding 是**空间相关的结构噪声**，其空间频率与低频天空梯度部分重叠 —— 用背景模型去"吸收"它是天文后期的经典误用（会在星云区留下方向性伪结构）。本实现明确拒绝并把责任推回采集/标定环节，同时给出可执行建议。PixInsight、Siril、GraXpert **均未提供此类自动分流**。

### 2.9 不可变基线事务与回滚正确性

- 基线冻结：`save stage3_bg_input`（`:1369`），失败则 `baseline_saved=False`，后续 `evaluate_attempts` 直接拒绝执行任何破坏性候选（`:1858-1862`）—— **fail closed 正确**。
- 每候选前 `restore_baseline(f"before:{label}")`（`:1864`），失败即 break（`:1865-1869`）。
- 候选失败 / 被拒 / 已评估三种路径均回滚（`:1962`、`:2136`、`:2208`）。
- 复合候选用 `try/finally` 保证即使异常也回滚（`:2849-2852`）。
- **回滚失败即作废已通过的候选**：`:2784-2790` —— 即使复合候选已通过全部验证并保存，只要基线无法恢复就判定 `rollback_failed` 并拒绝。**这是很强的一致性保证**。
- 全量回滚事件落盘 `rollback_events`（`:1367`、`:1380-1400`、归档于 `:1748` / `:3171`）。

**判定：事务语义正确、无回滚遗漏路径。**

### 2.10 降级路径

| 场景 | 行为 | 位置 |
| --- | --- | --- |
| 基线保存失败 | 跳过所有破坏性候选，走 preserve/review | `:1371-1375`、`:1858-1862` |
| 安全样点安装失败 | 记录 `safe_sample_install_failed`，跳过该候选，**绝不让 subsky 自动重采样** | `:1888-1907` |
| GraXpert 运行时错误 | 记录 `graxpert_runtime_error`，切下一候选，归档 `fallback_triggered_by_graxpert_error` | `:1929-1934`、`:3135-3137` |
| 全部候选被拒 | 回滚基线 + `_background_review_required=True` + `degraded` | `:3010-3013`、`:3305-3314` |
| 复合候选通过安全门但不够干净 | 接受但强制 review（`compound_selected_degraded`） | `:2982-2986`、`:3113-3122` |

### 2.11 D1 风险清单

| 编号 | 风险 | 严重度 | 位置 |
| --- | --- | --- | --- |
| R1 | **双标准共存**：硬门用 3σ 统计证据，`sufficient` / 候选排序仍用旧经验分数（`_stage3_background_score` + policy 固定阈值） | 高 | `:44-63`、`:199-247`、`policy_selector.py:29-35` |
| R2 | 最终选优后的 `max_bg_std_growth` / dirty>0.35 检查**只告警不回滚** | 中 | `:3027-3050` |
| R3 | `input_profile` 为空 dict 时 Stage3 恒 review_required（fail-closed 但静默失能） | 中 | `:1609-1618`、`background_sampling.py:1284-1287` |
| R4 | 复合候选改善判据（10%/0.03）为比例阈值，与 3σ 体系不同源 | 中 | `:2703-2743` |
| R5 | 未导出合成背景模型，事后无法复核模型形状是否吃掉了星云 | 中 | 全文件无 background-model 导出 |
| R6 | 仅支持 subtraction；判定为乘性渐晕时直接 review，不提供 division 路径 | 低 | `background_sampling.py:1319-1321` |

---

## 三、D2 门禁来源与归档

### 3.1 门禁全量清单

来源分类：**[C]** = PipelineConfig 字段（可配置 + CLAMP_RULES 钳制）；**[P]** = policy 文件/`DEFAULT_POLICY`；**[H]** = 代码硬编码字面量（不可配置、不钳制）。

| # | 门禁 | 位置 | 判据 | 阈值来源 | 钳制 | 硬门? | 归档字段 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 条件决策总开关 | `:1068-1075` | `stage3_conditional_decision_enabled` | [C] | bool | 是（关闭即无条件 apply） | `decision.source=compatibility_override` |
| G02 | 线性输入确认 | `background_sampling.py:1330-1331` | `state=="linear"` | 上游 `input_profile.json` | — | 是 | `process_evidence.linear_input` |
| G03 | 真实天空支撑 | `background_sampling.py:1292` | `sky_grid_cells >= 8` | **[H] 8** | 无 | 是 | `process_evidence.true_sky_support` |
| G04 | 空间显著性 3σ | `background_sampling.py:1243-1244` | `span > 3·SE_median` | 统计推导 | — | 是 | `process_evidence.spatial_models.spatial_significance_limit_3sigma` |
| G05 | 乘性/径向判别 | `background_sampling.py:1246-1251` | `ΔBIC>6` 且边缘落差<−3σ 且残差≤`max(span·0.40, 3σ)` | **[H] 6.0 / 0.40** | 无 | 是 | `process_evidence.spatial_models.radial_*` |
| G06 | 方向噪声检出 | `background_sampling.py:245-411` | `pattern_score >= 0.55` / walking `>= 0.50` | [C] `stage3_pattern_score_min` / `stage3_walking_noise_score_min` | 0.25–0.90 | 是 | `directional_pattern_noise` |
| G07 | 分流路由 | `background_sampling.py:1660-1688` | `spatial_variation_supported && mode=="subtract"` | 过程证据（生产分支） | — | 是 | `noise_route.gradient_evidence_basis` |
| G08 | 复合准入硬闸 | `:569-600` | 弥散/暗云气/低对比/需复核 → 排除 | [P] `protect_*` + target_type | — | 是 | `compound_fallback.target_guard` |
| G09 | 安全样点覆盖 | `background_sampling.py:698-704` | `≥min_count` ∧ `≥3象限` ∧ `≥8网格` ∧ `x/y_span≥0.55` | [C] `stage3_safe_sample_min_count` + **[H] 3 / 8 / 0.55** | 部分 | 是 | `safe_samples.coverage` |
| G10 | 样点安装审计 | `:768-807` | 无未知样点 ∧ 保留数 ≥ min | [C] `stage3_safe_sample_min_count`（8–64 钳制） | 是 | 是 | `attempts[].safe_samples` |
| G11 | 留出集拆分 | `background_sampling.py:858-1082` | 总数 ≥32、拟合 24、留出 8、双侧覆盖 | [C] `stage3_compound_*_count` | 是 | 是 | `safe_samples.fit_validation_split` |
| G12 | 目标保真门 | `background_sampling.py:1445-1523` | 通量 −3σ / 新增结构 +3σ | 统计推导 | — | 是（恒启用） | `attempts[].target_fidelity_gate` |
| G13 | 留出天空验证门 | `background_sampling.py:1386-1442` | 跨度改善 > 3σ ∧ RMS 不劣化 | 统计推导 | — | 是（恒启用） | `attempts[].validation_gate` |
| G14 | 方向噪声候选门 | `background_sampling.py:1733-1767` | 未引入 ∧ 增长 ≤ `growth_max` | [C] `stage3_pattern_score_growth_max` | 0.02–0.40 | 是 | `attempts[].pattern_quality_gate` |
| G15 | 色偏门 | `:2088-2101`、`:2602-2607` | `color_shift <= 0.18` | [P] `sufficient_color_shift_max` | — | 是（计入 `hard_gate_metrics_available`） | `attempts[].color_shift` |
| G16 | 源保真诊断门 | `stage_support.py:1573-1619` | **恒返回 True**，仅记录诊断 | — | — | **否（已退役）** | `attempts[].quality_message` |
| G17 | 复合残差显著性 | `:603-639` | `robust_span > 3σ` | 统计推导 | — | 是 | `compound_fallback.residual_gate` |
| G18 | 复合留出验证 | `background_sampling.py:1526-1639` | 改善 ≥10%、零点漂移 ≤0.01/15% | [C] `stage3_compound_validation_improvement_min` 等 | 是 | 是 | `compound_fallback.validation_gate` |
| G19 | 复合分数门 | `:642-690` | 绝对 ≥0.03 ∧ 相对 ≥10% | [C] `stage3_compound_score_*_min` | 是 | 是 | `compound_fallback.score_gate` |
| G20 | 复合回滚一致性 | `:2782-2790` | 回滚失败即作废候选 | — | — | 是 | `attempts[].status=rollback_failed` |
| G21 | 候选充分性 `sufficient` | `:199-247` | 分数/dirty/gradient 保留率/std 增长/色偏 综合 | **[P] 7 个固定字面量** | 无钳制 | **软门**（决定停搜索、触发复合、排序首键） | `attempts[].sufficient` |
| G22 | 最终选优 | `:2951-2958` | `(not sufficient, score)` 字典序最小 | — | — | 是 | `evaluated_candidate_order` + `attempts[].background_score` |
| G23 | 收尾背景增长检查 | `:3027-3050` | `std_growth > max_bg_std_growth` 或 `dirty>0.35 ∧ gradient_after ≥ 0.92·before` | [P] `max_bg_std_growth` + **[H] 0.35 / 0.92** | 部分 | **否（仅告警）** | `reason_code=background_improvement_limited` |

### 3.2 阈值来源分类统计

| 来源 | 数量 | 代表 |
| --- | --- | --- |
| **[C]** PipelineConfig + CLAMP_RULES | 14 项 | `stage3_safe_sample_*`（`models.py:130-134`）、`stage3_compound_*`（`:135-143`）、`stage3_pattern_*`（`:144-147`） |
| **[P]** policy（`policy_selector.py:17-37` + `configs/policies/*.yaml`） | 12 项 | `sufficient_max_background_score=0.34`、`sufficient_dirty_score_max=0.32`、`sufficient_color_shift_max=0.18`、`max_bg_std_growth=1.05` |
| **统计推导**（无自由阈值，仅 3σ 常数） | 5 项 | G04 / G12 / G13 / G17 及 `assess_single_background_validation` |
| **[H]** 硬编码字面量 | **≥9 处** | `8`（G03）、`6.0`/`0.40`（G05）、`3`/`8`/`0.55`（G09）、`0.50`（`background_sampling.py:1342` low_complexity 判据）、`0.45`（pattern penalty 系数 `:2156`/`:2691`）、`0.35`/`0.92`（G23） |

**问题**：**[H] 类中的 `8`、`0.55`、`0.50` 直接参与 preserve / apply / low_complexity 的授权判定**（G03 决定 `sky_supported` → `target_signal_or_sky_limited` → 硬阻断 → review_required），却既不在 `PipelineConfig`、不在 `CLAMP_RULES`、也不在 policy 文件中，因此**不出现在任何归档快照里**。

### 3.3 【重点验证】旧 `bg_*_ratio` 与 `stage3_gradient/dirty_*` 是否仍影响决策

文档 `pipeline/seestar_Superimpose_workflow.md` §3.1（L114-133）称这批字段「只保留兼容和诊断，不再决定生产任务」。**逐条代码验证结论：陈述属实**，但存在两处"僵尸读取"需要指出。

**（1）`bg_std_worsen_ratio_max` / `bg_median_drop_ratio_min` / `bg_object_preserve_ratio_min` / `bg_edge_black_rise_max` / `bg_star_preserve_ratio_min` / `bg_nebula_mean_change_max`（`models.py:116-121`）**

- 全仓引用点仅三处：`models.py` 定义、`seestar_Superimpose.py:314-318` 的 `AUTO_CLAMP_FIELDS`、`seestar_Superimpose.py:459-463` 的 `CLAMP_RULES`。
- 唯一的历史消费者 `_stage3_quality_gate`（`stage_support.py:1573-1619`）已被改写为**恒返回 `True`**，其 docstring 明确写道：「The former fixed ratios mixed sky-pedestal changes with source loss and are **intentionally no longer used to authorize or reject a candidate here**」（`:1579-1585`）。
- **判定：完全无活跃决策引用。✅ 文档属实。**

**（2）`stage3_gradient_skip_max` / `stage3_dirty_skip_max`（`models.py:125-126`）**

- 唯一读取点：`_stage3_background_decision` 的兼容回退段 `:1142-1143`。
- 该段的**可达性**：进入 `:1135` 之前必须先跳过 `:1124` 的 `if hasattr(pipeline, "input_profile")`。而 `seestar_Superimpose.py:1107` 已声明 `self.input_profile: Dict[str, Any] = {}` 作为实例属性 → `hasattr` **恒为 True** → `:1124-1133` 必定 return → **`:1135-1331` 近 200 行旧阈值路径在生产中不可达（dead code）**。
- **判定：无活跃决策引用。✅ 文档属实。**

**（3）`stage3_gradient_apply_min` / `stage3_dirty_apply_min`（`models.py:127-128`）—— 有"僵尸读取"**

- 除上述死代码外，**仍在生产路径被读取并传参**：`stage3_background_extraction.py:1644-1657` 将二者传入 `select_background_route`。
- 但在 `select_background_route`（`background_sampling.py:1660-1673`）中，只有 `process_report` 为空/非 dict 时才使用它们（`:1669-1672`）；生产路径在 `:1619-1639` 无论如何都会构造非空 dict（即使 `before_image is None` 也返回 4 键 dict），因此**永远走 `:1661-1665` 的过程证据分支**。
- 该分支还显式注释：「Production Stage 3 always supplies it」（`:1667-1668`），并把分支标记落盘为 `gradient_evidence_basis="source_masked_heldout_spatial_process"`（`:1665`、`:1723`）。
- **判定：读取但不生效，属"僵尸参数传递"。文档陈述仍然属实，但代码可读性上具有误导性 —— 静态阅读者会误以为这两个旧阈值仍在决定分流。**

**（4）`stage3_diffuse_auto_apply_enabled`（`models.py:129`）**：仅在死代码段与 clamp 中出现，无活跃引用。

**（5）release 副本核对**：`/Users/mz/dev/aiseestar_Superimpose.app/Contents/Resources/pipeline/` 下的同名文件与源码一致，旧字段同样仅出现在 `models.py` 与 clamp 逻辑中，无决策路径引用。

**总结论**：文档 §3.1 的断言 **属实**。全部 11 个旧兼容字段中，**0 个**对生产决策产生实际影响；其中 2 个（`stage3_gradient_apply_min` / `stage3_dirty_apply_min`）仍被读取传参但在生产分支被忽略，1 段约 200 行的旧阈值决策路径（`:1135-1331`）为不可达死代码。

### 3.4 参数集中化与环境变量控制

- **集中化**：所有可调 Stage3 参数集中于 `pipeline/models.py:100-147` 的 `PipelineConfig`，符合 `AGENTS.md` 的参数集中要求。
- **钳制**：`AUTO_CLAMP_FIELDS`（`seestar_Superimpose.py:303-446`）+ `CLAMP_RULES`（`:448-499+`），由 `clamp_config`（`:755-793`）与 `auto_tune_config`（`:796-969`）应用。抽查 `bg_samples`(12,32)、`bg_tolerance`(0.6,1.8)、`bg_smooth`(0.2,1.2) 与 Siril 官方参数语义（`-samples` 为每行样点数、`-tolerance` 为 MAD 倍数、`-smooth` 为 RBF 正则强度）一致且区间保守。
- **双重钳制**：`_stage3_cfg_float` / `_stage3_cfg_int`（`stage3_background_extraction.py:540-566`）在读取时再次夹紧，即使配置绕过 `clamp_config` 也不会越界。**纵深防御，正面评价。**
- **环境变量旁路**：`grep os.environ|getenv` 在 `stage3_background_extraction.py`、`background_sampling.py`、`policy_selector.py` 中**零命中** → **无环境变量后门**。✅

### 3.5 归档落盘清单

| 文件 | 写入位置 | Stage3 相关内容 |
| --- | --- | --- |
| `background_quality_report.json` | `:1737-1762`（非 apply 分支）/ `:3123-3188`（apply 分支） | `decision`（含 `process_evidence` 全量）、`attempts[]`（每候选门禁快照）、`rollback_events`、`safe_samples`（含实际样点坐标 + 掩膜报告 + 拒绝计数）、`compound_fallback`、`noise_route`、`directional_pattern_noise`、`before`/`after`、`protected_masks`、`fallback_reason` |
| `processing-plan.json` | `processor_runtime.py:1985-1990`（变换前冻结，双写 work_dir + process_dir） | `config` = 钳制后的完整 `PipelineConfig`（`:1979`）+ `plan_hash` |
| `pipeline_policy.json` | `seestar_Superimpose.py:1941`/`:1957` | 完整生效 policy（含 `stage3_background` 全部 `sufficient_*` 阈值） |
| `input_profile.json` | `processor_runtime.py:1541` | 线性域判据的上游依据 |
| `pipeline-result.json` | 经 `_record_stage`（`:1769-1848`、`:3290-3314`） | `status` / `execution` / `reason_code` / `components`（三个子组件的 status+method+reason_code+fallback_used） |
| review bundle | `:3195-3207` | 候选对比包（`candidates=attempt_records`） |

写入实现：`_write_stage_json`（`processor_runtime.py:2177-2178`）→ `write_stage_json(self.process_dir, ...)`。

### 3.6 可复现性判定：能否仅凭归档复现「为什么选 preserve 而非 apply」？

**结论：能，且可定量复算。**

`preserve` 与 `apply` 的分界完全由 `mechanism` 决定，而 `mechanism` 的全部输入都已落盘：

| 需要的量 | 归档路径 | 可否复算 |
| --- | --- | --- |
| 是否线性 | `decision.process_evidence.linear_input.confirmed` | ✅ |
| 天空支撑 | `...true_sky_support.usable_sky_grid_cells` / `.usable_sky_fraction` | ✅（与硬编码 8 比较） |
| 留出跨度 | `...spatial_models.robust_span` | ✅ |
| 3σ 门限 | `...spatial_models.spatial_significance_limit_3sigma` | ✅ |
| patch MAD / 半径 | `...spatial_models.patch_mad_median`、`safe_samples.patch_radius` | ✅（可独立复算 `1.2533·MAD/√N`） |
| 平面/径向 BIC | `...spatial_models.plane_model.bic` / `.radial_model.bic` | ✅ |
| 方向噪声 | `directional_pattern_noise` + `noise_route` | ✅ |
| 分支标识 | `noise_route.gradient_evidence_basis` | ✅（可确认走的是过程证据分支而非旧阈值分支） |
| 决策原因文本 | `decision.reason` + `decision.source` | ✅ |
| 实际样点坐标 | `safe_samples.points` | ✅ |

即：读到 `mechanism="no_measurable_low_frequency_gradient"` 时，可直接用 `robust_span` 与 `spatial_significance_limit_3sigma` 两个数验证「跨度未超采样不确定度」，**无需重跑代码**。这在同类项目中属于**优秀**水平。

### 3.7 归档缺口

| 编号 | 缺口 | 影响 |
| --- | --- | --- |
| A1 | **硬编码判据不入档**：G03 的 `8`、G09 的 `3/8/0.55`、G05 的 `6.0/0.40`、`low_complexity` 的 `0.50`、G23 的 `0.35/0.92`、pattern penalty 的 `0.45` 均无快照 | 换版本后无法确认当时用的是哪组常数 |
| A2 | **无代码版本指纹**：`processing-plan.json` 有 `plan_hash`（`processor_runtime.py:1981`）但仅覆盖 plan 内容；全仓 grep 无 git commit / 代码版本字段 | A1 无法通过版本回溯弥补 |
| A3 | **未导出合成背景模型**：Siril `subsky` 不返回模型，本实现也未做 before−after 差分导出 | 无法事后目视复核「模型是否吃掉了星云」，而这正是 PixInsight/GraXpert 的标配审查手段 |
| A4 | **`_stage3_background_score` 权重不入档**：`:44-63` 的分项权重为硬编码，归档只有最终 `background_score` 与 `base_background_score` | 分数不可分解，排序理由不可复现 |
| A5 | 非 apply 分支的归档缺 `compound_fallback` / `builtin_candidate_order` 等键（`:1740-1761` 与 `:3126-3187` 字段集不对称） | 下游消费者需做键存在性判断；分析脚本易漏 |

---

## 四、D3 算法行业标准符合度

### 4.1 本阶段使用的算法

1. **样点法背景建模**：自建 source/coverage 掩膜 → 空间分层选点 → `subsky -existing` 交给 Siril 拟合。
2. **模型族**：Siril Polynomial（degree 1–4）与 Siril RBF（薄板样条 + λ 正则）。
3. **复合模型**：Polynomial(deg 1) → 残差 RBF 串联。
4. **校正模式**：仅 subtraction。
5. **外部回退**：GraXpert（AI）/ ADBE / DBE / AutoDBE / NOX（`stage_support.py:1654-1662`）。

### 4.2 行业基线（含 URL）

| 工具 / 来源 | 关键事实 | URL |
| --- | --- | --- |
| **Siril Background Extraction（官方文档）** | 提供 Polynomial（最大 4 阶，「Beyond this, the model is generally unstable」）与 RBF（薄板样条 `φ(r)=r²log r`，加 `λI` 正则，Smoothing 即 λ）；tolerance 定义为 `median + tolerance × MAD`；subtraction 用于加性（光污染/月光），division 用于乘性（渐晕/大气消光，但「应当用 master-flat 修正」）；`-existing` 强制使用外部设定样点，否则 subsky 会自动重新生成 | <https://siril.readthedocs.io/en/stable/processing/background.html> |
| **Siril 梯度去除教程** | RBF 「allows us to process more complex gradients using fewer samples」，推荐用于单张图像；smoothing 过小会在样点间产生 over/undershoot，过大则无法适配大梯度差 | <https://siril.org/tutorials/gradient> |
| **Siril 文档源（含新版 Automatic 无样点法、样点优化、SetiAstro AutoDBE 式随机布点）** | 新版增加「Automatic (sample-free)」方法：对每个像素做迭代稳健剔除后直接拟合；样点可做「local minimum median」优化；可限制样点区域避开边缘无覆盖区 | <https://gitlab.com/free-astro/siril-doc/blob/main/doc/processing/background.rst> |
| **PixInsight MultiscaleGradientCorrection 官方教程（Vicent Peris）** | 明确指出 DBE 结果中「subtle local variations… we cannot distinguish between gradient residuals and actual nebular features」，主张用观测参考（第二台望远镜/MARS 巡天）而非纯软件解 | <https://pixinsight.com/tutorials/multiscale-gradient-correction> |
| **PixInsight GradientCorrection 参数解析** | 含 Scale / Smoothness / Low-High Threshold+Tolerance / **Structure Protection**（Protection Threshold + Protection Amount + 保护掩膜可视化）/ **Simplified Model**（Model Degree 1–8）/ Automatic Convergence；原文：「**Simplified model handles the large-scale gradient while the other settings handles the small-scale gradients**」 | <https://chaoticnebula.com/how-to-use-pixinsight-gradient-correction> |
| **PixInsight MGC 实战指南** | MGC 依赖 ImageSolver → SPFC → MGC → SPCC 的元数据链；Gradient scale 越小越激进、易过校正；建议 gradient correction 在 color calibration **之前** | <https://stirlingastrophoto.com/posts/multiscale-gradient-correction> |
| **MGC 吃掉 M42 核心的真实案例** | MGC 把 M42 强 Hα 发射误判为梯度，模型中出现「bright red blob centered exactly where M42's core should be」 | <https://spacecityastronomy.com/2026/01/27/taming-the-great-nebula-mgc-settings-for-m42> |
| **GraXpert 官方仓库** | 提供 RBF / Splines / Kriging（需手工选点）与 **AI 方法（无需任何样点）**；CLI：`-correction Subtraction\|Division`、`-smoothing 0.0–1.0`、`-bg` 导出背景模型 | <https://github.com/Steffenhir/GraXpert> |
| **GraXpert PixInsight 脚本说明** | 「default smoothing factor of 0.0 should be your best choice. **The smoothing factor does not influence the result produced by the AI model**; the resulting background model can be further smoothed using gaussian blur」 | <https://www.ideviceapps.de/graxpert-script.html> |
| **GraXpert 工作流约束** | 「GraXpert must be used **after cropping** and **before any other image processing**. GraXpert will fail on non-cropped images (leftover black frames) and on already post-processed images」 | <https://astroguide.starlust.de/html/GraXpert1.html> |
| **photutils Background2D（科研级基线）** | 网格分块 + sigma-clip（默认 3σ/10 iter）+ SExtractorBackground(`2.5·median − 1.5·mean`) + 中值滤波 + 三次样条插值；`mask`（源/坏像元）与 `coverage_mask`（无覆盖区）分离；`exclude_percentile` 控制块级剔除 | <https://photutils.readthedocs.io/en/latest/user_guide/background.html> |
| **SExtractor 原始文献（Bertin & Arnouts 1996, A&AS 117, 393）** | 天空背景 mesh 估计 + 3σ 迭代剔除 + 双三次样条插值的行业原点 | <https://doi.org/10.1051/aas:1996164> |
| **Wright 2003（Siril RBF 引用文献）** | RBF 插值的数值与解析发展，Siril 文档引用其证明薄板样条矩阵可逆性 | 由 Siril 文档引用：<https://siril.readthedocs.io/en/stable/processing/background.html> |
| **MSGR（多尺度梯度去除社区方法）** | 用宽场数据替换窄场大尺度分量，把复杂梯度降为低阶简单梯度后再用 ABE/DBE 处理；强调「low function degree, high smoothing, few points far apart」 | <https://nightphotons.com/guides/multiscale-gradient-removal/> |

### 4.3 「Polynomial → 残差 RBF」串联是否有先例？

**有明确先例，且属于当前主流做法之一。**

1. **PixInsight `GradientCorrection` 的 Simplified Model**：官方参数结构中，Simplified Model 是一个 degree 可调（实测 1–8）的低阶多项式**预处理**，专门吃掉光污染/渐晕/flat 误差造成的强大尺度梯度，剩余小尺度梯度交由主模型（Scale/Smoothness 控制的自适应模型）处理。原文：「Simplified model handles the large-scale gradient while the other settings handles the small-scale gradients」（<https://chaoticnebula.com/how-to-use-pixinsight-gradient-correction>）。本实现的 `subsky 1 -existing` → 残差 RBF **在结构上与之同构**。
2. **社区 MSGR 方法**：显式主张「先把复杂梯度降为低阶简单梯度，再用低阶模型收尾」（<https://nightphotons.com/guides/multiscale-gradient-removal/>），与本实现的分层动机一致（只是本实现用一阶多项式而非宽场参考图来完成第一层）。
3. **Siril 官方建议**：「a single image… generally follows a simple linear (degree 1) function」「Good results with the RBF algorithm generally require fewer samples than with the polynomial」（<https://siril.readthedocs.io/en/stable/processing/background.html>），即一阶多项式做主体、RBF 做柔性补充，是 Siril 文档默许的组合思路。
4. **科研侧类比**：photutils/SExtractor 的「粗网格稳健统计 → 中值滤波 → 样条插值」本质也是「先低阶稳健、再局部柔性」的两级结构（<https://photutils.readthedocs.io/en/latest/user_guide/background.html>）。

**差异点（本实现更严）**：上述所有先例中，两级模型的组合都是**人工试错决定**的；本实现是**在冻结留出集上做统计检验后自动决定**（G17 残差显著性 + G18 留出改善 + G19 分数改善三重门），并且弥散/暗云气目标直接禁用复合（G08）。**这是超出行业现状的加固。**

### 4.4 与 GraXpert AI 的取舍分析

| 维度 | 本实现（Siril 样点法 + 复合） | GraXpert AI |
| --- | --- | --- |
| 是否需要样点 | 需要，且自建掩膜 + 覆盖硬性检查 | 不需要（AI 直接推断） |
| 可解释性 | 高（样点坐标、掩膜、BIC、3σ 全部落盘） | 低（神经网络黑箱） |
| 可复现性 | 高（阈值 + 样点全归档） | 依赖 `-ai_version`；模型版本演进会改变结果 |
| 离线性 | 完全离线 | 首次需下载 AI 模型 |
| 复杂梯度能力 | 受限于一阶多项式 + RBF 柔性 | 通常更强（对多源混合梯度） |
| 弥散星云保护 | 由 G08/G12 显式硬闸保护 | 依赖训练数据分布，无保证 |
| 项目定位 | **主路径** | **外部回退**（`stage_support.py:1655-1656`，排在 ADBE/DBE 之前） |

**评价：取舍合理。** 本项目定位为「离线优先 + 可审计」，把可解释的样点法作为主路径、AI 作为回退是正确的优先级。且实现对 GraXpert 运行时错误做了专门归类与降级（`:1929-1934`、`:3132-3137`），符合 GraXpert 已知的运行环境脆弱性（未裁剪黑边即失败，见 <https://astroguide.starlust.de/html/GraXpert1.html>）。

**一处可优化**：GraXpert CLI 支持 `-bg` 导出背景模型（<https://github.com/Steffenhir/GraXpert>），本实现未使用（`stage_support.py:1655-1656` 仅传 `("gxp",)` / `("graxpert",)`），错失了免费的模型可视化证据。

### 4.5 行业已知陷阱 vs 本实现

| # | 行业已知陷阱 | 出处 | 本实现是否规避 | 证据 |
| --- | --- | --- | --- | --- |
| T1 | 样点落在星云/星系上 → 把目标当背景减掉 | Siril 文档「necessary to avoid placing samples on stars and/or objects」<https://siril.org/tutorials/gradient> | ✅ 完全规避 | source_mask + 亮度/纹理/星点分位拒绝（`background_sampling.py:414-746`） |
| T2 | 多项式阶数过高 → 过校正/不稳定 | Siril「maximum degree is 4… Beyond this, the model is generally unstable」 | ✅ 规避 | 复合第一级固定 degree 1（`:2353`）；`low_complexity_required` 时限制为 `polynomial_degree_1`（`background_sampling.py:1369-1373`） |
| T3 | RBF smoothing 过小 → 样点间 over/undershoot | Siril 教程同上 | ⚠ 部分规避 | `bg_smooth` 钳制 0.2–1.2（`seestar_Superimpose.py` CLAMP_RULES），但**未随证据强度自适应**；靠留出验证门事后拦截 |
| T4 | 把渐晕（乘性）当加性梯度减掉 | Siril「Division… mainly used to correct multiplicative phenomena」 | ✅ 规避且更严 | BIC 判别后走 `master_flat_review` 而非强行 division（`background_sampling.py:1319-1321`） |
| T5 | 背景模型吃掉大尺度星云 / 银河 cirrus | PixInsight「we cannot distinguish between gradient residuals and actual nebular features」<https://pixinsight.com/tutorials/multiscale-gradient-correction> | ✅ 规避 | G08 复合硬闸 + G12 目标保真 3σ 门 + `nebula_preservation_penalty_weight`（`:110-146`） |
| T6 | 强梯度目标核心被误判为梯度（MGC 吃 M42 核心） | <https://spacecityastronomy.com/2026/01/27/taming-the-great-nebula-mgc-settings-for-m42> | ✅ 规避 | `protect_bright_core` policy + 目标保真门的「新增结构 > 3σ」检测（`background_sampling.py:1491-1499`） |
| T7 | 16-bit + 弱梯度 → posterization / color banding | Siril「add dither option… strongly advise 32-bit」 | ⚠ **未验证** | `_stage3_subsky_rbf_candidates`（`stage_support.py:1683-1774`）未见 `-dither` 参数；未确认全链是否强制 32-bit float |
| T8 | 未裁剪黑边 → 边缘样点污染模型 | Siril「limit the area in which samples are placed… eliminating the problem with edge samples」；GraXpert 「will fail on non-cropped images」 | ✅ 规避 | coverage_mask + `margin_pixels`（`background_sampling.py:719`）+ patch 越界跳过（`:1183-1189`） |
| T9 | 条纹/walking noise 被背景模型吸收 | 天文后期通识；Siril/PixInsight 均无自动分流 | ✅ **超出行业现状** | 独立路由 + `pattern_candidate_gate`（`background_sampling.py:1642-1767`） |
| T10 | 背景校正与色彩定标顺序颠倒 | MGC 指南「recommended workflow is to apply color calibration **after** gradient correction」 | ✅ 规避 | Stage3（背景）在 Stage4（色彩定标）之前 |
| T11 | 无法回溯模型形状 | Siril/GraXpert/DBE 均支持导出背景模型 | ❌ **未规避** | 全文件未见背景模型导出（见 A3） |

### 4.6 偏离分析

| 偏离项 | 方向 | 评价 |
| --- | --- | --- |
| 不提供 division 校正 | 比行业**保守** | 合理：把乘性问题推回 flat 复核，避免掩盖标定缺陷 |
| 一阶多项式为复合第一级（行业可到 4–8 阶） | 比行业**保守** | 合理：Seestar 单站短焦数据梯度通常较简单；且高阶靠 RBF 补偿 |
| 强制 hold-out 统计验证 | 比行业**严格** | 行业无先例，正面 |
| 目标保真自动回滚 | 比行业**严格** | 行业依赖人眼，正面 |
| 方向噪声独立分流 | 比行业**严格** | 行业无先例，正面 |
| 无背景模型导出 | 比行业**落后** | 需补 |
| RBF smoothing 不自适应 | 与行业**持平**（行业也靠手调） | 但本项目是全自动，缺少手调环节，应考虑证据驱动的自适应 |
| 排序仍用经验分数 | 比行业**持平偏弱** | 行业靠人眼选优；本项目自动化后，经验分数成为唯一裁判，风险更高 |

### 4.7 未验证事项

- **未验证**：Siril 1.4.0 的 `subsky -rbf` 是否严格实现文档所述的薄板样条 + λI 正则 —— 未阅读 Siril C 源码，仅依据官方文档。
- **未验证**：`-dither` 是否在本 pipeline 的位深下必要 —— 未确认全链数据类型（32-bit float vs 16-bit）。
- **未验证**：PixInsight `GradientCorrection` 的官方参考手册页 —— 尝试访问 `pixinsight.com/doc/tools/AutomaticBackgroundExtractor/...` 返回 HTTP 404，D3 中 PixInsight 参数结构的结论来自第三方详解文章（chaoticnebula / stirlingastrophoto）与 PixInsight 官方教程页（pixinsight.com/tutorials/multiscale-gradient-correction），**非官方 reference 页**。
- **未验证**：`analyze_directional_pattern_noise` 的 `pattern_score` 计算是否对不同像元尺度/采样率鲁棒 —— 未做数值实验。
- **未验证**：resume 路径下 `input_profile` 是否在所有分支都先于 Stage3 完成重建（见 R3）。

---

## 五、关键发现 TOP 3

### 🔴 F1（最严重）—— 硬门用 3σ 统计证据，「够不够干净」却仍用旧经验分数，两套标准不同源

- **现象**：安全性判据（G04/G12/G13/G17）已全面统计化，但 `_stage3_candidate_sufficient`（`stage3_background_extraction.py:199-247`）仍用 7 个来自 `policy_selector.py:29-35` 的固定字面量（`0.34 / 0.32 / 0.88 / 0.04 / 0.06 / 0.96 / 0.18`）判定 `sufficient`；`_stage3_background_score`（`:44-63`）的权重亦为硬编码。
- **影响链**：`sufficient` 决定了 ①内置候选是否提前停止搜索（`:2210-2216`）②是否触发复合候选（`:2920-2935`）③最终选优的**第一排序键**（`:2951-2958`）④复合候选是否被标记为 degraded 强制 review（`:2982-2986`）。也就是说，**最终交付哪一张图，实质由旧经验分数拍板**，而非由统计证据拍板。
- **后果**：可能出现「A 候选统计上改善更显著、目标保真更好，但 B 候选经验分数更低而胜出」；反之在低 SNR 数据上，所有候选都达不到 `0.34` 的分数门槛 → 全部 `sufficient=False` → 无谓触发复合候选与外部插件链，增加处理时间与回滚风险。
- **证据**：`:44-63`、`:199-247`、`:2162-2167`、`:2210-2216`、`:2920-2935`、`:2951-2958`、`policy_selector.py:29-35`。

### 🟠 F2 —— 决定 preserve/apply 的部分判据是不入档的硬编码字面量，且归档无代码版本指纹

- **现象**：`sky_grid_cells >= 8`（`background_sampling.py:1292`）直接决定 `sky_supported` → 决定是否产生硬阻断 `insufficient_source_masked_true_sky_support` → 决定 `review_required`；同类还有样点覆盖的 `3 象限 / 8 网格 / 0.55 跨度`（`:698-704`）、`usable_sky_fraction < 0.50`（`:1342`）、BIC 的 `6.0` 与 `0.40`（`:1249-1250`）、pattern penalty 系数 `0.45`（`stage3_background_extraction.py:2156`、`:2691`）、收尾检查的 `0.35 / 0.92`（`:3042`）。
- **问题**：这些既不在 `PipelineConfig`（`models.py:100-147`）也不在 policy（`policy_selector.py`）中，因此**不出现在 `processing-plan.json` 的 config 快照或 `pipeline_policy.json` 里**；同时全仓无 git commit / 代码版本字段落盘（`run_manifest.py` 仅有 `plan_hash`）。
- **后果**：跨版本比对时，「同样的数据为什么这次 review 上次 apply」无法从归档中定位；与 `AGENTS.md` 的参数集中原则存在偏差。
- **证据**：`background_sampling.py:1292`、`:698-704`、`:1342`、`:1249-1250`；`stage3_background_extraction.py:2156`、`:2691`、`:3042`；`processor_runtime.py:1979-1981`。

### 🟡 F3 —— 旧字段断言属实，但存在「僵尸读取」与约 200 行不可达死代码，构成审计噪声

- **验证结论**：文档 §3.1 关于旧 `bg_*_ratio` 与 `stage3_gradient/dirty_*` 「只保留兼容和诊断」的陈述**属实**，全部 11 个旧字段对生产决策**零影响**。
- **但**：①`stage3_gradient_apply_min` / `stage3_dirty_apply_min` 仍在生产路径被读取并传参（`stage3_background_extraction.py:1644-1657`），只是在 `select_background_route` 内被过程证据分支旁路（`background_sampling.py:1660-1673`）；②`_stage3_background_decision` 的 `:1135-1331`（约 197 行）因 `hasattr(pipeline, "input_profile")` 恒真（`seestar_Superimpose.py:1107`）而**永不可达**。
- **后果**：静态阅读者（含未来的维护者与审计者）极易误判这些阈值仍在生效；死代码也会在重构时被误当作有效逻辑维护。
- **证据**：`stage3_background_extraction.py:1124-1133`、`:1135-1331`、`:1644-1657`；`background_sampling.py:1660-1673`；`stage_support.py:1579-1585`；`seestar_Superimpose.py:1107`、`:314-318`、`:459-463`。

---

## 六、改进建议

> 以下为审计建议，**本次审计未修改任何代码**。

### P0（建议优先处理）

| ID | 建议 | 位置 |
| --- | --- | --- |
| P0-1 | **统一「够好」的判据基准**：为 `_stage3_candidate_sufficient` 增加统计学出口 —— 当候选的留出天空跨度改善显著性（`validation_gate.span_improvement / sampling_uncertainty_3sigma`）超过设定 σ 数时，直接判 `sufficient`，不再依赖 `sufficient_max_background_score` 等经验字面量；经验分数降级为并列条件之一 | `stage3_background_extraction.py:199-247`；配合 `:2162-2167` |
| P0-2 | **最终选优改用统计量作为第一排序键**：把 `:2951-2958` 的 `(not sufficient, score)` 改为 `(not sufficient, -span_improvement_sigma, score)`，使统计证据优先于经验分数 | `stage3_background_extraction.py:2951-2958` |
| P0-3 | **硬编码授权判据入档**：将 G03 的 `8`、G09 的 `3/8/0.55`、`low_complexity` 的 `0.50` 上提为 `PipelineConfig` 字段并加入 `AUTO_CLAMP_FIELDS`/`CLAMP_RULES`；至少作为 `thresholds` 子对象写入 `background_quality_report.json` | `background_sampling.py:1292`、`:698-704`、`:1342`；`models.py:130-134`；`seestar_Superimpose.py:303-499` |
| P0-4 | **归档加入代码版本指纹**：在 `processing-plan.json` 增加 pipeline 代码版本/commit 字段，使硬编码常数可通过版本回溯 | `processor_runtime.py:1968-1981`；`run_manifest.py` |

### P1

| ID | 建议 | 位置 |
| --- | --- | --- |
| P1-1 | **导出合成背景模型**：在候选接受后计算 `baseline − candidate` 差分图并保存为 review bundle 的一部分（外部 GraXpert 分支可直接加 `-bg` 参数），补齐行业标配的模型可视化审查手段 | `stage3_background_extraction.py:3195-3207`；`stage_support.py:1655-1656` |
| P1-2 | **清理僵尸读取与死代码**：删除 `:1644-1657` 对 `stage3_gradient_apply_min`/`stage3_dirty_apply_min` 的传参（或在 `select_background_route` 签名中移除），并移除 `:1135-1331` 不可达分支；同时在 `models.py:125-129` 的注释中标注「仅供归档兼容读取，无任何调用点」 | `stage3_background_extraction.py:1135-1331`、`:1644-1657`；`background_sampling.py:1660-1673`；`models.py:125-129` |
| P1-3 | **收尾检查改为可回滚**：`:3027-3050` 的 `std_growth`/`dirty` 检查目前仅告警；建议在明显劣化时回滚到基线并转 `review_required`，或至少把 `0.35`/`0.92` 提升为 policy 字段 | `stage3_background_extraction.py:3027-3050`；`policy_selector.py:21` |
| P1-4 | **`input_profile` 缺失时给出显式诊断**：当 `input_profile` 为空 dict 时，在 `decision.reason` 中区分「上游未产出 profile」与「profile 判定为非线性」，避免静默失能被误读为「本图无梯度」 | `stage3_background_extraction.py:1609-1618`；`background_sampling.py:1284-1287`、`:1349-1355` |
| P1-5 | **统一两个归档分支的键集合**：让非 apply 分支（`:1740-1761`）与 apply 分支（`:3126-3187`）输出相同的键（缺失值填 `null`），便于下游分析脚本统一解析 | `stage3_background_extraction.py:1737-1762`、`:3123-3188` |

### P2

| ID | 建议 | 位置 |
| --- | --- | --- |
| P2-1 | **RBF smoothing 证据驱动自适应**：依据 `spatial_models.robust_span / significance_limit` 的比值调节 `bg_smooth`（梯度强 → 高平滑，弱 → 低平滑），对应 Siril 教程「high smoothing for large uniform gradients, lower for small local gradations」 | `stage_support.py:1683-1774`；参考 <https://siril.org/tutorials/gradient> |
| P2-2 | **补齐 `-dither` 决策**：确认全链位深，若存在 16-bit 路径则在弱梯度场景启用 `subsky ... -dither`，规避 posterization | `stage_support.py:1683-1774` |
| P2-3 | **归档 `_stage3_background_score` 的分项**：把 dirty/gradient/chroma/std_growth/color_shift 五个分量及权重一并写入 `attempts[]`，使排序理由可分解复现 | `stage3_background_extraction.py:44-63`、`:2168-2178` |
| P2-4 | **考虑 Siril 1.4 的 sample-free「Automatic」背景法作为额外候选**：官方新增的迭代稳健剔除法不依赖样点，可作为「安全样点覆盖不足」时的备选证据源（而非直接执行） | 参考 <https://gitlab.com/free-astro/siril-doc/blob/main/doc/processing/background.rst> |
| P2-5 | **`assess_background_process` 的 `evidence_basis` 增加数值摘要**：目前仅为文字列表（`background_sampling.py:1376-1382`），可附带关键数值便于机读 | `background_sampling.py:1376-1382` |

---

## 七、证据索引

### 7.1 代码证据（file:line）

**决策核心**
- `pipeline/stages/stage3_background_extraction.py:1059-1122` —— 三态决策生产路径
- `pipeline/stages/stage3_background_extraction.py:1124-1133` —— `input_profile` 存在但无 process_report 时的 fail-closed
- `pipeline/stages/stage3_background_extraction.py:1135-1331` —— 旧阈值决策路径（**不可达死代码**）
- `pipeline/background_sampling.py:1271-1383` —— `assess_background_process` 机制判定
- `pipeline/background_sampling.py:1313-1327` —— 五分支 mechanism 互斥判定
- `pipeline/background_sampling.py:1237-1244` —— 3σ 空间显著性构造
- `pipeline/background_sampling.py:1246-1251` —— 加性 vs 乘性 BIC 判别

**样点与掩膜**
- `pipeline/background_sampling.py:101-210` —— source/coverage 掩膜构建
- `pipeline/background_sampling.py:414-746` —— 安全样点选取
- `pipeline/background_sampling.py:698-704` —— 覆盖硬性要求
- `pipeline/background_sampling.py:858-1082` —— 确定性拟合/留出拆分
- `pipeline/stages/stage3_background_extraction.py:712-878` —— 样点安装审计（fail closed）

**门禁**
- `pipeline/background_sampling.py:1386-1442` —— 留出天空验证门
- `pipeline/background_sampling.py:1445-1523` —— 目标保真门
- `pipeline/background_sampling.py:1526-1639` —— 复合留出验证门
- `pipeline/background_sampling.py:1642-1730` —— 分流路由
- `pipeline/background_sampling.py:1733-1767` —— 方向噪声候选门
- `pipeline/stages/stage3_background_extraction.py:569-600` —— 复合准入硬闸
- `pipeline/stages/stage3_background_extraction.py:603-639` —— 复合残差显著性门
- `pipeline/stages/stage3_background_extraction.py:642-690` —— 复合分数门
- `pipeline/stage_support.py:1573-1619` —— 已退役的旧固定 ratio 门（恒 True）

**事务与回滚**
- `pipeline/stages/stage3_background_extraction.py:1365-1402` —— 基线冻结与回滚闭包
- `pipeline/stages/stage3_background_extraction.py:2385-2389`、`:2782-2790`、`:2849-2852` —— 复合候选事务

**评分与选优**
- `pipeline/stages/stage3_background_extraction.py:44-63` —— `_stage3_background_score`
- `pipeline/stages/stage3_background_extraction.py:199-247` —— `_stage3_candidate_sufficient`
- `pipeline/stages/stage3_background_extraction.py:2951-2958` —— 最终选优
- `pipeline/stages/stage3_background_extraction.py:3027-3050` —— 收尾告警（不回滚）

**配置与归档**
- `pipeline/models.py:100-147` —— PipelineConfig Stage3 字段
- `pipeline/models.py:116-129` —— 旧兼容字段
- `pipeline/seestar_Superimpose.py:303-446` —— `AUTO_CLAMP_FIELDS`
- `pipeline/seestar_Superimpose.py:448-499` —— `CLAMP_RULES`
- `pipeline/seestar_Superimpose.py:1107` —— `self.input_profile = {}` 初始化（决定死代码可达性）
- `pipeline/policy_selector.py:14-77` —— `DEFAULT_POLICY` 的 stage3 阈值
- `pipeline/stages/stage3_background_extraction.py:1737-1762` —— 非 apply 分支归档
- `pipeline/stages/stage3_background_extraction.py:3123-3188` —— apply 分支归档
- `pipeline/processor_runtime.py:1979-1998` —— `processing-plan.json` 冻结（含 config 快照）
- `pipeline/processor_runtime.py:2177-2178` —— `_write_stage_json`

**文档**
- `pipeline/seestar_Superimpose_workflow.md:75-110` —— §2.1 流程总览
- `pipeline/seestar_Superimpose_workflow.md:114-133` —— §3.1 旧字段兼容性声明（本次已验证属实）
- `pipeline/seestar_Superimpose_workflow.md:360-389` —— §5.3 阶段3执行顺序

### 7.2 行业证据（URL）

1. Siril 官方文档 · Background Extraction（RBF 薄板样条 + λI 正则、多项式最高 4 阶、tolerance = median + t·MAD、subtraction/division 语义、`-existing` 语义）—— <https://siril.readthedocs.io/en/stable/processing/background.html>
2. Siril 官方教程 · Removing gradients（RBF 少样点优势、smoothing 过大/过小的失效模式、避免在天体上布点）—— <https://siril.org/tutorials/gradient>
3. Siril 文档源仓库（新版 sample-free Automatic 法、样点优化、限制样点区域避开边缘）—— <https://gitlab.com/free-astro/siril-doc/blob/main/doc/processing/background.rst>
4. PixInsight 官方教程 · Multiscale Gradient Correction（DBE 残差可信性质疑、观测式梯度校正）—— <https://pixinsight.com/tutorials/multiscale-gradient-correction>
5. PixInsight GradientCorrection 参数详解（Simplified Model 处理大尺度、主模型处理小尺度；Structure Protection）—— <https://chaoticnebula.com/how-to-use-pixinsight-gradient-correction>
6. PixInsight MGC 实战指南（元数据依赖链、gradient correction 在 color calibration 之前）—— <https://stirlingastrophoto.com/posts/multiscale-gradient-correction>
7. MGC 误伤 M42 核心的实测案例 —— <https://spacecityastronomy.com/2026/01/27/taming-the-great-nebula-mgc-settings-for-m42>
8. GraXpert 官方仓库（RBF/Splines/Kriging/AI；`-correction`、`-smoothing`、`-bg`）—— <https://github.com/Steffenhir/GraXpert>
9. GraXpert PixInsight 脚本说明（AI 模式下 smoothing 不影响 AI 结果）—— <https://www.ideviceapps.de/graxpert-script.html>
10. GraXpert 工作流约束（必须先裁剪、必须在其他处理之前）—— <https://astroguide.starlust.de/html/GraXpert1.html>
11. photutils · Background Estimation（mask vs coverage_mask、sigma-clip、SExtractor 估计量、样条插值）—— <https://photutils.readthedocs.io/en/latest/user_guide/background.html>
12. photutils · Background2D API（`exclude_percentile`、`filter_size`、`BkgZoomInterpolator`）—— <https://photutils.readthedocs.io/en/1.13.0/api/photutils.background.Background2D.html>
13. Bertin & Arnouts 1996, A&AS 117, 393 · SExtractor（背景 mesh 估计的行业原点）—— <https://doi.org/10.1051/aas:1996164>
14. Multiscale Gradient Removal 社区方法（先降复杂度再低阶收尾）—— <https://nightphotons.com/guides/multiscale-gradient-removal/>

### 7.3 未验证声明汇总

1. **未验证：Siril 1.4.0 `subsky -rbf` 的实际数值实现** —— 仅依据官方文档描述（薄板样条 + λI），未阅读 Siril C 源码。
2. **未验证：PixInsight 官方 reference 手册页** —— `pixinsight.com/doc/tools/AutomaticBackgroundExtractor/AutomaticBackgroundExtractor.html` 返回 HTTP 404；D3 中 ABE/DBE/GradientCorrection 的参数结构结论来自 PixInsight 官方教程页与第三方详解文章，非官方 reference 页。
3. **未验证：本 pipeline 全链位深** —— 未确认是否始终 32-bit float，故 T7（posterization / `-dither` 必要性）无法定论。
4. **未验证：所有 resume 分支是否都在 Stage3 之前重建 `input_profile`** —— 仅确认 `seestar_Superimpose.py:2397`/`:2425` 存在重建调用，未逐分支验证。
5. **未验证：`analyze_directional_pattern_noise` 的 `pattern_score` 尺度鲁棒性** —— 纯静态审计，未做数值实验。
6. **未验证：`measure_stage3_signal_preservation`（`image_metrics.py:276-621`）内部各统计量的数值稳定性** —— 已确认其参考系与输出契约，未做边界数值验证。
7. **未验证：Stage3 在真实 Seestar 数据上的实际 mechanism 分布** —— 无运行数据，无法评估 preserve/apply/review 的实际比例是否合理。
