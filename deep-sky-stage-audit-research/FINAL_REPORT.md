# 深空天文后期 Pipeline 跨阶段汇总审计报告

- **项目**：`/Users/mz/dev/aiseestart`（Seestar 望远镜离线深空后期 pipeline，Python + Siril 1.4.0）
- **审计范围**：Stage 1–10，三维度（D1 逻辑专业合理性 / D2 门禁来源与归档 / D3 算法行业标准符合度）
- **输入**：10 份单阶段审计报告 `deep-sky-stage-audit-research/results/stage1.md` ～ `stage10.md`
- **审计性质**：纯研究汇总，**未读取或修改任何项目代码**（本文件为唯一新增产物）
- **汇总日期**：2026-08-05
- **方法论**：以 10 份报告正文为准综合；已知交叉校验分仅用于核对，冲突以报告正文为准；所有证据保留 `file:line`，行业结论保留 URL；无法验证处显式标注「未验证」，严禁编造。

> **命名错位背景（全仓固有）**：Stage 6「线性去星」实现在 `pipeline/stages/stage7_star_separation.py`，Stage 7「拉伸」实现在 `pipeline/stages/stage6_stretching.py`。本报告「Stage N」一律指逻辑阶段号，引用代码时给出真实文件名。各单报均已注明，不影响分析。

---

## 一、执行摘要

**整体健康度：7.7 / 10（加权总分）。** 这是一套架构取向专业、保守优先、可回退、可追的深空后期 pipeline。starless-first（线性去星 → 拉 starless → 受控回星）主线与 PixInsight / StarXTerminator / Siril 社区主流范式高度对齐，Stage 7 黑场不裁暗部与 Stage 9 Screen 回星的数学正确性达到「半专业级」上限。主要短板集中在**门禁来源可追溯性**（大量阈值硬编码且未标来源语义）与**末端降噪对星点的不可逆损伤**，以及贯穿多 stage 的**死配置 / 僵尸代码 / 可复现性断点**三类系统性技术债。

**最肯定的 3 件事：**
1. **starless-first 架构方向正确且工程化扎实**：四态状态机（accepted/rejected/tool_failed/target_bypass）在 Stage 6、Stage 8、Stage 9 形成完整「降级不崩溃」链路（Stage 6 `stage7_star_separation.py:436`、Stage 8 `stage8_nebula_enhancement.py:98`、Stage 9 `stage9_star_remixing.py:41`）。
2. **Stage 7 黑场处理是科学级实现**：从实测噪声地板反推 shadows 并留 3σ 余量，数学上不可能裁暗部（`stage6_services.py:2504-2529`），严于 PixInsight STF 默认的 -2.8σ 裁剪。
3. **Stage 9 回星数学正确**：Alpha+Screen 混合与逐像素标量增益保色（`stage9_quality.py:454-517`），数值示例验证与 RC Croman 社区公式完全等价，杜绝加法回星的星胀陷阱。

**最需立刻处理的 3 件事：**
1. **末端降噪无星点保护 mask（Stage 10，P0）**：`full`/`separate` 模式对整幅含星图无差别降噪、强度 0.5 硬编码（`:576,641`），而 starless-first 链路本已拥有星掩膜可零成本复用——属链路意图的自我抵消（D1/D3 最高专业风险）。
2. **可复现性断点（跨 Stage 5/6/7/8/10，P0）**：续跑指纹不绑定真实配置、模型权重身份未归档、AI 无 prompt 版本、Stage 8 断点无签名、两份顶层 JSON 不足以复现 Stage 10——科学级可复现性硬伤。
3. **降噪反向降级（Stage 5，P0）**：多尺度候选因细节保留不足被拒后落到 `denoise -mod=0.50 -indep` 更激进且无质量门无回滚——可能产出比候选更差的图。

---

## 二、评分总表（10 × 3 矩阵）

> 加权总分 = D1×0.35 + D2×0.35 + D3×0.30（沿用 Stage 9 报告口径）。Stage 4 单报未单列三维度，仅给总评 8.2，其三维格标注「见报告（仅总评 8.2）」。

| Stage | 阶段主题 | D1 | D2 | D3 | 加权总分 | 阶段总评出处 |
|---|---|---|---|---|---|---|
| 1 | 准备（debayer/register/stack） | 8.0 | 7.0 | 8.0 | **7.65** | stage1.md |
| 2 | 裁切 / 视场修正 | 7.5 | 6.0 | 7.0 | **6.83** | stage2.md |
| 3 | 背景提取 | 8.5 | 7.5 | 8.0 | **8.00** | stage3.md |
| 4 | plate solve + SPCC/PCC + 双窄带 | — | — | — | **8.20** | stage4.md（仅总评） |
| 5 | 线性反卷积 / 轻降噪 | 7.0 | 6.5 | 7.5 | **6.98** | stage5.md |
| 6 | 线性去星 / 星点层准备 | 7.5 | 7.0 | 7.0 | **7.18** | stage6.md |
| 7 | 主体拉伸 / Starless Stretch | 8.0 | 7.5 | 8.5 | **7.98** | stage7.md |
| 8 | Starless 深加工 / 星云增强 | 8.5 | 8.0 | 8.0 | **8.18** | stage8.md |
| 9 | 星点处理与合成 / 回星 | 9.0 | 9.0 | 8.0 | **8.70** | stage9.md |
| 10 | 最终降噪与导出 | 7.5 | 7.0 | 7.0 | **7.18** | stage10.md |
| | **D 维度平均**（9 份有值） | **7.94** | **7.28** | **7.67** | | |
| | **全局加权平均分** | | | | **7.69** | |

**最强 stage**：Stage 9（回星，8.70）——逻辑、归档、标准符合度均最高，Screen 混合与保色增益数学正确。
**最弱 stage**：Stage 2（裁切，6.83）——死配置（`stage2_color_artifact_max_crop` 已定义已钳制却从未读取）+ 纯像素统计误裁暗天区风险 + resume 路径 provenance 断点。Stage 5（6.98）与 Stage 10（7.18）紧随其后，短板分别为降噪反向降级与末端降噪无保护。

**跨维度观察**：
- D1（逻辑专业合理性）整体最高（均值 7.94），说明算法与工程架构选对了方向。
- D2（门禁来源与归档）整体最低（均值 7.28），是系统性短板——大量门禁阈值硬编码、来源语义不可见、模型/配置身份未归档。
- D3（行业标准符合度）均值 7.67，绝大多数对齐 PixInsight/Siril，偏离集中在「末端降噪无 mask」「GHS 自创映射」「缺几何星缩小」等少数点。

---

### 2.1 各 stage 一页式结论

- **Stage 1（7.65）**：准备阶段主线正确，但 CFA/比特深输入校验门禁缺失、register 统计未结构化——数据入口的可复现性弱端。
- **Stage 2（6.83）**：全仓最弱。死配置（`stage2_color_artifact_max_crop` 已定义已钳制却从不读取）+ 纯像素统计误裁暗天区 + resume provenance 断点。
- **Stage 3（8.00）**：背景提取是工程范本（三态+3σ 硬门+回滚），但 sufficient 判据两套标准并存、约 200 行死代码、授权判据硬编码未标来源。
- **Stage 4（8.20）**：plate solve + SPCC/PCC + 双窄带隔离架构合理；OSC 硬编码 IMX585 驱动 SPCC、双窄带质量门为宽带复用属偏离待验证。
- **Stage 5（6.98）**：线性反卷积/轻降噪主体合理，但降噪回退链反向降级（最激进档无门无回滚）、续跑指纹失真、plan_hash 名不副实。
- **Stage 6（7.18）**：starless-first 枢纽，四态证据链完整；arcsinh 预拉伸踩中 StarXTerminator 点名陷阱、SyQon 权重身份未归档。
- **Stage 7（7.98）**：全仓质量上限候选。黑场处理科学级；但两套拉伸架构并存、四条核心硬门无 env、AI 无 prompt 版本。
- **Stage 8（8.18）**：分区 mask+还原式保护+limited 不变量+9 项质量门；断点无签名、亮度/色度增强耦合、SASP 未验证。
- **Stage 9（8.70）**：全仓最强。Screen 数学正确+保色增益+五道闸；弱星/强星强度耦合、缺几何星缩小。
- **Stage 10（7.18）**：末端降噪定位正确、色彩管理超出社区平均；但无星点 mask、强度硬编码逃逸配置、review-only 时序缺陷、两份 JSON 不可复现。

## 三、D1 横向分析（逻辑专业合理性）

### 3.1 starless-first 架构自洽性

主线 **Stage 6 线性去星 → Stage 7 拉 starless → Stage 8 星云增强（仅 starless）→ Stage 9 回星** 在工程上自洽且保守：
- 去星在线性域、复用同一线性源做质量重试（不串联劣化，`stage7_star_separation.py:235` 重载 `stretched_name`）；
- Stage 7 是唯一点（`stage6_stretching.py:61`），之后全在非线性域；
- Stage 9 回星用 Screen + Alpha 约束层，杜绝加法星胀；
- 任一降级均回退到「无星底图」而非崩溃。

**自洽性结论：方向正确，是本项目质量上限所在。** 但存在两处架构意图被稀释：
- Stage 6 内部用 **arcsinh** 做可逆预拉伸（`SyQon-Starless.py:875/950`，`_IHS_TARGET=0.40`），而 StarXTerminator 官方明确警告 arcsinh 让星点像小椭圆星系、去星不全（见 §5.3 D3 偏离清单）——属「自洽但踩中行业点名陷阱」。
- Stage 10 末端降噪在已回星图上无差别作用（见 §3.4），相当于把 Stage 6 精心分离出的星点「再糊一遍」，是 starless-first 意图的部分自我抵消。

### 3.1.1 各 stage D1 关键证据速查（横向）

| Stage | D1 评分 | 关键正确点 | 关键逻辑风险 | 证据 |
|---|---|---|---|---|
| 1 | 8.0 | debayer/register/stack 主线正确 | CFA 相位/XISF 比特深无显式输入校验门禁 | stage1.md §2 |
| 2 | 7.5 | 裁切主体逻辑成立 | 纯像素统计无 footprint/coverage/weight 误裁暗天区；resume 不重建 crop_report | stage2.md §2 |
| 3 | 8.5 | 三态决策+3σ 硬门+hold-out 验证+事务回滚 | 硬门用 3σ 但 sufficient 排序仍用旧经验分数（两套标准） | stage3.md F1 |
| 4 | (8.2) | plate solve+SPCC/PCC+双窄带隔离 | 双窄带质量门为宽带复用 | stage4.md P1 |
| 5 | 7.0 | 线性反卷积主体合理 | 降噪回退链反向降级（见 §3.2） | stage5.md TOP1 |
| 6 | 7.5 | 线性域去星+可逆预拉伸+四态证据链完整 | arcsinh 预拉伸陷阱（见 §5.3 D3-1） | stage6.md §2.2 |
| 7 | 8.0 | 黑场三重保护+AI 只选 ID+确定性排序 | 两套拉伸架构并存（死代码） | stage7.md §2.4/§2.12 |
| 8 | 8.5 | 分区 mask+还原式保护+limited 不变量+9 项质量门 | 对比度/弱信号与饱和度不当耦合 | stage8.md §1.8 |
| 9 | 9.0 | Screen 数学正确+保色增益+五道闸防二次星点 | 弱星被强星降强度连带压低 | stage9.md §2.7/TOP3-1 |
| 10 | 7.5 | 双重降噪防护+色彩管理链路专业 | 末端降噪无星点 mask（见 §3.4）+ review-only 时序 | stage10.md §2.1/发现3 |

### 3.2 降噪次数与叠加

链路中有**两处降噪**：Stage 5 线性域轻降噪 + Stage 10 非线性末端降噪。两者的叠加由 `should_skip_final_denoise()`（`pipeline_safety.py:100-112`）防护——仅当「Stage5 已降噪 ∧ Stage8 未走降级 ∧ Stage8 质量良好」三者**与**关系成立才跳过 Stage 10 降噪。该规则专业且保守，直接规避「线性+非线性双重降噪导致塑料化」行业头号陷阱（Stage 10 报告 T1 规避）。

**实际问题在 Stage 5 与 Stage 10 各自的降噪质量**：
- Stage 5 降噪回退链「反向降级」：多尺度候选因细节保留不足被拒后，落到 `denoise -mod=0.50 -indep`（更激进、无质量门、无回滚）（stage5 报告 TOP1）。
- Stage 10 末端降噪强度固定 0.5、非自适应，且对星点无保护。

### 3.3 线性 / 非线性域边界

边界清晰且正确：
- Stage 7 是唯一「线性→非线性」不可逆转换点，具备三重黑场保护 + 四层高光保护（`stage6_services.py:2504-2529` / `:1543-1546` / `:2282-2315`）；
- Stage 10 末端降噪定位在「拉伸之后、回星之后」的非线性域（`:451-464`），与 NoiseXTerminator/AstroBackyard 建议顺序一致；
- Stage 10 的 PNG 导出明确跳过二次 autostretch（仅用 Stage 7 已验收渲染），保护影调（save_utils.py:246-263，T3 规避）；
- 不可逆几何裁剪集中在 Stage 2，Stage 10 内无裁剪（D1 正确）。

**边界结论：正确。** 唯一模糊点是 Stage 9 在「非线性域做保色星层拉伸」而非伪线性重组（D3 §4.6 部分符合），但数值上安全。

### 3.4 上下游契约断裂（横向风险）

跨 stage 存在多处契约不连续，均导致「下游无法感知上游意图」或「同义判定双源真值」：

| 断裂点 | 上游 → 下游 | 证据 | 后果 |
|---|---|---|---|
| Stage6.5 门建议被忽略 | Stage 6 预去星门算出 `recommended_starless_input` 却主动忽略 | `stage7_star_separation.py:575-582` | 该门对实际路由无约束力 |
| Stage 8 断点无签名 | Stage 8 产物不进 `checkpoint-manifest.json`（仅 1/2/5） | `stage_contracts.py:18` / `task_workspace.py:862` | 续跑依赖未校验磁盘文件 |
| 必需星点门双源真值 | Stage 9 与 Stage 10 各实现一次 `stars_required AND NOT stars_applied` | `stage9_star_remixing.py:858` + `stage10_export.py:411-415` + `processor_runtime.py:2010` | 双源真值隐患，改一处漏一处矛盾 |
| `starless_enhanced` 名不符实 | star-preserve bypass 路径下该文件实际含星 | `stage8_pixels.py:469,1479,1876,1899` | 候选链顺序阅读可疑，但正确性无损 |
| Stage 7 兼容检查点跳过 | 去星质量诊断能力一并失效；去星失败 review 路径无硬门 | `seestar_Superimpose.py:2530-2534` / `stage6_stretching.py:8` | review 路径产出无质量保护的非线性图 |

### 3.5 降级路径「沉默的劣化」

多个降级/失败分支仍报 success / partial_success，但产物已被静默劣化，人工难以归因：
- **Stage 10 后置 review-only 晚于降噪**：严格质量门 `needs_conservative_rerun` 在降噪执行后才置 review-only（`:1021`），而过度降噪本身会抬高 `artifact_score` 触发该门，形成「降噪→artifact 升高→标 review-only」自激，且 review 图已被降噪污染、无降噪前快照（stage10 报告发现 3 / D1-c）。
- **Stage 10 保存失败仅 degraded**：质量门整体包在 `elif stage_saved:` 内（`:983`），保存失败则质量门根本不跑，比质量不达标更严重的情形反而得到更宽松状态（D1-d）。
- **Stage 1 多个 degraded 串联仍报 success**：register 失败率等未结构化归档（G9 仅自由文本），跨 stage 劣化累积无量化（stage1 报告）。
- **Stage 10 全候选缺失兜底「沿用当前 Siril 图像」**：来源不确定的内存图以正式名 `result_processed`/`result_final` 交付（`:480-482`，D1-a）。

---

### 3.6 全局四态一致性（横向）

全局状态机（`processor_runtime.py:2001-2022`）优先级为 `failed > review_required > partial_success > success`，是跨 stage 降级的汇聚点。横向核查发现两类一致性隐患：

- **双源真值**：必需星点判定在 `stage10_export.py:411-415` 与 `processor_runtime.py:2010` 重复实现（见 §3.4），目前一致但属典型「改一处漏一处」隐患。
- **状态宽松倒置**：Stage 10 保存失败仅 `degraded`（partial_success），而质量不达标反而可能触发 review_required——更严重情形得到更宽松状态（见 §3.5 D1-d）。
- **降级仍报 success**：Stage 1 多个 degraded 串联无量化累积，`stage6_services.py` 各 fallback 仅记 `partial_success`，下游难以感知上游已劣化多少。

**横向结论**：四态框架本身完整，但「降级=劣化」的语义在多个节点被稀释（review 图被降噪污染、保存失败反宽松、串联 degraded 无量化），属 §3.5「沉默的劣化」的系统性表现。

## 四、D2 横向分析（门禁来源与归档）

### 4.1 阈值来源分布统计（横向）

汇总各 stage 门禁来源分类（基于各单报 D2 章节）：

| 来源分类 | 出现 stage | 典型代表 |
|---|---|---|
| PipelineConfig 默认值（集中，常被标「来源未标注」） | 1/2/3/5/6/7/8/9/10 | 绝大多数 `stageN_*` 阈值 |
| PipelineConfig + env 覆盖 + clamp（完整三段式） | 7/8/9/10（部分门） | Stage10 G1-G4；Stage7 约 25 个 env 键 |
| env only | 5/7/10 | `SEESTAR_SYQON_TIMEOUT_SEC`、`SEESTAR_SIRILPY_TIMEOUT_SEC` |
| 运行时推导 | 1/3/5/6/7/9/10 | 背景自适应阈值、remix_scale、denoise 模式 |
| 硬编码（代码内字面量，不入配置） | 2/3/5/6/7/10 | 见 §4.3 死配置与 §4.2 硬编码热点 |
| AI 返回（仅候选 ID） | 7/8/9 | `selected_candidate_id` 契约，参数归代码 |

**结构性结论**：门禁「集中配置」原则（AGENTS.md）在多数 stage 落地，但**来源语义缺失**是跨 stage 通病——`models.py` 仅给数值与一句用途注释，无可追溯引用（empirical / policy / documented）。Stage 7 报告指出的 env 白名单「覆盖面与阈值重要性负相关」在 Stage 10 同样存在（最核心的降噪强度反而无 env）。

### 4.1.1 各 stage 归档缺口对照（横向）

| Stage | 是否可复现（单 stage 判定） | 主要归档缺口 | 证据 |
|---|---|---|---|
| 1 | 部分 | register 失败率等统计未结构化（G9 仅自由文本） | stage1.md D2 |
| 2 | 部分 | resume 路径不重建 crop_report（provenance 断点） | stage2.md §3.4 |
| 3 | 可（主链） | 硬编码授权判据无代码版本指纹；约 200 行死代码 | stage3.md F2/F3 |
| 4 | 可 | processing-plan 未记录 SPCC sensor/filter/white_ref | stage4.md P2 |
| 5 | **不可**（断点） | 续跑指纹不覆盖真实配置；plan_hash 名不副实；GraXpert 模型无权重 SHA | stage5.md TOP2/TOP3 |
| 6 | 可（主链） | SyQon 模型权重身份未进结构化 JSON（仅日志） | stage6.md §3.4 |
| 7 | **可（有条件）** | AI 无 prompt_version/model_id；四条核心硬门不进 Stage7 JSON | stage7.md §3.6 |
| 8 | 可（主链） | Stage8 产物不在签名断点白名单 | stage8.md §2.3 |
| 9 | 可（优秀） | 跨档重试中间指标全量落盘未逐字段验证 | stage9.md §3.4 |
| 10 | **不可**（需 5 份产物） | 两份顶层 JSON 不足以复现；降噪强度不可见 | stage10.md §3.6 |

**横向结论**：可复现性呈「中间 stage（6/7/8/9）强、两端 stage（1/2/5/10）弱」的哑铃型分布。弱端恰好是数据入口（1/2）、降噪（5）、导出（10）三个最容易引入不可逆变更或最需要溯源的节点——风险敞口最大。

### 4.2 硬编码热点排名（横向，不入配置 / 未标来源）

| 排名 | 硬编码项 | 位置 | 影响 |
|---|---|---|---|
| 1 | 降噪强度 `0.5`（双写） | `stage10_export.py:576, 641` | 逃逸整个配置/钳制体系，两份 JSON 不可复现 |
| 2 | Stage 7 四条核心硬门（bg_median_min 等）无 env 覆盖 | `stage6_services.py:1533-1559` | 现场调参必须改代码 |
| 3 | Stage 7 排序权重 100% 硬编码 | `stage6_services.py:1747-1858, 2027-2059` | 死配置 YAML 误导 |
| 4 | Stage 6 arcsinh 预拉伸常量（`_IHS_B=6.0/_IHS_TARGET=0.40`） | `SyQon-Starless.py:126-127` | 来源（SyQon 训练分布 vs 自定）未验证 |
| 5 | Stage 3 授权判据字面量（G03 的 8、G09 的 3/8/0.55） | stage3 报告 F2 | 不入档、无代码版本指纹 |
| 6 | Stage 5 背景守卫 8 阈值全硬编码 | stage5 报告 TOP3 | 占门禁 44%，来源不可见 |
| 7 | Stage 4 OSC 传感器硬编码 `Sony IMX585` 驱动 SPCC | `stage4_color_calibration.py:882-905` | 模型存在性/曲线正确性未验证 |

### 4.3 可复现性总评（横向）

**判定：全链路可复现性为「局部可复现、关键节点有断点」。** 可复现的强项：Stage 7 候选选择（纯函数 9 元组，落盘可脱机复现）、Stage 9 回退/强度/回滚结构化归档、Stage 10 交付文件 SHA-256 + 原子写。

**不可复现 / 弱复现的断点（合并 P0）**：
- Stage 5：续跑指纹不覆盖真实生效配置（8 个 GUI 字段 vs 多个 env-only 参数）；`plan_hash` 实为 run-manifest 哈希；GraXpert 模型仅记版本目录名无权重 SHA。
- Stage 6：SyQon 模型权重身份（哪个 `.pt`、版本几）未进结构化 JSON，仅落日志（`:750-751`）。
- Stage 7：AI 请求无 `prompt_version`/`schema_version`/`model_id`；四条核心硬门阈值不进 Stage 7 自身 JSON。
- Stage 8：`stage8_enhanced.fit` 不在签名断点白名单。
- Stage 10：两份顶层 JSON 不足以复现（需 5 份产物：processing-plan + pipeline-result + final_quality_report + managed-output + stage 记录）。

### 4.4 死配置 / 僵尸代码清单（单独成节）

这是横向上最突出、可维护性危害最高的一类问题。**危害不在于运行风险（死代码不执行），而在于误导维护者下错判断。**

| 文件 / 位置 | 死组件 | 为何致命 | 证据 |
|---|---|---|---|
| `pipeline/models.py:108` | `stage2_color_artifact_max_crop` 已定义已钳制但从未读取 | 彩边裁切实际用硬编码 90% 门（L469）兜底，该配置形同虚设 | stage2.md §3 |
| `pipeline/models.py:103` | `crop_margin` 残留不驱动 | 死字段，易被误改 | stage2.md §3 |
| `pipeline/models.py:117` | `stage2_level_artifact_window` 仅 getattr 回退 81 | 未正式配置 | stage2.md §3 |
| `stage3` 报告 | 约 200 行不可达死代码（`:1135-1331`）+ 僵尸读取 `stage3_gradient_apply_min`/`stage3_dirty_apply_min` | 旧字段断言属实但代码不可达 | stage3.md F3 |
| `stage5` 报告 | `background_risk` 计算未用；`_stage5_denoise_mode` 三分支塌缩；`auto_tune denoise_mod` 未消费 | 死门禁 + 塌缩分支 | stage5.md TOP3 |
| `stage6_services.py:501-808` | 整套 legacy 拉伸架构（`_stage6_stretch_candidate`/`_stage6_strategy_from_features`/`_stage6_candidate_specs`/`_stage6_candidate_label`） | 命名比活架构（`_stage7_*`）更贴合文件名 `stage6_services.py`，极易改错 | stage7.md §2.12 |
| `stretch_candidate_evaluator.py` | 整模块死代码（被 import 未调用，仅测试引用） | 有测试覆盖给人「活代码」错觉 | stage7.md §2.12 |
| `policy_selector.py:45-232` + `configs/policies/*.yaml` | `stage6_stretch.scoring`/`hard_reject` 死配置 | 排序权重实际 100% 硬编码在活函数，看 YAML 会得出「权重可配置」的错误结论 | stage7.md §2.12 |
| `pipeline/task_plan.py:36,254-319,322-373` | `PROCESSING_PLAN_SCHEMA=v2` 及 `build/verify_processing_plan` | **仅被测试引用，生产零引用**；v1 实际产物反无等价测试，测试覆盖给出虚假信心 | stage10.md §3.7 |

**横向结论**：死配置/僵尸代码在 Stage 2/3/5/7/10 均有，Stage 7 最严重（两套完整架构并存）。建议统一清理（见 §7 路线图）。

### 4.5 env 覆盖白名单覆盖度矩阵（横向）

| Stage | 门禁总数（约） | 可 env 覆盖 | 不可 env 覆盖（典型） | 覆盖度评价 |
|---|---|---|---|---|
| 1 | 少 | 部分 | 输入校验门禁缺失 | 低（缺门禁本身） |
| 2 | 中 | 部分 | `stage2_color_artifact_max_crop`（死配置，本就不生效） | 低 |
| 3 | 多 | 部分 | 授权判据字面量（G03/G09） | 中 |
| 4 | 中 | 部分 | 双窄带质量门 | 中 |
| 5 | 多 | 部分 | 背景守卫 8 阈值全硬编码 | 中低 |
| 6 | 11（G1-G11） | 仅 G5 | 其余全 PipelineConfig 默认 | 中（阈值来源未标） |
| 7 | 15 硬门 | 9 | **H1-H4 四条核心安全门** | **负相关**（重要门不可覆盖） |
| 8 | 20+ | 多数 | limited 模式少数 | 高 |
| 9 | 30+ | 极少 | 全部 CLAMP 钳制，无 env | 高（钳制完备但不可调） |
| 10 | 24（G1-G24） | 4 阈值 + 3 bool | **G5 降噪强度** + G14/G15/G19/G21/G23 硬编码 | 中（核心强度逃逸） |

**横向结论**：env 覆盖度与「阈值重要性」呈**系统性负相关**——Stage 7 四条核心安全门、Stage 10 降噪强度这些最影响画质的参数反而不在白名单，而大量次要阈值可覆盖。这是 AGENTS.md「画质风险参数必须有上限或回退值」原则在「可调性」维度的落实缺口：有钳制但不可现场调参，A/B 实验必须改代码。

---

## 五、D3 横向分析（算法行业标准符合度）

### 5.1 与 PixInsight / Siril 对齐度（总览）

| 行业标准实践 | 本项目实现 | 符合度 | 涉及 stage |
|---|---|---|---|
| starless-first 工作流（去星→拉 starless→回星） | 主线架构 | ✅ 符合 | 6/7/8/9 |
| Asinh 主拉伸 + linked MTF | `stage6_services.py:2273,2537` | ✅ 符合 | 7 |
| 黑场不裁到 0（严于 STF -2.8σ） | `:2504-2529` 3σ 余量 | ✅✅ 严于行业 | 7 |
| 分区 mask + 局部增强（starless + masked local） | `stage8_generate_starless_masks` | ✅ 符合 | 8 |
| Screen 回星（非加法） | `stage9_quality.py:454-517` | ✅ 符合 | 9 |
| 保色星层独立拉伸（标量增益） | `stage9_quality.py:1313-1317` | ✅ 符合 | 9 |
| 色噪/亮噪分离降噪 | Stage10 四模式矩阵 | ✅ 符合 | 10 |
| 双重降噪防护 | `pipeline_safety.py:100-112` | ✅ 符合 | 5/10 |
| FITS 科学存档不可变 + SHA-256 | `managed_output.py:259-372` | ✅ 符合 | 10 |
| 16-bit sRGB/ICC 受管导出 | `managed_output.py:54-230` | ✅ 优于社区平均 | 10 |
| autostretch 仅作参照不作输出 | preview_ref `reference_only:True` | ✅ 优于常见做法 | 7 |

**对齐度结论：高。** 绝大多数核心算法与 PixInsight / StarXTerminator / Siril / scopetrader「5 条过度处理红线」一致。

### 5.1.1 各 stage D3 评分依据与代表 URL（横向）

| Stage | D3 评分 | 高度符合项 | 代表 URL（节选） |
|---|---|---|---|
| 1 | 8.0 | debayer/register/stack 符合 Siril 标准流程 | siril.readthedocs.io（Commands） |
| 2 | 7.0 | 裁切几何正确；彩边处理偏离 PixInsight 裁切经验 | pixinsight 论坛裁切讨论 |
| 3 | 8.0 | 背景提取 3σ 统计符合 GraXpert/APP 经验 | GraXpert 文档 |
| 4 | (8.2) | plate solve+SPCC 符合 PixInsight/Siril 流程 | pixinsight.com（SPCC） |
| 5 | 7.5 | 线性反卷积符合 PixInsight 反卷积范式 | pixinsight 论坛反卷积 |
| 6 | 7.0 | starless-first 符合社区主流；arcsinh 偏离 StarXTerminator | rc-astro.com/starxterminator-usage-notes；siril.org/2026/01/zenith |
| 7 | 8.5 | Asinh 主拉伸、linked MTF、黑场不裁、autostretch 仅参照 | siril.readthedocs.io（stretching）；pixinsight STF；rc-astro StarXTerminator |
| 8 | 8.0 | 分区 mask+局部增强符合 NGC7023-HDR；scopetrader 5 红线 | pixinsight.com/tutorials/NGC7023-HDR；scopetrader.com |
| 9 | 8.0 | Screen 回星、保色增益、unscreen 思路符合 RC Croman | rc-astro.com（StarXTerminator notes）；pixinsight 论坛 18602/21794 |
| 10 | 7.0 | FITS 不可变、16-bit sRGB/ICC、Astro-TIFF 符合；无 mask 偏离 | astrobackyard.com/astrophotography-noise/；astro-tiff.sourceforge.io；fits.gsfc.nasa.gov |

**横向结论**：D3 符合度与「是否在线性/非线性边界做标准操作」强相关——Stage 7（拉伸边界）、Stage 9（回星）符合度最高；Stage 6（预拉伸选型）与 Stage 10（末端降噪）因自创/偏离项最多而最低。

### 5.2 关键偏离清单及风险评级

| # | 偏离 | stage | 风险 | 依据 |
|---|---|---|---|---|
| D3-1 | arcsinh 预拉伸致小星系去星不全 | 6 | **高** | StarXTerminator 官方警告 `rc-astro.com/starxterminator-usage-notes`；`_IHS_TARGET=0.40`（`SyQon-Starless.py:126-127`） |
| D3-2 | GHS 参数自创 sigma 映射，未用 SP/LP/HP | 7 | 中 | Siril `autoghs`（`:960-1059`）；LP=0 无暗部保护 |
| D3-3 | GHS 未启用保色模式 | 7 | 中 | 靠事后色度门被动拦截，拦不住色相偏移（未验证 `-linked` 是否隐含保色） |
| D3-4 | 末端降噪无星点 mask | 10 | **高** | AstroBackyard 明确要求 mask（`astrobackyard.com/astrophotography-noise/`）；starless-first 本可复用 Stage9 掩膜 |
| D3-5 | 降噪强度非自适应（固定 0.5） | 10 | 中 | NXT 默认 0.85-1.0 更激进但随场景；本项目部分符合 |
| D3-6 | SASP 外部插件算法未验证 | 8 | 中（不影响可用性） | 仓库未含插件本体，`sasp_runner.py:641-730` |
| D3-7 | 单尺度增强（弱于多尺度 LHE/MMT） | 8 | 低 | `stage8_pixels.py:991-1399` |
| D3-8 | 缺几何星缩小（仅亮度抑制） | 9 | 中 | 无 erosion/MT 算子（§4.4） |
| D3-9 | 非伪线性重组 | 9 | 低 | 像素空间重组依赖 starless 已非线性 |
| D3-10 | 色彩契约「只审计不保证」 | 10 | 中 | `output_color.py:236-375` `rewrote_outputs=False`；关 managed 后全不符 |

### 5.3 Seestar 特化 vs 真风险

**Seestar 特化（合理，非真风险）：**
- OSC 传感器硬编码 `Sony IMX585` 驱动 SPCC 光谱模型（`stage4_color_calibration.py:882-905`）：属 Seestar 设备特化，但**预设存在性/曲线正确性未验证**（U1），应补验证而非视为缺陷。
- `chroma` 自创降噪模式（full 后恢复原始亮度，`stage10_export.py:347-370`）：方向正确（等价于只降色度），但色度-亮度一致性未验证（U1/U6），属「自创合理，待量化」。
- SyQon「需轻拉伸、最好后拉伸」被本项目以可逆轻拉伸替代：社区折中满足，应在文档声明偏离。

**真风险（应优先处理）：**
- D3-1 arcsinh 去星不全（小星系密集场系统性残星，不可复现权重身份叠加）。
- D3-4 末端降噪无 mask（星点 FWHM 软化、二次星点风险）。
- D3-2/D3-3 GHS 自创映射 + 未保色（色相漂移可能被事后门漏掉）。

### 5.4 行业陷阱规避率汇总（横向）

| Stage | 规避率 | 未规避/部分规避的关键陷阱 |
|---|---|---|
| 1 | — | 输入 CFA/比特深无校验门 |
| 2 | — | 纯像素统计误裁暗天区 |
| 3 | — | 两套 sufficient 标准并存 |
| 6 | — | arcsinh 使星点像小星系（D3-1） |
| 7 | 14 项中 11 完全 + 1 部分 + 2 未规避 | GHS 不设 LP（T13）、去星失败路径无硬门（T14） |
| 8 | 高 | SASP 多尺度增强未验证（D3-6） |
| 9 | 高 | 几何 bloat 无收缩（D3-8）、非伪线性重组（D3-9） |
| 10 | 12 项中 9 规避 + 1 部分 + 2 未规避 | 末端降噪软化星点（T2）、磁盘满静默截断（T12）、ICC 仅 managed 路径（T5） |

**横向结论**：行业陷阱规避率整体高（Stage 7/8/9 达 75%-90%+），未规避项集中在两类共性缺口——(a) **星点相关保护缺失**（Stage 10 降噪无 mask、Stage 9 无几何星缩小），(b) **环境/资源门缺失**（Stage 10 磁盘空间门）。这两类共性缺口应在路线图中优先闭环。

---

## 六、全局问题清单（P0 / P1 / P2）

> 分级原则：**P0 仅放真数据错误或不可复现**；P1 重要工程/专业问题；P2 技术债/长期优化。跨 stage 系统性问题已合并。

### P0（真数据错误 / 不可复现 — 必须处理）

**P0-1｜可复现性断点（跨 stage 系统性）**
- 所属 stage：5/6/7/8/10
- 证据：`stage5` 续跑指纹不覆盖 env-only 参数 + `plan_hash` 为 run-manifest 哈希；`stage7_star_separation.py:750-751` 模型权重仅落日志；`stage6_services.py:1747` 排序权重硬编码无 prompt 版本；`stage_contracts.py:18` Stage8 无签名断点；`stage10_export.py:301-324` 两份 JSON 不可复现（需 5 份）
- 影响：科学级可复现性硬伤，旧 run 无法定位当时模型/配置/决策依据
- 建议：建立「模型权重 SHA + 真实生效 config 快照 + 决策指标」三件套进 `pipeline-result.json`；Stage 8 纳入 soft-checkpoint（见 §7）

**P0-2｜末端降噪无星点保护 mask + 强度硬编码逃逸配置**
- 所属 stage：10
- 证据：`stage10_export.py:117-209`（四模式矩阵，full/separate 无 mask）、`:576,641`（强度 0.5 双写，无 env 无 clamp）
- 影响：已回星图被无差别降噪，软化星点 FWHM、可能引入塑料感/二次星点；starless-first 链路意图自我抵消；两份 JSON 不可复现
- 建议：复用 Stage9 `_stage9_last_star_overlay_mask`（`stage9_star_remixing.py:864`）做加权合并；把强度提升为受管配置项（`models.py` + env + clamp `[0.0,0.8]`）

**P0-3｜降噪回退链反向降级**
- 所属 stage：5
- 证据：stage5 报告 TOP1（多尺度候选被拒后落到 `denoise -mod=0.50 -indep` 更激进、无质量门、无回滚）
- 影响：回退结果可能比被拒候选更差，且静默生效（见 §3.5 沉默的劣化）
- 建议：为降噪回退链补质量门与回滚，拒绝则退回上一档而非更激进

**P0-4｜arcsinh 预拉伸去星不全风险**
- 所属 stage：6
- 证据：`SyQon-Starless.py:875/950`（`_IHS_TARGET=0.40`）；StarXTerminator 警告 `rc-astro.com/starxterminator-usage-notes`
- 影响：小星系密集场（M51、室女座星系团）系统性残星，且模型权重不可复现叠加
- 建议：评估 arcsinh→MTF 类替换或下压 `_IHS_TARGET`；在 `workflow.md` 明示风险与缓解

**P0-5｜Stage 7 去星失败 review 路径无硬门 + 兼容检查点跳过**
- 所属 stage：7
- 证据：`stage6_stretching.py:8`（`_run_with_stars_review_stretch` 仅 `autostretch -linked` 无硬门）；`seestar_Superimpose.py:2530-2534`（兼容检查点无条件 skipped）
- 影响：去星失败时产出无质量保护的非线性图进入后续流程
- 建议：review 路径至少跑 H1-H4 硬门并记录 `advisory_only`（stage7 报告 P1-4）

### P1（重要，应处理）

**P1-1｜死配置 / 僵尸代码跨 stage 清理**
- 所属 stage：2/3/5/7/10
- 证据：见 §4.4 清单（stage2 `stage2_color_artifact_max_crop`、stage3 约200行死代码、stage7 整套 legacy 架构、stage10 v2 影子 schema 等）
- 影响：误导维护者下错判断（如改 YAML 权重实则无效）
- 建议：先加 `# DEAD CODE` 标记（零风险），确认无依赖后删除；v2 schema 二选一合并或删除

**P1-2｜Stage 8 断点完整性缺口**
- 所属 stage：8
- 证据：`stage_contracts.py:18` / `task_workspace.py:862`；`stage8_enhanced.fit` 仅磁盘存在
- 影响：续跑依赖未校验磁盘文件，手动移动即回滚失败（`status='degraded'`）
- 建议：为 Stage8 关键产物补内容哈希或纳入 soft-checkpoint（同 P0-1）

**P1-3｜Stage 7 四条核心硬门无 env 覆盖**
- 所属 stage：7
- 证据：`stage6_services.py:1533-1559`；grep `PROJECT_ENV_ALLOWED_KEYS` 零匹配
- 影响：核心门无法做 A/B 实验，跨 run 对照必须改源码
- 建议：加 `SEESTAR_STAGE6_BG_MEDIAN_MIN` 等 4 键 + 钳制（stage7 报告 P0-2）

**P1-4｜Stage 9 弱星/强星强度耦合（牺牲弱星风险）**
- 所属 stage：9
- 证据：`stage9_weak_star_screen_intensity_min=0.40` + `remix_scale` 降强度连带压低弱星（TOP3-1）
- 影响：强星触发降强度时弱星同步消失，3 档固定重试无针对弱星的自适应补偿，可能回滚到无星底图牺牲全部弱星
- 建议：弱星保持独立 Screen 强度下限；重试档位与拒绝原因联动（stage9 报告 P0）

**P1-5｜Stage 10 review-only 时序 + 保存失败质量门跳过**
- 所属 stage：10
- 证据：`:610,1021`（后置 review-only 晚于降噪）；`:983`（`elif stage_saved:` 包裹质量门）
- 影响：review 图被降噪污染无快照；保存失败仅 degraded 非 review_required
- 建议：降噪前预判 strict gate 并保留 predenoise 快照；`not stage_saved` 直接置 review_only

**P1-6｜Stage 6.5 门建议被忽略**
- 所属 stage：6
- 证据：`stage7_star_separation.py:575-582`
- 影响：该门对实际路由无约束力，死逻辑
- 建议：消费 `recommended_starless_input` 或显式声明放弃（stage6 报告 P1-3）

**P1-7｜必需星点门双源真值**
- 所属 stage：9/10
- 证据：`stage10_export.py:411-415` 与 `processor_runtime.py:2010` 重复实现
- 影响：双源真值，改一处漏一处矛盾
- 建议：抽到 `pipeline_safety.py` 单一函数（stage10 报告 P1-4）

**P1-8｜阈值来源语义未标注（跨 stage）**
- 所属 stage：3/5/6/7/10
- 证据：各单报 D2（models.py 仅数值+用途注释，无 empirical/policy/documented 标注）
- 影响：审计不可追溯到行业依据
- 建议：为 `stageN_*` 门限 docstring 加 `source` + 参考链接（stage6 报告 P1-4）

**P1-9｜Stage 4 OSC 硬编码 SPCC 光谱模型未验证**
- 所属 stage：4
- 证据：`stage4_color_calibration.py:882-905`（`Sony IMX585`）；U1 预设存在性/曲线正确性未验证
- 影响：若传感器模型错配，校色基础偏差
- 建议：补 SPCC sensor/filter/white_ref 标识进 processing-plan；验证预设存在性

### P2（建议，技术债 / 长期优化）

**P2-1｜全仓 Stage 编号/文件名错位统一重命名**（Stage6 去星在 `stage7_star_separation.py`、Stage7 拉伸在 `stage6_stretching.py`）—— 降低维护误读，多份报告共同建议。
**P2-2｜Stage 10 导出前补磁盘空间门**（G24 缺失，`save_utils.py:209` 失败仅 degraded）—— 明确 failed 而非事后降级。
**P2-3｜Stage 10 色彩契约从「审计」升级为「保证」**（`output_color.py:236-375` `rewrote_outputs=False`）—— 不符时至少置 degraded。
**P2-4｜收敛复现所需产物数量**（Stage 10 需 5 份 → 把 `denoise_plan` 合入 `pipeline-result.json`）。
**P2-5｜为自定义 schema 提供 IVOA Provenance 映射**（四套 `seestar.*.v1`，`ivoa.net/documents/ProvenanceDM`）。
**P2-6｜Stage 7 GHS 补 LP/明确记录为有意不设**（D3-2/D3-3）。
**P2-7｜Stage 8 增强尺度表达（多尺度对比分支）**（D3-7，性能增强非缺陷）。
**P2-8｜Stage 9 增加几何星缩小算子（MT/erosion）**（D3-8）。
**P2-9｜Stage 10 `chroma` 自创模式色度-亮度一致性验证**（U1/U6）。

---

## 七、系统性改进路线图

> 验证方式统一参考 `AGENTS.md` Validation：`python -m py_compile` 编译检查、`bash -n` 语法检查、人工核对归档产物；禁止改项目代码（本研究仅产出本报告，路线图供维护者实施）。

### 批次一：立即修（P0，正确性 / 可复现性）

| 项 | 改动范围 | 验证方式 |
|---|---|---|
| P0-1 可复现性三件套（权重 SHA + 真实 config 快照 + 决策指标） | `stage7_star_separation.py:750` / `syqon_starless.py:199` 写 `syqon_model={name,weight_path,sha256}`；`pipeline_safety.py` 续跑指纹覆盖 env-only 参数；`processor_runtime.py` 写决策指标 | `py_compile` 各改文件；人工核对 `pipeline-result.json` 含权重 SHA 与 config 快照 |
| P0-2 末端降噪 mask + 强度受管 | `stage10_export.py:624-913` 接入 `_stage9_last_star_overlay_mask`；`models.py` 加 `stage10_denoise_strength`；`processor_runtime.py:1129-1133,1312-1315` 加 env+clamp | `py_compile`；构造含星测试图验证星点 FWHM 未软化 |
| P0-3 降噪回退链补质量门+回滚 | `stage5` 降噪回退逻辑 | `py_compile`；人工核对回退档不比候选更激进 |
| P0-4 arcsinh→MTF 评估 | `SyQon-Starless.py:126-127,875,950` | 文档声明风险；构造小星系测试场验证残星率 |
| P0-5 Stage7 review 路径补硬门 | `stage6_stretching.py:8` 调 `_validate_stage6_stretch_quality` | `py_compile`；人工核对 review 产物记录 H1-H4 |

### 批次二：下一迭代（P1，工程/专业质量）

| 项 | 改动范围 | 验证方式 |
|---|---|---|
| P1-1 死代码清理 | `models.py:103,108,117` / `stage3` 死代码 / `stage6_services.py:501-808` / `stretch_candidate_evaluator.py` / `policy_selector.py:45-232` / `task_plan.py` v2 | 先加 `# DEAD CODE` 标记零风险；`grep` 确认无外部引用后删除；`py_compile` + 跑相关测试 |
| P1-2 Stage8 断点完整性 | `task_workspace.py:853` 扩 soft stage 集合 / `processor_runtime.py:2060` 写 sha256 | 人工核对 `pipeline-result.json.outputs` 含 Stage8 哈希 |
| P1-3 Stage7 核心硬门 env | `processor_runtime.py:170-232,1036-1172` | `bash -n`；设 env 后跑单测核对生效 |
| P1-4 Stage9 弱星解耦 | `stage9_quality.py:477-481,2123` | 构造强星降强度场景验证弱星保留 |
| P1-5 Stage10 时序修复 | `stage10_export.py:610,983,1021` | 人工核对 review bundle 含 predenoise 快照 |
| P1-6/P1-7 死逻辑/双源真值 | `stage7_star_separation.py:575` / `pipeline_safety.py` 抽函数 | `py_compile`；`grep` 确认单点定义 |
| P1-8 阈值来源语义 | `models.py` 各 `stageN_*` docstring | 人工核对 docstring 含 source + URL |
| P1-9 Stage4 SPCC 验证 | `stage4_color_calibration.py:882` | 验证 IMX585 预设存在性，写标识进 plan |

### 批次三：技术债清理（P2，长期）

P2-1 重命名错位文件（影响面大，需全仓 grep 同步）→ P2-2 磁盘空间门 → P2-3 色彩契约保证 → P2-4 收敛复现产物 → P2-5 IVOA 映射文档 → P2-6/7/8/9 各算法增强与验证。

**优先序建议**：批次一（P0）> P1-1（死代码，因其放大所有其他修改的误改风险）> P1-2/3/4/5（质量与安全）> P1-6~9 > 批次三。

### 7.4 路线图预期收益汇总

| 批次 | 投入量级 | 预期收益 | 风险 |
|---|---|---|---|
| 一批（P0） | 中（5 项局部改动） | 闭合科学级可复现性断点；消除末端降噪对星点的不可逆损伤；消除降噪反向降级 | 低（多为加字段/加 mask，不改动主算法） |
| 二批（P1） | 中高（含死代码全仓清理） | 消除维护误导面；Stage8 断点完整；核心门可 A/B；弱星不再被误杀；时序/双源隐患闭合；门禁来源可追溯 | 中（死代码删除需全仓 grep 确认无引用） |
| 三批（P2） | 低-中（多为增强/文档） | 命名统一降低误读；资源门防静默失败；溯源对齐 IVOA；算法增强更接近 PixInsight 多尺度 | 低 |

**关键路径**：P0-1（可复现性三件套）应作为最高优先，因其为其余所有验证提供基线——若无法复现「本次跑了什么」，后续的调参/回滚均失去锚点。

---

## 八、未验证项汇总

> 以下为各单报声明「未验证」项的合并。全部为静态审计限界（未运行 pipeline、未实机验证第三方工具），严禁视为已确认缺陷。

| # | 未验证事项 | 涉及 stage | 为何未验证 | 如何验证 |
|---|---|---|---|---|
| U-1 | SyQon/Zenith 模型权重训练集、架构、版本校验和未公开 | 6 | siril.org 公告仅述设计哲学，仓库无权重本体 | 公开权重 SHA 或标注不可复现 |
| U-2 | `_IHS_TARGET=0.40`/`_IHS_B=6.0` 是否 SyQon 训练分布推荐值 | 6 | 代码注释未给出来源 | 查 SyQon 文档或作者确认 |
| U-3 | arcsinh 陷阱在真实 Seestar 数据上的触发率 | 6 | 未运行 pipeline | 构造小星系密集场（M51）实跑统计残星率 |
| U-4 | Stage7 `STAGE7_ASINH_STRETCH_MIN/MAX` 具体数值 | 7 | 仅确认钳制调用存在 | 读模块常量区逐行核对 |
| U-5 | Siril 1.4.0 `autoghs` 是否提供保色开关 / `-linked` 是否隐含保色 | 7 | 查 stable 分支未针对 1.4.0 | 查 1.4.0 命令参考 + 实机 |
| U-6 | `intensity_scale` 阶段归属（workflow 称 Stage7，grep 在 stage7_quality.py） | 7/9 | 文件阶段归属未审计 | 跨文件核对 |
| U-7 | GraXpert/SASP 模型权重训练集与算法细节 | 5/8 | 仓库未含插件本体 | 获取插件文档与版本 |
| U-8 | Stage4 SPCC 预设存在性 / 曲线正确性（IMX585） | 4 | 未实机验证 | 跑 SPCC 核对 sensor/filter 标识 |
| U-9 | Stage8 limited 模式实际触发频率与跳过率 | 8 | 未运行真实数据 | 批量运行统计 |
| U-10 | Stage10 `chroma` 自创模式色度-亮度一致性（高对比边缘色偏） | 10 | 需合成测试图量化 | 构造合成图对比 LAB chroma-only |
| U-11 | Stage10 受管 TIFF 在无系统 sRGB ICC 的 Linux 行为 | 10 | 以 macOS 路径为主未实测 | Linux 环境实跑 |
| U-12 | strict gate 各阈值（0.34/0.42/0.45/0.55/0.00016/0.00022/0.52/0.62）标定依据 | 10 | 代码注释无数据集 | 查标定记录或补实验 |
| U-13 | Stage9 跨档重试完整超标项数值是否全量落盘 | 9 | 未逐字段核对中间档 trace | 读 `stage9_remix_quality.json` 全字段 |
| U-14 | Stage9 回滚到无星底图时 Stage10 是否完全跳过星处理 | 9/10 | Stage10 消费分支未读验证 | 读 Stage10 消费 `stars_applied=false` 分支 |
| U-15 | 上游 starmask 实际 halo 残留量级 | 6/9 | 依赖 Stage6 去星质量 | 实跑生成真实 starmask 测 halo |
| U-16 | Stage2 裁切后 FITS 的 CRPIX 是否同步更新（FITS 标准 WCS） | 2/10 | 超出 Stage10 范围 | 查 Stage2 header 更新逻辑 |

---

## 跨报告冲突说明

经逐份核对，任务给定的交叉校验分（Stage1-3/5-10 各维分数、Stage4 总评 8.2）与 10 份报告正文**完全一致**，无实质矛盾。唯一不一致点：

- **Stage 4 三维度缺失**：任务说明「Stage4 三维度见报告」，但 `stage4.md` 实际仅给出总评 8.2，未单列 D1/D2/D3 分数。本报告评分总表据此标注「见报告（仅总评 8.2）」，加权时按总评计入，未做推断补全。
- 其余各 stage 报告内部 D1/D2/D3 与总评加权自洽（如 Stage7 8.0/7.5/8.5 → 7.98≈8.0；Stage9 9.0/9.0/8.0 → 8.70≈8.67），无数值冲突。

---

## 结论

该 pipeline 在**架构取向与算法专业性**上达到半专业级（D1 均值 7.94、D3 均值 7.67），starless-first 主线与 PixInsight/Siril 社区范式高度对齐。核心短板是**门禁来源可追溯性（D2 均值 7.28）**与**末端降噪对星点的不可逆损伤**，叠加跨 stage 的**死配置 / 可复现性断点**技术债。建议按 §7 路线图，优先处理 P0-1（可复现性三件套）、P0-2（末端降噪 mask+强度受管）、P0-3（降噪反向降级）、P0-4（arcsinh 陷阱）、P0-5（Stage7 review 无硬门），再清理 dead code 与补齐断点完整性。

*本报告为纯研究汇总，未修改被测项目任何代码。所有证据保留 file:line，行业结论保留 URL，未验证项已显式标注。*
