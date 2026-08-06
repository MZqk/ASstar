# Deep Research: Stage 3 背景提取 vs 深空天文行业标准
> Generated 2026-08-05 | Depth: standard | Sources: 22（外部 21 + 内部审计 1）

## TL;DR

Stage 3 当前的过程证据架构（线性确认→源/覆盖掩膜→真实天空样点→拟合/留出拆分→机制五分类→事务性候选+3σ验证门）在**设计与方法论层面符合、且局部领先于**深空背景提取的行业实践共识——机制分流、低阶模型上限、扩展源保护、留出验证与通量保真门都能在 Siril/Photutils/SExtractor/DrizzlePac/LSB 科研论文中找到对应原则，其中"冻结留出验证+自动化通量保真门"是行业交互工具完全没有、科研管线只靠人工做的能力。**结论：方法论上专业。** 唯一实质性短板是**缺少实证标定**（注入真值、分层标注集、shadow-mode 统计），这是"达到行业标准"对外可辩护性与否的分界线；该短板与今日早些时候的内部调研结论一致，且行业头部的 GraXpert AI 同样没有任何公开验证依据。

## Executive Summary

本次调研以项目内 2026-08-05 早间完成的阈值溯源调研（`deep-sky-background-extraction-research/report.md`）为基线，聚焦重构后的过程证据架构与行业主流方法的全面对标。

行业现状的关键事实是：**深空背景提取不存在任何跨工具、跨设备的统一数值阈值标准**。SExtractor 官方文档明示网格与滤波权衡"必须由用户自行找到折中"[8]；LSB 研究的共识是逐案检查加独立数据交叉验证 [9][10]；PixInsight 教材明确指出"样本越多越好"是常见误解 [7]。行业真正遵循的是一份**过程性清单**：线性已校准数据、高质量源掩膜、空间分布良好的纯天空样本、受控的模型复杂度、背景/RMS 残差检查、通量保真验证。

Stage 3 与这份清单逐项对标的结果是：线性前提确认、加性/乘性/方向性机制分流（与 Siril 的 Subtraction/Division 划分及"渐晕归 master-flat"的官方立场一致 [3][4]）、源掩膜与覆盖掩膜分离（与 Photutils `mask`/`coverage_mask` 的官方语义逐字对应 [19]）、低阶模型复杂度上限（与 Siril"高阶过校正"警告 [3] 和 NASIM 刻意采用二阶多项式的保守策略 [10] 一致）、扩展源占满视场时保留基线（正是 DrizzlePac"几乎必然要求关闭天空减除"的场景 [21]）。在验证环节，Stage 3 的冻结留出集+3σ空间跨度改善门+目标通量保真门，把科研管线中由人工执行的注入检查与残差审查自动化了——ABYSS 论文证明"每一种被测天空校正方法都系统性高估真实天空水平"[9]，这类验证不是锦上添花而是必需品，而 Siril/PixInsight/GraXpert 均未提供任何自动化验证机制（GraXpert 官方样点指导文档至今是 TODO 空页 [13]）。

差距集中在实证侧：仓库没有与门限数值关联的注入真值、盲评或 ROC 证据（基线调研已确认）；源掩膜未采用 NoiseChisel 类噪声驱动分割，而两篇 LSB 论文都指出这是天空估计精度的关键 [9][10]；3σ 门所用的 patch 中值采样不确定度是自研估计量，未经外部统计学校验。这些构成后续工程优先级，而非设计缺陷。

## 1. 行业标准的真实形态：过程共识，而非数值阈值 [Confidence: High]

讨论"是否符合行业标准"之前必须先回答"标准是什么"。证据显示，答案不是一组数字，而是一套过程性约束。

**不存在统一阈值。** SExtractor 官方文档在描述背景网格参数时直言："a good compromise must be found by the user"——网格过小会把延伸源通量吸收进背景图，过大则漏掉真实空间变化，没有任何通用合格值 [8]。PixInsight 教学材料同样否定量化教条，指出"一个通常的错误理解是样本数越多越好，更多的样本只会让算法更不准确"，因为"渐变一般是大尺度的"[7]。项目基线调研已在仓库内确认 Stage 3 旧版固定阈值自称"project internal engineering gate"，与行业无映射关系 [23]。

**行业共识是一份过程清单。** 综合官方文档与科研论文，专业背景提取的共同做法是：

1. **在线性、已校准数据上执行。** PixInsight 教材将 DBE 归入"线性处理"章节 [7]；Siril 流程将背景提取置于拉伸与校色之前，并建议先用直方图/假彩色模式目检梯度 [4]。
2. **区分机制。** Siril 官方文档明确 Subtraction 用于加性现象（光害、月光），Division 用于乘性现象（渐晕、大气吸收），且注明乘性校正"should be done by master-flat correction"[3]；官方教程进一步强调光学渐晕"is only perfectly corrected with the application of a master-flat"[4]。PixInsight 译本承认两类梯度可能共存，此时优先减法 [7]。
3. **掩膜质量决定天空纯度。** ABYSS 论文指出"Careful masking based on the noise-based non-parametric algorithms such as NoiseChisel … provides much more accurate estimations of the sky level"[9]；NASIM 用 `minskyfrac=0.9` 强制天空 tile 纯度 [10]；Photutils 在 API 层面把 `mask`（遮源/坏像素）与 `coverage_mask`（仅遮无数据区）做成两个不可混用的参数，明文禁止用 coverage_mask 遮源 [19]。
4. **压低模型复杂度。** Siril 多项式最高 4 阶且警告"a too high degree can give strange results like overcorrection"[3]，官方教程与 FAQ 推荐对单帧用 degree 1 处理时变梯度 [4][5]；NASIM 的残余天空模型刻意只用二阶多项式 [10]；DrizzlePac 对延伸源建议把天空统计改为 mode，无真实天空像素时"几乎必然要求关闭天空减除步骤"，窄带曝光默认关闭天空减除 [21]。
5. **验证扣除是否安全。** SExtractor 输出 BACKGROUND/BACKGROUND_RMS 双图供检查 [8]；ABYSS 用天空真值为零的 Illustris 模拟图像做基准测试，发现"every single tested sky-correction method systematically overestimates the true sky level"[9]；NASIM 用 2MASS/VHS 独立巡天的径向面亮度剖面交叉比对 [10]。过扣的典型症状是大星系外围截断与周围负通量环 [9][10]。

**AI 方法的验证现状值得单独说明。** GraXpert 是当前最流行的 AI 背景提取工具，但通过 arXiv 官方 API 检索 "GraXpert" 返回 0 条结果——没有任何同行评审论文描述其 BGE 模型的训练数据、损失函数或评估协议 [1]；官方 README、文档站、release notes 与模型仓库（CC BY-NC-SA 4.0、经 MinIO/S3 分发 ONNX）同样没有任何训练/验证描述 [12][13][14][15][16]；官方文档的 Sample Selection 页至今是 TODO 占位符（已核实），即连样点放置指导都没有公开 [13]。换言之，在"背景模型的验证依据"这一环，行业头部的 AI 方案同样拿不出公开证据——这不是为项目短板开脱，而是说明"实证标定缺失"是整个业余工具生态的普遍状态，科研管线（ABYSS/NASIM）才是验证方法的真正标杆。

## 2. 逐项对标：Stage 3 机制 vs 行业实践 [Confidence: High]

以下对照基于对 `pipeline/stages/stage3_background_extraction.py`、`pipeline/background_sampling.py` 与 `pipeline/seestar_Superimpose_workflow.md` §5.3 的源码审计，以及 `deep-sky-background-extraction-research/report.md` 的历史结论 [23]。

**线性前提确认 — 对齐。** Stage 3 从 `InputProfile` 确认数据线性，未知/冲突即 fail closed 保留像素。这与行业"背景提取属于线性阶段"的共识完全一致 [4][7]，且比交互式工具更严格：Siril/PixInsight 依赖用户自觉，Stage 3 把它做成了硬前置条件。

**机制分流 — 对齐且更细。** Siril 的二分法（加性减法/乘性除法，渐晕归 flat）[3][4] 在 Stage 3 中扩展为五分类：`additive_low_frequency_gradient`（允许减法）、`multiplicative_vignetting_or_flat_error`（禁止加性扣除、路由 master-flat/标定复核）、`directional_pattern_or_walking_noise`（条纹/walking noise 路由 bias/dark 与重叠加，不交给天空模型）、`target_signal_or_sky_limited`（保留基线+复核）、`no_measurable_low_frequency_gradient`（原样保存）。其中"方向性图样噪声不交给天空模型"这一条，在检索范围内没有找到任何交互式工具显式实现——Siril 仅有 `-dither` 缓解低动态梯度色带的提示 [6]。这是防止"背景模型静默吞掉结构噪声"的关键设计，与 SExtractor"网格过小吸收通量"警告 [8] 属于同一类风险意识。

**掩膜设计 — 逐字对齐 Photutils。** Stage 3 的 `source_mask`（迭代低阶天空面+正残差+星峰+连贯扩展结构生长）与 `coverage_mask`（仅非有限值/重复无数据底值，不把普通暗天空当无覆盖）的语义分离，与 Photutils 官方 docstring 的 `mask` vs `coverage_mask` 定义完全同构——后者明文"It should not be used to mask sources or bad pixels"[19]。SExtractor 同样把源污染作为背景估计的核心问题，用阈值+中值滤波压制亮星导致的局部高估 [8]。差异点：LSB 研究证明 NoiseChisel 类噪声驱动非参掩膜的天空估计精度显著优于简单阈值/结构掩膜 [9][10]，Stage 3 当前的迭代天空面+峰检测方案没有对标这一档。

**样点选取与空间覆盖 — 对齐。** 行业要求样点全图分布、只落天空、避开天体 [3][4]；Photutils 用 `exclude_percentile`（默认 10.0）控制每个 box 的允许污染占比并建议尽量压低 [19]；NASIM 用 minskyfrac=0.9 保证 tile 纯度 [10]。Stage 3 的做法是从未被掩膜污染、有限、非裁剪、低纹理、局部无星峰的 patch 中选点，最低 16 点、3 象限、8/16 网格、横纵 55% 跨度，且代码与文档均明示这些是"运行安全下限，不解释为行业通用天文阈值"[23]——这种诚实标注本身就是专业实践（对照：GraXpert 官方连样点指导页都是 TODO [13]）。每个候选执行前还通过 `set_image_bgsamples(..., recalculate=True)` 审计 Siril 实际消费的样点集合，要求其属于原拟合集且覆盖达标——这层"执行前复核"在 Siril 官方文档中没有对应物（Siril 的 `subsky -existing` 只说明可复用外部样点 [6]）。

**模型复杂度控制 — 对齐。** Stage 3 对大星系、弥散/暗弱结构、暗星云或天空覆盖受限场景把一阶 `subsky 1 -existing` 设为首位且作为复杂度上限，普通场景才允许 RBF。这对应 Siril 的高阶过校正警告 [3]、degree 1 推荐 [4][5]，以及 NASIM 的二阶上限 [10]。RBF 本身的数学（薄板样条核+smoothing 正则化）与 GraXpert 官方 Algorithms 页描述的 RBF（f(x)=Σwᵢφ(‖x−xᵢ‖)+offset，smoothing 为矩阵加 s·I，样点值经 3σ/grow=4 sigma-clipping 取中值）[13] 及 Siril RBF（thin-plate spline φ(|x|)=|x|²log|x| + Smoothing 正则）[3] 同族——Stage 3 通过 Siril `subsky -existing` 执行的正是这一主流方法。

**留出验证与通量保真门 — 领先于工具生态。** 这是 Stage 3 相对行业工具最突出的差异化设计。交互式工具的验证全部是人工的：Siril 依赖用户迭代试算与目检 [3][4]，PixInsight DBE 依赖删除标红坏样点与三维目检 [7]，GraXpert 无任何验证指导 [13]。科研管线的验证也是人工或离线的：ABYSS 用模拟真值基准 [9]，NASIM 用跨巡天剖面比对 [10]。Stage 3 把其中可自动化的部分固化成了硬门：冻结留出 patch 上 P90-P10 空间跨度改善必须超过 before/after 联合三倍采样不确定度且背景 RMS 不得变差；同一留出点拟合 before/after 一阶天空面，在固定目标 mask 上测参考通量、形态相关性、质心位移，通量损失>3σ 拒绝；候选还须复测方向性噪声不新增。需要说明的是，"holdout validation"在天文背景建模中没有检索到术语级先例——最接近的功能等价物是 NASIM 的纯天空 tile 留出 [10]。因此准确表述是：**该设计符合科研验证精神且自动化程度超过所有检索到的工具，但它本身不是既有标准条款，其 3σ 门限的统计有效性尚未经外部校验。**

**扩展源保护 — 对齐 DrizzlePac。** Stage 3 的 `target_signal_or_sky_limited` 机制（目标填满视场或真实天空不足→保留基线要求复核）与 DrizzlePac 官方建议逐字对应："Any observation where no true background (sky) pixels have been observed due to the presence of an extended source … will almost certainly require that the sky subtraction step be turned off"，且窄带曝光默认关闭天空减除 [21]。Siril 文档同样要求样点避开延伸天体、对填满画面的目标降低复杂度 [3]。

**事务性回滚与审计 — 工程层面领先。** 每个候选前重载不可变基线 `stage3_bg_input`，失败/拒绝/切换均回滚，回滚失败即停止搜索，全部证据写入 `background_quality_report.json`。Siril 的对应机制只是"多项式模型从内存中保留的原图反复计算"[3]，PixInsight/GraXpert 无等价审计链。科研管线的可复现性靠文档与版本控制，Stage 3 的逐候选证据报告达到了同等甚至更细的粒度。

**外部回退定位 — 合理。** GraXpert/GXP → GraXpert-BGE → ADBE → DBE → AutoBGE → NOX → VeraLux NOX 的插件链被放在内置候选全部被过程门拒绝之后，且必须通过与内置候选相同的留出/保真门。鉴于 GraXpert-BGE 没有任何公开的模型验证依据 [1][13][16]，把它作为受门禁约束的降级候选而非主路径，是正确的风险排序。

**对标总览：**

| 环节 | 行业实践 | Stage 3 | 评级 |
|---|---|---|---|
| 线性前提 | 线性阶段执行 [4][7] | InputProfile 硬确认，fail closed | 对齐+更严 |
| 机制分流 | 加性/乘性二分，渐晕归 flat [3][4] | 五分类路由，含方向性噪声分流 | 对齐+更细 |
| 掩膜 | mask/coverage 分离 [19]；NoiseChisel 更优 [9][10] | source/coverage 分离；未用噪声驱动分割 | 对齐，掩膜档级有差距 |
| 样点 | 天空纯度+空间覆盖 [3][19] | 低纹理筛选+16点/3象限/55%跨度+执行前复核 | 对齐 |
| 模型复杂度 | 低阶+保守 [3][10][21] | 目标感知一阶上限 | 对齐 |
| 验证门 | 人工目检/离线模拟 [3][9][10] | 自动留出验证+3σ通量保真门 | 领先 |
| 扩展源保护 | 关天空减除/降复杂度 [21][3] | preserve+review_required | 对齐 |
| 回滚与审计 | 无/靠文档 [3] | 事务性基线+逐候选证据报告 | 领先 |
| 实证标定 | 模拟真值/跨巡天比对 [9][10] | 无 | **缺失** |

## 3. 批判性评估：短板、风险与怀疑者视角 [Confidence: Medium]

**实证标定缺失是唯一硬短板。** ABYSS 的核心发现是所有天空校正方法都系统性高估天空 [9]，这意味着任何未经验证的背景门限——无论设计多精巧——都可能系统性偏向过扣或欠扣。Stage 3 的 3σ 门、留出划分比例（30/10）、样点下限（16/3象限/55%）目前都是工程判断而非标定值。基线调研给出的路径（shadow mode 记录建议不改像素、注入梯度真值标定、分层标注集、优先优化精确率而非覆盖率）[23] 与 NASIM/ABYSS 的做法（模拟基准+独立数据交叉验证）[9][10] 方向一致，应维持最高优先级。

**自研统计量的有效性未校验。** 留出验证用"patch 中值的三倍采样不确定度"作为显著性标尺。patch 中值作为天空估计是稳健的（与 SExtractorBackground 的 median 回退逻辑同族 [20]），但"三倍联合不确定度"这一具体倍数的错误率特性没有推导或蒙特卡洛佐证。行业没有现成答案（没有检索到天文背景模型的 holdout 先例），但至少应补 bootstrap/注入实验确认该门在已知梯度幅度下的行为曲线。

**掩膜档级低于 LSB 标杆。** 两篇 Tier 1 论文都把 NoiseChisel 类噪声驱动掩膜作为天空精度的关键 [9][10]。Stage 3 的迭代天空面+峰检测掩膜对明亮星点足够，但对"弱未掩膜源抬高背景"这类 LSB 场景的典型失效模式 [9] 缺少专门防线。考虑到项目目标用户以入门/进阶爱好者为主、目标多为中等尺度 DSO，这一差距对当前产品场景的影响有限，但应记入技术债。

**怀疑者视角。** 最强的反对意见是：这套过程证据架构本质上仍是手工规则的精致化——机制分类器的判据、硬阻断清单、3σ 门都是先验设定，没有数据驱动的校准闭环，因此"专业"只成立在结构上，不成立在证据上。这个批评部分有效，但需要两点限定：其一，交互式行业标准（Siril/PixInsight）的"验证"是人工目检，连先验规则都没有，Stage 3 至少把规则显式化、可审计化了；其二，行业 AI 标杆 GraXpert-BGE 连训练数据都不公开 [1][13]，"拿数据说话"在整个业余工具生态里无人做到。因此公平的结论是：**Stage 3 的方法论严谨度处于行业前列，实证成熟度处于行业普遍水平（即都缺），差距在科研管线一档。**

**证据边界。** 本轮检索未能取回 CloudyNights/PixInsight 论坛的社区批评一手帖（Cloudflare 拦截与接口节流），因此"GraXpert AI 过扣弥散星云"这类流传甚广的说法没有可引用证据，本报告不作断言；已核实的 GraXpert 失败案例仅限官方 issue tracker（插值法无法处理甜甜圈状锐边伪影 #54、反卷积模型条带伪影 #243）[17]。PixInsight DBE/ABE 官方文档已下线，相关表述依赖 Tier 2 教育译本 [7]。

## 4. Action Plan

- [ ] **注入真值标定（最高优先级）**：构造已知幅度的合成梯度注入集，量化留出验证门与 3σ 通量保真门的精确率/召回率曲线，替代工程判断值；先 shadow mode 记录不改像素。
- [ ] **校验自研统计量**：对"patch 中值三倍采样不确定度"做 bootstrap/蒙特卡洛检验，确认其在不同天空覆盖与噪声水平下的错误率特性。
- [ ] **评估 NoiseChisel/SNR 驱动掩膜**：作为 `source_mask` 的候选实现做 A/B，重点测弱未掩膜源场景的背景偏差。
- [ ] **与 GraXpert-BGE 做对照实验**：项目已将其接为受控回退；在同一目标集上对比内置 subsky 候选与 BGE 的留出指标，量化"AI 回退是否值得保留"。
- [ ] **跨工具交叉验证**：选 3-5 个代表性目标（大星系/弥散星云/亮核发射星云），与 Siril 手动 subsky、PixInsight DBE 的人工结果做盲比对。
- [ ] **维持诚实标注**：所有内部下限/门限在文档与报告中继续标注为"运行安全下限/工程门限"，不表述为行业阈值——这是当前实现已具备、且比行业普遍做法更专业的习惯。

## 5. Open Questions & Caveats

- GraXpert-BGE 的训练数据与评估协议完全不公开 [1][13][16]，无法判断其 AI 模型在"过扣弥散结构"上的真实风险；社区批评帖因抓取限制未能取证。
- "holdout validation 用于背景模型"无术语级行业先例，Stage 3 的留出设计属自主创新，其普适性只能靠自身标定数据证明。
- PixInsight 官方 DBE/ABE 文档（function degree、robustification、sample rejection 细节）已下线，本报告相关对比基于 Tier 2 译本，若需对外引用建议另行取证。
- 本次对标是**设计/方法论层面**的，未运行实验基准；"实现行为与文档一致"基于源码审计而非端到端验证。
- ESO/HST 管线的背景 QC 官方文档本轮未命中可靠链接，未纳入证据。

## Methodology

Standard 深度，lead-agent + 5 个子代理（Wave 1 检索 ×3、Wave 2 补漏 ×2、引用核验 ×1）。Wave 1 覆盖 GraXpert/AI 方法、Siril/PixInsight 传统方法、科研管线验证三个领域；质量门检查发现 GraXpert 官方源全 404、Photutils/DrizzlePac 文档 404，Wave 2 定向补漏（GraXpert 真实仓库定位为 github.com/Steffenhir/GraXpert；Photutils 改取 GitHub 源码 docstring）。引用核验 8 条高影响力论断：7 SUPPORTED、1 PARTIAL——PARTIAL 项（GraXpert 文档 TODO 页）已收窄为已核实表述：Sample Selection 页确认为 TODO 占位，Algorithms 页内容全部核实，Interpolation Method/Settings 页未逐一复核故不作断言。大纲相对模板做了证据驱动的调整：将通用"Status Quo/Trends"改为"行业形态/逐项对标"结构，因本课题是合规性评估而非趋势扫描。基线：`deep-sky-background-extraction-research/report.md`（2026-08-05）的阈值溯源结论作为增量起点复用，未重复检索。

## Bibliography

[1] arXiv 官方检索 API — Query: all:"GraXpert"（totalResults=0）— http://export.arxiv.org/api/query?search_query=all:%22GraXpert%22 — 2026-08-05 — Tier: 1
[2] Mac Observatory — The Role of AI in Astrophotography Image Processing: Tools and Controversies — https://www.macobservatory.com/blog/2025/2/8/the-role-of-ai-in-astrophotography-image-processing-tools-and-controversies — 2026-08-05 — Tier: 3
[3] Siril Team — Background Extraction (Siril 1.4 documentation) — https://siril.readthedocs.io/en/stable/processing/background.html — 2026-08-05 — Tier: 1
[4] Cyril Richard — Removing gradients (Siril tutorial) — https://siril.org/tutorials/gradient/ — 2026-08-05 — Tier: 1
[5] Siril Team — FAQ — https://siril.org/faq/ — 2026-08-05 — Tier: 1
[6] Siril Team — Commands Reference: subsky/seqsubsky — https://siril.readthedocs.io/en/stable/Commands.html — 2026-08-05 — Tier: 1
[7] 星空π对（PixInsight 教材中译）— PixInsight入门到精通 第三章 线性处理 第四节 DBE — http://pi.bestxtech.com/03_04/ — 2026-08-05 — Tier: 2
[8] SExtractor 官方文档 — Background estimation — https://sextractor.readthedocs.io/en/latest/Background.html — 2026-08-05 — Tier: 1
[9] Borlaff et al. — The missing light of the Hubble Ultra Deep Field (ABYSS) — A&A 621, A133 — https://www.aanda.org/articles/aa/full_html/2019/01/aa34312-18/aa34312-18.html — 2026-08-05 — Tier: 1 [foundational]
[10] Saremi et al. — Revealing the low surface brightness universe from legacy VISTA data (NASIM) — arXiv:2508.02780 — https://arxiv.org/html/2508.02780v1 — 2026-08-05 — Tier: 1
[11] GraXpert 官网 — https://www.graxpert.com/ — 2026-08-05 — Tier: 1
[12] Steffenhir/GraXpert — README.md — https://github.com/Steffenhir/GraXpert — 2026-08-05 — Tier: 1
[13] Dark-Matters-Astro/graxpert-web — Docs: Algorithms/Sample Selection — https://github.com/Dark-Matters-Astro/graxpert-web/tree/main/content/en/docs — 2026-08-05 — Tier: 1
[14] Dark-Matters-Astro/graxpert-ai-models — README + Releases — https://github.com/Dark-Matters-Astro/graxpert-ai-models — 2026-08-05 — Tier: 1
[15] Steffenhir/GraXpert — graxpert/ai_model_handling.py — https://github.com/Steffenhir/GraXpert/blob/main/graxpert/ai_model_handling.py — 2026-08-05 — Tier: 1
[16] Steffenhir/GraXpert — Releases v3.0.0–3.1.0rc2 — https://github.com/Steffenhir/GraXpert/releases — 2026-08-05 — Tier: 1
[17] GraXpert GitHub Issues #54 / #243 / #57 — https://github.com/Steffenhir/GraXpert/issues — 2026-08-05 — Tier: 3
[18] AstroBin 论坛 — GraXpert 3.0 on macOS 14.4.1 fails — https://www.astrobin.com/forum/c/equipment-forums/steffen-hirtle-graxpert/new-release-30-on-macos-1441-fails/ — 2026-08-05 — Tier: 3 [snippet only]
[19] astropy/photutils — background_2d.py (Background2D docstring) — https://raw.githubusercontent.com/astropy/photutils/main/photutils/background/background_2d.py — 2026-08-05 — Tier: 1
[20] astropy/photutils — core.py (SExtractorBackground docstring) — https://raw.githubusercontent.com/astropy/photutils/main/photutils/background/core.py — 2026-08-05 — Tier: 1
[21] STScI — DrizzlePac Handbook 6.3 Running AstroDrizzle — https://hst-docs.stsci.edu/drizzpac/chapter-6-reprocessing-with-the-drizzlepac-package/6-3-running-astrodrizzle — 2026-08-05 — Tier: 1
[22] 项目内部审计 — 深空天文图像背景提取判定与 Stage 3 阈值依据 — /Users/mz/dev/aiseestart/deep-sky-background-extraction-research/report.md — 2026-08-05 — Tier: 内部

## Source Extracts

### [1] arXiv "GraXpert" 检索 = 0
- **Summary:** arXiv 官方 Atom API 对 "GraXpert" 的 totalResults 为 0。GraXpert/GraXpert-BGE 无预印本或同行评审论文，AI 模型的训练与评估无公开学术依据。
- **Key quotes:** `<opensearch:totalResults>0</opensearch:totalResults>`
- **Source type:** 学术检索 API — Tier: 1

### [2] Mac Observatory AI 工具综述
- **Summary:** GraXpert 被描述为免费开源梯度去除工具，AI 模型"在多样化天文摄影图像上训练"；相对 DBE 自动化更高但"传统方法允许更精细的控制"；用户"仍应自行审查结果"。
- **Key quotes:** "trained on a diverse set of astrophotography images"; "traditional methods allow for precise control"
- **Source type:** vendor 博客 — Tier: 3

### [3] Siril 官方背景提取文档
- **Summary:** 样点按密度自动生成+Grid tolerance（median+N×sigma）排除亮区；RBF（thin-plate φ(|x|)=|x|²log|x|+Smoothing 正则）与多项式（最高 4 阶）；Subtraction=加性（光害/月光），Division=乘性（渐晕），后者"should be done by master-flat correction"；多项式模型从保留原图反复计算。
- **Key quotes:** "a too high degree can give strange results like overcorrection"; "Good results with the RBF algorithm generally require fewer samples than with the polynomial algorithm"
- **Source type:** 官方文档 — Tier: 1

### [4] Siril 官方梯度教程
- **Summary:** 区分外部光梯度与光学渐晕（后者仅 master-flat 完美校正）；样点须全图分布、只落天空、避开亮星；RBF 少量样本即可（示例 8 个）；堆栈梯度复杂时推荐单帧 degree 1 预去除。
- **Key quotes:** "only perfectly corrected with the application of a master-flat"; "polynomial interpolation at the lowest degree (degree 1) is the one we recommend"
- **Source type:** 官方教程 — Tier: 1

### [5] Siril FAQ
- **Summary:** 长曝光/时变梯度推荐对每张单帧去除一阶线性梯度（seqsubsky）；master-flat 完美也无法消除光害/月光/地平线梯度。
- **Key quotes:** "removing a linear (one-degree polynom) gradient from all individual exposures"
- **Source type:** 官方文档 — Tier: 1

### [6] Siril 命令参考 subsky
- **Summary:** `subsky { -rbf | degree } [-dither] [-samples=20] [-tolerance=1.0] [-smooth=0.5] [-existing]`；tolerance 以 MAD 单位定义；-dither 抑制低动态梯度色带；-existing 复用外部样点。
- **Key quotes:** "Tolerance is in MAD units: median + tolerance * mad"
- **Source type:** 官方文档 — Tier: 1

### [7] PixInsight DBE 中文译本
- **Summary:** DBE 归入线性处理；样点尺寸建议 10–16px、每行 5–7 个（渐变是大尺度现象）；星系/光晕内样本必须移除；"样本越多越好"是常见误解；两类梯度共存时优先减法。
- **Key quotes:** "一个通常的错误理解是样本数越多越好"; "渐变(gradients)一般是大尺度的"
- **Source type:** 教育译本 — Tier: 2

### [8] SExtractor 背景估计文档
- **Summary:** 网格内局部估计+双三次样条插值；BACK_SIZE 选择"very important"，过小吸收通量、过大漏变化，折中由用户负责；3×3 中值滤波压制亮星局部高估；输出背景与 RMS 双图。
- **Key quotes:** "The choice of the mesh size BACK_SIZE is very important"; "a good compromise must be found by the user"
- **Source type:** 官方文档 — Tier: 1

### [9] ABYSS / HUDF (A&A 621, A133)
- **Summary:** 标准管线偏袒致密源，最大天体外围被系统性过扣；用天空真值为零的 Illustris 模拟做基准，所有被测天空校正方法均系统性高估天空；推荐 NoiseChisel 类噪声驱动掩膜；激进天空扣除会破坏科学拼接图。
- **Key quotes:** "every single tested sky-correction method systematically overestimates the true sky level"; "the outskirts of the largest objects, which are usually over-subtracted"
- **Source type:** 同行评审论文 — Tier: 1 [foundational]

### [10] NASIM / VISTA LSB (arXiv:2508.02780)
- **Summary:** 每曝光 NoiseChisel 无源 tile 单一背景值（minskyfrac=0.9 保证纯度），叠加后强掩膜+二阶多项式残余校正；激进背景去除导致大星系周围负通量；用 2MASS/VHS 径向剖面交叉验证，达 Ks~27.7 mag/arcsec²。
- **Key quotes:** "Aggressive background removal introduces significant over-subtraction resulting in negative fluxes around large galaxies"; "enforced tile purity by setting minskyfrac to 0.9"
- **Source type:** arXiv 预印本 — Tier: 1

### [11][12] GraXpert 官网与 README
- **Summary:** 免费开源独立软件；真实仓库为 github.com/Steffenhir/GraXpert；两类方法：传统插值（RBF/Splines/Kriging）需手动样点，AI 方法无需用户输入；CLI 支持 -ai_version/-smoothing/-correction/-bg。
- **Key quotes:** "an AI method which does not require any user input"; "-smoothing … ranging from 0.0 (no smoothing) to 1 (maximum smoothing)"
- **Source type:** 官方文档 — Tier: 1

### [13] GraXpert 官方文档源码（Algorithms/Sample Selection）
- **Summary:** Algorithms 页：样点值用 astropy sigma_clipped_stats（3σ、grow=4）迭代剔除后取中值；RBF=Σwᵢφ(‖x−xᵢ‖)+offset，smoothing 为矩阵加 s·I；Kriging 不受 smoothing 影响；Subtraction=原图−模型+mean(模型)。**Sample Selection 页正文仅 "TODO" 占位**（已核实）；Interpolation Method/Settings 页疑似同样占位（未逐一复核）。Algorithms 页在 en 目录下实为德语原文。
- **Key quotes:** "Zusätzlich wird zu der Matrix auf der linken Seite der Summand s·I addiert"; Sample Selection 页 "{{% pageinfo color=\"warning\" %}} TODO {{% /pageinfo %}}"
- **Source type:** 官方文档源码 — Tier: 1

### [14][15][16] GraXpert-BGE 分发机制
- **Summary:** BGE 模型为 ONNX 格式，存 `bge-ai-models/<x.y.z>/model.onnx`；版本列表来自 MinIO/S3（内置只读密钥），不在 GitHub Releases 分发；模型仓库许可 CC BY-NC-SA 4.0；所有官方材料均无训练数据/训练方法/定量评估描述。
- **Key quotes:** "This repository provides GraXpert's ai models for manual installation released under CC BY-NC-SA 4.0"
- **Source type:** 官方仓库/源码 — Tier: 1

### [17] GraXpert 官方 issue 失败案例
- **Summary:** #54（开发团队自提）插值法无法处理"甜甜圈"状锐边伪影；#243 反卷积 AI 模型产生水平/垂直条带伪影（open）；#57 背景提取前必须先裁边去堆栈伪影。未见"过扣弥散星云"的一手投诉。
- **Key quotes:** "'Donut'-shaped image artifacts … cannot easily be removed by GraXperts interpolotation methods right now"
- **Source type:** 官方 issue tracker — Tier: 3

### [18] AstroBin GraXpert 论坛
- **Summary:** GraXpert 3.0 在 macOS 14.4.1 启动失败，与 release notes 的 macOS arm64 ≥13.6/14 要求互证。正文为 JS 渲染未能取证。
- **Source type:** 论坛 — Tier: 3 [snippet only]

### [19] Photutils Background2D 源码 docstring
- **Summary:** 网格 box 内 sigma-clip 统计（默认估计器 SExtractorBackground）→低分辨率背景/RMS mesh→插值全图双输出；`mask` 遮源/坏像素，`coverage_mask` 仅遮无覆盖区且明文禁止遮源，输出处填 fill_value；exclude_percentile 默认 10.0 宜压低。文档页已迁至 /user_guide/background.html（旧 URL 404 原因）。
- **Key quotes:** "It should not be used to mask sources or bad pixels (in that case use ``mask`` instead)"; "exclude_percentile should be kept as low as possible"
- **Source type:** 官方源码 docstring — Tier: 1

### [20] Photutils SExtractorBackground 源码 docstring
- **Summary:** SExtractor 兼容 mode 估计：(2.5×median)−(1.5×mean)，(mean−median)/std>0.3 时退回 median。
- **Key quotes:** "(2.5 * median) - (1.5 * mean)"; "If (mean - median) / std > 0.3 then the median is used instead"
- **Source type:** 官方源码 docstring — Tier: 1

### [21] STScI DrizzlePac 6.3
- **Summary:** 默认天空减除通常够用但会被大延伸源偏置；可改 skystat 为 mode；无真实天空像素时"几乎必然要求关闭天空减除"；窄带曝光 pipeline 默认关闭；自定义 skyuser 值必须各帧一致否则产生人为天空噪声。
- **Key quotes:** "will almost certainly require that the sky subtraction step be turned off"; "Sky subtraction will be turned off in the pipeline when processing any narrow-band exposures"
- **Source type:** 官方文档 — Tier: 1

### [22] 项目内部审计（基线）
- **Summary:** 旧版固定阈值 0.045/0.16/0.08/0.18/0.75 为项目内部工程门禁，无行业标准映射；行业共同做法为线性确认→机制区分→掩膜/覆盖/复杂度/残差/通量保真的过程性清单；推荐过程证据+留出验证+shadow mode 标定路径。
- **Source type:** 内部调研 — Tier: 内部
