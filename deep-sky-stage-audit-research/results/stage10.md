# Stage 10（最终降噪与导出）深度审计报告

- **审计对象**：`/Users/mz/dev/aiseestart`（Seestar 望远镜离线深空后期 pipeline，Python + Siril 1.4.0）
- **主实现文件**：`pipeline/stages/stage10_export.py`（1333 行，文件名与阶段号一致，无命名错位）
- **审计维度**：D1 逻辑专业合理性 / D2 门禁来源与归档 / D3 算法行业标准符合度
- **审计性质**：纯研究，**未修改任何项目代码**
- **审计日期**：2026-08-05

> 命名错位背景（本仓固有）：Stage 6「去星」实现在 `pipeline/stages/stage7_star_separation.py`，Stage 7「拉伸」实现在 `pipeline/stages/stage6_stretching.py`。Stage 10 无此问题。本报告中「Stage 6/7」一律指**逻辑阶段号**，引用代码时给出真实文件名。

---

## 一、摘要与总评分

### 1.1 总评分

| 维度 | 评分（0-10） | 一句话结论 |
|---|---|---|
| **D1 逻辑专业合理性** | **7.5** | 末端降噪定位（非线性域 + 回星后）与四态覆盖正确，色彩管理链路专业；**致命短板是全图降噪无任何星点保护 mask**，且质量门触发的 review-only 晚于降噪跳过门评估。 |
| **D2 门禁来源与归档** | **7.0** | 阈值集中 `PipelineConfig` + env 覆盖 + 双侧钳制，SHA-256 与原子写规范；但**降噪强度 0.5 硬编码逃逸出配置体系**、**无磁盘空间门**、`task_plan.py` 存在未被引用的 v2 影子 schema。 |
| **D3 算法行业标准符合度** | **7.0** | FITS 科学存档不可变、16-bit sRGB/ICC 受管导出、Astro-TIFF 均符合行业主流；**偏离点在星点保护缺失与降噪强度非自适应**，且未对齐 IVOA Provenance。 |
| **加权总评** | **7.2** | 工程规范性显著高于算法保守性；主要风险集中在「末端降噪对星点的不可逆损伤」与「该损伤无配置化闸门」。 |

### 1.2 一页式结论

Stage 10 的整体架构是**正确的**：它把最终降噪放在拉伸之后、回星之后的非线性域，先做色彩微调（带全局饱和度预算），再按运行时噪声指标动态选择 `full / chroma / separate / skip` 四种降噪模式，然后保存 `stage10_final`、跑最终质量门、生成受管 sRGB 衍生品、导出 TIFF/FITS/PNG，最后写只读色彩清单。数据真实性防线很强：科学存档 FITS 永不被重写，且有 SHA-256 前后比对（`pipeline/managed_output.py:259-372`）。

但在**天文后期专业性**上有一个结构性缺口：末端降噪在 `full` 与 `separate` 模式下对**整幅含星图像**无差别作用，代码中不存在任何星点保护 mask（`pipeline/stages/stage10_export.py` 全文 `mask` 仅出现在 `stage9_starmask_stretch_failed` 状态名与 Stage9 遗留字段中，无实际 mask 应用）。唯一的「保护」是 `chroma` 模式下通过 `_apply_stage10_chroma_only_result` 恢复原始亮度（L347-370），但该保护只覆盖四模式中的一种。行业共识（NoiseXTerminator 手册、AstroBackyard 降噪指南）明确要求末端降噪对高信噪区域（尤其恒星）施加 mask 或降低强度，否则产生「塑料感 / 星点软化」。

在**工程门禁**上有一个显眼的逃逸：降噪强度在两处硬编码为 `0.5`（`pipeline/stages/stage10_export.py:576` 与 `:641`），既不在 `PipelineConfig` 中，也无 `SEESTAR_STAGE10_*` env 覆盖，也无钳制表条目——而本仓其余 Stage10 参数（4 个阈值 + 1 个风险强度 + 1 个开关）全部完整走 config → env → clamp 三段式。这意味着「降噪强度上限」这条最关键的保守性门禁**在配置体系中不存在**，只能依赖 CosmicClarity 脚本内部的 `0..1` 钳制（`CosmicClarity_Denoise.py:253`）兜底。

在**可复现性**上：`pipeline-result.json` + `processing-plan.json` **不足以完整复现** Stage 10。原因是 Stage 10 的降噪模式是运行时指标（`chroma_noise_score` / `bg_std` / `background_mottling_score`）驱动的分支决策，这些指标本身不落在两份 JSON 中；`processing-plan.json` 的 `planned_steps` 只记录计划动作，不记录 Stage 10 分支的输入指标。实际决策证据落在 `final_quality_report.json` 与 stage 记录的 `denoise_plan` 字段里——需要**三份以上**产物才能复现。

---

## 二、D1 逻辑专业合理性审计

### 2.1 末端降噪的定位（非线性域 / 回星后）

| 检查项 | 结论 | 证据 |
|---|---|---|
| 是否在非线性域 | ✅ 是 | 输入候选首选 `stage9_remixed`（Stage9 回星产物，Stage7 已拉伸），`pipeline/stages/stage10_export.py:451-464` |
| 是否在回星后 | ✅ 是 | 候选优先级 `stage9_remixed > input_state_passthrough > starless_enhanced > stage7_stretched`，`:457-464` |
| 是否作用于星点 | ⚠️ **是（无保护）** | `full`/`separate` 模式对整幅含星图执行；仅 `chroma` 模式恢复亮度，`:347-370` |
| 是否有星点保护 mask | ❌ **无** | `stage10_export.py` 全文无 mask 构造/应用；`grep "mask"` 仅命中 Stage9 状态字段名 |
| 强度是否保守 | ⚠️ 部分 | 硬编码 `0.5`（`:576`、`:641`），相对 CosmicClarity 默认 0.5 属中位，但不随场景自适应 |
| 与 Stage5 线性降噪是否叠加过度 | ✅ 有防护 | `should_skip_final_denoise()`，`pipeline/pipeline_safety.py:100-112` |

**双重降噪防护逻辑**（`pipeline/pipeline_safety.py:100-112`）：

```
stage5_denoise_applied == True
  AND stage8_fallback_used == False
  AND stage8_final_quality ∈ {ok, conservative_skipped, star_preserve_bypass}
  → 跳过 Stage10 降噪
```

这条规则在专业上是**合理且保守**的：只要线性域已降噪、Stage8 未走降级路径、Stage8 质量判定良好，就不做第二次降噪。这直接规避了「线性 + 非线性双重降噪导致细节塑料化」这一行业头号陷阱。三个条件是**与**关系，任一不满足即允许末端降噪，符合「有疑问时允许补救」的工程直觉。

**降噪模式选择矩阵**（`_select_stage10_denoise_plan`，`pipeline/stages/stage10_export.py:117-209`）：

| 条件 | 选择模式 | 阈值来源 |
|---|---|---|
| 非彩色图像（channel_semantics ≠ broadband RGB OSC） | `full` | 逻辑分支，L~130 |
| 指标缺失（无 chroma/bg_std/mottling） | `full` | 保守兜底，L~140 |
| `chroma_noise_score ≥ 0.70` | `separate`（逐通道） | `stage10_separate_chroma_score_min`，`models.py:303` |
| `chroma ≥ 0.34` 且（`bg_std ≥ 0.018` 或 `mottling ≥ 0.45`） | `full` | `models.py:302,304,305` |
| `chroma ≥ 0.34` 且 亮度噪声未超标 | `chroma`（只降色度） | `models.py:302` |
| `bg_std ≥ 0.018` 或 `mottling ≥ 0.45`（chroma 未超标） | `full` | `models.py:304,305` |
| 以上皆否 | `skip` | 低噪保护 |

评价：这个矩阵**设计得相当专业**。核心洞察是「色噪 ≠ 亮噪」，色噪单独出现时只处理色度、保留亮度细节（`chroma` 模式），这正是 NoiseXTerminator「color noise separation」的思路。`separate`（逐通道）留给严重通道色噪，`skip` 留给低噪，都合理。**缺陷是**：矩阵完全不考虑星点密度或恒星 FWHM，无论视场里有 50 颗还是 5000 颗星，`full` 模式的行为完全一致。

**`chroma` 模式的实现细节**（`_apply_stage10_chroma_only_result`，`:347-370`）：先用 CosmicClarity 的 `full` 模型跑全图（`cosmic_clarity_mode` 映射 `chroma → full`，L~200），再把原图亮度合并回去（`_chroma_only_denoised_image`）。这是**正确的工程折中**——CosmicClarity CLI 只支持 `luminance/full/separate` 三个模式（`resources/siril_plugins/vendor/siril-scripts/processing/CosmicClarity_Denoise.py:617`），没有原生 chroma-only，所以用「full 后恢复 L」模拟。副作用是计算量翻倍且亮度通道的降噪结果被完全丢弃。

### 2.2 「验收来源」决策矩阵覆盖度

Stage 10 需要面对上游 Stage6（去星四态）× Stage7/8/9（各自 ok/degraded/failed）的组合。实际代码通过**两条正交机制**处理：

**机制 A：图像来源候选链**（`:451-486`）

```
preferred_final_source (= pipeline._stage9_final_source)
  → stage9_remixed
  → input_state_passthrough
  → starless_enhanced
  → pipeline.stretched_name / stage7_stretched
```

逐个 `Path.exists()` + `load` 尝试，失败记 `final_candidate_missing=` / `final_candidate_load_failed=` 消息；全部失败则 `status="degraded"` 并沿用当前 Siril 内存图像（`:480-482`）。若实际加载源 ≠ 首选源，置 `input_source_fallback_used=True`（`:483-485`）。

**机制 B：review-only 触发条件**（`:404-431`）

```
review_only_output =
      cfg.force_review_only_output                    (显式开关, models.py:308)
   OR pipeline._stage9_bypassed_bad_starless          (Stage9 判定无星图劣质而绕过)
   OR stage4_color_review_required                    (Stage4 校色需人工复核)
   OR stage9_missing_required_stars                   (:411-415)
   OR stage9_starmask_stretch_failed                  (:416-418)
```

其中 `stage9_missing_required_stars = _stage9_stars_applied 属性存在 AND _stage9_stars_required AND NOT _stage9_stars_applied`。而 `_stage9_stars_required = not _star_preserve_target_bypass`（`pipeline/stages/stage9_star_remixing.py:858-860`）。

**覆盖度判定**：

| 上游状态组合 | Stage10 行为 | 是否覆盖 |
|---|---|---|
| Stage6 去星成功 + Stage9 回星成功 | 加载 `stage9_remixed`，正常交付 | ✅ |
| Stage6 去星成功 + Stage9 回星失败（stars_required=True） | `stage9_missing_required_stars=True` → review-only | ✅ |
| Stage6 去星成功 + Stage9 星掩膜拉伸失败 | `stage9_starmask_stretch_failed=True` → review-only | ✅ |
| Stage6 去星失败/跳过（star_preserve_bypass） | `stars_required=False`；Stage8 仍以 `starless_enhanced` 命名保存**含星**图（`stage8_nebula_enhancement.py:110,699`），候选链可安全回退 | ✅（但命名语义误导） |
| Stage9 判定无星图劣质并绕过 | `_stage9_bypassed_bad_starless=True` → review-only | ✅ |
| Stage7 拉伸 degraded | 无 Stage10 专门分支；由 Stage7 自身状态汇入全局 partial_success | ⚠️ 间接覆盖 |
| Stage8 增强 failed | 无 `starless_enhanced` 产物 → 候选链退到 `stage7_stretched` | ✅ |
| 全部候选缺失 | `status="degraded"` + 沿用内存图 | ⚠️ 危险兜底（见下） |

**发现 D1-a（中风险）**：`:480-482` 的兜底「沿用当前 Siril 图像」是**未定义状态交付**。此时 Siril 内存中的图像来源不确定（可能是上一阶段残留），却只标 `degraded` → 全局 `partial_success`，仍走**正式交付命名** `result_processed`/`result_final`。专业上，来源不可知的图像不应以正式名交付，应强制 review-only。

**发现 D1-b（低风险，可维护性）**：`starless_enhanced.fit` 在 star-preserve bypass 路径下实际**包含星点**（Stage8 无论是否 bypass 都用该名保存，`stage8_pixels.py:469,1479,1876,1899`）。文件名与内容语义脱节，虽不影响正确性，但让候选链「无星图排在带星的 `stage7_stretched` 之前」这一顺序在阅读时显得可疑。

### 2.3 review-only 模式的触发与产物

**触发点（两处，时序不同）**：

1. **前置触发**（`:425-431`）：进入 Stage10 时依据上游状态判定，立即 `pipeline._final_output_review_only = False` 初始化（`:432`），并在 `:433-448` 打印原因日志。
2. **后置触发**（`:1021-1022`）：`final_quality_report` 返回 `needs_conservative_rerun=True` 时补设 `review_only_output = True`。

**产物差异**（`:1072-1087`）：

```
if review_only_output:
    base_filename    = "result_review"
    (linear 变体)    = "result_review_linear"
    fallback_fit     = "result_review_final"
    pipeline._final_output_review_only = True
    messages += "review_only_output=true; normal result_processed/result_final names withheld"
```

即**正式交付名被完全扣留**，只产出 `result_review*` 系列。全局状态经 `processor_runtime.py:2006` 读取 `_final_output_review_only` 判定为 `review_required`。这个设计是**正确且有力**的——把「不合格产物不许冒充成品」做成了文件名级别的物理隔离，比单纯写状态字段更难被下游误用。

**发现 D1-c（高风险，时序缺陷）**：降噪跳过门在 `:610` 评估 `review_only_denoise_skip = bool(review_only_output)`，此时读取的是**前置**触发结果。而后置触发发生在 `:1021-1022`，即降噪**已经执行完毕之后**。后果：

- 前置 review-only（如 Stage9 缺必需星点）→ 跳过降噪，快速导出 review 图。**符合设计意图**（`:626-627` 注释 "review-only fast export guard"）。
- 后置 review-only（质量门判定需保守重跑）→ 降噪**已执行**，且降噪本身可能正是质量劣化的成因（过度降噪 → artifact_score 升高 → 触发 strict gate）。此时产出的 `result_review*` 是**已被降噪污染**的图，人工复核时无法区分「原图就差」还是「降噪搞坏了」。

正确做法应是：后置触发时保留降噪前快照，或在 review bundle 中同时提供降噪前后对比。当前 `_create_stage_review_bundle`（`:987-1000`）的 context 里虽然带了 `denoise_plan`，但没有降噪前图像。

**发现 D1-d（中风险，门禁空洞）**：`final_quality_report` 整段被包在 `elif stage_saved:`（`:983`）分支内。若 `_save_stage_output("stage10_final")` 失败（`:979-982`），则**最终质量门完全不执行**，`needs_conservative_rerun` 永不为真，`review_only_output` 保持前置值。结果是「保存失败」只降级到 `partial_success`（因为 `status="degraded"`），而不是 `review_required`。保存失败本应是比质量不达标更严重的情况。

### 2.4 位深与色彩管理

存在**两条并行的导出路径**，行为差异明显：

**路径 1：受管输出（managed_output）** —— `pipeline/managed_output.py`，由 `cfg.stage10_managed_output_enabled`（默认 `True`，`models.py:307`）控制，`stage10_export.py:1037-1207` 调用。

| 格式 | 位深 | 色彩管理 | 证据 |
|---|---|---|---|
| PNG | 16-bit | 写 `sRGB` chunk + `gAMA`(45455) + `cHRM` | `managed_output.py:54-117` |
| TIFF | 16-bit little-endian | 嵌入 ICC profile（TIFF tag 34675），要求 profile 有效 sRGB，否则**抛错不生成** | `managed_output.py:142-230` |
| FITS | 原始 | **永不重写**，SHA-256 前后校验，`scientific_archive.policy="never_rewrite"` | `managed_output.py:259-372` |

sRGB profile 从系统路径探测（`managed_output.py:15-20`，macOS `/System/Library/ColorSync/Profiles/sRGB Profile.icc` 等）；找不到则不生成受管 TIFF（fail-closed，正确）。

**路径 2：Siril 原生导出（save_utils）** —— `pipeline/save_utils.py:153-286`，始终执行。

| 格式 | 命令 | 位深 | 色彩管理 | 证据 |
|---|---|---|---|---|
| TIFF | `savetif <base> -astro` | Siril 默认 16-bit | Astro-TIFF（FITS header 存 tag 270），**无 ICC** | `save_utils.py:194,201` |
| FITS | `save <base>` | 原始 | 科学归档，无色彩变换 | `save_utils.py:~225` |
| PNG | `savepng <base>` | 16-bit（当载入图为 16/32-bit） | **无 sRGB/ICC chunk** | `save_utils.py:265,272` |

另有纯 Python 的 `write_png_rgb16`（`save_utils.py:46-79`），写 16-bit big-endian RGB PNG，**同样无任何色彩 chunk**。

**PNG 预览拉伸逻辑**（`save_utils.py:246-263`）：若 `png_preview_stretch=True` 则先跑 `autostretch -linked` 再存；否则明确记录 `"PNG preview uses accepted nonlinear Stage7 rendering; second autostretch skipped"`。**这是专业正确的**——Stage7 已完成拉伸，再叠一次 autostretch 会破坏已验收的影调。

**只读色彩审计**（`build_output_color_manifest`，`pipeline/output_color.py:236-375`）：定义 desired_contract 为 FITS=科学归档无色彩变换、TIF=`editable_16bit` + sRGB ICC、PNG=`display` + sRGB，然后检查实际文件的 PNG sRGB/ICC chunk 与 TIFF ICC tag 34675，产出 `seestar.output-color-manifest.v1`（`stage10_export.py:1217`）。关键属性：`rewrote_outputs=False`——**只审计不修复**。

**发现 D1-e（中风险，契约与实现不一致）**：色彩清单声明的 desired_contract 要求 TIF/PNG 带 sRGB ICC，但 `save_utils` 路径产出的 TIFF/PNG **不满足该契约**。只有 managed 路径的衍生品满足。若 `stage10_managed_output_enabled=False`（可经 `SEESTAR_STAGE10_MANAGED_OUTPUT_ENABLE` 关闭，`processor_runtime.py:256,1170`），则**所有交付文件都不符合自己声明的色彩契约**，而清单只会记录不符，不会阻断交付。契约成了「记录事实」而非「保证事实」。

### 2.5 导出前的不可逆裁剪

**未发现 Stage 10 内有裁剪操作**。`stage10_export.py` 全文无 `crop` 命令调用。裁剪发生在 Stage 2（`stages/stage2_*`），Stage 10 只做色彩微调 + 降噪 + 保存。这是**正确的**——不可逆几何操作应集中在链路早期，末端只做像素级调整。

### 2.6 全局四态一致性

判定逻辑在 `pipeline/processor_runtime.py:2001-2022`，优先级：

```
1. failed          — 任一关键阶段 failed
2. review_required — _input_state_review_route
                   | _final_output_review_only          (Stage10 设置)
                   | _background_review_required
                   | _stage4_color_review_required
                   | (_stage9_stars_required AND NOT _stage9_stars_applied)
3. partial_success — 任一阶段 degraded 或 fallback_used
4. success         — 其余
```

Stage 10 对全局状态的三条输入通道：
- `_final_output_review_only`（`stage10_export.py:1079`）→ `review_required`
- `status="degraded"`（多处，如 `:481, 981, 1024, 1035`）→ `partial_success`
- `denoise_fallback_used=True`（降噪链降级）→ `partial_success`

**一致性评价**：Stage 9 的必需星点条件在 `processor_runtime.py:2010` 与 `stage10_export.py:411-415` **重复实现**了两次。两处逻辑目前一致，但属于典型的「双源真值」隐患——若一处修改而另一处未同步，会出现「Stage10 按正式名交付了，全局状态却是 review_required」或反之的矛盾。

---

## 三、D2 门禁来源与归档审计

### 3.1 Gate 穷举表

| # | Gate 名称 | 位置 | 阈值/条件 | 来源分类 | 钳制 | 落盘位置 |
|---|---|---|---|---|---|---|
| G1 | 色度聚焦门 | `stage10_export.py:22` / `models.py:302` | `chroma_noise_score ≥ 0.34` | PipelineConfig + env `SEESTAR_STAGE10_CHROMA_FOCUS_SCORE_MIN`（`processor_runtime.py:1129`） | `[0.10, 0.80]`（`processor_runtime.py:1312`） | `processing-plan.json` config 段；stage 记录 `denoise_plan` |
| G2 | 逐通道降噪门 | `stage10_export.py:23` / `models.py:303` | `chroma_noise_score ≥ 0.70` | PipelineConfig + env `..._SEPARATE_CHROMA_SCORE_MIN`（`:1130`） | `[0.35, 1.50]`（`:1313`） | 同上 |
| G3 | full 模式亮噪门 | `stage10_export.py:24` / `models.py:304` | `bg_std ≥ 0.018` | PipelineConfig + env `..._FULL_BG_STD_MIN`（`:1131`） | `[0.001, 0.10]`（`:1314`） | 同上 |
| G4 | full 模式斑驳门 | `stage10_export.py:25` / `models.py:305` | `mottling_score ≥ 0.45` | PipelineConfig + env `..._FULL_MOTTLING_SCORE_MIN`（`:1132`） | `[0.10, 1.00]`（`:1315`） | 同上 |
| G5 | **降噪强度** | `stage10_export.py:576, 641` | `strength = 0.5` | ❌ **硬编码**，无 config / 无 env / 无 clamp | 仅依赖 `CosmicClarity_Denoise.py:253` 的 `max(0, min(1, x))` | 仅出现在 CLI 参数日志 |
| G6 | 双重降噪门 | `pipeline_safety.py:100-112` | `stage5_applied AND NOT stage8_fallback AND stage8_quality ∈ {ok, conservative_skipped, star_preserve_bypass}` | 运行时状态传递（上游 Stage5/8） | N/A | stage 记录 `denoise_reason_code="duplicate_denoise_guard"` |
| G7 | 低噪跳过门 | `stage10_export.py:611` | `selected_denoise_mode == "skip"` | 运行时推导（G1-G4 综合） | N/A | `denoise_effective="low-noise input retained"`（`:633`） |
| G8 | review-only 降噪跳过 | `stage10_export.py:610` | `review_only_output == True`（**仅前置值**） | 上游状态传递 | N/A | `denoise_reason_code="review_only_output"`（`:1231`） |
| G9 | 降噪 CLI 超时门 | `stage10_export.py:667` → `cosmic_clarity.py:206-212` | `max(60, min(300, SEESTAR_SIRILPY_TIMEOUT_SEC + 60))`，默认 120+60=180s | env `SEESTAR_SIRILPY_TIMEOUT_SEC` | `[60, 300]` 硬编码 | 降噪失败消息 |
| G10 | SCUNet 二次回退抑制 | `stage10_export.py:373-390` | 主降噪错误含 "timeout"/"timed out" → 跳过 SCUNet | 运行时字符串匹配 | N/A | `_last_scunet_fallback_error` |
| G11 | 饱和度预算门 | `pipeline_safety.py:60-70` + `stage10_export.py:515-519` | `clamp_saturation_boost(requested, already_applied, limits)` | policy + Stage4 报告（`color_safety_limits`） | 全局预算上限 | review bundle `color_policy_limits` |
| G12 | 通道语义色彩门 | `stage10_export.py:497-514` | `channel_semantics == BROADBAND_RGB_OSC` 才允许全局调色 | 运行时推导（Stage1 输入画像） | N/A | messages `Stage10 global color adjustment skipped by channel semantics` |
| G13 | Stage9 局部色风险门 | `stage10_export.py:61-114` | `factor = max(0, 1 - risk_score × strength)`，`strength=1.0` | PipelineConfig `stage10_stage9_local_color_risk_strength`（`models.py:306`）+ env（`:1133`） | 无独立 clamp 条目 | review bundle `stage9_local_color_saturation_guard` |
| G14 | 最终质量门 | `stage8_pixels.py:476-800`，调用于 `stage10_export.py:1008` | 常规：chroma/mottling/patch/artifact 多项；strict：chroma 0.34/0.42、mottling 0.45/0.55、patch 0.00016/0.00022、artifact 0.52/0.62（`stage8_pixels.py:667-669`） | 硬编码于 `stage8_pixels.py` | N/A | **`final_quality_report.json`**（`stage10_export.py:1009`） |
| G15 | strict_gate 触发门 | `stage8_pixels.py:643-653` | `stage8_fallback_used AND NOT conservative_skip` \| `bypassed_bad_starless` \| `stage9_missing_required_stars` \| `starmask_stretch_failed` \| `halo>0.70` \| `compact_halo_limit_exceeded` | 上游状态 + 硬编码 0.70 | N/A | `final_quality_report.json` |
| G16 | 必需星点门 | `stage10_export.py:411-415` + `processor_runtime.py:2010` | `stars_required AND NOT stars_applied` | 上游状态（`stage9_star_remixing.py:858`） | N/A | `pipeline-result.json` `star_separation` 段（`:2127`） |
| G17 | review-only 强制开关 | `models.py:308` / `stage10_export.py:423` | `cfg.force_review_only_output` | PipelineConfig + env `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT`（`processor_runtime.py:1171`） | bool | `processing-plan.json` config 段 |
| G18 | 受管输出开关 | `models.py:307` / `stage10_export.py:1037-1039` | `stage10_managed_output_enabled`（默认 True） | PipelineConfig + env `SEESTAR_STAGE10_MANAGED_OUTPUT_ENABLE`（`:256, 1170`） | bool | `seestar.managed-output.v1`（`:1059, 1174`） |
| G19 | SHA-256 科学存档不变门 | `managed_output.py:272-341` | 导出前后 FITS SHA-256 必须一致 | 硬编码策略 `never_rewrite` | N/A | managed-output manifest |
| G20 | 输出记录 SHA-256 | `run_manifest.py:44-57, 60-107` | 每个交付文件记 path/size/sha256，按 `exported_after` 时间过滤（`stage10_export.py:1199`） | 运行时 | N/A | `pipeline-result.json` `outputs`/`checkpoints` |
| G21 | 清单哈希自校验 | `run_manifest.py:126-133` + `input_discovery.py:391-398` | `manifest_hash == canonical_payload_hash(unsigned)` | 硬编码 | N/A | `pipeline-result.json` `manifest_hash` |
| G22 | 导出格式门 | `models.py:425` | `output_format`：`all` 或逗号分隔 `tif/png/fit` | PipelineConfig | 字符串解析 | `processing-plan.json` config |
| G23 | 色彩契约审计门 | `output_color.py:236-375` | 检查 PNG sRGB/ICC chunk、TIFF tag 34675 | 硬编码 desired_contract | N/A | `seestar.output-color-manifest.v1`（`stage10_export.py:1217`） |
| **G24** | **磁盘空间门** | — | ❌ **不存在** | — | — | — |

### 3.2 关于磁盘空间门的确认

在 `pipeline/` 目录下检索 `disk_usage` / `statvfs` / `free_space` / `shutil.disk`，**零命中**。唯一命中在 `.claude/worktrees/laughing-shannon/gui/seestar_gui_app.py:1574`（GUI 层，非 pipeline 主线，且位于 worktree 内）。

结论：**Stage 10 导出前无任何磁盘空间前置检查**。Stage 10 会写出 FITS + TIFF（16-bit）+ PNG（16-bit）+ 受管 TIFF/PNG 衍生品，对一张 Seestar 拼接图可能是数百 MB 到 GB 级。磁盘满时的表现是 Siril `savetif`/`savepng` 抛异常 → 走 fallback 名再试 → 再失败 → `status="degraded"`（`save_utils.py:209, 235, 280`），即**降级为 partial_success 而非明确报错**。用户看到的是「部分成功」，实际原因是磁盘满，诊断路径很长。

### 3.3 阈值来源分类统计

| 分类 | 数量 | Gate 编号 |
|---|---|---|
| PipelineConfig + env + clamp（完整三段式） | 4 | G1, G2, G3, G4 |
| PipelineConfig + env（bool，无需 clamp） | 3 | G13, G17, G18 |
| PipelineConfig（无 env） | 1 | G22 |
| env only | 1 | G9 |
| 硬编码（Stage10 内） | 1 | **G5（降噪强度）** |
| 硬编码（其他文件） | 4 | G14, G15, G19, G21, G23 |
| 运行时推导 | 3 | G7, G10, G12 |
| 上游状态传递 | 4 | G6, G8, G11, G16 |

**发现 D2-a（高风险）**：G5 是**唯一一个属于 Stage10 自身语义、却完全逃逸配置体系的参数**。对比 G1-G4 都做了完整的 config → env → clamp，G5 的 `0.5` 直接写死在两处：

```
stage10_export.py:576   pipeline._cosmic_clarity_native_denoise_strength_override = "0.5"
stage10_export.py:641   "-denoise_strength", "0.5",
```

两处**字面量重复**，若未来只改其中一处会导致 in-process 路径与 CLI 路径行为分叉。且没有「降噪强度上限」这一保守性闸门——用户无法在不改代码的前提下把末端降噪调弱（这正是天文后期最常见的调参需求）。

**发现 D2-b（低风险）**：G13 的 `stage10_stage9_local_color_risk_strength` 有 config 和 env（`processor_runtime.py:1133`），但**不在钳制表**（`processor_runtime.py:1312-1316` 只有 4 条，均为 G1-G4）。env 传入负数会使 `factor = 1 - risk × negative > 1`，反而**放大**饱和度——与该 guard 的设计意图相反。虽然 `clamp_saturation_boost`（G11）会在下游再钳一次，实际风险被兜住，但 guard 自身缺少输入校验。

### 3.4 降噪执行链与降级路径

`stage10_export.py:617-913` 是一条四级回退链：

```
1. CosmicClarity in-process 脚本  (:646-651, _run_plugin_script_by_path)
   ↓ 失败
2. CosmicClarity CLI subprocess   (:667, timeout=_final_denoise_cli_timeout_sec())
   ↓ 失败
3. native fallback                (:710 附近)
   ↓ 失败
4. SCUNet fallback                (:768, :822, 经 _run_stage10_scunet_fallback 超时抑制)
   ↓ 失败
5. Aberration API                 (默认关闭)
```

每级失败置 `denoise_fallback_used=True` + `denoise_fallback_reason`，最终汇入全局 `partial_success`。链路设计**健壮**，且 G10 的「主降噪超时后不再尝试 SCUNet」是个务实的反超时叠加保护（注释直言 "Avoid another long AI wait"，`:378`）。

### 3.5 归档产物清单

| 产物 | Schema | 写入位置 | 内容要点 |
|---|---|---|---|
| `pipeline-result.json` | `seestar.pipeline-result.v1` | `processor_runtime.py:2038-2174`，原子写 work_dir + process_dir | `status`、`plan_hash`、`input_profile`、`star_separation`、`actual_steps`、`checkpoints`（含 SHA-256）、`outputs`、`manifest_hash` |
| `processing-plan.json` | `seestar.processing-plan.v1` | `processor_runtime.py:1883-1998`，原子写 | `pipeline_contract`、`input`、`channel_semantics`、`target`、`planned_steps`、`candidate_contracts`、`config`（redact_sensitive）、`plan_hash` |
| `final_quality_report.json` | （无显式 schema 字段） | `stage10_export.py:1009` | 最终质量门全部指标 + `needs_conservative_rerun` |
| managed-output manifest | `seestar.managed-output.v1` | `stage10_export.py:1059, 1174` | 受管衍生品清单 + `scientific_archive.policy="never_rewrite"` + SHA-256 |
| output-color manifest | `seestar.output-color-manifest.v1` | `stage10_export.py:1217` | desired vs actual 色彩契约对比，`rewrote_outputs=False` |
| review bundle | — | `stage10_export.py:987-1002` | `final_denoise_skipped`、`color_policy_limits`、`effective_final_saturation`、`channel_semantics`、`stage9_local_color_saturation_guard`、`denoise_plan` |

**原子写实现**（`run_manifest.py:136-161`）：临时文件 → `fsync` → `os.replace`。规范正确，可抵御写入中途断电导致的 JSON 截断。

### 3.6 「仅凭 pipeline-result.json + processing-plan.json 能否复现」判定

**判定：❌ 不能完整复现。**

**能复现的部分**：
- 输入身份与完整性：`processing-plan.json` 的 `input` 段 + `pipeline-result.json` 的 `checkpoints` SHA-256
- 计划路由与步骤：`planned_steps` + `actual_steps` 可比对是否走了计划路径
- 全部 PipelineConfig 参数值：`processing-plan.json` 的 `config` 段（含 G1-G4、G13、G17、G18、G22）
- 交付文件身份：`outputs` 段的 path/size/sha256
- 计划完整性自证：`plan_hash` + `manifest_hash` 双哈希

**不能复现的部分**：

1. **降噪模式的决策输入缺失**。`_select_stage10_denoise_plan` 依赖 `chroma_noise_score`、`bg_std`、`background_mottling_score` 三个运行时指标（由 `_stage10_denoise_input` 在 `:572-589` 取得），这三个值**不写入** `pipeline-result.json` 或 `processing-plan.json`。它们只出现在 stage 记录的 `denoise_plan` 与 `final_quality_report.json` 中。因此无法从两份 JSON 判定为何选了 `chroma` 而非 `full`。

2. **降噪强度不可见**。G5 硬编码 0.5，不在 config 段，两份 JSON 中无任何 `denoise_strength` 字段。

3. **实际执行的降噪后端不可见**。四级回退链走到第几级、`effective_denoise_mode` 是什么，记录在 stage 记录（`:1226-1333`）而非顶层两份 JSON。`pipeline-result.json` 只有 `actual_steps` 名称级信息。

4. **质量门指标缺失**。G14/G15 的判定依据全在 `final_quality_report.json`，两份 JSON 只能看到最终 `status`。

5. **系统环境依赖不可见**。受管 TIFF 是否生成取决于系统 sRGB ICC profile 是否存在（`managed_output.py:15-20`）。换台机器可能静默不产 TIFF，两份 JSON 无此记录（需查 managed-output manifest）。

**结论**：完整复现 Stage 10 至少需要 **5 份产物**：`processing-plan.json` + `pipeline-result.json` + `final_quality_report.json` + managed-output manifest + stage 记录（含 `denoise_plan`）。用户提出的「两份 JSON 复现」标准**不成立**。

### 3.7 关于 processing-plan schema 版本的核查更正

前序审计怀疑存在 v1/v2 版本错位。本轮**核查后更正**：

- `pipeline/processor_runtime.py:1884` 写入 `"schema": "seestar.processing-plan.v1"`
- `pipeline/input_discovery.py:402-406` 校验 `== "seestar.processing-plan.v1"`
- `pipeline/task_plan.py:36` 定义 `PROCESSING_PLAN_SCHEMA = "seestar.processing-plan.v2"`，并提供 `build_processing_plan` / `verify_processing_plan`

全仓检索 `build_processing_plan|verify_processing_plan|PROCESSING_PLAN_SCHEMA`（`*.py`，排除 `release/` 打包副本）：

| 引用方 | 位置 | 性质 |
|---|---|---|
| `tests/test_task_plan.py:51, 156, 161, 171, 185` | 单元测试 | **测试专用** |
| 生产代码（`pipeline/` 任何模块） | — | **零引用**（检索 `import task_plan\|from .task_plan\|task_plan\.` 于 `pipeline/` 无匹配） |

补充确认：`pipeline/task_plan.py` 模块本身**是活的**——`gui/task_intake.py:17, 40` 导入了同模块的 `build_resume_fingerprints`。但 v2 的 `build_processing_plan` / `verify_processing_plan` / `PROCESSING_PLAN_SCHEMA` 三者**仅被测试调用，从未在生产链路执行**。

**修正结论**：实际运行链路 v1 写（`processor_runtime.py:1884`）/ v1 读（`input_discovery.py:403`），**自洽无错位**，不存在运行时 bug。但存在一个更隐蔽的问题——**测试正在验证一份生产环境永不产出的 schema**：

1. `tests/test_task_plan.py` 对 v2 结构做了 hash 篡改、route 篡改、contract 篡改等 4 项校验测试（`:156-185`），全部通过，给出「processing-plan 已被充分测试」的假象；
2. 而生产实际产出的 v1（`processor_runtime.py:1883-1998`）与实际校验的 v1（`input_discovery.py:400-415`）**没有对应的等价测试覆盖**；
3. 两套定义在「计划是否携带 config 段」这一根本问题上不一致：v2 的 `build_processing_plan`（`task_plan.py:254-319`）**不含 config**，v1 实际产物**含 config（redact_sensitive）**。

对 Stage 10 可复现性判定的影响：3.6 节所依赖的「`processing-plan.json` 含全部 PipelineConfig 参数值」这一前提，成立于 v1 实际产物，**不成立于 v2 定义**。若未来切换到 v2，config 段消失，可复现性将进一步劣化。

---

## 四、D3 算法行业标准符合度审计

### 4.1 末端降噪的行业立场

| 行业来源 | 立场要点 | 本项目对标 |
|---|---|---|
| **NoiseXTerminator 2 / AI3 官方手册**<br>https://www.rc-astro.com/noisexterminator-2-ai3-user-manual-pixinsight/ | 线性、非线性域均可用；建议**优先在线性早期**降噪；提供 `intensity` 与 `color separation`（需彩色图）；默认强度 0.85-1.0；明确警告过度去噪产生塑料感（"plastic" look） | ✅ 线性早期降噪在 Stage5；✅ 色噪/亮噪分离（G1-G4 矩阵）符合 color separation 思路；⚠️ 末端强度 0.5 固定，无自适应 |
| **AstroBackyard 降噪指南**<br>https://astrobackyard.com/astrophotography-noise/ | 推荐顺序：线性轻降噪 → 去卷积 → 拉伸 → **可选**背景轻降噪；明确要求对**星点与高信号区域使用 mask** | ✅ 顺序完全一致（Stage5 → Stage7 → Stage10 可选）；❌ **无 mask，这是最明确的偏离** |
| **Seti Astro Cosmic Clarity**<br>https://www.setiastro.com/cosmic-clarity | Denoiser 支持 Full / Luminance 模式 | ✅ 项目使用该工具；⚠️ `chroma` 模式是本项目自创的「full 后恢复 L」组合，非官方模式 |
| **PixInsight TGVDenoise / MLT 社区共识** | 末端降噪普遍配合 range mask 或 star mask，仅作用于背景；强度取小值 | ❌ 偏离 |

**对标判定**：

- **符合**：降噪在流程中的位置（线性早期为主、非线性末端为辅）、色噪与亮噪分离处理、双重降噪防护、低噪时跳过。
- **部分符合**：强度 0.5 属中位偏保守（NXT 默认 0.85-1.0 更激进），但**不随场景变化**是部分符合而非完全符合。
- **偏离**：`full`/`separate` 模式无星点保护 mask，对整幅含星图无差别降噪。
- **自创**：`chroma` 模式（跑 full 模型后用 `_chroma_only_denoised_image` 恢复原始亮度，`stage10_export.py:347-370`）。这个自创方案**方向正确**（等价于只降色度），但计算浪费一半，且未见任何文献验证「full 模型的色度输出 + 原始亮度」在色度边缘处不会产生色偏。**未验证：该组合的色度-亮度一致性未做量化评估。**

### 4.2 行业陷阱规避清单

| # | 行业已知陷阱 | 本项目是否规避 | 证据 |
|---|---|---|---|
| T1 | 线性 + 非线性双重降噪 → 细节塑料化 | ✅ 规避 | `pipeline_safety.py:100-112` |
| T2 | 末端降噪软化星点、削 FWHM | ❌ **未规避** | 无 mask，`stage10_export.py` 全文 |
| T3 | 已拉伸图再叠 autostretch → 影调破坏 | ✅ 规避 | `save_utils.py:258-263` 明确跳过 |
| T4 | PNG 存 8-bit 丢位深 | ✅ 规避 | `managed_output.py:54-117`（16-bit）；Siril `savepng` 在 16/32-bit 载入时输出 16-bit |
| T5 | 导出不嵌 ICC → 跨软件色偏 | ⚠️ 部分规避 | managed 路径 ✅（`managed_output.py:142-230`）；save_utils 路径 ❌ |
| T6 | 重写 FITS 破坏科学数据 | ✅ 规避（做得很好） | `managed_output.py:259-372`，SHA-256 前后校验 + `never_rewrite` |
| T7 | 丢失 WCS / header | ✅ 规避 | FITS 用 Siril `save` 原样保存；TIFF 用 `-astro`（Astro-TIFF 把 FITS header 存 tag 270） |
| T8 | 过度饱和度堆叠 | ✅ 规避 | 全局饱和度预算链 `clamp_saturation_boost`（`pipeline_safety.py:60-70`）+ Stage9 局部色风险压制（`stage10_export.py:61-114`） |
| T9 | 窄带/单色图误用 RGB 调色 | ✅ 规避 | `channel_semantics == BROADBAND_RGB_OSC` 门（`stage10_export.py:497-514`） |
| T10 | 不合格产物冒充成品交付 | ✅ 规避 | review-only 文件名物理隔离（`stage10_export.py:1072-1087`） |
| T11 | 交付文件无完整性校验 | ✅ 规避 | 全部输出记 SHA-256（`run_manifest.py:44-57`） |
| T12 | 磁盘满导致静默截断 | ❌ **未规避** | 无空间门（G24），失败仅降级 |

规避率 9/12（75%），两个未规避项（T2 星点保护、T12 磁盘门）与一个部分项（T5 ICC）。

### 4.3 位深与色彩管理的行业对标

| 行业来源 | 建议 | 本项目 |
|---|---|---|
| **PixInsight 官方论坛（Juan Conejero）**<br>https://pixinsight.com/forum/index.php?threads/colors-are-off-when-saving-as-jpg-png-etc.23882/ | PixInsight 的 PNG 只支持 8-bit；只有非线性图才应存 PNG/JPEG；导出前应转到 sRGB 并**嵌入 ICC**；TIFF/JPEG 应开启 ICC 嵌入 | ✅ 本项目 PNG 做到 16-bit（**优于** PixInsight）；✅ 只在 Stage7 拉伸后（非线性）导出 PNG；⚠️ ICC 嵌入仅 managed 路径 |
| **Siril 命令参考**<br>https://free-astro.org/index.php?title=Siril:Commands | `savetif` 默认 16-bit，`-astro` 产出 Astro-TIFF；`savepng` 在载入图为 16/32-bit 时输出 16-bit | ✅ 用法正确（`save_utils.py:194, 265`） |
| **Astro-TIFF 规范 v1.0（2022-06-21）**<br>https://astro-tiff.sourceforge.io | 把完整 FITS header 序列化进 TIFF tag 270（ImageDescription）；Siril v1.0.0+ 支持 | ✅ 使用 `savetif -astro`，元数据随 TIFF 保留 |

**评价**：色彩管理这块**明显超出社区平均水平**。16-bit PNG + sRGB/gAMA/cHRM chunk 三件套（`managed_output.py:54-117`）比大多数天文后期流程做得细。ICC profile 无效时**拒绝生成 TIFF**（`managed_output.py:142-230`）而非静默降级，属于 fail-closed 的正确设计。

### 4.4 归档与溯源的行业对标

| 行业标准 | 要求 | 本项目 |
|---|---|---|
| **FITS Standard 4.0（IAU FWG 2016-07-22 批准，2018-08-13 语言编辑版）**<br>https://fits.gsfc.nasa.gov/fits_standard.html | 保留 WCS 关键词 `CDi_j`/`CRPIXj`/`CRVALi`/`RADESYS` 等；header 不应被有损修改 | ✅ FITS 通过 Siril `save` 原样写出，`never_rewrite` 策略保证受管流程不触碰（`managed_output.py:259-372`）。**未验证：pipeline 早期阶段（Stage2 裁切）是否同步更新了 CRPIX，Stage10 不负责此事但会继承其结果。** |
| **IVOA Provenance DM 1.0（Recommendation, 2020-04-11）**<br>https://ivoa.net/documents/ProvenanceDM/20200411/REC-ProvenanceDM-1.0-20200411.pdf | Entity / Activity / Agent 三元模型；支持 FAIR 原则的处理溯源 | ⚠️ **概念符合、形式不对齐**。项目的 `actual_steps`（Activity）、`checkpoints`（Entity + hash）、`pipeline_contract`（Agent 契约）在语义上覆盖了三元素，但未使用 IVOA 命名或可导出为 W3C PROV。属于「自创等效方案」。 |

**对标判定**：
- **符合**：FITS 4.0 数据不可变性、SHA-256 完整性链、原子写。
- **部分符合**：溯源信息**齐全但分散在 5 份产物**（见 3.6），不符合 Provenance DM 的单一可查询模型。
- **自创**：`seestar.pipeline-result.v1` / `seestar.processing-plan.v1` / `seestar.managed-output.v1` / `seestar.output-color-manifest.v1` 四套自定义 schema。质量不错（都有版本号、都有哈希自证），但缺少与 IVOA/W3C PROV 的映射说明。

---

## 五、关键发现 TOP 3

### 🔴 发现 1：末端降噪对星点无任何保护 mask（专业性最严重）

**严重度**：高 | **维度**：D1 + D3

`full` 与 `separate` 两种模式（占四种模式中的两种，且是「指标缺失时」与「非彩色图时」的默认兜底，见 `stage10_export.py:117-209`）对**整幅已回星的非线性图像**执行 AI 降噪，强度 0.5，无 mask、无亮度阈值排除、无恒星检测。

四模式中只有 `chroma` 通过 `_apply_stage10_chroma_only_result`（`:347-370`）恢复原始亮度，间接保护了星点亮度结构；`skip` 不做处理。也就是说，**一半的执行路径会直接软化星点**。

这与两条独立行业来源冲突：
- AstroBackyard 明确要求末端降噪对星点/高信号使用 mask（https://astrobackyard.com/astrophotography-noise/）
- NoiseXTerminator 手册警告过度去噪的塑料感（https://www.rc-astro.com/noisexterminator-2-ai3-user-manual-pixinsight/）

更微妙的是：本项目是 **starless-first 链路**——Stage6 已经把星点分离出来了。既然拥有星掩膜（Stage9 用它回星），在 Stage10 复用该掩膜做保护是**架构上几乎零成本**的。当前设计相当于「先精心把星点摘出来单独处理，最后又把它们和背景一起再糊一遍」，是链路设计意图的自我抵消。

**证据**：`pipeline/stages/stage10_export.py:117-209`（模式矩阵）、`:347-370`（唯一的亮度保护）、`:624-913`（执行链，无 mask 参与）、全文 `mask` 检索无实际应用。

---

### 🔴 发现 2：降噪强度 0.5 硬编码，逃逸整个配置与门禁体系

**严重度**：高 | **维度**：D2

Stage 10 的 6 个配置项（4 个阈值 + 1 个风险强度 + 1 个开关）全部走完整的 `PipelineConfig` → `SEESTAR_STAGE10_*` env → 钳制表三段式（`models.py:302-308`、`processor_runtime.py:1129-1133, 1170, 1312-1315`）。唯独**最影响画质的「降噪强度」不在其中**：

```
stage10_export.py:576   pipeline._cosmic_clarity_native_denoise_strength_override = "0.5"
stage10_export.py:641   "-denoise_strength", "0.5",
```

三重问题：
1. **无配置化**：用户想把末端降噪调弱（天文后期最高频的调参需求）必须改代码。
2. **无上限钳制**：用户要求审计的「降噪强度上限门」在本项目中**不存在**。仅靠 `CosmicClarity_Denoise.py:253` 的 `max(0.0, min(1.0, x))` 兜底，而那是第三方脚本的内部实现，不是本项目的门禁。
3. **字面量双写**：in-process 路径（`:576`）与 CLI 路径（`:641`）各写一次 `"0.5"`，改一处漏一处会导致两条路径行为分叉，且该分叉不会被任何测试或断言捕获。

附带影响：因为强度不在 config，它也不进 `processing-plan.json`，直接导致 3.6 节判定的「两份 JSON 不可复现」。

**证据**：`pipeline/stages/stage10_export.py:576, 641`；对比 `pipeline/models.py:302-308`、`pipeline/processor_runtime.py:1129-1133, 1312-1315`；兜底 `resources/siril_plugins/vendor/siril-scripts/processing/CosmicClarity_Denoise.py:253`。

---

### 🟠 发现 3：质量门触发的 review-only 晚于降噪跳过门，导致 review 图已被降噪污染

**严重度**：中高 | **维度**：D1

时序：

```
:610   review_only_denoise_skip = bool(review_only_output)    ← 只读到「前置」触发结果
:624   if skip_final_denoise: ... else: 执行四级降噪链
:979   _save_stage_output("stage10_final")
:1008  final_quality = _final_quality_report("stage10_final")
:1021  if needs_conservative_rerun: review_only_output = True  ← 「后置」触发，降噪已完成
:1072  if review_only_output: base_filename = "result_review"
```

`:626-627` 的 "review-only fast export guard" 注释表明设计意图是「进 review 就别浪费时间降噪」。但该 guard 只覆盖前置触发。后置触发（质量门判定 `needs_conservative_rerun`）时降噪早已执行完毕。

危害在于**归因困难**：strict gate 的判据里包含 `artifact_score`（阈值 0.52/0.62，`stage8_pixels.py:667-669`），而**过度降噪本身就会抬高 artifact_score**。于是可能出现「降噪 → artifact 升高 → 触发 strict gate → 标 review-only」的自激链条，而交付给人工复核的 `result_review*` 是**降噪后**的图。复核者看到的是结果，看不到原因，无法判断该保守重跑还是该关闭末端降噪。

review bundle 的 context 带了 `denoise_plan`（`:998`）但**不含降噪前图像**，无法离线对比。

**关联缺陷（发现 D1-d）**：整个质量门包在 `elif stage_saved:`（`:983`）内。若 `stage10_final` 保存失败，质量门根本不跑，`needs_conservative_rerun` 恒假，保存失败只降级为 `partial_success` 而非 `review_required`——比质量不达标更严重的情况反而得到更宽松的状态。

**证据**：`pipeline/stages/stage10_export.py:610, 624-627, 979-983, 1008, 1021-1022, 1072-1087`；`pipeline/stage8_pixels.py:643-653, 667-669, 794-796`。

---

## 六、改进建议

### P0（应优先处理）

**P0-1｜为 `full`/`separate` 模式接入星点保护 mask**
- 位置：`pipeline/stages/stage10_export.py:624-913`（降噪执行段），mask 来源可复用 Stage9 已有的 `_stage9_last_star_overlay_mask`（`pipeline/stages/stage9_star_remixing.py:864`）
- 方案：降噪后按 `mask` 做加权合并 `out = mask*original + (1-mask)*denoised`，与现有 `_apply_stage10_chroma_only_result`（`:347-370`）的合并模式同构，实现成本低
- 收益：直接消除 T2 陷阱，使 D1/D3 均提升约 1.5 分

**P0-2｜把降噪强度提升为受管配置项**
- 位置：新增 `models.py` 字段（如 `stage10_denoise_strength: float = 0.5`，紧邻 `:302-307`）+ `processor_runtime.py:1129-1133` 加 env 条目 + `:1312-1315` 加钳制条目（建议范围 `[0.0, 0.8]`，上限低于 1.0 以体现末端保守原则）
- 位置：消除 `stage10_export.py:576` 与 `:641` 的字面量双写，改为读同一变量
- 收益：补齐「降噪强度上限门」，同时让强度进入 `processing-plan.json` config 段，改善可复现性

**P0-3｜修正 review-only 时序，或保留降噪前快照**
- 方案 A（推荐）：在 `stage10_export.py:624` 降噪前，若判定有 strict gate 风险（可复用 `stage8_pixels.py:643-653` 的触发条件预判），先保存 `stage10_predenoise` 快照；后置触发 review-only 时把该快照一并放入 review bundle
- 方案 B：把 `final_quality_report` 提前到降噪前跑一次做预判
- 位置：`pipeline/stages/stage10_export.py:610, 624, 987-1000, 1021-1022`

### P1（建议处理）

**P1-1｜补齐导出前磁盘空间门**
- 位置：`pipeline/stages/stage10_export.py:1101` 附近（`export_started_at` 之前）
- 方案：按当前图像尺寸估算所需字节（FITS + TIFF + PNG + 受管衍生品，约 `W*H*3*2*4` 量级），`shutil.disk_usage(process_dir).free` 不足则明确 `failed` 而非事后 `degraded`
- 理由：当前磁盘满表现为 `partial_success`（`save_utils.py:209, 235, 280`），诊断成本高

**P1-2｜修复 `stage_saved` 失败时质量门被跳过的空洞**
- 位置：`pipeline/stages/stage10_export.py:979-983`
- 方案：`if not stage_saved` 分支应直接置 `review_only_output = True`（保存失败 = 交付物不可信，必须人工介入），而不只是 `status="degraded"`

**P1-3｜处理 `task_plan.py` 的 v2 影子实现与测试错配**
- 位置：`pipeline/task_plan.py:36, 254-319, 322-373`；测试 `tests/test_task_plan.py:51, 156-185`
- 现状：`PROCESSING_PLAN_SCHEMA = "seestar.processing-plan.v2"` 及其 `build_processing_plan`/`verify_processing_plan` **仅被单元测试调用，生产链路零引用**；实际写入/校验的是 v1（`processor_runtime.py:1884`、`input_discovery.py:403`），且两套定义在「是否携带 config 段」上不一致
- 风险：测试覆盖率给出虚假信心——被测的 v2 永不产出，实际产出的 v1 缺少等价测试
- 方案：二选一——(a) 让 `processor_runtime.py:1883-1998` 改调 `task_plan.build_processing_plan` 并把 config 段并入 v2，同步 `input_discovery.py:403` 的校验；(b) 删除 v2 三件套与对应测试，改为直接测 v1 产物

**P1-4｜消除必需星点门的双源真值**
- 位置：`pipeline/stages/stage10_export.py:411-415` 与 `pipeline/processor_runtime.py:2010` 重复实现同一判定
- 方案：抽到 `pipeline_safety.py` 单一函数（该文件已是同类守卫的归属地，参见 `:100-112`）

**P1-5｜为 `stage10_stage9_local_color_risk_strength` 补钳制**
- 位置：`pipeline/processor_runtime.py:1312-1315` 钳制表缺该项
- 方案：加 `("stage10_stage9_local_color_risk_strength", 0.0, 2.0)`，防止负值使 guard 反向放大饱和度（`stage10_export.py:103`）

### P2（长期优化）

**P2-1｜色彩契约从「审计」升级为「保证」**
- 位置：`pipeline/output_color.py:236-375`（`rewrote_outputs=False`）
- 现状：当 `stage10_managed_output_enabled=False` 时，所有交付文件都不满足自己声明的 desired_contract，但只记录不阻断
- 方案：契约不符时至少置 `status="degraded"` 并在 messages 明示；或让 managed 输出成为不可关闭的默认行为

**P2-2｜收敛复现所需产物数量**
- 现状：完整复现 Stage10 需 5 份产物（3.6 节）
- 方案：把 `denoise_plan`（含三个决策指标 + selected/effective mode + 实际使用的后端 + strength）合入 `pipeline-result.json` 的 stage 段，使「两份 JSON 复现」成立

**P2-3｜为自定义 schema 提供 IVOA Provenance 映射**
- 位置：四套 `seestar.*.v1` schema
- 方案：补一份映射文档，说明 `actual_steps`→Activity、`checkpoints`→Entity、`pipeline_contract`→Agent，参照 https://ivoa.net/documents/ProvenanceDM/20200411/REC-ProvenanceDM-1.0-20200411.pdf

**P2-4｜验证 `chroma` 自创模式的色度-亮度一致性**
- 位置：`pipeline/stages/stage10_export.py:347-370`
- 现状：**未验证**——「full 模型降噪结果的色度 + 原始亮度」在高对比边缘（星点边缘、星云锐利结构）是否产生色偏，无量化评估
- 方案：构造合成测试图对比 `chroma` 模式与真正的 LAB 空间 chroma-only 降噪

**P2-5｜重命名 `starless_enhanced` 以消除语义误导**
- 位置：`pipeline/stage8_pixels.py:469, 1479, 1876, 1899`、`pipeline/stages/stage8_nebula_enhancement.py:110, 234, 407, 699` 等
- 现状：star-preserve bypass 路径下该文件实际**含星**，名不符实（影响 `stage10_export.py:461` 候选链的可读性）

---

## 七、证据索引

### 7.1 代码证据（按文件）

**`pipeline/stages/stage10_export.py`（1333 行，主实现）**

| 行号 | 内容 |
|---|---|
| 22-25 | 四个降噪阈值常量默认值 |
| 61-114 | `_stage10_stage9_local_color_saturation_guard`，Stage9 局部色风险压制饱和度 |
| 74-93 | 必需星点判定（guard 内） |
| 96-110 | `risk_strength` 读取与 `factor = max(0, 1 - risk × strength)` |
| 117-209 | `_select_stage10_denoise_plan`，四模式决策矩阵 |
| 347-370 | `_apply_stage10_chroma_only_result`，chroma 模式亮度恢复（唯一的星点间接保护） |
| 373-390 | `_run_stage10_scunet_fallback`，超时后抑制二次回退 |
| 393 起 | `run_stage10_export` 主体 |
| 404-418 | Stage9 状态收集（stars_required / stars_applied / starmask_stretch_failed） |
| 423-431 | `review_only_output` 五项或条件 |
| 432-448 | review-only 初始化与原因日志 |
| 451-486 | 最终图像候选链与加载，`input_source_fallback_used` |
| 480-482 | 全候选失败兜底「沿用当前 Siril 图像」（发现 D1-a） |
| 488-519 | 色彩微调：`color_safety_limits` + 通道语义门 + `clamp_saturation_boost` |
| 497-514 | `BROADBAND_RGB_OSC` 通道语义门（T9 规避） |
| 572-589 | `_stage10_denoise_input` 取运行时指标 |
| **576** | **`_cosmic_clarity_native_denoise_strength_override = "0.5"`（硬编码 1/2）** |
| 599-609 | `should_skip_final_denoise` 调用 |
| 610-616 | 三条降噪跳过条件合并（发现 3 时序起点） |
| 624-635 | 跳过分支的三种原因文案 |
| **641** | **`"-denoise_strength", "0.5"`（硬编码 2/2）** |
| 646-651 | CosmicClarity in-process 调用 |
| 667 | CLI 超时门 `_final_denoise_cli_timeout_sec()` |
| 710, 768, 822 | native / SCUNet 回退调用点 |
| 915-961 | chroma 模式合并与 `effective_denoise_mode` 判定 |
| 979-982 | `_save_stage_output("stage10_final")` 与失败降级 |
| 983 | `elif stage_saved:` —— 质量门被包裹的分支（发现 D1-d） |
| 987-1002 | `_create_stage_review_bundle`，context 含 `denoise_plan` |
| 1008-1009 | `_final_quality_report` 调用与 `final_quality_report.json` 写入 |
| 1021-1022 | `needs_conservative_rerun → review_only_output = True`（后置触发） |
| 1023-1031 | `final_quality != "ok"` → degraded |
| 1037-1039 | 受管输出开关读取 |
| 1059, 1174 | `seestar.managed-output.v1` |
| 1072-1087 | review-only 文件命名隔离 |
| 1101-1102 | `export_started_at`（磁盘门应插入处） |
| 1114-1128 | `export_final_outputs` 调用 |
| 1198-1199 | `collect_output_records(exported_after=...)` |
| 1217 | `seestar.output-color-manifest.v1` |
| 1226-1333 | 组件状态汇总与 `_record_stage` |
| 1231 | `denoise_reason_code = "review_only_output"` |
| 1324 | stage 记录 `review_only_output` 字段 |

**`pipeline/pipeline_safety.py`（112 行）**

| 行号 | 内容 |
|---|---|
| 60-70 | `clamp_saturation_boost`，全局饱和度预算钳制 |
| 100-112 | `should_skip_final_denoise`，双重降噪防护（T1 规避） |

**`pipeline/save_utils.py`（286 行）**

| 行号 | 内容 |
|---|---|
| 46-79 | `write_png_rgb16`，16-bit big-endian，**无色彩 chunk** |
| 153-286 | `export_final_outputs` |
| 194, 201 | `savetif -astro` 主/备 |
| 209, 235, 280 | 各格式完全失败 → `status="degraded"` |
| 246-263 | PNG 预览拉伸开关与「跳过二次 autostretch」说明（T3 规避） |
| 265, 272 | `savepng` 主/备 |

**`pipeline/managed_output.py`（381 行）**

| 行号 | 内容 |
|---|---|
| 15-20 | `_SRGB_PROFILE_CANDIDATES` 系统 ICC 路径探测 |
| 54-117 | `write_managed_display_png`，16-bit + sRGB/gAMA(45455)/cHRM |
| 142-230 | `write_managed_edit_tiff`，16-bit LE + ICC tag 34675，无效则抛错 |
| 259-372 | `export_managed_outputs`，SHA-256 前后校验 + `never_rewrite`（T6 规避） |

**`pipeline/output_color.py`（381 行）**

| 行号 | 内容 |
|---|---|
| 236-375 | `build_output_color_manifest`，desired_contract 定义与实际检查，`rewrote_outputs=False` |

**`pipeline/processor_runtime.py`**

| 行号 | 内容 |
|---|---|
| 256 | `SEESTAR_STAGE10_MANAGED_OUTPUT_ENABLE` 声明 |
| 376-418 | `_result_output_basename`，动态命名与元数据完整性检查 |
| 1129-1133 | Stage10 五项 env 覆盖映射 |
| 1170-1171 | 受管输出开关 / `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT` |
| 1312-1315 | Stage10 四项钳制（G13 缺席） |
| 1883-1998 | `processing-plan.json` 构造与原子写 |
| **1884** | **`"schema": "seestar.processing-plan.v1"`** |
| 2001-2022 | `_pipeline_result_status` 四态判定 |
| 2006 | 读取 `_final_output_review_only` |
| 2010 | 必需星点条件（与 stage10 重复实现） |
| 2038-2174 | `_write_pipeline_result_manifest` |
| 2127 | `star_separation` 段写入 |

**`pipeline/run_manifest.py`（232 行）**

| 行号 | 内容 |
|---|---|
| 44-57 | `file_record`：path / size / sha256 |
| 60-107 | `collect_output_records`，按 `exported_after` 过滤 |
| 126-133 | `canonical_payload_hash` |
| 136-161 | `atomic_write_json`：临时文件 + fsync + replace |

**`pipeline/models.py`**

| 行号 | 内容 |
|---|---|
| 300 | `final_saturation = 0.15` |
| 302 | `stage10_chroma_focus_score_min = 0.34` |
| 303 | `stage10_separate_chroma_score_min = 0.70` |
| 304 | `stage10_full_bg_std_min = 0.018` |
| 305 | `stage10_full_mottling_score_min = 0.45` |
| 306 | `stage10_stage9_local_color_risk_strength = 1.0` |
| 307 | `stage10_managed_output_enabled = True` |
| 308 | `force_review_only_output = False` |
| 425 | `output_format = "all"` |

**`pipeline/stage8_pixels.py`（`final_quality_report` 所在）**

| 行号 | 内容 |
|---|---|
| 469, 1479, 1876, 1899 | `_save_stage_output("starless_enhanced")`（bypass 下含星，发现 D1-b） |
| 476-800 | `final_quality_report` 主体 |
| 512 | 读取 `_stage9_stars_required` |
| 643-653 | strict_gate 触发六条件 |
| 667-669 | strict 阈值：chroma 0.34/0.42、mottling 0.45/0.55、patch 0.00016/0.00022、artifact 0.52/0.62 |
| 794-796 | `needs_conservative_rerun` 返回 |

**`pipeline/cosmic_clarity.py`**

| 行号 | 内容 |
|---|---|
| 201, 268 | 超时参数传递 |
| 206-212 | `final_denoise_cli_timeout_sec`：`max(60, min(300, env+60))`，默认 180s |

**`pipeline/stages/stage9_star_remixing.py`**

| 行号 | 内容 |
|---|---|
| 858-860 | `_stage9_stars_required = not _star_preserve_target_bypass` |
| 864 | `_stage9_last_star_overlay_mask`（P0-1 可复用的掩膜） |
| 885 | `source_stem = _stage8_final_source or "starless_enhanced"` |
| 943, 1017 | 特定路径下强制 `stars_required = True` |

**`pipeline/input_discovery.py`**

| 行号 | 内容 |
|---|---|
| 391-398 | `manifest_hash` 自校验 |
| 400-415 | `processing-plan.json` schema **v1** 校验 + `plan_hash` 双向比对 |

**`pipeline/task_plan.py`（v2 影子实现，生产零引用）**

| 行号 | 内容 |
|---|---|
| 36 | `PROCESSING_PLAN_SCHEMA = "seestar.processing-plan.v2"`（仅测试引用） |
| 254-319 | `build_processing_plan`（仅测试引用，**不含 config 段**） |
| 302 | `"schema": PROCESSING_PLAN_SCHEMA` |
| 322-373 | `verify_processing_plan`（仅测试引用） |
| 405-411 | `__all__` 导出三件套 |

**`tests/test_task_plan.py`（测试错配证据）**

| 行号 | 内容 |
|---|---|
| 51 | `task_plan.build_processing_plan(**values)` |
| 156, 161, 171, 185 | `verify_processing_plan` 的 hash / route / contract 篡改校验（**均针对生产不产出的 v2**） |

**`gui/task_intake.py`（确认 task_plan 模块本身是活的）**

| 行号 | 内容 |
|---|---|
| 17, 40 | `from pipeline.task_plan import build_resume_fingerprints`（导入的是**其他**函数，非 v2 三件套） |

**`resources/siril_plugins/vendor/siril-scripts/processing/CosmicClarity_Denoise.py`（第三方）**

| 行号 | 内容 |
|---|---|
| 151-157 | `to_tiff_compatible`，float32 TIFF 交换 |
| 253 | `ds = max(0, min(1, strength)); blended = ds*full + (1-ds)*orig`（唯一的强度钳制） |
| 617 | `choices=["luminance","full","separate"]` |
| 645-647 | 默认 `mode=luminance`、`strength=0.5` |

### 7.2 行业标准与文献证据（URL）

| # | 来源 | URL | 用于论证 |
|---|---|---|---|
| R1 | NoiseXTerminator 2 / AI3 用户手册（RC-Astro 官方） | https://www.rc-astro.com/noisexterminator-2-ai3-user-manual-pixinsight/ | 降噪域选择、intensity/color separation、过度去噪塑料感警告 |
| R2 | AstroBackyard《Astrophotography Noise》指南 | https://astrobackyard.com/astrophotography-noise/ | 降噪流程顺序、**星点/高信号 mask 要求**（发现 1 核心依据） |
| R3 | Seti Astro Cosmic Clarity 官方页 | https://www.setiastro.com/cosmic-clarity | Denoiser 支持 Full/Luminance 模式 |
| R4 | PixInsight 官方论坛（Juan Conejero 答复色彩管理） | https://pixinsight.com/forum/index.php?threads/colors-are-off-when-saving-as-jpg-png-etc.23882/ | PNG 位深限制、非线性图才存 PNG/JPEG、sRGB 转换后嵌 ICC |
| R5 | FITS Standard 4.0（NASA/GSFC 官方，IAU FWG 2016-07-22 批准） | https://fits.gsfc.nasa.gov/fits_standard.html | WCS 关键词、header 不可有损修改 |
| R6 | IVOA Provenance Data Model 1.0（Recommendation 2020-04-11） | https://ivoa.net/documents/ProvenanceDM/20200411/REC-ProvenanceDM-1.0-20200411.pdf | Entity/Activity/Agent 溯源模型、FAIR 原则 |
| R7 | Astro-TIFF 规范 v1.0（2022-06-21） | https://astro-tiff.sourceforge.io | FITS header 存 TIFF tag 270，Siril v1.0.0+ 支持 |
| R8 | Siril 命令参考（free-astro 官方 wiki） | https://free-astro.org/index.php?title=Siril:Commands | `savetif -astro` / `savepng` 位深行为 |

### 7.3 未验证事项清单

| # | 事项 | 未验证原因 |
|---|---|---|
| U1 | `chroma` 自创模式（full 降噪 + 原亮度）在高对比边缘是否产生色偏 | 需构造合成测试图做量化对比，静态代码审计无法判定 |
| U2 | Stage2 裁切后 FITS 的 `CRPIX` 是否同步更新 | 超出 Stage10 审计范围；Stage10 只继承上游 header，不修改 |
| U3 | 受管 TIFF 在无系统 sRGB ICC 的 Linux 环境下的实际行为 | `managed_output.py:15-20` 的候选路径以 macOS 为主，未在 Linux 实测 |
| U4 | strict gate 各阈值（0.34/0.42/0.45/0.55/0.00016/0.00022/0.52/0.62）的标定依据 | 代码与注释均未记录标定数据集或推导过程 |
| U5 | 四级降噪回退链在真实失败场景下的实际触发频率 | 需运行日志统计，本次为静态审计 |
| U6 | `_chroma_only_denoised_image` 的具体色度空间（YCbCr / LAB / 简单比值） | 该函数实现未在本轮读取范围内，仅确认了调用点 `stage10_export.py:357` |

---

## 八、审计方法与边界声明

**方法**：
1. 全量阅读 `pipeline/stages/stage10_export.py`（L1-1333，分 6 段 offset 读取以规避截断）
2. 全量或定向阅读 8 个关联模块：`save_utils.py`、`output_color.py`、`managed_output.py`、`pipeline_safety.py`、`run_manifest.py`、`cosmic_clarity.py`、`models.py`、`processor_runtime.py`（多段）
3. 定向阅读上游状态源：`stage8_pixels.py`（`final_quality_report`）、`stage9_star_remixing.py`（`stars_required` 判定）
4. 第三方脚本核查：`CosmicClarity_Denoise.py`（强度语义与模式枚举）
5. 全仓交叉检索验证：`disk_usage|statvfs|free_space`（磁盘门）、`processing-plan.v`（schema 版本）、`build_processing_plan|PROCESSING_PLAN_SCHEMA` + `import task_plan`（影子实现与引用面确认，含测试目录）、`starless_enhanced`（产物语义）、`mask`（星点保护）
6. D3 联网检索 8 组来源，全部给出可验证 URL

**边界**：
- 本次为**静态代码审计**，未运行 pipeline，未生成实测数据
- 未修改任何项目代码（符合任务约束）
- 评分基于「专业天文后期实践 + 工程可维护性 + 行业标准符合度」三重视角的综合判断，非机械加权
- 所有代码结论均带 `文件:行号`；所有行业结论均带 URL；不确定处已在 7.3 节明确标注

---

*报告结束 · Stage 10 审计 · 总评 7.2 / 10*
