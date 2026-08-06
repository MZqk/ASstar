# Stage 2（裁切 / 视场修正）三维度深度审计报告

> 审计对象：`pipeline/stages/stage2_view_correction.py`（686 行）及其配置、归档、调用链
> 技术栈：Python + Siril 1.4.0，starless-first 离线深空后期链路
> 审计性质：**纯研究**，未修改任何项目代码
> 审计日期：2026-08-05

---

## 0. 摘要与总评分

| 维度 | 评分（0–10） | 一句话理由 |
|------|------|------|
| **D1 逻辑专业合理性** | **7.5** | 四边独立裁切 + 中心保护区 + 迭代复检 + 旋转角清理思路专业，但缺 ROI/信号保护、误裁真实暗天区风险未缓解、彩边门禁硬编码 |
| **D2 门禁来源与归档** | **6.0** | 配置已集中化且钳制完善，但 `stage2_color_artifact_max_crop` 已定义却未接线、存在未配置项 `stage2_level_artifact_window`、resume 不重建 crop_report 造成 provenance 断点 |
| **D3 算法行业标准符合度** | **7.0** | 边缘检测/迭代收敛思路与行业实践（Siril/PixInsight/APP）方向一致，但缺 footprint/coverage/weight 证据、未指定 seqapplyreg framing、非对称裁切破坏 mosaic 拼接风险未处理 |

**总体评价**：Stage 2 工程实现成熟度中上，核心裁切的安全护栏（中心保护区、guard_band、迭代收敛、旋转角清理）是亮点；主要短板集中在 **配置接线一致性（D2）** 与 **缺乏数据质量/视场覆盖证据（D3）**，二者叠加会在 resume 与 mosaic 场景下产生可复现性与科学保真度问题。

### 0.1 审计范围与方法

- **范围边界**：仅审计 Stage 2 裁切/视场修正，含其直接调用的 `image_metrics`、`device_geometry`（作为先验来源评估）、配置（`models.py` / `seestar_Superimpose.py` 钳制）与归档（`processor_runtime.py` / `task_workspace.py`）。上游 Stage 1 与下游 Stage 3–4 仅在接口契约层面触及。
- **方法**：静态代码通读 + 行号定位 + 配置/钳制交叉验证 + 调用链回溯 + 联网检索行业基线（Siril / PixInsight / APP / DSS / Astropy / DrizzlePac / Montage）。每条代码结论带 `文件:行号`，每条行业结论带 URL；无法确认处显式标注"未验证"。
- **命名提示**：全仓存在 stage 编号错位——Stage 6 去星在 `stages/stage7_star_separation.py`，Stage 7 拉伸在 `stages/stage6_stretching.py`；本报告严格按"Stage 2 = 裁切"语义，不受文件名干扰。

---

## 1. D1 逻辑专业合理性

### 1.1 Stage 目标与处理链

- **目标**（工作流层）：去除堆叠后黑边与窄幅彩色边缘，避免污染后续背景建模（Stage 3）与去星（Stage 6）。`pipeline/seestar_Superimpose_workflow.md:75-92`、`L300-328`
- **实际处理链**（`run_stage2_view_correction`，`stage2_view_correction.py:507`）：
  1. 初始自动边缘裁切（`is_adaptive=False`）→ L542
  2. 迭代复检（`stage2_adaptive_edge_crop_max_passes` 次，默认 3）→ L572-654
  3. 彩边裁切（`_edge_color_artifact_crop`）→ L657
  4. 保存 `stage2_corrected.fit` → L670
  5. 写 `stage2_crop_report.json` → L682

### 1.2 边界检测统计判据的稳健性

**判据构成**（`_detect_auto_edge_crop`，L68）：
- 背景估计：取中心 25%–75% 矩形区，取分位 0.45 以下像素作背景，中位 `bg_median`、标准差 `bg_std`（L79-89）
- 近黑阈值：`black_threshold = max(0.0015, min(0.018, bg_median*0.50))`（L91）——自适应、随底噪浮动，合理
- 三边证据融合（`_combine_edge_evidence`，L29）：硬近黑（`col_black > black_line_limit`）、软暗台阶（`col_median <= dark_level_limit`）、色偏、亮台阶（`col_level > col_high_limit`）、暗台阶（`col_level < col_low_limit`）五类
- 触发规则：硬近黑直接触发；软证据需至少两类同时出现（L124-136）

**稳健性优点**：
- `black_threshold` 与 `bg_median` 联动，避免固定阈值在亮背景目标（如亮星云）上误触（L91）
- **中心正交带评分**（L106-116）：仅对穿过图像中心的行/列做边缘统计，规避了「上下黑边抬高每一列的近黑比例」导致整图被判定为需大裁切的经典错误——这是一个专业且正确的设计
- 旋转三角清理（L143-173）：逐角用 5×5 近黑 patch 外扩，处理 dithering/registration 旋转后的黑角，符合行业对旋转叠加边缘处理的认知

**稳健性隐患**：
- **纯像素统计无法区分"无数据黑边"与"真实暗天区"**：`is_black = gray <= black_threshold`（L144）把一切低于阈值的像素当黑边。若目标本身含大面积暗星云 / 暗尘埃带（如马头星云暗部、星系际暗区），且这些区域恰好在边缘条带，会被当作黑边裁掉。当前无 footprint / coverage / weight 证据参与判断（见 D3.4）。**未验证**：实际 Seestar 单目标采集是否常出现边缘暗信号占优——需真实样本回测。
- `level_window = 81`（L117）为固定窗口做滚动中值，对极大图（>8000px）窗口占比过小、对极小图占比过大，缺乏与图像尺寸的归一化关系。

### 1.3 四边独立裁切 vs 对称裁切取舍

- 实现为**四边独立裁切**（L138-141 分别求 `left/right/top/bottom`），并由 `_stage2_constrain_crop_to_center`（L255）在中心保护区约束下独立应用。
- **专业合理性**：四边独立裁切能最大限度保留有效视场，这是堆叠后边缘不规则（旋转+抖动）下的正确选择。对称裁切会为迁就最差一边而牺牲信号，行业（PixInsight DynamicCrop、APP Crop 模式）也默认非对称裁切。方向正确。
- **风险（D3 重点）**：四边独立裁切使各帧输出尺寸/原点不同，**破坏 mosaic / 多通道共配准所需的同尺寸约束**（见 D3.4）。本项目为单目标 pipeline，风险可控，但若后续扩展 mosaic 需显式约束。

### 1.4 迭代复检收敛性与最大裁切比例保护

- 复检循环（L572-654）：每 pass 测 `edge_black_ratio`（`_measure_current_features`），若 `<= target`（默认 `stage2_edge_black_target=0.03`，L573 回退值 0.10）则停；否则自适应再裁（`is_adaptive=True`，上限 `stage2_adaptive_edge_crop_max_extra=0.035`，L185-192）
- **收敛保护**：
  - 单 pass 额外裁切上限 3.5%（L186-188）
  - 无改进提前退出：`after.edge_black >= before.edge_black - 0.003` 即停（L618）
  - 中心保护区约束：超过保护区即停（L303-304、L604-608）
  - 终态越界告警：`final.edge_black > target` 标 `degraded`（L649-653）
- **逻辑正确性**：迭代在「已裁图」上重估阈值，由于 `bg_median` 随图变化，阈值基准在变；但 `total_crop` 用累计 `total_left/total_top`（L405-406）记账，最终坐标可还原到原图。**注意**：迭代复检的 `target` 在 L573 回退默认是 `0.10`，而 `crop_report` 记录的是 `0.03`（L531），二者不一致（见 D2.2 注释）——实际以配置值 0.03 为准，回退值仅作缺失保护。

### 1.5 误裁真实暗天区风险

- 见 1.2 隐患。补充：中心保护区（默认保留原图 70% 面积，L222、`models.py:109`）提供了一道"至少保留中心信号"的硬护栏，但若暗信号位于边缘且目标主体偏移中心（偏心构图），保护区仍可能漏保边缘信号。
- **未验证**：在典型 Seestar 深空目标（M31/ M42/ 昴星团等）上，边缘暗区被误裁的比例——需真实样本统计。

### 1.6 裁切后 WCS / 坐标与后续 plate solve 一致性

- Stage 2 **不做 plate solve**（注释明确：天体测量在 Stage 4，`stage2_view_correction.py:511`）
- Stage 4 通过 `pipeline.stage2_crop_report["total_crop"]` 获取累计裁切量用于视场计算（`stage4_color_calibration.py:1434`）
- **一致性风险**：`total_crop` 仅记录四边像素数，**未记录裁切后的 WCS 原点偏移 / 像素尺度变化**。Siril `crop` 命令（`crop x y w h`，L404）会改写图像几何与 header。若 Stage 4 仅用 `total_crop` 估算视场而不重新 plate solve，在偏心裁切下会有视场定位偏差。**未验证**：Stage 4 是否依赖 crop_report 还是重新 plate solve（后者可自纠）。

### 1.7 unknown / nonlinear 输入执行裁切的安全性

- 调用点：默认线性路径（L2495）与 Stage1 resume（L2401）均走 `stage2_view_correction()`；unknown/nonlinear 走 `_activate_input_state_review_route`（L2522）——即非标准输入进入人工/状态审查分支，**不直接裸裁**，安全。
- `_detect_auto_edge_crop` 对图像尺寸有下限保护（`rgb.shape[1|2] < 80` 跳过，L74、L439），对极小图安全。
- **残留风险**：若上游传入 nonlinear（已拉伸）图，背景统计 `bg_median*0.50` 阈值仍按线性假设工作，可能误判。但进入裁切前已有状态审查路由，风险被上游吸收。

### 1.8 迭代复检的度量来源与语义

- 复检循环的判定量 `edge_black_ratio` 来自 `image_metrics.measure_image_features`（`pipeline/image_metrics.py:243-267`）：取图像边缘 `edge_w = max(2, int(min(shape)*0.05))`（即约 5% 宽度的边缘条带），`edge_black_ratio = mean(edge <= threshold)`，阈值与 Stage 2 内部 `black_threshold` 同源于背景估计，结果 clamp 到 [0,1]。
- **语义一致性**：复检用的"边缘近黑比例"与初始检测用的"列/行近黑比例"同源但统计口径不同（5% 边框 vs 中心正交带）。二者一致时收敛快；若旋转导致边缘近黑集中在角而非整边，边框法可能高估 `edge_black_ratio`，使迭代多跑无害的保守 pass——属安全侧偏差。
- **与 target 的关系**：`target = stage2_edge_black_target`（默认 0.03，L573）要求最终边框近黑比例 ≤ 3%。该目标针对"后续阶段不再临时裁黑边"的工作流承诺（`models.py:104` 注释），与背景拟合污染防护目标一致。

---

## 2. D2 门禁来源与归档

### 2.1 Gate 全量清单与来源分类

| Gate / 阈值 | 位置 | 来源分类 | 上下限钳制 |
|------|------|------|------|
| `black_threshold = max(0.0015, min(0.018, bg_median*0.50))` | L91 | 运行时推导（基于背景） | 硬编码 [0.0015, 0.018] |
| `black_line_limit = max(0.015, min(0.08, target*0.35))` | L99 | 运行时推导 + 配置 | 硬编码 [0.015, 0.08] |
| `dark_level_limit = max(black_threshold*1.25, bg_median - max(bg_std*2.0, 0.004))` | L100 | 运行时推导 | 软下限 |
| `cast_limit = max(0.010, center_cast*2.8)` | L101 | 运行时推导 | 下限 0.010 |
| `max_scan_x/y = max(4, int(dim*0.25))` | L102-103 | 运行时推导（图像尺寸） | 下限 4 |
| `stable_run = max(6, int(min*0.006))` | L104 | 运行时推导 | 下限 6 |
| `level_window = getattr(cfg,"stage2_level_artifact_window", 81)` | L117 | **⚠ 未集中配置**，仅回退默认 81 | 无钳制 |
| `guard_band = getattr(cfg,"stage2_guard_band_pixels", 3)` | L175 | `PipelineConfig`（L107）+ CLAMP [0,8] | 有钳制 |
| `max_extra_ratio = getattr(cfg,"stage2_adaptive_edge_crop_max_extra", 0.035)` | L186 | `PipelineConfig`（L106）+ CLAMP | 有钳制 |
| `center_protect` 70% | L222 / `models.py:109` | `PipelineConfig` + CLAMP [0.50,0.95] | 有钳制 |
| 彩边 `strip_w = max(8, int(width*0.025))` | L442 | 运行时推导 | 下限 8 |
| 彩边 `cast > max(0.010, center_cast*2.6)` | L459 | 运行时推导 | 下限 0.010 |
| 彩边 **剩余尺寸门 `crop_w <= width*0.90`** | L469 | **⚠ 硬编码 90%** | 无（应为 `stage2_color_artifact_max_crop`） |
| 迭代 `target = stage2_edge_black_target` | L573 | `PipelineConfig`（L104）+ CLAMP [0.03,0.18] | 有钳制 |
| 迭代 `max_passes` | L574 | `PipelineConfig`（L105）+ CLAMP [0,6] | 有钳制 |

### 2.2 来源分类汇总

- **集中化配置（PipelineConfig，`models.py`）**：`stage2_edge_black_target`、`stage2_adaptive_edge_crop_max_passes`、`stage2_adaptive_edge_crop_max_extra`、`stage2_guard_band_pixels`、`stage2_center_protect_area_ratio`、`crop_margin`（L103-109）。全部经 `seestar_Superimpose.py:303-446` AUTO_CLAMP_FIELDS 与 `L448-455` CLAMP_RULES 统一钳制，设计良好。
- **hardcoded 常量**：`black_threshold` 上下限、`black_line_limit` 上下限、`stable_run` 下限、`level_window` 默认值、`cast_limit` 下限——这些是算法内部安全护栏，硬编码可接受。
- **运行时推导**：背景估计类阈值（L91-104、L442-459）随图变化，专业合理。
- **设备几何先验（device_geometry.py）**：`build_device_geometry_report`（L169）从 FITS header / known_profile / env / legacy 多源解析，**纯 report_only（applied=False）**，`activate_device_geometry_report` 默认不 apply（enabled 默认 False，L422）。结论：**device_geometry 先验不直接驱动 Stage 2 裁切**，仅作 Stage 4 几何参考。因此 Stage 2 不依赖设备几何先验，无相关 archive 断点。

### 2.3 配置接线缺陷（关键）

- **`stage2_color_artifact_max_crop`（models.py:108，默认 0.15）已定义、已进 CLAMP [0.05,0.25]（seestar_Superimpose.py:454），但 `stage2_view_correction.py` 中从未读取**。彩边裁切实际用**硬编码 90% 剩余尺寸门**（L469）兜底。`grep "stage2_color_artifact_max_crop" pipeline/` 仅命中 models.py / seestar_Superimpose.py，**stage2 代码 0 命中**。→ 用户调该参数无效，属死配置。上游调研 `radiometric_fallback_detection.json:181`、`crop_decision_roi_stop.json:185` 已记录。
- **`stage2_level_artifact_window`（默认 81，L117）未在任何 PipelineConfig / CLAMP_RULES 定义**，仅以 `getattr` 回退值存在。用户无法配置、无法钳制、无上下限保护。上游 `current_stage2_gap_recommendations.json:188` 已记录。
- **`crop_margin`（models.py:103，默认 0.02）残留但 Stage 2 不读取**——无预裁逻辑使用它；属历史遗留/被新自适应算法取代但未清理。上游 `current_stage2_gap_recommendations.json:192` 已记录。

### 2.4 归档完整性

- **写入产物**：
  - `stage2_crop_report.json`（L682，经 `_write_stage_json`）：含 `original_shape / current_shape / final_shape / total_crop / crops[] / crop_limit_hits / target_edge_black_ratio / center_protection / status / messages`——**判定结果与裁切量记录完整**
  - `stage2_corrected.fit`：裁切后图像（L670，`_save_stage_output`）
  - `processing-plan.json`（`processor_runtime.py:1883-1998`）：含脱敏 config 与 planned_steps（Stage 2 计划动作见 `_planned_stage_actions` L1779）
  - `pipeline-result.json`（`processor_runtime.py:2038-2174`）：含 checkpoints、actual_steps，Stage 2 作为 checkpoint 记录（L2075）
  - `checkpoint-manifest.v1`（`task_workspace.py:853` build_checkpoint_manifest，schema L33）、`run-manifest.json`（L36）
  - 日志：`stage_start / stage_end / _record_stage`（L684-685）记录 status 与 messages
- **可复现性评估**：
  - 仅凭归档**可还原裁切矩形（total_crop + crops[]）与触发原因（messages / crop_limit_hits）**——决策透明度好。
  - **断点（provenance gap）**：resume 路径 `_prepare_stage2_corrected_resume_input`（seestar_Superimpose.py:1653）仅复制并 `load stage2_corrected.fit`（L1697-1700），**不读取/重建 `stage2_crop_report.json`**。若从 Stage 2 恢复，下游 Stage 4 依赖的 `pipeline.stage2_crop_report` 为空，视场计算可能退化。上游 `current_stage2_gap_recommendations.json:132` 已记录。→ 仅凭 resume 归档**不能**完全复现裁切决策。
  - **坐标基准缺失**：crop_report 记像素矩形但未记 WCS 原点偏移（见 D1.6），纯像素不足以在天文坐标框架内复现。
  - **复现性清单**：① 裁切矩形 ✅（total_crop + crops[]）；② 触发原因 ✅（messages / crop_limit_hits）；③ 配置快照 ✅（processing-plan.json 脱敏 config）；④ WCS 偏移 ❌（缺字段）；⑤ resume 后元数据 ❌（不重建 crop_report）。四项通过、两项缺口，总体属"可部分复现"。
  - **优先级指向**：上述两项缺口对应的修复即本报告 P0 建议（resume 重建 crop_report + WCS 偏移字段），属高性价比的可复现性补强。

### 2.5 device_geometry 先验归档

- `device_geometry.py` 产出 report 但不 apply（L422），其结论写入独立报告而非 stage2_crop_report。Stage 2 不读取它，故无"几何先验驱动裁切"的归档需求，也无断点。该职责划分清晰。

---

## 3. D3 算法行业标准符合度

### 3.1 行业基线（联网检索，附 URL）

- **Siril（本项目底层）**：
  - `crop` 命令语法 `crop [x y width height]`，无 autocrop 子命令；Rotate&Crop 旋转后缺区填黑（插值 clamping）— https://siril.readthedocs.io/en/latest/processing/geometry.html
  - seqapplyreg **framing** 选项：`current`(默认) / `min`(公共区) / `max`(包围盒) / `cog`(质心) — https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
  - 主界面 CFA 裁切约束（偶数坐标等）— https://siril.readthedocs.io/en/latest/GUI/main-interface.html
  - Commands 参考 — https://siril.readthedocs.io/en/stable/commands.html
- **PixInsight**：DynamicCrop 用于移除 stacking 边缘伪影（典型每边 ~70px），ImageIntegration 输出 rejection map；M31-Ha 示例展示边缘清理流程 — https://www.pixinsight.com/examples/M31-Ha/ ；大 mosaic 用 StarAlignment — https://pixinsight.com/forum/index.php?threads/building-large-mosaics-with-staralignment.1953/
- **Astro Pixel Processor (APP)**：Integrate 的 Full / Reference / Crop 三种合成模式；多通道共配准要求同尺寸 — https://www.astropixelprocessor.com/community/faq/rgb-combination-and-same-size-requirements/ ；mosaic 裁剪讨论 — https://www.astropixelprocessor.com/community/main-forum/cropping-a-mosaic/
- **DeepSkyStacker**：边缘/黑边处理在 ResultParameters 实现 — https://github.com/deepskystacker/DSS/blob/master/DeepSkyStacker/ResultParameters.cpp ；FAQ — https://deepskystacker.free.fr/english/faq.htm
- **Astropy reproject / DrizzlePac / Montage**（科学级 mosaic，强调 footprint / coverage / weight）：
  - footprint arrays — https://reproject.readthedocs.io/en/stable/footprints.html
  - mosaicking — https://reproject.readthedocs.io/en/stable/mosaicking.html
  - DrizzlePac Handbook — https://hst-docs.stsci.edu/drizzpac
  - Montage components — https://irsa.ipac.caltech.edu/Montage/docs/components.html

### 3.2 算法对标

| 本实现特性 | 行业基线 | 符合度 |
|------|------|------|
| 四边独立自动检测黑边 | PixInsight DynamicCrop / APP Crop / Siril 手动 crop | ✅ 符合 |
| 中心保护区防过裁 | PixInsight DynamicCrop 手动留边、APP Reference 模式 | ✅ 符合（自创自动版） |
| 迭代收敛 + 无改进退出 | 无标准工具显式迭代，但 DynamicCrop 交互式预览等价 | 🟡 部分符合（自创自动化） |
| 旋转角清理 | Siril Rotate&Crop 填黑、StarAlignment 处理旋转 | ✅ 符合 |
| guard_band 残余检查 | 无显式行业对应（属插值边界保护） | 🟡 自创，无通用证据 |
| 彩边色偏裁切 | 行业多依赖 CFA/去马赛克阶段，非独立彩边裁 | 🟡 部分符合（自创） |
| footprint/coverage/weight 证据 | Astropy/DrizzlePac/Montage 核心机制 | ❌ 偏离（纯像素统计） |
| 指定 seqapplyreg framing | Siril 支持 min/cog 以统一边缘 | ❌ 偏离（用默认 current） |

### 3.3 符合 / 偏离 / 自创小结

- **符合**：边缘自动检测、中心保护、旋转角清理方向正确。
- **部分符合 / 自创**：迭代复检（自动化创新，无对照基线）、彩边裁切（行业多在更早阶段处理）、guard_band（合理但无文献支撑）。
- **偏离**：
  1. **缺 footprint / coverage / weight 证据**：纯像素近黑统计无法区分"无数据"与"暗信号"，而科学级工具（Astropy reproject、DrizzlePac、Montage）均以权重图/覆盖图作为边缘判定核心。这是与行业最高标准的主要差距。
  2. **未指定 seqapplyreg framing**：`stage_support.py:1521` 仅 `seqapplyreg -filter-round=2.5k`，用默认 `current` framing。行业为获得规则边缘常用 `min`（公共区）或 `cog`。当前在 Stage 2 才修边缘，等于把 registration 本可统一的边缘问题后置，且四边独立裁切进一步放大各帧差异。

### 3.5 与上游调研结论的一致性

- 本审计的 D2/D3 关键结论与 `deep-sky-edge-cropping-research/` 上游调研相互印证：`stage2_color_artifact_max_crop` 未接线（`radiometric_fallback_detection.json:181`、`crop_decision_roi_stop.json:185`）、`stage2_level_artifact_window` 未正式配置（`current_stage2_gap_recommendations.json:188`）、`crop_margin` 残留不驱动（`current_stage2_gap_recommendations.json:192`）、resume 不重建 crop_report（`current_stage2_gap_recommendations.json:132`）、固定 3px guard 无通用行业证据（`interpolation_downstream_guard_band.json`）。
- 上游 `software_pipeline_practices.json` 提供的 19 个行业文档 URL 已收入本报告的"行业证据"索引（§6），作为 D3 对标的一手来源。
- **增量发现**：本审计额外定位了硬编码 90% 门的精确行号（L469）与 `getattr` 回退默认不一致（L531 vs L573/L98），并明确了 `device_geometry` 为 report_only、不直接驱动 Stage 2 裁切，补充了上游未细化的调用链结论。

### 3.4 行业已知陷阱规避情况

| 陷阱 | 是否被规避 | 说明 |
|------|------|------|
| 裁太狠丢信号 | ✅ 已规避 | 中心保护区 70% + adaptive 上限 3.5% + 无改进退出 |
| 裁太少污染背景拟合 | 🟡 部分 | 迭代把 edge_black 压到 target 0.03，但纯像素阈值可能残留暗台阶（软证据需两类同现，单类暗台阶可能漏） |
| 非对称裁切破坏 mosaic 拼接 | ❌ 未处理 | 四边独立裁切使各帧尺寸/原点不同；单目标 OK，mosaic 扩展需约束（APP 要求同尺寸） |
| 暗天区被误裁 | ❌ 未处理 | 无 footprint 区分，见 D1.2/D1.5 |
| CFA 偶数坐标 | 🟡 未验证 | Siril 文档主界面提 CFA 裁切约束（L 上文 URL），本项目 `crop` 命令未显式强制偶数坐标；未验证 Siril 1.4.0 是否内部处理 |
| 插值边界伪影（旋转填黑） | ✅ 规避 | guard_band 3px 清残余 + 旋转角清理 |
| 配置失效/死参数 | ❌ 存在 | `stage2_color_artifact_max_crop` 未接线（D2.3） |

---

## 4. 关键发现 TOP3

1. **【D2 配置断线】`stage2_color_artifact_max_crop` 已定义且已钳制，但 Stage 2 代码从未读取，彩边裁切实际由硬编码 90% 剩余尺寸门（L469）兜底。** 用户调参无效，属死配置；同时使彩边裁切量丧失上限可配置性（90% 门意味着最多裁 10%，比配置 15% 更保守但不可调）。
2. **【D3 科学保真】纯像素近黑统计无 footprint/coverage/weight 证据，无法区分"无数据黑边"与"真实暗天区"。** 在偏心构图或含边缘暗信号的目标上可能误裁科学信号；行业科学级工具以权重图为核心判据，本项目缺失该机制。
3. **【D2 provenance 断点】resume 路径（`seestar_Superimpose.py:1653`）只复制 `stage2_corrected.fit` 而不重建 `stage2_crop_report.json`，导致下游 Stage 4 视场计算依赖的裁切元数据在恢复后缺失。** 仅凭归档无法完整复现裁切决策。

---

## 5. 改进建议（P0/P1/P2，指明文件位置）

### P0（正确性 / 数据完整性，必修）
- **接线 `stage2_color_artifact_max_crop`**：在 `stage2_view_correction.py:_edge_color_artifact_crop`（L469）将硬编码 `width*0.90` 改为读取 `getattr(pipeline.cfg,"stage2_color_artifact_max_crop",0.15)`，使配置生效并与 CLAMP [0.05,0.25] 一致。
- **resume 重建 crop_report**：在 `seestar_Superimpose.py:_prepare_stage2_corrected_resume_input`（L1653）中，复制 `stage2_corrected.fit` 的同时加载/重建 `stage2_crop_report.json` 并赋值 `pipeline.stage2_crop_report`，消除 provenance 断点。

### P1（稳健性 / 行业对齐，建议修）
- **引入 footprint/weight 证据区分暗边与暗信号**：在 `_detect_auto_edge_crop`（L68-173）增加基于 stacking weight / 覆盖计数的判定分支（参考 Astropy reproject footprint — https://reproject.readthedocs.io/en/stable/footprints.html），对"低覆盖但非黑"的条带降级为软证据或不裁。
- **将 `stage2_level_artifact_window` 提升为正式配置**：在 `models.py`（~L109 后）增加字段并加入 `seestar_Superimpose.py` AUTO_CLAMP_FIELDS / CLAMP_RULES（参考 L303-454），移除 L117 的裸 `getattr` 回退。
- **指定 seqapplyreg framing**：在 `stage_support.py:1521` 增加 `-framing=min`（或 `cog`）以获得更规则公共区，减少 Stage 2 后置修边压力（与 Siril framing 文档对齐 — https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html）。

### P2（可维护性 / 清理，可选）
- **清理 `crop_margin` 残留**（`models.py:103`）：确认无人使用后移除或显式注释"已被自适应裁切取代"，避免误导。
- **统一 target 回退默认值**：`L531` 用 0.03、`L573`/`L98` 用 0.10 作为 `getattr` 回退不一致；建议统一回退为配置默认值 0.03，避免配置字段缺失时行为漂移。
- **裁切后 WCS 原点记录**：在 `stage2_crop_report.json` 增加 WCS 原点偏移字段，供 Stage 4 精确视场计算（D1.6）。
- **mosaic 兼容性预留**：若未来支持多目标拼接，在 `_stage2_constrain_crop_to_center` 增加"同尺寸/同原点"约束开关（对齐 APP 同尺寸要求 — https://www.astropixelprocessor.com/community/faq/rgb-combination-and-same-size-requirements/）。

---

## 6. 证据索引

### 代码证据（file:line）
- `pipeline/stages/stage2_view_correction.py:68` — `_detect_auto_edge_crop` 入口
- `…:91` — `black_threshold = max(0.0015, min(0.018, bg_median*0.50))`
- `…:99-104` — 各 limit / max_scan / stable_run
- `…:106-116` — 中心正交带评分（防上下黑边污染全列）
- `…:117` — `stage2_level_artifact_window` 未配置仅回退 81
- `…:138-173` — 四边独立检测 + 旋转角清理
- `…:175` — `guard_band` 默认 3
- `…:185-192` — adaptive 上限 `max_extra_ratio=0.035`
- `…:222-252` — `_stage2_center_protection` 默认 70%
- `…:255-353` — `_stage2_constrain_crop_to_center`
- `…:373-431` — `_stage2_apply_crop` 累计 total_crop
- `…:433-493` — `_edge_color_artifact_crop`，**L469 硬编码 90% 门**
- `…:496-504` — `_edge_color_cast_score`
- `…:507-686` — `run_stage2_view_correction` 全流程
- `…:531` vs `…:573` — target 回退默认值不一致（0.03 / 0.10）
- `…:682` — 写 `stage2_crop_report.json`
- `pipeline/models.py:103-109` — Stage 2 配置字段（含未接线 `stage2_color_artifact_max_crop` L108）
- `pipeline/seestar_Superimpose.py:303-455` — AUTO_CLAMP_FIELDS / CLAMP_RULES（stage2 参数钳制）
- `pipeline/seestar_Superimpose.py:1653` — `_prepare_stage2_corrected_resume_input`（不重建 crop_report）
- `pipeline/seestar_Superimpose.py:2401,2495,2522` — 调用点 / unknown-nonlinear 路由
- `pipeline/stage_support.py:1521` — `seqapplyreg -filter-round=2.5k`（未指定 framing）
- `pipeline/processor_runtime.py:1883-1998` — `processing-plan.json`；`2038-2174` — `pipeline-result.json`；`L1779` — planned actions
- `pipeline/image_metrics.py:243-267` — `measure_image_features` / `edge_black_ratio`
- `pipeline/device_geometry.py:169,422` — report_only，不 apply，不驱动 Stage 2
- `pipeline/stage4_color_calibration.py:1434` — 读取 `stage2_crop_report` 做视场计算
- `pipeline/stage_contracts.py:76-84` — Stage 2 contract（formal_resume_checkpoint=True）

### 行业证据（URL）
- Siril geometry/crop：https://siril.readthedocs.io/en/latest/processing/geometry.html
- Siril seqapplyreg framing：https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
- Siril CFA 约束：https://siril.readthedocs.io/en/latest/GUI/main-interface.html
- Siril Commands：https://siril.readthedocs.io/en/stable/commands.html
- PixInsight M31-Ha 示例：https://www.pixinsight.com/examples/M31-Ha/
- PixInsight StarAlignment mosaic：https://pixinsight.com/forum/index.php?threads/building-large-mosaics-with-staralignment.1953/
- APP 同尺寸要求：https://www.astropixelprocessor.com/community/faq/rgb-combination-and-same-size-requirements/
- APP mosaic 裁剪：https://www.astropixelprocessor.com/community/main-forum/cropping-a-mosaic/
- DeepSkyStacker 实现：https://github.com/deepskystacker/DSS/blob/master/DeepSkyStacker/ResultParameters.cpp
- Astropy reproject footprint：https://reproject.readthedocs.io/en/stable/footprints.html
- Astropy reproject mosaicking：https://reproject.readthedocs.io/en/stable/mosaicking.html
- DrizzlePac Handbook：https://hst-docs.stsci.edu/drizzpac
- Montage components：https://irsa.ipac.caltech.edu/Montage/docs/components.html

### 上游调研（deep-sky-edge-cropping-research/results/）
- `radiometric_fallback_detection.json:181` — `stage2_color_artifact_max_crop` 未接线
- `crop_decision_roi_stop.json:185` — 同上
- `current_stage2_gap_recommendations.json:132` — resume 不重建 crop_report
- `current_stage2_gap_recommendations.json:188` — `stage2_level_artifact_window` 未配置
- `current_stage2_gap_recommendations.json:192` — `crop_margin` 残留不驱动 Stage 2
- `interpolation_downstream_guard_band.json` — 固定 3px guard 无通用行业证据
- `software_pipeline_practices.json` — 上述 19 个行业文档 URL 源

### 未验证项（显式标注）
- **未验证**：真实 Seestar 目标上边缘暗信号被误裁的比例（需样本回测）
- **未验证**：Stage 4 是否重新 plate solve 自纠裁切坐标偏差（D1.6）
- **未验证**：Siril 1.4.0 `crop` 是否内部强制 CFA 偶数坐标（D3.4）
- **未验证**：`stage2_edge_black_target` 回退 0.10 在实际运行中是否被触发（正常配置下为 0.03）
