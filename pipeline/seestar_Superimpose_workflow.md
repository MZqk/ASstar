# `seestar_Superimpose.py` 流程与逻辑说明

本文档针对 [`seestar_Superimpose.py`](./seestar_Superimpose.py) 的当前实现，整理其入口流程、阶段职责、自动调参、回退策略和产物行为。目标是让后续维护时能快速判断某段逻辑位于哪一层、失败后会怎样降级、哪些文件会被读写。

## 1. 脚本定位

该脚本是 Seestar 后期处理的单一真源，运行前提是：

- 在 Siril Python 环境中执行，依赖 `numpy` 与 `sirilpy`
- Siril 版本至少为 1.4.0
- 工作目录中存在 `.fit/.fits` 输入
- SyQon-Starless / SASP Dark Star 可用时可执行去星；不可用时脚本会退化但不中断整条链路
- 可选插件（SetiAstroSuitePro、CosmicClarity、VeraLux 等）可用时提供增强处理路径

脚本职责不是做 GUI 调度或打包，而是在 Siril 已连接、工作目录已确定的前提下，完成一轮从输入 FITS 到最终 TIFF / PNG / FITS 导出的处理流水线，并在启用时追加可选 AI 后期副本导出。

实现组织上，阶段 11 的运行入口已从主脚本剥离到独立模块
`pipeline/stage11_ai_postprocess.py`，由主脚本调用执行。

## 2. 总体执行顺序

入口在 `SeestarPostProcessor.run()`，真实执行顺序如下：

```mermaid
flowchart TD
    A[connect Siril] --> B[requires 1.4.0]
    B --> C[stage1 前期准备]
    C --> D[auto tune 自动调参]
    D --> E[stage2 裁切]
    E --> F[stage3 背景提取]
    F --> G[stage4 图像解析+色彩校准]
    G --> H[stage5 星点矫正/锐化/初步降噪]
    H --> I[stage6 拉伸]
    I --> J[stage7 去星与星点层准备]
    J --> K[stage8 Starless 深加工]
    K --> L[stage9 星点处理与合成]
    L --> M[stage10 最终降噪与导出]
    M --> N[stage11 AI后期(可选)]
    N --> O[cleanup 清理]
    O --> P[汇总阶段结果]
```

其中：

- 阶段 1-5 视为线性阶段
- 阶段 6-10 视为非线性核心阶段
- 阶段 6-8 可在 AI 总开关开启且凭据齐全时启用参数优化与诊断记录；阶段 11 为可选 AI 后期副本（默认关闭，失败可降级）
- 自动调参发生在阶段 1 完成之后、阶段 2 开始之前
- 脚本不会因为单个非致命阶段失败就立刻退出，而是尽量记录 `ok / degraded / failed / skipped` 后继续处理
- 当 `SEESTAR_INPUT_MODE=result_linear_resume` 时，会改走显式续跑分支：
  `prepare result_linear.fit -> auto tune -> stage6 -> stage7 -> stage8 -> stage9 -> stage10 -> stage11 -> cleanup`
  同时把阶段 2-5 记录为 `skipped`

## 3. 核心对象分工

### 3.1 `PipelineConfig`

`PipelineConfig` 是全部可调参数的集中入口，主要分为几类：

- 重试控制：`max_retries`、`retry_delay`
- 阶段 1 质量门控：`stage1_register_fail_ratio_max`
- 裁切：`crop_margin`
- 背景提取：`bg_samples`、`bg_tolerance`、`bg_smooth`、`bg_quality_gate_enabled`、`bg_std_worsen_ratio_max`、`bg_median_drop_ratio_min`、`bg_object_preserve_ratio_min`、`bg_edge_black_rise_max`
- 线性降噪：`denoise_enabled`、`denoise_mod`、`denoise_safety_max`
- 拉伸：`asinh_stretch`、`asinh_offset`、`ghs_shadowsclip`、`ghs_stretchamount`
- 星云增强：`nebula_saturation`、`nebula_bg_factor`
- 星点混合：`star_intensity`、`star_fallback_intensity`（`remix_safe_blend`、`remix_nebula_weight` 为旧配置兼容字段）
- 最终导出：`final_saturation`、`final_bg_factor`
- AI：`ai_post_enabled`、`ai_endpoint`、`ai_model`、`ai_api_key`、`ai_timeout_sec`、`ai_strength`、`ai_prompt`、`ai_stage6_enabled`、`ai_stage7_enabled`、`ai_stage8_enabled`
- AI 诊断与 Stage11 门控：`ai_bg_median_delta_max`、`ai_color_ratio_delta_max`、`ai_core_growth_ratio_max`、`ai_star_growth_ratio_max`，以及阶段 6-8 的背景/裁剪/星点/蓝偏/饱和度/微对比诊断阈值
- 运行行为：`checkpoint_mode`、`debug_mode`、`auto_tune_enabled`、`auto_tune_debug`
- 工作流插件控制：`workflow_plugin_probe_enabled`（默认 `False`，仅在显式启用时探测插件命令）、`spcc_enabled`（默认 `True`，可通过 `SEESTAR_SPCC_ENABLE=0` 禁用）、`aberration_api_enabled`（默认 `False`，API 路径在 siril-cli 线程所有权场景下可能失败）、`optional_color_transform_enabled`（默认 `False`，启用时阶段 5/8/9 尝试 Alchemy/Hubble 等转色插件）
- 输入过滤：`exclude_prefixes`、`exclude_suffixes`、`exclude_substrings`

脚本没有把参数散落在各阶段函数中，后续维护时应优先检查这里。

### 3.2 `ImageFeatures` / `AutoTuneResult`

这两个数据结构用于自动调参：

- `ImageFeatures` 保存背景、中值、星密度、星点尺寸、目标面积、亮核比例、边缘黑边比例等测量值
- `AutoTuneResult` 保存识别出的目标类型、修改过的参数列表和告警说明

### 3.3 `StageResult`

每个阶段结束后都会追加一条 `StageResult`，状态只允许是：

- `ok`：阶段成功完成
- `degraded`：阶段未完全成功，但主流程继续
- `failed`：该阶段核心目标失败
- `skipped`：阶段被显式跳过

注意：`failed` 并不自动等于整条流水线退出。当前实现更偏向"最大化导出结果"。

### 3.4 `PipelineLogger`

`PipelineLogger` 只是一个轻量日志器，负责：

- 统一打印 `DEBUG / INFO / WARN / ERROR`
- 为每个阶段记录起止和耗时
- 为最终汇总提供足够可读的控制台输出

### 3.5 插件脚本基础设施

脚本提供了一套完整的插件脚本发现和执行体系：

- `_resolve_siril_scripts_root()`：从 `SEESTAR_SIRIL_PLUGIN_DIR` 环境变量指向的目录中定位 `siril-scripts` 根目录
- `_find_plugin_script()`：在 siril-scripts 根下按相对路径查找脚本文件
- `_validate_plugin_script_prerequisites()`：在执行前检查脚本所需的 Python 模块是否可用（如 `PyQt6`、`tiffile`、`astropy` 等），避免在离线环境中产生 traceback 噪声
- `_run_plugin_script_by_path()`：通过 Siril `pyscript` 命令在 Siril 进程内执行脚本
- `_run_plugin_script_cli_subprocess()`：以外部 Python 子进程调用脚本 CLI 模式（不走 Siril pyscript GUI 路径），支持超时控制和实时日志转发
- `_run_first_available_command()`：在多个候选命令中按顺序执行第一个可用命令；当 `workflow_plugin_probe_enabled=False` 且未设置 `allow_when_probe_disabled` 时直接跳过
- `_install_pyqt6_headless_stub()`：为 SASP Aberration API 等需要 PyQt6 的模块注入无头桩模块，使其在无 GUI 环境中可正常导入

已知脚本前置检查映射（`_SCRIPT_PREREQUISITE_MODULES`）：

| 脚本                         | 前置模块                           |
|------------------------------|-------------------------------------|
| `CosmicClarity_Sharpen.py`   | `PyQt6`, `tiffile`                 |
| `CosmicClarity_Denoise.py`   | `PyQt6`, `tiffile`                 |
| `SyQon-Starless.py`          | `PyQt6`, `PySide6`, `astropy`, `scipy` |

## 4. 自动调参逻辑

自动调参不是独立阶段，而是 `run()` 中插在 `stage1_preparation()` 与 `stage2_view_correction()` 之间的配置重写步骤。

### 4.1 输入来源

自动调参依赖当前已加载的图像像素数据和输入路径提示：

1. 先调用 `self.siril.get_image_pixeldata(preview=False)` 读取当前图像
2. 使用 `self.source_file` 作为文件名/目录上下文
3. 失败时直接回退到 `base_cfg`，不影响后续阶段继续执行

### 4.2 特征提取

`measure_image_features()` 会把图像归一化成 RGB 浮点数组，再提取：

- 背景亮度与波动：`bg_median`、`bg_std`
- 颜色倾向：`red_dominance`、`blue_dominance`
- 星场复杂度：`star_density`、`median_star_size`
- 目标覆盖特征：`object_area_ratio`、`diffuse_ratio`
- 高亮核心特征：`core_brightness_ratio`
- 黑边特征：`edge_black_ratio`

所有结果都会做安全限幅；如果测量异常，则整体回退为保守默认值。

### 4.3 目标识别顺序

`detect_target_type()` 分两层：

1. 先看路径和文件名关键词，例如 `m42`、`andromeda`、`jupiter`
2. 如果关键词无法命中，再按图像特征保守推断

支持的目标类型包括：

- `EMISSION_NEBULA`
- `REFLECTION_NEBULA`
- `GALAXY`
- `CLUSTER`
- `PLANETARY_NEBULA`
- `WIDEFIELD`
- `PLANETARY`
- `UNKNOWN`

### 4.4 调参顺序

`auto_tune_config()` 的处理链固定为：

1. 默认配置副本
2. 从测量特征归一化出 `noise/star/diffuse/core/edge/red/blue/object` 分数
3. 使用连续公式直接生成核心参数
4. 用 `clamp_config()` 做统一安全限幅

比较关键的调参倾向：

- 背景噪声高：增强 `denoise_mod`，并降低拉伸强度
- 星点密度高：降低 `star_intensity`
- 红色占优：降低 `nebula_saturation` / `final_saturation`
- 蓝色占优：适度提高 `nebula_saturation` / `final_saturation`
- 亮核比例高：提高 `asinh_offset`、收敛拉伸参数
- 边缘黑边明显：提高 `crop_margin`
- 主体覆盖面积大：降低 `bg_samples` 并提高 `bg_smooth`

`TargetType` 仍会识别并记录日志，但在当前版本中仅用于诊断说明，不再直接驱动 preset 套参。

### 4.5 强制运行时开关

自动调参完成后，`_apply_forced_runtime_switches()` 会检查 `SEESTAR_DENOISE_FORCE` 环境变量，允许在自动调参之后强制覆盖 `denoise_enabled` 的值。

自动调参失败时不会终止流程，只会恢复默认配置并记录告警。

## 5. 各阶段详细说明

### 5.1 阶段 1：前期准备

职责：创建干净的 `process/` 处理目录，并决定本轮输入来自"已叠加文件"还是 `Light_` 单帧。

处理逻辑：

- 先清理历史 `process_*` 目录
- 强制删除并重建 `process/`
- 一次遍历工作目录所有 `.fit/.fits`
- 根据前后缀、关键子串与 `exclude_substrings` 过滤中间产物，选出候选叠加图（例如 `sasp_*`、`*starless*`、`*starmask*` 会被跳过）
- 若存在候选叠加图，按修改时间降序选择最新文件
- 若没有叠加图但有 `Light_` 单帧，则转入预处理链
- 两类输入都没有时抛出 `SirilError`

预处理链只在 `Light_` 场景触发，顺序固定为：

```text
link -> calibrate -debayer -> register -2pass -> seqapplyreg -filter-round=2.5k -> stack
```

这里会先把本轮 `Light_` 文件隔离到 `process/_light_input/`，统一命名为 `lightsrc_00001.fit` 这种格式，目的是避免历史序列和大小写差异干扰 Siril 的 `link` / `stack`。

注册质量门控：预处理结束后会统计配准成功/失败帧数。当失败比例超过 `stage1_register_fail_ratio_max`（默认 0.10）时，阶段记为 `degraded`。

阶段输出：

- `process/working.fit`
- `process/stage1_prepared.fit`
- `self.source_file`
- `self._stage1_input_mode`（`"stacked"` 或 `"light_preprocess"`）
- Siril 当前加载图像切换到 `working`

### 5.2 阶段 2：裁切

职责：在仍处于线性阶段时裁切边缘。

当前版本阶段 2 仅做裁切，不再包含 mirrorx 翻转或 platesolve 天文定位（platesolve 已移至阶段 4）。

裁切逻辑：

1. 获取图像尺寸
2. 按 `crop_margin` 比例计算每边裁切像素
3. 执行 `crop` 命令
4. 测量 `edge_black_ratio`，若仍高于阶段 2 目标阈值，会在阶段 2 内迭代执行自适应黑边裁切并保存到 `stage2_corrected.fit`；后续阶段只诊断黑边，不再临时裁切去星输入

回退逻辑：

- 裁切失败时阶段记为 `degraded`
- 无法获取图像尺寸或图像太小时跳过裁切

阶段检查点：`stage2_corrected.fit`

### 5.3 阶段 3：背景提取

职责：消除背景梯度，优先保证背景平整，再考虑不误伤弱信号。

执行顺序：

1. 保存阶段 3 输入基线 `stage3_bg_input`，用于质量门控后回滚
2. 当 `workflow_plugin_probe_enabled=True` 时，按插件链尝试：`DBE -> ADBE -> GXP -> NOX -> AutoDBE -> AutoBGE -> VeraLux NOX`
3. 每个候选成功后都会做背景质量门控（背景噪声、背景中值、目标覆盖、边缘黑场）
4. 若质量门控不通过，自动回滚到基线并继续尝试下一个候选
5. 插件链不可用或被拒绝后，回退到 `subsky -rbf`（多组参数变体）
6. 若 RBF 仍失败，最后退回 `subsky 1`（线性多项式）
7. 若全部失败，阶段记为 `degraded`

RBF 参数变体：脚本会基于当前 `bg_samples` / `bg_tolerance` / `bg_smooth` 生成最多 3 组去重的参数组合，依次尝试。

阶段检查点：`stage3_bgremoved.fit`

### 5.4 阶段 4：图像解析 + 色彩校准

职责：执行天文定位（platesolve）并完成线性阶段的色彩平衡。

当前版本阶段 4 合并了原阶段 2 中的 platesolve 和色彩校准。

顺序：

1. 执行 `platesolve`
2. 当 `spcc_enabled=True` 且运行时允许时，优先尝试 `spcc`
3. `spcc` 失败时尝试 `pcc`
4. `spcc/pcc` 均不可用时尝试 `ccm` 灰世界色彩校准回退
5. 若 `platesolve` 失败，仍继续尝试 `spcc/pcc`，最后再回退 `ccm`

SPCC 运行时保护：

- 可通过 `SEESTAR_SPCC_ENABLE=0` 全局禁用 SPCC
- 当输入来自 `Light_` 预处理模式时，默认跳过 `spcc`（规避部分 `siril-cli` 运行环境崩溃风险），直接走 `pcc -> ccm`
- 如需强制在该模式下尝试 `spcc`，设置 `SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS=1`

CCM 灰世界回退实现：

- 从信号区域提取 R/G/B 中值
- 以 G 通道为基准计算 R/B 增益（限幅在 0.65–1.55）
- 通过 `ccm` 命令应用对角色彩矩阵

回退逻辑：

- 若 `platesolve_ok=False`，不会提前中断；仍会尝试 `spcc/pcc`
- `spcc/pcc` 任一步失败都会记录具体原因
- 若 `platesolve` 失败且进入 `CCM` 非光度回退，阶段会严格记为 `degraded`
- 阶段结果 `message` 会包含 `platesolve/SPCC/PCC/CCM` 的失败或回退信息

阶段检查点：`stage4_colorbalanced.fit`

### 5.5 阶段 5：星点矫正 / 锐化 / 初步降噪

职责：在拉伸前完成星点矫正、锐化和初步降噪，避免非线性放大后更难控制。

逻辑要点：

- 阶段开始先导出 `result_linear.fit`（预降噪版本，后续不会被覆盖）

星点矫正优先链：

1. 查找本地 Aberration ONNX 模型（`model_v2_0_1.onnx` 或 downloads 目录下最新 `.onnx`）
2. 若本地模型存在，通过 `SASP Aberration API`（`setiastro.saspro.aberration_ai.run_aberration_ai_on_array`）执行
3. 若本地模型不存在且 `aberration_api_enabled=True`，仍尝试 API 路径
4. API 路径会自动加载 `setiastrosuitepro` wheel 并在需要时注入 PyQt6 无头桩

锐化优先链：

1. `CosmicClarity_Sharpen.py`（通过 `pyscript` 在 Siril 内执行）
2. `Siril-CC Sharpen Both 0.1/3/0.5`（多种参数顺序别名，纯命令探测）
3. `CosmicClarity_Native.py --mode sharpen/denoise`（优先 Native 模型）
4. `unsharp 1.0 0.8`（内置回退）

初步降噪优先链：

1. `CosmicClarity_Denoise.py`（通过 `pyscript` 在 Siril 内执行）
2. 脚本失败后会触发 `Siril-CC Sharpen Both 0.1/3/0.5` 回退（若锐化阶段未使用过）
3. 若 `denoise_enabled=True`，进入内置 `denoise -mod=<denoise_mod>`
4. `denoise_mod` 会在执行前与动态安全上限比较：基线是 `denoise_safety_max`，在高星密（>0.004）/亮核（>0.08）/小目标高星场景会进一步收紧
5. 若 `denoise_enabled=False` 且未命中任何插件处理步骤，阶段记为 `skipped`

可选转色（当 `optional_color_transform_enabled=True`）：

- `VeraLux Alchemy` -> `Hubble Palette From Dual Band` -> `NB to RGB`

阶段检查点：`stage5_denoised.fit`

阶段额外产物：`result_linear.fit`（位于工作目录根目录，默认不会被下一轮输入扫描命中；可供 GUI 显式续跑模式继续执行阶段 6-11）

### 5.6 阶段 6：主体拉伸

职责：把线性图像变成可视化的非线性图像。

策略判断：

- 星团：`Asinh`，不做 GHS，避免星点膨胀
- 普通星系：`Asinh` + 轻曲线；GHS 只作为保守候选
- M42/猎户这类高动态星云：`Asinh+GHS` + 曲线
- 暗弱星云：`Asinh+GHS`，GHS 强度保守，并依赖阶段 6 质量门控保护噪声
- 背景很脏/信噪低：优先 `Asinh`，GHS 使用低强度谨慎候选，不做曲线
- 核心很亮、外围很暗：`Asinh+GHS`，必要时曲线平衡核心与外围

AI 介入（`SEESTAR_AI_ENABLED=1` 且 endpoint/model/key 齐全时）：

1. 拉伸前保存 `stage6_input.fit`
2. 把自动调参特征、目标类型和当前 GHS/Asinh 参数发给 AI，要求返回严格 JSON `stage6_stretch_plan`
3. AI 推荐候选必须被本地策略允许；随后继续执行本地策略候选；每个候选都从 `stage6_input` 重新载入
4. 所有候选都会保存诊断；最终按背景中值、黑场比例、高光裁剪和星点膨胀风险选择 least-risk 候选，避免第一条成功命令留下过暗结果
5. 记录背景、亮部和星点诊断，但诊断不作为硬门控，不会因为指标分数覆盖成功拉伸结果

候选来源：

1. 插件拉伸：`VeraLux HyperMetric Stretch` / `HyperMetricStretch`（仅当 `workflow_plugin_probe_enabled=True`）
2. 本地策略候选：`asinh`、`asinh` 后接 `autoghs -linked <shadowsclip> <stretchamount>`、单独 `autoghs`、`autostretch`

拉伸后可选微调：

- `VeraLux Curves` / `Curves`，仅在策略允许且 `workflow_plugin_probe_enabled=True` 时执行

关键行为：

- 无论拉伸是否成功，都会强制保存 `stage6_stretched.fit`
- 如果所有拉伸方法都失败，阶段状态记为 `failed`
- 但 `run()` 不会因此立即退出，后续阶段仍会继续尝试使用这个检查点
- 诊断输出 `stage6_stretch_quality.json`，记录策略、候选、指标和最终选择

这意味着阶段 6 的 `failed` 更像"拉伸目标未达成"，而不是"整条链路终止"。

### 5.7 阶段 7：去星与星点层准备

职责：调用去星工具生成去星图，并尽量构造一个可控的星点层。

去星优先链：

1. `SyQon-Starless.py` CLI 子进程（Zenith v1 模型，tile-size=512，overlap=64，no_gpu，带超时与日志转发）
2. `SASP Dark Star`（命令探测）

SyQon 路径补充逻辑：

- 脚本执行成功后，会在 `process/` 中收集产物：先按 `starless_{stretched_name}` / `starless` 查找，再按 glob 兜底
- 若 SyQon 脚本未生成 starless 产物，视为失败并回退到 SASP Dark Star

正常路径（去星成功后）：

1. 保存 `starless.fit`
2. 优先手动构建星点层：`stretched - starless`
3. 手动失败时，从可能的 `starmask` / `*_stars` 文件中兜底查找
4. 再不行则扫描 `process/` 中最新的星点蒙版
5. 导出 SASP 交换文件（`sasp_starless_input.fit`、`sasp_starmask_input.fit`）

AI 参数优化（`SEESTAR_AI_ENABLED=1` 且 endpoint/model/key 齐全时）：

- 去星前会先请求 `stage7_starless_plan`，将 AI 建议的 `tile_size`、`overlap`、`use_axiom` 实际传给 SyQon CLI；Axiom 只在本地模型存在时允许；`tile_size` 下限为 512、`overlap` 下限为 64，不再把 `tile=256/overlap=32` 作为优化方向
- 每次得到 `starless.fit` / `starmask.fit` 后生成一次质量记录，最终写入 `process/stage7_quality.json`
- 本地指标先判断 `starless` 残星、`starmask` 缺星和 `starmask` 过宽，再把指标发给 AI 做保守裁判
- 若残星、缺星或蒙版过宽指标偏差，会在 `SEESTAR_STAGE7_QUALITY_RETRY_MAX` 预算内用其他 SyQon 参数组合重跑，并把质量分最低的 `starless.fit` / `starmask.fit` 恢复为最终阶段产物
- 若最终 `residual_star_score` 仍高于阈值，AI 可返回 `residual_suppression_strength` 和 `stage9_star_intensity_scale`；脚本会做安全限幅后结合 `starmask` 与 `stage6_stretched - starless` residual map 实际修改 `starless.fit` 抑制残星，并把最终阶段 9 回混缩放写入 `stage7_quality.json.stage9_star_remix`
- 质量诊断不再把成功的 SyQon/SASP 结果直接降级为 `stage6_stretched`；只有所有去星工具失败才走退化路径

回退逻辑：

- 如果所有去星命令都失败，阶段记为 `degraded`
- 同时把 `stage6_stretched` 重新保存成 `starless.fit`，让阶段 8-10 仍然可以继续执行
- 因此去星工具不可用时，脚本退化为"不做真正去星"的流程

阶段检查点：`stage7_starless.fit`

阶段输出：

- `starless.fit`
- 可能存在的 `starmask.fit`
- `sasp_starless_input.fit`（工作目录，供外部工具使用）
- `sasp_starmask_input.fit`（工作目录，供外部工具使用）

### 5.8 阶段 8：Starless 深加工

职责：对去星结果做增强处理。阶段 8 默认生成 Starless soft masks 后分区增强，保护亮核心和背景区域；SASP WaveScale Dark Enhancer API 仍可作为增强源，但输出会通过 mask 回混，避免整图锐化/提亮。打包环境不再探测实验性 `sasp_*` Siril 命令，因为这些命令未由当前 SASP wheel 注册到 Siril CLI；API 不可用时直接使用确定性的内置分区增强链。

外部回写导入：

- 阶段开始时会在工作目录和 `process/` 中查找 `sasp_starless.fit` / `starless_sasp.fit` / `starless_from_sasp.fit`
- 若存在外部回写文件，导入并覆盖 `starless.fit`

分区保护：

1. 基于 `stage8_input_starless` 生成 `core_mask`、`nebula_mask`、`faint_nebula_mask`、`background_mask`
2. 核心区域优先回混原始 Starless，避免高光继续变白，并通过 soft mask 保持自然过渡
3. 背景区域强回混到“仅降噪版本”，不做锐化、局部对比或饱和度提升，并收紧 `bg_std_growth` 验收
4. 主体星云允许适度局部对比和轻锐化；外围暗云气只做轻量暗部提升和轻微饱和

插件处理链：

1. 默认调用 `setiastro.saspro.wavescalede.compute_wavescale_dse` 生成候选增强；当背景噪声或高光风险高时会降低 `boost_factor/mask_gamma` 上限
2. 候选增强只按主体/外围 mask 回混；核心和背景保持保护策略，高背景噪声或高光风险时会进一步压低回混权重
3. 可选调色（当 `optional_color_transform_enabled=True`）：`SASP Selective Color Correction` / `Selective Color Correction`

回退策略（插件不可用时）：

1. 若 SASP Python API 不可用，执行内置分区增强链；AI 开启且计划可用时使用 AI 生成的 `saturation/unsharp_amount` 作为分区强度输入
2. 内置链顺序为背景/外围轻降噪 -> 信号区蓝偏预抑制 -> 外围暗云气提升 -> 主体局部对比 -> 非核心轻锐化 -> 分区饱和度微调
3. 若分区增强失败，阶段记为 `degraded`

阶段检查点：`starless_enhanced.fit`、`stage8_enhanced.fit`

AI 参数优化（`SEESTAR_AI_ENABLED=1` 且 endpoint/model/key 齐全时）：

1. 处理前保存 `stage8_input_starless.fit`；启用分区增强时即使 AI 关闭也会保存，用于本地质量诊断
2. 基于输入 Starless 特征请求 `stage8_processing_plan`，并把 AI 生成的 `saturation/bg_factor/unsharp_radius/unsharp_amount/apply_after_plugins` 实际用于 Siril 命令
3. 插件链可用时，默认在插件处理后追加分区像素增强；AI 明确返回 `apply_after_plugins=false` 时才保留插件 mask 回混结果不追加
4. 插件链或内置增强前会先按信号区 R/G、B/G 做蓝偏预抑制并同步降低饱和度，后置 `ccm` 只处理残余明显蓝偏
5. 再判断蓝偏、饱和度增长、微对比增长、高亮裁剪、背景噪声增长、背景提亮、核心裁剪增长和纹理伪影增长，并把指标写入 `stage8_quality.json`；AI 开启时同时发给 AI 生成诊断
6. 若蓝偏超限，会按 AI 返回的 `target_blue_excess`（缺失时用本地阈值）对当前 `stage8_enhanced` 执行受限 `ccm` 蓝通道修正并重新保存 `starless_enhanced.fit`，随后重新计算 `stage8_quality` 并更新 `final`
7. 若饱和度、微对比、高光、背景噪声或纹理伪影风险仍偏高，会从 `stage8_input_starless.fit` 用保守分区增强重跑并覆盖 `starless_enhanced.fit`，避免过处理结果传入阶段 9

### 5.9 阶段 9：星点处理与合成

职责：把阶段 8 的 Starless 结果与星点信息重新组合。

外部回写导入：

- 阶段开始时会查找 `sasp_starmask.fit` / `starmask_sasp.fit` / `starmask_from_sasp.fit`
- 若存在外部回写文件，导入并覆盖 `starmask.fit`

星点预处理链（当 `workflow_plugin_probe_enabled=True` 且有 starmask 时）：

1. 星点拉伸：`SASP Star Stretch` / `NB to RGB Stars`
2. 星点去紫：`SASP Invert/SCNR` / `SCNR`
3. 星点微调：`SASP Curves Editor` / `Curves`
4. 处理后导出更新的 SASP 交换文件

Starless 二次细化（在合成前）：

1. `VeraLux Revela` / `Revela`
2. 可选调色：`VeraLux Vectra` / `Vectra`（当 `optional_color_transform_enabled=True`）
3. `VeraLux Curves` / `Curves`

星点合成优先级：

1. `VeraLux StarComposer` / `StarComposer`（当 `workflow_plugin_probe_enabled=True` 且 Stage7 未要求降低星点强度）
2. 上一阶段 Starless 图像的像素级 `starmask` 回混
3. 使用备用星点强度再次回混
4. 放弃回星，直接使用无星结果

默认回混公式：

```text
final = starless_enhanced + starmask * intensity
```

Stage7 联动：

- 若 `stage7_quality` 显示残星超标，阶段 9 会跳过不可控的 StarComposer，改走像素级回混
- 实际 `intensity` 和备用强度会乘以 Stage7 计算出的 `intensity_scale`，降低残星与回混星点叠加造成的二次星点风险

设计意图：

- 阶段 9 的主处理文件必须来自阶段 8 的 `starless_enhanced`
- 不再用 `stage6_stretched` 与 `starless_enhanced` 做安全混合，避免 `pm` 多文件表达式成功返回但实际保存回阶段 6 图像
- 合成后同时记录 `stage9_remixed` 对 `stage8_enhanced` 与 `stage6_stretched` 的差异，便于排查阶段 9 是否真正生效

回混存在两层强度：

1. `star_intensity`：自动调参和最终有效强度上限为 `1.05`
2. `star_fallback_intensity`：自动调参和最终有效强度上限为 `1.05`

阶段状态判断：

- StarComposer 或上一阶段像素级回混成功时为 `ok`
- 没有 `starmask` 时为 `skipped`
- 两轮像素级回混都失败时为 `degraded`

阶段检查点：`stage9_remixed.fit`

### 5.10 阶段 10：最终降噪与导出

职责：用当前最优可用图像做最终微调并导出。

最终图像加载优先级：

1. `stage9_remixed`
2. `starless_enhanced`
3. `stage6_stretched`

也就是说，即使阶段 9 或阶段 8 没有产出，脚本仍尝试导出一个最差也能接受的结果。

导出前会再做一次：

```text
satu <final_saturation> <final_bg_factor>
```

最终降噪优先链：

1. `CosmicClarity_Denoise.py` in-process script（复用已验证更稳定的 Siril 内执行路径，使用 classic 参数）
2. 若 in-process 失败，再尝试 `CosmicClarity_Denoise.py` CLI 子进程模式
3. 若 classic Denoise 不可用或失败，尝试 CosmicClarity Native Denoise，再尝试 `Siril-SCUNet Denoise`
4. 若 `Siril-SCUNet Denoise` 不可用且 `aberration_api_enabled=True`，再回退 `SASP Aberration API`
5. 若 `aberration_api_enabled=False`（默认），跳过 Aberration API 回退

状态规则：

- `CosmicClarity_Denoise.py`、`Siril-SCUNet Denoise`、`SASP Aberration API` 任一成功，阶段保持 `ok`（除非其他导出步骤失败）
- 仅当以上路径都不可用时，阶段记为 `degraded`（当 `aberration_api_enabled=True`）或记录跳过信息（当 `aberration_api_enabled=False`）
- 阶段结果 `message` 会写入最终降噪与导出失败原因，便于日志排查

阶段检查点：`stage10_final.fit`

然后切回原工作目录导出三类文件：

- TIFF：优先使用基于 FITS 元数据的动态命名，失败时回退 `result_processed.tif`
- PNG：优先动态命名，失败时回退 `result_processed.png`
- FITS：优先动态命名并追加 `_final`，失败时回退 `result_final.fit`

若本轮输入模式是 `result_linear_resume`：

- 主输出会统一追加 `_linear` 后缀，避免覆盖原始完整流程产物
- TIFF fallback 改为 `result_processed_linear.tif`
- PNG fallback 改为 `result_processed_linear.png`
- FITS fallback 改为 `result_final_linear.fit`

默认动态命名模板：

```text
$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec_$DATE-OBS:dm12$_processed
```

实际字符串中仍保留了 Siril 的格式占位符语法，由 Siril 在保存时展开。

### 5.11 阶段 11：AI 后期美化（可选）

职责：在阶段 10 产物不变的前提下，尝试生成 AI 后期副本（`*_ai`）。

实现模块：`pipeline/stage11_ai_postprocess.py`，由主脚本 `stage11_ai_postprocess()` 方法调用。若模块导入失败，阶段直接记为 `degraded` 并记录导入错误。

启用条件：

- `SEESTAR_AI_ENABLED=1/true`
- 同时提供 `SEESTAR_AI_ENDPOINT`、`SEESTAR_AI_MODEL`、`SEESTAR_AI_API_KEY`
- `SEESTAR_AI_ENDPOINT` 支持完整 endpoint 或 base URL；base URL 会自动尝试补全到 `/v1/chat/completions`
- GUI 运行时会把 `SEESTAR_AI_*` 从以下来源注入到 pipeline：
  - 进程环境
  - `<App>.app/Contents/Resources/ai.env`（打包默认值）
  - `~/Library/Application Support/SeestarSuperimpose/runtime_home/.seestar_ai.env`
  - `<work_dir>/.seestar_ai.env`
- 同名键优先级：进程环境 > 工作目录覆盖 > 运行时覆盖 > 打包资源默认值

关键流程：

1. 保存当前图像为 `stage11_ai_source.fit`
2. 基于图像特征调用 OpenAI-compatible Chat Completions，让模型输出严格 JSON 参数建议；参数可包含 `blend_strength`
3. 使用本地 Python/Numpy 执行实际增强，并写出 `stage11_ai_output.png`（16-bit RGB PNG）
4. 将本地增强结果转换为 FITS，再做保守混合：`final = source*(1-strength) + ai*strength`
5. 执行质量门控，必要时自动降强度（减半）重混一次
6. 通过门控后导出：
   - `result_processed_ai.tif`
   - `result_processed_ai.png`
   - `result_final_ai.fit`

AI Plan 解析容错：

- 优先解析严格 JSON -> 尝试 fenced code block -> 扫描第一个平衡 JSON 对象 -> 从纯文本中正则提取调整参数
- 阶段 6/7 的 AI advisory 支持从 reasoning 文本中抽取 `autostretch/GHS/Asinh`、`tile_size/overlap`、残星判断、残星抑制强度和 Stage9 回混缩放，避免模型未输出严格 JSON 时完全失效
- 若 API 成功返回但 JSON 解析全部失败，会写出 `process/ai_raw_stage11_adjustment_plan.*`，并按当前最终图像特征生成保守 fallback 参数
- 阶段会输出 `process/stage11_quality.json`，记录最终参数、混合强度、源/混合后特征和质量门控结果
- 支持 `reasoning_content` 字段兼容（部分推理模型 `content` 为空时的回退）

温度回退：

- 默认先用 `temperature=0.1`，若模型返回 "only 1 is allowed" 错误，自动回退到 `temperature=1.0`
- Kimi 系模型默认直接使用 `temperature=1.0`

质量门控关注点：

- 背景中值漂移
- 色彩比值漂移（`R/G`、`B/G`）
- 亮核占比增幅
- 星点中位尺寸增幅

降级策略：

- 未启用或缺少必需环境变量：阶段记为 `skipped`
- 模块导入失败：阶段记为 `degraded`
- API 错误、返回体异常、参数解析失败、尺寸不匹配：阶段记为 `degraded`
- 门控失败且降强度重试后仍不达标：阶段记为 `degraded`
- 无论阶段 11 状态如何，阶段 10 原始导出都保留

## 6. 运行时环境变量覆盖

`_apply_runtime_env_overrides()` 在 `run()` 入口处（连接 Siril 之前）集中读取并覆盖配置：

| 环境变量 | 对应配置 | 默认值 | 说明 |
|---|---|---|---|
| `SEESTAR_DEBUG_MODE` | `debug_mode` | `False` | 开启后保留 stage* 中间文件 |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` | `workflow_plugin_probe_enabled` | `False` | 启用后阶段 3/6/7/9 会探测更广泛的插件命令；阶段 8 固定使用 SASP Python API，不探测未注册的实验性 `sasp_*` Siril 命令 |
| `SEESTAR_SPCC_ENABLE` | `spcc_enabled` | `True` | 设为 0 可全局禁用 SPCC |
| `SEESTAR_ABERRATION_API_ENABLE` | `aberration_api_enabled` | `False` | 启用 SASP Aberration API 路径 |
| `SEESTAR_DENOISE_ENABLE` | `denoise_enabled` | `False` | 启用内置线性降噪 |
| `SEESTAR_DENOISE_FORCE` | `_force_denoise_enabled` | — | 自动调参后强制覆盖 denoise_enabled |
| `SEESTAR_AI_ENABLED` | `ai_post_enabled` | `False` | AI 总开关：启用阶段 6-8 参数优化/诊断与阶段 11 AI 副本 |
| `SEESTAR_AI_ENDPOINT` | `ai_endpoint` | `""` | AI API endpoint |
| `SEESTAR_AI_MODEL` | `ai_model` | `""` | AI 模型名 |
| `SEESTAR_AI_API_KEY` | `ai_api_key` | `""` | AI API 密钥 |
| `SEESTAR_AI_PROMPT` | `ai_prompt` | `""` | 自定义 AI 提示词 |
| `SEESTAR_AI_TIMEOUT_SEC` | `ai_timeout_sec` | `90` | API 超时（限幅 15–300） |
| `SEESTAR_AI_STRENGTH` | `ai_strength` | `0.12` | AI 混合强度（限幅 0.05–0.25） |
| `SEESTAR_AI_STAGE6_ENABLE` | `ai_stage6_enabled` | `True` | 单独控制阶段 6 AI 拉伸顾问 |
| `SEESTAR_AI_STAGE7_ENABLE` | `ai_stage7_enabled` | `True` | 单独控制阶段 7 AI SyQon 参数计划、诊断与重试择优 |
| `SEESTAR_AI_STAGE8_ENABLE` | `ai_stage8_enabled` | `True` | 单独控制阶段 8 AI Starless 参数计划、蓝色修正与保守重跑 |
| `SEESTAR_STAGE7_QUALITY_RETRY_MAX` | `stage7_quality_retry_max` | `2` | 阶段 7 参数优化预算配置（限幅 0–3） |
| `SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS` | — | `"0"` | 允许 Light_ 模式下尝试 SPCC |
| `SEESTAR_SIRIL_PLUGIN_DIR` | `siril_plugin_dir` | — | 插件目录路径 |
| `SIRIL_PYTHON_CLI` | — | — | CLI 子进程使用的 Python 解释器 |

## 7. 重试与异常处理策略

### 7.1 `cmd_with_check()`

几乎所有 Siril 命令都通过 `cmd_with_check()` 调用。它负责：

- 把命令失败统一收口为日志 + 异常
- 对可重试错误做有限重试
- 避免对非幂等命令做自动重试

### 7.2 非幂等命令

以下命令被显式列入 `_NON_IDEMPOTENT`，默认不自动重试：

- `stack`
- `calibrate`
- `register`
- `seqapplyreg`
- `link`
- `save`
- `savetif`
- `savepng`
- `savejpg`
- `pm`

理由是这些命令重复执行可能污染状态、覆盖文件或产生难以判断的副作用。

### 7.3 可重试错误判定

脚本通过两类信息判断瞬时错误：

- `CommandStatus` 中的已知可重试状态码（`CMD_GENERIC_ERROR`、`CMD_THREAD_RUNNING`）
- 错误文本中包含 `generic error`、`thread running`、`connection`、`timeout`、`busy`

如果满足可重试条件，按 `retry_delay * attempt` 做递增等待。

## 8. 文件与目录行为

### 8.1 工作目录

- `self.work_dir`：Siril 当前工作目录，也是最终导出目录
- `self.process_dir`：本轮处理中间目录，固定为 `work_dir/process`

### 8.2 常见中间文件

处理中经常出现这些名字：

- `working.fit`
- `stage1_prepared.fit`
- `stage2_corrected.fit`
- `stage3_bg_input.fit`（质量门控基线）
- `stage3_bgremoved.fit`
- `stage4_colorbalanced.fit`
- `stage5_denoised.fit`
- `stage6_input.fit`、`stage6_stretch_quality.json`（仅阶段 6 AI 介入时）
- `stage6_stretched.fit`
- `stage7_quality.json`（仅阶段 7 AI 诊断时）
- `stage7_starless.fit`
- `starless.fit`
- `stage8_input_starless.fit`、`stage8_quality.json`（仅阶段 8 AI 诊断时）
- `starless_enhanced.fit`
- `stage8_enhanced.fit`
- `starmask.fit`
- `stage9_remixed.fit`
- `stage10_final.fit`
- `result_linear.fit`（工作目录中的拉伸前线性中间文件）
- `sasp_starless_input.fit`、`sasp_starmask_input.fit`（工作目录中的 SASP 交换文件）
- `stage11_ai_source.fit`、`stage11_ai_output.png`、`stage11_ai_output_fit.fit`、`stage11_ai_blended.fit`（仅阶段 11 运行时出现）

是否保留取决于：

- `checkpoint_mode=False`：保存所有阶段检查点
- `checkpoint_mode=True`：只保存关键检查点
- `debug_mode=True`：保留 `stage*` 等中间文件，但会清理 `*lightsrc*` 预处理序列
- `debug_mode=False`：清理阶段会删除大多数中间文件

### 8.3 清理阶段

`cleanup()` 在非调试模式下会：

- 先通过 `_cleanup_lightsrc_intermediates()` 清理 `process/` 内 `*lightsrc*` 相关文件和 `_light_input/` 目录
- 清理 `process/` 内大部分 `fit/fits/seq/log/csv/lst`
- 默认保留 `starless.fit`、`starmask.fit`
- 如果存在拉伸名，还保留 `stage6_stretched_starmask.fit`
- 不会删除工作目录中的 `result_linear.fit`
- 阶段 11 临时文件会在非调试模式下由 `stage11_ai_postprocess.py` 主动删除
- 若阶段 11 成功，工作目录会新增 `result_processed_ai.tif/png` 与 `result_final_ai.fit`
- 若目录已空则直接删除 `process/`

因此，默认模式下最终更强调"输出整洁"，而不是保留完整现场。

## 9. 维护时最值得注意的点

1. 阶段顺序是核心契约：1-10 为稳定主链，阶段 11 仅可选追加，不可插到 1-10 中间。
2. 自动调参是"重写配置"，不是"给某个阶段附加参数"，排查参数异常时要先看自动调参日志。
3. `workflow_plugin_probe_enabled` 控制广泛插件探测开关：为 `False` 时大部分插件命令不会被尝试，只有标记 `allow_when_probe_disabled=True` 的命令（如 SASP Dark Star、unsharp、CC Sharpen Both）仍可执行；阶段 8 默认走 SASP Python API，不再先探测实验性 `sasp_*` Siril 命令。
4. 阶段 5、6、7、8、9、10、11 都有明显降级路径，改动时不能只看成功路径。
5. `process/` 每轮都会重建，任何依赖历史中间文件的思路都不成立。
6. `Light_` 输入路径使用隔离目录再 `link`，不要直接在工作目录对历史序列做操作。
7. 去星工具失败时并不会终止，而是退化为"把拉伸图当作 starless 继续"，这一点会直接影响阶段 8-10 的视觉预期。
8. 导出不是固定依赖 `stage9_remixed`，而是按 `stage9_remixed -> starless_enhanced -> stage6_stretched` 逐级回退。
9. SASP 交换文件（`sasp_starless_input.fit`、`sasp_starmask_input.fit`）和外部回写检测（`sasp_starless.fit` 等）构成了与 SetiAstroSuitePro 的双向数据通道。
10. 插件脚本通过 `_SCRIPT_PREREQUISITE_MODULES` 做前置检查，缺少模块时会跳过而不是抛出异常。

## 10. 快速排障索引

如果后续需要定位问题，可以优先按下面的思路查：

- 没有识别到输入：看阶段 1 的候选叠加过滤和 `Light_` 检测
- Light_ 配准质量差：看阶段 1 的 `_stage1_registration_stats` 和 `stage1_register_fail_ratio_max`
- 参数表现异常：看自动调参日志和 `AutoTuneResult.changed_params`
- 背景残留：看阶段 3 是否从 RBF 降级到了 `subsky 1`
- 色彩偏差：看阶段 4 的 `platesolve_ok`、`spcc_enabled` 是否被禁用、`pcc` 是失败还是被跳过，以及 `CCM` 回退是否成功
- 星点矫正无效：看阶段 5 是否找到本地 ONNX 模型、`aberration_api_enabled` 状态、PyQt6 桩注入结果
- 锐化无效：看阶段 5 的锐化链是命中了 CosmicClarity 还是回退到 unsharp
- 拉伸异常：看阶段 6 最终落到了插件、`asinh`、`autoghs` 还是 `autostretch`
- 去星无效：看阶段 7 是 SyQon 成功、回退到 SASP Dark Star、还是已经退化成"直接保存拉伸图为 starless"
- 星点回混异常：看阶段 9 走的是 StarComposer 还是上一阶段 `starless_enhanced` + `starmask` 像素级回混
- 最终没有某类导出：看阶段 10 是否命中了回退文件名 `result_processed` / `result_final`
- 最终降噪缺失：看阶段 10 是 CC Denoise、SCUNet、Aberration API 还是全部跳过
- AI 副本缺失：看阶段 11 是 `skipped`（未启用/缺少 env/模块导入失败）还是 `degraded`（接口或门控失败）
- 插件全部跳过：检查 `SEESTAR_WORKFLOW_PLUGIN_PROBE` 和 `SEESTAR_SIRIL_PLUGIN_DIR` 是否正确设置

## 11. 结论

当前实现的设计重点不是"每一步都必须成功"，而是：

- 保持阶段边界清晰
- 尽量通过回退链保证有结果可导出
- 把可调参数集中管理
- 在输入类型、自动调参、去星回混和导出命名几个高风险点上做保守处理
- 通过 `workflow_plugin_probe_enabled` 开关控制插件探测范围，避免在缺少插件的环境中产生大量无意义的探测开销
- 通过 SASP 交换文件和外部回写检测实现与第三方工具的双向集成

因此，后续改动若涉及阶段顺序、自动调参、去星回退、导出优先级或插件探测逻辑，应视为高风险变更，必须同步检查对应的降级路径是否仍然成立。
