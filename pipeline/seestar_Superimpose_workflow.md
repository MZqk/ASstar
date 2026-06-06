# `seestar_Superimpose.py` 流程与逻辑说明

本文档针对 [`seestar_Superimpose.py`](./seestar_Superimpose.py) 的当前实现，整理其入口流程、阶段职责、自动调参、回退策略和产物行为。目标是让后续维护时能快速判断某段逻辑位于哪一层、失败后会怎样降级、哪些文件会被读写。

本文档是 pipeline 阶段逻辑、参数和产物细节的单一来源；`README.md` 与 `INTEGRATION_README.md` 只保留面向用户或集成层的摘要和引用。

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
    E --> F[stage3 preflight 目标画像+策略]
    F --> G[stage3 背景提取]
    G --> H[stage4 解析 stage4_psolved]
    H --> I[stage4 校色 stage4_color]
    I --> J[stage5 RL反卷积/轻降噪]
    J --> K[stage6 线性去星与星点层准备]
    K --> L[stage7 拉伸 Starless]
    L --> M[stage7.5 兼容记录/默认跳过]
    M --> N[stage8 Starless 深加工]
    N --> O[stage9 星点处理与合成]
    O --> P[stage10 最终降噪与导出]
    P --> Q[stage11 AI后期(可选)]
    Q --> R[cleanup 清理]
    R --> S[汇总阶段结果]
```

其中：

- 阶段 1-5 视为线性阶段，目标画像与 policy 选择合并到 Stage 3/4 preflight，不再单独占一个阶段
- 阶段 6 是线性去星与星点层准备，先在线性图上分离星点；阶段 7 再拉伸 starless，避免拉伸放大的脏背景进入去星模型
- 因此阶段 7-10 是 starless-first 非线性核心处理
- 默认 starless-first 路径下阶段 7.5 只记录兼容 skip，旧的去星前质量门控仍保留给非默认/调试路径
- 阶段 6 去星、阶段 8 增强和阶段 11 可在 AI 总开关开启且凭据齐全时启用参数优化与诊断记录；当前阶段 7 主体拉伸固定为本地两候选，不再让 AI 或 policy 扩展候选集合
- 自动调参发生在阶段 1 完成之后、阶段 2 开始之前
- 脚本不会因为单个非致命阶段失败就立刻退出，而是尽量记录 `ok / degraded / failed / skipped` 后继续处理
- 当 `SEESTAR_INPUT_MODE=stage2_corrected_resume` 时，会使用工作目录根下或旧 `process/` 下的 `stage2_corrected.fit` 作为已完成裁切/视场修正的叠加后中间结果：
  `prepare stage2_corrected.fit -> auto tune -> stage3 background -> stage4 color -> stage5 linear -> stage6 starless -> stage7 stretch -> stage7.5 skipped -> stage8 -> stage9 -> stage10 -> stage11 -> cleanup`
  同时把阶段 1 记录为 `skipped`，阶段 2 记录为加载既有中间结果
- 当 `SEESTAR_INPUT_MODE=result_linear_resume` 时，会改走显式续跑分支：
  `prepare result_linear.fit -> auto tune -> target preflight -> stage6 starless -> stage7 stretch -> stage7.5 skipped -> stage8 -> stage9 -> stage10 -> stage11 -> cleanup`
  同时把阶段 2、3、4、5 记录为 `skipped`

### 2.1 实际天文后期逻辑总览

当前代码不是按“先拉伸整图再去星”的旧顺序运行，而是先在较干净的线性图上拆出 starless/starmask，再只拉伸 starless 主体，最后受控回星。

| 流程层 | 代码阶段 / 检查点 | 天文后期目的 | 实际处理逻辑 |
|---|---|---|---|
| 输入统一 | Stage 1 / `working.fit`、`stage1_prepared.fit` | 得到一张可处理的线性 FITS | 优先使用最新叠加 FITS；没有叠加图时隔离 `Light_` 单帧并执行 debayer、register、stack |
| 画面边界 | Stage 2 / `stage2_corrected.fit` | 去掉黑边和窄幅彩色边缘，避免污染背景建模和去星 | 自动扫描四边近黑/暗边界/红蓝叠加伪影并独立裁切，再按 `edge_black_ratio` 迭代复检 |
| 目标策略 | Stage 3/4 preflight / `target_profile.json`、`pipeline_policy.json` | 根据目标类型保护真实星云、暗云气、星系核心或星团星色 | 用像素特征、FITS metadata、目标库和路径上下文生成 profile；Stage 4 platesolve 后再刷新一次 |
| 线性校正 | Stage 3-5 / `stage3_bgremoved.fit`、`stage4_color.fit`、`stage5_linear.fit` | 在线性域完成背景、校色、可选反卷积和轻降噪 | 背景按 `GraXpert/ADBE/DBE/subsky RBF/NOX` 理论效果顺序尝试；有 Siril scripts 缓存时 GraXpert/ADBE/DBE/AutoDBE 通过 `pyscript` 调用 `GraXpert-AI.py` / `AutoBGE.py`，所有候选必须通过保留门控；校色按 platesolve 成功时 SPCC -> 多星表 PCC，解析失败时先尝试 header 坐标辅助解析 / PCC header fallback，再进入本地回退；Stage 5 先 PSF/RL，再 post-RL denoise |
| 线性去星 | Stage 6 / `stage6_starless.fit`、`starmask.fit` | 在噪声未被非线性放大前拆出主体和星点层 | 默认用最终 `stage5_linear` 做 SyQon，`stage5_deconv` 仅作检查点/回退；失败可轻微预拉伸重试，再回退 SASP；全部失败则把当前 Stage 6 输入保存为 `starless.fit` 继续 |
| 主体拉伸 | Stage 7 / `stage7_stretched.fit` | 把 starless 主体拉到可视亮度，同时避免黑场压死和核心过曝 | 只比较 `stage7_cand_a` 与 `stage7_cand_b`，`stage7_preview_ref` 只供参考；候选需通过像素分布和质量门控 |
| Starless 增强 | Stage 8 / `stage8_enhanced.fit`、`starless_enhanced.fit` | 分区提升星云主体和外围弱信号，保护背景与亮核 | 优先接收外部/SASP starless；默认 soft mask 分区增强，必要时 conservative skip、蓝偏修正或回滚 |
| 回星合成 | Stage 9 / `stage9_remixed.fit` | 把星点按质量诊断受控回混，避免二次星点、halo 或坏蒙版污染 | 可处理外部 starmask；残星/halo 风险高时绕过 StarComposer，改用像素级 `starless + stretched_starmask * intensity` |
| 交付导出 | Stage 10 / `stage10_final.fit` | 生成可交付 TIFF/PNG/FITS | 按 `stage9_remixed -> starless_enhanced -> stage7_stretched` 选择最终图，做最终饱和度、降噪回退链和格式导出 |
| 可选 AI 副本 | Stage 11 / `result_processed_ai.*` | 在不覆盖主结果的前提下生成保守 AI 后期副本 | 只写 `*_ai`，先本地应用模型建议参数，再质量门控和降强度重试 |

## 3. 核心对象分工

### 3.1 `PipelineConfig`

`PipelineConfig` 是全部可调参数的集中入口，主要分为几类：

- 重试控制：`max_retries`、`retry_delay`
- 阶段 1 质量门控：`stage1_register_fail_ratio_max`
- 裁切：`crop_margin`
- 背景提取：`bg_samples`、`bg_tolerance`、`bg_smooth`、`bg_quality_gate_enabled`、`bg_std_worsen_ratio_max`、`bg_median_drop_ratio_min`、`bg_object_preserve_ratio_min`、`bg_edge_black_rise_max`、`bg_star_preserve_ratio_min`、`bg_nebula_mean_change_max`，以及 policy 中的 `stage3_background.max_bg_std_growth`
- 线性降噪：`denoise_enabled`、`denoise_mod`、`denoise_safety_max`
- 拉伸：`asinh_stretch`、`asinh_offset`、`ghs_shadowsclip`、`ghs_stretchamount`
- 星云增强：`nebula_saturation`、`nebula_bg_factor`
- 星点混合：`star_intensity`、`star_fallback_intensity`（`remix_safe_blend`、`remix_nebula_weight` 为旧配置兼容字段）
- 最终导出：`final_saturation`、`final_bg_factor`
- AI：`ai_post_enabled`、`ai_endpoint`、`ai_model`、`ai_api_key`、`ai_timeout_sec`、`ai_strength`、`ai_prompt`、`ai_stage6_enabled`、`ai_stage7_enabled`、`ai_stage8_enabled`
- AI 诊断与 Stage11 门控：`ai_bg_median_delta_max`、`ai_color_ratio_delta_max`、`ai_core_growth_ratio_max`、`ai_star_growth_ratio_max`，以及阶段 6-8 的背景/裁剪/星点/蓝偏/饱和度/微对比诊断阈值
- 运行行为：`checkpoint_mode`、`debug_mode`、`auto_tune_enabled`、`auto_tune_debug`
- 工作流插件控制：`workflow_plugin_probe_enabled`（默认 `False`，仅在显式启用时探测插件命令）、`spcc_enabled`（默认 `True`，可通过 `SEESTAR_SPCC_ENABLE=0` 禁用）、`aberration_api_enabled`（默认 `False`，API 路径在 siril-cli 线程所有权场景下可能失败）、`optional_color_transform_enabled`（默认 `False`，启用时阶段 8/9 尝试可选调色插件）
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

### 3.4 Target Profile / Pipeline Policy

Stage 3/4 preflight 会生成运行时目标画像和策略：

- `target_profile.json`：目标名称猜测、置信度、target type、风险标记和识别来源
- `pipeline_policy.json`：阶段 3-6.5 使用的背景、校色、线性处理、拉伸候选和去星前门控策略
- 内置策略来自 `policy_selector.py`，配置文件来自 `pipeline/configs/policies/*.yaml`
- 若 profiler、metadata 或自适应特征读取失败，会回退为 `generic_low_snr_safe`

目标画像识别不再作为独立 Stage 2.5 记录到 stage summary；它合并为 Stage 3/4 preflight。Stage 3 背景提取前先生成初始 `target_profile.json` / `pipeline_policy.json`，Stage 4 在 `platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3` 后会再次基于 `stage4_psolved.fit` metadata 刷新目标画像，若 target type 或 policy 改变，会覆盖上述 JSON 并记录告警。

### 3.5 `PipelineLogger`

`PipelineLogger` 只是一个轻量日志器，负责：

- 统一打印 `DEBUG / INFO / WARN / ERROR`
- 为每个阶段记录起止和耗时
- 为最终汇总提供足够可读的控制台输出

### 3.6 插件脚本基础设施

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
- 边缘黑边明显：Stage 2 自动识别四边黑边和叠加伪影后裁切
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
link -> calibrate -debayer -> register -2pass -> seqapplyreg -filter-round=2.5k -> stack -> mirrorx_single
```

这里会先把本轮 `Light_` 文件隔离到 `process/_light_input/`，统一命名为 `lightsrc_00001.fit` 这种格式，目的是避免历史序列和大小写差异干扰 Siril 的 `link` / `stack`。
叠加输出在本项目中固定为 `working.fit`，因此 vendor Seestar 脚本中的 `mirrorx_single result` 在这里对应为 `mirrorx_single working`。

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

1. 获取当前图像像素和尺寸
2. 基于中心背景亮度估计黑边阈值，沿左/右/上/下使用完整列/行扫描近黑像素比例、暗边界、红/蓝色偏叠加伪影，以及平滑后的边缘背景亮度台阶
3. 按四条边独立计算裁切量并执行 `crop`，不再先按 `crop_margin` 做固定比例裁切
4. 测量 `edge_black_ratio`，若仍高于阶段 2 目标阈值，会继续在阶段 2 内复用自动边缘识别迭代补裁并保存到 `stage2_corrected.fit`；后续阶段只诊断黑边，不再临时裁切去星输入
5. 若黑边裁切后阶段仍正常，会再次比较边缘窄条与中心区域的色偏/chroma 分数；当左/右/上/下边缘存在明显红蓝彩色伪影时，最多裁掉每侧约 2.5% 的窄边，并拒绝会让宽高损失超过 10% 的彩色边缘裁切

回退逻辑：

- 裁切失败时阶段记为 `degraded`
- 无法获取图像尺寸或图像太小时跳过裁切
- 彩色边缘裁切命令失败时阶段记为 `degraded`；检测异常只记录跳过，不阻断流程

阶段检查点：`stage2_corrected.fit`

阶段额外产物：

- `stage2_crop_report.json`：记录原始尺寸、最终尺寸、每次裁切的相对 `crop` 参数，以及相对原始图像累计裁掉的左/上/右/下像素；Stage 4 会读取该运行时记录用于说明解析图像已经是裁切后的视场

### 5.2.5 Stage 3/4 preflight：目标画像识别与策略选择

职责：在背景提取前识别目标类型，选择后续阶段的保守处理策略；Stage 4 解析后再用 `stage4_psolved.fit` metadata 刷新一次。

输入来源：

- 当前已裁切 FITS 像素特征
- `stage2_corrected` 或源文件 FITS header metadata
- 路径、工作目录和文件名上下文
- 自动调参识别出的旧版 `TargetType` hint
- `pipeline/configs/target_catalog/popular_dso.json` 中的常见目标坐标、别名、特征和默认 policy

处理逻辑：

1. 使用自适应图像特征构建 profile
2. 用 FITS metadata 的坐标与目标库匹配；大目标会使用更宽的视场容差，避免 Horsehead/IC 434 这类构图中心偏离目标质心的场景被误判
3. 结合 feature flags 做二次加权；`dark_nebula` 和 `low_contrast` 会映射到弱外围云气特征
4. 按 profile 选择 policy，并同步到运行时 `self.target_profile` / `self.pipeline_policy`
5. 写出 `target_profile.json`、`pipeline_policy.json`，可用时额外写出 `stage3_target_preview.png`

回退逻辑：

- profiler 依赖不可用、像素读取失败、metadata 异常或 policy 选择异常时，记录到 Stage 3/4 message，不再新增独立 degraded 阶段
- 回退 profile 使用 `generic_low_snr_safe`，后续阶段继续执行

阶段额外产物：

- `target_profile.json`
- `pipeline_policy.json`
- `stage3_target_preview.png`（可选）

### 5.3 阶段 3：背景提取

职责：消除背景梯度，优先保证背景平整，再考虑不误伤弱信号。

执行顺序：

1. 保存阶段 3 输入基线 `stage3_bg_input`，用于质量门控后回滚
2. 按理论背景建模效果构建统一候选链：`GraXpert/GXP -> GraXpert-BGE -> ADBE -> DBE -> AutoDBE -> subsky -rbf 参数组 -> NOX -> VeraLux NOX -> subsky 1 安全兜底`。当 `resources/siril_plugins/vendor/siril-scripts` 可用时，GraXpert 使用 `pyscript GraXpert-AI.py -bge`，ADBE/DBE/AutoDBE 使用不同参数的 `pyscript AutoBGE.py`，避免把不存在的 `gxp/adbe/dbe/autodbe` 当作 Siril 原生命令。
3. `subsky -rbf` 会基于当前噪声/星密度/目标覆盖动态生成多组参数变体；若 target profile 识别为大面积/弥散发射星云，或像素统计显示暗弱星云信号（默认 `object_area_ratio >= 0.15`、`nebulosity_area_ratio >= 0.18`、`faint_structure_score >= 0.65`；另有 `nebulosity_area_ratio > 0.10 && faint_structure_score > 0.40` 的 `faint_nebula_protection` 通路不依赖 `target_type`），`subsky 1` 会排在 RBF 前，降低 RBF 过拟合吞掉大尺度星云或暗弱边缘的风险
4. 每个候选都从 `stage3_bg_input` 基线重新加载后执行；若命令失败，直接 fallback 到下一个候选。GraXpert/GraXpert-BGE 会额外识别 `GraXpert-AI.py`、ONNX runtime、`too many indices for array` 等脚本运行时错误，记录为 `graxpert_runtime_error`，并立即触发 ADBE/DBE/subsky 等后续背景提取候选，不再把该次 GraXpert 输出进入质量评估
5. 每个候选成功后都会做背景质量门控（背景噪声、背景中值、目标覆盖、边缘黑场、星点保留率、星云/弥散信号均值变化）；满足门控且达到“充分质量”时立即完成 Stage 3
6. 通过门控但未达到充分质量的候选会保存为 `stage3_candidate_*.fit`，并按残余脏背景分数、低频梯度、背景彩噪、背景噪声增长、颜色偏移，以及星点/星云保留惩罚计算 `background_score`；暗弱星云信号越强，`nebula_mean_change_ratio` 的惩罚权重会从默认 `1.6` 线性提高到最高 `2.5`
7. 若候选不充分，自动回滚到基线并继续尝试下一个候选；所有候选都不充分但存在通过门控的候选时，最终回载 `background_score` 最低的候选作为 `stage3_bgremoved.fit`
8. 若全部候选命令失败或质量门控不通过，阶段记为 `degraded`

RBF 参数变体：脚本会以 `bg_samples` / `bg_tolerance` / `bg_smooth` 为基线，再读取当前 `bg_std`、`star_density`、`object_area_ratio` 动态扩展候选。高噪声时会降低 tolerance、提高 smooth（最高约 2 倍并受上限钳制）；复杂星场或大目标会增加候选数量；低噪声场景保留更细的低 smooth 变体。当前候选数量按复杂度在 3-5 组之间去重生成。

新增保留门控：

- `star_retention_ratio = after_star_count / before_star_count`，当可测星点数足够时必须不低于 `bg_star_preserve_ratio_min`（默认 0.90）
- `nebula_mean_change_ratio = abs(after_mean - before_mean) / before_mean`，当可测弥散/星云区域足够时必须不高于 `bg_nebula_mean_change_max`（默认 0.10）
- `stage3_background.max_bg_std_growth` 同时用于候选是否“足够好”的提前停止判定；例如 `bright_nebula_hdr_conservative` 的 `1.03` 会比默认 `1.08` 更严格
- 这些指标会写入 `background_quality_report.json.attempts[].preservation`、`attempts[].nebula_preservation_penalty_weight`、`diffuse_nebula_context` 和 `selected_preservation`

阶段检查点：`stage3_bgremoved.fit`；Debug/保留中间产物时还可检查 `stage3_candidate_*.fit`

### 5.4 阶段 4：图像解析 + 色彩校准

职责：执行天文定位（platesolve）并完成线性阶段的色彩平衡。

当前版本阶段 4 拆分为解析检查点与校色检查点：`stage3_bgremoved.fit -> stage4_psolved.fit -> stage4_color.fit`。解析命令固定使用 `platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3`，对应 Seestar S30 Pro 远摄光路：160mm 焦距、Sony IMX585、2.9um 像元。Stage 2 裁边只改变当前图像宽高和视场范围，不改变焦距或像元尺寸；Stage 4 会把 Stage 2 累计裁切和裁切后视场写入 `color_calibration_report.json`。

顺序：

1. `load stage3_bgremoved`
2. 基于当前裁切后图像记录 Stage 4 几何：S30 Pro / Sony IMX585 / 160mm / 2.9um，以及 Stage 2 累计裁边
3. 默认使用 Gaia 星表做一次解析：`platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3`；如需多星表候选，可通过 `SEESTAR_STAGE4_PLATESOLVE_CATALOGS` 显式覆盖
4. 若普通解析候选全部失败且 FITS header 有 `RA/DEC`、`OBJCTRA/OBJCTDEC` 或 `CRVAL1/CRVAL2`，会追加一轮 header-center platesolve 候选：`platesolve <ra>,<dec> -focal=160 -pixelsize=2.90 -catalog=<catalog> -order=3`；可通过 `SEESTAR_STAGE4_PLATESOLVE_HEADER_RADIUS` 额外传入 `-radius=...`
5. `save stage4_psolved`
6. Stage4 preflight 刷新 `target_profile.json` 和 `pipeline_policy.json`
7. 若 platesolve 成功，OSC 默认使用 Siril 数据库项 `spcc "-oscsensor=Sony IMX585" "-oscfilter=ZWO Seestar LP" "-whiteref=Average Spiral Galaxy" -limitmag=10.5`；若 FITS header/path/profile 或显式配置显示无 LP 滤镜，则使用 `"-oscfilter=No filter"`；若显示 Ha/OIII 窄带或双窄带，则不传 `oscfilter`，改用 `spcc "-oscsensor=Sony IMX585" "-whiteref=Average Spiral Galaxy" -narrowband -rwl=656.28 -rbw=20 -gwl=500.70 -gbw=30 -bwl=500.70 -bbw=30 -limitmag=10.5`；SPCC sensor/filter/white reference 固定整段加引号；默认 bgtol 不显式传入
8. Mono/LRGB：改用 `-monosensor`、`-rfilter`、`-gfilter`、`-bfilter`
9. 若 SPCC 失败：按 `SEESTAR_STAGE4_PCC_CATALOGS` 依次尝试 PCC，默认 `localgaia,gaia,nomad,apass`；PCC 使用 Siril 默认 bgtol（默认即 -2.8/+2.0），避免 Siril 1.4.x CLI 对显式负数 tuple 参数报 invalid argument
10. 若 platesolve 仍失败但 FITS header 有中心坐标，且 `SEESTAR_STAGE4_PCC_HEADER_FALLBACK_ENABLE=1`，会尝试同一组 PCC catalog 作为 header-coordinate fallback；当前 Siril 1.4.x `pcc` 文档仍要求 WCS，失败会记录后继续本地回退
11. 若 PCC 失败：执行本地保守回退；先做背景中性化；背景采样默认使用 luminance Q5-Q45，但会根据 `target_profile.object_area_ratio` 动态收窄，大面积目标收窄到 Q5-Q25，target-aware 星云还会把 `nebulosity_area_ratio` 纳入有效覆盖率；非 target-aware 目标在未饱和、低色差、中等亮度星点样本足够时再做星点白平衡；发射/双窄带 target-aware 目标默认跳过本地星点白平衡以保护 Hα/OIII 色彩
12. `save stage4_color`，并保留兼容别名 `stage4_colorbalanced`

SPCC 运行时保护：

- 可通过 `SEESTAR_SPCC_ENABLE=0` 全局禁用 SPCC
- 可通过 `SEESTAR_STAGE4_PLATESOLVE_ENABLE=0` 显式关闭阶段 4 platesolve；默认开启
- 可通过 `SEESTAR_STAGE4_PLATESOLVE_CATALOGS` 调整解析星表候选顺序，默认 `gaia`；`SEESTAR_STAGE4_PLATESOLVE_ORDER` 默认 `3`
- 可通过 `SEESTAR_STAGE4_PCC_CATALOGS` 调整 PCC 星表候选顺序，默认 `localgaia,gaia,nomad,apass`
- `SEESTAR_STAGE4_PCC_HEADER_FALLBACK_ENABLE=1` 时，platesolve 失败且 FITS header 有中心坐标会尝试 PCC header fallback；该路径是兼容性尝试，失败不阻断本地回退
- OSC/SPCC 默认使用 `Sony IMX585` + `ZWO Seestar LP` + `Average Spiral Galaxy`。`SEESTAR_STAGE4_SPCC_OSC_FILTER=No filter` 或 header/path/profile 的 `No filter` / `clear` / `no lp` 线索会改用 `Sony IMX585` + `No filter`；`Ha+OIII` / `narrowband` / `dualband` / `L-eXtreme` 等线索会改用 Siril `-narrowband` 参数组，不再传 `oscfilter`
- `SEESTAR_STAGE4_SPCC_LIMITMAG` 默认 `10.5`，限制 Gaia SPCC 星表查询星等，避免超宽视场一次性对数千颗星做 aperture photometry
- 每次执行 `spcc` 前都会先运行 Siril 1.4.x 支持的 `setcpu 1`，SPCC 成功或失败后都会尝试恢复到 `SEESTAR_STAGE4_SPCC_RESTORE_CPU`；默认 `0` 表示恢复到当前机器 CPU 数
- S30 Pro 默认 LP 模式的硬兼容命令形态为：`spcc "-oscsensor=Sony IMX585" "-oscfilter=ZWO Seestar LP" "-whiteref=Average Spiral Galaxy" -limitmag=10.5`；该 sensor/filter 均为 Siril SPCC 数据库精确项，避免旧的 `Optolong L-eNhance` 映射进入 Siril `get_spectrum_from_args` 空谱崩溃路径
- GUI 写入临时 Siril 配置时会清空旧版 `catalogue_gaia_photo=/.../gaia_photometric.dat` 文件路径；该项在 Siril 1.4.x 中应为有效 SPCC Gaia 目录或留空，错误文件路径会导致 CLI 目录访问警告
- SPCC 白参考默认是 `Average Spiral Galaxy`；`SEESTAR_STAGE4_SPCC_ADAPTIVE_WHITE_REF=1` 时，发射星云/双窄带目标可自动改用 `SEESTAR_STAGE4_SPCC_NEBULA_WHITE_REF`，但默认关闭以匹配 Siril SPCC 常规方案
- SPCC 要求当前图像已 platesolve；PCC 正常路径也要求 WCS。若 Stage 4 解析失败或被禁用，会跳过 SPCC，只在 header 坐标存在且 header fallback 开启时尝试 PCC，失败后改走本地背景中性化/星点白平衡，避免 Siril 报 `Command requires plate-solved image`
- 发射星云/双窄带目标启用 target-aware color mapping 后，若该白参考下 SPCC 失败，不再回退到普通星系白参考；后续改走 PCC 或本地背景中性化/星点白平衡
- `SEESTAR_STAGE4_SPCC_WHITE_REF` 显式设置时优先级最高；`SEESTAR_STAGE4_SPCC_NEBULA_WHITE_REF` 可调整星云目标白参考；取值必须与 Siril `spcc_list whiteref` / SPCC 下拉列表完全一致；`SEESTAR_STAGE4_SPCC_ADAPTIVE_WHITE_REF=0` 可关闭目标类型自动切换
- 可通过 `SEESTAR_STAGE4_SPCC_SENSOR_MODE=mono_lrgb` 切换 Mono/LRGB，并配置 `SEESTAR_STAGE4_SPCC_MONO_SENSOR`、`SEESTAR_STAGE4_SPCC_R_FILTER`、`SEESTAR_STAGE4_SPCC_G_FILTER`、`SEESTAR_STAGE4_SPCC_B_FILTER`
- 当输入来自 `Light_` 预处理模式时，默认仍尝试 `spcc`，因为 Stage 4 的 SPCC/PCC 是线性校色关键步骤
- 如需规避特定 `siril-cli` 环境下的 SPCC 崩溃风险，可设置 `SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS=0`，此时直接走 `pcc -> 本地保守回退`
- GUI 运行时若检测到 `siril-cli` 在 SPCC 阶段以 `-11` 退出，会自动禁用 SPCC 重试一次完整流水线，使 Stage 4 改走 PCC/本地校色回退，而不是让整次处理失败

回退逻辑：

- 若 `platesolve_ok=False`，不会提前中断；会先尝试 header 坐标辅助解析 / PCC header fallback，再进入本地保守回退
- `spcc/pcc` 任一步失败都会记录具体原因
- 本地背景中性化的采样窗口会写入 `color_calibration_report.json.local_fallback.background_neutralization.sampling_window`：普通目标默认 Q5-Q45；`object_area_ratio` 或 target-aware `nebulosity_area_ratio` 较大时逐步收窄，覆盖率约 35% 以上时使用 Q5-Q25，减少星云/星系主体进入背景样本
- 若星点样本足够且不是 target-aware 发射/双窄带目标，本地回退为 `LOCAL_STAR_WB`：背景中性化 + 星点白平衡；采样下限默认 32 像素，并使用多档亮度/chroma 放宽和迭代 sigma-clip
- 若星点样本不足，本地回退为 `BACKGROUND_NEUTRALIZATION`：只做背景中性化，不做固定增强型 CCM
- target-aware 发射/双窄带目标本地回退默认为 `BACKGROUND_NEUTRALIZATION`，跳过星点白平衡以保护 Hα/OIII 色彩；如需强制启用可设 `SEESTAR_STAGE4_LOCAL_STAR_WB_TARGET_AWARE_ENABLE=1`
- 阶段结果 `message` 会包含 `platesolve/SPCC/PCC/local_fallback` 的失败或回退信息

阶段检查点：`stage4_psolved.fit`、`stage4_color.fit`（兼容别名：`stage4_colorbalanced.fit`）

### 5.5 阶段 5：线性反卷积 / 轻降噪

职责：在线性图像上先用未降噪的星点建立 PSF 并执行受控 RL 反卷积，再做轻量线性降噪，为后续线性去星提供保留微反差且背景不过度放大的输入。

逻辑要点：

- 输入固定优先 `stage4_color.fit`，阶段开始保存 `stage5_input_linear.fit`
- 默认 Siril 内核链路：`findstar -maxstars=200` -> `makepsf stars -sym -ks=33 -savepsf=stage5_psf.fit` -> `rl -loadpsf=stage5_psf.fit -iters=8 -alpha=3000 -tv -gdstep=0.0005 -stop=0.001` -> `save stage5_deconv` -> `denoise -mod=0.50 -indep` -> `save stage5_linear`
- `result_linear.fit` 在 `stage5_linear` 保存后导出，供 GUI 显式续跑模式使用
- Stage5 不再默认执行全局锐化、unsharp 或星点矫正，避免在线性暗背景中提前放大彩噪、星环和 halo
- CosmicClarity Denoise 只作为 Siril 内置降噪失败后的回退，强度映射限制在 `0.20~0.30`；`chroma_first` 使用 `0.30`，`luma_chroma_balanced` 使用 `0.25`
- GraXpert 反卷积只允许映射到 Stage5，建议强度 `0.20~0.40`；当前代码只记录配置占位，不在未确认本机工具调用协议时自动执行
- 若 RL 后背景指标明显变差，背景保护会丢弃 `stage5_deconv`，重新回到 `stage5_input_linear` 后再执行轻降噪；因此 `stage5_linear.fit` 始终是 Stage 5 的最终线性输出

当前 Stage 5 不执行可选转色。`optional_color_transform_enabled=True` 的可选颜色插件只在 Stage 8/9 使用，避免在线性阶段提前改变窄带/双窄带颜色语义。

阶段检查点：`stage5_linear.fit` 是最终线性输出；可选 `stage5_deconv.fit` 是 RL 后、降噪前检查点；兼容别名仍保存 `stage5_denoised.fit`

阶段额外产物：`result_linear.fit`（位于工作目录根目录，默认不会被下一轮输入扫描命中；可供 GUI 显式续跑模式继续执行阶段 6-11）

诊断输出：`stage5_linear_report.json` 会记录处理顺序、输入/输出、Siril 降噪参数、CosmicClarity 回退强度、RL 反卷积参数与 `background_guard` 的 before/after 自适应背景指标。

### 5.6 阶段 6：去星与星点层准备

职责：调用去星工具生成去星图，并尽量构造一个可控的星点层。当前默认在阶段 7 拉伸前执行，优先使用 `stage5_linear.fit`；`stage5_deconv.fit` 仅在最终线性输出缺失时作为回退检查点；`result_linear.fit` 续跑时会使用续跑加载得到的线性检查点。收益是避免非线性拉伸把低频脏背景、色斑和噪声放大后再交给去星模型；副作用是星点层也来自线性图，后续阶段 9 需要通过星点拉伸/回混强度控制恢复星点亮度。

星点分离输入模式：

- `linear_star_separation`：默认模式，直接用 Stage 5 线性结果去星，最能减少拉伸后星点膨胀、星色丢失和背景伪影。
- `mild_prestretch_star_separation`：仅为去星模型生成 `stage6_mild_prestretch_star_input.fit`，用轻微 Asinh 预拉伸帮助模型识别弱星；后续仍分别处理 starless 与 starmask，不把该预拉伸当作最终主体拉伸。
- 若线性 SyQon 初次失败，或虽然成功但质量门控发现 `starless` 动态范围塌缩（相对输入范围过低且峰值信号过低），且 `star_separation_fallback_to_mild_prestretch=True` / 保守重试开启，会自动生成轻微/保守预拉伸输入重试；失败原因、动态范围指标和最终 source 写入 `stage6_starless_quality.json` / `stage7_quality.json`。

说明：阶段 6 会导出新命名产物 `stage6_starless.fit`、`stage6_starless_quality.json`；同时保留旧别名 `stage7_starless.fit`、`stage7_quality.json`，用于兼容外部工具、续跑和旧报告。

去星优先链：

1. `SyQon-Starless.py` CLI 子进程（默认 Zenith v1 模型，tile-size=512，overlap=64；`SEESTAR_SYQON_GPU=0` 时追加 `--no_gpu`；带超时与日志转发）
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
- 每次得到 `starless.fit` / `starmask.fit` 后生成一次质量记录，最终写入 `process/stage6_starless_quality.json`，并保留兼容别名 `process/stage7_quality.json`
- 本地指标先判断 `starless` 残星、动态范围塌缩、低峰值信号、`starmask` 缺星和 `starmask` 过宽，再把指标发给 AI 做保守裁判
- 若残星、动态范围塌缩、缺星或蒙版过宽指标偏差，会在 `SEESTAR_STAGE7_QUALITY_RETRY_MAX` 预算内用其他 SyQon 参数组合或保守输入重跑，并把质量分最低的 `starless.fit` / `starmask.fit` 恢复为最终阶段产物
- 若最终 `residual_star_score` 仍高于阈值，AI 可返回 `residual_suppression_strength` 和 `stage9_star_intensity_scale`；脚本会做安全限幅后结合 `starmask` 与拉伸层 residual map 实际修改 `starless.fit` 抑制残星，并把最终阶段 9 回混缩放写入 `stage6_starless_quality.json.stage9_star_remix`
- 质量诊断不再把成功的 SyQon/SASP 结果直接降级为拉伸检查点；只有所有去星工具失败才走退化路径

回退逻辑：

- 如果所有去星命令都失败，阶段记为 `degraded`
- 同时把当前 Stage 6 输入检查点（通常是 `stage5_linear`、`stage5_deconv` 或轻微预拉伸输入）重新保存成 `starless.fit`，让阶段 7-10 仍然可以继续执行
- 因此去星工具不可用时，脚本退化为"不做真正去星"的流程

阶段检查点：`stage6_starless.fit`（兼容别名：`stage7_starless.fit`）

阶段输出：

- `starless.fit`
- 可能存在的 `starmask.fit`
- `sasp_starless_input.fit`（工作目录，供外部工具使用）
- `sasp_starmask_input.fit`（工作目录，供外部工具使用）

### 5.7 阶段 7：主体拉伸

职责：把阶段 6 生成的 starless 图像变成可视化的非线性图像。当前固定 I/O 为 `stage6_starless.fit -> stage7_stretched.fit`。

说明：阶段 7 只使用 Stage7 命名，不再把主体拉伸候选同时写成 `stage6_candidate_*` 与 `stage7_candidate_*`。

候选收缩为固定两条，另保留一个 preview 参考；候选名称固定，但参数会根据 `stage6_starless` 的 baseline 背景自适应：

1. `stage7_cand_a.fit`: `load stage6_starless` -> `asinh <adaptive_stretch> <adaptive_offset>`；普通背景默认 `2.2 / 0.002`
2. `stage7_cand_b.fit`: `load stage6_starless` -> `asinh <adaptive_stretch> <adaptive_offset>` -> `autoghs -linked -2.1 <adaptive_amount>`；普通背景默认 `2.1 / 0.002 / 1.05`
3. `stage7_preview_ref.fit`: `load stage6_starless` -> `autostretch -linked`，仅作参考，不允许成为最终图

关键行为：

- 当 baseline `bg_median <= 0.005` 时启用极低背景保护：降低 Asinh 强度并提高 offset；若 `stage6_starless` 的 `p99/max` 很低，会把 offset 重新限制到有效信号范围以下，避免 offset 高于整图信号导致候选全黑
- 当 baseline `0.005 < bg_median < 0.010` 时使用中等低背景保护，参数在默认值与极低背景值之间平滑过渡
- 候选像素分布若命中 `is_nearly_black`、`is_visibility_too_low`、`is_nearly_white` 或 `invalid_dynamic_range`，会被标记为非正常最终候选
- 若两个正式候选均未通过质量门控，但至少有一个候选成功保存，会选择风险较低者作为 degraded fallback
- 无论拉伸是否成功，最终只保存 `stage7_stretched.fit`
- 如果所有拉伸方法都失败，阶段状态记为 `failed`
- 但 `run()` 不会因此立即退出，后续阶段仍会继续尝试使用这个检查点
- 诊断输出 `stage7_stretch_quality.json` 和 `stretch_candidates_report.json`，记录两个正式候选、唯一 preview、`baseline_pixel_stats`、`stretch_adaptation`、候选像素分布和最终选择

这意味着阶段 7 的 `failed` 更像"拉伸目标未达成"，而不是"整条链路终止"。

### 5.7.5 阶段 7.5：兼容门控记录

职责：保留旧流程的去星前质量门控兼容点。当前默认 starless-first 顺序已经在阶段 7 拉伸前完成阶段 6 去星，因此主流程会把阶段 7.5 记录为 `skipped`，消息为 `starless-first mode: Stage6 already separated stars before Stage7 stretch`。

旧的非默认/调试路径仍可使用以下逻辑：在调用 SyQon/SASP 前判断当前拉伸图是否适合去星，并选择更安全的 starless 输入。

处理逻辑：

1. 以 `stage7_stretched` 或当前 `stretched_name` 为源读取自适应特征
2. 合并当前质量指标：黑场比例、高光裁剪、星点尺寸、边缘黑边等
3. 使用 `target_profile` 和 `pipeline_policy.stage6_5_pre_starless_gate` 评估 `ready_for_starless`
4. 当 policy 推荐 `stage7_conservative_asinh` 或 `stage7_ultra_conservative_asinh` 时，生成更保守的去星输入候选
5. 若推荐文件缺失，会回退到可用的保守输入或原始阶段 7 拉伸检查点

阶段输出：

- `stage7_5_pre_starless_gate_report.json`（兼容别名：`pre_starless_gate_report.json`）
- 可能生成 `stage7_conservative_asinh.fit` 或 `stage7_ultra_conservative_asinh.fit`

回退逻辑：

- 评估模块不可用或指标读取失败时，阶段记为 `degraded`
- 回退 report 会设置 `ready_for_starless=true`，推荐输入为阶段 7 拉伸检查点，避免门控故障阻断后续阶段

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
2. 上一阶段 Starless 图像的像素级 `starmask_stretched` 回混
3. 使用备用星点强度再次回混
4. 放弃回星，直接使用无星结果

默认回混公式：

```text
starmask_stretched = asinh(starmask, stretch=2.00, offset=0.001)
final = starless_enhanced + starmask_stretched * intensity
```

若 `SASP Star Stretch` / `NB to RGB Stars` 已经成功处理星点层，阶段 9 会把插件处理后的星点层另存为 `starmask_stretched.fit`，避免二次拉伸；否则使用内置温和 Asinh。这样避免把线性域 starmask 直接加到已经拉伸的 `starless_enhanced` 上，减少星点过暗或被星云背景吞没的问题。可通过 `SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE=0` 关闭，或用 `SEESTAR_STAGE9_STARMASK_ASINH_STRETCH` / `SEESTAR_STAGE9_STARMASK_ASINH_OFFSET` 调整。

Stage7 联动：

- 若 `stage6_starless_quality.json`（兼容别名：`stage7_quality.json`）显示残星超标，阶段 9 会跳过不可控的 StarComposer，改走像素级回混
- 实际 `intensity` 和备用强度会乘以 Stage7 计算出的 `intensity_scale`，降低残星与回混星点叠加造成的二次星点风险

设计意图：

- 阶段 9 的主处理文件必须来自阶段 8 的 `starless_enhanced`
- 不再用旧拉伸检查点与 `starless_enhanced` 做安全混合，避免 `pm` 多文件表达式成功返回但实际保存回拉伸阶段图像
- 合成后同时记录 `stage9_remixed` 对 `stage8_enhanced` 与阶段 7 拉伸检查点的差异，便于排查阶段 9 是否真正生效

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
3. `stage7_stretched`

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

然后切回原工作目录导出三类文件；当前导出顺序为 TIFF、FITS、PNG：

- TIFF：优先使用基于 FITS 元数据的动态命名，失败时回退 `result_processed.tif`
- PNG：优先动态命名，失败时回退 `result_processed.png`
- FITS：优先动态命名并追加 `_final`，失败时回退 `result_final.fit`

若本轮输入模式是 `result_linear_resume`：

- 主输出会统一追加 `_linear` 后缀，避免覆盖原始完整流程产物
- TIFF fallback 改为 `result_processed_linear.tif`
- PNG fallback 改为 `result_processed_linear.png`
- FITS fallback 改为 `result_final_linear.fit`

若本轮输入模式是 `stage2_corrected_resume`，输出命名沿用完整流程，不追加 `_linear` 后缀。

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

项目默认 env 会先从 `resources/default.env` 读入，再叠加 `resources/ai.env` 和当前目录 `.seestar_ai.env`；已有进程环境变量不会被覆盖。GUI 运行时额外读取 App Resources、runtime home 和工作目录的 `.seestar_ai.env`，然后再强制写入界面开关对应的 Debug、输入模式和 AI 开关。

`_apply_runtime_env_overrides()` 在 `run()` 入口处（连接 Siril 之前）集中读取并覆盖配置：

| 环境变量 | 对应配置 | 默认值 | 说明 |
|---|---|---|---|
| `SEESTAR_DEBUG_MODE` | `debug_mode` | `False` | 开启后保留 stage* 中间文件 |
| `SEESTAR_INPUT_MODE` | `input_mode` | `auto` | `auto` 正常流程；`stage2_corrected_resume` 从 `stage2_corrected.fit` 进入 stage 3；`result_linear_resume` 从 `result_linear.fit` 进入 stage 6 |
| `SEESTAR_OUTPUT_FORMAT` | `output_format` | `all` | 最终导出格式，可为 `all` 或逗号分隔 `tif/png/fit` |
| `SEESTAR_STAR_SEPARATION_MODE` | `star_separation_mode` | `linear_star_separation` | 阶段 6 去星输入模式；可设 `mild_prestretch_star_separation` |
| `SEESTAR_MILD_PRESTRETCH_STRENGTH` | `mild_prestretch_strength` | `1.35` | 轻微预拉伸去星强度，限幅 `1.05~1.80` |
| `SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH` | `star_separation_fallback_to_mild_prestretch` | `True` | 线性去星失败时是否用轻微预拉伸输入重试 |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` | `workflow_plugin_probe_enabled` | `False` | 启用后允许 `run_first_available_command` 路径探测更广泛的实验插件命令；Stage 6 的 SASP Dark Star fallback 可在关闭时运行，Stage 8 固定使用 SASP Python API，不探测未注册的实验性 `sasp_*` Siril 命令 |
| `SEESTAR_SPCC_ENABLE` | `spcc_enabled` | `True` | 设为 0 可全局禁用 SPCC |
| `SEESTAR_STAGE4_PLATESOLVE_ENABLE` | `stage4_platesolve_enabled` | `True` | 默认执行 `platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3` |
| `SEESTAR_STAGE4_PLATESOLVE_CATALOGS` / `SEESTAR_STAGE4_PLATESOLVE_ORDER` | env only | `gaia` / `3` | Stage 4 platesolve 星表候选顺序和多项式阶数 |
| `SEESTAR_STAGE4_PCC_CATALOGS` / `SEESTAR_STAGE4_PCC_HEADER_FALLBACK_ENABLE` | env only / `stage4_pcc_header_fallback_enabled` | `localgaia,gaia,nomad,apass` / `True` | PCC 星表候选顺序；platesolve 失败但 header 有坐标时允许尝试 PCC header fallback |
| `SEESTAR_STAGE4_SPCC_SENSOR_MODE` | `stage4_spcc_sensor_mode` | `osc` | SPCC 传感器模式：`osc` 或 `mono_lrgb` |
| `SEESTAR_STAGE4_SPCC_OSC_SENSOR` / `SEESTAR_STAGE4_SPCC_OSC_FILTER` | `stage4_spcc_osc_sensor` / `stage4_spcc_osc_filter` | `Sony IMX585` / `""` | OSC SPCC sensor/filter；filter 为空时默认 `ZWO Seestar LP`；可显式设 `No filter`；窄带/双窄带线索会改用 `-narrowband` 参数组 |
| `SEESTAR_STAGE4_SPCC_BUILTIN_DUALBAND_FILTER` | `stage4_spcc_builtin_dualband_filter_enabled` | `False` | 兼容旧配置；开启时按默认 LP 光害滤镜使用 `ZWO Seestar LP` |
| `SEESTAR_STAGE4_SPCC_*_FILTER` | `stage4_spcc_*_filter` | `""` | Mono/LRGB R/G/B filter 配置 |
| `SEESTAR_STAGE4_SPCC_WHITE_REF` / `SEESTAR_STAGE4_SPCC_BGTOL` / `SEESTAR_STAGE4_SPCC_LIMITMAG` / `SEESTAR_STAGE4_SPCC_RESTORE_CPU` | `stage4_spcc_white_ref` / `stage4_spcc_bgtol` / `stage4_spcc_limitmag` / `stage4_spcc_restore_cpu` | `Average Spiral Galaxy` / `-2.8,2.0` / `10.5` / `0` | SPCC 参数；默认 bgtol 不显式传入，非默认值才传；PCC 固定使用 Siril 默认 bgtol；SPCC 前强制 `setcpu 1`，结束后恢复，0 表示恢复到 CPU 数 |
| `SEESTAR_STAGE4_SPCC_ADAPTIVE_WHITE_REF` / `SEESTAR_STAGE4_SPCC_NEBULA_WHITE_REF` | `stage4_spcc_adaptive_white_ref_enabled` / `stage4_spcc_nebula_white_ref` | `False` / `Star, type G2(v)` | 可选的发射星云目标 SPCC 白参考自动切换，默认关闭 |
| `SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE` | `stage4_local_star_wb_enabled` | `True` | SPCC/PCC 均失败时是否允许本地星点白平衡回退 |
| `SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS` / `SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT` | `stage4_local_star_wb_min_pixels` / `stage4_local_star_wb_gain_limit` | `32` / `1.25` | 本地星点白平衡样本下限与单通道增益上限 |
| `SEESTAR_STAGE4_LOCAL_STAR_WB_TARGET_AWARE_ENABLE` | `stage4_local_star_wb_target_aware_enabled` | `False` | 发射/双窄带 target-aware 目标是否允许本地星点白平衡；默认关闭以保护窄带色彩 |
| `SEESTAR_ABERRATION_API_ENABLE` | `aberration_api_enabled` | `False` | 启用 SASP Aberration API 路径 |
| `SEESTAR_ABERRATION_PROVIDER` | — | — | SASP Aberration API provider 选择；可设 `cpu` 强制 CPU |
| `SEESTAR_DENOISE_ENABLE` | `denoise_enabled` | `False` | 启用内置线性降噪 |
| `SEESTAR_DENOISE_FORCE` | `_force_denoise_enabled` | — | 自动调参后强制覆盖 denoise_enabled |
| `SEESTAR_AI_ENABLED` | `ai_post_enabled` | `False` | AI 总开关：启用阶段 6 去星、阶段 8 增强诊断/参数建议与阶段 11 AI 副本；阶段 7 拉伸当前仍固定本地两候选 |
| `SEESTAR_AI_ENDPOINT` | `ai_endpoint` | `""` | AI API endpoint |
| `SEESTAR_AI_MODEL` | `ai_model` | `""` | AI 模型名 |
| `SEESTAR_AI_API_KEY` | `ai_api_key` | `""` | AI API 密钥 |
| `SEESTAR_AI_PROMPT` | `ai_prompt` | `""` | 自定义 AI 提示词 |
| `SEESTAR_AI_TIMEOUT_SEC` | `ai_timeout_sec` | `90` | API 超时（限幅 15–300） |
| `SEESTAR_AI_STRENGTH` | `ai_strength` | `0.12` | AI 混合强度（限幅 0.05–0.25） |
| `SEESTAR_AI_STAGE6_ENABLE` | `ai_stage6_enabled` | `True` | 保留的拉伸顾问开关；当前阶段 7 主体拉伸固定使用本地两候选，不扩展 AI 候选 |
| `SEESTAR_AI_STAGE7_ENABLE` | `ai_stage7_enabled` | `True` | 单独控制阶段 7 AI SyQon 参数计划、诊断与重试择优 |
| `SEESTAR_AI_STAGE8_ENABLE` | `ai_stage8_enabled` | `True` | 单独控制阶段 8 AI Starless 参数计划、蓝色修正与保守重跑 |
| `SEESTAR_STAGE7_QUALITY_RETRY_MAX` | `stage7_quality_retry_max` | `2` | 阶段 6 去星参数优化预算配置（限幅 0–3）；变量名保留 `stage7` 以兼容旧配置 |
| `SEESTAR_STAGE7_SKIP_UNREADY_STARLESS` | `stage7_skip_unready_starless` | `True` | 兼容门控判定不适合去星时是否跳过 SyQon/SASP |
| `SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH` | `stage7_soft_starless_asinh_stretch` | `1.35` | 阶段 6 质量差时追加更轻去星输入强度（限幅 1.05–ultra conservative） |
| `SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX` | `stage7_bright_nebula_halo_residue_score_max` | `0.60` | M42/亮核心星云 halo 验收上限 |
| `SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH` | `stage7_starless_repair_strength` | `0.68` | starless 小尺度残星修复强度（限幅 0–0.85） |
| `SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH` | `stage7_starless_halo_repair_strength` | `0.70` | starless 亮星 halo 修复强度（限幅 0–0.90） |
| `SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH` | `stage7_starless_chroma_denoise_strength` | `0.55` | starless 背景彩噪修复强度（限幅 0–0.90） |
| `SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE` | `stage7_starless_pixel_repair_enabled` | `True` | 是否启用阶段 6 starless 像素修复 |
| `SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR` | `stage8_force_conservative_after_stage7_repair` | `True` | 阶段 6 修复或不安全时是否强制 Stage 8 conservative skip |
| `SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE` | `stage9_starmask_stretch_enabled` | `True` | 阶段 9 像素回混前是否把 starmask 拉伸到非线性域 |
| `SEESTAR_STAGE9_STARMASK_ASINH_STRETCH` / `SEESTAR_STAGE9_STARMASK_ASINH_OFFSET` | `stage9_starmask_asinh_stretch` / `stage9_starmask_asinh_offset` | `2.00` / `0.001` | 阶段 9 starmask 温和 Asinh 拉伸参数 |
| `SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS` | — | `"1"` | 允许 Light_ 模式下尝试 SPCC；设为 `0` 时跳过 SPCC 直接走 PCC |
| `SEESTAR_SIRIL_PLUGIN_DIR` | `siril_plugin_dir` | — | 插件目录路径 |
| `SEESTAR_COSMIC_CLASSIC_ENABLE` | — | `0` | 是否启用 classic CosmicClarity executable 路径；默认使用 Native/SCUNet |
| `SEESTAR_COSMIC_CLARITY_EXECUTABLE` | — | — | classic CosmicClarity executable 路径 |
| `SEESTAR_COSMIC_CLASSIC_GPU` | — | `1` | classic CosmicClarity 是否允许 GPU/device auto；`0` 强制 `-no_gpu` |
| `SEESTAR_COSMIC_NATIVE_GPU` | — | `1` | CosmicClarity Native 是否允许 GPU/device auto；`0` 强制 CPU |
| `SEESTAR_SYQON_GPU` | — | `1` | SyQon Starless 是否允许 GPU backend；`0` 强制 CPU |
| `SEESTAR_SYQON_TIMEOUT_SEC` | — | `900` | SyQon Starless CLI 超时（限幅 60–1800） |
| `SEESTAR_SIRILPY_TIMEOUT_SEC` | — | GUI 默认 `120` | sirilpy/plugin 子进程基础超时；最终降噪 CLI 会在此基础上增加缓冲并限幅 |
| `SIRIL_PYTHON_CLI` | — | — | CLI 子进程使用的 Python 解释器 |
| `SEESTAR_SIRIL_PYTHON_CLI` | — | — | bundled wrapper 使用的稳定 Python 回退，避免第三方脚本污染 `SIRIL_PYTHON_CLI` |

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
- `target_profile.json`、`pipeline_policy.json`、`stage3_target_preview.png`（预览图可用时）
- `stage3_bg_input.fit`（质量门控基线）
- `stage3_bgremoved.fit`
- `stage4_psolved.fit`
- `stage4_color.fit`（兼容别名：`stage4_colorbalanced.fit`）
- `stage5_linear.fit`、`stage5_deconv.fit`（可选）、`stage5_denoised.fit`（兼容别名）
- `stage6_starless_quality.json`（兼容别名：`stage7_quality.json`）
- `stage6_starless.fit`（兼容别名：`stage7_starless.fit`）
- `stage7_conservative_asinh.fit`、`stage7_ultra_conservative_asinh.fit`（去星前门控需要时）
- `stage7_cand_a.fit`、`stage7_cand_b.fit`、`stage7_preview_ref.fit`
- `stage7_stretch_quality.json`、`stretch_candidates_report.json`
- `stage7_stretched.fit`
- `stage7_5_pre_starless_gate_report.json`（兼容别名：`pre_starless_gate_report.json`）
- `starless.fit`
- `stage8_input_starless.fit`、`stage8_quality.json`、`stage8_enhancement_report.json`
- `starless_enhanced.fit`
- `stage8_enhanced.fit`
- `starmask.fit`
- `starmask_stretched.fit`
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

Debug 模式下，每次 `_save_stage_output()` 成功保存阶段 FIT 后，会按统一格式输出质量指标：

- 日志行：`[STAGE_QUALITY_METRICS] schema=seestar.stage_quality.v1 stem=<stage_stem> ...`
- 单阶段文件：`process/<stage_stem>_quality_metrics.json`
- 汇总文件：`process/stage_quality_metrics.jsonl`

JSON 结构固定包含 `schema`、`sequence`、`stem`、`file`、`metrics`、`features`。`metrics` 使用 `QualityMetrics` 字段（背景、黑场、高光裁剪、星点、饱和度、微对比、蓝偏），`features` 使用 `ImageFeatures` 字段（边缘黑边、全局暗像素、目标面积、弥散比例、亮核比例等），便于后续比较阶段间变化。

### 8.3 清理阶段

`cleanup()` 在非调试模式下会：

- 先通过 `_cleanup_lightsrc_intermediates()` 清理 `process/` 内 `*lightsrc*` 相关文件和 `_light_input/` 目录
- 清理 `process/` 内大部分 `fit/fits/seq/log/csv/lst`
- 默认保留 `starless.fit`、`starmask.fit`
- 如果存在拉伸名，还保留对应的拉伸星点蒙版检查点
- 不会删除工作目录中的 `result_linear.fit`
- 阶段 11 临时文件会在非调试模式下由 `stage11_ai_postprocess.py` 主动删除
- 若阶段 11 成功，工作目录会新增 `result_processed_ai.tif/png` 与 `result_final_ai.fit`
- 若目录已空则直接删除 `process/`

因此，默认模式下最终更强调"输出整洁"，而不是保留完整现场。

## 9. 维护时最值得注意的点

1. 阶段顺序是核心契约：1-10 为稳定主链，阶段 11 仅可选追加，不可插到 1-10 中间。
2. 自动调参是"重写配置"，不是"给某个阶段附加参数"，排查参数异常时要先看自动调参日志。
3. Stage 3/4 preflight 的 `target_profile.json` / `pipeline_policy.json` 是 Stage 3-6.5 的策略输入；新增 target type 或候选时要同步内置 policy、配置文件和候选 evaluator。
4. Stage 7 候选不能只看命令成功，必须看 `quality_ok`、`pixel_stats` 和 `normal_selected`；degraded fallback 是可导出兜底，不是正常最佳结果。
5. `workflow_plugin_probe_enabled` 控制广泛插件探测开关：为 `False` 时大部分插件命令不会被尝试，只有标记 `allow_when_probe_disabled=True` 的命令（如 Stage 6 的 SASP Dark Star fallback）仍可执行；阶段 8 默认走 SASP Python API，不再先探测实验性 `sasp_*` Siril 命令。
6. 阶段 5、6、6.5、7、8、9、10、11 都有明显降级路径，改动时不能只看成功路径。
7. `process/` 每轮都会重建，任何依赖历史中间文件的思路都不成立。
8. `Light_` 输入路径使用隔离目录再 `link`，不要直接在工作目录对历史序列做操作。
9. 去星工具失败时并不会终止，而是退化为"把当前 Stage 6 输入当作 starless 继续"，这一点会直接影响阶段 7-10 的视觉预期。
10. 导出不是固定依赖 `stage9_remixed`，而是按 `stage9_remixed -> starless_enhanced -> stage7_stretched` 逐级回退。
11. SASP 交换文件（`sasp_starless_input.fit`、`sasp_starmask_input.fit`）和外部回写检测（`sasp_starless.fit` 等）构成了与 SetiAstroSuitePro 的双向数据通道。
12. 插件脚本通过 `_SCRIPT_PREREQUISITE_MODULES` 做前置检查，缺少模块时会跳过而不是抛出异常。

## 10. 快速排障索引

如果后续需要定位问题，可以优先按下面的思路查：

- 没有识别到输入：看阶段 1 的候选叠加过滤和 `Light_` 检测
- Light_ 配准质量差：看阶段 1 的 `_stage1_registration_stats` 和 `stage1_register_fail_ratio_max`
- 参数表现异常：看自动调参日志和 `AutoTuneResult.changed_params`
- 目标类型或策略不对：看 Stage 3/4 preflight 的 `target_profile.json`、`pipeline_policy.json`、FITS metadata source 和阶段 4 是否刷新过 profile
- 背景残留：看阶段 3 的 `background_quality_report.json`，确认 `builtin_order_reason`、`builtin_candidate_order`、最终 `model_used`，以及是否尝试了 GraXpert 补救
- 色彩偏差：看阶段 4 的 `platesolve_ok`、`spcc_enabled` 是否被禁用、`spcc_white_reference` 是否 target-aware、`pcc` 是失败还是被跳过，以及 `local_fallback` 是星点白平衡还是仅背景中性化
- Stage5 降噪无效：看 `stage5_linear_report.json` 的 `denoise.method`，确认 Siril `denoise -indep` 是否失败以及 CosmicClarity 低强度回退是否命中
- Stage5 反卷积无效：看 `stage5_linear_report.json` 的 `deconvolution.applied` 和日志中的 `findstar/makepsf/rl` 错误；失败时流程会回退使用 `stage5_linear.fit`
- 拉伸异常：看阶段 7 的 `stage7_stretch_quality.json` 和 `stretch_candidates_report.json`，确认 `stage7_cand_a` / `stage7_cand_b` 哪个被选中、是否为 degraded fallback，以及 `pixel_stats` 是否命中黑场/白场/动态范围门控
- 去星前直接跳过或输入过保守：看阶段 7.5 的 `stage7_5_pre_starless_gate_report.json`、`ready_for_starless` 和 `recommended_starless_input`
- 去星无效：看阶段 6 是 SyQon 成功、回退到 SASP Dark Star、还是已经退化成"直接保存当前 Stage 6 输入为 starless"
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
- 用 target profile / pipeline policy 把目标类型识别、背景保护、拉伸候选和去星前门控串起来
- 在输入类型、自动调参、去星回混和导出命名几个高风险点上做保守处理
- 通过 `workflow_plugin_probe_enabled` 开关控制插件探测范围，避免在缺少插件的环境中产生大量无意义的探测开销
- 通过 SASP 交换文件和外部回写检测实现与第三方工具的双向集成

因此，后续改动若涉及阶段顺序、自动调参、目标策略、Stage 7 候选评分、去星回退、导出优先级或插件探测逻辑，应视为高风险变更，必须同步检查对应的降级路径是否仍然成立。
