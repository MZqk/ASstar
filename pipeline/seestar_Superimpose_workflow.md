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
    L -.-> M[Stage 7 兼容检查点 非正式阶段/默认跳过]
    M --> N[stage8 Starless 深加工]
    N --> O[stage9 星点处理与合成]
    O --> P[stage10 最终降噪与导出]
    P --> Q[stage11 AI后期(可选)]
    Q --> T[AI艺术衍生实验(可选/非正式/只读Stage10)]
    T --> R[cleanup 清理]
    R --> S[汇总阶段结果]
```

其中：

- 正式阶段编号固定为 Stage 1-11，代码统一由 `models.PipelineStage` 提供阶段身份和显示标签
- AI 艺术衍生是 Stage 11 后的非正式隔离分支，不新增 Stage 12，也不改变正式阶段结果
- 阶段 1-6 视为线性阶段，目标画像与 policy 选择合并到 Stage 3/4 preflight，不再单独占一个阶段；`stage2_5_target_profiler()` 仅是兼容别名
- 阶段 6 是线性去星与星点层准备，先在线性图上分离星点；阶段 7 再拉伸 starless，避免拉伸放大的脏背景进入去星模型
- 主类正式入口为 `stage6_star_separation()` / `stage7_stretching()`，分别调用 `run_stage6_star_separation()` / `run_stage7_stretching()`；旧名称仅为兼容别名
- 因此阶段 7-10 是 starless-first 非线性核心处理
- 默认 starless-first 路径下只记录一个 Stage 7 兼容检查点（非正式阶段）；旧的去星前质量门控仍保留给非默认/调试路径，`stage6_5_pre_starless_gate()` 仅是兼容别名
- 阶段 6 去星、阶段 8 增强和阶段 11 可在 AI 总开关开启且凭据齐全时启用参数优化与诊断记录；当前阶段 7 主体拉伸固为本地两个主候选，AI 或 policy 不扩展候选集合，仅允许在背景色度门控失败时追加确定性救援候选
- 自动调参发生在阶段 1 完成之后、阶段 2 开始之前
- 脚本不会因为单个非致命阶段失败就立刻退出，而是尽量记录 `ok / degraded / failed / skipped` 后继续处理
- 当 `SEESTAR_INPUT_MODE=stage2_corrected_resume` 时，会使用工作目录根下或旧 `process/` 下的 `stage2_corrected.fit` 作为已完成裁切/视场修正的叠加后中间结果：
  `prepare stage2_corrected.fit -> auto tune -> stage3 background -> stage4 color -> stage5 linear -> stage6 starless -> stage7 stretch -> compatibility checkpoint skipped -> stage8 -> stage9 -> stage10 -> stage11 -> optional isolated artistic derivative -> cleanup`
  同时把阶段 1 记录为 `skipped`，阶段 2 记录为加载既有中间结果
- 当 `SEESTAR_INPUT_MODE=result_linear_resume` 时，会改走显式续跑分支：
  `prepare result_linear.fit -> auto tune -> target preflight -> stage6 starless -> stage7 stretch -> compatibility checkpoint skipped -> stage8 -> stage9 -> stage10 -> stage11 -> optional isolated artistic derivative -> cleanup`
  同时把阶段 2、3、4、5 记录为 `skipped`

### 2.1 实际天文后期逻辑总览

当前代码不是按“先拉伸整图再去星”的旧顺序运行，而是先在较干净的线性图上拆出 starless/starmask，再只拉伸 starless 主体，最后受控回星。

| 流程层 | 代码阶段 / 检查点 | 天文后期目的 | 实际处理逻辑 |
|---|---|---|---|
| 输入统一 | Stage 1 / `working.fit`、`stage1_prepared.fit` | 得到一张可处理的线性 FITS | 优先使用最新叠加 FITS；没有叠加图时隔离 `Light_` 单帧并执行 debayer、register、stack |
| 画面边界 | Stage 2 / `stage2_corrected.fit` | 去掉黑边和窄幅彩色边缘，避免污染背景建模和去星 | 自动扫描四边近黑/暗边界/红蓝叠加伪影并独立裁切，再按 `edge_black_ratio` 迭代复检 |
| 目标策略 | Stage 3/4 preflight / `target_profile.json`、`pipeline_policy.json` | 根据目标类型保护真实星云、暗云气、星系核心或星团星色 | 用像素特征、FITS metadata、目标库和路径上下文生成 profile；Stage 4 platesolve 后再刷新一次 |
| 线性校正 | Stage 3-5 / `stage3_bgremoved.fit`、`stage4_color.fit`、`stage5_linear.fit` | 在线性域完成背景、校色、可选反卷积和轻降噪 | 背景候选统一经过保留门控；Stage 4 只对确认线性的宽带 RGB/OSC 执行单次在线 Gaia PCC，并以不可变 `pre_pcc`、目标感知质量门和恒星软遮罩局部回退保护颜色；Stage 5 先 PSF/RL，再 post-RL denoise |
| 线性去星 | Stage 6 / `stage6_starless.fit`、`starmask_raw.fit`、`starmask_clean.fit` | 在噪声未被非线性放大前拆出主体和星点层 | 固定用最终可用线性检查点做 SyQon；质量重试只调整 SyQon 参数并复用同一线性输入；原始星点层与清理层分开保存，执行失败再回退 SASP |
| 主体拉伸 | Stage 7 / `stage7_stretched.fit` | 把 starless 主体拉到可视亮度，同时避免黑场压死和核心过曝 | 主选固定比较 `stage7_cand_a` 与 `stage7_cand_b`；仅因背景色度门控失败时追加受控救援候选；目标画像/policy 约束候选方法、P50/P99 标尺和核心/弱信号保护，所有候选都要通过核心、弱结构或暗云局部指标门控；`stage7_preview_ref` 不参与最终选择 |
| Starless 增强 | Stage 8 / `stage8_enhanced.fit`、`starless_enhanced.fit` | 分区提升星云主体和外围弱信号，保护背景与亮核 | 优先加载本轮验收通过的 `stage7_stretched`；Stage 7 失败时依次回退 `starless`、`stage6_starless`；默认 soft mask 分区增强，必要时 conservative skip、蓝偏修正或回滚 |
| 回星合成 | Stage 9 / `stage9_remixed.fit` | 把星点按质量诊断受控回混，避免二次星点、halo 或坏蒙版污染 | 从原始含星图建立独立星表并匹配 starmask；星点作为上层、Starless 作为底层执行受支持层约束的 Alpha+Screen，候选不通过 post-remix 门控时降强度重试或回滚无星底图 |
| 交付导出 | Stage 10 / `stage10_final.fit` | 生成可交付 TIFF/PNG/FITS | 按 `stage9_remixed -> starless_enhanced -> stage7_stretched` 选择最终图，做最终饱和度、降噪回退链和格式导出 |
| 可选 AI 副本 | Stage 11 / `result_processed_ai.*` | 在不覆盖主结果的前提下生成保守 AI 后期副本 | 只写 `*_ai`，先本地应用模型建议参数，再质量门控和降强度重试 |

### 2.2 GUI 最新阶段预览协议

阶段预览是只读观察链路，不参与候选选择、质量门控、阶段状态或最终导出：

- `_record_stage()` 先写 `[PIPELINE_STAGE_RESULT]`；summary-only 的 `ok_with_fallback` 对 GUI 映射为 `degraded`，`ok_skipped_optional` 映射为 `skipped`，但不改变流水线内部控制状态。仅当内部状态为 `ok` 或 `degraded` 时，才从该阶段已经验收并保存的最终产物生成预览。
- Stage 1-10 依次使用 `stage1_prepared`、`stage2_corrected`、`stage3_bgremoved`、`stage4_color`、`stage5_linear`、`stage6_starless/stage7_starless`、`stage7_stretched`、`stage8_enhanced`、`stage9_remixed`、`stage10_final`；Stage 11 仅在 AI 产物真实生成后使用工作目录下的 `result_final_ai.fit`。兼容别名只作为同阶段候选回退，不允许使用未验收候选。
- 像素直接映射为有界 16-bit RGB PNG，不执行 autostretch、Asinh、GHS、Gamma 或百分位归一化。Stage 1-6 属于线性数据，GUI 中偏暗是预期表现；Stage 7 以后显示阶段本身已有的非线性结果，但不增加额外显示拉伸。
- 预览先写临时文件，再原子替换 `process/ui_preview/latest.png`。成功时输出 `[PIPELINE_PREVIEW] {"stage":...,"title":...,"status":"ready","payload":".../latest.png"}`；读取或写出失败时输出 `status=unavailable` 和轻量原因。
- `failed/skipped` 阶段不发布新预览。预览生成失败只记录警告，GUI 保留上一张可靠图像，不展开失败流程，也不改变该阶段的 `ok/degraded` 结果。
- Stage 0 不属于正式 pipeline 阶段，由 GUI 异步读取实际输入：完整处理优先选择排序后的首个可读 `Light_` 样本，否则选择最近的可处理叠加 FITS；续跑模式直接使用 `stage2_corrected.fit` 或 `result_linear.fit`。

## 3. 核心对象分工

### 3.1 `PipelineConfig`

`PipelineConfig` 是全部可调参数的集中入口，主要分为几类：

- 重试控制：`max_retries`、`retry_delay`
- 阶段 1 质量门控：`stage1_register_fail_ratio_max`
- 裁切：`crop_margin`
- 背景提取：`bg_samples`、`bg_tolerance`、`bg_smooth`、`bg_quality_gate_enabled`、`bg_std_worsen_ratio_max`、`bg_median_drop_ratio_min`、`bg_object_preserve_ratio_min`、`bg_edge_black_rise_max`、`bg_star_preserve_ratio_min`、`bg_nebula_mean_change_max`，以及 policy 中的 `stage3_background.max_bg_std_growth`
- 线性降噪：`denoise_enabled`、`denoise_mod`、`denoise_safety_max`
- 拉伸：`asinh_stretch`、`asinh_offset`、`ghs_shadowsclip`、`ghs_stretchamount`，以及 `stage7_target_aware_stretch_enabled`、Stage 7 preview 标尺开关、两个 P50 目标比例、Asinh 强度上限和目标局部质量阈值
- 星云增强：`nebula_saturation`、`nebula_bg_factor`
- 星点混合：`star_intensity`、`stage9_fallback_intensity_levels`、兼容上限 `star_fallback_intensity`，以及混合星场单调多锚点曲线的弱/中/亮/极亮目标和预测覆盖上限（非混合场仍保留 Asinh 回退；`remix_safe_blend`、`remix_nebula_weight` 为旧配置兼容字段）
- 最终导出：`final_saturation`、`final_bg_factor`
- AI：`ai_post_enabled`、`ai_endpoint`、`ai_model`、`ai_api_key`、`ai_timeout_sec`、`ai_strength`、`ai_prompt`、`ai_advisor_mode`、`ai_stage6_enabled`、`ai_stage7_enabled`、`ai_stage8_enabled`
- AI 艺术衍生实验：`ai_artistic_derivative_enabled`、`ai_artistic_endpoint`、`ai_artistic_model`、`ai_artistic_api_key`、`ai_artistic_prompt`、`ai_artistic_timeout_sec`
- AI 诊断与 Stage11 门控：`ai_bg_median_delta_max`、`ai_color_ratio_delta_max`、`ai_core_growth_ratio_max`、`ai_star_growth_ratio_max`，以及阶段 6-8 的背景/裁剪/星点/蓝偏/饱和度/微对比诊断阈值
- 运行行为：`checkpoint_mode`、`debug_mode`、`auto_tune_enabled`、`auto_tune_debug`
- 工作流插件控制：`workflow_plugin_probe_enabled`（默认 `False`，仅在显式启用时探测插件命令）、`aberration_api_enabled`（默认 `False`，API 路径在 siril-cli 线程所有权场景下可能失败）、`optional_color_transform_enabled`（默认 `False`，启用时阶段 8/9 尝试可选调色插件）
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
- `pipeline_policy.json`：Stage 3-7 使用的背景、校色、线性处理、拉伸候选和兼容门控策略
- 内置策略来自 `policy_selector.py`，配置文件来自 `pipeline/configs/policies/*.yaml`
- 若 profiler、metadata 或自适应特征读取失败，会回退为 `generic_low_snr_safe`

目标画像识别不是独立 Stage 2.5，而是 Stage 3/4 内部 preflight，不记录到 stage summary。`stage2_5_target_profiler()` 仅为旧调用方保留，实际转发到 `target_profile_preflight()`。Stage 3 背景提取前先生成初始 `target_profile.json` / `pipeline_policy.json`，Stage 4 在 `platesolve -noflip -focal=160 -pixelsize=2.90 -catalog=gaia -order=3` 后会再次基于 `stage4_psolved.fit` metadata 刷新目标画像，若 target type 或 policy 改变，会覆盖上述 JSON 并记录告警。

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
| `GraXpert-AI.py`             | `onnx`, `onnxruntime`, `appdirs`, `cv2`, `PyQt6` |
| `AutoBGE.py`                 | `cv2`, `scipy`, `PyQt6`             |
| `CosmicClarity_Sharpen.py`   | `PyQt6`, `tiffile`, `lz4`, `zstandard`, `exifread`, `cv2` |
| `CosmicClarity_Denoise.py`   | `PyQt6`, `tiffile`, `lz4`, `zstandard`, `exifread`, `cv2` |
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

`TargetType` 仍会识别并记录日志，但在当前版本中仅用于诊断说明，不再直接驱动 preset 套参。目录坐标、视觉特征和自动提示的选择过程写入 `target_profile.json -> diagnostics` 并按 INFO 输出；只有会影响策略可靠性的异常才进入 `warnings`。

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
4. 每个候选都从 `stage3_bg_input` 基线重新加载后执行；若命令失败，直接 fallback 到下一个候选。所有 `pyscript` 候选都会先检查对应 Python 模块，并比较执行前后图像指纹；即使 Siril 外层返回成功，只要图像未变化，也会记录为 `graxpert_runtime_error` 或 `plugin_runtime_error` 并立即尝试后续候选，不再把原图作为背景提取成果进入质量评估。GraXpert 另外识别 `GraXpert-AI.py`、ONNX runtime、`too many indices for array` 等运行时错误
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

当前版本阶段 4 使用四个检查点：`stage3_bgremoved.fit -> stage4_psolved.fit -> stage4_pre_pcc.fit -> stage4_color.fit`。`stage4_pre_pcc.fit` 是 PCC 前不可变的线性回滚源；任何 PCC 超时、失败或质量门拒绝都必须先回到它，再执行本地回退。解析命令固定使用 `platesolve -noflip -focal=160 -pixelsize=2.90 -catalog=gaia -order=3`；`-noflip` 保留用户原始构图方向。

顺序：

1. `load stage3_bgremoved`
2. 基于当前裁切后图像记录 Stage 4 几何：S30 Pro / Sony IMX585 / 160mm / 2.9um，以及 Stage 2 累计裁边
3. 默认使用 Gaia 星表做一次解析：`platesolve -noflip -focal=160 -pixelsize=2.90 -catalog=gaia -order=3`；如需多星表候选，可通过 `SEESTAR_STAGE4_PLATESOLVE_CATALOGS` 显式覆盖
4. 若普通解析候选全部失败且 FITS header 有 `RA/DEC`、`OBJCTRA/OBJCTDEC` 或 `CRVAL1/CRVAL2`，会追加一轮 header-center platesolve 候选：`platesolve <ra>,<dec> -noflip -focal=160 -pixelsize=2.90 -catalog=<catalog> -order=3`；可通过 `SEESTAR_STAGE4_PLATESOLVE_HEADER_RADIUS` 额外传入 `-radius=...`
5. `save stage4_psolved`
6. Stage4 preflight 刷新 `target_profile.json` 和 `pipeline_policy.json`
7. 从 FITS `OBJECT`、坐标和目标目录自动刷新目标画像；例如 `OBJECT=M 42` 会识别为 M42，并选用亮发射/反射星云质量门
8. 判断线性状态与通道语义：Mono 正常跳过；非线性保色跳过；语义未知保色并标记 `review_required`；高置信度 HOO/SHO/Ha+OIII 双窄带跳过 PCC
9. 保存不可变 `stage4_pre_pcc.fit`
10. 仅当输入为确认线性的宽带 RGB/OSC、platesolve 成功且允许联网时，通过独立 `siril-cli` 执行一次 `pcc -catalog=gaia`。超时固定 30 秒，不重试、不切换 catalog
11. PCC 候选经过目标感知质量门：检查有限值、通道相对增益、背景综合色偏、高光裁剪增长和动态范围漂移。M42/发射星云允许真实红色主体占优；星系背景使用更严格的中性阈值
12. PCC 失败或质量门拒绝时先重新载入 `stage4_pre_pcc.fit`，再尝试恒星软遮罩局部色彩恢复。增益只作用于恒星及星翼，禁止全图白平衡和全图背景中性化
13. 本地恒星样本不足时不修改像素，保留输入颜色并标记 `review_required`
14. 高置信度窄带仅允许同一恒星软遮罩局部恢复；样本不足时直接保色，均按 `skipped_by_policy` 正常完成
15. `save stage4_color`，并保留兼容别名 `stage4_colorbalanced`

PCC 运行时保护：

- `SEESTAR_STAGE4_PLATESOLVE_ENABLE=0` 可关闭 platesolve；默认开启
- `SEESTAR_STAGE4_PLATESOLVE_CATALOGS` 调整解析星表候选顺序；默认 `gaia`
- `SEESTAR_STAGE4_FILTER_HINT` 仅在 FITS `FILTER` 缺失时补充 `broadband` 或 `dualband Ha OIII` 等通道语义，不直接套用固定调色矩阵
- `SEESTAR_STAGE4_PCC_TIMEOUT_SEC` 默认 30，运行时安全限幅 5–120 秒；产品 GUI 固定 30 秒
- PCC 使用独立 Siril 进程作为可终止边界；GUI 注入 `SEESTAR_SIRIL_CLI` 和本轮临时配置路径。找不到独立 CLI 时按单次失败处理，不在主连接中执行无界 PCC
- 通用 `cmd_with_check` 也把 `pcc` 列为非幂等命令，防止未来调用路径意外触发重试

回退逻辑：

- `platesolve_ok=False` 时不调用 PCC，直接从 `stage4_pre_pcc` 做局部恒星回退
- PCC 成功但质量门拒绝与 PCC 命令失败使用同一回滚路径
- 宽带本地回退成功或保色都写入 `requires_review=true`，阶段状态为 `degraded`
- Mono、非线性和高置信度窄带属于策略跳过，不把“未运行 PCC”误报为失败
- `color_calibration_report.json` 记录 `channel_policy`、单次 `pcc.attempts`、目标感知 `quality_gate`、`rollback`、局部软遮罩覆盖率和是否要求人工复核

阶段检查点：`stage4_psolved.fit`、`stage4_pre_pcc.fit`、`stage4_color.fit`（兼容别名：`stage4_colorbalanced.fit`）

### 5.5 阶段 5：线性反卷积 / 轻降噪

职责：在线性图像上优先执行本地 GraXpert Object Deconvolution；模型或脚本不可用、执行失败/空跑时用未降噪星点建立 PSF 并执行受控 Siril RL 回退，再做轻量线性降噪，为后续线性去星提供保留微反差且背景不过度放大的输入。

逻辑要点：

- 输入固定优先 `stage4_color.fit`，阶段开始保存 `stage5_input_linear.fit`
- 首选 GraXpert 链路：GUI 依次检测 Seestar 随包模型、本机 GraXpert 应用已安装模型，再读取 `SEESTAR_GRAXPERT_OBJECT_MODEL_PATH` 提供的外部 `model.onnx`、语义版本目录或模型家族目录；有效模型会只读链接到隔离 HOME 的 `deconvolution-object-ai-models/<version>/model.onnx`。版本目录必须使用官方语义格式（例如 `1.0.1`）。随后调用 `GraXpert-AI.py -deconv_obj -strength 0.30 -psfsize 5.0 -model <version> -nogpu`，成功后保存 `stage5_graxpert_deconv.fit`
- GraXpert 不可用或失败时进入 Siril 内核回退：`findstar -maxstars=200` -> `makepsf stars -sym -ks=33 -savepsf=stage5_psf.fit` -> `rl -loadpsf=stage5_psf.fit -iters=8 -alpha=3000 -tv -gdstep=0.0005 -stop=0.001` -> `save stage5_deconv` -> `denoise -mod=0.50 -indep` -> `save stage5_linear`
- `result_linear.fit` 在 `stage5_linear` 保存后导出，供 GUI 显式续跑模式使用
- Stage5 不再默认执行全局锐化、unsharp 或星点矫正，避免在线性暗背景中提前放大彩噪、星环和 halo
- CosmicClarity Denoise 只作为 Siril 内置降噪失败后的回退，强度映射限制在 `0.20~0.30`；`chroma_first` 使用 `0.30`，`luma_chroma_balanced` 使用 `0.25`
- GraXpert 反卷积只映射到 Stage5，强度限制为 `0.20~0.40`；模型不随此链路在线下载，缺失时稳定降级到 Siril RL
- 若 GraXpert 或 RL 后背景指标明显变差，背景保护会丢弃反卷积结果，重新回到 `stage5_input_linear` 后再执行轻降噪；因此 `stage5_linear.fit` 始终是 Stage 5 的最终线性输出

当前 Stage 5 不执行可选转色。`optional_color_transform_enabled=True` 的可选颜色插件只在 Stage 8/9 使用，避免在线性阶段提前改变窄带/双窄带颜色语义。

阶段检查点：`stage5_linear.fit` 是最终线性输出；可选 `stage5_graxpert_deconv.fit` 或 `stage5_deconv.fit` 是反卷积后、降噪前检查点；兼容别名仍保存 `stage5_denoised.fit`

阶段额外产物：`result_linear.fit`（位于工作目录根目录，默认不会被下一轮输入扫描命中；可供 GUI 显式续跑模式继续执行阶段 6-11）

诊断输出：`stage5_linear_report.json` 会记录处理顺序、输入/输出、Siril 降噪参数、CosmicClarity 回退强度、GraXpert 模型/失败原因、最终反卷积方法与 `background_guard` 的 before/after 自适应背景指标。

### 5.6 阶段 6：去星与星点层准备

职责：调用去星工具生成去星图，并尽量构造一个可控的星点层。当前默认在阶段 7 拉伸前执行，优先使用 `stage5_linear.fit`；GraXpert/RL 反卷积检查点仅在最终线性输出缺失时作为回退；`result_linear.fit` 续跑时会使用续跑加载得到的线性检查点。收益是避免非线性拉伸把低频脏背景、色斑和噪声放大后再交给去星模型；副作用是星点层也来自线性图，后续阶段 9 需要通过星点拉伸/回混强度控制恢复星点亮度。

星点分离输入模式：

- Stage 6 直接使用 Stage 5 的最终线性结果；输入域在质量报告中明确记录为 `input_domain=linear`，不再提供外部预拉伸模式。
- 当前内置 SyQon 会在脚本内部对线性图执行临时、可逆的 IHS 预拉伸，并在输出前逆变换回线性域。因此主流程不再额外执行 Asinh 预拉伸，避免 SyQon 内部变换与 Stage 7 主体拉伸叠加。
- `SEESTAR_STAR_SEPARATION_MODE`、`SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH` 与 `SEESTAR_MILD_PRESTRETCH_STRENGTH` 已退役且会被忽略，不会生成或选择任何外部预拉伸输入。
- 动态范围塌缩同时参考绝对峰值和峰值/背景比：极低背景下仍保留足够相对信号时不误报。若确认塌缩，只尝试切换 Zenith/Axiom 模型；不再仅修改 `tile_size`/`overlap` 重复运行同一模型。其他残星、halo 或黑洞问题仍可在同一线性 source 上调整参数补救。

目标感知旁路：当 target profile 为 `globular_cluster`、`open_cluster` 或 `reflection_nebula_cluster`（包括 M45 类星点主体）时，默认不调用 SyQon/SASP。Stage 6 会把线性输入作为保星检查点继续交给 Stage 7 拉伸，不生成 `starmask`；Stage 8 随后跳过 Starless 专用增强，Stage 9 因无星点层不再回混。可通过 `PipelineConfig.stage6_star_preserve_target_bypass_enabled=False` 显式关闭此保守旁路。

说明：阶段 6 会导出新命名产物 `stage6_starless.fit`、`stage6_starless_quality.json`；同时保留旧别名 `stage7_starless.fit`、`stage7_quality.json`，用于兼容外部工具、续跑和旧报告。

去星优先链：

1. `SyQon-Starless.py` CLI 子进程（默认 Zenith v1 模型，tile-size=512，overlap=64；`SEESTAR_SYQON_GPU=0` 时追加 `--no_gpu`；带超时与日志转发）
2. `SASP Dark Star`（命令探测）

SyQon 路径补充逻辑：

- 脚本执行成功后，会在 `process/` 中收集产物：先按 `starless_{stretched_name}` / `starless` 查找，再按 glob 兜底
- 若 SyQon 脚本未生成 starless 产物，视为失败并回退到 SASP Dark Star

正常路径（去星成功后）：

1. 保存 `starless.fit`
2. 优先手动构建线性星点层：`stage6_input - starless`，并立即保存不可变的 `starmask_raw.fit`
3. 手动失败时，从可能的 `starmask` / `*_stars` 文件中兜底查找
4. 再不行则扫描 `process/` 中最新的星点蒙版
5. 对 raw 星点层自身做多尺度清理：估计背景/MAD 噪声，分离紧致星点结构与低频弥散残差，只用统一标量权重修改 RGB，避免改变星色
6. 清理后验收紧致星核/星翼保留率和弥散残留；通过时另存 `starmask_clean.fit` 并令兼容别名 `starmask.fit` 指向 clean。紧致结构保留不足时回滚 raw；`diffuse_residual_ratio` 超过硬上限 `0.08` 时则禁用当前候选的 starmask，不允许用仍带弥散污染的 raw 继续回星。任何清理都不得覆盖 `starmask_raw.fit`
7. 导出 SASP 交换文件（`sasp_starless_input.fit`、`sasp_starmask_input.fit`）

星点层清理不再根据主体图的亮区直接判定“星云污染”，因此不会因为星点位于 M42 核心、发射星云丝状结构或星系盘面上就统一降权；也不再对全部小星额外降强或在 halo 区直接做模糊。现有兼容字段的新语义是：`stage7_starmask_small_star_scale` 为紧致弱星最低保留比例，`stage7_starmask_halo_blur_strength` 为非紧致宽 halo/低频残差衰减强度，`stage7_starmask_nebula_suppression` 为星点层自身的弥散污染扣除强度。清理指标及 `diffuse_hard_gate_failed` 写入 `stage6_starless_quality.json.starmask_cleanup`；硬门失败同时把所选 Stage 6 质量标为 `poor`，Stage 9 进入无 starmask 的安全降级路径。

AI 参数优化（`SEESTAR_AI_ENABLED=1` 且 endpoint/model/key 齐全时）：

- 去星前会先请求 `stage7_starless_plan`，将 AI 建议的 `tile_size`、`overlap`、`use_axiom` 实际传给 SyQon CLI；`use_axiom` 映射为 SyQon Axiom 2.1 的 `--axiom21`，仅在缓存中存在 `vendor/siril-scripts/Axiom2_1.pt` 或用户数据目录 `syqon_starless/axiom21.pt` 时启用；`tile_size` 下限为 512、`overlap` 下限为 64
- 每次得到 `starless.fit` / raw/clean starmask 后生成一次质量记录，最终写入 `process/stage6_starless_quality.json`，并保留兼容别名 `process/stage7_quality.json`
- 本地指标先判断 `starless` 残星、动态范围塌缩、低峰值信号、峰值/背景比、`starmask` 缺星和 `starmask` 过宽，再把指标发给 AI 做保守裁判。峰值/背景比默认下限为 `4.0`，可用 `SEESTAR_STAGE7_STARLESS_PEAK_BACKGROUND_RATIO_MIN` 调整
- 若残星、缺星或蒙版过宽指标偏差，会在 `SEESTAR_STAGE7_QUALITY_RETRY_MAX` 预算内用其他 SyQon 参数组合重跑；仅动态范围塌缩时最多切换一次可用模型，无备选模型则直接保留初次产物和降级诊断
- 若最终 `residual_star_score` 仍高于阈值，AI 可返回 `residual_suppression_strength` 和 `stage9_star_intensity_scale`；脚本会做安全限幅后结合 `starmask` 与拉伸层 residual map 实际修改 `starless.fit` 抑制残星，并把最终阶段 9 回混缩放写入 `stage6_starless_quality.json.stage9_star_remix`
- 最终 starless 质量仍差时允许执行背景综合色噪像素修复；除原有“残星或 halo 明显改善”路径外，当综合色噪下降比例和绝对下降量同时达标，且残星与 halo 均未恶化时也接受修复，诊断写入 `starless_pixel_repairs[].chroma_acceptance` 与 `acceptance_path`
- 质量诊断不再把成功的 SyQon/SASP 结果直接降级为拉伸检查点；只有所有去星工具失败才走退化路径

回退逻辑：

- 如果所有去星命令都失败，阶段记为 `degraded`
- 同时把当前线性 Stage 6 输入检查点（通常是 `stage5_linear`，缺失时使用更早的线性检查点）重新保存成 `starless.fit`，让阶段 7-10 仍然可以继续执行
- 因此去星工具不可用时，脚本退化为"不做真正去星"的流程

阶段检查点：`stage6_starless.fit`（兼容别名：`stage7_starless.fit`）

阶段输出：

- `starless.fit`
- 可能存在的 `starmask_raw.fit`、`starmask_clean.fit`；`starmask.fit` 是兼容别名
- `sasp_starless_input.fit`（工作目录，供外部工具使用）
- `sasp_starmask_input.fit`（工作目录，供外部工具使用）

### 5.7 阶段 7：主体拉伸

职责：把阶段 6 生成的 starless 图像变成可视化的非线性图像。当前固定 I/O 为 `stage6_starless.fit -> stage7_stretched.fit`。

说明：阶段 7 只使用 Stage7 命名，不再把主体拉伸候选同时写成 `stage6_candidate_*` 与 `stage7_candidate_*`。

候选收缩为固定两条，另保留一个 preview 参考；候选名称固定，但方法和参数会同时读取 Stage 3/4 的 `target_profile` / `pipeline_policy.stage6_stretch`，再根据 `stage6_starless` 的 baseline 背景自适应：

1. `stage7_cand_a.fit`: 通用目标为 `asinh <adaptive_stretch> <adaptive_offset>`；亮核心星云/暗云目标改用同一 Asinh 后的局部弱信号抬升、背景彩噪抑制和核心压缩；普通背景默认 `2.2 / 0.002`
2. `stage7_cand_b.fit`: 通用目标为 `asinh <adaptive_stretch> <adaptive_offset>` -> `autoghs -linked -2.1 <adaptive_amount>`；星团与反射星云星团改为第二条纯 Asinh 候选，避免 GHS 放大星点；普通背景默认 `2.1 / 0.002 / 1.05`
3. `stage7_preview_ref.fit`: `load stage6_starless` -> `autostretch -linked`，读取其 P50/P99 作为候选参数标尺，但不允许成为最终图

关键行为：

- 当 baseline `bg_median <= 0.005` 时启用极低背景保护：保持足够的 Asinh 强度，并用 `p01`、背景中值和背景噪声共同限制 offset，使其低于背景底部，避免弱信号被裁成黑场
- 当 baseline `0.005 < bg_median < 0.010` 时使用中等低背景保护，参数在默认值与极低背景值之间平滑过渡
- 通用目标以 preview P50 的约 35% / 25% 标定 A/B；亮核心星云和星团下调 P50 目标并把 preview P99 高光标尺从 90% 收紧到约 82%，星系收紧到约 85%，防止亮核/星核过曝
- 暗云提高 P50 目标但限制 GHS，用可见背景承载尘埃剪影；大视场发射星云适度提高 P50 目标以保留弥散弱信号；目标画像不可用时保持原通用参数
- `stage7_target_aware_stretch_enabled=False` 可关闭上述目标映射，只保留背景和 preview 自适应
- preview 生成后按 Siril Asinh 公式在 `1-1000` 范围内反解候选强度，并由各目标 profile 的 P99 标尺限制高光
- preview 标定有效时，正式候选实际 P50 默认必须处于标定目标的 55%–150%；低于下限或高于上限的候选会被拒绝，也不能作为“仅色度超限”的救援源；救援候选也会重新检查该上限
- preview 标尺无效或读取失败时保留原有背景自适应参数，不影响离线主流程继续运行
- 候选像素分布若命中 `is_nearly_black`、`is_visibility_too_low`、`is_nearly_white` 或 `invalid_dynamic_range`，会被标记为非正常最终候选
- 每个正式候选使用最终交付同口径的暗背景采样检查绝对综合色噪、低频背景斑驳，以及综合色偏差相对背景亮度的放大倍数；任一超限即 `quality_ok=false`，明暗结构、裁切与星点指标即使正常也不能进入正式 `stage7_stretched`
- 默认从线性 baseline 构建背景、暗结构、弱结构和亮核局部区域；亮核目标检查核心 P99/裁剪，星系、发射星云和暗云检查弱结构相对背景 SNR，暗云额外检查明暗分离。局部风险会并入候选总风险，超阈值时直接拒绝候选；指标不可用时只记录 unavailable，不阻断离线回退
- 若两个正式候选仅因背景综合色噪/色度负载增长超限而被拒绝，先以风险最低候选生成 `0.35 / 0.55 / 0.65` 三档背景限定、保亮度的色度抑制救援候选；星云、弱信号和核心 mask 受到保护，每档仍重新执行可见性、背景、核心和弱结构全部门控
- 主候选与本轮实际生成的全部救援候选统一执行“硬门优先、质量排序”择优：只有 `allowed_as_final=true` 的候选可进入正式选择，再按硬问题数、归一化超限量、风险分和背景质量负载选取最优项；通过门控的最优救援候选可保存为正式 `stage7_stretched` 并继续 Stage 8/9，Stage 7 状态保留为 `degraded`，但 `_stage7_stretch_accepted=true`
- 若救援仍未通过，不再按固定 A/B 文件顺序回退：先排除非有限值、近黑/近白、动态范围异常、核心裁剪或弱结构失败的候选，再从仅亮度上限/可复核色度问题的安全候选中选择 `actual_p50 / target_p50` 最低者作为 `_stage7_review_source`；该路径 Stage 7 记为 `degraded`、`_stage7_stretch_accepted=false`，后续仅允许导出 `result_review*`，不生成正常 `result_processed/result_final`
- 只有通过质量门控的候选才保存为 `stage7_stretched.fit`
- 如果所有拉伸方法都失败，阶段状态记为 `failed`，并回载 `stage6_starless`
- `run()` 不会因此立即退出，后续阶段继续使用各自已有的保守回退源
- 诊断输出 `stage7_stretch_quality.json` 和 `stretch_candidates_report.json`，记录两个正式候选、实际生成的救援候选、唯一 preview、`baseline_pixel_stats`、`stretch_adaptation.target_aware`、每个候选的 `preview_target_attainment`、`target_local_quality`、像素分布、`selection_rank`、`selection_role` 和最终/复核选择
- 候选背景硬门控的指标、阈值和拒绝原因记录在每个候选的 `background_quality_gate` 字段

这意味着阶段 7 的 `failed` 更像"拉伸目标未达成"，而不是"整条链路终止"。

### 5.7.1 Stage 7 兼容检查点：去星前质量门控

该检查点不是正式 Stage 6.5 或 Stage 7.5，不改变 Stage 1-11 的编号。职责是保留旧流程的去星前质量门控兼容点。当前默认 starless-first 顺序已经在 Stage 7 拉伸前完成 Stage 6 去星，因此主流程会把 “Stage 7 兼容检查点” 记录为 `skipped`。

旧的非默认/调试路径仍可计算以下诊断，但不再生成历史 Asinh 去星输入；任何非线性推荐都只记录在兼容报告中，实际输入始终保持已选定的线性检查点。

处理逻辑：

1. 以 `stage7_stretched` 或当前 `stretched_name` 为源读取自适应特征
2. 合并当前质量指标：黑场比例、高光裁剪、星点尺寸、边缘黑边等
3. 使用 `target_profile` 和历史兼容 policy key `pipeline_policy.stage6_5_pre_starless_gate` 评估 `ready_for_starless`
4. 当旧 policy 推荐 `stage7_conservative_asinh` 或 `stage7_ultra_conservative_asinh` 时，将推荐写入 `ignored_recommended_starless_input`
5. 报告的有效推荐重置为当前线性源；不执行 Asinh，也不会将 `stretched_name` 切换到非线性候选

阶段输出：

- `stage7_5_pre_starless_gate_report.json`（历史文件名，兼容别名：`pre_starless_gate_report.json`）

回退逻辑：

- 评估模块不可用或指标读取失败时，阶段记为 `degraded`
- 回退 report 会设置 `ready_for_starless=true`，推荐输入为阶段 7 拉伸检查点，避免门控故障阻断后续阶段

### 5.8 阶段 8：Starless 深加工

职责：对去星结果做增强处理。阶段 8 默认生成 Starless soft masks 后分区增强，保护亮核心和背景区域；SASP WaveScale Dark Enhancer API 仍可作为增强源，但输出会通过 mask 回混，避免整图锐化/提亮。打包环境不再探测实验性 `sasp_*` Siril 命令，因为这些命令未由当前 SASP wheel 注册到 Siril CLI；API 不可用时直接使用确定性的内置分区增强链。

输入选择固定为：本轮 `_stage7_stretch_accepted=true` 且文件存在时使用 `stage7_stretched.fit`；否则依次尝试 `starless.fit`、`stage6_starless.fit`。仅存在旧 `stage7_stretched.fit` 不足以被选中，避免失败任务误吃历史残留产物；实际选择写入日志、`stage8_input_selection.json` 和 Stage 8 报告的 `stage8_input_source/input_source`。

当 Stage 7 已请求 `_stage8_conservative_mode=true` 且输入门控决定跳过增强时，内部 `_stage8_final_quality`、`stage8_quality.json` 和 `stage8_enhancement_report.json` 统一记录 `conservative_skipped`；即使同时存在 `stage7_quality_status=poor` 等附加诊断原因也不改回普通 `skipped`。非保守模式触发的输入风险跳过仍记录 `skipped`。Stage 10 据此把保守跳过识别为已知安全旁路，不误启用 Stage 8 fallback 严格门限。

若 Stage 6 已命中星团/M45 保星旁路，本阶段只保存 `stage8_enhanced.fit` / `starless_enhanced.fit` 兼容检查点，不执行 Starless mask、锐化、局部对比或饱和度增强。

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
   - `saturation` 在执行前还会受到 Stage 4 `max_allowed_saturation_boost` 总预算限制；校色报告要求降饱和时预算减半。该预算由 Stage 8、10、11 共享，模型建议不能绕过。
3. 插件链可用时，默认在插件处理后追加分区像素增强；AI 明确返回 `apply_after_plugins=false` 时才保留插件 mask 回混结果不追加
4. 插件链或内置增强前会先按信号区 R/G、B/G 做蓝偏预抑制并同步降低饱和度，后置 `ccm` 只处理残余明显蓝偏
5. 再判断蓝偏、饱和度增长、微对比增长、高亮裁剪、背景噪声增长、背景提亮、核心裁剪增长和纹理伪影增长，并把指标写入 `stage8_quality.json`；AI 开启时同时发给 AI 生成诊断
6. 若蓝偏超限，会按 AI 返回的 `target_blue_excess`（缺失时用本地阈值）对当前 `stage8_enhanced` 执行受限 `ccm` 蓝通道修正并重新保存 `starless_enhanced.fit`，随后重新计算 `stage8_quality` 并更新 `final`
7. 若饱和度、微对比、高光、背景噪声或纹理伪影风险仍偏高，会从 `stage8_input_starless.fit` 用保守分区增强重跑并覆盖 `starless_enhanced.fit`，避免过处理结果传入阶段 9

### 5.9 阶段 9：星点处理与合成

职责：把阶段 8 的 Starless 结果与星点信息重新组合。

外部回写导入：

- 阶段开始时会查找 `sasp_starmask.fit` / `starmask_sasp.fit` / `starmask_from_sasp.fit`
- 若存在外部回写文件，导入为独立的 `starmask_external_raw.fit`，并更新当前兼容别名 `starmask.fit`；Stage 6 的 `starmask_raw.fit` / `starmask_clean.fit` 不被覆盖

星点预处理链（当 `workflow_plugin_probe_enabled=True` 且有 starmask 时）：

1. 星点拉伸：`SASP Star Stretch` / `NB to RGB Stars`
2. 星点去紫：`SASP Invert/SCNR` / `SCNR`
3. 星点微调：`SASP Curves Editor` / `Curves`
4. 处理后导出更新的 SASP 交换文件

Starless 二次细化（在合成前）：

1. `VeraLux Revela` / `Revela`
2. 可选调色：`VeraLux Vectra` / `Vectra`（当 `optional_color_transform_enabled=True`）
3. `VeraLux Curves` / `Curves`

星点合成路径：

1. 固定以上一阶段 Starless 为底层、`starmask_stretched` 为上层执行像素级 Alpha+Screen；正式路径不再调用顺序不可验证的 StarComposer
2. 主强度未通过门控时，回滚 Stage 8 source 并按 `0.75 / 0.55 / 0.40` 逐档降低亮星强度；弱星强度默认保持不低于 `1.05`
3. 所有候选均未通过时，放弃回星并把 Stage 8 无星结果保存为降级产物

默认回混公式：

```text
source_reference = compact_local_detail_components(stage5_linear, percentile=98.0, area >= 1)
if source_reference.single_pixel_ratio > 0.20 or (
    source_reference.density_per_megapixel > 2500
    and source_reference.single_pixel_ratio > 0.10
):
    reject_catalog_and_keep_stage8_source()
reference = match_5x5(source_reference.positions, original_starmask > background + 5σ)
weak_stars = lower_80_percent(reference.matched_starmask_peaks)
bright_stars = upper_20_percent(reference.matched_starmask_peaks)
compact_support = weak_cores(+1px) + bright_cores(+3px)  # strict: +1px / +2px
compact_starmask = where(compact_support, cleaned_starmask, 0)
if bright_peak_median / weak_peak_median >= 4 and weak_count >= 20 and bright_count >= 3:
    input_anchors = percentiles(reference.matched_starmask_peaks, [40, 80, 90, 99.7])
    output_anchors = [0.26, 0.50, 0.75, 0.90]
    starmask_stretched = color_preserving_monotonic_log_curve(
        compact_starmask, input_anchors -> output_anchors
    )
else:
    starmask_stretched = asinh(compact_starmask, bounded_single_layer_stretch)
intensity_map = weak_support * max(intensity, 1.05) + bright_support * intensity
star_term = clamp(starmask_stretched * intensity_map, 0, 1)
screened = 1 - (1 - starless_enhanced) * (1 - star_term)
final = alpha_over(starless_enhanced, screened, alpha=compact_support)
```

旧的直接相加会在亮星核心和亮星云重叠区域快速耗尽高光余量；Screen 公式随底图亮度自动减小新增量，仍保留星色和星点可见度，但更不容易把核心推到硬裁剪。

若 `SASP Star Stretch` / `NB to RGB Stars` 已经成功处理星点层，阶段 9 会把插件处理后的星点层另存为 `starmask_stretched.fit`，避免二次拉伸且不覆盖 raw/clean。目录不再由待验收的 starmask 自我定义：默认从 `stage5_linear.fit` 的局部高频细节建立独立紧致星表，再在原始 starmask 的 `5×5` 邻域内匹配有效星点。单像素弱星仍允许进入目录，但单像素组件比例超过 `0.20`，或组件密度超过 `2500/百万像素` 且伴随单像素噪点证据时，先把局部细节检测百分位从配置值按 `0.5` 逐档收紧到 `99.5`；只有全部档位仍疑似噪声/星云纹理污染才 fail-closed，不能通过放宽污染门限恢复回星。正常支持层扩张弱星 1 像素、亮星 3 像素，严格层为弱星 1 像素、亮星 2 像素。支持区外的广域低幅残差清零后保存 `starmask_compact.fit`，像素写回始终在 Siril `image_lock` 内执行。compact 支持层默认至少保留 `80%` 的弱星组件；旧支持层不足时按同一目录重建，禁止通过提高检测阈值丢弃已确认星表中的弱星核心。混合星场不再分别拉伸弱/亮层：统一以原始星峰的 `40 / 80 / 90 / 99.7` 分位建立对数输入轴上的单调多锚点曲线，映射到 `0.26 / 0.50 / 0.75 / 0.90`，并对 RGB 使用同一标量增益，因此亮度排序和星色同时保留。独立源图不可用时保留旧 starmask 目录兼容路径；目录、保留率或像素写回不可用时不会把未经验证的 raw starmask 作为正式交付候选，而是保留 Stage 8 source 并进入 review-only 降级路径。

Stage7 联动：

- 阶段 9 正式路径固定跳过不可控的 StarComposer，避免插件内部图层顺序与合成模式不可验证
- 实际主强度和各档备用强度会乘以 Stage7 计算出的 `intensity_scale`，降低残星与回混星点叠加造成的二次星点风险
- 线性 Starless 仅因相对动态范围/峰值偏低而被标记 `poor`，但 `stage7_stretched.fit` 已通过可见性、背景、核心和弱结构门控时，该早期诊断降为 advisory；Stage 9 仍使用现有 starmask 执行受控 Screen 回星。残星、halo、噪声或拉伸未验收等硬风险仍可触发安全旁路

Stage 9 保存前门控：

- 对主强度和各档备用强度 Alpha+Screen 候选比较 Stage 8 base，检查非有限像素、高光裁剪总量/增长、亮像素覆盖增长、暗背景抬升、低频背景斑驳增长、显著变化覆盖和异常变暗。斑驳评分只排除本次实际 Alpha 支持区；支持覆盖默认不得超过 `0.12`，支持区外显著变化不得超过 `0.01`，防止污染 starmask 反过来豁免自身斑驳
- 同一门控在 `candidate - Stage8` 的正增量中检查每颗独立参考星的 `3×3` 峰值、`7×7` 孔径通量和星翼通量，并检查回星后参考邻域的残余暗坑；默认要求弱星恢复率至少 `0.70`、全部星点与孔径通量恢复率至少 `0.75`、星翼恢复率至少 `0.65`，残余暗坑比例不超过 `0.15`。质量门开启且目录/恢复指标不可用时 fail-closed；显式关闭质量门时仍记录指标与 `gate_enabled=false`
- 候选还会对 `candidate - Stage8` 的显著正增量做独立连通形态检查：默认亮度增量超过 `0.05` 的闭合环内部空心面积不得超过 `64` 像素。该指标不依赖星表恢复率，用于拦截新生成的星环、空心星核和块状闭合结构
- 局部质量门不再依赖全图占比稀释：对亮度增量超过 `0.01` 的连通块检查最大面积 `256`、最长/最短边比例 `3.0`、包围盒填充率 `0.15` 和单像素组件比例 `0.20`；对局部青蓝增量团块限制单块面积 `64`；以 Stage 8 信号区第 `90` 百分位定义星云核心，归一化 RGB 跳变超过 `0.10` 的单块面积不得超过 `64`。任一局部指标超限即拒绝候选；局部连通分析不可用时 fail-closed
- `local_color_risk_score` 取青蓝团块和核心颜色突变相对各自面积上限的较大值并限幅到 `0–1`，既用于 Stage 9 报告，也作为 Stage 10 正向饱和度的风险输入
- 主候选不通过时立即回载 Stage 8 source。若是背景、高光或覆盖上限失败，可先换严格 compact 支持并按 `0.75 -> 0.55 -> 0.40` 逐级降低强度；若是星点恢复量不足则停止降 Screen 强度，因为继续降强度只会让恢复量更低。每个有效候选都从同一 Stage 8 source 重新回混，不得在已拒绝候选上叠加星点
- 所有候选和阈值写入 `stage9_remix_quality.json`；全部拒绝时阶段状态为 `degraded`，`stage9_remixed.fit` 保存回滚后的 Stage 8 source
- `stage9_remixed.fit` 保存成功后生成 `review_bundles/stage9_star_remixing/`，包含 Stage 8 source 与最终接受/回滚图像的 before、after、绝对/有符号 difference 和 `review.json`；JSON 保留全部候选状态、星表污染指标、starmask 标定与拒绝原因。复核包不参与确定性门控，生成失败只记录警告
- `stage9_remix_quality.json` 同时记录 `stars_required / stars_applied / stars_application_mode / starmask_stretch_failed`；只有 Alpha+Screen 候选通过且 `stage9_remixed.fit` 保存成功时，`stars_applied=true`
- `unsafe_starless_bypass`、`no_starmask`、`rejected_keep_starless` 及候选保存失败都必须保持 `stars_applied=false`；保星目标旁路另记 `stars_required=false`，不会被误判为缺星
- Stage 7 拉伸未验收时，Stage 9 不再向线性 Starless 继续回星；优先加载 Stage 7 对全部候选质量排序得到的 `_stage7_review_source`，只保存复核候选并记录 `degraded`，同时让最终质量报告强制进入 `needs_conservative_rerun`

设计意图：

- 阶段 9 的主处理文件必须来自阶段 8 的 `starless_enhanced`
- 不再用旧拉伸检查点与 `starless_enhanced` 做安全混合，避免 `pm` 多文件表达式成功返回但实际保存回拉伸阶段图像
- 合成后同时记录 `stage9_remixed` 对 `stage8_enhanced` 与阶段 7 拉伸检查点的差异，便于排查阶段 9 是否真正生效

回混强度结构：

1. `star_intensity`：自动调参和最终有效强度上限为 `1.05`
2. `stage9_fallback_intensity_levels`：默认 `0.75 / 0.55 / 0.40`；`star_fallback_intensity` 仅作为兼容上限

阶段状态判断：

- StarComposer 或 Screen 像素回混成功且通过门控时为 `ok`
- 没有 `starmask` 时为 `skipped`
- 所有像素级回混档位执行失败或均被门控拒绝时为 `degraded`

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

`final_saturation` 不是独立叠加额度：脚本会扣除 Stage 8 已使用的正向饱和度，只执行 Stage 4 色彩策略剩余的预算；预算耗尽时直接跳过本次 `satu`。正向剩余额度还会乘以 `1 - local_color_risk_score × stage10_stage9_local_color_risk_strength`（默认强度 `1.0`）；Stage 9 必须回星但未应用，或已接受候选缺少局部色彩指标时，风险按 `1.0` 处理并跳过最终增饱和。负向去饱和不受该门限制。诊断写入 `stage10_saturation_guard.json`。

最终降噪优先链：

0. 若进入 `review-only`（Stage 9 不安全旁路、starmask 拉伸失败或必须回星但未应用），直接跳过 CosmicClarity/SCUNet/Aberration 最终降噪链并快速导出复核源，避免为不可正式交付的结果继续等待耗时模型；否则，Stage 5 已成功降噪、Stage 8 最终质量为 `ok`/保星安全旁路且没有使用不安全 fallback 时，重复降噪保护会跳过 Stage 10 降噪链。
1. 读取最终降噪输入的 `chroma_noise_score`、`bg_std` 与 `background_mottling_score`，动态选择 `full/chroma/separate/skip`。`chroma` 路径用 CosmicClarity `full` 模型获得降噪色度后恢复输入亮度，因为打包版本原生只接受 `full/luminance/separate`，不能传入不存在的 `chroma` 参数
2. `CosmicClarity_Denoise.py` in-process script（复用已验证更稳定的 Siril 内执行路径，使用上述动态模式和 classic 参数）
3. 若 in-process 失败，再尝试 `CosmicClarity_Denoise.py` CLI 子进程模式
4. 若 classic Denoise 不可用或失败，尝试 CosmicClarity Native Denoise，再尝试 `Siril-SCUNet Denoise`；若 Native 已超时，为避免再次等待 300 秒，不再级联启动 SCUNet
5. 若 `Siril-SCUNet Denoise` 不可用且 `aberration_api_enabled=True`，再回退 `SASP Aberration API`
6. 若 `aberration_api_enabled=False`（默认），跳过 Aberration API 回退

模式规则与诊断：综合色噪达到 `0.70` 时使用 `separate`；达到 `0.34` 且亮度背景仍稳定时使用 `chroma`；综合色噪同时伴随 `bg_std>=0.018` 或斑驳分数 `>=0.45` 时使用 `full`。仅当有效彩色输入的综合色噪低于 `0.34`、`bg_std` 低于 `0.018` 且斑驳分数低于 `0.45` 时，Stage 10 以 `skip` 正常跳过昂贵的最终降噪；灰度输入或指标不可用仍保守使用 `full`。输入指标、原因、请求模式、底层 CosmicClarity 模式、最终有效模式及低噪声跳过标记写入 `stage10_denoise_plan.json`。

状态规则：

- 三项有效噪声指标均低于门限时以 `skipped_safe` 保持阶段 `ok`，不启动任何最终降噪模型
- `CosmicClarity_Denoise.py`、`Siril-SCUNet Denoise`、`SASP Aberration API` 任一成功，阶段保持 `ok`（除非其他导出步骤失败）
- 仅当以上路径都不可用时，阶段记为 `degraded`（当 `aberration_api_enabled=True`）或记录跳过信息（当 `aberration_api_enabled=False`）
- 阶段结果 `message` 会写入最终降噪与导出失败原因，便于日志排查

阶段检查点：`stage10_final.fit`

然后切回原工作目录导出三类文件；当前导出顺序为 TIFF、FITS、PNG。执行动态命名前会先检查 `OBJECT / STACKCNT / EXPTIME / DATE-OBS`。完整时使用 Siril 动态模板；仅缺少 `STACKCNT` 等次要字段时，用已有的 `OBJECT / EXPTIME / DATE-OBS` 生成安全字面名称（例如 `M_42_60sec_20260216_140234_processed`）；目标身份或日期也不可用时才退回通用名称，避免未解析占位符和不同任务相互覆盖：

- TIFF：优先使用基于 FITS 元数据的动态命名，失败时回退 `result_processed.tif`
- PNG：优先动态命名，失败时回退 `result_processed.png`
- FITS：优先动态命名并追加 `_final`，失败时回退 `result_final.fit`

PNG 默认直接使用已通过 Stage 7 门控的非线性渲染，不再对 TIFF/FITS 对应画面做第二次 autostretch；只有 Stage 7 未验收、导出诊断回退源时才执行 `autostretch -linked`，避免 PNG 与 TIFF/FITS 出现不同亮度和色彩关系。

若 `final_quality_report.json` 设置 `needs_conservative_rerun=true`，Stage 9 已因不安全 Starless 旁路回星、`starmask_stretch_failed=true`，回星契约表明 `stars_required=true` 但 `stars_applied=false`，或显式设置 `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT=1`，本轮只导出 `result_review.tif/png` 与 `result_review_final.fit`；线性续跑对应 `result_review_linear.*`。已在进入 Stage 10 前确定的复核输出会跳过耗时最终降噪。复核产物不会占用普通 `result_processed/result_final` 名称，Stage 11 也不会把它提升为 AI 正常成品。

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
   - Stage 11 的正向饱和度、红/蓝通道增益继续受 Stage 4 色彩策略和剩余饱和度预算限制。
6. 通过门控后导出：
   - `result_processed_ai.tif`
   - `result_processed_ai.png`
   - `result_final_ai.fit`

AI advisor 模式：

- `SEESTAR_AI_ADVISOR_MODE=text`（默认）保持现有纯指标/文本参数顾问。
- `SEESTAR_AI_ADVISOR_MODE=multimodal` 会把当前阶段的安全拉伸预览随参数请求发送给兼容图片输入的 OpenAI-compatible 模型；图片请求失败或模型不支持时，参数建议自动回退文本请求。
- 多模态模式会把预览图数据发送到用户配置的 `SEESTAR_AI_ENDPOINT`；需要完全离线或不希望外发图像时保持 `text`。
- Stage 3、6、7、8、11 的 review bundle 记录候选列表、算法选择状态和 `not_requested / unavailable / accepted / review_required / rejected` 视觉验收状态。未启用多模态时为 `not_requested`，同步调用无结论或失败时为 `unavailable`，不再留下无含义的长期 `pending`；视觉结论始终非阻断，不替代本地质量门控。

AI 艺术衍生实验（Stage 11 后的非正式分支）：

- 默认关闭，只有 `SEESTAR_AI_ARTISTIC_DERIVATIVE_ENABLED=1` 且独立 endpoint/model/key 全部配置时运行；不复用 advisor/Stage11 凭据。
- 固定只读 `process/stage10_final.fit`，生成显示预览后调用 OpenAI-compatible `/v1/images/edits`；接受 `b64_json` 或结果 URL。
- 只写 `<work_dir>/ai_artistic_derivative/source_preview.png`、`result_artistic_derivative.*` 和 `artistic_report.json`。
- 返回图片不会载入 Siril，不参与候选、验收、回混、导出或任何 Stage 状态；接口失败只写实验报告。
- 该输出明确标记为非科学、非校准的艺术衍生图，不能替代 Stage 10/11 天文处理结果。

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

`_apply_runtime_env_overrides()` 在 `run()` 入口处（连接 Siril 之前）集中读取并覆盖配置。下表变量均为可选；未设置时使用 `PipelineConfig` 和 `resources/default.env` 中列出的默认值：

| 环境变量 | 对应配置 | 默认值 | 说明 |
|---|---|---|---|
| `SEESTAR_DEBUG_MODE` | `debug_mode` | `False` | 开启后保留 stage* 中间文件 |
| `SEESTAR_INPUT_MODE` | `input_mode` | `auto` | `auto` 正常流程；`stage2_corrected_resume` 从 `stage2_corrected.fit` 进入 stage 3；`result_linear_resume` 从 `result_linear.fit` 进入 stage 6 |
| `SEESTAR_OUTPUT_FORMAT` | `output_format` | `all` | 最终导出格式，可为 `all` 或逗号分隔 `tif/png/fit` |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` | `workflow_plugin_probe_enabled` | `False` | 启用后允许 `run_first_available_command` 路径探测更广泛的实验插件命令；Stage 6 的 SASP Dark Star fallback 可在关闭时运行，Stage 8 固定使用 SASP Python API，不探测未注册的实验性 `sasp_*` Siril 命令 |
| `SEESTAR_STAGE4_PLATESOLVE_ENABLE` | `stage4_platesolve_enabled` | `True` | 默认执行 `platesolve -noflip -focal=160 -pixelsize=2.90 -catalog=gaia -order=3`，保留原始图像方向 |
| `SEESTAR_STAGE4_PLATESOLVE_CATALOGS` / `SEESTAR_STAGE4_PLATESOLVE_ORDER` | env only | `gaia` / `3` | Stage 4 platesolve 星表候选顺序和多项式阶数 |
| `SEESTAR_STAGE4_FILTER_HINT` | env only | `""` | FITS `FILTER` 缺失时补充宽带/双窄带通道语义；GUI 可选择自动、无滤镜、Seestar LP 或双窄带 Ha/OIII |
| `SEESTAR_STAGE4_PCC_TIMEOUT_SEC` | `stage4_pcc_timeout_sec` | `30` | 单次在线 Gaia PCC 超时；不重试、不切换星表 |
| `SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE` | `stage4_local_star_wb_enabled` | `True` | PCC 回退或窄带策略下是否允许恒星软遮罩局部色彩恢复 |
| `SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS` / `SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT` | `stage4_local_star_wb_min_pixels` / `stage4_local_star_wb_gain_limit` | `32` / `1.20` | 恒星样本下限与软遮罩内部单通道增益上限 |
| `SEESTAR_STAGE4_LOCAL_STAR_MASK_RADIUS` / `SEESTAR_STAGE4_LOCAL_STAR_MASK_COVERAGE_MAX` | `stage4_local_star_mask_radius` / `stage4_local_star_mask_coverage_max` | `2` / `0.12` | 星翼软遮罩半径与最大覆盖率；超限时不修改像素 |
| `SEESTAR_ABERRATION_API_ENABLE` | `aberration_api_enabled` | `False` | 启用 SASP Aberration API 路径 |
| `SEESTAR_ABERRATION_PROVIDER` | — | — | SASP Aberration API provider 选择；可设 `cpu` 强制 CPU |
| `SEESTAR_DENOISE_ENABLE` | `denoise_enabled` | `False` | 启用内置线性降噪 |
| `SEESTAR_DENOISE_FORCE` | `_force_denoise_enabled` | — | 自动调参后强制覆盖 denoise_enabled |
| `SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE` | `stage5_graxpert_deconvolution_enabled` | `True` | 是否在 Stage 5 优先尝试 GraXpert Object Deconvolution；关闭后保留 Siril RL 反卷积路径 |
| `SEESTAR_AI_ENABLED` | `ai_post_enabled` | `False` | AI 总开关：启用阶段 6 去星、阶段 8 增强诊断/参数建议与阶段 11 AI 副本；阶段 7 固定本地两个主候选和非 AI 色度救援 |
| `SEESTAR_AI_ENDPOINT` | `ai_endpoint` | `""` | AI API endpoint |
| `SEESTAR_AI_MODEL` | `ai_model` | `""` | AI 模型名 |
| `SEESTAR_AI_API_KEY` | `ai_api_key` | `""` | AI API 密钥 |
| `SEESTAR_AI_PROMPT` | `ai_prompt` | `""` | 自定义 AI 提示词 |
| `SEESTAR_AI_TIMEOUT_SEC` | `ai_timeout_sec` | `90` | API 超时（限幅 15–300） |
| `SEESTAR_AI_STRENGTH` | `ai_strength` | `0.12` | AI 混合强度（限幅 0.05–0.25） |
| `SEESTAR_AI_ADVISOR_MODE` | `ai_advisor_mode` | `text` | AI 顾问模式：`text` 或 `multimodal`；多模态参数请求失败自动回退文本 |
| `SEESTAR_AI_STAGE6_ENABLE` | `ai_stage6_enabled` | `True` | 保留的拉伸顾问开关；当前阶段 7 主体拉伸固定使用本地两个主候选，不扩展 AI 候选，色度救援始终为确定性本地路径 |
| `SEESTAR_AI_STAGE7_ENABLE` | `ai_stage7_enabled` | `True` | 单独控制阶段 7 AI SyQon 参数计划、诊断与重试择优 |
| `SEESTAR_AI_STAGE8_ENABLE` | `ai_stage8_enabled` | `True` | 单独控制阶段 8 AI Starless 参数计划、蓝色修正与保守重跑 |
| `SEESTAR_AI_ARTISTIC_DERIVATIVE_ENABLED` | `ai_artistic_derivative_enabled` | `False` | 完全隔离的艺术衍生实验开关；不受 Stage11 advisor 开关隐式开启 |
| `SEESTAR_AI_ARTISTIC_ENDPOINT` | `ai_artistic_endpoint` | `""` | 独立图片编辑 endpoint/base URL；base URL 自动补 `/v1/images/edits` |
| `SEESTAR_AI_ARTISTIC_MODEL` | `ai_artistic_model` | `""` | 独立图片编辑/生成模型名 |
| `SEESTAR_AI_ARTISTIC_API_KEY` | `ai_artistic_api_key` | `""` | 独立 API key，不回退复用 advisor key |
| `SEESTAR_AI_ARTISTIC_PROMPT` | `ai_artistic_prompt` | `""` | 艺术衍生提示词；为空使用带非科学声明的默认提示词 |
| `SEESTAR_AI_ARTISTIC_TIMEOUT_SEC` | `ai_artistic_timeout_sec` | `180` | 图片编辑请求超时，限幅 30–600 秒 |
| `SEESTAR_STAGE7_QUALITY_RETRY_MAX` | `stage7_quality_retry_max` | `2` | 阶段 6 去星参数优化预算配置（限幅 0–3）；变量名保留 `stage7` 以兼容旧配置 |
| `SEESTAR_STAGE7_SKIP_UNREADY_STARLESS` | `stage7_skip_unready_starless` | `True` | 兼容门控判定不适合去星时是否跳过 SyQon/SASP |
| `SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH` | `stage7_soft_starless_asinh_stretch` | `1.35` | 旧预拉伸重试兼容字段，当前正式 Stage 6 不使用 |
| `SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX` | `stage7_bright_nebula_halo_residue_score_max` | `0.60` | M42/亮核心星云 halo 验收上限 |
| `SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH` | `stage7_starless_repair_strength` | `0.68` | starless 小尺度残星修复强度（限幅 0–0.85） |
| `SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH` | `stage7_starless_halo_repair_strength` | `0.70` | starless 亮星 halo 修复强度（限幅 0–0.90） |
| `SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH` | `stage7_starless_chroma_denoise_strength` | `0.55` | starless 背景彩噪修复强度（限幅 0–0.90） |
| `SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_REDUCTION_MIN` | `stage7_starless_repair_chroma_reduction_min` | `0.20` | 综合色噪专用验收路径要求的最低下降比例（限幅 0.05–0.80） |
| `SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_DELTA_MIN` | `stage7_starless_repair_chroma_delta_min` | `0.0005` | 综合色噪专用验收路径要求的最小绝对下降量 |
| `SEESTAR_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX` | `stage7_starmask_diffuse_residual_ratio_max` | `0.08` | starmask 清理后弥散残留比例硬上限；超限禁用该候选星点层，不回退使用 raw 回星 |
| `SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE` | `stage7_starless_pixel_repair_enabled` | `True` | 是否启用阶段 6 starless 像素修复 |
| `SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR` | `stage8_force_conservative_after_stage7_repair` | `True` | 阶段 6 修复或不安全时是否强制 Stage 8 conservative skip |
| `SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE` | `stage9_starmask_stretch_enabled` | `True` | 阶段 9 像素回混前是否把 starmask 拉伸到非线性域 |
| `SEESTAR_STAGE9_STARMASK_ADAPTIVE_STRETCH_ENABLE` | `stage9_starmask_adaptive_stretch_enabled` | `True` | 混合星场启用单调多锚点曲线；非混合场保留自适应 Asinh 回退 |
| `SEESTAR_STAGE9_COMPACT_STARMASK_ENABLE` | `stage9_compact_starmask_enabled` | `True` | Asinh 前只保留连通紧致星核/窄星翼；覆盖异常时先重建严格紧致层再降强度 |
| `SEESTAR_STAGE9_SOURCE_STAR_DETAIL_PERCENTILE` | `stage9_source_star_detail_percentile` | `98.0` | 从原始含星图局部细节建立独立星表的百分位阈值（限幅 97–99.5） |
| `SEESTAR_STAGE9_SOURCE_COMPONENT_DENSITY_MAX` / `SEESTAR_STAGE9_SOURCE_SINGLE_PIXEL_RATIO_MAX` | 对应同名小写配置 | `2500.0` / `0.20` | 单像素比例独立超限，或组件密度超限且单像素比例高于 `0.10` 时，先提高细节百分位；达到 `99.5` 仍有污染则 fail-closed，避免把真实高密星场误判成噪声 |
| `SEESTAR_STAGE9_STAR_REFERENCE_SIGMA` | `stage9_star_reference_sigma` | `5.0` | 原始 starmask 星点目录阈值：背景加该倍数噪声标准差（限幅 3–8） |
| `SEESTAR_STAGE9_COMPACT_WEAK_STAR_RETENTION_MIN` | `stage9_compact_weak_star_retention_min` | `0.80` | compact 支持层必须保留的弱星组件数量比例下限 |
| `SEESTAR_STAGE9_MIXED_STAR_PEAK_RATIO_MIN` | `stage9_mixed_star_peak_ratio_min` | `4.0` | 亮星组/弱星组峰值中位数达到该倍率时才启用多锚点曲线 |
| `SEESTAR_STAGE9_MIXED_STAR_WEAK_COUNT_MIN` / `SEESTAR_STAGE9_MIXED_STAR_BRIGHT_COUNT_MIN` | 对应同名小写配置 | `20` / `3` | 启用混合亮星/弱星场多锚点曲线所需的最少组件数 |
| `SEESTAR_STAGE9_STARMASK_ASINH_STRETCH_MAX` | `stage9_starmask_asinh_stretch_max` | `1000.0` | 自适应 Asinh 强度反解上限 |
| `SEESTAR_STAGE9_STARMASK_FAINT_TARGET` / `SEESTAR_STAGE9_STARMASK_MID_TARGET` / `SEESTAR_STAGE9_STARMASK_BRIGHT_TARGET` / `SEESTAR_STAGE9_STARMASK_PEAK_TARGET` | 对应同名小写配置 | `0.26 / 0.50 / 0.75 / 0.90` | 多锚点曲线弱、中、亮、极亮星的有序输出目标；运行时保证至少相差 0.03 |
| `SEESTAR_STAGE9_STARMASK_CHROMA_REGULARIZATION_ENABLE` | `stage9_starmask_chroma_regularization_enabled` | `True` | 多锚点拉伸时以邻域星色约束低信号星翼，避免把单通道残差放大成蓝紫色块 |
| `SEESTAR_STAGE9_STARMASK_FAINT_CHROMA_MAX` / `SEESTAR_STAGE9_STARMASK_BRIGHT_CHROMA_MAX` | 对应同名小写配置 | `0.35` / `0.60` | 微弱星翼与亮星核心允许的最大通道跨度；随原始星点亮度连续过渡 |
| `SEESTAR_STAGE9_STARMASK_PREDICTED_CHANGE_RATIO_MAX` | `stage9_starmask_predicted_change_ratio_max` | `0.30` | 自适应拉伸求解阶段的预测显著变化覆盖上限；还会受正式 Stage 9 门控上限约束 |
| `SEESTAR_STAGE9_STARMASK_ASINH_STRETCH` / `SEESTAR_STAGE9_STARMASK_ASINH_OFFSET` | `stage9_starmask_asinh_stretch` / `stage9_starmask_asinh_offset` | `2.00` / `0.001` | 自适应统计不可用或关闭时的固定回退参数 |
| `SEESTAR_STAGE7_TARGET_LOCAL_METRICS_ENABLE` | `stage7_target_local_metrics_enabled` | `True` | 是否把目标局部核心、弱结构和暗云指标纳入 Stage 7 候选门控 |
| `SEESTAR_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX` | `stage7_local_core_clip_ratio_max` | `0.12` | 亮核局部裁剪比例上限 |
| `SEESTAR_STAGE7_LOCAL_FAINT_SNR_MIN` / `SEESTAR_STAGE7_LOCAL_DARK_SEPARATION_MIN` | 对应同名小写配置 | `0.25` / `0.001` | 弱结构相对背景 SNR 与暗云明暗分离下限 |
| `SEESTAR_STAGE7_STRETCH_CHROMA_NOISE_SCORE_MAX` | `stage7_stretch_chroma_noise_score_max` | `0.34` | 正式拉伸候选的背景绝对综合色噪上限 |
| `SEESTAR_STAGE7_STRETCH_BACKGROUND_MOTTLING_SCORE_MAX` | `stage7_stretch_background_mottling_score_max` | `0.45` | 正式拉伸候选的低频背景斑驳上限 |
| `SEESTAR_STAGE7_STRETCH_CHROMA_LOAD_GROWTH_MAX` | `stage7_stretch_chroma_load_growth_max` | `1.35` | 拉伸后综合色偏差相对背景亮度的最大放大倍数 |
| `SEESTAR_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_MAX` | `stage7_stretch_chroma_load_low_absolute_max` | `0.05` | 候选绝对 chroma load 不高于此值时，不因极低线性基线放大出的相对增长倍数单独拒绝；绝对色噪和斑驳门控仍生效 |
| `SEESTAR_STAGE7_PREVIEW_TARGET_P50_MIN_RATIO` | `stage7_preview_target_p50_min_ratio` | `0.55` | preview 标定有效时，正式候选实际 P50 相对目标 P50 的最低比例（限幅 0.25–0.90） |
| `SEESTAR_STAGE7_PREVIEW_TARGET_P50_MAX_RATIO` | `stage7_preview_target_p50_max_ratio` | `1.50` | preview 标定有效时，正式候选实际 P50 相对目标 P50 的最高比例（限幅 1.00–3.00） |
| `SEESTAR_STAGE7_CHROMA_RESCUE_ENABLE` | `stage7_chroma_rescue_enabled` | `True` | 两个主候选仅因背景色度门控失败时，是否允许默认 `0.35 / 0.55 / 0.65` 三档背景限定、保亮度的确定性救援；生成后与主候选统一质量择优 |
| `SEESTAR_STAGE9_QUALITY_GATE_ENABLE` | `stage9_quality_gate_enabled` | `True` | 是否启用 Stage 9 保存前回混门控与拒绝回滚 |
| `SEESTAR_STAGE9_HIGHLIGHT_CLIP_RATIO_MAX` / `SEESTAR_STAGE9_HIGHLIGHT_CLIP_GROWTH_MAX` | 对应同名小写配置 | `0.015` / `0.006` | 回混后高光裁剪总量与相对 Stage 8 增长上限 |
| `SEESTAR_STAGE9_BRIGHT_PIXEL_GROWTH_MAX` / `SEESTAR_STAGE9_BACKGROUND_LIFT_MAX` | 对应同名小写配置 | `0.025` / `0.010` | 亮像素扩张与暗背景中位抬升上限 |
| `SEESTAR_STAGE9_BACKGROUND_MOTTLING_GROWTH_MAX` | `stage9_background_mottling_growth_max` | `1.35` | Stage 9 候选相对 Stage 8 的低频背景斑驳评分增长倍数上限 |
| `SEESTAR_STAGE9_MOTTLING_EXEMPTION_CHANGED_PIXEL_RATIO_MAX` | `stage9_mottling_exemption_changed_pixel_ratio_max` | `0.12` | 低绝对斑驳豁免允许的最大显著变化覆盖；超过时仍执行斑驳增长门控 |
| `SEESTAR_STAGE9_CHANGED_PIXEL_RATIO_MAX` / `SEESTAR_STAGE9_DARKENING_RATIO_MAX` | 对应同名小写配置 | `0.35` / `0.005` | 显著变化覆盖和异常变暗像素占比上限 |
| `SEESTAR_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN` | `stage9_weak_star_recovery_ratio_min` | `0.70` | 候选相对 Stage 8 必须恢复的弱星组件数量比例下限 |
| `SEESTAR_STAGE9_STAR_RECOVERY_RATIO_MIN` | `stage9_star_recovery_ratio_min` | `0.75` | 候选相对 Stage 8 必须恢复的全部星点组件数量比例下限 |
| `SEESTAR_STAGE9_WEAK_STAR_SCREEN_INTENSITY_MIN` | `stage9_weak_star_screen_intensity_min` | `0.40` | 弱星 Screen 强度下限（限幅 0.10–1.05）；默认允许质量失败后的降强度阶梯真实生效，弱星恢复率门控防止过度降级 |
| `SEESTAR_STAGE9_STAR_SUPPORT_RATIO_MAX` / `SEESTAR_STAGE9_UNMATCHED_CHANGED_RATIO_MAX` | 对应同名小写配置 | `0.12` / `0.01` | 实际 Alpha 星点支持覆盖与支持区外显著变化比例上限 |
| `SEESTAR_STAGE9_CHROMATIC_ADDITION_PEAK_MIN` / `SEESTAR_STAGE9_CHROMATIC_ADDITION_SATURATION_MIN` / `SEESTAR_STAGE9_CHROMATIC_ADDITION_RATIO_MAX` | 对应同名小写配置 | `0.02` / `0.70` / `0.003` | 统计 Stage 9 相对 Stage 8 的强新增且高饱和像素；占比超限时拒绝候选，并在最终质量报告中二次拦截 |
| `SEESTAR_STAGE9_STAR_APERTURE_RECOVERY_RATIO_MIN` / `SEESTAR_STAGE9_STAR_WING_RECOVERY_RATIO_MIN` | 对应同名小写配置 | `0.75` / `0.65` | 独立星表的 7×7 孔径通量与星翼最低恢复比例 |
| `SEESTAR_STAGE9_RESIDUAL_DARK_HOLE_RATIO_MAX` | `stage9_residual_dark_hole_ratio_max` | `0.15` | 星点支持区回混后残余暗坑比例上限 |
| `SEESTAR_STAGE9_HOLLOW_STRUCTURE_DELTA_MIN` / `SEESTAR_STAGE9_NEW_HOLLOW_STRUCTURE_AREA_MAX` | 对应同名小写配置 | `0.05` / `64` | 新增闭合环/空心结构的最小亮度变化与最大允许空心面积 |
| `SEESTAR_STAGE9_LOCAL_COMPONENT_PEAK_MIN` / `SEESTAR_STAGE9_LOCAL_COMPONENT_AREA_MAX` | 对应同名小写配置 | `0.01` / `256` | Stage 9 局部新增连通块检测阈值与单块面积上限 |
| `SEESTAR_STAGE9_LOCAL_COMPONENT_ASPECT_RATIO_MAX` / `SEESTAR_STAGE9_LOCAL_COMPONENT_FILL_RATIO_MIN` / `SEESTAR_STAGE9_LOCAL_SINGLE_PIXEL_RATIO_MAX` | 对应同名小写配置 | `3.0` / `0.15` / `0.20` | 非星形细长/稀疏连通块和单像素碎片比例门限 |
| `SEESTAR_STAGE9_LOCAL_CYAN_BLUE_PEAK_MIN` / `SEESTAR_STAGE9_LOCAL_CYAN_BLUE_SATURATION_MIN` / `SEESTAR_STAGE9_LOCAL_CYAN_BLUE_COMPONENT_AREA_MAX` | 对应同名小写配置 | `0.01` / `0.50` / `64` | 局部青蓝增量团块的峰值、色彩跨度与最大单块面积 |
| `SEESTAR_STAGE9_CORE_PERCENTILE` / `SEESTAR_STAGE9_CORE_COLOR_JUMP_MIN` / `SEESTAR_STAGE9_CORE_COLOR_JUMP_COMPONENT_AREA_MAX` | 对应同名小写配置 | `90` / `0.10` / `64` | 星云核心定义百分位、归一化 RGB 突变阈值及最大突变块面积 |
| `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT` | `force_review_only_output` | `0` | 可选；不设置时使用配置默认值，设为 `1` 时仅生成 `result_review*`，不占用正式结果名 |
| `SEESTAR_STAGE10_CHROMA_FOCUS_SCORE_MIN` / `SEESTAR_STAGE10_SEPARATE_CHROMA_SCORE_MIN` | 对应同名小写配置 | `0.34` / `0.70` | Stage 10 选择 chroma 色度专用模式和 severe separate 模式的综合色噪门限 |
| `SEESTAR_STAGE10_FULL_BG_STD_MIN` / `SEESTAR_STAGE10_FULL_MOTTLING_SCORE_MIN` | 对应同名小写配置 | `0.018` / `0.45` | 色噪同时伴随亮度噪声或低频斑驳时切换为 full 模式的门限 |
| `SEESTAR_STAGE10_STAGE9_LOCAL_COLOR_RISK_STRENGTH` | `stage10_stage9_local_color_risk_strength` | `1.0` | Stage 10 按 Stage 9 局部青蓝/核心颜色风险缩减正向最终饱和度的强度（0–1） |
| `SEESTAR_SIRIL_PLUGIN_DIR` | `siril_plugin_dir` | — | 插件目录路径 |
| `SEESTAR_COSMIC_CLASSIC_ENABLE` | — | `1` | 是否启用 bundled classic CosmicClarity executable 路径；失败时仍可回退 Native/SCUNet |
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
- `stage5_linear.fit`、`stage5_graxpert_deconv.fit` / `stage5_deconv.fit`（二选一，可选）、`stage5_denoised.fit`（兼容别名）
- `stage6_starless_quality.json`（兼容别名：`stage7_quality.json`）
- `stage6_starless.fit`（兼容别名：`stage7_starless.fit`）
- `stage7_cand_a.fit`、`stage7_cand_b.fit`、`stage7_preview_ref.fit`
- `stage7_stretch_quality.json`、`stretch_candidates_report.json`
- `stage7_stretched.fit`
- `stage7_5_pre_starless_gate_report.json`（兼容别名：`pre_starless_gate_report.json`）
- `starless.fit`
- `stage8_input_starless.fit`、`stage8_quality.json`、`stage8_enhancement_report.json`
- `starless_enhanced.fit`
- `stage8_enhanced.fit`
- `starmask_raw.fit`、`starmask_clean.fit`（若清理通过）、`starmask_external_raw.fit`（若导入外部层）、`starmask.fit`（当前层兼容别名）
- `starmask_stretched.fit`
- `stage9_remix_quality.json`
- `stage9_remixed.fit`
- `stage10_final.fit`
- `result_linear.fit`（工作目录中的拉伸前线性中间文件）
- `sasp_starless_input.fit`、`sasp_starmask_input.fit`（工作目录中的 SASP 交换文件）
- `stage11_ai_source.fit`、`stage11_ai_output.png`、`stage11_ai_output_fit.fit`、`stage11_ai_blended.fit`（仅阶段 11 运行时出现）
- `<work_dir>/ai_artistic_derivative/source_preview.png`、`result_artistic_derivative.*`、`artistic_report.json`（仅隔离艺术实验显式开启时出现；不属于 Stage 产物）
- `review_bundles/<stage>/before.png`、`after.png`、`difference.png`、`signed_difference.png`、`review.json`：Stage 3-10 和可选 Stage 11 的统一视觉复核包；包含 before/after 指标、指标差值、目标策略上下文和待复核 checklist。Stage 3、6、7、8、9、11 还包含候选及觉验收状态；Stage 9 为确定性 P0 门控的非阻断证据包，多模态开启也不改写门控结论。生成或视觉调用失败只记录警告，不阻断离线主流程；可用 `PipelineConfig.review_bundle_enabled=False` 关闭。

是否保留取决于：

- `checkpoint_mode=False`：保存所有阶段检查点
- `checkpoint_mode=True`：只保存关键检查点
- `debug_mode=True`：保留 `stage*` 等中间文件，但会清理 `*lightsrc*` 预处理序列
- `debug_mode=False`：清理阶段会删除大多数中间文件

无论是否开启 Debug，只要存在轻量诊断文件，清理前都会在工作目录原子生成 `seestar_diagnostics.zip`。归档包含 JSON/JSONL、日志、CSV/TXT 和 review bundle PNG，不包含 FITS 等大体积科学图像；`manifest.json` 列出归档文件。归档失败只记警告，不中止主任务。

Debug 模式下，每次 `_save_stage_output()` 成功保存阶段 FIT 后，会按统一格式输出质量指标：

- 日志行：`[STAGE_QUALITY_METRICS] schema=seestar.stage_quality.v1 stem=<stage_stem> ...`
- 单阶段文件：`process/<stage_stem>_quality_metrics.json`
- 汇总文件：`process/stage_quality_metrics.jsonl`

JSON 结构固定包含 `schema`、`sequence`、`stem`、`file`、`metrics`、`features`。`metrics` 使用 `QualityMetrics` 字段（背景、黑场、高光裁剪、星点、饱和度、微对比、蓝偏），`features` 使用 `ImageFeatures` 字段（边缘黑边、全局暗像素、目标面积、弥散比例、亮核比例等），便于后续比较阶段间变化。

### 8.3 清理阶段

`cleanup()` 在非调试模式下会：

- 先生成工作目录下的 `seestar_diagnostics.zip`，再通过 `_cleanup_lightsrc_intermediates()` 清理 `process/` 内 `*lightsrc*` 相关文件和 `_light_input/` 目录
- 清理 `process/` 内大部分 `fit/fits/seq/log/csv/lst`
- 默认保留 `starless.fit`、各 raw/clean starmask、`starmask.fit` 和 `starmask_stretched.fit`（存在时）
- 如果存在拉伸名，还保留对应的拉伸星点蒙版检查点
- 不会删除工作目录中的 `result_linear.fit`
- 阶段 11 临时文件会在非调试模式下由 `stage11_ai_postprocess.py` 主动删除
- 若阶段 11 成功，工作目录会新增 `result_processed_ai.tif/png` 与 `result_final_ai.fit`
- 若目录已空则直接删除 `process/`

因此，默认模式下最终更强调"输出整洁"，而不是保留完整现场。

## 9. 维护时最值得注意的点

1. 阶段顺序是核心契约：1-10 为稳定主链，阶段 11 仅可选追加，不可插到 1-10 中间。
2. 自动调参是"重写配置"，不是"给某个阶段附加参数"，排查参数异常时要先看自动调参日志。
3. Stage 3/4 preflight 的 `target_profile.json` / `pipeline_policy.json` 是 Stage 3-7 的策略输入；新增 target type 或候选时要同步内置 policy、配置文件和候选 evaluator。`stage6_5_pre_starless_gate` 只是历史 policy key。
4. Stage 7 候选不能只看命令成功，必须看 `quality_ok`、`pixel_stats` 和 `normal_selected`；degraded fallback 是可导出兜底，不是正常最佳结果。
5. `workflow_plugin_probe_enabled` 控制广泛插件探测开关：为 `False` 时大部分插件命令不会被尝试，只有标记 `allow_when_probe_disabled=True` 的命令（如 Stage 6 的 SASP Dark Star fallback）仍可执行；阶段 8 默认走 SASP Python API，不再先探测实验性 `sasp_*` Siril 命令。
6. Stage 5-11 及 Stage 7 兼容门控检查点都有明显降级路径，改动时不能只看成功路径。
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
- 色彩偏差：看阶段 4 的 `channel_policy`、`pcc.policy_status`、`quality_gate.rejection_reasons`、`rollback.restored`、局部恒星软遮罩覆盖率与 `requires_review`
- Stage5 降噪无效：看 `stage5_linear_report.json` 的 `denoise.method`，确认 Siril `denoise -indep` 是否失败以及 CosmicClarity 低强度回退是否命中
- Stage5 反卷积无效：看 `stage5_linear_report.json` 的 `deconvolution.applied` 和日志中的 `findstar/makepsf/rl` 错误；失败时流程会回退使用 `stage5_linear.fit`
- 拉伸异常：看阶段 7 的 `stage7_stretch_quality.json` 和 `stretch_candidates_report.json`，确认 `stage7_cand_a` / `stage7_cand_b` 哪个被选中、是否为 degraded fallback，以及 `pixel_stats` 是否命中黑场/白场/动态范围门控
- 去星前直接跳过或输入过保守：看 Stage 7 兼容检查点的 `stage7_5_pre_starless_gate_report.json`、`ready_for_starless` 和 `recommended_starless_input`
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
