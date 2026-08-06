# Seestar Superimpose

面向 Seestar 天文图像的自动后期处理工具。以 **Siril 1.4+** 为核心，结合 `SyQon-Starless`、SASP/CosmicClarity 等插件链路，把 FITS/XISF 母版或 Seestar Light 自动导出为 TIFF/PNG/FITS。提供 macOS GUI 与离线打包。

系统要求：**macOS 14.0 或更高版本**，仅支持 **Apple Silicon（arm64）**，不支持 Intel Mac。

## Key Files

| Path | Role |
|---|---|
| `pipeline/seestar_Superimpose.py` | stage 1-10 主流程编排 |
| `pipeline/stages/` | stage 1-10 分离实现模块 |
| `pipeline/stage_contracts.py` / `pipeline/task_plan.py` | 阶段、正式断点、文件命名与冻结计划契约 |
| `pipeline/input_discovery.py` / `pipeline/task_workspace.py` | 输入识别、Light 分组、只读来源与任务历史 |
| `pipeline/stage11_ai_postprocess.py` | 已停用的隔离实验模块，不进入产品运行时 |
| `pipeline/ai_artistic_derivative.py` | 已停用的隔离艺术衍生实验模块 |
| `pipeline/seestar_Superimpose_workflow.md` | 详细流程说明 |
| `gui/seestar_gui_app.py` | GUI 入口 |
| `gui/seestar_gui_dev.py` | 使用系统 Siril/现有 seed 的免打包开发入口 |
| `gui/seestar_pipeline_dev.py` | 不启动 GUI 的源码核心流水线验证入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 主窗口与 Siril worker |
| `gui/ui_theme.py` / `gui/ui_platform.py` | 共享设计令牌、系统浅深色主题与桌面平台适配 |
| `build/build_macos_app.sh` | macOS 打包 |
| `requirements.txt` / `requirements.lock` | 直接依赖约束 / pip-tools 生成的完整依赖锁 |
| `resources/default.env` | 项目内默认 runtime env；无需手工 export |
| `resources/ai.env` | 开发者试用模型的本地构建输入；须为 `0600`，Key 不再明文复制进 App |
| `INTEGRATION_README.md` | 内部集成/runtime/验证细节 |

## Pipeline Summary

主流程是 starless-first 后期链路：输入统一和裁切 -> 输入线性状态判定 -> 冻结主目标/策略与处理计划 -> 条件背景提取、自动设备几何/platesolve、SPCC 优先物理校色（宽带异常时 PCC 回退；双窄带 HOO 艺术映射隔离）、噪声模型驱动线性反卷积/降噪 -> 线性去星与确定性星色修复 -> Starless 主体拉伸和本地曲线/蒙版增强 -> 受控回星 -> 最终降噪和科学/展示/编辑版导出。输入状态只接受可追溯证据，分为 `linear / nonlinear / unknown`；非线性或未知输入只执行安全裁切并进入复核导出，不运行破坏性的线性处理、去星或 Starless 增强。

Stage 3 不再用 `gradient_score / dirty_background_score` 的固定经验阈值决定跳过或执行。它先确认输入仍为线性数据，再建立显式源掩膜与无覆盖掩膜，从跨视场的真实天空区域选样，并冻结独立的拟合/留出集合；固定网格先冻结亮度、纹理和星点污染阈值，非弥散且真实天空占比充足的场景才允许用确定性暗斑搜索补充网格间候选，补充点仍须通过相同掩膜、阈值、全局最小间距和空间覆盖审计。弥散/暗弱星云或真实天空不足时禁用该增强，不能用“更多自动点”绕过保护门。留出天空的空间变化只有显著超过 patch 中值采样不确定度时才授权背景模型。方向性平面优先解释为加性光污染并使用减法；中心亮、边缘暗且径向模型显著更优时路由到 master-flat 复核；条带/walking noise 路由到标定或重叠加；真实天空不足时保留原图并要求复核。大星系、弥散/暗弱结构和天空覆盖受限场景将复杂度限制为一阶 Polynomial；RBF 与 GraXpert、ADBE、DBE、AutoBGE、NOX 等只作为受同一留出背景/RMS、目标通量和形态保真门约束的后备。所有候选都从同一不可变基线开始，失败、拒绝或切换前都会回滚。

Stage 5 不会在任务中联网下载 GraXpert 对象反卷积模型。自动模式优先使用 Seestar App 随包模型；随包模型缺失时读取本机 GraXpert 应用已安装的最新版模型，再尝试“处理参数…”里指定的官方 `model.onnx`、语义版本目录（如 `1.0.1`）或 `deconvolution-object-ai-models` 目录。App 会把有效模型只读链接到隔离运行目录；全部无效或缺失时安全回退 Siril RL。“自动加速”允许 GraXpert 通过 ONNX Runtime 自动选择 CoreML 等可用执行设备，初始化或推理失败时由插件回退 CPU；“CPU 兼容模式”会显式禁用该硬件加速。GUI 分开显示“反卷积”和“降噪”：若 GraXpert 已完成而降噪因低噪声或配置关闭被跳过，反卷积结果仍保存为 `stage5_linear.fit` 并继续传给 Stage 6。

Stage 6 会显式记录去星结果为 `accepted / target_bypass / rejected / tool_failed`。拒绝或工具失败时保留含星复核路径，不再把 passthrough 文件命名或解释为 Starless；Stage 8/9 跳过 Starless 专用逻辑，Stage 10 只生成复核产物。

亮发射/反射星云的 Stage 8 halo 门分为三级：不超过 `0.350` 正常增强，`0.350–0.600` 生成饱和度受限、无锐化的 masked 候选并由质量门决定，超过 `0.600` 或命中硬失败则安全旁路。受限候选保留为 `stage8_limited_candidate.fit`；拒绝时自动回滚。Stage 9 会把“使用 Stage 8 安全旁路源”与“Stage 9 自己使用降低强度/compact recovery 回退”分开显示。

产品流程固定在 Stage 10 结束。已停用的后续实验模块不会被 GUI 展示、预检、复制或执行，旧设置和内部参数也不能恢复入口；运行日志与阶段事件只接受 Stage 1-10。Stage 状态为 `ok / degraded / failed / skipped`；最终运行状态为 `success / partial_success / review_required / failed`。阶段状态由结构化执行、回退和旁路字段决定，不再根据日志说明文字中的 `skipped/fallback` 关键词猜测。

详细阶段顺序、检查点、质量门控和降级路径见 `pipeline/seestar_Superimpose_workflow.md`。

## GUI Usage

1. 启动 `SeestarSuperimpose.app`。首次启动或没有有效输入时会显示拖放空状态；已保存且仍存在的输入会直接进入任务设置。
   界面使用系统字体并跟随 macOS 浅色/深色外观；控件、状态与焦点样式来自同一套语义设计令牌。
2. 直接拖入一种输入：单个 `.fit/.fits/.fts/.xisf` 母版、包含 Light 的目录，或应用生成的产品任务目录。单文件始终从 Stage 1 只读导入；XISF 会先显示明确的等待提示，并在 Stage 1 转为任务内 FITS 后出现预览。Light 会递归发现，并按目标、滤镜、相机和几何尺寸拆为独立任务；Dark/Flat/Bias 首版不参与叠加。TIFF/PNG/JPEG 只进入复核路径并可直接预览。旧处理目录只有在 `processing-plan.json` 和 `pipeline-result.json` 互相匹配、断点声明为线性且 SHA-256 一致时才可只读迁移；迁移后仍从 Stage 1 安全导入。
3. 应用只显示一个自动处理计划，不提供手动阶段选择。只有带有效 `task-manifest.json`、`checkpoint-manifest.json`、阶段契约、累计配置指纹和产物 SHA-256 的产品任务，才可能从 Stage 1、2 或 5 后继续；普通目录和 `result_linear.fit` 文件名本身不能证明断点。展开高级设置后，“处理参数…”可配置输出格式、正式/复核输出、校色、滤镜、线性降噪、反卷积、GraXpert 模型和计算模式。参数修改后自动保存，并在开始时冻结。
4. 点击开始后，应用在后台计算来源 SHA-256，界面仍可响应并可停止；校验完成后建立相邻的 `SeestarSuperimpose/<task-id>/`。来源只记录绝对引用、大小和 SHA-256，不复制或改写；同一来源指纹复用同一任务，每次运行写入独立 `runs/<run-id>/`。多个 Light 分组严格串行执行。preflight 会分别检查运行环境卷与任务卷的磁盘空间，需求按来源大小和阶段峰值估算。
   运行环境准备在后台执行，可用“停止”取消；依赖锁、Siril Python ABI 与 App
   版本未变化时会复用已就绪 runtime，不重复执行 `pip install`。
   SyQon 与 CosmicClarity 模型直接从 App 或离线资源包只读使用，不再复制到
   `runtime_home`。
5. 快速 preflight 通过后，界面立即切换为只读运行视图：左侧显示冻结后的本次配置，中央只显示最新一张可靠预览，右侧完整列出 Stage 1-10。每个成功或降级阶段完成后刷新中央预览；跳过、失败或预览生成失败时保留上一张。当前运行阶段与预览来源阶段分开标注。
   Stage 0 和 Stage 1-6 会使用明确标注的链接屏幕拉伸，仅用于让线性数据快速可见，不写入 FITS 或改变流水线数据。Stage 7-10 显示阶段本身已有的非线性结果，不附加额外拉伸。中央预览支持适合窗口、1:1、滚轮缩放与拖动。
   阶段列表显示等待、运行中、已完成、已降级、失败或已跳过，并显示本阶段耗时和总耗时；不提供百分比或预计剩余时间。
   `CompletedWithWarning` 会显示黄色复核卡片，可直接打开最终质量报告。
6. 完成、失败或停止后仍停留在运行视图，可返回任务设置、按上次实际配置重新运行或打开结果目录；“打开结果”会打开 `latest-result.json` 指向且产物 SHA-256 仍有效的实际 run。失败会自动展开日志。Stage 1、2、5 一经线性状态和阶段结果验收，就会在正常清理前原子发布到任务级 `checkpoints/`，因此后续阶段失败仍可从最近可信断点继续。新结果发布后，保留策略只删除已签名旧 run 中哈希仍匹配的旧交付和中间目录；最新交付、当前质量报告、日志、正式断点与未登记文件保留。
7. 工具栏“历史记录”按任务分组显示新版中实际开始的运行，可按目标/输入名称搜索并按结果状态筛选。双击运行记录会复用只读处理页面显示仍可验证的阶段、预览和日志；“重新处理”只返回任务设置，不会直接执行。旧任务不会自动扫描或补录。空闲时可将一个验签通过且位于 `SeestarSuperimpose` 直接子目录下的完整任务移到 macOS 废纸篓；容器目录、符号链接、清单损坏或 `task_id` 不匹配的目标一律拒绝。

## Outputs

- `*.tif`：高质量结果
- `*.png`：预览；Stage 7 已验收的非线性结果不再二次拉伸，诊断回退源才使用 linked autostretch
- `*_final.fit`：归档 FITS
- `*_display_srgb.png`：独立 16-bit 展示版，写入 `sRGB/gAMA/cHRM`；不改动 Siril PNG 或 FITS
- `*_edit_srgb.tif`：独立 16-bit 可编辑版，内嵌 sRGB ICC；找不到有效 ICC 时不生成未标记替代品
- `process/managed_output_report.json` / `output_color_manifest.json`：记录受管理导出、容器元数据和科学 FITS 前后 SHA-256 不变性
- FITS 头完整时输出名包含目标、叠加数、曝光和拍摄时间；仅缺少 `STACKCNT` 等次要字段时，仍使用已有目标与拍摄时间生成可识别名称，避免退回通用名称覆盖其他任务
- `result_linear.fit`：Stage 5 线性交付文件；只有任务级清单同时验签时才能作为旧版只读迁移来源，文件名本身不触发续跑
- `SeestarSuperimpose/<task-id>/task-manifest.json`：来源只读引用、内容指纹与任务布局
- `SeestarSuperimpose/<task-id>/runs/<run-id>/`：每次运行独立的计划、日志、阶段文件与结果
- `SeestarSuperimpose/<task-id>/checkpoints/{stage1_prepared,stage2_corrected,stage5_linear}.fit`：仅有的正式跨运行断点；由 `checkpoint-manifest.json` 记录契约、配置指纹和 SHA-256
- `SeestarSuperimpose/<task-id>/results/latest-result.json`：最新成功/需复核 run 的签名索引；每个交付文件再次按 SHA-256 校验后才在 GUI 中打开
- `SeestarSuperimpose/<task-id>/results/retention.json`：记录已清理的旧交付/中间目录和因校验失败而保留的项目
- `~/Library/Application Support/SeestarSuperimpose/history-index.json`：GUI 原子维护的全局历史导航索引；只登记新版实际启动的 run，不能替代任务/运行清单验签，也不能单独授权删除文件
- `result_review*.tif/png/fit`：Stage 4 物理校色要求复核、最终质量门要求保守重跑，或显式设置 `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT=1` 时生成的复核产物；此时不会写普通 `result_processed/result_final` 名称，并会跳过耗时的 Stage 10 最终降噪链
- `process/stage*.fit`、`process/*.json`：开启“保留中间文件”后的阶段产物与诊断报告
- `process/final_quality_report.json`：Stage 10 最终质量报告；降级完成时可从 GUI 警告卡片打开
- `processing-plan.json`：变换前原子冻结的本轮计划，包含输入 SHA-256、输入状态、冻结主目标、通道语义、阶段动作、候选合约和规范化计划哈希
- `pipeline-result.json`：原子写出的最终运行清单，包含计划哈希、阶段结果、产物 SHA-256 和全局状态；`process/` 中保留镜像副本
- `process/review_bundles/stage9_star_remixing/`：Stage 8 底图与 Stage 9 选中/回滚结果的 before/after/difference 复核包，`review.json` 同时记录星表、starmask 标定和候选拒绝原因
- `process/*_quality_metrics.json`、`process/stage_quality_metrics.jsonl`：保留中间文件时输出的统一阶段质量指标，日志同步输出 `[STAGE_QUALITY_METRICS] schema=seestar.stage_quality.v1 ...`

完整中间文件清单见 `pipeline/seestar_Superimpose_workflow.md` 的“文件与目录行为”。

## Development Run

无需打包即可用系统 Siril 和已初始化的 Siril Python seed 启动源码 GUI：

```bash
cd /Users/mz/dev/aiseestart
.venv/bin/python gui/seestar_gui_dev.py \
  --siril-app /Applications/Siril.app \
  --siril-seed "$HOME/Library/Application Support/org.siril.Siril/siril"
```

两项参数均提供以上默认值，可直接运行
`.venv/bin/python gui/seestar_gui_dev.py`。launcher 会在临时目录创建资源覆盖层，
通过符号链接把系统 Siril、现有 seed 与项目 `resources/` 组合后交给原 GUI；
launcher 本身不会修改这些来源，也不会改变正式 App 入口。默认仍使用
`~/Library/Application Support/SeestarSuperimpose/runtime_home`，可通过
`--runtime-home /absolute/path` 指定独立开发 runtime。

只验证核心流水线输出、不启动 GUI 时：

```bash
cd /Users/mz/dev/aiseestart
.venv/bin/python -u -m gui.seestar_pipeline_dev \
  --work-dir "$HOME/SeeStar/sirildev" \
  --input-mode stage2_corrected_resume
```

该命令默认严格离线；只有明确需要在线 Gaia 解算时才添加 `--network`。

核心代码调整后的人工快速回归可使用专项脚本：

```bash
cd /Users/mz/dev/aiseestart
.venv/bin/python tests/manual_core_pipeline_smoke.py
```

无参数时可人工选择从 `stage2_corrected.fit`、`result_linear.fit` 或原始输入
开始。脚本复用上述无 GUI launcher，默认开启网络并保留调试中间文件，以便在
本地 Gaia 目录缺失时继续使用在线 Gaia 完成 Stage 4；完成后还会
校验本轮 `pipeline-result.json`、清单哈希和登记产物的 SHA-256。固定选择快速
验证 Stage 3-10 时可直接执行：

```bash
.venv/bin/python tests/manual_core_pipeline_smoke.py \
  --mode stage2_corrected_resume \
  --work-dir "$HOME/SeeStar/sirildev"
```

使用 `--dry-run` 可仅检查最终命令，`--no-debug` 可减少中间文件；需要验证严格
离线回退链路时添加 `--offline`（兼容别名 `--no-network`）。该脚本验证当前仓库核心源码，不替代发布前的
GUI 参数传递、打包资源和签名验证。

## Build

```bash
cd /Users/mz/dev/aiseestart
./build/build_macos_app.sh
```

Required packages:
- `packages/python-3.13.12-macos11.pkg`
- `packages/siril-1.4.4-arm64.dmg`

Optional bundled resources:
- `resources/ai.env`（开发者试用配置的构建输入；非空 Key 要求文件权限 `0600`）
- `resources/siril_plugins/`（可用 `bash resources/siril_plugins/download_siril_plugins.sh` 准备）

Default output: `release/SeestarSuperimpose.app`

构建脚本强制生成纯 `arm64` 主程序，并把 `LSMinimumSystemVersion=14.0`
写入 App 的 `Info.plist`；构建机必须是 Apple Silicon Mac。当前 Bundle 元数据为
`CFBundleIdentifier=StarunC`、`CFBundleShortVersionString=0.1`、
`CFBundleVersion=1`，运行时依赖缓存会使用 `0.1 (1)` 作为 App 版本指纹。

公开发行时建议传入固定签名身份：

```bash
./build/build_macos_app.sh \
  --codesign-identity "Developer ID Application: Your Name (TEAMID)"
```

默认 ad-hoc 签名只适合本地验证；固定 Developer ID 能让 App 更新后的 Keychain 访问身份更稳定。公证所需的 hardened runtime、timestamp 和 notarization 仍需单独配置。

默认构建仍为 Full Offline。也可把大型 wheels/模型拆为相邻的离线资源包：

```bash
./build/build_macos_app.sh --bundle-profile core
```

该命令生成 `SeestarSuperimpose.app` 和
`SeestarSuperimpose-OfflineResources/`；两者保持在同一目录即可自动发现。也可用
`--offline-resource-pack-dir` 指定资源包位置，或通过
`SEESTAR_OFFLINE_RESOURCE_ROOT` 指向已安装的资源包。

依赖更新后，GUI/App 锁仍使用 Python 3.13；Siril 插件锁必须使用与内置 Siril 一致的 Python 3.12：

```bash
python3.13 -m pip install -r requirements-dev.txt
PIP_CONFIG_FILE=/dev/null python3.13 -m piptools compile \
  --resolver=backtracking --strip-extras --allow-unsafe --generate-hashes \
  --index-url https://pypi.org/simple \
  --output-file requirements.lock requirements.txt
PIP_CONFIG_FILE=/dev/null python3.12 -m piptools compile \
  --resolver=backtracking --strip-extras --allow-unsafe --generate-hashes \
  --index-url https://pypi.org/simple \
  --constraint resources/siril_plugins/requirements-macos-arm64.lock \
  --output-file resources/siril_plugins/requirements.lock \
  resources/siril_plugins/requirements.txt
```

## Common Env

项目会自动读取 `resources/default.env`；若存在额外的 runtime env 文件，也只叠加产品处理参数。GUI 上的 Debug、处理模式和“处理参数…”任务快照优先；已停用功能的配置与凭据会被丢弃，不能进入子进程。

构建载荷用于避免安装包出现可直接搜索的明文 Key，但客户端仍包含恢复试用 Key 所需逻辑，只提高提取难度，不构成真正的凭据保密边界。正式长期发行仍应改为服务端代理或短期、可撤销的试用令牌。

| Variable | Purpose |
|---|---|
| `SEESTAR_NETWORK_MODE` | 默认 `0`；只有显式设为 `1` 才允许在线 Gaia SPCC/PCC 发起网络请求 |
| `SEESTAR_DEBUG_MODE` / `SEESTAR_INPUT_MODE` | GUI Debug 开关与自动冻结的内部路由；任务验签后可用 `stage1_prepared_resume`、`stage2_corrected_resume`、`result_linear_resume`，主界面不提供手动起点选择 |
| `SEESTAR_OUTPUT_FORMAT` | 最终导出格式：`all`（默认）或逗号分隔 `tif,png,fit` |
| `SEESTAR_STAGE10_MANAGED_OUTPUT_ENABLE` | 默认 `1`；为所请求的 PNG/TIFF 生成独立 sRGB/ICC 衍生文件，FITS 永不重写 |
| `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT` | 默认 `0`；设为 `1` 时仅导出 `result_review*`，适合人工复核而非正式交付 |
| `SEESTAR_WATCHDOG_IDLE_TIMEOUT_SEC` / `SEESTAR_EXPORT_TAIL_TIMEOUT_SEC` | GUI 无输出 watchdog；普通阶段默认 900 秒，确认本轮 PNG 产物后的收尾默认 120 秒；超时会记录最后命令、进程状态和产物 |
| `SEESTAR_STAGE7_*` | 阶段 6 去星重试、质量阈值和修复控制；输入固定保持线性，外部预拉伸配置已退役；变量名保留 `STAGE7` 以兼容旧配置 |
| `SEESTAR_STAGE9_*` | Stage 9 参考驱动星色修复、starmask Asinh、Screen 回混及保存前质量门控开关/阈值 |
| `SEESTAR_STAGE4_AUTO_GEOMETRY_*` / `SEESTAR_STAGE4_PLATESOLVE_ENABLE` | 无冲突且高置信的设备/FITS 几何可驱动 platesolve；显式环境覆盖优先，解算后 WCS 比例超限即回滚并禁止 SPCC/PCC |
| `SEESTAR_SPCC_ENABLE` / `SEESTAR_STAGE4_SPCC_*` / `SEESTAR_STAGE4_PCC_TIMEOUT_SEC` | 宽带与确认 Ha/OIII 映射的双窄带都先做单次 Gaia DR3 SPCC 物理校色；仅宽带 SPCC 异常时做单次 PCC；波长/带宽与超时受安全限幅 |
| `SEESTAR_STAGE4_NBN_*` / `SEESTAR_STAGE4_FILTER_HINT` | 双窄带 HOO 归一化只生成 `stage4_hoo_artistic.fit` 艺术派生图；后续主链恢复 `stage4_physical_color.fit`，单色和不确认映射保色 |
| `SEESTAR_STAGE8_LOCAL_ADJUSTMENT_ENGINE_ENABLE` | 默认 `1`；启用版本化本地曲线、软蒙版、形态学与局部调整配方 |
| `SEESTAR_ABERRATION_API_ENABLE` / `SEESTAR_ABERRATION_PROVIDER` | Stage 5/10 SASP Aberration API fallback；默认关闭 API，provider 可设 `cpu` |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` / `SEESTAR_SIRIL_PLUGIN_DIR` | 实验插件命令探测与插件目录 |

完整 pipeline env 说明见 `pipeline/seestar_Superimpose_workflow.md`；打包/runtime env 见 `INTEGRATION_README.md`。

Stage 4 在确认线性的宽带 RGB/OSC 和双窄带输入上保存不可变的 `stage4_pre_pcc.fit`，然后以独立 Siril CLI 进程执行一次 Gaia DR3 SPCC。有效本地 `xp_sampled` 分块存在时使用 `spcc -catalog=localgaia`，否则仅在开启联网时使用 `spcc -catalog=gaia`；宽带传入 Sony IMX585、Seestar LP/No filter 和白参考，双窄带传入 Ha/OIII 中心波长与带宽。Siril 的“解不精确”提示作为风险证据交给目标感知像素质量门，不再单独否决候选；SPCC 超时、命令失败或像素质量门拒绝时，流水线先回载不可变检查点：只有宽带可再执行一次 PCC 异常回退；双窄带禁止用宽带 PCC 替代物理校色。两种方法都失败时，宽带仅通过星点软掩膜做局部恢复并要求复核；双窄带保留物理输入并要求复核。

双窄带物理校色和艺术映射是两条明确分支：接受的 SPCC 物理结果保存为 `stage4_physical_color.fit`；HOO 归一化从它生成可选 `stage4_hoo_artistic.fit`，报告中固定 `feeds_main_pipeline=false`，随后重新载入物理母版再保存 `stage4_color.fit`。因此 HOO 艺术增益不会进入 Stage 5。Stage 4 一旦写出 `requires_review=true`，全局状态为 `review_required`，Stage 10 只写 `result_review*`，不会占用正式结果名。

完全离线的 platesolve/PCC 共用 `runtime_home/.local/share/siril/siril_cat_healpix8_astro.dat`；该官方 Gaia DR3 HEALPix 目录含恒星 `Teff`，可作为 PCC 的星色参考。离线 SPCC 使用独立的 `runtime_home/.local/share/siril/siril_cat1_healpix8_xpsamp/` 分块目录（完整集接近 21 GB），不与前者混用。GUI 下载器目前只安装较小的 astrometric/PCC 目录，使用 Zenodo 官方文件、双 SHA-256 校验和原子替换；构建脚本拒绝把任一 Gaia 星表带入 App。App 只打包固定版本并校验的 Sony IMX585、Seestar LP/No filter 与白参考 SPCC 元数据，不包含 Gaia 光谱星表。

## Limits

- 主要面向 Seestar 数据与 Siril 1.4+。
- 阶段 6 不再依赖 StarNet；去星链路仅保留 SyQon 与 SASP。全部不可用或结果被拒绝时保留含星复核路径，不会伪造 Starless 结果，最终只输出 `result_review*`。

## Tests

```bash
python -m pytest tests/ -v
```

重点覆盖 GUI runtime、pipeline fallback、输入过滤，以及已停用实验模块的隔离边界。
