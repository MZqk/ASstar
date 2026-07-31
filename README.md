# Seestar Superimpose

面向 Seestar 天文图像的自动后期处理工具。以 **Siril 1.4+** 为核心，结合 `SyQon-Starless`、SASP/CosmicClarity 等插件链路，把 `.fit/.fits` 输入自动导出为 TIFF/PNG/FITS。提供 macOS GUI 与离线打包。

系统要求：**macOS 14.0 或更高版本**，仅支持 **Apple Silicon（arm64）**，不支持 Intel Mac。

## Key Files

| Path | Role |
|---|---|
| `pipeline/seestar_Superimpose.py` | stage 1-10 主流程编排 |
| `pipeline/stages/` | stage 1-10 分离实现模块 |
| `pipeline/stage11_ai_postprocess.py` | 可选 AI 后期 |
| `pipeline/ai_artistic_derivative.py` | Stage 11 后完全隔离的 AI 艺术衍生实验输出 |
| `pipeline/seestar_Superimpose_workflow.md` | 详细流程说明 |
| `gui/seestar_gui_app.py` | GUI 入口 |
| `gui/seestar_gui_dev.py` | 使用系统 Siril/现有 seed 的免打包开发入口 |
| `gui/seestar_pipeline_dev.py` | 不启动 GUI 的源码核心流水线验证入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 主窗口与 Siril worker |
| `build/build_macos_app.sh` | macOS 打包 |
| `requirements.txt` / `requirements.lock` | 直接依赖约束 / pip-tools 生成的完整依赖锁 |
| `resources/default.env` | 项目内默认 runtime env；无需手工 export |
| `resources/ai.env` | 开发者试用模型的本地构建输入；须为 `0600`，Key 不再明文复制进 App |
| `INTEGRATION_README.md` | 内部集成/runtime/验证细节 |

## Pipeline Summary

主流程是 starless-first 后期链路：输入统一和裁切 -> 输入线性状态判定 -> 冻结主目标/策略与处理计划 -> 条件背景提取、自动设备几何/platesolve、宽带 PCC 或双窄带归一化、噪声模型驱动线性反卷积/降噪 -> 线性去星与确定性星色修复 -> Starless 主体拉伸和本地曲线/蒙版增强 -> 受控回星 -> 最终降噪和科学/展示/编辑版导出。输入状态只接受可追溯证据，分为 `linear / nonlinear / unknown`；非线性或未知输入只执行安全裁切并进入复核导出，不运行破坏性的线性处理、去星或 Starless 增强。

Stage 3 先判断背景是否确实需要处理：背景已干净时原样跳过；弥散目标或证据含糊时默认保留原图并要求复核；只有确定性高梯度才运行候选链。候选从同一不可变基线开始，失败、拒绝或切换候选前都会回滚，避免前一个候选的修改泄漏到后一个候选。有 Siril scripts 缓存时优先通过 `pyscript GraXpert-AI.py` / `pyscript AutoBGE.py` 尝试 GraXpert、ADBE、DBE、AutoDBE，再尝试动态 `subsky -rbf` 和 `subsky 1` 兜底。

Stage 5 不会在任务中联网下载 GraXpert 对象反卷积模型。自动模式优先使用 Seestar App 随包模型；随包模型缺失时读取本机 GraXpert 应用已安装的最新版模型，再尝试“处理参数…”里指定的官方 `model.onnx`、语义版本目录（如 `1.0.1`）或 `deconvolution-object-ai-models` 目录。App 会把有效模型只读链接到隔离运行目录；全部无效或缺失时安全回退 Siril RL。“自动加速”允许 GraXpert 通过 ONNX Runtime 自动选择 CoreML 等可用执行设备，初始化或推理失败时由插件回退 CPU；“CPU 兼容模式”会显式禁用该硬件加速。GUI 分开显示“反卷积”和“降噪”：若 GraXpert 已完成而降噪因低噪声或配置关闭被跳过，反卷积结果仍保存为 `stage5_linear.fit` 并继续传给 Stage 6。

Stage 6 会显式记录去星结果为 `accepted / target_bypass / rejected / tool_failed`。拒绝或工具失败时保留含星复核路径，不再把 passthrough 文件命名或解释为 Starless；Stage 8/9 跳过 Starless 专用逻辑，Stage 10 只生成复核产物。

亮发射/反射星云的 Stage 8 halo 门分为三级：不超过 `0.350` 正常增强，`0.350–0.600` 生成饱和度受限、无锐化的 masked 候选并由质量门决定，超过 `0.600` 或命中硬失败则安全旁路。受限候选保留为 `stage8_limited_candidate.fit`；拒绝时自动回滚。Stage 9 会把“使用 Stage 8 安全旁路源”与“Stage 9 自己使用降低强度/compact recovery 回退”分开显示。

Stage 11 的实现代码与测试保留给第二阶段，但第一阶段 macOS 产品入口隐藏，GUI worker 通过 release gate 硬禁用，旧设置、内部参数和 Keychain 凭据都不能把它打开；第一阶段正式交付以 Stage 10 为终点。Stage 状态为 `ok / degraded / failed / skipped`；最终运行状态为 `success / partial_success / review_required / failed`。阶段状态由结构化执行、回退和旁路字段决定，不再根据日志说明文字中的 `skipped/fallback` 关键词猜测。

详细阶段顺序、检查点、质量门控和降级路径见 `pipeline/seestar_Superimpose_workflow.md`。

## GUI Usage

1. 启动 `SeestarSuperimpose.app`。首次启动或没有有效目录时会显示拖放空状态；已保存的有效目录会直接进入任务设置。
2. 拖入或选择包含 `.fit/.fits` 的工作目录。应用会统计输入，并自动推荐“完整处理”“从裁切后继续”或“从线性处理后继续”，同时异步显示 Stage 0 输入预览。
3. 保持“自动推荐”，或手动指定处理方式；展开高级设置后，“处理参数…”会在任务卡片下方展开为单页 sheet，同页配置输出格式、正式/复核输出、Gaia PCC、拍摄滤镜、线性降噪、反卷积、GraXpert 对象模型和 CPU 兼容模式。“色彩校准”内可按需下载 Siril 官方离线 Gaia DR3 星色目录：压缩包约 1.1 GB、解压后约 1.52 GB，只写入应用 runtime home，不写项目、不随 App 打包。下方“专业细节”进一步开放 PCC 超时、恒星白平衡增益、线性降噪强度、GraXpert 强度、RL 迭代/PSF 星数、去星重试、残星/星晕/彩噪修复、星点 Asinh 拉伸和弱星恢复门槛；所有数值均受安全范围限制。参数修改后自动保存，每项都可通过鼠标悬浮查看默认值、影响和回退路径。AI 后期能力仍保留在 pipeline 中，但第一阶段 macOS UI 隐藏入口，GUI worker 也会硬禁用 Stage 11。
4. 点击开始；preflight 会分别检查运行环境所在卷与工作目录所在卷的磁盘空间。工作目录需求按处理方式、基准 FITS 大小和各阶段临时/调试产物数量估算，不使用固定容量门槛。
   运行环境准备在后台执行，可用“停止”取消；依赖锁、Siril Python ABI 与 App
   版本未变化时会复用已就绪 runtime，不重复执行 `pip install`。
   SyQon 与 CosmicClarity 模型直接从 App 或离线资源包只读使用，不再复制到
   `runtime_home`。
5. 快速 preflight 通过后，界面立即切换为只读运行视图：左侧显示冻结后的本次配置，中央只显示最新一张可靠预览，右侧完整列出 Stage 1-10。每个成功或降级阶段完成后刷新中央预览；跳过、失败或预览生成失败时保留上一张。当前运行阶段与预览来源阶段分开标注。
   Stage 0 和 Stage 1-6 预览按原始/线性像素显示，Stage 7-11 也不附加显示拉伸、Gamma 或直方图归一化。中央预览支持适合窗口、1:1、滚轮缩放与拖动。
   阶段列表显示等待、运行中、已完成、已降级、失败或已跳过，并显示本阶段耗时和总耗时；不提供百分比或预计剩余时间。
   `CompletedWithWarning` 会显示黄色复核卡片，可直接打开最终质量报告。
6. 完成、失败或停止后仍停留在运行视图，可返回任务设置、按上次实际配置重新运行或打开结果目录；失败会自动展开日志。运行输出收纳在可折叠的“详细日志”区域，Siril/插件逐百分比进度会压缩为约 10% 一档，重复的 ICCProfile 扩展提示也会折叠；错误、阶段切换和完成信息仍完整保留。完整说明位于“帮助”菜单。最近目录、处理方式、高级设置、日志展开状态和窗口位置/尺寸会自动保存。

## Outputs

- `*.tif`：高质量结果
- `*.png`：预览；Stage 7 已验收的非线性结果不再二次拉伸，诊断回退源才使用 linked autostretch
- `*_final.fit`：归档 FITS
- `*_display_srgb.png`：独立 16-bit 展示版，写入 `sRGB/gAMA/cHRM`；不改动 Siril PNG 或 FITS
- `*_edit_srgb.tif`：独立 16-bit 可编辑版，内嵌 sRGB ICC；找不到有效 ICC 时不生成未标记替代品
- `process/managed_output_report.json` / `output_color_manifest.json`：记录受管理导出、容器元数据和科学 FITS 前后 SHA-256 不变性
- FITS 头完整时输出名包含目标、叠加数、曝光和拍摄时间；仅缺少 `STACKCNT` 等次要字段时，仍使用已有目标与拍摄时间生成可识别名称，避免退回通用名称覆盖其他任务
- `result_linear.fit`：stage 5 线性中间结果，可用于续跑
- `*_linear.*`：`result_linear.fit` 续跑模式产物
- `result_review*.tif/png/fit`：最终质量门要求保守重跑，或显式设置 `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT=1` 时生成的复核产物；此时不会写普通 `result_processed/result_final` 名称，并会跳过耗时的 Stage 10 最终降噪链
- `process/stage*.fit`、`process/*.json`：开启“保留中间文件”后的阶段产物与诊断报告
- `process/final_quality_report.json`：Stage 10 最终质量报告；降级完成时可从 GUI 警告卡片打开
- `processing-plan.json`：变换前原子冻结的本轮计划，包含输入 SHA-256、输入状态、冻结主目标、通道语义、阶段动作、候选合约和规范化计划哈希
- `pipeline-result.json`：原子写出的最终运行清单，包含计划哈希、阶段结果、产物 SHA-256 和全局状态；`process/` 中保留镜像副本
- `process/review_bundles/stage9_star_remixing/`：Stage 8 底图与 Stage 9 选中/回滚结果的 before/after/difference 复核包，`review.json` 同时记录星表、starmask 标定和候选拒绝原因
- `process/*_quality_metrics.json`、`process/stage_quality_metrics.jsonl`：保留中间文件时输出的统一阶段质量指标，日志同步输出 `[STAGE_QUALITY_METRICS] schema=seestar.stage_quality.v1 ...`
- `result_processed_ai.tif/png`、`result_final_ai.fit`：第二阶段 Stage 11 启用后才可能出现；第一阶段 GUI 不生成

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
- `packages/siril-1.4.3-arm64.dmg`

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

项目会自动读取 `resources/default.env`；若存在 `resources/ai.env`、runtime home 或工作目录下的 `.seestar_ai.env`，也会叠加读取非敏感选项。GUI worker 不再从这些文件或父进程环境读取 API Key。GUI 上的 Debug、处理模式和“处理参数…”任务快照优先；第一阶段 macOS UI 隐藏 AI 后期入口，worker 无论旧设置或内部调用参数如何都会强制写入 `SEESTAR_AI_ENABLED=0`，并且不会向子进程注入 Keychain AI 凭据。

构建载荷用于避免安装包出现可直接搜索的明文 Key，但客户端仍包含恢复试用 Key 所需逻辑，只提高提取难度，不构成真正的凭据保密边界。正式长期发行仍应改为服务端代理或短期、可撤销的试用令牌。

| Variable | Purpose |
|---|---|
| `SEESTAR_NETWORK_MODE` | 默认 `0`；只有显式设为 `1` 才允许在线 Gaia PCC、AI 顾问、Stage 11 和艺术衍生分支发起网络请求 |
| `SEESTAR_AI_*` | AI 总配置；模型只返回代码定义候选的 ID，不接受模型注入数值参数。Stage 7 只允许在本地硬门已通过的非救援候选中选择；未知/不可用 ID 回退确定性本地择优 |
| `SEESTAR_DEBUG_MODE` / `SEESTAR_INPUT_MODE` | GUI Debug 开关与处理模式；`stage2_corrected_resume` 从 `stage2_corrected.fit` 进入 stage 3，`result_linear_resume` 从 `result_linear.fit` 续跑 |
| `SEESTAR_OUTPUT_FORMAT` | 最终导出格式：`all`（默认）或逗号分隔 `tif,png,fit` |
| `SEESTAR_STAGE10_MANAGED_OUTPUT_ENABLE` | 默认 `1`；为所请求的 PNG/TIFF 生成独立 sRGB/ICC 衍生文件，FITS 永不重写 |
| `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT` | 默认 `0`；设为 `1` 时仅导出 `result_review*`，并跳过 Stage 11，适合人工复核而非正式交付 |
| `SEESTAR_WATCHDOG_IDLE_TIMEOUT_SEC` / `SEESTAR_EXPORT_TAIL_TIMEOUT_SEC` | GUI 无输出 watchdog；普通阶段默认 900 秒，确认本轮 PNG 产物后的收尾默认 120 秒；超时会记录最后命令、进程状态和产物 |
| `SEESTAR_STAGE7_*` | 阶段 6 去星重试、质量阈值和修复控制；输入固定保持线性，外部预拉伸配置已退役；变量名保留 `STAGE7` 以兼容旧配置 |
| `SEESTAR_STAGE9_*` | Stage 9 参考驱动星色修复、starmask Asinh、Screen 回混及保存前质量门控开关/阈值 |
| `SEESTAR_STAGE4_AUTO_GEOMETRY_*` / `SEESTAR_STAGE4_PLATESOLVE_ENABLE` | 无冲突且高置信的设备/FITS 几何可驱动 platesolve；显式环境覆盖优先，解算后 WCS 比例超限即回滚并禁止 PCC |
| `SEESTAR_STAGE4_NBN_*` / `SEESTAR_STAGE4_PCC_TIMEOUT_SEC` / `SEESTAR_STAGE4_FILTER_HINT` | 宽带 RGB/OSC 单次 Gaia PCC；明确 Ha/OIII 映射的双窄带走 HOO 归一化；单色和不确认映射保色 |
| `SEESTAR_STAGE8_LOCAL_ADJUSTMENT_ENGINE_ENABLE` | 默认 `1`；启用版本化本地曲线、软蒙版、形态学与局部调整配方 |
| `SEESTAR_ABERRATION_API_ENABLE` / `SEESTAR_ABERRATION_PROVIDER` | Stage 5/10 SASP Aberration API fallback；默认关闭 API，provider 可设 `cpu` |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` / `SEESTAR_SIRIL_PLUGIN_DIR` | 实验插件命令探测与插件目录 |

完整 pipeline env 说明见 `pipeline/seestar_Superimpose_workflow.md`；打包/runtime env 见 `INTEGRATION_README.md`。

Stage 4 在确认线性的宽带 RGB/OSC 输入上保存不可变的 `stage4_pre_pcc.fit`，然后以独立 Siril CLI 进程执行一次 Gaia PCC：本地 Gaia DR3 目录有效时使用 `pcc -catalog=localgaia`，否则仅在开启联网时使用 `pcc -catalog=gaia`。候选除原有有限值、通道增益、高光和动态范围门外，还检查恒星综合色温分布、背景归一化色差和主体颜色漂移。超时、命令失败或质量门拒绝时，流水线必先回载该检查点，再仅通过星点软掩膜做局部颜色恢复；全图白平衡始终禁止，宽带回退结果会标记为需要复核。双窄带明确跳过 PCC；只有 Ha/OIII 通道映射达到置信门时，才从 `stage4_pre_nbn.fit` 生成亮度保持的 HOO 归一化候选，并检查背景改善、Ha/OIII 比例、星色、亮度和裁剪，拒绝时保留原色。单色不执行颜色处理。

完全离线的 platesolve/PCC 共用 `runtime_home/.local/share/siril/siril_cat_healpix8_astro.dat`；该官方 Gaia DR3 HEALPix 目录含恒星 `Teff`，可作为 PCC 的星色参考。GUI 下载器使用 Zenodo 官方文件、校验压缩和解压后 SHA-256，并以临时文件完成后原子替换；下载失败或取消不会发布半成品。代码只有在文件有效时才调用 `localgaia`，且构建脚本会拒绝把 Gaia 目录带入 App。

## Limits

- 主要面向 Seestar 数据与 Siril 1.4+。
- 阶段 6 不再依赖 StarNet；去星链路仅保留 SyQon 与 SASP。全部不可用或结果被拒绝时保留含星复核路径，不会伪造 Starless 结果，最终只输出 `result_review*`。

## Tests

```bash
python -m pytest tests/ -v
```

重点覆盖 GUI runtime、pipeline fallback、输入过滤、Stage11 单元与真实数据集成。
