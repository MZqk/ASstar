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
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 主窗口与 Siril worker |
| `build/build_macos_app.sh` | macOS 打包 |
| `requirements.txt` / `requirements.lock` | 直接依赖约束 / pip-tools 生成的完整依赖锁 |
| `resources/default.env` | 项目内默认 runtime env；无需手工 export |
| `resources/ai.env` | 开发者试用模型的本地构建输入；须为 `0600`，Key 不再明文复制进 App |
| `INTEGRATION_README.md` | 内部集成/runtime/验证细节 |

## Pipeline Summary

主流程是 starless-first 后期链路：输入统一和裁切 -> 目标画像/策略 -> 背景提取、platesolve、校色、线性反卷积/轻降噪 -> 线性去星与 starmask 准备 -> Starless 主体拉伸和分区增强 -> 受控回星 -> 最终降噪和 TIFF/PNG/FITS 导出。

Stage 3 背景提取按候选质量门控自动选择：有 Siril scripts 缓存时优先通过 `pyscript GraXpert-AI.py` / `pyscript AutoBGE.py` 尝试 GraXpert、ADBE、DBE、AutoDBE；再尝试动态 `subsky -rbf` 和 `subsky 1` 兜底。GraXpert BGE 模型会从 `resources/siril_plugins/model_v2_0_1.onnx` 同步到运行时 GraXpert 模型目录。

Stage 5 不会在任务中联网下载 GraXpert 对象反卷积模型。自动模式优先使用 Seestar App 随包模型；随包模型缺失时读取本机 GraXpert 应用已安装的最新版模型，再尝试“处理参数…”里指定的官方 `model.onnx`、语义版本目录（如 `1.0.1`）或 `deconvolution-object-ai-models` 目录。App 会把有效模型只读链接到隔离运行目录；全部无效或缺失时安全回退 Siril RL。

Stage 11 是可选 AI 后期，只生成 `*_ai` 副本，不覆盖 Stage 10 原始产物。Stage 状态为 `ok / degraded / failed / skipped`；summary 可显示 `ok_with_fallback`、`ok_skipped_optional`。

详细阶段顺序、检查点、质量门控和降级路径见 `pipeline/seestar_Superimpose_workflow.md`。

## GUI Usage

1. 启动 `SeestarSuperimpose.app`。首次启动或没有有效目录时会显示拖放空状态；已保存的有效目录会直接进入任务设置。
2. 拖入或选择包含 `.fit/.fits` 的工作目录。应用会统计输入，并自动推荐“完整处理”“从裁切后继续”或“从线性处理后继续”，同时异步显示 Stage 0 输入预览。
3. 保持“自动推荐”，或手动指定处理方式；展开高级设置后，“处理参数…”会在任务卡片下方展开为单页 sheet，同页配置输出格式、正式/复核输出、单次在线 Gaia PCC、拍摄滤镜、线性降噪、反卷积、GraXpert 对象模型和 CPU 兼容模式。下方“专业细节”进一步开放 PCC 超时、恒星白平衡增益、线性降噪强度、GraXpert 强度、RL 迭代/PSF 星数、去星重试、残星/星晕/彩噪修复、星点 Asinh 拉伸和弱星恢复门槛；所有数值均受安全范围限制。参数修改后自动保存，每项都可通过鼠标悬浮查看默认值、影响和回退路径。AI 后期能力仍保留在 pipeline 中，但当前 macOS UI 暂不开放。
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
- FITS 头完整时输出名包含目标、叠加数、曝光和拍摄时间；仅缺少 `STACKCNT` 等次要字段时，仍使用已有目标与拍摄时间生成可识别名称，避免退回通用名称覆盖其他任务
- `result_linear.fit`：stage 5 线性中间结果，可用于续跑
- `*_linear.*`：`result_linear.fit` 续跑模式产物
- `result_review*.tif/png/fit`：最终质量门要求保守重跑，或显式设置 `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT=1` 时生成的复核产物；此时不会写普通 `result_processed/result_final` 名称，并会跳过耗时的 Stage 10 最终降噪链
- `process/stage*.fit`、`process/*.json`：开启“保留中间文件”后的阶段产物与诊断报告
- `process/final_quality_report.json`：Stage 10 最终质量报告；降级完成时可从 GUI 警告卡片打开
- `process/review_bundles/stage9_star_remixing/`：Stage 8 底图与 Stage 9 选中/回滚结果的 before/after/difference 复核包，`review.json` 同时记录星表、starmask 标定和候选拒绝原因
- `process/*_quality_metrics.json`、`process/stage_quality_metrics.jsonl`：保留中间文件时输出的统一阶段质量指标，日志同步输出 `[STAGE_QUALITY_METRICS] schema=seestar.stage_quality.v1 ...`
- `result_processed_ai.tif/png`、`result_final_ai.fit`：可选 AI 后期产物

完整中间文件清单见 `pipeline/seestar_Superimpose_workflow.md` 的“文件与目录行为”。

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

项目会自动读取 `resources/default.env`；若存在 `resources/ai.env`、runtime home 或工作目录下的 `.seestar_ai.env`，也会叠加读取非敏感选项。GUI worker 不再从这些文件或父进程环境读取 API Key。GUI 上的 Debug、处理模式和“处理参数…”任务快照优先；当前 macOS UI 强制关闭未开放的 AI 后期入口。

构建载荷用于避免安装包出现可直接搜索的明文 Key，但客户端仍包含恢复试用 Key 所需逻辑，只提高提取难度，不构成真正的凭据保密边界。正式长期发行仍应改为服务端代理或短期、可撤销的试用令牌。

| Variable | Purpose |
|---|---|
| `SEESTAR_AI_*` | AI 总配置；开启后用于阶段 6 去星、阶段 8 增强诊断/参数建议和阶段 11 AI 副本；阶段 7 固定本地两个主候选，仅色度门控失败时追加三档确定性救援并统一质量择优 |
| `SEESTAR_DEBUG_MODE` / `SEESTAR_INPUT_MODE` | GUI Debug 开关与处理模式；`stage2_corrected_resume` 从 `stage2_corrected.fit` 进入 stage 3，`result_linear_resume` 从 `result_linear.fit` 续跑 |
| `SEESTAR_OUTPUT_FORMAT` | 最终导出格式：`all`（默认）或逗号分隔 `tif,png,fit` |
| `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT` | 默认 `0`；设为 `1` 时仅导出 `result_review*`，并跳过 Stage 11，适合人工复核而非正式交付 |
| `SEESTAR_WATCHDOG_IDLE_TIMEOUT_SEC` / `SEESTAR_EXPORT_TAIL_TIMEOUT_SEC` | GUI 无输出 watchdog；普通阶段默认 900 秒，确认本轮 PNG 产物后的收尾默认 120 秒；超时会记录最后命令、进程状态和产物 |
| `SEESTAR_STAGE7_*` | 阶段 6 去星重试、质量阈值和修复控制；输入固定保持线性，外部预拉伸配置已退役；变量名保留 `STAGE7` 以兼容旧配置 |
| `SEESTAR_STAGE9_*` | Stage 9 starmask Asinh、Screen 回混及保存前质量门控开关/阈值 |
| `SEESTAR_STAGE4_PLATESOLVE_ENABLE` / `SEESTAR_STAGE4_PCC_TIMEOUT_SEC` / `SEESTAR_STAGE4_FILTER_HINT` | Stage 4 色彩校准；仅在线性宽带 RGB/OSC 上单次调用 Gaia PCC（默认 30 秒），窄带/单色/非线性输入按策略跳过 |
| `SEESTAR_ABERRATION_API_ENABLE` / `SEESTAR_ABERRATION_PROVIDER` | Stage 5/10 SASP Aberration API fallback；默认关闭 API，provider 可设 `cpu` |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` / `SEESTAR_SIRIL_PLUGIN_DIR` | 实验插件命令探测与插件目录 |

完整 pipeline env 说明见 `pipeline/seestar_Superimpose_workflow.md`；打包/runtime env 见 `INTEGRATION_README.md`。

Stage 4 在确认线性的宽带 RGB/OSC 输入上保存不可变的 `stage4_pre_pcc.fit`，然后以独立 Siril CLI 进程执行一次 `pcc -catalog=gaia`。超时、命令失败或目标感知质量门拒绝时，流水线必先回载该检查点，再仅通过星点软掩膜做局部颜色恢复；全图白平衡始终禁止，宽带回退结果会标记为需要复核。窄带、HOO/SHO、双窄带和单色输入不会调用 PCC。

完全离线的 platesolve 可由 Siril Catalog Installer 在 runtime 目录安装 `siril_cat_healpix8_astro.dat`。代码只有在该文件有效时才调用 `-catalog=localgaia`；PCC 本身固定使用在线 Gaia，因此关闭联网时按策略跳过并进入安全回退，不尝试其他在线目录。

## Limits

- 主要面向 Seestar 数据与 Siril 1.4+。
- 阶段 6 不再依赖 StarNet；去星链路仅保留 SyQon 与 SASP，全部不可用时会退化为继续使用当前 Stage 6 输入。

## Tests

```bash
python -m pytest tests/ -v
```

重点覆盖 GUI runtime、pipeline fallback、输入过滤、Stage11 单元与真实数据集成。
