# Seestar Superimpose

面向 Seestar 天文图像的自动后期处理工具。以 **Siril 1.4+** 为核心，结合 `SyQon-Starless`、SASP/CosmicClarity 等插件链路，把 `.fit/.fits` 输入自动导出为 TIFF/PNG/FITS。提供 macOS GUI 与离线打包。

## Key Files

| Path | Role |
|---|---|
| `pipeline/seestar_Superimpose.py` | stage 1-10 主流程编排 |
| `pipeline/stages/` | stage 1-10 分离实现模块 |
| `pipeline/stage11_ai_postprocess.py` | 可选 AI 后期 |
| `pipeline/seestar_Superimpose_workflow.md` | 详细流程说明 |
| `gui/seestar_gui_app.py` | GUI 入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 主窗口与 Siril worker |
| `build/build_macos_app.sh` | macOS 打包 |
| `resources/default.env` | 项目内默认 runtime env；无需手工 export |
| `resources/ai.env` | 本地/打包覆盖 env；可放 API key，默认被 git 忽略 |
| `INTEGRATION_README.md` | 内部集成/runtime/验证细节 |

## Pipeline Summary

主流程是 starless-first 后期链路：输入统一和裁切 -> 目标画像/策略 -> 背景提取、platesolve、校色、线性反卷积/轻降噪 -> 线性去星与 starmask 准备 -> Starless 主体拉伸和分区增强 -> 受控回星 -> 最终降噪和 TIFF/PNG/FITS 导出。

Stage 3 背景提取按候选质量门控自动选择：有 Siril scripts 缓存时优先通过 `pyscript GraXpert-AI.py` / `pyscript AutoBGE.py` 尝试 GraXpert、ADBE、DBE、AutoDBE；再尝试动态 `subsky -rbf` 和 `subsky 1` 兜底。GraXpert BGE 模型会从 `resources/siril_plugins/model_v2_0_1.onnx` 同步到运行时 GraXpert 模型目录。

Stage 11 是可选 AI 后期，只生成 `*_ai` 副本，不覆盖 Stage 10 原始产物。Stage 状态为 `ok / degraded / failed / skipped`；summary 可显示 `ok_with_fallback`、`ok_skipped_optional`。

详细阶段顺序、检查点、质量门控和降级路径见 `pipeline/seestar_Superimpose_workflow.md`。

## GUI Usage

1. 启动 `SeestarSuperimpose.app`。
2. 选择包含 `.fit/.fits` 的工作目录。
3. 选择模式：
   - `Normal Pipeline`：完整流程。
   - `Postprocess From stage2_corrected.fit`：要求工作目录根下或 `process/` 下存在 `stage2_corrected.fit`，作为叠加后中间结果从 stage 3 继续完整后处理。
   - `Continue From result_linear.fit`：要求工作目录根下存在 `result_linear.fit`，从 stage 6 继续。
4. 按需切换：
   - `AI: ON/OFF` -> `SEESTAR_AI_ENABLED`
   - `Debug: ON/OFF` -> `SEESTAR_DEBUG_MODE`
   - `联网: ON/OFF` -> Siril 是否加 `--offline`
5. 点击开始；preflight 会检查 Siril、配置、pipeline、输入和磁盘空间。

## Outputs

- `*.tif`：高质量结果
- `*.png`：预览
- `*_final.fit`：归档 FITS
- `result_linear.fit`：stage 5 线性中间结果，可用于续跑
- `*_linear.*`：`result_linear.fit` 续跑模式产物
- `process/stage*.fit`、`process/*.json`：Debug 模式中间产物与诊断报告
- `process/*_quality_metrics.json`、`process/stage_quality_metrics.jsonl`：Debug 模式下每个阶段输出的统一质量指标，日志同步输出 `[STAGE_QUALITY_METRICS] schema=seestar.stage_quality.v1 ...`
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
- `resources/ai.env`
- `resources/siril_plugins/`（可用 `bash resources/siril_plugins/download_siril_plugins.sh` 准备）

Default output: `release/SeestarSuperimpose.app`

## Common Env

项目会自动读取 `resources/default.env`；若存在 `resources/ai.env`、runtime home 或工作目录下的 `.seestar_ai.env`，也会叠加读取。优先级为：进程环境 > 工作目录 `.seestar_ai.env` > runtime home `.seestar_ai.env` > `resources/ai.env` > `resources/default.env`。GUI 上的 Debug、处理模式和 AI 开关仍以界面选择为准。

| Variable | Purpose |
|---|---|
| `SEESTAR_AI_*` | AI 总配置；开启后用于阶段 6 去星、阶段 8 增强诊断/参数建议和阶段 11 AI 副本；阶段 7 拉伸当前固定本地两候选 |
| `SEESTAR_DEBUG_MODE` / `SEESTAR_INPUT_MODE` | GUI Debug 开关与处理模式；`stage2_corrected_resume` 从 `stage2_corrected.fit` 进入 stage 3，`result_linear_resume` 从 `result_linear.fit` 续跑 |
| `SEESTAR_OUTPUT_FORMAT` | 最终导出格式：`all`（默认）或逗号分隔 `tif,png,fit` |
| `SEESTAR_STAR_SEPARATION_MODE` / `SEESTAR_STAGE7_*` | 阶段 6 去星输入、重试、质量阈值和修复控制；变量名保留 `STAGE7` 以兼容旧配置 |
| `SEESTAR_STAGE9_STARMASK_*` | Stage 9 像素回混前的 starmask Asinh 拉伸控制 |
| `SEESTAR_SPCC_ENABLE` / `SEESTAR_STAGE4_PLATESOLVE_ENABLE` / `SEESTAR_STAGE4_SPCC_*` | Stage 4 色彩校准、platesolve、sensor/filter 和 target-aware white reference 行为 |
| `SEESTAR_ABERRATION_API_ENABLE` / `SEESTAR_ABERRATION_PROVIDER` | Stage 5/10 SASP Aberration API fallback；默认关闭 API，provider 可设 `cpu` |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` / `SEESTAR_SIRIL_PLUGIN_DIR` | 实验插件命令探测与插件目录 |

完整 pipeline env 说明见 `pipeline/seestar_Superimpose_workflow.md`；打包/runtime env 见 `INTEGRATION_README.md`。

## Limits

- 主要面向 Seestar 数据与 Siril 1.4+。
- 阶段 6 不再依赖 StarNet；去星链路仅保留 SyQon 与 SASP，全部不可用时会退化为继续使用当前 Stage 6 输入。

## Tests

```bash
python -m pytest tests/ -v
```

重点覆盖 GUI runtime、pipeline fallback、输入过滤、Stage11 单元与真实数据集成。
