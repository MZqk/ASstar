# Seestar Superimpose

面向 Seestar 天文图像的自动后期处理工具。以 **Siril 1.4+** 为核心，结合 `SyQon-Starless`、SASP/CosmicClarity 等插件链路，把 `.fit/.fits` 输入自动导出为 TIFF/PNG/FITS。提供 macOS GUI 与离线打包。

## Key Files

| Path | Role |
|---|---|
| `pipeline/seestar_Superimpose.py` | stage 1-10 主流程 |
| `pipeline/stage11_ai_postprocess.py` | 可选 AI 后期 |
| `pipeline/seestar_Superimpose_workflow.md` | 详细流程说明 |
| `gui/seestar_gui_app.py` | GUI 入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 主窗口与 Siril worker |
| `build/build_macos_app.sh` | macOS 打包 |
| `INTEGRATION_README.md` | 内部集成/runtime/验证细节 |

## Pipeline Summary

1. 输入检测；必要时处理 `Light_` 单帧；过滤 `sasp_*`、`*starless*`、`*starmask*` 等中间产物。
2. 裁切；阶段 2 会迭代处理明显黑边，后续去星阶段只诊断黑边风险，不再临时裁切。
3. Stage 2.5 目标画像识别：从当前 FITS 像素特征和目标库推断 target type，生成 `target_profile.json` 与 `pipeline_policy.json`；`result_linear.fit` 续跑也会执行此步骤，失败时回退 `generic_low_snr_safe`。
4. 背景提取：默认内置 `subsky` + 质量门控，并读取 target policy 输出 `background_quality_report.json`；实验 plugin command 仅在 `SEESTAR_WORKFLOW_PLUGIN_PROBE=1` 时探测。
5. 解析与校色：`SPCC -> PCC -> CCM`；输出 `color_calibration_report.json`，低置信校色会限制后续颜色/饱和增强。
6. 矫正、锐化、线性降噪；插件失败时回退到代码化/内置处理，并输出 `stage5_linear_report.json`。
7. 拉伸与曲线：Stage 6 会根据 target policy 生成多个候选并评分，输出 `stretch_candidates_report.json`；M42/亮核心星云会优先尝试 HDR masked stretch，并使用亮核心星云专用色噪门控，黑场、背景过低、核心过曝或动态范围异常的候选不能成为正常最终候选。
8. Stage 6.5 去星前质量门控：输出 `pre_starless_gate_report.json`，必要时生成 `stage7_ultra_conservative_asinh.fit`；若判定 `ready_for_starless=false`，Stage 7 默认跳过去星并进入 review export。
9. 去星与星点层：`SyQon-Starless -> SASP fallback -> degraded continuation`；Stage 7 会优先使用 Stage 6.5 推荐输入，质量差时会追加更低强度 Asinh 输入、择优去星，并对选中 starless 做残星、亮星 halo、暗坑和背景彩噪的保守像素修复。
10. Starless 深加工，支持 `sasp_starless.fit` 外部回写；默认生成核心、星云、外围云气和背景 soft mask 做分区增强，核心/背景强保护，SASP WaveScale 输出也会经 mask 回混；若 Stage 7 修复过、残星/halo/噪声不安全或 Stage 8 mask 覆盖不达标，会进入 conservative skip，禁止强锐化、强局部对比和强饱和增强。
11. 星点处理与合成，支持 `sasp_starmask.fit` 外部回写；若 Stage7 残星/halo 超标但仍可用，会绕过不可控 StarComposer 并降低像素级星点回混强度；若 Stage7 最终 starless 仍为 poor，则直接输出 Stage 6 有星安全版本，避免坏 starless/starmask 污染最终图。
12. 最终降噪与导出；classic CosmicClarity、Native、SCUNet、Siril alias、Aberration API 按可用性回退。
13. 可选 AI 后期：只生成 `*_ai` 副本，失败不影响原始产物。

Stage 状态：`ok / degraded / failed / skipped`；summary 可显示 `ok_with_fallback`、`ok_skipped_optional`。

## GUI Usage

1. 启动 `SeestarSuperimpose.app`。
2. 选择包含 `.fit/.fits` 的工作目录。
3. 选择模式：
   - `Normal Pipeline`：完整流程。
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
- `process/stage*.fit`：Debug 模式中间产物
- `process/target_profile.json`、`process/pipeline_policy.json`：Stage 2.5 目标类型、风险和处理策略
- `process/background_quality_report.json`、`process/color_calibration_report.json`、`process/stage5_linear_report.json`：Stage 3-5 policy-aware 诊断
- `process/stretch_candidates_report.json`：Stage 6 候选拉伸评分和最终选择；M42/脏背景场景下 autostretch 或黑场候选应显示 `allowed_as_final=false`，HDR masked 若仅比例色噪略高但绝对色噪低可作为正常候选
- `process/pre_starless_gate_report.json`：Stage 6.5 去星前质量门控和推荐 Starless 输入；`ready_for_starless=false` 时 Stage 7 默认跳过去星工具
- `process/stage7_quality.json`：Stage 7 去星参数、保守重试、starless 像素修复、Stage 8 conservative 标记与阶段 9 回混联动记录
- `process/stage11_quality.json`：Stage11 AI 后期参数、动态混合强度和质量门控记录（仅 AI 后期运行时）
- `process/ai_raw_*.json/txt`：AI 返回原文与响应结构；用于排查 JSON 解析失败和本阶段熔断原因
- `result_processed_ai.tif/png`、`result_final_ai.fit`：可选 AI 后期产物

## Build

```bash
cd /Users/mz/dev/aiseestart
./build/build_macos_app.sh
```

Required packages:
- `packages/python-3.13.12-macos11.pkg`
- `packages/siril-1.4.2-arm64.dmg`

Optional bundled resources:
- `resources/ai.env`
- `resources/siril_plugins/`（可用 `bash resources/siril_plugins/download_siril_plugins.sh` 准备）

Default output: `release/SeestarSuperimpose.app`

## Common Env

| Variable | Purpose |
|---|---|
| `SEESTAR_AI_*` | AI 总配置；开启后用于阶段 6-8 参数优化/诊断和阶段 11 AI 副本 |
| `SEESTAR_AI_STAGE6_ENABLE` / `SEESTAR_AI_STAGE7_ENABLE` / `SEESTAR_AI_STAGE8_ENABLE` | 可单独关闭阶段 6-8 AI 介入 |
| `SEESTAR_STAGE7_QUALITY_RETRY_MAX` | 阶段 7 残星/蒙版质量偏差时的 SyQon 参数重试预算；残星仍超标会联动降低阶段 9 回混强度 |
| `SEESTAR_STAGE7_SKIP_UNREADY_STARLESS` | Stage 6.5 判定不适合去星时是否跳过 SyQon/SASP；默认开启，设为 `0` 可强制继续 |
| `SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH` | 阶段 7 质量差时追加的更轻去星输入强度，默认 `1.35` |
| `SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX` | M42/亮核心星云的 Stage 7 halo 验收上限，默认 `0.60` |
| `SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH` / `SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH` / `SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH` | Stage 7 starless 残星、halo 和背景彩噪修复强度 |
| `SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE` / `SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR` | 控制 Stage 7 starless 像素修复和修复后 Stage 8 conservative skip |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` | 探测实验 workflow plugin commands |
| `SEESTAR_SPCC_ENABLE` / `SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS` | Stage 4 SPCC 行为 |
| `SEESTAR_ABERRATION_PROVIDER` | Stage 5/10 Aberration provider 选择（默认 arm64 优先 CoreML，再回退 CPU） |
| `SEESTAR_OPTIONAL_COLOR_TRANSFORM` | Stage 5/8/9 可选转色 |
| `SEESTAR_COSMIC_CLASSIC_ENABLE` / `SEESTAR_COSMIC_NATIVE_GPU` / `SEESTAR_COSMIC_CLASSIC_GPU` | Stage 5/10 CosmicClarity 行为；默认直接使用 Native，`SEESTAR_COSMIC_CLASSIC_ENABLE=1` 时才尝试 classic executable |
| `SEESTAR_SYQON_GPU` / `SEESTAR_SYQON_TIMEOUT_SEC` | Stage 7 SyQon 行为 |
| `SEESTAR_SIRILPY_TIMEOUT_SEC` | sirilpy/plugin subprocess timeout |
| `SEESTAR_DENOISE_ENABLE` / `SEESTAR_DENOISE_FORCE` | Stage 5 线性降噪 |

更完整的 runtime 与验证细节见 `INTEGRATION_README.md`。

## Limits

- 主要面向 Seestar 数据与 Siril 1.4+。
- 阶段 7 不再依赖 StarNet；去星链路仅保留 SyQon 与 SASP，全部不可用时会退化为继续使用拉伸图。

## Tests

```bash
python -m pytest tests/ -v
```

重点覆盖 GUI runtime、pipeline fallback、输入过滤、Stage11 单元与真实数据集成。
