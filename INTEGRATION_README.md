# Seestar Superimpose Integration Notes

内部集成速查。用户说明见 `README.md`；全仓约束见 `AGENTS.md`；pipeline 约束见 `pipeline/AGENTS.md`。

## Layout

| Path | Role |
|---|---|
| `pipeline/seestar_Superimpose.py` | stage 1-10 主流程与编排 |
| `pipeline/stage11_ai_postprocess.py` | 可选 Stage11 AI 后期 |
| `gui/seestar_gui_app.py` | GUI 入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 GUI、preflight、Siril worker |
| `build/build_macos_app.sh` | macOS App 打包 |
| `resources/siril_plugins/` | 可选 Siril/SASP 插件缓存 |
| `resources/config.1.4.ini.template` | Siril 配置模板 |
| `resources/ai.env.example` | AI 配置示例 |
| `tests/` | GUI/runtime/pipeline/Stage11 测试 |

## Build

```bash
cd /Users/mz/dev/aiseestart
./build/build_macos_app.sh
```

Inputs:
- Required: `packages/python-3.13.12-macos11.pkg`, `packages/siril-1.4.2-arm64.dmg`
- Required Siril seed source: `~/Library/Application Support/org.siril.Siril/siril/{venv,.python_module}`; missing时先运行 Siril 初始化。

Notes:
- 打包前清理同名 `.app` 与 PyInstaller onedir。
- seed venv 只保留最小运行依赖；重型插件依赖运行时从本地 wheels 离线安装。
- `resources/ai.env` 和 `resources/siril_plugins/` 存在时会打包进 App Resources。
- `resources/siril_plugins/cosmic_clarity/` 保留 CosmicClarity Native/classic wrapper 共用的最小模型集：`deep_denoise_{mono,color}_AI4.pth` 与 `deep_{sharp_stellar,nonstellar_sharp_conditional_psf}_AI4.pth`。
- `resources/siril_plugins/bin/CosmicClarity` 是 bundled standalone classic wrapper，兼容 Siril classic 插件的 `input/`、`output/` 目录协议。
- 可选参数：`--app-name`、`--output-dir`、`--gui-entry`、`--pipeline-src`、`--config-template`、`--ai-env`、`--siril-src`。

## Runtime

- GUI 强制 `SIRIL_PYTHON_CLI` 指向 bundled Siril Python，并同步注入 `SEESTAR_SIRIL_PYTHON_CLI`，供 bundled wrapper 在插件脚本把 `SIRIL_PYTHON_CLI` 改写成布尔值时继续找到稳定 Python。
- GUI 强制 Siril `HOME` 到 `~/Library/Application Support/SeestarSuperimpose/runtime_home`。
- runtime venv 位于该 HOME 下的 `Library/Application Support/org.siril.Siril/siril`。
- 每轮注入 Siril CPU 上限 `floor(logical_cpu * 0.8)`，最小 1。
- 每轮确保 `~/.local/share/siril/gaia_photometric.dat/` 与 `~/.local/share/siril-scripts/`。
- Worker 运行时复制外部 pipeline、Stage11、可选 `siril_plugins` 到临时目录，不内嵌 pipeline 源码字符串。

## Env

AI env 来源优先级：进程环境 > 工作目录 `.seestar_ai.env` > runtime home `.seestar_ai.env` > bundled `ai.env`。

| Variable | Default | Scope |
|---|---:|---|
| `SEESTAR_AI_ENABLED` | `0` | AI 总开关；GUI `AI: ON/OFF` 强制覆盖，启用阶段 6-8 参数优化/诊断与 Stage11 副本 |
| `SEESTAR_AI_ENDPOINT` | unset | OpenAI-compatible chat completions endpoint |
| `SEESTAR_AI_MODEL` | unset | AI model |
| `SEESTAR_AI_API_KEY` | unset | AI API key |
| `SEESTAR_AI_TIMEOUT_SEC` | `90` | AI request timeout |
| `SEESTAR_AI_STRENGTH` | `0.12` | Stage11 conservative blend strength |
| `SEESTAR_AI_PROMPT` | built-in | Optional prompt |
| `SEESTAR_AI_STAGE6_ENABLE` | `1` | AI 开启时允许 Stage 6 拉伸参数建议和诊断 |
| `SEESTAR_AI_STAGE7_ENABLE` | `1` | AI 开启时允许 Stage 7 SyQon 参数计划、残星诊断、残星抑制、参数重试择优与 Stage 9 回混降强度联动 |
| `SEESTAR_AI_STAGE8_ENABLE` | `1` | AI 开启时允许 Stage 8 `satu/unsharp` 参数计划、AI 蓝偏目标、蓝色修正、修正后重算质量与保守重跑 |
| `SEESTAR_STAGE7_QUALITY_RETRY_MAX` | `2` | Stage 7 质量偏差时的 SyQon 参数优化预算，钳制 `0~3`；SyQon 参数下限为 `tile=512/overlap=64` |
| `SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH` | `1.35` | Stage 7 质量差时生成更轻的去星输入，钳制到 `1.05~ultra_conservative`，用于减少黑洞和彩色残渣 |
| `SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX` | `0.60` | M42/亮核心星云的 Stage 7 halo 验收上限；普通目标仍使用 `0.35` |
| `SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH` | `0.68` | Stage 7 选中 starless 后的小尺度残星局部平滑/修补强度 |
| `SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH` | `0.70` | Stage 7 亮星 halo 局部平滑强度 |
| `SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH` | `0.55` | Stage 7 背景 chroma noise reduction 强度 |
| `SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE` | `1` | 是否启用 Stage 7 starless 像素修复 |
| `SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR` | `1` | Stage 7 修复或仍不安全时是否强制 Stage 8 conservative skip |
| `SEESTAR_WORKFLOW_PLUGIN_PROBE` | `0` | 探测实验 plugin commands |
| `SEESTAR_SPCC_ENABLE` | `1` | Stage 4 SPCC；`0` 强制 PCC/CCM |
| `SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS` | `0` | 允许 `Light_` 预处理模式尝试 SPCC；默认关闭以规避 `siril-cli` aperture photometry SIGSEGV |
| `SEESTAR_ABERRATION_API_ENABLE` | `0` | Stage 5/10 SASP Aberration API fallback |
| `SEESTAR_ABERRATION_PROVIDER` | unset | Stage 5/10 Aberration provider；默认 Apple Silicon 优先 `CoreMLExecutionProvider`，失败回退 CPU；可设 `cpu` |
| `SEESTAR_OPTIONAL_COLOR_TRANSFORM` | `0` | Stage 5/8/9 可选转色 |
| `SEESTAR_COSMIC_CLASSIC_ENABLE` | `0` | Stage 5/10 是否尝试 classic CosmicClarity；默认直接用 Native，避免 headless 下 classic 无变化/undo 回写失败 |
| `SEESTAR_COSMIC_CLARITY_EXECUTABLE` | unset | classic CosmicClarity executable；仅 `SEESTAR_COSMIC_CLASSIC_ENABLE=1` 时使用 |
| `SEESTAR_COSMIC_CLASSIC_GPU` | `1` | classic CosmicClarity 是否允许 GPU/device auto；`0` 强制 `-no_gpu` |
| `SEESTAR_COSMIC_NATIVE_GPU` | `1` | Stage 10 Native 自动用 MPS/CUDA/XPU/CPU；`0` 强制 CPU |
| `SEESTAR_SIRILPY_TIMEOUT_SEC` | `120` from GUI | sirilpy/plugin subprocess timeout |
| `SEESTAR_SYQON_GPU` | `1` | Stage 7 SyQon GPU backend；`0` 强制 CPU |
| `SEESTAR_SYQON_TIMEOUT_SEC` | `900` | Stage 7 SyQon timeout，钳制 `60~1800` |
| `SEESTAR_DENOISE_ENABLE` | unset | Stage 5 线性降噪基线 |
| `SEESTAR_DENOISE_FORCE` | unset | auto-tune 后保留 denoise enabled |
| `SEESTAR_INPUT_MODE` | `auto` | `auto` 或 `result_linear_resume` |
| `SEESTAR_DEBUG_MODE` | GUI toggle | 保留 `process/stage*.fit` |

## Pipeline Contracts

- Stage states: `ok`、`degraded`、`failed`、`skipped`；summary 可显示 `ok_with_fallback`、`ok_skipped_optional`。
- Stage 5 enhance fallback: classic CosmicClarity（env/config/常见安装位置自动发现） -> Native Sharpen/Denoise -> Siril-CC Sharpen / built-in denoise / unsharp。若仅是 classic executable 未配置且 Native 成功，summary 记为 `ok`，不记 `ok_with_fallback`。
- Stage 10 denoise fallback: in-process CosmicClarity Denoise -> classic CLI（env/config/常见安装位置自动发现） -> Native/SCUNet CLI/Siril alias -> Aberration API（若启用）。
- Stage 11 AI postprocess: AI plan 成功时使用模型参数和可选 `blend_strength`；解析失败时按最终图像特征生成保守 fallback，并写入 `stage11_quality.json` 与 `ai_raw_stage11_*`。
- Stage 7 starless fallback: SyQon -> SASP Dark Star / degraded continuation。质量差时会追加更轻 Asinh 输入重跑 SyQon，并对最终 starless 做残星、halo、暗坑和背景彩噪像素修复；修复结果写入 `stage7_quality.json`。
- Stage 8 Starless enhancement uses the bundled `setiastrosuitepro` WaveScale Dark Enhancer Python API. The packaged workflow does not probe experimental `sasp_*` Siril commands because the current SASP wheel does not register them with Siril CLI; the deterministic built-in `satu + unsharp` chain is the fallback when the API is unavailable. When Stage 7 marks starless as repaired or when bright-nebula halo is above the ordinary threshold but below the bright-nebula threshold, Stage 8 enters conservative skip and does not apply SASP/detail boost/saturation boost.
- Adaptive Stage 1-6.5: Stage 2.5 writes `target_profile.json` / `pipeline_policy.json` in both normal and `result_linear_resume` modes; Stage 3-5 write policy-aware reports; Stage 6 writes `stretch_candidates_report.json` and keeps `stage6_selected.fit` plus compatible `stage6_stretched.fit`; M42/bright-nebula HDR masked candidates use a mode-specific chroma gate with absolute chroma-noise protection; Stage 6.5 writes `pre_starless_gate_report.json` and can route or block Stage 7 based on starless readiness.
- Quality/processing guards: Stage 2 adaptive edge crop、Stage 6 weak-object tune/policy candidate scoring/AI stretch diagnostics + hard black-field gate、Stage 6.5 pre-starless gate、Stage 7 AI SyQon parameter plan + retry best-selection + residual-map smoothing + starless pixel repair + degraded status when the selected starless remains poor、Stage 8 conservative skip after repaired/unsafe starless + compact-target mask threshold + AI `satu/unsharp` plan + blue guard/corrective rerun/recalculated quality、Stage 9 intensity scaling + unsafe-starless skip + bad-starless bypass to Stage 6 review-safe output + `stage7_quality.json`、Stage 10 effective denoise path logging and missing-candidate fallback without noisy load errors。
- AI advisory JSON parse failures write `process/ai_raw_*.json/txt`; Stage 6/7 会先尝试从 reasoning 文本中抽取方法、tile/overlap、残星/回混参数，仍不可用时再打开本轮 per-stage circuit breaker。
- `result_linear_resume` 要求 `<work_dir>/result_linear.fit`，跳过 stage 2-5，但仍运行 Stage 2.5 目标画像识别，再从 Stage 6 继续。

## Plugin Cache

- 准备：`bash resources/siril_plugins/download_siril_plugins.sh`
- GUI 每轮检查插件脚本、SASP/SyQon/AberrationRemover、PyQt6/PySide6/astropy/scipy/tifffile/onnxruntime 等 wheels。
- 缺失时自动尝试下载；仍缺失则阻断运行。
- runtime offline pip 只从 bundled wheels 安装，并写入 `tiffile` shim 与 `sitecustomize.py` sirilpy timeout patch。

## Validation Matrix

| Changed area | Required checks |
|---|---|
| Pipeline | `python3 -m py_compile pipeline/seestar_Superimpose.py`; 核对 stage 顺序、降级、TIFF/PNG/FITS |
| Adaptive Stage 1-6.5 | `python3 -m unittest tests.test_adaptive_pipeline_phase1`; 检查 target fallback、policy fallback、dirty background 禁止 autostretch、Stage6.5 conservative input |
| Stage11 | `python3 -m py_compile pipeline/stage11_ai_postprocess.py`; 确认只写 `*_ai` 且失败不阻断 stage 10 产物 |
| AI Stage 6-8 | AI 关闭时确认沿用确定性路径；AI 缺凭据/API 失败时只记录计划 fallback；AI 开启时确认 Stage 7 SyQon 参数和 Stage 8 `satu/unsharp` 参数实际进入命令日志，并检查 Stage 7 重试择优、Stage 9 回混降强度、Stage 8 修正/重跑是否覆盖 FIT |
| GUI/preflight | 缺 Siril/config/pipeline/FITS/result_linear 的错误清晰；Stop 行为仍安全 |
| Build | `bash -n build/build_macos_app.sh`; app 仍嵌入 Siril、pipeline、config、可选资源 |
| Plugins | cache 完整性、缺失补齐/阻断、offline wheels 安装 |
| Toggles | `Debug`、`AI`、`联网`、处理模式映射到预期 env/Siril flags |

Baseline smoke: embedded Siril + external pipeline 可运行；stage 1-10 能导出 TIFF/PNG/FITS；Debug 保留 `process/stage*.fit`。
