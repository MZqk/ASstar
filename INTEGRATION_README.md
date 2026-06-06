# Seestar Superimpose Integration Notes

内部集成速查。用户说明见 `README.md`；全仓约束见 `AGENTS.md`；pipeline 约束见 `pipeline/AGENTS.md`。

## Layout

| Path | Role |
|---|---|
| `pipeline/seestar_Superimpose.py` | stage 1-10 主流程编排 |
| `pipeline/stages/` | stage 1-10 分离实现模块 |
| `pipeline/stage11_ai_postprocess.py` | 可选 Stage11 AI 后期 |
| `gui/seestar_gui_app.py` | GUI 入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 GUI、preflight、Siril worker |
| `build/build_macos_app.sh` | macOS App 打包 |
| `resources/siril_plugins/` | 可选 Siril/SASP 插件缓存 |
| `resources/config.1.4.ini.template` | Siril 配置模板 |
| `resources/default.env` | 项目默认 runtime env；打包时进入 App Resources |
| `resources/ai.env` | 本地/打包覆盖 env；可放 API key，默认被 git 忽略 |
| `resources/ai.env.example` | 本地覆盖示例 |
| `tests/` | GUI/runtime/pipeline/Stage11 测试 |

## Build

```bash
cd /Users/mz/dev/aiseestart
./build/build_macos_app.sh
```

Inputs:
- Required: `packages/python-3.13.12-macos11.pkg`, `packages/siril-1.4.3-arm64.dmg`
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
- Worker 同步复制 `pipeline/stages/` 和共享 helper 模块；阶段实现不应依赖仓库外相对路径。

## Env

Runtime env 来源优先级：进程环境 > 工作目录 `.seestar_ai.env` > runtime home `.seestar_ai.env` > bundled/project `ai.env` > bundled/project `default.env`。GUI worker 会在此基础上强制覆盖 `SEESTAR_DEBUG_MODE`、`SEESTAR_INPUT_MODE`、`SEESTAR_AI_ENABLED` 和 Siril runtime 路径类变量。

完整 pipeline env 解释以 `pipeline/seestar_Superimpose_workflow.md` 为准；本节只保留 runtime、打包和 GUI 注入相关项。

| Variable | Default | Scope |
|---|---:|---|
| `SEESTAR_DEBUG_MODE` | GUI toggle | GUI 强制写入；控制是否保留 `process/stage*.fit` |
| `SEESTAR_INPUT_MODE` | `auto` | GUI 处理模式；`stage2_corrected_resume` 要求工作目录根下或 `process/` 有 `stage2_corrected.fit`，从 Stage 3 继续；`result_linear_resume` 要求工作目录有 `result_linear.fit` |
| `SEESTAR_AI_ENABLED` | GUI toggle | GUI 强制写入；AI endpoint/model/key 仍按 env 优先级读取 |
| `SEESTAR_AI_ENDPOINT` / `SEESTAR_AI_MODEL` / `SEESTAR_AI_API_KEY` | unset | 可来自工作目录、runtime home、bundled/project env 文件 |
| `SIRIL_PYTHON_CLI` / `SEESTAR_SIRIL_PYTHON_CLI` | bundled Python | GUI 注入稳定 Siril Python；wrapper 用后者兜底 |
| `SEESTAR_SIRIL_PLUGIN_DIR` | bundled/runtime plugin dir | GUI 指向打包或复制后的插件缓存 |
| `SEESTAR_SIRILPY_TIMEOUT_SEC` | `120` from GUI | sirilpy/plugin subprocess timeout |

## Pipeline Contracts

Pipeline 细节以 `pipeline/seestar_Superimpose_workflow.md` 为准；这里仅保留集成层必须稳定的契约。

- Stage states: `ok`、`degraded`、`failed`、`skipped`；summary 可显示 `ok_with_fallback`、`ok_skipped_optional`。
- Worker 必须复制外部 `pipeline/stages/`、Stage11 和共享 helper 模块；不再内嵌 pipeline 源码字符串。
- Stage 11 必须保持可选，只写 `*_ai` 产物，失败不得阻断 Stage 10 原始输出。
- `stage2_corrected_resume` 要求 `<work_dir>/stage2_corrected.fit` 或 `<work_dir>/process/stage2_corrected.fit`，将其作为已完成裁切/视场修正的叠加后中间结果，并从 Stage 3 继续完整后处理。
- `result_linear_resume` 要求 `<work_dir>/result_linear.fit`，记录 stage 2-5 为 skipped，并从 Stage 6 继续。
- Stage 6/7 兼容别名属于外部和调试界面：`stage6_starless*` 与 `stage7_starless*`、`stage6_starless_quality.json` 与 `stage7_quality.json` 需继续兼容。
- Stage 10 导出必须保留 TIFF/PNG/FITS fallback 名称，包括续跑模式的 `_linear` fallbacks。
- Runtime plugin 路径必须保持离线可用：SyQon、SASP、CosmicClarity 资源来自 bundled/local cache；缺失时应作为 preflight/runtime 问题处理，不应要求 pipeline 内联网下载。

## Plugin Cache

- 准备：`bash resources/siril_plugins/download_siril_plugins.sh`；脚本按 Python 3.13 下载 wheel，并清理 `downloads/` 中同一库的旧版本/非 3.13 wheel。
- GUI 每轮检查插件脚本、SASP/SyQon/AberrationRemover、PyQt6/PySide6/astropy/scipy/tifffile/onnxruntime 等 wheels。
- 缺失时自动尝试下载；仍缺失则阻断运行。
- runtime offline pip 只从 bundled wheels 安装，并写入 `tiffile` shim 与 `sitecustomize.py` sirilpy timeout patch。

## Validation Matrix

| Changed area | Required checks |
|---|---|
| Pipeline | `python3 -m py_compile pipeline/seestar_Superimpose.py pipeline/stages/*.py`; 核对 stage 顺序、降级、TIFF/PNG/FITS |
| Target policy / gates | `python3 -m unittest tests.test_adaptive_pipeline_phase1`; 检查 target fallback、policy fallback、dirty background 保护和保守输入门控 |
| Stage11 | `python3 -m py_compile pipeline/stage11_ai_postprocess.py`; 确认只写 `*_ai` 且失败不阻断 stage 10 产物 |
| AI Stage 6/8/11 | AI 关闭时确认沿用确定性路径；AI 缺凭据/API 失败时只记录计划 fallback；AI 开启时确认阶段 6 SyQon 参数和 Stage 8 masked-enhancement 参数实际进入日志，并检查阶段 6 重试择优、Stage 9 回混降强度、Stage 8 修正/重跑是否覆盖 FIT；Stage 7 拉伸仍应保持固定两候选 |
| GUI/preflight | 缺 Siril/config/pipeline/FITS/result_linear 的错误清晰；Stop 行为仍安全 |
| Build | `bash -n build/build_macos_app.sh`; app 仍嵌入 Siril、pipeline、config、可选资源 |
| Plugins | cache 完整性、缺失补齐/阻断、offline wheels 安装 |
| Toggles | `Debug`、`AI`、`联网`、处理模式映射到预期 env/Siril flags |

Baseline smoke: embedded Siril + external pipeline 可运行；stage 1-10 能导出 TIFF/PNG/FITS；Debug 保留 `process/stage*.fit`。
