# pipeline/AGENTS.md

约束 `pipeline/seestar_Superimpose.py`（stage 1-10，含 Stage 3/4 target preflight 与 Stage 6.5 辅助门控）和 `pipeline/stage11_ai_postprocess.py`（可选 Stage11）。

## Principles

- 默认保守：保护数据真实性优先于亮度、饱和度和“好看”。
- 顺序遵循线性 -> 非线性：裁切/目标画像/背景/校色/线性降噪 -> 拉伸/去星前门控/去星/星云增强/星点回混/导出。
- 禁止默认引入明显破坏真实性的行为：大面积纯黑背景、星色失真、饱和溢出、弱信号被抹除。
- 激进增强必须可选，默认关闭或低风险。

## Stage Contract

- Stage11 必须可选；失败只降级/跳过；只写 `*_ai` 副本；不得让 API/模型成为离线主流程硬依赖。

## Code Rules

- 可调参数集中在 `PipelineConfig`；新增参数需中文注释、保守默认值、安全钳制。
- 画质风险参数（降噪、星点、饱和度、混合权重）必须有上限或回退值。
- 涉及 SyQon、SASP、CosmicClarity 的改动需保持离线 wheels/cache 可运行，且不得把联网安装变成主流程依赖。
- Siril 命令走统一封装（如 `cmd_with_check`）；重试只用于幂等/可重入命令。
- 输入过滤不得误吃中间产物；TIFF/PNG/FITS 导出是稳定交付面。
- 保持 Python 3.10+ 可读性；不新增无关依赖；不做无关重构。

## Minimum Checks

- 主流程：`python3 -m py_compile pipeline/seestar_Superimpose.py`。
- 分离阶段模块：`python3 -m py_compile pipeline/stages/stage*.py`。
- Stage11：`python3 -m py_compile pipeline/stage11_ai_postprocess.py`。
- 人工核对：stage 顺序、`PipelineConfig` 注释、`StageResult` 降级、TIFF/PNG/FITS 导出、Stage11 不覆盖主产物。

## Change Note

- 说明影响 stage、对星点/背景/色彩/噪声的预期影响、回退路径变化、已验证与未验证风险。
