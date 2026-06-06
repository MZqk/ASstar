# Siril Plugin Integration

本目录用于放置工作流对齐所需的第三方插件与脚本缓存，并由 GUI 在运行时复制到临时目录，通过环境变量 `SEESTAR_SIRIL_PLUGIN_DIR` 注入到 pipeline。

## 目录约定

- `downloads/`: 下载产物缓存（tar.gz / wheel）
- `vendor/`: 解压后的插件脚本或工具目录
- `syqon_starless/`: SyQon Starless 离线推理脚本、Zenith 模型、校验/日期文件
- `cosmic_clarity/`: CosmicClarity Native/classic wrapper 共用模型
- `bin/CosmicClarity`: classic CosmicClarity 兼容 wrapper
- `download_siril_plugins.sh`: 一键下载脚本
- `requirements.txt`: Python 3.13 runtime 直接依赖及兼容版本范围。
- `requirements.lock`: pip-tools 生成的固定传递依赖和 SHA256，供构建流程使用。
- `requirements-macos-arm64.lock`: macOS arm64 wheels 的固定版本与 SHA256；下载时由 pip `--require-hashes` 强制校验。
- `asset-checksums.sha256`: Siril scripts 与 SyQon 文件的可信 SHA256 清单。

## 下载与更新

在仓库根目录执行：

```bash
bash resources/siril_plugins/download_siril_plugins.sh
```

脚本会尝试下载：

1. 官方 Siril scripts 仓库归档（GitLab）
2. SetiAstroSuitePro（PyPI wheel，单独 `--no-deps` 缓存）
3. PyQt6 / PySide6 / astropy / scipy / tifffile / onnxruntime / coloredlogs / humanfriendly / sep / spandrel / lz4 / zstandard / exifread / opencv-python-headless / requests / setuptools / wheel 等 Python 3.13 运行时 wheels
4. SyQon Starless 所需的 PyTorch wheels
5. SyQon Starless 离线推理缓存：`syqon_starless_inference.py`、`zenith.pt`

所有下载均先写入临时文件并校验 SHA256，校验通过后才替换缓存；下载或校验失败会立即中断且不会覆盖已有缓存。第三方版本更新时必须同步审查并更新 lock/checksum 文件。当前 wheel 锁仅支持 macOS arm64。

## 当前缓存边界

- `syqon_starless/`: 随项目保存 SyQon Starless 推理文件与 Zenith 模型，GUI 启动前会同步到 Siril 运行时 user data 目录。
- `cosmic_clarity/`: CosmicClarity Native/classic wrapper 共用模型缓存目录。当前项目保留 denoise + sharpen 的最小模型集：`deep_denoise_mono_AI4.pth`、`deep_denoise_color_AI4.pth`、`deep_sharp_stellar_AI4.pth`、`deep_nonstellar_sharp_conditional_psf_AI4.pth`。
- `bin/CosmicClarity`: 兼容 Siril classic `CosmicClarity_Denoise.py` / `CosmicClarity_Sharpen.py` 协议的 standalone wrapper；内部使用 bundled `setiastrosuitepro`/`lz4`/`zstandard`/`exifread`/`opencv-python-headless`/`requests` wheel 和 `cosmic_clarity/` 模型；wrapper 会复用当前 Siril runtime 的 torch/torchvision，避免 SASPro 额外创建联网安装运行时。

## 运行时行为

- GUI 会在运行前把本目录复制到临时 runtime 插件目录，并通过 `SEESTAR_SIRIL_PLUGIN_DIR` 注入 pipeline。
- GUI 会把 `syqon_starless/` 同步到 Siril user data 目录，供 `SyQon-Starless.py` 离线使用。
- GUI 会把 `cosmic_clarity/` 模型同步到 runtime 插件目录，Native 优先使用本地模型；classic wrapper 仅在 `SEESTAR_COSMIC_CLASSIC_ENABLE=1` 时参与。
- `SIRIL_PYTHON_CLI` 可能被第三方脚本改写，wrapper 应优先使用 `SEESTAR_SIRIL_PYTHON_CLI` 作为稳定 Python 回退。
