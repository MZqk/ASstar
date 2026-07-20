# Siril Plugin Integration

本目录用于放置工作流对齐所需的第三方插件与脚本缓存。GUI 只把小型脚本复制到临时覆盖层；wheels 与模型通过只读链接和环境变量注入 pipeline。

## 目录约定

- `downloads/`: 下载产物缓存（tar.gz / wheel）
- `vendor/`: 解压后的插件脚本或工具目录
- `syqon_starless/`: SyQon Starless 离线推理脚本、Zenith 模型、校验/日期文件
- `cosmic_clarity/`: CosmicClarity Native/classic wrapper 共用模型
- `bin/CosmicClarity`: classic CosmicClarity 兼容 wrapper
- `download_siril_plugins.sh`: 一键下载脚本
- `requirements.txt`: Siril CPython 3.12 runtime 直接依赖及兼容版本范围。
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
3. PyQt6 / PySide6 / appdirs / astropy / scipy / tifffile / ml-dtypes / onnx / onnxruntime / coloredlogs / humanfriendly / sep / spandrel / lz4 / zstandard / exifread / opencv-python-headless / requests / setuptools / wheel 等 CPython 3.12 兼容 wheels（包括可在 CP312 上运行的早期 `abi3` wheel）
4. SyQon Starless 所需的 PyTorch wheels
5. SyQon Starless 离线推理缓存：`syqon_starless_inference.py`、`zenith.pt`

所有下载均先写入临时文件并校验 SHA256，校验通过后才替换缓存；随后会删除与 CP312 不兼容的 ABI 及同库旧版。下载或校验失败会立即中断。第三方版本更新时必须同步审查并更新 lock/checksum 文件。当前 wheel 锁仅支持 macOS arm64 + CPython 3.12。

## 当前缓存边界

- `syqon_starless/`: 随项目保存 SyQon Starless 推理文件与 Zenith 模型，运行时从 App/离线资源包直接读取。
- `cosmic_clarity/`: CosmicClarity Native/classic wrapper 共用模型缓存目录。当前项目保留 denoise + sharpen 的最小模型集：`deep_denoise_mono_AI4.pth`、`deep_denoise_color_AI4.pth`、`deep_sharp_stellar_AI4.pth`、`deep_nonstellar_sharp_conditional_psf_AI4.pth`。
- `bin/CosmicClarity`: 兼容 Siril classic `CosmicClarity_Denoise.py` / `CosmicClarity_Sharpen.py` 协议的 standalone wrapper；内部使用 bundled `setiastrosuitepro`/`lz4`/`zstandard`/`exifread`/`opencv-python-headless`/`requests` wheel 和 `cosmic_clarity/` 模型；wrapper 会复用当前 Siril runtime 的 torch/torchvision，避免 SASPro 额外创建联网安装运行时。

## 运行时行为

- GUI 会复制小型插件脚本到临时覆盖层，并通过 `SEESTAR_SIRIL_PLUGIN_DIR` 注入 pipeline；`downloads/`、`syqon_starless/`、`cosmic_clarity/` 和 GraXpert 模型保持只读链接。
- GraXpert 对象反卷积模型不随 App 分发，也不会由 pipeline 联网下载。用户可在工作目录或 runtime home 的 `.seestar_ai.env` 设置 `SEESTAR_GRAXPERT_OBJECT_MODEL_PATH`，指向官方 `model.onnx`、语义版本目录或 `deconvolution-object-ai-models` 家族目录；Stage 5 校验后只读链接到隔离 HOME，缺失或无效时继续回退 Siril RL。
- SyQon 与 CosmicClarity 分别通过 `SEESTAR_SYQON_MODEL_DIR`、`SEESTAR_COSMIC_CLARITY_MODEL_DIR` 直接读取 App/离线资源包模型；只清理与 bundled 文件同尺寸的旧 runtime 受管副本。
- classic wrapper 仅在 `SEESTAR_COSMIC_CLASSIC_ENABLE=1` 时参与，其可写缓存留在临时覆盖层，模型仍使用只读链接。
- runtime 安装 `opencv-python-headless` 后，GUI 会写入同版本的 `opencv-python` 兼容分发元数据；第三方脚本仍按发行包名调用 `ensure_installed("opencv-python")`，但实际导入继续使用 headless wheel 提供的 `cv2`。
- `SIRIL_PYTHON_CLI` 可能被第三方脚本改写，wrapper 应优先使用 `SEESTAR_SIRIL_PYTHON_CLI` 作为稳定 Python 回退。
- `PySide6_Addons` 暂时保留；只有完成断网、全新 runtime 的 SyQon GUI/CLI 回归后，才能从离线缓存和安装列表移除。
