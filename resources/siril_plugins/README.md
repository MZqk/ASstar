# Siril Plugin Integration

本目录用于放置工作流对齐所需的第三方插件与脚本缓存，并由 GUI 在运行时复制到临时目录，通过环境变量 `SEESTAR_SIRIL_PLUGIN_DIR` 注入到 pipeline。

## 目录约定

- `downloads/`: 下载产物缓存（tar.gz / wheel）
- `vendor/`: 解压后的插件脚本或工具目录
- `download_siril_plugins.sh`: 一键下载脚本
- `requirements.txt`: Siril runtime 插件依赖说明；对应 wheel 需缓存在 `downloads/`，供离线 venv 安装使用。

## 下载与更新

在仓库根目录执行：

```bash
bash resources/siril_plugins/download_siril_plugins.sh
```

脚本会尝试下载：

1. 官方 Siril scripts 仓库归档（GitLab）
2. SetiAstroSuitePro（PyPI wheel）
3. PyQt6 / PySide6 / astropy / scipy / tifffile / onnxruntime / sep / spandrel 等 Siril 插件运行时 wheels
4. SyQon Starless 所需的 PyTorch wheels
5. SyQon Starless 离线推理缓存：`syqon_starless_inference.py`、`zenith.pt`

下载失败不会删除已有缓存。

## 当前缓存边界

- `syqon_starless/`: 随项目保存 SyQon Starless 推理文件与 Zenith 模型，GUI 启动前会同步到 Siril 运行时 user data 目录。
- `cosmic_clarity/`: CosmicClarity Native/classic wrapper 共用模型缓存目录。当前项目保留 denoise + sharpen 的最小模型集：`deep_denoise_mono_AI4.pth`、`deep_denoise_color_AI4.pth`、`deep_sharp_stellar_AI4.pth`、`deep_nonstellar_sharp_conditional_psf_AI4.pth`。
- `bin/CosmicClarity`: 兼容 Siril classic `CosmicClarity_Denoise.py` / `CosmicClarity_Sharpen.py` 协议的 standalone wrapper；内部使用 bundled `setiastrosuitepro`/`lz4`/`zstandard`/`exifread`/`opencv-python-headless`/`requests` wheel 和 `cosmic_clarity/` 模型；wrapper 会复用当前 Siril runtime 的 torch/torchvision，避免 SASPro 额外创建联网安装运行时。
