# Siril Plugin Integration

本目录用于放置工作流对齐所需的第三方插件与脚本缓存。GUI 只把小型脚本复制到临时覆盖层；wheels 与模型通过只读链接和环境变量注入 pipeline。

## 目录约定

- `downloads/`: 下载产物缓存（tar.gz / wheel）
- `vendor/`: 解压后的插件脚本或工具目录
- `syqon_starless/`: 本机 SyQon Zenith 模型缓存、校验/日期文件（模型不随 Git 分发）
- `graxpert/deconvolution-object-ai-models/1.0.1/`: GraXpert Object Deconvolution 官方模型
- `graxpert/bge-ai-models/model_v2_0_1/`: Stage 3 固定的 GraXpert BGE 2.0.1 受控模型缓存
- `cosmic_clarity/`: 本机 CosmicClarity Native/classic wrapper 共用模型（denoise 模型不随 Git 分发）
- `bin/CosmicClarity`: classic CosmicClarity 兼容 wrapper
- `download_siril_plugins.sh`: 基础离线依赖下载脚本（不包含 CosmicClarity denoise 模型）
- `requirements.txt`: Siril CPython 3.12 runtime 直接依赖及兼容版本范围；同时覆盖随包脚本使用的离线绘图、星表查询、图像分析、数据表与 Qt 主题依赖。
- `requirements.lock`: pip-tools 生成的固定传递依赖和 SHA256，供构建流程使用。
- `requirements-macos-arm64.lock`: macOS arm64 wheels 的固定版本与 SHA256；下载时由 pip `--require-hashes` 强制校验。
- `asset-checksums.sha256`: Siril scripts、GraXpert 与 SyQon 文件的可信 SHA256 清单。

## 下载与更新

在仓库根目录执行：

```bash
bash resources/siril_plugins/download_siril_plugins.sh
```

GraXpert BGE 没有在公开 release 中提供可锁定下载 URL。首次准备缓存时必须设置 `STARUN_GRAXPERT_BGE_MODEL_SOURCE=/absolute/path/model_v2_0_1.onnx`；脚本只在 SHA-256 为 `26d9e68370dfc079698aece805240a41782364f48c75f18ee4ff262c3f2ea8d2` 时导入，之后可直接复用已校验缓存。

脚本会尝试下载：

1. 官方 Siril scripts 仓库归档（GitLab）
2. SetiAstroSuitePro（PyPI wheel，单独 `--no-deps` 缓存）
3. PyQt6 / PySide6 / appdirs / astropy / scipy / matplotlib / largestinteriorrectangle / tifffile / ml-dtypes / onnx / onnxruntime / coloredlogs / humanfriendly / psutil / sep / spandrel / lz4 / zstandard / exifread / opencv-python-headless / requests / setuptools / wheel 等 CPython 3.12 兼容 wheels（包括可在 CP312 上运行的早期 `abi3` wheel）
4. SyQon Starless 所需的 PyTorch wheels
5. GraXpert Object Deconvolution 1.0.1 官方模型
6. SyQon Zenith 离线模型 `zenith.pt`

所有下载均先写入临时文件后再替换缓存。Siril scripts 固定到提交 `4cc9e204f9ddfd6d03cc4283aac76c82d4d19167`，原始 `SyQon/Starless.py` 必须匹配 SHA-256 `d36818f24a6927b245ab66fc7c00eaaa3b330a47406b61f0e9beb0764e06ab11`，随后才通过 `apply_syqon_offline_model_patch.py` 生成派生脚本；补丁、派生脚本和 Zenith 模型同时受 `asset-checksums.sha256` 锁定。补丁锚点或任何摘要不匹配都会失败并要求人工审查。GraXpert 发布包解压后仍校验 `model.onnx`，其余受管资源继续校验 SHA256；随后会删除与 CP312 不兼容的 ABI 及同库旧版。下载或校验失败会立即中断。第三方版本更新时必须同步审查并更新 lock/checksum 文件。当前 wheel 锁仅支持 macOS arm64 + CPython 3.12。

该脚本不会下载 CosmicClarity mono/color denoise 模型，因此不能单独准备真实离线 E2E runner。专用 runner 必须预置仓库外、只读且已核验的完整资源包，并通过 `STARUN_OFFLINE_RESOURCE_ROOT` 提供给 workflow；真实 E2E 不在运行期联网补模型，也不把 SyQon/CosmicClarity 模型加入 Git/LFS。

## 当前缓存边界

- `syqon_starless/`: Zenith 模型仅保留在本机缓存，不随 Git 分发；执行 `download_siril_plugins.sh` 可下载并校验。正式推理由固定上游快照经离线补丁生成的 `vendor/siril-scripts/SyQon/Starless.py` 执行。
- `graxpert/deconvolution-object-ai-models/1.0.1/`: 随项目保存 GraXpert Object Deconvolution 1.0.1 模型；Stage 5 自动注入该路径，不在任务中联网下载。
- `graxpert/bge-ai-models/model_v2_0_1/`: 不随 Git 分发；full App/core 资源包构建前必须由受控来源预置并匹配固定摘要。Stage 3 只接受该路径，不读取顶层或 `downloads/` 中的同名文件。
- `cosmic_clarity/`: CosmicClarity Native/classic wrapper 共用模型缓存目录。当前项目在本机保留 denoise + sharpen 的最小模型集：`deep_denoise_mono_AI4.pth`、`deep_denoise_color_AI4.pth`、`deep_sharp_stellar_AI4.pth`、`deep_nonstellar_sharp_conditional_psf_AI4.pth`；其中两个 denoise 模型不随 Git 分发，打包前须由本机缓存提供。
- `bin/CosmicClarity`: 兼容 Siril classic `CosmicClarity_Denoise.py` / `CosmicClarity_Sharpen.py` 协议的 standalone wrapper；内部使用 bundled `setiastrosuitepro`/`lz4`/`zstandard`/`exifread`/`opencv-python-headless`/`requests` wheel 和 `cosmic_clarity/` 模型；wrapper 会复用当前 Siril runtime 的 torch/torchvision，避免 SASPro 额外创建联网安装运行时。

## 运行时行为

- GUI 会复制小型插件脚本到临时覆盖层，并通过 `STARUN_SIRIL_PLUGIN_DIR` 注入 pipeline；`downloads/`、`syqon_starless/`、`graxpert/`、`cosmic_clarity/` 和模型保持只读链接。
- GraXpert 对象反卷积模型 1.0.1 随 App 分发，且不会由 pipeline 联网下载。Stage 5 自动选择该模型并只读链接到隔离 HOME；只有随包模型缺失时，才尝试本机 GraXpert 模型或 `STARUN_GRAXPERT_OBJECT_MODEL_PATH` 指定的官方 `model.onnx`、语义版本目录或 `deconvolution-object-ai-models` 家族目录，全部无效时继续回退 Siril RL。
- Stage 3 GraXpert BGE 固定使用 `model_v2_0_1`、减法、CPU 与 `-keep_bg`，同时保存背景模型和脚本/模型摘要。缺失或摘要不符时不会联网或尝试未注册原生命令，而是继续确定性候选链。
- SyQon 与 CosmicClarity 分别通过 `STARUN_SYQON_MODEL_DIR`、`STARUN_COSMIC_CLARITY_MODEL_DIR` 直接读取 App/离线资源包模型；只清理与 bundled 文件同尺寸的旧 runtime 受管副本。Stage 6 同时核对 Zenith 固定项目摘要和旁车摘要；SyQon 对该只读模型跳过版本检查和下载，缺失/损坏会 fail-closed，不访问 `siril.syqon.it`。离线补丁使 Stage 6 文件模式只接受显式输入/输出/manifest 路径，由 Astropy 直接读写，不连接 Siril 或调用 Siril 回写；生产 profile 固定 Statistical `0.15`、unlinked、`bp_sigma=5.0`、black clip、subtraction、`512/64`、FP32。数值交换固定为有限 `float32 0..1` FITS，并在写出前剥离源整数存储的缩放卡；v2 报告记录资产、参数、padding/crop/grid/coverage、成对哈希、变换 roundtrip 和 `source ≈ starless + raw_starmask` 闭环。上游更新只通过补丁脚本重放，禁止直接维护派生 vendor 文件。
- classic wrapper 仅在 `STARUN_COSMIC_CLASSIC_ENABLE=1` 时参与，其可写缓存留在临时覆盖层，模型仍使用只读链接。
- runtime 安装 `opencv-python-headless` 后，GUI 会写入同版本的 `opencv-python` 兼容分发元数据；第三方脚本仍按发行包名调用 `ensure_installed("opencv-python")`，但实际导入继续使用 headless wheel 提供的 `cv2`。
- Stage 2 不调用第三方 `Autocrop.py`。第一方裁切检测器直接使用离线 runtime 中的 `opencv-python-headless` 和 `largestinteriorrectangle` 计算有效轮廓与最大内接矩形；后者异常时回退到项目内的确定性二值掩膜最大矩形算法。经纬仪场旋形成的非黑低覆盖楔区由同一第一方模块以边缘连通亮度噪声 + 色噪证据检测，不新增插件依赖。检测器只生成候选坐标，像素裁切仍由主 pipeline 统一执行。vendor 中随上游脚本缓存出现的 `Autocrop.py` 不属于产品执行路径。
- `SIRIL_PYTHON_CLI` 可能被第三方脚本改写，wrapper 应优先使用 `STARUN_SIRIL_PYTHON_CLI` 作为稳定 Python 回退。
- `PySide6_Addons` 暂时保留；只有完成断网、全新 runtime 的 SyQon GUI/CLI 回归后，才能从离线缓存和安装列表移除。
