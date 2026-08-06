# Seestar Superimpose Integration Notes

内部集成速查。用户说明见 `README.md`；全仓约束见 `AGENTS.md`；pipeline 约束见 `pipeline/AGENTS.md`。

## Layout

| Path | Role |
|---|---|
| `pipeline/seestar_Superimpose.py` | stage 1-10 主流程编排 |
| `pipeline/stages/` | stage 1-10 分离实现模块 |
| `pipeline/stage_contracts.py` / `pipeline/task_plan.py` | 产品阶段、正式续跑点、产物命名与共享冻结路由契约 |
| `pipeline/input_discovery.py` / `pipeline/task_workspace.py` | 文件/Light/产品任务识别、只读来源指纹、独立任务目录与断点验签 |
| `pipeline/stage11_ai_postprocess.py` | 已停用的隔离实验模块，不进入产品运行时 |
| `pipeline/ai_artistic_derivative.py` | 已停用的隔离艺术衍生实验模块 |
| `gui/seestar_gui_app.py` | GUI 入口 |
| `gui/seestar_gui_dev.py` | 系统 Siril + 现有 seed 的免打包开发入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 GUI、preflight、Siril worker |
| `gui/ui_theme.py` / `gui/ui_platform.py` | 语义设计令牌、浅深色主题、系统字体与平台策略 |
| `build/build_macos_app.sh` | macOS App 打包 |
| `resources/siril_plugins/` | 可选 Siril/SASP 插件缓存 |
| `resources/siril_spcc_database/` | 固定版本、带清单与 SHA-256 的最小 SPCC 传感器/滤镜/白参考种子；不含 Gaia 星表 |
| `resources/config.1.4.ini.template` | Siril 配置模板 |
| `resources/default.env` | 项目默认 runtime env；打包时进入 App Resources |
| `resources/ai.env` | 开发者试用配置的本地构建输入；非空 Key 时必须为 `0600` |
| `resources/ai.env.example` | 本地覆盖示例 |
| `tests/` | GUI/runtime/pipeline 与停用边界测试 |

## Build

```bash
cd /Users/mz/dev/aiseestart
./build/build_macos_app.sh
```

Inputs:
- Required: `packages/python-3.13.12-macos11.pkg`, `packages/siril-1.4.4-arm64.dmg`
- Required Siril module source: `~/Library/Application Support/org.siril.Siril/siril/.python_module`；缺失时先运行 Siril 初始化。用户目录中的 `venv` 不再作为构建输入。
- Build host: Apple Silicon Mac；最终 App 要求 macOS 14.0+，主程序必须是纯 `arm64`。

Notes:
- PyInstaller 在 `release/.seestar_build.*` 临时目录中完成构建；只有资源校验与深度签名全部通过后才替换同名 `.app`，构建中断不会把半成品发布到 `release/`。
- PyInstaller 强制使用 `arm64` 目标；构建后写入并校验 `LSMinimumSystemVersion=14.0`，同时用 `lipo` 拒绝非纯 `arm64` 主程序。
- Bundle 元数据固定为 `CFBundleIdentifier=StarunC`、`CFBundleShortVersionString=0.1`、`CFBundleVersion=1`；GUI runtime 指纹读取为 `0.1 (1)`，版本变化时不复用旧依赖缓存。
- 构建下载依赖时使用 `requirements.lock` 和 `resources/siril_plugins/requirements.lock`，并强制校验 pip-tools 生成的 SHA256。
- seed venv 由 App 内置 Siril Python 3.12 和本地 wheels 在构建时重新生成，不复用用户目录中可能被其他 Python 版本改写的 venv；seed 只保留最小运行依赖，重型插件依赖运行时再从本地 wheels 离线安装。
- 构建脚本校验 `resources/siril_spcc_database/SHA256SUMS` 后嵌入 `SirilSPCCDatabaseSeed/`；Worker 原子同步到隔离 HOME 的 `siril-spcc-database`。构建仍拒绝打包 astrometric 或 `xp_sampled` Gaia 星表。
- `resources/ai.env` 存在时，构建脚本只把 endpoint/model 等非敏感字段写入 App Resources；非空 Key 转为 `ai-trial.bootstrap`，首次使用后导入当前用户的 macOS Keychain。原始文件不会复制进 App。
- `resources/siril_plugins/cosmic_clarity/` 保留 CosmicClarity Native/classic wrapper 共用的最小模型集：`deep_denoise_{mono,color}_AI4.pth` 与 `deep_{sharp_stellar,nonstellar_sharp_conditional_psf}_AI4.pth`。
- `resources/siril_plugins/bin/CosmicClarity` 是 bundled standalone classic wrapper，兼容 Siril classic 插件的 `input/`、`output/` 目录协议。
- 默认 `--bundle-profile full` 生成单体 Full Offline App。`--bundle-profile core` 将完整 `siril_plugins` 输出到相邻的 `<AppName>-OfflineResources/`，App 启动时自动发现；可用 `--offline-resource-pack-dir` 自定义输出位置。
- 可选参数：`--app-name`、`--output-dir`、`--gui-entry`、`--pipeline-src`、`--config-template`、`--ai-env`、`--siril-src`、`--codesign-identity`、`--bundle-profile`、`--offline-resource-pack-dir`。
- `--codesign-identity` 默认仍为 `-`（ad-hoc，适合本地验证）。公开升级发行应使用固定 Developer ID Application 身份；稳定签名可降低 App 更新后 Keychain ACL 重新确认或无法读取的风险。完整公证仍需另行配置 hardened runtime、timestamp 和 notarization。

## Runtime

- 开发 launcher 通过临时资源覆盖层把显式 `--siril-app`、
  `--siril-seed` 与项目 `resources/` 组合后注入 `SeestarGui`。覆盖层只含符号链接，
  launcher 不改写来源，退出 GUI 后自动清理；正式 `gui/seestar_gui_app.py` 与
  frozen App 不受影响。
- GUI 强制 `SIRIL_PYTHON_CLI` 指向 bundled Siril Python，并同步注入 `SEESTAR_SIRIL_PYTHON_CLI`，供 bundled wrapper 在插件脚本把 `SIRIL_PYTHON_CLI` 改写成布尔值时继续找到稳定 Python。
- GUI 强制 Siril `HOME` 到 `~/Library/Application Support/SeestarSuperimpose/runtime_home`。
- runtime venv 位于该 HOME 下的 `Library/Application Support/org.siril.Siril/siril`。
- 每轮注入 Siril CPU 上限 `floor(logical_cpu * 0.8)`，最小 1。
- 每轮确保 `~/.local/share/siril-scripts/`，校验并同步最小 SPCC 元数据种子；元数据不可用时在启动 Siril 前禁用 SPCC，使宽带进入 PCC 异常回退。该种子只含 Sony IMX585、Seestar LP/No filter 和白参考曲线，不含 Gaia XP 光谱。
- 完全离线的 platesolve/PCC 可在 runtime 目录共用有效 `siril_cat_healpix8_astro.dat`；离线 SPCC 则读取独立的 `siril_cat1_healpix8_xpsamp/` 分块目录。Worker 将两条路径写入临时 Siril config，并分别注入 `SEESTAR_GAIA_ASTRO_CATALOG` / `SEESTAR_GAIA_PHOTO_CATALOG`。GUI 下载器目前只安装约 1.52 GB 的 astrometric/PCC 目录；项目、App 与离线资源包均不得携带任一 Gaia 目录。目录缺失时只有 `SEESTAR_NETWORK_MODE=1` 才允许对应的在线 Gaia 请求。
- Worker 运行时复制外部 Stage 1-10 pipeline、`pipeline/configs/` 目标目录/策略和小型插件脚本到临时目录；已停用的后续实验模块明确排除。`downloads`、`syqon_starless`、`cosmic_clarity` 使用指向 App/离线资源包的只读符号链接，不再产生约 1.7 GB 的临时副本。
- Worker 同步复制 `pipeline/stages/` 和共享 helper 模块；阶段实现不应依赖仓库外相对路径。
- GUI worker 将每轮 Siril 放入独立进程组：默认连续 900 秒无输出时记录最后命令、进程状态和本轮产物后终止整组；检测到本轮新 PNG 后，若连续 120 秒无输出且仍未退出，则按“导出成功、收尾异常”结束。GUI/任务日志将 Siril 与插件逐百分比输出压缩为约 10% 一档，并把重复的 ICCProfile HDU 跳过提示、Stage 5 NL-Bayes 已确认会继续执行的 `src fits` 初始化诊断各归并为一条 INFO；阶段变化、进度重启、完成行和其他非进度输出仍完整保留。原始输出仍逐行参与 watchdog 活跃度判断，已归并的无害诊断不会挤占崩溃诊断的最近有效输出缓冲区。
- GUI 先在主线程完成工作目录和输入的快速 preflight，再由 `BootstrapWorker` 分别预估系统运行时卷与工作目录卷；同卷时合并需求判断。工作目录预估以输入 FITS 实际大小为基准，按完整叠加流程 `44` 份、Stage 2 续跑 `40` 份、线性续跑 `28` 份的阶段/临时产物峰值计算，Light 模式另加预处理 sequence 预算，最后增加 `max(1 GiB, 15%)` 余量。检查通过后才执行 runtime seed、插件检查和离线依赖准备。SyQon/CosmicClarity 模型通过环境变量直接指向 App/离线资源包，并清理与 bundled 文件同尺寸的旧受管副本。准备阶段的“停止”会终止当前下载/安装子进程组。runtime venv 内的 `.seestar_runtime_ready.json` 使用依赖锁 SHA-256、Siril Python ABI 和 App 版本作为指纹，命中时跳过重复 `pip install`。
- GUI 使用稳定工具栏、任务输入/预览工作区、两阶段概览和固定状态栏作为主框架，并以显式 `empty / task / run` 状态切换内容区。首次启动或保存输入失效时进入拖放空状态；有效输入直接进入任务设置并重新检测。输入可为明确母版文件、递归 Light 目录或产品任务目录；主界面只显示由契约生成的唯一 Stage 1-10 计划，不再显示手动阶段/处理方式选择器。快速 preflight 通过后立即进入只读运行视图，完成、失败和停止也保持该视图，只有显式“返回任务设置”才退出。窗口默认 `1280×800`、最小 `980×680`，低于 1100 px 时折叠运行配置侧栏。文件/目录拖放、最近输入和 `QSettings` 持久化保留；窗口位置/尺寸、高级设置、全宽参数 sheet 与日志展开状态会恢复。“处理参数…”在任务输入区下方展开为可滚动的全宽单页 sheet，承载输出、校色、滤镜、降噪、反卷积、GraXpert 对象模型和计算兼容模式；“专业细节”以三组参数/行追加 12 个受 GUI 与 pipeline 双重限幅的 Stage 4/5/6/7/9 数值参数。每项同时设置 tooltip 与 accessibility description，修改后自动写入 `QSettings`，开始任务时冻结为只读快照；自动续跑会禁用已完成阶段参数。保留中间文件与联网仍为快捷开关，默认关闭。停用功能的控件、设置恢复、预检、阶段事件和日志输出均已移除；旧参数不能重新启用。主控件设置 label buddy、辅助功能名称/说明、状态播报和显式 Tab 顺序。
- 视觉系统由 `ui_theme.py` 的浅色/深色语义令牌与 `ui_platform.py` 的平台策略共同生成，不在业务窗口中复制 QSS。macOS 使用系统应用字体、原生菜单、统一标题栏/工具栏、30 px 控件与 Command 快捷键；Windows 复用同一 widget tree 和信息架构，改用系统字体、窗口内菜单、32 px 控件与 Control 快捷键。平台分支不得承载参数、preflight、worker 或 pipeline 行为。成功、旁路、降级、失败、停止和当前运行阶段通过动态语义属性统一着色，并始终保留符号与文字状态。
- GUI 的 Stage 0 预览由独立线程读取冻结输入：Light 使用首个分组样本，FITS/FTS 母版直接读取，产品任务只使用已验证 checkpoint，不再根据目录修改时间挑“最新 FITS”。Pipeline 在每次 `_record_stage` 后输出兼容的 `[PIPELINE_STAGE_RESULT] stage=... status=... duration=... title=...` 和 JSON `[PIPELINE_STAGE_DETAIL]`；后者包含 `execution / fallback_used / upstream_passthrough / reason_code / details / components`，Pipeline worker 通过 `stage_detail(stage, payload)` 交给 GUI。状态不得从 `message` 的 `skipped/fallback` 文字推导。`ok/degraded` 的最终验收产物可读时才原子更新 `process/ui_preview/latest.png` 并输出 `[PIPELINE_PREVIEW]` JSON。Stage 0 和 Stage 1-6 对预览 PNG 使用链接屏幕拉伸，全通道共用黑/白参考与亮度增益；该变换不回写 FITS、不参与质量门。Stage 7 以后不附加额外拉伸。跳过/失败/预览生成失败不替换现有图像，也不改变阶段科学结果或完成状态。GUI 只显示本阶段耗时和总耗时，不计算百分比或 ETA；`CompletedWithWarning` 显示黄色复核卡片并链接 `process/final_quality_report.json`。

## Env

非敏感 runtime env 来源优先级：进程环境 > 工作目录覆盖 > runtime home 覆盖 > bundled/project 配置。GUI worker 会移除已停用功能的全部变量和凭据，只注入产品处理参数、`SEESTAR_DEBUG_MODE`、`SEESTAR_INPUT_MODE` 和 Siril runtime 路径类变量。

完整 pipeline env 解释以 `pipeline/seestar_Superimpose_workflow.md` 为准；本节只保留 runtime、打包和 GUI 注入相关项。

| Variable | Default | Scope |
|---|---:|---|
| `SEESTAR_DEBUG_MODE` | GUI toggle | GUI 强制写入；控制是否保留 `process/stage*.fit` |
| `SEESTAR_INPUT_MODE` | `auto` | GUI 根据已验证任务自动注入；`stage1_prepared_resume`、`stage2_corrected_resume`、`result_linear_resume` 分别从 Stage 2、3、6 继续，不提供手动 UI |
| `SEESTAR_STAGE2_CENTER_PROTECT_AREA_RATIO` | `0.70` | Stage 2 自动裁切必须完整保留的原图中心保护区面积比例；运行时限幅为 `0.50–0.95` |
| `SEESTAR_FORCE_REVIEW_ONLY_OUTPUT` | `0` | 设为 `1` 时仅导出 `result_review*`；正式交付保持默认 `0`，质量门仍可独立要求复核输出 |
| `SEESTAR_NETWORK_MODE` | `0` from GUI | 出站网络总闸；只有显式设为 `1` 才允许在线 Gaia SPCC/PCC 请求 |
| `SIRIL_PYTHON_CLI` / `SEESTAR_SIRIL_PYTHON_CLI` | bundled Python | GUI 注入稳定 Siril Python；wrapper 用后者兜底 |
| `SEESTAR_SIRIL_PLUGIN_DIR` | bundled/runtime plugin dir | GUI 指向打包或复制后的插件缓存 |
| `SEESTAR_OFFLINE_RESOURCE_ROOT` | unset | Core App 的离线资源包根目录覆盖；可指向包含 `siril_plugins/` 的目录或插件目录本身 |
| `SEESTAR_SYQON_MODEL_DIR` / `SEESTAR_COSMIC_CLARITY_MODEL_DIR` | bundled cache | Worker 强制指向只读模型目录，避免 runtime 模型副本 |
| `SEESTAR_GRAXPERT_OBJECT_MODEL_PATH` | unset | 用户提供的 GraXpert 对象反卷积 `model.onnx`、语义版本目录或模型家族目录；解析后只读链接到隔离 HOME，不触发联网下载 |
| `SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE` | `1` | 是否在 Stage 5 优先尝试 Seestar 随包或本机 GraXpert 应用模型；GUI“仅 Siril RL”会写入 `0` |
| `SEESTAR_GRAXPERT_GPU` | `1` from GUI | `1` 允许 GraXpert/ONNX Runtime 自动选择可用硬件 provider 并在失败时回退 CPU；GUI“CPU 兼容模式”写入 `0` 并传递 `-nogpu` |
| `SEESTAR_SIRILPY_TIMEOUT_SEC` | `120` from GUI | sirilpy/plugin subprocess timeout |
| `SEESTAR_BOOTSTRAP_TIMEOUT_SEC` | `300` from GUI | pyscript bootstrap base timeout; GUI adds 120 seconds per GiB of top-level FITS input and clamps the result to 60–3600 seconds |
| `SEESTAR_WATCHDOG_IDLE_TIMEOUT_SEC` | `900` from GUI | no-output watchdog for ordinary runtime; configurable from 60–7200 seconds |
| `SEESTAR_EXPORT_TAIL_TIMEOUT_SEC` | `120` from GUI | no-output watchdog after a newly generated PNG is confirmed; clamped to 60–120 seconds |
| `SEESTAR_TEMP_CLEANUP_TIMEOUT_SEC` | `30` from GUI | maximum foreground wait for deleting the temporary embedded runtime; cleanup continues on a daemon thread after timeout |

## Pipeline Contracts

Pipeline 细节以 `pipeline/seestar_Superimpose_workflow.md` 为准；这里仅保留集成层必须稳定的契约。

- Stage states: `ok`、`degraded`、`failed`、`skipped`；summary 可显示 `ok_with_fallback`、`ok_safe_passthrough`、`ok_skipped_optional`。`fallback_used=true` 只代表本阶段实际选中回退，映射为兼容 `degraded`；`execution=safe_passthrough` 是中性安全旁路；`upstream_passthrough=true` 只描述输入来源，不能把当前阶段改成 fallback。全局 `partial_success` 只读取原始 `degraded` 或结构化 `fallback_used`。
- 输入必须通过 `InputProfile` 解析为 `linear / nonlinear / unknown`；文件名和扩展名不能单独证明线性状态。只有无冲突的线性证据或与 `pipeline-result.json` 中 checkpoint SHA-256 匹配的可信续跑来源才能进入 Stage 3-9；其余输入只允许安全裁切和 Stage 10 复核导出。
- `processing-plan.json` 必须在破坏性变换前原子冻结到工作目录，并镜像到 `process/`；内容包含输入 SHA-256、输入状态、冻结主目标、通道语义、阶段动作、候选合约、脱敏配置及规范化哈希。`pipeline-result.json` 必须原子记录计划哈希、阶段/产物结果与 `success / partial_success / review_required / failed`，并输出稳定的 `[PIPELINE_RESULT]` 行供 GUI 解析。
- 产品阶段与文件名以 `stage_contracts.py` 为单一来源：Stage 1-6 属于线性域，Stage 7-10 属于非线性域；跨运行正式续跑点仅允许 Stage 1、2、5。`stageN_*` 只表示阶段产物，`result_*` 只表示交付产物；旧名称仅可作为只读兼容输入。`task_plan.py` 提供 GUI 与 pipeline 共用的确定性路由、累计配置指纹和防篡改校验；旧 GUI 模式在分阶段迁移完成前仍由现有 runtime 兼容层承接。
- 输入动作以 `input_discovery.py` 的结论为准：明确拖入 FITS/FTS/XISF 是外部母版并从 Stage 1 导入；目录只可识别为递归 Light 集、带签名清单的产品任务或需要只读迁移的旧目录，禁止从“最新文件”或文件名推断阶段。旧目录必须同时验证 `processing-plan.json` 的哈希、`pipeline-result.json` 的哈希/计划引用、线性 checkpoint 状态与文件 SHA-256；迁移源作为外部母版从 Stage 1 重新校验，不把旧 Stage 5 冒充为当前契约断点。Light 按目标、滤镜、相机和几何尺寸分组，Dark/Flat/Bias 首版排除。`task_workspace.py` 对源文件只记录绝对引用、大小与 SHA-256，不复制或改写来源；GUI 通过可取消后台 worker 计算大文件/多 Light 指纹。同一内容指纹复用相邻 `SeestarSuperimpose/<task-id>/`，每个正式断点同时核验任务清单、阶段契约、配置累计指纹和产物 SHA-256。成功或需复核的 run 只有在 `pipeline-result.json` 及其交付哈希再验证后才更新 `results/latest-result.json`；保留策略仅清理已签名旧 run 中仍匹配清单的旧交付/中间目录，并保留当前 run 质量报告、日志、正式断点和未登记文件。
- GUI 历史索引由 `gui/history_store.py` 原子写入 `~/Library/Application Support/SeestarSuperimpose/history-index.json`，只登记功能上线后实际开始的 run，不扫描或补录旧任务。索引是可恢复的导航状态，不是信任根：历史详情重新验签 task/run/result 清单，交付文件重新核对 SHA-256；删除前还必须确认目标是非符号链接、父目录名严格为 `SeestarSuperimpose`、任务清单有效且 `task_id` 等于目录名。删除仅调用 Qt 废纸篓接口移动该任务根，不能接受容器路径或单次 run 路径，索引只在移动成功后移除。
- 每轮只能有一个冻结的 primary target 驱动 policy；Stage 4 的新观测和 secondary labels 只能增加保护信号，不能在处理中途改写主路由。
- Stage 3 的生产授权依据必须是过程证据：输入线性状态、显式 source/coverage mask、跨视场真实天空样点、冻结留出集、留出背景/RMS 及目标保真；`gradient_score / dirty_background_score` 仅供旧调用方兼容与候选排序，不得再决定生产任务的 `preserve / apply / review_required`。加性梯度走减法，径向暗角/平场误差走 master-flat 复核，条带/walking noise 走标定或重叠加复核，真实天空不足时保留基线。
- Stage 3 的 Siril 内置主候选必须使用代码筛选且通过源掩膜、覆盖掩膜和空间覆盖审计的自定义样点；规则网格先冻结安全阈值，只有非弥散且 source/sky mask 支持充分时才允许确定性暗斑搜索补充候选，增强点不得重估阈值，并与规则点统一执行去重、全局最小间距和空间覆盖门。弥散/暗弱目标或真实天空不足时增强必须 fail closed。拟合集与留出集必须分离。样点通过 `set_image_bgsamples(..., recalculate=True)` 安装，并核对 Siril 重算后的实际坐标、最低数量和空间覆盖；Siril 可丢弃个别无法重算的点，但不得引入未审计坐标或导致覆盖坍缩。所有 `subsky` 命令必须带 `-existing`，禁止回到 Siril 自动样点。候选执行是事务性的，插件和 AutoBGE 只能作为受留出背景/RMS与目标通量/形态门约束的回退。
- 通道语义固定为 `broadband / narrowband / mono / nonlinear / unknown`；全局饱和度和颜色变换只允许宽带路径使用，未知或非线性输入不能推断为宽带。
- Stage 6 必须显式记录 `accepted / target_bypass / rejected / tool_failed`。passthrough 不能命名或解释为 Starless；拒绝/失败后 Stage 8/9 走含星复核路径，Stage 10 只写 `result_review*`。
- 在线 SPCC/PCC 受 `SEESTAR_NETWORK_MODE` 总闸控制，默认离线。
- Worker 必须复制外部 `pipeline/stages/`、`pipeline/configs/` 和 Stage 1-10 共享 helper 模块，并明确排除停用实验模块；不再内嵌 pipeline 源码字符串。`SEESTAR_TASK_RUN_MANIFEST` 与 `SEESTAR_RESUME_CHECKPOINT_PATH` 必须进入 runtime 环境变量白名单并原样传到 Siril Python 进程，否则新任务会错误退回旧目录扫描。缺少 `configs/` 会让目标目录和非通用 policy 在临时运行时静默回落，必须由 runtime 测试覆盖。
- 停用实验模块只保留源码和隔离单元测试：正式 `run()` 固定结束于 Stage 10，GUI runtime 不复制模块、不接受事件、不显示或记录相关内容。
- `stage2_corrected_resume` 要求 `<work_dir>/stage2_corrected.fit` 或 `<work_dir>/process/stage2_corrected.fit`，将其作为已完成裁切/视场修正的叠加后中间结果，并从 Stage 3 继续完整后处理。
- `result_linear_resume` 要求 `<work_dir>/result_linear.fit`，记录 stage 2-5 为 skipped，并从 Stage 6 继续。
- Stage 4 必须先保存不可变的 `stage4_pre_pcc.fit`。宽带与确认 Ha/OIII 映射的双窄带都先通过独立 Siril CLI 执行一次 Gaia DR3 SPCC；本地优先、在线受总闸控制、默认 180 秒且不重试。候选解不精确、命令失败或质量门拒绝后必须回载检查点；只有宽带允许再执行一次 PCC 异常回退，禁止从 SPCC 候选串联到 PCC。
- 双窄带接受的物理结果保存为 `stage4_physical_color.fit`。HOO 归一化只写隔离的 `stage4_hoo_artistic.fit`，报告必须为 `feeds_main_pipeline=false`；保存后重新加载物理母版再写 `stage4_color.fit`。物理 SPCC 不可用时保色并要求复核，不得用宽带 PCC 或 HOO 艺术增益冒充物理校色。
- Stage 4 的本地回退只能作用于星点软掩膜，禁止手工全图白平衡。宽带 SPCC/PCC 均失败时标记 `requires_review`；该状态必须进入 `pipeline-result.json` 的全局 `review_required`，并让 Stage 10 只使用 `result_review*` 正式命名门。单色不执行颜色处理。
- 设备几何在无冲突且置信度达标时可驱动 platesolve，显式环境覆盖始终优先；解算后的 WCS 像素比例与预测值冲突时回滚 `stage3_bgremoved` 并禁止 SPCC/PCC。`stage5_noise_model.json` 同时驱动确定性多尺度亮度/对立色度降噪候选，细节或背景门不通过时回到 `stage5_pre_multiscale.fit` 并沿用原降噪回退链。
- Stage 8 的本地曲线/蒙版引擎使用版本化 recipe、单调曲线和显式 soft mask；配方候选未通过背景、核心、裁剪或 mask 外变化门时，保留配方前候选。
- Stage 8 亮星云 halo 必须执行 `full <= 0.350`、`limited <= 0.600`、`skip > 0.600/硬失败` 三级门；limited 只允许 masked builtin、饱和度 `<=0.05`、背景增益/锐化为零，并保存 `stage8_limited_candidate.fit`。starmask 环带门必须用高分位紧凑支撑排除弥散底座；通用品质门或环带纹理门拒绝后必须回滚 `stage8_input_starless`，且报告保留原始触发 advisory、修复后指标、接受上限和 outcome。
- Stage 9 在插件链之前用不可变线性含星参考修复星核/星翼色度，必须通过色度改善、通量、覆盖、裁剪和回混前复核；失败时恢复原 starmask，不改变既有回混梯级。
- 新任务只写 `stage6_starless*` 与 `stage6_starless_quality.json`；`stage7_starless*`、`stage7_quality.json`、`stage4_colorbalanced.fit`、`stage5_denoised.fit` 仅允许作为旧目录的只读迁移候选，不得作为新输出。拒绝/工具失败时不得创建任何 Starless 命名产物。
- Stage 9 必须在 `stage9_remix_quality.json` 记录 `stars_required / stars_applied / stars_application_mode`；只有合成候选验收且 `stage9_remixed.fit` 保存成功才能记 `stars_applied=true`。
- Stage 9 报告还必须分别记录 `upstream_passthrough` 与 `stage9_fallback_used/reason`；Stage 8 安全旁路 + Stage 9 主 Screen 成功的组合应为 `true / false`，GUI 显示“成功（使用 Stage 8 安全旁路源）”。
- Stage 9 在候选接受、全部拒绝回滚或 starmask fail-closed 且 `stage9_remixed.fit` 保存成功后，都要生成 `review_bundles/stage9_star_remixing/`；复核包生成失败只记录警告，不改变质量门的确定性结论。
- Stage 10 正常导出必须保留 TIFF/PNG/FITS fallback 名称，包括续跑模式的 `_linear` fallbacks；全模板因 `STACKCNT` 等字段不完整而不可用时，应先用 `OBJECT / EXPTIME / DATE-OBS` 生成可识别的安全字面名称，身份信息不足时才使用通用 fallback。若 Stage 4 色彩要求复核、最终质量报告要求保守重跑，或 `stars_required=true` 但 `stars_applied=false`，则只写 `result_review*`，不得覆盖普通 `result_processed/result_final`。
- Stage 10 的受管理导出只新增 `*_display_srgb.png` 与 `*_edit_srgb.tif`，不补写或重写 Siril 原产物；PNG 必须带 sRGB 声明，TIFF 必须内嵌有效 ICC，否则该衍生文件 fail-closed。`managed_output_report.json` 与 `output_color_manifest.json` 分别记录写入和容器复核，FITS 在导出前后以 SHA-256 验证不变。
- Runtime plugin 路径必须保持离线可用：SyQon、SASP、CosmicClarity 资源来自 bundled/local cache；缺失时应作为 preflight/runtime 问题处理，不应要求 pipeline 内联网下载。

## Plugin Cache

- 准备：`bash resources/siril_plugins/download_siril_plugins.sh`；脚本固定按 Siril CPython 3.12 / `cp312` 下载 wheel，并清理 `downloads/` 中同一库的旧版本和不兼容 ABI。早期 `cp37-cp311` 标记的 `abi3` wheel 在 CP312 可用时保留，`cp313` 及更高 ABI 一律拒绝。
- GUI 每轮检查插件脚本、SASP/SyQon/AberrationRemover、PyQt6/PySide6/astropy/scipy/tifffile/onnxruntime 等 wheels，并在安装前拦截非 CP312 兼容 ABI。
- 缺失时自动尝试下载；仍缺失则阻断运行。
- runtime offline pip 只从 bundled wheels 安装，并写入 `tiffile` shim 与 `sitecustomize.py` sirilpy timeout patch。
- `PySide6_Addons` 仍保留在 Full/Core 离线资源中；只有完成真实断网、全新 runtime 的 SyQon GUI/CLI 回归后，才允许改为 Essentials-only。

## Validation Matrix

| Changed area | Required checks |
|---|---|
| Pipeline | `python3 -m py_compile pipeline/seestar_Superimpose.py pipeline/stages/*.py`; 核对 stage 顺序、降级、TIFF/PNG/FITS |
| GUI history / task deletion | `python3 -m py_compile gui/history_store.py gui/main_window.py`; `QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_gui_history_store.py tests/test_gui_history_ui.py tests/test_gui_runtime_modes.py tests/test_gui_ui_system.py tests/test_task_workspace.py`; 核对不补录旧 run、异常终态恢复、历史详情验签，以及任务根/容器/符号链接/清单不匹配删除边界 |
| Structured stage status / Stage 5/8/9 | `python3 -m unittest tests.test_pipeline_plugin_fallbacks tests.test_gui_runtime_modes`; 检查消息关键词不改变状态、Stage 5 子状态、亮星云三级门/受限候选回滚，以及 Stage 9 upstream passthrough 与本地 fallback 分离 |
| Target policy / gates | `python3 -m unittest tests.test_adaptive_pipeline_phase1 tests.test_pipeline_stage_smoke`; 检查 target fallback、policy fallback、dirty background 保护、Stage 7 目标局部指标、raw/clean 星点层和 Stage 9 自适应回星门控 |
| Deterministic B/C/D engines | `python3 -m unittest tests.test_device_geometry tests.test_noise_model tests.test_narrowband_normalization tests.test_star_color_repair tests.test_local_adjustments tests.test_managed_output tests.test_output_color` |
| Disabled experiment boundary | `python3 -m py_compile pipeline/stage11_ai_postprocess.py pipeline/ai_artistic_derivative.py`; 确认正式 runtime 不复制、执行或报告停用模块 |
| GUI/preflight | 缺 Siril/config/pipeline/FITS/result_linear 的错误清晰；Stop 行为仍安全 |
| Dev launcher | `python3 -m unittest tests.test_gui_dev_launcher tests.test_pipeline_dev_launcher tests.test_manual_core_pipeline_smoke`; 检查显式 Siril/seed 校验、临时资源覆盖、无 GUI 核心入口和结果清单校验 |
| Build | `bash -n build/build_macos_app.sh`; app 仍嵌入 Siril、pipeline、config、可选资源 |
| Plugins | cache 完整性、缺失补齐/阻断、offline wheels 安装 |
| Toggles | `Debug`、`AI`、`联网`、处理模式映射到预期 env/Siril flags |

Baseline smoke: embedded Siril + external pipeline 可运行；stage 1-10 能导出 TIFF/PNG/FITS；Debug 保留 `process/stage*.fit`。

人工核心算法快速回归：`.venv/bin/python tests/manual_core_pipeline_smoke.py`，
菜单默认推荐 `stage2_corrected_resume`，网络默认开启；需要验证严格离线回退时
显式传入 `--offline`。脚本不会打包或启动 GUI，并在 launcher
成功后复核本轮结果清单与全部登记产物哈希；它不覆盖 frozen App 的资源布局、
GUI 参数注入、签名和 Finder 启动验证。
