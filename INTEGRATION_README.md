# Starun Integration Notes

内部集成速查。用户说明见 `README.md`；全仓约束见 `AGENTS.md`；pipeline 约束见 `pipeline/AGENTS.md`。

Starun 仅接受 `STARUN_*` runtime 环境变量，且使用 `starun.*` 任务/schema 契约；不得在新运行中注入旧 `SEESTAR_*` 键或复用旧版续跑清单。

## Layout

| Path | Role |
|---|---|
| `pipeline/starun.py` | stage 1-10 主流程编排 |
| `pipeline/stages/` | stage 1-10 分离实现模块 |
| `pipeline/stage_contracts.py` / `pipeline/task_plan.py` | 产品阶段、正式续跑点、产物命名与共享冻结路由契约 |
| `pipeline/input_discovery.py` / `pipeline/task_workspace.py` | 文件/Light/产品任务识别、只读来源指纹、独立任务目录与断点验签 |
| `pipeline/processing_parameters.py` | 当前 v7 参数注册表、历史 v4/v5/v6 兼容迁移、序列化、枚举与安全范围校验 |
| `gui/starun_gui_app.py` | GUI 入口 |
| `gui/starun_gui_dev.py` | 系统 Siril + 现有 seed 的免打包开发入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 GUI、preflight、Siril worker |
| `gui/ui_theme.py` / `gui/ui_platform.py` | 语义设计令牌、浅深色主题、系统字体与平台策略 |
| `build/build_macos_app.sh` | macOS App 打包 |
| `resources/siril_plugins/` | 可选 Siril/SASP 插件缓存 |
| `resources/siril_spcc_database/` | 固定版本、带清单与 SHA-256 的最小 SPCC 传感器/滤镜/白参考种子；不含 Gaia 星表 |
| `resources/config.1.4.ini.template` | Siril 配置模板 |
| `resources/default.env` | 项目默认 runtime env；打包时进入 App Resources |
| `tests/` | GUI/runtime/pipeline、计划验签和保留策略测试；fallback 按 Stage 分文件，共享 fake/helper 不参与收集 |
| `.github/workflows/tests.yml` / `.coveragerc` | 单元测试、65% 总体分支覆盖率门和真实 Siril 自托管 E2E 调度 |

## Build

```bash
cd /Users/mz/dev/aiseestart
python3.13 -m venv build/.venv
build/.venv/bin/python -m pip install --require-hashes \
  -r build/requirements-gui-build.lock
./build/build_macos_app.sh
```

Inputs:
- Required: `packages/siril-1.4.4-arm64-3.dmg`（内置 Siril CPython 3.12 runtime）
- Required Siril module source: `~/Library/Application Support/org.siril.Siril/siril/.python_module`；缺失时先运行 Siril 初始化。用户目录中的 `venv` 不再作为构建输入。
- Build host: Apple Silicon Mac；最终 App 要求 macOS 14.0+，主程序必须是纯 `arm64`。

Notes:
- `build/.venv` 只包含 `build/requirements-gui-build.lock` 锁定的 GUI 冻结依赖。脚本优先使用该环境，也可用 `--build-python` 显式覆盖；无论构建环境是否含重型科学库，PyInstaller 都排除仅由 Siril CP312 runtime 使用的模块，并在嵌入资源前验证 PySide6/numpy/astropy 存在且重型模块没有泄漏。
- PyInstaller 在 `release/.starun_build.*` 临时目录中完成构建；只有资源校验与深度签名全部通过后才替换同名 `.app`，构建中断不会把半成品发布到 `release/`。
- PyInstaller 强制使用 `arm64` 目标；构建后写入并校验 `LSMinimumSystemVersion=14.0`，同时用 `lipo` 拒绝非纯 `arm64` 主程序。
- Bundle 元数据固定为 `CFBundleIdentifier=StarunC`、`CFBundleShortVersionString=0.1`、`CFBundleVersion=1`；GUI runtime 指纹读取为 `0.1 (1)`，版本变化时不复用旧依赖缓存。
- 构建下载依赖时使用 `requirements.lock` 和 `resources/siril_plugins/requirements.lock`，并强制校验 pip-tools 生成的 SHA256。
- seed venv 由 App 内置 Siril Python 3.12 和本地 wheels 在构建时重新生成，不复用用户目录中可能被其他 Python 版本改写的 venv；seed 只保留最小运行依赖，重型插件依赖运行时再从本地 wheels 离线安装。
- 构建脚本校验 `resources/siril_spcc_database/SHA256SUMS` 后嵌入 `SirilSPCCDatabaseSeed/`；Worker 原子同步到隔离 HOME 的 `siril-spcc-database`。构建仍拒绝打包 astrometric 或 `xp_sampled` Gaia 星表。
- `resources/siril_plugins/cosmic_clarity/` 保留 CosmicClarity Native/classic wrapper 共用的最小模型集：`deep_denoise_{mono,color}_AI4.pth` 与 `deep_{sharp_stellar,nonstellar_sharp_conditional_psf}_AI4.pth`。
- `resources/siril_plugins/bin/CosmicClarity` 是 bundled standalone classic wrapper，兼容 Siril classic 插件的 `input/`、`output/` 目录协议。
- 默认 `--bundle-profile full` 生成单体 Full Offline App。`--bundle-profile core` 将完整 `siril_plugins` 输出到相邻的 `<AppName>-OfflineResources/`，App 启动时自动发现；可用 `--offline-resource-pack-dir` 自定义输出位置。
- 可选参数：`--app-name`、`--output-dir`、`--build-python`、`--gui-entry`、`--pipeline-src`、`--config-template`、`--siril-src`、`--codesign-identity`、`--bundle-profile`、`--offline-resource-pack-dir`。
- `--codesign-identity` 默认仍为 `-`（ad-hoc，适合本地验证）。公开升级发行应使用固定 Developer ID Application 身份；完整公证仍需另行配置 hardened runtime、timestamp 和 notarization。

## Runtime

- 开发 launcher 通过临时资源覆盖层把显式 `--siril-app`、
  `--siril-seed` 与项目 `resources/` 组合后注入 `StarunGui`。覆盖层只含符号链接，
  launcher 不改写来源，退出 GUI 后自动清理；正式 `gui/starun_gui_app.py` 与
  frozen App 不受影响。
- GUI 强制 `SIRIL_PYTHON_CLI` 指向 bundled Siril Python，并同步注入 `STARUN_SIRIL_PYTHON_CLI`，供 bundled wrapper 在插件脚本把 `SIRIL_PYTHON_CLI` 改写成布尔值时继续找到稳定 Python。
- GUI 强制 Siril `HOME` 到 `~/Library/Application Support/Starun/runtime_home`。
- runtime venv 位于该 HOME 下的 `Library/Application Support/org.siril.Siril/siril`。
- frozen GUI 只从当前进程所在的 `<实际 App>.app/Contents/Resources` 解析 Siril、模板和 pipeline；显式开发覆盖层同样是资源完整性边界，缺项时不得回退到源码 checkout 或 `release/Starun.app`。开发 launcher 会把源码 pipeline 以只读符号链接放进该覆盖层，以复用同一解析规则。
- 每个 `runs/<run-id>/` 在本地 preflight 前以独占方式创建 `starun_gui_run_<run-id>.log`，并原子创建带清单哈希的 `starun.run-state.v2`；`runtime-capabilities.json` 随后记录实际资源/runtime 路径和探测结果。bootstrap、预检失败与取消都先落终态再关闭日志；终态从已验签 pipeline result 与 worker fatal issue 合并，串行切换时主动清空旧诊断路径，pipeline 启动若拿不到属于当前 run 的已打开日志会 fail-closed。
- 每轮注入 Siril CPU 上限 `floor(logical_cpu * 0.8)`，最小 1。
- 每轮确保 `~/.local/share/siril-scripts/`，校验并同步最小 SPCC 元数据种子。元数据预检与 `spcc_list` 结果只记能力证据；若运行时仍不能确定传感器或滤镜，SPCC 以 `command_preparation_failed` 结束并从不可变基线转 PCC，绝不复用 Siril GUI 历史状态。该种子覆盖 Seestar S30/S30 Pro/S50 与 DWARF 2/3/mini 所需的官方 Siril 传感器响应、UV/IR/Seestar LP/DWARF mini Astro 滤镜和白参考曲线，不含 Gaia XP 光谱。
- 完全离线的 platesolve/PCC 使用 runtime 下大小精确匹配受支持版本的 `siril_cat_healpix8_astro.dat`；离线 SPCC 使用独立的 `siril_cat1_healpix8_xpsamp/`，预检要求编号 `0–47` 的 48 个有效分块全部存在。Worker 将两条路径写入临时 Siril config，并分别注入 `STARUN_GAIA_ASTRO_CATALOG` / `STARUN_GAIA_PHOTO_CATALOG`。GUI 下载器目前只安装约 1.52 GB 的 astrometric/PCC 目录；项目、App 与离线资源包均不得携带任一 Gaia 目录。显式离线时有 astrometric、无 XP 走 PCC-only；无 astrometric 时按 `stage4_offline_fallback_mode` 路由到 `auto_local_reference` 或 `preserve_input`。在线模式下端点探测结果只作 advisory，Stage 4 仍执行真实命令尝试。Siril、配置模板或流水线资源缺失仍然阻断。
- Worker 运行时复制 Stage 1-10 pipeline、`pipeline/configs/` 目标目录/策略和小型插件脚本到临时目录；`downloads`、`syqon_starless`、`cosmic_clarity` 使用指向 App/离线资源包的只读符号链接，不再产生约 1.7 GB 的临时副本。
- Worker 同步复制 `pipeline/stages/` 和共享 helper 模块；阶段实现不应依赖仓库外相对路径。
- GUI worker 将每轮 Siril 放入独立进程组：默认连续 900 秒无输出时记录最后命令、进程状态和本轮产物后终止整组；检测到本轮新 PNG 后，若连续 120 秒无输出且仍未退出，则按“导出成功、收尾异常”结束。GUI/任务日志将 Siril 与插件逐百分比输出压缩为约 10% 一档，并把重复的 ICCProfile HDU 跳过提示、Stage 5 NL-Bayes 已确认会继续执行的 `src fits` 初始化诊断各归并为一条 INFO；阶段变化、进度重启、完成行和其他非进度输出仍完整保留。原始输出仍逐行参与 watchdog 活跃度判断，已归并的无害诊断不会挤占崩溃诊断的最近有效输出缓冲区。
- GUI 先在主线程完成资源布局、模板/流水线语法与完整性、Gaia 本地目录、工作目录和输入的快速 preflight；再由 `BootstrapWorker` 实际执行 Siril `--version`、探测网络端点并分别预估系统运行时卷与工作目录卷，同卷时合并需求判断。最终 `runtime-capabilities.json.decisions.stage4_color_calibration` 使用 `starun.stage4-color-capability-decision.v2` 和 `attempt_policy=attempt_then_fallback` 冻结来源与 advisory 能力证据，并通过任务专用 `STARUN_RUNTIME_CAPABILITIES_MANIFEST` 传给流水线；读取端兼容 v1。在线预检不可达不得预先关闭 SPCC/PCC 或开启熔断，只有真实 SPCC 超时可写应用会话缓存。工作目录预估以输入 FITS 实际大小为基准，按完整叠加流程 `44` 份、Stage 2 续跑 `40` 份、线性续跑 `28` 份的阶段/临时产物峰值计算，Light 模式另加预处理 sequence 预算，最后增加 `max(1 GiB, 15%)` 余量。检查通过后才执行 runtime seed、插件检查和离线依赖准备。SyQon/CosmicClarity 模型通过环境变量直接指向 App/离线资源包，并清理与 bundled 文件同尺寸的旧受管副本。准备阶段的“停止”会终止当前下载/安装子进程组。runtime venv 内的 `.starun_runtime_ready.json` 使用依赖锁 SHA-256、Siril Python ABI 和 App 版本作为指纹，命中时跳过重复 `pip install`。
- GUI 使用稳定工具栏、任务输入/预览工作区、两阶段概览和固定状态栏作为主框架，并以显式 `empty / task / run` 状态切换内容区。首次启动或保存输入失效时进入拖放空状态；有效输入直接进入任务设置并重新检测。输入可为明确母版文件、递归 Light 目录或产品任务目录；主界面只显示由契约生成的唯一 Stage 1-10 计划，不再显示手动阶段选择器。快速 preflight 通过后立即进入只读运行视图，完成、失败和停止也保持该视图，只有显式“返回任务设置”才退出。窗口默认 `1280×800`、最小 `980×680`；主窗口以防抖方式分别保存正常矩形和最大化状态，启动时不恢复最小化/全屏幕，显示器移除后按当前 `availableGeometry` 校正。运行页使用可调宽的任务摘要侧边栏、主预览和检查器三栏：侧边栏保持一至两行的轻量任务信息，完整 Stage 2–9 冻结配置移入检查器“任务”页，阶段状态位于“阶段”页；两侧栏只响应用户的工具栏/显示菜单命令，不再因窗口低于 1100 px 自动消失。历史记录不再占用第四个主工作区状态，而由可恢复位置/尺寸的非模态单实例辅助窗口承载；搜索和筛选保留在该窗口内，处理期间允许只读浏览，但打开详情与删除锁定。文件/目录拖放、最近输入和 `QSettings` 持久化保留；窗口位置/尺寸、Task/Run 分隔宽度、两侧栏显示状态、检查器页、高级设置、全宽参数 sheet、专家参数可见性与日志展开状态会恢复。`PreferencesRole` / `Command-,` 打开相对主窗口放置、固定尺寸、非模态、单实例且不参与启动恢复的应用设置窗口，只维护联网、保留中间文件、完成后断点收敛、输出、正式/复核和计算设备等持久默认值；任务运行时窗口可查看但控件锁定。工具栏“任务选项”与“处理参数…”保留当前任务语义，通用配置之后始终显示全任务“默认 / 放松 3× / 无限 10×”门禁档位，Stage 2–9 再以下方单开手风琴显示；无限档位只在本任务内强制复核，不改写持久输出用途。建议参数始终可见，专家参数按执行策略、算法参数、过程门禁、质量验收、回退与失败分组；显式登记的数值门禁在跟随档位时只读显示派生有效值，取消跟随后才恢复原专家范围编辑，全局专家按钮只控制可见性且不清空值。档位与阶段字段只写入当前任务内存状态和签名 `run-manifest.json`，不进入 `QSettings`；切换输入时恢复默认档位并清空阶段覆盖，同一批次共享冻结快照，重新运行从验签清单恢复。恢复点会禁用已经完成的阶段。每项同时设置 tooltip 与 accessibility description，未知字段在 preflight 拒绝，专家合法越界值在共享注册表中安全限幅并留痕，档位派生值则只按显式物理域截断；新任务只写 v7 当前字段，历史 v4/v5/v6 仅在验签后兼容读取并执行明确迁移，Stage 10 门禁不属于 Stage 2–9 参数契约。保留中间文件与联网仍为快捷开关：前者默认关闭，后者默认开启。旧参数、旧 QSettings 键和已删除入口不会恢复。主控件设置 label buddy、辅助功能名称/说明、状态播报和显式 Tab 顺序。
- 任务页使用嵌套分隔器：输入计划/预览可调宽；展开处理参数后，输入区/参数区可调高。两组尺寸均写入 `QSettings` 并在重新启动或切换工作区后恢复。
- 文件、编辑、显示、处理、窗口和帮助菜单只编排共享 `QAction`；工具栏、运行预览和日志抽屉不再直接连接第二套业务回调。命令对象统一维护标题、图标、快捷键和启用状态，并使用 `PreferencesRole`、`QuitRole`、`AboutRole` 交给 macOS 应用菜单安置。文件菜单区分图像与文件夹，最近输入菜单与任务页列表同源；编辑命令只转发给当前焦点控件，显示菜单统一提供预览比例、缩放、日志和全屏幕，窗口菜单提供最小化、缩放、前置全部窗口，以及主窗口和 `Command-Y` 历史记录窗口入口。关闭、最小化和缩放命令作用于当前活动窗口。
- 视觉系统由 `ui_theme.py` 的浅色/深色语义令牌与 `ui_platform.py` 的平台策略共同生成，不在业务窗口中复制 QSS。macOS 使用系统应用字体、原生菜单、统一标题栏/工具栏、30 px 控件与 Command 快捷键；工具栏弹性空白区透传鼠标给原生统一标题栏作为拖拽/激活区域。主窗口与辅助窗口保留系统标题栏、红黄绿按钮或对应工具窗按钮，不使用 `FramelessWindowHint`。Windows 复用同一 widget tree 和信息架构，改用系统字体、窗口内菜单、32 px 控件与 Control 快捷键。平台分支不得承载参数、preflight、worker 或 pipeline 行为。成功、旁路、降级、失败、停止和当前运行阶段通过动态语义属性统一着色，并始终保留符号与文字状态。
- GUI 的 Stage 0 预览由独立线程读取冻结输入：Light 使用首个分组样本，FITS/FTS 母版直接读取，产品任务只使用已验证 checkpoint，不再根据目录修改时间挑“最新 FITS”。Pipeline 在每次 `_record_stage` 后输出兼容的 `[PIPELINE_STAGE_RESULT] stage=... status=... duration=... title=...` 和 `starun.pipeline-stage-detail.v2` JSON；后者包含 `execution / fallback_used / upstream_passthrough / reason_code / review_reasons / issues / details / components`，Pipeline worker 通过 `stage_detail(stage, payload)` 交给 GUI。状态不得从 `message` 的文字推导。`ok/degraded` 的最终验收产物可读时才原子更新 `process/ui_preview/latest.png` 并输出 `[PIPELINE_PREVIEW]` JSON。Stage 0 和 Stage 1-6 对预览 PNG 使用链接屏幕拉伸，全通道共用黑/白参考与亮度增益；该变换不回写 FITS、不参与质量门。Stage 7 以后不附加额外拉伸。跳过/失败/预览生成失败不替换现有图像，也不改变阶段科学结果或完成状态。GUI 只显示本阶段耗时和总耗时，不计算百分比或 ETA；终态按“降级完成 / 已使用回退 / 需要复核 / 失败”分别显示并可链接 `process/final_quality_report.json`。
- 选择单个 FITS/FTS 时，`pipeline/input_discovery.py` 以最多 128 个 FITS Header block 的固定上限同步读取主 Header，不读取像素数据；GUI 仅展示归一化后的设备、滤镜、曝光和常用拍摄/几何字段，不展示观察者或位置字段。Header 缺失、损坏以及当前不解析的 XISF/复核图像只显示非阻断状态，实际输入验收和 Stage 1 诊断仍按原契约执行。
- 新运行写入 `starun.processing-parameters.v7`；历史 `starun.processing-parameters.v4`、`starun.processing-parameters.v5`、`starun.processing-parameters.v6` 在验签后兼容读取。v6 显式 Stage 7 主体增色开关迁移到 `stage8_target_aware_chroma_enabled`，旧 Stage 4 HOO 像素字段和旧 Stage 3 经验门字段被移除并写入迁移审计；v7 继续携带这些退役字段、v1–v3、扁平载荷、未知 schema/字段均 fail-closed。共享注册表定义 Stage 1–10 与通用配置的 UI 级别、专家分组、依赖、范围、枚举、门禁方向和安全限幅；`checkpoint_mode=false` 为默认，`true` 时只有 Stage 10、v2 plan/result 引用、最终交付 SHA-256 和正式断点全部验证后才收敛中间文件，任何失败都保留现场。运行时优先级固定为“内置/当前环境默认 → Stage 1 分析与可选自动调参 → 签名门禁档位 → 签名专家覆盖 → 安全限幅 → Stage 执行”；安全钳制审计字段由钳制注册表派生。不可变 `run-manifest.json` 和唯一 `processing-plan.json` 保存原始选择、最终 `PipelineConfig`、门禁派生/限幅和失败策略，阶段详情与最终来源进入 `pipeline-result.json`。
- 累计指纹按产物影响划分：Stage 1 配准阈值变化使 Stage 1/2/5 断点失效；`auto_tune_enabled` 变化最多复用 Stage 1；`compute_mode` 变化最多复用 Stage 2，并使 Stage 5 断点失效。Stage 2 参数仍计入 Stage 2/5，Stage 3–5 只计入 Stage 5；重试、复核包、输出、Stage 9/10 参数不使早期线性断点失效。
- `stageN_failure_action` 在 Stage 2–10 仅允许 `auto_fallback / preserve_review / stop`。后两者在决定性失败后停止候选搜索并先写诊断：`preserve_review` 回载不可变阶段输入、生成规范旁路产物并强制复核；Stage 10 只允许回载已验证 Stage 9 含星源。`stop` 记录 failed 后由阶段包装层终止任务且 Stage 10 不生成最终导出。任何策略均不得强制接受硬门失败。Stage 8 用户模式是上游路由的强度上限；Stage 9/10 的 preserve 及最终兜底只接受实际存在、可加载的含星来源，无来源时设置 withheld 并失败。

## Env

runtime env 来源优先级：进程环境 > 工作目录覆盖 > runtime home 覆盖 > bundled/project 配置。GUI worker 仅允许当前产品处理参数、`STARUN_DEBUG_MODE`、验签后的 `STARUN_INPUT_MODE` 和 Siril runtime 路径类变量；旧 AI、凭据和 CLI 兼容变量不在允许列表中。

完整 pipeline env 解释以 `pipeline/starun_workflow.md` 为准；本节只保留 runtime、打包和 GUI 注入相关项。

| Variable | Default | Scope |
|---|---:|---|
| `STARUN_DEBUG_MODE` | GUI toggle | GUI 强制写入；控制是否保留 `process/stage*.fit` |
| `STARUN_INPUT_MODE` | `auto` | GUI 根据已验证任务自动注入；`stage1_prepared_resume`、`stage2_corrected_resume`、`stage5_linear_resume` 分别从 Stage 2、3、6 继续，不提供手动 UI |
| `STARUN_STAGE2_CENTER_PROTECT_AREA_RATIO` | `0.70` | Stage 2 自动裁切必须完整保留的原图中心保护区面积比例；运行时限幅为 `0.50–0.95` |
| `STARUN_FORCE_REVIEW_ONLY_OUTPUT` | `0` | 设为 `1` 时仅导出 `result_review*`；正式交付保持默认 `0`，质量门仍可独立要求复核输出 |
| `STARUN_NETWORK_MODE` | `1` from GUI | 出站网络总闸；默认允许在线 Gaia SPCC/PCC。显式设为 `0` 时只使用本地 Gaia；目录不足时按 Stage 4 离线回退配置继续，不允许任何联网路径；随包 SyQon 模型不检查更新 |
| `STARUN_STAGE4_SPCC_ONLINE_UNVERIFIED_TIMEOUT_SEC` | `300` | 在线 XP 尚未完成真实命令验证时的单次 SPCC 上限，安全范围 30–300 秒；实际取它与普通 SPCC 超时的较小值，完整 `localgaia` 不受此附加上限影响 |
| `STARUN_STAGE4_OFFLINE_FALLBACK_MODE` | `auto_local_reference` | 无 astrometric Gaia 时选择非物理自动局部参考或 `preserve` 保色继续；两者均强制复核 |
| `STARUN_STAGE4_AUTO_REFERENCE_GLOBAL_WHITE_ENABLE` | `1` | 自动矩形白参考默认启用；专家设为 `0` 时只允许背景平衡，白参考仍保留影子审计 |
| `STARUN_RUNTIME_CAPABILITIES_MANIFEST` | task-only | GUI 最终能力清单路径；Worker 会清除继承值，只接受当前任务显式覆盖，流水线还会校验 schema、run id 和工作目录边界 |
| `STARUN_PREFLIGHT_GAIA_ASTRO_ENDPOINTS` | VizieR + ESA TAP availability | 仅供 GUI runtime preflight；以空格、逗号或分号分隔的 astrometry 端点覆盖，不注入 pipeline |
| `STARUN_PREFLIGHT_GAIA_XP_ENDPOINTS` | Zenodo 17988559 XP chunk 0 | 仅供 GUI runtime preflight；以空格、逗号或分号分隔的 XP/SPCC 端点覆盖，不注入 pipeline |
| `SIRIL_PYTHON_CLI` / `STARUN_SIRIL_PYTHON_CLI` | bundled Python | GUI 注入稳定 Siril Python；wrapper 用后者兜底 |
| `STARUN_SIRIL_PLUGIN_DIR` | bundled/runtime plugin dir | GUI 指向打包或复制后的插件缓存 |
| `STARUN_OFFLINE_RESOURCE_ROOT` | unset | Core App 的离线资源覆盖；App 兼容资源包根或插件目录，真实 E2E CI 只接受绝对资源包根且其下必须包含 `siril_plugins/` |
| `STARUN_SYQON_MODEL_DIR` / `STARUN_COSMIC_CLARITY_MODEL_DIR` | bundled cache | Worker 强制指向只读模型目录，避免 runtime 模型副本；SyQon 还会校验同目录 `zenith.pt.sha256`，缺失/损坏时不启动下载 |
| `STARUN_GRAXPERT_OBJECT_MODEL_PATH` | unset | 用户提供的 GraXpert 对象反卷积 `model.onnx`、语义版本目录或模型家族目录；解析后只读链接到隔离 HOME，不触发联网下载 |
| `STARUN_STAGE5_GRAXPERT_DECONV_ENABLE` | `1` | 是否在 Stage 5 优先尝试 Starun 随包或本机 GraXpert 应用模型；GUI“仅 Siril RL”会写入 `0` |
| `STARUN_GRAXPERT_GPU` | `1` from GUI | `1` 允许 GraXpert/ONNX Runtime 自动选择可用硬件 provider 并在失败时回退 CPU；GUI“CPU 兼容模式”写入 `0` 并传递 `-nogpu` |
| `STARUN_SIRILPY_TIMEOUT_SEC` | `300` from GUI | sirilpy/plugin subprocess timeout；Stage 6 SyQon 使用纯 FITS 文件交换，不依赖该回写超时 |
| `STARUN_BOOTSTRAP_TIMEOUT_SEC` | `300` from GUI | pyscript bootstrap base timeout; GUI adds 120 seconds per GiB of top-level FITS input and clamps the result to 60–3600 seconds |
| `STARUN_WATCHDOG_IDLE_TIMEOUT_SEC` | `900` from GUI | no-output watchdog for ordinary runtime; configurable from 60–7200 seconds |
| `STARUN_EXPORT_TAIL_TIMEOUT_SEC` | `120` from GUI | no-output watchdog after a newly generated PNG is confirmed; clamped to 60–120 seconds |
| `STARUN_TEMP_CLEANUP_TIMEOUT_SEC` | `30` from GUI | maximum foreground wait for deleting the temporary embedded runtime; cleanup continues on a daemon thread after timeout |

## Pipeline Contracts

Pipeline 细节以 `pipeline/starun_workflow.md` 为准；这里仅保留集成层必须稳定的契约。

- Stage states 只允许 `ok / degraded / failed / skipped`。`fallback_used=true` 只代表本阶段实际选中回退，并强制规范化为 `degraded`；`execution=safe_passthrough` 是中性安全旁路，`upstream_passthrough=true` 只描述输入来源。review 与 degraded 正交，可选步骤正常跳过不降级。`pipeline-result.v2`、`pipeline-stage-detail.v2` 和 `run-state.v2` 共用 `had_errors / had_fatal_errors / had_degradations / had_fallbacks / review_required`；终态优先级固定为 failed → review_required → partial_success → success，所有判断只读取结构化字段。
- 输入必须通过 `InputProfile` 解析为 `linear / nonlinear / unknown`；文件名和扩展名不能单独证明线性状态。只有当前外部母版从 Stage 1 新建任务，或已验签 task-run manifest 明确引用的 Stage 1/2/5 正式断点，才能进入线性处理；非线性/未知输入只允许安全裁切和 Stage 10 复核导出。
- 工作目录只写一个 `starun.processing-plan.v2` `processing-plan.json`：runtime 在解析输入、目标和通道后调用共享 `task_plan.build_processing_plan()`，立即验签，再于破坏性变换前原子写入。`planned_steps` 只含规范动作，其他运行信息位于 `metadata`。发布 `pipeline-result.json`、最新结果索引或执行保留策略前，都必须再次验证 plan、计划哈希和 result→plan 引用。
- 产品阶段与文件名以 `stage_contracts.py` 的 `starun.pipeline-stage-contract.v2 / 2.0.0` 为单一来源：只登记 Stage 1–10，跨运行正式续跑点仅允许 Stage 1、2、5。GUI 摘要与 runtime 都调用 `task_plan.build_stage_steps()`；不保留另一套路由、旧阶段名或旧产物别名。
- 正式 Stage 1/2/5 checkpoint 必须由已验签 task-run manifest 引用，记录准确的 `run_manifest_hash`、阶段契约、累计配置指纹、线性状态和产物 SHA-256。Stage 2 的 `resume-semantics.v2` 冻结原始/最终尺寸、累计裁边、0–2 次场旋裁切、最终残余结论及 Stage 2 review；第一方轮廓为 `full_frame_is_valid` 且场旋无显著边缘异常时形成保留全画幅共识，旧检测器不得运行。Stage 1 会独立保留已验签、只读的原始来源文件身份，避免后续工作文件或断点路径覆盖设备文件名证据。S30 Pro 的 `S30 Pro_*` 头/文件名、每轴至少 95% 原生尺寸、NAXIS 一致和有效 WCS 或 RA/DEC+焦距+像元物理视场可共同形成全幅旁证；仅边缘相连但缺少角楔几何的异常在原生轮廓同时确认全幅时只记 advisory。通过角楔门的场旋候选不受该豁免，证据不足也继续复核，任何冲突均禁止自动裁切。旧检测器仅作为不可用/错误/结论不充分的降级路径，侵入中心保护区的候选以 `crop_detector_conflict` 整体拒绝。Stage 5 冻结通道、目标、policy、校色和 Stage 1–5 结构化 review。旧 Stage 5 v1 上游布尔值只在验签后按原字段映射到 Stage 2/3/4 并标记 `legacy_inferred`。`native_crop_v5` 会使旧 v4 Stage 2/5 累计指纹失效并回退到仍兼容的 Stage 1。任一字段缺失、清单不匹配、越界路径或哈希不符都拒绝续跑。
- Stage 5 只有在规范 `stage4_color.fit` 已加载并逐文件冻结 `stage5_input_linear.fit` 后，才允许发布 `starun.stage5-stage6-handoff.v1`。`stage5_stage6_handoff.json` 绑定当前 `run_id`、规范 `stage5_linear.fit` 文件名/SHA-256、线性状态、Stage 5 状态、反卷积/降噪完整性及输入基线；Stage 5 checkpoint 恢复还必须绑定已验签 manifest、累计配置指纹和语义上下文，并逐字节物化到当前 run。Stage 6 会重新验签 handoff 与实际字节，只接受当前 run 的 `stage5_linear.fit`，不扫描 Stage 1–4、Stage 5 内部候选、`working.fit`、旧拉伸名或目录同名旧文件。
- 输入动作以 `input_discovery.py` 为准：明确 FITS/FTS/XISF 从 Stage 1 导入，目录只可识别为递归 Light 集或当前契约的已签名产品任务。旧处理目录、根目录 checkpoint、result-only 目录和“最新文件”推断均明确拒绝；旧任务必须重新导入原始母版。`task_workspace.py` 对来源只记录绝对引用、大小与 SHA-256；成功或需复核的 run 只有在 v2 plan/result 与交付哈希复验后才更新 `results/latest-result.json`。
- 新建 GUI 队列按“来源指纹＋完整规范处理参数哈希”去除完全重复项，并在 `PreparedTaskQueue.skipped_duplicates` 留痕；参数不同或用户手动重新运行不去重。最终输出元数据必须拒绝未解析的 `$...$`、printf 模板、unknown/null/n/a、非法日期及非正/非有限曝光值；无有效 FITS 元数据时按高置信目标身份、同置信目标猜测、源文件/任务名、通用名依次生成安全文件名，模板字符不得进入交付名。
- GUI 历史索引由 `gui/history_store.py` 原子写入 `~/Library/Application Support/Starun/history-index.json`，只登记功能上线后实际开始的 run，不扫描或补录旧任务。索引是可恢复的导航状态，不是信任根：历史详情重新验签 task/run/result 清单，交付文件重新核对 SHA-256；删除前还必须确认目标是非符号链接、父目录名严格为 `Starun`、任务清单有效且 `task_id` 等于目录名。删除仅调用 Qt 废纸篓接口移动该任务根，不能接受容器路径或单次 run 路径，索引只在移动成功后移除。
- 每轮只能有一个冻结的 primary target 驱动 policy；Stage 4 的新观测和 secondary labels 只能增加保护信号，不能在处理中途改写主路由。
- Stage 3 的生产授权依据必须是过程证据：输入线性状态、显式 source/coverage mask、跨视场真实天空样点、冻结留出集、留出背景/RMS 及目标保真；`gradient_score / dirty_background_score` 仍参与候选充分性/排序和后续阶段自适应，但不得决定生产任务的 `preserve / apply / review_required`。方向性加性梯度走减法；径向低频形状只写 master-flat/标定 advisory；条带/walking noise 走标定或重叠加复核，真实天空不足时保留基线。
- Stage 3 普通路线获得执行授权后的默认链为“目标感知内置 Polynomial/RBF → 复合 Polynomial→RBF → GraXpert → ADBE → DBE → AutoDBE”。弥漫/大面积弱信号目标的规则网格不足时，只在线性状态、可用天空、valid/saturation/star catalog 和方向噪声证据全部可信时启用保守暗斑恢复；阈值和遮罩不放宽，验证集必须至少保留 4 个跨 3 象限/4 网格的 `regular_grid` 样本。恢复路线的 effective profile 固定为 `strict`，只执行一次 `subsky 1 -existing`，禁止 RBF、复合和外部插件补跑；RGB 把 Siril per-channel 结果投影为中心锚定的 Rec.709 中性轴一阶修正，丢弃 DC 与通道独立偏移并保持 `R-G/B-G`，单色维持原一阶结果；`graxpert_only` 下恢复不可用并 fail-closed。
- `stage3_gate_profile` 的缺省值是 `output_first`，冻结配置缺少该字段时也使用此值；另有 `balanced` 和 `strict`。v9 `background_quality_report.json` 记录 configured/effective profile、逐样本 provenance、恢复资格/缺口、`starun.stage3-neutral-axis-projection.v1` 与逐分量 `starun.stage3-spatial-opponent-projection.v1`、真实多尺度排除掩膜、统一 `reason_code` 和候选门禁。色度分量只在拟合 3σ、留出相关 `>=0.70`、RMS 改善 `>=15%` 且不反向时，以零 Rec.709 亮度的中心锚定增量写回；失败保留中性检查点并要求复核。完整的候选无关天空 support FITS、SHA、覆盖率、独立拟合/验证采样补丁计数和参考空间平面写入 `stage3_spatial_background_lineage.json`；Stage 5–10 不把稀疏采样补丁误作完整天空掩膜。密集星表路线仍保持 50% 天空基底、80% patch 支持和 `BGSample/recalculate=False` 契约。算法契约为 1.7.0；旧 v6/v7/v8 产物不重写。
- 通道语义固定为 `broadband / narrowband / mono / nonlinear / unknown`；全局饱和度和通用颜色变换只允许宽带路径使用。Ha/OIII 通道角色只能来自 Stage 4 冻结契约，后续阶段不得重新解析 FITS Header。唯一窄带例外是默认开启的 Stage 8 双窄带哈勃艺术分支：必须验证契约 schema、`osc_hoo_rgb`、`R→Ha`、`G/B→OIII` 和置信度，Stage 7 Starless 已验收且 Stage 8 为 `full/ok`；降级 PCC 父本允许映射但继续要求复核，缺失、畸形、未知或非线性输入不能进入该分支。
- Stage 6 必须显式记录 `accepted / target_bypass / rejected / tool_failed / review_required`。Stage 5 来源缺失、lineage 不可信或 `stage6_input.fit` 保存失败分别使用 `stage6_stage5_linear_unavailable / stage6_stage5_lineage_unverified / stage6_input_checkpoint_failed`：此时不调用去星工具、不生成 `stage6_passthrough`，质量报告写 `mode=upstream_source_rejected`，Stage 6 为致命失败并停止 Stage 7–10。只有可信 `stage6_input.fit` 已建立后发生的去星拒绝/工具失败，才保留既有含星 review passthrough；passthrough 不能命名或解释为 Starless。`target_bypass` 下 Stage 7 未验收时，Stage 8 必须按本轮 Stage 7 含星复核源→`stage7_review_with_stars`→Stage 6 passthrough→`stage6_passthrough` 的顺序旁路，禁止加载 `starless/stage6_starless`。SyQon 生产链只允许离线、哈希锁定的 Zenith，并通过显式 FITS 路径交换有限 `float32 0..1` pair。初始 profile 为 `512/64`、FP32；提交前统一归一化源图、Starless 和 starmask，并在排除星点、源图最高 30% 信号和无效像素后，以 16 px 微块、128 px 区域及 worker tile 边界检查相对高频纹理和局部传递残差。只有相对比值、`5σ` 绝对显著性和至少 15% 相连异常覆盖同时超限才记 `TILE_ARTIFACT`；随后只允许一次 `512/128、CPU、FP32` 重试。交换契约升级为 `starun.syqon-pixel-exchange.v3`，逐通道和全局 opponent 指标只作边界诊断；正式色度门使用候选无关的 Stage 3 valid/saturation/catalog 证据及源图主体 ROI，固定要求饱和度 P50/P95、opponent 能量和方向相关性保留率分别不低于 `0.35/0.50/0.40/0.70`，并写 `starun.stage6-subject-chroma-lineage.v1`。`SUBJECT_CHROMA_COLLAPSE` 只允许一次 linked-MTF FP32 重试且优先消费质量重试预算；重试或完整质量门失败、以及 `SUBJECT_CHROMA_LINEAGE_UNVERIFIED` 均清除 pair、恢复含星输入并限制下游复核，不运行色度回填或通用 pixel repair。低动态范围星罩覆盖优先使用可靠像素检测，失效时用冻结 Stage 5 星表的 `5σ` 峰值、`5σ√N` 通量和固定孔径恢复率；两者均不可用时记录 `measurement_unavailable` 并要求复核，不写伪数值零。`stage6_syqon_exchange.json` 保留全部尝试、配置、伪影指标和最终选中尝试；Stage 6 接受后仍写 `starun.stage6-pair-handoff.v1` 冻结 pair。
- 在线 SPCC/PCC 受 `STARUN_NETWORK_MODE` 总闸控制，GUI 默认在线；用户显式关闭联网后才强制本地 Gaia。首个 `online_unverified`、`catalog:gaia` SPCC 超时后，应用会话按 Siril 可执行文件/版本与 XP 端点证据指纹缓存 `operational_timeout_cached`；缓存键必须在 Siril 启动和网络能力预检完成后、pipeline worker 启动前计算。同一能力指纹的后续队列直接跳过在线 SPCC，但继续允许 localgaia、PCC 与离线降级。Siril/端点证据变化会生成新指纹并重新探测；旧的单队列 circuit 仍只约束当前批次。
- `catalog:gaia` SPCC 只对 408/429/500/502/503/504、连接重置、TLS EOF、DNS/临时下载错误以及伴随这些证据的 Siril 异常退出重试一次。重试前必须恢复不可变检查点并删除污染候选，新 CLI 预算不超过 120 秒。完整超时和非网络崩溃不重试；瞬态重试耗尽记录 `online_transient_exhausted`。运行时缓存升级为 `starun.stage4-spcc-operational-cache.v2`，兼容 v1，并区分 `operational_timeout_cached` 与 `operational_transient_failure_cached`。
- Worker 必须复制当前 `pipeline/stages/`、`pipeline/configs/` 和 Stage 1–10 共享 helper 模块，不再内嵌 pipeline 源码字符串。只有 `STARUN_TASK_RUN_MANIFEST` 进入 runtime 环境白名单；断点阶段、路径与 SHA-256 全部从该清单的验签 `resume` 记录解析，缺少已验签 task-run manifest 时任何续跑模式都 fail-closed。
- `stage1_prepared_resume`、`stage2_corrected_resume` 和 `stage5_linear_resume` 只是验签后 runtime 的内部模式；输入路径必须来自同一 task-run manifest 的 Stage 1/2/5 checkpoint 记录，不能读取工作目录根文件或仅凭文件名启动。
- Stage 4 公共入口必须在 platesolve 和 SPCC/PCC 前只调用一次统一窄带解析器，即使 `preserve` 也写出 `stage4_channel_mapping.json`。解析器将大小写不敏感的 `FILTER` 作为权威字段；值可分类时其他字段仅记为 ignored evidence，缺失或完全未知时才回退 `FILTER1/FILTER2/INSFLNAM/FILTERNAME/FILTNAME/FILTNAM`，且不读取 `OBJECT`。同一契约对象必须写入 `channel_profile.narrowband_mapping`、`color_calibration_report.channel_mapping`，并传给通道语义、SPCC 窄带参数和后续 Stage 8/9；Stage 4 不再执行 HOO 艺术像素变换。受控设备为 Seestar S30/S30 Pro/S50 的 `LP/LP_Starless` 与 DWARF 3/mini 的 `Duo-Band/Dual-Band`；裸 `LP/LP_Starless` 为 `0.86 / authoritative_filter_field_hint`，默认通过 `0.85` 门限，可通过提高门限关闭；DWARF 2 没有虚构的内置双窄带画像，但可走通用滤镜文字路径。
- Stage 4 必须先保存不可变的 `stage4_pre_pcc.fit`，路由固定为 `SPCC → PCC → 自动区域校色 → 原样保色`。宽带与冻结契约确认 Ha/OIII 映射的双窄带都优先通过独立 Siril CLI 执行一次 Gaia DR3 SPCC；默认请求在线 Gaia，显式关闭联网时物理命令只使用已验证的本地 Gaia。XP/astrometric 端点与 `spcc_list` 预检均为 advisory，不能代替真实命令；未知传感器或滤镜以 `command_preparation_failed` 转 PCC，不复用 GUI 状态。SPCC 的准备失败、真实超时/异常/非零退出、候选缺失或不可读、尺寸改变、非有限像素、保存或写回验证失败才属于技术失败。发生技术失败后必须逐像素验证检查点或内存基线精确恢复，再从同一基线执行一次最长 180 秒的 PCC；恢复失败时阻断主输出，禁止候选串联。
- SPCC/PCC 命令成功且候选通过技术完整性检查时立即采用。旧 PCC 质量门、亮核检查、增益/裁剪、SPCC 精度告警和双窄带源信号保留指标继续生成，但固定 `routing_effect=advisory_only`，不得修改、回滚成功候选或触发下一候选；`stage4_pcc_quality_gate_enabled` 仅兼容旧配置且已从 GUI 移除。宽带 PCC 仍是物理回退；双窄带 PCC 固定为 `PCC_NARROWBAND_DEGRADED*`，并写 `physical_color.accepted=false`、`degraded_color_correction.applied=true`、`requires_review=true`。Stage 4 只保留通道 mapping confidence 参数；HOO/SHO 等艺术像素均归 Stage 8。
- 只有 `linear + broadband_rgb_osc + auto_local_reference` 会在 SPCC/PCC 都技术失败后从不可变检查点运行 `starun.stage4-auto-local-reference.v2`。背景矩形按低亮度、低纹理和低梯度排序，不使用 RGB 色差，且只削减偏高通道；其余安全背景样点作为独立留出。背景候选合格后，以多尺度滑动矩形按亮度、结构、饱和度和污染指标选择单一白参考区域，区域内对象按空间确定性拆为拟合/留出；拟合统计生成最大 `1.10×` 的全局增益，再做一次背景中和。报告在 `reference_regions.background/white` 保存坐标、评分、阈值、偏移、增益、拟合/留出指标和事务回滚。两种自动方法都固定非物理、降级且强制复核；白参考拒绝时可采用背景平衡，背景也拒绝时精确保色。
- 自动候选拒绝或写入/保存失败后必须验证像素精确恢复 `stage4_pre_pcc`；旧星点软掩膜局部恢复已退出运行路由，不再修改像素。自动区域校色或保色均必须进入 `pipeline-result.json` 的全局 `review_required`，让 Stage 10 只使用 `result_review*` 命名。单色、非线性、未知通道和窄带不执行自动区域校色；双窄带两种物理方法失败后只保色复核。
- Stage 7 在 Stage 4 未接受物理校色时，对冻结 Stage 6 主体/银河信号外的背景执行独立绝对色偏复核门；默认 `background_chroma_load > 0.12` 或可信测量不可用要求复核。该门不修改像素、不执行全局白平衡，并由 Stage 10 与 `pipeline-result.json` 传播；不存在绕过普通命名门的强制交付例外。
- Stage 7 参数模式仍只有 `auto/manual`；任一签名 Asinh/GHS override 把 `auto` 规范化为 `manual`。默认 `stage7_rendition_intent=vivid_safe` 在同一冻结主体/背景 ROI 上测量 preview 与候选的可见度、主体跨度、饱和度中位数/P95、微对比度及亮度噪声，但 Stage 7 只生成亮度/tone 候选，主体低频色度修改统一归 Stage 8。v7 `hard_gate_continuous_quality_v7` 由 hard gate 单独决定正式资格，再比较严格亮核饱和状态、主体亮度下限、封顶亮度目标/效用及连续呈现安全分；advisory 和旧 `risk_score` 不进入正式排序。v6 键和假想选择只写影子审计。全任务 relaxed/unlimited 只缩放 Stage 7 可见度/P50 等呈现门，不缩放裁切、结构、LUT/MTF、背景色度、颜色方向和亮度噪声安全门。
- `starun.stage7-luma-noise-growth.v2` 对自动候选认证并重放完整 tone 链，1.25 门只判定 `actual/expected` 的 `excess_growth`；`raw_growth/expected_growth` 只作诊断。自动候选无法认证参数或摘要时 fail-closed，手动和旧任务明确保留 `legacy_raw_growth`。
- Stage 7 默认 `auto_display90` 仅在 Stage 6 Starless 为 `accepted` 且参数模式为 `auto` 时，在 A/B 外从同一 GUI linked D 合同生成 `stage7_cand_display70/82/90.fit`；默认高档 `0.90`，中/低档为 `-0.08/-0.20`，各自认证 65536 点共享 RGB LUT。`balanced` 只保留 D82，`conservative` 只保留 D70，`vivid_safe` 允许三档共同选优；不再生成 `stage7_cand_vivid_safe.fit`。确认 composite 会从已通过普通门的 Display tone 父级重放并追加 linked Rec.709 luminance gamma，形成独立 `stage7_cand_composite_tone.fit`，且不再享受 composite 专属 P50 放宽。`auto_dual` 保持 A/B 基础语义；`manual + display90_only` 仍拒绝。
- 本轮 Stage 7 可精确重放的 Display、`cand_b` linked-MTF、composite 与背景色度救援发布 `starun.stage7-matched-domain-transfer.v3`，adaptive-quantile 发布 v4。v3/v4 都认证实际 winner、父级参数或 LUT 摘要及有序步骤；背景色度救援的 v3 两步链还绑定 `cand_b` 父变换、救援强度和私有空间权重 NPZ 的容器/数组 SHA，adaptive-quantile 的 v4 冻结 calibration schema、anchors、curve digest、源/胜出候选 SHA 和 chain digest。Stage 9 对原始含星图与 Starless 精确重放同一整链，任一 schema、步骤、参数、私有 mask、SHA 或摘要不一致均 fail-closed；历史 v1/v2 只按兼容边界读取，不能冒充新双门证据。
- Display GUI 参考色度豁免只适用于 `narrowband_composite + SPCC_NARROWBAND accepted` 的自动 Starless 路径。每档完成 LUT 摘要和曲线一致性校验后，以同一 Stage 6 冻结背景 ROI 测量真实 GUI linked D；候选/参考负载比必须 `<=1.05` 且候选绝对负载 `<=0.30`。两个边界不乘共享 advisory 或全任务门禁档倍率；匹配只把线性基线增长失败改为 advisory，其他门禁不变。
- Stage 7 背景色度救援不再固定为 `0.10/0.20/0.35`，而是按当前 `chroma_load` 与有效绝对目标反解最多三档，默认上限 `0.90`，每档都从冻结源重放并重跑完整门禁。正式救援会把最终空间权重保存为 run 内私有 `float32` NPZ，并以文件 SHA、数组 SHA、形状和 dtype 写入 v3 两步重放合同；Stage 9 只接受与 `cand_b` 父链完全一致的 mask。若 `auto_fallback` 耗尽且 `stage7_forced_delivery_enabled=true`，该兼容字段只允许保留最佳失败候选作为诊断：报告必须写 `formal_accepted=false`、`delivery_class=review_only`。Stage 8 从冻结 Stage 6 含星输入生成安全预览，Stage 9 不运行无星重混，Stage 10 只写 `result_review*`；任何 Stage 7 硬画质失败均不能占用正式结果名。
- starmask 弥散残留统一为 `<=0.08` 正常、`0.08–0.16` advisory、`>0.16` hard。advisory 保留已验证 clean 星层，同时把 Stage 8 限为 limited、Stage 9 星强度缩放至最多 `0.70` 并锁定 Stage 10 review-only；边缘带状态和有效硬线必须写入 Stage 6/8/9/10 结构化结果，不能在单个任务中临时提高阈值。
- 设备几何在无冲突且置信度达标时可驱动 platesolve，显式环境覆盖始终优先；解算后的 WCS 像素比例与预测值冲突时回滚 `stage3_bgremoved` 并禁止 SPCC/PCC。Stage 5 降噪必须先冻结 `stage5_pre_denoise.fit` 与内存像素；多尺度、Siril、CosmicClarity 全部从该基线生成候选并通过同一细节/噪声/色噪/裁剪门，拒绝或异常时复核回滚，恢复失败则停止后续候选。
- Stage 8 的本地曲线/蒙版引擎使用版本化 recipe、单调曲线和显式 soft mask；配方候选未通过背景、核心、裁剪或 mask 外变化门时，保留配方前候选。冻结主目标为 open/globular cluster 且 secondary 同时包含 `large_nebulosity + emission_red` 时，仍保持恒星保护主路由，只在排除硬星核和背景后对弥散星云执行一次受限局部对比/饱和度叠加；星核最大变化必须为 0、背景 P95 变化不超过 0.002，读取、配方或质量门失败均原样透传。
- Stage 8 是唯一 Starless 像素责任阶段。基础结构门通过后，显式开启 `STARUN_WORKFLOW_PLUGIN_PROBE` 才按 Revela → 主体 Curves → 唯一颜色路线运行事务链；前两步只保留主体/大星系 disk soft mask 内的 linked Rec.709 luminance 增量，逐像素恢复背景、mask 外与硬亮核。每步独立保存前置基线和候选、重载验签，并记录 `stage8_starless_finish_report.json / starun.stage8-starless-finish.v1`；回滚失败立即限制下游并要求复核。`limited/background_only/skip/preserve`、Stage 7 未验收、未验证外部覆盖或受限 handoff 不探测这些插件。
- Stage 6 验收 pair 后从紧致 starmask 连通域发布 `starun.stage6-star-halo-guard.v1`，报告绑定 run ID、Starless/starmask/guard SHA 和局部径向一致性。有星点支撑却缺失/篡改 guard 时 Stage 8 只能受限复核。Stage 8 所有结构、主体色度、Vectra 与 palette mask 先扣除 soft guard，且每步再执行星关联局部门。`1.0 / 0.75 / 0.50 / 0.25` retained-delta 只属于阶段级结构 seam retry：复用同一已验证结构提案，由 seam 指标计算解析边界上限并选择全门合格的最强档，不重调插件，mask 外逐像素保持输入，全部失败才精确回滚。Starless finish 步骤不套用该 ladder；主体色度和 palette halo retry 保持既有固定 `0.5`，palette 的 `1.00/1.12/1.25/1.40` 保亮度色度闭环是独立的同候选尺度选择。
- Stage 8 双窄带哈勃艺术分支默认开启，由 `STARUN_STAGE8_DUALBAND_PALETTE_ENABLE=0` 显式关闭。三条正向颜色路线互斥且只消费一次预算：确认 Ha/OIII 双窄带只运行 palette；宽带默认运行主体增色；宽带、无冻结物理 SPCC/PCC 锚点且同时显式开启 `STARUN_OPTIONAL_COLOR_TRANSFORM` 与插件探测时，Vectra 预留全部预算并禁止通用 saturation/主体增色。Vectra 不可用、异常或拒绝时精确回滚且不补跑隐藏颜色路线。主体事务继续使用 `1 + (legacy_factor - 1) × clamp(effective_saturation/0.40, 0, 1)` 系数并写 `stage8_subject_chroma_report.json`。
- 任务参数 `stage8_dualband_palette_selection` 默认为 `auto`，也可手动指定 `HSO/SHO/OSH/OHS/HOS/HOO`；手动值是唯一 palette，拒绝后回滚而不切换色盘。处理计划一次冻结 `requested_palette / automatic_palette / palette / selection_mode / manual_override`，Stage 8 只消费该结论。该分支只读取 Stage 4/Stage 5 验签恢复的冻结映射，不得使用当前 Header 重新分类；契约、映射、色盘结论缺失或字段异常时明确跳过。双窄带 SPCC 失败后，通过安全门的降级 PCC 父本仍执行伪色映射，但不清除 `requires_review`。每个 palette 复用 Stage 6/7 冻结主体 ROI，并扣除核心、恒星和 halo guard；在同一 palette 内按 `1.00/1.12/1.25/1.40` 评估保亮度色度尺度，选择最弱的全门通过档。自动模式从冻结目标首选开始，在六种固定 `Classic` 色盘中执行受门控竞争，并按主体/背景色度分离、主体色度增益、亮度漂移、裁剪增长和稳定顺序确定性选择；通道公式为 `H=R`、`O=(G+B)/2`、`S*=(H+O)/2`，报告披露 `S*` 为合成代理。调色前必须保存 `stage8_pre_palette.fit`；接受后更新规范 `stage8_enhanced.fit` 并进入 Stage 9，拒绝或异常精确回滚，回滚失败则降级并要求人工复核。Stage 9 禁止再运行 Vectra 二次转色。
- Stage 8 亮星云 halo 必须执行 `full <= 0.350`、`limited <= 0.600`、`skip > 0.600/硬失败` 三级门；`skip` 不得依赖 AI/masked 开关才生效，`stage8_input_starless` 基线保存失败时任何策略都必须安全旁路。limited 只允许 masked builtin、饱和度 `<=0.05`、背景增益/锐化为零，并保存 `stage8_limited_candidate.fit`。质量基线/候选不可读必须 fail-closed；通用品质门或环带纹理门拒绝后必须回滚 `stage8_input_starless`，且报告保留原始触发 advisory、修复后指标、接受上限和 outcome。成功导入的显式外部 Starless 回写优先于 `stage7_stretched`，选择证据必须写入 `stage8_input_selection.json`。
- Stage 9 在插件链之前用不可变线性含星参考修复星核/星翼色度，必须通过色度改善、通量、覆盖、裁剪和回混前复核；失败时恢复原 starmask，不改变既有回混梯级。
- M8 类 `bright_composite_compact_flux_v1` 在 `0.06/0.26/0.50/0.75` 实际四锚曲线之后，把认证 Unscreen 峰值相对可信 Screen 峰值的恢复幅度固定为 `0.55`；其他 profile 保持 `1.0`。插值只作用于已有认证支持内的同坐标星层，沿用可信 RGB 比例，且仍重跑原恢复、色彩、背景、PSF、SEP 与持久化门。
- Stage 9 默认通过随包 Seti Astro Suite Pro 的无头适配器自动执行 `NB to RGB Stars -> SASP Star Stretch`，不依赖 `workflow_plugin_probe_enabled` 或未注册的 Siril `sasp_*` 命令。NB 工具只接受冻结的 `starun.narrowband-channel-mapping.v1`（`R→Ha、G/B→OIII`、置信度达到 `stage4_nbn_mapping_confidence_min`）和 `narrowband_composite` 语义；不满足时记录 skipped 而不猜通道。NB 成功先写 `starmask_nb_to_rgb.fit`，SASP 成功再写 `starmask_stretched.fit`；每个适配器在 `image_lock` 内原子读写并在失败时回写输入。`stage9_remix_quality.json.star_plugin_preprocessing.direct_plugin_reports` 必须记录 wheel、参数、映射校验、运行引擎和 rollback 状态。
- Stage 9 starmask 的“轻度”是输出合同而非固定 `k`，且不引入 LHE。普通场对 compact 支持像素测量 `P50/P75/P90/P99.7`，混合场对冻结星峰测量 `P40/P80/P90/P99.7`；通用 faint/mid/bright/peak 实际输出不高于 `0.26/0.50/0.75/0.90`（数值容差 `1e-4`）。目录目标为 `emission_nebula_widefield`、通道语义为 `narrowband_composite` 且签名参数未显式覆盖任一锚点时，runtime 复制配置并只对星层标定使用 `0.08/0.30/0.50/0.75`；`bright_emission_reflection_nebula + broadband_rgb(_osc)` 使用 `0.06/0.26/0.50/0.75`。原签名参数不被修改，用户显式锚点优先，适用证据及 configured/effective/实际曲线值写入 `stage9_starmask_target_profile.json`、preflight 与总报告，三者不一致即测试失败。两类 profile 仍重跑原有恢复、色彩、背景、PSF、SEP 和持久化门。内置 Asinh 分别反解四个最大 stretch 后取最保守值，并受 `min(stage9_starmask_predicted_change_ratio_max, 0.9×stage9_changed_pixel_ratio_max)` 收紧；多锚点曲线在覆盖超限时确定性二分统一缩放四个目标。插件必须测量实际输出与实际变化覆盖，测量不可用或输出超限即在 Screen 前拒绝；`stage9_starmask_asinh_stretch` 在 adaptive 关闭时也只是受同一硬门钳制的初始提案，标定不可用不得执行固定强度回退。v7 继续保留 calibration/preflight 的实际四锚点、目标、目标缩放因子、覆盖、派生 stretch 与拒绝原因，`k/stretch` 只作派生诊断。
- Stage 9 在首次 Screen 回混前，对同一冻结星表和 raw starmask 只读构造 normal/strict compact 支持摘要，复用 `stage9_star_support_ratio_max`、`stage9_starmask_predicted_change_ratio_max`、`stage9_changed_pixel_ratio_max` 和共享 `1.5×` advisory 规则。normal 清晰安全时只运行 normal；normal 位于 advisory 且 `auto_fallback` 生效时预备双候选；normal 硬越界/不可用而 strict 合格时只运行 strict；两者都不可用则在回混前 fail-closed。插件已拉伸星层必须以实际插件输出估算变化覆盖，实测不可用时不得借用内置 Asinh 预测。双候选不是并发 Siril 操作：它们在 `image_lock` 保护下串行从同一不可变 remix base 生成并各自保存检查点，过滤全部硬门拒绝后按无 advisory、最坏/平均 PSF 偏差、最低归一化恢复裕量、支持覆盖、高光增长和 normal 同分优先选择并事务恢复。主强度 `screen_compact_primary` 是正常候选，不记 fallback；降强度和运行期不可预测污染触发的 `screen_compact_recovery` 仍记 fallback。`preserve_review/stop` 不运行 advisory 双候选，首次实际拒绝后维持停止搜索语义。
- Stage 9 在建立支持候选前只解析一次并冻结 `stage9_spatial_scale / starun.stage9-fwhm-spatial-scale.v1`。尺度优先取有效、未饱和、隔离的 matched-display FWHM，其次取冻结 Stage 5 `fwhm_geometry`，最后取 raw starmask 连通星核半高直径；每层至少 4 星，后两类来源设置 `stage9_psf_review_required=true`，全部不足时以 `stage9_spatial_scale_unavailable` 在首次回混前 fail-closed。`4.0 px` 为 1× 锚点，半径/窗口/浮点匹配距离按 `FWHM/4` 缩放，面积按其平方缩放；支持半径向上取整、普通窗口/搜索半径四舍五入、浮点匹配距离不取整，3×3 结构元只表示拓扑邻接且不缩放。启动星表仅作尺度采样，正式星表必须用冻结尺度重建；所有 Screen、Unscreen、PSF、来源完整性和 compact 候选复用该结果，禁止逐候选重估。
- `stage9_remix_quality.json` 中的尺度对象记录来源、样本数、中位/P25/P75、4 px 锚点、radius/area scale 与公式，不写入逐星 NumPy 数组；每个 attempt 记录 `spatial_scale_source`、名义补翼档和实际支持半径范围，空间面积 `limits` 写实际有效值并并列保留名义配置值。
- Stage 9 同域映射以 `stage7_stretch_quality.json.matched_domain_transfer` 为权威合同。Display、linked-MTF、composite、背景色度救援和 adaptive-quantile winner 必须分别按其 v3/v4 合同重建并验签完整 tone 链；合同缺失、损坏、私有 mask SHA、curve/chain digest 或来源绑定不符时停用 Unscreen 并要求复核，禁止改用 cand_b MTF 或另一 LUT。
- Stage 9 保留 Stage 6 线性 `original-starless` 分离和 Alpha+Screen 基线，不调用 SyQon 伪 descreen 或 StarComposer。只有 Stage 6/7、不可变 pair handoff、独立源星表和 `matched_domain_transfer.v4` 全部验签时，才用 Stage 6 的 `B_pair` 构造 `U=(O-B_pair)/(1-B_pair)`；该 Unscreen 层只采用上行幅度并沿用可信 starmask 的 RGB 比例。正式保真与偏大 PSF 收缩使用的底图则必须是 `stage8-handoff.v3` 验签的实际 `B_stage8`，不得以 `B_pair` 替代。Screen/Unscreen 的 all/weak/bright 科学 FWHM 硬门保持 `0.93–1.10`，展示软目标为 `0.97–1.05`：偏小组只运行 fractional source-wing 搜索；偏大组先从认证 `O`、实际 `B_stage8` 和冻结目录执行 connected half-max boundary pruning，只缩放真实母星边界，不新增坐标，逐组件质心漂移 `<=0.05 px`。boundary 候选只运行一次完整 assess 和一次相对父候选零容差 non-regression；失败后才允许对 `u>0.45` 肩部使用 RGB-shared gamma fallback，`u<=0.45` 星翼、峰值和通道比例保持不变。各 Screen/Unscreen formal family 先独立闭环后统一排名，最终候选再执行 common closure；任一身份、坐标、恢复、色彩、背景、高光、SEP、PSF 或父候选保真退化都精确回滚。参考子组不足、合同不可重放或持久化复验失败均保持 review-only；正式像素回混仍记 `stars_application_mode=screen`。
- Stage 4/5/6 只写规范产物 `stage4_color.fit`、`stage5_linear.fit`、`stage6_starless.fit` 与 `stage6_starless_quality.json`；旧产物名不读取、不迁移、不写出。拒绝/工具失败时不得创建任何 Starless 命名产物。
- Stage 9 的 `stage9_remix_quality.json` 使用 `starun.stage9-remix-quality.v10`，固定 `selection_policy=sep_catalog_visibility_psf_fidelity_recovery_v8`，并包含 `starmask_support_preflight`（`starun.stage9` 的 v2 starmask support-preflight 契约）、`catalog_visibility`、独立 `stage9_sep_crossmatch.json` 摘要与 `persisted_output_validation`。插件 faint/mid/bright/peak 四锚点必须各达到目标的 `50%` 才有正式资格；不足、测量缺失或坐标合同不匹配时改用同一 raw starmask/normal support 构建的内置多锚点层与 strict 分支竞争。正式候选除既有 `0.93–1.10` FWHM、色度、高光、背景、裁切和光晕门外，还必须通过源目录绝对可见度（源局部对比度 `>=0.002`；全部/弱/亮星可见率分别 `>=0.75/0.70/0.90`）和独立 SEP 门，保存后重新加载 `stage9_remixed.fit` 核对尺寸、像素哈希、坐标域并重跑关键门。只有持久化、目录可见度与 SEP 复验全部通过才设置 `stars_applied=true / remix_formally_accepted=true`；旧 v9/v8 可读取但不能缺省提升为正式成功。正式路径失败后仍按冻结 Stage 8 Starless + 受控/原始 Starmask、最后 Stage 5 的顺序生成含星 review-only 回退；全部不可安全验证时必须 withheld，禁止把 Starless remix base 冒充最终或复核结果。
- Stage 9 报告还必须分别记录 `upstream_passthrough` 与 `stage9_fallback_used/reason`；Stage 8 安全旁路 + Stage 9 主 Screen 成功的组合应为 `true / false`，GUI 显示“成功（使用 Stage 8 安全旁路源）”。
- Stage 8 发布 `stage8_handoff.json / starun.stage8-handoff.v3`，显式绑定 `processing_route`、`formal_eligible`、`restricted_downstream`、实际源文件名、容器/像素 SHA、lineage、Starless finish 报告和最终质量。Stage 9 在任何星点处理前逐字段验签，不再从 `final_quality == "ok"` 推断资格；旧 v1/v2 只兼容为 review-only，身份、SHA、lineage 或 restricted 状态异常时禁止正式回星并转可信含星复核路径。Stage 9 的 `upstream_source_stem` 与 `remix_base_stem` 固定为验签后的 Stage 8 产物，不再生成 `stage9_starless_base.fit`。
- Stage 9 在候选接受、全部拒绝回滚或 starmask fail-closed 且 `stage9_remixed.fit` 保存成功后，都要生成 `review_bundles/stage9_star_remixing/`；复核 before 固定使用验签后的 Stage 8 产物。复核包生成失败只记录警告，不改变质量门的确定性结论。四张 16-bit 工作图只服务运行期判定；最终成功阶段只保留 `review.json`，异常阶段另保留一张 8-bit 紧凑 `review.png`。
- 非 Debug 交付清理必须删除工作目录精确命名的 `sasp_starless_input.fit / sasp_starmask_input.fit` 和全部 `process/*.fit/.fits`；`starless.fit`、`starmask_raw.fit`、`starmask_clean.fit` 和外部星层都不是正式断点或交付。`starun_diagnostics.zip` 只归档 JSON/JSONL、日志、CSV/TXT，不重复收录 review/UI PNG。
- Stage 10 正常导出必须保留 TIFF/PNG/FITS fallback 名称，包括续跑模式的 `_linear` fallbacks；全模板因 `STACKCNT` 等字段不完整而不可用时，应先用 `OBJECT / EXPTIME / DATE-OBS` 生成可识别的安全字面名称，身份信息不足时才使用通用 fallback。若 Stage 2 视场或 Stage 3 背景要求复核、Stage 4 色彩要求复核、target-bypass 的 Stage 7 未验收、`stage9_psf_review_required=true`、最终源不是 Stage 9 记录的首选源或无法确认、`stage10_final.fit`/最终质量门不可用或报告契约不一致、最终质量报告要求保守重跑，或 `stars_required=true`、`stars_applied=false` 但 `output_contains_stars=true`，则只写 `result_review*`，不得覆盖普通 `result_processed/result_final`。若 `stars_required=true` 且 `output_contains_stars=false`，Stage 10 必须在写任何正式或复核交付前以 `required_stars_output_withheld` 失败。
- Stage 10 实际启动末端降噪前必须冻结输入，并仅从 Stage 9 验证星表重建有界星点保护 mask；mask 不可用/超覆盖时安全跳过。所有后端成功结果统一回混硬星核与羽化边界，回混失败恢复冻结输入；统一强度来自 `stage10_final_denoise_strength`，不得在调用点另写固定强度。无 headless 强度参数的交互式 SCUNet GUI 脚本不得进入自动回退链。
- Stage 7 正式候选发布 `stage7_spatial_background_reference.json / starun.stage7-spatial-background-reference.v1`，绑定 Stage 3 support SHA、候选像素 SHA、认证 tone 链摘要及理论显示域指标。Stage 10 最终质量报告使用 `starun.final-quality.v4` 和 `starun.final-spatial-background-gradient.v2`：最终亮度或归一化 `R-G/B-G` 一阶平面相对同一 Stage 7 理论显示域参考新达到 3σ，或斜率增长超过 1.25× 时，使用 `final_spatial_background_gradient_unresolved` 禁止正式交付；认证参考本身的显著性仅作诊断，逐像素不变的来源不会被重复否决。Stage 3 线性指标只作诊断。参考缺失或篡改 fail-closed，且不触发隐藏色彩修复。
- 已进入 review-only 且 Stage 9 验签为含星的路径，Stage 10 在星表审计前冻结 observer-only display contract，并对 Stage 5 签名参考与复核候选重放同一映射；通过既有 0.002/16 点/可见率门后只生成 `result_review*`，原始 FITS 不改写。线性大星系若在映射前因动态范围压缩而被主体代理标为 `unmappable`，必须先验签可信含星 lineage 与至少 16 个 Stage 5/9 目录星，才允许建立 bounded linked mapping，并在映射后重跑主体和星表门。普通紧致峰不能替代星表，合同、SHA、尺寸或同域审计失败仍 withheld。`stage10_pre_export_visibility.json` 与 catalog resolution 使用 v2 并明确记录审计域和交付范围。
- Stage 6 拒绝 Starless pair 或工具失败后，Stage 7–9 的含星 review-only 检查点必须逐像素保持验签来源，不得在可信 FITS 上提前执行 `autostretch -linked`。唯一 linked 显示映射只用于 Stage 10 的 PNG 与同显示域星表审计，避免参考星表仍在线性域而候选已在非线性域的契约错配。
- Stage 10 的受管理导出只新增 `*_display_srgb.png` 与 `*_edit_srgb.tif`，不补写或重写 Siril 原产物。`audit_display_visibility`、managed output 与对应报告使用 v2，可携带显式源星目录及 `siril_pixel_buffer_bottom_up`/`display_array_top_down` 坐标域。PNG 除必须带 sRGB 声明外，还会先检查源显示亮度，必要时只对 display 衍生图执行 linked 可见度映射；在导出前像素和解码后的实际 PNG 上分别复核目录星、亮度与宽延展主体。通用 `compact_peak_count` 仅作诊断，不能满足 `stars_required`；目录不可用或任一目录可见度审计失败时，即使任务已因 Stage 3 等原因进入复核，也不得发布无星 `result_review*`。TIFF 必须内嵌有效 ICC，否则该衍生文件 fail-closed。`managed_output_report.json` 与 `output_color_manifest.json` 分别记录映射前/成品像素、写入和容器复核，FITS 在导出前后以 SHA-256 验证不变。
- Stage 3 的 `verified_noop` 只在完整真天空/方向梯度/图样噪声/像素/目标保真/绝对背景门通过、且全部候选仅因改善低于 `3σ` 不确定度而拒绝时成立；它与实际校正都发布 `starun.stage3-spatial-background-lineage.v2`，绑定 run、正式 route、输入/输出 artifact 与解码像素 SHA、support、拟合/验证点、参考平面和 chain digest。v1 只读且固定 `legacy_nonformal`。Stage 5 只消费验签后的冻结天空、主体结构和多尺度噪声证据，冻结模型必须以 schema、尺度、mask/input SHA 与 digest 验签并实际驱动滤波；按 `0.60/0.72/0.84` 选择最弱合格候选，正式门固定为噪声下降 `>=0.12`、细节保留 `>=0.90`，全拒时精确回滚且 handoff `formal_eligible=false`。
- Stage 7 色度和绝对可见噪声必须从 Stage 6 冻结 ROI/linked preview 计算；一般目标主体 nominal/hard 为 `0.70/0.55`、非背景 `0.45/0.30`，star-preserve 主体为 `0.35/0.23`。显示域天空噪声 `<=0.025` 正式通过、`0.025–0.0375` advisory、再高硬拒。正式 winner 另发布内部 `stage7_presentation_reference.fit/json` 并双 SHA 绑定。
- Stage 8 结构 retry 只复用同一已验证提案，在目标权重与核心/恒星/halo guard 内按 `1.0/0.75/0.5/0.25` retained-delta 选择最强全门合格候选；mask 外必须逐像素不变，冻结天空噪声增长 `<=1.10`。全失败精确回滚后，只有输入 SHA、色彩、背景、seam、halo、裁切和表现预检全部验签才允许 `safe_passthrough_color_only`；双窄带 palette 必须使用冻结且带保护的主体 ROI，内部按 `1.00/1.12/1.25/1.40` 选择最弱通过的保亮度色度尺度，还要求主体饱和度 `>=min(0.5*输入, 0.08)`。正式 v3 handoff 还要求 fresh 的 `stage8_final_cumulative_quality.json` 从冻结输入到最终持久化像素重跑 seam、累计噪声、mask 外恒等、完整质量、空间背景、halo、色彩表现和容器/FITS-data/decoded-pixel SHA。累计 mask 外恒等使用候选无关的授权变更并集：仅当相应事务被接受时，才并入冻结结构支持、Stage 6/7 冻结主体色度 ROI 或冻结 palette 主体 ROI；并集摘要和 SHA 写入报告，并集外任一像素变化都硬拒。结构候选若只在此累计门失败，必须先精确回滚，再从 Stage 7 基线走一次无插件色彩旁路并重跑全部独立门，不能复用失败结构或重调插件。若两条色彩事务分别以“明确不适用”或“质量拒绝且已验证回滚”的受审计终态结束，且最终像素与 Stage 7 基线精确相同，可记录 `verified_color_noop_after_structure_rollback`；schema/事务证据缺失、回滚失败或像素不等均拒绝。第二次累计门失败仍为 review-only。
- Stage 10 只对正式 Stage 9 来源执行末端降噪，不对 review-only 来源承担低频背景扣除。最终背景质量只消费 `stage10_background_support.json / starun.stage10-authenticated-background-support.v1`，由 Stage 3 v2 support 与 Stage 6/7 冻结 mask 交集生成，禁止从最终候选重选暗像素。`presentation_quality_report.json / starun.presentation-quality.v1` 以 Stage 7 冻结表现参考复核色彩 P50/P95、opponent energy、方向相关、可见度、目标 profile 亮度、微细节和 `0.97–1.05` 星点表现软目标；科学 PSF 硬门仍为 `0.93–1.10`。`final_artifact_identity_report.json / starun.final-artifact-identity.v1` 重新解码 Stage 10、正式 FITS 与请求的 managed PNG/TIFF，验证正式名称、文件 SHA 和量化后像素链。`pipeline-result.json.delivery_gates / starun.final-delivery-gates.v1` 只有科学门、表现门、无 review requirement、正式 artifact 身份四项同时通过才设置 `formal_delivery_accepted=true` 与 `delivery_eligible=true`。GUI 下载资格只读取该字段；已签名的历史 `pipeline-result` v1/v2/v3 均可读取，其中 v3 固定为 `legacy_schema_read_only`，任何历史结果缺少新双门字段时均标记 `legacy_delivery_contract` 且不可下载。
- `script/five_target_reference_qa.py` 位于生产 Stage 1–10 之外，只做外部参照注册、测量和报告/联图生成。SIFT 注册先取双向 Lowe-ratio 互认匹配以排除星密场的一向同形误配，互认点不足才使用单向稀疏回退；两路均须通过相同的最少匹配/内点、`>=0.30` 内点比例、重投影 P95 和视场重叠硬门。manifest 必须完整覆盖五个锁定目标/profile，baseline/optimized/reference 路径与 SHA 互异；baseline 验签所属历史 run，optimized 验签新 run 的 `pipeline-result.json`、正式 artifact SHA、`delivery_eligible` 与科学/表现双门。人工检查逐项绑定 optimized SHA，并要求全分辨率复核；工具执行前后复核三类输入 SHA 不变。报告固定输出 `external_reference_used=true`、`production_feedback=false`、`production_pixels_written=false`，目录必须位于各生产 run 和参照输入树之外，结果不得反向修改生产像素、阶段门、`delivery_gates` 或 GUI 下载资格。
- Runtime plugin 路径必须保持离线可用：SyQon、SASP、CosmicClarity 资源来自 bundled/local cache；缺失时应作为 preflight/runtime 问题处理，不应要求 pipeline 内联网下载。
- `final_artifact_identity_report.json` 对源绑定的观察专用 `linked_visibility_v2` 从冻结源精确重放黑白点、目标中值、gamma 与 RGB 共同比例映射，并比较 managed PNG 的解码 SHA；参数、来源或像素篡改均 fail-closed。
- 检查点压缩后的第二次结果清单复验允许 `process/stage10_final.fit` 已被清理，但只能以唯一正式科学 FITS 作为 `compacted_scientific_archive_anchor`：文件 SHA 必须匹配输出清单，解码像素 SHA 必须精确匹配 `managed_output_report.json` 中冻结且声明 `checkpoint=stage10_final.fit` 的源身份。随后所有 managed 角色继续按该同一像素源重放；缺失、多锚点或 source/report/output 任一不一致均 fail closed。

- 压缩底座的 `large_galaxy` Display 候选在现有 v2 合同内使用有界亮度分位数锚点；诊断写入 `quantile_tone_curve`，逐像素统一 RGB 增益、65536 点 LUT 摘要和 SHA-256 重建验签不变。

Stage 9 formal candidate score 先比较 `soft_target_closed`，再比较 visibility、PSF 偏差、reference fidelity、恢复率和高光风险。该优先级只影响多个已通过科学硬门的候选之间的选择，不改变 `0.93–1.10` 科学阈值或 `0.97–1.05` 展示阈值；common closure 和 persisted-output validation 仍在胜出后重新执行。

## Plugin Cache

- 准备：`bash resources/siril_plugins/download_siril_plugins.sh`；脚本固定按 Siril CPython 3.12 / `cp312` 下载 wheel，并清理 `downloads/` 中同一库的旧版本和不兼容 ABI。早期 `cp37-cp311` 标记的 `abi3` wheel 在 CP312 可用时保留，`cp313` 及更高 ABI 一律拒绝。
- GUI 每轮检查插件脚本、SASP/SyQon/AberrationRemover、PyQt6/PySide6/astropy/scipy/tifffile/onnxruntime 等 wheels，并在安装前拦截非 CP312 兼容 ABI。
- 缺失时自动尝试下载；仍缺失则阻断运行。
- runtime offline pip 只从 bundled wheels 安装，并写入 `tiffile` shim 与 `sitecustomize.py` sirilpy timeout patch。
- `PySide6_Addons` 仍保留在 Full/Core 离线资源中；只有完成真实断网、全新 runtime 的 SyQon GUI/CLI 回归后，才允许改为 Essentials-only。

## Validation Matrix

| Changed area | Required checks |
|---|---|
| Pipeline | `python3 -m py_compile pipeline/starun.py pipeline/stages/*.py`; 核对 stage 顺序、降级、TIFF/PNG/FITS |
| Stage 4 color/runtime | `python3 -m pytest -q tests/test_stage4_auto_reference.py tests/test_stage4_pcc_policy.py tests/test_runtime_capabilities.py tests/test_processing_parameters.py tests/test_pipeline_plugin_fallbacks_runtime_contracts.py tests/test_gui_runtime_modes.py`；检查物理成功跳过影子、在线未验证与本地默认均为 300 秒、星体 256 上限、留出尾部门和精确回滚阻断 |
| Task gate profiles | `python3 -m unittest tests.test_processing_parameters tests.test_processing_parameter_passthrough tests.test_processing_parameters_ui_state`; `QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_gui_history_ui.py`; 检查 v5 新写、历史 v4 兼容迁移、静态基线、1×/3×/10× 方向缩放、物理截断、专家优先、无限强制复核、签名审计与恢复指纹 |
| Stage 3 background gates | `python3 -m unittest tests.test_stage3_background_sampling tests.test_processing_parameters tests.test_pipeline_stage_smoke`，并运行 `tests/test_pipeline_plugin_fallbacks_stage3_background.py`；检查三档阈值、内置优先顺序、插件开关、正式 Pareto 选优、RGB 中性轴不变量/单次 subsky/headroom/重载门、软告警保留和全硬拒绝回滚 |
| GUI history / task deletion | `python3 -m py_compile gui/history_store.py gui/main_window.py`; `QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_gui_history_store.py tests/test_gui_history_ui.py tests/test_gui_runtime_modes.py tests/test_gui_ui_system.py tests/test_task_workspace.py`; 核对不补录旧 run、异常终态恢复、历史详情验签，以及任务根/容器/符号链接/清单不匹配删除边界 |
| Structured stage status / Stage 5/8/9 | `python3 -m pytest -q tests/test_pipeline_plugin_fallbacks_*.py tests/test_gui_runtime_modes.py`；检查消息关键词不改变状态、Stage 5 子状态、亮星云三级门/受限候选回滚，以及 Stage 9 upstream passthrough 与本地 fallback 分离 |
| Target policy / gates | `python3 -m unittest tests.test_adaptive_pipeline_phase1 tests.test_pipeline_stage_smoke`; 检查 target fallback、policy fallback、dirty background 保护、Stage 7 目标局部指标、raw/clean 星点层和 Stage 9 自适应回星门控 |
| Deterministic B/C/D engines | `python3 -m unittest tests.test_device_geometry tests.test_noise_model tests.test_narrowband_normalization tests.test_star_color_repair tests.test_local_adjustments tests.test_managed_output tests.test_output_color` |
| GUI/preflight | `python3 -m unittest tests.test_runtime_capabilities tests.test_gui_runtime_modes`; 缺 Siril/config/完整 pipeline/FITS/result_linear 的错误清晰；App Bundle 边界、48 个 XP 分块、网络/本地二选一、每 run 预检前日志与状态隔离、Stop 行为均保持安全 |
| Dev launcher | `python3 -m unittest tests.test_gui_dev_launcher tests.test_pipeline_dev_launcher tests.test_manual_core_pipeline_smoke`; 检查显式 Siril/seed 校验、临时资源覆盖、无 GUI 核心入口和结果清单校验 |
| Build / Cocoa launch | `bash -n build/build_macos_app.sh && bash -n script/build_and_run.sh`; `./script/build_and_run.sh --verify`；PyInstaller 边界检查通过；`dist/Starun.app` 仍嵌入 Siril、pipeline、config、CP312 wheels 与模型；`codesign --verify --deep --strict` 通过 |
| Plugins | cache 完整性、缺失补齐/阻断、offline wheels 安装 |
| Toggles | `Debug`、`checkpoint_mode`、联网和处理模式映射到预期 env/Siril flags |
| Coverage | `python3 -m pytest -q -rs --cov=gui --cov=pipeline --cov-branch --cov-config=.coveragerc --cov-report=term-missing --cov-fail-under=65` |
| Real Siril E2E | `python3 tests/real_siril_stage1_10_e2e.py`；专用 runner 必须验证真实输入、系统 Siril、完整离线 Gaia、仓库外预置的 SyQon/CosmicClarity 资源、Stage 1–10 正式状态、最终质量/星点交付、SyQon lineage、输出路径与 plan/result 哈希 |

Baseline smoke: embedded Siril + external pipeline 可运行；stage 1-10 能导出 TIFF/PNG/FITS；Debug 保留 `process/stage*.fit`。

新真实 E2E 除既有科学质量、星点和 lineage 验收外，还必须验证 `presentation_quality_report.json.accepted=true`、`pipeline-result.json.delivery_gates.formal_delivery_accepted=true`、顶层 `delivery_eligible=true`，并确认 GUI 读取同一四门合同。缺少新合同的历史任务只验证可读性，不得计入正式交付。

人工核心算法快速回归：`.venv/bin/python tests/manual_core_pipeline_smoke.py`，
脚本只从原始输入执行完整 Stage 1–10，网络默认开启；需要验证严格离线回退时
显式传入 `--offline`。脚本不会打包或启动 GUI，并在 launcher
成功后复核本轮结果清单与全部登记产物哈希；它不覆盖 frozen App 的资源布局、
GUI 参数注入、签名和 Finder 启动验证。

自动真实回归由 `.github/workflows/tests.yml` 的 `real-siril-stage1-10` job 执行。runner 必须带 `self-hosted / macOS / ARM64 / starun-e2e` 标签，并配置下列 repository variables：

- `STARUN_REAL_E2E_INPUT`：已审阅、会实际进入 SyQon 的线性 FITS/XISF 母版
- `STARUN_SIRIL_APP`：真实 Siril.app 路径
- `STARUN_SIRIL_SEED`：包含 `.python_module` 的 Siril seed
- `STARUN_RUNTIME_HOME`：其中 `.local/share/siril/` 已含精确大小的 astro catalog 和 `0–47` 共 48 个 XP 分块
- `STARUN_OFFLINE_RESOURCE_ROOT`：runner 上仓库外只读资源包的绝对路径；根目录必须包含 `siril_plugins/`、SyQon Zenith 及 CosmicClarity mono/color denoise 模型，模型不由 Git/LFS 提供
- `STARUN_REAL_E2E_READY=true`：仅在上述输入、runtime、Gaia 与完整插件资源全部准备并人工核验后设置

每周调度会在 runner 就绪时自动执行；`workflow_dispatch` 也可勾选 `run_real_siril_e2e`。运行目录与日志完整上传为 CI artifact。任务只有在原始 `starun.pipeline-result.v2` 为 `success`、Stage 1–10 按序全部 `ok/completed`、无降级/回退/复核、`starun.final-quality.v4` 为正式 `ok`、星点交付契约通过、SyQon accepted attempt 与 selected generation lineage 一致，且非 review 的 `*_final.fit` 位于本轮工作目录内时才通过；退出码为 0 本身不是验收证据。

真实 pipeline 通过 `--offline` 与 `STARUN_NETWORK_MODE=0` 禁止运行期模型/星表联网，`download_siril_plugins.sh` 也不会在该 job 中执行。工作流仍使用 `actions/setup-python` 并安装普通 Python 测试依赖，这些准备步骤可能联网；这里的“离线 E2E”特指 pipeline 执行与模型资源，不代表整个 CI job 完全 air-gapped。
