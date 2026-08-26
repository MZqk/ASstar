# Starun

面向 Seestar 天文图像的自动后期处理工具。以 **Siril 1.4+** 为核心，结合离线 Zenith-only `SyQon/Starless.py`、SASP/CosmicClarity 等插件链路，把 FITS/XISF 母版或 Seestar Light 自动导出为 TIFF/PNG/FITS。提供 macOS GUI 与离线打包。

### Starun 命名迁移

运行时环境变量统一使用 `STARUN_*`；旧的 `SEESTAR_*` 不再读取。任务清单和阶段报告的 schema 前缀也已改为 `starun.*`，因此旧版任务不能作为 Starun 的续跑输入，应新建任务。

系统要求：**macOS 14.0 或更高版本**，仅支持 **Apple Silicon（arm64）**，不支持 Intel Mac。

## Key Files

| Path | Role |
|---|---|
| `pipeline/starun.py` | stage 1-10 主流程编排 |
| `pipeline/stages/` | stage 1-10 分离实现模块 |
| `pipeline/stage_contracts.py` / `pipeline/task_plan.py` | 阶段、正式断点、文件命名与冻结计划契约 |
| `pipeline/input_discovery.py` / `pipeline/task_workspace.py` | 输入识别、Light 分组、只读来源与任务历史 |
| `pipeline/starun_workflow.md` | 详细流程说明 |
| `gui/starun_gui_app.py` | GUI 入口 |
| `gui/starun_gui_dev.py` | 使用系统 Siril/现有 seed 的免打包开发入口 |
| `gui/starun_pipeline_dev.py` | 不启动 GUI 的源码核心流水线验证入口 |
| `gui/main_window.py` / `gui/pipeline_worker.py` | PySide6 主窗口与 Siril worker |
| `gui/ui_theme.py` / `gui/ui_platform.py` | 共享设计令牌、系统浅深色主题与桌面平台适配 |
| `build/build_macos_app.sh` | macOS 打包 |
| `requirements.txt` / `requirements.lock` | 直接依赖约束 / pip-tools 生成的完整依赖锁 |
| `resources/default.env` | 项目内默认 runtime env；无需手工 export |
| `INTEGRATION_README.md` | 内部集成/runtime/验证细节 |

## Pipeline Summary

主流程是 starless-first 后期链路：输入统一和裁切 -> 输入线性状态判定 -> 冻结主目标/策略与处理计划 -> 条件背景提取、自动设备几何/platesolve、Stage 4 按 `SPCC → PCC → 自动背景/白参考区域校色 → 原样保色` 处理颜色并冻结窄带通道映射（SPCC/PCC 命令成功且候选技术完整即直通；自动区域校色只用于线性宽带 RGB/OSC 并强制复核；双窄带 PCC 仅作强制复核的降级基础校色）、噪声模型驱动线性反卷积/降噪 -> 线性去星与确定性星色修复 -> Starless 主体拉伸和 Stage 8 本地曲线/蒙版、主体低频增色及确认 Ha/OIII 的目标首选伪色 -> 受控回星 -> 最终降噪和科学/展示/编辑版导出。输入状态只接受可追溯证据，分为 `linear / nonlinear / unknown`；非线性或未知输入只执行安全裁切并进入复核导出，不运行破坏性的线性处理、去星或 Starless 增强。

Stage 5 的“自动”线性降噪默认进入候选链；是否实际执行只由其冻结基线低噪声门和候选质量门决定，Stage 1 不再以单一 `bg_std` 提前关闭降噪。

Stage 10 末端降噪会复用 Stage 9 已验证星表保护星核和星翼；保护 mask 不可信时安全跳过，不允许 fallback 无 mask 处理。处理参数可选择 `auto/preserve`、末端降噪与最终饱和度开关，以及 `auto_chain/cosmic_only/scunet_only` 后端策略；`preserve` 保持已验证 Stage 9 含星源的像素不变，但仍执行最终质量门和导出。降噪强度默认 `0.28`。

Stage 9 的像素几何以 `4.0 px` FWHM 为缩放锚点：半径、窗口和匹配距离按 FWHM 比例缩放，面积门按比例平方缩放。尺度优先来自同域显示 FWHM，其次为 Stage 5 冻结星表和原始 starmask 半高测量；后两种来源要求复核，三者都不足 4 星时会在首次回混前安全停止，不采用固定像素回退值。

Stage 9 在首次回星前会先评估普通与严格紧致两套星点支持；边界场景把两者作为同一不可变底图上的事务候选比较，已知宽支持越界时直接使用合格的紧致候选，不再先执行必然失败的宽掩膜。Siril 图像写回仍保持串行。Stage 9 保留线性 `original-starless` 星层和既有 Alpha+Screen 作为基线，并默认增加同域 Unscreen 幅度候选。Stage 6 验收时会冻结 `stage6_input/stage6_starless` 的路径、哈希、形状和像素域，Stage 9 不再受 Stage 8 可变 `starless_file` 指针影响。它按 Stage 7 的 `matched_domain_transfer` 建立同一显示域：本轮可精确重放的 Display、实际胜出的 `cand_b` linked-MTF 和 composite 候选只发布 v3 有序步骤合同，Stage 9 对原始含星图与 Starless 重放同一整链；无法序列化真实 winner 或参数/摘要不符时直接停用 Unscreen，禁止借用 `cand_b` 锚点，v1/v2 仅兼容旧任务。该域只用于恢复星点亮度与核翼幅度，RGB 比例继续来自已通过修复/正则化的可信 `starmask_stretched`；该 Unscreen 是保色幅度校正候选，不声明逐通道精确逆或科学测光保真。只有闭环 RGB MAE 同时改善至少 `10%` 和 `0.005`、不降低星点/星翼恢复且不新增色彩或结构风险时才会选中；否则事务式恢复 Screen 基线。Screen/Unscreen 都使用同星显示域 FWHM 闭环：未饱和且可测的全部、弱星和亮星组必须保持源星径的 `93%–110%`，饱和星单独观察；任一可测组低于 `97%`、且所有可测组都不高于 `105%`、同时没有其他非 PSF 拒绝时，Stage 9 才从不可变可信星层依次尝试 `+1 px`、必要时才尝试 `+2 px` 的源确认星翼。统一补翼达到全局上限后，会再按同星测量只选择 FWHM 比例低于 `1.08` 的未饱和星，保持严格星核不动，仅在真实源残差外翼内以最多 `+2 px`、`1.15×` 的受控候选补足分布低端；新增外翼按实际 Stage 8 底图反解 Screen 幅度，并钳在当前局部半高线的 `90%` 以下，增加可见外径而不扩大半高核心。若唯一硬失败是某个 weak/bright/all 组 FWHM 高于 `1.10`，则从不可变父星层只重建失败组，以 RGB 共用的 `u^(gamma-1)` 增益在 `1.0–2.5` 内二分收紧亚峰值翼部；峰值和通道比例保持不变，质心漂移超限的组件不处理。每个候选仍重跑恢复率、颜色、背景、亮星及 `0.93–1.10` PSF 全部门禁，选择最接近源 PSF 的正式候选；全失败恢复精确父状态，不放宽 FWHM 门。参考子组样本不足时保留含星候选但整轮转为 `result_review*`，候选已有足够参考却丢失可测星时仍硬拒绝。最终合成模式仍为 `screen`，不会改变 Stage 6 线性分离或 Stage 10 交付契约。

Stage 9 的 starmask “轻度拉伸”不由固定 Asinh `k` 定义，也不加入 LHE：普通星场按支持像素 `P50/P75/P90/P99.7`、混合星场按冻结星峰 `P40/P80/P90/P99.7` 实测输出，并分别受现有 `0.26/0.50/0.75/0.90` 上限及变化覆盖预算约束。内置 Asinh 的强度由这些上限反解，多锚点目标在必要时统一缩放；插件星层以实际输出执行同一硬门，无法测量或超限时在首次回混前拒绝。

Stage 9 默认直接运行随包的 `NB to RGB Stars` 和 `SASP Star Stretch` 无头适配器，不需要开启 `STARUN_WORKFLOW_PLUGIN_PROBE`。前者只处理已由 Stage 4 确认的 `R→Ha、G/B→OIII` 双窄带星层；普通 RGB、映射不足或缺少随包资源时安全跳过。后者随后进行唯一的插件星点拉伸；运行参数、使用的 wheel、映射验证和回滚结果会写入 `stage9_remix_quality.json`。可用 `STARUN_STAGE9_NB_TO_RGB_STARS_ENABLE`、`STARUN_STAGE9_NB_TO_RGB_STARS_RATIO`、`STARUN_STAGE9_SASP_STAR_STRETCH_ENABLE`、`STARUN_STAGE9_SASP_STAR_STRETCH_AMOUNT` 调整。

Stage 3 不再用 `gradient_score / dirty_background_score` 的固定经验阈值决定跳过或执行；两者仍参与候选充分性、排序和后续阶段自适应。Stage 3 先确认输入仍为线性数据，再建立显式源掩膜与无覆盖掩膜，从跨视场的真实天空区域选样，并冻结独立的拟合/留出集合；方向性平面优先解释为加性光污染并使用减法，径向形状只附带 master-flat/标定 advisory，条带/walking noise 则路由到标定或重叠加。获得执行授权后，默认按目标感知顺序评估本阶段全部内置 Polynomial/RBF 候选；若没有“无告警且足够干净”的候选，再依次升级到复合 Polynomial→RBF、GraXpert、ADBE、DBE、AutoDBE。`stage3_gate_profile` 默认使用“输出优先”，也可选择“平衡”或“严格保真”：一般的 `3σ` 变化、改善不足和轻中度形态/背景异常只记软告警并降权，极端通量、形态、质心、结构、背景或图案异常以及无效/未变化像素才硬拒绝。候选即使未获得显著改善也会进入混合统计排序；无软告警候选优先，再按残余背景、通量、形态、质心、结构和图案噪声的 Pareto 密集秩选择。所有候选都从同一不可变基线开始；全部硬拒绝或最终复检出现硬异常时回滚 Stage 3 输入，保存规范输出并以 `degraded/review_required` 继续流水线。关闭插件回退会同时排除 GraXpert 与 ADBE/DBE/AutoDBE。

Stage 2 的第一方轮廓检测与场旋检测会先形成裁切共识：轮廓确认 `full_frame_is_valid` 而场旋检测仍要求裁切时，按 `crop_detector_conflict` 保留全画幅、禁止旧边缘扫描器继续裁切并要求复核；只有检测器结论兼容时才允许实际裁切。轮廓确认全幅且场旋没有显著边缘异常时也直接保留全画幅。只有主检测器不可用、报错或结论不充分时才进入旧检测器；旧候选侵入中心保护区会以同一冲突原因整体拒绝，不再钳成恰好 70% 画幅。场旋裁切最多自动执行两次，第一次后仅在残余仍满足亮度噪声、色噪和边缘连通证据时尝试第二次，无改善即回滚。报告与 Stage 2 checkpoint 会冻结各检测器结论、共识、候选框、拒绝原因、累计裁边和复核语义。

Stage 5 不会在任务中联网下载 GraXpert 对象反卷积模型。自动模式优先使用 Starun App 随包模型；随包模型缺失时读取本机 GraXpert 应用已安装的最新版模型，再尝试“处理参数…”里指定且可读的官方 `model.onnx` 文件。App 会把有效模型只读链接到隔离运行目录；全部无效或缺失时安全回退 Siril RL。“自动加速”允许 GraXpert 通过 ONNX Runtime 自动选择 CoreML 等可用执行设备，初始化或推理失败时由插件回退 CPU；“CPU 兼容模式”会显式禁用该硬件加速。GUI 分开显示“反卷积”和“降噪”：若 GraXpert 已完成而降噪因低噪声或配置关闭被跳过，反卷积结果仍保存为 `stage5_linear.fit` 并继续传给 Stage 6。

Stage 5 成功后会在当前 run 写出 `stage5_stage6_handoff.json`，绑定规范 `stage5_linear.fit` 的 SHA-256、run ID、线性状态、输入基线及反卷积/降噪完整性；验签 checkpoint 恢复也必须逐字节物化为当前 run 的规范文件。Stage 6 只接受这份 handoff 与当前文件，不读取 Stage 1–4、Stage 5 中间候选、`working.fit` 或旧目录同名文件。来源缺失、lineage 异常或 `stage6_input.fit` 保存失败时会在调用去星工具前终止 Stage 6–10，不生成 passthrough；Stage 顺序仍固定为 1–10。

Stage 6 会显式记录去星结果为 `accepted / target_bypass / rejected / tool_failed / review_required`。自动链只运行免费、离线、哈希锁定的 Zenith 固定 profile；SASP 仅在显式 `sasp_only` 策略下运行。SyQon pair 提交前会在统一 `[0,1]` 域检查 16 px 微块、128 px 区域和 worker tile 边界的纹理/传递残差；硬伪影只允许一次 `512/128、CPU、FP32` 重试，仍失败则恢复含星 Stage 6 输入并要求复核，不会用模糊修补伪影。低动态范围星罩若像素覆盖测量失效，会改用冻结 Stage 5 星表和固定孔径测量；两种证据都不可用时记录 `measurement_unavailable`，不再伪造覆盖率零。所有尝试、参数、指标和最终选择均写入 `stage6_syqon_exchange.json`。NGC 6910 与 M45 等星点主体继续进入保星 target-bypass。

Stage 7 的 Stage 6 linked preview 仍只是屏幕显示参考。默认 `stage7_rendition_intent=vivid_safe` 与 `auto_display90` 会在 A/B 外生成同一 GUI D 曲线的 `cand_display70/82/90` 梯度，并在冻结的 Stage 6 主体/背景 ROI 上统一测量可见度、主体跨度、饱和度中位数、微对比度和背景安全余量；Stage 7 只决定亮度/tone，不再生成 `cand_vivid_safe` 或修改主体色度，相关测量作为 Stage 8 预算证据。v7 排序由 hard gate 单独决定正式资格，再按严格亮核饱和状态、主体亮度下限、封顶亮度目标与效用，以及背景色度/色噪/斑驳、颜色向量、新增裁切、亮核和高频亮度噪声的连续安全效用确定候选；advisory 数量和旧 `risk_score` 只写报告，不参与选择。完整 v6 结果仍作为影子审计记录，但不控制交付。Display GUI 参考色度比值仍须 `<=1.05`，绝对硬上限为 `0.30`；全任务“放松/无限”档不缩放 Stage 7 的裁切、结构、色度或亮度噪声安全门。所有候选出现硬画质失败时 `formal_accepted=false`：兼容字段 `stage7_forced_delivery_enabled` 只允许保留严重度最低的失败候选作为 `delivery_class=review_only` 诊断证据，Stage 8 从冻结 Stage 6 含星输入生成安全预览，Stage 9 不执行无星重混，Stage 10 只输出 `result_review*`。

Stage 4 的网络端点、Gaia 能力和 `spcc_list` 预检只提供审计证据，不会替代一次真实 Siril 命令；只有显式关闭联网/Stage 4、输入不适用或同应用会话已经发生真实 SPCC 超时，才会跳过相应尝试。SPCC/PCC 成功后，旧色偏、亮核、增益、窄带信号保留和精度指标仍写入报告，但仅为 `advisory_only`，不改写、回滚或切换成功候选；准备失败、命令失败/超时、候选缺失或不可读、尺寸变化、非有限像素及写回/保存验证失败才触发下一路由。

Stage 4 未接受 SPCC/PCC 物理校色时，Stage 7 会在排除主体、星云及可识别银河盘面信号后独立检查背景绝对色偏，Stage 10 对最终候选复测；`background_chroma_load` 默认超过 `0.12` 即要求复核。只有确认线性的宽带 RGB/OSC 可从不可变 `stage4_pre_pcc.fit` 运行自动区域校色：先按亮度、纹理和梯度选一个避开主体/恒星的背景矩形并做只削减偏高通道的全局加性平衡，再从亮度、结构、饱和度和污染证据选择一个白参考矩形，以空间分离的拟合/留出对象估计最大 `1.10×` 的全局通道增益并再次中和背景。白参考拒绝时保留合格的背景平衡；背景也拒绝或写回失败时精确恢复输入，不再运行旧星点软遮罩补偿。两种自动结果固定为非物理色彩并强制 `result_review*`；单色、非线性、未知通道和窄带继续保色。

该回退不自动操作 Siril 手工校色对话框。[Siril 手动校色](https://siril.readthedocs.io/en/stable/processing/color-calibration/manual.html)所述背景/白参考流程与依赖 Gaia 光谱及设备响应的 [SPCC](https://siril.readthedocs.io/en/latest/processing/color-calibration/spcc.html) 不是同一证据等级，因此自动局部参考报告始终披露“非物理色彩、需复核”。

亮发射/反射星云的 Stage 8 halo 门分为三级：不超过 `0.350` 正常增强，`0.350–0.600` 生成饱和度受限、无锐化的 masked 候选并由质量门决定，超过 `0.600` 或命中硬失败则安全旁路。受限候选保留为 `stage8_limited_candidate.fit`；拒绝时自动回滚。Stage 9 会把“使用 Stage 8 安全旁路源”与“Stage 9 自己使用降低强度/compact recovery 回退”分开显示。

Stage 4 会在任何颜色处理前只解析一次 FITS 设备/滤镜字段和显式用户提示，生成 `starun.narrowband-channel-mapping.v1` 并写入 `stage4_channel_mapping.json`。`FILTER` 是首要且权威字段：其值可分类时忽略其他滤镜字段；仅在缺失或完全无法识别时才按白名单回退到 `FILTER1/FILTER2/INSFLNAM/FILTERNAME/FILTNAME/FILTNAM`。受控 Seestar/DWARF 设备、明确 Ha/OIII、受控第三方滤镜和通用 `Dual-Band`/`Duo-Band` 分别保留不同证据等级；裸 `LP/LP_Starless` 使用 `0.86 / authoritative_filter_field_hint`，单窄带、SII/多窄带冲突及广角光路保持 `unknown`。SPCC、Stage 8、Stage 5 续跑和最终清单只复用这份冻结结论，不从后续 Header 重新猜测；Stage 4 不再生成 HOO 艺术像素，旧断点缺失映射时保留像素但进入 `resume_mapping_missing` 复核路径。

双窄带哈勃艺术色在 Stage 8 默认开启，可用 `STARUN_STAGE8_DUALBAND_PALETTE_ENABLE=0` 关闭。冻结契约中的通道必须已确认为 Ha/OIII，且 Stage 7 Starless 已验收、Stage 8 处于 `full/ok`；Stage 4 使用降级 PCC 基础校色不会阻止映射，但原有的 review-only 标记仍保留。`stage8_target_aware_chroma_enabled` 默认启用，只在 Stage 8 `auto/full` 且星云饱和度子步骤开启时对冻结主体 ROI 做保亮度、保余量的低频增色；`limited/background_only/preserve` 均旁路该正向增色。结构增强使用独立 `structure_scale`，因此窄带全局 saturation 为零时，受 mask 保护的暗弱星云提升和局部对比仍可执行；高背景噪声、亮目标和低 mask 覆盖分别降权，limited 模式强制归零。自动调色按“目标首选、HOO、其余映射”确定性生成六种候选，全部先通过亮度漂移、裁剪、背景和作用域硬门，再要求同一主体 mask 的主体/背景色度分离增益大于 `1e-4`；选择时依次比较色度分离增益、主体色度中位数增益、亮度漂移、裁剪增长和候选顺序。手动色盘仍只运行指定候选，新色度指标只记诊断、不推翻用户选择；自动候选全失败时恢复调色前结构结果。六套色盘统一使用上游脚本的常规 `Classic` 公式：`H=R`、`O=(G+B)/2`、`S*=(H+O)/2`，其中 `S*` 仅为合成代理。亮度漂移和裁剪增长超过名义门限但不超过 50% 时仍接受候选并只写提醒；超出该宽限、背景变化或 mask 外变化则拒绝该候选。

产品流程固定在 Stage 10 结束且仅包含本地处理算法；联网 AI、Stage 11、凭据注入和艺术衍生入口均已移除。运行日志与阶段事件只接受 Stage 1-10。Stage 状态为 `ok / degraded / failed / skipped`；最终运行状态为 `success / partial_success / review_required / failed`。阶段状态由结构化执行、回退和旁路字段决定，不再根据日志说明文字中的 `skipped/fallback` 关键词猜测。

详细阶段顺序、检查点、质量门控和降级路径见 `pipeline/starun_workflow.md`。

对压缩线性底座的 `large_galaxy`，Stage 7 Display 梯度在同一认证 LUT 内改用有界全画幅亮度分位数锚点，限制背景跨度并连续保留高光尾部，避免以 GUI P99.8 白点硬截断盘面或核心；RGB 仍按逐像素统一增益缩放。

## GUI Usage

1. 启动 `Starun.app`。首次启动或没有有效输入时会显示拖放空状态；已保存且仍存在的输入会直接进入任务设置。
   界面使用系统字体并跟随 macOS 浅色/深色外观；控件、状态与焦点样式来自同一套语义设计令牌。
   主窗口会恢复上次的正常位置、尺寸与最大化选择，但不会以最小化或全屏幕状态启动；若原显示器已移除，会自动回到当前显示器的可用区域。历史记录和设置等辅助窗口不会随启动自动重开。
   原生菜单按“文件 / 编辑 / 显示 / 处理 / 窗口 / 帮助”组织；工具栏与菜单复用同一命令和启用状态。`Command-O` 打开图像，`Command-Shift-O` 打开文件夹，`Command-Y` 激活历史记录窗口，“打开最近使用”与任务页最近输入保持同步；预览缩放、详细日志、全屏幕和标准编辑操作也可从菜单或系统快捷键调用。
2. 直接拖入一种输入：单个 `.fit/.fits/.fts/.xisf` 母版、包含 Light 的目录，或应用生成的产品任务目录。选择单个 FITS 后，任务设置会只扫描主 Header（不读取图像像素），显示设备、滤镜、单帧曝光，并按实际存在字段补充目标、拍摄时间、叠加数、图像尺寸/格式、Binning、增益、温度、焦距、像元和中心坐标等信息；字段缺失、Header 不可识别或当前为 XISF/复核图像都不会阻止处理。单文件始终从 Stage 1 只读导入；XISF 会先显示明确的等待提示，并在 Stage 1 转为任务内 FITS 后出现预览。Light 会递归发现，并按目标、滤镜、相机和几何尺寸拆为独立任务；Dark/Flat/Bias 首版不参与叠加。TIFF/PNG/JPEG 只进入复核路径并可直接预览。旧处理目录、根目录 checkpoint 和仅有结果文件的目录不再迁移或续跑；旧任务必须把原始母版作为新任务从 Stage 1 重新导入。
3. 应用只显示一个自动处理计划，不提供手动阶段选择。只有已验签 `run-manifest.json`、`checkpoint-manifest.json`、v2 阶段契约、累计配置指纹和产物 SHA-256 全部匹配的产品任务，才可能从 Stage 1、2 或 5 后继续；普通目录和 `result_linear.fit` 文件名本身不能证明断点。任务页可左右拖动“输入与处理计划 / 输入预览”分隔条；展开“处理参数…”后，也可上下拖动分隔条调整输入区与参数区高度，应用会恢复上次比例。macOS 的“Starun > 设置…”（`Command-,`）会在主窗口上方居中打开紧凑、不可缩放的独立窗口，维护联网、保留中间文件、完成后断点收敛、输出、正式/复核和计算设备等应用默认值；工具栏“任务选项”和任务页“处理参数…”只处理当前任务。处理参数顶部“通用配置”保存输出、审查、计算设备、自动调参、重试上限/间隔、复核包、受管理输出和 `checkpoint_mode`，并与应用默认值同步及冻结进任务清单；其后提供只缩放 Stage 2–9 已登记数值门禁的全任务“门禁策略”，下方按 Stage 1–10 单开分组，首次默认展开 Stage 2。Stage 4 普通配置可选择“自动区域校色（非物理色彩，需复核）”或“保留输入颜色（需复核）”；专家可关闭默认启用的白参考，只保留背景平衡。Stage 4 的通道增益、裁剪增长、星色温、背景色差和主体漂移只属于“诊断阈值”，可按任务覆盖但不参与候选路由，也不随门禁档位缩放。旧 `stage4_pcc_quality_gate_enabled` 不再显示，兼容值只影响报告指标。默认档位使用代码中的静态门禁默认值；放松模式将显式登记的数值验收阈值按宽严方向放宽 `3×`，无限模式放宽 `10×` 并临时强制本轮输出待复核。Stage 1、Stage 10 和最终交付安全门不参与放宽；布尔门、算法目标、采样/重试设置、Stage 4 advisory 诊断阈值和结构性安全检查也不参与缩放。专家模式中的登记门禁会只读显示档位派生后的真实值；取消“跟随档位”后可在原专家安全范围内逐项覆盖，且专家覆盖优先。档位和 Stage 1–10 覆盖仅属于当前输入，切换输入时恢复默认档位与自动参数；通用配置继续保留。批量任务在开始时冻结同一份签名快照，运行期间设置窗口保持可查看但禁止修改；同一新队列按“源文件指纹＋完整规范参数哈希”跳过完全重复项，不同参数和手动重新运行不去重。首个真实 `online_unverified` Gaia SPCC 超时后，本队列余下任务会跳过在线 SPCC，但继续允许 localgaia、PCC 和离线降级；预检不可达不会触发该缓存，新队列自动复位。Stage 1/2/5 断点分别禁用已完成的 Stage 1、Stage 1–2、Stage 1–5，Stage 10 始终可配置。Stage 2/3/4/6 可显式选择安全保留模式，仍生成规范阶段产物和诊断，不改变 Stage 1–10 顺序。
   专家项按“执行策略 / 算法参数 / 过程门禁 / 质量验收 / 诊断阈值 / 回退与失败”折叠显示；打开专家参数时优先展开执行策略。Stage 2–10 均可选择失败处置：`自动安全回退` 继续原候选链，`恢复输入并复核` 在决定性失败后停止搜索并回载本阶段输入，`写诊断后终止任务` 在阶段报告落盘后严格停止。Stage 10 的恢复来源只能是已验证 Stage 9 含星源；没有“忽略门禁并强制接受”，必须含星、最终质量报告以及相关性、通量和核心裁剪等交付不变量始终不可关闭。
   各阶段还可限制实际执行后端或子步骤：Stage 1 可调整配准失败率的降级线；Stage 2 第一方场旋覆盖检测与显式基础裁边，Stage 3 GraXpert/内置背景链，Stage 4 PCC/窄带降级/Header 引导解析，Stage 5 降噪后端与保留线性结果，Stage 6 SyQon/SASP，Stage 7 A/B/Display90 拉伸候选与 Display90 强度，Stage 8 `auto/limited/background_only/preserve` 强度上限、五个结构/饱和度子步骤及主体低频增色开关，Stage 9 可信含星保留、可调 PSF 恢复软目标及回星重试阶梯，Stage 10 保留源、末端降噪/饱和度/质量修复开关与后端策略。确认 Ha/OIII 的双窄带任务可在 Stage 8“哈勃色方案”中保持“自动（按目标类型）”，或手动指定 `HSO/SHO/OSH/OHS/HOS/HOO`；手动方案只改变艺术色盘，不改变 Stage 4 的 `R→Ha、G/B→OIII` 物理通道结论，也不能绕过 Starless、full 路由和画质门。用户模式只能收紧自动安全路由，不能把上游的 limited/skip 强制升级为完整处理。
4. 点击开始后，应用在后台计算来源 SHA-256，界面仍可响应并可停止；校验完成后建立相邻的 `Starun/<task-id>/`。来源只记录绝对引用、大小和 SHA-256，不复制或改写；同一来源指纹复用同一任务，每次运行写入独立 `runs/<run-id>/`。多个 Light 分组严格串行执行。每个 run 会先创建自己的日志和 `run-state.json`，再进入 preflight 并写出 `runtime-capabilities.json`；串行任务不会继承上一轮日志路径。能力清单从当前正在运行的 App Bundle（开发时为显式资源覆盖层）和隔离 `runtime_home` 解析 Siril、配置模板、完整 Stage 1–10 流水线、Gaia astro、48 个 Gaia XP 分块及网络端点，不会读取开发目录中的 `release/Starun.app`。preflight 还会分别检查运行环境卷与任务卷的磁盘空间，需求按来源大小和阶段峰值估算。
   运行环境准备在后台执行，可用“停止”取消；依赖锁、Siril Python ABI 与 App
   版本未变化时会复用已就绪 runtime，不重复执行 `pip install`。
   SyQon 与 CosmicClarity 模型直接从 App 或离线资源包只读使用，不再复制到
   `runtime_home`。
5. 快速 preflight 通过后，界面立即切换为只读运行视图：左侧显示冻结后的本次配置，中央只显示最新一张可靠预览，右侧完整列出 Stage 1-10。每个成功或降级阶段完成后刷新中央预览；跳过、失败或预览生成失败时保留上一张。当前运行阶段与预览来源阶段分开标注。
   运行视图采用稳定的“任务信息侧边栏 / 主预览 / 检查器”三栏结构：左栏只保留输入、任务目录、处理方式和结果操作；完整冻结配置移入右侧检查器的“任务”页，“阶段”页继续显示 Stage 1–10。工具栏或“显示”菜单可独立切换两侧栏（`Control-Command-S` / `Option-Command-I`），宽度、显示状态和检查器页会恢复；缩窄窗口不会擅自改变用户选择。
   Stage 0 和 Stage 1-6 会使用明确标注的链接屏幕拉伸，仅用于让线性数据快速可见，不写入 FITS 或改变流水线数据。Stage 7-10 显示阶段本身已有的非线性结果，不附加额外拉伸。中央预览支持适合窗口、1:1、滚轮缩放与拖动。
   阶段列表显示等待、运行中、已完成、已降级、失败或已跳过，并显示本阶段耗时和总耗时；不提供百分比或预计剩余时间。
   GUI 会按结构化终态分别显示“降级完成”“已使用回退”“需要复核”或“失败”；需要质量检查时可从提示卡片直接打开最终质量报告。
6. 完成、失败或停止后仍停留在运行视图，可返回任务设置、按上次实际配置重新运行或打开结果目录；“打开结果”会打开 `latest-result.json` 指向且产物 SHA-256 仍有效的实际 run。失败会自动展开日志。Stage 1、2、5 一经线性状态和阶段结果验收，就会在正常清理前原子发布到任务级 `checkpoints/`，因此后续阶段失败仍可从最近可信断点继续。新结果发布后，保留策略只删除已签名旧 run 中哈希仍匹配的旧交付和中间目录；最新交付、当前质量报告、日志、正式断点与未登记文件保留。每个旧 run 是否保留 `process/` 只读取该轮冻结的“保留中间文件”设置，不受后来运行的开关反向影响。
7. 工具栏或“窗口 > 历史记录”会打开单实例、非模态的独立历史窗口，按任务分组显示新版中实际开始的运行，可按目标/输入名称搜索并按结果状态筛选；窗口位置和尺寸会恢复，关闭后再次打开仍保留本次会话的搜索状态。选择运行后可点“查看运行”、按 Return 或双击，在主窗口复用只读处理视图显示仍可验证的阶段、预览和日志；“重新处理”只返回任务设置，不会直接执行。处理期间仍可浏览和搜索历史，但查看详情与删除会锁定。旧任务不会自动扫描或补录。空闲时可将一个验签通过且位于 `Starun` 直接子目录下的完整任务移到 macOS 废纸篓；容器目录、符号链接、清单损坏或 `task_id` 不匹配的目标一律拒绝。

## Outputs

- `*.tif`：高质量结果
- `*.png`：预览；Stage 7 已验收的非线性结果不再二次拉伸，诊断回退源才使用 linked autostretch
- `*_final.fit`：归档 FITS
- `*_display_srgb.png`：独立 16-bit 展示版，写入 `sRGB/gAMA/cHRM`，并对最终解码像素复核亮度、宽延展主体（星系/星云）与必需星点可见度；源画面过暗时只对该展示衍生图应用 linked 显示映射，复核仍失败则不以 display 名称发布，不改动 Siril PNG、可编辑 TIFF 或 FITS
- `*_edit_srgb.tif`：独立 16-bit 可编辑版，内嵌 sRGB ICC；找不到有效 ICC 时不生成未标记替代品
- `process/managed_output_report.json` / `output_color_manifest.json`：记录受管理导出的映射前/成品 PNG 可见度指标、容器元数据和科学 FITS 前后 SHA-256 不变性
- FITS 头完整时输出名包含目标、叠加数、曝光和拍摄时间；仅缺少 `STACKCNT` 等次要字段时，仍使用已有目标与拍摄时间生成可识别名称，避免退回通用名称覆盖其他任务
- `result_linear.fit`：Stage 5 线性交付文件；只作为交付产物，续跑必须使用已验签产品任务中的 Stage 5 正式 checkpoint
- `Starun/<task-id>/task-manifest.json`：来源只读引用、内容指纹与任务布局
- `Starun/<task-id>/runs/<run-id>/`：每次运行独立的计划、日志、阶段文件与结果
- `Starun/<task-id>/runs/<run-id>/run-manifest.json`：验签的来源、断点指纹、当前新写的 `starun.processing-parameters.v7` 任务参数快照及逐项门禁档位审计；历史 v4/v5/v6 仅在验签后兼容读取。v6 显式 `stage7_vivid_subject_chroma_enabled` 迁移为 Stage 8 的 `stage8_target_aware_chroma_enabled`，旧 Stage 4 HOO 像素参数被剥离并留迁移记录；v7 中继续携带这些旧字段、v1–v3、未知 schema、扁平载荷和未知字段都会在预检时报错，专家越界值与档位物理截断都会记录
- `Starun/<task-id>/runs/<run-id>/run-state.json`：在任何 preflight 之前创建并原子更新的 `starun.run-state.v2` 状态；终态与验签后的 pipeline result 合并，统一记录 `had_errors / had_fatal_errors / had_degradations / had_fallbacks / review_required / issues`
- `Starun/<task-id>/runs/<run-id>/runtime-capabilities.json`：本轮实际 App/runtime 路径和 Siril、模板、流水线、Gaia 本地目录、网络端点探测证据
- `stage4_auto_local_reference.json`：Stage 4 自动背景/白参考矩形坐标与评分、fit/holdout 指标、偏移/增益、独立安全门和检查点回滚证据；出现自动参考结果时固定披露“非物理色彩、需复核”
- `Starun/<task-id>/checkpoints/{stage1_prepared,stage2_corrected,stage5_linear}.fit`：仅有的正式跨运行断点；由 `checkpoint-manifest.json` 记录契约、配置指纹和 SHA-256，Stage 2/5 还携带验签的 v2 复核与上游语义
- `Starun/<task-id>/results/latest-result.json`：最新成功/需复核 run 的签名索引；每个交付文件再次按 SHA-256 校验后才在 GUI 中打开
- `Starun/<task-id>/results/retention.json`：记录已清理的旧交付/中间目录和因校验失败而保留的项目
- `~/Library/Application Support/Starun/history-index.json`：GUI 原子维护的全局历史导航索引；只登记新版实际启动的 run，不能替代任务/运行清单验签，也不能单独授权删除文件
- `result_review*.tif/png/fit`：Stage 2 视场残留或 Stage 3 背景要求复核、Stage 4 物理校色要求复核、target-bypass 的 Stage 7 拉伸未验收、Stage 9 弱/亮 PSF 子组证据不足、最终源降级/不可确认、`stage10_final.fit` 或最终质量门不可用、质量门要求保守重跑，或显式设置 `STARUN_FORCE_REVIEW_ONLY_OUTPUT=1` 时生成的复核产物；此时不会写普通 `result_processed/result_final` 名称。在最终降噪前已知的复核条件会直接跳过耗时降噪链
- Stage 7 未正式接受时没有普通命名例外；兼容的强制交付开关只保留最佳失败候选供诊断，后续固定进入含星复核路线并使用 `result_review*` 或 withheld
- 动态文件名只接受完整且有效的目标、叠加数、曝光和日期元数据；未解析的 `$...$`/printf 模板、unknown/null/n/a、非法日期及非正曝光会被拒绝，并依次回退到高置信目标身份、源文件/任务名或通用安全名称
- `process/stage*.fit`：`checkpoint_mode=false` 时仍由“保留中间文件”决定；`checkpoint_mode=true` 时只有 Stage 10 与交付、计划和断点全部验证通过才收敛，产品任务删除 `process/` 非关键 FITS 并保留任务级 Stage 1/2/5 正式断点，独立运行仅保留实际存在的规范 Stage 1/2/5 checkpoint。失败或验证失败会完整保留现场
- `process/*.json`：轻量阶段指标与诊断报告，默认保留
- `process/final_quality_report.json`：Stage 10 v2 最终质量报告；区分正常、可正式出图的软告警与必须复核的硬拒绝，并记录冻结背景采样、相对输入增长及一次受控修复/回滚结果；降级完成时可从 GUI 警告卡片打开
- `processing-plan.json`：唯一的 `starun.processing-plan.v2`，变换前经路由、阶段契约和计划哈希验证后原子冻结；确定性 `planned_steps` 与 GUI 同源，目标、通道、候选、参数调整和最终配置位于 `metadata`
- `pipeline-result.json`：发布前再次验证 v2 plan、计划哈希与 result→plan 引用；最新结果索引和旧 run 保留策略只接受该已验签关系
- `process/stage4_channel_mapping.json`：Stage 4 单次冻结的 Ha/OIII 通道映射契约；`channel_profile`、校色报告、Stage 5 断点和 Stage 8 只保存或消费同一结论
- `pipeline-result.json`：原子写出的 `starun.pipeline-result.v2` 最终运行清单，包含计划哈希、阶段结果、结构化复核/issue、五个统一状态布尔值、产物 SHA-256 和冻结窄带通道映射；读取器验签后兼容归一化历史 v1，旧文件不会被改写
- `process/review_bundles/<stage>/review.json`：默认保留 before/after 指标、候选和复核结论；成功阶段结束后删除视觉图片，`partial_success / review_required / failed` 只为异常阶段保留一张最大约 1024 px 的 8-bit `review.png` 四宫格
- `starun_diagnostics.zip`：只归档 JSON/JSONL、日志、CSV/TXT 等轻量文本诊断，不再复制 review/UI PNG 或 FITS
- `process/*_quality_metrics.json`、`process/stage_quality_metrics.jsonl`：保留中间文件时输出的统一阶段质量指标，日志同步输出 `[STAGE_QUALITY_METRICS] schema=starun.stage_quality.v1 ...`

完整中间文件清单见 `pipeline/starun_workflow.md` 的“文件与目录行为”。

## Development Run

无需打包即可用系统 Siril 和已初始化的 Siril Python seed 启动源码 GUI：

```bash
cd /Users/mz/dev/aiseestart
.venv/bin/python gui/starun_gui_dev.py \
  --siril-app /Applications/Siril.app \
  --siril-seed "$HOME/Library/Application Support/org.siril.Siril/siril"
```

两项参数均提供以上默认值，可直接运行
`.venv/bin/python gui/starun_gui_dev.py`。launcher 会在临时目录创建资源覆盖层，
通过符号链接把系统 Siril、现有 seed 与项目 `resources/` 组合后交给原 GUI；
launcher 本身不会修改这些来源，也不会改变正式 App 入口。默认仍使用
`~/Library/Application Support/Starun/runtime_home`，可通过
`--runtime-home /absolute/path` 指定独立开发 runtime。

只验证核心流水线输出、不启动 GUI 时：

```bash
cd /Users/mz/dev/aiseestart
.venv/bin/python -u -m gui.starun_pipeline_dev \
  --work-dir "$HOME/SeeStar/sirildev" \
  --input-mode auto
```

该命令默认允许在线 Gaia；明确要求纯离线时添加 `--offline`，此时需要先准备本地 Gaia 星表。

核心代码调整后的人工快速回归可使用专项脚本：

```bash
cd /Users/mz/dev/aiseestart
.venv/bin/python tests/manual_core_pipeline_smoke.py
```

脚本只从原始输入开始完整运行 Stage 1–10；续跑必须由 GUI 从验签产品任务发起，
不能把工作目录中的 checkpoint 或 `result_linear.fit` 作为入口。脚本复用上述无 GUI launcher，默认开启网络并保留调试中间文件，以便在
本地 Gaia 目录缺失时继续使用在线 Gaia 完成 Stage 4；完成后还会
校验本轮 `processing-plan.json`、`pipeline-result.json`、清单哈希和登记产物的 SHA-256。可直接执行：

```bash
.venv/bin/python tests/manual_core_pipeline_smoke.py \
  --mode auto \
  --work-dir "$HOME/SeeStar/sirildev"
```

使用 `--dry-run` 可仅检查最终命令，`--no-debug` 可减少中间文件；需要验证严格
离线回退链路时添加 `--offline`。该脚本验证当前仓库核心源码，不替代发布前的
GUI 参数传递、打包资源和签名验证。

## Build

开发期需要验证真实 Cocoa 窗口、包内资源和前台启动时，可使用 Codex Run 按钮或：

```bash
./script/build_and_run.sh --verify
```

该入口会停止已有的开发实例、构建到被 Git 忽略的 `dist/Starun.app`，再通过
Finder 启动并确认进程保持运行；不会替换 `release/Starun.app`。

```bash
cd /Users/mz/dev/aiseestart
python3.13 -m venv build/.venv
build/.venv/bin/python -m pip install --require-hashes \
  -r build/requirements-gui-build.lock
./build/build_macos_app.sh
```

`build/.venv` 是仅用于冻结 GUI 的最小构建环境。打包脚本优先使用它，并从
Python 3.13 GUI 中排除只会由 Siril Python 3.12 离线运行时使用的
Torch、ONNX Runtime、SciPy 等模块；`siril_plugins/downloads`、模型、Siril
脚本和 `SirilPythonSeed` 仍完整保留。需要使用其他等价环境时可传
`--build-python /path/to/python`，构建后的依赖边界检查仍会拒绝重复收集。

Required package:
- `packages/siril-1.4.4-arm64-3.dmg`（内置 Siril CPython 3.12 runtime）

Optional bundled resources:
- `resources/siril_plugins/`（可用 `bash resources/siril_plugins/download_siril_plugins.sh` 准备）

Default output: `release/Starun.app`

构建脚本强制生成纯 `arm64` 主程序，并把 `LSMinimumSystemVersion=14.0`
写入 App 的 `Info.plist`；构建机必须是 Apple Silicon Mac。当前 Bundle 元数据为
`CFBundleIdentifier=StarunC`、`CFBundleShortVersionString=0.1`、
`CFBundleVersion=1`，运行时依赖缓存会使用 `0.1 (1)` 作为 App 版本指纹。

公开发行时建议传入固定签名身份：

```bash
./build/build_macos_app.sh \
  --codesign-identity "Developer ID Application: Your Name (TEAMID)"
```

默认 ad-hoc 签名只适合本地验证。公证所需的 hardened runtime、timestamp 和 notarization 仍需单独配置。

默认构建仍为 Full Offline。也可把大型 wheels/模型拆为相邻的离线资源包：

```bash
./build/build_macos_app.sh --bundle-profile core
```

该命令生成 `Starun.app` 和
`Starun-OfflineResources/`；两者保持在同一目录即可自动发现。也可用
`--offline-resource-pack-dir` 指定资源包位置，或通过
`STARUN_OFFLINE_RESOURCE_ROOT` 指向已安装的资源包。

依赖更新后，GUI 构建锁和 GUI/App 锁仍使用 Python 3.13；Siril 插件锁必须使用与内置 Siril 一致的 Python 3.12：

```bash
python3.13 -m pip install -r requirements-dev.txt
PIP_CONFIG_FILE=/dev/null python3.13 -m piptools compile \
  --resolver=backtracking --strip-extras --allow-unsafe --generate-hashes \
  --index-url https://pypi.org/simple \
  --output-file build/requirements-gui-build.lock \
  build/requirements-gui-build.in
PIP_CONFIG_FILE=/dev/null python3.13 -m piptools compile \
  --resolver=backtracking --strip-extras --allow-unsafe --generate-hashes \
  --index-url https://pypi.org/simple \
  --output-file requirements.lock requirements.txt
PIP_CONFIG_FILE=/dev/null python3.12 -m piptools compile \
  --resolver=backtracking --strip-extras --allow-unsafe --generate-hashes \
  --index-url https://pypi.org/simple \
  --constraint resources/siril_plugins/requirements-macos-arm64.lock \
  --output-file resources/siril_plugins/requirements.lock \
  resources/siril_plugins/requirements.txt
```

## Common Env

项目会自动读取 `resources/default.env`；若存在额外的 runtime env 文件，也只叠加当前产品处理参数。GUI 上的 Debug、处理模式和“处理参数…”任务快照优先；旧 AI、凭据和兼容变量不在允许列表中。

| Variable | Purpose |
|---|---|
| `STARUN_NETWORK_MODE` | GUI 默认 `1`，Stage 4 默认在线请求 Gaia；显式设为 `0` 时只使用已验证的本地 Gaia，目录不足则按 `stage4_offline_fallback_mode` 继续并强制复核 |
| `STARUN_DEBUG_MODE` / `STARUN_INPUT_MODE` | GUI Debug 开关与自动冻结的内部路由；任务验签后可用 `stage1_prepared_resume`、`stage2_corrected_resume`、`stage5_linear_resume`，主界面不提供手动起点选择 |
| `STARUN_OUTPUT_FORMAT` | 最终导出格式：`all`（默认）或逗号分隔 `tif,png,fit` |
| `STARUN_STAGE10_MANAGED_OUTPUT_ENABLE` | 默认 `1`；为所请求的 PNG/TIFF 生成独立 sRGB/ICC 衍生文件，FITS 永不重写 |
| `STARUN_FORCE_REVIEW_ONLY_OUTPUT` | 默认 `0`；设为 `1` 时仅导出 `result_review*`，适合人工复核而非正式交付 |
| `STARUN_WATCHDOG_IDLE_TIMEOUT_SEC` / `STARUN_EXPORT_TAIL_TIMEOUT_SEC` | GUI 无输出 watchdog；普通阶段默认 900 秒，确认本轮 PNG 产物后的收尾默认 120 秒；超时会记录最后命令、进程状态和产物 |
| `STARUN_STAGE7_*` | 阶段 6 冻结质量阈值和事务式修复控制；SyQon 输入固定保持线性且不做质量参数盲扫；新任务写入 v7 参数注册表，历史 v4/v5/v6 仅兼容迁移 |
| `STARUN_STAGE9_*` | Stage 9 参考驱动星色修复、starmask Asinh、Screen 回混及保存前质量门控；同域 Unscreen 竞争参数通过签名任务参数传递 |
| `STARUN_STAGE4_AUTO_GEOMETRY_*` / `STARUN_STAGE4_PLATESOLVE_ENABLE` | 无冲突且高置信的设备/FITS 几何可驱动 platesolve；显式环境覆盖优先，解算后 WCS 比例超限即回滚并禁止 SPCC/PCC |
| `STARUN_SPCC_ENABLE` / `STARUN_STAGE4_SPCC_*` / `STARUN_STAGE4_PCC_TIMEOUT_SEC` | 宽带与确认 Ha/OIII 映射的双窄带都先做单次 Gaia DR3 SPCC 物理校色；Seestar S30/S30 Pro/S50 与 DWARF 2/3/mini 默认按 FITS 自动选择响应，显式覆盖优先；SPCC 异常后做单次 PCC，双窄带 PCC 明确标为非物理降级校色并强制复核 |
| `STARUN_STAGE4_NBN_MAPPING_CONFIDENCE_MIN` / `STARUN_STAGE4_FILTER_HINT` | Stage 4 单次生成 `stage4_channel_mapping.json` 并只冻结双窄带通道语义；艺术像素映射统一由 Stage 8 消费该契约，单色和不确认映射保色 |
| `STARUN_STAGE8_LOCAL_ADJUSTMENT_ENGINE_ENABLE` | 默认 `1`；启用版本化本地曲线、软蒙版、形态学与局部调整配方 |
| `STARUN_STAGE8_DUALBAND_PALETTE_ENABLE` / `STARUN_STAGE8_DUALBAND_PALETTE_*` | 双窄带哈勃色默认开启；可调混合强度、亮度/裁剪名义门限和默认 50% 的 warning-only 宽限 |
| `STARUN_ABERRATION_API_ENABLE` / `STARUN_ABERRATION_PROVIDER` | Stage 10 SASP Aberration API fallback；默认关闭 API，provider 可设 `cpu` |
| `STARUN_WORKFLOW_PLUGIN_PROBE` / `STARUN_SIRIL_PLUGIN_DIR` | 实验插件命令探测与插件目录 |

完整 pipeline env 说明见 `pipeline/starun_workflow.md`；打包/runtime env 见 `INTEGRATION_README.md`。

Stage 4 在确认线性的宽带 RGB/OSC 和双窄带输入上保存不可变的 `stage4_pre_pcc.fit`，然后以独立 Siril CLI 进程按 `SPCC → PCC` 各尝试一次。在 platesolve 或任何像素颜色操作前，它只生成一次 `stage4_channel_mapping.json`，不读取 `OBJECT` 猜滤镜；即使选择 `preserve` 也会写出该契约。大小写不敏感的 `FILTER` 可识别时拥有权威性，其他滤镜字段只作为 ignored evidence；`FILTER` 缺失或完全未知时才回退到 `FILTER1/FILTER2/INSFLNAM/FILTERNAME/FILTNAME/FILTNAM`，补充字段冲突则保持 `unknown`。默认使用在线 `-catalog=gaia`；显式关闭互联网时只使用预检通过的本地 `-catalog=localgaia`。在线端点、Gaia 能力和 `spcc_list` 预检只作 advisory，即使不可达也会执行真实命令，只有真实超时才影响后续任务。Seestar S30/S30 Pro/S50 与 DWARF 2/3/mini 的远摄天文帧会从 FITS 设备身份选择对应的 Siril 传感器、滤镜和几何；Header 中的焦距/像元仍优先，显式环境覆盖优先于自动画像。无法确定 SPCC 传感器或滤镜时记录 `command_preparation_failed` 并精确恢复后转 PCC，不复用 Siril GUI 历史状态。SPCC/PCC 命令成功且候选存在、可读、尺寸不变、像素有限并通过写回验证时直接采用；“解不精确”、色偏、亮核、增益、裁剪和窄带信号保留指标仍审计，但不再修改或拒绝候选。成功但低精度的 SPCC 不触发 PCC、回滚或 Stage 4 复核，只按目标策略将 Stage 8/10 后续饱和度预算减半。只有技术失败才精确回载检查点并转下一路由。

双窄带物理校色、降级校色和艺术映射是三种明确语义：只有接受的 SPCC 物理结果保存为 `stage4_physical_color.fit`；SPCC 失败后技术完整的 PCC 使用 `PCC_NARROWBAND_DEGRADED*` 方法名，质量与 Ha/OIII 信号保留指标仅作 advisory，只提供基础颜色矫正，不声明 `physical_color.accepted`，并固定 `requires_review=true`。Stage 4 只保存物理/PCC/保色父本与冻结通道映射，不再创建 `stage4_hoo_artistic.fit`；HOO/SHO 等艺术像素统一在 Stage 8 的 Starless 主体事务中生成，因此不会进入 Stage 5。降级 PCC 仍只能走 Stage 10 的 `result_review*`。

完全离线的 platesolve/PCC 共用 `runtime_home/.local/share/siril/siril_cat_healpix8_astro.dat`；该官方 Gaia DR3 HEALPix 目录含恒星 `Teff`，可作为 PCC 的星色参考。离线 SPCC 使用独立的 `runtime_home/.local/share/siril/siril_cat1_healpix8_xpsamp/` 分块目录（完整集接近 21 GB），不与前者混用。GUI 下载器目前只安装较小的 astrometric/PCC 目录，使用 Zenodo 官方文件、双 SHA-256 校验和原子替换；构建脚本拒绝把任一 Gaia 星表带入 App。App 打包固定提交并校验的 Seestar/DWARF 常见传感器、UV/IR/LP/Astro 滤镜与白参考 SPCC 元数据，不包含 Gaia 光谱星表。

## Limits

- 主要面向 Seestar、DWARF 智能望远镜数据与 Siril 1.4+。
- 阶段 6 不再依赖 StarNet；自动链只使用固定的离线 Zenith SyQon，SASP 仅允许显式选择。契约失败时保留含星复核路径；契约有效但质量失败时保留 limited/review pair。两者都不会伪造正式验收，最终只输出 `result_review*`。

## Tests

```bash
python3.13 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-dev.txt
.venv-test/bin/python -m pip install --require-hashes -r requirements.lock
.venv-test/bin/python -m pytest -q -rs
.venv-test/bin/python -m pytest -q -rs \
  --cov=gui --cov=pipeline --cov-branch \
  --cov-config=.coveragerc --cov-report=term-missing --cov-fail-under=65
```

重点覆盖 GUI/runtime 路由一致性、v2 计划与结果验签、断点收敛、pipeline fallback 和输入过滤。fallback 测试按 Stage 拆分在 `tests/test_pipeline_plugin_fallbacks_*.py`，共享 fake/helper 位于不参与收集的 `tests/pipeline_plugin_fallbacks_support.py`。

`.github/workflows/tests.yml` 在 push、pull request 和每周调度中运行完整单元测试并强制总体分支覆盖率不低于 65%。配置了 `starun-e2e` Apple Silicon 自托管 runner 后，每周调度或手动 workflow 还会运行真实 Siril 离线 Stage 1–10 回归。该任务必须使用已审阅、会实际进入 SyQon 的真实线性母版、完整离线 Gaia 目录，以及 `STARUN_OFFLINE_RESOURCE_ROOT` 指向的 runner 仓库外只读资源包；资源包须包含 SyQon Zenith 和 CosmicClarity mono/color denoise 模型，模型不由 Git/LFS 提供。只有输入、Siril/runtime、Gaia、SyQon 和 CosmicClarity 资源全部准备并人工核验后，才设置 `STARUN_REAL_E2E_READY=true`。

真实 E2E 的“通过”表示产物达到正式可交付条件，不等于进程退出成功。verifier 要求原始 `starun.pipeline-result.v2` 为 `success`、Stage 1–10 全部 `ok/completed`、无降级/回退/复核，`starun.final-quality.v2` 与正式星点交付通过，SyQon lineage 完整，并且本轮工作目录内存在非 review 的 `*_final.fit`。CI 的 pipeline 与模型推理按离线模式运行；`actions/setup-python` 和普通 Python 测试依赖安装仍可能联网，因此该 job 不声明为完全 air-gapped。
