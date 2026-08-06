# Stage 4 三维度深度审计报告：图像解析（Plate Solve）+ 色彩校准（SPCC/PCC）

> 审计对象：`pipeline/stages/stage4_color_calibration.py`（约 2872 行）及其依赖 `channel_semantics.py`、`narrowband_normalization.py`、`target_runtime.py`、`processor_runtime.py`、`policy_selector.py`、workflow 文档 `seestar_Superimpose_workflow.md`。
> 运行环境：Python + Siril 1.4.0（`requires 1.4.0`，见 `stage4_color_calibration.py:594`、`:759`）。
> 审计性质：纯研究 / 只读。**未修改任何项目代码**。
> 证据规则：代码结论带 `文件:行号`；行业结论带可访问 URL；不确定项显式标注「未验证：原因」。

---

## 0. 摘要与总评分

Stage 4 采用 **「先 plate solve → 再 SPCC 优先 → 失败回退 PCC（仅宽带）→ 再回退局部恒星软遮罩」** 的保守主路由，并对双窄带 HOO 做物理 SPCC 后隔离艺术派生（`feeds_main_pipeline=False`）。整体设计在**数据真实性保护、边界可控、可终止性、归档完整性**上达到生产级水准，且与 PixInsight / Siril 现行行业实践高度一致。

| 维度 | 评分（/10） | 一句话结论 |
|------|------------|-----------|
| D1 逻辑专业合理性 | 8.2 | 路由与前提校验保守合理；双窄带 SPCC 物理近似处理得当但质量门为宽带复用 |
| D2 门禁来源与归档 | 8.0 | 门禁齐全且有安全钳制；归档覆盖好，仅「意图层」未记录 SPCC 光谱标识 |
| D3 算法行业标准符合度 | 8.3 | SPCC 优先、线性域、plate solve 前置均与官方一致；OSC 传感器预设与窄带近似存未验证点 |
| **总评分** | **8.2（B+）** | 设计成熟，主要风险集中在「OSC 传感器光谱数据准确性（未验证）」与「窄带质量门复用」 |

**最严重问题（P1）**：`stage4_color_calibration.py:882-905` 将 OSC 传感器硬编码为 `Sony IMX585` 并直接传给 Siril SPCC 的 `oscsensor`。该标识的准确性决定整个校色光谱模型；**未验证：Siril 1.4 是否内置名为 "Sony IMX585" 的 OSC 预设、以及其合并响应曲线是否与 Seestar S50 实际 sensor+LP 滤镜一致**。若预设缺失/不匹配，SPCC 将用错误光谱模型产生系统性偏色，且当前质量门（增益比/背景色散）未必能捕获此类错误。

### 0.1 审计范围与方法
- **输入边界**：Stage 4 的输入为 `stage3_bgremoved`（线性、背景已去除），由 `run_stage4_color_calibration` 在 `:1928-1930` 载入；载入失败即 `hard_degraded` 并禁止所有 photometric 校准。
- **输出边界**：主产物 `stage4_psolved`（plate solve 后）、`stage4_color`（物理 SPCC 结果）、`stage4_hoo_artistic`（隔离艺术派生）；以及 `color_calibration_report.json`、`stage4_narrowband_normalization.json` 等归档。
- **方法**：只读审阅 + 三维度对照（逻辑 / 门禁归档 / 行业）。D3 通过 WebSearch 取证 PixInsight、Siril 官方文档与社区共识，URL 见第 6 节。

---

## 1. D1 — 逻辑专业合理性分析

### 1.1 Plate Solve 策略

- **候选枚举与顺序**（`_stage4_platesolve_variants`，`stage4_color_calibration.py:403-417`）：先按 catalog 列表（默认 `gaia`，由 `SEESTAR_STAGE4_PLATESOLVE_CATALOGS` 控制，`:347-357`）生成基础候选，再追加「header 中心坐标」候选（仅当 FITS 头含 `RA/DEC`、`OBJCTRA/OBJCTDEC` 或 `CRVAL1/CRVAL2` 时，`:374-400`）。两层候选覆盖了「已知目标坐标（hinted solve）」与「盲解（blind solve）」两类场景，符合 astrometry 常识。
- **方向保护**（`_stage4_platesolve_geometry_args`，`:302-319`）：始终带 `-noflip`。Siril 文档明确「除非 `-noflip`，检测到上下颠倒会翻转图像」（`https://siril.readthedocs.io/es/latest/astrometry/platesolving.html`）。保留方向对下游 starless/拉伸/合成的正确性至关重要，**正确**。算法层面，plate solve 只写入 WCS 元数据与（可能的）方向变换，不改变像素科学值——符合 AGENTS.md「保护数据真实性」。
- **几何参数注入**：`-focal`（默认 160mm，`DEFAULT_STAGE4_FOCAL_LENGTH_MM`，`:33`/`:290-291`）、`-pixelsize`（默认 2.90µm，`:294-295`）、`-order`（默认 3，`SIP` 畸变多项式阶，`:298-299`）。Seestar S50 焦距约 160mm、像素 2.9µm，参数合理；`order=3` 对广角镜畸变建模足够。
- **可终止边界**：`_stage4_run_platesolve`（`:434-476`）在执行期间临时将 `pipeline.cfg.max_retries` 置 0（`:436-438`、`:471-473`），禁止 Siril 内部自动重试，保证单次确定的求解行为；每个候选失败/跳过均记录 `attempts`（含 label/command/status/error），并由 `_stage4_platesolve_diagnostics`（`:479-520`）分类失败原因（目录服务不可用 / Siril 通用错误 / 星匹配失败）并给出下一步动作。**边界设计与审计可追溯性良好**。
- **catalog 选择逻辑**：`_stage4_platesolve_catalogs`（`:347-357`）默认 `gaia`（在线）；离线模式（`SEESTAR_NETWORK_MODE=0`）自动前置 `localgaia`（`:355-356`），确保离线优先本地星表。与 `_stage4_catalog_skip_reason`（`:420-431`）联动：localgaia 在本地目录缺失时跳过并给出路径/字节数；在线 catalog 在 `SEESTAR_NETWORK_MODE=0` 时跳过。**无悬空候选、无静默失败**。
- **失败后果**：plate solve 失败 → `pipeline.platesolve_ok=False` → SPCC/PCC 同时被禁用（`:2174`、`:2181`）→ 进入局部恒星白平衡或保色（`:2472-2506`）。优雅降级，**正确**。注意：无 WCS 时下游 stage5/6 仍可运行（starless/stretch 不依赖 WCS），但目标标注/合成对齐能力受限——属可接受的离线权衡。
- **轻微关注（非缺陷）**：hinted solve 仅依赖 FITS 头坐标；Seestar 原生输出头常含 `OBJECT` 但未必含精确 `CRVAL`。无头坐标时退化为纯盲解（`gaia` 在线 / `localgaia`），对稀疏天区可能失败。建议：在 workflow 文档中明确「离线盲解需本地 Gaia 星表覆盖目标纬度」。

### 1.2 SPCC 前提校验链路

- **线性域判定**（`_stage4_linearity`，`:965-981`）：优先读 FITS `LINEAR` 关键字（confidence 0.98），其次依赖 Stage3/4 线性检查点契约（confidence 0.96），未知则 confidence 0.0 并禁止。该判定为启发式（关键字 token 匹配 `nonlinear/non-linear/stretched/histogram`，`:966-971`）。
- **硬门控**：`checkpoint_loaded`（`:1928-1930`）在载入 `stage3_bgremoved` 失败时直接 `hard_degraded` 并禁止 SPCC/PCC（`:1932-1937`）。这是真正的「线性保证」，因为上一 stage 的 checkpoint 是已知线性产物。**双保险，符合「photometric 校准必须在线性未拉伸数据上进行」的硬约束（`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html` 警告段）**。
- **门控链路总览**：`spcc_allowed`（`:2171-2178`）= `physical_color_input ∧ spcc_runtime_enabled ∧ platesolve_ok ∧ pre_pcc_saved ∧ before_chw 可用 ∧ selected_spcc_catalog 非空`；`pcc_allowed`（`:2179-2185`）= `broadband_rgb_osc ∧ platesolve_ok ∧ pre_pcc_saved ∧ before_chw ∧ selected_pcc_catalog`。每个条件缺失都有对应 `spcc_skip_reason`/`pcc_skip_reason` 文案（`:2215-2223`、`:2374-2381`）。**链路严密、可诊断**。

### 1.3 Seestar OSC 光谱参数来源

- 参数来源（`_stage4_spcc_args`，`:863-962`）：
  - `common` 段（`:902-906`）：`-catalog={catalog}`、`-oscsensor={sensor}`、`-whiteref={white_ref}`。
  - sensor：`stage4_spcc_osc_sensor` 默认 `Sony IMX585`（`:882-885`，默认定义 `:35`）。
  - white_ref：默认 `Average Spiral Galaxy`（`:886-893`，默认 `:36`）。
  - osc_filter：经 `_stage4_effective_spcc_filter`（`:863-872`）解析——显式配置 > FITS/filter hint（含 `no_filter` 关键字回退 `No_filter`）> 默认 `ZWO Seestar LP`（`:37`）。
  - limitmag：默认 `10.5`（`:894-901`，默认 `:38`）。
- 行业对齐：Siril SPCC 对 OSC 相机要求在 Calibration 段选择「相机 sensor（含 Bayer CFA 的合并响应曲线）」+「滤镜」（`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html`）。代码传 `oscsensor`+`oscfilter` 正对应此模型，**与 Siril 的 OSC 建模一致**。
- **关键未验证点（P1）**：PixInsight 社区与官方指南强调 OSC 必须选「合并了 Bayer 矩阵的 sensor 响应」而非 mono「QE 曲线」（`https://www.pixinsight.com/forum/index.php?threads/is-this-the-correct-workflow-for-osc-dual-narrowband-spcc.19717/`、`https://www.pixinsight.com/forum/index.php?threads/choice-of-sensor-and-filter-for-a-osc-camera-in-spcc.21137/`）。Siril 用命名 OSC 预设承载该合并曲线——前提是 **"Sony IMX585" 必须确实是 Siril 1.4 的 OSC 预设且曲线正确**。未验证：Siril 1.4 安装包是否含该预设、曲线是否匹配 Seestar S50 实际 sensor。若预设名不匹配，SPCC 会直接报错或静默用近似曲线 → 系统性偏色，且增益比/背景门未必捕获。
- white_ref 选择：默认 `Average Spiral Galaxy` 是 PixInsight 与 Siril 的**共同默认**（`https://pixinsight.com/doc/docs/SPCC/SPCC.html`、`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html`），对星系目标科学正确；对强发射星云，社区有改用 G2V/光子通量或「整片星云作白参考」的做法（见 D3.4），但本 pipeline 通过目标感知质量门对发射星云做特殊豁免（`:1279-1381`），属可接受工程取舍。**注意**：代码未提供按目标类型切换 white_ref 的逻辑（始终用配置默认），对发射星云用星系白参考可能轻微偏冷——但质量门与后续拉伸可缓解，非阻断。
- limitmag 10.5：Siril 通常按 FOV 自动推算；显式 10.5 作为保守上限合理，避免稀疏场拉入过多暗星噪声。

### 1.4 单次执行合理性

- SPCC/PCC 均 **单次执行、不重试**：`max_attempts=1`（`:548`、`:695`），超时 `PCC_TIMEOUT_MIN_SEC..MAX_SEC` 钳制 5–180s（`:534`、`:681`）。理由（代码注释 `:672`、`:529`「behind a killable boundary」）：解算是确定性的，重复无收益；超时即转回退链。符合「可终止边界」原则（AGENTS.md Minimum Checks）。
- 执行载体：均通过独立 `siril-cli` 子进程 + 临时 `.ssf` 脚本（`:586-613` SPCC 变体 `:751-769`），`subprocess.run(..., timeout=timeout_sec)`，超时捕获 `TimeoutExpired` 后**不重试**（`:631-634`、`:796-799`）。与宿主 pipeline 进程隔离，避免 Siril 崩溃拖垮主流程。**隔离设计正确**。
- **轻微关注**：SPCC 因瞬态（catalog 抖动）失败时无重试直接落 PCC（精度更低，`:2384-2454`）。在弱网离线场景可能损失一次本可成功的更优校色。属可接受的稳健性/精度权衡，非缺陷；若需可加 1 次有限重试（见 P2-6）。

### 1.5 双窄带 SPCC 物理正确性

- 窄带分支（`_stage4_spcc_args`，`:907-949`）：`kind==narrowband_composite` 时传 `-narrowband` 与 R=Ha 656.28nm/20nm、G/B=OIII 500.70nm/30nm（默认值 `:926-931`，均带安全钳制 `:920-924`：波长 600–700(R)/450–550(GB)，带宽 1–100nm）。
- 映射置信门（`classify_dual_narrowband_mapping` + `:917-918`）：置信度 `< max(0.70, min(minimum,0.99))`（`minimum`=`stage4_nbn_mapping_confidence_min` 默认 0.85，`:914`）→ 抛 `ValueError` → SPCC preflight 拒绝（`color_warning="spcc_narrowband_metadata_unconfirmed"`，`:2234-2236`）→ 双窄带走 `PRESERVE_INPUT + requires_review`（`:2362-2371`）。**保守正确**：未确认的窄带映射绝不做物理 SPCC。
- 行业对齐：cloudynights 社区明确 HOO「true color like」用法即勾选 narrowband 模式、填入 Ha 656.3/带宽、OIII 500.7/带宽（`https://www.cloudynights.com/topic/932882-spcc-question-narrowband-filters-mode/`）；并强调 **SHO 为假彩色，SPCC 不适用，应走手动/其他**——本 pipeline 物理主路由仅做 HOO 真彩派生、SHO 不入主路由，**完全一致**。
- **物理近似性（固有，非缺陷）**：双窄带 OSC 的 R/G/B 通道含 Bayer 串扰（Ha 渗绿蓝、OIII 渗红），SPCC narrowband 模式假设各通道为纯窄带通量。社区公认此结果「接近真彩但非科学精确」（`https://www.cloudynights.com/forums/topic/983844-is-it-possible-to-completely-isolate-haoiii-from-dual-band-data/`）。本 pipeline 通过「隔离 HOO 艺术派生、物理 SPCC 结果留主路由」正确处理了这一近似，**设计优秀**（详见 1.6）。
- **未验证**：默认带宽（R=20nm、G/B=30nm）是否为 Seestar S50 双窄带滤镜真实 FWHM；配置可覆盖（`stage4_spcc_narrowband_*_wavelength_nm/_bandwidth_nm`），但默认值与实际滤镜规格的一致性未核对（U2）。

### 1.6 HOO 艺术派生隔离

- `_stage4_run_narrowband_normalization`（`:1728-1848`）+ 存档块（`:2508-2580`）：物理 SPCC 结果先存 `PHYSICAL_COLOR_STEM`（`:2509`，失败则 `hard_degraded`），随后仅由 `normalize_dual_narrowband_candidate` 派生 HOO 艺术图，并显式标记 `role=artistic_derivative`、`physical_parent=PHYSICAL_COLOR_STEM`、`feeds_main_pipeline=False`（`:2527-2533`），存 `stage4_narrowband_normalization.json`（`:2564-2567`）后恢复物理分支（`:2568-2580`）。
- 归一化本身带独立质量门（`normalize_dual_narrowband_candidate`，`narrowband_normalization.py:130-346`）：亮度保持 guard、`ha_oiii_ratio_drift_max`、`star_chroma_drift_max` 等多门限；mapping 置信不足（`<0.85`）则 `skipped_unconfirmed_mapping`（`:1754-1764`）；基线事务 `stage4_pre_nbn`（`:1765-1775`）确保不可变基线可回滚。
- 该隔离是本项目对「双窄带真彩 vs 假彩」争议的**最佳工程实践**：主路由保持物理可辩护的 SPCC 结果，艺术 HOO 仅作可选衍生，不影响下游科学与拼接。**D1 给满分项**。

### 1.7 通道语义判定与目标冻结

- `classify_channel_semantics`（`channel_semantics.py:149-210`）+ `_stage4_channel_policy`（`:984-1013`）：基于 `channels` 数、`LINEAR` 元数据、narrowband 关键字、filter hint。路由决策（`:2187-2204`）：
  - `mono` → `SKIPPED_MONO`（confidence 1.0，禁校准）。
  - `nonlinear_color` → `PRESERVE_INPUT`（confidence 0.80，禁 SPCC/PCC）。
  - `unknown` → `PRESERVE_INPUT` + `requires_review` + `status=degraded`（confidence 0.20）。
  - `broadband_rgb_osc` → SPCC→PCC（`:2205-2343`）。
  - `narrowband_composite` → SPCC(narrowband)（`:2206-2371`）。
  保守且自洽；对不确定输入一律「保色 + 标记 review」而非冒险校准，**符合数据真实性原则**。
- secondary labels 约束（`target_runtime.py:313-401`）：`_sync_runtime_policy_from_profile` 合并 secondary_labels，并在 observed primary 与 frozen primary 冲突时将 `policy_candidate` 置 `None`（`:325-370`），禁止切换已冻结的主路由。**符合「primary target 路由冻结、secondary labels 只增不改路由」的设计约束**。

### 1.8 局部恒星软遮罩恢复（最终兜底）

- `_stage4_local_color_fallback`（`:1851-1879`）+ `_stage4_star_white_balance`：仅对恒星做软遮罩局部白平衡，**明确禁止全局白平衡**（`global_white_balance.prohibited=True`，`:1868`）。当 SPCC/PCC 全失败且非窄带时触发（`:2472-2503`）。符合 AGENTS.md「禁止全图白平衡」，是合理的最终兜底——既不完全放弃色彩修正，又不污染星云/星系主体。

### 1.9 完整执行状态机与回退链（逻辑推演）

Stage 4 的决策可建模为一个确定性状态机，输入为通道语义与目标类型，输出为 `color_method` 与 `status`。以下为关键转移路径（行号见各函数）：

| 输入条件 | 转移 | 输出 color_method | status |
|---------|------|-------------------|--------|
| `kind=mono` | G5 | `SKIPPED_MONO` | ok |
| `kind=nonlinear_color` | G5 | `PRESERVE_INPUT` | ok |
| `kind=unknown` | G5 | `PRESERVE_INPUT` | degraded + review |
| 宽带 + SPCC 成功 + 质量门通过 | `:2227-2304` | `SPCC` / `SPCC_LOCAL_GAIA` | ok |
| 宽带 + SPCC 失败/拒 + PCC 成功 + 门通过 | `:2384-2424` | `PCC` / `PCC_LOCAL_GAIA` | ok（warning=spcc_exception_pcc_fallback） |
| 宽带 + SPCC/PCC 均失败 + 局部星 WB 成功 | `:2472-2503` | `LOCAL_STAR_COLOR_RESTORE` | degraded + review |
| 宽带 + 全部失败 | `:2472-2503` | `PRESERVE_INPUT` | degraded + review |
| 双窄带 + SPCC 成功 + 门过 | `:2249-2304` + `:2508-2534` | `SPCC_NARROWBAND` + 隔离 HOO | ok |
| 双窄带 + SPCC 失败/拒 | `:2362-2371` | `PRESERVE_INPUT` | degraded + review（PCC 禁止） |
| plate solve 失败（任何通道） | G3/G4 | 落入局部兜底或保色 | degraded |

- **关键不变量**：① 任何失败路径都先回滚到不可变 `PCC_CHECKPOINT_STEM`（G20，`:2347-2359`/`:2458-2470`）再尝试下一步，保证线性源不被覆盖；② `narrowband` 路径**绝不**进入 PCC（`:2370` 显式「broadband PCC is prohibited」），因为 PCC 的 Gaia Teff 模型对窄带数据无意义；③ 双窄带物理结果存盘失败直接 `hard_degraded`（`:2510-2514`），不允许「无物可继」。
- 该状态机**无悬空状态、无静默成功**：每个非 ok 终点都置 `requires_review=True` 或 `hard_degraded=True`，并被 `pipeline-result.json` 汇总。逻辑完备性在 D1 维度属优秀。

### 1.10 可终止边界与子进程隔离设计详述

- 所有 photometric 校色（SPCC/PCC）均在独立 `siril-cli` 子进程中执行，而非宿主 pipeline 进程内调用：
  - SPCC：`:740-749` 解析 cli 与 process_dir；`:751-769` 写临时 `.stage4_spcc_once.ssf`（含 `requires 1.4.0` / `cd` / `load PCC_CHECKPOINT_STEM` / `spcc ...` / `save SPCC_CANDIDATE_STEM` / `close`）。
  - PCC：`:575-613` 同理写 `.stage4_pcc_once.ssf`。
- 隔离价值：① Siril 崩溃/挂起不波及主流程；② 超时由 `subprocess.run(timeout=...)` 强控（`:786-795`、`:621-630`），超时即 `TimeoutExpired` 转回退，**绝不死等**；③ 临时脚本在 `finally` 清理（`:803-807`、`:638-642`），无残骸。
- 边界参数：超时 `max(PCC_TIMEOUT_MIN_SEC, min(timeout, PCC_TIMEOUT_MAX_SEC))` = 5–180s（`:673-681`、`:530-534`）。这是「可终止性」的硬保证，对应 AGENTS.md Minimum Checks 中「独立子进程 + 限幅超时」要求。**设计正确且必要**。
- 测试钩子：`_run_stage4_spcc_once` / `_run_stage4_pcc_once`（`:719-738`、`:564-573`）允许注入 mock runner，便于无 Siril 环境单测——可测试性良好，但未在审计范围验证测试覆盖。

### 1.11 网络模式与 catalog 选择的交互

- `_stage4_network_enabled`（`:114-118`）由 `SEESTAR_NETWORK_MODE` 控制；catalog 选择（`_stage4_preferred_spcc_catalog:210-215`、`_stage4_preferred_pcc_catalog:202-207`）严格「本地优先、在线次之、均无则 None（禁用）」。
- 离线（network=0）且本地星表缺失时：`spcc_allowed`/`pcc_allowed` 因 `selected_*_catalog is None` 而 False（`:2177`、`:2184`），直接落局部兜底——**不会尝试在线而静默泄露或卡顿**。符合离线隐私/确定性要求。
- 在线模式：先尝试本地（若已装），否则在线 Gaia。两种 catalog 在 `color_method` 上区分 `SPCC` vs `SPCC_LOCAL_GAIA`（`:2287-2295`），置信度本地略高（0.92 vs 0.88）——符合「本地 catalog 更可控、可复现」的工程判断。
- 交互无竞态：catalog 选择在一次 `run_stage4_color_calibration` 调用内确定，不随网络波动中途切换。

---

## 2. D2 — 门禁来源与归档穷举

### 2.1 门禁清单（file:line / 阈值来源 / 钳制 / 失败处理 / 归档）

| # | 门禁 | 位置 | 阈值来源 | 钳制 | 失败处理 | 归档 |
|---|------|------|---------|------|---------|------|
| G1 | 线性检查点载入 | `:1928-1930` | Stage3 契约 | — | `hard_degraded`，禁 SPCC/PCC | pipeline-result.json status |
| G2 | 线性域判定 | `:965-981` | FITS `LINEAR`/契约 | confidence 0.0–0.98 | unknown→保色 review | color_calibration_report.json |
| G3 | `spcc_allowed` | `:2171-2178` | 多条件与 | — | 记 `spcc_skip_reason` | messages 段 |
| G4 | `pcc_allowed` | `:2179-2185` | 仅 broadband | — | 记 `pcc_skip_reason` | messages 段 |
| G5 | 通道种类路由 | `:2187-2204` | `classify_channel_semantics` | confidence 0.90–1.0 | mono/nonlinear/unknown 分支 | channel_semantics 段 |
| G6 | 窄带波长置信 | `:917-918` | `classify_dual_narrowband_mapping` | `<max(0.70,min(0.85,0.99))` 抛错 | SPCC preflight 拒→保色 review | narrowband report |
| G7 | 质量门·通道增益比 | `:1282-1285`、`:1320-1321` | `stage4_pcc_channel_gain_ratio_max` 默认 1.80 | clamp 1.10–3.0 | `channel_gain_ratio_exceeded` | quality_gate.thresholds |
| G8 | 质量门·背景色散 | `:1281`、`:1325-1327` | 0.45(发射)/0.16(星系)/0.28(通用) | — | `background_chroma_exceeded` | — |
| G9 | 质量门·发射豁免 | `:1359-1381` | `emission_balance_gain_ratio_max` 默认 4.0 | clamp gain_ratio_max–5.0 | 移除 gain 拒绝理由 | target_aware_exemptions |
| G10 | 质量门·高光裁剪增长 | `:1298-1301`、`:1336-1340` | `stage4_pcc_clip_growth_max` 默认 0.005 | clamp 0–0.05 | `highlight_clip_growth_exceeded` | — |
| G11 | 质量门·动态范围漂移 | `:1341-1345` | 0.50–2.50 比 | — | `dynamic_range_shift_exceeded` | — |
| G12 | 星色温分布 | `:1150-1168` | `stage4_pcc_star_temperature_ratio_min/max` 0.45/2.20 | clamp 0.20–0.95 / 1.05–5.0 | `star_temperature_distribution_shift_exceeded` | post_calibration_checks |
| G13 | 背景色差 | `:1177-1192` | `stage4_pcc_background_color_delta_max` 0.22 | clamp 0.05–0.60 | `background_color_difference_exceeded`（需恶化） | — |
| G14 | 主体色漂移 | `:1198-1216` | 0.45(发射)/0.28(通用) | clamp 0.05–0.75 | `target_color_drift_exceeded` | — |
| G15 | SPCC/PCC 超时 | `:534`、`:681` | `stage4_*_timeout_sec` | clamp 5–180s | timeout→回退 | attempt.timeout_sec |
| G16 | 波长/带宽钳制 | `:920-924`、`:926-931` | 配置/默认 | 波长/带宽各自上下界 | — | spcc_parameters |
| G17 | 局部恒星白平衡 | `:1851-1879` | `stage4_local_star_wb_enabled` 默认 True | 禁全局 WB | 样本不足→保色 | local_fallback_report |
| G18 | 窄带归一化 mapping | `:1743-1764` | `stage4_nbn_mapping_confidence_min` 0.85 | — | skipped_unconfirmed_mapping | stage4_narrowband_normalization.json |
| G19 | 窄带归一化基线事务 | `:1765-1775` | `_save_stage_output("stage4_pre_nbn")` | — | prohibited（基线存失败） | — |
| G20 | 不可变预色检查点 | `:2347-2359`、`:2458-2470` | `PCC_CHECKPOINT_STEM` | — | 回滚失败→`hard_degraded` | rollback_report |

### 2.2 钳制（Clamp）审计

- 数值钳制贯彻一致：超时（5–180s，G15）、增益比（1.10–3.0，G7）、各 color delta（G13 0.05–0.60 / G14 0.05–0.75）、波长/带宽（G16）、星色温比（G12）、背景色散（G8 来自目标类型常量）均有上下界。符合 AGENTS.md「可调参数集中 + 安全钳制」原则。**未发现越界或未钳制的可调阈值**。
- `numeric()`（`:920-924`）对 NaN/Inf 回退默认，避免 Siril 收到非法参数。**正确**。
- 钳制上下界设计合理，逐条研判：
  - `gain_ratio_max`（1.10–3.0）：下限 1.10 防止「几乎无增益也被拒」，上限 3.0 防止极端偏色（如某通道被放大 3 倍以上）被放行。对发射星云另有 `emission_balance_gain_ratio_max`（4.0，clamp 至 5.0）放宽，但需满足背景平衡等附加条件（G9），非简单放开。
  - `clip_growth_max`（0–0.05）：上限 0.05 即「高光像素占比增长不超过 5‰」才可接受，防止校色引入饱和星点；下限 0 防止负值（无意义）。
  - `background_color_delta_max`（0.05–0.60）：下限 0.05 允许极小背景色差（噪声级），上限 0.60 防止明显色偏背景被接受。
  - `target_color_drift_max`（0.05–0.75）：发射星云放宽至 0.45（默认），因发射体本征色偏红/绿属物理真实。
  - 波长/带宽（G16）：R 波长 600–700nm 覆盖 Hα 附近；GB 450–550nm 覆盖 OIII 附近；带宽 1–100nm 覆盖超窄带到四窄带（与 3.4 PixInsight 教程一致）。**边界取值经过权衡，非随意**。

### 2.6 门禁触发后的可观测性

- 每个门禁拒绝/警告均产生命名 reason（如 `channel_gain_ratio_exceeded`、`background_color_difference_exceeded`、`spcc_narrowband_metadata_unconfirmed`），写入 `messages` 并最终进入 `color_calibration_report.json` 的 `rejection_reasons` / `messages` 与 `pipeline-result.json` 的 `color_warning`。
- 审计者可凭 `color_calibration_report.json` 的 `quality_gate.measurements` 与 `post_calibration_checks` 反向定位是哪个子门触发，无需重跑。**可诊断性强**。
- 唯一盲区：`_stage4_linearity` 的启发式漏检（见 7.1）——若 Stage 3 输出非线性但无对应关键字 token，`:970` 不会判 nonlinear，可能放行 SPCC 到错误数据上。该风险靠 `checkpoint_loaded` 契约缓解，但建议 Stage 3 显式写 `LINEAR` 关键字以启用 `:972-974` 强判定。

### 2.8 质量门数值示例（宽带健康场景推演）

假设宽带星系目标，校色前后通道中位数为 `before=[0.10,0.10,0.10]`、`after=[0.105,0.100,0.095]`：
- 增益 `gain=abs(after/before)=[1.05,1.00,0.95]`，`gain_ratio=1.05/0.95≈1.105` < `gain_ratio_max=1.80` → 通过（G7）。
- 背景通道色散 `bg_spread` 若从 0.05 降至 0.03 → `<bg_spread_max(galaxy=0.16)` 且改善 → 通过（G8）。
- 高光裁剪增长 `clip_growth≈0.001` < `0.005` → 通过（G10）。
- 动态范围比 `dynamic_ratio≈0.98` 在 0.50–2.50 → 通过（G11）。
- 星色温比 `star_ratio` 若 0.9 在 [0.45,2.20] → 通过（G12）。
- 背景色差 `background_delta≈0.02` < `0.22` 且未恶化 → 通过（G13）。
- 主体色漂移 `target_drift≈0.03` < `0.28` → 通过（G14）。
→ `reasons` 空 → `accepted=True`（`:1383-1384`）。该推演说明默认阈值对**健康校色**宽松通过、对**异常校色**（如增益比 >1.8 或背景色散 >0.16）才拒绝，符合预期。

### 2.9 门禁交互矩阵（组合后果）

| 触发门组合 | 结果 status | requires_review |
|-----------|------------|-----------------|
| 仅 G2 unknown | degraded | 是 |
| G3 不满足（无 catalog） | ok（落 PCC/兜底） | 视后续 |
| G6 抛错（窄带未确认） | degraded（保色） | 是 |
| G7/G8/G10/G11 任一拒（宽带） | 转 PCC；PCC 再拒→兜底 | 是 |
| G7 拒但 G9 发射豁免成立 | accepted（emission） | 否 |
| G20 回滚失败 | hard_degraded | 是 |
| 全部成功 | ok | 否 |

- 结论：门禁组合下，**没有任何路径会静默输出错误校色**；最坏情况为「保色 + review」或「hard_degraded」，数据真实性始终优先。这正是 AGENTS.md「默认保守」原则的体现。

### 2.10 可复现性缺口的实际影响评估

- D-A（plan 缺 SPCC 光谱标识）的实际影响：若运维仅保留 `processing-plan.json` 而丢失 `color_calibration_report.json`，则无法从 plan 得知当时用的是 `Sony IMX585`/`ZWO Seestar LP`/`Average Spiral Galaxy`。但**成品 `stage4_color.fit` 本身已内嵌 WCS 与（部分）处理历史**，且 `color_calibration_report.json` 通常与产物同目录归档（U3 待验证），故端到端复现仍可行。D-A 属「Plan 自述不完整」，非「不可复现」。
- D-B（catalog 分块未进 plan）影响更小：离线复现依赖相同本地分块集合，而分块由 `SEESTAR_GAIA_PHOTO_CATALOG` 路径 + 已下载文件决定，属环境配置而非运行参数；`color_calibration_report.json` 已记 `valid_chunks` 文件名，可据此重建环境。
- 综合：当前归档**结果层可复现、意图层基本可复现（补 D-A 后完整）**。P2-3 为提质项，非阻断项。

### 2.7 归档文件职责矩阵

| 文件 | 写入点 | 职责 | 复现价值 |
|------|--------|------|---------|
| `color_calibration_report.json` | `:2840` | 全量校色证据（参数+门+测量+回滚+艺术派生） | 高（结果层可复现） |
| `stage4_narrowband_normalization.json` | `:2564` | 双窄带隔离派生证据 | 中 |
| `processing-plan.json` | `processor_runtime.py:1848` | 意图层（输入/语义/目标/计划步骤/config 红字） | 中（缺 SPCC 光谱标识，D-A） |
| `pipeline-result.json` | `processor_runtime.py:2060` | 跨 stage 汇总 | 低（仅摘要） |
| `device_geometry_report.json` | `:1942` | plate solve 几何 | 中（支撑解算复现） |
| 不可变 `PCC_CHECKPOINT_STEM` | 全程 | 预色线性源，失败回滚目标 | 关键（数据真实性） |

### 2.3 归档（Archiving）逐项审计

- **`color_calibration_report.json`**（`:2716-2840` 组装，`:2840` 写出）：覆盖 `spcc_parameters`（sensor/white_reference/osc_filter/wavelengths/bandwidths，`:942-962`）、`pcc_parameters`、`quality_gate` 全阈值与测量、`post_calibration_checks`、`rollback_report`、`spcc_rollback_report`、`artistic_hoo_report`、`architectural` 元数据、`messages`。**覆盖最全，可复现所需信息基本齐备**。
- **`stage4_narrowband_normalization.json`**（`:2564-2567`）：含 `role/physical_parent/feeds_main_pipeline` 与归一化指标（background_improvement / ha_oiii_ratio_drift）。
- **`processing-plan.json`**（`processor_runtime.py:_write_processing_plan:1848-1988`）：含 input / channel_semantics / target(primary+secondary+policy) / planned_steps / candidate_contracts / config 红字（实际生效配置快照）。**意图层完整，但未单列 SPCC 使用的 sensor/filter/white_ref 标识**（见 2.4 缺口 D-A）。
- **`pipeline-result.json`**（`_write_pipeline_result_manifest:2060-2174`）：color_calibration 段仅汇 method/status/requires_review/physical_color/artistic_hoo。**汇总层，不重复细节**，合理。
- **`device_geometry_report.json`**（`:1942-1950` 由 `activate_device_geometry_report` 写出）：记录焦距/像素/裁剪几何，支撑 plate solve 可复现。
- **不可变检查点**：`PCC_CHECKPOINT_STEM`（预色线性源）全程不被覆盖，SPCC/PCC 失败均回滚至此（G20）。这是「数据真实性」的关键保障——任何失败的校色都不会污染原始线性数据。

### 2.4 可复现性缺口（P2）

- **D-A（意图层缺 SPCC 光谱标识）**：`processing-plan.json` 记录 config 红字但不显式记录 `oscsensor/oscfilter/white_ref`。若仅保留 plan 不保留 `color_calibration_report.json`，无法单凭 plan 复现校色光谱设定。**建议：plan 的 color_calibration 段写入实际 SPCC 参数快照**（P2-3）。
- **D-B（catalog 分块未记入计划层）**：离线 SPCC 依赖本地 Gaia DR3 xp_sampled 的 HEALpixel 分块（`_stage4_local_spcc_catalog_status`，`:168-188`）。报告记录了 `valid_chunks` 文件名，但 `processing-plan.json` 未记；在线 catalog 查询命中取决于当时可用镜像。**建议：确保 `color_calibration_report.json`（已含 valid_chunks）随产物归档**（U3）。
- **D-C（瞬态不可复现）**：在线 catalog 查询受网络/镜像状态影响，同一输入不同时间可能命中不同源。这是在线校色的固有属性，离线 localgaia 可消除。**建议：离线优先策略已就位，文档应强调离线复现需本地星表**。
- 上述缺口**不改变「结果层可复现」**（因 `color_calibration_report.json` 已含全部 SPCC 参数与 chunks），仅影响「仅凭计划层复现」。

### 2.5 门禁可追溯性总评

所有 G1–G20 门禁均：(a) 有明确代码位置；(b) 失败有命名 reason / warning；(c) 关键阈值来自 `PipelineConfig` 且可经 env 覆盖（`processor_runtime.py` env 覆盖与 `_clamp_float`）；(d) 失败动作写入 `messages` 并最终进入 `color_calibration_report.json` 与 `pipeline-result.json`。**审计可追溯性达到生产级**。

---

## 3. D3 — 算法行业标准符合度（联网取证）

### 3.1 SPCC 是现行标准，且优于 PCC

- PixInsight 官方 SPCC 文档：`https://pixinsight.com/doc/docs/SPCC/SPCC.html` —— SPCC 基于 Gaia DR3 `xp_sampled`（336–1020nm，2nm 步进，2.19 亿源），使用本地 XPSD 数据库；其默认白参考即 "Average Spiral Galaxy"，并明确「switching from PCC to SPCC is straightforward」「SPCC is more accurate」「renders PCC obsolete」。
- Siril 官方 SPCC 文档：`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html` —— SPCC 自 1.4.0 引入，用 Gaia DR3（在线或本地 48 分块 HEALpixel 目录）；明确「SPCC is a more accurate version of PCC and renders the latter obsolete」「must be carried out on a linear image whose histogram was not yet stretched」。
- **结论**：Stage 4「SPCC 优先、PCC 回退、强制线性、plate solve 前置」与两大主流工具官方立场**完全一致**。✓ 代码用 `spcc`/`pcc` Siril 命令（`:687`、`:540`）且 catalog 选择逻辑（`:210-215` 优先本地、其次在线）与 Siril「本地优先」理念一致。

### 3.2 Plate Solve 必须前置

- Siril 文档：`https://siril.readthedocs.io/es/latest/astrometry/platesolving.html` —— 「Many of Siril's tools, such as SPCC or PCC, need to know the coordinates … with sufficient accuracy」。SPCC 在无 WCS 时菜单置灰。
- ASTAP / astrometry.net 对比：`https://stellarnomads.com/plate-solving`、`https://www.cloudynights.com/forums/topic/963963-is-astap-still-the-best-plate-solver-for-nina/page/2/` —— 本地 solver（ASTAP / 本地 astrometry.net / Siril 本地 Gaia）均可在离线工作；Siril 1.3+ 本地 catalog 支持在目标坐标锥内近邻搜索（`:347-357` 的 header 候选 + localgaia 即对应此能力）。
- **结论**：Stage 4 先 plate solve 且离线优先 localgaia、在线回退 gaia，**符合行业离线实践**。✓ `-noflip` 保留方向（1.1）也与「solver 默认可能翻转」的警告契合。

### 3.3 OSC 相机光谱参数选择

- PixInsight 官方教程（astroguide）：`https://astroguide.starlust.de/html/SPCCSpectrophotometricColorCalib.html` —— 「When working with OSC cameras, we should always set the QE curve parameter to Ideal QE curve」。PixInsight 论坛（官方人员确认）：`https://www.pixinsight.com/forum/index.php?threads/is-this-the-correct-workflow-for-osc-dual-narrowband-spcc.19717/`、`https://www.pixinsight.com/forum/index.php?threads/choice-of-sensor-and-filter-for-a-osc-camera-in-spcc.21137/` —— OSC 必须选「合并了 Bayer 矩阵的 sensor 响应（Ideal QE / OSC preset）」，选 mono QE 曲线会重复计入（平方）sensor×filter 透射。
- Siril 建模差异：Siril 用**命名 OSC 预设**承载该合并曲线（即用 `oscsensor`），而非单独的「QE 曲线」开关。因此 Stage 4 传 `oscsensor=Sony IMX585` 是与 **Siril 模型一致**的正确做法。
- **结论**：方法正确；风险仅在「预设名/曲线数据是否真实匹配 Seestar S50」——见 P1（1.3 / U1）。⚠️ 另：默认值中 `DEFAULT_SPCC_WHITE_REF="Average Spiral Galaxy"` 与官方默认一致；但 PixInsight 社区对发射星云另有「整片星云作白参考」手法（`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html` 的 SHO 手动校准示例），代码未提供按目标切换，属取舍而非错误。

### 3.4 双窄带 SPCC 的争议与共识

- cloudynights 专帖：`https://www.cloudynights.com/topic/932882-spcc-question-narrowband-filters-mode/` —— ① narrowband 模式用于从窄带数据生成「true color like」图（如 HOO）；② SHO 为假彩色，SPCC 不适用（「there's no reason to use SPCC if you want to create an SHO image」）；③ 勾选 narrowband、填 Ha 656.3/带宽、OIII 500.7/带宽即可，若线性拟合差可取消 narrowband 或选相近曲线；④ 可勾选 "Optimize for stars"。
- cloudynights 隔离讨论：`https://www.cloudynights.com/forums/topic/983844-is-it-possible-to-completely-isolate-haoiii-from-dual-band-data/` —— 双窄带用软件无法完全分离（Bayer 渗漏 + 未知谱线相对强度），「近似可取但不应称精确」；「never be as clean as pure mono channels」。
- Siril 官方警告：`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html` —— 「Don't expect to retrieve the Hubble palette for SHO … huge green cast」；HOO 两通道同数据时波长/带宽应设相等（与代码 G/B 同 500.70/30nm 一致，`:928-931`）。
- PixInsight 窄带 SPCC 教程：`https://telescope.live/blog/pixinsight-spcc-narrowband-images` —— 窄带模式填各通道标称波长/带宽，超窄带可达 3nm，四窄带可达 35nm，与代码带宽钳制（1–100nm）一致。
- **结论**：Stage 4 物理主路由仅做 HOO 真彩 SPCC、SHO 不进主路由、HOO 艺术派生隔离（`feeds_main_pipeline=False`），**与社区共识高度一致**；对「无法精确分离」的固有近似通过隔离+保守质量门处理得当。✓（但质量门为宽带复用，见 3.5）

### 3.5 质量门与窄带场景的适配缺口（P1/P2）

- `_stage4_pcc_quality_gate`（`:1267-1418`）同时服务于宽带 SPCC/PCC 与**窄带 SPCC**（调用点 `:2258`、`:2399`）。其阈值（`bg_spread_max`、增益比、星色温比）按宽带物理色彩设计；发射星云豁免（`:1359-1381`）是「目标类型感知」而非「窄带感知」。
- 行业上窄带 HOO 的背景色度、通道增益行为与健康宽带不同（如 HOO 天然 R 通道强、G/B 弱，增益比大属正常）；复用宽带门可能：① 对合理窄带结果误拒，或 ② 放过宽带门不敏感但窄带特有的异常（如 Ha/OIII 线比异常）。**建议补充窄带专用校验（如 Ha/OIII 线比漂移、星彩度漂移），至少文档化该近似**。属 P1（因影响双窄带主路由判定）但已有隔离兜底，风险可控。

### 3.6 政策命名与代码行为错位（P2，已预告知）

- 所有 `pipeline/configs/policies/*.yaml` 的 `stage4_color.calibration_policy` 均声明 `single_gaia_pcc`（如 `generic_low_snr_safe.yaml:5`），但代码实际主路由为 **SPCC 优先**（`:2205-2343`）。`policy_selector.py:DEFAULT_POLICY` 同样写 `single_gaia_pcc`。
- 影响：策略名误导维护者；若未来有逻辑依赖 `calibration_policy` 字段做分支，将与真实行为不符。**建议统一命名（如 `spcc_first_then_pcc`）或在代码注释显式说明该字段为历史遗留、实际以代码路由为准**。非功能性缺陷。

### 3.7 SPCC 精度告警的处理（行业对标）

- Siril 在拟合色比偏差 >0.1 时输出精度告警；Stage 4 通过 `_stage4_spcc_output_is_imprecise`（`:659-661`、`:816-825`）捕获，但**不单凭此拒绝**，而是交由目标感知像素质量门裁决（`precision_warning_policy="defer_to_target_aware_pixel_quality_gate"`，`:822-825`、`:2265-2278`）。这与 PixInsight「SPCC 给出证书/拟合图供判断」的理念一致——告警是指示而非否决。**处理得当** ✓。

### 3.8 SPCC 与手动校准的边界

- Siril 文档与社区（`https://www.backyardastronomy.net/2026/06/21/siril-color-calibration-getting-accurate-colors-in-your-astrophotos/`）均指出：当 SPCC/PCC 不可用（离线、Gaia 宕机、无法解算）时，手动校准（使天空背景中性灰）仍远优于跳过。Stage 4 的局部恒星软遮罩兜底（1.8）正是这一理念的自动化实现——但更严格地限制了「仅恒星局部」，避免手动全图白平衡对主体的污染。**优于社区手动法的保守 variant** ✓。
- 注意：社区手动法对 SHO 假彩更推荐（3.4），而 Stage 4 对 SHO 根本不进主路由、仅 HOO 真彩走 SPCC，二者互不冲突。

### 3.9 限制星等（limit magnitude）行业实践

- Siril SPCC 的 limit magnitude 通常按 FOV 自动推算；PixInsight SPCC 默认勾选「Automatic limit magnitude」。Stage 4 显式设 `10.5`（`:894-901`）作为确定性上限，属合理工程化（避免不同场自动值波动导致不可复现）。行业教程（`https://telescope.live/blog/pixinsight-spcc-narrowband-images`）亦建议对稀疏场降低 SNR 下限而非硬改 limitmag；Stage 4 未暴露 SNR 下限参数，但对失败已通过 PCC/局部兜底覆盖，影响有限。

### 3.10 白参考（white reference）选择的科学边界

- Siril 官方：`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html` —— 默认 `Average Spiral Galaxy`（基于 SWIRE 模板），并提供全星系/恒星分类谱；G2V 太阳参考用于「人眼可见色」。Stage 4 默认星系白参考对星系目标科学正确；对发射星云，社区有「整片星云作白参考」手法（该文档 SHO 手动校准示例）。Stage 4 未做按目标切换（1.3），但发射星云经质量门豁免（`:1359-1381`）后通常可接受。**建议见 P2-7**。

### 3.11 本地 SPCC catalog 格式与行业一致

- Siril 1.4 本地 SPCC 目录为 Gaia DR3 数据，按 HEALpixel level-1 分 48 文件（`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html`）。Stage 4 的 `_stage4_local_spcc_catalog_status`（`:168-188`）按 `LOCAL_SPCC_FILE_PATTERN` glob 校验分块存在性与字节数，与官方分块模型一致。远程 catalog 自 1.4.1 起为本地目录的托管副本（HTTP RANGE 高效查询），Stage 4 在线回退路径（`:713-717`）与此兼容。**格式层对齐，无架构冲突**。
- 注意：离线 SPCC 需要 zenodo 下载的分块（`DOI 10.5281/zenodo.14697692`）或 Siril 自带安装脚本；Stage 4 仅校验存在性，不校验分块版本/完整性哈希。若分块部分损坏，`valid_chunk_count>0` 但通过校验失败，会落入 PCC/局部兜底——属可接受的鲁棒性边界。

### 3.12 与 PixInsight「Optimize for stars」的对照

- PixInsight 窄带 SPCC 提供「Optimize for stars」选项（cloudynights 3.4 专帖），因窄带对恒星表现不佳。Siril SPCC 当前命令参数（`_stage4_spcc_args`，`:863-962`）**未暴露该选项**。若 Seestar 双窄带 HOO 的星点色校准质量不足，Stage 4 缺少对应的「优化恒星」开关。这不影响物理正确性（质量门仍校验星色温），但可能损失星点观感。**建议（P2 可选）**：评估 Siril SPCC 是否支持等价参数并暴露。

### 3.13 与软件分离窄带（DBXtract 类）的关系

- 社区用 DBXtract 等脚本从双窄带 OSC 数据代数分离 Ha/OIII 灰阶图（`https://www.cloudynights.com/forums/topic/983844-is-it-possible-to-completely-isolate-haoiii-from-dual-band-data/`）。Stage 4 **未采用**该路径，而是直接对 OSC 双窄带图做 SPCC narrowband 模式 + HOO 艺术派生。二者路线不同：DBXtract 产出更「纯」的窄带通道供 SHO/HOO 合成，SPCC narrowband 直接产出真彩近似。Stage 4 的选择（真彩近似 + 隔离）与「物理主路由 + 艺术派生分离」的设计目标一致，**非缺陷**；但若用户需要纯 SHO 科学合成，需在 Stage 4 之外另行 DBXtract，属范围外。

---

## 8. 审计过程与局限

- **审计方法**：纯静态只读审阅 + 三维度对照 + D3 联网取证（WebSearch，URL 见第 6 节）。未运行 pipeline、未修改代码、未下载/校验本地星表。
- **已确认**：路由逻辑、门禁链路、钳制、归档文件职责、与 PixInsight/Siril 官方立场的一致性。
- **未验证（依赖运行环境）**：U1（Siril 预设存在性/曲线正确性）、U2（Seestar 滤镜真实 FWHM）、U3（报告归档落地）、U4（离线分块覆盖）。这些需在实际 Siril 1.4 环境 + 真实 Seestar 数据上闭环验证，属 P1 落地的必要前置。
- **评分置信度**：D1/D2 高（基于代码直接证据）；D3 高（基于官方文档与社区共识 URL）；总分 8.2/10 置信度中高，主要下修因素为 U1/U2 未验证带来的潜在系统性偏色风险。
- **范围边界**：本报告仅覆盖 Stage 4（plate solve + SPCC/PCC + 双窄带隔离）。上游 Stage 3 线性保证、下游 Stage 5/6/7 消费契约仅在本附录（第 7 节）做接口层提示，不构成对上下游的审计结论。

---

## 4. 关键发现 TOP3

1. **【P1】OSC 传感器光谱标识准确性未验证**（`stage4_color_calibration.py:882-905`）：硬编码 `Sony IMX585` + `ZWO Seestar LP` 直接驱动 Siril SPCC 光谱模型。若 Siril 1.4 无此 OSC 预设或曲线不匹配 Seestar S50 实际硬件，将产生系统性偏色且无质量门可捕获（增益比/背景门可能仍「通过」）。**未验证：需在目标 Siril 版本上确认预设存在与曲线正确性（U1）**。
2. **【P1】双窄带质量门为宽带复用**（`_stage4_pcc_quality_gate`，`:1267-1418`，窄带调用 `:2258`/`:2399`）：缺少窄带专用校验，发射星云豁免是目标类型感知而非窄带感知，可能在双窄带主路由上误判。因 HOO 已隔离、物理 SPCC 仅作真彩近似，风险可控但应在 3.5 层面补齐或文档化。
3. **【P2】意图层可复现性缺口 + 政策命名错位**：`processing-plan.json` 未记录 SPCC sensor/filter/white_ref 标识（结果层 `color_calibration_report.json` 已记录，仅计划层缺）；且 policies yaml 的 `single_gaia_pcc` 与代码 SPCC-first 实际路由不符（`policy_selector.py` / `configs/policies/*.yaml`）。二者均不影响成品正确性，但损害可维护性与「仅凭计划复现」。

---

## 5. 改进建议（P0/P1/P2）

### P0（无 — 无数据破坏性 / 阻断性缺陷）
当前实现未发现会损坏原始数据或阻断主路由正确性的缺陷。不可变检查点（G20）与全程回滚机制有效保护数据真实性。

### P1（高优先，建议在下一迭代处理）
1. **验证并固化 OSC 光谱标识**：在目标运行环境用 `siril-cli` 列出 SPCC 可用 `oscsensor` 预设，确认 `Sony IMX585` 存在且曲线匹配 Seestar S50；若不匹配，改为正确预设或自定义曲线文件（`_stage4_spcc_database_status` 已检查 `Sony_IMX585.json` 等四文件存在，`:237-268`，但**未验证其曲线数值正确性**）。并在 `color_calibration_report.json` 记录实际命中的预设名/曲线哈希。
2. **补充窄带感知质量门**：在 `_stage4_pcc_quality_gate` 增加 `narrowband` 分支（或新函数），校验 Ha/OIII 线比漂移、星彩度漂移等窄带特有指标（可复用 `narrowband_normalization.py` 的 `ha_oiii_ratio_drift_max` 等概念），避免宽带阈值对窄带误判。

### P2（中低优先，质量与可维护性）
3. **计划层记录 SPCC 光谱快照**：`_write_processing_plan`（`processor_runtime.py:1848-1988`）的 color_calibration 段写入 `oscsensor/oscfilter/white_ref/limitmag` 与 catalog 标识，使「仅凭计划复现」可行（D-A）。
4. **统一策略命名**：将 policies yaml 与 `policy_selector.py` 的 `single_gaia_pcc` 更名为与实际 SPCC-first 路由一致的名称，或在字段旁注释「历史遗留，实际路由见代码」（3.6）。
5. **文档化双窄带近似**：在 `seestar_Superimpose_workflow.md` §5.4 明确「双窄带 SPCC 为 HOO 真彩近似、HOO 艺术派生隔离、SHO 不进主路由」的物理边界与局限，引用 community 共识（3.4）。
6. **瞬态重试（可选）**：对 SPCC 在线 catalog 瞬时失败，可考虑 1 次有限重试再落 PCC，提升弱网下拿到更优校色的概率（需评估与「可终止边界」原则的冲突，1.4）。
7. **按目标切换 white_ref（可选）**：对强发射星云目标，考虑以「光子通量 / 整片星云」替代默认星系白参考，减少偏冷；需与目标感知质量门协同验证（1.3 / 3.3）。

---

## 6. 证据索引

### 6.1 代码证据（文件:行号）
- 入口与主路由：`stage4_color_calibration.py:1882`（run）、`:2167-2343`（SPCC 优先 / PCC 回退 / 窄带分支）、`:2344-2506`（回滚与局部兜底）
- Plate solve：`_stage4_run_platesolve:434-476`、`_stage4_platesolve_variants:403-417`、`_stage4_platesolve_geometry_args:302-319`、`_stage4_catalog_skip_reason:420-431`、`_stage4_platesolve_diagnostics:479-520`、`_stage4_platesolve_catalogs:347-357`、`_stage4_header_center_coordinates:374-386`
- 光谱参数：`_stage4_spcc_args:863-962`、`_stage4_effective_spcc_filter:863-872`、默认 `:33-38`
- 线性域：`_stage4_linearity:965-981`、检查点 `:1928-1930`
- 单次执行/超时：`_stage4_run_spcc:664-832`、`_stage4_run_pcc:523-656`、钳制 `:534/:681`、精度告警 `:659-661/:816-825`
- 质量门：`_stage4_pcc_quality_gate:1267-1418`、`_stage4_post_calibration_color_checks:1139-1264`、发射豁免 `:1359-1381`
- 双窄带与 HOO 隔离：`_stage4_run_narrowband_normalization:1728-1848`、隔离存档 `:2508-2580`、窄带波长 `:907-949`、置信门 `:917-918`
- 局部兜底：`_stage4_local_color_fallback:1851-1879`
- 通道语义：`channel_semantics.py:149-210`、`_stage4_channel_policy:984-1013`
- 目标冻结：`target_runtime.py:313-401`
- catalog/网络：`_stage4_network_enabled:114-118`、`_stage4_preferred_spcc_catalog:210-215`、`_stage4_local_spcc_catalog_status:168-188`、`_stage4_spcc_database_status:237-268`、`_stage4_pcc_catalog_status:191-199`
- 归档：`color_calibration_report` 组装 `:2716-2840`、`stage4_narrowband_normalization.json:2564-2567`；`processor_runtime.py:_write_processing_plan:1848-1988`、`_write_pipeline_result_manifest:2060-2174`
- 政策命名：`policy_selector.py:DEFAULT_POLICY`、`pipeline/configs/policies/generic_low_snr_safe.yaml:5`

### 6.2 行业证据（URL）
- PixInsight SPCC 官方：`https://pixinsight.com/doc/docs/SPCC/SPCC.html`
- Siril SPCC 官方：`https://siril.readthedocs.io/en/stable/processing/color-calibration/spcc.html`
- Siril Plate Solving 官方：`https://siril.readthedocs.io/es/latest/astrometry/platesolving.html`
- OSC 必须选合并响应（PixInsight 论坛，官方人员）：`https://www.pixinsight.com/forum/index.php?threads/is-this-the-correct-workflow-for-osc-dual-narrowband-spcc.19717/`、`https://www.pixinsight.com/forum/index.php?threads/choice-of-sensor-and-filter-for-a-osc-camera-in-spcc.21137/`
- OSC QE 曲线说明（astroguide）：`https://astroguide.starlust.de/html/SPCCSpectrophotometricColorCalib.html`
- 双窄带 SPCC 社区共识（HOO 真彩 / SHO 不适用）：`https://www.cloudynights.com/topic/932882-spcc-question-narrowband-filters-mode/`
- 双窄带无法精确分离：`https://www.cloudynights.com/forums/topic/983844-is-it-possible-to-completely-isolate-haoiii-from-dual-band-data/`
- PixInsight 窄带 SPCC 教程：`https://telescope.live/blog/pixinsight-spcc-narrowband-images`
- ASTAP / astrometry.net 对比：`https://stellarnomads.com/plate-solving`、`https://www.cloudynights.com/forums/topic/963963-is-astap-still-the-best-plate-solver-for-nina/page/2/`

### 6.3 未验证项汇总
- U1：Siril 1.4 是否内置 "Sony IMX585" OSC 预设及其曲线匹配 Seestar S50（影响 P1-1）。
- U2：Seestar S50 双窄带滤镜真实 FWHM 是否等于代码默认 20nm(Ha)/30nm(OIII)（影响 1.5 精度）。
- U3：`color_calibration_report.json` 在端到端运行中是否始终随产物归档（影响 2.4 复现性）。
- U4：弱网/离线混合下 `localgaia` + 本地 SPCC 分块是否覆盖 Seestar 典型视场（影响 1.1 盲解成功率）。

---

## 7. 附录：Stage 4 与上下游契约

### 7.1 上游契约（Stage 3 → Stage 4）
- 输入：`stage3_bgremoved`（线性、背景已去除）。Stage 4 在 `:1928-1930` 载入，失败即 `hard_degraded` 禁校准。这要求 Stage 3 必须产出**真实线性**产物——若 Stage 3 意外输出拉伸/非线性数据，`:965-981` 的启发式线性判定可能漏检（仅匹配关键字 token），是 D1 的**残余风险点**。
- 缓解：Stage 3 checkpoint 契约（`checkpoint_loaded`）作为主要保证；建议在 Stage 3 写出时显式写入 `LINEAR` 关键字，使 `:972-974` 的强判定生效，降低漏检概率。

### 7.2 下游契约（Stage 4 → Stage 5/6/7）
- `stage4_color`（物理 SPCC 结果，线性）应直接进入 Stage 5/6（去星 / 拉伸）。由于物理主路由保持线性，下游拉伸（Stage 6，按命名错位实际在 `stage6_stretching.py`）可在线性域安全进行。
- `stage4_hoo_artistic`（隔离派生，`feeds_main_pipeline=False`）不应进入主去星/合成链路，仅作可选艺术输出。下游若误消费该文件，会因「假彩 + 已派生」导致科学链路污染——建议在下游输入契约中显式排除 `stage4_hoo_artistic`（D-A 同类意图层缺口）。
- WCS：plate solve 写入的 WCS 随 `stage4_psolved`/`stage4_color` 保留，供下游标注/合成对齐使用；无 WCS 时下游仍可运行（starless/stretch 不依赖 WCS）。

### 7.3 命名错位提醒（与用户预告知一致）
- 按用户说明：Stage 6 去星实现位于 `stages/stage7_star_separation.py`，Stage 7 拉伸位于 `stages/stage6_stretching.py`。本报告聚焦 Stage 4，未据此错位调整——仅在本附录记录，避免下游契约分析混淆。

### 7.4 配置可观测性总结
- 所有可调阈值集中在 `PipelineConfig`（`models.py:169-204`），经 `processor_runtime.py` env 覆盖与 `_clamp_float` 钳制后生效；`processing-plan.json` 记录生效配置红字（`processor_runtime.py:1848-1988`）。**运维可观测性良好**，唯一缺口为 SPCC 光谱标识未进 plan（D-A / P2-3）。

---

*报告结束。总评分 8.2/10（B+）。无 P0 阻断性缺陷；P1 两项均围绕「光谱数据准确性验证」与「窄带质量门适配」，可在不改动主路由的前提下于下一迭代收敛。*
