# Stage 6（线性去星与星点层准备）深度审计报告

- **审计对象**：`/Users/mz/dev/aiseestart` — Seestar 望远镜离线深空后期 pipeline（Python + Siril 1.4.0）
- **审计阶段**：Stage 6 `star_separation`（线性域去星 + 星点层准备）
- **主代码**：`pipeline/stages/stage7_star_separation.py:436`（`run_stage6_star_separation`，文件名带 7 实为 Stage 6）、`pipeline/syqon_starless.py`、`pipeline/starmask_cleanup.py`、`pipeline/stage_support.py`、`pipeline/stage7_quality.py`、`pipeline/stage7_repair.py`、`pipeline/pipeline_safety.py`、`pipeline/quality_gate.py`、`pipeline/models.py`
- **去星模型**：`resources/siril_plugins/vendor/siril-scripts/processing/SyQon-Starless.py`（Zenith v1 / Axiom 2.1）
- **文档基准**：`pipeline/seestar_Superimpose_workflow.md` §5.6（L471-L534）、`pipeline/AGENTS.md`
- **审计性质**：纯只读研究，未修改任何项目代码
- **审计日期**：2026-08-05

> 说明：本阶段逻辑实现位于 `pipeline/stages/stage7_star_separation.py` 的 `run_stage6_star_separation`（L436）。同文件 `run_stage7_star_separation`（L1326）仅为历史兼容别名。下述"Stage 6"即指此函数。

---

## 一、摘要与总评分

Stage 6 是 starless-first 架构的**枢纽**：在线性域先把图像拆成 `starless`（主体）与 `starmask`（星点层），后续 Stage 7 只拉伸 starless，Stage 9 再受控回星。其输入为"最终可用线性检查点"，固定使用同一线性源做质量重试（不串联劣化），结果显式为四态之一：`accepted / target_bypass / rejected / tool_failed`。工程完备度（事务快照、结构化 JSON、SHA-256 断点契约）明显高过业余脚本水平。

**总体判断**：核心架构选择（线性域去星 + 可逆预拉伸 + starless-first）与 StarNet++、StarXTerminator 的官方线性处理范式**方向一致**，且比对社区对 SyQon/Zenith"需要轻拉伸"的经验做了可逆兼容。但存在三类系统性问题：(a) 对 SyQon 内部采用 **arcsinh 预拉伸**这一做法，未能完全规避 StarXTerminator 明确警告的"arcsinh 使星点轮廓与小型星系不可区分"风险；(b) **模型权重身份（哪个 .pt、哪一版本）未写入结构化归档**，仅凭 JSON 无法复现"用的是哪个模型"；(c) 质量门阈值几乎全部为代码内默认值，缺少外部 policy 的可观测来源标注，且 Stage6.5 门建议"拉伸输入"却被代码主动忽略。

| 维度 | 评分 | 理由 |
|---|---|---|
| **D1 逻辑专业合理性** | **7.5 / 10** | 线性域去星由 SyQon 脚本在内部做"临时、可逆 IHS(arcsinh) 预拉伸 → 推理 → 逆变换回线性"（`SyQon-Starless.py:875/887/950`，`_IHS_B=6.0`、`_IHS_TARGET=0.40` L126-127），与 StarNet++/StarXTerminator"线性输入需 MTF 预拉伸 + 精确逆变换"的官方范式一致；质量重试复用同一线性输入（L884-902、L235）设计正确，不串联劣化；四态结果证据链完整（`target_bypass` L494 / `rejected` L606、L1133 / `tool_failed` L1268 / `accepted` L1133-1137）；星点层 `starmask_raw→starmask_clean` 多尺度清理 + 硬门回滚完善（L794-813）。扣分点：用 **arcsinh** 而非文档更稳妥的 MTF 做预拉伸，落入 StarXTerminator 点名的"arcsinh 让星点像小星系"陷阱风险（见 D3）；Stage6.5 门已算出 `recommended_starless_input` 却被代码主动忽略（L575-582）。 |
| **D2 门禁来源与归档** | **7.0 / 10** | 门禁数量充足且结构化：target_bypass 门、Stage6.5 预去星门、preflight 门、工具可用性/超时门、starmask 清理硬门、综合质量门（残星/halo/黑洞/污染/噪声增益/覆盖/宽度）、动态范围门、重试上限门、像素修复接受门、Stage8 交接门。四态结果以 `star_separation_state` + `reason_code` + `reasons[]` + `stage8_handoff` 写入 `stage6_starless_quality.json` 并经 `_record_stage` 进 `pipeline-result.json`，可重建"为什么被拒绝"的主链。扣分点：**SyQon 模型版本/权重身份未进结构化 JSON**（"Zenith v1"/"Axiom 2.1"仅落日志 L750-751，`stage6_starless_quality.json` 只含 `use_axiom` 布尔与 `selected_candidate_id`，权重 `.pt` 路径与校验值缺失）；阈值几乎全为 `PipelineConfig` 代码默认值（models.py:339-417），缺"来源=policy/硬编码"标注；Stage6.5 门建议被忽略，导致该门对实际路由无约束力。 |
| **D3 算法行业标准符合度** | **7.0 / 10** | **符合**：starless-first 是社区主流（去星→拉 starless→回星，Siril Workflow PDF / bridgecameraastroimaging）；线性域去星 + 可逆预拉伸与 StarNet++（`siril.readthedocs.io`）、StarXTerminator（`rc-astro.com`）官方线性范式对齐；Siril 1.4 将 StarNet.py 与 SyQon Starless/StarXTerminator 并列为第三方 AI 模型（`siril.readthedocs.io/it/latest`）。**偏离/风险**：SyQon/Zenith 社区明确"used post stretch… needs at least a light statistical stretch"（Astrobin 论坛），本项目的线性+可逆轻拉伸是折中但非工具"首选"用法；**最严重风险**——StarXTerminator 文档明确警告"arcsinh / GHS 拉伸会让星点轮廓与小椭圆星系不可区分，导致去星不全"，而本项目预拉伸恰为 arcsinh。**自创**：starmask 多尺度清理（紧致保护+弥散抑制+halo 衰减+硬门回滚）属项目自研，行业无统一标准，需自身验证。SyQon 模型训练数据/权重校验未公开 → 标注"未验证"。 |
| **总评** | **7.2 / 10** | 架构与可观测性均达工程上乘，但"arcsinh 预拉伸陷阱未对冲 + 模型权重身份未归档 + 门阈值来源不可见"使其尚未达到科学级可复现线性阶段。 |

**最严重问题（一句话）**：SyQon 在线性域去星依赖 **arcsinh 预拉伸**，恰好踩中 StarXTerminator 官方点名的"arcsinh 使星点像小星系、去星不全"陷阱，且模型权重身份（哪个 `.pt`、版本几）未写入结构化归档，仅凭 `stage6_starless_quality.json` 无法复现"本次用的是哪个模型"。

---

## 二、D1 — 逻辑专业合理性

### 2.1 执行链路总览（含实际行号）

```
run_stage6_star_separation                          stage7_star_separation.py:436
├─ 初始化 _stage8_handoff（v1 schema）              :446-460
├─ _prepare_star_separation_source()                 :463-465   ← 选出固定线性输入 stem
├─ _active_target_type()                            :466-470
├─ should_bypass_star_separation()                   :476-481   ← TARGET_BYPASS 门
│   └─ 命中 → save stage6_passthrough, state=TARGET_BYPASS, return  :487-565
├─ 读 pre_starless_gate_report                       :573-582   ← Stage6.5 门（建议被忽略）
├─ gate.ready_for_starless=false → REJECTED         :585-674   ← state=REJECTED
├─ try:
│   ├─ _stage7_preflight_check()                     :701-704   ← preflight 门
│   ├─ _find_plugin_script(SyQon-Starless.py)        :713
│   ├─ _request_stage7_starless_plan()              :715       ← tile/overlap/axiom
│   ├─ _stage7_try_syqon_variant(initial)           :742       ← 首选去星
│   ├─ 失败 → SASP Dark Star 回退                    :768-785
│   ├─ 仍失败 → raise SirilError                      :786      → 落入 except=TOOL_FAILED
│   ├─ _stage7_prepare_starmask() (pm stretched-starless) :794  ← 星点层
│   ├─ _stage7_clean_starmask(initial)              :795       ← starmask 多尺度清理
│   ├─ _stage7_quality_assessment(initial)          :816       ← 综合质量门
│   ├─ _apply_starmask_cleanup_hard_gate()          :822
│   ├─ 若 status!=ok 且 conservative_repair：         :843-951
│   │    └─ 同线性源 SyQon 参数重试（tile/overlap/axiom）:884-902
│   ├─ 像素修复（halo/残星/彩噪局部）                :982-1107
│   ├─ separation_accepted = status==ok && starless && !hard_failed :1127-1137
│   ├─ state = ACCEPTED / REJECTED                  :1133-1137
│   ├─ _stage8_handoff_from_stage6()                :1148
│   ├─ _write_stage_json(stage6_starless_quality.json):1167
│   └─ save stage6_starless / stage6_passthrough     :1209
└─ except (CommandError,SirilError):
    └─ state=TOOL_FAILED, save stage6_passthrough, JSON :1261-1323
```

### 2.2 线性域去星的专业依据与代价（核心问题）

**关键澄清**：本项目并非"裸线性图直接喂给去星模型"。真正的做法在 SyQon 脚本内部：

- `SyQon-Starless.py:126-127`：`_IHS_B = 6.0`、`_IHS_TARGET = 0.40`（目标中值 0.40 的**轻量**拉伸）。
- `SyQon-Starless.py:875`：`img_for_model = apply_ihs_per_channel(img, ihs_params)`——推理前对线性图做 IHS(arcsinh) 临时预拉伸。
- `SyQon-Starless.py:950`：`result = apply_ihs_per_channel_inverse(result_stretched, ihs_params)`——输出前逆变换回线性域。

主流程（`workflow.md` §5.6 L471-534 确认）**不再额外预拉伸**，直接把线性检查点交给 SyQon，由脚本完成"可逆轻拉伸→去星→逆变换"。这与 StarNet++（Siril 文档："Pre-stretch linear image: 优化的 MTF 拉伸应用于运行前，完成后再逆拉伸"）和 StarXTerminator（RC Astro："处理线性图时，StarXTerminator 内部执行 MTF 拉伸，然后精确逆变换回线性"）的官方线性范式**本质相同**——都承认"去星网络需要被拉伸过的观感才能正确识别星点轮廓"，区别只是本项目的临时拉伸是 **arcsinh** 而官方文档多用 **MTF**。

**代价与风险**：
1. **arcsinh 陷阱（见 D3）**：StarXTerminator 文档明确点名 arcsinh / GHS 拉伸会让星点轮廓与小型椭圆星系不可区分，导致"不去星或只去一半"。本项目用 arcsinh（`_IHS_TARGET=0.40`）属轻拉伸，风险被削弱但未消除；对含大量小星系的场（如 M51、室女座星系团）可能系统性残留。
2. **可逆性依赖脚本正确实现**：若 `apply_ihs_per_channel_inverse` 与正向参数不完全对偶（例如通道间 IHS 参数估计误差），输出 starless 会带残余非线性，污染后续 Stage 7 拉伸。代码 `compute_ihs_per_channel_params`（L607-664）逐通道估计，理论上可对偶，但无断言校验逆变换残差。
3. **收益**：保持 starless 与原始线性检查点同域，Stage 7 可独立、确定地拉伸主体，Stage 9 回星时不引入拉伸耦合——这正是 starless-first 的核心收益，设计方向正确。

### 2.3 四态结果判定证据链

| 状态 | 触发条件 | 关键行号 | 落盘产物 |
|---|---|---|---|
| `target_bypass` | `should_bypass_star_separation()` 命中（globular/open cluster、reflection_nebula_cluster） | `pipeline_safety.py`；`:476-481,494` | `stage6_passthrough.fit` + JSON `mode=star_preserve_target_bypass` |
| `rejected`（Stage6.5 门） | `gate_report.ready_for_starless=false` 且 `stage7_skip_unready_starless=True` | `:585-606` | `stage6_passthrough.fit` + `reason_code=pre_starless_gate_rejected` |
| `rejected`（质量门） | `quality.status!=ok` 或 starmask 清理 `hard_failed` | `:1127-1146` | `stage6_passthrough.fit`（starless 置 None） |
| `tool_failed` | SyQon 缺失且 SASP 不可用 / 子进程异常 | `:1261-1268` | `stage6_passthrough.fit`（含星，复核路径） |
| `accepted` | `separation_accepted = status==ok && starless_file && !cleanup_hard_failed` | `:1127-1137` | `stage6_starless.fit` + `starmask_clean.fit` |

证据链**完整且互斥**，四个分支均显式赋值 `pipeline._star_separation_state` 并写入 `stage6_starless_quality.json["star_separation_state"]`，无"状态悬空"路径。`rejected`/`tool_failed` 都回退到 `stage6_passthrough`（含星复核路径），而非崩溃——降级设计稳健。

### 2.4 质量重试复用同一线性输入的正确性

重试循环（`:844-951`）对每个 variant 调用 `_stage7_try_syqon_variant(...)`（`syqon_starless.py:223`），其内部第一行即 `pipeline.cmd_with_check("load", pipeline.stretched_name)`（L235）——即**每次都从固定的最终线性检查点重新加载**，仅改变 `tile_size/overlap/axiom`。代码注释亦明确（L683 `syqon_failure_reason` 之外无串联），符合"质量重试不串联劣化"的架构契约。**正确**。

### 2.5 starmask_raw → starmask_clean 清理逻辑

- `starmask_raw` 由 `_stage7_prepare_starmask()` 经 `pm ${stretched} - ${starless}` 差分构建（`:794`；实现见 `stage_support.py:_build_manual_starmask` L1262）。
- `clean_starmask_pixels()`（`starmask_cleanup.py`）执行多尺度清理：背景 MAD 噪声估计、紧致星保护（`compact_floor=0.88`）、弥散残差抑制（`diffuse_strength=0.75`）、halo 衰减（`halo_strength=0.35`）。
- **硬门**：`diffuse_residual_ratio > max_diffuse_residual_ratio(0.08)`（`models.py:402`）→ `hard_rejected`/`rolled_back`，置 `pipeline._stage7_starmask_cleanup_hard_failed=True`（`:1114-1123`）→ 禁用 Stage 9 回星。逻辑自洽，避免脏星点层回混。

### 2.6 star holes（黑洞）检测与 target_bypass 判定

- **star holes**：由 `stage7_quality.py` 的 `black_hole_score` 度量，门限 `stage7_black_hole_score_max=0.08`（`models.py:387`）。去星模型在亮星处留下的暗坑/暗环被量化并拒收。
- **target_bypass**：`should_bypass_star_separation()`（`pipeline_safety.py`）对 `STAR_PRESERVE_TARGET_TYPES = {globular_cluster, open_cluster, reflection_nebula_cluster}` 返回 True；由 `stage6_star_preserve_target_bypass_enabled`（`models.py:413`，默认 True）开关。星团/反射星云的"星即主体"，绕过去星符合天体物理意图。

---

## 三、D2 — 门禁来源与归档

### 3.1 门禁穷举表

| # | 门禁 | 位置（file:line） | 阈值 | 阈值来源 | 钳制 | 是否落盘 |
|---|---|---|---|---|---|---|
| G1 | target_bypass 门 | `pipeline_safety.py` `should_bypass_star_separation`；启用 `models.py:413` | 类型集合硬编码 + `enabled` 开关 | 代码硬编码 | 无 | 是（JSON `mode`/`reasons`；pipeline-result `reason_code`） |
| G2 | Stage6.5 预去星门 | `quality_gate.py:15-80`；调用 `seestar_Superimpose.py:2053` | `max_bg_dirty=0.35, max_core_clip=0.01, max_star_halo_risk=0.65, max_global_dark=0.70, min_bg_median=0.005, max_edge_black=0.10` | policy `stage6_5_pre_starless_gate` 或 `quality_gate.py:22-27` 硬默认 | 无 | 是（`pre_starless_gate_report.json` + `stage6_starless_quality.json`） |
| G3 | preflight 门 | `stage_support.py:374-`（_stage7_preflight_check） | `edge_black_warn=0.10/high=0.18, bg_median_high=0.16, bg_std_high=0.055, bg_noise_ratio_high=0.55` | `PipelineConfig`（`models.py:379-383`） | 无 | 是（JSON `preflight`） |
| G4 | 工具可用性门 | `stage7_star_separation.py:713,762-786` | 脚本存在性 + SASP 回退 | 代码逻辑 | 无 | 部分（仅日志 + `tool_failed` 态） |
| G5 | 超时/GPU 门 | `syqon_starless.py` `_syqon_starless_cli_options` | `SEESTAR_SYQON_TIMEOUT_SEC`、`SEESTAR_SYQON_GPU` env；`tile_size∈[512,1024]`、`overlap∈[64,128]` | env + 代码钳制 | 是（512-1024 / 64-128） | 否（仅日志） |
| G6 | starmask 清理硬门 | `starmask_cleanup.py`（diffuse > 0.08） | `diffuse_residual_ratio_max=0.08` | `PipelineConfig`（`models.py:402`） | 无 | 是（JSON `starmask_cleanup[].status`，`hard_failed` 标志） |
| G7 | 综合质量门 | `stage7_quality.py` `stage7_quality_assessment` | `residual_star_max=0.45, halo_max=0.35(亮星云0.60), black_hole_max=0.08, starmask_contam_max=0.25, noise_gain_max=1.25, starmask_coverage_min=0.35, starmask_width_max=1.80` | `PipelineConfig`（`models.py:384-391`） | 无 | 是（JSON `attempts`/`selected`） |
| G8 | 动态范围门 | `stage7_quality.py` `stage7_dynamic_range_assessment` | `min_ratio=0.55, peak_signal_min=0.006, peak_bg_ratio_min=4.0` | `PipelineConfig`（`models.py:392-394`） | 无 | 是（JSON `derived`） |
| G9 | 重试上限门 | `stage7_star_separation.py:849-852,884-902` | `stage7_quality_retry_max=2` | `PipelineConfig`（`models.py:378`） | 无 | 是（JSON `retry_max`） |
| G10 | 像素修复接受门 | `stage7_star_separation.py:1044-1065` | `repair_max_score_growth=0.00, chroma_reduction_min=0.20, chroma_delta_min=0.0005` | `PipelineConfig`（`models.py:414-416`） | 无 | 是（JSON `starless_pixel_repairs`） |
| G11 | Stage8 交接门 | `_stage8_handoff_from_stage6` `stage7_star_separation.py:103-304` | 同 G7 + `bright_nebula_halo_advisory` 路径 | `PipelineConfig` | 无 | 是（JSON `stage8_handoff` + pipeline-result `details`） |

### 3.2 阈值来源分类

- **全部门禁阈值（除 G2 可被 policy 覆盖外）均为 `PipelineConfig` 代码内默认值**（`models.py:339-417`，每项带中文注释）。即"集中配置、可被 `pipeline_policy.json` 覆盖但默认硬编码"。
- **问题**：这些默认值的"来源语义"未标注——是来自某篇论文、PI 社区经验、还是作者拍定？`models.py` 仅给数值与一句用途说明，无可追溯引用。对科学级可审计阶段，建议在字段 docstring 标注 `source=empirical|policy|documented` 与参考出处。
- **钳制情况**：仅 G5（tile/overlap）有硬钳制；其余门限无上下钳，依赖调用方不传非法值。AGENTS.md 要求"画质风险参数必须有上限或回退值"——G7/G8 多数已是上限门，但**输入类参数**（如 `_IHS_TARGET`、`_IHS_B`、修复强度）的钳制在 Python 侧缺失，仅存在于 SyQon 脚本内部常量。

### 3.3 四态结果是否结构化归档

**是，且结构完整**。`stage6_starless_quality.json` 包含：
- `star_separation_state`（四态枚举值）
- `mode`（如 `star_preserve_target_bypass` / `skipped_by_pre_starless_gate` / `parameter_optimization`）
- `reasons[]`（每项含 `code` / `source_stage` / 量化 `value`/`limit`）
- `stage8_handoff`（policy / `reason_code` / `reasons` / `metrics`）
- `attempts[]` + `selected`（每次 SyQon 变体的完整质量记录，含 `derived` 指标）

并经 `_record_stage`（`seestar_Superimpose.py:1318`）写入 `pipeline-result.json` 的 stage detail（`reason_code` + `details.stage8_handoff`）。**仅凭归档，可重建"为什么被拒绝"的主链**（门触发 → 量化指标越限 → 四态）。

### 3.4 SyQon 模型版本/权重记录（D2 关键缺口）

- 模型名字符串 `"Zenith v1"` / `"Axiom 2.1"` **仅写入 `stage_messages`（日志）**（`stage7_star_separation.py:750-751`），**不进 `stage6_starless_quality.json`**。
- 结构化 JSON 仅含 `processing_plan.use_axiom`（布尔）+ `selected_candidate_id` + `tile_size` + `overlap`（`ai_advisory.py:997` 起）。
- 实际权重文件（`Axiom2_1.pt` / `axiom21.pt` / zenith 权重）的**路径与 SHA-256 校验值均未记录**（`syqon_starless.py:199-220` 仅做"是否存在"探测）。
- **结论**：仅凭归档**无法复现"本次用的是哪个模型权重、哪个版本"**。模型升级（Zenith 已到 v1.3，见 D3）后旧 run 的归档无法定位当时权重。这是可复现性硬伤，建议补 `syqon_model={name,weight_path,sha256,axiom}` 进 JSON。

### 3.5 能否仅凭归档复现"为什么被拒绝"

- 主链（门触发 + 量化越限 + 四态 + Stage8 路由）：**可**。
- 例外：模型权重身份缺失（§3.4）；`target_bypass` 分支的 `quality_record.derived` 全为 0（`:517-521`，因未实际去星），属预期而非缺陷。

---

## 四、D3 — 算法行业标准符合度（含 URL）

### 4.1 StarNet++ 官方线性要求

- Siril 官方："Pre-stretch linear image: 优化的 MTF 拉伸应用于运行 StarNet 前，完成后再应用逆拉伸。这对在线性阶段使用 StarNet 是**必需**的。" —— `https://siril.readthedocs.io/en/stable/processing/stars/starnet.html`
- Siril 集成教程："Starnet++ greatly prefers its input to have a midtone transfer function (MTF) autostretch applied… strongly recommended that you use Starnet++ before doing any custom stretching, and check the 'Pre-stretch Linear Image' option." —— `https://siril.org/tutorials/integrated-starnet`

**对标**：本项目"线性输入 + 脚本内临时预拉伸 + 逆变换回线性"与 StarNet++ 官方线性范式**一致**（仅预拉伸类型不同：本项目 arcsinh vs 文档 MTF）。

### 4.2 StarXTerminator 官方线性要求（最相关）

- RC Astro 用法说明："Use StarXTerminator as early in the processing flow as possible, ideally right after integration, with the data still in a linear state… StarXTerminator is trained on images stretched using a simple midtones transfer function (MTF). When processing linear images, StarXTerminator **internally performs such a stretch automatically, then precisely reverses it** after processing to return the image to a linear state."
- **关键警告**："In particular, an **arcsinh stretch** and a generalized hyperbolic stretch (GHS) can create star profiles that are **indistinguishable from small elliptical galaxies**, and will result in StarXTerminator not removing, or only partially removing, the stars." —— `https://www.rc-astro.com/starxterminator-usage-notes`

**对标**：前半段（线性 + 内部预拉伸 + 精确逆变换）与本项目**完全同构**；但本项目预拉伸恰为 **arcsinh**，正中 StarXTerminator 点名的陷阱。StarXTerminator 推荐的是 **MTF**（更接近 asinh 的温和变体而非纯 arcsinh）。**结论：部分符合，且存在被文档明确警告的风险**——建议评估将 IHS 预拉伸由 arcsinh 改为 MTF 类（如 `1/(1+k·x)` 或 asinh 低 B 值）以规避。

### 4.3 SyQon / Zenith 工具来源与"轻拉伸"经验

- Siril 官方公告（2026-01）：SyQon 专为 Siril 社区开发的免费去星脚本 Zenith，原生分辨率推理、本地聚合保结构。—— `https://siril.org/2026/01/a-brand-new-star-removal-script-comes-to-siril-zenith`
- Astrobin 社区实测："The only thing to keep in mind is that it is **used post stretch, not pre**. It doesn't need to be the final stretch but it **needs at least a light statistical stretch** to work properly." —— `https://ssr.app.astrobin.com/forum/topic/216451/processing/new-star-removal-tool`（镜像 `https://a813c8e227e3e9d16.awsglobalaccelerator.com/forum/topic/216451/processing/new-star-removal-tool`）
- Siril 1.4 将 StarNet.py 与 "SyQon Starless, RC Astro StarXterminator" 并列为第三方 AI 模型支持。—— `https://siril.readthedocs.io/it/latest/processing/stars/starnet.html`

**对标**：社区明确 SyQon"需轻拉伸、最好后拉伸"。本项目用线性 + 内部轻 arcsinh 预拉伸（target 中值 0.40）是**对"需轻拉伸"的折中满足**，但非工具"首选"用法。风险可控但应在文档中声明"本 pipeline 以可逆轻拉伸替代后拉伸，符合 SyQon 轻拉伸需求，但偏离其默认后拉伸工作流"。

### 4.4 starless-first 流程地位

经典社区工作流即"去星 → 拉伸 starless → 回星"：Siril Workflow PDF（步骤 8 去星、9 拉伸 starless、10 回星重组）；`https://www.fas37.org/wp/wp-content/uploads/2025/10/Siril-Workflow.pdf`；以及 `https://bridgecameraastroimaging.blogspot.com/2024/03/using-starnet-in-siril.html`。**对标：符合**——starless-first 是社区主流，本项目将其工程化为"线性域拆层 + 受控回星"，方向正确。

### 4.5 星点蒙版生成与 star removal 学术方法

- 行业通用 star mask 生成法即 `starless = model(input)`、`starmask = input - starless`（StarNet++ 文档："star mask is calculated as the difference between the original image and the starless image"）。本项目 `_build_manual_starmask` 用 `pm ${stretched} - ${starless}`（`stage_support.py:1262`）**一致**；并在其上加多尺度清理（紧致保护/弥散抑制/halo 衰减），属合理增强。
- "黑坑/残星/halo 残留"的学术度量（本项目 `black_hole_score`/`residual_star_score`/`halo_residue_score`）属经验性质量评分，行业无统一标准公式——**自创，需自身验证**，但度量维度（残星、halo、黑洞、噪声增益、动态范围）覆盖了已知去星 artifacts 全集，维度选择专业。

### 4.6 SyQon 工具来源结论

- "SyQon-Starless"是真实、Siril 官方公告背书、社区验证的去星工具，**存在性已验证**。
- 但**模型训练数据、权重文件、架构细节、版本校验值均未公开**（siril.org 公告仅述设计哲学）。→ **未验证：SyQon/Zenith 模型权重训练集与校验和未公开，无法做权重级复现性校验**。

### 4.7 行业已知陷阱的代码规避情况

| 行业陷阱 | 来源 | 本项目是否规避 |
|---|---|---|
| 拉伸放大脏背景进入去星模型 | workflow.md §5.6 | **已规避**：去星在线性域，Stage 7 才拉伸 starless |
| 去星后星点层含星云残差回混 | StarNet 经验 | **已规避**：starmask 多尺度清理 + 硬门回滚 |
| 重试串联劣化 | 工程常识 | **已规避**：同线性源重载（L235） |
| arcsinh 使星点像小星系、去星不全 | StarXTerminator 文档 | **未规避**：预拉伸即 arcsinh，仅以低 target(0.40) 削弱 |
| 模型权重不可复现 | 科学可复现要求 | **未规避**：权重身份未进归档（§3.4） |
| 黑坑（star holes） | 去星通病 | **已度量**（black_hole_score）但未在预去星阶段预防，仅事后拒收 |

---

## 五、关键发现 TOP3

1. **【D1/D3 风险】arcsinh 预拉伸踩中 StarXTerminator 官方点名陷阱**。SyQon 脚本用 arcsinh（`_IHS_TARGET=0.40`）做临时预拉伸（`SyQon-Starless.py:875/950`），而 RC Astro 文档明确"arcsinh 让星点轮廓与小椭圆星系不可区分，导致去星不全"（`rc-astro.com/starxterminator-usage-notes`）。对小星系密集场可能系统性残星。缓解：target 中值 0.40 属轻拉伸，但未根除。

2. **【D2 缺口】SyQon 模型权重身份未进结构化归档**。模型名仅落日志（`stage7_star_separation.py:750`），JSON 只含 `use_axiom` 布尔 + `selected_candidate_id`，权重 `.pt` 路径/SHA-256 缺失（`syqon_starless.py:199-220` 仅做存在性探测）。Zenith 已迭代至 v1.3（siril.org 公告），旧 run 无法定位当时权重——**科学级可复现性硬伤**。

3. **【D1 逻辑】Stage6.5 预去星门"算出建议却被忽略"**。门已算出 `recommended_starless_input`（如 `stage7_ultra_conservative_asinh`，`quality_gate.py:57`），但 `run_stage6_star_separation` 主动忽略、保留线性源（`:575-582` 仅 warn）。该门对实际路由**无约束力**，与"门禁应影响路由"的工程意图冲突。

---

## 六、改进建议

### P0（必须，影响正确性/可复现性）

1. **归档 SyQon 模型权重身份** — 文件 `pipeline/stages/stage7_star_separation.py:750-751` 与 `pipeline/syqon_starless.py:199-220`。在 `_write_stage_json("stage6_starless_quality.json", …)`（L1167）的 payload 中增加 `syqon_model={name, weight_path, sha256, axiom}`，由 `syqon_axiom_model_available` 探测路径后计算 SHA-256 写入。使旧 run 可定位权重、满足科学可复现。

2. **评估将 IHS 预拉伸由 arcsinh 改为 MTF 类** — 文件 `resources/siril_plugins/vendor/siril-scripts/processing/SyQon-Starless.py:126-127,875,950`。StarXTerminator 文档明确 arcsinh 会让星点像小星系。若 SyQon 训练分布允许，将 `_ihs` 的 arcsinh 替换为 MTF（`x/(x+k)` 或低 B asinh），或对 `_IHS_TARGET` 进一步下压（如 0.30）以削弱星点轮廓形变。至少应在 `workflow.md` §5.6 明示此风险与缓解。

### P1（应当，影响可观测性/门禁效力）

3. **让 Stage6.5 门建议真正参与路由或显式放弃** — 文件 `pipeline/stages/stage7_star_separation.py:575-582`。要么消费 `recommended_starless_input`（当其为拉伸候选时生成并喂 SyQon），要么在代码注释与 `pre_starless_gate_report.json` 中显式声明"本项目以线性+可逆轻拉伸替代后拉伸，故忽略该建议"，消除"算了却不用"的死逻辑。

4. **阈值来源语义标注** — 文件 `pipeline/models.py:339-417`。为每个 `stage6_/stage7_` 门限字段 docstring 增加 `source=empirical|policy|documented` 与参考链接（如 StarXTerminator 门限对应 rc-astro 文档），使审计可追溯到行业依据。

5. **输入类参数加钳制** — 文件 `pipeline/syqon_starless.py` 与 `SyQon-Starless.py` 常量。`_IHS_TARGET`、`_IHS_B`、各修复强度目前无 Python 侧上下钳；依 AGENTS.md"画质风险参数必须有上限或回退值"，应在 `PipelineConfig` 增加对应上限字段并传入。

### P2（建议，工程整洁度）

6. **逆变换残差断言** — 文件 `SyQon-Starless.py:950`。在输出前对 `apply_ihs_per_channel_inverse` 结果与原始线性域做残差断言（或记录 max|Δ| 进 JSON），防止可逆性实现偏差污染下游线性拉伸。

7. **`reasons[]` 统一 schema** — 所有分支（target_bypass / pre_starless / quality / tool_failed）的 `reasons[]` 已结构良好，但 `target_bypass` 分支 `derived` 全 0（`:517-521`）易与"未评估"混淆，建议标注 `evaluated=false` 区分"评估通过"与"未去星故无指标"。

---

## 七、证据索引

### 代码证据（file:line）
- `pipeline/stages/stage7_star_separation.py:436` — Stage 6 主函数 `run_stage6_star_separation`
- `:446-460` — `_stage8_handoff` 初始化（v1 schema）
- `:476-565` — `target_bypass` 分支（含 state 赋值 L494、JSON L523-544）
- `:575-582` — Stage6.5 门建议被忽略
- `:585-674` — `rejected`（pre_starless_gate）分支
- `:713-751` — SyQon 脚本定位、plan、CLI 选项、模型名仅落日志（L750-751）
- `:768-786` — SASP 回退 / `raise SirilError`
- `:794-813` — starmask 构建与清理
- `:816-826` — 综合质量评估 + 硬门应用
- `:844-951` — 同线性源参数重试
- `:982-1107` — 像素修复与接受/回滚
- `:1127-1146` — `separation_accepted` 与四态 `accepted/rejected` 赋值
- `:1148,1167-1190` — Stage8 交接 + `stage6_starless_quality.json` 写入
- `:1209` — `stage6_starless` / `stage6_passthrough` 落盘
- `:1261-1323` — `tool_failed` 分支
- `:1326-1331` — 历史别名 `run_stage7_star_separation`
- `:103-304` — `_stage8_handoff_from_stage6`（Stage8 路由门）
- `:307-` — `_apply_starmask_cleanup_hard_gate`
- `pipeline/syqon_starless.py:199-220` — `syqon_axiom_model_available`（权重存在性探测，无 SHA）
- `:223-265` — `stage7_try_syqon_variant`（L235 重载固定线性源）
- `pipeline/starmask_cleanup.py` — 多尺度清理 + `diffuse>0.08` 硬门
- `pipeline/stage_support.py:374-` — `_stage7_preflight_check`
- `:1262` — `_build_manual_starmask`（`pm stretched - starless`）
- `pipeline/pipeline_safety.py` — `should_bypass_star_separation` / `STAR_PRESERVE_TARGET_TYPES`
- `pipeline/quality_gate.py:15-80` — `evaluate_pre_starless_gate`（Stage6.5 门）
- `pipeline/models.py:339-417` — 全部 `stage6_/stage7_` 门限默认值
- `pipeline/seestar_Superimpose.py:1095` — `_star_separation_state` 初始化
- `:1318-` — `_record_stage`（进 pipeline-result.json）
- `resources/siril_plugins/vendor/siril-scripts/processing/SyQon-Starless.py:126-127` — `_IHS_B=6.0`、`_IHS_TARGET=0.40`
- `:875` / `:950` — IHS 预拉伸 / 逆变换应用点

### 行业证据（URL）
- StarNet++ 线性预拉伸官方说明：`https://siril.readthedocs.io/en/stable/processing/stars/starnet.html`
- StarNet++ 集成教程（MTF autostretch 推荐）：`https://siril.org/tutorials/integrated-starnet`
- StarXTerminator 用法（线性+内部 MTF+逆变换；arcsinh 警告）：`https://www.rc-astro.com/starxterminator-usage-notes`
- Siril 官方 Zenith/SyQon 公告：`https://siril.org/2026/01/a-brand-new-star-removal-script-comes-to-siril-zenith`
- SyQon 社区"需轻拉伸/后拉伸"经验：`https://ssr.app.astrobin.com/forum/topic/216451/processing/new-star-removal-tool`
- Siril 1.4 第三方 AI 模型并列（StarNet.py/SyQon/StarXTerminator）：`https://siril.readthedocs.io/it/latest/processing/stars/starnet.html`
- starless-first 经典工作流：
  - `https://www.fas37.org/wp/wp-content/uploads/2025/10/Siril-Workflow.pdf`
  - `https://bridgecameraastroimaging.blogspot.com/2024/03/using-starnet-in-siril.html`

### 未验证项
- **未验证**：SyQon/Zenith 模型权重训练集、架构细节、版本校验和均未公开（siril.org 公告仅述设计哲学），无法做权重级复现性校验。
- **未验证**：`_IHS_TARGET=0.40` / `_IHS_B=6.0` 两常量是否为 SyQon 训练分布的推荐值，还是本项目自定；代码注释未给出来源。
- **未验证**：本审计未运行 pipeline（纯静态阅读），所有"行为"结论基于代码路径与常量，未用真实 FITS 验证 arcsinh 陷阱在实际 Seestar 数据上的触发率。

---

*审计结束。本文件为独立研究产物，未修改被测项目任何代码。*
