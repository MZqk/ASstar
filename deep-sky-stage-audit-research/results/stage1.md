# Stage 1 审计报告

> 审计对象：`/Users/mz/dev/aiseestart`（Seestar 智能望远镜离线深空天文后期 pipeline，Python + Siril 1.4.0 驱动）
> 审计范围：Stage 1 前期准备（输入统一 / debayer / register / stack / InputProfile 线性判定 / stage1_prepared 检查点发布）
> 审计性质：纯研究，未修改任何项目代码。
> 覆盖代码：`pipeline/stages/stage1_preparation.py`、`pipeline/stage_support.py`、`pipeline/input_profile.py`、`pipeline/input_discovery.py`、`pipeline/run_manifest.py`、`pipeline/task_plan.py`、`pipeline/processor_runtime.py`、`pipeline/models.py`、`pipeline/seestar_Superimpose.py`、`pipeline/stage_contracts.py`
> 说明：全仓存在历史命名错位（Stage 6 去星实现在 `stage7_star_separation.py`，Stage 7 拉伸在 `stage6_stretching.py`），经核对**不影响 Stage 1 代码链路**。

---

## 摘要与总评分

| 维度 | 评分 | 一句话理由 |
|------|------|-----------|
| D1 逻辑专业合理性 | **8/10** | 预处理顺序（debayer→register→stack）、Winsorized sigma 3/3 加性归一化、InputProfile 多证据链均符合天文后期专业逻辑；扣分点为未做暗/平/偏校准、且 XISF/CFA 相位未校验。 |
| D2 门禁来源与归档 | **7/10** | 门禁清单完整、阈值来源分类清晰、多数可事后复现；扣分点为 register 失败统计仅落自由文本、且缺失 CFA/bit-depth 校验类门禁。 |
| D3 算法行业标准符合度 | **8/10** | debayer/register/stack 实现与 Siril、PixInsight WBPP 主流做法基本一致；扣分点为 XISF→FITS 隐式转换未显式处理 BZERO/BSCALE 与 CFA 相位，且 light 路径无校准帧。 |

**总评**：Stage 1 在专业逻辑与行业对标上总体扎实，核心风险集中在**输入域真实性校验的最后一公里**——依赖 Siril 隐式转换的 XISF/CFA/bit-depth 既无显式处理也无门禁拦截，错误会静默贯穿后续全部阶段。

---

## D1 逻辑专业合理性

### stage_objective
将异构输入（已叠加 `.fit`、单帧 `Light_`、显式母版/复核图、XISF）统一为一张处于**线性域**的 `stage1_prepared.fit`，并完成 OSC 去拜耳、配准与叠加，为后续线性阶段提供可信起点。

### actual_processing_chain（带 file:line）

存在**两条互斥输入路径**：

1. **签名任务清单路径（新任务，主路径）** — `stage1_preparation.py:113-124`
   - `_load_task_run_source()` 校验 run-manifest（schema/hash/contract/read_only/source kind/逐文件 SHA-256）— `stage1_preparation.py:19-80`
   - `light_directory` → `_preprocess_light_frames()` — `stage1_preparation.py:120-122`
   - `master_file` / `review_file` → `_load_explicit_master()`（XISF 经 Siril `load`+`save working` 转 FITS）— `stage1_preparation.py:83-98,124`

2. **旧目录扫描路径（兼容分支）** — `stage1_preparation.py:128-156`
   - `_find_fit_files()` 遍历工作目录 `.fit/.fits` — `stage_support.py:1241-1246`
   - `_is_candidate_stacked()` 用 exclude 前缀/子串/后缀过滤中间产物 — `stage_support.py:1223-1238`
   - 有候选叠加图 → `_load_stacked_file()` 拷贝最新到 `working.fit` — `stage_support.py:1453-1474`
   - 仅 `Light_` 单帧 → `_preprocess_light_frames()` — `stage1_preparation.py:150-152`

**轻帧预处理链**（两条路径共用）— `stage_support.py:1500-1551`，顺序固定：

| 步骤 | 命令 | 位置 |
|------|------|------|
| 1 链接隔离 Light | `link lightsrc -out=..`（先 `_prepare_isolated_light_input` 软链为 `lightsrc_00001.fit`） | `stage_support.py:1476-1497,1510` |
| 2 去拜耳校准 | `calibrate pp_seq -debayer` | `stage_support.py:1515` |
| 3 配准 | `register pp_seq -2pass` → `seqapplyreg pp_seq -filter-round=2.5k` | `stage_support.py:1520-1521` |
| 4 叠加 | `stack r_pp_seq rej 3 3 -norm=addscale -output_norm -rgb_equal -out=working` | `stage_support.py:1524-1527` |
| 5 翻转 | `mirrorx_single working`（Seestar vendor 脚本约定） | `stage_support.py:1529` |

随后 `stage1_prepared.fit` 落盘（`stage1_preparation.py:158`），执行 register 失败率门控（`stage1_preparation.py:164-186`），调用 `_resolve_input_profile()`（`processor_runtime.py:1489-1554`）写 `input_profile.json`，最终由运行入口发布 Stage 1 续跑检查点 `_publish_task_formal_checkpoint(1)`（`seestar_Superimpose.py:2399,2493`）。

### 处理链逐步专业评析（逐命令专业合理性）
- **link 隔离（`stage_support.py:1476-1497,1510`）**：先软链本轮 Light 到 `_light_input/lightsrc_NNNNN.fit`，避免历史 `pp_`/`r_` 序列与大小写差异污染 Siril `link`。这是防御性隔离的正确做法，符合 workflow.md:287 的设计意图；`OSError` 回退 `copy2` 保证跨文件系统可用。
- **calibrate -debayer（`stage_support.py:1515`）**：在 OSC 上先去拜耳再 register/stack 是标准顺序（D3.1）。但此处同时承担"校准"语义却未提供任何主校准帧，意味着 Seestar Light_ 单帧被当作"已机内校准的 RAW"处理。对 Seestar 产品形态可接受，但对通用深空后期属偏离（见 D3.1 偏离分析）。
- **register -2pass（`stage_support.py:1520`）**：2-pass 全局星点对齐对广角/有畸变场更稳健；未显式 `-ref` 取 Siril 默认参考帧（最可信帧）。对 Seestar 短曝光 OSC 合理，但牺牲了参考帧可复现确定性（见 D3.2）。
- **seqapplyreg -filter-round=2.5k（`stage_support.py:1521`）**：按 roundness 2.5k 过滤低质量配准帧，剔除畸变/拖线帧，是合理质量门；阈值为硬编码常量，未集中到 PipelineConfig（轻微违反 AGENTS.md 配置集中原则）。
- **stack rej 3 3 -norm=addscale -output_norm -rgb_equal（`stage_support.py:1524-1527`）**：Winsorized sigma 3/3 是 Siril 默认且对中等规模数据集稳健；加性缩放归一化是行业默认；`-rgb_equal` 在 Stage 4 才做 PCC/SPCC 之前等化 OSC 三通道背景，与后续色彩校准链路自洽；`-output_norm` 将结果缩放到 [0,1] 便于下游统一线性域约定。整体顺序与参数组合专业合理（见 D3.3/D3.4）。
- **mirrorx_single（`stage_support.py:1529`）**：对应 Seestar vendor 脚本的 result 翻转约定（`workflow.md:288`），属设备特化而非通用天文标准步骤；方向需与 Stage 4 platesolve 的天文定向自洽，当前无门禁验证（见 D3.5）。

### domain_assumption（线性/非线性域假设）
- 全部输出约定处于**线性域**：`stage_contracts.py:66-75` 将 Stage 1 标记为 `StagePhase.LINEAR`，primary_artifact `stage1_prepared.fit`。
- `input_profile.py:172-188` 中 `light_preprocess` 模式直接判定 `LINEAR(conf=1.0)`，因为本阶段刚生成 calibrate/register/stack 产物；`stacked`/`master` 模式则走证据链推断。
- 下游（Stage 2-5 线性去星、Stage 6-7 拉伸）依赖该线性假设；`InputProfile.safe_for_linear_steps`（`models.py`，`state==LINEAR and not conflicts`）是后续线性链的总闸。

### upstream_downstream_contract
- **输入契约**：两种路径分别由 run-manifest（签名 + SHA-256）或工作目录扫描决定。只读来源被强制（`stage1_preparation.py:47`）。
- **输出契约**：`stage1_prepared.fit`（checkpoint 主键，见 `stage_contracts.py:72`）；`working.fit` 为 process 内当前加载图。
- **耦合点**：`_stage1_input_mode`（`"stacked"`/`"light_preprocess"`/`"explicit_master"`）驱动 `infer_input_profile` 的证据权重（`processor_runtime.py:1530`）。

### degradation_paths（降级/跳过/失败分支）
- **register 失败率超阈**：阶段置 `degraded` 但**不中断**，继续发布（`stage1_preparation.py:178-186`）——符合「降级必须有回退」原则（`AGENTS.md`）。
- **stage1 输出保存失败**：`stage_status="degraded"` + message（`stage1_preparation.py:158-162`）。
- **无 `.fit` 文件**：抛 `SirilError` 终止（`stage1_preparation.py:153-156`）。
- **InputProfile=UNKNOWN / requires_review**：后续线性链被跳过，仅生成复核输出（`processor_runtime.py:1550-1553`）。
- **软链失败回退**：`_prepare_isolated_light_input` 在 `OSError` 时回退到 `copy2`（`stage_support.py:1491-1494`）。

### professional_soundness：合理（含两处存疑）
- 合理点：debayer 在 register/stack 之前（OSC 标准顺序，见 D3）；Winsorized sigma 3/3 对大量短曝光 OSC 子帧是稳健默认；`-norm=addscale` 加性缩放是行业默认归一化；`-rgb_equal` 在不使用 PCC/SPCC（Stage 4）前等化 OSC 背景是合理兜底；`mirrorx_single` 对应 Seestar vendor 脚本约定（`workflow.md:288`）。
- 存疑点 1：**light 路径不做暗/平/偏校准**——`calibrate -debayer` 未附带任何主校准帧（代码搜索确认 `pipeline/` 内 Stage 1 路径无 dark/flat/bias 引用）。这符合 Seestar「机内已校准、导出 Light_ 单帧」的产品形态，但偏离 PixInsight/Siril WBPP 完整预处理范式（见 D3）。
- 存疑点 2：**XISF/CFA 相位与 bit-depth 无显式处理**（见 D3 已知陷阱）。

### risks（已识别专业性风险）
1. **CFA 相位静默错误（高）**：`calibrate -debayer` 完全依赖 Siril 从 FITS 头 `BAYERPAT` 推断 CFA 图案；XISF→FITS 经 `load`/`save` 隐式转换（`stage1_preparation.py:89-97`），若转换未保留 `ROWORDER`/`XBAYROFF`/`YBAYROFF`，CFA 相位错位会导致系统性偏色且**无任何门禁拦截**，错误贯穿全部 10 个阶段。
2. **bit-depth / BZERO-BSCALE 未校验（中）**：Stage 1 未显式处理 `BZERO`/`BSCALE`/signed→unsigned 缩放（代码搜索确认 `pipeline/` 内无相关处理），依赖 Siril 隐式转换；整数缩放不当会污染线性统计与后续归一化。
3. **无校准帧（中，设备特化）**：Seestar Light_ 单帧未做暗场平场，弱信号本底与固定图形噪声未被抑制，影响 Stage 3 背景提取与 Stage 5 降噪前提。
4. **参考帧未显式指定（低）**：`register -2pass` 未用 `-ref` 指定参考帧，采用 Siril 默认（最亮/最多星帧）；对 Seestar 短曝光可接受，但缺乏可复现确定性。

---

## D2 门禁来源与归档

### gate_inventory（全部门禁）

| # | 门禁名称 | 判定位置 file:line | 触发动作 | 阈值来源分类 |
|---|----------|--------------------|----------|--------------|
| G1 | run-manifest schema 校验 | `stage1_preparation.py:30-31` | 抛错终止 | hardcoded_literal (`seestar.task-run.v1`) |
| G2 | manifest_hash 签名校验 | `stage1_preparation.py:35-38` | 抛错终止 | runtime_derived (`canonical_payload_hash`) |
| G3 | pipeline_contract 兼容校验 | `stage1_preparation.py:39-45` | 抛错终止 | hardcoded_literal (`PIPELINE_CONTRACT_SCHEMA/VERSION`) |
| G4 | read_only 来源校验 | `stage1_preparation.py:47-48` | 抛错终止 | hardcoded_literal |
| G5 | source kind 白名单 | `stage1_preparation.py:49-51` | 抛错终止 | hardcoded_literal (`_TASK_SOURCE_KINDS`) |
| G6 | 文件存在/大小/SHA-256 | `stage1_preparation.py:56-73` | 抛错终止 | runtime_derived (`run_manifest.sha256_file`) |
| G7 | 单文件任务数量=1 | `stage1_preparation.py:74-75` | 抛错终止 | hardcoded_literal |
| G8 | 无输入文件 | `stage1_preparation.py:153-156` | 抛错终止 | hardcoded_literal |
| G9 | register 失败率门控 | `stage1_preparation.py:164-186` | 置 `degraded`（不中断） | PipelineConfig (`stage1_register_fail_ratio_max=0.10`) |
| G10 | stage1 输出保存 | `stage1_preparation.py:158-162` | 置 `degraded` | runtime_derived (`_save_stage_output`) |
| G11 | InputProfile 线性/冲突门 | `input_profile.py:313-375` + `processor_runtime.py:1550-1553` | UNKNOWN→跳过线性链 | runtime_derived（多证据启发式） |

### threshold_provenance
- **hardcoded_literal**：G1、G3、G4、G5、G7、G8（schema 字符串、白名单集合、数量常量）——硬编码在代码中，无外部覆盖。
- **runtime_derived**：G2（`run_manifest.canonical_payload_hash`，`run_manifest.py:126-133`）、G6（逐文件 `sha256_file`，`run_manifest.py:20-41`）、G9 的 `fail_ratio`（运行时统计）、G10、G11。
- **PipelineConfig**：G9 的 `ratio_limit` 来自 `models.py:100`（`stage1_register_fail_ratio_max: float = 0.10`），经 `max(0.0, min(1.0, …))` 钳制（`stage1_preparation.py:170-173`）。
- **env_override**：run-manifest 路径由 `SEESTAR_TASK_RUN_MANIFEST` 环境变量提供（`stage1_preparation.py:12,22`），但该变量仅作为入口指针，不参与阈值计算。

### config_centralization（是否符合 AGENTS.md「参数集中 PipelineConfig」）
符合。可调阈值 `stage1_register_fail_ratio_max` 集中在 `PipelineConfig`（`models.py:100`）；排除规则 `exclude_prefixes/substrings/suffixes` 也集中在 `models.py:432-441`。无散落字面量阈值（除 schema 字符串等不可配置常量外）。

### env_override_control（env 覆盖是否有钳制/上下限）
- `SEESTAR_TASK_RUN_MANIFEST`：仅作路径指针，且强制要求位于当前运行目录内（`stage1_preparation.py:27-28` 跨目录即抛错），有位置钳制。
- `stage1_register_fail_ratio_max`：在门控处有 `max(0.0, min(1.0, …))` 上下限钳制（`stage1_preparation.py:170-173`），比例域合法。
- 其余门禁无 env 覆盖入口，无越界风险。

### archival_completeness（判定结果与阈值写入何处）

| 产物 | 内容 | 位置 |
|------|------|------|
| run-manifest（输入） | schema/hash/contract/read_only/source 文件清单+SHA-256 | `stage1_preparation.py:29-73` 读取并校验 |
| processing-plan.json | 冻结 config（含 `stage1_register_fail_ratio_max`）、`stage1_input_mode` | `processor_runtime._write_processing_plan`（摘要确认；未逐行核对） |
| pipeline-result.json | `input_profile` 全量证据链（`processor_runtime.py:2091`）、`actual_steps[].message`（含 register 统计文本，`processor_runtime.py:2148`）、`checkpoints` | `processor_runtime.py:2038-2174` |
| input_profile.json | InputProfile 字典（证据+冲突+state+confidence） | `processor_runtime.py:1541` |
| checkpoint-manifest.json | Stage 1 续跑断点：`stage1_prepared.fit` 的 sha256/state/config_fingerprint | `seestar_Superimpose.py:1425-1447` 经 `task_workspace.publish_formal_checkpoint` |
| 日志 | `[PIPELINE_STAGE_RESULT]`/`[PIPELINE_STAGE_DETAIL]`/`[TaskCheckpoint]` | `seestar_Superimpose.py:1364-1393,1440-1442` |

### reproducibility（仅凭归档产物能否事后复现该 gate 判定）
- **G1–G8、G10：可复现**——阈值与输入哈希均落盘，对照 run-manifest / processing-plan.json 可精确重算。
- **G9（register 失败率门控）：部分可复现**——`ratio_limit` 在 processing-plan.json 中可复现；但 `fail_ratio` 的**结构化数值**（total/registered/failed）仅以自由文本写入 `actual_steps[].message`（`stage1_preparation.py:174-186` 的 `stats_msg`），**未作为独立 JSON 字段**持久化。`_stage1_registration_stats` 仅存于运行时对象。r_pp_seq_*.fit 中间帧在下次 `_prepare_process_dir` 重建时会被清除（`stage1_preparation.py:111` 调用 `_prepare_process_dir`），故事后无法重新 glob 计数，只能依赖 message 文本解析。
- **G11（InputProfile 门）：可复现**——`input_profile.json` 与 pipeline-result.json 均保留完整证据链与冲突列表，可精确重算 UNKNOWN 决策。

### 门禁逐条可复现性叙述
- **G1 schema**：阈值 `seestar.task-run.v1` 硬编码（`stage1_preparation.py:13`），run-manifest 中 `schema` 字段落盘，对照即可复现。
- **G2 hash**：`canonical_payload_hash`（`run_manifest.py:126-133`）对去除 `manifest_hash` 后的 payload 做规范 JSON 哈希；源码与 run-manifest 双存档，可精确重算。
- **G3 contract**：`PIPELINE_CONTRACT_SCHEMA/VERSION` 硬编码（`stage_contracts.py:15-16`），run-manifest 中 `pipeline_contract` 落盘，可复现。
- **G4/G5**：`read_only=true` 与 `source.kind ∈ {master_file,light_directory,review_file}` 均为硬编码常量（`stage1_preparation.py:14-16,47,50`），可复现。
- **G6 文件 SHA-256**：逐文件 `sha256_file`（`run_manifest.py:20-41`）与 run-manifest 中 `files[].sha256`/`size` 双写，事后可逐文件重算比对，可复现。
- **G7 单文件数量**：`len(files)!=1` 硬编码于 `stage1_preparation.py:74`，可复现。
- **G8 无输入**：`stage1_preparation.py:153` 硬编码抛错，可复现。
- **G9 register 失败率**：`ratio_limit` 来自 processing-plan.json 的 `stage1_register_fail_ratio_max`（可复现）；但 `fail_ratio` 结构化数值未落独立 JSON 字段（见 gaps 1），**部分可复现**——只能从 `actual_steps[].message` 文本解析 `failed=X/Y (Z%, limit=W%)`。
- **G10 输出保存**：`_save_stage_output` 返回布尔（`stage1_preparation.py:158`）写入 message，可复现。
- **G11 InputProfile**：`input_profile.json` + pipeline-result.json 的 `input_profile` 均保留完整证据链与冲突（`processor_runtime.py:1541,2091`），可精确重算 UNKNOWN 决策，可复现。

### gaps（归档缺口清单）
1. **G9 数值未结构化**：register 失败统计（total/registered/failed/fail_ratio）应作为 `actual_steps[].details` 结构化字段（当前 `_record_stage` 调用未传 `details`，见 `stage1_preparation.py:189-194`、`seestar_Superimpose.py:1318-1348`），否则仅自由文本可溯源。
2. **缺失 CFA/bit-depth 校验门**：D1 风险 1/2 所指——Stage 1 没有任何门禁验证 `BAYERPAT`/`ROWORDER`/`XBAYROFF`/`BZERO`/`BSCALE` 的正确性，导致"输入域真实性"这一最关键门禁事实缺位。
3. **XISF→FITS 转换元数据未记录**：转换过程未把源 XISF 的 CFA/bit-depth 元数据写回归档证据，事后难证明转换无损。

---

## D3 算法行业标准符合度

### D3.1 Debayer（去拜耳）
- **algorithm_used**：`calibrate pp_seq -debayer`（`stage_support.py:1515`）。Siril 在校准后执行去拜耳，CFA 图案从 FITS 头 `BAYERPAT` 推断；插值方法取 Siril 首选项（代码未指定，默认由 Siril 配置决定）。
- **industry_baseline**：行业共识为「**先校准、后去拜耳**」，去拜耳必须在 RAW 未插值子帧上、且在 register/stack 之前完成。PixInsight：`https://telescope.live/blog/pixinsight-debayer-demosaicing-process-explained`；Altair：`https://www.altairastro.help/info-instructions/cmos/what-is-debayering-and-why-should-i-do-it-before-stacking-my-images/`。
- **conformance**：**符合**（顺序正确）。
- **deviation_analysis**：无校准帧（D1 存疑点 1）。PixInsight/Siril WBPP 默认带 master bias/dark/flat（`https://utahdesertremote.com/improve-your-astrophotography-with-weighted-batch-preprocessing`）。Seestar 机内已做校准、导出 Light_ 单帧，属设备特化，合理但偏离完整预处理范式。
- **known_pitfalls**：**CFA 相位错位**是最经典陷阱——`BAYERPAT` 正确但 `XBAYROFF`/`YBAYROFF`/`ROWORDER` 偏移会导致 RGGB→GRBG 级错位，产生系统性偏色且肉眼难辨（PixInsight 文档：`https://sh-cosmiccanvas.s3.us-west-2.amazonaws.com/Resources/20231210_PreprocessingOfRawImageDataWithPixInsight.pdf`；CloudyNights 讨论：`https://www.cloudynights.com/forums/topic/867412-rggb-grbg-gbrg-bggr-lets-call-the-whole-thing-off/`）。**代码未规避**：Stage 1 不读取/校验 `XBAYROFF`/`YBAYROFF`/`ROWORDER`，仅把 `BAYERPAT` 存在与否当作"线性采集"证据（`input_profile.py:32,243`），未用于驱动去拜耳；XISF→FITS 转换的 CFA 相位保真性无验证（D1 风险 1）。

### D3.2 Register（配准）
- **algorithm_used**：`register pp_seq -2pass` + `seqapplyreg pp_seq -filter-round=2.5k`（`stage_support.py:1520-1521`）。2-pass 全局星点对齐，参考帧由 Siril 默认选择（未用 `-ref`）。
- **industry_baseline**：Siril 注册默认采用 Homography（8 自由度）参考变换，参考帧为序列中质量最佳者（`https://gitlab.com/free-astro/siril-doc/blob/1.2/doc/preprocessing/registration.rst`）。`-filter-round=2.5k` 按 roundness 过滤低质量配准帧。
- **conformance**：**符合**。
- **deviation_analysis**：无显式 `-ref` 指定，依赖 Siril 默认；对 Seestar 短曝光 OSC 合理。
- **known_pitfalls**：参考帧选择不确定性影响可复现性；`-filter-round=2.5k` 阈值未集中到 PipelineConfig（硬编码在 `stage_support.py:1521`），轻微违反配置集中原则。

### D3.3 Stack rejection（叠加拒绝算法）
- **algorithm_used**：`stack r_pp_seq rej 3 3`（`stage_support.py:1524-1527`）→ **Winsorized Sigma Clipping**，低/高 sigma 均为 3。
- **industry_baseline**：Siril 平均叠加默认即 Winsorized sigma clipping（`https://siril.readthedocs.io/es/1.2/preprocessing/stacking.html`）。PixInsight 经验法则：3–5 帧用 Percentile/Averaged Sigma；5–10 用 Sigma；10–20 用 Winsorized 或 Linear Fit；>15–20 用 Linear Fit 更优（`http://www.astral-imaging.com/pi_processing_II.htm`）。PixInsight 默认/推荐亦支持 GESDT 与 Linear Fit Clipping（`https://pixinsight.com/forum.old/index.php?topic=2169.msg14163`、`https://utahdesertremote.com/improve-your-astrophotography-with-weighted-batch-preprocessing`）。
- **conformance**：**符合**（Winsorized 是 Siril 默认且对中等规模数据集稳健）。
- **deviation_analysis**：Seestar 通常产生大量（数十~数百）短曝光子帧，Winsorized 3/3 是合理默认；但 PixInsight 对大集更推荐 Linear Fit（抗梯度）或 GESDT（抗飞机/卫星轨迹）。当前固定 `rej 3 3` 未随帧数自适应。
- **known_pitfalls**：固定 sigma 3/3 在极端 outlier（如强卫星轨迹）下可能拒绝不足或过度；Siril 官方建议拒绝率应落在 0.1%–0.5%（`https://siril.org/tutorials/tuto-manual/`），代码未对实际拒绝率做反馈校验。

### D3.4 Normalization & rgb_equal
- **algorithm_used**：`-norm=addscale -output_norm -rgb_equal`（`stage_support.py:1524-1527`）。
- **industry_baseline**：Siril 平均叠加默认加性归一化（`https://siril.readthedocs.io/es/1.2/preprocessing/stacking.html`）；`-rgb_equal` 等化 OSC 三通道背景，文档明确"在 PCC/SPCC 或未链接 AUTOSTRETCH 不使用时有用"（`https://siril.readthedocs.io/it/stable/Commands.html`）。PixInsight 对应 "Additive with Scaling"（`https://chaoticnebula.com/pixinsight-image-integration/`）。
- **conformance**：**符合**（且 `-rgb_equal` 与 Stage 4 才做色彩校准的链路自洽）。
- **deviation_analysis**：无。
- **known_pitfalls**：`-output_norm` 将结果缩放到 [0,1]，若下游误将其当作非归一化线性数据使用会引入比例歧义——但本 pipeline 全程线性域约定明确，风险低。

### D3.5 mirrorx_single（翻转）
- **algorithm_used**：`mirrorx_single working`（`stage_support.py:1529`）。
- **industry_baseline**：非通用行业标准，而是 Seestar vendor 预处理脚本的约定（`workflow.md:288`："vendor Seestar 脚本中的 mirrorx_single result 在这里对应为 mirrorx_single working"）。
- **conformance**：**自创/设备特化**（符合 Seestar 输出约定，但非通用天文后期标准步骤）。
- **deviation_analysis**：合理——匹配 Seestar 导出图的方向约定，保证与厂商参考图一致。
- **known_pitfalls**：若后续 Stage 依赖绝对方向（如 platesolve 在 Stage 4），翻转需与天文定向一致；当前无门禁验证翻转后方向与元数据匹配。

### D3.6 XISF→FITS 转换
- **algorithm_used**：`_load_explicit_master` 中 `load`(XISF) + `save working`（`stage1_preparation.py:89-97`）。
- **industry_baseline**：XISF 是 PixInsight 原生格式，元数据比 FITS 更严格结构化；FITS 的 `BAYERPAT`/`XBAYROFF`/`YBAYROFF`/`ROWORDER` 为非标准关键字，不同软件支持不一致（`https://sh-cosmiccanvas.s3.us-west-2.amazonaws.com/Resources/20231210_PreprocessingOfRawImageDataWithPixInsight.pdf`）。
- **conformance**：**部分符合**（转换发生，但保真性无验证）。
- **deviation_analysis**：依赖 Siril 隐式转换，未显式映射/校验 CFA 与 bit-depth 元数据（D1 风险 1/2）。
- **known_pitfalls**：XISF→FITS 中 CFA 偏移与 `BZERO`/`BSCALE` 缩放若丢失，会导致 D3.1 的 CFA 错位与 D1 的 bit-depth 污染，且**无任何门禁可捕获**（与 D2 缺口 2 同源）。

### D3.7 InputProfile 线性判定（非传统"算法"，作为防御性控制评估）
- **algorithm_used**：多证据启发式（`input_profile.py:152-375`）：运行时 provenance(1.0) → verified_manifest(0.99) → FITS 头 LINEAR/STRETCHED/NONLINEAR/STRETCH(0.96) → processing_text 历史 token → acquisition_keys(BAYERPAT/STACKCNT)(0.88) → 像素分布(p50/p99/black/highlight, 0.68–0.80)；冲突→UNKNOWN(0.20)。
- **industry_baseline**：PixInsight/WBPP **假设输入即线性**，不自动检测传递函数状态；行业普遍依赖用户声明或 FITS 关键字。本 pipeline 的多证据 + 冲突→UNKNOWN→跳过线性链，实际**比行业通用做法更严谨**。
- **conformance**：**超越行业常规（自创但合理）**。
- **deviation_analysis**：像素分布阈值（如 `p50>=0.075` 判非线性，`input_profile.py:251-265`）为经验硬编码，未集中到 PipelineConfig；不同目标/曝光下可能误判，但 UNKNOWN 兜底安全。
- **known_pitfalls**：像素统计阈值对窄带/高背景目标可能过激判非线性；当前无校准帧（D3.1）使线性分布统计前提偏弱。

---

### D3 总体符合度矩阵

| 算法/控制 | 实现 | 行业对标 | 符合度 | 主要偏离 |
|-----------|------|----------|--------|----------|
| Debayer 顺序 | `calibrate -debayer`（先校准后去拜耳） | Siril/PixInsight 共识 | 符合 | 无校准帧（设备特化） |
| CFA 图案来源 | Siril 从 `BAYERPAT` 推断 | 依赖头关键字 | 部分符合 | 不校验 XBAYROFF/YBAYROFF/ROWORDER（D3.1） |
| Register | `register -2pass` + `seqapplyreg` | Siril 默认 Homography/2-pass | 符合 | 参考帧未显式指定 |
| Stack 拒绝 | `rej 3 3` Winsorized Sigma | Siril/Siril 默认；PixInsight 中小样本推荐 | 符合 | 大样本未切 Linear Fit/GESDT |
| 归一化 | `-norm=addscale` | PixInsight Additive with Scaling | 符合 | 无 |
| OSC 等化 | `-rgb_equal` | Siril 文档建议（无 PCC/SPCC 时） | 符合 | 无 |
| 方向翻转 | `mirrorx_single` | Seestar vendor 约定 | 自创/设备特化 | 非通用标准步骤 |
| XISF→FITS | `load`+`save` 隐式 | XISF 严格元数据 | 部分符合 | CFA/bit-depth 保真性无验证 |
| InputProfile 线性判定 | 多证据启发式+冲突→UNKNOWN | 行业多假设线性输入 | 超越常规（更严谨） | 像素阈值硬编码 |

### D3 偏离的"合理性"总判定
- **设备特化类（合理）**：无校准帧、mirrorx 翻转、`-rgb_equal` 兜底——均源于 Seestar"机内已校准 + 导出单帧/母版"的产品形态，与 Stage 4 才做色彩校准的链路自洽，属合理偏离。
- **隐式转换类（存疑/需补门禁）**：XISF→FITS 的 CFA 相位与 BZERO/BSCALE、以及固定 `rej 3 3` 未随帧数自适应——前者是"合理但缺校验"的真实风险，后者是"可接受默认但非最优"，均应在 P0/P1 中闭环。

## 关键发现 TOP3

1. **CFA 相位 / XISF 转换保真性既无显式处理也无门禁（最严重）**：`calibrate -debayer` 与 XISF→FITS 完全依赖 Siril 隐式转换；代码不读取 `XBAYROFF`/`YBAYROFF`/`ROWORDER`、不校验 `BZERO`/`BSCALE`（`stage_support.py:1500-1551`、`stage1_preparation.py:89-97`）。CFA 相位错位会系统性偏色并静默贯穿全部 10 阶段（依据 D3.1/D3.6 行业标准陷阱：`https://www.cloudynights.com/forums/topic/867412-rggb-grbg-gbrg-bggr-lets-call-the-whole-thing-off/`）。
2. **register 失败率门控数值未结构化归档**：G9 的 `fail_ratio` 仅以自由文本写入 `actual_steps[].message`（`stage1_preparation.py:174-186`），中间帧在下次运行被清，事后只能文本解析，**部分可复现**（D2 reproducibility）。
3. **light 路径缺校准帧、且 stack 拒绝算法未随帧数自适应**：`calibrate -debayer` 无 dark/flat/bias（`stage_support.py:1515`），固定 `rej 3 3` 未对大样本切到 Linear Fit/GESDT（D3.1/D3.3）；属 Seestar 设备特化可接受，但应在文档与门禁中明确声明。

---

## 改进建议（P0/P1/P2，可执行 + 文件位置）

### P0（必须修，影响数据真实性）
- **P0-1 增加 CFA/bit-depth 输入校验门禁**：在 `_load_explicit_master`（`stage1_preparation.py:83-98`）与 `_preprocess_light_frames`（`stage_support.py:1500-1551`）入口处，读取并打印 `BAYERPAT`/`XBAYROFF`/`YBAYROFF`/`ROWORDER`/`BZERO`/`BSCALE`；对 XISF 源在 `load`/`save` 前后比对 CFA 元数据一致性；缺失/错位即置 `degraded` 或终止，并写入证据。填补 D2 缺口 2 与 D1 风险 1/2。
- **P0-2 将 register 统计结构化为 `details`**：在 `stage1_preparation.py:189-194` 调用 `_record_stage` 时传入 `details=pipeline._stage1_registration_stats`，使 `actual_steps[].details` 携带 total/registered/failed/fail_ratio（`seestar_Superimpose.py:1318-1348` 已支持 `details` 字段）。修复 D2 部分可复现。

### P1（应修，提升专业度与可复现性）
- **P1-1 校准帧可选支持**：在 `stage_support.py:1515` 的 `calibrate -debayer` 增加可选的 `-dark/-flat/-bias` 分支（从 run-manifest 或 config 读取），至少对 Seestar 导出的 RAW 单帧开放；即便默认空，也明确声明"无校准"并记入 InputProfile 证据。
- **P1-2 stack 拒绝算法随帧数自适应**：将 `rej 3 3` 改为由 PipelineConfig 驱动（`models.py:100` 附近新增 `stage1_stack_rejection=(3,3)` 及大样本→Linear Fit 的切换阈值），对齐 PixInsight 经验法则（`http://www.astral-imaging.com/pi_processing_II.htm`）。
- **P1-3 参考帧确定性**：在 `stage_support.py:1520` 增加显式 `-ref` 选择（如最多星帧）或记录 Siril 选定参考帧名到证据，提升可复现性。

### P2（建议，长期完善）
- **P2-1 配置集中**：将 `-filter-round=2.5k`（`stage_support.py:1521`）、像素分布阈值（`input_profile.py:251-265`）等硬编码常量迁入 `PipelineConfig`（`models.py`），落实 AGENTS.md 配置集中原则。
- **P2-2 拒绝率反馈校验**：stack 后读取 Siril 拒绝率，与 Siril 官方 0.1%–0.5% 区间（`https://siril.org/tutorials/tuto-manual/`）比对，越界告警。
- **P2-3 XISF 转换元数据归档**：在 `_load_explicit_master` 将源 XISF 的 CFA/bit-depth 元数据写入 `input_profile.json` 证据（`processor_runtime.py:1541`），补全 D2 缺口 3。

---

## 证据索引

**代码证据（file:line）**
- `pipeline/stages/stage1_preparation.py:19-80` — `_load_task_run_source` 清单/schema/hash/contract/read_only/source/SHA-256 校验
- `pipeline/stages/stage1_preparation.py:83-98` — `_load_explicit_master` XISF→FITS 隐式转换（`load`+`save`）
- `pipeline/stages/stage1_preparation.py:113-124` — 签名路径分支
- `pipeline/stages/stage1_preparation.py:128-156` — 旧目录扫描兼容路径
- `pipeline/stages/stage1_preparation.py:158-162` — stage1 输出保存门
- `pipeline/stages/stage1_preparation.py:164-186` — register 失败率门控 + 钳制
- `pipeline/stages/stage1_preparation.py:189-194` — `_record_stage` 调用（未传 details）
- `pipeline/stage_support.py:1223-1238` — `_is_candidate_stacked` 排除规则
- `pipeline/stage_support.py:1241-1246` — `_find_fit_files`
- `pipeline/stage_support.py:1453-1474` — `_load_stacked_file`
- `pipeline/stage_support.py:1476-1497` — `_prepare_isolated_light_input` 软链隔离
- `pipeline/stage_support.py:1500-1551` — `_preprocess_light_frames` 预处理链（link/calibrate -debayer/register -2pass/seqapplyreg/stack rej 3 3 -norm=addscale -output_norm -rgb_equal/mirrorx_single）
- `pipeline/stage_support.py:292-300` — `_count_sequence_products` 配准计数
- `pipeline/input_profile.py:152-375` — `infer_input_profile` 多证据线性判定与冲突→UNKNOWN
- `pipeline/input_profile.py:32,243` — `BAYERPAT` 仅作线性证据，未驱动去拜耳
- `pipeline/run_manifest.py:20-41` — `sha256_file`
- `pipeline/run_manifest.py:126-133` — `canonical_payload_hash`
- `pipeline/models.py:100` — `stage1_register_fail_ratio_max=0.10`
- `pipeline/models.py:432-441` — `exclude_prefixes/substrings/suffixes`
- `pipeline/processor_runtime.py:1489-1554` — `_resolve_input_profile` 写 input_profile.json
- `pipeline/processor_runtime.py:2038-2174` — `_write_pipeline_result_manifest`（input_profile:2091, actual_steps message:2148）
- `pipeline/seestar_Superimpose.py:1318-1423` — `_record_stage` / PIPELINE_STAGE_DETAIL 日志
- `pipeline/seestar_Superimpose.py:1425-1447` — `_publish_task_formal_checkpoint` 发布 stage1_prepared 断点
- `pipeline/seestar_Superimpose.py:2399,2493` — `_publish_task_formal_checkpoint(1)` 调用点
- `pipeline/stage_contracts.py:18,66-75` — `FORMAL_RESUME_STAGES=(1,2,5)`；Stage 1 线性域 + primary_artifact=stage1_prepared.fit
- `pipeline/seestar_Superimpose_workflow.md:267-300` — §5.1 阶段 1 说明（预处理链顺序、失败率门控、mirrorx 约定）

**行业基准证据（URL）**
- Siril 手动预处理（拒绝率 0.1–0.5%、register/stack 顺序）：`https://siril.org/tutorials/tuto-manual/`
- Siril 叠加拒绝算法（Winsorized Sigma / Linear Fit / GESDT）：`https://siril.readthedocs.io/es/1.2/preprocessing/stacking.html`
- Siril 命令参考（stack rej / -norm=addscale / -rgb_equal / -output_norm / -filter-round）：`https://siril.readthedocs.io/it/stable/Commands.html`
- Siril 注册文档（参考帧、2-pass、Homography 默认）：`https://gitlab.com/free-astro/siril-doc/blob/1.2/doc/preprocessing/registration.rst`
- PixInsight Linear Fit Clipping 算法说明：`https://pixinsight.com/forum.old/index.php?topic=2169.msg14163`
- PixInsight WBPP 拒绝算法建议（Winsorized/GESDT）：`https://utahdesertremote.com/improve-your-astrophotography-with-weighted-batch-preprocessing`
- PixInsight 拒绝算法按数据集规模选择经验法则：`http://www.astral-imaging.com/pi_processing_II.htm`
- PixInsight Additive with Scaling 归一化：`https://chaoticnebula.com/pixinsight-image-integration/`
- 去拜耳须先于校准后、顺序共识：`https://telescope.live/blog/pixinsight-debayer-demosaicing-process-explained`、`https://www.altairastro.help/info-instructions/cmos/what-is-debayering-and-why-should-i-do-it-before-stacking-my-images/`
- CFA 相位/ROWORDER/XBAYROFF/BAYERPAT 错位陷阱：`https://www.cloudynights.com/forums/topic/867412-rggb-grbg-gbrg-bggr-lets-call-the-whole-thing-off/`、`https://sh-cosmiccanvas.s3.us-west-2.amazonaws.com/Resources/20231210_PreprocessingOfRawImageDataWithPixInsight.pdf`

**未验证项（uncertainty）**
- `processing-plan.json` 的精确字段结构（config 冻结、`stage1_input_mode` 落盘）依据摘要推断，未逐行核对 `processor_runtime._write_processing_plan`（约 1848–1998 行）——**未验证：未读取该段源码**。
- Seestar Light_ 单帧是否含机内校准信息（影响 D3.1 偏离合理性判断）——**未验证：未在样本数据上检查 FITS 头**。
- Siril `calibrate -debayer` 默认插值方法（VNG/Bilinear/SuperPixel）及其对 Seestar RGGB 的适配——**未验证：取决于运行环境 Siril 首选项，代码未指定**。
- 本次审计未运行 pipeline，所有结论基于静态代码与文档研读，未经动态执行验证。
