# Stage 7（主体拉伸 / Starless Stretch）深度审计报告

- **审计对象**：`/Users/mz/dev/aiseestart` — Seestar 望远镜离线深空后期 pipeline（Python + Siril 1.4.0）
- **审计阶段**：Stage 7 `stretching` — 全链路唯一的「线性 → 非线性」不可逆转换点
- **命名错位警告**：Stage 7 的阶段封装位于 `pipeline/stages/stage6_stretching.py`（文件名带 6、语义是 7）；`run_stage7_stretching()` 在 `stage6_stretching.py:61`，`run_stage6_stretching()`（`:221`）是弃用兼容别名。主服务实现 `_run_stage6_ai_stretching()` 位于 `pipeline/stage6_services.py:2646-3477`（函数名带 6、语义是 7）。
- **主代码**：`pipeline/stages/stage6_stretching.py`（227 行）、`pipeline/stage6_services.py`（Stage 7 相关约 1400 行）、`pipeline/stage7_stretch_metrics.py`、`pipeline/ai_advisory.py`、`pipeline/models.py`
- **归档/契约**：`pipeline/save_utils.py`、`pipeline/processor_runtime.py`、`pipeline/policy_selector.py`
- **文档基准**：`pipeline/seestar_Superimpose_workflow.md` §5.7（L534-577）、§5.7.1（L578-601）
- **审计性质**：纯只读研究，未修改任何项目代码
- **审计日期**：2026-08-05

---

## 一、摘要与总评分

Stage 7 是整条 pipeline 中「唯一一次不可逆的域转换」：它把 Stage 6 输出的 starless 线性图变换到非线性显示域，之后 Stage 8（星云增强）、Stage 9（回星）、Stage 10（导出）全部在此结果上叠加。一旦此处压死暗部或烧掉核心，后续任何阶段都无法恢复。

总体判断：**这是本项目审计至今架构最严谨的阶段之一**。它具备三项在业余自动化 pipeline 中罕见的正确设计：(a) 黑场从实测噪声地板反推并强制留出安全余量，从数学上杜绝了裁剪暗部；(b) AI 只被允许返回候选 ID，所有数值参数由代码拥有，且 AI 只能在硬门之后的白名单里挑；(c) 候选级度量向量与逐项门禁结果完整落盘，可脱机复现选择过程。

但同时存在三类系统性问题：**(1) 大量 legacy 死代码与死配置与新架构并存**（旧多候选策略、`stretch_candidate_evaluator.py`、policy YAML 的 `scoring`/`hard_reject` 权重全部未被调用），审计者与后续维护者极易依据错误代码路径下判断；**(2) 最核心的四条硬门阈值没有任何 env 覆盖通道**，与周边大量次要阈值可 env 覆盖形成割裂；**(3) AI 请求未归档 prompt 版本标识**，prompt 漂移后无法追溯。

| 维度 | 评分 | 理由 |
|---|---|---|
| **D1 逻辑专业合理性** | **8.0 / 10** | 算法选型（Asinh 主候选 + Asinh/GHS 或 noise-floor linked MTF 副候选）契合深空线性数据特征；黑场处理是全项目最专业的一处——`stage6_services.py:2515-2529` 把 MTF shadows 取在 `min(p01, min) − shadow_margin` 并 `max(0.0, ...)` 钳制，`:2282-2315` 对极低背景把 Asinh offset 压到背景地板之下，从构造上不可能裁暗部；高光侧有 `highlight_clip_ratio_max=0.010` 硬门 + `highlight_scale` 预标定 + 局部核心裁剪门三重防护；AI 契约（只返 ID、只在 accepted 白名单里选、色度救援候选不暴露）设计正确。扣分点：legacy 死代码大面积并存（`:501/:541/:738` 与 `stretch_candidate_evaluator.py` 全链未调用）；Stage 7 兼容检查点在主流程被无条件记为 `skipped`（`seestar_Superimpose.py:2530-2534`），其去星前质量诊断能力实际失效；`_run_with_stars_review_stretch`（`stage6_stretching.py:8`）在去星失败时只做 `autostretch -linked` 复核预览，完全不走本阶段任何硬门。 |
| **D2 门禁来源与归档** | **7.5 / 10** | 归档面在全项目中最完整：`stage7_stretch_quality.json`（`stage6_services.py:3360-3383`）含每候选完整 attempt 对象 + `selection_summary` + baseline/preview 参照；`stretch_candidates_report.json`（`:3384-3439`）含逐候选 `params`/`quality_ok`/`risk_score`/`pixel_stats`/`background_quality_gate`/`diagnostics`/`selection_rank`/`selection_role`；AI 原始响应写 `ai_raw_*.json`（`save_utils.py:112-150`）。**仅凭归档可复现「为何选 A」——判定为「可复现」**（排序键是纯函数且输入全部落盘）。扣分点：`stage6_bg_median_min`/`stage6_black_pixel_ratio_max`/`stage6_highlight_clip_ratio_max`/`stage6_star_growth_ratio_max` 四条核心硬门**不在 `PROJECT_ENV_ALLOWED_KEYS`**（`processor_runtime.py:170-232` grep 无匹配），而 20+ 条次要阈值可覆盖；AI 请求无 `prompt_version`/`schema_version` 归档；`pipeline-result.json`（`processor_runtime.py:2038-2169`）只有步骤级状态、无候选级度量，跨运行比对需回读逐 run 的 stage JSON。 |
| **D3 算法行业标准符合度** | **8.5 / 10** | 与 PixInsight/Siril/GHS 主流实践高度对齐：Asinh 作深空主拉伸是 Siril 官方推荐路径；linked MTF 的 `[shadows, midtones, 1.0]` 三元组与 PixInsight HistogramTransformation 语义一致；「shadows 不裁到 0」严格遵守行业共识；目标背景 P50 落在 0.15-0.22（`models.py:352-355`），略低于 PixInsight STF 默认 0.25，属保守方向的合理偏离；starless 单独拉伸再回星的工作流与 RC-Astro / Russell Croman 的官方 usage notes 一致。扣分点：GHS 参数 `ghs_shadowsclip=-2.1`（`stage6_services.py:2277`）沿用 STF 风格的 sigma 单位而非 GHS 官方的 SP/LP/HP 语义，属自创映射；未使用 GHS 官方推荐的「Colour / 保色模式」而靠事后色度门补救；`autostretch -linked` 只作 preview 参照不作最终输出，比多数业余流程更严谨。 |
| **总评** | **8.0 / 10** | 算法与安全契约达到「可审计的半专业级」，是本项目质量上限所在的阶段。主要风险不在算法本身，而在**代码考古学风险**：新旧两套架构并存且旧架构语法完整、命名更「正统」，任何未做全量 grep 的维护者都会改错地方。 |

**最严重问题（一句话）**：`pipeline/stage6_services.py` 中并存两套完整的拉伸候选架构——新的固定双候选（`_stage7_compact_stretch_candidates`，`:2246`）在跑，旧的多模式策略（`_stage6_strategy_from_features` `:541` / `_stage6_candidate_specs` `:738` / `stretch_candidate_evaluator.py` 全模块 / `policy_selector.py:45-232` 与 `configs/policies/*.yaml` 的 `scoring`+`hard_reject` 权重）语法完整、命名更正统、却**全部未被运行时调用**，构成高危误导面。

---

## 二、D1 — 逻辑专业合理性

### 2.1 执行链路总览（含实际行号）

```
seestar_Superimpose.py:2519  stage6_star_separation()          ← 去星在前
seestar_Superimpose.py:2520  stage7_stretching()
  └─ stages/stage6_stretching.py:61  run_stage7_stretching()
       ├─ 分离 REJECTED/TOOL_FAILED → :8 _run_with_stars_review_stretch()  ← 仅 autostretch -linked，不进任何硬门
       └─ :97-106  _run_stage6_ai_stretching(allow_ai)  → stage6_services.py:2646
              :2756  allow_ai 判定（网络门 + ai_post_enabled + ai_stage6_enabled）
              :2783  preview_ref = autostretch -linked    ← reference_only
              :2246  _stage7_compact_stretch_candidates()  ← 固定 cand_a / cand_b
              逐候选 :1187 _stage7_evaluate_stretch_candidate()
                  ├─ :1522 _validate_stage6_stretch_quality()   硬门 ×4
                  ├─ :1448 可见度门 ｜ :1581 色度/斑驳门 ×5 ｜ :1265 秩漂移门
                  └─ :2027 _stage6_stretch_risk_score()         软风险分
              :1062 亮度闭环 retry(max=1) ｜ :2607 quantile 兜底 ｜ :2969 chroma_rescue(0.10/0.20/0.35)
              :3230  deterministic_best = min(accepted, key=selection_key)
              :3237  allow_ai → _request_stage7_stretch_selection()
              :3255  AI 选中且可复验 → 采纳；:3258 否则 warn + 退回 deterministic
              :3360 写 stage7_stretch_quality.json ｜ :3384 写 stretch_candidates_report.json
       └─ save stage7_stretched.fit / 置 _stage7_stretch_accepted
seestar_Superimpose.py:2530  pre_starless_compatibility_gate → 硬编码记为 skipped
```

### 2.2 拉伸算法类型与选型

**cand_a — Asinh**（`stage6_services.py:2273`，默认 `{asinh_stretch: 2.2, asinh_offset: 0.002}`）；若目标画像为亮星云 HDR，切换为 `bright_nebula_hdr_masked`（`:2085-2243` 的 profile 映射 + `:960-1059` 的执行分派）。

**cand_b — 两种形态**：
1. `asinh_ghs`（`:2274-2279`，`{asinh_stretch: 2.1, asinh_offset: 0.002, ghs_shadowsclip: -2.1, ghs_stretchamount: 1.05}`）
2. `linked_mtf`（`:2537-2544`），在 starless 台基极窄时启用，参数 `[shadows, midtones, 1.0]`

**专业性评价**：Asinh 作为深空主拉伸是正确选择——它对暗部近似线性放大、对亮部对数压缩，且是**保色**变换（三通道用同一 luminance-driven 因子缩放），这正是深空图既要提暗部又要保星色的核心诉求。GHS 作为二次微调、MTF 作为窄台基展开，分工合理。

**扣分**：`cand_b` 的两种形态由数据条件自动切换（`:2440-2586` 的 try 块），但两者的参数语义完全不同（GHS 的 sigma 风格 vs MTF 的归一化三元组），归档时都写在 `params` 字段下靠 `method` 区分，人工审阅时容易误读。

### 2.3 候选生成策略：固定双候选 + 目标感知 profile

架构是「固定 2 个候选 + preview 参照」，而非「多方案穷举打分」。这在自动化 pipeline 中是**正确的收敛性设计**：候选越多，AI 或排序器选到边缘方案的概率越大，而深空后期没有客观真值可裁决。

目标感知层 `_stage7_target_stretch_profile()`（`:2085-2243`）按 `target_type` 输出 5 类 profile：
- `bright_core_protect`（亮核心保护）
- `star_colour_preserve`（星色保持）
- `galaxy_core_halo_balance`（星系核-晕平衡）
- `dark_nebula_separation`（暗云分离）
- `widefield_nebulosity`（广域星云）

profile 通过 `cand_a_stretch_multiplier` / `cand_b_stretch_multiplier` / `cand_b_ghs_amount` / `highlight_scale` / `cand_a_p50_multiplier` 调制参数（应用点 `:2336-2362`），且乘完后立刻用 `_clamp_float(..., STAGE7_ASINH_STRETCH_MIN, STAGE7_ASINH_STRETCH_MAX)` 钳制（`:2337-2354`）——乘子不能越界，设计正确。

**未验证**：`STAGE7_ASINH_STRETCH_MIN/MAX` 的具体数值未逐行核对（定义在 `stage6_services.py` 模块常量区），仅确认钳制调用存在。

### 2.4 黑场点：是否压死暗部 —— 本阶段最强项

这是整份审计中发现的**最专业的一段代码**。三条独立机制共同保证黑场不裁：

**机制一：MTF shadows 从实测噪声地板反推并留余量**（`:2504-2529`）
```python
shadow_margin = max(noise_sigma * bg_std,            # 默认 sigma=3.0，钳制 [2.0, 6.0]
                    (source_p50 - p01) * 0.25,
                    1e-6)
shadow_candidates = [v - shadow_margin for v in (p01, min_value) if v > 0.0]
shadows = max(0.0, min(shadow_candidates)) if shadow_candidates else 0.0
shadows = min(shadows, max(0.0, source_p50 - shadow_margin))
```
shadows 被取在 **`min(p01, min) 再减去 3σ 余量`**，且被 `max(0.0, ...)` 兜底、被 `source_p50 - margin` 上限二次钳制。这意味着黑场永远在实测最暗像素之下 3σ，**数学上不可能裁掉真实信号**。这比 PixInsight STF 默认的 `-2.80 sigma` clipping 更保守（后者本质上是允许约 0.26% 的像素被裁到 0）。

**机制二：极低背景时 Asinh offset 压到背景地板之下**（`:2282-2315`）
```python
if 0.0 < bg_median <= 0.005:
    safe_offset_candidates = [bg_median * 0.50]
    if p01 > 1e-6: safe_offset_candidates.append(p01 * 0.80)
    noise_floor = bg_median - 2.5 * bg_std
    if noise_floor > 1e-6: safe_offset_candidates.append(noise_floor)
    safe_offset = max(1e-6, min(safe_offset_candidates))
```
取三个候选的**最小值**（最保守），并记录 `offset_cap.reason = "keep Asinh offset below the measured background floor"`（`:2311`）。

**机制三：黑像素比例硬门**（`:1538-1541`）
```python
if metrics.black_pixel_ratio > self.cfg.stage6_black_pixel_ratio_max:   # 0.35
```
以及背景中值下限硬门（`:1533`，`stage6_effective_bg_median_min(cfg.stage6_bg_median_min)`，配置值 0.020 减去容差）。

**与项目自身规范的一致性**：`pipeline/AGENTS.md:5` 明确写「禁止默认引入明显破坏真实性的行为：大面积纯黑背景、星色失真、饱和溢出、弱信号被抹除」。上述三条机制**逐条对应**该规范的四项禁止内容，实现与规范一致。

**判定：不压死暗部，且防护等级高于行业平均。**

### 2.5 高光 / 星系核心过曝防护

四层防护：

| 层 | 位置 | 机制 |
|---|---|---|
| 参数预标定 | `:2416` / `:2425` | `highlight_scale`（默认 0.90）参与 `_stage7_preview_calibrated_stretch`，使预测 P99 留出 10% 顶部余量 |
| offset 上限 | `:2364-2388` | `offset_cap = min(p99*0.85, max*0.80)`，理由 `"asinh_offset must stay below starless effective signal range"` |
| 全局硬门 | `:1543-1546` | `highlight_clip_ratio > 0.010` → reject |
| 局部核心门 | `models.py:368` | `stage7_local_core_clip_ratio_max = 0.12`，只看目标核心 ROI |

**评价**：全局 1.0% 裁剪门对深空图是合理的宽松上限（星点本身就该接近饱和）；真正保护星系核心的是局部 ROI 门 0.12——这是相当专业的做法，因为全局比例会被大量背景像素稀释，无法反映核心是否烧掉。

**扣分**：`stage7_local_core_clip_ratio_max=0.12` 意味着允许核心区 12% 像素裁剪，对 M31/M42 这类高动态目标偏宽松；且 `bright_nebula_star_growth_ratio_max` 放宽到 1.50（`models.py:343`）时，核心门并未同步收紧。

### 2.6 starless 图专门拉伸的利用

Stage 7 输入是 Stage 6 的 starless 结果，代码有三处针对 starless 特性的专门处理：

1. **窄台基识别与 linked MTF 分支**（`:2440-2586`）：starless 图去掉星点后直方图动态范围骤降、台基极窄，Asinh 在此类分布上容易过度压缩。代码检测到该情形后切换为 noise-floor linked MTF，理由字符串 `"expand the narrow Starless pedestal from a measured noise-floor shadow"`（`:2574-2578`）——这是对 starless 数据分布特性的正确认知。

2. **starless 结构秩漂移门**（`:1264-1290`、`:1406-1420`，`stage7_stretch_metrics.assess_starless_structure_growth`）：用原星点邻域掩膜内的亮度秩结构 P95 漂移（阈值 0.18，`models.py:363`）替代通用星点膨胀检测。`models.py:362` 的注释说明了原因——「避免通用阈值把星云纹理误判为星点膨胀」。这是很到位的设计：starless 图上本就没有星点，用 `median_star_size` 做增长比检测会失真。

3. **星点增长门的条件启用**（`:1507-1520` `_stage7_should_enforce_star_growth`）：当 starless 结构门可用（status ∈ {ok, rejected}）时，通用星点增长门被让位。

**判定：对 starless 特性的利用充分且有针对性，明显优于「把 starless 图当普通线性图拉」的常见做法。**

### 2.7 色度保持

`_stage7_stretch_background_gate()`（`:1581`）实现 5 条色度相关门：

| 指标 | 阈值 | 来源 |
|---|---|---|
| `chroma_noise_score` | 0.34 | `models.py:371` |
| `background_mottling_score` | 0.45 | `models.py:372` |
| `chroma_load_growth` | 1.35 | `models.py:373` |
| `chroma_load_low_absolute` | 0.05 | `models.py:374` |
| `chroma_load_signal_excluded` | 0.06 | `models.py:360` |

**设计亮点**：`chroma_load_growth` 是**增长比**而非绝对值——它衡量拉伸前后色度负载的放大倍数，正确捕捉了「拉伸放大色噪」这一物理机制，而非惩罚数据本身固有的颜色。同时保留 `chroma_load_low_absolute`（低背景绝对上限）作为增长比在极低基数时失真的补丁（`:1776-1792` 有 `chroma_load_growth_low_absolute_exempted` 豁免逻辑）。

**扣分**：Asinh 本身是保色变换，但 `asinh_ghs` 组合中的 GHS 环节若未启用保色模式则会移动色相。代码未见任何 GHS 保色模式开关（`:960-1059` 的 `autoghs -linked` 分派只传 `-linked`），依赖事后色度门被动拦截而非主动保色。**未验证**：Siril `autoghs` 是否在 `-linked` 下自动等价于保色处理——需实测确认。

### 2.8 色度救援候选不暴露给 AI 的安全性

**机制**：色度救援候选（`method = "background_chroma_rescue"`，`:2012`）在 `_stage7_validated_fallback_reason()`（`:1862-1869`）被标记为 `"validated_chroma_rescue"`，进而在 attempt 上置 `explicit_fallback=True`。AI 白名单构造时显式排除：

```python
# stage6_services.py:3334-3338
"allowed_candidate_ids": [
    str(attempt.get("name"))
    for attempt in accepted_attempts
    if not bool(attempt.get("explicit_fallback"))
],
```

同样的过滤在 AI 选择复验时**二次执行**（`:3245-3254`，条件含 `and not bool(attempt.get("explicit_fallback"))`），以及在 `ai_advisory.request_stage7_stretch_selection()`（`ai_advisory.py:870-930`）内部第三次执行。

**评价：三重过滤，安全性判定为「充分」。** 救援候选只能通过确定性排序在无其他 accepted 候选时被选中，AI 无法主动索取。

**触发前置条件也很严**：`_stage7_attempt_allows_chroma_rescue()`（`:1946-1972`）要求候选**仅因**背景色噪被拒（其它硬门必须全过），才允许从冻结线性源重放。这避免了用色度抑制掩盖真正的拉伸失败。

### 2.9 AI 只选 ID 不给数值的安全性

契约由三处共同保证：

1. **schema 强制**（`ai_advisory.py:870-930`）：`model_must_not_return_parameters: True`、`all_candidates_passed_hard_gates: True`
2. **归一化验证**（`ai_advisory.py:842-867` `normalize_stage7_stretch_selection`）：`selection_contract: candidate_id_only_after_hard_gates`，校验返回的 `candidate_id` 必须在 allowed 列表内
3. **调用侧复验**（`stage6_services.py:3245-3254`）：即使归一化通过，本地仍从 `accepted_attempts` 里再查一次，查不到就丢弃

归档字段明确记录了所有权：`"model_output_fields": ["selected_candidate_id"]`、`"parameters_owned_by": "code"`（`:3332-3333`）。

**评价：这是 AI 参与后期决策的正确范式。** AI 只在「代码已证明安全的选项集合」上做主观偏好选择，无法引入代码未验证过的参数组合。相较于让 AI 直接返回 `asinh_stretch=4.7` 这类做法，风险面缩小了一个数量级。

**扣分**：AI 是否被采纳只记录布尔（`"ai_selection": ai_selection`，`:3339`）与 selector 字符串（`:3327-3331`），**AI 未被采纳时的具体原因只落在日志 warn**（`:3258-3261`），未结构化写入 JSON。见 D2 §3.7 缺口 G3。

### 2.10 无效选择回退本地排序的确定性

`_stage7_candidate_selection_key()`（`:1747-1858`）是纯 `@staticmethod`，输入只有 attempt dict，输出 9 元组：

```
(status_penalty,          # 0/1  status=="ok" 且有 stem
 final_penalty,           # 0/1  allowed_as_final
 hard_issue_count,        # 硬诊断条数（排除两类软前缀）
 normalized_excess,       # Σ max(0, metric/limit − 1) + 亮度区间越界惩罚
 len(diagnostics),        # 全部诊断条数
 normalized_quality_load, # Σ metric/limit
 brightness_distance,     # |ln(attainment_ratio)|
 risk_score,              # _stage6_stretch_risk_score 输出
 name)                    # 字符串，最终 tie-break
```

**确定性判定：完全确定性。** 无随机数、无时间依赖、无字典遍历顺序依赖（最后一位用 `name` 字符串保证全序），且 `min()` 在 Python 中对相等键取首个——但因 `name` 唯一，不存在相等键。

**软/硬分层正确**：`soft_prefixes = ("background_chroma_noise_score", "background_chroma_load_growth")`（`:1756-1759`）——这两类色度诊断不计入 `hard_issue_count`，只影响后位的 `normalized_quality_load`。设计意图是「色度问题可救援，结构问题不可」，与 §2.8 的救援触发条件一致。

**兜底完整**：`risk_score` 解析失败或非有限值时置 `1_000_000.0`（`:1842-1847`）；`brightness_distance` 在 ratio ≤ 0 时置 `math.inf`（`:1836-1840`）——异常候选被自动排到末尾而非抛异常。

### 2.11 Stage 7 兼容检查点默认跳过的隐患

`pre_starless_compatibility_gate()` 实现在 `seestar_Superimpose.py:2005-2087`，但主流程中**无条件**记为 skipped：

```python
# seestar_Superimpose.py:2530-2534
self._record_skipped_stage(
    PipelineCheckpoint.PRE_STARLESS_COMPATIBILITY_GATE.label,
    "not a formal stage; starless-first mode completes Stage 6 "
    "before Stage 7",
)
```

注意这段代码在 `if input_profile.safe_for_linear_steps:` **之外**（`:2518` 的分支在 `:2529` 已闭合），即无论何种路径都记 skipped。

`workflow.md` §5.7.1（L578-601）对此有明确说明：该检查点是「非正规阶段」，旧路径仍计算诊断但不生成历史 Asinh 输入；输出 `stage7_5_pre_starless_gate_report.json`；评估模块不可用时回退 `ready_for_starless=true`。实现侧 `:2042-2045` 确实把 Asinh 非线性推荐强制回退为当前线性源。

**隐患评估（中等，非高危）**：
- 在 starless-first 架构下，去星（Stage 6）已在 Stage 7 之前完成，该检查点的原始职责（「去星前是否需要先拉伸」）确实已不适用，跳过在**语义上正确**。
- 但它同时携带的**去星质量诊断**能力一并失效。当前架构下，Stage 6 的分离质量判断完全由 `run_stage7_stretching()` 开头的分离状态检查承担（`stage6_stretching.py:61-96`），而该检查只区分 REJECTED/TOOL_FAILED 两态，粒度远低于原检查点。
- 更实质的隐患在 `_run_with_stars_review_stretch()`（`stage6_stretching.py:8`）：去星失败时该路径**只做 `autostretch -linked` 生成复核预览**，Stage 7 的全部硬门（黑场、高光、色度、结构）**一条都不执行**。虽然此路径输出被标记为 review-only，但它仍会产生一张非线性图进入后续流程。

### 2.12 Legacy 死代码与死配置（本阶段头号技术债）

grep 全量确认以下代码/配置**存在完整实现但无运行时调用点**：

| 死组件 | 位置 | 状态 |
|---|---|---|
| `_stage6_stretch_candidate()` | `stage6_services.py:501-540` | 仅被 `_stage6_strategy_from_features` 调用 |
| `_stage6_strategy_from_features()` | `stage6_services.py:541-694` | 无调用点（构建 asinh/ghs/autostretch 三组旧策略） |
| `_stage6_candidate_specs()` | `stage6_services.py:738-779` | 无调用点 |
| `_stage6_candidate_label()` | `stage6_services.py:782-808` | 仅被上述死函数调用 |
| `Stage6StretchStrategy` | 类型定义 | 仅出现在死函数签名与 import |
| `stretch_candidate_evaluator.py` | 整个模块（`build_candidate_spec` `:56` / `score_candidate` `:134` / `allowed_as_final` `:180` / `choose_best` `:213`） | 被 `stage6_services.py:49-65` import 但未调用；仅 `tests/test_adaptive_pipeline_phase1.py` 引用 |
| `policy_selector.py` 的 `stage6_stretch` policy | `:45-232`（`candidate_mode` / `forbidden_when_dirty` / `hard_reject.max_bg_dirty_score=0.42` / `scoring.core_blowout_weight` / `fallback_candidate`） | `forbidden_when_dirty` 仅被死模块 `stretch_candidate_evaluator.py` 消费 |
| `configs/policies/*.yaml` 的 `stage6_stretch.scoring` / `.hard_reject` | 多个 YAML（如 `bright_nebula_hdr_conservative.yaml`） | 与 `PipelineConfig` 硬编码阈值并存，新架构不消费 |

**危害等级：高。** 原因不是运行风险（死代码不执行），而是：
1. 旧命名（`_stage6_*`、`stage6_stretch` policy）比新命名（`_stage7_*`）更贴合文件名 `stage6_stretching.py` / `stage6_services.py`，误导性极强；
2. policy YAML 里的 `scoring` 权重看起来是「排序权重来源」，实际排序权重全部硬编码在 `_stage7_candidate_selection_key` 里——审计 D2「排序权重硬编码还是配置」这个问题，如果只看 YAML 会得出完全错误的结论；
3. `stretch_candidate_evaluator.py` 有测试覆盖，会给人「这是活代码」的错觉。

### 2.13 D1 小结

- ✅ 拉伸算法类型合理（Asinh 主 + GHS/MTF 副，分工正确）
- ✅ 候选生成策略收敛（固定双候选 + 目标 profile，参数全钳制）
- ✅✅ 黑场不压死暗部（三重机制，从构造上不可能裁剪）
- ✅ 高光/核心过曝防护（四层：预标定/offset cap/全局门/局部 ROI 门；局部门阈值偏宽）
- ✅✅ starless 特性利用（窄台基 MTF 分支 + 秩漂移门替代星点增长门）
- ⚠️ 色度保持（门禁完备但被动；GHS 保色模式未主动启用）
- ✅✅ 救援候选对 AI 隐藏（三重过滤）｜✅✅ AI 只选 ID（schema + 归一化 + 本地复验）｜✅✅ 无效选择回退确定（纯函数 9 元组全序）
- ⚠️ 兼容检查点跳过（语义正确但诊断能力丢失；review 路径无硬门）
- ❌ 死代码（两套架构并存，高误导风险）

---

## 三、D2 — 门禁来源与归档

### 3.1 硬门（hard gate）穷举

| # | 门名 | 判定位置 | 阈值 | 默认值 | 来源分类 | env 覆盖 | 钳制 |
|---|---|---|---|---|---|---|---|
| H1 | 背景中值下限 | `stage6_services.py:1533-1536` | `stage6_effective_bg_median_min(cfg.stage6_bg_median_min)` | 0.020（`models.py:339`）经 `:377` 减容差 | hardcoded → 运行时推导 | ❌ **无** | 由 `:377` 函数内部实现 |
| H2 | 黑像素比例上限 | `:1538-1541` | `cfg.stage6_black_pixel_ratio_max` | 0.35（`models.py:340`） | hardcoded | ❌ **无** | 风险分处 `max(..., 1e-4)`（`:2043`） |
| H3 | 高光裁剪上限 | `:1543-1546` | `cfg.stage6_highlight_clip_ratio_max` | 0.010（`models.py:341`） | hardcoded | ❌ **无** | 风险分处 `max(..., 1e-5)`（`:2046`） |
| H4 | 星点增长上限 | `:1555-1559` | `_stage7_effective_star_growth_ratio_max()`（`:1485`） | 1.25 / 亮星云 1.50（`models.py:342-343`） | hardcoded + 运行时分支 | ❌ **无** | 无显式钳制 |
| H5 | starless 秩漂移 P95 | `:1406-1420` | `stage7_starless_masked_rank_drift_p95_max` | 0.18（`models.py:363`） | hardcoded | ✅ `SEESTAR_STAGE7_STARLESS_MASKED_RANK_DRIFT_P95_MAX`（`processor_runtime.py:1063`） | 未见 |
| H6 | 弥散可见度下限 | `:1448-1482` | `stage7_diffuse_visibility_score_min` | 0.08（`models.py:359`） | hardcoded | ❌ 未在白名单 | 未见 |
| H7 | 色度噪声上限 | `:1581+` | `stage7_stretch_chroma_noise_score_max` | 0.34（`models.py:371`） | hardcoded | ✅ `SEESTAR_STAGE7_STRETCH_CHROMA_NOISE_SCORE_MAX`（`:178`） | 未见 |
| H8 | 背景斑驳上限 | `:1581+` | `stage7_stretch_background_mottling_score_max` | 0.45（`models.py:372`） | hardcoded | ✅（`:179`） | 未见 |
| H9 | 色度负载增长上限 | `:1581+` | `stage7_stretch_chroma_load_growth_max` | 1.35（`models.py:373`） | hardcoded | ✅（`:180`） | 未见 |
| H10 | 色度负载低基绝对上限 | `:1581+` | `stage7_stretch_chroma_load_low_absolute_max` | 0.05（`models.py:374`） | hardcoded | ✅（`:181`） | 有容差键（`:182`） |
| H11 | 信号排除色度负载上限 | `:1581+` | `stage7_stretch_chroma_load_signal_excluded_max` | 0.06（`models.py:360`） | hardcoded | ❌ 未在白名单 | 未见 |
| H12 | 局部核心裁剪上限 | 局部度量模块 | `stage7_local_core_clip_ratio_max` | 0.12（`models.py:368`） | hardcoded | ✅ `SEESTAR_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX`（`:230`） | 未见 |
| H13 | 局部弱信号 SNR 下限 | 同上 | `stage7_local_faint_snr_min` | 0.25（`models.py:369`） | hardcoded | ✅（`:231`） | 未见 |
| H14 | 局部暗云分离下限 | 同上 | `stage7_local_dark_separation_min` | 0.001（`models.py:370`） | hardcoded | ✅（`:232`） | 未见 |
| H15 | preview P50 达标区间 | `preview_target_attainment` | `stage7_preview_target_p50_min_ratio` / `_max_ratio` | 0.55 / 1.50（`models.py:357-358`） | hardcoded | ✅（`:184-185`） | 未见 |

**统计**：15 条硬门中，**9 条可 env 覆盖、6 条不可**；不可覆盖的 6 条里包含了 H1-H4 这四条**最核心的图像安全门**。

### 3.2 软排序权重（soft ranking）

排序权重**全部硬编码在两个函数体内**，无任何配置化：

**`_stage7_candidate_selection_key()`（`:1747-1858`）** — 字典序 9 元组，权重体现为「位次」而非数值系数：

| 位次 | 分量 | 计算 | 硬编码位置 |
|---|---|---|---|
| 1 | status_penalty | `status=="ok" and stem` → 0 else 1 | `:1749` |
| 2 | final_penalty | `allowed_as_final` → 0 else 1 | `:1750` |
| 3 | hard_issue_count | 非软前缀诊断计数 | `:1756-1762` |
| 4 | normalized_excess | Σ `max(0, m/l − 1)` + 亮度越界罚 | `:1802-1835` |
| 5 | len(diagnostics) | 全部诊断计数 | `:1853` |
| 6 | normalized_quality_load | Σ `m/l` | `:1808` |
| 7 | brightness_distance | `\|ln(attainment_ratio)\|` | `:1836-1840` |
| 8 | risk_score | 见下 | `:1842-1847` |
| 9 | name | 字符串 tie-break | `:1857` |

**`_stage6_stretch_risk_score()`（`:2027-2059`）** — 加权和，系数全为字面量：

| 项 | 系数 | 行号 |
|---|---|---|
| `len(issues)` | ×5.0 | `:2037` |
| 背景中值不足 | ×8.0（相对偏差） | `:2040` |
| 背景中值 < 0.005 | 额外 ×12.0 | `:2042` |
| 黑像素超标 | ×3.0 | `:2045` |
| 高光裁剪超标 | ×6.0 | `:2048` |
| 星点增长超标 | ×8.0（绝对超出） | `:2058` |
| metrics 为 None | `1_000_000.0 + len(issues)` | `:2036` |

**判定：排序权重 100% 硬编码，零配置化，零 env 覆盖。** 注意 `configs/policies/*.yaml` 和 `policy_selector.py:45-232` 里确实存在 `stage6_stretch.scoring.core_blowout_weight` 之类的权重定义——但如 §2.12 所述，**这些是死配置，不参与实际排序**。

### 3.3 阈值来源分类汇总

| 分类 | 数量 | 说明 |
|---|---|---|
| **hardcoded（PipelineConfig 字面量默认值）** | 约 30 项 | `models.py:339-408` 集中定义；全部可经 PipelineConfig 传参，但 GUI/task intake 是否暴露 → **未验证** |
| **env 覆盖** | 约 25 键 | `processor_runtime.py:170-232` 白名单 + `:1036-1172` 的 float/int/bool 三类 setattr |
| **运行时推导** | H1（`stage6_effective_bg_median_min`）、cand_a/b 全部参数（背景自适应 + preview 标定 + target profile 三级推导） | `stage6_services.py:377`、`:2265-2604` |
| **AI 返回** | 仅 `selected_candidate_id`（1 个字符串） | 契约 `parameters_owned_by: "code"`（`:3333`） |

**关键结构性问题**：env 白名单覆盖面与阈值重要性**负相关**。次要的 `SEESTAR_STAGE7_STARLESS_HALO_DETAIL_DELTA_MIN`（`:190`）可覆盖，而决定图像是否被判定为「暗部压死」的 `stage6_bg_median_min` 不可覆盖。grep 确认：

```
grep "stage6_bg_median_min|stage6_black_pixel_ratio_max|
      stage6_highlight_clip_ratio_max|stage6_star_growth_ratio_max|
      stage7_preview_cand_a_p50_ratio|stage7_chroma_rescue_strength_levels"
  pipeline/processor_runtime.py  →  0 matches
```

### 3.4 归档矩阵

| 内容 | `stage7_stretch_quality.json` | `stretch_candidates_report.json` | `ai_raw_*.json` | `processing-plan.json` | `pipeline-result.json` | 日志 |
|---|---|---|---|---|---|---|
| preview_ref 参照 + 像素统计 | ✅ `:3365-3372` | ✅ 文件名 `:3415` | — | — | — | — |
| baseline 自适应/背景质量/冻结采样/像素统计 | ✅ `:3373-3376` | ✅ `:3390-3392` | — | — | — | — |
| 拉伸自适应决策（含 offset_cap 理由） | ✅ `:3377` | ✅ `:3389` | — | — | — | — |
| **每候选完整 attempt 对象** | ✅ `:3378` | — | — | — | — | — |
| 每候选 method / params / adaptation | — | ✅ `:3396-3399` | — | — | — | — |
| 每候选 quality_ok / risk_score / status / pixel_stats | — | ✅ `:3400-3406` | — | — | — | 选中项 `:3465-3470` |
| 每候选 preview_target_attainment / target_local_quality | — | ✅ `:3403-3404` | — | — | — | — |
| 每候选 background_quality_gate（metrics+limits） | — | ✅ `:3405` | — | — | — | — |
| 每候选 diagnostics（逐项失败原因）/ feedback | — | ✅ `:3407-3408` | — | — | — | — |
| 每候选 selection_rank / selection_role | ✅（attempt 内 `:3288-3295`） | ✅ `:3410-3411` | — | — | — | — |
| starless_structure_quality | ✅（attempt 内 `:1441`） | ❌ 不在字段白名单 | — | — | — | — |
| selection_summary（策略/选择器/白名单/结果） | ✅ `:3381` | ✅ `:3437` | — | — | — | — |
| AI 是否被采纳 + 允许候选 ID 列表 | ✅ `:3334-3339` | ✅ 同 | — | — | — | ✅ `:3299-3309` |
| AI 未采纳原因 | ❌ | ❌ | ❌ | — | — | ⚠️ 仅 warn `:3258-3261` |
| AI endpoint / temperature / json_mode / content / response | — | — | ✅ `save_utils.py:112-150` | — | — | — |
| **AI prompt 版本 / schema 版本 / model id** | ❌ | ❌ | ⚠️ 仅 endpoint | ❌ | ❌ | ❌ |
| 选中候选 + fallback_used + reason_code | ✅ `:3379`/`:3351` | ✅ `:3416-3427` | — | — | ✅ 步骤级 | ✅ `:3457-3462` |
| stage7_stretch 候选契约 + 完整 cfg 快照（红隐后） | — | — | — | ✅ `processor_runtime.py:1848-1990` | — | — |
| 阶段 status / fallback_used / reason_code | — | — | — | — | ✅ `:2038-2169` | ✅ |
| **候选级度量向量** | ✅ | ✅ | — | ❌ | ❌ | ❌ |

### 3.5 AI 决策归档细节

`request_stage_ai_advisory()`（`ai_advisory.py:464-624`）在每次调用后写 `ai_raw_{counter}_{stage}.json` + 同名 `.txt`，字段含：`endpoint`、`temperature`、`json_mode`、`error`、`content_preview`、`content`、`response`（`save_utils.py:112-150`）。

温度策略：kimi 系模型固定 1.0，其余尝试 0.1 后 1.0（`ai_advisory.py:464-624` 内的 temperatures 序列）。

**缺失项**：
- **prompt 版本标识**：grep `prompt_version|prompt_id|schema_version` 在 `pipeline/` 下**无任何匹配**。system_prompt 与 user_prompt 是运行时拼接的字符串（含 `DEFAULT_AI_PROMPT`，`ai_advisory.py:20`），一旦被修改，历史归档无法区分「同一 endpoint 下 prompt 已变更」。
- **模型标识**：只有 `endpoint` URL，无明确的 model name/version 字段。同一 endpoint 后端换模型不可察觉。
- **未采纳原因结构化**：`:3258-3261` 的 warn 文案 `"stage7 stretch selection was not revalidated"` 只进日志。

### 3.6 可复现性判定：**可复现（有条件）**

**判定：仅凭 `stretch_candidates_report.json` + `stage7_stretch_quality.json` 即可脱机复现「为何最终选了候选 A」——成立。**

复现路径：
1. 从 `candidates[]` 取出每个候选的 `status`、`quality_ok`、`diagnostics`、`background_quality_gate.{metrics,limits}`、`preview_target_attainment`、`risk_score`
2. 这恰好是 `_stage7_candidate_selection_key()`（`:1747`）的**全部输入**——该函数是纯静态方法，不读 cfg、不读文件
3. 手工重算 9 元组即可复现 `selection_rank`（该字段本身也已落盘 `:3410`，可交叉校验）
4. `selection.selector` 字段（`:3327-3331`）直接告知是 `ai_candidate_id` 还是 `deterministic_quality_rank`
5. `selection.allowed_candidate_ids`（`:3334`）告知 AI 当时可见的白名单

**限制条件（使其为「有条件」）**：
- 若 `selector == "ai_candidate_id"`，**AI 为何在白名单中选 A 而非 B 不可复现**——AI 原始 response 在 `ai_raw_*.json` 里，但需人工关联两个文件（无交叉 ID）；且无 prompt 版本，无法确认当时的提示词。
- `attempt.allowed_as_final` 的判定逻辑本身未落盘（只落布尔结果），若要复核该布尔是否正确，仍需读代码。
- 硬门阈值本身**不在这两个 JSON 里**——`background_quality_gate.limits` 只覆盖色度门（H7-H11），H1-H4 的阈值需回读 `processing-plan.json` 的 cfg 快照。跨文件关联才能完整复现。

### 3.7 归档缺口清单

| ID | 缺口 | 影响 | 严重度 |
|---|---|---|---|
| G1 | H1-H4 四条核心硬门无 env 覆盖 | 现场调参必须改代码；A/B 对照实验无法通过环境变量做 | 高 |
| G2 | AI 请求无 `prompt_version` / `schema_version` / `model_id` | prompt 漂移不可追溯；跨 run 对比 AI 行为时无法排除提示词变更因素 | 高 |
| G3 | AI 选择被拒的原因未结构化落盘（只 warn） | 归档中 `ai_selection=null` 时无法区分「AI 未调用」「AI 返回空」「AI 返回无效 ID」 | 中 |
| G4 | `pipeline-result.json` 无候选级度量 | 批量跨 run 分析必须逐 run 打开 stage JSON | 中 |
| G5 | H1-H4 阈值不写入 Stage 7 自身 JSON | 单文件不自洽，复现需关联 `processing-plan.json` | 中 |
| G6 | `starless_structure_quality` 未进入 `stretch_candidates_report.json` 字段白名单（`:3394-3412`） | 该报告是给人看的主报告，秩漂移门结果缺席 | 中 |
| G7 | 两个 stage JSON 与 `ai_raw_*.json` 无交叉引用 ID | 需靠 counter 与时间顺序人工对齐 | 低 |
| G8 | policy YAML 的 `stage6_stretch.scoring`/`hard_reject` 是死配置但仍进 cfg 快照 | 审计者会误认为这些权重生效 | 中 |

---

## 四、D3 — 算法行业标准符合度

### 4.1 对标基准

本节全部结论基于 13 个一手来源（PixInsight 官方文档与 PCL 源码、Siril 官方文档与命令参考、GHS 官网与工具文档、GHS 论坛大黑场讨论、SetiAstro GHS 数学实现、RC-Astro StarXTerminator usage notes、PixInsight 论坛 Russell Croman 回星讨论、StargazersLounge 背景电平共识、Lodriguss《BGAIP》）。**完整 URL 见 §7.2**，下文逐项对标时按需引用。

### 4.2 逐项对标

#### (1) Asinh 作为深空主拉伸 — **符合**

Siril 官方拉伸文档明确把 Asinh 列为深空图的主要拉伸方式，并强调其相对 MTF 的优势是**保持色彩**（因为三通道用同一个基于亮度的因子缩放）。本项目把 Asinh 定为 cand_a 主候选（`stage6_services.py:2273`），完全符合。

Siril `asinh` 命令的 stretch 参数范围为 1-1000（Commands 文档），本项目实际使用 2.1-2.2 起步、经 preview 标定后由 `_clamp_float(..., STAGE7_ASINH_STRETCH_MIN, STAGE7_ASINH_STRETCH_MAX)` 钳制、`stage7_preview_asinh_stretch_max` 默认 1000.0（`:2406-2410`）——在官方范围内。

#### (2) linked MTF 的三元组语义 — **符合**

PixInsight HistogramTransformation 的 MTF 由 `(shadows, midtones, highlights)` 定义，midtones=0.5 时为恒等（线性）。本项目 cand_b 的 linked MTF 参数 `{mtf_shadows, mtf_midtones, mtf_highlights: 1.0}`（`:2538-2544`）与该语义一致，highlights 固定 1.0（不压缩顶部）也是主流做法。

`_stage7_linked_mtf_midtones(normalized_source_p50, target_p50)`（`:2533`）从「源 P50 → 目标 P50」反解 midtones，这正是 PixInsight STF AutoStretch 的核心算法（给定 target background 反解 m）。**判定为标准实现的正确复刻。**

#### (3) 目标背景值 0.15-0.22 vs 行业 0.25 — **部分符合（保守偏离，合理）**

PixInsight STF AutoStretch 的 Target background 默认 **0.25**（STF 文档 + PCL 源码常量）；Siril `autostretch` 命令的 `targetbg` 默认同为 **0.25**（Siril Commands 文档）。

本项目的 starless linked MTF 目标 P50 区间为 **0.15-0.22**（`models.py:352-355`：`p50_min=0.15`、`diffuse_p50_min=0.17`、`p50_max=0.22`）。

**评价：这是有意识的保守偏离，方向正确。** 理由：
- 0.25 是 STF 的**屏幕预览**目标，用于让线性图在屏幕上「看得见」，不是最终成品的背景电平；把 0.25 直接当成品背景会明显偏亮。
- Jerry Lodriguss 与社区共识（StargazersLounge 讨论）建议成品背景落在 **0.09-0.12** 附近（8-bit 约 20-30/255）。
- 本项目的 0.15-0.22 介于「STF 预览值」与「成品建议值」之间，考虑到这是 starless 中间产物（后续 Stage 8 增强 + Stage 9 回星还会改变整体观感），选在中间偏低是合理的。

**扣分点**：0.15-0.22 这三个数值在 `models.py` 只有中文注释，无对标依据说明，也无 env 覆盖通道（不在白名单）。

#### (4) 黑场不裁到 0 — **完全符合，且严于行业**

行业共识（PixInsight HistogramTransformation 文档、Lodriguss BGAIP §3.03h、StargazersLounge 讨论）一致强调：**shadows clipping 必须留出余量，绝不能把背景峰裁到 0**，否则暗部信号永久丢失且无法恢复。

PixInsight STF 的默认 `-2.80 sigma`（PCL `DEFAULT_AUTOSTRETCH_SCLIP`）本质上**允许约 0.26% 的最暗像素被裁**——这是屏幕预览可接受、成品不可接受的。Siril `autostretch` 的 `shadowsclip` 默认同为 **-2.8**（Siril Commands 文档）。

本项目做法（`:2515-2529`）：
```
shadows = min(p01, min_value) − max(3σ·bg_std, (p50−p01)·0.25, 1e-6)
shadows = max(0.0, shadows)
shadows = min(shadows, max(0.0, p50 − shadow_margin))
```
即黑场取在**实测最小值之下再减 3σ**，然后 clamp 到 ≥0。**这意味着裁剪像素数在数学上为 0**，比 -2.8σ 严格得多。

同时项目的 `stage7_starless_linked_mtf_shadow_noise_sigma` 默认 3.0、钳制 `[2.0, 6.0]`（`:2506-2514`），即使配置到最小值 2.0 仍在最暗像素之下。

**这是本审计中发现的、与行业最佳实践对齐度最高的一处实现。**

#### (5) GHS 参数语义 — **偏离（自创映射）**

GHS 官方参数体系为 `SP`（symmetry point）、`LP`（local protection，暗部保护）、`HP`（highlight protection）、`D`（stretch amount）、`b`（local intensity）。SetiAstro 的实现说明进一步阐明：GHS 的正确用法是**先减黑点，把 SP 置于背景峰位置，用 LP/HP 生成直线段保护两端**。

Siril `autoghs` 命令的默认值为 **B=13、HP=0.7、LP=0**（Siril Commands 文档）。

本项目 cand_b 的 GHS 参数是 `{ghs_shadowsclip: -2.1, ghs_stretchamount: 1.05}`（`:2277-2278`），且 `PipelineConfig` 默认为 `ghs_shadowsclip=-2.8`、`ghs_stretchamount=2.0`（`models.py:210-211`）。

**问题**：
- `ghs_shadowsclip` 用的是 **STF 风格的 sigma 单位**（-2.1、-2.8），这是 `autostretch`/`autoghs` 的 shadows clipping 参数语义，不是 GHS 论文里的 SP/LP。
- **未见任何 SP / LP / HP 的显式设置**。执行侧 `:960-1059` 的分派只传 `autoghs -linked` 加两个参数，`HP` 与 `LP` 走 Siril 默认（HP=0.7、LP=0）。
- **LP=0 意味着没有暗部局部保护**——GHS 论坛讨论明确指出，处理大黑场数据时应通过 LP 生成暗部直线段来避免暗部被过度压缩。

**判定：部分符合。** 主拉伸幅度受控（`ghs_stretchamount` 从 2.0 降到 1.05，属保守），且 Asinh 已在前级完成主要提升，GHS 只做微调，风险有限。但参数语义映射是自创的，且未利用 GHS 最有价值的 LP/HP 保护机制。

#### (6) GHS 保色（Colour）模式 — **偏离**

GHS 论坛讨论与官方文档均提及 GHS 有专门的 **Colour / 保色模式**，用于在拉伸时保持色相不漂移。本项目在 GHS 环节未启用任何保色开关（`:960-1059` 的命令构造中无相关参数），改为在事后用色度门（H7-H11）被动拦截。

**判定：偏离。** 事后门禁能拦住「色噪放大过头」的候选，但拦不住「色相整体偏移」——后者不体现在 `chroma_noise_score` 或 `chroma_load_growth` 上。

**未验证**：Siril 1.4.0 的 `autoghs` 是否提供保色开关，以及 `-linked` 模式是否已隐含链接三通道从而避免色相漂移。需查 Siril 1.4.0 命令参考确认。

#### (7) starless 单独拉伸再回星工作流 — **符合**

RC-Astro（StarXTerminator 作者 Russell Croman）的官方 usage notes 明确推荐：**在线性阶段分离星点 → 对 starless 与 stars 分别独立拉伸 → 用 Screen 或 PixelMath 合成回去**。PixInsight 论坛的 "After StarXTerminator" 讨论中，Croman 本人进一步说明了 unscreen/re-screen 的数学关系。

本项目的链路完全对应：Stage 6 在线性域分离（`seestar_Superimpose.py:2519`）→ Stage 7 只拉 starless（`:2520`）→ Stage 9 回星。**架构层面完全符合当代主流工作流。**

进一步的加分项：`workflow.md` §5.9（L734-760）记载 Stage 9 使用 `intensity_scale` 降低回星后的二次星点风险。**未验证归属**：grep 显示 `_stage9_star_intensity_scale` 实际由 `pipeline/stage7_quality.py:685/729/775/785` 设置，而非 `stage6_services.py` 的 Stage 7 拉伸主流程——文档描述与实现归属存在偏差，需进一步核对 `stage7_quality.py` 的阶段归属。

#### (8) autostretch 只作参照不作输出 — **符合且优于行业常见做法**

本项目 `preview_ref` 用 `autostretch -linked` 生成，明确标记 `"reference_only": True`（`:3369`），只用于反推 P50/P99 目标（`:2411-2440`），**不会成为最终输出**。

行业上，大量业余自动化脚本直接把 STF/autostretch 结果 apply 成永久拉伸——这是被 PixInsight 社区反复批评的做法（STF 是预览工具，其激进的 -2.8σ clipping 不适合成品）。本项目正确区分了「参照」与「输出」。

### 4.3 行业陷阱清单与规避情况

| # | 行业陷阱 | 规避 | 证据 |
|---|---|---|---|
| T1 | 把 STF/AutoStretch 直接固化为成品拉伸 | ✅ 规避 | `preview_ref` 标 `reference_only:True`（`:3369`），只用于标定 |
| T2 | 黑场裁到 0 导致暗部永久丢失 | ✅✅ 规避（严于行业） | `:2515-2529` shadows 在最小值之下 3σ |
| T3 | 一次性重度拉伸导致星点膨胀 | ✅ 规避 | 星点增长门 1.25/1.50（`:1555`）+ starless 秩漂移门 0.18（`:1406`） |
| T4 | 拉伸后星系核心/亮星云烧白 | ✅ 规避 | 高光门 0.010（`:1543`）+ 局部核心门 0.12 + `highlight_scale=0.90` |
| T5 | 非保色拉伸导致星色漂移 | ⚠️ 部分 | Asinh 保色 ✅；GHS 未启保色模式，靠事后门 |
| T6 | 拉伸放大背景色噪 | ✅ 规避 | `chroma_load_growth` 增长比门 1.35（`:1581+`） |
| T7 | 背景压得过暗（「太空太黑」） | ✅ 规避 | 背景中值下限 H1 + 弥散可见度门 0.08 + preview P50 达标区间 0.55-1.50 |
| T8 | 背景提得过亮（雾感） | ✅ 规避 | preview P50 上限比 1.50 + linked MTF 目标 P50 上限 0.22 |
| T9 | starless 图用普通线性图的拉伸参数 | ✅✅ 规避 | 窄台基检测 + linked MTF 分支（`:2440-2586`） |
| T10 | 自动化流程无兜底，失败即中断 | ✅ 规避 | quantile fallback（`:2607`）+ chroma rescue 三级（`:2969`）+ safe review 候选（`:3268-3280`） |
| T11 | AI 直接输出拉伸数值 | ✅✅ 规避 | 三重契约（§2.9） |
| T12 | 拉伸决策不可追溯 | ✅ 规避 | 两个 stage JSON 完整落盘（§3.6 判定可复现） |
| T13 | GHS 不设 LP 导致暗部过度压缩 | ❌ 未规避 | 未见 LP 设置，走 Siril 默认 LP=0 |
| T14 | 去星失败路径无质量保护 | ❌ 未规避 | `_run_with_stars_review_stretch`（`stage6_stretching.py:8`）不走任何硬门 |

**规避率：14 项中完全规避 11 项、部分规避 1 项、未规避 2 项。**

### 4.4 D3 综合判定

- **符合**：Asinh 主拉伸｜linked MTF 三元组｜MTF midtones 反解算法（STF 算法正确复刻）｜starless 分离拉伸回星工作流
- **符合且严于行业**：黑场不裁到 0
- **符合且优于常见做法**：autostretch 仅作参照不作输出
- **部分符合**：目标背景 0.15-0.22（保守偏离，方向正确）
- **偏离**：GHS 参数语义（自创 sigma 映射，未用 SP/LP/HP）｜GHS 保色模式（未启用）
- **自创**：候选排序权重体系（无行业对标物，但设计自洽）

---

## 五、关键发现 TOP 3

### 发现 1（高危 · 可维护性）：两套完整拉伸架构并存，旧架构全死但命名更「正统」

`pipeline/stage6_services.py` 中同时存在：

- **活架构**：`_stage7_compact_stretch_candidates()`（`:2246`）固定双候选 → `_stage7_evaluate_stretch_candidate()`（`:1187`）→ `_stage7_candidate_selection_key()`（`:1747`）
- **死架构**：`_stage6_stretch_candidate()`（`:501`）→ `_stage6_strategy_from_features()`（`:541`，构建 asinh/ghs/autostretch 三组多模式策略）→ `_stage6_candidate_specs()`（`:738`）→ `stretch_candidate_evaluator.py` 全模块（`build_candidate_spec:56` / `score_candidate:134` / `allowed_as_final:180` / `choose_best:213`）→ `policy_selector.py:45-232` 的 `stage6_stretch` policy（`candidate_mode` / `forbidden_when_dirty` / `hard_reject.max_bg_dirty_score` / `scoring.*_weight`）→ `configs/policies/*.yaml`

死架构**语法完整、有测试覆盖（`tests/test_adaptive_pipeline_phase1.py`）、被 import（`stage6_services.py:49-65`）、且命名与文件名 `stage6_services.py` 更匹配**。

**具体危害**：审计 D2「排序权重硬编码还是配置」这个问题，如果只看 `configs/policies/*.yaml` 的 `scoring.core_blowout_weight`，会得出「权重可配置」的**完全错误**结论；实际权重 100% 硬编码在 `:1747-1858` 与 `:2027-2059`。同理，任何维护者想调整硬门阈值时，`policy_selector.py` 的 `hard_reject.max_bg_dirty_score=0.42` 看起来就是入口，改了却毫无效果。

### 发现 2（高危 · 可运维性）：四条最核心硬门无 env 覆盖，与 20+ 条次要阈值形成割裂

`stage6_bg_median_min`（背景中值下限）、`stage6_black_pixel_ratio_max`（黑像素比例）、`stage6_highlight_clip_ratio_max`（高光裁剪）、`stage6_star_growth_ratio_max`（星点增长）——这四条是决定「一张拉伸结果是否被接受」的最基础判据（`stage6_services.py:1533-1559`），grep 确认**全部不在 `PROJECT_ENV_ALLOWED_KEYS`**（`processor_runtime.py:170-232`）。

同时，`SEESTAR_STAGE7_STARLESS_HALO_DETAIL_DELTA_MIN`（`:190`）、`SEESTAR_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX`（`:195`）这类次要阈值却可覆盖。

**具体危害**：想验证「把黑像素上限从 0.35 收紧到 0.20 后有多少候选被拒」这类假设，必须改代码、必须重新打包、必须污染 git 工作区——而这恰恰是审计与调参最常见的需求。同时也意味着这四条门的实际生效值**无法在不同 run 之间变化**，跨 run 对照实验只能靠改源码。

### 发现 3（中危 · 可审计性）：AI 决策链缺 prompt 版本与拒绝原因，AI 路径不可完整复现

三个缺口叠加：

1. **无 prompt 版本**：grep `prompt_version|prompt_id|schema_version` 在整个 `pipeline/` 下零匹配。`ai_raw_*.json` 只记 `endpoint`/`temperature`/`json_mode`/`content`/`response`（`save_utils.py:112-150`）。prompt 是运行时拼接的字符串（`ai_advisory.py:464-624`），改动后历史归档无法区分。
2. **无模型标识**：只有 endpoint URL，后端换模型不可察觉。
3. **拒绝原因只进日志**：`stage6_services.py:3258-3261` 的 warn `"stage7 stretch selection was not revalidated"` 未结构化落盘；归档里 `ai_selection: null` 无法区分「未调用 AI」「AI 返回空」「AI 返回了无效 ID」三种情况。

**具体危害**：`selection.selector == "deterministic_quality_rank"` 时，无法从归档判断这是「本来就没开 AI」还是「AI 给了非法答案被丢弃」。后者是需要告警的质量事件，前者是正常配置——两者在归档里长得一模一样。

---

## 六、改进建议

### P0（应尽快处理）

**P0-1 · 删除或隔离 legacy 死架构**
- 位置：`pipeline/stage6_services.py:501-540`（`_stage6_stretch_candidate`）、`:541-694`（`_stage6_strategy_from_features`）、`:738-779`（`_stage6_candidate_specs`）、`:782-808`（`_stage6_candidate_label`）、`:49-65`（`stretch_candidate_evaluator` 的未用 import）
- 位置：`pipeline/stretch_candidate_evaluator.py` 整个模块
- 位置：`pipeline/policy_selector.py:45-232` 中 `stage6_stretch` 的 `candidate_mode` / `forbidden_when_dirty` / `hard_reject` / `scoring` 段，及 `pipeline/configs/policies/*.yaml` 中对应段
- 做法：先在每处加 `# DEAD CODE — not called by stage7 pipeline as of <date>; see stage6_services.py:2246` 标记（零风险），确认无外部依赖后整体删除；`stretch_candidate_evaluator.py` 若要保留仅供测试，移入 `tests/legacy/`
- 收益：消除审计与维护的头号误导面

**P0-2 · 把四条核心硬门加入 env 白名单**
- 位置：`pipeline/processor_runtime.py:170-232` 的 `PROJECT_ENV_ALLOWED_KEYS`，与 `:1036-1172` 的 float 映射表
- 新增键：`SEESTAR_STAGE6_BG_MEDIAN_MIN` → `stage6_bg_median_min`；`SEESTAR_STAGE6_BLACK_PIXEL_RATIO_MAX` → `stage6_black_pixel_ratio_max`；`SEESTAR_STAGE6_HIGHLIGHT_CLIP_RATIO_MAX` → `stage6_highlight_clip_ratio_max`；`SEESTAR_STAGE6_STAR_GROWTH_RATIO_MAX` → `stage6_star_growth_ratio_max`
- 建议同时加钳制：bg_median_min ∈ [0.002, 0.10]、black_ratio_max ∈ [0.05, 0.60]、highlight_clip_max ∈ [0.001, 0.05]、star_growth_max ∈ [1.05, 2.00]
- 收益：使核心门可做 A/B 实验，且不破坏现有默认行为

**P0-3 · 补 prompt 版本与模型标识归档**
- 位置：`pipeline/ai_advisory.py:464-624`（构造请求处）与 `pipeline/save_utils.py:112-150`（`write_ai_raw_response`）
- 做法：在 `ai_advisory.py` 模块级定义 `STAGE7_SELECTION_PROMPT_VERSION = "v1"`（每次改 prompt 手动 bump），并在 raw response payload 中加 `prompt_version`、`prompt_sha256`（对拼接后的 system+user prompt 求哈希，自动防漂移）、`model_id`（从响应体或配置读取）
- 收益：AI 路径可完整复现

### P1（应在下一轮迭代处理）

**P1-1 · 结构化归档 AI 拒绝原因**
- 位置：`pipeline/stage6_services.py:3235-3262`
- 做法：把 `ai_selection` 从 `Optional[Dict]` 扩展为始终写入的诊断对象，含 `invoked: bool`（对应 `allow_ai and accepted_attempts`）、`raw_candidate_id: str|null`、`accepted: bool`、`rejection_reason: "not_in_allowed_list" | "explicit_fallback" | "empty_response" | null`，写进 `selection_summary`（`:3325-3358`）
- 收益：填补发现 3 的第三个缺口

**P1-2 · 硬门阈值快照写入 Stage 7 自身 JSON**
- 位置：`pipeline/stage6_services.py:3360-3383`（`stage7_stretch_quality.json` 的 payload 构造）
- 做法：新增 `"hard_gate_limits"` 段，落盘 H1-H4 的实际生效值（含 `stage6_effective_bg_median_min()` 推导后的结果，而非配置原值）+ H5/H6/H11 等未落盘的阈值
- 收益：单文件自洽，无需交叉 `processing-plan.json` 即可完整复现

**P1-3 · 把 `starless_structure_quality` 加入候选报告字段白名单**
- 位置：`pipeline/stage6_services.py:3394-3412` 的字段列表
- 做法：追加 `"starless_structure_quality": item.get("starless_structure_quality")`
- 收益：主报告不再缺失秩漂移门这一 starless 专属判据

**P1-4 · 为去星失败的 review 路径补最小硬门**
- 位置：`pipeline/stages/stage6_stretching.py:8`（`_run_with_stars_review_stretch`）
- 做法：即使只做 `autostretch -linked`，也应调用 `_validate_stage6_stretch_quality()`（`stage6_services.py:1522`）计算 H1-H4 并把结果写入 stage JSON（可以不拒绝、只记录 `advisory_only: true`）
- 收益：填补 T14，使 review 路径也有质量可观测性

**P1-5 · 补 GHS 的 LP（暗部保护）设置或明确记录为有意不设**
- 位置：`pipeline/stage6_services.py:960-1059` 的 `autoghs` 命令构造
- 做法：先实测确认 Siril 1.4.0 `autoghs` 是否支持 LP 参数；若支持，对 `bg_median < 0.010` 的低背景数据设置非零 LP（参考 GHS 官方文档的 SP/LP 用法）；若不支持或有意不设，在 `models.py:210-211` 附近加注释说明依据
- 依据：https://www.ghsastro.co.uk/doc/tools/GeneralizedHyperbolicStretch/GeneralizedHyperbolicStretch.html 与 https://www.pixinsight.com/forum/index.php?threads/limitation-for-ghs-in-data-with-large-black-level.21043/

### P2（可择机改善）

**P2-1 · 目标背景值 0.15-0.22 补对标注释与 env 覆盖**
- 位置：`pipeline/models.py:352-355`
- 做法：注释中标明「PixInsight STF / Siril autostretch 的 targetbg 默认 0.25 为屏幕预览值；成品建议 0.09-0.12；本项目取中间偏低的 0.15-0.22 因后续 Stage 8/9 仍会改变整体观感」，并加入 env 白名单

**P2-2 · 收紧亮星云场景下的局部核心裁剪门**
- 位置：`pipeline/models.py:343`（`stage7_bright_nebula_star_growth_ratio_max=1.50`）与 `:368`（`stage7_local_core_clip_ratio_max=0.12`）
- 做法：当星点增长门放宽到 1.50 时，同步把局部核心裁剪门收紧（如 0.08），避免两个宽松条件叠加

**P2-3 · 建立 stage JSON 与 `ai_raw_*.json` 的交叉引用**
- 位置：`pipeline/save_utils.py:112-150` 与 `pipeline/stage6_services.py:3325-3358`
- 做法：`write_ai_raw_response` 返回其写入的文件名，由调用方写入 `selection_summary.ai_raw_file`
- 收益：填补 G7

**P2-4 · 把候选级关键度量摘要上卷到 `pipeline-result.json`**
- 位置：`pipeline/processor_runtime.py:2038-2169`
- 做法：Stage 7 的 `actual_steps` 条目下增加精简摘要（选中候选名、method、p50/p99、risk_score、fallback_used、AI 是否采纳）
- 收益：批量跨 run 分析无需逐 run 打开 stage JSON

**P2-5 · 澄清 `intensity_scale` 的阶段归属**
- 位置：`pipeline/seestar_Superimpose_workflow.md` §5.9（L734-760）与 `pipeline/stage7_quality.py:685/729/775/785`
- 做法：核实该值究竟由哪个阶段产生，统一文档与代码的阶段命名（这与全局的 stage6/stage7 命名错位是同一类问题）

---

## 七、证据索引

### 7.1 代码证据

**`pipeline/stages/stage6_stretching.py`**：Stage 7 入口 `:61`；去星失败复核路径（无硬门）`:8`；弃用别名 `:221`

**`pipeline/seestar_Superimpose.py`**：主流程 Stage 6→7 调用 `:2519-2520`；兼容检查点无条件 skipped `:2530-2534`；检查点实现 `:2005-2087`

**`pipeline/stage6_services.py`**（Stage 7 核心）：
- 背景中值下限推导 `:377`｜AI 选择转发 `:728-735`
- **死代码**：`_stage6_stretch_candidate` `:501-540`、`_stage6_strategy_from_features` `:541-694`、`_stage6_candidate_specs` `:738-779`、`_stage6_candidate_label` `:782-808`
- 候选执行分派（Siril 命令）`:960-1059`｜亮度闭环重试 `:1062-1186`｜候选评估主函数 `:1187-1445`
- starless 结构门 `:1264-1290`/`:1406-1420`｜弥散可见度门 `:1448-1482`｜有效星点增长上限 `:1485-1506`｜条件启用 `:1507-1520`
- **四条核心硬门 `:1522-1560`**｜背景色度门 `:1581-1723`｜救援强度列表 `:1724-1746`
- **确定性排序键 `:1747-1858`**｜救援原因标记 `:1862-1883`｜safe review 判定 `:1884-1945`｜救援前置条件 `:1946-1972`｜救援像素实现 `:1974-2026`
- 软风险分（硬编码系数）`:2027-2059`｜目标感知 profile `:2085-2243`
- 候选构建 `:2246-2604`（默认参数 `:2273-2279`、极低背景 offset 保护 `:2282-2315`、低背景自适应 `:2316-2334`、profile 乘子钳制 `:2336-2362`、offset 上限 `:2364-2388`、preview 标定 `:2390-2440`、**noise-floor MTF shadows `:2504-2529`**、linked MTF 参数 `:2537-2579`、双候选返回 `:2588-2604`）
- quantile 兜底 `:2607+`｜色度救援循环 `:2946-3230`｜确定性最优+AI 选择 `:3230-3262`｜rank/role 写回 `:3281-3295`｜selection_summary `:3325-3358`
- 写 `stage7_stretch_quality.json` `:3360-3383`｜写 `stretch_candidates_report.json` `:3384-3439`｜选中日志 `:3452-3476`
- 主服务范围：`:2646-3477`

**`pipeline/ai_advisory.py`**：`DEFAULT_AI_PROMPT` `:20`；通用请求与温度策略 `:464-624`；选择归一化契约 `:842-867`；选择请求与白名单过滤 `:870-930`

**`pipeline/models.py`**：Asinh/GHS 全局默认 `:208-211`；四条核心硬门 `:339-343`；目标感知与 preview 比例 `:348-350`；linked MTF 目标 P50 `:352-355`；preview 达标区间与可见度 `:357-359`；色度与重试 `:360-361`；starless 结构门 `:362-363`；quantile 与局部门 `:366-370`；色度门四阈值 `:371-374`；救援强度 `:377`；旧兼容参数 `:405-408`

**`pipeline/processor_runtime.py`**：env 白名单 `:170-232`；env 覆盖应用 `:1036-1172`；`processing-plan.json` `:1848-1990`；`pipeline-result.json` `:2038-2169`

**其它**：`save_utils.py:93`（`write_stage_json`）、`:112-150`（`write_ai_raw_response`）｜`policy_selector.py:45-232`（**死配置**）｜`stretch_candidate_evaluator.py` 全文件（**死模块**）｜`stage7_quality.py:685/729/775/785`（`intensity_scale`，归属存疑）｜`pipeline/AGENTS.md:5`（禁止纯黑背景/星色失真/饱和溢出/弱信号抹除）｜`seestar_Superimpose_workflow.md:534-577`（§5.7）、`:578-601`（§5.7.1）、`:734-760`（§5.9）

### 7.2 行业标准来源

1. PixInsight ScreenTransferFunction / AutoStretch 文档 — https://astroguide.starlust.de/html/STFDocumentation.html
2. PixInsight PCL 源码（`ScreenTransferFunctionInterface.cpp`，含 `DEFAULT_AUTOSTRETCH_SCLIP`） — https://gitlab.com/pixinsight/PCL/-/blob/master/src/modules/processes/IntensityTransformations/ScreenTransferFunctionInterface.cpp
3. PixInsight HistogramTransformation / MTF 官方说明 — https://pixinsight.com/doc/legacy/LE/16_histograms/histogram_manipulation/histogram_manipulation.html
4. Siril 拉伸处理文档 — https://siril.readthedocs.io/en/1.2/processing/stretching.html
5. Siril 命令参考（`autostretch` shadowsclip -2.8 / targetbg 0.25；`autoghs` B=13/HP=0.7/LP=0；`asinh` stretch 1-1000） — https://siril.readthedocs.io/en/stable/Commands.html
6. GHS 官网 — http://ghsastro.co.uk/
7. GHS 工具文档（SP/LP/HP/D/b 语义） — https://www.ghsastro.co.uk/doc/tools/GeneralizedHyperbolicStretch/GeneralizedHyperbolicStretch.html
8. GHS 大黑场数据讨论（含 Adam Block 讲解、Colour 保色模式） — https://www.pixinsight.com/forum/index.php?threads/limitation-for-ghs-in-data-with-large-black-level.21043/
9. SetiAstro GHS 数学实现说明 — https://github.com/setiastro/setiastrosuitepro/wiki/Function:-Hyperbolic-Stretch
10. RC-Astro StarXTerminator 官方 usage notes（线性期分离 + 分别拉伸 + Screen 回星） — https://www.rc-astro.com/starxterminator-usage-notes
11. PixInsight 论坛「After StarXTerminator」（Russell Croman unscreen/re-screen） — https://pixinsight.com/forum/index.php?threads/after-starxterminator.18773/
12. StargazersLounge「Levels and curves routine」（背景电平 y≈0.09-0.10、避免裁剪） — https://stargazerslounge.com/topic/427107-levels-and-curves-routine/
13. Jerry Lodriguss《BGAIP》§3.03h（背景电平设定） — https://astropix.com/books/BGAIP/chapter3/303h.html

### 7.3 未验证事项清单

- **U1** `STAGE7_ASINH_STRETCH_MIN/MAX` 具体数值 —— 未验证：仅确认钳制调用存在（`stage6_services.py:2337-2354`），未核对模块常量定义
- **U2/U3** Siril 1.4.0 `autoghs` 是否提供保色开关、`-linked` 是否隐含三通道链接从而避免色相漂移 —— 未验证：查阅文档为 stable 分支，未针对 1.4.0 逐项核对，未实机验证
- **U4** `intensity_scale` 阶段归属 —— 未验证：`workflow.md` §5.9 称由 Stage 7 计算，grep 显示实际在 `stage7_quality.py:685/729/775/785`，该文件阶段归属未审计
- **U5** GUI / task intake 是否暴露 Stage 7 的 PipelineConfig 字段 —— 未验证：未审计 `gui/task_intake.py` 的字段映射
- **U6** `_stage7_stretch_background_gate()`（`:1581-1723`）内部逐条门的精确判定行号 —— 未验证：仅读取起始段与排序键引用的 limits 键名，未逐行展开全部 142 行
- **U7** `stage7_stretch_metrics.assess_starless_structure_growth()` 内部实现 —— 未验证：未展开 `pipeline/stage7_stretch_metrics.py`
- **U8** `run-manifest.json` / `checkpoint-manifest.json` 中 Stage 7 的具体字段 —— 未验证：本次未读取其生成逻辑
- **U9** policy YAML 的 `stage6_stretch` 段是否在其它阶段被消费 —— 未验证：仅确认 `forbidden_when_dirty` 被死模块引用，未穷举全部键的消费方

*本报告为纯只读审计产出，审计过程中未修改 `/Users/mz/dev/aiseestart` 下任何项目代码或配置。*
