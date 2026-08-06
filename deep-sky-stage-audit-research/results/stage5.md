# Stage 5（线性反卷积 / 轻降噪）深度审计报告

- **审计对象**：`/Users/mz/dev/aiseestart` — Seestar 望远镜离线深空后期 pipeline（Python + Siril 1.4.0）
- **审计阶段**：Stage 5 `linear_cleanup`（GraXpert Object Deconvolution → Siril RL 回退 → 轻量线性降噪）
- **主代码**：`pipeline/stages/stage5_linear_denoise.py`（1015 行）、`pipeline/noise_model.py`（541 行）、`pipeline/scunet_denoise.py`（87 行）、`pipeline/cosmic_clarity.py`（270 行）
- **契约/归档**：`pipeline/stage_contracts.py`、`pipeline/task_workspace.py`、`pipeline/task_plan.py`、`pipeline/processor_runtime.py`、`gui/task_intake.py`
- **文档基准**：`pipeline/seestar_Superimpose_workflow.md` §5.5（L443-L471）
- **审计性质**：纯只读研究，未修改任何项目代码
- **审计日期**：2026-08-05

---

## 一、摘要与总评分

Stage 5 是本 pipeline 中"线性域最后一次结构性修改"，同时是三个跨运行正式续跑点之一（`FORMAL_RESUME_STAGES = (1, 2, 5)`，`pipeline/stage_contracts.py:18`）。其输出 `stage5_linear.fit` 直接决定 Stage 6 去星模型的输入质量，因此噪声放大与细节损失在此处的代价高于任何其他线性阶段。

总体判断：**架构设计意识明显高于行业业余平均水平**（有背景守卫、有确定性候选+质量门+事务回滚、有完整 JSON 归档、有 SHA-256 断点契约），但存在**三类系统性缺陷**：(a) 部分门禁计算了却未接线（死门禁）；(b) 降噪链路的质量门只覆盖首选算法，回退路径完全裸奔；(c) 续跑指纹与实际生效配置不同源，env-only 参数可静默复用不兼容断点。

| 维度 | 评分 | 理由 |
|---|---|---|
| **D1 逻辑专业合理性** | **7.0 / 10** | 主链路顺序（反卷积→背景守卫→降噪）与 starless-first 目标一致，多尺度候选具备"基线→候选→质量门→回滚"完整事务语义（`stage5_linear_denoise.py:129-238`）；扣分点：`background_risk` 计算后未参与任何决策（死门禁，L710/L992）、`_stage5_denoise_mode()` 三分支塌缩为单一返回值（L62-66）、自动调参产出的 `denoise_mod` 从未被 Stage5 消费（`seestar_Superimpose.py:862-865` vs `stage5_linear_denoise.py:108`）、降噪结果无任何 before/after 质量门与回滚。 |
| **D2 门禁来源与归档** | **6.5 / 10** | 归档面广且结构化（`stage5_linear_report.json`/`stage5_noise_model.json`/`stage5_multiscale_denoise.json` + `checkpoint-manifest.json` 携带 SHA-256/state/config_fingerprint）；扣分点：Stage5 续跑指纹只取 8 个 GUI 字段（`gui/task_intake.py:171-183`），RL `alpha/gdstep/stop/ks`、全部 multiscale 门限、`SEESTAR_GRAXPERT_GPU` 等 env-only 参数不入指纹；断点内 `plan_hash` 实际写入的是 run-manifest 自身哈希而非 `processing-plan.json` 的 `plan_hash`（`task_workspace.py:570`）；GraXpert 模型只记录版本目录名，无权重 SHA-256（`stage5_linear_denoise.py:572`）；`status=degraded` 仍发布正式断点（`seestar_Superimpose.py:1417-1423`）。 |
| **D3 算法行业标准符合度** | **7.5 / 10** | 与 Siril 官方文档、GraXpert CLI 语义、PixInsight 主流线性期实践总体对齐：线性域做反卷积、RL 采用梯度下降 + TV 正则 + early-stop（Siril 对线性图的明确推荐）、GraXpert 强度 0.30 比官方默认 0.5 更保守、降噪置于反卷积之后符合 RC Astro 的强立场；扣分点：`stage5_rl_alpha` 的代码注释"越高越保守"与 Siril 官方"lower value = more regularization"**方向相反**；对低 SNR 线性数据未启用 Siril 官方为 Poisson 噪声准备的 `-vst`（Anscombe）；把 AI 黑盒 `-deconv_obj` 作为"物理链路"默认首选且无输出质量门，与行业对 AI 反卷积"可能生成不存在细节"的争议未做任何对冲。 |
| **总评** | **7.0 / 10** | 工程完备度高、可观测性强，但"门禁写了不用 / 门禁只护一条路 / 指纹不覆盖真实配置"三点使其尚未达到可审计的科学级线性阶段标准。 |

**最严重问题（一句话）**：Stage 5 的降噪侧完全没有 before/after 质量门与回滚——首选多尺度候选被质量门 `rejected` 后，系统会直接回退到强度更高、无任何验证的 `denoise -mod=0.50 -indep`，并把结果作为跨运行正式断点发布。

---

## 二、D1 — 逻辑专业合理性

### 2.1 执行链路总览（含实际行号）

```
load stage4_color                                   stage5_linear_denoise.py:655
save stage5_input_linear（不可变基线）                :662
_adaptive_features_current() → before_adaptive       :667-671
build_noise_model_report()（report_only，不改像素）    :673-685
_stage5_background_risk() → background_risk          :710      ← 计算后未使用
_run_stage5_graxpert_deconvolution()                 :712      ← 首选
  └─ 失败 → _run_stage5_rl_deconvolution()           :723      ← 回退
背景守卫 _stage5_background_worsened()               :750-769  ← 仅护反卷积
  └─ 触发 → load stage5_input_linear，丢弃反卷积       :762-766
降噪分支                                              :778-847
  ├─ denoise_enabled=False → skip                    :778
  ├─ multiscale 候选（质量门 + 事务回滚）              :789
  ├─ status=skipped_low_noise → 不再回退               :799-804
  ├─ Siril denoise -mod -indep                       :805  ← 无质量门
  └─ CosmicClarity Denoise 回退                       :810  ← 无质量门
save stage5_linear                                   :849
export result_linear.fit                             :871
写 stage5_linear_report.json                          :933
```

### 2.2 PSF 估计稳健性

RL 路径的 PSF 由 `findstar -maxstars=200` → `makepsf stars -sym -ks=33 -savepsf=stage5_psf.fit` 构造（`stage5_linear_denoise.py:342-350`），参数钳制为 `maxstars∈[20,1000]`（L331）、`ks∈[9,99]` 且强制奇数（L332-334）。

**合理之处**：
- 使用 `stars` 模式而非 `blind`，且强制 `-sym`（对称化），对 Seestar 这类小口径、场曲/彗差不严重的设备是稳健选择；Siril 官方明确指出 star-PSF 只在线性数据上有效，本项目正是在线性域调用，符合约束。
- `ks=33` 对 Seestar S30/S50 的典型 FWHM（约 3–6 px）留有足够冗余，不会截断 PSF 尾翼。

**不足**：
1. **无 PSF 质量校验**。`makepsf stars` 成功返回即视为可用，未检查实际参与拟合的星数、FWHM 离散度、椭圆度。若 `findstar` 只找到极少数星（薄云、月光、大片星云遮挡），会得到高方差 PSF，后续 RL 会把噪声当结构放大。代码路径中没有任何 `stage5_psf.fit` 的统计回读。
2. **`maxstars` 未随图像分辨率/星密度自适应**。`stage5_rl_maxstars` 默认 200（`models.py:160`），auto_tune 完全没有为其生成公式（`seestar_Superimpose.py:836-960` 中无 `stage5_rl_maxstars`）。星场密集的目标与稀疏星系场使用同一上限。
3. **GraXpert 路径的 `psf_size` 被硬编码为 5.0**（`stage5_linear_denoise.py:532`），既不来自配置也不来自实测 FWHM，且不可通过 env 覆盖。GraXpert 的 `-psfsize` 在官方语义中是需要与实际 PSF 匹配的参数，固定值意味着在 seeing 差异较大的批次间行为不一致。

### 2.3 RL 迭代次数与振铃抑制

实际命令（`stage5_linear_denoise.py:356-364`）：

```
rl -loadpsf=stage5_psf.fit -iters=8 -alpha=3000 -tv -gdstep=0.0005 -stop=0.001
```

**合理之处**：
- `iters=8` 低于 Siril CLI 默认 10，属保守取值；配置注释也明确"过高易放大噪声和星环"（`models.py:162`）。
- 使用 `-tv`（Total Variation）而非无正则，且使用梯度下降（未加 `-mul`）——这正是 Siril 官方对**线性图像**的推荐组合（"For linear images it is usually best to use the gradient descent Richardson-Lucy methods"）。
- 启用 `-stop=0.001` 提前停止，Siril 文档把 stopping criterion 作为抑制星周暗环的首选手段。

**问题（严重）**：`stage5_rl_alpha` 的语义方向理解错误。

- `pipeline/models.py:163` 注释：`stage5_rl_alpha: float = 3000.0  # RL TV 正则 alpha，越高越保守`
- Siril 官方文档：`-alpha=` provides the regularization strength (**lower value = more regularization**, default = 3000)

即 **alpha 越低正则越强、结果越平滑越保守；alpha 越高正则越弱、越锐利、越容易放大噪声**。代码注释写反了。当前默认值 3000 恰好等于 Siril 默认值，因此**默认行为没有实际损害**；但一旦有人依注释"提高 alpha 求保守"（钳制上限 10000，`stage5_linear_denoise.py:336`），实际会得到**正则强度约 1/3、噪声放大与星环风险显著上升**的结果。这是一个高危的语义陷阱。

另有一处相关不足：`gdstep=0.0005` 是 Siril 默认步长，但官方指出"if ringing occurs around bright stars then reduce the step size"——项目没有把"检测到星环 → 降低 gdstep 重试"的自适应逻辑接进来，唯一的兜底是事后的背景守卫（只看背景统计，不看星周环）。

### 2.4 线性域正确性

**正确的部分**：
- 输入固定为 `stage4_color`（线性、已校色），且阶段起始立刻保存不可变基线 `stage5_input_linear`（L655-662）。文档 §5.5 L449 与代码一致。
- Stage 5 明确不做全局锐化 / unsharp / 星点矫正（函数 docstring L623 + 文档 L453），避免在线性暗背景提前放大彩噪与 halo——这是正确的工程克制。
- 多尺度候选在**亮度 + 对立色度（R−G / B−G）**空间做软阈值（`noise_model.py:448-477`），而非直接在 RGB 上处理，符合"色度噪声比亮度噪声更容易在拉伸后暴露、且色度可更激进平滑"的行业共识（色度阈值倍数 1.90 vs 亮度 1.35，`noise_model.py:418-419`）。

**风险点**：
- `multiscale_denoise_candidate` 在 L490 对候选做 `np.clip(candidate, 0.0, 1.0)`。对 Siril 32-bit float（通常归一化到 [0,1]）无害，但若上游返回超出 [0,1] 的浮点线性数据，高光将被硬裁剪，破坏线性性。**未验证：未能确认 `siril.get_image_pixeldata(preview=False)` 在本项目全部输入路径下是否恒定返回 [0,1] 归一化浮点，需实机验证。**
- `_as_chw_view` 在 L27 已将通道截断为前 3 个，因此 `multiscale_denoise_candidate` L478 的 `if before.shape[0] > 3` 分支恒不成立（死代码）；对 RGB 无影响，但说明多通道（如带 alpha）路径未真正被测试。

### 2.5 噪声放大对 Stage 6（去星）的影响

Stage 6 去星以 `stage5_linear.fit` 为首选输入（文档 L473）。去星模型（StarNet/StarXTerminator 类）对**背景噪声与星周伪影高度敏感**：噪声被反卷积放大后，模型容易把噪点识别为微弱星点并挖除，产生"背景被啃出麻点"的典型失败模式。

项目对此**部分设防**：
- 反卷积后立即用 `_stage5_background_worsened()` 比较 before/after 的 `bg_std` / `chroma_noise_score` / 色度-背景比 / `dirty_background_score`，任一超阈值即整体丢弃反卷积（L750-769）。这是一个设计良好的、面向下游的保护。
- 阈值：`std_growth>1.12`、`chroma_growth>1.15`、`chroma_ratio_growth>1.35`、`dirty_delta>0.06`（L47-52）。以背景噪声允许增长 12% 为界，对反卷积而言是合理偏严的设定。

**但保护存在结构性缺口**：
1. 守卫**只覆盖反卷积**，不覆盖降噪。降噪之后计算了 `after_linear_adaptive`（L876-880）却只写进报告（L997-998），从未与 `before_adaptive` 比较。若 `denoise -mod=0.50` 或 CosmicClarity 把背景抹成塑料感 / 抹掉微弱结构，没有任何机制发现或回滚。
2. 守卫是**全局背景统计**，不含星周局部指标（ringing/暗环）。RL 的典型失败恰恰是"背景 σ 几乎不变、但亮星周围出现暗环"，当前守卫对此完全盲视。
3. 守卫触发后 reload 失败仅置 `status="degraded"`（L767-769）并继续，此时内存中的图像是**已被反卷积恶化的版本**，随后被保存为 `stage5_linear` 并（见 D2）作为正式断点发布。

### 2.6 轻降噪强度的保守性

| 通道 | 参数 | 默认值 | 钳制 | 位置 |
|---|---|---|---|---|
| 多尺度候选 | `stage5_multiscale_denoise_strength` | 0.72 | [0.10, 1.0] | `stage5_linear_denoise.py:143-155` |
| 多尺度门限 | `detail_retention_min` | 0.82 | [0.70, 0.98] | `noise_model.py:493` |
| 多尺度门限 | `noise_reduction_min` | 0.05 | [0.0, 0.50] | `noise_model.py:494` |
| Siril 内置 | `stage5_builtin_denoise_mod` | 0.50 | [0.20, 0.55] | `stage5_linear_denoise.py:108` |
| CosmicClarity | strength | "0.25"（chroma_first 为 "0.30"） | 字符串常量 | `stage5_linear_denoise.py:69-75` |

**评价**：
- 多尺度候选虽然名义强度 0.72 偏高，但有信号权重衰减（`blend = strength * (1 - 0.78*signal_weight)`，`noise_model.py:446`），亮区实际强度降到 0.72×0.22 ≈ 0.16，加上 5 项质量门（细节保留 / 噪声下降 / 裁剪增长 / 背景中值漂移 / 高光扩散），**这条路径的保守性是可信的**。
- **`stage5_builtin_denoise_mod=0.50` 是全链路最激进的一环**。Siril `denoise` 的 `-mod` 是"降噪结果与原图的混合比例"，0.50 意味着 NL-Bayes 全强度输出占一半权重。文档 §5.5 把 Stage 5 定位为"轻降噪"，而 0.50 的 NL-Bayes 在低 SNR Seestar 线性数据上并不"轻"，且**它恰好是多尺度候选被质量门拒绝后的下一站**——即"首选算法因为太伤细节被拒绝 → 改用一个更强且不检查细节的算法"。这是逻辑上的反向降级（见 TOP3-1）。
- CosmicClarity 回退强度 0.20–0.30 与文档 L454 一致，属保守。

### 2.7 「AI 降噪是否应在线性域」的项目立场

项目实际立场：**默认使用确定性算法（多尺度软阈值），AI 降噪（CosmicClarity / SCUNet）仅作为末端回退**。

- `pipeline/scunet_denoise.py:14-60` 的 `run_siril_scunet_denoise_fallback()` 优先找 `SCUNet_Denoise.py` 脚本，否则做命令别名探测（`siril_scunet_denoise` / `scunet_denoise` 等），强度经 `_clamp_float(strength, 0.0, 1.0)` 钳制（L48）。
- `pipeline/cosmic_clarity.py:215-256` 的 native denoise 默认 `mode="luminance"`、`strength="0.5"`，但 Stage5 会通过 `_cosmic_clarity_native_denoise_mode_override` 覆盖为 `full` / `0.25`（`stage5_linear_denoise.py:284-285`）。

这个"确定性优先、AI 兜底、AI 强度被显式压低"的分层是**审慎且值得肯定的**。它与项目在反卷积侧的选择形成明显反差：反卷积把 AI（GraXpert `-deconv_obj`）放在**首选**位置，且没有等价的质量门。同一阶段内两种相反的风险态度，是设计一致性上的缺陷。

### 2.8 续跑契约完整性（逻辑侧）

Stage 5 是正式续跑点，契约定义于 `pipeline/stage_contracts.py:102`：
- `primary_artifact = "stage5_linear.fit"`
- `formal_resume_checkpoint = True`
- `legacy_read_aliases = ("stage5_denoised.fit", "result_linear.fit")`

发布逻辑 `_publish_task_formal_checkpoint()`（`seestar_Superimpose.py:1425-1447`）在阶段结束后调用，前置条件（L1417-1423）：`stage ∈ (2,5)` 且 `status ∈ {ok, degraded}` 且 `input_profile.state == "linear"`。

**逻辑缺陷**：`degraded` 也发布。Stage 5 可能因以下原因 degraded 却仍被当作可信续跑点：
- `load stage4_color` 失败，用"当前图像"继续（L657-659）——输入根本不是约定的 Stage4 产物；
- 背景守卫触发但 reload 失败（L767-769）——图像是被判定为"背景恶化"的版本；
- 全部降噪器失败（L821-826）；
- `result_linear.fit` 导出失败（L872-874）。

前两种情况下发布的 `stage5_linear.fit` 在科学意义上是**不合格产物**，但它会成为后续所有运行的续跑起点，且带有完整 SHA-256 与签名，看起来"完全可信"。

### 2.9 失败降级路径完整性

降级链条本身设计完备，每一步都有兜底：

| 步骤 | 失败动作 | 行号 |
|---|---|---|
| GraXpert 脚本缺失/模型缺失 | 记 reason，转 RL | :548-566 |
| GraXpert 执行失败/空跑 | reload `stage5_input_linear`，转 RL | :592-607 |
| RL 命令失败 | reload `fallback_stem`，`deconv_applied=False` | :371-378 |
| 背景守卫触发 | reload 基线，清理 stale 产物 | :761-776 |
| 多尺度基线保存失败 | `status="prohibited"`，不执行 | :129-138 |
| 多尺度异常 | reload `stage5_pre_multiscale`，记 `rollback_performed` | :224-233 |
| Siril 内置降噪失败 | 转 CosmicClarity | :805-810 |
| 全部失败 | `status="degraded"`，保留当前图 | :821-826 |
| 噪声模型不可用 | 写 `status="unavailable"`，不影响主链 | :700-709 |

**值得肯定**：stale 产物清理（L771-776 删除 `stage5_deconv.fit` / `stage5_graxpert_deconv.fit`）避免了下游误读上一轮的反卷积中间件——这是很多 pipeline 会踩的坑。

**遗留**：多尺度基线 `stage5_pre_multiscale.fit` 在候选被拒后不会被清理，会与最终产物同处 `process/`，可能被人工误读；且 `stage5_multiscale_candidate.fit` 只在接受时写出，拒绝时无法离线复盘"到底差在哪一项"（虽然 JSON 里有 metrics）。

---

## 三、D2 — 门禁来源、钳制与归档

### 3.1 全部门禁穷举

| # | 门禁 | 文件:行号 | 阈值/取值 | 来源分类 | 是否可 env 覆盖 | 是否落盘 |
|---|---|---|---|---|---|---|
| G1 | 背景风险判定 `dirty` | `stage5_linear_denoise.py:28` | `>= 0.30` | **hardcoded** | 否 | 是（`background_guard.risk`） |
| G2 | 背景风险判定 `chroma` | 同上 :28 | `>= 0.08` | hardcoded | 否 | 是 |
| G3 | 背景风险判定 `gradient` | 同上 :28 | `>= 0.10` | hardcoded | 否 | 是 |
| G4 | 背景风险判定 `bg_std` | 同上 :28 | `>= 0.030` | hardcoded | 否 | 是 |
| G5 | 守卫 `std_growth` | :48 | `> 1.12` | hardcoded | 否 | 是（reason 字符串） |
| G6 | 守卫 `chroma_growth` | :49 | `> 1.15` | hardcoded | 否 | 是 |
| G7 | 守卫 `chroma_ratio_growth` | :50 | `> 1.35` | hardcoded | 否 | 是 |
| G8 | 守卫 `dirty_delta` | :51 | `> 0.06` | hardcoded | 否 | 是 |
| G9 | RL `maxstars` | :331 | `[20,1000]`，默认 200 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_RL_MAXSTARS` | 部分（消息串/plan） |
| G10 | RL `kernel_size` | :332-334 | `[9,99]` 强制奇数，默认 33 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_RL_PSF_KS` | 部分 |
| G11 | RL `iters` | :335 | `[1,40]`，默认 8 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_RL_ITERS` | 部分 |
| G12 | RL `alpha` | :336 | `[100,10000]`，默认 3000 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_RL_ALPHA` | **否**（未写入 report） |
| G13 | RL `gdstep` | :337 | `[1e-5,0.01]`，默认 5e-4 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_RL_GDSTEP` | **否** |
| G14 | RL `stop` | :338 | `[1e-4,0.05]`，默认 1e-3 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_RL_STOP` | **否** |
| G15 | GraXpert `strength` | :525-531 | `[0.20,0.40]`，默认 0.30 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_GRAXPERT_DECONV_STRENGTH` | 是（`deconvolution.graxpert.strength`） |
| G16 | GraXpert `psf_size` | :532 | **固定 5.0** | **hardcoded** | 否 | 是 |
| G17 | GraXpert 模型版本格式 | :18, :381-385 | 必须 `^\d+\.\d+\.\d+$` | hardcoded 正则 | 否 | 是（reason code） |
| G18 | GraXpert GPU 开关 | :91-104 | 默认 True | env-only | `SEESTAR_GRAXPERT_GPU` | 是（`hardware_acceleration`） |
| G19 | 多尺度 `strength` | :143-155 | `[0.10,1.0]`，默认 0.72 | PipelineConfig + **调用点内联钳制** | `SEESTAR_STAGE5_MULTISCALE_DENOISE_STRENGTH` | 是（candidate JSON） |
| G20 | 多尺度 `detail_retention_min` | `noise_model.py:493` | `[0.70,0.98]`，默认 0.82 | PipelineConfig + **函数内钳制** | `SEESTAR_STAGE5_MULTISCALE_DETAIL_RETENTION_MIN` | 是（`limits`） |
| G21 | 多尺度 `noise_reduction_min` | `noise_model.py:494` | `[0.0,0.50]`，默认 0.05 | 同上 | `SEESTAR_STAGE5_MULTISCALE_NOISE_REDUCTION_MIN` | 是 |
| G22 | 多尺度 `clip_growth_max` | `noise_model.py:495` | `0.001` | **hardcoded** | 否 | 是 |
| G23 | 多尺度 `background_median_drift_max` | `noise_model.py:496` | `0.003` | hardcoded | 否 | 是 |
| G24 | 多尺度 `bright_spread_growth_max` | `noise_model.py:497` | `0.03` | hardcoded | 否 | 是 |
| G25 | 低噪跳过判定 | `noise_model.py:510` | `bg_luma_sigma_before <= 1e-5` | hardcoded | 否 | 是（`status`） |
| G26 | 阈值倍数（亮度/色度） | `noise_model.py:418-419` | 1.35 / 1.90 | **函数默认参数，调用点未传** | 否 | 否 |
| G27 | Siril 内置 `-mod` | `stage5_linear_denoise.py:108` | `[0.20,0.55]`，默认 0.50 | PipelineConfig + CLAMP_RULES | `SEESTAR_STAGE5_BUILTIN_DENOISE_MOD` | 是（`siril_builtin_mod`） |
| G28 | CosmicClarity strength | :69-75 | `"0.30"` / `"0.25"` | **hardcoded 字符串** | 否 | 是 |
| G29 | 降噪总开关 | :778 | `denoise_enabled` 默认 **False** | PipelineConfig + auto_tune | `SEESTAR_DENOISE_ENABLE` / `SEESTAR_DENOISE_FORCE` | 是（reason_code） |
| G30 | 反卷积总开关 | :328/:538 | 默认 True | PipelineConfig | `SEESTAR_STAGE5_DECONV_ENABLE` | 是 |
| G31 | GraXpert 分支开关 | :541-546 | 默认 True | PipelineConfig | `SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE` | 是 |
| G32 | 断点发布状态门 | `seestar_Superimpose.py:1417-1423` | `status ∈ {ok, degraded}` 且 `state == "linear"` | hardcoded | 否 | 是（checkpoint-manifest） |

**统计**：32 项门禁中，**14 项为硬编码**（G1-G8、G16、G17、G22-G24、G26、G28、G32），占 44%。其中 G1-G8（背景风险 + 背景守卫全部 8 个阈值）是 Stage 5 唯一的图像质量保护机制，却完全没有配置化、没有 env 覆盖、没有 auto_tune 参与——这意味着**任何目标类型、任何 SNR 条件下都用同一套阈值**，也无法在不改代码的前提下做灵敏度实验。

### 3.2 「死门禁」明确清单

1. **`background_risk`（G1-G4）计算但从不参与决策**。`stage5_linear_denoise.py:710` 赋值，此后唯一引用是 L992 写进 `background_guard.risk`。`_stage5_background_risk()` 还要求 `stage5_policy["protect_background"]` 为真才计算（L22-23），否则直接返回 False。也就是说：策略层辛苦算出的"这张图背景很脏，反卷积要小心"从未转化为任何参数收敛（例如降低 iters、降低 GraXpert strength、收紧守卫阈值）。

2. **`_stage5_denoise_mode()` 三分支塌缩**（L62-66）：`chroma_first` / `luma_chroma_balanced` / `full` 三种输入全部返回 `"full"`。`pipeline/policy_selector.py:46,100,167,217` 精心为不同目标类型设置了 `denoise_mode`（多处为 `chroma_first`），到 Stage 5 全部失效。只有 `_stage5_denoise_strength()`（L69-75）还残留区分（0.30 / 0.25 / 0.25），且仅作用于 CosmicClarity 这条最末端回退。

3. **`denoise_mod` / `denoise_safety_max` 从未被 Stage 5 消费**。auto_tune 为 `denoise_mod` 生成了噪声自适应公式 `0.24 + 0.22*noise_score - 0.06*core_score`（`seestar_Superimpose.py:861-865`），并在 `clamp_config()` 中以 `denoise_safety_max` 二次限幅（L764-768）。但 Stage 5 实际读取的是 `stage5_builtin_denoise_mod`（`stage5_linear_denoise.py:108`），后者**不在 auto_tune 的调参列表中**，恒为 0.50 或 GUI/env 值。结果是：**降噪强度自适应链路整条断开**，唯一还生效的自适应量是 `denoise_enabled = noise_score > 0.35`（L856-860）。

4. **`noise_model.py:418-419` 的阈值倍数**（1.35/1.90）只存在于函数默认参数，`stage5_linear_denoise.py:141-170` 的调用点未传递，也无配置项、无 env、不落盘——属于事实上无法调节且无法追溯的隐藏常量。

### 3.3 归档路径穷举

| 产物 | 写入位置 | 内容要点 | 完整性评价 |
|---|---|---|---|
| `stage5_input_linear.fit` | `process/`，:662 | 不可变基线 | 无 SHA 记录 |
| `stage5_graxpert_deconv.fit` | `process/`，:609 | GraXpert 中间件 | 失败时被清理（:773） |
| `stage5_deconv.fit` | `process/`，:365 | RL 中间件 | 同上 |
| `stage5_pre_multiscale.fit` | `process/`，:129 | 多尺度事务基线 | **拒绝后不清理** |
| `stage5_multiscale_candidate.fit` | `process/`，:197 | 仅接受时写出 | 拒绝时无候选可复盘 |
| `stage5_linear.fit` | `process/`，:849 | **契约主产物** | 进入断点，有 SHA-256 |
| `result_linear.fit` | `work_dir/`，:871 | 旧 CLI 兼容导出 | 记入 pipeline-result（`processor_runtime.py:2049-2055`，含 sha256） |
| `stage5_noise_model.json` | `_write_stage_json`，:682 | 噪声模型（report_only），schema `seestar.multiscale-noise-model.v1` | 完整 |
| `stage5_multiscale_denoise.json` | :792 | 候选 metrics/limits/issues/transaction | 完整 |
| `stage5_linear_report.json` | :933 | 处理顺序、policy、denoise/deconvolution 子状态、background_guard before/after、components、messages | **最完整的一份**，但缺 RL 实参 |
| `processing-plan.json` | `processor_runtime.py:1883-1981` | `redact_sensitive(asdict(cfg))` 全量配置 + `plan_hash` | 全量 cfg 在此，含 RL alpha 等 |
| `pipeline-result.json` | `processor_runtime.py:2038-2174` | checkpoints（含 `result_linear` 的 sha256）、actual_steps、`manifest_hash` | 完整 |
| `run-manifest.json` | `task_workspace.py:406-423` | resume 记录、`checkpoint_fingerprints`、`manifest_hash` | 完整 |
| `checkpoints/stage5_linear.fit` + `checkpoint-manifest.json` | `task_workspace.py:519-581` | sha256、state="linear"、plan_hash、config_fingerprint、run_id、completed_at | 见下方缺口 |

### 3.4 续跑点契约的四个归档缺口

**缺口 1 — 配置指纹字段不全（高危）**

Stage 5 的 `config_fingerprint` 由 `build_resume_fingerprints()`（`task_plan.py:179-229`）对 Stage1-5 的 `stage_config` 做累计 canonical hash 生成，而 Stage 5 的 `stage_config` 只有 8 个字段（`gui/task_intake.py:171-183`）：

```
denoise_mode, deconvolution_mode, graxpert_model_path, compute_mode,
builtin_denoise_strength, graxpert_deconv_strength, rl_iterations, rl_maxstars
```

**未被纳入指纹但确实改变输出的参数**：
- `stage5_rl_alpha` / `stage5_rl_gdstep` / `stage5_rl_stop` / `stage5_rl_psf_kernel_size`（env-only，`processor_runtime.py:897-901`）
- `stage5_multiscale_denoise_enabled` / `_strength` / `_detail_retention_min` / `_noise_reduction_min`（env-only，`processor_runtime.py:821-829, 879-894`）
- `SEESTAR_GRAXPERT_GPU`（影响 ONNX provider，可能带来数值差异）
- `denoise_enabled` 的 auto_tune 结果（依赖图像特征，不进 GUI 设置）

后果：用 `SEESTAR_STAGE5_RL_ALPHA=9000` 跑一轮后，再以默认值发起新运行，`latest_compatible_resume_stage()`（`task_plan.py:232-240`）会判定 Stage 5 断点兼容并直接复用——**复用的是一个用完全不同正则强度产生的图像**，且日志中不会有任何不兼容提示。

**缺口 2 — `plan_hash` 名不副实**

`task_workspace.py:570`：`"plan_hash": str(run_payload.get("manifest_hash") or "")`。`manifest_hash` 是 run-manifest 自身经 `_signed_payload()`（L76-78）计算的签名哈希，与 `processing-plan.json` 中真正携带全量 `cfg` 的 `plan_hash` 不是同一个值。因此断点里"看起来绑定了处理计划"，实际只绑定了 run-manifest 的自洽性。`begin_task_run()` 对它的校验也只是"非空"（L400-403），不做任何比对。全量有效配置虽然归档在 `processing-plan.json` 里，但**没有任何链路把它绑进续跑判定**。

**缺口 3 — GraXpert 模型版本归档不足**

`stage5_linear_denoise.py:572` 记录 `"model": model.parent.name`（即版本目录名，如 `1.0.1`），L573 记录 `resolved_model_path`。**未记录 `model.onnx` 的 SHA-256、文件大小、mtime**。同一版本号目录下的权重文件被替换（用户手动覆盖、GraXpert 应用自更新）不会被察觉，而模型是 AI 反卷积结果的决定性因素。`graxpert_model_path` 虽在续跑指纹里，但那只是**路径字符串**，不是内容摘要。

**缺口 4 — degraded 断点无降级标记**

`checkpoint-manifest.json` 的 Stage 5 记录（`task_workspace.py:563-574`）包含 `stage/artifact/path/size/sha256/state/plan_hash/config_fingerprint/run_id/completed_at`，**没有 `stage_status` 字段**。一个 `status="degraded"`（例如背景守卫回滚失败）的产物与一个完美的 `ok` 产物在断点清单中完全无法区分。

### 3.5 归档正确性的加分项

需要明确肯定的部分：
- `publish_formal_checkpoint()` 做了**复制后二次 SHA-256 校验**（`task_workspace.py:539-541`），并在失败时用硬链接备份回滚（L521-534）——这是罕见的严谨做法。
- **下游断点级联失效**：发布 Stage N 时会遍历所有 `> N` 的已存断点，若其 `config_fingerprint` 与当前不符则删除（L543-561），避免"Stage 2 重跑了但 Stage 5 老断点还在"的脏续跑。
- `begin_task_run()` 的续跑校验链完整：路径必须在 `checkpoints/` 内（L389-391）、文件名符合契约（L392-393）、SHA-256 逐字节比对（L394-397）、`state == "linear"`（L398-399）。

---

## 四、D3 — 算法行业标准符合度

### 4.1 反卷积：位置与前提

| 行业基准 | 立场 | 本项目 | 判定 |
|---|---|---|---|
| RC Astro（BlurXTerminator 作者 Russell Croman） | "Deconvolution, to truly be called deconvolution, **requires linear data that has not been noise-reduced**" | Stage5 在线性域、Stage4 之后、任何降噪之前执行反卷积 | ✅ 完全符合 |
| Telescope.live / PixInsight 主流教程 | BXT "should be used in the **linear stage**, immediately after integration, channel combination, color calibration and background flattening" | Stage3 背景 → Stage4 校色 → Stage5 反卷积，顺序一致 | ✅ 符合 |
| Siril 官方文档 | "Images containing stars, especially linear (unstretched) data, **should always be deconvolved using the Richardson-Lucy methods**"（禁用 Split Bregman / Wiener） | 仅使用 `rl`，未使用 `sb` / `wiener` | ✅ 符合 |
| Siril 官方文档 | 线性图应用**梯度下降** RL；拉伸图才用 `-mul` | 未加 `-mul`，默认梯度下降 | ✅ 符合 |
| Siril 官方文档 | star-PSF（`makepsf stars`）**只在线性图上有效** | 线性域调用 | ✅ 符合 |

**结论**：反卷积的"何时做"这一最关键的行业问题，本项目答对了。

参考：
- https://www.rc-astro.com/noisexterminator-2-ai3-user-manual-pixinsight/
- https://telescope.live/blog/pixinsight-blur-xterminator-process
- https://siril.readthedocs.io/en/latest/processing/deconvolution.html

### 4.2 RL 参数对标 Siril 官方

| 参数 | Siril 默认/建议 | 本项目 | 判定 |
|---|---|---|---|
| `-iters` | CLI 默认 10；文档建议"逐渐增加直到星周出现暗环再回退" | 8（钳制 1-40） | ✅ 保守合理 |
| 正则类型 | `-tv`（Total Variation）或 `-fh`（Frobenius-Hessian），默认无 | `-tv` | ✅ 符合（TV 是天文 RL 的学术主流，见 4.5） |
| `-alpha` | 默认 3000，**lower = more regularization** | 3000，钳制 [100,10000] | ⚠️ 值正确，**注释语义反向**（`models.py:163`） |
| 求解方式 | 线性图用梯度下降（默认），拉伸图可用 `-mul` | 梯度下降 | ✅ 符合 |
| `-gdstep` | 默认 0.0005；出现星环时**减小步长** | 0.0005，无自适应 | ➖ 值符合，缺自适应 |
| `-stop` | 早停可减少星周振铃；星环时**增大 stop 值** | 0.001，无自适应 | ➖ 同上 |
| 反卷积前轻降噪 | Siril 文档建议：背景噪声放大时，反卷积前用 **Anscombe VST 轻降噪（modulation 50-60%）** | **未实现**（降噪一律在反卷积之后） | ⚠️ 与 Siril 建议不符，但与 RC Astro 立场一致（见 4.4） |

参考：https://siril.readthedocs.io/en/latest/processing/deconvolution.html

### 4.3 GraXpert Object Deconvolution 对标

Siril 1.4 起内置 `GraXpert_AI.py` pyscript，官方参数语义（https://siril.readthedocs.io/fr/latest/processing/graxpert.html）：
- 操作：`-denoise` / `-deconv_obj` / `-deconv_stellar` / `-bge`
- `-strength`：0.0–1.0
- `-psfsize`：PSF 尺寸
- `-model`：模型版本（如 `"3.0.2"`），省略则取最新
- `-listmodels`：列出可用版本
- `-gpu` / `-nogpu`

本项目调用（`stage5_linear_denoise.py:576-585`）：`-deconv_obj -strength 0.30 -psfsize 5.0 -model <version> -gpu|-nogpu`。

| 项 | 行业默认 | 本项目 | 判定 |
|---|---|---|---|
| strength | 0.5（GraXpert 官方/Siril 社区实测默认） | 0.30，钳制 [0.20,0.40] | ✅ 更保守，适合低 SNR Seestar |
| psfsize | 5.0（默认） | 5.0 **硬编码** | ➖ 值合理，但不随 seeing 自适应 |
| model 版本 | 支持显式指定 | 显式指定（本地最高语义版本） | ✅ 可复现性优于"取最新" |
| 模型联网下载 | GraXpert 支持远程模型 | **禁止**，缺失即降级 RL（文档 L455） | ✅ 符合离线定位 |
| `-listmodels` 校验 | 官方提供 | **未使用**，靠文件系统 glob 自行发现 | ➖ 与官方发现机制并行，存在版本不一致风险 |
| 只做 `-deconv_obj`，不做 `-deconv_stellar` | 两者可分别使用 | 只用 object | ✅ 合理（星点锐化在 starless-first 链路中风险更高） |

参考：
- https://siril.readthedocs.io/fr/latest/processing/graxpert.html
- https://github.com/Steffenhir/GraXpert/
- https://bbs.imufu.cn/forum.php?mod=viewthread&action=printable&tid=822518（Siril 1.4.0 beta3 集成 GraXpert AI 脚本的实测记录，含 strength/PSF 默认值 0.5/5.0）

### 4.4 「降噪放线性期还是拉伸后」— 多方观点

这是用户明确要求查清的争议点。检索到的立场分布如下：

**观点 A — 线性期降噪，且必须在反卷积之后（RC Astro / Russell Croman，工具作者，最强立场）**
> "Noise reduction of any kind should **never** be applied before deconvolution of any kind, including BlurXTerminator... NoiseXTerminator can just as easily process linear data as non-linear (stretched) data... Internally, NoiseXTerminator applies a stretch prior to reducing noise, and then precisely reverses it afterwards."
> 
> "Deconvolution, to truly be called deconvolution, requires linear data that has not been noise-reduced."
> — https://www.rc-astro.com/noisexterminator-2-ai3-user-manual-pixinsight/
> （同一结论被第三方教程站转述：https://astroguide.starlust.de/html/NoiseXTerminator1.html）

**观点 B — 线性期轻降噪 + 拉伸后再补一次（AstroBackyard / Trevor Jones，实践派主流）**
> "Run noise reduction on the data in its linear, unstretched state... **Avoid heavy noise reduction before deconvolution**... Light noise reduction early (linear) > Deconvolution/BlurXTerminator > Stretch > Optional Light Noise Reduction (background only)."
> — https://astrobackyard.com/astrophotography-noise/

注意 B 与 A 有微妙冲突：B 允许"轻降噪在反卷积之前"，A 则完全禁止。

**观点 C — 拉伸后降噪效果更好（PixInsight 论坛用户实践）**
> "My experience is that it works best very near the end of all the processing, on a non-linear image."
> — https://www.pixinsight.com/forum/index.php?threads/when-to-use-noise-xterminator-linear-of-non-linear-stage.22069/

**观点 D — Siril 官方：反卷积前用 Anscombe VST 轻降噪来抑制背景噪声放大**
> 文档在讨论 RL 的噪声放大时建议：反卷积前用 Anscombe VST 做轻度降噪（modulation 50–60%）。
> — https://siril.readthedocs.io/en/latest/processing/deconvolution.html
> Siril `denoise` 的 `-vst`（Anscombe 变换）专为 Poisson / Poisson-Gaussian 噪声设计，适用于"photon-starved"与"very low SNR"图像。
> — https://siril.readthedocs.io/zh-tw/1.2/processing/denoising.html

**观点 E — 中文社区批判视角：AI 降噪必须在拉伸前低强度运行**
> "NoiseXTerminator（NX-T）必须在拉伸前以低强度运行……如果将强力降噪放在非线性阶段，AI 将不可避免地在拉伸后的高对比度区域产生伪影。"
> — https://tsight.io/articles/19711668

**本项目的定位与判定**：
项目采用 **反卷积 → 降噪，全部在线性域**（`stage5_linear_denoise.py:712 → :778`），这与观点 A（最权威的工具作者立场）和观点 B 的主体顺序一致，是**可辩护的主流选择**。

但存在一个针对性缺口：项目的目标数据是 Seestar 智能望远镜的短曝光叠加结果，属于典型的 **photon-starved / low SNR / Poisson 主导**场景，而 Siril 官方恰恰为这种数据提供了 `-vst`（Anscombe）路径（观点 D）。项目的 `denoise -mod=0.50 -indep`（L111）**未启用 `-vst`**，等于用为 AWGN 设计的 NL-Bayes 直接处理 Poisson 主导的噪声。这在算法适配性上是明确的次优选择。

（补充事实：Siril 文档指出 `-vst` 不能与 `-da3d` / `-sos` 组合，但与 `-mod` / `-indep` 可共存，因此不存在技术冲突。**未验证：本项目 Siril 1.4.0 构建下 `-vst -indep` 组合的实际行为，需实机验证。**）

### 4.5 RL 正则化的学术基准

- **TV 正则（Rudin-Osher-Fatemi 1992）+ RL** 是天文/自适应光学图像复原的标准组合。近期代表性工作：Guo, Lu & Li, *Richardson–Lucy Iterative Blind Deconvolution with Gaussian Total Variation Constraints for Space Extended Object Images*, Photonics 11(6):576, 2024 — https://doi.org/10.3390/photonics11060576 。该文明确指出 TV 约束能改善 RL 的收敛性并抑制噪声放大，同时强调**正则化参数选择方案（regularization parameter selection scheme）本身是核心难点**。
- 对照本项目：TV 类型选对了（`-tv`），但正则强度 `alpha` 是固定常数 3000，**没有任何基于图像 SNR 的自适应选择方案**，而学术界公认这是决定 RL 复原质量的关键。同时 `noise_model.py` 已经在同一阶段测出了 `background_luma_sigma`、多尺度 MAD、信号-噪声曲线（`build_noise_model_report`，L173+），这些量完全具备驱动 alpha 自适应的信息基础，却只用于 `report_only`。这是"数据已在手边但未闭环"的典型遗憾。
- Siril 的另一选项 `-fh`（Frobenius norm of the Hessian，基于二阶导数）在抑制 TV 常见的"阶梯状伪影（staircasing）"上更优，项目未做对比评估。**未验证：本项目是否曾评估过 `-fh`，代码与文档中均无痕迹。**

### 4.6 AI 反卷积的行业争议与项目暴露面

行业对 AI 反卷积（BXT / GraXpert deconv）的分歧非常尖锐：

- **支持方**：Vicent Peris（PixInsight 核心圈）在硬测试后承认 BXT V2 相较 V1 有巨大进步，"I still have to find an image with an artifact"，但同时警告"A ML tool is a black box, it will never be perfect... once you find a problem, you cannot know how to correct it" — https://pixinsight.com/forum/index.php?threads/clarification-regarding-use-of-blurxterminator-in-mars-pi-survey.24241/
- **批评方**：StarTools 作者 Ivo 的对比表直指要害 —— 传统反卷积"Risk: amplifies visible noise (**but readily detectable**)"，神经网络"Risk: **hallucinations and artifacts that look plausible**"；且 BXT 厂商"divulges precisely 0 about the actual workings"（不公开训练数据、架构、参数量） — https://forum.startools.org/viewtopic.php?t=3063
- **社区常识**：Cloudy Nights 讨论串中反复出现"AI-based deblurring, in many/most cases, invents details... consider the results to be more artwork than any sort of scientific reference" — https://www.cloudynights.com/forums/topic/855011-experimenting-with-the-new-blurxterminator-from-rc-astro/
- **中文批判视角**：AI 三件套组合易产生"塑料感"，且警告"必须避免将 BlurXTerminator 与强力的形态学锐化重复叠加" — https://tsight.io/articles/19711668

**对本项目的判定**：
项目在文档中把自身定位为保守的"物理链路"（Stage 5 明确不做全局锐化、不做转色、GraXpert 强度压到 0.30），这一点值得肯定。但存在两处与行业争议未对冲的暴露面：

1. **AI 反卷积是默认首选而非可选增强**。`stage5_graxpert_deconvolution_enabled` 默认 True（`models.py:159`），只有模型缺失才降级 RL。对一个强调可复现与物理正确性的 pipeline，把不可解释的黑盒放在确定性算法之前，是价值取向上的不一致（对比：降噪侧恰恰是"确定性优先、AI 兜底"）。
2. **GraXpert 输出无内容级质量门**。背景守卫只看 4 项背景统计（`stage5_linear_denoise.py:47-52`），无法检测 AI 幻觉最典型的失效形式——**背景干净、但星系旋臂/星云丝状结构被"补全"出不存在的细节**。相比之下，多尺度降噪候选有 5 项质量门 + 事务回滚，两者的严谨度差距显著。

---

## 五、关键发现 TOP 3

### TOP 1 —（严重）降噪回退链是"反向降级"，且全程无质量门与回滚

**证据**：
- 首选路径：多尺度候选，5 项质量门（`noise_model.py:492-516`），失败即事务回滚（`stage5_linear_denoise.py:224-233`）。
- 拒绝分支：`status="rejected"` 时（例如 `signal_detail_retention < 0.82`，即"太伤细节"），代码在 `stage5_linear_denoise.py:799` 只识别 `skipped_low_noise`，其余全部落到 L805 `_run_builtin_linear_denoise()`。
- 该函数执行 `denoise -mod=0.50 -indep`（L111），**Siril NL-Bayes 全强度输出占 50% 权重**，比刚被拒绝的候选（亮区有效强度约 0.16）激进得多，且**没有任何 before/after 指标比较、没有回滚**。
- `after_linear_adaptive` 在 L876 被采集，但仅写入报告（L997-998），从不与 `before_adaptive` 比较。

**影响**：质量门起到了"筛掉温和方案、放行激进方案"的反效果。在 Seestar 常见的低 SNR + 强信号目标（如明亮星云核心）上，候选最容易因细节保留不足被拒，随后被 `-mod=0.50` 处理，Stage 6 去星模型将拿到一张被过度平滑的输入。

### TOP 2 —（严重）跨运行续跑指纹不覆盖真实生效配置，可静默复用不兼容断点

**证据**：
- Stage 5 `config_fingerprint` 仅由 8 个 GUI 字段生成（`gui/task_intake.py:171-183` → `task_plan.py:179-229`）。
- 至少 9 个改变输出的参数为 env-only、不入指纹：`stage5_rl_alpha` / `_gdstep` / `_stop` / `_psf_kernel_size`（`processor_runtime.py:897-901`）、4 个 multiscale 参数（L879-894 + L821-829）、`SEESTAR_GRAXPERT_GPU`（`stage5_linear_denoise.py:15,91-104`）。
- `plan_hash` 实为 run-manifest 自身哈希（`task_workspace.py:570`），且只做非空校验（L400-403），全量 cfg 虽归档在 `processing-plan.json`（`processor_runtime.py:1883-1981`）却不参与任何比对。
- GraXpert 模型只存版本目录名（`stage5_linear_denoise.py:572`），无权重 SHA-256。

**影响**：续跑机制表面严谨（SHA-256 + 签名 + 级联失效），实际存在一条静默通道：改 env 重跑 → 系统认为配置未变 → 直接复用旧 Stage 5 断点 → 用户以为跑了新参数，实际拿到旧结果。这对一个以"可审计、可复现"为卖点的离线 pipeline 是根本性缺陷。

### TOP 3 —（中高）背景风险门禁与自适应降噪强度整条断链，且 `alpha` 注释语义反向

**证据**：
- `background_risk`（4 个阈值，`stage5_linear_denoise.py:21-28`）在 L710 计算，唯一去处是 L992 的报告字段，从不影响任何参数。
- `_stage5_denoise_mode()`（L62-66）三分支恒返回 `"full"`，`policy_selector.py:46/100/167/217` 设置的 `chroma_first` 等策略全部失效。
- auto_tune 的 `denoise_mod` 公式（`seestar_Superimpose.py:861-865`）+ `denoise_safety_max` 限幅（L764-768）产出的自适应强度，Stage 5 从不读取；实际读的是不参与 auto_tune 的 `stage5_builtin_denoise_mod`（`stage5_linear_denoise.py:108`）。
- `models.py:163` 注释"RL TV 正则 alpha，越高越保守"与 Siril 官方"lower value = more regularization"方向相反（https://siril.readthedocs.io/en/latest/processing/deconvolution.html）。

**影响**：Stage 5 对外呈现"策略驱动 + 自动调参 + 风险感知"的完整形态，实际生效的自适应量只有 `denoise_enabled` 一个布尔值。加上 alpha 注释误导，后续维护者按注释调参会朝错误方向走。

---

## 六、改进建议

### P0（必须修，直接影响科学正确性与可复现性）

**P0-1｜为降噪结果补 before/after 质量门与回滚**
- 位置：`pipeline/stages/stage5_linear_denoise.py:805-847`（Siril 内置与 CosmicClarity 两条回退分支）
- 做法：把 `_run_multiscale_linear_denoise` 已有的事务模式（保存基线 → 执行 → `_quality_metrics` 比较 → 不达标 reload）抽为通用包装，套用到 `_run_builtin_linear_denoise()` 与 `_run_cosmic_clarity_linear_denoise()`。至少复用 `noise_model._quality_metrics()`（`pipeline/noise_model.py:335-410`）中的 `signal_detail_retention` 与 `background_median_drift`。
- 同时：修正 L799 的分支语义——多尺度候选因 `signal_detail_retention` 不足被拒时，**不应**升级到更激进的 `-mod=0.50`，应改为降低 multiscale strength 重试一次，或直接跳过降噪并记 `reason_code="detail_guard_skip"`。

**P0-2｜续跑指纹改为绑定实际生效配置**
- 位置：`gui/task_intake.py:171-183`（Stage 5 字段表）与 `pipeline/task_plan.py:179-229`
- 做法：Stage 5 的 `stage_config` 补入 `rl_alpha`、`rl_gdstep`、`rl_stop`、`rl_psf_kernel_size`、`multiscale_denoise_enabled/strength/detail_retention_min/noise_reduction_min`、`graxpert_gpu`。更彻底的方案：改为对 `redact_sensitive(asdict(cfg))` 中所有 `stage5_*` 与 `denoise_*` 键做 canonical hash（`processor_runtime.py:1883-1981` 已有该 payload），从根本上消除 GUI 字段表与实际配置的同步负担。

**P0-3｜断点绑定真实 plan_hash 并记录 GraXpert 权重摘要与阶段状态**
- 位置：`pipeline/task_workspace.py:563-574`
- 做法：(a) `plan_hash` 改为读取 `processing-plan.json` 的 `plan_hash` 字段，并在 `begin_task_run()`（L400-403）中做实际比对而非仅非空检查；(b) 新增 `stage_status` 字段，`degraded` 的产物默认不可作为续跑起点（或需显式 env 放行）；(c) `stage5_linear_denoise.py:569-575` 的 `details` 增加 `model_sha256` 与 `model_size`，并透传到报告与断点。

**P0-4｜修正 `stage5_rl_alpha` 注释语义**
- 位置：`pipeline/models.py:163`
- 做法：改为"RL TV 正则 alpha（Siril 语义：**值越低正则越强、结果越保守**；默认 3000 为 Siril 官方默认）"。同步检查 `pipeline/seestar_Superimpose.py:478` 的钳制区间说明与任何 GUI 提示文案。

### P1（重要，显著提升算法适配性与可控性）

**P1-1｜为低 SNR 线性数据启用 Anscombe VST**
- 位置：`pipeline/stages/stage5_linear_denoise.py:111`
- 做法：新增 `stage5_builtin_denoise_vst_enabled` 配置（默认按 `noise_model` 报告的背景 σ 与信号-噪声曲线自动判定），命中时命令改为 `denoise -mod=<m> -vst -indep`。依据：Siril 官方将 `-vst` 定位为 photon-starved / Poisson 主导数据的专用路径（https://siril.readthedocs.io/zh-tw/1.2/processing/denoising.html），与 Seestar 数据特性高度吻合。

**P1-2｜接线 `background_risk`，让风险感知真正影响参数**
- 位置：`pipeline/stages/stage5_linear_denoise.py:710`
- 做法：`background_risk=True` 时对反卷积做收敛：GraXpert strength 取区间下限（0.20）、RL `iters` 降至 4–6、`stop` 放宽（增大以更早停止，符合 Siril 抑制振铃的建议）、守卫阈值收紧（如 `std_growth > 1.06`）。所有收敛动作写入 `stage5_linear_report.json` 的新字段 `risk_adaptation`。

**P1-3｜把自适应降噪强度接回 Stage 5**
- 位置：`pipeline/seestar_Superimpose.py:861-865`（auto_tune）与 `pipeline/stages/stage5_linear_denoise.py:108`
- 做法：二选一 —— (a) 为 `stage5_builtin_denoise_mod` 与 `stage5_multiscale_denoise_strength` 增加 auto_tune 公式（可复用 `noise_score`）；(b) 让 Stage 5 直接消费已有的 `denoise_mod`（并保留 `stage5_builtin_denoise_mod` 作为显式覆盖）。同时把 `stage5_multiscale_*` 四个字段补入 `AUTO_CLAMP_FIELDS`（L330-337）与 `CLAMP_RULES`（L474-481），目前它们只在调用点内联钳制，未纳入统一限幅体系。

**P1-4｜为 GraXpert 输出增加结构级质量门**
- 位置：`pipeline/stages/stage5_linear_denoise.py:750-769`（背景守卫之后）
- 做法：在现有 4 项背景指标之外，增加针对 AI 幻觉的检测：(a) 高频能量增益上限（反卷积后细节能量增长超过 N 倍即回滚）；(b) 星点 FWHM 收缩比下限/上限（过度收缩说明被"钉子化"）；(c) 复用 `_quality_metrics` 的 `bright_spread_growth`。依据：行业对 AI 反卷积"生成看似合理的伪细节"的普遍警告（https://forum.startools.org/viewtopic.php?t=3063）。

**P1-5｜PSF 质量校验**
- 位置：`pipeline/stages/stage5_linear_denoise.py:342-350`
- 做法：`findstar` 后回读星表统计（星数、FWHM 中位数与 MAD、椭圆度），星数低于阈值（如 30）或 FWHM 离散度过大时跳过 RL 并记 `reason_code="psf_unreliable"`，而不是用低质量 PSF 硬跑。同时把 PSF 统计写入 `stage5_linear_report.json`。

### P2（改善可维护性、可观测性与一致性）

**P2-1｜落盘 RL 实参**
- `stage5_linear_report.json` 的 `deconvolution` 段（`pipeline/stages/stage5_linear_denoise.py:975-990`）目前不含 `iters/alpha/gdstep/stop/maxstars/kernel_size` 的实际取值（仅出现在 messages 字符串里）。应结构化写入，与 `denoise` 段的 `siril_builtin_mod` 保持对称。

**P2-2｜清理死代码与塌缩分支**
- `_stage5_denoise_mode()`（`stage5_linear_denoise.py:62-66`）：要么实现 `chroma_first` / `luma_chroma_balanced` 的真实差异化，要么删除分支并在文档中声明策略层 `denoise_mode` 对 Stage 5 无效。
- `noise_model.py:478` 的 `before.shape[0] > 3` 分支恒不成立（上游 `_as_chw_view` L27 已截断到 3 通道），应删除或修正 `_as_chw_view`。

**P2-3｜暴露隐藏常量**
- `noise_model.py:418-419` 的 `luma_threshold_multiplier=1.35` / `chroma_threshold_multiplier=1.90` 与 L495-497 的三个硬编码上限，应提升为 `PipelineConfig` 字段并落盘到 `limits`（目前后三者已落盘，前两者未落盘）。
- `stage5_linear_denoise.py:532` 的 `psf_size=5.0` 应配置化，并考虑由实测 FWHM 推导。

**P2-4｜清理事务中间件**
- 多尺度候选被拒后 `stage5_pre_multiscale.fit` 未清理（对比 L771-776 对反卷积中间件的清理）。建议对齐处理，或在报告中显式说明其保留用途。

**P2-5｜使用 GraXpert 官方 `-listmodels` 校验版本**
- `stage5_linear_denoise.py:494-514` 用文件系统 glob 自行发现模型版本目录。建议在调用前用 `-listmodels` 交叉验证，避免自行发现的目录与 GraXpert 运行时实际可用版本不一致。

**P2-6｜评估 `-fh` 正则替代/对比 `-tv`**
- Siril 提供 Frobenius-Hessian 正则（二阶导数），对 TV 常见的 staircasing 伪影更友好。建议在离线评测集上与 `-tv` 做 A/B，结论写入 workflow 文档。

**P2-7｜文档与代码的模型发现顺序对齐**
- `pipeline/seestar_Superimpose_workflow.md:450` 描述的发现顺序为"随包模型 → 本机 GraXpert 已装模型 → `SEESTAR_GRAXPERT_OBJECT_MODEL_PATH`"，而 `stage5_linear_denoise.py:474-514` 的实际顺序是"env 配置路径优先 → 本机 model roots"。**未验证：GUI 侧是否在设置 env 前已完成前两级探测，若是则文档与代码可自洽，需核对 `gui/main_window.py:2438-2443` 的模型路径来源。** 建议在文档中显式区分"GUI 探测顺序"与"pipeline 解析顺序"。

---

## 七、证据索引

### 7.1 代码证据

| 编号 | 位置 | 内容 |
|---|---|---|
| C01 | `pipeline/stages/stage5_linear_denoise.py:21-28` | `_stage5_background_risk()`，4 个硬编码阈值 |
| C02 | `pipeline/stages/stage5_linear_denoise.py:31-59` | `_stage5_background_worsened()`，4 个硬编码增长阈值 |
| C03 | `pipeline/stages/stage5_linear_denoise.py:62-66` | `_stage5_denoise_mode()` 三分支塌缩为 `"full"` |
| C04 | `pipeline/stages/stage5_linear_denoise.py:69-75` | CosmicClarity 强度 0.30 / 0.25 硬编码 |
| C05 | `pipeline/stages/stage5_linear_denoise.py:78-88` | `_stage5_disabled_denoise_reason()` 三种跳过原因 |
| C06 | `pipeline/stages/stage5_linear_denoise.py:91-104` | `SEESTAR_GRAXPERT_GPU` 解析，非法值 warn 后默认 True |
| C07 | `pipeline/stages/stage5_linear_denoise.py:107-117` | `denoise -mod -indep`，mod 钳制 [0.20,0.55] |
| C08 | `pipeline/stages/stage5_linear_denoise.py:120-238` | 多尺度候选完整事务（基线/候选/回滚） |
| C09 | `pipeline/stages/stage5_linear_denoise.py:143-169` | multiscale 三参数取值与内联钳制 |
| C10 | `pipeline/stages/stage5_linear_denoise.py:241-319` | CosmicClarity 回退（脚本 → native） |
| C11 | `pipeline/stages/stage5_linear_denoise.py:322-378` | RL 反卷积全部参数与命令 |
| C12 | `pipeline/stages/stage5_linear_denoise.py:381-436` | GraXpert 模型语义版本解析与择新 |
| C13 | `pipeline/stages/stage5_linear_denoise.py:439-514` | 模型隔离 HOME 链接与发现根列表 |
| C14 | `pipeline/stages/stage5_linear_denoise.py:517-615` | GraXpert 反卷积执行、失败回退、模型记录（L572） |
| C15 | `pipeline/stages/stage5_linear_denoise.py:654-710` | 主流程前段：load / baseline / noise model / risk |
| C16 | `pipeline/stages/stage5_linear_denoise.py:712-776` | 反卷积调度、背景守卫、stale 清理 |
| C17 | `pipeline/stages/stage5_linear_denoise.py:778-847` | 降噪四分支与降级链 |
| C18 | `pipeline/stages/stage5_linear_denoise.py:849-880` | 保存、导出、after 指标采集 |
| C19 | `pipeline/stages/stage5_linear_denoise.py:882-1015` | components 状态机与报告落盘 |
| C20 | `pipeline/noise_model.py:11-12` | `DEFAULT_MAX_SIDE=1024`、`DEFAULT_SCALES=(1,2,4,8,16)` |
| C21 | `pipeline/noise_model.py:274-299` | `_full_float_chw` / `_restore_like`（整数归一化、浮点不归一） |
| C22 | `pipeline/noise_model.py:413-535` | `multiscale_denoise_candidate` 全流程与质量门 |
| C23 | `pipeline/noise_model.py:492-517` | 5 项 limits、issues、`accepted` 判定与三态 status |
| C24 | `pipeline/scunet_denoise.py:14-60` | SCUNet 回退与强度钳制 |
| C25 | `pipeline/cosmic_clarity.py:215-256` | native denoise CLI 选项，默认 luminance/0.5 |
| C26 | `pipeline/stage_contracts.py:18` | `FORMAL_RESUME_STAGES = (1, 2, 5)` |
| C27 | `pipeline/stage_contracts.py:102` | Stage5 契约（primary_artifact / legacy aliases） |
| C28 | `pipeline/models.py:149-166` | Stage5 全部默认配置（含 L163 alpha 反向注释） |
| C29 | `pipeline/seestar_Superimpose.py:329-337` | `AUTO_CLAMP_FIELDS` 中的 stage5 字段（无 multiscale） |
| C30 | `pipeline/seestar_Superimpose.py:474-481` | `CLAMP_RULES` 中的 stage5 区间（无 multiscale） |
| C31 | `pipeline/seestar_Superimpose.py:764-772` | `denoise_mod` 安全限幅、RL kernel 奇数修正 |
| C32 | `pipeline/seestar_Superimpose.py:856-865` | auto_tune 的 `denoise_enabled` / `denoise_mod` 公式 |
| C33 | `pipeline/seestar_Superimpose.py:1417-1447` | Stage 2/5 断点发布条件与 `_publish_task_formal_checkpoint` |
| C34 | `pipeline/seestar_Superimpose.py:1992-1993` | `stage5_linear_denoise()` 薄封装 |
| C35 | `pipeline/processor_runtime.py:809-910` | Stage5 全部 env 覆盖（bool 与数值两组） |
| C36 | `pipeline/processor_runtime.py:1883-1981` | `_write_processing_plan()`，全量 cfg + plan_hash |
| C37 | `pipeline/processor_runtime.py:2025-2055` | `result_linear` 记录（含 sha256）写入 pipeline-result |
| C38 | `pipeline/task_workspace.py:76-90` | `_signed_payload()` 与 `manifest_hash` |
| C39 | `pipeline/task_workspace.py:374-404` | 续跑记录校验链（路径/文件名/SHA-256/state/指纹非空） |
| C40 | `pipeline/task_workspace.py:453-583` | `publish_formal_checkpoint()`：二次 SHA 校验、下游断点级联失效、记录字段 |
| C41 | `pipeline/task_plan.py:179-229` | `build_resume_fingerprints()` 累计哈希 |
| C42 | `gui/task_intake.py:171-183` | Stage5 续跑指纹的 8 个字段 |
| C43 | `gui/main_window.py:329-350` | `DEFAULT_PROCESSING_SETTINGS` 中的 Stage5 相关键 |
| C44 | `gui/main_window.py:2406-2443` | GUI → env 映射（仅 4 个 Stage5 数值参数） |
| C45 | `pipeline/policy_selector.py:46,100,167,217` | 策略层 `denoise_mode`（chroma_first 等） |
| C46 | `pipeline/seestar_Superimpose_workflow.md:443-471` | §5.5 阶段 5 官方文档描述 |

### 7.2 行业标准与文献证据（URL）

| 编号 | 来源 | URL | 用途 |
|---|---|---|---|
| R01 | Siril 官方文档 — Deconvolution | https://siril.readthedocs.io/en/latest/processing/deconvolution.html | `rl` 参数语义、alpha 方向、线性 vs 拉伸建议、振铃/噪声警告、反卷积前 VST 轻降噪建议 |
| R02 | Siril 官方文档 — Commands | https://siril.readthedocs.io/en/latest/Commands.html | `makepsf` 完整语法与 `stars -sym -ks=` 语义 |
| R03 | Siril 1.2 文档 — Deconvolution | https://siril.readthedocs.io/en/1.2/processing/deconvolution.html | RL/SB/Wiener 对比、PSF 生成方式 |
| R04 | Siril 文档 — Noise Reduction | https://siril.readthedocs.io/zh-tw/1.2/processing/denoising.html | NL-Bayes、Anscombe VST（Poisson）、DA3D/SOS、算法选型准则 |
| R05 | Siril 源码解读（denoise 命令） | https://deepwiki.com/michal2229/siril__mirror/7.5-noise-reduction | `-mod` / `-indep` / `-vst` / `-da3d` / `-sos` / `-rho` 参数表与 VST 组合限制 |
| R06 | Siril 官方文档 — GraXpert 接口 | https://siril.readthedocs.io/fr/latest/processing/graxpert.html | `-deconv_obj` / `-strength` / `-psfsize` / `-model` / `-listmodels` / `-gpu` 语义 |
| R07 | GraXpert GitHub | https://github.com/Steffenhir/GraXpert/ | CLI 参数、denoising strength 默认 0.5、`-ai_version` |
| R08 | Siril 1.4.0 beta3 + GraXpert 实测记录 | https://bbs.imufu.cn/forum.php?mod=viewthread&action=printable&tid=822518 | AI 反卷积（恒星/星云）默认 strength 0.5、PSF 5.0 |
| R09 | RC Astro — NoiseXTerminator 2/AI3 User Manual | https://www.rc-astro.com/noisexterminator-2-ai3-user-manual-pixinsight/ | "NR 永远不得在反卷积之前"、NXT 线性/非线性皆可、内部先拉伸再反变换 |
| R10 | Starlust AstroGuide — NoiseXTerminator | https://astroguide.starlust.de/html/NoiseXTerminator1.html | 第三方对 R09 结论的转述与线性期定位 |
| R11 | AstroBackyard — Astrophotography Noise | https://astrobackyard.com/astrophotography-noise/ | "线性期降噪"主流实践、反卷积前避免重降噪、完整顺序建议 |
| R12 | PixInsight 论坛 — NXT linear or non-linear | https://www.pixinsight.com/forum/index.php?threads/when-to-use-noise-xterminator-linear-of-non-linear-stage.22069/ | 反方观点：拉伸后降噪效果更好 |
| R13 | Telescope.live — BlurXTerminator Process | https://telescope.live/blog/pixinsight-blur-xterminator-process | BXT 应在线性期使用（integration/校色/背景之后） |
| R14 | xray-echo — Seestar PixInsight Linear Processing | https://xray-echo.com/amateurastronomy/2024/seestar-pixinsight-processing-tutorial-linear-processing | Seestar 数据的线性期标准流程（SPCC → GraXpert → BXT → 拉伸） |
| R15 | ChaoticNebula — PixInsight Deconvolution | https://chaoticnebula.com/pixinsight-deconvolution | 经典 Regularized RL 参数实践（iters 10-20、deringing、wavelet regularization） |
| R16 | NRStellar — LRGB Workflow | https://nrstellar.com/blogs/articles/lrgb-editing-workflow-for-pixinsight | 线性期顺序（Decon → StarX → LinearFit → 拉伸 → 降噪）的另一种主流变体 |
| R17 | Guo, Lu & Li (2024), Photonics 11(6):576 | https://doi.org/10.3390/photonics11060576 | RL + Gaussian TV 约束；强调正则化参数选择方案的关键性 |
| R18 | Cloudy Nights — BlurXTerminator Review | https://www.cloudynights.com/articles/astro-gear-today/reviews/software/blurxterminator-review-a-new-era-for-astroimaging-r4655/ | "linear state, before any noise reduction"；需要良好 SNR 才有效 |
| R19 | Cloudy Nights 讨论串 — Experimenting with BXT | https://www.cloudynights.com/forums/topic/855011-experimenting-with-the-new-blurxterminator-from-rc-astro/ | AI 反卷积"发明细节"的社区争议 |
| R20 | StarTools 论坛 — SV Decon vs BlurX | https://forum.startools.org/viewtopic.php?t=3063 | 传统反卷积 vs 神经网络的风险对比表、黑盒透明度批评 |
| R21 | PixInsight 论坛 — BXT in Mars-Pi survey | https://pixinsight.com/forum/index.php?threads/clarification-regarding-use-of-blurxterminator-in-mars-pi-survey.24241/ | Vicent Peris：V1 普遍伪影、V2 显著改善、ML 黑盒不可修正 |
| R22 | tsight.io — AI 的悖论：PixInsight 管线批判 | https://tsight.io/articles/19711668 | 中文批判视角：AI 降噪须拉伸前低强度、"塑料感"、避免叠加锐化 |
| R23 | istarshooter — PixInsight LRGB 全流程 | https://istarshooter.com/zh/articles/268 | 中文主流流程佐证：线性期 BXT 位置 |
| R24 | explorethecosmos — AI 降噪工具综述 | https://explorethecosmos.org/using-ai-to-clean-up-noisy-astrophotography-images | GraXpert 降噪的星点软化风险与保守设置建议 |

### 7.3 未验证项汇总

| 编号 | 未验证内容 | 原因 |
|---|---|---|
| U01 | `siril.get_image_pixeldata(preview=False)` 在本项目全部输入路径下是否恒返回 [0,1] 归一化浮点 | 纯静态审计无法确认运行时数值范围；若存在 >1 的线性数据，`noise_model.py:490` 的 clip 会破坏高光线性性 |
| U02 | Siril 1.4.0 实际构建下 `denoise -mod -vst -indep` 组合的行为 | 文档说明 `-vst` 不可与 `-da3d`/`-sos` 组合，但未明确 `-indep` 组合；需实机验证 |
| U03 | 项目是否曾评估 `-fh`（Frobenius-Hessian）正则 | 代码与 workflow 文档中均无任何痕迹，无法判断是"评估后放弃"还是"未考虑" |
| U04 | GUI 侧 GraXpert 模型探测顺序是否与 workflow 文档 L450 描述一致 | 已确认 `gui/main_window.py:2438-2443` 只做"有则设 env、无则 unset"，未见前两级探测实现；需进一步核对 GUI 模型选择控件的数据来源 |
| U05 | `stage5_rl_alpha` 提高到钳制上限 10000 时的实际画质退化程度 | 属于需要真实数据集的实验验证，静态审计只能依据 Siril 文档的语义推断方向 |
| U06 | GraXpert Object Deconvolution 在 Seestar 低 SNR 数据上的伪细节发生率 | 需要建立带真值的评测集；行业争议（R19-R21）表明风险真实存在但量级依数据而异 |
| U07 | 背景守卫 4 个阈值（1.12/1.15/1.35/0.06）的标定依据 | 代码、注释、workflow 文档中均无推导过程或实验记录 |

---

*报告完*
