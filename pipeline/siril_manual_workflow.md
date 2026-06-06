# Seestar Superimpose Siril 手工命令流程

本文只保留一种执行方式：在 Siril 中按阶段手工执行命令。自动流程中的 Python 像素统计、质量门控、候选评分和局部像素处理无法完全用 Siril 命令复刻；手工执行时按本文保存同名检查点，并根据“参数来源”和“回退”说明人工判断。

## 0. 准备

1. 在 Siril 中把工作目录设为 FITS 所在目录。
2. 新建或清空 `process/`。
3. 若使用插件脚本，确认 Siril scripts 目录中存在：
   - `processing/GraXpert-AI.py`
   - `processing/AutoBGE.py`
   - `processing/SyQon-Starless.py`
   - `processing/CosmicClarity_Denoise.py`
4. 若使用 GraXpert BGE，确认 `model_v2_0_1.onnx` 已安装到 GraXpert 模型目录。
5. 以下命令中的 `<work_dir>`、`<siril-scripts>`、`<x>` 等占位符需按实际路径或人工测量值替换。

参数总来源：

- 默认值来自 `PipelineConfig`。
- 自动调参值来自当前图像像素特征、FITS header、文件名和目录上下文。
- 目标策略来自 `target_profile.json` / `pipeline_policy.json`；手工执行时用目标类型经验替代。
- 完整参数解释见 `pipeline/seestar_Superimpose_workflow.md`。

## 1. 输入准备

已有叠加 FITS 时，把最新非中间产物复制或另存为 `process/working.fit`，再执行：

```text
cd "<work_dir>/process"
load working
save stage1_prepared
```

只有 `Light_*.fit/.fits` 单帧时，先把本轮 Light 文件整理到 `process/_light_input/`，统一命名为 `lightsrc_00001.fit`、`lightsrc_00002.fit`，再执行：

```text
cd "<work_dir>/process/_light_input"
link lightsrc -out=..
cd "<work_dir>/process"
calibrate lightsrc -debayer
register pp_lightsrc -2pass
seqapplyreg pp_lightsrc -filter-round=2.5k
stack r_pp_lightsrc rej 3 3 -norm=addscale -output_norm -rgb_equal -out=working
mirrorx_single working
load working
save stage1_prepared
```

参数来源：

- 输入过滤规则来自 `exclude_prefixes`、`exclude_suffixes`、`exclude_substrings`。
- 注册失败比例阈值来自 `stage1_register_fail_ratio_max`，默认 `0.10`。

回退：

- 有叠加 FITS 时不做 Light 预处理。
- Light 注册失败比例偏高时仍可继续，但应人工检查星点拖影。
- 无叠加 FITS 且无 Light 单帧时停止。

## 2. 裁切

打开 `stage1_prepared.fit`，根据黑边和彩色边缘人工确定裁切矩形：

```text
load stage1_prepared
crop <x> <y> <width> <height>
save stage2_corrected
```

参数来源：

- `<x> <y> <width> <height>` 来自四边黑边、暗边界、红蓝边缘伪影的人工测量。
- 自动流程会独立裁切四边，并最多额外裁掉每侧约 `2.5%` 的彩色伪影边。

回退：

- 图像太小或无法判断边界时跳过裁切：

```text
load stage1_prepared
save stage2_corrected
```

- 不要让单次彩色边缘裁切造成宽高损失超过约 `10%`。

## 3. 目标策略判断

此阶段无必须执行的 Siril 命令。人工读取 FITS header 和文件名，判断目标类型：

```text
load stage2_corrected
```

参数来源：

- `OBJECT`、`RA/DEC`、`FILTER`、`INSTRUME`。
- 目录名和文件名中的目标线索。
- 目标类型按发射星云、反射星云、星系、星团、行星状星云、宽场或未知判断。

回退：

- 不确定时按 `generic_low_snr_safe`：少扣背景、少饱和、少锐化、回星强度偏低。

## 4. 背景提取

先保存背景提取基线：

```text
load stage2_corrected
save stage3_bg_input
```

按顺序尝试候选。每个候选执行前都重新加载 `stage3_bg_input`，执行后人工检查背景是否变平、弱信号是否被保留、星点是否明显损失。选择最佳候选后保存 `stage3_bgremoved`。

```text
load stage3_bg_input
pyscript "<siril-scripts>/processing/GraXpert-AI.py" -bge -correction subtraction -smoothing 0.50
save stage3_candidate_graxpert

load stage3_bg_input
pyscript "<siril-scripts>/processing/GraXpert-AI.py" -bge -correction subtraction -smoothing 0.35
save stage3_candidate_graxpert_bge

load stage3_bg_input
pyscript "<siril-scripts>/processing/AutoBGE.py" -npoints 80 -polydegree 2 -rbfsmooth 0.08
save stage3_candidate_adbe

load stage3_bg_input
pyscript "<siril-scripts>/processing/AutoBGE.py" -npoints 120 -polydegree 3 -rbfsmooth 0.12
save stage3_candidate_dbe

load stage3_bg_input
pyscript "<siril-scripts>/processing/AutoBGE.py" -npoints 100 -polydegree 2 -rbfsmooth 0.10
save stage3_candidate_autodbe

load stage3_bg_input
subsky -rbf -samples=<samples> -tolerance=<tolerance> -smooth=<smooth>
save stage3_candidate_subsky_rbf

load stage3_bg_input
subsky 1
save stage3_candidate_subsky_poly

load <best_stage3_candidate>
save stage3_bgremoved
```

参数来源：

- `<samples>` 来自 `bg_samples`，自动流程限幅 `12-32`。
- `<tolerance>` 来自 `bg_tolerance`，自动流程限幅 `0.6-1.8`。
- `<smooth>` 来自 `bg_smooth`，自动流程限幅 `0.2-1.2`。
- 高噪声时降低 tolerance、提高 smooth。
- 复杂星场或大目标时增加候选数量。
- 大面积星云、暗弱云气或弥散目标时优先检查 `subsky 1`，避免 RBF 过拟合扣掉主体。

回退：

- GraXpert 脚本或模型不可用时，从 AutoBGE 开始。
- AutoBGE 不可用时，从 `subsky -rbf` 开始。
- RBF 过度扣星云时使用 `subsky 1`。
- 所有背景提取都不理想时：

```text
load stage3_bg_input
save stage3_bgremoved
```

## 5. 图像解析与色彩校准

默认解析和 SPCC：

```text
load stage3_bgremoved
platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3
save stage4_psolved
setcpu 1
spcc "-oscsensor=Sony IMX585" "-oscfilter=ZWO Seestar LP" "-whiteref=Average Spiral Galaxy" -limitmag=10.5
setcpu <restore_cpu>
save stage4_color
save stage4_colorbalanced
```

无 LP 滤镜时，把 SPCC 命令替换为：

```text
spcc "-oscsensor=Sony IMX585" "-oscfilter=No filter" "-whiteref=Average Spiral Galaxy" -limitmag=10.5
```

窄带或双窄带时，把 SPCC 命令替换为：

```text
spcc "-oscsensor=Sony IMX585" "-whiteref=Average Spiral Galaxy" -narrowband -rwl=656.28 -rbw=20 -gwl=500.70 -gbw=30 -bwl=500.70 -bbw=30 -limitmag=10.5
```

解析失败但 header 有中心坐标时重试：

```text
platesolve <ra>,<dec> -focal=160 -pixelsize=2.90 -catalog=gaia -order=3
```

SPCC 失败时按顺序尝试 PCC：

```text
pcc -catalog=localgaia
pcc -catalog=gaia
pcc -catalog=nomad
pcc -catalog=apass
save stage4_color
save stage4_colorbalanced
```

参数来源：

- 焦距和像元：Seestar S30 Pro 远摄光路 `160 mm / 2.90 um`。
- 解析星表默认 `gaia`。
- PCC 星表默认 `localgaia,gaia,nomad,apass`。
- SPCC sensor 默认 `Sony IMX585`。
- SPCC filter 来自显式配置、FITS header、路径/文件名或目标画像。
- `<restore_cpu>` 默认机器 CPU 数。

回退：

- 普通 platesolve 失败后用 header 坐标重试。
- SPCC 失败后用 PCC。
- PCC 失败后人工做背景中性化；发射星云/双窄带目标不要强行星点白平衡。

## 6. 线性反卷积与轻降噪

```text
load stage4_color
save stage5_input_linear
findstar -maxstars=200
makepsf stars -sym -ks=33 -savepsf=stage5_psf.fit
rl -loadpsf=stage5_psf.fit -iters=8 -alpha=3000 -tv -gdstep=0.0005 -stop=0.001
save stage5_deconv
denoise -mod=0.50 -indep
save stage5_linear
save stage5_denoised
cd "<work_dir>"
save result_linear
cd "<work_dir>/process"
```

参数来源：

- `findstar -maxstars` 来自 `stage5_rl_maxstars`，默认 `200`。
- `makepsf -ks` 来自 `stage5_rl_psf_kernel_size`，默认 `33`，必须为奇数。
- `rl -iters` 默认 `8`。
- `rl -alpha` 默认 `3000`。
- `rl -gdstep` 默认 `0.0005`。
- `rl -stop` 默认 `0.001`。
- `denoise -mod` 来自 `stage5_builtin_denoise_mod`，默认 `0.50`。

回退：

- `findstar`、`makepsf` 或 `rl` 失败时：

```text
load stage5_input_linear
denoise -mod=0.50 -indep
save stage5_linear
save stage5_denoised
```

- RL 后背景变脏或星环明显时，也回到 `stage5_input_linear`。
- `denoise` 失败时跳过降噪：

```text
load stage5_deconv
save stage5_linear
save stage5_denoised
```

## 7. 线性去星与星点层准备

先保存去星输入：

```text
load stage5_linear
save stage6_input
```

在 Siril 中运行 `SyQon-Starless.py`，输入当前 `stage5_linear`，输出 `starless.fit` 和 `starmask.fit`。成功后执行：

```text
load starless
save stage6_starless
save stage7_starless
```

如果线性图去星失败，生成轻微预拉伸输入后重试 `SyQon-Starless.py`：

```text
load stage5_linear
asinh 1.35 0.0025
save stage6_mild_prestretch_star_input
```

参数来源：

- SyQon 默认模型：Zenith v1。
- SyQon 默认 tile/overlap：`512 / 64`。
- 轻微预拉伸强度来自 `SEESTAR_MILD_PRESTRETCH_STRENGTH`，默认 `1.35`。
- GPU 开关来自 `SEESTAR_SYQON_GPU`。

回退：

- SyQon 失败时尝试轻微预拉伸输入。
- 仍失败时可用 SASP Dark Star 手工去星。
- 所有去星工具失败时，退化为不真正去星：

```text
load stage5_linear
save starless
save stage6_starless
save stage7_starless
```

- `starmask.fit` 缺失时，后续 Stage 9 跳过回星。

## 8. 主体拉伸

生成预览和两个正式候选：

```text
load stage6_starless
autostretch -linked
save stage7_preview_ref

load stage6_starless
asinh 2.2 0.002
save stage7_cand_a

load stage6_starless
asinh 2.1 0.002
autoghs -linked -2.1 1.05
save stage7_cand_b

load <best_stage7_candidate>
save stage7_stretched
```

参数来源：

- `asinh_stretch/asinh_offset` 来自自动调参；普通背景默认约 `2.2 / 0.002`。
- `autoghs` amount 普通默认约 `1.05`。
- 极低背景时降低 Asinh 强度并提高 offset，但 offset 不应高于有效信号范围。

回退：

- `stage7_cand_b` 过曝或发白时选 `stage7_cand_a`。
- 两个候选都不理想时选动态范围更正常、核心不过曝、背景不死黑的候选。
- 所有拉伸失败时：

```text
load stage6_starless
save stage7_stretched
```

## 9. Starless 增强

```text
load starless
save stage8_input_starless
satu <nebula_saturation> <nebula_bg_factor>
save starless_enhanced
save stage8_enhanced
```

参数来源：

- `<nebula_saturation>` 来自 `nebula_saturation`。
- `<nebula_bg_factor>` 来自 `nebula_bg_factor`。
- 发射星云可略提高饱和；星系、反射星云和低信噪目标应保守。

回退：

- 如果存在外部增强结果，可先导入并另存：

```text
load sasp_starless
save starless
save stage8_input_starless
```

- 增强后背景变脏、蓝偏、核心过曝或纹理伪影明显时：

```text
load stage8_input_starless
save starless_enhanced
save stage8_enhanced
```

## 10. 星点处理与合成

如果有外部 starmask，先导入：

```text
load sasp_starmask
save starmask
```

如果有 `starmask.fit`，先拉伸星点层：

```text
load starmask
asinh 2.000 0.00100
save starmask_stretched
```

回混：

```text
load starless_enhanced
pm $starless_enhanced$ + $starmask_stretched$ * <intensity>
save stage9_remixed
```

参数来源：

- starmask Asinh 默认来自 `SEESTAR_STAGE9_STARMASK_ASINH_STRETCH=2.00` 和 `SEESTAR_STAGE9_STARMASK_ASINH_OFFSET=0.001`。
- `<intensity>` 来自 `star_intensity`，自动调参后上限 `1.05`。
- 残星明显时降低 `<intensity>`，避免二次星点。

回退：

- 主强度回混星点太重时，改用 `star_fallback_intensity` 或人工降低到 `0.45-0.80`。
- 没有 starmask 时：

```text
load starless_enhanced
save stage9_remixed
```

- 回混失败时继续使用 `starless_enhanced`。

## 11. 最终降噪与导出

按优先级加载最终候选：

```text
load stage9_remixed
```

若缺失：

```text
load starless_enhanced
```

若仍缺失：

```text
load stage7_stretched
```

最终色彩和保存：

```text
satu <final_saturation> <final_bg_factor>
save stage10_final
```

可选最终降噪：

```text
pyscript "<siril-scripts>/processing/CosmicClarity_Denoise.py" -denoising_mode luminance -denoise_strength 0.5 -use_gpu <classic_executable_args>
save stage10_final
```

导出：

```text
cd "<work_dir>"
savetif "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec_$DATE-OBS:dm12$_processed" -astro
save "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec_$DATE-OBS:dm12$_processed_final"
autostretch
savepng "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec_$DATE-OBS:dm12$_processed"
```

参数来源：

- `<final_saturation>` 来自 `final_saturation`。
- `<final_bg_factor>` 来自 `final_bg_factor`。
- 输出格式默认 TIFF、FITS、PNG 全部导出。
- 动态文件名模板由 Siril 保存时展开 FITS header。

回退：

- CosmicClarity 不可用时跳过最终降噪。
- 动态命名导出失败时：

```text
savetif result_processed -astro
save result_final
savepng result_processed
```

- 从 `result_linear.fit` 续跑时使用线性续跑回退名：

```text
savetif result_processed_linear -astro
save result_final_linear
savepng result_processed_linear
```

## 12. 可选 AI 后期副本

Stage 11 不覆盖 Stage 10 主输出，只生成 `_ai` 副本。手工执行时先用外部 AI/图像工具生成 `stage11_ai_output_fit.fit`，再混合：

```text
save stage11_ai_source
load stage11_ai_source
pm $stage11_ai_source$ * <source_weight> + $stage11_ai_output_fit$ * <ai_strength>
save stage11_ai_blended
load stage11_ai_blended
savetif result_processed_ai -astro
savepng result_processed_ai
save result_final_ai
```

参数来源：

- `<ai_strength>` 来自 `SEESTAR_AI_STRENGTH`，默认 `0.12`，限幅 `0.05-0.25`。
- `<source_weight>` = `1 - ai_strength`。

回退：

- AI 输出导致背景漂移、色彩漂移、亮核增长或星点膨胀时降低 `<ai_strength>` 后重试。
- 仍不合格时不导出 AI 副本，保留 Stage 10 主输出。

## 13. 验收

1. `stage2_corrected.fit`：无明显黑边和彩色窄边。
2. `stage3_bgremoved.fit`：背景平整，但星云/暗云气没有被扣空。
3. `stage4_color.fit`：星色自然，背景不过度偏绿/偏蓝/偏红。
4. `stage5_linear.fit`：反卷积没有明显星环，背景噪声没有被放大。
5. `stage6_starless.fit`：主体保留，残星不过多；`starmask.fit` 不应包含大量星云结构。
6. `stage7_stretched.fit`：主体可见，核心不过曝，背景不死黑。
7. `stage8_enhanced.fit`：增强不过度，背景不脏，蓝偏不过量。
8. `stage9_remixed.fit`：回星不重复、不膨胀、不压过星云主体。
9. `stage10_final.fit` / 导出文件：TIFF/PNG/FITS 均可打开，PNG 仅作预览。

## 14. 续跑入口

从裁切后结果续跑：

```text
load stage2_corrected
```

然后从“背景提取”继续。

从线性结果续跑：

```text
load result_linear
```

然后从“线性去星与星点层准备”继续，并使用 `result_processed_linear.*` / `result_final_linear.fit` 作为回退输出名。
