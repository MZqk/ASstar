"""Stage 5 linear cleanup: optional RL deconvolution first, then light denoise."""
from __future__ import annotations

from typing import List, Optional

from models import PipelineStage
from sirilpy.exceptions import CommandError, SirilError


def _stage5_background_risk(adaptive: dict, stage5_policy: dict) -> bool:
    if not bool(stage5_policy.get("protect_background", False)):
        return False
    dirty = float(stage5_policy.get("dirty_background_score", adaptive.get("dirty_background_score", 0.0)) or 0.0)
    chroma = float(stage5_policy.get("chroma_noise_score", adaptive.get("chroma_noise_score", 0.0)) or 0.0)
    gradient = float(stage5_policy.get("gradient_score", adaptive.get("gradient_score", 0.0)) or 0.0)
    bg_std = float(adaptive.get("bg_std", 0.0) or 0.0)
    return dirty >= 0.30 or chroma >= 0.08 or gradient >= 0.10 or bg_std >= 0.030


def _stage5_background_worsened(before: dict, after: dict) -> tuple[bool, str]:
    if not before or not after:
        return False, ""
    before_std = max(float(before.get("bg_std", 0.0) or 0.0), 1e-7)
    after_std = float(after.get("bg_std", 0.0) or 0.0)
    before_chroma = max(float(before.get("chroma_noise_score", 0.0) or 0.0), 1e-7)
    after_chroma = float(after.get("chroma_noise_score", 0.0) or 0.0)
    before_dirty = float(before.get("dirty_background_score", 0.0) or 0.0)
    after_dirty = float(after.get("dirty_background_score", 0.0) or 0.0)

    std_growth = after_std / before_std
    chroma_growth = after_chroma / before_chroma
    before_chroma_ratio = before_chroma / before_std
    after_chroma_ratio = after_chroma / max(after_std, 1e-7)
    chroma_ratio_growth = after_chroma_ratio / max(before_chroma_ratio, 1e-7)
    dirty_delta = after_dirty - before_dirty
    worsened = (
        std_growth > 1.12
        or chroma_growth > 1.15
        or chroma_ratio_growth > 1.35
        or dirty_delta > 0.06
    )
    reason = (
        f"bg_std_growth={std_growth:.3f}, "
        f"chroma_growth={chroma_growth:.3f}, "
        f"chroma_bg_ratio_growth={chroma_ratio_growth:.3f}, "
        f"dirty_delta={dirty_delta:.3f}"
    )
    return worsened, reason


def _stage5_denoise_mode(stage5_policy: dict) -> str:
    mode = str(stage5_policy.get("denoise_mode", "") or "").strip().lower()
    if mode in {"chroma_first", "luma_chroma_balanced", "full"}:
        return "full"
    return "full"


def _stage5_denoise_strength(stage5_policy: dict) -> str:
    mode = str(stage5_policy.get("denoise_mode", "") or "").strip().lower()
    if mode == "chroma_first":
        return "0.30"
    if mode == "luma_chroma_balanced":
        return "0.25"
    return "0.25"


def _run_builtin_linear_denoise(pipeline, messages: List[str]) -> bool:
    denoise_mod = max(0.20, min(0.55, float(getattr(pipeline.cfg, "stage5_builtin_denoise_mod", 0.50))))
    try:
        pipeline.log.info(f"[Stage5] Siril linear denoise: denoise -mod={denoise_mod:.2f} -indep")
        pipeline.cmd_with_check("denoise", f"-mod={denoise_mod:.2f}", "-indep")
        messages.append(f"Siril linear denoise applied (mod={denoise_mod:.2f}, indep=True)")
        return True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"[Stage5] Siril linear denoise failed: {e}")
        messages.append(f"Siril linear denoise failed: {pipeline._short_text(e, 160)}")
        return False


def _run_cosmic_clarity_linear_denoise(
    pipeline,
    *,
    denoise_mode: str,
    denoise_strength: str,
    messages: List[str],
) -> Optional[str]:
    denoise_script = pipeline._find_plugin_script(("processing/CosmicClarity_Denoise.py",))
    denoise_executable_args = pipeline._classic_cosmic_clarity_args(
        "sirilcc_denoise.conf",
        "CosmicClarity Denoise",
    )
    denoise_device_args, denoise_device_note = pipeline._classic_cosmic_clarity_device_args()
    failure_reason = ""
    used = None

    if denoise_script is not None and denoise_executable_args is not None:
        used = pipeline._run_plugin_script_by_path(
            "线性降噪回退",
            "CosmicClarity Denoise",
            denoise_script,
            args=(
                "-denoising_mode",
                denoise_mode,
                "-denoise_strength",
                denoise_strength,
                *denoise_device_args,
                *denoise_executable_args,
            ),
        )
        if not used:
            failure_reason = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or denoise_script.name
            )
    elif denoise_script is not None:
        failure_reason = "CosmicClarity executable not configured"
    else:
        failure_reason = "CosmicClarity_Denoise.py script missing"

    if not used:
        previous_native_mode = getattr(pipeline, "_cosmic_clarity_native_denoise_mode_override", None)
        previous_native_strength = getattr(pipeline, "_cosmic_clarity_native_denoise_strength_override", None)
        pipeline._cosmic_clarity_native_denoise_mode_override = denoise_mode
        pipeline._cosmic_clarity_native_denoise_strength_override = denoise_strength
        try:
            used = pipeline._run_cosmic_clarity_native_denoise_fallback("线性降噪回退")
        finally:
            if previous_native_mode is None:
                try:
                    delattr(pipeline, "_cosmic_clarity_native_denoise_mode_override")
                except AttributeError:
                    pass
            else:
                pipeline._cosmic_clarity_native_denoise_mode_override = previous_native_mode
            if previous_native_strength is None:
                try:
                    delattr(pipeline, "_cosmic_clarity_native_denoise_strength_override")
                except AttributeError:
                    pass
            else:
                pipeline._cosmic_clarity_native_denoise_strength_override = previous_native_strength

    if used:
        messages.append(
            f"CosmicClarity linear denoise fallback applied "
            f"(mode={denoise_mode}, strength={denoise_strength}, tool={used})"
        )
        return used

    messages.append(
        pipeline._fallback_summary(
            "Siril linear denoise",
            "native denoise command failed",
            f"CosmicClarity Denoise ({denoise_device_note or failure_reason})",
            False,
        )
    )
    return None


def _run_stage5_rl_deconvolution(
    pipeline,
    messages: List[str],
    *,
    fallback_stem: str = "stage5_input_linear",
) -> bool:
    if not bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)):
        messages.append("Stage5 RL deconvolution disabled")
        return False

    maxstars = max(20, min(1000, int(getattr(pipeline.cfg, "stage5_rl_maxstars", 200))))
    kernel_size = max(9, min(99, int(getattr(pipeline.cfg, "stage5_rl_psf_kernel_size", 33))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iters = max(1, min(40, int(getattr(pipeline.cfg, "stage5_rl_iters", 8))))
    alpha = max(100.0, min(10000.0, float(getattr(pipeline.cfg, "stage5_rl_alpha", 3000.0))))
    gdstep = max(0.00001, min(0.01, float(getattr(pipeline.cfg, "stage5_rl_gdstep", 0.0005))))
    stop = max(0.0001, min(0.05, float(getattr(pipeline.cfg, "stage5_rl_stop", 0.001))))

    try:
        pipeline.log.info(f"[Stage5] findstar -maxstars={maxstars}")
        pipeline.cmd_with_check("findstar", f"-maxstars={maxstars}")
        pipeline.log.info(f"[Stage5] makepsf stars -sym -ks={kernel_size} -savepsf=stage5_psf.fit")
        pipeline.cmd_with_check(
            "makepsf",
            "stars",
            "-sym",
            f"-ks={kernel_size}",
            "-savepsf=stage5_psf.fit",
        )
        pipeline.log.info(
            "[Stage5] rl "
            f"-loadpsf=stage5_psf.fit -iters={iters} -alpha={alpha:g} "
            f"-tv -gdstep={gdstep:g} -stop={stop:g}"
        )
        pipeline.cmd_with_check(
            "rl",
            "-loadpsf=stage5_psf.fit",
            f"-iters={iters}",
            f"-alpha={alpha:g}",
            "-tv",
            f"-gdstep={gdstep:g}",
            f"-stop={stop:g}",
        )
        pipeline._save_stage_output("stage5_deconv")
        messages.append(
            "Stage5 RL deconvolution applied "
            f"(maxstars={maxstars}, ks={kernel_size}, iters={iters}, alpha={alpha:g})"
        )
        return True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"[Stage5] RL deconvolution skipped: {e}")
        messages.append(f"Stage5 RL deconvolution skipped: {pipeline._short_text(e, 160)}")
        try:
            pipeline.cmd_with_check("load", fallback_stem)
        except (CommandError, SirilError) as load_error:
            pipeline.log.warn(f"[Stage5] reload {fallback_stem} failed after RL failure: {load_error}")
        return False


def run_stage5_linear_denoise(pipeline) -> None:
    """
    阶段 5: 线性整理
    - 输入固定为 stage4_color，先用原始线性数据生成 PSF 并执行可选 RL 反卷积。
    - RL 后再做轻量线性降噪，最终保存 stage5_linear。
    - 不再默认执行全局锐化，避免在线性暗背景中放大彩噪和星环。
    """
    stage_name = PipelineStage.LINEAR_DENOISE.label
    pipeline.log.stage_start(stage_name)
    status = "ok"
    messages: List[str] = []
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage5_policy = policy.get("stage5_linear", {}) if isinstance(policy, dict) else {}
    denoise_mode = _stage5_denoise_mode(stage5_policy)
    denoise_strength = _stage5_denoise_strength(stage5_policy)
    background_risk = False
    before_adaptive = {}
    after_deconv_adaptive = {}
    after_linear_adaptive = {}
    guard_triggered = False
    guard_reason = ""
    denoise_used = "none"
    deconv_applied = False
    denoise_input = "stage5_input_linear"
    final_stem = "stage5_linear"

    try:
        pipeline.cmd_with_check("load", "stage4_color")
        messages.append("loaded stage4_color")
    except (CommandError, SirilError) as e:
        status = "degraded"
        messages.append(f"load stage4_color failed, using current image: {pipeline._short_text(e, 160)}")

    try:
        pipeline.cmd_with_check("save", "stage5_input_linear")
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"stage5 baseline save failed: {e}")
        messages.append(f"stage5 input baseline save failed: {pipeline._short_text(e, 160)}")

    before_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )
    background_risk = _stage5_background_risk(before_adaptive, stage5_policy)

    deconv_applied = _run_stage5_rl_deconvolution(
        pipeline,
        messages,
        fallback_stem="stage5_input_linear",
    )
    if deconv_applied:
        denoise_input = "stage5_deconv"
        after_deconv_adaptive = (
            pipeline._adaptive_features_current()
            if hasattr(pipeline, "_adaptive_features_current")
            else {}
        )
        guard_triggered, guard_reason = _stage5_background_worsened(
            before_adaptive,
            after_deconv_adaptive,
        )
        if guard_triggered:
            pipeline.log.warn(f"[Stage5] background guard dropped RL result: {guard_reason}")
            messages.append(f"Stage5 background guard dropped RL result: {guard_reason}")
            try:
                pipeline.cmd_with_check("load", "stage5_input_linear")
                denoise_input = "stage5_input_linear"
                deconv_applied = False
            except (CommandError, SirilError) as e:
                status = "degraded"
                messages.append(f"Stage5 background guard reload failed: {pipeline._short_text(e, 160)}")

    if not deconv_applied and pipeline.process_dir:
        try:
            (pipeline.process_dir / "stage5_deconv.fit").unlink(missing_ok=True)
        except OSError as e:
            pipeline.log.warn(f"[Stage5] stale stage5_deconv cleanup failed: {e}")

    if _run_builtin_linear_denoise(pipeline, messages):
        denoise_used = "siril_builtin"
    else:
        plugin_used = _run_cosmic_clarity_linear_denoise(
            pipeline,
            denoise_mode=denoise_mode,
            denoise_strength=denoise_strength,
            messages=messages,
        )
        if plugin_used:
            denoise_used = plugin_used
        else:
            status = "degraded"
            messages.append(
                "linear denoise unavailable; stage5_linear keeps current "
                f"{'deconvolved' if deconv_applied else 'linear'} image"
            )

    linear_saved = pipeline._save_stage_output("stage5_linear")
    if not linear_saved:
        status = "degraded"
        messages.append("stage5_linear save failed")
    else:
        diff_note = pipeline._stage_diff_note("stage5_linear", denoise_input)
        if diff_note:
            messages.append(diff_note)

    linear_export_ok = pipeline._export_linear_intermediate()
    if not linear_export_ok:
        status = "degraded"
        messages.append("导出 result_linear.fit 失败")

    after_linear_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )

    compatibility_saved = pipeline._save_stage_output("stage5_denoised")
    if not compatibility_saved:
        status = "degraded"
        messages.append("stage5_denoised compatibility save failed")

    graxpert_strength = max(
        0.20,
        min(0.40, float(getattr(pipeline.cfg, "stage5_graxpert_deconv_strength", 0.30))),
    )
    pipeline._write_stage_json(
        "stage5_linear_report.json",
        {
            "stage": "stage5_linear",
            "input": "stage4_color",
            "linear_output": "stage5_linear",
            "final_linear_source": final_stem,
            "processing_order": [
                "load stage4_color",
                "save stage5_input_linear",
                "findstar/makepsf/rl if enabled",
                "denoise",
                "save stage5_linear",
            ],
            "policy": (
                pipeline._active_policy_name()
                if hasattr(pipeline, "_active_policy_name")
                else str(policy.get("policy_name", "generic_low_snr_safe"))
            ),
            "target_type": (
                pipeline._active_target_type()
                if hasattr(pipeline, "_active_target_type")
                else "generic_low_snr_safe"
            ),
            "denoise": {
                "method": denoise_used,
                "input": denoise_input,
                "output": "stage5_linear",
                "siril_builtin_mod": max(
                    0.20,
                    min(0.55, float(getattr(pipeline.cfg, "stage5_builtin_denoise_mod", 0.50))),
                ),
                "siril_independent_channels": True,
                "cosmic_clarity_fallback_mode": denoise_mode,
                "cosmic_clarity_fallback_strength": denoise_strength,
            },
            "deconvolution": {
                "enabled": bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)),
                "method": "siril_rl",
                "applied": deconv_applied,
                "output": "stage5_deconv" if deconv_applied else None,
                "runs_before_denoise": True,
                "graxpert_strength_mapping": graxpert_strength,
                "graxpert_note": (
                    "GraXpert deconvolution is intentionally not auto-invoked here; "
                    "wire the exact local invocation only after the installed tool protocol is confirmed."
                ),
            },
            "background_guard": {
                "risk": background_risk,
                "rollback": guard_triggered and not deconv_applied,
                "reason": guard_reason,
                "before": before_adaptive,
                "after_deconvolution": after_deconv_adaptive,
                "after_linear": after_linear_adaptive,
                "after_final": after_linear_adaptive,
            },
            "status": status,
            "messages": messages,
        },
    )

    elapsed = pipeline.log.stage_end(stage_name)
    pipeline._record_stage(stage_name, status, elapsed, "；".join(messages))
