"""Stage 5 linear cleanup: optional RL deconvolution first, then light denoise."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from models import PipelineStage
import stage5_deconvolution_quality
import stage8_pixels
from noise_model import (
    assess_denoise_candidate,
    build_noise_model_report,
    multiscale_denoise_candidate,
)
from sirilpy.exceptions import CommandError, SirilError


GRAXPERT_OBJECT_MODEL_ENV = "STARUN_GRAXPERT_OBJECT_MODEL_PATH"
GRAXPERT_GPU_ENV = "STARUN_GRAXPERT_GPU"
_ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_GRAXPERT_MODEL_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
STAGE5_DECONV_BG_STD_GROWTH_MAX = 1.38


def _stage5_background_risk(adaptive: dict, stage5_policy: dict) -> bool:
    if not bool(stage5_policy.get("protect_background", False)):
        return False
    dirty = float(stage5_policy.get("dirty_background_score", adaptive.get("dirty_background_score", 0.0)) or 0.0)
    chroma = float(stage5_policy.get("chroma_noise_score", adaptive.get("chroma_noise_score", 0.0)) or 0.0)
    gradient = float(stage5_policy.get("gradient_score", adaptive.get("gradient_score", 0.0)) or 0.0)
    bg_std = float(adaptive.get("bg_std", 0.0) or 0.0)
    return dirty >= 0.30 or chroma >= 0.08 or gradient >= 0.10 or bg_std >= 0.030


def _stage5_background_worsened(
    before: dict,
    after: dict,
    pipeline=None,
) -> tuple[bool, str]:
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
    cfg = getattr(pipeline, "cfg", None)
    std_growth_max = float(
        getattr(cfg, "stage5_deconv_bg_std_growth_max", 1.38)
    )
    chroma_growth_max = float(
        getattr(cfg, "stage5_deconv_chroma_growth_max", 1.15)
    )
    chroma_ratio_growth_max = float(
        getattr(cfg, "stage5_deconv_chroma_ratio_growth_max", 1.35)
    )
    dirty_delta_max = float(
        getattr(cfg, "stage5_deconv_dirty_delta_max", 0.06)
    )
    worsened = (
        std_growth > std_growth_max
        or chroma_growth > chroma_growth_max
        or chroma_ratio_growth > chroma_ratio_growth_max
        or dirty_delta > dirty_delta_max
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


def _stage5_disabled_denoise_reason(pipeline) -> str:
    """Explain why the optional denoise component did not run."""
    if getattr(pipeline, "_force_denoise_enabled", None) is False:
        return "user_disabled"
    manual_fields = getattr(pipeline, "_task_manual_override_fields", ()) or ()
    if "denoise_enabled" in manual_fields:
        return "user_disabled"
    return "config_disabled"


def _stage5_graxpert_hardware_acceleration_enabled(pipeline) -> bool:
    """Allow ONNX provider auto-selection unless CPU compatibility is requested."""
    raw_value = os.getenv(GRAXPERT_GPU_ENV)
    if raw_value is None:
        return True
    normalized = raw_value.strip().lower()
    if normalized in _ENV_TRUE_VALUES:
        return True
    if normalized in _ENV_FALSE_VALUES:
        return False
    pipeline.log.warn(
        f"{GRAXPERT_GPU_ENV} has invalid value; defaulting to automatic hardware acceleration"
    )
    return True


def _stage5_builtin_denoise_mod(pipeline) -> float:
    configured = float(getattr(pipeline.cfg, "denoise_mod", 0.35))
    safety_max = max(
        0.20,
        min(0.55, float(getattr(pipeline.cfg, "denoise_safety_max", 0.55))),
    )
    return max(0.20, min(safety_max, configured))


def _run_builtin_linear_denoise(pipeline, messages: List[str]) -> bool:
    denoise_mod = _stage5_builtin_denoise_mod(pipeline)
    try:
        pipeline.log.info(
            f"[Stage5] Siril linear denoise candidate: "
            f"denoise -mod={denoise_mod:.2f} -indep"
        )
        pipeline.cmd_with_check("denoise", f"-mod={denoise_mod:.2f}", "-indep")
        messages.append(
            f"Siril linear denoise candidate generated "
            f"(mod={denoise_mod:.2f}, indep=True)"
        )
        return True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"[Stage5] Siril linear denoise failed: {e}")
        messages.append(f"Siril linear denoise failed: {pipeline._short_text(e, 160)}")
        return False


def _run_multiscale_linear_denoise(
    pipeline,
    messages: List[str],
    *,
    baseline_pixels: Any,
) -> tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "schema": "starun.multiscale-denoise-candidate.v1",
        "status": "unavailable",
        "accepted": False,
    }
    try:
        candidate, report = multiscale_denoise_candidate(
            baseline_pixels,
            strength=max(
                0.10,
                min(
                    1.0,
                    float(
                        getattr(
                            pipeline.cfg,
                            "stage5_multiscale_denoise_strength",
                            0.72,
                        )
                    ),
                ),
            ),
            detail_retention_min=float(
                getattr(
                    pipeline.cfg,
                    "stage5_multiscale_detail_retention_min",
                    0.82,
                )
            ),
            noise_reduction_min=float(
                getattr(
                    pipeline.cfg,
                    "stage5_multiscale_noise_reduction_min",
                    0.05,
                )
            ),
            chroma_noise_growth_max=float(
                getattr(
                    pipeline.cfg,
                    "stage5_denoise_chroma_noise_growth_max",
                    1.05,
                )
            ),
        )
        report["transaction"]["baseline_saved"] = True
        report["transaction"]["pixels_mutated"] = False
        if not bool(report.get("accepted")):
            messages.append(
                "Stage5 multiscale candidate not applied: "
                f"status={report.get('status')}, "
                f"issues={','.join(report.get('issues') or []) or 'none'}"
            )
            return False, report

        safe_writer = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(safe_writer):
            safe_writer(candidate, label="Stage5 multiscale linear denoise")
        else:
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                raise RuntimeError("Siril pixel writer unavailable")
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(candidate)
            else:
                pipeline.log.warn(
                    "Stage5 multiscale denoise: image_lock unavailable"
                )
                set_pixels(candidate)
        report["transaction"]["pixels_mutated"] = True

        if not pipeline._save_stage_output("stage5_multiscale_candidate"):
            raise RuntimeError("candidate checkpoint save failed")
        report["transaction"].update(
            candidate_saved=True,
            rollback_performed=False,
        )
        metrics = report.get("metrics") or {}
        messages.append(
            "Stage5 deterministic multiscale denoise accepted "
            f"(noise_reduction={float(metrics.get('background_noise_reduction', 0.0)):.3f}, "
            f"detail_retention={float(metrics.get('signal_detail_retention', 0.0)):.3f})"
        )
        return True, report
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        report.update(
            status="failed",
            accepted=False,
            error=str(error),
        )
        messages.append(
            "Stage5 multiscale denoise candidate failed: "
            f"{pipeline._short_text(error, 160)}"
        )
        return False, report


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
            f"CosmicClarity linear denoise candidate generated "
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


def _stage5_current_pixels(pipeline) -> np.ndarray:
    pixels = pipeline.siril.get_image_pixeldata(preview=False)
    array = np.asarray(pixels)
    if array.size == 0:
        raise ValueError("empty Stage5 image buffer")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("nonfinite Stage5 image buffer")
    return np.array(array, copy=True)


def _stage5_restore_denoise_baseline(
    pipeline,
    *,
    baseline_stem: str,
    baseline_pixels: Optional[np.ndarray],
    rollback_label: str = "Stage5 denoise baseline rollback",
) -> Dict[str, Any]:
    """优先回载不可变检查点，失败时用冻结内存像素进行第二重恢复。"""

    def verify_restored_pixels() -> None:
        if baseline_pixels is None:
            return
        restored = _stage5_current_pixels(pipeline)
        if restored.shape != baseline_pixels.shape or not np.allclose(
            restored,
            baseline_pixels,
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        ):
            raise RuntimeError(
                "restored pixels do not match frozen Stage5 baseline"
            )

    try:
        pipeline.cmd_with_check("load", baseline_stem)
        verify_restored_pixels()
        return {
            "required": True,
            "completed": True,
            "method": "checkpoint_reload",
            "pixel_verified": baseline_pixels is not None,
        }
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as checkpoint_error:
        checkpoint_error_text = str(checkpoint_error)

    try:
        if baseline_pixels is None:
            raise RuntimeError("frozen Stage5 baseline pixels unavailable")
        safe_writer = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(safe_writer):
            safe_writer(
                np.array(baseline_pixels, copy=True),
                label=rollback_label,
            )
        else:
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                raise RuntimeError("Siril pixel writer unavailable")
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(np.array(baseline_pixels, copy=True))
            else:
                set_pixels(np.array(baseline_pixels, copy=True))
        verify_restored_pixels()
        return {
            "required": True,
            "completed": True,
            "method": "frozen_pixel_restore",
            "checkpoint_error": checkpoint_error_text,
        }
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as pixel_error:
        return {
            "required": True,
            "completed": False,
            "method": "failed",
            "checkpoint_error": checkpoint_error_text,
            "pixel_restore_error": str(pixel_error),
        }


def _stage5_quality_gate(
    pipeline,
    baseline_pixels: np.ndarray,
    candidate_pixels: np.ndarray,
) -> Dict[str, Any]:
    return assess_denoise_candidate(
        baseline_pixels,
        candidate_pixels,
        detail_retention_min=float(
            getattr(
                pipeline.cfg,
                "stage5_multiscale_detail_retention_min",
                0.82,
            )
        ),
        noise_reduction_min=float(
            getattr(
                pipeline.cfg,
                "stage5_multiscale_noise_reduction_min",
                0.05,
            )
        ),
        chroma_noise_growth_max=float(
            getattr(
                pipeline.cfg,
                "stage5_denoise_chroma_noise_growth_max",
                1.05,
            )
        ),
    )


def _run_external_denoise_candidate(
    pipeline,
    *,
    method: str,
    checkpoint_stem: str,
    baseline_stem: str,
    baseline_pixels: np.ndarray,
    runner,
    messages: List[str],
) -> tuple[Optional[str], Dict[str, Any], bool]:
    """从同一基线运行外部候选；拒绝或异常时必须恢复。"""
    report: Dict[str, Any] = {
        "schema": "starun.stage5-denoise-attempt.v1",
        "method": method,
        "status": "not_run",
        "accepted": False,
        "baseline": f"{baseline_stem}.fit",
        "candidate": f"{checkpoint_stem}.fit",
    }
    initial_restore = _stage5_restore_denoise_baseline(
        pipeline,
        baseline_stem=baseline_stem,
        baseline_pixels=baseline_pixels,
    )
    report["initial_baseline_restore"] = initial_restore
    if not bool(initial_restore.get("completed")):
        report.update(
            status="prohibited",
            issues=["baseline_restore_failed"],
        )
        messages.append(
            f"Stage5 {method} candidate prohibited: baseline restore failed"
        )
        return None, report, False

    used: Any = None
    try:
        used = runner()
        if not used:
            raise RuntimeError("candidate backend unavailable or command failed")
        candidate_pixels = _stage5_current_pixels(pipeline)
        gate = _stage5_quality_gate(
            pipeline,
            baseline_pixels,
            candidate_pixels,
        )
        report["quality_gate"] = gate
        report.update(
            status=str(gate.get("status") or "rejected"),
            accepted=bool(gate.get("accepted")),
            issues=list(gate.get("issues") or []),
        )
        if report["accepted"]:
            if not pipeline._save_stage_output(checkpoint_stem):
                report.update(
                    status="failed",
                    accepted=False,
                    issues=[*report["issues"], "candidate_checkpoint_save_failed"],
                )
                raise RuntimeError("candidate checkpoint save failed")
            report["transaction"] = {
                "candidate_saved": True,
                "rollback_required": False,
                "rollback_completed": False,
            }
            metrics = gate.get("metrics") or {}
            messages.append(
                f"Stage5 {method} denoise candidate accepted "
                f"(noise_reduction={float(metrics.get('background_noise_reduction', 0.0)):.3f}, "
                f"detail_retention={float(metrics.get('signal_detail_retention', 0.0)):.3f})"
            )
            return str(used), report, True

        rollback = _stage5_restore_denoise_baseline(
            pipeline,
            baseline_stem=baseline_stem,
            baseline_pixels=baseline_pixels,
        )
        report["transaction"] = {
            "candidate_saved": False,
            "rollback_required": True,
            "rollback_completed": bool(rollback.get("completed")),
            "rollback": rollback,
        }
        messages.append(
            f"Stage5 {method} denoise candidate rejected and rolled back: "
            f"issues={','.join(report['issues']) or 'none'}"
        )
        return None, report, bool(rollback.get("completed"))
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        rollback = _stage5_restore_denoise_baseline(
            pipeline,
            baseline_stem=baseline_stem,
            baseline_pixels=baseline_pixels,
        )
        report.update(
            status="failed",
            accepted=False,
            error=str(error),
        )
        report["transaction"] = {
            "candidate_saved": False,
            "rollback_required": True,
            "rollback_completed": bool(rollback.get("completed")),
            "rollback": rollback,
        }
        messages.append(
            f"Stage5 {method} denoise candidate failed and rolled back: "
            f"{pipeline._short_text(error, 160)}"
        )
        return None, report, bool(rollback.get("completed"))


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

    psf_report = stage5_deconvolution_quality.unavailable_psf_quality_report(
        "RL audit has not started",
        findstar_succeeded=False,
    )
    pipeline._stage5_psf_quality_report = psf_report
    pipeline._stage5_rl_star_catalog = []
    findstar_succeeded = False
    try:
        pipeline.cmd_with_check("load", fallback_stem)
        pipeline.log.info(
            "[Stage5] findstar "
            f"-out=stage5_psf_stars.csv -maxstars={maxstars}"
        )
        pipeline.cmd_with_check(
            "findstar",
            "-out=stage5_psf_stars.csv",
            f"-maxstars={maxstars}",
        )
        findstar_succeeded = True
        get_stars = getattr(pipeline.siril, "get_image_stars", None)
        if not callable(get_stars):
            psf_report = (
                stage5_deconvolution_quality.unavailable_psf_quality_report(
                    "sirilpy get_image_stars is unavailable",
                    findstar_succeeded=True,
                )
            )
            star_catalog: List[Dict[str, Any]] = []
        else:
            try:
                psf_report, star_catalog = (
                    stage5_deconvolution_quality.build_psf_quality_report(
                        get_stars(),
                        getattr(pipeline, "_stage5_input_linear_pixels", (0, 0)),
                        target_structure_mask=getattr(
                            pipeline,
                            "_stage5_target_structure_mask",
                            None,
                        ),
                        source_checkpoint="stage5_input_linear.fit",
                        catalog_role="rl_psf",
                        findstar_succeeded=True,
                    )
                )
            except (
                AttributeError,
                RuntimeError,
                SirilError,
                TypeError,
                ValueError,
            ) as error:
                psf_report = (
                    stage5_deconvolution_quality.unavailable_psf_quality_report(
                        f"get_image_stars failed: {error}",
                        findstar_succeeded=True,
                    )
                )
                star_catalog = []
        pipeline._stage5_psf_quality_report = psf_report
        pipeline._stage5_rl_star_catalog = star_catalog
        pipeline._write_stage_json("stage5_psf_quality.json", psf_report)
        if bool(psf_report.get("hard_skip_rl")):
            reason = str(psf_report.get("reason_code") or "invalid_psf_catalog")
            messages.append(
                "Stage5 Siril RL skipped by structural PSF audit: " + reason
            )
            pipeline.cmd_with_check("load", fallback_stem)
            return False

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
        if not findstar_succeeded:
            psf_report = (
                stage5_deconvolution_quality.unavailable_psf_quality_report(
                    f"findstar failed: {e}",
                    findstar_succeeded=False,
                )
            )
            psf_report.update(
                decision="skip_rl_command_failed",
                reason_code="findstar_failed",
            )
            pipeline._stage5_psf_quality_report = psf_report
            pipeline._write_stage_json("stage5_psf_quality.json", psf_report)
        pipeline.log.warn(f"[Stage5] RL deconvolution skipped: {e}")
        messages.append(f"Stage5 RL deconvolution skipped: {pipeline._short_text(e, 160)}")
        try:
            pipeline.cmd_with_check("load", fallback_stem)
        except (CommandError, SirilError) as load_error:
            pipeline.log.warn(f"[Stage5] reload {fallback_stem} failed after RL failure: {load_error}")
        return False


def _stage5_capture_star_reference(
    pipeline,
    messages: List[str],
    *,
    baseline_pixels: Optional[np.ndarray],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Freeze star coordinates before either Stage 5 deconvolution method."""
    if not bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)):
        return (
            {
                "schema": stage5_deconvolution_quality.STAR_REFERENCE_SCHEMA,
                "status": "not_run",
                "reason_code": "deconvolution_disabled",
                "source_checkpoint": "stage5_input_linear.fit",
                "role": "fixed_local_star_reference",
            },
            [],
        )
    if baseline_pixels is None:
        return (
            {
                "schema": stage5_deconvolution_quality.STAR_REFERENCE_SCHEMA,
                "status": "unavailable",
                "reason_code": "baseline_pixels_unavailable",
                "source_checkpoint": "stage5_input_linear.fit",
                "role": "fixed_local_star_reference",
            },
            [],
        )
    maxstars = max(
        20,
        min(1000, int(getattr(pipeline.cfg, "stage5_rl_maxstars", 200))),
    )
    findstar_succeeded = False
    try:
        pipeline.cmd_with_check("load", "stage5_input_linear")
        pipeline.cmd_with_check(
            "findstar",
            "-out=stage5_star_reference.csv",
            f"-maxstars={maxstars}",
        )
        findstar_succeeded = True
        get_stars = getattr(pipeline.siril, "get_image_stars", None)
        if not callable(get_stars):
            raise AttributeError("sirilpy get_image_stars is unavailable")
        report, catalog = stage5_deconvolution_quality.build_psf_quality_report(
            get_stars(),
            baseline_pixels,
            target_structure_mask=getattr(
                pipeline,
                "_stage5_target_structure_mask",
                None,
            ),
            source_checkpoint="stage5_input_linear.fit",
            catalog_role="fixed_local_star_reference",
            findstar_succeeded=True,
        )
        report = {
            **report,
            "schema": stage5_deconvolution_quality.STAR_REFERENCE_SCHEMA,
            "role": "fixed_local_star_reference",
            "participates_in_deconvolution_acceptance": False,
            "fixed_before_deconvolution": True,
        }
        messages.append(
            "Stage5 fixed star reference captured "
            f"(stars={int((report.get('counts') or {}).get('total', 0))})"
        )
        return report, catalog
    except (
        AttributeError,
        CommandError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        messages.append(
            "Stage5 fixed star reference unavailable; GraXpert remains eligible: "
            f"{pipeline._short_text(error, 160)}"
        )
        return (
            {
                "schema": stage5_deconvolution_quality.STAR_REFERENCE_SCHEMA,
                "status": "unavailable",
                "reason_code": "star_reference_api_unavailable",
                "reason": str(error),
                "findstar_succeeded": findstar_succeeded,
                "source_checkpoint": "stage5_input_linear.fit",
                "role": "fixed_local_star_reference",
                "participates_in_deconvolution_acceptance": False,
                "fixed_before_deconvolution": True,
            },
            [],
        )


def _stage5_graxpert_model_version(model: Path) -> Optional[tuple[int, int, int]]:
    match = _GRAXPERT_MODEL_VERSION_RE.fullmatch(model.parent.name)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _stage5_graxpert_models_from_path(path: Path) -> List[Path]:
    def nonempty_model(candidate: Path) -> bool:
        try:
            return (
                candidate.name == "model.onnx"
                and candidate.is_file()
                and candidate.stat().st_size > 0
            )
        except OSError:
            return False

    try:
        if path.is_file():
            # A configured file is already an explicit user choice.  GraXpert
            # receives it through an isolated ``model.onnx`` link, so its
            # source basename does not need to match the cache convention.
            return [path] if path.stat().st_size > 0 else []
        if not path.is_dir():
            return []
    except OSError:
        return []

    roots = [path]
    family_root = path / "deconvolution-object-ai-models"
    try:
        if family_root.is_dir():
            roots.insert(0, family_root)
    except OSError:
        pass
    candidates: List[Path] = []
    for root in roots:
        direct_model = root / "model.onnx"
        if nonempty_model(direct_model):
            candidates.append(direct_model)
        try:
            candidates.extend(
                model
                for model in root.glob("*/model.onnx")
                if nonempty_model(model)
            )
        except OSError:
            continue
    return list(dict.fromkeys(candidates))


def _stage5_latest_graxpert_model(candidates: List[Path]) -> Optional[Path]:
    versioned = [
        (version, model)
        for model in candidates
        if (version := _stage5_graxpert_model_version(model)) is not None
    ]
    return max(versioned, key=lambda item: item[0])[1] if versioned else None


def _stage5_model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage5_install_user_graxpert_model(source: Path, home: Path) -> tuple[Optional[Path], str]:
    version = _stage5_graxpert_model_version(source)
    try:
        model_name = (
            ".".join(str(part) for part in version)
            if version is not None
            else f"user-{_stage5_model_sha256(source)[:12]}"
        )
    except OSError as error:
        return None, f"model_hash_failed:{error}"
    target = (
        home
        / "Library"
        / "Application Support"
        / "GraXpert"
        / "deconvolution-object-ai-models"
        / model_name
        / "model.onnx"
    )
    try:
        if target.exists():
            if target.samefile(source):
                return target, ""
            if (
                target.stat().st_size == source.stat().st_size
                and _stage5_model_sha256(target) == _stage5_model_sha256(source)
            ):
                return target, ""
            return None, "model_version_conflicts_with_isolated_home"
        if target.is_symlink():
            return None, "isolated_home_model_symlink_is_broken"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
        return target, ""
    except OSError as error:
        return None, f"model_link_failed:{error}"


def _stage5_graxpert_object_model(cfg=None) -> tuple[Optional[Path], dict]:
    home = Path(os.environ.get("HOME") or Path.home())
    details = {
        "configured_path": "",
        "source": "",
        "resolved_model_path": "",
        "discovery_reason": "",
    }
    configured_path = str(
        getattr(cfg, "graxpert_object_model_path", "")
        or os.environ.get(GRAXPERT_OBJECT_MODEL_ENV, "")
    ).strip()
    if configured_path:
        configured = Path(configured_path).expanduser()
        if not configured.is_absolute():
            configured = Path.cwd() / configured
        details["configured_path"] = str(configured)
        configured_models = _stage5_graxpert_models_from_path(configured)
        user_model = (
            configured_models[0]
            if configured.is_file() and configured_models
            else _stage5_latest_graxpert_model(configured_models)
        )
        if user_model is not None:
            installed, reason = _stage5_install_user_graxpert_model(user_model, home)
            if installed is not None:
                details["source"] = "user_provided"
                return installed, details
            details["discovery_reason"] = reason
        elif configured_models:
            details["discovery_reason"] = "model_version_directory_must_be_semver"
        else:
            details["discovery_reason"] = "configured_model_not_found_or_invalid"
        return None, details

    model_roots = (
        home / "Library" / "Application Support" / "GraXpert"
        / "GraXpert" / "deconvolution-object-ai-models",
        home / "Library" / "Application Support" / "GraXpert"
        / "deconvolution-object-ai-models",
        home / ".local" / "share" / "GraXpert"
        / "deconvolution-object-ai-models",
        home / "AppData" / "Local" / "GraXpert"
        / "deconvolution-object-ai-models",
    )
    local_model = _stage5_latest_graxpert_model(
        [
            model
            for root in model_roots
            for model in _stage5_graxpert_models_from_path(root)
        ]
    )
    if local_model is not None:
        details["source"] = "isolated_home"
        return local_model, details
    return None, details


def _run_stage5_graxpert_deconvolution(
    pipeline,
    messages: List[str],
    *,
    strength_override: Optional[float] = None,
    checkpoint_stem: str = "stage5_graxpert_deconv",
) -> tuple[bool, dict]:
    configured_strength = float(
        getattr(pipeline.cfg, "stage5_graxpert_deconv_strength", 0.30)
    )
    details = {
        "attempted": False,
        "available": False,
        "model": None,
        "strength": max(
            0.20,
            min(
                0.40,
                configured_strength
                if strength_override is None
                else float(strength_override),
            ),
        ),
        "psf_size": 5.0,
        "reason": "",
        "configured_path": "",
        "source": "",
        "resolved_model_path": "",
        "checkpoint": checkpoint_stem,
        "strength_override": strength_override is not None,
    }
    if not bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)):
        details["reason"] = "deconvolution_disabled"
        return False, details
    if not bool(
        getattr(pipeline.cfg, "stage5_graxpert_deconvolution_enabled", True)
    ):
        details["reason"] = "graxpert_deconvolution_disabled"
        messages.append("Stage5 GraXpert object deconvolution disabled; using Siril RL")
        return False, details

    script = pipeline._find_plugin_script(("processing/GraXpert-AI.py",))
    if script is None:
        details["reason"] = "script_missing"
        return False, details
    model, discovery = _stage5_graxpert_object_model(pipeline.cfg)
    details.update(discovery)
    if model is None:
        details["reason"] = (
            discovery.get("discovery_reason")
            or "object_deconvolution_model_missing"
        )
        if discovery.get("configured_path"):
            messages.append(
                "Stage5 GraXpert user model unavailable: "
                f"{details['reason']} ({discovery['configured_path']})"
            )
        else:
            messages.append("Stage5 GraXpert object deconvolution unavailable: local model missing")
        return False, details

    hardware_acceleration = _stage5_graxpert_hardware_acceleration_enabled(pipeline)
    details.update({
        "attempted": True,
        "available": True,
        "model": model.parent.name,
        "resolved_model_path": str(model),
        "hardware_acceleration": "auto" if hardware_acceleration else "cpu",
    })
    script_args = [
        "-deconv_obj",
        "-strength", f"{details['strength']:.2f}",
        "-psfsize", f"{details['psf_size']:.1f}",
        "-model", model.parent.name,
    ]
    if hardware_acceleration:
        script_args.append("-gpu")
    else:
        script_args.append("-nogpu")
    used = pipeline._run_plugin_script_by_path(
        "Stage5 GraXpert反卷积",
        "GraXpert AI Object Deconvolution",
        script,
        args=tuple(script_args),
    )
    if not used:
        details["reason"] = (
            getattr(pipeline, "_last_plugin_script_error", None)
            or "plugin_failed_or_noop"
        )
        messages.append(
            "Stage5 GraXpert object deconvolution failed; falling back to Siril RL: "
            f"{pipeline._short_text(details['reason'], 160)}"
        )
        try:
            pipeline.cmd_with_check("load", "stage5_input_linear")
        except (CommandError, SirilError) as load_error:
            pipeline.log.warn(
                f"[Stage5] reload baseline after GraXpert failure failed: {load_error}"
            )
        return False, details

    pipeline._save_stage_output(checkpoint_stem)
    messages.append(
        "Stage5 GraXpert object deconvolution applied "
        f"(model={model.parent.name}, strength={details['strength']:.2f}, psf=5.0, "
        f"hardware={'auto' if hardware_acceleration else 'cpu'})"
    )
    return True, details


def run_stage5_linear_denoise(pipeline) -> None:
    """
    阶段 5: 线性整理
    - 输入固定为 stage4_color，优先 GraXpert Object Deconvolution，失败回退 RL。
    - 反卷积后再做轻量线性降噪，最终保存 stage5_linear。
    - 不再默认执行全局锐化，避免在线性暗背景中放大彩噪和星环。
    """
    stage_name = PipelineStage.LINEAR_DENOISE.label
    pipeline._clear_stage_reviews(5)
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
    background_guard_triggered = False
    background_guard_reason = ""
    local_guard_triggered = False
    local_guard_reason = ""
    deconv_integrity_ok = True
    denoise_used = "none"
    denoise_reason_code = ""
    denoise_fallback_used = False
    deconv_applied = False
    deconv_method = "none"
    deconv_attempted_method = "none"
    deconv_fallback_used = False
    deconv_rollback_used = False
    graxpert_details = {}
    graxpert_attempts: List[Dict[str, Any]] = []
    stage5_input_linear_pixels: Optional[np.ndarray] = None
    target_structure_report: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "stage5 input pixels unavailable",
    }
    star_reference_report: Dict[str, Any] = {
        "schema": stage5_deconvolution_quality.STAR_REFERENCE_SCHEMA,
        "status": "unavailable",
        "reason_code": "not_captured",
        "source_checkpoint": "stage5_input_linear.fit",
        "role": "fixed_local_star_reference",
    }
    star_reference_catalog: List[Dict[str, Any]] = []
    psf_quality_report = stage5_deconvolution_quality.not_run_psf_quality_report()
    local_star_guard = (
        stage5_deconvolution_quality.not_run_local_star_guard_report()
    )
    local_star_guard_attempts: List[Dict[str, Any]] = []
    multiscale_report: Dict[str, Any] = {
        "status": "not_requested",
        "accepted": False,
    }
    denoise_attempts: List[Dict[str, Any]] = []
    denoise_baseline_transaction: Dict[str, Any] = {
        "status": "not_requested",
        "checkpoint": "stage5_pre_denoise.fit",
    }
    denoise_integrity_ok = True
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
    try:
        stage5_input_linear_pixels = _stage5_current_pixels(pipeline)
        pipeline._stage5_input_linear_pixels = stage5_input_linear_pixels
        noise_model_report = build_noise_model_report(
            stage5_input_linear_pixels,
            source_checkpoint="stage5_input_linear.fit",
            channel_semantics=str(
                getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
            ),
        )
        pipeline._stage5_noise_model_report = noise_model_report
        pipeline._write_stage_json(
            "stage5_noise_model.json",
            noise_model_report,
        )
        messages.append(
            "stage5_noise_model=report_only "
            f"scales={len(noise_model_report['scales'])} "
            f"background_samples={int(noise_model_report['background']['sample_count'])}"
        )
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        noise_model_report = {
            "schema": "starun.multiscale-noise-model.v1",
            "mode": "report_only",
            "applied_to_pixels": False,
            "status": "unavailable",
            "error": str(error),
        }
        pipeline._stage5_noise_model_report = noise_model_report
        pipeline.log.warn(f"Stage5 noise model report unavailable: {error}")
        messages.append("stage5 noise model unavailable; existing denoise policy unchanged")
    background_risk = _stage5_background_risk(before_adaptive, stage5_policy)

    if stage5_input_linear_pixels is not None:
        try:
            target_masks, target_structure_report = (
                stage8_pixels.build_signal_excluded_background_masks(
                    pipeline,
                    stage5_input_linear_pixels,
                )
            )
            pipeline._stage5_target_structure_mask = (
                stage5_deconvolution_quality.build_target_structure_mask(
                    target_masks,
                    stage5_input_linear_pixels,
                )
            )
            target_structure_report = {
                **target_structure_report,
                "union_threshold": (
                    stage5_deconvolution_quality.TARGET_STRUCTURE_MASK_THRESHOLD
                ),
                "union_available": (
                    pipeline._stage5_target_structure_mask is not None
                ),
            }
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            FloatingPointError,
        ) as error:
            pipeline._stage5_target_structure_mask = None
            target_structure_report = {
                "status": "unavailable",
                "reason": str(error),
                "union_threshold": (
                    stage5_deconvolution_quality.TARGET_STRUCTURE_MASK_THRESHOLD
                ),
            }
    else:
        pipeline._stage5_target_structure_mask = None

    star_reference_report, star_reference_catalog = (
        _stage5_capture_star_reference(
            pipeline,
            messages,
            baseline_pixels=stage5_input_linear_pixels,
        )
    )
    pipeline._stage5_star_reference_report = star_reference_report
    pipeline._stage5_star_reference_catalog = star_reference_catalog
    pipeline._stage5_psf_quality_report = psf_quality_report

    processing_mode = str(
        getattr(pipeline.cfg, "stage5_processing_mode", "auto")
    )
    failure_action = str(
        getattr(pipeline.cfg, "stage5_failure_action", "auto_fallback")
    )
    denoise_backend_policy = str(
        getattr(pipeline.cfg, "stage5_denoise_backend_policy", "auto_chain")
    )
    if processing_mode == "preserve":
        try:
            pipeline.cmd_with_check("load", "stage5_input_linear")
        except (CommandError, SirilError) as error:
            status = "degraded"
            messages.append(
                "Stage5 preserve reload failed; current immutable-equivalent "
                f"linear image retained: {pipeline._short_text(error, 160)}"
            )
        linear_saved = pipeline._save_stage_output("stage5_linear")
        linear_export_ok = pipeline._export_linear_intermediate()
        if not linear_saved or not linear_export_ok:
            status = "degraded"
        pipeline._stage5_denoise_applied = False
        messages.append(
            "用户选择保留 Stage4 线性结果；反卷积与降噪候选未执行"
        )
        preserve_deconvolution_component = {
            "status": "skipped",
            "reason_code": "user_preserve",
            "fallback_used": False,
        }
        pipeline._stage5_deconvolution_acceptance = {
            "schema": "starun.stage5-deconvolution-acceptance.v1",
            "accepted": False,
            "method": "none",
            "accepted_checkpoint": None,
            "component": dict(preserve_deconvolution_component),
            "local_star_guard": local_star_guard,
            "attempts": [],
            "integrity_ok": True,
        }
        report = {
            "stage": "stage5_linear",
            "processing_mode": processing_mode,
            "failure_action": failure_action,
            "denoise_backend_policy": denoise_backend_policy,
            "execution": "safe_passthrough",
            "reason_code": "user_preserve",
            "input": "stage5_input_linear",
            "linear_output": "stage5_linear" if linear_saved else None,
            "noise_model_report": noise_model_report,
            "target_structure_mask": target_structure_report,
            "star_reference": star_reference_report,
            "deconvolution": {
                **preserve_deconvolution_component,
                "local_star_guard": local_star_guard,
                "local_star_guard_attempts": [],
                "integrity_ok": True,
            },
            "denoise": {"status": "skipped", "reason_code": "user_preserve"},
            "status": status,
            "messages": messages,
        }
        pipeline._write_stage_json("stage5_linear_report.json", report)
        elapsed = pipeline.log.stage_end(stage_name)
        pipeline._record_stage(
            stage_name,
            status,
            elapsed,
            "；".join(messages),
            execution="safe_passthrough",
            upstream_passthrough=True,
            reason_code="user_preserve",
            details={
                "output": "stage5_linear" if linear_saved else None,
                "diagnostics_complete": True,
            },
            components={
                "deconvolution": preserve_deconvolution_component,
                "denoise": {
                    "status": "skipped",
                    "reason_code": "user_preserve",
                    "fallback_used": False,
                },
            },
        )
        return

    def assess_deconvolution_candidate(
        method: str,
        *,
        attempt: str,
    ) -> Dict[str, Any]:
        try:
            if stage5_input_linear_pixels is None:
                raise RuntimeError(
                    "immutable stage5_input_linear pixels unavailable"
                )
            candidate_pixels = _stage5_current_pixels(pipeline)
            local_catalog = (
                list(getattr(pipeline, "_stage5_rl_star_catalog", []) or [])
                if method == "siril_rl"
                else star_reference_catalog
            )
            report = stage5_deconvolution_quality.assess_local_star_guard(
                stage5_input_linear_pixels,
                candidate_pixels,
                local_catalog,
                method=method,
            )
        except (
            AttributeError,
            CommandError,
            RuntimeError,
            SirilError,
            TypeError,
            ValueError,
        ) as error:
            report = (
                stage5_deconvolution_quality.unavailable_local_star_guard_report(
                    str(error),
                    method=method,
                )
            )
        report = {**report, "attempt": attempt}
        local_star_guard_attempts.append(report)
        return report

    def restore_deconvolution_baseline(reason: str) -> Dict[str, Any]:
        restored = _stage5_restore_denoise_baseline(
            pipeline,
            baseline_stem="stage5_input_linear",
            baseline_pixels=stage5_input_linear_pixels,
            rollback_label="Stage5 deconvolution baseline rollback",
        )
        restored["trigger"] = reason
        return restored

    primary_graxpert_applied, primary_graxpert_details = (
        _run_stage5_graxpert_deconvolution(
            pipeline,
            messages,
        )
    )
    graxpert_attempts.append(dict(primary_graxpert_details))
    graxpert_details = dict(primary_graxpert_details)
    deconv_applied = primary_graxpert_applied
    if deconv_applied:
        deconv_method = "graxpert_object"
        deconv_attempted_method = "graxpert_object"
        denoise_input = str(
            primary_graxpert_details.get("checkpoint")
            or "stage5_graxpert_deconv"
        )
        local_star_guard = assess_deconvolution_candidate(
            deconv_method,
            attempt="graxpert_primary",
        )
        if not bool(local_star_guard.get("accepted", False)):
            local_guard_triggered = True
            local_guard_reason = str(
                local_star_guard.get("reason_code")
                or "local_star_guard_rejected"
            )
            rollback = restore_deconvolution_baseline(local_guard_reason)
            local_star_guard["rollback"] = rollback
            deconv_rollback_used = True
            deconv_integrity_ok = bool(rollback.get("completed", False))
            deconv_applied = False
            deconv_method = "none"
            denoise_input = "stage5_input_linear"
            messages.append(
                "Stage5 enforced local-star guard rolled back GraXpert "
                f"primary candidate ({local_guard_reason})"
            )

            retry_strength = max(
                0.20,
                min(
                    float(
                        getattr(
                            pipeline.cfg,
                            "stage5_graxpert_guard_retry_strength",
                            0.25,
                        )
                    ),
                    float(primary_graxpert_details.get("strength", 0.30))
                    - 0.05,
                ),
            )
            retry_eligible = bool(
                deconv_integrity_ok
                and failure_action == "auto_fallback"
                and local_star_guard.get("status") == "available"
                and retry_strength
                < float(primary_graxpert_details.get("strength", 0.30)) - 1e-9
            )
            if retry_eligible:
                retry_applied, retry_details = (
                    _run_stage5_graxpert_deconvolution(
                        pipeline,
                        messages,
                        strength_override=retry_strength,
                        checkpoint_stem=(
                            "stage5_graxpert_deconv_guard_retry"
                        ),
                    )
                )
                retry_details["guard_retry"] = True
                graxpert_attempts.append(dict(retry_details))
                if retry_applied:
                    retry_guard = assess_deconvolution_candidate(
                        "graxpert_object",
                        attempt="graxpert_guard_retry",
                    )
                    local_star_guard = retry_guard
                    if bool(retry_guard.get("accepted", False)):
                        deconv_applied = True
                        deconv_method = "graxpert_object"
                        denoise_input = str(
                            retry_details.get("checkpoint")
                            or "stage5_graxpert_deconv_guard_retry"
                        )
                        deconv_fallback_used = True
                        messages.append(
                            "Stage5 lower-strength GraXpert retry passed "
                            "the enforced local-star guard"
                        )
                    else:
                        local_guard_reason = str(
                            retry_guard.get("reason_code")
                            or "local_star_guard_rejected"
                        )
                        retry_rollback = restore_deconvolution_baseline(
                            local_guard_reason
                        )
                        retry_guard["rollback"] = retry_rollback
                        deconv_integrity_ok = bool(
                            retry_rollback.get("completed", False)
                        )
                        messages.append(
                            "Stage5 enforced local-star guard rejected the "
                            "lower-strength GraXpert retry"
                        )
                else:
                    retry_rollback = restore_deconvolution_baseline(
                        "graxpert_guard_retry_execution_failed"
                    )
                    deconv_integrity_ok = bool(
                        retry_rollback.get("completed", False)
                    )
    else:
        if bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)):
            deconv_attempted_method = "siril_rl"
        deconv_applied = _run_stage5_rl_deconvolution(
            pipeline,
            messages,
            fallback_stem="stage5_input_linear",
        )
        psf_quality_report = dict(
            getattr(
                pipeline,
                "_stage5_psf_quality_report",
                psf_quality_report,
            )
            or psf_quality_report
        )
        if deconv_applied:
            deconv_method = "siril_rl"
            denoise_input = "stage5_deconv"
            deconv_fallback_used = bool(
                getattr(
                    pipeline.cfg,
                    "stage5_graxpert_deconvolution_enabled",
                    True,
                )
                and primary_graxpert_details.get("reason")
                not in {
                    "deconvolution_disabled",
                    "graxpert_deconvolution_disabled",
                }
            )
            local_star_guard = assess_deconvolution_candidate(
                deconv_method,
                attempt="siril_rl",
            )
            if not bool(local_star_guard.get("accepted", False)):
                local_guard_triggered = True
                local_guard_reason = str(
                    local_star_guard.get("reason_code")
                    or "local_star_guard_rejected"
                )
                rollback = restore_deconvolution_baseline(local_guard_reason)
                local_star_guard["rollback"] = rollback
                deconv_rollback_used = True
                deconv_integrity_ok = bool(rollback.get("completed", False))
                deconv_applied = False
                deconv_method = "none"
                denoise_input = "stage5_input_linear"
                messages.append(
                    "Stage5 enforced local-star guard rolled back Siril RL "
                    f"candidate ({local_guard_reason})"
                )

    graxpert_details["attempts"] = graxpert_attempts
    graxpert_details["accepted_checkpoint"] = (
        denoise_input if deconv_method == "graxpert_object" else None
    )
    graxpert_details["accepted_strength"] = (
        next(
            (
                float(attempt.get("strength"))
                for attempt in reversed(graxpert_attempts)
                if str(attempt.get("checkpoint") or "") == denoise_input
            ),
            None,
        )
        if deconv_method == "graxpert_object"
        else None
    )

    if deconv_applied:
        after_deconv_adaptive = (
            pipeline._adaptive_features_current()
            if hasattr(pipeline, "_adaptive_features_current")
            else {}
        )
        background_guard_triggered, background_guard_reason = (
            _stage5_background_worsened(
                before_adaptive,
                after_deconv_adaptive,
                pipeline,
            )
        )
        if background_guard_triggered:
            pipeline.log.warn(
                "[Stage5] background guard dropped "
                f"{deconv_method} result: {background_guard_reason}"
            )
            messages.append(
                "Stage5 background guard dropped "
                f"{deconv_method} result: {background_guard_reason}"
            )
            rollback = restore_deconvolution_baseline(
                "background_guard_rejected"
            )
            deconv_integrity_ok = bool(rollback.get("completed", False))
            denoise_input = "stage5_input_linear"
            deconv_applied = False
            deconv_method = "none"
            deconv_rollback_used = True
            if not deconv_integrity_ok:
                status = "degraded"
                messages.append(
                    "Stage5 background guard rollback failed: "
                    f"{pipeline._short_text(rollback, 160)}"
                )

    policy_abort_reason = ""
    if (
        (local_guard_triggered and not deconv_applied)
        or background_guard_triggered
    ) and failure_action != "auto_fallback":
        if local_guard_triggered and not deconv_applied:
            policy_abort_reason = (
                "deconvolution local-star gate rejected candidate: "
                + local_guard_reason
            )
            pipeline._require_review(
                5,
                "deconvolution_local_star_gate_rejected",
            )
        else:
            policy_abort_reason = (
                "deconvolution background gate rejected candidate: "
                + background_guard_reason
            )
            pipeline._require_review(
                5,
                "deconvolution_background_gate_rejected",
            )
        messages.append(
            "Stage5 candidate search stopped by failure policy after "
            "deconvolution rollback"
        )

    if not deconv_integrity_ok and not policy_abort_reason:
        policy_abort_reason = "deconvolution_rollback_failed"
        pipeline._require_review(5, "deconvolution_rollback_failed")

    if not deconv_applied and pipeline.process_dir:
        try:
            for stale_name in (
                "stage5_deconv.fit",
                "stage5_graxpert_deconv.fit",
                "stage5_graxpert_deconv_guard_retry.fit",
            ):
                (pipeline.process_dir / stale_name).unlink(missing_ok=True)
        except OSError as e:
            pipeline.log.warn(f"[Stage5] stale deconvolution output cleanup failed: {e}")

    if policy_abort_reason:
        denoise_reason_code = "failure_policy_candidate_abort"
    elif not bool(getattr(pipeline.cfg, "denoise_enabled", False)):
        denoise_reason_code = _stage5_disabled_denoise_reason(pipeline)
        denoise_reason_text = {
            "user_disabled": "linear denoise disabled by user",
            "config_disabled": "linear denoise skipped by configuration",
        }[denoise_reason_code]
        messages.append(denoise_reason_text)
    else:
        baseline_stem = "stage5_pre_denoise"
        baseline_saved = pipeline._save_stage_output(baseline_stem)
        baseline_pixels: Optional[np.ndarray] = None
        baseline_error = ""
        if baseline_saved:
            try:
                baseline_pixels = _stage5_current_pixels(pipeline)
            except (
                AttributeError,
                CommandError,
                OSError,
                RuntimeError,
                SirilError,
                TypeError,
                ValueError,
            ) as error:
                baseline_error = str(error)
        else:
            baseline_error = "immutable baseline checkpoint save failed"

        if baseline_pixels is None:
            denoise_baseline_transaction.update(
                status="prohibited",
                checkpoint_saved=bool(baseline_saved),
                frozen_pixels_available=False,
                error=baseline_error,
            )
            denoise_reason_code = "immutable_baseline_unavailable"
            status = "degraded"
            messages.append(
                "Stage5 denoise prohibited: immutable checkpoint and frozen "
                "pixel baseline are both required"
            )
        else:
            denoise_baseline_transaction.update(
                status="ready",
                checkpoint_saved=True,
                frozen_pixels_available=True,
                shape=[int(value) for value in baseline_pixels.shape],
            )
            baseline_probe = _stage5_quality_gate(
                pipeline,
                baseline_pixels,
                baseline_pixels,
            )
            denoise_baseline_transaction["low_noise_probe"] = baseline_probe
            low_noise_input = bool(
                baseline_probe.get("status") == "skipped_low_noise"
                and getattr(
                    pipeline.cfg,
                    "stage5_low_noise_auto_skip_enabled",
                    True,
                )
            )
            multiscale_enabled = bool(
                getattr(
                    pipeline.cfg,
                    "stage5_multiscale_denoise_enabled",
                    True,
                )
                and denoise_backend_policy
                in {"auto_chain", "multiscale_only"}
            )

            if low_noise_input:
                denoise_reason_code = "auto_low_noise"
                messages.append(
                    "Stage5 low-noise guard skipped all denoise candidates"
                )
            elif multiscale_enabled:
                multiscale_applied, multiscale_report = (
                    _run_multiscale_linear_denoise(
                        pipeline,
                        messages,
                        baseline_pixels=baseline_pixels,
                    )
                )
                denoise_attempts.append(multiscale_report)
                pipeline._write_stage_json(
                    "stage5_multiscale_denoise.json",
                    multiscale_report,
                )
                if multiscale_applied:
                    denoise_used = "deterministic_multiscale"
                    denoise_reason_code = "accepted"
                elif bool(
                    (multiscale_report.get("transaction") or {}).get(
                        "pixels_mutated"
                    )
                ):
                    rollback = _stage5_restore_denoise_baseline(
                        pipeline,
                        baseline_stem=baseline_stem,
                        baseline_pixels=baseline_pixels,
                    )
                    multiscale_report.setdefault("transaction", {}).update(
                        rollback_required=True,
                        rollback_completed=bool(rollback.get("completed")),
                        rollback=rollback,
                    )
                    denoise_integrity_ok = bool(rollback.get("completed"))

            if (
                denoise_used == "none"
                and denoise_integrity_ok
                and not low_noise_input
                and denoise_backend_policy in {"auto_chain", "siril_only"}
            ):
                builtin_used, builtin_report, denoise_integrity_ok = (
                    _run_external_denoise_candidate(
                        pipeline,
                        method="siril_builtin",
                        checkpoint_stem="stage5_siril_denoise_candidate",
                        baseline_stem=baseline_stem,
                        baseline_pixels=baseline_pixels,
                        runner=lambda: _run_builtin_linear_denoise(
                            pipeline,
                            messages,
                        ),
                        messages=messages,
                    )
                )
                denoise_attempts.append(builtin_report)
                if builtin_used:
                    denoise_used = "siril_builtin"
                    denoise_reason_code = (
                        "primary_rejected_or_failed"
                        if multiscale_enabled
                        else "accepted"
                    )
                    denoise_fallback_used = multiscale_enabled

            if (
                denoise_used == "none"
                and denoise_integrity_ok
                and not low_noise_input
                and denoise_backend_policy
                in {"auto_chain", "cosmic_clarity_only"}
            ):
                plugin_used, plugin_report, denoise_integrity_ok = (
                    _run_external_denoise_candidate(
                        pipeline,
                        method="cosmic_clarity",
                        checkpoint_stem="stage5_cosmic_clarity_denoise_candidate",
                        baseline_stem=baseline_stem,
                        baseline_pixels=baseline_pixels,
                        runner=lambda: _run_cosmic_clarity_linear_denoise(
                            pipeline,
                            denoise_mode=denoise_mode,
                            denoise_strength=denoise_strength,
                            messages=messages,
                        ),
                        messages=messages,
                    )
                )
                denoise_attempts.append(plugin_report)
                if plugin_used:
                    denoise_used = plugin_used
                    denoise_reason_code = "primary_rejected_or_failed"
                    denoise_fallback_used = True

            if not denoise_integrity_ok:
                status = "degraded"
                denoise_reason_code = "denoise_rollback_failed"
                messages.append(
                    "Stage5 denoise integrity failure: rollback failed; "
                    "no further denoise candidates were executed"
                )
            elif denoise_used == "none" and not low_noise_input:
                status = "degraded"
                rejected = any(
                    attempt.get("status") == "rejected"
                    for attempt in denoise_attempts
                )
                denoise_reason_code = (
                    "all_candidates_rejected_safe_passthrough"
                    if rejected
                    else "all_denoisers_failed_safe_passthrough"
                )
                messages.append(
                    "Stage5 kept the immutable pre-denoise baseline because "
                    "no denoise candidate passed the common quality gate"
                )

    if (
        failure_action != "auto_fallback"
        and not policy_abort_reason
        and denoise_reason_code
        in {
            "denoise_rollback_failed",
            "all_candidates_rejected_safe_passthrough",
            "all_denoisers_failed_safe_passthrough",
        }
    ):
        policy_abort_reason = denoise_reason_code
        pipeline._require_review(5, str(denoise_reason_code))

    if policy_abort_reason:
        policy_restore_failed = False
        try:
            pipeline.cmd_with_check("load", "stage5_input_linear")
            denoise_input = "stage5_input_linear"
            deconv_applied = False
            deconv_method = "none"
            deconv_rollback_used = True
        except (CommandError, SirilError) as error:
            policy_restore_failed = True
            messages.append(
                "Stage5 failure-policy baseline restore failed: "
                f"{pipeline._short_text(error, 160)}"
            )
        if hasattr(pipeline, "_record_stage_policy_event"):
            pipeline._record_stage_policy_event(
                5,
                event="candidate_search_stopped",
                reason=policy_abort_reason,
                source="candidate_gate",
            )
        status = (
            "failed"
            if failure_action == "stop" or policy_restore_failed
            else "degraded"
        )

    linear_saved = pipeline._save_stage_output("stage5_linear")
    pipeline._stage5_denoise_applied = denoise_used != "none"
    if not linear_saved:
        if status != "failed":
            status = "degraded"
        messages.append("stage5_linear save failed")
    else:
        diff_note = pipeline._stage_diff_note("stage5_linear", denoise_input)
        if diff_note:
            messages.append(diff_note)
        if hasattr(pipeline, "_create_stage_review_bundle"):
            review = pipeline._create_stage_review_bundle(
                "stage5_linear_cleanup",
                "stage5_input_linear",
                "stage5_linear",
                context={
                    "denoise_method": denoise_used,
                    "deconvolution_applied": deconv_applied,
                },
            )
            if review.get("report_path"):
                messages.append(f"review_bundle={review['report_path']}")

    linear_export_ok = pipeline._export_linear_intermediate()
    if not linear_export_ok:
        if status != "failed":
            status = "degraded"
        messages.append("导出 result_linear.fit 失败")

    after_linear_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )

    if deconv_applied:
        deconv_component_status = "applied"
        deconv_reason_code = "accepted"
    elif background_guard_triggered:
        deconv_component_status = "rolled_back"
        deconv_reason_code = "background_guard_rollback"
    elif local_guard_triggered:
        deconv_component_status = (
            "rolled_back" if deconv_integrity_ok else "failed"
        )
        deconv_reason_code = (
            "local_star_guard_rollback"
            if deconv_integrity_ok
            else "deconvolution_rollback_failed"
        )
    elif not bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)):
        deconv_component_status = "skipped"
        deconv_reason_code = (
            "user_disabled"
            if str(os.getenv("STARUN_STAGE5_DECONV_ENABLE") or "").strip().lower()
            in {"0", "false", "no", "off"}
            else "config_disabled"
        )
    elif bool(psf_quality_report.get("hard_skip_rl")):
        deconv_component_status = "skipped"
        deconv_reason_code = "psf_structurally_invalid"
    else:
        deconv_component_status = "failed"
        deconv_reason_code = "deconvolution_unavailable"

    if denoise_used != "none":
        denoise_component_status = "applied"
        denoise_reason_code = denoise_reason_code or "accepted"
    elif denoise_reason_code in {
        "auto_low_noise",
        "user_disabled",
        "config_disabled",
    }:
        denoise_component_status = "skipped"
    elif denoise_reason_code == "immutable_baseline_unavailable":
        denoise_component_status = "prohibited"
    elif denoise_reason_code in {
        "all_candidates_rejected_safe_passthrough",
        "all_denoisers_failed_safe_passthrough",
    }:
        denoise_component_status = "rolled_back"
    else:
        denoise_component_status = "failed"
        denoise_reason_code = denoise_reason_code or "all_denoisers_failed"

    components = {
        "deconvolution": {
            "status": deconv_component_status,
            "method": deconv_method,
            "reason_code": deconv_reason_code,
            "input": "stage5_input_linear",
            "output": denoise_input if deconv_applied else None,
            "fallback_used": deconv_fallback_used or deconv_rollback_used,
        },
        "denoise": {
            "status": denoise_component_status,
            "method": denoise_used,
            "reason_code": denoise_reason_code,
            "input": denoise_input,
            "output": "stage5_linear" if linear_saved else None,
            "fallback_used": denoise_fallback_used,
        },
    }
    pipeline._stage5_deconvolution_acceptance = {
        "schema": "starun.stage5-deconvolution-acceptance.v1",
        "accepted": bool(deconv_applied),
        "method": deconv_method,
        "accepted_checkpoint": denoise_input if deconv_applied else None,
        "component": dict(components["deconvolution"]),
        "local_star_guard": local_star_guard,
        "attempts": local_star_guard_attempts,
        "integrity_ok": bool(deconv_integrity_ok),
    }
    stage_fallback_used = (
        deconv_fallback_used
        or deconv_rollback_used
        or denoise_fallback_used
    )
    if policy_abort_reason:
        status = "failed" if failure_action == "stop" else "degraded"
        stage_reason_code = (
            "failure_policy_stop"
            if failure_action == "stop"
            else "failure_policy_preserve_review"
        )
    elif denoise_reason_code in {
        "immutable_baseline_unavailable",
        "denoise_rollback_failed",
        "all_candidates_rejected_safe_passthrough",
        "all_denoisers_failed_safe_passthrough",
    }:
        status = "degraded"
        stage_reason_code = denoise_reason_code
    elif deconv_component_status == "failed":
        status = "degraded"
        stage_reason_code = "deconvolution_unavailable"
        messages.append(
            "Stage5 degraded: all enabled deconvolution methods were unavailable"
        )
    elif (
        deconv_component_status == "rolled_back"
        and denoise_component_status != "applied"
    ):
        status = "degraded"
        stage_reason_code = "deconvolution_rollback_without_denoise"
        messages.append(
            "Stage5 degraded: deconvolution was rolled back and denoise was not applied"
        )
    elif stage_fallback_used:
        stage_reason_code = "component_fallback_used"
    else:
        stage_reason_code = ""

    pipeline._write_stage_json(
        "stage5_denoise_attempts.json",
        {
            "schema": "starun.stage5-denoise-transaction.v1",
            "baseline": denoise_baseline_transaction,
            "attempts": denoise_attempts,
            "accepted_method": denoise_used,
            "reason_code": denoise_reason_code,
            "integrity_ok": denoise_integrity_ok,
        },
    )
    pipeline._write_stage_json(
        "stage5_linear_report.json",
        {
            "stage": "stage5_linear",
            "processing_mode": processing_mode,
            "failure_action": failure_action,
            "failure_policy_triggered": bool(policy_abort_reason),
            "failure_policy_reason": policy_abort_reason or None,
            "denoise_backend_policy": denoise_backend_policy,
            "input": "stage4_color",
            "linear_output": "stage5_linear",
            "final_linear_source": final_stem,
            "processing_order": [
                "load stage4_color",
                "save stage5_input_linear",
                "GraXpert object deconvolution when local model is available",
                "findstar/makepsf/rl fallback if enabled",
                "freeze immutable stage5_pre_denoise baseline",
                "evaluate every denoise candidate from the same baseline",
                "accept through common quality gate or rollback",
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
                "status": denoise_component_status,
                "reason_code": denoise_reason_code,
                "fallback_used": denoise_fallback_used,
                "method": denoise_used,
                "input": denoise_input,
                "output": "stage5_linear",
                "noise_model_report": noise_model_report,
                "multiscale_candidate": multiscale_report,
                "baseline_transaction": denoise_baseline_transaction,
                "candidate_attempts": denoise_attempts,
                "integrity_ok": denoise_integrity_ok,
                "quality_gate": {
                    "detail_retention_min": float(
                        getattr(
                            pipeline.cfg,
                            "stage5_multiscale_detail_retention_min",
                            0.82,
                        )
                    ),
                    "noise_reduction_min": float(
                        getattr(
                            pipeline.cfg,
                            "stage5_multiscale_noise_reduction_min",
                            0.05,
                        )
                    ),
                    "chroma_noise_growth_max": float(
                        getattr(
                            pipeline.cfg,
                            "stage5_denoise_chroma_noise_growth_max",
                            1.05,
                        )
                    ),
                },
                "siril_builtin_mod": _stage5_builtin_denoise_mod(pipeline),
                "siril_builtin_mod_source": "denoise_mod clamped by denoise_safety_max",
                "siril_independent_channels": True,
                "cosmic_clarity_fallback_mode": denoise_mode,
                "cosmic_clarity_fallback_strength": denoise_strength,
            },
            "deconvolution": {
                "status": deconv_component_status,
                "reason_code": deconv_reason_code,
                "fallback_used": deconv_fallback_used or deconv_rollback_used,
                "enabled": bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)),
                "method": deconv_method,
                "attempted_method": deconv_attempted_method,
                "applied": deconv_applied,
                "output": denoise_input if deconv_applied else None,
                "runs_before_denoise": True,
                "graxpert": {
                    **graxpert_details,
                    "accepted": deconv_applied and deconv_method == "graxpert_object",
                },
                "fallback": "siril_rl" if deconv_method == "siril_rl" else None,
                "star_reference": star_reference_report,
                "psf_quality": psf_quality_report,
                "target_structure_mask": target_structure_report,
                "local_star_guard": local_star_guard,
                "local_star_guard_attempts": local_star_guard_attempts,
                "integrity_ok": deconv_integrity_ok,
            },
            "background_guard": {
                "risk": background_risk,
                "rollback": background_guard_triggered and not deconv_applied,
                "reason": background_guard_reason,
                "before": before_adaptive,
                "after_deconvolution": after_deconv_adaptive,
                "after_linear": after_linear_adaptive,
                "after_final": after_linear_adaptive,
                "thresholds": {
                    "bg_std_growth_max": float(
                        getattr(
                            pipeline.cfg,
                            "stage5_deconv_bg_std_growth_max",
                            1.38,
                        )
                    ),
                    "chroma_growth_max": float(
                        getattr(
                            pipeline.cfg,
                            "stage5_deconv_chroma_growth_max",
                            1.15,
                        )
                    ),
                    "chroma_ratio_growth_max": float(
                        getattr(
                            pipeline.cfg,
                            "stage5_deconv_chroma_ratio_growth_max",
                            1.35,
                        )
                    ),
                    "dirty_delta_max": float(
                        getattr(
                            pipeline.cfg,
                            "stage5_deconv_dirty_delta_max",
                            0.06,
                        )
                    ),
                },
            },
            "components": components,
            "status": status,
            "reason_code": stage_reason_code,
            "messages": messages,
        },
    )

    elapsed = pipeline.log.stage_end(stage_name)
    pipeline._record_stage(
        stage_name,
        status,
        elapsed,
        "；".join(messages),
        execution=(
            "safe_passthrough" if policy_abort_reason else "completed"
        ),
        fallback_used=bool(stage_fallback_used or policy_abort_reason),
        upstream_passthrough=bool(policy_abort_reason),
        reason_code=stage_reason_code,
        details={
            "failure_action": failure_action,
            "denoise_backend_policy": denoise_backend_policy,
            "failure_policy_reason": policy_abort_reason or None,
        },
        components=components,
        review_reasons=pipeline._stage_review_reasons(5),
    )
