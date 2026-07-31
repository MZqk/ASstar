"""Stage 5 linear cleanup: optional RL deconvolution first, then light denoise."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import PipelineStage
from noise_model import build_noise_model_report, multiscale_denoise_candidate
from sirilpy.exceptions import CommandError, SirilError


GRAXPERT_OBJECT_MODEL_ENV = "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH"
GRAXPERT_GPU_ENV = "SEESTAR_GRAXPERT_GPU"
_ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_GRAXPERT_MODEL_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


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


def _stage5_disabled_denoise_reason(pipeline) -> str:
    """Explain why the optional denoise component did not run."""
    if getattr(pipeline, "_force_denoise_enabled", None) is False:
        return "user_disabled"
    if bool(getattr(pipeline.cfg, "auto_tune_enabled", False)) and getattr(
        pipeline,
        "auto_tune_result",
        None,
    ) is not None:
        return "auto_low_noise"
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


def _run_multiscale_linear_denoise(
    pipeline,
    messages: List[str],
) -> tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "schema": "seestar.multiscale-denoise-candidate.v1",
        "status": "unavailable",
        "accepted": False,
    }
    baseline_saved = pipeline._save_stage_output("stage5_pre_multiscale")
    if not baseline_saved:
        report.update(
            status="prohibited",
            issues=["immutable_baseline_save_failed"],
        )
        messages.append(
            "Stage5 multiscale denoise prohibited: immutable baseline save failed"
        )
        return False, report
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        candidate, report = multiscale_denoise_candidate(
            image_data,
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
        )
        report["transaction"]["baseline_saved"] = True
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
        try:
            pipeline.cmd_with_check("load", "stage5_pre_multiscale")
            report.setdefault("transaction", {}).update(
                rollback_performed=True,
            )
        except (CommandError, SirilError) as rollback_error:
            report.setdefault("transaction", {}).update(
                rollback_performed=False,
                rollback_error=str(rollback_error),
            )
        messages.append(
            "Stage5 multiscale denoise failed; restored baseline: "
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
            return [path] if nonempty_model(path) else []
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


def _stage5_install_user_graxpert_model(source: Path, home: Path) -> tuple[Optional[Path], str]:
    version = _stage5_graxpert_model_version(source)
    if version is None:
        return None, "model_version_directory_must_be_semver"
    target = (
        home
        / "Library"
        / "Application Support"
        / "GraXpert"
        / "deconvolution-object-ai-models"
        / ".".join(str(part) for part in version)
        / "model.onnx"
    )
    try:
        if target.exists():
            if target.samefile(source):
                return target, ""
            return None, "model_version_conflicts_with_isolated_home"
        if target.is_symlink():
            return None, "isolated_home_model_symlink_is_broken"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
        return target, ""
    except OSError as error:
        return None, f"model_link_failed:{error}"


def _stage5_graxpert_object_model() -> tuple[Optional[Path], dict]:
    home = Path(os.environ.get("HOME") or Path.home())
    details = {
        "configured_path": "",
        "source": "",
        "resolved_model_path": "",
        "discovery_reason": "",
    }
    configured_path = os.environ.get(GRAXPERT_OBJECT_MODEL_ENV, "").strip()
    if configured_path:
        configured = Path(configured_path).expanduser()
        if not configured.is_absolute():
            configured = Path.cwd() / configured
        details["configured_path"] = str(configured)
        configured_models = _stage5_graxpert_models_from_path(configured)
        user_model = _stage5_latest_graxpert_model(configured_models)
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
) -> tuple[bool, dict]:
    details = {
        "attempted": False,
        "available": False,
        "model": None,
        "strength": max(
            0.20,
            min(
                0.40,
                float(getattr(pipeline.cfg, "stage5_graxpert_deconv_strength", 0.30)),
            ),
        ),
        "psf_size": 5.0,
        "reason": "",
        "configured_path": "",
        "source": "",
        "resolved_model_path": "",
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
    model, discovery = _stage5_graxpert_object_model()
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

    pipeline._save_stage_output("stage5_graxpert_deconv")
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
    denoise_reason_code = ""
    denoise_fallback_used = False
    deconv_applied = False
    deconv_method = "none"
    deconv_attempted_method = "none"
    deconv_fallback_used = False
    graxpert_details = {}
    multiscale_report: Dict[str, Any] = {
        "status": "not_requested",
        "accepted": False,
    }
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
        noise_pixels = pipeline.siril.get_image_pixeldata(preview=False)
        noise_model_report = build_noise_model_report(
            noise_pixels,
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
            "schema": "seestar.multiscale-noise-model.v1",
            "mode": "report_only",
            "applied_to_pixels": False,
            "status": "unavailable",
            "error": str(error),
        }
        pipeline._stage5_noise_model_report = noise_model_report
        pipeline.log.warn(f"Stage5 noise model report unavailable: {error}")
        messages.append("stage5 noise model unavailable; existing denoise policy unchanged")
    background_risk = _stage5_background_risk(before_adaptive, stage5_policy)

    deconv_applied, graxpert_details = _run_stage5_graxpert_deconvolution(
        pipeline,
        messages,
    )
    if deconv_applied:
        deconv_method = "graxpert_object"
        deconv_attempted_method = "graxpert_object"
        denoise_input = "stage5_graxpert_deconv"
    else:
        if bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)):
            deconv_attempted_method = "siril_rl"
        deconv_applied = _run_stage5_rl_deconvolution(
            pipeline,
            messages,
            fallback_stem="stage5_input_linear",
        )
        if deconv_applied:
            deconv_method = "siril_rl"
            deconv_fallback_used = bool(
                getattr(
                    pipeline.cfg,
                    "stage5_graxpert_deconvolution_enabled",
                    True,
                )
                and graxpert_details.get("reason")
                not in {
                    "deconvolution_disabled",
                    "graxpert_deconvolution_disabled",
                }
            )
    if deconv_applied:
        if deconv_method == "siril_rl":
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
            pipeline.log.warn(
                f"[Stage5] background guard dropped {deconv_method} result: {guard_reason}"
            )
            messages.append(
                f"Stage5 background guard dropped {deconv_method} result: {guard_reason}"
            )
            try:
                pipeline.cmd_with_check("load", "stage5_input_linear")
                denoise_input = "stage5_input_linear"
                deconv_applied = False
                deconv_method = "none"
                deconv_fallback_used = False
            except (CommandError, SirilError) as e:
                status = "degraded"
                messages.append(f"Stage5 background guard reload failed: {pipeline._short_text(e, 160)}")

    if not deconv_applied and pipeline.process_dir:
        try:
            for stale_name in ("stage5_deconv.fit", "stage5_graxpert_deconv.fit"):
                (pipeline.process_dir / stale_name).unlink(missing_ok=True)
        except OSError as e:
            pipeline.log.warn(f"[Stage5] stale deconvolution output cleanup failed: {e}")

    if not bool(getattr(pipeline.cfg, "denoise_enabled", False)):
        denoise_reason_code = _stage5_disabled_denoise_reason(pipeline)
        denoise_reason_text = {
            "user_disabled": "linear denoise disabled by user",
            "auto_low_noise": "linear denoise skipped by automatic low-noise policy",
            "config_disabled": "linear denoise skipped by configuration",
        }[denoise_reason_code]
        messages.append(denoise_reason_text)
    elif bool(
        getattr(pipeline.cfg, "stage5_multiscale_denoise_enabled", True)
    ):
        multiscale_applied, multiscale_report = (
            _run_multiscale_linear_denoise(pipeline, messages)
        )
        pipeline._write_stage_json(
            "stage5_multiscale_denoise.json",
            multiscale_report,
        )
        if multiscale_applied:
            denoise_used = "deterministic_multiscale"
            denoise_reason_code = "accepted"
        elif multiscale_report.get("status") == "skipped_low_noise":
            denoise_used = "none"
            denoise_reason_code = "auto_low_noise"
            messages.append(
                "Stage5 low-noise guard skipped fallback denoisers"
            )
        elif _run_builtin_linear_denoise(pipeline, messages):
            denoise_used = "siril_builtin"
            denoise_reason_code = "primary_failed"
            denoise_fallback_used = True
        else:
            plugin_used = _run_cosmic_clarity_linear_denoise(
                pipeline,
                denoise_mode=denoise_mode,
                denoise_strength=denoise_strength,
                messages=messages,
            )
            if plugin_used:
                denoise_used = plugin_used
                denoise_reason_code = "primary_failed"
                denoise_fallback_used = True
            else:
                status = "degraded"
                denoise_reason_code = "all_denoisers_failed"
                messages.append(
                    "linear denoise unavailable; stage5_linear keeps current "
                    f"{'deconvolved' if deconv_applied else 'linear'} image"
                )
    elif _run_builtin_linear_denoise(pipeline, messages):
        denoise_used = "siril_builtin"
        denoise_reason_code = "accepted"
    else:
        plugin_used = _run_cosmic_clarity_linear_denoise(
            pipeline,
            denoise_mode=denoise_mode,
            denoise_strength=denoise_strength,
            messages=messages,
        )
        if plugin_used:
            denoise_used = plugin_used
            denoise_reason_code = "primary_failed"
            denoise_fallback_used = True
        else:
            status = "degraded"
            denoise_reason_code = "all_denoisers_failed"
            messages.append(
                "linear denoise unavailable; stage5_linear keeps current "
                f"{'deconvolved' if deconv_applied else 'linear'} image"
            )

    linear_saved = pipeline._save_stage_output("stage5_linear")
    pipeline._stage5_denoise_applied = denoise_used != "none"
    if not linear_saved:
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

    if deconv_applied:
        deconv_component_status = "applied"
        deconv_reason_code = "accepted"
    elif guard_triggered:
        deconv_component_status = "rolled_back"
        deconv_reason_code = "background_guard_rollback"
    elif not bool(getattr(pipeline.cfg, "stage5_deconvolution_enabled", True)):
        deconv_component_status = "skipped"
        deconv_reason_code = (
            "user_disabled"
            if str(os.getenv("SEESTAR_STAGE5_DECONV_ENABLE") or "").strip().lower()
            in {"0", "false", "no", "off"}
            else "config_disabled"
        )
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
            "fallback_used": deconv_fallback_used,
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
    stage_fallback_used = deconv_fallback_used or denoise_fallback_used

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
                "GraXpert object deconvolution when local model is available",
                "findstar/makepsf/rl fallback if enabled",
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
                "status": denoise_component_status,
                "reason_code": denoise_reason_code,
                "fallback_used": denoise_fallback_used,
                "method": denoise_used,
                "input": denoise_input,
                "output": "stage5_linear",
                "noise_model_report": noise_model_report,
                "multiscale_candidate": multiscale_report,
                "siril_builtin_mod": max(
                    0.20,
                    min(0.55, float(getattr(pipeline.cfg, "stage5_builtin_denoise_mod", 0.50))),
                ),
                "siril_independent_channels": True,
                "cosmic_clarity_fallback_mode": denoise_mode,
                "cosmic_clarity_fallback_strength": denoise_strength,
            },
            "deconvolution": {
                "status": deconv_component_status,
                "reason_code": deconv_reason_code,
                "fallback_used": deconv_fallback_used,
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
            "components": components,
            "status": status,
            "messages": messages,
        },
    )

    elapsed = pipeline.log.stage_end(stage_name)
    pipeline._record_stage(
        stage_name,
        status,
        elapsed,
        "；".join(messages),
        fallback_used=stage_fallback_used,
        reason_code=("component_fallback_used" if stage_fallback_used else ""),
        components=components,
    )
