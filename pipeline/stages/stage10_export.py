"""Stage 10 final denoise and export."""
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sirilpy.exceptions import CommandError, SirilError

from channel_semantics import BROADBAND_RGB_OSC
from managed_output import export_managed_outputs
from models import PipelineStage
from output_color import build_output_color_manifest
from pipeline_safety import (
    clamp_saturation_boost,
    color_safety_limits,
    should_skip_final_denoise,
)
from save_utils import export_final_outputs


_STAGE10_CHROMA_FOCUS_MIN = 0.34
_STAGE10_SEPARATE_MIN = 0.70
_STAGE10_FULL_BG_STD_MIN = 0.018
_STAGE10_FULL_MOTTLING_MIN = 0.45


def _metric_value(metrics: Dict[str, Any], name: str) -> float:
    try:
        value = float(metrics.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _has_complete_noise_metrics(metrics: Dict[str, Any]) -> bool:
    for name in (
        "chroma_noise_score",
        "bg_std",
        "background_mottling_score",
    ):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value) or value < 0.0:
            return False
    return True


def _config_value(cfg: Any, name: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(getattr(cfg, name, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(lower, min(upper, value))


def _stage10_stage9_local_color_saturation_guard(
    pipeline,
    saturation: float,
) -> Tuple[float, Dict[str, Any]]:
    """Reduce positive saturation when Stage 9 reports localized color risk."""
    requested = float(saturation)
    selected = getattr(pipeline, "_stage9_selected_remix_quality", None)
    metrics = (
        selected.get("metrics") or {}
        if isinstance(selected, dict)
        else {}
    )
    local_status = str(metrics.get("local_quality_status", "") or "")
    stage9_contract_known = hasattr(pipeline, "_stage9_stars_applied")
    stars_required = bool(getattr(pipeline, "_stage9_stars_required", False))
    stars_applied = bool(getattr(pipeline, "_stage9_stars_applied", False))
    reason = "no_local_color_risk"
    risk_score = 0.0
    if stage9_contract_known and stars_required and not stars_applied:
        risk_score = 1.0
        reason = "stage9_required_stars_not_applied"
    elif isinstance(selected, dict) and bool(selected.get("accepted", False)):
        if local_status == "ok":
            risk_score = max(
                0.0,
                min(
                    1.0,
                    _metric_value(metrics, "local_color_risk_score"),
                ),
            )
            if risk_score > 0.0:
                reason = "stage9_local_color_risk"
        elif stage9_contract_known and stars_required:
            risk_score = 1.0
            reason = "stage9_local_color_metrics_unavailable"
    strength = _config_value(
        getattr(pipeline, "cfg", None),
        "stage10_stage9_local_color_risk_strength",
        1.0,
        0.0,
        1.0,
    )
    factor = max(0.0, 1.0 - risk_score * strength)
    guarded = requested * factor if requested > 0.0 else requested
    return guarded, {
        "requested_saturation": requested,
        "effective_saturation": guarded,
        "local_quality_status": local_status or "not_available",
        "local_color_risk_score": risk_score,
        "risk_strength": strength,
        "saturation_factor": factor,
        "applied": bool(requested > 0.0 and guarded < requested - 1e-8),
        "reason": reason,
    }


def _select_stage10_denoise_plan(
    metrics: Optional[Dict[str, Any]],
    *,
    color_input: bool,
    cfg: Any = None,
) -> Dict[str, Any]:
    """Select full/chroma/separate/skip from input background metrics."""
    measured = dict(metrics or {})
    metrics_available = _has_complete_noise_metrics(measured)
    chroma = _metric_value(measured, "chroma_noise_score")
    bg_std = _metric_value(measured, "bg_std")
    mottling = _metric_value(measured, "background_mottling_score")
    chroma_focus_min = _config_value(
        cfg,
        "stage10_chroma_focus_score_min",
        _STAGE10_CHROMA_FOCUS_MIN,
        0.10,
        0.80,
    )
    separate_min = max(
        chroma_focus_min,
        _config_value(
            cfg,
            "stage10_separate_chroma_score_min",
            _STAGE10_SEPARATE_MIN,
            0.35,
            1.50,
        ),
    )
    full_bg_std_min = _config_value(
        cfg,
        "stage10_full_bg_std_min",
        _STAGE10_FULL_BG_STD_MIN,
        0.001,
        0.10,
    )
    full_mottling_min = _config_value(
        cfg,
        "stage10_full_mottling_score_min",
        _STAGE10_FULL_MOTTLING_MIN,
        0.10,
        1.00,
    )

    if not color_input:
        selected_mode = "full"
        reason = "non-color input"
    elif not metrics_available:
        selected_mode = "full"
        reason = "input metrics unavailable; balanced fallback"
    elif chroma >= separate_min:
        selected_mode = "separate"
        reason = "severe channel-specific background color noise"
    elif chroma >= chroma_focus_min:
        if bg_std >= full_bg_std_min or mottling >= full_mottling_min:
            selected_mode = "full"
            reason = "material luminance/background noise accompanies color noise"
        else:
            selected_mode = "chroma"
            reason = "color noise dominates a comparatively stable luminance background"
    elif bg_std >= full_bg_std_min or mottling >= full_mottling_min:
        selected_mode = "full"
        reason = "material luminance/background noise requires full denoise"
    else:
        selected_mode = "skip"
        reason = "all measured background noise is below final-denoise thresholds"

    # Bundled CosmicClarity supports full/luminance/separate. Chroma-only is
    # produced by running full and restoring the original luminance afterwards.
    # The low-noise skip path never invokes CosmicClarity.
    cosmic_clarity_mode = (
        "none"
        if selected_mode == "skip"
        else "full" if selected_mode == "chroma" else selected_mode
    )
    return {
        "selected_mode": selected_mode,
        "cosmic_clarity_mode": cosmic_clarity_mode,
        "reason": reason,
        "color_input": bool(color_input),
        "input_metrics_available": metrics_available,
        "input_metrics": {
            "chroma_noise_score": chroma,
            "bg_std": bg_std,
            "background_mottling_score": mottling,
        },
        "thresholds": {
            "chroma_focus_min": chroma_focus_min,
            "separate_min": separate_min,
            "full_bg_std_min": full_bg_std_min,
            "full_mottling_min": full_mottling_min,
        },
    }


def _is_color_image(image: np.ndarray) -> bool:
    arr = np.asarray(image)
    return bool(
        arr.ndim == 3
        and (arr.shape[0] in (3, 4) or arr.shape[-1] in (3, 4))
    )


def _rgb_float(image: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected RGB image, got shape={arr.shape}")
    if arr.shape[0] >= 3 and arr.shape[-1] not in (1, 3, 4):
        rgb = arr[:3]
        layout = "chw"
    elif arr.shape[-1] >= 3:
        rgb = np.transpose(arr[..., :3], (2, 0, 1))
        layout = "hwc"
    elif arr.shape[0] >= 3:
        rgb = arr[:3]
        layout = "chw"
    else:
        raise ValueError(f"expected RGB image, got shape={arr.shape}")

    rgb = np.nan_to_num(
        np.asarray(rgb, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
    else:
        peak = float(np.max(np.abs(rgb))) if rgb.size else 0.0
        if peak <= 1.5:
            scale = 1.0
        elif peak <= 255.0 * 1.05:
            scale = 255.0
        elif peak <= 65535.0 * 1.05:
            scale = 65535.0
        else:
            scale = max(peak, 1.0)
    return np.clip(rgb / max(scale, 1e-12), 0.0, 1.0), layout


def _restore_rgb_like(source: np.ndarray, rgb: np.ndarray, layout: str) -> np.ndarray:
    source_arr = np.asarray(source)
    restored = np.asarray(rgb, dtype=np.float32)
    if layout == "hwc":
        restored = np.transpose(restored, (1, 2, 0))
    if np.issubdtype(source_arr.dtype, np.integer):
        scale = float(np.iinfo(source_arr.dtype).max)
        return np.rint(np.clip(restored, 0.0, 1.0) * scale).astype(
            source_arr.dtype,
            copy=False,
        )
    peak = float(np.max(np.abs(source_arr))) if source_arr.size else 0.0
    if peak <= 1.5:
        scale = 1.0
    elif peak <= 255.0 * 1.05:
        scale = 255.0
    elif peak <= 65535.0 * 1.05:
        scale = 65535.0
    else:
        scale = max(peak, 1.0)
    return (np.clip(restored, 0.0, 1.0) * scale).astype(
        source_arr.dtype,
        copy=False,
    )


def _chroma_only_denoised_image(
    original: np.ndarray,
    denoised: np.ndarray,
) -> np.ndarray:
    """Keep denoised chroma while restoring the pre-denoise luminance."""
    if np.asarray(original).shape != np.asarray(denoised).shape:
        raise ValueError(
            "chroma-only merge shape mismatch: "
            f"original={np.asarray(original).shape}, denoised={np.asarray(denoised).shape}"
        )
    original_rgb, original_layout = _rgb_float(original)
    denoised_rgb, denoised_layout = _rgb_float(denoised)
    if original_layout != denoised_layout:
        raise ValueError(
            f"chroma-only merge layout mismatch: {original_layout}!={denoised_layout}"
        )
    original_luma = (
        0.2126 * original_rgb[0]
        + 0.7152 * original_rgb[1]
        + 0.0722 * original_rgb[2]
    )
    denoised_luma = (
        0.2126 * denoised_rgb[0]
        + 0.7152 * denoised_rgb[1]
        + 0.0722 * denoised_rgb[2]
    )
    chroma_only = np.clip(
        denoised_rgb + (original_luma - denoised_luma)[None, :, :],
        0.0,
        1.0,
    )
    return _restore_rgb_like(np.asarray(denoised), chroma_only, denoised_layout)


def _stage10_denoise_input(pipeline) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("empty Stage10 denoise input")
        image = np.asarray(image_data)
        metric_reader = getattr(pipeline, "_background_quality_metrics", None)
        metrics = metric_reader(image) if callable(metric_reader) else {}
        plan = _select_stage10_denoise_plan(
            metrics if isinstance(metrics, dict) else {},
            color_input=_is_color_image(image),
            cfg=getattr(pipeline, "cfg", None),
        )
        # Only chroma mode needs the pre-denoise pixels to restore luminance.
        # Avoid retaining a second full-resolution image for full/separate/skip.
        snapshot = (
            np.array(image, copy=True)
            if plan["selected_mode"] == "chroma"
            else None
        )
        return snapshot, plan
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        pipeline.log.warn(f"Stage10 denoise input metrics unavailable: {error}")
        return None, _select_stage10_denoise_plan(
            {},
            color_input=True,
            cfg=getattr(pipeline, "cfg", None),
        )


def _apply_stage10_chroma_only_result(
    pipeline,
    original: Optional[np.ndarray],
) -> Tuple[bool, str]:
    if original is None:
        return False, "original Stage10 input unavailable"
    try:
        denoised = pipeline.siril.get_image_pixeldata(preview=False)
        if denoised is None:
            raise RuntimeError("empty denoised image")
        merged = _chroma_only_denoised_image(original, np.asarray(denoised))
        setter = getattr(pipeline, "_set_current_image_pixeldata", None)
        if callable(setter):
            setter(merged, label="Stage10 chroma-only merge")
        else:
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    pipeline.siril.set_image_pixeldata(merged)
            else:
                pipeline.siril.set_image_pixeldata(merged)
        return True, "original luminance restored after full-model chroma denoise"
    except (AttributeError, CommandError, SirilError, RuntimeError, TypeError, ValueError) as error:
        return False, str(error)


def _run_stage10_scunet_fallback(
    pipeline,
    step_key: str,
    strength: float,
) -> Optional[str]:
    """Avoid another long AI wait after the primary denoiser already timed out."""
    primary_error = str(
        getattr(pipeline, "_last_plugin_script_error", None) or ""
    ).strip()
    if "timeout" in primary_error.lower() or "timed out" in primary_error.lower():
        reason = (
            "SCUNet skipped after primary denoiser timeout to avoid a second "
            "long-running fallback"
        )
        pipeline._last_scunet_fallback_error = reason
        pipeline.log.warn(f"{step_key}: {reason}")
        return None
    return pipeline._run_siril_scunet_denoise_fallback(step_key, strength)


def run_stage10_export(pipeline) -> None:
    """
    阶段 10: 最终降噪与导出
    - 最终色彩微调
    - SCUNet 最终降噪（若可用）
    - 导出 TIFF/PNG/FITS
    """
    stage_label = PipelineStage.EXPORT.label
    pipeline.log.stage_start(stage_label)
    status = "ok"
    messages: List[str] = []
    stage9_contract_known = hasattr(pipeline, "_stage9_stars_applied")
    stage9_stars_required = bool(
        getattr(pipeline, "_stage9_stars_required", False)
    )
    stage9_stars_applied = bool(
        getattr(pipeline, "_stage9_stars_applied", False)
    )
    stage9_missing_required_stars = bool(
        stage9_contract_known
        and stage9_stars_required
        and not stage9_stars_applied
    )
    stage9_starmask_stretch_failed = bool(
        getattr(pipeline, "_stage9_starmask_stretch_failed", False)
    )
    forced_review_only = bool(
        getattr(pipeline.cfg, "force_review_only_output", False)
    )
    review_only_output = forced_review_only or bool(
        getattr(pipeline, "_stage9_bypassed_bad_starless", False)
    ) or stage9_missing_required_stars or stage9_starmask_stretch_failed
    pipeline._final_output_review_only = False
    if stage9_missing_required_stars:
        messages.append(
            "stage9_stars_applied=false while stars_required=true; "
            "normal delivery is not allowed"
        )
    if stage9_starmask_stretch_failed:
        messages.append(
            "stage9_starmask_stretch_failed=true; normal delivery is not allowed"
        )
    if forced_review_only:
        messages.append(
            "force_review_only_output=true; normal delivery names are disabled"
        )

    # 按优先级加载最终图像
    final_file = "stage9_remixed"
    final_loaded = False
    preferred_final_source = str(
        getattr(pipeline, "_stage9_final_source", None) or final_file
    )
    final_candidates = [
        preferred_final_source,
        final_file,
        "input_state_passthrough",
        "starless_enhanced",
        pipeline.stretched_name or "stage7_stretched",
        "stage7_stretched",
    ]
    for candidate in dict.fromkeys(
        str(item) for item in final_candidates if item
    ):
        candidate_path = pipeline.process_dir / f"{candidate}.fit"
        if not candidate_path.exists():
            messages.append(f"final_candidate_missing={candidate}")
            continue
        try:
            pipeline.cmd_with_check("load", candidate)
            final_file = candidate
            final_loaded = True
            break
        except (CommandError, SirilError):
            messages.append(f"final_candidate_load_failed={candidate}")
            continue
    if not final_loaded:
        status = "degraded"
        messages.append("最终候选图加载失败，沿用当前 Siril 图像")
    input_source_fallback_used = bool(
        final_loaded and final_file != preferred_final_source
    )
    pipeline.log.info(f"使用最终图像: {final_file}")

    # 色彩微调
    pipeline.log.info("色彩最终优化...")
    color_limits = color_safety_limits(
        getattr(pipeline, "pipeline_policy", {}) or {},
        getattr(pipeline, "color_calibration_report", {}) or {},
    )
    channel_semantics = str(
        getattr(pipeline, "_channel_semantics", "unknown") or "unknown"
    )
    channel_color_adjustments_allowed = channel_semantics == BROADBAND_RGB_OSC
    requested_final_saturation = (
        0.0
        if (
            bool(getattr(pipeline, "_skip_stage10_color_adjustments", False))
            or not channel_color_adjustments_allowed
        )
        else float(pipeline.cfg.final_saturation)
    )
    if bool(getattr(pipeline, "_skip_stage10_color_adjustments", False)):
        messages.append(
            "Stage10 color adjustment skipped by input-state review guard"
        )
    elif not channel_color_adjustments_allowed:
        messages.append(
            "Stage10 global color adjustment skipped by channel semantics "
            f"({channel_semantics})"
        )
    color_budget_saturation = clamp_saturation_boost(
        requested_final_saturation,
        already_applied=float(getattr(pipeline, "_saturation_boost_applied", 0.0)),
        limits=color_limits,
    )
    if color_budget_saturation != requested_final_saturation:
        messages.append(
            "Stage4 color policy capped Stage10 saturation "
            f"{requested_final_saturation:.3f}->{color_budget_saturation:.3f} "
            f"(budget={color_limits['max_saturation_boost']:.3f})"
        )
    effective_final_saturation, stage9_color_guard = (
        _stage10_stage9_local_color_saturation_guard(
            pipeline,
            color_budget_saturation,
        )
    )
    pipeline._stage10_saturation_guard = dict(stage9_color_guard)
    if stage9_color_guard["applied"]:
        messages.append(
            "Stage9 local color risk capped Stage10 saturation "
            f"{color_budget_saturation:.3f}->{effective_final_saturation:.3f} "
            f"(risk={stage9_color_guard['local_color_risk_score']:.3f}, "
            f"factor={stage9_color_guard['saturation_factor']:.3f}, "
            f"reason={stage9_color_guard['reason']})"
        )
    stage_json_writer = getattr(pipeline, "_write_stage_json", None)
    if callable(stage_json_writer):
        try:
            stage_json_writer(
                "stage10_saturation_guard.json",
                stage9_color_guard,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            pipeline.log.warn(f"Stage10 saturation guard report failed: {error}")
    if abs(effective_final_saturation) > 1e-8:
        try:
            pipeline.cmd_with_check(
                "satu",
                f"{effective_final_saturation:.6f}",
                str(pipeline.cfg.final_bg_factor),
            )
            pipeline._saturation_boost_applied = float(
                getattr(pipeline, "_saturation_boost_applied", 0.0)
            ) + max(0.0, effective_final_saturation)
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"最终饱和度调整跳过: {e}")
            status = "degraded"
            messages.append(f"最终饱和度调整失败: {e}")
    else:
        if stage9_color_guard["applied"]:
            messages.append(
                "Stage10 saturation skipped by Stage9 local color risk guard"
            )
        else:
            messages.append("Stage10 saturation skipped: Stage4 color budget exhausted")

    denoise_input_pixels, denoise_plan = _stage10_denoise_input(pipeline)
    selected_denoise_mode = str(denoise_plan["selected_mode"])
    cosmic_clarity_mode = str(denoise_plan["cosmic_clarity_mode"])
    pipeline._cosmic_clarity_native_denoise_mode_override = cosmic_clarity_mode
    pipeline._cosmic_clarity_native_denoise_strength_override = "0.5"
    input_metrics = denoise_plan["input_metrics"]
    pipeline.log.info(
        "[Stage10] denoise mode selection "
        f"selected={selected_denoise_mode}, cosmic_clarity={cosmic_clarity_mode}, "
        f"chroma={float(input_metrics['chroma_noise_score']):.3f}, "
        f"bg_std={float(input_metrics['bg_std']):.5f}, "
        f"mottling={float(input_metrics['background_mottling_score']):.3f}"
    )
    messages.append(
        "Stage10 denoise mode "
        f"selected={selected_denoise_mode}, cosmic_clarity={cosmic_clarity_mode}, "
        f"reason={denoise_plan['reason']}"
    )

    final_denoise_used = None
    final_scunet_used = None
    denoise_primary = "CosmicClarity Denoise in-process script"
    denoise_primary_status = "skipped"
    denoise_effective = "none"
    denoise_effective_status = "skipped"
    denoise_fallback_used = False
    denoise_fallback_reason = ""
    duplicate_denoise_skip = should_skip_final_denoise(
        stage5_denoise_applied=bool(
            getattr(pipeline, "_stage5_denoise_applied", False)
        ),
        stage8_final_quality=str(
            getattr(pipeline, "_stage8_final_quality", "unknown")
        ),
        stage8_fallback_used=bool(
            getattr(pipeline, "_stage8_fallback_used", False)
        ),
    )
    review_only_denoise_skip = bool(review_only_output)
    low_noise_denoise_skip = selected_denoise_mode == "skip"
    skip_final_denoise = (
        review_only_denoise_skip
        or duplicate_denoise_skip
        or low_noise_denoise_skip
    )
    final_denoise_script = pipeline._find_plugin_script(
        ("processing/CosmicClarity_Denoise.py",)
    )
    final_denoise_executable_args = pipeline._classic_cosmic_clarity_args(
        "sirilcc_denoise.conf",
        "CosmicClarity Denoise",
    )
    if skip_final_denoise:
        if review_only_denoise_skip:
            denoise_primary = "review-only fast export guard"
            denoise_effective = "review source retained"
        elif duplicate_denoise_skip:
            denoise_primary = "duplicate-denoise guard"
            denoise_effective = "stage5 denoise retained"
        else:
            denoise_primary = "low-noise metric guard"
            denoise_effective = "low-noise input retained"
        denoise_primary_status = "skipped"
        denoise_effective_status = "skipped_safe"
    elif final_denoise_script is not None and final_denoise_executable_args is not None:
        cli_args: List[str] = [
            "-denoising_mode",
            cosmic_clarity_mode,
            "-denoise_strength",
            "0.5",
            "-use_gpu",
            *final_denoise_executable_args,
        ]

        final_denoise_used = pipeline._run_plugin_script_by_path(
            "最终降噪",
            "CosmicClarity Denoise",
            final_denoise_script,
            args=tuple(cli_args),
        )
        if final_denoise_used:
            denoise_primary_status = "success"
            denoise_effective = final_denoise_used
            denoise_effective_status = "success"
        else:
            denoise_primary_status = "failed"
            script_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or final_denoise_script.name
            )
            cli_denoise_used = pipeline._run_plugin_script_cli_subprocess(
                "最终降噪",
                "CosmicClarity Denoise",
                final_denoise_script,
                args=tuple(cli_args),
                timeout_sec=pipeline._final_denoise_cli_timeout_sec(),
            )
            if cli_denoise_used:
                final_denoise_used = cli_denoise_used
                denoise_effective = cli_denoise_used
                denoise_effective_status = "success"
                denoise_fallback_used = True
                denoise_fallback_reason = "inprocess_to_cli"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Denoise in-process",
                        script_error,
                        cli_denoise_used,
                        True,
                    )
                )
            else:
                cli_error = (
                    getattr(pipeline, "_last_plugin_script_error", None)
                    or final_denoise_script.name
                )
                native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
                    "最终降噪回退"
                )
                if native_denoise_used:
                    final_denoise_used = native_denoise_used
                    denoise_effective = native_denoise_used
                    denoise_effective_status = "success"
                    denoise_fallback_used = True
                    denoise_fallback_reason = "cli_to_native"
                    messages.append(
                        pipeline._fallback_summary(
                            "CosmicClarity Denoise CLI subprocess",
                            cli_error,
                            native_denoise_used,
                            True,
                        )
                    )
                else:
                    native_error = (
                        getattr(pipeline, "_last_plugin_script_error", None)
                        or "CosmicClarity_Native.py unavailable"
                    )
                    final_scunet_used = _run_stage10_scunet_fallback(
                        pipeline,
                        "最终降噪回退",
                        0.28,
                    )
                    if final_scunet_used:
                        denoise_effective = final_scunet_used
                        denoise_effective_status = "success"
                        denoise_fallback_used = True
                        denoise_fallback_reason = "native_to_scunet"
                        messages.append(
                            pipeline._fallback_summary(
                                "CosmicClarity Native Denoise",
                                native_error,
                                final_scunet_used,
                                True,
                            )
                        )
                    else:
                        scunet_reason = getattr(
                            pipeline,
                            "_last_scunet_fallback_error",
                            None,
                        )
                        messages.append(
                            pipeline._fallback_summary(
                                "CosmicClarity Native Denoise",
                                native_error,
                                "Siril-SCUNet Denoise",
                                False,
                            )
                        )
                        if scunet_reason:
                            messages.append(
                                f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}"
                            )
                        else:
                            messages.append("Siril-SCUNet Denoise 回退不可用")
    elif final_denoise_script is not None:
        denoise_primary = "CosmicClarity Native Denoise cli-subprocess"
        pipeline.log.info(
            "CosmicClarity Denoise classic 路径未启用，使用 Native Denoise"
        )
        native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
            "最终降噪"
        )
        if native_denoise_used:
            final_denoise_used = native_denoise_used
            denoise_effective = native_denoise_used
            denoise_primary_status = "success"
            denoise_effective_status = "success"
            messages.append("CosmicClarity classic 路径未启用，已选择 Native Denoise")
        else:
            denoise_primary_status = "failed"
            native_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or "CosmicClarity_Native.py unavailable"
            )
            final_scunet_used = _run_stage10_scunet_fallback(
                pipeline,
                "最终降噪回退",
                0.28,
            )
            if final_scunet_used:
                denoise_effective = final_scunet_used
                denoise_effective_status = "success"
                denoise_fallback_used = True
                denoise_fallback_reason = "native_to_scunet"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        final_scunet_used,
                        True,
                    )
                )
            else:
                scunet_reason = getattr(pipeline, "_last_scunet_fallback_error", None)
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        "Siril-SCUNet Denoise",
                        False,
                    )
                )
                if scunet_reason:
                    messages.append(f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}")
    else:
        denoise_primary_status = "missing"
        native_denoise_used = pipeline._run_cosmic_clarity_native_denoise_fallback(
            "最终降噪回退"
        )
        if native_denoise_used:
            final_denoise_used = native_denoise_used
            denoise_effective = native_denoise_used
            denoise_effective_status = "success"
            denoise_fallback_used = True
            denoise_fallback_reason = "script_missing_to_native"
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity_Denoise.py",
                    "script missing",
                    native_denoise_used,
                    True,
                )
            )
        else:
            native_error = (
                getattr(pipeline, "_last_plugin_script_error", None)
                or "CosmicClarity_Native.py unavailable"
            )
            final_scunet_used = _run_stage10_scunet_fallback(
                pipeline,
                "最终降噪回退",
                0.28,
            )
            if final_scunet_used:
                denoise_effective = final_scunet_used
                denoise_effective_status = "success"
                denoise_fallback_used = True
                denoise_fallback_reason = "script_missing_to_scunet"
                messages.append(
                    pipeline._fallback_summary(
                        "CosmicClarity Native Denoise",
                        native_error,
                        final_scunet_used,
                        True,
                    )
                )
            else:
                scunet_reason = getattr(pipeline, "_last_scunet_fallback_error", None)
                if scunet_reason:
                    messages.append(
                        pipeline._fallback_summary(
                            "CosmicClarity Native Denoise",
                            native_error,
                            "Siril-SCUNet Denoise",
                            False,
                        )
                    )
                    messages.append(f"Siril-SCUNet Denoise 回退不可用: {scunet_reason}")

    if review_only_denoise_skip:
        pipeline.log.info(
            "Stage10 review-only output skips expensive final denoise"
        )
        messages.append(
            "Stage10 review-only fast export skipped final denoise"
        )
    elif duplicate_denoise_skip:
        pipeline.log.info("Stage5 降噪已通过后续质量门，跳过 Stage10 重复降噪")
        messages.append(
            "Stage10 duplicate-denoise guard skipped final denoise "
            f"(stage8_quality={getattr(pipeline, '_stage8_final_quality', 'unknown')})"
        )
    elif low_noise_denoise_skip:
        pipeline.log.info(
            "Stage10 输入噪声指标均低于门限，跳过昂贵的最终降噪"
        )
        messages.append(
            "Stage10 low-noise guard skipped final denoise "
            f"(chroma={float(input_metrics['chroma_noise_score']):.3f}, "
            f"bg_std={float(input_metrics['bg_std']):.5f}, "
            "mottling="
            f"{float(input_metrics['background_mottling_score']):.3f})"
        )
    elif final_denoise_used:
        pipeline.log.info("已执行最终降噪（作为最后一步处理）")
        messages.append(f"最终降噪使用 {final_denoise_used}")
    elif final_scunet_used:
        pipeline.log.info("已执行 Siril-SCUNet 最终降噪（代码回退）")
        messages.append(f"最终降噪使用 {final_scunet_used}")
    elif getattr(pipeline.cfg, "aberration_api_enabled", False):
        pipeline.log.warn("最终降噪脚本不可用，尝试 Aberration API 作为回退")
        local_model = pipeline._resolve_local_aberration_model()
        aberration_used = pipeline._run_aberration_api("最终降噪", model_path=local_model)
        if aberration_used:
            denoise_effective = aberration_used
            denoise_effective_status = "success"
            denoise_fallback_used = True
            denoise_fallback_reason = "denoiser_chain_to_aberration"
            messages.append(
                pipeline._fallback_summary(
                    "CosmicClarity/SCUNet final denoise",
                    "previous denoise candidates unavailable",
                    aberration_used,
                    True,
                )
            )
        else:
            pipeline.log.warn("最终降噪脚本与 Aberration API 均不可用，跳过最终降噪")
            if pipeline._last_aberration_api_error:
                messages.append(
                    "Aberration API 不可用: "
                    f"{pipeline._short_text(pipeline._last_aberration_api_error, 160)}"
                )
            messages.append("最终降噪脚本与 Aberration API 均不可用")
            denoise_effective_status = "failed"
            status = "degraded" if status == "ok" else status
    else:
        pipeline.log.info("最终降噪脚本不可用，且 Aberration API 默认关闭，跳过最终降噪")
        messages.append("最终降噪未执行（script/scunet unavailable, Aberration API disabled）")
        status = "degraded" if status == "ok" else status

    effective_denoise_mode = "skipped"
    if denoise_effective_status == "success":
        if selected_denoise_mode == "chroma":
            chroma_applied, chroma_note = _apply_stage10_chroma_only_result(
                pipeline,
                denoise_input_pixels,
            )
            if chroma_applied:
                effective_denoise_mode = "chroma"
                messages.append(f"Stage10 chroma-only merge: {chroma_note}")
            else:
                effective_denoise_mode = "full_fallback"
                denoise_fallback_used = True
                denoise_fallback_reason = (
                    denoise_fallback_reason or "chroma_merge_to_full"
                )
                pipeline.log.warn(
                    "Stage10 chroma-only merge unavailable; retaining full denoise: "
                    f"{chroma_note}"
                )
                messages.append(
                    "Stage10 chroma-only merge failed; retained full denoise: "
                    f"{chroma_note}"
                )
        elif selected_denoise_mode == "separate" and final_denoise_used:
            effective_denoise_mode = "separate"
        elif selected_denoise_mode == "separate":
            effective_denoise_mode = "full_fallback"
            messages.append(
                "Stage10 separate mode unavailable on the effective fallback denoiser; "
                "retained fallback output"
            )
        else:
            effective_denoise_mode = "full"

    denoise_plan.update(
        {
            "effective_mode": effective_denoise_mode,
            "effective_component": denoise_effective,
            "effective_status": denoise_effective_status,
            "skipped_by_duplicate_guard": bool(duplicate_denoise_skip),
            "skipped_by_review_only": bool(review_only_denoise_skip),
            "skipped_by_low_noise_guard": bool(low_noise_denoise_skip),
            "fallback_used": bool(denoise_fallback_used),
            "fallback_reason": denoise_fallback_reason or None,
        }
    )
    stage_json_writer = getattr(pipeline, "_write_stage_json", None)
    if callable(stage_json_writer):
        try:
            stage_json_writer("stage10_denoise_plan.json", denoise_plan)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            pipeline.log.warn(f"Stage10 denoise plan report failed: {error}")
            messages.append("stage10_denoise_plan.json 写入失败")

    messages.append(
        f"final_denoise_primary={denoise_primary}; "
        f"primary_status={denoise_primary_status}; "
        f"final_denoise_effective={denoise_effective}; "
        f"effective_status={denoise_effective_status}; "
        f"selected_mode={selected_denoise_mode}; "
        f"effective_mode={effective_denoise_mode}"
    )

    stage_saved = pipeline._save_stage_output("stage10_final")
    if not stage_saved and status == "ok":
        status = "degraded"
        messages.append("stage10 输出保存失败")
    elif stage_saved:
        diff_note = pipeline._stage_diff_note("stage10_final", final_file)
        if diff_note:
            messages.append(diff_note)
        if hasattr(pipeline, "_create_stage_review_bundle"):
            review = pipeline._create_stage_review_bundle(
                "stage10_export",
                final_file,
                "stage10_final",
                context={
                    "final_denoise_skipped": skip_final_denoise,
                    "color_policy_limits": color_limits,
                    "effective_final_saturation": effective_final_saturation,
                    "channel_semantics": channel_semantics,
                    "stage9_local_color_saturation_guard": stage9_color_guard,
                    "denoise_plan": denoise_plan,
                },
            )
            if review.get("report_path"):
                messages.append(f"review_bundle={review['report_path']}")
        feature_note = pipeline._feature_summary_note("最终导出前特征")
        if feature_note:
            messages.append(feature_note)
        if hasattr(pipeline, "_final_quality_report"):
            try:
                final_quality = pipeline._final_quality_report("stage10_final")
                pipeline._write_stage_json("final_quality_report.json", final_quality)
                pipeline.log.info(
                    "[Stage10] final_quality="
                    f"{final_quality.get('final_quality')} "
                    f"status={final_quality.get('status')} "
                    f"needs_conservative_rerun={str(bool(final_quality.get('needs_conservative_rerun'))).lower()}"
                )
                messages.append(
                    "final_quality="
                    f"{final_quality.get('final_quality')} "
                    f"status={final_quality.get('status')}"
                )
                if bool(final_quality.get("needs_conservative_rerun", False)):
                    review_only_output = True
                if final_quality.get("final_quality") != "ok":
                    status = "degraded" if status == "ok" else status
                    issues = final_quality.get("issues", [])
                    if isinstance(issues, list) and issues:
                        issue_text = ", ".join(str(x) for x in issues[:2])
                        pipeline.log.warn(f"[Stage10] final_quality_issues={issue_text}")
                        messages.append("final_quality_issues=" + issue_text)
                    else:
                        messages.append("final_quality=poor")
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                pipeline.log.warn(f"final quality report failed: {e}")
                messages.append("final_quality_report 写入失败")
                status = "degraded" if status == "ok" else status

    managed_export_enabled = bool(
        getattr(pipeline.cfg, "stage10_managed_output_enabled", True)
    )
    managed_pixels: Optional[np.ndarray] = None
    managed_export_report: Optional[Dict[str, Any]] = None
    if managed_export_enabled:
        try:
            if stage_saved:
                pipeline.cmd_with_check("load", "stage10_final")
            current_pixels = pipeline.siril.get_image_pixeldata(preview=False)
            if current_pixels is None:
                raise RuntimeError("Stage10 managed-export pixels unavailable")
            managed_pixels = np.array(current_pixels, copy=True)
        except (
            AttributeError,
            CommandError,
            RuntimeError,
            SirilError,
            TypeError,
            ValueError,
        ) as error:
            managed_export_report = {
                "schema": "seestar.managed-output.v1",
                "status": "partial",
                "ready": False,
                "mode": "independent_managed_derivatives",
                "issues": [f"source_pixels_unavailable: {error}"],
            }
            messages.append(
                "managed output source unavailable; independent derivatives skipped"
            )

    # 切换回原工作目录导出
    pipeline.cmd_with_check("cd", f'"{pipeline.work_dir}"')

    if review_only_output:
        review_suffix = "_linear" if pipeline._stage1_input_mode == "linear_resume" else ""
        base_filename = f"result_review{review_suffix}"
        fallback_base = base_filename
        fallback_fit_base = f"{base_filename}_final"
        pipeline.main_output_basename_template = base_filename
        pipeline.main_output_fit_basename_template = fallback_fit_base
        pipeline._final_output_review_only = True
        status = "degraded" if status == "ok" else status
        messages.append(
            "review_only_output=true; normal result_processed/result_final names withheld"
        )
        pipeline.log.warn(
            "最终质量门控要求保守重跑；本轮仅导出 result_review* 复核产物，"
            "不写入普通 result_processed/result_final 名称"
        )
    else:
        base_filename = pipeline._result_output_basename()
        fallback_base = "result_processed"
        fallback_fit_base = "result_final"
        if pipeline._stage1_input_mode == "linear_resume":
            fallback_base = "result_processed_linear"
            fallback_fit_base = "result_final_linear"

    fit_filename = getattr(
        pipeline,
        "main_output_fit_basename_template",
        base_filename + "_final",
    )
    export_started_at = time.time()
    pipeline._final_export_started_at = export_started_at
    pipeline._final_output_basenames = tuple(
        dict.fromkeys(
            (
                base_filename,
                fit_filename,
                fallback_base,
                fallback_fit_base,
            )
        )
    )
    export_report: Dict[str, Any] = {}
    status, messages = export_final_outputs(
        pipeline.cmd_with_check,
        pipeline.log,
        base_filename=base_filename,
        fit_filename=fit_filename,
        fallback_base=fallback_base,
        fallback_fit_base=fallback_fit_base,
        output_format=getattr(pipeline.cfg, "output_format", "all"),
        png_preview_stretch=not bool(
            getattr(pipeline, "_stage7_stretch_accepted", False)
        ),
        status=status,
        messages=messages,
        export_report=export_report,
    )
    pipeline._write_stage_json("stage10_export_report.json", export_report)
    if managed_export_enabled and managed_pixels is not None:
        scientific_names = {
            str(fit_filename or ""),
            str(fallback_fit_base or ""),
        }
        scientific_paths = [
            pipeline.work_dir / f"{name}.{extension}"
            for name in scientific_names
            if name
            for extension in ("fit", "fits")
        ]
        for extension in ("fit", "fits"):
            for candidate in pipeline.work_dir.glob(f"*.{extension}"):
                try:
                    if candidate.stat().st_mtime >= export_started_at - 1.0:
                        scientific_paths.append(candidate)
                except OSError:
                    continue
        scientific_paths = list(dict.fromkeys(scientific_paths))
        try:
            managed_export_report = export_managed_outputs(
                managed_pixels,
                work_dir=pipeline.work_dir,
                base_filename=base_filename,
                output_format=getattr(pipeline.cfg, "output_format", "all"),
                scientific_paths=scientific_paths,
            )
            pipeline._write_stage_json(
                "managed_output_report.json",
                managed_export_report,
            )
            messages.append(
                "managed_output="
                f"{managed_export_report.get('status', 'unknown')} "
                f"artifacts={len(managed_export_report.get('artifacts') or [])} "
                "scientific_unchanged="
                f"{str(bool((managed_export_report.get('scientific_archive') or {}).get('unchanged', False))).lower()}"
            )
            if not bool(managed_export_report.get("ready", False)):
                messages.append(
                    "managed output incomplete; primary Siril/FITS exports remain valid"
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            managed_export_report = {
                "schema": "seestar.managed-output.v1",
                "status": "partial",
                "ready": False,
                "mode": "independent_managed_derivatives",
                "issues": [str(error)],
            }
            pipeline._write_stage_json(
                "managed_output_report.json",
                managed_export_report,
            )
            pipeline.log.warn(f"Stage10 managed export unavailable: {error}")
            messages.append("managed output export failed")
    elif not managed_export_enabled:
        messages.append("managed output disabled by configuration")

    try:
        output_color_manifest = build_output_color_manifest(
            work_dir=pipeline.work_dir,
            base_filename=base_filename,
            fit_filename=fit_filename,
            fallback_base=fallback_base,
            fallback_fit_base=fallback_fit_base,
            output_format=getattr(pipeline.cfg, "output_format", "all"),
            channel_semantics=channel_semantics,
            review_only=review_only_output,
            exported_after=export_started_at,
            managed_export_report=managed_export_report,
        )
        pipeline._output_color_manifest = output_color_manifest
        pipeline._write_stage_json(
            "output_color_manifest.json",
            output_color_manifest,
        )
        color_summary = output_color_manifest["summary"]
        messages.append(
            "output_color_audit="
            f"{output_color_manifest.get('mode', 'unknown')} "
            f"artifacts={int(color_summary['artifact_count'])} "
            "managed_export_ready="
            f"{str(bool(color_summary.get('managed_export_ready', False))).lower()}"
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        pipeline._output_color_manifest = {
            "schema": "seestar.output-color-manifest.v1",
            "mode": "report_only",
            "rewrote_outputs": False,
            "status": "unavailable",
            "error": str(error),
        }
        pipeline.log.warn(f"Stage10 output color audit unavailable: {error}")
        messages.append("output color audit unavailable; exported files unchanged")

    if denoise_effective_status == "success":
        denoise_component_status = "applied"
        denoise_reason_code = denoise_fallback_reason or "accepted"
    elif review_only_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "review_only_output"
    elif duplicate_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "duplicate_denoise_guard"
    elif low_noise_denoise_skip:
        denoise_component_status = "skipped"
        denoise_reason_code = "auto_low_noise"
    else:
        denoise_component_status = "failed"
        denoise_reason_code = "all_final_denoisers_failed"
    denoise_component = {
        "status": denoise_component_status,
        "method": denoise_effective,
        "primary": denoise_primary,
        "primary_status": denoise_primary_status,
        "selected_mode": selected_denoise_mode,
        "effective_mode": effective_denoise_mode,
        "reason_code": denoise_reason_code,
        "fallback_used": bool(denoise_fallback_used),
        "input": final_file if final_loaded else None,
        "output": "stage10_final" if stage_saved else None,
    }
    input_source_component = {
        "status": "applied" if final_loaded else "failed",
        "method": final_file if final_loaded else None,
        "reason_code": (
            "final_source_recovery"
            if input_source_fallback_used
            else "accepted"
            if final_loaded
            else "unavailable"
        ),
        "fallback_used": input_source_fallback_used,
    }
    export_outputs = export_report.get("outputs") or {}
    export_failed = any(
        isinstance(item, dict) and item.get("status") == "failed"
        for item in export_outputs.values()
    )
    export_fallback_used = bool(export_report.get("fallback_used", False))
    export_component = {
        "status": (
            "failed"
            if export_failed
            else "applied"
            if export_outputs
            else "skipped"
        ),
        "method": {
            key: value.get("selected")
            for key, value in export_outputs.items()
            if isinstance(value, dict) and value.get("selected")
        },
        "reason_code": (
            "final_export_failed"
            if export_failed
            else "final_export_fallback"
            if export_fallback_used
            else "accepted"
            if export_outputs
            else "not_requested"
        ),
        "fallback_used": export_fallback_used,
    }
    stage_denoise_fallback_used = bool(
        denoise_fallback_used and denoise_effective_status == "success"
    )
    stage_fallback_used = bool(
        input_source_fallback_used
        or stage_denoise_fallback_used
        or export_fallback_used
    )
    stage_reason_code = (
        denoise_fallback_reason
        if stage_denoise_fallback_used
        else "final_source_recovery"
        if input_source_fallback_used
        else "final_export_fallback"
        if export_fallback_used
        else ""
    )

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(
        stage_label,
        status,
        elapsed,
        "；".join(messages),
        fallback_used=stage_fallback_used,
        reason_code=stage_reason_code,
        details={
            "review_only_output": bool(review_only_output),
            "final_source": final_file,
        },
        components={
            "input_source": input_source_component,
            "denoise": denoise_component,
            "export": export_component,
        },
    )
