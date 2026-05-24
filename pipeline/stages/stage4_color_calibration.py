"""Stage 4 plate solving and color calibration."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from sirilpy.exceptions import CommandError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_SPCC_WHITE_REF = "Average Spiral Galaxy"
NEBULA_SPCC_WHITE_REF = "Star, type G2(v)"
EMISSION_NEBULA_TARGET_TYPES = frozenset(
    {
        "emission_nebula",
        "emission_nebula_widefield",
        "bright_emission_reflection_nebula",
    }
)
DUAL_NARROWBAND_KEYWORDS = frozenset(
    {
        "dual",
        "duo",
        "narrowband",
        "narrow-band",
        "l-extreme",
        "l-enhance",
        "l-ultimate",
        "ha-oiii",
        "ha_oiii",
        "haoiii",
        "h-alpha",
        "oiii",
        "o-iii",
        "triband",
        "tri-band",
    }
)


def _stage4_active_target_type(pipeline) -> str:
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        target_type = str(profile.get("target_type") or "").strip().lower()
        if target_type:
            return target_type
    if hasattr(pipeline, "_active_target_type"):
        return str(pipeline._active_target_type() or "").strip().lower()
    return ""


def _stage4_target_aware_color_mapping(pipeline) -> bool:
    target_type = _stage4_active_target_type(pipeline)
    if target_type in EMISSION_NEBULA_TARGET_TYPES:
        return True

    candidates = [
        getattr(pipeline.cfg, "stage4_spcc_osc_filter", ""),
        getattr(pipeline.cfg, "stage4_spcc_osc_sensor", ""),
        getattr(pipeline, "source_file", ""),
    ]
    profile = getattr(pipeline, "target_profile", None)
    if isinstance(profile, dict):
        candidates.extend(
            [
                profile.get("filter", ""),
                profile.get("filter_name", ""),
                profile.get("instrument", ""),
                profile.get("target_type", ""),
            ]
        )

    text = " ".join(str(item or "").lower() for item in candidates)
    return any(keyword in text for keyword in DUAL_NARROWBAND_KEYWORDS)


def _stage4_effective_spcc_white_ref(pipeline) -> Tuple[str, str]:
    configured = str(
        getattr(pipeline.cfg, "stage4_spcc_white_ref", DEFAULT_SPCC_WHITE_REF)
        or DEFAULT_SPCC_WHITE_REF
    )
    if configured != DEFAULT_SPCC_WHITE_REF:
        return configured, "explicit_config"

    if (
        bool(getattr(pipeline.cfg, "stage4_spcc_adaptive_white_ref_enabled", True))
        and _stage4_target_aware_color_mapping(pipeline)
    ):
        target_type = _stage4_active_target_type(pipeline) or "dual_narrowband"
        nebula_ref = str(
            getattr(pipeline.cfg, "stage4_spcc_nebula_white_ref", NEBULA_SPCC_WHITE_REF)
            or NEBULA_SPCC_WHITE_REF
        )
        return nebula_ref, f"target_profile:{target_type}"

    return configured, "default"


def _stage4_spcc_args(pipeline, *, whiteref: Optional[str] = None) -> Tuple[Tuple[str, ...], List[str]]:
    mode = str(getattr(pipeline.cfg, "stage4_spcc_sensor_mode", "osc") or "osc").lower()
    whiteref = str(whiteref or DEFAULT_SPCC_WHITE_REF)
    messages: List[str] = []

    if mode in {"mono", "mono_lrgb", "lrgb"}:
        pipeline.cmd_with_check("spcc_list", "monosensor")
        pipeline.cmd_with_check("spcc_list", "redfilter")
        pipeline.cmd_with_check("spcc_list", "greenfilter")
        pipeline.cmd_with_check("spcc_list", "bluefilter")
        sensor = str(getattr(pipeline.cfg, "stage4_spcc_mono_sensor", "") or "")
        r_filter = str(getattr(pipeline.cfg, "stage4_spcc_r_filter", "") or "")
        g_filter = str(getattr(pipeline.cfg, "stage4_spcc_g_filter", "") or "")
        b_filter = str(getattr(pipeline.cfg, "stage4_spcc_b_filter", "") or "")
        if not all((sensor, r_filter, g_filter, b_filter)):
            messages.append("Mono/LRGB SPCC config incomplete")
        return (
            (
                _stage4_siril_named_arg("monosensor", sensor),
                _stage4_siril_named_arg("rfilter", r_filter),
                _stage4_siril_named_arg("gfilter", g_filter),
                _stage4_siril_named_arg("bfilter", b_filter),
                _stage4_siril_named_arg("whiteref", whiteref),
            )
            + _stage4_spcc_limitmag_args(pipeline)
            + _stage4_spcc_bgtol_args(pipeline),
            messages,
        )

    pipeline.cmd_with_check("spcc_list", "oscsensor")
    pipeline.cmd_with_check("spcc_list", "oscfilter")
    sensor = str(
        getattr(pipeline.cfg, "stage4_spcc_osc_sensor", "seestar s30pro")
        or "seestar s30pro"
    )
    osc_filter = str(
        getattr(pipeline.cfg, "stage4_spcc_osc_filter", "seestar s30pro")
        or "seestar s30pro"
    )
    return (
        (
            _stage4_siril_named_arg("oscsensor", sensor),
            _stage4_siril_named_arg("oscfilter", osc_filter),
            _stage4_siril_named_arg("whiteref", whiteref),
        )
        + _stage4_spcc_limitmag_args(pipeline)
        + _stage4_spcc_bgtol_args(pipeline),
        messages,
    )


def _stage4_siril_named_arg(name: str, value: str) -> str:
    arg = f"-{name}={str(value or '').strip()}"
    if any(ch.isspace() for ch in arg):
        return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return arg


def _stage4_spcc_bgtol_args(pipeline) -> Tuple[str, ...]:
    bgtol = str(getattr(pipeline.cfg, "stage4_spcc_bgtol", "-2.8,2.0") or "").strip()
    normalized = bgtol.replace(" ", "").replace("+", "")
    if not bgtol or normalized in {"-2.8,2.0", "-2.80,2.00"}:
        return ()
    return (f"-bgtol={bgtol}",)


def _stage4_spcc_limitmag_args(pipeline) -> Tuple[str, ...]:
    limitmag = str(getattr(pipeline.cfg, "stage4_spcc_limitmag", "11.5") or "").strip()
    if not limitmag:
        return ()
    try:
        value = float(limitmag)
    except (TypeError, ValueError):
        pipeline.log.warn(f"忽略无效 SPCC limit magnitude: {limitmag}")
        return ()
    if value <= 0:
        return ()
    return (f"-limitmag={value:g}",)


def _stage4_pcc_args() -> Tuple[str, ...]:
    # Siril's PCC default bgtol is already -2.8,+2.0. Keeping it implicit avoids
    # Siril 1.4.x rejecting explicit negative tuple forms in some CLI paths.
    return ("-catalog=localgaia",)


def _stage4_image_as_chw(image_data: Any) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    arr = np.asarray(image_data)
    if arr.size == 0:
        raise ValueError("image buffer is empty")

    original_dtype = arr.dtype
    if arr.ndim == 2:
        work = arr[np.newaxis, :, :].astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(chw[0], original_dtype)

        return work, restore

    if arr.ndim != 3:
        raise ValueError(f"unsupported image ndim: {arr.ndim}")

    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        work = arr.astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(chw, original_dtype)

        return work, restore

    if arr.shape[-1] in (1, 3, 4):
        work = np.moveaxis(arr, -1, 0).astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(np.moveaxis(chw, 0, -1), original_dtype)

        return work, restore

    if arr.shape[0] >= 3:
        work = arr.astype(np.float32, copy=True)

        def restore(chw: np.ndarray) -> np.ndarray:
            return _stage4_cast_like(chw, original_dtype)

        return work, restore

    raise ValueError(f"unsupported image shape: {arr.shape}")


def _stage4_cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype, copy=False)
    return values.astype(dtype, copy=False)


def _stage4_luminance(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] < 3:
        return rgb[0]
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)


def _stage4_background_neutralize(chw: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    output = np.nan_to_num(chw.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    if output.shape[0] < 3:
        return np.clip(output, 0.0, None), {"applied": False, "reason": "mono image"}

    rgb = output[:3]
    lum = _stage4_luminance(rgb)
    finite_mask = np.isfinite(lum)
    if int(np.count_nonzero(finite_mask)) < 64:
        raise ValueError("not enough finite pixels for background neutralization")

    lo = float(np.quantile(lum[finite_mask], 0.05))
    hi = float(np.quantile(lum[finite_mask], 0.45))
    bg_mask = finite_mask & (lum >= lo) & (lum <= hi)
    if int(np.count_nonzero(bg_mask)) < 256:
        hi = float(np.quantile(lum[finite_mask], 0.60))
        bg_mask = finite_mask & (lum <= hi)
    if int(np.count_nonzero(bg_mask)) < 64:
        bg_mask = finite_mask

    medians = np.array([float(np.median(channel[bg_mask])) for channel in rgb], dtype=np.float32)
    target = float(np.median(medians))
    offsets = target - medians
    output[:3] = np.clip(rgb + offsets[:, np.newaxis, np.newaxis], 0.0, None)
    return output, {
        "applied": True,
        "sample_pixels": int(np.count_nonzero(bg_mask)),
        "channel_medians_before": [float(v) for v in medians],
        "target_median": target,
        "offsets": [float(v) for v in offsets],
    }


def _stage4_star_white_balance(chw: np.ndarray, pipeline) -> Tuple[np.ndarray, Dict[str, Any]]:
    output = np.clip(chw.astype(np.float32, copy=True), 0.0, None)
    if output.shape[0] < 3:
        return output, {"applied": False, "reason": "mono image", "white_reference_pixels": 0}

    rgb = output[:3]
    lum = _stage4_luminance(rgb)
    finite_mask = np.isfinite(lum)
    if int(np.count_nonzero(finite_mask)) < 256:
        return output, {"applied": False, "reason": "not enough finite pixels", "white_reference_pixels": 0}

    valid_lum = lum[finite_mask]
    median_lum = float(np.median(valid_lum))
    std_lum = float(np.std(valid_lum))
    low = max(float(np.quantile(valid_lum, 0.985)), median_lum + 3.0 * std_lum)
    high = float(np.quantile(valid_lum, 0.9995))
    if high <= low:
        high = float(np.max(valid_lum))

    max_channel = np.max(rgb, axis=0)
    min_channel = np.min(rgb, axis=0)
    chroma = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    saturation_limit = float(np.quantile(max_channel[finite_mask], 0.999))

    star_mask = (
        finite_mask
        & (lum >= low)
        & (lum <= high)
        & (max_channel <= saturation_limit)
        & (chroma <= 0.35)
    )
    if int(np.count_nonzero(star_mask)) < 32:
        star_mask = finite_mask & (lum >= low) & (lum <= high) & (max_channel <= saturation_limit)

    star_pixels = int(np.count_nonzero(star_mask))
    min_pixels = int(getattr(pipeline.cfg, "stage4_local_star_wb_min_pixels", 80) or 80)
    if star_pixels < min_pixels:
        return output, {
            "applied": False,
            "reason": "insufficient unsaturated low-chroma star samples",
            "white_reference_pixels": star_pixels,
            "required_pixels": min_pixels,
        }

    medians = np.array([float(np.median(channel[star_mask])) for channel in rgb], dtype=np.float32)
    if float(np.min(medians)) <= 0.0:
        return output, {
            "applied": False,
            "reason": "invalid star channel medians",
            "white_reference_pixels": star_pixels,
            "channel_medians_before": [float(v) for v in medians],
        }

    target = float(np.median(medians))
    gain_limit = float(getattr(pipeline.cfg, "stage4_local_star_wb_gain_limit", 1.25) or 1.25)
    gain_limit = max(1.01, min(gain_limit, 1.50))
    gains = np.clip(target / medians, 1.0 / gain_limit, gain_limit)
    output[:3] = np.clip(rgb * gains[:, np.newaxis, np.newaxis], 0.0, None)
    return output, {
        "applied": True,
        "white_reference_pixels": star_pixels,
        "channel_medians_before": [float(v) for v in medians],
        "target_median": target,
        "gains": [float(v) for v in gains],
        "gain_limit": gain_limit,
        "sample_policy": "medium-bright unsaturated low-chroma stars",
    }


def _stage4_write_image_pixels(pipeline, pixels: np.ndarray) -> None:
    lock_factory = getattr(pipeline.siril, "image_lock", None)
    if callable(lock_factory):
        with lock_factory():
            pipeline.siril.set_image_pixeldata(pixels)
        return
    pipeline.siril.set_image_pixeldata(pixels)


def _stage4_local_color_fallback(pipeline, *, target_aware: bool = False) -> Tuple[bool, str, str, float, Dict[str, Any], str]:
    image_data = pipeline.siril.get_image_pixeldata(preview=False)
    chw, restore = _stage4_image_as_chw(image_data)
    neutralized, bg_report = _stage4_background_neutralize(chw)

    wb_enabled = bool(getattr(pipeline.cfg, "stage4_local_star_wb_enabled", True))
    star_report: Dict[str, Any] = {"applied": False, "reason": "disabled", "white_reference_pixels": 0}
    balanced = neutralized
    if wb_enabled:
        balanced, star_report = _stage4_star_white_balance(neutralized, pipeline)

    restored = restore(balanced)
    _stage4_write_image_pixels(pipeline, restored)

    report = {
        "target_aware": bool(target_aware),
        "background_neutralization": bg_report,
        "star_white_balance": star_report,
    }
    if star_report.get("applied"):
        message = "local background neutralization + star white balance fallback ok"
        if target_aware:
            message += " (target-aware color mapping)"
        return True, "LOCAL_STAR_WB", "local_star_white_balance_fallback", 0.58, report, message

    message = "local background neutralization fallback ok; star samples insufficient"
    if target_aware:
        message += " (target-aware color mapping)"
    return True, "BACKGROUND_NEUTRALIZATION", "background_neutralization_only", 0.35, report, message


def run_stage4_color_calibration(pipeline) -> None:
    """
    阶段 4: 图像解析 + 色彩校准
    - 显式从 stage3_bgremoved 进入
    - platesolve -noflip -downscale 后保存 stage4_psolved
    - SPCC 指定 Seestar S30 Pro OSC 参数，失败后 PCC，最后本地背景中性化/星点白平衡回退
    """
    stage_label = "阶段 4: 图像解析 + 色彩校准"
    pipeline.log.stage_start(stage_label)
    status = "ok"
    color_ok = False
    color_method = "none"
    color_warning = ""
    color_confidence = 0.0
    messages: List[str] = []
    spcc_white_ref = DEFAULT_SPCC_WHITE_REF
    spcc_white_ref_reason = "default"
    spcc_white_ref_fallback = None
    local_fallback_report: Optional[Dict[str, Any]] = None
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}

    try:
        pipeline.cmd_with_check("load", "stage3_bgremoved")
        messages.append("input=stage3_bgremoved")
    except (CommandError, SirilError) as e:
        status = "degraded"
        messages.append(f"stage3_bgremoved load failed; using current image: {e}")
        pipeline.log.warn(f"stage4 load stage3_bgremoved failed, using current image: {e}")

    pipeline.platesolve_ok = False
    platesolve_attempted = True
    if bool(getattr(pipeline.cfg, "stage4_platesolve_enabled", True)):
        try:
            pipeline.log.info("执行图像解析: platesolve -noflip -downscale")
            pipeline.cmd_with_check("platesolve", "-noflip", "-downscale")
            pipeline.platesolve_ok = True
            messages.append("platesolve -noflip -downscale ok")
        except (CommandError, SirilError) as e:
            status = "degraded"
            messages.append(f"platesolve -noflip -downscale failed: {e}")
            pipeline.log.warn(f"图像解析失败: {e}")
    else:
        platesolve_attempted = False
        status = "degraded"
        messages.append("platesolve disabled by config; stage4_psolved will mirror input")
        pipeline.log.warn("Stage4 platesolve disabled by config")

    ps_saved = pipeline._save_stage_output("stage4_psolved")
    if not ps_saved:
        status = "degraded"
        messages.append("stage4_psolved 保存失败")

    if hasattr(pipeline, "_run_target_profile_preflight"):
        profile_msg = pipeline._run_target_profile_preflight(
            source="Stage4 preflight",
            metadata_candidates=("stage4_psolved", "stage3_bgremoved", getattr(pipeline, "source_file", None)),
        )
        if profile_msg:
            messages.append(profile_msg)
            policy = getattr(pipeline, "pipeline_policy", {}) or {}
            stage4_policy = policy.get("stage4_color", {}) if isinstance(policy, dict) else {}

    spcc_runtime_allowed = bool(getattr(pipeline.cfg, "spcc_enabled", True))
    target_aware_color = _stage4_target_aware_color_mapping(pipeline)
    allow_light_spcc = (
        os.getenv("SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS", "1").strip().lower()
        in ENV_TRUE_VALUES
    )
    if (
        spcc_runtime_allowed
        and getattr(pipeline, "_stage1_input_mode", "") == "light_preprocess"
        and not allow_light_spcc
    ):
        spcc_runtime_allowed = False
        messages.append("SPCC skipped on Light_ preprocess mode to avoid siril-cli crash risk")

    if spcc_runtime_allowed:
        try:
            spcc_white_ref, spcc_white_ref_reason = _stage4_effective_spcc_white_ref(pipeline)
            spcc_args, spcc_messages = _stage4_spcc_args(pipeline, whiteref=spcc_white_ref)
            messages.extend(spcc_messages)
            messages.append(
                f"SPCC whiteref={spcc_white_ref} ({spcc_white_ref_reason})"
            )
            pipeline.cmd_with_check("spcc", *spcc_args)
            color_ok = True
            color_method = "SPCC"
            color_confidence = 0.90
            messages.append("SPCC ok")
            pipeline.log.info("分光光度色彩校准成功 (SPCC)")
        except (CommandError, SirilError) as e:
            primary_error = e
            if (
                spcc_white_ref != DEFAULT_SPCC_WHITE_REF
                and spcc_white_ref_reason != "explicit_config"
                and not target_aware_color
            ):
                try:
                    spcc_white_ref_fallback = DEFAULT_SPCC_WHITE_REF
                    messages.append(
                        "SPCC adaptive whiteref failed; retrying "
                        f"{DEFAULT_SPCC_WHITE_REF}: {e}"
                    )
                    fallback_args, fallback_messages = _stage4_spcc_args(
                        pipeline,
                        whiteref=DEFAULT_SPCC_WHITE_REF,
                    )
                    messages.extend(fallback_messages)
                    pipeline.cmd_with_check("spcc", *fallback_args)
                    color_ok = True
                    color_method = "SPCC"
                    color_confidence = 0.86
                    messages.append("SPCC ok with default whiteref fallback")
                    pipeline.log.info(
                        "分光光度色彩校准成功 (SPCC, default whiteref fallback)"
                    )
                except (CommandError, SirilError) as fallback_error:
                    messages.append(
                        f"SPCC failed: {primary_error}; default whiteref retry failed: {fallback_error}"
                    )
                    pipeline.log.warn(
                        f"SPCC 失败: {primary_error}; default whiteref retry failed: {fallback_error}"
                    )
            elif target_aware_color and spcc_white_ref != DEFAULT_SPCC_WHITE_REF:
                messages.append(
                    f"SPCC failed: {e}; ordinary galaxy white reference fallback disabled for target-aware color mapping"
                )
                pipeline.log.warn(
                    f"SPCC 失败: {e}; target-aware 目标禁止回退到普通星系白参考"
                )
            else:
                messages.append(f"SPCC failed: {e}")
                pipeline.log.warn(f"SPCC 失败: {e}")
    else:
        messages.append("SPCC disabled; using PCC fallback")
        pipeline.log.warn("SPCC 已禁用，尝试 PCC")

    if not color_ok:
        try:
            pcc_args = _stage4_pcc_args()
            pipeline.cmd_with_check("pcc", *pcc_args)
            color_ok = True
            color_method = "PCC"
            color_confidence = 0.72
            messages.append("PCC localgaia ok (default bgtol)")
            pipeline.log.info("PCC 色彩校准成功 (localgaia)")
        except (CommandError, SirilError) as e:
            messages.append(f"PCC localgaia failed: {e}")
            pipeline.log.warn(f"PCC 失败: {e}")

    if not color_ok:
        try:
            (
                color_ok,
                color_method,
                color_warning,
                color_confidence,
                local_fallback_report,
                fallback_message,
            ) = _stage4_local_color_fallback(pipeline, target_aware=target_aware_color)
            messages.append(fallback_message)
            pipeline.log.info(fallback_message)
        except Exception as e:
            color_ok = False
            status = "degraded"
            messages.append(f"local color fallback failed: {e}")
            pipeline.log.warn(f"本地色彩回退失败: {e}")

    if not color_ok:
        status = "degraded"
        color_method = "none"
        color_warning = color_warning or "color_calibration_failed"
    elif color_method == "BACKGROUND_NEUTRALIZATION":
        status = "degraded"

    if (
        color_warning
        and stage4_policy.get("reduce_saturation_if_solution_imprecise", False)
    ):
        messages.append("color policy limits later saturation/color gains due to imprecise solution")
    if color_method in {"LOCAL_STAR_WB", "BACKGROUND_NEUTRALIZATION"} and platesolve_attempted and not pipeline.platesolve_ok:
        messages.append("platesolve 失败，已使用本地背景中性化/星点白平衡回退")

    color_saved = pipeline._save_stage_output("stage4_color")
    if not color_saved:
        status = "degraded"
        messages.append("stage4_color 输出保存失败")

    pipeline.color_calibration_report = {
        "stage": "stage4_color",
        "input": "stage3_bgremoved",
        "platesolve": {
            "attempted": platesolve_attempted,
            "ok": bool(pipeline.platesolve_ok),
            "command": "platesolve -noflip -downscale",
            "output": "stage4_psolved.fit",
        },
        "method": color_method,
        "target_aware_color_mapping": bool(target_aware_color),
        "spcc_white_reference": {
            "requested": spcc_white_ref,
            "reason": spcc_white_ref_reason,
            "fallback": spcc_white_ref_fallback,
            "ordinary_galaxy_fallback_allowed": not bool(target_aware_color),
        },
        "local_fallback": local_fallback_report,
        "status": (
            "success_with_warning"
            if color_ok and color_warning
            else ("success" if color_ok else "degraded")
        ),
        "warning": color_warning or None,
        "color_confidence": color_confidence,
        "policy": (
            pipeline._active_policy_name()
            if hasattr(pipeline, "_active_policy_name")
            else str(policy.get("policy_name", "generic_low_snr_safe"))
        ),
        "policy_adjustments": {
            "reduce_saturation_boost": bool(
                stage4_policy.get("reduce_saturation_if_solution_imprecise", False)
                and color_warning
            ),
            "blue_gain_limit": stage4_policy.get("blue_gain_limit"),
            "red_gain_limit": stage4_policy.get("red_gain_limit"),
            "max_allowed_saturation_boost": stage4_policy.get("max_allowed_saturation_boost"),
        },
        "outputs": {
            "psolved": "stage4_psolved.fit",
            "color": "stage4_color.fit",
            "compat_color": "stage4_colorbalanced.fit",
        },
        "messages": messages,
    }
    pipeline._write_stage_json("color_calibration_report.json", pipeline.color_calibration_report)

    elapsed = pipeline.log.stage_end(stage_label)
    pipeline._record_stage(stage_label, status, elapsed, "；".join(messages))
