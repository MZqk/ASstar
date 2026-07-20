"""Stage 9 star processing and remixing."""
from typing import Any, Dict, List, Optional

import numpy as np

from models import PipelineStage
import stage9_quality
from sirilpy.exceptions import CommandError, SirilError


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _stage9_remix_intensity_candidates(
    pipeline,
    *,
    primary_intensity: float,
    remix_scale: float,
) -> List[tuple[str, float]]:
    """Build a genuinely descending remix ladder after the primary candidate."""
    candidates: List[tuple[str, float]] = [("primary", primary_intensity)]
    configured_levels = getattr(
        pipeline.cfg,
        "stage9_fallback_intensity_levels",
        (0.75, 0.55, 0.40),
    )
    if not isinstance(configured_levels, (list, tuple)):
        configured_levels = (0.75, 0.55, 0.40)
    compatibility_cap = _clamp_float(
        getattr(pipeline.cfg, "star_fallback_intensity", 0.95),
        0.40,
        1.05,
    )
    previous = primary_intensity
    for raw_level in configured_levels:
        try:
            base_level = min(float(raw_level), compatibility_cap)
        except (TypeError, ValueError):
            continue
        effective = _clamp_float(base_level * remix_scale, 0.10, 1.05)
        if effective >= previous - 1e-6:
            continue
        label = f"fallback_{int(round(effective * 100)):03d}"
        candidates.append((label, effective))
        previous = effective
    return candidates


def _assess_stage9_candidate(
    pipeline,
    source_stem: str,
    *,
    attempt: str,
    formula: str,
) -> Dict[str, Any]:
    assessor = getattr(pipeline, "_stage9_assess_current_remix", None)
    if not callable(assessor):
        return {
            "attempt": attempt,
            "formula": formula,
            "status": "not_measured",
            "accepted": True,
            "gate_enabled": False,
            "issues": ["quality assessor unavailable"],
            "metrics": {},
        }
    return assessor(
        source_stem,
        attempt=attempt,
        formula=formula,
    )


def _stage9_needs_compact_mask_recovery(quality: Dict[str, Any]) -> bool:
    """Return whether a rejected candidate indicates broad starmask contamination."""
    if bool(quality.get("accepted", False)):
        return False
    issue_text = " ".join(str(item) for item in quality.get("issues", [])).lower()
    if any(
        token in issue_text
        for token in (
            "background_mottling_growth",
            "changed_pixel_ratio",
            "background_lift",
            "chromatic_star_addition_ratio",
            "new_hollow_structure_max_area",
            "local_connected_component_max_area",
            "local_nonstellar_shape_component_count",
            "local_single_pixel_component_ratio",
            "local_cyan_blue_component_max_area",
            "core_color_jump_component_max_area",
        )
    ):
        return True
    metrics = quality.get("metrics") or {}
    limits = quality.get("limits") or {}
    try:
        changed_ratio = float(metrics.get("changed_pixel_ratio", 0.0) or 0.0)
        recovery_limit = min(
            float(limits.get("changed_pixel_ratio", 0.35) or 0.35),
            float(
                limits.get(
                    "background_mottling_low_absolute_changed_pixel_ratio_max",
                    0.12,
                )
                or 0.12
            ),
        )
    except (TypeError, ValueError):
        return False
    return changed_ratio > recovery_limit


def _stage9_has_recovery_shortfall(quality: Dict[str, Any]) -> bool:
    """Return whether lowering Screen intensity cannot improve the rejection."""
    issue_text = " ".join(str(item) for item in quality.get("issues", [])).lower()
    return any(
        token in issue_text
        for token in (
            "weak_star_recovery_ratio",
            "star_recovery_ratio",
            "star_aperture_recovery_ratio",
            "star_wing_recovery_ratio",
            "residual_dark_hole_ratio",
            "new_hollow_structure_max_area",
            "star_recovery_metrics_unavailable",
        )
    )


def _prepare_stage9_star_reference(
    pipeline,
    starmask_name: str,
    messages: List[str],
) -> Dict[str, Any]:
    """Build a source-confirmed star catalog before any star plugin runs."""
    catalog: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "original starmask pixels unavailable",
    }
    reference_source = "starmask_only"
    try:
        pipeline.cmd_with_check("load", starmask_name)
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        if callable(get_pixels):
            pixels = get_pixels(preview=False)
            if pixels is not None:
                starmask_pixels = np.array(pixels, copy=True)
                source_pixels = None
                process_dir = getattr(pipeline, "process_dir", None)
                for source_stem in ("stage5_linear", "stage6_input", "working"):
                    source_path = (
                        process_dir / f"{source_stem}.fit"
                        if process_dir is not None
                        else None
                    )
                    if source_path is None or not source_path.exists():
                        continue
                    pipeline.cmd_with_check("load", source_stem)
                    source_data = get_pixels(preview=False)
                    if source_data is not None:
                        source_pixels = np.array(source_data, copy=True)
                        reference_source = source_stem
                        break
                catalog = stage9_quality.build_star_reference_catalog(
                    starmask_pixels,
                    pipeline.cfg,
                    source_image=source_pixels,
                )
                catalog["reference_source"] = reference_source
                pipeline.cmd_with_check("load", starmask_name)
    except (
        CommandError,
        SirilError,
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        IndexError,
    ) as error:
        catalog = {"status": "unavailable", "reason": str(error)}

    try:
        pipeline.cmd_with_check("load", starmask_name)
    except (CommandError, SirilError, RuntimeError, AttributeError) as restore_error:
        if catalog.get("status") == "ok":
            catalog = {
                "status": "unavailable",
                "reason": f"failed to restore starmask after cataloging: {restore_error}",
            }

    pipeline._stage9_star_reference_catalog = catalog
    summary = stage9_quality.star_reference_summary(catalog)
    pipeline._stage9_star_reference_summary = summary
    if summary.get("status") == "ok":
        messages.append(
            "Stage9 source-confirmed star reference "
            f"source={summary.get('reference_source', reference_source)}, "
            f"method={summary.get('method', 'legacy_starmask_catalog')}, "
            f"components={int(summary.get('component_count', 0))}, "
            f"weak={int(summary.get('weak_component_count', 0))}, "
            f"bright={int(summary.get('bright_component_count', 0))}, "
            f"peak_ratio={float(summary.get('bright_to_weak_peak_ratio', 0.0)):.2f}, "
            f"mixed={str(bool(summary.get('mixed_star_field', False))).lower()}, "
            "detail_percentile="
            f"{float(summary.get('source_detail_percentile', 0.0)):.1f}"
        )
    else:
        reason = str(summary.get("reason") or "unknown")
        messages.append(f"Stage9 original starmask reference unavailable: {reason}")
        pipeline.log.warn(
            "Stage9 original starmask reference unavailable; enabled quality gate "
            f"will fail closed: {reason}"
        )
    return catalog


def _write_stage9_quality_report(
    pipeline,
    attempts: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    *,
    source_stem: str,
    mode: str,
) -> None:
    writer = getattr(pipeline, "_write_stage_json", None)
    if not callable(writer):
        return
    stars_required = bool(getattr(pipeline, "_stage9_stars_required", True))
    stars_applied = bool(getattr(pipeline, "_stage9_stars_applied", False))
    stars_application_mode = str(
        getattr(pipeline, "_stage9_stars_application_mode", mode) or mode
    )
    pipeline.log.info(
        "[Stage9] star application contract "
        f"required={str(stars_required).lower()}, "
        f"applied={str(stars_applied).lower()}, "
        f"mode={stars_application_mode}"
    )
    report_formula = str(
        (selected or {}).get("formula")
        or (attempts[-1].get("formula") if attempts else "none")
    )
    writer(
        "stage9_remix_quality.json",
        {
            "mode": mode,
            "formula": report_formula,
            "source_stem": source_stem,
            "stars_required": stars_required,
            "stars_applied": stars_applied,
            "stars_application_mode": stars_application_mode,
            "star_reference": getattr(
                pipeline,
                "_stage9_star_reference_summary",
                {"status": "unavailable", "reason": "not prepared"},
            ),
            "starmask_calibration": getattr(
                pipeline,
                "_stage9_starmask_calibration",
                None,
            ),
            "starmask_stretch_failed": bool(
                getattr(pipeline, "_stage9_starmask_stretch_failed", False)
            ),
            "attempts": attempts,
            "selected": selected,
        },
    )


def _append_stage9_review_bundle(
    pipeline,
    messages: List[str],
    attempts: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    *,
    source_stem: str,
    mode: str,
    stage_saved: bool,
) -> None:
    """Create Stage 9 before/after evidence for accepted and degraded outputs."""
    creator = getattr(pipeline, "_create_stage_review_bundle", None)
    if not stage_saved or not callable(creator):
        return
    review_candidates = list(attempts)
    selected_attempt = str((selected or {}).get("attempt") or "").strip() or None
    if attempts and selected is None:
        selected_attempt = "stage9_safe_rollback"
        review_candidates.append(
            {
                "id": selected_attempt,
                "name": "stage9_remixed",
                "status": "selected",
                "selected": True,
                "reason": "all remix candidates rejected; retained safe source",
            }
        )
    review = creator(
        "stage9_star_remixing",
        source_stem,
        "stage9_remixed",
        context={
            "mode": mode,
            "stars_required": bool(
                getattr(pipeline, "_stage9_stars_required", True)
            ),
            "stars_applied": bool(
                getattr(pipeline, "_stage9_stars_applied", False)
            ),
            "stars_application_mode": str(
                getattr(pipeline, "_stage9_stars_application_mode", mode) or mode
            ),
            "star_reference": getattr(
                pipeline,
                "_stage9_star_reference_summary",
                {"status": "unavailable", "reason": "not prepared"},
            ),
            "starmask_calibration": getattr(
                pipeline,
                "_stage9_starmask_calibration",
                None,
            ),
        },
        candidates=review_candidates,
        selected_candidate=selected_attempt,
    )
    if review.get("status") == "ready":
        messages.append(f"review_bundle={review['report_path']}")


def _prepare_stage9_starmask_for_pixel_remix(
    pipeline,
    starmask_name: str,
    *,
    star_stretch_used: bool,
    messages: List[str],
    strict_support: bool = False,
    output_name: str = "starmask_stretched",
) -> str:
    stretch_enabled = bool(
        getattr(pipeline.cfg, "stage9_starmask_stretch_enabled", True)
    )
    stretched_name = output_name
    if star_stretch_used:
        pipeline._stage9_starmask_calibration = {
            "status": "plugin_stretched",
            "reason": "plugin-provided nonlinear star layer",
        }
        messages.append("Stage9 starmask uses plugin-stretched star layer for pixel remix")
        return stretched_name
    try:
        pipeline.cmd_with_check("load", starmask_name)
        calibration: Dict[str, Any] = {
            "status": "unavailable",
            "reason": "starmask pixels unavailable",
        }
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        starmask_data = None
        support_mask = None
        if callable(get_pixels):
            starmask_data = get_pixels(preview=False)
            if starmask_data is not None:
                calibration = stage9_quality.calibrate_starmask_asinh(
                    starmask_data,
                    pipeline.cfg,
                    include_support_mask=True,
                    strict_support=strict_support,
                    reference_catalog=getattr(
                        pipeline,
                        "_stage9_star_reference_catalog",
                        None,
                    ),
                )
                support_mask = calibration.get("_compact_support_mask")
        compact_enabled = bool(
            getattr(pipeline.cfg, "stage9_compact_starmask_enabled", True)
        )
        compact_applied = False

        def write_pixels(pixels, *, label: str) -> bool:
            safe_pixel_writer = getattr(
                pipeline,
                "_set_current_image_pixeldata",
                None,
            )
            if callable(safe_pixel_writer):
                safe_pixel_writer(pixels, label=label)
                return True
            set_pixels = getattr(pipeline.siril, "set_image_pixeldata", None)
            if not callable(set_pixels):
                return False
            lock_factory = getattr(pipeline.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    set_pixels(pixels)
            else:
                pipeline.log.warn(
                    f"{label}: image_lock unavailable, writing pixels without thread lock"
                )
                set_pixels(pixels)
            return True

        if calibration.get("status") == "ok" and compact_enabled:
            if starmask_data is not None and support_mask is not None:
                compact_pixels = stage9_quality.apply_compact_starmask_support(
                    starmask_data,
                    support_mask,
                )
                compact_applied = write_pixels(
                    compact_pixels,
                    label="Stage9 compact starmask",
                )
            if compact_applied:
                compact_name = (
                    "starmask_compact_recovery"
                    if strict_support
                    else "starmask_compact"
                )
                pipeline.cmd_with_check("save", compact_name)
                calibration["compact_layer_applied"] = True
                calibration["compact_layer_stem"] = compact_name
                messages.append(
                    "Stage9 compact starmask applied before Asinh "
                    f"(mode={calibration.get('support_mode', 'normal')}, "
                    f"support={float(calibration.get('compact_support_coverage', 0.0)):.3f}, "
                    "removed_predicted_change="
                    f"{float(calibration.get('removed_predicted_change_ratio', 0.0)):.3f})"
                )
            else:
                calibration["adaptive_status"] = "ok"
                calibration["status"] = "fallback_safe"
                calibration["reason"] = "compact support pixel write unavailable"
                calibration["compact_layer_applied"] = False
        elif calibration.get("status") == "ok":
            calibration["compact_layer_applied"] = False
            calibration["compact_layer_disabled"] = True
        if calibration.get("status") != "ok":
            calibration.setdefault("compact_layer_applied", compact_applied)
            if compact_enabled and starmask_data is not None:
                public_calibration = {
                    key: value
                    for key, value in calibration.items()
                    if not str(key).startswith("_")
                }
                public_calibration["fail_closed"] = True
                pipeline._stage9_starmask_calibration = public_calibration
                if not strict_support:
                    pipeline._stage9_starmask_stretch_failed = True
                reason = str(calibration.get("reason") or "compact support unavailable")
                messages.append(
                    "Stage9 compact starmask rejected; raw starmask is not eligible "
                    f"for formal delivery ({reason})"
                )
                return starmask_name
        if calibration.get("status") == "ok":
            stretch = _clamp_float(
                float(calibration["stretch"]),
                1.10,
                1000.0,
            )
            offset = _clamp_float(
                float(calibration["offset"]),
                0.00001,
                0.0060,
            )
            messages.append(
                "Stage9 adaptive starmask calibration "
                f"samples={int(calibration.get('star_sample_count', 0))}, "
                f"components={int(calibration.get('compact_component_count', 0))}, "
                f"faint={float(calibration.get('faint_value', 0.0)):.5f}, "
                f"peak={float(calibration.get('peak_value', 0.0)):.5f}, "
                "predicted_change="
                f"{float(calibration.get('predicted_change_ratio', 0.0)):.3f}/"
                f"{float(calibration.get('predicted_change_ratio_limit', 0.0)):.3f}"
            )
        else:
            stretch = _clamp_float(
                getattr(pipeline.cfg, "stage9_starmask_asinh_stretch", 2.0),
                1.10,
                3.00,
            )
            offset = _clamp_float(
                getattr(pipeline.cfg, "stage9_starmask_asinh_offset", 0.001),
                0.0005,
                0.0060,
            )
            messages.append(
                "Stage9 adaptive starmask calibration unavailable; "
                f"using configured fallback ({calibration.get('reason', 'unknown')})"
            )

        if calibration.get("status") == "ok" and not stretch_enabled:
            calibration["starmask_stretch_disabled"] = True
            calibration["stretch_applied"] = False
            pipeline.cmd_with_check("save", stretched_name)
            pipeline._stage9_starmask_calibration = {
                key: value
                for key, value in calibration.items()
                if not str(key).startswith("_")
            }
            messages.append(
                "Stage9 starmask stretch disabled; retained validated compact support"
            )
            if pipeline.process_dir:
                pipeline._stage9_stretched_starmask_file = (
                    pipeline.process_dir / f"{stretched_name}.fit"
                )
            return stretched_name

        multi_anchor_curve = bool(
            calibration.get("status") == "ok"
            and calibration.get("multi_anchor_curve", False)
        )
        if multi_anchor_curve:
            curved_pixels = stage9_quality.apply_calibrated_starmask(
                starmask_data,
                calibration,
            )
            if not write_pixels(
                curved_pixels,
                label="Stage9 monotonic multi-anchor starmask",
            ):
                raise RuntimeError("multi-anchor starmask pixel write unavailable")
            calibration["multi_anchor_curve_applied"] = True
            calibration["stretch_applied"] = True
            stretch_method = "monotonic_multi_anchor_star_curve"
        else:
            pipeline.cmd_with_check("asinh", f"{stretch:.3f}", f"{offset:.5f}")
            calibration["stretch_applied"] = True
            stretch_method = "asinh"
        if calibration.get("status") != "ok":
            calibration.setdefault(
                "adaptive_status",
                str(calibration.get("status") or "unavailable"),
            )
            calibration["status"] = "fallback_safe"
            calibration["fallback_stretch_applied"] = True
        pipeline._stage9_starmask_calibration = {
            key: value
            for key, value in calibration.items()
            if not str(key).startswith("_")
        }
        messages.append(
            "Stage9 starmask stretched before pixel remix "
            f"(method={stretch_method}, stretch={stretch:.3f}, offset={offset:.5f})"
        )
        pipeline.cmd_with_check("save", stretched_name)
        if pipeline.process_dir:
            pipeline._stage9_stretched_starmask_file = (
                pipeline.process_dir / f"{stretched_name}.fit"
            )
        return stretched_name
    except (
        CommandError,
        SirilError,
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        IndexError,
    ) as e:
        pipeline._stage9_starmask_calibration = {
            "status": "failed",
            "reason": str(e),
            "strict_support": bool(strict_support),
        }
        if not strict_support:
            pipeline._stage9_starmask_stretch_failed = True
        pipeline.log.warn(f"Stage9 starmask stretch failed: {e}")
        messages.append(f"Stage9 starmask stretch failed: {e}")
        return starmask_name


def run_stage9_star_remixing(pipeline) -> None:
    """
    阶段 9: 星点处理与合成
    - 对齐工作流中的 Star Stretch / SCNR / Curves / StarComposer
    - 插件不可用时使用阶段 8 的 starless_enhanced 作为主图，再回混非线性 starmask
    """
    stage_label = PipelineStage.STAR_REMIXING.label
    pipeline.log.stage_start(stage_label)
    messages: List[str] = []
    pipeline._stage9_bypassed_bad_starless = False
    pipeline._stage9_stars_required = not bool(
        getattr(pipeline, "_star_preserve_target_bypass", False)
    )
    pipeline._stage9_stars_applied = False
    pipeline._stage9_stars_application_mode = "pending"
    pipeline._stage9_starmask_stretch_failed = False
    pipeline._stage9_last_star_overlay_mask = None
    pipeline._stage9_last_weak_overlay_mask = None
    pipeline._stage9_last_bright_overlay_mask = None
    pipeline._stage9_last_star_layer = None
    pipeline._stage9_selected_remix_quality = None
    pipeline._stage9_star_reference_catalog = {
        "status": "unavailable",
        "reason": "not prepared",
    }
    pipeline._stage9_star_reference_summary = stage9_quality.star_reference_summary(
        pipeline._stage9_star_reference_catalog
    )
    source_stem = getattr(pipeline, "_stage8_final_source", "starless_enhanced") or "starless_enhanced"
    fallback_used = bool(getattr(pipeline, "_stage8_fallback_used", False))
    messages.append(
        "stage9_starless_source="
        f"{source_stem}; stage8_fallback_used={str(fallback_used).lower()}"
    )
    if bool(getattr(pipeline, "_star_preserve_target_bypass", False)):
        try:
            pipeline.cmd_with_check("load", source_stem)
            stage_saved = pipeline._save_stage_output("stage9_remixed")
            pipeline._stage9_final_source = source_stem
            pipeline._stage9_stars_application_mode = "not_required_star_preserve"
            _write_stage9_quality_report(
                pipeline,
                [],
                None,
                source_stem=source_stem,
                mode="star_preserve_target_bypass",
            )
            _append_stage9_review_bundle(
                pipeline,
                messages,
                [],
                None,
                source_stem=source_stem,
                mode="star_preserve_target_bypass",
                stage_saved=stage_saved,
            )
            messages.append(
                "star-preserve target bypassed starmask import and star remix "
                f"(source={source_stem})"
            )
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                "skipped" if stage_saved else "degraded",
                elapsed,
                "；".join(messages),
            )
            return
        except (CommandError, SirilError) as error:
            pipeline._stage9_stars_required = True
            pipeline._stage9_stars_application_mode = (
                "pending_after_star_preserve_bypass_failure"
            )
            pipeline.log.warn(
                "Star-preserve Stage9 bypass failed; continuing with regular remix path: "
                f"{error}"
            )
            messages.append(
                "star-preserve Stage9 bypass failed: "
                f"{pipeline._short_text(error, 160)}"
            )
    bad_starless_reason = pipeline._stage9_bad_starless_reason()
    starless_advisories = list(
        getattr(pipeline, "_stage9_starless_advisories", []) or []
    )
    if starless_advisories:
        advisory_text = ", ".join(str(item) for item in starless_advisories)
        messages.append(
            "Stage9 accepted-stretch advisory; continuing controlled star remix: "
            f"{advisory_text}"
        )
        pipeline.log.info(
            "Stage9 continues controlled star remix because Stage7 stretched output "
            f"passed its quality gate: {advisory_text}"
        )
    if bad_starless_reason:
        safe_source = pipeline._stage9_review_safe_source()
        pipeline.log.warn(
            "Stage9 bypasses starless remix because selected starless is unsafe: "
            f"{bad_starless_reason}"
        )
        messages.append(
            "stage9_bad_starless_bypass fallback_used=true "
            f"source={safe_source}; reason={bad_starless_reason}"
        )
        try:
            pipeline.cmd_with_check("load", safe_source)
            stage_saved = pipeline._save_stage_output("stage9_remixed")
            pipeline._stage9_bypassed_bad_starless = True
            pipeline._stage9_final_source = safe_source
            pipeline._stage9_stars_application_mode = "unsafe_starless_bypass"
            _write_stage9_quality_report(
                pipeline,
                [],
                None,
                source_stem=safe_source,
                mode="unsafe_starless_bypass",
            )
            _append_stage9_review_bundle(
                pipeline,
                messages,
                [],
                None,
                source_stem=safe_source,
                mode="unsafe_starless_bypass",
                stage_saved=stage_saved,
            )
            diff_note = pipeline._stage_diff_note("stage9_remixed", safe_source)
            if diff_note:
                messages.append(diff_note)
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                "degraded",
                elapsed,
                "；".join(messages),
            )
            return
        except (CommandError, SirilError) as e:
            messages.append(f"stage9 safe-source fallback failed: {e}")
            pipeline.log.warn(f"Stage9 safe-source fallback failed: {e}")

    external_starmask = pipeline._find_external_fit(
        [
            "sasp_starmask.fit",
            "starmask_sasp.fit",
            "starmask_from_sasp.fit",
        ]
    )
    if external_starmask:
        try:
            imported = pipeline._import_external_fit(external_starmask, "starmask_external")
            if imported:
                pipeline.cmd_with_check("save", "starmask_external_raw")
                pipeline.cmd_with_check("save", "starmask")
                pipeline.starmask_file = pipeline.process_dir / "starmask_external_raw.fit"
                pipeline.log.info(f"已导入外部 Starmask: {external_starmask.name}")
        except (OSError, CommandError, SirilError) as e:
            pipeline.log.warn(f"导入外部 Starmask 失败，继续使用本地 starmask: {e}")

    star_stretch_used = False
    if pipeline.starmask_file and pipeline.starmask_file.exists():
        _prepare_stage9_star_reference(
            pipeline,
            pipeline.starmask_file.stem,
            messages,
        )
        try:
            pipeline.cmd_with_check("load", pipeline.starmask_file.stem)
            star_stretch_label = pipeline._run_first_available_command(
                "星点拉伸",
                [
                    ("SASP Star Stretch", ("sasp_star_stretch",)),
                    ("NB to RGB Stars", ("nb_to_rgb_stars",)),
                ],
            )
            pipeline._run_first_available_command(
                "星点去紫",
                [
                    ("SASP Invert/SCNR", ("sasp_invert_scnr",)),
                    ("SCNR", ("scnr",)),
                ],
            )
            pipeline._run_first_available_command(
                "星点微调",
                [
                    ("SASP Curves Editor", ("sasp_curves_editor",)),
                    ("Curves", ("curves",)),
                ],
            )
            pipeline.cmd_with_check("save", "starmask_stretched")
            star_stretch_used = bool(star_stretch_label)
            if star_stretch_used:
                pipeline._stage9_stretched_starmask_file = (
                    pipeline.process_dir / "starmask_stretched.fit"
                )
        except (CommandError, SirilError) as e:
            star_stretch_used = False
            pipeline.log.warn(f"星点处理插件链失败，使用原始 starmask: {e}")

    # 按工作流先在 Siril 侧做 Starless 二次细化，再进行星点合成
    if fallback_used:
        messages.append("Stage9 skipped starless secondary enhancement because Stage8 used fallback")
    else:
        try:
            pipeline.cmd_with_check("load", source_stem)
            pipeline._run_first_available_command(
                "细节/结构增强2",
                [
                    ("VeraLux Revela", ("veralux_revela",)),
                    ("Revela", ("revela",)),
                ],
            )
            if pipeline.cfg.optional_color_transform_enabled:
                pipeline._run_first_available_command(
                    "调色2（可选）",
                    [
                        ("VeraLux Vectra", ("veralux_vectra",)),
                        ("Vectra", ("vectra",)),
                    ],
                )
            pipeline._run_first_available_command(
                "最终微调颜色",
                [
                    ("VeraLux Curves", ("veralux_curves",)),
                    ("Curves", ("curves",)),
                ],
            )
            pipeline.cmd_with_check("save", source_stem)
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"Starless 二次细化失败，沿用当前 {source_stem}: {e}")

    remix_scale = _clamp_float(
        getattr(pipeline, "_stage9_star_intensity_scale", 1.0),
        0.45,
        1.0,
    )
    if fallback_used:
        messages.append("Stage8 fallback source active; using controlled pixel remix")
        remix_scale = min(remix_scale, 0.95 / max(float(pipeline.cfg.star_intensity), 1e-6))
        messages.append("Stage8 fallback star remix intensity capped at 0.950")
    messages.append(
        "Stage9 bypassed StarComposer; formal remix uses explicit "
        "starmask-top/starless-bottom Alpha+Screen composition"
    )
    composer_used = None
    remix_attempts: List[Dict[str, Any]] = []
    selected_remix: Optional[Dict[str, Any]] = None
    composer_rejected = False
    if composer_used:
        composer_quality = _assess_stage9_candidate(
            pipeline,
            source_stem,
            attempt="starcomposer",
            formula="plugin_starcomposer",
        )
        remix_attempts.append(composer_quality)
        if not bool(composer_quality.get("accepted", False)):
            composer_rejected = True
            issues = ", ".join(str(item) for item in composer_quality.get("issues", [])[:3])
            messages.append(f"Stage9 gate rejected StarComposer: {issues}")
            pipeline.log.warn(f"Stage9 gate rejected StarComposer: {issues}")
            try:
                pipeline.cmd_with_check("load", source_stem)
            except (CommandError, SirilError) as error:
                messages.append(f"Stage9 StarComposer rollback failed: {error}")
            composer_used = None
        else:
            selected_remix = composer_quality
            pipeline._stage9_selected_remix_quality = dict(composer_quality)

    if composer_used and selected_remix is not None:
        stage_saved = pipeline._save_stage_output("stage9_remixed")
        pipeline._stage9_stars_applied = bool(stage_saved)
        pipeline._stage9_stars_application_mode = (
            "starcomposer" if stage_saved else "starcomposer_save_failed"
        )
        pipeline._stage9_final_source = "stage9_remixed" if stage_saved else source_stem
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            selected_remix,
            source_stem=source_stem,
            mode="starcomposer",
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            remix_attempts,
            selected_remix,
            source_stem=source_stem,
            mode="starcomposer",
            stage_saved=stage_saved,
        )
        diff_note = pipeline._stage_diff_note("stage9_remixed", "stage8_enhanced")
        if diff_note:
            messages.append(diff_note)
        stage7_diff_note = pipeline._stage_diff_note("stage9_remixed", "stage7_stretched")
        if stage7_diff_note:
            messages.append(stage7_diff_note)
        elapsed = pipeline.log.stage_end(stage_label)
        if stage_saved:
            pipeline._record_stage(
                stage_label,
                'ok',
                elapsed,
                "；".join(messages),
            )
        else:
            messages.append("stage9 输出保存失败")
            pipeline._record_stage(
                stage_label,
                'degraded',
                elapsed,
                "；".join(messages),
            )
        return

    pipeline.log.info("执行基于上一阶段的星点合成...")
    if not pipeline.starmask_file or not pipeline.starmask_file.exists():
        pipeline.log.warn("无星点蒙版，跳过混合阶段")
        if composer_rejected:
            stage_saved = pipeline._save_stage_output("stage9_remixed")
            pipeline._stage9_stars_application_mode = "rejected_keep_starless"
            _write_stage9_quality_report(
                pipeline,
                remix_attempts,
                None,
                source_stem=source_stem,
                mode="rejected_keep_starless",
            )
            _append_stage9_review_bundle(
                pipeline,
                messages,
                remix_attempts,
                None,
                source_stem=source_stem,
                mode="rejected_keep_starless",
                stage_saved=stage_saved,
            )
            elapsed = pipeline.log.stage_end(stage_label)
            pipeline._record_stage(
                stage_label,
                "degraded",
                elapsed,
                "；".join(messages + ["StarComposer rejected; kept Stage8 starless source"]),
            )
            return
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._stage9_stars_application_mode = "no_starmask"
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            None,
            source_stem=source_stem,
            mode="no_starmask",
        )
        pipeline._record_stage(
            stage_label, 'skipped', elapsed, "无星点蒙版")
        return

    intensity = _clamp_float(pipeline.cfg.star_intensity * remix_scale, 0.10, 1.05)
    if remix_scale < 0.999:
        reason = getattr(pipeline, "_stage9_star_intensity_reason", "")
        if not reason:
            reason = "stage8 fallback star intensity cap" if fallback_used else "stage7 residual stars"
        messages.append(
            "Stage9 star remix intensity reduced from safety diagnostics "
            f"(base={pipeline.cfg.star_intensity:.3f}, effective={intensity:.3f}, "
            f"reason={reason})"
        )
    starmask_name = pipeline.starmask_file.stem
    remix_starmask_name = _prepare_stage9_starmask_for_pixel_remix(
        pipeline,
        starmask_name,
        star_stretch_used=star_stretch_used,
        messages=messages,
    )
    if bool(getattr(pipeline, "_stage9_starmask_stretch_failed", False)):
        try:
            pipeline.cmd_with_check("load", source_stem)
        except (CommandError, SirilError) as error:
            messages.append(f"Stage9 starmask failure rollback failed: {error}")
        stage_saved = pipeline._save_stage_output("stage9_remixed")
        pipeline._stage9_final_source = source_stem
        pipeline._stage9_stars_application_mode = "starmask_stretch_failed"
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            None,
            source_stem=source_stem,
            mode="starmask_stretch_failed",
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            remix_attempts,
            None,
            source_stem=source_stem,
            mode="starmask_stretch_failed",
            stage_saved=stage_saved,
        )
        messages.append(
            "Stage9 did not remix the original linear starmask after stretch failure; "
            "kept Stage8 source for review-only export"
        )
        if not stage_saved:
            messages.append("stage9 输出保存失败")
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(stage_label, "degraded", elapsed, "；".join(messages))
        return
    candidates = _stage9_remix_intensity_candidates(
        pipeline,
        primary_intensity=intensity,
        remix_scale=remix_scale,
    )
    messages.append(
        "Stage9 remix intensity ladder="
        + " -> ".join(f"{value:.3f}" for _, value in candidates)
    )

    for attempt_label, candidate_intensity in candidates:
        applied = pipeline._apply_previous_stage_star_remix(
            source_stem,
            remix_starmask_name,
            candidate_intensity,
        )
        if not applied:
            remix_attempts.append(
                {
                    "attempt": attempt_label,
                    "formula": "screen",
                    "intensity": candidate_intensity,
                    "status": "failed",
                    "accepted": False,
                    "issues": ["pixel remix execution failed"],
                    "metrics": {},
                }
            )
            messages.append(
                f"Stage9 {attempt_label} Screen remix execution failed "
                f"(intensity={candidate_intensity:.3f})"
            )
            continue

        quality = _assess_stage9_candidate(
            pipeline,
            source_stem,
            attempt=f"screen_{attempt_label}",
            formula="screen",
        )
        quality["intensity"] = candidate_intensity
        remix_attempts.append(quality)
        if bool(quality.get("accepted", False)):
            selected_remix = quality
            pipeline._stage9_selected_remix_quality = dict(quality)
            messages.append(
                "previous_stage_star_remix "
                f"source={source_stem}, starmask={remix_starmask_name}, "
                f"attempt={attempt_label}, formula=screen, "
                f"intensity={candidate_intensity:.3f}"
            )
            break

        issues = ", ".join(str(item) for item in quality.get("issues", [])[:3])
        messages.append(
            f"Stage9 gate rejected {attempt_label} Screen remix: {issues}"
        )
        pipeline.log.warn(
            f"Stage9 gate rejected {attempt_label} Screen remix: {issues}"
        )
        try:
            pipeline.cmd_with_check("load", source_stem)
        except (CommandError, SirilError) as error:
            messages.append(f"Stage9 {attempt_label} rollback failed: {error}")

        if (
            attempt_label == "primary"
            and _stage9_needs_compact_mask_recovery(quality)
        ):
            initial_calibration = dict(
                getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
            )
            recovery_starmask_name = _prepare_stage9_starmask_for_pixel_remix(
                pipeline,
                starmask_name,
                star_stretch_used=False,
                messages=messages,
                strict_support=True,
                output_name="starmask_stretched_recovery",
            )
            recovery_calibration = dict(
                getattr(pipeline, "_stage9_starmask_calibration", {}) or {}
            )
            recovery_applied = bool(
                recovery_calibration.get("compact_layer_applied", False)
            )
            combined_calibration = dict(recovery_calibration)
            combined_calibration["recovery_attempted"] = True
            combined_calibration["recovery_applied"] = recovery_applied
            combined_calibration["initial"] = initial_calibration
            pipeline._stage9_starmask_calibration = combined_calibration
            if not recovery_applied:
                messages.append(
                    "Stage9 compact-mask recovery unavailable; continuing the "
                    "existing intensity ladder"
                )
                if _stage9_has_recovery_shortfall(quality):
                    messages.append(
                        "Stage9 stopped intensity fallback because lowering Screen "
                        "intensity cannot improve star recovery"
                    )
                    break
                continue

            remix_starmask_name = recovery_starmask_name
            messages.append(
                "Stage9 broad starmask coverage detected; regenerated strict "
                "compact support before lowering remix intensity"
            )
            recovered = pipeline._apply_previous_stage_star_remix(
                source_stem,
                remix_starmask_name,
                candidate_intensity,
            )
            if not recovered:
                remix_attempts.append(
                    {
                        "attempt": "screen_compact_recovery",
                        "formula": "screen",
                        "intensity": candidate_intensity,
                        "status": "failed",
                        "accepted": False,
                        "issues": ["compact-mask remix execution failed"],
                        "metrics": {},
                    }
                )
                messages.append("Stage9 compact-mask recovery remix execution failed")
                if _stage9_has_recovery_shortfall(quality):
                    messages.append(
                        "Stage9 stopped intensity fallback because lowering Screen "
                        "intensity cannot improve star recovery"
                    )
                    break
                continue

            recovery_quality = _assess_stage9_candidate(
                pipeline,
                source_stem,
                attempt="screen_compact_recovery",
                formula="screen",
            )
            recovery_quality["intensity"] = candidate_intensity
            recovery_quality["starmask"] = remix_starmask_name
            remix_attempts.append(recovery_quality)
            if bool(recovery_quality.get("accepted", False)):
                selected_remix = recovery_quality
                pipeline._stage9_selected_remix_quality = dict(recovery_quality)
                messages.append(
                    "previous_stage_star_remix "
                    f"source={source_stem}, starmask={remix_starmask_name}, "
                    "attempt=compact_recovery, formula=screen, "
                    f"intensity={candidate_intensity:.3f}"
                )
                break

            recovery_issues = ", ".join(
                str(item) for item in recovery_quality.get("issues", [])[:3]
            )
            messages.append(
                "Stage9 gate rejected strict compact-mask recovery: "
                f"{recovery_issues}"
            )
            pipeline.log.warn(
                "Stage9 gate rejected strict compact-mask recovery: "
                f"{recovery_issues}"
            )
            try:
                pipeline.cmd_with_check("load", source_stem)
            except (CommandError, SirilError) as error:
                messages.append(f"Stage9 compact recovery rollback failed: {error}")
            quality = recovery_quality
            if _stage9_has_recovery_shortfall(recovery_quality):
                messages.append(
                    "Stage9 stopped intensity fallback after strict support because "
                    "lowering Screen intensity cannot improve star recovery"
                )
                break

        if _stage9_has_recovery_shortfall(quality):
            messages.append(
                "Stage9 stopped intensity fallback because lowering Screen intensity "
                "cannot improve star recovery"
            )
            break

    if selected_remix is None:
        try:
            pipeline.cmd_with_check("load", source_stem)
        except (CommandError, SirilError) as error:
            messages.append(f"Stage9 final rollback failed: {error}")
        stage_saved = pipeline._save_stage_output("stage9_remixed")
        pipeline._stage9_final_source = source_stem
        pipeline._stage9_stars_application_mode = "rejected_keep_starless"
        _write_stage9_quality_report(
            pipeline,
            remix_attempts,
            None,
            source_stem=source_stem,
            mode="rejected_keep_starless",
        )
        _append_stage9_review_bundle(
            pipeline,
            messages,
            remix_attempts,
            None,
            source_stem=source_stem,
            mode="rejected_keep_starless",
            stage_saved=stage_saved,
        )
        elapsed = pipeline.log.stage_end(stage_label)
        messages.append("Stage9 gate rejected all remix candidates; kept Stage8 starless source")
        if not stage_saved:
            messages.append("stage9 输出保存失败")
        pipeline._record_stage(stage_label, "degraded", elapsed, "；".join(messages))
        return

    stage_saved = pipeline._save_stage_output("stage9_remixed")
    pipeline._stage9_stars_applied = bool(stage_saved)
    pipeline._stage9_stars_application_mode = (
        "screen" if stage_saved else "screen_save_failed"
    )
    pipeline._stage9_final_source = "stage9_remixed" if stage_saved else source_stem
    _write_stage9_quality_report(
        pipeline,
        remix_attempts,
        selected_remix,
        source_stem=source_stem,
        mode="screen",
    )
    _append_stage9_review_bundle(
        pipeline,
        messages,
        remix_attempts,
        selected_remix,
        source_stem=source_stem,
        mode="screen",
        stage_saved=stage_saved,
    )
    diff_note = pipeline._stage_diff_note("stage9_remixed", "stage8_enhanced")
    if diff_note:
        messages.append(diff_note)
    stage7_diff_note = pipeline._stage_diff_note("stage9_remixed", "stage7_stretched")
    if stage7_diff_note:
        messages.append(stage7_diff_note)

    elapsed = pipeline.log.stage_end(stage_label)
    if stage_saved:
        pipeline._record_stage(
            stage_label,
            'ok',
            elapsed,
            "；".join(messages),
        )
    else:
        messages.append("stage9 输出保存失败")
        pipeline._record_stage(
            stage_label,
            'degraded',
            elapsed,
            "；".join(messages),
        )
