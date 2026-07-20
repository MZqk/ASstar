#!/usr/bin/env python3
"""Stage11 AI postprocess runner extracted from the main pipeline."""

from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from models import PipelineStage
from pipeline_safety import clamp_ai_color_adjustments, color_safety_limits
from review_bundle import apply_visual_acceptance, create_image_review_bundle
from sirilpy.exceptions import CommandError, DataError, SirilError


def run_stage11_ai_postprocess(
    owner: Any,
    *,
    write_png_rgb16_func: Callable[[Path, Any], None],
) -> None:
    """
    Execute optional Stage11 AI postprocess.

    The owner object must provide the runtime state and callbacks used by
    stage11 (cfg/log/siril/cmd_with_check/etc.).
    """
    stage_label = PipelineStage.AI_POSTPROCESS.label
    owner.log.stage_start(stage_label)
    status = "ok"
    message = ""
    owner.ai_outputs_generated = False

    if not owner.cfg.ai_post_enabled:
        elapsed = owner.log.stage_end(stage_label)
        owner._record_stage(
            stage_label,
            "skipped",
            elapsed,
            "SEESTAR_AI_ENABLED not enabled",
        )
        return

    if bool(getattr(owner, "_final_output_review_only", False)):
        elapsed = owner.log.stage_end(stage_label)
        owner._record_stage(
            stage_label,
            "skipped",
            elapsed,
            "Stage10 output is review-only; AI postprocess is not allowed to promote it",
        )
        return

    missing_env: list[str] = []
    if not owner.cfg.ai_endpoint.strip():
        missing_env.append("SEESTAR_AI_ENDPOINT")
    if not owner.cfg.ai_model.strip():
        missing_env.append("SEESTAR_AI_MODEL")
    if not owner.cfg.ai_api_key.strip():
        missing_env.append("SEESTAR_AI_API_KEY")
    if missing_env:
        elapsed = owner.log.stage_end(stage_label)
        owner._record_stage(
            stage_label,
            "skipped",
            elapsed,
            f"missing required env: {', '.join(missing_env)}",
        )
        return

    source_name = "stage11_ai_source"
    output_name = "stage11_ai_output"
    output_fit_name = "stage11_ai_output_fit"
    blended_name = "stage11_ai_blended"
    candidate_records: list[dict[str, Any]] = []

    temp_files = [
        owner.work_dir / f"{source_name}.fit",
        owner.work_dir / f"{output_name}.png",
        owner.work_dir / f"{output_fit_name}.fit",
        owner.work_dir / f"{blended_name}.fit",
    ]

    try:
        owner.cmd_with_check("cd", f'"{owner.work_dir}"')
        owner.cmd_with_check("save", source_name)

        source_features = owner._measure_current_features()
        if source_features is None:
            raise RuntimeError("failed to extract baseline image features")
        source_image_data = owner.siril.get_image_pixeldata(preview=False)
        if source_image_data is None:
            raise RuntimeError("failed to read current image pixel data")

        owner.log.info(
            f"[AI] Requesting adjustment plan model={owner.cfg.ai_model} "
            f"timeout={owner.cfg.ai_timeout_sec}s"
        )
        ai_adjustments, ai_summary = owner._request_ai_adjustments(source_features)
        color_limits = color_safety_limits(
            getattr(owner, "pipeline_policy", {}) or {},
            getattr(owner, "color_calibration_report", {}) or {},
        )
        original_adjustments = dict(ai_adjustments)
        ai_adjustments = clamp_ai_color_adjustments(
            ai_adjustments,
            already_applied=float(getattr(owner, "_saturation_boost_applied", 0.0)),
            limits=color_limits,
        )
        if ai_summary:
            owner.log.info(f"[AI] Plan summary: {ai_summary}")
        if ai_adjustments != original_adjustments:
            owner.log.info(
                "[AI] Stage4 color policy constrained Stage11 color adjustments "
                f"(limits={color_limits})"
            )
        owner.log.info(f"[AI] Normalized adjustments: {ai_adjustments}")

        ai_rgb = owner._apply_local_ai_adjustments(source_image_data, ai_adjustments)
        output_png_path = owner.work_dir / f"{output_name}.png"
        write_png_rgb16_func(output_png_path, ai_rgb)
        candidate_records.append(
            {
                "name": "local_adjusted",
                "file": f"{output_name}.png",
                "status": "generated",
                "adjustments": ai_adjustments,
            }
        )
        owner.log.info("[AI] Local AI-adjusted PNG written")

        owner.cmd_with_check("load", output_name)
        owner.cmd_with_check("save", output_fit_name)

        owner.cmd_with_check("load", source_name)
        source_shape = owner.siril.get_image_shape()
        owner.cmd_with_check("load", output_fit_name)
        output_shape = owner.siril.get_image_shape()
        if source_shape and output_shape and source_shape != output_shape:
            raise RuntimeError(
                f"AI output shape mismatch: source={source_shape}, ai={output_shape}"
            )

        blend_strength = ai_adjustments.get("blend_strength", owner.cfg.ai_strength)
        owner._blend_ai_images(source_name, output_fit_name, blended_name, blend_strength)
        owner.cmd_with_check("load", blended_name)
        blended_features = owner._measure_current_features()
        if blended_features is None:
            raise RuntimeError("failed to extract blended image features")

        quality_ok, quality_issues = owner._validate_ai_quality(
            source_features, blended_features
        )
        candidate_records.append(
            {
                "name": "blend_primary",
                "stem": blended_name,
                "status": "accepted" if quality_ok else "rejected",
                "blend_strength": blend_strength,
                "quality_issues": list(quality_issues),
            }
        )
        if not quality_ok:
            reduced_strength = max(0.05, round(blend_strength * 0.5, 4))
            if reduced_strength < blend_strength:
                owner.log.warn(
                    "[AI] Quality gate failed, retrying with lower blend strength "
                    f"{blend_strength:.3f}->{reduced_strength:.3f}"
                )
                blend_strength = reduced_strength
                owner._blend_ai_images(
                    source_name, output_fit_name, blended_name, blend_strength
                )
                owner.cmd_with_check("load", blended_name)
                blended_features = owner._measure_current_features()
                if blended_features is None:
                    raise RuntimeError(
                        "failed to extract blended image features after retry"
                    )
                quality_ok, quality_issues = owner._validate_ai_quality(
                    source_features, blended_features
                )
                candidate_records.append(
                    {
                        "name": "blend_reduced",
                        "stem": blended_name,
                        "status": "accepted" if quality_ok else "rejected",
                        "blend_strength": blend_strength,
                        "quality_issues": list(quality_issues),
                        "selected": True,
                    }
                )

        selected_candidate = (
            "blend_reduced"
            if any(item.get("name") == "blend_reduced" for item in candidate_records)
            else "blend_primary"
        )
        for item in candidate_records:
            item["selected"] = item.get("name") == selected_candidate

        review_report_path = None
        if (
            bool(getattr(owner.cfg, "review_bundle_enabled", True))
            and getattr(owner, "process_dir", None)
        ):
            try:
                owner.cmd_with_check("load", blended_name)
                blended_image_data = owner.siril.get_image_pixeldata(preview=False)
                if blended_image_data is None:
                    raise RuntimeError("failed to read Stage11 blended pixels for review")
                review = create_image_review_bundle(
                    source_image_data,
                    blended_image_data,
                    output_dir=owner.process_dir / "review_bundles" / "stage11_ai_postprocess",
                    stage_key="stage11_ai_postprocess",
                    source={
                        "before_path": str(owner.work_dir / f"{source_name}.fit"),
                        "after_path": str(owner.work_dir / f"{blended_name}.fit"),
                    },
                    context={
                        "target_type": (
                            owner._active_target_type()
                            if hasattr(owner, "_active_target_type")
                            else "generic_low_snr_safe"
                        ),
                        "adjustments": ai_adjustments,
                        "blend_strength": blend_strength,
                        "quality_ok": quality_ok,
                        "quality_issues": quality_issues,
                        "color_policy_limits": color_limits,
                    },
                    candidates=candidate_records,
                    selected_candidate=selected_candidate,
                )
                advisor_mode = str(
                    getattr(owner.cfg, "ai_advisor_mode", "text") or "text"
                ).strip().lower()
                try:
                    visual_verdict = owner._request_visual_acceptance(
                        "stage11_ai_postprocess",
                        review,
                    )
                    review = apply_visual_acceptance(
                        review,
                        visual_verdict,
                        advisor_mode=advisor_mode,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as visual_error:
                    owner.log.warn(
                        f"[AI] Stage11 multimodal visual acceptance unavailable: {visual_error}"
                    )
                    review = apply_visual_acceptance(
                        review,
                        None,
                        advisor_mode=advisor_mode,
                        error=str(visual_error)[:180],
                    )
                review_report_path = review.get("report_path")
                visual_acceptance = review.get("visual_review")
            except (OSError, RuntimeError, TypeError, ValueError) as review_error:
                owner.log.warn(f"[AI] Stage11 review bundle skipped: {review_error}")

        if not quality_ok:
            status = "degraded"
            message = (
                "quality gate rejected AI blend: "
                + "; ".join(quality_issues[:3])
            )
            try:
                owner._write_stage_json(
                    "stage11_quality.json",
                    {
                        "summary": ai_summary,
                        "adjustments": ai_adjustments,
                        "blend_strength": blend_strength,
                        "status": status,
                        "quality_issues": quality_issues,
                        "source_features": asdict(source_features),
                        "blended_features": asdict(blended_features),
                        "review_bundle": review_report_path,
                        "candidates": candidate_records,
                        "visual_acceptance": locals().get("visual_acceptance"),
                        "plan_parse_fallback": bool(
                            getattr(owner, "_ai_plan_parse_fallback", False)
                        ),
                        "plan_parse_fallback_reason": getattr(
                            owner, "_ai_plan_parse_fallback_reason", None
                        ),
                    },
                )
            except (OSError, RuntimeError, TypeError, ValueError) as json_error:
                owner.log.warn(f"[AI] stage11 quality JSON write failed: {json_error}")
        else:
            owner.cmd_with_check("load", blended_name)
            owner._saturation_boost_applied = float(
                getattr(owner, "_saturation_boost_applied", 0.0)
            ) + max(
                0.0,
                float(ai_adjustments.get("global_saturation_delta", 0.0))
                * float(blend_strength),
            )
            export_errors: list[str] = []

            try:
                owner.cmd_with_check("savetif", "result_processed_ai", "-astro")
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                export_errors.append(f"TIFF export failed: {e}")

            try:
                owner.cmd_with_check("savepng", "result_processed_ai")
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                export_errors.append(f"PNG export failed: {e}")

            try:
                owner.cmd_with_check("save", "result_final_ai")
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                export_errors.append(f"FITS export failed: {e}")

            if export_errors:
                status = "degraded"
                message = "; ".join(export_errors)
            else:
                owner.ai_outputs_generated = True
                message = (
                    "AI outputs exported successfully "
                    f"(blend={blend_strength:.3f})"
                )
                try:
                    owner._write_stage_json(
                        "stage11_quality.json",
                        {
                            "summary": ai_summary,
                            "adjustments": ai_adjustments,
                            "blend_strength": blend_strength,
                            "status": status,
                            "quality_issues": quality_issues,
                            "source_features": asdict(source_features),
                            "blended_features": asdict(blended_features),
                            "review_bundle": review_report_path,
                            "candidates": candidate_records,
                            "visual_acceptance": locals().get("visual_acceptance"),
                            "plan_parse_fallback": bool(
                                getattr(owner, "_ai_plan_parse_fallback", False)
                            ),
                            "plan_parse_fallback_reason": getattr(
                                owner, "_ai_plan_parse_fallback_reason", None
                            ),
                        },
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as json_error:
                    owner.log.warn(f"[AI] stage11 quality JSON write failed: {json_error}")
                plan_parse_fallback = bool(
                    getattr(owner, "_ai_plan_parse_fallback", False)
                )
                if plan_parse_fallback:
                    status = "degraded" if status == "ok" else status
                    fallback_reason = str(
                        getattr(owner, "_ai_plan_parse_fallback_reason", "")
                    ).strip()
                    if fallback_reason:
                        message = f"{message}; AI plan fallback: {fallback_reason}"
                    else:
                        message = f"{message}; AI plan fallback triggered"

    except (
        CommandError,
        DataError,
        SirilError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as e:
        status = "degraded"
        message = f"AI postprocess failed: {e}"
        owner.log.warn(message)
        if owner.cfg.debug_mode:
            owner.log.debug(traceback.format_exc())
    finally:
        if not owner.cfg.debug_mode:
            for path in temp_files:
                owner._safe_unlink(path)
        elapsed = owner.log.stage_end(stage_label)
        owner._record_stage(stage_label, status, elapsed, message)
