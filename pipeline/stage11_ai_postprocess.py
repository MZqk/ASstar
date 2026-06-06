#!/usr/bin/env python3
"""Stage11 AI postprocess runner extracted from the main pipeline."""

from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
    owner.log.stage_start("阶段 11: AI 后期美化")
    status = "ok"
    message = ""
    owner.ai_outputs_generated = False

    if not owner.cfg.ai_post_enabled:
        elapsed = owner.log.stage_end("阶段 11: AI 后期美化")
        owner._record_stage(
            "阶段 11: AI 后期美化",
            "skipped",
            elapsed,
            "SEESTAR_AI_ENABLED not enabled",
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
        elapsed = owner.log.stage_end("阶段 11: AI 后期美化")
        owner._record_stage(
            "阶段 11: AI 后期美化",
            "skipped",
            elapsed,
            f"missing required env: {', '.join(missing_env)}",
        )
        return

    source_name = "stage11_ai_source"
    output_name = "stage11_ai_output"
    output_fit_name = "stage11_ai_output_fit"
    blended_name = "stage11_ai_blended"

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
        if ai_summary:
            owner.log.info(f"[AI] Plan summary: {ai_summary}")
        owner.log.info(f"[AI] Normalized adjustments: {ai_adjustments}")

        ai_rgb = owner._apply_local_ai_adjustments(source_image_data, ai_adjustments)
        output_png_path = owner.work_dir / f"{output_name}.png"
        write_png_rgb16_func(output_png_path, ai_rgb)
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
        elapsed = owner.log.stage_end("阶段 11: AI 后期美化")
        owner._record_stage("阶段 11: AI 后期美化", status, elapsed, message)
