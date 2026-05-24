"""Stage 3 background extraction."""
import re
from typing import Any, Dict, List, Tuple

from sirilpy.exceptions import CommandError, SirilError


def _stage3_candidate_stem(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip().lower()).strip("_")
    return f"stage3_candidate_{safe or 'background'}"


def _stage3_background_score(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> float:
    before = before or {}
    after = after or {}
    before_std = max(float(before.get("bg_std", 0.0) or 0.0), 1e-7)
    after_std = float(after.get("bg_std", 0.0) or 0.0)
    dirty = float(after.get("dirty_background_score", 0.0) or 0.0)
    gradient = float(after.get("gradient_score", 0.0) or 0.0)
    chroma = float(after.get("chroma_noise_score", 0.0) or 0.0)
    color_shift = _stage3_color_shift(before, after)
    std_growth = max(0.0, after_std / before_std - 1.0)
    return (
        dirty * 1.25
        + gradient * 0.85
        + chroma * 0.45
        + std_growth * 0.35
        + color_shift * 0.45
    )


def _stage3_color_shift(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> float:
    before = before or {}
    after = after or {}
    shifts: List[float] = []
    for key in ("red_dominance", "blue_dominance", "green_cast"):
        if key not in before or key not in after:
            continue
        before_value = float(before.get(key, 1.0) or 1.0)
        after_value = float(after.get(key, 1.0) or 1.0)
        shifts.append(abs(after_value - before_value))
    if "color_balance_score" in before and "color_balance_score" in after:
        shifts.append(
            max(
                0.0,
                float(before.get("color_balance_score", 1.0) or 1.0)
                - float(after.get("color_balance_score", 1.0) or 1.0),
            )
        )
    return max(shifts) if shifts else 0.0


def _stage3_candidate_sufficient(
    before: Dict[str, Any],
    after: Dict[str, Any],
    score: float,
) -> bool:
    before = before or {}
    after = after or {}
    dirty = float(after.get("dirty_background_score", 0.0) or 0.0)
    gradient_before = float(before.get("gradient_score", 0.0) or 0.0)
    gradient_after = float(after.get("gradient_score", 0.0) or 0.0)
    before_std = max(float(before.get("bg_std", 0.0) or 0.0), 1e-7)
    after_std = float(after.get("bg_std", 0.0) or 0.0)
    color_shift = _stage3_color_shift(before, after)
    return score <= 0.34 and not (
        dirty > 0.32 and gradient_after >= max(gradient_before * 0.88, 0.04)
    ) and not (
        gradient_before >= 0.06 and gradient_after > gradient_before * 0.96
    ) and not (
        after_std / before_std > 1.08
    ) and not (
        color_shift > 0.18
    )


def _stage3_graxpert_candidates() -> List[Tuple[str, Tuple[str, ...], str]]:
    return [
        ("GraXpert", ("gxp",), "graxpert"),
        ("GraXpert-BGE", ("graxpert",), "graxpert"),
    ]


def _stage3_prefers_poly_first(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    *,
    object_area_min: float = 0.35,
) -> bool:
    profile = target_profile or {}
    target_type = str(profile.get("target_type") or "").strip().lower()
    emission_types = {
        "emission_nebula",
        "emission_nebula_widefield",
        "bright_emission_reflection_nebula",
    }
    if target_type not in emission_types:
        return False

    object_stats = profile.get("object_stats") if isinstance(profile, dict) else {}
    object_area = None
    if isinstance(object_stats, dict):
        object_area = object_stats.get("object_area_ratio")
    if object_area is None:
        object_area = (adaptive or {}).get("object_area_ratio")
    try:
        object_area_ratio = float(object_area or 0.0)
    except (TypeError, ValueError):
        object_area_ratio = 0.0
    return object_area_ratio >= object_area_min


def run_stage3_background_extraction(pipeline) -> None:
    """
    阶段 3: 背景提取
    - 先尝试内置 subsky RBF / 线性多项式候选
    - 每个候选成功后执行质量门控，避免过度扣背景
    - 内置候选未达到充分质量时，再尝试 GraXpert 背景提取补救
    """
    pipeline.log.stage_start("阶段 3: 背景提取")
    bg_ok = False
    selected_source = ""
    preflight_message = ""
    if hasattr(pipeline, "_run_target_profile_preflight"):
        preflight_message = pipeline._run_target_profile_preflight(
            source="Stage3 preflight",
            metadata_candidates=("stage2_corrected", getattr(pipeline, "source_file", None)),
            preview_name="stage3_target_preview.png",
        )
    stage_message = preflight_message
    policy = getattr(pipeline, "pipeline_policy", {}) or {}
    policy_name = policy.get("policy_name", "generic_low_snr_safe") if isinstance(policy, dict) else "generic_low_snr_safe"
    stage3_policy = policy.get("stage3_background", {}) if isinstance(policy, dict) else {}
    pipeline.log.info(
        "[Stage3] Background policy: "
        f"policy={policy_name} protect_nebulosity={bool(stage3_policy.get('protect_nebulosity', False))} "
        f"model={','.join(stage3_policy.get('model_priority', []) or [])}"
    )

    baseline_stem = "stage3_bg_input"
    baseline_saved = False
    try:
        pipeline.cmd_with_check("save", baseline_stem)
        baseline_saved = True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(f"stage3 baseline save failed, fallback without rollback: {e}")

    before_feat = pipeline._stage3_measure_features("before")
    before_image = None
    try:
        before_image = pipeline.siril.get_image_pixeldata(preview=False)
    except Exception as e:
        pipeline.log.debug(f"stage3 baseline image sampling skipped: {e}")
    before_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )

    attempt_records: List[Dict[str, Any]] = []
    selected_preservation: Dict[str, Any] = {}
    accepted_candidates: List[Dict[str, Any]] = []
    builtin_sufficient = False
    graxpert_attempted = False
    selected_label = ""
    builtin_order_reason = "default_rbf_first"

    def evaluate_attempts(
        attempts: List[Tuple[str, Tuple[str, ...], str]],
        *,
        phase: str,
    ) -> bool:
        nonlocal baseline_saved
        phase_sufficient = False
        for label, command, source in attempts:
            if baseline_saved:
                try:
                    pipeline.cmd_with_check("load", baseline_stem, quiet=True)
                except (CommandError, SirilError) as e:
                    pipeline.log.warn(f"failed to restore stage3 baseline: {e}")
                    baseline_saved = False

            pipeline.log.info(f"尝试背景提取: {label}")
            if not pipeline._try_cmd(*command):
                attempt_records.append(
                    {
                        "label": label,
                        "source": source,
                        "phase": phase,
                        "status": "command_failed",
                    }
                )
                continue

            after_feat = pipeline._stage3_measure_features(label)
            after_image = None
            try:
                after_image = pipeline.siril.get_image_pixeldata(preview=False)
            except Exception as e:
                pipeline.log.debug(f"stage3 candidate image sampling skipped ({label}): {e}")
            preservation = pipeline._stage3_signal_preservation_metrics(
                before_image,
                after_image,
            )
            gate_ok, gate_msg = pipeline._stage3_quality_gate(
                before_feat,
                after_feat,
                preservation,
            )
            after_adaptive_candidate = (
                pipeline._adaptive_features_current()
                if hasattr(pipeline, "_adaptive_features_current")
                else {}
            )
            color_shift = _stage3_color_shift(before_adaptive, after_adaptive_candidate)
            record = {
                "label": label,
                "source": source,
                "phase": phase,
                "status": "accepted" if gate_ok else "rejected",
                "quality_message": gate_msg,
                "preservation": preservation,
                "after_adaptive": after_adaptive_candidate,
                "color_shift": color_shift,
            }
            if not gate_ok:
                attempt_records.append(record)
                pipeline.log.warn(
                    f"{label} rejected by quality gate, try next candidate: {gate_msg}"
                )
                continue

            candidate_stem = _stage3_candidate_stem(label)
            candidate_saved = pipeline._save_stage_output(candidate_stem)
            candidate_score = _stage3_background_score(before_adaptive, after_adaptive_candidate)
            sufficient = _stage3_candidate_sufficient(
                before_adaptive,
                after_adaptive_candidate,
                candidate_score,
            )
            record.update(
                {
                    "candidate_stem": candidate_stem if candidate_saved else None,
                    "background_score": candidate_score,
                    "sufficient": sufficient,
                }
            )
            attempt_records.append(record)
            if candidate_saved:
                accepted_candidates.append(
                    {
                        "label": label,
                        "source": source,
                        "phase": phase,
                        "stem": candidate_stem,
                        "score": candidate_score,
                        "quality_message": gate_msg,
                        "preservation": preservation,
                        "after_adaptive": after_adaptive_candidate,
                        "color_shift": color_shift,
                        "sufficient": sufficient,
                    }
                )
            if sufficient and candidate_saved:
                pipeline.log.info(
                    f"背景提取候选足够干净: {label} score={candidate_score:.3f}"
                )
                phase_sufficient = True
                break
            pipeline.log.info(
                f"背景提取候选通过但残余背景偏高，继续搜索: {label} score={candidate_score:.3f}"
            )
        return phase_sufficient

    rbf_attempts: List[Tuple[str, Tuple[str, ...], str]] = []
    for idx, cmd in enumerate(pipeline._stage3_subsky_rbf_candidates(), start=1):
        rbf_attempts.append((f"subsky-rbf-{idx}", cmd, "builtin"))
    poly_attempt = [("subsky-poly", ("subsky", "1"), "builtin")]

    target_profile = getattr(pipeline, "target_profile", {}) or {}
    poly_first = _stage3_prefers_poly_first(target_profile, before_adaptive)
    if poly_first:
        builtin_attempts = poly_attempt + rbf_attempts
        builtin_order_reason = "large_emission_nebula_poly_first"
        pipeline.log.info(
            "[Stage3] Large emission nebula detected; trying subsky 1 before RBF"
        )
    else:
        builtin_attempts = rbf_attempts + poly_attempt

    builtin_sufficient = evaluate_attempts(builtin_attempts, phase="builtin")

    if not builtin_sufficient:
        pipeline.log.warn(
            "内置 subsky/RBF 背景提取未达到充分质量，尝试 GraXpert 背景提取补救"
        )
        graxpert_attempted = True
        evaluate_attempts(_stage3_graxpert_candidates(), phase="graxpert_fallback")

    if accepted_candidates:
        selected = min(accepted_candidates, key=lambda item: float(item.get("score", 999.0)))
        try:
            pipeline.cmd_with_check("load", str(selected["stem"]))
        except (CommandError, SirilError) as e:
            pipeline.log.warn(f"failed to load best stage3 candidate, keeping current image: {e}")
        bg_ok = True
        selected_source = str(selected.get("source") or "")
        selected_label = str(selected.get("label") or "")
        selected_preservation = selected.get("preservation") or {}
        selected_message = (
            f"method={selected.get('label')}; {selected.get('quality_message')}; "
            f"background_score={float(selected.get('score', 0.0)):.3f}"
        )
        stage_message = (
            f"{preflight_message}; {selected_message}"
            if preflight_message
            else selected_message
        )
        if selected_source == "plugin":
            pipeline.workflow_command_used["背景提取插件链"] = str(selected.get("label"))
        elif selected_source == "graxpert":
            pipeline.workflow_command_used["GraXpert 背景提取"] = str(selected.get("label"))
        pipeline.log.info(
            "背景提取最终选择: "
            f"{selected.get('label')} score={float(selected.get('score', 0.0)):.3f}"
        )
    elif not bg_ok:
        pipeline.log.error("背景提取完全失败，图像可能有梯度残留")

    stage_saved = pipeline._save_stage_output("stage3_bgremoved")
    after_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )
    max_bg_std_growth = float(stage3_policy.get("max_bg_std_growth", 1.10) or 1.10)
    fallback_warning = False
    if before_adaptive and after_adaptive:
        before_std = max(float(before_adaptive.get("bg_std", 0.0) or 0.0), 1e-7)
        after_std = float(after_adaptive.get("bg_std", 0.0) or 0.0)
        dirty = float(after_adaptive.get("dirty_background_score", 0.0) or 0.0)
        gradient_before = float(before_adaptive.get("gradient_score", 0.0) or 0.0)
        gradient_after = float(after_adaptive.get("gradient_score", 0.0) or 0.0)
        if after_std / before_std > max_bg_std_growth or (
            dirty > 0.35 and gradient_after >= gradient_before * 0.92
        ):
            fallback_warning = True
            warning_msg = (
                "background improvement limited "
                f"(dirty={dirty:.3f}, std_growth={after_std / before_std:.3f})"
            )
            pipeline.log.warn(f"[Stage3] {warning_msg}")
            stage_message = f"{stage_message}; {warning_msg}" if stage_message else warning_msg
    if hasattr(pipeline, "_write_stage_json"):
        pipeline._write_stage_json(
            "background_quality_report.json",
            {
                "stage": "stage3_background",
                "policy": policy_name,
                "model_used": selected_label or None,
                "graxpert_attempted": graxpert_attempted,
                "builtin_order_reason": builtin_order_reason,
                "builtin_candidate_order": [record[0] for record in builtin_attempts],
                "protected_masks": [
                    name
                    for name, enabled in (
                        ("nebulosity_mask", stage3_policy.get("protect_nebulosity")),
                        ("bright_core_mask", stage3_policy.get("protect_bright_core")),
                        ("star_halo_mask", stage3_policy.get("protect_star_halo")),
                        ("outer_halo_mask", stage3_policy.get("protect_outer_halo")),
                        ("dark_structure_mask", stage3_policy.get("protect_dark_structure")),
                    )
                    if enabled
                ],
                "before": before_adaptive,
                "after": after_adaptive,
                "attempts": attempt_records,
                "selected_preservation": selected_preservation,
                "quality": "warning" if fallback_warning else ("ok" if bg_ok else "degraded"),
                "fallback_used": selected_source == "graxpert" or not bg_ok or fallback_warning,
            },
        )
    if not stage_saved:
        stage_message = (
            f"{stage_message}; stage3 输出保存失败"
            if stage_message
            else "stage3 输出保存失败"
        )

    elapsed = pipeline.log.stage_end("阶段 3: 背景提取")
    if bg_ok:
        status = "ok" if stage_saved else "degraded"
        pipeline._record_stage("阶段 3: 背景提取", status, elapsed, stage_message)
        if selected_source == "builtin":
            pipeline.log.info("阶段3按策略使用内置 subsky/RBF 背景提取")
    else:
        degrade_message = "背景提取失败，图像可能有梯度残留"
        if not stage_saved:
            degrade_message += "；stage3 输出保存失败"
        pipeline._record_stage(
            "阶段 3: 背景提取",
            "degraded",
            elapsed,
            degrade_message,
        )
