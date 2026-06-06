"""Stage 3 background extraction."""
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sirilpy.exceptions import CommandError, SirilError


EMISSION_NEBULA_TARGET_TYPES = {
    "emission_nebula",
    "emission_nebula_widefield",
    "bright_emission_reflection_nebula",
}
DEFAULT_DIFFUSE_OBJECT_AREA_MIN = 0.15
DEFAULT_DIFFUSE_NEBULOSITY_AREA_MIN = 0.18
DEFAULT_DIFFUSE_FAINT_STRUCTURE_MIN = 0.65
DEFAULT_FAINT_NEBULA_AREA_MIN = 0.10
DEFAULT_FAINT_NEBULA_STRUCTURE_MIN = 0.40
DEFAULT_NEBULA_PRESERVATION_WEIGHT = 1.6
DEFAULT_FAINT_NEBULA_PRESERVATION_WEIGHT_MAX = 2.5


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


def _stage3_policy_float(
    stage3_policy: Optional[Dict[str, Any]],
    key: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    policy = stage3_policy or {}
    try:
        value = float(policy.get(key, default) if isinstance(policy, dict) else default)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def _stage3_nebula_preservation_weight(
    diffuse_context: Optional[Dict[str, Any]] = None,
    stage3_policy: Optional[Dict[str, Any]] = None,
) -> float:
    base_weight = _stage3_policy_float(
        stage3_policy,
        "nebula_preservation_penalty_weight",
        DEFAULT_NEBULA_PRESERVATION_WEIGHT,
        minimum=0.0,
        maximum=10.0,
    )
    max_weight = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_preservation_penalty_weight_max",
        DEFAULT_FAINT_NEBULA_PRESERVATION_WEIGHT_MAX,
        minimum=0.0,
        maximum=10.0,
    )
    max_weight = max(base_weight, max_weight)
    context = diffuse_context or {}
    try:
        faint_structure_score = float(context.get("faint_structure_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        faint_structure_score = 0.0
    faint_min = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_structure_min",
        DEFAULT_FAINT_NEBULA_STRUCTURE_MIN,
        minimum=0.0,
        maximum=1.0,
    )
    if not context.get("faint_nebula_protection") and faint_structure_score <= faint_min:
        return base_weight
    span = max(1e-6, 1.0 - faint_min)
    t = max(0.0, min(1.0, (faint_structure_score - faint_min) / span))
    return base_weight + (max_weight - base_weight) * t


def _stage3_preservation_penalty(
    preservation: Dict[str, Any],
    *,
    diffuse_context: Optional[Dict[str, Any]] = None,
    stage3_policy: Optional[Dict[str, Any]] = None,
    nebula_weight: Optional[float] = None,
) -> float:
    if not isinstance(preservation, dict) or not preservation.get("available"):
        return 0.0

    penalty = 0.0
    nebula_weight = (
        float(nebula_weight)
        if nebula_weight is not None
        else _stage3_nebula_preservation_weight(diffuse_context, stage3_policy)
    )
    nebula_change = preservation.get("nebula_mean_change_ratio")
    if nebula_change is not None:
        try:
            change = max(0.0, float(nebula_change))
            penalty += max(0.0, change - 0.015) * nebula_weight
        except (TypeError, ValueError):
            pass

    star_retention = preservation.get("star_retention_ratio")
    if star_retention is not None:
        try:
            retention = float(star_retention)
            penalty += max(0.0, 1.0 - retention) * 0.45
        except (TypeError, ValueError):
            pass
    return penalty


def _stage3_candidate_sufficient(
    before: Dict[str, Any],
    after: Dict[str, Any],
    score: float,
    stage3_policy: Optional[Dict[str, Any]] = None,
) -> bool:
    before = before or {}
    after = after or {}
    max_score = _stage3_policy_float(stage3_policy, "sufficient_max_background_score", 0.34, minimum=0.0)
    dirty_max = _stage3_policy_float(stage3_policy, "sufficient_dirty_score_max", 0.32, minimum=0.0)
    dirty_gradient_ratio = _stage3_policy_float(
        stage3_policy,
        "sufficient_dirty_gradient_retention_ratio",
        0.88,
        minimum=0.0,
        maximum=1.5,
    )
    dirty_gradient_floor = _stage3_policy_float(
        stage3_policy,
        "sufficient_dirty_gradient_floor",
        0.04,
        minimum=0.0,
    )
    initial_gradient_min = _stage3_policy_float(
        stage3_policy,
        "sufficient_initial_gradient_min",
        0.06,
        minimum=0.0,
    )
    high_gradient_ratio = _stage3_policy_float(
        stage3_policy,
        "sufficient_high_gradient_retention_ratio",
        0.96,
        minimum=0.0,
        maximum=1.5,
    )
    max_std_growth = _stage3_policy_float(
        stage3_policy,
        "max_bg_std_growth",
        1.08,
        minimum=1.0,
        maximum=2.0,
    )
    color_shift_max = _stage3_policy_float(
        stage3_policy,
        "sufficient_color_shift_max",
        0.18,
        minimum=0.0,
        maximum=2.0,
    )
    dirty = float(after.get("dirty_background_score", 0.0) or 0.0)
    gradient_before = float(before.get("gradient_score", 0.0) or 0.0)
    gradient_after = float(after.get("gradient_score", 0.0) or 0.0)
    before_std = max(float(before.get("bg_std", 0.0) or 0.0), 1e-7)
    after_std = float(after.get("bg_std", 0.0) or 0.0)
    color_shift = _stage3_color_shift(before, after)
    return score <= max_score and not (
        dirty > dirty_max and gradient_after >= max(gradient_before * dirty_gradient_ratio, dirty_gradient_floor)
    ) and not (
        gradient_before >= initial_gradient_min and gradient_after > gradient_before * high_gradient_ratio
    ) and not (
        after_std / before_std > max_std_growth
    ) and not (
        color_shift > color_shift_max
    )


def _stage3_quote_arg(pipeline, value: Path | str) -> str:
    if hasattr(pipeline, "_quote_siril_arg"):
        return pipeline._quote_siril_arg(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _stage3_find_script(pipeline, *relative_candidates: str) -> Optional[Path]:
    if hasattr(pipeline, "_find_plugin_script"):
        try:
            found = pipeline._find_plugin_script(tuple(relative_candidates))
            if found is not None:
                return found
        except Exception as exc:
            if hasattr(pipeline, "log"):
                pipeline.log.debug(f"stage3 plugin script lookup skipped: {exc}")
    scripts_root = None
    if hasattr(pipeline, "_resolve_siril_scripts_root"):
        try:
            scripts_root = pipeline._resolve_siril_scripts_root()
        except Exception:
            scripts_root = None
    if scripts_root is None:
        plugin_dir = getattr(pipeline, "siril_plugin_dir", None)
        if plugin_dir:
            root = Path(plugin_dir)
            for candidate_root in (
                root / "vendor" / "siril-scripts",
                root / "vendor" / "siril-scripts" / "siril-scripts",
            ):
                if (candidate_root / "processing").is_dir():
                    scripts_root = candidate_root
                    break
    if scripts_root is None:
        return None
    for rel in relative_candidates:
        candidate = Path(scripts_root) / rel
        if candidate.is_file():
            return candidate
    return None


def _stage3_ensure_graxpert_bge_model(pipeline) -> bool:
    plugin_dir = getattr(pipeline, "siril_plugin_dir", None)
    if not plugin_dir:
        return False
    plugin_root = Path(plugin_dir)
    source_candidates = (
        plugin_root / "graxpert" / "bge-ai-models" / "model_v2_0_1" / "model.onnx",
        plugin_root / "bge-ai-models" / "model_v2_0_1" / "model.onnx",
        plugin_root / "model_v2_0_1.onnx",
        plugin_root / "downloads" / "model_v2_0_1.onnx",
    )
    source = next((candidate for candidate in source_candidates if candidate.is_file()), None)
    if source is None:
        if hasattr(pipeline, "log"):
            pipeline.log.warn("Stage3 GraXpert BGE model missing: model_v2_0_1.onnx")
        return False

    target = (
        Path(os.path.expanduser("~/Library/Application Support/GraXpert"))
        / "bge-ai-models"
        / "model_v2_0_1"
        / "model.onnx"
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
            if hasattr(pipeline, "log"):
                pipeline.log.info(f"Stage3 GraXpert BGE model installed: {target}")
        return target.is_file()
    except Exception as exc:
        if hasattr(pipeline, "log"):
            pipeline.log.warn(f"Stage3 GraXpert BGE model install failed: {exc}")
        return False


def _stage3_graxpert_candidates(pipeline=None) -> List[Tuple[str, Tuple[str, ...], str]]:
    if pipeline is not None:
        script = _stage3_find_script(
            pipeline,
            "processing/GraXpert-AI.py",
            "processing/GraXpert.py",
        )
        if script is not None and _stage3_ensure_graxpert_bge_model(pipeline):
            script_arg = _stage3_quote_arg(pipeline, script)
            return [
                (
                    "GraXpert",
                    ("pyscript", script_arg, "-bge", "-correction", "subtraction", "-smoothing", "0.50"),
                    "graxpert",
                ),
                (
                    "GraXpert-BGE",
                    ("pyscript", script_arg, "-bge", "-correction", "subtraction", "-smoothing", "0.35"),
                    "graxpert",
                ),
            ]
    return [
        ("GraXpert", ("gxp",), "graxpert"),
        ("GraXpert-BGE", ("graxpert",), "graxpert"),
    ]


def _stage3_theoretical_plugin_candidates(pipeline=None) -> List[Tuple[str, Tuple[str, ...], str]]:
    # Order by expected background-model quality first, then by automation risk.
    # Every candidate still goes through the same Stage3 quality gate.
    autobge = None
    if pipeline is not None:
        autobge = _stage3_find_script(pipeline, "processing/AutoBGE.py")
    if autobge is not None:
        script_arg = _stage3_quote_arg(pipeline, autobge)
        background_plugins = [
            ("ADBE", ("pyscript", script_arg, "-npoints", "80", "-polydegree", "2", "-rbfsmooth", "0.08"), "plugin"),
            ("DBE", ("pyscript", script_arg, "-npoints", "120", "-polydegree", "3", "-rbfsmooth", "0.12"), "plugin"),
            ("AutoDBE", ("pyscript", script_arg, "-npoints", "100", "-polydegree", "2", "-rbfsmooth", "0.10"), "plugin"),
        ]
    else:
        background_plugins = [
            ("ADBE", ("adbe",), "plugin"),
            ("DBE", ("dbe",), "plugin"),
            ("AutoDBE", ("autodbe",), "plugin"),
        ]
    return [
        *_stage3_graxpert_candidates(pipeline),
        *background_plugins,
        ("NOX", ("nox",), "plugin"),
        ("VeraLux NOX", ("veralux_nox",), "plugin"),
    ]


def _stage3_background_candidate_chain(
    pipeline,
    *,
    rbf_attempts: List[Tuple[str, Tuple[str, ...], str]],
    poly_attempt: List[Tuple[str, Tuple[str, ...], str]],
    poly_first: bool,
) -> Tuple[List[Tuple[str, Tuple[str, ...], str]], List[str], str]:
    plugin_attempts = _stage3_theoretical_plugin_candidates(pipeline)
    if poly_first:
        builtin_attempts = poly_attempt + rbf_attempts
        builtin_order_reason = "diffuse_signal_subsky_poly_before_rbf"
    else:
        builtin_attempts = rbf_attempts + poly_attempt
        builtin_order_reason = "default_rbf_before_poly"

    chain = (
        plugin_attempts[:5]
        + rbf_attempts
        + plugin_attempts[5:]
        + ([] if poly_first else poly_attempt)
    )
    if poly_first:
        chain = plugin_attempts[:5] + poly_attempt + rbf_attempts + plugin_attempts[5:]

    seen = set()
    ordered: List[Tuple[str, Tuple[str, ...], str]] = []
    for label, command, source in chain:
        key = (label, command)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((label, command, source))

    if hasattr(pipeline, "log"):
        pipeline.log.info(
            "[Stage3] Background extraction chain: "
            + " -> ".join(label for label, _command, _source in ordered)
        )
    return ordered, [record[0] for record in builtin_attempts], builtin_order_reason


def _stage3_is_graxpert_attempt(label: str, command: Tuple[str, ...], source: str) -> bool:
    if str(source).lower() == "graxpert":
        return True
    text = " ".join(str(part) for part in (label, *command)).lower()
    return "graxpert-ai.py" in text or "graxpert" in text or "gxp" in text


def _stage3_graxpert_runtime_error_reason(
    error: Exception,
    command: Tuple[str, ...],
) -> Optional[str]:
    text = str(error).strip()
    lowered = text.lower()
    command_text = " ".join(str(part) for part in command).lower()
    runtime_markers = (
        "graxpert-ai.py",
        "graxpert ai",
        "background extraction",
        "too many indices for array",
        "onnxruntime",
        "error initializing application",
        "traceback",
        "model_v2_0_1",
    )
    if "graxpert-ai.py" in command_text or any(marker in lowered for marker in runtime_markers):
        return f"graxpert_runtime_error: {text or type(error).__name__}"
    return None


def _stage3_try_background_command(
    pipeline,
    label: str,
    command: Tuple[str, ...],
    source: str,
) -> Tuple[bool, Optional[str]]:
    try:
        pipeline.cmd_with_check(*command, quiet=True)
        return True, None
    except Exception as exc:
        if _stage3_is_graxpert_attempt(label, command, source):
            reason = _stage3_graxpert_runtime_error_reason(exc, command)
            if reason is None:
                reason = f"graxpert_command_failed: {str(exc).strip() or type(exc).__name__}"
            return False, reason
        return False, f"command_failed: {str(exc).strip() or type(exc).__name__}"


def _stage3_metric(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    key: str,
) -> float:
    profile = target_profile or {}
    for section_name in ("object_stats", "image_stats", "color_stats", "star_stats"):
        section = profile.get(section_name) if isinstance(profile, dict) else None
        if isinstance(section, dict) and key in section:
            try:
                return float(section.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    if isinstance(adaptive, dict) and key in adaptive:
        try:
            return float(adaptive.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _stage3_diffuse_nebula_context(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    *,
    stage3_policy: Optional[Dict[str, Any]] = None,
    object_area_min: float = DEFAULT_DIFFUSE_OBJECT_AREA_MIN,
    nebulosity_area_min: float = DEFAULT_DIFFUSE_NEBULOSITY_AREA_MIN,
    faint_structure_min: float = DEFAULT_DIFFUSE_FAINT_STRUCTURE_MIN,
) -> Tuple[bool, Dict[str, Any]]:
    profile = target_profile or {}
    object_area_min = _stage3_policy_float(
        stage3_policy,
        "diffuse_nebula_object_area_min",
        object_area_min,
        minimum=0.0,
        maximum=1.0,
    )
    nebulosity_area_min = _stage3_policy_float(
        stage3_policy,
        "diffuse_nebula_nebulosity_area_min",
        nebulosity_area_min,
        minimum=0.0,
        maximum=1.0,
    )
    faint_structure_min = _stage3_policy_float(
        stage3_policy,
        "diffuse_nebula_faint_structure_min",
        faint_structure_min,
        minimum=0.0,
        maximum=1.0,
    )
    faint_nebula_area_min = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_nebulosity_area_min",
        DEFAULT_FAINT_NEBULA_AREA_MIN,
        minimum=0.0,
        maximum=1.0,
    )
    faint_nebula_structure_min = _stage3_policy_float(
        stage3_policy,
        "faint_nebula_structure_min",
        DEFAULT_FAINT_NEBULA_STRUCTURE_MIN,
        minimum=0.0,
        maximum=1.0,
    )
    target_type = str(profile.get("target_type") or "").strip().lower()
    features = profile.get("features") if isinstance(profile, dict) else {}
    feature_large = bool(
        isinstance(features, dict)
        and features.get("large_nebulosity")
    )
    object_area_ratio = _stage3_metric(profile, adaptive or {}, "object_area_ratio")
    nebulosity_area_ratio = _stage3_metric(profile, adaptive or {}, "nebulosity_area_ratio")
    faint_structure_score = _stage3_metric(profile, adaptive or {}, "faint_structure_score")
    is_emission = target_type in EMISSION_NEBULA_TARGET_TYPES
    emission_diffuse = bool(
        is_emission
        and (
            object_area_ratio >= object_area_min
            or nebulosity_area_ratio >= nebulosity_area_min
            or faint_structure_score >= faint_structure_min
            or feature_large
        )
    )
    faint_nebula_protection = bool(
        nebulosity_area_ratio > faint_nebula_area_min
        and faint_structure_score > faint_nebula_structure_min
    )
    pixel_signal_protection = bool(faint_nebula_protection or feature_large)
    diffuse = bool(emission_diffuse or pixel_signal_protection)
    if emission_diffuse:
        protection_reason = "emission_target_signal"
    elif faint_nebula_protection:
        protection_reason = "faint_nebula_signal"
    elif feature_large:
        protection_reason = "large_nebulosity_feature"
    else:
        protection_reason = "none"
    return diffuse, {
        "target_type": target_type,
        "is_emission_target": is_emission,
        "emission_diffuse": emission_diffuse,
        "faint_nebula_protection": faint_nebula_protection,
        "pixel_signal_protection": pixel_signal_protection,
        "protection_reason": protection_reason,
        "object_area_ratio": object_area_ratio,
        "nebulosity_area_ratio": nebulosity_area_ratio,
        "faint_structure_score": faint_structure_score,
        "large_nebulosity_feature": feature_large,
        "object_area_min": object_area_min,
        "nebulosity_area_min": nebulosity_area_min,
        "faint_structure_min": faint_structure_min,
        "faint_nebula_nebulosity_area_min": faint_nebula_area_min,
        "faint_nebula_structure_min": faint_nebula_structure_min,
    }


def _stage3_prefers_poly_first(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    *,
    stage3_policy: Optional[Dict[str, Any]] = None,
    object_area_min: float = DEFAULT_DIFFUSE_OBJECT_AREA_MIN,
) -> bool:
    diffuse, _context = _stage3_diffuse_nebula_context(
        target_profile,
        adaptive,
        stage3_policy=stage3_policy,
        object_area_min=object_area_min,
    )
    return diffuse


def _stage3_should_exhaust_builtin_search(
    target_profile: Dict[str, Any],
    adaptive: Dict[str, Any],
    stage3_policy: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    diffuse, context = _stage3_diffuse_nebula_context(
        target_profile,
        adaptive,
        stage3_policy=stage3_policy,
    )
    policy_requests_protection = bool(
        (stage3_policy or {}).get("reject_samples_on_nebula")
        or (stage3_policy or {}).get("protect_nebulosity")
    )
    pixel_signal_requests_protection = bool(
        context.get("faint_nebula_protection")
        or context.get("large_nebulosity_feature")
    )
    return bool(diffuse and (policy_requests_protection or pixel_signal_requests_protection)), context


def run_stage3_background_extraction(pipeline) -> None:
    """
    阶段 3: 背景提取
    - 按理论效果尝试 GraXpert / ADBE / DBE / subsky RBF / NOX 候选
    - 每个候选成功后执行质量门控，避免过度扣背景
    - 候选命令失败或未达到充分质量时，继续 fallback 到下一个候选
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
    graxpert_runtime_error = False
    graxpert_error_reasons: List[str] = []
    selected_label = ""
    builtin_order_reason = "default_rbf_before_poly"
    builtin_search_mode = "theoretical_effect_order"
    diffuse_context: Dict[str, Any] = {}

    def evaluate_attempts(
        attempts: List[Tuple[str, Tuple[str, ...], str]],
        *,
        phase: str,
        stop_on_sufficient: bool = True,
    ) -> bool:
        nonlocal baseline_saved, graxpert_runtime_error
        phase_sufficient = False
        for label, command, source in attempts:
            if baseline_saved:
                try:
                    pipeline.cmd_with_check("load", baseline_stem, quiet=True)
                except (CommandError, SirilError) as e:
                    pipeline.log.warn(f"failed to restore stage3 baseline: {e}")
                    baseline_saved = False

            pipeline.log.info(f"尝试背景提取: {label}")
            command_ok, failure_reason = _stage3_try_background_command(
                pipeline,
                label,
                command,
                source,
            )
            if not command_ok:
                is_graxpert = _stage3_is_graxpert_attempt(label, command, source)
                is_graxpert_runtime = bool(
                    is_graxpert
                    and failure_reason
                    and failure_reason.startswith("graxpert_runtime_error:")
                )
                if is_graxpert_runtime:
                    graxpert_runtime_error = True
                    graxpert_error_reasons.append(failure_reason)
                    pipeline.log.warn(
                        f"{label} 运行失败，自动切换到下一个背景提取候选: {failure_reason}"
                    )
                attempt_records.append(
                    {
                        "label": label,
                        "source": source,
                        "phase": phase,
                        "status": (
                            "graxpert_runtime_error"
                            if is_graxpert_runtime
                            else "command_failed"
                        ),
                        "failure_reason": failure_reason,
                        "fallback_triggered": bool(is_graxpert),
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
            base_candidate_score = _stage3_background_score(before_adaptive, after_adaptive_candidate)
            nebula_preservation_weight = _stage3_nebula_preservation_weight(
                diffuse_context,
                stage3_policy,
            )
            preservation_penalty = _stage3_preservation_penalty(
                preservation,
                diffuse_context=diffuse_context,
                stage3_policy=stage3_policy,
                nebula_weight=nebula_preservation_weight,
            )
            candidate_score = base_candidate_score + preservation_penalty
            sufficient = _stage3_candidate_sufficient(
                before_adaptive,
                after_adaptive_candidate,
                candidate_score,
                stage3_policy,
            )
            record.update(
                {
                    "candidate_stem": candidate_stem if candidate_saved else None,
                    "base_background_score": base_candidate_score,
                    "preservation_penalty": preservation_penalty,
                    "nebula_preservation_penalty_weight": nebula_preservation_weight,
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
                        "base_score": base_candidate_score,
                        "preservation_penalty": preservation_penalty,
                        "nebula_preservation_penalty_weight": nebula_preservation_weight,
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
                if stop_on_sufficient:
                    break
                pipeline.log.info(
                    f"{label} 已合格；继续评估剩余候选以保护弥散星云"
                )
                continue
            pipeline.log.info(
                f"背景提取候选通过但残余背景偏高，继续搜索: {label} score={candidate_score:.3f}"
            )
        return phase_sufficient

    rbf_attempts: List[Tuple[str, Tuple[str, ...], str]] = []
    for idx, cmd in enumerate(pipeline._stage3_subsky_rbf_candidates(), start=1):
        rbf_attempts.append((f"subsky-rbf-{idx}", cmd, "builtin"))
    poly_attempt = [("subsky-poly", ("subsky", "1"), "builtin")]

    target_profile = getattr(pipeline, "target_profile", {}) or {}
    poly_first = _stage3_prefers_poly_first(
        target_profile,
        before_adaptive,
        stage3_policy=stage3_policy,
    )
    exhaustive_builtin, diffuse_context = _stage3_should_exhaust_builtin_search(
        target_profile,
        before_adaptive,
        stage3_policy,
    )
    if exhaustive_builtin:
        builtin_search_mode = "theoretical_effect_order_with_diffuse_signal_protection"
        pipeline.log.info(
            "[Stage3] Diffuse nebulosity protection enabled; every candidate still requires preservation gate"
        )

    ordered_attempts, builtin_attempt_labels, builtin_order_reason = (
        _stage3_background_candidate_chain(
            pipeline,
            rbf_attempts=rbf_attempts,
            poly_attempt=poly_attempt,
            poly_first=poly_first,
        )
    )
    builtin_sufficient = evaluate_attempts(
        ordered_attempts,
        phase="theoretical_chain",
        stop_on_sufficient=True,
    )
    graxpert_attempted = any(
        record.get("source") == "graxpert" for record in attempt_records
    )
    graxpert_runtime_error = graxpert_runtime_error or any(
        record.get("status") == "graxpert_runtime_error"
        for record in attempt_records
    )

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
    max_bg_std_growth = _stage3_policy_float(
        stage3_policy,
        "max_bg_std_growth",
        1.10,
        minimum=1.0,
        maximum=2.0,
    )
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
                "graxpert_runtime_error": graxpert_runtime_error,
                "graxpert_error_reasons": graxpert_error_reasons,
                "fallback_triggered_by_graxpert_error": bool(
                    graxpert_runtime_error and selected_source != "graxpert"
                ),
                "builtin_order_reason": builtin_order_reason,
                "candidate_order": [record[0] for record in ordered_attempts],
                "builtin_candidate_order": builtin_attempt_labels,
                "builtin_search_mode": builtin_search_mode,
                "diffuse_nebula_context": diffuse_context,
                "protected_masks": [
                    name
                    for name, enabled in (
                        ("nebulosity_mask", stage3_policy.get("protect_nebulosity")),
                        ("faint_nebula_signal", diffuse_context.get("faint_nebula_protection")),
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
