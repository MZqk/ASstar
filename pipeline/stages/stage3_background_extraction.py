"""Stage 3 background extraction."""
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from background_sampling import (
    STAGE3_PROCESS_EVIDENCE_SCHEMA,
    assess_background_process,
    assess_compound_background_validation,
    assess_single_background_validation,
    assess_target_fidelity,
    analyze_directional_pattern_noise,
    background_span_standard_error,
    build_safe_background_samples,
    measure_background_validation,
    pattern_candidate_gate,
    select_background_route,
    split_background_sample_points,
)
from models import PipelineStage
from sirilpy.exceptions import CommandError, SirilError
from stage3_contract import (
    STAGE3_ALGORITHM_CONTRACT_VERSION,
    STAGE3_BACKGROUND_QUALITY_SCHEMA,
    STAGE3_BACKGROUND_SCORE_WEIGHTS,
    STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT,
    STAGE3_FINAL_DIRTY_WARNING_MIN,
    STAGE3_FINAL_GRADIENT_RETENTION_WARNING,
    STAGE3_MIN_AXIS_SPAN_RATIO,
    STAGE3_MIN_SPATIAL_GRID_CELLS,
    STAGE3_MIN_SPATIAL_QUADRANTS,
    STAGE3_MIN_VALIDATION_PATCHES,
    STAGE3_SIGNIFICANCE_SIGMA,
    stage3_static_contract_manifest,
)


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
    return float(_stage3_background_score_components(before, after)["total"])


def _stage3_background_score_components(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    """Return an auditable decomposition of the legacy ranking score."""
    before = before or {}
    after = after or {}
    before_std = max(float(before.get("bg_std", 0.0) or 0.0), 1e-7)
    after_std = float(after.get("bg_std", 0.0) or 0.0)
    dirty = float(after.get("dirty_background_score", 0.0) or 0.0)
    gradient = float(after.get("gradient_score", 0.0) or 0.0)
    chroma = float(after.get("chroma_noise_score", 0.0) or 0.0)
    color_shift = _stage3_color_shift(before, after)
    std_growth = max(0.0, after_std / before_std - 1.0)
    components = {
        "dirty_background_score": dirty,
        "gradient_score": gradient,
        "chroma_noise_score": chroma,
        "bg_std_growth": std_growth,
        "color_shift": color_shift,
    }
    weighted = {
        name: float(value) * float(STAGE3_BACKGROUND_SCORE_WEIGHTS[name])
        for name, value in components.items()
    }
    return {
        "components": components,
        "weights": dict(STAGE3_BACKGROUND_SCORE_WEIGHTS),
        "weighted_components": weighted,
        "total": sum(weighted.values()),
    }


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
    target_flux_retention = preservation.get("target_flux_retention_ratio")
    if target_flux_retention is not None:
        try:
            retention = float(target_flux_retention)
            penalty += max(0.0, 1.0 - retention) * nebula_weight
        except (TypeError, ValueError):
            pass
    else:
        # Compatibility for older callers that have not supplied held-out sky
        # referenced target flux metrics.
        nebula_change = preservation.get("nebula_mean_change_ratio")
        if nebula_change is not None:
            try:
                change = max(0.0, float(nebula_change))
                penalty += max(0.0, change - 0.015) * nebula_weight
            except (TypeError, ValueError):
                pass

    morphology = preservation.get("target_morphology_correlation")
    if morphology is not None:
        try:
            penalty += max(0.0, 1.0 - float(morphology)) * 0.75
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
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if hasattr(pipeline, "log"):
                pipeline.log.debug(f"stage3 plugin script lookup skipped: {exc}")
    scripts_root = None
    if hasattr(pipeline, "_resolve_siril_scripts_root"):
        try:
            scripts_root = pipeline._resolve_siril_scripts_root()
        except (OSError, RuntimeError, TypeError, ValueError):
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
    except (OSError, RuntimeError, shutil.Error) as exc:
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
        builtin_order_reason = "diffuse_signal_safe_samples_poly_before_rbf"
    else:
        builtin_attempts = rbf_attempts + poly_attempt
        builtin_order_reason = "safe_samples_rbf_before_poly"

    # Custom, audited samples are the primary Stage 3 contract. Plugins remain
    # bounded fallbacks if Siril cannot consume the samples or the candidate is
    # rejected by the same preservation gate.
    chain = builtin_attempts + plugin_attempts

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
        "onnx",
        "onnxruntime",
        "error initializing application",
        "traceback",
        "model_v2_0_1",
    )
    if "graxpert-ai.py" in command_text or any(marker in lowered for marker in runtime_markers):
        return f"graxpert_runtime_error: {text or type(error).__name__}"
    return None


def _stage3_pyscript_path(command: Tuple[str, ...]) -> Optional[Path]:
    if len(command) < 2 or str(command[0]).lower() != "pyscript":
        return None
    raw_path = str(command[1]).strip()
    if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] == '"':
        raw_path = raw_path[1:-1]
    raw_path = raw_path.replace('\\"', '"').replace("\\\\", "\\")
    return Path(raw_path) if raw_path else None


def _stage3_image_fingerprint(pipeline) -> Optional[str]:
    if not hasattr(pipeline, "_current_image_fingerprint"):
        return None
    try:
        return pipeline._current_image_fingerprint()
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if hasattr(pipeline, "log"):
            pipeline.log.debug(f"stage3 image fingerprint skipped: {exc}")
        return None


def _stage3_try_background_command(
    pipeline,
    label: str,
    command: Tuple[str, ...],
    source: str,
) -> Tuple[bool, Optional[str]]:
    is_graxpert = _stage3_is_graxpert_attempt(label, command, source)
    script_path = _stage3_pyscript_path(command)
    runtime_error_prefix = (
        "graxpert_runtime_error" if is_graxpert else "plugin_runtime_error"
    )
    if script_path is not None and hasattr(
        pipeline,
        "_validate_plugin_script_prerequisites",
    ):
        try:
            prerequisites_ok, prerequisites_reason = (
                pipeline._validate_plugin_script_prerequisites(script_path)
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            prerequisites_ok = False
            prerequisites_reason = f"prerequisite check failed: {exc}"
        if not prerequisites_ok:
            return False, (
                f"{runtime_error_prefix}: prerequisites unavailable: "
                f"{prerequisites_reason or 'unknown reason'}"
            )

    before_fingerprint = (
        _stage3_image_fingerprint(pipeline) if script_path is not None else None
    )
    try:
        pipeline.cmd_with_check(*command, quiet=True)
        after_fingerprint = (
            _stage3_image_fingerprint(pipeline) if script_path is not None else None
        )
        if (
            before_fingerprint
            and after_fingerprint
            and before_fingerprint == after_fingerprint
        ):
            return False, (
                f"{runtime_error_prefix}: command returned success "
                "but image did not change"
            )
        return True, None
    except (CommandError, SirilError, OSError, RuntimeError) as exc:
        if is_graxpert:
            reason = _stage3_graxpert_runtime_error_reason(exc, command)
            if reason is None:
                reason = f"graxpert_command_failed: {str(exc).strip() or type(exc).__name__}"
            return False, reason
        return False, f"command_failed: {str(exc).strip() or type(exc).__name__}"


def _stage3_cfg_float(
    pipeline,
    name: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    try:
        value = float(getattr(pipeline.cfg, name, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def _stage3_cfg_int(
    pipeline,
    name: str,
    default: int,
    lower: int,
    upper: int,
) -> int:
    try:
        value = int(getattr(pipeline.cfg, name, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def _stage3_decision_thresholds(
    pipeline,
    stage3_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze the effective Stage 3 decision and ranking thresholds."""
    contract = stage3_static_contract_manifest()
    return {
        **contract,
        "safe_samples": {
            "target_count": _stage3_cfg_int(
                pipeline, "stage3_safe_sample_target_count", 40, 16, 64
            ),
            "minimum_count": _stage3_cfg_int(
                pipeline, "stage3_safe_sample_min_count", 16, 12, 48
            ),
            "patch_radius": _stage3_cfg_int(
                pipeline, "stage3_safe_sample_patch_radius", 12, 4, 24
            ),
            "brightness_quantile_max": _stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_brightness_quantile_max",
                0.70,
                0.50,
                0.85,
            ),
            "texture_quantile_max": _stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_texture_quantile_max",
                0.55,
                0.25,
                0.75,
            ),
        },
        "candidate_sufficiency": {
            "maximum_background_score": _stage3_policy_float(
                stage3_policy,
                "sufficient_max_background_score",
                0.34,
                minimum=0.0,
            ),
            "maximum_dirty_score": _stage3_policy_float(
                stage3_policy,
                "sufficient_dirty_score_max",
                0.32,
                minimum=0.0,
            ),
            "maximum_bg_std_growth": _stage3_policy_float(
                stage3_policy,
                "max_bg_std_growth",
                1.08,
                minimum=1.0,
                maximum=2.0,
            ),
            "maximum_color_shift": _stage3_policy_float(
                stage3_policy,
                "sufficient_color_shift_max",
                0.18,
                minimum=0.0,
                maximum=2.0,
            ),
        },
        "compound_candidate": {
            "minimum_span_improvement_ratio": _stage3_cfg_float(
                pipeline,
                "stage3_compound_validation_improvement_min",
                0.10,
                0.10,
                0.40,
            ),
            "minimum_score_absolute_improvement": _stage3_cfg_float(
                pipeline,
                "stage3_compound_score_abs_improvement_min",
                0.03,
                0.03,
                0.15,
            ),
            "minimum_score_relative_improvement": _stage3_cfg_float(
                pipeline,
                "stage3_compound_score_rel_improvement_min",
                0.10,
                0.10,
                0.40,
            ),
        },
        "directional_pattern": {
            "pattern_score_min": _stage3_cfg_float(
                pipeline, "stage3_pattern_score_min", 0.55, 0.25, 0.90
            ),
            "walking_noise_score_min": _stage3_cfg_float(
                pipeline,
                "stage3_walking_noise_score_min",
                0.50,
                0.25,
                0.90,
            ),
            "maximum_pattern_score_growth": _stage3_cfg_float(
                pipeline,
                "stage3_pattern_score_growth_max",
                0.12,
                0.02,
                0.40,
            ),
        },
        "final_output_revalidation": {
            "enforced_for_profiled_runs": True,
            "heldout_span_improvement": "strictly_greater_than_three_sigma",
            "background_rms": "not_worse_beyond_three_sigma",
            "maximum_bg_std_growth_warning": _stage3_policy_float(
                stage3_policy,
                "max_bg_std_growth",
                1.10,
                minimum=1.0,
                maximum=2.0,
            ),
        },
        "statistical_selection_shadow": {
            "method": "pareto_dense_rank_sum_v1",
            "runtime_selection_affected": False,
            "lower_is_better": [
                "residual_span_significance_sigma",
                "target_preservation_penalty",
                "directional_pattern_penalty",
            ],
        },
    }


def _stage3_statistical_shadow_selection(
    candidates: List[Dict[str, Any]],
    current_selected: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare a statistical/Pareto order without changing runtime selection."""
    current_label = str((current_selected or {}).get("label") or "") or None
    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        validation = candidate.get("validation") or {}
        gate = candidate.get("validation_gate") or {}
        if validation.get("status") != "ready" or gate.get("accepted") is False:
            continue
        try:
            residual_span = float(validation["robust_span"])
            span_standard_error = float(
                background_span_standard_error(validation)
            )
            preservation_penalty = float(
                candidate.get("preservation_penalty", 0.0) or 0.0
            )
            pattern_penalty = float(
                candidate.get("directional_pattern_penalty", 0.0) or 0.0
            )
            runtime_score = float(candidate.get("score", 999.0))
        except (KeyError, TypeError, ValueError):
            continue
        values = (
            residual_span,
            span_standard_error,
            preservation_penalty,
            pattern_penalty,
            runtime_score,
        )
        if not all(math.isfinite(value) for value in values):
            continue
        if span_standard_error <= 0.0:
            continue
        uncertainty_3sigma = gate.get("sampling_uncertainty_3sigma")
        span_improvement = gate.get("span_improvement")
        improvement_sigma = None
        try:
            uncertainty_value = float(uncertainty_3sigma)
            improvement_value = float(span_improvement)
            if uncertainty_value > 0.0:
                improvement_sigma = (
                    STAGE3_SIGNIFICANCE_SIGMA
                    * improvement_value
                    / uncertainty_value
                )
        except (TypeError, ValueError):
            pass
        rows.append(
            {
                "candidate_index": index,
                "label": str(candidate.get("label") or f"candidate_{index + 1}"),
                "source": candidate.get("source"),
                "runtime_selected": str(candidate.get("label") or "")
                == current_label,
                "residual_span": residual_span,
                "residual_span_standard_error": span_standard_error,
                "residual_span_significance_sigma": (
                    residual_span / span_standard_error
                ),
                "span_improvement_sigma": improvement_sigma,
                "target_preservation_penalty": preservation_penalty,
                "directional_pattern_penalty": pattern_penalty,
                "runtime_background_score": runtime_score,
            }
        )

    if not rows:
        return {
            "status": "unavailable",
            "method": "pareto_dense_rank_sum_v1",
            "runtime_selection_affected": False,
            "current_runtime_candidate": current_label,
            "reason": "no candidate has complete held-out statistical evidence",
            "candidates": [],
        }

    criteria = (
        "residual_span_significance_sigma",
        "target_preservation_penalty",
        "directional_pattern_penalty",
    )

    def dominates(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        return bool(
            all(float(left[key]) <= float(right[key]) for key in criteria)
            and any(float(left[key]) < float(right[key]) for key in criteria)
        )

    pareto_labels = [
        str(row["label"])
        for row in rows
        if not any(
            other is not row and dominates(other, row)
            for other in rows
        )
    ]
    for key in criteria:
        ordered_values = sorted({float(row[key]) for row in rows})
        ranks = {value: rank for rank, value in enumerate(ordered_values)}
        for row in rows:
            row[f"{key}_rank"] = ranks[float(row[key])]
    for row in rows:
        row["balanced_rank_sum"] = sum(
            int(row[f"{key}_rank"])
            for key in criteria
        )
        row["pareto_front"] = str(row["label"]) in pareto_labels

    shadow_order = sorted(
        rows,
        key=lambda row: (
            int(row["balanced_rank_sum"]),
            not bool(row["pareto_front"]),
            float(row["residual_span_significance_sigma"]),
            float(row["runtime_background_score"]),
            str(row["label"]),
        ),
    )
    proposed_label = str(shadow_order[0]["label"])
    return {
        "status": "ready",
        "method": "pareto_dense_rank_sum_v1",
        "runtime_selection_affected": False,
        "current_runtime_candidate": current_label,
        "shadow_recommended_candidate": proposed_label,
        "selection_would_change": bool(
            current_label and proposed_label != current_label
        ),
        "pareto_front": pareto_labels,
        "statistical_order": [str(row["label"]) for row in shadow_order],
        "candidates": shadow_order,
    }


def _stage3_final_output_validation(
    pipeline,
    *,
    baseline_validation: Dict[str, Any],
    validation_points: List[Tuple[float, float]],
    patch_radius: int,
    minimum_count: int,
    enforced: bool,
) -> Dict[str, Any]:
    """Re-run the held-out hard gate on the reloaded, saved final buffer."""
    report: Dict[str, Any] = {
        "status": "not_enforced" if not enforced else "running",
        "enforced": bool(enforced),
        "evidence_basis": "reloaded_saved_output_heldout_sky_three_sigma",
    }
    if not enforced:
        report["reason"] = "legacy caller has no profiled production input"
        return report
    try:
        image = pipeline.siril.get_image_pixeldata(preview=False)
        candidate_validation = measure_background_validation(
            image,
            validation_points,
            patch_radius=patch_radius,
            minimum_count=minimum_count,
            value_scale=baseline_validation.get("value_scale"),
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
        candidate_validation = {
            "status": "unavailable",
            "reason": str(error),
        }
    accepted, gate = assess_single_background_validation(
        baseline_validation,
        candidate_validation,
    )
    report.update(
        {
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "validation": candidate_validation,
            "validation_gate": gate,
        }
    )
    return report


def _stage3_compound_target_guard(
    target_profile: Dict[str, Any],
    diffuse_context: Dict[str, Any],
    stage3_policy: Dict[str, Any],
    noise_route: Dict[str, Any],
) -> Dict[str, Any]:
    """Hard-disable compound fitting where large-scale signal is ambiguous."""
    target_type = str((target_profile or {}).get("target_type") or "").lower()
    reasons: List[str] = []
    context = diffuse_context or {}
    policy = stage3_policy or {}
    if bool((noise_route or {}).get("requires_review", False)):
        reasons.append("directional_pattern_noise_requires_review")
    if bool(
        context.get("diffuse")
        or context.get("emission_diffuse")
        or context.get("large_nebulosity_feature")
    ):
        reasons.append("large_or_diffuse_nebula_signal")
    if bool(context.get("faint_nebula_protection")):
        reasons.append("low_contrast_faint_structure")
    if "dark_nebula" in target_type or bool(policy.get("protect_dark_structure")):
        reasons.append("dark_nebula_structure")
    if "low_contrast" in target_type:
        reasons.append("low_contrast_target_profile")
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "status": "eligible" if not unique_reasons else "excluded",
        "eligible": not unique_reasons,
        "reasons": unique_reasons,
        "target_type": target_type or None,
    }


def _stage3_compound_residual_gate(
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Require residual sky variation above held-out sampling uncertainty."""
    if (validation or {}).get("status") != "ready":
        return {
            "status": "not_supported",
            "supported": False,
            "reason": "held-out validation is unavailable",
        }
    try:
        robust_span = float(validation["robust_span"])
        patch_rms = float(validation["patch_mad_median"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "not_supported",
            "supported": False,
            "reason": "held-out validation is incomplete",
        }
    span_standard_error = background_span_standard_error(validation)
    significance_limit = max(
        STAGE3_SIGNIFICANCE_SIGMA * span_standard_error,
        1e-12,
    )
    supported = bool(
        math.isfinite(robust_span)
        and math.isfinite(significance_limit)
        and robust_span > significance_limit
    )
    return {
        "status": "supported" if supported else "not_supported",
        "supported": supported,
        "robust_span": robust_span,
        "patch_rms": patch_rms,
        "patch_median_standard_error": validation.get(
            "patch_median_uncertainty"
        ),
        "span_standard_error": span_standard_error,
        "spatial_significance_limit_3sigma": significance_limit,
        "evidence_basis": "correlation_aware_heldout_sky_sampling_uncertainty",
    }


def _stage3_compound_score_gate(
    best_single_score: float,
    compound_score: float,
    *,
    absolute_improvement_min: float = 0.03,
    relative_improvement_min: float = 0.10,
) -> Tuple[bool, Dict[str, Any]]:
    try:
        best_score = float(best_single_score)
        candidate_score = float(compound_score)
    except (TypeError, ValueError):
        return False, {
            "status": "rejected",
            "accepted": False,
            "issues": ["background scores are unavailable"],
        }
    if not math.isfinite(best_score) or not math.isfinite(candidate_score):
        return False, {
            "status": "rejected",
            "accepted": False,
            "issues": ["background scores are non-finite"],
        }
    absolute_min = max(0.0, float(absolute_improvement_min))
    relative_min = max(0.0, float(relative_improvement_min))
    absolute_improvement = best_score - candidate_score
    relative_improvement = absolute_improvement / max(abs(best_score), 1e-7)
    issues: List[str] = []
    if absolute_improvement + 1e-12 < absolute_min:
        issues.append(
            "absolute score improvement "
            f"{absolute_improvement:.3f}<{absolute_min:.3f}"
        )
    if relative_improvement + 1e-12 < relative_min:
        issues.append(
            "relative score improvement "
            f"{relative_improvement:.3f}<{relative_min:.3f}"
        )
    accepted = not issues
    return accepted, {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "issues": issues,
        "best_single_score": best_score,
        "compound_score": candidate_score,
        "absolute_improvement": absolute_improvement,
        "absolute_improvement_min": absolute_min,
        "relative_improvement": relative_improvement,
        "relative_improvement_min": relative_min,
    }


def _stage3_clear_background_samples(pipeline) -> None:
    clear_samples = getattr(pipeline.siril, "clear_image_bgsamples", None)
    if not callable(clear_samples):
        return
    try:
        clear_samples()
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        if hasattr(pipeline, "log"):
            pipeline.log.debug(f"Stage3 background sample cleanup skipped: {error}")


def _stage3_install_safe_background_samples(
    pipeline,
    points: List[Tuple[float, float]],
) -> Tuple[bool, Dict[str, Any]]:
    """Install and audit Siril's recalculated sample set before ``-existing``.

    Siril may discard a sample whose 25-pixel statistics cannot be
    recalculated.  That is not itself a failure: the returned set remains
    usable only when every surviving coordinate came from the audited request,
    the configured minimum is retained, and its spatial coverage is still
    sufficient.  Unknown samples or collapsed coverage fail closed.
    """
    setter = getattr(pipeline.siril, "set_image_bgsamples", None)
    if not points:
        return False, {
            "status": "unavailable",
            "installed": False,
            "reason": "safe background sample set is empty",
        }
    if not callable(setter):
        return False, {
            "status": "unsupported",
            "installed": False,
            "reason": "Siril Python API does not expose set_image_bgsamples",
        }
    try:
        _stage3_clear_background_samples(pipeline)
        result = setter(
            points,
            show_samples=False,
            recalculate=True,
        )
        if result is False:
            raise RuntimeError("set_image_bgsamples returned false")
        observed_count = None
        observed_positions: List[Tuple[float, float]] = []
        rejected_positions: List[List[float]] = []
        observed_coverage: Dict[str, Any] = {}
        getter = getattr(pipeline.siril, "get_image_bgsamples", None)
        if callable(getter):
            observed = getter()
            if observed is None:
                raise RuntimeError(
                    "Siril did not return the installed background samples"
                )
            observed_count = len(observed)
            for sample in observed:
                position = getattr(sample, "position", sample)
                if not isinstance(position, (tuple, list)) or len(position) < 2:
                    raise RuntimeError(
                        "Siril returned a background sample without coordinates"
                    )
                observed_positions.append(
                    (float(position[0]), float(position[1]))
                )

            remaining = list(observed_positions)
            for requested_x, requested_y in points:
                match_index = next(
                    (
                        index
                        for index, (observed_x, observed_y) in enumerate(remaining)
                        if math.hypot(
                            observed_x - float(requested_x),
                            observed_y - float(requested_y),
                        ) <= 0.75
                    ),
                    None,
                )
                if match_index is None:
                    rejected_positions.append(
                        [float(requested_x), float(requested_y)]
                    )
                else:
                    remaining.pop(match_index)
            if remaining:
                raise RuntimeError(
                    "Siril returned background samples outside the audited set"
                )

            try:
                minimum_count = int(
                    getattr(
                        getattr(pipeline, "cfg", None),
                        "stage3_safe_sample_min_count",
                        16,
                    )
                )
            except (TypeError, ValueError):
                minimum_count = 16
            minimum_count = max(
                STAGE3_MIN_VALIDATION_PATCHES,
                min(64, minimum_count),
            )
            if observed_count < minimum_count:
                raise RuntimeError(
                    "Siril retained too few audited background samples: "
                    f"minimum={minimum_count} observed={observed_count}"
                )

            requested_x = [float(point[0]) for point in points]
            requested_y = [float(point[1]) for point in points]
            x_low, x_high = min(requested_x), max(requested_x)
            y_low, y_high = min(requested_y), max(requested_y)
            x_span = max(x_high - x_low, 1.0)
            y_span = max(y_high - y_low, 1.0)
            cells = {
                (
                    min(3, max(0, int((x - x_low) / x_span * 4.0))),
                    min(3, max(0, int((y - y_low) / y_span * 4.0))),
                )
                for x, y in observed_positions
            }
            quadrants = {
                (int(x >= (x_low + x_high) / 2.0), int(y >= (y_low + y_high) / 2.0))
                for x, y in observed_positions
            }
            observed_coverage = {
                "quadrants": len(quadrants),
                "grid_cells": len(cells),
                "x_span_ratio_of_requested_envelope": (
                    max(x for x, _y in observed_positions)
                    - min(x for x, _y in observed_positions)
                ) / x_span,
                "y_span_ratio_of_requested_envelope": (
                    max(y for _x, y in observed_positions)
                    - min(y for _x, y in observed_positions)
                ) / y_span,
            }
            if (
                len(quadrants) < STAGE3_MIN_SPATIAL_QUADRANTS
                or len(cells) < STAGE3_MIN_SPATIAL_GRID_CELLS
                or observed_coverage["x_span_ratio_of_requested_envelope"]
                < STAGE3_MIN_AXIS_SPAN_RATIO
                or observed_coverage["y_span_ratio_of_requested_envelope"]
                < STAGE3_MIN_AXIS_SPAN_RATIO
            ):
                raise RuntimeError(
                    "Siril recalculation collapsed audited sample coverage: "
                    f"quadrants={len(quadrants)} grid_cells={len(cells)} "
                    "x_span_ratio="
                    f"{observed_coverage['x_span_ratio_of_requested_envelope']:.3f} "
                    "y_span_ratio="
                    f"{observed_coverage['y_span_ratio_of_requested_envelope']:.3f}"
                )
        return True, {
            "status": "installed",
            "installed": True,
            "sample_count": observed_count if observed_count is not None else len(points),
            "requested_count": len(points),
            "observed_count": observed_count,
            "siril_rejected_count": len(rejected_positions),
            "siril_rejected_positions": rejected_positions,
            "observed_coverage": observed_coverage,
            "command_contract": "subsky -existing",
        }
    except (
        AttributeError,
        CommandError,
        OSError,
        RuntimeError,
        SirilError,
        TypeError,
        ValueError,
    ) as error:
        _stage3_clear_background_samples(pipeline)
        return False, {
            "status": "failed",
            "installed": False,
            "sample_count": len(points),
            "reason": str(error),
        }


def _stage3_subsky_uses_existing(command: Tuple[str, ...]) -> bool:
    return bool(
        command
        and str(command[0]).lower() == "subsky"
        and "-existing" in command
    )


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
    secondary_labels = {
        str(label).strip()
        for label in (profile.get("secondary_labels") or [])
    }
    features = profile.get("features") if isinstance(profile, dict) else {}
    feature_large = bool(
        isinstance(features, dict)
        and features.get("large_nebulosity")
    )
    feature_large = bool(
        feature_large or "large_nebulosity" in secondary_labels
    )
    object_area_ratio = _stage3_metric(profile, adaptive or {}, "object_area_ratio")
    nebulosity_area_ratio = _stage3_metric(profile, adaptive or {}, "nebulosity_area_ratio")
    faint_structure_score = _stage3_metric(profile, adaptive or {}, "faint_structure_score")
    is_emission = target_type in EMISSION_NEBULA_TARGET_TYPES
    emission_context = bool(is_emission or "emission_red" in secondary_labels)
    emission_diffuse = bool(
        emission_context
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
        "secondary_labels": sorted(secondary_labels),
        "secondary_emission_context": bool(
            "emission_red" in secondary_labels
        ),
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
    target_type = str((target_profile or {}).get("target_type") or "").lower()
    if target_type in {"large_galaxy", "galaxy", "dark_nebula"}:
        return True
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


def _stage3_background_decision(
    pipeline,
    adaptive: Dict[str, Any],
    *,
    diffuse_context: Optional[Dict[str, Any]] = None,
    process_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve Stage 3 from process evidence before destructive commands."""
    cfg = pipeline.cfg
    if not bool(getattr(cfg, "stage3_conditional_decision_enabled", True)):
        return {
            "decision": "apply",
            "source": "compatibility_override",
            "confidence": 1.0,
            "reason": "conditional background decision disabled by configuration",
            "threshold_basis": "explicit configuration override",
        }

    if isinstance(process_report, dict) and process_report:
        mechanism = str(process_report.get("mechanism") or "unknown")
        common_process = {
            "metrics": dict(adaptive or {}),
            "diffuse_context": dict(diffuse_context or {}),
            "process_evidence": process_report,
            "process_evidence_schema": process_report.get(
                "schema_version",
                STAGE3_PROCESS_EVIDENCE_SCHEMA,
            ),
            "threshold_basis": "source-masked held-out process evidence",
        }
        hard_blocks = list(process_report.get("hard_block_reasons") or [])
        if hard_blocks:
            return {
                **common_process,
                "decision": "review_required",
                "source": "process_evidence",
                "confidence": 1.0,
                "reason": "; ".join(str(reason) for reason in hard_blocks),
            }
        if bool(process_report.get("should_evaluate", False)):
            return {
                **common_process,
                "decision": "apply",
                "source": "process_evidence",
                "confidence": 1.0,
                "reason": (
                    "linear input, source-masked true sky and held-out spatial "
                    "variation authorize a bounded low-complexity model"
                ),
            }
        if mechanism == "no_measurable_low_frequency_gradient":
            return {
                **common_process,
                "decision": "preserve",
                "source": "process_evidence",
                "confidence": 1.0,
                "reason": (
                    "source-masked held-out sky shows no spatial variation above "
                    "patch-median uncertainty"
                ),
            }
        return {
            **common_process,
            "decision": "review_required",
            "source": "process_evidence",
            "confidence": 1.0,
            "reason": f"background mechanism requires review: {mechanism}",
        }

    if hasattr(pipeline, "input_profile"):
        return {
            "decision": "review_required",
            "source": "process_evidence",
            "confidence": 0.0,
            "reason": "source-masked Stage 3 process evidence is unavailable",
            "metrics": dict(adaptive or {}),
            "diffuse_context": dict(diffuse_context or {}),
            "threshold_basis": "process evidence required",
        }

    def bounded(name: str, default: float, lower: float, upper: float) -> float:
        try:
            value = float(getattr(cfg, name, default))
        except (TypeError, ValueError):
            value = default
        return max(lower, min(upper, value))

    gradient_skip_max = bounded("stage3_gradient_skip_max", 0.045, 0.0, 0.30)
    dirty_skip_max = bounded("stage3_dirty_skip_max", 0.16, 0.0, 0.60)
    gradient_apply_min = bounded("stage3_gradient_apply_min", 0.08, 0.01, 0.80)
    dirty_apply_min = bounded("stage3_dirty_apply_min", 0.18, 0.01, 0.80)
    confidence_min = bounded("stage3_apply_confidence_min", 0.75, 0.50, 0.99)
    thresholds = {
        "gradient_skip_max": gradient_skip_max,
        "dirty_skip_max": dirty_skip_max,
        "gradient_apply_min": gradient_apply_min,
        "dirty_apply_min": dirty_apply_min,
        "apply_confidence_min": confidence_min,
    }
    required_keys = {"gradient_score", "dirty_background_score"}
    if not isinstance(adaptive, dict) or not required_keys.issubset(adaptive):
        return {
            "decision": "review_required",
            "source": "diagnostics",
            "confidence": 0.0,
            "reason": "gradient diagnostics unavailable or incomplete",
            "metrics": dict(adaptive or {}),
            "thresholds": thresholds,
            "threshold_basis": "project internal engineering gate",
        }

    try:
        gradient = float(adaptive.get("gradient_score") or 0.0)
        dirty = float(adaptive.get("dirty_background_score") or 0.0)
    except (TypeError, ValueError):
        gradient = 0.0
        dirty = 0.0
    metrics = {
        "gradient_score": gradient,
        "dirty_background_score": dirty,
        "chroma_noise_score": float(adaptive.get("chroma_noise_score") or 0.0),
    }
    target_profile = getattr(pipeline, "target_profile", {}) or {}
    advisory: Any = getattr(pipeline, "_stage3_background_advisory", None)
    if not isinstance(advisory, dict) and isinstance(target_profile, dict):
        raw_decision = target_profile.get("dbe_decision")
        if raw_decision:
            advisory = {
                "decision": raw_decision,
                "confidence": target_profile.get("dbe_confidence", 0.0),
                "reason": target_profile.get("dbe_reason", ""),
            }
    if not isinstance(advisory, dict):
        advisory = {}

    raw_advisory_decision = str(
        advisory.get("decision") or advisory.get("dbe_decision") or ""
    ).strip().lower()
    normalized_advisory = {
        "review_chromatic": "review_required",
        "review": "review_required",
    }.get(raw_advisory_decision, raw_advisory_decision)
    try:
        advisory_confidence = max(
            0.0,
            min(1.0, float(advisory.get("confidence") or 0.0)),
        )
    except (TypeError, ValueError):
        advisory_confidence = 0.0

    common = {
        "metrics": metrics,
        "thresholds": thresholds,
        "threshold_basis": "project internal engineering gate",
        "diffuse_context": dict(diffuse_context or {}),
    }
    if normalized_advisory == "skip":
        return {
            **common,
            "decision": "skip",
            "source": "validated_advisory",
            "confidence": advisory_confidence,
            "reason": str(advisory.get("reason") or "advisor selected skip"),
        }
    if normalized_advisory == "review_required":
        return {
            **common,
            "decision": "review_required",
            "source": "validated_advisory",
            "confidence": advisory_confidence,
            "reason": str(
                advisory.get("reason")
                or "advisor requires chromatic/background review"
            ),
        }

    diffuse_risk = bool(
        (diffuse_context or {}).get("diffuse")
        or (diffuse_context or {}).get("emission_diffuse")
        or (diffuse_context or {}).get("pixel_signal_protection")
    )
    if diffuse_risk and not bool(
        getattr(cfg, "stage3_diffuse_auto_apply_enabled", False)
    ):
        if (
            dirty <= dirty_skip_max
            and normalized_advisory != "apply"
            and bool(
                (diffuse_context or {}).get("pixel_signal_protection")
                or (diffuse_context or {}).get("emission_diffuse")
            )
        ):
            return {
                **common,
                "decision": "skip",
                "source": "target_protection_policy",
                "confidence": 0.85,
                "reason": (
                    "dirty-background evidence is low and the measured "
                    "gradient overlaps protected diffuse target signal"
                ),
            }
        return {
            **common,
            "decision": "review_required",
            "source": (
                "validated_advisory"
                if normalized_advisory == "apply"
                else "diagnostics"
            ),
            "confidence": advisory_confidence,
            "reason": "diffuse or large-scale target signal may contaminate DBE samples",
        }

    if gradient <= gradient_skip_max and dirty <= dirty_skip_max:
        return {
            **common,
            "decision": "skip",
            "source": "diagnostics",
            "confidence": 0.90,
            "reason": "no material low-frequency gradient detected",
        }

    eligible = gradient >= gradient_apply_min and dirty >= dirty_apply_min
    if normalized_advisory == "apply":
        if advisory_confidence < confidence_min:
            return {
                **common,
                "decision": "review_required",
                "source": "validated_advisory",
                "confidence": advisory_confidence,
                "reason": "apply advisory confidence is below the execution gate",
            }
        if not eligible:
            return {
                **common,
                "decision": "review_required",
                "source": "validated_advisory",
                "confidence": advisory_confidence,
                "reason": "apply advisory is not supported by deterministic gradient evidence",
            }
        return {
            **common,
            "decision": "apply",
            "source": "validated_advisory",
            "confidence": advisory_confidence,
            "reason": str(advisory.get("reason") or "high-confidence apply advisory"),
        }

    if eligible and bool(
        getattr(cfg, "stage3_deterministic_auto_apply_enabled", True)
    ):
        confidence = min(
            0.95,
            0.75
            + 0.20
            * min(
                gradient / max(gradient_apply_min, 1e-6) - 1.0,
                dirty / max(dirty_apply_min, 1e-6) - 1.0,
                1.0,
            ),
        )
        return {
            **common,
            "decision": "apply",
            "source": "deterministic_offline_policy",
            "confidence": max(0.75, confidence),
            "reason": "directional gradient and dirty-background evidence exceed apply gates",
        }

    return {
        **common,
        "decision": "review_required",
        "source": "diagnostics",
        "confidence": 0.50,
        "reason": "background evidence is ambiguous; preserve the baseline",
    }


def run_stage3_background_extraction(pipeline) -> None:
    """
    阶段 3: 背景提取
    - 先生成空间分布安全样点，并以 subsky -existing 建模低频天空梯度
    - 单阶段内置候选均不充分时，可在冻结验证门下尝试 Polynomial→残差RBF
    - 复合候选不适用、失败或仍不充分时，再尝试 GraXpert / ADBE / DBE / NOX 回退
    - 条纹和 walking noise 与天空梯度分流，禁止用背景模型静默吞掉结构噪声
    - 每个候选成功后执行质量门控，避免过度扣背景
    - 候选命令失败或未达到充分质量时，继续 fallback 到下一个候选
    """
    stage_label = PipelineStage.BACKGROUND_EXTRACTION.label
    pipeline.log.stage_start(stage_label)
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
    decision_thresholds = _stage3_decision_thresholds(pipeline, stage3_policy)
    pipeline.log.info(
        "[Stage3] Background policy: "
        f"policy={policy_name} protect_nebulosity={bool(stage3_policy.get('protect_nebulosity', False))} "
        f"model={','.join(stage3_policy.get('model_priority', []) or [])}"
    )

    baseline_stem = "stage3_bg_input"
    baseline_saved = False
    rollback_events: List[Dict[str, Any]] = []
    try:
        pipeline.cmd_with_check("save", baseline_stem)
        baseline_saved = True
    except (CommandError, SirilError) as e:
        pipeline.log.warn(
            "stage3 baseline save failed; skip destructive background candidates: "
            f"{e}"
        )

    def restore_baseline(context: str) -> bool:
        nonlocal baseline_saved
        if not baseline_saved:
            rollback_events.append(
                {
                    "context": context,
                    "status": "unavailable",
                    "reason": "stage3 baseline checkpoint unavailable",
                }
            )
            return False
        try:
            pipeline.cmd_with_check("load", baseline_stem, quiet=True)
            rollback_events.append({"context": context, "status": "restored"})
            return True
        except (CommandError, SirilError) as e:
            baseline_saved = False
            rollback_events.append(
                {
                    "context": context,
                    "status": "failed",
                    "reason": str(e),
                }
            )
            pipeline.log.warn(f"failed to restore stage3 baseline ({context}): {e}")
            return False

    before_feat = pipeline._stage3_measure_features("before")
    before_image = None
    try:
        before_image = pipeline.siril.get_image_pixeldata(preview=False)
    except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.debug(f"stage3 baseline image sampling skipped: {e}")
    before_adaptive = (
        pipeline._adaptive_features_current()
        if hasattr(pipeline, "_adaptive_features_current")
        else {}
    )

    pattern_routing_enabled = bool(
        getattr(pipeline.cfg, "stage3_pattern_routing_enabled", True)
    )
    if pattern_routing_enabled and before_image is not None:
        pattern_report = analyze_directional_pattern_noise(
            before_image,
            detection_threshold=_stage3_cfg_float(
                pipeline,
                "stage3_pattern_score_min",
                0.55,
                0.25,
                0.90,
            ),
            walking_threshold=_stage3_cfg_float(
                pipeline,
                "stage3_walking_noise_score_min",
                0.50,
                0.25,
                0.90,
            ),
        )
    else:
        pattern_report = {
            "status": "disabled" if not pattern_routing_enabled else "unavailable",
            "detected": False,
            "reason": (
                "disabled by configuration"
                if not pattern_routing_enabled
                else "baseline pixels unavailable"
            ),
        }
    noise_route: Dict[str, Any] = {}

    attempt_records: List[Dict[str, Any]] = []
    selected_preservation: Dict[str, Any] = {}
    accepted_candidates: List[Dict[str, Any]] = []
    builtin_sufficient = False
    graxpert_attempted = False
    graxpert_runtime_error = False
    graxpert_error_reasons: List[str] = []
    selected_label = ""
    selected_pattern_report: Dict[str, Any] = {}
    builtin_order_reason = "safe_samples_rbf_before_poly"
    builtin_search_mode = "safe_samples_primary"
    diffuse_context: Dict[str, Any] = {}
    safe_sample_points: List[Tuple[float, float]] = []
    safe_sample_report: Dict[str, Any] = {
        "status": "not_run",
        "sample_count": 0,
    }
    compound_fit_points: List[Tuple[float, float]] = []
    compound_validation_points: List[Tuple[float, float]] = []
    compound_split_report: Dict[str, Any] = {
        "status": "not_run",
    }
    compound_target_guard: Dict[str, Any] = {
        "status": "not_evaluated",
        "eligible": False,
        "reasons": [],
    }
    baseline_validation: Dict[str, Any] = {
        "status": "not_run",
    }
    compound_report: Dict[str, Any] = {
        "status": "not_triggered",
        "triggered": False,
    }
    compound_selected = False
    compound_selected_degraded = False
    selection_shadow: Dict[str, Any] = {
        "status": "not_run",
        "method": "pareto_dense_rank_sum_v1",
        "runtime_selection_affected": False,
    }
    final_output_validation: Dict[str, Any] = {
        "status": "not_run",
        "enforced": False,
    }
    final_output_validation_rejected = False
    attempted_selected_label = ""

    target_profile = getattr(pipeline, "target_profile", {}) or {}
    profile_fallback_used = bool(
        isinstance(target_profile, dict)
        and str(
            target_profile.get("classification_method") or ""
        ).strip().lower()
        == "fallback"
    )
    _diffuse, diffuse_context = _stage3_diffuse_nebula_context(
        target_profile,
        before_adaptive,
        stage3_policy=stage3_policy,
    )
    diffuse_context["diffuse"] = bool(_diffuse)
    sample_refinement_blocked = bool(
        diffuse_context.get("diffuse")
        or diffuse_context.get("emission_diffuse")
        or diffuse_context.get("large_nebulosity_feature")
        or diffuse_context.get("faint_nebula_protection")
        or diffuse_context.get("pixel_signal_protection")
    )

    if before_image is not None:
        safe_sample_points, safe_sample_report = build_safe_background_samples(
            before_image,
            target_count=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_target_count",
                40,
                16,
                64,
            ),
            min_count=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_min_count",
                16,
                12,
                48,
            ),
            patch_radius=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_patch_radius",
                12,
                4,
                24,
            ),
            brightness_quantile_max=_stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_brightness_quantile_max",
                0.70,
                0.50,
                0.85,
            ),
            texture_quantile_max=_stage3_cfg_float(
                pipeline,
                "stage3_safe_sample_texture_quantile_max",
                0.55,
                0.25,
                0.75,
            ),
            candidate_refinement=not sample_refinement_blocked,
        )
    else:
        safe_sample_report = {
            "status": "unavailable",
            "sample_count": 0,
            "error": "baseline pixels unavailable",
        }

    if before_image is not None and safe_sample_points:
        (
            compound_fit_points,
            compound_validation_points,
            compound_split_report,
        ) = split_background_sample_points(
            safe_sample_points,
            before_image,
            validation_ratio=_stage3_cfg_float(
                pipeline,
                "stage3_compound_validation_ratio",
                0.25,
                0.15,
                0.35,
            ),
            minimum_total=_stage3_cfg_int(
                pipeline,
                "stage3_compound_min_sample_count",
                32,
                32,
                64,
            ),
            minimum_fit=_stage3_cfg_int(
                pipeline,
                "stage3_compound_fit_min_count",
                24,
                24,
                56,
            ),
            minimum_validation=_stage3_cfg_int(
                pipeline,
                "stage3_compound_validation_min_count",
                8,
                8,
                20,
            ),
        )
        if compound_split_report.get("status") == "ready":
            baseline_validation = measure_background_validation(
                before_image,
                compound_validation_points,
                patch_radius=_stage3_cfg_int(
                    pipeline,
                    "stage3_safe_sample_patch_radius",
                    12,
                    4,
                    24,
                ),
                minimum_count=_stage3_cfg_int(
                    pipeline,
                    "stage3_compound_validation_min_count",
                    8,
                    8,
                    20,
                ),
            )
    else:
        compound_split_report = {
            "status": "unavailable",
            "reason": "audited samples or baseline pixels are unavailable",
        }

    input_profile = getattr(pipeline, "input_profile", None)
    process_profile = (
        input_profile
        if isinstance(input_profile, dict)
        else {
            "state": "linear",
            "safe_for_linear_steps": True,
            "source": "legacy_stage3_contract",
        }
    )
    process_report = assess_background_process(
        before_image,
        safe_sample_points,
        safe_sample_report,
        baseline_validation,
        pattern_report,
        input_profile=process_profile,
        diffuse_context=diffuse_context,
        patch_radius=_stage3_cfg_int(
            pipeline,
            "stage3_safe_sample_patch_radius",
            12,
            4,
            24,
        ),
    ) if before_image is not None else {
        "status": "review_required",
        "should_evaluate": False,
        "mechanism": "unavailable",
        "hard_block_reasons": ["baseline_pixels_unavailable"],
    }
    noise_route = select_background_route(
        before_adaptive,
        pattern_report,
        process_report=process_report,
        gradient_apply_min=_stage3_cfg_float(
            pipeline,
            "stage3_gradient_apply_min",
            0.08,
            0.01,
            0.80,
        ),
        dirty_apply_min=_stage3_cfg_float(
            pipeline,
            "stage3_dirty_apply_min",
            0.18,
            0.01,
            0.80,
        ),
    )
    if not pattern_routing_enabled:
        noise_route.update(
            route="routing_disabled",
            pattern_detected=False,
            subsky_existing_allowed=True,
            requires_review=False,
            reason="directional-pattern routing disabled by configuration",
        )
    pipeline._stage3_pattern_noise_report = {
        "analysis": pattern_report,
        "route": noise_route,
    }
    compound_target_guard = _stage3_compound_target_guard(
        target_profile,
        diffuse_context,
        stage3_policy,
        noise_route,
    )
    safe_sample_report = {
        **safe_sample_report,
        "fit_validation_split": compound_split_report,
        # Compatibility name retained for existing diagnostics consumers.
        "compound_split": compound_split_report,
        "compound_target_guard": compound_target_guard,
        "baseline_validation": baseline_validation,
        "compound_baseline_validation": baseline_validation,
    }
    pipeline._stage3_safe_sample_report = safe_sample_report
    pipeline.log.info(
        "[Stage3] Process evidence: "
        f"linear={bool((process_report.get('linear_input') or {}).get('confirmed'))} "
        f"samples={safe_sample_report.get('status')}:"
        f"{int(safe_sample_report.get('sample_count') or 0)} "
        f"holdout={compound_split_report.get('status')} "
        f"mechanism={process_report.get('mechanism')}"
    )
    background_decision = _stage3_background_decision(
        pipeline,
        before_adaptive,
        diffuse_context=diffuse_context,
        process_report=(
            process_report
            if isinstance(input_profile, dict)
            else None
        ),
    )
    if noise_route.get("route") == "pattern_noise_deferred":
        background_decision = {
            **background_decision,
            "decision": "review_required",
            "source": "pattern_noise_router",
            "confidence": max(
                float(background_decision.get("confidence") or 0.0),
                float(pattern_report.get("pattern_score") or 0.0),
            ),
            "reason": str(noise_route.get("reason") or "pattern noise deferred"),
            "pre_route_decision": dict(background_decision),
            "noise_route": noise_route,
        }
    else:
        background_decision["noise_route"] = noise_route
    pipeline._stage3_background_decision = background_decision
    decision = str(background_decision.get("decision") or "review_required")
    pipeline.log.info(
        "[Stage3] Background decision: "
        f"decision={decision} source={background_decision.get('source')} "
        f"confidence={float(background_decision.get('confidence') or 0.0):.2f}"
    )
    if decision != "apply":
        restore_baseline(f"decision:{decision}")
        stage_saved = pipeline._save_stage_output("stage3_bgremoved")
        after_adaptive = dict(before_adaptive or {})
        if decision == "review_required":
            pipeline._background_review_required = True
        reason = str(
            background_decision.get("reason")
            or "background extraction was not authorized"
        )
        if hasattr(pipeline, "_write_stage_json"):
            pipeline._write_stage_json(
                "background_quality_report.json",
                {
                    "schema_version": STAGE3_BACKGROUND_QUALITY_SCHEMA,
                    "algorithm_contract_version": (
                        STAGE3_ALGORITHM_CONTRACT_VERSION
                    ),
                    "stage": "stage3_background",
                    "policy": policy_name,
                    "decision_thresholds": decision_thresholds,
                    "decision": background_decision,
                    "process_evidence": process_report,
                    "model_used": None,
                    "candidate_order": [],
                    "attempts": [],
                    "selection_shadow": {
                        **selection_shadow,
                        "status": "not_applicable",
                        "reason": f"decision={decision}",
                    },
                    "final_output_validation": {
                        **final_output_validation,
                        "status": "not_applicable",
                        "reason": f"decision={decision}",
                    },
                    "rollback_events": rollback_events,
                    "diffuse_nebula_context": diffuse_context,
                    "directional_pattern_noise": pattern_report,
                    "noise_route": noise_route,
                    "safe_samples": safe_sample_report,
                    "before": before_adaptive,
                    "after": after_adaptive,
                    "quality": (
                        "unchanged"
                        if decision in {"skip", "preserve"}
                        else "review_required"
                    ),
                    "fallback_used": False,
                },
            )
        message = f"decision={decision}; {reason}"
        if preflight_message:
            message = f"{preflight_message}; {message}"
        if not stage_saved:
            message += "; stage3 输出保存失败"
        elapsed = pipeline.log.stage_end(stage_label)
        pipeline._record_stage(
            stage_label,
            (
                "skipped"
                if decision == "skip" and stage_saved
                else "ok"
                if decision == "preserve" and stage_saved
                else "degraded"
            ),
            elapsed,
            message,
            execution=(
                "skipped"
                if decision == "skip" and stage_saved
                else "safe_passthrough"
                if decision == "preserve" and stage_saved
                else "safe_passthrough"
            ),
            fallback_used=profile_fallback_used,
            reason_code=(
                "stage3_output_save_failed"
                if not stage_saved
                else "background_not_required"
                if decision == "skip"
                else "background_not_required_after_process_validation"
                if decision == "preserve"
                else "pattern_noise_deferred"
                if noise_route.get("route") == "pattern_noise_deferred"
                else "background_review_required"
            ),
            components={
                "target_profile": {
                    "status": "applied",
                    "method": target_profile.get("classification_method"),
                    "reason_code": (
                        "target_profiler_fallback"
                        if profile_fallback_used
                        else "accepted"
                    ),
                    "fallback_used": profile_fallback_used,
                },
                "background_extraction": {
                    "status": (
                        "skipped"
                        if decision == "skip"
                        else "preserved"
                        if decision == "preserve"
                        else "rolled_back"
                    ),
                    "method": None,
                    "reason_code": (
                        "background_not_required"
                        if decision == "skip"
                        else "background_not_required_after_process_validation"
                        if decision == "preserve"
                        else "background_review_required"
                    ),
                    "input": baseline_stem,
                    "output": "stage3_bgremoved" if stage_saved else None,
                    "fallback_used": False,
                },
                "directional_pattern_router": {
                    "status": (
                        "review_required"
                        if noise_route.get("requires_review")
                        else "accepted"
                    ),
                    "method": noise_route.get("route"),
                    "reason_code": (
                        "pattern_noise_deferred"
                        if noise_route.get("route") == "pattern_noise_deferred"
                        else "mixed_gradient_pattern_noise_review"
                        if noise_route.get("route") == "mixed_gradient_and_pattern_noise"
                        else "no_directional_pattern_detected"
                    ),
                    "fallback_used": False,
                },
            },
        )
        return

    def evaluate_attempts(
        attempts: List[Tuple[str, Tuple[str, ...], str]],
        *,
        phase: str,
        stop_on_sufficient: bool = True,
    ) -> bool:
        nonlocal baseline_saved, graxpert_runtime_error
        phase_sufficient = False
        if not baseline_saved:
            pipeline.log.warn(
                f"[Stage3] Skip {phase}: rollback checkpoint is unavailable"
            )
            return False
        for label, command, source in attempts:
            if not restore_baseline(f"before:{label}"):
                pipeline.log.warn(
                    f"[Stage3] Stop {phase}: unable to establish clean baseline "
                    f"before {label}"
                )
                break

            sample_install_report: Dict[str, Any] = {
                "status": "not_required",
                "installed": False,
            }
            if _stage3_subsky_uses_existing(command):
                installed_points = (
                    compound_fit_points
                    if source == "builtin"
                    and compound_split_report.get("status") == "ready"
                    else safe_sample_points
                )
                sample_ok, sample_install_report = (
                    _stage3_install_safe_background_samples(
                        pipeline,
                        installed_points,
                    )
                )
                if not sample_ok:
                    attempt_records.append(
                        {
                            "label": label,
                            "source": source,
                            "phase": phase,
                            "command": list(command),
                            "status": "safe_sample_install_failed",
                            "failure_reason": sample_install_report.get("reason"),
                            "safe_samples": sample_install_report,
                            "fallback_triggered": True,
                        }
                    )
                    pipeline.log.warn(
                        f"{label} 禁止执行：自定义安全样点未完整安装；"
                        "不会让 subsky 自动重采样"
                    )
                    if not restore_baseline(f"sample_install_failed:{label}"):
                        break
                    continue
            else:
                _stage3_clear_background_samples(pipeline)

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
                is_plugin_runtime = bool(
                    failure_reason
                    and failure_reason.startswith("plugin_runtime_error:")
                )
                if is_graxpert_runtime:
                    graxpert_runtime_error = True
                    graxpert_error_reasons.append(failure_reason)
                    pipeline.log.warn(
                        f"{label} 运行失败，自动切换到下一个背景提取候选: {failure_reason}"
                    )
                elif is_plugin_runtime:
                    pipeline.log.warn(
                        f"{label} 未产生有效图像变更，自动切换到下一个背景提取候选: "
                        f"{failure_reason}"
                    )
                attempt_records.append(
                    {
                        "label": label,
                        "source": source,
                        "phase": phase,
                        "command": list(command),
                        "status": (
                            "graxpert_runtime_error"
                            if is_graxpert_runtime
                            else (
                                "plugin_runtime_error"
                                if is_plugin_runtime
                                else "command_failed"
                            )
                        ),
                        "failure_reason": failure_reason,
                        "safe_samples": sample_install_report,
                        "fallback_triggered": bool(
                            is_graxpert or is_plugin_runtime
                        ),
                    }
                )
                if not restore_baseline(f"failed:{label}"):
                    break
                continue

            after_feat = pipeline._stage3_measure_features(label)
            after_image = None
            try:
                after_image = pipeline.siril.get_image_pixeldata(preview=False)
            except (CommandError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
                pipeline.log.debug(f"stage3 candidate image sampling skipped ({label}): {e}")
            preservation = pipeline._stage3_signal_preservation_metrics(
                before_image,
                after_image,
            )
            fidelity_ok, fidelity_gate = assess_target_fidelity(
                preservation,
                low_complexity_required=bool(
                    process_report.get("low_complexity_required", False)
                ),
            )
            fidelity_enforced = isinstance(input_profile, dict)
            gate_ok, gate_msg = pipeline._stage3_quality_gate(
                before_feat,
                after_feat,
                preservation,
            )
            if fidelity_enforced and not fidelity_ok:
                gate_ok = False
                issues = ", ".join(fidelity_gate.get("issues") or [])
                gate_msg = (
                    f"{gate_msg}; target fidelity gate rejected"
                    + (f": {issues}" if issues else "")
                )
            if pattern_routing_enabled and after_image is not None:
                after_pattern_report = analyze_directional_pattern_noise(
                    after_image,
                    detection_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_pattern_score_min",
                        0.55,
                        0.25,
                        0.90,
                    ),
                    walking_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_walking_noise_score_min",
                        0.50,
                        0.25,
                        0.90,
                    ),
                )
            else:
                after_pattern_report = {
                    "status": "unavailable",
                    "detected": False,
                }
            pattern_ok, pattern_gate_report = pattern_candidate_gate(
                pattern_report,
                after_pattern_report,
                growth_max=_stage3_cfg_float(
                    pipeline,
                    "stage3_pattern_score_growth_max",
                    0.12,
                    0.02,
                    0.40,
                ),
            )
            if not pattern_ok:
                gate_ok = False
                gate_msg = (
                    f"{gate_msg}; directional pattern gate rejected "
                    f"growth={float(pattern_gate_report.get('pattern_score_growth') or 0.0):.3f}"
                )
            after_adaptive_candidate = (
                pipeline._adaptive_features_current()
                if hasattr(pipeline, "_adaptive_features_current")
                else {}
            )
            if compound_validation_points and after_image is not None:
                candidate_validation = measure_background_validation(
                    after_image,
                    compound_validation_points,
                    patch_radius=_stage3_cfg_int(
                        pipeline,
                        "stage3_safe_sample_patch_radius",
                        12,
                        4,
                        24,
                    ),
                    minimum_count=_stage3_cfg_int(
                        pipeline,
                        "stage3_compound_validation_min_count",
                        8,
                        8,
                        20,
                    ),
                    value_scale=baseline_validation.get("value_scale"),
                )
            else:
                candidate_validation = {
                    "status": "not_run",
                    "reason": "held-out validation pool is unavailable",
                }
            validation_ok, validation_gate = assess_single_background_validation(
                baseline_validation,
                candidate_validation,
            )
            validation_enforced = isinstance(input_profile, dict)
            if validation_enforced and not validation_ok:
                gate_ok = False
                issues = ", ".join(validation_gate.get("issues") or [])
                gate_msg = (
                    f"{gate_msg}; held-out background/RMS gate rejected"
                    + (f": {issues}" if issues else "")
                )
            color_shift = _stage3_color_shift(before_adaptive, after_adaptive_candidate)
            required_adaptive_metrics = {
                "bg_std",
                "gradient_score",
                "dirty_background_score",
                "chroma_noise_score",
            }
            adaptive_metrics_available = bool(
                required_adaptive_metrics.issubset(before_adaptive)
                and required_adaptive_metrics.issubset(after_adaptive_candidate)
            )
            color_shift_limit = _stage3_policy_float(
                stage3_policy,
                "sufficient_color_shift_max",
                0.18,
                minimum=0.0,
                maximum=2.0,
            )
            hard_gate_metrics_available = bool(
                before_feat is not None
                and after_feat is not None
                and after_image is not None
                and preservation.get("available")
                and adaptive_metrics_available
                and color_shift <= color_shift_limit
                and (not validation_enforced or validation_ok)
                and (
                    not pattern_routing_enabled
                    or (
                        pattern_report.get("status") == "ok"
                        and after_pattern_report.get("status") == "ok"
                    )
                )
            )
            record = {
                "label": label,
                "source": source,
                "phase": phase,
                "command": list(command),
                "status": "accepted" if gate_ok else "rejected",
                "quality_message": gate_msg,
                "preservation": preservation,
                "target_fidelity_gate": fidelity_gate,
                "target_fidelity_enforced": fidelity_enforced,
                "safe_samples": sample_install_report,
                "directional_pattern_noise": after_pattern_report,
                "pattern_quality_gate": pattern_gate_report,
                "after_adaptive": after_adaptive_candidate,
                "color_shift": color_shift,
                "validation": candidate_validation,
                "validation_gate": validation_gate,
                "validation_enforced": validation_enforced,
                "hard_gate_metrics_available": hard_gate_metrics_available,
            }
            if not gate_ok:
                attempt_records.append(record)
                pipeline.log.warn(
                    f"{label} rejected by quality gate, try next candidate: {gate_msg}"
                )
                if not restore_baseline(f"rejected:{label}"):
                    break
                continue

            candidate_stem = _stage3_candidate_stem(label)
            candidate_saved = pipeline._save_stage_output(candidate_stem)
            background_score_components = _stage3_background_score_components(
                before_adaptive,
                after_adaptive_candidate,
            )
            base_candidate_score = float(background_score_components["total"])
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
            pattern_penalty = max(
                0.0,
                float(pattern_gate_report.get("pattern_score_growth", 0.0) or 0.0),
            ) * STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT
            candidate_score = (
                base_candidate_score
                + preservation_penalty
                + pattern_penalty
            )
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
                    "background_score_components": background_score_components,
                    "preservation_penalty": preservation_penalty,
                    "directional_pattern_penalty": pattern_penalty,
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
                        "command": list(command),
                        "stem": candidate_stem,
                        "base_score": base_candidate_score,
                        "background_score_components": (
                            background_score_components
                        ),
                        "preservation_penalty": preservation_penalty,
                        "directional_pattern_penalty": pattern_penalty,
                        "nebula_preservation_penalty_weight": nebula_preservation_weight,
                        "score": candidate_score,
                        "quality_message": gate_msg,
                        "preservation": preservation,
                        "target_fidelity_gate": fidelity_gate,
                        "target_fidelity_enforced": fidelity_enforced,
                        "directional_pattern_noise": after_pattern_report,
                        "pattern_quality_gate": pattern_gate_report,
                        "after_adaptive": after_adaptive_candidate,
                        "color_shift": color_shift,
                        "validation": candidate_validation,
                        "validation_gate": validation_gate,
                        "validation_enforced": validation_enforced,
                        "hard_gate_metrics_available": hard_gate_metrics_available,
                        "sufficient": sufficient,
                    }
                )
            if not restore_baseline(f"evaluated:{label}"):
                break
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

    def evaluate_compound_candidate() -> bool:
        nonlocal compound_report
        label = "subsky-poly-residual-rbf"
        phase = "compound_fallback"
        accepted_builtin_candidates = [
            candidate
            for candidate in accepted_candidates
            if candidate.get("source") == "builtin"
        ]
        unverified_builtin_candidates = [
            candidate.get("label")
            for candidate in accepted_builtin_candidates
            if not bool(candidate.get("hard_gate_metrics_available", False))
        ]
        builtin_candidates = [
            candidate
            for candidate in accepted_builtin_candidates
            if bool(candidate.get("hard_gate_metrics_available", False))
        ]
        rbf_candidates = [
            candidate
            for candidate in builtin_candidates
            if "-rbf" in tuple(candidate.get("command") or ())
        ]
        hard_rejections = [
            record.get("label")
            for record in attempt_records
            if record.get("source") == "builtin"
            and record.get("status") == "rejected"
        ]
        eligibility_issues: List[str] = []
        if not compound_target_guard.get("eligible"):
            eligibility_issues.extend(
                str(reason)
                for reason in compound_target_guard.get("reasons", [])
            )
        if compound_split_report.get("status") != "ready":
            eligibility_issues.append("deterministic_fit_validation_split_unavailable")
        if baseline_validation.get("status") != "ready":
            eligibility_issues.append("baseline_validation_unavailable")
        if not bool(getattr(pipeline.cfg, "bg_quality_gate_enabled", True)):
            eligibility_issues.append("stage3_quality_gate_disabled")
        if hard_rejections:
            eligibility_issues.append("single_stage_hard_gate_rejection_present")
        if unverified_builtin_candidates:
            eligibility_issues.append("single_stage_hard_gate_metrics_unavailable")
        if not builtin_candidates:
            eligibility_issues.append("no_hard_gate_accepted_builtin_candidate")
        if not rbf_candidates:
            eligibility_issues.append("no_hard_gate_accepted_rbf_candidate")

        best_single = (
            min(
                builtin_candidates,
                key=lambda item: float(item.get("score", 999.0)),
            )
            if builtin_candidates
            else None
        )
        best_rbf = (
            min(
                rbf_candidates,
                key=lambda item: float(item.get("score", 999.0)),
            )
            if rbf_candidates
            else None
        )
        if best_single is not None:
            residual_gate = _stage3_compound_residual_gate(
                best_single.get("validation") or {},
            )
            if not residual_gate.get("supported"):
                eligibility_issues.append("low_frequency_residual_not_supported")
            if (best_single.get("validation") or {}).get("status") != "ready":
                eligibility_issues.append("best_single_validation_unavailable")
        else:
            residual_gate = {"status": "not_run", "supported": False}
        if best_rbf is not None and not bool(
            best_rbf.get("hard_gate_metrics_available", False)
        ):
            eligibility_issues.append("best_rbf_hard_gate_metrics_unavailable")

        eligibility_issues = list(dict.fromkeys(eligibility_issues))
        if eligibility_issues:
            compound_report = {
                "status": "not_triggered",
                "triggered": False,
                "eligibility_issues": eligibility_issues,
                "target_guard": compound_target_guard,
                "sample_split": compound_split_report,
                "residual_gate": residual_gate,
                "hard_rejected_single_candidates": hard_rejections,
                "unverified_single_candidates": unverified_builtin_candidates,
                "best_single_candidate": (
                    best_single.get("label") if best_single else None
                ),
                "reused_rbf_candidate": (
                    best_rbf.get("label") if best_rbf else None
                ),
            }
            return False

        assert best_single is not None
        assert best_rbf is not None
        compound_report = {
            "status": "running",
            "triggered": True,
            "target_guard": compound_target_guard,
            "sample_split": compound_split_report,
            "residual_gate": residual_gate,
            "best_single_candidate": best_single.get("label"),
            "reused_rbf_candidate": best_rbf.get("label"),
            "reused_rbf_command": list(best_rbf.get("command") or []),
        }
        pipeline.log.info(
            "[Stage3] Single-stage built-ins remain insufficient; "
            "try frozen-validation Polynomial→residual-RBF before external plugins"
        )
        base_record: Dict[str, Any] = {
            "label": label,
            "source": "compound",
            "phase": phase,
            "status": "running",
            "fallback_triggered": True,
            "best_single_candidate": best_single.get("label"),
            "reused_rbf_candidate": best_rbf.get("label"),
            "steps": [
                ["subsky", "1", "-existing"],
                list(best_rbf.get("command") or []),
            ],
            "sample_split": compound_split_report,
            "baseline_validation": baseline_validation,
        }

        def reject(
            status: str,
            reason: str,
            **details: Any,
        ) -> bool:
            record = {
                **base_record,
                **details,
                "status": status,
                "failure_reason": reason,
            }
            attempt_records.append(record)
            compound_report.update(
                {
                    "status": status,
                    "accepted": False,
                    "failure_reason": reason,
                    **details,
                }
            )
            pipeline.log.warn(
                f"[Stage3] Compound Polynomial→RBF candidate rejected: {reason}"
            )
            return False

        if not restore_baseline(f"before:{label}"):
            return reject(
                "rollback_unavailable",
                "immutable baseline could not be restored",
            )

        rollback_attempted = False
        rollback_completed = False
        try:
            poly_ok, poly_samples = _stage3_install_safe_background_samples(
                pipeline,
                compound_fit_points,
            )
            if not poly_ok:
                return reject(
                    "safe_sample_install_failed",
                    str(poly_samples.get("reason") or "Polynomial fit samples unavailable"),
                    polynomial_samples=poly_samples,
                )
            polynomial_command = ("subsky", "1", "-existing")
            command_ok, failure_reason = _stage3_try_background_command(
                pipeline,
                f"{label}-polynomial",
                polynomial_command,
                "compound",
            )
            if not command_ok:
                return reject(
                    "command_failed",
                    str(failure_reason or "Polynomial command failed"),
                    polynomial_samples=poly_samples,
                )
            intermediate_stem = "stage3_compound_poly_intermediate"
            if not pipeline._save_stage_output(intermediate_stem):
                return reject(
                    "intermediate_save_failed",
                    "Polynomial transaction intermediate could not be saved",
                    polynomial_samples=poly_samples,
                    intermediate_stem=intermediate_stem,
                )
            try:
                polynomial_image = pipeline.siril.get_image_pixeldata(preview=False)
            except (
                CommandError,
                SirilError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                return reject(
                    "validation_unavailable",
                    f"Polynomial intermediate pixels unavailable: {error}",
                    polynomial_samples=poly_samples,
                    intermediate_stem=intermediate_stem,
                )
            polynomial_validation = measure_background_validation(
                polynomial_image,
                compound_validation_points,
                patch_radius=_stage3_cfg_int(
                    pipeline,
                    "stage3_safe_sample_patch_radius",
                    12,
                    4,
                    24,
                ),
                minimum_count=_stage3_cfg_int(
                    pipeline,
                    "stage3_compound_validation_min_count",
                    8,
                    8,
                    20,
                ),
                value_scale=baseline_validation.get("value_scale"),
            )
            if polynomial_validation.get("status") != "ready":
                return reject(
                    "validation_unavailable",
                    "Polynomial held-out validation is unavailable",
                    polynomial_samples=poly_samples,
                    polynomial_validation=polynomial_validation,
                    intermediate_stem=intermediate_stem,
                )

            rbf_ok, rbf_samples = _stage3_install_safe_background_samples(
                pipeline,
                compound_fit_points,
            )
            if not rbf_ok:
                return reject(
                    "safe_sample_install_failed",
                    str(rbf_samples.get("reason") or "RBF fit samples unavailable"),
                    polynomial_samples=poly_samples,
                    rbf_samples=rbf_samples,
                    polynomial_validation=polynomial_validation,
                    intermediate_stem=intermediate_stem,
                )
            rbf_command = tuple(best_rbf.get("command") or ())
            command_ok, failure_reason = _stage3_try_background_command(
                pipeline,
                f"{label}-rbf",
                rbf_command,
                "compound",
            )
            if not command_ok:
                return reject(
                    "command_failed",
                    str(failure_reason or "residual RBF command failed"),
                    polynomial_samples=poly_samples,
                    rbf_samples=rbf_samples,
                    polynomial_validation=polynomial_validation,
                    intermediate_stem=intermediate_stem,
                )

            after_feat = pipeline._stage3_measure_features(label)
            try:
                after_image = pipeline.siril.get_image_pixeldata(preview=False)
            except (
                CommandError,
                SirilError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                after_image = None
                pipeline.log.debug(
                    f"stage3 compound candidate image sampling skipped: {error}"
                )
            preservation = pipeline._stage3_signal_preservation_metrics(
                before_image,
                after_image,
            )
            fidelity_ok, fidelity_gate = assess_target_fidelity(
                preservation,
                low_complexity_required=bool(
                    process_report.get("low_complexity_required", False)
                ),
            )
            fidelity_enforced = isinstance(input_profile, dict)
            gate_ok, gate_message = pipeline._stage3_quality_gate(
                before_feat,
                after_feat,
                preservation,
            )
            if fidelity_enforced and not fidelity_ok:
                gate_ok = False
                issues = ", ".join(fidelity_gate.get("issues") or [])
                gate_message = (
                    f"{gate_message}; target fidelity gate rejected"
                    + (f": {issues}" if issues else "")
                )
            if pattern_routing_enabled and after_image is not None:
                after_pattern_report = analyze_directional_pattern_noise(
                    after_image,
                    detection_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_pattern_score_min",
                        0.55,
                        0.25,
                        0.90,
                    ),
                    walking_threshold=_stage3_cfg_float(
                        pipeline,
                        "stage3_walking_noise_score_min",
                        0.50,
                        0.25,
                        0.90,
                    ),
                )
            else:
                after_pattern_report = {
                    "status": "unavailable",
                    "detected": False,
                }
            pattern_ok, pattern_gate_report = pattern_candidate_gate(
                pattern_report,
                after_pattern_report,
                growth_max=_stage3_cfg_float(
                    pipeline,
                    "stage3_pattern_score_growth_max",
                    0.12,
                    0.02,
                    0.40,
                ),
            )
            if not pattern_ok:
                gate_ok = False
                gate_message = (
                    f"{gate_message}; directional pattern gate rejected"
                )
            after_adaptive_candidate = (
                pipeline._adaptive_features_current()
                if hasattr(pipeline, "_adaptive_features_current")
                else {}
            )
            color_shift = _stage3_color_shift(
                before_adaptive,
                after_adaptive_candidate,
            )
            required_adaptive_metrics = {
                "bg_std",
                "gradient_score",
                "dirty_background_score",
                "chroma_noise_score",
            }
            adaptive_metrics_available = bool(
                required_adaptive_metrics.issubset(before_adaptive)
                and required_adaptive_metrics.issubset(after_adaptive_candidate)
            )
            color_shift_limit = _stage3_policy_float(
                stage3_policy,
                "sufficient_color_shift_max",
                0.18,
                minimum=0.0,
                maximum=2.0,
            )
            if color_shift > color_shift_limit:
                gate_ok = False
                gate_message = (
                    f"{gate_message}; color shift "
                    f"{color_shift:.3f}>{color_shift_limit:.3f}"
                )
            hard_gate_metrics_available = bool(
                before_feat is not None
                and after_feat is not None
                and after_image is not None
                and preservation.get("available")
                and adaptive_metrics_available
                and color_shift <= color_shift_limit
                and (
                    not pattern_routing_enabled
                    or (
                        pattern_report.get("status") == "ok"
                        and after_pattern_report.get("status") == "ok"
                    )
                )
            )
            if not hard_gate_metrics_available:
                gate_ok = False
                gate_message = (
                    f"{gate_message}; compound hard-gate metrics unavailable"
                )
            candidate_validation = (
                measure_background_validation(
                    after_image,
                    compound_validation_points,
                    patch_radius=_stage3_cfg_int(
                        pipeline,
                        "stage3_safe_sample_patch_radius",
                        12,
                        4,
                        24,
                    ),
                    minimum_count=_stage3_cfg_int(
                        pipeline,
                        "stage3_compound_validation_min_count",
                        8,
                        8,
                        20,
                    ),
                    value_scale=baseline_validation.get("value_scale"),
                )
                if after_image is not None
                else {"status": "unavailable", "reason": "pixels unavailable"}
            )
            common_details = {
                "polynomial_samples": poly_samples,
                "rbf_samples": rbf_samples,
                "intermediate_stem": intermediate_stem,
                "polynomial_validation": polynomial_validation,
                "validation": candidate_validation,
                "quality_message": gate_message,
                "preservation": preservation,
                "target_fidelity_gate": fidelity_gate,
                "target_fidelity_enforced": fidelity_enforced,
                "directional_pattern_noise": after_pattern_report,
                "pattern_quality_gate": pattern_gate_report,
                "after_adaptive": after_adaptive_candidate,
                "color_shift": color_shift,
                "hard_gate_metrics_available": hard_gate_metrics_available,
            }
            if not gate_ok:
                return reject(
                    "rejected",
                    gate_message,
                    **common_details,
                )

            background_score_components = _stage3_background_score_components(
                before_adaptive,
                after_adaptive_candidate,
            )
            base_candidate_score = float(background_score_components["total"])
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
            pattern_penalty = max(
                0.0,
                float(pattern_gate_report.get("pattern_score_growth", 0.0) or 0.0),
            ) * STAGE3_DIRECTIONAL_PATTERN_PENALTY_WEIGHT
            candidate_score = (
                base_candidate_score
                + preservation_penalty
                + pattern_penalty
            )
            validation_ok, validation_gate = (
                assess_compound_background_validation(
                    baseline_validation,
                    best_single.get("validation") or {},
                    polynomial_validation,
                    candidate_validation,
                    improvement_min=_stage3_cfg_float(
                        pipeline,
                        "stage3_compound_validation_improvement_min",
                        0.10,
                        0.10,
                        0.40,
                    ),
                    zero_point_abs_max=_stage3_cfg_float(
                        pipeline,
                        "stage3_compound_zero_point_abs_max",
                        0.01,
                        0.002,
                        0.010,
                    ),
                    zero_point_rel_max=_stage3_cfg_float(
                        pipeline,
                        "stage3_compound_zero_point_rel_max",
                        0.15,
                        0.05,
                        0.15,
                    ),
                )
            )
            score_ok, score_gate = _stage3_compound_score_gate(
                float(best_single.get("score", 999.0)),
                candidate_score,
                absolute_improvement_min=_stage3_cfg_float(
                    pipeline,
                    "stage3_compound_score_abs_improvement_min",
                    0.03,
                    0.03,
                    0.15,
                ),
                relative_improvement_min=_stage3_cfg_float(
                    pipeline,
                    "stage3_compound_score_rel_improvement_min",
                    0.10,
                    0.10,
                    0.40,
                ),
            )
            common_details.update(
                {
                    "base_background_score": base_candidate_score,
                    "background_score_components": background_score_components,
                    "preservation_penalty": preservation_penalty,
                    "directional_pattern_penalty": pattern_penalty,
                    "nebula_preservation_penalty_weight": (
                        nebula_preservation_weight
                    ),
                    "background_score": candidate_score,
                    "validation_gate": validation_gate,
                    "score_gate": score_gate,
                }
            )
            if not validation_ok:
                return reject(
                    "validation_rejected",
                    "; ".join(validation_gate.get("issues") or [
                        "held-out validation rejected the compound candidate"
                    ]),
                    **common_details,
                )
            if not score_ok:
                return reject(
                    "score_rejected",
                    "; ".join(score_gate.get("issues") or [
                        "compound score improvement is insufficient"
                    ]),
                    **common_details,
                )

            candidate_stem = _stage3_candidate_stem(label)
            if not pipeline._save_stage_output(candidate_stem):
                return reject(
                    "candidate_save_failed",
                    "validated compound candidate could not be saved",
                    **common_details,
                )
            _stage3_clear_background_samples(pipeline)
            rollback_attempted = True
            rollback_completed = restore_baseline(f"evaluated:{label}")
            if not rollback_completed:
                return reject(
                    "rollback_failed",
                    "validated compound candidate was invalidated because the immutable baseline could not be restored",
                    candidate_stem=candidate_stem,
                    **common_details,
                )
            sufficient = _stage3_candidate_sufficient(
                before_adaptive,
                after_adaptive_candidate,
                candidate_score,
                stage3_policy,
            )
            record = {
                **base_record,
                **common_details,
                "status": "accepted",
                "candidate_stem": candidate_stem,
                "sufficient": sufficient,
            }
            attempt_records.append(record)
            accepted_candidates.append(
                {
                    "label": label,
                    "source": "compound",
                    "phase": phase,
                    "command": list(rbf_command),
                    "stem": candidate_stem,
                    "base_score": base_candidate_score,
                    "background_score_components": background_score_components,
                    "preservation_penalty": preservation_penalty,
                    "directional_pattern_penalty": pattern_penalty,
                    "nebula_preservation_penalty_weight": (
                        nebula_preservation_weight
                    ),
                    "score": candidate_score,
                    "quality_message": gate_message,
                    "preservation": preservation,
                    "directional_pattern_noise": after_pattern_report,
                    "pattern_quality_gate": pattern_gate_report,
                    "after_adaptive": after_adaptive_candidate,
                    "color_shift": color_shift,
                    "validation": candidate_validation,
                    "validation_gate": validation_gate,
                    "score_gate": score_gate,
                    "hard_gate_metrics_available": True,
                    "sufficient": sufficient,
                }
            )
            compound_report.update(
                {
                    "status": "accepted" if sufficient else "accepted_degraded",
                    "accepted": True,
                    "sufficient": sufficient,
                    "candidate_stem": candidate_stem,
                    "intermediate_stem": intermediate_stem,
                    "validation_gate": validation_gate,
                    "score_gate": score_gate,
                    "background_score": candidate_score,
                }
            )
            pipeline.log.info(
                "[Stage3] Compound Polynomial→RBF candidate accepted: "
                f"score={candidate_score:.3f} sufficient={sufficient}"
            )
            return bool(sufficient)
        finally:
            if not rollback_attempted:
                _stage3_clear_background_samples(pipeline)
                restore_baseline(f"evaluated:{label}")

    mixed_pattern_route = bool(
        noise_route.get("route") == "mixed_gradient_and_pattern_noise"
    )
    if mixed_pattern_route:
        pipeline._background_review_required = True
        builtin_search_mode = "mixed_gradient_pattern_noise_review"
        pipeline.log.warn(
            "[Stage3] Directional pattern noise coexists with a supported gradient; "
            "only the low-frequency gradient may be modeled; pattern noise remains review-only"
        )

    rbf_attempts: List[Tuple[str, Tuple[str, ...], str]] = []
    for idx, raw_command in enumerate(
        pipeline._stage3_subsky_rbf_candidates(),
        start=1,
    ):
        command = tuple(raw_command)
        if "-existing" not in command:
            command += ("-existing",)
        rbf_attempts.append((f"subsky-rbf-existing-{idx}", command, "builtin"))
    poly_command = ("subsky", "1", "-existing")
    poly_attempt = [("subsky-poly-existing", poly_command, "builtin")]

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
        builtin_search_mode = (
            "mixed_gradient_pattern_noise_review"
            if mixed_pattern_route
            else "safe_samples_with_diffuse_signal_protection"
        )
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
    builtin_attempts_ordered = [
        record
        for record in ordered_attempts
        if record[2] == "builtin"
    ]
    external_attempts_ordered = [
        record
        for record in ordered_attempts
        if record[2] != "builtin"
    ]
    evaluate_attempts(
        builtin_attempts_ordered,
        phase="single_stage_builtin",
        stop_on_sufficient=True,
    )
    builtin_sufficient = any(
        record.get("source") == "builtin"
        and bool(record.get("sufficient"))
        for record in attempt_records
    )
    compound_sufficient = False
    if builtin_sufficient:
        compound_report = {
            "status": "not_required",
            "triggered": False,
            "reason": "single_stage_builtin_candidate_is_sufficient",
            "target_guard": compound_target_guard,
            "sample_split": compound_split_report,
        }
    else:
        compound_sufficient = evaluate_compound_candidate()
    if not builtin_sufficient and not compound_sufficient:
        evaluate_attempts(
            external_attempts_ordered,
            phase="external_fallback",
            stop_on_sufficient=True,
        )
    _stage3_clear_background_samples(pipeline)
    graxpert_attempted = any(
        record.get("source") == "graxpert" for record in attempt_records
    )
    graxpert_runtime_error = graxpert_runtime_error or any(
        record.get("status") == "graxpert_runtime_error"
        for record in attempt_records
    )

    if accepted_candidates:
        selected = min(
            accepted_candidates,
            key=lambda item: (
                not bool(item.get("sufficient")),
                float(item.get("score", 999.0)),
            ),
        )
        attempted_selected_label = str(selected.get("label") or "")
        selection_shadow = _stage3_statistical_shadow_selection(
            accepted_candidates,
            selected,
        )
        if selection_shadow.get("selection_would_change"):
            pipeline.log.info(
                "[Stage3] Statistical selection shadow differs: "
                f"runtime={selection_shadow.get('current_runtime_candidate')} "
                f"shadow={selection_shadow.get('shadow_recommended_candidate')} "
                "(observational only)"
            )
        selected_loaded = False
        try:
            pipeline.cmd_with_check("load", str(selected["stem"]))
            selected_loaded = True
        except (CommandError, SirilError) as e:
            pipeline.log.warn(
                "failed to load best stage3 candidate; restoring baseline: "
                f"{e}"
            )
            restore_baseline(f"selected_load_failed:{selected.get('label')}")
            failure_message = (
                f"selected candidate load failed: {selected.get('label')}"
            )
            stage_message = (
                f"{preflight_message}; {failure_message}"
                if preflight_message
                else failure_message
            )
        if selected_loaded:
            bg_ok = True
            selected_source = str(selected.get("source") or "")
            selected_label = str(selected.get("label") or "")
            compound_selected = selected_source == "compound"
            compound_selected_degraded = bool(
                compound_selected and not selected.get("sufficient")
            )
            if compound_selected_degraded:
                pipeline._background_review_required = True
            selected_preservation = selected.get("preservation") or {}
            selected_pattern_report = (
                selected.get("directional_pattern_noise") or {}
            )
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
            elif selected_source == "compound":
                pipeline.workflow_command_used["复合背景提取"] = str(selected.get("label"))
            pipeline.log.info(
                "背景提取最终选择: "
                f"{selected.get('label')} score={float(selected.get('score', 0.0)):.3f}"
            )
    elif not bg_ok:
        restore_baseline("no_candidate_accepted")
        pipeline._background_review_required = True
        pipeline.log.error("背景提取完全失败，图像可能有梯度残留")

    _stage3_clear_background_samples(pipeline)
    pipeline._stage3_pattern_noise_report = {
        "analysis": pattern_report,
        "route": noise_route,
        "selected_candidate": selected_pattern_report or None,
    }
    stage_saved = pipeline._save_stage_output("stage3_bgremoved")
    if bg_ok and stage_saved:
        final_output_validation = _stage3_final_output_validation(
            pipeline,
            baseline_validation=baseline_validation,
            validation_points=compound_validation_points,
            patch_radius=_stage3_cfg_int(
                pipeline,
                "stage3_safe_sample_patch_radius",
                12,
                4,
                24,
            ),
            minimum_count=_stage3_cfg_int(
                pipeline,
                "stage3_compound_validation_min_count",
                8,
                8,
                20,
            ),
            enforced=isinstance(input_profile, dict),
        )
        final_output_validation["selected_candidate"] = (
            attempted_selected_label or None
        )
        if (
            final_output_validation.get("enforced")
            and not final_output_validation.get("accepted", False)
        ):
            final_output_validation_rejected = True
            pipeline._background_review_required = True
            issues = ", ".join(
                str(issue)
                for issue in (
                    final_output_validation.get("validation_gate") or {}
                ).get("issues", [])
            )
            validation_message = (
                "final saved output rejected by held-out background/RMS gate"
                + (f": {issues}" if issues else "")
            )
            pipeline.log.warn(f"[Stage3] {validation_message}")
            stage_message = (
                f"{stage_message}; {validation_message}"
                if stage_message
                else validation_message
            )
            rollback_completed = restore_baseline(
                "final_output_validation_rejected"
            )
            final_output_validation["rollback"] = {
                "attempted": True,
                "completed": rollback_completed,
            }
            if not rollback_completed:
                raise RuntimeError(
                    "Stage3 final output failed its hard gate and the immutable "
                    "baseline could not be restored"
                )
            stage_saved = pipeline._save_stage_output("stage3_bgremoved")
            final_output_validation["rollback"]["output_saved"] = stage_saved
            bg_ok = False
            selected_source = ""
            selected_label = ""
            selected_preservation = {}
            selected_pattern_report = {}
            compound_selected = False
            compound_selected_degraded = False
            pipeline._stage3_pattern_noise_report["selected_candidate"] = None
    elif bg_ok:
        final_output_validation = {
            "status": "not_run",
            "enforced": isinstance(input_profile, dict),
            "selected_candidate": attempted_selected_label or None,
            "reason": "stage3 final output could not be saved",
        }
    else:
        final_output_validation = {
            "status": "not_applicable",
            "enforced": False,
            "reason": "no background candidate was selected",
        }
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
    if bg_ok and before_adaptive and after_adaptive:
        before_std = max(float(before_adaptive.get("bg_std", 0.0) or 0.0), 1e-7)
        after_std = float(after_adaptive.get("bg_std", 0.0) or 0.0)
        dirty = float(after_adaptive.get("dirty_background_score", 0.0) or 0.0)
        gradient_before = float(before_adaptive.get("gradient_score", 0.0) or 0.0)
        gradient_after = float(after_adaptive.get("gradient_score", 0.0) or 0.0)
        if after_std / before_std > max_bg_std_growth or (
            dirty > STAGE3_FINAL_DIRTY_WARNING_MIN
            and gradient_after
            >= gradient_before * STAGE3_FINAL_GRADIENT_RETENTION_WARNING
        ):
            fallback_warning = True
            pipeline._background_review_required = True
            warning_msg = (
                "background improvement limited "
                f"(dirty={dirty:.3f}, std_growth={after_std / before_std:.3f})"
            )
            pipeline.log.warn(f"[Stage3] {warning_msg}")
            stage_message = f"{stage_message}; {warning_msg}" if stage_message else warning_msg
    graxpert_runtime_fallback_used = bool(
        bg_ok and graxpert_runtime_error and selected_source != "graxpert"
    )
    custom_sample_attempted = any(
        record.get("source") == "builtin" for record in attempt_records
    )
    custom_sample_fallback_used = bool(
        bg_ok
        and selected_source not in ("", "builtin", "compound")
        and custom_sample_attempted
    )
    safe_sample_install_failed = any(
        record.get("status") == "safe_sample_install_failed"
        for record in attempt_records
    )
    safe_sample_report = {
        **safe_sample_report,
        "subsky_existing_enforced": True,
        "install_failed": safe_sample_install_failed,
        "selected_source": selected_source or None,
        "fallback_used": custom_sample_fallback_used,
    }
    pipeline._stage3_safe_sample_report = safe_sample_report
    stage_fallback_used = bool(
        profile_fallback_used
        or graxpert_runtime_fallback_used
        or custom_sample_fallback_used
        or compound_selected
        or final_output_validation_rejected
    )
    fallback_reasons = [
        reason
        for reason, enabled in (
            ("custom_safe_sample_candidate_fallback", custom_sample_fallback_used),
            ("compound_poly_residual_rbf_fallback", compound_selected),
            ("graxpert_runtime_fallback", graxpert_runtime_fallback_used),
            ("target_profiler_fallback", profile_fallback_used),
            (
                "final_output_validation_rejected_baseline_restored",
                final_output_validation_rejected,
            ),
        )
        if enabled
    ]
    pattern_review_required = bool(noise_route.get("requires_review", False))
    background_review_required = bool(
        pattern_review_required
        or compound_selected_degraded
        or final_output_validation_rejected
        or fallback_warning
    )
    report_quality = (
        "review_required"
        if background_review_required
        else "ok"
        if bg_ok
        else "degraded"
    )
    if pattern_review_required:
        pattern_note = (
            "directional pattern noise remains unresolved; "
            f"route={noise_route.get('route')}"
        )
        stage_message = (
            f"{stage_message}; {pattern_note}"
            if stage_message
            else pattern_note
        )
    if compound_selected_degraded:
        compound_note = (
            "compound Polynomial→RBF fallback passed safety validation but "
            "did not reach sufficient background quality; review-only output required"
        )
        stage_message = (
            f"{stage_message}; {compound_note}"
            if stage_message
            else compound_note
        )
    if hasattr(pipeline, "_write_stage_json"):
        pipeline._write_stage_json(
            "background_quality_report.json",
            {
                "schema_version": STAGE3_BACKGROUND_QUALITY_SCHEMA,
                "algorithm_contract_version": STAGE3_ALGORITHM_CONTRACT_VERSION,
                "stage": "stage3_background",
                "policy": policy_name,
                "decision_thresholds": decision_thresholds,
                "decision": background_decision,
                "process_evidence": process_report,
                "model_used": selected_label or None,
                "attempted_selected_model": attempted_selected_label or None,
                "graxpert_attempted": graxpert_attempted,
                "graxpert_runtime_error": graxpert_runtime_error,
                "graxpert_error_reasons": graxpert_error_reasons,
                "fallback_triggered_by_graxpert_error": bool(
                    graxpert_runtime_error and selected_source != "graxpert"
                ),
                "builtin_order_reason": builtin_order_reason,
                "candidate_order": [record[0] for record in ordered_attempts],
                "evaluated_candidate_order": [
                    str(record.get("label") or "")
                    for record in attempt_records
                ],
                "builtin_candidate_order": builtin_attempt_labels,
                "builtin_search_mode": builtin_search_mode,
                "builtin_sufficient": builtin_sufficient,
                "compound_fallback": compound_report,
                "diffuse_nebula_context": diffuse_context,
                "safe_samples": safe_sample_report,
                "subsky_existing_enforced": True,
                "directional_pattern_noise": pattern_report,
                "selected_directional_pattern_noise": (
                    selected_pattern_report or None
                ),
                "noise_route": noise_route,
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
                "selection_shadow": selection_shadow,
                "final_output_validation": final_output_validation,
                "rollback_events": rollback_events,
                "selected_preservation": selected_preservation,
                "quality": report_quality,
                "fallback_used": stage_fallback_used,
                "fallback_reasons": fallback_reasons,
                "fallback_reason": (
                    "final_output_validation_rejected_baseline_restored"
                    if final_output_validation_rejected
                    else "compound_poly_residual_rbf_fallback"
                    if compound_selected
                    else "graxpert_runtime_fallback"
                    if graxpert_runtime_fallback_used
                    else "custom_safe_sample_candidate_fallback"
                    if custom_sample_fallback_used
                    else "target_profiler_fallback"
                    if profile_fallback_used
                    else None
                ),
            },
        )
    if not stage_saved:
        stage_message = (
            f"{stage_message}; stage3 输出保存失败"
            if stage_message
            else "stage3 输出保存失败"
        )
    elif hasattr(pipeline, "_create_stage_review_bundle"):
        review = pipeline._create_stage_review_bundle(
            "stage3_background_extraction",
            baseline_stem,
            "stage3_bgremoved",
            context={
                "method": selected_label or None,
                "quality": report_quality,
                "noise_route": noise_route.get("route"),
            },
            candidates=attempt_records,
            selected_candidate=selected_label or None,
        )
        if review.get("report_path"):
            review_note = f"review_bundle={review['report_path']}"
            stage_message = f"{stage_message}; {review_note}" if stage_message else review_note

    elapsed = pipeline.log.stage_end(stage_label)
    components = {
        "target_profile": {
            "status": "applied",
            "method": target_profile.get("classification_method"),
            "reason_code": (
                "target_profiler_fallback"
                if profile_fallback_used
                else "accepted"
            ),
            "fallback_used": profile_fallback_used,
        },
        "background_extraction": {
            "status": (
                "rolled_back"
                if final_output_validation_rejected
                else "review_required"
                if compound_selected_degraded
                else "applied"
                if bg_ok
                else "rolled_back"
            ),
            "method": selected_label or None,
            "attempted_method": attempted_selected_label or None,
            "reason_code": (
                "final_output_validation_rejected"
                if final_output_validation_rejected
                else "compound_poly_residual_rbf_degraded_review"
                if compound_selected_degraded
                else "compound_poly_residual_rbf_fallback_accepted"
                if compound_selected
                else "accepted"
                if bg_ok
                else "no_candidate_accepted"
            ),
            "input": baseline_stem,
            "output": "stage3_bgremoved" if stage_saved else None,
            "fallback_used": bool(
                graxpert_runtime_fallback_used
                or custom_sample_fallback_used
                or compound_selected
                or final_output_validation_rejected
            ),
        },
        "directional_pattern_router": {
            "status": (
                "review_required" if pattern_review_required else "accepted"
            ),
            "method": noise_route.get("route"),
            "reason_code": (
                "mixed_gradient_pattern_noise_review"
                if noise_route.get("route") == "mixed_gradient_and_pattern_noise"
                else "no_directional_pattern_detected"
            ),
            "fallback_used": False,
        },
    }
    reason_code = (
        "final_output_validation_rejected"
        if final_output_validation_rejected
        else "no_background_candidate_accepted"
        if not bg_ok
        else "stage3_output_save_failed"
        if not stage_saved
        else "mixed_gradient_pattern_noise_review"
        if pattern_review_required
        else "compound_poly_residual_rbf_degraded_review"
        if compound_selected_degraded
        else "compound_poly_residual_rbf_fallback_accepted"
        if compound_selected
        else "graxpert_runtime_fallback"
        if graxpert_runtime_fallback_used
        else "custom_safe_sample_candidate_fallback"
        if custom_sample_fallback_used
        else "target_profiler_fallback"
        if profile_fallback_used
        else "background_improvement_limited"
        if fallback_warning
        else ""
    )
    if bg_ok:
        status = (
            "degraded"
            if background_review_required or not stage_saved
            else "ok"
        )
        pipeline._record_stage(
            stage_label,
            status,
            elapsed,
            stage_message,
            fallback_used=stage_fallback_used,
            reason_code=reason_code,
            components=components,
        )
        if selected_source == "builtin":
            pipeline.log.info("阶段3按策略使用内置 subsky/RBF 背景提取")
    else:
        degrade_message = (
            stage_message
            if final_output_validation_rejected and stage_message
            else "背景提取失败，图像可能有梯度残留"
        )
        if not stage_saved:
            degrade_message += "；stage3 输出保存失败"
        pipeline._record_stage(
            stage_label,
            "degraded",
            elapsed,
            degrade_message,
            execution="safe_passthrough",
            fallback_used=stage_fallback_used,
            reason_code=reason_code,
            components=components,
        )
