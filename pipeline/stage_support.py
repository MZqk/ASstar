"""Service mixins for StarunPostProcessor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import textwrap
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import cosmic_clarity
import plugin_runner
import review_bundle
import sasp_runner
import scunet_denoise
import syqon_starless
import stage7_quality
import stage7_repair
import stage8_pixels
import stage9_quality
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    _to_rgb_float_image,
    format_feature_summary,
    measure_image_features,
    measure_quality_metrics,
    measure_stage3_signal_preservation,
)
from models import ImageFeatures, QualityMetrics, StageResult, TargetType
from save_utils import save_stage_output, write_stage_json

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except ImportError:
    CommandError = RuntimeError
    DataError = RuntimeError
    SirilError = RuntimeError

try:
    from image_feature_analyzer import analyze_image as analyze_adaptive_image
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from target_profiler import build_target_profile
except (ImportError, RuntimeError):
    analyze_adaptive_image = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage7_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_target_profile = None

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "STARUN_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "STARUN_INPUT_MODE"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "stage5_linear_resume"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)

def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))

class PluginServiceMixin:
    def _run_first_available_command(
        self,
        step_key: str,
        candidates: List[Tuple[str, Tuple[str, ...]]],
        *,
        allow_when_probe_disabled: bool = False,
    ) -> Optional[str]:
        return plugin_runner.run_first_available_command(
            self, step_key, candidates, allow_when_probe_disabled=allow_when_probe_disabled
        )


    def _quote_siril_arg(self, value: Path | str) -> str:
        return plugin_runner.quote_siril_arg(self, value)


    def _resolve_siril_scripts_root(self) -> Optional[Path]:
        return plugin_runner.resolve_siril_scripts_root(self)


    def _find_plugin_script(self, relative_candidates: Tuple[str, ...]) -> Optional[Path]:
        return plugin_runner.find_plugin_script(self, relative_candidates)


    def _is_python_module_available(self, module_name: str) -> bool:
        return plugin_runner.is_python_module_available(self, module_name)


    def _validate_plugin_script_prerequisites(
        self,
        script_path: Path,
        python_executable: Optional[str] = None,
    ) -> Tuple[bool, str]:
        return plugin_runner.validate_plugin_script_prerequisites(
            self,
            script_path,
            python_executable,
        )


    def _run_plugin_script_by_path(
        self,
        step_key: str,
        label: str,
        script_path: Path,
        *,
        args: Tuple[str, ...] = (),
    ) -> Optional[str]:
        return plugin_runner.run_plugin_script_by_path(
            self, step_key, label, script_path, args=args
        )


    def _current_image_fingerprint(self) -> Optional[str]:
        return plugin_runner.current_image_fingerprint(self)


    def _plugin_output_failure_reason(self, script_name: str, output_text: str) -> Optional[str]:
        return plugin_runner.plugin_output_failure_reason(self, script_name, output_text)


    def _classic_cosmic_clarity_args(
        self,
        config_name: str,
        label: str,
    ) -> Optional[Tuple[str, ...]]:
        return cosmic_clarity.classic_cosmic_clarity_args(self, config_name, label)


    def _classic_cosmic_clarity_candidate_error(self, candidate: Path) -> Optional[str]:
        return cosmic_clarity.classic_cosmic_clarity_candidate_error(self, candidate)


    def _classic_cosmic_clarity_auto_candidates(self) -> List[Path]:
        return cosmic_clarity.classic_cosmic_clarity_auto_candidates(self)


    def _persist_classic_cosmic_clarity_config(
        self,
        config_path: Path,
        executable: Path,
        label: str,
    ) -> None:
        return cosmic_clarity.persist_classic_cosmic_clarity_config(
            self,
            config_path,
            executable,
            label,
        )


    def _classic_cosmic_clarity_device_args(self) -> Tuple[Tuple[str, ...], str]:
        return cosmic_clarity.classic_cosmic_clarity_device_args(self)


    def _cosmic_clarity_native_sharpen_cli_options(self) -> Tuple[Tuple[str, ...], str]:
        return cosmic_clarity.cosmic_clarity_native_sharpen_cli_options(self)


    def _run_cosmic_clarity_native_sharpen_fallback(self, step_key: str) -> Optional[str]:
        return cosmic_clarity.run_cosmic_clarity_native_sharpen_fallback(self, step_key)


    def _run_plugin_script_cli_subprocess(
        self,
        step_key: str,
        label: str,
        script_path: Path,
        *,
        args: Tuple[str, ...] = (),
        timeout_sec: int = 1800,
        verify_image_change: bool = True,
        uses_siril_connection: bool = True,
    ) -> Optional[str]:
        return plugin_runner.run_plugin_script_cli_subprocess(
            self,
            step_key,
            label,
            script_path,
            args=args,
            timeout_sec=timeout_sec,
            verify_image_change=verify_image_change,
            uses_siril_connection=uses_siril_connection,
        )


    def _run_siril_cc_sharpen_fallback(self, step_key: str) -> Optional[str]:
        """
        尝试以纯命令方式执行 Siril-CC Sharpen Both 0.1/3/0.5 回退链。
        参数约定按用户请求固定为:
        - mode: Both
        - non-stellar amount: 0.1
        - non-stellar strength: 3
        - stellar amount: 0.5
        """
        return self._run_first_available_command(
            step_key,
            [
                (
                    "Siril-CC Sharpen Both 0.1/3/0.5",
                    ("siril_cc_sharpen", "Both", "0.5", "0.1", "3"),
                ),
                (
                    "Siril-CC Sharpen Both 0.1/3/0.5",
                    ("siril_cc_sharpen", "Both", "0.1", "3", "0.5"),
                ),
                (
                    "Siril-CC Sharpen Both 0.1/3/0.5",
                    (
                        "siril_cc_sharpen",
                        "-mode=Both",
                        "-stellar_amount=0.5",
                        "-non_stellar_amount=0.1",
                        "-non_stellar_strength=3",
                    ),
                ),
                (
                    "Siril-CC Sharpen Both 0.1/3/0.5",
                    ("siril_cc_sharpen_both", "0.1", "3", "0.5"),
                ),
                (
                    "Siril-CC Sharpen Both 0.1/3/0.5",
                    ("siril_cc_sharpen_both", "0.5", "0.1", "3"),
                ),
            ],
            allow_when_probe_disabled=True,
        )


    def _run_siril_scunet_denoise_fallback(
        self, step_key: str, strength: float
    ) -> Optional[str]:
        return scunet_denoise.run_siril_scunet_denoise_fallback(
            self,
            step_key,
            strength,
            command_error_types=(CommandError,),
            recoverable_error_types=(SirilError, DataError),
        )


    def _final_denoise_cli_timeout_sec(self) -> int:
        return cosmic_clarity.final_denoise_cli_timeout_sec(self)


    def _cosmic_clarity_native_denoise_cli_options(self) -> Tuple[Tuple[str, ...], str]:
        return cosmic_clarity.cosmic_clarity_native_denoise_cli_options(self)


    def _run_cosmic_clarity_native_denoise_fallback(self, step_key: str) -> Optional[str]:
        return cosmic_clarity.run_cosmic_clarity_native_denoise_fallback(self, step_key)


    def _syqon_starless_cli_options(
        self,
        *,
        profile: syqon_starless.SyQonAttemptProfile = (
            syqon_starless.SYQON_BASELINE_PROFILE
        ),
    ) -> Tuple[Tuple[str, ...], int, str]:
        return syqon_starless.syqon_starless_cli_options(
            self,
            profile=profile,
        )


    def _count_sequence_products(self, prefix: str) -> int:
        if not self.process_dir:
            return 0
        seen: set[Path] = set()
        for ext in ("fit", "fits"):
            for path in self.process_dir.glob(f"{prefix}_*.{ext}"):
                if path.is_file():
                    seen.add(path)
        return len(seen)


    def _clear_star_separation_outputs(self) -> None:
        syqon_starless.clear_star_separation_outputs(self)


    def _fallback_summary(
        self,
        primary: str,
        reason: str,
        fallback: str,
        success: bool,
    ) -> str:
        return plugin_runner.fallback_summary(self, primary, reason, fallback, success)


    def _is_classic_cc_not_configured(self, reason: str) -> bool:
        return plugin_runner.is_classic_cc_not_configured(self, reason)


    def _is_siril_connection_failure(self, value: object) -> bool:
        return plugin_runner.is_siril_connection_failure(self, value)


    def _subprocess_output_tail(self, output_text: str, max_lines: int = 12) -> str:
        return plugin_runner.subprocess_output_tail(
            self,
            output_text,
            max_lines=max_lines,
        )


class Stage7ServiceMixin:
    def _stage7_preflight_summary(self, preflight: Dict[str, Any]) -> str:
        metrics = preflight.get("metrics") if isinstance(preflight, dict) else None
        if not isinstance(metrics, dict):
            return "stage7 preflight unavailable"
        return (
            f"stage7 preflight {preflight.get('risk_level', 'unknown')}: "
            f"edge_black={float(metrics.get('edge_black_ratio', 0.0)):.3f}, "
            f"bg_median={float(metrics.get('bg_median', 0.0)):.4f}, "
            f"bg_std={float(metrics.get('bg_std', 0.0)):.4f}, "
            f"star_size={float(metrics.get('median_star_size', 0.0)):.3f}"
        )


    def _stage7_preflight_check(self) -> Dict[str, Any]:
        source_stem = self.stretched_name or "stage7_stretched"
        result: Dict[str, Any] = {
            "source": source_stem,
            "risk_level": "ok",
            "issues": [],
            "recommendations": [],
            "metrics": None,
            "initial_metrics": None,
        }

        def add_issue(level: str, issue: str, recommendation: str) -> None:
            issues = result.setdefault("issues", [])
            recommendations = result.setdefault("recommendations", [])
            if issue not in issues:
                issues.append(issue)
            if recommendation and recommendation not in recommendations:
                recommendations.append(recommendation)
            if level == "high":
                result["risk_level"] = "high"
            elif level == "warn" and result.get("risk_level") == "ok":
                result["risk_level"] = "warn"

        def collect_metrics(
            features: ImageFeatures,
            quality: QualityMetrics,
            star_growth_ratio: Optional[float],
        ) -> Dict[str, Optional[float]]:
            bg_noise_ratio = features.bg_std / max(features.bg_median, 1e-4)
            return {
                "edge_black_ratio": float(features.edge_black_ratio),
                "bg_median": float(features.bg_median),
                "bg_std": float(features.bg_std),
                "bg_noise_ratio": float(bg_noise_ratio),
                "median_star_size": float(quality.median_star_size),
                "star_growth_ratio": (
                    float(star_growth_ratio)
                    if star_growth_ratio is not None
                    else None
                ),
            }

        try:
            self.cmd_with_check("load", source_stem)
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                add_issue("warn", "preflight_sampling_unavailable", "continue with standard starless flow")
                self.log.warn("Stage7 preflight skipped: image buffer is empty")
                return result

            features = measure_image_features(np.asarray(image_data))
            quality = measure_quality_metrics(np.asarray(image_data))
            baseline_quality: Optional[QualityMetrics] = None
            if self.process_dir and (self.process_dir / "stage6_input.fit").exists():
                baseline_data = self._read_image_by_stem("stage6_input")
                if baseline_data is not None:
                    baseline_quality = measure_quality_metrics(baseline_data)

            star_growth_ratio: Optional[float] = None
            if (
                baseline_quality
                and baseline_quality.median_star_size > 0.2
                and quality.median_star_size > 0
            ):
                star_growth_ratio = (
                    quality.median_star_size
                    / max(baseline_quality.median_star_size, 1e-4)
                )

            initial_metrics = collect_metrics(features, quality, star_growth_ratio)
            result["initial_metrics"] = dict(initial_metrics)
            result["metrics"] = dict(initial_metrics)

            if features.edge_black_ratio > self.cfg.stage7_edge_black_warn:
                add_issue(
                    "warn",
                    (
                        "edge_black "
                        f"{features.edge_black_ratio:.3f}>{self.cfg.stage7_edge_black_warn:.3f}"
                    ),
                    "rerun from stage2 so black borders are removed before background/stretch/starless stages",
                )

            final_metrics = result.get("metrics") or initial_metrics
            edge_black = float(final_metrics.get("edge_black_ratio") or 0.0)
            bg_median = float(final_metrics.get("bg_median") or 0.0)
            bg_std = float(final_metrics.get("bg_std") or 0.0)
            bg_noise_ratio = float(final_metrics.get("bg_noise_ratio") or 0.0)

            if edge_black > self.cfg.stage7_edge_black_high:
                add_issue(
                    "high",
                    (
                        "edge_black_high "
                        f"{edge_black:.3f}>{self.cfg.stage7_edge_black_high:.3f}"
                    ),
                    "prefer cropped or conservative starless input on future retry",
                )

            bg_level_high = (
                bg_median > self.cfg.stage7_bg_median_high
                and bg_std > self.cfg.stage7_bg_std_high
            )
            bg_ratio_high = bg_noise_ratio > self.cfg.stage7_bg_noise_ratio_high
            if bg_level_high or bg_ratio_high:
                level = "high" if bg_level_high and bg_ratio_high else "warn"
                add_issue(
                    level,
                    (
                        "bright_noisy_background "
                        f"bg_median={bg_median:.4f}, bg_std={bg_std:.4f}, "
                        f"ratio={bg_noise_ratio:.3f}"
                    ),
                    "use lighter pre-starless stretch or conservative starless parameters on retry",
                )

            if star_growth_ratio is not None:
                growth_limit = float(self.cfg.stage7_star_growth_ratio_max)
                if star_growth_ratio > growth_limit:
                    level = "high" if star_growth_ratio > growth_limit * 1.20 else "warn"
                    add_issue(
                        level,
                        (
                            "star_growth "
                            f"{star_growth_ratio:.3f}>{growth_limit:.3f}"
                        ),
                        "protect or shrink bloated stars before future starless retry",
                    )

            self.cmd_with_check("load", source_stem)
            summary = self._stage7_preflight_summary(result)
            if result["risk_level"] == "ok":
                self.log.info(summary)
            else:
                self.log.warn(summary + "; " + "; ".join(result.get("issues", [])[:3]))
            return result
        except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
            add_issue("warn", "preflight_failed", "continue with standard starless flow")
            result["error"] = self._short_text(e, 180)
            self.log.warn(f"Stage7 preflight failed: {e}")
            try:
                self.cmd_with_check("load", source_stem)
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                self.log.warn(f"Stage7 preflight input restore failed: {e}")
            return result


    def _stage7_starless_artifact_scores(
        self,
        source_data: Optional[np.ndarray],
        starless_data: Optional[np.ndarray],
        starmask_data: Optional[np.ndarray],
        source_features: ImageFeatures,
        starless_features: ImageFeatures,
    ) -> Dict[str, float]:
        return stage7_quality.stage7_starless_artifact_scores(
            self,
            source_data,
            starless_data,
            starmask_data,
            source_features,
            starless_features,
        )


    def _stage7_clean_starmask(
        self,
        *,
        label: str = "initial",
        source_stem: Optional[str] = None,
    ) -> Dict[str, Any]:
        return stage7_repair.stage7_clean_starmask(
            self,
            label=label,
            source_stem=source_stem,
        )


    def _stage7_quality_assessment(
        self,
        attempt_name: str,
        *,
        tool_label: str,
        source_stem: Optional[str] = None,
    ) -> Dict[str, Any]:
        return stage7_quality.stage7_quality_assessment(
            self,
            attempt_name,
            tool_label=tool_label,
            source_stem=source_stem,
        )


    def _stage7_prepare_starmask(self) -> None:
        if not self.starmask_file and not self._build_manual_starmask():
            self.log.warn("手动星点层失败，未生成规范 starmask_raw.fit")


    def _stage7_try_syqon_variant(
        self,
        syqon_script: Path,
        *,
        attempt_name: str,
        profile: syqon_starless.SyQonAttemptProfile = (
            syqon_starless.SYQON_BASELINE_PROFILE
        ),
    ) -> Optional[str]:
        return syqon_starless.stage7_try_syqon_variant(
            self,
            syqon_script,
            attempt_name=attempt_name,
            profile=profile,
        )


    def _stage7_quality_score(self, quality: Optional[Dict[str, Any]]) -> float:
        return stage7_quality.stage7_quality_score(self, quality)


    def _stage7_snapshot_current_outputs(self, suffix: str) -> Dict[str, Optional[str]]:
        snapshot: Dict[str, Optional[str]] = {
            "starless": None,
            "starmask": None,
            "starmask_raw": None,
            "starmask_kind": None,
            "pair_id": getattr(self, "_selected_syqon_pair_id", None),
            "attempt_id": getattr(self, "_selected_syqon_attempt_id", None),
        }
        if not self.process_dir:
            return snapshot
        if self.starless_file and self.starless_file.exists():
            target = self.process_dir / f"starless_{suffix}.fit"
            syqon_starless._atomic_copy(self.starless_file, target)
            snapshot["starless"] = target.stem
        if self.starmask_file and self.starmask_file.exists():
            target = self.process_dir / f"starmask_{suffix}.fit"
            syqon_starless._atomic_copy(self.starmask_file, target)
            snapshot["starmask"] = target.stem
            snapshot["starmask_kind"] = (
                "clean" if self.starmask_file.stem == "starmask_clean" else "raw"
            )
        raw_starmask = self.process_dir / "starmask_raw.fit"
        if raw_starmask.exists():
            target_raw = self.process_dir / f"starmask_raw_{suffix}.fit"
            syqon_starless._atomic_copy(raw_starmask, target_raw)
            snapshot["starmask_raw"] = target_raw.stem
        return snapshot


    def _stage7_restore_snapshot(self, snapshot: Dict[str, Optional[str]]) -> None:
        if not self.process_dir or not snapshot.get("starless"):
            return
        starless_src = self.process_dir / f"{snapshot['starless']}.fit"
        target_starless = self.process_dir / "starless.fit"
        restore_plan: List[Tuple[Path, Path]] = [(starless_src, target_starless)]
        raw_starmask_stem = snapshot.get("starmask_raw")
        if raw_starmask_stem:
            raw_src = self.process_dir / f"{raw_starmask_stem}.fit"
            restore_plan.append((raw_src, self.process_dir / "starmask_raw.fit"))
        starmask_stem = snapshot.get("starmask")
        restored_starmask: Optional[Path] = None
        if starmask_stem:
            starmask_src = self.process_dir / f"{starmask_stem}.fit"
            if snapshot.get("starmask_kind") == "clean":
                target_starmask = self.process_dir / "starmask_clean.fit"
            else:
                target_starmask = self.process_dir / "starmask_raw.fit"
            restore_plan.append((starmask_src, target_starmask))
            restored_starmask = target_starmask

        missing_sources = [source for source, _target in restore_plan if not source.is_file()]
        if missing_sources:
            missing_names = ", ".join(source.name for source in missing_sources)
            raise FileNotFoundError(f"Stage7 snapshot is incomplete: {missing_names}")

        # Validate every source before replacing any live output.  A corrupt
        # snapshot must not leave starless and starmask from different retries.
        for source, target in restore_plan:
            syqon_starless._atomic_copy(source, target)
        self.starless_file = target_starless
        self.starmask_file = restored_starmask
        self._selected_syqon_pair_id = snapshot.get("pair_id")
        self._selected_syqon_attempt_id = snapshot.get("attempt_id")


    def _stage7_repair_triggers(self, quality: Optional[Dict[str, Any]]) -> List[str]:
        return stage7_quality.stage7_repair_triggers(self, quality)


    def _stage7_try_syqon_with_source(
        self,
        syqon_script: Path,
        *,
        source_stem: str,
        attempt_name: str,
        profile: syqon_starless.SyQonAttemptProfile = (
            syqon_starless.SYQON_BASELINE_PROFILE
        ),
    ) -> Optional[str]:
        return syqon_starless.stage7_try_syqon_with_source(
            self,
            syqon_script,
            source_stem=source_stem,
            attempt_name=attempt_name,
            profile=profile,
        )


    def _stage7_update_star_remix_from_quality(
        self,
        quality: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return stage7_quality.stage7_update_star_remix_from_quality(self, quality)


    def _stage7_residual_suppression_strength(
        self,
        quality: Optional[Dict[str, Any]],
    ) -> float:
        return stage7_quality.stage7_residual_suppression_strength(self, quality)


    def _apply_stage7_residual_suppression(
        self,
        strength: float,
        *,
        source_stem: Optional[str] = None,
    ) -> Optional[str]:
        return stage7_repair.apply_stage7_residual_suppression(
            self,
            strength,
            source_stem=source_stem,
        )


    def _apply_stage7_starless_pixel_repair(
        self,
        *,
        source_stem: str,
        label: str,
    ) -> Dict[str, Any]:
        return stage7_repair.apply_stage7_starless_pixel_repair(
            self,
            source_stem=source_stem,
            label=label,
        )


class SaspServiceMixin:
    def _find_latest_sasp_wheel(self) -> Optional[Path]:
        return sasp_runner.find_latest_sasp_wheel(self)


    def _install_pyqt6_headless_stub(self) -> bool:
        return sasp_runner.install_pyqt6_headless_stub(self)


    def _load_sasp_aberration_module(self):
        return sasp_runner.load_sasp_aberration_module(self)


    def _prepare_aberration_input(self, image_data):
        return sasp_runner.prepare_aberration_input(self, image_data)


    def _restore_aberration_output(self, output_data, layout: str, src_dtype):
        return sasp_runner.restore_aberration_output(self, output_data, layout, src_dtype)


    def _resolve_local_aberration_model(self) -> Optional[Path]:
        return sasp_runner.resolve_local_aberration_model(self)


    def _preferred_aberration_providers(self, module) -> Tuple[Optional[List[str]], str]:
        return sasp_runner.preferred_aberration_providers(self, module)


    def _run_aberration_api(self, step_key: str, model_path: Optional[Path] = None):
        return sasp_runner.run_aberration_api(self, step_key, model_path=model_path)


    def _load_sasp_stage8_module(self):
        return sasp_runner.load_sasp_stage8_module(self)


    def _install_sasp_stage8_widget_import_shims(self, wheel_path: Path) -> None:
        return sasp_runner.install_sasp_stage8_widget_import_shims(self, wheel_path)


    def _prepare_stage8_sasp_input(self, image_data):
        return sasp_runner.prepare_stage8_sasp_input(self, image_data)


    def _restore_stage8_sasp_output(
        self,
        output_data,
        layout: str,
        src_dtype,
        scale_back: Optional[float],
    ):
        return sasp_runner.restore_stage8_sasp_output(self, output_data, layout, src_dtype, scale_back)


    def _run_sasp_stage8_api(self, plan: Optional[Dict[str, Any]] = None):
        return sasp_runner.run_sasp_stage8_api(self, plan)


    def _load_sasp_star_stretch_module(self):
        return sasp_runner.load_sasp_star_stretch_module(self)


    def _run_sasp_star_stretch_api(self):
        return sasp_runner.run_sasp_star_stretch_api(self)


    def _run_nb_to_rgb_stars_api(self):
        return sasp_runner.run_nb_to_rgb_stars_api(self)


class Stage8ServiceMixin:
    def _apply_starless_blue_guard(self, feat: ImageFeatures) -> Optional[str]:
        blue_excess = feat.blue_dominance - max(1.08, feat.red_dominance + 0.12)
        if blue_excess <= 0:
            return None
        b_gain = _clamp_float(1.0 / max(feat.blue_dominance, 1e-6), 0.86, 0.98)
        self.cmd_with_check(
            "ccm",
            "1.000000",
            "0",
            "0",
            "0",
            "1.000000",
            "0",
            "0",
            "0",
            f"{b_gain:.6f}",
        )
        return (
            "starless blue guard applied "
            f"(blue_dom={feat.blue_dominance:.3f}, red_dom={feat.red_dominance:.3f}, "
            f"b_gain={b_gain:.3f})"
        )


    def _stage8_restore_rgb_like(self, source_data: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        return stage8_pixels.stage8_restore_rgb_like(self, source_data, rgb)


    def _stage8_soften_mask(self, mask: np.ndarray, passes: int = 3) -> np.ndarray:
        return stage8_pixels.stage8_soften_mask(self, mask, passes=passes)


    def _stage8_generate_starless_masks(self, image_data: np.ndarray) -> Dict[str, Any]:
        return stage8_pixels.stage8_generate_starless_masks(self, image_data)


    def _stage8_starless_readiness_report(
        self,
        image_data: np.ndarray,
        masks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return stage8_pixels.stage8_starless_readiness_report(
            self,
            image_data,
            masks,
        )


    def _stage8_masked_metrics(
        self,
        image_data: Optional[np.ndarray],
        masks: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        return stage8_pixels.stage8_masked_metrics(self, image_data, masks)


    def _background_quality_metrics(
        self,
        image_data: Optional[np.ndarray],
        masks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return stage8_pixels.background_quality_metrics(self, image_data, masks)


    def _stage8_enhancement_quality_report(self) -> Dict[str, Any]:
        return stage8_pixels.stage8_enhancement_quality_report(self)


    def _rollback_stage8_to_input(self) -> bool:
        return stage8_pixels.rollback_stage8_to_input(self)


    def _final_quality_report(self, stem: str = "stage10_final") -> Dict[str, Any]:
        return stage8_pixels.final_quality_report(self, stem)


    def _stage7_halo_residue_score(self) -> float:
        quality = getattr(self, "_stage7_selected_quality", None)
        derived = quality.get("derived") if isinstance(quality, dict) else None
        if isinstance(derived, dict):
            try:
                return float(derived.get("halo_residue_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0


    def _stage7_effective_halo_threshold(self) -> float:
        base = float(self.cfg.stage7_halo_residue_score_max)
        target_type = self._active_target_type() if hasattr(self, "_active_target_type") else ""
        if target_type == "large_galaxy":
            return max(
                base,
                float(self.cfg.stage7_large_galaxy_halo_residue_score_max),
            )
        if target_type == "bright_emission_reflection_nebula":
            return max(base, float(self.cfg.stage7_bright_nebula_halo_residue_score_max))
        if target_type in stage7_quality.DIFFUSE_EMISSION_NEBULA_TARGET_TYPES:
            return max(
                base,
                stage7_quality.DIFFUSE_EMISSION_HALO_RESIDUE_SCORE_MAX,
            )
        if stage7_quality.stage7_has_diffuse_nebula_context(self):
            return max(
                base,
                stage7_quality.DIFFUSE_EMISSION_HALO_RESIDUE_SCORE_MAX,
            )
        return base


    def _stage8_input_enhancement_guard(self) -> Dict[str, Any]:
        return stage8_pixels.stage8_input_enhancement_guard(self)

    def _box_blur_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Return a deterministic 3x3 RGB blur used by local repair paths."""
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"expected RGB CHW array, got shape={arr.shape}")
        height, width = arr.shape[1], arr.shape[2]
        padded = np.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="reflect")
        accumulated = np.zeros_like(arr, dtype=np.float32)
        for y_offset in range(3):
            for x_offset in range(3):
                accumulated += padded[
                    :,
                    y_offset:y_offset + height,
                    x_offset:x_offset + width,
                ]
        return accumulated / 9.0


    def _apply_stage8_masked_pixel_enhancement(
        self,
        image_data: np.ndarray,
        plan: Dict[str, Any],
        *,
        label: str,
        plugin_candidate: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
        return stage8_pixels.apply_stage8_masked_pixel_enhancement(
            self,
            image_data,
            plan,
            label=label,
            plugin_candidate=plugin_candidate,
        )


    def _apply_stage8_builtin_enhancement(
        self,
        plan: Dict[str, Any],
        *,
        label: str,
    ) -> List[str]:
        return stage8_pixels.apply_stage8_builtin_enhancement(self, plan, label=label)


    def _apply_stage8_color_correction_from_quality(
        self,
        quality_record: Dict[str, Any],
    ) -> Optional[str]:
        return stage8_pixels.apply_stage8_color_correction_from_quality(self, quality_record)


    def _stage8_target_blue_excess(self, quality_record: Optional[Dict[str, Any]]) -> float:
        return stage8_pixels.stage8_target_blue_excess(self, quality_record)


    def _stage8_quality_assessment(
        self,
        *,
        baseline_stem: str = "stage8_input_starless",
        candidate_stem: str = "stage8_enhanced",
    ) -> Dict[str, Any]:
        return stage8_pixels.stage8_quality_assessment(
            self,
            baseline_stem=baseline_stem,
            candidate_stem=candidate_stem,
        )


    def _stage8_conservative_rerun(self, original_saturation: float) -> Dict[str, Any]:
        return stage8_pixels.stage8_conservative_rerun(self, original_saturation)


    def _stage8_needs_conservative_rerun(self, quality_record: Dict[str, Any]) -> bool:
        return stage8_pixels.stage8_needs_conservative_rerun(self, quality_record)


class StageSupportMixin:
    def _create_stage_review_bundle(
        self,
        stage_key: str,
        before_stem: str,
        after_stem: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        selected_candidate: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write non-blocking visual evidence for later human or multimodal review."""
        try:
            payload = review_bundle.create_stage_review_bundle(
                self,
                stage_key=stage_key,
                before_stem=before_stem,
                after_stem=after_stem,
                context={
                    "target_type": (
                        self._active_target_type()
                        if hasattr(self, "_active_target_type")
                        else "generic_low_snr_safe"
                    ),
                    "policy": (
                        self._active_policy_name()
                        if hasattr(self, "_active_policy_name")
                        else "generic_low_snr_safe"
                    ),
                    **(context or {}),
                },
                candidates=candidates,
                selected_candidate=selected_candidate,
            )
            if payload.get("status") == "ready":
                self.log.info(
                    f"[{stage_key}] review bundle ready: {payload.get('report_path')}"
                )
            return payload
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.log.warn(f"[{stage_key}] review bundle skipped: {error}")
            return {
                "stage": stage_key,
                "status": "unavailable",
                "reason": self._short_text(error, 180),
            }

    def _is_candidate_stacked(self, f):
        """判断文件是否为候选叠加文件（排除中间产物）"""
        name_lower = f.name.lower()
        stem_lower = f.stem.lower()
        for prefix in self.cfg.exclude_prefixes:
            if name_lower.startswith(prefix):
                return False
        for substring in self.cfg.exclude_substrings:
            if substring in stem_lower:
                return False
        for suffix in self.cfg.exclude_suffixes:
            if stem_lower.endswith(suffix):
                return False
        if f.parent != self.work_dir:
            return False
        return True


    def _find_fit_files(self):
        """一次遍历工作目录，返回所有 .fit/.fits 文件"""
        return [
            f for f in self.work_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ('.fit', '.fits')
        ]


    def _build_manual_starmask(self):
        """
        从原拉伸图与去星图构建星点层:
        stars = stretched - starless
        注: 某些 Siril 版本 PixelMath 不支持 max(a,b) 语法，
        此处仅生成差分星点层作为回退方案。
        """
        try:
            self.log.info("构建手动星点层: stretched - starless")
            self.cmd_with_check("load", self.stretched_name)
            self.cmd_with_check("pm", f"${self.stretched_name}$ - $starless$")
            self.cmd_with_check("save", "starmask_raw")
            self.log.info("已生成手动星点层: starmask_raw.fit")

            self.starmask_file = self.process_dir / "starmask_raw.fit"
            return True
        except (CommandError, SirilError) as e:
            self.log.warn(f"手动构建星点层失败: {e}")
            return False


    def _match_star_layer_shape(self, stars: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Return a star layer compatible with the base image layout."""
        if stars.shape == base.shape:
            return stars
        if base.ndim == 3 and stars.ndim == 2:
            return np.broadcast_to(stars, base.shape)
        if base.ndim == 2 and stars.ndim == 3:
            return np.max(stars, axis=0)
        if base.ndim == 3 and stars.ndim == 3:
            if stars.shape[0] == 1 and base.shape[0] == 3 and stars.shape[1:] == base.shape[1:]:
                return np.broadcast_to(stars, base.shape)
            if stars.shape[0] == 3 and base.shape[0] == 1 and stars.shape[1:] == base.shape[1:]:
                return np.max(stars, axis=0, keepdims=True)
        raise ValueError(
            f"starmask shape {stars.shape} is incompatible with base shape {base.shape}"
        )


    def _stage9_capture_remix_base_identity(self, source_stem: str) -> Dict[str, Any]:
        """Hash the immutable Stage 9 base used by every remix candidate."""
        process_dir = getattr(self, "process_dir", None)
        if not process_dir:
            return {
                "status": "unavailable",
                "source_stem": source_stem,
                "reason": "process directory unavailable",
            }
        source_path = next(
            (
                Path(process_dir) / f"{source_stem}{suffix}"
                for suffix in (".fit", ".fits", ".fts")
                if (Path(process_dir) / f"{source_stem}{suffix}").is_file()
            ),
            None,
        )
        if source_path is None:
            return {
                "status": "unavailable",
                "source_stem": source_stem,
                "reason": "remix base FITS unavailable",
            }
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "status": "locked",
            "source_stem": source_stem,
            "path": str(source_path),
            "bytes": int(source_path.stat().st_size),
            "sha256": digest.hexdigest(),
        }


    def _stage9_verify_remix_base_identity(self, source_stem: str) -> Dict[str, Any]:
        expected = getattr(self, "_stage9_remix_base_identity", None)
        if not isinstance(expected, dict) or expected.get("source_stem") != source_stem:
            expected = self._stage9_capture_remix_base_identity(source_stem)
            self._stage9_remix_base_identity = expected
        if expected.get("status") != "locked":
            return dict(expected)
        current = self._stage9_capture_remix_base_identity(source_stem)
        if current.get("status") != "locked":
            return current
        if (
            current.get("sha256") != expected.get("sha256")
            or current.get("bytes") != expected.get("bytes")
        ):
            return {
                "status": "rejected",
                "source_stem": source_stem,
                "reason": "immutable Stage9 remix base changed between candidates",
                "expected": expected,
                "actual": current,
            }
        return {**expected, "status": "verified"}


    def _apply_previous_stage_star_remix(self, source_stem: str, starmask_name: str, intensity: float) -> bool:
        """
        Compose stars onto the immutable Stage 9 remix base in pixel space.

        Siril's `pm $stage6$ ... $starless$ ...` expression can report success while
        leaving the currently loaded stage6 image unchanged. Reading both buffers and
        writing the composed image avoids silently saving stage6 as stage9.
        """
        try:
            base_identity = self._stage9_verify_remix_base_identity(source_stem)
            if base_identity.get("status") != "verified":
                raise RuntimeError(
                    "Stage9 immutable remix base contract failed: "
                    f"{base_identity.get('reason', base_identity.get('status'))}"
                )
            self.cmd_with_check("load", source_stem)
            base_data = self.siril.get_image_pixeldata(preview=False)
            if base_data is None:
                raise RuntimeError(f"{source_stem} image buffer is empty")
            base = np.asarray(base_data)

            self.cmd_with_check("load", starmask_name)
            star_data = self.siril.get_image_pixeldata(preview=False)
            if star_data is None:
                raise RuntimeError(f"{starmask_name} image buffer is empty")
            stars = self._match_star_layer_shape(np.asarray(star_data), base)

            weak_mask = None
            bright_mask = None
            alpha_mask = None
            candidate_masks = dict(
                getattr(self, "_stage9_candidate_overlay_masks", {}) or {}
            ).get(starmask_name)
            if isinstance(candidate_masks, dict):
                weak_mask = candidate_masks.get("weak")
                bright_mask = candidate_masks.get("bright")
                alpha_mask = candidate_masks.get("alpha")
            catalog = getattr(self, "_stage9_star_reference_catalog", None)
            if (
                alpha_mask is None
                and isinstance(catalog, dict)
                and catalog.get("status") == "ok"
            ):
                calibration = dict(
                    getattr(self, "_stage9_starmask_calibration", {}) or {}
                )
                strict_overlay = str(calibration.get("support_mode") or "") == (
                    "strict_recovery"
                )
                weak_mask, bright_mask, alpha_mask = (
                    stage9_quality.build_star_overlay_masks(
                        catalog,
                        strict=strict_overlay,
                        cfg=self.cfg,
                        extra_pixels=int(
                            calibration.get("psf_support_retry_pixels", 0) or 0
                        ),
                    )
                )
            configured_weak_intensity = max(
                0.10,
                min(
                    1.05,
                    float(
                        getattr(
                            self.cfg,
                            "stage9_weak_star_screen_intensity_min",
                            0.40,
                        )
                    ),
                ),
            )
            weak_intensity = max(float(intensity), configured_weak_intensity)
            mixed = stage9_quality.screen_blend(
                base,
                stars,
                intensity,
                alpha_mask=alpha_mask,
                weak_mask=weak_mask,
                bright_mask=bright_mask,
                weak_intensity=weak_intensity,
            )
            self._stage9_last_star_overlay_mask = alpha_mask
            self._stage9_last_weak_overlay_mask = weak_mask
            self._stage9_last_bright_overlay_mask = bright_mask
            self._stage9_last_star_layer = np.array(stars, copy=True)

            # After reading starmask, switch the active Siril image back to the
            # immutable Stage 9 remix base before replacing its pixels with the
            # composed result.
            self.cmd_with_check("load", source_stem)
            lock_factory = getattr(self.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    self.siril.set_image_pixeldata(mixed)
            else:
                self.log.warn(
                    "阶段9 Siril image_lock 不可用，尝试直接写回合成图像"
                )
                self.siril.set_image_pixeldata(mixed)
            self.log.info(
                "阶段9 使用显式上层 Alpha+Screen 向 Starless 底图回混星点 "
                f"(source={source_stem}, starmask={starmask_name}, "
                f"bright_intensity={intensity}, weak_intensity={weak_intensity})"
            )
            return True
        except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
            self.log.warn(f"Stage9 remix base 星点合成失败: {e}")
            return False

    def _stage9_assess_current_remix(
        self,
        source_stem: str,
        *,
        attempt: str,
        formula: str,
    ) -> Dict[str, Any]:
        """Measure the active Stage 9 candidate, restoring it after loading its base."""
        try:
            candidate_data = self.siril.get_image_pixeldata(preview=False)
            if candidate_data is None:
                raise RuntimeError("Stage9 candidate image buffer is empty")
            candidate = np.array(candidate_data, copy=True)

            self.cmd_with_check("load", source_stem)
            base_data = self.siril.get_image_pixeldata(preview=False)
            if base_data is None:
                raise RuntimeError(f"{source_stem} image buffer is empty")
            base = np.array(base_data, copy=True)
            report = stage9_quality.assess_remix(
                base,
                candidate,
                self.cfg,
                attempt=attempt,
                formula=formula,
                star_reference=getattr(
                    self,
                    "_stage9_star_reference_catalog",
                    None,
                ),
                star_overlay_mask=getattr(
                    self,
                    "_stage9_last_star_overlay_mask",
                    None,
                ),
            )

            self.cmd_with_check("load", source_stem)
            lock_factory = getattr(self.siril, "image_lock", None)
            if callable(lock_factory):
                with lock_factory():
                    self.siril.set_image_pixeldata(candidate)
            else:
                self.siril.set_image_pixeldata(candidate)
            return report
        except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
            self.log.warn(f"Stage9 remix quality assessment failed: {e}")
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "rejected",
                "accepted": False,
                "gate_enabled": bool(
                    getattr(self.cfg, "stage9_quality_gate_enabled", True)
                ),
                "issues": [self._short_text(e, 180)],
                "metrics": {},
            }

    # --------------------------------------------------
    # 连接
    # --------------------------------------------------


    def _load_stacked_file(self, stacked_files):
        """加载已叠加的文件"""
        self._stage1_input_mode = "stacked"
        stacked_files = sorted(
            stacked_files, key=lambda x: x.stat().st_mtime, reverse=True
        )
        if len(stacked_files) > 1:
            self.log.info(
                f"找到 {len(stacked_files)} 个叠加文件，使用最新的:")
            for f in stacked_files[:3]:
                self.log.info(f"    - {f.name}")

        self.source_file = stacked_files[0]
        self.log.info(f"源文件: {self.source_file.name}")

        working_file = self.process_dir / "working.fit"
        shutil.copy2(self.source_file, working_file)
        self.log.info("已复制到处理目录")

        self.cmd_with_check("cd", f'"{self.process_dir}"')
        self.cmd_with_check("load", "working")


    def _prepare_isolated_light_input(self, light_files):
        """创建隔离输入目录，确保预处理只使用本轮检测到的 Light 帧"""
        input_dir = self.process_dir / "_light_input"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)

        # 统一重命名为 lightsrc_00001.fit，避免污染和大小写差异
        sorted_lights = sorted(light_files, key=lambda p: p.name)
        for idx, src in enumerate(sorted_lights, start=1):
            ext = src.suffix.lower()
            if ext not in ('.fit', '.fits'):
                ext = '.fit'
            dest = input_dir / f"lightsrc_{idx:05d}{ext}"
            try:
                dest.symlink_to(src)
            except OSError:
                # 某些文件系统不支持软链接时回退到复制
                shutil.copy2(src, dest)

        self.log.info(f"已隔离 Light 输入: {len(sorted_lights)} 帧")
        return input_dir, "lightsrc"


    def _preprocess_light_frames(self, light_files):
        """
        对 Light_ 单帧执行预处理:
        link → calibrate (debayer) → register → stack
        """
        self._stage1_input_mode = "light_preprocess"
        # 在隔离目录执行 link，避免历史文件混入
        input_dir, seq_name = self._prepare_isolated_light_input(light_files)
        self.cmd_with_check("cd", f'"{input_dir}"')
        self.log.info("[预处理 1/4] 链接 Light 帧...")
        self.cmd_with_check("link", seq_name, "-out=..")

        self.cmd_with_check("cd", f'"{self.process_dir}"')

        self.log.info("[预处理 2/4] 去拜耳校准...")
        self.cmd_with_check("calibrate", seq_name, "-debayer")
        pp_seq = f"pp_{seq_name}"
        r_pp_seq = f"r_{pp_seq}"

        self.log.info("[预处理 3/4] 图像配准...")
        self.cmd_with_check("register", pp_seq, "-2pass")
        self.cmd_with_check("seqapplyreg", pp_seq, "-filter-round=2.5k")

        self.log.info("[预处理 4/4] 叠加...")
        self.cmd_with_check(
            "stack", r_pp_seq, "rej", "3", "3",
            "-norm=addscale", "-output_norm", "-rgb_equal",
            "-out=working")
        # Seestar preprocessing scripts flip the stacked output before loading it.
        self.cmd_with_check("mirrorx_single", "working")

        total_frames = len(light_files)
        registered_count = self._count_sequence_products(r_pp_seq)
        failed_count = max(total_frames - registered_count, 0)
        fail_ratio = (
            (failed_count / total_frames)
            if total_frames > 0 else 0.0
        )
        self.log.info(
            "预处理配准统计: "
            f"registered={registered_count}/{total_frames}, failed={failed_count}"
        )

        self.cmd_with_check("load", "working")
        self.source_file = self.process_dir / "working.fit"
        self.log.info("预处理完成: 叠加结果已加载")
        return {
            "total": total_frames,
            "registered": registered_count,
            "failed": failed_count,
            "fail_ratio": fail_ratio,
        }

    # ========================================
    # 阶段 2: 视图与画面修正
    # ========================================


    def _stage3_measure_features(self, tag: str) -> Optional[ImageFeatures]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            feat = measure_image_features(image_data)
            self.log.debug(
                f"[stage3:{tag}] bg_median={feat.bg_median:.4f}, "
                f"bg_std={feat.bg_std:.4f}, object_ratio={feat.object_area_ratio:.4f}, "
                f"edge_black={feat.edge_black_ratio:.4f}"
            )
            return feat
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.debug(f"[stage3:{tag}] feature sampling skipped: {e}")
            return None


    def _stage3_signal_preservation_metrics(
        self,
        before_image: Optional[np.ndarray],
        after_image: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        if before_image is None or after_image is None:
            return {"available": False, "notes": ["image sampling unavailable"]}
        safe_report = getattr(self, "_stage3_safe_sample_report", {}) or {}
        split_report = safe_report.get("fit_validation_split") or {}
        sky_points = split_report.get("validation_points") or []
        return measure_stage3_signal_preservation(
            before_image,
            after_image,
            sky_points=[
                (float(point[0]), float(point[1]))
                for point in sky_points
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ],
            sky_patch_radius=int(safe_report.get("patch_radius", 12) or 12),
        )


    def _stage3_subsky_rbf_candidates(self) -> List[Tuple[str, ...]]:
        samples = _clamp_int(self.cfg.bg_samples, 12, 32)
        tolerance = _clamp_float(self.cfg.bg_tolerance, 0.6, 1.8)
        smooth = _clamp_float(self.cfg.bg_smooth, 0.2, 1.2)

        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            feat = measure_image_features(image_data)
        except (CommandError, DataError, SirilError, RuntimeError, ValueError):
            feat = None

        bg_std = float(feat.bg_std) if feat is not None else 0.03
        star_density = float(feat.star_density) if feat is not None else 0.002
        object_area = float(feat.object_area_ratio) if feat is not None else 0.20
        high_noise = bg_std >= 0.045
        low_noise = bg_std <= 0.020
        complex_field = star_density >= 0.0045 or object_area >= 0.35
        variant_budget = 5 if (high_noise or complex_field) else 4 if low_noise else 3

        variants = [
            (samples, tolerance, smooth),
        ]
        if high_noise:
            variants.extend(
                [
                    (
                        _clamp_int(samples - 4, 12, 32),
                        _clamp_float(tolerance - 0.20, 0.6, 1.8),
                        _clamp_float(smooth * 2.0, 0.2, 1.2),
                    ),
                    (
                        _clamp_int(samples - 6, 12, 32),
                        _clamp_float(tolerance - 0.30, 0.6, 1.8),
                        _clamp_float(smooth * 1.6, 0.2, 1.2),
                    ),
                ]
            )
        else:
            variants.append(
                (
                    _clamp_int(samples - 4, 12, 32),
                    _clamp_float(tolerance + 0.20, 0.6, 1.8),
                    _clamp_float(smooth + 0.20, 0.2, 1.2),
                )
            )
        variants.append(
            (
                _clamp_int(samples + 4, 12, 32),
                _clamp_float(tolerance - 0.10, 0.6, 1.8),
                _clamp_float(smooth - 0.10, 0.2, 1.2),
            )
        )
        if complex_field:
            variants.append(
                (
                    _clamp_int(samples + 6, 12, 32),
                    _clamp_float(tolerance + 0.10, 0.6, 1.8),
                    _clamp_float(smooth + 0.35, 0.2, 1.2),
                )
            )
        if low_noise:
            variants.append(
                (
                    _clamp_int(samples + 8, 12, 32),
                    _clamp_float(tolerance + 0.25, 0.6, 1.8),
                    _clamp_float(smooth - 0.15, 0.2, 1.2),
                )
            )

        seen = set()
        commands: List[Tuple[str, ...]] = []
        for s_count, tol, sm in variants[:variant_budget]:
            key = (int(s_count), round(float(tol), 3), round(float(sm), 3))
            if key in seen:
                continue
            seen.add(key)
            commands.append(
                (
                    "subsky",
                    "-rbf",
                    f"-samples={s_count}",
                    f"-tolerance={tol:.3f}",
                    f"-smooth={sm:.3f}",
                    "-existing",
                )
            )
        self.log.info(
            "[Stage3] Dynamic RBF candidates with custom -existing samples: "
            f"count={len(commands)}, bg_std={bg_std:.4f}, "
            f"star_density={star_density:.5f}, object_area={object_area:.3f}"
        )
        return commands


    def _compute_ccm_fallback_gains(self) -> Tuple[float, float, int]:
        image_data = self.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("image buffer is empty")

        rgb = _to_rgb_float_image(np.asarray(image_data))
        r, g, b = rgb[0], rgb[1], rgb[2]
        gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)

        signal_threshold = max(
            float(np.quantile(gray, 0.55)),
            float(np.median(gray) + 0.5 * np.std(gray)),
        )
        signal_mask = gray > signal_threshold
        signal_count = int(np.count_nonzero(signal_mask))
        if signal_count < 128:
            signal_mask = gray > float(np.quantile(gray, 0.50))
            signal_count = int(np.count_nonzero(signal_mask))
        if signal_count < 32:
            signal_mask = np.ones_like(gray, dtype=bool)
            signal_count = int(np.count_nonzero(signal_mask))

        eps = 1e-6
        r_median = float(np.median(r[signal_mask])) + eps
        g_median = float(np.median(g[signal_mask])) + eps
        b_median = float(np.median(b[signal_mask])) + eps

        r_gain = _clamp_float(g_median / r_median, 0.65, 1.55)
        b_gain = _clamp_float(g_median / b_median, 0.65, 1.55)
        return r_gain, b_gain, signal_count


    def _run_ccm_color_fallback(self) -> Tuple[bool, str]:
        try:
            r_gain, b_gain, sample_pixels = self._compute_ccm_fallback_gains()
            self.cmd_with_check(
                "ccm",
                f"{r_gain:.6f}",
                "0",
                "0",
                "0",
                "1.000000",
                "0",
                "0",
                "0",
                f"{b_gain:.6f}",
                "1.0",
            )
            msg = (
                "使用 CCM 回退完成色彩校准 "
                f"(r_gain={r_gain:.3f}, b_gain={b_gain:.3f}, sample_pixels={sample_pixels})"
            )
            self.log.info(msg)
            return True, msg
        except (CommandError, SirilError, DataError, RuntimeError, ValueError) as e:
            return False, self._short_text(e, 200)


    def _stage9_bad_starless_reason(self) -> str:
        reasons: List[str] = []
        advisories: List[str] = []
        accepted_stretch = bool(
            getattr(self, "_stage7_stretch_accepted", False)
        )
        accepted_stretch_stem = str(
            getattr(self, "_stage7_stretch_output", None)
            or getattr(self, "stretched_name", None)
            or ""
        ).strip()
        process_dir = getattr(self, "process_dir", None)
        if accepted_stretch and accepted_stretch_stem and process_dir is not None:
            accepted_stretch = (
                process_dir / f"{accepted_stretch_stem}.fit"
            ).is_file()
        else:
            accepted_stretch = False
        if not accepted_stretch:
            reasons.append("stage7_stretch_not_accepted")
        if bool(getattr(self, "_stage7_starless_skipped", False)):
            reasons.append("stage7_starless_skipped")
        quality = getattr(self, "_stage7_selected_quality", None)
        if isinstance(quality, dict):
            status = str(quality.get("status", "") or "").lower()
            quality_issues = [
                str(item).strip().lower()
                for item in (quality.get("issues") or [])
                if str(item).strip()
            ]
            advisories.extend(
                str(item).strip()
                for item in (quality.get("advisories") or [])
                if str(item).strip()
            )
            dynamic_range_only = bool(quality_issues) and all(
                item.startswith("starless_dynamic_range_collapse")
                for item in quality_issues
            )
            if status and status != "ok":
                status_reason = f"stage7_quality_status={status}"
                if accepted_stretch and dynamic_range_only:
                    advisories.append(status_reason)
                else:
                    reasons.append(status_reason)
            derived = quality.get("derived") if isinstance(quality.get("derived"), dict) else {}
            residual = float(derived.get("residual_star_score", 0.0) or 0.0)
            halo = float(derived.get("halo_residue_score", 0.0) or 0.0)
            if stage7_quality.stage7_single_local_galaxy_halo_override_active(
                self,
                quality,
            ):
                # Keep the local galaxy-disk measurement in the report, but do
                # not let one uncorroborated sample revoke an otherwise safe
                # starless pair at the final remix boundary.
                halo = max(
                    float(
                        derived.get("global_halo_residue_score", 0.0) or 0.0
                    ),
                    float(
                        derived.get("compact_halo_residue_score", 0.0) or 0.0
                    ),
                )
            noise_gain = float(derived.get("starless_noise_gain", 0.0) or 0.0)
            dynamic_range_ratio = float(
                derived.get("starless_dynamic_range_ratio", 1.0) or 0.0
            )
            peak_signal = float(derived.get("starless_peak_signal", 1.0) or 0.0)
            residual_gate = stage7_quality.stage7_upper_quality_gate(
                self.cfg,
                value=residual,
                accepted_limit=self.cfg.stage7_residual_star_score_max,
            )
            if residual_gate["hard_failed"]:
                reasons.append(
                    f"stage7_residual_star_score {residual:.3f}>{self.cfg.stage7_residual_star_score_max:.3f}"
                )
            elif residual_gate["advisory"]:
                advisories.append(
                    f"stage7_residual_star_score {residual:.3f}>"
                    f"{self.cfg.stage7_residual_star_score_max:.3f} advisory"
                )
            halo_threshold = self._stage7_effective_halo_threshold()
            halo_gate = stage7_quality.stage7_upper_quality_gate(
                self.cfg,
                value=halo,
                accepted_limit=halo_threshold,
            )
            if halo_gate["hard_failed"]:
                reasons.append(
                    f"stage7_halo_residue_score {halo:.3f}>{halo_threshold:.3f}"
                )
            elif halo_gate["advisory"]:
                advisories.append(
                    f"stage7_halo_residue_score {halo:.3f}>"
                    f"{halo_threshold:.3f} advisory"
                )
            noise_gate = stage7_quality.stage7_upper_quality_gate(
                self.cfg,
                value=noise_gain,
                accepted_limit=self.cfg.stage7_starless_noise_gain_max,
            )
            if noise_gate["hard_failed"]:
                reasons.append(
                    f"stage7_starless_noise_gain {noise_gain:.3f}>{self.cfg.stage7_starless_noise_gain_max:.3f}"
                )
            elif noise_gate["advisory"]:
                advisories.append(
                    f"stage7_starless_noise_gain {noise_gain:.3f}>"
                    f"{self.cfg.stage7_starless_noise_gain_max:.3f} advisory"
                )
            dynamic_threshold = float(
                getattr(self.cfg, "stage7_starless_dynamic_range_min_ratio", 0.55)
            )
            peak_threshold = float(
                getattr(self.cfg, "stage7_starless_peak_signal_min", 0.006)
            )
            collapse = derived.get("dynamic_range_collapse")
            if collapse is None:
                collapse = (
                    dynamic_range_ratio < dynamic_threshold
                    and peak_signal < peak_threshold
                )
            if bool(collapse):
                dynamic_reason = (
                    "stage7_starless_dynamic_range "
                    f"{dynamic_range_ratio:.3f}<{dynamic_threshold:.3f}, "
                    f"peak={peak_signal:.5f}<{peak_threshold:.5f}"
                )
                if accepted_stretch:
                    advisories.append(dynamic_reason)
                else:
                    reasons.append(dynamic_reason)
        self._stage9_starless_advisories = list(dict.fromkeys(advisories))
        return ", ".join(dict.fromkeys(reasons))


    def _stage9_review_safe_source(self) -> str:
        candidates: List[Optional[str]] = [
            getattr(self, "_stage7_review_source", None)
        ]
        if bool(getattr(self, "_stage7_stretch_accepted", False)):
            candidates.extend(
                [getattr(self, "_stage7_stretch_output", None), "stage7_stretched"]
            )
        candidates.extend(
            [
                "stage7_cand_rescue_3",
                "stage7_cand_rescue_2",
                "stage7_cand_rescue_1",
                "stage7_cand_composite_tone",
                "stage7_cand_display90",
                "stage7_cand_display86",
                "stage7_cand_display82",
                "stage7_cand_display70",
                "stage7_cand_b",
                "stage7_cand_a",
                self.stretched_name,
            ]
        )
        seen = set()
        for stem in candidates:
            if not stem or stem in seen:
                continue
            seen.add(stem)
            if self.process_dir and (self.process_dir / f"{stem}.fit").exists():
                return stem
        return self.stretched_name or "stage7_stretched"


    def _cleanup_lightsrc_intermediates(self):
        """清理 Stage1 Light 预处理生成的 lightsrc 相关中间文件。"""
        if not self.process_dir or not self.process_dir.exists():
            return 0

        targets = set(self.process_dir.glob("*lightsrc*"))
        light_input_dir = self.process_dir / "_light_input"
        if light_input_dir.exists():
            targets.add(light_input_dir)

        deleted_count = 0
        for path in sorted(targets, key=lambda item: item.name):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                deleted_count += 1
            except OSError as e:
                self.log.warn(f"清理 lightsrc 中间文件失败: {path.name} ({e})")
        return deleted_count


    def _summary_message_lines(
        self,
        message: str,
        *,
        max_parts: int = 10,
        max_len: int = 180,
    ) -> List[str]:
        """Split long stage summary messages into stable, readable log lines."""
        text = str(message or "").replace("\r", " ").replace("\n", "；")
        parts = [part.strip() for part in text.split("；") if part.strip()]
        if not parts and text.strip():
            parts = [text.strip()]

        lines: List[str] = []
        hidden = 0
        for part in parts:
            if len(lines) >= max_parts:
                hidden += 1
                continue
            if len(part) <= max_len:
                lines.append(part)
                continue
            chunks = textwrap.wrap(
                part,
                width=max_len,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [part[:max_len]]
            for chunk in chunks:
                if len(lines) >= max_parts:
                    hidden += 1
                    break
                lines.append(chunk)
        if hidden:
            lines.append(f"... {hidden} more summary item(s); see detailed stage logs and JSON reports")
        return lines
