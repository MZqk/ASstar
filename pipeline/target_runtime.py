"""Service mixins for SeestarPostProcessor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ai_advisory
import cosmic_clarity
import plugin_runner
import sasp_runner
import scunet_denoise
import syqon_starless
import stage7_quality
import stage7_repair
import stage8_pixels
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    format_feature_summary,
    measure_quality_metrics,
)
from models import ImageFeatures, QualityMetrics, Stage6StretchStrategy, StageResult, TargetType
from save_utils import save_stage_output, write_ai_raw_response, write_stage_json

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except ImportError:
    CommandError = RuntimeError
    DataError = RuntimeError
    SirilError = RuntimeError

try:
    from image_feature_analyzer import analyze_image as analyze_adaptive_image
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from stretch_candidate_evaluator import (
        build_candidate_spec,
        candidate_modes,
        choose_best as choose_best_stretch_candidate,
        score_candidate as score_stretch_candidate,
    )
    from target_profiler import build_target_profile, normalize_target_profile
except (ImportError, RuntimeError):
    analyze_adaptive_image = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage6_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_candidate_spec = None
    candidate_modes = None
    choose_best_stretch_candidate = None
    score_stretch_candidate = None
    build_target_profile = None
    normalize_target_profile = None

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "SEESTAR_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "SEESTAR_INPUT_MODE"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "result_linear_resume"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)

class TargetRuntimeMixin:
    def _auto_target_hint(self) -> str:
        result = getattr(self, "auto_tune_result", None)
        target_type = getattr(result, "target_type", None)
        name = getattr(target_type, "name", "")
        value = getattr(target_type, "value", "")
        return str(name or value or "")


    def _target_profile_context_text(self) -> str:
        values = [
            self.source_file,
            self.work_dir,
            self.process_dir,
            getattr(self, "input_dir", None),
        ]
        return " ".join(str(value) for value in values if value is not None)


    def _refresh_target_profile_from_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        stage_label: str,
    ) -> Optional[str]:
        if not metadata or build_target_profile is None or analyze_adaptive_image is None:
            return None
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                return None
            profile = build_target_profile(
                analyze_adaptive_image(image_data),
                metadata=metadata,
                context_text=self._target_profile_context_text(),
            )
            policy = profile.pop("policy", None)
            previous_type = (
                self.target_profile.get("target_type")
                if isinstance(getattr(self, "target_profile", None), dict)
                else None
            )
            previous_policy = self._active_policy_name() if hasattr(self, "_active_policy_name") else ""
            policy = self._sync_runtime_policy_from_profile(
                profile,
                source=stage_label,
                policy_candidate=policy if isinstance(policy, dict) else None,
            )
            self._write_stage_json("target_profile.json", profile)
            self._write_stage_json("pipeline_policy.json", policy)
            new_policy = self._active_policy_name()
            if (
                (previous_type and previous_type != profile.get("target_type"))
                or previous_policy != new_policy
            ):
                self.log.warn(
                    f"[{stage_label}] target profile updated: "
                    f"{profile.get('target_name_guess') or ''} {profile.get('target_type')} policy={new_policy}"
                )
            return (
                f"{stage_label} target_profile={profile.get('target_name_guess') or profile.get('target_type')} "
                f"policy={new_policy} previous_policy={previous_policy}"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.warn(f"[{stage_label}] target profile metadata refresh failed: {e}")
            return None


    def _adaptive_features_from_image(self, image_data: Any) -> Dict[str, Any]:
        if analyze_adaptive_image is None:
            return {}
        try:
            features = analyze_adaptive_image(image_data)
            return features.to_dict()
        except (RuntimeError, TypeError, ValueError, IndexError, FloatingPointError) as e:
            self.log.warn(f"自适应图像特征分析失败: {e}")
            return {}


    def _adaptive_features_current(self) -> Dict[str, Any]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                return {}
            return self._adaptive_features_from_image(image_data)
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            self.log.warn(f"当前图像自适应特征读取失败: {e}")
            return {}


    def _adaptive_features_by_stem(self, stem: str) -> Dict[str, Any]:
        data = self._read_image_by_stem(stem)
        if data is None:
            return {}
        return self._adaptive_features_from_image(data)


    def _pixel_distribution_stats(self, image_data: Any) -> Dict[str, Any]:
        try:
            arr = np.asarray(image_data, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.size == 0:
                raise ValueError("empty image")
            flat = arr.reshape(-1)
            p01, p50, p90, p99 = np.percentile(flat, [1.0, 50.0, 90.0, 99.0])
            min_v = float(np.min(flat))
            max_v = float(np.max(flat))
            median = float(np.median(flat))
            dynamic = max_v - min_v
            global_dark_ratio = float(np.mean(flat <= 0.010))
            object_signal_ratio = float((float(p99) - float(p50)) / max(float(p50), 0.003))
            safe_preview_visibility = float((float(p99) - float(p50)) / max(float(p99), 0.010))
            core_peak_ratio = float(max_v / max(float(p50), 0.003))
            visibility_too_low = (
                global_dark_ratio > 0.985
                and float(p99) < 0.020
                and safe_preview_visibility < 0.35
            )
            nearly_black = (
                visibility_too_low
                or
                (
                    float(p99) <= 0.006
                    and float(p90) <= 0.003
                    and safe_preview_visibility < 0.12
                    and core_peak_ratio < 3.0
                )
                or (
                    float(p99) <= 0.012
                    and dynamic <= 0.004
                    and object_signal_ratio < 1.0
                    and safe_preview_visibility < 0.15
                    and core_peak_ratio < 2.5
                )
            )
            nearly_white = (float(p01) >= 0.92) or (median >= 0.98)
            invalid_dynamic = (not math.isfinite(dynamic)) or dynamic <= 1e-5 or float(p99) <= float(p01) + 1e-5
            return {
                "min": min_v,
                "max": max_v,
                "median": median,
                "p01": float(p01),
                "p50": float(p50),
                "p90": float(p90),
                "p99": float(p99),
                "dynamic_range": float(dynamic),
                "global_dark_ratio": global_dark_ratio,
                "object_signal_ratio": object_signal_ratio,
                "safe_preview_visibility_score": safe_preview_visibility,
                "core_peak_ratio": core_peak_ratio,
                "is_visibility_too_low": bool(visibility_too_low),
                "is_nearly_black": bool(nearly_black),
                "is_nearly_white": bool(nearly_white),
                "invalid_dynamic_range": bool(invalid_dynamic),
            }
        except (RuntimeError, TypeError, ValueError, IndexError, FloatingPointError) as e:
            return {
                "error": self._short_text(e, 160),
                "is_nearly_black": False,
                "is_nearly_white": False,
                "invalid_dynamic_range": True,
            }


    def _current_pixel_distribution_stats(self) -> Dict[str, Any]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                raise RuntimeError("image buffer is empty")
            return self._pixel_distribution_stats(image_data)
        except (CommandError, DataError, SirilError, OSError, RuntimeError, TypeError, ValueError) as e:
            return {
                "error": self._short_text(e, 160),
                "is_nearly_black": False,
                "is_nearly_white": False,
                "invalid_dynamic_range": True,
            }


    def _active_policy_name(self) -> str:
        policy = getattr(self, "pipeline_policy", None)
        if isinstance(policy, dict):
            return str(policy.get("policy_name") or "generic_low_snr_safe")
        return "generic_low_snr_safe"


    def _active_target_type(self) -> str:
        frozen = getattr(self, "_frozen_primary_target", None)
        if isinstance(frozen, dict) and frozen.get("type"):
            return str(frozen["type"])
        profile = getattr(self, "target_profile", None)
        if isinstance(profile, dict):
            primary = profile.get("primary_target")
            if isinstance(primary, dict) and primary.get("type"):
                return str(primary["type"])
            return str(profile.get("target_type") or "generic_low_snr_safe")
        return "generic_low_snr_safe"

    def _freeze_primary_target(self) -> Dict[str, Any]:
        """Freeze the sole routing target before processing transforms start."""
        if (
            bool(getattr(self, "_target_primary_frozen", False))
            and isinstance(getattr(self, "_frozen_primary_target", None), dict)
        ):
            return dict(self._frozen_primary_target)
        profile = dict(getattr(self, "target_profile", {}) or {})
        if callable(normalize_target_profile):
            profile = normalize_target_profile(profile)
        primary = dict(profile.get("primary_target") or {})
        primary.update(
            {
                "type": str(
                    primary.get("type")
                    or profile.get("target_type")
                    or "generic_low_snr_safe"
                ),
                "frozen": True,
                "frozen_at": "processing_plan",
            }
        )
        profile["primary_target"] = primary
        profile["target_type"] = primary["type"]
        profile.setdefault("routing_contract", {})["primary_frozen"] = True
        self.target_profile = profile
        self._frozen_primary_target = copy.deepcopy(primary)
        self._target_primary_frozen = True
        self.log.info(
            "[TargetProfile] primary frozen "
            f"type={primary['type']} confidence={float(primary.get('confidence', 0.0)):.2f}"
        )
        return dict(primary)


    def _sync_runtime_policy_from_profile(
        self,
        profile: Dict[str, Any],
        *,
        source: str,
        policy_candidate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if callable(normalize_target_profile):
            normalized = normalize_target_profile(profile)
            profile.clear()
            profile.update(normalized)

        if bool(getattr(self, "_target_primary_frozen", False)):
            frozen = dict(getattr(self, "_frozen_primary_target", {}) or {})
            observed_primary = dict(profile.get("primary_target") or {})
            current = dict(getattr(self, "target_profile", {}) or {})
            current_secondary = current.get("secondary_labels", [])
            observed_secondary = profile.get("secondary_labels", [])
            merged_secondary = [
                label
                for label in (
                    "bright_core",
                    "large_nebulosity",
                    "faint_outer_cloud",
                    "dense_star_field",
                    "reflection_blue",
                    "emission_red",
                )
                if label in set(current_secondary or ())
                or label in set(observed_secondary or ())
            ]
            profile["secondary_labels"] = merged_secondary
            merged_evidence = dict(current.get("secondary_label_evidence") or {})
            merged_evidence.update(profile.get("secondary_label_evidence") or {})
            profile["secondary_label_evidence"] = {
                label: merged_evidence.get(label, {})
                for label in merged_secondary
            }
            profile["target_name_guess"] = frozen.get("name")
            profile["target_confidence"] = frozen.get("confidence", 0.0)
            profile["target_type"] = frozen.get(
                "type", "generic_low_snr_safe"
            )
            profile["classification_method"] = frozen.get("method", "unknown")
            profile["primary_target"] = copy.deepcopy(frozen)
            profile["pipeline"] = str(
                current.get("pipeline")
                or getattr(self, "pipeline_policy", {}).get("policy_name")
                or "generic_low_snr_safe"
            )
            profile.setdefault("routing_contract", {})["primary_frozen"] = True
            if observed_primary.get("type") != frozen.get("type"):
                profile["observed_primary_target"] = observed_primary
                profile.setdefault("diagnostics", []).append(
                    "primary_target_frozen: "
                    f"observed={observed_primary.get('type')} "
                    f"retained={frozen.get('type')}"
                )
                policy_candidate = None

        desired_name = str(
            profile.get("pipeline")
            or profile.get("default_policy")
            or "generic_low_snr_safe"
        )
        policy = policy_candidate if isinstance(policy_candidate, dict) else None
        if (
            not isinstance(policy, dict)
            or str(policy.get("policy_name") or "") != desired_name
        ):
            policy = (
                policy_for_profile(profile)
                if callable(policy_for_profile)
                else copy.deepcopy(DEFAULT_POLICY)
            )
        if str(policy.get("policy_name") or "") != desired_name:
            loaded_name = policy.get("policy_name") if isinstance(policy, dict) else "none"
            policy = copy.deepcopy(policy if isinstance(policy, dict) else DEFAULT_POLICY)
            policy["policy_name"] = desired_name
            policy.setdefault("applies_to", {})["target_types"] = [
                str(profile.get("target_type") or "generic_low_snr_safe")
            ]
            self.log.warn(
                f"[Policy] loader returned {loaded_name}; forcing runtime policy={desired_name}"
            )
        self.target_profile = profile
        self.pipeline_policy = policy
        self.log.info(f"[Policy] runtime policy refreshed: {self._active_policy_name()} ({source})")
        return policy


    def _fallback_target_profile(self, reason: str) -> Dict[str, Any]:
        profile = {
            "target_name_guess": None,
            "target_confidence": 0.0,
            "target_type": "generic_low_snr_safe",
            "pipeline": "generic_low_snr_safe",
            "classification_method": "fallback",
            "secondary_labels": [],
            "secondary_label_evidence": {},
            "features": {},
            "risks": {},
            "image_stats": {},
            "object_stats": {},
            "star_stats": {},
            "color_stats": {},
            "warnings": [reason],
        }
        if callable(normalize_target_profile):
            profile = normalize_target_profile(profile)
        policy = (
            policy_for_profile(profile)
            if callable(policy_for_profile)
            else copy.deepcopy(DEFAULT_POLICY)
        )
        profile["policy"] = policy
        return profile
