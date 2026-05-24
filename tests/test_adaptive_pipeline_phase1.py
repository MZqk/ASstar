#!/usr/bin/env python3
"""Phase 1 tests for adaptive target policy and stretch scoring."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "numpy" not in sys.modules:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_numpy.float32 = float
    fake_numpy.isfinite = lambda value: True
    sys.modules["numpy"] = fake_numpy

from image_feature_analyzer import AdaptiveImageFeatures  # noqa: E402
from policy_selector import load_policy  # noqa: E402
from quality_gate import evaluate_pre_starless_gate  # noqa: E402
from stretch_candidate_evaluator import (  # noqa: E402
    allowed_as_final,
    build_candidate_spec,
    candidate_modes,
    choose_best,
    score_candidate,
)
from target_profiler import BUILTIN_CATALOG, build_target_profile  # noqa: E402


class AdaptivePipelinePhase1Tests(unittest.TestCase):
    def test_target_profiler_classifies_m42_like_features(self) -> None:
        features = AdaptiveImageFeatures(
            bg_median=0.02,
            bg_std=0.018,
            dirty_background_score=0.42,
            object_area_ratio=0.35,
            bright_core_score=0.78,
            core_peak_ratio=0.86,
            nebulosity_area_ratio=0.34,
            faint_structure_score=0.55,
            dense_star_field_score=0.48,
            red_dominance=1.22,
            blue_dominance=1.18,
            halo_risk_score=0.52,
        )

        profile = build_target_profile(features, context_text="M42 stacked.fit")

        self.assertEqual(profile["target_type"], "bright_emission_reflection_nebula")
        self.assertEqual(profile["pipeline"], "bright_nebula_hdr_conservative")
        self.assertGreaterEqual(profile["target_confidence"], 0.55)

    def test_policy_missing_falls_back_to_generic(self) -> None:
        policy = load_policy("does_not_exist")

        self.assertEqual(policy["policy_name"], "generic_low_snr_safe")

    def test_named_policy_uses_builtin_overlay_when_config_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = load_policy("large_galaxy_core_protect", policy_dir=Path(tmpdir))

        self.assertEqual(policy["policy_name"], "large_galaxy_core_protect")
        self.assertIn("low_contrast_masked_lift", policy["stage6_stretch"]["candidate_mode"])
        self.assertEqual(policy["stage6_stretch"]["fallback_candidate"], "low_contrast_masked_lift")

    def test_target_profiler_matches_m42_from_plate_solve_coordinates(self) -> None:
        features = AdaptiveImageFeatures(
            dirty_background_score=0.52,
            object_area_ratio=0.04,
            bright_core_score=0.42,
            nebulosity_area_ratio=0.12,
            elongation_score=0.36,
        )

        profile = build_target_profile(
            features,
            metadata={
                "OBJCTRA": "05:35:39.735",
                "OBJCTDEC": "-05:27:33.020",
            },
        )

        self.assertEqual(profile["target_name_guess"], "M42")
        self.assertEqual(profile["target_type"], "bright_emission_reflection_nebula")
        self.assertEqual(profile["pipeline"], "bright_nebula_hdr_conservative")
        self.assertIn("catalog_coordinate", profile["classification_method"])

    def test_target_profiler_prefers_ic434_catalog_context_over_visual_galaxy_guess(self) -> None:
        features = AdaptiveImageFeatures(
            bg_median=0.0023,
            bg_std=0.0001,
            object_area_ratio=0.0003,
            nebulosity_area_ratio=0.0004,
            faint_structure_score=0.36,
            elongation_score=0.33,
            bright_core_score=0.31,
            blue_dominance=1.16,
            color_balance_score=0.82,
        )

        profile = build_target_profile(
            features,
            metadata={"CRVAL1": 85.9125, "CRVAL2": -2.5575},
            context_text="/Users/mz/SeeStar/IC 434_sub/process",
        )

        self.assertEqual(profile["target_name_guess"], "Horsehead Nebula")
        self.assertEqual(profile["target_type"], "dark_nebula_low_contrast")
        self.assertEqual(profile["pipeline"], "dark_nebula_low_contrast")

    def test_target_profiler_builtin_catalog_keeps_ic434_dark_nebula_with_auto_hint(self) -> None:
        features = AdaptiveImageFeatures(
            bg_median=0.0023,
            bg_std=0.0001,
            object_area_ratio=0.0003,
            nebulosity_area_ratio=0.0004,
            faint_structure_score=0.36,
            elongation_score=0.33,
            bright_core_score=0.31,
            blue_dominance=1.16,
            color_balance_score=0.82,
        )

        profile = build_target_profile(
            features,
            metadata={"AUTO_TARGET_TYPE": "EMISSION_NEBULA"},
            context_text="/Users/mz/SeeStar/IC 434_sub/process",
            catalog=BUILTIN_CATALOG,
        )

        self.assertEqual(profile["target_name_guess"], "Horsehead Nebula")
        self.assertEqual(profile["target_type"], "dark_nebula_low_contrast")
        self.assertEqual(profile["pipeline"], "dark_nebula_low_contrast")

    def test_target_profiler_preserves_auto_emission_hint_for_low_snr_rosette(self) -> None:
        features = AdaptiveImageFeatures(
            bg_median=0.0020,
            bg_std=0.0001,
            dirty_background_score=0.44,
            object_area_ratio=0.0002,
            nebulosity_area_ratio=0.0002,
            faint_structure_score=0.34,
            bright_core_score=0.20,
            red_dominance=1.03,
            blue_dominance=1.02,
            color_balance_score=0.82,
        )

        profile = build_target_profile(
            features,
            metadata={"AUTO_TARGET_TYPE": "bright_emission_reflection_nebula"},
            context_text="/Users/mz/SeeStar/NGC2237_sub0111/process/stage4_colorbalanced.fit",
        )

        self.assertEqual(profile["target_type"], "bright_emission_reflection_nebula")
        self.assertEqual(profile["pipeline"], "bright_nebula_hdr_conservative")
        self.assertIn("auto_target_hint", ",".join(profile["warnings"]))

    def test_dirty_background_forbids_autostretch_final(self) -> None:
        policy = load_policy("bright_nebula_hdr_conservative")
        allowed, reason = allowed_as_final(
            "autostretch_reference",
            {"dirty_background_score": 0.48, "core_clip_ratio": 0.002},
            policy,
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "dirty_background_policy_forbids_autostretch")

    def test_choose_best_ignores_rejected_autostretch(self) -> None:
        candidates = [
            {"name": "asinh_core_protect", "status": "ok", "score": 0.72, "allowed_as_final": True},
            {"name": "autostretch_reference", "status": "ok", "score": 0.95, "allowed_as_final": False},
        ]

        selected = choose_best(candidates, "asinh_core_protect")

        self.assertIsNotNone(selected)
        self.assertEqual(selected["name"], "asinh_core_protect")

    def test_dark_nebula_policy_uses_masked_lift_before_reference_autostretch(self) -> None:
        policy = load_policy("dark_nebula_low_contrast")

        modes = candidate_modes(policy, "dark_nebula_low_contrast")
        spec = build_candidate_spec(modes[0], SimpleNamespace(asinh_stretch=3.0, asinh_offset=0.001))

        self.assertEqual(modes[0], "dark_nebula_masked_lift")
        self.assertEqual(spec["method"], "bright_nebula_hdr_masked")
        self.assertGreater(spec["params"]["bg_pedestal"], 0.0)
        self.assertLess(spec["params"]["faint_saturation_boost"], 0.02)

    def test_large_galaxy_policy_has_low_contrast_rescue_candidate(self) -> None:
        policy = load_policy("large_galaxy_core_protect")

        modes = candidate_modes(policy, "small_galaxy")
        spec = build_candidate_spec("low_contrast_masked_lift", SimpleNamespace(asinh_stretch=3.0, asinh_offset=0.001))

        self.assertIn("low_contrast_masked_lift", modes)
        self.assertEqual(spec["method"], "bright_nebula_hdr_masked")
        self.assertGreater(spec["params"]["bg_pedestal"], 0.0)

    def test_choose_best_does_not_select_invalid_fallback_candidate(self) -> None:
        candidates = [
            {
                "name": "asinh_core_protect",
                "status": "ok",
                "score": 0.72,
                "allowed_as_final": False,
                "reject_reason": "nearly_black",
            },
            {
                "name": "autostretch_reference",
                "status": "ok",
                "score": 0.95,
                "allowed_as_final": False,
                "reject_reason": "autostretch_reference_only",
            },
        ]

        self.assertIsNone(choose_best(candidates, "asinh_core_protect"))

    def test_stage6_5_recommends_conservative_input_for_dirty_bright_nebula(self) -> None:
        policy = load_policy("bright_nebula_hdr_conservative")
        report = evaluate_pre_starless_gate(
            {
                "dirty_background_score": 0.48,
                "core_clip_ratio": 0.02,
                "halo_risk_score": 0.70,
            },
            {"target_type": "bright_emission_reflection_nebula"},
            policy,
        )

        self.assertFalse(report["ready_for_starless"])
        self.assertEqual(
            report["recommended_starless_input"],
            "stage7_ultra_conservative_asinh",
        )


if __name__ == "__main__":
    unittest.main()
