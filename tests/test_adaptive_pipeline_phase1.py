#!/usr/bin/env python3
"""Phase 1 tests for adaptive target policy and stretch scoring."""

from __future__ import annotations

import sys
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
from target_profiler import build_target_profile  # noqa: E402


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
