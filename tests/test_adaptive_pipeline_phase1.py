#!/usr/bin/env python3
"""Phase 1 tests for adaptive target policy and stretch scoring."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    import numpy  # noqa: F401
except ImportError:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_numpy.float32 = float
    fake_numpy.isfinite = lambda value: True
    sys.modules["numpy"] = fake_numpy

from image_feature_analyzer import AdaptiveImageFeatures  # noqa: E402
from policy_selector import load_policy  # noqa: E402
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
        self.assertEqual(
            profile["primary_target"]["type"],
            "bright_emission_reflection_nebula",
        )
        self.assertIn("large_nebulosity", profile["secondary_labels"])

    def test_cluster_primary_is_not_promoted_by_nebula_secondary_label(self) -> None:
        features = AdaptiveImageFeatures(
            bright_core_score=0.44,
            compactness_score=0.32,
            nebulosity_area_ratio=0.19,
            faint_structure_score=0.48,
            dense_star_field_score=0.82,
            red_dominance=1.18,
        )

        profile = build_target_profile(features)

        self.assertEqual(profile["target_type"], "globular_cluster")
        self.assertEqual(profile["pipeline"], "globular_cluster_star_preserve")
        self.assertIn("large_nebulosity", profile["secondary_labels"])
        self.assertIn("emission_red", profile["secondary_labels"])
        self.assertFalse(profile["routing_contract"]["secondary_labels_can_route"])

    def test_policy_missing_falls_back_to_generic(self) -> None:
        policy = load_policy("does_not_exist")

        self.assertEqual(policy["policy_name"], "generic_low_snr_safe")
        self.assertEqual(
            policy["stage4_color"]["calibration_policy"],
            "spcc_first_then_pcc",
        )

    def test_stage7_policy_does_not_expose_dead_scoring_weights(self) -> None:
        for policy_name in (
            "generic_low_snr_safe",
            "bright_nebula_hdr_conservative",
            "dark_nebula_low_contrast",
            "large_galaxy_core_protect",
        ):
            with self.subTest(policy=policy_name):
                policy = load_policy(policy_name)
                self.assertNotIn("scoring", policy["stage7_stretch"])

    def test_named_policy_uses_builtin_overlay_when_config_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = load_policy("large_galaxy_core_protect", policy_dir=Path(tmpdir))

        self.assertEqual(policy["policy_name"], "large_galaxy_core_protect")
        self.assertIn("low_contrast_masked_lift", policy["stage7_stretch"]["candidate_mode"])
        self.assertEqual(policy["stage7_stretch"]["fallback_candidate"], "low_contrast_masked_lift")

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

    def test_ngc6910_is_fixed_as_open_cluster_in_main_and_builtin_catalogs(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.01,
            bright_core_score=0.28,
            dense_star_field_score=0.78,
        )
        for catalog in (None, BUILTIN_CATALOG):
            profile = build_target_profile(
                features,
                metadata={"OBJECT": "NGC 6910"},
                context_text="/Users/mz/SeeStar/NGC6910/process",
                **({"catalog": catalog} if catalog is not None else {}),
            )
            self.assertEqual(profile["target_name_guess"], "NGC 6910")
            self.assertEqual(profile["target_type"], "open_cluster")
            self.assertEqual(profile["pipeline"], "open_cluster_color_preserve")
            self.assertIn("large_nebulosity", profile["secondary_labels"])
            self.assertIn("emission_red", profile["secondary_labels"])
            self.assertEqual(
                profile["secondary_label_evidence"]["emission_red"]["source"],
                "catalog_context",
            )

    def test_catalog_identifiers_accept_space_underscore_and_hyphen_boundaries(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.01,
            bright_core_score=0.15,
        )
        cases = {
            "/tasks/M_33/source.fit": ("M33", "large_galaxy"),
            "/tasks/IC-434/source.fit": (
                "Horsehead Nebula",
                "dark_nebula_low_contrast",
            ),
            "/tasks/M-42/source.fit": (
                "M42",
                "bright_emission_reflection_nebula",
            ),
            "/tasks/NGC_6910/source.fit": ("NGC 6910", "open_cluster"),
        }
        for context, expected in cases.items():
            with self.subTest(context=context):
                profile = build_target_profile(
                    features,
                    context_text=context,
                )
                self.assertEqual(profile["target_name_guess"], expected[0])
                self.assertEqual(profile["target_type"], expected[1])
                self.assertGreaterEqual(profile["target_confidence"], 0.90)

    def test_ngc1579_is_bright_nebula_in_main_and_builtin_catalogs(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.02,
            bright_core_score=0.42,
            nebulosity_area_ratio=0.08,
        )
        for catalog in (None, BUILTIN_CATALOG):
            profile = build_target_profile(
                features,
                context_text="/tasks/NGC-1579/source.fit",
                **({"catalog": catalog} if catalog is not None else {}),
            )
            self.assertEqual(profile["target_name_guess"], "NGC 1579")
            self.assertEqual(
                profile["target_type"],
                "bright_emission_reflection_nebula",
            )
            self.assertEqual(
                profile["pipeline"],
                "bright_nebula_hdr_conservative",
            )

    def test_unknown_visual_galaxy_is_demoted_to_generic_safe_policy(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.05,
            bright_core_score=0.42,
            nebulosity_area_ratio=0.10,
            elongation_score=0.42,
        )

        profile = build_target_profile(
            features,
            context_text="/tasks/Xiaoyinh/process/working.fit",
        )

        self.assertIsNone(profile["target_name_guess"])
        self.assertEqual(profile["target_type"], "generic_low_snr_safe")
        self.assertEqual(profile["pipeline"], "generic_low_snr_safe")
        self.assertEqual(
            profile["visual_hypothesis"]["target_type"],
            "small_galaxy",
        )

    def test_explicit_name_coordinate_conflict_fails_safe_and_requires_review(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.05,
            bright_core_score=0.42,
            nebulosity_area_ratio=0.10,
            elongation_score=0.42,
        )

        profile = build_target_profile(
            features,
            metadata={
                "OBJCTRA": "05:35:39.735",
                "OBJCTDEC": "-05:27:33.020",
            },
            context_text="/tasks/M33/source.fit",
        )

        self.assertEqual(profile["identity_status"], "conflict")
        self.assertTrue(profile["target_identity_conflict"])
        self.assertTrue(profile["requires_review"])
        self.assertIsNone(profile["target_name_guess"])
        self.assertEqual(profile["target_type"], "generic_low_snr_safe")
        self.assertEqual(
            profile["identity_evidence"]["name"]["target"],
            "M33",
        )
        self.assertEqual(
            profile["identity_evidence"]["coordinate"]["target"],
            "M42",
        )

    def test_same_type_targets_inside_wcs_field_resolve_as_composite(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.20,
            bright_core_score=0.46,
            nebulosity_area_ratio=0.32,
            red_dominance=1.12,
        )

        profile = build_target_profile(
            features,
            metadata={
                "CRVAL1": 270.868376824048,
                "CRVAL2": -23.52,
                "CDELT1": -0.001022709,
                "CDELT2": 0.001022709,
                "NAXIS1": 2146,
                "NAXIS2": 3174,
            },
            context_text="/tasks/M8/source.fit",
        )

        self.assertEqual(profile["identity_status"], "composite_resolved")
        self.assertFalse(profile["target_identity_conflict"])
        self.assertFalse(profile["requires_review"])
        self.assertEqual(profile["target_name_guess"], "Lagoon Nebula")
        self.assertEqual(
            profile["target_type"],
            "bright_emission_reflection_nebula",
        )
        self.assertEqual(
            {item["name"] for item in profile["composite_targets"]},
            {"Lagoon Nebula", "Trifid Nebula"},
        )
        self.assertIn("reflection_blue", profile["secondary_labels"])
        self.assertEqual(
            profile["secondary_label_evidence"]["reflection_blue"]["source"],
            "catalog_composite_context",
        )
        self.assertFalse(
            profile["routing_contract"]["secondary_labels_can_route"]
        )

    def test_same_type_targets_inside_physical_field_resolve_as_composite(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.20,
            bright_core_score=0.46,
            nebulosity_area_ratio=0.32,
            red_dominance=1.12,
        )

        profile = build_target_profile(
            features,
            metadata={
                "RA": 271.311860502958,
                "DEC": -23.5068506982248,
                "FOCALLEN": 160.0,
                "XPIXSZ": 2.9,
                "YPIXSZ": 2.9,
                "NAXIS1": 2146,
                "NAXIS2": 3772,
            },
            context_text="/tasks/M8/source.fit",
        )

        self.assertEqual(profile["identity_status"], "composite_resolved")
        self.assertFalse(profile["target_identity_conflict"])
        self.assertFalse(profile["requires_review"])
        self.assertEqual(profile["target_name_guess"], "Lagoon Nebula")
        self.assertEqual(
            {item["name"] for item in profile["composite_targets"]},
            {"Lagoon Nebula", "Trifid Nebula"},
        )
        self.assertIn("reflection_blue", profile["secondary_labels"])

    def test_target_profiler_prefers_explicit_rosette_name_over_auto_hint(self) -> None:
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

        self.assertEqual(profile["target_name_guess"], "Rosette Nebula")
        self.assertEqual(profile["target_type"], "emission_nebula_widefield")
        self.assertEqual(profile["pipeline"], "emission_nebula_widefield")
        self.assertIn("auto_target_hint", ",".join(profile["diagnostics"]))

    def test_catalog_visual_difference_is_diagnostic_not_warning(self) -> None:
        features = AdaptiveImageFeatures(
            object_area_ratio=0.35,
            bright_core_score=0.30,
            nebulosity_area_ratio=0.34,
            red_dominance=1.22,
            blue_dominance=1.18,
        )

        profile = build_target_profile(
            features,
            metadata={"OBJCTRA": "05:35:39.735", "OBJCTDEC": "-05:27:33.020"},
        )

        self.assertIn("catalog_visual_type_resolution", ",".join(profile["diagnostics"]))
        self.assertEqual(profile["warnings"], [])

if __name__ == "__main__":
    unittest.main()
