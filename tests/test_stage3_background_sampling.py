#!/usr/bin/env python3
"""Focused tests for Stage 3 safe samples and pattern-noise routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from background_sampling import (  # noqa: E402
    assess_background_process,
    assess_compound_background_validation,
    assess_single_background_validation,
    assess_target_fidelity,
    analyze_directional_pattern_noise,
    build_safe_background_samples,
    measure_background_validation,
    pattern_candidate_gate,
    select_background_route,
    split_background_sample_points,
)
from image_metrics import measure_stage3_signal_preservation  # noqa: E402


def _gradient_image(*, seed: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = 256, 320
    y, x = np.mgrid[:height, :width]
    return (
        0.04
        + 0.06 * x / (width - 1)
        + 0.025 * y / (height - 1)
        + rng.normal(0.0, 0.001, (height, width))
    )


def _compound_grid_points(*, height: int, width: int):
    return [
        (
            (cell_x + fraction) / 4.0 * (width - 1),
            (cell_y + fraction) / 4.0 * (height - 1),
        )
        for cell_y in range(4)
        for cell_x in range(4)
        for fraction in (0.30, 0.70)
    ]


class Stage3BackgroundSamplingTests(unittest.TestCase):
    def test_safe_samples_are_spatial_and_avoid_nebula_and_stars(self):
        image = _gradient_image()
        height, width = image.shape
        y, x = np.mgrid[:height, :width]
        nebula_center = (width * 0.50, height * 0.48)
        image += 0.12 * np.exp(
            -(
                (x - nebula_center[0]) ** 2
                + (y - nebula_center[1]) ** 2
            )
            / (2 * 38**2)
        )
        stars = [(40, 40), (270, 60), (90, 200), (235, 190)]
        for star_x, star_y in stars:
            image += 0.30 * np.exp(
                -((x - star_x) ** 2 + (y - star_y) ** 2) / (2 * 3**2)
            )

        points, report = build_safe_background_samples(image)

        self.assertEqual(report["status"], "ready")
        self.assertGreaterEqual(len(points), report["minimum_count"])
        self.assertEqual(report["coordinate_system"], "siril_bottom_left")
        self.assertGreaterEqual(report["coverage"]["quadrants"], 3)
        self.assertGreaterEqual(report["coverage"]["grid_cells"], 8)
        self.assertGreaterEqual(report["coverage"]["x_span_ratio"], 0.55)
        self.assertGreaterEqual(report["coverage"]["y_span_ratio"], 0.55)

        top_left_points = [
            (point_x, height - 1 - point_y)
            for point_x, point_y in points
        ]
        center_distance = min(
            np.hypot(
                point_x - nebula_center[0],
                point_y - nebula_center[1],
            )
            for point_x, point_y in top_left_points
        )
        star_distance = min(
            min(
                np.hypot(point_x - star_x, point_y - star_y)
                for star_x, star_y in stars
            )
            for point_x, point_y in top_left_points
        )
        self.assertGreater(center_distance, 65.0)
        self.assertGreater(star_distance, 12.0)

    def test_safe_samples_fail_closed_without_dynamic_range(self):
        points, report = build_safe_background_samples(
            np.full((128, 128), 0.05, dtype=np.float32)
        )

        self.assertEqual(points, [])
        self.assertEqual(report["status"], "unavailable")

    def test_shared_scene_support_excludes_invalid_saturated_and_catalog_patches(self):
        image = _gradient_image(seed=44)
        height, width = image.shape
        catalog = [
            {
                "x": float(x),
                "y": float(y),
                "fwhm_px": 20.0,
            }
            for y in range(0, height, 32)
            for x in range(0, width, 32)
        ]
        cases = (
            (
                {"shared_valid_mask": np.zeros((height, width), dtype=np.uint8)},
                "valid_mask",
                "shared_invalid_region",
            ),
            (
                {"shared_saturation_map": np.ones((height, width), dtype=np.uint8)},
                "saturation_map",
                "shared_saturated",
            ),
            (
                {"shared_star_catalog": catalog},
                "star_catalog",
                "shared_catalog_star",
            ),
        )

        for kwargs, component, rejection in cases:
            with self.subTest(component=component):
                points, report = build_safe_background_samples(image, **kwargs)
                self.assertEqual(points, [])
                self.assertEqual(
                    report["shared_scene_support"][component],
                    "applied",
                )
                self.assertGreater(report["rejection_counts"][rejection], 0)

    def test_multiscale_mask_evidence_is_applied_and_catalog_halo_has_zero_overlap(self):
        image = _gradient_image(seed=52)
        height, width = image.shape
        star = {"x": width / 2, "y": height / 2, "fwhm_px": 8.0}

        points, report = build_safe_background_samples(
            image,
            shared_star_catalog=[star],
            protection_policy={"protect_star_halo": True},
        )

        self.assertEqual(report["status"], "ready")
        evidence = report["mask_evidence"]
        self.assertTrue(evidence["applied_to_sampling"])
        self.assertGreater(evidence["combined_excluded_fraction"], 0.0)
        self.assertGreater(evidence["usable_sky_fraction"], 0.0)
        self.assertLess(evidence["usable_sky_fraction"], 1.0)
        for layer in (
            "invalid_or_uncovered",
            "image_stars_and_sources",
            "scene_support_stars",
            "positive_structure_nebulosity",
            "bright_core",
            "dark_structure",
            "outer_halo",
        ):
            self.assertIn(layer, evidence["layers"])
            self.assertIn("requested", evidence["layers"][layer])
            self.assertIn("available", evidence["layers"][layer])
            self.assertIn("applied", evidence["layers"][layer])
            self.assertIn("pixel_fraction", evidence["layers"][layer])
            self.assertIn("method", evidence["layers"][layer])
            self.assertIn("reason", evidence["layers"][layer])
        self.assertTrue(evidence["layers"]["scene_support_stars"]["applied"])
        self.assertEqual(
            evidence["layers"]["scene_support_stars"]["method"],
            "scene_support_catalog_4x_fwhm",
        )
        self.assertFalse(evidence["layers"]["outer_halo"]["requested"])
        for point_x, point_y in points:
            top_y = height - 1 - point_y
            self.assertGreater(
                np.hypot(point_x - star["x"], top_y - star["y"]),
                4.0 * star["fwhm_px"] + report["patch_radius"],
            )

    def test_bright_core_mask_requires_coherent_positive_structure(self):
        image = _gradient_image(seed=57)
        height, width = image.shape
        y, x = np.mgrid[:height, :width]
        rng = np.random.default_rng(57)
        for star_x, star_y in zip(
            rng.integers(8, width - 8, size=240),
            rng.integers(8, height - 8, size=240),
        ):
            image += 0.18 * np.exp(
                -((x - star_x) ** 2 + (y - star_y) ** 2) / (2 * 1.1**2)
            )

        _points, report = build_safe_background_samples(image)

        layers = report["masks"]["layer_fractions"]
        self.assertLessEqual(
            layers["bright_core"],
            layers["extended_structure"] + 1e-12,
        )
        self.assertLess(layers["bright_core"], 0.20)

    def test_dense_scene_catalog_expands_search_without_relaxing_star_mask(self):
        image = _gradient_image(seed=58)
        height, width = image.shape
        catalog = [
            {"x": float(x), "y": float(y), "fwhm_px": 3.0}
            for y in range(8, height, 18)
            for x in range(8, width, 18)
        ]

        _points, report = build_safe_background_samples(
            image,
            shared_star_catalog=catalog,
        )

        search = report["candidate_search"]
        self.assertTrue(search["dense_star_field_expansion"])
        self.assertEqual(search["grid_multiplier"], 6)
        self.assertGreaterEqual(search["scene_support_star_fraction"], 0.15)
        self.assertGreater(report["rejection_counts"]["shared_catalog_star"], 0)

    def test_dark_patch_refinement_is_deterministic_and_keeps_frozen_thresholds(self):
        image = _gradient_image(seed=31)
        height, width = image.shape
        y, x = np.mgrid[:height, :width]
        image += 0.10 * np.exp(
            -((x - width * 0.52) ** 2 + (y - height * 0.47) ** 2)
            / (2 * 42**2)
        )

        baseline_points, baseline_report = build_safe_background_samples(
            image,
            candidate_refinement=False,
        )
        first_points, first_report = build_safe_background_samples(image)
        second_points, second_report = build_safe_background_samples(image)

        self.assertEqual(first_points, second_points)
        self.assertEqual(first_report, second_report)
        self.assertTrue(first_report["refinement"]["enabled"])
        self.assertGreater(
            first_report["refinement"]["accepted_candidate_count"],
            0,
        )
        self.assertGreater(len(first_points), len(baseline_points))
        self.assertEqual(first_report["thresholds"], baseline_report["thresholds"])
        self.assertEqual(len(first_points), len(set(first_points)))

        top_left_points = [
            (point_x, height - 1 - point_y)
            for point_x, point_y in first_points
        ]
        minimum_spacing = max(
            first_report["patch_radius"] * 2,
            int(round(min(height, width) * 0.035)),
        )
        observed_spacing = min(
            np.hypot(left_x - right_x, left_y - right_y)
            for index, (left_x, left_y) in enumerate(top_left_points)
            for right_x, right_y in top_left_points[index + 1 :]
        )
        self.assertGreaterEqual(observed_spacing, minimum_spacing)

    def test_dark_patch_refinement_stays_disabled_for_sky_limited_structure(self):
        rng = np.random.default_rng(91)
        height, width = 256, 320
        y, x = np.mgrid[:height, :width]
        image = (
            0.03
            + 0.02 * x / (width - 1)
            + rng.normal(0.0, 0.0005, (height, width))
            + 0.06 * np.maximum(0.0, np.sin(x / 5.0) + np.sin(y / 6.5))
        )

        _points, report = build_safe_background_samples(image)

        self.assertFalse(report["refinement"]["enabled"])
        self.assertEqual(
            report["refinement"]["accepted_candidate_count"],
            0,
        )
        self.assertIn(
            "usable_sky_fraction_below_0_50",
            report["refinement"]["block_reasons"],
        )
        self.assertIn(
            "source_mask_fraction_above_0_50",
            report["refinement"]["block_reasons"],
        )

    def test_compound_split_is_deterministic_disjoint_and_spatial(self):
        rng = np.random.default_rng(27)
        height, width = 512, 640
        y, x = np.mgrid[:height, :width]
        image = (
            0.04
            + 0.06 * x / (width - 1)
            + 0.025 * y / (height - 1)
            + rng.normal(0.0, 0.001, (height, width))
        )
        points = _compound_grid_points(height=height, width=width)

        first_fit, first_validation, first_report = (
            split_background_sample_points(points, image)
        )
        second_fit, second_validation, second_report = (
            split_background_sample_points(points, image)
        )

        self.assertEqual(first_report["status"], "ready")
        self.assertEqual(first_fit, second_fit)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(first_report, second_report)
        self.assertGreaterEqual(len(first_fit), 24)
        self.assertGreaterEqual(len(first_validation), 8)
        self.assertFalse(set(first_fit) & set(first_validation))
        self.assertEqual(set(first_fit) | set(first_validation), set(points))
        self.assertGreaterEqual(
            first_report["fit_coverage"]["grid_cells"],
            8,
        )
        self.assertGreaterEqual(
            first_report["validation_coverage"]["grid_cells"],
            8,
        )
        fit, validation, report = split_background_sample_points(
            points[:11],
            image,
        )
        self.assertEqual((fit, validation), ([], []))
        self.assertEqual(report["status"], "insufficient_samples")

    def test_compound_validation_requires_improvement_and_bounded_drift(self):
        height, width = 512, 640
        y, x = np.mgrid[:height, :width]
        base = 0.05 + 0.08 * x / (width - 1) + 0.02 * y / (height - 1)
        points = _compound_grid_points(height=height, width=width)
        _fit, validation, split_report = split_background_sample_points(
            points,
            base,
        )
        self.assertEqual(split_report["status"], "ready")
        best_single = 0.05 + 0.045 * x / (width - 1) + 0.012 * y / (height - 1)
        polynomial = 0.05 + 0.025 * x / (width - 1) + 0.006 * y / (height - 1)
        compound = 0.0505 + 0.012 * x / (width - 1) + 0.003 * y / (height - 1)

        baseline_metrics = measure_background_validation(base, validation)
        scale = baseline_metrics["value_scale"]
        single_metrics = measure_background_validation(
            best_single,
            validation,
            value_scale=scale,
        )
        polynomial_metrics = measure_background_validation(
            polynomial,
            validation,
            value_scale=scale,
        )
        compound_metrics = measure_background_validation(
            compound,
            validation,
            value_scale=scale,
        )
        accepted, gate = assess_compound_background_validation(
            baseline_metrics,
            single_metrics,
            polynomial_metrics,
            compound_metrics,
        )

        self.assertTrue(accepted, gate)
        self.assertGreaterEqual(gate["span_improvement_ratio"], 0.10)
        self.assertLessEqual(
            gate["zero_point_drift"],
            gate["zero_point_limit"],
        )

        drifted_metrics = measure_background_validation(
            compound + 0.03,
            validation,
            value_scale=scale,
        )
        accepted, gate = assess_compound_background_validation(
            baseline_metrics,
            single_metrics,
            polynomial_metrics,
            drifted_metrics,
        )
        self.assertTrue(accepted)
        self.assertEqual(gate["severity"], "soft_warning")
        self.assertTrue(
            any("zero-point drift" in issue for issue in gate["issues"])
        )
        strict_accepted, strict_gate = assess_compound_background_validation(
            baseline_metrics,
            single_metrics,
            polynomial_metrics,
            drifted_metrics,
            gate_profile="strict",
        )
        self.assertFalse(strict_accepted)
        self.assertEqual(strict_gate["severity"], "hard_rejected")

    def test_validation_uses_correlation_aware_patch_uncertainty(self):
        rng = np.random.default_rng(44)
        height, width = 256, 320
        y, x = np.mgrid[:height, :width]
        correlated = np.repeat(
            np.repeat(rng.normal(0.0, 0.001, (64, 80)), 4, axis=0),
            4,
            axis=1,
        )
        image = (
            0.04
            + 0.03 * x / (width - 1)
            + 0.01 * y / (height - 1)
            + correlated
        )
        points = _compound_grid_points(height=height, width=width)

        validation = measure_background_validation(image, points)

        naive_pixel_standard_error = (
            1.2533
            * validation["patch_mad_median"]
            / np.sqrt((2 * validation["patch_radius"] + 1) ** 2)
        )
        self.assertEqual(
            validation["schema_version"],
            "starun.stage3-background-validation.v1",
        )
        self.assertEqual(
            validation["uncertainty_method"],
            "correlation_aware_nonoverlapping_block_medians",
        )
        self.assertGreaterEqual(
            validation["patch_median_uncertainty"],
            naive_pixel_standard_error,
        )

    def test_process_evidence_authorizes_low_order_additive_gradient(self):
        image = _gradient_image()
        points = _compound_grid_points(
            height=image.shape[0],
            width=image.shape[1],
        )
        baseline = measure_background_validation(image, points)
        process = assess_background_process(
            image,
            points,
            {
                "status": "ready",
                "masks": {
                    "usable_sky_fraction": 0.62,
                    "usable_sky_grid_cells": 16,
                },
            },
            baseline,
            {"status": "ok", "detected": False},
            input_profile={
                "state": "linear",
                "safe_for_linear_steps": True,
                "confidence": 0.9,
                "source": "test_metadata",
            },
            diffuse_context={"target_type": "large_galaxy"},
        )

        self.assertEqual(process["mechanism"], "additive_low_frequency_gradient")
        self.assertEqual(
            process["schema_version"],
            "starun.stage3-process-evidence.v1",
        )
        self.assertTrue(process["should_evaluate"])
        self.assertTrue(process["low_complexity_required"])
        self.assertEqual(process["model_complexity_limit"], "polynomial_degree_1")

    def test_process_evidence_reports_combined_mask_and_safe_candidate_geometry(self):
        image = _gradient_image()
        process = assess_background_process(
            image,
            [],
            {
                "status": "insufficient_safe_coverage",
                "masks": {
                    "usable_sky_fraction": 0.60,
                    "usable_sky_grid_cells": 16,
                },
                "mask_evidence": {"usable_sky_fraction": 0.455},
                "coverage": {"available_grid_cells": 0},
            },
            {"status": "not_run"},
            {"status": "ok", "detected": False},
            input_profile={
                "state": "linear",
                "safe_for_linear_steps": True,
            },
        )

        self.assertEqual(
            process["true_sky_support"]["usable_sky_fraction"],
            0.455,
        )
        self.assertEqual(
            process["true_sky_support"]["usable_sky_grid_cells"],
            0,
        )
        self.assertFalse(process["true_sky_support"]["supported"])

    def test_process_evidence_evaluates_radial_gradient_with_flat_advisory(self):
        rng = np.random.default_rng(61)
        height, width = 320, 400
        y, x = np.mgrid[:height, :width]
        nx = (x - (width - 1) / 2.0) / max(width - 1, 1)
        ny = (y - (height - 1) / 2.0) / max(height - 1, 1)
        image = (
            0.08
            - 0.06 * (nx * nx + ny * ny)
            + rng.normal(0.0, 0.0002, (height, width))
        )
        points = _compound_grid_points(height=height, width=width)
        baseline = measure_background_validation(image, points)
        process = assess_background_process(
            image,
            points,
            {
                "status": "ready",
                "masks": {
                    "usable_sky_fraction": 0.70,
                    "usable_sky_grid_cells": 16,
                },
            },
            baseline,
            {"status": "ok", "detected": False},
            input_profile={
                "state": "linear",
                "safe_for_linear_steps": True,
            },
        )

        self.assertEqual(
            process["mechanism"],
            "radial_low_frequency_gradient_ambiguous",
        )
        self.assertTrue(process["should_evaluate"])
        self.assertEqual(
            process["correction_mode"],
            "subtract_with_master_flat_advisory",
        )
        self.assertTrue(process["radial_shape_supported"])
        self.assertIn(
            "radial_shape_cannot_distinguish_additive_gradient_from_flat_error",
            process["advisory_reasons"],
        )
        self.assertEqual(process["hard_block_reasons"], [])
        route = select_background_route({}, {"detected": False}, process_report=process)
        self.assertTrue(route["gradient_supported"])
        self.assertEqual(route["route"], "low_frequency_gradient")

    def test_target_fidelity_softens_moderate_source_loss_by_default(self):
        rng = np.random.default_rng(73)
        height, width = 256, 320
        y, x = np.mgrid[:height, :width]
        plane = 0.035 + 0.018 * x / (width - 1) + 0.010 * y / (height - 1)
        target = 0.10 * np.exp(
            -(
                ((x - width * 0.52) / 48.0) ** 2
                + ((y - height * 0.49) / 36.0) ** 2
            )
            / 2.0
        )
        before = plane + target + rng.normal(0.0, 0.0004, (height, width))
        corrected = before - plane + 0.045
        oversubtracted = corrected - 0.35 * target
        sky_points = [
            (24.0, 24.0),
            (295.0, 24.0),
            (24.0, 230.0),
            (295.0, 230.0),
        ]

        good = measure_stage3_signal_preservation(
            before,
            corrected,
            sky_points=sky_points,
        )
        good_ok, good_gate = assess_target_fidelity(
            good,
            low_complexity_required=True,
        )
        bad = measure_stage3_signal_preservation(
            before,
            oversubtracted,
            sky_points=sky_points,
        )
        bad_ok, bad_gate = assess_target_fidelity(
            bad,
            low_complexity_required=True,
        )

        self.assertTrue(good_ok, good_gate)
        self.assertEqual(good["target_sky_reference"], "heldout_sky_plane_degree_1")
        self.assertAlmostEqual(
            good["target_flux_retention_ratio"],
            1.0,
            delta=0.002,
        )
        self.assertTrue(bad_ok, bad_gate)
        self.assertEqual(bad_gate["severity"], "soft_warning")
        self.assertLess(bad["target_flux_change_significance"], -3.0)
        self.assertTrue(
            any("target flux loss" in issue for issue in bad_gate["issues"])
        )
        strict_ok, strict_gate = assess_target_fidelity(
            bad,
            low_complexity_required=True,
            gate_profile="strict",
        )
        self.assertFalse(strict_ok)
        self.assertEqual(strict_gate["severity"], "hard_rejected")

    def test_target_fidelity_output_first_rejects_only_extreme_change(self):
        report = {
            "available": True,
            "target_sky_reference": "heldout_sky_plane_degree_1",
            "target_flux_retention_ratio": 4.5,
            "target_flux_change_significance": 135.0,
            "target_morphology_correlation": 0.45,
            "target_change_residual_significance": 1.0,
            "target_centroid_shift_fraction": 0.13,
        }

        accepted, gate = assess_target_fidelity(
            report,
            low_complexity_required=False,
        )

        self.assertFalse(accepted)
        self.assertEqual(gate["profile"], "output_first")
        self.assertEqual(gate["severity"], "hard_rejected")
        self.assertTrue(any("flux growth is excessive" in issue for issue in gate["hard_issues"]))

    def test_target_fidelity_balanced_uses_intermediate_hard_limits(self):
        base = {
            "available": True,
            "target_sky_reference": "heldout_sky_plane_degree_1",
            "target_flux_retention_ratio": 1.0,
            "target_flux_change_significance": 0.0,
            "target_morphology_correlation": 0.55,
            "target_change_residual_significance": 1.0,
            "target_centroid_shift_fraction": 0.11,
        }

        output_ok, output_gate = assess_target_fidelity(
            base,
            low_complexity_required=False,
            gate_profile="output_first",
        )
        balanced_ok, balanced_gate = assess_target_fidelity(
            base,
            low_complexity_required=False,
            gate_profile="balanced",
        )

        self.assertTrue(output_ok, output_gate)
        self.assertEqual(output_gate["severity"], "soft_warning")
        self.assertFalse(balanced_ok)
        self.assertEqual(balanced_gate["severity"], "hard_rejected")
        self.assertEqual(balanced_gate["profile"], "balanced")

    def test_batch_regression_m31_m33_softens_graxpert_rejections(self):
        # Frozen metrics from the 2026-08-11 M31/M33 Stage 3 reports.
        samples = {
            "M31": (0.7021789879, -13.5163206, 0.9742678011, 0.0033138954, 19.3588124),
            "M33": (0.4198179414, -6.1838996, 0.9465315208, 0.0124553912, 4.3946548),
        }
        for label, values in samples.items():
            with self.subTest(label=label):
                flux, significance, morphology, centroid, structure = values
                accepted, gate = assess_target_fidelity(
                    {
                        "available": True,
                        "target_sky_reference": "heldout_sky_plane_degree_1",
                        "target_flux_retention_ratio": flux,
                        "target_flux_change_significance": significance,
                        "target_morphology_correlation": morphology,
                        "target_centroid_shift_fraction": centroid,
                        "target_change_residual_significance": structure,
                    },
                    low_complexity_required=True,
                    gate_profile="output_first",
                )
                self.assertTrue(accepted, gate)
                self.assertEqual(gate["severity"], "soft_warning")

    def test_batch_regression_dwarf_ic434_extreme_plugins_stay_hard_rejected(self):
        # ADBE/DBE/AutoDBE flux growth observed in the Dwarf IC434 batch.
        samples = (
            (5.4603697378, 176.6057993, 0.4638830871, 0.1691496149),
            (4.4231739503, 134.8946439, 0.4396387106, 0.1308457798),
            (5.3597662307, 172.0481022, 0.4502337119, 0.1642541962),
        )
        for flux, significance, morphology, centroid in samples:
            with self.subTest(flux=flux):
                accepted, gate = assess_target_fidelity(
                    {
                        "available": True,
                        "target_sky_reference": "heldout_sky_plane_degree_1",
                        "target_flux_retention_ratio": flux,
                        "target_flux_change_significance": significance,
                        "target_morphology_correlation": morphology,
                        "target_centroid_shift_fraction": centroid,
                        "target_change_residual_significance": 1.0,
                    },
                    low_complexity_required=True,
                    gate_profile="output_first",
                )
                self.assertFalse(accepted)
                self.assertEqual(gate["severity"], "hard_rejected")

    def test_missing_gate_metrics_degrade_unless_strict(self):
        accepted, gate = assess_target_fidelity(
            {},
            low_complexity_required=True,
        )
        strict_accepted, strict_gate = assess_target_fidelity(
            {},
            low_complexity_required=True,
            gate_profile="strict",
        )

        self.assertTrue(accepted)
        self.assertEqual(gate["severity"], "soft_warning")
        self.assertFalse(strict_accepted)
        self.assertEqual(strict_gate["severity"], "hard_rejected")

    def test_single_validation_uses_sampling_uncertainty_not_score_thresholds(self):
        baseline = {
            "status": "ready",
            "robust_span": 0.0010,
            "patch_mad_median": 0.0002,
            "patch_radius": 12,
            "median": 0.05,
        }
        improved = {
            "status": "ready",
            "robust_span": 0.0004,
            "patch_mad_median": 0.0002,
            "patch_radius": 12,
            "median": 0.05,
        }

        accepted, gate = assess_single_background_validation(
            baseline,
            improved,
        )

        self.assertTrue(accepted, gate)
        self.assertTrue(gate["material_improvement"])
        self.assertIn("sampling_uncertainty_3sigma", gate)

    def test_single_validation_is_conservative_for_correlated_patches(self):
        baseline = {
            "status": "ready",
            "robust_span": 0.0010,
            "patch_mad_median": 0.0002,
            "patch_median_uncertainty": 0.0002,
            "patch_radius": 12,
            "median": 0.05,
        }
        weak_improvement = {
            "status": "ready",
            "robust_span": 0.0004,
            "patch_mad_median": 0.0002,
            "patch_median_uncertainty": 0.0002,
            "patch_radius": 12,
            "median": 0.05,
        }

        accepted, gate = assess_single_background_validation(
            baseline,
            weak_improvement,
        )

        self.assertTrue(accepted, gate)
        self.assertEqual(gate["severity"], "soft_warning")
        self.assertEqual(
            gate["uncertainty_method"],
            "correlation_aware_span_difference",
        )
        self.assertFalse(gate["material_improvement"])
        strict_accepted, strict_gate = assess_single_background_validation(
            baseline,
            weak_improvement,
            gate_profile="strict",
        )
        self.assertFalse(strict_accepted)
        self.assertEqual(strict_gate["severity"], "hard_rejected")

    def test_single_validation_balanced_rejects_intermediate_span_worsening(self):
        baseline = {
            "status": "ready",
            "robust_span": 0.010,
            "patch_mad_median": 0.004,
            "patch_median_uncertainty": 0.0005,
        }
        candidate = {
            "status": "ready",
            "robust_span": 0.020,
            "patch_mad_median": 0.004,
            "patch_median_uncertainty": 0.0005,
        }

        output_ok, output_gate = assess_single_background_validation(
            baseline,
            candidate,
            gate_profile="output_first",
        )
        balanced_ok, balanced_gate = assess_single_background_validation(
            baseline,
            candidate,
            gate_profile="balanced",
        )

        self.assertTrue(output_ok, output_gate)
        self.assertEqual(output_gate["severity"], "soft_warning")
        self.assertFalse(balanced_ok)
        self.assertEqual(balanced_gate["severity"], "hard_rejected")

    def test_smooth_gradient_is_not_directional_pattern_noise(self):
        report = analyze_directional_pattern_noise(_gradient_image())

        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["detected"])
        self.assertEqual(report["kind"], "none")

    def test_isotropic_correlated_texture_is_not_directional_pattern_noise(self):
        image = _gradient_image()
        y, x = np.mgrid[: image.shape[0], : image.shape[1]]
        image += 0.012 * np.exp(
            -((x - 155) ** 2 + (y - 126) ** 2) / (2 * 52**2)
        )
        for _ in range(4):
            padded = np.pad(image, 1, mode="reflect")
            image = sum(
                padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ) / 9.0

        report = analyze_directional_pattern_noise(image)

        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["detected"])
        self.assertLess(
            report["directional_score"],
            report["thresholds"]["pattern_score_min"],
        )

    def test_horizontal_banding_is_deferred_from_subsky(self):
        image = _gradient_image()
        y = np.arange(image.shape[0], dtype=np.float64)[:, None]
        report = analyze_directional_pattern_noise(
            image + 0.025 * np.sin(2 * np.pi * y / 8)
        )
        route = select_background_route(
            {
                "gradient_score": 0.03,
                "dirty_background_score": 0.08,
            },
            report,
        )

        self.assertTrue(report["detected"])
        self.assertEqual(report["kind"], "horizontal_banding")
        self.assertEqual(route["route"], "pattern_noise_deferred")
        self.assertEqual(route["pattern_branch"], "banding_review")
        self.assertFalse(route["subsky_existing_allowed"])
        self.assertEqual(
            route["correction_owner"],
            "calibration_or_sensor_review",
        )

    def test_diagonal_walking_noise_has_its_own_route(self):
        image = _gradient_image()
        y, x = np.mgrid[: image.shape[0], : image.shape[1]]
        report = analyze_directional_pattern_noise(
            image + 0.025 * np.sin(2 * np.pi * (x - y) / 11)
        )

        self.assertTrue(report["detected"])
        self.assertEqual(report["kind"], "diagonal_walking_noise")
        self.assertGreaterEqual(
            report["walking_noise_score"],
            report["thresholds"]["walking_noise_score_min"],
        )
        route = select_background_route(
            {"gradient_score": 0.02, "dirty_background_score": 0.06},
            report,
        )
        self.assertEqual(route["pattern_branch"], "walking_noise_review")

    def test_mixed_gradient_and_pattern_noise_allows_only_reviewed_subsky(self):
        image = _gradient_image()
        y = np.arange(image.shape[0], dtype=np.float64)[:, None]
        report = analyze_directional_pattern_noise(
            image + 0.025 * np.sin(2 * np.pi * y / 8)
        )
        route = select_background_route(
            {
                "gradient_score": 0.15,
                "dirty_background_score": 0.30,
            },
            report,
        )

        self.assertEqual(
            route["route"],
            "mixed_gradient_and_pattern_noise",
        )
        self.assertTrue(route["subsky_existing_allowed"])
        self.assertTrue(route["requires_review"])

    def test_candidate_gate_softens_introduced_pattern_noise_by_default(self):
        before = analyze_directional_pattern_noise(_gradient_image())
        image = _gradient_image()
        y, x = np.mgrid[: image.shape[0], : image.shape[1]]
        after = analyze_directional_pattern_noise(
            image + 0.025 * np.sin(2 * np.pi * (x - y) / 11)
        )

        accepted, gate = pattern_candidate_gate(before, after)

        self.assertTrue(accepted, gate)
        self.assertTrue(gate["introduced_pattern_noise"])
        self.assertEqual(gate["status"], "accepted_with_warnings")
        strict_accepted, strict_gate = pattern_candidate_gate(
            before,
            after,
            gate_profile="strict",
        )
        self.assertFalse(strict_accepted)
        self.assertEqual(strict_gate["status"], "rejected")

    def test_pattern_gate_balanced_rejects_intermediate_extreme(self):
        before = {
            "status": "ok",
            "pattern_score": 0.40,
            "detected": False,
        }
        after = {
            "status": "ok",
            "pattern_score": 0.88,
            "detected": True,
            "thresholds": {"pattern_score_min": 0.55},
        }

        output_ok, output_gate = pattern_candidate_gate(
            before,
            after,
            gate_profile="output_first",
        )
        balanced_ok, balanced_gate = pattern_candidate_gate(
            before,
            after,
            gate_profile="balanced",
        )

        self.assertTrue(output_ok, output_gate)
        self.assertEqual(output_gate["severity"], "soft_warning")
        self.assertFalse(balanced_ok)
        self.assertEqual(balanced_gate["severity"], "hard_rejected")


if __name__ == "__main__":
    unittest.main()
