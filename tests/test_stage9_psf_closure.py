#!/usr/bin/env python3
"""Display-domain same-star FWHM closure tests for Stage 9."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from models import PipelineConfig  # noqa: E402
import stage9_quality  # noqa: E402


def _gaussian_field(
    shape: tuple[int, int],
    coordinates: list[tuple[int, int]],
    *,
    sigma: float,
    amplitude: float = 0.65,
) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    plane = np.zeros(shape, dtype=np.float32)
    for center_y, center_x in coordinates:
        plane += amplitude * np.exp(
            -(
                (xx - center_x) ** 2
                + (yy - center_y) ** 2
            )
            / (2.0 * sigma * sigma)
        ).astype(np.float32)
    return np.stack([plane, plane * 0.92, plane * 0.81])


class Stage9PsfClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PipelineConfig()
        self.coordinates = [
            (y, x)
            for y in (16, 38, 60, 82, 104)
            for x in (16, 38, 60, 82, 104)
        ]

    def _reference(self, source: np.ndarray) -> dict:
        source_peak = np.max(source, axis=0)
        fwhm = []
        halfmax_area = []
        saturated = []
        for center_y, center_x in self.coordinates:
            measured = stage9_quality._measure_connected_halfmax_fwhm(
                source_peak,
                center_y,
                center_x,
            )
            self.assertEqual(measured["status"], "ok", measured)
            fwhm.append(measured["fwhm_px"])
            halfmax_area.append(measured["half_max_area"])
            saturated.append(measured["saturated"])
        count = len(self.coordinates)
        weak = np.asarray([index < 20 for index in range(count)], dtype=bool)
        return {
            "status": "ok",
            "source_matched": True,
            "psf_reference_status": "ready",
            "psf_sample_count": count,
            "_display_source_fwhm_px": np.asarray(fwhm, dtype=np.float32),
            "_display_source_halfmax_area_px": np.asarray(
                halfmax_area,
                dtype=np.float32,
            ),
            "_psf_valid_flags": np.ones(count, dtype=bool),
            "_psf_saturated_flags": np.asarray(saturated, dtype=bool),
            "_peak_y": np.asarray([item[0] for item in self.coordinates]),
            "_peak_x": np.asarray([item[1] for item in self.coordinates]),
            "_weak_flags": weak,
        }

    def test_spatial_scale_prefers_matched_display_and_scales_math(self):
        catalog = self._reference(
            _gaussian_field((128, 128), self.coordinates, sigma=1.35)
        )
        catalog["_psf_isolated_flags"] = np.ones(
            len(self.coordinates), dtype=bool
        )
        report = stage9_quality.resolve_stage9_spatial_scale(
            catalog,
            stage5_stars=[
                {"geometry_valid": True, "saturated": False, "fwhm_geometry": 8.0}
                for _ in range(4)
            ],
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["source"], "matched_display_fwhm")
        self.assertFalse(report["stage9_psf_review_required"])
        for fwhm, expected_radius, expected_area in (
            (2.0, 0.5, 0.25),
            (4.0, 1.0, 1.0),
            (8.0, 2.0, 4.0),
        ):
            catalog["stage9_spatial_scale"] = {
                "status": "ready",
                "fwhm_median_px": fwhm,
                "radius_scale": expected_radius,
                "area_scale": expected_area,
            }
            self.assertEqual(
                stage9_quality.stage9_scale_distance(1.0, catalog),
                expected_radius,
            )
            self.assertEqual(
                stage9_quality.stage9_scale_area(64, catalog),
                int(64 * expected_area),
            )

    def test_spatial_scale_stage5_fallback_requires_review(self):
        catalog = {"status": "ok", "_peak_y": np.arange(4)}
        report = stage9_quality.resolve_stage9_spatial_scale(
            catalog,
            stage5_stars=[
                {
                    "geometry_valid": True,
                    "saturated": False,
                    "fwhm_geometry": value,
                }
                for value in (3.0, 4.0, 5.0, 6.0)
            ],
        )

        self.assertEqual(report["source"], "stage5_fwhm_geometry")
        self.assertTrue(report["stage9_psf_review_required"])
        self.assertEqual(report["sample_count"], 4)

    def test_spatial_scale_fails_closed_with_too_few_samples(self):
        report = stage9_quality.resolve_stage9_spatial_scale(
            {"status": "ok"},
            stage5_stars=[
                {
                    "geometry_valid": True,
                    "saturated": False,
                    "fwhm_geometry": value,
                }
                for value in (3.0, 4.0, 5.0)
            ],
        )

        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(
            report["reason_code"], "stage9_spatial_scale_unavailable"
        )

    def test_frozen_per_star_windows_follow_each_fwhm(self):
        catalog = {
            "status": "ok",
            "_peak_y": np.asarray([5, 10, 15, 20]),
            "_peak_x": np.asarray([5, 10, 15, 20]),
            "_stage9_spatial_fwhm_px": np.asarray([2.0, 4.0, 8.0, 4.0]),
            "stage9_spatial_scale": {
                "status": "ready",
                "fwhm_median_px": 4.0,
                "radius_scale": 1.0,
                "area_scale": 1.0,
            },
        }
        raw = np.zeros((3, 27, 27), dtype=np.float32)
        for y, x in zip(catalog["_peak_y"], catalog["_peak_x"]):
            raw[:, y, x] = 1.0

        geometry = stage9_quality.freeze_stage9_spatial_geometry(catalog, raw)

        self.assertEqual(
            catalog["_stage9_inner_window_size_px"].tolist(),
            [3, 3, 5, 3],
        )
        self.assertEqual(
            catalog["_stage9_outer_window_size_px"].tolist(),
            [5, 7, 13, 7],
        )
        self.assertEqual(geometry["star_count"], 4)

    def test_same_star_fwhm_accepts_natural_size_and_rejects_small_or_large(self):
        source = _gaussian_field((128, 128), self.coordinates, sigma=1.35)
        reference = self._reference(source)

        natural = stage9_quality.assess_stage9_psf_closure(
            source,
            reference,
            self.cfg,
        )
        undersized = stage9_quality.assess_stage9_psf_closure(
            _gaussian_field((128, 128), self.coordinates, sigma=0.70),
            reference,
            self.cfg,
        )
        oversized = stage9_quality.assess_stage9_psf_closure(
            _gaussian_field((128, 128), self.coordinates, sigma=2.10),
            reference,
            self.cfg,
        )

        self.assertTrue(natural["accepted"], natural)
        self.assertAlmostEqual(
            natural["groups"]["all"]["fwhm_ratio_median"],
            1.0,
            places=6,
        )
        self.assertFalse(undersized["accepted"])
        self.assertLess(
            undersized["groups"]["all"]["fwhm_ratio_median"],
            0.90,
        )
        self.assertFalse(oversized["accepted"])
        self.assertGreater(
            oversized["groups"]["all"]["fwhm_ratio_median"],
            1.10,
        )

    def test_candidate_must_retain_minimum_same_star_fwhm_samples(self):
        source = _gaussian_field((128, 128), self.coordinates, sigma=1.35)
        reference = self._reference(source)
        partial = _gaussian_field(
            (128, 128),
            self.coordinates[:10],
            sigma=1.35,
        )

        result = stage9_quality.assess_stage9_psf_closure(
            partial,
            reference,
            self.cfg,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["accepted"])
        self.assertLess(
            result["candidate_sample_count"],
            result["minimum_sample_count"],
        )
        self.assertIn("star_psf_fwhm_sample_count", result["issues"][0])

    def test_insufficient_bright_reference_group_requires_review(self):
        source = _gaussian_field((128, 128), self.coordinates, sigma=1.35)
        reference = self._reference(source)
        keep = 16
        for key in (
            "_display_source_fwhm_px",
            "_display_source_halfmax_area_px",
            "_psf_valid_flags",
            "_psf_saturated_flags",
            "_peak_y",
            "_peak_x",
            "_weak_flags",
        ):
            reference[key] = np.asarray(reference[key])[:keep]
        reference["_weak_flags"] = np.asarray(
            [index < 13 for index in range(keep)],
            dtype=bool,
        )
        reference["psf_sample_count"] = keep

        result = stage9_quality.assess_stage9_psf_closure(
            source,
            reference,
            self.cfg,
        )

        self.assertEqual(result["schema"], "starun.stage9-psf-closure.v3")
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["accepted"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["groups"]["bright"]["status"], "not_assessed")
        self.assertEqual(result["groups"]["bright"]["reference_sample_count"], 3)

    def _closure_with_measured_ratio(
        self,
        ratio: float,
        *,
        source_area_available: bool = True,
    ) -> dict:
        source = _gaussian_field(
            (128, 128),
            self.coordinates,
            sigma=1.35,
        )
        reference = self._reference(source)
        source_fwhm = np.asarray(
            reference["_display_source_fwhm_px"],
            dtype=np.float64,
        )
        source_area = np.asarray(
            reference["_display_source_halfmax_area_px"],
            dtype=np.float64,
        )
        if not source_area_available:
            reference["_display_source_halfmax_area_px"] = np.full(
                source_area.shape,
                np.nan,
                dtype=np.float32,
            )
        measurements = [
            {
                "status": "ok",
                "fwhm_px": float(value * ratio),
                "half_max_area": float(area * ratio * ratio),
                "saturated": False,
                "offset_y": 0,
                "offset_x": 0,
            }
            for value, area in zip(source_fwhm, source_area)
        ]
        with patch.object(
            stage9_quality,
            "_measure_connected_halfmax_fwhm",
            side_effect=measurements,
        ):
            return stage9_quality.assess_stage9_psf_closure(
                source,
                reference,
                self.cfg,
            )

    def test_upper_boundary_can_be_accepted_within_measurement_uncertainty(self):
        result = self._closure_with_measured_ratio(1.101946)

        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["status"], "advisory")
        self.assertFalse(result["review_required"])
        self.assertTrue(result["uncertainty_exemption_used"])
        group = result["groups"]["all"]
        self.assertFalse(group["strict_accepted"])
        self.assertTrue(group["accepted_within_uncertainty"])
        self.assertEqual(group["decision"], "accepted_within_uncertainty")

    def test_ratio_beyond_maximum_uncertainty_is_rejected(self):
        result = self._closure_with_measured_ratio(1.13)

        self.assertFalse(result["accepted"])
        group = result["groups"]["all"]
        self.assertEqual(group["decision"], "rejected")
        self.assertLessEqual(
            group["measurement_uncertainty"]["u95_effective"],
            0.020,
        )

    def test_lower_boundary_uses_same_uncertainty_rule(self):
        result = self._closure_with_measured_ratio(0.9281)

        self.assertTrue(result["accepted"], result)
        self.assertTrue(
            result["groups"]["all"]["accepted_within_uncertainty"]
        )

    def test_missing_halfmax_area_keeps_auditable_floor_only_tolerance(self):
        result = self._closure_with_measured_ratio(
            1.1019,
            source_area_available=False,
        )

        uncertainty = result["groups"]["all"]["measurement_uncertainty"]
        self.assertTrue(result["accepted"], result)
        self.assertEqual(uncertainty["pixel_area_sample_count"], 0)
        self.assertEqual(uncertainty["pixel_se"], 0.0)
        self.assertAlmostEqual(uncertainty["u95_effective"], 0.002)

    def test_candidate_losing_measurable_bright_group_is_rejected(self):
        source = _gaussian_field((128, 128), self.coordinates, sigma=1.35)
        reference = self._reference(source)
        candidate = _gaussian_field(
            (128, 128),
            self.coordinates[:23],
            sigma=1.35,
        )

        result = stage9_quality.assess_stage9_psf_closure(
            candidate,
            reference,
            self.cfg,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["review_required"])
        self.assertEqual(result["groups"]["bright"]["status"], "insufficient")
        self.assertEqual(result["groups"]["bright"]["reference_sample_count"], 5)
        self.assertEqual(result["groups"]["bright"]["candidate_sample_count"], 3)

    def test_saturated_reference_stars_are_observed_not_fwhm_gated(self):
        source = _gaussian_field((128, 128), self.coordinates, sigma=1.35)
        reference = self._reference(source)
        reference["_weak_flags"] = np.asarray(
            [index < 16 for index in range(len(self.coordinates))],
            dtype=bool,
        )
        saturated = np.zeros(len(self.coordinates), dtype=bool)
        saturated[-4:] = True
        reference["_psf_saturated_flags"] = saturated

        result = stage9_quality.assess_stage9_psf_closure(
            source,
            reference,
            self.cfg,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reference_total_sample_count"], 25)
        self.assertEqual(result["reference_sample_count"], 21)
        self.assertEqual(result["groups"]["bright"]["reference_sample_count"], 5)
        self.assertEqual(result["groups"]["saturated"]["status"], "observed")
        self.assertEqual(result["groups"]["saturated"]["reference_sample_count"], 4)
        self.assertAlmostEqual(
            result["groups"]["saturated"]["halfmax_area_ratio_median"],
            1.0,
            places=6,
        )

    def test_disabled_psf_gate_records_partial_evidence_without_review(self):
        source = _gaussian_field((128, 128), self.coordinates, sigma=1.35)
        reference = self._reference(source)
        keep = 16
        for key in (
            "_display_source_fwhm_px",
            "_display_source_halfmax_area_px",
            "_psf_valid_flags",
            "_psf_saturated_flags",
            "_peak_y",
            "_peak_x",
            "_weak_flags",
        ):
            reference[key] = np.asarray(reference[key])[:keep]
        reference["_weak_flags"] = np.asarray(
            [index < 13 for index in range(keep)],
            dtype=bool,
        )
        self.cfg.stage9_psf_size_gate_enabled = False

        result = stage9_quality.assess_stage9_psf_closure(
            source,
            reference,
            self.cfg,
        )

        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["accepted"])
        self.assertFalse(result["review_required"])

    def test_display_reference_preserves_source_and_layer_coordinates(self):
        source = _gaussian_field((128, 128), self.coordinates, sigma=1.30)
        count = len(self.coordinates)
        source_y = np.asarray([item[0] for item in self.coordinates])
        source_x = np.asarray([item[1] for item in self.coordinates])
        catalog = {
            "status": "ok",
            "source_matched": True,
            "_source_peak_y": source_y,
            "_source_peak_x": source_x,
            "_peak_y": source_y + 1,
            "_peak_x": source_x,
            "_weak_flags": np.asarray(
                [index < 20 for index in range(count)],
                dtype=bool,
            ),
        }

        enriched = stage9_quality.enrich_star_reference_with_display_psf(
            catalog,
            source,
            self.cfg,
        )

        self.assertEqual(enriched["psf_reference_status"], "ready")
        self.assertGreaterEqual(enriched["psf_sample_count"], 16)
        np.testing.assert_array_equal(enriched["_source_peak_y"], source_y)
        np.testing.assert_array_equal(enriched["_peak_y"], source_y + 1)
        self.assertEqual(
            enriched["psf_confirmation_method"],
            "trusted_starmask_candidates_confirmed_in_mtf_O",
        )

    def test_normal_catalog_starts_from_starmask_and_confirms_in_mtf_o(self):
        starmask = _gaussian_field(
            (128, 128),
            self.coordinates,
            sigma=1.20,
            amplitude=0.45,
        )
        shifted_coordinates = [(y + 1, x) for y, x in self.coordinates]
        original_display = _gaussian_field(
            (128, 128),
            shifted_coordinates,
            sigma=1.35,
            amplitude=0.65,
        )

        catalog = stage9_quality.build_display_confirmed_starmask_catalog(
            starmask,
            original_display,
            self.cfg,
        )

        self.assertEqual(catalog["status"], "ok", catalog)
        self.assertTrue(catalog["source_matched"])
        self.assertEqual(
            catalog["method"],
            "trusted_starmask_candidates_confirmed_in_mtf_O",
        )
        self.assertEqual(catalog["source_detail_role"], "diagnostic_compatibility_only")
        self.assertTrue(
            all(
                attempt["percentile"] == 98.0
                for attempt in catalog["source_detail_attempts"]
            )
        )
        np.testing.assert_array_equal(
            np.asarray(catalog["_source_peak_y"]),
            np.asarray(catalog["_peak_y"]) + 1,
        )
        self.assertEqual(catalog["psf_reference_status"], "ready")

    def test_o_confirmation_sigma_retry_does_not_thin_by_percentile(self):
        starmask = np.zeros((3, 128, 128), dtype=np.float32)
        for y in range(4, 125, 4):
            for x in range(4, 125, 4):
                starmask[:, y, x] = (0.50, 0.45, 0.40)

        catalog = stage9_quality.build_display_confirmed_starmask_catalog(
            starmask,
            starmask,
            self.cfg,
        )

        self.assertEqual(catalog["status"], "rejected", catalog)
        self.assertTrue(catalog["fail_closed"])
        self.assertGreater(len(catalog["source_detail_attempts"]), 1)
        self.assertTrue(
            all(
                attempt["percentile"] == 98.0
                for attempt in catalog["source_detail_attempts"]
            )
        )
        self.assertGreater(
            catalog["source_detail_attempts"][-1]["reference_sigma"],
            catalog["source_detail_attempts"][0]["reference_sigma"],
        )
        self.assertIn("single_pixel_component_ratio", catalog["reason"])

    def test_fwhm_support_radius_and_source_wing_retry_are_bounded(self):
        shape = (48, 48)
        weak_core = np.zeros(shape, dtype=bool)
        bright_core = np.zeros(shape, dtype=bool)
        weak_core[24, 24] = True
        catalog = {
            "status": "ok",
            "source_matched": True,
            "_weak_core_mask": weak_core,
            "_bright_core_mask": bright_core,
            "_display_source_fwhm_px": np.asarray([4.0], dtype=np.float32),
            "_psf_valid_flags": np.asarray([True]),
            "_peak_y": np.asarray([24]),
            "_peak_x": np.asarray([24]),
            "_weak_flags": np.asarray([True]),
        }

        normal = stage9_quality.build_star_overlay_masks(
            catalog,
            strict=False,
            cfg=self.cfg,
        )[2]
        strict = stage9_quality.build_star_overlay_masks(
            catalog,
            strict=True,
            cfg=self.cfg,
        )[2]
        retried = stage9_quality.build_star_overlay_masks(
            catalog,
            strict=False,
            cfg=self.cfg,
            extra_pixels=1,
        )[2]

        self.assertTrue(normal[24, 27])
        self.assertFalse(normal[24, 28])
        self.assertTrue(strict[24, 26])
        self.assertFalse(strict[24, 27])
        self.assertTrue(retried[24, 28])
        self.assertFalse(retried[24, 29])

    def test_saturated_core_remains_measurable_without_gaussian_replacement(self):
        yy, xx = np.mgrid[:31, :31]
        saturated = np.clip(
            2.0 * np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / (2.0 * 1.8**2)),
            0.0,
            1.0,
        ).astype(np.float32)
        measured = stage9_quality._measure_connected_halfmax_fwhm(
            saturated,
            15,
            15,
        )
        self.assertEqual(measured["status"], "ok", measured)
        self.assertTrue(measured["saturated"])
        self.assertGreater(measured["fwhm_px"], 1.0)

    def test_unscreen_fwhm_may_not_regress_from_accepted_baseline(self):
        def candidate(mae: float, ratio: float) -> dict:
            return {
                "accepted": True,
                "reference_fidelity": {"status": "ok", "support_rgb_mae": mae},
                "star_color_validation": {
                    "metrics": {"median_chroma_error": 0.05}
                },
                "metrics": {
                    "weak_star_recovery_ratio": 0.90,
                    "star_recovery_ratio": 0.90,
                    "star_positive_delta_window_recovery_ratio": 0.90,
                    "star_wing_recovery_ratio": 0.90,
                    "star_psf_fwhm_ratio_all": ratio,
                },
                "advisories": [],
            }

        comparison = stage9_quality.compare_unscreen_candidate(
            candidate(0.10, 0.99),
            candidate(0.08, 0.90),
            self.cfg,
        )
        self.assertFalse(comparison["selected"])
        self.assertFalse(comparison["checks"]["fwhm_non_regression"])

    def test_zero_edit_operator_audit_reports_exact_raw_closure_and_alpha_invariant(self):
        starless = np.full((3, 12, 12), 0.12, dtype=np.float32)
        raw_stars = np.zeros_like(starless)
        raw_stars[:, 3:9, 3:9] = np.asarray(
            (0.25, 0.18, 0.12),
            dtype=np.float32,
        )[:, np.newaxis, np.newaxis]
        support = np.zeros((12, 12), dtype=bool)
        support[3:9, 3:9] = True
        original = stage9_quality.screen_blend(starless, raw_stars, 1.0)
        stabilized = raw_stars * 0.90

        linear = stage9_quality.assess_linear_decomposition_roundtrip(
            original,
            starless,
        )
        display = stage9_quality.assess_unscreen_operator_roundtrip(
            original,
            starless,
            stabilized,
            support,
            denominator_floor=0.08,
        )

        self.assertEqual(linear["status"], "ok")
        self.assertLess(linear["error"]["rgb_max"], 1e-6)
        self.assertEqual(display["status"], "ok")
        self.assertLess(
            display["raw_unscreen_screen_roundtrip"]["rgb_max"],
            1e-6,
        )
        self.assertGreater(
            display["stabilized_deviation_from_raw"]["rgb_mae"],
            0.0,
        )
        self.assertEqual(
            display["alpha_support_outside_change"]["rgb_max"],
            0.0,
        )

    def test_source_wing_feather_uses_matched_outer_ring_only(self):
        starless = np.full((3, 21, 21), 0.10, dtype=np.float32)
        source_stars = _gaussian_field(
            (21, 21),
            [(10, 10)],
            sigma=2.0,
            amplitude=0.55,
        )
        original = stage9_quality.screen_blend(starless, source_stars, 1.0)
        strict = np.zeros((21, 21), dtype=bool)
        strict[8:13, 8:13] = True
        expanded = np.zeros((21, 21), dtype=bool)
        expanded[6:15, 6:15] = True
        stabilized = np.where(strict[np.newaxis, ...], source_stars, 0.0)

        feathered, support, report = (
            stage9_quality.build_source_wing_feather_candidate(
                original,
                starless,
                stabilized,
                strict,
                expanded,
                self.cfg,
                feather_strength=0.90,
            )
        )

        self.assertEqual(report["status"], "ready", report)
        self.assertTrue(np.all(feathered[:, ~expanded] == 0.0))
        self.assertTrue(np.any(feathered[:, expanded & ~strict] > 0.0))
        self.assertTrue(np.all(support[strict]))
        self.assertFalse(report.get("recursive_dilation", False))

    def test_selective_source_wing_changes_only_undersized_star_outer_wing(self):
        shape = (61, 61)
        coordinates = [(18, 18), (42, 42)]
        source = _gaussian_field(
            shape,
            coordinates,
            sigma=1.70,
            amplitude=0.52,
        )
        starless = np.full((3, *shape), 0.08, dtype=np.float32)
        original = stage9_quality.screen_blend(starless, source, 1.0)
        candidate_stars = _gaussian_field(
            shape,
            [coordinates[0]],
            sigma=0.90,
            amplitude=0.52,
        )
        candidate_stars += _gaussian_field(
            shape,
            [coordinates[1]],
            sigma=1.80,
            amplitude=0.52,
        )
        candidate_display = stage9_quality.screen_blend(
            starless,
            candidate_stars,
            1.0,
        )
        # Build the reference in the same delivered display domain used by the
        # closure assessor.  The constant test base must not be re-normalized
        # as an integer-scale image, so use the explicit Screen result.
        source_peak = np.max(original, axis=0)
        fwhm = []
        areas = []
        for y, x in coordinates:
            measured = stage9_quality._measure_connected_halfmax_fwhm(
                source_peak,
                y,
                x,
            )
            self.assertEqual(measured["status"], "ok", measured)
            fwhm.append(measured["fwhm_px"])
            areas.append(measured["half_max_area"])
        reference = {
            "status": "ok",
            "source_matched": True,
            "_display_source_fwhm_px": np.asarray(fwhm, dtype=np.float32),
            "_display_source_halfmax_area_px": np.asarray(
                areas, dtype=np.float32
            ),
            "_psf_valid_flags": np.ones(2, dtype=bool),
            "_psf_saturated_flags": np.zeros(2, dtype=bool),
            "_peak_y": np.asarray([item[0] for item in coordinates]),
            "_peak_x": np.asarray([item[1] for item in coordinates]),
            "_weak_flags": np.asarray([True, False], dtype=bool),
            "_weak_core_mask": np.zeros(shape, dtype=bool),
            "_bright_core_mask": np.zeros(shape, dtype=bool),
        }
        reference["_weak_core_mask"][coordinates[0]] = True
        reference["_bright_core_mask"][coordinates[1]] = True
        current_support = np.zeros(shape, dtype=bool)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        for y, x in coordinates:
            current_support |= (yy - y) ** 2 + (xx - x) ** 2 <= 3**2

        selective, support, report = (
            stage9_quality.build_selective_source_wing_candidate(
                original,
                starless,
                candidate_stars,
                candidate_display,
                current_support,
                reference,
                self.cfg,
                remix_base=starless,
                screen_intensity=1.0,
                fwhm_ratio_target=1.08,
                feather_strength=1.15,
                extra_pixels=2,
            )
        )

        self.assertEqual(report["status"], "ready", report)
        self.assertTrue(report["changed"])
        self.assertEqual(report["selected_star_count"], 1)
        self.assertEqual(report["selected_weak_star_count"], 1)
        self.assertEqual(report["selected_bright_star_count"], 0)
        self.assertTrue(report["strict_core_immutable"])
        self.assertTrue(report["halfmax_core_immutable"])
        self.assertEqual(report["halfmax_ceiling_fraction"], 0.45)
        baseline_stars = np.where(
            current_support[np.newaxis, ...],
            candidate_stars,
            0.0,
        )
        delta = np.max(np.abs(selective - baseline_stars), axis=0)
        first_outer = ((yy - 18) ** 2 + (xx - 18) ** 2 >= 3**2) & (
            (yy - 18) ** 2 + (xx - 18) ** 2 <= 5**2
        )
        second_patch = (yy - 42) ** 2 + (xx - 42) ** 2 <= 5**2
        self.assertTrue(np.any(delta[first_outer] > 0.0))
        self.assertTrue(np.all(delta[second_patch] == 0.0))
        self.assertTrue(np.all(support[current_support]))
        delivered = stage9_quality.screen_blend(
            starless,
            selective,
            1.0,
            alpha_mask=support.astype(np.float32),
        )
        before_fwhm = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(candidate_display, axis=0),
            *coordinates[0],
        )["fwhm_px"]
        after_fwhm = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(delivered, axis=0),
            *coordinates[0],
        )["fwhm_px"]
        self.assertLessEqual(after_fwhm, before_fwhm)

    def test_selective_source_wing_skips_saturated_reference_star(self):
        shape = (31, 31)
        coordinate = (15, 15)
        source = _gaussian_field(
            shape,
            [coordinate],
            sigma=1.70,
            amplitude=0.52,
        )
        starless = np.full((3, *shape), 0.08, dtype=np.float32)
        original = stage9_quality.screen_blend(starless, source, 1.0)
        candidate_stars = _gaussian_field(
            shape,
            [coordinate],
            sigma=0.80,
            amplitude=0.52,
        )
        candidate_display = stage9_quality.screen_blend(
            starless,
            candidate_stars,
            1.0,
        )
        measured = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(original, axis=0),
            *coordinate,
        )
        reference = {
            "status": "ok",
            "source_matched": True,
            "_display_source_fwhm_px": np.asarray(
                [measured["fwhm_px"]], dtype=np.float32
            ),
            "_psf_valid_flags": np.ones(1, dtype=bool),
            "_psf_saturated_flags": np.ones(1, dtype=bool),
            "_peak_y": np.asarray([coordinate[0]]),
            "_peak_x": np.asarray([coordinate[1]]),
            "_weak_flags": np.ones(1, dtype=bool),
            "_weak_core_mask": np.zeros(shape, dtype=bool),
            "_bright_core_mask": np.zeros(shape, dtype=bool),
        }
        reference["_weak_core_mask"][coordinate] = True
        current_support = np.zeros(shape, dtype=bool)
        current_support[13:18, 13:18] = True

        selective, support, report = (
            stage9_quality.build_selective_source_wing_candidate(
                original,
                starless,
                candidate_stars,
                candidate_display,
                current_support,
                reference,
                self.cfg,
            )
        )

        self.assertIsNone(selective)
        self.assertIsNone(support)
        self.assertEqual(report["status"], "not_needed")
        self.assertEqual(report["selected_star_count"], 0)
        self.assertEqual(report["ordinary_reference_sample_count"], 0)

    def test_linked_autostretch_reference_recovers_low_wings_not_halfmax_core(self):
        shape = (81, 81)
        coordinate = (40, 40)
        # The matched-MTF source and the delivered candidate intentionally have
        # the same half-max core.  Only the linked-autostretch morphology
        # reference carries a wider 5%--10% wing.
        core = _gaussian_field(
            shape,
            [coordinate],
            sigma=1.35,
            amplitude=0.48,
        )
        visible_reference = _gaussian_field(
            shape,
            [coordinate],
            sigma=2.15,
            amplitude=0.72,
        )
        starless = np.full((3, *shape), 0.08, dtype=np.float32)
        original = stage9_quality.screen_blend(starless, core, 1.0)
        candidate_display = stage9_quality.screen_blend(starless, core, 1.0)
        source_measurement = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(original, axis=0),
            *coordinate,
        )
        reference = {
            "status": "ok",
            "source_matched": True,
            "_display_source_fwhm_px": np.asarray(
                [source_measurement["fwhm_px"]], dtype=np.float32
            ),
            "_display_source_halfmax_area_px": np.asarray(
                [source_measurement["half_max_area"]], dtype=np.float32
            ),
            "_psf_valid_flags": np.ones(1, dtype=bool),
            "_psf_saturated_flags": np.zeros(1, dtype=bool),
            "_peak_y": np.asarray([coordinate[0]]),
            "_peak_x": np.asarray([coordinate[1]]),
            "_weak_flags": np.ones(1, dtype=bool),
            "_weak_core_mask": np.zeros(shape, dtype=bool),
            "_bright_core_mask": np.zeros(shape, dtype=bool),
        }
        reference["_weak_core_mask"][coordinate] = True
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        current_support = (yy - coordinate[0]) ** 2 + (
            xx - coordinate[1]
        ) ** 2 <= 3**2

        before_halfmax = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(candidate_display, axis=0),
            *coordinate,
        )["fwhm_px"]
        before_visible = (
            stage9_quality._measure_connected_peak_fraction_footprint(
                np.max(candidate_display, axis=0),
                *coordinate,
                peak_fraction=0.10,
                patch_radius=10,
            )["equivalent_diameter_px"]
        )
        source_visible = (
            stage9_quality._measure_connected_peak_fraction_footprint(
                np.max(visible_reference, axis=0),
                *coordinate,
                peak_fraction=0.10,
                patch_radius=10,
            )["equivalent_diameter_px"]
        )
        selective, support, report = (
            stage9_quality.build_selective_source_wing_candidate(
                original,
                starless,
                core,
                candidate_display,
                current_support,
                reference,
                self.cfg,
                remix_base=starless,
                visible_wing_reference=visible_reference,
                screen_intensity=1.0,
            )
        )

        self.assertEqual(report["status"], "ready", report)
        self.assertTrue(report["changed"])
        self.assertEqual(
            report["selection_mode"],
            "same_source_linked_autostretch_5pct_10pct_25pct_footprint",
        )
        self.assertEqual(report["selected_star_count"], 1)
        self.assertEqual(report["selected_at_10pct_star_count"], 1)
        self.assertLess(report["before_visible_wing_ratio_median"], 1.0)
        self.assertGreater(report["visible_target_pixel_count"], 0)
        delivered = stage9_quality.screen_blend(
            starless,
            selective,
            1.0,
            alpha_mask=support.astype(np.float32),
        )
        after_halfmax = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(delivered, axis=0),
            *coordinate,
        )["fwhm_px"]
        after_visible = (
            stage9_quality._measure_connected_peak_fraction_footprint(
                np.max(delivered, axis=0),
                *coordinate,
                peak_fraction=0.10,
                patch_radius=10,
            )["equivalent_diameter_px"]
        )
        self.assertGreater(after_visible, before_visible)
        self.assertLessEqual(after_visible, source_visible * 1.05)
        self.assertLessEqual(after_halfmax, before_halfmax)
        audit = stage9_quality.assess_stage9_visible_wing_closure(
            delivered,
            visible_reference,
            reference,
            self.cfg,
        )
        self.assertEqual(audit["status"], "measured", audit)
        self.assertFalse(audit["hard_gate"])
        self.assertFalse(audit["scientific_photometry_claim"])
        self.assertGreater(audit["fractions"]["0.10"]["sample_count"], 0)

    def test_linked_autostretch_restores_missing_25pct_mid_wing_without_peak_or_fwhm_growth(self):
        shape = (81, 81)
        coordinate = (40, 40)
        visible_reference = _gaussian_field(
            shape,
            [coordinate],
            sigma=2.15,
            amplitude=0.72,
        )
        candidate_stars = _gaussian_field(
            shape,
            [coordinate],
            sigma=2.15,
            amplitude=0.72,
        )
        # Retain the peak/half-max core but remove the 25%--45% body.  A
        # 5%/10%-only closure can miss this visually-small-star failure mode.
        candidate_stars = np.where(
            candidate_stars >= 0.45 * float(np.max(candidate_stars)),
            candidate_stars,
            0.0,
        ).astype(np.float32)
        starless = np.full((3, *shape), 0.08, dtype=np.float32)
        original = stage9_quality.screen_blend(
            starless,
            visible_reference,
            1.0,
        )
        candidate_display = stage9_quality.screen_blend(
            starless,
            candidate_stars,
            1.0,
        )
        source_measurement = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(original, axis=0),
            *coordinate,
        )
        reference = {
            "status": "ok",
            "source_matched": True,
            "_display_source_fwhm_px": np.asarray(
                [source_measurement["fwhm_px"]], dtype=np.float32
            ),
            "_display_source_halfmax_area_px": np.asarray(
                [source_measurement["half_max_area"]], dtype=np.float32
            ),
            "_psf_valid_flags": np.ones(1, dtype=bool),
            "_psf_saturated_flags": np.zeros(1, dtype=bool),
            "_peak_y": np.asarray([coordinate[0]]),
            "_peak_x": np.asarray([coordinate[1]]),
            "_weak_flags": np.ones(1, dtype=bool),
            "_weak_core_mask": np.zeros(shape, dtype=bool),
            "_bright_core_mask": np.zeros(shape, dtype=bool),
        }
        reference["_weak_core_mask"][coordinate] = True
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        current_support = (yy - coordinate[0]) ** 2 + (
            xx - coordinate[1]
        ) ** 2 <= 3**2

        before_peak = float(np.max(candidate_display))
        before_fwhm = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(candidate_display, axis=0),
            *coordinate,
        )["fwhm_px"]
        before_mid = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(candidate_display, axis=0),
            *coordinate,
            peak_fraction=0.25,
            patch_radius=10,
        )["equivalent_diameter_px"]
        source_mid = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(visible_reference, axis=0),
            *coordinate,
            peak_fraction=0.25,
            patch_radius=10,
        )["equivalent_diameter_px"]
        selective, support, report = (
            stage9_quality.build_selective_source_wing_candidate(
                original,
                starless,
                candidate_stars,
                candidate_display,
                current_support,
                reference,
                self.cfg,
                remix_base=starless,
                visible_wing_reference=visible_reference,
                screen_intensity=1.0,
            )
        )

        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(report["selected_at_25pct_star_count"], 1)
        self.assertLess(report["before_visible_mid_ratio_median"], 1.0)
        delivered = stage9_quality.screen_blend(
            starless,
            selective,
            1.0,
            alpha_mask=support.astype(np.float32),
        )
        after_peak = float(np.max(delivered))
        after_fwhm = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(delivered, axis=0),
            *coordinate,
        )["fwhm_px"]
        after_mid = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(delivered, axis=0),
            *coordinate,
            peak_fraction=0.25,
            patch_radius=10,
        )["equivalent_diameter_px"]
        self.assertGreater(after_mid, before_mid)
        self.assertLessEqual(after_mid, source_mid * 1.05)
        self.assertEqual(after_fwhm, before_fwhm)
        self.assertAlmostEqual(after_peak, before_peak, places=6)

    def test_linked_autostretch_can_restore_only_5pct_outer_skirt(self):
        shape = (81, 81)
        coordinate = (40, 40)
        visible_reference = _gaussian_field(
            shape,
            [coordinate],
            sigma=2.15,
            amplitude=0.72,
        )
        candidate_stars = _gaussian_field(
            shape,
            [coordinate],
            sigma=2.35,
            amplitude=0.72,
        )
        # Keep a slightly wider 10% body but remove the lowest skirt.  This is
        # the field failure mode that a 10%-only selector cannot see.
        candidate_stars = np.where(
            candidate_stars >= 0.095 * float(np.max(candidate_stars)),
            candidate_stars,
            0.0,
        ).astype(np.float32)
        starless = np.full((3, *shape), 0.08, dtype=np.float32)
        original = stage9_quality.screen_blend(
            starless,
            visible_reference,
            1.0,
        )
        candidate_display = stage9_quality.screen_blend(
            starless,
            candidate_stars,
            1.0,
        )
        source_measurement = stage9_quality._measure_connected_halfmax_fwhm(
            np.max(original, axis=0),
            *coordinate,
        )
        reference = {
            "status": "ok",
            "source_matched": True,
            "_display_source_fwhm_px": np.asarray(
                [source_measurement["fwhm_px"]], dtype=np.float32
            ),
            "_display_source_halfmax_area_px": np.asarray(
                [source_measurement["half_max_area"]], dtype=np.float32
            ),
            "_psf_valid_flags": np.ones(1, dtype=bool),
            "_psf_saturated_flags": np.zeros(1, dtype=bool),
            "_peak_y": np.asarray([coordinate[0]]),
            "_peak_x": np.asarray([coordinate[1]]),
            "_weak_flags": np.ones(1, dtype=bool),
            "_weak_core_mask": np.zeros(shape, dtype=bool),
            "_bright_core_mask": np.zeros(shape, dtype=bool),
        }
        reference["_weak_core_mask"][coordinate] = True
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        current_support = (yy - coordinate[0]) ** 2 + (
            xx - coordinate[1]
        ) ** 2 <= 6**2

        before_10 = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(candidate_display, axis=0),
            *coordinate,
            peak_fraction=0.10,
            patch_radius=10,
        )["equivalent_diameter_px"]
        before_5 = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(candidate_display, axis=0),
            *coordinate,
            peak_fraction=0.05,
            patch_radius=10,
        )["equivalent_diameter_px"]
        source_10 = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(visible_reference, axis=0),
            *coordinate,
            peak_fraction=0.10,
            patch_radius=10,
        )["equivalent_diameter_px"]
        source_5 = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(visible_reference, axis=0),
            *coordinate,
            peak_fraction=0.05,
            patch_radius=10,
        )["equivalent_diameter_px"]
        self.assertGreaterEqual(before_10 / source_10, 1.03)
        self.assertLess(before_5 / source_5, 1.03)

        selective, support, report = (
            stage9_quality.build_selective_source_wing_candidate(
                original,
                starless,
                candidate_stars,
                candidate_display,
                current_support,
                reference,
                self.cfg,
                remix_base=starless,
                visible_wing_reference=visible_reference,
                screen_intensity=1.0,
            )
        )

        self.assertEqual(report["status"], "ready", report)
        self.assertEqual(report["selected_at_10pct_star_count"], 0)
        self.assertEqual(report["selected_at_5pct_only_star_count"], 1)
        delivered = stage9_quality.screen_blend(
            starless,
            selective,
            1.0,
            alpha_mask=support.astype(np.float32),
        )
        after_10 = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(delivered, axis=0),
            *coordinate,
            peak_fraction=0.10,
            patch_radius=10,
        )["equivalent_diameter_px"]
        after_5 = stage9_quality._measure_connected_peak_fraction_footprint(
            np.max(delivered, axis=0),
            *coordinate,
            peak_fraction=0.05,
            patch_radius=10,
        )["equivalent_diameter_px"]
        self.assertEqual(after_10, before_10)
        self.assertGreater(after_5, before_5)
        self.assertLessEqual(after_5, source_5 * 1.05)

    def _assert_stage5_completion_coordinate_domain(
        self,
        coordinate_domain,
        *,
        target_array_y,
        target_siril_y,
        existing_array_y,
        existing_siril_y,
    ):
        shape = (47, 53)
        target_x = 37
        existing_x_value = 11
        original = _gaussian_field(
            shape,
            [(existing_array_y, existing_x_value), (target_array_y, target_x)],
            sigma=2.0,
            amplitude=0.70,
        )
        trusted = original.copy()
        existing_y = np.asarray([existing_array_y], dtype=np.int32)
        existing_x = np.asarray([existing_x_value], dtype=np.int32)
        catalog = {
            "status": "ok",
            "reference_threshold": 1e-4,
            "_display_source_peak_y": existing_y,
            "_display_source_peak_x": existing_x,
        }
        stage5_stars = [
            {
                "index": 1,
                "x": float(target_x),
                "y": float(target_siril_y),
                "fwhm_geometry": 10.0,
                "saturated": True,
            },
            {
                "index": 2,
                "x": float(existing_x_value),
                "y": float(existing_siril_y),
                "fwhm_geometry": 10.0,
                "saturated": True,
            },
        ]

        completion = stage9_quality.build_stage5_bright_star_completion(
            stage5_stars,
            catalog,
            original,
            trusted,
            self.cfg,
            coordinate_domain=coordinate_domain,
        )

        self.assertEqual(completion["status"], "ready", completion)
        self.assertEqual(
            completion["schema"],
            "starun.stage9-stage5-bright-star-completion.v2",
        )
        self.assertTrue(completion["coordinate_contract"]["validated"])
        self.assertEqual(
            completion["coordinate_contract"]["array_coordinate_domain"],
            coordinate_domain,
        )
        self.assertEqual(completion["selected_star_count"], 1)
        self.assertEqual(completion["selected_saturated_count"], 1)
        self.assertEqual(
            completion["source_star_layer_counts"],
            {"ordinary": 0, "bright": 0, "saturated": 2},
        )
        self.assertEqual(
            completion["selected_star_layer_counts"],
            {"ordinary": 0, "bright": 0, "saturated": 1},
        )
        self.assertFalse(completion["ordinary_fwhm_gate_member"])
        mirror_y = shape[0] - 1 - target_array_y
        self.assertTrue(completion["_support_mask"][target_array_y, target_x])
        self.assertFalse(completion["_support_mask"][mirror_y, target_x])

        base = np.full_like(original, 0.05)
        completed, support, applied_report = (
            stage9_quality.apply_stage5_bright_star_completion(
                original,
                base,
                np.zeros_like(original),
                completion,
                self.cfg,
                remix_base=base,
                screen_intensity=1.0,
            )
        )
        self.assertEqual(applied_report["status"], "ready", applied_report)
        self.assertTrue(support[target_array_y, target_x])
        self.assertFalse(support[mirror_y, target_x])
        self.assertGreater(float(np.max(completed[:, target_array_y, target_x])), 0.0)
        self.assertEqual(float(np.max(completed[:, mirror_y, target_x])), 0.0)
        delivered = stage9_quality.screen_blend(
            base,
            completed,
            1.0,
            alpha_mask=support,
        )
        self.assertGreater(
            float(np.max(delivered[:, target_array_y, target_x] - base[:, target_array_y, target_x])),
            0.0,
        )
        self.assertEqual(
            float(np.max(delivered[:, mirror_y, target_x] - base[:, mirror_y, target_x])),
            0.0,
        )

    def test_stage5_saturated_star_uses_siril_pixel_buffer_coordinates_directly(self):
        self._assert_stage5_completion_coordinate_domain(
            "siril_pixel_buffer_bottom_up",
            target_array_y=8,
            target_siril_y=8,
            existing_array_y=31,
            existing_siril_y=31,
        )

    def test_stage5_saturated_star_converts_only_explicit_top_down_fits_array(self):
        self._assert_stage5_completion_coordinate_domain(
            "fits_array_top_down",
            target_array_y=38,
            target_siril_y=8,
            existing_array_y=15,
            existing_siril_y=31,
        )

    def test_bright_star_presence_observation_detects_restoration(self):
        base = np.full((3, 21, 21), 0.10, dtype=np.float32)
        stars = _gaussian_field(
            (21, 21),
            [(10, 10)],
            sigma=2.0,
            amplitude=0.50,
        )
        candidate = stage9_quality.screen_blend(base, stars, 1.0)
        report = {
            "status": "ready",
            "stars": [
                {"x": 10, "y": 10, "saturated": True},
            ],
        }

        observed = stage9_quality.assess_stage5_bright_star_presence(
            base,
            candidate,
            report,
        )

        self.assertEqual(observed["status"], "observed")
        self.assertEqual(observed["recovery_ratio"], 1.0)
        self.assertEqual(observed["saturated_recovery_ratio"], 1.0)
        self.assertFalse(observed["ordinary_fwhm_gate_member"])

    def test_bright_star_completion_caps_core_to_source_on_enhanced_base(self):
        original = np.full((3, 9, 9), 0.10, dtype=np.float32)
        starless = np.full_like(original, 0.10)
        remix_base = np.full_like(original, 0.40)
        original[:, 4, 4] = 0.90
        support = np.zeros((9, 9), dtype=bool)
        support[4, 4] = True
        completion = {
            "schema": "starun.stage9-stage5-bright-star-completion.v2",
            "status": "ready",
            "available": True,
            "coordinate_contract": {
                "schema": "starun.pixel-coordinate-contract.v1",
                "source_coordinate_domain": "siril_star_catalog_bottom_up",
                "array_coordinate_domain": "siril_pixel_buffer_bottom_up",
                "conversion": "y_array = y_siril",
                "validated": True,
            },
            "stars": [
                {"x": 4, "y": 4, "saturated": True},
            ],
            "_support_mask": support,
        }

        stars, effective_support, report = (
            stage9_quality.apply_stage5_bright_star_completion(
                original,
                starless,
                np.full_like(original, 0.80),
                completion,
                self.cfg,
                remix_base=remix_base,
                screen_intensity=1.05,
            )
        )

        self.assertEqual(report["status"], "ready", report)
        self.assertTrue(report["source_peak_cap_applied"])
        self.assertGreater(report["source_peak_cap_reduced_pixel_count"], 0)
        self.assertTrue(effective_support[4, 4])
        composed = stage9_quality.screen_blend(
            remix_base,
            stars,
            1.05,
            alpha_mask=effective_support,
        )
        self.assertLessEqual(float(np.max(composed[:, 4, 4])), 0.900001)
        self.assertTrue(np.all(composed[:, ~effective_support] == 0.40))

    def test_local_chroma_recovery_preserves_strict_core_and_psf_geometry(self):
        base = np.full((3, 9, 9), 0.10, dtype=np.float32)
        stars = np.zeros_like(base)
        stars[:, 4, 4] = np.asarray([0.70, 0.08, 0.03])
        stars[:, 4, 5] = np.asarray([0.32, 0.03, 0.01])
        stars[:, 2, 2] = np.asarray([0.04, 0.04, 0.04])
        support = np.zeros((9, 9), dtype=bool)
        support[4, 4:6] = True
        support[2, 2] = True
        strict_core = np.zeros_like(support)
        strict_core[4, 4] = True
        candidate = stage9_quality.screen_blend(
            base,
            stars,
            1.0,
            alpha_mask=support,
        )

        recovered, report = stage9_quality.build_local_chroma_recovery_layer(
            stars,
            base,
            candidate,
            support,
            strict_core,
            self.cfg,
            attenuation=0.50,
        )

        self.assertEqual(report["status"], "ready", report)
        self.assertTrue(report["strict_core_immutable"])
        self.assertEqual(report["strict_core_change_max"], 0.0)
        np.testing.assert_array_equal(recovered[:, 4, 4], stars[:, 4, 4])
        np.testing.assert_allclose(
            recovered[:, 4, 5],
            0.5 * stars[:, 4, 5],
            atol=1e-7,
        )
        np.testing.assert_array_equal(recovered[:, 2, 2], stars[:, 2, 2])

    def test_preflight_output_adequacy_routes_normal_and_strict_competition(self):
        shape = (9, 9)
        normal_mask = np.zeros(shape, dtype=bool)
        normal_mask[4, 4] = True
        strict_mask = np.zeros(shape, dtype=bool)
        strict_mask[4, 4:6] = True

        def calibration(mode, mask, actual_scale):
            return {
                "status": "ok",
                "support_status": "ok",
                "support_mode": mode,
                "_compact_support_mask": mask,
                "compact_support_coverage": float(np.mean(mask)),
                "predicted_change_ratio": 0.01,
                "predicted_change_ratio_limit": 0.30,
                "weak_star_retention": 1.0,
                "weak_star_retention_min": 0.80,
                "star_retention": 1.0,
                "output_profile": {
                    "status": "ok",
                    "accepted": True,
                    "hard_failed": False,
                    "actual": {
                        anchor: actual_scale
                        for anchor in ("faint", "mid", "bright", "peak")
                    },
                    "targets": {
                        anchor: 1.0
                        for anchor in ("faint", "mid", "bright", "peak")
                    },
                    "exceeded_anchors": [],
                },
            }

        normal = calibration("normal", normal_mask, 0.40)
        strict = calibration("strict_compact", strict_mask, 0.90)
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[normal, strict],
        ):
            report = stage9_quality.assess_starmask_support_preflight(
                np.zeros((3, *shape), dtype=np.float32),
                self.cfg,
                failure_action="auto_fallback",
            )

        self.assertEqual(report["route"], "dual_competition")
        self.assertEqual(
            report["reason_code"],
            "stage9_support_preflight_output_adequacy_dual",
        )
        self.assertTrue(
            report["candidates"]["normal"]["output_adequacy"][
                "psf_undersize_risk"
            ]
        )
        self.assertFalse(
            report["candidates"]["normal"]["output_adequacy"]["formal_gate"]
        )

    def test_plugin_output_adequacy_uses_inclusive_fifty_percent_boundary(self):
        shape = (12, 14)
        support = np.zeros(shape, dtype=bool)
        support[5:7, 6:8] = True

        def calibration(mode: str) -> dict:
            return {
                "status": "ok",
                "support_status": "ok",
                "support_mode": mode,
                "_compact_support_mask": support.copy(),
                "compact_support_coverage": 0.05,
                "predicted_change_ratio": 0.05,
                "predicted_change_ratio_limit": 0.30,
                "weak_star_retention": 1.0,
                "weak_star_retention_min": 0.80,
                "star_retention": 1.0,
                "output_profile": {
                    "status": "ok",
                    "accepted": True,
                    "hard_failed": False,
                    "actual": {
                        anchor: 0.80
                        for anchor in ("faint", "mid", "bright", "peak")
                    },
                    "targets": {
                        anchor: 1.0
                        for anchor in ("faint", "mid", "bright", "peak")
                    },
                    "exceeded_anchors": [],
                },
            }

        for actual, eligible in ((0.50, True), (0.499, False)):
            with self.subTest(actual=actual):
                measured_profile = {
                    "status": "ok",
                    "accepted": True,
                    "hard_failed": False,
                    "actual": {
                        anchor: actual
                        for anchor in ("faint", "mid", "bright", "peak")
                    },
                    "targets": {
                        anchor: 1.0
                        for anchor in ("faint", "mid", "bright", "peak")
                    },
                    "exceeded_anchors": [],
                }
                with (
                    patch.object(
                        stage9_quality,
                        "calibrate_starmask_asinh",
                        side_effect=[
                            calibration("normal"),
                            calibration("strict_compact"),
                        ],
                    ),
                    patch.object(
                        stage9_quality,
                        "measure_starmask_output_profile",
                        return_value=measured_profile,
                    ),
                ):
                    report = stage9_quality.assess_starmask_support_preflight(
                        np.zeros((3, *shape), dtype=np.float32),
                        self.cfg,
                        failure_action="auto_fallback",
                        plugin_stretched_stars=np.full(
                            (3, *shape),
                            0.2,
                            dtype=np.float32,
                        ),
                    )
                plugin = report["plugin_formal_eligibility"]
                self.assertEqual(plugin["eligible"], eligible)
                self.assertEqual(
                    report["selected_stretch_source"],
                    "plugin_stretched" if eligible else "builtin_calibrated",
                )
                if not eligible:
                    self.assertIn(
                        "stage9_plugin_starmask_output_inadequate",
                        str(report["fallback_reason"]),
                    )

    def test_catalog_visibility_accepts_both_explicit_coordinate_domains(self):
        shape = (72, 96)
        coordinates = [
            (y, x)
            for y in (8, 22, 39, 58)
            for x in (9, 31, 55, 82)
        ]
        image = _gaussian_field(
            shape,
            coordinates,
            sigma=0.75,
            amplitude=0.55,
        )
        count = len(coordinates)
        reference = {
            "status": "ok",
            "source_matched": True,
            "_source_peak_y": np.asarray(
                [item[0] for item in coordinates],
                dtype=np.int32,
            ),
            "_source_peak_x": np.asarray(
                [item[1] for item in coordinates],
                dtype=np.int32,
            ),
            "_weak_flags": np.asarray(
                [index < count // 2 for index in range(count)],
                dtype=bool,
            ),
            "_reference_local_contrast": np.full(
                count,
                0.30,
                dtype=np.float32,
            ),
            "_stage9_visibility_inner_window_size_px": np.full(
                count,
                3,
                dtype=np.int32,
            ),
            "_stage9_visibility_outer_window_size_px": np.full(
                count,
                7,
                dtype=np.int32,
            ),
        }
        bottom_up = stage9_quality.assess_catalog_star_visibility(
            image,
            reference,
            self.cfg,
            coordinate_domain="siril_pixel_buffer_bottom_up",
        )
        top_down = stage9_quality.assess_catalog_star_visibility(
            np.flip(image, axis=1),
            reference,
            self.cfg,
            coordinate_domain="display_array_top_down",
        )
        mirrored_wrong_domain = stage9_quality.assess_catalog_star_visibility(
            image,
            reference,
            self.cfg,
            coordinate_domain="display_array_top_down",
        )

        self.assertTrue(bottom_up["passed"], bottom_up)
        self.assertTrue(top_down["passed"], top_down)
        self.assertFalse(mirrored_wrong_domain["passed"])
        self.assertIn(
            "stage9_catalog_visibility_failed",
            mirrored_wrong_domain["reason_code"],
        )

    def test_legacy_v8_report_cannot_be_promoted_without_persisted_audit(self):
        legacy = stage9_quality.interpret_stage9_remix_quality_report(
            {
                "schema": "starun.stage9-remix-quality.v8",
                "formal_accepted": True,
            }
        )
        current = stage9_quality.interpret_stage9_remix_quality_report(
            {
                "schema": "starun.stage9-remix-quality.v10",
                "formal_accepted": True,
                "persisted_output_validation": {
                    "accepted": True,
                    "sep_crossmatch_accepted": True,
                },
                "sep_crossmatch": {
                    "schema": "starun.stage9-sep-crossmatch.v1",
                    "accepted": True,
                    "artifact_sha256": "0" * 64,
                },
            }
        )

        self.assertTrue(legacy["supported"])
        self.assertFalse(legacy["formal_accepted"])
        self.assertTrue(legacy["requires_review"])
        self.assertTrue(current["formal_accepted"])


if __name__ == "__main__":
    unittest.main()
