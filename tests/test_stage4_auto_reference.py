#!/usr/bin/env python3
"""Pure-algorithm tests for the Stage 4 non-physical fallback."""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from stage4_auto_reference import (  # noqa: E402
    AUTO_REFERENCE_SCHEMA,
    BACKGROUND_METHOD,
    STAR_ENSEMBLE_OBJECT_CAP,
    STAR_ENSEMBLE_METHOD,
    _background_holdout_comparison,
    _background_holdout_tail_reasons,
    _safety_gate,
    _select_star_records_for_analysis,
    _split_star_records,
    evaluate_auto_local_reference,
)
import stage4_auto_reference as auto_reference  # noqa: E402


def _broadband_fixture(*, stars: bool = True) -> np.ndarray:
    height = width = 256
    y, x = np.mgrid[:height, :width]
    rng = np.random.default_rng(7)
    background = (
        0.025
        + 0.002 * x / width
        + 0.001 * y / height
        + rng.normal(0.0, 0.0003, (height, width))
    )
    image = np.stack(
        [background + 0.012, background + 0.006, background], axis=0
    ).astype(np.float32)
    subject = np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2.0 * 30.0**2)) * 0.12
    image += np.stack([subject * 1.2, subject * 0.8, subject * 0.6])
    if stars:
        for cy in (28, 65, 102, 154, 205, 230):
            for cx in (25, 73, 121, 175, 225):
                star = (
                    np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 1.2**2))
                    * 0.45
                )
                image += np.stack([star * 1.08, star, star * 0.92])
    return image


def _custom_star_field(positions, color_ratios) -> np.ndarray:
    image = _broadband_fixture(stars=False)
    y, x = np.mgrid[:256, :256]
    for (cy, cx), (red_ratio, blue_ratio) in zip(positions, color_ratios):
        star = (
            np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 1.2**2))
            * 0.35
        )
        image += np.stack(
            [star * red_ratio, star, star * blue_ratio]
        )
    return image


class Stage4AutoReferenceTests(unittest.TestCase):
    def test_dense_star_records_are_spatially_capped_and_split_in_bounded_time(self) -> None:
        records = []
        width = 1000
        height = 800
        for index in range(10_000):
            x = float((index * 37) % width)
            y = float((index * 53) % height)
            records.append(
                {
                    "x": x,
                    "y": y,
                    "quadrant": int(x >= width / 2) + 2 * int(y >= height / 2),
                    "aspect_ratio": 1.0 + (index % 7) * 0.01,
                    "compactness": 0.8,
                    "saturation_fraction": 0.0,
                }
            )

        started = time.monotonic()
        selected, selection = _select_star_records_for_analysis(records)
        fit, holdout, split = _split_star_records(
            selected,
            validation_ratio=0.25,
        )
        elapsed = time.monotonic() - started
        second, second_selection = _select_star_records_for_analysis(records)

        self.assertLess(elapsed, 5.0)
        self.assertEqual(len(selected), STAR_ENSEMBLE_OBJECT_CAP)
        self.assertEqual(selection["valid_object_count"], 10_000)
        self.assertEqual(selection["analysis_object_count"], 256)
        self.assertTrue(selection["object_cap_applied"])
        self.assertEqual(
            [(item["x"], item["y"]) for item in selected],
            [(item["x"], item["y"]) for item in second],
        )
        self.assertEqual(selection, second_selection)
        self.assertEqual(split["status"], "ready")
        self.assertGreaterEqual(split["fit_quadrants"], 3)
        self.assertGreaterEqual(split["validation_quadrants"], 3)
        self.assertTrue({id(item) for item in fit}.isdisjoint({id(item) for item in holdout}))

    def test_background_tail_gate_rejects_median_masked_spatial_regression(self) -> None:
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                  [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
        before_values = np.asarray(
            [[0.030, 0.020, 0.018]] * 8,
            dtype=np.float64,
        )
        after_values = np.asarray(
            [
                [0.022, 0.020, 0.019],
                [0.022, 0.020, 0.019],
                [0.022, 0.020, 0.019],
                [0.022, 0.020, 0.019],
                [0.022, 0.020, 0.019],
                [0.022, 0.020, 0.019],
                [0.040, 0.020, 0.018],
                [0.041, 0.020, 0.018],
            ],
            dtype=np.float64,
        )
        before = {"points": points, "channel_medians": before_values.tolist()}
        after = {"points": points, "channel_medians": after_values.tolist()}

        comparison = _background_holdout_comparison(before, after)
        reasons = _background_holdout_tail_reasons(comparison)

        self.assertLess(
            float(np.median(np.ptp(after_values, axis=1))),
            float(np.median(np.ptp(before_values, axis=1))),
        )
        self.assertIn("heldout_background_p90_regressed", reasons)
        self.assertIn(
            "heldout_background_spatial_chroma_gradient_growth_exceeded",
            reasons,
        )

    def test_rejected_background_skips_star_analysis(self) -> None:
        with patch.object(
            auto_reference,
            "_find_star_ensemble",
            side_effect=AssertionError("star analysis must be skipped"),
        ):
            candidate, report = evaluate_auto_local_reference(
                _broadband_fixture(),
                config=SimpleNamespace(
                    stage4_auto_reference_target_chroma_drift_max=0.01,
                ),
            )

        self.assertIsNone(candidate)
        self.assertEqual(report["sampling"]["stars"]["status"], "not_run")
        skipped_selection = report["sampling"]["stars"]["selection"]
        self.assertEqual(skipped_selection["status"], "not_run")
        self.assertEqual(skipped_selection["valid_object_count"], 0)
        self.assertEqual(skipped_selection["quadrants"], 0)
        self.assertEqual(skipped_selection["rejection_counts"], {})
        star = report["candidates"][STAR_ENSEMBLE_METHOD]
        self.assertTrue(star["analysis_skipped"])
        self.assertIn(
            "background_neutralization_prerequisite_rejected",
            star["rejection_reasons"],
        )

    def test_background_candidate_is_deterministic_and_uses_disjoint_holdout(self) -> None:
        image = _broadband_fixture()

        first, first_report = evaluate_auto_local_reference(
            image,
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=False,
            ),
        )
        second, second_report = evaluate_auto_local_reference(
            image,
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=False,
            ),
        )

        self.assertIsNotNone(first)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_report["schema"], AUTO_REFERENCE_SCHEMA)
        self.assertEqual(first_report["selection"]["method"], BACKGROUND_METHOD)
        self.assertFalse(first_report["physical_color"]["accepted"])
        self.assertTrue(first_report["requires_review"])
        split = first_report["sampling"]["background"]["split"]
        fit = {tuple(point) for point in split["fit_points"]}
        validation = {tuple(point) for point in split["validation_points"]}
        self.assertTrue(fit)
        self.assertTrue(validation)
        self.assertTrue(fit.isdisjoint(validation))
        self.assertEqual(
            first_report["candidates"][BACKGROUND_METHOD]["channel_offsets"][-1],
            0.0,
        )
        json.dumps(first_report, allow_nan=False)
        self.assertEqual(first_report, second_report)

    def test_layout_and_integer_dtype_are_preserved(self) -> None:
        source = np.moveaxis(
            np.clip(np.rint(_broadband_fixture() * 65535.0), 0, 65535).astype(
                np.uint16
            ),
            0,
            -1,
        )

        candidate, report = evaluate_auto_local_reference(
            source,
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=False,
            ),
        )

        self.assertEqual(report["selection"]["method"], BACKGROUND_METHOD)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.shape, source.shape)
        self.assertEqual(candidate.dtype, source.dtype)
        self.assertTrue(np.all(candidate[..., 0] <= source[..., 0]))
        self.assertTrue(np.all(candidate[..., 1] <= source[..., 1]))
        self.assertTrue(np.array_equal(candidate[..., 2], source[..., 2]))

    def test_white_reference_rectangle_is_enabled_by_default(self) -> None:
        candidate, report = evaluate_auto_local_reference(
            _broadband_fixture(),
            config=SimpleNamespace(),
        )

        star = report["candidates"][STAR_ENSEMBLE_METHOD]
        self.assertIsNotNone(candidate)
        self.assertTrue(star["would_accept"], star)
        self.assertFalse(star["shadow_only"])
        self.assertTrue(star["pixels_authorized"])
        self.assertEqual(report["selection"]["method"], STAR_ENSEMBLE_METHOD)
        self.assertEqual(star["reference"], "single_rectangular_white_reference")
        self.assertTrue(all((1 / 1.10) <= gain <= 1.10 for gain in star["gains"]))
        for name in ("background", "white"):
            region = report["reference_regions"][name]
            self.assertEqual(region["status"], "selected")
            rectangle = region["rectangle"]
            self.assertGreater(rectangle["width"], 0)
            self.assertGreater(rectangle["height"], 0)
            self.assertGreaterEqual(rectangle["x"], 0)
            self.assertGreaterEqual(rectangle["y"], 0)
            self.assertLessEqual(rectangle["x"] + rectangle["width"], 256)
            self.assertLessEqual(rectangle["y"] + rectangle["height"], 256)
        self.assertFalse(
            report["reference_regions"]["white"]["color_values_used_for_selection"]
        )
        split = report["sampling"]["stars"]["split"]
        self.assertTrue(
            set(split["fit_indexes"]).isdisjoint(split["validation_indexes"])
        )

    def test_expert_switch_can_disable_white_reference_application(self) -> None:
        candidate, report = evaluate_auto_local_reference(
            _broadband_fixture(),
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=False,
            ),
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(report["selection"]["method"], BACKGROUND_METHOD)
        white = report["candidates"][STAR_ENSEMBLE_METHOD]
        self.assertTrue(white["shadow_only"])
        self.assertFalse(white["pixels_authorized"])

    def test_sparse_star_evidence_cannot_authorize_pseudo_white_reference(self) -> None:
        candidate, report = evaluate_auto_local_reference(
            _broadband_fixture(stars=False),
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=True,
            ),
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(report["selection"]["method"], BACKGROUND_METHOD)
        star = report["candidates"][STAR_ENSEMBLE_METHOD]
        self.assertFalse(star["accepted"])
        self.assertIn(
            "insufficient_independent_star_objects",
            star["rejection_reasons"],
        )

    def test_saturated_star_objects_are_excluded(self) -> None:
        image = _broadband_fixture()
        for cy in (28, 65, 102, 154, 205, 230):
            for cx in (25, 73, 121, 175, 225):
                image[:, cy - 1 : cy + 2, cx - 1 : cx + 2] = 1.0

        candidate, report = evaluate_auto_local_reference(
            image,
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=True,
            ),
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(report["selection"]["method"], BACKGROUND_METHOD)
        selection = report["sampling"]["stars"]["selection"]
        self.assertGreaterEqual(selection["rejection_counts"]["saturated"], 16)
        self.assertFalse(report["candidates"][STAR_ENSEMBLE_METHOD]["accepted"])

    def test_star_quadrant_and_color_dispersion_gates_reject_weak_evidence(self) -> None:
        left_positions = [
            (cy, cx)
            for cy in (35, 70, 105, 150, 185, 220)
            for cx in (20, 42, 64, 86)
        ]
        left_image = _custom_star_field(
            left_positions,
            [(1.08, 0.92)] * len(left_positions),
        )
        _candidate, left_report = evaluate_auto_local_reference(
            left_image,
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=True,
            ),
        )
        left_star = left_report["candidates"][STAR_ENSEMBLE_METHOD]
        self.assertGreaterEqual(
            left_report["sampling"]["stars"]["selection"]["valid_object_count"],
            16,
        )
        self.assertIn(
            "insufficient_star_quadrant_coverage",
            left_star["rejection_reasons"],
        )

        dispersed_positions = [
            (cy, cx)
            for cy in (25, 75, 125, 175, 225)
            for cx in (25, 75, 125, 175, 225)
        ]
        count = len(dispersed_positions)
        dispersed_ratios = [
            (
                float(np.exp(-0.4 + 0.8 * index / (count - 1))),
                float(np.exp(0.4 - 0.8 * index / (count - 1))),
            )
            for index in range(count)
        ]
        dispersed_image = _custom_star_field(
            dispersed_positions,
            dispersed_ratios,
        )
        _candidate, dispersed_report = evaluate_auto_local_reference(
            dispersed_image,
            config=SimpleNamespace(
                stage4_auto_reference_global_white_enabled=True,
            ),
        )
        dispersed_star = dispersed_report["candidates"][STAR_ENSEMBLE_METHOD]
        self.assertGreater(
            dispersed_report["sampling"]["stars"]["fit_ratio_statistics"][
                "maximum_ratio_mad"
            ],
            0.12,
        )
        self.assertIn(
            "star_color_ratio_dispersion_exceeded",
            dispersed_star["rejection_reasons"],
        )

    def test_independent_safety_gate_rejects_clip_structure_and_subject_drift(self) -> None:
        before = np.full((3, 64, 64), 0.20, dtype=np.float64)
        after = before.copy()
        checker = (np.indices((64, 64)).sum(axis=0) % 2) * 0.04 - 0.02
        after += checker[np.newaxis]
        after[:, 0, 0] = 1.0
        after[:, 0, 1] = 0.0
        subject_mask = np.zeros((64, 64), dtype=bool)
        subject_mask[20:44, 20:44] = True
        after[0, subject_mask] += 0.20

        reasons, metrics = _safety_gate(
            before,
            after,
            validation_points=((16, 16), (48, 16), (16, 48), (48, 48)),
            patch_radius=5,
            subject_mask=subject_mask,
            config=SimpleNamespace(
                stage4_auto_reference_highlight_clip_growth_max=0.0,
                stage4_auto_reference_black_clip_growth_max=0.0,
                stage4_auto_reference_gradient_growth_max=1.0,
                stage4_auto_reference_texture_growth_max=1.0,
                stage4_auto_reference_target_chroma_drift_max=0.01,
            ),
        )

        self.assertIn("highlight_clip_growth_exceeded", reasons)
        self.assertIn("black_clip_growth_exceeded", reasons)
        self.assertIn("heldout_gradient_growth_exceeded", reasons)
        self.assertIn("heldout_texture_growth_exceeded", reasons)
        self.assertIn("subject_chromaticity_drift_exceeded", reasons)
        self.assertTrue(metrics["shape_preserved"])
        self.assertTrue(metrics["finite"])

    def test_subject_drift_limit_rejects_an_otherwise_valid_background_candidate(self) -> None:
        candidate, report = evaluate_auto_local_reference(
            _broadband_fixture(),
            config=SimpleNamespace(
                stage4_auto_reference_target_chroma_drift_max=0.01,
                stage4_auto_reference_global_white_enabled=True,
            ),
        )

        self.assertIsNone(candidate)
        background = report["candidates"][BACKGROUND_METHOD]
        self.assertIn(
            "subject_chromaticity_drift_exceeded",
            background["rejection_reasons"],
        )
        self.assertIn(
            "background_neutralization_prerequisite_rejected",
            report["candidates"][STAR_ENSEMBLE_METHOD]["rejection_reasons"],
        )

    def test_uniform_full_frame_signal_is_preserved(self) -> None:
        image = np.empty((3, 256, 256), dtype=np.float32)
        image[0] = 0.08
        image[1] = 0.06
        image[2] = 0.04

        candidate, report = evaluate_auto_local_reference(
            image,
            config=SimpleNamespace(),
        )

        self.assertIsNone(candidate)
        self.assertEqual(report["selection"]["method"], "PRESERVE_INPUT")
        reasons = report["candidates"].get(BACKGROUND_METHOD, {}).get(
            "rejection_reasons", []
        )
        self.assertTrue(
            "insufficient_safe_background_coverage" in reasons
            or "insufficient_background_subject_separation" in reasons
        )
        star = report["candidates"][STAR_ENSEMBLE_METHOD]
        self.assertTrue(star["analysis_skipped"])
        self.assertEqual(report["sampling"]["stars"]["status"], "not_run")
        json.dumps(report, allow_nan=False)

    def test_non_broadband_non_linear_and_non_finite_inputs_are_rejected(self) -> None:
        image = _broadband_fixture()
        for kwargs, reason in (
            ({"channel_kind": "narrowband_composite"}, "unsupported_channel_semantics"),
            ({"linear": False}, "nonlinear_input"),
        ):
            with self.subTest(reason=reason):
                candidate, report = evaluate_auto_local_reference(
                    image,
                    config=SimpleNamespace(),
                    **kwargs,
                )
                self.assertIsNone(candidate)
                self.assertEqual(report["eligibility"]["reason"], reason)

        invalid = image.copy()
        invalid[0, 0, 0] = np.nan
        candidate, report = evaluate_auto_local_reference(
            invalid,
            config=SimpleNamespace(),
        )
        self.assertIsNone(candidate)
        self.assertEqual(report["eligibility"]["reason"], "non_finite_input")


if __name__ == "__main__":
    unittest.main()
