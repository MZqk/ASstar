#!/usr/bin/env python3
"""Smoke-level integration tests for all ten pipeline stage entry points."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _install_sirilpy_stub() -> None:
    if "sirilpy" in sys.modules:
        return
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class SirilError(Exception):
        pass

    class CommandError(SirilError):
        pass

    class DataError(SirilError):
        pass

    sirilpy.SirilInterface = object
    exceptions.SirilError = SirilError
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions


_install_sirilpy_stub()

from models import (  # noqa: E402
    ImageFeatures,
    PipelineConfig,
    PipelineStage,
)
import stage9_quality  # noqa: E402
import stage5_handoff  # noqa: E402
import stage7_quality  # noqa: E402
import stage7_stretch_metrics  # noqa: E402
import stage7_repair  # noqa: E402
import starmask_cleanup  # noqa: E402
from stages import (  # noqa: E402
    stage10_export,
    stage1_preparation,
    stage2_view_correction,
    stage3_background_extraction,
    stage4_color_calibration,
    stage5_linear_denoise,
    stage7_stretching,
    stage6_star_separation,
    stage8_nebula_enhancement,
    stage9_star_remixing,
)


def _synthetic_stage9_mixed_star_field():
    """Return a deterministic sparse-bright/dense-faint Stage 9 star field."""
    rng = np.random.default_rng(20260717)
    height = width = 192
    yy, xx = np.mgrid[:height, :width]
    noise = np.clip(
        rng.normal(0.0, 0.000035, (height, width)),
        0.0,
        None,
    )
    diffuse = 0.00003 * np.sin(xx / 13.0) ** 2 * np.sin(yy / 17.0) ** 2
    floor = noise + diffuse
    stars = np.stack([floor, floor * 0.9, floor * 0.8]).astype(np.float32)
    coordinates = [
        (center_y, center_x)
        for center_y in range(10, 183, 16)
        for center_x in range(10, 183, 16)
    ]
    rng.shuffle(coordinates)
    weak_levels = np.concatenate(
        [
            rng.uniform(0.008, 0.011, 80),
            rng.uniform(0.035, 0.040, 20),
        ]
    )
    for (center_y, center_x), amplitude in zip(
        coordinates[:100],
        weak_levels,
    ):
        profile = np.exp(
            -(
                (xx - center_x) ** 2
                + (yy - center_y) ** 2
            )
            / (2.0 * 1.05**2)
        )
        stars += np.stack(
            [profile * amplitude, profile * amplitude * 0.85, profile * amplitude * 0.70]
        ).astype(np.float32)

    extreme_coordinates = coordinates[100:104]
    for (center_y, center_x), amplitude in zip(
        extreme_coordinates,
        (0.65, 0.72, 0.82, 0.90),
    ):
        profile = np.exp(
            -(
                (xx - center_x) ** 2
                + (yy - center_y) ** 2
            )
            / (2.0 * 1.35**2)
        )
        stars += np.stack(
            [profile * amplitude, profile * amplitude * 0.78, profile * amplitude * 0.58]
        ).astype(np.float32)
    return stars, extreme_coordinates


def _stage9_source_image(stars):
    """Embed the synthetic stars in a smooth full-image nebula background."""
    height, width = stars.shape[-2:]
    yy, xx = np.mgrid[:height, :width]
    smooth = (
        0.025
        + 0.012 * (xx / max(width - 1, 1))
        + 0.018 * np.exp(-((xx - 104) ** 2 + (yy - 88) ** 2) / (2.0 * 42.0**2))
    ).astype(np.float32)
    return np.clip(stars + np.stack([smooth, smooth * 0.92, smooth * 0.84]), 0.0, 1.0)


class _Log:
    def stage_start(self, _name: str) -> None:
        return

    def stage_end(self, _name: str | None = None) -> float:
        return 0.01

    def info(self, _message: str) -> None:
        return

    def warn(self, _message: str) -> None:
        return

    def error(self, _message: str) -> None:
        return

    def debug(self, _message: str) -> None:
        return


class PipelineStageTests(unittest.TestCase):
    def test_stage9_manual_like_defaults_are_ordered_and_support_safe(self):
        cfg = PipelineConfig()

        self.assertEqual(cfg.star_intensity, 1.05)
        self.assertEqual(cfg.stage9_psf_fwhm_ratio_min, 0.93)
        self.assertEqual(cfg.stage9_psf_fwhm_ratio_max, 1.10)
        self.assertEqual(cfg.stage9_psf_review_fwhm_ratio_max, 1.65)
        self.assertEqual(cfg.stage9_psf_recovery_target_min, 0.97)
        self.assertEqual(cfg.stage9_psf_recovery_target_max, 1.05)
        self.assertEqual(cfg.stage9_psf_support_radius_max, 6)
        self.assertEqual(cfg.stage9_psf_support_retry_pixels, 2)
        self.assertEqual(cfg.stage9_source_star_detail_percentile, 98.0)
        self.assertEqual(
            (
                cfg.stage9_starmask_faint_target,
                cfg.stage9_starmask_mid_target,
                cfg.stage9_starmask_bright_target,
                cfg.stage9_starmask_peak_target,
            ),
            (0.26, 0.50, 0.75, 0.90),
        )
        self.assertEqual(cfg.stage9_weak_star_screen_intensity_min, 0.55)
        self.assertTrue(cfg.stage9_starmask_chroma_regularization_enabled)
        self.assertEqual(cfg.stage9_starmask_faint_chroma_max, 0.35)
        self.assertEqual(cfg.stage9_starmask_bright_chroma_max, 0.60)
        self.assertEqual(cfg.stage9_chromatic_addition_ratio_max, 0.003)
        self.assertEqual(cfg.stage9_star_support_ratio_max, 0.12)
        self.assertEqual(cfg.stage9_source_component_density_max, 2500.0)
        self.assertEqual(cfg.stage9_source_single_pixel_ratio_max, 0.20)
        self.assertEqual(cfg.stage9_hollow_structure_delta_min, 0.05)
        self.assertEqual(cfg.stage9_new_hollow_structure_area_max, 64)
        self.assertEqual(cfg.stage9_local_component_area_max, 256)
        self.assertEqual(cfg.stage9_local_single_pixel_ratio_max, 0.20)
        self.assertEqual(cfg.stage9_local_cyan_blue_component_area_max, 64)
        self.assertEqual(cfg.stage9_core_color_jump_component_area_max, 64)
        self.assertTrue(cfg.stage9_unscreen_candidate_enabled)
        self.assertEqual(cfg.stage9_unscreen_denominator_floor, 0.08)
        self.assertEqual(cfg.stage9_unscreen_reliable_support_min, 0.80)
        self.assertEqual(cfg.stage9_unscreen_peak_max, 0.95)
        self.assertEqual(
            cfg.stage9_unscreen_roundtrip_relative_improvement_min,
            0.10,
        )
        self.assertEqual(
            cfg.stage9_unscreen_roundtrip_absolute_improvement_min,
            0.005,
        )
        self.assertEqual(cfg.stage7_starmask_diffuse_residual_ratio_max, 0.08)
        self.assertEqual(cfg.stage7_quality_advisory_multiplier, 2.0)
        self.assertEqual(cfg.stage7_9_quality_advisory_multiplier, 1.5)
        self.assertEqual(cfg.stage7_starmask_diffuse_uncertainty_abs, 0.0005)
        self.assertEqual(
            cfg.stage7_starmask_diffuse_borderline_star_intensity_scale,
            0.70,
        )
        self.assertEqual(cfg.stage10_final_denoise_strength, 0.28)
        self.assertEqual(cfg.stage10_star_protection_coverage_max, 0.35)
        self.assertEqual(
            cfg.stage10_large_galaxy_local_patch_variance_max,
            0.00032,
        )
        self.assertEqual(cfg.stage10_stage9_local_color_risk_strength, 1.0)

    def test_stage7_9_quality_gate_uses_exact_fifty_percent_advisory_band(self):
        cfg = PipelineConfig(stage7_9_quality_advisory_multiplier=1.5)

        upper_boundary = stage7_quality.stage7_9_upper_quality_gate(
            cfg,
            value=1.5,
            accepted_limit=1.0,
        )
        upper_hard = stage7_quality.stage7_9_upper_quality_gate(
            cfg,
            value=1.500001,
            accepted_limit=1.0,
        )
        lower_boundary = stage7_quality.stage7_9_lower_quality_gate(
            cfg,
            value=2.0 / 3.0,
            accepted_limit=1.0,
        )
        lower_hard = stage7_quality.stage7_9_lower_quality_gate(
            cfg,
            value=(2.0 / 3.0) - 0.000001,
            accepted_limit=1.0,
        )

        self.assertEqual(upper_boundary["status"], "advisory")
        self.assertEqual(upper_hard["status"], "hard_failed")
        self.assertEqual(lower_boundary["status"], "advisory")
        self.assertEqual(lower_hard["status"], "hard_failed")

    def test_stage7_vectorized_linked_mtf_matches_scalar_reference(self):
        values = np.linspace(0.0, 1.0, 257, dtype=np.float32)
        mapped = stage7_stretch_metrics.apply_linked_mtf(
            values,
            0.012,
            0.08,
            1.0,
        )
        expected = np.asarray(
            [
                stage7_stretch_metrics.linked_mtf_sample(
                    value,
                    0.012,
                    0.08,
                    1.0,
                )
                for value in values
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(mapped, expected, atol=1e-6)

    def test_formal_stage_labels_are_unique_and_contiguous(self):
        labels = [stage.label for stage in PipelineStage]

        self.assertEqual(len(labels), 10)
        self.assertEqual(len(set(labels)), 10)
        for number, label in enumerate(labels, start=1):
            self.assertTrue(label.startswith(f"阶段 {number}:"))

    def test_stage9_screen_blend_preserves_highlight_headroom(self):
        base = np.full((3, 4, 4), 0.80, dtype=np.float32)
        stars = np.full((3, 4, 4), 0.80, dtype=np.float32)

        mixed = stage9_quality.screen_blend(base, stars, 1.0)

        self.assertAlmostEqual(float(mixed[0, 0, 0]), 0.96, places=5)
        self.assertLess(float(mixed.max()), 1.0)

    def test_stage9_unscreen_is_exact_inverse_screen_in_matched_domain(self):
        rng = np.random.default_rng(20260812)
        background = rng.uniform(0.01, 0.85, (3, 32, 32)).astype(np.float32)
        stars = rng.uniform(0.0, 0.80, (3, 32, 32)).astype(np.float32)
        original = stage9_quality.screen_blend(background, stars, 1.0)

        recovered = stage9_quality.unscreen_layer(
            original,
            background,
            denominator_floor=0.08,
        )
        closed = stage9_quality.screen_blend(background, recovered, 1.0)

        self.assertLessEqual(float(np.max(np.abs(closed - original))), 1e-6)
        dark = stage9_quality.unscreen_layer(
            np.asarray([[0.11]], dtype=np.float32),
            np.asarray([[0.10]], dtype=np.float32),
        )
        bright = stage9_quality.unscreen_layer(
            np.asarray([[0.91]], dtype=np.float32),
            np.asarray([[0.90]], dtype=np.float32),
        )
        self.assertAlmostEqual(float(dark[0, 0]), 0.011111, places=5)
        self.assertAlmostEqual(float(bright[0, 0]), 0.10, places=5)

    def test_stage9_unscreen_stabilization_preserves_trusted_rgb_ratios(self):
        cfg = PipelineConfig(stage9_unscreen_reliable_support_min=0.50)
        background = np.full((3, 12, 12), 0.55, dtype=np.float32)
        trusted = np.zeros_like(background)
        trusted[:, 4:8, 4:8] = np.asarray([0.20, 0.10, 0.05])[:, None, None]
        raw_star = np.zeros_like(background)
        raw_star[:, 4:8, 4:8] = np.asarray([0.60, 0.20, 0.50])[:, None, None]
        original = stage9_quality.screen_blend(background, raw_star, 1.0)
        support = np.zeros((12, 12), dtype=bool)
        support[4:8, 4:8] = True

        stabilized, report = stage9_quality.build_chroma_stable_unscreen_layer(
            original,
            background,
            trusted,
            support,
            cfg,
        )

        self.assertIsNotNone(stabilized)
        self.assertEqual(report["status"], "ready")
        output = np.asarray(stabilized)
        ratio = output[:, 5, 5] / output[0, 5, 5]
        np.testing.assert_allclose(ratio, [1.0, 0.5, 0.25], atol=1e-6)
        self.assertGreater(float(output[0, 5, 5]), float(trusted[0, 5, 5]))
        self.assertEqual(float(np.max(output[:, ~support])), 0.0)

    def test_stage9_unscreen_reliability_fails_closed(self):
        cfg = PipelineConfig(stage9_unscreen_reliable_support_min=0.80)
        background = np.full((3, 10, 10), 0.94, dtype=np.float32)
        trusted = np.full((3, 10, 10), 0.02, dtype=np.float32)
        support = np.ones((10, 10), dtype=bool)
        original = stage9_quality.screen_blend(background, trusted, 1.0)

        layer, report = stage9_quality.build_chroma_stable_unscreen_layer(
            original,
            background,
            trusted,
            support,
            cfg,
        )

        self.assertIsNone(layer)
        self.assertEqual(
            report["reason_code"],
            "stage9_unscreen_reliability_insufficient",
        )
        self.assertEqual(report["reliable_support_ratio"], 0.0)

    def test_stage9_unscreen_competition_requires_non_regressing_gain(self):
        cfg = PipelineConfig()

        def candidate(mae, chroma=0.10, recovery=0.90, advisories=()):
            return {
                "accepted": True,
                "reference_fidelity": {
                    "status": "ok",
                    "support_rgb_mae": mae,
                },
                "star_color_validation": {
                    "metrics": {"median_chroma_error": chroma}
                },
                "metrics": {
                    "weak_star_recovery_ratio": recovery,
                    "star_recovery_ratio": recovery,
                    "star_positive_delta_window_recovery_ratio": recovery,
                    "star_wing_recovery_ratio": recovery,
                },
                "advisories": list(advisories),
            }

        baseline = candidate(0.050)
        boundary_winner = candidate(0.045)
        comparison = stage9_quality.compare_unscreen_candidate(
            baseline,
            boundary_winner,
            cfg,
        )
        self.assertTrue(comparison["selected"], comparison)

        chroma_regression = candidate(0.040, chroma=0.121)
        comparison = stage9_quality.compare_unscreen_candidate(
            baseline,
            chroma_regression,
            cfg,
        )
        self.assertFalse(comparison["selected"])
        self.assertIn("chroma_non_regression", comparison["failed_checks"])

        advisory_regression = candidate(
            0.040,
            advisories=("background_lift 0.02>0.01",),
        )
        comparison = stage9_quality.compare_unscreen_candidate(
            baseline,
            advisory_regression,
            cfg,
        )
        self.assertFalse(comparison["selected"])
        self.assertIn(
            "no_new_advisory_category",
            comparison["failed_checks"],
        )

    def test_stage9_regularizes_faint_single_channel_artifacts(self):
        source = np.zeros((3, 17, 17), dtype=np.float32)
        mapped = np.zeros_like(source)
        source[:, 8, 8] = (0.80, 0.64, 0.48)
        mapped[:, 8, 8] = (0.80, 0.64, 0.48)
        source[:, 8, 10] = (0.0, 0.0, 0.0005)
        mapped[:, 8, 10] = (0.0, 0.0, 0.25)

        regularized, diagnostics = (
            stage9_quality._regularize_amplified_starmask_chroma(
                source,
                mapped,
                faint_input=0.001,
                bright_input=0.20,
                faint_chroma_max=0.35,
                bright_chroma_max=0.60,
            )
        )

        faint_rgb = regularized[:, 8, 10]
        faint_saturation = (float(faint_rgb.max()) - float(faint_rgb.min())) / float(
            faint_rgb.max()
        )
        self.assertLessEqual(faint_saturation, 0.351)
        np.testing.assert_allclose(
            regularized[:, 8, 8],
            mapped[:, 8, 8],
            rtol=0.0,
            atol=1e-6,
        )
        self.assertGreater(diagnostics["regularized_pixel_ratio"], 0.0)

    def test_stage9_gate_warns_for_moderate_chromatic_additions(self):
        cfg = PipelineConfig()
        base = np.full((3, 64, 64), 0.04, dtype=np.float32)
        candidate = np.array(base, copy=True)
        candidate[2, 8:12, 8:12] += 0.40

        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="chromatic_artifact",
            formula="screen",
        )

        self.assertGreater(
            report["metrics"]["chromatic_star_addition_ratio"],
            report["limits"]["chromatic_star_addition_ratio"],
        )
        self.assertFalse(
            any(
                issue.startswith("chromatic_star_addition_ratio")
                for issue in report["issues"]
            )
        )
        self.assertTrue(
            any(
                advisory.startswith("chromatic_star_addition_ratio")
                for advisory in report["advisories"]
            )
        )
        self.assertEqual(
            report["quality_gates"]["chromatic_star_addition_ratio"]["status"],
            "advisory",
        )

        hard_candidate = np.array(base, copy=True)
        hard_candidate[2, 8:13, 8:13] += 0.40
        hard_report = stage9_quality.assess_remix(
            base,
            hard_candidate,
            cfg,
            attempt="hard_chromatic_artifact",
            formula="screen",
        )
        self.assertEqual(
            hard_report["quality_gates"]["chromatic_star_addition_ratio"][
                "status"
            ],
            "hard_failed",
        )
        self.assertTrue(
            any(
                issue.startswith("chromatic_star_addition_ratio")
                for issue in hard_report["issues"]
            )
        )

    def test_stage9_local_gates_reject_small_frame_artifacts(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[:192, :192]
        base_gray = 0.04 + 0.09 * np.exp(
            -((xx - 96) ** 2 + (yy - 96) ** 2) / (2.0 * 30.0**2)
        )
        base = np.stack([base_gray, base_gray, base_gray]).astype(np.float32)

        cases = {}

        nonstellar = np.array(base, copy=True)
        nonstellar[:, 16:20, 20:90] += 0.08
        cases["local_connected_component_max_area"] = nonstellar

        cyan_blue = np.array(base, copy=True)
        cyan_blue[0, 10:18, 10:20] += 0.01
        cyan_blue[1, 10:18, 10:20] += 0.08
        cyan_blue[2, 10:18, 10:20] += 0.10
        cases["local_cyan_blue_component_max_area"] = cyan_blue

        single_pixels = np.array(base, copy=True)
        for y in range(6, 190, 16):
            for x in range(6, 190, 16):
                single_pixels[:, y, x] += 0.08
        cases["local_single_pixel_component_ratio"] = single_pixels

        core_jump = np.array(base, copy=True)
        core_jump[0, 92:100, 92:101] += 0.10
        cases["core_color_jump_component_max_area"] = core_jump

        expected_gate_status = {
            "local_connected_component_max_area": "advisory",
            "local_cyan_blue_component_max_area": "advisory",
            "local_single_pixel_component_ratio": "hard_failed",
            "core_color_jump_component_max_area": "advisory",
        }
        for expected_metric, candidate in cases.items():
            with self.subTest(metric=expected_metric):
                report = stage9_quality.assess_remix(
                    base,
                    candidate,
                    cfg,
                    attempt=expected_metric,
                    formula="screen",
                )

                self.assertFalse(report["accepted"])
                self.assertGreater(
                    report["metrics"][expected_metric],
                    report["limits"][expected_metric],
                )
                self.assertEqual(
                    report["quality_gates"][expected_metric]["status"],
                    expected_gate_status[expected_metric],
                )
                messages = (
                    report["issues"]
                    if expected_gate_status[expected_metric] == "hard_failed"
                    else report["advisories"]
                )
                self.assertTrue(
                    any(message.startswith(expected_metric) for message in messages),
                    messages,
                )

        nonstellar_report = stage9_quality.assess_remix(
            base,
            nonstellar,
            cfg,
            attempt="nonstellar_shape_check",
            formula="screen",
        )
        self.assertGreater(
            nonstellar_report["metrics"]["local_nonstellar_shape_component_count"],
            0,
        )
        self.assertTrue(
            any(
                issue.startswith("local_nonstellar_shape_component_count")
                for issue in nonstellar_report["issues"]
            )
        )

        self.assertLess(
            stage9_quality.assess_remix(
                base,
                cyan_blue,
                cfg,
                attempt="cyan_global_ratio_check",
                formula="screen",
            )["metrics"]["chromatic_star_addition_ratio"],
            cfg.stage9_chromatic_addition_ratio_max,
        )

    def test_stage9_local_cyan_gate_exempts_source_confirmed_star_support(self):
        cfg = PipelineConfig()
        base = np.full((3, 96, 96), 0.04, dtype=np.float32)
        candidate = np.array(base, copy=True)
        candidate[0, 10:18, 10:20] += 0.01
        candidate[1, 10:18, 10:20] += 0.08
        candidate[2, 10:18, 10:20] += 0.10
        positive_change = np.maximum(candidate - base, 0.0)
        confirmed_support = np.zeros((96, 96), dtype=bool)
        confirmed_support[10:18, 10:20] = True

        unmatched = stage9_quality._stage9_local_quality_metrics(
            base,
            candidate,
            positive_change,
            cfg,
        )
        confirmed = stage9_quality._stage9_local_quality_metrics(
            base,
            candidate,
            positive_change,
            cfg,
            confirmed_star_support=confirmed_support,
        )

        self.assertGreater(
            unmatched["metrics"]["local_cyan_blue_component_max_area"],
            unmatched["limits"]["local_cyan_blue_component_max_area"],
        )
        self.assertEqual(
            confirmed["metrics"]["local_cyan_blue_component_max_area"],
            0,
        )
        self.assertEqual(
            confirmed["metrics"]["local_cyan_blue_component_max_area_raw"],
            80,
        )
        self.assertGreater(
            confirmed["metrics"]["local_cyan_blue_confirmed_star_pixel_ratio"],
            0.0,
        )

    def test_stage9_local_shape_gates_exempt_source_confirmed_star_support(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[:192, :192]
        base_gray = 0.04 + 0.09 * np.exp(
            -((xx - 96) ** 2 + (yy - 96) ** 2) / (2.0 * 30.0**2)
        )
        base = np.stack([base_gray, base_gray, base_gray]).astype(np.float32)
        candidate = np.array(base, copy=True)
        candidate[0, 86:106, 86:106] += 0.12
        positive_change = np.maximum(candidate - base, 0.0)
        confirmed_support = np.zeros((192, 192), dtype=bool)
        confirmed_support[86:106, 86:106] = True

        unmatched = stage9_quality._stage9_local_quality_metrics(
            base,
            candidate,
            positive_change,
            cfg,
        )
        confirmed = stage9_quality._stage9_local_quality_metrics(
            base,
            candidate,
            positive_change,
            cfg,
            confirmed_star_support=confirmed_support,
        )

        self.assertGreater(
            unmatched["metrics"]["local_connected_component_max_area"],
            unmatched["limits"]["local_connected_component_max_area"],
        )
        self.assertGreater(
            unmatched["metrics"]["core_color_jump_component_max_area"],
            unmatched["limits"]["core_color_jump_component_max_area"],
        )
        self.assertEqual(
            confirmed["metrics"]["local_connected_component_max_area"],
            0,
        )
        self.assertEqual(
            confirmed["metrics"]["core_color_jump_component_max_area"],
            0,
        )
        self.assertEqual(
            confirmed["metrics"]["local_connected_component_max_area_raw"],
            400,
        )
        self.assertEqual(
            confirmed["metrics"]["core_color_jump_component_max_area_raw"],
            400,
        )

    def test_stage9_alpha_screen_keeps_starless_bottom_outside_star_support(self):
        base = np.full((3, 5, 5), 0.20, dtype=np.float32)
        stars = np.full((3, 5, 5), 0.50, dtype=np.float32)
        support = np.zeros((5, 5), dtype=bool)
        support[2, 2] = True

        mixed = stage9_quality.screen_blend(
            base,
            stars,
            1.0,
            alpha_mask=support,
        )

        np.testing.assert_allclose(mixed[:, ~support], base[:, ~support])
        self.assertAlmostEqual(float(mixed[0, 2, 2]), 0.60, places=5)

    def test_stage9_gate_rejects_diffuse_starmask_background_lift(self):
        cfg = PipelineConfig()
        base = np.full((3, 20, 20), 0.05, dtype=np.float32)
        candidate = np.full(
            (3, 20, 20),
            0.075,
            dtype=np.float32,
        )

        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="diffuse_mask",
            formula="screen",
        )

        self.assertFalse(report["accepted"])
        self.assertTrue(
            any("background_lift" in issue for issue in report["issues"])
        )

    def test_stage9_gate_rejects_background_mottling_growth(self):
        cfg = PipelineConfig()
        cfg.stage9_background_mottling_growth_max = 1.20
        yy, xx = np.mgrid[:96, :96]
        base_gray = 0.05 + 0.002 * np.sin(xx / 19.0)
        broad_mottling = 0.009 * (1.0 + np.sin(xx / 5.0) * np.sin(yy / 7.0))
        base = np.stack([base_gray, base_gray, base_gray]).astype(np.float32)
        candidate_gray = base_gray + broad_mottling
        candidate = np.stack(
            [candidate_gray, candidate_gray, candidate_gray]
        ).astype(np.float32)

        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="mottled_background",
            formula="screen",
        )

        self.assertFalse(report["accepted"])
        self.assertGreater(
            report["metrics"]["background_mottling_growth"],
            report["limits"]["background_mottling_growth"],
        )
        self.assertTrue(
            any("background_mottling_growth" in issue for issue in report["issues"])
        )

    def test_stage9_mottling_gate_ignores_low_absolute_compact_star_growth(self):
        cfg = PipelineConfig()
        cfg.stage9_psf_min_sample_count = 4
        rng = np.random.default_rng(42)
        yy, xx = np.mgrid[:128, :128]
        base_gray = 0.04 + rng.normal(0.0, 0.0005, (128, 128))
        base = np.stack([base_gray, base_gray, base_gray]).astype(np.float32)
        stars = np.zeros_like(base)
        for center_y, center_x, amplitude in (
            (16, 17, 0.12),
            (24, 54, 0.16),
            (38, 96, 0.20),
            (55, 27, 0.24),
            (68, 73, 0.30),
            (84, 111, 0.34),
            (102, 45, 0.38),
            (112, 91, 0.42),
        ):
            profile = amplitude * np.exp(
                -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * 1.2**2)
            )
            stars += np.stack([profile, profile * 0.9, profile * 0.8]).astype(
                np.float32
            )
        candidate = stage9_quality.screen_blend(base, stars, 1.0)

        catalog = stage9_quality.build_star_reference_catalog(stars, cfg)
        catalog["_weak_flags"] = np.asarray(
            [True, True, True, True, False, False, False, False],
            dtype=bool,
        )
        stage9_quality.enrich_star_reference_with_display_psf(
            catalog,
            candidate,
            cfg,
        )
        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="compact_stars",
            formula="screen",
            star_reference=catalog,
        )

        self.assertTrue(report["accepted"])
        self.assertTrue(
            report["metrics"]["background_mottling_low_absolute_exempted"]
        )
        self.assertFalse(
            any("background_mottling_growth" in issue for issue in report["issues"])
        )

    def test_stage9_low_absolute_mottling_exemption_rejects_broad_changes(self):
        cfg = PipelineConfig()
        cfg.stage9_background_mottling_growth_max = 1.20
        yy, xx = np.mgrid[:128, :128]
        base_gray = 0.04 + 0.0002 * np.sin(xx / 23.0)
        broad_change = 0.004 * np.sin(xx / 4.0) * np.sin(yy / 5.0)
        base = np.stack([base_gray, base_gray, base_gray]).astype(np.float32)
        candidate_gray = base_gray + broad_change
        candidate = np.stack(
            [candidate_gray, candidate_gray, candidate_gray]
        ).astype(np.float32)

        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="broad_low_absolute_change",
            formula="screen",
        )

        self.assertGreater(
            report["metrics"]["changed_pixel_ratio"],
            report["limits"][
                "background_mottling_low_absolute_changed_pixel_ratio_max"
            ],
        )
        self.assertFalse(
            report["metrics"]["background_mottling_low_absolute_exempted"]
        )
        self.assertTrue(
            any("background_mottling_growth" in issue for issue in report["issues"])
        )

    def test_stage9_adaptive_starmask_stretch_uses_more_gain_for_faint_stars(self):
        cfg = PipelineConfig(stage9_starmask_faint_target=0.22)
        yy, xx = np.mgrid[-32:32, -32:32]
        profile = np.exp(-(xx * xx + yy * yy) / (2.0 * 1.4**2))
        faint = np.stack([profile * 0.035, profile * 0.030, profile * 0.025]).astype(np.float32)
        bright = np.stack([profile * 0.75, profile * 0.64, profile * 0.53]).astype(np.float32)

        faint_plan = stage9_quality.calibrate_starmask_asinh(faint, cfg)
        bright_plan = stage9_quality.calibrate_starmask_asinh(bright, cfg)

        self.assertEqual(faint_plan["status"], "ok")
        self.assertEqual(bright_plan["status"], "ok")
        faint_direct = stage9_quality._solve_asinh_stretch(
            0.035,
            0.001,
            0.22,
            1000.0,
        )
        bright_direct = stage9_quality._solve_asinh_stretch(
            0.75,
            0.001,
            0.22,
            1000.0,
        )
        self.assertGreater(faint_direct, bright_direct)
        self.assertGreater(
            faint_plan["derived_asinh_stretch"],
            bright_plan["derived_asinh_stretch"],
        )
        for plan in (faint_plan, bright_plan):
            self.assertTrue(plan["output_profile"]["accepted"])
            for name, value in plan["output_profile"]["actual"].items():
                self.assertLessEqual(
                    value,
                    plan["output_profile"]["targets"][name] + 1e-4,
                )
        self.assertLessEqual(faint_plan["predicted_peak"], cfg.stage9_starmask_peak_target + 1e-4)

    def test_stage9_mixed_field_preserves_weak_stars_and_compresses_bright_stars(self):
        cfg = PipelineConfig()
        stars, extreme_coordinates = _synthetic_stage9_mixed_star_field()
        source = _stage9_source_image(stars)
        catalog = stage9_quality.build_star_reference_catalog(
            stars,
            cfg,
            source_image=source,
        )
        component_peaks = np.asarray(
            catalog["_component_peaks"],
            dtype=np.float32,
        )
        bright_indices = np.argsort(component_peaks)[-4:]
        catalog["_weak_flags"] = np.ones(component_peaks.size, dtype=bool)
        catalog["_weak_flags"][bright_indices] = False
        catalog["_reference_local_contrast"] = np.full(
            component_peaks.size,
            0.01,
            dtype=np.float32,
        )
        plan = stage9_quality.calibrate_starmask_asinh(
            stars,
            cfg,
            include_support_mask=True,
            reference_catalog=catalog,
        )
        stretched = stage9_quality.apply_calibrated_starmask(stars, plan)

        self.assertEqual(catalog["status"], "ok")
        self.assertTrue(catalog["source_matched"])
        self.assertTrue(catalog["mixed_star_field"])
        self.assertGreaterEqual(catalog["weak_component_count"], 80)
        self.assertGreaterEqual(catalog["bright_component_count"], 3)
        self.assertGreaterEqual(
            catalog["bright_to_weak_peak_ratio"],
            cfg.stage9_mixed_star_peak_ratio_min,
        )
        self.assertEqual(plan["method"], "monotonic_multi_anchor_star_curve")
        self.assertTrue(plan["multi_anchor_curve"])
        self.assertTrue(plan["chroma_regularization_enabled"])
        self.assertGreaterEqual(
            plan["weak_star_retention"],
            cfg.stage9_compact_weak_star_retention_min,
        )
        self.assertGreaterEqual(plan["star_retention"], 0.99)
        self.assertLessEqual(plan["predicted_faint"], 0.2601)
        self.assertLessEqual(plan["predicted_mid"], 0.5001)
        self.assertLessEqual(plan["predicted_bright"], 0.7501)
        self.assertLessEqual(plan["predicted_peak"], 0.9001)
        self.assertTrue(plan["output_profile"]["accepted"])
        self.assertLessEqual(plan["output_target_scale"], 1.0)

        stretched_norm = stage9_quality._normalized(stretched)
        stretched_peak = stage9_quality._pixel_peak(stretched_norm)
        bright_flags = np.asarray(catalog["_weak_flags"], dtype=bool) == 0
        bright_y = np.asarray(catalog["_peak_y"])[bright_flags]
        bright_x = np.asarray(catalog["_peak_x"])[bright_flags]
        self.assertLessEqual(float(np.max(stretched_peak[bright_y, bright_x])), 0.9001)

        component_peaks = np.asarray(catalog["_component_peaks"], dtype=np.float32)
        output_peaks = stretched_peak[
            np.asarray(catalog["_peak_y"]),
            np.asarray(catalog["_peak_x"]),
        ]
        q80, q90, q997 = np.percentile(component_peaks, (80.0, 90.0, 99.7))
        rank_groups = (
            component_peaks <= q80,
            (component_peaks > q80) & (component_peaks <= q90),
            (component_peaks > q90) & (component_peaks <= q997),
            component_peaks > q997,
        )

        ordered_medians = [
            float(np.median(output_peaks[group]))
            for group in rank_groups
            if np.any(group)
        ]
        self.assertTrue(
            all(
                later > earlier
                for earlier, later in zip(ordered_medians, ordered_medians[1:])
            ),
            ordered_medians,
        )

        center_y, center_x = extreme_coordinates[-1]
        gains = stretched_norm[:, center_y, center_x] / np.maximum(
            stars[:, center_y, center_x],
            1e-12,
        )
        self.assertLess(float(np.max(gains) - np.min(gains)), 1e-4)

        base = np.full_like(stars, 0.04)
        weak_mask, bright_mask, overlay_mask = stage9_quality.build_star_overlay_masks(
            catalog,
            strict=False,
        )
        candidate = stage9_quality.screen_blend(
            base,
            stretched,
            0.75,
            alpha_mask=overlay_mask,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            weak_intensity=1.0,
        )
        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="mixed_screen",
            formula="screen",
            star_reference=catalog,
            star_overlay_mask=overlay_mask,
        )

        self.assertTrue(report["accepted"], report["issues"])
        self.assertEqual(report["metrics"]["local_quality_status"], "ok")
        self.assertEqual(report["metrics"]["local_color_risk_score"], 0.0)
        self.assertGreaterEqual(
            report["metrics"]["weak_star_recovery_ratio"],
            cfg.stage9_weak_star_recovery_ratio_min,
        )
        self.assertGreaterEqual(
            report["metrics"]["star_recovery_ratio"],
            cfg.stage9_star_recovery_ratio_min,
        )
        self.assertLessEqual(
            report["metrics"]["star_support_ratio"],
            cfg.stage9_star_support_ratio_max,
        )
        self.assertLessEqual(
            report["metrics"]["unmatched_changed_ratio"],
            cfg.stage9_unmatched_changed_ratio_max,
        )
        self.assertGreaterEqual(
            report["metrics"]["star_positive_delta_window_recovery_ratio"],
            cfg.stage9_star_positive_delta_window_recovery_ratio_min,
        )
        self.assertGreaterEqual(
            report["metrics"]["star_positive_delta_window_restored_count"],
            0,
        )
        self.assertGreaterEqual(
            report["metrics"]["star_wing_recovery_ratio"],
            cfg.stage9_star_wing_recovery_ratio_min,
        )
        for metric_name in (
            "highlight_clip_ratio_after",
            "highlight_clip_growth",
            "bright_pixel_growth",
            "background_lift",
            "changed_pixel_ratio",
            "chromatic_star_addition_ratio",
            "darkening_ratio",
        ):
            self.assertLessEqual(
                report["metrics"][metric_name],
                report["limits"][metric_name],
            )

        advisory_cfg = PipelineConfig(stage9_star_support_ratio_max=0.08)
        advisory_report = stage9_quality.assess_remix(
            base,
            candidate,
            advisory_cfg,
            attempt="mixed_screen_advisory",
            formula="screen",
            star_reference=catalog,
            star_overlay_mask=overlay_mask,
        )
        self.assertTrue(advisory_report["accepted"], advisory_report["issues"])
        self.assertEqual(
            advisory_report["quality_gates"]["star_support_ratio"]["status"],
            "advisory",
        )
        self.assertTrue(
            any(
                advisory.startswith("star_support_ratio")
                for advisory in advisory_report["advisories"]
            )
        )

    def test_stage9_multi_anchor_uniformly_scales_targets_to_change_budget(self):
        cfg = PipelineConfig(
            stage9_starmask_predicted_change_ratio_max=0.05,
            stage9_changed_pixel_ratio_max=0.05,
        )
        stars, _extreme_coordinates = _synthetic_stage9_mixed_star_field()
        catalog = stage9_quality.build_star_reference_catalog(
            stars,
            cfg,
            source_image=_stage9_source_image(stars),
        )

        plan = stage9_quality.calibrate_starmask_asinh(
            stars,
            cfg,
            include_support_mask=True,
            reference_catalog=catalog,
        )

        self.assertEqual(plan["status"], "ok")
        self.assertTrue(plan["multi_anchor_curve"])
        self.assertTrue(plan["coverage_limited"])
        self.assertLess(plan["output_target_scale"], 1.0)
        self.assertLessEqual(
            plan["predicted_change_ratio"],
            plan["predicted_change_ratio_limit"] + 1e-12,
        )
        actual = plan["output_profile"]["actual"]
        self.assertLess(actual["faint"], actual["mid"])
        self.assertLess(actual["mid"], actual["bright"])
        self.assertLess(actual["bright"], actual["peak"])

    def test_stage9_compatibility_stretch_is_only_a_clamped_proposal(self):
        cfg = PipelineConfig(
            stage9_starmask_adaptive_stretch_enabled=False,
            stage9_starmask_asinh_stretch=3.0,
        )
        yy, xx = np.mgrid[:128, :128]
        stars = np.zeros((3, 128, 128), dtype=np.float32)
        for cy, cx, amplitude in (
            (18, 21, 0.04),
            (37, 92, 0.07),
            (69, 60, 0.12),
            (99, 31, 0.20),
            (101, 104, 0.35),
        ):
            profile = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.5**2)
            )
            stars += np.stack(
                [profile, profile * 0.85, profile * 0.70]
            ).astype(np.float32) * amplitude

        plan = stage9_quality.calibrate_starmask_asinh(stars, cfg)

        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["configured_stretch_proposal"], 3.0)
        self.assertLessEqual(plan["stretch"], 3.0)
        self.assertTrue(plan["output_profile"]["accepted"])
        for name, value in plan["output_profile"]["actual"].items():
            self.assertLessEqual(
                value,
                plan["output_profile"]["targets"][name] + 1e-4,
            )

    def test_stage9_rejects_when_minimum_asinh_exceeds_output_targets(self):
        cfg = PipelineConfig()
        stars = np.zeros((3, 128, 128), dtype=np.float32)
        for cy, cx in ((20, 20), (35, 90), (65, 60), (95, 30), (100, 103)):
            stars[:, cy - 1 : cy + 2, cx - 1 : cx + 2] = np.asarray(
                [0.50, 0.45, 0.40],
                dtype=np.float32,
            )[:, np.newaxis, np.newaxis]

        plan = stage9_quality.calibrate_starmask_asinh(stars, cfg)

        self.assertEqual(plan["status"], "rejected")
        self.assertFalse(plan["light_stretch_usable"])
        self.assertEqual(
            plan["reason_code"],
            "stage9_starmask_output_target_exceeded_at_minimum_stretch",
        )
        self.assertEqual(plan["stretch"], 1.10)
        self.assertTrue(plan["output_profile"]["hard_failed"])

    def test_stage9_recovery_gate_rejects_missing_stars_and_missing_catalog(self):
        cfg = PipelineConfig()
        stars, _ = _synthetic_stage9_mixed_star_field()
        catalog = stage9_quality.build_star_reference_catalog(
            stars,
            cfg,
            source_image=_stage9_source_image(stars),
        )
        base = np.full_like(stars, 0.04)

        missing_stars = stage9_quality.assess_remix(
            base,
            base.copy(),
            cfg,
            attempt="missing_stars",
            formula="screen",
            star_reference=catalog,
        )
        self.assertFalse(missing_stars["accepted"])
        self.assertEqual(missing_stars["metrics"]["weak_star_recovery_ratio"], 0.0)
        self.assertEqual(missing_stars["metrics"]["star_recovery_ratio"], 0.0)
        self.assertTrue(
            any(
                issue.startswith("weak_star_recovery_ratio")
                for issue in missing_stars["issues"]
            )
        )

        missing_catalog = stage9_quality.assess_remix(
            base,
            stage9_quality.screen_blend(base, stars, 1.0),
            cfg,
            attempt="missing_catalog",
            formula="screen",
        )
        self.assertFalse(missing_catalog["accepted"])
        self.assertIn(
            "star_recovery_metrics_unavailable",
            " ".join(missing_catalog["issues"]),
        )

        cfg.stage9_quality_gate_enabled = False
        disabled_gate = stage9_quality.assess_remix(
            base,
            base.copy(),
            cfg,
            attempt="disabled_gate",
            formula="screen",
            star_reference=catalog,
        )
        self.assertFalse(disabled_gate["accepted"])
        self.assertFalse(disabled_gate["gate_enabled"])
        self.assertEqual(disabled_gate["metrics"]["star_recovery_ratio"], 0.0)
        self.assertIn(
            "catalog_star_visibility_unavailable",
            " ".join(disabled_gate["issues"]),
        )

    def test_stage9_gate_rejects_new_closed_hollow_structure(self):
        cfg = PipelineConfig()
        stars, extreme_coordinates = _synthetic_stage9_mixed_star_field()
        catalog = stage9_quality.build_star_reference_catalog(
            stars,
            cfg,
            source_image=_stage9_source_image(stars),
        )
        plan = stage9_quality.calibrate_starmask_asinh(
            stars,
            cfg,
            include_support_mask=True,
            reference_catalog=catalog,
        )
        stretched = stage9_quality.apply_calibrated_starmask(stars, plan)
        weak_mask, bright_mask, overlay_mask = stage9_quality.build_star_overlay_masks(
            catalog,
            strict=False,
        )
        base = np.full_like(stars, 0.04)
        candidate = stage9_quality.screen_blend(
            base,
            stretched,
            0.75,
            alpha_mask=overlay_mask,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            weak_intensity=1.0,
        )

        center_y, center_x = extreme_coordinates[0]
        yy, xx = np.mgrid[: stars.shape[1], : stars.shape[2]]
        radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        artificial_ring = (radius >= 10.0) & (radius <= 13.0)
        candidate[:, artificial_ring] = np.maximum(
            candidate[:, artificial_ring],
            base[:, artificial_ring] + 0.25,
        )
        ring_support = overlay_mask | artificial_ring

        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="artificial_ring",
            formula="screen",
            star_reference=catalog,
            star_overlay_mask=ring_support,
        )

        self.assertFalse(report["accepted"])
        self.assertGreater(
            report["metrics"]["new_hollow_structure_max_area"],
            report["limits"]["new_hollow_structure_max_area"],
        )
        self.assertIn(
            "new_hollow_structure_max_area",
            " ".join(report["issues"]),
        )

    def test_stage9_source_catalog_fails_closed_on_dense_single_pixel_noise(self):
        cfg = PipelineConfig()
        rng = np.random.default_rng(20260717)
        noisy_plane = np.clip(
            0.02 + rng.normal(0.0, 0.003, (256, 256)),
            0.0,
            1.0,
        ).astype(np.float32)
        noisy_source = np.stack(
            [noisy_plane, noisy_plane * 0.93, noisy_plane * 0.86]
        )

        catalog = stage9_quality.build_star_reference_catalog(
            noisy_source,
            cfg,
            source_image=noisy_source,
        )

        self.assertEqual(catalog["status"], "rejected")
        self.assertTrue(catalog["fail_closed"])
        self.assertLess(
            catalog["source_component_density_per_megapixel"],
            catalog["source_component_density_max"],
        )
        self.assertGreater(
            catalog["source_raw_component_density_per_megapixel"],
            catalog["source_component_density_max"],
        )
        self.assertGreater(
            catalog["source_single_pixel_component_ratio"],
            catalog["source_single_pixel_component_ratio_max"],
        )
        self.assertIn("source_star_catalog_contamination_risk", catalog["reason"])
        self.assertEqual(catalog["source_detail_percentile"], 98.0)
        self.assertGreater(len(catalog["source_detail_attempts"]), 1)
        self.assertGreater(
            catalog["source_detail_attempts"][-1]["reference_sigma"],
            catalog["source_detail_attempts"][0]["reference_sigma"],
        )

    def test_stage9_source_catalog_tightens_detail_before_accepting(self):
        cfg = PipelineConfig()
        rng = np.random.default_rng(42)
        yy, xx = np.mgrid[:256, :256]
        source = np.full((3, 256, 256), 0.02, dtype=np.float32)
        starmask = np.zeros_like(source)
        for center_y, center_x in rng.integers(10, 246, size=(60, 2)):
            amplitude = float(rng.uniform(0.015, 0.05))
            profile = np.exp(
                -(
                    (xx - center_x) ** 2
                    + (yy - center_y) ** 2
                )
                / (2.0 * 1.0**2)
            ).astype(np.float32)
            star = np.stack(
                [
                    profile * amplitude,
                    profile * amplitude * 0.90,
                    profile * amplitude * 0.80,
                ]
            )
            source += star
            starmask += star
        salt = np.zeros((256, 256), dtype=np.float32)
        salt.reshape(-1)[rng.choice(256 * 256, size=900, replace=False)] = 0.006
        source += np.stack([salt, salt * 0.93, salt * 0.86])

        catalog = stage9_quality.build_star_reference_catalog(
            starmask,
            cfg,
            source_image=source,
        )

        self.assertEqual(catalog["status"], "ok")
        self.assertFalse(catalog["source_detail_adaptive_retry"])
        self.assertEqual(catalog["source_detail_percentile_requested"], 98.0)
        self.assertEqual(catalog["source_detail_percentile"], 98.0)
        self.assertGreaterEqual(catalog["matched_component_count"], 60)
        self.assertFalse(
            catalog["source_detail_attempts"][0]["contamination_risk"]
        )
        self.assertGreater(
            catalog["source_raw_single_pixel_component_ratio"],
            catalog["source_single_pixel_component_ratio"],
        )

    def test_stage9_source_catalog_rejects_single_pixel_noise_independently(self):
        cfg = PipelineConfig()
        cfg.stage9_source_component_density_max = 10000.0
        cfg.stage9_source_single_pixel_ratio_max = 0.10
        rng = np.random.default_rng(20260718)
        noisy_plane = np.clip(
            0.02 + rng.normal(0.0, 0.003, (256, 256)),
            0.0,
            1.0,
        ).astype(np.float32)
        noisy_source = np.stack(
            [noisy_plane, noisy_plane * 0.93, noisy_plane * 0.86]
        )

        catalog = stage9_quality.build_star_reference_catalog(
            noisy_source,
            cfg,
            source_image=noisy_source,
        )

        self.assertEqual(catalog["status"], "rejected")
        contaminated_attempt = next(
            attempt
            for attempt in reversed(catalog["source_detail_attempts"])
            if attempt["contamination_risk"]
        )
        self.assertFalse(contaminated_attempt["density_limit_exceeded"])
        self.assertTrue(contaminated_attempt["single_pixel_limit_exceeded"])
        self.assertIn("single_pixel_component_ratio=", catalog["reason"])
        self.assertNotIn("component_density_per_megapixel=", catalog["reason"])

    def test_stage9_source_catalog_excludes_unmatched_diffuse_starmask_residual(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[:160, :160]
        source = np.full((3, 160, 160), 0.025, dtype=np.float32)
        starmask = np.zeros_like(source)
        for index, (cy, cx) in enumerate(
            (y, x)
            for y in (18, 48, 78, 108, 138)
            for x in (20, 58, 96, 134)
        ):
            amplitude = 0.015 + 0.002 * index
            profile = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.1**2))
            rgb_star = np.stack(
                [profile * amplitude, profile * amplitude * 0.85, profile * amplitude * 0.70]
            ).astype(np.float32)
            source += rgb_star
            starmask += rgb_star

        false_residual = 0.025 * np.exp(
            -((xx - 80) ** 2 + (yy - 28) ** 2) / (2.0 * 10.0**2)
        )
        starmask += np.stack(
            [false_residual, false_residual * 0.9, false_residual * 0.8]
        ).astype(np.float32)

        catalog = stage9_quality.build_star_reference_catalog(
            starmask,
            cfg,
            source_image=source,
        )
        _weak, _bright, support = stage9_quality.build_star_overlay_masks(
            catalog,
            strict=False,
        )

        self.assertEqual(catalog["status"], "ok")
        self.assertTrue(catalog["source_matched"])
        self.assertGreaterEqual(catalog["component_count"], 16)
        self.assertFalse(bool(support[28, 80]))
        self.assertLess(float(np.mean(support)), cfg.stage9_star_support_ratio_max)

    def test_stage9_adaptive_starmask_samples_only_connected_compact_support(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[:128, :128]
        diffuse_floor = 0.00003 + 0.00002 * np.sin(xx / 7.0) ** 2
        stars = np.stack(
            [diffuse_floor, diffuse_floor * 0.9, diffuse_floor * 0.8]
        ).astype(np.float32)
        for cy, cx, amplitude in (
            (20, 22, 0.04),
            (38, 94, 0.06),
            (70, 60, 0.08),
            (101, 29, 0.05),
            (96, 104, 0.12),
        ):
            profile = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.5**2))
            stars += np.stack(
                [profile * amplitude, profile * amplitude * 0.85, profile * amplitude * 0.70]
            ).astype(np.float32)

        plan = stage9_quality.calibrate_starmask_asinh(stars, cfg)

        self.assertEqual(plan["status"], "ok")
        self.assertEqual(
            plan["method"],
            "connected_compact_distribution_calibrated_asinh",
        )
        self.assertGreater(plan["compact_component_count"], 0)
        self.assertLess(plan["compact_support_coverage"], 0.10)
        self.assertLess(plan["star_sample_count"], stars.shape[1] * stars.shape[2] * 0.10)

    def test_stage9_compact_starmask_clears_diffuse_residual_before_asinh(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[:128, :128]
        diffuse_floor = 0.00004 + 0.00003 * np.sin(xx / 6.0) ** 2
        stars = np.stack(
            [diffuse_floor, diffuse_floor * 0.9, diffuse_floor * 0.8]
        ).astype(np.float32)
        for cy, cx, amplitude in (
            (22, 24, 0.05),
            (41, 92, 0.08),
            (73, 61, 0.12),
            (99, 31, 0.06),
            (96, 105, 0.18),
        ):
            profile = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.4**2))
            stars += np.stack(
                [profile * amplitude, profile * amplitude * 0.85, profile * amplitude * 0.70]
            ).astype(np.float32)

        plan = stage9_quality.calibrate_starmask_asinh(
            stars,
            cfg,
            include_support_mask=True,
        )
        support = plan.pop("_compact_support_mask")
        compact = stage9_quality.apply_compact_starmask_support(stars, support)
        distance = stage9_quality.scipy_ndimage.distance_transform_edt(~support)
        feather = (~support) & (distance <= np.sqrt(2.0) + 1.0e-6)
        outside = (~support) & ~feather

        self.assertEqual(plan["status"], "ok")
        self.assertTrue(np.any(stars[:, ~support] > 0.0))
        self.assertTrue(np.any(compact[:, feather] > 0.0))
        self.assertTrue(np.all(compact[:, outside] == 0.0))
        self.assertTrue(np.all(compact[:, feather] < stars[:, feather]))
        np.testing.assert_allclose(compact[:, support], stars[:, support])
        self.assertLessEqual(
            plan["predicted_change_ratio"],
            plan["full_layer_predicted_change_ratio"],
        )
        self.assertGreaterEqual(plan["removed_predicted_change_ratio"], 0.0)

    def test_stage9_strict_compact_recovery_uses_narrower_support(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[:160, :160]
        stars = np.zeros((3, 160, 160), dtype=np.float32)
        for cy, cx, amplitude in (
            (28, 30, 0.08),
            (48, 126, 0.10),
            (83, 79, 0.16),
            (128, 39, 0.12),
            (121, 132, 0.20),
        ):
            profile = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.8**2))
            stars += np.stack([profile, profile * 0.85, profile * 0.70]).astype(
                np.float32
            ) * amplitude

        normal = stage9_quality.calibrate_starmask_asinh(stars, cfg)
        strict = stage9_quality.calibrate_starmask_asinh(
            stars,
            cfg,
            strict_support=True,
        )

        self.assertEqual(normal["status"], "ok")
        self.assertEqual(strict["status"], "ok")
        self.assertEqual(strict["support_mode"], "strict_recovery")
        self.assertLess(
            strict["compact_support_coverage"],
            normal["compact_support_coverage"],
        )
        self.assertLess(strict["compact_wing_iterations"], normal["compact_wing_iterations"])

    @staticmethod
    def _stage9_preflight_calibration(
        *,
        coverage: float,
        predicted: float = 0.10,
        status: str = "ok",
        mask: np.ndarray | None = None,
    ) -> dict:
        support = (
            np.asarray(mask, dtype=bool)
            if mask is not None
            else np.zeros((10, 10), dtype=bool)
        )
        return {
            "status": status,
            "reason": "mock unavailable" if status != "ok" else "",
            "support_mode": "normal",
            "compact_support_coverage": coverage,
            "predicted_change_ratio": predicted,
            "predicted_change_ratio_limit": 0.30,
            "weak_star_retention": 1.0 if status == "ok" else 0.0,
            "weak_star_retention_min": 0.80,
            "star_retention": 1.0 if status == "ok" else 0.0,
            "faint_target": 0.26,
            "mid_target": 0.50,
            "bright_target": 0.75,
            "peak_target": 0.90,
            "output_targets": {
                "faint": 0.26,
                "mid": 0.50,
                "bright": 0.75,
                "peak": 0.90,
            },
            "output_profile_mode": "ordinary_support_pixel_percentiles",
            "output_profile": {
                "status": "ok",
                "accepted": status == "ok",
                "hard_failed": status != "ok",
                "actual": {
                    "faint": 0.20,
                    "mid": 0.30,
                    "bright": 0.40,
                    "peak": 0.50,
                },
                "targets": {
                    "faint": 0.26,
                    "mid": 0.50,
                    "bright": 0.75,
                    "peak": 0.90,
                },
                "tolerance": 1e-4,
                "exceeded_anchors": [],
            },
            "_compact_support_mask": support,
            "_output_profile_sample_mask": support,
            "star_reference": {
                "psf_support_radius_max": 6,
                "psf_support_radius_median_px": 2.0,
                "psf_support_radius_p95_px": 3.0,
            },
        }

    def test_stage9_support_preflight_routes_clear_boundary_and_hard_normal(self):
        cfg = PipelineConfig()
        stars = np.zeros((3, 10, 10), dtype=np.float32)
        masks = (
            np.eye(10, dtype=bool),
            np.eye(10, dtype=bool) & (np.indices((10, 10))[0] < 5),
        )
        cases = (
            (0.10, "auto_fallback", "normal_only"),
            (0.13, "auto_fallback", "dual_competition"),
            (0.19, "auto_fallback", "strict_only"),
            (0.13, "preserve_review", "normal_only"),
        )
        for normal_coverage, failure_action, expected_route in cases:
            with self.subTest(
                normal_coverage=normal_coverage,
                failure_action=failure_action,
            ):
                normal = self._stage9_preflight_calibration(
                    coverage=normal_coverage,
                    mask=masks[0],
                )
                strict = self._stage9_preflight_calibration(
                    coverage=0.08,
                    mask=masks[1],
                )
                strict["support_mode"] = "strict_recovery"
                with patch.object(
                    stage9_quality,
                    "calibrate_starmask_asinh",
                    side_effect=[normal, strict],
                ):
                    report = stage9_quality.assess_starmask_support_preflight(
                        stars,
                        cfg,
                        failure_action=failure_action,
                    )
                self.assertEqual(report["route"], expected_route)

    def test_stage9_support_preflight_deduplicates_and_handles_unavailable_modes(self):
        cfg = PipelineConfig()
        stars = np.zeros((3, 10, 10), dtype=np.float32)
        shared_mask = np.eye(10, dtype=bool)
        normal = self._stage9_preflight_calibration(
            coverage=0.10,
            mask=shared_mask,
        )
        strict = self._stage9_preflight_calibration(
            coverage=0.10,
            mask=shared_mask,
        )
        strict["support_mode"] = "strict_recovery"
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[normal, strict],
        ):
            equivalent = stage9_quality.assess_starmask_support_preflight(
                stars,
                cfg,
            )
        self.assertEqual(equivalent["route"], "normal_only")
        self.assertTrue(equivalent["support_masks_equivalent"])
        self.assertTrue(equivalent["compact_support_enabled"])
        self.assertFalse(equivalent["pre_stretch_compact_enabled"])

        unavailable = self._stage9_preflight_calibration(
            coverage=0.0,
            status="unavailable",
        )
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[unavailable, unavailable],
        ):
            rejected = stage9_quality.assess_starmask_support_preflight(
                stars,
                cfg,
            )
        self.assertEqual(rejected["route"], "unavailable")
        self.assertEqual(rejected["status"], "rejected")

        advisory = self._stage9_preflight_calibration(
            coverage=0.13,
            mask=np.eye(10, dtype=bool),
        )
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[advisory, unavailable],
        ):
            strict_unavailable = (
                stage9_quality.assess_starmask_support_preflight(
                    stars,
                    cfg,
                )
            )
        self.assertEqual(strict_unavailable["route"], "normal_only")
        self.assertEqual(
            strict_unavailable["reason_code"],
            "stage9_support_preflight_boundary_strict_unavailable",
        )
        self.assertEqual(
            strict_unavailable["skipped_candidates"][0]["support_mode"],
            "strict_compact",
        )

        cfg.stage9_compact_starmask_enabled = False
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            return_value=normal,
        ) as calibrate:
            disabled = stage9_quality.assess_starmask_support_preflight(
                stars,
                cfg,
            )
        self.assertEqual(disabled["route"], "normal_only")
        self.assertEqual(calibrate.call_count, 1)
        self.assertFalse(disabled["compact_support_enabled"])
        self.assertFalse(disabled["pre_stretch_compact_enabled"])

        cfg.stage9_starmask_pre_stretch_compact_enabled = True
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            return_value=normal,
        ) as calibrate:
            compact_normal_only = (
                stage9_quality.assess_starmask_support_preflight(
                    stars,
                    cfg,
                )
            )
        self.assertEqual(compact_normal_only["route"], "normal_only")
        self.assertEqual(calibrate.call_count, 1)
        self.assertFalse(compact_normal_only["compact_support_enabled"])
        self.assertTrue(
            compact_normal_only["pre_stretch_compact_enabled"]
        )

    def test_stage9_support_preflight_uses_actual_plugin_change_coverage(self):
        cfg = PipelineConfig()
        stars = np.zeros((3, 10, 10), dtype=np.float32)
        normal_mask = np.eye(10, dtype=bool)
        strict_mask = normal_mask & (np.indices((10, 10))[0] < 5)
        normal = self._stage9_preflight_calibration(
            coverage=0.10,
            predicted=0.01,
            mask=normal_mask,
        )
        strict = self._stage9_preflight_calibration(
            coverage=0.05,
            predicted=0.02,
            mask=strict_mask,
        )
        strict["support_mode"] = "strict_recovery"
        plugin = np.zeros_like(stars)
        plugin[:, normal_mask] = 0.25

        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[normal, strict],
        ):
            measured = stage9_quality.assess_starmask_support_preflight(
                stars,
                cfg,
                plugin_stretched_stars=plugin,
            )

        normal_summary = measured["candidates"]["normal"]
        plugin_summary = measured["candidates"]["plugin_normal"]
        self.assertEqual(measured["route"], "dual_competition")
        self.assertEqual(
            measured["reason_code"],
            "stage9_plugin_starmask_output_inadequate_builtin_dual_fallback",
        )
        self.assertEqual(
            normal_summary["predicted_change_source"],
            "calibrated_builtin_stretch",
        )
        self.assertAlmostEqual(
            normal_summary["predicted_change_ratio"],
            0.01,
        )
        self.assertEqual(
            plugin_summary["predicted_change_source"],
            "actual_plugin_stretched_pixels",
        )
        self.assertAlmostEqual(
            plugin_summary["predicted_change_ratio"],
            0.10,
        )
        self.assertFalse(plugin_summary["formal_eligible"])
        self.assertEqual(measured["selected_stretch_source"], "builtin_calibrated")
        self.assertTrue(normal_summary["output_profile"]["accepted"])

        over_target_plugin = np.zeros_like(stars)
        over_target_plugin[:, normal_mask] = 0.95
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[normal, strict],
        ):
            output_rejected = stage9_quality.assess_starmask_support_preflight(
                stars,
                cfg,
                plugin_stretched_stars=over_target_plugin,
            )
        rejected_normal = output_rejected["candidates"]["plugin_normal"]
        self.assertEqual(output_rejected["route"], "dual_competition")
        self.assertTrue(
            rejected_normal["gates"]["starmask_output_targets"]["hard_failed"]
        )
        self.assertIn(
            "peak",
            rejected_normal["output_profile"]["exceeded_anchors"],
        )

        advisory_cfg = PipelineConfig(
            stage9_star_support_ratio_max=0.30,
            stage9_starmask_predicted_change_ratio_max=0.10,
        )
        advisory_mask = np.zeros((10, 10), dtype=bool)
        advisory_mask[:1, :] = True
        advisory_mask[1, :3] = True
        advisory_normal = self._stage9_preflight_calibration(
            coverage=0.13,
            predicted=0.01,
            mask=advisory_mask,
        )
        advisory_normal["predicted_change_ratio_limit"] = 0.10
        advisory_plugin = np.zeros_like(stars)
        advisory_plugin[:, advisory_mask] = 0.25
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[advisory_normal, strict],
        ):
            plugin_advisory = stage9_quality.assess_starmask_support_preflight(
                stars,
                advisory_cfg,
                plugin_stretched_stars=advisory_plugin,
            )
        self.assertEqual(plugin_advisory["route"], "dual_competition")
        self.assertEqual(
            plugin_advisory["candidates"]["plugin_normal"]["gates"][
                "predicted_change_ratio"
            ]["status"],
            "advisory",
        )

        wide_mask = np.zeros((10, 10), dtype=bool)
        wide_mask[:2, :] = True
        wide_normal = self._stage9_preflight_calibration(
            coverage=0.20,
            predicted=0.01,
            mask=wide_mask,
        )
        wide_plugin = np.zeros_like(stars)
        wide_plugin[:, wide_mask] = 0.25
        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[wide_normal, strict],
        ):
            plugin_strict = stage9_quality.assess_starmask_support_preflight(
                stars,
                cfg,
                plugin_stretched_stars=wide_plugin,
            )
        self.assertEqual(plugin_strict["route"], "strict_only")
        self.assertEqual(
            plugin_strict["candidates"]["plugin_normal"][
                "predicted_change_source"
            ],
            "actual_plugin_stretched_pixels",
        )

        with patch.object(
            stage9_quality,
            "calibrate_starmask_asinh",
            side_effect=[normal, strict],
        ):
            unavailable_measurement = (
                stage9_quality.assess_starmask_support_preflight(
                    stars,
                    cfg,
                    plugin_stretched_stars=np.zeros(
                        (3, 8, 8), dtype=np.float32
                    ),
                )
            )
        unavailable_normal = unavailable_measurement["candidates"]["plugin_normal"]
        self.assertEqual(
            unavailable_normal["predicted_change_source"],
            "plugin_measurement_unavailable",
        )
        self.assertIsNone(unavailable_normal["predicted_change_ratio"])
        self.assertTrue(unavailable_normal["hard_failed"])
        self.assertEqual(unavailable_measurement["route"], "dual_competition")

    def test_stage9_support_preflight_public_report_strips_private_masks(self):
        public = stage9_quality.public_starmask_support_preflight(
            {
                "schema": (
                    "starun.stage9-starmas"
                    "k-support-preflight.v2"
                ),
                "status": "ready",
                "route": "normal_only",
                "_calibrations": {"normal": {"_compact_support_mask": np.ones((2, 2))}},
            }
        )
        self.assertNotIn("_calibrations", public)
        self.assertEqual(public["route"], "normal_only")
        json.dumps(public)

    def test_stage9_support_candidate_order_prefers_clean_psf_then_normal_tie(self):
        def quality(
            *,
            ratio: float,
            advisory: bool = False,
            uncertainty_exemption: bool = False,
        ) -> dict:
            return {
                "accepted": True,
                "advisories": ["boundary"] if advisory else [],
                "quality_gates": {},
                "psf_closure": {
                    "uncertainty_exemption_used": uncertainty_exemption,
                    "groups": {
                        "all": {
                            "status": "ok",
                            "fwhm_ratio_median": ratio,
                            "accepted_within_uncertainty": (
                                uncertainty_exemption
                            ),
                        }
                    }
                },
                "metrics": {
                    "weak_star_recovery_ratio": 0.80,
                    "star_support_ratio": 0.08,
                    "highlight_clip_growth": 0.001,
                    "bright_pixel_growth": 0.002,
                },
                "limits": {"weak_star_recovery_ratio": 0.70},
            }

        normal = stage9_star_remixing._stage9_support_candidate_score(
            quality(ratio=0.97),
            support_mode="normal",
        )
        compact = stage9_star_remixing._stage9_support_candidate_score(
            quality(ratio=0.99),
            support_mode="strict_compact",
        )
        self.assertLess(compact, normal)

        normal_tie = stage9_star_remixing._stage9_support_candidate_score(
            quality(ratio=0.99),
            support_mode="normal",
        )
        self.assertLess(normal_tie, compact)

        compact_advisory = stage9_star_remixing._stage9_support_candidate_score(
            quality(ratio=1.0, advisory=True),
            support_mode="strict_compact",
        )
        self.assertLess(normal_tie, compact_advisory)

        strict_farther = stage9_star_remixing._stage9_support_candidate_score(
            quality(ratio=0.98),
            support_mode="normal",
        )
        uncertainty_closer = (
            stage9_star_remixing._stage9_support_candidate_score(
                quality(
                    ratio=1.001,
                    uncertainty_exemption=True,
                ),
                support_mode="normal",
            )
        )
        self.assertLess(strict_farther, uncertainty_closer)

    def test_stage9_adaptive_starmask_caps_predicted_change_coverage(self):
        peak_map = np.geomspace(1e-7, 0.02, 20000, dtype=np.float32).reshape(100, 200)

        stretch, coverage, _threshold, limited = stage9_quality._coverage_limited_stretch(
            peak_map,
            requested_stretch=1000.0,
            offset=0.00002,
            intensity=1.05,
            coverage_limit=0.30,
        )

        self.assertTrue(limited)
        self.assertLess(stretch, 1000.0)
        self.assertLessEqual(coverage, 0.3001)

    def test_stage7_target_local_gate_rejects_bright_core_clipping(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[-32:32, -32:32]
        source = 0.01 + 0.30 * np.exp(-(xx * xx + yy * yy) / (2.0 * 7.0**2))
        baseline = np.stack([source, source * 0.9, source * 0.8]).astype(np.float32)
        candidate = np.clip(baseline * 2.4, 0.0, 1.0)
        candidate[:, 25:39, 25:39] = 1.0

        report = stage7_stretch_metrics.assess_target_local_stretch(
            baseline,
            candidate,
            "bright_emission_reflection_nebula",
            cfg,
        )

        self.assertFalse(report["accepted"])
        self.assertTrue(any("local_core_clip_ratio" in issue for issue in report["issues"]))

    def test_stage7_dark_nebula_local_gate_requires_structure_separation(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[-32:32, -32:32]
        source = 0.01 + 0.10 * ((xx + 32) / 64.0) + 0.02 * np.exp(
            -(xx * xx + yy * yy) / (2.0 * 12.0**2)
        )
        baseline = np.stack([source, source, source]).astype(np.float32)
        flat_candidate = np.full_like(baseline, 0.05)

        report = stage7_stretch_metrics.assess_target_local_stretch(
            baseline,
            flat_candidate,
            "dark_nebula_low_contrast",
            cfg,
        )

        self.assertFalse(report["accepted"])
        self.assertTrue(any("local_dark_separation" in issue for issue in report["issues"]))

    def test_starmask_cleanup_preserves_compact_star_color_and_reduces_diffuse_residual(self):
        cfg = PipelineConfig()
        yy, xx = np.mgrid[-32:32, -32:32]
        diffuse = 0.012 + 0.025 * np.exp(-(xx * xx + yy * yy) / (2.0 * 15.0**2))
        compact = 0.55 * np.exp(-((xx - 4) ** 2 + (yy + 3) ** 2) / (2.0 * 1.25**2))
        faint = 0.09 * np.exp(-((xx + 15) ** 2 + (yy - 10) ** 2) / (2.0 * 1.0**2))
        stars = np.stack(
            [diffuse + compact * 1.00 + faint * 0.80,
             diffuse + compact * 0.72 + faint * 0.90,
             diffuse + compact * 0.48 + faint * 1.00],
            axis=0,
        ).astype(np.float32)

        cleaned, metrics = starmask_cleanup.clean_starmask_pixels(stars, cfg)

        self.assertTrue(metrics["accepted"])
        self.assertGreaterEqual(
            metrics["compact_retention"],
            cfg.stage7_starmask_compact_retention_min,
        )
        self.assertGreaterEqual(
            metrics["faint_compact_retention"],
            cfg.stage7_starmask_small_star_scale - 0.01,
        )
        self.assertLess(metrics["diffuse_residual_ratio"], 1.0)
        self.assertLessEqual(
            metrics["diffuse_residual_ratio"],
            cfg.stage7_starmask_diffuse_residual_ratio_max,
        )
        core_before = stars[:, 29, 36]
        core_after = cleaned[:, 29, 36]
        before_ratio = core_before / core_before[1]
        after_ratio = core_after / core_after[1]
        np.testing.assert_allclose(after_ratio, before_ratio, rtol=1e-4, atol=1e-4)

    def test_starmask_cleanup_hard_rejects_diffuse_residual_above_limit(self):
        cfg = PipelineConfig(stage7_starmask_diffuse_residual_ratio_max=0.01)
        yy, xx = np.mgrid[-32:32, -32:32]
        diffuse = 0.012 + 0.025 * np.exp(-(xx * xx + yy * yy) / (2.0 * 15.0**2))
        compact = 0.55 * np.exp(
            -((xx - 4) ** 2 + (yy + 3) ** 2) / (2.0 * 1.25**2)
        )
        stars = np.stack(
            [diffuse + compact, diffuse + compact * 0.72, diffuse + compact * 0.48]
        ).astype(np.float32)

        _cleaned, metrics = starmask_cleanup.clean_starmask_pixels(stars, cfg)

        self.assertFalse(metrics["accepted"])
        self.assertTrue(metrics["diffuse_hard_gate_failed"])
        self.assertGreater(
            metrics["diffuse_residual_ratio"],
            metrics["limits"]["max_diffuse_residual_ratio"],
        )

        quality = stage6_star_separation._apply_starmask_cleanup_hard_gate(
            {"status": "ok", "issues": [], "derived": {}},
            {"metrics": metrics},
        )
        self.assertEqual(quality["status"], "poor")
        self.assertTrue(quality["derived"]["starmask_cleanup_hard_failed"])

    def test_starmask_diffuse_gate_uses_two_x_advisory_band(self):
        borderline = starmask_cleanup.classify_diffuse_residual_gate(
            0.16,
            0.08,
            0.0005,
        )
        hard_failed = starmask_cleanup.classify_diffuse_residual_gate(
            0.160001,
            0.08,
            0.0005,
        )

        self.assertEqual(borderline["status"], "borderline")
        self.assertTrue(borderline["borderline"])
        self.assertFalse(borderline["hard_failed"])
        self.assertAlmostEqual(borderline["effective_hard_limit"], 0.16)
        self.assertAlmostEqual(borderline["advisory_multiplier"], 2.0)
        self.assertEqual(hard_failed["status"], "hard_failed")
        self.assertTrue(hard_failed["hard_failed"])

    def test_starmask_lower_gate_uses_half_threshold_advisory_band(self):
        borderline = starmask_cleanup.classify_lower_bound_gate(0.41, 0.82)
        hard_failed = starmask_cleanup.classify_lower_bound_gate(0.409999, 0.82)

        self.assertEqual(borderline["status"], "borderline")
        self.assertTrue(borderline["borderline"])
        self.assertAlmostEqual(borderline["effective_hard_limit"], 0.41)
        self.assertEqual(hard_failed["status"], "hard_failed")
        self.assertTrue(hard_failed["hard_failed"])

    def test_starmask_borderline_gate_keeps_quality_ok_but_marks_review(self):
        quality = stage6_star_separation._apply_starmask_cleanup_hard_gate(
            {"status": "ok", "issues": [], "derived": {}},
            {
                "metrics": {
                    "diffuse_residual_ratio": 0.12,
                    "diffuse_borderline": True,
                    "diffuse_hard_gate_failed": False,
                    "limits": {
                        "max_diffuse_residual_ratio": 0.08,
                        "diffuse_uncertainty_abs": 0.0005,
                        "advisory_multiplier": 2.0,
                        "effective_diffuse_hard_limit": 0.16,
                    },
                }
            },
        )

        self.assertEqual(quality["status"], "ok")
        self.assertTrue(quality["derived"]["starmask_cleanup_borderline"])
        self.assertFalse(quality["derived"]["starmask_cleanup_hard_failed"])
        self.assertTrue(quality["advisories"])

    def test_starmask_compact_advisory_propagates_without_rejecting_candidate(self):
        advisory = "compact_retention within advisory band: 0.700<0.820"
        quality = stage6_star_separation._apply_starmask_cleanup_hard_gate(
            {"status": "ok", "issues": [], "derived": {}},
            {
                "metrics": {
                    "advisories": [advisory],
                    "diffuse_residual_ratio": 0.05,
                    "diffuse_borderline": False,
                    "diffuse_hard_gate_failed": False,
                    "limits": {
                        "max_diffuse_residual_ratio": 0.08,
                        "advisory_multiplier": 2.0,
                        "effective_diffuse_hard_limit": 0.16,
                    },
                }
            },
        )

        self.assertEqual(quality["status"], "ok")
        self.assertFalse(quality["issues"])
        self.assertIn(advisory, quality["advisories"])
        self.assertIn(advisory, quality["local_advisories"])

    def test_starmask_cleanup_retries_diffuse_only_pixels_before_hard_reject(self):
        cfg = PipelineConfig(stage7_starmask_nebula_suppression=0.65)
        yy, xx = np.mgrid[-32:32, -32:32]
        diffuse = 0.012 + 0.025 * np.exp(
            -(xx * xx + yy * yy) / (2.0 * 15.0**2)
        )
        compact = 0.55 * np.exp(
            -((xx - 4) ** 2 + (yy + 3) ** 2) / (2.0 * 1.25**2)
        )
        faint = 0.09 * np.exp(
            -((xx + 15) ** 2 + (yy - 10) ** 2) / (2.0 * 1.0**2)
        )
        stars = np.stack(
            [
                diffuse + compact + faint * 0.80,
                diffuse + compact * 0.72 + faint * 0.90,
                diffuse + compact * 0.48 + faint,
            ]
        ).astype(np.float32)

        _cleaned, metrics = starmask_cleanup.clean_starmask_pixels(stars, cfg)

        retry = metrics["diffuse_retry"]
        self.assertTrue(retry["attempted"])
        self.assertTrue(retry["applied"])
        self.assertGreater(retry["ratio_before"], cfg.stage7_starmask_diffuse_residual_ratio_max)
        self.assertLessEqual(
            retry["ratio_after"],
            cfg.stage7_starmask_diffuse_residual_ratio_max,
        )
        self.assertTrue(metrics["accepted"])
        self.assertGreaterEqual(
            metrics["compact_retention"],
            cfg.stage7_starmask_compact_retention_min,
        )
        self.assertGreaterEqual(
            metrics["faint_compact_retention"],
            cfg.stage7_starmask_small_star_scale - 0.01,
        )

    def test_starmask_cleanup_applies_compact_retention_advisory_without_rollback(self):
        yy, xx = np.mgrid[-16:16, -16:16]
        compact = 0.65 * np.exp(-(xx * xx + yy * yy) / (2.0 * 1.2**2))
        diffuse = 0.02 + 0.02 * np.exp(-(xx * xx + yy * yy) / (2.0 * 10.0**2))
        pixels = np.stack(
            [diffuse + compact, diffuse + compact * 0.75, diffuse + compact * 0.50],
            axis=0,
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            process_dir = Path(tmp)
            starmask_path = process_dir / "starmask.fit"
            starmask_path.touch()
            writes = []
            commands = []

            class _CleanupPipeline:
                cfg = PipelineConfig(stage7_starmask_compact_retention_min=0.98)
                starmask_file = starmask_path
                stretched_name = "stage5_linear"
                log = _Log()
                siril = SimpleNamespace(
                    get_image_pixeldata=lambda preview=False: pixels.copy(),
                    set_image_pixeldata=lambda output: writes.append(output),
                )

                def cmd_with_check(self, *args):
                    commands.append(args)

                @staticmethod
                def _short_text(value, _limit=180):
                    return str(value)

            result = stage7_repair.stage7_clean_starmask(_CleanupPipeline())

        self.assertEqual(result["status"], "applied")
        self.assertTrue(writes)
        self.assertIn(("save", "starmask_clean"), commands)
        self.assertEqual(result["metrics"]["compact_gate_status"], "borderline")
        self.assertTrue(
            any(
                item.startswith("compact_retention within advisory band")
                for item in result["metrics"]["advisories"]
            )
        )

    def test_starmask_cleanup_preserves_raw_and_writes_clean_layer(self):
        yy, xx = np.mgrid[-24:24, -24:24]
        compact = 0.60 * np.exp(-(xx * xx + yy * yy) / (2.0 * 1.2**2))
        diffuse = 0.008 + 0.018 * np.exp(-(xx * xx + yy * yy) / (2.0 * 12.0**2))
        pixels = np.stack(
            [diffuse + compact, diffuse + compact * 0.78, diffuse + compact * 0.55],
            axis=0,
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            process_dir = Path(tmp)
            starmask_path = process_dir / "starmask_raw.fit"
            starmask_path.write_bytes(b"raw-star-layer")
            writes = []
            commands = []

            class _CleanupPipeline:
                cfg = PipelineConfig()
                starmask_file = starmask_path
                stretched_name = "stage5_linear"
                log = _Log()
                siril = SimpleNamespace(
                    get_image_pixeldata=lambda preview=False: pixels.copy(),
                    set_image_pixeldata=lambda output: writes.append(output),
                )

                def cmd_with_check(self, *args):
                    commands.append(args)

                @staticmethod
                def _short_text(value, _limit=180):
                    return str(value)

            pipeline = _CleanupPipeline()
            result = stage7_repair.stage7_clean_starmask(pipeline)
            raw_bytes = (process_dir / "starmask_raw.fit").read_bytes()

        self.assertEqual(result["status"], "applied")
        self.assertEqual(raw_bytes, b"raw-star-layer")
        self.assertTrue(writes)
        self.assertIn(("save", "starmask_clean"), commands)
        self.assertNotIn(("save", "starmask"), commands)
        self.assertEqual(pipeline.starmask_file.name, "starmask_clean.fit")


class _Siril:
    def get_image_shape(self):
        return (3, 100, 100)

    def get_image_pixeldata(self, preview: bool = False):
        return None


class _Pipeline:
    def __init__(self, root: Path) -> None:
        self.cfg = PipelineConfig()
        self.log = _Log()
        self.siril = _Siril()
        self.work_dir = root
        self.process_dir = root / "process"
        self.process_dir.mkdir()
        self.results: list[tuple[str, str, float, str]] = []
        self.commands: list[tuple[object, ...]] = []
        self.pipeline_policy = {}
        self.workflow_command_used = {}
        self.source_file = root / "stacked.fit"
        self.stretched_name = "stage7_stretched"
        self.starless_file = None
        self.starmask_file = None
        self._stage1_input_mode = "stacked"
        self._last_plugin_script_error = None
        self._last_sasp_stage8_api_error = None
        self._last_aberration_api_error = None
        self._last_scunet_fallback_error = None
        self._review_requirements: dict[tuple[int, str], dict[str, object]] = {}

    def _clear_stage_reviews(self, stage: int) -> None:
        self._review_requirements = {
            key: value
            for key, value in self._review_requirements.items()
            if key[0] != int(stage)
        }

    def _require_review(
        self,
        stage: int,
        code: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self._review_requirements[(int(stage), str(code))] = {
            "stage": int(stage),
            "code": str(code),
            "details": dict(details or {}),
        }

    def _stage_review_reasons(self, stage: int) -> list[str]:
        return [
            value["code"]
            for key, value in self._review_requirements.items()
            if key[0] == int(stage)
        ]

    def _review_requirements_payload(self) -> list[dict[str, object]]:
        return [
            dict(value)
            for _key, value in sorted(self._review_requirements.items())
        ]

    def _record_stage(
        self,
        name: str,
        status: str,
        duration: float,
        message: str = "",
        **_metadata: object,
    ) -> None:
        self.results.append((name, status, duration, message))

    def cmd_with_check(self, *args, **_kwargs) -> None:
        self.commands.append(tuple(args))

    def _save_stage_output(self, stem: str) -> bool:
        (self.process_dir / f"{stem}.fit").touch()
        return True

    def _write_stage_json(self, _name: str, payload: dict) -> None:
        # Match the production writer's JSON boundary so recursive report
        # structures cannot pass the stage smoke suite unnoticed.
        json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _short_text(self, value, _limit: int = 160) -> str:
        return str(value)

    def _stage_diff_note(self, *_args) -> str:
        return ""

    def _measure_current_features(self):
        return ImageFeatures(edge_black_ratio=0.0)

    def _adaptive_features_current(self):
        return {}

    def _feature_summary_note(self, _label: str) -> str:
        return ""


class PipelineStageSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pipeline = _Pipeline(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_stage1_preparation_smoke(self) -> None:
        source = self.root / "stacked.fit"
        source.touch()
        self.pipeline._prepare_process_dir = lambda: None
        self.pipeline._find_fit_files = lambda: [source]
        self.pipeline._is_candidate_stacked = lambda _path: True
        self.pipeline._load_stacked_file = lambda files: self.commands_append("load_stacked", files)
        self.pipeline._preprocess_light_frames = lambda _files: None

        stage1_preparation.run_stage1_preparation(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")

    def commands_append(self, *args) -> None:
        self.pipeline.commands.append(tuple(args))

    def test_stage2_view_correction_smoke(self) -> None:
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=(
                    None,
                    "native contour crop skipped: no candidate",
                    {
                        "method": "native_contour",
                        "accepted": False,
                        "reason": "no_candidate",
                        "candidate": None,
                    },
                ),
            ),
            patch.object(stage2_view_correction, "_detect_auto_edge_crop", return_value=(None, "no crop")),
            patch.object(stage2_view_correction, "_edge_color_artifact_crop", return_value=""),
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage2_corrected.fit").exists())
        self.assertEqual(self.pipeline.stage2_crop_report["mode"], "native_no_crop")
        self.assertEqual(
            self.pipeline.stage2_crop_report["stacked_master_footprint"]["status"],
            "unavailable",
        )

    def test_stage2_footprint_failure_does_not_change_stage_result(self) -> None:
        self.pipeline.siril.get_image_pixeldata = lambda preview=False: np.ones(
            (3, 100, 100),
            dtype=np.float32,
        )
        with (
            patch.object(
                stage2_view_correction,
                "infer_stacked_master_footprint",
                side_effect=Exception("forced observer failure"),
            ),
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=(
                    None,
                    "native contour crop skipped: no candidate",
                    {
                        "method": "native_contour",
                        "accepted": False,
                        "reason": "no_candidate",
                        "candidate": None,
                    },
                ),
            ),
            patch.object(
                stage2_view_correction,
                "_detect_auto_edge_crop",
                return_value=(None, "no crop"),
            ),
            patch.object(
                stage2_view_correction,
                "_edge_color_artifact_crop",
                return_value="",
            ),
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(
            self.pipeline.stage2_crop_report["stacked_master_footprint"]["status"],
            "unavailable",
        )
        self.assertEqual(self.pipeline.stage2_crop_report["advisories"], [])
        self.assertFalse(self.pipeline.stage2_crop_report["requires_review"])

    def test_stage2_footprint_observer_runs_before_crop_without_owning_decision(self) -> None:
        observed_before_commands = []
        footprint = {
            "schema": "starun.stage2-stacked-footprint-evidence.v1",
            "status": "available",
            "source_mode": "stacked_master_inference",
            "observer_only": True,
            "captured_before_crop": True,
            "source_artifact": "stage1_prepared.fit",
            "source_sha256": "a" * 64,
            "input_shape": {"channels": 3, "height": 100, "width": 100},
            "layers": {
                "fill_support": {"status": "available"},
                "relative_coverage": {"status": "available"},
            },
            "limitations": [
                "not_per_frame_registration_footprint",
                "not_crop_authority",
            ],
        }

        def observe(_pipeline, _shape):
            observed_before_commands.append(list(self.pipeline.commands))
            return footprint

        with (
            patch.object(
                stage2_view_correction,
                "_stage2_stacked_master_footprint",
                side_effect=observe,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=(
                    (4, 4, 92, 92),
                    "native contour crop detected",
                    {
                        "method": "native_contour",
                        "accepted": True,
                        "reason": "near_black_boundary_confirmed",
                        "candidate": {"x": 4, "y": 4, "width": 92, "height": 92},
                    },
                ),
            ),
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as fallback,
            patch.object(stage2_view_correction, "_edge_color_artifact_crop") as color,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(observed_before_commands, [[]])
        self.assertEqual(self.pipeline.commands, [("crop", "4", "4", "92", "92")])
        fallback.assert_not_called()
        color.assert_not_called()
        self.assertIs(
            self.pipeline.stage2_crop_report["stacked_master_footprint"],
            footprint,
        )
        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertFalse(self.pipeline.stage2_crop_report["requires_review"])

    def test_stage2_preserve_mode_keeps_footprint_without_status_effect(self) -> None:
        self.pipeline.cfg.stage2_processing_mode = "preserve"
        footprint = {
            "schema": "starun.stage2-stacked-footprint-evidence.v1",
            "status": "unavailable",
            "source_mode": "stacked_master_inference",
            "observer_only": True,
            "captured_before_crop": True,
            "source_artifact": None,
            "source_sha256": None,
            "input_shape": {"channels": 3, "height": 100, "width": 100},
            "layers": {},
            "limitations": [
                "not_per_frame_registration_footprint",
                "not_crop_authority",
            ],
        }
        with patch.object(
            stage2_view_correction,
            "_stage2_stacked_master_footprint",
            return_value=footprint,
        ) as observer:
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        observer.assert_called_once()
        self.assertEqual(self.pipeline.commands, [])
        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(self.pipeline.stage2_crop_report["mode"], "user_preserve")
        self.assertIs(
            self.pipeline.stage2_crop_report["stacked_master_footprint"],
            footprint,
        )
        self.assertEqual(self.pipeline.stage2_crop_report["advisories"], [])

    def test_stage2_native_contour_uses_guarded_crop_and_skips_fallback(self) -> None:
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=(
                    (4, 4, 92, 92),
                    "native contour crop detected",
                    {
                        "method": "native_contour",
                        "accepted": True,
                        "reason": "near_black_boundary_confirmed",
                        "candidate": {"x": 4, "y": 4, "width": 92, "height": 92},
                    },
                ),
            ),
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as detect,
            patch.object(stage2_view_correction, "_edge_color_artifact_crop") as color_crop,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        detect.assert_not_called()
        color_crop.assert_not_called()
        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(self.pipeline.commands, [("crop", "4", "4", "92", "92")])
        self.assertEqual(self.pipeline.stage2_crop_report["mode"], "native_contour")
        self.assertEqual(
            self.pipeline.stage2_crop_report["total_crop"],
            {"left": 4, "top": 4, "right": 4, "bottom": 4},
        )

    def test_stage2_primary_safe_consensus_suppresses_legacy_crop(self) -> None:
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=(
                    None,
                    "native contour crop skipped: full frame valid",
                    {
                        "method": "native_contour",
                        "accepted": False,
                        "reason": "full_frame_is_valid",
                        "candidate": None,
                    },
                ),
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                return_value=(
                    None,
                    "field rotation clear",
                    {
                        "method": "native_field_rotation",
                        "accepted": False,
                        "reason": "no_significant_edge_connected_coverage_anomaly",
                        "candidate": None,
                    },
                ),
            ),
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as legacy,
            patch.object(stage2_view_correction, "_edge_color_artifact_crop") as color,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        legacy.assert_not_called()
        color.assert_not_called()
        self.assertEqual(self.pipeline.commands, [])
        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(
            self.pipeline.stage2_crop_report["detector_consensus"]["decision"],
            "preserve_full_frame",
        )
        self.assertEqual(
            self.pipeline.stage2_crop_report["reason_code"],
            "primary_detectors_preserve_full_frame",
        )

    def test_stage2_legacy_crop_intruding_center_is_rejected_not_clamped(self) -> None:
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=(
                    None,
                    "native detector inconclusive",
                    {
                        "method": "native_contour",
                        "accepted": False,
                        "reason": "no_candidate",
                        "candidate": None,
                    },
                ),
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                return_value=(
                    None,
                    "field detector unavailable",
                    {
                        "method": "native_field_rotation",
                        "accepted": False,
                        "reason": "insufficient_valid_samples",
                        "candidate": None,
                    },
                ),
            ),
            patch.object(
                stage2_view_correction,
                "_detect_auto_edge_crop",
                return_value=((15, 15, 70, 70), "legacy crop candidate"),
            ),
            patch.object(stage2_view_correction, "_edge_color_artifact_crop") as color,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        color.assert_not_called()
        self.assertEqual(self.pipeline.commands, [])
        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertEqual(
            self.pipeline.stage2_crop_report["reason_code"],
            "crop_detector_conflict",
        )
        self.assertEqual(
            self.pipeline.stage2_crop_report["crop_limit_hits"][-1]["applied"],
            None,
        )
        self.assertIn(
            "crop_detector_conflict",
            self.pipeline._stage_review_reasons(2),
        )

    def test_stage2_real_native_detector_applies_one_siril_crop(self) -> None:
        pixels = np.full((3, 100, 120), 0.02, dtype=np.float32)
        pixels[:, :8, :] = 0.0
        pixels[:, -8:, :] = 0.0
        pixels[:, :, :8] = 0.0
        pixels[:, :, -8:] = 0.0
        self.pipeline.siril = SimpleNamespace(
            get_image_shape=lambda: (3, 100, 120),
            get_image_pixeldata=lambda preview=False: pixels,
        )

        with (
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as detect,
            patch.object(stage2_view_correction, "_edge_color_artifact_crop") as color_crop,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        detect.assert_not_called()
        color_crop.assert_not_called()
        self.assertEqual(
            self.pipeline.commands,
            [("crop", "8", "8", "104", "84")],
        )
        self.assertEqual(self.pipeline.stage2_crop_report["mode"], "native_contour")
        self.assertTrue(
            self.pipeline.stage2_crop_report["native_contour"]["accepted"]
        )

    def test_stage2_field_rotation_crop_runs_once_and_skips_edge_chain(self) -> None:
        no_contour = (
            None,
            "native contour detector was inconclusive",
            {
                "method": "native_contour",
                "accepted": False,
                "reason": "no_candidate",
                "candidate": None,
            },
        )
        field_candidate = (
            (20, 20, 60, 60),
            "field-rotation coverage crop detected",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "candidate": {"x": 20, "y": 20, "width": 60, "height": 60},
            },
        )
        residual_clear = (
            None,
            "field-rotation coverage crop skipped: no residual",
            {
                "method": "native_field_rotation",
                "accepted": False,
                "reason": "no_significant_edge_connected_coverage_anomaly",
                "candidate": None,
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                side_effect=(field_candidate, residual_clear),
            ) as field_detect,
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as edge_detect,
            patch.object(stage2_view_correction, "_edge_color_artifact_crop") as color_crop,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(field_detect.call_count, 2)
        edge_detect.assert_not_called()
        color_crop.assert_not_called()
        self.assertEqual(self.pipeline.commands, [("crop", "8", "8", "84", "84")])
        self.assertEqual(
            self.pipeline.stage2_crop_report["mode"],
            "native_field_rotation",
        )
        self.assertFalse(self.pipeline.stage2_crop_report["requires_review"])
        self.assertFalse(
            self.pipeline.stage2_crop_report["field_rotation"]["residual"][
                "accepted"
            ]
        )
        self.assertEqual(
            self.pipeline.stage2_crop_report["field_rotation"]["actual_passes"],
            1,
        )
        field_rotation = self.pipeline.stage2_crop_report["field_rotation"]
        first_pass = field_rotation["passes"][0]
        self.assertIsNot(first_pass["detector"], field_rotation)
        self.assertIsNot(first_pass["verification"], field_rotation["residual"])
        json.loads(json.dumps(self.pipeline.stage2_crop_report))

    def test_stage2_field_rotation_area_limit_marks_stage2_review(self) -> None:
        self.pipeline.cfg.stage2_center_protect_area_ratio = 1.0
        no_contour = (
            None,
            "no contour",
            {"method": "native_contour", "accepted": False},
        )
        field_candidate = (
            (10, 10, 80, 80),
            "field rotation",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "evidence": {"edge_connected_ratio": 0.25},
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                return_value=field_candidate,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_auto_edge_crop",
                return_value=(None, "no generic crop"),
            ),
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(
            self.pipeline._stage_review_reasons(2),
            ["field_rotation_residual_review"],
        )
        self.assertEqual(self.pipeline._stage_review_reasons(3), [])
        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertEqual(
            self.pipeline.stage2_crop_report["field_rotation"]["actual_passes"],
            1,
        )
        json.loads(json.dumps(self.pipeline.stage2_crop_report))

    def test_stage2_full_frame_and_field_rotation_conflict_preserves_frame(self) -> None:
        no_contour = (
            None,
            "native contour crop skipped: full frame valid",
            {
                "method": "native_contour",
                "accepted": False,
                "reason": "full_frame_is_valid",
                "candidate": None,
            },
        )
        field_candidate = (
            (20, 20, 60, 60),
            "field-rotation coverage crop detected",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "candidate": {"x": 20, "y": 20, "width": 60, "height": 60},
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                return_value=field_candidate,
            ),
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as edge_detect,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        edge_detect.assert_not_called()
        self.assertEqual(self.pipeline.commands, [])
        self.assertTrue(self.pipeline.stage2_crop_report["requires_review"])
        self.assertTrue(self.pipeline._stage2_view_review_required)
        self.assertFalse(
            getattr(self.pipeline, "_background_review_required", False)
        )
        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertEqual(
            self.pipeline.stage2_crop_report["reason_code"],
            "crop_detector_conflict",
        )
        self.assertEqual(
            self.pipeline.stage2_crop_report["field_rotation"][
                "application_status"
            ],
            "rejected_detector_conflict",
        )
        self.assertEqual(
            self.pipeline.stage2_crop_report["detector_consensus"]["decision"],
            "crop_detector_conflict",
        )
        json.loads(json.dumps(self.pipeline.stage2_crop_report))

    def test_stage2_known_native_frame_and_wcs_downgrade_conflict_to_advisory(self) -> None:
        self.pipeline.siril.get_image_shape = lambda: (3, 3840, 2160)
        self.pipeline._read_fits_header_metadata = lambda *_args: {
            "_header_source": "stage1_prepared.fit",
            "TELESCOP": "ZWO Seestar S30 Pro",
            "NAXIS1": 2160,
            "NAXIS2": 3840,
            "CRVAL1": 270.868,
            "CRVAL2": -23.52,
            "CDELT1": -0.00102,
            "CDELT2": 0.00102,
        }
        no_contour = (
            None,
            "native contour crop skipped: full frame valid",
            {
                "method": "native_contour",
                "accepted": False,
                "reason": "full_frame_is_valid",
                "candidate": None,
            },
        )
        field_candidate = (
            (0, 598, 2160, 3242),
            "field-rotation coverage crop detected",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "candidate": {
                    "x": 0,
                    "y": 598,
                    "width": 2160,
                    "height": 3242,
                },
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                return_value=field_candidate,
            ),
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as edge_detect,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        edge_detect.assert_not_called()
        report = self.pipeline.stage2_crop_report
        self.assertEqual(self.pipeline.commands, [])
        self.assertFalse(report["requires_review"])
        self.assertFalse(self.pipeline._stage2_view_review_required)
        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(
            report["reason_code"],
            "field_rotation_full_frame_advisory",
        )
        self.assertEqual(
            report["detector_consensus"]["decision"],
            "preserve_full_frame_advisory",
        )
        self.assertTrue(
            report["full_frame_context"]["advisory_authorized"]
        )
        self.assertTrue(report["advisories"])

    def test_stage2_s30_physical_geometry_downgrades_non_wedge_anomaly(self) -> None:
        self.pipeline.source_file = self.root / (
            "SeestarS30Pro-M8-60s169_IRCUT.xisf"
        )
        self.pipeline.siril.get_image_shape = lambda: (3, 3772, 2146)
        self.pipeline._read_fits_header_metadata = lambda *_args: {
            "_header_source": "stage1_prepared.fit",
            "TELESCOP": "S30 Pro_6a110e03",
            "INSTRUME": "imx585",
            "NAXIS1": 2146,
            "NAXIS2": 3772,
            "RA": 271.311860502958,
            "DEC": -23.5068506982248,
            "FOCALLEN": 160.0,
            "XPIXSZ": 2.9,
            "YPIXSZ": 2.9,
        }
        no_contour = (
            None,
            "native contour crop skipped: full frame valid",
            {
                "method": "native_contour",
                "accepted": False,
                "reason": "full_frame_is_valid",
                "candidate": None,
            },
        )
        non_wedge_anomaly = (
            None,
            "field-rotation anomaly lacks corner wedge geometry",
            {
                "method": "native_field_rotation",
                "accepted": False,
                "reason": "edge_connected_anomaly_lacks_corner_wedge_geometry",
                "candidate": None,
                "evidence": {
                    "edge_connected_ratio": 0.08,
                    "wedge_geometry_confirmed": False,
                    "corner_hit_count": 1,
                    "boundary_side_count": 2,
                },
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                return_value=non_wedge_anomaly,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_auto_edge_crop",
            ) as edge_detect,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        edge_detect.assert_not_called()
        report = self.pipeline.stage2_crop_report
        self.assertEqual(self.pipeline.commands, [])
        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertFalse(report["requires_review"])
        self.assertEqual(
            report["reason_code"],
            "field_rotation_full_frame_advisory",
        )
        self.assertEqual(report["full_frame_context"]["known_device"], "seestar s30 pro")
        self.assertTrue(report["full_frame_context"]["native_frame_match"])
        self.assertFalse(report["full_frame_context"]["wcs_valid"])
        self.assertTrue(
            report["full_frame_context"]["astrometric_geometry_valid"]
        )
        self.assertTrue(report["full_frame_context"]["advisory_authorized"])
        self.assertEqual(
            report["field_rotation"]["application_status"],
            "rejected_full_frame_advisory",
        )

    def test_stage2_signed_source_name_supplies_s30_identity_for_full_frame(self) -> None:
        self.pipeline.source_file = None
        self.pipeline._stage1_original_source_file = self.root / (
            "SeestarS30Pro-NGC6888-60s720_LP.xisf"
        )
        self.pipeline.siril.get_image_shape = lambda: (3, 3840, 2160)
        self.pipeline._read_fits_header_metadata = lambda *_args: {
            "_header_source": "stage1_prepared.fit",
            "INSTRUME": "imx585",
            "NAXIS1": 2160,
            "NAXIS2": 3840,
            "RA": 303.280899104167,
            "DEC": 38.4100161652774,
            "FOCALLEN": 160.0,
            "XPIXSZ": 2.9,
            "YPIXSZ": 2.9,
        }
        no_contour = (
            None,
            "native contour crop skipped: full frame valid",
            {
                "method": "native_contour",
                "accepted": False,
                "reason": "full_frame_is_valid",
                "candidate": None,
            },
        )
        non_wedge_anomaly = (
            None,
            "field-rotation anomaly lacks corner wedge geometry",
            {
                "method": "native_field_rotation",
                "accepted": False,
                "reason": "edge_connected_anomaly_lacks_corner_wedge_geometry",
                "candidate": None,
                "evidence": {
                    "edge_connected_ratio": 0.0326,
                    "wedge_geometry_confirmed": False,
                    "corner_hit_count": 2,
                    "boundary_side_count": 4,
                },
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                return_value=non_wedge_anomaly,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_auto_edge_crop",
            ) as edge_detect,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        edge_detect.assert_not_called()
        report = self.pipeline.stage2_crop_report
        self.assertFalse(report["requires_review"])
        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(
            report["reason_code"],
            "field_rotation_full_frame_advisory",
        )
        self.assertEqual(
            report["full_frame_context"]["known_device"],
            "seestar s30 pro",
        )
        self.assertTrue(report["full_frame_context"]["native_frame_match"])
        self.assertTrue(
            report["full_frame_context"]["astrometric_geometry_valid"]
        )

    def test_stage2_second_field_rotation_crop_clears_residual(self) -> None:
        no_contour = (
            None,
            "no contour",
            {"method": "native_contour", "accepted": False},
        )
        first = (
            (4, 4, 92, 92),
            "first field crop",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "evidence": {"edge_connected_ratio": 0.24},
            },
        )
        residual = (
            (4, 4, 84, 84),
            "residual field crop",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "evidence": {"edge_connected_ratio": 0.12},
            },
        )
        clear = (
            None,
            "clear",
            {
                "method": "native_field_rotation",
                "accepted": False,
                "reason": "no_significant_edge_connected_coverage_anomaly",
                "evidence": {"edge_connected_ratio": 0.0},
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                side_effect=(first, residual, clear),
            ) as detect,
            patch.object(stage2_view_correction, "_detect_auto_edge_crop") as edge,
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(detect.call_count, 3)
        edge.assert_not_called()
        self.assertEqual(
            self.pipeline.commands,
            [
                ("crop", "4", "4", "92", "92"),
                ("crop", "4", "4", "84", "84"),
            ],
        )
        report = self.pipeline.stage2_crop_report
        self.assertFalse(report["requires_review"])
        self.assertEqual(report["total_crop"], {"left": 8, "top": 8, "right": 8, "bottom": 8})
        self.assertEqual(len(report["field_rotation"]["passes"]), 2)
        self.assertTrue(report["field_rotation"]["passes"][1]["accepted"])
        self.assertEqual(report["field_rotation"]["actual_passes"], 2)
        self.assertGreaterEqual(report["crops"][-1]["retained_area_ratio"], 0.70)
        first_pass, second_pass = report["field_rotation"]["passes"]
        self.assertIsNot(first_pass["detector"], report["field_rotation"])
        self.assertIsNot(
            first_pass["verification"],
            second_pass["detector"],
        )
        self.assertIsNot(
            second_pass["verification"],
            report["field_rotation"]["residual"],
        )
        json.loads(json.dumps(report))

    def test_stage2_second_field_rotation_crop_rolls_back_without_improvement(self) -> None:
        no_contour = (
            None,
            "no contour",
            {"method": "native_contour", "accepted": False},
        )
        first = (
            (2, 2, 96, 96),
            "first field crop",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "evidence": {"edge_connected_ratio": 0.20},
            },
        )
        residual = (
            (2, 2, 92, 92),
            "residual field crop",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "evidence": {"edge_connected_ratio": 0.18},
            },
        )
        worse = (
            (2, 2, 88, 88),
            "still residual",
            {
                "method": "native_field_rotation",
                "accepted": True,
                "reason": "edge_connected_field_rotation_confirmed",
                "evidence": {"edge_connected_ratio": 0.19},
            },
        )
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=no_contour,
            ),
            patch.object(
                stage2_view_correction,
                "_detect_field_rotation_candidate",
                side_effect=(first, residual, worse),
            ),
            patch.object(stage2_view_correction, "_detect_auto_edge_crop"),
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        report = self.pipeline.stage2_crop_report
        second = report["field_rotation"]["passes"][1]
        self.assertTrue(second["rolled_back"])
        self.assertEqual(
            second["rollback_reason"],
            "field_rotation_residual_not_improved",
        )
        self.assertEqual(report["total_crop"], {"left": 2, "top": 2, "right": 2, "bottom": 2})
        self.assertEqual(report["field_rotation"]["actual_passes"], 1)
        self.assertTrue(report["requires_review"])
        self.assertEqual(self.pipeline._stage_review_reasons(2), ["field_rotation_residual_review"])
        self.assertIn(("load", "stage2_field_rotation_pass1"), self.pipeline.commands)
        first_pass, second_pass = report["field_rotation"]["passes"]
        self.assertIsNot(first_pass["detector"], report["field_rotation"])
        self.assertIsNot(
            first_pass["verification"],
            second_pass["detector"],
        )
        json.loads(json.dumps(report))

    def test_stage2_center_area_protection_constrains_aggressive_crop(self) -> None:
        shape = {"channels": 3, "height": 100, "width": 100}
        crop_report = {
            "original_shape": shape,
            "current_shape": shape,
            "total_left": 0,
            "total_top": 0,
            "crops": [],
            "crop_limit_hits": [],
            "center_protection": stage2_view_correction._stage2_center_protection(
                shape,
                self.pipeline.cfg.stage2_center_protect_area_ratio,
            ),
        }

        applied_rect, note = stage2_view_correction._stage2_apply_crop(
            self.pipeline,
            crop_report,
            20,
            20,
            60,
            60,
            reason="test_aggressive_crop",
        )

        self.assertEqual(applied_rect, (8, 8, 84, 84))
        self.assertIn("center-area protection constrained crop", note)
        self.assertGreaterEqual((84 * 84) / float(100 * 100), 0.70)
        self.assertEqual(self.pipeline.commands[-1], ("crop", "8", "8", "84", "84"))
        self.assertTrue(crop_report["crops"][-1]["center_protection_limited"])
        self.assertEqual(len(crop_report["crop_limit_hits"]), 1)

    def test_stage2_safe_crop_is_expanded_to_even_boundaries(self) -> None:
        shape = {"channels": 3, "height": 100, "width": 100}
        crop_report = {
            "original_shape": shape,
            "current_shape": shape,
            "total_left": 0,
            "total_top": 0,
            "crops": [],
            "crop_limit_hits": [],
            "center_protection": stage2_view_correction._stage2_center_protection(
                shape,
                self.pipeline.cfg.stage2_center_protect_area_ratio,
            ),
        }

        applied_rect, note = stage2_view_correction._stage2_apply_crop(
            self.pipeline,
            crop_report,
            2,
            3,
            95,
            94,
            reason="test_safe_crop",
        )

        self.assertEqual(applied_rect, (2, 2, 96, 96))
        self.assertIn("center-area protection constrained crop", note)
        self.assertEqual(len(crop_report["crop_limit_hits"]), 1)

    def test_stage2_center_area_protection_is_cfa_alignment_safe(self) -> None:
        shape = {"channels": 3, "height": 3753, "width": 2111}

        protection = stage2_view_correction._stage2_center_protection(shape, 0.70)

        self.assertGreaterEqual(protection["actual_area_ratio"], 0.70)
        self.assertEqual(protection["left"] % 2, 0)
        self.assertEqual(protection["top"] % 2, 0)
        self.assertEqual(protection["width"] % 2, 0)
        self.assertEqual(protection["height"] % 2, 0)

    def test_stage2_soft_edge_evidence_requires_two_independent_signals(self) -> None:
        hard_black = np.array([False, False, True])
        dark_step = np.array([True, True, False])
        color_cast = np.array([False, True, False])

        combined = stage2_view_correction._combine_edge_evidence(
            hard_black,
            dark_step,
            color_cast,
        )

        np.testing.assert_array_equal(combined, np.array([False, True, True]))

    def test_stage2_rectangular_black_border_does_not_cross_contaminate_axes(self) -> None:
        height, width = 400, 240
        pixels = np.full((3, height, width), 0.02, dtype=np.float32)
        pixels[:, :20, :] = 0.0
        pixels[:, -20:, :] = 0.0
        pixels[:, :, :12] = 0.0
        pixels[:, :, -12:] = 0.0
        self.pipeline.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels,
        )

        crop_rect, _note = stage2_view_correction._detect_auto_edge_crop(
            self.pipeline,
            is_adaptive=False,
        )

        self.assertEqual(crop_rect, (15, 23, 210, 354))

    @property
    def process_dir(self) -> Path:
        return self.pipeline.process_dir

    def test_stage3_background_extraction_smoke(self) -> None:
        self.pipeline._stage3_measure_features = lambda _label: ImageFeatures()
        self.pipeline._stage3_signal_preservation_metrics = lambda *_args: {}
        self.pipeline._stage3_quality_gate = lambda *_args: (True, "ok")
        self.pipeline._stage3_subsky_rbf_candidates = lambda: []
        self.pipeline.workflow_plugin_probe_enabled = False
        with (
            patch.object(
                stage3_background_extraction,
                "_stage3_background_candidate_chain",
                return_value=([], [], "smoke_no_candidates"),
            ),
            patch.object(stage3_background_extraction, "_stage3_theoretical_plugin_candidates", return_value=[]),
            patch.object(stage3_background_extraction, "_stage3_graxpert_candidates", return_value=[]),
        ):
            stage3_background_extraction.run_stage3_background_extraction(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertTrue((self.process_dir / "stage3_bgremoved.fit").exists())

    def test_stage4_color_calibration_smoke(self) -> None:
        self.pipeline.cfg.stage4_platesolve_enabled = False
        self.pipeline._read_fits_metadata = lambda *_args: {}
        with (
            patch.object(stage4_color_calibration, "_stage4_header_metadata", return_value={}),
            patch.object(stage4_color_calibration, "_stage4_image_geometry", return_value={"current_shape": {}}),
        ):
            stage4_color_calibration.run_stage4_color_calibration(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertEqual(self.pipeline.color_calibration_report["method"], "PRESERVE_INPUT")
        self.assertEqual(
            self.pipeline.color_calibration_report["local_fallback"]["status"],
            "retired",
        )
        self.assertTrue(self.pipeline.color_calibration_report["requires_review"])
        self.assertEqual(
            self.pipeline.color_calibration_report["channel_policy"]["kind"],
            "broadband_rgb_osc",
        )

    def test_stage5_linear_denoise_smoke(self) -> None:
        (self.process_dir / "stage4_color.fit").write_bytes(b"stage4")
        self.pipeline._export_linear_intermediate = lambda: True
        self.pipeline._active_policy_name = lambda: "generic_low_snr_safe"
        self.pipeline._active_target_type = lambda: "generic_low_snr_safe"
        self.pipeline._find_plugin_script = lambda _paths: None
        with (
            patch.object(stage5_linear_denoise, "_run_stage5_rl_deconvolution", return_value=False),
            patch.object(stage5_linear_denoise, "_run_builtin_linear_denoise", return_value=True),
        ):
            stage5_linear_denoise.run_stage5_linear_denoise(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertTrue((self.process_dir / "stage5_linear.fit").exists())

    def test_stage7_stretching_smoke(self) -> None:
        self.pipeline._run_stage7_stretching_candidates = lambda: (
            True,
            False,
            ["local candidates"],
            "asinh",
        )

        stage7_stretching.run_stage7_stretching(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage7_stretched.fit").exists())

    def test_stage6_star_separation_source_is_always_linear(self) -> None:
        (self.process_dir / "stage5_linear.fit").touch()
        (self.process_dir / "stage4_color.fit").write_bytes(b"stage4")
        (self.process_dir / "stage5_input_linear.fit").write_bytes(b"baseline")
        input_lineage = stage5_handoff.freeze_stage5_input_lineage(
            self.pipeline,
            upstream_loaded=True,
            baseline_saved=True,
        )
        handoff = stage5_handoff.freeze_stage5_handoff(
            self.pipeline,
            origin=stage5_handoff.CURRENT_RUN_ORIGIN,
            stage_status="ok",
            deconvolution_integrity_ok=True,
            denoise_integrity_ok=True,
            input_lineage=input_lineage,
        )

        source, mode, records = stage6_star_separation._prepare_star_separation_source(
            self.pipeline
        )

        self.assertEqual(source, "stage5_linear")
        self.assertEqual(mode, "linear_star_separation")
        self.assertEqual(self.pipeline.stretched_name, "stage5_linear")
        self.assertFalse(any(command[0] == "asinh" for command in self.pipeline.commands))
        self.assertEqual(records[0]["source_stem"], "stage5_linear")
        self.assertEqual(records[0]["source_lineage"], handoff)

    def test_stage6_rejects_older_checkpoint_without_stage5_handoff(self) -> None:
        (self.process_dir / "stage5_graxpert_deconv.fit").touch()
        (self.process_dir / "stage4_color.fit").touch()
        (self.process_dir / "working.fit").touch()
        reports = {}
        metadata = {}
        self.pipeline._write_stage_json = (
            lambda name, payload: reports.__setitem__(name, payload)
        )
        original_record_stage = self.pipeline._record_stage

        def capture_record_stage(*args, **kwargs):
            metadata.update(kwargs)
            return original_record_stage(*args, **kwargs)

        self.pipeline._record_stage = capture_record_stage

        with self.assertRaises(stage6_star_separation.SirilError):
            stage6_star_separation.run_stage6_star_separation(self.pipeline)

        report = reports["stage6_starless_quality.json"]
        self.assertEqual(report["mode"], "upstream_source_rejected")
        self.assertEqual(
            report["reason_code"],
            stage5_handoff.REASON_LINEAGE_UNVERIFIED,
        )
        self.assertEqual(self.pipeline.results[-1][1], "failed")
        self.assertEqual(metadata["components"]["input_source"]["status"], "failed")
        self.assertTrue(metadata["components"]["input_source"]["fatal"])
        self.assertTrue(
            metadata["details"]["stage8_handoff"]["restricted_downstream"]
        )
        self.assertFalse((self.process_dir / "stage6_passthrough.fit").exists())
        self.assertFalse(
            any(
                command and command[0] in {"script", "pyscript", "syqon"}
                for command in self.pipeline.commands
            )
        )

    def test_stage8_nebula_enhancement_smoke(self) -> None:
        self.pipeline.cfg.stage8_masked_enhancement_enabled = True
        self.pipeline._find_external_fit = lambda _names: None
        self.pipeline._ai_stage_advisory_enabled = lambda _name: False
        self.pipeline._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": True,
            "reasons": ["unsafe starless input"],
        }

        stage8_nebula_enhancement.run_stage8_nebula_enhancement(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertEqual(self.pipeline._stage8_final_source, "stage8_input_starless")
        self.assertEqual(self.pipeline._stage8_final_quality, "skipped")

    def test_stage9_star_remixing_smoke(self) -> None:
        self.pipeline._stage9_bad_starless_reason = lambda: "poor starless"
        (self.process_dir / "stage7_review_with_stars.fit").touch()

        stage9_star_remixing.run_stage9_star_remixing(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertTrue(self.pipeline._stage9_bypassed_bad_starless)
        self.assertTrue(self.pipeline._stage9_output_contains_stars)

    def test_stage10_export_smoke(self) -> None:
        (self.process_dir / "stage9_remixed.fit").touch()
        self.pipeline._find_plugin_script = lambda _paths: None
        self.pipeline._classic_cosmic_clarity_args = lambda *_args: None
        self.pipeline._run_cosmic_clarity_native_denoise_fallback = lambda _label: None
        self.pipeline._run_siril_scunet_denoise_fallback = lambda *_args: None
        self.pipeline._result_output_basename = lambda: "result_processed"
        with patch.object(
            stage10_export,
            "export_final_outputs",
            side_effect=lambda _cmd, _log, **kwargs: (
                kwargs["status"],
                kwargs["messages"],
            ),
        ):
            stage10_export.run_stage10_export(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][0], "阶段 10: 最终降噪与导出")
        self.assertIn(self.pipeline.results[-1][1], {"ok", "degraded"})

    def test_stage10_denoise_plan_selects_chroma_for_color_dominant_noise(self) -> None:
        plan = stage10_export._select_stage10_denoise_plan(
            {
                "chroma_noise_score": 0.431,
                "bg_std": 0.003,
                "background_mottling_score": 0.144,
            },
            color_input=True,
        )

        self.assertEqual(plan["selected_mode"], "chroma")
        self.assertEqual(plan["cosmic_clarity_mode"], "full")

    def test_stage10_saturation_is_scaled_by_stage9_local_color_risk(self) -> None:
        pipeline = SimpleNamespace(
            cfg=PipelineConfig(),
            _stage9_selected_remix_quality={
                "accepted": True,
                "metrics": {
                    "local_quality_status": "ok",
                    "local_color_risk_score": 0.75,
                },
            },
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
        )

        guarded, report = (
            stage10_export._stage10_stage9_local_color_saturation_guard(
                pipeline,
                0.12,
            )
        )

        self.assertAlmostEqual(guarded, 0.03, places=6)
        self.assertAlmostEqual(report["saturation_factor"], 0.25, places=6)
        self.assertTrue(report["applied"])
        self.assertEqual(report["reason"], "stage9_local_color_risk")

    def test_stage10_saturation_is_skipped_when_required_stars_fail_gate(self) -> None:
        pipeline = SimpleNamespace(
            cfg=PipelineConfig(),
            _stage9_selected_remix_quality=None,
            _stage9_stars_required=True,
            _stage9_stars_applied=False,
        )

        guarded, report = (
            stage10_export._stage10_stage9_local_color_saturation_guard(
                pipeline,
                0.12,
            )
        )

        self.assertEqual(guarded, 0.0)
        self.assertEqual(report["local_color_risk_score"], 1.0)
        self.assertEqual(report["reason"], "stage9_required_stars_not_applied")

    def test_stage10_denoise_plan_selects_full_for_combined_noise(self) -> None:
        plan = stage10_export._select_stage10_denoise_plan(
            {
                "chroma_noise_score": 0.50,
                "bg_std": 0.025,
                "background_mottling_score": 0.20,
            },
            color_input=True,
        )

        self.assertEqual(plan["selected_mode"], "full")
        self.assertEqual(plan["cosmic_clarity_mode"], "full")

    def test_stage10_denoise_plan_skips_when_all_noise_is_below_thresholds(self) -> None:
        plan = stage10_export._select_stage10_denoise_plan(
            {
                "chroma_noise_score": 0.058,
                "bg_std": 0.00084,
                "background_mottling_score": 0.031,
            },
            color_input=True,
        )

        self.assertEqual(plan["selected_mode"], "skip")
        self.assertEqual(plan["cosmic_clarity_mode"], "none")
        self.assertIn("below final-denoise thresholds", plan["reason"])

    def test_stage10_low_noise_input_does_not_retain_full_image_copy(self) -> None:
        pixels = np.full((3, 32, 32), 0.05, dtype=np.float32)
        pipeline = SimpleNamespace(
            cfg=PipelineConfig(),
            log=_Log(),
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: pixels,
            ),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.058,
                "bg_std": 0.00084,
                "background_mottling_score": 0.031,
            },
        )

        snapshot, plan = stage10_export._stage10_denoise_input(pipeline)

        self.assertIsNone(snapshot)
        self.assertEqual(plan["selected_mode"], "skip")

    def test_stage10_active_denoise_freezes_input_for_protected_merge(self) -> None:
        pixels = np.full((3, 32, 32), 0.05, dtype=np.float32)
        pipeline = SimpleNamespace(
            cfg=PipelineConfig(),
            log=_Log(),
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: pixels,
            ),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.20,
                "bg_std": 0.025,
                "background_mottling_score": 0.20,
            },
        )

        snapshot, plan = stage10_export._stage10_denoise_input(pipeline)

        self.assertEqual(plan["selected_mode"], "full")
        self.assertIsNotNone(snapshot)
        self.assertIsNot(snapshot, pixels)
        np.testing.assert_array_equal(snapshot, pixels)

    def test_stage10_star_protected_merge_restores_hard_core(self) -> None:
        original = np.full((3, 12, 12), 0.8, dtype=np.float32)
        denoised = np.full((3, 12, 12), 0.2, dtype=np.float32)
        protection = np.zeros((12, 12), dtype=np.float32)
        protection[5, 5] = 1.0
        protection[5, 6] = 0.5

        merged = stage10_export._star_protected_denoised_image(
            original,
            denoised,
            protection,
        )

        np.testing.assert_array_equal(merged[:, 5, 5], original[:, 5, 5])
        np.testing.assert_array_equal(merged[:, 0, 0], denoised[:, 0, 0])
        np.testing.assert_allclose(merged[:, 5, 6], 0.5, atol=1e-7)

    def test_stage10_star_protection_uses_validated_stage9_catalog(self) -> None:
        weak_core = np.zeros((32, 32), dtype=bool)
        bright_core = np.zeros((32, 32), dtype=bool)
        weak_core[7, 8] = True
        bright_core[20, 22] = True
        pipeline = SimpleNamespace(
            cfg=PipelineConfig(),
            _stage9_stars_applied=True,
            _stage9_star_reference_catalog={
                "status": "ok",
                "source_matched": True,
                "_weak_core_mask": weak_core,
                "_bright_core_mask": bright_core,
            },
        )

        mask, report = stage10_export._build_stage10_star_protection_mask(
            pipeline,
            np.zeros((3, 32, 32), dtype=np.float32),
        )

        self.assertIsNotNone(mask)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(float(mask[7, 8]), 1.0)
        self.assertEqual(float(mask[20, 22]), 1.0)
        self.assertLessEqual(
            report["protected_coverage"],
            report["coverage_max"],
        )

    def test_stage10_star_protection_fails_closed_without_catalog(self) -> None:
        pipeline = SimpleNamespace(
            cfg=PipelineConfig(),
            _stage9_stars_applied=True,
            _stage9_star_reference_catalog={
                "status": "unavailable",
                "reason": "catalog gate rejected",
            },
        )

        mask, report = stage10_export._build_stage10_star_protection_mask(
            pipeline,
            np.zeros((3, 32, 32), dtype=np.float32),
        )

        self.assertIsNone(mask)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["reason"], "catalog gate rejected")

    def test_stage10_low_chroma_still_uses_full_for_background_noise(self) -> None:
        plan = stage10_export._select_stage10_denoise_plan(
            {
                "chroma_noise_score": 0.058,
                "bg_std": 0.021,
                "background_mottling_score": 0.20,
            },
            color_input=True,
        )

        self.assertEqual(plan["selected_mode"], "full")
        self.assertEqual(plan["cosmic_clarity_mode"], "full")

    def test_stage10_missing_metrics_keeps_balanced_full_fallback(self) -> None:
        plan = stage10_export._select_stage10_denoise_plan(
            {"chroma_noise_score": 0.01},
            color_input=True,
        )

        self.assertEqual(plan["selected_mode"], "full")
        self.assertFalse(plan["input_metrics_available"])

    def test_stage10_non_color_input_keeps_full_denoise(self) -> None:
        plan = stage10_export._select_stage10_denoise_plan(
            {
                "chroma_noise_score": 0.0,
                "bg_std": 0.0002,
                "background_mottling_score": 0.02,
            },
            color_input=False,
        )

        self.assertEqual(plan["selected_mode"], "full")
        self.assertEqual(plan["reason"], "non-color input")

    def test_stage10_denoise_plan_selects_separate_for_severe_color_noise(self) -> None:
        plan = stage10_export._select_stage10_denoise_plan(
            {
                "chroma_noise_score": 0.80,
                "bg_std": 0.004,
                "background_mottling_score": 0.20,
            },
            color_input=True,
        )

        self.assertEqual(plan["selected_mode"], "separate")
        self.assertEqual(plan["cosmic_clarity_mode"], "separate")

    def test_stage10_chroma_only_merge_restores_input_luminance(self) -> None:
        original = np.array(
            [
                [[0.20, 0.30], [0.40, 0.50]],
                [[0.18, 0.28], [0.38, 0.48]],
                [[0.16, 0.26], [0.36, 0.46]],
            ],
            dtype=np.float32,
        )
        denoised = np.array(
            [
                [[0.22, 0.29], [0.42, 0.47]],
                [[0.20, 0.29], [0.40, 0.47]],
                [[0.18, 0.29], [0.38, 0.47]],
            ],
            dtype=np.float32,
        )

        merged = stage10_export._chroma_only_denoised_image(original, denoised)
        original_luma = (
            0.2126 * original[0] + 0.7152 * original[1] + 0.0722 * original[2]
        )
        merged_luma = 0.2126 * merged[0] + 0.7152 * merged[1] + 0.0722 * merged[2]

        np.testing.assert_allclose(merged_luma, original_luma, atol=1e-6)
        self.assertFalse(np.allclose(merged, original))


if __name__ == "__main__":
    unittest.main()
