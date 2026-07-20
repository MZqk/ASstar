#!/usr/bin/env python3
"""Smoke-level integration tests for all ten pipeline stage entry points."""

from __future__ import annotations

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
    PipelineCheckpoint,
    PipelineConfig,
    PipelineStage,
)
import stage9_quality  # noqa: E402
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
    stage6_stretching,
    stage7_star_separation,
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
        self.assertEqual(cfg.stage9_weak_star_screen_intensity_min, 0.40)
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
        self.assertEqual(cfg.stage7_starmask_diffuse_residual_ratio_max, 0.08)
        self.assertEqual(cfg.stage10_stage9_local_color_risk_strength, 1.0)

    def test_formal_stage_labels_are_unique_and_contiguous(self):
        labels = [stage.label for stage in PipelineStage]

        self.assertEqual(len(labels), 11)
        self.assertEqual(len(set(labels)), 11)
        for number, label in enumerate(labels, start=1):
            self.assertTrue(label.startswith(f"阶段 {number}:"))

        self.assertNotIn(
            PipelineCheckpoint.PRE_STARLESS_COMPATIBILITY_GATE.label,
            labels,
        )

    def test_stage9_screen_blend_preserves_highlight_headroom(self):
        base = np.full((3, 4, 4), 0.80, dtype=np.float32)
        stars = np.full((3, 4, 4), 0.80, dtype=np.float32)

        mixed = stage9_quality.screen_blend(base, stars, 1.0)

        self.assertAlmostEqual(float(mixed[0, 0, 0]), 0.96, places=5)
        self.assertLess(float(mixed.max()), 1.0)

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

    def test_stage9_gate_rejects_extreme_chromatic_additions(self):
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
        self.assertTrue(
            any(
                issue.startswith("chromatic_star_addition_ratio")
                for issue in report["issues"]
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
                self.assertTrue(
                    any(issue.startswith(expected_metric) for issue in report["issues"]),
                    report["issues"],
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
        rng = np.random.default_rng(42)
        yy, xx = np.mgrid[:128, :128]
        base_gray = 0.04 + rng.normal(0.0, 0.0005, (128, 128))
        base = np.stack([base_gray, base_gray, base_gray]).astype(np.float32)
        stars = np.zeros_like(base)
        for center_y, center_x, amplitude in (
            (24, 31, 0.25),
            (45, 90, 0.40),
            (76, 54, 0.18),
            (102, 109, 0.30),
        ):
            profile = amplitude * np.exp(
                -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * 1.2**2)
            )
            stars += np.stack([profile, profile * 0.9, profile * 0.8]).astype(
                np.float32
            )
        candidate = stage9_quality.screen_blend(base, stars, 1.0)

        report = stage9_quality.assess_remix(
            base,
            candidate,
            cfg,
            attempt="compact_stars",
            formula="screen",
            star_reference=stage9_quality.build_star_reference_catalog(
                stars,
                cfg,
            ),
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
        bright = np.stack([profile * 0.45, profile * 0.38, profile * 0.30]).astype(np.float32)

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
            0.45,
            0.001,
            0.22,
            1000.0,
        )
        self.assertGreater(faint_direct, bright_direct)
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
        self.assertAlmostEqual(plan["predicted_faint"], 0.26, places=3)
        self.assertAlmostEqual(plan["predicted_mid"], 0.50, places=3)
        self.assertAlmostEqual(plan["predicted_bright"], 0.75, places=3)
        self.assertLessEqual(plan["predicted_peak"], 0.9001)

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
            report["metrics"]["star_aperture_recovery_ratio"],
            cfg.stage9_star_aperture_recovery_ratio_min,
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
        self.assertTrue(disabled_gate["accepted"])
        self.assertFalse(disabled_gate["gate_enabled"])
        self.assertEqual(disabled_gate["metrics"]["star_recovery_ratio"], 0.0)

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
        self.assertGreater(
            catalog["source_component_density_per_megapixel"],
            catalog["source_component_density_max"],
        )
        self.assertGreater(
            catalog["source_single_pixel_component_ratio"],
            catalog["source_single_pixel_component_ratio_max"],
        )
        self.assertIn("source_star_catalog_contamination_risk", catalog["reason"])
        self.assertEqual(catalog["source_detail_percentile"], 99.5)
        self.assertGreater(len(catalog["source_detail_attempts"]), 1)

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
        self.assertTrue(catalog["source_detail_adaptive_retry"])
        self.assertEqual(catalog["source_detail_percentile_requested"], 98.0)
        self.assertEqual(catalog["source_detail_percentile"], 99.5)
        self.assertEqual(catalog["matched_component_count"], 60)
        self.assertTrue(catalog["source_detail_attempts"][0]["contamination_risk"])
        self.assertFalse(
            catalog["source_detail_attempts"][-1]["contamination_risk"]
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
        last_attempt = catalog["source_detail_attempts"][-1]
        self.assertFalse(last_attempt["density_limit_exceeded"])
        self.assertTrue(last_attempt["single_pixel_limit_exceeded"])
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

        self.assertEqual(plan["status"], "ok")
        self.assertTrue(np.any(stars[:, ~support] > 0.0))
        self.assertTrue(np.all(compact[:, ~support] == 0.0))
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
        cfg = PipelineConfig(stage7_starmask_diffuse_residual_ratio_max=0.05)
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

        quality = stage7_star_separation._apply_starmask_cleanup_hard_gate(
            {"status": "ok", "issues": [], "derived": {}},
            {"metrics": metrics},
        )
        self.assertEqual(quality["status"], "poor")
        self.assertTrue(quality["derived"]["starmask_cleanup_hard_failed"])

    def test_starmask_cleanup_rolls_back_before_write_when_compact_stars_are_overcleaned(self):
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

        self.assertEqual(result["status"], "rolled_back")
        self.assertFalse(writes)
        self.assertNotIn(("save", "starmask"), commands)

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
            starmask_path = process_dir / "starmask.fit"
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
        self.assertIn(("save", "starmask"), commands)
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
        self.pre_starless_gate_report = {}
        self._stage1_input_mode = "stacked"
        self._last_plugin_script_error = None
        self._last_sasp_stage8_api_error = None
        self._last_aberration_api_error = None
        self._last_scunet_fallback_error = None

    def _record_stage(self, name: str, status: str, duration: float, message: str = "") -> None:
        self.results.append((name, status, duration, message))

    def cmd_with_check(self, *args, **_kwargs) -> None:
        self.commands.append(tuple(args))

    def _save_stage_output(self, stem: str) -> bool:
        (self.process_dir / f"{stem}.fit").touch()
        return True

    def _write_stage_json(self, *_args, **_kwargs) -> None:
        return

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
            patch.object(stage2_view_correction, "_detect_auto_edge_crop", return_value=(None, "no crop")),
            patch.object(stage2_view_correction, "_edge_color_artifact_crop", return_value=""),
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage2_corrected.fit").exists())

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
            patch.object(
                stage4_color_calibration,
                "_stage4_local_color_fallback",
                return_value=(True, "LOCAL_STAR_WB", "", 0.6, {}, "local fallback"),
            ),
        ):
            stage4_color_calibration.run_stage4_color_calibration(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(self.pipeline.color_calibration_report["method"], "LOCAL_STAR_WB")

    def test_stage5_linear_denoise_smoke(self) -> None:
        self.pipeline._export_linear_intermediate = lambda: True
        self.pipeline._active_policy_name = lambda: "generic_low_snr_safe"
        self.pipeline._active_target_type = lambda: "generic_low_snr_safe"
        with (
            patch.object(stage5_linear_denoise, "_run_stage5_rl_deconvolution", return_value=False),
            patch.object(stage5_linear_denoise, "_run_builtin_linear_denoise", return_value=True),
        ):
            stage5_linear_denoise.run_stage5_linear_denoise(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage5_linear.fit").exists())

    def test_stage6_stretching_smoke(self) -> None:
        self.pipeline._ai_stage_advisory_enabled = lambda _name: False
        self.pipeline._run_stage6_ai_stretching = lambda allow_ai: (
            True,
            False,
            [f"allow_ai={allow_ai}"],
            "asinh",
        )

        stage6_stretching.run_stage6_stretching(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage7_stretched.fit").exists())

    def test_stage7_star_separation_smoke(self) -> None:
        (self.process_dir / "stage5_linear.fit").touch()
        (self.process_dir / "stage7_conservative_asinh.fit").touch()
        self.pipeline.pre_starless_gate_report = {
            "ready_for_starless": False,
            "reason": ["unsafe input"],
            "recommended_starless_input": "stage7_conservative_asinh.fit",
        }
        self.pipeline.cfg.stage7_skip_unready_starless = True
        self.pipeline._stage7_update_star_remix_from_quality = lambda _record: {}
        self.pipeline._export_sasp_exchange_files = lambda: None
        stage7_star_separation.run_stage7_star_separation(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "skipped")
        self.assertTrue(self.pipeline._stage7_starless_skipped)
        self.assertIn(("load", "stage5_linear"), self.pipeline.commands)
        self.assertNotIn(("load", "stage7_conservative_asinh"), self.pipeline.commands)

    def test_stage7_legacy_mild_prestretch_mode_keeps_linear_input(self) -> None:
        (self.process_dir / "stage5_linear.fit").touch()
        self.pipeline.cfg.star_separation_mode = "mild_prestretch_star_separation"

        source, mode, records = stage7_star_separation._prepare_star_separation_source(
            self.pipeline
        )

        self.assertEqual(source, "stage5_linear")
        self.assertEqual(mode, "linear_star_separation")
        self.assertEqual(self.pipeline.stretched_name, "stage5_linear")
        self.assertFalse(any(command[0] == "asinh" for command in self.pipeline.commands))
        self.assertEqual(records[-1]["status"], "ignored_compatibility")

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
        self.pipeline._stage9_review_safe_source = lambda: "stage7_stretched"

        stage9_star_remixing.run_stage9_star_remixing(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertTrue(self.pipeline._stage9_bypassed_bad_starless)

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
