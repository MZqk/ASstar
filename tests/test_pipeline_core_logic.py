#!/usr/bin/env python3
"""Unit tests for pipeline configuration, tuning formulas, and target matching."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
PIPELINE_MODULE_PATH = PIPELINE_DIR / "seestar_Superimpose.py"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _install_sirilpy_stub() -> None:
    if "sirilpy" in sys.modules:
        return
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")
    enums = types.ModuleType("sirilpy.enums")

    class SirilError(Exception):
        pass

    class SirilConnectionError(SirilError):
        pass

    class CommandError(SirilError):
        pass

    class DataError(SirilError):
        pass

    class SirilInterface:
        pass

    class CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    sirilpy.SirilInterface = SirilInterface
    exceptions.SirilError = SirilError
    exceptions.SirilConnectionError = SirilConnectionError
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    enums.CommandStatus = CommandStatus
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums


def _load_pipeline_module() -> Any:
    _install_sirilpy_stub()
    spec = importlib.util.spec_from_file_location(
        "seestar_pipeline_core_test_module",
        PIPELINE_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {PIPELINE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = _load_pipeline_module()


class ClampConfigTests(unittest.TestCase):
    def test_clamps_low_values_and_preserves_input(self) -> None:
        cfg = pipeline.PipelineConfig()
        cfg.crop_margin = -1.0
        cfg.bg_samples = -100
        cfg.denoise_mod = -1.0
        cfg.stage5_rl_psf_kernel_size = 2
        cfg.stage7_edge_black_warn = -1.0
        cfg.stage7_edge_black_high = -1.0
        cfg.stage7_conservative_asinh_stretch = -1.0
        cfg.stage7_ultra_conservative_asinh_stretch = -1.0
        cfg.stage7_soft_starless_asinh_stretch = -1.0

        tuned = pipeline.clamp_config(cfg)

        self.assertEqual(cfg.crop_margin, -1.0)
        self.assertEqual(tuned.crop_margin, 0.0)
        self.assertEqual(tuned.bg_samples, 12)
        self.assertEqual(tuned.denoise_mod, 0.2)
        self.assertEqual(tuned.stage5_rl_psf_kernel_size, 9)
        self.assertEqual(tuned.stage7_edge_black_warn, 0.04)
        self.assertEqual(tuned.stage7_edge_black_high, 0.04)
        self.assertEqual(tuned.stage7_conservative_asinh_stretch, 1.6)
        self.assertEqual(tuned.stage7_ultra_conservative_asinh_stretch, 1.2)
        self.assertEqual(tuned.stage7_soft_starless_asinh_stretch, 1.05)

    def test_clamps_high_values_and_enforces_related_limits(self) -> None:
        cfg = pipeline.PipelineConfig()
        cfg.crop_margin = 1.0
        cfg.bg_samples = 100
        cfg.denoise_safety_max = 0.40
        cfg.denoise_mod = 1.0
        cfg.stage5_rl_psf_kernel_size = 98
        cfg.stage7_edge_black_warn = 0.20
        cfg.stage7_edge_black_high = 0.10
        cfg.stage7_conservative_asinh_stretch = 2.2
        cfg.stage7_ultra_conservative_asinh_stretch = 3.0
        cfg.stage7_soft_starless_asinh_stretch = 3.0

        tuned = pipeline.clamp_config(cfg)

        self.assertEqual(tuned.crop_margin, 0.06)
        self.assertEqual(tuned.bg_samples, 32)
        self.assertEqual(tuned.denoise_mod, 0.40)
        self.assertEqual(tuned.stage5_rl_psf_kernel_size, 99)
        self.assertEqual(tuned.stage7_edge_black_high, 0.20)
        self.assertEqual(tuned.stage7_ultra_conservative_asinh_stretch, 2.2)
        self.assertEqual(tuned.stage7_soft_starless_asinh_stretch, 2.2)

    def test_exact_boundary_values_are_stable(self) -> None:
        cfg = pipeline.PipelineConfig(
            crop_margin=0.06,
            bg_samples=12,
            bg_tolerance=1.8,
            stage5_rl_psf_kernel_size=99,
            asinh_stretch=1.6,
            star_intensity=1.05,
            final_saturation=0.05,
        )

        tuned = pipeline.clamp_config(cfg)

        self.assertEqual(tuned.crop_margin, 0.06)
        self.assertEqual(tuned.bg_samples, 12)
        self.assertEqual(tuned.bg_tolerance, 1.8)
        self.assertEqual(tuned.stage5_rl_psf_kernel_size, 99)
        self.assertEqual(tuned.asinh_stretch, 1.6)
        self.assertEqual(tuned.star_intensity, 1.05)
        self.assertEqual(tuned.final_saturation, 0.05)


class AutoTuneConfigTests(unittest.TestCase):
    def test_feature_formula_derivation(self) -> None:
        cfg = pipeline.PipelineConfig()
        features = pipeline.ImageFeatures(
            bg_std=0.05,
            star_density=0.005,
            diffuse_ratio=0.60,
            core_brightness_ratio=0.08,
            edge_black_ratio=0.30,
            red_dominance=1.25,
            blue_dominance=1.25,
            object_area_ratio=0.40,
        )

        tuned, result = pipeline.auto_tune_config(
            cfg,
            pipeline.TargetType.GALAXY,
            features,
        )

        self.assertAlmostEqual(tuned.crop_margin, 0.025)
        self.assertEqual(tuned.bg_samples, 25)
        self.assertAlmostEqual(tuned.bg_tolerance, 0.875)
        self.assertAlmostEqual(tuned.bg_smooth, 0.69)
        self.assertTrue(tuned.denoise_enabled)
        self.assertAlmostEqual(tuned.denoise_mod, 0.32)
        self.assertAlmostEqual(tuned.asinh_stretch, 2.09)
        self.assertAlmostEqual(tuned.asinh_offset, 0.00175)
        self.assertAlmostEqual(tuned.ghs_shadowsclip, -2.65)
        self.assertAlmostEqual(tuned.ghs_stretchamount, 1.85)
        self.assertAlmostEqual(tuned.nebula_saturation, 0.212)
        self.assertAlmostEqual(tuned.star_intensity, 0.972)
        self.assertAlmostEqual(tuned.star_fallback_intensity, 0.912)
        self.assertAlmostEqual(tuned.final_saturation, 0.082)
        self.assertEqual(result.target_type, pipeline.TargetType.GALAXY)
        reasons = {name: reason for name, _old, _new, reason in result.changed_params}
        self.assertEqual(reasons["crop_margin"], "feature_formula:crop_margin")
        self.assertEqual(reasons["final_saturation"], "feature_formula:final_saturation")

    def test_extreme_features_are_safely_clamped(self) -> None:
        tuned, result = pipeline.auto_tune_config(
            pipeline.PipelineConfig(),
            pipeline.TargetType.UNKNOWN,
            pipeline.ImageFeatures(
                bg_std=10.0,
                star_density=10.0,
                diffuse_ratio=10.0,
                core_brightness_ratio=10.0,
                edge_black_ratio=10.0,
                red_dominance=10.0,
                blue_dominance=10.0,
                object_area_ratio=10.0,
            ),
        )

        self.assertLessEqual(tuned.crop_margin, 0.06)
        self.assertGreaterEqual(tuned.bg_samples, 12)
        self.assertLessEqual(tuned.denoise_mod, tuned.denoise_safety_max)
        self.assertLessEqual(tuned.star_intensity, 1.05)
        self.assertIn("UNKNOWN", " ".join(result.notes))


class TargetMatchingTests(unittest.TestCase):
    def test_path_keyword_matching_handles_compact_catalog_numbers(self) -> None:
        target = pipeline.detect_target_type(
            Path("/data/IC 434/session/stacked.fit")
        )
        self.assertEqual(target, pipeline.TargetType.EMISSION_NEBULA)

    def test_more_specific_planetary_nebula_precedes_generic_text(self) -> None:
        target = pipeline.detect_target_type(
            Path("/data/M57_ring_nebula/galaxy_stack.fit")
        )
        self.assertEqual(target, pipeline.TargetType.PLANETARY_NEBULA)

    def test_feature_profiles_cover_all_inference_branches(self) -> None:
        cases = [
            (
                pipeline.ImageFeatures(
                    edge_black_ratio=0.01,
                    bg_median=0.4,
                    object_area_ratio=0.8,
                    star_density=0.001,
                ),
                pipeline.TargetType.PLANETARY,
            ),
            (
                pipeline.ImageFeatures(
                    diffuse_ratio=0.7,
                    object_area_ratio=0.3,
                    red_dominance=1.3,
                ),
                pipeline.TargetType.EMISSION_NEBULA,
            ),
            (
                pipeline.ImageFeatures(
                    diffuse_ratio=0.7,
                    object_area_ratio=0.3,
                    blue_dominance=1.3,
                ),
                pipeline.TargetType.REFLECTION_NEBULA,
            ),
            (
                pipeline.ImageFeatures(
                    object_area_ratio=0.25,
                    core_brightness_ratio=0.08,
                    diffuse_ratio=0.4,
                    star_density=0.002,
                ),
                pipeline.TargetType.GALAXY,
            ),
            (
                pipeline.ImageFeatures(
                    object_area_ratio=0.1,
                    core_brightness_ratio=0.1,
                    diffuse_ratio=0.3,
                    red_dominance=1.2,
                ),
                pipeline.TargetType.PLANETARY_NEBULA,
            ),
            (
                pipeline.ImageFeatures(
                    object_area_ratio=0.1,
                    star_density=0.006,
                    diffuse_ratio=0.1,
                ),
                pipeline.TargetType.CLUSTER,
            ),
            (
                pipeline.ImageFeatures(
                    object_area_ratio=0.4,
                    star_density=0.006,
                    diffuse_ratio=0.1,
                ),
                pipeline.TargetType.WIDEFIELD,
            ),
        ]

        for features, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    pipeline._infer_target_type_from_features(features),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
