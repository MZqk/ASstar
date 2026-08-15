#!/usr/bin/env python3
"""Tests for Stage 7 conditional cand_a stretch algorithms."""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _install_fake_sirilpy() -> None:
    if "sirilpy.exceptions" in sys.modules:
        return
    package = types.ModuleType("sirilpy")
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

    package.SirilInterface = SirilInterface
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    exceptions.SirilError = SirilError
    exceptions.SirilConnectionError = SirilConnectionError
    enums.CommandStatus = CommandStatus
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums


_install_fake_sirilpy()

import stage7_stretch_metrics as stretch_metrics  # noqa: E402
from models import PipelineConfig  # noqa: E402
from processing_parameters import SPECS_BY_FIELD  # noqa: E402
from stage6_services import Stage6ServiceMixin  # noqa: E402


class _StretchProcessor(Stage6ServiceMixin):
    def __init__(self, target_type: str) -> None:
        self.cfg = PipelineConfig()
        self.pipeline_policy = {}
        self.target_type = target_type
        self._task_manual_override_fields = ()

    def _active_target_type(self) -> str:
        return self.target_type

    @staticmethod
    def _short_text(value, _limit: int) -> str:
        return str(value)


class Stage7ConditionalStretchTests(unittest.TestCase):
    @staticmethod
    def _profile(**overrides):
        profile = {
            "status": "available",
            "trusted_background": True,
            "trusted_galaxy_roi": True,
            "background_median": 0.012,
            "background_sigma": 0.0015,
            "subject_mask_available": True,
            "subject_p90": 0.025,
            "p99_9": 0.160,
            "faint_signal_snr_proxy": 4.0,
            "stretch_noise_regime": "medium_snr_proxy",
        }
        profile.update(overrides)
        return profile

    def test_config_and_processing_registry_expose_bounded_controls(self) -> None:
        cfg = PipelineConfig()
        self.assertTrue(cfg.stage7_iterative_masked_mtf_enabled)
        self.assertEqual(cfg.stage7_iterative_masked_mtf_iterations, 16)
        self.assertTrue(cfg.stage7_dual_stage_mtf_ghs_enabled)
        self.assertEqual(cfg.stage7_dual_stage_weak_snr_max, 8.0)
        self.assertEqual(cfg.stage7_dual_stage_ghs_search_steps, 47)
        for field in (
            "stage7_iterative_masked_mtf_enabled",
            "stage7_iterative_masked_mtf_iterations",
            "stage7_dual_stage_mtf_ghs_enabled",
            "stage7_dual_stage_weak_snr_max",
            "stage7_dual_stage_subject_p90_min",
            "stage7_dual_stage_subject_p90_max",
            "stage7_dual_stage_ghs_b",
            "stage7_dual_stage_ghs_d_min",
            "stage7_dual_stage_ghs_d_max",
            "stage7_dual_stage_ghs_search_steps",
            "stage7_conditional_lut_max_derivative",
        ):
            self.assertIn(field, SPECS_BY_FIELD)

    def test_source_profile_uses_frozen_background_and_galaxy_roi(self) -> None:
        rng = np.random.default_rng(31)
        height = width = 128
        yy, xx = np.mgrid[:height, :width]
        radius2 = (xx - 64) ** 2 + (yy - 64) ** 2
        galaxy = np.exp(-radius2 / 650.0).astype(np.float32)
        gray = np.clip(
            0.010
            + rng.normal(0.0, 0.00045, size=(height, width))
            + 0.0040 * galaxy,
            0.0,
            1.0,
        ).astype(np.float32)
        image = np.stack(
            [gray * 1.02, gray, gray * 0.98],
            axis=0,
        )
        galaxy_mask = (galaxy > 0.16).astype(np.float32)
        background_mask = (galaxy < 0.08).astype(np.float32)
        masks = {
            "background_mask": background_mask,
            "core_mask": np.zeros_like(gray),
            "nebula_mask": np.zeros_like(gray),
            "faint_nebula_mask": np.zeros_like(gray),
            "galaxy_signal_mask": galaxy_mask,
        }
        sampling = {
            "status": "available",
            "method": "frozen_signal_excluded_background_mask_v2",
            "candidate_independent": True,
            "coverage_gt_0_50": float(np.mean(background_mask > 0.50)),
            "galaxy_signal_exclusion": {
                "applicable": True,
                "available": True,
                "coverage": float(np.mean(galaxy_mask > 0.12)),
            },
        }

        profile = stretch_metrics.build_conditional_stretch_source_profile(
            image,
            masks,
            sampling,
        )

        self.assertEqual(profile["status"], "available", profile)
        self.assertTrue(profile["trusted_background"])
        self.assertTrue(profile["trusted_galaxy_roi"])
        self.assertGreaterEqual(profile["background_sample_count"], 64)
        self.assertGreaterEqual(profile["subject_sample_count"], 64)
        self.assertGreater(profile["subject_p90"], profile["background_median"])
        self.assertFalse(profile["physical_snr"])
        self.assertEqual(
            profile["extended_filter"]["method"],
            "reflect_box_mean_v1",
        )

    def test_iterative_masked_mtf_has_stable_golden_lut(self) -> None:
        calibration = stretch_metrics.calibrate_iterative_masked_mtf(
            self._profile(),
            target_background=0.085,
            iterations=16,
            max_derivative=5000.0,
            source_p50=0.018,
            source_p99=0.120,
        )

        self.assertEqual(calibration["status"], "ok", calibration)
        self.assertEqual(
            calibration["lut_contract"]["sha256"],
            "b231fabb0b340443f11d5b0f547cd476c5c4eb690d1be7ef06fd6ce025913f60",
        )
        self.assertTrue(calibration["lut_contract"]["monotonic"])
        self.assertAlmostEqual(
            calibration["resolved"]["mapped_background"],
            0.085,
            places=5,
        )
        lut, contract = stretch_metrics.rebuild_conditional_stretch_lut(
            calibration
        )
        self.assertEqual(contract["sha256"], calibration["lut_contract"]["sha256"])
        self.assertEqual(lut.size, 65536)

        source = np.asarray(
            [
                [[0.01, 0.03], [0.10, 0.40]],
                [[0.02, 0.04], [0.12, 0.45]],
                [[0.03, 0.05], [0.14, 0.50]],
            ],
            dtype=np.float32,
        )
        stretched = stretch_metrics.apply_conditional_linked_rgb_stretch(
            source,
            calibration,
        )
        grid = np.linspace(0.0, 1.0, lut.size, dtype=np.float64)
        expected = np.interp(source.reshape(-1), grid, lut).reshape(source.shape)
        np.testing.assert_allclose(stretched, expected, atol=1e-7)
        self.assertLessEqual(float(np.max(stretched)), 0.995)

        tampered = {**calibration, "lut_contract": dict(calibration["lut_contract"])}
        tampered["lut_contract"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            stretch_metrics.rebuild_conditional_stretch_lut(tampered)

    def test_dual_stage_mtf_ghs_has_stable_golden_lut(self) -> None:
        calibration = stretch_metrics.calibrate_dual_stage_mtf_ghs(
            self._profile(),
            target_background=0.085,
            target_subject_p90=0.230,
            ghs_b=5.0,
            ghs_d_min=0.5,
            ghs_d_max=12.0,
            ghs_search_steps=47,
            max_derivative=5000.0,
            source_p50=0.018,
            source_p99=0.120,
        )

        self.assertEqual(calibration["status"], "ok", calibration)
        self.assertEqual(
            calibration["lut_contract"]["sha256"],
            "bf3130eda7236285b7a4af73636f20e92b129db6a8a0537015984c69760980cb",
        )
        self.assertTrue(calibration["lut_contract"]["monotonic"])
        self.assertLessEqual(
            calibration["lut_contract"]["maximum_derivative"],
            calibration["lut_contract"]["maximum_derivative_limit"],
        )
        self.assertEqual(
            calibration["resolved"]["ghs_stretch_factor_semantics"],
            "ln(D+1)",
        )
        self.assertTrue(calibration["resolved"]["background_target_attained"])
        self.assertTrue(calibration["resolved"]["subject_target_attained"])
        self.assertAlmostEqual(
            calibration["resolved"]["mapped_background"],
            0.085,
            places=5,
        )
        self.assertLessEqual(
            calibration["resolved"]["subject_target_error"],
            calibration["resolved"]["subject_target_tolerance"],
        )
        _lut, rebuilt = stretch_metrics.rebuild_conditional_stretch_lut(
            calibration
        )
        self.assertEqual(rebuilt["sha256"], calibration["lut_contract"]["sha256"])

    def test_iterative_bounds_and_invalid_inputs_fail_closed(self) -> None:
        lower = stretch_metrics.calibrate_iterative_masked_mtf(
            self._profile(),
            target_background=0.085,
            iterations=1,
            max_derivative=5000.0,
            source_p50=0.018,
            source_p99=0.120,
        )
        upper = stretch_metrics.calibrate_iterative_masked_mtf(
            self._profile(),
            target_background=0.085,
            iterations=100,
            max_derivative=5000.0,
            source_p50=0.018,
            source_p99=0.120,
        )
        invalid = stretch_metrics.calibrate_iterative_masked_mtf(
            self._profile(background_median=float("nan")),
            target_background=0.085,
            iterations=16,
            max_derivative=5000.0,
            source_p50=0.018,
            source_p99=0.120,
        )

        self.assertEqual(lower["status"], "ok", lower)
        self.assertEqual(lower["resolved"]["iterations"], 8)
        self.assertEqual(upper["status"], "ok", upper)
        self.assertEqual(upper["resolved"]["iterations"], 32)
        self.assertEqual(invalid["status"], "unavailable")

        infeasible_dual = stretch_metrics.calibrate_dual_stage_mtf_ghs(
            self._profile(subject_p90=0.070),
            target_background=0.085,
            target_subject_p90=0.230,
            ghs_b=5.0,
            ghs_d_min=0.5,
            ghs_d_max=12.0,
            ghs_search_steps=47,
            max_derivative=5000.0,
            source_p50=0.018,
            source_p99=0.120,
        )
        self.assertEqual(infeasible_dual["status"], "unavailable")
        self.assertIn("infeasible", infeasible_dual["reason"])

    def test_strict_routes_replace_only_candidate_a(self) -> None:
        baseline_stats = {
            "p01": 0.004,
            "p50": 0.018,
            "p99": 0.120,
            "max": 0.300,
        }
        preview_stats = {"p50": 0.250, "p99": 0.800}
        for target_type in (
            "globular_cluster",
            "open_cluster",
            "reflection_nebula_cluster",
        ):
            with self.subTest(target_type=target_type):
                processor = _StretchProcessor(target_type)
                candidates, adaptation = (
                    processor._stage7_compact_stretch_candidates(
                        None,
                        {"bg_median": 0.012, "bg_std": 0.0015},
                        baseline_stats,
                        preview_stats,
                        starless_recomposition_planned=False,
                        source_profile=self._profile(),
                        source_stem="stage6_passthrough",
                        star_separation_state="target_bypass",
                    )
                )
                self.assertEqual(len(candidates), 2)
                self.assertEqual(candidates[0]["method"], "iterative_masked_mtf")
                self.assertEqual(candidates[1]["method"], "asinh")
                self.assertEqual(
                    adaptation["candidate_a_replacement"]["status"],
                    "applied",
                )

        for target_type in ("large_galaxy", "small_galaxy"):
            with self.subTest(target_type=target_type):
                processor = _StretchProcessor(target_type)
                candidates, adaptation = (
                    processor._stage7_compact_stretch_candidates(
                        None,
                        {"bg_median": 0.012, "bg_std": 0.0015},
                        baseline_stats,
                        preview_stats,
                        starless_recomposition_planned=True,
                        source_profile=self._profile(),
                        source_stem="stage6_starless",
                        star_separation_state="accepted",
                    )
                )
                self.assertEqual(len(candidates), 2)
                self.assertEqual(candidates[0]["method"], "dual_stage_mtf_ghs")
                self.assertEqual(candidates[1]["method"], "linked_mtf")
                self.assertEqual(
                    adaptation["candidate_a_replacement"]["status"],
                    "applied",
                )
                self.assertGreaterEqual(
                    adaptation["candidate_a_replacement"]["target_subject_p90"],
                    adaptation["candidate_a_replacement"]["calibration"][
                        "resolved"
                    ]["target_background"]
                    * 2.0,
                )
                self.assertEqual(
                    adaptation["preview_calibration"]["candidate_a"][
                        "calibration_method"
                    ],
                    "dual_stage_mtf_ghs",
                )
                serialized = json.dumps(adaptation, allow_nan=False)
                self.assertNotIn('"lut":', serialized)
                self.assertNotIn('"background_mask":', serialized)

    def test_replacement_keeps_cand_b_and_fixed_pool_unchanged(self) -> None:
        baseline_stats = {
            "p01": 0.004,
            "p50": 0.018,
            "p99": 0.120,
            "max": 0.300,
        }
        preview_stats = {"p50": 0.250, "p99": 0.800}
        enabled = _StretchProcessor("large_galaxy")
        enabled_candidates, _enabled_adaptation = (
            enabled._stage7_compact_stretch_candidates(
                None,
                {"bg_median": 0.012, "bg_std": 0.0015},
                baseline_stats,
                preview_stats,
                starless_recomposition_planned=True,
                source_profile=self._profile(),
                source_stem="stage6_starless",
                star_separation_state="accepted",
            )
        )
        disabled = _StretchProcessor("large_galaxy")
        disabled.cfg.stage7_dual_stage_mtf_ghs_enabled = False
        disabled_candidates, _disabled_adaptation = (
            disabled._stage7_compact_stretch_candidates(
                None,
                {"bg_median": 0.012, "bg_std": 0.0015},
                baseline_stats,
                preview_stats,
                starless_recomposition_planned=True,
                source_profile=self._profile(),
                source_stem="stage6_starless",
                star_separation_state="accepted",
            )
        )

        self.assertEqual(
            [candidate["name"] for candidate in enabled_candidates],
            ["cand_a", "cand_b"],
        )
        self.assertEqual(len(disabled_candidates), 2)
        self.assertEqual(enabled_candidates[0]["method"], "dual_stage_mtf_ghs")
        self.assertEqual(disabled_candidates[0]["method"], "asinh")
        self.assertEqual(
            enabled_candidates[1]["method"],
            disabled_candidates[1]["method"],
        )
        self.assertEqual(
            enabled_candidates[1]["params"],
            disabled_candidates[1]["params"],
        )

    def test_strict_route_failures_keep_legacy_candidate_a(self) -> None:
        baseline_stats = {
            "p01": 0.004,
            "p50": 0.018,
            "p99": 0.120,
            "max": 0.300,
        }
        preview_stats = {"p50": 0.250, "p99": 0.800}
        cases = (
            (
                "high_snr",
                self._profile(faint_signal_snr_proxy=8.0),
                "auto",
                True,
                "signal_not_weak",
            ),
            (
                "untrusted_roi",
                self._profile(trusted_galaxy_roi=False),
                "auto",
                True,
                "trusted_galaxy_roi_unavailable",
            ),
            (
                "infeasible_subject_anchor",
                self._profile(subject_p90=0.070),
                "auto",
                True,
                "calibration_failed",
            ),
            (
                "manual",
                self._profile(),
                "manual",
                True,
                "manual_parameter_mode",
            ),
            (
                "disabled",
                self._profile(),
                "auto",
                False,
                "algorithm_disabled",
            ),
        )
        for name, profile, mode, enabled, reason_code in cases:
            with self.subTest(case=name):
                processor = _StretchProcessor("large_galaxy")
                processor.cfg.stage7_processing_mode = mode
                processor.cfg.stage7_dual_stage_mtf_ghs_enabled = enabled
                candidates, adaptation = (
                    processor._stage7_compact_stretch_candidates(
                        None,
                        {"bg_median": 0.012, "bg_std": 0.0015},
                        baseline_stats,
                        preview_stats,
                        starless_recomposition_planned=True,
                        source_profile=profile,
                        source_stem="stage6_starless",
                        star_separation_state="accepted",
                    )
                )
                self.assertEqual(candidates[0]["method"], "asinh")
                self.assertEqual(
                    adaptation["candidate_a_replacement"]["reason_code"],
                    reason_code,
                )

    def test_explicit_candidate_policies_and_target_switch_keep_legacy_a(self) -> None:
        baseline_stats = {
            "p01": 0.004,
            "p50": 0.018,
            "p99": 0.120,
            "max": 0.300,
        }
        preview_stats = {"p50": 0.250, "p99": 0.800}
        for candidate_policy in ("candidate_a_only", "candidate_b_only"):
            with self.subTest(candidate_policy=candidate_policy):
                processor = _StretchProcessor("large_galaxy")
                processor.cfg.stage7_candidate_policy = candidate_policy
                candidates, adaptation = (
                    processor._stage7_compact_stretch_candidates(
                        None,
                        {"bg_median": 0.012, "bg_std": 0.0015},
                        baseline_stats,
                        preview_stats,
                        starless_recomposition_planned=True,
                        source_profile=self._profile(),
                        source_stem="stage6_starless",
                        star_separation_state="accepted",
                    )
                )
                self.assertEqual(candidates[0]["method"], "asinh")
                self.assertEqual(
                    adaptation["candidate_a_replacement"]["reason_code"],
                    "candidate_policy_preserves_configured_algorithms",
                )

        processor = _StretchProcessor("large_galaxy")
        processor.cfg.stage7_target_aware_stretch_enabled = False
        candidates, adaptation = processor._stage7_compact_stretch_candidates(
            None,
            {"bg_median": 0.012, "bg_std": 0.0015},
            baseline_stats,
            preview_stats,
            starless_recomposition_planned=True,
            source_profile=self._profile(),
            source_stem="stage6_starless",
            star_separation_state="accepted",
        )
        self.assertEqual(candidates[0]["method"], "asinh")
        self.assertEqual(
            adaptation["candidate_a_replacement"]["reason_code"],
            "target_aware_stretch_disabled",
        )

    def test_quantile_fallback_keeps_original_preview_targets(self) -> None:
        source = np.linspace(
            0.001,
            0.35,
            3 * 16 * 16,
            dtype=np.float32,
        ).reshape(3, 16, 16)
        adaptation = {
            "preview_calibration": {
                "candidate_a": {
                    "calibration_method": "dual_stage_mtf_ghs",
                    "target_p50": 0.117,
                    "target_p99": 0.910,
                    "auto_asinh_target_p50": 0.080,
                    "auto_asinh_target_p99": 0.700,
                }
            }
        }

        calibration = stretch_metrics.calibrate_adaptive_quantile_stretch(
            source,
            adaptation,
            PipelineConfig(),
        )

        self.assertEqual(calibration["status"], "ok", calibration)
        self.assertEqual(calibration["target_p50"], 0.080)
        self.assertEqual(calibration["target_p99"], 0.700)

    def test_executor_rebuilds_linked_lut_without_external_runtime(self) -> None:
        calibration = stretch_metrics.calibrate_iterative_masked_mtf(
            self._profile(),
            target_background=0.085,
            iterations=16,
            max_derivative=5000.0,
            source_p50=0.018,
            source_p99=0.120,
        )
        processor = _StretchProcessor("globular_cluster")
        source = np.linspace(0.001, 0.40, 3 * 16 * 16, dtype=np.float32).reshape(
            3,
            16,
            16,
        )

        class _Siril:
            def get_image_pixeldata(self, preview=False):
                _ = preview
                return source.copy()

        processor.siril = _Siril()
        processor._stage8_restore_rgb_like = lambda _source, rgb: rgb
        captured = {}
        processor._set_current_image_pixeldata = (
            lambda image, **_kwargs: captured.setdefault("image", image)
        )
        ok, used = processor._execute_stage7_stretch_candidate(
            {
                "method": "iterative_masked_mtf",
                "params": {"calibration": calibration},
            }
        )

        self.assertTrue(ok, used)
        self.assertEqual(used, "iterative_masked_mtf")
        self.assertEqual(captured["image"].shape, source.shape)
        self.assertTrue(np.all(np.isfinite(captured["image"])))
        self.assertLessEqual(float(np.max(captured["image"])), 0.995)

        mismatch_ok, mismatch_reason = (
            processor._execute_stage7_stretch_candidate(
                {
                    "method": "dual_stage_mtf_ghs",
                    "params": {"calibration": calibration},
                }
            )
        )
        self.assertFalse(mismatch_ok)
        self.assertIn("mismatch", mismatch_reason)


if __name__ == "__main__":
    unittest.main()
