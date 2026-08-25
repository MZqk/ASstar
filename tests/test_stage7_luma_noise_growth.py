from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


if "sirilpy.exceptions" not in sys.modules:
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

    class CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    class SirilInterface:
        def cmd(self, *_args, **_kwargs):
            return None

    exceptions.SirilError = SirilError
    exceptions.SirilConnectionError = SirilConnectionError
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    enums.CommandStatus = CommandStatus
    package.exceptions = exceptions
    package.SirilInterface = SirilInterface
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums


from models import PipelineConfig  # noqa: E402
from processing_parameters import PROCESSING_GATE_PARAMETER_SPECS  # noqa: E402
from stage6_services import Stage6ServiceMixin  # noqa: E402
from stage7_stretch_metrics import (  # noqa: E402
    LUMA_NOISE_GROWTH_SCHEMA,
    LUMA_VISIBLE_NOISE_REFERENCE_FLOOR,
    assess_frozen_background_luma_noise_growth,
)


class Stage7LumaNoiseGrowthTests(unittest.TestCase):
    @staticmethod
    def _checkerboard(size: int = 64) -> np.ndarray:
        yy, xx = np.indices((size, size))
        return np.where((xx + yy) % 2 == 0, -1.0, 1.0).astype(np.float32)

    @classmethod
    def _image(cls, amplitude: float, *, base: float = 0.10) -> np.ndarray:
        gray = base + amplitude * cls._checkerboard()
        return np.repeat(gray[None, ...], 3, axis=0).astype(np.float32)

    @staticmethod
    def _all_background(size: int = 64) -> dict[str, np.ndarray]:
        return {"background_mask": np.ones((size, size), dtype=np.float32)}

    def _assess(
        self,
        source: np.ndarray,
        candidate: np.ndarray,
        masks: dict[str, np.ndarray] | None = None,
        *,
        limit: float = 1.25,
    ) -> dict:
        cfg = PipelineConfig(stage7_stretch_luma_noise_growth_max=limit)
        return assess_frozen_background_luma_noise_growth(
            source,
            candidate,
            masks if masks is not None else self._all_background(),
            cfg,
        )

    def test_default_and_expert_parameter_contract(self):
        self.assertEqual(
            PipelineConfig().stage7_stretch_luma_noise_growth_max,
            1.25,
        )
        spec = next(
            item
            for item in PROCESSING_GATE_PARAMETER_SPECS
            if item.field == "stage7_stretch_luma_noise_growth_max"
        )
        self.assertEqual(spec.stage, 7)
        self.assertEqual((spec.minimum, spec.maximum, spec.step), (1.0, 3.0, 0.05))
        self.assertEqual(spec.suffix, "×")

    def test_schema_and_uniform_scaling_keep_visible_noise_stable(self):
        source = self._image(0.002)
        report = self._assess(source, source * 2.0)

        self.assertEqual(report["schema"], LUMA_NOISE_GROWTH_SCHEMA)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(
            report["metrics"]["visible_noise_growth"],
            1.0,
            places=5,
        )

    def test_nominal_advisory_and_hard_boundaries(self):
        source = self._image(0.002)
        for factor, expected in (
            (1.25, "ok"),
            (1.50, "advisory"),
            (1.875, "advisory"),
            (1.90, "poor"),
        ):
            with self.subTest(factor=factor):
                candidate = self._image(0.002 * factor)
                report = self._assess(source, candidate)
                self.assertEqual(report["status"], expected)
                self.assertEqual(report["accepted"], expected != "poor")

    def test_low_absolute_noise_exempts_unstable_ratio(self):
        source = self._image(0.00001)
        candidate = self._image(0.00010)

        report = self._assess(source, candidate)

        self.assertTrue(report["accepted"])
        self.assertTrue(report["low_absolute_exempted"])
        self.assertEqual(report["quality_gate"]["status"], "ok")
        self.assertTrue(
            report["metrics"]["visible_noise_reference_floor_applied"]
        )

    def test_dark_linear_reference_uses_visible_noise_floor(self):
        source = self._image(0.00001, base=0.015)
        acceptable = self._image(0.0012, base=0.10)
        excessive = self._image(0.0060, base=0.10)

        accepted_report = self._assess(source, acceptable)
        rejected_report = self._assess(source, excessive)

        self.assertTrue(
            accepted_report["metrics"][
                "visible_noise_reference_floor_applied"
            ]
        )
        self.assertEqual(
            accepted_report["metrics"]["visible_noise_reference_score"],
            LUMA_VISIBLE_NOISE_REFERENCE_FLOOR,
        )
        self.assertTrue(accepted_report["accepted"])
        self.assertFalse(rejected_report["accepted"])

    def test_low_frequency_gradient_is_not_high_frequency_noise(self):
        source = self._image(0.001)
        gradient = np.linspace(0.0, 0.03, 64, dtype=np.float32)[None, :]
        candidate = source + gradient[None, ...]

        report = self._assess(source, candidate)

        self.assertTrue(report["accepted"])
        self.assertFalse(report["quality_gate"]["hard_failed"])

    def test_opponent_chroma_noise_does_not_create_rec709_luma_noise(self):
        source = np.full((3, 64, 64), 0.10, dtype=np.float32)
        pattern = self._checkerboard()
        candidate = source.copy()
        candidate[0] += pattern * 0.01
        candidate[1] -= pattern * 0.01 * (0.2126 / 0.7152)

        report = self._assess(source, candidate)

        self.assertTrue(report["accepted"])
        self.assertLess(
            report["metrics"]["candidate"]["sigma"],
            1e-6,
        )

    def test_frozen_signal_exclusion_ignores_subject_noise(self):
        source = self._image(0.001)
        subject = np.zeros((64, 64), dtype=np.float32)
        subject[16:48, 16:48] = 1.0
        masks = {
            "background_mask": np.ones((64, 64), dtype=np.float32),
            "subject_mask": subject,
        }
        subject_candidate = source.copy()
        subject_candidate[:, 16:48, 16:48] += (
            0.02 * self._checkerboard(32)[None, ...]
        )
        background_candidate = self._image(0.006)

        subject_report = self._assess(source, subject_candidate, masks)
        background_report = self._assess(source, background_candidate, masks)

        self.assertTrue(subject_report["accepted"])
        self.assertEqual(
            subject_report["mask_source"],
            "stage6_frozen_background_signal_excluded",
        )
        self.assertFalse(background_report["accepted"])

    def test_source_p35_fallback_is_frozen_for_both_images(self):
        source_gray = np.linspace(0.02, 0.20, 4096, dtype=np.float32).reshape(
            64, 64
        )
        source = np.repeat(source_gray[None, ...], 3, axis=0)
        candidate = source.copy()
        threshold = np.quantile(source_gray, 0.35)
        candidate[:, source_gray > threshold] = 0.90
        frozen_masks = {
            "luma_noise_background_mask": (source_gray <= threshold).astype(
                np.float32
            )
        }

        report = self._assess(source, candidate, frozen_masks)

        self.assertEqual(report["mask_source"], "stage6_source_luma_p35")
        self.assertGreaterEqual(report["support_count"], 64)
        self.assertTrue(report["accepted"])

    def test_invalid_measurements_fail_closed(self):
        source = self._image(0.001)
        non_finite = source.copy()
        non_finite[0, 0, 0] = np.nan
        invalid_cases = (
            (source, source[:, :-1, :], self._all_background()),
            (source, non_finite, self._all_background()),
            (
                source,
                source,
                {"background_mask": np.zeros((64, 64), dtype=np.float32)},
            ),
        )
        for baseline, candidate, masks in invalid_cases:
            with self.subTest(candidate_shape=candidate.shape):
                report = self._assess(baseline, candidate, masks)
                self.assertFalse(report["accepted"])
                self.assertEqual(report["status"], "unavailable")
                self.assertTrue(report["quality_gate"]["hard_failed"])

    def test_luma_failure_is_reviewable_but_does_not_trigger_chroma_rescue(self):
        processor = object.__new__(Stage6ServiceMixin)
        processor.cfg = PipelineConfig()
        attempt = {
            "status": "ok",
            "stem": "stage7_cand_a",
            "diagnostics": [
                "background_luma_noise_growth 1.900>1.250"
            ],
            "pixel_stats": {
                "p50": 0.10,
                "p99": 0.80,
                "dynamic_range": 0.79,
            },
            "preview_target_attainment": {"attainment_ratio": 1.0},
            "target_local_quality": {"accepted": True},
        }

        self.assertTrue(processor._stage7_review_candidate_is_safe(attempt))
        self.assertFalse(processor._stage7_attempt_allows_chroma_rescue(attempt))

    @staticmethod
    def _ranking_attempt(name: str) -> dict:
        return {
            "name": name,
            "status": "ok",
            "stem": f"stage7_{name}",
            "allowed_as_final": True,
            "technical_safe": True,
            "presentation_score": {
                "policy": "hard_gate_continuous_quality_v7",
                "score": 0.60,
            },
            "presentation_score_v6": {"score": 0.60},
            "advisories": [],
            "risk_score": 0.10,
        }

    def test_v7_sort_is_invariant_to_advisory_and_legacy_risk_state(self):
        baseline = self._ranking_attempt("same")
        baseline["presentation_score"] = (
            Stage6ServiceMixin._stage7_presentation_score(baseline)
        )
        changed = {
            **baseline,
            "advisories": ["two", "one", "one"],
            "risk_score": 999.0,
        }
        changed["presentation_score"] = (
            Stage6ServiceMixin._stage7_presentation_score(changed)
        )

        self.assertEqual(
            baseline["presentation_score"]["score"],
            changed["presentation_score"]["score"],
        )
        self.assertNotEqual(
            baseline["presentation_score"]["advisory_count"],
            changed["presentation_score"]["advisory_count"],
        )
        self.assertEqual(
            Stage6ServiceMixin._stage7_candidate_selection_key(baseline),
            Stage6ServiceMixin._stage7_candidate_selection_key(changed),
        )
        self.assertNotEqual(
            Stage6ServiceMixin._stage7_candidate_selection_key_v6(baseline),
            Stage6ServiceMixin._stage7_candidate_selection_key_v6(changed),
        )

    def test_v7_quality_score_precedes_brightness_goal_tiebreakers(self):
        lower_quality_brighter = self._ranking_attempt("brighter")
        lower_quality_brighter["presentation_score"]["score"] = 0.60
        lower_quality_brighter["subject_brightness_selection"] = {
            "formal_floor_passed": True,
            "ranking": {"goal_count": 2, "utility": 1.0},
        }
        higher_quality = self._ranking_attempt("quality")
        higher_quality["presentation_score"]["score"] = 0.80
        higher_quality["subject_brightness_selection"] = {
            "formal_floor_passed": True,
            "ranking": {"goal_count": 0, "utility": 0.60},
        }

        selected = min(
            (lower_quality_brighter, higher_quality),
            key=Stage6ServiceMixin._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "quality")

    def test_continuous_quality_formula_contracts(self):
        upper = Stage6ServiceMixin._stage7_continuous_upper_utility
        lower = Stage6ServiceMixin._stage7_continuous_lower_utility

        self.assertEqual(upper(0.0, 0.30), 1.0)
        self.assertEqual(upper(0.30, 0.30), 0.0)
        self.assertEqual(upper(1.0, 1.875, ideal_value=1.0), 1.0)
        self.assertEqual(upper(1.875, 1.875, ideal_value=1.0), 0.0)
        self.assertAlmostEqual(lower(0.40, 0.60, 0.40), 0.0)
        self.assertAlmostEqual(lower(0.60, 0.60, 0.40), 1.0)
        self.assertEqual(upper(None, 1.0), 0.5)
        self.assertEqual(lower(None, 0.6, 0.4), 0.5)

    def test_v7_safety_weights_and_missing_optional_limit_are_stable(self):
        generic = self._ranking_attempt("generic")
        generic["background_quality_gate"] = {
            "metrics": {"chroma_load_growth": 1.2},
            "limits": {},
        }
        generic_report = Stage6ServiceMixin._stage7_presentation_score(generic)
        self.assertEqual(
            generic_report["continuous_safety_utilities"][
                "background_chroma_load"
            ],
            0.5,
        )
        self.assertEqual(
            generic_report["continuous_safety_weights"],
            {
                "background_chroma_load": 0.25,
                "chroma_noise": 0.15,
                "mottling": 0.10,
                "luma_noise_growth": 0.15,
                "color_vector": 0.20,
                "new_hard_clip": 0.15,
            },
        )

        nebula = self._ranking_attempt("nebula")
        nebula["adaptation"] = {
            "target_aware": {"name": "widefield_nebulosity"}
        }
        nebula_report = Stage6ServiceMixin._stage7_presentation_score(nebula)
        self.assertEqual(
            nebula_report["continuous_safety_weights"],
            {
                "background_chroma_load": 0.20,
                "chroma_noise": 0.10,
                "mottling": 0.10,
                "luma_noise_growth": 0.10,
                "color_vector": 0.15,
                "new_hard_clip": 0.15,
                "target_core_safety": 0.20,
            },
        )

    def test_luma_quality_is_continuous_across_advisory_band(self):
        def attempt(name: str, growth: float, advisory: bool) -> dict:
            item = self._ranking_attempt(name)
            item["advisories"] = (
                ["background_luma_noise_growth advisory"]
                if advisory
                else []
            )
            item["luma_noise_quality"] = {
                "status": "advisory" if advisory else "ok",
                "low_absolute_exempted": False,
                "metrics": {"visible_noise_growth": growth},
                "quality_gate": {"hard_limit": 1.875},
            }
            item["presentation_score"] = (
                Stage6ServiceMixin._stage7_presentation_score(item)
            )
            return item

        just_over = attempt("just_over", 1.251, True)
        near_hard = attempt("near_hard", 1.874, True)

        self.assertGreater(
            just_over["presentation_score"]["continuous_safety_utilities"][
                "luma_noise_growth"
            ],
            near_hard["presentation_score"]["continuous_safety_utilities"][
                "luma_noise_growth"
            ],
        )
        self.assertLess(
            Stage6ServiceMixin._stage7_candidate_selection_key(just_over),
            Stage6ServiceMixin._stage7_candidate_selection_key(near_hard),
        )

    def test_low_absolute_luma_uses_sigma_instead_of_unstable_growth(self):
        def score(sigma: float, growth: float) -> float:
            item = self._ranking_attempt(f"sigma_{sigma}")
            item["luma_noise_quality"] = {
                "status": "ok",
                "low_absolute_exempted": True,
                "metrics": {
                    "candidate": {"sigma": sigma},
                    "visible_noise_growth": growth,
                    "low_absolute_sigma_max": 0.00075,
                },
                "quality_gate": {"hard_limit": 1.875},
            }
            return Stage6ServiceMixin._stage7_presentation_score(item)[
                "continuous_safety_utilities"
            ]["luma_noise_growth"]

        self.assertGreater(score(0.00010, 20.0), score(0.00070, 2.0))

    def test_review_ranking_uses_luma_severity_and_puts_unavailable_last(self):
        def attempt(name: str, value, status: str = "hard_failed") -> dict:
            item = self._ranking_attempt(name)
            item["allowed_as_final"] = False
            item["diagnostics"] = [
                "background_luma_noise_growth_measurement_unavailable"
                if status == "unavailable"
                else f"background_luma_noise_growth {value:.3f}>1.250"
            ]
            item["pixel_stats"] = {"p50": 0.1, "dynamic_range": 0.8}
            item["preview_target_attainment"] = {"attainment_ratio": 1.0}
            item["target_local_quality"] = {"accepted": True}
            item["luma_noise_quality"] = {
                "status": status,
                "quality_gate": {
                    "status": status,
                    "hard_failed": True,
                    "value": value,
                    "hard_limit": 1.875,
                },
            }
            return item

        lower = attempt("lower", 1.90)
        higher = attempt("higher", 2.50)
        unavailable = attempt("unavailable", None, "unavailable")
        ordered = sorted(
            [unavailable, higher, lower],
            key=Stage6ServiceMixin._stage7_review_candidate_selection_key,
        )

        self.assertEqual(
            [item["name"] for item in ordered],
            ["lower", "higher", "unavailable"],
        )

    def test_review_ranking_treats_lower_bound_failures_as_measurable(self):
        def subject_failure(name: str, retention: float) -> dict:
            item = self._ranking_attempt(name)
            item["allowed_as_final"] = False
            item["diagnostics"] = ["stage7_subject_brightness_floor_unmet"]
            item["subject_brightness_selection"] = {
                "formal_floor_passed": False,
                "retention": {
                    "subject_p50": retention,
                    "subject_lift": None,
                },
                "floors": {
                    "subject_p50_retention": 0.60,
                    "subject_lift_retention": None,
                },
            }
            return item

        near_floor = subject_failure("near_floor", 0.59)
        far_below = subject_failure("far_below", 0.20)
        unavailable = self._ranking_attempt("unavailable")
        unavailable["allowed_as_final"] = False
        unavailable["diagnostics"] = [
            "background_luma_noise_growth_measurement_unavailable"
        ]
        unavailable["luma_noise_quality"] = {
            "status": "unavailable",
            "quality_gate": {
                "status": "unavailable",
                "hard_failed": True,
                "value": None,
                "hard_limit": 1.875,
            },
        }

        ordered = sorted(
            [unavailable, far_below, near_floor],
            key=Stage6ServiceMixin._stage7_review_candidate_selection_key,
        )
        self.assertEqual(
            [item["name"] for item in ordered],
            ["near_floor", "far_below", "unavailable"],
        )

    def test_hard_gate_eligibility_still_precedes_continuous_score(self):
        eligible = self._ranking_attempt("eligible")
        eligible["presentation_score"]["score"] = 0.10
        rejected = self._ranking_attempt("rejected")
        rejected["allowed_as_final"] = False
        rejected["presentation_score"]["score"] = 1.0

        selected = min(
            [rejected, eligible],
            key=Stage6ServiceMixin._stage7_candidate_selection_key,
        )
        self.assertEqual(selected["name"], "eligible")


if __name__ == "__main__":
    unittest.main()
