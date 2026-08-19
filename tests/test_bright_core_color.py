import unittest
from unittest import mock
import sys
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import bright_core_color


STRICT_PROFILE = {
    "secondary_labels": ["bright_core"],
    "features": {"bright_core": True},
    "risks": {"core_blowout": "high"},
}
REC709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


def _strict_core_image(size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    luminance = (
        0.01
        + 0.70
        * np.exp(
            -(
                (yy - size / 2.0) ** 2
                + (xx - size / 2.0) ** 2
            )
            / (2.0 * (size * 0.09) ** 2)
        )
        + 0.000001 * (yy * size + xx)
    )
    direction = np.asarray((0.75, 0.18, 0.07), dtype=np.float64)
    scale = luminance / float(direction @ REC709)
    return direction[:, None, None] * scale[None, :, :]


def _flip_to_blue(before: np.ndarray, mask: np.ndarray) -> np.ndarray:
    after = before.copy()
    luminance = np.tensordot(REC709, before, axes=(0, 0))
    direction = np.asarray((0.05, 0.15, 0.80), dtype=np.float64)
    replacement = direction[:, None, None] * (
        luminance / float(direction @ REC709)
    )[None, :, :]
    after[:, mask] = replacement[:, mask]
    return after


class BrightCoreColorIntegrityTests(unittest.TestCase):
    def _assess(self, before, after, *, target_type=None):
        return bright_core_color.assess_spcc_bright_core_color(
            before,
            after,
            target_type=(
                target_type or "bright_emission_reflection_nebula"
            ),
            target_profile=STRICT_PROFILE,
        )

    def test_m42_type_connected_channel_flip_is_locally_repaired(self):
        before = _strict_core_image()
        mask = np.zeros(before.shape[1:], dtype=bool)
        mask[126:129, 126:129] = True
        after = _flip_to_blue(before, mask)

        candidate, report = (
            bright_core_color.evaluate_and_repair_spcc_bright_core(
                before,
                after,
                target_type="bright_emission_reflection_nebula",
                target_profile=STRICT_PROFILE,
            )
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(report["status"], "repaired")
        self.assertTrue(report["accepted"])
        self.assertTrue(report["repair"]["passed"])
        self.assertLessEqual(
            report["repair"]["checks"]["largest_component_ratio"]["value"],
            0.005,
        )
        self.assertLessEqual(
            report["repair"]["support_ratio_of_image"],
            0.015,
        )

    def test_uint16_spcc_candidate_is_assessed_normalized_and_restored(self):
        before_float = np.clip(_strict_core_image(), 0.0, 1.0)
        mask = np.zeros(before_float.shape[1:], dtype=bool)
        mask[126:129, 126:129] = True
        after_float = np.clip(_flip_to_blue(before_float, mask), 0.0, 1.0)
        before = np.rint(before_float * 65535.0).astype(np.uint16)
        after = np.rint(after_float * 65535.0).astype(np.uint16)

        candidate, report = (
            bright_core_color.evaluate_and_repair_spcc_bright_core(
                before,
                after,
                target_type="bright_emission_reflection_nebula",
                target_profile=STRICT_PROFILE,
            )
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(report["status"], "repaired")
        self.assertEqual(
            report["numeric_domain"]["before"]["domain"],
            "uint16_full_scale",
        )
        self.assertLessEqual(report["roi"]["threshold"], 1.0)
        self.assertGreater(float(np.max(candidate)), 1.0)
        support_ratio = report["repair"]["support_ratio_of_image"]
        self.assertGreater(support_ratio, 0.0)
        support_pixels = report["repair"]["support_pixels"]
        self.assertLess(support_pixels, candidate.shape[1] * candidate.shape[2])

    def test_normal_m8_route_does_not_apply_strict_guard(self):
        before = _strict_core_image(128)
        after = before * np.asarray((1.05, 0.98, 1.02))[:, None, None]

        candidate, report = (
            bright_core_color.evaluate_and_repair_spcc_bright_core(
                before,
                after,
                target_type="bright_emission_reflection_nebula",
                target_profile={
                    "target_name_guess": "Lagoon Nebula",
                    "secondary_labels": ["bright_core", "large_nebulosity"],
                    "features": {"bright_core": True},
                    "risks": {"core_blowout": "medium"},
                },
            )
        )

        self.assertIsNotNone(candidate)
        self.assertFalse(report["applicable"])
        self.assertEqual(report["status"], "not_applicable")

    def test_compact_colored_star_does_not_trigger_repair(self):
        before = _strict_core_image()
        mask = np.zeros(before.shape[1:], dtype=bool)
        mask[128, 128] = True
        after = _flip_to_blue(before, mask)

        report, _context = self._assess(before, after)

        self.assertTrue(report["accepted"])
        self.assertIn(report["status"], {"ok", "advisory"})
        self.assertNotEqual(
            report["final_action"],
            "attempt_local_core_chroma_rollback",
        )

    def test_insufficient_roi_rejects_strict_spcc(self):
        before = _strict_core_image(32)
        candidate, report = (
            bright_core_color.evaluate_and_repair_spcc_bright_core(
                before,
                before,
                target_type="bright_emission_reflection_nebula",
                target_profile=STRICT_PROFILE,
            )
        )

        self.assertIsNone(candidate)
        self.assertEqual(report["status"], "hard_failed")
        self.assertIn(
            "bright_core_roi_support_insufficient",
            report["trigger_reasons"],
        )

    def test_oversized_repair_support_and_new_clip_fail_closed(self):
        before = _strict_core_image(80)
        base_report, context = self._assess(before, before)
        self.assertEqual(base_report["roi"]["support_pixels"], 64)
        after = _flip_to_blue(before, context["roi"])

        with mock.patch.object(
            bright_core_color,
            "BROAD_PLATFORM_STRONG_ANOMALY_RATIO_MIN",
            1.0,
        ):
            candidate, report = (
                bright_core_color.evaluate_and_repair_spcc_bright_core(
                    before,
                    after,
                    target_type="bright_emission_reflection_nebula",
                    target_profile=STRICT_PROFILE,
                )
            )

        self.assertIsNone(candidate)
        self.assertEqual(report["status"], "hard_failed")
        checks = report["repair"]["checks"]
        self.assertFalse(checks["modified_support_ratio"]["passed"])
        self.assertFalse(checks["new_clip_ratio"]["passed"])

    def test_broad_core_chroma_platform_attempts_once_then_fails_closed(self):
        before = _strict_core_image()
        base_report, context = self._assess(before, before)
        self.assertGreaterEqual(base_report["roi"]["support_pixels"], 64)
        after = _flip_to_blue(before, context["roi"])

        candidate, report = (
            bright_core_color.evaluate_and_repair_spcc_bright_core(
                before,
                after,
                target_type="bright_emission_reflection_nebula",
                target_profile=STRICT_PROFILE,
            )
        )

        self.assertIsNone(candidate)
        self.assertEqual(report["status"], "hard_failed")
        self.assertEqual(report["final_action"], "reject_spcc_to_pcc")
        self.assertIn("broad_core_chroma_platform", report["trigger_reasons"])
        measurements = report["measurements"]
        self.assertGreater(
            measurements["anomaly_ratio_of_roi"],
            0.02,
        )
        self.assertGreater(
            measurements["broad_platform_largest_component_ratio_of_roi"],
            0.05,
        )
        self.assertEqual(
            report["repair"]["method"],
            "SPCC_BROAD_CORE_CHROMA_ROLLBACK",
        )
        self.assertFalse(report["repair"]["passed"])
        self.assertFalse(
            report["repair"]["checks"]["modified_support_ratio"]["passed"]
        )

    def test_bounded_broad_core_chroma_platform_is_repaired_for_review(self):
        before = np.clip(_strict_core_image(), 0.0, 0.90)
        base_report, context = self._assess(before, before)
        self.assertGreaterEqual(base_report["roi"]["support_pixels"], 64)
        platform = np.zeros(before.shape[1:], dtype=bool)
        platform[124:133, 124:133] = True
        platform &= context["roi"]
        after = _flip_to_blue(before, platform)

        candidate, report = (
            bright_core_color.evaluate_and_repair_spcc_bright_core(
                before,
                after,
                target_type="bright_emission_reflection_nebula",
                target_profile=STRICT_PROFILE,
            )
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(report["status"], "repaired")
        self.assertTrue(report["accepted"])
        self.assertTrue(report["repaired"])
        self.assertTrue(report["requires_review"])
        self.assertEqual(
            report["final_action"],
            "accept_bounded_broad_core_chroma_rollback_for_review",
        )
        repair = report["repair"]
        self.assertEqual(
            repair["method"],
            "SPCC_BROAD_CORE_CHROMA_ROLLBACK",
        )
        self.assertLessEqual(repair["support_ratio_of_image"], 0.015)
        self.assertLessEqual(
            repair["checks"]["luma_abs_error_p99"]["value"],
            0.002,
        )
        self.assertLessEqual(
            repair["checks"]["new_clip_ratio"]["value"],
            0.001,
        )
        self.assertTrue(
            repair["checks"]["post_repair_full_assessment"]["passed"]
        )

    def test_luminance_revalidation_failure_rejects_repair(self):
        before = _strict_core_image()
        mask = np.zeros(before.shape[1:], dtype=bool)
        mask[126:129, 126:129] = True
        after = _flip_to_blue(before, mask)

        with mock.patch.object(
            bright_core_color,
            "REPAIR_LUMA_ERROR_P99_MAX",
            -1.0,
        ):
            candidate, report = (
                bright_core_color.evaluate_and_repair_spcc_bright_core(
                    before,
                    after,
                    target_type="bright_emission_reflection_nebula",
                    target_profile=STRICT_PROFILE,
                )
            )

        self.assertIsNone(candidate)
        self.assertFalse(
            report["repair"]["checks"]["luma_abs_error_p99"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
