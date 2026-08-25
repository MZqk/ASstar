import unittest
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
    def _assess(self, before, after, *, target_type=None, target_profile=None):
        return bright_core_color.assess_spcc_bright_core_color(
            before,
            after,
            target_type=(
                target_type or "bright_emission_reflection_nebula"
            ),
            target_profile=(
                STRICT_PROFILE if target_profile is None else target_profile
            ),
        )

    def test_connected_channel_flip_is_flagged_without_pixel_repair(self):
        before = _strict_core_image()
        mask = np.zeros(before.shape[1:], dtype=bool)
        mask[126:129, 126:129] = True
        after = _flip_to_blue(before, mask)

        report, _context = self._assess(before, after)

        self.assertEqual(report["status"], "high_risk")
        self.assertFalse(report["accepted"])
        self.assertEqual(
            report["final_action"],
            "flag_compact_core_chroma_anomaly",
        )
        self.assertNotIn("repair", report)
        self.assertFalse(
            hasattr(
                bright_core_color,
                "evaluate_and_repair_spcc_bright_core",
            )
        )

    def test_uint16_candidate_is_assessed_in_normalized_domain(self):
        before_float = np.clip(_strict_core_image(), 0.0, 1.0)
        mask = np.zeros(before_float.shape[1:], dtype=bool)
        mask[126:129, 126:129] = True
        after_float = np.clip(_flip_to_blue(before_float, mask), 0.0, 1.0)
        before = np.rint(before_float * 65535.0).astype(np.uint16)
        after = np.rint(after_float * 65535.0).astype(np.uint16)

        report, context = self._assess(before, after)

        self.assertEqual(
            report["numeric_domain"]["before"]["domain"],
            "uint16_full_scale",
        )
        self.assertLessEqual(report["roi"]["threshold"], 1.0)
        self.assertEqual(context["after_native"].dtype, np.float64)

    def test_normal_m8_route_does_not_apply_strict_guard(self):
        before = _strict_core_image(128)
        after = before * np.asarray((1.05, 0.98, 1.02))[:, None, None]

        report, context = self._assess(
            before,
            after,
            target_profile={
                "target_name_guess": "Lagoon Nebula",
                "secondary_labels": ["bright_core", "large_nebulosity"],
                "features": {"bright_core": True},
                "risks": {"core_blowout": "medium"},
            },
        )

        self.assertFalse(report["applicable"])
        self.assertEqual(report["status"], "not_applicable")
        self.assertEqual(context, {})

    def test_compact_colored_star_does_not_trigger_high_risk(self):
        before = _strict_core_image()
        mask = np.zeros(before.shape[1:], dtype=bool)
        mask[128, 128] = True
        after = _flip_to_blue(before, mask)

        report, _context = self._assess(before, after)

        self.assertTrue(report["accepted"])
        self.assertIn(report["status"], {"ok", "advisory"})
        self.assertNotIn("repair", report["fixed_limits"])

    def test_insufficient_roi_is_flagged_without_routing_action(self):
        before = _strict_core_image(32)

        report, _context = self._assess(before, before)

        self.assertEqual(report["status"], "hard_failed")
        self.assertEqual(
            report["final_action"],
            "flag_insufficient_roi_support",
        )
        self.assertIn(
            "bright_core_roi_support_insufficient",
            report["trigger_reasons"],
        )

    def test_broad_core_chroma_platform_is_advisory_evidence_only(self):
        before = _strict_core_image()
        base_report, context = self._assess(before, before)
        self.assertGreaterEqual(base_report["roi"]["support_pixels"], 64)
        after = _flip_to_blue(before, context["roi"])

        report, _context = self._assess(before, after)

        self.assertEqual(report["status"], "hard_failed")
        self.assertEqual(
            report["final_action"],
            "flag_broad_core_chroma_platform",
        )
        self.assertIn("broad_core_chroma_platform", report["trigger_reasons"])
        self.assertEqual(
            report["fixed_limits"]["broad_core_chroma_platform"]["action"],
            "flag_advisory_only",
        )


if __name__ == "__main__":
    unittest.main()
