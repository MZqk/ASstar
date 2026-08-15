#!/usr/bin/env python3
"""Read-only Stage 9 PSF/scale diagnostics must never synthesize zero support."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import stage9_quality  # noqa: E402


class Stage9PsfShadowTests(unittest.TestCase):
    def test_insufficient_psf_support_is_unavailable_not_zero(self) -> None:
        result = stage9_quality._stage9_psf_scale_shadow(
            np.zeros((3, 16, 16), dtype=np.float32),
            {
                "status": "ok",
                "_source_fwhm_px": np.asarray([2.0, 2.2, 2.1]),
                "_peak_y": np.asarray([3, 7, 11]),
                "_peak_x": np.asarray([3, 7, 11]),
                "_source_chroma": np.full((3, 3), 1.0 / 3.0),
            },
            star_overlay_mask=None,
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["support_count"], 3)
        self.assertNotIn("source_fwhm_median_px", result)

    def test_source_confirmed_psf_shadow_reports_normalized_morphology(self) -> None:
        positive = np.zeros((3, 24, 24), dtype=np.float32)
        peak_y = np.asarray([4, 8, 12, 18])
        peak_x = np.asarray([5, 16, 11, 19])
        for y, x in zip(peak_y, peak_x):
            positive[:, y, x] = (0.04, 0.03, 0.02)
        support = np.zeros((24, 24), dtype=bool)
        support[peak_y, peak_x] = True
        result = stage9_quality._stage9_psf_scale_shadow(
            positive,
            {
                "status": "ok",
                "_source_fwhm_px": np.asarray([2.0, 2.2, 2.4, 2.6]),
                "_peak_y": peak_y,
                "_peak_x": peak_x,
                "_source_chroma": np.tile(
                    np.asarray([[4.0 / 9.0, 3.0 / 9.0, 2.0 / 9.0]]),
                    (4, 1),
                ),
            },
            star_overlay_mask=support,
        )

        self.assertEqual(result["status"], "shadow")
        self.assertEqual(result["support_count"], 4)
        self.assertGreater(result["source_fwhm_median_px"], 0.0)
        self.assertIsNotNone(result["positive_component_area_over_fwhm2_median"])
        self.assertEqual(
            result["positive_component_area_over_fwhm2_status"],
            "shadow",
        )
        self.assertEqual(
            result["outside_confirmed_star_change_status"],
            "shadow",
        )
        self.assertAlmostEqual(result["source_chroma_error_median"], 0.0, places=6)

    def test_missing_overlay_support_is_explicitly_unavailable(self) -> None:
        result = stage9_quality._stage9_psf_scale_shadow(
            np.zeros((3, 24, 24), dtype=np.float32),
            {
                "status": "ok",
                "_source_fwhm_px": np.asarray([2.0, 2.2, 2.4, 2.6]),
                "_peak_y": np.asarray([4, 8, 12, 18]),
                "_peak_x": np.asarray([5, 16, 11, 19]),
                "_source_chroma": np.full((4, 3), 1.0 / 3.0),
            },
            star_overlay_mask=None,
        )

        self.assertEqual(result["status"], "shadow")
        self.assertIsNone(result["outside_confirmed_star_change_ratio"])
        self.assertEqual(
            result["outside_confirmed_star_change_status"],
            "unavailable",
        )
        self.assertEqual(
            result["positive_component_area_over_fwhm2_status"],
            "unavailable",
        )


if __name__ == "__main__":
    unittest.main()
