from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from image_metrics import measure_image_features  # noqa: E402
from models import PipelineConfig  # noqa: E402
from stage7_quality import (  # noqa: E402
    stage7_calibrate_starless_dynamic_range,
    stage7_dynamic_range_assessment,
    stage7_starless_artifact_scores,
)


class SyqonDynamicRangeCalibrationTests(unittest.TestCase):
    @staticmethod
    def _dense_star_nebula() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height = width = 256
        yy, xx = np.mgrid[:height, :width]
        background = 0.00135 + 0.00008 * xx.astype(np.float32) / width
        nebula = (
            0.00110
            * np.exp(
                -(
                    ((xx - 132.0) / 58.0) ** 2
                    + ((yy - 124.0) / 44.0) ** 2
                )
            )
        ).astype(np.float32)
        starless_gray = background + nebula
        starmask_gray = np.zeros_like(starless_gray)
        for cy in range(10, height, 20):
            for cx in range(10, width, 20):
                radius2 = (yy - cy) ** 2 + (xx - cx) ** 2
                starmask_gray += 0.006 * np.exp(-radius2 / 4.0)
        source_gray = starless_gray + starmask_gray
        return tuple(
            np.repeat(layer[None, :, :], 3, axis=0).astype(np.float32)
            for layer in (source_gray, starless_gray, starmask_gray)
        )

    @staticmethod
    def _pipeline() -> SimpleNamespace:
        return SimpleNamespace(
            cfg=PipelineConfig(),
            _active_target_type=lambda: "emission_nebula_widefield",
            target_profile={
                "secondary_labels": [
                    "large_nebulosity",
                    "faint_outer_cloud",
                    "emission_red",
                ],
                "features": {},
            },
        )

    def test_dense_stars_do_not_make_preserved_nebula_look_collapsed(self) -> None:
        source, starless, starmask = self._dense_star_nebula()
        pipeline = self._pipeline()

        scores = stage7_starless_artifact_scores(
            pipeline,
            source,
            starless,
            starmask,
            measure_image_features(source),
            measure_image_features(starless),
        )
        assessment = stage7_dynamic_range_assessment(
            pipeline.cfg,
            dynamic_range_ratio=float(scores["starless_dynamic_range_ratio"]),
            peak_signal=float(scores["starless_peak_signal"]),
            background_level=float(measure_image_features(starless).bg_median),
        )

        self.assertLess(scores["starless_dynamic_range_ratio_raw"], 0.55)
        self.assertEqual(scores["dynamic_range_calibration_available"], 1.0)
        self.assertGreater(scores["starless_dynamic_range_ratio"], 0.90)
        self.assertGreater(
            scores["dynamic_range_calibration_correlation"],
            0.85,
        )
        self.assertFalse(assessment["collapsed"])

    def test_flat_starless_output_still_fails_closed(self) -> None:
        source, starless, starmask = self._dense_star_nebula()
        flat = np.full_like(starless, 0.0014)
        pipeline = self._pipeline()

        scores = stage7_starless_artifact_scores(
            pipeline,
            source,
            flat,
            starmask,
            measure_image_features(source),
            measure_image_features(flat),
        )
        assessment = stage7_dynamic_range_assessment(
            pipeline.cfg,
            dynamic_range_ratio=float(scores["starless_dynamic_range_ratio"]),
            peak_signal=float(scores["starless_peak_signal"]),
            background_level=float(measure_image_features(flat).bg_median),
        )

        self.assertEqual(scores["dynamic_range_calibration_available"], 0.0)
        self.assertTrue(assessment["collapsed"])
        self.assertTrue(assessment["hard_failed"])

    def test_missing_starmask_keeps_original_fallback_measurement(self) -> None:
        source, starless, _starmask = self._dense_star_nebula()
        source_gray = source[0]
        starless_gray = starless[0]

        calibration = stage7_calibrate_starless_dynamic_range(
            source_gray,
            starless_gray,
            None,
        )

        self.assertFalse(calibration["available"])
        self.assertEqual(calibration["method"], "full_frame_percentile_fallback")
        self.assertEqual(calibration["reason"], "starmask_unavailable")


if __name__ == "__main__":
    unittest.main()
