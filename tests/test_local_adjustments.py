from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from local_adjustments import (  # noqa: E402
    LOCAL_ADJUSTMENT_SCHEMA,
    apply_local_adjustment_recipe,
    build_local_masks,
    dilate_mask,
    erode_mask,
    feather_mask,
)


def _synthetic_nebula() -> np.ndarray:
    height, width = 180, 240
    y_grid, x_grid = np.mgrid[:height, :width]
    nebula = np.exp(
        -(((x_grid - 125) / 58) ** 2 + ((y_grid - 90) / 42) ** 2)
    ).astype(np.float32)
    image = np.full((3, height, width), 0.035, dtype=np.float32)
    image += (
        np.asarray([0.26, 0.15, 0.10], dtype=np.float32)[:, None, None]
        * nebula[None]
    )
    return np.clip(image, 0.0, 1.0)


class LocalAdjustmentTests(unittest.TestCase):
    def test_masks_and_morphology_are_bounded(self) -> None:
        image = _synthetic_nebula()
        report = build_local_masks(image)
        mask = report["masks"]["subject"]

        dilated = dilate_mask(mask > 0.5, 1)
        eroded = erode_mask(mask > 0.5, 1)
        feathered = feather_mask(mask, 2)

        self.assertGreaterEqual(np.sum(dilated), np.sum(mask > 0.5))
        self.assertLessEqual(np.sum(eroded), np.sum(mask > 0.5))
        self.assertGreaterEqual(float(np.min(feathered)), 0.0)
        self.assertLessEqual(float(np.max(feathered)), 1.0)

    def test_recipe_is_deterministic_guarded_and_nonmutating(self) -> None:
        image = _synthetic_nebula()
        original = image.copy()
        recipe = {
            "schema": LOCAL_ADJUSTMENT_SCHEMA,
            "id": "test_nebula",
            "operations": [
                {
                    "type": "curve",
                    "mask": "faint",
                    "points": (
                        (0.0, 0.0),
                        (0.20, 0.21),
                        (0.55, 0.56),
                        (1.0, 1.0),
                    ),
                    "opacity": 0.30,
                },
                {
                    "type": "saturation",
                    "mask": "subject",
                    "amount": 0.03,
                    "opacity": 0.50,
                },
            ],
        }

        first, first_report = apply_local_adjustment_recipe(image, recipe)
        second, second_report = apply_local_adjustment_recipe(image, recipe)

        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_report, second_report)
        self.assertTrue(first_report["accepted"], first_report)
        self.assertGreater(
            first_report["metrics"]["changed_pixel_ratio"],
            0.0,
        )
        json.dumps(first_report)

    def test_nonmonotonic_curve_is_rejected(self) -> None:
        recipe = {
            "schema": LOCAL_ADJUSTMENT_SCHEMA,
            "operations": [
                {
                    "type": "curve",
                    "mask": "subject",
                    "points": ((0.0, 0.0), (0.5, 0.7), (1.0, 0.6)),
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "monotonic"):
            apply_local_adjustment_recipe(_synthetic_nebula(), recipe)


if __name__ == "__main__":
    unittest.main()
