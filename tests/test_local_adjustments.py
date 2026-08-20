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
        saturation = first_report["operations"][1]
        self.assertEqual(saturation["requested_amount"], 0.03)
        self.assertEqual(saturation["amount"], 0.03)
        self.assertGreater(saturation["effective_amount_peak"], 0.0)
        self.assertEqual(saturation["effect_status"], "effective")
        self.assertEqual(
            first_report["saturation_effect"]["status"],
            "effective",
        )
        json.dumps(first_report)

    def test_saturation_no_effect_is_reported_explicitly(self) -> None:
        image = np.full((3, 64, 64), 0.20, dtype=np.float32)
        recipe = {
            "schema": LOCAL_ADJUSTMENT_SCHEMA,
            "id": "test_no_effect",
            "operations": [
                {
                    "type": "saturation",
                    "mask": "subject",
                    "amount": 0.20,
                    "opacity": 0.50,
                }
            ],
        }

        candidate, report = apply_local_adjustment_recipe(
            image,
            recipe,
            masks={"subject": np.ones((64, 64), dtype=np.float32)},
        )

        np.testing.assert_array_equal(candidate, image)
        self.assertFalse(report["accepted"])
        self.assertIn("no_effect", report["issues"])
        self.assertEqual(report["saturation_effect"]["status"], "no_effect")
        self.assertEqual(report["operations"][0]["effect_status"], "no_effect")

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

    def test_hue_selective_saturation_targets_only_requested_color_band(self) -> None:
        image = np.empty((3, 120, 160), dtype=np.float32)
        image[:, :, :80] = np.asarray([0.32, 0.16, 0.14], dtype=np.float32)[
            :, None, None
        ]
        image[:, :, 80:] = np.asarray([0.14, 0.18, 0.32], dtype=np.float32)[
            :, None, None
        ]
        recipe = {
            "schema": LOCAL_ADJUSTMENT_SCHEMA,
            "id": "test_hue_selective",
            "operations": [
                {
                    "type": "hue_selective_saturation",
                    "mask": "subject",
                    "profile": "test_red",
                    "bands": [
                        {
                            "id": "red",
                            "center": 0.0,
                            "width": 30.0,
                            "feather": 0.80,
                            "amount": 0.10,
                        }
                    ],
                    "opacity": 1.0,
                }
            ],
        }

        candidate, report = apply_local_adjustment_recipe(
            image,
            recipe,
            masks={"subject": np.ones((120, 160), dtype=np.float32)},
        )

        before_luma = np.tensordot(
            np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
            image,
            axes=(0, 0),
        )
        after_luma = np.tensordot(
            np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
            candidate,
            axes=(0, 0),
        )
        before_chroma = np.max(image, axis=0) - np.min(image, axis=0)
        after_chroma = np.max(candidate, axis=0) - np.min(candidate, axis=0)

        self.assertTrue(report["accepted"], report)
        self.assertLess(float(np.max(np.abs(after_luma - before_luma))), 1e-6)
        self.assertGreater(
            float(np.mean(after_chroma[:, :80] - before_chroma[:, :80])),
            0.005,
        )
        self.assertLess(
            float(np.max(np.abs(candidate[:, :, 80:] - image[:, :, 80:]))),
            1e-6,
        )
        operation = report["operations"][0]
        self.assertEqual(operation["type"], "hue_selective_saturation")
        self.assertEqual(operation["profile"], "test_red")
        self.assertEqual(operation["bands"][0]["id"], "red")

    def test_hue_selective_saturation_rejects_excessive_chroma_growth(self) -> None:
        image = np.empty((3, 64, 64), dtype=np.float32)
        image[:] = np.asarray([0.75, 0.10, 0.10], dtype=np.float32)[
            :, None, None
        ]
        recipe = {
            "schema": LOCAL_ADJUSTMENT_SCHEMA,
            "operations": [
                {
                    "type": "hue_selective_saturation",
                    "mask": "subject",
                    "bands": [
                        {
                            "id": "red",
                            "center": 0.0,
                            "width": 40.0,
                            "feather": 0.80,
                            "amount": 0.15,
                        }
                    ],
                }
            ],
        }

        _candidate, report = apply_local_adjustment_recipe(
            image,
            recipe,
            masks={"subject": np.ones((64, 64), dtype=np.float32)},
        )

        self.assertFalse(report["accepted"], report)
        self.assertIn("active_chroma_p95_growth", report["issues"])


if __name__ == "__main__":
    unittest.main()
