from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from narrowband_normalization import (  # noqa: E402
    classify_dual_narrowband_mapping,
    normalize_dual_narrowband_candidate,
)


def _synthetic_hoo() -> np.ndarray:
    rng = np.random.default_rng(4)
    height, width = 192, 256
    y_grid, x_grid = np.mgrid[:height, :width]
    ha = np.exp(
        -(((x_grid - 132) / 55) ** 2 + ((y_grid - 96) / 38) ** 2)
    )
    oiii = np.exp(
        -(((x_grid - 104) / 48) ** 2 + ((y_grid - 90) / 46) ** 2)
    )
    image = np.empty((3, height, width), dtype=np.float32)
    image[0] = 0.045 + 0.36 * ha
    image[1] = 0.068 + 0.18 * oiii
    image[2] = 0.082 + 0.25 * oiii
    image += rng.normal(0.0, 0.002, image.shape).astype(np.float32)
    for x_pos, y_pos, amplitude in (
        (50, 40, 0.55),
        (180, 70, 0.65),
        (120, 140, 0.45),
        (210, 150, 0.50),
    ):
        star = np.exp(
            -(
                (x_grid - x_pos) ** 2
                + (y_grid - y_pos) ** 2
            )
            / 2.5
        )
        image += amplitude * star[None, :, :]
    return np.clip(image, 0.0, 1.0)


class NarrowbandNormalizationTests(unittest.TestCase):
    def test_mapping_requires_identified_emission_lines(self) -> None:
        explicit = classify_dual_narrowband_mapping(
            {"FILTER": "Ha + OIII dual-band"}
        )
        generic = classify_dual_narrowband_mapping(
            {"FILTER": "generic dualband"}
        )
        unknown = classify_dual_narrowband_mapping({})

        self.assertGreaterEqual(explicit["confidence"], 0.85)
        self.assertLess(generic["confidence"], 0.85)
        self.assertEqual(unknown["mapping"], "unknown")

    def test_candidate_is_guarded_and_does_not_mutate_input(self) -> None:
        image = _synthetic_hoo()
        original = image.copy()

        candidate, report = normalize_dual_narrowband_candidate(
            image,
            metadata={"FILTER": "Ha OIII dual-band"},
        )

        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(candidate, image))
        self.assertTrue(report["accepted"], report)
        self.assertLessEqual(
            report["metrics"]["ha_oiii_ratio_drift"],
            report["limits"]["ha_oiii_ratio_drift_max"],
        )
        self.assertLessEqual(
            report["metrics"]["star_chroma_drift"],
            report["limits"]["star_chroma_drift_max"],
        )
        self.assertLessEqual(
            report["metrics"]["star_mask_coverage"],
            report["limits"]["star_mask_coverage_max"],
        )
        self.assertGreater(
            report["metrics"]["background_color_improvement"],
            0.0,
        )
        json.dumps(report)

    def test_unconfirmed_mapping_is_rejected_before_processing(self) -> None:
        with self.assertRaisesRegex(ValueError, "mapping confidence"):
            normalize_dual_narrowband_candidate(
                _synthetic_hoo(),
                metadata={"FILTER": "generic dualband"},
            )


if __name__ == "__main__":
    unittest.main()
