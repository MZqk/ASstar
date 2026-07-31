from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from star_color_repair import (  # noqa: E402
    assess_repaired_star_layer,
    public_star_color_report,
    repair_star_layer_colors,
)


def _synthetic_star_data() -> tuple[np.ndarray, np.ndarray]:
    height, width = 160, 220
    y_grid, x_grid = np.mgrid[:height, :width]
    stars = np.full((3, height, width), 0.0001, dtype=np.float32)
    reference = np.full((3, height, width), 0.05, dtype=np.float32)
    definitions = (
        (40, 35, 0.65, (0.90, 0.65, 0.45)),
        (150, 55, 0.55, (0.55, 0.72, 0.90)),
        (100, 115, 0.45, (0.85, 0.82, 0.75)),
        (190, 120, 0.50, (0.70, 0.50, 0.85)),
    )
    for x_pos, y_pos, amplitude, color in definitions:
        profile = np.exp(
            -(
                (x_grid - x_pos) ** 2
                + (y_grid - y_pos) ** 2
            )
            / 5.0
        ).astype(np.float32)
        true_star = (
            np.asarray(color, dtype=np.float32)[:, None, None]
            * amplitude
            * profile[None, :, :]
        )
        reference += true_star
        distorted = true_star.copy()
        distorted[0] *= 1.25
        distorted[2] *= 1.35
        stars += distorted
    return np.clip(stars, 0.0, 1.0), np.clip(reference, 0.0, 1.0)


class StarColorRepairTests(unittest.TestCase):
    def test_repair_improves_chroma_without_mutating_input(self) -> None:
        stars, reference = _synthetic_star_data()
        original = stars.copy()

        candidate, report = repair_star_layer_colors(stars, reference)

        np.testing.assert_array_equal(stars, original)
        self.assertTrue(report["accepted"], report)
        self.assertGreater(
            report["metrics"]["star_chroma_improvement"],
            report["limits"]["chroma_improvement_min"],
        )
        self.assertLessEqual(
            report["metrics"]["star_flux_drift"],
            report["limits"]["star_flux_drift_max"],
        )
        self.assertEqual(report["metrics"]["nonstar_changed_ratio"], 0.0)
        validation = assess_repaired_star_layer(
            candidate,
            report["_reference_samples"],
        )
        self.assertTrue(validation["accepted"], validation)
        json.dumps(public_star_color_report(report))

    def test_post_validation_rejects_extreme_color_rewrite(self) -> None:
        stars, reference = _synthetic_star_data()
        candidate, report = repair_star_layer_colors(stars, reference)
        candidate[0] = 0.0
        candidate[2] = np.maximum(candidate[2], candidate[1] * 1.8)

        validation = assess_repaired_star_layer(
            candidate,
            report["_reference_samples"],
            chroma_error_max=0.12,
        )

        self.assertFalse(validation["accepted"])
        self.assertTrue(validation["issues"])

    def test_post_validation_ignores_samples_removed_from_final_support(self) -> None:
        stars, reference = _synthetic_star_data()
        candidate, report = repair_star_layer_colors(stars, reference)
        samples = report["_reference_samples"]
        support = np.zeros(candidate.shape[1:], dtype=bool)
        y_coord = np.asarray(samples["y"])
        x_coord = np.asarray(samples["x"])
        retained_count = y_coord.size // 2
        support[y_coord[:retained_count], x_coord[:retained_count]] = True
        compact_candidate = np.array(candidate, copy=True)
        compact_candidate[:, ~support] = 0.0

        unscoped = assess_repaired_star_layer(
            compact_candidate,
            samples,
        )
        scoped = assess_repaired_star_layer(
            compact_candidate,
            samples,
            support_mask=support,
        )

        self.assertFalse(unscoped["accepted"])
        self.assertTrue(scoped["accepted"], scoped)
        self.assertTrue(scoped["metrics"]["support_filtered"])
        self.assertLess(
            scoped["metrics"]["sample_count"],
            scoped["metrics"]["reference_sample_count"],
        )


if __name__ == "__main__":
    unittest.main()
