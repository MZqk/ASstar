from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from noise_model import (  # noqa: E402
    build_noise_model_report,
    multiscale_denoise_candidate,
)


class NoiseModelReportTests(unittest.TestCase):
    def test_color_report_is_deterministic_and_does_not_mutate_input(self) -> None:
        rng = np.random.default_rng(17)
        image = np.full((3, 192, 256), 0.08, dtype=np.float32)
        image += rng.normal(0.0, 0.004, size=image.shape).astype(np.float32)
        image[0] += rng.normal(0.0, 0.002, size=image.shape[1:]).astype(np.float32)
        original = image.copy()

        first = build_noise_model_report(
            image,
            source_checkpoint="stage5_input_linear.fit",
            channel_semantics="broadband_rgb_osc",
            max_side=128,
        )
        second = build_noise_model_report(
            image,
            source_checkpoint="stage5_input_linear.fit",
            channel_semantics="broadband_rgb_osc",
            max_side=128,
        )

        np.testing.assert_array_equal(image, original)
        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "report_only")
        self.assertFalse(first["applied_to_pixels"])
        self.assertFalse(first["consumed_by_denoiser"])
        self.assertGreater(first["background"]["sample_count"], 100)
        self.assertGreater(len(first["scales"]), 1)
        self.assertGreater(first["scales"][0]["luma_sigma"], 0.0)
        self.assertIn("r_minus_g", first["aggregate"]["chroma_sigma"])
        json.dumps(first)

    def test_mono_hwc_and_nonfinite_pixels_are_supported(self) -> None:
        image = np.linspace(0.02, 0.2, 96 * 128, dtype=np.float32).reshape(
            96,
            128,
        )
        image[0, 0] = np.nan

        report = build_noise_model_report(
            image,
            source_checkpoint="mono.fit",
            channel_semantics="mono",
            max_side=128,
        )

        self.assertEqual(report["input"]["shape_chw"], [1, 96, 128])
        self.assertEqual(report["future_advisory"]["mode"], "luminance")
        self.assertEqual(report["aggregate"]["chroma_sigma"], {})

    def test_multiscale_candidate_reduces_noise_and_preserves_structure(self) -> None:
        rng = np.random.default_rng(4)
        y, x = np.mgrid[:192, :256]
        image = np.full((3, 192, 256), 0.06, dtype=np.float32)
        nebula = np.exp(
            -(((x - 128) / 45) ** 2 + ((y - 96) / 30) ** 2)
        ).astype(np.float32)
        image += (
            nebula[None, :, :]
            * np.array([0.18, 0.126, 0.081], dtype=np.float32)[:, None, None]
        )
        image = np.clip(
            image
            + rng.normal(0.0, 0.006, size=image.shape).astype(np.float32),
            0.0,
            1.0,
        )
        original = image.copy()

        candidate, report = multiscale_denoise_candidate(image)

        np.testing.assert_array_equal(image, original)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["status"], "accepted")
        self.assertGreater(
            report["metrics"]["background_noise_reduction"],
            0.05,
        )
        self.assertGreaterEqual(
            report["metrics"]["signal_detail_retention"],
            report["limits"]["detail_retention_min"],
        )
        self.assertEqual(candidate.shape, image.shape)

    def test_multiscale_candidate_skips_low_noise_input(self) -> None:
        image = np.full((3, 128, 128), 0.05, dtype=np.float32)

        _candidate, report = multiscale_denoise_candidate(image)

        self.assertFalse(report["accepted"])
        self.assertEqual(report["status"], "skipped_low_noise")


if __name__ == "__main__":
    unittest.main()
