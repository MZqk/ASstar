from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from noise_model import (  # noqa: E402
    assess_denoise_candidate,
    build_noise_model_report,
    multiscale_denoise_candidate,
    noise_model_digest_sha256,
    validate_noise_model_report,
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

        background = nebula < 0.02
        noise_report = build_noise_model_report(
            image,
            source_checkpoint="stage5_pre_denoise.fit",
            background_mask=background,
        )
        candidate, report = multiscale_denoise_candidate(
            image,
            background_mask=background,
            signal_mask=nebula > 0.05,
            noise_model_report=noise_report,
        )

        np.testing.assert_array_equal(image, original)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["status"], "accepted")
        self.assertGreater(
            report["metrics"]["background_noise_reduction"],
            0.12,
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

    def test_common_gate_rejects_candidate_without_noise_reduction(self) -> None:
        rng = np.random.default_rng(41)
        before = np.clip(
            0.08 + rng.normal(0.0, 0.008, size=(3, 128, 128)),
            0.0,
            1.0,
        ).astype(np.float32)

        report = assess_denoise_candidate(before, before.copy())

        self.assertFalse(report["accepted"])
        self.assertEqual(report["status"], "rejected")
        self.assertIn("insufficient_noise_reduction", report["issues"])

    def test_common_gate_rejects_ngc7000_like_seven_point_nine_percent_reduction(
        self,
    ) -> None:
        rng = np.random.default_rng(7000)
        before = np.clip(
            0.08 + rng.normal(0.0, 0.008, size=(3, 160, 192)),
            0.0,
            1.0,
        ).astype(np.float32)
        after = (0.08 + 0.921 * (before - 0.08)).astype(np.float32)

        report = assess_denoise_candidate(before, after)

        reduction = report["metrics"]["background_noise_reduction"]
        self.assertGreater(reduction, 0.07)
        self.assertLess(reduction, 0.09)
        self.assertFalse(report["accepted"])
        self.assertIn("insufficient_noise_reduction", report["issues"])
        self.assertNotIn("signal_detail_retention", report["issues"])

    def test_multiscale_candidate_consumes_frozen_sky_target_and_noise_model(
        self,
    ) -> None:
        rng = np.random.default_rng(35)
        image = np.clip(
            0.08 + rng.normal(0.0, 0.007, size=(3, 160, 192)),
            0.0,
            1.0,
        ).astype(np.float32)
        background = np.zeros((160, 192), dtype=bool)
        background[:64, :] = True
        signal = np.zeros((160, 192), dtype=bool)
        signal[70:145, 50:150] = True
        noise_report = build_noise_model_report(
            image,
            source_checkpoint="stage5_pre_denoise.fit",
            background_mask=background,
        )

        _candidate, report = multiscale_denoise_candidate(
            image,
            background_mask=background,
            signal_mask=signal,
            noise_model_report=noise_report,
        )

        context = report["frozen_context"]
        self.assertEqual(
            context["background_mask_source"],
            "stage3_spatial_background_lineage",
        )
        self.assertEqual(
            context["signal_mask_source"],
            "stage5_frozen_target_structure",
        )
        self.assertTrue(context["noise_model_verified"])
        self.assertTrue(context["noise_model_consumed"])
        self.assertEqual(
            report["component_scales"]["luma"][0]["sigma_source"],
            "frozen_multiscale_noise_model",
        )

    def test_noise_model_bindings_and_digest_fail_closed(self) -> None:
        rng = np.random.default_rng(351)
        image = np.clip(
            0.08 + rng.normal(0.0, 0.007, size=(3, 160, 192)),
            0.0,
            1.0,
        ).astype(np.float32)
        background = np.zeros((160, 192), dtype=bool)
        background[:80, :] = True
        report = build_noise_model_report(
            image,
            source_checkpoint="stage5_pre_denoise.fit",
            background_mask=background,
        )

        validation = validate_noise_model_report(
            report,
            image=image,
            background_mask=background,
            source_checkpoint="stage5_pre_denoise.fit",
        )
        self.assertTrue(validation["accepted"], validation)
        self.assertEqual(len(report["input"]["pixel_sha256"]), 64)
        self.assertEqual(len(report["background"]["mask_sha256"]), 64)
        self.assertEqual(len(report["model_digest_sha256"]), 64)

        mutations = {
            "missing_input_sha": lambda value: value["input"].pop(
                "pixel_sha256"
            ),
            "mask_sha": lambda value: value["background"].update(
                mask_sha256="0" * 64
            ),
            "model_digest": lambda value: value.update(
                model_digest_sha256="0" * 64
            ),
            "missing_scales": lambda value: value.update(scales=[]),
            "nonfinite_scale": lambda value: value["scales"][0].update(
                luma_sigma=float("nan")
            ),
            "nonpositive_scale": lambda value: value["scales"][0].update(
                luma_sigma=0.0
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                tampered = copy.deepcopy(report)
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, "frozen noise model rejected"):
                    multiscale_denoise_candidate(
                        image,
                        background_mask=background,
                        noise_model_report=tampered,
                    )

        with self.assertRaisesRegex(ValueError, "frozen noise model rejected"):
            multiscale_denoise_candidate(
                image,
                background_mask=background,
                noise_model_report=None,
            )

        changed_image = image.copy()
        changed_image[0, 0, 0] += np.float32(0.001)
        with self.assertRaisesRegex(ValueError, "input_pixel_sha256"):
            multiscale_denoise_candidate(
                changed_image,
                background_mask=background,
                noise_model_report=report,
            )

        changed_mask = background.copy()
        changed_mask[120:, :] = True
        with self.assertRaisesRegex(ValueError, "background_mask_sha256"):
            multiscale_denoise_candidate(
                image,
                background_mask=changed_mask,
                noise_model_report=report,
            )

    def test_frozen_scale_sigmas_drive_candidate_thresholds(self) -> None:
        rng = np.random.default_rng(352)
        image = np.clip(
            0.08 + rng.normal(0.0, 0.008, size=(3, 160, 192)),
            0.0,
            1.0,
        ).astype(np.float32)
        background = np.zeros((160, 192), dtype=bool)
        background[:80, :] = True
        report = build_noise_model_report(
            image,
            source_checkpoint="stage5_pre_denoise.fit",
            background_mask=background,
        )
        stronger = copy.deepcopy(report)
        for scale in stronger["scales"]:
            scale["luma_sigma"] *= 1.8
            scale["opponent_sigma"]["r_minus_g"] *= 1.8
            scale["opponent_sigma"]["b_minus_g"] *= 1.8
        stronger["model_digest_sha256"] = noise_model_digest_sha256(stronger)

        first, first_report = multiscale_denoise_candidate(
            image,
            background_mask=background,
            noise_model_report=report,
        )
        second, second_report = multiscale_denoise_candidate(
            image,
            background_mask=background,
            noise_model_report=stronger,
        )

        self.assertFalse(np.array_equal(first, second))
        self.assertLess(
            first_report["component_scales"]["luma"][0]["threshold"],
            second_report["component_scales"]["luma"][0]["threshold"],
        )

    def test_common_gate_rejects_background_chroma_noise_growth(self) -> None:
        rng = np.random.default_rng(73)
        base = 0.08 + rng.normal(0.0, 0.007, size=(128, 128))
        before = np.stack((base, base, base), axis=0).astype(np.float32)
        after = before.copy()
        opponent_noise = rng.normal(0.0, 0.004, size=(128, 128)).astype(
            np.float32
        )
        after[0] += opponent_noise
        after[2] += opponent_noise

        report = assess_denoise_candidate(before, after)

        self.assertFalse(report["accepted"])
        self.assertIn("background_chroma_noise_growth", report["issues"])

    def test_common_gate_rejects_nonfinite_candidate_pixels(self) -> None:
        rng = np.random.default_rng(97)
        before = np.clip(
            0.08 + rng.normal(0.0, 0.006, size=(3, 128, 128)),
            0.0,
            1.0,
        ).astype(np.float32)
        after = before.copy()
        after[0, 4, 7] = np.nan

        report = assess_denoise_candidate(before, after)

        self.assertFalse(report["accepted"])
        self.assertIn("nonfinite_output", report["issues"])


if __name__ == "__main__":
    unittest.main()
