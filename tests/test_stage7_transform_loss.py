from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import stage7_stretch_metrics  # noqa: E402


class Stage7TransformLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = np.asarray(
            [
                [[0.0, 0.002], [0.020, 0.999]],
                [[1e-8, 0.003], [0.500, 0.994]],
                [[0.100, 0.004], [0.996, 0.800]],
            ],
            dtype=np.float32,
        )
        self.candidate = np.asarray(
            [
                [[0.0, 0.0], [0.020, 1.0]],
                [[0.0, 0.003], [0.500, 0.994]],
                [[0.100, 0.0], [1.0, 0.800]],
            ],
            dtype=np.float32,
        )

    def test_exact_zero_clip_channel_and_frozen_roi_ratios(self) -> None:
        frozen_roi = np.asarray([[True, True], [False, False]])

        report = stage7_stretch_metrics.assess_transform_loss(
            self.source,
            self.candidate,
            method="asinh",
            params={"asinh_offset": 0.003},
            background_mask=frozen_roi,
        )

        self.assertEqual(report["schema"], "starun.stage7-transform-loss.v1")
        self.assertEqual(report["role"], "report_only")
        self.assertFalse(report["participates_in_selection"])
        self.assertEqual(report["zero_epsilon"], 1e-7)
        self.assertEqual(report["near_black"], 0.010)
        self.assertEqual(report["near_highlight"], 0.995)

        full = report["global"]
        self.assertAlmostEqual(full["zero"]["before"], 2.0 / 12.0)
        self.assertAlmostEqual(full["zero"]["after"], 4.0 / 12.0)
        self.assertAlmostEqual(full["zero"]["newly_zeroed_ratio"], 2.0 / 12.0)
        self.assertAlmostEqual(full["zero_ratio_before"], 2.0 / 12.0)
        self.assertAlmostEqual(full["zero_ratio_after"], 4.0 / 12.0)
        self.assertAlmostEqual(full["newly_zeroed_ratio"], 2.0 / 12.0)
        self.assertAlmostEqual(
            full["unexpected_newly_zeroed_ratio"],
            1.0 / 12.0,
        )
        self.assertAlmostEqual(full["source_below_blackpoint_ratio"], 4.0 / 12.0)
        self.assertAlmostEqual(
            full["zero"]["newly_zeroed_ratio_by_channel"]["r"],
            0.25,
        )
        self.assertAlmostEqual(
            full["zero"]["newly_zeroed_ratio_by_channel"]["g"],
            0.0,
        )
        self.assertAlmostEqual(
            full["zero"]["newly_zeroed_ratio_by_channel"]["b"],
            0.25,
        )
        self.assertAlmostEqual(
            full["hard_high_clip"]["newly_clipped_ratio"],
            2.0 / 12.0,
        )
        self.assertAlmostEqual(
            full["effective_blackpoint"]["source_below_blackpoint_ratio"],
            4.0 / 12.0,
        )

        roi = report["background_roi"]
        self.assertEqual(roi["spatial_pixel_count"], 2)
        self.assertAlmostEqual(roi["zero"]["newly_zeroed_ratio"], 2.0 / 6.0)
        self.assertAlmostEqual(
            roi["unexpected_newly_zeroed_ratio"],
            1.0 / 6.0,
        )
        self.assertAlmostEqual(
            roi["effective_blackpoint"]["source_below_blackpoint_ratio"],
            4.0 / 6.0,
        )

    def test_effective_blackpoint_is_method_specific(self) -> None:
        linked = stage7_stretch_metrics.assess_transform_loss(
            self.source,
            self.candidate,
            method="linked_mtf",
            params={"mtf_shadows": 0.004},
        )
        quantile = stage7_stretch_metrics.assess_transform_loss(
            self.source,
            self.candidate,
            method="adaptive_quantile",
            params={},
        )

        self.assertEqual(linked["effective_blackpoint"]["source"], "mtf_shadows")
        self.assertEqual(
            quantile["effective_blackpoint"]["status"],
            "not_applicable",
        )

    def test_asinh_semantics_fix_rgbblend_without_human_weighting(self) -> None:
        for method in ("asinh", "asinh_ghs", "bright_nebula_hdr_masked"):
            with self.subTest(method=method):
                semantics = stage7_stretch_metrics.build_siril_stretch_semantics(
                    method,
                    {"asinh_stretch": 3.2, "asinh_offset": 0.0012},
                )
                asinh = semantics["steps"][0]
                self.assertEqual(semantics["luminance_mode"], "mean_rgb")
                self.assertFalse(semantics["human_weighted"])
                self.assertEqual(semantics["clip_mode"], "rgbblend")
                self.assertIn("-clipmode=rgbblend", asinh["full_argv"])
                self.assertNotIn("-human", asinh["full_argv"])
                self.assertEqual(semantics["bundled_reference_version"], "1.4.4")


class Stage7RenditionTests(unittest.TestCase):
    @staticmethod
    def _scene() -> tuple[np.ndarray, dict[str, np.ndarray]]:
        height = width = 64
        yy, xx = np.mgrid[:height, :width]
        texture = 0.015 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
        image = np.full((3, height, width), 0.08, dtype=np.float32)
        subject = np.zeros((height, width), dtype=np.float32)
        subject[12:52, 12:52] = 1.0
        image[0] += subject * (0.34 + texture)
        image[1] += subject * (0.20 + texture)
        image[2] += subject * (0.11 + texture)
        core = np.zeros_like(subject)
        core[27:37, 27:37] = 1.0
        star = np.zeros_like(subject)
        star[17:22, 17:22] = 1.0
        return image, {
            "subject_mask": subject,
            "background_mask": 1.0 - subject,
            "core_mask": core,
            "star_mask": star,
        }

    def test_frozen_roi_is_candidate_invariant(self) -> None:
        image, masks = self._scene()
        muted = image.copy()
        luminance = (
            0.2126 * muted[0] + 0.7152 * muted[1] + 0.0722 * muted[2]
        )
        muted = luminance[None, :, :] + 0.45 * (
            muted - luminance[None, :, :]
        )

        vivid_report = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            image,
            masks,
        )
        muted_report = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            muted,
            masks,
        )

        self.assertEqual(vivid_report["status"], "available")
        self.assertEqual(vivid_report["mask_source"], "stage6_frozen_roi")
        self.assertEqual(
            vivid_report["subject_coverage"],
            muted_report["subject_coverage"],
        )
        self.assertEqual(
            vivid_report["background_coverage"],
            muted_report["background_coverage"],
        )
        self.assertGreater(
            vivid_report["metrics"]["saturation_median"],
            muted_report["metrics"]["saturation_median"],
        )

    def test_subject_chroma_boost_preserves_luminance_background_and_headroom(self) -> None:
        image, masks = self._scene()
        rendered, metadata = (
            stage7_stretch_metrics.apply_subject_chroma_rendition(
                image,
                masks,
                factor=1.18,
            )
        )
        source_luma = (
            0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]
        )
        rendered_luma = (
            0.2126 * rendered[0]
            + 0.7152 * rendered[1]
            + 0.0722 * rendered[2]
        )
        background = masks["background_mask"] > 0.5
        before = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            image,
            masks,
        )
        after = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            rendered,
            masks,
        )

        np.testing.assert_allclose(rendered_luma, source_luma, atol=2e-7)
        np.testing.assert_allclose(rendered[:, background], image[:, background])
        self.assertLessEqual(float(np.max(rendered)), 0.995001)
        self.assertGreaterEqual(float(np.min(rendered)), 0.0)
        self.assertEqual(metadata["newly_clipped_ratio"], 0.0)
        self.assertTrue(metadata["background_unchanged"])
        self.assertTrue(metadata["star_protection_applied"])
        self.assertGreater(
            after["metrics"]["saturation_median"],
            before["metrics"]["saturation_median"],
        )

        with self.assertRaisesRegex(ValueError, "frozen Stage 6 ROI"):
            stage7_stretch_metrics.apply_subject_chroma_rendition(
                image,
                None,
                factor=1.18,
            )


if __name__ == "__main__":
    unittest.main()
