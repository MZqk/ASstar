from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import stage7_stretch_metrics  # noqa: E402
import stage8_color_rendition  # noqa: E402


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


class Stage7MetricsAndStage8ColorRenditionTests(unittest.TestCase):
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

        colorful_report = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            image,
            masks,
        )
        muted_report = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            muted,
            masks,
        )

        self.assertEqual(colorful_report["status"], "available")
        self.assertEqual(colorful_report["mask_source"], "stage6_frozen_roi")
        self.assertEqual(
            colorful_report["subject_coverage"],
            muted_report["subject_coverage"],
        )
        self.assertEqual(
            colorful_report["background_coverage"],
            muted_report["background_coverage"],
        )
        self.assertGreater(
            colorful_report["metrics"]["saturation_median"],
            muted_report["metrics"]["saturation_median"],
        )

    def test_rendition_reports_broad_non_background_saturation(self) -> None:
        image, masks = self._scene()
        narrow_subject = masks["subject_mask"].copy()
        narrow_subject[12:32, 12:52] = 0.0
        masks["subject_mask"] = narrow_subject
        muted_signal = image.copy()
        luma = (
            0.2126 * muted_signal[0]
            + 0.7152 * muted_signal[1]
            + 0.0722 * muted_signal[2]
        )
        muted_signal[:, 12:32, 12:52] = luma[None, 12:32, 12:52]

        report = stage7_stretch_metrics.measure_frozen_rendition_metrics(
            muted_signal,
            masks,
        )

        self.assertGreater(
            report["non_background_signal_coverage"],
            report["subject_coverage"],
        )
        self.assertLess(
            report["metrics"]["non_background_saturation_median"],
            report["metrics"]["saturation_median"],
        )

    def test_subject_chroma_boost_preserves_luminance_background_and_headroom(self) -> None:
        image, masks = self._scene()
        rendered, metadata = (
            stage8_color_rendition.apply_subject_chroma_rendition(
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

        with self.assertRaisesRegex(ValueError, "valid frozen ROI"):
            stage8_color_rendition.apply_subject_chroma_rendition(
                image,
                None,
                factor=1.18,
            )

    def test_high_factor_subject_chroma_is_still_luminance_and_gamut_bounded(self) -> None:
        image, masks = self._scene()
        rendered, metadata = stage8_color_rendition.apply_subject_chroma_rendition(
            image,
            masks,
            factor=6.0,
        )
        source_luma = (
            0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]
        )
        rendered_luma = (
            0.2126 * rendered[0]
            + 0.7152 * rendered[1]
            + 0.0722 * rendered[2]
        )

        np.testing.assert_allclose(rendered_luma, source_luma, atol=2e-7)
        self.assertLessEqual(float(np.max(rendered)), 0.995001)
        self.assertGreaterEqual(float(np.min(rendered)), 0.0)
        self.assertEqual(metadata["factor"], 4.0)
        self.assertEqual(metadata["newly_clipped_ratio"], 0.0)

    def test_faint_signal_expansion_uses_frozen_non_background_boundary(self) -> None:
        image, masks = self._scene()
        explicit_subject = masks["subject_mask"].copy()
        explicit_subject[12:32, 12:52] = 0.0
        masks["subject_mask"] = explicit_subject

        rendered, metadata = stage8_color_rendition.apply_subject_chroma_rendition(
            image,
            masks,
            factor=2.0,
            expand_faint_signal=True,
        )

        # This pixel is outside the narrow explicit subject mask but remains
        # inside the independently frozen non-background signal region.
        self.assertFalse(np.allclose(rendered[:, 24, 30], image[:, 24, 30]))
        # Frozen sky background is still bit-for-bit outside the operation.
        np.testing.assert_allclose(rendered[:, 4, 4], image[:, 4, 4])
        self.assertTrue(metadata["non_background_signal_expansion_applied"])
        self.assertEqual(metadata["non_background_signal_smoothing_passes"], 8)
        self.assertEqual(metadata["non_background_signal_opening_passes"], 8)
        self.assertEqual(metadata["chroma_smoothing_passes"], 6)
        self.assertFalse(metadata["star_protection_applied"])
        self.assertTrue(
            metadata["star_protection_skipped_for_starless_expansion"]
        )
        self.assertTrue(metadata["background_unchanged"])

    def test_starless_faint_signal_expansion_excludes_isolated_star_aperture(
        self,
    ) -> None:
        image, masks = self._scene()
        masks["background_mask"] = np.ones(
            image.shape[1:], dtype=np.float32
        )
        masks["background_mask"][20:28, 20:28] = 0.0
        masks["subject_mask"] = np.zeros(
            image.shape[1:], dtype=np.float32
        )
        masks["core_mask"] = None
        masks["nebula_mask"] = None
        masks["faint_nebula_mask"] = None
        masks["galaxy_signal_mask"] = None
        masks["star_mask"] = np.zeros(image.shape[1:], dtype=np.float32)
        masks["star_mask"][20:28, 20:28] = 1.0

        with self.assertRaisesRegex(ValueError, "insufficient support"):
            stage8_color_rendition.apply_subject_chroma_rendition(
                image,
                masks,
                factor=2.0,
                expand_faint_signal=True,
            )

    def test_starless_expansion_does_not_reintroduce_aggregate_star_island(
        self,
    ) -> None:
        image, masks = self._scene()
        masks["background_mask"] = np.ones(
            image.shape[1:], dtype=np.float32
        )
        masks["subject_mask"] = np.zeros(
            image.shape[1:], dtype=np.float32
        )
        masks["subject_mask"][20:28, 20:28] = 1.0
        masks["core_mask"] = None
        masks["nebula_mask"] = None
        masks["faint_nebula_mask"] = None
        masks["galaxy_signal_mask"] = None
        masks["star_mask"] = np.zeros(image.shape[1:], dtype=np.float32)
        masks["star_mask"][20:28, 20:28] = 1.0

        with self.assertRaisesRegex(ValueError, "insufficient support"):
            stage8_color_rendition.apply_subject_chroma_rendition(
                image,
                masks,
                factor=4.0,
                expand_faint_signal=True,
            )

    def test_composite_tone_darkens_luminance_with_one_linked_rgb_gain(self) -> None:
        image = np.asarray(
            [
                np.full((8, 8), 0.24),
                np.full((8, 8), 0.18),
                np.full((8, 8), 0.12),
            ],
            dtype=np.float32,
        )
        rendered, report = (
            stage7_stretch_metrics.apply_composite_preserving_tone(
                image,
                source_background=0.18,
                target_background=0.11,
            )
        )

        self.assertTrue(report["linked_rgb_gain"])
        self.assertGreater(report["gamma"], 1.0)
        self.assertLess(float(np.median(rendered)), float(np.median(image)))
        np.testing.assert_allclose(
            rendered[0] / rendered[1],
            image[0] / image[1],
            atol=1e-6,
        )
        self.assertEqual(report["newly_clipped_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
