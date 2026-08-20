#!/usr/bin/env python3
"""Tests for the pure first-party Stage 2 crop detector."""

from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.stage2_crop_detector import (
    crop_black_boundary_evidence,
    detect_field_rotation_crop,
    detect_native_contour_crop,
    largest_rectangle_in_mask,
)


class Stage2NativeCropDetectorTests(unittest.TestCase):
    def test_largest_rectangle_fallback_stays_inside_mask(self) -> None:
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[1:7, 2:9] = 255
        mask[1:3, 2:4] = 0

        rect = largest_rectangle_in_mask(mask)

        self.assertEqual(rect, (4, 1, 5, 6))

    def test_rectangular_black_border_is_accepted_for_rgb(self) -> None:
        image = np.full((3, 100, 120), 0.02, dtype=np.float32)
        image[:, :8, :] = 0.0
        image[:, -8:, :] = 0.0
        image[:, :, :8] = 0.0
        image[:, :, -8:] = 0.0

        result = detect_native_contour_crop(image)

        self.assertTrue(result.accepted, result.as_report())
        self.assertEqual(result.rect, (8, 8, 104, 84))
        self.assertEqual(
            result.evidence["rectangle_source"], "largestinteriorrectangle"
        )

    def test_rotated_valid_footprint_produces_interior_rectangle(self) -> None:
        height = width = 128
        yy, xx = np.mgrid[:height, :width]
        valid = np.abs(xx - 63.5) + np.abs(yy - 63.5) <= 62
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[valid] = (0.020, 0.021, 0.019)

        result = detect_native_contour_crop(image)

        self.assertTrue(result.accepted, result.as_report())
        self.assertIsNotNone(result.rect)
        x, y, rect_width, rect_height = result.rect
        self.assertTrue(np.all(valid[y : y + rect_height, x : x + rect_width]))

    def test_valid_dark_sky_and_edge_signal_are_not_cropped(self) -> None:
        yy, xx = np.mgrid[:96, :120]
        gray = 0.0010 + xx.astype(np.float32) * 0.000002
        image = np.stack((gray, gray * 0.92, gray * 1.08), axis=0)
        image[0, :, :6] += 0.0008

        result = detect_native_contour_crop(image)

        self.assertFalse(result.accepted)
        self.assertIsNone(result.rect)
        self.assertEqual(result.reason, "full_frame_is_valid")

    def test_boundary_gate_rejects_valid_sky_crop(self) -> None:
        image = np.full((100, 120, 3), 0.02, dtype=np.float32)

        accepted, evidence = crop_black_boundary_evidence(
            image,
            (8, 8, 104, 84),
        )

        self.assertFalse(accepted)
        self.assertFalse(evidence["accepted"])
        self.assertEqual(
            evidence["reason"], "insufficient_near_black_boundary_evidence"
        )

    def test_mono_black_border_is_supported(self) -> None:
        image = np.full((80, 96), 0.03, dtype=np.float32)
        image[:6, :] = 0.0
        image[-6:, :] = 0.0
        image[:, :6] = 0.0
        image[:, -6:] = 0.0

        result = detect_native_contour_crop(image)

        self.assertTrue(result.accepted, result.as_report())
        self.assertIsNotNone(result.rect)

    def test_lir_failure_uses_deterministic_mask_fallback(self) -> None:
        import cv2

        class BrokenLir:
            @staticmethod
            def lir(_contour):
                raise RuntimeError("forced failure")

        image = np.full((3, 80, 96), 0.03, dtype=np.float32)
        image[:, :6, :] = 0.0
        image[:, -6:, :] = 0.0
        image[:, :, :6] = 0.0
        image[:, :, -6:] = 0.0

        result = detect_native_contour_crop(
            image,
            cv2_module=cv2,
            lir_module=BrokenLir(),
        )

        self.assertTrue(result.accepted, result.as_report())
        self.assertEqual(result.evidence["rectangle_source"], "binary_mask_fallback")

    def test_opencv_import_failure_returns_safe_no_candidate(self) -> None:
        image = np.full((3, 80, 96), 0.02, dtype=np.float32)
        real_import = builtins.__import__

        def reject_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("forced missing cv2")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_cv2):
            result = detect_native_contour_crop(image)

        self.assertFalse(result.accepted)
        self.assertIsNone(result.rect)
        self.assertEqual(result.reason, "opencv_unavailable")

    @staticmethod
    def _field_rotation_image() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(20260811)
        height, width = 432, 576
        image = np.clip(
            0.02 + rng.normal(0.0, 0.001, (height, width, 3)),
            0.0,
            1.0,
        ).astype(np.float32)
        yy, xx = np.mgrid[:height, :width]
        coverage_wedges = (
            (xx + yy < 144)
            | ((width - 1 - xx) + yy < 144)
            | (xx + (height - 1 - yy) < 144)
            | ((width - 1 - xx) + (height - 1 - yy) < 144)
        )
        image[coverage_wedges] = np.clip(
            image[coverage_wedges]
            + rng.normal(0.0, 0.006, (int(np.count_nonzero(coverage_wedges)), 3)),
            0.0,
            1.0,
        )
        return image, coverage_wedges

    def test_field_rotation_wedges_produce_interior_crop(self) -> None:
        image, coverage_wedges = self._field_rotation_image()

        result = detect_field_rotation_crop(image)

        self.assertTrue(result.accepted, result.as_report())
        self.assertEqual(result.method, "native_field_rotation")
        self.assertIsNotNone(result.rect)
        x, y, rect_width, rect_height = result.rect
        self.assertFalse(
            np.any(coverage_wedges[y : y + rect_height, x : x + rect_width])
        )
        self.assertGreater(result.evidence["edge_connected_ratio"], 0.10)

    def test_field_rotation_requires_independent_chroma_evidence(self) -> None:
        image, coverage_wedges = self._field_rotation_image()
        rng = np.random.default_rng(20260812)
        clean = np.clip(
            0.02 + rng.normal(0.0, 0.001, image.shape),
            0.0,
            1.0,
        ).astype(np.float32)
        correlated_noise = rng.normal(
            0.0,
            0.006,
            (int(np.count_nonzero(coverage_wedges)), 1),
        )
        clean[coverage_wedges] = np.clip(
            clean[coverage_wedges] + correlated_noise,
            0.0,
            1.0,
        )

        result = detect_field_rotation_crop(clean)

        self.assertFalse(result.accepted, result.as_report())
        self.assertIsNone(result.rect)

    def test_single_noisy_edge_band_lacks_corner_wedge_geometry(self) -> None:
        rng = np.random.default_rng(20260820)
        height, width = 432, 576
        image = np.clip(
            0.02 + rng.normal(0.0, 0.001, (height, width, 3)),
            0.0,
            1.0,
        ).astype(np.float32)
        image[:72] = np.clip(
            image[:72]
            + rng.normal(0.0, 0.006, image[:72].shape),
            0.0,
            1.0,
        )

        result = detect_field_rotation_crop(image)

        self.assertFalse(result.accepted, result.as_report())
        self.assertIsNone(result.rect)
        self.assertEqual(
            result.reason,
            "edge_connected_anomaly_lacks_corner_wedge_geometry",
        )
        self.assertLess(result.evidence["corner_hit_count"], 3)

    def test_interior_noise_does_not_authorize_boundary_crop(self) -> None:
        rng = np.random.default_rng(20260813)
        image = np.clip(
            0.02 + rng.normal(0.0, 0.001, (432, 576, 3)),
            0.0,
            1.0,
        ).astype(np.float32)
        image[144:288, 216:360] = np.clip(
            image[144:288, 216:360]
            + rng.normal(0.0, 0.006, (144, 144, 3)),
            0.0,
            1.0,
        )

        result = detect_field_rotation_crop(image)

        self.assertFalse(result.accepted, result.as_report())
        self.assertEqual(
            result.reason,
            "no_significant_edge_connected_coverage_anomaly",
        )

    def test_dark_gradient_nebula_and_edge_stars_are_preserved(self) -> None:
        rng = np.random.default_rng(20260814)
        height, width = 432, 576
        yy, xx = np.mgrid[:height, :width]
        background = (
            0.003
            + 0.004 * (xx / float(width - 1))
            + 0.005
            * np.exp(-((xx - 220) ** 2 + (yy - 216) ** 2) / (2.0 * 130**2))
        )
        image = np.stack(
            (background * 1.05, background, background * 0.92),
            axis=-1,
        )
        image += rng.normal(0.0, 0.00045, image.shape)
        for center_y, center_x in (
            (10, 20),
            (25, 560),
            (100, 8),
            (420, 90),
            (410, 550),
            (5, 300),
            (300, 570),
        ):
            image[
                max(0, center_y - 2) : center_y + 3,
                max(0, center_x - 2) : center_x + 3,
                :,
            ] += np.asarray((0.05, 0.045, 0.04))

        result = detect_field_rotation_crop(
            np.clip(image, 0.0, 1.0).astype(np.float32)
        )

        self.assertFalse(result.accepted, result.as_report())
        self.assertIsNone(result.rect)


if __name__ == "__main__":
    unittest.main()
