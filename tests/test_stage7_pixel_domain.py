from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


if "sirilpy.exceptions" not in sys.modules:
    package = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")
    enums = types.ModuleType("sirilpy.enums")

    class SirilError(Exception):
        pass

    class SirilConnectionError(SirilError):
        pass

    class CommandError(SirilError):
        pass

    class DataError(SirilError):
        pass

    class CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    class SirilInterface:
        def cmd(self, *_args, **_kwargs):
            return None

    exceptions.SirilError = SirilError
    exceptions.SirilConnectionError = SirilConnectionError
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    enums.CommandStatus = CommandStatus
    package.exceptions = exceptions
    package.SirilInterface = SirilInterface
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums


from stage7_pixel_domain import (  # noqa: E402
    Stage7PixelDomainError,
    canonicalize_stage7_pixels_01,
)
from stage6_services import Stage6ServiceMixin  # noqa: E402
from target_runtime import TargetRuntimeMixin  # noqa: E402


class _Runtime(TargetRuntimeMixin):
    @staticmethod
    def _short_text(value, _limit):
        return str(value)


class Stage7PixelDomainTests(unittest.TestCase):
    def test_uint16_uses_dtype_full_scale_not_observed_maximum(self):
        pixels = np.array([[0, 32768]], dtype=np.uint16)

        canonical, provenance = canonicalize_stage7_pixels_01(pixels)

        self.assertEqual(canonical.dtype, np.float32)
        self.assertAlmostEqual(float(canonical[0, 1]), 32768.0 / 65535.0, places=7)
        self.assertLess(float(canonical[0, 1]), 0.501)
        self.assertEqual(provenance["source_dtype"], "uint16")
        self.assertEqual(provenance["normalization_scale"], 65535.0)
        self.assertTrue(provenance["normalization_applied"])

    def test_uint16_and_equivalent_float32_statistics_match(self):
        codes = np.array(
            [0, 655, 1121, 1122, 1525, 7864, 32768, 60000, 65535],
            dtype=np.uint16,
        ).reshape(3, 3)
        floats = codes.astype(np.float32) / 65535.0
        runtime = _Runtime()

        integer_stats = runtime._pixel_distribution_stats(codes)
        float_stats = runtime._pixel_distribution_stats(floats)

        for key in (
            "min",
            "max",
            "median",
            "p01",
            "p50",
            "p90",
            "p99",
            "dynamic_range",
            "global_dark_ratio",
            "object_signal_ratio",
            "safe_preview_visibility_score",
            "core_peak_ratio",
        ):
            self.assertAlmostEqual(integer_stats[key], float_stats[key], places=7)
        for key in (
            "is_visibility_too_low",
            "is_nearly_black",
            "is_nearly_white",
            "invalid_dynamic_range",
        ):
            self.assertEqual(integer_stats[key], float_stats[key])
        self.assertEqual(
            integer_stats["pixel_domain"]["normalization_scale"],
            65535.0,
        )
        self.assertFalse(
            float_stats["pixel_domain"]["normalization_applied"]
        )

    def test_stage7_snapshot_preserves_siril_source_dtype_provenance(self):
        raw = np.full((3, 4, 4), 1122, dtype=np.uint16)
        raw[:, 0, 0] = 1110
        raw[:, -1, -1] = 1161
        runtime = _Runtime()
        runtime.siril = types.SimpleNamespace(
            get_image_pixeldata=lambda preview=False: raw.copy()
        )

        pixels, stats = Stage6ServiceMixin._stage7_current_pixel_snapshot(
            runtime
        )

        self.assertEqual(pixels.dtype, np.float32)
        self.assertAlmostEqual(stats["p50"], 1122.0 / 65535.0, places=7)
        self.assertEqual(stats["pixel_domain"]["source_dtype"], "uint16")
        self.assertEqual(
            stats["pixel_domain"]["normalization_scale"],
            65535.0,
        )

    def test_uint32_normalizes_by_its_own_full_scale(self):
        uint16_codes = np.array([0, 1, 32768, 65535], dtype=np.uint16)
        uint32_codes = (
            uint16_codes.astype(np.uint64) * np.uint64(65537)
        ).astype(np.uint32)

        normalized16, _ = canonicalize_stage7_pixels_01(uint16_codes)
        normalized32, provenance32 = canonicalize_stage7_pixels_01(uint32_codes)

        self.assertTrue(np.array_equal(normalized16, normalized32))
        self.assertEqual(provenance32["source_dtype"], "uint32")
        self.assertEqual(provenance32["normalization_scale"], 4294967295.0)

    def test_m8_incident_medians_are_in_the_unit_domain(self):
        runtime = _Runtime()

        for raw_p50, expected in (
            (1122, 0.017120622568093384),
            (1525, 0.023270008392462044),
            (7864, 0.11999694819562066),
        ):
            with self.subTest(raw_p50=raw_p50):
                pixels = np.full((3, 4, 4), raw_p50, dtype=np.uint16)
                stats = runtime._pixel_distribution_stats(pixels)
                self.assertAlmostEqual(stats["p50"], expected, places=7)
                self.assertFalse(stats["is_nearly_white"])

    def test_invalid_float_and_ambiguous_signed_integer_fail_closed(self):
        for pixels in (
            np.array([0.0, np.nan], dtype=np.float32),
            np.array([0.0, np.inf], dtype=np.float32),
            np.array([-0.01, 0.5], dtype=np.float32),
            np.array([0.5, 1.01], dtype=np.float32),
            np.array([0, 32767], dtype=np.int16),
        ):
            with self.subTest(dtype=pixels.dtype, values=pixels.tolist()):
                with self.assertRaises(Stage7PixelDomainError):
                    canonicalize_stage7_pixels_01(pixels)

        stats = _Runtime()._pixel_distribution_stats(
            np.array([0.0, np.nan], dtype=np.float32)
        )
        self.assertTrue(stats["invalid_dynamic_range"])
        self.assertIn("NaN or Inf", stats["error"])


if __name__ == "__main__":
    unittest.main()
