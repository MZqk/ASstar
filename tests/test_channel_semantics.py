from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from channel_semantics import (  # noqa: E402
    BROADBAND_RGB_OSC,
    MONO,
    NARROWBAND_COMPOSITE,
    NONLINEAR_COLOR,
    UNKNOWN,
    classify_channel_semantics,
)


class ChannelSemanticsTests(unittest.TestCase):
    def test_single_channel_is_mono(self):
        profile = classify_channel_semantics(
            channels=1,
            input_state="linear",
        )
        self.assertEqual(profile["kind"], MONO)
        self.assertEqual(profile["action"], "skip_color_calibration")

    def test_linear_rgb_with_narrowband_filter_preserves_composite(self):
        profile = classify_channel_semantics(
            channels=3,
            input_state="linear",
            metadata={"FILTER": "Ha+OIII dual-band"},
        )
        self.assertEqual(profile["kind"], NARROWBAND_COMPOSITE)
        self.assertTrue(profile["narrowband_detected"])

    def test_linear_rgb_without_filter_evidence_is_broadband(self):
        profile = classify_channel_semantics(
            channels=3,
            input_state="linear",
            metadata={"FILTER": "LP"},
        )
        self.assertEqual(profile["kind"], BROADBAND_RGB_OSC)

    def test_unknown_linearity_does_not_guess_from_three_channels(self):
        profile = classify_channel_semantics(channels=3)
        self.assertEqual(profile["kind"], UNKNOWN)
        self.assertEqual(profile["action"], "preserve_input_review")

    def test_nonlinear_input_is_preserved(self):
        profile = classify_channel_semantics(
            channels=3,
            input_state="nonlinear",
            metadata={"FILTER": "LP"},
        )
        self.assertEqual(profile["kind"], NONLINEAR_COLOR)
        self.assertEqual(profile["action"], "preserve_input")


if __name__ == "__main__":
    unittest.main()
