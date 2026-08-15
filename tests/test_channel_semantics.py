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

    def test_authoritative_lp_without_known_device_is_narrowband(self):
        profile = classify_channel_semantics(
            channels=3,
            input_state="linear",
            metadata={"INSTRUME": "imx585", "FILTER": "LP"},
        )
        self.assertEqual(profile["kind"], NARROWBAND_COMPOSITE)
        self.assertTrue(profile["narrowband_detected"])
        self.assertEqual(
            profile["axes"]["spectral"]["kind"],
            "dualband_ha_oiii",
        )

    def test_authoritative_filter_ignores_conflicting_secondary_field(self):
        narrowband = classify_channel_semantics(
            channels=3,
            input_state="linear",
            metadata={"FILTER": "LP", "FILTER1": "SII"},
        )
        broadband = classify_channel_semantics(
            channels=3,
            input_state="linear",
            metadata={"FILTER": "IRCUT", "FILTER1": "Dual-Band"},
        )

        self.assertEqual(narrowband["kind"], NARROWBAND_COMPOSITE)
        self.assertEqual(broadband["kind"], BROADBAND_RGB_OSC)

    def test_starun_lp_filter_aliases_are_vendor_confirmed_narrowband(self):
        for telescope in ("Seestar S30", "Seestar S30 Pro", "Seestar S50"):
            for filter_name in ("LP      ", "LP_Starless"):
                with self.subTest(telescope=telescope, filter_name=filter_name):
                    profile = classify_channel_semantics(
                        channels=3,
                        input_state="linear",
                        metadata={"TELESCOP": telescope, "FILTER": filter_name},
                    )
                    self.assertEqual(profile["kind"], NARROWBAND_COMPOSITE)
                    self.assertTrue(profile["narrowband_detected"])
                    self.assertEqual(
                        profile["device_filter_match"]["header_key"],
                        "FILTER",
                    )

    def test_dwarf_duoband_filter_aliases_are_vendor_confirmed_narrowband(self):
        for telescope in ("DWARF 3", "DWARF mini"):
            for filter_name in ("Duo-Band      ", "Dual-Band      "):
                with self.subTest(telescope=telescope, filter_name=filter_name):
                    profile = classify_channel_semantics(
                        channels=3,
                        input_state="linear",
                        metadata={"TELESCOP": telescope, "FILTER": filter_name},
                    )
                    self.assertEqual(profile["kind"], NARROWBAND_COMPOSITE)
                    self.assertTrue(profile["narrowband_detected"])
                    self.assertIsNotNone(profile["device_filter_match"])

    def test_starun_no_lp_filter_is_not_misclassified(self):
        profile = classify_channel_semantics(
            channels=3,
            input_state="linear",
            metadata={"TELESCOP": "Seestar S50", "FILTER": "No LP"},
        )
        self.assertEqual(profile["kind"], BROADBAND_RGB_OSC)
        self.assertFalse(profile["narrowband_detected"])

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

    def test_nonlinear_state_does_not_erase_narrowband_identity(self):
        profile = classify_channel_semantics(
            channels=3,
            input_state="nonlinear",
            metadata={"FILTER": "Ha+OIII dual-band"},
        )

        self.assertEqual(profile["kind"], NONLINEAR_COLOR)
        self.assertEqual(
            profile["axes"]["spectral"]["kind"],
            "dualband_ha_oiii",
        )
        self.assertEqual(profile["axes"]["transfer"]["kind"], "nonlinear")
        self.assertFalse(profile["color_operations_authorized"])

    def test_four_channel_layout_fails_closed_without_explicit_roles(self):
        profile = classify_channel_semantics(
            channels=4,
            input_state="linear",
            metadata={"FILTER": "UV/IR Cut"},
        )

        self.assertEqual(profile["kind"], UNKNOWN)
        self.assertEqual(profile["action"], "preserve_input_review")
        self.assertEqual(profile["axes"]["layout"]["kind"], "multichannel")
        self.assertIn(
            "multichannel_layout_requires_explicit_channel_roles",
            profile["unsupported_reasons"],
        )
        self.assertFalse(profile["color_operations_authorized"])


if __name__ == "__main__":
    unittest.main()
