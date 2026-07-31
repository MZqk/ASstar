from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from device_geometry import (  # noqa: E402
    activate_device_geometry_report,
    build_device_geometry_report,
    validate_active_geometry,
)


class DeviceGeometryReportTests(unittest.TestCase):
    def test_header_geometry_is_high_confidence_but_report_only(self) -> None:
        report = build_device_geometry_report(
            {
                "FOCALLEN": "250.0 mm",
                "XPIXSZ": 2.9,
                "YPIXSZ": 2.9,
                "XBINNING": 2,
                "YBINNING": 2,
                "INSTRUME": "Unknown telescope",
            },
            image_shape={"channels": 3, "height": 1080, "width": 1920},
            environ={},
        )

        self.assertEqual(report["mode"], "report_only")
        self.assertFalse(report["applied"])
        self.assertEqual(
            report["selected"]["focal_length_mm"]["source"],
            "fits_header",
        )
        self.assertAlmostEqual(
            report["selected"]["effective_pixel_size_um"]["x"],
            5.8,
        )
        self.assertTrue(report["decision"]["would_auto_apply"])
        self.assertAlmostEqual(
            report["selected"]["predicted_plate_scale_arcsec_per_pixel"]["x"],
            206.265 * 5.8 / 250.0,
        )
        json.dumps(report)

    def test_asymmetric_pixel_metadata_blocks_future_auto_apply(self) -> None:
        report = build_device_geometry_report(
            {
                "FOCALLEN": 160,
                "XPIXSZ": 2.9,
                "YPIXSZ": 3.8,
            },
            environ={},
        )

        self.assertFalse(report["decision"]["would_auto_apply"])
        self.assertTrue(
            any(
                item["reason"] == "x_y_pixel_size_mismatch"
                for item in report["conflicts"]
            )
        )

    def test_known_profile_is_advisory_and_default_is_low_confidence(self) -> None:
        profile_report = build_device_geometry_report(
            {"INSTRUME": "ZWO Seestar S30 Pro"},
            environ={},
        )
        default_report = build_device_geometry_report({}, environ={})

        self.assertEqual(
            profile_report["selected"]["focal_length_mm"]["source"],
            "known_device_profile",
        )
        self.assertTrue(profile_report["decision"]["would_auto_apply"])
        self.assertEqual(
            default_report["selected"]["focal_length_mm"]["source"],
            "legacy_default",
        )
        self.assertFalse(default_report["decision"]["would_auto_apply"])
        self.assertTrue(default_report["current_runtime"]["unchanged_by_report"])

    def test_activation_preserves_explicit_environment_override(self) -> None:
        report = build_device_geometry_report(
            {"FOCALLEN": 250, "XPIXSZ": 2.4, "YPIXSZ": 2.4},
            environ={},
        )
        activated = activate_device_geometry_report(
            report,
            enabled=True,
            environ={
                "SEESTAR_STAGE4_PLATESOLVE_FOCAL": "160",
                "SEESTAR_STAGE4_PLATESOLVE_PIXELSIZE": "2.9",
            },
        )

        self.assertFalse(activated["activation"]["applied"])
        self.assertEqual(
            activated["activation"]["reason"],
            "explicit_environment_override",
        )
        self.assertEqual(
            activated["activation"]["runtime_geometry"]["focal_length_mm"],
            160.0,
        )

    def test_wcs_scale_validation_accepts_match_and_rejects_conflict(self) -> None:
        report = activate_device_geometry_report(
            build_device_geometry_report(
                {"FOCALLEN": 250, "XPIXSZ": 2.4, "YPIXSZ": 2.4},
                environ={},
            ),
            enabled=True,
            environ={},
        )
        predicted = 206.265 * 2.4 / 250.0
        accepted = validate_active_geometry(
            report,
            {"CDELT1": -(predicted / 3600.0), "CDELT2": predicted / 3600.0},
        )
        rejected = validate_active_geometry(
            report,
            {"SECPIX": predicted * 1.20},
        )

        self.assertTrue(accepted["activation"]["validation"]["accepted"])
        self.assertFalse(rejected["activation"]["validation"]["accepted"])


if __name__ == "__main__":
    unittest.main()
