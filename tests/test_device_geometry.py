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
    resolve_device_report_identity,
    resolve_smart_device_profile,
    resolve_spcc_sensor_from_metadata,
    validate_active_geometry,
)


class DeviceGeometryReportTests(unittest.TestCase):
    def test_common_smart_telescope_profiles_resolve_from_fits_identity(self) -> None:
        cases = (
            ("ZWO Seestar S30", "seestar_s30_imx662", "ZWO Seestar S30", 150.0, 2.9),
            (
                "ZWO Seestar S30 Pro",
                "seestar_s30_pro_imx585",
                "Sony IMX585",
                160.0,
                2.9,
            ),
            ("Seestar S50", "seestar_s50_imx462", "ZWO Seestar S50", 250.0, 2.9),
            ("DWARFII", "dwarflab_dwarf_2_imx415", "Sony IMX415", 100.0, 1.45),
            ("DWARF 3", "dwarflab_dwarf_3_imx678", "Sony IMX678", 150.0, 2.0),
            (
                "DWARF mini",
                "dwarflab_dwarf_mini_imx662",
                "Sony IMX662",
                150.0,
                2.9,
            ),
        )
        for identity, profile_id, spcc_sensor, focal, pixel in cases:
            with self.subTest(identity=identity):
                profile = resolve_smart_device_profile({"TELESCOP": identity})
                self.assertIsNotNone(profile)
                self.assertEqual(profile["id"], profile_id)
                self.assertEqual(profile["spcc_sensor"], spcc_sensor)
                report = build_device_geometry_report(
                    {"TELESCOP": identity},
                    environ={},
                )
                self.assertEqual(
                    report["selected"]["focal_length_mm"]["value"],
                    focal,
                )
                self.assertEqual(
                    report["selected"]["pixel_size_um"]["value"],
                    pixel,
                )
                self.assertTrue(report["decision"]["would_auto_apply"])

    def test_sensor_only_header_resolves_spcc_without_assuming_geometry(self) -> None:
        sensor, source = resolve_spcc_sensor_from_metadata(
            {"DETECTOR": "SONY IMX678 STARVIS 2"}
        )
        report = build_device_geometry_report(
            {"DETECTOR": "SONY IMX678 STARVIS 2"},
            environ={},
        )

        self.assertEqual(sensor, "Sony IMX678")
        self.assertEqual(source, "fits_header:DETECTOR")
        self.assertIsNone(report["identity"]["matched_profile"])
        self.assertFalse(report["decision"]["would_auto_apply"])

    def test_unknown_qhy_setup_uses_header_identity_without_starun_fallback(self) -> None:
        metadata = {
            "INSTRUME": "QHY268M",
            "TELESCOP": "Askar 107PHQ",
            "FOCALLEN": 746.608,
            "XPIXSZ": 3.76,
            "YPIXSZ": 3.76,
        }
        identity = resolve_device_report_identity(metadata)
        report = build_device_geometry_report(metadata, environ={})

        self.assertEqual(identity["source"], "header_derived")
        self.assertEqual(identity["instrument"], "Askar 107PHQ")
        self.assertEqual(identity["instrument_source"], "fits_header:TELESCOP")
        self.assertEqual(identity["sensor"], "QHY268M")
        self.assertEqual(identity["sensor_source"], "fits_header:INSTRUME")
        self.assertIsNone(identity["device_profile_id"])
        self.assertIsNone(report["identity"]["matched_profile"])
        self.assertEqual(report["identity"]["source"], "header_derived")
        self.assertNotIn("Seestar", json.dumps(report))
        self.assertEqual(
            report["selected"]["focal_length_mm"]["value"],
            746.608,
        )
        self.assertEqual(report["selected"]["pixel_size_um"]["value"], 3.76)
        self.assertTrue(report["decision"]["would_auto_apply"])

    def test_missing_device_identity_is_reported_as_unknown(self) -> None:
        identity = resolve_device_report_identity({})

        self.assertEqual(identity["source"], "unknown")
        self.assertEqual(identity["instrument"], "unknown")
        self.assertEqual(identity["sensor"], "unknown")
        self.assertIsNone(identity["device_profile_id"])

    def test_creator_header_can_identify_a_supported_smart_telescope(self) -> None:
        profile = resolve_smart_device_profile(
            {"CREATOR": "DWARFLAB DWARF 3"}
        )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["id"], "dwarflab_dwarf_3_imx678")

    def test_wide_camera_evidence_blocks_tele_profile_geometry(self) -> None:
        report = build_device_geometry_report(
            {
                "TELESCOP": "Seestar S30 Pro",
                "SENSOR": "Sony IMX586",
                "FOCALLEN": 6.0,
            },
            environ={},
        )

        self.assertIsNone(report["identity"]["matched_profile"])
        self.assertIn(
            "imx586",
            report["identity"]["profile_rejection_reason"],
        )
        self.assertFalse(report["decision"]["would_auto_apply"])

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
                "STARUN_STAGE4_PLATESOLVE_FOCAL": "160",
                "STARUN_STAGE4_PLATESOLVE_PIXELSIZE": "2.9",
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
