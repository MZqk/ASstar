from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from managed_output import audit_display_visibility, export_managed_outputs  # noqa: E402
from output_color import (  # noqa: E402
    build_output_color_manifest,
    inspect_output_artifact,
)


def _test_icc_profile() -> bytes:
    profile = bytearray(128)
    profile[:4] = (128).to_bytes(4, "big")
    profile[36:40] = b"acsp"
    return bytes(profile)


def _star_visibility_fixture(
    coordinates: tuple[tuple[int, int], ...],
) -> tuple[dict, SimpleNamespace]:
    count = len(coordinates)
    y = np.asarray([item[0] for item in coordinates], dtype=np.int32)
    x = np.asarray([item[1] for item in coordinates], dtype=np.int32)
    weak = np.zeros(count, dtype=bool)
    weak[::2] = True
    return (
        {
            "status": "ok",
            "source_matched": True,
            "_source_peak_y": y,
            "_source_peak_x": x,
            "_peak_y": y.copy(),
            "_peak_x": x.copy(),
            "_weak_flags": weak,
            "_reference_local_contrast": np.full(
                count,
                0.10,
                dtype=np.float32,
            ),
            "_stage9_visibility_inner_window_size_px": np.full(
                count,
                3,
                dtype=np.int32,
            ),
            "_stage9_visibility_outer_window_size_px": np.full(
                count,
                7,
                dtype=np.int32,
            ),
        },
        SimpleNamespace(
            stage9_psf_min_sample_count=4,
            stage9_catalog_star_visibility_contrast_min=0.002,
            stage9_star_recovery_ratio_min=0.75,
            stage9_weak_star_recovery_ratio_min=0.70,
            stage9_bright_star_visibility_ratio_min=0.90,
        ),
    )


class ManagedOutputTests(unittest.TestCase):
    def test_visibility_audit_normalizes_integer_pixel_buffers(self) -> None:
        image = np.full((3, 96, 128), 0.04, dtype=np.float32)
        image[:, 24:72, 32:96] = 0.16
        encoded = np.rint(image * 65535.0).astype(np.uint16)

        float_audit = audit_display_visibility(
            image,
            target_type="emission_nebula_widefield",
        )
        integer_audit = audit_display_visibility(
            encoded,
            target_type="emission_nebula_widefield",
        )

        self.assertEqual(
            integer_audit["exposure_state"],
            float_audit["exposure_state"],
        )
        self.assertAlmostEqual(
            integer_audit["metrics"]["luminance_median"],
            float_audit["metrics"]["luminance_median"],
            places=4,
        )

    def test_generic_compact_peaks_cannot_satisfy_required_stars(self) -> None:
        image = np.full((3, 96, 128), 0.14, dtype=np.float32)
        for y, x in (
            (10, 12),
            (18, 52),
            (27, 96),
            (42, 30),
            (55, 75),
            (70, 111),
            (82, 20),
            (88, 90),
        ):
            image[:, y, x] = 0.92

        audit = audit_display_visibility(
            image,
            target_type="emission_nebula_widefield",
            stars_required=True,
        )

        star_check = audit["checks"]["star_visibility"]
        self.assertTrue(
            star_check["compact_peak_diagnostic"]["passed"],
            audit,
        )
        self.assertFalse(star_check["passed"])
        self.assertIn("star_visibility", audit["failed_checks"])

    def test_managed_derivatives_are_tagged_and_fits_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scientific = root / "result_final.fit"
            scientific.write_bytes(b"immutable scientific archive")
            original_scientific = scientific.read_bytes()
            y_grid, x_grid = np.mgrid[:80, :120]
            image = np.stack(
                (
                    0.70 * x_grid / 119.0,
                    0.70 * y_grid / 79.0,
                    0.70 * (x_grid + y_grid) / 198.0,
                ),
                axis=0,
            ).astype(np.float32)

            report = export_managed_outputs(
                image,
                work_dir=root,
                base_filename="result_processed",
                output_format="all",
                scientific_paths=(scientific,),
                icc_profile_bytes=_test_icc_profile(),
            )
            display = inspect_output_artifact(
                root / "result_processed_display_srgb.png"
            )
            editable = inspect_output_artifact(
                root / "result_processed_edit_srgb.tif"
            )
            manifest = build_output_color_manifest(
                work_dir=root,
                base_filename="result_processed",
                fit_filename="result_final",
                fallback_base="result_processed",
                fallback_fit_base="result_final",
                output_format="all",
                channel_semantics="broadband_rgb_osc",
                review_only=False,
                managed_export_report=report,
            )

            self.assertEqual(scientific.read_bytes(), original_scientific)
            self.assertTrue(report["ready"], report)
            self.assertTrue(
                report["scientific_archive"]["unchanged"]
            )
            self.assertTrue(display["display_profile_verified"])
            self.assertEqual(display["bit_depth"], 16)
            self.assertTrue(editable["icc_profile_present"])
            self.assertEqual(editable["bits_per_sample"], [16, 16, 16])
            self.assertEqual(
                manifest["mode"],
                "managed_derivatives_active",
            )
            self.assertTrue(
                manifest["summary"]["managed_export_ready"]
            )
            self.assertTrue(
                manifest["summary"]["display_visibility_verified"]
            )

    def test_dark_galaxy_display_is_stretched_then_audited_from_final_png(self) -> None:
        height, width = 180, 240
        y_grid, x_grid = np.mgrid[:height, :width]
        galaxy = np.exp(
            -0.5
            * (
                ((x_grid - width * 0.50) / (width * 0.20)) ** 2
                + ((y_grid - height * 0.52) / (height * 0.12)) ** 2
            )
        )
        green = 0.075 + 0.055 * galaxy
        image = np.stack((green * 1.03, green, green * 0.94)).astype(np.float32)
        star_coordinates = (
            (20, 30, 0.45),
            (40, 180, 0.35),
            (80, 40, 0.40),
            (130, 190, 0.50),
            (155, 90, 0.30),
            (60, 120, 0.40),
            (105, 210, 0.38),
            (145, 55, 0.42),
        )
        for y_pos, x_pos, value in star_coordinates:
            image[:, y_pos, x_pos] = np.clip(
                image[:, y_pos, x_pos] + value,
                0.0,
                1.0,
            )
        star_reference, visibility_config = _star_visibility_fixture(
            tuple((y, x) for y, x, _value in star_coordinates)
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = export_managed_outputs(
                image,
                work_dir=root,
                base_filename="result_review",
                output_format="png",
                target_type="large_galaxy",
                stars_required=True,
                star_reference=star_reference,
                star_visibility_config=visibility_config,
            )

            self.assertTrue(report["ready"], report)
            visibility = report["display_visibility"]
            self.assertEqual(visibility["input_pixels"]["status"], "failed")
            self.assertLess(
                visibility["input_pixels"]["metrics"]["green_median"],
                0.08,
            )
            self.assertEqual(
                visibility["transform"]["name"],
                "linked_review_visibility_v2",
            )
            final_png = visibility["final_png"]
            self.assertEqual(final_png["source"], "decoded_final_png")
            self.assertEqual(final_png["status"], "passed")
            self.assertAlmostEqual(
                final_png["metrics"]["luminance_median"],
                0.18,
                delta=0.015,
            )
            self.assertTrue(final_png["checks"]["galaxy_visibility"]["passed"])
            self.assertTrue(final_png["checks"]["star_visibility"]["passed"])
            self.assertTrue(
                (root / "result_review_display_srgb.png").is_file()
            )

    def test_unreadable_display_candidate_is_not_published_as_display(self) -> None:
        image = np.full((3, 96, 128), 0.075, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale_display = root / "result_review_display_srgb.png"
            stale_display.write_bytes(b"stale managed derivative")
            report = export_managed_outputs(
                image,
                work_dir=root,
                base_filename="result_review",
                output_format="png",
                target_type="large_galaxy",
                stars_required=True,
            )

            self.assertFalse(report["ready"])
            self.assertFalse(stale_display.exists())
            self.assertEqual(report["display_visibility"]["status"], "failed")
            self.assertEqual(
                report["artifacts"][0]["status"],
                "rejected_not_published",
            )
            failed = report["display_visibility"]["final_png"]["failed_checks"]
            self.assertIn("pixel_brightness", failed)
            self.assertIn("galaxy_visibility", failed)
            self.assertIn("star_visibility", failed)
            self.assertTrue(
                any("display_png_visibility_failed" in issue for issue in report["issues"])
            )


if __name__ == "__main__":
    unittest.main()
