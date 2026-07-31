from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from managed_output import export_managed_outputs  # noqa: E402
from output_color import (  # noqa: E402
    build_output_color_manifest,
    inspect_output_artifact,
)


def _test_icc_profile() -> bytes:
    profile = bytearray(128)
    profile[:4] = (128).to_bytes(4, "big")
    profile[36:40] = b"acsp"
    return bytes(profile)


class ManagedOutputTests(unittest.TestCase):
    def test_managed_derivatives_are_tagged_and_fits_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scientific = root / "result_final.fit"
            scientific.write_bytes(b"immutable scientific archive")
            original_scientific = scientific.read_bytes()
            y_grid, x_grid = np.mgrid[:80, :120]
            image = np.stack(
                (
                    x_grid / 119.0,
                    y_grid / 79.0,
                    (x_grid + y_grid) / 198.0,
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


if __name__ == "__main__":
    unittest.main()
