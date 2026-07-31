#!/usr/bin/env python3
"""Tests for the headless source-pipeline development launcher."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gui import seestar_pipeline_dev as launcher


class PipelineDevLauncherTests(unittest.TestCase):
    def test_parser_is_offline_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            args = launcher.build_parser().parse_args(
                ["--work-dir", td]
            )

        self.assertFalse(args.network)
        self.assertFalse(args.debug)
        self.assertEqual(args.input_mode, launcher.INPUT_MODE_AUTO)

    def test_stage2_resume_accepts_root_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "stage2_corrected.fit").touch()

            launcher.validate_work_dir(
                work_dir,
                launcher.INPUT_MODE_STAGE2_CORRECTED_RESUME,
            )

    def test_stage2_resume_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(
                ValueError,
                "stage2_corrected.fit",
            ):
                launcher.validate_work_dir(
                    Path(td),
                    launcher.INPUT_MODE_STAGE2_CORRECTED_RESUME,
                )


if __name__ == "__main__":
    unittest.main()
