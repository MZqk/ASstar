#!/usr/bin/env python3
"""Tests for the headless source-pipeline development launcher."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gui import starun_pipeline_dev as launcher


class PipelineDevLauncherTests(unittest.TestCase):
    def test_parser_is_online_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            args = launcher.build_parser().parse_args(
                ["--work-dir", td]
            )

        self.assertTrue(args.network)
        self.assertFalse(args.debug)
        self.assertEqual(args.input_mode, launcher.INPUT_MODE_AUTO)

    def test_parser_accepts_explicit_offline_mode(self):
        with tempfile.TemporaryDirectory() as td:
            args = launcher.build_parser().parse_args(
                ["--work-dir", td, "--offline"]
            )

        self.assertFalse(args.network)

    def test_root_checkpoint_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "stage2_corrected.fit").touch()

            with self.assertRaisesRegex(
                ValueError,
                "验签产品任务",
            ):
                launcher.validate_work_dir(
                    work_dir,
                    "stage2_corrected_resume",
                )

    def test_parser_rejects_manual_resume_mode(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(SystemExit):
            launcher.build_parser().parse_args(
                [
                    "--work-dir",
                    td,
                    "--input-mode",
                    "stage5_linear_resume",
                ]
            )


if __name__ == "__main__":
    unittest.main()
