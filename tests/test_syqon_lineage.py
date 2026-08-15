#!/usr/bin/env python3
"""Atomic generation and selected-pair integrity tests for SyQon Stage 6."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import syqon_starless  # noqa: E402


class SyqonLineageTests(unittest.TestCase):
    def _pipeline(self, root: Path) -> SimpleNamespace:
        process_dir = root / "process"
        process_dir.mkdir()
        starless = process_dir / "starless.fit"
        starmask = process_dir / "starmask.fit"
        source = np.linspace(0.01, 0.25, 3 * 16 * 16, dtype=np.float32).reshape(
            3, 16, 16
        )
        fits.PrimaryHDU(data=source * 0.9).writeto(starless)
        fits.PrimaryHDU(data=source * 0.1).writeto(starmask)
        return SimpleNamespace(
            process_dir=process_dir,
            starless_file=starless,
            starmask_file=starmask,
            _selected_syqon_pair_id="raw-parent-pair",
            _selected_syqon_attempt_id="raw-parent-attempt",
            _last_syqon_exchange_report={},
            log=SimpleNamespace(warn=lambda _message: None),
        )

    def test_derived_pair_commits_atomically_and_verifies_canonical_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipeline = self._pipeline(Path(td))
            selected = syqon_starless.record_syqon_derived_generation(
                pipeline,
                generation="clean",
                details={"test": True},
            )

            self.assertIsNotNone(selected)
            pointer = pipeline.process_dir / "stage6_syqon_selected.json"
            self.assertTrue(pointer.is_file())
            attempts_root = pipeline.process_dir / ".stage6_syqon"
            self.assertFalse(any(attempts_root.glob("*.tmp")))
            verification = syqon_starless.verify_selected_syqon_pair(pipeline)
            self.assertTrue(verification["accepted"])
            self.assertEqual(verification["generation"], "clean")

            fits.PrimaryHDU(
                data=np.zeros((3, 16, 16), dtype=np.float32)
            ).writeto(pipeline.starmask_file, overwrite=True)
            rejected = syqon_starless.verify_selected_syqon_pair(pipeline)
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["failure_code"], "PAIR_MISMATCH")
            self.assertIn("canonical starmask", rejected["reason"])

    def test_stage6_pair_handoff_ignores_mutable_stage8_starless_alias(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipeline = self._pipeline(Path(td))
            pipeline._selected_syqon_pair_id = None
            pipeline._selected_syqon_attempt_id = None
            source_path = pipeline.process_dir / "stage6_input.fit"
            starless_path = pipeline.process_dir / "stage6_starless.fit"
            source = np.linspace(
                0.01,
                0.25,
                3 * 16 * 16,
                dtype=np.float32,
            ).reshape(3, 16, 16)
            fits.PrimaryHDU(data=source).writeto(source_path)
            fits.PrimaryHDU(data=source * 0.9).writeto(starless_path)

            handoff = syqon_starless.record_stage6_pair_handoff(
                pipeline,
                source_path=source_path,
                starless_path=starless_path,
            )
            self.assertTrue(handoff["accepted"])

            stage8_starless = pipeline.process_dir / "stage8_enhanced.fit"
            fits.PrimaryHDU(data=source * 0.8).writeto(stage8_starless)
            pipeline.starless_file = stage8_starless
            verified = syqon_starless.verify_stage6_pair_handoff(pipeline)
            self.assertTrue(verified["accepted"])
            self.assertEqual(
                Path(verified["paths"]["stage6_starless"]),
                starless_path.resolve(),
            )

    def test_stage6_pair_handoff_rejects_hashed_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipeline = self._pipeline(Path(td))
            pipeline._selected_syqon_pair_id = None
            source_path = pipeline.process_dir / "stage6_input.fit"
            starless_path = pipeline.process_dir / "stage6_starless.fit"
            pixels = np.full((3, 16, 16), 0.1, dtype=np.float32)
            fits.PrimaryHDU(data=pixels).writeto(source_path)
            fits.PrimaryHDU(data=pixels * 0.9).writeto(starless_path)
            syqon_starless.record_stage6_pair_handoff(
                pipeline,
                source_path=source_path,
                starless_path=starless_path,
            )

            fits.PrimaryHDU(data=np.zeros_like(pixels)).writeto(
                starless_path,
                overwrite=True,
            )
            rejected = syqon_starless.verify_stage6_pair_handoff(pipeline)
            self.assertFalse(rejected["accepted"])
            self.assertEqual(
                rejected["reason_code"],
                "stage9_stage6_pair_mismatch",
            )

    def test_old_checkpoint_without_pair_handoff_is_compatible_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipeline = self._pipeline(Path(td))

            result = syqon_starless.verify_stage6_pair_handoff(pipeline)

            self.assertFalse(result["accepted"])
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(
                result["reason_code"],
                "stage9_stage6_pair_handoff_unavailable",
            )


if __name__ == "__main__":
    unittest.main()
