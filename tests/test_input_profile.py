#!/usr/bin/env python3
"""Regression tests for evidence-backed input-state routing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from input_profile import infer_input_profile  # noqa: E402
from models import InputState  # noqa: E402


class InputProfileTests(unittest.TestCase):
    def test_current_run_light_stack_is_trusted_linear(self) -> None:
        profile = infer_input_profile(
            input_mode="light_preprocess",
            source_path=None,
            image_data=np.full((16, 16), 0.4, dtype=np.float32),
        )

        self.assertEqual(profile.state, InputState.LINEAR)
        self.assertTrue(profile.safe_for_linear_steps)
        self.assertEqual(profile.confidence, 1.0)

    def test_result_linear_filename_without_evidence_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "result_linear.fit"
            source.write_bytes(b"not-a-fits-header")
            profile = infer_input_profile(
                input_mode="linear_resume",
                source_path=source,
                metadata={},
                image_data=None,
            )

        self.assertEqual(profile.state, InputState.UNKNOWN)
        self.assertFalse(profile.safe_for_linear_steps)
        self.assertTrue(profile.requires_review)

    def test_explicit_stretch_metadata_is_nonlinear(self) -> None:
        profile = infer_input_profile(
            input_mode="stacked",
            source_path=None,
            metadata={"STRETCHED": True},
            image_data=np.full((16, 16), 0.1, dtype=np.float32),
        )

        self.assertEqual(profile.state, InputState.NONLINEAR)
        self.assertFalse(profile.safe_for_linear_steps)

    def test_conflicting_explicit_metadata_is_unknown(self) -> None:
        profile = infer_input_profile(
            input_mode="stacked",
            source_path=None,
            metadata={"LINEAR": True, "STRETCHED": True},
            image_data=None,
        )

        self.assertEqual(profile.state, InputState.UNKNOWN)
        self.assertTrue(profile.conflicts)

    def test_acquisition_metadata_identifies_linear_stack(self) -> None:
        pixels = np.linspace(0.0, 0.08, 32 * 32, dtype=np.float32).reshape(
            32,
            32,
        )
        profile = infer_input_profile(
            input_mode="stacked",
            source_path=None,
            metadata={
                "STACKCNT": 120,
                "EXPTIME": 10.0,
                "INSTRUME": "Seestar S50",
            },
            image_data=pixels,
        )

        self.assertEqual(profile.state, InputState.LINEAR)
        self.assertTrue(profile.safe_for_linear_steps)

    def test_verified_manifest_can_authorize_linear_resume(self) -> None:
        profile = infer_input_profile(
            input_mode="linear_resume",
            source_path=None,
            metadata={},
            image_data=None,
            trusted_provenance={
                "verified": True,
                "state": "linear",
                "detail": "matching sha256",
            },
        )

        self.assertEqual(profile.state, InputState.LINEAR)
        self.assertEqual(profile.source, "verified_manifest")
        self.assertTrue(profile.safe_for_linear_steps)


if __name__ == "__main__":
    unittest.main()
