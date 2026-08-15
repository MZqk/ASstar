#!/usr/bin/env python3
"""Regression tests for the copied GraXpert runtime compatibility patch."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHER_PATH = (
    REPO_ROOT
    / "resources"
    / "siril_plugins"
    / "patches"
    / "apply_graxpert_ai_runtime_patch.py"
)
UPSTREAM_SCRIPT = (
    REPO_ROOT
    / "resources"
    / "siril_plugins"
    / "vendor"
    / "siril-scripts"
    / "processing"
    / "GraXpert-AI.py"
)


def _load_patcher():
    spec = importlib.util.spec_from_file_location("graxpert_runtime_patch_test", PATCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load patcher: {PATCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GraXpertRuntimePatchTests(unittest.TestCase):
    def test_patch_requests_first_onnx_output_and_validates_shape(self) -> None:
        patcher = _load_patcher()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "GraXpert-AI.py"
            target.write_bytes(UPSTREAM_SCRIPT.read_bytes())

            self.assertTrue(patcher.apply_patch(target))
            patched = target.read_text(encoding="utf-8")

            self.assertIn("return_first_output=True", patched)
            self.assertIn("background = np.asarray(background)", patched)
            self.assertIn("Invalid GraXpert background shape", patched)
            self.assertIn('parser.add_argument("-nogpu"', patched)
            self.assertIn("BGE is CPU-only", patched)
            self.assertNotIn(
                "removing both leading axes collapses it to 2D",
                patched,
            )
            self.assertFalse(patcher.apply_patch(target))

    def test_patch_upgrades_first_generation_layout_patch(self) -> None:
        patcher = _load_patcher()
        upstream = UPSTREAM_SCRIPT.read_text(encoding="utf-8")
        legacy_pre = patcher.NEW_PRE_INFERENCE.replace(
            patcher.NEW_PATCHED_RUN_SUFFIX,
            patcher.OLD_PATCHED_RUN_SUFFIX,
        ).replace(
            patcher.NEW_PATCHED_OUTPUT_NORMALIZATION,
            patcher.OLD_PATCHED_OUTPUT_NORMALIZATION,
        )
        legacy = upstream.replace(patcher.OLD_PRE_INFERENCE, legacy_pre, 1).replace(
            patcher.OLD_PADDING,
            patcher.NEW_PADDING,
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "GraXpert-AI.py"
            target.write_text(legacy, encoding="utf-8")

            self.assertTrue(patcher.apply_patch(target))
            patched = target.read_text(encoding="utf-8")
            self.assertIn("return_first_output=True", patched)
            self.assertIn("Invalid GraXpert background shape", patched)
            self.assertIn('parser.add_argument("-nogpu"', patched)
            self.assertFalse(patcher.apply_patch(target))


if __name__ == "__main__":
    unittest.main()
