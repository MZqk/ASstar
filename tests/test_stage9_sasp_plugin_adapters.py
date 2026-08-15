#!/usr/bin/env python3
"""Focused tests for Stage 9's direct SASP star-plugin adapters."""
from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import sasp_runner  # noqa: E402


class _Log:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warn(self, message: str) -> None:
        self.messages.append(("warn", message))


class _Pipeline:
    def __init__(self, root: Path, *, semantics: str = "narrowband_composite") -> None:
        self.siril_plugin_dir = root
        downloads = root / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        # The direct adapter only needs a bundled SASP resource to be present.
        # Force the NumPy-compatible kernel below so this test stays offline and
        # independent of optional PyQt6/Numba/OpenCV runtime dependencies.
        (downloads / "setiastrosuitepro-test.whl").write_bytes(b"fixture")
        self._sasp_star_stretch_module = None
        self._sasp_star_stretch_module_error = "forced test fallback"
        self._store: dict[str, object] = {}
        self.log = _Log()
        self.cfg = SimpleNamespace(
            stage9_sasp_star_stretch_enabled=True,
            stage9_sasp_star_stretch_amount=1.0,
            stage9_nb_to_rgb_stars_enabled=True,
            stage9_nb_to_rgb_stars_ratio=0.25,
            stage4_nbn_mapping_confidence_min=0.85,
        )
        self._channel_semantics = semantics
        self.narrowband_channel_mapping = {
            "schema": "starun.narrowband-channel-mapping.v1",
            "mapping": "osc_hoo_rgb",
            "ha_channel": "R",
            "oiii_channels": ["G", "B"],
            "confidence": 0.95,
        }
        self.channel_profile = {}
        self.workflow_command_used: dict[str, str] = {}
        self.pixels = np.zeros((3, 4, 4), dtype=np.float32)
        self.pixels[0] = 0.10
        self.pixels[1] = 0.20
        self.pixels[2] = 0.40
        self.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: self.pixels.copy(),
            set_image_pixeldata=lambda pixels: setattr(
                self,
                "pixels",
                np.array(pixels, copy=True),
            ),
            image_lock=lambda: nullcontext(),
        )

    def _find_latest_sasp_wheel(self):
        return sasp_runner.find_latest_sasp_wheel(self)

    @staticmethod
    def _short_text(error: object) -> str:
        return str(error)


class Stage9SaspPluginAdapterTests(unittest.TestCase):
    def test_confirmed_narrowband_runs_nb_to_rgb_then_star_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _Pipeline(Path(tmp))

            nb_label = sasp_runner.run_nb_to_rgb_stars_api(pipeline)
            np.testing.assert_allclose(
                pipeline.pixels[:, 0, 0],
                np.asarray((0.10, 0.25, 0.30), dtype=np.float32),
            )
            stretch_label = sasp_runner.run_sasp_star_stretch_api(pipeline)

        self.assertEqual(nb_label, "NB to RGB Stars API")
        self.assertEqual(stretch_label, "SASP Star Stretch API")
        np.testing.assert_allclose(
            pipeline.pixels[:, 0, 0],
            np.asarray((0.25, 0.50, 0.5625), dtype=np.float32),
        )
        self.assertEqual(pipeline._stage9_nb_to_rgb_stars_report["status"], "applied")
        self.assertEqual(
            pipeline._stage9_sasp_star_stretch_report["status"],
            "applied",
        )
        self.assertEqual(
            pipeline._stage9_sasp_star_stretch_report["engine"],
            "numpy_compatible_fallback",
        )

    def test_nb_to_rgb_fails_closed_for_broadband_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _Pipeline(Path(tmp), semantics="broadband_rgb_osc")
            before = pipeline.pixels.copy()

            label = sasp_runner.run_nb_to_rgb_stars_api(pipeline)

        self.assertIsNone(label)
        np.testing.assert_array_equal(pipeline.pixels, before)
        self.assertEqual(pipeline._stage9_nb_to_rgb_stars_report["status"], "skipped")
        self.assertEqual(
            pipeline._stage9_nb_to_rgb_stars_report["reason"],
            "channel_semantics_not_narrowband_composite",
        )


if __name__ == "__main__":
    unittest.main()
