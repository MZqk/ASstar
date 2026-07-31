#!/usr/bin/env python3
"""Tests for the isolated AI artistic-derivative experiment."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy" not in sys.modules:
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")
    enums = types.ModuleType("sirilpy.enums")

    class _SirilError(Exception):
        pass

    class _SirilConnectionError(_SirilError):
        pass

    class _SirilInterface:
        pass

    class _CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    exceptions.CommandError = _SirilError
    exceptions.DataError = _SirilError
    exceptions.SirilError = _SirilError
    exceptions.SirilConnectionError = _SirilConnectionError
    sirilpy.SirilInterface = _SirilInterface
    enums.CommandStatus = _CommandStatus
    sirilpy.exceptions = exceptions
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums

from ai_artistic_derivative import (  # noqa: E402
    build_image_edit_endpoint_candidates,
    run_ai_artistic_derivative,
)


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.events.append(("info", message))

    def warn(self, message: str) -> None:
        self.events.append(("warn", message))


class _Siril:
    def get_image_pixeldata(self, preview: bool = False) -> Any:
        _ = preview
        image = np.zeros((3, 32, 48), dtype=np.float32)
        image[:, 8:24, 12:36] = 0.16
        return image


class _Owner:
    def __init__(self, root: Path) -> None:
        self.work_dir = root
        self.process_dir = root / "process"
        self.process_dir.mkdir()
        (self.process_dir / "stage10_final.fit").write_bytes(b"FITS")
        self.cfg = SimpleNamespace(
            ai_artistic_derivative_enabled=True,
            ai_artistic_endpoint="https://example.invalid/v1/images/edits",
            ai_artistic_model="image-model",
            ai_artistic_api_key="secret",
            ai_artistic_prompt="artistic test",
            ai_artistic_timeout_sec=180,
        )
        self.log = _Logger()
        self.siril = _Siril()
        self.cmd_calls: list[tuple[Any, ...]] = []
        self.results = [("stage10", "ok")]

    def cmd_with_check(self, *args: Any) -> bool:
        self.cmd_calls.append(args)
        return True


def _write_png(path: Path, _image: Any) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\nsource")


class ArtisticDerivativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._network_patch = patch.dict(
            os.environ,
            {"SEESTAR_NETWORK_MODE": "1"},
        )
        self._network_patch.start()

    def tearDown(self) -> None:
        self._network_patch.stop()

    def test_endpoint_base_url_expands_to_image_edits(self) -> None:
        self.assertEqual(
            build_image_edit_endpoint_candidates("https://api.example.com"),
            ["https://api.example.com/v1/images/edits"],
        )
        self.assertEqual(
            build_image_edit_endpoint_candidates("https://api.example.com/v1"),
            ["https://api.example.com/v1/images/edits"],
        )

    def test_disabled_experiment_has_no_files_or_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            owner = _Owner(Path(tmpdir))
            owner.cfg.ai_artistic_derivative_enabled = False

            result = run_ai_artistic_derivative(
                owner,
                write_png_rgb16_func=_write_png,
            )

            self.assertIsNone(result)
            self.assertFalse((owner.work_dir / "ai_artistic_derivative").exists())
            self.assertEqual(owner.cmd_calls, [])
            self.assertEqual(owner.results, [("stage10", "ok")])

    def test_network_disabled_writes_skipped_report_without_request(self) -> None:
        calls: list[tuple[Any, ...]] = []

        def fake_request(*args: Any):
            calls.append(args)
            return b"", {}

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "0"}):
            with tempfile.TemporaryDirectory() as tmpdir:
                owner = _Owner(Path(tmpdir))
                result = run_ai_artistic_derivative(
                    owner,
                    write_png_rgb16_func=_write_png,
                    request_func=fake_request,
                )

                self.assertIsNone(result)
                self.assertEqual(calls, [])
                self.assertEqual(owner.cmd_calls, [])
                report = json.loads(
                    (
                        owner.work_dir
                        / "ai_artistic_derivative"
                        / "artistic_report.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(report["status"], "skipped")
                self.assertIn("NETWORK_MODE", report["reason"])

    def test_missing_isolated_credentials_only_writes_skipped_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            owner = _Owner(Path(tmpdir))
            owner.cfg.ai_artistic_api_key = ""

            result = run_ai_artistic_derivative(
                owner,
                write_png_rgb16_func=_write_png,
            )

            self.assertIsNone(result)
            report_path = owner.work_dir / "ai_artistic_derivative" / "artistic_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "skipped")
            self.assertTrue(report["isolated"])
            self.assertEqual(owner.cmd_calls, [])

    def test_success_writes_derivative_without_importing_it_into_siril(self) -> None:
        calls: list[tuple[Any, ...]] = []

        def fake_request(*args: Any):
            calls.append(args)
            source_path = Path(args[4])
            self.assertTrue(source_path.is_file())
            return b"\x89PNG\r\n\x1a\nderivative", {"transport": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            owner = _Owner(Path(tmpdir))
            result = run_ai_artistic_derivative(
                owner,
                write_png_rgb16_func=_write_png,
                request_func=fake_request,
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.is_file())
            self.assertTrue(owner.ai_artistic_output_generated)
            self.assertEqual(owner.cmd_calls, [("load", "stage10_final")])
            self.assertEqual(owner.results, [("stage10", "ok")])
            self.assertEqual(len(calls), 1)
            report = json.loads(
                (owner.work_dir / "ai_artistic_derivative" / "artistic_report.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "ok")
            self.assertFalse(report["reimported_into_siril"])
            self.assertFalse(report["affects_pipeline_status"])
            self.assertNotIn("secret", json.dumps(report))

    def test_request_failure_is_reported_without_raising(self) -> None:
        def failing_request(*_args: Any):
            raise RuntimeError("mock image endpoint failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            owner = _Owner(Path(tmpdir))
            result = run_ai_artistic_derivative(
                owner,
                write_png_rgb16_func=_write_png,
                request_func=failing_request,
            )

            self.assertIsNone(result)
            report = json.loads(
                (owner.work_dir / "ai_artistic_derivative" / "artistic_report.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertIn("mock image endpoint failure", report["error"])
            self.assertEqual(owner.results, [("stage10", "ok")])


if __name__ == "__main__":
    unittest.main()
