#!/usr/bin/env python3
"""Targeted tests for isolated Stage11 runner."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy" not in sys.modules:
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class _SirilError(Exception):
        pass

    exceptions.CommandError = _SirilError
    exceptions.DataError = _SirilError
    exceptions.SirilError = _SirilError
    sirilpy.exceptions = exceptions
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions

STAGE11_MODULE_PATH = REPO_ROOT / "pipeline" / "stage11_ai_postprocess.py"


def _load_stage11_module():
    spec = importlib.util.spec_from_file_location(
        "stage11_ai_postprocess_test_module", STAGE11_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {STAGE11_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage11_module = _load_stage11_module()
run_stage11_ai_postprocess = stage11_module.run_stage11_ai_postprocess


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def stage_start(self, name: str) -> None:
        self.events.append(("stage_start", name))

    def stage_end(self, _name: str | None = None) -> float:
        self.events.append(("stage_end", _name or ""))
        return 0.01

    def info(self, msg: str) -> None:
        self.events.append(("info", msg))

    def warn(self, msg: str) -> None:
        self.events.append(("warn", msg))

    def debug(self, msg: str) -> None:
        self.events.append(("debug", msg))


class FakeSiril:
    def __init__(self) -> None:
        self.current_name = "stage11_ai_source"
        self.shape_map = {
            "stage11_ai_source": (3, 100, 100),
            "stage11_ai_output_fit": (3, 100, 100),
            "stage11_ai_blended": (3, 100, 100),
        }

    def get_image_pixeldata(self, preview: bool = False) -> Any:
        _ = preview
        return [[0.0]]

    def get_image_shape(self) -> Any:
        return self.shape_map.get(self.current_name, (3, 100, 100))


class FakeProcessor:
    def __init__(self, work_dir: Path) -> None:
        self.cfg = SimpleNamespace(
            ai_post_enabled=True,
            ai_endpoint="https://example.invalid/v1/chat/completions",
            ai_model="kimi-k2.5",
            ai_api_key="dummy-key",
            ai_timeout_sec=90,
            ai_strength=0.12,
            debug_mode=False,
            review_bundle_enabled=False,
        )
        self.log = FakeLogger()
        self.work_dir = work_dir
        self.process_dir = work_dir / "process"
        self.process_dir.mkdir(exist_ok=True)
        self.siril = FakeSiril()
        self.ai_outputs_generated = False
        self._ai_plan_parse_fallback = False
        self._ai_plan_parse_fallback_reason: str | None = None

        self.results: list[tuple[str, str, float, str]] = []
        self.cmd_calls: list[tuple[Any, ...]] = []
        self.unlinked: list[Path] = []
        self.stage_json: dict[str, Any] = {}
        self.request_should_raise: Exception | None = None
        self.quality_results: list[tuple[bool, list[str]]] = [(True, [])]

    def cmd_with_check(self, *args: Any) -> bool:
        self.cmd_calls.append(args)
        if args and args[0] == "load" and len(args) > 1:
            self.siril.current_name = str(args[1])
        return True

    def _record_stage(self, name: str, status: str, duration: float = 0.0, message: str = "") -> None:
        self.results.append((name, status, duration, message))

    def _measure_current_features(self) -> Any:
        return SimpleNamespace()

    def _request_ai_adjustments(self, _source_features: Any):
        if self.request_should_raise is not None:
            raise self.request_should_raise
        return {"background_protection": 0.9}, "test-plan"

    def _apply_local_ai_adjustments(self, _image_data: Any, _adjustments: Any) -> Any:
        return [[0.0]]

    def _blend_ai_images(self, _source: str, _ai: str, _out: str, _strength: float) -> None:
        return None

    def _validate_ai_quality(self, _baseline: Any, _candidate: Any):
        if self.quality_results:
            return self.quality_results.pop(0)
        return True, []

    def _safe_unlink(self, path: Path) -> None:
        self.unlinked.append(path)

    def _write_stage_json(self, name: str, payload: Any) -> None:
        self.stage_json[name] = payload


class Stage11RunnerTests(unittest.TestCase):
    def _run_with_processor(self, processor: FakeProcessor) -> FakeProcessor:
        def fake_png_writer(path: Path, _rgb: Any) -> None:
            path.write_bytes(b"PNG")

        run_stage11_ai_postprocess(
            processor,
            write_png_rgb16_func=fake_png_writer,
        )
        return processor

    def test_stage11_skipped_when_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            processor = FakeProcessor(Path(td))
            processor.cfg.ai_post_enabled = False

            self._run_with_processor(processor)

            _name, status, _dur, message = processor.results[-1]
            self.assertEqual(status, "skipped")
            self.assertIn("SEESTAR_AI_ENABLED not enabled", message)

    def test_stage11_skips_review_only_stage10_output(self):
        with tempfile.TemporaryDirectory() as td:
            processor = FakeProcessor(Path(td))
            processor._final_output_review_only = True

            self._run_with_processor(processor)

            _name, status, _dur, message = processor.results[-1]
            self.assertEqual(status, "skipped")
            self.assertIn("review-only", message)
            self.assertFalse(processor.cmd_calls)
            self.assertFalse(processor.ai_outputs_generated)

    def test_stage11_skipped_when_required_env_missing(self):
        with tempfile.TemporaryDirectory() as td:
            processor = FakeProcessor(Path(td))
            processor.cfg.ai_endpoint = ""

            self._run_with_processor(processor)

            _name, status, _dur, message = processor.results[-1]
            self.assertEqual(status, "skipped")
            self.assertIn("missing required env", message)
            self.assertIn("SEESTAR_AI_ENDPOINT", message)

    def test_stage11_success_exports_ai_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            processor = FakeProcessor(Path(td))
            processor.quality_results = [(True, [])]

            self._run_with_processor(processor)

            _name, status, _dur, message = processor.results[-1]
            self.assertEqual(status, "ok")
            self.assertTrue(processor.ai_outputs_generated)
            self.assertIn("AI outputs exported successfully", message)
            cmds = [" ".join(map(str, x)) for x in processor.cmd_calls]
            self.assertTrue(any(call.startswith("savetif result_processed_ai") for call in cmds))
            self.assertTrue(any(call.startswith("savepng result_processed_ai") for call in cmds))
            self.assertTrue(any(call == "save result_final_ai" for call in cmds))

    def test_stage11_marks_degraded_when_ai_plan_parse_fallback_triggered(self):
        with tempfile.TemporaryDirectory() as td:
            processor = FakeProcessor(Path(td))
            processor.quality_results = [(True, [])]
            processor._ai_plan_parse_fallback = True
            processor._ai_plan_parse_fallback_reason = "ai plan json parse failed"

            self._run_with_processor(processor)

            _name, status, _dur, message = processor.results[-1]
            self.assertEqual(status, "degraded")
            self.assertTrue(processor.ai_outputs_generated)
            self.assertIn("AI outputs exported successfully", message)
            self.assertIn("AI plan fallback", message)

    def test_stage11_quality_gate_rejected_after_retry(self):
        with tempfile.TemporaryDirectory() as td:
            processor = FakeProcessor(Path(td))
            processor.quality_results = [
                (False, ["star growth"]),
                (False, ["star growth"]),
            ]

            self._run_with_processor(processor)

            _name, status, _dur, message = processor.results[-1]
            self.assertEqual(status, "degraded")
            self.assertIn("quality gate rejected AI blend", message)
            self.assertFalse(processor.ai_outputs_generated)

    def test_stage11_degraded_when_request_raises(self):
        with tempfile.TemporaryDirectory() as td:
            processor = FakeProcessor(Path(td))
            processor.request_should_raise = RuntimeError("mock request failed")

            self._run_with_processor(processor)

            _name, status, _dur, message = processor.results[-1]
            self.assertEqual(status, "degraded")
            self.assertIn("AI postprocess failed", message)
            self.assertIn("mock request failed", message)


if __name__ == "__main__":
    unittest.main()
