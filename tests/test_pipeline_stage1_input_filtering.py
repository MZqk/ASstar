#!/usr/bin/env python3
"""Stage 1 input filtering regression tests."""

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
PIPELINE_MODULE_PATH = PIPELINE_DIR / "starun.py"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _ensure_fake_sirilpy() -> None:
    if "sirilpy" in sys.modules:
        return

    fake_sirilpy = types.ModuleType("sirilpy")
    fake_exceptions = types.ModuleType("sirilpy.exceptions")
    fake_enums = types.ModuleType("sirilpy.enums")

    class _SirilError(Exception):
        pass

    class _SirilConnectionError(_SirilError):
        pass

    class _CommandError(_SirilError):
        pass

    class _DataError(_SirilError):
        pass

    class _CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    class _SirilInterface:
        def cmd(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    fake_sirilpy.SirilInterface = _SirilInterface
    fake_exceptions.SirilError = _SirilError
    fake_exceptions.SirilConnectionError = _SirilConnectionError
    fake_exceptions.CommandError = _CommandError
    fake_exceptions.DataError = _DataError
    fake_enums.CommandStatus = _CommandStatus

    sys.modules["sirilpy"] = fake_sirilpy
    sys.modules["sirilpy.exceptions"] = fake_exceptions
    sys.modules["sirilpy.enums"] = fake_enums


def _ensure_fake_numpy() -> None:
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # type: ignore

        _ = numpy
        return
    except Exception:
        pass

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = float
    fake_numpy.uint16 = int
    fake_numpy.uint8 = int
    fake_numpy.integer = int
    fake_numpy.ndarray = object

    def _asarray(value: Any):
        return value

    def _transpose(value: Any, _axes: Any):
        return value

    def _issubdtype(_lhs: Any, _rhs: Any) -> bool:
        return False

    def _clip(value: Any, _vmin: Any, _vmax: Any):
        return value

    fake_numpy.asarray = _asarray
    fake_numpy.transpose = _transpose
    fake_numpy.issubdtype = _issubdtype
    fake_numpy.clip = _clip
    sys.modules["numpy"] = fake_numpy


def _load_pipeline_module():
    _ensure_fake_numpy()
    _ensure_fake_sirilpy()
    spec = importlib.util.spec_from_file_location(
        "starun_pipeline_stage1_filter_test_module",
        PIPELINE_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {PIPELINE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline_module = _load_pipeline_module()


class _FakeLogger:
    def stage_start(self, _name: str) -> None:
        return

    def stage_end(self, _name: str | None = None) -> float:
        return 0.01

    def info(self, _msg: str) -> None:
        return

    def warn(self, _msg: str) -> None:
        return

    def error(self, _msg: str) -> None:
        return

    def debug(self, _msg: str) -> None:
        return


class _Stage1Probe:
    def __init__(self, module: Any, work_dir: Path) -> None:
        self.module = module
        self.cfg = module.PipelineConfig()
        self.work_dir = work_dir
        self.process_dir = work_dir / "process"
        self.log = _FakeLogger()
        self.used_stacked = False
        self.used_light = False
        self.saved_stage = False
        self.source_file = None
        self.linear_intermediate_path = None
        self._stage1_input_mode = "unknown"
        self._task_resume_checkpoint_path = None
        self.cmd_calls: list[tuple[Any, ...]] = []

    def _clear_stage_reviews(self, _stage: int) -> None:
        return None

    def _find_fit_files(self) -> list[Path]:
        return self.module.StarunPostProcessor._find_fit_files(self)

    def _is_candidate_stacked(self, path: Path) -> bool:
        return self.module.StarunPostProcessor._is_candidate_stacked(self, path)

    def _prepare_process_dir(self) -> None:
        return self.module.StarunPostProcessor._prepare_process_dir(self)

    def _load_stacked_file(self, _stacked_files: list[Path]) -> None:
        self.used_stacked = True

    def _preprocess_light_frames(self, _light_files: list[Path]) -> None:
        self.used_light = True

    def _save_stage_output(self, _stem: str) -> bool:
        self.saved_stage = True
        return True

    def _record_stage(
        self,
        _name: str,
        _status: str,
        _duration: float = 0.0,
        _message: str = "",
    ) -> None:
        return

    def cmd_with_check(self, *args: Any, **_kwargs: Any) -> bool:
        self.cmd_calls.append(args)
        return True


class _Stage1GateProbe(_Stage1Probe):
    def __init__(self, module: Any, work_dir: Path) -> None:
        super().__init__(module, work_dir)
        self.preprocess_stats = {
            "total": 0,
            "registered": 0,
            "failed": 0,
            "fail_ratio": 0.0,
        }
        self.stage_save_ok = True
        self.stage_records: list[tuple[str, str, str]] = []

    def _preprocess_light_frames(self, _light_files: list[Path]):
        self.used_light = True
        return self.preprocess_stats

    def _save_stage_output(self, _stem: str) -> bool:
        self.saved_stage = True
        return self.stage_save_ok

    def _record_stage(
        self,
        name: str,
        status: str,
        _duration: float = 0.0,
        message: str = "",
    ) -> None:
        self.stage_records.append((name, status, message))

    def _record_skipped_stage(self, name: str, message: str) -> None:
        self.stage_records.append((name, "skipped", message))


class _LinearResumeProbe(_Stage1Probe):
    def __init__(self, module: Any, work_dir: Path) -> None:
        super().__init__(module, work_dir)
        self.stage_records: list[tuple[str, str, str]] = []

    def _record_stage(
        self,
        name: str,
        status: str,
        _duration: float = 0.0,
        message: str = "",
    ) -> None:
        self.stage_records.append((name, status, message))

    def _record_skipped_stage(self, name: str, message: str) -> None:
        self.stage_records.append((name, "skipped", message))


class _Stage1PreprocessProbe(_Stage1Probe):
    def _prepare_isolated_light_input(self, light_files: list[Path]):
        return self.module.StarunPostProcessor._prepare_isolated_light_input(
            self,
            light_files,
        )

    def _count_sequence_products(self, _seq_name: str) -> int:
        return 2


class PipelineStage1InputFilteringTests(unittest.TestCase):
    def test_is_candidate_stacked_rejects_sasp_exchange_files(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            probe = SimpleNamespace(cfg=pipeline_module.PipelineConfig(), work_dir=work_dir)

            for name in (
                "sasp_starmask_input.fit",
                "sasp_starless_input.fit",
                "manual_starless_export.fit",
                "manual_starmask_export.fits",
            ):
                path = work_dir / name
                path.write_bytes(b"")
                self.assertFalse(
                    pipeline_module.StarunPostProcessor._is_candidate_stacked(
                        probe,
                        path,
                    ),
                    msg=f"expected non-candidate: {name}",
                )

            candidate = work_dir / "IC2177_master.fit"
            candidate.write_bytes(b"")
            self.assertTrue(
                pipeline_module.StarunPostProcessor._is_candidate_stacked(
                    probe,
                    candidate,
                )
            )

    def test_stage1_prefers_light_when_only_sasp_candidates_exist(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "sasp_starmask_input.fit").write_bytes(b"")
            (work_dir / "sasp_starless_input.fit").write_bytes(b"")
            (work_dir / "Light_0001.fit").write_bytes(b"")
            (work_dir / "Light_0002.fit").write_bytes(b"")

            probe = _Stage1Probe(pipeline_module, work_dir)

            pipeline_module.StarunPostProcessor.stage1_preparation(probe)

            self.assertFalse(probe.used_stacked)
            self.assertTrue(probe.used_light)
            self.assertTrue(probe.saved_stage)

    def test_stage1_marks_degraded_when_register_fail_ratio_exceeds_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "Light_0001.fit").write_bytes(b"")
            probe = _Stage1GateProbe(pipeline_module, work_dir)
            probe.preprocess_stats = {
                "total": 100,
                "registered": 86,
                "failed": 14,
                "fail_ratio": 0.14,
            }

            pipeline_module.StarunPostProcessor.stage1_preparation(probe)

            _name, status, message = probe.stage_records[-1]
            self.assertEqual(status, "degraded")
            self.assertIn("registration failed=14/100", message)

    def test_stage1_keeps_ok_when_register_fail_ratio_within_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "Light_0001.fit").write_bytes(b"")
            probe = _Stage1GateProbe(pipeline_module, work_dir)
            probe.preprocess_stats = {
                "total": 80,
                "registered": 73,
                "failed": 7,
                "fail_ratio": 0.0875,
            }

            pipeline_module.StarunPostProcessor.stage1_preparation(probe)

            _name, status, message = probe.stage_records[-1]
            self.assertEqual(status, "ok")
            self.assertEqual(message, "")

    def test_light_preprocess_mirrors_stacked_working_before_load(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            light_files = [
                work_dir / "Light_0001.fit",
                work_dir / "Light_0002.fit",
            ]
            for path in light_files:
                path.write_bytes(b"")
            probe = _Stage1PreprocessProbe(pipeline_module, work_dir)
            probe.process_dir.mkdir()

            stats = pipeline_module.StarunPostProcessor._preprocess_light_frames(
                probe,
                light_files,
            )

            self.assertEqual(stats["registered"], 2)
            self.assertIn(("mirrorx_single", "working"), probe.cmd_calls)
            stack_index = next(
                idx for idx, call in enumerate(probe.cmd_calls)
                if call and call[0] == "stack"
            )
            mirror_index = probe.cmd_calls.index(("mirrorx_single", "working"))
            load_index = probe.cmd_calls.index(("load", "working"))
            self.assertLess(stack_index, mirror_index)
            self.assertLess(mirror_index, load_index)

    def test_prepare_linear_resume_input_uses_verified_stage5_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            linear_path = work_dir / "checkpoints" / "stage5" / "stage5_linear.fit"
            linear_path.parent.mkdir(parents=True)
            linear_path.write_bytes(b"linear-fit")
            probe = _LinearResumeProbe(pipeline_module, work_dir)
            probe._task_resume_checkpoint_path = linear_path

            pipeline_module.StarunPostProcessor._prepare_linear_resume_input(probe)

            self.assertEqual(probe.source_file, linear_path)
            self.assertEqual(probe.linear_intermediate_path, linear_path)
            self.assertEqual(probe._stage1_input_mode, "linear_resume")
            self.assertTrue((work_dir / "process" / "working.fit").is_file())
            self.assertIn(("load", "working"), probe.cmd_calls)
            self.assertEqual(len(probe.stage_records), 1)
            name, status, message = probe.stage_records[0]
            self.assertEqual(name, "阶段 1: 前期准备")
            self.assertEqual(status, "ok")
            self.assertIn("loaded verified Stage 5 checkpoint stage5_linear.fit", message)

    def test_prepare_stage2_corrected_resume_uses_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            stage2_path = work_dir / "checkpoints" / "stage2" / pipeline_module.STAGE2_CORRECTED_INPUT_NAME
            stage2_path.parent.mkdir(parents=True)
            stage2_path.write_bytes(b"stage2-fit")
            old_process = work_dir / "process"
            old_process.mkdir()
            stale_path = old_process / "stale.fit"
            stale_path.write_bytes(b"stale")
            probe = _LinearResumeProbe(pipeline_module, work_dir)
            probe._task_resume_checkpoint_path = stage2_path

            pipeline_module.StarunPostProcessor._prepare_stage2_corrected_resume_input(probe)

            self.assertEqual(probe.source_file, stage2_path)
            self.assertIsNone(probe.linear_intermediate_path)
            self.assertEqual(probe._stage1_input_mode, "stage2_corrected_resume")
            self.assertEqual((work_dir / "process" / "working.fit").read_bytes(), b"stage2-fit")
            self.assertEqual(
                (work_dir / "process" / pipeline_module.STAGE2_CORRECTED_INPUT_NAME).read_bytes(),
                b"stage2-fit",
            )
            self.assertFalse(stale_path.exists())
            self.assertIn(("load", "stage2_corrected"), probe.cmd_calls)
            self.assertEqual(len(probe.stage_records), 2)
            self.assertEqual(
                probe.stage_records[0],
                (
                    "阶段 1: 前期准备",
                    "skipped",
                    "skipped by stage2 corrected resume mode",
                ),
            )
            name, status, message = probe.stage_records[1]
            self.assertEqual(name, "阶段 2: 裁切")
            self.assertEqual(status, "ok")
            self.assertIn("continue from stage3", message)

    def test_resume_preparation_rejects_root_result_without_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "result_linear.fit").write_bytes(b"legacy-result")
            probe = _LinearResumeProbe(pipeline_module, work_dir)

            with self.assertRaisesRegex(
                pipeline_module.SirilError,
                "task-run manifest",
            ):
                pipeline_module.StarunPostProcessor._prepare_linear_resume_input(
                    probe
                )


if __name__ == "__main__":
    unittest.main()
