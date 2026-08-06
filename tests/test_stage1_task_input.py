#!/usr/bin/env python3
"""Regression tests for signed, source-read-only Stage 1 task intake."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


class _SirilError(RuntimeError):
    pass


if "sirilpy.exceptions" not in sys.modules:
    sirilpy_module = types.ModuleType("sirilpy")
    exceptions_module = types.ModuleType("sirilpy.exceptions")
    exceptions_module.SirilError = _SirilError
    sirilpy_module.exceptions = exceptions_module
    sys.modules["sirilpy"] = sirilpy_module
    sys.modules["sirilpy.exceptions"] = exceptions_module

import task_workspace  # noqa: E402
from stages import stage1_preparation  # noqa: E402


class _Log:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warn(self, message: str) -> None:
        self.messages.append(message)


class Stage1TaskInputTests(unittest.TestCase):
    def _run(self, root: Path):
        source = root / "source" / "NGC6910.xisf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"XISF0100-linear-master")
        source_record = task_workspace.build_source_record(
            source_kind="master_file",
            selected_path=source,
            files=(source,),
        )
        workspace = task_workspace.ensure_task_workspace(
            source_record=source_record,
            selected_path=source,
        )
        run = task_workspace.begin_task_run(
            workspace=workspace,
            source_record=source_record,
            run_id="run-1",
        )
        return source, run

    def test_signed_run_manifest_resolves_exact_external_master(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source, run = self._run(Path(td))
            pipeline = SimpleNamespace(work_dir=run.root, log=_Log())

            with patch.dict(
                os.environ,
                {stage1_preparation.TASK_RUN_MANIFEST_ENV: str(run.manifest_path)},
                clear=False,
            ):
                kind, files = stage1_preparation._load_task_run_source(pipeline)

        self.assertEqual(kind, "master_file")
        self.assertEqual(files, [source.resolve()])
        self.assertTrue(any("verified" in item for item in pipeline.log.messages))

    def test_source_change_after_freeze_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source, run = self._run(Path(td))
            source.write_bytes(b"changed-after-freeze")
            pipeline = SimpleNamespace(work_dir=run.root, log=_Log())

            with patch.dict(
                os.environ,
                {stage1_preparation.TASK_RUN_MANIFEST_ENV: str(run.manifest_path)},
                clear=False,
            ):
                with self.assertRaisesRegex(stage1_preparation.SirilError, "已变化"):
                    stage1_preparation._load_task_run_source(pipeline)

    def test_xisf_import_saves_working_inside_task_process_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source" / "NGC6910.xisf"
            process_dir = root / "task" / "process"
            source.parent.mkdir(parents=True)
            process_dir.mkdir(parents=True)
            source.write_bytes(b"XISF0100-linear-master")
            commands: list[tuple[str, ...]] = []
            pipeline = SimpleNamespace(
                process_dir=process_dir,
                log=_Log(),
                cmd_with_check=lambda *args: commands.append(tuple(args)),
            )

            stage1_preparation._load_explicit_master(
                pipeline,
                source,
                "master_file",
            )

        self.assertEqual(pipeline.source_file, source)
        self.assertEqual(pipeline._stage1_input_mode, "explicit_master")
        self.assertIn(("cd", f'"{source.parent}"'), commands)
        self.assertIn(("load", f'"{source.name}"'), commands)
        self.assertIn(("cd", f'"{process_dir}"'), commands)
        self.assertIn(("save", "working"), commands)


if __name__ == "__main__":
    unittest.main()
