#!/usr/bin/env python3
"""Regression tests for pipeline-side task checkpoint provenance."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


class _SirilError(RuntimeError):
    pass


exceptions_module = sys.modules.get("sirilpy.exceptions")
if exceptions_module is None:
    sirilpy_module = types.ModuleType("sirilpy")
    exceptions_module = types.ModuleType("sirilpy.exceptions")
    sirilpy_module.exceptions = exceptions_module
    sys.modules["sirilpy"] = sirilpy_module
    sys.modules["sirilpy.exceptions"] = exceptions_module
for exception_name in ("CommandError", "DataError", "SirilError"):
    if not hasattr(exceptions_module, exception_name):
        setattr(exceptions_module, exception_name, _SirilError)

import processor_runtime  # noqa: E402
import task_plan  # noqa: E402
from task_workspace import (  # noqa: E402
    begin_task_run,
    build_source_record,
    ensure_task_workspace,
    inspect_task_workspace,
    publish_formal_checkpoint,
)


class _Log:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, value: str) -> None:
        self.messages.append(value)

    def warn(self, value: str) -> None:
        self.messages.append(value)


class TaskResumeRuntimeTests(unittest.TestCase):
    def _resume_run(self, root: Path):
        source = root / "source" / "master.fit"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"linear-master")
        source_record = build_source_record(
            source_kind="master_file",
            selected_path=source,
            files=(source,),
        )
        workspace = ensure_task_workspace(
            source_record=source_record,
            selected_path=source,
        )
        fingerprints = task_plan.build_resume_fingerprints(
            input_fingerprint=source_record["fingerprint"],
            stage_config={stage: {"value": stage} for stage in range(1, 6)},
        )
        first_run = begin_task_run(
            workspace=workspace,
            source_record=source_record,
            run_id="first-run",
            checkpoint_fingerprints=fingerprints,
        )
        process_dir = first_run.root / "process"
        process_dir.mkdir()
        stage5 = process_dir / "stage5_linear.fit"
        stage5.write_bytes(b"verified-stage5")
        publish_formal_checkpoint(
            run_manifest_path=first_run.manifest_path,
            stage_number=5,
            artifact_path=stage5,
        )
        inspection = inspect_task_workspace(
            workspace.root,
            current_resume_fingerprints=fingerprints,
        )
        second_run = begin_task_run(
            workspace=workspace,
            source_record=source_record,
            run_id="second-run",
            resume_record=inspection["resume_record"],
            checkpoint_fingerprints=fingerprints,
        )
        return workspace, second_run

    def test_pipeline_accepts_matching_task_run_stage5_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _workspace, run = self._resume_run(Path(td))
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = run.root
            runtime.input_mode = processor_runtime.INPUT_MODE_LINEAR_RESUME
            runtime.log = _Log()

            with patch.dict(
                os.environ,
                {processor_runtime.ENV_TASK_RUN_MANIFEST_KEY: str(run.manifest_path)},
                clear=False,
            ):
                result = runtime._load_trusted_input_provenance_for_resume()

        self.assertTrue(result["verified"])
        self.assertEqual(result["state"], "linear")
        self.assertEqual(runtime._task_resume_checkpoint_path.name, "stage5_linear.fit")

    def test_pipeline_rejects_checkpoint_changed_after_run_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace, run = self._resume_run(Path(td))
            (workspace.checkpoints_dir / "stage5_linear.fit").write_bytes(b"changed")
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = run.root
            runtime.input_mode = processor_runtime.INPUT_MODE_LINEAR_RESUME
            runtime.log = _Log()

            with patch.dict(
                os.environ,
                {processor_runtime.ENV_TASK_RUN_MANIFEST_KEY: str(run.manifest_path)},
                clear=False,
            ):
                result = runtime._load_trusted_input_provenance_for_resume()

        self.assertFalse(result["verified"])
        self.assertIn("SHA-256", result["detail"])


if __name__ == "__main__":
    unittest.main()
