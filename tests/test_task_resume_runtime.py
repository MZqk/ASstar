#!/usr/bin/env python3
"""Regression tests for pipeline-side task checkpoint provenance."""
from __future__ import annotations

import json
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
import run_manifest  # noqa: E402
import task_plan  # noqa: E402
from stage_contracts import stage_contract  # noqa: E402
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
    @staticmethod
    def _semantic_context() -> dict[str, object]:
        mapping = {
            "schema": "starun.narrowband-channel-mapping.v1",
            "mapping": "osc_hoo_rgb",
            "ha_channel": "R",
            "oiii_channels": ["G", "B"],
            "confidence": 0.97,
            "evidence": "verified_device_profile",
            "evidence_detail": {
                "device_profile_id": "seestar_s30_pro_imx585",
            },
        }
        return {
            "schema": "starun.resume-semantics.v1",
            "checkpoint_stage": 5,
            "channel_semantics": "narrowband_composite",
            "channel_profile": {
                "kind": "narrowband_composite",
                "confidence": 0.98,
                "narrowband_mapping": mapping,
            },
            "narrowband_channel_mapping": mapping,
            "target_profile": {
                "target_type": "emission_nebula_widefield",
                "target_confidence": 0.93,
            },
            "pipeline_policy": {
                "policy_name": "emission_nebula_widefield",
            },
            "color_calibration_report": {
                "status": "success",
                "method": "SPCC_NARROWBAND",
                "physical_color": {
                    "accepted": True,
                    "method": "SPCC_NARROWBAND",
                },
            },
            "stage5_star_reference_report": {
                "schema": "starun.stage5-star-reference.v1",
                "status": "ready",
                "stars": [
                    {
                        "index": 7,
                        "x": 11.0,
                        "y": 13.0,
                        "fwhm_geometry": 9.0,
                        "saturated": True,
                    }
                ],
            },
            "upstream_review": {
                "stage2_view_review_required": False,
                "background_review_required": False,
                "color_review_required": False,
            },
        }

    def _resume_run(
        self,
        root: Path,
        *,
        stage_number: int = 5,
        with_semantics: bool = True,
        semantic_context: dict[str, object] | None = None,
    ):
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
        contract = stage_contract(stage_number)
        checkpoint = process_dir / contract.primary_artifact
        checkpoint.write_bytes(f"verified-stage{stage_number}".encode("ascii"))
        publish_formal_checkpoint(
            run_manifest_path=first_run.manifest_path,
            stage_number=stage_number,
            artifact_path=checkpoint,
            semantic_context=(
                semantic_context
                if semantic_context is not None
                else self._semantic_context()
                if with_semantics and stage_number == 5
                else None
            ),
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

    def test_pipeline_accepts_matching_task_run_stage1_and_stage2_provenance(self) -> None:
        modes = {
            1: processor_runtime.INPUT_MODE_STAGE1_PREPARED_RESUME,
            2: processor_runtime.INPUT_MODE_STAGE2_CORRECTED_RESUME,
        }
        with tempfile.TemporaryDirectory() as td:
            for stage_number, input_mode in modes.items():
                with self.subTest(stage=stage_number):
                    _workspace, run = self._resume_run(
                        Path(td) / f"stage{stage_number}",
                        stage_number=stage_number,
                    )
                    runtime = processor_runtime.ProcessorRuntimeMixin()
                    runtime.work_dir = run.root
                    runtime.input_mode = input_mode
                    runtime.log = _Log()

                    with patch.dict(
                        os.environ,
                        {
                            processor_runtime.ENV_TASK_RUN_MANIFEST_KEY: str(
                                run.manifest_path
                            )
                        },
                        clear=False,
                    ):
                        result = runtime._load_trusted_input_provenance_for_resume()

                    self.assertTrue(result["verified"])
                    self.assertEqual(result["checkpoint"], f"stage{stage_number}")
                    self.assertEqual(
                        runtime._task_resume_checkpoint_path.name,
                        stage_contract(stage_number).primary_artifact,
                    )

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
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    runtime._load_trusted_input_provenance_for_resume()

    def test_stage5_resume_restores_signed_color_and_channel_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _workspace, run = self._resume_run(
                Path(td),
                with_semantics=True,
            )
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = run.root
            runtime.input_mode = processor_runtime.INPUT_MODE_LINEAR_RESUME
            runtime.log = _Log()
            runtime._background_review_required = False
            runtime._stage4_color_review_required = False
            runtime.color_calibration_report = {}
            runtime.channel_profile = {}
            runtime.target_profile = {}
            runtime.pipeline_policy = {}

            with patch.dict(
                os.environ,
                {processor_runtime.ENV_TASK_RUN_MANIFEST_KEY: str(run.manifest_path)},
                clear=False,
            ):
                provenance = runtime._load_trusted_input_provenance_for_resume()
                restored = runtime._apply_trusted_resume_semantics()

        self.assertTrue(provenance["verified"])
        self.assertEqual(provenance["semantic_context_status"], "verified")
        self.assertTrue(restored)
        self.assertEqual(runtime._channel_semantics, "narrowband_composite")
        self.assertEqual(
            runtime.color_calibration_report["method"],
            "SPCC_NARROWBAND",
        )
        self.assertTrue(
            runtime.color_calibration_report["physical_color"]["accepted"]
        )
        self.assertEqual(
            runtime.narrowband_channel_mapping["evidence"],
            "verified_device_profile",
        )
        self.assertFalse(runtime._stage4_color_review_required)
        self.assertFalse(runtime._stage2_view_review_required)
        self.assertEqual(
            runtime._stage5_star_reference_report["stars"][0]["index"],
            7,
        )

    def test_stage5_resume_restores_stage2_review_without_stage3_pollution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            context = self._semantic_context()
            context["upstream_review"]["stage2_view_review_required"] = True
            _workspace, run = self._resume_run(
                Path(td),
                semantic_context=context,
            )
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = run.root
            runtime.input_mode = processor_runtime.INPUT_MODE_LINEAR_RESUME
            runtime.log = _Log()
            runtime._stage2_view_review_required = False
            runtime._background_review_required = False
            runtime._stage4_color_review_required = False
            runtime.color_calibration_report = {}

            with patch.dict(
                os.environ,
                {
                    processor_runtime.ENV_TASK_RUN_MANIFEST_KEY: str(
                        run.manifest_path
                    )
                },
                clear=False,
            ):
                runtime._load_trusted_input_provenance_for_resume()
                restored = runtime._apply_trusted_resume_semantics()

        self.assertTrue(restored)
        self.assertTrue(runtime._stage2_view_review_required)
        self.assertFalse(runtime._background_review_required)

    def test_stage5_checkpoint_without_semantics_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "语义契约"):
                self._resume_run(Path(td), with_semantics=False)

    def test_stage5_checkpoint_without_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            context = self._semantic_context()
            context.pop("narrowband_channel_mapping", None)
            channel_profile = dict(context["channel_profile"])
            channel_profile.pop("narrowband_mapping", None)
            context["channel_profile"] = channel_profile
            with self.assertRaisesRegex(RuntimeError, "通道映射契约"):
                self._resume_run(
                    Path(td),
                    semantic_context=context,
                )

    def test_pipeline_result_persists_frozen_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = Path(td)
            runtime.process_dir = None
            runtime.log = _Log()
            runtime.results = []
            runtime.input_profile = {"state": "linear"}
            runtime._channel_semantics = "narrowband_composite"
            runtime.target_profile = {}
            runtime.color_calibration_report = {}
            runtime._stage9_stars_required = True
            runtime._stage9_stars_applied = True
            runtime._stage9_output_contains_stars = True
            runtime._stage9_remix_formally_accepted = False
            runtime._stage9_review_candidate_selected = True
            runtime._stage9_psf_review_required = True
            runtime.narrowband_channel_mapping = dict(
                self._semantic_context()["narrowband_channel_mapping"]
            )
            plan = task_plan.build_processing_plan(
                run_id="mapping-run",
                generated_at="2026-08-14T00:00:00Z",
                input_record={"kind": "master_file"},
                input_state="linear",
                input_trust="recognized",
            )
            run_manifest.atomic_write_json(
                runtime.work_dir / "processing-plan.json",
                plan,
            )
            runtime._run_id = plan["run_id"]
            runtime._processing_plan_hash = plan["plan_hash"]

            written = runtime._write_pipeline_result_manifest()
            payload = json.loads(
                (runtime.work_dir / "pipeline-result.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(written)
        self.assertEqual(
            payload["narrowband_channel_mapping"],
            runtime.narrowband_channel_mapping,
        )
        self.assertFalse(
            payload["star_separation"]["remix_formally_accepted"]
        )
        self.assertTrue(
            payload["star_separation"]["review_candidate_selected"]
        )
        self.assertTrue(
            payload["review_requirements"]["stage9_review_candidate_selected"]
        )


if __name__ == "__main__":
    unittest.main()
