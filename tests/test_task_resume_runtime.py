#!/usr/bin/env python3
"""Regression tests for pipeline-side task checkpoint provenance."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits


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
import scene_support  # noqa: E402
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

    @staticmethod
    def _footprint_evidence() -> dict[str, object]:
        decoded = bytes((100, 100, 0, 0))
        grid = {
            "rows": 2,
            "columns": 2,
            "encoding": "rle-u8-row-major-v1",
            "runs": [[100, 2], [0, 2]],
            "sha256": hashlib.sha256(decoded).hexdigest(),
        }
        return {
            "schema": "starun.stage2-stacked-footprint-evidence.v1",
            "status": "available",
            "source_mode": "stacked_master_inference",
            "observer_only": True,
            "captured_before_crop": True,
            "source_artifact": "stage1_prepared.fit",
            "source_sha256": "b" * 64,
            "input_shape": {"channels": 3, "height": 1080, "width": 1920},
            "layers": {
                "fill_support": {
                    "status": "available",
                    "reason": "full_frame_is_valid",
                    "grid": dict(grid),
                },
                "relative_coverage": {
                    "status": "available",
                    "reason": "no_significant_edge_connected_coverage_anomaly",
                    "grid": dict(grid),
                },
            },
            "limitations": [
                "not_per_frame_registration_footprint",
                "not_crop_authority",
            ],
        }

    @staticmethod
    def _stage2_semantic_context() -> dict[str, object]:
        return {
            "schema": "starun.resume-semantics.v2",
            "checkpoint_stage": 2,
            "review_requirements": [],
            "stage2_crop": {
                "original_dimensions": {"width": 1920, "height": 1080},
                "final_dimensions": {"width": 1600, "height": 960},
                "cumulative_crop": {
                    "left": 160,
                    "top": 60,
                    "right": 160,
                    "bottom": 60,
                },
                "field_rotation_passes": 1,
                "final_residual_detection": {
                    "accepted": False,
                    "reason": "no_significant_edge_connected_coverage_anomaly",
                },
                "stacked_master_footprint": (
                    TaskResumeRuntimeTests._footprint_evidence()
                ),
            },
        }

    def _resume_run(
        self,
        root: Path,
        *,
        stage_number: int = 5,
        with_semantics: bool = True,
        semantic_context: dict[str, object] | None = None,
        with_scene_support: bool = False,
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
        selected_context = (
            semantic_context
            if semantic_context is not None
            else self._semantic_context()
            if with_semantics and stage_number == 5
            else self._stage2_semantic_context()
            if with_semantics and stage_number == 2
            else None
        )
        auxiliary_artifacts = None
        if with_scene_support:
            image = np.full((3, 24, 32), 0.1, dtype=np.float32)
            scene_source = process_dir / "stage3_bg_input.fit"
            scene_source.write_bytes(b"stage3-bg-input")
            manifest = scene_support.build_scene_support(
                image,
                process_dir,
                source_path=scene_source,
                sep_module=None,
            )
            if isinstance(selected_context, dict):
                selected_context["stage3_scene_support"] = (
                    scene_support.scene_support_summary(manifest)
                )
            auxiliary_artifacts = {
                name: process_dir / name
                for name in (
                    scene_support.SCENE_SUPPORT_JSON,
                    scene_support.SCENE_SUPPORT_ARRAYS,
                )
            }
        publish_formal_checkpoint(
            run_manifest_path=first_run.manifest_path,
            stage_number=stage_number,
            artifact_path=checkpoint,
            semantic_context=selected_context,
            auxiliary_artifacts=auxiliary_artifacts,
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

    def test_stage2_resume_restores_crop_and_review_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            context = self._stage2_semantic_context()
            context["review_requirements"] = [
                {
                    "stage": 2,
                    "code": "field_rotation_residual_review",
                    "details": {"retained_area_ratio": 0.70},
                }
            ]
            _workspace, run = self._resume_run(
                Path(td),
                stage_number=2,
                semantic_context=context,
            )
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = run.root
            runtime.input_mode = processor_runtime.INPUT_MODE_STAGE2_CORRECTED_RESUME
            runtime.log = _Log()

            with patch.dict(
                os.environ,
                {processor_runtime.ENV_TASK_RUN_MANIFEST_KEY: str(run.manifest_path)},
                clear=False,
            ):
                provenance = runtime._load_trusted_input_provenance_for_resume()
                restored = runtime._apply_trusted_resume_semantics()

        self.assertTrue(provenance["verified"])
        self.assertTrue(restored)
        self.assertEqual(runtime.stage2_crop_report["field_rotation_passes"], 1)
        self.assertEqual(
            runtime.stage2_crop_report["stacked_master_footprint"],
            context["stage2_crop"]["stacked_master_footprint"],
        )
        self.assertEqual(
            runtime._stage_review_reasons(2),
            ["field_rotation_residual_review"],
        )
        self.assertEqual(runtime._stage_review_reasons(3), [])

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
        self.assertEqual(
            runtime._stage3_scene_support["manifest"]["reason_code"],
            "legacy_checkpoint_without_scene_support",
        )

    def test_stage5_resume_restores_scene_support_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _workspace, run = self._resume_run(
                Path(td),
                with_scene_support=True,
            )
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = run.root
            runtime.process_dir = run.root / "process"
            runtime.process_dir.mkdir()
            runtime.input_mode = processor_runtime.INPUT_MODE_LINEAR_RESUME
            runtime.log = _Log()
            runtime.color_calibration_report = {}
            runtime.channel_profile = {}
            runtime.target_profile = {}
            runtime.pipeline_policy = {}

            with patch.dict(
                os.environ,
                {processor_runtime.ENV_TASK_RUN_MANIFEST_KEY: str(run.manifest_path)},
                clear=False,
            ):
                runtime._load_trusted_input_provenance_for_resume()
                restored = runtime._apply_trusted_resume_semantics()

            self.assertTrue(restored)
            self.assertIn(runtime._stage3_scene_support["status"], {"available", "partial"})
            self.assertTrue(
                (runtime.process_dir / scene_support.SCENE_SUPPORT_JSON).is_file()
            )
            self.assertTrue(
                (runtime.process_dir / scene_support.SCENE_SUPPORT_ARRAYS).is_file()
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
            runtime._require_review(9, "stage9_review_candidate_selected")
            runtime._require_review(
                9,
                "stage9_psf_subgroup_evidence_insufficient",
            )

            written = runtime._write_pipeline_result_manifest()
            payload = json.loads(
                (runtime.work_dir / "pipeline-result.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(written)
        self.assertEqual(payload["schema"], "starun.pipeline-result.v2")
        self.assertTrue(payload["review_required"])
        self.assertFalse(payload["had_fatal_errors"])
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
        self.assertFalse(
            payload["delivery_gates"]["formal_delivery_accepted"]
        )
        self.assertFalse(payload["delivery_gates"]["review"]["accepted"])
        self.assertIn(
            {"stage": 9, "code": "stage9_review_candidate_selected", "details": {}},
            payload["review_requirements"],
        )

    def test_pipeline_result_requires_science_presentation_and_artifact_gates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = Path(td)
            runtime.process_dir = runtime.work_dir / "process"
            runtime.process_dir.mkdir()
            runtime.log = _Log()
            runtime.results = []
            runtime.input_profile = {"state": "linear"}
            runtime._channel_semantics = "broadband_rgb_osc"
            runtime.target_profile = {}
            runtime.color_calibration_report = {}
            runtime.narrowband_channel_mapping = {}
            runtime._scientific_quality_accepted = True
            runtime._presentation_quality_accepted = True
            runtime._presentation_quality_report = {"status": "ok"}
            runtime._final_output_review_only = False
            runtime._final_output_basenames = ("result_processed",)
            formal = runtime.work_dir / "result_processed.fit"
            formal_pixels = np.linspace(
                0.01,
                0.25,
                num=3 * 8 * 8,
                dtype=np.float32,
            ).reshape(3, 8, 8)
            fits.PrimaryHDU(formal_pixels).writeto(formal)
            fits.PrimaryHDU(formal_pixels).writeto(
                runtime.process_dir / "stage10_final.fit"
            )
            plan = task_plan.build_processing_plan(
                run_id="delivery-gates-run",
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
        gates = payload["delivery_gates"]
        self.assertEqual(gates["schema"], "starun.final-delivery-gates.v1")
        self.assertTrue(gates["scientific"]["accepted"])
        self.assertTrue(gates["presentation"]["accepted"])
        self.assertTrue(gates["artifacts"]["accepted"])
        self.assertTrue(gates["review"]["accepted"])
        self.assertTrue(gates["formal_delivery_accepted"])
        self.assertTrue(payload["delivery_eligible"])
        self.assertEqual(gates["artifacts"]["formal_count"], 1)

    def test_failed_run_cannot_become_delivery_eligible_from_four_green_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = processor_runtime.ProcessorRuntimeMixin()
            runtime.work_dir = Path(temporary)
            runtime.process_dir = None
            runtime.log = _Log()
            runtime.results = []
            runtime.input_profile = {"state": "linear"}
            runtime._channel_semantics = "broadband_rgb_osc"
            runtime.target_profile = {}
            runtime.color_calibration_report = {}
            runtime.narrowband_channel_mapping = {}
            runtime._scientific_quality_accepted = True
            runtime._presentation_quality_accepted = True
            runtime._presentation_quality_report = {"status": "ok"}
            runtime._final_output_review_only = False
            runtime._final_output_basenames = ("result_final",)
            (runtime.work_dir / "result_final.fit").write_bytes(b"formal")
            plan = task_plan.build_processing_plan(
                run_id="run-failed-delivery",
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

            self.assertTrue(
                runtime._write_pipeline_result_manifest(
                    failure_reason="fatal"
                )
            )
            payload = json.loads(
                (runtime.work_dir / "pipeline-result.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(payload["status"], "failed")
        self.assertFalse(
            payload["delivery_gates"]["formal_delivery_accepted"]
        )
        self.assertFalse(payload["delivery_eligible"])


if __name__ == "__main__":
    unittest.main()
