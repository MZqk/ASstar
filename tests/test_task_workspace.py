#!/usr/bin/env python3
"""Regression tests for source-read-only tasks and formal checkpoints."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_manifest  # noqa: E402
import task_plan  # noqa: E402
from input_discovery import InputKind, discover_input  # noqa: E402
from stage_contracts import FORMAL_RESUME_STAGES, stage_contract  # noqa: E402
from task_workspace import (  # noqa: E402
    CHECKPOINT_MANIFEST_REL,
    TASK_CONTAINER_NAME,
    WorkspaceError,
    apply_task_retention,
    begin_task_run,
    build_checkpoint_manifest,
    build_source_record,
    ensure_task_workspace,
    inspect_task_workspace,
    latest_result_directory,
    latest_result_files,
    publish_formal_checkpoint,
    publish_latest_result_index,
)


class TaskWorkspaceTests(unittest.TestCase):
    def _workspace(self, root: Path):
        source = root / "source" / "NGC6910.xisf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"XISF0100-linear-master")
        source_record = build_source_record(
            source_kind="master_file",
            selected_path=source,
            files=(source,),
        )
        workspace = ensure_task_workspace(
            source_record=source_record,
            selected_path=source,
            created_at="2026-08-04T00:00:00Z",
        )
        return source, source_record, workspace

    @staticmethod
    def _fingerprints(
        *,
        stage4: str = "physical",
        stage2: str = "auto",
    ) -> dict[str, dict[str, object]]:
        return task_plan.build_resume_fingerprints(
            input_fingerprint="source-fingerprint",
            stage_config={
                1: {"import": "xisf_to_fits"},
                2: {"crop": stage2},
                3: {"background": "safe"},
                4: {"color": stage4},
                5: {"denoise": "conservative"},
            },
        )

    @staticmethod
    def _semantic_context() -> dict[str, object]:
        mapping = {
            "schema": "starun.narrowband-channel-mapping.v1",
            "mapping": "unknown",
            "confidence": 0.0,
            "evidence": "not_narrowband",
        }
        return {
            "schema": "starun.resume-semantics.v1",
            "checkpoint_stage": 5,
            "channel_semantics": "broadband",
            "channel_profile": {"kind": "broadband"},
            "narrowband_channel_mapping": mapping,
            "target_profile": {},
            "pipeline_policy": {},
            "color_calibration_report": {},
            "upstream_review": {},
        }

    @staticmethod
    def _stage2_semantic_context() -> dict[str, object]:
        return {
            "schema": "starun.resume-semantics.v2",
            "checkpoint_stage": 2,
            "review_requirements": [],
            "stage2_crop": {
                "original_dimensions": {"width": 1920, "height": 1080},
                "final_dimensions": {"width": 1920, "height": 1080},
                "cumulative_crop": {
                    "left": 0,
                    "top": 0,
                    "right": 0,
                    "bottom": 0,
                },
                "field_rotation_passes": 0,
                "final_residual_detection": {
                    "accepted": False,
                    "reason": "not_run",
                },
            },
        }

    def _semantic_context_for_stage(self, stage_number: int):
        if stage_number == 2:
            return self._stage2_semantic_context()
        if stage_number == 5:
            return self._semantic_context()
        return None

    def _write_checkpoints(self, workspace, fingerprints) -> dict[str, object]:
        records = {}
        for stage_number in FORMAL_RESUME_STAGES:
            contract = stage_contract(stage_number)
            path = workspace.checkpoints_dir / contract.primary_artifact
            path.write_bytes(f"checkpoint-{stage_number}".encode("ascii"))
            key = f"stage{stage_number}"
            records[key] = {
                "stage": stage_number,
                "artifact": contract.primary_artifact,
                "path": path.relative_to(workspace.root).as_posix(),
                "sha256": run_manifest.sha256_file(path),
                "state": "linear",
                "run_manifest_hash": "run-manifest-hash",
                "config_fingerprint": fingerprints[key]["fingerprint"],
                "semantic_context": self._semantic_context_for_stage(stage_number),
                "semantic_context_status": (
                    "verified" if stage_number in {2, 5} else "not_applicable"
                ),
            }
        manifest = build_checkpoint_manifest(
            task_id=workspace.task_id,
            source_fingerprint=workspace.source_fingerprint,
            checkpoints=records,
            generated_at="2026-08-04T01:00:00Z",
        )
        run_manifest.atomic_write_json(
            workspace.root / CHECKPOINT_MANIFEST_REL,
            manifest,
        )
        return records

    def test_task_is_created_in_sibling_container_and_reused_by_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, source_record, workspace = self._workspace(root)
            reused = ensure_task_workspace(
                source_record=source_record,
                selected_path=source,
            )

            self.assertEqual(
                workspace.root.parent,
                (source.parent / TASK_CONTAINER_NAME).resolve(),
            )
            self.assertTrue(workspace.manifest_path.is_file())
            self.assertEqual(source.read_bytes(), b"XISF0100-linear-master")
            self.assertFalse(any(workspace.root.rglob("*.xisf")))
            self.assertFalse(workspace.reused)
            self.assertTrue(reused.reused)
            self.assertEqual(reused.root, workspace.root)

    def test_light_source_manifest_holds_references_not_copies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_root = root / "capture" / "Light"
            light1 = source_root / "Light_001.fit"
            light2 = source_root / "Light_002.fit"
            source_root.mkdir(parents=True)
            light1.write_bytes(b"one")
            light2.write_bytes(b"two")
            record = build_source_record(
                source_kind="light_directory",
                selected_path=source_root,
                files=(light1, light2),
                group={
                    "target": "NGC 6910",
                    "filter": "Seestar LP",
                    "camera": "Seestar S50",
                    "geometry": "1080x1920@1x1",
                },
            )
            workspace = ensure_task_workspace(
                source_record=record,
                selected_path=source_root,
            )

            manifest = run_manifest.load_json(workspace.manifest_path)

        self.assertIsNotNone(manifest)
        self.assertTrue(manifest["source"]["read_only"])
        self.assertEqual(manifest["source"]["file_count"], 2)
        self.assertEqual(
            {Path(item["path"]).name for item in manifest["source"]["files"]},
            {"Light_001.fit", "Light_002.fit"},
        )

    def test_task_run_freezes_same_read_only_source_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source, source_record, workspace = self._workspace(Path(td))

            run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="20260804T010203-run1",
                generated_at="2026-08-04T01:02:03Z",
            )
            payload = run_manifest.load_json(run.manifest_path)

        self.assertEqual(run.root.parent, workspace.runs_dir)
        self.assertEqual(payload["task_id"], workspace.task_id)
        self.assertEqual(payload["source"]["files"][0]["path"], str(source.resolve()))
        self.assertTrue(payload["source"]["read_only"])

    def test_accepted_stages_publish_durable_formal_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            fingerprints = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={stage: {"value": stage} for stage in range(1, 6)},
            )
            run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="publish-run",
                checkpoint_fingerprints=fingerprints,
            )
            process_dir = run.root / "process"
            process_dir.mkdir()
            for stage_number in FORMAL_RESUME_STAGES:
                contract = stage_contract(stage_number)
                artifact = process_dir / contract.primary_artifact
                artifact.write_bytes(f"accepted-{stage_number}".encode("ascii"))
                publish_formal_checkpoint(
                    run_manifest_path=run.manifest_path,
                    stage_number=stage_number,
                    artifact_path=artifact,
                    semantic_context=self._semantic_context_for_stage(stage_number),
                )

            inspection = inspect_task_workspace(
                workspace.root,
                current_resume_fingerprints=fingerprints,
            )

        self.assertTrue(inspection["verified"])
        self.assertEqual(inspection["resume_after_stage"], 5)

    def test_earlier_checkpoint_prunes_incompatible_later_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            baseline = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={stage: {"value": stage} for stage in range(1, 6)},
            )
            first_run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="first-run",
                checkpoint_fingerprints=baseline,
            )
            first_process = first_run.root / "process"
            first_process.mkdir()
            for stage_number in FORMAL_RESUME_STAGES:
                contract = stage_contract(stage_number)
                artifact = first_process / contract.primary_artifact
                artifact.write_bytes(f"first-{stage_number}".encode("ascii"))
                publish_formal_checkpoint(
                    run_manifest_path=first_run.manifest_path,
                    stage_number=stage_number,
                    artifact_path=artifact,
                    semantic_context=self._semantic_context_for_stage(stage_number),
                )

            changed = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={
                    1: {"value": 1},
                    2: {"value": 2},
                    3: {"value": 3},
                    4: {"value": "changed"},
                    5: {"value": 5},
                },
            )
            second_run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="second-run",
                checkpoint_fingerprints=changed,
            )
            second_process = second_run.root / "process"
            second_process.mkdir()
            stage2 = second_process / "stage2_corrected.fit"
            stage2.write_bytes(b"second-stage2")
            publish_formal_checkpoint(
                run_manifest_path=second_run.manifest_path,
                stage_number=2,
                artifact_path=stage2,
                semantic_context=self._stage2_semantic_context(),
            )
            manifest = run_manifest.load_json(
                workspace.root / CHECKPOINT_MANIFEST_REL
            )

        self.assertIn("stage2", manifest["checkpoints"])
        self.assertNotIn("stage5", manifest["checkpoints"])

    def test_latest_result_index_references_only_hash_verified_run_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="result-run",
            )
            preview = run.root / "result_processed.png"
            preview.write_bytes(b"preview")
            stale = run.root / "result_final.fit"
            stale.write_bytes(b"stale")
            result = {
                "schema": "starun.pipeline-result.v2",
                "run_id": run.run_id,
                "status": "success",
                "review_requirements": [],
                "actual_steps": [],
                "outputs": {
                    preview.name: run_manifest.file_record(
                        preview,
                        base_dir=run.root,
                    ),
                    stale.name: {
                        **run_manifest.file_record(stale, base_dir=run.root),
                        "sha256": "wrong",
                    },
                },
            }
            plan = task_plan.build_processing_plan(
                run_id=run.run_id,
                generated_at="2026-08-04T01:00:00Z",
                input_record={"fingerprint": source_record["fingerprint"]},
                input_state="linear",
                input_trust="recognized",
            )
            run_manifest.atomic_write_json(run.root / "processing-plan.json", plan)
            result["plan_hash"] = plan["plan_hash"]
            result["manifest_hash"] = run_manifest.canonical_payload_hash(result)
            run_manifest.atomic_write_json(run.root / "pipeline-result.json", result)

            latest = publish_latest_result_index(
                run_manifest_path=run.manifest_path
            )
            files = latest_result_files(workspace.root)
            result_directory = latest_result_directory(workspace.root)

        self.assertEqual(latest["run_id"], "result-run")
        self.assertEqual([path.name for path in files], ["result_processed.png"])
        self.assertEqual(result_directory, run.root)

    def test_failed_manifest_update_restores_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            fingerprints = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={stage: {"value": stage} for stage in range(1, 6)},
            )
            first_run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="checkpoint-first",
                checkpoint_fingerprints=fingerprints,
            )
            first_process = first_run.root / "process"
            first_process.mkdir()
            first_artifact = first_process / "stage2_corrected.fit"
            first_artifact.write_bytes(b"trusted-old-checkpoint")
            publish_formal_checkpoint(
                run_manifest_path=first_run.manifest_path,
                stage_number=2,
                artifact_path=first_artifact,
                semantic_context=self._stage2_semantic_context(),
            )
            manifest_path = workspace.root / CHECKPOINT_MANIFEST_REL
            previous_manifest = manifest_path.read_bytes()

            second_run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="checkpoint-second",
                checkpoint_fingerprints=fingerprints,
            )
            second_process = second_run.root / "process"
            second_process.mkdir()
            second_artifact = second_process / "stage2_corrected.fit"
            second_artifact.write_bytes(b"new-checkpoint")
            real_atomic_write = run_manifest.atomic_write_json

            def fail_checkpoint_manifest(path, payload):
                if Path(path) == manifest_path:
                    raise OSError("simulated manifest failure")
                return real_atomic_write(path, payload)

            with mock.patch.object(
                run_manifest,
                "atomic_write_json",
                side_effect=fail_checkpoint_manifest,
            ):
                with self.assertRaises(OSError):
                    publish_formal_checkpoint(
                        run_manifest_path=second_run.manifest_path,
                        stage_number=2,
                        artifact_path=second_artifact,
                        semantic_context=self._stage2_semantic_context(),
                    )

            checkpoint = workspace.checkpoints_dir / "stage2_corrected.fit"
            self.assertEqual(checkpoint.read_bytes(), b"trusted-old-checkpoint")
            self.assertEqual(manifest_path.read_bytes(), previous_manifest)

    def test_retention_keeps_latest_delivery_and_only_prunes_verified_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))

            def completed_run(
                run_id: str,
                output_name: str,
                content: bytes,
                *,
                debug_mode: bool = False,
            ):
                run = begin_task_run(
                    workspace=workspace,
                    source_record=source_record,
                    run_id=run_id,
                )
                output = run.root / output_name
                output.write_bytes(content)
                process = run.root / "process"
                process.mkdir()
                (process / "temporary.fit").write_bytes(b"temporary")
                plan = task_plan.build_processing_plan(
                    run_id=run.run_id,
                    generated_at="2026-08-04T01:00:00Z",
                    input_record={"fingerprint": source_record["fingerprint"]},
                    input_state="linear",
                    input_trust="recognized",
                    metadata={
                        "config": {
                            "debug_mode": debug_mode,
                            "checkpoint_mode": False,
                        }
                    },
                )
                run_manifest.atomic_write_json(
                    run.root / "processing-plan.json",
                    plan,
                )
                result = {
                    "schema": "starun.pipeline-result.v1",
                    "run_id": run.run_id,
                    "status": "success",
                    "plan_hash": plan["plan_hash"],
                    "outputs": {
                        output.name: run_manifest.file_record(
                            output,
                            base_dir=run.root,
                        )
                    },
                }
                result["manifest_hash"] = run_manifest.canonical_payload_hash(result)
                run_manifest.atomic_write_json(
                    run.root / "pipeline-result.json",
                    result,
                )
                return run, output, process

            old_run, old_output, old_process = completed_run(
                "old-run",
                "old_processed.fit",
                b"old-delivery",
            )
            unrelated = old_run.root / "user-note.txt"
            unrelated.write_text("keep", encoding="utf-8")
            debug_run, debug_output, debug_process = completed_run(
                "debug-run",
                "debug_processed.fit",
                b"debug-delivery",
                debug_mode=True,
            )
            latest_run, latest_output, latest_process = completed_run(
                "latest-run",
                "latest_processed.fit",
                b"latest-delivery",
                debug_mode=True,
            )
            publish_latest_result_index(
                run_manifest_path=latest_run.manifest_path
            )

            report = apply_task_retention(
                workspace.root,
            )

            self.assertFalse(old_output.exists())
            self.assertFalse(debug_output.exists())
            self.assertTrue(latest_output.is_file())
            self.assertTrue(unrelated.is_file())
            self.assertFalse(old_process.exists())
            self.assertTrue(debug_process.is_dir())
            self.assertTrue(latest_process.is_dir())
            self.assertIn(
                old_output.relative_to(workspace.root).as_posix(),
                report["deleted_files"],
            )
            self.assertIn(
                debug_process.relative_to(workspace.root).as_posix(),
                report["preserved_debug_process_directories"],
            )
            self.assertEqual(
                report["retention_scope"],
                "per_run_frozen_debug_and_checkpoint_settings",
            )

    def test_latest_verified_compatible_checkpoint_selects_stage5(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            fingerprints = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={stage: {"value": stage} for stage in range(1, 6)},
            )
            self._write_checkpoints(workspace, fingerprints)

            inspection = inspect_task_workspace(
                workspace.root,
                current_resume_fingerprints=fingerprints,
            )
            discovery = discover_input(
                workspace.root,
                current_resume_fingerprints=fingerprints,
            )

        self.assertTrue(inspection["verified"])
        self.assertEqual(inspection["resume_after_stage"], 5)
        self.assertEqual(discovery.kind, InputKind.PRODUCT_TASK)
        self.assertEqual(discovery.resume_after_stage, 5)

    def test_corrupt_stage5_falls_back_to_verified_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            fingerprints = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={stage: {"value": stage} for stage in range(1, 6)},
            )
            self._write_checkpoints(workspace, fingerprints)
            (workspace.checkpoints_dir / "stage5_linear.fit").write_bytes(b"tampered")
            (workspace.root / "result_linear.fit").write_bytes(b"legacy-alias")

            inspection = inspect_task_workspace(
                workspace.root,
                current_resume_fingerprints=fingerprints,
            )

        self.assertTrue(inspection["verified"])
        self.assertEqual(inspection["resume_after_stage"], 2)
        self.assertIn("SHA-256", inspection["rejections"]["stage5"])

    def test_changed_stage4_configuration_falls_back_to_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            baseline = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={
                    1: {"import": "xisf"},
                    2: {"crop": "auto"},
                    3: {"background": "safe"},
                    4: {"color": "physical"},
                    5: {"denoise": "safe"},
                },
            )
            current = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={
                    1: {"import": "xisf"},
                    2: {"crop": "auto"},
                    3: {"background": "safe"},
                    4: {"color": "changed"},
                    5: {"denoise": "safe"},
                },
            )
            self._write_checkpoints(workspace, baseline)

            inspection = inspect_task_workspace(
                workspace.root,
                current_resume_fingerprints=current,
            )

        self.assertEqual(inspection["resume_after_stage"], 2)

    def test_native_crop_v4_stage2_and_stage5_fall_back_to_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, source_record, workspace = self._workspace(Path(td))
            legacy = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={
                    1: {"source_import": "read_only"},
                    2: {"boundary_correction": "native_crop_v4"},
                    3: {"background": "safe"},
                    4: {"color": "physical"},
                    5: {"denoise": "safe"},
                },
            )
            current = task_plan.build_resume_fingerprints(
                input_fingerprint=source_record["fingerprint"],
                stage_config={
                    1: {"source_import": "read_only"},
                    2: {"boundary_correction": "native_crop_v5"},
                    3: {"background": "safe"},
                    4: {"color": "physical"},
                    5: {"denoise": "safe"},
                },
            )
            self._write_checkpoints(workspace, legacy)

            inspection = inspect_task_workspace(
                workspace.root,
                current_resume_fingerprints=current,
            )

        self.assertTrue(inspection["verified"])
        self.assertEqual(inspection["resume_after_stage"], 1)
        self.assertIn("incompatible", inspection["rejections"]["stage5"])
        self.assertIn("incompatible", inspection["rejections"]["stage2"])
        self.assertIn("incompatible", inspection["rejections"]["stage5"])

    def test_nonformal_checkpoint_record_is_rejected(self) -> None:
        with self.assertRaisesRegex(WorkspaceError, "非正式续跑阶段"):
            build_checkpoint_manifest(
                task_id="task-1",
                source_fingerprint="fingerprint",
                checkpoints={"stage4": {}},
            )


if __name__ == "__main__":
    unittest.main()
