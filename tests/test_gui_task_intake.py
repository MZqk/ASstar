#!/usr/bin/env python3
"""Regression tests for deterministic serial GUI task preparation."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.task_intake import (  # noqa: E402
    RESUME_CHECKPOINT_PATH_ENV,
    TASK_RUN_MANIFEST_ENV,
    describe_input_plan,
    discover_input_for_processing_settings,
    prepare_task_queue,
    stage_config_from_processing_settings,
)
from pipeline import run_manifest, task_plan  # noqa: E402
from pipeline.input_discovery import (  # noqa: E402
    DiscoveryTrust,
    InputDiscovery,
    InputKind,
    LightGroup,
    discover_input,
)
from pipeline.stage_contracts import FORMAL_RESUME_STAGES, stage_contract  # noqa: E402
from pipeline.task_workspace import (  # noqa: E402
    CHECKPOINT_MANIFEST_REL,
    build_checkpoint_manifest,
    build_source_record,
    ensure_task_workspace,
)


SETTINGS = {
    "color_calibration": "pcc",
    "filter_hint": "auto",
    "pcc_timeout_sec": 180,
    "local_wb_gain_limit": 1.2,
    "denoise_mode": "auto",
    "deconvolution_mode": "auto",
    "graxpert_model_path": "",
    "compute_mode": "auto",
    "builtin_denoise_strength": 0.5,
    "graxpert_deconv_strength": 0.3,
    "rl_iterations": 8,
    "rl_maxstars": 2000,
}


class GuiTaskIntakeTests(unittest.TestCase):
    def test_plan_copy_explains_review_and_verified_resume_without_mode_choice(self) -> None:
        review = InputDiscovery(
            selected_path=Path("/tmp/review.png"),
            kind=InputKind.REVIEW_FILE,
            trust=DiscoveryTrust.REVIEW_REQUIRED,
            summary="review",
            master_file=Path("/tmp/review.png"),
        )
        resume = InputDiscovery(
            selected_path=Path("/tmp/task"),
            kind=InputKind.PRODUCT_TASK,
            trust=DiscoveryTrust.VERIFIED,
            summary="resume",
            task_directory=Path("/tmp/task"),
            resume_after_stage=5,
        )

        review_plan = describe_input_plan(review)
        resume_plan = describe_input_plan(resume)

        self.assertIn("Stage 3–9", review_plan.summary)
        self.assertIn("Stage 6", resume_plan.summary)
        self.assertIn("✓ Stage 1–5", resume_plan.linear_phase)

    def test_explicit_master_creates_one_independent_auto_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "capture" / "NGC6910.xisf"
            source.parent.mkdir()
            source.write_bytes(b"XISF0100-master")
            discovery = discover_input(source)

            queue = prepare_task_queue(
                discovery,
                processing_settings=SETTINGS,
                now=datetime(2026, 8, 4, tzinfo=timezone.utc),
            )
            task = queue.tasks[0]

        self.assertEqual(len(queue.tasks), 1)
        self.assertEqual(task.input_mode, "auto")
        self.assertIsNone(task.resume_after_stage)
        self.assertEqual(
            task.runtime_overrides[TASK_RUN_MANIFEST_ENV],
            str(task.run.manifest_path),
        )
        self.assertTrue(task.source_record["read_only"])

    def test_source_hashing_can_be_cancelled_before_task_creation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "capture" / "master.fit"
            source.parent.mkdir()
            source.write_bytes(b"master")
            discovery = discover_input(source)

            with self.assertRaises(InterruptedError):
                prepare_task_queue(
                    discovery,
                    processing_settings=SETTINGS,
                    cancel_check=lambda: True,
                )

            self.assertFalse((source.parent / "SeestarSuperimpose").exists())

    def test_verified_legacy_directory_creates_read_only_stage1_migration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "legacy"
            root.mkdir()
            checkpoint = root / "result_linear.fit"
            checkpoint.write_bytes(b"trusted-legacy-linear")
            plan = {
                "schema": "seestar.processing-plan.v1",
                "run_id": "legacy-run",
            }
            plan["plan_hash"] = run_manifest.canonical_payload_hash(plan)
            run_manifest.atomic_write_json(root / "processing-plan.json", plan)
            result = {
                "schema": "seestar.pipeline-result.v1",
                "status": "success",
                "plan_hash": plan["plan_hash"],
                "checkpoints": {
                    "result_linear": {
                        **run_manifest.file_record(checkpoint, base_dir=root),
                        "state": "linear",
                    }
                },
            }
            result["manifest_hash"] = run_manifest.canonical_payload_hash(result)
            run_manifest.atomic_write_json(root / "pipeline-result.json", result)

            queue = prepare_task_queue(
                discover_input(root),
                processing_settings=SETTINGS,
            )
            task = queue.tasks[0]

        self.assertEqual(task.input_mode, "auto")
        self.assertIsNone(task.resume_after_stage)
        self.assertEqual(task.source_record["kind"], "master_file")
        self.assertEqual(
            task.source_record["group"]["migration"],
            "legacy_read_only_v1",
        )
        self.assertEqual(
            Path(task.source_record["files"][0]["path"]),
            checkpoint.resolve(),
        )

    def test_multiple_light_groups_become_a_serial_task_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "capture"
            files = []
            for index in range(2):
                path = root / f"Light_{index}.fit"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"light-{index}".encode("ascii"))
                files.append(path)
            groups = (
                LightGroup(
                    key="lp",
                    target="NGC 6910",
                    filter_name="Seestar LP",
                    camera="Seestar S50",
                    geometry="1080x1920@1x1",
                    files=(files[0],),
                    total_bytes=files[0].stat().st_size,
                ),
                LightGroup(
                    key="clear",
                    target="NGC 6910",
                    filter_name="No filter",
                    camera="Seestar S50",
                    geometry="1080x1920@1x1",
                    files=(files[1],),
                    total_bytes=files[1].stat().st_size,
                ),
            )
            discovery = InputDiscovery(
                selected_path=root.resolve(),
                source_root=root.resolve(),
                kind=InputKind.LIGHT_DIRECTORY,
                trust=DiscoveryTrust.RECOGNIZED,
                summary="two groups",
                light_groups=groups,
            )

            queue = prepare_task_queue(
                discovery,
                processing_settings=SETTINGS,
                now=datetime(2026, 8, 4, tzinfo=timezone.utc),
            )

        self.assertEqual(len(queue.tasks), 2)
        self.assertEqual([task.queue_index for task in queue.tasks], [1, 2])
        self.assertEqual({task.queue_total for task in queue.tasks}, {2})
        self.assertNotEqual(queue.tasks[0].workspace.root, queue.tasks[1].workspace.root)
        self.assertTrue(all(task.input_mode == "auto" for task in queue.tasks))

    def test_product_task_selects_latest_checkpoint_compatible_with_settings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source" / "master.fit"
            source.parent.mkdir()
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
                stage_config=stage_config_from_processing_settings(SETTINGS),
            )
            checkpoint_records = {}
            for stage_number in FORMAL_RESUME_STAGES:
                contract = stage_contract(stage_number)
                path = workspace.checkpoints_dir / contract.primary_artifact
                path.write_bytes(f"stage-{stage_number}".encode("ascii"))
                key = f"stage{stage_number}"
                checkpoint_records[key] = {
                    "stage": stage_number,
                    "artifact": contract.primary_artifact,
                    "path": path.relative_to(workspace.root).as_posix(),
                    "sha256": run_manifest.sha256_file(path),
                    "state": "linear",
                    "plan_hash": "previous-plan",
                    "config_fingerprint": fingerprints[key]["fingerprint"],
                }
            checkpoint_manifest = build_checkpoint_manifest(
                task_id=workspace.task_id,
                source_fingerprint=workspace.source_fingerprint,
                checkpoints=checkpoint_records,
            )
            run_manifest.atomic_write_json(
                workspace.root / CHECKPOINT_MANIFEST_REL,
                checkpoint_manifest,
            )
            discovery = discover_input(workspace.root)

            queue = prepare_task_queue(
                discovery,
                processing_settings=SETTINGS,
            )
            task = queue.tasks[0]
            frozen_run = run_manifest.load_json(task.run.manifest_path)

        self.assertEqual(task.resume_after_stage, 5)
        self.assertEqual(task.input_mode, "result_linear_resume")
        self.assertEqual(
            Path(task.runtime_overrides[RESUME_CHECKPOINT_PATH_ENV]).name,
            "stage5_linear.fit",
        )
        self.assertEqual(frozen_run["resume"]["stage"], 5)

    def test_product_task_card_rechecks_checkpoint_against_current_settings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source" / "master.fit"
            source.parent.mkdir()
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
                stage_config=stage_config_from_processing_settings(SETTINGS),
            )
            checkpoint_records = {}
            for stage_number in FORMAL_RESUME_STAGES:
                contract = stage_contract(stage_number)
                path = workspace.checkpoints_dir / contract.primary_artifact
                path.write_bytes(f"stage-{stage_number}".encode("ascii"))
                key = f"stage{stage_number}"
                checkpoint_records[key] = {
                    "stage": stage_number,
                    "artifact": contract.primary_artifact,
                    "path": path.relative_to(workspace.root).as_posix(),
                    "sha256": run_manifest.sha256_file(path),
                    "state": "linear",
                    "plan_hash": "previous-plan",
                    "config_fingerprint": fingerprints[key]["fingerprint"],
                }
            run_manifest.atomic_write_json(
                workspace.root / CHECKPOINT_MANIFEST_REL,
                build_checkpoint_manifest(
                    task_id=workspace.task_id,
                    source_fingerprint=workspace.source_fingerprint,
                    checkpoints=checkpoint_records,
                ),
            )
            changed_settings = {**SETTINGS, "denoise_mode": "off"}

            discovery = discover_input_for_processing_settings(
                workspace.root,
                processing_settings=changed_settings,
            )

        self.assertEqual(discovery.resume_after_stage, 2)
        self.assertIn("Stage 3", discovery.summary)


if __name__ == "__main__":
    unittest.main()
