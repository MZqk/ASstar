#!/usr/bin/env python3
"""Regression tests for deterministic serial GUI task preparation."""
from __future__ import annotations

import sys
import tempfile
import unittest
import copy
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.task_intake import (  # noqa: E402
    TASK_RUN_MANIFEST_ENV,
    describe_input_plan,
    discover_input_for_processing_settings,
    prepare_task_queue,
    stage4_online_spcc_timeout_detected,
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
from pipeline.processing_parameters import default_processing_parameters  # noqa: E402
from pipeline.task_workspace import (  # noqa: E402
    CHECKPOINT_MANIFEST_REL,
    build_checkpoint_manifest,
    build_source_record,
    ensure_task_workspace,
)


SETTINGS = default_processing_parameters()
SETTINGS["stages"]["5"]["overrides"].update(
    {
        "denoise_mod": 0.5,
        "stage5_graxpert_deconv_strength": 0.3,
        "stage5_rl_iters": 8,
        "stage5_rl_maxstars": 2000,
    }
)


def _stage5_resume_semantics() -> dict[str, object]:
    return {
        "schema": "starun.resume-semantics.v1",
        "checkpoint_stage": 5,
        "channel_semantics": "broadband",
        "channel_profile": {"kind": "broadband"},
        "narrowband_channel_mapping": {
            "schema": "starun.narrowband-channel-mapping.v1",
            "mapping": "unknown",
            "confidence": 0.0,
            "evidence": "not_narrowband",
        },
        "target_profile": {},
        "pipeline_policy": {},
        "color_calibration_report": {},
        "upstream_review": {},
    }


def _stage2_resume_semantics() -> dict[str, object]:
    return {
        "schema": "starun.resume-semantics.v2",
        "checkpoint_stage": 2,
        "review_requirements": [],
        "stage2_crop": {
            "original_dimensions": {"width": 1920, "height": 1080},
            "final_dimensions": {"width": 1920, "height": 1080},
            "cumulative_crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "field_rotation_passes": 0,
            "final_residual_detection": {
                "accepted": False,
                "reason": "not_run",
            },
        },
    }


def _resume_semantics(stage_number: int):
    if stage_number == 2:
        return _stage2_resume_semantics()
    if stage_number == 5:
        return _stage5_resume_semantics()
    return None


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

            self.assertFalse((source.parent / "Starun").exists())

    def test_legacy_directory_is_rejected_without_migration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "legacy"
            root.mkdir()
            checkpoint = root / "result_linear.fit"
            checkpoint.write_bytes(b"trusted-legacy-linear")
            plan = {
                "schema": "starun.processing-plan.v1",
                "run_id": "legacy-run",
            }
            plan["plan_hash"] = run_manifest.canonical_payload_hash(plan)
            run_manifest.atomic_write_json(root / "processing-plan.json", plan)
            result = {
                "schema": "starun.pipeline-result.v1",
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

            with self.assertRaisesRegex(Exception, "Stage 1"):
                prepare_task_queue(
                    discover_input(root),
                    processing_settings=SETTINGS,
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
            frozen_parameters = [
                run_manifest.load_json(task.run.manifest_path)[
                    "processing_parameters"
                ]
                for task in queue.tasks
            ]

        self.assertEqual(len(queue.tasks), 2)
        self.assertEqual([task.queue_index for task in queue.tasks], [1, 2])
        self.assertEqual({task.queue_total for task in queue.tasks}, {2})
        self.assertNotEqual(queue.tasks[0].workspace.root, queue.tasks[1].workspace.root)
        self.assertTrue(all(task.input_mode == "auto" for task in queue.tasks))
        self.assertEqual(frozen_parameters[0], frozen_parameters[1])
        self.assertEqual(
            frozen_parameters[0]["stages"]["5"]["overrides"]["stage5_rl_maxstars"],
            1000,
        )

    def test_exact_duplicate_source_and_processing_is_skipped_in_new_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "capture"
            source = root / "Light.fit"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"same-light")
            duplicate_group = LightGroup(
                key="same",
                target="M45",
                filter_name="Seestar LP",
                camera="Seestar S50",
                geometry="1080x1920@1x1",
                files=(source,),
                total_bytes=source.stat().st_size,
            )
            groups = (duplicate_group, duplicate_group)
            discovery = InputDiscovery(
                selected_path=root.resolve(),
                source_root=root.resolve(),
                kind=InputKind.LIGHT_DIRECTORY,
                trust=DiscoveryTrust.RECOGNIZED,
                summary="duplicate groups",
                light_groups=groups,
            )

            queue = prepare_task_queue(
                discovery,
                processing_settings=SETTINGS,
                now=datetime(2026, 8, 4, tzinfo=timezone.utc),
            )

        self.assertEqual(len(queue.tasks), 1)
        self.assertEqual(len(queue.skipped_duplicates), 1)
        self.assertEqual(
            queue.skipped_duplicates[0]["reason_code"],
            "exact_source_and_processing_duplicate",
        )

    def test_online_spcc_timeout_detection_ignores_localgaia_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_root = Path(td)
            report = run_root / "process" / "color_calibration_report.json"
            report.parent.mkdir()
            cases = (
                ("timeout", "catalog:gaia", "online_unverified", True),
                ("failed", "catalog:gaia", "online_unverified", False),
                ("timeout", "catalog:localgaia", "local_verified", False),
            )
            for status, label, readiness, expected in cases:
                with self.subTest(status=status, label=label):
                    run_manifest.atomic_write_json(
                        report,
                        {
                            "spcc": {
                                "attempts": [
                                    {
                                        "status": status,
                                        "label": label,
                                        "spcc_readiness": readiness,
                                    }
                                ]
                            }
                        },
                    )
                    self.assertEqual(
                        stage4_online_spcc_timeout_detected(run_root),
                        expected,
                    )

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
                    "run_manifest_hash": "previous-run-manifest",
                    "config_fingerprint": fingerprints[key]["fingerprint"],
                    "semantic_context": _resume_semantics(stage_number),
                    "semantic_context_status": (
                        "verified" if stage_number in {2, 5} else "not_applicable"
                    ),
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
        self.assertEqual(task.input_mode, "stage5_linear_resume")
        self.assertEqual(frozen_run["resume"]["stage"], 5)
        self.assertEqual(Path(frozen_run["resume"]["path"]).name, "stage5_linear.fit")
        self.assertEqual(
            set(task.runtime_overrides),
            {TASK_RUN_MANIFEST_ENV},
        )

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
                    "run_manifest_hash": "previous-run-manifest",
                    "config_fingerprint": fingerprints[key]["fingerprint"],
                    "semantic_context": _resume_semantics(stage_number),
                    "semantic_context_status": (
                        "verified" if stage_number in {2, 5} else "not_applicable"
                    ),
                }
            run_manifest.atomic_write_json(
                workspace.root / CHECKPOINT_MANIFEST_REL,
                build_checkpoint_manifest(
                    task_id=workspace.task_id,
                    source_fingerprint=workspace.source_fingerprint,
                    checkpoints=checkpoint_records,
                ),
            )
            changed_settings = copy.deepcopy(SETTINGS)
            changed_settings["stages"]["5"]["overrides"]["denoise_mod"] = 0.0

            discovery = discover_input_for_processing_settings(
                workspace.root,
                processing_settings=changed_settings,
            )

        self.assertEqual(discovery.resume_after_stage, 2)
        self.assertIn("Stage 3", discovery.summary)


if __name__ == "__main__":
    unittest.main()
