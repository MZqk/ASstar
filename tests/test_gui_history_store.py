#!/usr/bin/env python3
"""Regression tests for GUI processing history and safe task deletion."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from gui.history_store import (
    HISTORY_SCHEMA,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_LABELS,
    STATUS_PARTIAL_SUCCESS,
    STATUS_PREPARING,
    STATUS_REVIEW_REQUIRED,
    STATUS_STOPPED,
    STATUS_SUCCESS,
    HistoryStore,
    UnsafeTaskDeletionError,
    load_verified_pipeline_result,
    validate_deletable_task_root,
    verified_result_files,
    verify_history_run,
)
from pipeline import run_manifest
from pipeline.task_workspace import (
    TASK_CONTAINER_NAME,
    begin_task_run,
    build_source_record,
    ensure_task_workspace,
    open_task_workspace,
)


class GuiHistoryStoreTests(unittest.TestCase):
    def _task(self, root: Path, *, name: str = "M81.fit"):
        source = root / "capture" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"linear-master")
        source_record = build_source_record(
            source_kind="master_file",
            selected_path=source,
            files=(source,),
        )
        workspace = ensure_task_workspace(
            source_record=source_record,
            selected_path=source,
            created_at="2026-08-05T01:00:00Z",
        )
        run = begin_task_run(
            workspace=workspace,
            source_record=source_record,
            run_id="run-1",
            generated_at="2026-08-05T01:02:00Z",
        )
        return source_record, workspace, run

    @staticmethod
    def _register(store, source_record, workspace, run):
        return store.register_run(
            task_id=workspace.task_id,
            task_directory=workspace.root,
            source_fingerprint=workspace.source_fingerprint,
            source_record=source_record,
            run_id=run.run_id,
            run_directory=run.root,
            input_mode="auto",
            started_at="2026-08-05T01:02:00Z",
        )

    def test_history_only_contains_explicitly_registered_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _source, _workspace, _run = self._task(root)
            store = HistoryStore(root / "history.json", session_id="test")

            self.assertEqual(store.tasks(), [])

    def test_register_update_and_sort_runs_inside_one_task(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_record, workspace, run = self._task(root)
            store = HistoryStore(root / "history.json", session_id="test")
            task = self._register(store, source_record, workspace, run)

            self.assertEqual(task["display_name"], "M81")
            self.assertEqual(task["latest_status"], STATUS_PREPARING)
            store.update_run(
                task_key=task["task_key"],
                run_id=run.run_id,
                status=STATUS_SUCCESS,
                completed_at="2026-08-05T01:10:00Z",
            )
            run2 = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="run-2",
                generated_at="2026-08-05T02:00:00Z",
            )
            self._register(store, source_record, workspace, run2)
            store.update_run(
                task_key=task["task_key"],
                run_id=run2.run_id,
                status=STATUS_FAILED,
                completed_at="2026-08-05T02:05:00Z",
                failure_reason="preflight failed",
                exit_code=2,
            )

            tasks = store.tasks()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["latest_status"], STATUS_FAILED)
        self.assertEqual(
            [item["run_id"] for item in tasks[0]["runs"]],
            ["run-2", "run-1"],
        )
        self.assertEqual(tasks[0]["runs"][0]["failure_reason"], "preflight failed")

    def test_stale_active_run_becomes_interrupted_when_owner_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_record, workspace, run = self._task(root)
            first = HistoryStore(
                root / "history.json",
                session_id="old-session",
                owner_pid=999_999_999,
            )
            self._register(first, source_record, workspace, run)
            second = HistoryStore(root / "history.json", session_id="new-session")

            changed = second.mark_incomplete_runs_interrupted()
            recorded = second.tasks()[0]["runs"][0]

        self.assertEqual(changed, 1)
        self.assertEqual(recorded["status"], STATUS_INTERRUPTED)
        self.assertTrue(recorded["completed_at"])

    def test_all_terminal_statuses_are_preserved_with_user_labels(self) -> None:
        terminal_statuses = (
            STATUS_SUCCESS,
            STATUS_PARTIAL_SUCCESS,
            STATUS_REVIEW_REQUIRED,
            STATUS_FAILED,
            STATUS_STOPPED,
            STATUS_INTERRUPTED,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_record, workspace, first_run = self._task(root)
            store = HistoryStore(root / "history.json", session_id="test")
            registered = self._register(store, source_record, workspace, first_run)
            store.update_run(
                task_key=registered["task_key"],
                run_id=first_run.run_id,
                status=terminal_statuses[0],
            )
            for index, status in enumerate(terminal_statuses[1:], start=2):
                run = begin_task_run(
                    workspace=workspace,
                    source_record=source_record,
                    run_id=f"run-{index}",
                )
                self._register(store, source_record, workspace, run)
                store.update_run(
                    task_key=registered["task_key"],
                    run_id=run.run_id,
                    status=status,
                )

            recorded_statuses = {
                run["status"] for run in store.tasks()[0]["runs"]
            }

        self.assertEqual(recorded_statuses, set(terminal_statuses))
        self.assertTrue(all(STATUS_LABELS[status] for status in terminal_statuses))

    def test_missing_task_can_be_relocated_only_when_a_new_run_is_registered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_record, workspace, run = self._task(root)
            store = HistoryStore(root / "history.json", session_id="test")
            first = self._register(store, source_record, workspace, run)

            destination_container = root / "moved" / TASK_CONTAINER_NAME
            destination_container.mkdir(parents=True)
            moved_root = destination_container / workspace.root.name
            shutil.move(str(workspace.root), str(moved_root))
            moved_workspace, moved_source = open_task_workspace(moved_root)
            run2 = begin_task_run(
                workspace=moved_workspace,
                source_record=moved_source,
                run_id="run-2",
            )
            moved = self._register(
                store,
                moved_source,
                moved_workspace,
                run2,
            )

            tasks = store.tasks()

        self.assertEqual(len(tasks), 1)
        self.assertEqual(first["task_id"], moved["task_id"])
        self.assertEqual(tasks[0]["task_directory"], str(moved_root.resolve()))
        self.assertEqual(len(tasks[0]["runs"]), 2)

    def test_corrupt_index_is_backed_up_before_starting_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root / "history.json"
            index.write_text("not-json", encoding="utf-8")
            store = HistoryStore(index, session_id="test")

            self.assertEqual(store.tasks(), [])
            self.assertIsNotNone(store.last_recovery_path)
            self.assertTrue(store.last_recovery_path.is_file())
            self.assertFalse(index.exists())

    def test_remove_task_only_changes_history_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_record, workspace, run = self._task(root)
            store = HistoryStore(root / "history.json", session_id="test")
            task = self._register(store, source_record, workspace, run)

            removed = store.remove_task(task["task_key"])

            self.assertTrue(removed)
            self.assertTrue(workspace.root.is_dir())
            self.assertEqual(store.tasks(), [])

    def test_verified_run_and_pipeline_result_require_valid_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _source_record, workspace, run = self._task(root)
            result = {
                "schema": "starun.pipeline-result.v1",
                "run_id": run.run_id,
                "status": "success",
                "outputs": {},
            }
            result["manifest_hash"] = run_manifest.canonical_payload_hash(result)
            run_manifest.atomic_write_json(run.root / "pipeline-result.json", result)

            verified_workspace, payload, run_root = verify_history_run(
                workspace.root,
                run.run_id,
            )
            loaded = load_verified_pipeline_result(run_root)

            self.assertEqual(verified_workspace.task_id, workspace.task_id)
            self.assertEqual(payload["run_id"], run.run_id)
            self.assertEqual(loaded["status"], "success")

            tampered = json.loads((run.root / "pipeline-result.json").read_text())
            tampered["status"] = "failed"
            run_manifest.atomic_write_json(run.root / "pipeline-result.json", tampered)
            with self.assertRaisesRegex(Exception, "哈希无效"):
                load_verified_pipeline_result(run.root)

    def test_verified_result_files_reject_missing_hashes_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _source_record, _workspace, run = self._task(root)
            preview = run.root / "result.png"
            preview.write_bytes(b"preview")
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            result = {
                "outputs": {
                    "valid": run_manifest.file_record(preview, base_dir=run.root),
                    "escape": {
                        "path": "../../outside.png",
                        "sha256": run_manifest.sha256_file(outside),
                    },
                    "missing_hash": {"path": "result.png", "sha256": ""},
                }
            }

            files = verified_result_files(run.root, result, suffixes={".png"})

        self.assertEqual(files, (preview.resolve(),))

    def test_deletion_validator_accepts_only_exact_verified_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _source_record, workspace, _run = self._task(root)

            verified = validate_deletable_task_root(workspace.root)
            self.assertEqual(verified.task_id, workspace.task_id)

            with self.assertRaises(UnsafeTaskDeletionError):
                validate_deletable_task_root(workspace.root.parent)

            link = workspace.root.parent / "linked-task"
            link.symlink_to(workspace.root, target_is_directory=True)
            with self.assertRaisesRegex(UnsafeTaskDeletionError, "符号链接"):
                validate_deletable_task_root(link)

            renamed = workspace.root.with_name("not-the-task-id")
            workspace.root.rename(renamed)
            with self.assertRaisesRegex(UnsafeTaskDeletionError, "task_id"):
                validate_deletable_task_root(renamed)

    def test_written_index_uses_versioned_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_record, workspace, run = self._task(root)
            index = root / "history.json"
            store = HistoryStore(index, session_id="test")
            self._register(store, source_record, workspace, run)

            payload = json.loads(index.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], HISTORY_SCHEMA)


if __name__ == "__main__":
    unittest.main()
