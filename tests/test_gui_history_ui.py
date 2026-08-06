#!/usr/bin/env python3
"""Offscreen checks for the history page and reused historical run view."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
if not getattr(PySide6, "__version__", None):
    pytest.skip("requires real PySide6 modules", allow_module_level=True)

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.history_store import (  # noqa: E402
    STATUS_FAILED,
    STATUS_SUCCESS,
    HistoryStore,
)
import gui.main_window as main_window_module  # noqa: E402
from gui.main_window import SeestarGui  # noqa: E402
from gui.task_intake import PreparedTask  # noqa: E402
from pipeline.task_workspace import (  # noqa: E402
    begin_task_run,
    build_source_record,
    ensure_task_workspace,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def _history_task(root: Path, *, name: str, status: str, run_id: str):
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
    )
    run = begin_task_run(
        workspace=workspace,
        source_record=source_record,
        run_id=run_id,
    )
    return source_record, workspace, run, status


def _register(store: HistoryStore, fixture):
    source_record, workspace, run, status = fixture
    task = store.register_run(
        task_id=workspace.task_id,
        task_directory=workspace.root,
        source_fingerprint=workspace.source_fingerprint,
        source_record=source_record,
        run_id=run.run_id,
        run_directory=run.root,
        input_mode="auto",
    )
    store.update_run(
        task_key=task["task_key"],
        run_id=run.run_id,
        status=status,
        failure_reason="test failure" if status == STATUS_FAILED else None,
    )
    return task, workspace, run


def _window(root: Path) -> SeestarGui:
    settings = QSettings(
        str(root / "settings.ini"),
        QSettings.Format.IniFormat,
    )
    return SeestarGui(
        resources_override=Path(__file__).resolve().parents[1] / "resources",
        runtime_home_override=root / "runtime",
        settings_override=settings,
        history_path_override=root / "history.json",
    )


def test_history_page_groups_runs_and_opens_reused_run_view(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = HistoryStore(root / "history.json", session_id="seed")
        task, _workspace, run = _register(
            store,
            _history_task(
                root,
                name="M81.fit",
                status=STATUS_FAILED,
                run_id="run-1",
            ),
        )
        window = _window(root)
        try:
            window._show_history()
            top = window.history_tree.topLevelItem(0)

            assert top.text(0) == "M81"
            assert top.childCount() == 1
            assert "失败" in window.history_tree.itemWidget(top, 2).text()

            window._open_history_run(task["task_key"], run.run_id)

            assert window._history_detail_mode is True
            assert window.workspace_stack.currentWidget() is window.run_page
            assert "只读历史记录" in window.run_options_label.text()
            assert window.return_task_btn.text() == "返回历史记录"
            assert window.rerun_btn.text() == "重新处理"

            window._return_to_task_setup()
            assert window.workspace_stack.currentWidget() is window.history_page

            run_count = len(list(_workspace.runs_dir.iterdir()))
            window._open_history_run(task["task_key"], run.run_id)
            window._rerun_last_task()
            assert window.workspace_stack.currentWidget() is window.task_page
            assert Path(window.dir_edit.text()) == _workspace.root
            assert len(list(_workspace.runs_dir.iterdir())) == run_count
        finally:
            if window.preview_worker is not None:
                window.preview_worker.wait(2000)
            window.close()


def test_history_search_and_status_filter_apply_to_tasks(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = HistoryStore(root / "history.json", session_id="seed")
        _register(
            store,
            _history_task(
                root / "first",
                name="M81.fit",
                status=STATUS_SUCCESS,
                run_id="success-run",
            ),
        )
        _register(
            store,
            _history_task(
                root / "second",
                name="M42.fit",
                status=STATUS_FAILED,
                run_id="failed-run",
            ),
        )
        window = _window(root)
        try:
            window._show_history()
            assert window.history_tree.topLevelItemCount() == 2

            window.history_search_edit.setText("M42")
            assert window.history_tree.topLevelItemCount() == 1
            assert window.history_tree.topLevelItem(0).text(0) == "M42"

            window.history_search_edit.clear()
            window.history_status_combo.setCurrentIndex(1)
            assert window.history_tree.topLevelItemCount() == 1
            assert window.history_tree.topLevelItem(0).text(0) == "M81"
        finally:
            window.close()


def test_delete_moves_only_selected_verified_task_then_removes_index(
    app,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = HistoryStore(root / "history.json", session_id="seed")
        task, workspace, _run = _register(
            store,
            _history_task(
                root,
                name="M31.fit",
                status=STATUS_SUCCESS,
                run_id="run-1",
            ),
        )
        window = _window(root)

        class _FakeMessageBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()

            class StandardButton:
                Cancel = object()

            warnings = []

            def __init__(self, *_args, **_kwargs):
                self._move_button = None

            def setWindowTitle(self, *_args):
                return None

            def setIcon(self, *_args):
                return None

            def setText(self, *_args):
                return None

            def setInformativeText(self, *_args):
                return None

            def addButton(self, value, *_args):
                if value == "移到废纸篓":
                    self._move_button = object()
                    return self._move_button
                return object()

            def setDefaultButton(self, *_args):
                return None

            def exec(self):
                return None

            def clickedButton(self):
                return self._move_button

            @classmethod
            def warning(cls, *_args):
                cls.warnings.append(_args)

        class _FakeFile:
            moved_paths = []

            @classmethod
            def moveToTrash(cls, path):
                cls.moved_paths.append(path)
                return True, "/mock-trash/task"

        monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)
        monkeypatch.setattr(main_window_module, "QFile", _FakeFile)
        try:
            window._last_task_root = workspace.root
            window._show_history()
            window.history_tree.setCurrentItem(window.history_tree.topLevelItem(0))

            window._delete_selected_history_task()

            assert _FakeFile.moved_paths == [str(workspace.root.resolve())]
            assert window.history_store.find_task(task["task_key"]) is None
            assert window.workspace_stack.currentWidget() is window.empty_page
            assert _FakeMessageBox.warnings == []
        finally:
            window.close()


def test_gui_lifecycle_hooks_register_and_finish_history_run(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source_record, workspace, run, _status = _history_task(
            root,
            name="NGC7000.fit",
            status=STATUS_SUCCESS,
            run_id="run-1",
        )
        prepared = PreparedTask(
            queue_index=1,
            queue_total=1,
            workspace=workspace,
            run=run,
            source_record=source_record,
            input_mode="auto",
            resume_after_stage=None,
            checkpoint_fingerprints={},
            runtime_overrides={},
            display_label="NGC7000",
        )
        window = _window(root)
        try:
            window._register_history_run(prepared)
            window._update_active_history_run(STATUS_FAILED, failure_reason="preflight")

            recorded = window.history_store.tasks()[0]

            assert recorded["task_id"] == workspace.task_id
            assert recorded["runs"][0]["run_id"] == run.run_id
            assert recorded["runs"][0]["status"] == STATUS_FAILED
            assert recorded["runs"][0]["failure_reason"] == "preflight"
        finally:
            window.close()
