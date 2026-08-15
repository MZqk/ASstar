#!/usr/bin/env python3
"""Offscreen checks for the history window and reused historical run view."""

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
from gui.main_window import StarunGui, WORKSPACE_TASK  # noqa: E402
from gui.task_intake import PreparedTask  # noqa: E402
from pipeline.task_workspace import (  # noqa: E402
    begin_task_run,
    build_source_record,
    ensure_task_workspace,
)
from pipeline.processing_parameters import default_processing_parameters  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def _history_task(
    root: Path,
    *,
    name: str,
    status: str,
    run_id: str,
    processing_parameters=None,
):
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
        processing_parameters=processing_parameters,
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


def _window(root: Path) -> StarunGui:
    settings = QSettings(
        str(root / "settings.ini"),
        QSettings.Format.IniFormat,
    )
    return StarunGui(
        resources_override=Path(__file__).resolve().parents[1] / "resources",
        runtime_home_override=root / "runtime",
        settings_override=settings,
        history_path_override=root / "history.json",
    )


def _fits_card(keyword: str, value: object) -> str:
    if isinstance(value, str):
        rendered = f"'{value}'"
    elif isinstance(value, bool):
        rendered = "T" if value else "F"
    else:
        rendered = str(value)
    return f"{keyword:<8}= {rendered:<20}".ljust(80)[:80]


def _write_header_only_fits(path: Path, **metadata: object) -> None:
    cards = [_fits_card("SIMPLE", True)]
    cards.extend(_fits_card(key, value) for key, value in metadata.items())
    cards.append("END".ljust(80))
    header = "".join(cards).encode("ascii")
    header += b" " * ((2880 - len(header) % 2880) % 2880)
    path.write_bytes(header)


def test_selected_fits_displays_header_summary_in_task_settings(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "M31.fit"
        _write_header_only_fits(
            source,
            TELESCOP="Seestar S50",
            FILTER="No filter",
            EXPTIME=10,
            STACKCNT=30,
            OBJECT="M 31",
            NAXIS1=1080,
            NAXIS2=1920,
            BITPIX=16,
        )
        window = _window(root)
        try:
            window._schedule_initial_preview = lambda *_args, **_kwargs: None

            window._apply_input_path(source, remember=False)

            assert window.source_header_group.isHidden() is False
            assert window.source_header_device_label.text() == "Seestar S50"
            assert window.source_header_filter_label.text() == "No filter"
            assert window.source_header_exposure_label.text() == "10 秒/帧"
            assert "目标：M 31" in window.source_header_details_label.text()
            assert "叠加帧数：30" in window.source_header_details_label.text()
            assert "图像尺寸：1080 × 1920" in (
                window.source_header_details_label.text()
            )
            assert "未读取图像像素" in window.source_header_status_label.text()
        finally:
            window.close()


def test_history_window_groups_runs_and_opens_reused_run_view(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        processing_parameters = default_processing_parameters(
            general={"output_formats": ["fit"], "compute_mode": "cpu"}
        )
        processing_parameters["stages"]["7"]["overrides"][
            "asinh_stretch"
        ] = 3.1
        store = HistoryStore(root / "history.json", session_id="seed")
        task, _workspace, run = _register(
            store,
            _history_task(
                root,
                name="M81.fit",
                status=STATUS_FAILED,
                run_id="run-1",
                processing_parameters=processing_parameters,
            ),
        )
        window = _window(root)
        try:
            window._show_history()
            top = window.history_tree.topLevelItem(0)

            assert window.history_window.isVisible() is True
            assert window.workspace_stack.currentWidget() is window.empty_page
            assert top.text(0) == "M81"
            assert top.childCount() == 1
            assert "失败" in window.history_tree.itemWidget(top, 2).text()
            top.setExpanded(True)
            window.history_tree.setCurrentItem(top.child(0))
            assert window.history_open_btn.isEnabled() is True
            assert window.history_delete_btn.isEnabled() is False
            window._refresh_history_view()
            top = window.history_tree.topLevelItem(0)
            assert top.isExpanded() is True
            assert window.history_tree.currentItem() is top.child(0)

            window._open_history_run(task["task_key"], run.run_id)

            assert window._history_detail_mode is True
            assert window.history_window.isVisible() is False
            assert window.workspace_stack.currentWidget() is window.run_page
            assert "只读历史记录" in window.run_options_label.text()
            assert "CPU 兼容" in window.run_options_label.text()
            assert "Stage 7 主体拉伸：" in window.run_options_label.text()
            assert "项自定义" in window.run_options_label.text()
            assert window.return_task_btn.text() == "返回历史记录"
            assert window.rerun_btn.text() == "重新处理"

            window._return_to_task_setup()
            assert window._history_detail_mode is False
            assert window.history_window.isVisible() is True
            assert window.workspace_stack.currentWidget() is window.empty_page

            run_count = len(list(_workspace.runs_dir.iterdir()))
            window._open_history_run(task["task_key"], run.run_id)
            window._rerun_last_task()
            assert window.workspace_stack.currentWidget() is window.task_page
            assert Path(window.dir_edit.text()) == _workspace.root
            assert len(list(_workspace.runs_dir.iterdir())) == run_count
            assert window.processing_parameters["general"]["compute_mode"] == "cpu"
            assert (
                window.processing_parameters["stages"]["7"]["overrides"][
                    "asinh_stretch"
                ]
                == 3.1
            )
        finally:
            if window.preview_worker is not None:
                window.preview_worker.wait(2000)
            window.close()


def test_switching_input_clears_stage_overrides_but_keeps_general(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first = root / "first"
        second = root / "second"
        first.mkdir()
        second.mkdir()
        window = _window(root)
        try:
            payload = default_processing_parameters(
                general={"output_formats": ["fit"], "compute_mode": "cpu"}
            )
            payload["stages"]["8"]["overrides"]["nebula_saturation"] = 0.5
            window._restore_processing_settings(payload)
            window._processing_parameter_input_path = first.resolve()
            window._analyze_selected_directory = lambda: None

            window._apply_input_path(second, remember=False)

            assert window.processing_parameters["general"]["compute_mode"] == "cpu"
            assert window.processing_parameters["general"]["output_formats"] == [
                "fit"
            ]
            assert all(
                not entry["overrides"] and entry["mode"] == "auto"
                for entry in window.processing_parameters["stages"].values()
            )
        finally:
            window.close()


def test_qsettings_persists_only_general_and_expert_visibility(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        payload = default_processing_parameters(
            general={"output_formats": ["fit"], "compute_mode": "cpu"}
        )
        payload["stages"]["5"]["overrides"]["stage5_rl_iters"] = 12
        legacy_settings = QSettings(
            str(root / "settings.ini"),
            QSettings.Format.IniFormat,
        )
        legacy_settings.setValue("processing/rlIterations", 12)
        legacy_settings.sync()
        window = _window(root)
        try:
            assert "processing/rlIterations" in window.settings.allKeys()
            assert window.processing_parameters["stages"]["5"]["overrides"] == {}
            window._restore_processing_settings(payload)
            window._set_processing_expert_visible(True)
            window._save_settings()
        finally:
            window.close()

        settings = QSettings(
            str(root / "settings.ini"),
            QSettings.Format.IniFormat,
        )
        processing_keys = {
            key for key in settings.allKeys() if key.startswith("processing/")
        }
        assert processing_keys == {
            "processing/autoTuneEnabled",
            "processing/checkpointMode",
            "processing/computeMode",
            "processing/managedOutputEnabled",
            "processing/maxRetries",
            "processing/outputFormats",
            "processing/reviewOnly",
            "processing/retryDelay",
            "processing/reviewBundleEnabled",
            "processing/rlIterations",
        }

        restored = _window(root)
        try:
            assert restored.processing_parameters["general"]["compute_mode"] == "cpu"
            assert restored.processing_parameters["general"]["output_formats"] == [
                "fit"
            ]
            assert restored.processing_expert_visible is True
            assert all(
                not entry["overrides"] and entry["mode"] == "auto"
                for entry in restored.processing_parameters["stages"].values()
            )
        finally:
            restored.close()


def test_editing_an_automatic_stage_value_creates_task_override(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        try:
            field = "asinh_stretch"
            automatic = window._stage_parameter_auto_checks[field]
            control = window._stage_parameter_controls[field]

            assert automatic.isChecked() is True
            assert control.isEnabled() is True
            control.setValue(control.value() + 0.05)
            app.processEvents()

            assert automatic.isChecked() is False
            assert (
                window.processing_parameters["stages"]["7"]["overrides"][field]
                == control.value()
            )

            window._reset_stage_processing_parameters(7)
            assert automatic.isChecked() is True
            assert field not in window.processing_parameters["stages"]["7"]["overrides"]
        finally:
            window.close()


def test_stage8_palette_combo_creates_manual_ohs_override(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        try:
            field = "stage8_dualband_palette_selection"
            automatic = window._stage_parameter_auto_checks[field]
            control = window._stage_parameter_controls[field]

            assert automatic.isChecked() is True
            assert control.currentData() == "auto"
            assert control.isEnabled() is True
            control.setCurrentIndex(control.findData("OHS"))
            app.processEvents()

            assert automatic.isChecked() is False
            assert control.currentData() == "OHS"
            assert (
                window.processing_parameters["stages"]["8"]["overrides"][field]
                == "OHS"
            )

            window._reset_stage_processing_parameters(8)
            assert automatic.isChecked() is True
            assert control.currentData() == "auto"
            assert field not in window.processing_parameters["stages"]["8"][
                "overrides"
            ]
        finally:
            window.close()


def test_task_gate_profile_updates_effective_values_and_forces_review(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        try:
            field = "stage5_multiscale_detail_retention_min"
            control = window._stage_parameter_controls[field]
            effective = window._stage_parameter_effective_labels[field]
            follows_profile = window._stage_parameter_auto_checks[field]
            profile_combo = window.processing_gate_profile_combo

            assert follows_profile.text() == "跟随档位"
            assert control.isHidden() is True
            assert "0.82" in effective.text()

            profile_combo.setCurrentIndex(profile_combo.findData("relaxed"))
            app.processEvents()
            assert window.processing_parameters["gate_profile"] == "relaxed"
            assert "0.27333333" in effective.text()
            assert window.review_combo.isEnabled() is True

            profile_combo.setCurrentIndex(profile_combo.findData("unlimited"))
            app.processEvents()
            assert window.processing_parameters["gate_profile"] == "unlimited"
            assert "0.082" in effective.text()
            assert window.review_combo.currentData() is True
            assert window.review_combo.isEnabled() is False
            assert window.review_only is False
            assert window.processing_gate_profile_banner.property("tone") == "warning"
            runtime_env, _unset = window._processing_runtime_configuration("auto")
            assert runtime_env["STARUN_FORCE_REVIEW_ONLY_OUTPUT"] == "1"

            profile_combo.setCurrentIndex(profile_combo.findData("default"))
            app.processEvents()
            assert window.review_combo.currentData() is False
            assert window.review_combo.isEnabled() is True
            follows_profile.setChecked(False)
            app.processEvents()
            assert control.isHidden() is False
            assert field in window.processing_parameters["stages"]["5"]["overrides"]
        finally:
            window.close()


def test_processing_stage_accordion_expert_groups_and_dependencies(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _window(Path(td))
        try:
            assert window._stage_parameter_headers[2].isChecked() is True
            assert sum(
                int(header.isChecked())
                for header in window._stage_parameter_headers.values()
            ) == 1

            window._set_processing_expert_visible(True)
            for stage, headers in window._stage_expert_section_headers.items():
                if "execution" in headers:
                    assert headers["execution"].isChecked() is True
                assert all(
                    not header.isChecked()
                    for section, header in headers.items()
                    if section != "execution"
                )

            mode = window._stage_parameter_controls["stage8_processing_mode"]
            mode.setCurrentIndex(mode.findData("preserve"))
            app.processEvents()
            saturation_toggle = window._stage_parameter_controls[
                "stage8_nebula_saturation_enabled"
            ]
            assert saturation_toggle.isEnabled() is False

            failure = window._stage_parameter_controls["stage8_failure_action"]
            failure.setCurrentIndex(failure.findData("stop"))
            app.processEvents()
            assert (
                window.processing_parameters["stages"]["8"]["overrides"][
                    "stage8_failure_action"
                ]
                == "stop"
            )
            window._set_processing_expert_visible(False)
            window._set_processing_expert_visible(True)
            assert failure.currentData() == "stop"
        finally:
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


def test_history_window_is_singleton_and_restores_geometry(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        window = _window(root)
        window.show()
        try:
            window._show_history()
            history_window = window.history_window
            history_window.resize(880, 560)
            history_window.move(72, 84)
            window.history_search_edit.setText("M42")
            app.processEvents()

            history_window.close()
            app.processEvents()
            assert history_window.isVisible() is False

            window._show_history()
            app.processEvents()
            assert window.history_window is history_window
            assert window.history_search_edit.text() == "M42"
            assert history_window.size().width() == 880
            assert history_window.size().height() == 560
        finally:
            window.close()

        restored = _window(root)
        try:
            # Qt clamps restored geometry to the current display; the offscreen
            # test display is only 800 px wide.
            assert 760 <= restored.history_window.size().width() <= 880
            assert restored.history_window.size().height() == 560
        finally:
            restored.close()


def test_return_from_history_detail_restores_prior_main_workspace(app) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = HistoryStore(root / "history.json", session_id="seed")
        task, _workspace, run = _register(
            store,
            _history_task(
                root,
                name="NGC7000.fit",
                status=STATUS_SUCCESS,
                run_id="run-1",
            ),
        )
        original_task_root = root / "selected-task"
        original_task_root.mkdir()
        window = _window(root)
        try:
            window.dir_edit.setText(str(original_task_root))
            window._last_task_root = original_task_root
            window._show_workspace(WORKSPACE_TASK)
            window._show_history()

            window._open_history_run(task["task_key"], run.run_id)
            assert window.workspace_stack.currentWidget() is window.run_page

            window._return_to_task_setup()
            assert window.workspace_stack.currentWidget() is window.task_page
            assert window._last_task_root == original_task_root
            assert window.history_window.isVisible() is True
        finally:
            window.close()


def test_running_task_keeps_history_browsable_but_locks_mutations(
    app,
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = HistoryStore(root / "history.json", session_id="seed")
        task, _workspace, run = _register(
            store,
            _history_task(
                root,
                name="M101.fit",
                status=STATUS_SUCCESS,
                run_id="run-1",
            ),
        )
        window = _window(root)
        window.show()
        messages: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "information",
            lambda *args: messages.append(args),
        )
        try:
            window._show_history()
            child = window.history_tree.topLevelItem(0).child(0)
            window.history_tree.setCurrentItem(child)
            window._set_running(True)
            app.processEvents()

            assert window.history_window.isVisible() is True
            assert window.history_action.isEnabled() is True
            assert window.history_mode_label.isVisible() is True
            assert window.history_open_btn.isEnabled() is False
            assert window.history_delete_btn.isEnabled() is False

            window._open_history_run(task["task_key"], run.run_id)
            assert messages
            assert messages[0][0] is window.history_window
            assert window.history_window.isVisible() is True
            assert window.workspace_stack.currentWidget() is window.empty_page
            assert window._history_detail_mode is False
        finally:
            window._set_running(False)
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
