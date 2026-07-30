#!/usr/bin/env python3
"""GUI runtime mode and plugin integrity helper tests."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_MODULE_PATH = REPO_ROOT / "gui" / "seestar_gui_app.py"


def _ensure_fake_pyside6() -> None:
    if "PySide6" in sys.modules:
        return

    class _DummySignal:
        def connect(self, *_args, **_kwargs):
            return None

        def emit(self, *_args, **_kwargs):
            return None

    class _DummyObject:
        def __init__(self, *_args, **_kwargs):
            pass

    qt_mod = types.ModuleType("PySide6")
    qt_core = types.ModuleType("PySide6.QtCore")
    qt_gui = types.ModuleType("PySide6.QtGui")
    qt_widgets = types.ModuleType("PySide6.QtWidgets")

    qt_core.QThread = type("QThread", (_DummyObject,), {})
    qt_core.QSettings = type("QSettings", (_DummyObject,), {})
    qt_core.QTimer = type("QTimer", (_DummyObject,), {})
    qt_core.QUrl = type("QUrl", (_DummyObject,), {})
    qt_core.Signal = lambda *_a, **_k: _DummySignal()
    qt_core.Qt = SimpleNamespace()

    qt_gui.QAction = type("QAction", (_DummyObject,), {})
    qt_gui.QDesktopServices = type("QDesktopServices", (_DummyObject,), {})
    qt_gui.QImageReader = type("QImageReader", (_DummyObject,), {})
    qt_gui.QPixmap = type("QPixmap", (_DummyObject,), {})
    qt_gui.QTextCursor = type("QTextCursor", (_DummyObject,), {})

    for name in (
        "QApplication",
        "QCheckBox",
        "QComboBox",
        "QDialog",
        "QDialogButtonBox",
        "QDoubleSpinBox",
        "QFileDialog",
        "QFormLayout",
        "QFrame",
        "QGraphicsScene",
        "QGraphicsView",
        "QGridLayout",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QMainWindow",
        "QMessageBox",
        "QPushButton",
        "QPlainTextEdit",
        "QProgressBar",
        "QScrollArea",
        "QSizePolicy",
        "QSplitter",
        "QSpinBox",
        "QStackedWidget",
        "QToolBar",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(qt_widgets, name, type(name, (_DummyObject,), {}))

    qt_mod.QtCore = qt_core
    qt_mod.QtGui = qt_gui
    qt_mod.QtWidgets = qt_widgets

    sys.modules["PySide6"] = qt_mod
    sys.modules["PySide6.QtCore"] = qt_core
    sys.modules["PySide6.QtGui"] = qt_gui
    sys.modules["PySide6.QtWidgets"] = qt_widgets


def _load_gui_module():
    _ensure_fake_pyside6()
    spec = importlib.util.spec_from_file_location(
        "seestar_gui_runtime_test_module",
        GUI_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {GUI_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gui_module = _load_gui_module()
bootstrap_module = sys.modules[gui_module.BootstrapWorker.__module__]
pipeline_worker_module = sys.modules[gui_module.PipelineWorker.__module__]


class GuiRuntimeModesTests(unittest.TestCase):
    def test_processing_parameters_expand_inside_task_sheet(self):
        class _Widget:
            def __init__(self):
                self.visible = False
                self.text = ""
                self.accessible_name = ""

            def setVisible(self, visible):
                self.visible = bool(visible)

            def setText(self, text):
                self.text = str(text)

            def setAccessibleName(self, name):
                self.accessible_name = str(name)

        panel = _Widget()
        button = _Widget()
        proxy = SimpleNamespace(
            processing_parameters_expanded=False,
            processing_params_panel=panel,
            processing_params_btn=button,
            advanced_toggle_btn=SimpleNamespace(isChecked=lambda: True),
            _restoring_settings=True,
            _update_processing_sheet_availability=lambda: None,
        )

        gui_module.SeestarGui._set_processing_parameters_expanded(proxy, True)

        self.assertTrue(proxy.processing_parameters_expanded)
        self.assertTrue(panel.visible)
        self.assertEqual(button.text, "收起处理参数")
        self.assertEqual(button.accessible_name, "收起处理参数设置")

    def test_processing_runtime_configuration_maps_safe_ui_settings(self):
        proxy = SimpleNamespace(
            output_formats=("tif", "png"),
            review_only=True,
            color_calibration="pcc",
            filter_hint="seestar_lp",
            denoise_mode="auto",
            deconvolution_mode="rl",
            graxpert_model_path="",
            compute_mode="cpu",
            pcc_timeout_sec=45,
            local_wb_gain_limit=1.18,
            builtin_denoise_strength=0.35,
            graxpert_deconv_strength=0.26,
            rl_iterations=12,
            rl_maxstars=320,
            starless_retry_max=3,
            starless_repair_strength=0.60,
            starless_halo_repair_strength=0.65,
            starless_chroma_strength=0.45,
            starmask_asinh_stretch=2.40,
            weak_star_recovery_ratio=0.80,
        )

        overrides, unset_keys = (
            gui_module.SeestarGui._processing_runtime_configuration(
                proxy,
                gui_module.INPUT_MODE_AUTO,
            )
        )

        self.assertEqual(overrides["SEESTAR_OUTPUT_FORMAT"], "tif,png")
        self.assertEqual(overrides["SEESTAR_FORCE_REVIEW_ONLY_OUTPUT"], "1")
        self.assertEqual(
            overrides["SEESTAR_STAGE4_FILTER_HINT"],
            "broadband Seestar LP",
        )
        self.assertEqual(overrides["SEESTAR_STAGE5_DECONV_ENABLE"], "1")
        self.assertEqual(
            overrides["SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE"],
            "0",
        )
        self.assertEqual(overrides["SEESTAR_SYQON_GPU"], "0")
        self.assertEqual(overrides["SEESTAR_STAGE4_PCC_TIMEOUT_SEC"], "45")
        self.assertEqual(
            overrides["SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT"], "1.18"
        )
        self.assertEqual(overrides["SEESTAR_STAGE5_BUILTIN_DENOISE_MOD"], "0.35")
        self.assertEqual(overrides["SEESTAR_STAGE5_RL_ITERS"], "12")
        self.assertEqual(overrides["SEESTAR_STAGE5_RL_MAXSTARS"], "320")
        self.assertEqual(overrides["SEESTAR_STAGE7_QUALITY_RETRY_MAX"], "3")
        self.assertEqual(
            overrides["SEESTAR_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN"], "0.80"
        )
        self.assertIn("SEESTAR_DENOISE_FORCE", unset_keys)
        self.assertIn("SEESTAR_GRAXPERT_OBJECT_MODEL_PATH", unset_keys)

    def test_processing_runtime_configuration_omits_completed_linear_stages(self):
        proxy = SimpleNamespace(
            output_formats=("fit",),
            review_only=False,
            color_calibration="pcc",
            filter_hint="no_filter",
            denoise_mode="on",
            deconvolution_mode="off",
            graxpert_model_path="/tmp/model.onnx",
            compute_mode="auto",
            pcc_timeout_sec=30,
            local_wb_gain_limit=1.20,
            builtin_denoise_strength=0.50,
            graxpert_deconv_strength=0.30,
            rl_iterations=8,
            rl_maxstars=200,
            starless_retry_max=2,
            starless_repair_strength=0.68,
            starless_halo_repair_strength=0.70,
            starless_chroma_strength=0.55,
            starmask_asinh_stretch=2.00,
            weak_star_recovery_ratio=0.70,
        )

        overrides, unset_keys = (
            gui_module.SeestarGui._processing_runtime_configuration(
                proxy,
                gui_module.INPUT_MODE_LINEAR_RESUME,
            )
        )

        self.assertEqual(overrides["SEESTAR_OUTPUT_FORMAT"], "fit")
        self.assertNotIn("SEESTAR_STAGE4_FILTER_HINT", overrides)
        self.assertNotIn("SEESTAR_STAGE4_PCC_TIMEOUT_SEC", overrides)
        self.assertNotIn("SEESTAR_STAGE5_RL_ITERS", overrides)
        self.assertNotIn("SEESTAR_DENOISE_FORCE", overrides)
        self.assertNotIn("SEESTAR_STAGE5_DECONV_ENABLE", overrides)
        self.assertEqual(overrides["SEESTAR_STAGE7_QUALITY_RETRY_MAX"], "2")
        self.assertEqual(
            overrides["SEESTAR_STAGE9_STARMASK_ASINH_STRETCH"], "2.00"
        )
        self.assertFalse(unset_keys)

    def test_processing_professional_settings_are_safely_clamped(self):
        proxy = SimpleNamespace(_sync_processing_controls_from_state=lambda: None)

        gui_module.SeestarGui._restore_processing_settings(
            proxy,
            {
                "pcc_timeout_sec": 999,
                "local_wb_gain_limit": 9.0,
                "builtin_denoise_strength": -1.0,
                "graxpert_deconv_strength": "invalid",
                "rl_iterations": 99,
                "rl_maxstars": 1,
                "starless_retry_max": 9,
                "starless_repair_strength": 2.0,
                "starless_halo_repair_strength": -1.0,
                "starless_chroma_strength": 2.0,
                "starmask_asinh_stretch": 9.0,
                "weak_star_recovery_ratio": 0.1,
            },
        )

        self.assertEqual(proxy.pcc_timeout_sec, 120)
        self.assertEqual(proxy.local_wb_gain_limit, 1.50)
        self.assertEqual(proxy.builtin_denoise_strength, 0.20)
        self.assertEqual(proxy.graxpert_deconv_strength, 0.30)
        self.assertEqual(proxy.rl_iterations, 40)
        self.assertEqual(proxy.rl_maxstars, 20)
        self.assertEqual(proxy.starless_retry_max, 3)
        self.assertEqual(proxy.starless_repair_strength, 0.85)
        self.assertEqual(proxy.starless_halo_repair_strength, 0.0)
        self.assertEqual(proxy.starless_chroma_strength, 0.90)
        self.assertEqual(proxy.starmask_asinh_stretch, 3.00)
        self.assertEqual(proxy.weak_star_recovery_ratio, 0.40)

    def test_pipeline_worker_runtime_overrides_can_unset_env_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
                runtime_overrides={"SEESTAR_OUTPUT_FORMAT": "tif,png"},
                runtime_unset_keys={"SEESTAR_DENOISE_FORCE"},
            )

            with patch.dict(
                os.environ,
                {"SEESTAR_DENOISE_FORCE": "1"},
                clear=False,
            ):
                env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(env["SEESTAR_OUTPUT_FORMAT"], "tif,png")
            self.assertNotIn("SEESTAR_DENOISE_FORCE", env)

    def test_pipeline_worker_prefers_bundled_graxpert_object_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plugin_dir = root / "plugins"
            bundled_model = (
                plugin_dir
                / "deconvolution-object-ai-models"
                / "1.2.3"
                / "model.onnx"
            )
            bundled_model.parent.mkdir(parents=True)
            bundled_model.write_bytes(b"bundled-model")
            user_model = root / "user" / "9.9.9" / "model.onnx"
            user_model.parent.mkdir(parents=True)
            user_model.write_bytes(b"user-model")
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=plugin_dir,
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
                runtime_overrides={
                    "SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE": "1",
                    "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": str(user_model),
                },
            )

            env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(
                env["SEESTAR_GRAXPERT_OBJECT_MODEL_PATH"],
                str(bundled_model),
            )

    def test_default_graxpert_model_uses_installed_app_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            app_model = (
                root
                / "home"
                / "Library"
                / "Application Support"
                / "GraXpert"
                / "GraXpert"
                / "deconvolution-object-ai-models"
                / "1.0.1"
                / "model.onnx"
            )
            app_model.parent.mkdir(parents=True)
            app_model.write_bytes(b"graxpert-app-model")

            selected, source = pipeline_worker_module.default_graxpert_object_model(
                root / "plugins",
                user_home=root / "home",
            )

            self.assertEqual(selected, app_model)
            self.assertEqual(source, "graxpert_app")

    def test_result_preview_uses_newest_known_output_across_name_families(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            stale = work_dir / "result_processed.png"
            current = work_dir / "M_42_60sec_20260216_140234_processed.png"
            stale.write_bytes(b"stale")
            current.write_bytes(b"current")
            os.utime(stale, (100.0, 100.0))
            os.utime(current, (200.0, 200.0))

            selected = gui_module.SeestarGui._find_result_preview(None, work_dir)

            self.assertEqual(selected, current)

    def test_directory_recommendation_prefers_latest_resume_point(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "Light_001.fit").write_bytes(b"fits")
            (work_dir / gui_module.LINEAR_RESUME_INPUT_NAME).write_bytes(b"fits")
            proxy = SimpleNamespace(
                _fits_in_work_dir=lambda wd: list(wd.glob("*.fit")),
                _linear_resume_input_path=lambda wd: wd / gui_module.LINEAR_RESUME_INPUT_NAME,
                _stage2_corrected_resume_input_path=lambda wd: wd / gui_module.STAGE2_CORRECTED_INPUT_NAME,
                _is_candidate_stacked_input=lambda _path, _wd: False,
            )

            mode, detected = gui_module.SeestarGui._directory_recommendation(
                proxy,
                work_dir,
            )

            self.assertEqual(mode, gui_module.INPUT_MODE_LINEAR_RESUME)
            self.assertIn("线性处理进度", detected)

    def test_directory_recommendation_counts_light_frames(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            for index in range(3):
                (work_dir / f"Light_{index:03d}.fit").write_bytes(b"fits")
            proxy = SimpleNamespace(
                _fits_in_work_dir=lambda wd: list(wd.glob("*.fit")),
                _linear_resume_input_path=lambda wd: wd / gui_module.LINEAR_RESUME_INPUT_NAME,
                _stage2_corrected_resume_input_path=lambda wd: wd / gui_module.STAGE2_CORRECTED_INPUT_NAME,
                _is_candidate_stacked_input=lambda _path, _wd: False,
            )

            mode, detected = gui_module.SeestarGui._directory_recommendation(
                proxy,
                work_dir,
            )

            self.assertEqual(mode, gui_module.INPUT_MODE_AUTO)
            self.assertEqual(detected, "3 个 Light FITS")

    def test_quality_report_path_prefers_process_report(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            report = work_dir / "process" / "final_quality_report.json"
            report.parent.mkdir()
            report.write_text("{}\n", encoding="utf-8")

            selected = gui_module.SeestarGui._quality_report_path(
                SimpleNamespace(),
                work_dir,
            )

            self.assertEqual(selected, report)

    def test_bootstrap_worker_emits_cancelled_when_stopped(self):
        cancelled = []
        succeeded = []
        worker = gui_module.BootstrapWorker(
            lambda _stop_event, _progress: {"ready": True}
        )
        worker.cancelled = SimpleNamespace(emit=lambda: cancelled.append(True))
        worker.succeeded = SimpleNamespace(emit=lambda value: succeeded.append(value))

        worker.stop()
        worker.run()

        self.assertEqual(cancelled, [True])
        self.assertEqual(succeeded, [])

    def test_bootstrap_subprocess_can_be_cancelled(self):
        stop_event = threading.Event()
        timer = threading.Timer(0.1, stop_event.set)
        started_at = time.monotonic()
        timer.start()
        try:
            with self.assertRaises(bootstrap_module.BootstrapCancelled):
                bootstrap_module.run_cancellable_process(
                    ["/bin/sh", "-c", "sleep 30 & wait"],
                    stop_event=stop_event,
                    capture_output=True,
                    text=True,
                )
        finally:
            timer.cancel()

        self.assertLess(time.monotonic() - started_at, 5)

    def test_bootstrap_fingerprint_changes_with_dependency_lock(self):
        with tempfile.TemporaryDirectory() as td:
            plugin_dir = Path(td)
            lock_path = plugin_dir / "requirements.lock"
            lock_path.write_text("numpy==1\n", encoding="utf-8")
            proxy = SimpleNamespace(
                siril_plugin_dir=plugin_dir,
                _plugin_requirements_path=lambda: plugin_dir / "requirements.txt",
                _bootstrap_app_version=lambda: "1.2.3",
            )

            first = gui_module.SeestarGui._bootstrap_fingerprint(proxy)
            lock_path.write_text("numpy==2\n", encoding="utf-8")
            second = gui_module.SeestarGui._bootstrap_fingerprint(proxy)

            self.assertEqual(first["python_abi"], "cp312")
            self.assertEqual(first["app_version"], "1.2.3")
            self.assertNotEqual(
                first["dependency_lock_sha256"],
                second["dependency_lock_sha256"],
            )

    def test_bootstrap_runtime_skips_installs_when_fingerprint_matches(self):
        calls = []
        progress = []
        fingerprint = {
            "schema": 1,
            "python_abi": "cp312",
            "app_version": "1.2.3",
            "dependency_lock_sha256": "abc",
        }
        proxy = SimpleNamespace(
            _bootstrap_stop_event=None,
            _check_bootstrap_cancelled=lambda: None,
            _estimate_disk_space=lambda _work_dir, *, input_mode=None: calls.append("work_disk") or None,
            _estimate_runtime_disk_space=lambda _fingerprint: calls.append("runtime_disk") or SimpleNamespace(
                    volume_path=Path("/tmp"),
                    volume_device=-1,
                    current_runtime_bytes=0,
                    seed_growth_bytes=0,
                    support_growth_bytes=0,
                    dependency_growth_bytes=0,
                    required_free_bytes=0,
                    available_bytes=1,
                    bootstrap_cache_hit=True,
                ),
            _ensure_offline_siril_python_seed=lambda: calls.append("seed"),
            _ensure_siril_plugins_ready=lambda: calls.append("plugins") or True,
            _ensure_runtime_spcc_database_seed=lambda: calls.append("spcc"),
            _ensure_runtime_siril_support_dirs=lambda: calls.append("support"),
            _bootstrap_fingerprint=lambda: fingerprint,
            _bootstrap_state_is_current=lambda value: value == fingerprint,
            _append_event=lambda message: calls.append(str(message)),
            _disk_space_error_message=lambda _estimate: "disk error",
            _runtime_disk_space_error_message=lambda _estimate: "runtime disk error",
            _runtime_disk_space_summary_lines=lambda _estimate: [],
            _ensure_runtime_requirements_ready=lambda: calls.append("install"),
        )

        result = gui_module.SeestarGui._bootstrap_runtime(
            proxy,
            Path("/tmp/work"),
            gui_module.INPUT_MODE_AUTO,
            threading.Event(),
            progress.append,
        )

        self.assertTrue(result["bootstrap_cache_hit"])
        self.assertNotIn("install", calls)
        self.assertLess(calls.index("work_disk"), calls.index("seed"))
        self.assertLess(calls.index("runtime_disk"), calls.index("seed"))
        self.assertTrue(any("跳过重复安装" in item for item in progress))

    def test_append_text_holds_log_lock_while_writing(self):
        lock = threading.Lock()

        class LogFile:
            def __init__(self):
                self.writes = []
                self.flush_count = 0

            def write(self, text):
                self.assert_locked()
                self.writes.append(text)

            def flush(self):
                self.assert_locked()
                self.flush_count += 1

            @staticmethod
            def assert_locked():
                if not lock.locked():
                    raise AssertionError("run log lock is not held")

        log_file = LogFile()
        log_view = SimpleNamespace(
            moveCursor=lambda *_args: None,
            insertPlainText=lambda *_args: None,
        )
        proxy = SimpleNamespace(
            log_view=log_view,
            run_log_file=log_file,
            _run_log_lock=lock,
        )
        append_text = gui_module.SeestarGui._append_text
        method_globals = append_text.__globals__
        original_cursor = method_globals["QTextCursor"]
        method_globals["QTextCursor"] = SimpleNamespace(
            MoveOperation=SimpleNamespace(End=object())
        )
        try:
            append_text(proxy, "worker output\n")
        finally:
            method_globals["QTextCursor"] = original_cursor

        self.assertEqual(log_file.writes, ["worker output\n"])
        self.assertEqual(log_file.flush_count, 1)

    def test_close_event_forces_termination_after_wait_timeout(self):
        calls = []

        class Worker:
            def isRunning(self):
                return True

            def wait(self, timeout=None):
                calls.append(("wait", timeout))
                return timeout is None

            def terminate(self):
                calls.append(("terminate", None))

        class Event:
            accepted = False
            ignored = False

            def accept(self):
                self.accepted = True

            def ignore(self):
                self.ignored = True

        worker = Worker()
        proxy = SimpleNamespace(
            worker=worker,
            _stop_run=lambda: (_ for _ in ()).throw(RuntimeError("stop failed")),
            _append_event=lambda _message: (_ for _ in ()).throw(OSError("log failed")),
        )
        event = Event()
        close_event = gui_module.SeestarGui.closeEvent
        method_globals = close_event.__globals__
        original_message_box = method_globals["QMessageBox"]
        yes = 1
        method_globals["QMessageBox"] = SimpleNamespace(
            StandardButton=SimpleNamespace(Yes=yes, No=2),
            question=lambda *_args, **_kwargs: yes,
        )
        try:
            close_event(proxy, event)
        finally:
            method_globals["QMessageBox"] = original_message_box

        self.assertEqual(
            calls,
            [("wait", 8000), ("terminate", None), ("wait", None)],
        )
        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)

    def _make_gui_proxy(self, work_dir: Path, *, input_mode: str):
        resources = work_dir / "resources"
        resources.mkdir(exist_ok=True)
        pipeline_path = work_dir / "pipeline" / "seestar_Superimpose.py"
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_path.write_text("# mock pipeline\n", encoding="utf-8")
        stage11_path = pipeline_path.with_name("stage11_ai_postprocess.py")
        stage11_path.write_text("# mock stage11\n", encoding="utf-8")
        config_template = work_dir / "resources" / "config.1.4.ini.template"
        config_template.parent.mkdir(parents=True, exist_ok=True)
        config_template.write_text("[core]\n", encoding="utf-8")
        siril_cli = work_dir / "resources" / "siril-cli"
        siril_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        siril_cli.chmod(0o755)
        plugin_dir = work_dir / "resources" / "siril_plugins"
        plugin_dir.mkdir(parents=True, exist_ok=True)

        proxy = SimpleNamespace(
            pipeline_path=pipeline_path,
            config_template=config_template,
            siril_plugin_dir=plugin_dir,
            _resolve_siril_candidates=lambda: [siril_cli],
            _display_path=lambda path: str(path),
            _current_input_mode=lambda: input_mode,
            _linear_resume_input_path=lambda wd: gui_module.SeestarGui._linear_resume_input_path(proxy, wd),
            _stage2_corrected_resume_input_path=lambda wd: gui_module.SeestarGui._stage2_corrected_resume_input_path(proxy, wd),
            _disk_space_mode_label=lambda estimate: gui_module.SeestarGui._disk_space_mode_label(proxy, estimate),
            _disk_space_summary_lines=lambda estimate: gui_module.SeestarGui._disk_space_summary_lines(proxy, estimate),
        )
        proxy._fits_in_work_dir = lambda wd: gui_module.SeestarGui._fits_in_work_dir(proxy, wd)
        proxy._is_candidate_stacked_input = (
            lambda path, wd: gui_module.SeestarGui._is_candidate_stacked_input(proxy, path, wd)
        )
        return proxy

    def test_build_siril_cli_command_respects_offline_mode(self):
        work_dir = Path("/tmp/work")
        run_ini = Path("/tmp/config.ini")
        run_ssf = Path("/tmp/run.ssf")
        cli = Path("/opt/siril-cli")

        offline_cmd = gui_module.build_siril_cli_command(
            siril_cli=cli,
            work_dir=work_dir,
            run_ini=run_ini,
            run_ssf=run_ssf,
            offline_mode=True,
        )
        self.assertIn("--offline", offline_cmd)

        online_cmd = gui_module.build_siril_cli_command(
            siril_cli=cli,
            work_dir=work_dir,
            run_ini=run_ini,
            run_ssf=run_ssf,
            offline_mode=False,
        )
        self.assertNotIn("--offline", online_cmd)

    def test_resolve_siril_scripts_root_accepts_direct_and_nested_layout(self):
        with tempfile.TemporaryDirectory() as td:
            plugin_root = Path(td)
            direct = plugin_root / "vendor" / "siril-scripts"
            nested = direct / "siril-scripts"

            (direct / "processing").mkdir(parents=True, exist_ok=True)
            (direct / "processing" / "AberrationRemover.py").write_text("x", encoding="utf-8")
            resolved_direct = gui_module.resolve_siril_scripts_root(plugin_root)
            self.assertEqual(resolved_direct, direct)

            (direct / "processing" / "AberrationRemover.py").unlink()
            (nested / "processing").mkdir(parents=True, exist_ok=True)
            (nested / "processing" / "AberrationRemover.py").write_text("x", encoding="utf-8")
            resolved_nested = gui_module.resolve_siril_scripts_root(plugin_root)
            self.assertEqual(resolved_nested, nested)

    def test_missing_plugin_artifacts_requires_runtime_wheels(self):
        with tempfile.TemporaryDirectory() as td:
            plugin_root = Path(td)
            (plugin_root / "downloads").mkdir(parents=True, exist_ok=True)
            (plugin_root / "vendor" / "siril-scripts" / "processing").mkdir(
                parents=True, exist_ok=True
            )
            (plugin_root / "download_siril_plugins.sh").write_text("#!/bin/bash\n", encoding="utf-8")
            (plugin_root / "downloads" / "setiastrosuitepro-1.0-py3-none-any.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "vendor" / "siril-scripts" / "processing" / "AberrationRemover.py").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "vendor" / "siril-scripts" / "processing" / "SyQon-Starless.py").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "syqon_starless").mkdir(parents=True, exist_ok=True)
            (plugin_root / "syqon_starless" / "syqon_starless_inference.py").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "syqon_starless" / "zenith.pt").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "cosmic_clarity").mkdir(parents=True, exist_ok=True)
            for name in (
                "deep_denoise_mono_AI4.pth",
                "deep_denoise_color_AI4.pth",
                "deep_sharp_stellar_AI4.pth",
                "deep_nonstellar_sharp_conditional_psf_AI4.pth",
            ):
                (plugin_root / "cosmic_clarity" / name).write_text("x", encoding="utf-8")

            proxy = SimpleNamespace(siril_plugin_dir=plugin_root)
            proxy._plugin_download_script_path = (
                lambda: gui_module.SeestarGui._plugin_download_script_path(proxy)
            )
            proxy._plugin_downloads_dir = (
                lambda: gui_module.SeestarGui._plugin_downloads_dir(proxy)
            )
            proxy._onnxruntime_wheels = (
                lambda: gui_module.SeestarGui._onnxruntime_wheels(proxy)
            )
            proxy._onnx_wheels = (
                lambda: gui_module.SeestarGui._onnx_wheels(proxy)
            )
            proxy._pyqt6_wheels = (
                lambda: gui_module.SeestarGui._pyqt6_wheels(proxy)
            )
            proxy._pyqt6_qt6_wheels = (
                lambda: gui_module.SeestarGui._pyqt6_qt6_wheels(proxy)
            )
            proxy._pyqt6_sip_wheels = (
                lambda: gui_module.SeestarGui._pyqt6_sip_wheels(proxy)
            )
            proxy._pyside6_wheels = (
                lambda: gui_module.SeestarGui._pyside6_wheels(proxy)
            )
            proxy._pyside6_addons_wheels = (
                lambda: gui_module.SeestarGui._pyside6_addons_wheels(proxy)
            )
            proxy._pyside6_essentials_wheels = (
                lambda: gui_module.SeestarGui._pyside6_essentials_wheels(proxy)
            )
            proxy._shiboken6_wheels = (
                lambda: gui_module.SeestarGui._shiboken6_wheels(proxy)
            )
            proxy._astropy_wheels = (
                lambda: gui_module.SeestarGui._astropy_wheels(proxy)
            )
            proxy._scipy_wheels = (
                lambda: gui_module.SeestarGui._scipy_wheels(proxy)
            )
            proxy._tifffile_wheels = (
                lambda: gui_module.SeestarGui._tifffile_wheels(proxy)
            )
            proxy._sep_wheels = (
                lambda: gui_module.SeestarGui._sep_wheels(proxy)
            )
            proxy._spandrel_wheels = (
                lambda: gui_module.SeestarGui._spandrel_wheels(proxy)
            )
            proxy._einops_wheels = (
                lambda: gui_module.SeestarGui._einops_wheels(proxy)
            )
            proxy._safetensors_wheels = (
                lambda: gui_module.SeestarGui._safetensors_wheels(proxy)
            )
            proxy._torch_wheels = (
                lambda: gui_module.SeestarGui._torch_wheels(proxy)
            )
            proxy._torchvision_wheels = (
                lambda: gui_module.SeestarGui._torchvision_wheels(proxy)
            )

            missing = gui_module.SeestarGui._missing_plugin_artifacts(proxy)
            self.assertIn("onnx wheel 缺失", missing)
            self.assertIn("onnxruntime wheel 缺失", missing)
            self.assertIn("PyQt6 wheel 缺失", missing)
            self.assertIn("PyQt6_Qt6 wheel 缺失", missing)
            self.assertIn("pyqt6_sip wheel 缺失", missing)
            self.assertIn("tifffile wheel 缺失", missing)
            self.assertIn("PySide6 wheel 缺失", missing)
            self.assertIn("PySide6_Addons wheel 缺失", missing)
            self.assertIn("PySide6_Essentials wheel 缺失", missing)
            self.assertIn("shiboken6 wheel 缺失", missing)
            self.assertIn("astropy wheel 缺失", missing)
            self.assertIn("scipy wheel 缺失", missing)
            self.assertIn("sep wheel 缺失", missing)
            self.assertIn("spandrel wheel 缺失", missing)
            self.assertIn("einops wheel 缺失", missing)
            self.assertIn("safetensors wheel 缺失", missing)

            (plugin_root / "downloads" / "onnxruntime-1.19.2-cp312.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "onnx-1.22.0-cp312.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "pyqt6-6.8.1-cp39-abi3-macosx_11_0_universal2.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "pyqt6_qt6-6.8.2-py3-none-macosx_11_0_universal2.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "pyqt6_sip-13.10.2-cp312-cp312-macosx_11_0_universal2.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "tifffile-2025.3.30-py3-none-any.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "pyside6-6.8.2-cp39-abi3-macosx_11_0_universal2.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "pyside6_addons-6.8.2-cp39-abi3-macosx_11_0_universal2.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "pyside6_essentials-6.8.2-cp39-abi3-macosx_11_0_universal2.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "shiboken6-6.8.2-cp39-abi3-macosx_11_0_universal2.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "astropy-7.1.0-cp312-cp312-macosx_11_0_arm64.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "scipy-1.14.1-cp312-cp312-macosx_11_0_arm64.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "sep-1.4.1-cp312-cp312-macosx_11_0_arm64.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "spandrel-0.4.2-py3-none-any.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "einops-0.8.2-py3-none-any.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "safetensors-0.7.0-cp38-abi3-macosx_11_0_arm64.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "torch-2.11.0-cp312-cp312-macosx_11_0_arm64.whl").write_text(
                "x", encoding="utf-8"
            )
            (plugin_root / "downloads" / "torchvision-0.26.0-cp312-cp312-macosx_11_0_arm64.whl").write_text(
                "x", encoding="utf-8"
            )
            missing_after = gui_module.SeestarGui._missing_plugin_artifacts(proxy)
            self.assertEqual(missing_after, [])

    def test_siril_plugin_wheel_abi_is_limited_to_cp312(self):
        compatible = (
            "numpy-2.4.6-cp312-cp312-macosx_14_0_arm64.whl",
            "astropy-7.2.0-cp311-abi3-macosx_11_0_arm64.whl",
            "opencv_python_headless-4.13.0-cp37-abi3-macosx_13_0_arm64.whl",
            "tifffile-2026.5.15-py3-none-any.whl",
        )
        incompatible = (
            "numpy-2.4.6-cp313-cp313-macosx_14_0_arm64.whl",
            "future_abi-1.0-cp313-abi3-macosx_14_0_arm64.whl",
            "numpy-2.4.6-cp311-cp311-macosx_14_0_arm64.whl",
        )

        for filename in compatible:
            self.assertTrue(
                gui_module.is_siril_cp312_wheel_compatible(Path(filename)),
                filename,
            )
        for filename in incompatible:
            self.assertFalse(
                gui_module.is_siril_cp312_wheel_compatible(Path(filename)),
                filename,
            )

    def test_timeout_patch_overrides_sirilpy_method_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            runtime_home = Path(td) / "runtime_home"
            state_root = gui_module.siril_state_root_from_home(runtime_home)
            site_dir = state_root / "venv" / "lib" / "python3.12" / "site-packages"
            site_dir.mkdir(parents=True, exist_ok=True)
            events: list[str] = []

            proxy = SimpleNamespace(
                _siril_state_root=lambda: state_root,
                _append_event=lambda msg: events.append(str(msg)),
                _display_path=lambda path: str(path),
            )

            gui_module.SeestarGui._ensure_runtime_sirilpy_timeout_patch(proxy)
            patch_path = site_dir / "sitecustomize.py"
            self.assertTrue(patch_path.is_file())
            patch_text = patch_path.read_text(encoding="utf-8")
            self.assertIn("def _patch_default_timeout", patch_text)
            self.assertIn("_recv_exact", patch_text)
            self.assertIn("_request_data", patch_text)
            self.assertTrue(any("timeout 补丁" in msg for msg in events))

    def test_opencv_headless_distribution_alias_satisfies_opencv_python_check(self):
        with tempfile.TemporaryDirectory() as td:
            runtime_home = Path(td) / "runtime_home"
            state_root = gui_module.siril_state_root_from_home(runtime_home)
            site_dir = state_root / "venv" / "lib" / "python3.12" / "site-packages"
            (site_dir / "cv2").mkdir(parents=True, exist_ok=True)
            headless_dist = site_dir / "opencv_python_headless-4.13.0.92.dist-info"
            headless_dist.mkdir(parents=True, exist_ok=True)
            (headless_dist / "METADATA").write_text(
                "Metadata-Version: 2.1\n"
                "Name: opencv-python-headless\n"
                "Version: 4.13.0.92\n",
                encoding="utf-8",
            )
            events: list[str] = []

            proxy = SimpleNamespace(
                _siril_state_root=lambda: state_root,
                _append_event=lambda msg: events.append(str(msg)),
                _display_path=lambda path: str(path),
            )

            gui_module.SeestarGui._ensure_runtime_opencv_distribution_alias(proxy)

            alias_metadata = (
                site_dir / "opencv_python-4.13.0.92.dist-info" / "METADATA"
            ).read_text(encoding="utf-8")
            self.assertIn("Name: opencv-python\n", alias_metadata)
            self.assertIn("Version: 4.13.0.92\n", alias_metadata)
            distributions = {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions(path=[str(site_dir)])
            }
            self.assertEqual(distributions["opencv-python"], "4.13.0.92")
            self.assertTrue(
                any("opencv-python 兼容分发元数据" in msg for msg in events)
            )

    def test_pipeline_worker_build_env_includes_input_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
                input_mode=gui_module.INPUT_MODE_LINEAR_RESUME,
            )

            contaminated_qt_env = {
                "QT_PLUGIN_PATH": "/app/PySide6/Qt/plugins",
                "QT_QPA_PLATFORM_PLUGIN_PATH": "/app/PySide6/Qt/plugins/platforms",
                "QML2_IMPORT_PATH": "/app/PySide6/Qt/qml",
                "QT_QPA_PLATFORM": "cocoa",
            }
            with patch.dict(os.environ, contaminated_qt_env):
                env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(
                env.get("SEESTAR_INPUT_MODE"),
                gui_module.INPUT_MODE_LINEAR_RESUME,
            )
            self.assertEqual(env.get("PYTHONUNBUFFERED"), "1")
            self.assertEqual(env.get("QT_QPA_PLATFORM"), "offscreen")
            self.assertNotIn("QT_PLUGIN_PATH", env)
            self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", env)
            self.assertNotIn("QML2_IMPORT_PATH", env)
            self.assertEqual(env.get("SEESTAR_BOOTSTRAP_TIMEOUT_SEC"), "300")
            self.assertEqual(env.get("SEESTAR_WATCHDOG_IDLE_TIMEOUT_SEC"), "900")
            self.assertEqual(env.get("SEESTAR_EXPORT_TAIL_TIMEOUT_SEC"), "120")
            self.assertEqual(env.get("SEESTAR_TEMP_CLEANUP_TIMEOUT_SEC"), "30")
            self.assertEqual(env.get("SEESTAR_NETWORK_MODE"), "1")
            self.assertNotIn("SEESTAR_SPCC_ENABLE", env)
            self.assertNotIn("SEESTAR_GAIA_PHOTO_CATALOG", env)

    def test_pipeline_worker_uses_keychain_runtime_override_not_env_file_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resources = root / "resources"
            resources.mkdir()
            (resources / "ai.env").write_text(
                "SEESTAR_AI_ENDPOINT=https://bundled.example/v1\n"
                "SEESTAR_AI_MODEL=bundled-model\n"
                "SEESTAR_AI_API_KEY=plaintext-file-key\n",
                encoding="utf-8",
            )
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=resources,
                runtime_home=root / "runtime_home",
                siril_candidates=[],
                ai_stage_enabled=True,
                ai_runtime_overrides={
                    "SEESTAR_AI_ENDPOINT": "https://custom.example/v1",
                    "SEESTAR_AI_MODEL": "custom-model",
                    "SEESTAR_AI_API_KEY": "keychain-runtime-key",
                },
            )

            with patch.dict(
                os.environ,
                {"SEESTAR_AI_API_KEY": "parent-process-key"},
                clear=False,
            ):
                env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(env["SEESTAR_AI_ENDPOINT"], "https://custom.example/v1")
            self.assertEqual(env["SEESTAR_AI_MODEL"], "custom-model")
            self.assertEqual(env["SEESTAR_AI_API_KEY"], "keychain-runtime-key")
            self.assertNotIn("plaintext-file-key", env.values())
            self.assertNotIn("parent-process-key", env.values())
            self.assertNotIn("SEESTAR_GAIA_PHOTO_CATALOG", env)
            self.assertEqual(
                env.get("SEESTAR_GAIA_ASTRO_CATALOG"),
                str(
                    root
                    / "runtime_home"
                    / ".local"
                    / "share"
                    / "siril"
                    / "siril_cat_healpix8_astro.dat"
                ),
            )
            self.assertNotIn("SEESTAR_SPCC_DATABASE_DIR", env)
            self.assertEqual(worker._temp_cleanup_timeout_sec, 30)
            self.assertEqual(worker._watchdog_idle_timeout_sec, 900)
            self.assertEqual(worker._export_tail_timeout_sec, 120)

    def test_pipeline_worker_bootstrap_timeout_uses_env_and_fits_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            work_dir.mkdir()
            large_fit = work_dir / "large_input.fit"
            with large_fit.open("wb") as stream:
                stream.truncate(2 * 1024 * 1024 * 1024)
            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            effective, base, input_bytes = worker._bootstrap_timeout_sec(
                {"SEESTAR_BOOTSTRAP_TIMEOUT_SEC": "600"}
            )

            self.assertEqual(base, 600)
            self.assertEqual(input_bytes, 2 * 1024 * 1024 * 1024)
            self.assertEqual(effective, 840)

    def test_pipeline_worker_bootstrap_timeout_invalid_value_uses_default_and_caps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            work_dir.mkdir()
            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            events = []
            worker.log = SimpleNamespace(emit=lambda text: events.append(str(text)))

            effective, base, input_bytes = worker._bootstrap_timeout_sec(
                {"SEESTAR_BOOTSTRAP_TIMEOUT_SEC": "invalid"}
            )
            capped, capped_base, _ = worker._bootstrap_timeout_sec(
                {"SEESTAR_BOOTSTRAP_TIMEOUT_SEC": "99999"}
            )

            self.assertEqual((effective, base, input_bytes), (300, 300, 0))
            self.assertEqual((capped, capped_base), (3600, 3600))
            self.assertTrue(any("使用默认值 300s" in event for event in events))

    def test_pipeline_worker_tracks_last_command_and_saving_png(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            worker._inspect_output_for_errors(
                'input command:savepng "M42_processed"'
            )

            self.assertIn("savepng", worker._last_command)
            self.assertIsNotNone(worker._saving_png_seen_at)

    def test_pipeline_worker_compacts_repetitive_progress_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            emitted = []
            worker.log = SimpleNamespace(emit=emitted.append)
            worker._current_pipeline_stage = 5

            for percent in (0.0, 1.56, 9.38, 10.94, 12.50, 20.31, 100.0):
                worker._emit_process_output(
                    f"progress: NL-Bayes denoising..., {percent:.2f}%\n"
                )
            worker._emit_process_output(
                "progress: NL-Bayes denoising..., 0.00%\n"
            )
            worker._emit_process_output("log: final denoise complete\n")

            self.assertEqual(len(emitted), 6)
            self.assertIn("0.00%", emitted[0])
            self.assertIn("10.94%", emitted[1])
            self.assertIn("20.31%", emitted[2])
            self.assertIn("100.00%", emitted[3])
            self.assertIn("0.00%", emitted[4])
            self.assertEqual(emitted[5], "log: final denoise complete\n")

    def test_pipeline_worker_compacts_plugin_progress_by_phase(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            emitted = []
            worker.log = SimpleNamespace(emit=emitted.append)
            worker._current_pipeline_stage = 10

            for line in (
                "log: [INFO] [CosmicClarity_Native.py] [####----------------] 21% Denoising luma\n",
                "log: [INFO] [CosmicClarity_Native.py] [####----------------] 22% Denoising luma\n",
                "log: [INFO] [CosmicClarity_Native.py] [######--------------] 30% Denoising luma\n",
                "log: [INFO] [CosmicClarity_Native.py] [##########----------] 51% Denoising colour\n",
                "log: [INFO] [CosmicClarity_Native.py] [##########----------] 52% Denoising colour\n",
                "log: [INFO] [CosmicClarity_Native.py] [############--------] 60% Denoising colour\n",
            ):
                worker._emit_process_output(line)

            self.assertEqual(len(emitted), 4)
            self.assertIn("21%", emitted[0])
            self.assertIn("30%", emitted[1])
            self.assertIn("51%", emitted[2])
            self.assertIn("60%", emitted[3])

    def test_pipeline_worker_compacts_known_benign_siril_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            emitted = []
            worker.log = SimpleNamespace(emit=emitted.append)
            worker._current_pipeline_stage = 5
            worker._last_command = "log: Running command: denoise"

            lines = (
                "Skipping HDU 2 with EXTNAME=ICCProfile\n",
                "Skipping HDU 2 with EXTNAME=ICCProfile\n",
                "error: no suitable data in src fits\n",
            )
            for line in lines:
                worker._inspect_output_for_errors(line)
                worker._emit_process_output(line)

            self.assertEqual(len(emitted), 2)
            self.assertIn("ICCProfile", emitted[0])
            self.assertIn("Stage 5", emitted[1])
            self.assertEqual(list(worker._recent_process_output), [])

    def test_pipeline_worker_resolves_user_graxpert_model_path_before_home_isolated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            work_dir.mkdir()
            model_dir = work_dir / "models" / "1.0.1"
            model_dir.mkdir(parents=True)
            (model_dir / "model.onnx").write_bytes(b"onnx")
            (work_dir / ".seestar_ai.env").write_text(
                "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH=models/1.0.1\n",
                encoding="utf-8",
            )
            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            worker.log = SimpleNamespace(emit=lambda _text: None)

            with patch.dict(
                os.environ,
                {"SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": ""},
                clear=False,
            ):
                env = worker._build_env(root / "siril-cli")

            self.assertEqual(
                env["SEESTAR_GRAXPERT_OBJECT_MODEL_PATH"],
                str(model_dir),
            )
            self.assertEqual(env["HOME"], str(root / "runtime_home"))

    def test_pipeline_worker_progress_does_not_hide_recent_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            worker._inspect_output_for_errors("log: useful diagnostic\n")
            for percent in range(80):
                worker._inspect_output_for_errors(
                    f"progress: inference, {percent:.2f}%\n"
                )

            self.assertEqual(list(worker._recent_process_output), ["log: useful diagnostic"])

    def test_pipeline_worker_emits_stage_progress_and_terminal_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            emitted = []
            worker.progress = SimpleNamespace(
                emit=lambda stage, title, state: emitted.append((stage, title, state))
            )

            worker._inspect_output_for_errors("[INFO] 阶段 4: 色彩校准")
            worker._inspect_output_for_errors("[DEBUG] 阶段 4: 继续校准")
            worker._inspect_output_for_errors(
                "[INFO] [PIPELINE_STAGE_RESULT] "
                "stage=4 status=degraded duration=12.7 title=图像解析 + 色彩校准"
            )
            worker._inspect_output_for_errors("[INFO] 阶段 4: 色彩校准重试")
            worker._inspect_output_for_errors("[INFO] 阶段 5: 线性降噪")

            self.assertEqual(
                emitted,
                [
                    (4, "色彩校准", "running"),
                    (4, "图像解析 + 色彩校准", "degraded"),
                    (4, "色彩校准重试", "running"),
                    (5, "线性降噪", "running"),
                ],
            )
            self.assertEqual(worker._pipeline_stage_durations[4], 12.7)

    def test_pipeline_worker_watchdog_uses_only_current_run_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stale = root / "old_result.png"
            stale.write_bytes(b"old")
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            worker._capture_artifact_snapshot()
            fresh = root / "new_result.png"
            fresh.write_bytes(b"new")

            generated = worker._generated_artifacts()

            self.assertIn(fresh, generated)
            self.assertNotIn(stale, generated)

    def test_pipeline_worker_export_tail_watchdog_precedes_idle_watchdog(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            worker._capture_artifact_snapshot()
            (root / "new_result.png").write_bytes(b"png")
            worker._saving_png_seen_at = 10.0
            worker._last_output_ts = 10.0
            worker._export_tail_timeout_sec = 60
            worker._watchdog_idle_timeout_sec = 900

            worker._refresh_export_tail_state(11.0)
            reason = worker._watchdog_timeout_reason(71.0)

            self.assertEqual(reason, ("export_tail", 60.0))

    def test_pipeline_worker_export_tail_timeout_returns_warning_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_path = root / "result.png"
            fake_cli = root / "fake-siril-cli"
            fake_cli.write_text(
                "#!/bin/sh\n"
                "printf 'Running command: pyscript mock.py\\n'\n"
                "printf '[INFO] Stage 1\\n'\n"
                "printf 'input command:savepng result\\n'\n"
                "printf 'png' > \"$TEST_OUTPUT\"\n"
                "printf 'Saving PNG: result.png\\n'\n"
                "sleep 30\n",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            events: list[str] = []
            worker.log = SimpleNamespace(emit=lambda text: events.append(str(text)))

            def build_env(_siril_cli):
                worker._watchdog_idle_timeout_sec = 10
                worker._export_tail_timeout_sec = 0.05
                env = os.environ.copy()
                env["TEST_OUTPUT"] = str(output_path)
                return env

            worker._build_env = build_env

            success, exit_code = worker._run_once(
                fake_cli,
                root / "run.ssf",
                root / "config.ini",
            )

            self.assertTrue(success)
            self.assertNotEqual(exit_code, 0)
            self.assertTrue(worker._export_tail_timeout_recovered)
            self.assertEqual(worker._completed_run_status(), "CompletedWithWarning")
            self.assertIn("导出成功、收尾异常", "".join(events))

    def test_pipeline_worker_stage11_disarms_short_export_tail_watchdog(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
                ai_stage_enabled=True,
            )
            worker._inspect_output_for_errors("Saving PNG: result.png")
            worker._inspect_output_for_errors("[INFO] 阶段 11: AI 后期美化")

            self.assertTrue(worker._export_tail_disarmed)

    def test_pipeline_worker_watchdog_diagnostics_include_required_context(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            events: list[str] = []
            worker.log = SimpleNamespace(emit=lambda text: events.append(str(text)))
            worker._capture_artifact_snapshot()
            (root / "result.png").write_bytes(b"png")
            worker._last_command = "input command:savepng result"
            worker._proc = SimpleNamespace(pid=123, poll=lambda: None)
            worker._proc_pgid = 123
            worker._process_group_snapshot = lambda: [
                "pid=123 ppid=1 pgid=123 stat=S etime=01:00 cpu=0.0% mem=0.1% cmd=siril-cli"
            ]

            worker._append_watchdog_diagnostics(
                reason="export_tail",
                idle_sec=120.0,
            )

            rendered = "".join(events)
            self.assertIn("Watchdog 最后命令", rendered)
            self.assertIn("savepng", rendered)
            self.assertIn("Watchdog 进程状态", rendered)
            self.assertIn("pid=123", rendered)
            self.assertIn("Watchdog 已生成产物", rendered)
            self.assertIn("result.png", rendered)

    def test_pipeline_worker_terminates_the_run_process_group(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            proc = subprocess.Popen(
                ["/bin/sh", "-c", "sleep 30 & wait"],
                start_new_session=True,
            )
            worker._proc = proc
            worker._proc_pgid = proc.pid
            try:
                worker._terminate_active_processes(grace_sec=0.2)
                proc.wait(timeout=2)
                self.assertFalse(worker._process_group_alive())
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if proc.poll() is None:
                    proc.wait(timeout=2)

    def test_gui_displays_export_success_with_teardown_warning(self):
        display = gui_module.SeestarGui._display_status(
            SimpleNamespace(),
            "CompletedWithWarning",
        )
        self.assertEqual(display, "已完成（有降级/需复核）")

    def test_pipeline_worker_temp_cleanup_timeout_does_not_block(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            temp_dir = root / "embedded"
            temp_dir.mkdir()
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            events = []
            worker.log = SimpleNamespace(emit=lambda text: events.append(str(text)))
            release_cleanup = threading.Event()
            cleanup_finished = threading.Event()
            original_rmtree = gui_module.PipelineWorker._cleanup_temp_dir.__globals__[
                "shutil"
            ].rmtree

            def slow_rmtree(_path):
                release_cleanup.wait(1)
                cleanup_finished.set()

            gui_module.PipelineWorker._cleanup_temp_dir.__globals__[
                "shutil"
            ].rmtree = slow_rmtree
            try:
                cleaned = worker._cleanup_temp_dir(temp_dir, timeout_sec=0.01)
            finally:
                release_cleanup.set()
                cleanup_finished.wait(1)
                gui_module.PipelineWorker._cleanup_temp_dir.__globals__[
                    "shutil"
                ].rmtree = original_rmtree

            self.assertFalse(cleaned)
            self.assertTrue(any("已转为后台清理" in event for event in events))

    def test_pipeline_worker_temp_cleanup_removes_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            temp_dir = root / "embedded"
            temp_dir.mkdir()
            (temp_dir / "runtime.bin").write_bytes(b"x")
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            cleaned = worker._cleanup_temp_dir(temp_dir, timeout_sec=1)

            self.assertTrue(cleaned)
            self.assertFalse(temp_dir.exists())

    def test_pipeline_worker_accepts_stage2_corrected_resume_input_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
                input_mode=gui_module.INPUT_MODE_STAGE2_CORRECTED_RESUME,
            )

            env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(
                env.get("SEESTAR_INPUT_MODE"),
                gui_module.INPUT_MODE_STAGE2_CORRECTED_RESUME,
            )

    def test_pipeline_worker_build_env_points_pip_to_bundled_wheels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            downloads = root / "resources" / "siril_plugins" / "downloads"
            runtime_downloads = root / "runtime_plugins" / "downloads"
            downloads.mkdir(parents=True)
            runtime_downloads.mkdir(parents=True)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            worker._runtime_plugin_dir = root / "runtime_plugins"

            env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(env.get("PIP_NO_INDEX"), "1")
            self.assertEqual(
                env.get("PIP_FIND_LINKS"),
                f"{downloads} {runtime_downloads}",
            )

    def test_pipeline_worker_uses_lightweight_plugin_overlay_for_large_resources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            work_dir.mkdir()
            config_template = root / "config.ini"
            config_template.write_text("[core]\nextension=.fit\n", encoding="utf-8")
            pipeline_path = root / "pipeline.py"
            pipeline_path.write_text("# mock\n", encoding="utf-8")
            pipeline_path.with_name("stage11_ai_postprocess.py").write_text(
                "# mock stage11\n",
                encoding="utf-8",
            )
            plugin_dir = root / "plugins"
            for name in ("downloads", "syqon_starless", "cosmic_clarity"):
                resource_dir = plugin_dir / name
                resource_dir.mkdir(parents=True)
                (resource_dir / "large.bin").write_bytes(b"x" * 1024)
            (plugin_dir / "syqon_starless" / "zenith.pt").write_bytes(b"model")
            (plugin_dir / "model_v2_0_1.onnx").write_bytes(b"graxpert-model")
            (plugin_dir / "bin").mkdir()
            (plugin_dir / "bin" / "helper").write_text("small", encoding="utf-8")

            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=config_template,
                pipeline_path=pipeline_path,
                siril_plugin_dir=plugin_dir,
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            temp_dir = root / "temp"
            temp_dir.mkdir()

            worker._prepare_runtime_files(temp_dir)
            overlay = temp_dir / "siril_plugins"

            self.assertTrue((overlay / "downloads").is_symlink())
            self.assertTrue((overlay / "syqon_starless").is_symlink())
            self.assertTrue((overlay / "cosmic_clarity").is_symlink())
            self.assertTrue((overlay / "model_v2_0_1.onnx").is_symlink())
            self.assertTrue((overlay / "bin" / "helper").is_file())

            env = worker._build_env(Path("/tmp/siril-cli"))
            self.assertEqual(
                env.get("SEESTAR_SYQON_MODEL_DIR"),
                str(plugin_dir / "syqon_starless"),
            )
            self.assertEqual(
                env.get("SEESTAR_COSMIC_CLARITY_MODEL_DIR"),
                str(plugin_dir / "cosmic_clarity"),
            )

    def test_direct_model_resources_remove_only_matching_legacy_copies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plugin_dir = root / "plugins"
            syqon_bundle = plugin_dir / "syqon_starless"
            syqon_bundle.mkdir(parents=True)
            (syqon_bundle / "zenith.pt").write_bytes(b"bundled-model")
            runtime_syqon = root / "runtime" / "syqon_starless"
            runtime_syqon.mkdir(parents=True)
            (runtime_syqon / "zenith.pt").write_bytes(b"bundled-model")
            (runtime_syqon / "user-model.pt").write_bytes(b"keep")
            events: list[str] = []
            proxy = SimpleNamespace(
                siril_plugin_dir=plugin_dir,
                _check_bootstrap_cancelled=lambda: None,
                _runtime_syqon_starless_dir=lambda: runtime_syqon,
                _display_path=str,
                _append_event=events.append,
                _remove_matching_legacy_runtime_files=lambda bundle, target, names: gui_module.SeestarGui._remove_matching_legacy_runtime_files(
                    proxy,
                    bundle,
                    target,
                    names,
                ),
            )

            gui_module.SeestarGui._sync_syqon_starless_bundle(proxy)

            self.assertFalse((runtime_syqon / "zenith.pt").exists())
            self.assertTrue((runtime_syqon / "user-model.pt").exists())
            self.assertTrue(any("只读离线资源" in event for event in events))

    def test_runtime_disk_estimate_counts_dependencies_without_model_copies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime_home = root / "runtime_home"
            state_root = runtime_home / "state"
            (state_root / "venv").mkdir(parents=True)
            (state_root / ".python_module" / "sirilpy").mkdir(parents=True)
            seed_dir = root / "seed"
            (seed_dir / "venv").mkdir(parents=True)
            (seed_dir / ".python_module" / "sirilpy").mkdir(parents=True)
            plugin_dir = root / "plugins"
            downloads = plugin_dir / "downloads"
            downloads.mkdir(parents=True)
            (downloads / "deps.whl").write_bytes(b"x" * 100)
            proxy = SimpleNamespace(
                runtime_home=runtime_home,
                siril_seed_dir=seed_dir,
                siril_plugin_dir=plugin_dir,
                _siril_state_root=lambda: state_root,
                _runtime_siril_scripts_repo_dir=lambda: runtime_home / "scripts",
                _bootstrap_state_is_current=lambda _fingerprint: False,
                _plugin_downloads_dir=lambda: downloads,
            )
            estimate = gui_module.SeestarGui._estimate_runtime_disk_space(
                proxy,
                {"fingerprint": "changed"},
            )

            self.assertEqual(estimate.seed_growth_bytes, 0)
            self.assertEqual(estimate.support_growth_bytes, 0)
            self.assertEqual(
                estimate.dependency_growth_bytes,
                int(100 * gui_module.SeestarGui._estimate_runtime_disk_space.__globals__["RUNTIME_DEPENDENCY_EXPANSION_FACTOR"]),
            )

    def test_core_app_discovers_adjacent_offline_resource_pack(self):
        with tempfile.TemporaryDirectory() as td:
            distribution_root = Path(td)
            resources = (
                distribution_root
                / "SeestarSuperimpose.app"
                / "Contents"
                / "Resources"
            )
            resources.mkdir(parents=True)
            external_plugins = (
                distribution_root
                / "SeestarSuperimpose-OfflineResources"
                / "siril_plugins"
            )
            external_plugins.mkdir(parents=True)
            resolver = gui_module.SeestarGui.__init__.__globals__[
                "default_siril_plugin_dir"
            ]
            resolver_globals = resolver.__globals__
            original_is_frozen = resolver_globals["is_frozen"]
            resolver_globals["is_frozen"] = lambda: True
            try:
                resolved = resolver(resources)
            finally:
                resolver_globals["is_frozen"] = original_is_frozen

            self.assertEqual(resolved, external_plugins)

            embedded_plugins = resources / "siril_plugins"
            embedded_plugins.mkdir()
            resolver_globals["is_frozen"] = lambda: True
            try:
                resolved_full = resolver(resources)
            finally:
                resolver_globals["is_frozen"] = original_is_frozen
            self.assertEqual(resolved_full, embedded_plugins)

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_spcc_crash_retry_env_disables_spcc(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            worker._force_disable_spcc_for_retry = True

            env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(env.get("SEESTAR_SPCC_ENABLE"), "0")

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_spcc_crash_retry_resumes_current_stage4_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            process_dir = work_dir / "process"
            process_dir.mkdir(parents=True)
            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            worker._capture_artifact_snapshot()
            checkpoint = process_dir / gui_module.STAGE4_PSOLVED_INPUT_NAME
            checkpoint.write_bytes(b"current-stage4-psolved")

            selected = worker._prepare_spcc_crash_retry()
            env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(selected, checkpoint)
            self.assertTrue(worker._force_disable_spcc_for_retry)
            self.assertTrue(worker._resume_stage4_psolved_for_retry)
            self.assertEqual(env.get("SEESTAR_SPCC_ENABLE"), "0")
            self.assertEqual(
                env.get("SEESTAR_INPUT_MODE"),
                gui_module.INPUT_MODE_STAGE4_PSOLVED_RESUME,
            )

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_does_not_resume_stale_stage4_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            process_dir = work_dir / "process"
            process_dir.mkdir(parents=True)
            checkpoint = process_dir / gui_module.STAGE4_PSOLVED_INPUT_NAME
            checkpoint.write_bytes(b"stale-stage4-psolved")
            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            worker._capture_artifact_snapshot()

            selected = worker._prepare_spcc_crash_retry()

            self.assertIsNone(selected)
            self.assertTrue(worker._force_disable_spcc_for_retry)
            self.assertFalse(worker._resume_stage4_psolved_for_retry)

    def test_pipeline_worker_marks_native_termination_marker_as_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            worker._inspect_output_for_errors(
                "[SIRIL_NATIVE_PROCESS_TERMINATED] connection closed"
            )

            self.assertTrue(worker._run_had_errors)
            self.assertTrue(worker._native_process_terminated_detected)

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_keeps_minus11_retry_and_uses_stage4_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            process_dir = work_dir / "process"
            process_dir.mkdir(parents=True)
            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[Path("/tmp/fake-siril-cli")],
            )
            events: list[str] = []
            done_calls: list[tuple[object, ...]] = []
            worker.log = SimpleNamespace(emit=lambda text: events.append(str(text)))
            worker.state = SimpleNamespace(emit=lambda *_args: None)
            worker.done = SimpleNamespace(emit=lambda *args: done_calls.append(args))
            worker._prepare_runtime_files = lambda temp: (
                temp / "run.ssf",
                temp / "run.ini",
                temp / "pipeline.py",
            )
            run_calls: list[int] = []

            def run_once(*_args):
                run_calls.append(len(run_calls) + 1)
                if len(run_calls) == 1:
                    worker._artifact_snapshot = {}
                    (process_dir / gui_module.STAGE4_PSOLVED_INPUT_NAME).write_bytes(
                        b"current-stage4-checkpoint"
                    )
                    worker._spcc_cli_crash_detected = True
                    return False, -11
                self.assertTrue(worker._force_disable_spcc_for_retry)
                self.assertTrue(worker._resume_stage4_psolved_for_retry)
                return True, 0

            worker._run_once = run_once

            worker.run()

            self.assertEqual(run_calls, [1, 2])
            self.assertTrue(done_calls)
            self.assertEqual(done_calls[-1][0], "Completed")
            self.assertIn("stage4_psolved.fit 检查点", "".join(events))

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_detects_spcc_command_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            worker._inspect_output_for_errors('input command:spcc "-oscsensor=Sony IMX585"')

            self.assertTrue(worker._spcc_seen_in_run)

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_does_not_misattribute_stage6_native_failure_to_spcc(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            worker._inspect_output_for_errors("log: [INFO] 阶段 4: 色彩校准")
            worker._inspect_output_for_errors('input command:spcc "-oscsensor=Sony IMX585"')
            worker._inspect_output_for_errors("log: [INFO] 阶段 6: 星点分离")
            worker._inspect_output_for_errors(
                "log: [SIRIL_NATIVE_PROCESS_TERMINATED] Bad file descriptor"
            )

            self.assertEqual(worker._native_termination_stage, 6)
            self.assertFalse(worker._is_spcc_crash_context(0))

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_attributes_stage4_native_failure_to_spcc(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            worker._inspect_output_for_errors("log: [INFO] 阶段 4: 色彩校准")
            worker._inspect_output_for_errors('input command:spcc "-oscsensor=Sony IMX585"')
            worker._inspect_output_for_errors(
                "log: [SIRIL_NATIVE_PROCESS_TERMINATED] SPCC connection closed"
            )

            self.assertTrue(worker._is_spcc_crash_context(0))

    def test_pipeline_worker_marks_degraded_pipeline_summary_as_warning(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            worker._inspect_output_for_errors(
                "log: [INFO] [PIPELINE_RUN_SUMMARY] failed=1 degraded=2"
            )

            self.assertEqual(worker._completed_run_status(), "CompletedWithWarning")

    @unittest.skip("SPCC runtime and crash-retry path retired")
    def test_pipeline_worker_spcc_crash_diagnostics_include_recent_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = gui_module.PipelineWorker(
                work_dir=root / "work",
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )
            events: list[str] = []
            worker.log = SimpleNamespace(emit=lambda text: events.append(str(text)))

            worker._inspect_output_for_errors('input command:spcc "-oscsensor=Sony IMX585"')
            worker._inspect_output_for_errors("log: Applying aperture photometry to 991 stars.")
            worker._append_spcc_crash_diagnostics(-11)

            rendered = "".join(events)
            self.assertIn("SPCC 崩溃诊断", rendered)
            self.assertIn("input command:spcc", rendered)
            self.assertIn("Applying aperture photometry to 991 stars", rendered)

    def test_pipeline_worker_configures_runtime_local_spcc_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            work_dir.mkdir(parents=True, exist_ok=True)
            config_template = root / "config.ini"
            config_template.write_text(
                "[core]\n"
                "extension=.fit\n"
                "catalogue_gaia_astro=/Users/mz/.local/share/siril/gaia_astrometric.dat\n"
                "catalogue_gaia_photo=/Users/mz/.local/share/siril/gaia_photometric.dat\n",
                encoding="utf-8",
            )
            pipeline_path = root / "pipeline.py"
            pipeline_path.write_text("# mock\n", encoding="utf-8")
            pipeline_path.with_name("stage11_ai_postprocess.py").write_text("# mock stage11\n", encoding="utf-8")
            resources = root / "resources"
            resources.mkdir(exist_ok=True)
            plugin_dir = root / "plugins"
            plugin_dir.mkdir(exist_ok=True)

            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=config_template,
                pipeline_path=pipeline_path,
                siril_plugin_dir=plugin_dir,
                resources=resources,
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            temp_dir = root / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            _run_ssf, run_ini, _run_py = worker._prepare_runtime_files(temp_dir)
            rendered = run_ini.read_text(encoding="utf-8")

        self.assertIn("catalogue_gaia_photo=\n", rendered)
        expected_astro = (
            root
            / "runtime_home"
            / ".local"
            / "share"
            / "siril"
            / "siril_cat_healpix8_astro.dat"
        )
        self.assertIn(f"catalogue_gaia_astro={expected_astro}\n", rendered)
        self.assertNotIn("gaia_photometric.dat", rendered)
        self.assertNotIn("gaia_astrometric.dat", rendered)

    def test_pipeline_worker_normalizes_config_template_without_starnet_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            work_dir.mkdir(parents=True, exist_ok=True)
            config_template = root / "config.ini"
            config_template.write_text("[core]\nstarnet_exe=old\nstarnet_weights=old\n", encoding="utf-8")
            pipeline_path = root / "pipeline.py"
            pipeline_path.write_text("# mock\n", encoding="utf-8")
            pipeline_path.with_name("stage11_ai_postprocess.py").write_text("# mock stage11\n", encoding="utf-8")
            resources = root / "resources"
            resources.mkdir(exist_ok=True)
            plugin_dir = root / "plugins"
            plugin_dir.mkdir(exist_ok=True)

            worker = gui_module.PipelineWorker(
                work_dir=work_dir,
                config_template=config_template,
                pipeline_path=pipeline_path,
                siril_plugin_dir=plugin_dir,
                resources=resources,
                runtime_home=root / "runtime_home",
                siril_candidates=[],
            )

            temp_dir = root / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            _run_ssf, run_ini, _run_py = worker._prepare_runtime_files(temp_dir)

            rendered = run_ini.read_text(encoding="utf-8")
            self.assertIn("[core]\n", rendered)
            self.assertIn("extension=.fit\n", rendered)
            self.assertIn("[gui]\n", rendered)
            self.assertNotIn("starnet_exe", rendered)
            self.assertNotIn("starnet_weights", rendered)
            runtime_catalog_root = (
                root / "runtime_home" / ".local" / "share" / "siril"
            )
            self.assertIn("catalogue_gaia_photo=\n", rendered)
            self.assertIn(
                "catalogue_gaia_astro="
                f"{runtime_catalog_root / 'siril_cat_healpix8_astro.dat'}\n",
                rendered,
            )

    def test_preflight_errors_require_result_linear_for_resume_mode(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "M42_master.fit").write_bytes(b"stacked")
            proxy = self._make_gui_proxy(
                work_dir,
                input_mode=gui_module.INPUT_MODE_LINEAR_RESUME,
            )

            errors = gui_module.SeestarGui._preflight_errors(proxy, work_dir)

            self.assertTrue(errors)
            self.assertTrue(
                any(gui_module.LINEAR_RESUME_INPUT_NAME in err for err in errors)
            )

    def test_preflight_errors_require_stage2_corrected_for_resume_mode(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "M42_master.fit").write_bytes(b"stacked")
            proxy = self._make_gui_proxy(
                work_dir,
                input_mode=gui_module.INPUT_MODE_STAGE2_CORRECTED_RESUME,
            )

            errors = gui_module.SeestarGui._preflight_errors(proxy, work_dir)

            self.assertTrue(errors)
            self.assertTrue(
                any(gui_module.STAGE2_CORRECTED_INPUT_NAME in err for err in errors)
            )

    def test_linear_resume_disk_estimate_and_summary_use_dedicated_mode(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            linear_path = work_dir / gui_module.LINEAR_RESUME_INPUT_NAME
            linear_path.write_bytes(b"x" * 1024)
            proxy = self._make_gui_proxy(
                work_dir,
                input_mode=gui_module.INPUT_MODE_LINEAR_RESUME,
            )

            estimate = gui_module.SeestarGui._estimate_disk_space(proxy, work_dir)
            self.assertIsNotNone(estimate)
            assert estimate is not None
            self.assertEqual(estimate.mode, "linear_resume")
            self.assertEqual(estimate.selected_input_label, gui_module.LINEAR_RESUME_INPUT_NAME)
            self.assertEqual(
                estimate.estimated_peak_growth_bytes,
                int(
                    1024
                    * gui_module.SeestarGui._estimate_disk_space.__globals__[
                        "LINEAR_RESUME_STAGE_ARTIFACT_COPIES"
                    ]
                ),
            )

            summary_lines = gui_module.SeestarGui._disk_space_summary_lines(proxy, estimate)
            self.assertTrue(any("从线性处理后继续" in line for line in summary_lines))
            self.assertTrue(
                any(gui_module.LINEAR_RESUME_INPUT_NAME in line for line in summary_lines)
            )

    def test_stage2_corrected_resume_disk_estimate_and_summary_use_dedicated_mode(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            process_dir = work_dir / "process"
            process_dir.mkdir()
            stage2_path = process_dir / gui_module.STAGE2_CORRECTED_INPUT_NAME
            stage2_path.write_bytes(b"x" * 1024)
            proxy = self._make_gui_proxy(
                work_dir,
                input_mode=gui_module.INPUT_MODE_STAGE2_CORRECTED_RESUME,
            )

            estimate = gui_module.SeestarGui._estimate_disk_space(proxy, work_dir)
            self.assertIsNotNone(estimate)
            assert estimate is not None
            self.assertEqual(estimate.mode, "stage2_corrected_resume")
            self.assertEqual(estimate.selected_input_label, gui_module.STAGE2_CORRECTED_INPUT_NAME)
            self.assertEqual(
                estimate.estimated_peak_growth_bytes,
                int(
                    1024
                    * gui_module.SeestarGui._estimate_disk_space.__globals__[
                        "STAGE2_RESUME_STAGE_ARTIFACT_COPIES"
                    ]
                ),
            )
            self.assertGreaterEqual(
                gui_module.SeestarGui._estimate_disk_space.__globals__[
                    "STAGE2_RESUME_STAGE_ARTIFACT_COPIES"
                ],
                40.0,
            )

            summary_lines = gui_module.SeestarGui._disk_space_summary_lines(proxy, estimate)
            self.assertTrue(any("从裁切后继续" in line for line in summary_lines))
            self.assertTrue(
                any(gui_module.STAGE2_CORRECTED_INPUT_NAME in line for line in summary_lines)
            )

    def test_stage0_preview_candidates_follow_input_mode(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            light_b = work_dir / "Light_002.fit"
            light_a = work_dir / "Light_001.fit"
            resume = work_dir / gui_module.LINEAR_RESUME_INPUT_NAME
            for path in (light_b, light_a, resume):
                path.write_bytes(b"fits")

            proxy = SimpleNamespace(
                _current_input_mode=lambda: gui_module.INPUT_MODE_AUTO,
                _linear_resume_input_path=lambda wd: wd / gui_module.LINEAR_RESUME_INPUT_NAME,
                _stage2_corrected_resume_input_path=lambda wd: wd / gui_module.STAGE2_CORRECTED_INPUT_NAME,
                _fits_in_work_dir=lambda _wd: [light_b, resume, light_a],
                _is_candidate_stacked_input=lambda _path, _wd: False,
            )

            candidates, label = gui_module.SeestarGui._initial_preview_candidates(
                proxy,
                work_dir,
                gui_module.INPUT_MODE_AUTO,
            )
            self.assertEqual(candidates, [light_a, light_b])
            self.assertEqual(label, "输入样本 · 2 帧")

            resume_candidates, resume_label = (
                gui_module.SeestarGui._initial_preview_candidates(
                    proxy,
                    work_dir,
                    gui_module.INPUT_MODE_LINEAR_RESUME,
                )
            )
            self.assertEqual(resume_candidates, [resume])
            self.assertEqual(resume_label, "续跑输入")

    def test_pipeline_worker_emits_preview_events_for_stage1_through_stage11(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            preview_path = root / "latest.png"
            preview_path.write_bytes(b"png")
            worker = gui_module.PipelineWorker(
                work_dir=root,
                config_template=root / "config.ini",
                pipeline_path=root / "pipeline.py",
                siril_plugin_dir=root / "plugins",
                resources=root / "resources",
                runtime_home=root / "runtime",
                siril_candidates=[],
            )
            events = []
            worker.preview = SimpleNamespace(
                emit=lambda *values: events.append(values)
            )

            for stage in range(1, 12):
                worker._inspect_output_for_errors(
                    "[INFO] [PIPELINE_PREVIEW] "
                    f'{{"stage":{stage},"title":"阶段 {stage}",'
                    f'"status":"ready","payload":"{preview_path}"}}'
                )

            self.assertEqual([event[0] for event in events], list(range(1, 12)))
            self.assertTrue(all(event[2] == "ready" for event in events))
            self.assertTrue(all(event[3] == str(preview_path) for event in events))

    def test_preview_failure_retains_previous_reliable_stage(self):
        class Label:
            def __init__(self):
                self.value = ""

            def setText(self, value):
                self.value = value

        events = []
        activity = Label()
        status = Label()
        proxy = SimpleNamespace(
            _latest_preview_stage=4,
            preview_activity_label=activity,
            preview_status_label=status,
            _display_latest_preview=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unavailable preview must not replace the current image")
            ),
            _append_event=events.append,
        )

        gui_module.SeestarGui._on_pipeline_preview(
            proxy,
            5,
            "线性降噪",
            "unavailable",
            "mock decode failure",
        )

        self.assertIn("保留上一张", activity.value)
        self.assertIn("Stage 4", status.value)
        self.assertTrue(any("不影响处理结果" in item for item in events))


if __name__ == "__main__":
    unittest.main()
