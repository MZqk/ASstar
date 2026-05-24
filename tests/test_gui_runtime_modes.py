#!/usr/bin/env python3
"""GUI runtime mode and plugin integrity helper tests."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


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
    qt_core.QUrl = type("QUrl", (_DummyObject,), {})
    qt_core.Signal = lambda *_a, **_k: _DummySignal()

    qt_gui.QAction = type("QAction", (_DummyObject,), {})
    qt_gui.QDesktopServices = type("QDesktopServices", (_DummyObject,), {})
    qt_gui.QTextCursor = type("QTextCursor", (_DummyObject,), {})

    for name in (
        "QApplication",
        "QComboBox",
        "QFileDialog",
        "QGridLayout",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QMainWindow",
        "QMessageBox",
        "QPushButton",
        "QPlainTextEdit",
        "QSizePolicy",
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


class GuiRuntimeModesTests(unittest.TestCase):
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

            env = worker._build_env(Path("/tmp/siril-cli"))

            self.assertEqual(
                env.get("SEESTAR_INPUT_MODE"),
                gui_module.INPUT_MODE_LINEAR_RESUME,
            )

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

            worker._inspect_output_for_errors('input command:spcc "-oscsensor=seestar s30pro"')

            self.assertTrue(worker._spcc_seen_in_run)

    def test_normalize_siril_config_blanks_legacy_gaia_photo_file_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "work"
            work_dir.mkdir(parents=True, exist_ok=True)
            config_template = root / "config.ini"
            config_template.write_text(
                "[core]\n"
                "extension=.fit\n"
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
        self.assertNotIn("gaia_photometric.dat", rendered)

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

            summary_lines = gui_module.SeestarGui._disk_space_summary_lines(proxy, estimate)
            self.assertTrue(
                any("result_linear.fit 后期模式" in line for line in summary_lines)
            )
            self.assertTrue(
                any(gui_module.LINEAR_RESUME_INPUT_NAME in line for line in summary_lines)
            )


if __name__ == "__main__":
    unittest.main()
