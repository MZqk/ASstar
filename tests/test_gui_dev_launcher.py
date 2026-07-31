#!/usr/bin/env python3
"""Tests for the no-package source GUI development launcher."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_LAUNCHER_PATH = REPO_ROOT / "gui" / "seestar_gui_dev.py"


def _load_dev_launcher():
    spec = importlib.util.spec_from_file_location(
        "seestar_gui_dev_test_module",
        DEV_LAUNCHER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {DEV_LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dev_launcher = _load_dev_launcher()


class GuiDevLauncherTests(unittest.TestCase):
    def _make_siril_app(self, root: Path) -> Path:
        siril_app = root / "System Siril.app"
        cli = siril_app / "Contents/MacOS/siril-cli"
        python = (
            siril_app
            / "Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
        )
        cli.parent.mkdir(parents=True)
        python.parent.mkdir(parents=True)
        cli.write_text("#!/bin/sh\n", encoding="utf-8")
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        cli.chmod(0o755)
        python.chmod(0o755)
        return siril_app

    def _make_seed(self, root: Path) -> Path:
        seed = root / "existing seed"
        (seed / "venv").mkdir(parents=True)
        (seed / ".python_module/sirilpy").mkdir(parents=True)
        return seed

    def test_resource_overlay_links_project_resources_siril_and_seed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project_resources = root / "project resources"
            project_resources.mkdir()
            config = project_resources / "config.1.4.ini.template"
            config.write_text("[core]\n", encoding="utf-8")
            plugin_dir = project_resources / "siril_plugins"
            plugin_dir.mkdir()
            siril_app = self._make_siril_app(root)
            seed = self._make_seed(root)

            overlay = dev_launcher.prepare_resource_overlay(
                root / "overlay",
                project_resources=project_resources,
                siril_app=siril_app,
                siril_seed=seed,
            )

            self.assertEqual(
                (overlay / "config.1.4.ini.template").resolve(),
                config.resolve(),
            )
            self.assertEqual(
                (overlay / "siril_plugins").resolve(),
                plugin_dir.resolve(),
            )
            self.assertEqual((overlay / "Siril.app").resolve(), siril_app.resolve())
            self.assertEqual(
                (overlay / "SirilPythonSeed").resolve(),
                seed.resolve(),
            )

    def test_validation_rejects_incomplete_siril_app(self):
        with tempfile.TemporaryDirectory() as td:
            siril_app = Path(td) / "Siril.app"
            siril_app.mkdir()
            with self.assertRaisesRegex(
                dev_launcher.DevLauncherError,
                "Siril 运行文件缺失",
            ):
                dev_launcher.validate_siril_app(siril_app)

    def test_validation_rejects_seed_without_sirilpy(self):
        with tempfile.TemporaryDirectory() as td:
            seed = Path(td) / "seed"
            (seed / "venv").mkdir(parents=True)
            with self.assertRaisesRegex(
                dev_launcher.DevLauncherError,
                "Siril seed 不完整",
            ):
                dev_launcher.validate_siril_seed(seed)


if __name__ == "__main__":
    unittest.main()
