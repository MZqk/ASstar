#!/usr/bin/env python3
"""Tests for the headless source-pipeline development launcher."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gui import starun_pipeline_dev as launcher


class PipelineDevLauncherTests(unittest.TestCase):
    def test_parser_is_online_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            args = launcher.build_parser().parse_args(
                ["--work-dir", td]
            )

        self.assertTrue(args.network)
        self.assertFalse(args.debug)
        self.assertEqual(args.input_mode, launcher.INPUT_MODE_AUTO)
        self.assertIsNone(args.offline_resource_root)
        self.assertIsNone(args.task_run_manifest)

    def test_parser_accepts_explicit_offline_mode(self):
        with tempfile.TemporaryDirectory() as td:
            args = launcher.build_parser().parse_args(
                ["--work-dir", td, "--offline"]
            )

        self.assertFalse(args.network)

    def test_external_resource_root_selects_runner_plugin_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "offline-resources"
            plugin_dir = root / "siril_plugins"
            plugin_dir.mkdir(parents=True)
            args = launcher.build_parser().parse_args(
                [
                    "--work-dir",
                    td,
                    "--offline-resource-root",
                    str(root),
                ]
            )

            resolved = launcher.resolve_siril_plugin_dir(
                args.offline_resource_root
            )

        self.assertEqual(resolved, plugin_dir.resolve())

    def test_external_resource_root_does_not_fallback_to_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            missing_root = Path(td) / "missing-resources"

            with self.assertRaisesRegex(ValueError, "缺少 siril_plugins"):
                launcher.resolve_siril_plugin_dir(missing_root)

    def test_pipeline_worker_receives_external_plugin_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plugin_dir = root / "offline-resources/siril_plugins"
            plugin_dir.mkdir(parents=True)
            args = launcher.build_parser().parse_args(
                [
                    "--work-dir",
                    td,
                    "--runtime-home",
                    str(root / "runtime-home"),
                    "--offline-resource-root",
                    str(plugin_dir.parent),
                ]
            )
            with (
                mock.patch.object(
                    launcher,
                    "verify_siril_offline_seed_venv",
                    return_value=(True, "ready"),
                ),
                mock.patch.object(
                    launcher,
                    "prepare_resource_overlay",
                    return_value=root / "overlay",
                ),
                mock.patch.object(launcher, "PipelineWorker") as worker_class,
            ):
                worker = worker_class.return_value

                def finish_run() -> None:
                    callback = worker.done.connect.call_args.args[0]
                    callback("Completed", 0, False, "siril-cli")

                worker.run.side_effect = finish_run

                exit_code = launcher.run_pipeline(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            worker_class.call_args.kwargs["siril_plugin_dir"],
            plugin_dir.resolve(),
        )

    def test_pipeline_worker_receives_task_run_manifest_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "run-manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            args = launcher.build_parser().parse_args(
                [
                    "--work-dir",
                    td,
                    "--runtime-home",
                    str(root / "runtime-home"),
                    "--task-run-manifest",
                    str(manifest),
                ]
            )
            with (
                mock.patch.object(
                    launcher,
                    "verify_siril_offline_seed_venv",
                    return_value=(True, "ready"),
                ),
                mock.patch.object(
                    launcher,
                    "prepare_resource_overlay",
                    return_value=root / "overlay",
                ),
                mock.patch.object(launcher, "PipelineWorker") as worker_class,
            ):
                worker = worker_class.return_value

                def finish_run() -> None:
                    callback = worker.done.connect.call_args.args[0]
                    callback("Completed", 0, False, "siril-cli")

                worker.run.side_effect = finish_run

                exit_code = launcher.run_pipeline(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            worker_class.call_args.kwargs["runtime_overrides"],
            {"STARUN_TASK_RUN_MANIFEST": str(manifest.resolve())},
        )

    def test_root_checkpoint_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            (work_dir / "stage2_corrected.fit").touch()

            with self.assertRaisesRegex(
                ValueError,
                "验签产品任务",
            ):
                launcher.validate_work_dir(
                    work_dir,
                    "stage2_corrected_resume",
                )

    def test_parser_rejects_manual_resume_mode(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(SystemExit):
            launcher.build_parser().parse_args(
                [
                    "--work-dir",
                    td,
                    "--input-mode",
                    "stage5_linear_resume",
                ]
            )


if __name__ == "__main__":
    unittest.main()
