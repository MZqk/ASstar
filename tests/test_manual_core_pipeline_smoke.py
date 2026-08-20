#!/usr/bin/env python3
"""Tests for the manual, packaging-free core pipeline smoke script."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from pipeline import run_manifest
from pipeline import task_plan
from tests import manual_core_pipeline_smoke as smoke


class ManualCorePipelineSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_plan(work_dir: Path, run_id: str = "smoke-run") -> dict[str, object]:
        plan = task_plan.build_processing_plan(
            run_id=run_id,
            generated_at="2026-08-14T00:00:00Z",
            input_record={"fingerprint": "smoke-input"},
            input_state="linear",
            input_trust="recognized",
        )
        run_manifest.atomic_write_json(work_dir / "processing-plan.json", plan)
        return plan

    def test_launcher_command_defaults_to_network_and_explicit_runtime(self) -> None:
        args = smoke.build_parser().parse_args(
            [
                "--mode",
                "auto",
                "--work-dir",
                "/tmp/core-work",
                "--siril-app",
                "/Applications/Siril.app",
                "--siril-seed",
                "/tmp/siril-seed",
                "--runtime-home",
                "/tmp/runtime-home",
            ]
        )

        command = smoke.build_launcher_command(args, args.mode)

        self.assertIn("gui.starun_pipeline_dev", command)
        self.assertIn("auto", command)
        self.assertIn("--debug", command)
        self.assertNotIn("--network", command)
        self.assertNotIn("--offline-resource-root", command)
        self.assertNotIn("starun_gui_dev", command)

    def test_launcher_command_can_force_offline_mode(self) -> None:
        args = smoke.build_parser().parse_args(
            [
                "--mode",
                "auto",
                "--work-dir",
                "/tmp/core-work",
                "--offline",
            ]
        )

        command = smoke.build_launcher_command(args, args.mode)

        self.assertFalse(args.network)
        self.assertNotIn("--network", command)
        self.assertIn("--offline", command)

    def test_launcher_command_forwards_external_resource_root(self) -> None:
        args = smoke.build_parser().parse_args(
            [
                "--mode",
                "auto",
                "--work-dir",
                "/tmp/core-work",
                "--offline-resource-root",
                "/Volumes/starun-e2e-resources",
                "--offline",
            ]
        )

        command = smoke.build_launcher_command(args, args.mode)

        option_index = command.index("--offline-resource-root")
        self.assertEqual(
            command[option_index + 1],
            "/Volumes/starun-e2e-resources",
        )
        self.assertIn("--offline", command)

    def test_verify_result_manifest_checks_current_outputs_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            output = work_dir / "M_42_processed_final.fit"
            output.write_bytes(b"fits-output")
            plan = self._write_plan(work_dir)
            payload = {
                "schema": "starun.pipeline-result.v1",
                "run_id": plan["run_id"],
                "plan_hash": plan["plan_hash"],
                "status": "partial_success",
                "outputs": {
                    output.name: run_manifest.file_record(
                        output,
                        base_dir=work_dir,
                    )
                },
            }
            payload["manifest_hash"] = run_manifest.canonical_payload_hash(payload)
            manifest = work_dir / "pipeline-result.json"
            run_manifest.atomic_write_json(manifest, payload)
            started_at = manifest.stat().st_mtime - 0.5

            verified, details = smoke.verify_result_manifest(
                work_dir,
                run_started_at=started_at,
            )

        self.assertTrue(verified)
        self.assertIn("pipeline_status=partial_success", details)
        self.assertIn("verified_outputs=1", details)

    def test_verify_result_manifest_rejects_stale_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            output = work_dir / "result_final.fit"
            output.write_bytes(b"fits-output")
            plan = self._write_plan(work_dir)
            payload = {
                "run_id": plan["run_id"],
                "plan_hash": plan["plan_hash"],
                "status": "success",
                "outputs": {
                    output.name: run_manifest.file_record(output, base_dir=work_dir)
                },
            }
            payload["manifest_hash"] = run_manifest.canonical_payload_hash(payload)
            manifest = work_dir / "pipeline-result.json"
            run_manifest.atomic_write_json(manifest, payload)
            os.utime(manifest, (100.0, 100.0))

            verified, details = smoke.verify_result_manifest(
                work_dir,
                run_started_at=200.0,
            )

        self.assertFalse(verified)
        self.assertIn("不是本轮生成", details[0])

    def test_verify_result_manifest_rejects_modified_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td)
            output = work_dir / "result_final.fit"
            output.write_bytes(b"original")
            plan = self._write_plan(work_dir)
            payload = {
                "run_id": plan["run_id"],
                "plan_hash": plan["plan_hash"],
                "status": "success",
                "outputs": {
                    output.name: run_manifest.file_record(output, base_dir=work_dir)
                },
            }
            payload["manifest_hash"] = run_manifest.canonical_payload_hash(payload)
            run_manifest.atomic_write_json(work_dir / "pipeline-result.json", payload)
            output.write_bytes(b"modified")

            verified, details = smoke.verify_result_manifest(work_dir)

        self.assertFalse(verified)
        self.assertIn(output.name, details[0])


if __name__ == "__main__":
    unittest.main()
