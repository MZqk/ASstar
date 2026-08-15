#!/usr/bin/env python3
"""Contract tests for the opt-in real Siril Stage 1-10 runner."""
from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from pipeline import run_manifest, task_plan
from tests import real_siril_stage1_10_e2e as real_e2e


class RealSirilStage110E2EContractTests(unittest.TestCase):
    @staticmethod
    def _write_complete_run(root: Path) -> dict[str, object]:
        output = root / "result_final.fit"
        output.write_bytes(b"verified-final-fits")
        plan = task_plan.build_processing_plan(
            run_id="real-e2e-contract",
            generated_at="2026-08-14T00:00:00Z",
            input_record={"fingerprint": "real-e2e-input"},
            input_state="linear",
            input_trust="recognized",
        )
        run_manifest.atomic_write_json(root / "processing-plan.json", plan)
        result: dict[str, object] = {
            "schema": "starun.pipeline-result.v1",
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "status": "success",
            "actual_steps": [
                {"name": f"阶段 {stage}: contract", "status": "ok"}
                for stage in real_e2e.EXPECTED_STAGES
            ],
            "outputs": {
                output.name: run_manifest.file_record(output, base_dir=root)
            },
        }
        result["manifest_hash"] = run_manifest.canonical_payload_hash(result)
        run_manifest.atomic_write_json(root / "pipeline-result.json", result)
        process_dir = root / "process"
        process_dir.mkdir()
        run_manifest.atomic_write_json(
            process_dir / "final_quality_report.json",
            {"status": "accepted"},
        )
        run_manifest.atomic_write_json(
            process_dir / "stage6_syqon_exchange.json",
            {"status": "accepted", "accepted": True},
        )
        return result

    def test_verifier_requires_full_stage_chain_final_export_and_syqon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete_run(root)

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertTrue(verified)
        self.assertIn("verified_stages=1-10", details)
        self.assertIn("offline_syqon=accepted", details)

    def test_verifier_rejects_missing_formal_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._write_complete_run(root)
            result["actual_steps"] = result["actual_steps"][:-1]
            result.pop("manifest_hash", None)
            result["manifest_hash"] = run_manifest.canonical_payload_hash(result)
            run_manifest.atomic_write_json(root / "pipeline-result.json", result)

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertFalse(verified)
        self.assertTrue(any("未覆盖完整 Stage 1-10" in item for item in details))

    def test_verifier_rejects_mocked_or_bypassed_syqon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete_run(root)
            run_manifest.atomic_write_json(
                root / "process/stage6_syqon_exchange.json",
                {"status": "bypassed", "accepted": False},
            )

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertFalse(verified)
        self.assertIn("真实离线 SyQon 交换未验收", details)

    def test_environment_validation_reports_work_dir_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_path = root / "not-a-directory"
            work_path.write_text("occupied", encoding="utf-8")
            args = Namespace(
                input=root / "missing.fit",
                work_dir=work_path,
                siril_app=root / "missing-siril.app",
                siril_seed=root / "missing-seed",
                runtime_home=None,
                offline_resource_root=root / "missing-resources",
            )

            errors = real_e2e.validate_environment(args)

        self.assertIn(f"工作目录路径不是目录：{work_path}", errors)


if __name__ == "__main__":
    unittest.main()
