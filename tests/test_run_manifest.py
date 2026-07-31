#!/usr/bin/env python3
"""Regression tests for durable plan/result provenance helpers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_manifest  # noqa: E402


class RunManifestTests(unittest.TestCase):
    def _write_result_manifest(
        self,
        root: Path,
        checkpoint: Path,
        *,
        state: str = "linear",
    ) -> None:
        payload = {
            "schema": "seestar.pipeline-result.v1",
            "status": "success",
            "plan_hash": "plan-hash",
            "checkpoints": {
                "result_linear": {
                    **run_manifest.file_record(checkpoint, base_dir=root),
                    "state": state,
                }
            },
        }
        payload["manifest_hash"] = run_manifest.canonical_payload_hash(payload)
        run_manifest.atomic_write_json(root / "pipeline-result.json", payload)

    def test_matching_signed_linear_checkpoint_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "result_linear.fit"
            checkpoint.write_bytes(b"linear-checkpoint")
            self._write_result_manifest(root, checkpoint)

            result = run_manifest.verify_resume_provenance(
                work_dir=root,
                input_path=checkpoint,
                checkpoint_name="result_linear",
            )

        self.assertTrue(result["verified"])
        self.assertEqual(result["state"], "linear")
        self.assertEqual(result["plan_hash"], "plan-hash")

    def test_modified_checkpoint_is_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "result_linear.fit"
            checkpoint.write_bytes(b"linear-checkpoint")
            self._write_result_manifest(root, checkpoint)
            checkpoint.write_bytes(b"modified")

            result = run_manifest.verify_resume_provenance(
                work_dir=root,
                input_path=checkpoint,
                checkpoint_name="result_linear",
            )

        self.assertFalse(result["verified"])
        self.assertIn("does not match", result["detail"])

    def test_tampered_manifest_is_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "result_linear.fit"
            checkpoint.write_bytes(b"linear-checkpoint")
            self._write_result_manifest(root, checkpoint)
            manifest_path = root / "pipeline-result.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["checkpoints"]["result_linear"]["state"] = "unknown"
            run_manifest.atomic_write_json(manifest_path, payload)

            result = run_manifest.verify_resume_provenance(
                work_dir=root,
                input_path=checkpoint,
                checkpoint_name="result_linear",
            )

        self.assertFalse(result["verified"])
        self.assertIn("manifest hash", result["detail"])

    def test_sensitive_config_values_are_redacted_recursively(self) -> None:
        redacted = run_manifest.redact_sensitive(
            {
                "api_key": "secret-value",
                "nested": {"access_token": "token-value", "mode": "offline"},
            }
        )

        self.assertEqual(redacted["api_key"], "<redacted>")
        self.assertEqual(redacted["nested"]["access_token"], "<redacted>")
        self.assertEqual(redacted["nested"]["mode"], "offline")

    def test_collect_output_records_uses_current_export_names_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stale_review = root / "result_review.png"
            stale_review.write_bytes(b"stale-review")
            stale_custom = root / "Old_Target_processed.png"
            stale_custom.write_bytes(b"stale-custom")
            for path in (stale_review, stale_custom):
                os.utime(path, (100.0, 100.0))

            export_started_at = 200.0
            current_files = (
                root / "M_42_processed.png",
                root / "M_42_processed.tif",
                root / "M_42_processed_final.fit",
                root / "M_42_processed_display_srgb.png",
            )
            for path in current_files:
                path.write_bytes(path.name.encode("utf-8"))
                os.utime(path, (201.0, 201.0))

            durable_files = (
                root / "processing-plan.json",
                root / "result_linear.fit",
                root / "seestar_diagnostics.zip",
            )
            for path in durable_files:
                path.write_bytes(path.name.encode("utf-8"))
                os.utime(path, (100.0, 100.0))

            outputs = run_manifest.collect_output_records(
                root,
                output_basenames=("M_42_processed", "M_42_processed_final"),
                exported_after=export_started_at,
            )

        self.assertEqual(
            set(outputs),
            {
                *(path.name for path in current_files),
                *(path.name for path in durable_files),
            },
        )
        self.assertNotIn(stale_review.name, outputs)
        self.assertNotIn(stale_custom.name, outputs)


if __name__ == "__main__":
    unittest.main()
