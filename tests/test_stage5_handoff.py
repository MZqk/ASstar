#!/usr/bin/env python3
"""Regression tests for the strict Stage 5 -> Stage 6 source contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_manifest  # noqa: E402
import stage5_handoff  # noqa: E402


class Stage5HandoffTests(unittest.TestCase):
    def _pipeline(self, root: Path, *, run_id: str = "run-1") -> SimpleNamespace:
        process_dir = root / "process"
        process_dir.mkdir()
        return SimpleNamespace(
            process_dir=process_dir,
            _run_id=run_id,
            _stage5_linear_handoff={},
        )

    def _current_input_lineage(self, pipeline: SimpleNamespace) -> dict:
        (pipeline.process_dir / stage5_handoff.STAGE5_UPSTREAM_ARTIFACT).write_bytes(
            b"canonical-stage4"
        )
        (pipeline.process_dir / stage5_handoff.STAGE5_INPUT_ARTIFACT).write_bytes(
            b"saved-stage5-input"
        )
        return stage5_handoff.freeze_stage5_input_lineage(
            pipeline,
            upstream_loaded=True,
            baseline_saved=True,
        )

    def test_current_run_handoff_persists_separate_integrity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self._pipeline(Path(temporary_directory))
            artifact = pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT
            artifact.write_bytes(b"current-run-linear")
            input_lineage = self._current_input_lineage(pipeline)

            record = stage5_handoff.freeze_stage5_handoff(
                pipeline,
                origin=stage5_handoff.CURRENT_RUN_ORIGIN,
                stage_status="degraded",
                deconvolution_integrity_ok=True,
                denoise_integrity_ok=True,
                formal_eligible=True,
                input_lineage=input_lineage,
            )

            persisted = run_manifest.load_json(
                pipeline.process_dir / stage5_handoff.STAGE5_HANDOFF_REPORT
            )
            self.assertTrue(record["accepted"])
            self.assertTrue(record["deconvolution_integrity_ok"])
            self.assertTrue(record["denoise_integrity_ok"])
            self.assertTrue(record["input_integrity_ok"])
            self.assertEqual(
                record["input_lineage"]["upstream"]["artifact"],
                "stage4_color.fit",
            )
            self.assertTrue(
                record["input_lineage"]["baseline"]["save_verified"]
            )
            self.assertEqual(persisted, record)
            self.assertEqual(stage5_handoff.verify_stage5_handoff(pipeline), record)

    def test_current_run_handoff_rejects_failed_component_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self._pipeline(Path(temporary_directory))
            (pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT).write_bytes(
                b"invalid-transaction"
            )
            input_lineage = self._current_input_lineage(pipeline)

            record = stage5_handoff.freeze_stage5_handoff(
                pipeline,
                origin=stage5_handoff.CURRENT_RUN_ORIGIN,
                stage_status="degraded",
                deconvolution_integrity_ok=False,
                denoise_integrity_ok=True,
                formal_eligible=True,
                input_lineage=input_lineage,
            )

            self.assertFalse(record["accepted"])
            self.assertFalse(record["deconvolution_integrity_ok"])
            with self.assertRaises(stage5_handoff.Stage5HandoffError):
                stage5_handoff.verify_stage5_handoff(pipeline)

    def test_current_run_handoff_rejects_revoked_formal_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self._pipeline(Path(temporary_directory))
            (pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT).write_bytes(
                b"review-only-linear"
            )
            input_lineage = self._current_input_lineage(pipeline)

            record = stage5_handoff.freeze_stage5_handoff(
                pipeline,
                origin=stage5_handoff.CURRENT_RUN_ORIGIN,
                stage_status="degraded",
                deconvolution_integrity_ok=True,
                denoise_integrity_ok=True,
                formal_eligible=False,
                input_lineage=input_lineage,
            )

            self.assertFalse(record["accepted"])
            self.assertFalse(record["formal_eligible"])
            self.assertEqual(
                record["reason_code"],
                stage5_handoff.REASON_FORMAL_INELIGIBLE,
            )
            with self.assertRaises(stage5_handoff.Stage5HandoffError):
                stage5_handoff.verify_stage5_handoff(pipeline)

    def test_handoff_rejects_artifact_changed_after_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self._pipeline(Path(temporary_directory))
            artifact = pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT
            artifact.write_bytes(b"original")
            input_lineage = self._current_input_lineage(pipeline)
            stage5_handoff.freeze_stage5_handoff(
                pipeline,
                origin=stage5_handoff.CURRENT_RUN_ORIGIN,
                stage_status="ok",
                deconvolution_integrity_ok=True,
                denoise_integrity_ok=True,
                formal_eligible=True,
                input_lineage=input_lineage,
            )
            artifact.write_bytes(b"changed")

            with self.assertRaisesRegex(
                stage5_handoff.Stage5HandoffError,
                "changed after handoff freeze",
            ):
                stage5_handoff.verify_stage5_handoff(pipeline)

    def test_handoff_is_bound_to_the_current_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self._pipeline(Path(temporary_directory), run_id="run-a")
            (pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT).write_bytes(
                b"run-a"
            )
            input_lineage = self._current_input_lineage(pipeline)
            stage5_handoff.freeze_stage5_handoff(
                pipeline,
                origin=stage5_handoff.CURRENT_RUN_ORIGIN,
                stage_status="ok",
                deconvolution_integrity_ok=True,
                denoise_integrity_ok=True,
                formal_eligible=True,
                input_lineage=input_lineage,
            )
            pipeline._run_id = "run-b"

            with self.assertRaisesRegex(
                stage5_handoff.Stage5HandoffError,
                "run_id mismatch",
            ):
                stage5_handoff.verify_stage5_handoff(pipeline)

    def test_current_run_handoff_rejects_unverified_input_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self._pipeline(Path(temporary_directory))
            (pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT).write_bytes(
                b"unbound-current-buffer"
            )
            lineage = stage5_handoff.freeze_stage5_input_lineage(
                pipeline,
                upstream_loaded=False,
                baseline_saved=False,
            )

            record = stage5_handoff.freeze_stage5_handoff(
                pipeline,
                origin=stage5_handoff.CURRENT_RUN_ORIGIN,
                stage_status="degraded",
                deconvolution_integrity_ok=True,
                denoise_integrity_ok=True,
                formal_eligible=True,
                input_lineage=lineage,
            )

            self.assertFalse(record["accepted"])
            self.assertFalse(record["input_integrity_ok"])
            self.assertIn("input lineage contract", record["detail"])

    def test_handoff_rejects_input_artifacts_changed_after_freeze(self) -> None:
        for artifact_name in (
            stage5_handoff.STAGE5_UPSTREAM_ARTIFACT,
            stage5_handoff.STAGE5_INPUT_ARTIFACT,
        ):
            with self.subTest(artifact=artifact_name), tempfile.TemporaryDirectory() as td:
                pipeline = self._pipeline(Path(td))
                (pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT).write_bytes(
                    b"linear"
                )
                input_lineage = self._current_input_lineage(pipeline)
                stage5_handoff.freeze_stage5_handoff(
                    pipeline,
                    origin=stage5_handoff.CURRENT_RUN_ORIGIN,
                    stage_status="ok",
                    deconvolution_integrity_ok=True,
                    denoise_integrity_ok=True,
                    formal_eligible=True,
                    input_lineage=input_lineage,
                )
                (pipeline.process_dir / artifact_name).write_bytes(b"tampered")

                with self.assertRaisesRegex(
                    stage5_handoff.Stage5HandoffError,
                    "changed or disappeared after input freeze",
                ):
                    stage5_handoff.verify_stage5_handoff(pipeline)

    def test_verified_resume_records_manifest_and_config_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pipeline = self._pipeline(Path(temporary_directory), run_id="resume-run")
            artifact = pipeline.process_dir / stage5_handoff.STAGE5_SOURCE_ARTIFACT
            artifact.write_bytes(b"verified-resume")
            provenance = {
                "verified": True,
                "checkpoint": "stage5",
                "state": "linear",
                "semantic_context_status": "verified",
                "run_manifest_hash": "manifest-sha256",
                "config_fingerprint": "config-sha256",
                "actual_sha256": run_manifest.sha256_file(artifact),
            }
            pipeline._trusted_input_provenance = dict(provenance)

            record = stage5_handoff.freeze_stage5_handoff(
                pipeline,
                origin=stage5_handoff.VERIFIED_RESUME_ORIGIN,
                stage_status="verified_resume",
                deconvolution_integrity_ok=True,
                denoise_integrity_ok=True,
                formal_eligible=True,
                provenance=provenance,
            )

            self.assertTrue(record["accepted"])
            self.assertEqual(record["resume"]["checkpoint"], "stage5")
            self.assertEqual(
                record["resume"]["run_manifest_hash"],
                "manifest-sha256",
            )
            self.assertEqual(
                record["resume"]["config_fingerprint"],
                "config-sha256",
            )
            self.assertEqual(stage5_handoff.verify_stage5_handoff(pipeline), record)


if __name__ == "__main__":
    unittest.main()
