#!/usr/bin/env python3
"""Fail-closed tests for the Qt-free run presentation contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from gui.history_store import HistoryStoreError, load_verified_run_bundle
from gui.run_presentation import RunOutcome, build_run_presentation
from pipeline import run_manifest, task_plan
from pipeline.task_workspace import (
    begin_task_run,
    build_source_record,
    ensure_task_workspace,
)


class GuiRunPresentationTests(unittest.TestCase):
    def _run(self, root: Path):
        source = root / "capture" / "M81.fit"
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
            created_at="2026-08-26T01:00:00Z",
        )
        run = begin_task_run(
            workspace=workspace,
            source_record=source_record,
            run_id="run-presentation",
            generated_at="2026-08-26T01:01:00Z",
        )
        return source_record, workspace, run

    @staticmethod
    def _write_plan(
        source_record,
        run,
        *,
        input_state: str = "linear",
        output: Mapping[str, Any] | None = None,
    ):
        run_payload = run_manifest.load_json(run.manifest_path)
        plan = task_plan.build_processing_plan(
            run_id=run.run_id,
            generated_at="2026-08-26T01:02:00Z",
            input_record={"fingerprint": source_record["fingerprint"]},
            input_state=input_state,
            input_trust="recognized",
            metadata={
                "task_run_manifest_hash": run_payload["manifest_hash"],
            },
            output=output,
        )
        run_manifest.atomic_write_json(run.root / "processing-plan.json", plan)
        return plan

    @staticmethod
    def _write_result(
        run,
        *,
        plan_hash: str,
        outputs: Mapping[str, Mapping[str, Any]],
        step_status: str = "ok",
        review_requirements: tuple[Mapping[str, Any], ...] = (),
        failure_reason: str | None = None,
        schema: str = "starun.pipeline-result.v2",
        include_delivery_gates: bool = True,
        include_formal_outputs: bool = True,
        formal_outputs: tuple[str, ...] | None = None,
        formal_count: int | None = None,
        legacy_delivery_contract: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": schema,
            "run_id": run.run_id,
            "plan_hash": plan_hash,
            "status": "success",
            "actual_steps": [
                {
                    "stage": 10,
                    "name": "阶段 10: 导出与报告",
                    "status": step_status,
                    "execution": "completed",
                    "fallback_used": step_status == "degraded",
                    "issues": [],
                }
            ],
            "outputs": dict(outputs),
        }
        if schema == "starun.pipeline-result.v2":
            payload["review_requirements"] = [
                dict(value) for value in review_requirements
            ]
            if include_delivery_gates:
                if formal_outputs is None:
                    formal_outputs = tuple(
                        str(name)
                        for name in outputs
                        if Path(str(name)).suffix.casefold()
                        in {
                            ".fit",
                            ".fits",
                            ".fts",
                            ".xisf",
                            ".tif",
                            ".tiff",
                            ".png",
                            ".jpg",
                            ".jpeg",
                        }
                        and "review" not in str(name).casefold()
                    )
                artifact_gate: dict[str, Any] = {
                    "accepted": True,
                    "formal_count": (
                        len(formal_outputs)
                        if formal_count is None
                        else formal_count
                    ),
                }
                if include_formal_outputs:
                    artifact_gate["formal_outputs"] = list(formal_outputs)
                payload["delivery_gates"] = {
                    "schema": "starun.final-delivery-gates.v1",
                    "legacy_delivery_contract": legacy_delivery_contract,
                    "scientific": {"accepted": True},
                    "presentation": {"accepted": True},
                    "artifacts": artifact_gate,
                    "review": {"accepted": not review_requirements},
                    "formal_delivery_accepted": not review_requirements,
                }
        if failure_reason is not None:
            payload["failure_reason"] = failure_reason
        payload["manifest_hash"] = run_manifest.canonical_payload_hash(payload)
        run_manifest.atomic_write_json(run.root / "pipeline-result.json", payload)
        return payload

    def test_success_has_verified_formal_delivery_and_manifest_only_preview(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            preview = run.root / "result_processed_display_srgb.png"
            preview.write_bytes(b"verified-preview")
            formal = run.root / "result_processed.tif"
            formal.write_bytes(b"verified-formal")
            unlisted = run.root / "result_processed_newer.png"
            unlisted.write_bytes(b"not-in-result-manifest")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    preview.name: run_manifest.file_record(
                        preview,
                        base_dir=run.root,
                    ),
                    formal.name: run_manifest.file_record(
                        formal,
                        base_dir=run.root,
                    ),
                },
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(
                bundle,
                fallback_status="failed",
            )

        self.assertTrue(bundle.plan_verified)
        self.assertTrue(bundle.result_verified)
        self.assertTrue(bundle.lineage_verified)
        self.assertEqual(bundle.verified_png, preview.resolve())
        self.assertNotIn(unlisted.resolve(), [item.path for item in bundle.verified_outputs])
        self.assertEqual(presentation.status, RunOutcome.SUCCESS)
        self.assertEqual(presentation.output_kind, "formal")
        self.assertTrue(presentation.delivery_eligible)
        self.assertTrue(presentation.download_enabled)
        self.assertIs(bundle.processing_plan, bundle.plan)
        self.assertIs(bundle.pipeline_result, bundle.result)

    def test_target_named_output_is_formal_for_verified_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            target_named = run.root / "M81_LRGB_20260826.tif"
            target_named.write_bytes(b"target-named-formal-result")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    target_named.name: run_manifest.file_record(
                        target_named,
                        base_dir=run.root,
                    )
                },
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(bundle.verified_outputs[0].kind, "formal")
        self.assertEqual(presentation.status, RunOutcome.SUCCESS)
        self.assertEqual(presentation.output_kind, "formal")
        self.assertTrue(presentation.delivery_eligible)

    def test_only_artifact_allowlist_image_is_promoted_to_formal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            formal = run.root / "M81_verified_formal.tif"
            formal.write_bytes(b"decoded-pixel-identity-verified")
            sha_only = run.root / "M81_sha_only_preview.png"
            sha_only.write_bytes(b"file-sha-only")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    formal.name: run_manifest.file_record(
                        formal,
                        base_dir=run.root,
                    ),
                    sha_only.name: run_manifest.file_record(
                        sha_only,
                        base_dir=run.root,
                    ),
                },
                formal_outputs=(formal.name,),
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        kinds = {output.name: output.kind for output in bundle.verified_outputs}
        self.assertEqual(kinds[formal.name], "formal")
        self.assertEqual(kinds[sha_only.name], "auxiliary")
        self.assertIsNone(bundle.verified_png)
        self.assertTrue(presentation.delivery_eligible)
        self.assertEqual(presentation.formal_output_names, (formal.name,))
        self.assertIsNone(presentation.preview_path)

    def test_phantom_allowlist_and_wrong_formal_count_block_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            formal = run.root / "result_processed_display_srgb.png"
            formal.write_bytes(b"verified-existing-formal")
            phantom_name = "missing_formal.tif"
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    formal.name: run_manifest.file_record(
                        formal,
                        base_dir=run.root,
                    )
                },
                formal_outputs=(formal.name, phantom_name),
                formal_count=1,
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(bundle.verified_outputs[0].kind, "formal")
        self.assertEqual(presentation.status, RunOutcome.SUCCESS)
        self.assertFalse(presentation.delivery_eligible)
        self.assertEqual(presentation.output_kind, "review")
        self.assertEqual(presentation.preview_path, formal.resolve())
        self.assertTrue(
            any(
                requirement.get("code") == "delivery_gates_rejected"
                for requirement in presentation.review_requirements
            )
        )

    def test_missing_artifact_allowlist_is_legacy_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            image = run.root / "result_processed.png"
            image.write_bytes(b"sha-verified-but-not-formally-identified")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    image.name: run_manifest.file_record(
                        image,
                        base_dir=run.root,
                    )
                },
                include_formal_outputs=False,
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(bundle.verified_outputs[0].kind, "auxiliary")
        self.assertIsNone(bundle.verified_png)
        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertFalse(presentation.delivery_eligible)
        self.assertEqual(presentation.output_kind, "none")
        self.assertEqual(presentation.formal_output_names, ())
        self.assertTrue(
            any(
                requirement.get("code") == "legacy_delivery_contract"
                for requirement in presentation.review_requirements
            )
        )

    def test_success_with_review_only_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            review = run.root / "result_review.png"
            review.write_bytes(b"review-only-output")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    review.name: run_manifest.file_record(
                        review,
                        base_dir=run.root,
                    )
                },
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertTrue(bundle.lineage_verified)
        self.assertEqual(bundle.verified_outputs[0].kind, "review")
        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertEqual(presentation.output_kind, "review")
        self.assertFalse(presentation.delivery_eligible)
        self.assertNotIn("安全下载", presentation.summary)

    def test_review_png_cannot_become_formal_preview_beside_formal_tiff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            formal = run.root / "result_processed.tif"
            formal.write_bytes(b"formal-tiff")
            review = run.root / "result_review.png"
            review.write_bytes(b"review-png")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    formal.name: run_manifest.file_record(
                        formal,
                        base_dir=run.root,
                    ),
                    review.name: run_manifest.file_record(
                        review,
                        base_dir=run.root,
                    ),
                },
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertTrue(presentation.delivery_eligible)
        self.assertIsNone(bundle.verified_png)
        self.assertIsNone(presentation.preview_path)

    def test_partial_success_remains_formally_deliverable_without_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            formal = run.root / "result_processed.png"
            formal.write_bytes(b"degraded-but-formal")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    formal.name: run_manifest.file_record(
                        formal,
                        base_dir=run.root,
                    )
                },
                step_status="degraded",
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(bundle.result["status"], "partial_success")
        self.assertEqual(presentation.status, RunOutcome.PARTIAL_SUCCESS)
        self.assertTrue(presentation.delivery_eligible)
        self.assertEqual(presentation.output_kind, "formal")
        self.assertIn("正式结果可用", presentation.title)

    def test_missing_dual_gate_contract_is_legacy_and_not_deliverable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            formal = run.root / "result_processed.tif"
            formal.write_bytes(b"legacy-formal-output")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    formal.name: run_manifest.file_record(
                        formal,
                        base_dir=run.root,
                    )
                },
                include_delivery_gates=False,
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(bundle.verified_outputs[0].kind, "auxiliary")
        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertFalse(presentation.delivery_eligible)
        self.assertEqual(presentation.output_kind, "none")
        self.assertTrue(
            any(
                requirement.get("code") == "legacy_delivery_contract"
                for requirement in presentation.review_requirements
            )
        )

    def test_explicit_legacy_dual_gate_contract_is_not_deliverable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            formal = run.root / "result_processed.tif"
            formal.write_bytes(b"explicit-legacy-formal-output")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    formal.name: run_manifest.file_record(
                        formal,
                        base_dir=run.root,
                    )
                },
                legacy_delivery_contract=True,
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(presentation.status, RunOutcome.SUCCESS)
        self.assertFalse(presentation.delivery_eligible)
        self.assertFalse(presentation.download_enabled)
        self.assertEqual(presentation.output_kind, "review")
        self.assertTrue(
            any(
                requirement.get("code") == "legacy_delivery_contract"
                for requirement in presentation.review_requirements
            )
        )

    def test_frozen_review_only_plan_overrides_success_result(self) -> None:
        cases = (
            ("linear", {"review_only": True}),
            ("nonlinear", None),
        )
        for input_state, output in cases:
            with self.subTest(input_state=input_state, output=output):
                with tempfile.TemporaryDirectory() as td:
                    source_record, workspace, run = self._run(Path(td))
                    plan = self._write_plan(
                        source_record,
                        run,
                        input_state=input_state,
                        output=output,
                    )
                    formal = run.root / "result_processed.tif"
                    formal.write_bytes(b"must-remain-review-only")
                    self._write_result(
                        run,
                        plan_hash=plan["plan_hash"],
                        outputs={
                            formal.name: run_manifest.file_record(
                                formal,
                                base_dir=run.root,
                            )
                        },
                    )

                    bundle = load_verified_run_bundle(
                        workspace.root,
                        run.run_id,
                    )
                    presentation = build_run_presentation(bundle)

                self.assertEqual(
                    presentation.status,
                    RunOutcome.REVIEW_REQUIRED,
                )
                self.assertEqual(presentation.output_kind, "review")
                self.assertFalse(presentation.delivery_eligible)
                self.assertTrue(
                    any(
                        requirement.get("code")
                        == "frozen_plan_review_only"
                        for requirement in presentation.review_requirements
                    )
                )

    def test_review_required_uses_verified_review_png_and_never_delivers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            review = run.root / "result_review_display_srgb.png"
            review.write_bytes(b"verified-review")
            processed = run.root / "result_processed.png"
            processed.write_bytes(b"also-verified")
            unlisted = run.root / "result_review_newer.png"
            unlisted.write_bytes(b"unlisted")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    processed.name: run_manifest.file_record(
                        processed,
                        base_dir=run.root,
                    ),
                    review.name: run_manifest.file_record(
                        review,
                        base_dir=run.root,
                    ),
                },
                review_requirements=(
                    {
                        "stage": 9,
                        "code": "user_preserve_with_stars",
                        "details": {},
                    },
                ),
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(bundle.verified_png, review.resolve())
        self.assertNotEqual(bundle.verified_png, unlisted.resolve())
        self.assertEqual(presentation.status, RunOutcome.REVIEW_REQUIRED)
        self.assertEqual(presentation.output_kind, "review")
        self.assertFalse(presentation.delivery_eligible)
        self.assertEqual(presentation.review_requirements[0]["stage"], 9)

    def test_failed_signed_result_stays_failed_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={},
                step_status="failed",
                failure_reason="Stage 10 export failed",
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(
                bundle,
                fallback_status="success",
            )

        self.assertEqual(presentation.status, RunOutcome.FAILED)
        self.assertFalse(presentation.delivery_eligible)
        self.assertIn("Stage 10 export failed", presentation.summary)

    def test_signed_user_interruption_is_presented_as_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={},
                failure_reason="user interrupted",
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(
                bundle,
                fallback_status="stopped",
            )

        self.assertEqual(bundle.result["status"], "failed")
        self.assertEqual(presentation.status, RunOutcome.STOPPED)
        self.assertFalse(presentation.delivery_eligible)
        self.assertIn("中止", presentation.title)

    def test_success_with_only_auxiliary_output_cannot_deliver(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            diagnostics = run.root / "quality-report.json"
            diagnostics.write_text("{}", encoding="utf-8")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    diagnostics.name: run_manifest.file_record(
                        diagnostics,
                        base_dir=run.root,
                    )
                },
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertEqual(bundle.verified_outputs[0].kind, "auxiliary")
        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertEqual(presentation.output_kind, "none")
        self.assertFalse(presentation.delivery_eligible)

    def test_output_sha_tamper_blocks_success_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            plan = self._write_plan(source_record, run)
            preview = run.root / "result_processed.png"
            preview.write_bytes(b"before")
            self._write_result(
                run,
                plan_hash=plan["plan_hash"],
                outputs={
                    preview.name: run_manifest.file_record(
                        preview,
                        base_dir=run.root,
                    )
                },
            )
            preview.write_bytes(b"after")

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(
                bundle,
                fallback_status="success",
            )

        self.assertEqual(bundle.verified_outputs, ())
        self.assertIsNone(bundle.verified_png)
        self.assertTrue(any("SHA-256" in value for value in bundle.integrity_errors))
        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertFalse(presentation.delivery_eligible)
        self.assertIsNotNone(presentation.integrity_error)

    def test_plan_hash_mismatch_keeps_preview_but_blocks_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_record, workspace, run = self._run(Path(td))
            self._write_plan(source_record, run)
            preview = run.root / "result_processed.png"
            preview.write_bytes(b"verified-output")
            self._write_result(
                run,
                plan_hash="not-the-plan-hash",
                outputs={
                    preview.name: run_manifest.file_record(
                        preview,
                        base_dir=run.root,
                    )
                },
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(bundle)

        self.assertFalse(bundle.lineage_verified)
        self.assertEqual(bundle.verified_png, preview.resolve())
        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertEqual(presentation.output_kind, "review")
        self.assertFalse(presentation.delivery_eligible)

    def test_signed_v1_result_normalizes_but_untrusted_plan_cannot_deliver(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _source_record, workspace, run = self._run(Path(td))
            preview = run.root / "result_processed.png"
            preview.write_bytes(b"legacy-preview")
            self._write_result(
                run,
                plan_hash="legacy-plan",
                outputs={
                    preview.name: run_manifest.file_record(
                        preview,
                        base_dir=run.root,
                    )
                },
                schema="starun.pipeline-result.v1",
            )

            bundle = load_verified_run_bundle(workspace.root, run.run_id)
            presentation = build_run_presentation(
                bundle,
                fallback_status="success",
            )

        self.assertEqual(bundle.result["source_schema"], "starun.pipeline-result.v1")
        self.assertFalse(bundle.plan_verified)
        self.assertFalse(bundle.lineage_verified)
        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertFalse(presentation.delivery_eligible)

    def test_fallback_success_cannot_replace_a_verified_bundle(self) -> None:
        presentation = build_run_presentation(None, fallback_status="success")

        self.assertEqual(presentation.status, RunOutcome.VERIFICATION_FAILED)
        self.assertFalse(presentation.delivery_eligible)
        self.assertFalse(presentation.download_enabled)

    def test_run_manifest_identity_is_a_hard_loader_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _source_record, workspace, run = self._run(Path(td))
            payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            payload["run_id"] = "different-run"
            payload.pop("manifest_hash")
            payload["manifest_hash"] = run_manifest.canonical_payload_hash(payload)
            run_manifest.atomic_write_json(run.manifest_path, payload)

            with self.assertRaisesRegex(HistoryStoreError, "身份不匹配"):
                load_verified_run_bundle(workspace.root, run.run_id)


if __name__ == "__main__":
    unittest.main()
