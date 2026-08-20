#!/usr/bin/env python3
"""Contract tests for the opt-in real Siril Stage 1-10 runner."""
from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from pipeline import outcome, run_manifest, task_plan
from tests import real_siril_stage1_10_e2e as real_e2e


class RealSirilStage110E2EContractTests(unittest.TestCase):
    @staticmethod
    def _write_result(root: Path, result: dict[str, object]) -> None:
        result.pop("manifest_hash", None)
        result["manifest_hash"] = run_manifest.canonical_payload_hash(result)
        run_manifest.atomic_write_json(root / "pipeline-result.json", result)

    @classmethod
    def _write_complete_run(cls, root: Path) -> dict[str, object]:
        output = root / "M_42_processed_final.fit"
        output.write_bytes(b"verified-final-fits")
        plan = task_plan.build_processing_plan(
            run_id="real-e2e-contract",
            generated_at="2026-08-14T00:00:00Z",
            input_record={"fingerprint": "real-e2e-input"},
            input_state="linear",
            input_trust="recognized",
        )
        run_manifest.atomic_write_json(root / "processing-plan.json", plan)
        actual_steps = [
            {
                "stage": stage,
                "name": f"阶段 {stage}: contract",
                "status": "ok",
                "execution": "completed",
                "fallback_used": False,
                "upstream_passthrough": False,
                "review_required": False,
                "review_reasons": [],
                "issues": [],
            }
            for stage in real_e2e.EXPECTED_STAGES
        ]
        result: dict[str, object] = {
            "schema": outcome.PIPELINE_RESULT_SCHEMA_V2,
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "failure_reason": None,
            "review_requirements": [],
            "color_calibration": {
                "requires_review": False,
                "stage7_forced_delivery": False,
            },
            "star_separation": {
                "state": "accepted",
                "stars_required": True,
                "stars_applied": True,
                "output_contains_stars": True,
                "output_withheld": False,
                "starmask_borderline_review_required": False,
                "psf_review_required": False,
                "remix_formally_accepted": True,
                "delivery_contract_accepted": True,
                "review_candidate_selected": False,
                "final_source": "stage9_remixed",
            },
            "actual_steps": actual_steps,
            "outputs": {
                output.name: run_manifest.file_record(output, base_dir=root)
            },
        }
        result.update(outcome.summarize_outcome(actual_steps, []))
        cls._write_result(root, result)
        process_dir = root / "process"
        process_dir.mkdir()
        run_manifest.atomic_write_json(
            process_dir / "final_quality_report.json",
            {
                "schema": real_e2e.FINAL_QUALITY_SCHEMA,
                "status": "ok",
                "final_quality": "ok",
                "needs_conservative_rerun": False,
                "issues": [],
                "hard_issues": [],
                "warnings": ["允许不阻断交付的 advisory"],
            },
        )
        attempt_id = "initial-contract-attempt"
        pair_id = "a" * 64
        selected_pointer = {
            "schema": real_e2e.SYQON_SELECTED_SCHEMA,
            "attempt_id": attempt_id,
            "pair_id": pair_id,
            "generation": "raw",
            "stop_reason": "CONTRACT_VALID_PAIR_COMMITTED",
            "attempt_manifest": (
                ".stage6_syqon/initial-contract-attempt/attempt-manifest.json"
            ),
            "starless": ".stage6_syqon/initial-contract-attempt/starless.fit",
            "starmask_raw": ".stage6_syqon/initial-contract-attempt/starmask.fit",
        }
        run_manifest.atomic_write_json(
            process_dir / "stage6_syqon_selected.json",
            selected_pointer,
        )
        run_manifest.atomic_write_json(
            process_dir / "stage6_syqon_exchange.json",
            {
                "schema": real_e2e.SYQON_EXCHANGE_SCHEMA,
                "status": "accepted",
                "accepted": True,
                "attempt_id": attempt_id,
                "selected_attempt_id": attempt_id,
                "pair_id": pair_id,
                "generation": "raw",
                "stop_reason": "CONTRACT_VALID_PAIR_COMMITTED",
                "files": {"source": {"sha256": "source-sha256"}},
                "assets": {"model": {"sha256": "model-sha256"}},
                "runtime": {"python": "3.12"},
                "worker": {"status": "success"},
                "attempts": [
                    {
                        "attempt_id": attempt_id,
                        "status": "accepted",
                        "accepted": True,
                    }
                ],
                "generations": [
                    {
                        "attempt_id": attempt_id,
                        "pair_id": pair_id,
                        "generation": "raw",
                        "stop_reason": "CONTRACT_VALID_PAIR_COMMITTED",
                    }
                ],
                "selected_pointer": selected_pointer,
            },
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
        self.assertIn("formal_delivery=accepted", details)
        self.assertIn("final_quality=ok", details)

    def test_verifier_rejects_legacy_or_non_success_result(self) -> None:
        cases = (
            ("schema", outcome.PIPELINE_RESULT_SCHEMA_V1, "v2 schema"),
            ("status", "partial_success", "不是正式 success"),
            ("status", "review_required", "不是正式 success"),
        )
        for field, value, expected_detail in cases:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                result = self._write_complete_run(root)
                result[field] = value
                self._write_result(root, result)

                verified, details = real_e2e.verify_e2e_artifacts(root)

                self.assertFalse(verified)
                self.assertTrue(
                    any(expected_detail in item for item in details),
                    details,
                )

    def test_verifier_rejects_run_level_review_error_or_fallback_flags(self) -> None:
        for field in (
            "review_required",
            "had_errors",
            "had_fatal_errors",
            "had_degradations",
            "had_fallbacks",
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                result = self._write_complete_run(root)
                result[field] = True
                self._write_result(root, result)

                verified, details = real_e2e.verify_e2e_artifacts(root)

                self.assertFalse(verified)
                self.assertTrue(any(field in item for item in details), details)

    def test_verifier_rejects_malformed_top_level_issue(self) -> None:
        cases = ({"code": "missing_severity"}, {"severity": "catastrophic"})
        for issue in cases:
            with (
                self.subTest(issue=issue),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                result = self._write_complete_run(root)
                result["issues"] = [issue]
                self._write_result(root, result)

                verified, details = real_e2e.verify_e2e_artifacts(root)

                self.assertFalse(verified)
                self.assertTrue(
                    any("issues 不是规范对象数组" in item for item in details),
                    details,
                )

    def test_verifier_rejects_nonempty_review_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._write_complete_run(root)
            result["review_requirements"] = [
                {"stage": 9, "code": "review_candidate", "details": {}}
            ]
            self._write_result(root, result)

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertFalse(verified)
        self.assertTrue(any("review_requirements" in item for item in details))

    def test_verifier_rejects_missing_formal_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._write_complete_run(root)
            steps = result["actual_steps"]
            self.assertIsInstance(steps, list)
            result["actual_steps"] = steps[:-1]
            self._write_result(root, result)

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertFalse(verified)
        self.assertTrue(any("未覆盖完整 Stage 1-10" in item for item in details))

    def test_verifier_rejects_duplicate_or_out_of_order_stage_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._write_complete_run(root)
            steps = result["actual_steps"]
            self.assertIsInstance(steps, list)
            self.assertIsInstance(steps[1], dict)
            steps[1]["stage"] = 1
            self._write_result(root, result)

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertFalse(verified)
        self.assertTrue(any("顺序必须严格为 1-10" in item for item in details))

    def test_verifier_rejects_nonformal_stage_semantics(self) -> None:
        cases = (
            ("status", "degraded", "status=degraded"),
            ("execution", "safe_passthrough", "execution=safe_passthrough"),
            ("fallback_used", True, "fallback_used=false"),
            ("upstream_passthrough", True, "upstream_passthrough=false"),
            ("review_required", True, "review_required=false"),
            ("review_reasons", ["manual_review"], "review_reasons"),
            (
                "issues",
                [
                    {
                        "stage": 1,
                        "component": "contract",
                        "severity": "error",
                        "code": "recovered_error",
                        "recovered": True,
                        "message": "recovered error still blocks formal acceptance",
                    }
                ],
                "error/fatal issues",
            ),
        )
        for field, value, expected_detail in cases:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                result = self._write_complete_run(root)
                steps = result["actual_steps"]
                self.assertIsInstance(steps, list)
                self.assertIsInstance(steps[0], dict)
                steps[0][field] = value
                self._write_result(root, result)

                verified, details = real_e2e.verify_e2e_artifacts(root)

                self.assertFalse(verified)
                self.assertTrue(
                    any(expected_detail in item for item in details),
                    details,
                )

    def test_verifier_rejects_nonaccepted_final_quality(self) -> None:
        cases = (
            ({"status": "accepted"}, "final-quality.v2"),
            (
                {
                    "schema": real_e2e.FINAL_QUALITY_SCHEMA,
                    "status": "needs_conservative_rerun",
                    "final_quality": "poor",
                    "needs_conservative_rerun": True,
                    "issues": ["noise_regression"],
                    "hard_issues": ["noise_regression"],
                },
                "conservative rerun",
            ),
        )
        for quality, expected_detail in cases:
            with (
                self.subTest(quality=quality),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self._write_complete_run(root)
                run_manifest.atomic_write_json(
                    root / "process/final_quality_report.json",
                    quality,
                )

                verified, details = real_e2e.verify_e2e_artifacts(root)

                self.assertFalse(verified)
                self.assertTrue(
                    any(expected_detail in item for item in details),
                    details,
                )

    def test_verifier_rejects_incomplete_star_delivery_contract(self) -> None:
        cases = (
            ("stars_applied", False, "stars_applied=true"),
            ("output_withheld", True, "output_withheld=false"),
            (
                "delivery_contract_accepted",
                False,
                "delivery_contract_accepted=true",
            ),
        )
        for field, value, expected_detail in cases:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                result = self._write_complete_run(root)
                star_separation = result["star_separation"]
                self.assertIsInstance(star_separation, dict)
                star_separation[field] = value
                self._write_result(root, result)

                verified, details = real_e2e.verify_e2e_artifacts(root)

                self.assertFalse(verified)
                self.assertTrue(
                    any(expected_detail in item for item in details),
                    details,
                )

    def test_verifier_rejects_review_only_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._write_complete_run(root)
            review_output = root / "result_review_final.fit"
            review_output.write_bytes(b"review-only-fits")
            result["outputs"] = {
                review_output.name: run_manifest.file_record(
                    review_output,
                    base_dir=root,
                )
            }
            self._write_result(root, result)

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertFalse(verified)
        self.assertTrue(any("review-only 输出" in item for item in details))

    def test_verifier_rejects_output_path_outside_work_dir(self) -> None:
        for path_kind in ("absolute", "parent"):
            with (
                self.subTest(path_kind=path_kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                container = Path(directory)
                root = container / "work"
                root.mkdir()
                result = self._write_complete_run(root)
                outside = container / "outside_processed_final.fit"
                outside.write_bytes(b"stale-external-fits")
                record = run_manifest.file_record(outside)
                record["path"] = (
                    str(outside)
                    if path_kind == "absolute"
                    else "../outside_processed_final.fit"
                )
                result["outputs"] = {outside.name: record}
                self._write_result(root, result)

                verified, details = real_e2e.verify_e2e_artifacts(root)

                self.assertFalse(verified)
                expected = "绝对路径" if path_kind == "absolute" else "逃逸工作目录"
                self.assertTrue(any(expected in item for item in details), details)

    def test_verifier_accepts_committed_derived_syqon_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete_run(root)
            exchange_path = root / "process/stage6_syqon_exchange.json"
            exchange = run_manifest.load_json(exchange_path)
            self.assertIsInstance(exchange, dict)
            pointer = dict(exchange["selected_pointer"])
            pointer.update(
                {
                    "attempt_id": "clean-contract-attempt",
                    "pair_id": "b" * 64,
                    "generation": "clean",
                    "stop_reason": "DERIVED_GENERATION_COMMITTED",
                    "attempt_manifest": (
                        ".stage6_syqon/clean-contract-attempt/attempt-manifest.json"
                    ),
                    "starless": ".stage6_syqon/clean-contract-attempt/starless.fit",
                    "starmask_clean": (
                        ".stage6_syqon/clean-contract-attempt/starmask.fit"
                    ),
                }
            )
            exchange.update(
                {
                    "pair_id": pointer["pair_id"],
                    "generation": pointer["generation"],
                    "stop_reason": pointer["stop_reason"],
                    "selected_pointer": pointer,
                }
            )
            generations = exchange["generations"]
            self.assertIsInstance(generations, list)
            generations.append(
                {
                    "attempt_id": pointer["attempt_id"],
                    "pair_id": pointer["pair_id"],
                    "generation": pointer["generation"],
                    "stop_reason": pointer["stop_reason"],
                }
            )
            run_manifest.atomic_write_json(exchange_path, exchange)
            run_manifest.atomic_write_json(
                root / "process/stage6_syqon_selected.json",
                pointer,
            )

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertTrue(verified, details)

    def test_verifier_rejects_accepted_syqon_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete_run(root)
            run_manifest.atomic_write_json(
                root / "process/stage6_syqon_exchange.json",
                {
                    "schema": real_e2e.SYQON_EXCHANGE_SCHEMA,
                    "status": "accepted",
                    "accepted": True,
                },
            )

            verified, details = real_e2e.verify_e2e_artifacts(root)

        self.assertFalse(verified)
        self.assertTrue(any("provenance" in item for item in details), details)

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

    def test_core_smoke_command_forwards_external_resource_root(self) -> None:
        args = Namespace(
            work_dir=Path("/tmp/real-e2e-work"),
            siril_app=Path("/Applications/Siril.app"),
            siril_seed=Path("/tmp/siril-seed"),
            runtime_home=Path("/tmp/runtime-home"),
            offline_resource_root=Path("/Volumes/starun-e2e-resources"),
        )

        command = real_e2e.build_core_smoke_command(args)

        option_index = command.index("--offline-resource-root")
        self.assertEqual(
            command[option_index + 1],
            "/Volumes/starun-e2e-resources",
        )
        self.assertIn("--offline", command)


if __name__ == "__main__":
    unittest.main()
