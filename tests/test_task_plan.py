#!/usr/bin/env python3
"""Regression tests for shared, frozen task-plan routing."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_manifest  # noqa: E402
import task_plan  # noqa: E402


class TaskPlanTests(unittest.TestCase):
    def _fingerprints(
        self,
        *,
        input_fingerprint: str = "input-sha256",
        stage_config: dict[int, dict[str, object]] | None = None,
    ) -> dict[str, dict[str, object]]:
        return task_plan.build_resume_fingerprints(
            input_fingerprint=input_fingerprint,
            stage_config=stage_config
            or {
                1: {"import": "fits"},
                2: {"crop": "auto"},
                3: {"background": "safe"},
                4: {"color": "physical"},
                5: {"denoise": "conservative"},
            },
        )

    def _plan(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "run_id": "run-1",
            "generated_at": "2026-08-04T00:00:00Z",
            "input_record": {
                "path": "/source/master.xisf",
                "fingerprint": "input-sha256",
            },
            "input_state": "linear",
            "input_trust": "recognized",
        }
        values.update(overrides)
        return task_plan.build_processing_plan(**values)

    @staticmethod
    def _actions(plan: dict[str, object]) -> list[str]:
        return [
            str(step["action"])
            for step in plan["planned_steps"]  # type: ignore[index,union-attr]
        ]

    def test_external_linear_master_runs_stage_1_through_10(self) -> None:
        plan = self._plan()

        self.assertEqual(self._actions(plan), ["execute"] * 10)
        self.assertFalse(plan["route"]["review_only"])  # type: ignore[index]

    def test_verified_stage5_checkpoint_resumes_at_stage6(self) -> None:
        plan = self._plan(
            input_trust="verified",
            resume_after_stage=5,
            checkpoint_fingerprints=self._fingerprints(),
        )

        self.assertEqual(
            self._actions(plan),
            ["verified"] * 5 + ["execute"] * 5,
        )

    def test_nonlinear_and_unknown_inputs_are_review_only(self) -> None:
        for state in ("nonlinear", "unknown"):
            with self.subTest(state=state):
                plan = self._plan(
                    input_state=state,
                    input_trust="review_required",
                )
                self.assertEqual(
                    self._actions(plan),
                    ["execute", "execute"]
                    + ["input_state_guard"] * 7
                    + ["review_export"],
                )
                self.assertTrue(plan["route"]["review_only"])  # type: ignore[index]

    def test_resume_rejects_recognized_or_nonformal_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "verified task provenance"):
            self._plan(
                resume_after_stage=5,
                checkpoint_fingerprints=self._fingerprints(),
            )
        with self.assertRaisesRegex(ValueError, "formal resume boundary"):
            self._plan(
                input_trust="verified",
                resume_after_stage=4,
                checkpoint_fingerprints=self._fingerprints(),
            )

    def test_resume_requires_matching_checkpoint_contract_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "verified stage5 fingerprint"):
            self._plan(input_trust="verified", resume_after_stage=5)

        fingerprints = self._fingerprints()
        fingerprints["stage5"]["artifact"] = "result_linear.fit"
        with self.assertRaisesRegex(ValueError, "wrong artifact"):
            self._plan(
                input_trust="verified",
                resume_after_stage=5,
                checkpoint_fingerprints=fingerprints,
            )

    def test_latest_compatible_resume_follows_cumulative_configuration(self) -> None:
        baseline = self._fingerprints()
        stage4_changed = self._fingerprints(
            stage_config={
                1: {"import": "fits"},
                2: {"crop": "auto"},
                3: {"background": "safe"},
                4: {"color": "changed"},
                5: {"denoise": "conservative"},
            }
        )
        stage2_changed = self._fingerprints(
            stage_config={
                1: {"import": "fits"},
                2: {"crop": "changed"},
            }
        )
        input_changed = self._fingerprints(input_fingerprint="different-input")

        self.assertEqual(
            task_plan.latest_compatible_resume_stage(baseline, baseline),
            5,
        )
        self.assertEqual(
            task_plan.latest_compatible_resume_stage(baseline, stage4_changed),
            2,
        )
        self.assertEqual(
            task_plan.latest_compatible_resume_stage(baseline, stage2_changed),
            1,
        )
        self.assertIsNone(
            task_plan.latest_compatible_resume_stage(baseline, input_changed)
        )

    def test_plan_hash_route_and_contract_are_verified(self) -> None:
        plan = self._plan()
        self.assertTrue(task_plan.verify_processing_plan(plan)["verified"])

        hash_tampered = copy.deepcopy(plan)
        hash_tampered["route"]["input_state"] = "unknown"
        self.assertFalse(
            task_plan.verify_processing_plan(hash_tampered)["verified"]
        )

        route_tampered = copy.deepcopy(plan)
        route_tampered["route"]["input_state"] = "unknown"
        route_tampered["plan_hash"] = run_manifest.canonical_payload_hash(
            {key: value for key, value in route_tampered.items() if key != "plan_hash"}
        )
        self.assertIn(
            "steps do not match",
            task_plan.verify_processing_plan(route_tampered)["detail"],
        )

        contract_tampered = copy.deepcopy(plan)
        contract_tampered["pipeline_contract"]["stages"][0]["title"] = "篡改"
        contract_tampered["plan_hash"] = run_manifest.canonical_payload_hash(
            {
                key: value
                for key, value in contract_tampered.items()
                if key != "plan_hash"
            }
        )
        self.assertIn(
            "contract was modified",
            task_plan.verify_processing_plan(contract_tampered)["detail"],
        )


if __name__ == "__main__":
    unittest.main()
