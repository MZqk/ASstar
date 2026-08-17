#!/usr/bin/env python3
"""Canonical v2 outcome truth-table and v1 compatibility tests."""

from __future__ import annotations

import unittest
import sys
import types
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

_installed_sirilpy_stub = "sirilpy" not in sys.modules
if _installed_sirilpy_stub:
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class _SirilError(RuntimeError):
        pass

    exceptions.CommandError = _SirilError
    exceptions.DataError = _SirilError
    exceptions.SirilConnectionError = _SirilError
    exceptions.SirilError = _SirilError
    sirilpy.exceptions = exceptions
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions

from models import StageResult
from outcome import (
    normalize_pipeline_result,
    normalize_run_state,
    summarize_outcome,
)
from processor_runtime import ProcessorRuntimeMixin

if _installed_sirilpy_stub:
    sys.modules.pop("sirilpy.exceptions", None)
    sys.modules.pop("sirilpy", None)


class OutcomeSemanticsTests(unittest.TestCase):
    def test_status_truth_table(self) -> None:
        cases = (
            (
                "normal",
                [{"stage": 1, "status": "ok"}],
                [],
                [],
                "success",
            ),
            (
                "optional_skip_and_safe_passthrough",
                [
                    {"stage": 1, "status": "skipped", "execution": "skipped"},
                    {
                        "stage": 2,
                        "status": "ok",
                        "execution": "safe_passthrough",
                        "upstream_passthrough": True,
                    },
                ],
                [],
                [],
                "success",
            ),
            (
                "fallback",
                [{"stage": 5, "status": "ok", "fallback_used": True}],
                [],
                [],
                "partial_success",
            ),
            (
                "recovered_error",
                [{"stage": 5, "status": "ok"}],
                [],
                [
                    {
                        "stage": 5,
                        "component": "denoise",
                        "severity": "error",
                        "code": "plugin_execution_failed",
                        "recovered": True,
                        "message": "plugin failed; Siril fallback accepted",
                    }
                ],
                "partial_success",
            ),
            (
                "review_wins_over_degraded",
                [{"stage": 9, "status": "degraded"}],
                [
                    {
                        "stage": 9,
                        "code": "user_preserve_with_stars",
                        "details": {},
                    }
                ],
                [],
                "review_required",
            ),
            (
                "fatal_wins_over_review",
                [{"stage": 10, "status": "failed"}],
                [{"stage": 9, "code": "psf_review", "details": {}}],
                [],
                "failed",
            ),
        )
        for label, steps, reviews, issues, expected in cases:
            with self.subTest(label=label):
                summary = summarize_outcome(
                    steps,
                    reviews,
                    extra_issues=issues,
                )
                self.assertEqual(summary["status"], expected)

        recovered = summarize_outcome(cases[3][1], [], extra_issues=cases[3][3])
        self.assertTrue(recovered["had_errors"])
        self.assertFalse(recovered["had_fatal_errors"])
        self.assertFalse(recovered["had_fallbacks"])

    def test_review_registry_deduplicates_by_stage_and_code(self) -> None:
        runtime = ProcessorRuntimeMixin()
        runtime._require_review(2, "residual", {"pass": 1})
        runtime._require_review(2, "residual", {"pass": 2})
        runtime._require_review(9, "residual", {"kind": "psf"})

        self.assertEqual(
            runtime._review_requirements_payload(),
            [
                {"stage": 2, "code": "residual", "details": {"pass": 2}},
                {"stage": 9, "code": "residual", "details": {"kind": "psf"}},
            ],
        )
        runtime._clear_stage_reviews(2)
        self.assertEqual(runtime._stage_review_reasons(2), [])
        self.assertEqual(runtime._stage_review_reasons(9), ["residual"])

    def test_late_review_registration_updates_owning_stage_result(self) -> None:
        runtime = ProcessorRuntimeMixin()
        runtime.results = [StageResult("阶段 1: 前期准备", "ok")]

        runtime._require_review(1, "input_state_review_required")

        self.assertEqual(
            runtime.results[0].review_reasons,
            ["input_state_review_required"],
        )

    def test_v1_result_normalizes_fallback_and_stage3_review(self) -> None:
        normalized = normalize_pipeline_result(
            {
                "schema": "starun.pipeline-result.v1",
                "status": "review_required",
                "review_requirements": {
                    "stage3_background_review_required": True,
                },
                "actual_steps": [
                    {
                        "name": "阶段 5: 线性整理",
                        "status": "ok",
                        "fallback_used": True,
                    }
                ],
            }
        )

        self.assertEqual(normalized["schema"], "starun.pipeline-result.v2")
        self.assertEqual(normalized["actual_steps"][0]["status"], "degraded")
        self.assertTrue(normalized["had_fallbacks"])
        self.assertEqual(
            normalized["review_requirements"],
            [
                {
                    "stage": 3,
                    "code": "stage3_background_review_required",
                    "details": {},
                    "legacy_inferred": True,
                }
            ],
        )

    def test_v1_run_state_normalizes_recovered_error(self) -> None:
        normalized = normalize_run_state(
            {
                "schema": "starun.run-state.v1",
                "status": "CompletedWithWarning",
                "had_errors": True,
                "errors": ["legacy recovered error"],
            }
        )

        self.assertEqual(normalized["status"], "partial_success")
        self.assertTrue(normalized["had_errors"])
        self.assertFalse(normalized["had_fatal_errors"])
        self.assertTrue(normalized["issues"][0]["recovered"])

    def test_stage_result_canonicalizes_fallback(self) -> None:
        result = StageResult("阶段 5: 线性整理", "ok", fallback_used=True)
        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.display_status, "degraded")
        with self.assertRaises(ValueError):
            StageResult("阶段 X", "warning")


if __name__ == "__main__":
    unittest.main()
