#!/usr/bin/env python3
"""Tests for multimodal/human review evidence generation."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy" not in sys.modules:
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")
    enums = types.ModuleType("sirilpy.enums")

    class _SirilError(Exception):
        pass

    class _SirilConnectionError(_SirilError):
        pass

    class _SirilInterface:
        pass

    class _CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    exceptions.CommandError = _SirilError
    exceptions.DataError = _SirilError
    exceptions.SirilError = _SirilError
    exceptions.SirilConnectionError = _SirilConnectionError
    sirilpy.SirilInterface = _SirilInterface
    enums.CommandStatus = _CommandStatus
    sirilpy.exceptions = exceptions
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums

from review_bundle import (  # noqa: E402
    apply_visual_acceptance,
    create_image_review_bundle,
    prune_review_bundles,
)


class ReviewBundleTests(unittest.TestCase):
    def test_bundle_writes_previews_diffs_metrics_and_explicit_review_state(self) -> None:
        before = np.zeros((3, 24, 32), dtype=np.float32)
        before[:, 6:18, 8:24] = 0.10
        after = before.copy()
        after[0, 8:16, 10:22] += 0.05
        after[:, 10:14, 14:18] += 0.10

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "review"
            payload = create_image_review_bundle(
                before,
                after,
                output_dir=output_dir,
                stage_key="stage_test",
                source={"before_stem": "before", "after_stem": "after"},
                context={"target_type": "large_galaxy"},
                candidates=[
                    {"name": "candidate_a", "status": "ok"},
                    {"name": "candidate_b", "status": "rejected"},
                ],
                selected_candidate="candidate_a",
            )

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["visual_review"]["status"], "not_requested")
            self.assertFalse(payload["visual_review"]["acceptance_blocking"])
            self.assertEqual(payload["candidates"][0]["selection_status"], "selected")
            self.assertEqual(
                payload["candidates"][0]["visual_acceptance_status"], "not_requested"
            )
            self.assertEqual(
                payload["candidates"][1]["visual_acceptance_status"], "unavailable"
            )
            for preview_path in payload["previews"].values():
                self.assertTrue(Path(preview_path).is_file())
                self.assertGreater(Path(preview_path).stat().st_size, 0)
            compact_path = Path(payload["compact_preview"])
            self.assertTrue(compact_path.is_file())
            self.assertEqual(compact_path.read_bytes()[24], 8)
            report_path = Path(payload["report_path"])
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("features", report["metrics"]["delta"])
            self.assertIn("quality", report["metrics"]["delta"])
            self.assertEqual(report["context"]["target_type"], "large_galaxy")

            updated = apply_visual_acceptance(
                payload,
                {
                    "verdict": "accept",
                    "confidence": 0.91,
                    "summary": "looks safe",
                    "issues": [],
                    "recommended_parameter_ranges": {},
                },
                advisor_mode="multimodal",
            )
            self.assertEqual(updated["visual_review"]["status"], "accepted")
            self.assertEqual(
                updated["candidates"][0]["visual_acceptance_status"], "accepted"
            )
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["visual_review"]["status"], "accepted")

    def test_final_retention_keeps_only_problem_stage_compact_preview(self) -> None:
        before = np.zeros((3, 24, 32), dtype=np.float32)
        after = before.copy()
        after[:, 8:18, 9:23] = 0.15

        with tempfile.TemporaryDirectory() as tmpdir:
            review_root = Path(tmpdir) / "review_bundles"
            for stage_key in (
                "stage4_color_calibration",
                "stage5_linear_cleanup",
            ):
                create_image_review_bundle(
                    before,
                    after,
                    output_dir=review_root / stage_key,
                    stage_key=stage_key,
                    source={"before_stem": "before", "after_stem": "after"},
                )

            result = prune_review_bundles(
                review_root,
                pipeline_status="review_required",
                problem_stage_numbers={4},
            )

            stage4 = review_root / "stage4_color_calibration"
            stage5 = review_root / "stage5_linear_cleanup"
            self.assertEqual(
                {path.name for path in stage4.glob("*.png")},
                {"review.png"},
            )
            self.assertEqual(list(stage5.glob("*.png")), [])
            stage4_report = json.loads(
                (stage4 / "review.json").read_text(encoding="utf-8")
            )
            stage5_report = json.loads(
                (stage5 / "review.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                stage4_report["retention"]["mode"],
                "compact_problem_evidence",
            )
            self.assertEqual(stage5_report["retention"]["mode"], "metrics_only")
            self.assertNotIn("compact_preview", stage5_report)
            self.assertNotIn("compact_layout", stage5_report)
            self.assertEqual(
                set(stage4_report["previews"]),
                {"compact_review"},
            )
            self.assertGreater(len(result["removed_images"]), 0)

    def test_missing_synchronous_visual_verdict_is_not_left_pending(self) -> None:
        payload = {
            "visual_review": {},
            "candidates": [{"selection_status": "selected"}],
        }

        updated = apply_visual_acceptance(
            payload,
            None,
            advisor_mode="multimodal",
        )

        self.assertEqual(updated["visual_review"]["status"], "unavailable")
        self.assertEqual(
            updated["candidates"][0]["visual_acceptance_status"], "unavailable"
        )


if __name__ == "__main__":
    unittest.main()
