#!/usr/bin/env python3
"""Tests for cross-stage target, color, and denoise safety rules."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy" not in sys.modules:
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class _SirilError(Exception):
        pass

    exceptions.CommandError = _SirilError
    exceptions.DataError = _SirilError
    exceptions.SirilError = _SirilError
    sirilpy.exceptions = exceptions
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions

from pipeline_safety import (  # noqa: E402
    clamp_ai_color_adjustments,
    clamp_saturation_boost,
    color_safety_limits,
    should_bypass_star_separation,
    should_skip_final_denoise,
)
from stages.stage7_star_separation import run_stage7_star_separation  # noqa: E402
from stages.stage8_nebula_enhancement import run_stage8_nebula_enhancement  # noqa: E402
from stages.stage9_star_remixing import run_stage9_star_remixing  # noqa: E402


class _Log:
    def stage_start(self, _name: str) -> None:
        return None

    def stage_end(self, _name: str) -> float:
        return 0.01

    def info(self, _message: str) -> None:
        return None

    def warn(self, _message: str) -> None:
        return None


class _StarPreservePipeline:
    def __init__(self, process_dir: Path) -> None:
        self.process_dir = process_dir
        self.process_dir.mkdir(parents=True, exist_ok=True)
        (self.process_dir / "stage5_linear.fit").write_bytes(b"FIT")
        self.cfg = SimpleNamespace(
            star_separation_mode="linear_star_separation",
            stage6_star_preserve_target_bypass_enabled=True,
        )
        self.log = _Log()
        self.stretched_name = None
        self.starless_file = None
        self.starmask_file = None
        self.pipeline_policy = {}
        self.color_calibration_report = {}
        self.pre_starless_gate_report = {}
        self.reports: dict[str, object] = {}
        self.records: list[tuple[str, str, float, str]] = []
        self.commands: list[tuple[object, ...]] = []
        self._star_preserve_target_bypass = False
        self._stage7_starless_skipped = False
        self._stage8_conservative_mode = False
        self._stage8_final_source = "starless_enhanced"
        self._stage8_final_quality = "unknown"
        self._stage8_fallback_used = False
        self._stage9_final_source = ""

    def cmd_with_check(self, *args: object) -> None:
        self.commands.append(args)
        if args and args[0] == "save" and len(args) > 1:
            (self.process_dir / f"{args[1]}.fit").write_bytes(b"FIT")

    def _save_stage_output(self, stem: str) -> bool:
        (self.process_dir / f"{stem}.fit").write_bytes(b"FIT")
        if stem == "stage7_starless":
            (self.process_dir / "stage6_starless.fit").write_bytes(b"FIT")
        return True

    def _active_target_type(self) -> str:
        return "globular_cluster"

    def _write_stage_json(self, name: str, payload: object) -> None:
        self.reports[name] = payload

    def _record_stage(self, name: str, status: str, elapsed: float, message: str) -> None:
        self.records.append((name, status, elapsed, message))


class PipelineSafetyTests(unittest.TestCase):
    def test_star_subject_targets_bypass_star_separation(self) -> None:
        for target_type in (
            "globular_cluster",
            "open_cluster",
            "reflection_nebula_cluster",
        ):
            with self.subTest(target_type=target_type):
                self.assertTrue(should_bypass_star_separation(target_type))
        self.assertFalse(should_bypass_star_separation("large_galaxy"))
        self.assertFalse(
            should_bypass_star_separation("globular_cluster", enabled=False)
        )

    def test_color_report_limits_total_saturation_budget(self) -> None:
        policy = {
            "stage4_color": {
                "max_allowed_saturation_boost": 0.14,
                "red_gain_limit": 1.08,
                "blue_gain_limit": 0.90,
            }
        }
        report = {"policy_adjustments": {"reduce_saturation_boost": True}}
        limits = color_safety_limits(policy, report)

        self.assertAlmostEqual(limits["max_saturation_boost"], 0.07)
        self.assertAlmostEqual(
            clamp_saturation_boost(
                0.15,
                already_applied=0.05,
                limits=limits,
            ),
            0.02,
        )

    def test_stage11_color_adjustments_obey_remaining_budget_and_gain_caps(self) -> None:
        limits = {
            "max_saturation_boost": 0.10,
            "red_gain_limit": 1.05,
            "blue_gain_limit": 0.90,
        }
        adjusted = clamp_ai_color_adjustments(
            {
                "global_saturation_delta": 0.08,
                "red_balance_delta": 0.08,
                "blue_balance_delta": 0.08,
                "background_protection": 0.9,
            },
            already_applied=0.06,
            limits=limits,
        )

        self.assertAlmostEqual(adjusted["global_saturation_delta"], 0.04)
        self.assertAlmostEqual(adjusted["red_balance_delta"], 0.05)
        self.assertAlmostEqual(adjusted["blue_balance_delta"], 0.0)

    def test_final_denoise_skips_only_after_safe_later_quality(self) -> None:
        self.assertTrue(
            should_skip_final_denoise(
                stage5_denoise_applied=True,
                stage8_final_quality="ok",
                stage8_fallback_used=False,
            )
        )
        self.assertTrue(
            should_skip_final_denoise(
                stage5_denoise_applied=True,
                stage8_final_quality="star_preserve_bypass",
                stage8_fallback_used=False,
            )
        )
        self.assertFalse(
            should_skip_final_denoise(
                stage5_denoise_applied=True,
                stage8_final_quality="poor",
                stage8_fallback_used=True,
            )
        )

    def test_star_preserve_route_skips_starless_tool_and_stage8_enhancement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _StarPreservePipeline(Path(tmpdir))

            run_stage7_star_separation(pipeline)

            self.assertTrue(pipeline._star_preserve_target_bypass)
            self.assertTrue(pipeline._stage7_starless_skipped)
            quality_report = pipeline.reports["stage7_quality.json"]
            self.assertEqual(quality_report["mode"], "star_preserve_target_bypass")
            self.assertFalse(any(command[0] == "script" for command in pipeline.commands))

            pipeline.stretched_name = "stage7_stretched"
            (pipeline.process_dir / "stage7_stretched.fit").write_bytes(b"FIT")
            run_stage8_nebula_enhancement(pipeline)

            self.assertEqual(pipeline._stage8_final_quality, "star_preserve_bypass")
            stage8_report = pipeline.reports["stage8_enhancement_report.json"]
            self.assertEqual(stage8_report["mode"], "star_preserve_target_bypass")
            self.assertEqual(pipeline.records[-1][1], "skipped")

            run_stage9_star_remixing(pipeline)

            self.assertEqual(pipeline.records[-1][1], "skipped")
            self.assertEqual(pipeline._stage9_final_source, "stage8_enhanced")
            self.assertTrue((pipeline.process_dir / "stage9_remixed.fit").exists())


if __name__ == "__main__":
    unittest.main()
