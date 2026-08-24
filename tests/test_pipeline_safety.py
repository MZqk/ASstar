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
    clamp_saturation_boost,
    color_safety_limits,
    should_bypass_star_separation,
    should_skip_final_denoise,
)
from models import StarSeparationState  # noqa: E402
from stages.stage7_stretching import run_stage7_stretching  # noqa: E402
from stages.stage6_star_separation import run_stage6_star_separation  # noqa: E402
from stages.stage8_nebula_enhancement import run_stage8_nebula_enhancement  # noqa: E402
from stages.stage9_star_remixing import run_stage9_star_remixing  # noqa: E402
from sirilpy.exceptions import CommandError  # noqa: E402


class _Log:
    def stage_start(self, _name: str) -> None:
        return None

    def stage_end(self, _name: str) -> float:
        return 0.01

    def info(self, _message: str) -> None:
        return None

    def warn(self, _message: str) -> None:
        return None

    def error(self, _message: str) -> None:
        return None


class _StarPreservePipeline:
    def __init__(self, process_dir: Path) -> None:
        self.process_dir = process_dir
        self.process_dir.mkdir(parents=True, exist_ok=True)
        (self.process_dir / "stage5_linear.fit").write_bytes(b"FIT")
        self.cfg = SimpleNamespace(
            stage6_star_preserve_target_bypass_enabled=True,
        )
        self.log = _Log()
        self.stretched_name = None
        self.starless_file = None
        self.starmask_file = None
        self.pipeline_policy = {}
        self.color_calibration_report = {}
        self.reports: dict[str, object] = {}
        self.records: list[tuple[str, str, float, str]] = []
        self.commands: list[tuple[object, ...]] = []
        self._star_preserve_target_bypass = False
        self._stage7_starless_skipped = False
        self._stage8_conservative_mode = False
        self._stage8_final_source = "stage8_enhanced"
        self._stage8_final_quality = "unknown"
        self._stage8_fallback_used = False
        self._stage9_final_source = ""
        self._review_requirements: dict[tuple[int, str], dict[str, object]] = {}

    def _clear_stage_reviews(self, stage: int) -> None:
        self._review_requirements = {
            key: value
            for key, value in self._review_requirements.items()
            if key[0] != int(stage)
        }

    def _require_review(
        self,
        stage: int,
        code: str,
        details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        requirement = {
            "stage": int(stage),
            "code": str(code),
            "details": dict(details or {}),
        }
        self._review_requirements[(int(stage), str(code))] = requirement
        return requirement

    def _stage_review_reasons(self, stage: int) -> list[str]:
        return [
            str(value["code"])
            for key, value in self._review_requirements.items()
            if key[0] == int(stage)
        ]

    def _review_requirements_payload(self) -> list[dict[str, object]]:
        return [
            dict(value)
            for _key, value in sorted(self._review_requirements.items())
        ]

    def cmd_with_check(self, *args: object) -> None:
        self.commands.append(args)
        if args and args[0] == "save" and len(args) > 1:
            (self.process_dir / f"{args[1]}.fit").write_bytes(b"FIT")

    def _save_stage_output(self, stem: str) -> bool:
        (self.process_dir / f"{stem}.fit").write_bytes(b"FIT")
        return True

    def _active_target_type(self) -> str:
        return "globular_cluster"

    @staticmethod
    def _short_text(value: object, limit: int) -> str:
        return str(value)[:limit]

    def _write_stage_json(self, name: str, payload: object) -> None:
        self.reports[name] = payload

    def _record_stage(
        self,
        name: str,
        status: str,
        elapsed: float,
        message: str,
        **_metadata: object,
    ) -> None:
        self.records.append((name, status, elapsed, message))


class _StarFailurePipeline(_StarPreservePipeline):
    def __init__(self, process_dir: Path) -> None:
        super().__init__(process_dir)
        self.cfg = SimpleNamespace(
            stage6_star_preserve_target_bypass_enabled=True,
            stage7_quality_retry_max=0,
        )

    def _active_target_type(self) -> str:
        return "large_galaxy"

    def _find_plugin_script(self, _candidates: object):
        return None

    def _run_first_available_command(self, *_args: object, **_kwargs: object):
        return None

    def _stage7_update_star_remix_from_quality(self, _quality: object):
        self._stage9_star_intensity_scale = 1.0
        self._stage9_star_intensity_reason = "star separation unavailable"
        return {"scale": 1.0, "reason": self._stage9_star_intensity_reason}

    def _export_sasp_exchange_files(self) -> None:
        return None

    def _short_text(self, value: object, limit: int) -> str:
        return str(value)[:limit]


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
        self.assertTrue(
            should_skip_final_denoise(
                stage5_denoise_applied=True,
                stage8_final_quality="star_preserve_secondary_nebulosity",
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

            run_stage6_star_separation(pipeline)

            self.assertTrue(pipeline._star_preserve_target_bypass)
            self.assertTrue(pipeline._stage7_starless_skipped)
            self.assertEqual(
                pipeline._star_separation_state,
                StarSeparationState.TARGET_BYPASS.value,
            )
            self.assertTrue((pipeline.process_dir / "stage6_passthrough.fit").exists())
            self.assertFalse((pipeline.process_dir / "starless.fit").exists())
            quality_report = pipeline.reports["stage6_starless_quality.json"]
            self.assertEqual(quality_report["mode"], "star_preserve_target_bypass")
            self.assertFalse(any(command[0] == "script" for command in pipeline.commands))

            pipeline.stretched_name = "stage7_stretched"
            (pipeline.process_dir / "stage7_stretched.fit").write_bytes(b"FIT")
            pipeline._stage7_stretch_accepted = True
            pipeline._stage7_stretch_output = "stage7_stretched"
            run_stage8_nebula_enhancement(pipeline)

            self.assertEqual(pipeline._stage8_final_quality, "star_preserve_bypass")
            stage8_report = pipeline.reports["stage8_enhancement_report.json"]
            self.assertEqual(stage8_report["mode"], "star_preserve_target_bypass")
            self.assertEqual(pipeline.records[-1][1], "skipped")

            run_stage9_star_remixing(pipeline)

            self.assertEqual(pipeline.records[-1][1], "skipped")
            self.assertEqual(pipeline._stage9_final_source, "stage8_enhanced")
            self.assertTrue((pipeline.process_dir / "stage9_remixed.fit").exists())

    def test_target_bypass_rejected_stage7_keeps_with_stars_review_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _StarPreservePipeline(Path(tmpdir))
            run_stage6_star_separation(pipeline)
            pipeline._stage7_stretch_accepted = False
            pipeline._stage7_stretch_output = None
            pipeline._stage7_review_source = "stage7_review_with_stars"
            (pipeline.process_dir / "stage7_review_with_stars.fit").write_bytes(
                b"FIT"
            )

            command_start = len(pipeline.commands)
            run_stage8_nebula_enhancement(pipeline)

            stage8_commands = pipeline.commands[command_start:]
            loaded_stems = [
                str(command[1])
                for command in stage8_commands
                if command and command[0] == "load" and len(command) > 1
            ]
            self.assertEqual(loaded_stems, ["stage7_review_with_stars"])
            self.assertNotIn("starless", loaded_stems)
            self.assertNotIn("stage6_starless", loaded_stems)
            self.assertEqual(
                pipeline._stage8_final_source,
                "stage8_review_with_stars",
            )
            stage8_report = pipeline.reports["stage8_enhancement_report.json"]
            self.assertEqual(
                stage8_report["handoff"]["reason_code"],
                "stage7_stretch_not_accepted_target_bypass",
            )
            self.assertTrue(stage8_report["handoff"]["restricted_downstream"])

            run_stage9_star_remixing(pipeline)

            self.assertTrue(pipeline._stage9_bypassed_bad_starless)
            self.assertFalse(pipeline._stage9_stars_required)
            self.assertEqual(
                pipeline._stage9_stars_application_mode,
                "not_required_star_preserve_review",
            )
            self.assertEqual(pipeline.records[-1][1], "degraded")

    def test_target_bypass_rejected_stage7_withholds_when_no_with_stars_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _StarPreservePipeline(Path(tmpdir))
            run_stage6_star_separation(pipeline)
            pipeline._stage7_stretch_accepted = False
            pipeline._stage7_stretch_output = None
            pipeline._stage7_review_source = "missing_stage7_review_with_stars"
            pipeline._stage6_passthrough_source = "missing_stage6_passthrough"
            (pipeline.process_dir / "stage6_passthrough.fit").unlink()
            original_cmd_with_check = pipeline.cmd_with_check

            def checked_cmd_with_check(*args: object) -> None:
                if args and args[0] == "load" and len(args) > 1:
                    pipeline.commands.append(args)
                    source_path = pipeline.process_dir / f"{args[1]}.fit"
                    if not source_path.exists():
                        raise CommandError(
                            f"missing with-stars source: {args[1]}"
                        )
                    return
                original_cmd_with_check(*args)

            pipeline.cmd_with_check = checked_cmd_with_check

            command_start = len(pipeline.commands)
            run_stage8_nebula_enhancement(pipeline)

            loaded_stems = [
                str(command[1])
                for command in pipeline.commands[command_start:]
                if command and command[0] == "load" and len(command) > 1
            ]
            self.assertNotIn("starless", loaded_stems)
            self.assertNotIn("stage6_starless", loaded_stems)
            self.assertEqual(pipeline.records[-1][1], "failed")
            self.assertIsNone(
                pipeline.reports["stage8_enhancement_report.json"]["source"]
            )
            self.assertEqual(
                pipeline._stage8_final_source,
                "stage8_review_with_stars",
            )

            run_stage9_star_remixing(pipeline)

            self.assertTrue(pipeline._stage9_bypassed_bad_starless)
            self.assertTrue(pipeline._stage9_output_withheld)
            self.assertEqual(
                pipeline._stage9_stars_application_mode,
                "with_stars_review_source_unavailable",
            )
            self.assertEqual(pipeline.records[-1][1], "failed")
            self.assertFalse(
                (pipeline.process_dir / "stage9_remixed.fit").exists()
            )

    def test_star_tool_failure_uses_with_stars_review_path_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _StarFailurePipeline(Path(tmpdir))
            stale_starless_artifacts = (
                "starless.fit",
                "starless_ai_best_initial.fit",
                "stage6_starless.fit",
                "stage6_starless_repaired.fit",
                "starmask_raw.fit",
            )
            for name in stale_starless_artifacts:
                (pipeline.process_dir / name).write_bytes(b"STALE")

            run_stage6_star_separation(pipeline)

            self.assertEqual(
                pipeline._star_separation_state,
                StarSeparationState.TOOL_FAILED.value,
            )
            self.assertIsNone(pipeline.starless_file)
            self.assertIsNone(pipeline.starmask_file)
            self.assertTrue((pipeline.process_dir / "stage6_passthrough.fit").exists())
            self.assertTrue(
                all(
                    not (pipeline.process_dir / name).exists()
                    for name in stale_starless_artifacts
                )
            )

            run_stage7_stretching(pipeline)
            self.assertFalse(pipeline._stage7_stretch_accepted)
            self.assertEqual(
                pipeline._stage7_review_source,
                "stage7_review_with_stars",
            )

            run_stage8_nebula_enhancement(pipeline)
            stage8_report = pipeline.reports["stage8_enhancement_report.json"]
            self.assertEqual(
                stage8_report["mode"],
                "with_stars_review_passthrough",
            )
            self.assertFalse(stage8_report["starless_enhancement_applied"])
            self.assertEqual(
                pipeline._stage8_final_source,
                "stage8_review_with_stars",
            )

            run_stage9_star_remixing(pipeline)
            self.assertTrue(pipeline._stage9_stars_required)
            self.assertFalse(pipeline._stage9_stars_applied)
            self.assertEqual(
                pipeline._stage9_stars_application_mode,
                "not_applied_star_separation_unavailable",
            )
            self.assertTrue((pipeline.process_dir / "stage9_remixed.fit").exists())


if __name__ == "__main__":
    unittest.main()
