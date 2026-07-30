#!/usr/bin/env python3
"""Focused tests for the PCC-only Stage 4 contract."""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _install_fake_sirilpy() -> None:
    if "sirilpy.exceptions" in sys.modules:
        return
    package = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class SirilError(Exception):
        pass

    class CommandError(SirilError):
        pass

    exceptions.SirilError = SirilError
    exceptions.CommandError = CommandError
    package.exceptions = exceptions
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions


_install_fake_sirilpy()
stage4 = importlib.import_module("stages.stage4_color_calibration")
target_profiler = importlib.import_module("target_profiler")


def _pipeline(shape=(3, 64, 64)):
    cfg = SimpleNamespace(
        stage4_pcc_timeout_sec=30,
        stage4_pcc_quality_gate_enabled=True,
        stage4_pcc_channel_gain_ratio_max=1.80,
        stage4_pcc_clip_growth_max=0.005,
        stage4_local_star_wb_enabled=True,
        stage4_local_star_wb_min_pixels=16,
        stage4_local_star_wb_gain_limit=1.20,
        stage4_local_star_mask_radius=2,
        stage4_local_star_mask_coverage_max=0.12,
    )
    return SimpleNamespace(
        cfg=cfg,
        siril=SimpleNamespace(get_image_shape=lambda: shape),
        target_profile={},
    )


class Stage4PccPolicyTests(unittest.TestCase):
    def test_object_header_m42_is_parsed_as_catalog_target(self):
        text = target_profiler._metadata_text({"OBJECT": "M 42"}, "")
        matched = target_profiler._catalog_name_match(
            target_profiler.load_catalog(),
            text,
        )
        self.assertIsNotNone(matched)
        item, confidence = matched
        self.assertEqual(item["name"], "M42")
        self.assertEqual(item["type"], "bright_emission_reflection_nebula")
        self.assertGreaterEqual(confidence, 0.90)

    def test_broadband_m42_remains_eligible_for_pcc(self):
        pipeline = _pipeline()
        pipeline.target_profile = {
            "target_name_guess": "M42",
            "target_type": "bright_emission_reflection_nebula",
        }
        policy = stage4._stage4_channel_policy(
            pipeline,
            {"OBJECT": "M 42", "FILTER": "LP"},
            checkpoint_loaded=True,
        )
        self.assertEqual(policy["kind"], "broadband_rgb_osc")
        self.assertEqual(policy["action"], "single_pcc")

    def test_dual_narrowband_skips_pcc_by_policy(self):
        pipeline = _pipeline()
        policy = stage4._stage4_channel_policy(
            pipeline,
            {"OBJECT": "M 42", "FILTER": "Ha+OIII dual-band"},
            checkpoint_loaded=True,
        )
        self.assertEqual(policy["kind"], "narrowband_composite")
        self.assertEqual(policy["action"], "skip_pcc_local_star_only")

    def test_unknown_linearity_preserves_input_for_review(self):
        policy = stage4._stage4_channel_policy(
            _pipeline(),
            {"OBJECT": "M 42"},
            checkpoint_loaded=False,
        )
        self.assertEqual(policy["kind"], "unknown")
        self.assertEqual(policy["action"], "preserve_input_review")

    def test_pcc_runner_is_exactly_one_attempt(self):
        pipeline = _pipeline()
        calls = []

        def run_once(*, timeout_sec: int, catalog: str):
            calls.append((timeout_sec, catalog))
            return False, "timeout"

        pipeline._run_stage4_pcc_once = run_once
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)
        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            ok, _detail, attempts = stage4._stage4_run_pcc(
                pipeline,
                phase="linear_broadband",
            )
        self.assertFalse(ok)
        self.assertEqual(calls, [(30, "gaia")])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["max_attempts"], 1)

    def test_local_star_restore_does_not_apply_global_white_balance(self):
        pipeline = _pipeline(shape=(3, 96, 96))
        y, x = np.mgrid[:96, :96]
        image = np.full((3, 96, 96), 0.02, dtype=np.float32)
        for cy, cx in ((20, 20), (20, 70), (48, 48), (72, 24), (72, 72)):
            star = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 3.0) * 0.55
            image[0] += star * 1.15
            image[1] += star * 0.90
            image[2] += star * 0.80

        restored, report = stage4._stage4_star_white_balance(image, pipeline)
        self.assertTrue(report["applied"])
        self.assertEqual(report["application"], "star_soft_mask_only")
        changed = np.max(np.abs(restored - image), axis=0) > 1e-7
        self.assertLess(float(np.mean(changed)), pipeline.cfg.stage4_local_star_mask_coverage_max)
        self.assertTrue(np.allclose(restored[:, 0, 0], image[:, 0, 0]))

    def test_m42_quality_gate_allows_red_subject_but_checks_background(self):
        pipeline = _pipeline()
        pipeline.target_profile = {
            "target_name_guess": "M42",
            "target_type": "bright_emission_reflection_nebula",
        }
        before = np.full((3, 64, 64), 0.05, dtype=np.float32)
        after = before.copy()
        after[0, 20:44, 20:44] *= 1.35
        accepted, report = stage4._stage4_pcc_quality_gate(before, after, pipeline)
        self.assertTrue(accepted, report)
        self.assertEqual(
            report["target_aware_profile"],
            "emission_nebula_red_dominance_allowed",
        )

    def test_failed_pcc_restores_pre_pcc_and_marks_review_required(self):
        image = np.full((3, 64, 64), 0.04, dtype=np.float32)
        saved = {"stage3_bgremoved": image.copy()}
        commands = []
        results = []

        class Log:
            def stage_start(self, _label):
                pass

            def stage_end(self, _label):
                return 0.1

            def info(self, _message):
                pass

            def warn(self, _message):
                pass

            def debug(self, _message):
                pass

        pipeline = _pipeline()
        pipeline.cfg.stage4_platesolve_enabled = True
        pipeline.cfg.max_retries = 0
        pipeline.log = Log()
        pipeline.current = image.copy()
        pipeline.process_dir = REPO_ROOT
        pipeline.work_dir = REPO_ROOT
        pipeline.source_file = REPO_ROOT / "M42.fit"
        pipeline.pipeline_policy = {}
        pipeline.target_profile = {
            "target_name_guess": "M42",
            "target_type": "bright_emission_reflection_nebula",
        }

        def command(*args, quiet=False):
            _ = quiet
            commands.append(tuple(args))
            if args[0] == "load":
                pipeline.current = saved[str(args[1])].copy()
            return True

        pipeline.cmd_with_check = command
        pipeline.siril = SimpleNamespace(
            get_image_shape=lambda: pipeline.current.shape,
            get_image_pixeldata=lambda preview=False: pipeline.current.copy(),
            set_image_pixeldata=lambda pixels: setattr(
                pipeline, "current", np.asarray(pixels).copy()
            ),
        )
        pipeline._save_stage_output = lambda stem: saved.setdefault(
            stem, pipeline.current.copy()
        ) is not None
        pipeline._read_fits_header_metadata = lambda *_args: {
            "OBJECT": "M 42",
            "FILTER": "LP",
            "CRVAL1": 83.822,
            "CRVAL2": -5.391,
        }
        pipeline._run_target_profile_preflight = lambda **_kwargs: ""
        pipeline._run_stage4_pcc_once = lambda **_kwargs: (False, "timeout")
        pipeline._write_stage_json = lambda *_args: None
        pipeline._record_stage = lambda *args: results.append(args)
        pipeline._active_policy_name = lambda: "bright_nebula_hdr_conservative"

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["pcc"]["max_attempts"], 1)
        self.assertEqual(len(report["pcc"]["attempts"]), 1)
        self.assertTrue(report["pcc"]["rollback"]["restored"])
        self.assertTrue(report["requires_review"])
        self.assertEqual(results[-1][1], "degraded")
        self.assertIn(("load", "stage4_pre_pcc"), commands)
        self.assertFalse(any(call and call[0] == "spcc" for call in commands))


if __name__ == "__main__":
    unittest.main()
