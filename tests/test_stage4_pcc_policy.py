#!/usr/bin/env python3
"""Focused tests for the SPCC-first Stage 4 color contract."""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import types
import tempfile
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
        stage4_spcc_enabled=True,
        stage4_spcc_timeout_sec=180,
        stage4_spcc_osc_sensor="Sony IMX585",
        stage4_spcc_osc_filter="",
        stage4_spcc_white_ref="Average Spiral Galaxy",
        stage4_spcc_limit_magnitude=10.5,
        stage4_spcc_narrowband_r_wavelength_nm=656.28,
        stage4_spcc_narrowband_r_bandwidth_nm=20.0,
        stage4_spcc_narrowband_g_wavelength_nm=500.70,
        stage4_spcc_narrowband_g_bandwidth_nm=30.0,
        stage4_spcc_narrowband_b_wavelength_nm=500.70,
        stage4_spcc_narrowband_b_bandwidth_nm=30.0,
        stage4_narrowband_normalization_enabled=True,
        stage4_nbn_mapping_confidence_min=0.85,
        stage4_pcc_timeout_sec=180,
        stage4_pcc_quality_gate_enabled=True,
        stage4_pcc_channel_gain_ratio_max=1.80,
        stage4_pcc_emission_balance_gain_ratio_max=4.0,
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


def _stage4_integration_fixture(*, filter_name: str = "LP"):
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

    def save(stem):
        saved[stem] = pipeline.current.copy()
        return True

    pipeline.cmd_with_check = command
    pipeline.siril = SimpleNamespace(
        get_image_shape=lambda: pipeline.current.shape,
        get_image_pixeldata=lambda preview=False: pipeline.current.copy(),
        set_image_pixeldata=lambda pixels: setattr(
            pipeline,
            "current",
            np.asarray(pixels).copy(),
        ),
    )
    pipeline._save_stage_output = save
    pipeline._read_fits_header_metadata = lambda *_args: {
        "OBJECT": "M 42",
        "FILTER": filter_name,
        "CRVAL1": 83.822,
        "CRVAL2": -5.391,
    }
    pipeline._run_target_profile_preflight = lambda **_kwargs: ""
    pipeline._write_stage_json = lambda *_args: None
    pipeline._record_stage = lambda *args, **metadata: results.append(
        (args, metadata)
    )
    pipeline._active_policy_name = lambda: "bright_nebula_hdr_conservative"
    return pipeline, saved, commands, results


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

    def test_object_header_m8_uses_packaged_lagoon_catalog_entry(self):
        text = target_profiler._metadata_text({"OBJECT": "M 8"}, "")
        matched = target_profiler._catalog_name_match(
            target_profiler.load_catalog(),
            text,
        )

        self.assertIsNotNone(matched)
        item, confidence = matched
        self.assertEqual(item["name"], "Lagoon Nebula")
        self.assertEqual(item["type"], "bright_emission_reflection_nebula")
        self.assertGreaterEqual(confidence, 0.90)

    def test_broadband_m42_routes_spcc_then_pcc(self):
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
        self.assertEqual(policy["action"], "spcc_then_pcc")

    def test_dual_narrowband_routes_physical_spcc_and_isolated_hoo(self):
        pipeline = _pipeline()
        policy = stage4._stage4_channel_policy(
            pipeline,
            {"OBJECT": "M 42", "FILTER": "Ha+OIII dual-band"},
            checkpoint_loaded=True,
        )
        self.assertEqual(policy["kind"], "narrowband_composite")
        self.assertEqual(policy["action"], "spcc_narrowband_with_isolated_hoo")

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
        self.assertEqual(calls, [(180, "gaia")])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["max_attempts"], 1)

    def test_spcc_runner_is_exactly_one_attempt(self):
        pipeline = _pipeline()
        calls = []

        def run_once(**kwargs):
            calls.append(kwargs)
            return False, "timeout"

        pipeline._run_stage4_spcc_once = run_once
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)
        args, _parameters = stage4._stage4_spcc_args(
            pipeline,
            {"FILTER": "LP"},
            {"kind": "broadband_rgb_osc"},
            catalog="gaia",
        )
        self.assertEqual(_parameters["limit_magnitude"], "10.5")
        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            ok, _detail, attempts = stage4._stage4_run_spcc(
                pipeline,
                phase="linear_broadband",
                catalog="gaia",
                args=args,
                narrowband=False,
            )

        self.assertFalse(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timeout_sec"], 180)
        self.assertEqual(calls[0]["catalog"], "gaia")
        self.assertEqual(attempts[0]["max_attempts"], 1)

    def test_spcc_database_provenance_records_selected_file_hashes(self):
        pipeline = _pipeline()
        commit = "1" * 40
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir)
            expected_hashes = {}
            for _kind, _label, relative, _required in stage4.SPCC_METADATA_FILES:
                path = database / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = f"metadata:{relative}\n".encode("utf-8")
                path.write_bytes(content)
                expected_hashes[relative] = hashlib.sha256(content).hexdigest()
            (database / stage4.SPCC_SEED_MARKER_NAME).write_text(
                f"source_commit={commit}\n",
                encoding="utf-8",
            )
            pipeline.spcc_database_dir = database

            status = stage4._stage4_spcc_database_status(pipeline)
            _args, parameters = stage4._stage4_spcc_args(
                pipeline,
                {"FILTER": "LP"},
                {"kind": "broadband_rgb_osc"},
                catalog="gaia",
            )
            selected = stage4._stage4_selected_spcc_metadata(
                status,
                parameters,
            )

        self.assertEqual(status["source_commit"], commit)
        self.assertEqual(
            status["source_commit_source"],
            stage4.SPCC_SEED_MARKER_NAME,
        )
        self.assertEqual(selected["source_commit"], commit)
        self.assertFalse(selected["unresolved"])
        self.assertEqual(len(selected["files"]), 3)
        for record in selected["files"]:
            self.assertEqual(record["sha256"], expected_hashes[record["path"]])

    def test_spcc_imprecise_advisory_defers_to_pixel_quality_gate(self):
        pipeline = _pipeline()
        advisory = (
            "Spectrophotometric Color Calibration succeeded. "
            "The photometric color calibration seems to have found an "
            "imprecise solution, consider correcting the image gradient first"
        )
        pipeline._run_stage4_spcc_once = lambda **_kwargs: (True, advisory)
        pipeline.log = SimpleNamespace(
            warn=lambda *_args: None,
            info=lambda *_args: None,
        )
        args, _parameters = stage4._stage4_spcc_args(
            pipeline,
            {"FILTER": "LP"},
            {"kind": "broadband_rgb_osc"},
            catalog="gaia",
        )

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            ok, _detail, attempts = stage4._stage4_run_spcc(
                pipeline,
                phase="linear_broadband",
                catalog="gaia",
                args=args,
                narrowband=False,
            )

        self.assertTrue(ok)
        self.assertEqual(attempts[0]["status"], "ok")
        self.assertEqual(
            attempts[0]["precision_warning"],
            "spcc_imprecise_solution",
        )
        self.assertEqual(
            attempts[0]["precision_warning_policy"],
            "defer_to_target_aware_pixel_quality_gate",
        )

    def test_pcc_runner_clamps_timeout_to_180_seconds(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_pcc_timeout_sec = 999
        calls = []
        pipeline._run_stage4_pcc_once = lambda **kwargs: (
            calls.append(kwargs) or False,
            "timeout",
        )
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            stage4._stage4_run_pcc(pipeline, phase="linear_broadband")

        self.assertEqual(calls[0]["timeout_sec"], 180)

    def test_local_gaia_catalog_drives_pcc_with_network_disabled(self):
        pipeline = _pipeline()
        calls = []
        pipeline._run_stage4_pcc_once = lambda **kwargs: (
            calls.append(kwargs) or True,
            "ok",
        )
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / stage4.LOCAL_ASTROMETRIC_FILENAME
            catalog.write_bytes(b"x" * stage4.MIN_LOCAL_CATALOG_FILE_BYTES)
            with patch.dict(
                os.environ,
                {
                    "SEESTAR_NETWORK_MODE": "0",
                    "SEESTAR_GAIA_ASTRO_CATALOG": str(catalog),
                },
                clear=False,
            ):
                self.assertEqual(
                    stage4._stage4_preferred_pcc_catalog(pipeline),
                    "localgaia",
                )
                ok, _detail, attempts = stage4._stage4_run_pcc(
                    pipeline,
                    phase="linear_broadband",
                    catalog="localgaia",
                )

        self.assertTrue(ok)
        self.assertEqual(calls, [{"timeout_sec": 180, "catalog": "localgaia"}])
        self.assertTrue(attempts[0]["offline"])

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
        self.assertEqual(
            set(report["post_calibration_checks"]),
            {
                "star_color_temperature_distribution",
                "background_color_difference",
                "target_color_drift",
            },
        )

    def test_emission_pcc_accepts_large_gain_only_after_verified_background_balance(self):
        pipeline = _pipeline()
        pipeline.target_profile = {
            "target_name_guess": "Lagoon Nebula",
            "target_type": "bright_emission_reflection_nebula",
        }
        before = np.empty((3, 64, 64), dtype=np.float32)
        before[0] = 0.060
        before[1] = 0.020
        before[2] = 0.018
        after = np.full((3, 64, 64), 0.030, dtype=np.float32)

        accepted, report = stage4._stage4_pcc_quality_gate(
            before,
            after,
            pipeline,
        )

        self.assertTrue(accepted, report)
        self.assertGreater(
            report["measurements"]["channel_gain_ratio"],
            pipeline.cfg.stage4_pcc_channel_gain_ratio_max,
        )
        self.assertIn(
            "large_gain_accepted_after_verified_background_balance",
            report["target_aware_exemptions"],
        )

    def test_emission_pcc_large_gain_is_rejected_when_background_remains_unbalanced(self):
        pipeline = _pipeline()
        pipeline.target_profile = {
            "target_name_guess": "Lagoon Nebula",
            "target_type": "bright_emission_reflection_nebula",
        }
        before = np.empty((3, 64, 64), dtype=np.float32)
        before[0] = 0.060
        before[1] = 0.020
        before[2] = 0.018
        after = np.empty((3, 64, 64), dtype=np.float32)
        after[0] = 0.030
        after[1] = 0.030
        after[2] = 0.010

        accepted, report = stage4._stage4_pcc_quality_gate(
            before,
            after,
            pipeline,
        )

        self.assertFalse(accepted)
        self.assertIn("channel_gain_ratio_exceeded", report["rejection_reasons"])
        self.assertIn("background_chroma_exceeded", report["rejection_reasons"])

    def test_spcc_then_failed_pcc_restores_checkpoint_and_marks_review(self):
        pipeline, _saved, commands, results = _stage4_integration_fixture()
        calibration_order = []
        pipeline._run_stage4_spcc_once = lambda **_kwargs: (
            calibration_order.append("spcc") or False,
            "spcc timeout",
        )
        pipeline._run_stage4_pcc_once = lambda **_kwargs: (
            calibration_order.append("pcc") or False,
            "pcc timeout",
        )

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(calibration_order, ["spcc", "pcc"])
        self.assertEqual(len(report["spcc"]["attempts"]), 1)
        self.assertEqual(report["pcc"]["max_attempts"], 1)
        self.assertEqual(len(report["pcc"]["attempts"]), 1)
        self.assertTrue(report["pcc"]["rollback"]["restored"])
        self.assertTrue(report["requires_review"])
        self.assertTrue(pipeline._stage4_color_review_required)
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertIn(("load", "stage4_pre_pcc"), commands)
        self.assertFalse(any(call and call[0] == "spcc" for call in commands))

    def test_spcc_success_does_not_invoke_pcc(self):
        pipeline, saved, commands, _results = _stage4_integration_fixture()
        pcc_calls = []

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc ok"

        pipeline._run_stage4_spcc_once = spcc_success
        pipeline._run_stage4_pcc_once = lambda **kwargs: (
            pcc_calls.append(kwargs) or True,
            "pcc should not run",
        )

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "SPCC")
        self.assertFalse(report["requires_review"])
        self.assertFalse(pipeline._stage4_color_review_required)
        self.assertFalse(pcc_calls)
        self.assertIn(("load", stage4.SPCC_CANDIDATE_STEM), commands)

    def test_spcc_failure_then_pcc_success_is_exception_fallback_without_review(self):
        pipeline, saved, _commands, results = _stage4_integration_fixture()
        calibration_order = []

        pipeline._run_stage4_spcc_once = lambda **_kwargs: (
            calibration_order.append("spcc") or False,
            "spcc timeout",
        )

        def pcc_success(**_kwargs):
            calibration_order.append("pcc")
            saved[stage4.PCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "pcc ok"

        pipeline._run_stage4_pcc_once = pcc_success

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(calibration_order, ["spcc", "pcc"])
        self.assertEqual(report["method"], "PCC")
        self.assertTrue(report["pcc"]["used"])
        self.assertEqual(report["pcc"]["role"], "exception_fallback_broadband_only")
        self.assertFalse(report["requires_review"])
        self.assertFalse(pipeline._stage4_color_review_required)
        self.assertEqual(results[-1][0][1], "ok")
        self.assertTrue(results[-1][1]["fallback_used"])
        self.assertEqual(
            results[-1][1]["reason_code"],
            "spcc_exception_pcc_fallback",
        )

    def test_narrowband_hoo_derivative_does_not_feed_main_pipeline(self):
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            filter_name="Ha+OIII dual-band"
        )
        spcc_calls = []

        def spcc_success(**kwargs):
            spcc_calls.append(kwargs)
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc narrowband ok"

        def hoo_derivative(_pipeline, _metadata):
            _pipeline.current = np.clip(_pipeline.current * 1.5, 0.0, 1.0)
            return True, {"accepted": True, "status": "accepted"}, "HOO accepted"

        pipeline._run_stage4_spcc_once = spcc_success
        pipeline._run_stage4_pcc_once = lambda **_kwargs: self.fail(
            "narrowband must not invoke PCC"
        )

        with (
            patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=False),
            patch.object(stage4, "_stage4_run_narrowband_normalization", hoo_derivative),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        self.assertTrue(spcc_calls[0]["narrowband"])
        self.assertTrue(
            np.allclose(
                saved["stage4_color"],
                saved[stage4.PHYSICAL_COLOR_STEM],
            )
        )
        self.assertFalse(
            np.allclose(
                saved[stage4.HOO_ARTISTIC_STEM],
                saved[stage4.PHYSICAL_COLOR_STEM],
            )
        )
        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "SPCC_NARROWBAND")
        self.assertFalse(report["artistic_hoo"]["feeds_main_pipeline"])
        self.assertEqual(
            report["artistic_hoo"]["physical_parent"],
            "stage4_physical_color.fit",
        )


if __name__ == "__main__":
    unittest.main()
