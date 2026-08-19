#!/usr/bin/env python3
"""Focused tests for the SPCC-first Stage 4 color contract."""

from __future__ import annotations

import hashlib
import importlib
import json
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

    class DataError(SirilError):
        pass

    exceptions.SirilError = SirilError
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    package.SirilInterface = object
    package.exceptions = exceptions
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions


_install_fake_sirilpy()
from dualband_palette import resolve_palette_selection  # noqa: E402

stage4 = importlib.import_module("stages.stage4_color_calibration")
stage8 = importlib.import_module("stages.stage8_nebula_enhancement")
target_profiler = importlib.import_module("target_profiler")


def _pipeline(shape=(3, 64, 64)):
    cfg = SimpleNamespace(
        stage4_spcc_enabled=True,
        stage4_spcc_timeout_sec=300,
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


def _channel_policy(pipeline, metadata, *, checkpoint_loaded):
    mapping = stage4.resolve_dual_narrowband_mapping(metadata)
    return stage4._stage4_channel_policy(
        pipeline,
        metadata,
        checkpoint_loaded=checkpoint_loaded,
        narrowband_mapping=mapping,
    )


def _stage4_integration_fixture(
    *,
    filter_name: str = "IRCUT",
    image: np.ndarray | None = None,
):
    if image is None:
        image = np.full((3, 64, 64), 0.04, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
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
    pipeline._review_requirements = {}
    pipeline._clear_stage_reviews = lambda stage: setattr(
        pipeline,
        "_review_requirements",
        {
            key: value
            for key, value in pipeline._review_requirements.items()
            if key[0] != int(stage)
        },
    )
    pipeline._require_review = lambda stage, code, details=None: (
        pipeline._review_requirements.setdefault(
            (int(stage), str(code)),
            {
                "stage": int(stage),
                "code": str(code),
                "details": dict(details or {}),
            },
        )
    )
    pipeline._stage_review_reasons = lambda stage: [
        value["code"]
        for key, value in pipeline._review_requirements.items()
        if key[0] == int(stage)
    ]
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
    pipeline.stage_json = {}
    pipeline._write_stage_json = lambda name, payload: pipeline.stage_json.__setitem__(
        name,
        payload,
    )
    pipeline._record_stage = lambda *args, **metadata: results.append(
        (args, metadata)
    )
    pipeline._active_policy_name = lambda: "bright_nebula_hdr_conservative"
    return pipeline, saved, commands, results


def _auto_reference_image() -> np.ndarray:
    height = width = 256
    y, x = np.mgrid[:height, :width]
    rng = np.random.default_rng(7)
    background = (
        0.025
        + 0.002 * x / width
        + 0.001 * y / height
        + rng.normal(0.0, 0.0003, (height, width))
    )
    image = np.stack(
        [background + 0.012, background + 0.006, background], axis=0
    ).astype(np.float32)
    subject = np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2.0 * 30.0**2)) * 0.12
    image += np.stack([subject * 1.2, subject * 0.8, subject * 0.6])
    for cy in (28, 65, 102, 154, 205, 230):
        for cx in (25, 73, 121, 175, 225):
            star = (
                np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 1.2**2))
                * 0.45
            )
            image += np.stack([star * 1.08, star, star * 0.92])
    return image


def _strict_bright_core_image(size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    background = (
        0.02
        + 0.002 * xx / float(size)
        + 0.001 * yy / float(size)
    )
    core = 0.50 * np.exp(
        -(
            (yy - size / 2.0) ** 2
            + (xx - size / 2.0) ** 2
        )
        / (2.0 * (size * 0.09) ** 2)
    )
    return np.stack(
        (
            background + core,
            background + 0.45 * core,
            background + 0.28 * core,
        )
    ).astype(np.float32)


def _strict_profile() -> dict:
    return {
        "target_name_guess": "M42",
        "target_type": "bright_emission_reflection_nebula",
        "secondary_labels": ["bright_core"],
        "features": {"bright_core": True},
        "risks": {"core_blowout": "high"},
    }


def _blue_flip_candidate(before: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rec709 = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    luminance = np.tensordot(rec709, before, axes=(0, 0))
    direction = np.asarray((0.05, 0.15, 0.80), dtype=np.float64)
    replacement = direction[:, None, None] * (
        luminance / float(direction @ rec709)
    )[None, :, :]
    candidate = before.copy()
    candidate[:, mask] = replacement[:, mask]
    return candidate.astype(np.float32)


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
        policy = _channel_policy(
            pipeline,
            {"OBJECT": "M 42", "FILTER": "IRCUT"},
            checkpoint_loaded=True,
        )
        self.assertEqual(policy["kind"], "broadband_rgb_osc")
        self.assertEqual(policy["action"], "spcc_then_pcc")

    def test_starun_lp_header_routes_narrowband_without_line_keywords(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_osc_sensor = ""
        metadata = {"TELESCOP": "Seestar S50", "FILTER": "LP      "}

        with patch.dict(os.environ, {}, clear=True):
            policy = _channel_policy(
                pipeline,
                metadata,
                checkpoint_loaded=True,
            )
            args, parameters = stage4._stage4_spcc_args(
                pipeline,
                metadata,
                policy,
                catalog="gaia",
            )

        self.assertEqual(policy["kind"], "narrowband_composite")
        self.assertEqual(parameters["device_profile_id"], "seestar_s50_imx462")
        self.assertEqual(
            parameters["mapping"]["evidence"],
            "verified_device_profile",
        )
        self.assertEqual(parameters["bandwidths_nm"]["r"], 20.0)
        self.assertIn("-narrowband", args)
        self.assertFalse(any("oscfilter" in arg for arg in args))

    def test_imx585_bare_lp_routes_frozen_mapping_to_narrowband_spcc(self):
        pipeline = _pipeline()
        metadata = {"INSTRUME": "imx585", "FILTER": "LP"}
        mapping = stage4.resolve_dual_narrowband_mapping(metadata)
        policy = stage4._stage4_channel_policy(
            pipeline,
            metadata,
            checkpoint_loaded=True,
            narrowband_mapping=mapping,
        )
        args, parameters = stage4._stage4_spcc_args(
            pipeline,
            metadata,
            policy,
            catalog="gaia",
        )

        self.assertEqual(mapping["mapping"], "osc_hoo_rgb")
        self.assertEqual(mapping["confidence"], 0.86)
        self.assertEqual(
            mapping["evidence"],
            "authoritative_filter_field_hint",
        )
        self.assertEqual(policy["kind"], "narrowband_composite")
        self.assertIs(parameters["mapping"], mapping)
        self.assertIn("-narrowband", args)
        self.assertFalse(any("oscfilter" in arg for arg in args))

    def test_dual_narrowband_routes_spcc_then_degraded_pcc_and_isolated_hoo(self):
        pipeline = _pipeline()
        policy = _channel_policy(
            pipeline,
            {"OBJECT": "M 42", "FILTER": "Ha+OIII dual-band"},
            checkpoint_loaded=True,
        )
        self.assertEqual(policy["kind"], "narrowband_composite")
        self.assertEqual(
            policy["action"],
            "spcc_narrowband_then_degraded_pcc_with_isolated_hoo",
        )

    def test_unknown_linearity_preserves_input_for_review(self):
        policy = _channel_policy(
            _pipeline(),
            {"OBJECT": "M 42"},
            checkpoint_loaded=False,
        )
        self.assertEqual(policy["kind"], "unknown")
        self.assertEqual(policy["action"], "preserve_input_review")

    def test_preserve_mode_still_freezes_stage4_channel_mapping(self):
        pipeline, _saved, _commands, _results = _stage4_integration_fixture(
            filter_name="Ha+OIII dual-band"
        )
        pipeline.cfg.stage4_processing_mode = "preserve"
        original_resolver = stage4.resolve_dual_narrowband_mapping

        with patch.object(
            stage4,
            "resolve_dual_narrowband_mapping",
            wraps=original_resolver,
        ) as resolver:
            stage4.run_stage4_color_calibration(pipeline)

        self.assertEqual(resolver.call_count, 1)
        mapping = pipeline.narrowband_channel_mapping
        self.assertEqual(mapping["mapping"], "osc_hoo_rgb")
        self.assertEqual(
            pipeline.stage_json["stage4_channel_mapping.json"],
            mapping,
        )
        self.assertEqual(
            pipeline.channel_profile["narrowband_mapping"],
            mapping,
        )
        self.assertEqual(
            pipeline.color_calibration_report["channel_mapping"],
            mapping,
        )

    def test_stage4_reuses_one_mapping_for_spcc_and_hoo(self):
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            filter_name="Ha+OIII dual-band"
        )
        captured = {}

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc narrowband ok"

        def hoo_derivative(_pipeline, mapping):
            captured["mapping"] = mapping
            return False, {
                "status": "skipped_test",
                "accepted": False,
                "mapping": mapping,
            }, "HOO skipped by test"

        pipeline._run_stage4_spcc_once = spcc_success
        original_resolver = stage4.resolve_dual_narrowband_mapping
        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(
                stage4,
                "resolve_dual_narrowband_mapping",
                wraps=original_resolver,
            ) as resolver,
            patch.object(
                stage4,
                "_stage4_run_narrowband_normalization",
                hoo_derivative,
            ),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        mapping = pipeline.narrowband_channel_mapping
        self.assertEqual(resolver.call_count, 1)
        self.assertIs(
            pipeline.channel_profile["narrowband_mapping"],
            mapping,
        )
        self.assertEqual(captured["mapping"], mapping)
        self.assertIs(captured["mapping"], mapping)
        self.assertEqual(
            pipeline.color_calibration_report["spcc"]["parameters"]["mapping"],
            mapping,
        )
        self.assertEqual(
            pipeline.color_calibration_report["channel_mapping"],
            mapping,
        )

    def test_stage8_uses_frozen_mapping_not_changed_header(self):
        frozen_mapping = stage4.resolve_dual_narrowband_mapping(
            {"INSTRUME": "Seestar S30 Pro", "FILTER": "LP"}
        )
        written = {}
        pipeline = SimpleNamespace(
            cfg=SimpleNamespace(
                stage8_dualband_palette_enabled=True,
                stage4_nbn_mapping_confidence_min=0.85,
            ),
            narrowband_channel_mapping=frozen_mapping,
            channel_profile={"narrowband_mapping": frozen_mapping},
            _stage4_header_metadata={"FILTER": "SII/OIII Dual-Band"},
            _stage8_final_quality="degraded",
            _stage7_stretch_accepted=True,
            _stage8_palette_report={},
            _stage8_palette_selection=resolve_palette_selection(
                {"type": "generic_low_snr_safe", "frozen": True},
                "auto",
            ),
            target_profile={},
            color_calibration_report={},
            _write_stage_json=lambda name, payload: written.__setitem__(
                name,
                payload,
            ),
        )

        report = stage8._stage8_run_dualband_palette(
            pipeline,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="narrowband_composite",
            processing_policy="full",
            external_override=False,
        )

        self.assertEqual(report["mapping_evidence"], frozen_mapping)
        self.assertEqual(report["mapping_source"], "stage4_runtime_contract")
        self.assertNotIn("conflicting_filter_lines", str(report))
        self.assertEqual(written["stage8_palette_report.json"], report)

    def test_manual_palette_does_not_bypass_invalid_mapping(self):
        written = {}
        pipeline = SimpleNamespace(
            cfg=SimpleNamespace(
                stage8_dualband_palette_enabled=True,
                stage4_nbn_mapping_confidence_min=0.85,
            ),
            narrowband_channel_mapping={
                "schema": "starun.narrowband-channel-mapping.v1",
                "mapping": "unknown",
                "ha_channel": None,
                "oiii_channels": [],
                "confidence": 0.0,
            },
            channel_profile={},
            _stage8_final_quality="ok",
            _stage7_stretch_accepted=True,
            _stage8_palette_report={},
            _stage8_palette_selection=resolve_palette_selection(
                {"type": "emission_nebula_widefield", "frozen": True},
                "OHS",
            ),
            color_calibration_report={},
            _write_stage_json=lambda name, payload: written.__setitem__(
                name,
                payload,
            ),
        )

        report = stage8._stage8_run_dualband_palette(
            pipeline,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="narrowband_composite",
            processing_policy="full",
            external_override=False,
        )

        self.assertEqual(report["status"], "skipped_ineligible")
        self.assertEqual(report["planned_palette"], "OHS")
        self.assertEqual(report["automatic_palette"], "SHO")
        self.assertIn("channel_mapping_kind_invalid", report["eligibility"]["issues"])
        self.assertIn("ha_oiii_mapping_unconfirmed", report["eligibility"]["issues"])
        self.assertEqual(written["stage8_palette_report.json"], report)

    def test_stage8_missing_frozen_palette_selection_is_fail_closed(self):
        frozen_mapping = stage4.resolve_dual_narrowband_mapping(
            {"INSTRUME": "Seestar S30 Pro", "FILTER": "LP"}
        )
        pipeline = SimpleNamespace(
            cfg=SimpleNamespace(
                stage8_dualband_palette_enabled=True,
                stage4_nbn_mapping_confidence_min=0.85,
            ),
            narrowband_channel_mapping=frozen_mapping,
            channel_profile={"narrowband_mapping": frozen_mapping},
            _stage8_final_quality="ok",
            _stage7_stretch_accepted=True,
            _stage8_palette_report={},
            color_calibration_report={},
            _write_stage_json=lambda _name, _payload: None,
        )

        report = stage8._stage8_run_dualband_palette(
            pipeline,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="narrowband_composite",
            processing_policy="full",
            external_override=False,
        )

        self.assertEqual(report["status"], "skipped_ineligible")
        self.assertIsNone(report["planned_palette"])
        self.assertIn(
            "stage8_palette_selection_missing",
            report["eligibility"]["issues"],
        )

    def test_pcc_runner_is_exactly_one_attempt(self):
        pipeline = _pipeline()
        calls = []

        def run_once(*, timeout_sec: int, catalog: str):
            calls.append((timeout_sec, catalog))
            return False, "timeout"

        pipeline._run_stage4_pcc_once = run_once
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)
        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
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
        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            ok, _detail, attempts = stage4._stage4_run_spcc(
                pipeline,
                phase="linear_broadband",
                catalog="gaia",
                args=args,
                narrowband=False,
            )

        self.assertFalse(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timeout_sec"], 90)
        self.assertEqual(calls[0]["catalog"], "gaia")
        self.assertEqual(attempts[0]["max_attempts"], 1)
        self.assertEqual(attempts[0]["configured_timeout_sec"], 300)
        self.assertEqual(attempts[0]["timeout_policy"], "online_unverified_cap")
        self.assertTrue(attempts[0]["online_unverified_cap_applied"])

    def test_spcc_online_unverified_timeout_respects_lower_user_budget(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_timeout_sec = 45
        calls = []
        pipeline._run_stage4_spcc_once = lambda **kwargs: (
            calls.append(kwargs) or False,
            "timeout",
        )
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            _ok, _detail, attempts = stage4._stage4_run_spcc(
                pipeline,
                phase="linear_broadband",
                catalog="gaia",
                args=("-oscsensor=x",),
                narrowband=False,
            )

        self.assertEqual(calls[0]["timeout_sec"], 45)
        self.assertEqual(attempts[0]["timeout_sec"], 45)

    def test_spcc_batch_timeout_circuit_skips_online_gaia(self):
        pipeline = _pipeline()
        calls = []
        pipeline._run_stage4_spcc_once = lambda **kwargs: (
            calls.append(kwargs) or True,
            "unexpected",
        )
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)

        with patch.dict(
            os.environ,
            {
                "STARUN_NETWORK_MODE": "1",
                stage4.SPCC_ONLINE_CIRCUIT_ENV: "1",
            },
            clear=False,
        ):
            ok, detail, attempts = stage4._stage4_run_spcc(
                pipeline,
                phase="linear_broadband",
                catalog="gaia",
                args=("-oscsensor=x",),
                narrowband=False,
            )

        self.assertFalse(ok)
        self.assertEqual(calls, [])
        self.assertIn("circuit is open", detail)
        self.assertEqual(attempts[0]["status"], "skipped")
        self.assertEqual(
            attempts[0]["reason_code"],
            "batch_online_timeout_circuit_open",
        )

    def test_spcc_batch_timeout_circuit_does_not_skip_localgaia(self):
        pipeline = _pipeline()
        calls = []
        pipeline._run_stage4_spcc_once = lambda **kwargs: (
            calls.append(kwargs) or False,
            "local attempt",
        )
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)

        with (
            patch.dict(
                os.environ,
                {
                    "STARUN_NETWORK_MODE": "0",
                    stage4.SPCC_ONLINE_CIRCUIT_ENV: "1",
                },
                clear=False,
            ),
            patch.object(
                stage4,
                "_stage4_local_spcc_catalog_status",
                return_value={"available": True},
            ),
        ):
            _ok, _detail, attempts = stage4._stage4_run_spcc(
                pipeline,
                phase="linear_broadband",
                catalog="localgaia",
                args=("-oscsensor=x",),
                narrowband=False,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(attempts[0]["label"], "catalog:localgaia")

    def test_spcc_localgaia_keeps_configured_timeout(self):
        pipeline = _pipeline()
        calls = []
        pipeline._run_stage4_spcc_once = lambda **kwargs: (
            calls.append(kwargs) or False,
            "timeout",
        )
        pipeline.log = SimpleNamespace(warn=lambda *_args: None, info=lambda *_args: None)

        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "0"}, clear=False),
            patch.object(
                stage4,
                "_stage4_local_spcc_catalog_status",
                return_value={"available": True},
            ),
        ):
            _ok, _detail, attempts = stage4._stage4_run_spcc(
                pipeline,
                phase="linear_broadband",
                catalog="localgaia",
                args=("-oscsensor=x",),
                narrowband=False,
            )

        self.assertEqual(calls[0]["timeout_sec"], 300)
        self.assertEqual(attempts[0]["timeout_policy"], "configured_localgaia")
        self.assertFalse(attempts[0]["online_unverified_cap_applied"])

    def test_common_smart_devices_select_profile_sensor_and_filter(self):
        cases = (
            (
                {"TELESCOP": "ZWO Seestar S30", "FILTER": "Clear"},
                "ZWO Seestar S30",
                "UV/IR Block",
            ),
            (
                {"TELESCOP": "ZWO Seestar S30 Pro", "FILTER": "UV/IR Cut"},
                "Sony IMX585",
                "UV/IR Block",
            ),
            (
                {"INSTRUME": "Seestar S50", "FILTER": "Clear"},
                "ZWO Seestar S50",
                "UV/IR Block",
            ),
            (
                {"TELESCOP": "DWARFII", "FILTER": "Astro"},
                "Sony IMX415",
                "UV/IR Block",
            ),
            (
                {"TELESCOP": "DWARFIII", "FILTER": "Astro"},
                "Sony IMX678",
                "UV/IR Block",
            ),
            (
                {"TELESCOP": "DWARF mini", "FILTER": "Astro"},
                "Sony IMX662",
                "Dwarf Mini Astro",
            ),
        )
        for metadata, expected_sensor, expected_filter in cases:
            with self.subTest(metadata=metadata):
                pipeline = _pipeline()
                pipeline.cfg.stage4_spcc_osc_sensor = ""
                args, parameters = stage4._stage4_spcc_args(
                    pipeline,
                    metadata,
                    {"kind": "broadband_rgb_osc"},
                    catalog="gaia",
                )
                self.assertEqual(parameters["sensor"], expected_sensor)
                self.assertEqual(parameters["osc_filter"], expected_filter)
                self.assertEqual(parameters["sensor_source"], "smart_device_profile")
                self.assertIn(f'"-oscsensor={expected_sensor}"', args)
                self.assertIn(f'"-oscfilter={expected_filter}"', args)

    def test_dwarf_dualband_aliases_supply_confirmed_ha_oiii_bandwidths(self):
        for filter_name in ("Duo-Band      ", "Dual-Band      "):
            with self.subTest(filter_name=filter_name):
                pipeline = _pipeline()
                pipeline.cfg.stage4_spcc_osc_sensor = ""
                metadata = {"TELESCOP": "DWARF 3", "FILTER": filter_name}
                with patch.dict(os.environ, {}, clear=True):
                    policy = _channel_policy(
                        pipeline,
                        metadata,
                        checkpoint_loaded=True,
                    )
                    args, parameters = stage4._stage4_spcc_args(
                        pipeline,
                        metadata,
                        policy,
                        catalog="gaia",
                    )

                self.assertEqual(policy["kind"], "narrowband_composite")
                self.assertEqual(parameters["sensor"], "Sony IMX678")
                self.assertEqual(
                    parameters["mapping"]["evidence"],
                    "verified_device_profile",
                )
                self.assertEqual(parameters["bandwidths_nm"]["r"], 18.0)
                self.assertEqual(
                    parameters["parameter_sources"]["r_bandwidth_nm"],
                    "smart_device_profile",
                )
                self.assertIn("-rbw=18", args)

    def test_recognized_starun_without_filter_hint_defaults_to_uv_ir_block(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_osc_sensor = ""
        _args, parameters = stage4._stage4_spcc_args(
            pipeline,
            {"TELESCOP": "Seestar S50"},
            {"kind": "broadband_rgb_osc"},
            catalog="gaia",
        )

        self.assertEqual(parameters["osc_filter"], "UV/IR Block")
        self.assertEqual(
            parameters["osc_filter_reason"],
            "smart_device_profile_default",
        )

    def test_explicit_sensor_override_wins_over_detected_device(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_osc_sensor = "Sony IMX585"
        _args, parameters = stage4._stage4_spcc_args(
            pipeline,
            {"TELESCOP": "DWARF 3", "FILTER": "Astro"},
            {"kind": "broadband_rgb_osc"},
            catalog="gaia",
        )

        self.assertEqual(parameters["sensor"], "Sony IMX585")
        self.assertEqual(parameters["sensor_source"], "explicit_config")

    def test_unknown_device_does_not_silently_use_imx585(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_osc_sensor = ""
        with self.assertRaises(stage4._Stage4SpccDeviceMetadataMissing):
            stage4._stage4_spcc_args(
                pipeline,
                {"TELESCOP": "Unknown Smart Telescope", "FILTER": "Astro"},
                {"kind": "broadband_rgb_osc"},
                catalog="gaia",
            )

    def test_qhy_header_identity_is_reported_without_weakening_spcc_fail_closed(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_osc_sensor = ""
        metadata = {
            "INSTRUME": "QHY268M",
            "TELESCOP": "Askar 107PHQ",
            "FOCALLEN": 746.608,
            "XPIXSZ": 3.76,
            "YPIXSZ": 3.76,
            "FILTER": "L",
        }

        geometry = stage4._stage4_image_geometry(pipeline, metadata)

        self.assertEqual(geometry["instrument"], "Askar 107PHQ")
        self.assertEqual(geometry["instrument_source"], "fits_header:TELESCOP")
        self.assertEqual(geometry["sensor"], "QHY268M")
        self.assertEqual(geometry["sensor_source"], "fits_header:INSTRUME")
        self.assertEqual(geometry["identity_source"], "header_derived")
        self.assertIsNone(geometry["device_profile_id"])
        with self.assertRaises(stage4._Stage4SpccDeviceMetadataMissing):
            stage4._stage4_spcc_args(
                pipeline,
                metadata,
                {"kind": "broadband_rgb_osc"},
                catalog="gaia",
            )

    def test_missing_auto_sensor_config_field_is_also_fail_closed(self):
        pipeline = _pipeline()
        del pipeline.cfg.stage4_spcc_osc_sensor

        with self.assertRaises(stage4._Stage4SpccDeviceMetadataMissing):
            stage4._stage4_spcc_args(
                pipeline,
                {"TELESCOP": "Unknown", "FILTER": "L"},
                {"kind": "broadband_rgb_osc"},
                catalog="gaia",
            )

    def test_known_device_does_not_guess_an_unsupported_filter_curve(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_osc_sensor = ""
        for filter_name in ("VIS", "LP"):
            with self.subTest(filter_name=filter_name), self.assertRaises(
                stage4._Stage4SpccDeviceMetadataMissing
            ):
                stage4._stage4_spcc_args(
                    pipeline,
                    {"TELESCOP": "DWARF 3", "FILTER": filter_name},
                    {"kind": "broadband_rgb_osc"},
                    catalog="gaia",
                )

    def test_smart_telescope_wide_camera_does_not_use_tele_sensor_profile(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_spcc_osc_sensor = ""
        with self.assertRaises(stage4._Stage4SpccDeviceMetadataMissing):
            stage4._stage4_spcc_args(
                pipeline,
                {
                    "TELESCOP": "Seestar S30 Pro",
                    "SENSOR": "Sony IMX586",
                    "FOCALLEN": 6.0,
                    "FILTER": "Clear",
                },
                {"kind": "broadband_rgb_osc"},
                catalog="gaia",
            )

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

    def test_bundled_database_contains_common_smart_device_responses(self):
        pipeline = _pipeline()
        pipeline.spcc_database_dir = REPO_ROOT / "resources" / "siril_spcc_database"
        status = stage4._stage4_spcc_database_status(pipeline)
        records = {
            (record["kind"], record["label"]): record
            for record in status["files"]
        }
        expected = {
            ("osc_sensor", "Sony IMX415"),
            ("osc_sensor", "Sony IMX462"),
            ("osc_sensor", "Sony IMX585"),
            ("osc_sensor", "Sony IMX662"),
            ("osc_sensor", "Sony IMX678"),
            ("osc_sensor", "ZWO Seestar S30"),
            ("osc_sensor", "ZWO Seestar S50"),
            ("osc_filter", "UV/IR Block"),
            ("osc_filter", "Dwarf Mini Astro"),
            ("osc_filter", "ZWO Seestar LP"),
        }

        self.assertTrue(expected.issubset(records))
        self.assertTrue(all(records[key]["available"] for key in expected))
        self.assertTrue(all(records[key]["sha256"] for key in expected))

    def test_spcc_runtime_capabilities_verify_requested_metadata(self):
        pipeline = _pipeline()
        outputs = {
            "oscsensor": "log: OSC Sensors\nlog: Sony IMX585\nlog: Sony IMX662\n",
            "oscfilter": "log: OSC Filters\nlog: No filter\nlog: ZWO Seestar LP\n",
            "whiteref": (
                "log: White References\n"
                "log: Average Spiral Galaxy\n"
                "log: Star, type G2(v)\n"
            ),
        }
        calls = []
        pipeline._run_stage4_spcc_list_once = lambda **kwargs: (
            calls.append(kwargs) or True,
            outputs[kwargs["kind"]],
        )
        _args, parameters = stage4._stage4_spcc_args(
            pipeline,
            {"FILTER": "LP"},
            {"kind": "broadband_rgb_osc"},
            catalog="gaia",
        )

        report = stage4._stage4_spcc_runtime_capabilities(
            pipeline,
            parameters,
        )

        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["decision"], "allow")
        self.assertFalse(report["blocking_missing"])
        self.assertEqual(
            [call["kind"] for call in calls],
            ["oscsensor", "whiteref", "oscfilter"],
        )
        self.assertTrue(report["categories"]["osc_sensor"]["found"])
        self.assertEqual(
            report["categories"]["osc_filter"]["matched_label"],
            "ZWO Seestar LP",
        )

    def test_spcc_list_parser_accepts_localized_category_heading(self):
        output = (
            "log: Running command: spcc_list\n"
            "1786001710: running command spcc_list\n"
            "log: OSC 传感器\n"
            "log: Sony IMX585\n"
            "log: Sony IMX662\n"
            "log: Running command: close\n"
        )

        values = stage4._stage4_parse_spcc_list_output(output, "oscsensor")

        self.assertEqual(values, ["Sony IMX585", "Sony IMX662"])

    def test_spcc_runtime_capabilities_reject_confirmed_missing_sensor(self):
        pipeline = _pipeline()
        outputs = {
            "oscsensor": "log: OSC Sensors\nlog: Sony IMX662\n",
            "oscfilter": "log: OSC Filters\nlog: ZWO Seestar LP\n",
            "whiteref": "log: White References\nlog: Average Spiral Galaxy\n",
        }
        pipeline._run_stage4_spcc_list_once = lambda **kwargs: (
            True,
            outputs[kwargs["kind"]],
        )
        _args, parameters = stage4._stage4_spcc_args(
            pipeline,
            {"FILTER": "LP"},
            {"kind": "broadband_rgb_osc"},
            catalog="gaia",
        )

        report = stage4._stage4_spcc_runtime_capabilities(
            pipeline,
            parameters,
        )

        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["decision"], "reject")
        self.assertEqual(report["blocking_missing"], ["osc_sensor=Sony IMX585"])
        self.assertFalse(report["categories"]["osc_sensor"]["found"])

    def test_spcc_runtime_probe_failure_is_nonblocking_and_audited(self):
        pipeline = _pipeline()
        pipeline._run_stage4_spcc_list_once = lambda **_kwargs: (
            False,
            "probe unavailable",
        )
        _args, parameters = stage4._stage4_spcc_args(
            pipeline,
            {"FILTER": "LP"},
            {"kind": "broadband_rgb_osc"},
            catalog="gaia",
        )

        report = stage4._stage4_spcc_runtime_capabilities(
            pipeline,
            parameters,
        )

        self.assertEqual(report["status"], "unverified")
        self.assertEqual(report["decision"], "allow_unverified")
        self.assertFalse(report["blocking_missing"])
        self.assertEqual(
            set(report["unverified_requirements"]),
            {"osc_sensor", "osc_filter", "white_reference"},
        )

    def test_narrowband_runtime_probe_does_not_require_osc_filter(self):
        pipeline = _pipeline()
        outputs = {
            "oscsensor": "log: OSC Sensors\nlog: Sony IMX585\n",
            "whiteref": "log: White References\nlog: Average Spiral Galaxy\n",
        }
        calls = []

        def runtime_list(**kwargs):
            calls.append(kwargs["kind"])
            return True, outputs[kwargs["kind"]]

        pipeline._run_stage4_spcc_list_once = runtime_list
        _args, parameters = stage4._stage4_spcc_args(
            pipeline,
            {"FILTER": "Ha+OIII dual-band"},
            {
                "kind": "narrowband_composite",
                "filter_hint": "Ha+OIII dual-band",
                "narrowband_mapping": stage4.resolve_dual_narrowband_mapping(
                    {"FILTER": "Ha+OIII dual-band"}
                ),
            },
            catalog="gaia",
        )

        report = stage4._stage4_spcc_runtime_capabilities(
            pipeline,
            parameters,
        )

        self.assertEqual(report["decision"], "allow")
        self.assertEqual(calls, ["oscsensor", "whiteref"])
        self.assertNotIn("osc_filter", report["categories"])

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

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
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

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
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
            with catalog.open("wb") as handle:
                handle.truncate(stage4.LOCAL_ASTROMETRIC_EXPECTED_SIZE_BYTES)
            with patch.dict(
                os.environ,
                {
                    "STARUN_NETWORK_MODE": "0",
                    "STARUN_GAIA_ASTRO_CATALOG": str(catalog),
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

    def test_direct_offline_spcc_requires_all_expected_xp_chunks(self):
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as temp_dir:
            photo = Path(temp_dir) / stage4.LOCAL_SPCC_DIRNAME
            photo.mkdir()
            (photo / f"{stage4.LOCAL_SPCC_FILE_PREFIX}0.dat").write_bytes(
                b"x" * stage4.MIN_LOCAL_CATALOG_FILE_BYTES
            )
            pipeline.local_gaia_photo_catalog = photo
            with patch.dict(
                os.environ,
                {"STARUN_NETWORK_MODE": "0"},
                clear=False,
            ):
                self.assertIsNone(stage4._stage4_preferred_spcc_catalog(pipeline))
                for index in range(1, stage4.LOCAL_SPCC_EXPECTED_CHUNKS):
                    (photo / f"{stage4.LOCAL_SPCC_FILE_PREFIX}{index}.dat").write_bytes(
                        b"x" * stage4.MIN_LOCAL_CATALOG_FILE_BYTES
                    )
                self.assertEqual(
                    stage4._stage4_preferred_spcc_catalog(pipeline),
                    "localgaia",
                )

    def test_default_catalog_policy_prefers_online_gaia_even_when_local_exists(self):
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            astro = root / stage4.LOCAL_ASTROMETRIC_FILENAME
            astro.write_bytes(b"x" * stage4.MIN_LOCAL_CATALOG_FILE_BYTES)
            photo = root / stage4.LOCAL_SPCC_DIRNAME
            photo.mkdir()
            (photo / "siril_cat1_healpix8_xpsamp_0.dat").write_bytes(
                b"x" * stage4.MIN_LOCAL_CATALOG_FILE_BYTES
            )
            pipeline.local_gaia_astro_catalog = astro
            pipeline.local_gaia_photo_catalog = photo

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(stage4._stage4_network_enabled())
                self.assertEqual(
                    stage4._stage4_preferred_pcc_catalog(pipeline),
                    "gaia",
                )
                self.assertEqual(
                    stage4._stage4_preferred_spcc_catalog(pipeline),
                    "gaia",
                )

    def test_explicit_offline_without_local_gaia_disables_photometric_catalogs(self):
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.local_gaia_astro_catalog = root / "missing_astro.dat"
            pipeline.local_gaia_photo_catalog = root / "missing_photo"

            with patch.dict(
                os.environ,
                {"STARUN_NETWORK_MODE": "0"},
                clear=False,
            ):
                self.assertFalse(stage4._stage4_network_enabled())
                self.assertIsNone(stage4._stage4_preferred_pcc_catalog(pipeline))
                self.assertIsNone(stage4._stage4_preferred_spcc_catalog(pipeline))

    def test_runtime_capability_manifest_is_run_scoped_and_routes_stage4_commands(self):
        pipeline = _pipeline()
        pipeline._run_id = "run-1"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.work_dir = root
            manifest_path = root / "runtime-capabilities.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": stage4.RUNTIME_CAPABILITIES_SCHEMA,
                        "run_id": "run-1",
                        "status": "degraded_allowed",
                        "decisions": {
                            "stage4_color_calibration": {
                                "schema": stage4.RUNTIME_COLOR_DECISION_SCHEMA,
                                "status": "degraded_allowed",
                                "route": "auto_local_reference",
                                "offline_fallback_mode": "auto_local_reference",
                                "commands": {
                                    "platesolve": False,
                                    "spcc": False,
                                    "pcc": False,
                                },
                                "skip_photometric_commands": [
                                    "platesolve",
                                    "spcc",
                                    "pcc",
                                ],
                                "requires_review": True,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {stage4.RUNTIME_CAPABILITIES_ENV: str(manifest_path)},
                clear=False,
            ):
                decision = stage4._stage4_runtime_color_decision(pipeline)

            self.assertTrue(decision["trusted"])
            self.assertEqual(decision["source"], "runtime_capabilities_manifest")
            self.assertFalse(decision["commands"]["platesolve"])
            self.assertEqual(decision["spcc_readiness"], "unavailable")

            malformed_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            malformed_payload["decisions"]["stage4_color_calibration"][
                "commands"
            ]["platesolve"] = "false"
            manifest_path.write_text(
                json.dumps(malformed_payload),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    stage4.RUNTIME_CAPABILITIES_ENV: str(manifest_path),
                    "STARUN_NETWORK_MODE": "1",
                },
                clear=False,
            ):
                malformed = stage4._stage4_runtime_color_decision(pipeline)
            self.assertFalse(malformed["trusted"])
            self.assertIn("must use booleans", malformed["manifest_error"])

            malformed_payload["decisions"]["stage4_color_calibration"][
                "commands"
            ]["platesolve"] = False
            manifest_path.write_text(
                json.dumps(malformed_payload),
                encoding="utf-8",
            )

            pipeline._run_id = "another-run"
            with patch.dict(
                os.environ,
                {
                    stage4.RUNTIME_CAPABILITIES_ENV: str(manifest_path),
                    "STARUN_NETWORK_MODE": "1",
                },
                clear=False,
            ):
                rejected = stage4._stage4_runtime_color_decision(pipeline)
            self.assertFalse(rejected["trusted"])
            self.assertIn("run_id mismatch", rejected["manifest_error"])
            self.assertTrue(rejected["commands"]["platesolve"])

    def test_legacy_runtime_manifest_infers_local_spcc_readiness_from_xp_source(self):
        pipeline = _pipeline()
        pipeline._run_id = "run-legacy"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.work_dir = root
            manifest_path = root / "runtime-capabilities.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": stage4.RUNTIME_CAPABILITIES_SCHEMA,
                        "run_id": "run-legacy",
                        "status": "ready",
                        "decisions": {
                            "stage4_color_calibration": {
                                "schema": stage4.RUNTIME_COLOR_DECISION_SCHEMA,
                                "status": "ready",
                                "route": "physical_spcc_then_pcc",
                                "offline_fallback_mode": "auto_local_reference",
                                "astrometric_source": "localgaia",
                                "xp_source": "localgaia",
                                "commands": {
                                    "platesolve": True,
                                    "spcc": True,
                                    "pcc": True,
                                },
                                "skip_photometric_commands": [],
                                "requires_review": False,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {stage4.RUNTIME_CAPABILITIES_ENV: str(manifest_path)},
                clear=False,
            ):
                decision = stage4._stage4_runtime_color_decision(pipeline)

        self.assertTrue(decision["trusted"])
        self.assertEqual(decision["spcc_readiness"], "local_verified")

    def test_offline_without_gaia_skips_photometric_commands_and_applies_background_candidate(self):
        source = _auto_reference_image()
        pipeline, saved, commands, results = _stage4_integration_fixture(
            image=source
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.local_gaia_astro_catalog = root / "missing_astro.dat"
            pipeline.local_gaia_photo_catalog = root / "missing_photo"
            pipeline._run_stage4_spcc_once = lambda **_kwargs: self.fail(
                "SPCC must be skipped when Gaia capability is confirmed absent"
            )
            pipeline._run_stage4_pcc_once = lambda **_kwargs: self.fail(
                "PCC must be skipped when Gaia capability is confirmed absent"
            )

            with patch.dict(
                os.environ,
                {
                    "STARUN_NETWORK_MODE": "0",
                    stage4.RUNTIME_CAPABILITIES_ENV: "",
                },
                clear=False,
            ):
                stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertFalse(report["platesolve"]["attempted"])
        self.assertEqual(report["method"], stage4.AUTO_BACKGROUND_METHOD)
        self.assertFalse(report["physical_color"]["accepted"])
        self.assertTrue(report["degraded_color_correction"]["applied"])
        self.assertTrue(report["requires_review"])
        self.assertEqual(
            report["runtime_capability_decision"]["route"],
            "auto_local_reference",
        )
        self.assertTrue(
            np.array_equal(saved["stage4_psolved"], saved["stage3_bgremoved"])
        )
        self.assertFalse(
            np.array_equal(saved["stage4_color"], saved[stage4.PCC_CHECKPOINT_STEM])
        )
        self.assertFalse(
            any(call and call[0] in {"platesolve", "spcc", "pcc"} for call in commands)
        )
        self.assertIn(stage4.AUTO_REFERENCE_REPORT_NAME, pipeline.stage_json)
        self.assertEqual(results[-1][0][1], "degraded")

    def test_offline_preserve_mode_keeps_exact_pre_color_and_skips_local_restore(self):
        source = _auto_reference_image()
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            image=source
        )
        pipeline.cfg.stage4_offline_fallback_mode = "preserve"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.local_gaia_astro_catalog = root / "missing_astro.dat"
            pipeline.local_gaia_photo_catalog = root / "missing_photo"
            with (
                patch.dict(
                    os.environ,
                    {
                        "STARUN_NETWORK_MODE": "0",
                        stage4.RUNTIME_CAPABILITIES_ENV: "",
                    },
                    clear=False,
                ),
                patch.object(
                    stage4,
                    "_stage4_local_color_fallback",
                    side_effect=AssertionError("preserve mode must skip local restore"),
                ),
            ):
                stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "PRESERVE_INPUT")
        self.assertTrue(report["requires_review"])
        self.assertEqual(
            report["auto_local_reference"]["status"],
            "preserved_by_configuration",
        )
        self.assertTrue(
            np.array_equal(saved["stage4_color"], saved[stage4.PCC_CHECKPOINT_STEM])
        )

    def test_auto_reference_write_failure_restores_exact_pre_color(self):
        source = _auto_reference_image()
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            image=source
        )
        pipeline.cfg.stage4_local_star_wb_enabled = False
        original_save = pipeline._save_stage_output

        def save(stem):
            if stem == "stage4_auto_reference_candidate":
                return False
            return original_save(stem)

        pipeline._save_stage_output = save
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.local_gaia_astro_catalog = root / "missing_astro.dat"
            pipeline.local_gaia_photo_catalog = root / "missing_photo"
            with patch.dict(
                os.environ,
                {
                    "STARUN_NETWORK_MODE": "0",
                    stage4.RUNTIME_CAPABILITIES_ENV: "",
                },
                clear=False,
            ):
                stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "PRESERVE_INPUT")
        self.assertEqual(
            report["auto_local_reference"]["status"],
            "apply_failed_rolled_back",
        )
        self.assertTrue(
            report["auto_local_reference"]["transaction"]["rollback_performed"]
        )
        self.assertTrue(
            np.array_equal(saved["stage4_color"], saved[stage4.PCC_CHECKPOINT_STEM])
        )

    def test_missing_immutable_checkpoint_prohibits_all_color_heuristics(self):
        source = _auto_reference_image()
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            image=source
        )
        original_save = pipeline._save_stage_output

        def fail_checkpoint_save(stem):
            if stem == stage4.PCC_CHECKPOINT_STEM:
                saved[stem] = pipeline.current.copy()
                return False
            return original_save(stem)

        pipeline._save_stage_output = fail_checkpoint_save
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.local_gaia_astro_catalog = root / "missing_astro.dat"
            pipeline.local_gaia_photo_catalog = root / "missing_photo"
            with (
                patch.dict(
                    os.environ,
                    {
                        "STARUN_NETWORK_MODE": "0",
                        stage4.RUNTIME_CAPABILITIES_ENV: "",
                    },
                    clear=False,
                ),
                patch.object(
                    stage4,
                    "evaluate_auto_local_reference",
                    side_effect=AssertionError(
                        "missing immutable checkpoint must prohibit auto reference"
                    ),
                ),
                patch.object(
                    stage4,
                    "_stage4_local_color_fallback",
                    side_effect=AssertionError(
                        "missing immutable checkpoint must prohibit local restore"
                    ),
                ),
            ):
                stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "PRESERVE_INPUT")
        self.assertEqual(report["auto_local_reference"]["status"], "unavailable")
        self.assertEqual(
            report["auto_local_reference"]["eligibility"]["reason"],
            "immutable_pre_color_checkpoint_unavailable",
        )
        self.assertTrue(np.array_equal(saved["stage4_color"], source))

    def test_auto_reference_post_write_mismatch_restores_exact_pre_color(self):
        source = _auto_reference_image()
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            image=source
        )
        saved[stage4.PCC_CHECKPOINT_STEM] = source.copy()
        writes = 0

        def corrupt_first_write(pixels):
            nonlocal writes
            writes += 1
            pipeline.current = np.asarray(pixels).copy()
            if writes == 1:
                pipeline.current.flat[0] += np.float32(0.001)

        pipeline.siril.set_image_pixeldata = corrupt_first_write
        report = stage4._stage4_empty_auto_reference_report(
            status="accepted",
            reason="test_candidate",
        )
        report["selection"].update(
            method=stage4.AUTO_BACKGROUND_METHOD,
            applied=True,
        )

        applied, result = stage4._stage4_apply_auto_reference_candidate(
            pipeline,
            source + np.float32(0.002),
            expected_pre_color=source,
            report=report,
        )

        self.assertFalse(applied)
        self.assertEqual(result["status"], "apply_failed_rolled_back")
        self.assertTrue(result["transaction"]["rollback_performed"])
        self.assertTrue(np.array_equal(pipeline.current, source))

    def test_local_star_write_mismatch_restores_exact_pre_color(self):
        source = _auto_reference_image()
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            image=source
        )

        def corrupt_write(pixels):
            pipeline.current = np.asarray(pixels).copy()
            pipeline.current.flat[0] += np.float32(0.001)

        pipeline.siril.set_image_pixeldata = corrupt_write
        rejected_report = stage4._stage4_empty_auto_reference_report(
            status="rejected",
            reason="test_auto_candidate_rejected",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pipeline.local_gaia_astro_catalog = root / "missing_astro.dat"
            pipeline.local_gaia_photo_catalog = root / "missing_photo"
            with (
                patch.dict(
                    os.environ,
                    {
                        "STARUN_NETWORK_MODE": "0",
                        stage4.RUNTIME_CAPABILITIES_ENV: "",
                    },
                    clear=False,
                ),
                patch.object(
                    stage4,
                    "evaluate_auto_local_reference",
                    return_value=(None, rejected_report),
                ),
                patch.object(
                    stage4,
                    "_stage4_select_local_star_reference_candidate",
                    side_effect=lambda chw, _pipeline: (
                        chw + np.float32(0.002),
                        {"applied": True, "application": "star_soft_mask_only"},
                    ),
                ),
            ):
                stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "PRESERVE_INPUT")
        self.assertEqual(
            report["local_fallback"]["star_white_balance"]["reason"],
            "application_failed",
        )
        self.assertTrue(
            report["local_fallback"]["transaction"]["rollback_performed"]
        )
        self.assertTrue(
            np.array_equal(saved["stage4_color"], saved[stage4.PCC_CHECKPOINT_STEM])
        )

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

    def test_relaxed_default_gate_accepts_large_ic434_style_color_correction(self):
        pipeline = _pipeline()
        pipeline.cfg.stage4_pcc_channel_gain_ratio_max = 10.0
        pipeline.cfg.stage4_pcc_target_color_drift_max = 0.40
        pipeline.target_profile = {
            "target_name_guess": "Horsehead Nebula",
            "target_type": "dark_nebula_low_contrast",
        }
        before = np.empty((3, 64, 64), dtype=np.float32)
        before[0] = 0.0051749204
        before[1] = 0.0009517903
        before[2] = 0.0028257146
        after = np.empty((3, 64, 64), dtype=np.float32)
        after[0] = 0.0029956400
        after[1] = 0.0029641059
        after[2] = 0.0029707856

        accepted, report = stage4._stage4_pcc_quality_gate(
            before,
            after,
            pipeline,
        )

        self.assertTrue(accepted, report)
        self.assertEqual(report["thresholds"]["channel_gain_ratio_max"], 10.0)
        self.assertEqual(
            report["post_calibration_checks"]["target_color_drift"]["maximum_delta"],
            0.40,
        )
        self.assertNotIn(
            "channel_gain_ratio_exceeded",
            report["rejection_reasons"],
        )
        self.assertNotIn(
            "target_color_drift_exceeded",
            report["rejection_reasons"],
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

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
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

    def test_spcc_restore_verifies_checkpoint_and_uses_exact_memory_fallback(self):
        source = np.full((3, 64, 64), 0.04, dtype=np.float32)
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            image=source
        )
        original_command = pipeline.cmd_with_check
        pcc_saw_exact_baseline = []

        def command(*args, **kwargs):
            if args[:2] == ("load", stage4.PCC_CHECKPOINT_STEM):
                return True
            return original_command(*args, **kwargs)

        def spcc_failure(**_kwargs):
            pipeline.current = source + np.float32(0.20)
            return False, "spcc failed after mutating pixels"

        def pcc_success(**_kwargs):
            pcc_saw_exact_baseline.append(np.array_equal(pipeline.current, source))
            saved[stage4.PCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "pcc ok"

        pipeline.cmd_with_check = command
        pipeline._run_stage4_spcc_once = spcc_failure
        pipeline._run_stage4_pcc_once = pcc_success

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        self.assertEqual(pcc_saw_exact_baseline, [True])
        restore = pipeline.color_calibration_report["spcc"]["rollback"][
            "exact_restore"
        ]
        self.assertTrue(restore["verified_exact"])
        self.assertEqual(restore["source"], "in_memory_pre_color")

    def test_double_pre_color_restore_failure_blocks_pcc_and_main_output(self):
        source = np.full((3, 64, 64), 0.04, dtype=np.float32)
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            image=source
        )
        original_command = pipeline.cmd_with_check
        original_write = pipeline.siril.set_image_pixeldata
        pcc_calls = []

        def command(*args, **kwargs):
            if args[:2] == ("load", stage4.PCC_CHECKPOINT_STEM):
                raise stage4.CommandError("checkpoint unreadable")
            return original_command(*args, **kwargs)

        def spcc_failure(**_kwargs):
            pipeline.current = source + np.float32(0.20)
            pipeline.siril.set_image_pixeldata = lambda _pixels: (_ for _ in ()).throw(
                RuntimeError("in-memory restore unavailable")
            )
            return False, "spcc failed after mutating pixels"

        pipeline.cmd_with_check = command
        pipeline._run_stage4_spcc_once = spcc_failure
        pipeline._run_stage4_pcc_once = lambda **kwargs: (
            pcc_calls.append(kwargs) or False,
            "must not run",
        )

        try:
            with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "immutable pre-color baseline could not be restored",
                ):
                    stage4.run_stage4_color_calibration(pipeline)
        finally:
            pipeline.siril.set_image_pixeldata = original_write

        report = pipeline.color_calibration_report
        self.assertFalse(pcc_calls)
        self.assertTrue(report["main_output_blocked"])
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("stage4_color", saved)
        self.assertEqual(results[-1][0][1], "failed")

    def test_runtime_metadata_rejection_skips_spcc_and_writes_capability_report(self):
        pipeline, saved, _commands, _results = _stage4_integration_fixture()
        outputs = {
            "oscsensor": "log: OSC Sensors\nlog: Sony IMX662\n",
            "oscfilter": "log: OSC Filters\nlog: ZWO Seestar LP\n",
            "whiteref": "log: White References\nlog: Average Spiral Galaxy\n",
        }
        written = {}
        calibration_order = []
        pipeline._run_stage4_spcc_list_once = lambda **kwargs: (
            True,
            outputs[kwargs["kind"]],
        )
        pipeline._run_stage4_spcc_once = lambda **_kwargs: self.fail(
            "confirmed missing runtime metadata must skip SPCC"
        )

        def pcc_success(**_kwargs):
            calibration_order.append("pcc")
            saved[stage4.PCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "pcc ok"

        pipeline._run_stage4_pcc_once = pcc_success
        pipeline._write_stage_json = lambda name, payload: written.update(
            {name: payload}
        )

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        runtime = pipeline.color_calibration_report["spcc"]["runtime_capabilities"]
        self.assertEqual(calibration_order, ["pcc"])
        self.assertEqual(runtime["decision"], "reject")
        self.assertEqual(
            pipeline.color_calibration_report["spcc"]["quality_gate"]["status"],
            "runtime_preflight_rejected",
        )
        self.assertIsNone(
            pipeline.color_calibration_report["outputs"]["spcc_candidate"]
        )
        self.assertEqual(
            written["stage4_spcc_capabilities.json"]["blocking_missing"],
            ["osc_sensor=Sony IMX585", "osc_filter=UV/IR Block"],
        )

    def test_unresolved_auto_device_skips_spcc_and_uses_broadband_pcc(self):
        pipeline, saved, _commands, _results = _stage4_integration_fixture()
        pipeline.cfg.stage4_spcc_osc_sensor = ""
        calibration_order = []
        pipeline._run_stage4_spcc_once = lambda **_kwargs: self.fail(
            "unresolved device must not invoke SPCC"
        )

        def pcc_success(**_kwargs):
            calibration_order.append("pcc")
            saved[stage4.PCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "pcc ok"

        pipeline._run_stage4_pcc_once = pcc_success
        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(calibration_order, ["pcc"])
        self.assertEqual(report["method"], "PCC")
        self.assertEqual(
            report["spcc"]["quality_gate"]["status"],
            "device_preflight_rejected",
        )
        self.assertEqual(
            report["spcc"]["metadata_database"]["device_resolution"]["status"],
            "rejected",
        )

    def test_spcc_success_does_not_invoke_pcc(self):
        pipeline, saved, commands, _results = _stage4_integration_fixture()
        pcc_calls = []
        events = []
        original_save = pipeline._save_stage_output
        original_write = pipeline._write_stage_json

        def save(stem):
            if stem == "stage4_color":
                events.append("stage4_color_saved")
            return original_save(stem)

        def write(name, payload):
            if name == stage4.AUTO_REFERENCE_REPORT_NAME:
                events.append("auto_reference_report_written")
            return original_write(name, payload)

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc ok"

        pipeline._run_stage4_spcc_once = spcc_success
        pipeline._save_stage_output = save
        pipeline._write_stage_json = write
        pipeline._run_stage4_pcc_once = lambda **kwargs: (
            pcc_calls.append(kwargs) or True,
            "pcc should not run",
        )

        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(
                stage4,
                "evaluate_auto_local_reference",
                side_effect=AssertionError(
                    "physical success must not invoke auto-reference shadow"
                ),
            ),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "SPCC")
        self.assertFalse(report["requires_review"])
        self.assertFalse(pipeline._stage4_color_review_required)
        self.assertFalse(pcc_calls)
        self.assertIn(("load", stage4.SPCC_CANDIDATE_STEM), commands)
        shadow = report["auto_local_reference"]
        self.assertEqual(
            shadow["status"],
            "shadow_skipped_physical_color_accepted",
        )
        self.assertEqual(
            shadow["shadow_comparison"]["physical_method_preserved"],
            "SPCC",
        )
        self.assertFalse(shadow["shadow_comparison"]["enabled"])
        self.assertFalse(shadow["shadow_comparison"]["pixels_written"])
        self.assertFalse(shadow["physical_color"]["accepted"])
        self.assertLess(
            events.index("stage4_color_saved"),
            events.index("auto_reference_report_written"),
        )

    def test_strict_m42_spcc_core_flip_is_repaired_without_forcing_review(self):
        source = _strict_bright_core_image()
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            image=source
        )
        pipeline.target_profile = _strict_profile()
        mask = np.zeros(source.shape[1:], dtype=bool)
        mask[126:129, 126:129] = True
        bad_spcc = _blue_flip_candidate(source, mask)
        pcc_calls = []

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = bad_spcc.copy()
            return True, "spcc imprecise candidate"

        pipeline._run_stage4_spcc_once = spcc_success
        pipeline._run_stage4_pcc_once = lambda **kwargs: (
            pcc_calls.append(kwargs) or True,
            "pcc must not run after a valid local repair",
        )

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        integrity = report["bright_core_color_integrity"]
        self.assertEqual(report["method"], "SPCC_LOCAL_CORE_CHROMA_ROLLBACK")
        self.assertEqual(integrity["status"], "repaired")
        self.assertTrue(integrity["repair"]["passed"])
        self.assertFalse(report["requires_review"])
        self.assertFalse(pcc_calls)
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertTrue(results[-1][1]["fallback_used"])

    def test_strict_m42_broad_core_platform_uses_bounded_review_rollback(self):
        source = _strict_bright_core_image()
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            image=source
        )
        pipeline.target_profile = _strict_profile()
        _base_report, context = (
            stage4.bright_core_color.assess_spcc_bright_core_color(
                source,
                source,
                target_type="bright_emission_reflection_nebula",
                target_profile=pipeline.target_profile,
            )
        )
        platform = np.zeros(source.shape[1:], dtype=bool)
        platform[124:133, 124:133] = True
        platform &= context["roi"]
        bad_spcc = _blue_flip_candidate(source, platform)
        pcc_calls = []

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = bad_spcc.copy()
            return True, "spcc broad core candidate"

        pipeline._run_stage4_spcc_once = spcc_success
        pipeline._run_stage4_pcc_once = lambda **kwargs: (
            pcc_calls.append(kwargs) or True,
            "pcc must not run after bounded broad-core repair",
        )

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        integrity = report["bright_core_color_integrity"]
        self.assertEqual(
            report["method"],
            "SPCC_BROAD_CORE_CHROMA_ROLLBACK",
        )
        self.assertEqual(integrity["status"], "repaired")
        self.assertTrue(integrity["repair"]["passed"])
        self.assertTrue(report["requires_review"])
        self.assertTrue(pipeline._stage4_color_review_required)
        self.assertFalse(report["physical_color"]["accepted"])
        self.assertTrue(report["degraded_color_correction"]["applied"])
        self.assertEqual(
            report["degraded_color_correction"]["method"],
            "SPCC_BROAD_CORE_CHROMA_ROLLBACK",
        )
        self.assertEqual(report["status"], "review_required")
        self.assertFalse(pcc_calls)
        self.assertEqual(
            report["auto_local_reference"]["status"],
            "shadow_skipped_degraded_color_accepted",
        )
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertTrue(results[-1][1]["fallback_used"])
        self.assertEqual(
            results[-1][1]["reason_code"],
            "spcc_broad_core_chroma_rollback_review_required",
        )

    def test_failed_strict_core_repair_rolls_back_before_pcc(self):
        source = _strict_bright_core_image(80)
        pipeline, saved, _commands, _results = _stage4_integration_fixture(
            image=source
        )
        pipeline.target_profile = _strict_profile()
        base_report, context = (
            stage4.bright_core_color.assess_spcc_bright_core_color(
                source,
                source,
                target_type="bright_emission_reflection_nebula",
                target_profile=pipeline.target_profile,
            )
        )
        self.assertEqual(base_report["roi"]["support_pixels"], 64)
        bad_spcc = _blue_flip_candidate(source, context["roi"])
        calibration_order = []

        def spcc_success(**_kwargs):
            calibration_order.append("spcc")
            saved[stage4.SPCC_CANDIDATE_STEM] = bad_spcc.copy()
            return True, "spcc bad core"

        def pcc_success(**_kwargs):
            calibration_order.append("pcc")
            saved[stage4.PCC_CANDIDATE_STEM] = source.copy()
            return True, "pcc ok"

        pipeline._run_stage4_spcc_once = spcc_success
        pipeline._run_stage4_pcc_once = pcc_success

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        integrity = report["bright_core_color_integrity"]
        self.assertEqual(calibration_order, ["spcc", "pcc"])
        self.assertEqual(report["method"], "PCC")
        self.assertEqual(integrity["status"], "ok")
        self.assertEqual(
            integrity["final_action"],
            "bad_spcc_rejected_and_safe_fallback_selected",
        )
        self.assertEqual(integrity["resolved_by"], "PCC")
        self.assertEqual(integrity["spcc_assessment_status"], "hard_failed")
        self.assertEqual(
            integrity["spcc_assessment_final_action"],
            "reject_spcc_to_pcc",
        )
        self.assertIn(
            "broad_core_chroma_platform",
            integrity["trigger_reasons"],
        )
        self.assertIn(
            "broad_core_chroma_platform",
            integrity["spcc_rejection_reasons"],
        )
        self.assertGreater(
            integrity["measurements"]["anomaly_ratio_of_roi"],
            0.02,
        )
        self.assertGreater(
            integrity["measurements"][
                "broad_platform_largest_component_ratio_of_roi"
            ],
            0.05,
        )
        np.testing.assert_array_equal(saved["stage4_color"], source)

    def test_strict_broad_platform_in_pcc_preserves_precolor_input(self):
        source = _strict_bright_core_image(80)
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            image=source
        )
        pipeline.target_profile = _strict_profile()
        _base_report, context = (
            stage4.bright_core_color.assess_spcc_bright_core_color(
                source,
                source,
                target_type="bright_emission_reflection_nebula",
                target_profile=pipeline.target_profile,
            )
        )
        broad_platform = _blue_flip_candidate(source, context["roi"])
        calibration_order = []

        def spcc_success(**_kwargs):
            calibration_order.append("spcc")
            saved[stage4.SPCC_CANDIDATE_STEM] = broad_platform.copy()
            return True, "spcc bad core"

        def pcc_success(**_kwargs):
            calibration_order.append("pcc")
            saved[stage4.PCC_CANDIDATE_STEM] = broad_platform.copy()
            return True, "pcc bad core"

        pipeline._run_stage4_spcc_once = spcc_success
        pipeline._run_stage4_pcc_once = pcc_success

        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(
                stage4,
                "evaluate_auto_local_reference",
                side_effect=AssertionError(
                    "unsafe PCC must preserve the exact pre-color source"
                ),
            ),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        integrity = report["bright_core_color_integrity"]
        pcc_integrity = report["pcc"]["quality_gate"][
            "bright_core_color_integrity"
        ]
        self.assertEqual(calibration_order, ["spcc", "pcc"])
        self.assertEqual(report["method"], "PRESERVE_INPUT")
        self.assertEqual(
            report["warning"],
            "pcc_bright_core_color_integrity_rejected_preserve_input",
        )
        self.assertFalse(report["pcc"]["quality_gate"]["accepted"])
        self.assertIn(
            "bright_core_color_integrity_rejected",
            report["pcc"]["quality_gate"]["rejection_reasons"],
        )
        self.assertEqual(pcc_integrity["status"], "hard_failed")
        self.assertIn(
            "broad_core_chroma_platform",
            pcc_integrity["trigger_reasons"],
        )
        self.assertEqual(integrity["status"], "ok")
        self.assertEqual(integrity["resolved_by"], "PRESERVE_INPUT")
        self.assertEqual(
            integrity["pcc_fallback_assessment"]["status"],
            "hard_failed",
        )
        np.testing.assert_array_equal(saved["stage4_color"], source)
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertTrue(results[-1][1]["fallback_used"])

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

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
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
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
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
        self.assertEqual(
            report["auto_local_reference"]["status"],
            "shadow_skipped_physical_color_accepted",
        )
        self.assertFalse(report["artistic_hoo"]["feeds_main_pipeline"])
        self.assertEqual(
            report["artistic_hoo"]["physical_parent"],
            "stage4_physical_color.fit",
        )

    def test_narrowband_spcc_timeout_uses_degraded_pcc_for_review(self):
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            filter_name="Ha+OIII dual-band"
        )
        pcc_calls = []
        pipeline._run_stage4_spcc_once = lambda **_kwargs: (
            False,
            "SPCC linear_dual_narrowband_physical timed out after 300s",
        )

        def pcc_success(**kwargs):
            pcc_calls.append(kwargs)
            saved[stage4.PCC_CANDIDATE_STEM] = np.full(
                (3, 64, 64),
                0.05,
                dtype=np.float32,
            )
            return True, "pcc degraded ok"

        def hoo_derivative(_pipeline, _metadata):
            _pipeline.current = np.clip(_pipeline.current * 1.5, 0.0, 1.0)
            return True, {"accepted": True, "status": "accepted"}, "HOO accepted"

        pipeline._run_stage4_pcc_once = pcc_success
        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(stage4, "_stage4_run_narrowband_normalization", hoo_derivative),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(len(pcc_calls), 1)
        self.assertEqual(
            report["pcc"]["attempts"][0]["phase"],
            "dual_narrowband_spcc_degraded_fallback",
        )
        self.assertEqual(report["method"], "PCC_NARROWBAND_DEGRADED")
        self.assertTrue(report["pcc"]["used"])
        self.assertTrue(report["pcc"]["degraded"])
        self.assertFalse(report["pcc"]["physical_color"])
        self.assertTrue(report["requires_review"])
        self.assertTrue(pipeline._stage4_color_review_required)
        self.assertFalse(report["physical_color"]["accepted"])
        self.assertIsNone(report["physical_color"]["output"])
        self.assertNotIn(stage4.PHYSICAL_COLOR_STEM, saved)
        self.assertTrue(report["degraded_color_correction"]["applied"])
        self.assertEqual(
            report["artistic_hoo"]["source_parent"],
            "stage4_pcc_candidate.fit",
        )
        self.assertIsNone(report["artistic_hoo"]["physical_parent"])
        self.assertTrue(
            np.allclose(saved["stage4_color"], saved[stage4.PCC_CANDIDATE_STEM])
        )
        self.assertFalse(
            np.allclose(saved["stage4_color"], saved[stage4.HOO_ARTISTIC_STEM])
        )
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertTrue(results[-1][1]["fallback_used"])
        self.assertEqual(
            results[-1][1]["reason_code"],
            "narrowband_pcc_degraded_fallback",
        )

    def test_narrowband_physical_restore_failure_uses_pre_color_for_main_pipeline(self):
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            filter_name="Ha+OIII dual-band"
        )
        original_command = pipeline.cmd_with_check

        def command(*args, **kwargs):
            if args[:2] == ("load", stage4.PHYSICAL_COLOR_STEM):
                raise stage4.CommandError("physical checkpoint unreadable")
            return original_command(*args, **kwargs)

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc narrowband ok"

        def hoo_derivative(_pipeline, _metadata):
            _pipeline.current = np.clip(_pipeline.current * 1.5, 0.0, 1.0)
            return True, {"accepted": True, "status": "accepted"}, "HOO accepted"

        pipeline.cmd_with_check = command
        pipeline._run_stage4_spcc_once = spcc_success

        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(stage4, "_stage4_run_narrowband_normalization", hoo_derivative),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertTrue(report["requires_review"])
        self.assertEqual(report["method"], "PRESERVE_INPUT")
        self.assertFalse(report["physical_color"]["accepted"])
        self.assertEqual(
            report["physical_color"]["main_pipeline_restore"]["source"],
            "stage4_pre_pcc.fit",
        )
        self.assertTrue(
            report["physical_color"]["main_pipeline_restore"]["fallback_used"]
        )
        self.assertTrue(
            np.allclose(saved["stage4_color"], saved[stage4.PCC_CHECKPOINT_STEM])
        )
        self.assertFalse(
            np.allclose(saved["stage4_color"], saved[stage4.HOO_ARTISTIC_STEM])
        )
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertTrue(results[-1][1]["fallback_used"])

    def test_narrowband_restore_uses_in_memory_pre_color_after_file_failures(self):
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            filter_name="Ha+OIII dual-band"
        )
        original_command = pipeline.cmd_with_check

        def command(*args, **kwargs):
            if args[:2] in {
                ("load", stage4.PHYSICAL_COLOR_STEM),
                ("load", stage4.PCC_CHECKPOINT_STEM),
            }:
                raise stage4.CommandError("checkpoint unreadable")
            return original_command(*args, **kwargs)

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc narrowband ok"

        def hoo_derivative(_pipeline, _metadata):
            _pipeline.current = np.clip(_pipeline.current * 1.5, 0.0, 1.0)
            return True, {"accepted": True, "status": "accepted"}, "HOO accepted"

        pipeline.cmd_with_check = command
        pipeline._run_stage4_spcc_once = spcc_success

        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(stage4, "_stage4_run_narrowband_normalization", hoo_derivative),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(
            report["physical_color"]["main_pipeline_restore"]["source"],
            "in_memory_pre_color",
        )
        self.assertTrue(report["requires_review"])
        self.assertTrue(
            np.allclose(saved["stage4_color"], saved["stage3_bgremoved"])
        )
        self.assertFalse(
            np.allclose(saved["stage4_color"], saved[stage4.HOO_ARTISTIC_STEM])
        )
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertTrue(results[-1][1]["fallback_used"])

    def test_narrowband_restore_failure_blocks_main_output_when_no_safe_source_exists(self):
        pipeline, saved, _commands, results = _stage4_integration_fixture(
            filter_name="Ha+OIII dual-band"
        )
        original_command = pipeline.cmd_with_check

        def command(*args, **kwargs):
            if args[:2] in {
                ("load", stage4.PHYSICAL_COLOR_STEM),
                ("load", stage4.PCC_CHECKPOINT_STEM),
            }:
                raise stage4.CommandError("checkpoint unreadable")
            return original_command(*args, **kwargs)

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc narrowband ok"

        def hoo_derivative(_pipeline, _metadata):
            _pipeline.current = np.clip(_pipeline.current * 1.5, 0.0, 1.0)
            return True, {"accepted": True, "status": "accepted"}, "HOO accepted"

        pipeline.cmd_with_check = command
        pipeline._run_stage4_spcc_once = spcc_success
        pipeline.siril.set_image_pixeldata = lambda _pixels: (_ for _ in ()).throw(
            RuntimeError("in-memory restore rejected")
        )

        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(stage4, "_stage4_run_narrowband_normalization", hoo_derivative),
            self.assertRaisesRegex(
                RuntimeError,
                "immutable pre-color baseline could not be restored",
            ),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        self.assertNotIn("stage4_color", saved)
        self.assertIsNone(pipeline.color_calibration_report["outputs"]["color"])
        self.assertEqual(pipeline.color_calibration_report["status"], "failed")
        self.assertTrue(pipeline.color_calibration_report["main_output_blocked"])
        self.assertEqual(
            pipeline.color_calibration_report["components"]["color_calibration"][
                "status"
            ],
            "failed",
        )
        self.assertFalse(
            pipeline.color_calibration_report["physical_color"][
                "feeds_main_pipeline"
            ]
        )
        self.assertEqual(results[-1][0][1], "failed")
        self.assertEqual(results[-1][1]["reason_code"], "stage4_main_output_blocked")

    def test_local_star_restore_is_not_reported_as_accepted_physical_calibration(self):
        pipeline, _saved, _commands, _results = _stage4_integration_fixture()
        pipeline._run_stage4_spcc_once = lambda **_kwargs: (False, "spcc failed")
        pipeline._run_stage4_pcc_once = lambda **_kwargs: (False, "pcc failed")
        fallback_report = {
            "star_white_balance": {"applied": True},
            "global_white_balance": {"applied": False, "prohibited": True},
        }

        with (
            patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False),
            patch.object(
                stage4,
                "_stage4_local_color_fallback",
                return_value=(
                    True,
                    "LOCAL_STAR_COLOR_RESTORE",
                    "local_star_mask_fallback",
                    0.55,
                    fallback_report,
                    "local fallback",
                ),
            ),
        ):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "LOCAL_STAR_COLOR_RESTORE")
        self.assertEqual(report["components"]["color_calibration"]["status"], "applied")
        self.assertFalse(report["physical_color"]["accepted"])
        self.assertTrue(report["requires_review"])

    def test_unsaved_color_output_is_not_declared_in_report(self):
        pipeline, saved, _commands, results = _stage4_integration_fixture()
        original_save = pipeline._save_stage_output

        def save(stem):
            if stem == "stage4_color":
                return False
            return original_save(stem)

        def spcc_success(**_kwargs):
            saved[stage4.SPCC_CANDIDATE_STEM] = pipeline.current.copy()
            return True, "spcc ok"

        pipeline._save_stage_output = save
        pipeline._run_stage4_spcc_once = spcc_success

        with patch.dict(os.environ, {"STARUN_NETWORK_MODE": "1"}, clear=False):
            stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertIsNone(report["outputs"]["color"])
        self.assertEqual(report["status"], "review_required")
        self.assertIsNone(report["components"]["color_calibration"]["output"])
        self.assertEqual(report["components"]["color_calibration"]["status"], "failed")
        self.assertFalse(report["physical_color"]["feeds_main_pipeline"])
        self.assertEqual(results[-1][0][1], "degraded")

    def test_disabled_platesolve_keeps_mono_stage_degraded(self):
        pipeline, _saved, _commands, results = _stage4_integration_fixture()
        pipeline.cfg.stage4_platesolve_enabled = False
        pipeline.siril.get_image_shape = lambda: (1, 64, 64)

        stage4.run_stage4_color_calibration(pipeline)

        report = pipeline.color_calibration_report
        self.assertEqual(report["method"], "SKIPPED_MONO")
        self.assertFalse(report["requires_review"])
        self.assertEqual(results[-1][0][1], "degraded")
        self.assertEqual(results[-1][1]["reason_code"], "stage4_platesolve_disabled")


if __name__ == "__main__":
    unittest.main()
