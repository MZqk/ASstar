#!/usr/bin/env python3
"""Safe-passthrough tests for user-preserve processing modes."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import run_manifest
from pipeline import stage5_handoff
from pipeline.models import (
    ImageFeatures,
    InputProfile,
    InputState,
    PipelineConfig,
    StarSeparationState,
)
from pipeline.processing_parameters import (
    GATE_PROFILE_UNLIMITED,
    default_processing_parameters,
    normalize_processing_parameters,
    processing_gate_profile_audit,
)

# GUI tests may have installed a deliberately minimal sirilpy stub first.
# Complete that shared stub before importing the pipeline stage smoke fixtures.
if "sirilpy" in sys.modules:
    exceptions = sys.modules.get("sirilpy.exceptions")
    if exceptions is None:
        exceptions = types.ModuleType("sirilpy.exceptions")
        sys.modules["sirilpy.exceptions"] = exceptions

    class _SirilError(Exception):
        pass

    if not hasattr(exceptions, "SirilError"):
        exceptions.SirilError = _SirilError
    if not hasattr(exceptions, "SirilConnectionError"):
        exceptions.SirilConnectionError = type(
            "SirilConnectionError", (exceptions.SirilError,), {}
        )
    if not hasattr(exceptions, "CommandError"):
        exceptions.CommandError = type(
            "CommandError", (exceptions.SirilError,), {}
        )
    if not hasattr(exceptions, "DataError"):
        exceptions.DataError = type("DataError", (exceptions.SirilError,), {})
    if not hasattr(sys.modules["sirilpy"], "SirilInterface"):
        sys.modules["sirilpy"].SirilInterface = object
from tests.test_pipeline_stage_smoke import (
    _Pipeline,
    stage2_view_correction,
    stage3_background_extraction,
    stage4_color_calibration,
    stage5_linear_denoise,
    stage6_star_separation,
)
from pipeline.stage6_services import Stage6ServiceMixin
from processor_runtime import ProcessorRuntimeMixin


class ProcessingParameterPassthroughTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.pipeline = _Pipeline(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _freeze_current_stage5_handoff(self) -> dict:
        (self.pipeline.process_dir / "stage4_color.fit").write_bytes(b"stage4")
        (self.pipeline.process_dir / "stage5_input_linear.fit").write_bytes(
            b"baseline"
        )
        input_lineage = stage5_handoff.freeze_stage5_input_lineage(
            self.pipeline,
            upstream_loaded=True,
            baseline_saved=True,
        )
        return stage5_handoff.freeze_stage5_handoff(
            self.pipeline,
            origin=stage5_handoff.CURRENT_RUN_ORIGIN,
            stage_status="ok",
            deconvolution_integrity_ok=True,
            denoise_integrity_ok=True,
            input_lineage=input_lineage,
        )

    def test_project_default_env_enables_stage5_denoise(self) -> None:
        runtime = ProcessorRuntimeMixin()
        runtime.cfg = PipelineConfig(
            denoise_enabled=False,
            stage8_target_aware_chroma_enabled=False,
        )
        runtime._force_denoise_enabled = None
        runtime.log = self.pipeline.log
        runtime._sync_logger_level = lambda: None

        with patch.dict(os.environ, {}, clear=True):
            runtime._load_project_env_defaults()
            runtime._apply_runtime_env_overrides()

            self.assertEqual(os.environ["STARUN_DENOISE_ENABLE"], "1")
            self.assertTrue(runtime.cfg.denoise_enabled)
            self.assertEqual(
                os.environ["STARUN_STAGE8_TARGET_AWARE_CHROMA_ENABLE"],
                "1",
            )
            self.assertTrue(runtime.cfg.stage8_target_aware_chroma_enabled)
            for retired in (
                "STARUN_STAGE4_NBN_ENABLE",
                "STARUN_STAGE4_NBN_STRENGTH",
                "STARUN_STAGE4_NBN_GAIN_LIMIT",
                "STARUN_STAGE4_NBN_LINE_RATIO_DRIFT_MAX",
            ):
                self.assertNotIn(retired, os.environ)
            self.assertNotIn("STARUN_STAGE4_LOCAL_STAR_WB_ENABLE", os.environ)
            self.assertNotIn(
                "STARUN_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT",
                os.environ,
            )

    def test_stage8_target_aware_chroma_env_override_is_applied(self) -> None:
        runtime = ProcessorRuntimeMixin()
        runtime.cfg = PipelineConfig(stage8_target_aware_chroma_enabled=True)
        runtime.log = self.pipeline.log
        runtime._sync_logger_level = lambda: None

        with patch.dict(
            os.environ,
            {"STARUN_STAGE8_TARGET_AWARE_CHROMA_ENABLE": "0"},
            clear=True,
        ):
            runtime._apply_runtime_env_overrides()

        self.assertFalse(runtime.cfg.stage8_target_aware_chroma_enabled)

    def test_stage7_luma_noise_growth_env_is_loaded_and_clamped(self) -> None:
        runtime = ProcessorRuntimeMixin()
        runtime.cfg = PipelineConfig()
        runtime.log = self.pipeline.log
        runtime._sync_logger_level = lambda: None

        with patch.dict(
            os.environ,
            {"STARUN_STAGE7_STRETCH_LUMA_NOISE_GROWTH_MAX": "9.0"},
            clear=False,
        ):
            runtime._apply_runtime_env_overrides()

        self.assertEqual(runtime.cfg.stage7_stretch_luma_noise_growth_max, 3.0)

    def test_stage2_user_preserve_writes_canonical_artifact(self) -> None:
        self.pipeline.cfg.stage2_processing_mode = "preserve"

        stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.pipeline.process_dir / "stage2_corrected.fit").is_file())
        self.assertEqual(self.pipeline.stage2_crop_report["mode"], "user_preserve")
        self.assertEqual(
            self.pipeline.stage2_crop_report["execution"], "safe_passthrough"
        )

    def test_stage2_explicit_base_crop_runs_before_builtin_detection(self) -> None:
        self.pipeline.cfg.stage2_base_crop_enabled = True
        self.pipeline.cfg.stage2_base_crop_margin = 0.02
        self.pipeline.cfg.stage2_color_edge_cleanup_enabled = False
        with (
            patch.object(
                stage2_view_correction,
                "_detect_native_contour_candidate",
                return_value=(
                    None,
                    "native contour crop skipped: no additional crop",
                    {
                        "method": "native_contour",
                        "accepted": False,
                        "reason": "no_candidate",
                        "candidate": None,
                    },
                ),
            ),
            patch.object(
                stage2_view_correction,
                "_detect_auto_edge_crop",
                return_value=(None, "no additional crop"),
            ),
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(
            self.pipeline.stage2_crop_report["crops"][0]["reason"],
            "user_base_crop",
        )
        self.assertTrue((self.pipeline.process_dir / "stage2_corrected.fit").is_file())

    def test_stage3_user_preserve_keeps_diagnostics_and_writes_artifact(self) -> None:
        self.pipeline.cfg.stage3_processing_mode = "preserve"
        self.pipeline._stage3_measure_features = lambda _label: ImageFeatures()
        self.pipeline._stage3_signal_preservation_metrics = lambda *_args: {}
        self.pipeline._stage3_quality_gate = lambda *_args: (True, "ok")
        self.pipeline._stage3_subsky_rbf_candidates = lambda: []
        self.pipeline.workflow_plugin_probe_enabled = False
        with (
            patch.object(
                stage3_background_extraction,
                "_stage3_background_candidate_chain",
                return_value=([], [], "user_preserve"),
            ),
            patch.object(
                stage3_background_extraction,
                "_stage3_theoretical_plugin_candidates",
                return_value=[],
            ),
            patch.object(
                stage3_background_extraction,
                "_stage3_graxpert_candidates",
                return_value=[],
            ),
        ):
            stage3_background_extraction.run_stage3_background_extraction(
                self.pipeline
            )

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.pipeline.process_dir / "stage3_bgremoved.fit").is_file())
        self.assertEqual(
            self.pipeline._stage3_background_decision["source"],
            "user_processing_parameters",
        )

    def test_stage4_user_preserve_skips_solve_and_color_commands(self) -> None:
        self.pipeline.cfg.stage4_processing_mode = "preserve"
        self.pipeline._read_fits_metadata = lambda *_args: {}
        with (
            patch.object(
                stage4_color_calibration,
                "_stage4_header_metadata",
                return_value={},
            ),
            patch.object(
                stage4_color_calibration,
                "_stage4_image_geometry",
                return_value={"current_shape": {}},
            ),
        ):
            stage4_color_calibration.run_stage4_color_calibration(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.pipeline.process_dir / "stage4_color.fit").is_file())
        self.assertEqual(
            self.pipeline.color_calibration_report["reason_code"],
            "user_preserve",
        )
        self.assertFalse(any(command[0] == "platesolve" for command in self.pipeline.commands))

    def test_stage5_user_preserve_keeps_linear_artifact_and_diagnostics(self) -> None:
        (self.pipeline.process_dir / "stage4_color.fit").write_bytes(b"stage4")
        self.pipeline.cfg.stage5_processing_mode = "preserve"
        self.pipeline._export_linear_intermediate = lambda: True
        reports = {}
        self.pipeline._write_stage_json = (
            lambda name, payload: reports.__setitem__(name, payload)
        )

        stage5_linear_denoise.run_stage5_linear_denoise(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.pipeline.process_dir / "stage5_linear.fit").is_file())
        report = reports["stage5_linear_report.json"]
        self.assertEqual(report["reason_code"], "user_preserve")
        self.assertEqual(report["deconvolution"]["status"], "skipped")
        self.assertEqual(report["denoise"]["status"], "skipped")

    def test_stage6_user_preserve_establishes_starless_bypass_state(self) -> None:
        (self.pipeline.process_dir / "stage5_linear.fit").touch()
        self._freeze_current_stage5_handoff()
        self.pipeline.cfg.stage6_processing_mode = "preserve"

        stage6_star_separation.run_stage6_star_separation(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.pipeline.process_dir / "stage6_passthrough.fit").is_file())
        self.assertEqual(
            self.pipeline._star_separation_state,
            StarSeparationState.TARGET_BYPASS.value,
        )
        self.assertTrue(self.pipeline._stage7_starless_skipped)
        self.assertTrue(self.pipeline._stage8_handoff["passthrough"])
        self.assertEqual(self.pipeline._stage8_handoff["reason_code"], "user_preserve")

    def test_stage6_user_preserve_never_falls_through_when_save_fails(self) -> None:
        (self.pipeline.process_dir / "stage5_linear.fit").touch()
        self._freeze_current_stage5_handoff()
        self.pipeline.cfg.stage6_processing_mode = "preserve"
        original_save = self.pipeline._save_stage_output
        self.pipeline._save_stage_output = lambda stem: (
            False if stem == "stage6_passthrough" else original_save(stem)
        )

        stage6_star_separation.run_stage6_star_separation(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertEqual(self.pipeline._stage6_passthrough_source, "stage5_linear")
        self.assertEqual(
            self.pipeline._star_separation_state,
            StarSeparationState.TARGET_BYPASS.value,
        )
        self.assertEqual(self.pipeline._stage8_handoff["source_stem"], "stage5_linear")
        self.assertFalse(any(command[0] == "syqon" for command in self.pipeline.commands))

    def test_stage7_candidates_keep_manual_asinh_and_ghs_values(self) -> None:
        stretch_service = Stage6ServiceMixin()
        stretch_service.cfg = self.pipeline.cfg
        stretch_service.pipeline_policy = {}
        stretch_service.cfg.asinh_stretch = 3.4
        stretch_service.cfg.asinh_offset = 0.0042
        stretch_service.cfg.ghs_stretchamount = 2.6
        stretch_service._task_manual_override_fields = (
            "asinh_stretch",
            "asinh_offset",
            "ghs_stretchamount",
        )

        candidates, adaptation = (
            stretch_service._stage7_compact_stretch_candidates(
                None,
                {"bg_median": 0.02, "bg_std": 0.002},
                {"p50": 0.02, "p99": 0.20, "max": 0.30},
                {"p50": 0.25, "p99": 0.80},
                starless_recomposition_planned=True,
            )
        )

        self.assertEqual(candidates[0]["params"]["asinh_stretch"], 3.4)
        self.assertEqual(candidates[0]["params"]["asinh_offset"], 0.0042)
        self.assertEqual(candidates[1]["params"]["asinh_stretch"], 3.4)
        self.assertEqual(candidates[1]["params"]["ghs_stretchamount"], 2.6)
        self.assertEqual(candidates[1]["method"], "asinh_ghs")
        self.assertEqual(
            adaptation["manual_parameter_overrides"]["fields"],
            ["asinh_offset", "asinh_stretch", "ghs_stretchamount"],
        )
        self.assertTrue(
            adaptation["manual_parameter_overrides"][
                "adaptive_replacement_disabled"
            ]
        )
        self.assertEqual(adaptation["parameter_mode"], "manual")
        candidate_a_contract = adaptation["preview_calibration"]["candidate_a"]
        self.assertEqual(
            candidate_a_contract["calibration_method"],
            "manual_contract_rebased",
        )
        self.assertEqual(
            candidate_a_contract["target_p50"],
            candidate_a_contract["predicted_p50"],
        )
        self.assertEqual(
            adaptation["preview_calibration"]["candidate_b"],
            {},
        )
        self.assertEqual(
            adaptation["manual_parameter_overrides"]["target_contracts"][
                "candidate_b"
            ]["mode"],
            "manual_independent_safety_gates",
        )

    def test_stage7_override_automatically_selects_manual_parameter_mode(self) -> None:
        parameters = default_processing_parameters()
        parameters["stages"]["7"]["overrides"]["asinh_stretch"] = 3.3

        normalized, adjustments = normalize_processing_parameters(parameters)

        self.assertEqual(normalized["stages"]["7"]["mode"], "manual")
        self.assertTrue(
            any(
                record.get("reason") == "manual_overrides_require_manual_mode"
                for record in adjustments
            )
        )

    def test_failure_action_preserve_marks_review_and_stop_raises_after_result(self) -> None:
        failed_result = types.SimpleNamespace(
            name="阶段 8: 星云增强",
            status="failed",
            reason_code="quality_gate_failed",
            message="",
        )
        runtime = ProcessorRuntimeMixin()
        runtime.cfg = PipelineConfig()
        runtime.log = self.pipeline.log
        runtime.results = [failed_result]
        runtime._stage_policy_events = []
        runtime.cfg.stage8_failure_action = "preserve_review"

        runtime._enforce_stage_failure_action(8)

        self.assertEqual(
            runtime._stage_review_reasons(8),
            ["failure_policy_preserve_review"],
        )
        self.assertEqual(runtime._stage_review_reasons(3), [])
        self.assertEqual(runtime._stage_policy_events[-1]["event"], "decisive_failure")

        runtime.cfg.stage8_failure_action = "stop"
        with self.assertRaisesRegex(RuntimeError, "用户严格停止"):
            runtime._enforce_stage_failure_action(8)

    def test_signed_manifest_general_and_stage_overrides_are_authoritative(self) -> None:
        parameters = default_processing_parameters(
            general={
                "output_formats": ["fit"],
                "review_only": True,
                "compute_mode": "cpu",
                "auto_tune_enabled": False,
                "max_retries": 3,
                "retry_delay": 2.5,
                "review_bundle_enabled": False,
                "managed_output_enabled": False,
            }
        )
        parameters["stages"]["1"]["overrides"][
            "stage1_register_fail_ratio_max"
        ] = 0.20
        parameters["stages"]["7"]["overrides"]["asinh_stretch"] = 3.3
        parameters["stages"]["7"]["overrides"][
            "stage7_9_quality_advisory_multiplier"
        ] = 1.6
        parameters["stages"]["8"]["overrides"][
            "stage8_dualband_palette_selection"
        ] = "OHS"
        parameters["stages"]["9"]["overrides"].update(
            {
                "stage9_quality_gate_enabled": False,
                "stage9_highlight_clip_growth_max": 0.01,
                "stage9_psf_recovery_target_min": 0.99,
                "stage9_targeted_recovery_enabled": False,
                "stage9_targeted_recovery_retry_max": 2,
            }
        )
        parameters["stages"]["10"].update(
            {
                "mode": "preserve",
                "overrides": {
                    "stage10_denoise_backend_policy": "scunet_only",
                    "stage10_quality_repair_enabled": False,
                },
            }
        )
        manifest = {
            "schema": "starun.task-run.v1",
            "processing_parameters": parameters,
            "processing_gate_profile": processing_gate_profile_audit(parameters),
            "processing_parameter_adjustments": [],
        }
        manifest["manifest_hash"] = run_manifest.canonical_payload_hash(manifest)
        manifest_path = self.root / "run-manifest.json"
        run_manifest.atomic_write_json(manifest_path, manifest)
        runtime = ProcessorRuntimeMixin()
        runtime.cfg = PipelineConfig()
        runtime.log = self.pipeline.log

        with patch.dict(
            os.environ,
            {
                "STARUN_TASK_RUN_MANIFEST": str(manifest_path),
                "STARUN_SYQON_GPU": "legacy_ignored",
            },
            clear=False,
        ):
            runtime._load_task_processing_parameters()
            runtime._apply_task_processing_parameter_overrides()

            self.assertEqual(os.environ["STARUN_SYQON_GPU"], "legacy_ignored")
            self.assertEqual(os.environ["STARUN_GRAXPERT_GPU"], "0")

        self.assertEqual(runtime.cfg.output_format, "fit")
        self.assertTrue(runtime.cfg.force_review_only_output)
        self.assertFalse(runtime.cfg.auto_tune_enabled)
        self.assertEqual(runtime.cfg.max_retries, 3)
        self.assertEqual(runtime.cfg.retry_delay, 2.5)
        self.assertFalse(runtime.cfg.review_bundle_enabled)
        self.assertFalse(runtime.cfg.stage10_managed_output_enabled)
        self.assertEqual(runtime.cfg.stage1_register_fail_ratio_max, 0.20)
        self.assertEqual(runtime.cfg.asinh_stretch, 3.3)
        self.assertEqual(runtime.cfg.stage7_9_quality_advisory_multiplier, 1.6)
        self.assertEqual(runtime.cfg.stage7_processing_mode, "manual")
        self.assertEqual(runtime.cfg.stage8_dualband_palette_selection, "OHS")
        self.assertFalse(runtime.cfg.stage9_quality_gate_enabled)
        self.assertEqual(runtime.cfg.stage9_highlight_clip_growth_max, 0.01)
        self.assertEqual(runtime.cfg.stage9_psf_recovery_target_min, 0.99)
        self.assertFalse(runtime.cfg.stage9_targeted_recovery_enabled)
        self.assertEqual(runtime.cfg.stage9_targeted_recovery_retry_max, 2)
        self.assertEqual(runtime.cfg.stage10_processing_mode, "preserve")
        self.assertEqual(
            runtime.cfg.stage10_denoise_backend_policy,
            "scunet_only",
        )
        self.assertFalse(runtime.cfg.stage10_quality_repair_enabled)
        self.assertIn("asinh_stretch", runtime._task_manual_override_fields)
        self.assertIn(
            "stage9_highlight_clip_growth_max",
            runtime._task_manual_override_fields,
        )
        self.assertIn(
            "stage7_9_quality_advisory_multiplier",
            runtime._task_manual_override_fields,
        )
        self.assertIn(
            "stage7_processing_mode",
            runtime._task_manual_override_fields,
        )
        self.assertIn(
            "stage8_dualband_palette_selection",
            runtime._task_manual_override_fields,
        )

    def test_processing_plan_freezes_manual_palette_for_stage8(self) -> None:
        runtime = ProcessorRuntimeMixin()
        runtime.cfg = PipelineConfig()
        runtime.cfg.stage8_dualband_palette_selection = "OHS"
        runtime.work_dir = self.root
        runtime.process_dir = self.root / "process"
        runtime.process_dir.mkdir(exist_ok=True)
        runtime.input_mode = "auto"
        runtime._stage1_input_mode = "stacked"
        runtime._channel_semantics = "narrowband_composite"
        runtime.target_profile = {
            "target_type": "emission_nebula_widefield",
            "target_confidence": 0.95,
            "primary_target": {
                "type": "emission_nebula_widefield",
                "confidence": 0.95,
                "method": "catalog",
                "frozen": True,
            },
        }
        runtime.pipeline_policy = {}
        runtime.log = self.pipeline.log
        runtime._resolve_channel_profile = lambda _profile: {
            "kind": "narrowband_composite"
        }
        runtime._processing_software_identity = lambda: {}
        parameters = default_processing_parameters()
        parameters["stages"]["8"]["overrides"][
            "stage8_dualband_palette_selection"
        ] = "OHS"
        runtime._task_processing_parameters = parameters
        runtime._task_processing_parameter_request = parameters
        profile = InputProfile(
            state=InputState.LINEAR,
            confidence=1.0,
            source="test",
            input_mode="auto",
        )

        self.assertTrue(runtime._write_processing_plan(profile))

        selection = runtime._processing_plan["metadata"]["candidate_contracts"][
            "stage8_enhancement"
        ]["dualband_palette"]["selection"]
        self.assertEqual(selection, runtime._stage8_palette_selection)
        self.assertEqual(selection["requested_palette"], "OHS")
        self.assertEqual(selection["automatic_palette"], "SHO")
        self.assertEqual(selection["palette"], "OHS")
        self.assertEqual(selection["selection_mode"], "explicit_user_palette")
        self.assertTrue(selection["manual_override"])
        persisted = run_manifest.load_json(self.root / "processing-plan.json")
        self.assertEqual(
            persisted["metadata"]["candidate_contracts"]["stage8_enhancement"][
                "dualband_palette"
            ]["selection"],
            selection,
        )

    def test_unlimited_gate_profile_forces_review_and_static_gate_baseline(self) -> None:
        parameters = default_processing_parameters(
            general={"review_only": False}
        )
        parameters["gate_profile"] = GATE_PROFILE_UNLIMITED
        manifest = {
            "schema": "starun.task-run.v1",
            "processing_parameters": parameters,
            "processing_parameter_adjustments": [],
        }
        manifest["manifest_hash"] = run_manifest.canonical_payload_hash(manifest)
        manifest_path = self.root / "unlimited-run-manifest.json"
        run_manifest.atomic_write_json(manifest_path, manifest)
        runtime = ProcessorRuntimeMixin()
        runtime.cfg = PipelineConfig()
        runtime.cfg.stage8_mask_signal_coverage_min = 0.001
        runtime.log = self.pipeline.log

        with patch.dict(
            os.environ,
            {"STARUN_TASK_RUN_MANIFEST": str(manifest_path)},
            clear=False,
        ):
            runtime._load_task_processing_parameters()
            runtime._apply_task_processing_parameter_overrides()

        self.assertTrue(runtime.cfg.force_review_only_output)
        self.assertAlmostEqual(
            runtime.cfg.stage8_mask_signal_coverage_min,
            PipelineConfig().stage8_mask_signal_coverage_min / 10.0,
        )
        self.assertEqual(runtime._task_gate_profile_audit["profile"], "unlimited")


if __name__ == "__main__":
    unittest.main()
