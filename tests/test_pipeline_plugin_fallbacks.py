#!/usr/bin/env python3
"""Fallback and degrade behavior tests for pipeline stages 4/5/7/10."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
PIPELINE_MODULE_PATH = REPO_ROOT / "pipeline" / "seestar_Superimpose.py"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _ensure_fake_sirilpy() -> None:
    if "sirilpy" in sys.modules:
        return

    fake_sirilpy = types.ModuleType("sirilpy")
    fake_exceptions = types.ModuleType("sirilpy.exceptions")
    fake_enums = types.ModuleType("sirilpy.enums")

    class _SirilError(Exception):
        pass

    class _SirilConnectionError(_SirilError):
        pass

    class _CommandError(_SirilError):
        pass

    class _DataError(_SirilError):
        pass

    class _CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    class _SirilInterface:
        def cmd(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    fake_sirilpy.SirilInterface = _SirilInterface
    fake_exceptions.SirilError = _SirilError
    fake_exceptions.SirilConnectionError = _SirilConnectionError
    fake_exceptions.CommandError = _CommandError
    fake_exceptions.DataError = _DataError
    fake_enums.CommandStatus = _CommandStatus

    sys.modules["sirilpy"] = fake_sirilpy
    sys.modules["sirilpy.exceptions"] = fake_exceptions
    sys.modules["sirilpy.enums"] = fake_enums


def _ensure_fake_numpy() -> None:
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # type: ignore

        _ = numpy
        return
    except Exception:
        pass

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = float
    fake_numpy.uint16 = int
    fake_numpy.uint8 = int
    fake_numpy.integer = int
    fake_numpy.ndarray = object

    def _asarray(value: Any):
        return value

    def _transpose(value: Any, _axes: Any):
        return value

    def _issubdtype(_lhs: Any, _rhs: Any) -> bool:
        return False

    def _clip(value: Any, _vmin: Any, _vmax: Any):
        return value

    fake_numpy.asarray = _asarray
    fake_numpy.transpose = _transpose
    fake_numpy.issubdtype = _issubdtype
    fake_numpy.clip = _clip
    sys.modules["numpy"] = fake_numpy


def _load_pipeline_module():
    _ensure_fake_numpy()
    _ensure_fake_sirilpy()
    spec = importlib.util.spec_from_file_location(
        "seestar_pipeline_test_module",
        PIPELINE_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {PIPELINE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline_module = _load_pipeline_module()
stage_support_module = sys.modules["stage_support"]
stage7_star_separation_module = sys.modules["stages.stage7_star_separation"]
stage7_quality_module = sys.modules["stage7_quality"]
stage4_color_calibration = pipeline_module.SeestarPostProcessor.stage4_color_calibration
stage5_linear_denoise = pipeline_module.SeestarPostProcessor.stage5_linear_denoise
stage2_view_correction = pipeline_module.SeestarPostProcessor.stage2_view_correction
# Test the canonical Stage 6/7 entry points. The similarly named methods below
# are compatibility aliases retained for old callers, not the production order.
stage7_stretching = pipeline_module.SeestarPostProcessor.stage7_stretching
stage6_star_separation = pipeline_module.SeestarPostProcessor.stage6_star_separation
stage8_nebula_enhancement = pipeline_module.SeestarPostProcessor.stage8_nebula_enhancement
stage9_star_remixing = pipeline_module.SeestarPostProcessor.stage9_star_remixing
stage10_export = pipeline_module.SeestarPostProcessor.stage10_export


class FakeLogger:
    _LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.min_level = self._LEVELS["DEBUG"]

    def stage_start(self, name: str) -> None:
        self.events.append(("stage_start", name))

    def stage_end(self, name: str | None = None) -> float:
        self.events.append(("stage_end", name or ""))
        return 0.01

    def info(self, msg: str) -> None:
        self.events.append(("info", msg))

    def warn(self, msg: str) -> None:
        self.events.append(("warn", msg))

    def error(self, msg: str) -> None:
        self.events.append(("error", msg))

    def debug(self, msg: str) -> None:
        self.events.append(("debug", msg))


class Stage3TransactionFake:
    """Minimal buffer model for Stage 3 rollback regression tests."""

    def __init__(
        self,
        *,
        gate_ok: bool,
        fail_selected_load: bool = False,
    ) -> None:
        self.log = FakeLogger()
        self.cfg = SimpleNamespace(workflow_plugin_probe_enabled=False)
        self.pipeline_policy = {
            "policy_name": "test",
            "stage3_background": {},
        }
        self.target_profile: dict[str, Any] = {}
        self.current_image = "baseline"
        self.fail_selected_load = fail_selected_load
        self.gate_ok = gate_ok
        self.saved_sources: dict[str, str] = {}
        self.cmd_calls: list[tuple[Any, ...]] = []
        self.workflow_command_used: dict[str, str] = {}
        self.results: list[tuple[str, str, float, str]] = []
        self.result_metadata: list[dict[str, Any]] = []
        self.report: dict[str, Any] = {}
        self.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: None)

    def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
        _ = quiet
        self.cmd_calls.append(args)
        if args[0] == "save":
            self.saved_sources[str(args[1])] = self.current_image
            return True
        if args[0] == "load":
            stem = str(args[1])
            if self.fail_selected_load and stem.startswith("stage3_candidate_"):
                raise pipeline_module.CommandError("mock selected candidate load failure")
            self.current_image = self.saved_sources[stem]
            return True
        self.current_image = f"candidate:{args[0]}"
        return True

    def _stage3_subsky_rbf_candidates(self):
        return []

    def _stage3_measure_features(self, _label: str):
        return None

    def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
        return {"available": False}

    def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
        return self.gate_ok, "accepted" if self.gate_ok else "mock rejection"

    def _adaptive_features_current(self):
        return {
            "bg_std": 0.0001,
            "gradient_score": 0.10,
            "dirty_background_score": 0.20,
            "chroma_noise_score": 0.03,
            "red_dominance": 1.0,
            "blue_dominance": 1.0,
            "green_cast": 1.0,
        }

    def _save_stage_output(self, stem: str) -> bool:
        self.saved_sources[stem] = self.current_image
        return True

    def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
        self.report = payload

    def _record_stage(
        self,
        name: str,
        status: str,
        elapsed: float,
        message: str,
        **metadata: Any,
    ) -> None:
        self.results.append((name, status, elapsed, message))
        self.result_metadata.append(dict(metadata))


class FakeProcessor:
    def __init__(self, module: Any, work_dir: Path) -> None:
        self.module = module
        self.log = FakeLogger()
        self.work_dir = work_dir
        self.process_dir = work_dir / "process"
        self.process_dir.mkdir(exist_ok=True)
        catalog_root = work_dir / "catalogs"
        self.local_gaia_photo_catalog = (
            catalog_root / "siril_cat1_healpix8_xpsamp"
        )
        self.local_gaia_photo_catalog.mkdir(parents=True, exist_ok=True)
        (
            self.local_gaia_photo_catalog
            / "siril_cat1_healpix8_xpsamp_14.dat"
        ).write_bytes(b"x" * 2048)
        self.local_gaia_astro_catalog = (
            catalog_root / "siril_cat_healpix8_astro.dat"
        )
        self.local_gaia_astro_catalog.write_bytes(b"x" * 2048)
        self.spcc_database_dir = REPO_ROOT / "resources" / "siril_spcc_database"

        self.cfg = SimpleNamespace(
            denoise_enabled=True,
            denoise_mod=0.35,
            denoise_safety_max=0.55,
            asinh_stretch=3.0,
            asinh_offset=0.001,
            ghs_shadowsclip=-2.8,
            ghs_stretchamount=2.0,
            nebula_saturation=0.4,
            nebula_bg_factor=1,
            stage8_bg_std_growth_max=1.08,
            remix_nebula_weight=0.18,
            star_intensity=1.0,
            star_fallback_intensity=0.95,
            remix_safe_blend=True,
            optional_color_transform_enabled=False,
            final_saturation=0.15,
            final_bg_factor=1,
            debug_mode=False,
            workflow_plugin_probe_enabled=True,
            aberration_api_enabled=False,
            spcc_enabled=True,
            stage4_platesolve_enabled=True,
            stage4_spcc_restore_cpu=8,
            stage4_pcc_header_fallback_enabled=True,
            stage4_local_star_wb_enabled=True,
            stage4_local_star_wb_min_pixels=32,
            stage4_local_star_wb_gain_limit=1.25,
            stage4_local_star_wb_target_aware_enabled=False,
        )
        self.auto_tune_result = None
        self.stretched_name = "stage6_stretched"
        self.platesolve_ok = False
        self.image_shape = (3, 1000, 1000)
        self.siril = SimpleNamespace(get_image_shape=lambda: self.image_shape)

        self.export_linear_ok = True
        self.fail_commands: set[str] = set()
        self.command_labels: dict[str, str | None] = {}
        self.available_commands: set[str] = set()
        self.script_labels: dict[str, str | None] = {}
        self.available_scripts: set[str] = set()
        self.script_fail_steps: set[str] = set()
        self.cli_fail_steps: set[str] = set()
        self.cli_failure_errors: dict[str, str] = {}
        self.classic_cc_args: tuple[str, ...] | None = ("-executable", "/mock/cc")
        self.script_calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.syqon_output_mode: str = "none"
        self._last_plugin_script_error: str | None = None
        self._last_scunet_fallback_error: str | None = None
        self.aberration_labels: dict[str, str | None] = {}
        self.aberration_errors: dict[str, str] = {}
        self._last_aberration_api_error: str | None = None
        self.local_aberration_model: Path | None = None
        self.ccm_fallback_ok = True
        self._stage1_input_mode = "stacked"
        self._channel_semantics = "broadband_rgb_osc"
        self.ccm_fallback_message = (
            "使用 CCM 回退完成色彩校准 (r_gain=1.010, b_gain=0.990, sample_pixels=2048)"
        )
        self.main_output_basename_template = pipeline_module.RESULT_BASENAME_TEMPLATE
        self.feature_measurements: list[Any] = []
        self.adaptive_measurements: list[dict[str, Any]] = []

        self.cmd_calls: list[tuple[Any, ...]] = []
        self.command_chain_calls: list[str] = []
        self.aberration_calls: list[str] = []
        self.checkpoints: list[str] = []
        self.results: list[tuple[str, str, float, str]] = []
        self.result_metadata: list[dict[str, Any]] = []
        self.workflow_command_used: dict[str, str] = {}
        self.starmask_file: Path | None = None
        self.starless_file: Path | None = None
        self.previous_stage_remix_calls: list[tuple[str, str, float]] = []
        self.fail_previous_stage_remix = False
        self.sasp_stage8_label: str | None = None
        self.sasp_stage8_calls: list[dict[str, Any] | None] = []
        self.stage_json_reports: dict[str, dict[str, Any]] = {}
        self.header_metadata: dict[str, Any] = {}

    def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
        _ = quiet
        self.cmd_calls.append(args)
        cmd = str(args[0]) if args else ""
        if cmd in self.fail_commands:
            raise self.module.CommandError(f"mock failure: {cmd}")
        if cmd in {"siril_scunet_denoise", "scunet_denoise", "siril_scunet", "scunet"}:
            if cmd not in self.available_commands:
                raise self.module.CommandError(f"Command '{cmd}' failed: Command not found")
        return True

    def _run_first_available_command(
        self,
        step_key: str,
        candidates: list[tuple[str, tuple[Any, ...]]],
        allow_when_probe_disabled: bool = False,
    ):
        _ = candidates
        if not self.cfg.workflow_plugin_probe_enabled and not allow_when_probe_disabled:
            return None
        self.command_chain_calls.append(step_key)
        label = self.command_labels.get(step_key)
        if label:
            self.workflow_command_used[step_key] = label
        return label

    def _find_plugin_script(self, relative_candidates: tuple[str, ...]):
        for rel in relative_candidates:
            if rel not in self.available_scripts:
                continue
            script_path = self.work_dir / "mock_scripts" / rel
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("# mock script\n", encoding="utf-8")
            return script_path
        return None

    def _run_plugin_script_by_path(
        self,
        step_key: str,
        label: str,
        script_path: Path,
        *,
        args: tuple[str, ...] = (),
    ):
        self.script_calls.append((step_key, script_path.name, args))
        if step_key in self.script_fail_steps:
            self._last_plugin_script_error = f"{script_path.name}: mock script failure"
            return None
        self._last_plugin_script_error = None
        if step_key == "去星":
            if self.syqon_output_mode in {"starless", "both"}:
                (self.process_dir / f"starless_{self.stretched_name}.fit").write_bytes(b"")
            if self.syqon_output_mode == "both":
                (self.process_dir / f"starmask_{self.stretched_name}.fit").write_bytes(b"")
        used = self.script_labels.get(step_key)
        if used is None:
            used = f"{label} script ({script_path.name})"
        self.workflow_command_used[step_key] = used
        return used

    def _run_plugin_script_cli_subprocess(
        self,
        step_key: str,
        label: str,
        script_path: Path,
        *,
        args: tuple[str, ...] = (),
        timeout_sec: int = 1800,
        **_kwargs: Any,
    ):
        _ = timeout_sec
        if step_key in self.cli_fail_steps:
            self._last_plugin_script_error = self.cli_failure_errors.get(
                step_key,
                f"{script_path.name}: mock cli failure",
            )
            return None
        used = self._run_plugin_script_by_path(
            step_key,
            label,
            script_path,
            args=args,
        )
        if used is None:
            return None
        cli_used = f"{label} cli-subprocess ({script_path.name})"
        self.workflow_command_used[step_key] = cli_used
        return cli_used

    def _classic_cosmic_clarity_args(self, config_name: str, label: str):
        _ = (config_name, label)
        return self.classic_cc_args

    def _classic_cosmic_clarity_device_args(self):
        return pipeline_module.SeestarPostProcessor._classic_cosmic_clarity_device_args(self)

    def _is_classic_cc_not_configured(self, reason: str):
        return pipeline_module.SeestarPostProcessor._is_classic_cc_not_configured(self, reason)

    def _run_siril_cc_sharpen_fallback(self, step_key: str):
        return pipeline_module.SeestarPostProcessor._run_siril_cc_sharpen_fallback(
            self,
            step_key,
        )

    def _run_cosmic_clarity_native_sharpen_fallback(self, step_key: str):
        return pipeline_module.SeestarPostProcessor._run_cosmic_clarity_native_sharpen_fallback(
            self,
            step_key,
        )

    def _cosmic_clarity_native_sharpen_cli_options(self):
        return pipeline_module.SeestarPostProcessor._cosmic_clarity_native_sharpen_cli_options(self)

    def _run_siril_scunet_denoise_fallback(self, step_key: str, strength: float):
        return pipeline_module.SeestarPostProcessor._run_siril_scunet_denoise_fallback(
            self,
            step_key,
            strength,
        )

    def _run_cosmic_clarity_native_denoise_fallback(self, step_key: str):
        return pipeline_module.SeestarPostProcessor._run_cosmic_clarity_native_denoise_fallback(
            self,
            step_key,
        )

    def _cosmic_clarity_native_denoise_cli_options(self):
        return pipeline_module.SeestarPostProcessor._cosmic_clarity_native_denoise_cli_options(self)

    def _syqon_starless_cli_options(
        self,
        *,
        tile_size: int = 512,
        overlap: int = 64,
        axiom: bool = False,
    ):
        return pipeline_module.syqon_starless.syqon_starless_cli_options(
            self,
            tile_size=tile_size,
            overlap=overlap,
            axiom=axiom,
        )

    def _final_denoise_cli_timeout_sec(self) -> int:
        return pipeline_module.SeestarPostProcessor._final_denoise_cli_timeout_sec(self)

    def _collect_star_separation_outputs(self):
        return pipeline_module.SeestarPostProcessor._collect_star_separation_outputs(self)

    def _run_aberration_api(self, step_key: str, model_path=None):
        _ = model_path
        self.aberration_calls.append(step_key)
        label = self.aberration_labels.get(step_key)
        if label:
            self._last_aberration_api_error = None
            return label
        self._last_aberration_api_error = self.aberration_errors.get(step_key)
        return None

    def _resolve_local_aberration_model(self):
        return self.local_aberration_model

    def _run_sasp_stage8_api(self, plan=None):
        self.sasp_stage8_calls.append(plan)
        if self.sasp_stage8_label:
            self.workflow_command_used["SASP Starless 深加工 API"] = self.sasp_stage8_label
            self._last_sasp_stage8_error = None
            return self.sasp_stage8_label
        self._last_sasp_stage8_error = "mock SASP stage8 API unavailable"
        return None

    def _export_linear_intermediate(self) -> bool:
        return self.export_linear_ok

    def _result_output_basename(self) -> str:
        return pipeline_module.SeestarPostProcessor._result_output_basename(self)

    def _run_ccm_color_fallback(self) -> tuple[bool, str]:
        if self.ccm_fallback_ok:
            return True, self.ccm_fallback_message
        return False, "mock ccm fallback failed"

    def _checkpoint_save(self, name: str, critical: bool = False) -> None:
        _ = critical
        self.checkpoints.append(name)

    def _save_stage_output(self, _stem: str) -> bool:
        return True

    def _read_fits_header_metadata(self, *_candidates: str):
        metadata = {
            "CRVAL1": 303.051891667,
            "CRVAL2": 38.331575278,
        }
        metadata.update(self.header_metadata)
        return metadata

    def _auto_target_hint(self):
        return None

    def _refresh_target_profile_from_metadata(self, _metadata: dict[str, Any], *, stage_label: str = ""):
        _ = stage_label
        return ""

    def _active_policy_name(self):
        return "generic_low_snr_safe"

    def _stage_diff_note(self, _current_stem: str, _previous_stem: str):
        return None

    def _fallback_summary(
        self,
        failed_component: str,
        failure_reason: str,
        fallback_component: str,
        fallback_succeeded: bool,
    ) -> str:
        return pipeline_module.SeestarPostProcessor._fallback_summary(
            self,
            failed_component,
            failure_reason,
            fallback_component,
            fallback_succeeded,
        )

    def _is_siril_connection_failure(self, value: object) -> bool:
        return pipeline_module.SeestarPostProcessor._is_siril_connection_failure(
            self,
            value,
        )

    def _build_manual_starmask(self) -> bool:
        return True

    def _export_sasp_exchange_files(self) -> None:
        return None

    def _find_external_fit(self, _candidate_names: list[str]):
        return None

    def _record_stage(
        self,
        name: str,
        status: str,
        duration: float = 0.0,
        message: str = "",
        **metadata: Any,
    ) -> None:
        self.results.append((name, status, duration, message))
        self.result_metadata.append(dict(metadata))

    def _short_text(self, value: Any, max_len: int = 240) -> str:
        text = str(value).strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _measure_current_features(self):
        if self.feature_measurements:
            return self.feature_measurements.pop(0)
        return None

    def _adaptive_features_current(self):
        if self.adaptive_measurements:
            return self.adaptive_measurements.pop(0)
        return {}

    def _feature_summary_note(self, label: str):
        return pipeline_module.SeestarPostProcessor._feature_summary_note(self, label)

    def _apply_adaptive_edge_crop(self, feat):
        return pipeline_module.SeestarPostProcessor._apply_adaptive_edge_crop(self, feat)

    def _apply_weak_object_tuning(self):
        return pipeline_module.SeestarPostProcessor._apply_weak_object_tuning(self)

    def _apply_starless_blue_guard(self, feat):
        return pipeline_module.SeestarPostProcessor._apply_starless_blue_guard(self, feat)

    def _ai_stage_advisory_enabled(self, _attr_name: str) -> bool:
        return False

    def _request_stage8_processing_plan(self):
        return None

    def _apply_stage8_builtin_enhancement(self, plan: dict[str, Any], *, label: str):
        return pipeline_module.SeestarPostProcessor._apply_stage8_builtin_enhancement(
            self,
            plan,
            label=label,
        )

    def _stage8_quality_assessment(self):
        return {"status": "ok", "issues": []}

    def _write_stage_json(self, name: str, payload: dict[str, Any]) -> None:
        self.stage_json_reports[name] = payload

    def _apply_previous_stage_star_remix(self, source_stem: str, starmask_name: str, intensity: float):
        self.previous_stage_remix_calls.append((source_stem, starmask_name, intensity))
        return not self.fail_previous_stage_remix

    def _stage9_bad_starless_reason(self) -> str:
        return ""

    def _stage9_review_safe_source(self) -> str:
        return "stage6_stretched"

    def _stage8_soften_mask(self, mask, passes: int = 3):
        return pipeline_module.SeestarPostProcessor._stage8_soften_mask(
            self,
            mask,
            passes=passes,
        )


class PipelinePluginFallbackTests(unittest.TestCase):
    def _new_processor(self) -> FakeProcessor:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return FakeProcessor(pipeline_module, Path(td.name))

    @staticmethod
    def _stage10_final_input(processor: FakeProcessor) -> None:
        (processor.process_dir / "stage9_remixed.fit").write_bytes(b"mock")

    def _copy_spcc_database(self, processor: FakeProcessor) -> Path:
        target = processor.work_dir / "siril-spcc-database"
        shutil.copytree(processor.spcc_database_dir, target)
        processor.spcc_database_dir = target
        return target

    def test_stage_review_bundle_skips_visual_request_when_ai_is_disabled(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.ai_post_enabled = False
        processor.cfg.ai_advisor_mode = "multimodal"
        processor.cfg.ai_endpoint = "https://example.invalid/v1/chat/completions"
        processor.cfg.ai_model = "vision-model"
        processor.cfg.ai_api_key = "configured-but-disabled"
        payload = {
            "status": "ready",
            "stage": "stage3_background_extraction",
            "visual_review": {
                "status": "not_requested",
                "advisor_mode": "not_requested",
            },
            "candidates": [
                {
                    "selection_status": "selected",
                    "visual_acceptance_status": "not_requested",
                }
            ],
        }

        with patch.object(
            stage_support_module.review_bundle,
            "create_stage_review_bundle",
            return_value=payload,
        ), patch.object(
            stage_support_module.ai_advisory,
            "request_visual_acceptance",
            return_value=None,
        ) as request_visual_acceptance:
            result = processor._create_stage_review_bundle(
                "stage3_background_extraction",
                "before",
                "after",
            )

        request_visual_acceptance.assert_not_called()
        self.assertEqual(result["visual_review"]["status"], "not_requested")
        self.assertEqual(result["visual_review"]["advisor_mode"], "not_requested")

    def test_stage11_disabled_skips_before_optional_module_import_failure(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.ai_post_enabled = False

        with patch.object(
            pipeline_module,
            "run_stage11_ai_postprocess",
            None,
        ), patch.object(
            pipeline_module,
            "STAGE11_IMPORT_ERROR",
            RuntimeError("mock optional import failure"),
        ):
            processor.stage11_ai_postprocess()

        result = processor.results[-1]
        self.assertEqual(result.status, "skipped")
        self.assertIn("SEESTAR_AI_ENABLED not enabled", result.message)
        self.assertFalse(processor.ai_outputs_generated)

    def test_stage_preview_publishes_only_accepted_artifact_pixels(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        process_dir = work_dir / "process"
        process_dir.mkdir()
        (process_dir / "stage1_prepared.fit").write_bytes(b"accepted")
        processor = pipeline_module.SeestarPostProcessor()
        processor.work_dir = work_dir
        processor.process_dir = process_dir
        processor.log = FakeLogger()
        processor.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: np.array(
                [[0.0, 0.25], [0.5, 0.75]],
                dtype=np.float32,
            )
        )
        commands = []
        processor.cmd_with_check = lambda *args: commands.append(args)

        processor._publish_stage_preview(1, "前期准备", "ok")

        preview_path = process_dir / "ui_preview" / "latest.png"
        self.assertTrue(preview_path.is_file())
        self.assertIn(("load", "stage1_prepared"), commands)
        self.assertTrue(
            any(
                "[PIPELINE_PREVIEW]" in message and '"status":"ready"' in message
                for level, message in processor.log.events
                if level == "info"
            )
        )

    def test_failed_or_skipped_stage_does_not_publish_preview(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.process_dir = Path("/tmp/unused-preview-test")
        processor.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: (_ for _ in ()).throw(
                AssertionError("failed/skipped stage must not decode")
            )
        )

        processor._publish_stage_preview(3, "背景提取", "failed")
        processor._publish_stage_preview(4, "校色", "skipped")

        self.assertFalse(processor.log.events)

    def test_unexpected_preview_error_cannot_change_stage_result(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.results = []
        processor.log = FakeLogger()
        processor._publish_stage_preview = lambda *_args: (_ for _ in ()).throw(
            AttributeError("mock preview failure")
        )

        processor._record_stage("阶段 2: 裁切", "ok", 0.2, "accepted")

        self.assertEqual(processor.results[-1].status, "ok")
        self.assertTrue(
            any(
                "预览观察链路异常" in message
                for level, message in processor.log.events
                if level == "warn"
            )
        )

    def test_record_stage_emits_gui_state_from_structured_fields_only(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.results = []
        processor.log = FakeLogger()
        processor._publish_stage_preview = lambda *_args: None

        processor._record_stage(
            "阶段 9: 星点处理与合成",
            "ok",
            1.2,
            "fallback_used=true; controlled Screen remix",
        )
        processor._record_stage(
            "阶段 10: 最终降噪与导出",
            "ok",
            0.5,
            "final denoise skipped because input is already low-noise",
        )
        processor._record_stage(
            "阶段 6: 去星与 Halo 修复",
            "ok",
            0.8,
            "primary tool failed; alternate accepted",
            fallback_used=True,
            reason_code="alternate_star_separation",
        )
        processor._record_stage(
            "阶段 8: Starless 深加工",
            "ok",
            0.4,
            "guard retained the accepted Stage 7 source",
            execution="safe_passthrough",
            reason_code="bright_nebula_halo_advisory",
        )

        events = [
            message
            for level, message in processor.log.events
            if level == "info" and "[PIPELINE_STAGE_RESULT]" in message
        ]
        detail_events = [
            json.loads(message.split("[PIPELINE_STAGE_DETAIL] ", 1)[1])
            for level, message in processor.log.events
            if level == "info" and "[PIPELINE_STAGE_DETAIL]" in message
        ]
        self.assertIn("stage=9 status=ok", events[0])
        self.assertIn("stage=10 status=ok", events[1])
        self.assertIn("stage=6 status=degraded", events[2])
        self.assertIn("stage=8 status=ok", events[3])
        self.assertEqual(detail_events[0]["display_status"], "ok")
        self.assertEqual(detail_events[1]["display_status"], "ok")
        self.assertTrue(detail_events[2]["fallback_used"])
        self.assertEqual(detail_events[2]["display_status"], "ok_with_fallback")
        self.assertEqual(detail_events[3]["execution"], "safe_passthrough")
        self.assertEqual(
            detail_events[3]["display_status"],
            "ok_safe_passthrough",
        )
        self.assertEqual(processor.results[0].status, "ok")
        self.assertEqual(processor.results[1].status, "ok")

    def test_stage_output_aliases_follow_display_stage_names(self):
        calls: list[tuple[str, str]] = []
        log = FakeLogger()

        saved = pipeline_module.save_stage_output(
            lambda *args: calls.append(tuple(str(item) for item in args)),
            log,
            "stage7_starless",
        )

        self.assertTrue(saved)
        self.assertIn(("save", "stage7_starless"), calls)
        self.assertIn(("save", "stage6_starless"), calls)

        calls.clear()
        saved = pipeline_module.save_stage_output(
            lambda *args: calls.append(tuple(str(item) for item in args)),
            log,
            "stage7_stretched",
        )

        self.assertTrue(saved)
        self.assertIn(("save", "stage7_stretched"), calls)
        self.assertNotIn(("save", "stage6_stretched"), calls)

    def test_stage_json_aliases_follow_display_stage_names(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name)
        log = FakeLogger()

        pipeline_module.write_stage_json(
            process_dir,
            log,
            "stage7_stretch_quality.json",
            {"stage": "stage7_stretch"},
        )

        self.assertTrue((process_dir / "stage7_stretch_quality.json").exists())

        pipeline_module.write_stage_json(
            process_dir,
            log,
            "stage6_starless_quality.json",
            {"stage": "stage6_starless"},
        )

        self.assertTrue((process_dir / "stage6_starless_quality.json").exists())
        self.assertTrue((process_dir / "stage7_quality.json").exists())

    def test_legacy_stage_method_aliases_warn_before_dispatch(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        calls: list[str] = []
        processor.stage7_stretching = lambda: calls.append("stage7_stretching")
        processor.stage6_star_separation = lambda: calls.append("stage6_star_separation")

        processor.stage6_stretching()
        processor.stage7_star_separation()

        self.assertEqual(calls, ["stage7_stretching", "stage6_star_separation"])
        warnings = [
            message for level, message in processor.log.events if level == "warn"
        ]
        self.assertTrue(any("stage6_stretching() is a legacy alias" in message for message in warnings))
        self.assertTrue(
            any("stage7_star_separation() is a legacy alias" in message for message in warnings)
        )

    def test_debug_stage_save_writes_quality_metrics(self):
        import json

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name)
        processor = pipeline_module.SeestarPostProcessor()
        processor.cfg.debug_mode = True
        processor.log = FakeLogger()
        processor.process_dir = process_dir
        processor.siril = SimpleNamespace(
            cmd=lambda *_args: None,
            get_image_pixeldata=lambda preview=False: object(),
        )

        metric_globals = processor._write_debug_quality_metrics.__globals__
        with patch.dict(
            metric_globals,
            {
                "measure_quality_metrics": lambda _image: pipeline_module.QualityMetrics(
                    bg_median=0.123,
                ),
                "measure_image_features": lambda _image: pipeline_module.ImageFeatures(
                    edge_black_ratio=0.045,
                ),
            },
        ):
            self.assertTrue(processor._save_stage_output("stage_debug_probe"))

        metrics_path = process_dir / "stage_debug_probe_quality_metrics.json"
        self.assertTrue(metrics_path.exists())
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "seestar.stage_quality.v1")
        self.assertEqual(payload["stem"], "stage_debug_probe")
        self.assertIn("bg_median", payload["metrics"])
        self.assertIn("edge_black_ratio", payload["features"])

        jsonl_path = process_dir / "stage_quality_metrics.jsonl"
        self.assertTrue(jsonl_path.exists())
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["stem"], "stage_debug_probe")
        self.assertTrue(
            any(
                "[STAGE_QUALITY_METRICS]" in message
                for level, message in processor.log.events
                if level == "info"
            )
        )

    def test_stage5_keeps_builtin_denoise_primary_with_legacy_plugins_available(self):
        processor = self._new_processor()
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["矫正星点"] = "SASP Aberration API (CPU)"
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Denoise.py",
            }
        )
        processor.script_labels["锐化"] = "CosmicClarity Sharpen script (CosmicClarity_Sharpen.py)"
        processor.script_labels["初步降噪"] = "CosmicClarity Denoise script (CosmicClarity_Denoise.py)"

        stage5_linear_denoise(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Siril linear denoise applied", message)
        self.assertNotIn("矫正星点", processor.aberration_calls)
        sharpen_calls = [args for step, _name, args in processor.script_calls if step == "锐化"]
        self.assertFalse(sharpen_calls)

    def test_stage5_does_not_reintroduce_legacy_global_sharpen_for_local_model(self):
        processor = self._new_processor()
        processor.cfg.aberration_api_enabled = False
        processor.local_aberration_model = Path("/tmp/model_v2_0_1.onnx")
        processor.aberration_labels["矫正星点"] = "SASP Aberration API (CPU) [model_v2_0_1.onnx]"
        processor.available_scripts.add("processing/CosmicClarity_Sharpen.py")
        processor.script_fail_steps.add("锐化")
        processor.command_labels["锐化"] = "Unsharp fallback"

        stage5_linear_denoise(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Siril linear denoise applied", message)
        self.assertNotIn("矫正星点", processor.aberration_calls)

    def test_stage5_falls_back_to_internal_denoise_when_scripts_unavailable(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = True
        processor.command_labels["锐化"] = "Unsharp fallback"

        stage5_linear_denoise(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Siril linear denoise applied", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("denoise", cmds)

    def test_stage5_runs_deconvolution_before_linear_denoise(self):
        processor = self._new_processor()

        stage5_linear_denoise(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertLess(cmds.index("findstar"), cmds.index("denoise"))
        self.assertLess(cmds.index("makepsf"), cmds.index("denoise"))
        self.assertLess(cmds.index("rl"), cmds.index("denoise"))
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["final_linear_source"], "stage5_linear")
        self.assertEqual(report["denoise"]["input"], "stage5_deconv")
        self.assertTrue(report["deconvolution"]["runs_before_denoise"])

    def test_stage5_prefers_graxpert_object_deconvolution_when_model_exists(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": "",
            },
            clear=False,
        ):
            os.environ.pop("SEESTAR_GRAXPERT_GPU", None)
            stage5_linear_denoise(processor)

        graxpert_call = next(
            args
            for step, _name, args in processor.script_calls
            if step == "Stage5 GraXpert反卷积"
        )
        self.assertIn("-deconv_obj", graxpert_call)
        self.assertIn("1.0.1", graxpert_call)
        self.assertIn("-gpu", graxpert_call)
        self.assertNotIn("-nogpu", graxpert_call)
        self.assertNotIn("rl", [str(call[0]) for call in processor.cmd_calls])
        self.assertNotIn("denoise", [str(call[0]) for call in processor.cmd_calls])
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "graxpert_object")
        self.assertEqual(report["denoise"]["input"], "stage5_graxpert_deconv")
        self.assertEqual(report["components"]["deconvolution"]["status"], "applied")
        self.assertEqual(report["components"]["denoise"]["status"], "skipped")
        self.assertEqual(
            report["deconvolution"]["graxpert"]["hardware_acceleration"],
            "auto",
        )
        self.assertEqual(
            report["components"]["denoise"]["reason_code"],
            "config_disabled",
        )
        self.assertEqual(
            processor.result_metadata[-1]["components"],
            report["components"],
        )
        self.assertEqual(report["final_linear_source"], "stage5_linear")

    def test_stage5_graxpert_cpu_compatibility_disables_hardware_acceleration(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": "",
                "SEESTAR_GRAXPERT_GPU": "0",
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        graxpert_call = next(
            args
            for step, _name, args in processor.script_calls
            if step == "Stage5 GraXpert反卷积"
        )
        self.assertIn("-nogpu", graxpert_call)
        self.assertNotIn("-gpu", graxpert_call)
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(
            report["deconvolution"]["graxpert"]["hardware_acceleration"],
            "cpu",
        )

    def test_stage5_graxpert_failure_reloads_baseline_then_falls_back_to_rl(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        processor.script_fail_steps.add("Stage5 GraXpert反卷积")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": "",
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        self.assertIn("rl", [str(call[0]) for call in processor.cmd_calls])
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "siril_rl")
        self.assertTrue(report["deconvolution"]["graxpert"]["attempted"])

    def test_stage5_links_user_provided_graxpert_model_into_isolated_home(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        external_model = (
            processor.work_dir
            / "user-models"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        external_model.parent.mkdir(parents=True)
        external_model.write_bytes(b"user supplied onnx")
        isolated_home = processor.work_dir / "isolated-home"

        with patch.dict(
            os.environ,
            {
                "HOME": str(isolated_home),
                "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": str(external_model.parent.parent),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        linked_model = (
            isolated_home
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        self.assertTrue(linked_model.is_symlink())
        self.assertEqual(linked_model.resolve(), external_model.resolve())
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "graxpert_object")
        self.assertEqual(
            report["deconvolution"]["graxpert"]["source"], "user_provided"
        )
        self.assertEqual(
            report["deconvolution"]["graxpert"]["resolved_model_path"],
            str(linked_model),
        )

    def test_stage5_invalid_user_model_path_falls_back_to_rl(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        missing_model = processor.work_dir / "missing-model"

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir / "isolated-home"),
                "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": str(missing_model),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        graxpert = report["deconvolution"]["graxpert"]
        self.assertEqual(report["deconvolution"]["method"], "siril_rl")
        self.assertEqual(graxpert["reason"], "configured_model_not_found_or_invalid")
        self.assertEqual(graxpert["configured_path"], str(missing_model))

    def test_stage5_rejects_user_model_without_semantic_version_directory(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = processor.work_dir / "user-model" / "model.onnx"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"user supplied onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir / "isolated-home"),
                "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH": str(model),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        graxpert = report["deconvolution"]["graxpert"]
        self.assertEqual(report["deconvolution"]["method"], "siril_rl")
        self.assertEqual(
            graxpert["reason"], "model_version_directory_must_be_semver"
        )

    def test_stage5_rl_failure_reloads_input_before_denoise(self):
        processor = self._new_processor()
        processor.fail_commands.add("rl")

        stage5_linear_denoise(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertLess(cmds.index("rl"), cmds.index("denoise"))
        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertFalse(report["deconvolution"]["applied"])
        self.assertEqual(report["denoise"]["input"], "stage5_input_linear")

    def test_stage7_prefers_final_stage5_linear_over_deconv_checkpoint(self):
        processor = self._new_processor()
        (processor.process_dir / "stage5_deconv.fit").write_bytes(b"mock")
        (processor.process_dir / "stage5_graxpert_deconv.fit").write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"mock")
        stage7_module = sys.modules["stages.stage7_star_separation"]

        self.assertEqual(stage7_module._stage7_linear_source(processor), "stage5_linear")

    def test_stage7_uses_graxpert_checkpoint_when_final_stage5_outputs_are_missing(self):
        processor = self._new_processor()
        (processor.process_dir / "stage5_graxpert_deconv.fit").write_bytes(b"mock")
        stage7_module = sys.modules["stages.stage7_star_separation"]

        self.assertEqual(
            stage7_module._stage7_linear_source(processor),
            "stage5_graxpert_deconv",
        )

    def test_stage5_skips_classic_cosmic_clarity_without_executable(self):
        processor = self._new_processor()
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Denoise.py",
            }
        )
        processor.classic_cc_args = None

        stage5_linear_denoise(processor)

        classic_calls = [
            call
            for call in processor.script_calls
            if call[1] in {"CosmicClarity_Sharpen.py", "CosmicClarity_Denoise.py"}
        ]
        self.assertFalse(classic_calls)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("denoise", cmds)

    def test_stage5_prefers_builtin_denoise_when_cosmic_native_is_available(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": False,
                "denoise_mode": "chroma_first",
            },
        }
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Denoise.py",
                "processing/CosmicClarity_Native.py",
            }
        )
        processor.classic_cc_args = None

        stage5_linear_denoise(processor)

        native_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Native.py"
        ]
        self.assertFalse(native_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Siril linear denoise applied", message)

    def test_stage5_skips_global_sharpen_when_background_policy_protects_background(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "dark_nebula_low_contrast",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": True,
                "denoise_mode": "chroma_first",
                "sharpen_mode": "minimal",
            },
        }
        processor.adaptive_measurements.extend(
            [
                {
                    "dirty_background_score": 0.02,
                    "chroma_noise_score": 0.01,
                    "gradient_score": 0.01,
                    "bg_std": 0.0001,
                },
                {
                    "dirty_background_score": 0.02,
                    "chroma_noise_score": 0.01,
                    "gradient_score": 0.01,
                    "bg_std": 0.0001,
                },
            ]
        )
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Native.py",
            }
        )

        stage5_linear_denoise(processor)

        sharpen_calls = [call for call in processor.script_calls if call[0] == "锐化"]
        self.assertFalse(sharpen_calls)
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("Siril linear denoise applied", message)

    def test_stage5_uses_builtin_denoise_for_chroma_first_policy(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": True,
                "denoise_mode": "chroma_first",
            },
        }
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_labels["初步降噪"] = "CosmicClarity Denoise script (CosmicClarity_Denoise.py)"

        stage5_linear_denoise(processor)

        denoise_calls = [args for step, _name, args in processor.script_calls if step == "初步降噪"]
        self.assertFalse(denoise_calls)
        self.assertIn(("denoise", "-mod=0.50", "-indep"), processor.cmd_calls)

    def test_stage5_rolls_back_when_background_chroma_gets_worse(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "generic_low_snr_safe",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": False,
            },
        }
        processor.cfg.denoise_enabled = True
        processor.cfg.denoise_mod = 0.35
        processor.adaptive_measurements.extend(
            [
                {
                    "dirty_background_score": 0.25,
                    "chroma_noise_score": 0.08,
                    "bg_std": 0.00010,
                },
                {
                    "dirty_background_score": 0.34,
                    "chroma_noise_score": 0.12,
                    "bg_std": 0.00013,
                },
            ]
        )

        stage5_linear_denoise(processor)

        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        self.assertIn(("denoise", "-mod=0.50", "-indep"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 background guard dropped siril_rl result", message)

    def test_stage5_rolls_back_when_chroma_becomes_more_visible_than_luma_noise(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": False,
                "denoise_mode": "chroma_first",
            },
        }
        processor.cfg.denoise_enabled = True
        processor.cfg.denoise_mod = 0.35
        processor.adaptive_measurements.extend(
            [
                {
                    "dirty_background_score": 0.0051,
                    "chroma_noise_score": 0.01346,
                    "bg_std": 0.000055,
                },
                {
                    "dirty_background_score": 0.0048,
                    "chroma_noise_score": 0.01388,
                    "bg_std": 0.000033,
                },
            ]
        )

        stage5_linear_denoise(processor)

        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        self.assertIn(("denoise", "-mod=0.50", "-indep"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 background guard dropped siril_rl result", message)
        self.assertIn("chroma_bg_ratio_growth", message)

    def test_stage4_spcc_success_stays_ok_with_platesolve_by_default(self):
        processor = self._new_processor()

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn(
            "platesolve -noflip -focal=160 -pixelsize=2.90 "
            "-catalog=gaia -order=3 ok",
            message,
        )
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("platesolve", cmds)
        platesolve_calls = [call for call in processor.cmd_calls if call[0] == "platesolve"]
        self.assertEqual(
            platesolve_calls[0],
            (
                "platesolve",
                "-noflip",
                "-focal=160",
                "-pixelsize=2.90",
                "-catalog=gaia",
                "-order=3",
            ),
        )
        self.assertIn("spcc", cmds)
        self.assertIn(("spcc_list", "whiteref"), processor.cmd_calls)
        spcc_call = next(call for call in processor.cmd_calls if call[0] == "spcc")
        self.assertIn("-catalog=localgaia", spcc_call)
        self.assertEqual(processor.color_calibration_report["spcc"]["catalog"], "localgaia")
        metadata_report = processor.color_calibration_report["spcc"]["metadata_database"]
        self.assertTrue(metadata_report["available"])
        self.assertEqual(metadata_report["reason"], "ok")
        self.assertTrue(all(item["found"] for item in metadata_report["requirements"]))
        self.assertNotIn("cc", cmds)
        self.assertFalse({"mirrorx", "mirrory", "flip", "rotate"} & set(cmds))

    def test_stage4_imprecise_spcc_log_restores_psolved_and_uses_pcc(self):
        processor = self._new_processor()
        before = "existing Siril log\n"
        warning = (
            "The photometric color calibration seems to have found an imprecise "
            "solution, consider correcting the image gradient first\n"
        )
        snapshots = iter((before, before + warning))
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: processor.image_shape,
            get_siril_log=lambda: next(snapshots),
        )

        stage4_color_calibration(processor)

        spcc_index = next(
            index for index, call in enumerate(processor.cmd_calls) if call[0] == "spcc"
        )
        restore_index = processor.cmd_calls.index(("load", "stage4_psolved"), spcc_index)
        pcc_index = next(
            index for index, call in enumerate(processor.cmd_calls) if call[0] == "pcc"
        )
        self.assertLess(spcc_index, restore_index)
        self.assertLess(restore_index, pcc_index)
        report = processor.color_calibration_report
        self.assertEqual(report["method"], "PCC")
        self.assertEqual(report["warning"], "spcc_imprecise_solution_pcc_fallback")
        self.assertEqual(report["color_confidence"], 0.62)
        self.assertEqual(report["status"], "success_with_warning")
        solution = report["spcc"]["solution_quality"]
        self.assertEqual(solution["status"], "imprecise")
        self.assertTrue(solution["imprecise"])
        self.assertTrue(solution["fallback_triggered"])
        self.assertTrue(solution["fallback_source_restored"])
        self.assertEqual(solution["confidence"], 0.45)
        self.assertIn("imprecise solution", solution["matched_messages"][0])
        self.assertEqual(
            report["pcc"]["attempts"][0]["phase"],
            "spcc_imprecise_recovery",
        )

    def test_stage4_spcc_imprecise_detector_recognizes_chinese_log_delta(self):
        stage4_module = sys.modules["stages.stage4_color_calibration"]
        before = "旧日志\n"
        after = before + "测光法色彩校准似乎不能精确校准，考虑先修正图像渐变\n"

        report = stage4_module._stage4_spcc_solution_quality(before, after)

        self.assertEqual(report["status"], "imprecise")
        self.assertTrue(report["imprecise"])
        self.assertEqual(report["warning_code"], "spcc_imprecise_solution")
        self.assertEqual(report["confidence"], 0.45)

    def test_stage4_spcc_imprecise_detector_ignores_warning_from_old_log(self):
        stage4_module = sys.modules["stages.stage4_color_calibration"]
        warning = (
            "The photometric color calibration seems to have found an imprecise "
            "solution, consider correcting the image gradient first\n"
        )
        before = warning + "old command finished\n"
        after = before + "Spectrophotometric Color Calibration succeeded.\n"

        report = stage4_module._stage4_spcc_solution_quality(before, after)

        self.assertEqual(report["status"], "accepted")
        self.assertFalse(report["imprecise"])
        self.assertFalse(report["matched_messages"])

    def test_stage4_imprecise_spcc_restore_failure_keeps_degraded_result(self):
        processor = self._new_processor()
        before = "existing Siril log\n"
        warning = (
            "The photometric color calibration seems to have found an imprecise "
            "solution, consider correcting the image gradient first\n"
        )
        snapshots = iter((before, before + warning))
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: processor.image_shape,
            get_siril_log=lambda: next(snapshots),
        )
        original_cmd = processor.cmd_with_check

        def fail_recovery_load(*args: Any, quiet: bool = False):
            if args == ("load", "stage4_psolved"):
                processor.cmd_calls.append(args)
                raise processor.module.CommandError("mock recovery load failure")
            return original_cmd(*args, quiet=quiet)

        processor.cmd_with_check = fail_recovery_load

        stage4_color_calibration(processor)

        report = processor.color_calibration_report
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(report["method"], "SPCC_IMPRECISE")
        self.assertEqual(report["color_confidence"], 0.45)
        self.assertEqual(report["status"], "success_with_warning")
        self.assertEqual(
            report["warning"],
            "spcc_imprecise_solution_restore_failed",
        )
        self.assertFalse(
            report["spcc"]["solution_quality"]["fallback_source_restored"]
        )
        self.assertFalse(any(call[0] == "pcc" for call in processor.cmd_calls))

    def test_stage4_all_platesolve_variants_disable_orientation_flip(self):
        stage4_module = sys.modules["stages.stage4_color_calibration"]
        processor = self._new_processor()

        variants = stage4_module._stage4_platesolve_variants(
            processor,
            {"RA": 10.0, "DEC": 20.0},
        )

        self.assertTrue(variants)
        for _label, args in variants:
            self.assertIn("-noflip", args)

    def test_cmd_with_check_treats_closed_connection_as_fatal_without_retry(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        calls: list[tuple[Any, ...]] = []

        def dead_cmd(*args: Any) -> None:
            calls.append(args)
            raise pipeline_module.SirilConnectionError("connection closed by Siril")

        processor.siril = SimpleNamespace(cmd=dead_cmd)

        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.cmd_with_check("spcc")

        self.assertTrue(processor._siril_process_terminated)
        self.assertEqual(calls, [("spcc",)])
        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.cmd_with_check("pcc")
        self.assertEqual(calls, [("spcc",)])

    def test_direct_siril_api_connection_death_uses_same_fatal_fuse(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor._siril_ever_connected = True
        calls: list[str] = []

        def dead_pixel_read(*_args: Any, **_kwargs: Any):
            calls.append("get_image_pixeldata")
            raise pipeline_module.SirilConnectionError("broken pipe")

        processor.siril = pipeline_module._FatalSirilInterfaceProxy(
            processor,
            SimpleNamespace(get_image_pixeldata=dead_pixel_read),
        )

        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.siril.get_image_pixeldata(preview=False)
        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.siril.get_image_pixeldata(preview=False)

        self.assertEqual(calls, ["get_image_pixeldata"])

    def test_stage4_native_death_skips_cpu_restore_pcc_save_and_completion(self):
        processor = self._new_processor()
        original_cmd_with_check = processor.cmd_with_check
        saved_stems: list[str] = []

        def fatal_spcc(*args: Any, quiet: bool = False):
            if args and args[0] == "spcc":
                processor.cmd_calls.append(args)
                processor._siril_process_terminated = True
                raise pipeline_module.SirilNativeProcessTerminated(
                    "spcc",
                    pipeline_module.SirilConnectionError("connection closed"),
                )
            return original_cmd_with_check(*args, quiet=quiet)

        processor.cmd_with_check = fatal_spcc
        processor._save_stage_output = lambda stem: saved_stems.append(stem) or True

        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertEqual(
            [call for call in processor.cmd_calls if call[0] == "setcpu"],
            [("setcpu", "1")],
        )
        self.assertNotIn("pcc", cmds)
        self.assertIn("stage4_psolved", saved_stems)
        self.assertNotIn("stage4_color", saved_stems)
        self.assertFalse(processor.results)
        self.assertFalse(
            any(level == "stage_end" for level, _message in processor.log.events)
        )

    def test_stage4_skips_all_spcc_commands_when_sensor_is_not_in_database(self):
        processor = self._new_processor()
        processor.cfg.stage4_spcc_osc_sensor = "Unsupported OSC sensor"

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc_list", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["metadata_database"]
        self.assertEqual(report["reason"], "required_metadata_missing")
        self.assertIn("osc_sensor=Unsupported OSC sensor", report["missing"])

    def test_stage4_skips_all_spcc_commands_when_filter_is_not_in_database(self):
        processor = self._new_processor()
        processor.cfg.stage4_spcc_osc_filter = "Unsupported OSC filter"

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc_list", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["metadata_database"]
        self.assertIn("osc_filter=Unsupported OSC filter", report["missing"])

    def test_stage4_skips_all_spcc_commands_when_selected_json_is_invalid(self):
        processor = self._new_processor()
        database = self._copy_spcc_database(processor)
        (database / "osc_sensors" / "Sony_IMX585.json").write_text(
            "{invalid-json",
            encoding="utf-8",
        )

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc_list", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["metadata_database"]
        self.assertEqual(report["reason"], "invalid_metadata_json")
        self.assertTrue(report["invalid_json_files"])

    def test_stage4_skips_all_spcc_commands_when_selected_json_array_is_empty(self):
        processor = self._new_processor()
        database = self._copy_spcc_database(processor)
        (database / "osc_filters" / "ZWO_Seestar_LP.json").write_text(
            "[]\n",
            encoding="utf-8",
        )

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc_list", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["metadata_database"]
        self.assertEqual(report["reason"], "empty_metadata_json_array")
        self.assertIn("osc_filters/ZWO_Seestar_LP.json", report["empty_json_files"])

    def test_stage4_skips_all_spcc_commands_when_response_arrays_are_empty(self):
        processor = self._new_processor()
        database = self._copy_spcc_database(processor)
        (database / "osc_filters" / "ZWO_Seestar_LP.json").write_text(
            '[{"name":"ZWO Seestar LP","type":"OSC_FILTER",'
            '"wavelength":{"value":[]},"values":{"value":[]}}]\n',
            encoding="utf-8",
        )

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc_list", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["metadata_database"]
        self.assertEqual(report["reason"], "invalid_metadata_response_arrays")
        self.assertTrue(report["invalid_entry_files"])

    def test_stage4_skips_spcc_before_siril_when_catalog_is_empty(self):
        processor = self._new_processor()
        for path in processor.local_gaia_photo_catalog.glob("*.dat"):
            path.unlink()

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["local_catalog"]
        self.assertFalse(report["available"])
        self.assertEqual(report["reason"], "no_valid_catalog_chunks")
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("SPCC skipped before Siril call", message)

    def test_stage4_skips_spcc_when_target_healpix_is_not_installed(self):
        processor = self._new_processor()
        chunk14 = (
            processor.local_gaia_photo_catalog
            / "siril_cat1_healpix8_xpsamp_14.dat"
        )
        chunk14.rename(
            processor.local_gaia_photo_catalog
            / "siril_cat1_healpix8_xpsamp_13.dat"
        )

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["local_catalog"]
        self.assertEqual(report["reason"], "target_healpix_not_installed")
        self.assertIn(14, report["required_pixels"])
        self.assertIn(14, report["missing_pixels"])

    def test_stage4_pcc_skips_missing_localgaia_and_uses_explicit_online_catalog(self):
        processor = self._new_processor()
        processor.local_gaia_astro_catalog.unlink()
        processor.fail_commands.add("spcc")

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}):
            stage4_color_calibration(processor)

        pcc_calls = [call for call in processor.cmd_calls if call[0] == "pcc"]
        self.assertEqual(pcc_calls, [("pcc", "-catalog=gaia")])
        local_attempt = processor.color_calibration_report["pcc"]["attempts"][0]
        self.assertEqual(local_attempt["label"], "catalog:localgaia")
        self.assertEqual(local_attempt["status"], "skipped")
        self.assertIn("local Gaia astrometric catalog unavailable", local_attempt["error"])

    def test_stage4_offline_pcc_uses_installed_local_astrometric_catalog_only(self):
        processor = self._new_processor()
        processor.fail_commands.add("spcc")

        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "0"}):
            stage4_color_calibration(processor)

        pcc_calls = [call for call in processor.cmd_calls if call[0] == "pcc"]
        self.assertEqual(pcc_calls, [("pcc", "-catalog=localgaia")])
        platesolve_calls = [
            call for call in processor.cmd_calls if call[0] == "platesolve"
        ]
        self.assertTrue(platesolve_calls)
        self.assertIn("-catalog=localgaia", platesolve_calls[0])

    def test_stage4_wraps_spcc_with_single_thread_setcpu_guard(self):
        processor = self._new_processor()

        stage4_color_calibration(processor)

        spcc_index = processor.cmd_calls.index(
            next(call for call in processor.cmd_calls if call[0] == "spcc")
        )
        set_one_index = processor.cmd_calls.index(("setcpu", "1"))
        restore_index = processor.cmd_calls.index(("setcpu", "8"))
        self.assertLess(set_one_index, spcc_index)
        self.assertLess(spcc_index, restore_index)
        self.assertEqual(
            processor.color_calibration_report["spcc"]["cpu_guard"],
            {"runtime": 1, "restore": 8},
        )

    def test_stage4_platesolve_can_be_disabled_explicitly(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = False

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertIn("platesolve disabled by config", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("platesolve", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertNotIn("pcc", cmds)

    def test_stage4_psolved_resume_reuses_wcs_and_runs_pcc_without_platesolve(self):
        processor = self._new_processor()
        processor._stage1_input_mode = "stage4_psolved_resume"
        processor.cfg.spcc_enabled = False
        saved_stems: list[str] = []
        processor._save_stage_output = lambda stem: saved_stems.append(stem) or True

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn(("load", "stage4_psolved"), processor.cmd_calls)
        self.assertNotIn("platesolve", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        self.assertNotIn("stage4_psolved", saved_stems)
        self.assertIn("stage4_color", saved_stems)
        report = processor.color_calibration_report
        self.assertEqual(report["input"], "stage4_psolved")
        self.assertFalse(report["platesolve"]["attempted"])
        self.assertTrue(report["platesolve"]["ok"])

    def test_stage4_uses_average_spiral_white_ref_and_lp_filter_by_default(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.target_profile = {"target_type": "emission_nebula_widefield"}

        stage4_color_calibration(processor)

        spcc_calls = [call for call in processor.cmd_calls if call[0] == "spcc"]
        self.assertTrue(spcc_calls)
        self.assertIn('"-whiteref=Average Spiral Galaxy"', spcc_calls[0])
        self.assertIn('"-oscsensor=Sony IMX585"', spcc_calls[0])
        self.assertIn('"-oscfilter=ZWO Seestar LP"', spcc_calls[0])
        self.assertIn("-limitmag=10.5", spcc_calls[0])
        self.assertNotIn("-bgtol=-2.8,2.0", spcc_calls[0])
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SPCC whiteref=Average Spiral Galaxy", message)

    def test_stage4_uses_zwo_seestar_lp_when_builtin_dualband_filter_enabled(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.cfg.stage4_spcc_builtin_dualband_filter_enabled = True

        stage4_color_calibration(processor)

        spcc_calls = [call for call in processor.cmd_calls if call[0] == "spcc"]
        self.assertTrue(spcc_calls)
        self.assertIn('"-oscsensor=Sony IMX585"', spcc_calls[0])
        self.assertIn('"-oscfilter=ZWO Seestar LP"', spcc_calls[0])
        self.assertIn('"-whiteref=Average Spiral Galaxy"', spcc_calls[0])

    def test_stage4_uses_s30pro_spcc_database_filter_for_lp_header(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.target_profile = {"target_type": "emission_nebula_widefield"}
        processor.header_metadata = {
            "OBJECT": "NGC 2237",
            "RA": 98.01666,
            "DEC": 4.966111,
            "FILTER": "LP",
            "INSTRUME": "imx585",
            "TELESCOP": "S30 Pro_90f61d23",
        }

        stage4_color_calibration(processor)

        spcc_calls = [call for call in processor.cmd_calls if call[0] == "spcc"]
        self.assertTrue(spcc_calls)
        self.assertIn('"-oscsensor=Sony IMX585"', spcc_calls[0])
        self.assertIn('"-oscfilter=ZWO Seestar LP"', spcc_calls[0])
        self.assertIn('"-whiteref=Average Spiral Galaxy"', spcc_calls[0])
        self.assertIn("-limitmag=10.5", spcc_calls[0])
        self.assertEqual(
            processor.color_calibration_report["spcc"]["osc_filter"],
            "ZWO Seestar LP",
        )
        self.assertFalse(processor.color_calibration_report["spcc"]["narrowband"])

    def test_stage4_uses_no_filter_when_lp_is_explicitly_absent(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.header_metadata = {"FILTER": "No filter"}

        stage4_color_calibration(processor)

        spcc_calls = [call for call in processor.cmd_calls if call[0] == "spcc"]
        self.assertTrue(spcc_calls)
        self.assertIn('"-oscsensor=Sony IMX585"', spcc_calls[0])
        self.assertIn('"-oscfilter=No filter"', spcc_calls[0])
        self.assertIn('"-whiteref=Average Spiral Galaxy"', spcc_calls[0])
        self.assertIn("-limitmag=10.5", spcc_calls[0])
        self.assertFalse(processor.color_calibration_report["spcc"]["narrowband"])

    def test_stage4_uses_narrowband_spcc_args_for_ha_oiii_filter_hint(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.header_metadata = {"FILTER": "Ha + OIII narrowband"}

        stage4_color_calibration(processor)

        spcc_calls = [call for call in processor.cmd_calls if call[0] == "spcc"]
        self.assertTrue(spcc_calls)
        flat = " ".join(str(arg) for arg in spcc_calls[0])
        self.assertIn('"-oscsensor=Sony IMX585"', spcc_calls[0])
        self.assertIn('"-whiteref=Average Spiral Galaxy"', spcc_calls[0])
        self.assertNotIn("-oscfilter=", flat)
        self.assertIn("-narrowband", spcc_calls[0])
        self.assertIn("-rwl=656.28", spcc_calls[0])
        self.assertIn("-rbw=20", spcc_calls[0])
        self.assertIn("-gwl=500.70", spcc_calls[0])
        self.assertIn("-gbw=30", spcc_calls[0])
        self.assertIn("-bwl=500.70", spcc_calls[0])
        self.assertIn("-bbw=30", spcc_calls[0])
        self.assertIn("-limitmag=10.5", spcc_calls[0])
        self.assertTrue(processor.color_calibration_report["spcc"]["narrowband"])
        self.assertEqual(processor.color_calibration_report["spcc"]["osc_filter"], "")

    def test_stage4_rejects_explicit_white_ref_missing_from_database(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.target_profile = {"target_type": "emission_nebula_widefield"}
        processor.cfg.stage4_spcc_white_ref = "Photon Flux"

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc_list", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        report = processor.color_calibration_report["spcc"]["metadata_database"]
        self.assertIn("white_reference=Photon Flux", report["missing"])

    def test_stage4_target_aware_white_ref_does_not_retry_average_galaxy(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.cfg.stage4_spcc_adaptive_white_ref_enabled = True
        processor.target_profile = {"target_type": "emission_nebula_widefield"}
        original_cmd_with_check = processor.cmd_with_check

        def fail_nebula_white_ref_once(*args, quiet=False):
            if args and args[0] == "spcc" and '"-whiteref=Star, type G2(v)"' in args:
                processor.cmd_calls.append(args)
                raise pipeline_module.CommandError("mock invalid white ref")
            return original_cmd_with_check(*args, quiet=quiet)

        processor.cmd_with_check = fail_nebula_white_ref_once

        stage4_color_calibration(processor)

        spcc_calls = [call for call in processor.cmd_calls if call[0] == "spcc"]
        self.assertEqual(len(spcc_calls), 1)
        self.assertIn('"-whiteref=Star, type G2(v)"', spcc_calls[0])
        self.assertNotIn(
            '"-whiteref=Average Spiral Galaxy"',
            [arg for call in spcc_calls for arg in call],
        )
        self.assertIn("pcc", [str(call[0]) for call in processor.cmd_calls])
        self.assertEqual(processor.color_calibration_report["method"], "PCC")
        self.assertIsNone(processor.color_calibration_report["spcc_white_reference"]["fallback"])
        self.assertFalse(
            processor.color_calibration_report["spcc_white_reference"][
                "ordinary_galaxy_fallback_allowed"
            ]
        )
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("ordinary galaxy white reference fallback disabled", message)

    def test_stage4_pcc_fallback_stays_ok(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.fail_commands.add("spcc")

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SPCC failed", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("pcc", cmds)
        self.assertNotIn("cc", cmds)

    def test_stage4_skips_spcc_when_setcpu_guard_fails(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.fail_commands.add("setcpu")

        stage4_color_calibration(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("setcpu", cmds)
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SPCC failed: mock failure: setcpu", message)

    def test_stage4_restores_setcpu_before_pcc_after_spcc_failure(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.fail_commands.add("spcc")

        stage4_color_calibration(processor)

        spcc_index = processor.cmd_calls.index(
            next(call for call in processor.cmd_calls if call[0] == "spcc")
        )
        restore_index = processor.cmd_calls.index(("setcpu", "8"))
        pcc_index = processor.cmd_calls.index(
            next(call for call in processor.cmd_calls if call[0] == "pcc")
        )
        self.assertLess(spcc_index, restore_index)
        self.assertLess(restore_index, pcc_index)

    def test_stage4_pcc_uses_siril_default_bgtol_for_142_compat(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.fail_commands.add("spcc")

        stage4_color_calibration(processor)

        pcc_calls = [call for call in processor.cmd_calls if call[0] == "pcc"]
        self.assertEqual(pcc_calls, [("pcc", "-catalog=localgaia")])
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("pcc -catalog=localgaia ok (default bgtol)", message)

    def test_stage4_tries_header_pcc_when_platesolve_failed_with_coordinates(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.target_profile = {"target_type": "emission_nebula_widefield"}
        processor.header_metadata = {
            "OBJECT": "NGC 2237",
            "RA": 98.01666,
            "DEC": 4.966111,
            "FILTER": "LP",
            "INSTRUME": "imx585",
            "TELESCOP": "S30 Pro_90f61d23",
        }
        processor.fail_commands.add("platesolve")

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("platesolve failed: mock failure: platesolve", message)
        self.assertIn("trying PCC header-coordinate fallback", message)
        self.assertIn("pcc -catalog=localgaia ok using FITS header coordinates", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        self.assertNotIn("cc", cmds)
        header_platesolve_calls = [
            call
            for call in processor.cmd_calls
            if call[0] == "platesolve" and str(call[1]).startswith("98.01666,4.966111")
        ]
        self.assertTrue(header_platesolve_calls)
        self.assertEqual(
            processor.color_calibration_report["spcc_white_reference"]["requested"],
            "Average Spiral Galaxy",
        )
        self.assertEqual(
            processor.color_calibration_report["spcc"]["osc_filter"],
            "ZWO Seestar LP",
        )
        self.assertEqual(
            processor.color_calibration_report["platesolve"]["diagnostics"]["failure_kind"],
            "siril_generic_failure",
        )
        self.assertTrue(
            processor.color_calibration_report["platesolve"]["diagnostics"][
                "has_header_coordinates"
            ]
        )
        self.assertEqual(processor.color_calibration_report["method"], "PCC_HEADER")
        self.assertEqual(
            processor.color_calibration_report["warning"],
            "pcc_header_coordinate_fallback",
        )
        self.assertTrue(
            processor.color_calibration_report["pcc"][
                "header_coordinate_fallback_allowed"
            ]
        )

    def test_stage4_local_fallback_skips_star_wb_for_target_aware_nebula(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.target_profile = {"target_type": "emission_nebula_widefield"}
        processor.header_metadata = {
            "OBJECT": "NGC 2237",
            "RA": 98.01666,
            "DEC": 4.966111,
            "FILTER": "LP",
        }
        processor.fail_commands.update({"platesolve", "pcc"})

        stage4_module = sys.modules["stages.stage4_color_calibration"]
        image = stage4_module.np.full((3, 64, 64), 0.05, dtype=stage4_module.np.float32)
        image[0] *= 1.15
        image[2] *= 0.90
        written_pixels = []
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: image.shape,
            get_image_pixeldata=lambda preview=False: image.copy(),
            set_image_pixeldata=lambda pixels: written_pixels.append(pixels),
        )
        with patch.object(
            stage4_module,
            "_stage4_star_white_balance",
            wraps=stage4_module._stage4_star_white_balance,
        ) as star_wb:
            stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertIn("target-aware star white balance skipped", message)
        self.assertTrue(written_pixels)
        star_wb.assert_not_called()
        self.assertEqual(
            processor.color_calibration_report["method"],
            "BACKGROUND_NEUTRALIZATION",
        )
        self.assertEqual(
            processor.color_calibration_report["warning"],
            "target_aware_background_neutralization_only",
        )
        self.assertEqual(
            processor.color_calibration_report["local_fallback"]["star_white_balance"][
                "reason"
            ],
            "target-aware emission/dualband color preservation",
        )

    def test_stage4_background_neutralization_adapts_window_for_large_targets(self):
        stage4_module = sys.modules["stages.stage4_color_calibration"]
        lum = stage4_module.np.linspace(
            0.001,
            0.100,
            10000,
            dtype=stage4_module.np.float32,
        ).reshape(100, 100)
        image = stage4_module.np.stack(
            [lum * 1.10, lum, lum * 0.92],
            axis=0,
        )
        processor = self._new_processor()

        processor.target_profile = {
            "target_type": "large_galaxy",
            "object_stats": {"object_area_ratio": 0.42},
        }
        _large_pixels, large_report = stage4_module._stage4_background_neutralize(
            image,
            processor,
        )

        processor.target_profile = {
            "target_type": "open_cluster",
            "object_stats": {"object_area_ratio": 0.02},
        }
        _small_pixels, small_report = stage4_module._stage4_background_neutralize(
            image,
            processor,
        )

        self.assertEqual(
            large_report["sampling_window"]["mode"],
            "large_target_q5_q25",
        )
        self.assertAlmostEqual(
            large_report["sampling_window"]["upper_quantile"],
            0.25,
        )
        self.assertAlmostEqual(
            small_report["sampling_window"]["upper_quantile"],
            0.45,
        )
        self.assertLess(
            large_report["sample_pixels"],
            small_report["sample_pixels"],
        )

    def test_stage4_background_neutralization_uses_nebulosity_for_target_aware_fields(self):
        stage4_module = sys.modules["stages.stage4_color_calibration"]
        image = stage4_module.np.full((3, 80, 80), 0.05, dtype=stage4_module.np.float32)
        processor = self._new_processor()
        processor.target_profile = {
            "target_type": "emission_nebula_widefield",
            "object_stats": {
                "object_area_ratio": 0.12,
                "nebulosity_area_ratio": 0.41,
            },
        }

        _pixels, report = stage4_module._stage4_background_neutralize(image, processor)

        self.assertEqual(
            report["sampling_window"]["mode"],
            "large_target_q5_q25",
        )
        self.assertAlmostEqual(
            report["sampling_window"]["effective_area_ratio"],
            0.41,
        )

    def test_stage4_local_star_wb_fallback_replaces_fixed_ccm(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.fail_commands.update({"platesolve", "spcc", "pcc"})

        stage4_module = sys.modules["stages.stage4_color_calibration"]
        local_report = {
            "background_neutralization": {"applied": True},
            "star_white_balance": {"applied": True, "white_reference_pixels": 256},
        }
        with patch.object(
            stage4_module,
            "_stage4_local_color_fallback",
            return_value=(
                True,
                "LOCAL_STAR_WB",
                "local_star_white_balance_fallback",
                0.58,
                local_report,
                "local background neutralization + star white balance fallback ok",
            ),
        ):
            stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("local background neutralization + star white balance fallback ok", message)
        self.assertIn("platesolve 失败，已使用本地背景中性化/星点白平衡回退", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("cc", cmds)
        self.assertEqual(processor.color_calibration_report["method"], "LOCAL_STAR_WB")
        self.assertEqual(
            processor.color_calibration_report["warning"],
            "local_star_white_balance_fallback",
        )

    def test_stage4_background_neutralization_only_when_stars_insufficient(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.fail_commands.update({"spcc", "pcc"})

        stage4_module = sys.modules["stages.stage4_color_calibration"]
        local_report = {
            "background_neutralization": {"applied": True},
            "star_white_balance": {
                "applied": False,
                "white_reference_pixels": 12,
                "reason": "insufficient unsaturated low-chroma star samples",
            },
        }
        with patch.object(
            stage4_module,
            "_stage4_local_color_fallback",
            return_value=(
                True,
                "BACKGROUND_NEUTRALIZATION",
                "background_neutralization_only",
                0.35,
                local_report,
                "local background neutralization fallback ok; star samples insufficient",
            ),
        ):
            stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertIn("star samples insufficient", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("cc", cmds)
        self.assertEqual(
            processor.color_calibration_report["method"],
            "BACKGROUND_NEUTRALIZATION",
        )
        self.assertEqual(
            processor.color_calibration_report["warning"],
            "background_neutralization_only",
        )

    def test_stage4_allows_spcc_on_light_preprocess_mode_by_default(self):
        processor = self._new_processor()
        processor._stage1_input_mode = "light_preprocess"

        with patch.dict(os.environ, {"SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS": "1"}):
            stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("spcc", cmds)
        self.assertNotIn("pcc", cmds)
        self.assertNotIn("SPCC skipped on Light_ preprocess mode", message)

    def test_stage4_skips_spcc_on_light_preprocess_when_env_disabled(self):
        processor = self._new_processor()
        processor._stage1_input_mode = "light_preprocess"

        with patch.dict(os.environ, {"SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS": "0"}):
            stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SPCC skipped on Light_ preprocess mode", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)
        self.assertEqual(processor.color_calibration_report["method"], "PCC")

    def test_stage2_applies_adaptive_edge_crop_when_edge_black_remains_high(self):
        processor = self._new_processor()
        processor.cfg.crop_margin = 0.02
        processor.feature_measurements.extend(
            [
                pipeline_module.ImageFeatures(edge_black_ratio=0.19),
                pipeline_module.ImageFeatures(edge_black_ratio=0.05),
                pipeline_module.ImageFeatures(edge_black_ratio=0.05),
            ]
        )
        stage2_module = sys.modules["stages.stage2_view_correction"]

        with (
            patch.object(
                stage2_module,
                "_detect_auto_edge_crop",
                side_effect=[
                    ((10, 10, 980, 980), "initial edge crop"),
                    ((8, 8, 964, 964), "adaptive edge crop"),
                ],
            ),
            patch.object(stage2_module, "_edge_color_artifact_crop", return_value=""),
        ):
            stage2_view_correction(processor)

        crop_calls = [call for call in processor.cmd_calls if call[0] == "crop"]
        self.assertGreaterEqual(len(crop_calls), 2)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("adaptive edge crop", message)

    def test_stage7_weak_object_tuning_uses_current_stretch_configuration(self):
        processor = self._new_processor()
        processor.cfg.asinh_stretch = 2.2
        processor.cfg.nebula_saturation = 0.16
        processor.auto_tune_result = pipeline_module.AutoTuneResult(
            features=pipeline_module.ImageFeatures(
                object_area_ratio=0.002,
                diffuse_ratio=0.0,
                core_brightness_ratio=0.20,
            )
        )

        note = processor._apply_weak_object_tuning()

        self.assertAlmostEqual(processor.cfg.asinh_stretch, 2.45)
        self.assertAlmostEqual(processor.cfg.nebula_saturation, 0.22)
        self.assertIn("weak-object tuning applied", note)

    def test_auto_tune_lifts_low_signal_emission_nebula_without_boosting_stars(self):
        cfg = pipeline_module.PipelineConfig()
        tuned, result = pipeline_module.auto_tune_config(
            cfg,
            pipeline_module.TargetType.EMISSION_NEBULA,
            pipeline_module.ImageFeatures(
                bg_median=0.0020,
                bg_std=0.0001,
                red_dominance=1.02,
                blue_dominance=1.01,
                star_density=0.00008,
                object_area_ratio=0.0002,
                diffuse_ratio=0.0,
                core_brightness_ratio=0.19,
            ),
        )

        self.assertGreaterEqual(tuned.nebula_saturation, 0.30)
        self.assertGreaterEqual(tuned.final_saturation, 0.12)
        self.assertLessEqual(tuned.star_intensity, 0.95)
        self.assertGreaterEqual(tuned.asinh_stretch, 1.85)
        self.assertIn("low-signal emission nebula", " ".join(result.notes))

    def test_detect_target_type_recognizes_ic434_path_as_nebula(self):
        target_type = pipeline_module.detect_target_type(
            Path("/Users/mz/SeeStar/IC 434_sub/process/working.fit")
        )

        self.assertEqual(target_type, pipeline_module.TargetType.EMISSION_NEBULA)

    def test_stage6_bg_gate_allows_sampling_edge_near_configured_floor(self):
        stage6_services = sys.modules["stage6_services"]

        self.assertLessEqual(stage6_services.stage6_effective_bg_median_min(0.020), 0.0199)

    def test_stage7_bright_nebula_uses_target_specific_star_growth_gate(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.cfg = pipeline_module.PipelineConfig()
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        processor._measure_current_quality = lambda: pipeline_module.QualityMetrics(
            bg_median=0.03,
            median_star_size=1.414,
        )
        baseline = pipeline_module.QualityMetrics(
            bg_median=0.003,
            median_star_size=1.0,
        )

        accepted, issues, _metrics = (
            pipeline_module.SeestarPostProcessor._validate_stage6_stretch_quality(
                processor,
                baseline,
            )
        )

        self.assertTrue(accepted)
        self.assertEqual(issues, [])

        processor._active_target_type = lambda: "large_galaxy"
        accepted, issues, _metrics = (
            pipeline_module.SeestarPostProcessor._validate_stage6_stretch_quality(
                processor,
                baseline,
            )
        )

        self.assertFalse(accepted)
        self.assertEqual(issues, ["star_size_growth 1.414>1.250"])

    def test_stage7_compact_stretch_adapts_extreme_low_background(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.002),
                {"bg_std": 0.00005},
            )
        )

        self.assertEqual(adaptation["mode"], "extreme_low_background")
        self.assertEqual(candidates[0]["name"], "cand_a")
        self.assertAlmostEqual(candidates[0]["params"]["asinh_stretch"], 2.4)
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.001)
        self.assertEqual(candidates[1]["name"], "cand_b")
        self.assertAlmostEqual(candidates[1]["params"]["asinh_stretch"], 2.2)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.0005)
        self.assertLessEqual(candidates[1]["params"]["ghs_stretchamount"], 1.01)

    def test_stage7_preview_target_attainment_enforces_brightness_band(self):
        stage6_services = sys.modules["stage6_services"]
        adaptation = {
            "preview_calibration": {
                "candidate_a": {
                    "target_p50": 0.09990,
                    "calibrated_stretch": 1000.0,
                    "stretch_max": 1000.0,
                    "predicted_p50": 0.0388,
                },
                "candidate_b": {
                    "target_p50": 0.06882,
                    "calibrated_stretch": 850.0,
                    "stretch_max": 1000.0,
                    "predicted_p50": 0.06882,
                },
            }
        }

        dark = stage6_services._stage7_preview_target_attainment(
            "cand_a", {"p50": 0.04007}, adaptation
        )
        balanced = stage6_services._stage7_preview_target_attainment(
            "cand_a", {"p50": 0.11880}, adaptation
        )
        overbright = stage6_services._stage7_preview_target_attainment(
            "cand_b", {"p50": 0.26384}, adaptation
        )

        self.assertFalse(dark["accepted"])
        self.assertTrue(dark["stretch_saturated"])
        self.assertIn("preview_target_p50_ratio", dark["issues"][0])
        self.assertTrue(balanced["accepted"])
        self.assertAlmostEqual(balanced["attainment_ratio"], 1.189189, places=5)
        self.assertFalse(overbright["accepted"])
        self.assertAlmostEqual(overbright["attainment_ratio"], 3.833769, places=5)
        self.assertEqual(overbright["maximum_ratio"], 1.50)
        self.assertIn(
            "preview_target_p50_ratio_above_max",
            overbright["issues"][0],
        )

    def test_stage7_compact_stretch_caps_offset_below_low_signal_starless(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.002),
                {"bg_std": 0.00005},
                {"p01": 0.00080, "p99": 0.00224, "max": 0.00298},
            )
        )

        self.assertEqual(adaptation["mode"], "extreme_low_background")
        self.assertIn("offset_cap", adaptation)
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.00064, places=6)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.00050, places=6)
        self.assertLess(candidates[0]["params"]["asinh_offset"], 0.002)
        self.assertLess(candidates[1]["params"]["asinh_offset"], 0.002)

    def test_stage7_compact_stretch_uses_safe_offset_for_sh2_296_statistics(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.0004478424),
                {"bg_std": 0.0000242532},
                {
                    "p01": 0.0002327696,
                    "p99": 0.0013253181,
                    "max": 0.0273016952,
                },
            )
        )

        self.assertEqual(adaptation["mode"], "extreme_low_background")
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.000186, places=6)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.000112, places=6)
        self.assertLess(candidates[0]["params"]["asinh_offset"], 0.0004478424)
        self.assertLess(candidates[1]["params"]["asinh_offset"], 0.0004478424)

    def test_stage7_preview_ref_calibrates_sh2_296_asinh_strength(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.0004478424),
                {"bg_std": 0.0000242532},
                {
                    "p01": 0.0002327696,
                    "p50": 0.0004245928,
                    "p99": 0.0013253181,
                    "max": 0.0273016952,
                },
                {
                    "p50": 0.124821,
                    "p99": 0.800000,
                },
            )
        )

        calibration = adaptation["preview_calibration"]
        self.assertEqual(calibration["source"], "stage7_preview_ref")
        self.assertEqual(candidates[0]["params"]["asinh_stretch"], 1000.0)
        self.assertGreater(candidates[1]["params"]["asinh_stretch"], 100.0)
        self.assertLess(candidates[1]["params"]["asinh_stretch"], 1000.0)
        self.assertGreaterEqual(
            calibration["candidate_a"]["predicted_p50"],
            0.025,
        )
        self.assertLessEqual(
            calibration["candidate_b"]["predicted_p99"],
            calibration["candidate_b"]["target_p99"],
        )

    def test_stage7_preview_calibration_can_be_disabled(self):
        processor = self._new_processor()
        processor.cfg.stage7_preview_calibration_enabled = False
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.0004478424),
                {"bg_std": 0.0000242532},
                {
                    "p01": 0.0002327696,
                    "p50": 0.0004245928,
                    "p99": 0.0013253181,
                    "max": 0.0273016952,
                },
                {"p50": 0.124821, "p99": 0.800000},
            )
        )

        self.assertNotIn("preview_calibration", adaptation)
        self.assertEqual(candidates[0]["params"]["asinh_stretch"], 2.4)
        self.assertEqual(candidates[1]["params"]["asinh_stretch"], 2.2)

    def test_stage7_compact_stretch_keeps_default_for_normal_background(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertEqual(adaptation["mode"], "default_compact")
        self.assertEqual(
            candidates[0]["params"],
            {"asinh_stretch": 2.2, "asinh_offset": 0.002},
        )
        self.assertEqual(candidates[1]["params"]["asinh_offset"], 0.002)

    def test_stage7_compact_stretch_restores_bright_nebula_target_profile(self):
        processor = self._new_processor()
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage6_stretch": {
                "candidate_mode": ["bright_nebula_hdr_masked", "asinh_core_protect"]
            },
        }
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertEqual(adaptation["target_aware"]["name"], "bright_core_protect")
        self.assertEqual(candidates[0]["method"], "bright_nebula_hdr_masked")
        self.assertLess(candidates[0]["params"]["asinh_stretch"], 2.2)
        self.assertAlmostEqual(candidates[0]["params"]["core_protection"], 0.72)
        self.assertLessEqual(candidates[1]["params"]["ghs_stretchamount"], 1.0)

    def test_stage7_compact_stretch_uses_asinh_only_for_star_preserve_targets(self):
        processor = self._new_processor()
        processor._active_target_type = lambda: "open_cluster"
        processor.pipeline_policy = {
            "policy_name": "open_cluster_color_preserve",
            "stage6_stretch": {"candidate_mode": ["star_color_preserving_stretch"]},
        }
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertEqual(adaptation["target_aware"]["name"], "star_colour_preserve")
        self.assertEqual([item["method"] for item in candidates], ["asinh", "asinh"])
        self.assertLess(candidates[1]["params"]["asinh_stretch"], 2.1)

    def test_stage7_target_aware_stretch_can_be_disabled(self):
        processor = self._new_processor()
        processor.cfg.stage7_target_aware_stretch_enabled = False
        processor._active_target_type = lambda: "dark_nebula_low_contrast"
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.SeestarPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertFalse(adaptation["target_aware"]["enabled"])
        self.assertEqual(candidates[0]["method"], "asinh")
        self.assertEqual(candidates[0]["params"]["asinh_stretch"], 2.2)

    def test_stage8_applies_blue_guard_when_starless_layer_is_too_blue(self):
        processor = self._new_processor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.946,
                blue_dominance=1.168,
            )
        )
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.960,
                blue_dominance=1.080,
            )
        )

        stage8_nebula_enhancement(processor)

        ccm_calls = [call for call in processor.cmd_calls if call[0] == "ccm"]
        self.assertTrue(ccm_calls)
        self.assertEqual(ccm_calls[-1][-3:], ("0", "0", "0.860000"))
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("starless blue guard applied", message)
        self.assertIn("Starless 蓝色门控后特征", message)

    def test_stage8_rolls_back_blue_guard_when_feature_gets_worse(self):
        processor = self._new_processor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.933,
                blue_dominance=1.129,
            )
        )
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.932,
                blue_dominance=2.013,
            )
        )

        stage8_nebula_enhancement(processor)

        self.assertIn(("load", "stage8_enhanced"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Starless 蓝色门控回滚", message)

    def test_stage8_narrowband_skips_blue_guard(self):
        processor = self._new_processor()
        processor._channel_semantics = "narrowband_composite"
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.946,
                blue_dominance=1.168,
            )
        )

        stage8_nebula_enhancement(processor)

        self.assertFalse(any(call[0] == "ccm" for call in processor.cmd_calls))
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn(
            "Stage8 global color transforms skipped by channel semantics "
            "(narrowband_composite)",
            message,
        )

    def test_stage8_bg_growth_gate_allows_low_absolute_background_noise(self):
        processor = self._new_processor()
        helper = pipeline_module.stage8_pixels._stage8_bg_noise_growth_issue

        issue = helper(
            processor,
            growth=2.744,
            baseline_std=0.000169,
            candidate_std=0.000464,
            candidate_dirty_score=0.025,
        )

        self.assertIsNone(issue)

    def test_stage8_bg_growth_gate_rejects_material_background_noise(self):
        processor = self._new_processor()
        helper = pipeline_module.stage8_pixels._stage8_bg_noise_growth_issue

        issue = helper(
            processor,
            growth=2.744,
            baseline_std=0.00030,
            candidate_std=0.00120,
            candidate_dirty_score=0.080,
        )

        self.assertIn("bg_std_growth", issue)

    def test_stage10_script_failure_with_aberration_fallback_is_ok(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/SCUNet_Denoise.py",
            }
        )
        processor.script_fail_steps.add("最终降噪")
        processor.script_fail_steps.add("最终降噪回退")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=SASP Aberration API", message)
        self.assertIn("fallback_status=success", message)
        self.assertIn("final_denoise_effective=SASP Aberration API", message)
        self.assertIn("effective_status=success", message)
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "denoiser_chain_to_aberration",
        )

    def test_stage10_narrowband_skips_global_saturation(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._channel_semantics = "narrowband_composite"

        stage10_export(processor)

        self.assertFalse(any(call[0] == "satu" for call in processor.cmd_calls))
        self.assertIn(
            "Stage10 global color adjustment skipped by channel semantics "
            "(narrowband_composite)",
            processor.results[-1][3],
        )

    def test_stage10_color_dominant_input_uses_chroma_plan_with_full_model(self):
        processor = self._new_processor()
        (processor.process_dir / "stage9_remixed.fit").write_bytes(b"mock")
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        pixels = np.full((3, 32, 32), 0.05, dtype=np.float32)
        processor.siril.get_image_pixeldata = lambda preview=False: pixels
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.431,
            "bg_std": 0.003,
            "background_mottling_score": 0.144,
        }
        processor._set_current_image_pixeldata = lambda _image, **_kwargs: None

        stage10_export(processor)

        denoise_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Denoise.py"
        ]
        self.assertTrue(denoise_calls)
        args = denoise_calls[0][2]
        mode_index = args.index("-denoising_mode") + 1
        self.assertEqual(args[mode_index], "full")
        report = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(report["selected_mode"], "chroma")
        self.assertEqual(report["effective_mode"], "chroma")

    def test_stage10_low_noise_input_skips_expensive_denoiser(self):
        processor = self._new_processor()
        (processor.process_dir / "stage9_remixed.fit").write_bytes(b"mock")
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        pixels = np.full((3, 32, 32), 0.05, dtype=np.float32)
        processor.siril.get_image_pixeldata = lambda preview=False: pixels
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.058,
            "bg_std": 0.00084,
            "background_mottling_score": 0.031,
        }

        stage10_export(processor)

        self.assertFalse(
            any(step == "最终降噪" for step, _name, _args in processor.script_calls)
        )
        report = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(report["selected_mode"], "skip")
        self.assertEqual(report["effective_status"], "skipped_safe")
        self.assertTrue(report["skipped_by_low_noise_guard"])
        self.assertFalse(report["skipped_by_review_only"])
        self.assertFalse(report["skipped_by_duplicate_guard"])
        self.assertIn("Stage10 low-noise guard", processor.results[-1][3])
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["components"]["denoise"]["reason_code"],
            "auto_low_noise",
        )

    def test_stage10_script_failure_prefers_scunet_command_fallback(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_fail_steps.add("最终降噪")
        processor.available_commands.add("siril_scunet_denoise")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=Siril-SCUNet Denoise", message)
        self.assertIn("fallback_status=success", message)
        self.assertNotIn("fallback_component=SASP Aberration API", message)
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["components"]["denoise"]["method"],
            "Siril-SCUNet Denoise",
        )

    def test_stage10_uses_in_process_cosmic_clarity_by_default(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("final_denoise_primary=CosmicClarity Denoise in-process script", message)
        self.assertIn("final_denoise_effective=CosmicClarity Denoise script", message)
        self.assertNotIn("fallback_component=", message)
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])

    def test_stage10_export_filename_fallback_is_structured(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage5_denoise_applied = True
        processor._stage8_final_quality = "ok"
        processor._stage8_fallback_used = False
        processor._result_output_basename = lambda: "primary_result"
        processor.main_output_fit_basename_template = "primary_final"
        command_calls: list[tuple[Any, ...]] = []

        def command(*args: Any, quiet: bool = False) -> bool:
            _ = quiet
            command_calls.append(args)
            if args[:2] == ("savetif", "primary_result"):
                raise pipeline_module.CommandError("mock primary TIFF failure")
            return True

        processor.cmd_with_check = command

        stage10_export(processor)

        report = processor.stage_json_reports["stage10_export_report.json"]
        self.assertTrue(report["fallback_used"])
        self.assertEqual(report["fallback_formats"], ["tif"])
        self.assertEqual(report["outputs"]["tif"]["status"], "fallback")
        self.assertIn(("savetif", "result_processed", "-astro"), command_calls)
        metadata = processor.result_metadata[-1]
        self.assertTrue(metadata["fallback_used"])
        self.assertEqual(metadata["reason_code"], "final_export_fallback")
        self.assertTrue(metadata["components"]["export"]["fallback_used"])

    def test_stage10_in_process_path_does_not_depend_on_cli_connection(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.cli_failure_errors["最终降噪"] = (
            "CosmicClarity_Denoise.py: subprocess exited with code 1; "
            "output_tail=Error: Failed to connect to Siril"
        )

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertNotIn("CLI Siril 连接失败", message)
        self.assertIn("primary_status=success", message)

    def test_stage10_uses_native_cosmic_clarity_without_classic_executable(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/CosmicClarity_Native.py",
            }
        )
        processor.classic_cc_args = None

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        classic_calls = [
            call
            for call in processor.script_calls
            if call[1] == "CosmicClarity_Denoise.py"
        ]
        self.assertFalse(classic_calls)
        self.assertIn("CosmicClarity classic 路径未启用，已选择 Native Denoise", message)
        self.assertIn("final_denoise_primary=CosmicClarity Native Denoise cli-subprocess", message)
        self.assertIn("primary_status=success", message)
        self.assertNotIn("fallback_component=CosmicClarity Native Denoise", message)
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])

    def test_cosmic_clarity_native_uses_device_auto_by_default(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/CosmicClarity_Native.py")

        with patch.dict(os.environ, {}, clear=False):
            processor._run_cosmic_clarity_native_denoise_fallback("最终降噪")

        native_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Native.py"
        ]
        self.assertTrue(native_calls)
        self.assertNotIn("--cpu", native_calls[-1][2])

    def test_cosmic_clarity_native_can_force_cpu(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/CosmicClarity_Native.py")

        with patch.dict(os.environ, {"SEESTAR_COSMIC_NATIVE_GPU": "0"}, clear=False):
            processor._run_cosmic_clarity_native_denoise_fallback("最终降噪")

        native_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Native.py"
        ]
        self.assertTrue(native_calls)
        self.assertIn("--cpu", native_calls[-1][2])

    def test_stage10_script_failure_without_scunet_falls_back_to_aberration(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_fail_steps.add("最终降噪")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Siril-SCUNet Denoise 回退不可用", message)
        self.assertIn("fallback_component=SASP Aberration API", message)
        self.assertNotIn("fallback_component=in-process CosmicClarity Denoise script (CosmicClarity_Denoise.py)", message)

    def test_stage10_script_failure_prefers_scunet_script_fallback(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/SCUNet_Denoise.py",
            }
        )
        processor.script_fail_steps.add("最终降噪")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=Siril-SCUNet Denoise", message)
        self.assertNotIn("fallback_component=SASP Aberration API", message)

    def test_stage10_degraded_when_final_denoise_skipped_even_exports_succeed(self):
        processor = self._new_processor()
        processor.cfg.aberration_api_enabled = False

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertIn("最终降噪未执行（script/scunet unavailable, Aberration API disabled）", message)

    def test_stage6_records_post_stretch_feature_summary(self):
        processor = self._new_processor()
        processor.feature_measurements.extend(
            [
                pipeline_module.ImageFeatures(bg_median=0.2),
                pipeline_module.ImageFeatures(bg_median=0.2),
            ]
        )
        processor._run_stage6_ai_stretching = lambda allow_ai: (
            True,
            False,
            [f"allow_ai={allow_ai}"],
            "asinh",
        )

        stage7_stretching(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("stage7_features=", message)
        self.assertIn("bg_median=0.2000", message)

    def test_stage8_records_post_starless_feature_summary(self):
        processor = self._new_processor()
        processor.feature_measurements.append(pipeline_module.ImageFeatures(object_area_ratio=0.33))

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Starless 后特征", message)
        self.assertIn("object_area=0.330", message)

    def test_stage8_conservative_skip_status_survives_additional_guard_reasons(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor._stage8_conservative_mode = True
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": True,
            "conservative_mode": True,
            "status": "skipped",
            "final_quality": "skipped",
            "reasons": [
                "stage8_conservative_mode_after_stage7_starless_repair",
                "stage7_quality_status=poor",
            ],
        }

        stage8_nebula_enhancement(processor)

        self.assertEqual(processor._stage8_final_quality, "conservative_skipped")
        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        quality = processor.stage_json_reports["stage8_quality.json"]
        self.assertEqual(report["status"], "conservative_skipped")
        self.assertEqual(report["final_quality"], "conservative_skipped")
        self.assertEqual(quality["initial"]["status"], "conservative_skipped")
        self.assertEqual(quality["final"]["final_quality"], "conservative_skipped")

    def test_final_quality_recognizes_stage8_conservative_skip(self):
        probe = SimpleNamespace(
            _stage8_final_quality="conservative_skipped",
            _stage8_fallback_used=True,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.35,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "emission_nebula_widefield",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertFalse(report["strict_gate"])
        self.assertTrue(report["metrics"]["stage8_conservative_skipped"])
        self.assertEqual(report["final_quality"], "ok")

    def test_final_quality_rejects_hidden_compact_stage7_halo(self):
        probe = SimpleNamespace(
            _stage8_final_quality="conservative_skipped",
            _stage8_fallback_used=True,
            _stage9_bypassed_bad_starless=False,
            _stage7_selected_quality={
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.427,
                    "global_halo_residue_score": 0.427,
                    "compact_halo_residue_score": 0.857,
                },
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.427,
            _stage7_effective_halo_threshold=lambda: 0.60,
            _active_policy_name=lambda: "bright_emission_reflection_nebula",
            _active_target_type=lambda: "bright_emission_reflection_nebula",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertEqual(report["final_quality"], "poor")
        self.assertIn(
            "stage7_compact_halo_residue_score 0.857>0.600",
            report["issues"],
        )
        self.assertEqual(
            report["metrics"]["stage7_compact_halo_residue_score"],
            0.857,
        )

    def test_final_quality_accepts_near_limit_compact_halo_after_safe_star_remix(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_stars_application_mode="screen",
            _stage9_starmask_stretch_failed=False,
            _stage9_selected_remix_quality={
                "metrics": {
                    "chromatic_star_addition_ratio": 0.00001,
                    "local_color_risk_score": 0.66,
                },
                "limits": {"chromatic_star_addition_ratio": 0.003},
            },
            _stage7_selected_quality={
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.343,
                    "global_halo_residue_score": 0.343,
                    "compact_halo_residue_score": 0.637,
                    "compact_residual_star_score": 0.001,
                    "compact_residual_coverage": 0.0001,
                },
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.01,
                "background_mottling_score": 0.03,
                "local_patch_variance": 0.000001,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.06,
                "bg_dirty_score": 0.04,
                "bg_std": 0.004,
            },
            _stage7_halo_residue_score=lambda: 0.343,
            _stage7_effective_halo_threshold=lambda: 0.60,
            _active_policy_name=lambda: "bright_nebula_hdr_conservative",
            _active_target_type=lambda: "bright_emission_reflection_nebula",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertFalse(report["strict_gate"])
        self.assertEqual(report["final_quality"], "ok")
        self.assertTrue(
            report["metrics"]["stage7_compact_halo_raw_limit_exceeded"]
        )
        self.assertTrue(
            report["metrics"]["stage7_compact_halo_target_aware_exempted"]
        )

    def test_final_quality_rejects_selected_stage9_chromatic_artifacts(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_stars_application_mode="screen",
            _stage9_selected_remix_quality={
                "metrics": {"chromatic_star_addition_ratio": 0.010},
                "limits": {"chromatic_star_addition_ratio": 0.003},
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "emission_nebula_widefield",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertEqual(report["final_quality"], "poor")
        self.assertIn(
            "stage9_chromatic_star_addition_ratio 0.010000>0.003000",
            report["issues"],
        )
        self.assertEqual(
            report["metrics"]["stage9_chromatic_star_addition_ratio"],
            0.010,
        )

    def test_final_quality_requires_review_after_unsafe_stage9_bypass(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=True,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertEqual(report["final_quality"], "poor")
        self.assertTrue(report["needs_conservative_rerun"])

    def test_final_quality_requires_review_when_required_stars_not_applied(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=False,
            _stage9_stars_application_mode="no_starmask",
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertIn("stage9_required_stars_not_applied", report["issues"])
        self.assertTrue(report["metrics"]["stage9_stars_required"])
        self.assertFalse(report["metrics"]["stage9_stars_applied"])
        self.assertTrue(report["needs_conservative_rerun"])

    def test_final_quality_requires_review_after_starmask_stretch_failure(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_starmask_stretch_failed=True,
            _stage9_stars_application_mode="screen",
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertIn("stage9_starmask_stretch_failed", report["issues"])
        self.assertTrue(report["metrics"]["stage9_starmask_stretch_failed"])
        self.assertTrue(report["needs_conservative_rerun"])

    def test_stage8_input_guard_classifies_conservative_skip_with_other_reasons(self):
        probe = SimpleNamespace(
            _stage8_conservative_mode=True,
            _stage7_selected_quality={"status": "poor", "derived": {}},
            cfg=SimpleNamespace(
                stage7_residual_star_score_max=0.45,
                stage7_starless_noise_gain_max=1.25,
                stage8_mask_signal_coverage_min=0.002,
            ),
            siril=SimpleNamespace(get_image_pixeldata=lambda preview=False: None),
            _stage7_halo_residue_score=lambda: 0.0,
            _stage7_effective_halo_threshold=lambda: 0.35,
            _short_text=lambda value, _limit=120: str(value),
        )

        report = pipeline_module.stage8_pixels.stage8_input_enhancement_guard(probe)

        self.assertEqual(report["status"], "conservative_skipped")
        self.assertEqual(report["final_quality"], "conservative_skipped")
        self.assertIn("stage7_quality_status=poor", report["reasons"])

    def test_stage8_input_guard_allows_limited_m42_candidate_with_valid_mask(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        starmask_file = Path(td.name) / "starmask.fit"
        starmask_file.write_bytes(b"mock")
        reason_text = (
            "bright_nebula_halo_advisory: "
            "0.488 > 0.350, accepted_limit=0.600"
        )
        probe = SimpleNamespace(
            _stage8_handoff={
                "processing_policy": "limited",
                "reason_code": "bright_nebula_halo_advisory",
                "reason_text": reason_text,
                "reasons": [
                    {
                        "code": "bright_nebula_halo_advisory",
                        "value": 0.488,
                        "effective_value": 0.493,
                        "base_limit": 0.35,
                        "accepted_limit": 0.60,
                    }
                ],
            },
            _stage7_selected_quality={
                "status": "ok",
                "derived": {
                    "residual_star_score": 0.10,
                    "halo_residue_score": 0.488,
                    "compact_halo_residue_score": 0.493,
                    "starless_noise_gain": 1.0,
                },
            },
            _stage7_starless_skipped=False,
            starmask_file=starmask_file,
            cfg=SimpleNamespace(
                stage8_masked_enhancement_enabled=True,
                stage7_residual_star_score_max=0.45,
                stage7_halo_residue_score_max=0.35,
                stage7_starless_noise_gain_max=1.25,
                stage8_mask_signal_coverage_min=0.002,
            ),
            siril=SimpleNamespace(get_image_pixeldata=lambda preview=False: None),
            _stage7_halo_residue_score=lambda: 0.488,
            _stage7_effective_halo_threshold=lambda: 0.60,
            _active_target_type=lambda: "bright_emission_reflection_nebula",
            _short_text=lambda value, _limit=120: str(value),
        )

        report = pipeline_module.stage8_pixels.stage8_input_enhancement_guard(probe)

        self.assertFalse(report["skip_enhancement"])
        self.assertEqual(report["processing_policy"], "limited")
        self.assertEqual(report["reason_code"], "bright_nebula_halo_advisory")
        self.assertEqual(report["advisories"], [reason_text])
        self.assertAlmostEqual(
            report["derived"]["compact_halo_residue_score"],
            0.493,
        )

    def test_stage8_limited_halo_texture_gate_rejects_new_ring_detail(self):
        cfg = SimpleNamespace(
            stage8_limited_halo_texture_growth_max=1.05,
            stage8_limited_halo_texture_delta_max=0.00075,
        )
        probe = SimpleNamespace(cfg=cfg)
        baseline = np.full((3, 64, 64), 0.12, dtype=np.float32)
        candidate = baseline.copy()
        starmask = np.zeros_like(baseline)
        starmask[:, 29:35, 29:35] = 1.0
        yy, xx = np.indices((64, 64))
        radius = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
        ring = (radius >= 5.0) & (radius <= 10.0)
        checker = np.where((xx + yy) % 2 == 0, 0.035, -0.035)
        candidate[:, ring] = np.clip(
            candidate[:, ring] + checker[ring],
            0.0,
            1.0,
        )

        report = pipeline_module.stage8_pixels.stage8_limited_halo_texture_report(
            probe,
            baseline,
            candidate,
            starmask,
        )
        unchanged = pipeline_module.stage8_pixels.stage8_limited_halo_texture_report(
            probe,
            baseline,
            baseline.copy(),
            starmask,
        )

        self.assertTrue(report["available"])
        self.assertFalse(report["accepted"])
        self.assertGreater(report["growth"], 1.05)
        self.assertGreater(report["absolute_delta"], 0.00075)
        self.assertTrue(unchanged["accepted"])

    def test_stage8_limited_halo_gate_extracts_compact_support_from_diffuse_starmask(self):
        cfg = SimpleNamespace(
            stage8_limited_halo_texture_growth_max=1.05,
            stage8_limited_halo_texture_delta_max=0.00075,
        )
        probe = SimpleNamespace(cfg=cfg)
        baseline = np.full((3, 128, 128), 0.12, dtype=np.float32)
        rng = np.random.default_rng(42)
        diffuse = rng.uniform(0.0, 0.10, size=(128, 128)).astype(np.float32)
        for y, x in ((20, 20), (32, 96), (64, 64), (96, 28), (105, 105)):
            diffuse[y, x] = 1.0
        starmask = np.repeat(diffuse[None, ...], 3, axis=0)

        report = pipeline_module.stage8_pixels.stage8_limited_halo_texture_report(
            probe,
            baseline,
            baseline.copy(),
            starmask,
        )

        self.assertTrue(report["available"])
        self.assertTrue(report["accepted"])
        self.assertLessEqual(report["ring_coverage"], 0.45)
        self.assertGreaterEqual(report["support_quantile"], 0.90)
        self.assertLess(report["core_coverage"], 0.05)

    def test_stage8_prefers_accepted_stage7_stretched_input(self):
        processor = self._new_processor()
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")

        stage8_nebula_enhancement(processor)

        self.assertIn(("load", "stage7_stretched"), processor.cmd_calls)
        self.assertNotIn(("load", "starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "stage7_stretched")
        self.assertIn("stage8_input_source=stage7_stretched", processor.results[-1][3])

    def test_stage8_ignores_stale_stage7_output_when_not_accepted(self):
        processor = self._new_processor()
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = False
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"stale")

        stage8_nebula_enhancement(processor)

        self.assertNotIn(("load", "stage7_stretched"), processor.cmd_calls)
        self.assertIn(("load", "starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "starless")
        self.assertIn("Stage7 output not accepted", processor.results[-1][3])

    def test_stage8_falls_back_when_accepted_stage7_file_is_missing(self):
        processor = self._new_processor()
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"

        stage8_nebula_enhancement(processor)

        self.assertNotIn(("load", "stage7_stretched"), processor.cmd_calls)
        self.assertIn(("load", "starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "starless")
        self.assertIn("preferred Stage7 input missing", processor.results[-1][3])

    def test_stage8_falls_back_to_stage6_starless_when_starless_load_fails(self):
        processor = self._new_processor()
        processor._stage7_stretch_accepted = False
        original_cmd_with_check = processor.cmd_with_check

        def selective_load_failure(*args: Any, quiet: bool = False) -> bool:
            if args == ("load", "starless"):
                processor.cmd_calls.append(args)
                raise pipeline_module.CommandError("mock missing starless")
            return original_cmd_with_check(*args, quiet=quiet)

        processor.cmd_with_check = selective_load_failure

        stage8_nebula_enhancement(processor)

        self.assertIn(("load", "starless"), processor.cmd_calls)
        self.assertIn(("load", "stage6_starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "stage6_starless")
        self.assertTrue(processor._stage8_input_fallback_used)
        selection = processor.stage_json_reports["stage8_input_selection.json"]
        self.assertEqual(selection["selected_source"], "stage6_starless")
        self.assertTrue(selection["fallback_used"])

    def test_stage7_marks_saved_quality_candidate_as_accepted_for_stage8(self):
        processor = self._new_processor()
        processor._run_stage6_ai_stretching = lambda allow_ai=False: (
            True,
            False,
            ["quality_ok=true"],
            "Asinh",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertTrue(processor._stage7_stretch_accepted)
        self.assertEqual(processor._stage7_stretch_output, "stage7_stretched")

    def test_stage7_marks_validated_rescue_as_accepted_and_ok(self):
        processor = self._new_processor()
        processor._stage7_stretch_validated_rescue = True
        processor._run_stage6_ai_stretching = lambda allow_ai=False: (
            True,
            True,
            ["stage7 background chroma rescue accepted"],
            "background_chroma_rescue",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertTrue(processor._stage7_stretch_accepted)
        self.assertEqual(processor._stage7_stretch_output, "stage7_stretched")
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "validated_chroma_rescue",
        )

    def test_stage7_marks_review_only_candidate_as_degraded_not_accepted(self):
        processor = self._new_processor()

        def review_only_stretch(allow_ai=False):
            processor._stage7_review_source = "stage7_cand_a"
            return False, False, ["selected safe review source"], ""

        processor._run_stage6_ai_stretching = review_only_stretch

        pipeline_module.run_stage7_stretching(processor)

        self.assertFalse(processor._stage7_stretch_accepted)
        self.assertIsNone(processor._stage7_stretch_output)
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIn("仅保留 Stage7 复核候选", processor.results[-1][3])

    def test_stage7_chroma_rescue_only_allows_exclusive_chroma_rejection(self):
        processor = pipeline_module.SeestarPostProcessor()
        chroma_only = {
            "status": "ok",
            "stem": "stage7_cand_a",
            "target_local_quality": {"accepted": True},
            "diagnostics": [
                "background_chroma_noise_score 0.431>0.340",
                "background_chroma_load_growth 3.210>1.350",
            ],
        }

        self.assertTrue(processor._stage7_attempt_allows_chroma_rescue(chroma_only))
        self.assertFalse(
            processor._stage7_attempt_allows_chroma_rescue(
                {
                    **chroma_only,
                    "diagnostics": [
                        *chroma_only["diagnostics"],
                        "background_mottling_score 0.600>0.450",
                    ],
                }
            )
        )
        self.assertFalse(
            processor._stage7_attempt_allows_chroma_rescue(
                {
                    **chroma_only,
                    "diagnostics": [
                        *chroma_only["diagnostics"],
                        "preview_target_p50_ratio_above_max 3.833>1.500",
                    ],
                }
            )
        )
        processor.cfg.stage7_chroma_rescue_enabled = False
        self.assertFalse(
            processor._stage7_attempt_allows_chroma_rescue(chroma_only)
        )

    def test_stage7_chroma_rescue_uses_three_strength_levels(self):
        processor = pipeline_module.SeestarPostProcessor()

        self.assertEqual(
            processor._stage7_chroma_rescue_strengths(),
            [0.35, 0.55, 0.65],
        )

    def test_stage7_candidate_selection_prefers_best_post_rescue_quality(self):
        processor = pipeline_module.SeestarPostProcessor()
        gate_limits = {
            "chroma_noise_score_max": 0.34,
            "background_mottling_score_max": 0.45,
            "chroma_load_growth_max": 1.35,
        }

        def attempt(name: str, chroma: float, risk: float) -> dict[str, Any]:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": False,
                "diagnostics": [
                    f"background_chroma_noise_score {chroma:.3f}>0.340"
                ],
                "risk_score": risk,
                "background_quality_gate": {
                    "metrics": {
                        "chroma_noise_score": chroma,
                        "background_mottling_score": 0.10,
                        "chroma_load_growth": 1.10,
                    },
                    "limits": gate_limits,
                },
            }

        candidates = [
            attempt("cand_b", 0.751, 10.0),
            attempt("chroma_rescue_1", 0.515, 8.0),
            attempt("chroma_rescue_2", 0.380, 5.0),
        ]

        selected = min(
            candidates,
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "chroma_rescue_2")

    def test_stage7_candidate_selection_keeps_hard_gate_for_final_output(self):
        processor = pipeline_module.SeestarPostProcessor()
        rejected_low_risk = {
            "name": "rejected_low_risk",
            "stem": "stage7_rejected_low_risk",
            "status": "ok",
            "allowed_as_final": False,
            "diagnostics": ["background_chroma_noise_score 0.350>0.340"],
            "risk_score": 1.0,
        }
        accepted_higher_risk = {
            "name": "accepted_higher_risk",
            "stem": "stage7_accepted_higher_risk",
            "status": "ok",
            "allowed_as_final": True,
            "diagnostics": [],
            "risk_score": 4.0,
        }

        selected = min(
            [rejected_low_risk, accepted_higher_risk],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "accepted_higher_risk")

    def test_stage7_review_selection_prefers_lowest_safe_preview_ratio(self):
        processor = pipeline_module.SeestarPostProcessor()

        def review_attempt(
            name: str,
            ratio: float,
            diagnostics: list[str],
            *,
            local_ok: bool = True,
        ) -> dict[str, Any]:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": False,
                "diagnostics": diagnostics,
                "risk_score": 1.0,
                "pixel_stats": {
                    "p50": ratio * 0.10,
                    "p99": 0.80,
                    "dynamic_range": 0.79,
                },
                "preview_target_attainment": {
                    "attainment_ratio": ratio,
                },
                "target_local_quality": {"accepted": local_ok},
            }

        candidate_a = review_attempt(
            "cand_a",
            1.19,
            ["background_chroma_noise_score 0.431>0.340"],
        )
        candidate_b = review_attempt(
            "cand_b",
            3.83,
            [
                "background_chroma_noise_score 0.401>0.340",
                "preview_target_p50_ratio_above_max 3.830>1.500",
            ],
        )
        unsafe_lower_ratio = review_attempt(
            "unsafe_core",
            1.05,
            ["background_chroma_noise_score 0.410>0.340"],
            local_ok=False,
        )

        safe_candidates = [
            attempt
            for attempt in (candidate_b, unsafe_lower_ratio, candidate_a)
            if processor._stage7_review_candidate_is_safe(attempt)
        ]
        selected = min(
            safe_candidates,
            key=processor._stage7_review_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "cand_a")
        self.assertNotIn(unsafe_lower_ratio, safe_candidates)

    def test_stage7_chroma_rescue_preserves_luminance_and_signal_region(self):
        processor = pipeline_module.SeestarPostProcessor()
        rgb = np.empty((3, 8, 8), dtype=np.float32)
        rgb[0] = 0.34
        rgb[1] = 0.22
        rgb[2] = 0.30
        luma = (
            0.2126 * rgb[0]
            + 0.7152 * rgb[1]
            + 0.0722 * rgb[2]
        ).astype(np.float32)
        background_mask = np.ones((8, 8), dtype=np.float32)
        background_mask[3:5, 3:5] = 0.0
        processor._stage8_generate_starless_masks = lambda _image: {
            "rgb": rgb.copy(),
            "gray": luma.copy(),
            "background_mask": background_mask,
            "coverage": {
                "core": 4.0 / 64.0,
                "nebula": 0.0,
                "faint_nebula": 0.0,
            },
        }
        processor._stage8_restore_rgb_like = (
            lambda source, rescued_rgb: rescued_rgb.astype(source.dtype, copy=False)
        )

        rescued, metadata = processor._stage7_background_chroma_rescue_pixels(
            rgb,
            strength=0.55,
        )
        rescued_luma = (
            0.2126 * rescued[0]
            + 0.7152 * rescued[1]
            + 0.0722 * rescued[2]
        )
        before_chroma = np.std(rgb[:, 0, 0] - luma[0, 0])
        after_chroma = np.std(rescued[:, 0, 0] - rescued_luma[0, 0])

        self.assertTrue(np.allclose(rescued_luma, luma, atol=1e-6))
        self.assertLess(after_chroma, before_chroma * 0.50)
        self.assertTrue(np.allclose(rescued[:, 3:5, 3:5], rgb[:, 3:5, 3:5]))
        self.assertTrue(metadata["luminance_preserved"])

    def test_stage7_background_gate_rejects_case_candidates_a_and_b(self):
        processor = pipeline_module.SeestarPostProcessor()
        baseline = {
            "chroma_noise_score": 0.003310588,
            "background_mottling_score": 0.000616090,
            "bg_std": 0.000019856,
            "bg_median": 0.000494266,
        }
        candidate_a = {
            "chroma_noise_score": 0.430704,
            "background_mottling_score": 0.078502,
            "bg_std": 0.00253955,
            "bg_median": 0.03274113,
        }
        candidate_b = {
            "chroma_noise_score": 0.795766,
            "background_mottling_score": 0.20,
            "bg_std": 0.00941930,
            "bg_median": 0.19506431,
        }

        gate_a = processor._stage7_stretch_background_gate(baseline, candidate_a)
        gate_b = processor._stage7_stretch_background_gate(baseline, candidate_b)

        self.assertFalse(gate_a["accepted"])
        self.assertFalse(gate_b["accepted"])
        self.assertTrue(
            any("background_chroma_noise_score" in issue for issue in gate_a["issues"])
        )
        self.assertTrue(
            any("background_chroma_load_growth" in issue for issue in gate_a["issues"])
        )
        self.assertTrue(
            any("background_chroma_noise_score" in issue for issue in gate_b["issues"])
        )

    def test_background_chroma_noise_ignores_smooth_colour_bias(self):
        processor = pipeline_module.SeestarPostProcessor()
        smooth_red = np.full((3, 64, 64), 0.04, dtype=np.float32)
        smooth_red[0] = 0.16

        metrics = processor._background_quality_metrics(smooth_red)

        self.assertLess(metrics["chroma_noise_score"], 0.05)
        self.assertGreater(metrics["background_chroma_load"], 0.50)

    def test_background_chroma_noise_detects_high_frequency_colour_variation(self):
        processor = pipeline_module.SeestarPostProcessor()
        yy, xx = np.mgrid[:64, :64]
        checker = ((xx + yy) % 2).astype(np.float32)
        noisy = np.full((3, 64, 64), 0.08, dtype=np.float32)
        noisy[0] += checker * 0.10
        noisy[2] += (1.0 - checker) * 0.10

        metrics = processor._background_quality_metrics(noisy)

        self.assertGreater(metrics["chroma_noise_score"], 0.34)

    def test_stage7_background_gate_prefers_direct_chroma_load_metric(self):
        processor = pipeline_module.SeestarPostProcessor()
        baseline = {
            "chroma_noise_score": 0.02,
            "background_chroma_load": 0.80,
            "background_mottling_score": 0.01,
            "bg_std": 0.001,
            "bg_median": 0.002,
        }
        candidate = {
            "chroma_noise_score": 0.03,
            "background_chroma_load": 0.60,
            "background_mottling_score": 0.02,
            "bg_std": 0.005,
            "bg_median": 0.05,
        }

        gate = processor._stage7_stretch_background_gate(baseline, candidate)

        self.assertTrue(gate["accepted"], gate)
        self.assertAlmostEqual(gate["metrics"]["chroma_load"], 0.60)
        self.assertAlmostEqual(gate["metrics"]["chroma_load_growth"], 0.75)

    def test_stage7_background_gate_checks_mottling_and_accepts_safe_candidate(self):
        processor = pipeline_module.SeestarPostProcessor()
        baseline = {
            "chroma_noise_score": 0.0033,
            "background_mottling_score": 0.001,
            "bg_std": 0.00002,
            "bg_median": 0.0005,
        }
        safe = {
            "chroma_noise_score": 0.20,
            "background_mottling_score": 0.20,
            "bg_std": 0.003,
            "bg_median": 0.05,
        }
        mottled = {**safe, "background_mottling_score": 0.60}

        self.assertTrue(
            processor._stage7_stretch_background_gate(baseline, safe)["accepted"]
        )
        mottled_gate = processor._stage7_stretch_background_gate(baseline, mottled)
        self.assertFalse(mottled_gate["accepted"])
        self.assertTrue(
            any("background_mottling_score" in issue for issue in mottled_gate["issues"])
        )

    def test_stage7_background_gate_exempts_low_absolute_load_in_extreme_low_background(self):
        processor = pipeline_module.SeestarPostProcessor()
        baseline = {
            "chroma_noise_score": 0.0012268481441424228,
            "background_mottling_score": 0.0004914217773451431,
            "bg_std": 0.000014195245057635475,
            "bg_median": 0.0005103948060423136,
        }
        candidate_a = {
            "chroma_noise_score": 0.15973878325894475,
            "background_mottling_score": 0.06247950174535314,
            "bg_std": 0.001812034985050559,
            "bg_median": 0.03351139277219772,
        }
        candidate_b = {
            "chroma_noise_score": 0.4499879052243117,
            "background_mottling_score": 0.1006566743036724,
            "bg_std": 0.007080338895320892,
            "bg_median": 0.18866348266601562,
        }

        gate_a = processor._stage7_stretch_background_gate(baseline, candidate_a)
        gate_b = processor._stage7_stretch_background_gate(baseline, candidate_b)

        self.assertTrue(gate_a["accepted"])
        self.assertAlmostEqual(gate_a["metrics"]["chroma_load"], 0.04766700815594566)
        self.assertTrue(
            gate_a["metrics"]["chroma_load_growth_low_absolute_exempted"]
        )
        self.assertTrue(gate_a["metrics"]["extreme_low_background"])
        self.assertFalse(gate_b["accepted"])
        self.assertTrue(
            any("background_chroma_noise_score" in issue for issue in gate_b["issues"])
        )

    def test_stage7_background_gate_still_rejects_high_absolute_load_growth(self):
        processor = pipeline_module.SeestarPostProcessor()
        baseline = {
            "chroma_noise_score": 0.10,
            "background_mottling_score": 0.01,
            "bg_std": 0.002,
            "bg_median": 0.05,
        }
        candidate = {
            "chroma_noise_score": 0.30,
            "background_mottling_score": 0.10,
            "bg_std": 0.004,
            "bg_median": 0.03,
        }

        gate = processor._stage7_stretch_background_gate(baseline, candidate)

        self.assertFalse(gate["accepted"])
        self.assertFalse(gate["metrics"]["chroma_load_growth_low_absolute_exempted"])
        self.assertTrue(
            any("background_chroma_load_growth" in issue for issue in gate["issues"])
        )

    def test_stage7_background_gate_does_not_exempt_normal_background_growth(self):
        processor = pipeline_module.SeestarPostProcessor()
        baseline = {
            "chroma_noise_score": 0.10,
            "background_mottling_score": 0.01,
            "bg_std": 0.002,
            "bg_median": 0.05,
        }
        candidate = {
            "chroma_noise_score": 0.20,
            "background_mottling_score": 0.10,
            "bg_std": 0.003,
            "bg_median": 0.05,
        }

        gate = processor._stage7_stretch_background_gate(baseline, candidate)

        self.assertLessEqual(
            gate["metrics"]["chroma_load"],
            gate["limits"]["chroma_load_low_absolute_max"],
        )
        self.assertFalse(gate["metrics"]["extreme_low_background"])
        self.assertFalse(gate["accepted"])
        self.assertTrue(
            any("background_chroma_load_growth" in issue for issue in gate["issues"])
        )

    def test_stage7_pixel_repair_accepts_significant_chroma_reduction(self):
        cfg = pipeline_module.PipelineConfig()
        assessment = stage7_star_separation_module._stage7_chroma_repair_acceptance(
            cfg,
            {"chroma_noise_score": 0.003310588},
            {"chroma_noise_score": 0.001822228},
            residual_not_worse=True,
            halo_not_worse=True,
        )

        self.assertTrue(assessment["accepted"])
        self.assertGreater(assessment["reduction_ratio"], 0.40)

    def test_stage7_pixel_repair_rejects_chroma_gain_when_halo_worsens(self):
        cfg = pipeline_module.PipelineConfig()
        assessment = stage7_star_separation_module._stage7_chroma_repair_acceptance(
            cfg,
            {"chroma_noise_score": 0.003310588},
            {"chroma_noise_score": 0.001822228},
            residual_not_worse=True,
            halo_not_worse=False,
        )

        self.assertFalse(assessment["accepted"])

    def test_stage7_does_not_accept_degraded_candidate_for_stage8(self):
        processor = self._new_processor()
        processor._run_stage6_ai_stretching = lambda allow_ai=False: (
            True,
            True,
            ["quality_ok=false"],
            "Asinh",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertFalse(processor._stage7_stretch_accepted)
        self.assertIsNone(processor._stage7_stretch_output)

    def test_stage8_builtin_saturation_fallback_is_reported_as_internal_processing(self):
        processor = self._new_processor()

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("内置 Starless satu", message)
        self.assertIn("内置 Starless unsharp", message)
        self.assertNotIn("插件未命中", message)

    def test_stage8_uses_builtin_without_siril_command_probe_when_api_unavailable_by_default(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("内置 Starless", message)
        self.assertNotIn("曲线/蒙版工具2", processor.command_chain_calls)
        self.assertNotIn("细节/结构增强", processor.command_chain_calls)
        self.assertTrue(processor.sasp_stage8_calls)

    def test_stage8_uses_sasp_python_api_when_siril_commands_unavailable(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.sasp_stage8_label = "SASP WaveScale Dark Enhancer API"

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SASP Starless 深加工使用 SASP WaveScale Dark Enhancer API", message)
        self.assertNotIn("内置 Starless", message)
        self.assertEqual(
            processor.workflow_command_used.get("SASP Starless 深加工 API"),
            "SASP WaveScale Dark Enhancer API",
        )
        self.assertNotIn("曲线/蒙版工具2", processor.command_chain_calls)
        self.assertNotIn("细节/结构增强", processor.command_chain_calls)

    def test_stage8_limited_candidate_uses_masked_builtin_only_and_is_accepted(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor.cfg.stage8_limited_saturation_max = 0.05
        processor.cfg.optional_color_transform_enabled = True
        processor._stage8_handoff = {
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": (
                "bright_nebula_halo_advisory: "
                "0.488 > 0.350, accepted_limit=0.600"
            ),
            "reasons": [],
        }
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": False,
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": processor._stage8_handoff["reason_text"],
            "reason_details": [],
            "advisories": [processor._stage8_handoff["reason_text"]],
        }
        captured_plans: list[dict[str, Any]] = []
        processor._apply_stage8_builtin_enhancement = (
            lambda plan, *, label: (
                captured_plans.append(dict(plan))
                or [f"{label} masked limited candidate"]
            )
        )
        processor._stage8_quality_assessment = lambda: {
            "status": "ok",
            "issues": [],
        }
        saved_stems: list[str] = []
        processor._save_stage_output = lambda stem: saved_stems.append(stem) or True

        stage8_nebula_enhancement(processor)

        self.assertEqual(processor.results[-1][1], "ok")
        self.assertEqual(processor._stage8_final_source, "stage8_enhanced")
        self.assertFalse(processor._stage8_handoff["passthrough"])
        self.assertTrue(processor._stage8_handoff["restricted_downstream"])
        self.assertEqual(
            processor._stage8_handoff["outcome_reason_code"],
            "stage8_limited_candidate_accepted",
        )
        self.assertIn("stage8_limited_candidate", saved_stems)
        self.assertFalse(processor.sasp_stage8_calls)
        self.assertNotIn("调色1（可选）", processor.command_chain_calls)
        self.assertEqual(captured_plans[0]["bg_factor"], 0)
        self.assertEqual(captured_plans[0]["unsharp_radius"], 0.0)
        self.assertEqual(captured_plans[0]["unsharp_amount"], 0.0)
        self.assertLessEqual(captured_plans[0]["saturation"], 0.05)
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "completed",
        )

    def test_stage8_limited_candidate_rejection_preserves_candidate_and_rolls_back(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor.cfg.stage8_limited_saturation_max = 0.05
        processor._stage8_handoff = {
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": (
                "bright_nebula_halo_advisory: "
                "0.488 > 0.350, accepted_limit=0.600"
            ),
            "reasons": [],
        }
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": False,
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": processor._stage8_handoff["reason_text"],
            "reason_details": [],
            "advisories": [processor._stage8_handoff["reason_text"]],
        }
        processor._apply_stage8_builtin_enhancement = (
            lambda _plan, *, label: [f"{label} masked limited candidate"]
        )
        processor._stage8_quality_assessment = lambda: {
            "status": "poor",
            "issues": ["stage8_limited_halo_texture_growth_exceeded"],
        }
        rollback_calls: list[str] = []
        processor._rollback_stage8_to_input = (
            lambda: rollback_calls.append("stage8_input_starless") or True
        )
        saved_stems: list[str] = []
        processor._save_stage_output = lambda stem: saved_stems.append(stem) or True

        stage8_nebula_enhancement(processor)

        self.assertEqual(rollback_calls, ["stage8_input_starless"])
        self.assertIn("stage8_limited_candidate", saved_stems)
        self.assertEqual(processor._stage8_final_source, "stage8_input_starless")
        self.assertEqual(
            processor._stage8_final_quality,
            "limited_candidate_rejected",
        )
        self.assertTrue(processor._stage8_handoff["passthrough"])
        self.assertEqual(
            processor._stage8_handoff["outcome_reason_code"],
            "stage8_limited_candidate_rejected",
        )
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "safe_passthrough",
        )
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])

    def test_pyqt6_headless_stub_includes_sasp_stage8_widget_imports(self):
        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "PyQt6" or name.startswith("PyQt6.")
        }
        try:
            processor = self._new_processor()
            installed = pipeline_module.SeestarPostProcessor._install_pyqt6_headless_stub(
                processor
            )
            self.assertTrue(installed)
            from PyQt6.QtGui import QDoubleValidator, QIntValidator, QPainter
            from PyQt6.QtCore import QCoreApplication, QPoint, QUrl, pyqtProperty
            from PyQt6.QtQuickWidgets import QQuickWidget

            self.assertIsNotNone(QIntValidator)
            self.assertIsNotNone(QDoubleValidator)
            self.assertIsNotNone(QPainter)
            self.assertIsNotNone(QCoreApplication)
            self.assertIsNotNone(QPoint)
            self.assertIsNotNone(QUrl)
            self.assertIsNotNone(pyqtProperty)
            self.assertIsNotNone(QQuickWidget)
        finally:
            for name in list(sys.modules):
                if name == "PyQt6" or name.startswith("PyQt6."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)

    def test_sasp_stage8_widget_shim_avoids_widget_package_side_effects(self):
        if not hasattr(pipeline_module.np, "array"):
            self.skipTest("real numpy is not available in this test interpreter")
        wheel_dir = REPO_ROOT / "resources" / "siril_plugins" / "downloads"
        wheels = sorted(wheel_dir.glob("setiastrosuitepro-*.whl"))
        if not wheels:
            self.skipTest("setiastrosuitepro wheel not bundled")

        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "setiastro.saspro.widgets"
            or name.startswith("setiastro.saspro.widgets.")
        }
        try:
            processor = self._new_processor()
            pipeline_module.SeestarPostProcessor._install_pyqt6_headless_stub(processor)
            pipeline_module.SeestarPostProcessor._install_sasp_stage8_widget_import_shims(
                processor,
                wheels[-1],
            )

            self.assertIn("setiastro.saspro.widgets.wavelet_utils", sys.modules)
            widgets_pkg = sys.modules.get("setiastro.saspro.widgets")
            self.assertIsNotNone(widgets_pkg)
            self.assertFalse(getattr(widgets_pkg, "__file__", None))
            wavelet = sys.modules["setiastro.saspro.widgets.wavelet_utils"]
            self.assertTrue(hasattr(wavelet, "atrous_decompose"))
        finally:
            for name in list(sys.modules):
                if name == "setiastro.saspro.widgets" or name.startswith(
                    "setiastro.saspro.widgets."
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)

    def test_stage8_skips_invalid_sasp_siril_commands_when_plugin_probe_enabled(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["曲线/蒙版工具2"] = "SASP CreateMask"
        processor.command_labels["细节/结构增强"] = "SASP Texture and Clarity"

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SASP Siril 深加工命令不可用", message)
        self.assertIn("内置 Starless", message)
        self.assertNotIn("曲线/蒙版工具2", processor.command_chain_calls)
        self.assertNotIn("细节/结构增强", processor.command_chain_calls)

    def test_stage8_low_dynamic_starless_builds_nonzero_faint_nebula_mask(self):
        thresholds = pipeline_module.stage8_pixels.stage8_low_signal_thresholds(
            bg_median=0.01993,
            bg_std=0.00005,
            p90=0.02010,
            p99=0.02115,
        )

        self.assertTrue(thresholds["low_signal"])
        self.assertLessEqual(thresholds["nebula_floor"], 0.0010)
        self.assertLess(thresholds["faint_floor"], 0.008)
        self.assertLess(thresholds["std_floor"], 0.01)

    def test_stage9_uses_previous_stage_starless_for_star_remix(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        stage9_star_remixing(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [("starless_enhanced", "starmask_stretched", processor.cfg.star_intensity)],
        )
        self.assertIn("previous_stage_star_remix source=starless_enhanced", message)
        self.assertIn("starmask=starmask_stretched", message)
        self.assertIn(("asinh", "2.000", "0.00100"), processor.cmd_calls)
        self.assertIn(("save", "starmask_stretched"), processor.cmd_calls)
        pm_calls = [call for call in processor.cmd_calls if call[0] == "pm"]
        self.assertFalse(pm_calls)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertTrue(report["stars_required"])
        self.assertTrue(report["stars_applied"])
        self.assertEqual(report["stars_application_mode"], "screen")

    def test_stage9_caps_star_remix_when_stage8_used_fallback(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.star_intensity = 1.05
        processor.cfg.star_fallback_intensity = 1.05
        processor._stage8_final_source = "stage8_input_starless"
        processor._stage8_handoff = {
            "schema": "seestar.stage8-handoff.v1",
            "source_stem": "stage8_input_starless",
            "passthrough": True,
            "restricted_downstream": True,
            "reason_code": "bright_nebula_halo_advisory",
        }
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        stage9_star_remixing(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [("stage8_input_starless", "starmask_stretched", 0.95)],
        )
        self.assertIn("Stage8 restricted source active", message)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertTrue(report["upstream_passthrough"])
        self.assertFalse(report["stage9_fallback_used"])
        self.assertIsNone(report["stage9_fallback_reason"])
        metadata = processor.result_metadata[-1]
        self.assertTrue(metadata["upstream_passthrough"])
        self.assertFalse(metadata["fallback_used"])
        self.assertEqual(metadata["reason_code"], "upstream_safe_passthrough")
        self.assertEqual(metadata["details"]["reason_text"], "使用 Stage 8 安全旁路源")

    def test_stage9_uses_plugin_stretched_starmask_without_builtin_asinh(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["星点拉伸"] = "SASP Star Stretch"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        stage9_star_remixing(processor)

        self.assertEqual(
            processor.previous_stage_remix_calls,
            [("starless_enhanced", "starmask_stretched", processor.cfg.star_intensity)],
        )
        asinh_calls = [call for call in processor.cmd_calls if call[0] == "asinh"]
        self.assertFalse(asinh_calls)
        self.assertIn(("save", "starmask_stretched"), processor.cmd_calls)

    def test_stage9_gate_rolls_back_primary_and_accepts_fallback_intensity(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        review_calls = []
        processor._create_stage_review_bundle = (
            lambda *args, **kwargs: (
                review_calls.append((args, kwargs))
                or {
                    "status": "ready",
                    "report_path": str(
                        processor.process_dir
                        / "review_bundles"
                        / "stage9_star_remixing"
                        / "review.json"
                    ),
                }
            )
        )

        def assess(_source_stem, *, attempt, formula):
            accepted = attempt == "screen_fallback_075"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else ["background_lift 0.020000>0.010000"],
                "metrics": {"background_lift": 0.0 if accepted else 0.02},
            }

        processor._stage9_assess_current_remix = assess

        stage9_star_remixing(processor)

        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                ("starless_enhanced", "starmask_stretched", processor.cfg.star_intensity),
                ("starless_enhanced", "starmask_stretched", 0.75),
            ],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_fallback_075")
        self.assertTrue(report["stage9_fallback_used"])
        self.assertEqual(report["stage9_fallback_reason"], "intensity_fallback")
        self.assertEqual(len(review_calls), 1)
        review_args, review_kwargs = review_calls[0]
        self.assertEqual(
            review_args,
            ("stage9_star_remixing", "starless_enhanced", "stage9_remixed"),
        )
        self.assertEqual(
            review_kwargs["selected_candidate"],
            "screen_fallback_075",
        )
        self.assertEqual(review_kwargs["context"]["mode"], "screen")
        self.assertEqual(len(review_kwargs["candidates"]), 2)
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "intensity_fallback",
        )

    def test_stage9_local_fallback_classifier_excludes_upstream_passthrough(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        cases = (
            ("screen", {"attempt": "primary"}, "screen", False, None),
            (
                "screen",
                {"attempt": "screen_fallback_075"},
                "screen",
                True,
                "intensity_fallback",
            ),
            (
                "screen",
                {"attempt": "screen_compact_recovery"},
                "screen",
                True,
                "compact_mask_recovery",
            ),
            (
                "unsafe_starless_bypass",
                None,
                "unsafe_starless_bypass",
                True,
                "unsafe_starless_bypass",
            ),
            (
                "rejected_keep_starless",
                None,
                "rejected_keep_starless",
                True,
                "all_remix_candidates_rejected",
            ),
            (
                "starmask_stretch_failed",
                None,
                "starmask_stretch_failed",
                True,
                "starmask_stretch_failed_keep_upstream",
            ),
            ("no_starmask", None, "no_starmask", False, None),
            (
                "star_preserve_target_bypass",
                None,
                "not_required_star_preserve",
                False,
                None,
            ),
        )
        for mode, selected, application_mode, expected_used, expected_reason in cases:
            with self.subTest(mode=mode, selected=selected):
                used, reason = stage9_module._stage9_local_fallback(
                    mode,
                    selected,
                    application_mode,
                )
                self.assertEqual(used, expected_used)
                self.assertEqual(reason, expected_reason)

    def test_stage9_review_bundle_marks_safe_rollback_as_selected(self):
        processor = self._new_processor()
        review_calls = []
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = False
        processor._stage9_stars_application_mode = "rejected_keep_starless"
        processor._stage9_star_reference_summary = {
            "status": "rejected",
            "reason": "source_star_catalog_contamination_risk",
        }
        processor._create_stage_review_bundle = (
            lambda *args, **kwargs: (
                review_calls.append((args, kwargs))
                or {"status": "ready", "report_path": "/tmp/stage9-review.json"}
            )
        )
        messages = []
        rejected_attempt = {
            "attempt": "screen_primary",
            "status": "rejected",
            "accepted": False,
            "issues": ["new_hollow_structure_max_area 1250>64"],
        }

        stage9_module = sys.modules["stages.stage9_star_remixing"]
        stage9_module._append_stage9_review_bundle(
            processor,
            messages,
            [rejected_attempt],
            None,
            source_stem="starless_enhanced",
            mode="rejected_keep_starless",
            stage_saved=True,
        )

        self.assertEqual(len(review_calls), 1)
        _args, kwargs = review_calls[0]
        self.assertEqual(kwargs["selected_candidate"], "stage9_safe_rollback")
        self.assertEqual(kwargs["candidates"][0], rejected_attempt)
        self.assertEqual(kwargs["candidates"][-1]["id"], "stage9_safe_rollback")
        self.assertTrue(kwargs["candidates"][-1]["selected"])
        self.assertIn("review_bundle=/tmp/stage9-review.json", messages)

    def test_stage9_rebuilds_strict_compact_mask_before_lowering_intensity(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        yy, xx = np.mgrid[:128, :128]
        diffuse_floor = 0.00004 + 0.00003 * np.sin(xx / 6.0) ** 2
        pixels = np.stack(
            [diffuse_floor, diffuse_floor * 0.9, diffuse_floor * 0.8]
        ).astype(np.float32)
        for cy, cx, amplitude in (
            (22, 24, 0.05),
            (41, 92, 0.08),
            (73, 61, 0.12),
            (99, 31, 0.06),
            (96, 105, 0.18),
        ):
            profile = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.4**2))
            pixels += np.stack(
                [profile * amplitude, profile * amplitude * 0.85, profile * amplitude * 0.70]
            ).astype(np.float32)
        written_pixels = []
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=lambda output: written_pixels.append(output.copy()),
        )

        def assess(_source_stem, *, attempt, formula):
            accepted = attempt == "screen_compact_recovery"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else [
                    "background_mottling_growth 2.000000>1.350000"
                ],
                "metrics": {"changed_pixel_ratio": 0.25 if not accepted else 0.03},
                "limits": {
                    "changed_pixel_ratio": 0.35,
                    "background_mottling_low_absolute_changed_pixel_ratio_max": 0.12,
                },
            }

        processor._stage9_assess_current_remix = assess

        stage9_star_remixing(processor)

        self.assertEqual(len(written_pixels), 2)
        normal_coverage = float(np.mean(np.max(written_pixels[0], axis=0) > 0.0))
        recovery_coverage = float(np.mean(np.max(written_pixels[1], axis=0) > 0.0))
        self.assertLess(recovery_coverage, normal_coverage)
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                ("starless_enhanced", "starmask_stretched", processor.cfg.star_intensity),
                (
                    "starless_enhanced",
                    "starmask_stretched_recovery",
                    processor.cfg.star_intensity,
                ),
            ],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_compact_recovery")
        self.assertTrue(report["stage9_fallback_used"])
        self.assertEqual(report["stage9_fallback_reason"], "compact_mask_recovery")
        self.assertTrue(report["starmask_calibration"]["recovery_attempted"])
        self.assertTrue(report["starmask_calibration"]["recovery_applied"])
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage9_compact_starmask_write_claims_image_lock(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        pixels = np.full((3, 8, 8), 0.01, dtype=np.float32)
        writes = []
        lock_events = []

        class ImageLock:
            def __enter__(self):
                lock_events.append("enter")

            def __exit__(self, _type, _value, _traceback):
                lock_events.append("exit")

        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=lambda output: writes.append(output.copy()),
            image_lock=lambda: ImageLock(),
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        calibration = {
            "status": "ok",
            "stretch": 12.0,
            "offset": 0.001,
            "star_sample_count": 64,
            "compact_component_count": 4,
            "_compact_support_mask": np.ones((8, 8), dtype=bool),
        }

        with patch.object(
            stage9_module.stage9_quality,
            "calibrate_starmask_asinh",
            return_value=calibration,
        ):
            stage9_star_remixing(processor)

        self.assertEqual(lock_events, ["enter", "exit"])
        self.assertEqual(len(writes), 1)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertFalse(report["starmask_stretch_failed"])
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage9_starmask_pixel_write_failure_stops_raw_linear_remix(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        pixels = np.full((3, 8, 8), 0.01, dtype=np.float32)

        def fail_pixel_write(_output):
            raise pipeline_module.SirilError("processing thread is not claimed")

        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=fail_pixel_write,
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        calibration = {
            "status": "ok",
            "stretch": 12.0,
            "offset": 0.001,
            "star_sample_count": 64,
            "compact_component_count": 4,
            "_compact_support_mask": np.ones((8, 8), dtype=bool),
        }

        with patch.object(
            stage9_module.stage9_quality,
            "calibrate_starmask_asinh",
            return_value=calibration,
        ):
            stage9_star_remixing(processor)

        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertTrue(processor._stage9_starmask_stretch_failed)
        self.assertFalse(processor._stage9_stars_applied)
        self.assertEqual(processor.results[-1][1], "degraded")
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertTrue(report["starmask_stretch_failed"])
        self.assertEqual(report["mode"], "starmask_stretch_failed")

    def test_stage9_dynamic_collapse_advisory_still_remixes_after_accepted_stretch(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage7_residual_star_score_max = 0.28
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_starless_noise_gain_max = 2.2
        processor.cfg.stage7_starless_dynamic_range_min_ratio = 0.55
        processor.cfg.stage7_starless_peak_signal_min = 0.006
        processor.cfg.stage9_starmask_stretch_enabled = False
        processor.cfg.stage9_fallback_intensity_levels = (0.75, 0.55, 0.40)
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor._stage7_selected_quality = {
            "status": "poor",
            "issues": [
                "starless_dynamic_range_collapse 0.117<0.550, "
                "peak=0.00551<0.00600"
            ],
            "derived": {
                "residual_star_score": 0.0,
                "halo_residue_score": 0.053,
                "starless_noise_gain": 0.78,
                "starless_dynamic_range_ratio": 0.117,
                "starless_peak_signal": 0.00551,
            },
        }
        processor._stage8_final_source = "stage7_stretched"
        processor._stage8_fallback_used = True
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage7_effective_halo_threshold = lambda: (
            processor.cfg.stage7_halo_residue_score_max
        )

        stage9_star_remixing(processor)

        self.assertTrue(processor.previous_stage_remix_calls)
        self.assertEqual(
            processor.stage_json_reports["stage9_remix_quality.json"]["mode"],
            "screen",
        )
        self.assertIn(
            "accepted-stretch advisory",
            processor.results[-1][3],
        )

    def test_stage9_bypasses_remix_and_degrades_when_stage7_was_not_accepted(self):
        processor = self._new_processor()
        processor._stage7_stretch_accepted = False
        processor._stage7_stretch_output = None
        processor.stretched_name = "stage7_stretched"
        (processor.process_dir / "stage7_cand_a.fit").write_bytes(b"review-only")
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage9_review_safe_source = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage9_review_safe_source,
            processor,
        )

        stage9_star_remixing(processor)

        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertIn(("load", "stage7_cand_a"), processor.cmd_calls)
        self.assertTrue(processor._stage9_bypassed_bad_starless)
        self.assertTrue(processor._stage9_stars_required)
        self.assertFalse(processor._stage9_stars_applied)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertFalse(report["stars_applied"])
        self.assertEqual(report["stars_application_mode"], "unsafe_starless_bypass")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIn("stage7_stretch_not_accepted", processor.results[-1][3])

    def test_stage9_review_bypass_prefers_stage7_quality_ranked_source(self):
        processor = self._new_processor()
        processor._stage7_stretch_accepted = False
        processor._stage7_stretch_output = None
        processor._stage7_review_source = "stage7_cand_rescue_2"
        processor.stretched_name = "stage7_stretched"
        (processor.process_dir / "stage7_cand_a.fit").write_bytes(b"candidate-a")
        (processor.process_dir / "stage7_cand_rescue_2.fit").write_bytes(
            b"quality-ranked"
        )
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage9_review_safe_source = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage9_review_safe_source,
            processor,
        )

        stage9_star_remixing(processor)

        self.assertIn(("load", "stage7_cand_rescue_2"), processor.cmd_calls)
        self.assertNotIn(("load", "stage7_cand_a"), processor.cmd_calls)
        self.assertTrue(processor._stage9_bypassed_bad_starless)

    def test_stage9_no_starmask_records_required_stars_not_applied(self):
        processor = self._new_processor()

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "no_starmask")
        self.assertTrue(report["stars_required"])
        self.assertFalse(report["stars_applied"])
        self.assertEqual(report["stars_application_mode"], "no_starmask")

    def test_stage9_star_preserve_bypass_records_stars_not_required(self):
        processor = self._new_processor()
        processor._star_preserve_target_bypass = True
        processor._stage8_final_source = "stage8_enhanced"

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "star_preserve_target_bypass")
        self.assertFalse(report["stars_required"])
        self.assertFalse(report["stars_applied"])
        self.assertEqual(
            report["stars_application_mode"],
            "not_required_star_preserve",
        )
        self.assertEqual(processor.results[-1][1], "skipped")

    def test_stage9_all_rejected_records_required_stars_not_applied(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage9_assess_current_remix = lambda *_args, **_kwargs: {
            "attempt": "rejected",
            "formula": "screen",
            "status": "rejected",
            "accepted": False,
            "issues": ["mock rejection"],
            "metrics": {},
        }

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "rejected_keep_starless")
        self.assertTrue(report["stars_required"])
        self.assertFalse(report["stars_applied"])
        self.assertEqual(
            report["stars_application_mode"],
            "rejected_keep_starless",
        )

    def test_stage9_accepted_screen_save_failure_does_not_claim_stars_applied(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._save_stage_output = lambda stem: stem != "stage9_remixed"

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "screen")
        self.assertTrue(report["stars_required"])
        self.assertFalse(report["stars_applied"])
        self.assertEqual(report["stars_application_mode"], "screen_save_failed")
        self.assertEqual(processor.results[-1][1], "degraded")

    def test_stage7_dynamic_range_gate_accepts_peak_well_above_extreme_background(self):
        cfg = pipeline_module.PipelineConfig()

        assessment = stage7_quality_module.stage7_dynamic_range_assessment(
            cfg,
            dynamic_range_ratio=0.09246493544624469,
            peak_signal=0.0051499465480446815,
            background_level=0.000503482879139483,
        )

        self.assertFalse(assessment["collapsed"])
        self.assertGreater(assessment["peak_background_ratio"], 10.0)

    def test_stage7_dynamic_range_gate_rejects_flat_low_signal_output(self):
        cfg = pipeline_module.PipelineConfig()

        assessment = stage7_quality_module.stage7_dynamic_range_assessment(
            cfg,
            dynamic_range_ratio=0.09,
            peak_signal=0.001,
            background_level=0.0005,
        )

        self.assertTrue(assessment["collapsed"])
        self.assertEqual(assessment["peak_background_ratio"], 2.0)

    def test_stage7_dynamic_collapse_skips_same_model_parameter_retries(self):
        without_axiom = SimpleNamespace(
            _syqon_axiom_model_available=lambda: False,
        )
        with_axiom = SimpleNamespace(
            _syqon_axiom_model_available=lambda: True,
        )

        self.assertEqual(
            stage7_star_separation_module._syqon_refinement_variants(
                without_axiom,
                repair_triggers=["dynamic_range_collapse"],
                initial_variant=(512, 64, False),
            ),
            [],
        )
        self.assertEqual(
            stage7_star_separation_module._syqon_refinement_variants(
                with_axiom,
                repair_triggers=["dynamic_range_collapse"],
                initial_variant=(512, 64, False),
            ),
            [(512, 64, True)],
        )

    def test_stage7_calibrated_dynamic_range_does_not_trigger_refinement(self):
        processor = pipeline_module.SeestarPostProcessor()
        quality = {
            "derived": {
                "residual_star_score": 0.0,
                "halo_residue_score": 0.03,
                "black_hole_score": 0.0,
                "starless_dynamic_range_ratio": 0.09246493544624469,
                "starless_peak_signal": 0.0051499465480446815,
                "dynamic_range_collapse": False,
            }
        }

        self.assertEqual(processor._stage7_repair_triggers(quality), [])

    def test_stage9_hard_halo_risk_still_bypasses_after_accepted_stretch(self):
        processor = self._new_processor()
        processor.cfg.stage7_residual_star_score_max = 0.28
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_starless_noise_gain_max = 2.2
        processor.cfg.stage7_starless_dynamic_range_min_ratio = 0.55
        processor.cfg.stage7_starless_peak_signal_min = 0.006
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor._stage7_selected_quality = {
            "status": "poor",
            "issues": [
                "starless_dynamic_range_collapse 0.117<0.550, "
                "peak=0.00551<0.00600"
            ],
            "derived": {
                "residual_star_score": 0.0,
                "halo_residue_score": 0.60,
                "starless_noise_gain": 0.78,
                "starless_dynamic_range_ratio": 0.117,
                "starless_peak_signal": 0.00551,
            },
        }
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.SeestarPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage7_effective_halo_threshold = lambda: (
            processor.cfg.stage7_halo_residue_score_max
        )

        reason = processor._stage9_bad_starless_reason()

        self.assertIn("stage7_halo_residue_score", reason)
        self.assertNotIn("stage7_starless_dynamic_range", reason)

    def test_stage9_uses_descending_fallback_intensity_ladder(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.star_intensity = 1.05
        processor.cfg.star_fallback_intensity = 1.0
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        def assess(_source_stem, *, attempt, formula):
            accepted = attempt == "screen_fallback_040"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else ["changed_pixel_ratio"],
                "metrics": {},
            }

        processor._stage9_assess_current_remix = assess

        stage9_star_remixing(processor)

        self.assertEqual(
            [call[2] for call in processor.previous_stage_remix_calls],
            [1.05, 0.75, 0.55, 0.40],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_fallback_040")

    def test_stage9_does_not_lower_screen_intensity_after_recovery_shortfall(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage9_assess_current_remix = lambda *_args, **kwargs: {
            "attempt": kwargs["attempt"],
            "formula": kwargs["formula"],
            "status": "rejected",
            "accepted": False,
            "issues": ["weak_star_recovery_ratio 0.420000<0.700000"],
            "metrics": {
                "weak_star_recovery_ratio": 0.42,
                "star_recovery_ratio": 0.50,
            },
        }

        stage9_star_remixing(processor)

        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                (
                    "starless_enhanced",
                    "starmask_stretched",
                    processor.cfg.star_intensity,
                )
            ],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "rejected_keep_starless")
        self.assertFalse(report["stars_applied"])
        self.assertEqual(processor.results[-1][1], "degraded")

    def test_stage10_degraded_when_script_and_aberration_unavailable(self):
        processor = self._new_processor()
        processor.cfg.aberration_api_enabled = True
        processor.aberration_errors["最终降噪"] = "import failed: No module named 'PyQt6'"
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_fail_steps.add("最终降噪")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertTrue(message.strip())
        self.assertIn("最终降噪脚本与 Aberration API 均不可用", message)
        self.assertIn("Aberration API 不可用", message)

    def test_stage10_skips_scunet_after_primary_timeout(self):
        processor = self._new_processor()
        processor.classic_cc_args = None
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/CosmicClarity_Native.py",
                "processing/SCUNet_Denoise.py",
            }
        )
        processor.cli_fail_steps.add("最终降噪")
        processor.cli_failure_errors["最终降噪"] = (
            "CosmicClarity_Native.py: subprocess timeout after 180s"
        )

        stage10_export(processor)

        scunet_calls = [
            call for call in processor.script_calls if call[1] == "SCUNet_Denoise.py"
        ]
        self.assertFalse(scunet_calls)
        self.assertIn("skipped after primary denoiser timeout", processor.results[-1][3])

    def test_stage10_uses_linear_resume_output_suffixes(self):
        processor = self._new_processor()
        processor._stage1_input_mode = "linear_resume"

        stage10_export(processor)

        linear_base = "result_processed_linear"
        self.assertIn(("savetif", linear_base, "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", linear_base), processor.cmd_calls)
        self.assertIn(("save", "result_final_linear"), processor.cmd_calls)
        self.assertEqual(processor.main_output_basename_template, linear_base)

    def test_stage10_withholds_normal_names_when_quality_requires_rerun(self):
        processor = self._new_processor()
        processor._final_quality_report = lambda _stem: {
            "final_quality": "poor",
            "status": "needs_conservative_rerun",
            "needs_conservative_rerun": True,
            "issues": ["background_chroma_noise_score 0.431>0.340"],
        }

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", "result_review"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertFalse(
            any(
                "result_processed" in str(item) or "result_final" in str(item)
                for call in processor.cmd_calls
                for item in call
            )
        )
        self.assertTrue(processor._final_output_review_only)
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIn("review_only_output=true", processor.results[-1][3])

    def test_stage10_stage9_bypass_uses_linear_review_only_names(self):
        processor = self._new_processor()
        processor._stage1_input_mode = "linear_resume"
        processor._stage9_bypassed_bad_starless = True
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")

        stage10_export(processor)

        self.assertIn(
            ("savetif", "result_review_linear", "-astro"),
            processor.cmd_calls,
        )
        self.assertIn(("savepng", "result_review_linear"), processor.cmd_calls)
        self.assertIn(
            ("save", "result_review_linear_final"),
            processor.cmd_calls,
        )
        self.assertTrue(processor._final_output_review_only)
        self.assertFalse(
            any(step == "最终降噪" for step, _name, _args in processor.script_calls)
        )
        denoise_plan = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertTrue(denoise_plan["skipped_by_review_only"])
        self.assertFalse(denoise_plan["skipped_by_duplicate_guard"])

    def test_stage10_missing_required_star_remix_forces_review_only_output(self):
        processor = self._new_processor()
        processor._stage9_bypassed_bad_starless = False
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = False
        processor._stage9_stars_application_mode = "no_starmask"

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertIn("stage9_stars_applied=false", processor.results[-1][3])

    def test_stage10_explicit_review_only_setting_withholds_normal_names(self):
        processor = self._new_processor()
        processor.cfg.force_review_only_output = True
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", "result_review"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertFalse(
            any(
                "result_processed" in str(item) or "result_final" in str(item)
                for call in processor.cmd_calls
                for item in call
            )
        )
        self.assertIn("force_review_only_output=true", processor.results[-1][3])

    def test_stage10_applied_required_star_remix_keeps_normal_output_names(self):
        processor = self._new_processor()
        processor._stage9_bypassed_bad_starless = False
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_stars_application_mode = "screen"

        stage10_export(processor)

        self.assertIn(
            ("savetif", "result_processed", "-astro"),
            processor.cmd_calls,
        )
        self.assertIn(("save", "result_final"), processor.cmd_calls)
        self.assertFalse(processor._final_output_review_only)

    def test_result_output_basename_uses_template_when_headers_are_complete(self):
        processor = self._new_processor()
        processor.header_metadata.update(
            {
                "OBJECT": "NGC7000",
                "STACKCNT": 120,
                "EXPTIME": 10.0,
                "DATE-OBS": "2026-07-15T12:00:00",
            }
        )

        base_name = processor._result_output_basename()

        self.assertEqual(base_name, pipeline_module.RESULT_BASENAME_TEMPLATE)
        self.assertEqual(
            processor.main_output_fit_basename_template,
            pipeline_module.RESULT_BASENAME_TEMPLATE + "_final",
        )

    def test_result_output_basename_uses_partial_metadata_when_stack_count_missing(self):
        processor = self._new_processor()
        processor.header_metadata.update(
            {
                "OBJECT": "M 42",
                "EXPTIME": 60.0,
                "DATE-OBS": "2026-02-16T14:02:34.608000",
            }
        )

        base_name = processor._result_output_basename()

        self.assertEqual(base_name, "M_42_60sec_20260216_140234_processed")
        self.assertEqual(
            processor.main_output_fit_basename_template,
            "M_42_60sec_20260216_140234_processed_final",
        )
        self.assertNotIn("$", base_name)
        self.assertTrue(
            any(
                "通用结果名覆盖" in message
                for level, message in processor.log.events
                if level == "warn"
            )
        )

    def test_result_output_basename_keeps_generic_fallback_without_identity(self):
        processor = self._new_processor()

        base_name = processor._result_output_basename()

        self.assertEqual(base_name, "result_processed")
        self.assertEqual(processor.main_output_fit_basename_template, "result_final")

    def test_stage10_saves_fits_before_preview_autostretch_png(self):
        processor = self._new_processor()

        stage10_export(processor)

        commands = [call[0] for call in processor.cmd_calls]
        self.assertLess(commands.index("save"), commands.index("autostretch"))
        self.assertLess(commands.index("autostretch"), commands.index("savepng"))
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("PNG preview stretch applied", message)

    def test_stage10_skips_second_preview_stretch_for_accepted_stage7_output(self):
        processor = self._new_processor()
        processor._stage7_stretch_accepted = True

        stage10_export(processor)

        commands = [call[0] for call in processor.cmd_calls]
        self.assertNotIn("autostretch", commands)
        self.assertIn("savepng", commands)
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("second autostretch skipped", message)

    def test_stage6_allows_sasp_fallback_when_probe_disabled(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.command_labels["去星"] = "SASP Dark Star"

        used = processor._run_first_available_command(
            "去星",
            [("SASP Dark Star", ("sasp_dark_star",))],
            allow_when_probe_disabled=True,
        )

        self.assertEqual(used, "SASP Dark Star")
        self.assertEqual(processor.workflow_command_used.get("去星"), "SASP Dark Star")

    def test_cli_subprocess_failure_records_output_tail(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)
        script_path = Path(td.name) / "SyQon-Starless.py"
        script_path.write_text("# mock\n", encoding="utf-8")

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            return SimpleNamespace(
                returncode=1,
                stdout="\n".join(f"line{i}" for i in range(1, 16)) + "\n",
            )

        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )
        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "去星",
                    "SyQon Starless",
                    script_path,
                )

        self.assertIsNone(used)
        self.assertIn("output_tail=", processor._last_plugin_script_error or "")
        self.assertIn("line4", processor._last_plugin_script_error or "")
        self.assertIn("line15", processor._last_plugin_script_error or "")
        self.assertNotIn("output_tail=line1", processor._last_plugin_script_error or "")

    def test_cleanup_preserves_raw_clean_and_stretched_starmask_layers(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.debug_mode = False
        processor.stretched_name = "stage7_stretched"

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        keep = {
            "starless.fit",
            "starmask.fit",
            "starmask_raw.fit",
            "starmask_clean.fit",
            "starmask_external_raw.fit",
            "starmask_stretched.fit",
            "stage8_limited_candidate.fit",
        }
        for name in keep | {"temporary.fit"}:
            (processor.process_dir / name).write_bytes(name.encode("utf-8"))

        processor.cleanup()

        self.assertTrue(all((processor.process_dir / name).exists() for name in keep))
        self.assertFalse((processor.process_dir / "temporary.fit").exists())

    def test_review_route_artifacts_are_available_as_stage_previews(self):
        processor = pipeline_module.SeestarPostProcessor()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        expected = {
            6: "stage6_passthrough.fit",
            7: "stage7_review_with_stars.fit",
            8: "stage8_review_with_stars.fit",
        }
        for stage, filename in expected.items():
            self.assertIn(
                processor.process_dir / filename,
                processor._stage_preview_candidates(stage),
            )

    def test_project_env_allowlist_includes_acceleration_and_quality_gates(self):
        allowed = sys.modules["processor_runtime"].PROJECT_ENV_ALLOWED_KEYS
        self.assertIn("SEESTAR_GRAXPERT_GPU", allowed)
        self.assertIn(
            "SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_REDUCTION_MIN",
            allowed,
        )
        self.assertIn(
            "SEESTAR_STAGE7_STARLESS_REPAIR_CHROMA_DELTA_MIN",
            allowed,
        )
        self.assertIn(
            "SEESTAR_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX",
            allowed,
        )
        self.assertIn("SEESTAR_STAGE8_LIMITED_SATURATION_MAX", allowed)
        self.assertIn(
            "SEESTAR_STAGE8_LIMITED_HALO_TEXTURE_GROWTH_MAX",
            allowed,
        )
        self.assertIn(
            "SEESTAR_STAGE8_LIMITED_HALO_TEXTURE_DELTA_MAX",
            allowed,
        )

    def test_stage6_halo_threshold_is_target_aware_for_diffuse_emission(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60

        processor._active_target_type = lambda: "galaxy"
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.35)
        processor._active_target_type = lambda: "emission_nebula_widefield"
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.45)
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.60)

    def test_stage6_bright_nebula_advisory_halo_triggers_pixel_repair(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        quality = {
            "status": "ok",
            "derived": {
                "halo_residue_score": 0.488,
                "compact_halo_residue_score": 0.493,
            },
        }

        trigger = (
            stage7_star_separation_module._stage7_starless_pixel_repair_trigger(
                processor,
                quality,
            )
        )

        self.assertTrue(trigger["triggered"])
        self.assertEqual(trigger["reason"], "bright_nebula_halo_advisory")
        self.assertTrue(trigger["within_target_limit"])
        self.assertAlmostEqual(trigger["measured_halo_score"], 0.493)

    def test_stage6_stage8_handoff_uses_three_level_bright_nebula_gate(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"

        cases = (
            (0.3500, 0.3400, "full"),
            (0.4880, 0.4930, "limited"),
            (0.6000, 0.5900, "limited"),
            (0.6001, 0.5900, "skip"),
            (0.3000, 0.6001, "skip"),
        )
        for global_halo, compact_halo, expected_policy in cases:
            with self.subTest(
                global_halo=global_halo,
                compact_halo=compact_halo,
            ):
                quality = {
                    "status": "ok",
                    "derived": {
                        "halo_residue_score": global_halo,
                        "compact_halo_residue_score": compact_halo,
                        "residual_star_score": 0.10,
                        "starless_noise_gain": 1.0,
                    },
                }
                handoff = stage7_star_separation_module._stage8_handoff_from_stage6(
                    processor,
                    quality,
                    [],
                    separation_accepted=True,
                )
                self.assertEqual(handoff["processing_policy"], expected_policy)

        m42_handoff = stage7_star_separation_module._stage8_handoff_from_stage6(
            processor,
            {
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.488,
                    "compact_halo_residue_score": 0.493,
                    "residual_star_score": 0.10,
                    "starless_noise_gain": 1.0,
                },
            },
            [],
            separation_accepted=True,
        )
        self.assertEqual(
            m42_handoff["reason_text"],
            "bright_nebula_halo_advisory: 0.488 > 0.350, accepted_limit=0.600",
        )
        self.assertEqual(m42_handoff["reason_code"], "bright_nebula_halo_advisory")

        repaired_handoff = stage7_star_separation_module._stage8_handoff_from_stage6(
            processor,
            {
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.445,
                    "compact_halo_residue_score": 0.445,
                    "residual_star_score": 0.056,
                    "starless_noise_gain": 0.625,
                },
            },
            [
                {
                    "accepted": True,
                    "acceptance_path": "residual_or_halo",
                    "trigger": {
                        "reason": "bright_nebula_halo_advisory",
                        "halo_residue_score": 0.488,
                        "compact_halo_residue_score": 0.493,
                    },
                }
            ],
            separation_accepted=True,
        )
        self.assertEqual(
            repaired_handoff["reason_text"],
            "bright_nebula_halo_advisory: 0.488 > 0.350, accepted_limit=0.600",
        )
        self.assertAlmostEqual(
            repaired_handoff["metrics"]["halo_residue_score"],
            0.445,
        )
        self.assertAlmostEqual(
            repaired_handoff["metrics"]["trigger_effective_halo_residue_score"],
            0.493,
        )
        self.assertAlmostEqual(
            m42_handoff["metrics"]["effective_halo_residue_score"],
            0.493,
        )

    def test_stage6_safe_bright_nebula_halo_does_not_trigger_pixel_repair(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        quality = {
            "status": "ok",
            "derived": {
                "halo_residue_score": 0.32,
                "compact_halo_residue_score": 0.34,
            },
        }

        trigger = (
            stage7_star_separation_module._stage7_starless_pixel_repair_trigger(
                processor,
                quality,
            )
        )

        self.assertFalse(trigger["triggered"])
        self.assertEqual(trigger["reason"], "")

    def test_stage6_poor_quality_still_triggers_pixel_repair(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor._active_target_type = lambda: "galaxy"
        quality = {
            "status": "poor",
            "derived": {
                "halo_residue_score": 0.10,
                "compact_halo_residue_score": 0.08,
            },
        }

        trigger = (
            stage7_star_separation_module._stage7_starless_pixel_repair_trigger(
                processor,
                quality,
            )
        )

        self.assertTrue(trigger["triggered"])
        self.assertEqual(trigger["reason"], "quality_status=poor")

    def test_stage6_compact_halo_triggers_refinement_and_candidate_penalty(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"

        safe = {
            "derived": {
                "residual_star_score": 0.03,
                "halo_residue_score": 0.49,
                "compact_halo_residue_score": 0.58,
                "black_hole_score": 0.0,
                "starmask_contamination": 0.0,
                "starless_noise_gain": 1.0,
                "starless_dynamic_range_ratio": 1.0,
                "starless_peak_signal": 1.0,
                "starmask_coverage_ratio": 1.0,
                "starmask_width_ratio": 1.0,
            }
        }
        compact_halo = {
            "derived": {
                **safe["derived"],
                "compact_halo_residue_score": 0.89,
            }
        }

        self.assertEqual(processor._stage7_repair_triggers(safe), [])
        self.assertIn(
            "compact_halo_residue",
            processor._stage7_repair_triggers(compact_halo),
        )
        self.assertGreater(
            processor._stage7_quality_score(compact_halo),
            processor._stage7_quality_score(safe),
        )

    def test_stage6_compact_halo_measurement_stays_local_in_dense_star_field(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor._active_target_type = lambda: "galaxy"
        height = width = 256
        yy, xx = np.mgrid[:height, :width]
        background = 0.015 + xx.astype(np.float32) / width * 0.018
        source_gray = background.copy()
        starmask_gray = np.zeros_like(background)
        for cy in range(12, height, 20):
            for cx in range(12, width, 20):
                radius2 = (yy - cy) ** 2 + (xx - cx) ** 2
                star = 0.22 * np.exp(-radius2 / 5.0)
                source_gray += star
                starmask_gray += star
        source = np.repeat(source_gray[None, :, :], 3, axis=0)
        starless = np.repeat(background[None, :, :], 3, axis=0)
        starmask = np.repeat(starmask_gray[None, :, :], 3, axis=0)

        scores = pipeline_module.stage7_quality.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )

        self.assertLess(scores["compact_halo_mask_coverage"], 0.50)
        self.assertLess(scores["compact_halo_residue_score"], 0.60)

    def test_cleanup_archives_lightweight_diagnostics_before_deleting_logs(self):
        import zipfile

        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.debug_mode = False

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.work_dir = Path(td.name)
        processor.process_dir = processor.work_dir / "process"
        review_dir = processor.process_dir / "review_bundles" / "stage5"
        review_dir.mkdir(parents=True)
        (processor.process_dir / "stage5_linear_report.json").write_text(
            '{"status":"ok"}', encoding="utf-8"
        )
        (processor.process_dir / "stage.log").write_text("diagnostic", encoding="utf-8")
        (review_dir / "preview.png").write_bytes(b"png")
        (processor.process_dir / "large.fit").write_bytes(b"fits")

        processor.cleanup()

        archive_path = processor.work_dir / "seestar_diagnostics.zip"
        self.assertTrue(archive_path.exists())
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
        self.assertIn("manifest.json", names)
        self.assertIn("process/stage5_linear_report.json", names)
        self.assertIn("process/review_bundles/stage5/preview.png", names)
        self.assertNotIn("process/large.fit", names)

    def test_plugin_fingerprint_uses_preview_and_bounded_sample(self):
        preview_calls: list[bool] = []
        image = np.zeros((3, 512, 512), dtype=np.float32)
        processor = SimpleNamespace(
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: (
                    preview_calls.append(preview) or image
                )
            ),
            log=FakeLogger(),
        )

        before = pipeline_module.plugin_runner.current_image_fingerprint(processor)
        image[:] = 1.0
        after = pipeline_module.plugin_runner.current_image_fingerprint(processor)

        self.assertEqual(preview_calls, [True, True])
        self.assertNotEqual(before, after)

    def test_runtime_env_applies_adaptive_star_and_target_local_controls(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        overrides = {
            "SEESTAR_STAGE9_STARMASK_ADAPTIVE_STRETCH_ENABLE": "0",
            "SEESTAR_STAGE9_SOURCE_STAR_DETAIL_PERCENTILE": "98.2",
            "SEESTAR_STAGE9_SOURCE_COMPONENT_DENSITY_MAX": "2300",
            "SEESTAR_STAGE9_SOURCE_SINGLE_PIXEL_RATIO_MAX": "0.36",
            "SEESTAR_STAGE9_STARMASK_ASINH_STRETCH_MAX": "640",
            "SEESTAR_STAGE9_STARMASK_FAINT_TARGET": "0.19",
            "SEESTAR_STAGE9_STARMASK_MID_TARGET": "0.48",
            "SEESTAR_STAGE9_STARMASK_BRIGHT_TARGET": "0.73",
            "SEESTAR_STAGE9_STARMASK_PEAK_TARGET": "0.79",
            "SEESTAR_STAGE9_STARMASK_CHROMA_REGULARIZATION_ENABLE": "0",
            "SEESTAR_STAGE9_STARMASK_FAINT_CHROMA_MAX": "0.32",
            "SEESTAR_STAGE9_STARMASK_BRIGHT_CHROMA_MAX": "0.58",
            "SEESTAR_STAGE9_STARMASK_PREDICTED_CHANGE_RATIO_MAX": "0.27",
            "SEESTAR_STAGE9_STAR_REFERENCE_SIGMA": "5.5",
            "SEESTAR_STAGE9_COMPACT_WEAK_STAR_RETENTION_MIN": "0.83",
            "SEESTAR_STAGE9_MIXED_STAR_PEAK_RATIO_MIN": "4.5",
            "SEESTAR_STAGE9_MIXED_STAR_WEAK_COUNT_MIN": "24",
            "SEESTAR_STAGE9_MIXED_STAR_BRIGHT_COUNT_MIN": "4",
            "SEESTAR_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN": "0.72",
            "SEESTAR_STAGE9_STAR_RECOVERY_RATIO_MIN": "0.78",
            "SEESTAR_STAGE9_WEAK_STAR_SCREEN_INTENSITY_MIN": "0.96",
            "SEESTAR_STAGE9_STAR_SUPPORT_RATIO_MAX": "0.11",
            "SEESTAR_STAGE9_UNMATCHED_CHANGED_RATIO_MAX": "0.008",
            "SEESTAR_STAGE9_CHROMATIC_ADDITION_PEAK_MIN": "0.025",
            "SEESTAR_STAGE9_CHROMATIC_ADDITION_SATURATION_MIN": "0.74",
            "SEESTAR_STAGE9_CHROMATIC_ADDITION_RATIO_MAX": "0.0025",
            "SEESTAR_STAGE9_STAR_APERTURE_RECOVERY_RATIO_MIN": "0.76",
            "SEESTAR_STAGE9_STAR_WING_RECOVERY_RATIO_MIN": "0.66",
            "SEESTAR_STAGE9_RESIDUAL_DARK_HOLE_RATIO_MAX": "0.14",
            "SEESTAR_STAGE9_HOLLOW_STRUCTURE_DELTA_MIN": "0.06",
            "SEESTAR_STAGE9_NEW_HOLLOW_STRUCTURE_AREA_MAX": "48",
            "SEESTAR_STAGE9_LOCAL_COMPONENT_PEAK_MIN": "0.012",
            "SEESTAR_STAGE9_LOCAL_COMPONENT_AREA_MAX": "320",
            "SEESTAR_STAGE9_LOCAL_COMPONENT_ASPECT_RATIO_MAX": "4.0",
            "SEESTAR_STAGE9_LOCAL_COMPONENT_FILL_RATIO_MIN": "0.12",
            "SEESTAR_STAGE9_LOCAL_SINGLE_PIXEL_RATIO_MAX": "0.18",
            "SEESTAR_STAGE9_LOCAL_CYAN_BLUE_PEAK_MIN": "0.015",
            "SEESTAR_STAGE9_LOCAL_CYAN_BLUE_SATURATION_MIN": "0.55",
            "SEESTAR_STAGE9_LOCAL_CYAN_BLUE_COMPONENT_AREA_MAX": "72",
            "SEESTAR_STAGE9_CORE_PERCENTILE": "92",
            "SEESTAR_STAGE9_CORE_COLOR_JUMP_MIN": "0.11",
            "SEESTAR_STAGE9_CORE_COLOR_JUMP_COMPONENT_AREA_MAX": "80",
            "SEESTAR_STAGE10_STAGE9_LOCAL_COLOR_RISK_STRENGTH": "0.8",
            "SEESTAR_FORCE_REVIEW_ONLY_OUTPUT": "1",
            "SEESTAR_STAGE7_TARGET_LOCAL_METRICS_ENABLE": "0",
            "SEESTAR_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX": "0.08",
            "SEESTAR_STAGE7_LOCAL_FAINT_SNR_MIN": "0.31",
            "SEESTAR_STAGE7_LOCAL_DARK_SEPARATION_MIN": "0.002",
            "SEESTAR_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_MAX": "0.045",
            "SEESTAR_STAGE7_PREVIEW_TARGET_P50_MAX_RATIO": "1.42",
            "SEESTAR_STAGE7_STARLESS_PEAK_BACKGROUND_RATIO_MIN": "5.0",
            "SEESTAR_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX": "0.07",
        }

        with patch.dict(os.environ, overrides, clear=False):
            processor._apply_runtime_env_overrides()

        self.assertFalse(processor.cfg.stage9_starmask_adaptive_stretch_enabled)
        self.assertEqual(processor.cfg.stage9_source_star_detail_percentile, 98.2)
        self.assertEqual(processor.cfg.stage9_source_component_density_max, 2300.0)
        self.assertEqual(processor.cfg.stage9_source_single_pixel_ratio_max, 0.36)
        self.assertEqual(processor.cfg.stage9_starmask_asinh_stretch_max, 640.0)
        self.assertEqual(processor.cfg.stage9_starmask_faint_target, 0.19)
        self.assertEqual(processor.cfg.stage9_starmask_mid_target, 0.48)
        self.assertEqual(processor.cfg.stage9_starmask_bright_target, 0.73)
        self.assertEqual(processor.cfg.stage9_starmask_peak_target, 0.79)
        self.assertFalse(processor.cfg.stage9_starmask_chroma_regularization_enabled)
        self.assertEqual(processor.cfg.stage9_starmask_faint_chroma_max, 0.32)
        self.assertEqual(processor.cfg.stage9_starmask_bright_chroma_max, 0.58)
        self.assertEqual(
            processor.cfg.stage9_starmask_predicted_change_ratio_max,
            0.27,
        )
        self.assertEqual(processor.cfg.stage9_star_reference_sigma, 5.5)
        self.assertEqual(
            processor.cfg.stage9_compact_weak_star_retention_min,
            0.83,
        )
        self.assertEqual(processor.cfg.stage9_mixed_star_peak_ratio_min, 4.5)
        self.assertEqual(processor.cfg.stage9_mixed_star_weak_count_min, 24)
        self.assertEqual(processor.cfg.stage9_mixed_star_bright_count_min, 4)
        self.assertEqual(processor.cfg.stage9_weak_star_recovery_ratio_min, 0.72)
        self.assertEqual(processor.cfg.stage9_star_recovery_ratio_min, 0.78)
        self.assertEqual(processor.cfg.stage9_weak_star_screen_intensity_min, 0.96)
        self.assertEqual(processor.cfg.stage9_star_support_ratio_max, 0.11)
        self.assertEqual(processor.cfg.stage9_unmatched_changed_ratio_max, 0.008)
        self.assertEqual(processor.cfg.stage9_chromatic_addition_peak_min, 0.025)
        self.assertEqual(
            processor.cfg.stage9_chromatic_addition_saturation_min,
            0.74,
        )
        self.assertEqual(processor.cfg.stage9_chromatic_addition_ratio_max, 0.0025)
        self.assertEqual(
            processor.cfg.stage9_star_aperture_recovery_ratio_min,
            0.76,
        )
        self.assertEqual(processor.cfg.stage9_star_wing_recovery_ratio_min, 0.66)
        self.assertEqual(processor.cfg.stage9_residual_dark_hole_ratio_max, 0.14)
        self.assertEqual(processor.cfg.stage9_hollow_structure_delta_min, 0.06)
        self.assertEqual(processor.cfg.stage9_new_hollow_structure_area_max, 48.0)
        self.assertEqual(processor.cfg.stage9_local_component_peak_min, 0.012)
        self.assertEqual(processor.cfg.stage9_local_component_area_max, 320.0)
        self.assertEqual(processor.cfg.stage9_local_component_aspect_ratio_max, 4.0)
        self.assertEqual(processor.cfg.stage9_local_component_fill_ratio_min, 0.12)
        self.assertEqual(processor.cfg.stage9_local_single_pixel_ratio_max, 0.18)
        self.assertEqual(processor.cfg.stage9_local_cyan_blue_peak_min, 0.015)
        self.assertEqual(processor.cfg.stage9_local_cyan_blue_saturation_min, 0.55)
        self.assertEqual(
            processor.cfg.stage9_local_cyan_blue_component_area_max,
            72.0,
        )
        self.assertEqual(processor.cfg.stage9_core_percentile, 92.0)
        self.assertEqual(processor.cfg.stage9_core_color_jump_min, 0.11)
        self.assertEqual(
            processor.cfg.stage9_core_color_jump_component_area_max,
            80.0,
        )
        self.assertEqual(
            processor.cfg.stage10_stage9_local_color_risk_strength,
            0.8,
        )
        self.assertTrue(processor.cfg.force_review_only_output)
        self.assertFalse(processor.cfg.stage7_target_local_metrics_enabled)
        self.assertEqual(processor.cfg.stage7_local_core_clip_ratio_max, 0.08)
        self.assertEqual(processor.cfg.stage7_local_faint_snr_min, 0.31)
        self.assertEqual(processor.cfg.stage7_local_dark_separation_min, 0.002)
        self.assertEqual(
            processor.cfg.stage7_stretch_chroma_load_low_absolute_max,
            0.045,
        )
        self.assertEqual(
            processor.cfg.stage7_preview_target_p50_max_ratio,
            1.42,
        )
        self.assertEqual(
            processor.cfg.stage7_starmask_diffuse_residual_ratio_max,
            0.07,
        )
        self.assertEqual(
            processor.cfg.stage7_starless_peak_background_ratio_min,
            5.0,
        )

    def test_runtime_retires_external_mild_prestretch_controls(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        overrides = {
            "SEESTAR_STAR_SEPARATION_MODE": "mild_prestretch_star_separation",
            "SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH": "1",
            "SEESTAR_MILD_PRESTRETCH_STRENGTH": "1.70",
        }

        with patch.dict(os.environ, overrides, clear=False):
            processor._apply_runtime_env_overrides()

        self.assertFalse(hasattr(processor.cfg, "star_separation_mode"))
        self.assertFalse(
            hasattr(processor.cfg, "star_separation_fallback_to_mild_prestretch")
        )
        self.assertFalse(hasattr(processor.cfg, "mild_prestretch_strength"))
        warnings = [
            message for level, message in processor.log.events if level == "warn"
        ]
        self.assertTrue(
            any(
                "SEESTAR_STAR_SEPARATION_MODE is retired" in message
                for message in warnings
            )
        )
        self.assertTrue(
            any(
                "SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH is retired"
                in message
                for message in warnings
            )
        )
        self.assertTrue(
            any(
                "SEESTAR_MILD_PRESTRETCH_STRENGTH is retired" in message
                for message in warnings
            )
        )

    def test_cosmic_clarity_wrapper_uses_stable_python_when_siril_env_is_boolean(self):
        wrapper = REPO_ROOT / "resources" / "siril_plugins" / "bin" / "CosmicClarity"

        proc = subprocess.run(
            [str(wrapper), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "SIRIL_PYTHON_CLI": "1",
                "SEESTAR_SIRIL_PYTHON_CLI": sys.executable,
            },
        )

        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("Bundled Cosmic Clarity classic wrapper", proc.stdout)

    def test_legacy_cosmic_clarity_wrapper_is_rejected(self):
        processor = pipeline_module.SeestarPostProcessor()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        plugin_root = Path(td.name) / "siril_plugins"
        bin_dir = plugin_root / "bin"
        bin_dir.mkdir(parents=True)
        wrapper = bin_dir / "CosmicClarity"
        wrapper.write_text(
            "#!/bin/sh\n"
            '"exec" "${SIRIL_PYTHON_CLI:-python3}" "$0" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        processor.siril_plugin_dir = plugin_root

        reason = processor._classic_cosmic_clarity_candidate_error(wrapper)

        self.assertIsNotNone(reason)
        self.assertIn("boolean SIRIL_PYTHON_CLI", reason or "")

    def test_stage6_syqon_variant_collects_script_outputs(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/SyQon-Starless.py")
        processor.syqon_output_mode = "both"
        processor._clear_star_separation_outputs = (  # type: ignore[method-assign]
            lambda: pipeline_module.syqon_starless.clear_star_separation_outputs(processor)
        )
        processor._stage7_prepare_starmask = lambda: None  # type: ignore[method-assign]
        script = processor._find_plugin_script(("processing/SyQon-Starless.py",))
        self.assertIsNotNone(script)

        used = pipeline_module.syqon_starless.stage7_try_syqon_variant(
            processor,
            script,
            attempt_name="initial",
            tile_size=512,
            overlap=64,
            axiom=False,
        )

        self.assertIsNotNone(used)
        self.assertIn("SyQon Starless initial", processor.workflow_command_used["去星"])
        syqon_calls = [
            args
            for step, script_name, args in processor.script_calls
            if step == "去星" and script_name == "SyQon-Starless.py"
        ]
        self.assertTrue(syqon_calls)
        self.assertIn("--tile-size", syqon_calls[0])
        self.assertIn("--overlap", syqon_calls[0])
        self.assertNotIn("--no_gpu", syqon_calls[0])
        self.assertNotIn("--axiom", syqon_calls[0])
        self.assertTrue((processor.process_dir / "starless.fit").exists())
        self.assertTrue((processor.process_dir / "starmask.fit").exists())

    def test_stage7_retry_cleanup_preserves_all_best_snapshot_layers(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        retained = {
            "starless_ai_best_syqon_refine_1.fit",
            "starmask_ai_best_syqon_refine_1.fit",
            "starmask_raw_ai_best_syqon_refine_1.fit",
        }
        removable = {
            "starless.fit",
            "starmask.fit",
            "starmask_raw.fit",
        }
        for name in retained | removable:
            (processor.process_dir / name).write_bytes(name.encode("utf-8"))

        processor._clear_star_separation_outputs()

        self.assertTrue(all((processor.process_dir / name).exists() for name in retained))
        self.assertTrue(all(not (processor.process_dir / name).exists() for name in removable))

    def test_stage7_restore_rejects_incomplete_snapshot_without_partial_restore(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        current_starless = processor.process_dir / "starless.fit"
        current_starmask = processor.process_dir / "starmask.fit"
        current_raw = processor.process_dir / "starmask_raw.fit"
        current_starless.write_bytes(b"current-starless")
        current_starmask.write_bytes(b"current-starmask")
        current_raw.write_bytes(b"current-raw")
        (processor.process_dir / "starless_best.fit").write_bytes(b"best-starless")
        (processor.process_dir / "starmask_best.fit").write_bytes(b"best-starmask")
        processor.starless_file = current_starless
        processor.starmask_file = current_starmask

        with self.assertRaisesRegex(FileNotFoundError, "Stage7 snapshot is incomplete"):
            processor._stage7_restore_snapshot(
                {
                    "starless": "starless_best",
                    "starmask": "starmask_best",
                    "starmask_raw": "starmask_raw_missing",
                    "starmask_kind": "raw",
                }
            )

        self.assertEqual(current_starless.read_bytes(), b"current-starless")
        self.assertEqual(current_starmask.read_bytes(), b"current-starmask")
        self.assertEqual(current_raw.read_bytes(), b"current-raw")
        self.assertEqual(processor.starless_file, current_starless)
        self.assertEqual(processor.starmask_file, current_starmask)

    def test_stage7_restore_replaces_all_layers_from_one_snapshot(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        current_starless = processor.process_dir / "starless.fit"
        current_starmask = processor.process_dir / "starmask.fit"
        current_raw = processor.process_dir / "starmask_raw.fit"
        for path in (current_starless, current_starmask, current_raw):
            path.write_bytes(b"retry-2")
        (processor.process_dir / "starless_best.fit").write_bytes(b"retry-1-starless")
        (processor.process_dir / "starmask_best.fit").write_bytes(b"retry-1-starmask")
        (processor.process_dir / "starmask_raw_best.fit").write_bytes(b"retry-1-raw")

        processor._stage7_restore_snapshot(
            {
                "starless": "starless_best",
                "starmask": "starmask_best",
                "starmask_raw": "starmask_raw_best",
                "starmask_kind": "raw",
            }
        )

        self.assertEqual(current_starless.read_bytes(), b"retry-1-starless")
        self.assertEqual(current_starmask.read_bytes(), b"retry-1-starmask")
        self.assertEqual(current_raw.read_bytes(), b"retry-1-raw")
        self.assertEqual(processor.starless_file, current_starless)
        self.assertEqual(processor.starmask_file, current_starmask)

    def test_stage6_can_force_syqon_cpu_with_env(self):
        processor = self._new_processor()

        with patch.dict(os.environ, {pipeline_module.ENV_SYQON_GPU_KEY: "0"}, clear=False):
            args, _timeout, _note = processor._syqon_starless_cli_options(
                tile_size=512,
                overlap=64,
                axiom=False,
            )

        self.assertIn("--no_gpu", args)

    def test_syqon_axiom21_cli_matches_bundled_script_interface(self):
        processor = self._new_processor()

        args, _timeout, note = pipeline_module.syqon_starless.syqon_starless_cli_options(
            processor,
            tile_size=512,
            overlap=64,
            axiom=True,
        )

        self.assertIn("--axiom21", args)
        self.assertNotIn("--axiom", args)
        self.assertIn("Axiom 2.1", note)

    def test_syqon_axiom21_model_probe_matches_script_paths(self):
        processor = self._new_processor()
        processor.siril_plugin_dir = processor.work_dir / "siril_plugins"
        bundled_model = (
            processor.siril_plugin_dir
            / "vendor"
            / "siril-scripts"
            / "Axiom2_1.pt"
        )
        bundled_model.parent.mkdir(parents=True, exist_ok=True)
        bundled_model.write_bytes(b"mock-model")

        available = pipeline_module.syqon_starless.syqon_axiom_model_available(processor)

        self.assertTrue(available)

    def test_run_linear_resume_without_evidence_uses_review_only_route(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        linear_path = work_dir / pipeline_module.LINEAR_RESUME_INPUT_NAME
        linear_path.write_bytes(b"linear-fit")

        calls: list[str] = []

        processor.connect = lambda: setattr(processor, "work_dir", work_dir)  # type: ignore[method-assign]
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: None,
        )

        def _prepare_linear_resume() -> None:
            calls.append("prepare_linear_resume")
            processor._stage1_input_mode = "linear_resume"
            processor.linear_intermediate_path = linear_path

        processor._prepare_linear_resume_input = _prepare_linear_resume  # type: ignore[method-assign]
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")  # type: ignore[method-assign]
        processor.stage1_preparation = lambda: calls.append("stage1")  # type: ignore[method-assign]
        processor.stage2_view_correction = lambda: calls.append("stage2")  # type: ignore[method-assign]
        processor.stage3_background_extraction = lambda: calls.append("stage3")  # type: ignore[method-assign]
        processor.stage4_color_calibration = lambda: calls.append("stage4")  # type: ignore[method-assign]
        processor.stage5_linear_denoise = lambda: calls.append("stage5")  # type: ignore[method-assign]
        processor.stage6_star_separation = lambda: calls.append("stage6")  # type: ignore[method-assign]
        processor.stage7_stretching = lambda: calls.append("stage7")  # type: ignore[method-assign]
        processor.stage8_nebula_enhancement = lambda: calls.append("stage8")  # type: ignore[method-assign]
        processor.stage9_star_remixing = lambda: calls.append("stage9")  # type: ignore[method-assign]
        processor.stage10_export = lambda: calls.append("stage10")  # type: ignore[method-assign]
        processor.stage11_ai_postprocess = lambda: calls.append("stage11")  # type: ignore[method-assign]
        processor.cleanup = lambda: calls.append("cleanup")  # type: ignore[method-assign]

        with patch.dict(
            os.environ,
            {pipeline_module.ENV_INPUT_MODE_KEY: pipeline_module.INPUT_MODE_LINEAR_RESUME},
            clear=False,
        ):
            processor.run()

        self.assertEqual(
            calls,
            [
                "prepare_linear_resume",
                "auto_tune",
                "stage10",
                "cleanup",
            ],
        )
        skipped = {
            (result.name, result.status, result.message)
            for result in processor.results
            if result.status == "skipped"
        }
        self.assertIn(
            ("阶段 2: 裁切", "skipped", "skipped by linear resume mode"),
            skipped,
        )
        skipped_names = {name for name, _status, _message in skipped}
        self.assertTrue(
            {
                pipeline_module.PipelineStage.BACKGROUND_EXTRACTION.label,
                pipeline_module.PipelineStage.COLOR_CALIBRATION.label,
                pipeline_module.PipelineStage.LINEAR_DENOISE.label,
                pipeline_module.PipelineStage.STAR_SEPARATION.label,
                pipeline_module.PipelineStage.STRETCHING.label,
                pipeline_module.PipelineStage.NEBULA_ENHANCEMENT.label,
                pipeline_module.PipelineStage.STAR_REMIXING.label,
                pipeline_module.PipelineStage.AI_POSTPROCESS.label,
            }.issubset(skipped_names)
        )
        self.assertTrue(processor.cfg.force_review_only_output)
        self.assertEqual(processor.input_profile["state"], "unknown")
        plan = json.loads(
            (work_dir / "processing-plan.json").read_text(encoding="utf-8")
        )
        result_manifest = json.loads(
            (work_dir / "pipeline-result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(plan["input"]["profile"]["state"], "unknown")
        self.assertEqual(
            plan["planned_steps"][9]["action"],
            "review_export_only",
        )
        self.assertEqual(result_manifest["status"], "review_required")
        self.assertEqual(result_manifest["plan_hash"], plan["plan_hash"])

    def test_run_stage2_resume_without_evidence_uses_review_only_route(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        stage2_path = work_dir / pipeline_module.STAGE2_CORRECTED_INPUT_NAME
        stage2_path.write_bytes(b"stage2-fit")

        calls: list[str] = []

        processor.connect = lambda: setattr(processor, "work_dir", work_dir)  # type: ignore[method-assign]
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: None,
        )

        def _prepare_stage2_corrected_resume() -> None:
            calls.append("prepare_stage2_corrected_resume")
            processor._stage1_input_mode = "stage2_corrected_resume"
            processor.source_file = stage2_path

        processor._prepare_stage2_corrected_resume_input = _prepare_stage2_corrected_resume  # type: ignore[method-assign]
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")  # type: ignore[method-assign]
        processor.stage1_preparation = lambda: calls.append("stage1")  # type: ignore[method-assign]
        processor.stage2_view_correction = lambda: calls.append("stage2")  # type: ignore[method-assign]
        processor.stage3_background_extraction = lambda: calls.append("stage3")  # type: ignore[method-assign]
        processor.stage4_color_calibration = lambda: calls.append("stage4")  # type: ignore[method-assign]
        processor.stage5_linear_denoise = lambda: calls.append("stage5")  # type: ignore[method-assign]
        processor.stage6_star_separation = lambda: calls.append("stage6")  # type: ignore[method-assign]
        processor.stage7_stretching = lambda: calls.append("stage7")  # type: ignore[method-assign]
        processor.stage8_nebula_enhancement = lambda: calls.append("stage8")  # type: ignore[method-assign]
        processor.stage9_star_remixing = lambda: calls.append("stage9")  # type: ignore[method-assign]
        processor.stage10_export = lambda: calls.append("stage10")  # type: ignore[method-assign]
        processor.stage11_ai_postprocess = lambda: calls.append("stage11")  # type: ignore[method-assign]
        processor.cleanup = lambda: calls.append("cleanup")  # type: ignore[method-assign]

        with patch.dict(
            os.environ,
            {pipeline_module.ENV_INPUT_MODE_KEY: pipeline_module.INPUT_MODE_STAGE2_CORRECTED_RESUME},
            clear=False,
        ):
            processor.run()

        self.assertEqual(
            calls,
            [
                "prepare_stage2_corrected_resume",
                "auto_tune",
                "stage10",
                "cleanup",
            ],
        )
        self.assertTrue(processor.cfg.force_review_only_output)
        self.assertEqual(processor.input_profile["state"], "unknown")
        result_manifest = json.loads(
            (work_dir / "pipeline-result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result_manifest["status"], "review_required")

    def test_run_stage4_psolved_resume_skips_stages_1_through_3(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        calls: list[str] = []

        def connect() -> None:
            processor.work_dir = work_dir
            processor._siril_ever_connected = True

        processor.connect = connect
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: calls.append("disconnect"),
        )

        def prepare_resume() -> None:
            calls.append("prepare_stage4_psolved_resume")
            processor._stage1_input_mode = "stage4_psolved_resume"

        processor._prepare_stage4_psolved_resume_input = prepare_resume
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")
        processor.stage1_preparation = lambda: calls.append("stage1")
        processor.stage2_view_correction = lambda: calls.append("stage2")
        processor.stage3_background_extraction = lambda: calls.append("stage3")
        processor.stage4_color_calibration = lambda: calls.append("stage4")
        processor.stage5_linear_denoise = lambda: calls.append("stage5")
        processor.stage6_star_separation = lambda: calls.append("stage6")
        processor.stage7_stretching = lambda: calls.append("stage7")
        processor.stage8_nebula_enhancement = lambda: calls.append("stage8")
        processor.stage9_star_remixing = lambda: calls.append("stage9")
        processor.stage10_export = lambda: calls.append("stage10")
        processor.stage11_ai_postprocess = lambda: calls.append("stage11")
        processor.cleanup = lambda: calls.append("cleanup")

        with patch.dict(
            os.environ,
            {
                pipeline_module.ENV_INPUT_MODE_KEY:
                pipeline_module.INPUT_MODE_STAGE4_PSOLVED_RESUME
            },
            clear=False,
        ):
            processor.run()

        self.assertEqual(
            calls,
            [
                "prepare_stage4_psolved_resume",
                "auto_tune",
                "stage4",
                "stage5",
                "stage6",
                "stage7",
                "stage8",
                "stage9",
                "stage10",
                "stage11",
                "cleanup",
                "disconnect",
            ],
        )
        skipped_names = {
            result.name for result in processor.results if result.status == "skipped"
        }
        self.assertTrue(
            {
                pipeline_module.PipelineStage.PREPARATION.label,
                pipeline_module.PipelineStage.VIEW_CORRECTION.label,
                pipeline_module.PipelineStage.BACKGROUND_EXTRACTION.label,
            }.issubset(skipped_names)
        )

    def test_prepare_stage4_psolved_resume_preserves_checkpoint_while_rebuilding_process_dir(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        process_dir = work_dir / "process"
        process_dir.mkdir()
        checkpoint = process_dir / pipeline_module.STAGE4_PSOLVED_INPUT_NAME
        checkpoint.write_bytes(b"stage4-psolved-checkpoint")
        calls: list[tuple[Any, ...]] = []
        processor.work_dir = work_dir
        processor.process_dir = process_dir
        processor.cmd_with_check = lambda *args, **_kwargs: calls.append(args) or True

        processor._prepare_stage4_psolved_resume_input()

        restored = process_dir / pipeline_module.STAGE4_PSOLVED_INPUT_NAME
        self.assertEqual(restored.read_bytes(), b"stage4-psolved-checkpoint")
        self.assertEqual(processor.source_file, restored)
        self.assertEqual(processor._stage1_input_mode, "stage4_psolved_resume")
        self.assertIn(("load", "stage4_psolved"), calls)

    def test_run_native_death_in_stage4_does_not_continue_stage5_or_cleanup(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        calls: list[str] = []

        processor.connect = lambda: setattr(processor, "work_dir", work_dir)
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: calls.append("disconnect"),
        )
        processor.stage1_preparation = lambda: calls.append("stage1")
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")
        processor.stage2_view_correction = lambda: calls.append("stage2")
        processor.stage3_background_extraction = lambda: calls.append("stage3")

        def fatal_stage4() -> None:
            calls.append("stage4")
            processor._siril_process_terminated = True
            raise pipeline_module.SirilNativeProcessTerminated(
                "spcc",
                pipeline_module.SirilConnectionError("connection closed"),
            )

        processor.stage4_color_calibration = fatal_stage4
        processor.stage5_linear_denoise = lambda: calls.append("stage5")
        processor.cleanup = lambda: calls.append("cleanup")

        with patch.dict(
            os.environ,
            {pipeline_module.ENV_INPUT_MODE_KEY: pipeline_module.INPUT_MODE_AUTO},
            clear=False,
        ):
            with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
                processor.run()

        self.assertEqual(calls, ["stage1", "auto_tune", "stage2", "stage3", "stage4"])

    def test_run_reraises_generic_stage_failure_for_siril_and_gui(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        calls: list[str] = []

        def connect() -> None:
            processor.work_dir = work_dir
            processor._siril_ever_connected = True

        processor.connect = connect
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: calls.append("disconnect"),
        )
        processor.stage1_preparation = lambda: calls.append("stage1")
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")

        def fail_stage2() -> None:
            calls.append("stage2")
            raise RuntimeError("simulated stage2 failure")

        processor.stage2_view_correction = fail_stage2
        processor.stage3_background_extraction = lambda: calls.append("stage3")

        with patch.dict(
            os.environ,
            {pipeline_module.ENV_INPUT_MODE_KEY: pipeline_module.INPUT_MODE_AUTO},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated stage2 failure"):
                processor.run()

        self.assertEqual(calls, ["stage1", "auto_tune", "stage2", "disconnect"])
        self.assertTrue(
            any(
                level == "error" and "程序中断: simulated stage2 failure" in message
                for level, message in processor.log.events
            )
        )

    def test_plugin_script_prereq_check_skips_runtime_execution_when_modules_missing(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        def _unexpected_cmd(*_args, **_kwargs):
            raise AssertionError("pyscript should be skipped before runtime call")

        processor.cmd_with_check = _unexpected_cmd  # type: ignore[method-assign]
        script_path = Path("/tmp/CosmicClarity_Denoise.py")

        used = processor._run_plugin_script_by_path(
            "最终降噪",
            "CosmicClarity Denoise",
            script_path,
            args=("-denoising_mode", "luminance"),
        )

        self.assertIsNone(used)
        self.assertIsNotNone(processor._last_plugin_script_error)
        self.assertIn("missing python modules", processor._last_plugin_script_error)

    def test_cli_subprocess_ignores_boolean_siril_python_cli_env(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "CosmicClarity_Denoise.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")

        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        captured_cmd: dict[str, list[str]] = {}
        captured_env: dict[str, str] = {}

        def _fake_run(cmd: list[str], **kwargs: Any):
            captured_cmd["value"] = list(cmd)
            proc_env = kwargs.get("env") or {}
            if isinstance(proc_env, dict):
                captured_env.update({str(k): str(v) for k, v in proc_env.items()})
            return SimpleNamespace(returncode=0, stdout="")

        contaminated_env = {
            "SIRIL_PYTHON_CLI": "1",
            "QT_PLUGIN_PATH": "/app/PySide6/Qt/plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/app/PySide6/Qt/plugins/platforms",
            "QML2_IMPORT_PATH": "/app/PySide6/Qt/qml",
            "QT_QPA_PLATFORM": "cocoa",
        }
        with patch.dict(os.environ, contaminated_env, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "最终降噪",
                    "CosmicClarity Denoise",
                    script_path,
                    args=("-denoising_mode", "luminance"),
                )

        self.assertIsNotNone(used)
        self.assertIn("value", captured_cmd)
        self.assertNotEqual(captured_cmd["value"][0], "1")
        self.assertIn(str(script_path), captured_cmd["value"])
        self.assertEqual(captured_env.get("SEESTAR_SIRILPY_TIMEOUT_SEC"), "120")
        self.assertNotEqual(captured_env.get("SIRIL_PYTHON_CLI"), "1")
        self.assertEqual(captured_env.get("QT_QPA_PLATFORM"), "offscreen")
        self.assertNotIn("QT_PLUGIN_PATH", captured_env)
        self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", captured_env)
        self.assertNotIn("QML2_IMPORT_PATH", captured_env)

    def test_cli_subprocess_releases_parent_siril_connection(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "SyQon-Starless.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")

        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        events: list[str] = []

        class _ConnectedSiril:
            connected = True

            def disconnect(self) -> None:
                events.append("disconnect")
                self.connected = False

            def connect(self) -> bool:
                events.append("connect")
                self.connected = True
                return True

        processor.siril = _ConnectedSiril()

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            events.append("run")
            return SimpleNamespace(returncode=0, stdout="")

        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "去星",
                    "SyQon Starless",
                    script_path,
                    args=("--tile-size", "512"),
                )

        self.assertIsNotNone(used)
        self.assertEqual(events, ["disconnect", "run", "connect"])
        self.assertTrue(processor.siril.connected)

    def test_cli_subprocess_heartbeat_stays_local_while_parent_siril_is_disconnected(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "SyQon-Starless.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")
        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        live_messages: list[str] = []
        connection_events: list[str] = []

        class SirilConnectionError(Exception):
            pass

        class _ConnectedSiril:
            connected = True

            def disconnect(self) -> None:
                connection_events.append("disconnect")
                self.connected = False

            def connect(self) -> bool:
                connection_events.append("connect")
                self.connected = True
                return True

            def log(self, line: str) -> None:
                if not self.connected:
                    raise SirilConnectionError(
                        "Error in _send_command(): [Errno 9] Bad file descriptor"
                    )
                live_messages.append(line)

        processor.siril = pipeline_module._FatalSirilInterfaceProxy(
            processor,
            _ConnectedSiril(),
        )
        processor._siril_ever_connected = True
        processor.log = pipeline_module.PipelineLogger("DEBUG")
        log_path = Path(td.name) / "pipeline.log"
        processor.log.set_file_path(log_path)
        processor.log.set_sink(processor.siril.log)

        class _ImmediateFirstWaitEvent:
            def __init__(self) -> None:
                self._first_wait = True
                self._stopped = False

            def wait(self, _timeout: float) -> bool:
                if self._stopped:
                    return True
                if self._first_wait:
                    self._first_wait = False
                    return False
                return True

            def set(self) -> None:
                self._stopped = True

        class _SynchronousThread:
            def __init__(self, *, target: Any, daemon: bool) -> None:
                _ = daemon
                self._target = target

            def start(self) -> None:
                self._target()

            def join(self, timeout: float | None = None) -> None:
                _ = timeout

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            return SimpleNamespace(returncode=0, stdout="")

        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(
                pipeline_module.plugin_runner.threading,
                "Event",
                _ImmediateFirstWaitEvent,
            ):
                with patch.object(
                    pipeline_module.plugin_runner.threading,
                    "Thread",
                    _SynchronousThread,
                ):
                    with patch.object(pipeline_module.subprocess, "run", _fake_run):
                        used = processor._run_plugin_script_cli_subprocess(
                            "去星",
                            "SyQon Starless",
                            script_path,
                            args=("--tile-size", "512"),
                        )

        self.assertIsNotNone(used)
        self.assertEqual(connection_events, ["disconnect", "connect"])
        self.assertFalse(processor._siril_process_terminated)
        self.assertIn("CLI 子进程仍在运行", log_path.read_text(encoding="utf-8"))
        self.assertFalse(any("CLI 子进程仍在运行" in line for line in live_messages))
        self.assertTrue(any("CLI 子进程成功" in line for line in live_messages))

    def test_cli_subprocess_timeout_applies_when_child_produces_no_output(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "sleeping_plugin.py"
        script_path.write_text(
            "import time\n"
            "time.sleep(2)\n",
            encoding="utf-8",
        )

        started = time.monotonic()
        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            used = processor._run_plugin_script_cli_subprocess(
                "测试脚本",
                "Sleeping Plugin",
                script_path,
                timeout_sec=1,
            )
        elapsed = time.monotonic() - started

        self.assertIsNone(used)
        self.assertLess(elapsed, 1.8)
        self.assertIn("subprocess timeout", processor._last_plugin_script_error or "")

    def test_final_denoise_cli_timeout_tracks_sirilpy_timeout_with_cap(self):
        processor = pipeline_module.SeestarPostProcessor()

        with patch.dict(os.environ, {"SEESTAR_SIRILPY_TIMEOUT_SEC": "120"}, clear=False):
            self.assertEqual(processor._final_denoise_cli_timeout_sec(), 180)
        with patch.dict(os.environ, {"SEESTAR_SIRILPY_TIMEOUT_SEC": "999"}, clear=False):
            self.assertEqual(processor._final_denoise_cli_timeout_sec(), 300)
        with patch.dict(os.environ, {"SEESTAR_SIRILPY_TIMEOUT_SEC": "bad"}, clear=False):
            self.assertEqual(processor._final_denoise_cli_timeout_sec(), 180)

    def test_stage_diff_note_detects_identical_and_changed_outputs(self):
        processor = pipeline_module.SeestarPostProcessor()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name) / "process"
        process_dir.mkdir(parents=True, exist_ok=True)
        processor.process_dir = process_dir

        previous_path = process_dir / "stage4_colorbalanced.fit"
        current_path = process_dir / "stage5_denoised.fit"
        payload = b"mock-fit-bytes"
        previous_path.write_bytes(payload)
        current_path.write_bytes(payload)

        same_note = processor._stage_diff_note("stage5_denoised", "stage4_colorbalanced")
        self.assertIsNotNone(same_note)
        self.assertIn("内容一致", same_note)

        current_path.write_bytes(payload + b"-changed")
        diff_note = processor._stage_diff_note("stage5_denoised", "stage4_colorbalanced")
        self.assertIsNotNone(diff_note)
        self.assertIn("内容有变化", diff_note)

    def test_stage_result_display_status_uses_structured_fields_only(self):
        message_only_result = pipeline_module.StageResult(
            "阶段 X",
            "ok",
            message="fallback: failed_component=A; fallback_component=B; fallback_status=success",
        )
        skipped_message_result = pipeline_module.StageResult(
            "阶段 Y",
            "ok",
            message="SPCC skipped on Light_ preprocess mode",
        )
        fallback_result = pipeline_module.StageResult(
            "阶段 A",
            "ok",
            fallback_used=True,
        )
        skipped_result = pipeline_module.StageResult(
            "阶段 B",
            "ok",
            execution="skipped",
        )
        passthrough_result = pipeline_module.StageResult(
            "阶段 C",
            "ok",
            execution="safe_passthrough",
        )
        degraded_result = pipeline_module.StageResult("阶段 Z", "degraded")

        self.assertEqual(message_only_result.display_status, "ok")
        self.assertEqual(skipped_message_result.display_status, "ok")
        self.assertEqual(fallback_result.display_status, "ok_with_fallback")
        self.assertEqual(skipped_result.display_status, "ok_skipped_optional")
        self.assertEqual(passthrough_result.display_status, "ok_safe_passthrough")
        self.assertEqual(degraded_result.display_status, "degraded")

    def test_ai_plan_text_fallback_extracts_candidate_id_without_numeric_params(self):
        processor = pipeline_module.SeestarPostProcessor()
        raw_text = (
            "Recommendation summary\n"
            "selected_candidate_id: conservative\n"
            "global_saturation_delta: 0.99\n"
        )

        parsed = processor._extract_first_json_object(raw_text)

        self.assertEqual(parsed["selected_candidate_id"], "conservative")
        self.assertNotIn("adjustments", parsed)

    def test_stage3_plugin_order_uses_theoretical_effect_chain(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()

        high_noise = pipeline_module.ImageFeatures(bg_std=0.060, star_density=0.001)
        high_noise_order = processor._stage3_plugin_candidates(
            high_noise,
            {"dirty_background_score": 0.42},
        )
        self.assertEqual(
            [label for label, _cmd, _source in high_noise_order],
            ["GraXpert", "GraXpert-BGE", "ADBE", "DBE", "AutoDBE", "NOX", "VeraLux NOX"],
        )

    def test_stage3_quality_gate_rejects_star_or_nebula_loss(self):
        processor = pipeline_module.SeestarPostProcessor()
        preservation = {
            "available": True,
            "star_retention_ratio": 0.82,
            "nebula_mean_change_ratio": 0.14,
            "before_star_count": 100,
            "after_star_count": 82,
        }
        gate_ok, gate_msg = processor._stage3_quality_gate(
            pipeline_module.ImageFeatures(bg_std=0.02, bg_median=0.08, object_area_ratio=0.20),
            pipeline_module.ImageFeatures(bg_std=0.02, bg_median=0.08, object_area_ratio=0.20),
            preservation,
        )

        self.assertFalse(gate_ok)
        self.assertIn("star retention ratio", gate_msg)
        self.assertIn("nebula mean change", gate_msg)

    def test_stage3_background_score_prefers_cleaner_low_gradient_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        before = {
            "bg_std": 0.00010,
            "gradient_score": 0.10,
            "dirty_background_score": 0.42,
            "red_dominance": 1.00,
            "blue_dominance": 1.00,
            "green_cast": 1.00,
        }
        dirty_candidate = {
            "bg_std": 0.00011,
            "gradient_score": 0.09,
            "dirty_background_score": 0.38,
            "chroma_noise_score": 0.12,
            "red_dominance": 1.32,
            "blue_dominance": 0.92,
            "green_cast": 1.18,
        }
        cleaner_candidate = {
            "bg_std": 0.00010,
            "gradient_score": 0.04,
            "dirty_background_score": 0.20,
            "chroma_noise_score": 0.05,
            "red_dominance": 1.02,
            "blue_dominance": 1.01,
            "green_cast": 0.99,
        }

        dirty_score = stage3_module._stage3_background_score(before, dirty_candidate)
        cleaner_score = stage3_module._stage3_background_score(before, cleaner_candidate)

        self.assertLess(cleaner_score, dirty_score)
        self.assertFalse(stage3_module._stage3_candidate_sufficient(before, dirty_candidate, dirty_score))
        self.assertTrue(stage3_module._stage3_candidate_sufficient(before, cleaner_candidate, cleaner_score))

    def test_stage3_candidate_sufficient_uses_policy_std_growth_limit(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        before = {
            "bg_std": 0.00010,
            "gradient_score": 0.08,
            "dirty_background_score": 0.28,
            "red_dominance": 1.00,
            "blue_dominance": 1.00,
            "green_cast": 1.00,
        }
        candidate = {
            "bg_std": 0.000107,
            "gradient_score": 0.02,
            "dirty_background_score": 0.12,
            "chroma_noise_score": 0.02,
            "red_dominance": 1.01,
            "blue_dominance": 1.00,
            "green_cast": 0.99,
        }

        self.assertTrue(stage3_module._stage3_candidate_sufficient(before, candidate, 0.12))
        self.assertFalse(
            stage3_module._stage3_candidate_sufficient(
                before,
                candidate,
                0.12,
                {"max_bg_std_growth": 1.03},
            )
        )

    def test_stage3_large_emission_nebula_prefers_poly_first(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.48},
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.18},
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "bright_emission_reflection_nebula",
                    "object_stats": {
                        "object_area_ratio": 0.16,
                        "nebulosity_area_ratio": 0.42,
                    },
                },
                {},
            )
        )
        self.assertFalse(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "large_galaxy",
                    "object_stats": {"object_area_ratio": 0.48},
                },
                {},
            )
        )

    def test_stage3_faint_nebula_signal_protects_generic_profile(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        profile = {
            "target_type": "generic_low_snr_safe",
            "object_stats": {
                "nebulosity_area_ratio": 0.12,
                "faint_structure_score": 0.45,
            },
        }
        protect, context = stage3_module._stage3_should_exhaust_builtin_search(
            profile,
            {},
            {},
        )

        self.assertTrue(stage3_module._stage3_prefers_poly_first(profile, {}))
        self.assertTrue(protect)
        self.assertTrue(context["faint_nebula_protection"])
        self.assertEqual(context["protection_reason"], "faint_nebula_signal")

    def test_stage3_faint_structure_increases_nebula_preservation_penalty(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        preservation = {
            "available": True,
            "nebula_mean_change_ratio": 0.08,
            "star_retention_ratio": 1.0,
        }
        base_penalty = stage3_module._stage3_preservation_penalty(
            preservation,
            diffuse_context={"faint_structure_score": 0.40},
        )
        strong_penalty = stage3_module._stage3_preservation_penalty(
            preservation,
            diffuse_context={
                "faint_nebula_protection": True,
                "faint_structure_score": 0.90,
            },
        )

        self.assertGreater(strong_penalty, base_penalty)
        self.assertLessEqual(
            stage3_module._stage3_nebula_preservation_weight(
                {"faint_nebula_protection": True, "faint_structure_score": 1.0}
            ),
            2.5,
        )

    def test_stage3_theoretical_chain_falls_back_until_candidate_is_sufficient(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(workflow_plugin_probe_enabled=False)
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: None)
                self.try_calls: list[tuple[str, ...]] = []
                self.cmd_calls: list[tuple[Any, ...]] = []
                self.saved: list[str] = []
                self.workflow_command_used: dict[str, str] = {}
                self.results: list[tuple[str, str, float, str]] = []
                self.report: dict[str, Any] = {}
                self.adaptive_measurements = [
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.12,
                        "dirty_background_score": 0.44,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.11,
                        "dirty_background_score": 0.39,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.10,
                        "dirty_background_score": 0.37,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return True

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

            def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
                return True, "quality gate ok"

            def _adaptive_features_current(self):
                return self.adaptive_measurements.pop(0)

            def _save_stage_output(self, stem: str) -> bool:
                self.saved.append(stem)
                return True

            def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
                self.report = payload

            def _record_stage(
                self,
                name: str,
                status: str,
                elapsed: float,
                message: str,
                **_metadata: Any,
            ) -> None:
                self.results.append((name, status, elapsed, message))

        processor = Stage3Fake()
        stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(
            background_attempts[:3],
            [("gxp",), ("graxpert",), ("adbe",)],
        )
        self.assertIn(("load", "stage3_candidate_adbe"), processor.cmd_calls)
        self.assertEqual(processor.workflow_command_used["背景提取插件链"], "ADBE")
        self.assertTrue(processor.report["graxpert_attempted"])
        self.assertFalse(processor.report["fallback_used"])
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage3_all_candidates_rejected_restores_baseline(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(gate_ok=False)

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [("rejected", ("subsky", "1"), "builtin")],
                ["rejected"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(processor.report["attempts"][0]["status"], "rejected")
        self.assertIn(
            {"context": "rejected:rejected", "status": "restored"},
            processor.report["rollback_events"],
        )

    def test_stage3_selected_candidate_load_failure_restores_baseline(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(
            gate_ok=True,
            fail_selected_load=True,
        )

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [("accepted", ("subsky", "1"), "builtin")],
                ["accepted"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIsNone(processor.report["model_used"])
        self.assertIn(
            {
                "context": "selected_load_failed:accepted",
                "status": "restored",
            },
            processor.report["rollback_events"],
        )

    def test_stage3_graxpert_runtime_error_triggers_background_fallback(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(workflow_plugin_probe_enabled=False)
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: None)
                self.cmd_calls: list[tuple[Any, ...]] = []
                self.saved: list[str] = []
                self.workflow_command_used: dict[str, str] = {}
                self.results: list[tuple[str, str, float, str]] = []
                self.report: dict[str, Any] = {}
                self.adaptive_measurements = [
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.12,
                        "dirty_background_score": 0.44,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                if args and args[0] in ("gxp", "graxpert"):
                    raise RuntimeError(
                        "GraXpert-AI.py Error: too many indices for array: "
                        "array is 2-dimensional, but 3 were indexed"
                    )
                return True

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

            def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
                return True, "quality gate ok"

            def _adaptive_features_current(self):
                return self.adaptive_measurements.pop(0)

            def _save_stage_output(self, stem: str) -> bool:
                self.saved.append(stem)
                return True

            def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
                self.report = payload

            def _record_stage(
                self,
                name: str,
                status: str,
                elapsed: float,
                message: str,
                **_metadata: Any,
            ) -> None:
                self.results.append((name, status, elapsed, message))

        processor = Stage3Fake()
        stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(background_attempts[:3], [("gxp",), ("graxpert",), ("adbe",)])
        self.assertEqual(processor.workflow_command_used["背景提取插件链"], "ADBE")
        self.assertTrue(processor.report["graxpert_runtime_error"])
        self.assertTrue(processor.report["fallback_triggered_by_graxpert_error"])
        self.assertTrue(processor.report["fallback_used"])
        self.assertEqual(
            processor.report["fallback_reason"],
            "graxpert_runtime_fallback",
        )
        self.assertEqual(
            [record["status"] for record in processor.report["attempts"][:2]],
            ["graxpert_runtime_error", "graxpert_runtime_error"],
        )

    def test_stage3_graxpert_success_without_image_change_is_runtime_error(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.fingerprints = iter(("unchanged", "unchanged"))
                self.cmd_calls: list[tuple[Any, ...]] = []

            def _validate_plugin_script_prerequisites(self, _script_path: Path):
                return True, ""

            def _current_image_fingerprint(self):
                return next(self.fingerprints)

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

        processor = Stage3Fake()
        command = (
            "pyscript",
            '"/mock/siril-scripts/processing/GraXpert-AI.py"',
            "-bge",
        )

        ok, reason = stage3_module._stage3_try_background_command(
            processor,
            "GraXpert",
            command,
            "graxpert",
        )

        self.assertFalse(ok)
        self.assertEqual(processor.cmd_calls, [command])
        self.assertEqual(
            reason,
            "graxpert_runtime_error: command returned success but image did not change",
        )

    def test_stage3_graxpert_missing_onnx_is_rejected_before_execution(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cmd_calls: list[tuple[Any, ...]] = []

            def _validate_plugin_script_prerequisites(self, script_path: Path):
                self.validated_script = script_path
                return False, "missing python modules: onnx"

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

        processor = Stage3Fake()
        command = (
            "pyscript",
            '"/mock/siril scripts/processing/GraXpert-AI.py"',
            "-bge",
        )

        ok, reason = stage3_module._stage3_try_background_command(
            processor,
            "GraXpert",
            command,
            "graxpert",
        )

        self.assertFalse(ok)
        self.assertFalse(processor.cmd_calls)
        self.assertEqual(
            processor.validated_script,
            Path("/mock/siril scripts/processing/GraXpert-AI.py"),
        )
        self.assertEqual(
            reason,
            "graxpert_runtime_error: prerequisites unavailable: "
            "missing python modules: onnx",
        )

    def test_stage3_autobge_success_without_image_change_is_rejected(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.fingerprints = iter(("unchanged", "unchanged"))
                self.cmd_calls: list[tuple[Any, ...]] = []

            def _validate_plugin_script_prerequisites(self, script_path: Path):
                self.validated_script = script_path
                return True, ""

            def _current_image_fingerprint(self):
                return next(self.fingerprints)

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

        processor = Stage3Fake()
        command = (
            "pyscript",
            '"/mock/siril scripts/processing/AutoBGE.py"',
        )

        ok, reason = stage3_module._stage3_try_background_command(
            processor,
            "AutoBGE",
            command,
            "plugin",
        )

        self.assertFalse(ok)
        self.assertEqual(
            processor.validated_script,
            Path("/mock/siril scripts/processing/AutoBGE.py"),
        )
        self.assertEqual(processor.cmd_calls, [command])
        self.assertEqual(
            reason,
            "plugin_runtime_error: command returned success but image did not change",
        )

    def test_autobge_prerequisites_check_cv2_import_name(self):
        modules = pipeline_module.SeestarPostProcessor._SCRIPT_PREREQUISITE_MODULES[
            "AutoBGE.py"
        ]

        self.assertIn("cv2", modules)
        self.assertNotIn("opencv-python", modules)

    def test_graxpert_prerequisites_include_script_import_names(self):
        modules = pipeline_module.SeestarPostProcessor._SCRIPT_PREREQUISITE_MODULES[
            "GraXpert-AI.py"
        ]

        self.assertIn("onnx", modules)
        self.assertIn("appdirs", modules)
        self.assertNotIn("platformdirs", modules)

    def test_stage3_stops_when_first_theoretical_candidate_is_sufficient(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(workflow_plugin_probe_enabled=False)
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: None)
                self.try_calls: list[tuple[str, ...]] = []
                self.cmd_calls: list[tuple[Any, ...]] = []
                self.saved: list[str] = []
                self.workflow_command_used: dict[str, str] = {}
                self.results: list[tuple[str, str, float, str]] = []
                self.report: dict[str, Any] = {}
                self.adaptive_measurements = [
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.12,
                        "dirty_background_score": 0.44,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return True

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

            def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
                return True, "quality gate ok"

            def _adaptive_features_current(self):
                return self.adaptive_measurements.pop(0)

            def _save_stage_output(self, stem: str) -> bool:
                self.saved.append(stem)
                return True

            def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
                self.report = payload

            def _record_stage(
                self,
                name: str,
                status: str,
                elapsed: float,
                message: str,
                **_metadata: Any,
            ) -> None:
                self.results.append((name, status, elapsed, message))

        processor = Stage3Fake()
        stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(background_attempts, [("gxp",)])
        self.assertIn(("load", "stage3_candidate_graxpert"), processor.cmd_calls)
        self.assertEqual(processor.workflow_command_used["GraXpert 背景提取"], "GraXpert")

    def test_stage3_large_emission_nebula_tries_poly_before_rbf(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(
                    workflow_plugin_probe_enabled=False,
                    stage3_diffuse_auto_apply_enabled=True,
                )
                self.target_profile = {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.46},
                }
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: None)
                self.try_calls: list[tuple[str, ...]] = []
                self.cmd_calls: list[tuple[Any, ...]] = []
                self.saved: list[str] = []
                self.workflow_command_used: dict[str, str] = {}
                self.results: list[tuple[str, str, float, str]] = []
                self.report: dict[str, Any] = {}
                self.adaptive_measurements = [
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.12,
                        "dirty_background_score": 0.44,
                        "object_area_ratio": 0.46,
                        "nebulosity_area_ratio": 0.42,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.11,
                        "dirty_background_score": 0.39,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.02,
                        "dirty_background_score": 0.14,
                        "chroma_noise_score": 0.03,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.02,
                        "dirty_background_score": 0.14,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                if args and args[0] not in ("save", "load", "subsky"):
                    raise RuntimeError(f"mock plugin unavailable: {args[0]}")
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return args and args[0] == "subsky"

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

            def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
                return True, "quality gate ok"

            def _adaptive_features_current(self):
                return self.adaptive_measurements.pop(0)

            def _save_stage_output(self, stem: str) -> bool:
                self.saved.append(stem)
                return True

            def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
                self.report = payload

            def _record_stage(
                self,
                name: str,
                status: str,
                elapsed: float,
                message: str,
                **_metadata: Any,
            ) -> None:
                self.results.append((name, status, elapsed, message))

        processor = Stage3Fake()
        stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertLess(
            background_attempts.index(("subsky", "1")),
            background_attempts.index(("subsky", "-rbf")),
        )
        self.assertIn(("load", "stage3_candidate_subsky_rbf_1"), processor.cmd_calls)
        self.assertEqual(processor.report["builtin_order_reason"], "diffuse_signal_subsky_poly_before_rbf")
        self.assertEqual(
            processor.report["builtin_search_mode"],
            "theoretical_effect_order_with_diffuse_signal_protection",
        )

    def test_stage3_decision_skips_clean_background(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.02,
                "dirty_background_score": 0.08,
                "chroma_noise_score": 0.01,
            },
            diffuse_context={"diffuse": False},
        )

        self.assertEqual(decision["decision"], "skip")

    def test_stage3_decision_requires_review_for_diffuse_signal(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={"target_type": "emission_nebula_widefield"},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.20,
                "dirty_background_score": 0.40,
                "chroma_noise_score": 0.08,
            },
            diffuse_context={
                "diffuse": True,
                "emission_diffuse": True,
            },
        )

        self.assertEqual(decision["decision"], "review_required")

    def test_stage3_decision_skips_low_dirty_gradient_when_diffuse_signal_is_protected(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={"target_type": "emission_nebula_widefield"},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.13,
                "dirty_background_score": 0.088,
                "chroma_noise_score": 0.15,
            },
            diffuse_context={
                "diffuse": True,
                "emission_diffuse": True,
                "pixel_signal_protection": True,
            },
        )

        self.assertEqual(decision["decision"], "skip")
        self.assertEqual(decision["source"], "target_protection_policy")
        self.assertGreaterEqual(decision["confidence"], 0.80)

    def test_stage3_decision_applies_high_confidence_offline_gradient(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.14,
                "dirty_background_score": 0.32,
                "chroma_noise_score": 0.04,
            },
            diffuse_context={"diffuse": False},
        )

        self.assertEqual(decision["decision"], "apply")
        self.assertEqual(decision["source"], "deterministic_offline_policy")

    def test_stage3_dynamic_rbf_candidates_expand_for_noisy_complex_fields(self):
        processor = pipeline_module.SeestarPostProcessor()
        processor.log = FakeLogger()
        processor.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: object())

        with patch.object(
            sys.modules["stage_support"],
            "measure_image_features",
            return_value=pipeline_module.ImageFeatures(
                bg_std=0.070,
                star_density=0.006,
                object_area_ratio=0.42,
            ),
        ):
            candidates = processor._stage3_subsky_rbf_candidates()

        self.assertGreaterEqual(len(candidates), 4)
        command_text = [" ".join(cmd) for cmd in candidates]
        self.assertTrue(any("-smooth=1.000" in text or "-smooth=1.200" in text for text in command_text))
        self.assertTrue(any("-tolerance=0.800" in text or "-tolerance=0.700" in text for text in command_text))


for _legacy_name, _legacy_test in list(vars(PipelinePluginFallbackTests).items()):
    if _legacy_name.startswith("test_") and "stage4" in _legacy_name:
        setattr(
            PipelinePluginFallbackTests,
            _legacy_name,
            unittest.skip(
                "superseded by the PCC-only contract in test_stage4_pcc_policy.py"
            )(_legacy_test),
        )


if __name__ == "__main__":
    unittest.main()
