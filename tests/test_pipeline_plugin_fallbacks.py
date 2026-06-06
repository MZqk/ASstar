#!/usr/bin/env python3
"""Fallback and degrade behavior tests for pipeline stages 4/5/7/10."""

from __future__ import annotations

import importlib.util
import os
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


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MODULE_PATH = REPO_ROOT / "pipeline" / "seestar_Superimpose.py"


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
stage4_color_calibration = pipeline_module.SeestarPostProcessor.stage4_color_calibration
stage5_linear_denoise = pipeline_module.SeestarPostProcessor.stage5_linear_denoise
stage2_view_correction = pipeline_module.SeestarPostProcessor.stage2_view_correction
stage6_stretching = pipeline_module.SeestarPostProcessor.stage6_stretching
stage7_star_separation = pipeline_module.SeestarPostProcessor.stage7_star_separation
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


class FakeProcessor:
    def __init__(self, module: Any, work_dir: Path) -> None:
        self.module = module
        self.log = FakeLogger()
        self.work_dir = work_dir
        self.process_dir = work_dir / "process"
        self.process_dir.mkdir(exist_ok=True)

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

    def _syqon_starless_cli_options(self):
        return pipeline_module.SeestarPostProcessor._syqon_starless_cli_options(self)

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
        return dict(self.header_metadata)

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
    ) -> None:
        self.results.append((name, status, duration, message))

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
        self.assertIn(("save", "stage6_stretched"), calls)

    def test_stage_json_aliases_follow_display_stage_names(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name)
        log = FakeLogger()

        pipeline_module.write_stage_json(
            process_dir,
            log,
            "stage6_stretch_quality.json",
            {"stage": "stage7_stretch"},
        )

        self.assertTrue((process_dir / "stage6_stretch_quality.json").exists())
        self.assertTrue((process_dir / "stage7_stretch_quality.json").exists())

        pipeline_module.write_stage_json(
            process_dir,
            log,
            "stage6_starless_quality.json",
            {"stage": "stage6_starless"},
        )

        self.assertTrue((process_dir / "stage6_starless_quality.json").exists())
        self.assertTrue((process_dir / "stage7_quality.json").exists())

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

    def test_stage5_uses_aberration_api_with_script_preferred_sharpen_and_denoise(self):
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
        self.assertIn("星点矫正使用 SASP Aberration API", message)
        self.assertIn("矫正星点", processor.aberration_calls)
        self.assertEqual(
            processor.workflow_command_used.get("初步降噪"),
            "CosmicClarity Denoise script (CosmicClarity_Denoise.py)",
        )
        sharpen_calls = [args for step, _name, args in processor.script_calls if step == "锐化"]
        self.assertTrue(sharpen_calls)
        self.assertIn("-non_stellar_strength", sharpen_calls[0])
        self.assertIn("3", sharpen_calls[0])

    def test_stage5_uses_local_model_api_even_when_aberration_env_disabled(self):
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
        self.assertIn("星点矫正使用 SASP Aberration API", message)
        self.assertIn("矫正星点", processor.aberration_calls)

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
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"mock")
        stage7_module = sys.modules["stages.stage7_star_separation"]

        self.assertEqual(stage7_module._stage7_linear_source(processor), "stage5_linear")

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

    def test_stage5_prefers_native_cosmic_clarity_when_classic_unconfigured(self):
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
        self.assertEqual(len(native_calls), 2)
        self.assertIn("--mode", native_calls[0][2])
        self.assertIn("sharpen", native_calls[0][2])
        self.assertIn("denoise", native_calls[1][2])
        self.assertIn("--denoise-mode", native_calls[1][2])
        self.assertIn("full", native_calls[1][2])
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("CosmicClarity Sharpen classic executable 未配置，已直接选择", message)
        self.assertIn("CosmicClarity Denoise classic executable 未配置，已直接选择", message)

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
        self.assertIn("Stage5 background guard skipped global sharpen", message)

    def test_stage5_classic_denoise_uses_full_mode_for_chroma_first_policy(self):
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
        self.assertTrue(denoise_calls)
        self.assertIn("-denoising_mode", denoise_calls[0])
        self.assertIn("full", denoise_calls[0])

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
        self.assertIn(("denoise", "-mod=0.24"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 background guard rollback", message)

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
        self.assertIn(("denoise", "-mod=0.24"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("chroma_bg_ratio_growth", message)

    def test_stage4_spcc_success_stays_ok_with_platesolve_by_default(self):
        processor = self._new_processor()

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("platesolve -focal=160 -pixelsize=2.90 -catalog=gaia -order=3 ok", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("platesolve", cmds)
        platesolve_calls = [call for call in processor.cmd_calls if call[0] == "platesolve"]
        self.assertEqual(
            platesolve_calls[0],
            ("platesolve", "-focal=160", "-pixelsize=2.90", "-catalog=gaia", "-order=3"),
        )
        self.assertIn("spcc", cmds)
        self.assertNotIn("cc", cmds)
        self.assertFalse({"mirrorx", "mirrory", "flip", "rotate"} & set(cmds))

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

    def test_stage4_explicit_white_ref_overrides_emission_nebula_profile(self):
        processor = self._new_processor()
        processor.cfg.stage4_platesolve_enabled = True
        processor.target_profile = {"target_type": "emission_nebula_widefield"}
        processor.cfg.stage4_spcc_white_ref = "Photon Flux"

        stage4_color_calibration(processor)

        spcc_calls = [call for call in processor.cmd_calls if call[0] == "spcc"]
        self.assertTrue(spcc_calls)
        self.assertIn('"-whiteref=Photon Flux"', spcc_calls[0])
        self.assertNotIn('"-whiteref=Star, type G2(v)"', spcc_calls[0])

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
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(edge_black_ratio=0.19)
        )

        stage2_view_correction(processor)

        crop_calls = [call for call in processor.cmd_calls if call[0] == "crop"]
        self.assertGreaterEqual(len(crop_calls), 2)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("adaptive edge crop", message)

    def test_stage6_applies_weak_object_conservative_tuning_before_asinh(self):
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

        stage6_stretching(processor)

        self.assertIn(("asinh", "2.45", "0.001"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("weak-object tuning applied", message)

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
        self.assertAlmostEqual(candidates[0]["params"]["asinh_stretch"], 1.5)
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.008)
        self.assertEqual(candidates[1]["name"], "cand_b")
        self.assertAlmostEqual(candidates[1]["params"]["asinh_stretch"], 1.6)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.007)
        self.assertLessEqual(candidates[1]["params"]["ghs_stretchamount"], 1.01)

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
                {"p99": 0.00224, "max": 0.00298},
            )
        )

        self.assertEqual(adaptation["mode"], "extreme_low_background")
        self.assertIn("offset_cap", adaptation)
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.0019, places=4)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.0019, places=4)
        self.assertLess(candidates[0]["params"]["asinh_offset"], 0.00224)
        self.assertLess(candidates[1]["params"]["asinh_offset"], 0.00224)

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

    def test_stage8_applies_blue_guard_when_starless_layer_is_too_blue(self):
        processor = self._new_processor()
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

    def test_stage10_script_failure_prefers_scunet_command_fallback(self):
        processor = self._new_processor()
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

    def test_stage10_cli_failure_prefers_in_process_cosmic_clarity_fallback(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.cli_fail_steps.add("最终降噪")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=in-process CosmicClarity Denoise script", message)
        self.assertIn("fallback_status=success", message)
        self.assertIn("final_denoise_primary=CosmicClarity Denoise CLI subprocess", message)
        self.assertIn("final_denoise_effective=CosmicClarity Denoise script", message)
        self.assertNotIn("Siril-SCUNet Denoise 回退不可用", message)

    def test_stage10_cli_siril_connection_failure_is_reported(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.cli_fail_steps.add("最终降噪")
        processor.cli_failure_errors["最终降噪"] = (
            "CosmicClarity_Denoise.py: subprocess exited with code 1; "
            "output_tail=Error: Failed to connect to Siril"
        )

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("CLI Siril 连接失败", message)
        self.assertIn("fallback_component=in-process CosmicClarity Denoise script", message)

    def test_stage10_uses_native_cosmic_clarity_without_classic_executable(self):
        processor = self._new_processor()
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
        self.assertIn("CosmicClarity classic executable 未配置，已选择 Native Denoise", message)
        self.assertIn("final_denoise_primary=CosmicClarity Native Denoise cli-subprocess", message)
        self.assertIn("primary_status=success", message)
        self.assertNotIn("fallback_component=CosmicClarity Native Denoise", message)

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
        processor.feature_measurements.append(pipeline_module.ImageFeatures(bg_median=0.2))

        stage6_stretching(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("拉伸后特征", message)
        self.assertIn("bg_median=0.2000", message)

    def test_stage8_records_post_starless_feature_summary(self):
        processor = self._new_processor()
        processor.feature_measurements.append(pipeline_module.ImageFeatures(object_area_ratio=0.33))

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Starless 后特征", message)
        self.assertIn("object_area=0.330", message)

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

    def test_stage9_caps_star_remix_when_stage8_used_fallback(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.star_intensity = 1.05
        processor.cfg.star_fallback_intensity = 1.05
        processor._stage8_fallback_used = True
        processor._stage8_final_source = "stage8_input_starless"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        stage9_star_remixing(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [("stage8_input_starless", "starmask_stretched", 0.95)],
        )
        self.assertIn("Stage8 fallback source active", message)

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

    def test_stage10_uses_linear_resume_output_suffixes(self):
        processor = self._new_processor()
        processor._stage1_input_mode = "linear_resume"

        stage10_export(processor)

        linear_base = pipeline_module.RESULT_BASENAME_TEMPLATE + "_linear"
        self.assertIn(("savetif", linear_base, "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", linear_base), processor.cmd_calls)
        self.assertIn(("save", linear_base + "_final"), processor.cmd_calls)
        self.assertEqual(processor.main_output_basename_template, linear_base)

    def test_stage10_saves_fits_before_preview_autostretch_png(self):
        processor = self._new_processor()

        stage10_export(processor)

        commands = [call[0] for call in processor.cmd_calls]
        self.assertLess(commands.index("save"), commands.index("autostretch"))
        self.assertLess(commands.index("autostretch"), commands.index("savepng"))
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("PNG preview stretch applied", message)

    def test_stage7_uses_sasp_when_probe_disabled(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.command_labels["去星"] = "SASP Dark Star"

        stage7_star_separation(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=SASP Dark Star", message)
        self.assertIn("fallback_status=success", message)
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

    def test_stage7_uses_syqon_script_outputs_when_available(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/SyQon-Starless.py")
        processor.syqon_output_mode = "both"

        stage7_star_separation(processor)

        _name, status, _dur, _message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertEqual(
            processor.workflow_command_used.get("去星"),
            "SyQon Starless cli-subprocess (SyQon-Starless.py)",
        )
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
        info_messages = [msg for level, msg in processor.log.events if level == "info"]
        self.assertTrue(any("Zenith v1" in msg for msg in info_messages))
        self.assertTrue((processor.process_dir / "starless.fit").exists())
        self.assertTrue((processor.process_dir / "starmask.fit").exists())

    def test_stage7_can_force_syqon_cpu_with_env(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/SyQon-Starless.py")
        processor.syqon_output_mode = "starless"

        with patch.dict(os.environ, {pipeline_module.ENV_SYQON_GPU_KEY: "0"}, clear=False):
            stage7_star_separation(processor)

        syqon_calls = [
            args
            for step, script_name, args in processor.script_calls
            if step == "去星" and script_name == "SyQon-Starless.py"
        ]
        self.assertTrue(syqon_calls)
        self.assertIn("--no_gpu", syqon_calls[0])

    def test_run_linear_resume_skips_stages_2_through_5(self):
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
        processor.stage6_stretching = lambda: calls.append("stage6")  # type: ignore[method-assign]
        processor.stage7_star_separation = lambda: calls.append("stage7")  # type: ignore[method-assign]
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
                "stage7",
                "stage6",
                "stage8",
                "stage9",
                "stage10",
                "stage11",
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
        self.assertIn(
            ("阶段 3: 背景提取", "skipped", "skipped by linear resume mode"),
            skipped,
        )
        self.assertIn(
            ("阶段 4: 图像解析 + 色彩校准", "skipped", "skipped by linear resume mode"),
            skipped,
        )
        self.assertIn(
            ("阶段 5: 线性反卷积 / 轻降噪", "skipped", "skipped by linear resume mode"),
            skipped,
        )

    def test_run_stage2_corrected_resume_continues_from_stage3(self):
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
        processor.stage6_stretching = lambda: calls.append("stage6")  # type: ignore[method-assign]
        processor.stage7_star_separation = lambda: calls.append("stage7")  # type: ignore[method-assign]
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
                "stage3",
                "stage4",
                "stage5",
                "stage7",
                "stage6",
                "stage8",
                "stage9",
                "stage10",
                "stage11",
                "cleanup",
            ],
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

        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": "1"}, clear=False):
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

    def test_stage_result_display_status_marks_ok_variants(self):
        fallback_result = pipeline_module.StageResult(
            "阶段 X",
            "ok",
            message="fallback: failed_component=A; fallback_component=B; fallback_status=success",
        )
        skipped_result = pipeline_module.StageResult(
            "阶段 Y",
            "ok",
            message="SPCC skipped on Light_ preprocess mode",
        )
        degraded_result = pipeline_module.StageResult("阶段 Z", "degraded")

        self.assertEqual(fallback_result.display_status, "ok_with_fallback")
        self.assertEqual(skipped_result.display_status, "ok_skipped_optional")
        self.assertEqual(degraded_result.display_status, "degraded")

    def test_ai_plan_text_fallback_extracts_adjustments_without_strict_json(self):
        processor = pipeline_module.SeestarPostProcessor()
        raw_text = (
            "Recommendation summary\n"
            "background_protection: 0.92\n"
            "global_contrast_delta=-0.02\n"
            "global_saturation_delta: 0.04\n"
            "red_balance_delta: -0.01\n"
            "blue_balance_delta: 0.02\n"
            "denoise_strength: 0.07\n"
            "detail_boost=0.05\n"
        )

        parsed = processor._extract_first_json_object(raw_text)

        self.assertIn("adjustments", parsed)
        self.assertAlmostEqual(parsed["adjustments"]["background_protection"], 0.92)
        self.assertAlmostEqual(parsed["adjustments"]["detail_boost"], 0.05)

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

            def _record_stage(self, name: str, status: str, elapsed: float, message: str) -> None:
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
        self.assertEqual(processor.results[-1][1], "ok")

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

            def _record_stage(self, name: str, status: str, elapsed: float, message: str) -> None:
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
        self.assertEqual(
            [record["status"] for record in processor.report["attempts"][:2]],
            ["graxpert_runtime_error", "graxpert_runtime_error"],
        )

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

            def _record_stage(self, name: str, status: str, elapsed: float, message: str) -> None:
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
                self.cfg = SimpleNamespace(workflow_plugin_probe_enabled=False)
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

            def _record_stage(self, name: str, status: str, elapsed: float, message: str) -> None:
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


if __name__ == "__main__":
    unittest.main()
