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
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

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

    def _write_stage_json(self, _name: str, _payload: dict[str, Any]) -> None:
        return None

    def _apply_previous_stage_star_remix(self, source_stem: str, starmask_name: str, intensity: float):
        self.previous_stage_remix_calls.append((source_stem, starmask_name, intensity))
        return not self.fail_previous_stage_remix


class PipelinePluginFallbackTests(unittest.TestCase):
    def _new_processor(self) -> FakeProcessor:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return FakeProcessor(pipeline_module, Path(td.name))

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
        self.assertIn("未执行星点矫正", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("denoise", cmds)

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
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("CosmicClarity Sharpen classic executable 未配置，已直接选择", message)
        self.assertIn("CosmicClarity Denoise classic executable 未配置，已直接选择", message)
        self.assertEqual(
            pipeline_module.StageResult("阶段 5", status, message=message).display_status,
            "ok",
        )

    def test_stage4_spcc_success_stays_ok(self):
        processor = self._new_processor()

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertEqual(message, "")
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("platesolve", cmds)
        self.assertIn("spcc", cmds)
        self.assertNotIn("cc", cmds)

    def test_stage4_pcc_fallback_stays_ok(self):
        processor = self._new_processor()
        processor.fail_commands.add("spcc")

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SPCC 失败", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("pcc", cmds)
        self.assertNotIn("cc", cmds)

    def test_stage4_still_tries_spcc_when_platesolve_failed(self):
        processor = self._new_processor()
        processor.fail_commands.add("platesolve")

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("platesolve 失败", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("spcc", cmds)
        self.assertNotIn("cc", cmds)

    def test_stage4_cc_fallback_marks_degraded(self):
        processor = self._new_processor()
        processor.fail_commands.update({"platesolve", "spcc", "pcc"})

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertIn("使用 CCM 回退完成色彩校准", message)
        self.assertIn("platesolve 失败，已使用非光度 CCM 回退", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("cc", cmds)

    def test_stage4_skips_spcc_on_light_preprocess_mode_by_default(self):
        processor = self._new_processor()
        processor._stage1_input_mode = "light_preprocess"

        stage4_color_calibration(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SPCC skipped on Light_ preprocess mode", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertNotIn("spcc", cmds)
        self.assertIn("pcc", cmds)

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
            [("starless_enhanced", "starmask", processor.cfg.star_intensity)],
        )
        self.assertIn("previous_stage_star_remix source=starless_enhanced", message)
        pm_calls = [call for call in processor.cmd_calls if call[0] == "pm"]
        self.assertFalse(pm_calls)

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
                "stage6",
                "stage7",
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
            ("阶段 5: 星点矫正 / 锐化 / 初步降噪", "skipped", "skipped by linear resume mode"),
            skipped,
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


if __name__ == "__main__":
    unittest.main()
