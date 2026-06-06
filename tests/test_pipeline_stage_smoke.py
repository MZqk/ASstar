#!/usr/bin/env python3
"""Smoke-level integration tests for all ten pipeline stage entry points."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def _install_sirilpy_stub() -> None:
    if "sirilpy" in sys.modules:
        return
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class SirilError(Exception):
        pass

    class CommandError(SirilError):
        pass

    class DataError(SirilError):
        pass

    sirilpy.SirilInterface = object
    exceptions.SirilError = SirilError
    exceptions.CommandError = CommandError
    exceptions.DataError = DataError
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions


_install_sirilpy_stub()

from models import (  # noqa: E402
    ImageFeatures,
    PipelineCheckpoint,
    PipelineConfig,
    PipelineStage,
)
from stages import (  # noqa: E402
    stage10_export,
    stage1_preparation,
    stage2_view_correction,
    stage3_background_extraction,
    stage4_color_calibration,
    stage5_linear_denoise,
    stage6_stretching,
    stage7_star_separation,
    stage8_nebula_enhancement,
    stage9_star_remixing,
)


class _Log:
    def stage_start(self, _name: str) -> None:
        return

    def stage_end(self, _name: str | None = None) -> float:
        return 0.01

    def info(self, _message: str) -> None:
        return

    def warn(self, _message: str) -> None:
        return

    def error(self, _message: str) -> None:
        return

    def debug(self, _message: str) -> None:
        return


class PipelineStageTests(unittest.TestCase):
    def test_formal_stage_labels_are_unique_and_contiguous(self):
        labels = [stage.label for stage in PipelineStage]

        self.assertEqual(len(labels), 11)
        self.assertEqual(len(set(labels)), 11)
        for number, label in enumerate(labels, start=1):
            self.assertTrue(label.startswith(f"阶段 {number}:"))

        self.assertNotIn(
            PipelineCheckpoint.PRE_STARLESS_COMPATIBILITY_GATE.label,
            labels,
        )


class _Siril:
    def get_image_shape(self):
        return (3, 100, 100)

    def get_image_pixeldata(self, preview: bool = False):
        return None


class _Pipeline:
    def __init__(self, root: Path) -> None:
        self.cfg = PipelineConfig()
        self.log = _Log()
        self.siril = _Siril()
        self.work_dir = root
        self.process_dir = root / "process"
        self.process_dir.mkdir()
        self.results: list[tuple[str, str, float, str]] = []
        self.commands: list[tuple[object, ...]] = []
        self.pipeline_policy = {}
        self.workflow_command_used = {}
        self.source_file = root / "stacked.fit"
        self.stretched_name = "stage7_stretched"
        self.starless_file = None
        self.starmask_file = None
        self.pre_starless_gate_report = {}
        self._stage1_input_mode = "stacked"
        self._last_plugin_script_error = None
        self._last_sasp_stage8_api_error = None
        self._last_aberration_api_error = None
        self._last_scunet_fallback_error = None

    def _record_stage(self, name: str, status: str, duration: float, message: str = "") -> None:
        self.results.append((name, status, duration, message))

    def cmd_with_check(self, *args, **_kwargs) -> None:
        self.commands.append(tuple(args))

    def _save_stage_output(self, stem: str) -> bool:
        (self.process_dir / f"{stem}.fit").touch()
        return True

    def _write_stage_json(self, *_args, **_kwargs) -> None:
        return

    def _short_text(self, value, _limit: int = 160) -> str:
        return str(value)

    def _stage_diff_note(self, *_args) -> str:
        return ""

    def _measure_current_features(self):
        return ImageFeatures(edge_black_ratio=0.0)

    def _adaptive_features_current(self):
        return {}

    def _feature_summary_note(self, _label: str) -> str:
        return ""


class PipelineStageSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pipeline = _Pipeline(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_stage1_preparation_smoke(self) -> None:
        source = self.root / "stacked.fit"
        source.touch()
        self.pipeline._prepare_process_dir = lambda: None
        self.pipeline._find_fit_files = lambda: [source]
        self.pipeline._is_candidate_stacked = lambda _path: True
        self.pipeline._load_stacked_file = lambda files: self.commands_append("load_stacked", files)
        self.pipeline._preprocess_light_frames = lambda _files: None

        stage1_preparation.run_stage1_preparation(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")

    def commands_append(self, *args) -> None:
        self.pipeline.commands.append(tuple(args))

    def test_stage2_view_correction_smoke(self) -> None:
        with (
            patch.object(stage2_view_correction, "_detect_auto_edge_crop", return_value=(None, "no crop")),
            patch.object(stage2_view_correction, "_edge_color_artifact_crop", return_value=""),
        ):
            stage2_view_correction.run_stage2_view_correction(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage2_corrected.fit").exists())

    @property
    def process_dir(self) -> Path:
        return self.pipeline.process_dir

    def test_stage3_background_extraction_smoke(self) -> None:
        self.pipeline._stage3_measure_features = lambda _label: ImageFeatures()
        self.pipeline._stage3_signal_preservation_metrics = lambda *_args: {}
        self.pipeline._stage3_quality_gate = lambda *_args: (True, "ok")
        self.pipeline._stage3_subsky_rbf_candidates = lambda: []
        self.pipeline.workflow_plugin_probe_enabled = False
        with (
            patch.object(
                stage3_background_extraction,
                "_stage3_background_candidate_chain",
                return_value=([], [], "smoke_no_candidates"),
            ),
            patch.object(stage3_background_extraction, "_stage3_theoretical_plugin_candidates", return_value=[]),
            patch.object(stage3_background_extraction, "_stage3_graxpert_candidates", return_value=[]),
        ):
            stage3_background_extraction.run_stage3_background_extraction(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertTrue((self.process_dir / "stage3_bgremoved.fit").exists())

    def test_stage4_color_calibration_smoke(self) -> None:
        self.pipeline.cfg.stage4_platesolve_enabled = False
        self.pipeline._read_fits_metadata = lambda *_args: {}
        with (
            patch.object(stage4_color_calibration, "_stage4_header_metadata", return_value={}),
            patch.object(stage4_color_calibration, "_stage4_image_geometry", return_value={"current_shape": {}}),
            patch.object(
                stage4_color_calibration,
                "_stage4_local_color_fallback",
                return_value=(True, "LOCAL_STAR_WB", "", 0.6, {}, "local fallback"),
            ),
        ):
            stage4_color_calibration.run_stage4_color_calibration(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertEqual(self.pipeline.color_calibration_report["method"], "LOCAL_STAR_WB")

    def test_stage5_linear_denoise_smoke(self) -> None:
        self.pipeline._export_linear_intermediate = lambda: True
        self.pipeline._active_policy_name = lambda: "generic_low_snr_safe"
        self.pipeline._active_target_type = lambda: "generic_low_snr_safe"
        with (
            patch.object(stage5_linear_denoise, "_run_stage5_rl_deconvolution", return_value=False),
            patch.object(stage5_linear_denoise, "_run_builtin_linear_denoise", return_value=True),
        ):
            stage5_linear_denoise.run_stage5_linear_denoise(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage5_linear.fit").exists())

    def test_stage6_stretching_smoke(self) -> None:
        self.pipeline._ai_stage_advisory_enabled = lambda _name: False
        self.pipeline._run_stage6_ai_stretching = lambda allow_ai: (
            True,
            False,
            [f"allow_ai={allow_ai}"],
            "asinh",
        )

        stage6_stretching.run_stage6_stretching(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue((self.process_dir / "stage7_stretched.fit").exists())

    def test_stage7_star_separation_smoke(self) -> None:
        self.pipeline.pre_starless_gate_report = {
            "ready_for_starless": False,
            "reason": ["unsafe input"],
        }
        self.pipeline.cfg.stage7_skip_unready_starless = True
        self.pipeline._stage7_update_star_remix_from_quality = lambda _record: {}
        self.pipeline._export_sasp_exchange_files = lambda: None
        with patch.object(
            stage7_star_separation,
            "_prepare_star_separation_source",
            return_value=("stage7_stretched", "linear_star_separation", []),
        ):
            stage7_star_separation.run_stage7_star_separation(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "skipped")
        self.assertTrue(self.pipeline._stage7_starless_skipped)

    def test_stage8_nebula_enhancement_smoke(self) -> None:
        self.pipeline.cfg.stage8_masked_enhancement_enabled = True
        self.pipeline._find_external_fit = lambda _names: None
        self.pipeline._ai_stage_advisory_enabled = lambda _name: False
        self.pipeline._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": True,
            "reasons": ["unsafe starless input"],
        }

        stage8_nebula_enhancement.run_stage8_nebula_enhancement(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "degraded")
        self.assertEqual(self.pipeline._stage8_final_source, "stage8_input_starless")

    def test_stage9_star_remixing_smoke(self) -> None:
        self.pipeline._stage9_bad_starless_reason = lambda: "poor starless"
        self.pipeline._stage9_review_safe_source = lambda: "stage7_stretched"

        stage9_star_remixing.run_stage9_star_remixing(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][1], "ok")
        self.assertTrue(self.pipeline._stage9_bypassed_bad_starless)

    def test_stage10_export_smoke(self) -> None:
        (self.process_dir / "stage9_remixed.fit").touch()
        self.pipeline._find_plugin_script = lambda _paths: None
        self.pipeline._classic_cosmic_clarity_args = lambda *_args: None
        self.pipeline._run_cosmic_clarity_native_denoise_fallback = lambda _label: None
        self.pipeline._run_siril_scunet_denoise_fallback = lambda *_args: None
        self.pipeline._result_output_basename = lambda: "result_processed"
        with patch.object(
            stage10_export,
            "export_final_outputs",
            side_effect=lambda _cmd, _log, **kwargs: (
                kwargs["status"],
                kwargs["messages"],
            ),
        ):
            stage10_export.run_stage10_export(self.pipeline)

        self.assertEqual(self.pipeline.results[-1][0], "阶段 10: 最终降噪与导出")
        self.assertIn(self.pipeline.results[-1][1], {"ok", "degraded"})


if __name__ == "__main__":
    unittest.main()
