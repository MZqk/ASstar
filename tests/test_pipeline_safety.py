#!/usr/bin/env python3
"""Tests for cross-stage target, color, and denoise safety rules."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits


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
import stage5_handoff  # noqa: E402
import spatial_background_lineage  # noqa: E402
from stage8_starless_finish import pixel_sha256  # noqa: E402
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
        height, width = 96, 128
        y, x = np.mgrid[:height, :width]
        diffuse = 0.04 + 0.12 * np.exp(
            -(((x - 63.5) / 28.0) ** 2 + ((y - 47.5) / 20.0) ** 2)
        )
        self.image_pixels = np.stack(
            (diffuse * 1.20, diffuse * 0.92, diffuse * 0.75),
            axis=0,
        ).astype(np.float32)
        for row, col in (
            (20, 25),
            (20, 102),
            (35, 55),
            (35, 72),
            (60, 55),
            (60, 72),
            (75, 25),
            (75, 102),
        ):
            self.image_pixels[:, row, col] = (0.80, 0.75, 0.70)
        self.saved_image_pixels: dict[str, np.ndarray] = {}
        self._loaded_source: str | None = None
        self._seed_image("stage5_linear")
        self._seed_image("stage4_color")
        self._seed_image("stage5_input_linear")
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
        self.target_profile = {
            "primary_target": {
                "name": "NGC6910",
                "type": "open_cluster",
                "frozen": True,
            },
            "secondary_labels": [
                "bright_core",
                "large_nebulosity",
                "emission_red",
            ],
        }
        self.siril = SimpleNamespace(
            get_image_pixeldata=(
                lambda preview=False: np.array(self.image_pixels, copy=True)
            ),
            set_image_pixeldata=self._set_image_pixeldata,
        )
        self._write_spatial_lineage()
        input_lineage = stage5_handoff.freeze_stage5_input_lineage(
            self,
            upstream_loaded=True,
            baseline_saved=True,
        )
        stage5_handoff.freeze_stage5_handoff(
            self,
            origin=stage5_handoff.CURRENT_RUN_ORIGIN,
            stage_status="ok",
            deconvolution_integrity_ok=True,
            denoise_integrity_ok=True,
            formal_eligible=True,
            input_lineage=input_lineage,
        )

    def _seed_image(
        self,
        stem: str,
        pixels: np.ndarray | None = None,
    ) -> None:
        values = np.array(
            self.image_pixels if pixels is None else pixels,
            copy=True,
        )
        self.saved_image_pixels[stem] = values
        (self.process_dir / f"{stem}.fit").write_bytes(b"FIT")

    def _set_image_pixeldata(self, pixels: object) -> None:
        self.image_pixels = np.asarray(pixels).copy()

    def _set_current_image_pixeldata(
        self,
        pixels: object,
        **_kwargs: object,
    ) -> None:
        self._set_image_pixeldata(pixels)

    def _write_spatial_lineage(self) -> None:
        height, width = self.image_pixels.shape[1:]
        support = np.ones((height, width), dtype=np.uint8)
        points = [
            (
                (cell_x + 0.5) / 4.0 * (width - 1),
                (cell_y + 0.5) / 4.0 * (height - 1),
            )
            for cell_y in range(4)
            for cell_x in range(4)
        ]
        support_path = (
            self.process_dir / "stage3_spatial_background_support.fit"
        )
        input_path = self.process_dir / "stage3_bg_input.fit"
        output_path = self.process_dir / "stage3_bgremoved.fit"
        fits.PrimaryHDU(support).writeto(support_path)
        fits.PrimaryHDU(self.image_pixels).writeto(input_path)
        fits.PrimaryHDU(self.image_pixels).writeto(output_path)
        reference_metrics = (
            spatial_background_lineage.measure_spatial_background_planes(
                self.image_pixels,
                support.astype(bool),
                points,
                patch_radius=2,
            )
        )
        reference_plane = {
            name: {
                "coefficients": list(component.get("coefficients") or []),
                "slope_span": component.get("slope_span"),
                "slope_significance_sigma": component.get(
                    "slope_significance_sigma"
                ),
            }
            for name, component in reference_metrics.items()
        }
        lineage = spatial_background_lineage.seal_lineage({
            "schema": spatial_background_lineage.LINEAGE_SCHEMA,
            "status": "accepted",
            "accepted": True,
            "review_required": False,
            "run_id": "pipeline-safety-fixture",
            "processing_route": "verified_noop",
            "image_shape": [height, width],
            "channel_layout": "rgb_chw",
            "support_artifact": support_path.name,
            "support_kind": "candidate_independent_full_sky_mask",
            "support_pixel_count": int(np.count_nonzero(support)),
            "support_coverage": 1.0,
            "sample_patch_support_pixel_count": int(
                np.count_nonzero(
                    spatial_background_lineage.build_sample_patch_support(
                        support.shape,
                        points,
                        support.astype(bool),
                        patch_radius=2,
                    )[0]
                )
            ),
            "sample_patch_min_support_pixel_count": 25,
            "support_sha256": hashlib.sha256(
                support_path.read_bytes()
            ).hexdigest(),
            "stage3_input_sha256": hashlib.sha256(
                input_path.read_bytes()
            ).hexdigest(),
            "stage3_input_pixel_sha256": (
                spatial_background_lineage._array_sha256(self.image_pixels)
            ),
            "stage3_output_sha256": hashlib.sha256(
                output_path.read_bytes()
            ).hexdigest(),
            "stage3_output_pixel_sha256": (
                spatial_background_lineage._array_sha256(self.image_pixels)
            ),
            "fit_points": [list(point) for point in points[:12]],
            "validation_points": [list(point) for point in points[12:]],
            "patch_radius": 2,
            "reference_metrics": reference_metrics,
            "reference_plane": {
                "coordinate_system": "normalized_image_xy",
                "components": reference_plane,
                "sha256": spatial_background_lineage._json_sha256(
                    reference_plane
                ),
            },
            "projection_schema": None,
            "projection_reason_code": None,
            "selected_components": [],
            "unresolved_components": [],
        })
        (self.process_dir / "stage3_spatial_background_lineage.json").write_text(
            json.dumps(lineage),
            encoding="utf-8",
        )
        candidate_path = self.process_dir / "stage7_spatial_reference.fit"
        fits.PrimaryHDU(self.image_pixels).writeto(candidate_path)
        spatial = spatial_background_lineage.assess_stage7_spatial_chroma(
            self.process_dir,
            self.image_pixels,
            self.image_pixels,
            transform_identity={
                "status": "ok",
                "method": "synthetic_authenticated_tone",
                "digest": "pipeline-safety",
            },
        )
        reference = spatial_background_lineage.build_stage7_display_reference(
            self.process_dir,
            {
                "name": "synthetic_pipeline_safety",
                "file": candidate_path.name,
                "spatial_chroma_quality": spatial,
            },
            {
                "status": "active",
                "schema": "starun.stage7-matched-domain-transfer.v4",
                "method": "synthetic_authenticated_tone",
                "chain_contract": {"sha256": "pipeline-safety"},
            },
        )
        (
            self.process_dir
            / spatial_background_lineage.STAGE7_REFERENCE_NAME
        ).write_text(json.dumps(reference), encoding="utf-8")

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
        if args and args[0] == "load" and len(args) > 1:
            stem = str(args[1])
            source_path = self.process_dir / f"{stem}.fit"
            pixels = self.saved_image_pixels.get(stem)
            if not source_path.is_file() or pixels is None:
                raise CommandError(f"missing image source: {stem}")
            self.image_pixels = np.array(pixels, copy=True)
            self._loaded_source = stem
        elif args and args[0] == "save" and len(args) > 1:
            self._seed_image(str(args[1]))

    def _save_stage_output(self, stem: str) -> bool:
        if self._loaded_source is None:
            return False
        self._seed_image(stem)
        return True

    def _read_image_by_stem(self, stem: str):
        path = self.process_dir / f"{stem}.fit"
        pixels = self.saved_image_pixels.get(stem)
        if not path.is_file() or pixels is None:
            return None
        return np.array(pixels, copy=True)

    def _fits_stage_fingerprint(self, path: Path) -> dict[str, str] | None:
        pixels = self.saved_image_pixels.get(path.stem)
        if pixels is None:
            return None
        return {"data_sha256": pixel_sha256(pixels)}

    def _active_target_type(self) -> str:
        return "open_cluster"

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
            pipeline._seed_image("stage7_stretched")
            pipeline._stage7_stretch_accepted = True
            pipeline._stage7_stretch_output = "stage7_stretched"
            run_stage8_nebula_enhancement(pipeline)

            self.assertEqual(
                pipeline._stage8_final_quality,
                "star_preserve_secondary_nebulosity",
            )
            stage8_report = pipeline.reports["stage8_enhancement_report.json"]
            self.assertEqual(
                stage8_report["mode"],
                "star_preserve_secondary_nebulosity",
            )
            self.assertTrue(
                stage8_report["secondary_nebulosity_overlay"]["accepted"]
            )
            self.assertEqual(
                stage8_report["handoff"]["processing_route"],
                "star_preserve_secondary_nebulosity",
            )
            self.assertTrue(stage8_report["handoff"]["formal_eligible"])
            self.assertFalse(stage8_report["handoff"]["restricted_downstream"])
            self.assertEqual(pipeline.records[-1][1], "ok")

            run_stage9_star_remixing(pipeline)

            self.assertEqual(pipeline.records[-1][1], "skipped")
            self.assertEqual(pipeline._stage9_final_source, "stage9_remixed")
            self.assertEqual(
                pipeline._stage9_stars_application_mode,
                "stars_not_required",
            )
            self.assertTrue(pipeline._stage9_remix_formally_accepted)
            self.assertFalse(pipeline._stage9_output_withheld)
            self.assertTrue((pipeline.process_dir / "stage9_remixed.fit").exists())

    def test_target_bypass_rejected_stage7_keeps_with_stars_review_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _StarPreservePipeline(Path(tmpdir))
            run_stage6_star_separation(pipeline)
            pipeline._stage7_stretch_accepted = False
            pipeline._stage7_stretch_output = None
            pipeline._stage7_review_source = "stage7_review_with_stars"
            pipeline._seed_image("stage7_review_with_stars")

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
                "with_stars_review_fallback",
            )
            self.assertTrue(pipeline._stage9_output_contains_stars)
            self.assertFalse(pipeline._stage9_output_withheld)
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
            pipeline.saved_image_pixels.pop("stage6_passthrough", None)
            (pipeline.process_dir / "stage5_linear.fit").unlink()
            pipeline.saved_image_pixels.pop("stage5_linear", None)

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

            self.assertTrue(pipeline._stage9_output_withheld)
            self.assertEqual(
                pipeline._stage9_stars_application_mode,
                "withheld_no_with_stars_review_source",
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
                "with_stars_review_fallback",
            )
            self.assertTrue(pipeline._stage9_output_contains_stars)
            self.assertFalse(pipeline._stage9_remix_formally_accepted)
            self.assertTrue((pipeline.process_dir / "stage9_remixed.fit").exists())


if __name__ == "__main__":
    unittest.main()
