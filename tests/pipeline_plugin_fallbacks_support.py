#!/usr/bin/env python3
"""Fallback and degrade behavior tests for pipeline stages 4/5/7/10."""

from __future__ import annotations

import copy
import hashlib
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
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
PIPELINE_MODULE_PATH = REPO_ROOT / "pipeline" / "starun.py"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from dualband_palette import resolve_palette_selection  # noqa: E402
from narrowband_normalization import resolve_dual_narrowband_mapping  # noqa: E402


def _ensure_fake_sirilpy() -> None:
    if "sirilpy" in sys.modules:
        fake_sirilpy = sys.modules["sirilpy"]
        fake_exceptions = sys.modules.setdefault(
            "sirilpy.exceptions",
            types.ModuleType("sirilpy.exceptions"),
        )
        siril_error = getattr(fake_exceptions, "SirilError", Exception)
        if not hasattr(fake_exceptions, "SirilConnectionError"):
            fake_exceptions.SirilConnectionError = type(
                "SirilConnectionError",
                (siril_error,),
                {},
            )
        if not hasattr(fake_exceptions, "CommandError"):
            fake_exceptions.CommandError = type(
                "CommandError",
                (siril_error,),
                {},
            )
        if not hasattr(fake_exceptions, "DataError"):
            fake_exceptions.DataError = type(
                "DataError",
                (siril_error,),
                {},
            )
        fake_enums = sys.modules.setdefault(
            "sirilpy.enums",
            types.ModuleType("sirilpy.enums"),
        )
        if not hasattr(fake_enums, "CommandStatus"):
            fake_enums.CommandStatus = type(
                "CommandStatus",
                (),
                {"CMD_GENERIC_ERROR": 1, "CMD_THREAD_RUNNING": 2},
            )
        if not hasattr(fake_sirilpy, "SirilInterface"):
            fake_sirilpy.SirilInterface = object
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
        "starun_pipeline_test_module",
        PIPELINE_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {PIPELINE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline_module = _load_pipeline_module()
stage_support_module = sys.modules["stage_support"]
stage6_services_module = sys.modules["stage6_services"]
stage6_star_separation_module = sys.modules["stages.stage6_star_separation"]
stage7_quality_module = sys.modules["stage7_quality"]
stage7_repair_module = sys.modules["stage7_repair"]
stage5_linear_denoise_module = sys.modules["stages.stage5_linear_denoise"]
stage10_export_module = sys.modules["stages.stage10_export"]
stage4_color_calibration = pipeline_module.StarunPostProcessor.stage4_color_calibration
stage5_linear_denoise = pipeline_module.StarunPostProcessor.stage5_linear_denoise
stage2_view_correction = pipeline_module.StarunPostProcessor.stage2_view_correction
# Test the canonical Stage 6/7 entry points and production order.
stage7_stretching = pipeline_module.StarunPostProcessor.stage7_stretching
stage6_star_separation = pipeline_module.StarunPostProcessor.stage6_star_separation
stage8_nebula_enhancement = pipeline_module.StarunPostProcessor.stage8_nebula_enhancement
_stage9_star_remixing_impl = pipeline_module.StarunPostProcessor.stage9_star_remixing
_stage10_export_impl = pipeline_module.StarunPostProcessor.stage10_export


def stage9_star_remixing(processor):
    """Run Stage 9 while keeping legacy shallow fakes focused on scheduling.

    Older tests deliberately omit a readable starmask pixel buffer.  The
    production preflight must fail closed in that situation, but those tests
    exercise downstream fallback/selection behavior rather than support
    calibration.  Preserve their original scope with a small, valid support
    plan only when the real preflight reports that every support is unavailable.
    """

    stage9_module = sys.modules["stages.stage9_star_remixing"]
    real_preflight = stage9_module._stage9_starmask_support_preflight
    real_persisted_validator = (
        stage9_module._validate_stage9_persisted_output
    )

    def compatibility_persisted_validator(
        pipeline,
        source_stem,
        selected_quality,
    ):
        if bool(
            getattr(
                pipeline,
                "_stage9_test_use_real_persisted_validation",
                False,
            )
        ):
            return real_persisted_validator(
                pipeline,
                source_stem,
                selected_quality,
            )
        report = {
            "schema": "starun.stage9-persisted-output-validation.v1",
            "status": "ok",
            "accepted": True,
            "reason_code": "test_double_persisted_validation_compatibility",
            "selected_attempt": str(
                selected_quality.get("attempt") or "unknown"
            ),
            "selected_formula": str(
                selected_quality.get("formula") or "screen"
            ),
            "catalog_visibility_groups_passed": True,
            "sep_crossmatch_accepted": True,
            "test_double_compatibility": True,
        }
        sep_summary = {
            "schema": "starun.stage9-sep-crossmatch.v1",
            "status": "ok",
            "accepted": True,
            "reason_code": "test_double_sep_crossmatch_compatibility",
            "artifact": "stage9_sep_crossmatch.json",
            "artifact_sha256": "0" * 64,
            "test_double_compatibility": True,
        }
        report["sep_crossmatch"] = sep_summary
        pipeline._stage9_sep_crossmatch_report = {
            **sep_summary,
            "formal_gate_applied": True,
        }
        pipeline._stage9_sep_crossmatch_summary = sep_summary
        pipeline._stage9_persisted_output_validation = report
        return report

    def compatibility_preflight(pipeline, *args, **kwargs):
        report = real_preflight(pipeline, *args, **kwargs)
        if report.get("route") != "unavailable":
            return report

        # These shallow scheduling fakes predate the frozen FWHM contract and
        # intentionally provide neither matched-display nor Stage 5 geometry.
        # Keep their compatibility preflight at the 4 px anchor; production
        # still fails closed before reaching this test-only wrapper.
        pipeline._stage9_spatial_scale = {
            "schema": "starun.stage9-fwhm-spatial-scale.v1",
            "status": "ready",
            "reason_code": "test_double_4px_anchor",
            "source": "test_double_4px_anchor",
            "sample_count": 4,
            "fwhm_median_px": 4.0,
            "fwhm_p25_px": 4.0,
            "fwhm_p75_px": 4.0,
            "anchor_fwhm_px": 4.0,
            "radius_scale": 1.0,
            "area_scale": 1.0,
            "stage9_psf_review_required": False,
        }
        pipeline._stage9_spatial_scale_review_required = False
        pipeline._stage9_psf_review_required = False

        height, width = 16, 16
        get_pixels = getattr(pipeline.siril, "get_image_pixeldata", None)
        if callable(get_pixels):
            try:
                pixels = np.asarray(get_pixels(preview=False))
                height, width = pixels.shape[-2:]
            except Exception:
                pass
        elif isinstance(getattr(pipeline, "image_pixels", None), np.ndarray):
            height, width = pipeline.image_pixels.shape[-2:]

        support = np.zeros((height, width), dtype=bool)
        support[max(0, height // 2 - 1) : height // 2 + 1,
                max(0, width // 2 - 1) : width // 2 + 1] = True
        coverage = float(np.mean(support))
        had_pixel_reader = callable(
            getattr(pipeline.siril, "get_image_pixeldata", None)
        )
        had_pixel_writer = callable(
            getattr(pipeline.siril, "set_image_pixeldata", None)
        )
        if not had_pixel_reader:
            pipeline.siril.get_image_pixeldata = (
                lambda preview=False: pipeline.image_pixels.copy()
            )
        if not had_pixel_writer:
            pipeline.siril.set_image_pixeldata = lambda image: setattr(
                pipeline,
                "image_pixels",
                np.array(image, copy=True),
            )
        shallow_pixel_compatibility = not (had_pixel_reader and had_pixel_writer)
        if shallow_pixel_compatibility:
            pipeline.cfg.stage9_compact_starmask_enabled = False
        # Keep legacy shallow scheduling tests on the legacy coordinator. New
        # targeted-recovery behavior is covered by dedicated stateful tests.
        pipeline.cfg.stage9_targeted_recovery_enabled = False
        write_support = np.ones((height, width), dtype=bool)
        calibration = {
            "status": "ok",
            "support_mode": "normal",
            "stretch": 2.0,
            "offset": 0.001,
            "star_sample_count": 4,
            "compact_component_count": 1,
            "compact_support_coverage": coverage,
            "predicted_change_ratio": coverage,
            "weak_star_retention": 1.0,
            "star_retention": 1.0,
            "_compact_support_mask": write_support,
            "_weak_support_mask": write_support.copy(),
            "_bright_support_mask": np.zeros_like(support),
        }
        candidate = {
            "support_mode": "normal",
            "status": "ok",
            "usable": True,
            "hard_failed": False,
            "advisory": False,
            "support_coverage": coverage,
            "predicted_change_ratio": coverage,
            "weak_star_retention": 1.0,
            "gate_statuses": {},
        }
        report = {
            "schema": "starun.stage9-starmask-support-preflight.v2",
            "status": "ready",
            "route": "normal_only",
            "reason_code": "test_double_support_preflight_compatibility",
            "compact_enabled": not shallow_pixel_compatibility,
            "candidates": {
                "normal": candidate,
                "strict_compact": {**candidate, "support_mode": "strict_compact"},
            },
            "prepared_candidates": [],
            "executed_candidates": [],
            "selected_support_mode": None,
            "_calibrations": {
                "normal": calibration,
                "strict_compact": {**calibration, "support_mode": "strict_compact"},
            },
        }
        pipeline._stage9_starmask_support_preflight = (
            stage9_module.stage9_quality.public_starmask_support_preflight(report)
        )
        return report

    with (
        patch.object(
            stage9_module,
            "_stage9_starmask_support_preflight",
            side_effect=compatibility_preflight,
        ),
        patch.object(
            stage9_module,
            "_validate_stage9_persisted_output",
            side_effect=compatibility_persisted_validator,
        ),
    ):
        return _stage9_star_remixing_impl(processor)


def stage10_export(processor):
    """Keep legacy Stage10 tests scoped while production catalog gates stay hard."""
    stage10_module = sys.modules["stages.stage10_export"]
    managed_module = sys.modules["managed_output"]
    real_stage10_audit = stage10_module.audit_display_visibility
    real_managed_audit = managed_module.audit_display_visibility

    def compatibility_audit(*args, **kwargs):
        audit = real_managed_audit(*args, **kwargs)
        if (
            not bool(kwargs.get("stars_required", False))
            or bool(
                getattr(
                    processor,
                    "_stage10_test_use_real_catalog_visibility",
                    False,
                )
            )
        ):
            return audit
        star_check = ((audit.get("checks") or {}).get("star_visibility") or {})
        if star_check.get("passed") is not True:
            star_check.update(
                passed=True,
                detected=True,
                method="test_double_catalog_visibility_compatibility",
            )
            failed = [
                name
                for name in list(audit.get("failed_checks") or [])
                if name != "star_visibility"
            ]
            audit["failed_checks"] = failed
            audit["passed"] = not failed
            audit["status"] = "passed" if not failed else "failed"
        return audit

    def stage10_compatibility_audit(*args, **kwargs):
        _ = real_stage10_audit
        return compatibility_audit(*args, **kwargs)

    with (
        patch.object(
            stage10_module,
            "audit_display_visibility",
            side_effect=stage10_compatibility_audit,
        ),
        patch.object(
            managed_module,
            "audit_display_visibility",
            side_effect=compatibility_audit,
        ),
    ):
        return _stage10_export_impl(processor)


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


class ReviewRegistryTestDouble:
    def _review_registry(self) -> dict[tuple[int, str], dict[str, Any]]:
        registry = getattr(self, "_review_requirements", None)
        if not isinstance(registry, dict):
            registry = {}
            self._review_requirements = registry
        return registry

    def _clear_stage_reviews(self, stage: int) -> None:
        stage = int(stage)
        self._review_requirements = {
            key: value
            for key, value in self._review_registry().items()
            if key[0] != stage
        }

    def _require_review(
        self,
        stage: int,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        requirement = {
            "stage": int(stage),
            "code": str(code),
            "details": dict(details or {}),
        }
        self._review_registry()[(int(stage), str(code))] = requirement
        return requirement

    def _stage_review_reasons(self, stage: int) -> list[str]:
        return [
            value["code"]
            for key, value in self._review_registry().items()
            if key[0] == int(stage)
        ]

    def _review_requirements_payload(
        self,
        *,
        through_stage: int | None = None,
    ) -> list[dict[str, Any]]:
        limit = int(through_stage) if through_stage is not None else 10
        return [
            dict(value)
            for key, value in sorted(self._review_registry().items())
            if key[0] <= limit
        ]


class Stage3SampleSiril:
    """Small Siril API double with auditable background-sample state."""

    def __init__(self) -> None:
        rng = np.random.default_rng(37)
        height, width = 256, 320
        y, x = np.mgrid[:height, :width]
        self.image = (
            0.04
            + 0.06 * x / (width - 1)
            + 0.025 * y / (height - 1)
            + rng.normal(0.0, 0.001, (height, width))
        ).astype(np.float32)
        self.samples: list[tuple[float, float]] = []
        self.set_calls: list[dict[str, Any]] = []

    def get_image_pixeldata(self, preview: bool = False):
        _ = preview
        return self.image.copy()

    def clear_image_bgsamples(self) -> None:
        self.samples = []

    def set_image_bgsamples(
        self,
        points: list[tuple[float, float]],
        *,
        show_samples: bool = False,
        recalculate: bool = True,
    ) -> bool:
        self.samples = list(points)
        self.set_calls.append(
            {
                "count": len(points),
                "show_samples": show_samples,
                "recalculate": recalculate,
            }
        )
        return True

    def get_image_bgsamples(self):
        return list(self.samples)


class Stage3TransactionFake(ReviewRegistryTestDouble):
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


class Stage3CompoundSiril:
    def __init__(self, owner: "Stage3CompoundFake") -> None:
        self.owner = owner
        self.samples: list[tuple[float, float]] = []
        self.set_calls: list[dict[str, Any]] = []

    def get_image_pixeldata(self, preview: bool = False):
        _ = preview
        return self.owner.images[self.owner.state].copy()

    def clear_image_bgsamples(self) -> None:
        self.samples = []

    def set_image_bgsamples(
        self,
        points: list[tuple[float, float]],
        *,
        show_samples: bool = False,
        recalculate: bool = True,
    ) -> bool:
        self.samples = list(points)
        self.set_calls.append(
            {
                "points": list(points),
                "count": len(points),
                "show_samples": show_samples,
                "recalculate": recalculate,
            }
        )
        return True

    def get_image_bgsamples(self):
        return list(self.samples)


class Stage3CompoundFake(ReviewRegistryTestDouble):
    """Image-aware Stage 3 double for Polynomial→residual-RBF transactions."""

    def __init__(
        self,
        *,
        compound_mode: str = "sufficient",
        external_success: bool = False,
    ) -> None:
        self.log = FakeLogger()
        self.cfg = SimpleNamespace(
            workflow_plugin_probe_enabled=False,
            bg_quality_gate_enabled=True,
            stage3_gate_profile="strict",
        )
        self.pipeline_policy = {
            "policy_name": "test",
            "stage3_background": {},
        }
        self.target_profile: dict[str, Any] = {}
        height, width = 512, 640
        y, x = np.mgrid[:height, :width]
        noise = np.random.default_rng(1).normal(
            0.0,
            0.001,
            (height, width),
        )
        baseline = (
            0.04
            + 0.060 * x / (width - 1)
            + 0.025 * y / (height - 1)
            + noise
        ).astype(np.float32)
        compound = (
            0.0405
            + 0.012 * x / (width - 1)
            + 0.003 * y / (height - 1)
            + noise
        ).astype(np.float32)
        if compound_mode == "validation_rejected":
            compound = compound + np.float32(0.03)
        self.images = {
            "baseline": baseline,
            "single_rbf": (
                0.04
                + 0.040 * x / (width - 1)
                + 0.014 * y / (height - 1)
                + noise
            ).astype(np.float32),
            "polynomial": (
                0.04
                + 0.025 * x / (width - 1)
                + 0.007 * y / (height - 1)
                + noise
            ).astype(np.float32),
            "compound": compound,
            "plugin": (
                0.04
                + 0.010 * x / (width - 1)
                + 0.002 * y / (height - 1)
                + noise
            ).astype(np.float32),
        }
        self.metrics = {
            "baseline": {
                "bg_std": 0.0010,
                "gradient_score": 0.12,
                "dirty_background_score": 0.44,
                "chroma_noise_score": 0.03,
                "red_dominance": 1.0,
                "blue_dominance": 1.0,
                "green_cast": 1.0,
            },
            "single_rbf": {
                "bg_std": 0.0010,
                "gradient_score": 0.10,
                "dirty_background_score": 0.38,
                "chroma_noise_score": 0.06,
                "red_dominance": 1.01,
                "blue_dominance": 1.01,
                "green_cast": 0.99,
            },
            "polynomial": {
                "bg_std": 0.0010,
                "gradient_score": 0.085,
                "dirty_background_score": 0.35,
                "chroma_noise_score": 0.05,
                "red_dominance": 1.01,
                "blue_dominance": 1.01,
                "green_cast": 0.99,
            },
            "compound": {
                "bg_std": 0.0010,
                "gradient_score": 0.06 if compound_mode == "insufficient" else 0.02,
                "dirty_background_score": 0.25 if compound_mode == "insufficient" else 0.12,
                "chroma_noise_score": 0.03 if compound_mode == "insufficient" else 0.02,
                "red_dominance": 1.0,
                "blue_dominance": 1.0,
                "green_cast": 1.0,
            },
            "plugin": {
                "bg_std": 0.0010,
                "gradient_score": 0.015,
                "dirty_background_score": 0.10,
                "chroma_noise_score": 0.02,
                "red_dominance": 1.0,
                "blue_dominance": 1.0,
                "green_cast": 1.0,
            },
        }
        self.compound_mode = compound_mode
        self.external_success = external_success
        self.state = "baseline"
        self.saved_states: dict[str, str] = {}
        self.saved_images: dict[str, np.ndarray] = {}
        self.cmd_calls: list[tuple[Any, ...]] = []
        self.workflow_command_used: dict[str, str] = {}
        self.results: list[tuple[str, str, float, str]] = []
        self.result_metadata: list[dict[str, Any]] = []
        self.report: dict[str, Any] = {}
        self._background_review_required = False
        self.siril = Stage3CompoundSiril(self)

    def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
        _ = quiet
        self.cmd_calls.append(tuple(args))
        command = str(args[0])
        if command == "save":
            stem = str(args[1])
            self.saved_states[stem] = self.state
            self.saved_images[stem] = self.images[self.state].copy()
            return True
        if command == "load":
            stem = str(args[1])
            self.state = self.saved_states[stem]
            self.images[self.state] = self.saved_images[stem].copy()
            return True
        if command == "subsky":
            if "-rbf" in args:
                self.state = (
                    "compound"
                    if self.state == "polynomial"
                    else "single_rbf"
                )
            else:
                self.state = "polynomial"
            return True
        if command in {
            "gxp",
            "graxpert",
            "adbe",
            "dbe",
            "autodbe",
        }:
            if not self.external_success:
                raise pipeline_module.CommandError("mock external failure")
            self.state = "plugin"
            return True
        return True

    def _stage3_subsky_rbf_candidates(self):
        return [
            (
                "subsky",
                "-rbf",
                "-samples=20",
                "-tolerance=1.000",
                "-smooth=0.500",
            )
        ]

    def _stage3_measure_features(self, _label: str):
        return SimpleNamespace(state=self.state)

    def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
        return {
            "available": True,
            "star_retention_ratio": 1.0,
            "nebula_mean_change_ratio": 0.0,
        }

    def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
        return True, "quality gate ok"

    def _adaptive_features_current(self):
        return dict(self.metrics[self.state])

    def _save_stage_output(self, stem: str) -> bool:
        return self.cmd_with_check("save", stem)

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


class FakeProcessor(ReviewRegistryTestDouble):
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
            stage5_multiscale_denoise_enabled=False,
            stage5_multiscale_detail_retention_min=0.82,
            stage5_multiscale_noise_reduction_min=0.05,
            stage5_denoise_chroma_noise_growth_max=1.05,
            asinh_stretch=3.0,
            asinh_offset=0.001,
            ghs_shadowsclip=-2.8,
            ghs_stretchamount=2.0,
            nebula_saturation=0.4,
            nebula_bg_factor=1,
            stage8_bg_std_growth_max=1.08,
            star_intensity=1.0,
            stage9_fallback_intensity_cap=0.95,
            stage9_targeted_recovery_enabled=False,
            stage9_targeted_recovery_retry_max=3,
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
        self.stretched_name = "stage7_stretched"
        self.platesolve_ok = False
        self.image_shape = (3, 1000, 1000)
        rng = np.random.default_rng(913)
        yy, xx = np.mgrid[:96, :128]
        structure = 0.05 + 0.18 * np.exp(
            -(((xx - 64) / 24.0) ** 2 + ((yy - 48) / 18.0) ** 2)
        )
        self.image_pixels = np.clip(
            np.stack(
                (structure, structure * 0.88, structure * 0.72),
                axis=0,
            )
            + rng.normal(0.0, 0.006, size=(3, 96, 128)),
            0.0,
            1.0,
        ).astype(np.float32)
        self.saved_image_pixels: dict[str, np.ndarray] = {}
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
        self._stage8_handoff = {
            "schema": "starun.stage8-handoff.v1",
            "source_stem": "stage8_enhanced",
            "passthrough": False,
            "restricted_downstream": False,
            "reason_code": "stage8_enhancement_accepted",
            "processing_policy": "full",
        }
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
        if cmd == "save" and len(args) >= 2:
            self.saved_image_pixels[str(args[1])] = self.image_pixels.copy()
        elif cmd == "load" and len(args) >= 2:
            saved = self.saved_image_pixels.get(str(args[1]))
            if saved is not None:
                self.image_pixels = saved.copy()
        elif cmd == "denoise":
            padded = np.pad(
                self.image_pixels,
                ((0, 0), (1, 1), (1, 1)),
                mode="reflect",
            )
            smooth = sum(
                padded[:, row : row + 96, col : col + 128]
                for row in range(3)
                for col in range(3)
            ) / 9.0
            self.image_pixels = (
                self.image_pixels * 0.65 + smooth.astype(np.float32) * 0.35
            ).astype(np.float32)
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

    def _run_sasp_star_stretch_api(self):
        """Model the current direct SASP adapter, not removed CLI aliases."""
        label = self.command_labels.get("星点拉伸")
        if not label:
            return None
        self.command_chain_calls.append("星点拉伸")
        self._stage9_sasp_star_stretch_report = {
            "status": "applied",
            "implementation": label,
        }
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
            def _output_path(flag: str, fallback: Path) -> Path:
                try:
                    index = args.index(flag)
                    return Path(args[index + 1])
                except (ValueError, IndexError):
                    return fallback

            def _arg_value(flag: str, default: str) -> str:
                try:
                    return str(args[args.index(flag) + 1])
                except (ValueError, IndexError):
                    return default

            input_path = _output_path(
                "--input-file",
                self.process_dir / f"{self.stretched_name}.fit",
            )
            with fits.open(input_path, memmap=False) as hdul:
                source = np.asarray(hdul[0].data, dtype=np.float32)
            starless = np.clip(source * np.float32(0.90), 0.0, 1.0)
            starmask = np.clip(source - starless, 0.0, 1.0)
            if self.syqon_output_mode in {"starless", "both"}:
                fits.PrimaryHDU(data=starless).writeto(
                    _output_path(
                        "--starless-output",
                        self.process_dir / f"starless_{self.stretched_name}.fit",
                    ),
                    overwrite=True,
                )
            if self.syqon_output_mode == "both":
                fits.PrimaryHDU(data=starmask).writeto(
                    _output_path(
                        "--starmask-output",
                        self.process_dir / f"starmask_{self.stretched_name}.fit",
                    ),
                    overwrite=True,
                )
            if "--manifest-output" in args:
                height, width = source.shape[-2:]
                tile_size = int(_arg_value("--tile-size", "512"))
                overlap = int(_arg_value("--overlap", "64"))
                stretch_method = _arg_value("--stretch-method", "statistical")
                target_median = float(_arg_value("--target-median", "0.15"))
                stat_bp_sigma = float(_arg_value("--stat-bp-sigma", "5.0"))
                mask_method = _arg_value("--mask-method", "subtraction")
                manifest = {
                    "schema": "starun.syqon-worker.v1",
                    "status": "accepted",
                    "model": "zenith",
                    "requested": {
                        "tile_size": tile_size,
                        "overlap": overlap,
                        "use_gpu": "--no_gpu" not in args,
                        "use_amp": "--use-amp" in args,
                        "stretch_method": stretch_method,
                        "target_median": target_median,
                        "linked_stretch": "--linked-stretch" in args,
                        "stat_bp_sigma": stat_bp_sigma,
                        "no_black_clip": "--no-black-clip" in args,
                        "mask_method": mask_method,
                    },
                    "actual": {
                        "mode": "full_frame",
                        "model": "zenith",
                        "tile_size": tile_size,
                        "overlap": overlap,
                        "actual_amp": False,
                        "mask_method": mask_method,
                        "stretch": {
                            "method": stretch_method,
                            "target_median": target_median,
                            "linked": "--linked-stretch" in args,
                            "stat_bp_sigma": stat_bp_sigma,
                            "no_black_clip": "--no-black-clip" in args,
                        },
                        "coverage_min": 1.0,
                        "coverage_max": 1.0,
                        "crop_shape": [height, width],
                        "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "padded_shape": [height, width],
                        "grid": {"rows": 1, "columns": 1, "tiles": 1},
                    },
                    "shadow_metrics": {
                        "transform_roundtrip": {"status": "shadow", "mae": 0.0}
                    },
                }
                _output_path(
                    "--manifest-output",
                    self.process_dir / "worker-manifest.json",
                ).write_text(json.dumps(manifest), encoding="utf-8")
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
        return pipeline_module.StarunPostProcessor._classic_cosmic_clarity_device_args(self)

    def _is_classic_cc_not_configured(self, reason: str):
        return pipeline_module.StarunPostProcessor._is_classic_cc_not_configured(self, reason)

    def _run_siril_cc_sharpen_fallback(self, step_key: str):
        return pipeline_module.StarunPostProcessor._run_siril_cc_sharpen_fallback(
            self,
            step_key,
        )

    def _run_cosmic_clarity_native_sharpen_fallback(self, step_key: str):
        return pipeline_module.StarunPostProcessor._run_cosmic_clarity_native_sharpen_fallback(
            self,
            step_key,
        )

    def _cosmic_clarity_native_sharpen_cli_options(self):
        return pipeline_module.StarunPostProcessor._cosmic_clarity_native_sharpen_cli_options(self)

    def _run_siril_scunet_denoise_fallback(self, step_key: str, strength: float):
        return pipeline_module.StarunPostProcessor._run_siril_scunet_denoise_fallback(
            self,
            step_key,
            strength,
        )

    def _run_cosmic_clarity_native_denoise_fallback(self, step_key: str):
        return pipeline_module.StarunPostProcessor._run_cosmic_clarity_native_denoise_fallback(
            self,
            step_key,
        )

    def _cosmic_clarity_native_denoise_cli_options(self):
        return pipeline_module.StarunPostProcessor._cosmic_clarity_native_denoise_cli_options(self)

    def _syqon_starless_cli_options(
        self,
        *,
        profile=None,
    ):
        if profile is None:
            profile = pipeline_module.syqon_starless.SYQON_BASELINE_PROFILE
        return pipeline_module.syqon_starless.syqon_starless_cli_options(
            self,
            profile=profile,
        )

    def _final_denoise_cli_timeout_sec(self) -> int:
        return pipeline_module.StarunPostProcessor._final_denoise_cli_timeout_sec(self)

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
        return pipeline_module.StarunPostProcessor._result_output_basename(self)

    def _run_ccm_color_fallback(self) -> tuple[bool, str]:
        if self.ccm_fallback_ok:
            return True, self.ccm_fallback_message
        return False, "mock ccm fallback failed"

    def _checkpoint_save(self, name: str, critical: bool = False) -> None:
        _ = critical
        self.checkpoints.append(name)

    def _save_stage_output(self, _stem: str) -> bool:
        self.saved_image_pixels[_stem] = self.image_pixels.copy()
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

    def _final_quality_report(self, _stem: str) -> dict[str, Any]:
        return {
            "final_quality": "ok",
            "status": "ok",
            "needs_conservative_rerun": False,
            "issues": [],
        }

    def _fallback_summary(
        self,
        failed_component: str,
        failure_reason: str,
        fallback_component: str,
        fallback_succeeded: bool,
    ) -> str:
        return pipeline_module.StarunPostProcessor._fallback_summary(
            self,
            failed_component,
            failure_reason,
            fallback_component,
            fallback_succeeded,
        )

    def _is_siril_connection_failure(self, value: object) -> bool:
        return pipeline_module.StarunPostProcessor._is_siril_connection_failure(
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
        return pipeline_module.StarunPostProcessor._feature_summary_note(self, label)

    def _apply_adaptive_edge_crop(self, feat):
        return pipeline_module.StarunPostProcessor._apply_adaptive_edge_crop(self, feat)

    def _apply_weak_object_tuning(self):
        return pipeline_module.StarunPostProcessor._apply_weak_object_tuning(self)

    def _apply_starless_blue_guard(self, feat):
        return pipeline_module.StarunPostProcessor._apply_starless_blue_guard(self, feat)

    def _ai_stage_advisory_enabled(self, _attr_name: str) -> bool:
        return False

    def _request_stage8_processing_plan(self):
        return None

    def _stage8_input_enhancement_guard(self):
        return {
            "skip_enhancement": False,
            "processing_policy": "full",
            "reason_details": [],
        }

    def _apply_stage8_builtin_enhancement(self, plan: dict[str, Any], *, label: str):
        return pipeline_module.StarunPostProcessor._apply_stage8_builtin_enhancement(
            self,
            plan,
            label=label,
        )

    def _stage8_quality_assessment(self):
        return {"status": "ok", "issues": []}

    def _write_stage_json(self, name: str, payload: dict[str, Any]) -> None:
        self.stage_json_reports[name] = payload

    def _stage9_capture_remix_base_identity(self, source_stem: str):
        return pipeline_module.StarunPostProcessor._stage9_capture_remix_base_identity(
            self,
            source_stem,
        )

    def _stage9_verify_remix_base_identity(self, source_stem: str):
        return pipeline_module.StarunPostProcessor._stage9_verify_remix_base_identity(
            self,
            source_stem,
        )

    def _apply_previous_stage_star_remix(self, source_stem: str, starmask_name: str, intensity: float):
        self.previous_stage_remix_calls.append((source_stem, starmask_name, intensity))
        if self.fail_previous_stage_remix:
            return False
        if bool(getattr(self, "_stage9_minimal_fallback_active", False)):
            base = self.saved_image_pixels.get(source_stem)
            if base is None:
                base = self.image_pixels.copy()
            support = np.zeros(base.shape[-2:], dtype=bool)
            center_y = base.shape[-2] // 2
            center_x = base.shape[-1] // 2
            support[
                max(0, center_y - 2) : center_y + 3,
                max(0, center_x - 2) : center_x + 3,
            ] = True
            star_layer = np.zeros_like(base, dtype=np.float32)
            star_layer[..., support] = 0.04
            self.image_pixels = np.clip(
                base + (1.0 - base) * star_layer * float(intensity),
                0.0,
                1.0,
            ).astype(np.float32)
            self._stage9_last_star_layer = star_layer
            self._stage9_last_star_overlay_mask = support
            self._stage9_last_weak_overlay_mask = support.copy()
            self._stage9_last_bright_overlay_mask = np.zeros_like(support)
        return True

    def _stage9_bad_starless_reason(self) -> str:
        return ""

    def _stage9_review_safe_source(self) -> str:
        return "stage7_stretched"

    def _stage8_soften_mask(self, mask, passes: int = 3):
        return pipeline_module.StarunPostProcessor._stage8_soften_mask(
            self,
            mask,
            passes=passes,
        )


_stage5_linear_denoise_impl = stage5_linear_denoise


def stage5_linear_denoise(processor: FakeProcessor) -> None:
    """仅为 Stage 5 测试注入可读写像素缓冲和可测星场。"""
    stars = [
        SimpleNamespace(
            xpos=x,
            ypos=y,
            fwhmx=2.0,
            fwhmy=2.2,
            A=0.5,
            B=0.01,
            rmse=0.001,
            sat=1.0,
            has_saturated=False,
        )
        for x, y in (
            (16, 16),
            (48, 16),
            (80, 16),
            (112, 16),
            (16, 80),
            (112, 80),
        )
    ]
    if not callable(getattr(processor.siril, "get_image_stars", None)):
        processor.siril.get_image_stars = lambda: list(stars)
    if not bool(getattr(processor, "_stage5_test_star_field_installed", False)):
        yy, xx = np.mgrid[
            : processor.image_pixels.shape[-2],
            : processor.image_pixels.shape[-1],
        ]
        star_signal = np.zeros_like(yy, dtype=np.float32)
        for star in stars:
            star_signal += 0.28 * np.exp(
                -(
                    (xx - float(star.xpos)) ** 2
                    + (yy - float(star.ypos)) ** 2
                )
                / (2.0 * (2.1 / 2.355) ** 2)
            ).astype(np.float32)
        processor.image_pixels = np.clip(
            processor.image_pixels + star_signal[None, :, :],
            0.0,
            1.0,
        ).astype(np.float32)
        processor._stage5_test_star_field_installed = True
    processor.siril.get_image_pixeldata = (
        lambda preview=False: processor.image_pixels.copy()
    )
    processor.siril.set_image_pixeldata = lambda image: setattr(
        processor,
        "image_pixels",
        np.array(image, copy=True),
    )
    processor._set_current_image_pixeldata = lambda image, **_kwargs: setattr(
        processor,
        "image_pixels",
        np.array(image, copy=True),
    )
    empty_mask = np.zeros(processor.image_pixels.shape[-2:], dtype=np.float32)
    with patch.object(
        stage5_linear_denoise_module.stage8_pixels,
        "build_signal_excluded_background_masks",
        return_value=(
            {
                "core_mask": empty_mask,
                "nebula_mask": empty_mask,
                "faint_nebula_mask": empty_mask,
                "galaxy_signal_mask": empty_mask,
            },
            {"status": "test_fixture", "reason": "isolated star field"},
        ),
    ):
        _stage5_linear_denoise_impl(processor)


class PipelinePluginFallbackTestBase(unittest.TestCase):
    def _new_processor(self) -> FakeProcessor:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return FakeProcessor(pipeline_module, Path(td.name))

    def _dualband_palette_processor(
        self,
        *,
        requested_palette: str,
    ) -> FakeProcessor:
        processor = self._new_processor()
        processor._channel_semantics = "narrowband_composite"
        mapping = resolve_dual_narrowband_mapping(
            {"FILTER": "Ha OIII dual-band"}
        )
        processor.narrowband_channel_mapping = mapping
        processor.channel_profile = {
            "kind": "narrowband_composite",
            "filter_hint": "Ha OIII dual-band",
            "narrowband_mapping": mapping,
        }
        processor._stage4_header_metadata = {"FILTER": "Ha OIII dual-band"}
        processor._stage7_stretch_accepted = True
        processor._stage8_final_quality = "ok"
        processor._frozen_primary_target = {
            "type": "emission_nebula_widefield",
            "confidence": 0.95,
            "method": "catalog",
            "frozen": True,
        }
        processor._stage8_palette_selection = resolve_palette_selection(
            processor._frozen_primary_target,
            requested_palette,
        )
        processor.color_calibration_report = {
            "method": "PCC_NARROWBAND_DEGRADED",
            "requires_review": True,
            "physical_color": {"accepted": False},
            "degraded_color_correction": {"applied": True},
        }
        default_cfg = pipeline_module.PipelineConfig()
        self.assertTrue(default_cfg.stage8_dualband_palette_enabled)
        processor.cfg.stage8_dualband_palette_enabled = (
            default_cfg.stage8_dualband_palette_enabled
        )
        processor.cfg.stage8_dualband_palette_selection = requested_palette
        processor.cfg.stage8_dualband_palette_strength = 0.85
        processor.cfg.stage8_dualband_palette_luma_drift_max = 0.005
        processor.cfg.stage8_dualband_palette_clip_growth_max = 0.002
        processor.cfg.stage8_dualband_palette_quality_warning_tolerance = 0.50
        processor.cfg.optional_color_transform_enabled = False
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"mock")

        height, width = processor.image_pixels.shape[1:]
        subject = np.zeros((height, width), dtype=np.float32)
        subject[20:76, 24:104] = 1.0
        background = 1.0 - subject
        processor._stage8_generate_starless_masks = lambda _image: {
            "core_mask": np.zeros_like(subject),
            "nebula_mask": subject,
            "faint_nebula_mask": np.zeros_like(subject),
            "background_mask": background,
        }
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        processor._set_current_image_pixeldata = lambda image, **_kwargs: setattr(
            processor,
            "image_pixels",
            np.array(image, copy=True),
        )
        return processor

    @staticmethod
    def _stage5_psf_stars() -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                xpos=x,
                ypos=y,
                fwhmx=2.0,
                fwhmy=2.2,
                A=0.5,
                B=0.01,
                rmse=0.001,
                sat=1.0,
                has_saturated=False,
            )
            for x, y in (
                (16, 16),
                (48, 16),
                (80, 16),
                (112, 16),
                (16, 80),
                (112, 80),
            )
        ]

    @staticmethod
    def _stage10_final_input(processor: FakeProcessor) -> None:
        (processor.process_dir / "stage9_remixed.fit").write_bytes(b"mock")
        pixels = np.full((3, 32, 32), 0.12, dtype=np.float32)
        coordinates = tuple(
            (y, x)
            for y in (5, 12, 19, 26)
            for x in (5, 12, 19, 26)
        )
        for index, (y, x) in enumerate(coordinates):
            peak = np.float32(0.42 if index < 8 else 0.62)
            pixels[:, y, x] = peak
            pixels[:, y - 1 : y + 2, x - 1 : x + 2] = np.maximum(
                pixels[:, y - 1 : y + 2, x - 1 : x + 2],
                peak * np.float32(0.45),
            )
            pixels[:, y, x] = peak
        state = {"pixels": pixels}
        processor.siril.get_image_pixeldata = (
            lambda preview=False: state["pixels"].copy()
        )
        processor._set_current_image_pixeldata = (
            lambda image, **_kwargs: state.__setitem__(
                "pixels",
                np.array(image, copy=True),
            )
        )
        weak_core = np.zeros((32, 32), dtype=bool)
        bright_core = np.zeros((32, 32), dtype=bool)
        weak_core[8, 9] = True
        weak_core[23, 6] = True
        bright_core[17, 22] = True
        y = np.asarray([item[0] for item in coordinates], dtype=np.int32)
        x = np.asarray([item[1] for item in coordinates], dtype=np.int32)
        weak_flags = np.asarray(
            [index < 8 for index in range(len(coordinates))],
            dtype=bool,
        )
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_output_contains_stars = True
        processor._stage9_star_reference_catalog = {
            "status": "ok",
            "source_matched": True,
            "_weak_core_mask": weak_core,
            "_bright_core_mask": bright_core,
            "_source_peak_y": y,
            "_source_peak_x": x,
            "_peak_y": y.copy(),
            "_peak_x": x.copy(),
            "_weak_flags": weak_flags,
            "_reference_local_contrast": np.full(
                len(coordinates),
                0.30,
                dtype=np.float32,
            ),
            "_stage9_visibility_inner_window_size_px": np.full(
                len(coordinates),
                3,
                dtype=np.int32,
            ),
            "_stage9_visibility_outer_window_size_px": np.full(
                len(coordinates),
                7,
                dtype=np.int32,
            ),
        }

    @staticmethod
    def _stage10_quality_noise_report(
        *,
        chroma: float,
        hard: bool,
    ) -> dict[str, Any]:
        hard_issues = (
            [f"background_chroma_noise_extreme {chroma:.3f}>0.900"]
            if hard
            else []
        )
        return {
            "schema": "starun.final-quality.v2",
            "severity": "hard_reject" if hard else "soft_warning",
            "status": "needs_conservative_rerun" if hard else "ok",
            "final_quality": "poor" if hard else "ok",
            "needs_conservative_rerun": hard,
            "hard_issues": hard_issues,
            "issues": list(hard_issues),
            "warnings": [],
            "advisories": [],
            "metrics": {
                "background_chroma_noise_score": chroma,
                "background_mottling_score": 0.10,
                "starless_artifact_score": 0.10,
                "local_texture_residual_outlier_score": 0.1,
                "local_texture_affected_patch_ratio": 0.0,
                "noise_gate_limits": {
                    "chroma_advisory_max": 0.42,
                    "mottling_advisory_max": 0.55,
                    "artifact_advisory_max": 0.62,
                    "texture_outlier_score_hard_max": 4.0,
                    "texture_affected_ratio_hard_max": 0.35,
                },
            },
        }

    @staticmethod
    def _synthetic_galaxy_starless_layers(
        *,
        disk_halo_amplitude: float = 0.0,
        core_damage: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height = width = 256
        yy, xx = np.mgrid[:height, :width]
        dx = xx.astype(np.float32) - 128.0
        dy = yy.astype(np.float32) - 128.0
        angle = np.deg2rad(35.0)
        major = dx * np.cos(angle) + dy * np.sin(angle)
        minor = -dx * np.sin(angle) + dy * np.cos(angle)
        radius2 = (major / 72.0) ** 2 + (minor / 28.0) ** 2
        background = 0.012 + 0.003 * xx / width + 0.002 * yy / height
        disk = 0.055 * np.exp(-1.5 * radius2)
        bulge = 0.30 * np.exp(-((major / 14.0) ** 2 + (minor / 10.0) ** 2))
        theta = np.arctan2(minor / 28.0, major / 72.0)
        arms = (
            0.010
            * np.maximum(np.sin(14.0 * np.sqrt(radius2) + 2.0 * theta), 0.0)
            * np.exp(-1.6 * radius2)
        )
        galaxy = background + disk + bulge + arms
        source = galaxy.copy()
        starmask = np.zeros_like(galaxy)
        stars = (
            (70, 80, 0.20),
            (115, 145, 0.16),
            (145, 158, 0.22),
            (178, 112, 0.18),
            (44, 205, 0.24),
            (210, 40, 0.18),
            (90, 190, 0.14),
        )
        for center_y, center_x, amplitude in stars:
            star_radius2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
            star = amplitude * np.exp(-star_radius2 / 3.0)
            halo = amplitude * 0.12 * np.exp(-star_radius2 / 45.0)
            source += star + halo
            starmask += star + halo

        starless = galaxy.copy()
        if disk_halo_amplitude > 0.0:
            halo_radius2 = (xx - 158) ** 2 + (yy - 145) ** 2
            starless += disk_halo_amplitude * np.exp(-halo_radius2 / 45.0)
        if core_damage:
            core_hole = np.exp(
                -((major / 13.0) ** 2 + (minor / 9.0) ** 2)
            )
            starless = np.clip(starless - 0.22 * core_hole, 0.0, None)

        def rgb(layer: np.ndarray) -> np.ndarray:
            return np.repeat(layer[None, :, :], 3, axis=0).astype(np.float32)

        return rgb(source), rgb(starless), rgb(starmask)

    def _copy_spcc_database(self, processor: FakeProcessor) -> Path:
        target = processor.work_dir / "siril-spcc-database"
        shutil.copytree(processor.spcc_database_dir, target)
        processor.spcc_database_dir = target
        return target


__all__ = [name for name in globals() if not name.startswith("__")]
