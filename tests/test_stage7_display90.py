#!/usr/bin/env python3
"""Regression tests for the Stage 7 Display90 candidate and Stage 9 handoff."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits


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

    package.SirilInterface = object
    exceptions.SirilError = SirilError
    exceptions.CommandError = type("CommandError", (SirilError,), {})
    exceptions.DataError = type("DataError", (SirilError,), {})
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions


_install_fake_sirilpy()

import stage7_stretch_metrics as stretch_metrics  # noqa: E402
import stage8_pixels  # noqa: E402
import ui_preview  # noqa: E402
from models import PipelineConfig, QualityMetrics, StarSeparationState  # noqa: E402
from processing_parameters import (  # noqa: E402
    SPECS_BY_FIELD,
    default_processing_parameters,
    normalize_processing_parameters,
)
from stage6_services import (  # noqa: E402
    Stage6ServiceMixin,
    _stage7_candidates_for_policy,
    _stage7_matched_domain_transfer_contract,
)
from stages import stage9_star_remixing  # noqa: E402


def _linear_rgb(height: int = 96, width: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    gradient = 0.001 + 0.032 * (xx.astype(np.float32) / max(width - 1, 1))
    subject = 0.018 * np.exp(
        -(((xx - width * 0.55) / (width * 0.22)) ** 2)
        - (((yy - height * 0.48) / (height * 0.20)) ** 2)
    )
    base = gradient + subject.astype(np.float32)
    return np.stack((base * 1.08, base, base * 0.82), axis=0).astype(np.float32)


def _pixel_stats(image: np.ndarray) -> dict[str, float]:
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    p01, p50, p90, p99 = np.percentile(flat, (1.0, 50.0, 90.0, 99.0))
    return {
        "min": float(np.min(flat)),
        "p01": float(p01),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(np.max(flat)),
    }


def _calibration(image: np.ndarray, strength: float = 0.90) -> dict:
    curve = ui_preview.build_linked_display_curve_contract(image)
    return stretch_metrics.calibrate_display90_linked_lut(
        image,
        curve,
        strength=strength,
        max_derivative=5000.0,
    )


def _eligible_calibration(image: np.ndarray, strength: float = 0.90) -> dict:
    calibration = _calibration(image, strength)
    calibration["eligibility"] = {
        "policy_requested": True,
        "automatic_parameter_mode": True,
        "starless_recomposition_planned": True,
        "source_stem": "stage6_starless",
        "star_separation_state": StarSeparationState.ACCEPTED.value,
        "baseline_pixels_available": True,
    }
    return calibration


def _display90_reference_evidence(
    reference_load: float,
    *,
    channel_semantics: str = "narrowband_composite",
    physical_accepted: bool = True,
    physical_method: str = "SPCC_NARROWBAND",
    signal_exclusion_applied: bool = True,
    curve_authenticated: bool = True,
    non_background_hard_gates_accepted: bool = True,
) -> dict:
    return {
        "schema": "starun.stage7-display90-background-reference.v1",
        "status": "available",
        "applicable": True,
        "matched": False,
        "reason_code": "gui_linked_reference_measured",
        "candidate_name": "cand_display90",
        "candidate_method": "display90_linked_lut",
        "source_stem": "stage6_starless",
        "automatic_accepted_starless_route": True,
        "channel_semantics": channel_semantics,
        "physical_calibration_accepted": physical_accepted,
        "physical_calibration_method": physical_method,
        "curve_conformance_accepted": True,
        "curve_authenticated": curve_authenticated,
        "non_background_hard_gates_accepted": (
            non_background_hard_gates_accepted
        ),
        "signal_exclusion_applied": signal_exclusion_applied,
        "signal_exclusion_keys": ["nebula_mask"],
        "mask_scope": "stage6_frozen_signal_excluded_background_mask",
        "reference_application": "gui_rec709_luminance_gain",
        "candidate_application": "linked_rgb_common_lut",
        "lut_sha256": "a" * 64,
        "reference_chroma_load": reference_load,
        "reference_metrics": {
            "background_chroma_load": reference_load,
            "signal_exclusion_applied": signal_exclusion_applied,
        },
    }


class _StretchProcessor(Stage6ServiceMixin):
    def __init__(self) -> None:
        self.cfg = PipelineConfig()
        self.pipeline_policy = {}
        self._task_manual_override_fields = ()
        self._channel_semantics = "narrowband_composite"
        self.color_calibration_report = {
            "method": "SPCC_NARROWBAND",
            "physical_color": {
                "accepted": True,
                "method": "SPCC_NARROWBAND",
            },
        }
        self.log = types.SimpleNamespace(warn=lambda _message: None)

    @staticmethod
    def _active_target_type() -> str:
        return "bright_emission_reflection_nebula"

    @staticmethod
    def _short_text(value: object, limit: int) -> str:
        return str(value)[:limit]

    def _stage8_soften_mask(
        self,
        mask: np.ndarray,
        passes: int = 3,
    ) -> np.ndarray:
        return stage8_pixels.stage8_soften_mask(self, mask, passes=passes)

    def _stage8_generate_starless_masks(
        self,
        image: np.ndarray,
    ) -> dict:
        return stage8_pixels.stage8_generate_starless_masks(self, image)

    def _background_quality_metrics(
        self,
        image: np.ndarray,
        masks: dict | None = None,
    ) -> dict:
        return stage8_pixels.background_quality_metrics(self, image, masks)


class Display90CurveTests(unittest.TestCase):
    def test_gui_reference_rebuilds_exact_existing_linked_display(self) -> None:
        image = _linear_rgb()
        calibration = _calibration(image)
        reference, contract = (
            stretch_metrics.build_display90_gui_linked_reference(
                image,
                calibration,
            )
        )
        expected = ui_preview.apply_linked_display_curve_contract(
            image,
            calibration["display_curve"],
        )

        np.testing.assert_array_equal(reference, expected)
        self.assertEqual(
            contract["sha256"],
            calibration["lut_contract"]["sha256"],
        )

    def test_formula_lut_contract_and_curve_conformance(self) -> None:
        image = _linear_rgb()
        calibration = _calibration(image)

        self.assertEqual(calibration["status"], "ok", calibration)
        self.assertEqual(
            calibration["schema"],
            stretch_metrics.DISPLAY90_STRETCH_SCHEMA_V2,
        )
        self.assertEqual(
            calibration["method"],
            stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD,
        )
        lut, contract = stretch_metrics.rebuild_display90_linked_lut(calibration)
        grid = np.linspace(0.0, 1.0, lut.size, dtype=np.float64)
        curve = calibration["display_curve"]
        display = np.power(
            np.clip(
                (grid - curve["black"])
                / (curve["white"] - curve["black"]),
                0.0,
                1.0,
            ),
            curve["gamma"],
        )
        expected = 0.10 * grid + 0.90 * display

        np.testing.assert_array_equal(lut, expected.astype(np.float32))
        self.assertEqual(lut.size, 65536)
        self.assertEqual(float(lut[0]), 0.0)
        self.assertEqual(float(lut[-1]), 1.0)
        self.assertTrue(contract["monotonic"])
        self.assertLessEqual(
            contract["maximum_derivative"],
            contract["maximum_derivative_limit"],
        )

        mapped = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            calibration,
        )
        conformance = stretch_metrics.assess_display90_curve_conformance(
            calibration,
            mapped,
        )
        self.assertTrue(conformance["accepted"], conformance)

        drifted = np.clip(mapped * 0.70, 0.0, 0.995)
        rejected = stretch_metrics.assess_display90_curve_conformance(
            calibration,
            drifted,
        )
        self.assertFalse(rejected["accepted"], rejected)

    def test_v2_luminance_lut_preserves_rgb_direction_and_uniform_gamut(self) -> None:
        image = _linear_rgb()
        image[:, 20:30, 40:50] = np.asarray(
            [0.92, 0.18, 0.04],
            dtype=np.float32,
        )[:, None, None]
        calibration = _calibration(image, strength=0.95)
        mapped = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            calibration,
        )

        source_sum = np.sum(image, axis=0)
        mapped_sum = np.sum(mapped, axis=0)
        support = (source_sum > 1e-8) & (mapped_sum > 1e-8)
        source_direction = image[:, support] / source_sum[support]
        mapped_direction = mapped[:, support] / mapped_sum[support]
        np.testing.assert_allclose(
            mapped_direction,
            source_direction,
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertLessEqual(float(np.max(mapped)), 0.995001)

    def test_compressed_pedestal_opens_shadow_shoulder_without_median_drift(self) -> None:
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[:192, :256]
        pedestal = 0.02578 + rng.normal(0.0, 0.00005, size=xx.shape)
        galaxy = 0.008 * np.exp(
            -(((xx - 132.0) / 46.0) ** 2)
            - (((yy - 98.0) / 20.0) ** 2)
        )
        base = np.asarray(pedestal + galaxy, dtype=np.float32)
        image = np.stack(
            (base * 1.015, base, base * 0.985),
            axis=0,
        ).astype(np.float32)
        source_curve = ui_preview.build_linked_display_curve_contract(image)
        self.assertLess(source_curve["median_normalized"], 0.08)

        calibration = _calibration(image)
        self.assertEqual(calibration["status"], "ok", calibration)
        protection = calibration["shadow_protection"]
        self.assertEqual(protection["status"], "applied", protection)
        self.assertLess(
            protection["source_below_low_input_ratio"],
            0.0001,
        )
        self.assertLess(protection["low_input"], protection["join_input"])
        self.assertLess(protection["low_output"], protection["join_output"])

        mapped = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            calibration,
        )
        mapped_luminance = (
            0.2126 * mapped[0]
            + 0.7152 * mapped[1]
            + 0.0722 * mapped[2]
        )
        p0_2, p50 = np.percentile(mapped_luminance, (0.2, 50.0))
        self.assertGreater(float(p0_2 / p50), 0.20)
        expected_p50 = float(calibration["predicted_p50"])
        self.assertAlmostEqual(float(p50), expected_p50, delta=0.002)

    def test_large_galaxy_compressed_pedestal_uses_bounded_quantile_tail(self) -> None:
        rng = np.random.default_rng(17)
        yy, xx = np.mgrid[:240, :320]
        pedestal = 0.02578 + rng.normal(0.0, 0.00005, size=xx.shape)
        galaxy = 0.030 * np.exp(
            -(((xx - 164.0) / 62.0) ** 2)
            - (((yy - 122.0) / 25.0) ** 2)
        )
        base = np.asarray(pedestal + galaxy, dtype=np.float32)
        image = np.stack(
            (base * 1.015, base, base * 0.985),
            axis=0,
        ).astype(np.float32)
        curve = ui_preview.build_linked_display_curve_contract(image)

        calibration = stretch_metrics.calibrate_display90_linked_lut(
            image,
            curve,
            strength=0.90,
            max_derivative=5000.0,
            target_type="large_galaxy",
        )

        self.assertEqual(calibration["status"], "ok", calibration)
        tone = calibration["quantile_tone_curve"]
        self.assertEqual(tone["status"], "applied", tone)
        self.assertEqual(
            calibration["shadow_protection"]["policy"],
            "large_galaxy_quantile_curve_owns_shadows",
        )
        mapped = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            calibration,
        )
        luminance = (
            0.2126 * mapped[0]
            + 0.7152 * mapped[1]
            + 0.0722 * mapped[2]
        )
        p10, p50, p90, p99 = np.percentile(
            luminance,
            (10.0, 50.0, 90.0, 99.0),
        )
        self.assertAlmostEqual(float(p10), 0.134, delta=0.004)
        self.assertAlmostEqual(float(p50), 0.165, delta=0.004)
        self.assertAlmostEqual(float(p90), 0.330, delta=0.006)
        self.assertAlmostEqual(float(p99), 0.700, delta=0.010)
        self.assertLess(float(np.mean(luminance >= 0.995)), 0.001)
        conformance = stretch_metrics.assess_display90_curve_conformance(
            calibration,
            mapped,
        )
        self.assertTrue(conformance["accepted"], conformance)

    def test_legacy_v1_calibration_replays_per_channel_semantics(self) -> None:
        image = _linear_rgb()
        calibration = _calibration(image)
        legacy = copy.deepcopy(calibration)
        legacy["schema"] = stretch_metrics.DISPLAY90_STRETCH_SCHEMA_V1
        legacy["method"] = stretch_metrics.DISPLAY90_LEGACY_METHOD
        legacy.pop("application_contract", None)

        lut, _contract = stretch_metrics.rebuild_display90_linked_lut(legacy)
        replayed = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            legacy,
        )
        expected = stretch_metrics._apply_authenticated_lut_rgb(image, lut)
        np.testing.assert_array_equal(replayed, expected)
        self.assertFalse(
            np.array_equal(
                replayed,
                stretch_metrics.apply_display90_linked_rgb_stretch(
                    image,
                    calibration,
                ),
            )
        )

    def test_strength_bounds_and_digest_tampering_fail_closed(self) -> None:
        image = _linear_rgb()
        for strength in (0.50, 0.95):
            self.assertEqual(_calibration(image, strength)["status"], "ok")
        for strength in (0.49, 0.96):
            self.assertEqual(
                _calibration(image, strength)["status"],
                "unavailable",
            )
        curve = ui_preview.build_linked_display_curve_contract(image)
        derivative_rejected = stretch_metrics.calibrate_display90_linked_lut(
            image,
            curve,
            strength=0.90,
            max_derivative=1.0,
        )
        self.assertEqual(derivative_rejected["status"], "unavailable")
        self.assertIn("derivative", derivative_rejected["reason"])

        calibration = _calibration(image)
        tampered_strength = copy.deepcopy(calibration)
        tampered_strength["parameters"]["strength"] = 0.80
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            stretch_metrics.rebuild_display90_linked_lut(tampered_strength)

        tampered_digest = copy.deepcopy(calibration)
        tampered_digest["lut_contract"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            stretch_metrics.rebuild_display90_linked_lut(tampered_digest)

        tampered_headroom = copy.deepcopy(calibration)
        tampered_headroom["parameters"]["output_headroom"] = 1.0
        with self.assertRaisesRegex(ValueError, "headroom contract mismatch"):
            stretch_metrics.rebuild_display90_linked_lut(tampered_headroom)

        tampered_summary = copy.deepcopy(calibration)
        tampered_summary["lut_contract"]["maximum_derivative_limit"] *= 2.0
        with self.assertRaisesRegex(ValueError, "summary mismatch"):
            stretch_metrics.rebuild_display90_linked_lut(tampered_summary)

    def test_ngc6888_default_strength_regression(self) -> None:
        source_path = Path(
            "/Users/mz/SeeStar/Starun/Starun/ngc6888-15a8abd195d9/"
            "runs/20260813T145916Z-79662361/process/stage6_starless.fit"
        )
        if not source_path.is_file():
            self.skipTest("user-provided NGC6888 Stage6 FITS is unavailable")
        image = fits.getdata(source_path, memmap=False)
        calibration = _calibration(image)
        targets = calibration["d90_target_quantiles"]["rec709_luminance"]

        self.assertAlmostEqual(targets["p50"], 0.189, delta=0.002)
        self.assertAlmostEqual(targets["p90"], 0.401, delta=0.002)
        self.assertAlmostEqual(targets["p99"], 0.608, delta=0.002)
        mapped = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            calibration,
        )
        self.assertEqual(float(np.mean(mapped <= 1e-5)), 0.0)
        self.assertEqual(float(np.mean(mapped >= 0.995)), 0.0)

        processor = _StretchProcessor()
        frozen_masks, _mask_report = (
            stage8_pixels.build_signal_excluded_background_masks(
                processor,
                image,
            )
        )
        candidate_masks = dict(frozen_masks)
        generated = processor._stage8_generate_starless_masks(mapped)
        for key in ("core_mask", "nebula_mask", "faint_nebula_mask"):
            candidate_masks[key] = generated[key]
        candidate_metrics = processor._background_quality_metrics(
            mapped,
            candidate_masks,
        )
        gui_reference, _contract = (
            stretch_metrics.build_display90_gui_linked_reference(
                image,
                calibration,
            )
        )
        reference_metrics = processor._background_quality_metrics(
            gui_reference,
            candidate_masks,
        )
        candidate_load = float(candidate_metrics["background_chroma_load"])
        reference_load = float(reference_metrics["background_chroma_load"])
        self.assertAlmostEqual(candidate_load, 0.365, delta=0.015)
        self.assertAlmostEqual(reference_load, 0.368, delta=0.015)
        self.assertLessEqual(candidate_load / reference_load, 1.05)


class Display90BackgroundReferenceGateTests(unittest.TestCase):
    @staticmethod
    def _candidate_metrics(
        chroma_load: float,
        *,
        chroma_noise: float = 0.07,
        mottling: float = 0.05,
        signal_exclusion_applied: bool = True,
    ) -> dict:
        return {
            "background_chroma_load": chroma_load,
            "chroma_noise_score": chroma_noise,
            "background_mottling_score": mottling,
            "signal_exclusion_applied": signal_exclusion_applied,
            "bg_median": 0.10,
            "bg_std": 0.01,
            "background_red_mean": 0.08,
            "background_green_mean": 0.11,
            "background_blue_mean": 0.11,
            "background_green_excess": 0.15,
        }

    def _gate(
        self,
        candidate_load: float,
        reference_load: float,
        *,
        evidence: dict | None = None,
        candidate_name: str = "cand_display90",
        candidate_method: str = "display90_linked_lut",
        chroma_noise: float = 0.07,
        mottling: float = 0.05,
        signal_exclusion_applied: bool = True,
    ) -> dict:
        processor = _StretchProcessor()
        baseline = {
            "background_chroma_load": 0.05,
            "bg_median": 0.001,
        }
        return processor._stage7_stretch_background_gate(
            baseline,
            self._candidate_metrics(
                candidate_load,
                chroma_noise=chroma_noise,
                mottling=mottling,
                signal_exclusion_applied=signal_exclusion_applied,
            ),
            candidate_name=candidate_name,
            candidate_method=candidate_method,
            display90_reference=(
                evidence
                if evidence is not None
                else _display90_reference_evidence(reference_load)
            ),
        )

    def test_reference_measurement_uses_authenticated_curve_and_same_mask(
        self,
    ) -> None:
        image = _linear_rgb()
        calibration = _eligible_calibration(image)
        candidate = {
            "name": "cand_display90",
            "method": "display90_linked_lut",
            "params": {"calibration": calibration},
        }
        mapped = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            calibration,
        )
        curve_quality = stretch_metrics.assess_display90_curve_conformance(
            calibration,
            mapped,
        )
        shape = tuple(image.shape[1:])
        masks = {
            "background_mask": np.ones(shape, dtype=np.float32),
            "nebula_mask": np.zeros(shape, dtype=np.float32),
        }
        processor = _StretchProcessor()
        evidence = processor._stage7_display90_background_reference(
            candidate,
            source_stem="stage6_starless",
            baseline_image_data=image,
            background_masks=masks,
            signal_exclusion_applied=True,
            curve_quality=curve_quality,
            non_background_hard_gates_accepted=True,
        )

        self.assertEqual(evidence["status"], "available", evidence)
        self.assertTrue(evidence["curve_authenticated"])
        self.assertEqual(
            evidence["mask_scope"],
            "stage6_frozen_signal_excluded_background_mask",
        )
        self.assertEqual(
            evidence["lut_sha256"],
            calibration["lut_contract"]["sha256"],
        )
        self.assertGreater(evidence["reference_chroma_load"], 0.0)

        tampered = copy.deepcopy(candidate)
        tampered["params"]["calibration"]["lut_contract"]["sha256"] = (
            "f" * 64
        )
        rejected = processor._stage7_display90_background_reference(
            tampered,
            source_stem="stage6_starless",
            baseline_image_data=image,
            background_masks=masks,
            signal_exclusion_applied=True,
            curve_quality=curve_quality,
            non_background_hard_gates_accepted=True,
        )
        self.assertEqual(rejected["status"], "unavailable")
        self.assertFalse(rejected["curve_authenticated"])

    def test_ratio_and_absolute_boundaries_become_one_advisory(self) -> None:
        candidate_load = 0.30
        result = self._gate(candidate_load, candidate_load / 1.05)

        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["status"], "advisory")
        self.assertEqual(len(result["advisories"]), 1)
        reference = result["display90_reference_match"]
        self.assertEqual(reference["status"], "matched_advisory")
        self.assertTrue(reference["matched"])
        self.assertTrue(
            reference["nominal_growth_gate"]["hard_failed"],
            reference,
        )
        growth_gate = result["quality_gates"][
            "background_chroma_load_growth"
        ]
        self.assertFalse(growth_gate["hard_failed"])
        self.assertTrue(growth_gate["advisory"])
        self.assertTrue(growth_gate["nominal_gate"]["hard_failed"])
        self.assertTrue(
            result["metrics"][
                "chroma_load_growth_display_reference_exempted"
            ]
        )
        self.assertAlmostEqual(
            result["metrics"]["display_reference_chroma_load_ratio"],
            1.05,
        )

    def test_reference_ratio_or_absolute_limit_excess_remains_hard(self) -> None:
        ratio_failed = self._gate(0.29, 0.29 / 1.051)
        self.assertFalse(ratio_failed["accepted"])
        self.assertEqual(
            ratio_failed["display90_reference_match"]["reason_code"],
            "display90_reference_chroma_ratio_exceeded",
        )

        absolute_failed = self._gate(0.301, 0.301)
        self.assertFalse(absolute_failed["accepted"])
        self.assertEqual(
            absolute_failed["display90_reference_match"]["reason_code"],
            "display90_reference_absolute_chroma_load_exceeded",
        )

    def test_context_and_evidence_fail_closed(self) -> None:
        variants = {
            "broadband": _display90_reference_evidence(
                0.38,
                channel_semantics="broadband_rgb",
            ),
            "uncalibrated": _display90_reference_evidence(
                0.38,
                physical_accepted=False,
            ),
            "wrong_method": _display90_reference_evidence(
                0.38,
                physical_method="SPCC",
            ),
            "missing_signal_exclusion": _display90_reference_evidence(
                0.38,
                signal_exclusion_applied=False,
            ),
            "unauthenticated_curve": _display90_reference_evidence(
                0.38,
                curve_authenticated=False,
            ),
            "other_gate_failed": _display90_reference_evidence(
                0.38,
                non_background_hard_gates_accepted=False,
            ),
        }
        for name, evidence in variants.items():
            with self.subTest(name=name):
                result = self._gate(0.39, 0.38, evidence=evidence)
                self.assertFalse(result["accepted"], result)
                self.assertFalse(
                    result["metrics"][
                        "chroma_load_growth_display_reference_exempted"
                    ]
                )

        candidate_a = self._gate(
            0.39,
            0.38,
            candidate_name="cand_a",
            candidate_method="asinh",
        )
        self.assertFalse(candidate_a["accepted"])
        self.assertFalse(
            candidate_a["metrics"][
                "chroma_load_growth_display_reference_exempted"
            ]
        )

    def test_reference_match_never_bypasses_noise_or_mottling_hard_gate(
        self,
    ) -> None:
        noisy = self._gate(0.39, 0.39, chroma_noise=0.60)
        self.assertFalse(noisy["accepted"])
        self.assertIn(
            "background_chroma_noise_score",
            " ".join(noisy["issues"]),
        )
        self.assertFalse(
            noisy["metrics"][
                "chroma_load_growth_display_reference_exempted"
            ]
        )

        mottled = self._gate(0.39, 0.39, mottling=0.70)
        self.assertFalse(mottled["accepted"])
        self.assertIn(
            "background_mottling_score",
            " ".join(mottled["issues"]),
        )

    def test_v7_continuous_quality_can_override_retention_advantage(self) -> None:
        processor = _StretchProcessor()
        display_gate = self._gate(0.39, 0.39)
        safe_gate = processor._stage7_stretch_background_gate(
            {"background_chroma_load": 0.05, "bg_median": 0.001},
            self._candidate_metrics(0.06),
            candidate_name="cand_b",
            candidate_method="linked_mtf",
        )
        self.assertEqual(
            safe_gate["display90_reference_match"]["status"],
            "not_applicable",
        )
        self.assertFalse(
            safe_gate["display90_reference_match"]["quality_gates"][
                "candidate_to_gui_reference_ratio"
            ]["hard_failed"]
        )

        def attempt(
            name: str,
            gate: dict,
            retention: float,
        ) -> dict:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "diagnostics": [],
                "advisories": gate.get("advisories") or [],
                "risk_score": 0.1,
                "background_quality_gate": gate,
                "preview_target_attainment": {
                    "attainment_ratio": 1.0,
                    "minimum_ratio": 0.90,
                    "hard_minimum_ratio": 0.80,
                    "maximum_ratio": 1.50,
                    "hard_maximum_ratio": 2.25,
                },
                "preview_retention": {
                    "metrics": {
                        key: {
                            "available": True,
                            "ranking_ratio": retention,
                        }
                        for key in (
                            "visibility",
                            "object_signal",
                            "subject_span",
                            "saturation_p95",
                            "microcontrast",
                        )
                    }
                },
            }

        display = attempt("cand_display90", display_gate, 1.0)
        candidate_b = attempt("cand_b", safe_gate, 0.80)
        selected = min(
            (display, candidate_b),
            key=processor._stage7_candidate_selection_key,
        )
        self.assertEqual(selected["name"], "cand_b")

        display["preview_retention"] = copy.deepcopy(
            candidate_b["preview_retention"]
        )
        selected_without_retention_advantage = min(
            (display, candidate_b),
            key=processor._stage7_candidate_selection_key,
        )
        self.assertEqual(selected_without_retention_advantage["name"], "cand_b")


class Display90RoutingTests(unittest.TestCase):
    def test_candidate_is_limited_to_accepted_auto_starless_route(self) -> None:
        image = _linear_rgb()
        stats = _pixel_stats(image)
        preview_stats = {"p50": 0.25, "p99": 0.80}
        processor = _StretchProcessor()

        candidates, adaptation = processor._stage7_compact_stretch_candidates(
            QualityMetrics(bg_median=stats["p50"]),
            {"bg_std": 0.001},
            stats,
            preview_stats,
            starless_recomposition_planned=True,
            source_stem="stage6_starless",
            star_separation_state=StarSeparationState.ACCEPTED.value,
            baseline_image_data=image,
        )
        self.assertEqual(
            [candidate["name"] for candidate in candidates],
            [
                "cand_a",
                "cand_b",
                "cand_display82",
                "cand_display70",
                "cand_display90",
            ],
        )
        self.assertEqual(adaptation["display90_calibration"]["status"], "ok")
        self.assertEqual(
            [
                item["name"]
                for item in adaptation["display_ladder"]["tiers"]
            ],
            ["cand_display82", "cand_display70", "cand_display90"],
        )

        processor.cfg.stage7_processing_mode = "manual"
        manual_candidates, _adaptation = (
            processor._stage7_compact_stretch_candidates(
                QualityMetrics(bg_median=stats["p50"]),
                {"bg_std": 0.001},
                stats,
                preview_stats,
                starless_recomposition_planned=True,
                source_stem="stage6_starless",
                star_separation_state=StarSeparationState.ACCEPTED.value,
                baseline_image_data=image,
            )
        )
        self.assertNotIn(
            "cand_display90",
            [candidate["name"] for candidate in manual_candidates],
        )

        processor.cfg.stage7_processing_mode = "auto"
        bypass_candidates, bypass_adaptation = (
            processor._stage7_compact_stretch_candidates(
                QualityMetrics(bg_median=stats["p50"]),
                {"bg_std": 0.001},
                stats,
                preview_stats,
                starless_recomposition_planned=False,
                source_stem="stage6_passthrough",
                star_separation_state=StarSeparationState.TARGET_BYPASS.value,
                baseline_image_data=image,
            )
        )
        self.assertNotIn(
            "cand_display90",
            [candidate["name"] for candidate in bypass_candidates],
        )
        self.assertEqual(
            bypass_adaptation["display90_calibration"]["status"],
            "not_applicable",
        )

        degenerate = np.zeros((3, 32, 32), dtype=np.float32)
        degenerate_candidates, degenerate_adaptation = (
            processor._stage7_compact_stretch_candidates(
                QualityMetrics(bg_median=0.0),
                {"bg_std": 0.0},
                _pixel_stats(degenerate),
                {"p50": 0.0, "p99": 0.0},
                starless_recomposition_planned=True,
                source_stem="stage6_starless",
                star_separation_state=StarSeparationState.ACCEPTED.value,
                baseline_image_data=degenerate,
            )
        )
        self.assertNotIn(
            "cand_display90",
            [candidate["name"] for candidate in degenerate_candidates],
        )
        self.assertEqual(
            degenerate_adaptation["display90_calibration"]["status"],
            "unavailable",
        )

    def test_galaxy_route_adds_bounded_mid_high_display_candidate(self) -> None:
        image = _linear_rgb()
        stats = _pixel_stats(image)
        processor = _StretchProcessor()
        processor._active_target_type = lambda: "large_galaxy"
        processor._channel_semantics = "broadband_rgb_osc"

        candidates, adaptation = processor._stage7_compact_stretch_candidates(
            QualityMetrics(bg_median=stats["p50"]),
            {"bg_std": 0.001},
            stats,
            {"p50": 0.25, "p99": 0.80},
            starless_recomposition_planned=True,
            source_stem="stage6_starless",
            star_separation_state=StarSeparationState.ACCEPTED.value,
            baseline_image_data=image,
        )

        names = [candidate["name"] for candidate in candidates]
        self.assertIn("cand_display86", names)
        tier = next(
            item
            for item in adaptation["display_ladder"]["tiers"]
            if item["name"] == "cand_display86"
        )
        self.assertAlmostEqual(tier["strength"], 0.86)
        self.assertEqual(tier["tier"], "mid_high")

    def test_galaxy_brightness_floor_preserves_core_safe_tradeoff(self) -> None:
        preview = {
            "metrics": {
                "subject_p50": 0.50,
                "subject_lift": 0.20,
                "background_mad": 0.02,
            }
        }
        candidate = {
            "metrics": {
                "subject_p50": 0.295,
                "subject_lift": 0.11,
                "background_mad": 0.01,
            }
        }

        galaxy = stretch_metrics.subject_brightness_selection(
            candidate,
            preview,
            profile_name="galaxy_core_halo_balance",
        )
        generic = stretch_metrics.subject_brightness_selection(
            candidate,
            preview,
            profile_name="generic_balanced",
        )

        self.assertTrue(galaxy["formal_floor_passed"])
        self.assertEqual(galaxy["floors"]["subject_p50_retention"], 0.58)
        self.assertFalse(generic["formal_floor_passed"])

    def test_policy_keeps_legacy_auto_dual_and_all_only_modes(self) -> None:
        candidates = [
            {"name": "cand_a"},
            {"name": "cand_b"},
            {"name": "cand_display90"},
        ]
        expected = {
            "auto_display90": ["cand_a", "cand_b", "cand_display90"],
            "auto_dual": ["cand_a", "cand_b"],
            "candidate_a_only": ["cand_a"],
            "candidate_b_only": ["cand_b"],
            "display90_only": ["cand_display90"],
        }
        for policy, names in expected.items():
            self.assertEqual(
                [
                    item["name"]
                    for item in _stage7_candidates_for_policy(candidates, policy)
                ],
                names,
            )

    def test_processing_contract_defaults_clamps_and_rejects_manual_only(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.stage7_candidate_policy, "auto_display90")
        self.assertEqual(cfg.stage7_display90_strength, 0.90)
        self.assertEqual(
            cfg.stage7_display90_reference_chroma_load_ratio_max,
            1.05,
        )
        self.assertEqual(
            cfg.stage7_display90_reference_chroma_load_absolute_max,
            0.30,
        )
        self.assertEqual(cfg.stage7_rendition_intent, "vivid_safe")
        self.assertTrue(cfg.stage7_forced_delivery_enabled)
        self.assertIn("stage7_display90_strength", SPECS_BY_FIELD)
        self.assertIn(
            "stage7_display90_reference_chroma_load_ratio_max",
            SPECS_BY_FIELD,
        )

        payload = default_processing_parameters()
        payload["stages"]["7"]["overrides"][
            "stage7_display90_strength"
        ] = 1.50
        payload["stages"]["7"]["overrides"][
            "stage7_display90_reference_chroma_load_ratio_max"
        ] = 2.0
        payload["stages"]["7"]["overrides"][
            "stage7_display90_reference_chroma_load_absolute_max"
        ] = 0.10
        normalized, adjustments = normalize_processing_parameters(payload)
        self.assertEqual(
            normalized["stages"]["7"]["overrides"][
                "stage7_display90_strength"
            ],
            0.95,
        )
        self.assertEqual(
            normalized["stages"]["7"]["overrides"][
                "stage7_display90_reference_chroma_load_ratio_max"
            ],
            1.20,
        )
        self.assertEqual(
            normalized["stages"]["7"]["overrides"][
                "stage7_display90_reference_chroma_load_absolute_max"
            ],
            0.15,
        )
        self.assertTrue(
            any(
                item.get("field") == "stage7_display90_strength"
                for item in adjustments
            )
        )

        invalid = default_processing_parameters()
        invalid["stages"]["7"]["mode"] = "manual"
        invalid["stages"]["7"]["overrides"][
            "stage7_candidate_policy"
        ] = "display90_only"
        with self.assertRaisesRegex(ValueError, "display90_only"):
            normalize_processing_parameters(invalid)

class Stage7SubjectBrightnessSelectionTests(unittest.TestCase):
    @staticmethod
    def _report(subject_p50: float, background: float, mad: float) -> dict:
        lift = max(0.0, subject_p50 - background)
        sigma = 1.4826 * mad
        return {
            "status": "available",
            "metrics": {
                "subject_p50": subject_p50,
                "subject_lift": lift,
                "subject_lift_sigma": lift / sigma if sigma > 0.0 else 0.0,
                "background_median": background,
                "background_mad": mad,
            },
        }

    def test_reliable_subject_lift_and_p50_use_bounded_goals(self) -> None:
        preview = self._report(0.10, 0.02, 0.005)
        at_goal = stretch_metrics.subject_brightness_selection(
            self._report(0.08, 0.02, 0.005),
            preview,
        )
        above_goal = stretch_metrics.subject_brightness_selection(
            self._report(0.14, 0.02, 0.005),
            preview,
        )

        self.assertTrue(at_goal["formal_floor_passed"], at_goal)
        self.assertEqual(at_goal["ranking"]["goal_count"], 2)
        self.assertEqual(at_goal["ranking"]["utility"], 1.0)
        self.assertEqual(above_goal["ranking"]["utility"], 1.0)
        self.assertEqual(
            above_goal["ranking"]["above_goal_reward"],
            "capped",
        )

    def test_subject_brightness_floor_rejects_dim_candidate(self) -> None:
        report = stretch_metrics.subject_brightness_selection(
            self._report(0.05, 0.02, 0.005),
            self._report(0.10, 0.02, 0.005),
        )

        self.assertFalse(report["formal_floor_passed"], report)
        self.assertEqual(
            report["reason_code"],
            "stage7_subject_brightness_floor_unmet",
        )
        self.assertIn("subject_p50_retention_below_floor", report["issues"])
        self.assertIn("subject_lift_retention_below_floor", report["issues"])

    def test_star_subject_contract_does_not_require_diffuse_lift(self) -> None:
        preview = self._report(0.10, 0.02, 0.005)
        report = stretch_metrics.subject_brightness_selection(
            self._report(0.025, 0.02, 0.005),
            preview,
            profile_name="star_colour_preserve",
        )

        self.assertTrue(report["formal_floor_passed"], report)
        self.assertEqual(report["subject_kind"], "stellar")
        self.assertEqual(report["contract_profile"], "star_colour_preserve")
        self.assertFalse(
            report["preview_reliability"]["subject_lift_applicable"]
        )
        self.assertIsNone(report["retention"]["subject_lift"])
        self.assertEqual(report["floors"]["subject_p50_retention"], 0.20)

    def test_unreliable_preview_lift_does_not_participate(self) -> None:
        preview = self._report(0.039, 0.02, 0.005)
        candidate = self._report(0.025, 0.02, 0.005)
        report = stretch_metrics.subject_brightness_selection(
            candidate,
            preview,
        )

        self.assertFalse(
            report["preview_reliability"]["subject_lift_reliable"]
        )
        self.assertIsNone(report["retention"]["subject_lift"])
        self.assertTrue(report["formal_floor_passed"], report)
        self.assertEqual(report["ranking"]["applicable_goal_count"], 1)

    def test_selection_prefers_brightness_only_after_safety_gates(self) -> None:
        preview = self._report(0.10, 0.02, 0.005)
        dim = stretch_metrics.subject_brightness_selection(
            self._report(0.065, 0.02, 0.005),
            preview,
        )
        bright = stretch_metrics.subject_brightness_selection(
            self._report(0.08, 0.02, 0.005),
            preview,
        )

        def attempt(name: str, brightness: dict, *, technical: bool = True) -> dict:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "technical_safe": technical,
                "subject_brightness_selection": brightness,
                "presentation_score": {"score": 0.5},
                "advisories": [],
                "risk_score": 0.1,
            }

        safe_winner = min(
            [attempt("dim", dim), attempt("bright", bright)],
            key=Stage6ServiceMixin._stage7_candidate_selection_key,
        )
        self.assertEqual(safe_winner["name"], "bright")

        unsafe_bright = attempt("unsafe_bright", bright, technical=False)
        safe_dim = attempt("safe_dim", dim)
        safety_winner = min(
            [unsafe_bright, safe_dim],
            key=Stage6ServiceMixin._stage7_candidate_selection_key,
        )
        self.assertEqual(safety_winner["name"], "safe_dim")

    def test_v7_continuous_quality_precedes_brightness_goal(self) -> None:
        preview = self._report(0.039, 0.02, 0.005)
        lower_p50 = stretch_metrics.subject_brightness_selection(
            self._report(0.025, 0.02, 0.005),
            preview,
        )
        higher_p50 = stretch_metrics.subject_brightness_selection(
            self._report(0.034, 0.02, 0.005),
            preview,
        )

        def attempt(
            name: str,
            brightness: dict,
            *,
            score: float,
            advisories: list[str],
            risk: float,
        ) -> dict:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "technical_safe": True,
                "subject_brightness_selection": brightness,
                "presentation_score": {"score": score},
                "advisories": advisories,
                "risk_score": risk,
            }

        display70 = attempt(
            "cand_display70",
            lower_p50,
            score=0.84,
            advisories=[],
            risk=0.00001,
        )
        higher_brightness = attempt(
            "cand_display82",
            higher_p50,
            score=0.81,
            advisories=["background_chroma_load_growth"],
            risk=0.02,
        )
        winner = min(
            [higher_brightness, display70],
            key=Stage6ServiceMixin._stage7_candidate_selection_key,
        )

        self.assertFalse(
            lower_p50["preview_reliability"]["subject_lift_reliable"]
        )
        self.assertEqual(winner["name"], "cand_display70")


class Display90Stage9ContractTests(unittest.TestCase):
    def test_stage9_reuses_exact_display90_lut(self) -> None:
        image = _linear_rgb()
        calibration = _calibration(image)
        selected = {
            "name": "cand_display90",
            "params": {"calibration": calibration},
        }
        transfer = _stage7_matched_domain_transfer_contract(selected, {})
        self.assertEqual(
            transfer["schema"],
            stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V3,
        )
        self.assertEqual(len(transfer["chain_contract"]["steps"]), 1)
        with tempfile.TemporaryDirectory() as td:
            pipeline = types.SimpleNamespace(
                _stage7_matched_domain_transfer=transfer,
                _stage7_closed_form_mtf_reference=None,
                process_dir=Path(td),
            )
            resolved = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                pipeline
            )

        self.assertEqual(resolved["status"], "ready", resolved)
        self.assertEqual(
            resolved["method"],
            stretch_metrics.DISPLAY_LUMINANCE_VECTOR_METHOD,
        )
        stage9_pixels = stage9_star_remixing._stage9_apply_matched_domain_transfer(
            image,
            resolved,
        )
        stage7_pixels = stretch_metrics.apply_display90_linked_rgb_stretch(
            image,
            calibration,
        )
        np.testing.assert_array_equal(stage9_pixels, stage7_pixels)

        tampered = {
            **transfer,
            "chain_contract": {
                **transfer["chain_contract"],
                "sha256": "0" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            rejected = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                types.SimpleNamespace(
                    _stage7_matched_domain_transfer=tampered,
                    _stage7_closed_form_mtf_reference=None,
                    process_dir=Path(td),
                )
            )
        self.assertEqual(rejected["status"], "unavailable")
        self.assertEqual(rejected["reason_code"], "stage9_display90_transfer_invalid")

    def test_composite_tone_v3_replays_exact_authenticated_chain(self) -> None:
        image = _linear_rgb()
        parent_calibration = _calibration(image, strength=0.82)
        calibration = stretch_metrics.build_composite_tone_calibration(
            parent_calibration,
            source_background=0.18,
            target_background=0.11,
        )
        selected = {
            "name": "cand_composite_tone",
            "method": stretch_metrics.COMPOSITE_TONE_TRANSFER_METHOD,
            "params": {"calibration": calibration},
        }
        transfer = _stage7_matched_domain_transfer_contract(selected, {})
        with tempfile.TemporaryDirectory() as td:
            pipeline = types.SimpleNamespace(
                _stage7_matched_domain_transfer=transfer,
                _stage7_closed_form_mtf_reference=None,
                process_dir=Path(td),
            )
            resolved = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                pipeline
            )

        self.assertEqual(
            transfer["schema"],
            stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V3,
        )
        self.assertEqual(transfer["selected_candidate_id"], "cand_composite_tone")
        self.assertEqual(transfer["tone_candidate_id"], "cand_composite_tone")
        self.assertEqual(resolved["status"], "ready", resolved)
        self.assertEqual(resolved["tone_candidate_id"], "cand_composite_tone")
        stage9_pixels = stage9_star_remixing._stage9_apply_matched_domain_transfer(
            image,
            resolved,
        )
        expected = stretch_metrics.apply_composite_tone_transfer(
            image,
            calibration,
        )
        np.testing.assert_array_equal(stage9_pixels, expected)

        corrupted = dict(transfer)
        corrupted["chain_contract"] = {
            **dict(transfer["chain_contract"]),
            "sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as td:
            rejected = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                types.SimpleNamespace(
                    _stage7_matched_domain_transfer=corrupted,
                    _stage7_closed_form_mtf_reference=None,
                    process_dir=Path(td),
                )
            )
        self.assertEqual(rejected["status"], "unavailable")
        self.assertEqual(
            rejected["reason_code"],
            "stage9_composite_tone_transfer_invalid",
        )

    def test_stage9_replays_legacy_v1_display_transfer_without_semantic_upgrade(
        self,
    ) -> None:
        image = _linear_rgb()
        legacy = _calibration(image)
        legacy["schema"] = stretch_metrics.DISPLAY90_STRETCH_SCHEMA_V1
        legacy["method"] = stretch_metrics.DISPLAY90_LEGACY_METHOD
        legacy.pop("application_contract", None)
        selected = {
            "name": "cand_display90",
            "method": stretch_metrics.DISPLAY90_LEGACY_METHOD,
            "params": {"calibration": legacy},
        }
        generated = _stage7_matched_domain_transfer_contract(selected, {})
        self.assertEqual(
            generated["schema"],
            stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V3,
        )
        transfer = dict(generated)
        transfer["schema"] = (
            stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V1
        )
        transfer.pop("chain_contract", None)
        with tempfile.TemporaryDirectory() as td:
            pipeline = types.SimpleNamespace(
                _stage7_matched_domain_transfer=transfer,
                _stage7_closed_form_mtf_reference=None,
                process_dir=Path(td),
            )
            resolved = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                pipeline
            )

        self.assertEqual(resolved["status"], "ready", resolved)
        self.assertEqual(
            resolved["method"],
            stretch_metrics.DISPLAY90_LEGACY_METHOD,
        )
        np.testing.assert_array_equal(
            stage9_star_remixing._stage9_apply_matched_domain_transfer(
                image,
                resolved,
            ),
            stretch_metrics.apply_display90_linked_rgb_stretch(image, legacy),
        )

    def test_closed_form_mtf_uses_v3_single_step_and_rejects_tamper(self) -> None:
        image = _linear_rgb()
        params = {
            "mtf_shadows": 0.01,
            "mtf_midtones": 0.22,
            "mtf_highlights": 0.98,
        }
        reference = {
            "schema": "starun.stage7-mtf-reference.v1",
            "status": "active",
            "active_anchor": {
                "candidate": "cand_b",
                "method": "closed_form_linked_mtf",
                "params": params,
            },
        }
        transfer = _stage7_matched_domain_transfer_contract(
            {"name": "cand_b", "method": "linked_mtf", "params": params},
            reference,
        )
        self.assertEqual(
            transfer["schema"],
            stretch_metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V3,
        )
        self.assertEqual(
            [step["method"] for step in transfer["chain_contract"]["steps"]],
            ["closed_form_linked_mtf"],
        )
        with tempfile.TemporaryDirectory() as td:
            resolved = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                types.SimpleNamespace(
                    _stage7_matched_domain_transfer=transfer,
                    _stage7_closed_form_mtf_reference=reference,
                    process_dir=Path(td),
                )
            )
        self.assertEqual(resolved["status"], "ready", resolved)
        np.testing.assert_array_equal(
            stage9_star_remixing._stage9_apply_matched_domain_transfer(
                image,
                resolved,
            ),
            stretch_metrics.apply_linked_mtf(
                image,
                params["mtf_shadows"],
                params["mtf_midtones"],
                params["mtf_highlights"],
            ),
        )

        forged_winner = copy.deepcopy(transfer)
        forged_winner["selected_candidate_id"] = "cand_a"
        forged_winner["tone_candidate_id"] = "cand_a"
        with tempfile.TemporaryDirectory() as td:
            forged_rejected = (
                stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                    types.SimpleNamespace(
                        _stage7_matched_domain_transfer=forged_winner,
                        _stage7_closed_form_mtf_reference=reference,
                        process_dir=Path(td),
                    )
                )
            )
        self.assertEqual(forged_rejected["status"], "unavailable")
        self.assertEqual(
            forged_rejected["reason_code"],
            "stage9_matched_domain_transfer_invalid",
        )

        tampered = {
            **transfer,
            "chain_contract": {
                **transfer["chain_contract"],
                "steps": [
                    {
                        **transfer["chain_contract"]["steps"][0],
                        "params": {**params, "mtf_midtones": 0.23},
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            rejected = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                types.SimpleNamespace(
                    _stage7_matched_domain_transfer=tampered,
                    _stage7_closed_form_mtf_reference=reference,
                    process_dir=Path(td),
                )
            )
        self.assertEqual(rejected["status"], "unavailable")
        self.assertEqual(
            rejected["reason_code"],
            "stage9_matched_domain_transfer_invalid",
        )

    def test_non_cand_b_winner_cannot_borrow_closed_form_mtf_reference(self) -> None:
        params = {
            "mtf_shadows": 0.01,
            "mtf_midtones": 0.22,
            "mtf_highlights": 0.98,
        }
        reference = {
            "schema": "starun.stage7-mtf-reference.v1",
            "status": "active",
            "active_anchor": {
                "candidate": "cand_b",
                "method": "closed_form_linked_mtf",
                "params": params,
            },
        }

        transfer = _stage7_matched_domain_transfer_contract(
            {
                "name": "cand_a",
                "method": "asinh",
                "params": {"asinh_stretch": 2.2, "asinh_offset": 0.002},
            },
            reference,
        )

        self.assertEqual(transfer["status"], "unavailable")
        self.assertEqual(transfer["selected_candidate_id"], "cand_a")
        self.assertNotIn("chain_contract", transfer)

    def test_cand_b_params_must_match_closed_form_mtf_reference(self) -> None:
        selected_params = {
            "mtf_shadows": 0.01,
            "mtf_midtones": 0.22,
            "mtf_highlights": 0.98,
        }
        reference = {
            "schema": "starun.stage7-mtf-reference.v1",
            "status": "active",
            "active_anchor": {
                "candidate": "cand_b",
                "method": "closed_form_linked_mtf",
                "params": {**selected_params, "mtf_midtones": 0.24},
            },
        }

        transfer = _stage7_matched_domain_transfer_contract(
            {"name": "cand_b", "method": "linked_mtf", "params": selected_params},
            reference,
        )

        self.assertEqual(transfer["status"], "unavailable")
        self.assertNotIn("chain_contract", transfer)

    def test_corrupt_or_missing_display90_contract_never_uses_mtf(self) -> None:
        image = _linear_rgb()
        calibration = _calibration(image)
        selected = {
            "name": "cand_display90",
            "params": {"calibration": calibration},
        }
        transfer = _stage7_matched_domain_transfer_contract(selected, {})
        transfer["calibration"]["lut_contract"]["sha256"] = "f" * 64
        reference = {
            "status": "active",
            "active_anchor": {
                "params": {
                    "mtf_shadows": 0.0,
                    "mtf_midtones": 0.2,
                    "mtf_highlights": 1.0,
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            pipeline = types.SimpleNamespace(
                _stage7_matched_domain_transfer=transfer,
                _stage7_closed_form_mtf_reference=reference,
                process_dir=Path(td),
            )
            corrupt = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                pipeline
            )
            self.assertEqual(corrupt["status"], "unavailable")
            self.assertEqual(
                corrupt["reason_code"],
                "stage9_display90_transfer_invalid",
            )

            outer_tampered_transfer = _stage7_matched_domain_transfer_contract(
                selected,
                {},
            )
            outer_tampered_transfer["lut_contract"]["sha256"] = "e" * 64
            outer_tampered = types.SimpleNamespace(
                _stage7_matched_domain_transfer=outer_tampered_transfer,
                _stage7_closed_form_mtf_reference=reference,
                process_dir=Path(td),
            )
            outer_rejected = (
                stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                    outer_tampered
                )
            )
            self.assertEqual(outer_rejected["status"], "unavailable")
            self.assertEqual(
                outer_rejected["reason_code"],
                "stage9_display90_transfer_invalid",
            )

            report_path = Path(td) / "stage7_stretch_quality.json"
            report_path.write_text(
                json.dumps(
                    {
                        "selected": {"name": "cand_display90"},
                        "closed_form_mtf_reference": reference,
                    }
                ),
                encoding="utf-8",
            )
            resumed = types.SimpleNamespace(
                _stage7_matched_domain_transfer=None,
                _stage7_closed_form_mtf_reference=None,
                process_dir=Path(td),
            )
            missing = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                resumed
            )
            self.assertEqual(missing["status"], "unavailable")
            self.assertEqual(
                missing["reason_code"],
                "stage9_display90_transfer_invalid",
            )

    def test_legacy_closed_form_mtf_report_remains_supported(self) -> None:
        reference = {
            "schema": "starun.stage7-mtf-reference.v1",
            "status": "active",
            "active_anchor": {
                "method": "closed_form_linked_mtf",
                "params": {
                    "mtf_shadows": 0.01,
                    "mtf_midtones": 0.20,
                    "mtf_highlights": 1.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as td:
            pipeline = types.SimpleNamespace(
                _stage7_matched_domain_transfer=None,
                _stage7_closed_form_mtf_reference=reference,
                process_dir=Path(td),
            )
            resolved = stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                pipeline
            )
        self.assertEqual(resolved["status"], "ready", resolved)
        self.assertEqual(resolved["method"], "closed_form_linked_mtf")


if __name__ == "__main__":
    unittest.main()
