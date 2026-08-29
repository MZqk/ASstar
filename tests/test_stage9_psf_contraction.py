from __future__ import annotations

import copy
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


if "sirilpy.exceptions" not in sys.modules:
    package = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class SirilError(Exception):
        pass

    package.SirilInterface = object
    exceptions.SirilError = SirilError
    exceptions.CommandError = type("CommandError", (SirilError,), {})
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions

import stage9_quality  # noqa: E402
from stages import stage9_star_remixing as stage9_remix  # noqa: E402
from stage8_starless_finish import (  # noqa: E402
    DECODED_PIXEL_SHA256_METHOD,
    FITS_DATA_SHA256_METHOD,
    canonical_decoded_pixel_sha256,
)


def _quality(*, weak: float, bright: float, all_ratio: float) -> dict:
    accepted = all(
        0.93 <= value <= 1.10 for value in (weak, bright, all_ratio)
    )
    issues = []
    if not accepted:
        for group, value in (
            ("all", all_ratio),
            ("weak", weak),
            ("bright", bright),
        ):
            if not 0.93 <= value <= 1.10:
                issues.append(
                    f"star_psf_fwhm_ratio_{group} {value:.6f} "
                    "outside 0.930000..1.100000"
                )
    return {
        "attempt": "candidate",
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "issues": issues,
        "metrics": {},
        "reference_fidelity": {
            "schema": "starun.stage9-reference-fidelity.v1",
            "status": "ok",
            "support_rgb_mae": 0.02,
            "support_rgb_p95": 0.05,
            "reference_base_role": "actual_stage8_remix_base_B",
            "reference_base_pixel_sha256": "a" * 64,
            "reference_base_identity_verified": True,
            "matched_domain_authenticated": True,
            "stage6_starless_used_as_base": False,
        },
        "psf_closure": {
            "status": "accepted" if accepted else "rejected",
            "limits": {
                "stage9_psf_fwhm_ratio_min": 0.93,
                "stage9_psf_fwhm_ratio_max": 1.10,
            },
            "groups": {
                "all": {"status": "ok", "fwhm_ratio_median": all_ratio},
                "weak": {"status": "ok", "fwhm_ratio_median": weak},
                "bright": {"status": "ok", "fwhm_ratio_median": bright},
            },
        },
    }


class Stage9StarmaskTargetProfileTests(unittest.TestCase):
    @staticmethod
    def _pipeline(*, target_type: str, semantics: str, manual_fields=()):
        reports = {}
        pipeline = SimpleNamespace(
            cfg=SimpleNamespace(
                stage9_starmask_faint_target=0.26,
                stage9_starmask_mid_target=0.50,
                stage9_starmask_bright_target=0.75,
                stage9_starmask_peak_target=0.90,
            ),
            _channel_semantics=semantics,
            _task_manual_override_fields=manual_fields,
            _active_target_type=lambda: target_type,
            _write_stage_json=lambda name, payload: reports.__setitem__(
                name, copy.deepcopy(payload)
            ),
        )
        return pipeline, reports

    def test_narrowband_widefield_profile_uses_lowest_existing_anchor_tier(self):
        pipeline, reports = self._pipeline(
            target_type="emission_nebula_widefield",
            semantics="narrowband_composite",
        )

        effective, report = stage9_remix._stage9_effective_starmask_profile(
            pipeline
        )

        self.assertEqual(report["status"], "active")
        self.assertEqual(
            report["profile_id"],
            "narrowband_widefield_compact_flux_v1",
        )
        self.assertEqual(effective.stage9_starmask_faint_target, 0.08)
        self.assertEqual(effective.stage9_starmask_mid_target, 0.30)
        self.assertEqual(effective.stage9_starmask_bright_target, 0.50)
        self.assertEqual(effective.stage9_starmask_peak_target, 0.75)
        self.assertEqual(pipeline.cfg.stage9_starmask_faint_target, 0.26)
        self.assertTrue(report["scientific_gates_unchanged"])
        self.assertEqual(report["formal_psf_gate_unchanged"], [0.93, 1.10])
        self.assertIn("stage9_starmask_target_profile.json", reports)

    def test_profile_does_not_override_signed_anchor_parameters(self):
        pipeline, _reports = self._pipeline(
            target_type="emission_nebula_widefield",
            semantics="narrowband_composite",
            manual_fields=("stage9_starmask_faint_target",),
        )
        pipeline.cfg.stage9_starmask_faint_target = 0.12

        effective, report = stage9_remix._stage9_effective_starmask_profile(
            pipeline
        )

        self.assertEqual(report["status"], "not_applicable")
        self.assertEqual(
            report["reason_code"],
            "stage9_starmask_anchor_manual_override",
        )
        self.assertEqual(effective.stage9_starmask_faint_target, 0.12)
        self.assertEqual(effective.stage9_starmask_mid_target, 0.50)

    def test_bright_composite_profile_uses_compact_anchor_tier(self):
        pipeline, reports = self._pipeline(
            target_type="bright_emission_reflection_nebula",
            semantics="broadband_rgb_osc",
        )

        effective, report = stage9_remix._stage9_effective_starmask_profile(
            pipeline
        )

        self.assertEqual(report["status"], "active")
        self.assertEqual(
            report["profile_id"],
            "bright_composite_compact_flux_v1",
        )
        self.assertEqual(effective.stage9_starmask_faint_target, 0.06)
        self.assertEqual(effective.stage9_starmask_mid_target, 0.26)
        self.assertEqual(effective.stage9_starmask_bright_target, 0.50)
        self.assertEqual(effective.stage9_starmask_peak_target, 0.75)
        self.assertEqual(pipeline.cfg.stage9_starmask_faint_target, 0.26)
        self.assertTrue(report["scientific_gates_unchanged"])
        self.assertEqual(report["presentation_target_unchanged"], [0.97, 1.05])
        self.assertIn("stage9_starmask_target_profile.json", reports)

        actual_targets = stage9_quality._stage9_starmask_output_targets(
            effective
        )
        self.assertEqual(
            actual_targets,
            {
                "faint": 0.06,
                "mid": 0.26,
                "bright": 0.50,
                "peak": 0.75,
            },
        )

    def test_profile_requires_both_target_and_channel_evidence(self):
        for target_type, semantics in (
            ("large_galaxy", "narrowband_composite"),
            ("emission_nebula_widefield", "broadband_rgb"),
            ("bright_emission_reflection_nebula", "narrowband_composite"),
        ):
            with self.subTest(target_type=target_type, semantics=semantics):
                pipeline, _reports = self._pipeline(
                    target_type=target_type,
                    semantics=semantics,
                )
                effective, report = (
                    stage9_remix._stage9_effective_starmask_profile(pipeline)
                )
                self.assertEqual(report["status"], "not_applicable")
                self.assertEqual(effective.stage9_starmask_faint_target, 0.26)


class _FakePipeline:
    def __init__(self, stars: np.ndarray, catalog: dict) -> None:
        self.cfg = SimpleNamespace(
            stage9_targeted_recovery_retry_max=3,
            stage9_psf_fwhm_ratio_min=0.93,
            stage9_psf_fwhm_ratio_max=1.10,
        )
        self._stage9_star_reference_catalog = catalog
        self._stage9_last_star_layer = np.array(stars, copy=True)
        self._stage9_last_star_overlay_mask = None
        self._stage9_last_weak_overlay_mask = None
        self._stage9_last_bright_overlay_mask = None
        self._stage9_star_color_post_validation = None
        self._stage9_starmask_calibration = None
        self.loaded: list[str] = []
        self.saved: list[str] = []
        self.applied: list[tuple[str, str, float]] = []

    def cmd_with_check(self, command: str, stem: str) -> bool:
        if command == "load":
            self.loaded.append(stem)
        return True

    def _save_stage_output(self, stem: str) -> bool:
        self.saved.append(stem)
        return True

    def _apply_previous_stage_star_remix(
        self,
        source_stem: str,
        starmask: str,
        intensity: float,
    ) -> bool:
        self.applied.append((source_stem, starmask, intensity))
        return True


class Stage9PsfContractionTests(unittest.TestCase):
    def _fixture(self):
        height = width = 64
        yy, xx = np.indices((height, width))
        weak = 0.30 * np.exp(
            -((yy - 18.0) ** 2 + (xx - 18.0) ** 2) / (2.0 * 2.2**2)
        )
        bright = 0.82 * np.exp(
            -((yy - 45.0) ** 2 + (xx - 44.0) ** 2) / (2.0 * 2.2**2)
        )
        scalar = (weak + bright).astype(np.float32)
        stars = np.stack((scalar, scalar * 0.70, scalar * 0.40))
        cfg = SimpleNamespace(
            stage9_star_reference_sigma=3.0,
            stage9_mixed_star_weak_count_min=1,
            stage9_mixed_star_bright_count_min=1,
            stage9_mixed_star_peak_ratio_min=2.0,
        )
        catalog = stage9_quality.build_star_reference_catalog(
            stars,
            cfg,
            background=0.0,
            noise_sigma=0.001,
        )
        self.assertEqual(catalog["status"], "ok", catalog)
        weak_mask, bright_mask, support = (
            stage9_quality.build_star_overlay_masks(
                catalog,
                strict=False,
                cfg=cfg,
            )
        )
        return stars, catalog, weak_mask, bright_mask, support

    def _reference_guided_fixture(self, *, dtype=np.float32):
        height = width = 192
        yy, xx = np.indices((height, width), dtype=np.float32)
        centers = [
            (y, x)
            for y in (24, 60, 96, 132, 168)
            for x in (30, 74, 118, 162)
        ]
        source_scalar = np.zeros((height, width), dtype=np.float32)
        parent_scalar = np.zeros_like(source_scalar)
        for index, (y, x) in enumerate(centers):
            amplitude = 0.20 if index < 16 else 0.75
            source_scalar += amplitude * np.exp(
                -((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * 1.20**2)
            )
            parent_scalar += amplitude * np.exp(
                -((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * 1.85**2)
            )
        source_stars = np.stack(
            (source_scalar, source_scalar * 0.78, source_scalar * 0.56)
        ).astype(np.float32)
        parent_stars = np.stack(
            (parent_scalar, parent_scalar * 0.78, parent_scalar * 0.56)
        ).astype(np.float32)
        base = np.full_like(source_stars, 0.08)
        original = stage9_quality.screen_blend(base, source_stars, 1.0)
        cfg = SimpleNamespace(
            stage9_star_reference_sigma=3.0,
            stage9_source_component_density_max=2500.0,
            stage9_source_single_pixel_ratio_max=0.20,
            stage9_mixed_star_weak_count_min=4,
            stage9_mixed_star_bright_count_min=3,
            stage9_mixed_star_peak_ratio_min=2.0,
            stage9_psf_min_sample_count=16,
            stage9_psf_size_gate_enabled=True,
            stage9_psf_fwhm_ratio_min=0.93,
            stage9_psf_fwhm_ratio_max=1.10,
        )
        catalog = stage9_quality.build_display_confirmed_starmask_catalog(
            source_stars,
            original,
            cfg,
        )
        self.assertEqual(catalog["status"], "ok", catalog)
        support = np.zeros((height, width), dtype=bool)
        weak_mask = np.zeros_like(support)
        bright_mask = np.zeros_like(support)
        for index, (y, x) in enumerate(
            zip(catalog["_peak_y"], catalog["_peak_x"])
        ):
            disk = (yy - int(y)) ** 2 + (xx - int(x)) ** 2 <= 8.0**2
            support |= disk
            if bool(catalog["_weak_flags"][index]):
                weak_mask |= disk
            else:
                bright_mask |= disk
        scale = 65535.0 if np.dtype(dtype) == np.dtype(np.float64) else 1.0
        return (
            (parent_stars.astype(dtype) * scale),
            (base.astype(dtype) * scale),
            (original.astype(dtype) * scale),
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        )

    def _reference_guided_transaction_fixture(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()
        cfg.stage9_targeted_recovery_retry_max = 3
        cfg.stage9_psf_recovery_target_min = 0.97
        cfg.stage9_psf_recovery_target_max = 1.05
        pipeline = _FakePipeline(stars, catalog)
        pipeline.cfg = cfg
        pipeline._stage9_remix_base_stem = "stage8_enhanced"
        base_pixel_sha = canonical_decoded_pixel_sha256(base)
        fits_data_sha = "c" * 64
        container_sha = "b" * 64
        pipeline._stage9_remix_base_identity = {
            "status": "locked",
            "source_stem": "stage8_enhanced",
            "sha256": container_sha,
            "fits_data_sha256": fits_data_sha,
            "fits_data_sha256_method": FITS_DATA_SHA256_METHOD,
            "decoded_pixel_sha256": base_pixel_sha,
            "decoded_pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
        }
        pipeline._stage9_stage8_handoff_verification = {
            "handoff_schema": "starun.stage8-handoff.v3",
            "status": "verified",
            "verified": True,
            "artifact": {
                "expected_sha256": container_sha,
                "actual_sha256": container_sha,
                "expected_pixel_sha256": fits_data_sha,
                "actual_pixel_sha256": fits_data_sha,
                "expected_fits_data_sha256": fits_data_sha,
                "actual_fits_data_sha256": fits_data_sha,
                "fits_data_sha256_method": FITS_DATA_SHA256_METHOD,
                "expected_decoded_pixel_sha256": base_pixel_sha,
                "actual_decoded_pixel_sha256": base_pixel_sha,
                "decoded_pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
            },
        }
        candidate_store: dict[str, np.ndarray] = {}
        pipeline._current_pixels = np.zeros_like(base)
        pipeline.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: np.array(
                pipeline._current_pixels,
                copy=True,
            )
        )

        def capture_candidate(*_args, **kwargs):
            candidate_store["stars"] = np.array(kwargs["stars"], copy=True)
            return True

        def apply_candidate(source_stem, starmask, intensity):
            pipeline.applied.append((source_stem, starmask, intensity))
            candidate = candidate_store.get("stars")
            if candidate is None:
                return False
            pipeline._current_pixels = stage9_quality.screen_blend(
                base,
                candidate,
                intensity,
                alpha_mask=support,
                weak_mask=weak_mask,
                bright_mask=bright_mask,
                weak_intensity=max(float(intensity), 0.55),
            )
            return True

        pipeline._apply_previous_stage_star_remix = apply_candidate
        parent_quality = _quality(
            weak=1.08,
            bright=1.08,
            all_ratio=1.08,
        )
        parent_quality["attempt"] = "parent"
        context = {
            "available": True,
            "stars": stars,
            "unscreen_stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "parent_stars",
            "support_starmask": "parent_stars",
            "support_mode": "normal",
            "original_display": original,
            "starless_display": base,
            "remix_base": base,
            "remix_base_stem": "stage8_enhanced",
            "matched_domain_authenticated": True,
        }
        return (
            pipeline,
            parent_quality,
            context,
            candidate_store,
            capture_candidate,
        )

    def test_reference_guided_halfmax_pruning_closes_all_groups(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars,
            base,
            original,
            catalog,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.0,
        )

        self.assertIsNotNone(pruned, report)
        self.assertTrue(report["accepted"], report)
        self.assertGreater(report["changed_pixel_count"], 0)
        self.assertTrue(report["reference_fidelity_nonregression"])
        self.assertTrue(report["exact_source_pixel_preservation_verified"])
        self.assertTrue(
            all(0.97 <= value <= 1.05 for value in report["fwhm_ratios"].values())
        )
        self.assertGreater(report["timings_seconds"]["total"], 0.0)

    def test_reference_guided_wing_scope_is_frozen_before_threshold_crossing(
        self,
    ):
        immutable_peak = np.asarray([[1.0, 0.46, 0.20]], dtype=np.float32)
        frozen_wings = stage9_quality._freeze_stage9_assigned_halfmax_wings(
            immutable_peak,
            np.asarray([0, 1, 2], dtype=np.int64),
            {7: (0, 3)},
            {7: 1.0},
            wing_ceiling=0.45,
        )
        final_peak = np.array(immutable_peak, copy=True)
        final_peak[0, 1] = 0.40
        delta = np.abs(final_peak - immutable_peak)

        self.assertFalse(frozen_wings[0, 1])
        self.assertTrue(frozen_wings[0, 2])
        self.assertGreater(immutable_peak[0, 1], 0.45)
        self.assertLessEqual(final_peak[0, 1], 0.45)
        self.assertEqual(float(np.max(delta[frozen_wings])), 0.0)

    def test_reference_guided_halfmax_pruning_rejects_fidelity_regression(self):
        (
            stars,
            base,
            _original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()
        parent_display = stage9_quality.screen_blend(base, stars, 1.0)

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars,
            base,
            parent_display,
            catalog,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.0,
        )

        self.assertIsNone(pruned)
        self.assertFalse(report["accepted"])

    def test_reference_guided_float64_exactly_preserves_immutable_pixels(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture(dtype=np.float64)
        original_stars = np.array(stars, copy=True)

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars,
            base,
            original,
            catalog,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.0,
        )

        self.assertIsNotNone(pruned, report)
        assert pruned is not None
        self.assertEqual(pruned.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(pruned[:, ~support], original_stars[:, ~support])
        np.testing.assert_array_equal(
            pruned[:, catalog["_peak_y"], catalog["_peak_x"]],
            original_stars[:, catalog["_peak_y"], catalog["_peak_x"]],
        )
        self.assertEqual(report["outside_support_max_abs_change"], 0.0)
        self.assertEqual(report["protected_wing_max_abs_change"], 0.0)
        self.assertTrue(report["centroid_guard_passed"], report)
        self.assertLessEqual(report["centroid_drift_max_px"], 0.05)

    def test_reference_guided_freezes_overlapping_saturated_global_peak_anchor(
        self,
    ):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()
        baseline, baseline_report = (
            stage9_quality.prune_star_layer_halfmax_boundaries(
                stars,
                base,
                original,
                catalog,
                cfg,
                support_mask=support,
                weak_mask=weak_mask,
                bright_mask=bright_mask,
                target_groups=("weak", "bright"),
                intensity=1.0,
            )
        )
        self.assertIsNotNone(baseline, baseline_report)
        assert baseline is not None
        baseline_delta = np.max(
            np.abs(
                np.asarray(baseline, dtype=np.float64)
                - np.asarray(stars, dtype=np.float64)
            ),
            axis=0,
        )
        changed_yx = np.argwhere(baseline_delta > 0.0)
        self.assertGreater(changed_yx.shape[0], 0)
        anchor_y, anchor_x = (int(value) for value in changed_yx[0])

        anchored_catalog = copy.deepcopy(catalog)
        saturated_index = int(
            np.flatnonzero(
                anchored_catalog["_psf_valid_flags"]
                & anchored_catalog["_weak_flags"]
            )[0]
        )
        anchored_catalog["_psf_saturated_flags"][saturated_index] = True
        for key, coordinate in (
            ("_peak_y", anchor_y),
            ("_peak_x", anchor_x),
            ("_source_peak_y", anchor_y),
            ("_source_peak_x", anchor_x),
            ("_display_source_peak_y", anchor_y),
            ("_display_source_peak_x", anchor_x),
        ):
            anchored_catalog[key][saturated_index] = coordinate
        overlapping_weak = np.array(weak_mask, copy=True)
        overlapping_bright = np.array(bright_mask, copy=True)
        overlapping_weak[anchor_y, anchor_x] = True
        overlapping_bright[anchor_y, anchor_x] = True

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars,
            base,
            original,
            anchored_catalog,
            cfg,
            support_mask=support,
            weak_mask=overlapping_weak,
            bright_mask=overlapping_bright,
            target_groups=("weak", "bright"),
            intensity=1.0,
        )

        self.assertIsNotNone(pruned, report)
        assert pruned is not None
        np.testing.assert_array_equal(
            pruned[:, anchor_y, anchor_x],
            stars[:, anchor_y, anchor_x],
        )
        self.assertEqual(report["catalog_peak_max_abs_change"], 0.0)
        self.assertEqual(report["source_peak_max_abs_change"], 0.0)
        self.assertEqual(report["global_peak_anchor_max_abs_change"], 0.0)
        self.assertTrue(report["exact_source_pixel_preservation_verified"])

    def test_reference_guided_catalog_coordinate_tamper_fails_closed(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()
        tampered = copy.deepcopy(catalog)
        tampered["_peak_x"][0] = stars.shape[-1] + 5

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars,
            base,
            original,
            tampered,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.0,
        )

        self.assertIsNone(pruned)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("coordinates", report["reason"])

    def test_reference_guided_soft_target_cannot_be_relaxed(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars,
            base,
            original,
            catalog,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.0,
            target_min=0.96,
            target_max=1.06,
        )

        self.assertIsNone(pruned)
        self.assertEqual(report["status"], "unavailable")
        self.assertIn("0.97..1.05", report["reason"])

    def test_reference_guided_rejects_four_channel_layer(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()
        stars_rgba = np.concatenate(
            (stars, np.zeros_like(stars[:1])),
            axis=0,
        )
        base_rgba = np.concatenate(
            (base, np.zeros_like(base[:1])),
            axis=0,
        )
        original_rgba = np.concatenate(
            (original, np.zeros_like(original[:1])),
            axis=0,
        )

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars_rgba,
            base_rgba,
            original_rgba,
            catalog,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.0,
        )

        self.assertIsNone(pruned)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("RGB", report["reason"])

    def test_reference_guided_uses_exact_alpha_clipped_screen(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()
        source_stars = np.divide(
            original - base,
            1.0 - base,
            out=np.zeros_like(original),
            where=(1.0 - base) > 1.0e-8,
        )
        alpha = support.astype(np.float32) * 0.55
        clipped_original = stage9_quality.screen_blend(
            base,
            source_stars,
            1.60,
            alpha_mask=alpha,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            weak_intensity=1.60,
        )

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            stars,
            base,
            clipped_original,
            catalog,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.60,
            weak_intensity=1.60,
            alpha_mask=alpha,
        )

        self.assertIsNotNone(pruned, report)
        self.assertTrue(report["accepted"], report)
        assert pruned is not None
        direct = stage9_quality.assess_unscreen_reference_fidelity(
            clipped_original,
            base,
            pruned,
            intensity=1.60,
            support_mask=support,
            alpha_mask=alpha,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            weak_intensity=1.60,
        )
        self.assertAlmostEqual(
            report["candidate_reference_fidelity"]["support_rgb_mae"],
            direct["support_rgb_mae"],
            places=10,
        )
        self.assertAlmostEqual(
            report["candidate_reference_fidelity"]["support_rgb_p95"],
            direct["support_rgb_p95"],
            places=10,
        )

    def test_reference_guided_asymmetric_pruning_respects_centroid_guard(self):
        (
            stars,
            base,
            original,
            catalog,
            cfg,
            weak_mask,
            bright_mask,
            support,
        ) = self._reference_guided_fixture()
        asymmetric_stars = np.zeros_like(stars)
        for index, (y, x) in enumerate(
            zip(catalog["_peak_y"], catalog["_peak_x"])
        ):
            amplitude = 0.20 if bool(catalog["_weak_flags"][index]) else 0.75
            color = np.asarray(
                [amplitude, amplitude * 0.78, amplitude * 0.56],
                dtype=np.float32,
            )
            asymmetric_stars[:, int(y), int(x)] = color
            for dy, dx in (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ):
                asymmetric_stars[:, int(y) + dy, int(x) + dx] = 0.60 * color
            asymmetric_stars[:, int(y), int(x) + 1] = 0.99 * color
        parent_display = stage9_quality.screen_blend(
            base,
            asymmetric_stars,
            1.0,
        )
        one_sided_original = np.array(parent_display, copy=True)
        asymmetric_scope = np.zeros_like(support)
        for y, x in zip(catalog["_peak_y"], catalog["_peak_x"]):
            asymmetric_scope[int(y), int(x) + 1] = True
        one_sided_original[:, asymmetric_scope] = base[:, asymmetric_scope]

        pruned, report = stage9_quality.prune_star_layer_halfmax_boundaries(
            asymmetric_stars,
            base,
            one_sided_original,
            catalog,
            cfg,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak", "bright"),
            intensity=1.0,
        )

        if pruned is None:
            self.assertFalse(report["accepted"], report)
            self.assertGreater(
                report.get("centroid_guard_skipped_operation_count", 0),
                0,
                report,
            )
        else:
            self.assertTrue(report["centroid_guard_passed"], report)
            self.assertLessEqual(report["centroid_drift_max_px"], 0.05)

    def test_component_operator_tightens_only_weak_group(self) -> None:
        stars, catalog, weak_mask, bright_mask, support = self._fixture()

        contracted, report = stage9_quality.contract_star_layer_components(
            stars,
            catalog,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak",),
            gamma=2.0,
        )

        self.assertIsNotNone(contracted)
        assert contracted is not None
        self.assertTrue(report["changed"], report)
        self.assertEqual(
            report["schema"],
            "starun.stage9-psf-component-contraction.v2",
        )
        self.assertTrue(report["wing_pixels_immutable_by_construction"])
        self.assertEqual(report["preserved_wing_ceiling_fraction"], 0.45)
        self.assertTrue(report["peak_preserved"], report)
        self.assertEqual(report["outside_target_max_abs_change"], 0.0)
        self.assertLessEqual(report["centroid_drift_max_px"], 0.05)
        np.testing.assert_array_equal(
            contracted[:, bright_mask],
            stars[:, bright_mask],
        )
        np.testing.assert_array_equal(
            contracted[:, ~weak_mask],
            stars[:, ~weak_mask],
        )

        weak_y, weak_x = 18, 18
        weak_peak_before = float(np.max(stars[:, weak_y, weak_x]))
        weak_peak_after = float(np.max(contracted[:, weak_y, weak_x]))
        self.assertEqual(weak_peak_after, weak_peak_before)
        frozen_wings = (
            weak_mask
            & (stars[0] > 0.0)
            & (stars[0] <= weak_peak_before * 0.45)
        )
        self.assertTrue(np.any(frozen_wings))
        np.testing.assert_array_equal(
            contracted[:, frozen_wings],
            stars[:, frozen_wings],
        )
        before_area = int(
            np.count_nonzero(stars[0] >= weak_peak_before * 0.5)
        )
        after_area = int(
            np.count_nonzero(contracted[0] >= weak_peak_after * 0.5)
        )
        self.assertLess(after_area, before_area)

        changed = weak_mask & (stars[0] > 1.0e-6)
        channel_gain = contracted[:, changed] / stars[:, changed]
        np.testing.assert_allclose(channel_gain[0], channel_gain[1], atol=1e-7)
        np.testing.assert_allclose(channel_gain[0], channel_gain[2], atol=1e-7)

    def test_component_operator_allows_stronger_centroid_guarded_request(self) -> None:
        stars, catalog, weak_mask, bright_mask, support = self._fixture()

        contracted, report = stage9_quality.contract_star_layer_components(
            stars,
            catalog,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak",),
            gamma=3.2,
        )

        self.assertIsNotNone(contracted)
        self.assertTrue(report["changed"], report)
        self.assertEqual(report["gamma_bounds"], [1.0, 4.0])
        self.assertLessEqual(report["centroid_drift_max_px"], 0.05)
        self.assertTrue(report["peak_preserved"], report)

    def test_scaled_float64_operator_exactly_preserves_immutable_pixels(self):
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        scaled = stars.astype(np.float64) * 65535.0

        contracted, report = stage9_quality.contract_star_layer_components(
            scaled,
            catalog,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak",),
            gamma=2.0,
        )

        self.assertIsNotNone(contracted)
        assert contracted is not None
        self.assertEqual(contracted.dtype, np.dtype(np.float64))
        weak_peak = float(np.max(scaled[:, 18, 18]))
        protected_wing = (
            weak_mask
            & (scaled[0] > 0.0)
            & (scaled[0] <= weak_peak * 0.45)
        )
        np.testing.assert_array_equal(
            contracted[:, ~weak_mask],
            scaled[:, ~weak_mask],
        )
        np.testing.assert_array_equal(
            contracted[:, protected_wing],
            scaled[:, protected_wing],
        )
        np.testing.assert_array_equal(contracted[:, 18, 18], scaled[:, 18, 18])
        np.testing.assert_array_equal(
            contracted[:, catalog["_peak_y"], catalog["_peak_x"]],
            scaled[:, catalog["_peak_y"], catalog["_peak_x"]],
        )
        self.assertEqual(report["outside_target_max_abs_change"], 0.0)
        self.assertEqual(report["protected_wing_max_abs_change"], 0.0)
        self.assertEqual(report["unchanged_scope_max_abs_change"], 0.0)
        self.assertEqual(report["peak_max_abs_drift"], 0.0)
        self.assertTrue(report["exact_source_pixel_preservation_verified"])

    def test_large_only_router_closes_accepted_advisory_but_rejects_hard_failure(self):
        pipeline = SimpleNamespace(
            cfg=SimpleNamespace(stage9_psf_fwhm_ratio_max=1.10)
        )
        quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)

        self.assertEqual(
            stage9_remix._stage9_psf_contraction_target_groups(
                pipeline,
                quality,
            ),
            ("weak",),
        )
        self.assertTrue(
            stage9_remix._stage9_is_psf_large_only_failure(
                pipeline,
                quality,
            )
        )
        hard_failure = _quality(weak=1.18, bright=1.02, all_ratio=1.12)
        self.assertFalse(
            stage9_remix._stage9_is_psf_large_only_failure(
                pipeline,
                hard_failure,
            )
        )
        advisory_accepted = copy.deepcopy(hard_failure)
        advisory_accepted.update(
            accepted=True,
            status="ok",
            issues=[],
            advisories=[
                "PSF ratio accepted within halfmax pixel quantization"
            ],
        )
        self.assertTrue(
            stage9_remix._stage9_is_psf_large_only_failure(
                pipeline,
                advisory_accepted,
            )
        )

    def test_asymmetric_component_uses_maximum_centroid_safe_gamma(self) -> None:
        height = width = 64
        yy, xx = np.indices((height, width))
        weak = 0.30 * np.exp(
            -((yy - 18.0) ** 2 + (xx - 18.0) ** 2) / (2.0 * 2.2**2)
        )
        weak += 0.14 * np.exp(
            -((yy - 18.0) ** 2 + (xx - 22.0) ** 2) / (2.0 * 1.8**2)
        )
        bright = 0.82 * np.exp(
            -((yy - 45.0) ** 2 + (xx - 44.0) ** 2) / (2.0 * 2.2**2)
        )
        scalar = (weak + bright).astype(np.float32)
        stars = np.stack((scalar, scalar * 0.70, scalar * 0.40))
        cfg = SimpleNamespace(
            stage9_star_reference_sigma=3.0,
            stage9_mixed_star_weak_count_min=1,
            stage9_mixed_star_bright_count_min=1,
            stage9_mixed_star_peak_ratio_min=2.0,
        )
        catalog = stage9_quality.build_star_reference_catalog(
            stars,
            cfg,
            background=0.0,
            noise_sigma=0.001,
        )
        weak_mask, bright_mask, support = (
            stage9_quality.build_star_overlay_masks(
                catalog,
                strict=False,
                cfg=cfg,
            )
        )

        contracted, report = stage9_quality.contract_star_layer_components(
            stars,
            catalog,
            support_mask=support,
            weak_mask=weak_mask,
            bright_mask=bright_mask,
            target_groups=("weak",),
            gamma=2.5,
            centroid_drift_max_px=0.002,
        )

        self.assertIsNotNone(contracted)
        self.assertEqual(report["centroid_guard_adjusted_component_count"], 1)
        self.assertEqual(
            report["centroid_moment_compensated_component_count"],
            0,
        )
        self.assertEqual(report["centroid_guard_backoff_component_count"], 1)
        self.assertEqual(report["centroid_guard_skipped_component_count"], 0)
        self.assertEqual(report["centroid_compensation_pixel_count"], 0)
        self.assertEqual(
            report["centroid_guard_strategy"],
            "per_component_uniform_gamma_backoff",
        )
        self.assertGreater(report["effective_gamma"]["median"], 1.0)
        self.assertLess(report["effective_gamma"]["median"], 2.5)
        self.assertLessEqual(report["centroid_drift_max_px"], 0.002)
        self.assertTrue(report["peak_preserved"], report)

    def test_compact_support_feathers_only_the_adjacent_source_ring(self) -> None:
        stars = np.ones((3, 9, 9), dtype=np.float32)
        support = np.zeros((9, 9), dtype=bool)
        support[4, 4] = True

        compact = stage9_quality.apply_compact_starmask_support(
            stars,
            support,
        )

        np.testing.assert_array_equal(compact[:, support], stars[:, support])
        self.assertGreater(float(compact[0, 4, 5]), 0.0)
        self.assertGreater(float(compact[0, 3, 3]), 0.0)
        self.assertLess(float(compact[0, 3, 3]), float(compact[0, 4, 5]))
        self.assertEqual(float(compact[0, 4, 6]), 0.0)

    def test_calibrated_curve_uses_the_same_tapered_support(self) -> None:
        stars = np.full((3, 9, 9), 0.5, dtype=np.float32)
        support = np.zeros((9, 9), dtype=bool)
        support[4, 4] = True
        calibration = {
            "_compact_support_mask": support,
            "anchor_input_values": [1.0e-6, 0.5, 1.0],
            "anchor_output_targets": [0.0, 0.5, 1.0],
            "chroma_regularization_enabled": False,
        }

        curved = stage9_quality.apply_calibrated_starmask(
            stars,
            calibration,
        )

        self.assertEqual(float(curved[0, 4, 4]), 0.5)
        self.assertGreater(float(curved[0, 4, 5]), 0.0)
        self.assertGreater(float(curved[0, 3, 3]), 0.0)
        self.assertLess(float(curved[0, 3, 3]), float(curved[0, 4, 5]))
        self.assertEqual(float(curved[0, 4, 6]), 0.0)

    def test_search_selects_formal_candidate_closest_to_source_psf(self) -> None:
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)
        parent_quality["attempt"] = "parent"
        context = {
            "stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "parent_stars",
            "support_starmask": "parent_stars",
        }
        assessed = (
            _quality(weak=1.06, bright=1.02, all_ratio=1.04),
            _quality(weak=0.98, bright=1.02, all_ratio=1.00),
            _quality(weak=1.01, bright=1.02, all_ratio=1.01),
        )
        immutable_inputs: list[np.ndarray] = []
        real_contract = stage9_quality.contract_star_layer_components

        def contract(source, catalog_arg, **kwargs):
            immutable_inputs.append(np.array(source, copy=True))
            return real_contract(source, catalog_arg, **kwargs)

        with (
            patch.object(
                stage9_quality,
                "contract_star_layer_components",
                side_effect=contract,
            ),
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                return_value=True,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                side_effect=assessed,
            ),
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_targeted_psf_contraction(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertTrue(selected["accepted"], selected)
        self.assertAlmostEqual(selected["recovery_strength"], 1.258668585216)
        self.assertEqual(selected["recovery_target_groups"], ["weak"])
        self.assertEqual(
            len(selected["psf_contraction_candidate_comparison"]),
            3,
        )
        comparisons = selected["psf_contraction_candidate_comparison"]
        self.assertAlmostEqual(comparisons[0]["feedback_input_ratio"], 1.08)
        self.assertIn("feedback_next_gamma", comparisons[0])
        self.assertNotEqual(
            [round(item["gamma"], 4) for item in comparisons],
            [1.75, 2.125, 2.3125],
        )
        self.assertTrue(selected["psf_contraction_rollback"]["selected"])
        self.assertIn("psf_contraction", selected_context)
        self.assertEqual(len(pipeline.applied), 3)
        self.assertEqual(len(immutable_inputs), 3)
        for candidate_input in immutable_inputs:
            np.testing.assert_array_equal(candidate_input, stars)

    def test_reference_guided_transaction_runs_one_full_assessment(self):
        (
            pipeline,
            parent_quality,
            context,
            _candidate_store,
            capture_candidate,
        ) = self._reference_guided_transaction_fixture()
        candidate_quality = _quality(
            weak=1.00,
            bright=1.00,
            all_ratio=1.00,
        )

        with (
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                side_effect=capture_candidate,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                return_value=copy.deepcopy(candidate_quality),
            ) as assess_candidate,
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_targeted_psf_contraction(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertTrue(selected["accepted"], selected)
        self.assertEqual(
            selected["recovery_kind"],
            "reference_guided_halfmax_boundary_pruning",
        )
        self.assertTrue(
            selected["psf_boundary_pruning"]["centroid_guard_passed"]
        )
        self.assertLessEqual(
            selected["psf_boundary_pruning"]["centroid_drift_max_px"],
            0.05,
        )
        comparison = selected["psf_contraction_candidate_comparison"][0]
        self.assertEqual(comparison["full_assessment_count"], 1)
        self.assertEqual(comparison["exact_nonregression_count"], 1)
        self.assertTrue(comparison["candidate_buffer_identity"]["accepted"])
        self.assertTrue(
            comparison["post_assessment_catalog_identity"]["accepted"]
        )
        self.assertEqual(assess_candidate.call_count, 1)
        self.assertEqual(len(pipeline.applied), 2)
        self.assertEqual(
            selected_context["starmask"],
            "starmask_normal_psf_ref_boundary",
        )

    def test_reference_guided_transaction_rejects_missing_parent_fidelity(self):
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.08, bright=1.08, all_ratio=1.08)
        parent_quality.pop("reference_fidelity")
        context = {
            "stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "parent_stars",
            "support_starmask": "parent_stars",
        }

        with patch.object(
            stage9_quality,
            "contract_star_layer_components",
        ) as old_contract:
            selected, selected_context = (
                stage9_remix._stage9_targeted_psf_contraction(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertIs(selected, parent_quality)
        self.assertIs(selected_context, context)
        self.assertEqual(
            selected["psf_contraction_reference_fidelity"]["status"],
            "rejected",
        )
        old_contract.assert_not_called()

    def test_reference_guided_full_psf_tamper_is_restored(self):
        (
            pipeline,
            parent_quality,
            context,
            _candidate_store,
            _capture_candidate,
        ) = self._reference_guided_transaction_fixture()
        original_valid = np.array(
            pipeline._stage9_star_reference_catalog["_psf_valid_flags"],
            copy=True,
        )
        real_prune = stage9_quality.prune_star_layer_halfmax_boundaries

        def mutating_prune(*args, **kwargs):
            candidate, report = real_prune(*args, **kwargs)
            args[3]["_psf_valid_flags"][0] = ~args[3][
                "_psf_valid_flags"
            ][0]
            return candidate, report

        with (
            patch.object(
                stage9_quality,
                "prune_star_layer_halfmax_boundaries",
                side_effect=mutating_prune,
            ),
            patch.object(
                stage9_quality,
                "contract_star_layer_components",
                return_value=(
                    None,
                    {
                        "status": "unavailable",
                        "changed": False,
                        "reason": "stopped after tamper audit",
                    },
                ),
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
            ) as assess_candidate,
        ):
            selected, selected_context = (
                stage9_remix._stage9_targeted_psf_contraction(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertIs(selected, parent_quality)
        self.assertIs(selected_context, context)
        np.testing.assert_array_equal(
            pipeline._stage9_star_reference_catalog["_psf_valid_flags"],
            original_valid,
        )
        boundary = selected["psf_contraction_candidate_comparison"][0]
        self.assertFalse(
            boundary["catalog_coordinate_identity"]["accepted"]
        )
        self.assertTrue(
            boundary["catalog_coordinate_identity"]["restored"]
        )
        assess_candidate.assert_not_called()

    def test_reference_guided_bright_audit_restore_failure_rolls_back(self):
        (
            pipeline,
            parent_quality,
            context,
            _candidate_store,
            capture_candidate,
        ) = self._reference_guided_transaction_fixture()
        parent_quality["source_presence"] = {
            "stage5_bright_star_completion": {"status": "ok"}
        }
        candidate_quality = _quality(
            weak=1.00,
            bright=1.00,
            all_ratio=1.00,
        )
        stable_apply = pipeline._apply_previous_stage_star_remix
        apply_count = 0

        def fail_post_audit_reapply(source_stem, starmask, intensity):
            nonlocal apply_count
            apply_count += 1
            if apply_count == 2:
                return False
            return stable_apply(source_stem, starmask, intensity)

        pipeline._apply_previous_stage_star_remix = fail_post_audit_reapply
        with (
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                side_effect=capture_candidate,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                return_value=copy.deepcopy(candidate_quality),
            ) as assess_candidate,
            patch.object(
                stage9_remix,
                "_stage9_observe_bright_star_presence",
                return_value={"status": "ok", "recovery_ratio": 1.0},
            ),
            patch.object(
                stage9_quality,
                "contract_star_layer_components",
            ) as old_contract,
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_targeted_psf_contraction(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertIs(selected, parent_quality)
        self.assertIs(selected_context, context)
        self.assertTrue(selected["psf_contraction_rollback"]["performed"])
        comparison = selected["psf_contraction_candidate_comparison"][0]
        self.assertEqual(
            comparison["candidate_buffer_identity"]["status"],
            "rejected",
        )
        self.assertFalse(
            comparison["candidate_buffer_identity"]["accepted"]
        )
        self.assertEqual(assess_candidate.call_count, 1)
        old_contract.assert_not_called()

    def test_failed_search_restores_exact_parent(self) -> None:
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)
        parent_quality["attempt"] = "parent"
        context = {
            "stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "parent_stars",
            "support_starmask": "parent_stars",
        }
        with (
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                return_value=True,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                side_effect=lambda *_args, **_kwargs: _quality(
                    weak=1.16,
                    bright=1.02,
                    all_ratio=1.11,
                ),
            ),
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_targeted_psf_contraction(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertIs(selected, parent_quality)
        self.assertIs(selected_context, context)
        self.assertTrue(selected["psf_contraction_rollback"]["performed"])
        self.assertEqual(
            selected["psf_contraction_rollback"]["restored"],
            "immutable_parent",
        )
        self.assertEqual(
            pipeline.loaded[-1],
            "stage9_candidate_normal_000_psf_contraction_parent",
        )
        np.testing.assert_array_equal(pipeline._stage9_last_star_layer, stars)

    def test_post_source_presence_closure_selects_soft_target_candidate(self):
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)
        parent_quality.update(
            attempt="screen_unscreen_source_presence_95",
            accepted=True,
            status="advisory",
            issues=[],
        )
        parent_quality["metrics"]["star_wing_recovery_ratio"] = 0.94
        context = {
            "available": True,
            "stars": stars,
            "unscreen_stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "starmask_unscreen_normal",
            "support_starmask": "starmask_stretched",
            "support_mode": "normal",
        }
        candidate_quality = _quality(
            weak=1.03,
            bright=1.01,
            all_ratio=1.02,
        )
        candidate_quality["metrics"]["star_wing_recovery_ratio"] = 0.94
        attempts: list[dict] = []

        with (
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                return_value=True,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                side_effect=lambda *_args, **kwargs: {
                    **copy.deepcopy(candidate_quality),
                    "attempt": kwargs["attempt"],
                    "formula": kwargs["formula"],
                },
            ),
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_close_accepted_psf_soft_target(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=attempts,
                    review_candidate_registry=[],
                )
            )

        self.assertIn("_psf_contract_", selected["attempt"])
        self.assertEqual(
            selected["parent_attempt"],
            "screen_unscreen_source_presence_95",
        )
        self.assertEqual(
            stage9_remix._stage9_psf_group_ratios(selected),
            {"all": 1.02, "weak": 1.03, "bright": 1.01},
        )
        self.assertTrue(selected["psf_soft_target_closure"]["accepted"])
        self.assertEqual(
            selected["psf_soft_target_closure"]["status"],
            "closed",
        )
        self.assertIsNot(selected_context, context)
        self.assertTrue(attempts)

    def test_final_primary_screen_selection_runs_common_soft_target_closure(self):
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        pipeline._stage9_last_star_overlay_mask = np.array(support, copy=True)
        pipeline._stage9_last_weak_overlay_mask = np.array(weak_mask, copy=True)
        pipeline._stage9_last_bright_overlay_mask = np.array(
            bright_mask,
            copy=True,
        )
        parent_quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)
        parent_quality.update(
            attempt="screen_primary",
            accepted=True,
            status="advisory",
            issues=[],
            intensity=1.0,
            starmask="starmask_stretched",
            support_mode="normal",
            support_starmask="starmask_stretched",
        )
        candidate_quality = _quality(
            weak=1.03,
            bright=1.01,
            all_ratio=1.02,
        )

        with (
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                return_value=True,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                side_effect=lambda *_args, **kwargs: {
                    **copy.deepcopy(candidate_quality),
                    "attempt": kwargs["attempt"],
                    "formula": kwargs["formula"],
                },
            ),
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_close_final_selected_psf_soft_target(
                    pipeline,
                    source_stem="stage8_enhanced",
                    selected_quality=parent_quality,
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        assert selected is not None
        self.assertTrue(selected["accepted"], selected)
        self.assertIn("_psf_contract_", selected["attempt"])
        self.assertEqual(
            selected["psf_soft_target_closure"]["selection_scope"],
            "final_selected_remix",
        )
        self.assertEqual(
            stage9_remix._stage9_psf_group_ratios(selected),
            {"all": 1.02, "weak": 1.03, "bright": 1.01},
        )
        self.assertEqual(
            selected_context["starmask"],
            selected["starmask"],
        )

    def test_post_source_presence_wing_regression_restores_exact_parent(self):
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)
        parent_quality.update(
            attempt="screen_unscreen_source_presence_95",
            accepted=True,
            status="advisory",
            issues=[],
        )
        parent_quality["metrics"]["star_wing_recovery_ratio"] = 0.9400
        context = {
            "available": True,
            "stars": stars,
            "unscreen_stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "starmask_unscreen_normal",
            "support_starmask": "starmask_stretched",
            "support_mode": "normal",
        }
        candidate_quality = _quality(
            weak=1.03,
            bright=1.01,
            all_ratio=1.02,
        )
        candidate_quality["metrics"]["star_wing_recovery_ratio"] = float(
            np.nextafter(0.9400, 0.0)
        )

        with (
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                return_value=True,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                side_effect=lambda *_args, **kwargs: {
                    **copy.deepcopy(candidate_quality),
                    "attempt": kwargs["attempt"],
                    "formula": kwargs["formula"],
                },
            ),
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_close_accepted_psf_soft_target(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertIs(selected, parent_quality)
        self.assertIs(selected_context, context)
        self.assertTrue(selected["psf_contraction_rollback"]["performed"])
        self.assertFalse(selected["psf_soft_target_closure"]["accepted"])
        self.assertEqual(
            selected["psf_soft_target_closure"]["status"],
            "rolled_back_unclosed",
        )
        wing_comparison = selected["psf_contraction_candidate_comparison"][0][
            "nonregression"
        ]["metrics"]["star_wing_recovery_ratio"]
        self.assertEqual(wing_comparison["tolerance"], 0.0)
        self.assertFalse(wing_comparison["accepted"])
        self.assertEqual(
            pipeline.loaded[-1],
            "stage9_candidate_normal_000_psf_soft_target_parent",
        )
        np.testing.assert_array_equal(pipeline._stage9_last_star_layer, stars)

    def test_weak_soft_target_without_all_group_rolls_back_exact_parent(self):
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)
        parent_quality.update(
            attempt="screen_primary",
            accepted=True,
            status="advisory",
            issues=[],
        )
        context = {
            "available": True,
            "stars": stars,
            "unscreen_stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "starmask_stretched",
            "support_starmask": "starmask_stretched",
            "support_mode": "normal",
        }
        candidate_quality = _quality(
            weak=1.03,
            bright=1.01,
            all_ratio=1.08,
        )

        with (
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
                return_value=True,
            ),
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
                side_effect=lambda *_args, **kwargs: {
                    **copy.deepcopy(candidate_quality),
                    "attempt": kwargs["attempt"],
                    "formula": kwargs["formula"],
                },
            ),
            patch.object(stage9_remix, "_stage9_consider_review_candidate"),
        ):
            selected, selected_context = (
                stage9_remix._stage9_close_accepted_psf_soft_target(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertIs(selected, parent_quality)
        self.assertIs(selected_context, context)
        self.assertFalse(selected["accepted"])
        self.assertFalse(selected["psf_soft_target_closure"]["accepted"])
        self.assertEqual(
            selected["psf_soft_target_closure"]["status"],
            "rolled_back_unclosed",
        )
        comparisons = selected["psf_contraction_candidate_comparison"]
        self.assertTrue(comparisons)
        self.assertTrue(
            all(not item["transaction_eligible"] for item in comparisons)
        )
        self.assertTrue(
            all(item["fwhm_ratios"]["all"] == 1.08 for item in comparisons)
        )
        self.assertEqual(
            pipeline.loaded[-1],
            "stage9_candidate_normal_000_psf_soft_target_parent",
        )
        np.testing.assert_array_equal(pipeline._stage9_last_star_layer, stars)

    def test_catalog_coordinate_mutation_fails_closed_and_is_restored(self):
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.08, bright=1.02, all_ratio=1.06)
        parent_quality.update(
            attempt="screen_unscreen_source_presence_95",
            accepted=True,
            status="advisory",
            issues=[],
        )
        original_peak_x = np.array(catalog["_peak_x"], copy=True)
        context = {
            "available": True,
            "stars": stars,
            "unscreen_stars": stars,
            "support_mask": support,
            "weak_mask": weak_mask,
            "bright_mask": bright_mask,
            "starmask": "starmask_unscreen_normal",
            "support_starmask": "starmask_stretched",
            "support_mode": "normal",
        }
        real_contract = stage9_quality.contract_star_layer_components

        def mutating_contract(source, catalog_arg, **kwargs):
            catalog_arg["_peak_x"][0] += 1
            return real_contract(source, catalog_arg, **kwargs)

        with (
            patch.object(
                stage9_quality,
                "contract_star_layer_components",
                side_effect=mutating_contract,
            ),
            patch.object(
                stage9_remix,
                "_save_stage9_candidate_star_layer",
            ) as save_candidate,
            patch.object(
                stage9_remix,
                "_assess_stage9_candidate",
            ) as assess_candidate,
        ):
            selected, selected_context = (
                stage9_remix._stage9_close_accepted_psf_soft_target(
                    pipeline,
                    source_stem="stage8_enhanced",
                    parent_quality=parent_quality,
                    parent_context=context,
                    intensity=1.0,
                    support_mode="normal",
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertIs(selected, parent_quality)
        self.assertIs(selected_context, context)
        np.testing.assert_array_equal(catalog["_peak_x"], original_peak_x)
        self.assertTrue(selected["psf_contraction_rollback"]["performed"])
        comparison = selected["psf_contraction_candidate_comparison"][0]
        self.assertIn(
            "psf_contraction_catalog_coordinates_changed",
            comparison["issues"],
        )
        self.assertTrue(
            comparison["catalog_coordinate_identity"]["restored"]
        )
        save_candidate.assert_not_called()
        assess_candidate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
