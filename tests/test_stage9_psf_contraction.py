from __future__ import annotations

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

    def test_large_only_router_targets_failed_group_and_rejects_mixed_failure(self):
        pipeline = SimpleNamespace(
            cfg=SimpleNamespace(stage9_psf_fwhm_ratio_max=1.10)
        )
        quality = _quality(weak=1.18, bright=1.02, all_ratio=1.12)

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
        mixed = {**quality, "issues": [*quality["issues"], "background lift"]}
        self.assertFalse(
            stage9_remix._stage9_is_psf_large_only_failure(
                pipeline,
                mixed,
            )
        )

    def test_search_selects_formal_candidate_closest_to_source_psf(self) -> None:
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.18, bright=1.02, all_ratio=1.12)
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
        self.assertAlmostEqual(selected["recovery_strength"], 1.9375)
        self.assertEqual(selected["recovery_target_groups"], ["weak"])
        self.assertEqual(
            len(selected["psf_contraction_candidate_comparison"]),
            3,
        )
        self.assertTrue(selected["psf_contraction_rollback"]["selected"])
        self.assertIn("psf_contraction", selected_context)
        self.assertEqual(len(pipeline.applied), 3)
        self.assertEqual(len(immutable_inputs), 3)
        for candidate_input in immutable_inputs:
            np.testing.assert_array_equal(candidate_input, stars)

    def test_failed_search_restores_exact_parent(self) -> None:
        stars, catalog, weak_mask, bright_mask, support = self._fixture()
        pipeline = _FakePipeline(stars, catalog)
        parent_quality = _quality(weak=1.18, bright=1.02, all_ratio=1.12)
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


if __name__ == "__main__":
    unittest.main()
