#!/usr/bin/env python3
"""Focused Stage 6-7 tests for strict bright-core nebula protection."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from models import PipelineConfig  # noqa: E402
import stage7_quality  # noqa: E402
import stage7_stretch_metrics  # noqa: E402
import syqon_starless  # noqa: E402

if "sirilpy.exceptions" not in sys.modules:
    fake_sirilpy = types.ModuleType("sirilpy")
    fake_exceptions = types.ModuleType("sirilpy.exceptions")
    fake_enums = types.ModuleType("sirilpy.enums")
    fake_exceptions.CommandError = RuntimeError
    fake_exceptions.DataError = RuntimeError
    fake_exceptions.SirilConnectionError = RuntimeError
    fake_exceptions.SirilError = RuntimeError
    fake_enums.CommandStatus = type(
        "CommandStatus",
        (),
        {"CMD_GENERIC_ERROR": 1, "CMD_THREAD_RUNNING": 2},
    )
    fake_sirilpy.SirilInterface = object
    fake_sirilpy.exceptions = fake_exceptions
    sys.modules.setdefault("sirilpy", fake_sirilpy)
    sys.modules.setdefault("sirilpy.exceptions", fake_exceptions)
    sys.modules.setdefault("sirilpy.enums", fake_enums)

from stages import stage6_star_separation  # noqa: E402


STRICT_PROFILE = {
    "secondary_labels": ["bright_core", "large_nebulosity"],
    "features": {"bright_core": True},
    "risks": {"core_blowout": "high"},
}


def _linear_core(size: int = 128):
    half = size // 2
    yy, xx = np.mgrid[-half : size - half, -half : size - half]
    gray = 0.02 + 0.55 * np.exp(
        -(xx * xx + yy * yy) / (2.0 * (size * 0.14) ** 2)
    )
    source = np.stack([gray, gray * 0.92, gray * 0.85]).astype(np.float32)
    return source, np.zeros_like(source)


class BrightCoreStage6IntegrityTests(unittest.TestCase):
    def _assess(self, source, starless, starmask, profile=STRICT_PROFILE):
        return stage7_quality.assess_bright_core_integrity(
            source,
            starless,
            starmask,
            target_type="bright_emission_reflection_nebula",
            target_profile=profile,
        )

    def test_normal_bright_core_is_safe(self):
        source, starmask = _linear_core()
        report = self._assess(source, source.copy(), starmask)

        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(report["roi"]["support"], 64)
        self.assertFalse(report["hard_failed"])

    def test_m42_style_channel_platform_is_hard_rejected(self):
        source, starmask = _linear_core()
        roi, _evidence = stage7_quality.build_bright_core_roi(source, starmask)
        starless = source.copy()
        coordinates = np.argwhere(roi)[:48]
        starless[0, coordinates[:, 0], coordinates[:, 1]] = 1.0

        report = self._assess(source, starless, starmask)

        self.assertEqual(report["status"], "hard_failed")
        self.assertTrue(
            report["gates"]["new_channel_cap_ratio_max"]["hard_failed"]
        )
        self.assertTrue(
            report["gates"]["largest_cap_component_ratio"]["hard_failed"]
        )

    def test_two_by_two_phase_artifact_is_reported(self):
        source, starmask = _linear_core()
        roi, _evidence = stage7_quality.build_bright_core_roi(source, starmask)
        starless = source.copy()
        checker = np.indices(roi.shape).sum(axis=0) % 2 == 0
        starless[0, roi & checker] += 0.012
        starless[0, roi & ~checker] -= 0.012

        report = self._assess(source, starless, starmask)

        self.assertTrue(
            report["gates"]["parity_phase_span_max"]["hard_failed"]
        )

    def test_preexisting_source_clipping_is_not_new_clipping(self):
        source, starmask = _linear_core()
        roi, _evidence = stage7_quality.build_bright_core_roi(source, starmask)
        source[:, roi] = 1.0

        report = self._assess(source, source.copy(), starmask)

        self.assertEqual(
            report["metrics"]["new_channel_cap_ratio_max"],
            0.0,
        )
        self.assertEqual(report["status"], "ok")

    def test_confirmed_star_mask_excludes_local_platform(self):
        source, starmask = _linear_core()
        roi, _evidence = stage7_quality.build_bright_core_roi(source, starmask)
        coordinates = np.argwhere(roi)[:12]
        starless = source.copy()
        starless[0, coordinates[:, 0], coordinates[:, 1]] = 1.0
        starmask[:, coordinates[:, 0], coordinates[:, 1]] = 0.03

        report = self._assess(source, starless, starmask)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(
            report["metrics"]["new_channel_cap_ratio_max"],
            0.0,
        )

    def test_insufficient_roi_is_hard_rejected(self):
        source, starmask = _linear_core(16)
        report = self._assess(source, source.copy(), starmask)

        self.assertEqual(report["status"], "hard_failed")
        self.assertIn("roi_support_insufficient", report["trigger_reasons"])

    def test_m8_like_non_strict_profile_keeps_existing_path(self):
        source, starmask = _linear_core()
        profile = {
            "secondary_labels": [
                "bright_core",
                "large_nebulosity",
                "emission_red",
            ],
            "features": {"bright_core": True},
            "risks": {"core_blowout": "medium"},
        }
        report = self._assess(source, source.copy(), starmask, profile=profile)
        retry_plan = (
            stage6_star_separation._stage6_bright_core_retry_plan(
                {"status": "ok", "bright_core_integrity": report},
                retry_max=1,
                syqon_available=True,
            )
        )

        self.assertFalse(report["applicable"])
        self.assertEqual(report["status"], "not_applicable")
        self.assertFalse(retry_plan["triggered"])
        self.assertFalse(retry_plan["should_attempt"])
        self.assertEqual(retry_plan["status"], "not_triggered")

    def test_high_core_blowout_is_strict_without_bright_core_label(self):
        source, starmask = _linear_core()
        profile = {
            "secondary_labels": ["large_nebulosity", "emission_red"],
            "features": {"bright_core": False},
            "risks": {"core_blowout": "high"},
        }

        report = self._assess(source, source.copy(), starmask, profile=profile)

        self.assertTrue(report["applicable"])
        self.assertEqual(report["status"], "ok")
        self.assertFalse(
            report["strict_target_evidence"]["bright_core_label"]
        )

    def test_recovery_profile_is_fixed_ihs_fp32_subtraction(self):
        manifest = syqon_starless.SYQON_BRIGHT_CORE_RECOVERY_PROFILE.manifest()

        self.assertEqual(manifest["profile_id"], "zenith_bright_core_ihs_recovery")
        self.assertEqual(manifest["stretch_method"], "ihs")
        self.assertEqual(manifest["target_median"], 0.15)
        self.assertEqual(manifest["tile_size"], 512)
        self.assertEqual(manifest["overlap"], 64)
        self.assertEqual(manifest["precision"], "fp32")
        self.assertEqual(manifest["mask_method"], "subtraction")

    def test_rejected_pair_purge_clears_pointer_and_memory_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            process_dir = Path(directory)
            for name in ("starless.fit", "starmask.fit", "stage6_syqon_selected.json"):
                (process_dir / name).write_bytes(b"test")
            pipeline = SimpleNamespace(
                process_dir=process_dir,
                work_dir=None,
                starless_file=process_dir / "starless.fit",
                starmask_file=process_dir / "starmask.fit",
                _selected_syqon_pair_id="pair-id",
                _selected_syqon_attempt_id="attempt-id",
                _stage6_pair_handoff={"pair_id": "pair-id"},
            )

            removed = syqon_starless.purge_unaccepted_star_separation_outputs(
                pipeline
            )

            self.assertIn("stage6_syqon_selected.json", removed)
            self.assertIsNone(pipeline.starless_file)
            self.assertIsNone(pipeline.starmask_file)
            self.assertIsNone(pipeline._selected_syqon_pair_id)
            self.assertIsNone(pipeline._selected_syqon_attempt_id)
            self.assertIsNone(pipeline._stage6_pair_handoff)

    def test_failed_recovery_contract_is_review_only_and_never_hdr_eligible(self):
        pipeline = SimpleNamespace(
            _active_target_type=lambda: "bright_emission_reflection_nebula",
            target_profile=STRICT_PROFILE,
            _selected_syqon_pair_id="pair-live",
        )
        quality = {
            "attempt": "bright_core_ihs_recovery",
            "pair_id": "pair-rejected",
            "bright_core_integrity": {
                "applicable": True,
                "hard_failed": True,
                "trigger_reasons": ["new_channel_cap_ratio_max"],
                "roi": {"available": True, "support": 128, "reason": "ok"},
                "gates": {
                    "new_channel_cap_ratio_max": {"hard_failed": True}
                },
            },
        }

        contract = (
            stage6_star_separation._stage6_bright_core_with_stars_fallback_contract(
                pipeline,
                quality,
                {"attempted": True, "accepted": False, "status": "rejected"},
                separation_accepted=False,
            )
        )

        self.assertFalse(contract["eligible"])
        self.assertFalse(contract["accepted"])
        self.assertTrue(contract["review_only"])
        self.assertEqual(contract["status"], "rejected_to_review")
        self.assertEqual(contract["review_output"], "stage7_review_with_stars")
        self.assertEqual(contract["rejected_pair_id"], "pair-rejected")

    def test_insufficient_roi_contract_still_routes_to_review_only(self):
        pipeline = SimpleNamespace(
            _active_target_type=lambda: "bright_emission_reflection_nebula",
            target_profile=STRICT_PROFILE,
            _selected_syqon_pair_id="pair-insufficient-roi",
        )
        quality = {
            "attempt": "zenith_baseline",
            "pair_id": "pair-insufficient-roi",
            "bright_core_integrity": {
                "applicable": True,
                "hard_failed": True,
                "trigger_reasons": ["roi_support_insufficient"],
                "roi": {
                    "available": False,
                    "support": 24,
                    "reason": "roi_support_insufficient",
                },
                "gates": {},
            },
        }

        contract = (
            stage6_star_separation._stage6_bright_core_with_stars_fallback_contract(
                pipeline,
                quality,
                {"attempted": False, "status": "direct_reject"},
                separation_accepted=False,
            )
        )

        self.assertEqual(contract["status"], "rejected_to_review")
        self.assertTrue(contract["review_only"])
        self.assertEqual(contract["source_stem"], "stage6_input")
        self.assertIn("roi_support_insufficient", contract["blocked_reasons"])

    def test_ihs_ordinary_quality_failure_still_routes_to_review_only(self):
        pipeline = SimpleNamespace(
            _active_target_type=lambda: "bright_emission_reflection_nebula",
            target_profile=STRICT_PROFILE,
            _selected_syqon_pair_id="ihs-pair",
        )
        quality = {
            "attempt": "bright_core_ihs_recovery",
            "pair_id": "ihs-pair",
            "status": "poor",
            "bright_core_integrity": {
                "applicable": True,
                "hard_failed": False,
                "trigger_reasons": [],
                "roi": {"available": True, "support": 128, "reason": "ok"},
                "gates": {},
            },
        }

        contract = (
            stage6_star_separation._stage6_bright_core_with_stars_fallback_contract(
                pipeline,
                quality,
                {
                    "attempted": True,
                    "accepted": False,
                    "status": "rejected",
                    "trigger_reasons": ["new_channel_cap_ratio_max"],
                },
                separation_accepted=False,
            )
        )

        self.assertEqual(contract["status"], "rejected_to_review")
        self.assertTrue(contract["review_only"])
        self.assertIn("new_channel_cap_ratio_max", contract["trigger_reasons"])


class BrightCoreStage7IntegrityTests(unittest.TestCase):
    def _assess(self, candidate, *, source=None, starmask=None, available=True):
        if source is None or starmask is None:
            default_source, default_starmask = _linear_core()
            source = default_source if source is None else source
            starmask = default_starmask if starmask is None else starmask
        return stage7_stretch_metrics.assess_target_local_stretch(
            source,
            candidate,
            "bright_emission_reflection_nebula",
            PipelineConfig(),
            target_profile=STRICT_PROFILE,
            starmask=starmask,
            frozen_reference_available=available,
        )

    def test_channel_clipping_and_colored_platform_are_hard_gates(self):
        source, starmask = _linear_core()
        roi, _evidence = stage7_quality.build_bright_core_roi(source, starmask)
        candidate = np.clip(source * 1.5, 0.0, 0.95)
        coordinates = np.argwhere(roi)[:8]
        candidate[0, coordinates[:, 0], coordinates[:, 1]] = 1.0

        report = self._assess(candidate, source=source, starmask=starmask)

        self.assertTrue(
            report["quality_gates"]["local_core_clip_ratio"]["hard_failed"]
        )
        self.assertTrue(
            report["quality_gates"][
                "local_core_colored_plateau_component_ratio"
            ]["hard_failed"]
        )

    def test_core_phase_span_is_a_hard_gate(self):
        source, starmask = _linear_core()
        roi, _evidence = stage7_quality.build_bright_core_roi(source, starmask)
        candidate = np.clip(source * 1.4, 0.0, 0.90)
        phase = np.indices(roi.shape).sum(axis=0) % 2 == 0
        candidate[1, roi & phase] += 0.02
        candidate[1, roi & ~phase] -= 0.02

        report = self._assess(candidate, source=source, starmask=starmask)

        self.assertTrue(
            report["quality_gates"][
                "local_core_parity_phase_span"
            ]["hard_failed"]
        )

    def test_missing_frozen_reference_is_non_overridable(self):
        source, starmask = _linear_core()
        report = self._assess(
            source.copy(),
            source=source,
            starmask=starmask,
            available=False,
        )

        self.assertFalse(report["accepted"])
        self.assertTrue(
            report["quality_gates"][
                "local_core_reference_available"
            ]["hard_failed"]
        )

if __name__ == "__main__":
    unittest.main()
