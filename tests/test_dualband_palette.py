from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from dualband_palette import (  # noqa: E402
    PALETTE_CHANNELS,
    build_dualband_palette_candidate,
    compose_palette,
    derive_classic_dualband_channels,
    evaluate_palette_quality_metrics,
    resolve_palette_selection,
    select_primary_target_palette,
    stage8_palette_eligibility,
)


class DualbandPaletteSelectionTests(unittest.TestCase):
    def test_frozen_primary_target_selects_only_first_choice(self) -> None:
        cases = (
            ({"type": "generic_low_snr_safe"}, "HOO"),
            ({"type": "emission_nebula_widefield"}, "SHO"),
            ({"type": "dark_nebula_low_contrast"}, "HSO"),
            ({"type": "bright_emission_reflection_nebula"}, "HOS"),
            ({"type": "planetary_nebula"}, "OHS"),
            (
                {
                    "name": "Veil Nebula",
                    "type": "emission_nebula_widefield",
                },
                "OSH",
            ),
        )

        for target, expected in cases:
            with self.subTest(target=target):
                selection = select_primary_target_palette(target)
                self.assertEqual(selection["palette"], expected)
                self.assertEqual(selection["selected_rank"], 1)
                self.assertEqual(selection["candidate_count"], 1)
                self.assertEqual(
                    selection["selection_mode"],
                    "frozen_primary_target_first_choice",
                )

    def test_auto_resolution_preserves_target_first_choice(self) -> None:
        selection = resolve_palette_selection(
            {
                "type": "emission_nebula_widefield",
                "confidence": 0.95,
                "method": "catalog",
                "frozen": True,
            },
            " AUTO ",
        )

        self.assertEqual(selection["requested_palette"], "auto")
        self.assertEqual(selection["automatic_palette"], "SHO")
        self.assertEqual(selection["palette"], "SHO")
        self.assertEqual(selection["selection_mode"], "automatic_target_mapping")
        self.assertFalse(selection["manual_override"])
        self.assertTrue(selection["target"]["frozen"])

    def test_explicit_palette_forces_each_regular_mapping(self) -> None:
        target = {
            "type": "emission_nebula_widefield",
            "confidence": 0.95,
            "method": "catalog",
            "frozen": True,
        }
        for palette in PALETTE_CHANNELS:
            with self.subTest(palette=palette):
                selection = resolve_palette_selection(
                    target,
                    f"  {palette.lower()}  ",
                )
                self.assertEqual(selection["requested_palette"], palette)
                self.assertEqual(selection["automatic_palette"], "SHO")
                self.assertEqual(selection["palette"], palette)
                self.assertEqual(
                    selection["selection_mode"],
                    "explicit_user_palette",
                )
                self.assertTrue(selection["manual_override"])
                self.assertEqual(selection["selected_rank"], 1)
                self.assertEqual(selection["candidate_count"], 1)

    def test_invalid_explicit_palette_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dual-band palette"):
            resolve_palette_selection({"type": "planetary_nebula"}, "XYZ")

    def test_regular_classic_formula_and_six_permutations_are_exact(self) -> None:
        rgb = np.array(
            [
                [[0.50, 0.20]],
                [[0.30, 0.10]],
                [[0.40, 0.20]],
            ],
            dtype=np.float32,
        )
        channels = derive_classic_dualband_channels(rgb)

        np.testing.assert_allclose(channels["H"], [[0.50, 0.20]], atol=1e-7)
        np.testing.assert_allclose(channels["O"], [[0.35, 0.15]], atol=1e-7)
        np.testing.assert_allclose(channels["S"], [[0.425, 0.175]], atol=1e-7)
        for palette, names in PALETTE_CHANNELS.items():
            expected = np.stack([channels[name] for name in names])
            np.testing.assert_array_equal(
                compose_palette(channels, palette),
                expected,
            )

    def test_eligibility_is_fail_closed(self) -> None:
        mapping = {
            "schema": "starun.narrowband-channel-mapping.v1",
            "mapping": "osc_hoo_rgb",
            "ha_channel": "R",
            "oiii_channels": ["G", "B"],
            "confidence": 0.97,
        }
        accepted = stage8_palette_eligibility(
            enabled=True,
            channel_semantics="narrowband_composite",
            mapping=mapping,
            mapping_confidence_min=0.85,
            processing_policy="full",
            stage8_quality="ok",
            stage7_accepted=True,
            external_override=False,
        )
        self.assertTrue(accepted["eligible"])

        rejected = stage8_palette_eligibility(
            enabled=True,
            channel_semantics="narrowband_composite",
            mapping={**mapping, "confidence": 0.76},
            mapping_confidence_min=0.85,
            processing_policy="limited",
            stage8_quality="degraded",
            stage7_accepted=False,
            external_override=True,
        )
        self.assertFalse(rejected["eligible"])
        self.assertIn("ha_oiii_mapping_unconfirmed", rejected["issues"])
        self.assertIn("stage8_policy_not_full", rejected["issues"])
        self.assertIn(
            "external_starless_mapping_provenance_unverified",
            rejected["issues"],
        )

        nonfinite = stage8_palette_eligibility(
            enabled=True,
            channel_semantics="narrowband_composite",
            mapping={**mapping, "confidence": float("nan")},
            mapping_confidence_min=0.85,
            processing_policy="full",
            stage8_quality="ok",
            stage7_accepted=True,
            external_override=False,
        )
        self.assertFalse(nonfinite["eligible"])
        self.assertEqual(nonfinite["mapping_confidence"], 0.0)

        malformed = stage8_palette_eligibility(
            enabled=True,
            channel_semantics="narrowband_composite",
            mapping={**mapping, "oiii_channels": ["B", "G"]},
            mapping_confidence_min=0.85,
            processing_policy="full",
            stage8_quality="ok",
            stage7_accepted=True,
            external_override=False,
        )
        self.assertFalse(malformed["eligible"])
        self.assertIn("oiii_channels_not_green_blue", malformed["issues"])


class DualbandPaletteCandidateTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[np.ndarray, dict[str, np.ndarray]]:
        height, width = 96, 128
        y_grid, x_grid = np.mgrid[:height, :width]
        ha = np.exp(
            -(((x_grid - 68) / 30) ** 2 + ((y_grid - 48) / 22) ** 2)
        )
        oiii = np.exp(
            -(((x_grid - 52) / 24) ** 2 + ((y_grid - 42) / 25) ** 2)
        )
        image = np.empty((3, height, width), dtype=np.float32)
        image[0] = 0.040 + 0.42 * ha
        image[1] = 0.050 + 0.24 * oiii
        image[2] = 0.055 + 0.31 * oiii

        nebula = np.clip((ha - 0.08) / 0.75, 0.0, 1.0).astype(np.float32)
        faint = np.clip(
            (np.maximum(ha, oiii) - 0.02) / 0.40,
            0.0,
            1.0,
        ).astype(np.float32)
        faint *= 1.0 - 0.70 * nebula
        core = np.clip((ha - 0.72) / 0.28, 0.0, 1.0).astype(np.float32)
        subject = np.maximum.reduce((core, nebula, faint))
        background = np.clip(1.0 - 1.50 * subject, 0.0, 1.0).astype(
            np.float32
        )
        return image, {
            "core_mask": core,
            "nebula_mask": nebula,
            "faint_nebula_mask": faint,
            "background_mask": background,
        }

    def test_candidate_is_masked_luminance_safe_and_nonmutating(self) -> None:
        image, masks = self._fixture()
        original = image.copy()

        candidate, report = build_dualband_palette_candidate(
            image,
            palette="SHO",
            **masks,
        )

        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(candidate, image))
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["role"], "artistic_false_color")
        self.assertTrue(report["synthetic_sii"])
        self.assertEqual(report["formula"]["mode"], "regular")
        self.assertEqual(report["formula"]["preset"], "Classic")
        self.assertGreater(report["metrics"]["subject_change_p95"], 0.01)
        self.assertLessEqual(
            report["metrics"]["luminance_drift_p95"],
            report["limits"]["luminance_drift_p95_max"],
        )
        self.assertEqual(report["metrics"]["background_change_p95"], 0.0)
        self.assertEqual(
            report["metrics"]["outside_subject_max_abs_change"],
            0.0,
        )
        background = masks["background_mask"] >= 0.80
        np.testing.assert_array_equal(candidate[:, background], image[:, background])
        json.dumps(report)

    def test_quality_excess_within_fifty_percent_is_warning_only(self) -> None:
        quality = evaluate_palette_quality_metrics(
            {
                "luminance_drift_p95": 0.0074,
                "clip_growth": 0.0029,
                "outside_subject_max_abs_change": 0.0,
                "background_change_p95": 0.0,
            },
            luma_drift_p95_max=0.005,
            clip_growth_max=0.002,
            warning_tolerance=0.50,
        )

        self.assertTrue(quality["accepted"])
        self.assertEqual(quality["status"], "accepted_with_warning")
        self.assertIn(
            "luminance_drift_within_warning_tolerance",
            quality["warnings"],
        )
        self.assertIn(
            "clip_growth_within_warning_tolerance",
            quality["warnings"],
        )

    def test_quality_excess_beyond_fifty_percent_still_rejects(self) -> None:
        quality = evaluate_palette_quality_metrics(
            {
                "luminance_drift_p95": 0.0076,
                "clip_growth": 0.0031,
                "outside_subject_max_abs_change": 0.0,
                "background_change_p95": 0.0,
            },
            luma_drift_p95_max=0.005,
            clip_growth_max=0.002,
            warning_tolerance=0.50,
        )

        self.assertFalse(quality["accepted"])
        self.assertIn("luminance_drift", quality["issues"])
        self.assertIn("clip_growth", quality["issues"])


if __name__ == "__main__":
    unittest.main()
