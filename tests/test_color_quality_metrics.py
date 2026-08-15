from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from color_quality_metrics import (  # noqa: E402
    build_color_quality_report,
    physical_broadband_anchor_accepted,
    resolve_color_contract,
)


class ColorContractTests(unittest.TestCase):
    def test_accepted_broadband_physical_color_freezes_anchor(self) -> None:
        channel_profile = {
            "kind": "broadband_rgb_osc",
            "axes": {"spectral": {"kind": "broadband"}},
        }
        color_report = {
            "physical_color": {
                "accepted": True,
                "feeds_main_pipeline": True,
                "method": "SPCC",
            }
        }

        contract = resolve_color_contract(
            channel_profile=channel_profile,
            color_report=color_report,
        )

        self.assertEqual(contract["rendition_intent"], "photometrically_anchored")
        self.assertTrue(contract["physical_anchor_accepted"])
        self.assertFalse(
            contract["operation_policy"]["repeat_global_white_balance"]
        )
        self.assertFalse(contract["operation_policy"]["unconditional_scnr"])
        self.assertTrue(
            physical_broadband_anchor_accepted(channel_profile, color_report)
        )

    def test_dualband_palette_is_disclosed_as_artistic(self) -> None:
        contract = resolve_color_contract(
            channel_profile={"kind": "narrowband_composite"},
            color_report={},
            palette_report={
                "accepted": True,
                "palette": "SHO",
                "synthetic_sii": True,
            },
        )

        self.assertEqual(contract["rendition_intent"], "artistic_false_color")
        self.assertEqual(contract["disclosure"]["palette"], "SHO")
        self.assertTrue(contract["disclosure"]["synthetic_sii"])


class ColorQualityReportTests(unittest.TestCase):
    def test_report_measures_color_change_without_becoming_a_gate(self) -> None:
        baseline = np.full((3, 48, 64), 0.04, dtype=np.float32)
        baseline[0, 12:36, 16:48] = 0.30
        baseline[1, 12:36, 16:48] = 0.18
        baseline[2, 12:36, 16:48] = 0.12
        candidate = baseline.copy()
        candidate[0, 12:36, 16:48] *= 1.08
        subject = np.zeros((48, 64), dtype=np.float32)
        subject[12:36, 16:48] = 1.0
        masks = {
            "core_mask": np.zeros_like(subject),
            "nebula_mask": subject,
            "faint_nebula_mask": np.zeros_like(subject),
            "background_mask": 1.0 - subject,
        }
        contract = resolve_color_contract(
            channel_profile={"kind": "broadband_rgb_osc"},
            color_report={},
        )

        report = build_color_quality_report(
            baseline,
            candidate,
            stage="stage8",
            baseline_name="stage8_input_starless.fit",
            candidate_name="stage8_enhanced.fit",
            contract=contract,
            masks=masks,
            requested_saturation=0.10,
            effective_saturation=0.08,
            applied_saturation=0.08,
            operation="masked_single_chroma_recovery",
        )

        self.assertEqual(report["status"], "reported")
        self.assertEqual(report["mode"], "report_only")
        self.assertFalse(report["used_for_gate"])
        self.assertGreater(
            report["rois"]["subject"]["delta"][
                "chromaticity_distance_p95"
            ],
            0.0,
        )
        self.assertEqual(report["profile_dependent_metrics"]["delta_e00"], None)
        self.assertEqual(report["ledger_entry"]["applied_saturation"], 0.08)

    def test_shape_mismatch_is_reported_unavailable(self) -> None:
        contract = resolve_color_contract(
            channel_profile={"kind": "broadband_rgb_osc"},
            color_report={},
        )
        report = build_color_quality_report(
            np.zeros((3, 8, 8), dtype=np.float32),
            np.zeros((3, 9, 8), dtype=np.float32),
            stage="stage8",
            baseline_name="before.fit",
            candidate_name="after.fit",
            contract=contract,
        )

        self.assertEqual(report["status"], "unavailable")
        self.assertFalse(report["used_for_gate"])


if __name__ == "__main__":
    unittest.main()
