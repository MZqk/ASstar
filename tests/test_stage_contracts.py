#!/usr/bin/env python3
"""Regression tests for stable product-stage and artifact contracts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import stage_contracts  # noqa: E402


class StageContractTests(unittest.TestCase):
    def test_product_stage_order_and_phase_boundary_are_stable(self) -> None:
        contracts = stage_contracts.product_stage_contracts()

        self.assertEqual([item.number for item in contracts], list(range(1, 11)))
        self.assertEqual(
            [item.phase.value for item in contracts],
            ["linear"] * 6 + ["nonlinear"] * 4,
        )
        self.assertEqual(contracts[5].key, "linear_star_separation")

    def test_only_stage_1_2_5_are_formal_resume_boundaries(self) -> None:
        self.assertEqual(stage_contracts.FORMAL_RESUME_STAGES, (1, 2, 5))
        self.assertEqual(
            [item.number for item in stage_contracts.formal_resume_contracts()],
            [1, 2, 5],
        )

    def test_stage_and_delivery_names_use_separate_namespaces(self) -> None:
        for contract in stage_contracts.product_stage_contracts():
            self.assertTrue(
                contract.primary_artifact.startswith(f"stage{contract.number}_")
            )
            self.assertFalse(contract.primary_artifact.startswith("result_"))
        for artifact in stage_contracts.RESULT_ARTIFACT_FAMILIES.values():
            self.assertTrue(artifact.startswith("result_"))

    def test_legacy_names_are_read_aliases_not_primary_artifacts(self) -> None:
        stage5 = stage_contracts.stage_contract(5)

        self.assertEqual(stage5.primary_artifact, "stage5_linear.fit")
        self.assertIn("result_linear.fit", stage5.legacy_read_aliases)
        self.assertNotIn(
            "result_linear.fit",
            [item.primary_artifact for item in stage_contracts.product_stage_contracts()],
        )

    def test_manifest_exposes_product_contract(self) -> None:
        manifest = stage_contracts.pipeline_contract_manifest()

        self.assertEqual(manifest["formal_resume_stages"], [1, 2, 5])
        self.assertEqual(manifest["linear_stages"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(manifest["nonlinear_stages"], [7, 8, 9, 10])
        self.assertFalse(manifest["stages"][-1]["product_stage"])


if __name__ == "__main__":
    unittest.main()
