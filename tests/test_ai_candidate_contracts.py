#!/usr/bin/env python3
"""Contract tests for code-owned AI candidate selection."""
from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy" not in sys.modules:
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")
    enums = types.ModuleType("sirilpy.enums")

    class _SirilError(Exception):
        pass

    class _SirilConnectionError(_SirilError):
        pass

    class _SirilInterface:
        pass

    class _CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    exceptions.CommandError = _SirilError
    exceptions.DataError = _SirilError
    exceptions.SirilError = _SirilError
    exceptions.SirilConnectionError = _SirilConnectionError
    sirilpy.SirilInterface = _SirilInterface
    enums.CommandStatus = _CommandStatus
    sirilpy.exceptions = exceptions
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums

import ai_advisory  # noqa: E402


class _Pipeline:
    def __init__(self, *, axiom_available: bool = False) -> None:
        self.cfg = SimpleNamespace(
            asinh_stretch=2.4,
            asinh_offset=0.0015,
            ghs_shadowsclip=-2.8,
            ghs_stretchamount=1.7,
            nebula_saturation=0.30,
            nebula_bg_factor=2,
            ai_strength=0.14,
        )
        self._axiom_available = axiom_available

    @staticmethod
    def _short_text(value: object, max_len: int = 180) -> str:
        return str(value)[:max_len]

    def _syqon_axiom_model_available(self) -> bool:
        return self._axiom_available


class AiCandidateContractTests(unittest.TestCase):
    def test_network_mode_is_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(ai_advisory.network_mode_enabled())
            with self.assertRaisesRegex(RuntimeError, "NETWORK_MODE"):
                ai_advisory.post_json_with_auth(
                    "https://example.invalid/v1/chat/completions",
                    {},
                    "secret",
                    1,
                )
        with patch.dict(os.environ, {"SEESTAR_NETWORK_MODE": "1"}, clear=True):
            self.assertTrue(ai_advisory.network_mode_enabled())

    def test_stage6_ignores_model_numeric_parameters(self) -> None:
        pipeline = _Pipeline()
        plan = ai_advisory.normalize_stage6_ai_plan(
            pipeline,
            {
                "stage6_stretch_selection": {
                    "selected_candidate_id": "asinh",
                    "params": {
                        "asinh_stretch": 99,
                        "ghs_stretchamount": 99,
                    },
                }
            },
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["selected_candidate_id"], "asinh")
        self.assertEqual(plan["params"]["asinh_stretch"], 2.4)
        self.assertEqual(plan["params"]["ghs_stretchamount"], 1.7)

    def test_stage7_stretch_rejects_id_outside_post_gate_allowlist(self) -> None:
        pipeline = _Pipeline()
        invalid = ai_advisory.normalize_stage7_stretch_selection(
            pipeline,
            {
                "stage7_stretch_selection": {
                    "selected_candidate_id": "invented_candidate",
                    "params": {"asinh_stretch": 99},
                }
            },
            ["cand_a", "cand_b"],
        )
        self.assertIsNone(invalid)
        valid = ai_advisory.normalize_stage7_stretch_selection(
            pipeline,
            {
                "stage7_stretch_selection": {
                    "selected_candidate_id": "cand_b",
                    "confidence": 0.8,
                }
            },
            ["cand_a", "cand_b"],
        )
        self.assertEqual(valid["selected_candidate_id"], "cand_b")

    def test_stage7_stretch_request_exposes_only_hard_gate_candidate_ids(self) -> None:
        pipeline = _Pipeline()
        captured: dict[str, object] = {}
        pipeline.log = SimpleNamespace(warn=lambda _message: None)
        pipeline._ai_stage_advisory_enabled = lambda _name: True
        pipeline._active_target_type = lambda: "galaxy"

        def request(_stage: str, _schema: str, observations: dict):
            captured.update(observations)
            return {
                "stage7_stretch_selection": {
                    "selected_candidate_id": "cand_b",
                }
            }

        pipeline._request_stage_ai_advisory = request
        selection = ai_advisory.request_stage7_stretch_selection(
            pipeline,
            [
                {"name": "cand_a", "method": "asinh", "risk_score": 0.2},
                {"name": "cand_b", "method": "asinh_ghs", "risk_score": 0.3},
                {
                    "name": "chroma_rescue_1",
                    "method": "background_chroma_rescue",
                    "risk_score": 0.1,
                    "explicit_fallback": True,
                },
            ],
        )

        self.assertEqual(selection["selected_candidate_id"], "cand_b")
        self.assertEqual(
            captured["constraints"]["allowed_candidate_ids"],
            ["cand_a", "cand_b"],
        )
        self.assertNotIn(
            "chroma_rescue_1",
            [
                item["candidate_id"]
                for item in captured["allowed_candidates"]
            ],
        )

    def test_starless_axiom_candidate_requires_local_capability(self) -> None:
        unavailable = ai_advisory.normalize_stage7_starless_plan(
            _Pipeline(axiom_available=False),
            {
                "stage7_starless_selection": {
                    "selected_candidate_id": "syqon_axiom_standard",
                    "tile_size": 4096,
                }
            },
        )
        self.assertIsNone(unavailable)

        available = ai_advisory.normalize_stage7_starless_plan(
            _Pipeline(axiom_available=True),
            {
                "stage7_starless_selection": {
                    "selected_candidate_id": "syqon_axiom_standard",
                    "tile_size": 4096,
                }
            },
        )
        self.assertEqual(available["tile_size"], 512)
        self.assertEqual(available["overlap"], 64)
        self.assertTrue(available["use_axiom"])

    def test_stage8_maps_id_to_immutable_local_preset(self) -> None:
        pipeline = _Pipeline()
        plan = ai_advisory.normalize_stage8_processing_plan(
            pipeline,
            {
                "stage8_processing_selection": {
                    "selected_candidate_id": "conservative",
                    "saturation": 0.65,
                    "bg_factor": 3,
                    "unsharp_amount": 0.60,
                }
            },
        )
        self.assertEqual(plan["saturation"], 0.08)
        self.assertEqual(plan["bg_factor"], 1)
        self.assertEqual(plan["unsharp_amount"], 0.18)

    def test_quality_advisories_map_ids_instead_of_numeric_values(self) -> None:
        pipeline = _Pipeline()
        stage7 = ai_advisory.normalize_stage7_ai_quality(
            pipeline,
            {
                "stage7_quality": {
                    "verdict": "poor",
                    "star_remix_candidate_id": "conservative",
                    "residual_suppression_candidate_id": "guarded",
                    "stage9_star_intensity_scale": 0.99,
                    "residual_suppression_strength": 0.24,
                }
            },
        )
        self.assertEqual(stage7["stage9_star_intensity_scale"], 0.50)
        self.assertEqual(stage7["residual_suppression_strength"], 0.16)

        stage8 = ai_advisory.normalize_stage8_ai_quality(
            pipeline,
            {
                "stage8_quality": {
                    "blue_bias": True,
                    "blue_guard_candidate_id": "strict",
                    "target_blue_excess": 0.16,
                }
            },
        )
        self.assertEqual(stage8["target_blue_excess"], 0.07)

    def test_stage11_ignores_adjustment_payload_and_rejects_unknown_id(self) -> None:
        pipeline = _Pipeline()
        adjustments = ai_advisory.normalize_ai_adjustments(
            pipeline,
            {
                "stage11_adjustment_selection": {
                    "selected_candidate_id": "conservative",
                    "adjustments": {
                        "global_saturation_delta": 0.99,
                        "detail_boost": 0.99,
                    },
                }
            },
        )
        self.assertEqual(adjustments["global_saturation_delta"], 0.015)
        self.assertEqual(adjustments["detail_boost"], 0.015)
        with self.assertRaises(ValueError):
            ai_advisory.normalize_ai_adjustments(
                pipeline,
                {"selected_candidate_id": "invented"},
            )


if __name__ == "__main__":
    unittest.main()
