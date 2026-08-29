"""Stage 7 presentation scoring and non-overridable safety-contract tests."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403
import stage8_color_rendition


class PipelinePluginFallbackStage7ContractTests(PipelinePluginFallbackTestBase):
    def test_stage7_bounded_score_stops_rewarding_oversaturation(self):
        processor = pipeline_module.StarunPostProcessor()

        def attempt(name: str, saturation: float, safety_load: float) -> dict[str, Any]:
            retention = {
                metric: {
                    "available": True,
                    "ratio": saturation if metric == "saturation_median" else 0.90,
                }
                for metric in (
                    "visibility",
                    "subject_span",
                    "saturation_median",
                    "microcontrast",
                )
            }
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "technical_safe": True,
                "adaptation": {"target_aware": {"name": "widefield_nebulosity"}},
                "rendition_metrics": {"retention": {"metrics": retention}},
                "background_quality_gate": {
                    "metrics": {
                        "chroma_load": safety_load,
                        "chroma_load_low_absolute_effective_max": 0.06,
                        "chroma_noise_score": safety_load * 5.0,
                        "background_mottling_score": safety_load * 6.0,
                    },
                    "limits": {
                        "chroma_load_signal_excluded_max": 0.06,
                        "chroma_noise_score_max": 0.34,
                        "background_mottling_score_max": 0.45,
                    },
                },
                "color_vector_gate": {
                    "metrics": {
                        "chromaticity_l1_half_p95": min(0.079, safety_load)
                    },
                    "limits": {"chromaticity_l1_half_p95_hard_max": 0.08},
                },
                "transform_loss_gate": {
                    "metrics": {"newly_hard_clipped_ratio": safety_load / 120.0},
                    "limits": {"newly_hard_clipped_ratio_max": 0.0005},
                },
                "diagnostics": [],
                "advisories": [],
                "risk_score": 1.0,
            }

        color_heavy = attempt("a_oversaturated", 3.0, 0.059)
        bounded = attempt("z_bounded", 0.90, 0.0)

        selected = min(
            [color_heavy, bounded],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "z_bounded")
        self.assertEqual(
            processor._stage7_presentation_score(color_heavy)["utilities"][
                "saturation_median"
            ],
            1.0,
        )

    def test_m8_composite_profile_uses_nebula_chroma_goal(self):
        processor = pipeline_module.StarunPostProcessor()
        attempt = {
            "adaptation": {
                "target_aware": {"name": "bright_core_composite_reveal"}
            },
            "rendition_metrics": {
                "candidate": {
                    "status": "available",
                    "metrics": {"saturation_median": 0.11},
                },
                "retention": {
                    "metrics": {
                        name: {
                            "available": True,
                            "ratio": 0.87 if name == "saturation_median" else 0.90,
                        }
                        for name in (
                            "visibility",
                            "subject_span",
                            "saturation_median",
                            "microcontrast",
                        )
                    }
                }
            },
        }

        report = processor._stage7_presentation_score_v6(attempt)

        self.assertEqual(report["profile"], "nebula")
        self.assertEqual(report["goals"]["saturation_median"], 0.90)
        self.assertLess(report["utilities"]["saturation_median"], 1.0)
        self.assertIsNone(report["absolute_subject_saturation_goal"])
        self.assertIsNone(report["absolute_subject_saturation_utility"])
        self.assertEqual(
            report["absolute_subject_saturation_role"],
            "diagnostic_only_stage8_owns_goal",
        )
        factor = stage8_color_rendition.target_aware_chroma_factor(
            "bright_core_composite_reveal",
            subject_saturation=0.11,
            effective_saturation_budget=0.40,
        )
        self.assertEqual(factor["raw_factor"], 4.0)
        self.assertEqual(factor["factor"], 4.0)

    def test_stage7_ranking_does_not_use_stage8_color_utility(self):
        processor = pipeline_module.StarunPostProcessor()

        def attempt(name, absolute_utility, score):
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "technical_safe": True,
                "presentation_score": {
                    "policy": "hard_gate_continuous_quality_v7",
                    "score": score,
                    "absolute_subject_saturation_utility": absolute_utility,
                },
            }

        pale = attempt("cand_b", 0.36, 0.95)
        color_rich = attempt("cand_display82", 0.74, 0.88)
        selected = min(
            [pale, color_rich],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "cand_b")

    def test_stage7_forced_delivery_rejects_any_technical_damage(self):
        processor = pipeline_module.StarunPostProcessor()
        appearance_only = {
            "name": "appearance_only",
            "stem": "stage7_appearance_only",
            "status": "ok",
            "pixel_stats": {
                "p50": 0.20,
                "p99": 0.90,
                "max": 0.95,
                "dynamic_range": 0.88,
            },
            "quality_gates": {"base_quality": {}},
            "target_local_quality": {
                "accepted": False,
                "quality_gates": {
                    "local_core_clip_ratio": {"hard_failed": False}
                },
            },
            "starless_structure_quality": {"accepted": True},
            "transform_loss_gate": {"accepted": True},
            "mtf_reference_quality": {"status": "not_applicable"},
            "display90_curve_quality": {"status": "not_applicable"},
            "diagnostics": ["local_faint_snr 0.20<0.40"],
        }
        self.assertTrue(
            processor._stage7_candidate_is_technically_safe(appearance_only)
        )

        clipped = copy.deepcopy(appearance_only)
        clipped["transform_loss_gate"] = {"accepted": False}
        self.assertFalse(processor._stage7_candidate_is_technically_safe(clipped))

        core_overflow = copy.deepcopy(appearance_only)
        core_overflow["target_local_quality"]["quality_gates"][
            "local_core_clip_ratio"
        ]["hard_failed"] = True
        self.assertFalse(
            processor._stage7_candidate_is_technically_safe(core_overflow)
        )

        for gate_name in (
            "local_core_colored_plateau_component_ratio",
            "local_core_parity_phase_span",
            "local_core_reference_available",
        ):
            with self.subTest(gate=gate_name):
                damaged = copy.deepcopy(appearance_only)
                damaged["target_local_quality"]["quality_gates"] = {
                    gate_name: {"hard_failed": True}
                }
                self.assertFalse(
                    processor._stage7_candidate_is_technically_safe(damaged)
                )

        corrupt = copy.deepcopy(appearance_only)
        corrupt["starless_structure_quality"] = {"accepted": False}
        self.assertFalse(processor._stage7_candidate_is_technically_safe(corrupt))

        unverified_mtf = copy.deepcopy(appearance_only)
        unverified_mtf["method"] = "linked_mtf"
        unverified_mtf["mtf_reference_quality"] = {
            "status": "unavailable",
            "accepted": False,
        }
        self.assertFalse(
            processor._stage7_candidate_is_technically_safe(unverified_mtf)
        )
