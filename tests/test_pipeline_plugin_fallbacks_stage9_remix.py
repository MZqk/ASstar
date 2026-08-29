"""Pipeline/plugin fallback tests for stage9 remix."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage9RemixTests(PipelinePluginFallbackTestBase):
    def test_stage9_legacy_accepted_hdr_state_cannot_bypass_review_contract(self):
        processor = self._new_processor()
        processor._star_separation_state = (
            pipeline_module.StarSeparationState.REJECTED.value
        )
        processor._stage8_final_source = "stage8_enhanced"
        processor._stage8_final_quality = "bright_core_with_stars_hdr"
        processor._stage8_handoff.update(
            source_stem="stage8_enhanced",
            passthrough=True,
            restricted_downstream=False,
            reason_code="bright_core_with_stars_hdr_passthrough",
        )
        processor._bright_core_with_stars_fallback = {
            "eligible": True,
            "accepted": True,
            "status": "accepted",
            "output_stem": "stage7_with_stars_hdr",
        }

        stage9_star_remixing(processor)

        self.assertTrue(processor._stage9_stars_required)
        self.assertFalse(processor._stage9_stars_applied)
        self.assertFalse(processor._stage9_output_contains_stars)
        self.assertFalse(processor._stage9_star_delivery_contract_accepted)
        self.assertEqual(processor._stage9_final_source, "")
        self.assertEqual(
            processor._stage9_stars_application_mode,
            "withheld_no_with_stars_review_source",
        )
        self.assertFalse(any(call[0] == "pm" for call in processor.cmd_calls))
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertFalse(report["star_delivery_contract_accepted"])
        self.assertTrue(report["stars_required"])
        self.assertFalse(report["output_contains_stars"])
        self.assertTrue(report["output_withheld"])
        self.assertEqual(processor.results[-1][1], "failed")

    def test_stage9_uses_previous_stage_starless_for_star_remix(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        stage9_star_remixing(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [("stage8_enhanced", "starmask_stretched", processor.cfg.star_intensity)],
        )
        self.assertIn("previous_stage_star_remix source=stage8_enhanced", message)
        self.assertIn("starmask=starmask_stretched", message)
        self.assertIn(
            ("asinh", "2.000", "0.00100", "-clipmode=rgbblend"),
            processor.cmd_calls,
        )
        self.assertIn(("save", "starmask_stretched"), processor.cmd_calls)
        pm_calls = [call for call in processor.cmd_calls if call[0] == "pm"]
        self.assertFalse(pm_calls)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        json.dumps(report)
        self.assertEqual(report["schema"], "starun.stage9-remix-quality.v10")
        self.assertEqual(
            report["selection_policy"],
            "sep_catalog_visibility_psf_fidelity_recovery_v8",
        )
        self.assertTrue(report["sep_crossmatch"]["accepted"])
        self.assertEqual(report["selection_class"], "formal")
        self.assertTrue(report["formal_accepted"])
        self.assertFalse(report["review_candidate_selected"])
        self.assertEqual(
            report["star_layer_decomposition"]["final_composition"],
            "screen",
        )
        self.assertEqual(
            report["objective"]["reconstruction_target"],
            "faithful_same_source_star_display_restoration",
        )
        self.assertFalse(report["objective"]["scientific_photometry_claim"])
        self.assertEqual(
            report["operator_contract"]["alpha_semantics"],
            "binary_compact_spatial_support",
        )
        self.assertNotIn("metric_compatibility", report)
        self.assertEqual(
            report["zero_edit_operator_audit"]["schema"],
            "starun.stage9-operator-audit.v1",
        )
        self.assertTrue(report["stars_required"])
        self.assertTrue(report["stars_applied"])
        self.assertEqual(report["stars_application_mode"], "screen")

    def test_stage9_partial_psf_evidence_without_triplet_fails_closed(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: {
                "attempt": attempt,
                "formula": formula,
                "status": "partial",
                "accepted": True,
                "review_required": True,
                "issues": [],
                "advisories": ["PSF groups not assessed: bright"],
                "reason_codes": [
                    "STAGE9_PSF_SUBGROUP_EVIDENCE_INSUFFICIENT"
                ],
                "metrics": {},
                "psf_closure": {
                    "schema": "starun.stage9-psf-closure.v3",
                    "status": "partial",
                    "accepted": True,
                    "review_required": True,
                    "groups": {
                        "bright": {
                            "status": "not_assessed",
                            "reference_sample_count": 3,
                            "minimum_sample_count": 4,
                        }
                    },
                },
            }
        )

        stage9_star_remixing(processor)

        self.assertEqual(processor.results[-1][1], "failed")
        self.assertFalse(processor._stage9_stars_applied)
        self.assertTrue(processor._stage9_output_withheld)
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "required_stars_output_withheld",
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertIsNone(report["selected"])
        self.assertEqual(
            report["attempts"][-1]["psf_soft_target_closure"]["status"],
            "rolled_back_unclosed",
        )
        self.assertEqual(
            report["attempts"][-1]["psf_soft_target_closure"][
                "final_missing_formal_groups"
            ],
            ["all", "weak", "bright"],
        )

    def test_stage9_selects_unscreen_only_after_material_non_regressing_gain(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        baseline_stars = np.full((3, 4, 4), 0.20, dtype=np.float32)
        unscreen_stars = np.full((3, 4, 4), 0.35, dtype=np.float32)
        state = {
            "star_layer": baseline_stars,
            "alpha_mask": np.ones((4, 4), dtype=np.float32),
            "weak_mask": np.ones((4, 4), dtype=bool),
            "bright_mask": np.zeros((4, 4), dtype=bool),
            "star_color_validation": None,
        }

        def assess(_source_stem, *, attempt, formula):
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok",
                "accepted": True,
                "issues": [],
                "advisories": [],
                "star_color_validation": {
                    "metrics": {"median_chroma_error": 0.10}
                },
                "metrics": {
                    "weak_star_recovery_ratio": 0.90,
                    "star_recovery_ratio": 0.90,
                    "star_positive_delta_window_recovery_ratio": 0.90,
                    "star_wing_recovery_ratio": 0.90,
                },
            }

        processor._stage9_assess_current_remix = assess
        context = {
            "available": True,
            "report": {
                "schema": "starun.stage9-unscreen-reference.v1",
                "status": "ready",
                "available": True,
                "reason_code": "stage9_unscreen_reference_ready",
            },
            "original_display": np.zeros((3, 4, 4), dtype=np.float32),
            "starless_display": np.zeros((3, 4, 4), dtype=np.float32),
            "trusted_stars": baseline_stars,
            "unscreen_stars": unscreen_stars,
            "support_mask": np.ones((4, 4), dtype=bool),
        }

        def fidelity(_pipeline, _context, stars, _intensity, _state):
            return {
                "status": "ok",
                "support_rgb_mae": (
                    0.040 if stars is unscreen_stars else 0.050
                ),
            }

        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                return_value=context,
            ),
            patch.object(
                stage9_module,
                "_stage9_extend_rescue_with_source_presence",
                side_effect=lambda _pipeline, **kwargs: (
                    kwargs["accepted_quality"],
                    kwargs["accepted_context"],
                ),
            ),
            patch.object(
                stage9_module,
                "_capture_stage9_candidate_state",
                return_value=state,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                side_effect=fidelity,
            ),
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_unscreen_primary")
        self.assertEqual(
            report["star_layer_decomposition"]["selected"],
            "matched_mtf_unscreen_chroma_stabilized",
        )
        self.assertEqual(
            report["unscreen_reference"]["reason_code"],
            "stage9_unscreen_selected",
        )
        self.assertFalse(report["stage9_fallback_used"])
        self.assertEqual(
            processor.previous_stage_remix_calls[-1][1],
            "starmask_unscreen_stabilized",
        )

    def test_stage9_unscreen_recovers_small_psf_from_source_wing_support(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        baseline_stars = np.full((3, 4, 4), 0.20, dtype=np.float32)
        initial_unscreen_stars = np.full((3, 4, 4), 0.28, dtype=np.float32)
        one_pixel_unscreen_stars = np.full((3, 4, 4), 0.32, dtype=np.float32)
        two_pixel_unscreen_stars = np.full((3, 4, 4), 0.35, dtype=np.float32)
        state = {
            "star_layer": baseline_stars,
            "alpha_mask": np.ones((4, 4), dtype=np.float32),
            "weak_mask": np.ones((4, 4), dtype=bool),
            "bright_mask": np.zeros((4, 4), dtype=bool),
            "star_color_validation": None,
        }

        def quality(attempt: str, *, accepted: bool, fwhm: float) -> dict[str, Any]:
            return {
                "attempt": attempt,
                "formula": "screen",
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else [
                    f"star_psf_fwhm_ratio_all {fwhm:.3f} outside 0.930..1.100"
                ],
                "advisories": [],
                "star_color_validation": {
                    "metrics": {"median_chroma_error": 0.10}
                },
                "metrics": {
                    "weak_star_recovery_ratio": 0.90,
                    "star_recovery_ratio": 0.90,
                    "star_positive_delta_window_recovery_ratio": 0.90,
                    "star_wing_recovery_ratio": 0.90,
                    "star_psf_fwhm_ratio_all": fwhm,
                },
                "psf_closure": {
                    "status": "ok" if accepted else "rejected",
                    "limits": {
                        "stage9_psf_fwhm_ratio_min": 0.93,
                        "stage9_psf_fwhm_ratio_max": 1.10,
                    },
                    "groups": {
                        "all": {
                            "status": "ok",
                            "fwhm_ratio_median": fwhm,
                        }
                    },
                },
            }

        def assess(_source_stem, *, attempt, formula):
            if attempt == "screen_unscreen_primary":
                return quality(attempt, accepted=False, fwhm=0.80)
            if attempt == "screen_unscreen_psf_support_recovery_1px":
                return quality(attempt, accepted=True, fwhm=0.95)
            if attempt == "screen_unscreen_psf_support_recovery_2px":
                return quality(attempt, accepted=True, fwhm=1.00)
            return quality(attempt, accepted=True, fwhm=1.00)

        processor._stage9_assess_current_remix = assess
        initial_context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": initial_unscreen_stars,
        }
        one_pixel_context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": one_pixel_unscreen_stars,
        }
        two_pixel_context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": two_pixel_unscreen_stars,
        }
        original_prepare = stage9_module._prepare_stage9_starmask_for_pixel_remix

        def prepare_with_psf_retry(*args, **kwargs):
            output_name = str(kwargs.get("output_name") or "")
            if output_name.startswith("starmask_stretched_unscreen_psf_recovery_"):
                return output_name
            return original_prepare(*args, **kwargs)

        def fidelity(_pipeline, _context, stars, _intensity, _state):
            if stars is baseline_stars:
                mae = 0.050
            elif stars is two_pixel_unscreen_stars:
                mae = 0.040
            elif stars is one_pixel_unscreen_stars:
                mae = 0.045
            else:
                mae = 0.048
            return {"status": "ok", "support_rgb_mae": mae}

        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                side_effect=[
                    initial_context,
                    one_pixel_context,
                    two_pixel_context,
                ],
            ),
            patch.object(
                stage9_module,
                "_prepare_stage9_starmask_for_pixel_remix",
                side_effect=prepare_with_psf_retry,
            ),
            patch.object(
                stage9_module,
                "_capture_stage9_candidate_state",
                return_value=state,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                side_effect=fidelity,
            ),
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(
            report["selected"]["attempt"],
            "screen_unscreen_psf_support_recovery_2px",
        )
        self.assertFalse(report["stage9_fallback_used"])
        recovery_attempt = next(
            attempt
            for attempt in report["attempts"]
            if attempt.get("attempt")
            == "screen_unscreen_psf_support_recovery_2px"
        )
        self.assertEqual(recovery_attempt["psf_support_recovery_pixels"], 2)
        self.assertTrue(
            any(
                attempt.get("attempt")
                == "screen_unscreen_psf_support_recovery_1px"
                for attempt in report["attempts"]
            )
        )
        self.assertTrue(
            any(
                attempt.get("attempt") == "screen_unscreen_primary"
                for attempt in report["attempts"]
            )
        )

    def test_stage9_psf_soft_targets_cannot_expand_hard_gate(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.stage9_psf_fwhm_ratio_min = 0.93
        processor.cfg.stage9_psf_fwhm_ratio_max = 1.10
        processor.cfg.stage9_psf_recovery_target_min = 0.50
        processor.cfg.stage9_psf_recovery_target_max = 1.50
        quality = {
            "psf_closure": {
                "limits": {
                    "stage9_psf_fwhm_ratio_min": 0.94,
                    "stage9_psf_fwhm_ratio_max": 1.08,
                }
            }
        }

        self.assertEqual(
            stage9_module._stage9_psf_recovery_target_min(processor, quality),
            0.97,
        )
        self.assertEqual(
            stage9_module._stage9_psf_recovery_target_max(processor, quality),
            1.05,
        )

    def test_stage9_screen_psf_recovery_uses_one_pixel_before_two(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_unscreen_candidate_enabled = False
        processor.cfg.stage9_psf_support_retry_pixels = 2
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        def quality(attempt: str, fwhm: float) -> dict[str, Any]:
            return {
                "attempt": attempt,
                "formula": "screen",
                "status": "ok",
                "accepted": True,
                "issues": [],
                "advisories": [],
                "metrics": {
                    "star_psf_fwhm_ratio_all": fwhm,
                    "highlight_clip_growth": 0.001,
                    "bright_pixel_growth": 0.002,
                },
                "psf_closure": {
                    "status": "ok",
                    "limits": {
                        "stage9_psf_fwhm_ratio_min": 0.93,
                        "stage9_psf_fwhm_ratio_max": 1.10,
                    },
                    "groups": {
                        "all": {
                            "status": "ok",
                            "fwhm_ratio_median": fwhm,
                        }
                    },
                },
            }

        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: quality(
                attempt,
                0.99 if attempt.endswith("_1px") else 0.95,
            )
        )
        original_prepare = stage9_module._prepare_stage9_starmask_for_pixel_remix
        recovery_outputs: list[str] = []

        def prepare_with_progressive_retry(*args, **kwargs):
            output_name = str(kwargs.get("output_name") or "")
            if output_name.startswith("starmask_stretched_psf_recovery_"):
                recovery_outputs.append(output_name)
                return output_name
            return original_prepare(*args, **kwargs)

        with patch.object(
            stage9_module,
            "_prepare_stage9_starmask_for_pixel_remix",
            side_effect=prepare_with_progressive_retry,
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(
            report["selected"]["attempt"],
            "screen_psf_support_recovery_1px",
        )
        self.assertEqual(
            recovery_outputs,
            ["starmask_stretched_psf_recovery_1px"],
        )
        self.assertEqual(
            report["selected"]["psf_support_recovery_pixels"],
            1,
        )
        self.assertEqual(
            report["psf_size_recovery_policy"]["candidate_pixels"],
            [0, 1, 2],
        )
        self.assertEqual(
            report["psf_size_recovery_policy"]["soft_recovery_target_min"],
            0.97,
        )
        self.assertEqual(
            report["psf_size_recovery_policy"]["soft_recovery_target_max"],
            1.05,
        )
        self.assertFalse(
            report["psf_size_recovery_policy"]["synthetic_or_recursive_dilation"]
        )

    def test_stage9_screen_psf_recovery_rolls_back_oversized_candidate(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_unscreen_candidate_enabled = False
        processor.cfg.stage9_psf_support_retry_pixels = 2
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        def assess(_source, *, attempt, formula):
            fwhm = 1.08 if attempt.endswith("_1px") else 0.95
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok",
                "accepted": True,
                "issues": [],
                "advisories": [],
                "metrics": {
                    "star_psf_fwhm_ratio_all": fwhm,
                    "highlight_clip_growth": 0.001,
                    "bright_pixel_growth": 0.002,
                },
                "psf_closure": {
                    "status": "ok",
                    "limits": {
                        "stage9_psf_fwhm_ratio_min": 0.93,
                        "stage9_psf_fwhm_ratio_max": 1.10,
                    },
                    "groups": {
                        "all": {
                            "status": "ok",
                            "fwhm_ratio_median": fwhm,
                        }
                    },
                },
            }

        processor._stage9_assess_current_remix = assess
        original_prepare = stage9_module._prepare_stage9_starmask_for_pixel_remix

        def prepare_with_progressive_retry(*args, **kwargs):
            output_name = str(kwargs.get("output_name") or "")
            if output_name.startswith("starmask_stretched_psf_recovery_"):
                return output_name
            return original_prepare(*args, **kwargs)

        with patch.object(
            stage9_module,
            "_prepare_stage9_starmask_for_pixel_remix",
            side_effect=prepare_with_progressive_retry,
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertIsNone(report["selected"])
        primary = next(
            attempt
            for attempt in report["attempts"]
            if attempt.get("attempt") == "screen_primary"
        )
        self.assertEqual(
            primary["psf_soft_target_closure"]["status"],
            "rolled_back_unclosed",
        )
        self.assertFalse(primary["psf_soft_target_closure"]["accepted"])
        self.assertFalse(
            any(
                attempt.get("attempt") == "screen_psf_support_recovery_2px"
                for attempt in report["attempts"]
            )
        )

    def test_stage9_psf_support_recovery_uses_soft_target_without_changing_hard_gate(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        rejected = {
            "psf_closure": {
                "status": "rejected",
                "limits": {
                    "stage9_psf_fwhm_ratio_min": 0.93,
                    "stage9_psf_fwhm_ratio_max": 1.10,
                },
                "groups": {
                    "all": {"status": "ok", "fwhm_ratio_median": 0.90}
                },
            }
        }

        self.assertEqual(stage9_module._stage9_psf_size_direction(rejected), "small")
        accepted = copy.deepcopy(rejected)
        accepted["psf_closure"]["status"] = "ok"
        accepted["psf_closure"]["groups"]["all"]["fwhm_ratio_median"] = 0.95
        self.assertIsNone(stage9_module._stage9_psf_size_direction(accepted))
        self.assertTrue(
            stage9_module._stage9_needs_progressive_psf_recovery(
                processor,
                accepted,
            )
        )
        accepted["psf_closure"]["groups"]["all"]["fwhm_ratio_median"] = 0.98
        self.assertFalse(
            stage9_module._stage9_needs_progressive_psf_recovery(
                processor,
                accepted,
            )
        )
        accepted["psf_closure"]["groups"] = {
            "weak": {"status": "ok", "fwhm_ratio_median": 1.06},
            "bright": {"status": "ok", "fwhm_ratio_median": 0.95},
        }
        self.assertFalse(
            stage9_module._stage9_needs_progressive_psf_recovery(
                processor,
                accepted,
            )
        )
        mixed_large_rejection = {
            "accepted": False,
            "issues": [
                "star_psf_fwhm_ratio_all 1.206045 outside 0.930000..1.100000",
                "star_psf_fwhm_ratio_weak 1.239239 outside 0.930000..1.100000",
            ],
            "psf_closure": {
                "status": "rejected",
                "limits": {
                    "stage9_psf_fwhm_ratio_min": 0.93,
                    "stage9_psf_fwhm_ratio_max": 1.10,
                },
                "groups": {
                    "all": {"status": "ok", "fwhm_ratio_median": 1.206045},
                    "weak": {"status": "ok", "fwhm_ratio_median": 1.239239},
                    "bright": {"status": "ok", "fwhm_ratio_median": 0.967204},
                },
            },
        }
        self.assertFalse(
            stage9_module._stage9_needs_progressive_psf_recovery(
                processor,
                mixed_large_rejection,
            )
        )
        processor.cfg.stage9_psf_size_gate_enabled = False
        accepted["psf_closure"]["groups"] = {
            "all": {"status": "ok", "fwhm_ratio_median": 0.90}
        }
        self.assertFalse(
            stage9_module._stage9_needs_progressive_psf_recovery(
                processor,
                accepted,
            )
        )

    def test_stage9_restores_baseline_when_unscreen_gain_is_insufficient(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        baseline_stars = np.full((3, 4, 4), 0.20, dtype=np.float32)
        unscreen_stars = np.full((3, 4, 4), 0.25, dtype=np.float32)
        state = {
            "star_layer": baseline_stars,
            "alpha_mask": np.ones((4, 4), dtype=np.float32),
            "weak_mask": np.ones((4, 4), dtype=bool),
            "bright_mask": np.zeros((4, 4), dtype=bool),
            "star_color_validation": None,
        }
        processor._stage9_assess_current_remix = lambda _source, *, attempt, formula: {
            "attempt": attempt,
            "formula": formula,
            "status": "ok",
            "accepted": True,
            "issues": [],
            "advisories": [],
            "star_color_validation": {
                "metrics": {"median_chroma_error": 0.10}
            },
            "metrics": {
                "weak_star_recovery_ratio": 0.90,
                "star_recovery_ratio": 0.90,
                "star_positive_delta_window_recovery_ratio": 0.90,
                "star_wing_recovery_ratio": 0.90,
            },
        }
        context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": unscreen_stars,
        }
        fidelity_values = iter((0.050, 0.047))
        restore_calls: list[str] = []

        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                return_value=context,
            ),
            patch.object(
                stage9_module,
                "_capture_stage9_candidate_state",
                return_value=state,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                side_effect=lambda *_args: {
                    "status": "ok",
                    "support_rgb_mae": next(fidelity_values),
                },
            ),
            patch.object(
                stage9_module,
                "_restore_stage9_candidate_state",
                side_effect=lambda _pipeline, _state, *, checkpoint_stem: (
                    restore_calls.append(checkpoint_stem)
                ),
            ),
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_primary")
        self.assertEqual(
            report["unscreen_reference"]["reason_code"],
            "stage9_unscreen_no_material_improvement",
        )
        self.assertEqual(restore_calls, ["stage9_candidate_subtraction_screen"])
        self.assertEqual(
            report["star_layer_decomposition"]["selected"],
            "linear_original_minus_starless_stretched",
        )

    def test_stage9_unscreen_can_rescue_after_all_baselines_are_rejected(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        unscreen_stars = np.full((3, 4, 4), 0.30, dtype=np.float32)

        def assess(_source_stem, *, attempt, formula):
            accepted = attempt == "screen_unscreen_primary"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else ["background_lift 0.02>0.01"],
                "advisories": [],
                "metrics": {
                    "weak_star_recovery_ratio": 0.90,
                    "star_recovery_ratio": 0.90,
                    "star_positive_delta_window_recovery_ratio": 0.90,
                    "star_wing_recovery_ratio": 0.90,
                },
            }

        processor._stage9_assess_current_remix = assess
        context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": unscreen_stars,
        }
        state = {
            "star_layer": np.full((3, 4, 4), 0.10, dtype=np.float32),
            "alpha_mask": np.ones((4, 4), dtype=np.float32),
            "weak_mask": np.ones((4, 4), dtype=bool),
            "bright_mask": np.zeros((4, 4), dtype=bool),
            "star_color_validation": None,
        }

        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                return_value=context,
            ),
            patch.object(
                stage9_module,
                "_capture_stage9_candidate_state",
                return_value=state,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                return_value={"status": "ok", "support_rgb_mae": 0.01},
            ),
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_unscreen_primary")
        self.assertTrue(report["unscreen_reference"]["rescue_without_baseline"])
        self.assertFalse(report["stage9_fallback_used"])
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage9_unscreen_rescue_runs_source_presence_extension(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        unscreen_stars = np.full((3, 4, 4), 0.30, dtype=np.float32)
        context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": unscreen_stars,
        }
        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: {
                "attempt": attempt,
                "formula": formula,
                "status": (
                    "ok" if attempt == "screen_unscreen_primary" else "rejected"
                ),
                "accepted": attempt == "screen_unscreen_primary",
                "issues": (
                    []
                    if attempt == "screen_unscreen_primary"
                    else ["background_lift 0.02>0.01"]
                ),
                "advisories": [],
                "metrics": {},
            }
        )
        extended_quality = {
            "attempt": "screen_unscreen_source_presence",
            "formula": "screen",
            "status": "ok",
            "accepted": True,
            "issues": [],
            "metrics": {},
            "decomposition_method": "matched_mtf_unscreen_source_presence",
            "psf_closure": {
                "status": "ok",
                "accepted": True,
                "limits": {
                    "stage9_psf_fwhm_ratio_min": 0.93,
                    "stage9_psf_fwhm_ratio_max": 1.10,
                },
                "groups": {
                    group: {
                        "status": "ok",
                        "accepted": True,
                        "fwhm_ratio_median": 1.0,
                    }
                    for group in ("all", "weak", "bright")
                },
            },
        }
        calls = []

        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                return_value=context,
            ),
            patch.object(
                stage9_module,
                "_stage9_extend_rescue_with_source_presence",
                side_effect=lambda _pipeline, **kwargs: (
                    calls.append(kwargs["accepted_quality"]["attempt"])
                    or (extended_quality, kwargs["accepted_context"])
                ),
            ),
            patch.object(
                stage9_module,
                "_capture_stage9_candidate_state",
                return_value={
                    "star_layer": np.zeros((3, 4, 4), dtype=np.float32),
                    "alpha_mask": np.ones((4, 4), dtype=np.float32),
                    "weak_mask": np.ones((4, 4), dtype=bool),
                    "bright_mask": np.zeros((4, 4), dtype=bool),
                    "star_color_validation": None,
                },
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                return_value={"status": "ok", "support_rgb_mae": 0.01},
            ),
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(calls, ["screen_unscreen_primary"])
        self.assertEqual(
            report["selected"]["attempt"],
            "screen_unscreen_source_presence",
        )
        self.assertEqual(
            report["star_layer_decomposition"]["selected"],
            "matched_mtf_unscreen_source_presence",
        )

    def test_stage9_source_presence_retries_lower_feather_after_psf_overshoot(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor._stage9_source_presence_report = {}
        base_stars = np.full((3, 4, 4), 0.20, dtype=np.float32)
        accepted_context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": base_stars,
            "support_mask": np.ones((4, 4), dtype=bool),
            "weak_mask": np.ones((4, 4), dtype=bool),
            "bright_mask": np.zeros((4, 4), dtype=bool),
            "original_display": np.full((3, 4, 4), 0.30, dtype=np.float32),
            "starless_display": np.full((3, 4, 4), 0.10, dtype=np.float32),
        }
        accepted_quality = {
            "attempt": "screen_unscreen_primary",
            "status": "ok",
            "accepted": True,
            "issues": [],
            "metrics": {},
            "support_mode": "normal",
            "support_starmask": "starmask_stretched",
            "base_source_stem": "stage8_enhanced",
        }
        attempted_strengths = []
        qualities = iter(
            (
                {
                    "status": "rejected",
                    "accepted": False,
                    "issues": [
                        "star_psf_fwhm_ratio_weak 1.102 outside 0.930..1.100"
                    ],
                    "metrics": {"star_psf_fwhm_ratio_weak": 1.102},
                },
                {
                    "status": "ok",
                    "accepted": True,
                    "issues": [],
                    "metrics": {"star_psf_fwhm_ratio_weak": 1.096},
                },
            )
        )

        def prepare(
            _pipeline,
            context,
            _messages,
            *,
            feather_strength=0.90,
            screen_intensity=1.0,
        ):
            attempted_strengths.append(feather_strength)
            self.assertEqual(screen_intensity, 1.0)
            stars = np.full_like(base_stars, feather_strength)
            return {
                **context,
                "unscreen_stars": stars,
                "source_presence_report": {
                    "status": "ready",
                    "available": True,
                    "changed": True,
                    "source_wing_feather": {
                        "status": "ready",
                        "feather_strength": feather_strength,
                    },
                    "stage5_bright_star_completion": {"status": "ready"},
                },
            }

        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: {
                **next(qualities),
                "attempt": attempt,
                "formula": formula,
            }
        )
        remix_attempts = []
        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_source_presence_candidate",
                side_effect=prepare,
            ),
            patch.object(
                stage9_module,
                "_save_stage9_unscreen_context_layer",
                return_value=True,
            ),
            patch.object(
                stage9_module,
                "_stage9_observe_bright_star_presence",
                return_value={"status": "observed"},
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                return_value={"status": "ok"},
            ),
        ):
            selected, selected_context = (
                stage9_module._stage9_extend_rescue_with_source_presence(
                    processor,
                    source_stem="stage8_enhanced",
                    accepted_context=accepted_context,
                    accepted_quality=accepted_quality,
                    intensity=1.0,
                    messages=[],
                    remix_attempts=remix_attempts,
                )
            )

        self.assertEqual(attempted_strengths, [0.95, 0.90])
        self.assertEqual(selected["attempt"], "screen_unscreen_source_presence_90")
        self.assertEqual(selected["source_wing_feather_strength"], 0.90)
        self.assertEqual(selected["support_mode"], "normal")
        self.assertEqual(selected["support_starmask"], "starmask_stretched")
        self.assertIs(selected_context["unscreen_stars"].dtype, base_stars.dtype)
        self.assertEqual(len(remix_attempts), 2)
        self.assertTrue(remix_attempts[-1]["accepted"])

    def test_stage9_selective_size_retries_without_relaxing_psf_gate(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.stage9_psf_selective_wing_enabled = True
        processor.cfg.stage9_psf_selective_wing_target_ratio = 1.08
        processor.cfg.stage9_psf_selective_wing_strength_max = 1.15
        processor.cfg.stage9_psf_support_retry_pixels = 2
        processor._stage9_source_presence_report = {
            "status": "ready",
            "available": True,
        }
        base_stars = np.full((3, 4, 4), 0.20, dtype=np.float32)
        accepted_context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "unscreen_stars": base_stars,
            "support_mask": np.ones((4, 4), dtype=bool),
            "weak_mask": np.ones((4, 4), dtype=bool),
            "bright_mask": np.zeros((4, 4), dtype=bool),
            "original_display": np.full((3, 4, 4), 0.30, dtype=np.float32),
            "starless_display": np.full((3, 4, 4), 0.10, dtype=np.float32),
            "source_presence_report": {"status": "ready"},
        }
        accepted_quality = {
            "attempt": "screen_unscreen_source_presence_90",
            "status": "ok",
            "accepted": True,
            "issues": [],
            "metrics": {},
        }
        processor.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: np.full(
                (3, 4, 4), 0.25, dtype=np.float32
            )
        )
        attempted = []
        qualities = iter(
            (
                {
                    "status": "rejected",
                    "accepted": False,
                    "issues": [
                        "star_psf_fwhm_ratio_weak 1.105 outside 0.930..1.100"
                    ],
                    "metrics": {"star_psf_fwhm_ratio_weak": 1.105},
                    "psf_closure": {
                        "status": "rejected",
                        "limits": {
                            "stage9_psf_fwhm_ratio_min": 0.93,
                            "stage9_psf_fwhm_ratio_max": 1.10,
                        },
                        "groups": {
                            "weak": {
                                "status": "ok",
                                "fwhm_ratio_median": 1.105,
                            }
                        },
                    },
                },
                {
                    "status": "ok",
                    "accepted": True,
                    "issues": [],
                    "metrics": {"star_psf_fwhm_ratio_weak": 1.098},
                    "psf_closure": {
                        "status": "ok",
                        "limits": {
                            "stage9_psf_fwhm_ratio_min": 0.93,
                            "stage9_psf_fwhm_ratio_max": 1.10,
                        },
                        "groups": {
                            "weak": {
                                "status": "ok",
                                "fwhm_ratio_median": 1.098,
                            }
                        },
                    },
                },
            )
        )

        def prepare(
            _pipeline,
            context,
            _candidate_display,
            _messages,
            *,
            screen_intensity,
            fwhm_ratio_target,
            feather_strength,
            support_extra_pixels,
        ):
            attempted.append((support_extra_pixels, feather_strength))
            self.assertEqual(screen_intensity, 1.0)
            self.assertEqual(fwhm_ratio_target, 1.08)
            return {
                **context,
                "unscreen_stars": np.full_like(base_stars, feather_strength),
                "selective_source_wing_report": {
                    "schema": "starun.stage9-selective-source-wing.v1",
                    "status": "ready",
                    "available": True,
                    "changed": True,
                    "selected_star_count": 8,
                    "selected_star_ratio": 0.25,
                },
            }

        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: {
                **next(qualities),
                "attempt": attempt,
                "formula": formula,
            }
        )
        remix_attempts = []
        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_selective_size_candidate",
                side_effect=prepare,
            ),
            patch.object(
                stage9_module,
                "_save_stage9_unscreen_context_layer",
                return_value=True,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                return_value={"status": "ok"},
            ),
        ):
            selected, selected_context = (
                stage9_module._stage9_extend_with_selective_size(
                    processor,
                    source_stem="stage8_enhanced",
                    accepted_context=accepted_context,
                    accepted_quality=accepted_quality,
                    intensity=1.0,
                    messages=[],
                    remix_attempts=remix_attempts,
                )
            )

        self.assertEqual(attempted, [(2, 1.15), (2, 1.10)])
        self.assertEqual(
            selected["attempt"],
            "screen_unscreen_selective_size_2px_110",
        )
        self.assertTrue(selected["accepted"])
        self.assertEqual(selected["selective_source_wing_extra_pixels"], 2)
        self.assertEqual(selected["selective_source_wing_strength"], 1.10)
        self.assertIs(selected_context["unscreen_stars"].dtype, base_stars.dtype)
        self.assertEqual(len(remix_attempts), 2)
        self.assertEqual(
            processor._stage9_source_presence_report[
                "selective_size_rescue"
            ]["candidate_comparison"][-1]["fwhm_ratios"]["weak"],
            1.098,
        )

    def test_stage9_reference_rejection_uses_strict_low_intensity_candidate(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["星点拉伸"] = "SASP Star Stretch"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"mock")

        yy, xx = np.mgrid[:128, :128]
        starmask = np.zeros((3, 128, 128), dtype=np.float32)
        for cy, cx, amplitude in (
            (20, 22, 0.05),
            (35, 92, 0.08),
            (62, 58, 0.10),
            (88, 28, 0.06),
            (101, 98, 0.14),
            (72, 108, 0.09),
        ):
            profile = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.4**2)
            ).astype(np.float32)
            starmask += np.stack(
                [
                    profile * amplitude,
                    profile * amplitude * 0.88,
                    profile * amplitude * 0.74,
                ]
            )
        source = np.clip(starmask + 0.02, 0.0, 1.0)
        processor.saved_image_pixels["starmask"] = starmask.copy()
        processor.saved_image_pixels["stage5_linear"] = source.copy()
        processor.image_pixels = starmask.copy()
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: starmask.shape,
            get_image_pixeldata=lambda preview=False: processor.image_pixels.copy(),
            set_image_pixeldata=lambda output: setattr(
                processor,
                "image_pixels",
                np.array(output, copy=True),
            ),
        )

        stage9_module = sys.modules["stages.stage9_star_remixing"]
        original_builder = stage9_module.stage9_quality.build_star_reference_catalog

        def reject_source_reference(stars, cfg, **kwargs):
            if kwargs.get("source_image") is not None:
                return {
                    "status": "rejected",
                    "reason": (
                        "source_star_catalog_contamination_risk: "
                        "single_pixel_component_ratio=0.236>0.200"
                    ),
                    "source_component_density_per_megapixel": 611.0,
                    "source_component_density_max": 2500.0,
                    "source_single_pixel_component_ratio": 0.236,
                    "source_single_pixel_component_ratio_max": 0.20,
                }
            return original_builder(stars, cfg, **kwargs)

        with patch.object(
            stage9_module.stage9_quality,
            "build_star_reference_catalog",
            side_effect=reject_source_reference,
        ):
            stage9_star_remixing(processor)

        self.assertFalse(processor._stage9_starmask_stretch_failed)
        self.assertFalse(processor._stage9_starmask_preparation_failed)
        self.assertTrue(processor._stage9_star_reference_degraded)
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                (
                    "stage8_enhanced",
                    "starmask_stretched_reference_fallback",
                    0.40,
                )
            ],
        )
        self.assertTrue(any(call[0] == "asinh" for call in processor.cmd_calls))
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(
            report["star_plugin_preprocessing"]["status"],
            "plugin_stretch_bypassed_reference_degraded",
        )
        self.assertTrue(report["star_reference_degraded"])
        self.assertEqual(report["star_reference_primary"]["status"], "rejected")
        self.assertEqual(
            report["selected"]["attempt"],
            "screen_reference_degraded_strict",
        )
        self.assertFalse(report["delivery_fallback_used"])
        self.assertTrue(report["candidate_recovery_used"])
        self.assertEqual(
            report["starmask_calibration"]["support_mode"],
            "strict_recovery",
        )

    def test_stage9_restricted_stage8_handoff_routes_to_with_stars_review(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.star_intensity = 1.05
        processor.cfg.stage9_fallback_intensity_cap = 1.05
        processor._stage8_final_source = "stage8_input_starless"
        restricted_path = processor.process_dir / "stage8_input_starless.fit"
        restricted_path.write_bytes(b"restricted-stage8")
        processor._stage8_handoff = {
            "schema": "starun.stage8-handoff.v2",
            "source_stem": "stage8_input_starless",
            "passthrough": True,
            "restricted_downstream": True,
            "reason_code": "bright_nebula_halo_advisory",
            "final_quality": "ok",
            "lineage_verified": True,
            "source_artifact": {
                "artifact": "stage8_input_starless.fit",
                "sha256": hashlib.sha256(restricted_path.read_bytes()).hexdigest(),
                "pixel_sha256": None,
            },
        }
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage7_review_with_stars.fit").write_bytes(
            b"trusted-with-stars"
        )

        stage9_star_remixing(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertIn("prohibited formal star remix", message)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertTrue(report["upstream_passthrough"])
        self.assertTrue(report["stage9_fallback_used"])
        metadata = processor.result_metadata[-1]
        self.assertTrue(metadata["upstream_passthrough"])
        self.assertTrue(metadata["fallback_used"])
        self.assertEqual(
            metadata["reason_code"],
            "required_stars_preserved_in_review_fallback",
        )
        self.assertEqual(metadata["details"]["reason_text"], "使用 Stage 8 安全旁路源")

    def test_stage9_rebuilds_inadequate_plugin_starmask_with_builtin_asinh(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["星点拉伸"] = "SASP Star Stretch"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        stage9_star_remixing(processor)

        self.assertEqual(
            processor.previous_stage_remix_calls,
            [("stage8_enhanced", "starmask_stretched", processor.cfg.star_intensity)],
        )
        asinh_calls = [call for call in processor.cmd_calls if call[0] == "asinh"]
        self.assertTrue(asinh_calls)
        self.assertIn(("save", "starmask_stretched"), processor.cmd_calls)

    def test_stage9_never_runs_starless_finish_plugins(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["细节/结构增强2"] = "VeraLux Revela"
        processor.command_labels["最终微调颜色"] = "VeraLux Curves"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        saved_stems = []
        processor._save_stage_output = (
            lambda stem: saved_stems.append(stem) or True
        )

        stage9_star_remixing(processor)

        self.assertNotIn("stage9_starless_base", saved_stems)
        self.assertNotIn("细节/结构增强2", processor.command_chain_calls)
        self.assertNotIn("最终微调颜色", processor.command_chain_calls)
        self.assertNotIn("调色2（可选）", processor.command_chain_calls)
        self.assertNotIn(("save", "stage8_enhanced"), processor.cmd_calls)
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                (
                    "stage8_enhanced",
                    "starmask_stretched",
                    processor.cfg.star_intensity,
                )
            ],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["upstream_source_stem"], "stage8_enhanced")
        self.assertEqual(report["remix_base_stem"], "stage8_enhanced")
        self.assertEqual(report["source_stem"], "stage8_enhanced")

    def test_stage9_rejects_tampered_stage8_handoff_before_remix(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["细节/结构增强2"] = "VeraLux Revela"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage7_review_with_stars.fit").write_bytes(
            b"trusted-with-stars"
        )
        processor._stage8_handoff["source_artifact"]["sha256"] = "0" * 64

        stage9_star_remixing(processor)

        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertEqual(
            processor._stage9_final_source,
            "stage9_review_with_stars",
        )
        verification = processor.stage_json_reports[
            "stage9_stage8_handoff_verification.json"
        ]
        self.assertFalse(verification["verified"])
        self.assertIn("stage8_handoff_sha256_mismatch", verification["issues"])
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertNotIn("细节/结构增强2", processor.command_chain_calls)

    def test_stage9_legacy_v1_handoff_is_review_only(self):
        processor = self._new_processor()
        processor._stage8_handoff = {
            "schema": "starun.stage8-handoff.v1",
            "source_stem": "stage8_enhanced",
            "passthrough": False,
            "restricted_downstream": False,
        }
        (processor.process_dir / "stage7_review_with_stars.fit").write_bytes(
            b"trusted-with-stars"
        )

        stage9_star_remixing(processor)

        verification = processor.stage_json_reports[
            "stage9_stage8_handoff_verification.json"
        ]
        self.assertEqual(
            verification["status"],
            "legacy_stage8_handoff_review_only",
        )
        self.assertIn("legacy_delivery_contract", verification["issues"])
        self.assertFalse(processor._stage9_remix_formally_accepted)
        self.assertEqual(processor._stage9_final_source, "stage9_review_with_stars")

    def test_stage9_handoff_v3_rejects_identity_route_and_eligibility_tampering(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        for scenario, expected_issue in (
            ("filename", "stage8_handoff_artifact_filename_mismatch"),
            ("pixel", "stage8_handoff_pixel_identity_mismatch"),
            ("route", "stage8_handoff_processing_route_invalid"),
            ("formal", "stage8_handoff_not_formal_eligible"),
            (
                "top_artifact_sha",
                "stage8_handoff_top_level_artifact_sha256_mismatch",
            ),
        ):
            with self.subTest(scenario=scenario):
                processor = self._new_processor()
                if scenario == "filename":
                    processor._stage8_handoff["source_artifact"][
                        "artifact"
                    ] = "wrong.fit"
                elif scenario == "pixel":
                    processor._stage8_handoff["source_artifact"][
                        "pixel_sha256"
                    ] = "f" * 64
                elif scenario == "route":
                    processor._stage8_handoff["processing_route"] = "review_only"
                elif scenario == "formal":
                    processor._stage8_handoff["formal_eligible"] = False
                else:
                    processor._stage8_handoff["artifact_sha256"] = "e" * 64

                report = stage9_module._stage9_verify_stage8_handoff(
                    processor,
                    "stage8_enhanced",
                    processor._stage8_handoff,
                )

                self.assertFalse(report["verified"])
                self.assertIn(expected_issue, report["issues"])

    def test_stage9_handoff_v3_does_not_trust_or_reject_quality_string(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor._stage8_handoff["final_quality"] = "arbitrary_status_text"

        report = stage9_module._stage9_verify_stage8_handoff(
            processor,
            "stage8_enhanced",
            processor._stage8_handoff,
        )

        self.assertTrue(report["verified"], report)

    def test_stage9_handoff_v3_verifies_fits_and_decoded_pixel_domains(self):
        from processor_runtime import _read_fits_stage_fingerprint
        from stage8_starless_finish import (
            DECODED_PIXEL_SHA256_METHOD,
            FITS_DATA_SHA256_METHOD,
            persisted_fits_decoded_pixel_sha256,
        )

        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        source_path = processor.process_dir / "stage8_enhanced.fit"
        pixels = np.linspace(
            0.01,
            0.91,
            num=3 * 16 * 12,
            dtype=np.float32,
        ).reshape(3, 16, 12)
        fits.PrimaryHDU(pixels).writeto(source_path, overwrite=True)
        processor.saved_image_pixels["stage8_enhanced"] = pixels.copy()
        processor._fits_stage_fingerprint = _read_fits_stage_fingerprint
        fingerprint = _read_fits_stage_fingerprint(source_path)
        fits_data_sha = fingerprint["data_sha256"]
        decoded_pixel_sha = persisted_fits_decoded_pixel_sha256(source_path)
        self.assertNotEqual(fits_data_sha, decoded_pixel_sha)
        container_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        artifact = {
            "artifact": source_path.name,
            "sha256": container_sha,
            "pixel_sha256": fits_data_sha,
            "pixel_sha256_method": FITS_DATA_SHA256_METHOD,
            "fits_data_sha256": fits_data_sha,
            "fits_data_sha256_method": FITS_DATA_SHA256_METHOD,
            "decoded_pixel_sha256": decoded_pixel_sha,
            "decoded_pixel_sha256_method": DECODED_PIXEL_SHA256_METHOD,
            "identity_status": "verified",
        }
        handoff = processor._make_stage8_handoff("stage8_enhanced")
        handoff.update(
            source_artifact=artifact,
            artifact_sha256=container_sha,
            pixel_sha256=fits_data_sha,
            pixel_sha256_method=FITS_DATA_SHA256_METHOD,
            fits_data_sha256=fits_data_sha,
            fits_data_sha256_method=FITS_DATA_SHA256_METHOD,
            decoded_pixel_sha256=decoded_pixel_sha,
            decoded_pixel_sha256_method=DECODED_PIXEL_SHA256_METHOD,
        )
        handoff["lineage"]["output_artifact"] = artifact
        handoff = seal_stage8_handoff(handoff)

        report = stage9_module._stage9_verify_stage8_handoff(
            processor,
            "stage8_enhanced",
            handoff,
        )

        self.assertTrue(report["verified"], report)
        self.assertEqual(
            report["artifact"]["actual_fits_data_sha256"],
            fits_data_sha,
        )
        self.assertEqual(
            report["artifact"]["actual_decoded_pixel_sha256"],
            decoded_pixel_sha,
        )

        for field, replacement, expected_issue in (
            (
                "fits_data_sha256",
                "1" * 64,
                "stage8_handoff_fits_data_identity_mismatch",
            ),
            (
                "decoded_pixel_sha256",
                "2" * 64,
                "stage8_handoff_decoded_pixel_identity_mismatch",
            ),
            (
                "decoded_pixel_sha256_method",
                "untrusted_method",
                "stage8_handoff_decoded_pixel_identity_method_invalid",
            ),
        ):
            with self.subTest(field=field):
                tampered = copy.deepcopy(handoff)
                tampered["source_artifact"][field] = replacement
                tampered["lineage"]["output_artifact"] = dict(
                    tampered["source_artifact"]
                )
                tampered[field] = replacement
                rejected = stage9_module._stage9_verify_stage8_handoff(
                    processor,
                    "stage8_enhanced",
                    tampered,
                )
                self.assertFalse(rejected["verified"], rejected)
                self.assertIn(expected_issue, rejected["issues"])

    def test_stage9_accepts_complete_limited_safe_passthrough_contract(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        handoff = processor._stage8_handoff
        artifact = dict(handoff["source_artifact"])
        eligibility = {
            "schema": "starun.stage8-limited-safe-passthrough-eligibility.v1",
            "status": "eligible",
            "accepted": True,
            "checks": {
                "limited_policy": True,
                "guard_status": True,
                "guard_hard_reasons_clear": True,
                "guard_subject_reasons_clear": True,
                "upstream_quality_ok": True,
                "upstream_reason_is_safe": True,
                "upstream_hard_metric_evidence_complete": True,
                "upstream_hard_metrics_clear": True,
                "final_candidate_ready": True,
                "processing_mode_safe": True,
                "no_external_override": True,
                "review_requirement_free": True,
            },
            "issues": [],
            "review_requirements": [],
            "upstream_hard_metrics": {
                name: {"accepted": True}
                for name in (
                    "residual_star_score",
                    "starless_noise_gain",
                    "effective_halo_residue_score",
                )
            },
        }
        preflight = {
            "schema": "starun.stage8-safe-passthrough-preflight.v1",
            "status": "accepted",
            "accepted": True,
            "source_mode": "limited_safe_passthrough",
            "eligibility": eligibility,
            "checks": {
                name: {"accepted": True}
                for name in (
                    "exact_structure_rollback",
                    "stage7_presentation_reference",
                    "spatial_background",
                    "subject_boundary_seam",
                    "star_halo",
                    "clipping",
                )
            },
        }
        final_validation = {
            "schema": "starun.stage8-safe-passthrough-final.v1",
            "status": "accepted",
            "accepted": True,
            "checks": {
                "color": {"accepted": True, "mode": "verified_pixel_identity"},
                "background_seam_clip_presentation": {"status": "ok"},
                "spatial_background": {"accepted": True},
                "star_halo": {"accepted": True},
                "artifact": {"accepted": True},
            },
        }
        handoff.update(
            processing_route="safe_passthrough_color_only",
            formal_eligible=True,
            restricted_downstream=False,
            passthrough=True,
            safe_passthrough_color_only={
                "limited_eligibility": eligibility,
                "preflight": preflight,
                "final_validation": final_validation,
                "color_gate_verified": True,
            },
            color_gate={
                "used_for_gate": True,
                "status": "reported",
                "guard_lineage_verified": True,
                "final_pixel_identity": {
                    "sha256": artifact["sha256"],
                    "pixel_sha256": artifact["pixel_sha256"],
                },
            },
            star_halo_guard={"status": "ok", "verified": True},
        )
        handoff = seal_stage8_handoff(handoff)

        report = stage9_module._stage9_verify_stage8_handoff(
            processor,
            "stage8_enhanced",
            handoff,
        )

        self.assertTrue(report["verified"], report)
        self.assertEqual(report["processing_route"], "safe_passthrough_color_only")

    def test_stage9_safe_passthrough_contract_tampering_fails_closed(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        def safe_handoff(processor):
            handoff = processor._stage8_handoff
            artifact = dict(handoff["source_artifact"])
            eligibility = {
                "accepted": True,
                "issues": [],
                "review_requirements": [],
                "checks": {"review_requirement_free": True},
                "upstream_hard_metrics": {
                    "halo": {"accepted": True},
                },
            }
            preflight = {
                "accepted": True,
                "source_mode": "limited_safe_passthrough",
                "eligibility": eligibility,
                "checks": {
                    name: {"accepted": True}
                    for name in (
                        "exact_structure_rollback",
                        "stage7_presentation_reference",
                        "spatial_background",
                        "subject_boundary_seam",
                        "star_halo",
                        "clipping",
                    )
                },
            }
            handoff.update(
                processing_route="safe_passthrough_color_only",
                formal_eligible=True,
                restricted_downstream=False,
                passthrough=True,
                safe_passthrough_color_only={
                    "limited_eligibility": eligibility,
                    "preflight": preflight,
                    "final_validation": {
                        "accepted": True,
                        "checks": {
                            "color": {"accepted": True},
                            "background_seam_clip_presentation": {"status": "ok"},
                            "spatial_background": {"accepted": True},
                            "star_halo": {"accepted": True},
                            "artifact": {"accepted": True},
                        },
                    },
                    "color_gate_verified": True,
                },
                color_gate={
                    "used_for_gate": True,
                    "status": "reported",
                    "guard_lineage_verified": True,
                    "final_pixel_identity": {
                        "sha256": artifact["sha256"],
                        "pixel_sha256": artifact["pixel_sha256"],
                    },
                },
                star_halo_guard={"status": "ok", "verified": True},
            )
            return seal_stage8_handoff(handoff)

        for scenario, expected_issue in (
            (
                "limited_eligibility",
                "stage8_limited_safe_passthrough_eligibility_unverified",
            ),
            (
                "seam",
                "stage8_safe_passthrough_preflight_gate_unverified",
            ),
            (
                "final_quality",
                "stage8_safe_passthrough_final_gate_unverified",
            ),
            (
                "color",
                "stage8_safe_passthrough_color_gate_unverified",
            ),
            (
                "halo",
                "stage8_safe_passthrough_halo_guard_unverified",
            ),
        ):
            with self.subTest(scenario=scenario):
                processor = self._new_processor()
                handoff = safe_handoff(processor)
                safe = handoff["safe_passthrough_color_only"]
                if scenario == "limited_eligibility":
                    safe["limited_eligibility"]["checks"][
                        "review_requirement_free"
                    ] = False
                elif scenario == "seam":
                    safe["preflight"]["checks"]["subject_boundary_seam"][
                        "accepted"
                    ] = False
                elif scenario == "final_quality":
                    safe["final_validation"]["checks"][
                        "background_seam_clip_presentation"
                    ]["status"] = "poor"
                elif scenario == "color":
                    safe["color_gate_verified"] = False
                else:
                    handoff["star_halo_guard"]["verified"] = False

                report = stage9_module._stage9_verify_stage8_handoff(
                    processor,
                    "stage8_enhanced",
                    handoff,
                )

                self.assertFalse(report["verified"], report)
                self.assertIn(expected_issue, report["issues"])

    def test_stage9_partial_star_plugin_result_feeds_builtin_stretch(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["星点去紫"] = "SCNR"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        original_cmd = processor.cmd_with_check

        def cmd_with_saved_partial(*args, **kwargs):
            result = original_cmd(*args, **kwargs)
            if args[:2] == ("save", "starmask_plugin_processed"):
                (processor.process_dir / "starmask_plugin_processed.fit").write_bytes(
                    b"partial"
                )
            return result

        processor.cmd_with_check = cmd_with_saved_partial

        stage9_star_remixing(processor)

        self.assertIn(
            ("load", "starmask_plugin_processed"),
            processor.cmd_calls,
        )
        self.assertIn(
            ("asinh", "2.000", "0.00100", "-clipmode=rgbblend"),
            processor.cmd_calls,
        )
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                (
                    "stage8_enhanced",
                    "starmask_stretched",
                    processor.cfg.star_intensity,
                )
            ],
        )
        preprocessing = processor.stage_json_reports[
            "stage9_remix_quality.json"
        ]["star_plugin_preprocessing"]
        self.assertEqual(preprocessing["status"], "partial_plugin_processing")
        self.assertTrue(preprocessing["builtin_stretch_required"])
        self.assertEqual(preprocessing["applied_steps"][0]["step"], "scnr")

    def test_stage9_star_color_post_gate_is_independent(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.stage9_quality_gate_enabled = False
        processor._stage9_star_color_reference_samples = {"samples": [1]}
        processor._stage9_last_star_layer = np.zeros((3, 4, 4), dtype=np.float32)
        processor._stage9_last_star_overlay_mask = np.ones((4, 4), dtype=bool)
        processor._stage9_assess_current_remix = lambda *_args, **_kwargs: {
            "attempt": "screen_primary",
            "formula": "screen",
            "status": "ok",
            "accepted": True,
            "gate_enabled": False,
            "issues": [],
            "metrics": {},
        }
        rejected_color = {
            "status": "rejected",
            "accepted": False,
            "issues": ["post_chroma_error"],
        }

        with patch.object(
            stage9_module,
            "assess_repaired_star_layer",
            side_effect=lambda *_args, **_kwargs: rejected_color.copy(),
        ) as color_assessor:
            processor.cfg.stage9_star_color_post_validation_enabled = True
            enforced = stage9_module._assess_stage9_candidate(
                processor,
                "stage8_enhanced",
                attempt="screen_primary",
                formula="screen",
            )
            processor.cfg.stage9_star_color_post_validation_enabled = False
            observed_only = stage9_module._assess_stage9_candidate(
                processor,
                "stage8_enhanced",
                attempt="screen_primary",
                formula="screen",
            )
            processor._stage9_assess_current_remix = (
                lambda *_args, **_kwargs: {
                    "attempt": "screen_primary",
                    "formula": "screen",
                    "status": "rejected",
                    "accepted": False,
                    "issues": ["preliminary_psf_failure"],
                    "metrics": {},
                }
            )
            processor.cfg.stage9_star_color_post_validation_enabled = True
            preliminary_rejected = stage9_module._assess_stage9_candidate(
                processor,
                "stage8_enhanced",
                attempt="screen_primary",
                formula="screen",
            )

        self.assertFalse(enforced["accepted"])
        self.assertTrue(enforced["star_color_validation"]["gate_enabled"])
        self.assertTrue(observed_only["accepted"])
        self.assertFalse(
            observed_only["star_color_validation"]["gate_enabled"]
        )
        self.assertFalse(observed_only["star_color_validation"]["enforced"])
        self.assertFalse(preliminary_rejected["accepted"])
        self.assertIn("star_color_validation", preliminary_rejected)
        self.assertEqual(color_assessor.call_count, 3)

    def test_stage9_star_color_post_gate_warns_within_fifty_percent_band(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor._stage9_star_color_reference_samples = {"samples": [1]}
        processor._stage9_last_star_layer = np.zeros((3, 4, 4), dtype=np.float32)
        processor._stage9_last_star_overlay_mask = np.ones((4, 4), dtype=bool)
        processor._stage9_assess_current_remix = lambda *_args, **_kwargs: {
            "attempt": "screen_primary",
            "formula": "screen",
            "status": "ok",
            "accepted": True,
            "gate_enabled": True,
            "issues": [],
            "metrics": {},
        }
        moderate_color = {
            "status": "rejected",
            "accepted": False,
            "issues": ["post_stretch_star_chroma_error"],
            "metrics": {
                "median_chroma_error": 0.30,
                "extreme_chroma_outlier_ratio": 0.22,
            },
            "limits": {
                "median_chroma_error_max": 0.22,
                "extreme_chroma_outlier_ratio_max": 0.20,
            },
        }
        hard_color = {
            **moderate_color,
            "metrics": {
                **moderate_color["metrics"],
                "median_chroma_error": 0.34,
            },
        }

        with patch.object(
            stage9_module,
            "assess_repaired_star_layer",
            side_effect=lambda *_args, **_kwargs: moderate_color.copy(),
        ):
            advisory = stage9_module._assess_stage9_candidate(
                processor,
                "stage8_enhanced",
                attempt="screen_primary",
                formula="screen",
            )
        with patch.object(
            stage9_module,
            "assess_repaired_star_layer",
            side_effect=lambda *_args, **_kwargs: hard_color.copy(),
        ):
            rejected = stage9_module._assess_stage9_candidate(
                processor,
                "stage8_enhanced",
                attempt="screen_primary",
                formula="screen",
            )

        self.assertTrue(advisory["accepted"])
        self.assertEqual(
            advisory["star_color_validation"]["status"],
            "advisory",
        )
        self.assertTrue(advisory["advisories"])
        self.assertFalse(rejected["accepted"])
        self.assertEqual(
            rejected["star_color_validation"]["quality_gates"][
                "median_chroma_error"
            ]["status"],
            "hard_failed",
        )

    def test_stage9_gate_rolls_back_primary_and_accepts_fallback_intensity(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        review_calls = []
        processor._create_stage_review_bundle = (
            lambda *args, **kwargs: (
                review_calls.append((args, kwargs))
                or {
                    "status": "ready",
                    "report_path": str(
                        processor.process_dir
                        / "review_bundles"
                        / "stage9_star_remixing"
                        / "review.json"
                    ),
                }
            )
        )

        def assess(_source_stem, *, attempt, formula):
            accepted = attempt == "screen_fallback_075"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else ["background_lift 0.020000>0.010000"],
                "metrics": {"background_lift": 0.0 if accepted else 0.02},
            }

        processor._stage9_assess_current_remix = assess

        stage9_star_remixing(processor)

        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                ("stage8_enhanced", "starmask_stretched", processor.cfg.star_intensity),
                ("stage8_enhanced", "starmask_stretched", 0.75),
            ],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_fallback_075")
        self.assertFalse(report["stage9_fallback_used"])
        self.assertIsNone(report["stage9_fallback_reason"])
        self.assertTrue(report["candidate_recovery_used"])
        self.assertFalse(report["delivery_fallback_used"])
        self.assertEqual(len(review_calls), 1)
        review_args, review_kwargs = review_calls[0]
        self.assertEqual(
            review_args,
            ("stage9_star_remixing", "stage8_enhanced", "stage9_remixed"),
        )
        self.assertEqual(
            review_kwargs["selected_candidate"],
            "screen_fallback_075",
        )
        self.assertEqual(review_kwargs["context"]["mode"], "screen")
        self.assertEqual(len(review_kwargs["candidates"]), 2)
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(processor.result_metadata[-1]["reason_code"], "")

    def test_stage9_local_fallback_classifier_excludes_upstream_passthrough(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        cases = (
            ("screen", {"attempt": "primary"}, "screen", False, None),
            (
                "screen",
                {"attempt": "screen_compact_primary"},
                "screen",
                False,
                None,
            ),
            (
                "screen",
                {"attempt": "screen_fallback_075"},
                "screen",
                False,
                None,
            ),
            (
                "screen",
                {"attempt": "screen_compact_recovery"},
                "screen",
                False,
                None,
            ),
            (
                "screen",
                {"attempt": "screen_reference_degraded_strict"},
                "screen",
                False,
                None,
            ),
            (
                "best_failed_review_candidate",
                {"attempt": "screen_unscreen_fallback_040"},
                "screen_review_candidate",
                True,
                "best_failed_candidate_review",
            ),
            (
                "stage8_starmask_review_fallback",
                {"attempt": "screen_stage8_starmask_raw_fallback"},
                "screen_minimal_review_fallback",
                True,
                "stage8_starmask_review_fallback",
            ),
            (
                "stage5_review_fallback",
                None,
                "with_stars_review_fallback",
                True,
                "all_remix_candidates_outside_review_range_stage5",
            ),
            (
                "unsafe_starless_bypass",
                None,
                "unsafe_starless_bypass",
                True,
                "unsafe_starless_bypass",
            ),
            (
                "rejected_keep_starless",
                None,
                "rejected_keep_starless",
                True,
                "all_remix_candidates_rejected",
            ),
            (
                "starmask_preparation_failed",
                None,
                "starmask_preparation_failed",
                True,
                "starmask_preparation_failed_keep_upstream",
            ),
            (
                "starmask_stretch_failed",
                None,
                "starmask_stretch_failed",
                True,
                "starmask_stretch_failed_keep_upstream",
            ),
            ("no_starmask", None, "no_starmask", False, None),
            (
                "star_preserve_target_bypass",
                None,
                "not_required_star_preserve",
                False,
                None,
            ),
        )
        for mode, selected, application_mode, expected_used, expected_reason in cases:
            with self.subTest(mode=mode, selected=selected):
                used, reason = stage9_module._stage9_local_fallback(
                    mode,
                    selected,
                    application_mode,
                )
                self.assertEqual(used, expected_used)
                self.assertEqual(reason, expected_reason)

    def test_stage9_review_bundle_marks_safe_rollback_as_selected(self):
        processor = self._new_processor()
        review_calls = []
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = False
        processor._stage9_stars_application_mode = "rejected_keep_starless"
        processor._stage9_star_reference_summary = {
            "status": "rejected",
            "reason": "source_star_catalog_contamination_risk",
        }
        processor._create_stage_review_bundle = (
            lambda *args, **kwargs: (
                review_calls.append((args, kwargs))
                or {"status": "ready", "report_path": "/tmp/stage9-review.json"}
            )
        )
        messages = []
        rejected_attempt = {
            "attempt": "screen_primary",
            "status": "rejected",
            "accepted": False,
            "issues": ["new_hollow_structure_max_area 1250>64"],
        }

        stage9_module = sys.modules["stages.stage9_star_remixing"]
        stage9_module._append_stage9_review_bundle(
            processor,
            messages,
            [rejected_attempt],
            None,
            source_stem="stage8_enhanced",
            mode="rejected_keep_starless",
            stage_saved=True,
        )

        self.assertEqual(len(review_calls), 1)
        _args, kwargs = review_calls[0]
        self.assertEqual(kwargs["selected_candidate"], "stage9_safe_rollback")
        self.assertEqual(kwargs["candidates"][0], rejected_attempt)
        self.assertEqual(kwargs["candidates"][-1]["id"], "stage9_safe_rollback")
        self.assertTrue(kwargs["candidates"][-1]["selected"])
        self.assertIn("review_bundle=/tmp/stage9-review.json", messages)

    def test_stage9_rebuilds_strict_compact_mask_before_lowering_intensity(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_starmask_pre_stretch_compact_enabled = True
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"mock")
        yy, xx = np.mgrid[:128, :128]
        diffuse_floor = 0.00004 + 0.00003 * np.sin(xx / 6.0) ** 2
        pixels = np.stack(
            [diffuse_floor, diffuse_floor * 0.9, diffuse_floor * 0.8]
        ).astype(np.float32)
        for cy, cx, amplitude in (
            (22, 24, 0.05),
            (41, 92, 0.08),
            (73, 61, 0.12),
            (99, 31, 0.06),
            (96, 105, 0.18),
        ):
            profile = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.4**2))
            pixels += np.stack(
                [profile * amplitude, profile * amplitude * 0.85, profile * amplitude * 0.70]
            ).astype(np.float32)
        written_pixels = []
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=lambda output: written_pixels.append(output.copy()),
        )

        def assess(_source_stem, *, attempt, formula):
            accepted = attempt == "screen_compact_recovery"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else [
                    "background_mottling_growth 2.000000>1.350000"
                ],
                "metrics": {"changed_pixel_ratio": 0.25 if not accepted else 0.03},
                "limits": {
                    "changed_pixel_ratio": 0.35,
                    "background_mottling_low_absolute_changed_pixel_ratio_max": 0.12,
                },
            }

        processor._stage9_assess_current_remix = assess

        stage9_star_remixing(processor)

        self.assertEqual(len(written_pixels), 2)
        normal_coverage = float(np.mean(np.max(written_pixels[0], axis=0) > 0.0))
        recovery_coverage = float(np.mean(np.max(written_pixels[1], axis=0) > 0.0))
        self.assertLess(recovery_coverage, normal_coverage)
        self.assertEqual(
            processor.previous_stage_remix_calls,
            [
                ("stage8_enhanced", "starmask_stretched", processor.cfg.star_intensity),
                (
                    "stage8_enhanced",
                    "starmask_stretched_recovery",
                    processor.cfg.star_intensity,
                ),
            ],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_compact_recovery")
        self.assertFalse(report["stage9_fallback_used"])
        self.assertTrue(report["candidate_recovery_used"])
        self.assertFalse(report["delivery_fallback_used"])
        self.assertTrue(report["starmask_calibration"]["recovery_attempted"])
        self.assertTrue(report["starmask_calibration"]["recovery_applied"])
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertTrue(report["psf_review_required"])
        self.assertEqual(
            report["stage9_spatial_scale"]["source"],
            "raw_starmask_halfmax",
        )

    def test_stage9_preflight_skips_hard_wide_support_and_selects_compact_primary(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"mock")
        yy, xx = np.mgrid[:128, :128]
        pixels = np.zeros((3, 128, 128), dtype=np.float32)
        for cy, cx, amplitude in (
            (22, 24, 0.05),
            (41, 92, 0.08),
            (73, 61, 0.12),
            (99, 31, 0.06),
            (96, 105, 0.18),
        ):
            profile = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.4**2)
            )
            pixels += np.stack(
                [
                    profile * amplitude,
                    profile * amplitude * 0.85,
                    profile * amplitude * 0.70,
                ]
            ).astype(np.float32)
        processor.image_pixels = pixels.copy()
        processor.saved_image_pixels["starmask"] = pixels.copy()
        processor.saved_image_pixels["stage5_linear"] = np.clip(
            pixels + 0.02,
            0.0,
            1.0,
        )
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: processor.image_pixels.copy(),
            set_image_pixeldata=lambda output: setattr(
                processor,
                "image_pixels",
                np.array(output, copy=True),
            ),
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        def strict_only_preflight(pipeline, *_args, **_kwargs):
            report = stage9_module.stage9_quality.assess_starmask_support_preflight(
                pixels,
                pipeline.cfg,
                reference_catalog=pipeline._stage9_star_reference_catalog,
            )
            report["route"] = "strict_only"
            report["reason_code"] = "stage9_support_preflight_normal_hard_failed"
            report["planned_candidates"] = ["strict_compact"]
            report["skipped_candidates"] = [
                {
                    "support_mode": "normal",
                    "reason_code": (
                        "stage9_support_preflight_normal_hard_failed"
                    ),
                }
            ]
            pipeline._stage9_starmask_support_preflight = (
                stage9_module.stage9_quality.public_starmask_support_preflight(
                    report
                )
            )
            return report

        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: {
                "attempt": attempt,
                "formula": formula,
                "status": "ok",
                "accepted": True,
                "issues": [],
                "advisories": [],
                "metrics": {},
            }
        )
        with patch.object(
            stage9_module,
            "_stage9_starmask_support_preflight",
            side_effect=strict_only_preflight,
        ):
            stage9_star_remixing(processor)

        self.assertEqual(
            processor.previous_stage_remix_calls[0][1],
            "starmask_stretched_compact_primary",
        )
        self.assertFalse(
            any(
                call[1] == "starmask_stretched"
                for call in processor.previous_stage_remix_calls
            )
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_compact_primary")
        self.assertFalse(report["stage9_fallback_used"])
        self.assertIn(
            "stage9_support_preflight_strict_selected",
            report["selected"]["reason_codes"],
        )
        self.assertEqual(
            report["starmask_support_preflight"]["selected_support_mode"],
            "strict_compact",
        )

    def test_stage9_boundary_preflight_compares_both_and_prefers_clean_compact(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_unscreen_candidate_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"mock")
        yy, xx = np.mgrid[:128, :128]
        pixels = np.zeros((3, 128, 128), dtype=np.float32)
        for cy, cx, amplitude in (
            (22, 24, 0.05),
            (41, 92, 0.08),
            (73, 61, 0.12),
            (99, 31, 0.06),
            (96, 105, 0.18),
        ):
            profile = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.4**2)
            )
            pixels += np.stack(
                [
                    profile * amplitude,
                    profile * amplitude * 0.85,
                    profile * amplitude * 0.70,
                ]
            ).astype(np.float32)
        processor.image_pixels = pixels.copy()
        processor.saved_image_pixels["starmask"] = pixels.copy()
        processor.saved_image_pixels["stage5_linear"] = np.clip(
            pixels + 0.02,
            0.0,
            1.0,
        )
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: processor.image_pixels.copy(),
            set_image_pixeldata=lambda output: setattr(
                processor,
                "image_pixels",
                np.array(output, copy=True),
            ),
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        def dual_preflight(pipeline, *_args, **_kwargs):
            report = stage9_module.stage9_quality.assess_starmask_support_preflight(
                pixels,
                pipeline.cfg,
                reference_catalog=pipeline._stage9_star_reference_catalog,
            )
            report["route"] = "dual_competition"
            report["reason_code"] = "stage9_support_preflight_boundary_dual"
            report["planned_candidates"] = ["normal", "strict_compact"]
            report["skipped_candidates"] = []
            pipeline._stage9_starmask_support_preflight = (
                stage9_module.stage9_quality.public_starmask_support_preflight(
                    report
                )
            )
            return report

        def assess(_source, *, attempt, formula):
            normal = attempt == "screen_primary"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "advisory" if normal else "ok",
                "accepted": True,
                "issues": [],
                "advisories": ["star_support_ratio advisory"] if normal else [],
                "quality_gates": {
                    "star_support_ratio": {
                        "status": "advisory",
                        "advisory": True,
                        "hard_failed": False,
                        "severity_ratio": 1.1,
                    }
                }
                if normal
                else {},
                "metrics": {
                    "star_support_ratio": 0.13 if normal else 0.08,
                    "highlight_clip_growth": 0.001,
                    "bright_pixel_growth": 0.002,
                },
                "limits": {},
            }

        processor._stage9_assess_current_remix = assess
        with patch.object(
            stage9_module,
            "_stage9_starmask_support_preflight",
            side_effect=dual_preflight,
        ):
            stage9_star_remixing(processor)

        self.assertEqual(
            [call[1] for call in processor.previous_stage_remix_calls[:2]],
            ["starmask_stretched", "starmask_stretched_compact_primary"],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_compact_primary")
        self.assertFalse(report["stage9_fallback_used"])
        self.assertIn(
            ("load", "stage9_candidate_support_strict_compact_primary"),
            processor.cmd_calls,
        )
        self.assertIn(
            "stage9_support_dual_compact_selected",
            report["selected"]["reason_codes"],
        )

    def test_stage9_compact_primary_then_lower_intensity_is_fallback(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_unscreen_candidate_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        def strict_only_preflight(pipeline, *_args, **_kwargs):
            pipeline.siril.get_image_pixeldata = (
                lambda preview=False: pipeline.image_pixels.copy()
            )
            pipeline.siril.set_image_pixeldata = lambda image: setattr(
                pipeline,
                "image_pixels",
                np.array(image, copy=True),
            )
            support = np.ones(pipeline.image_pixels.shape[-2:], dtype=bool)
            strict_calibration = {
                "status": "ok",
                "support_mode": "strict_recovery",
                "stretch": 2.0,
                "offset": 0.001,
                "star_sample_count": 8,
                "compact_component_count": 1,
                "compact_support_coverage": 0.08,
                "predicted_change_ratio": 0.08,
                "weak_star_retention": 1.0,
                "star_retention": 1.0,
                "_compact_support_mask": support,
                "_weak_support_mask": support.copy(),
                "_bright_support_mask": np.zeros_like(support),
            }
            report = {
                "schema": "starun.stage9-starmask-support-preflight.v2",
                "status": "ready",
                "route": "strict_only",
                "reason_code": "stage9_support_preflight_normal_hard_failed",
                "planned_candidates": ["strict_compact"],
                "skipped_candidates": [],
                "candidates": {},
                "executed_candidates": [],
                "selected_support_mode": None,
                "_calibrations": {
                    "strict_compact": strict_calibration,
                },
            }
            pipeline._stage9_starmask_support_preflight = (
                stage9_module.stage9_quality.public_starmask_support_preflight(
                    report
                )
            )
            return report

        def assess(_source, *, attempt, formula):
            accepted = attempt == "screen_fallback_075"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else [
                    "highlight_clip_growth 0.020000>0.010000"
                ],
                "advisories": [],
                "metrics": {
                    "highlight_clip_growth": 0.0 if accepted else 0.02,
                },
                "limits": {"highlight_clip_growth": 0.01},
            }

        processor._stage9_assess_current_remix = assess
        with patch.object(
            stage9_module,
            "_stage9_starmask_support_preflight",
            side_effect=strict_only_preflight,
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_fallback_075")
        self.assertEqual(report["selected"]["support_mode"], "strict_compact")
        self.assertFalse(report["stage9_fallback_used"])
        self.assertTrue(report["candidate_recovery_used"])
        self.assertFalse(report["delivery_fallback_used"])
        self.assertTrue(
            all(
                "support_mode" in attempt
                and "support_preflight_route" in attempt
                for attempt in report["attempts"]
            )
        )

    def test_stage9_dual_strict_preparation_failure_keeps_normal_candidate(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_unscreen_candidate_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        def dual_preflight(pipeline, *_args, **_kwargs):
            report = {
                "schema": "starun.stage9-starmask-support-preflight.v2",
                "status": "ready",
                "route": "dual_competition",
                "reason_code": "stage9_support_preflight_boundary_dual",
                "planned_candidates": ["normal", "strict_compact"],
                "skipped_candidates": [],
                "candidates": {},
                "executed_candidates": [],
                "selected_support_mode": None,
                "_calibrations": {},
            }
            pipeline._stage9_starmask_support_preflight = (
                stage9_module.stage9_quality.public_starmask_support_preflight(
                    report
                )
            )
            return report

        def prepare(
            pipeline,
            starmask_name,
            *,
            strict_support,
            output_name,
            **_kwargs,
        ):
            if strict_support:
                pipeline._stage9_starmask_calibration = {
                    "status": "failed",
                    "reason": "strict preparation mock failure",
                    "failure_phase": "starmask_preparation",
                }
                return starmask_name
            pipeline._stage9_starmask_calibration = {"status": "ok"}
            return output_name

        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: {
                "attempt": attempt,
                "formula": formula,
                "status": "ok",
                "accepted": True,
                "issues": [],
                "advisories": [],
                "metrics": {},
            }
        )
        with (
            patch.object(
                stage9_module,
                "_stage9_starmask_support_preflight",
                side_effect=dual_preflight,
            ),
            patch.object(
                stage9_module,
                "_prepare_stage9_starmask_for_pixel_remix",
                side_effect=prepare,
            ),
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_primary")
        self.assertFalse(report["starmask_preparation_failed"])
        self.assertFalse(report["starmask_stretch_failed"])
        self.assertEqual(
            report["starmask_support_preflight"]["preparation_failures"][0][
                "support_mode"
            ],
            "strict_compact",
        )

    def test_stage9_all_preflight_supports_unavailable_preserves_with_stars(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        def unavailable_preflight(pipeline, *_args, **_kwargs):
            report = {
                "schema": "starun.stage9-starmask-support-preflight.v2",
                "status": "rejected",
                "route": "unavailable",
                "reason_code": "stage9_support_preflight_no_usable_candidate",
                "planned_candidates": [],
                "skipped_candidates": [
                    {"support_mode": "normal", "status": "unavailable"},
                    {
                        "support_mode": "strict_compact",
                        "status": "unavailable",
                    },
                ],
                "candidates": {},
                "executed_candidates": [],
                "selected_support_mode": None,
                "_calibrations": {},
            }
            pipeline._stage9_starmask_support_preflight = (
                stage9_module.stage9_quality.public_starmask_support_preflight(
                    report
                )
            )
            return report

        with patch.object(
            stage9_module,
            "_stage9_starmask_support_preflight",
            side_effect=unavailable_preflight,
        ):
            _stage9_star_remixing_impl(processor)

        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertTrue(processor._stage9_starmask_preparation_failed)
        self.assertFalse(processor._stage9_starmask_stretch_failed)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "with_stars_review_fallback")
        self.assertTrue(report["output_contains_stars"])
        self.assertFalse(report["output_withheld"])
        self.assertEqual(report["attempts"], [])

    def test_stage9_compact_starmask_write_claims_image_lock(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_starmask_pre_stretch_compact_enabled = True
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        pixels = np.full((3, 8, 8), 0.01, dtype=np.float32)
        writes = []
        lock_events = []

        class ImageLock:
            def __enter__(self):
                lock_events.append("enter")

            def __exit__(self, _type, _value, _traceback):
                lock_events.append("exit")

        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=lambda output: writes.append(output.copy()),
            image_lock=lambda: ImageLock(),
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        calibration = {
            "status": "ok",
            "stretch": 12.0,
            "offset": 0.001,
            "star_sample_count": 64,
            "compact_component_count": 4,
            "_compact_support_mask": np.ones((8, 8), dtype=bool),
        }

        with patch.object(
            stage9_module.stage9_quality,
            "calibrate_starmask_asinh",
            return_value=calibration,
        ):
            stage9_star_remixing(processor)

        self.assertEqual(lock_events, ["enter", "exit"])
        self.assertEqual(len(writes), 1)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertFalse(report["starmask_stretch_failed"])
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage9_starmask_pixel_write_failure_stops_raw_linear_remix(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_starmask_pre_stretch_compact_enabled = True
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        pixels = np.full((3, 8, 8), 0.01, dtype=np.float32)

        def fail_pixel_write(_output):
            raise pipeline_module.SirilError("processing thread is not claimed")

        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=fail_pixel_write,
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        calibration = {
            "status": "ok",
            "stretch": 12.0,
            "offset": 0.001,
            "star_sample_count": 64,
            "compact_component_count": 4,
            "_compact_support_mask": np.ones((8, 8), dtype=bool),
        }

        with patch.object(
            stage9_module.stage9_quality,
            "calibrate_starmask_asinh",
            return_value=calibration,
        ):
            stage9_star_remixing(processor)

        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertFalse(processor._stage9_starmask_stretch_failed)
        self.assertTrue(processor._stage9_starmask_preparation_failed)
        self.assertFalse(processor._stage9_stars_applied)
        self.assertEqual(processor.results[-1][1], "degraded")
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertFalse(report["starmask_stretch_failed"])
        self.assertTrue(report["starmask_preparation_failed"])
        self.assertEqual(report["mode"], "with_stars_review_fallback")
        self.assertTrue(report["output_contains_stars"])
        self.assertFalse(report["output_withheld"])

    def test_stage9_asinh_command_failure_is_reported_as_stretch_failure(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor.fail_commands.add("asinh")
        pixels = np.full((3, 8, 8), 0.01, dtype=np.float32)
        processor.siril = SimpleNamespace(
            get_image_shape=lambda: pixels.shape,
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=lambda _output: None,
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        calibration = {
            "status": "ok",
            "stretch": 12.0,
            "offset": 0.001,
            "star_sample_count": 64,
            "compact_component_count": 4,
            "_compact_support_mask": np.ones((8, 8), dtype=bool),
        }

        with patch.object(
            stage9_module.stage9_quality,
            "calibrate_starmask_asinh",
            return_value=calibration,
        ):
            stage9_star_remixing(processor)

        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertTrue(processor._stage9_starmask_stretch_failed)
        self.assertFalse(processor._stage9_starmask_preparation_failed)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertTrue(report["starmask_stretch_failed"])
        self.assertFalse(report["starmask_preparation_failed"])
        self.assertEqual(report["mode"], "with_stars_review_fallback")
        self.assertTrue(report["output_contains_stars"])
        self.assertEqual(
            report["starmask_calibration"]["failure_phase"],
            "stretch_execution",
        )

    def test_stage9_pre_stretch_compaction_is_independent_from_support_routing(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        raw = np.full((3, 8, 8), 0.2, dtype=np.float32)
        support = np.zeros((8, 8), dtype=bool)
        support[2:6, 2:6] = True
        combinations = (
            (True, False, False),
            (True, True, True),
            (False, False, False),
            (False, True, True),
        )

        for support_enabled, pre_compact_enabled, expected_apply in combinations:
            with self.subTest(
                support_enabled=support_enabled,
                pre_compact_enabled=pre_compact_enabled,
            ):
                processor = self._new_processor()
                processor.cfg.stage9_compact_starmask_enabled = support_enabled
                processor.cfg.stage9_starmask_pre_stretch_compact_enabled = (
                    pre_compact_enabled
                )
                writes = []
                processor.siril = SimpleNamespace(
                    get_image_pixeldata=lambda preview=False: raw.copy(),
                    set_image_pixeldata=lambda output: writes.append(
                        np.asarray(output).copy()
                    ),
                )
                calibration = {
                    "status": "ok",
                    "support_mode": "normal",
                    "stretch": 2.0,
                    "offset": 0.001,
                    "star_sample_count": 32,
                    "compact_component_count": 4,
                    "predicted_change_ratio": 0.01,
                    "predicted_change_ratio_limit": 0.30,
                    "_compact_support_mask": support,
                }

                with patch.object(
                    stage9_module.stage9_quality,
                    "apply_compact_starmask_support",
                    wraps=(
                        stage9_module.stage9_quality.apply_compact_starmask_support
                    ),
                ) as compact:
                    prepared = (
                        stage9_module._prepare_stage9_starmask_for_pixel_remix(
                            processor,
                            "starmask",
                            star_stretch_used=False,
                            messages=[],
                            precomputed_calibration=calibration,
                        )
                    )

                self.assertEqual(prepared, "starmask_stretched")
                self.assertEqual(compact.call_count, int(expected_apply))
                report = processor._stage9_starmask_calibration
                self.assertEqual(
                    report["compact_support_enabled"], support_enabled
                )
                self.assertEqual(
                    report["pre_stretch_compact_enabled"],
                    pre_compact_enabled,
                )
                self.assertEqual(
                    report["compact_layer_applied"], expected_apply
                )
                if expected_apply:
                    self.assertEqual(len(writes), 1)
                    weights = (
                        stage9_module.stage9_quality._compact_starmask_support_weights(
                            support
                        )
                    )
                    np.testing.assert_allclose(
                        writes[0],
                        raw * weights[np.newaxis, ...],
                    )
                else:
                    self.assertEqual(writes, [])

    def test_stage9_sasp_stretched_layer_bypasses_pre_stretch_compaction(self):
        processor = self._new_processor()
        processor.cfg.stage9_starmask_pre_stretch_compact_enabled = True
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        with patch.object(
            stage9_module.stage9_quality,
            "apply_compact_starmask_support",
        ) as compact:
            prepared = stage9_module._prepare_stage9_starmask_for_pixel_remix(
                processor,
                "starmask",
                star_stretch_used=True,
                messages=[],
                precomputed_calibration={"status": "ok"},
            )

        self.assertEqual(prepared, "starmask_stretched")
        compact.assert_not_called()
        report = processor._stage9_starmask_calibration
        self.assertEqual(report["status"], "plugin_stretched")
        self.assertTrue(report["pre_stretch_compact_enabled"])
        self.assertFalse(report["compact_layer_applied"])

    def test_stage9_unavailable_calibration_does_not_run_fixed_asinh_fallback(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        pixels = np.full((3, 8, 8), 0.01, dtype=np.float32)
        processor.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: pixels.copy(),
            set_image_pixeldata=lambda _output: None,
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        messages = []

        prepared = stage9_module._prepare_stage9_starmask_for_pixel_remix(
            processor,
            "starmask",
            star_stretch_used=False,
            messages=messages,
            precomputed_calibration={
                "status": "rejected",
                "reason": "stage9_starmask_output_target_exceeded",
                "stretch": 2.0,
                "offset": 0.001,
                "_compact_support_mask": np.ones((8, 8), dtype=bool),
            },
        )

        self.assertEqual(prepared, "starmask")
        self.assertFalse(any(call[0] == "asinh" for call in processor.cmd_calls))
        self.assertTrue(processor._stage9_starmask_preparation_failed)
        self.assertIn("unmeasured configured Asinh fallback", " ".join(messages))

    def test_stage9_dynamic_collapse_advisory_still_remixes_after_accepted_stretch(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage7_residual_star_score_max = 0.28
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_starless_noise_gain_max = 2.2
        processor.cfg.stage7_starless_dynamic_range_min_ratio = 0.55
        processor.cfg.stage7_starless_peak_signal_min = 0.006
        processor.cfg.stage9_starmask_stretch_enabled = False
        processor.cfg.stage9_fallback_intensity_levels = (0.75, 0.55, 0.40)
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor._stage7_selected_quality = {
            "status": "poor",
            "issues": [
                "starless_dynamic_range_collapse 0.117<0.550, "
                "peak=0.00551<0.00600"
            ],
            "derived": {
                "residual_star_score": 0.0,
                "halo_residue_score": 0.053,
                "starless_noise_gain": 0.78,
                "starless_dynamic_range_ratio": 0.117,
                "starless_peak_signal": 0.00551,
            },
        }
        processor._stage8_final_source = "stage7_stretched"
        processor._stage8_fallback_used = True
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")
        processor.refresh_stage8_handoff("stage7_stretched")
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.StarunPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage7_effective_halo_threshold = lambda: (
            processor.cfg.stage7_halo_residue_score_max
        )

        stage9_star_remixing(processor)

        self.assertTrue(processor.previous_stage_remix_calls)
        self.assertEqual(
            processor.stage_json_reports["stage9_remix_quality.json"]["mode"],
            "screen",
        )
        self.assertIn(
            "accepted-stretch advisory",
            processor.results[-1][3],
        )

    def test_stage9_bypasses_remix_and_degrades_when_stage7_was_not_accepted(self):
        processor = self._new_processor()
        processor._stage7_stretch_accepted = False
        processor._stage7_stretch_output = None
        processor.stretched_name = "stage7_stretched"
        (processor.process_dir / "stage7_cand_a.fit").write_bytes(b"review-only")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.StarunPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage9_review_safe_source = types.MethodType(
            pipeline_module.StarunPostProcessor._stage9_review_safe_source,
            processor,
        )

        stage9_star_remixing(processor)

        self.assertFalse(processor.previous_stage_remix_calls)
        self.assertNotIn(("load", "stage7_cand_a"), processor.cmd_calls)
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertIn(("autostretch", "-linked"), processor.cmd_calls)
        self.assertTrue(processor._stage9_bypassed_bad_starless)
        self.assertTrue(processor._stage9_stars_required)
        self.assertFalse(processor._stage9_stars_applied)
        self.assertTrue(processor._stage9_output_contains_stars)
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertFalse(report["stars_applied"])
        self.assertTrue(report["output_contains_stars"])
        self.assertEqual(
            report["stars_application_mode"],
            "with_stars_review_fallback",
        )
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIn("stage7_stretch_not_accepted", processor.results[-1][3])

    def test_stage9_review_bypass_never_uses_starless_review_source(self):
        processor = self._new_processor()
        processor._stage7_stretch_accepted = False
        processor._stage7_stretch_output = None
        processor._stage7_review_source = "stage7_cand_rescue_2"
        processor.stretched_name = "stage7_stretched"
        (processor.process_dir / "stage7_cand_a.fit").write_bytes(b"candidate-a")
        (processor.process_dir / "stage7_cand_rescue_2.fit").write_bytes(
            b"quality-ranked"
        )
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.StarunPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage9_review_safe_source = types.MethodType(
            pipeline_module.StarunPostProcessor._stage9_review_safe_source,
            processor,
        )

        stage9_star_remixing(processor)

        self.assertNotIn(("load", "stage7_cand_rescue_2"), processor.cmd_calls)
        self.assertNotIn(("load", "stage7_cand_a"), processor.cmd_calls)
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertTrue(processor._stage9_bypassed_bad_starless)
        self.assertTrue(processor._stage9_output_contains_stars)

    def test_stage9_no_starmask_records_required_stars_not_applied(self):
        processor = self._new_processor()
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "with_stars_review_fallback")
        self.assertTrue(report["stars_required"])
        self.assertFalse(report["stars_applied"])
        self.assertTrue(report["output_contains_stars"])
        self.assertEqual(
            report["stars_application_mode"],
            "with_stars_review_fallback",
        )

    def test_stage9_star_preserve_bypass_records_stars_not_required(self):
        processor = self._new_processor()
        processor._star_preserve_target_bypass = True
        processor._stage8_final_source = "stage8_enhanced"
        for y, x in (
            (12, 16),
            (20, 42),
            (31, 78),
            (45, 55),
            (52, 101),
            (62, 30),
            (70, 88),
            (82, 116),
        ):
            processor.image_pixels[:, y, x] = 0.95
        processor.refresh_stage8_handoff(
            processing_route="star_preserve_secondary_nebulosity",
        )
        stage9_module = sys.modules["stages.stage9_star_remixing"]

        with patch.object(
            stage9_module.spatial_background_lineage,
            "assess_final_spatial_background",
            return_value={
                "schema": "starun.final-spatial-background.v1",
                "status": "ok",
                "accepted": True,
                "support_sha256": "1" * 64,
                "issues": [],
            },
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "stars_not_required", report)
        self.assertNotEqual(report["unscreen_reference"]["status"], "ready")
        self.assertFalse(report["stars_required"])
        self.assertFalse(report["stars_applied"])
        self.assertEqual(
            report["stars_application_mode"],
            "stars_not_required",
        )
        self.assertTrue(report["formal_accepted"])
        self.assertEqual(processor._stage9_final_source, "stage9_remixed")
        self.assertFalse(processor._stage9_output_withheld)
        self.assertEqual(processor.results[-1][1], "skipped")

    def test_stage9_review_candidate_boundaries_and_failure_classes(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.stage9_psf_review_fwhm_ratio_max = 1.65

        formal = self._psf_quality(
            "formal",
            {"all": 1.10, "weak": 1.00, "bright": 1.10},
        )
        formal_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            formal,
            attempt_order=0,
        )
        self.assertFalse(formal_result["eligible"])
        self.assertEqual(formal_result["reasons"], ["formally_accepted"])

        at_ceiling = self._psf_quality(
            "at_ceiling",
            {"all": 1.04, "weak": 1.00, "bright": 1.65},
        )
        ceiling_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            at_ceiling,
            attempt_order=1,
        )
        self.assertTrue(ceiling_result["eligible"])
        self.assertEqual(at_ceiling["structural_failures"], [])
        self.assertEqual(
            at_ceiling["numeric_failures"][0]["group"],
            "bright",
        )

        advisory = self._psf_quality(
            "advisory",
            {"all": 1.04, "weak": 1.00, "bright": 1.36},
        )
        advisory["quality_gates"]["background_lift"] = {
            "status": "advisory",
            "advisory": True,
            "hard_failed": False,
            "value": 0.014,
            "accepted_limit": 0.010,
            "severity_ratio": 1.4,
        }
        advisory_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            advisory,
            attempt_order=2,
        )
        self.assertTrue(advisory_result["eligible"])

        above_ceiling = self._psf_quality(
            "above_ceiling",
            {"all": 1.04, "weak": 1.00, "bright": 1.6501},
        )
        above_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            above_ceiling,
            attempt_order=3,
        )
        self.assertFalse(above_result["eligible"])
        self.assertIn("psf_above_review_upper_limit", above_result["reasons"])

        nonfinite = self._psf_quality(
            "nonfinite",
            {"all": 1.04, "weak": 1.00, "bright": float("nan")},
        )
        nonfinite_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            nonfinite,
            attempt_order=4,
        )
        self.assertFalse(nonfinite_result["eligible"])
        self.assertIn("structural_failure", nonfinite_result["reasons"])

        below_formal = self._psf_quality(
            "below_formal",
            {"all": 1.00, "weak": 0.92, "bright": 1.00},
        )
        below_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            below_formal,
            attempt_order=5,
        )
        self.assertFalse(below_result["eligible"])
        self.assertIn("psf_below_formal_lower_limit", below_result["reasons"])

        two_groups = self._psf_quality(
            "two_groups",
            {"all": 1.15, "weak": 1.00, "bright": 1.36},
        )
        two_group_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            two_groups,
            attempt_order=6,
        )
        self.assertFalse(two_group_result["eligible"])
        self.assertIn("psf_failure_count_not_one", two_group_result["reasons"])

        non_psf_hard = self._psf_quality(
            "non_psf_hard",
            {"all": 1.04, "weak": 1.00, "bright": 1.36},
        )
        non_psf_issue = "background_lift 0.020000>0.010000"
        non_psf_hard["issues"].append(non_psf_issue)
        non_psf_hard["gate_issues"].append(non_psf_issue)
        non_psf_hard["quality_gates"]["background_lift"] = {
            "status": "hard_failed",
            "hard_failed": True,
            "value": 0.020,
            "accepted_limit": 0.010,
            "severity_ratio": 2.0,
        }
        non_psf_result = stage9_module._stage9_review_candidate_eligibility(
            processor,
            non_psf_hard,
            attempt_order=7,
        )
        self.assertFalse(non_psf_result["eligible"])
        self.assertIn(
            "non_psf_numeric_hard_failure",
            non_psf_result["reasons"],
        )

        missing_color = self._psf_quality(
            "missing_color",
            {"all": 1.04, "weak": 1.00, "bright": 1.36},
        )
        missing_color.pop("star_color_validation")
        missing_color_result = (
            stage9_module._stage9_review_candidate_eligibility(
                processor,
                missing_color,
                attempt_order=8,
            )
        )
        self.assertFalse(missing_color_result["eligible"])
        self.assertIn("structural_failure", missing_color_result["reasons"])

        insufficient = self._psf_quality(
            "insufficient",
            {"all": 1.04, "weak": 1.00, "bright": 1.36},
        )
        insufficient["psf_closure"]["groups"]["weak"] = {
            "status": "insufficient",
            "candidate_sample_count": 3,
            "minimum_sample_count": 4,
            "reason": "candidate lost measurable same-star samples",
        }
        insufficient_result = (
            stage9_module._stage9_review_candidate_eligibility(
                processor,
                insufficient,
                attempt_order=9,
            )
        )
        self.assertFalse(insufficient_result["eligible"])
        self.assertIn("structural_failure", insufficient_result["reasons"])

        incomplete_state = self._psf_quality(
            "incomplete_state",
            {"all": 1.04, "weak": 1.00, "bright": 1.36},
        )
        registry = []
        stage9_module._stage9_consider_review_candidate(
            processor,
            incomplete_state,
            attempt_order=10,
            registry=registry,
            messages=[],
        )
        self.assertEqual(registry, [])
        self.assertEqual(
            incomplete_state["review_eligibility"]["reasons"],
            ["candidate_state_checkpoint_incomplete"],
        )
        self.assertEqual(
            incomplete_state["review_eligibility"]["checkpoint_state"][
                "status"
            ],
            "incomplete",
        )

    def test_stage9_selects_star_complete_bounded_unscreen_review_candidate(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.star_intensity = 1.05
        processor.cfg.stage9_psf_review_fwhm_ratio_max = 1.65
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage8_enhanced.fit").write_bytes(
            b"stage8-starless"
        )
        processor.refresh_stage8_handoff()
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor.saved_image_pixels["stage8_enhanced"] = (
            processor.image_pixels.copy()
        )
        processor.saved_image_pixels["starmask"] = np.zeros_like(
            processor.image_pixels
        )

        def assess(_source_stem, *, attempt, formula):
            if attempt.startswith("screen_unscreen_"):
                ratios = {"all": 1.044466, "weak": 1.0, "bright": 1.362770}
                intensity_growth = {
                    "screen_unscreen_primary": 0.00910,
                    "screen_unscreen_fallback_075": 0.00008,
                    "screen_unscreen_fallback_055": 0.000006,
                    "screen_unscreen_fallback_040": 0.000004,
                }.get(attempt, 0.0)
            else:
                ratios = {"all": 1.154701, "weak": 1.080123, "bright": 1.362770}
                intensity_growth = 0.00037
            quality = self._psf_quality(
                attempt,
                ratios,
                bright_growth=intensity_growth,
            )
            quality["formula"] = formula
            processor._stage9_last_star_layer = np.full(
                (3, 4, 4),
                0.30,
                dtype=np.float32,
            )
            processor._stage9_last_star_overlay_mask = np.ones(
                (4, 4),
                dtype=bool,
            )
            processor._stage9_last_weak_overlay_mask = np.ones(
                (4, 4),
                dtype=bool,
            )
            processor._stage9_last_bright_overlay_mask = np.ones(
                (4, 4),
                dtype=bool,
            )
            processor._stage9_starmask_calibration = {"status": "ok"}
            processor._stage9_star_color_post_validation = dict(
                quality["star_color_validation"]
            )
            return quality

        processor._stage9_assess_current_remix = assess
        unscreen_stars = np.full((3, 4, 4), 0.30, dtype=np.float32)
        context = {
            "available": True,
            "report": {"status": "ready", "available": True},
            "original_display": np.zeros((3, 4, 4), dtype=np.float32),
            "starless_display": np.zeros((3, 4, 4), dtype=np.float32),
            "unscreen_stars": unscreen_stars,
            "support_mask": np.ones((4, 4), dtype=bool),
        }

        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                return_value=context,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                return_value={"status": "ok", "support_rgb_mae": 0.01},
            ),
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "best_failed_review_candidate")
        self.assertEqual(report["selection_class"], "review_candidate")
        self.assertEqual(report["selected"]["attempt"], "screen_unscreen_primary")
        self.assertEqual(report["selected"]["intensity"], 1.05)
        self.assertFalse(report["formal_accepted"])
        self.assertTrue(report["review_candidate_selected"])
        self.assertTrue(report["stars_applied"])
        self.assertTrue(report["output_contains_stars"])
        selected_eligibility = report["selected"]["review_eligibility"]
        self.assertTrue(selected_eligibility["selected"])
        self.assertTrue(selected_eligibility["checkpoint_saved"])
        self.assertEqual(
            selected_eligibility["checkpoint_state"]["status"],
            "saved",
        )
        self.assertEqual(selected_eligibility["restore_status"], "restored")
        self.assertEqual(selected_eligibility["final_save_status"], "saved")
        self.assertEqual(
            report["stage9_fallback_reason"],
            "best_failed_candidate_review",
        )
        self.assertEqual(report["fallback_remix"]["status"], "not_attempted")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "best_failed_candidate_review",
        )
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertNotIn(("autostretch", "-linked"), processor.cmd_calls)
        self.assertIn("stage9_review_with_stars", processor.saved_image_pixels)
        self.assertIn("stage9_remixed", processor.saved_image_pixels)

    def test_stage9_review_checkpoint_failure_falls_back_to_stage5(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")

        def assess(_source, *, attempt, formula):
            quality = {
                **self._psf_quality(
                    attempt,
                    {"all": 1.04, "weak": 1.00, "bright": 1.36},
                ),
                "formula": formula,
            }
            processor._stage9_last_star_layer = np.full(
                (3, 4, 4),
                0.30,
                dtype=np.float32,
            )
            processor._stage9_last_star_overlay_mask = np.ones(
                (4, 4),
                dtype=bool,
            )
            processor._stage9_last_weak_overlay_mask = np.ones(
                (4, 4),
                dtype=bool,
            )
            processor._stage9_last_bright_overlay_mask = np.ones(
                (4, 4),
                dtype=bool,
            )
            processor._stage9_starmask_calibration = {"status": "ok"}
            processor._stage9_star_color_post_validation = dict(
                quality["star_color_validation"]
            )
            return quality

        processor._stage9_assess_current_remix = assess
        real_save = processor._save_stage_output
        processor._save_stage_output = lambda stem: (
            False
            if stem.startswith("stage9_review_candidate_")
            else real_save(stem)
        )

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selection_class"], "stage5_fallback")
        self.assertEqual(report["mode"], "stage5_review_fallback")
        self.assertFalse(report["stars_applied"])
        self.assertTrue(report["output_contains_stars"])
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertIn(("autostretch", "-linked"), processor.cmd_calls)
        self.assertTrue(
            all(
                attempt["review_eligibility"]["checkpoint_state"]["status"]
                in {"not_attempted", "save_failed"}
                for attempt in report["attempts"]
            )
        )

    def test_stage9_above_review_ceiling_falls_back_to_stage5(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_psf_review_fwhm_ratio_max = 1.65
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor._stage9_assess_current_remix = (
            lambda _source, *, attempt, formula: {
                **self._psf_quality(
                    attempt,
                    {"all": 1.04, "weak": 1.00, "bright": 1.6501},
                ),
                "formula": formula,
            }
        )

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selection_class"], "stage5_fallback")
        self.assertEqual(report["mode"], "stage5_review_fallback")
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertTrue(
            any(
                "psf_above_review_upper_limit"
                in attempt["review_eligibility"]["reasons"]
                for attempt in report["attempts"]
            )
        )
        self.assertFalse(
            any(
                stem.startswith("stage9_review_candidate_")
                for stem in processor.saved_image_pixels
            )
        )

    def test_stage9_all_rejected_uses_stage8_stretched_starmask_fallback(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage8_enhanced.fit").write_bytes(
            b"stage8-starless"
        )
        processor.refresh_stage8_handoff()
        (processor.process_dir / "starmask_fallback_stretched.fit").write_bytes(
            b"fallback-stretched"
        )
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        stage8_pixels = processor.image_pixels.copy()
        raw_starmask = np.zeros_like(stage8_pixels)
        raw_starmask[:, 42:47, 61:66] = 0.08
        processor.saved_image_pixels["stage8_enhanced"] = stage8_pixels
        processor.saved_image_pixels["starmask"] = raw_starmask
        processor._stage9_assess_current_remix = lambda *_args, **_kwargs: {
            "attempt": "rejected",
            "formula": "screen",
            "status": "rejected",
            "accepted": False,
            "issues": ["mock rejection"],
            "metrics": {},
        }

        stage9_module = sys.modules["stages.stage9_star_remixing"]
        calibration = {
            "status": "ok",
            "support_mode": "normal",
            "stretch": 2.0,
            "offset": 0.001,
            "star_sample_count": 25,
            "compact_component_count": 1,
            "stretch_applied": False,
        }
        with patch.object(
            stage9_module.stage9_quality,
            "calibrate_starmask_asinh",
            return_value=calibration,
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "stage8_starmask_review_fallback")
        self.assertEqual(
            report["selection_class"],
            "stage8_starmask_fallback",
        )
        self.assertTrue(report["stars_required"])
        self.assertTrue(report["stars_applied"])
        self.assertTrue(report["output_contains_stars"])
        self.assertEqual(
            report["stars_application_mode"],
            "screen_minimal_review_fallback",
        )
        self.assertEqual(
            report["fallback_remix"]["selected_variant"],
            "stretch_only",
        )
        self.assertEqual(
            report["fallback_remix"]["base_source_stem"],
            "stage8_enhanced",
        )
        self.assertFalse(report["remix_formally_accepted"])
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertNotIn(("autostretch", "-linked"), processor.cmd_calls)
        self.assertIn("base=stage8_enhanced", processor.results[-1][3])
        self.assertEqual(processor.results[-1][1], "degraded")

    def test_stage9_minimal_fallback_uses_raw_starmask_when_stretch_fails(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage8_enhanced.fit").write_bytes(
            b"stage8-starless"
        )
        processor.refresh_stage8_handoff()
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor.saved_image_pixels["stage8_enhanced"] = (
            processor.image_pixels.copy()
        )
        raw_starmask = np.zeros_like(processor.image_pixels)
        raw_starmask[:, 42:47, 61:66] = 0.08
        processor.saved_image_pixels["starmask"] = raw_starmask
        processor._stage9_assess_current_remix = lambda *_args, **_kwargs: {
            "attempt": "rejected",
            "formula": "screen",
            "status": "rejected",
            "accepted": False,
            "issues": ["mock rejection"],
            "metrics": {},
        }
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        real_prepare = stage9_module._prepare_stage9_starmask_for_pixel_remix

        def prepare(pipeline, starmask_name, **kwargs):
            if kwargs.get("output_name") == "starmask_fallback_stretched":
                pipeline._stage9_starmask_calibration = {
                    "status": "failed",
                    "reason": "mock fallback stretch failure",
                    "stretch_applied": False,
                }
                return starmask_name
            return real_prepare(pipeline, starmask_name, **kwargs)

        with patch.object(
            stage9_module,
            "_prepare_stage9_starmask_for_pixel_remix",
            side_effect=prepare,
        ):
            stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "stage8_starmask_review_fallback")
        self.assertEqual(report["fallback_remix"]["selected_variant"], "raw")
        self.assertTrue(report["stars_applied"])
        self.assertFalse(report["remix_formally_accepted"])
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertTrue(
            any(
                call[:3]
                == (
                    "stage8_enhanced",
                    "starmask",
                    processor.cfg.star_intensity,
                )
                for call in processor.previous_stage_remix_calls
            )
        )

    def test_stage9_minimal_fallback_fatal_pixel_checks(self):
        processor = self._new_processor()
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        domain = {"float_tolerance": 2e-6}
        base = np.full((3, 4, 4), 0.2, dtype=np.float32)
        processor._stage9_last_star_layer = np.full_like(base, 0.05)
        processor._stage9_last_star_overlay_mask = np.ones(
            (4, 4),
            dtype=bool,
        )

        nonfinite = base.copy()
        nonfinite[0, 0, 0] = np.nan
        shape_mismatch = np.full((3, 3, 4), 0.2, dtype=np.float32)
        unchanged = base.copy()
        darkened = base.copy()
        darkened[:, 1, 1] -= 0.01

        for label, candidate, expected_issue in (
            ("nonfinite", nonfinite, "non-finite"),
            ("shape", shape_mismatch, "shape differs"),
            ("unchanged", unchanged, "did not add measurable signal"),
            ("darkened", darkened, "material negative delta"),
        ):
            with self.subTest(label=label):
                report = stage9_module._stage9_minimal_fallback_safety(
                    processor,
                    base,
                    candidate,
                    base_domain=domain,
                    candidate_domain=domain,
                )
                self.assertEqual(report["status"], "failed")
                self.assertTrue(
                    any(
                        expected_issue in issue
                        for issue in report["issues"]
                    )
                )

        processor._stage9_last_star_overlay_mask = None
        processor._stage9_last_star_layer = np.zeros_like(base)
        processor._stage9_last_star_layer[:, 2, 2] = 0.05
        outside_support = base.copy()
        outside_support[:, 0, 0] += 0.01
        outside_report = stage9_module._stage9_minimal_fallback_safety(
            processor,
            base,
            outside_support,
            base_domain=domain,
            candidate_domain=domain,
        )
        self.assertEqual(outside_report["status"], "failed")
        self.assertEqual(
            outside_report["checks"]["supported_positive_delta_pixel_count"],
            0,
        )

        inside_support = base.copy()
        inside_support[:, 2, 2] += 0.01
        inside_report = stage9_module._stage9_minimal_fallback_safety(
            processor,
            base,
            inside_support,
            base_domain=domain,
            candidate_domain=domain,
        )
        self.assertEqual(inside_report["status"], "passed")

    def test_stage9_minimal_fallback_rejects_invalid_starmask_source(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        for label, starmask_pixels, expected_issue in (
            (
                "nonfinite",
                np.full((3, 96, 128), np.nan, dtype=np.float32),
                "pixel domain invalid",
            ),
            (
                "shape",
                np.zeros((3, 95, 128), dtype=np.float32),
                "dimensions are incompatible",
            ),
        ):
            with self.subTest(label=label):
                processor = self._new_processor()
                (processor.process_dir / "stage8_enhanced.fit").write_bytes(
                    b"stage8-starless"
                )
                (processor.process_dir / "starmask.fit").write_bytes(b"starmask")
                processor.saved_image_pixels["stage8_enhanced"] = (
                    processor.image_pixels.copy()
                )
                processor.saved_image_pixels["starmask"] = starmask_pixels
                processor.siril.get_image_pixeldata = (
                    lambda preview=False: processor.image_pixels.copy()
                )
                processor._stage9_remix_base_stem = "stage9_starless_base"
                attempts = []

                selected, _ = (
                    stage9_module._stage9_try_stage8_starmask_review_fallback(
                        processor,
                        [],
                        attempts,
                        trigger_reason="test_invalid_starmask",
                        stage8_source_stem="stage8_enhanced",
                        raw_starmask_stem="starmask",
                        intensity=processor.cfg.star_intensity,
                        allow_stretch=False,
                    )
                )

                self.assertFalse(selected)
                self.assertEqual(
                    processor._stage9_remix_base_stem,
                    "stage9_starless_base",
                )
                self.assertTrue(
                    any(
                        expected_issue in issue
                        for attempt in attempts
                        for issue in attempt.get("issues", [])
                    )
                )

    def test_stage9_minimal_fallback_diagnostic_failure_is_non_blocking(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        (processor.process_dir / "stage8_enhanced.fit").write_bytes(
            b"stage8-starless"
        )
        processor.refresh_stage8_handoff()
        (processor.process_dir / "starmask.fit").write_bytes(b"starmask")
        base = processor.image_pixels.copy()
        starmask = np.zeros_like(base)
        starmask[:, 42:47, 61:66] = 0.08
        processor.saved_image_pixels["stage8_enhanced"] = base
        processor.saved_image_pixels["starmask"] = starmask
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        processor.siril.set_image_pixeldata = lambda image: setattr(
            processor,
            "image_pixels",
            np.array(image, copy=True),
        )
        attempts = []

        with patch.object(
            stage9_module,
            "_assess_stage9_candidate",
            side_effect=RuntimeError("mock formal diagnostic failure"),
        ):
            selected, selected_attempt = (
                stage9_module._stage9_try_stage8_starmask_review_fallback(
                    processor,
                    [],
                    attempts,
                    trigger_reason="test_diagnostic_failure",
                    stage8_source_stem="stage8_enhanced",
                    raw_starmask_stem="starmask",
                    intensity=processor.cfg.star_intensity,
                    allow_stretch=False,
                )
            )

        self.assertTrue(selected)
        self.assertEqual(selected_attempt["fallback_variant"], "raw")
        self.assertEqual(
            selected_attempt["formal_quality"]["status"],
            "unavailable",
        )
        self.assertGreater(
            float(
                np.max(
                    processor.saved_image_pixels["stage9_remixed"] - base
                )
            ),
            0.0,
        )

    def test_stage9_accepted_screen_save_failure_does_not_claim_stars_applied(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage8_enhanced.fit").write_bytes(
            b"stage8-starless"
        )
        processor.refresh_stage8_handoff()
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor.saved_image_pixels["stage8_enhanced"] = (
            processor.image_pixels.copy()
        )
        processor.saved_image_pixels["starmask"] = np.full_like(
            processor.image_pixels,
            0.02,
        )
        processor._save_stage_output = lambda stem: stem != "stage9_remixed"

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "with_stars_review_fallback")
        self.assertTrue(report["stars_required"])
        self.assertFalse(report["stars_applied"])
        self.assertTrue(report["output_contains_stars"])
        self.assertEqual(
            report["stars_application_mode"],
            "with_stars_review_fallback",
        )
        self.assertEqual(report["fallback_remix"]["status"], "failed")
        self.assertIn(("load", "stage5_linear"), processor.cmd_calls)
        self.assertEqual(processor.results[-1][1], "degraded")

    def test_stage9_withholds_required_stars_output_when_no_safe_source_exists(self):
        processor = self._new_processor()

        stage9_star_remixing(processor)

        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "required_stars_output_withheld")
        self.assertFalse(report["output_contains_stars"])
        self.assertTrue(report["output_withheld"])
        self.assertEqual(
            report["stars_application_mode"],
            "withheld_no_with_stars_review_source",
        )
        self.assertEqual(processor.results[-1][1], "failed")
