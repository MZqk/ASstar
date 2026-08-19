"""Focused Stage 9 source-presence routing regressions."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class Stage9SourcePresenceRoutingTests(PipelinePluginFallbackTestBase):
    def test_frozen_evidence_does_not_require_transient_trusted_array(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        processor._stage9_star_reference_catalog = {"status": "ok"}
        frozen_evidence = np.full((3, 7, 7), 0.40, dtype=np.float32)
        processor._stage9_immutable_trusted_starmask_peak = frozen_evidence
        processor._stage5_star_reference_report = {
            "stars": [{"x": 4.0, "y": 2.0, "fwhm_geometry": 9.0}]
        }
        base_stars = np.full((3, 7, 7), 0.10, dtype=np.float32)
        base_support = np.zeros((7, 7), dtype=bool)
        base_support[3, 3] = True
        completion_support = np.zeros((7, 7), dtype=bool)
        completion_support[2, 4] = True
        context = {
            "available": True,
            "original_display": np.full((3, 7, 7), 0.50, dtype=np.float32),
            "starless_display": np.full((3, 7, 7), 0.05, dtype=np.float32),
            "unscreen_stars": base_stars,
            "support_mask": base_support,
        }

        def build_completion(
            _stage5_stars,
            _catalog,
            _original,
            trusted_stars,
            _cfg,
            *,
            coordinate_domain,
        ):
            np.testing.assert_array_equal(trusted_stars, frozen_evidence)
            self.assertEqual(
                coordinate_domain,
                "siril_pixel_buffer_bottom_up",
            )
            return {
                "schema": "starun.stage9-stage5-bright-star-completion.v2",
                "status": "ready",
            }

        completed = base_stars.copy()
        completed[:, 2, 4] = 0.35
        completion_report = {
            "schema": "starun.stage9-stage5-bright-star-completion.v2",
            "status": "ready",
            "coordinate_contract": {
                "array_coordinate_domain": "siril_pixel_buffer_bottom_up",
                "conversion": "y_array = y_siril",
                "validated": True,
            },
        }
        with (
            patch.object(
                stage9_module.stage9_quality,
                "build_star_overlay_masks",
                return_value=(base_support, base_support, base_support),
            ),
            patch.object(
                stage9_module.stage9_quality,
                "build_source_wing_feather_candidate",
                return_value=(None, None, {"status": "not_needed"}),
            ),
            patch.object(
                stage9_module.stage9_quality,
                "build_stage5_bright_star_completion",
                side_effect=build_completion,
            ),
            patch.object(
                stage9_module.stage9_quality,
                "apply_stage5_bright_star_completion",
                return_value=(completed, completion_support, completion_report),
            ),
        ):
            prepared = stage9_module._prepare_stage9_source_presence_candidate(
                processor,
                context,
                [],
            )

        report = prepared["source_presence_report"]
        self.assertTrue(report["changed"])
        self.assertEqual(
            report["stage5_bright_star_completion"]["schema"],
            "starun.stage9-stage5-bright-star-completion.v2",
        )
        self.assertTrue(prepared["support_mask"][2, 4])
        self.assertAlmostEqual(float(prepared["unscreen_stars"][0, 2, 4]), 0.35)

    def test_targeted_unscreen_winner_runs_source_presence_extension(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        shape = (3, 4, 4)
        state = {"star_layer": np.full(shape, 0.20, dtype=np.float32)}
        screen_quality = {
            "attempt": "screen_primary",
            "accepted": False,
            "intensity": 1.0,
        }
        unscreen_quality = {
            "attempt": "screen_unscreen_normal_primary",
            "formula": "screen",
            "status": "ok",
            "accepted": True,
            "issues": [],
            "metrics": {
                "weak_star_recovery_ratio": 1.0,
                "star_recovery_ratio": 1.0,
                "star_positive_delta_window_recovery_ratio": 1.0,
                "star_wing_recovery_ratio": 1.0,
                "chromatic_star_addition_ratio": 0.0,
                "highlight_clip_growth": 0.0,
                "bright_pixel_growth": 0.0,
            },
            "psf_closure": {
                "groups": {
                    "all": {
                        "status": "ok",
                        "fwhm_ratio_median": 1.0,
                    }
                }
            },
        }
        context = {
            "available": True,
            "report": {
                "schema": "starun.stage9-unscreen-reference.v1",
                "status": "ready",
                "available": True,
            },
            "starmask": "starmask_unscreen_normal",
            "unscreen_stars": np.full(shape, 0.25, dtype=np.float32),
        }
        extension_parents = []

        def extend(pipeline, **kwargs):
            parent = dict(kwargs["accepted_quality"])
            extension_parents.append(parent["attempt"])
            report = {
                "schema": "starun.stage9-source-presence.v1",
                "status": "ready",
                "available": True,
                "changed": True,
            }
            extended = {
                **parent,
                "attempt": "screen_unscreen_source_presence_95",
                "starmask": "starmask_unscreen_normal",
                "source_presence": report,
            }
            kwargs["remix_attempts"].append(dict(extended))
            pipeline._stage9_source_presence_report = report
            return extended, {
                **kwargs["accepted_context"],
                "source_presence_report": report,
            }

        with (
            patch.object(
                stage9_module,
                "_activate_stage9_candidate_state",
            ),
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                return_value=context,
            ),
            patch.object(
                stage9_module,
                "_assess_stage9_candidate",
                return_value=unscreen_quality,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                return_value={"status": "ok", "support_rgb_mae": 0.0},
            ),
            patch.object(
                stage9_module,
                "_stage9_consider_review_candidate",
            ),
            patch.object(
                stage9_module,
                "_stage9_targeted_soft_psf_recovery",
                side_effect=lambda _pipeline, **kwargs: (
                    kwargs["parent_quality"],
                    kwargs["parent_context"],
                ),
            ),
            patch.object(
                stage9_module,
                "_stage9_targeted_local_chroma_recovery",
                side_effect=lambda _pipeline, **kwargs: (
                    kwargs["parent_quality"],
                    kwargs["parent_context"],
                ),
            ),
            patch.object(
                stage9_module,
                "_capture_stage9_candidate_state",
                return_value=state,
            ),
            patch.object(
                stage9_module,
                "_restore_stage9_candidate_state",
            ),
            patch.object(
                stage9_module,
                "_stage9_extend_rescue_with_source_presence",
                side_effect=extend,
            ),
            patch.object(
                processor,
                "_apply_previous_stage_star_remix",
                return_value=True,
            ),
            patch.object(processor, "_save_stage_output", return_value=True),
        ):
            selected, starmask = (
                stage9_module._stage9_targeted_unscreen_competition(
                    processor,
                    source_stem="stage8_enhanced",
                    primary_support_results=[
                        {
                            "support_mode": "normal",
                            "starmask": "starmask_stretched",
                            "quality": screen_quality,
                            "state": state,
                        }
                    ],
                    selected_screen=None,
                    messages=[],
                    remix_attempts=[],
                    review_candidate_registry=[],
                )
            )

        self.assertEqual(
            extension_parents,
            ["screen_unscreen_normal_primary"],
        )
        self.assertEqual(selected["attempt"], "screen_unscreen_source_presence_95")
        self.assertEqual(starmask, "starmask_unscreen_normal")

