"""Stage 9 formal candidate ordering and persisted-output acceptance tests."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403

import run_manifest


class PipelinePluginFallbackStage9AcceptanceTests(PipelinePluginFallbackTestBase):
    def test_formal_ranking_ignores_lower_chromatic_addition(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        base = {
            "accepted": True,
            "metrics": {
                "catalog_star_visibility_ratio_all": 0.95,
                "catalog_star_visibility_ratio_weak": 0.92,
                "catalog_star_visibility_ratio_bright": 1.0,
                "weak_star_recovery_ratio": 0.90,
                "star_recovery_ratio": 0.92,
                "star_positive_delta_window_recovery_ratio": 0.91,
                "star_wing_recovery_ratio": 0.80,
                "highlight_clip_growth": 0.001,
                "bright_pixel_growth": 0.002,
            },
            "psf_closure": {
                "groups": {"all": {"status": "ok", "fwhm_ratio_median": 1.0}}
            },
            "reference_fidelity": {"support_rgb_mae": 0.01},
        }
        low_chroma = copy.deepcopy(base)
        low_chroma["metrics"]["chromatic_star_addition_ratio"] = 0.00001
        high_chroma = copy.deepcopy(base)
        high_chroma["metrics"]["chromatic_star_addition_ratio"] = 0.002

        low_score = stage9_module._stage9_formal_candidate_score(
            low_chroma, support_mode="normal"
        )
        high_score = stage9_module._stage9_formal_candidate_score(
            high_chroma, support_mode="normal"
        )
        self.assertEqual(low_score, high_score)

        lower_visibility = copy.deepcopy(base)
        lower_visibility["metrics"]["catalog_star_visibility_ratio_weak"] = 0.75
        self.assertLess(
            low_score,
            stage9_module._stage9_formal_candidate_score(
                lower_visibility, support_mode="normal"
            ),
        )

    def test_formal_ranking_prefers_candidate_inside_presentation_psf_target(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        base = {
            "accepted": True,
            "metrics": {
                "catalog_star_visibility_ratio_all": 1.0,
                "catalog_star_visibility_ratio_weak": 1.0,
                "catalog_star_visibility_ratio_bright": 1.0,
                "weak_star_recovery_ratio": 1.0,
                "star_recovery_ratio": 1.0,
                "star_positive_delta_window_recovery_ratio": 1.0,
                "star_wing_recovery_ratio": 0.95,
                "highlight_clip_growth": 0.0,
                "bright_pixel_growth": 0.0,
            },
            "reference_fidelity": {"support_rgb_mae": 0.01},
        }

        closer_but_outside = copy.deepcopy(base)
        closer_but_outside["psf_closure"] = {
            "groups": {
                "all": {"status": "ok", "fwhm_ratio_median": 1.0},
                "weak": {"status": "ok", "fwhm_ratio_median": 1.0},
                "bright": {"status": "ok", "fwhm_ratio_median": 0.966},
            }
        }
        inside_target = copy.deepcopy(base)
        inside_target["psf_closure"] = {
            "groups": {
                "all": {"status": "ok", "fwhm_ratio_median": 1.0},
                "weak": {"status": "ok", "fwhm_ratio_median": 1.0},
                "bright": {"status": "ok", "fwhm_ratio_median": 1.049},
            }
        }

        self.assertLess(
            stage9_module._stage9_formal_candidate_score(
                inside_target,
                support_mode="normal",
            ),
            stage9_module._stage9_formal_candidate_score(
                closer_but_outside,
                support_mode="normal",
            ),
        )

    def test_persisted_validation_rejects_frame_mismatch(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        active = np.zeros((3, 12, 16), dtype=np.float32)
        active[:, 5, 7] = 0.5
        persisted = active.copy()
        persisted[:, 2, 3] = 0.1
        state = {"pixels": active}

        class PersistedPipeline:
            cfg = SimpleNamespace(stage9_star_color_post_validation_enabled=False)
            siril = SimpleNamespace(
                get_image_pixeldata=lambda preview=False: state["pixels"].copy()
            )
            _stage9_star_color_reference_samples = None
            _stage9_last_star_layer = None

            @staticmethod
            def cmd_with_check(command, stem):
                self.assertEqual(command, "load")
                if stem == "stage9_remixed":
                    state["pixels"] = persisted

            @staticmethod
            def _stage9_assess_current_remix(_source, *, attempt, formula):
                return {
                    "attempt": attempt,
                    "formula": formula,
                    "status": "ok",
                    "accepted": True,
                    "catalog_visibility": {
                        "available": True,
                        "passed": True,
                        "coordinate_contract": {"validated": True},
                        "groups": {
                            name: {"passed": True}
                            for name in ("all", "weak", "bright")
                        },
                    },
                }

        report = stage9_module._validate_stage9_persisted_output(
            PersistedPipeline(),
            "stage8_enhanced",
            {"attempt": "screen_primary", "formula": "screen"},
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["pixels_match"])
        self.assertIn("persisted_frame_pixels_mismatch", report["failures"])
        self.assertNotEqual(
            report["active_pixel_hash"], report["persisted_pixel_hash"]
        )

    def test_persisted_validation_requires_sep_artifact_and_restores_c(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        candidate = np.zeros((3, 12, 16), dtype=np.float32)
        candidate[:, 5, 7] = 0.5
        base = np.zeros_like(candidate)
        original = candidate.copy()
        state = {"stem": "active"}

        with tempfile.TemporaryDirectory() as temp_dir:
            process_dir = Path(temp_dir)

            class PersistedPipeline:
                cfg = SimpleNamespace(
                    stage9_star_color_post_validation_enabled=False
                )
                siril = SimpleNamespace(
                    get_image_pixeldata=lambda preview=False: {
                        "active": candidate,
                        "stage9_remixed": candidate,
                        "stage8_enhanced": base,
                    }[state["stem"]].copy()
                )
                _stage9_star_color_reference_samples = None
                _stage9_last_star_layer = None
                _stage9_remix_base_stem = "stage8_enhanced"
                _stage9_spatial_scale = {
                    "status": "ready",
                    "fwhm_median_px": 4.0,
                }
                _stage9_matched_domain_context = {
                    "available": True,
                    "original_display": original,
                }

                @staticmethod
                def cmd_with_check(command, stem):
                    self.assertEqual(command, "load")
                    state["stem"] = stem

                @staticmethod
                def _stage9_assess_current_remix(_source, *, attempt, formula):
                    return {
                        "attempt": attempt,
                        "formula": formula,
                        "status": "ok",
                        "accepted": True,
                        "catalog_visibility": {
                            "available": True,
                            "coordinate_contract": {"validated": True},
                            "groups": {
                                name: {"passed": True}
                                for name in ("all", "weak", "bright")
                            },
                        },
                    }

                @staticmethod
                def _write_stage_json(name, payload):
                    run_manifest.atomic_write_json(process_dir / name, payload)

            sep_evidence = {
                "schema": "starun.stage9-sep-crossmatch.v1",
                "status": "ok",
                "accepted": True,
                "reason_code": "stage9_sep_crossmatch_accepted",
                "catalogs": {},
                "matches": {},
                "formal_set": {},
                "report_sha256": "1" * 64,
            }
            with patch.object(
                stage9_module.stage9_quality,
                "assess_independent_sep_crossmatch",
                return_value=sep_evidence,
            ):
                persisted_pipeline = PersistedPipeline()
                persisted_pipeline.process_dir = process_dir
                report = stage9_module._validate_stage9_persisted_output(
                    persisted_pipeline,
                    "stage8_enhanced",
                    {"attempt": "screen_primary", "formula": "screen"},
                )

            self.assertTrue(report["accepted"], report)
            self.assertTrue(report["sep_crossmatch_accepted"])
            self.assertTrue(report["restored_after_sep"])
            self.assertEqual(state["stem"], "stage9_remixed")
            self.assertTrue(
                (process_dir / "stage9_sep_crossmatch.json").is_file()
            )
            self.assertEqual(
                len(report["sep_crossmatch"]["artifact_sha256"]),
                64,
            )

    def test_sep_recovery_formal_closure_rejects_soft_psf_and_identity_drift(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        parent = {
            "metrics": {"star_wing_recovery_ratio": 0.90},
            "psf_closure": {
                "groups": {
                    name: {"status": "ok", "fwhm_ratio_median": 1.02}
                    for name in ("all", "weak", "bright")
                }
            },
        }
        candidate = copy.deepcopy(parent)
        candidate["metrics"]["star_wing_recovery_ratio"] = 0.89
        candidate["psf_closure"]["groups"]["weak"][
            "fwhm_ratio_median"
        ] = 1.06
        candidate_hash = "a" * 64
        retry = {
            "accepted": True,
            "shape_matches": True,
            "pixels_match": True,
            "restored_after_sep": True,
            "active_pixel_hash": candidate_hash,
            "persisted_pixel_hash": candidate_hash,
            "restored_pixel_hash": "b" * 64,
            "coordinate_contract": {"validated": True},
            "reloaded_quality": candidate,
        }

        report = (
            stage9_module._stage9_validate_persisted_sep_recovery_closure(
                parent,
                retry,
                candidate_pixel_sha256=candidate_hash,
            )
        )

        self.assertFalse(report["accepted"])
        self.assertEqual(
            report["failures"],
            [
                "psf_soft_target_unclosed",
                "parent_zero_tolerance_nonregression_failed",
                "persisted_pixel_identity_failed",
            ],
        )
        self.assertFalse(report["soft_psf_closure"]["accepted"])
        self.assertFalse(
            report["parent_zero_tolerance_nonregression"]["accepted"]
        )
        self.assertFalse(report["identity"]["accepted"])

    def test_persisted_sep_recovery_soft_psf_failure_rolls_back_exact_parent(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        parent = np.zeros((3, 12, 16), dtype=np.float32)
        parent[:, 5, 7] = 0.50
        candidate = parent.copy()
        candidate[:, 6, 8] = 0.25
        base = np.zeros_like(parent)
        original = parent.copy()
        state = {
            "stem": "active",
            "current": parent.copy(),
            "files": {
                "active": parent.copy(),
                "stage9_remixed": parent.copy(),
                "stage8_enhanced": base.copy(),
            },
        }

        def quality_for(pixels, *, attempt, formula):
            recovered = np.array_equal(pixels, candidate)
            ratio = 1.06 if recovered else 1.02
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok",
                "accepted": True,
                "issues": [],
                "metrics": {"star_wing_recovery_ratio": 0.90},
                "psf_closure": {
                    "status": "ok",
                    "limits": {
                        "stage9_psf_fwhm_ratio_min": 0.93,
                        "stage9_psf_fwhm_ratio_max": 1.10,
                    },
                    "groups": {
                        name: {
                            "status": "ok",
                            "fwhm_ratio_median": ratio,
                        }
                        for name in ("all", "weak", "bright")
                    },
                },
                "catalog_visibility": {
                    "available": True,
                    "coordinate_contract": {"validated": True},
                    "groups": {
                        name: {"passed": True}
                        for name in ("all", "weak", "bright")
                    },
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            process_dir = Path(temp_dir)
            (process_dir / "stage9_remixed.fit").write_bytes(b"fixture")

            class PersistedPipeline:
                cfg = SimpleNamespace(
                    stage9_star_color_post_validation_enabled=False
                )
                _stage9_star_color_reference_samples = None
                _stage9_last_star_layer = None
                _stage9_remix_base_stem = "stage8_enhanced"
                _stage9_spatial_scale = {
                    "status": "ready",
                    "fwhm_median_px": 4.0,
                }
                _stage9_matched_domain_context = {
                    "available": True,
                    "original_display": original,
                }
                siril = SimpleNamespace(
                    get_image_pixeldata=lambda preview=False: np.array(
                        state["current"], copy=True
                    )
                )

                @staticmethod
                def cmd_with_check(command, stem):
                    self.assertEqual(command, "load")
                    state["stem"] = stem
                    state["current"] = np.array(
                        state["files"][stem], copy=True
                    )

                @staticmethod
                def _set_current_image_pixeldata(pixels, *, label):
                    state["current"] = np.array(pixels, copy=True)

                @staticmethod
                def _save_stage_output(stem):
                    state["files"][stem] = np.array(
                        state["current"], copy=True
                    )
                    return True

                @staticmethod
                def _stage9_assess_current_remix(
                    _source, *, attempt, formula
                ):
                    return quality_for(
                        state["current"], attempt=attempt, formula=formula
                    )

            def sep_assessment(_o, _b, persisted, *_args, **_kwargs):
                recovered = np.array_equal(persisted, candidate)
                return {
                    "schema": "starun.stage9-sep-crossmatch.v1",
                    "status": "ok" if recovered else "rejected",
                    "accepted": recovered,
                    "failed_gates": (
                        [] if recovered else ["source_recovery_ratio"]
                    ),
                }

            def sep_summary(_pipeline, evidence):
                return {
                    "schema": "starun.stage9-sep-crossmatch.v1",
                    "accepted": bool(evidence.get("accepted", False)),
                }

            with (
                patch.object(
                    stage9_module.stage9_quality,
                    "assess_independent_sep_crossmatch",
                    side_effect=sep_assessment,
                ),
                patch.object(
                    stage9_module,
                    "_stage9_bind_sep_o_source_evidence",
                    return_value={"accepted": True},
                ),
                patch.object(
                    stage9_module,
                    "_persist_stage9_sep_crossmatch_evidence",
                    side_effect=sep_summary,
                ),
                patch.object(
                    stage9_module,
                    "_stage9_build_same_source_sep_recovery",
                    return_value=(
                        candidate.copy(),
                        {"status": "ready", "accepted": True},
                    ),
                ),
            ):
                persisted_pipeline = PersistedPipeline()
                persisted_pipeline.process_dir = process_dir
                report = stage9_module._validate_stage9_persisted_output(
                    persisted_pipeline,
                    "stage8_enhanced",
                    {"attempt": "screen_primary", "formula": "screen"},
                )

        self.assertFalse(report["accepted"], report)
        recovery = report["same_source_sep_recovery"]
        self.assertFalse(recovery["formal_closure"]["accepted"])
        self.assertIn(
            "psf_soft_target_unclosed",
            recovery["formal_closure"]["failures"],
        )
        self.assertEqual(recovery["rollback"]["status"], "restored")
        self.assertEqual(
            recovery["rollback"]["pixel_sha256"],
            stage9_module._stage9_pixel_hash(parent),
        )
        np.testing.assert_array_equal(
            state["files"]["stage9_remixed"], parent
        )

    def test_stage9_preserve_review_stops_after_first_rejected_candidate(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_failure_action = "preserve_review"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor._stage9_assess_current_remix = lambda *_args, **kwargs: {
            **self._psf_quality(
                kwargs["attempt"],
                {"all": 1.04, "weak": 1.00, "bright": 1.36},
            ),
            "formula": kwargs["formula"],
        }

        stage9_star_remixing(processor)

        self.assertEqual(len(processor.previous_stage_remix_calls), 1)
        self.assertTrue(processor._stage_review_reasons(9))
        self.assertEqual(processor._stage_review_reasons(3), [])
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertFalse(
            any(
                stem.startswith("stage9_review_candidate_")
                for stem in processor.saved_image_pixels
            )
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(
            report["attempts"][0]["review_eligibility"]["reasons"],
            ["review_policy_disabled_for_failure_action"],
        )

    def test_stage9_preserve_mode_requires_and_uses_verified_with_stars_source(self):
        processor = self._new_processor()
        processor.cfg.stage9_processing_mode = "preserve_with_stars"
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")

        stage9_star_remixing(processor)

        self.assertEqual(processor.previous_stage_remix_calls, [])
        self.assertTrue(processor._stage9_output_contains_stars)
        self.assertFalse(processor._stage9_output_withheld)
        self.assertEqual(
            processor._stage_review_reasons(9),
            ["user_preserve_with_stars"],
        )
        self.assertEqual(processor._stage_review_reasons(3), [])
        self.assertEqual(processor.results[-1][1], "ok")
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["mode"], "user_preserve_with_stars")

    def test_stage9_does_not_lower_screen_intensity_after_recovery_shortfall(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"with-stars")
        processor._stage9_assess_current_remix = lambda *_args, **kwargs: {
            "attempt": kwargs["attempt"],
            "formula": kwargs["formula"],
            "status": "rejected",
            "accepted": False,
            "issues": ["weak_star_recovery_ratio 0.420000<0.700000"],
            "metrics": {
                "weak_star_recovery_ratio": 0.42,
                "star_recovery_ratio": 0.50,
            },
        }

        stage9_star_remixing(processor)

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
        self.assertEqual(report["mode"], "stage5_review_fallback")
        self.assertFalse(report["stars_applied"])
        self.assertTrue(report["output_contains_stars"])
        self.assertEqual(processor.results[-1][1], "degraded")
