"""Pipeline/plugin fallback tests for stage3 background."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage3BackgroundTests(PipelinePluginFallbackTestBase):
    def test_background_chroma_noise_ignores_smooth_colour_bias(self):
        processor = pipeline_module.StarunPostProcessor()
        smooth_red = np.full((3, 64, 64), 0.04, dtype=np.float32)
        smooth_red[0] = 0.16

        metrics = processor._background_quality_metrics(smooth_red)

        self.assertLess(metrics["chroma_noise_score"], 0.05)
        self.assertGreater(metrics["background_chroma_load"], 0.50)

    def test_background_chroma_noise_detects_high_frequency_colour_variation(self):
        processor = pipeline_module.StarunPostProcessor()
        yy, xx = np.mgrid[:64, :64]
        checker = ((xx + yy) % 2).astype(np.float32)
        noisy = np.full((3, 64, 64), 0.08, dtype=np.float32)
        noisy[0] += checker * 0.10
        noisy[2] += (1.0 - checker) * 0.10

        metrics = processor._background_quality_metrics(noisy)

        self.assertGreater(metrics["chroma_noise_score"], 0.34)

    def test_pipeline_status_uncalibrated_background_cast_is_review_required(self):
        probe = pipeline_module.StarunPostProcessor()
        probe.results = []
        probe._require_review(
            7,
            "uncalibrated_background_color_review_required",
        )

        status = probe._pipeline_result_status()

        self.assertEqual(status, "review_required")

    def test_plugin_fingerprint_uses_preview_and_bounded_sample(self):
        preview_calls: list[bool] = []
        image = np.zeros((3, 512, 512), dtype=np.float32)
        processor = SimpleNamespace(
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: (
                    preview_calls.append(preview) or image
                )
            ),
            log=FakeLogger(),
        )

        before = pipeline_module.plugin_runner.current_image_fingerprint(processor)
        image[:] = 1.0
        after = pipeline_module.plugin_runner.current_image_fingerprint(processor)

        self.assertEqual(preview_calls, [True, True])
        self.assertNotEqual(before, after)

    def test_stage3_plugin_order_uses_theoretical_effect_chain(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()

        high_noise = pipeline_module.ImageFeatures(bg_std=0.060, star_density=0.001)
        high_noise_order = processor._stage3_plugin_candidates(
            high_noise,
            {"dirty_background_score": 0.42},
        )
        self.assertEqual(
            [label for label, _cmd, _source in high_noise_order],
            ["GraXpert", "ADBE", "DBE", "AutoDBE"],
        )

    def test_stage3_legacy_ratio_gate_is_diagnostic_only(self):
        processor = pipeline_module.StarunPostProcessor()
        preservation = {
            "available": True,
            "star_retention_ratio": 0.82,
            "nebula_mean_change_ratio": 0.14,
            "before_star_count": 100,
            "after_star_count": 82,
        }
        gate_ok, gate_msg = processor._stage3_quality_gate(
            pipeline_module.ImageFeatures(bg_std=0.02, bg_median=0.08, object_area_ratio=0.20),
            pipeline_module.ImageFeatures(bg_std=0.02, bg_median=0.08, object_area_ratio=0.20),
            preservation,
        )

        self.assertTrue(gate_ok)
        self.assertIn("held-out sky validation owns acceptance", gate_msg)
        self.assertIn("star_retention=0.820", gate_msg)

    def test_stage3_background_score_prefers_cleaner_low_gradient_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        before = {
            "bg_std": 0.00010,
            "gradient_score": 0.10,
            "dirty_background_score": 0.42,
            "red_dominance": 1.00,
            "blue_dominance": 1.00,
            "green_cast": 1.00,
        }
        dirty_candidate = {
            "bg_std": 0.00011,
            "gradient_score": 0.09,
            "dirty_background_score": 0.38,
            "chroma_noise_score": 0.12,
            "red_dominance": 1.32,
            "blue_dominance": 0.92,
            "green_cast": 1.18,
        }
        cleaner_candidate = {
            "bg_std": 0.00010,
            "gradient_score": 0.04,
            "dirty_background_score": 0.20,
            "chroma_noise_score": 0.05,
            "red_dominance": 1.02,
            "blue_dominance": 1.01,
            "green_cast": 0.99,
        }

        dirty_score = stage3_module._stage3_background_score(before, dirty_candidate)
        cleaner_score = stage3_module._stage3_background_score(before, cleaner_candidate)

        self.assertLess(cleaner_score, dirty_score)
        self.assertFalse(stage3_module._stage3_candidate_sufficient(before, dirty_candidate, dirty_score))
        self.assertTrue(stage3_module._stage3_candidate_sufficient(before, cleaner_candidate, cleaner_score))

    def test_stage3_background_score_archives_components_and_weights(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        before = {
            "bg_std": 0.01,
            "red_dominance": 1.0,
            "blue_dominance": 1.0,
            "green_cast": 1.0,
        }
        after = {
            "bg_std": 0.011,
            "dirty_background_score": 0.20,
            "gradient_score": 0.04,
            "chroma_noise_score": 0.03,
            "red_dominance": 1.02,
            "blue_dominance": 0.99,
            "green_cast": 1.01,
        }

        report = stage3_module._stage3_background_score_components(before, after)

        self.assertAlmostEqual(
            report["total"],
            stage3_module._stage3_background_score(before, after),
        )
        self.assertEqual(
            set(report["components"]),
            {
                "dirty_background_score",
                "gradient_score",
                "chroma_noise_score",
                "bg_std_growth",
                "color_shift",
            },
        )
        self.assertAlmostEqual(
            report["total"],
            sum(report["weighted_components"].values()),
        )

    def test_stage3_statistical_selection_is_runtime_authoritative(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        current = {
            "label": "runtime-score-winner",
            "source": "builtin",
            "score": 0.10,
            "preservation_penalty": 0.0,
            "directional_pattern_penalty": 0.0,
            "validation": {
                "status": "ready",
                "robust_span": 0.08,
                "patch_median_uncertainty": 0.01,
            },
            "validation_gate": {
                "accepted": True,
                "span_improvement": 0.02,
                "sampling_uncertainty_3sigma": 0.01,
            },
        }
        statistical = {
            "label": "statistical-shadow-winner",
            "source": "builtin",
            "score": 0.16,
            "preservation_penalty": 0.05,
            "directional_pattern_penalty": 0.0,
            "validation": {
                "status": "ready",
                "robust_span": 0.02,
                "patch_median_uncertainty": 0.01,
            },
            "validation_gate": {
                "accepted": True,
                "span_improvement": 0.08,
                "sampling_uncertainty_3sigma": 0.01,
            },
        }

        report = stage3_module._stage3_statistical_shadow_selection(
            [current, statistical],
            current,
        )

        self.assertEqual(report["current_runtime_candidate"], current["label"])
        self.assertEqual(
            report["shadow_recommended_candidate"],
            statistical["label"],
        )
        self.assertTrue(report["selection_would_change"])
        self.assertTrue(report["runtime_selection_affected"])

    def test_stage3_output_first_flux_penalty_is_symmetric(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        loss = stage3_module._stage3_preservation_penalty(
            {"available": True, "target_flux_retention_ratio": 0.5},
            gate_profile="output_first",
        )
        growth = stage3_module._stage3_preservation_penalty(
            {"available": True, "target_flux_retention_ratio": 1.5},
            gate_profile="output_first",
        )

        self.assertAlmostEqual(loss, growth)

    def test_stage3_batch_regression_m8a_selects_best_soft_warning_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        candidates = [
            {
                "label": "M8-A subsky-rbf-existing-1",
                "source": "builtin",
                "score": 0.30,
                "sufficient": False,
                "severity": "soft_warning",
                "gate_warnings": ["improvement below sampling uncertainty"],
                "validation": {
                    "status": "ready",
                    "robust_span": 0.000008489204242,
                    "patch_median_uncertainty": 0.000001644499422,
                },
                "validation_gate": {"accepted": True},
                "preservation": {
                    "target_flux_retention_ratio": 0.9969786552,
                    "target_morphology_correlation": 0.9999974168,
                    "target_centroid_shift_fraction": 0.0001479180,
                    "target_change_residual_significance": 0.0475881849,
                },
            },
            {
                "label": "M8-A DBE",
                "source": "plugin",
                "score": 0.28,
                "sufficient": False,
                "severity": "soft_warning",
                "gate_warnings": ["improvement below sampling uncertainty"],
                "validation": {
                    "status": "ready",
                    "robust_span": 0.000010031891356,
                    "patch_median_uncertainty": 0.000001644499423,
                },
                "validation_gate": {"accepted": True},
                "preservation": {
                    "target_flux_retention_ratio": 1.1922317502,
                    "target_morphology_correlation": 0.9996902601,
                    "target_centroid_shift_fraction": 0.0007474976,
                    "target_change_residual_significance": 8.4440638041,
                },
            },
        ]

        report = stage3_module._stage3_statistical_shadow_selection(
            candidates,
            candidates[1],
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(
            report["shadow_recommended_candidate"],
            "M8-A subsky-rbf-existing-1",
        )
        self.assertTrue(report["selection_would_change"])

    def test_stage3_pixel_integrity_gate_hard_rejects_invalid_outputs(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        baseline = np.ones((8, 10), dtype=np.float32)

        for candidate, expected_issue in (
            (baseline.copy(), "did not change"),
            (np.ones((7, 10), dtype=np.float32), "dimensions changed"),
            (np.full((8, 10), np.nan, dtype=np.float32), "non-finite"),
        ):
            with self.subTest(expected_issue=expected_issue):
                accepted, gate = stage3_module._stage3_candidate_pixel_gate(
                    baseline,
                    candidate,
                )
                self.assertFalse(accepted)
                self.assertEqual(gate["severity"], "hard_rejected")
                self.assertTrue(
                    any(expected_issue in issue for issue in gate["hard_issues"])
                )

    def test_stage3_candidate_sufficient_uses_policy_std_growth_limit(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        before = {
            "bg_std": 0.00010,
            "gradient_score": 0.08,
            "dirty_background_score": 0.28,
            "red_dominance": 1.00,
            "blue_dominance": 1.00,
            "green_cast": 1.00,
        }
        candidate = {
            "bg_std": 0.000107,
            "gradient_score": 0.02,
            "dirty_background_score": 0.12,
            "chroma_noise_score": 0.02,
            "red_dominance": 1.01,
            "blue_dominance": 1.00,
            "green_cast": 0.99,
        }

        self.assertTrue(stage3_module._stage3_candidate_sufficient(before, candidate, 0.12))
        self.assertFalse(
            stage3_module._stage3_candidate_sufficient(
                before,
                candidate,
                0.12,
                {"max_bg_std_growth": 1.03},
                "strict",
            )
        )

    def test_stage3_large_emission_nebula_prefers_poly_first(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.48},
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.18},
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "bright_emission_reflection_nebula",
                    "object_stats": {
                        "object_area_ratio": 0.16,
                        "nebulosity_area_ratio": 0.42,
                    },
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "large_galaxy",
                    "object_stats": {"object_area_ratio": 0.48},
                },
                {},
            )
        )

    def test_stage3_faint_nebula_signal_protects_generic_profile(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        profile = {
            "target_type": "generic_low_snr_safe",
            "object_stats": {
                "nebulosity_area_ratio": 0.12,
                "faint_structure_score": 0.45,
            },
        }
        protect, context = stage3_module._stage3_should_exhaust_builtin_search(
            profile,
            {},
            {},
        )

        self.assertTrue(stage3_module._stage3_prefers_poly_first(profile, {}))
        self.assertTrue(protect)
        self.assertTrue(context["faint_nebula_protection"])
        self.assertEqual(context["protection_reason"], "faint_nebula_signal")

    def test_stage3_faint_structure_increases_nebula_preservation_penalty(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        preservation = {
            "available": True,
            "nebula_mean_change_ratio": 0.08,
            "star_retention_ratio": 1.0,
        }
        base_penalty = stage3_module._stage3_preservation_penalty(
            preservation,
            diffuse_context={"faint_structure_score": 0.40},
        )
        strong_penalty = stage3_module._stage3_preservation_penalty(
            preservation,
            diffuse_context={
                "faint_nebula_protection": True,
                "faint_structure_score": 0.90,
            },
        )

        self.assertGreater(strong_penalty, base_penalty)
        self.assertLessEqual(
            stage3_module._stage3_nebula_preservation_weight(
                {"faint_nebula_protection": True, "faint_structure_score": 1.0}
            ),
            2.5,
        )

    def test_stage3_theoretical_chain_falls_back_until_candidate_is_sufficient(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(
                    workflow_plugin_probe_enabled=False,
                    stage3_gate_profile="strict",
                )
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = Stage3SampleSiril()
                self.try_calls: list[tuple[str, ...]] = []
                self.cmd_calls: list[tuple[Any, ...]] = []
                self.saved: list[str] = []
                self.workflow_command_used: dict[str, str] = {}
                self.results: list[tuple[str, str, float, str]] = []
                self.report: dict[str, Any] = {}
                self.adaptive_measurements = [
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.12,
                        "dirty_background_score": 0.44,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.11,
                        "dirty_background_score": 0.39,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.10,
                        "dirty_background_score": 0.37,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return True

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

            def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
                return True, "quality gate ok"

            def _adaptive_features_current(self):
                return self.adaptive_measurements.pop(0)

            def _save_stage_output(self, stem: str) -> bool:
                self.saved.append(stem)
                return True

            def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
                self.report = payload

            def _record_stage(
                self,
                name: str,
                status: str,
                elapsed: float,
                message: str,
                **_metadata: Any,
            ) -> None:
                self.results.append((name, status, elapsed, message))

        processor = Stage3Fake()
        stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(
            background_attempts[:3],
            [
                ("subsky", "-rbf", "-existing"),
                ("subsky", "1", "-existing"),
                ("gxp",),
            ],
        )
        self.assertIn(
            ("load", "stage3_candidate_graxpert_native_alias"),
            processor.cmd_calls,
        )
        self.assertEqual(
            processor.workflow_command_used["GraXpert 背景提取"],
            "GraXpert native alias",
        )
        self.assertTrue(processor.report["graxpert_attempted"])
        self.assertTrue(processor.report["backup_used"])
        self.assertEqual(
            processor.report["backup_reason"],
            "builtin_and_compound_not_clean_graxpert_selected",
        )
        self.assertFalse(processor.report["fallback_used"])
        self.assertEqual(processor.report["fallback_reasons"], [])
        self.assertTrue(processor.report["subsky_existing_enforced"])
        self.assertTrue(processor.siril.set_calls)
        self.assertTrue(processor.siril.set_calls[0]["recalculate"])
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage3_compound_target_guard_excludes_protected_structure(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        eligible = stage3_module._stage3_compound_target_guard(
            {"target_type": "open_cluster"},
            {"diffuse": False},
            {},
            {"requires_review": False},
        )
        dark = stage3_module._stage3_compound_target_guard(
            {"target_type": "dark_nebula_low_contrast"},
            {"diffuse": False},
            {"protect_dark_structure": True},
            {"requires_review": False},
        )
        diffuse = stage3_module._stage3_compound_target_guard(
            {"target_type": "emission_nebula_widefield"},
            {"diffuse": True, "emission_diffuse": True},
            {"protect_nebulosity": True},
            {"requires_review": False},
        )

        self.assertTrue(eligible["eligible"])
        self.assertFalse(dark["eligible"])
        self.assertIn("dark_nebula_structure", dark["reasons"])
        self.assertFalse(diffuse["eligible"])
        self.assertIn("large_or_diffuse_nebula_signal", diffuse["reasons"])

    def test_stage3_compound_config_cannot_relax_safety_contract(self):
        cfg = pipeline_module.PipelineConfig()
        cfg.stage3_compound_min_sample_count = 10
        cfg.stage3_compound_fit_min_count = 6
        cfg.stage3_compound_validation_min_count = 2
        cfg.stage3_compound_score_abs_improvement_min = 0.01
        cfg.stage3_compound_score_rel_improvement_min = 0.02
        cfg.stage3_compound_validation_improvement_min = 0.03
        cfg.stage3_compound_zero_point_abs_max = 0.05
        cfg.stage3_compound_zero_point_rel_max = 0.50

        clamped = pipeline_module.clamp_config(cfg)

        self.assertEqual(clamped.stage3_compound_min_sample_count, 12)
        self.assertEqual(clamped.stage3_compound_fit_min_count, 8)
        self.assertEqual(clamped.stage3_compound_validation_min_count, 4)
        self.assertEqual(
            clamped.stage3_compound_score_abs_improvement_min,
            0.03,
        )
        self.assertEqual(
            clamped.stage3_compound_score_rel_improvement_min,
            0.10,
        )
        self.assertEqual(
            clamped.stage3_compound_validation_improvement_min,
            0.10,
        )
        self.assertEqual(clamped.stage3_compound_zero_point_abs_max, 0.01)
        self.assertEqual(clamped.stage3_compound_zero_point_rel_max, 0.15)

    def test_stage3_compound_runs_before_plugins_and_reuses_safe_rbf(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()

        stage3_module.run_stage3_background_extraction(processor)

        subsky_calls = [
            call for call in processor.cmd_calls if call and call[0] == "subsky"
        ]
        self.assertEqual(len(subsky_calls), 4)
        self.assertEqual(subsky_calls[0], subsky_calls[-1])
        self.assertEqual(subsky_calls[1], ("subsky", "1", "-existing"))
        self.assertEqual(subsky_calls[2], ("subsky", "1", "-existing"))
        self.assertNotIn(("gxp",), processor.cmd_calls)
        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "accepted",
        )
        self.assertEqual(
            processor.report["model_used"],
            "subsky-poly-residual-rbf",
        )
        self.assertIsNone(processor.report["fallback_reason"])
        self.assertEqual(
            processor.report["schema_version"],
            "starun.stage3-background-quality.v4",
        )
        self.assertEqual(processor.report["algorithm_contract_version"], "1.2.0")
        self.assertIn("spatial_coverage", processor.report["decision_thresholds"])
        self.assertEqual(processor.report["selection_shadow"]["status"], "ready")
        self.assertTrue(
            any(
                "background_score_components" in attempt
                for attempt in processor.report["attempts"]
                if attempt.get("status") == "accepted"
            )
        )
        split = processor.report["safe_samples"]["compound_split"]
        validation_points = {
            tuple(point) for point in split["validation_points"]
        }
        self.assertGreaterEqual(split["fit_count"], 24)
        self.assertGreaterEqual(split["validation_count"], 8)
        self.assertEqual(
            split["fit_count"] + split["validation_count"],
            split["sample_count"],
        )
        self.assertTrue(processor.siril.set_calls)
        self.assertTrue(
            all(call["count"] == split["fit_count"] for call in processor.siril.set_calls)
        )
        self.assertTrue(
            all(
                validation_points.isdisjoint(set(call["points"]))
                for call in processor.siril.set_calls
            )
        )
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertTrue(processor.report["backup_used"])

    def test_stage3_candidate_attempt_limit_counts_compound_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.cfg.stage3_candidate_attempt_limit = 2

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(len(processor.report["attempts"]), 2)
        self.assertEqual(
            processor.report["compound_fallback"]["reason"],
            "candidate_attempt_limit_reached",
        )
        self.assertFalse(
            any(
                attempt.get("source") in {"compound", "graxpert", "plugin"}
                for attempt in processor.report["attempts"]
            )
        )

    def test_stage3_final_saved_output_hard_gate_rolls_back_profiled_run(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.input_profile = {
            "state": "linear",
            "safe_for_linear_steps": True,
            "confidence": 1.0,
            "source": "test",
        }
        original_save = processor._save_stage_output
        tampered = False

        def tamper_first_final_save(stem: str) -> bool:
            nonlocal tampered
            saved = original_save(stem)
            if stem == "stage3_bgremoved" and not tampered:
                tampered = True
                processor.images[processor.state] = processor.images["baseline"].copy()
            return saved

        processor._save_stage_output = tamper_first_final_save
        with patch.object(
            stage3_module,
            "assess_target_fidelity",
            return_value=(True, {"status": "accepted", "accepted": True}),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        final_validation = processor.report["final_output_validation"]
        self.assertEqual(final_validation["status"], "rejected")
        self.assertTrue(final_validation["rollback"]["completed"])
        self.assertTrue(final_validation["rollback"]["output_saved"])
        self.assertIsNone(processor.report["model_used"])
        self.assertEqual(
            processor.report["attempted_selected_model"],
            "subsky-poly-residual-rbf",
        )
        self.assertEqual(processor.state, "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "final_output_validation_rejected",
        )

    def test_stage3_final_saved_output_soft_warning_remains_accepted(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        baseline_image = np.zeros((32, 40), dtype=np.float32)
        candidate_image = np.full((32, 40), 0.01, dtype=np.float32)
        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_gate_profile="output_first"),
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: candidate_image.copy()
            ),
        )
        baseline_validation = {
            "status": "ready",
            "robust_span": 0.010,
            "patch_mad_median": 0.004,
            "patch_median_uncertainty": 0.001,
        }
        candidate_validation = {
            "status": "ready",
            "robust_span": 0.011,
            "patch_mad_median": 0.004,
            "patch_median_uncertainty": 0.001,
        }

        with patch.object(
            stage3_module,
            "measure_background_validation",
            return_value=candidate_validation,
        ):
            report = stage3_module._stage3_final_output_validation(
                processor,
                baseline_image=baseline_image,
                baseline_validation=baseline_validation,
                validation_points=[(8.0, 8.0)] * 4,
                patch_radius=4,
                minimum_count=4,
                enforced=True,
            )

        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["severity"], "soft_warning")
        self.assertEqual(
            report["validation_gate"]["severity"],
            "soft_warning",
        )

    def test_stage3_final_consistency_warning_requires_review_without_rollback(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        original_save = processor._save_stage_output
        original_adaptive = processor._adaptive_features_current
        final_saved = False

        def mark_final_save(stem: str) -> bool:
            nonlocal final_saved
            saved = original_save(stem)
            if stem == "stage3_bgremoved":
                final_saved = True
            return saved

        def final_growth_metrics():
            metrics = original_adaptive()
            if final_saved:
                metrics["bg_std"] = 0.002
            return metrics

        processor._save_stage_output = mark_final_save
        processor._adaptive_features_current = final_growth_metrics
        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(processor.report["quality"], "review_required")
        self.assertEqual(processor.state, "compound")
        self.assertEqual(
            processor.report["final_output_validation"]["status"],
            "not_enforced",
        )

    def test_stage3_compound_validation_rejection_continues_to_plugin(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(
            compound_mode="validation_rejected",
            external_success=True,
        )

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "validation_rejected",
        )
        self.assertIn(("gxp",), processor.cmd_calls)
        compound_rbf_index = max(
            index
            for index, call in enumerate(processor.cmd_calls)
            if call and call[0] == "subsky" and "-rbf" in call
        )
        self.assertLess(compound_rbf_index, processor.cmd_calls.index(("gxp",)))
        self.assertEqual(
            processor.report["model_used"],
            "GraXpert native command",
        )
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertIn(
            {"context": "evaluated:subsky-poly-residual-rbf", "status": "restored"},
            processor.report["rollback_events"],
        )

    def test_stage3_single_hard_rejection_blocks_compound_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(external_success=True)

        def quality_gate(_before: Any, _after: Any, _preservation: Any):
            if processor.state == "polynomial":
                return False, "mock protected-signal rejection"
            return True, "quality gate ok"

        processor._stage3_quality_gate = quality_gate
        stage3_module.run_stage3_background_extraction(processor)

        compound = processor.report["compound_fallback"]
        self.assertEqual(compound["status"], "not_triggered")
        self.assertIn(
            "single_stage_hard_gate_rejection_present",
            compound["eligibility_issues"],
        )
        self.assertNotIn(
            "stage3_compound_poly_intermediate",
            processor.saved_states,
        )
        self.assertIn(("gxp",), processor.cmd_calls)

    def test_stage3_compound_rollback_failure_invalidates_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        original_cmd = processor.cmd_with_check

        def fail_compound_rollback(*args: Any, quiet: bool = False):
            if (
                tuple(args) == ("load", "stage3_bg_input")
                and "stage3_candidate_subsky_poly_residual_rbf"
                in processor.saved_states
            ):
                raise pipeline_module.CommandError("mock compound rollback failure")
            return original_cmd(*args, quiet=quiet)

        processor.cmd_with_check = fail_compound_rollback
        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "rollback_failed",
        )
        self.assertNotEqual(
            processor.report["model_used"],
            "subsky-poly-residual-rbf",
        )
        self.assertIn(
            {
                "context": "evaluated:subsky-poly-residual-rbf",
                "status": "failed",
                "reason": "mock compound rollback failure",
            },
            processor.report["rollback_events"],
        )

    def test_stage3_insufficient_compound_is_degraded_review_output(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(compound_mode="insufficient")

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "accepted_degraded",
        )
        self.assertEqual(
            processor.report["model_used"],
            "subsky-poly-residual-rbf",
        )
        self.assertEqual(processor.report["quality"], "review_required")
        self.assertTrue(processor._background_review_required)
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "compound_poly_residual_rbf_degraded_review",
        )

    def test_stage3_all_candidates_rejected_restores_baseline(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(gate_ok=False)

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [("rejected", ("subsky", "1"), "builtin")],
                ["rejected"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(processor.report["quality"], "review_required")
        self.assertTrue(processor.report["review_required"])
        self.assertEqual(processor.report["attempts"][0]["status"], "rejected")
        self.assertIn(
            {"context": "rejected:rejected", "status": "restored"},
            processor.report["rollback_events"],
        )

    def test_stage3_subsky_existing_fails_closed_without_safe_samples(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(gate_ok=True)

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [
                    (
                        "safe-samples",
                        ("subsky", "-rbf", "-existing"),
                        "builtin",
                    )
                ],
                ["safe-samples"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertNotIn(
            ("subsky", "-rbf", "-existing"),
            processor.cmd_calls,
        )
        self.assertEqual(
            processor.report["attempts"][0]["status"],
            "safe_sample_install_failed",
        )
        self.assertTrue(processor.report["subsky_existing_enforced"])

    def test_stage3_safe_sample_install_rejects_collapsed_coverage(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": []}

        def set_samples(points, **_kwargs):
            state["samples"] = list(points)
            return True

        processor = SimpleNamespace(
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: state["samples"][:-1],
            ),
        )
        points = [(float(index), float(index)) for index in range(16)]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
        )

        self.assertFalse(accepted)
        self.assertEqual(report["status"], "failed")
        self.assertIn("collapsed audited sample coverage", report["reason"])
        self.assertEqual(state["samples"], [])

    def test_stage3_safe_sample_install_accepts_siril_recalculated_subset(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": []}

        def set_samples(points, **_kwargs):
            state["samples"] = list(points)
            return True

        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_safe_sample_min_count=16),
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: state["samples"][:-1],
            ),
        )
        points = [
            (float(x), float(y))
            for y in (20, 80, 140, 200)
            for x in (20, 85, 150, 215, 280)
        ]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
        )

        self.assertTrue(accepted, report)
        self.assertEqual(report["requested_count"], 20)
        self.assertEqual(report["observed_count"], 19)
        self.assertEqual(report["siril_rejected_count"], 1)
        self.assertGreaterEqual(report["observed_coverage"]["grid_cells"], 8)

    def test_stage3_compound_fit_uses_its_own_sample_minimum(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": []}

        def set_samples(points, **_kwargs):
            state["samples"] = list(points)
            return True

        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_safe_sample_min_count=12),
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: state["samples"][:-1],
            ),
        )
        points = [
            (float(x), float(y))
            for y in (20, 100, 180)
            for x in (20, 100, 180, 260)
        ][:10]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
            minimum_count=8,
            sample_contract="compound_fit",
        )

        self.assertTrue(accepted, report)
        self.assertEqual(report["observed_count"], 9)
        self.assertEqual(report["minimum_count"], 8)
        self.assertEqual(report["sample_contract"], "compound_fit")

    def test_stage3_pattern_noise_route_preserves_baseline_and_requires_review(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(gate_ok=True)
        processor.siril = Stage3SampleSiril()
        y = np.arange(processor.siril.image.shape[0], dtype=np.float64)[:, None]
        processor.siril.image += 0.025 * np.sin(2 * np.pi * y / 8)
        processor._adaptive_features_current = lambda: {
            "bg_std": 0.0001,
            "gradient_score": 0.03,
            "dirty_background_score": 0.08,
            "chroma_noise_score": 0.02,
            "red_dominance": 1.0,
            "blue_dominance": 1.0,
            "green_cast": 1.0,
        }

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertTrue(processor._background_review_required)
        self.assertEqual(
            processor.report["noise_route"]["route"],
            "mixed_gradient_and_pattern_noise",
        )
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "background_review_required",
        )

    def test_stage3_selected_candidate_load_failure_restores_baseline(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(
            gate_ok=True,
            fail_selected_load=True,
        )

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [("accepted", ("subsky", "1"), "builtin")],
                ["accepted"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIsNone(processor.report["model_used"])
        self.assertIn(
            {
                "context": "selected_load_failed:accepted",
                "status": "restored",
            },
            processor.report["rollback_events"],
        )

    def test_stage3_candidate_chain_respects_backend_and_plugin_controls(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        primary = (
            "GraXpert-AI BGE CPU",
            ("pyscript", "GraXpert-AI.py", "-bge", "-nogpu"),
            "graxpert",
        )
        plugin_backup = ("ADBE", ("adbe",), "plugin")
        scenarios = (
            ("auto_chain", True, 0, ["poly", "rbf", primary[0], "ADBE"]),
            ("builtin_only", True, 0, ["poly", "rbf"]),
            ("graxpert_only", True, 0, [primary[0]]),
            ("auto_chain", False, 0, ["poly", "rbf"]),
            ("auto_chain", True, 2, ["poly", "rbf"]),
        )

        for backend, plugins_enabled, limit, expected in scenarios:
            with self.subTest(
                backend=backend,
                plugins_enabled=plugins_enabled,
                limit=limit,
            ):
                processor = SimpleNamespace(
                    log=FakeLogger(),
                    cfg=SimpleNamespace(
                        stage3_backend_policy=backend,
                        stage3_plugin_fallback_enabled=plugins_enabled,
                        stage3_candidate_attempt_limit=limit,
                    ),
                )
                with patch.object(
                    stage3_module,
                    "_stage3_theoretical_plugin_candidates",
                    return_value=[primary, plugin_backup],
                ):
                    ordered, _builtin_labels, _reason = (
                        stage3_module._stage3_background_candidate_chain(
                            processor,
                            rbf_attempts=[
                                (
                                    "rbf",
                                    ("subsky", "-rbf", "-existing"),
                                    "builtin",
                                )
                            ],
                            poly_attempt=[
                                ("poly", ("subsky", "1", "-existing"), "builtin")
                            ],
                            poly_first=True,
                        )
                    )
                self.assertEqual(
                    [label for label, _command, _source in ordered],
                    expected,
                )

    def test_stage3_theoretical_plugins_exclude_unintegrated_nox(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(log=FakeLogger())

        with (
            patch.object(stage3_module, "_stage3_graxpert_candidates", return_value=[]),
            patch.object(stage3_module, "_stage3_find_script", return_value=None),
        ):
            candidates = stage3_module._stage3_theoretical_plugin_candidates(processor)

        self.assertEqual(
            [label for label, _command, _source in candidates],
            ["ADBE", "DBE", "AutoDBE"],
        )

    def test_stage3_autobge_success_without_image_change_is_rejected(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.fingerprints = iter(("unchanged", "unchanged"))
                self.cmd_calls: list[tuple[Any, ...]] = []

            def _validate_plugin_script_prerequisites(self, script_path: Path):
                self.validated_script = script_path
                return True, ""

            def _current_image_fingerprint(self):
                return next(self.fingerprints)

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

        processor = Stage3Fake()
        command = (
            "pyscript",
            '"/mock/siril scripts/processing/AutoBGE.py"',
        )

        ok, reason = stage3_module._stage3_try_background_command(
            processor,
            "AutoBGE",
            command,
            "plugin",
        )

        self.assertFalse(ok)
        self.assertEqual(
            processor.validated_script,
            Path("/mock/siril scripts/processing/AutoBGE.py"),
        )
        self.assertEqual(processor.cmd_calls, [command])
        self.assertEqual(
            reason,
            "plugin_runtime_error: command returned success but image did not change",
        )

    def test_stage3_evaluates_all_builtin_candidates_before_selecting(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(workflow_plugin_probe_enabled=False)
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = Stage3SampleSiril()
                self.try_calls: list[tuple[str, ...]] = []
                self.cmd_calls: list[tuple[Any, ...]] = []
                self.saved: list[str] = []
                self.workflow_command_used: dict[str, str] = {}
                self.results: list[tuple[str, str, float, str]] = []
                self.report: dict[str, Any] = {}
                self.adaptive_measurements = [
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.12,
                        "dirty_background_score": 0.44,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.04,
                        "dirty_background_score": 0.18,
                        "chroma_noise_score": 0.04,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return True

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

            def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
                return True, "quality gate ok"

            def _adaptive_features_current(self):
                return self.adaptive_measurements.pop(0)

            def _save_stage_output(self, stem: str) -> bool:
                self.saved.append(stem)
                return True

            def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
                self.report = payload

            def _record_stage(
                self,
                name: str,
                status: str,
                elapsed: float,
                message: str,
                **_metadata: Any,
            ) -> None:
                self.results.append((name, status, elapsed, message))

        processor = Stage3Fake()
        stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(
            background_attempts,
            [
                ("subsky", "-rbf", "-existing"),
                ("subsky", "1", "-existing"),
            ],
        )
        self.assertIn(
            ("load", "stage3_candidate_subsky_rbf_existing_1"),
            processor.cmd_calls,
        )
        self.assertNotIn("GraXpert 背景提取", processor.workflow_command_used)

    def test_stage3_large_emission_nebula_tries_poly_before_rbf(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(
                    workflow_plugin_probe_enabled=False,
                    stage3_diffuse_auto_apply_enabled=True,
                )
                self.target_profile = {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.46},
                }
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = Stage3SampleSiril()
                self.try_calls: list[tuple[str, ...]] = []
                self.cmd_calls: list[tuple[Any, ...]] = []
                self.saved: list[str] = []
                self.workflow_command_used: dict[str, str] = {}
                self.results: list[tuple[str, str, float, str]] = []
                self.report: dict[str, Any] = {}
                self.adaptive_measurements = [
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.12,
                        "dirty_background_score": 0.44,
                        "object_area_ratio": 0.46,
                        "nebulosity_area_ratio": 0.42,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.11,
                        "dirty_background_score": 0.39,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.02,
                        "dirty_background_score": 0.14,
                        "chroma_noise_score": 0.03,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.02,
                        "dirty_background_score": 0.14,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                if args and args[0] not in ("save", "load", "subsky"):
                    raise RuntimeError(f"mock plugin unavailable: {args[0]}")
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return args and args[0] == "subsky"

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

            def _stage3_quality_gate(self, _before: Any, _after: Any, _preservation: Any):
                return True, "quality gate ok"

            def _adaptive_features_current(self):
                return self.adaptive_measurements.pop(0)

            def _save_stage_output(self, stem: str) -> bool:
                self.saved.append(stem)
                return True

            def _write_stage_json(self, _name: str, payload: dict[str, Any]) -> None:
                self.report = payload

            def _record_stage(
                self,
                name: str,
                status: str,
                elapsed: float,
                message: str,
                **_metadata: Any,
            ) -> None:
                self.results.append((name, status, elapsed, message))

        processor = Stage3Fake()
        stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertLess(
            background_attempts.index(("subsky", "1", "-existing")),
            background_attempts.index(("subsky", "-rbf", "-existing")),
        )
        self.assertIn(
            ("load", "stage3_candidate_subsky_rbf_existing_1"),
            processor.cmd_calls,
        )
        self.assertEqual(
            processor.report["builtin_order_reason"],
            "diffuse_signal_safe_samples_poly_before_rbf",
        )
        self.assertEqual(
            processor.report["builtin_search_mode"],
            "safe_samples_with_diffuse_signal_protection",
        )

    def test_stage3_decision_skips_clean_background(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.02,
                "dirty_background_score": 0.08,
                "chroma_noise_score": 0.01,
            },
            diffuse_context={"diffuse": False},
        )

        self.assertEqual(decision["decision"], "skip")

    def test_stage3_decision_requires_review_for_diffuse_signal(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={"target_type": "emission_nebula_widefield"},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.20,
                "dirty_background_score": 0.40,
                "chroma_noise_score": 0.08,
            },
            diffuse_context={
                "diffuse": True,
                "emission_diffuse": True,
            },
        )

        self.assertEqual(decision["decision"], "review_required")

    def test_stage3_decision_skips_low_dirty_gradient_when_diffuse_signal_is_protected(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={"target_type": "emission_nebula_widefield"},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.13,
                "dirty_background_score": 0.088,
                "chroma_noise_score": 0.15,
            },
            diffuse_context={
                "diffuse": True,
                "emission_diffuse": True,
                "pixel_signal_protection": True,
            },
        )

        self.assertEqual(decision["decision"], "skip")
        self.assertEqual(decision["source"], "target_protection_policy")
        self.assertGreaterEqual(decision["confidence"], 0.80)

    def test_stage3_decision_applies_high_confidence_offline_gradient(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.14,
                "dirty_background_score": 0.32,
                "chroma_noise_score": 0.04,
            },
            diffuse_context={"diffuse": False},
        )

        self.assertEqual(decision["decision"], "apply")
        self.assertEqual(decision["source"], "deterministic_offline_policy")

    def test_stage3_dynamic_rbf_candidates_expand_for_noisy_complex_fields(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: object())

        with patch.object(
            sys.modules["stage_support"],
            "measure_image_features",
            return_value=pipeline_module.ImageFeatures(
                bg_std=0.070,
                star_density=0.006,
                object_area_ratio=0.42,
            ),
        ):
            candidates = processor._stage3_subsky_rbf_candidates()

        self.assertGreaterEqual(len(candidates), 4)
        command_text = [" ".join(cmd) for cmd in candidates]
        self.assertTrue(any("-smooth=1.000" in text or "-smooth=1.200" in text for text in command_text))
        self.assertTrue(any("-tolerance=0.800" in text or "-tolerance=0.700" in text for text in command_text))
        self.assertTrue(all("-existing" in command for command in candidates))
