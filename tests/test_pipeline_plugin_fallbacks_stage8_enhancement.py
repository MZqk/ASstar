"""Pipeline/plugin fallback tests for stage8 enhancement."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage8EnhancementTests(PipelinePluginFallbackTestBase):
    def test_stage8_legacy_accepted_hdr_state_is_review_passthrough(self):
        processor = self._new_processor()
        processor._star_separation_state = (
            pipeline_module.StarSeparationState.REJECTED.value
        )
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_with_stars_hdr"
        processor._bright_core_with_stars_fallback = {
            "eligible": True,
            "accepted": True,
            "status": "accepted",
            "output_stem": "stage7_with_stars_hdr",
        }

        stage8_nebula_enhancement(processor)

        self.assertEqual(
            processor._stage8_final_source,
            "stage8_review_with_stars",
        )
        self.assertEqual(
            processor._stage8_final_quality,
            "star_separation_unavailable",
        )
        self.assertTrue(processor._stage8_fallback_used)
        self.assertTrue(processor._stage8_handoff["restricted_downstream"])
        self.assertEqual(processor.results[-1][1], "degraded")
        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        self.assertEqual(report["mode"], "with_stars_review_passthrough")
        self.assertFalse(report["starless_enhancement_applied"])

    def test_stage8_applies_blue_guard_when_starless_layer_is_too_blue(self):
        processor = self._new_processor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.946,
                blue_dominance=1.168,
            )
        )
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.960,
                blue_dominance=1.080,
            )
        )

        stage8_nebula_enhancement(processor)

        ccm_calls = [call for call in processor.cmd_calls if call[0] == "ccm"]
        self.assertTrue(ccm_calls)
        self.assertEqual(ccm_calls[-1][-3:], ("0", "0", "0.860000"))
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("starless blue guard applied", message)
        self.assertIn("Starless 蓝色门控后特征", message)

    def test_stage8_rolls_back_blue_guard_when_feature_gets_worse(self):
        processor = self._new_processor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.933,
                blue_dominance=1.129,
            )
        )
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.932,
                blue_dominance=2.013,
            )
        )

        stage8_nebula_enhancement(processor)

        self.assertIn(("load", "stage8_enhanced"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Starless 蓝色门控回滚", message)

    def test_stage8_narrowband_skips_blue_guard(self):
        processor = self._new_processor()
        processor._channel_semantics = "narrowband_composite"
        processor.feature_measurements.append(
            pipeline_module.ImageFeatures(
                red_dominance=0.946,
                blue_dominance=1.168,
            )
        )

        stage8_nebula_enhancement(processor)

        self.assertFalse(any(call[0] == "ccm" for call in processor.cmd_calls))
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn(
            "Stage8 global color transforms skipped by channel semantics "
            "(narrowband_composite)",
            message,
        )

    def test_stage8_manual_ohs_palette_accepts_degraded_pcc_parent(self):
        processor = self._dualband_palette_processor(requested_palette="OHS")

        stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_palette_report.json"]
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["requested_palette"], "OHS")
        self.assertEqual(report["automatic_palette"], "SHO")
        self.assertEqual(report["palette"], "OHS")
        self.assertEqual(report["selection_mode"], "explicit_user_palette")
        self.assertTrue(report["manual_override"])
        self.assertTrue(report["feeds_main_pipeline"])
        self.assertTrue(report["color_parent"]["degraded_pcc_applied"])
        self.assertTrue(report["color_parent"]["requires_review"])
        self.assertFalse(processor.cfg.optional_color_transform_enabled)
        self.assertEqual(
            processor.result_metadata[-1]["details"]["dualband_palette"],
            report,
        )

    def test_stage8_auto_palette_keeps_frozen_target_mapping(self):
        processor = self._dualband_palette_processor(requested_palette="auto")

        stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_palette_report.json"]
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["requested_palette"], "auto")
        self.assertEqual(report["automatic_palette"], "SHO")
        self.assertEqual(report["palette"], "SHO")
        self.assertEqual(report["selection_mode"], "automatic_target_mapping")
        self.assertFalse(report["manual_override"])

    def test_stage8_manual_palette_rejection_rolls_back_without_auto_fallback(self):
        processor = self._dualband_palette_processor(requested_palette="OHS")
        rejected_candidate = (
            processor.image_pixels.copy(),
            {
                "accepted": False,
                "status": "rejected",
                "palette": "OHS",
                "issues": ["luminance_drift"],
                "warnings": [],
            },
        )

        with patch(
            "stages.stage8_nebula_enhancement.build_dualband_palette_candidate",
            return_value=rejected_candidate,
        ) as builder:
            stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_palette_report.json"]
        self.assertFalse(report["accepted"])
        self.assertEqual(report["status"], "rejected_by_palette_quality_gate")
        self.assertEqual(report["planned_palette"], "OHS")
        self.assertEqual(report["automatic_palette"], "SHO")
        self.assertTrue(report["transaction"]["rollback_performed"])
        self.assertTrue(report["transaction"]["rollback_ok"])
        builder.assert_called_once()
        self.assertEqual(builder.call_args.kwargs["palette"], "OHS")

    def test_stage8_bg_growth_gate_allows_low_absolute_background_noise(self):
        processor = self._new_processor()
        helper = pipeline_module.stage8_pixels._stage8_bg_noise_growth_issue

        issue = helper(
            processor,
            growth=2.744,
            baseline_std=0.000169,
            candidate_std=0.000464,
            candidate_dirty_score=0.025,
        )

        self.assertIsNone(issue)

    def test_stage8_bg_growth_gate_rejects_material_background_noise(self):
        processor = self._new_processor()
        helper = pipeline_module.stage8_pixels._stage8_bg_noise_growth_issue

        issue = helper(
            processor,
            growth=2.744,
            baseline_std=0.00030,
            candidate_std=0.00120,
            candidate_dirty_score=0.080,
        )

        self.assertIn("bg_std_growth", issue)

    def test_stage8_quality_metrics_exclude_enhanced_signal_from_background(self):
        processor = self._new_processor()
        baseline = np.full((3, 32, 32), 0.02, dtype=np.float32)
        candidate = baseline.copy()
        checker = (np.indices((32, 16)).sum(axis=0) % 2).astype(np.float32)
        candidate[:, :, :16] += 0.08 + 0.04 * checker
        signal = np.zeros((32, 32), dtype=np.float32)
        signal[:, :16] = 1.0
        masks = {
            # Reproduce a feathered-mask overlap: the old quality metric treated
            # the whole enhanced half as background because this mask stayed 1.
            "background_mask": np.ones((32, 32), dtype=np.float32),
            "core_mask": np.zeros((32, 32), dtype=np.float32),
            "nebula_mask": signal,
            "faint_nebula_mask": np.zeros((32, 32), dtype=np.float32),
        }

        baseline_masked = pipeline_module.stage8_pixels.stage8_masked_metrics(
            processor, baseline, masks
        )
        candidate_masked = pipeline_module.stage8_pixels.stage8_masked_metrics(
            processor, candidate, masks
        )
        baseline_background = pipeline_module.stage8_pixels.background_quality_metrics(
            processor, baseline, masks
        )
        candidate_background = pipeline_module.stage8_pixels.background_quality_metrics(
            processor, candidate, masks
        )

        self.assertAlmostEqual(
            candidate_masked["background_std"],
            baseline_masked["background_std"],
            places=7,
        )
        self.assertAlmostEqual(
            candidate_background["bg_std"],
            baseline_background["bg_std"],
            places=7,
        )

    def test_stage8_quality_metrics_still_detect_true_background_noise(self):
        processor = self._new_processor()
        baseline = np.full((3, 32, 32), 0.02, dtype=np.float32)
        candidate = baseline.copy()
        checker = (np.indices((32, 16)).sum(axis=0) % 2).astype(np.float32)
        candidate[:, :, 16:] += 0.04 * checker
        signal = np.zeros((32, 32), dtype=np.float32)
        signal[:, :16] = 1.0
        masks = {
            "background_mask": np.ones((32, 32), dtype=np.float32),
            "core_mask": np.zeros((32, 32), dtype=np.float32),
            "nebula_mask": signal,
            "faint_nebula_mask": np.zeros((32, 32), dtype=np.float32),
        }

        baseline_metrics = pipeline_module.stage8_pixels.stage8_masked_metrics(
            processor, baseline, masks
        )
        candidate_metrics = pipeline_module.stage8_pixels.stage8_masked_metrics(
            processor, candidate, masks
        )

        self.assertGreater(candidate_metrics["background_std"], 0.015)
        self.assertGreater(
            candidate_metrics["background_std"],
            baseline_metrics["background_std"] + 0.015,
        )

    def test_stage8_records_post_starless_feature_summary(self):
        processor = self._new_processor()
        processor.feature_measurements.append(pipeline_module.ImageFeatures(object_area_ratio=0.33))

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Starless 后特征", message)
        self.assertIn("object_area=0.330", message)

    def test_stage8_conservative_skip_status_survives_additional_guard_reasons(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor._stage8_conservative_mode = True
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": True,
            "conservative_mode": True,
            "status": "skipped",
            "final_quality": "skipped",
            "reasons": [
                "stage8_conservative_mode_after_stage7_starless_repair",
                "stage7_quality_status=poor",
            ],
        }

        stage8_nebula_enhancement(processor)

        self.assertEqual(processor._stage8_final_quality, "conservative_skipped")
        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        quality = processor.stage_json_reports["stage8_quality.json"]
        self.assertEqual(report["status"], "conservative_skipped")
        self.assertEqual(report["final_quality"], "conservative_skipped")
        self.assertEqual(quality["initial"]["status"], "conservative_skipped")
        self.assertEqual(quality["final"]["final_quality"], "conservative_skipped")

    def test_stage8_background_only_route_keeps_immutable_baseline(self):
        processor = self._new_processor()
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": False,
            "background_only": True,
            "processing_policy": "background_only",
            "status": "background_only_passthrough",
            "final_quality": "background_only_passthrough",
            "reason_code": "stage8_subject_risk_background_only",
            "reason_text": "mock subject-only risk",
            "reasons": ["stage7_quality_status=poor"],
            "reason_details": [],
        }

        stage8_nebula_enhancement(processor)

        self.assertFalse(processor.sasp_stage8_calls)
        self.assertFalse(
            any(call and call[0] in {"satu", "unsharp", "ccm"} for call in processor.cmd_calls)
        )
        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        self.assertEqual(report["processing_policy"], "background_only")
        self.assertEqual(report["background_operation"], "none")
        self.assertFalse(report["fallback_used"])
        self.assertEqual(
            processor._stage8_final_quality,
            "background_only_passthrough",
        )
        self.assertEqual(
            processor.stage_json_reports["stage8_color_quality_report.json"]["status"],
            "reported",
        )

    def test_stage8_input_guard_classifies_conservative_skip_with_other_reasons(self):
        probe = SimpleNamespace(
            _stage8_conservative_mode=True,
            _stage7_selected_quality={"status": "poor", "derived": {}},
            cfg=SimpleNamespace(
                stage7_residual_star_score_max=0.45,
                stage7_starless_noise_gain_max=1.25,
                stage8_mask_signal_coverage_min=0.002,
            ),
            siril=SimpleNamespace(get_image_pixeldata=lambda preview=False: None),
            _stage7_halo_residue_score=lambda: 0.0,
            _stage7_effective_halo_threshold=lambda: 0.35,
            _short_text=lambda value, _limit=120: str(value),
        )

        report = pipeline_module.stage8_pixels.stage8_input_enhancement_guard(probe)

        self.assertEqual(report["status"], "conservative_skipped")
        self.assertEqual(report["final_quality"], "conservative_skipped")
        self.assertIn("stage7_quality_status=poor", report["reasons"])

    def test_stage8_input_guard_routes_subject_risk_to_background_only(self):
        image = np.full((3, 32, 32), 0.05, dtype=np.float32)
        probe = SimpleNamespace(
            _stage8_handoff={"processing_policy": "full"},
            _stage7_selected_quality={"status": "poor", "derived": {}},
            _stage7_starless_skipped=False,
            cfg=SimpleNamespace(
                stage8_masked_enhancement_enabled=True,
                stage7_residual_star_score_max=0.45,
                stage7_halo_residue_score_max=0.35,
                stage7_starless_noise_gain_max=1.25,
                stage8_mask_signal_coverage_min=0.002,
            ),
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: image.copy()
            ),
            _stage7_halo_residue_score=lambda: 0.0,
            _stage7_effective_halo_threshold=lambda: 0.35,
            _active_target_type=lambda: "large_galaxy",
            _stage8_generate_starless_masks=lambda _image: {
                "background_mask": np.ones((32, 32), dtype=np.float32),
                "coverage": {"nebula": 0.20, "faint_nebula": 0.10},
            },
            _stage8_starless_readiness_report=lambda _image, _masks: {
                "schema": "starun.stage8-starless-readiness.v1",
                "status": "reported",
                "mode": "report_only",
                "used_for_gate": False,
            },
            _short_text=lambda value, _limit=120: str(value),
        )

        report = pipeline_module.stage8_pixels.stage8_input_enhancement_guard(probe)

        self.assertEqual(report["processing_policy"], "background_only")
        self.assertTrue(report["background_only"])
        self.assertFalse(report["skip_enhancement"])
        self.assertTrue(report["background_available"])
        self.assertIn("stage7_quality_status=poor", report["subject_reasons"])

    def test_stage8_user_mode_only_tightens_the_upstream_policy(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        starmask_file = Path(td.name) / "starmask.fit"
        starmask_file.write_bytes(b"mock")
        image = np.full((3, 32, 32), 0.05, dtype=np.float32)
        cfg = SimpleNamespace(
            stage8_processing_mode="limited",
            stage8_masked_enhancement_enabled=True,
            stage7_residual_star_score_max=0.45,
            stage7_halo_residue_score_max=0.35,
            stage7_starless_noise_gain_max=1.25,
            stage8_mask_signal_coverage_min=0.002,
        )
        probe = SimpleNamespace(
            _stage8_handoff={"processing_policy": "full"},
            _stage7_selected_quality={
                "status": "ok",
                "derived": {
                    "residual_star_score": 0.0,
                    "halo_residue_score": 0.0,
                    "starless_noise_gain": 1.0,
                },
            },
            _stage7_starless_skipped=False,
            starmask_file=starmask_file,
            cfg=cfg,
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: image.copy()
            ),
            _stage7_halo_residue_score=lambda: 0.0,
            _stage7_effective_halo_threshold=lambda: 0.35,
            _active_target_type=lambda: "large_galaxy",
            _stage8_generate_starless_masks=lambda _image: {
                "background_mask": np.ones((32, 32), dtype=np.float32),
                "coverage": {"nebula": 0.20, "faint_nebula": 0.10},
            },
            _stage8_starless_readiness_report=lambda _image, _masks: {
                "schema": "starun.stage8-starless-readiness.v1",
                "status": "reported",
                "mode": "report_only",
                "used_for_gate": False,
            },
            _short_text=lambda value, _limit=120: str(value),
        )

        limited = pipeline_module.stage8_pixels.stage8_input_enhancement_guard(
            probe
        )
        self.assertEqual(limited["upstream_requested_policy"], "full")
        self.assertEqual(limited["processing_policy"], "limited")

        cfg.stage8_processing_mode = "background_only"
        background_only = (
            pipeline_module.stage8_pixels.stage8_input_enhancement_guard(probe)
        )
        self.assertEqual(background_only["processing_policy"], "background_only")

        cfg.stage8_processing_mode = "preserve"
        preserve = pipeline_module.stage8_pixels.stage8_input_enhancement_guard(
            probe
        )
        self.assertTrue(preserve["skip_enhancement"])
        self.assertEqual(preserve["reason_code"], "user_preserve")

    def test_stage8_input_guard_allows_limited_m42_candidate_with_valid_mask(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        starmask_file = Path(td.name) / "starmask.fit"
        starmask_file.write_bytes(b"mock")
        reason_text = (
            "bright_nebula_halo_advisory: "
            "0.488 > 0.350, accepted_limit=0.600"
        )
        probe = SimpleNamespace(
            _stage8_handoff={
                "processing_policy": "limited",
                "reason_code": "bright_nebula_halo_advisory",
                "reason_text": reason_text,
                "reasons": [
                    {
                        "code": "bright_nebula_halo_advisory",
                        "value": 0.488,
                        "effective_value": 0.493,
                        "base_limit": 0.35,
                        "accepted_limit": 0.60,
                    }
                ],
            },
            _stage7_selected_quality={
                "status": "ok",
                "derived": {
                    "residual_star_score": 0.10,
                    "halo_residue_score": 0.488,
                    "compact_halo_residue_score": 0.493,
                    "starless_noise_gain": 1.0,
                },
            },
            _stage7_starless_skipped=False,
            starmask_file=starmask_file,
            cfg=SimpleNamespace(
                stage8_masked_enhancement_enabled=True,
                stage7_residual_star_score_max=0.45,
                stage7_halo_residue_score_max=0.35,
                stage7_starless_noise_gain_max=1.25,
                stage8_mask_signal_coverage_min=0.002,
            ),
            siril=SimpleNamespace(get_image_pixeldata=lambda preview=False: None),
            _stage7_halo_residue_score=lambda: 0.488,
            _stage7_effective_halo_threshold=lambda: 0.60,
            _active_target_type=lambda: "bright_emission_reflection_nebula",
            _short_text=lambda value, _limit=120: str(value),
        )

        report = pipeline_module.stage8_pixels.stage8_input_enhancement_guard(probe)

        self.assertFalse(report["skip_enhancement"])
        self.assertEqual(report["processing_policy"], "limited")
        self.assertEqual(report["reason_code"], "bright_nebula_halo_advisory")
        self.assertEqual(report["advisories"], [reason_text])
        self.assertAlmostEqual(
            report["derived"]["compact_halo_residue_score"],
            0.493,
        )

    def test_stage8_limited_halo_texture_gate_rejects_new_ring_detail(self):
        cfg = SimpleNamespace(
            stage8_limited_halo_texture_growth_max=1.05,
            stage8_limited_halo_texture_delta_max=0.00075,
        )
        probe = SimpleNamespace(cfg=cfg)
        baseline = np.full((3, 64, 64), 0.12, dtype=np.float32)
        candidate = baseline.copy()
        starmask = np.zeros_like(baseline)
        starmask[:, 29:35, 29:35] = 1.0
        yy, xx = np.indices((64, 64))
        radius = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
        ring = (radius >= 5.0) & (radius <= 10.0)
        checker = np.where((xx + yy) % 2 == 0, 0.035, -0.035)
        candidate[:, ring] = np.clip(
            candidate[:, ring] + checker[ring],
            0.0,
            1.0,
        )

        report = pipeline_module.stage8_pixels.stage8_limited_halo_texture_report(
            probe,
            baseline,
            candidate,
            starmask,
        )
        advisory_baseline = baseline.copy()
        advisory_candidate = baseline.copy()
        advisory_baseline[:, ring] = np.clip(
            advisory_baseline[:, ring] + 0.005 * np.sign(checker[ring]),
            0.0,
            1.0,
        )
        advisory_candidate[:, ring] = np.clip(
            advisory_candidate[:, ring] + 0.006 * np.sign(checker[ring]),
            0.0,
            1.0,
        )
        advisory = pipeline_module.stage8_pixels.stage8_limited_halo_texture_report(
            probe,
            advisory_baseline,
            advisory_candidate,
            starmask,
        )
        unchanged = pipeline_module.stage8_pixels.stage8_limited_halo_texture_report(
            probe,
            baseline,
            baseline.copy(),
            starmask,
        )

        self.assertTrue(report["available"])
        self.assertFalse(report["accepted"])
        self.assertGreater(report["growth"], 1.05)
        self.assertGreater(report["absolute_delta"], 0.00075)
        self.assertTrue(advisory["accepted"])
        self.assertTrue(advisory["advisory"])
        self.assertEqual(
            advisory["quality_gates"]["growth"]["status"],
            "advisory",
        )
        self.assertEqual(
            advisory["quality_gates"]["absolute_delta"]["status"],
            "advisory",
        )
        self.assertTrue(unchanged["accepted"])

    def test_stage8_limited_halo_gate_extracts_compact_support_from_diffuse_starmask(self):
        cfg = SimpleNamespace(
            stage8_limited_halo_texture_growth_max=1.05,
            stage8_limited_halo_texture_delta_max=0.00075,
        )
        probe = SimpleNamespace(cfg=cfg)
        baseline = np.full((3, 128, 128), 0.12, dtype=np.float32)
        rng = np.random.default_rng(42)
        diffuse = rng.uniform(0.0, 0.10, size=(128, 128)).astype(np.float32)
        for y, x in ((20, 20), (32, 96), (64, 64), (96, 28), (105, 105)):
            diffuse[y, x] = 1.0
        starmask = np.repeat(diffuse[None, ...], 3, axis=0)

        report = pipeline_module.stage8_pixels.stage8_limited_halo_texture_report(
            probe,
            baseline,
            baseline.copy(),
            starmask,
        )

        self.assertTrue(report["available"])
        self.assertTrue(report["accepted"])
        self.assertLessEqual(report["ring_coverage"], 0.45)
        self.assertGreaterEqual(report["support_quantile"], 0.90)
        self.assertLess(report["core_coverage"], 0.05)

    def test_stage8_prefers_accepted_stage7_stretched_input(self):
        processor = self._new_processor()
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")

        stage8_nebula_enhancement(processor)

        self.assertIn(("load", "stage7_stretched"), processor.cmd_calls)
        self.assertNotIn(("load", "starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "stage7_stretched")
        self.assertIn("stage8_input_source=stage7_stretched", processor.results[-1][3])

    def test_stage8_ignores_stale_stage7_output_when_not_accepted(self):
        processor = self._new_processor()
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = False
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"stale")

        stage8_nebula_enhancement(processor)

        self.assertNotIn(("load", "stage7_stretched"), processor.cmd_calls)
        self.assertIn(("load", "starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "starless")
        self.assertIn("Stage7 output not accepted", processor.results[-1][3])

    def test_stage8_falls_back_when_accepted_stage7_file_is_missing(self):
        processor = self._new_processor()
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"

        stage8_nebula_enhancement(processor)

        self.assertNotIn(("load", "stage7_stretched"), processor.cmd_calls)
        self.assertIn(("load", "starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "starless")
        self.assertIn("preferred Stage7 input missing", processor.results[-1][3])

    def test_stage8_falls_back_to_stage6_starless_when_starless_load_fails(self):
        processor = self._new_processor()
        processor._stage7_stretch_accepted = False
        original_cmd_with_check = processor.cmd_with_check

        def selective_load_failure(*args: Any, quiet: bool = False) -> bool:
            if args == ("load", "starless"):
                processor.cmd_calls.append(args)
                raise pipeline_module.CommandError("mock missing starless")
            return original_cmd_with_check(*args, quiet=quiet)

        processor.cmd_with_check = selective_load_failure

        stage8_nebula_enhancement(processor)

        self.assertIn(("load", "starless"), processor.cmd_calls)
        self.assertIn(("load", "stage6_starless"), processor.cmd_calls)
        self.assertEqual(processor._stage8_input_source, "stage6_starless")
        self.assertTrue(processor._stage8_input_fallback_used)
        selection = processor.stage_json_reports["stage8_input_selection.json"]
        self.assertEqual(selection["selected_source"], "stage6_starless")
        self.assertTrue(selection["fallback_used"])

    def test_stage8_external_starless_explicitly_overrides_accepted_stage7(self):
        processor = self._new_processor()
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")
        external = processor.work_dir / "sasp_starless.fit"
        external.write_bytes(b"external")
        processor._find_external_fit = lambda _names: external
        processor._import_external_fit = lambda source, _stem: source

        stage8_nebula_enhancement(processor)

        selection = processor.stage_json_reports["stage8_input_selection.json"]
        self.assertEqual(processor._stage8_input_source, "starless")
        self.assertEqual(selection["selected_source"], "starless")
        self.assertTrue(selection["external_override"])
        self.assertEqual(selection["external_source_file"], str(external))

    def test_stage8_skip_handoff_forces_guard_when_masked_mode_is_disabled(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = False
        processor._stage8_handoff = {
            "processing_policy": "skip",
            "reason_code": "stage6_halo_residue_hard_limit_exceeded",
            "reason_text": "stage6_halo_residue_hard_limit_exceeded",
            "reasons": [],
        }
        guard_calls: list[str] = []
        processor._stage8_input_enhancement_guard = lambda: (
            guard_calls.append("guard")
            or {
                "skip_enhancement": True,
                "processing_policy": "skip",
                "conservative_mode": True,
                "reasons": ["stage6_halo_residue_hard_limit_exceeded"],
                "reason_code": "stage6_halo_residue_hard_limit_exceeded",
                "reason_text": "stage6_halo_residue_hard_limit_exceeded",
                "reason_details": [],
            }
        )

        stage8_nebula_enhancement(processor)

        self.assertEqual(guard_calls, ["guard"])
        self.assertFalse(processor.sasp_stage8_calls)
        self.assertEqual(processor._stage8_final_source, "stage8_input_starless")
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "safe_passthrough",
        )

    def test_stage8_baseline_save_failure_is_safe_passthrough(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        original_save = processor._save_stage_output
        processor._save_stage_output = lambda stem: (
            False if stem == "stage8_input_starless" else original_save(stem)
        )

        stage8_nebula_enhancement(processor)

        self.assertFalse(processor.sasp_stage8_calls)
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(processor._stage8_final_quality, "baseline_save_failed")
        self.assertEqual(processor._stage8_final_source, "starless")
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "safe_passthrough",
        )
        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        self.assertEqual(report["mode"], "baseline_save_failed_safe_passthrough")

    def test_stage8_builtin_saturation_fallback_is_reported_as_internal_processing(self):
        processor = self._new_processor()

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("内置 Starless satu", message)
        self.assertIn("内置 Starless unsharp", message)
        self.assertNotIn("插件未命中", message)

    def test_stage8_uses_builtin_without_siril_command_probe_when_api_unavailable_by_default(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("内置 Starless", message)
        self.assertNotIn("曲线/蒙版工具2", processor.command_chain_calls)
        self.assertNotIn("细节/结构增强", processor.command_chain_calls)
        self.assertTrue(processor.sasp_stage8_calls)

    def test_stage8_uses_sasp_python_api_when_siril_commands_unavailable(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.sasp_stage8_label = "SASP WaveScale Dark Enhancer API"

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SASP Starless 深加工使用 SASP WaveScale Dark Enhancer API", message)
        self.assertNotIn("内置 Starless", message)
        self.assertEqual(
            processor.workflow_command_used.get("SASP Starless 深加工 API"),
            "SASP WaveScale Dark Enhancer API",
        )

    def test_stage8_sasp_masked_color_pass_is_not_replayed(self):
        processor = self._new_processor()
        processor._ai_stage_advisory_enabled = lambda _name: True
        processor._request_stage8_processing_plan = lambda: {
            "selected_candidate_id": "balanced",
            "saturation": 0.10,
            "bg_factor": 1,
            "unsharp_radius": 0.8,
            "unsharp_amount": 0.35,
            "apply_after_plugins": True,
            "summary": "mock balanced plan",
        }

        def run_sasp(_plan=None):
            processor._stage8_saturation_execution = {
                "requested": 0.10,
                "applied": True,
                "applied_amount": 0.02,
                "passes": 1,
                "position": "after_structure_and_plugin_blend",
                "method": "local_adjustment_recipe",
            }
            return "SASP WaveScale Dark Enhancer API"

        processor._run_sasp_stage8_api = run_sasp
        processor._apply_stage8_builtin_enhancement = lambda *_args, **_kwargs: (
            self.fail("SASP masked enhancement must not replay the built-in color pass")
        )

        stage8_nebula_enhancement(processor)

        self.assertEqual(processor._stage8_final_source, "stage8_enhanced")
        self.assertEqual(processor._stage8_saturation_execution["passes"], 1)
        self.assertAlmostEqual(processor._saturation_boost_applied, 0.02)
        self.assertNotIn("曲线/蒙版工具2", processor.command_chain_calls)
        self.assertNotIn("细节/结构增强", processor.command_chain_calls)

    def test_stage8_limited_candidate_uses_masked_builtin_only_and_is_accepted(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor.cfg.stage8_limited_saturation_max = 0.05
        processor.cfg.optional_color_transform_enabled = True
        processor._stage8_handoff = {
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": (
                "bright_nebula_halo_advisory: "
                "0.488 > 0.350, accepted_limit=0.600"
            ),
            "reasons": [],
        }
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": False,
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": processor._stage8_handoff["reason_text"],
            "reason_details": [],
            "advisories": [processor._stage8_handoff["reason_text"]],
        }
        captured_plans: list[dict[str, Any]] = []
        processor._apply_stage8_builtin_enhancement = (
            lambda plan, *, label: (
                captured_plans.append(dict(plan))
                or [f"{label} masked limited candidate"]
            )
        )
        processor._stage8_quality_assessment = lambda: {
            "status": "ok",
            "issues": [],
        }
        saved_stems: list[str] = []
        processor._save_stage_output = lambda stem: saved_stems.append(stem) or True

        stage8_nebula_enhancement(processor)

        self.assertEqual(processor.results[-1][1], "ok")
        self.assertEqual(processor._stage8_final_source, "stage8_enhanced")
        self.assertFalse(processor._stage8_handoff["passthrough"])
        self.assertTrue(processor._stage8_handoff["restricted_downstream"])
        self.assertEqual(
            processor._stage8_handoff["outcome_reason_code"],
            "stage8_limited_candidate_accepted",
        )
        self.assertIn("stage8_limited_candidate", saved_stems)
        self.assertFalse(processor.sasp_stage8_calls)
        self.assertNotIn("调色1（可选）", processor.command_chain_calls)
        self.assertEqual(captured_plans[0]["bg_factor"], 0)
        self.assertEqual(captured_plans[0]["unsharp_radius"], 0.0)
        self.assertEqual(captured_plans[0]["unsharp_amount"], 0.0)
        self.assertLessEqual(captured_plans[0]["saturation"], 0.05)
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "completed",
        )

    def test_stage8_limited_candidate_rejection_preserves_candidate_and_rolls_back(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor.cfg.stage8_limited_saturation_max = 0.05
        processor._stage8_handoff = {
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": (
                "bright_nebula_halo_advisory: "
                "0.488 > 0.350, accepted_limit=0.600"
            ),
            "reasons": [],
        }
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": False,
            "processing_policy": "limited",
            "reason_code": "bright_nebula_halo_advisory",
            "reason_text": processor._stage8_handoff["reason_text"],
            "reason_details": [],
            "advisories": [processor._stage8_handoff["reason_text"]],
        }
        processor._apply_stage8_builtin_enhancement = (
            lambda _plan, *, label: [f"{label} masked limited candidate"]
        )
        processor._stage8_quality_assessment = lambda: {
            "status": "poor",
            "issues": ["stage8_limited_halo_texture_growth_exceeded"],
        }
        rollback_calls: list[str] = []
        processor._rollback_stage8_to_input = (
            lambda: rollback_calls.append("stage8_input_starless") or True
        )
        saved_stems: list[str] = []
        processor._save_stage_output = lambda stem: saved_stems.append(stem) or True

        stage8_nebula_enhancement(processor)

        self.assertEqual(rollback_calls, ["stage8_input_starless"])
        self.assertIn("stage8_limited_candidate", saved_stems)
        self.assertEqual(processor._stage8_final_source, "stage8_input_starless")
        self.assertEqual(
            processor._stage8_final_quality,
            "limited_candidate_rejected",
        )
        self.assertTrue(processor._stage8_handoff["passthrough"])
        self.assertEqual(
            processor._stage8_handoff["outcome_reason_code"],
            "stage8_limited_candidate_rejected",
        )
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "safe_passthrough",
        )
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])

    def test_stage8_quality_assessment_fails_closed_when_candidate_is_unreadable(self):
        probe = SimpleNamespace(
            _stage8_handoff={"processing_policy": "full"},
            _last_stage8_masked_diagnostics={},
            _read_image_by_stem=lambda _stem: None,
        )

        quality = pipeline_module.stage8_pixels.stage8_quality_assessment(probe)

        self.assertEqual(quality["status"], "poor")
        self.assertIn(
            "stage8_candidate_unavailable=stage8_enhanced",
            quality["issues"],
        )

    def test_stage8_unhandled_poor_assessment_rolls_back_instead_of_becoming_ok(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor._stage8_input_enhancement_guard = lambda: {
            "skip_enhancement": False,
            "processing_policy": "full",
            "reason_details": [],
        }
        processor._apply_stage8_builtin_enhancement = (
            lambda _plan, *, label: [f"{label} candidate"]
        )
        processor._stage8_quality_assessment = lambda: {
            "status": "poor",
            "issues": ["ai_verdict=review"],
        }
        processor._apply_stage8_color_correction_from_quality = lambda _quality: None
        processor._stage8_needs_conservative_rerun = lambda _quality: False
        processor._stage8_enhancement_quality_report = lambda: {
            "status": "ok",
            "issues": [],
        }
        rollback_calls: list[str] = []
        processor._rollback_stage8_to_input = (
            lambda: rollback_calls.append("stage8_input_starless") or True
        )

        stage8_nebula_enhancement(processor)

        self.assertEqual(rollback_calls, ["stage8_input_starless"])
        self.assertEqual(processor._stage8_final_quality, "poor")
        self.assertEqual(processor._stage8_final_source, "stage8_input_starless")
        self.assertTrue(processor._stage8_handoff["restricted_downstream"])

    def test_stage8_skips_invalid_sasp_siril_commands_when_plugin_probe_enabled(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor.command_labels["曲线/蒙版工具2"] = "SASP CreateMask"
        processor.command_labels["细节/结构增强"] = "SASP Texture and Clarity"

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("SASP Siril 深加工命令不可用", message)
        self.assertIn("内置 Starless", message)
        self.assertNotIn("曲线/蒙版工具2", processor.command_chain_calls)
        self.assertNotIn("细节/结构增强", processor.command_chain_calls)

    def test_stage8_low_dynamic_starless_builds_nonzero_faint_nebula_mask(self):
        thresholds = pipeline_module.stage8_pixels.stage8_low_signal_thresholds(
            bg_median=0.01993,
            bg_std=0.00005,
            p90=0.02010,
            p99=0.02115,
        )

        self.assertTrue(thresholds["low_signal"])
        self.assertLessEqual(thresholds["nebula_floor"], 0.0010)
        self.assertLess(thresholds["faint_floor"], 0.008)
        self.assertLess(thresholds["std_floor"], 0.01)

    def test_stage8_core_mask_detects_post_transform_color_core_below_old_floor(self):
        processor = pipeline_module.StarunPostProcessor()
        image = np.full((3, 96, 96), 0.04, dtype=np.float32)
        image[0, 43:53, 43:53] = 0.995
        image[1, 43:53, 43:53] = 0.50
        image[2, 43:53, 43:53] = 0.35

        masks = processor._stage8_generate_starless_masks(image)

        self.assertLess(masks["core_threshold"], 0.82)
        self.assertGreater(float(np.mean(masks["core_hard_mask"] > 0.50)), 0.0)
        self.assertGreater(
            masks["coverage"]["limited_core_exclusion_hard"],
            masks["coverage"]["core"],
        )
        self.assertEqual(masks["limited_core_exclusion_expand"], 8)

    def test_stage8_limited_pixels_are_core_excluded_and_weak_signal_only(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "limited"}
        processor._active_target_type = (
            lambda: "bright_emission_reflection_nebula"
        )
        processor._stage7_halo_residue_score = lambda: 0.48
        processor._stage7_effective_halo_threshold = lambda: 0.60
        image = np.full((3, 96, 96), 0.04, dtype=np.float32)
        yy, xx = np.indices((96, 96))
        radius = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
        weak = (radius >= 22.0) & (radius <= 34.0)
        image[0, weak] = 0.15
        image[1, weak] = 0.12
        image[2, weak] = 0.10
        image[0, 43:53, 43:53] = 0.995
        image[1, 43:53, 43:53] = 0.50
        image[2, 43:53, 43:53] = 0.35

        masks = processor._stage8_generate_starless_masks(image)
        enhanced, diagnostics, _messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.05, "unsharp_amount": 0.0},
                label="test",
            )
        )
        hard_core = masks["limited_core_exclusion_hard_mask"] > 0.50
        weak_signal = (
            masks["faint_nebula_mask"]
            * (1.0 - masks["nebula_mask"])
            * (1.0 - masks["limited_core_exclusion_mask"])
        )
        weak_signal = np.clip((weak_signal - 0.05) / 0.95, 0.0, 1.0)
        delta = np.max(np.abs(enhanced - image), axis=0)

        self.assertTrue(np.array_equal(enhanced[:, hard_core], image[:, hard_core]))
        self.assertEqual(float(np.max(delta[weak_signal <= 0.0])), 0.0)
        self.assertGreater(float(np.max(delta[weak_signal > 0.0])), 0.0)
        self.assertEqual(
            diagnostics["processing_scope"]["mode"],
            "limited_weak_signal_only",
        )
        self.assertEqual(
            diagnostics["processing_scope"]["core_operation_weight_max"],
            0.0,
        )
        self.assertEqual(
            diagnostics["processing_scope"]["core_max_abs_change"],
            0.0,
        )
        self.assertEqual(
            diagnostics["processing_scope"][
                "outside_weak_signal_max_abs_change"
            ],
            0.0,
        )
        operations = diagnostics["local_adjustment_engine"]["operations"]
        self.assertTrue(operations)
        self.assertEqual(
            {operation["mask"] for operation in operations},
            {"limited_weak_signal"},
        )

    def test_stage8_full_broadband_uses_target_hue_selective_saturation(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "broadband_rgb_osc"
        processor._active_target_type = lambda: "large_galaxy"
        processor._frozen_primary_target = {
            "type": "large_galaxy",
            "confidence": 0.92,
            "method": "target_profiler",
        }
        processor._stage7_halo_residue_score = lambda: 0.10
        processor._stage7_effective_halo_threshold = lambda: 0.35
        height, width = 120, 160
        yy, xx = np.indices((height, width))
        signal = np.exp(
            -(((xx - 80) / 38.0) ** 2 + ((yy - 60) / 28.0) ** 2)
        ).astype(np.float32)
        image = np.full((3, height, width), 0.035, dtype=np.float32)
        image += np.asarray([0.28, 0.15, 0.10], dtype=np.float32)[
            :, None, None
        ] * signal[None]

        _enhanced, diagnostics, messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.10, "unsharp_amount": 0.0},
                label="test",
            )
        )

        operations = diagnostics["local_adjustment_engine"]["operations"]
        selective = [
            operation
            for operation in operations
            if operation["type"] == "hue_selective_saturation"
        ]
        self.assertEqual(len(selective), 1)
        self.assertEqual(selective[0]["profile"], "galaxy")
        self.assertEqual(
            {band["id"] for band in selective[0]["bands"]},
            {"warm_core", "blue_structure"},
        )
        self.assertNotIn("saturation", {operation["type"] for operation in operations})
        saturation_execution = diagnostics["saturation_execution"]
        self.assertTrue(saturation_execution["applied"])
        self.assertEqual(saturation_execution["passes"], 1)
        self.assertEqual(
            saturation_execution["position"],
            "after_structure_and_plugin_blend",
        )
        self.assertGreater(saturation_execution["applied_amount"], 0.0)
        self.assertTrue(
            any("broadband hue-selective saturation accepted" in item for item in messages),
            messages,
        )

    def test_stage8_selective_saturation_fails_closed_to_generic_routing(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._stage7_halo_residue_score = lambda: 0.10
        processor._stage7_effective_halo_threshold = lambda: 0.35
        image = np.full((3, 96, 96), 0.04, dtype=np.float32)
        image[0, 24:72, 24:72] = 0.26
        image[1, 24:72, 24:72] = 0.15
        image[2, 24:72, 24:72] = 0.10

        for channel_semantics, target_type in (
            ("narrowband_composite", "large_galaxy"),
            ("broadband_rgb_osc", "generic_low_snr_safe"),
        ):
            with self.subTest(
                channel_semantics=channel_semantics,
                target_type=target_type,
            ):
                processor._channel_semantics = channel_semantics
                processor._active_target_type = lambda value=target_type: value
                _enhanced, diagnostics, _messages = (
                    processor._apply_stage8_masked_pixel_enhancement(
                        image,
                        {"saturation": 0.10, "unsharp_amount": 0.0},
                        label="test",
                    )
                )
                operation_types = {
                    operation["type"]
                    for operation in diagnostics["local_adjustment_engine"][
                        "operations"
                    ]
                }
                self.assertIn("saturation", operation_types)
                self.assertNotIn("hue_selective_saturation", operation_types)

    def test_stage8_physical_color_anchor_disables_blue_channel_rebalance(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "broadband_rgb_osc"
        processor.channel_profile = {"kind": "broadband_rgb_osc"}
        processor.color_calibration_report = {
            "physical_color": {
                "accepted": True,
                "feeds_main_pipeline": True,
                "method": "SPCC",
            }
        }
        processor._active_target_type = lambda: "large_galaxy"
        processor._stage7_halo_residue_score = lambda: 0.10
        processor._stage7_effective_halo_threshold = lambda: 0.35
        image = np.full((3, 96, 96), 0.04, dtype=np.float32)
        image[0, 24:72, 24:72] = 0.14
        image[1, 24:72, 24:72] = 0.13
        image[2, 24:72, 24:72] = 0.42

        _enhanced, diagnostics, messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.10, "unsharp_amount": 0.0},
                label="test",
            )
        )

        self.assertTrue(
            any("physical color anchor frozen" in item for item in messages),
            messages,
        )
        self.assertFalse(
            any("blue pre-control" in item for item in messages),
            messages,
        )
        self.assertLessEqual(diagnostics["saturation_execution"]["passes"], 1)

    def test_stage8_low_confidence_target_uses_generic_color_preserve(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "broadband_rgb_osc"
        processor.target_profile = {
            "primary_target": {
                "type": "large_galaxy",
                "confidence": 0.30,
                "method": "fallback",
            }
        }
        processor._active_target_type = lambda: "large_galaxy"
        processor._stage7_halo_residue_score = lambda: 0.10
        processor._stage7_effective_halo_threshold = lambda: 0.35
        image = np.full((3, 96, 96), 0.04, dtype=np.float32)
        image[0, 24:72, 24:72] = 0.26
        image[1, 24:72, 24:72] = 0.15
        image[2, 24:72, 24:72] = 0.10

        _enhanced, diagnostics, messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.10, "unsharp_amount": 0.0},
                label="test",
            )
        )

        operations = diagnostics["local_adjustment_engine"]["operations"]
        operation_types = {operation["type"] for operation in operations}
        self.assertIn("saturation", operation_types)
        self.assertNotIn("hue_selective_saturation", operation_types)
        self.assertEqual(
            diagnostics["local_adjustment_engine"]["color_route"],
            "generic_color_preserve",
        )
        self.assertTrue(
            any("failed closed to generic preserve" in item for item in messages),
            messages,
        )

    def test_stage8_physical_anchor_skips_unmasked_global_saturation(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.stage8_masked_enhancement_enabled = False
        processor.channel_profile = {"kind": "broadband_rgb_osc"}
        processor.color_calibration_report = {
            "physical_color": {
                "accepted": True,
                "feeds_main_pipeline": True,
                "method": "SPCC",
            }
        }
        calls: list[tuple[Any, ...]] = []
        processor.cmd_with_check = lambda *args, **_kwargs: calls.append(args) or True

        messages = processor._apply_stage8_builtin_enhancement(
            {
                "saturation": 0.10,
                "bg_factor": 1,
                "unsharp_radius": 0.8,
                "unsharp_amount": 0.20,
            },
            label="test",
        )

        self.assertFalse(any(call[0] == "satu" for call in calls))
        self.assertTrue(any(call[0] == "unsharp" for call in calls))
        self.assertTrue(
            any("requires masked color recovery" in item for item in messages),
            messages,
        )
        self.assertEqual(processor._stage8_saturation_execution["passes"], 0)

    def test_stage8_masked_substep_switches_make_the_recipe_a_noop(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "broadband_rgb_osc"
        processor._active_target_type = lambda: "large_galaxy"
        processor._stage7_halo_residue_score = lambda: 0.0
        processor._stage7_effective_halo_threshold = lambda: 0.35
        processor.cfg.stage8_nebula_saturation_enabled = False
        processor.cfg.stage8_background_denoise_enabled = False
        processor.cfg.stage8_faint_nebula_boost_enabled = False
        processor.cfg.stage8_nebula_contrast_enabled = False
        processor.cfg.stage8_masked_unsharp_enabled = False
        image = np.full((3, 96, 96), 0.05, dtype=np.float32)
        image[:, 24:72, 24:72] = 0.20

        enhanced, diagnostics, _messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.20, "unsharp_amount": 0.30},
                label="test",
            )
        )

        self.assertTrue(np.allclose(enhanced, image, atol=1e-7))
        self.assertEqual(diagnostics["saturation_execution"]["passes"], 0)
        self.assertEqual(
            diagnostics["local_adjustment_engine"].get("operations", []),
            [],
        )

    def test_stage8_conservative_rerun_respects_disabled_substeps(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = False
        processor.cfg.stage8_nebula_saturation_enabled = False
        processor.cfg.stage8_masked_unsharp_enabled = False
        calls: list[tuple[Any, ...]] = []
        processor.cmd_with_check = lambda *args, **_kwargs: calls.append(args) or True
        processor._save_stage_output = lambda _stem: True
        processor._stage8_quality_assessment = lambda: {"status": "ok"}

        result = pipeline_module.stage8_pixels.stage8_conservative_rerun(
            processor,
            0.18,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["safe_saturation"], 0.0)
        self.assertFalse(any(call[0] in {"satu", "unsharp"} for call in calls))

    def test_stage8_core_clip_growth_gate_remains_at_point_zero_one(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        baseline = np.full((3, 64, 64), 0.04, dtype=np.float32)
        baseline[0, 27:37, 27:37] = 0.990
        baseline[1, 27:37, 27:37] = 0.50
        baseline[2, 27:37, 27:37] = 0.35
        candidate = baseline.copy()
        candidate[0, 27:37, 27:37] = 1.0
        processor._read_image_by_stem = (
            lambda stem: baseline
            if stem == "stage8_input_starless"
            else candidate
        )
        processor._request_stage8_quality_ai = lambda _observations: None

        quality = processor._stage8_quality_assessment()

        self.assertGreater(quality["derived"]["core_clip_growth"], 0.0100)
        self.assertTrue(
            any(
                issue.startswith("core_clip_growth ")
                and issue.endswith(">0.0100")
                for issue in quality["issues"]
            )
        )
