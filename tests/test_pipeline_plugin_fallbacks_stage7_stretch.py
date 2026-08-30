"""Pipeline/plugin fallback tests for stage7 stretch."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403
import stage8_color_rendition


class PipelinePluginFallbackStage7StretchTests(PipelinePluginFallbackTestBase):
    def test_stage8_galaxy_chroma_factor_is_budget_bounded(self):
        factor = stage8_color_rendition.target_aware_chroma_factor(
            "galaxy_core_halo_balance",
            subject_saturation=0.08,
            effective_saturation_budget=0.40,
        )

        self.assertEqual(factor["factor"], 1.08)

    def test_stage6_accepts_only_frozen_final_stage5_linear(self):
        processor = self._new_processor()
        (processor.process_dir / "stage5_deconv.fit").write_bytes(b"mock")
        (processor.process_dir / "stage5_graxpert_deconv.fit").write_bytes(b"mock")
        (processor.process_dir / "stage5_linear.fit").write_bytes(b"mock")
        (processor.process_dir / "stage4_color.fit").write_bytes(b"stage4")
        (processor.process_dir / "stage5_input_linear.fit").write_bytes(b"baseline")
        stage6_module = sys.modules["stages.stage6_star_separation"]
        handoff_module = sys.modules["stage5_handoff"]
        input_lineage = handoff_module.freeze_stage5_input_lineage(
            processor,
            upstream_loaded=True,
            baseline_saved=True,
        )
        handoff_module.freeze_stage5_handoff(
            processor,
            origin=handoff_module.CURRENT_RUN_ORIGIN,
            stage_status="ok",
            deconvolution_integrity_ok=True,
            denoise_integrity_ok=True,
            formal_eligible=True,
            input_lineage=input_lineage,
        )

        source, mode, records = stage6_module._prepare_star_separation_source(
            processor
        )

        self.assertEqual(source, "stage5_linear")
        self.assertEqual(mode, "linear_star_separation")
        self.assertEqual(records[0]["source_stem"], "stage5_linear")

    def test_stage6_rejects_graxpert_checkpoint_without_final_stage5_handoff(self):
        processor = self._new_processor()
        (processor.process_dir / "stage5_graxpert_deconv.fit").write_bytes(b"mock")
        stage6_module = sys.modules["stages.stage6_star_separation"]
        handoff_module = sys.modules["stage5_handoff"]

        with self.assertRaises(handoff_module.Stage5HandoffError):
            stage6_module._prepare_star_separation_source(processor)

    def test_stage7_weak_object_tuning_uses_current_stretch_configuration(self):
        processor = self._new_processor()
        processor.cfg.asinh_stretch = 2.2
        processor.cfg.nebula_saturation = 0.16
        processor.auto_tune_result = pipeline_module.AutoTuneResult(
            features=pipeline_module.ImageFeatures(
                object_area_ratio=0.002,
                diffuse_ratio=0.0,
                core_brightness_ratio=0.20,
            )
        )

        note = processor._apply_weak_object_tuning()

        self.assertAlmostEqual(processor.cfg.asinh_stretch, 2.45)
        self.assertAlmostEqual(processor.cfg.nebula_saturation, 0.22)
        self.assertIn("weak-object tuning applied", note)

    def test_stage7_bg_gate_allows_sampling_edge_near_configured_floor(self):
        stage6_services = sys.modules["stage6_services"]

        self.assertLessEqual(stage6_services.stage7_effective_bg_median_min(0.020), 0.0199)

    def test_stage7_bright_nebula_uses_target_specific_star_growth_gate(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg = pipeline_module.PipelineConfig()
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        processor._measure_current_quality = lambda: pipeline_module.QualityMetrics(
            bg_median=0.03,
            median_star_size=2.0,
        )
        baseline = pipeline_module.QualityMetrics(
            bg_median=0.003,
            median_star_size=1.0,
        )

        accepted, issues, _metrics = (
            pipeline_module.StarunPostProcessor._validate_stage7_stretch_quality(
                processor,
                baseline,
            )
        )

        self.assertTrue(accepted)
        self.assertEqual(issues, [])
        self.assertTrue(
            getattr(_metrics, "_stage7_quality_advisories", [])
        )

        processor._active_target_type = lambda: "large_galaxy"
        accepted, issues, _metrics = (
            pipeline_module.StarunPostProcessor._validate_stage7_stretch_quality(
                processor,
                baseline,
            )
        )

        self.assertFalse(accepted)
        self.assertEqual(issues, ["star_size_growth 2.000>1.250"])

    def test_stage7_starless_structure_gate_replaces_generic_star_size_gate(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg = pipeline_module.PipelineConfig()
        processor._active_target_type = lambda: "large_galaxy"
        processor._measure_current_quality = lambda: pipeline_module.QualityMetrics(
            bg_median=0.03,
            median_star_size=1.60,
        )
        baseline = pipeline_module.QualityMetrics(
            bg_median=0.003,
            median_star_size=1.0,
        )

        accepted, issues, _metrics = (
            pipeline_module.StarunPostProcessor._validate_stage7_stretch_quality(
                processor,
                baseline,
                enforce_star_growth=False,
            )
        )

        self.assertTrue(accepted)
        self.assertEqual(issues, [])

    def test_stage7_bright_nebula_keeps_star_size_gate_with_structure_gate(self):
        processor = SimpleNamespace(
            _active_target_type=lambda: "bright_emission_reflection_nebula"
        )

        self.assertTrue(
            pipeline_module.StarunPostProcessor._stage7_should_enforce_star_growth(
                processor,
                True,
            )
        )
        processor._active_target_type = lambda: "large_galaxy"
        self.assertFalse(
            pipeline_module.StarunPostProcessor._stage7_should_enforce_star_growth(
                processor,
                True,
            )
        )
        self.assertTrue(
            pipeline_module.StarunPostProcessor._stage7_should_enforce_star_growth(
                processor,
                False,
            )
        )

    def test_stage7_bright_nebula_cand_a_reduces_mask_local_star_detail(self):
        height = width = 128
        yy, xx = np.mgrid[:height, :width]
        background = (
            0.08
            + 0.01 * xx.astype(np.float32) / width
            + 0.03 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 1800.0)
        ).astype(np.float32)
        starmask_gray = np.zeros((height, width), dtype=np.float32)
        stretched_gray = background.copy()
        for cy, cx in ((34, 40), (45, 94), (89, 37), (91, 92)):
            radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
            starmask_gray += 0.45 * np.exp(-radius2 / 4.0)
            stretched_gray += 0.10 * np.exp(-radius2 / 10.0)
        stretched = np.repeat(stretched_gray[None, :, :], 3, axis=0)
        starmask = np.repeat(starmask_gray[None, :, :], 3, axis=0)

        outputs: dict[str, np.ndarray] = {}
        processor = SimpleNamespace(
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: stretched.copy()
            ),
            _stage8_restore_rgb_like=lambda source, rgb: rgb,
            _set_current_image_pixeldata=lambda data, label: outputs.__setitem__(
                label,
                np.asarray(data).copy(),
            ),
        )
        params = {
            "bg_pedestal": 0.012,
            "faint_boost": 0.0,
            "core_protection": 0.0,
            "shadow_chroma_damping": 0.0,
            "star_mask_expand": 4,
            "star_faint_suppression": 0.85,
            "star_detail_suppression": 0.18,
        }

        pipeline_module.StarunPostProcessor._apply_stage7_bright_nebula_hdr_masked(
            processor,
            params,
            starmask,
        )

        protected = outputs["stage6 bright-nebula HDR masked"]
        support = starmask_gray > 0.02
        self.assertLess(
            float(np.mean(protected[:, support])),
            float(np.mean(stretched[:, support])),
        )
        self.assertLessEqual(float(np.max(protected)), float(np.max(stretched)))
        self.assertLess(
            float(np.mean(np.abs(protected[:, ~support] - stretched[:, ~support]))),
            0.001,
        )

    def test_stage7_bright_nebula_hdr_has_no_obsolete_formal_headroom_knee(self):
        height = width = 96
        image = np.full((3, height, width), 0.20, dtype=np.float32)
        image[:, 42:54, 42:54] = np.asarray(
            [0.995, 0.82, 0.61], dtype=np.float32
        )[:, None, None]
        outputs: dict[str, np.ndarray] = {}
        processor = SimpleNamespace(
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: image.copy()
            ),
            _stage8_restore_rgb_like=lambda source, rgb: rgb,
            _set_current_image_pixeldata=lambda data, label: outputs.__setitem__(
                label,
                np.asarray(data).copy(),
            ),
        )
        params = {
            "bg_pedestal": 0.024,
            "faint_boost": 0.012,
            "core_protection": 0.90,
            "shadow_chroma_damping": 0.28,
        }

        pipeline_module.StarunPostProcessor._apply_stage7_bright_nebula_hdr_masked(
            processor,
            params,
            None,
        )

        candidate = outputs["stage6 bright-nebula HDR masked"]
        self.assertGreater(float(np.max(candidate)), 0.9921)

    def test_stage7_starless_structure_gate_is_invariant_to_monotonic_stretch(self):
        height = width = 160
        yy, xx = np.mgrid[:height, :width]
        background = (
            0.008
            + 0.05 * (xx.astype(np.float32) / width)
            + 0.04 * np.exp(-((xx - 80) ** 2 + (yy - 80) ** 2) / 1800.0)
        )
        starmask_gray = np.zeros((height, width), dtype=np.float32)
        starless_gray = background.copy()
        star_centres = ((35, 40), (50, 120), (115, 45), (105, 110))
        for cy, cx in star_centres:
            radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
            starmask_gray += 0.30 * np.exp(-radius2 / 4.0)
            starless_gray += 0.002 * np.exp(-radius2 / 5.0)
        baseline = np.repeat(starless_gray[None, :, :], 3, axis=0)
        candidate_gray = np.arcsinh(12.0 * starless_gray) / np.arcsinh(12.0)
        candidate = np.repeat(candidate_gray[None, :, :], 3, axis=0)
        starmask = np.repeat(starmask_gray[None, :, :], 3, axis=0)

        assessment = (
            sys.modules["stage7_stretch_metrics"].assess_starless_structure_growth(
                baseline,
                candidate,
                starmask,
                pipeline_module.PipelineConfig(),
            )
        )

        self.assertTrue(assessment["accepted"], assessment)
        self.assertLess(
            assessment["metrics"]["masked_rank_drift_p95"],
            0.01,
        )

    def test_stage7_starless_structure_gate_rejects_new_mask_local_halos(self):
        height = width = 160
        yy, xx = np.mgrid[:height, :width]
        baseline_gray = 0.01 + 0.04 * (xx.astype(np.float32) / width)
        starmask_gray = np.zeros((height, width), dtype=np.float32)
        candidate_gray = np.arcsinh(12.0 * baseline_gray) / np.arcsinh(12.0)
        for cy, cx in ((35, 40), (50, 120), (115, 45), (105, 110)):
            radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
            starmask_gray += 0.30 * np.exp(-radius2 / 4.0)
            candidate_gray += 0.18 * np.exp(-radius2 / 80.0)
        baseline = np.repeat(baseline_gray[None, :, :], 3, axis=0)
        candidate = np.repeat(
            np.clip(candidate_gray, 0.0, 1.0)[None, :, :],
            3,
            axis=0,
        )
        starmask = np.repeat(starmask_gray[None, :, :], 3, axis=0)

        assessment = (
            sys.modules["stage7_stretch_metrics"].assess_starless_structure_growth(
                baseline,
                candidate,
                starmask,
                pipeline_module.PipelineConfig(),
            )
        )

        self.assertFalse(assessment["accepted"], assessment)
        self.assertTrue(
            any(
                issue.startswith("starless_masked_rank_drift_p95")
                for issue in assessment["issues"]
            ),
            assessment,
        )

    def test_stage7_compact_stretch_adapts_extreme_low_background(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )
        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.002),
                {"bg_std": 0.00005},
            )
        )

        self.assertEqual(adaptation["mode"], "extreme_low_background")
        self.assertEqual(candidates[0]["name"], "cand_a")
        self.assertAlmostEqual(candidates[0]["params"]["asinh_stretch"], 2.4)
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.001)
        self.assertEqual(candidates[1]["name"], "cand_b")
        self.assertAlmostEqual(candidates[1]["params"]["asinh_stretch"], 2.2)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.0005)
        self.assertLessEqual(candidates[1]["params"]["ghs_stretchamount"], 1.01)

    def test_stage7_preview_target_attainment_enforces_brightness_band(self):
        stage6_services = sys.modules["stage6_services"]
        adaptation = {
            "preview_calibration": {
                "candidate_a": {
                    "target_p50": 0.09990,
                    "calibrated_stretch": 1000.0,
                    "stretch_max": 1000.0,
                    "predicted_p50": 0.0388,
                },
                "candidate_b": {
                    "target_p50": 0.06882,
                    "calibrated_stretch": 850.0,
                    "stretch_max": 1000.0,
                    "predicted_p50": 0.06882,
                },
            }
        }

        dark = stage6_services._stage7_preview_target_attainment(
            "cand_a", {"p50": 0.04007}, adaptation
        )
        balanced = stage6_services._stage7_preview_target_attainment(
            "cand_a", {"p50": 0.11880}, adaptation
        )
        overbright = stage6_services._stage7_preview_target_attainment(
            "cand_b", {"p50": 0.26384}, adaptation
        )
        advisory = stage6_services._stage7_preview_target_attainment(
            "cand_a", {"p50": 0.084915}, adaptation
        )
        hard_dark = stage6_services._stage7_preview_target_attainment(
            "cand_a", {"p50": 0.078921}, adaptation
        )

        self.assertFalse(dark["accepted"])
        self.assertTrue(dark["stretch_saturated"])
        self.assertIn("preview_target_p50_ratio", dark["issues"][0])
        self.assertTrue(balanced["accepted"])
        self.assertAlmostEqual(balanced["attainment_ratio"], 1.189189, places=5)
        self.assertFalse(overbright["accepted"])
        self.assertAlmostEqual(overbright["attainment_ratio"], 3.833769, places=5)
        self.assertEqual(overbright["maximum_ratio"], 1.50)
        self.assertIn(
            "preview_target_p50_ratio_above_max",
            overbright["issues"][0],
        )
        self.assertTrue(advisory["accepted"])
        self.assertAlmostEqual(advisory["attainment_ratio"], 0.85, places=5)
        self.assertTrue(advisory["advisories"])
        self.assertFalse(hard_dark["accepted"])
        self.assertAlmostEqual(hard_dark["attainment_ratio"], 0.79, places=5)

    def test_stage7_preview_retention_reports_subject_color_and_detail(self):
        retention = stage6_services_module._stage7_preview_retention(
            {
                "p50": 0.14,
                "p99": 0.70,
                "object_signal_ratio": 0.80,
                "safe_preview_visibility_score": 0.30,
            },
            {
                "saturation_median": 0.08,
                "saturation_p95": 0.30,
                "microcontrast": 0.012,
            },
            {
                "p50": 0.20,
                "p99": 0.90,
                "object_signal_ratio": 1.00,
                "safe_preview_visibility_score": 0.50,
            },
            {
                "saturation_median": 0.10,
                "saturation_p95": 0.50,
                "microcontrast": 0.020,
            },
        )

        self.assertEqual(
            retention["schema"],
            "starun.stage7-preview-retention.v2",
        )
        self.assertAlmostEqual(
            retention["metrics"]["visibility"]["ratio"],
            0.60,
        )
        self.assertAlmostEqual(
            retention["metrics"]["object_signal"]["ratio"],
            0.80,
        )
        self.assertAlmostEqual(
            retention["metrics"]["subject_span"]["ratio"],
            0.80,
        )
        self.assertAlmostEqual(
            retention["metrics"]["saturation_p95"]["ratio"],
            0.60,
        )
        self.assertAlmostEqual(
            retention["metrics"]["microcontrast"]["ratio"],
            0.60,
        )

    def test_stage7_strong_display_targets_keep_profiles_and_p50_cap(self):
        baseline_stats = {
            "p01": 0.004,
            "p50": 0.010,
            "p99": 0.20,
            "max": 0.80,
        }
        preview_stats = {"p50": 0.20, "p99": 0.80}

        generic = self._new_processor()
        generic._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            generic,
        )
        _generic_candidates, generic_adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                generic,
                pipeline_module.QualityMetrics(bg_median=0.010),
                {"bg_std": 0.001},
                baseline_stats,
                preview_stats,
            )
        )
        self.assertAlmostEqual(
            generic_adaptation["preview_calibration"]["candidate_a"][
                "target_p50"
            ],
            0.17,
        )

        bright = self._new_processor()
        bright._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            bright,
        )
        bright._active_target_type = lambda: "bright_emission_reflection_nebula"
        bright.pipeline_policy = {"policy_name": "bright_nebula_hdr_conservative"}
        _bright_candidates, bright_adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                bright,
                pipeline_module.QualityMetrics(bg_median=0.010),
                {"bg_std": 0.001},
                baseline_stats,
                preview_stats,
            )
        )
        self.assertAlmostEqual(
            bright_adaptation["preview_calibration"]["candidate_a"][
                "target_p50"
            ],
            0.1394,
        )

        cluster = self._new_processor()
        cluster._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            cluster,
        )
        cluster._active_target_type = lambda: "open_cluster"
        cluster.pipeline_policy = {"policy_name": "open_cluster_color_preserve"}
        _cluster_candidates, cluster_adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                cluster,
                pipeline_module.QualityMetrics(bg_median=0.010),
                {"bg_std": 0.001},
                baseline_stats,
                preview_stats,
                source_stem="stage6_passthrough",
                star_separation_state="target_bypass",
            )
        )
        self.assertAlmostEqual(
            cluster_adaptation["preview_calibration"]["candidate_a"][
                "target_p50"
            ],
            0.1445,
        )

        _capped_candidates, capped_adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                generic,
                pipeline_module.QualityMetrics(bg_median=0.010),
                {"bg_std": 0.001},
                baseline_stats,
                {"p50": 0.80, "p99": 0.95},
            )
        )
        self.assertAlmostEqual(
            capped_adaptation["preview_calibration"]["candidate_a"][
                "target_p50"
            ],
            0.26,
        )

    def test_stage7_feedback_retry_calibrates_asinh_from_post_ghs_p50(self):
        processor = pipeline_module.StarunPostProcessor()
        candidate = {
            "name": "cand_b",
            "stem": "stage7_cand_b",
            "method": "asinh_ghs",
            "params": {
                "asinh_stretch": 69.544,
                "asinh_offset": 0.001029,
                "ghs_shadowsclip": -2.1,
                "ghs_stretchamount": 1.0,
            },
            "adaptation": {"preview_calibration": {}},
        }

        def rejected(name: str, stem: str, ratio: float) -> dict[str, Any]:
            return {
                "name": name,
                "status": "ok",
                "stem": stem,
                "diagnostics": [
                    "preview_target_p50_ratio_above_max "
                    f"{ratio:.3f}>1.500 (actual=0.07596, target=0.04512)"
                ],
                "preview_target_attainment": {
                    "attainment_ratio": ratio,
                    "actual_p50": ratio * 0.04512,
                    "target_p50": 0.04512,
                    "minimum_ratio": 0.55,
                    "maximum_ratio": 1.50,
                },
            }

        retry_1 = processor._stage7_feedback_retry_candidate(
            candidate,
            rejected("cand_b", "stage7_cand_b", 1.683),
            1,
        )
        self.assertIsNotNone(retry_1)
        assert retry_1 is not None
        self.assertEqual(retry_1["method"], "asinh_ghs")
        self.assertEqual(
            retry_1["feedback"]["adjustment"],
            "scale_asinh_from_post_transform_p50",
        )
        self.assertAlmostEqual(retry_1["params"]["asinh_stretch"], 60.743, places=3)
        self.assertAlmostEqual(retry_1["params"]["ghs_stretchamount"], 1.0)
        self.assertTrue(retry_1["feedback"]["ghs_retained"])
        self.assertEqual(
            retry_1["feedback"]["mode"],
            "post_transform_p50_calibration",
        )
        self.assertAlmostEqual(
            retry_1["feedback"]["measured_post_transform_p50"],
            1.683 * 0.04512,
        )
        self.assertEqual(retry_1["calibration_candidate"], "cand_b")
        self.assertTrue(retry_1["explicit_fallback"])

        retry_2 = processor._stage7_feedback_retry_candidate(
            retry_1,
            rejected("cand_b_feedback_1", "stage7_cand_b_feedback_1", 1.600),
            2,
        )
        self.assertIsNone(retry_2)

    def test_stage7_feedback_retry_uses_brightness_only_rejections(self):
        processor = pipeline_module.StarunPostProcessor()
        candidate = {
            "name": "cand_b",
            "stem": "stage7_cand_b",
            "method": "asinh_ghs",
            "params": {"ghs_stretchamount": 1.0},
        }
        attempt = {
            "status": "ok",
            "stem": "stage7_cand_b",
            "diagnostics": [
                "preview_target_p50_ratio_above_max 1.683>1.500",
                "core_clip_ratio 0.150>0.120",
            ],
            "preview_target_attainment": {
                "attainment_ratio": 1.683,
                "minimum_ratio": 0.55,
                "maximum_ratio": 1.50,
            },
        }

        self.assertIsNone(
            processor._stage7_feedback_retry_candidate(candidate, attempt, 1)
        )

    def test_stage7_feedback_retry_increases_asinh_after_underbright_ghs(self):
        processor = pipeline_module.StarunPostProcessor()
        candidate = {
            "name": "cand_b",
            "stem": "stage7_cand_b",
            "method": "asinh_ghs",
            "params": {
                "asinh_stretch": 40.0,
                "ghs_stretchamount": 1.0,
            },
        }
        attempt = {
            "status": "ok",
            "stem": "stage7_cand_b",
            "diagnostics": ["preview_target_p50_ratio 0.400<0.550"],
            "preview_target_attainment": {
                "attainment_ratio": 0.40,
                "minimum_ratio": 0.55,
                "maximum_ratio": 1.50,
            },
        }

        retry = processor._stage7_feedback_retry_candidate(candidate, attempt, 1)

        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual(retry["method"], "asinh_ghs")
        self.assertEqual(
            retry["feedback"]["adjustment"],
            "scale_asinh_from_post_transform_p50",
        )
        self.assertAlmostEqual(retry["params"]["asinh_stretch"], 56.1)
        self.assertEqual(retry["params"]["ghs_stretchamount"], 1.0)

    def test_stage7_adaptive_quantile_fallback_hits_preview_targets_monotonically(self):
        rng = np.random.default_rng(7)
        baseline = np.clip(
            rng.lognormal(mean=-6.0, sigma=0.9, size=(3, 128, 128)),
            0.0,
            0.40,
        ).astype(np.float32)
        adaptation = {
            "preview_calibration": {
                "candidate_a": {
                    "target_p50": 0.052,
                    "target_p99": 0.680,
                }
            }
        }
        metrics_module = sys.modules["stage7_stretch_metrics"]

        calibration = metrics_module.calibrate_adaptive_quantile_stretch(
            baseline,
            adaptation,
            pipeline_module.PipelineConfig(),
        )
        stretched = metrics_module.apply_adaptive_quantile_stretch(
            baseline,
            calibration,
        )

        self.assertEqual(calibration["status"], "ok")
        self.assertTrue(calibration["brightness_ordering_preserved"])
        self.assertAlmostEqual(float(np.percentile(stretched, 50.0)), 0.052, places=3)
        self.assertAlmostEqual(float(np.percentile(stretched, 99.0)), 0.680, places=3)
        source_order = np.argsort(baseline.reshape(-1))
        ordered_output = stretched.reshape(-1)[source_order]
        self.assertTrue(np.all(np.diff(ordered_output) >= -1e-7))
        self.assertTrue(np.all(np.isfinite(stretched)))

    def test_stage7_quantile_fallback_candidate_is_explicit_and_reuses_cand_a_gate(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = np.linspace(0.0002, 0.20, 3 * 32 * 32, dtype=np.float32).reshape(
            3,
            32,
            32,
        )
        adaptation = {
            "preview_calibration": {
                "candidate_a": {
                    "target_p50": 0.050,
                    "target_p99": 0.650,
                }
            }
        }

        candidate, calibration = processor._stage7_quantile_fallback_candidate(
            baseline,
            adaptation,
        )

        self.assertEqual(calibration["status"], "ok")
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["method"], "adaptive_quantile")
        self.assertEqual(candidate["calibration_candidate"], "cand_a")
        self.assertTrue(candidate["explicit_fallback"])
        self.assertEqual(
            candidate["params"]["calibration"]["method"],
            "linked_piecewise_linear_quantile_curve",
        )

    def test_stage7_compact_stretch_caps_offset_below_low_signal_starless(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.002),
                {"bg_std": 0.00005},
                {"p01": 0.00080, "p99": 0.00224, "max": 0.00298},
            )
        )

        self.assertEqual(adaptation["mode"], "extreme_low_background")
        self.assertIn("offset_cap", adaptation)
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.00064, places=6)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.00050, places=6)
        self.assertLess(candidates[0]["params"]["asinh_offset"], 0.002)
        self.assertLess(candidates[1]["params"]["asinh_offset"], 0.002)

    def test_stage7_compact_stretch_uses_safe_offset_for_sh2_296_statistics(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.0004478424),
                {"bg_std": 0.0000242532},
                {
                    "p01": 0.0002327696,
                    "p99": 0.0013253181,
                    "max": 0.0273016952,
                },
            )
        )

        self.assertEqual(adaptation["mode"], "extreme_low_background")
        self.assertAlmostEqual(candidates[0]["params"]["asinh_offset"], 0.000186, places=6)
        self.assertAlmostEqual(candidates[1]["params"]["asinh_offset"], 0.000112, places=6)
        self.assertLess(candidates[0]["params"]["asinh_offset"], 0.0004478424)
        self.assertLess(candidates[1]["params"]["asinh_offset"], 0.0004478424)

    def test_stage7_preview_ref_calibrates_sh2_296_asinh_strength(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.0004478424),
                {"bg_std": 0.0000242532},
                {
                    "p01": 0.0002327696,
                    "p50": 0.0004245928,
                    "p99": 0.0013253181,
                    "max": 0.0273016952,
                },
                {
                    "p50": 0.124821,
                    "p99": 0.800000,
                },
            )
        )

        calibration = adaptation["preview_calibration"]
        self.assertEqual(calibration["source"], "stage7_preview_ref")
        self.assertEqual(candidates[0]["params"]["asinh_stretch"], 1000.0)
        self.assertEqual(candidates[1]["params"]["asinh_stretch"], 1000.0)
        self.assertGreaterEqual(
            calibration["candidate_a"]["predicted_p50"],
            0.025,
        )
        self.assertLessEqual(
            calibration["candidate_b"]["predicted_p99"],
            calibration["candidate_b"]["target_p99"],
        )

    def test_stage7_starless_cand_b_uses_noise_floor_linked_mtf(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )
        baseline_p50 = 0.002586460905149579

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.0023227420169860125),
                {"bg_std": 0.00007896655006334186},
                {
                    "p01": 0.00206859961617738,
                    "p50": baseline_p50,
                    "p99": 0.008578826813027247,
                    "max": 0.3814423680305481,
                },
                {"p50": 0.18092394590377808, "p99": 0.90},
                starless_recomposition_planned=True,
            )
        )

        cand_b = candidates[1]
        self.assertEqual(cand_b["method"], "linked_mtf")
        shadows = cand_b["params"]["mtf_shadows"]
        self.assertGreater(shadows, 0.0)
        self.assertLess(shadows, 0.00206859961617738)
        self.assertEqual(cand_b["params"]["mtf_highlights"], 1.0)
        midpoint = cand_b["params"]["mtf_midtones"]
        normalized_p50 = (baseline_p50 - shadows) / (1.0 - shadows)
        mapped_p50 = (
            (midpoint - 1.0) * normalized_p50
            / ((2.0 * midpoint - 1.0) * normalized_p50 - midpoint)
        )
        target_p50 = adaptation["preview_calibration"]["candidate_b"][
            "target_p50"
        ]
        self.assertGreaterEqual(target_p50, 0.15)
        self.assertAlmostEqual(mapped_p50, target_p50, places=7)
        self.assertEqual(
            adaptation["starless_recomposition_candidate"]["method"],
            "noise_floor_linked_mtf",
        )

    def test_stage7_uint16_m8_stats_produce_unit_domain_linked_mtf(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )
        processor._pixel_distribution_stats_from_canonical = types.MethodType(
            pipeline_module.StarunPostProcessor._pixel_distribution_stats_from_canonical,
            processor,
        )
        baseline = np.full(3 * 32 * 32, 1122, dtype=np.uint16)
        baseline[:32] = 1121
        baseline[-32:] = 1161
        baseline[-1] = 65534
        baseline = baseline.reshape(3, 32, 32)
        preview = np.full(3 * 32 * 32, 16573, dtype=np.uint16)
        preview[:32] = 15340
        preview[-32:] = 20197
        preview[-1] = 65535
        preview = preview.reshape(3, 32, 32)
        baseline_stats = (
            pipeline_module.StarunPostProcessor._pixel_distribution_stats(
                processor,
                baseline,
            )
        )
        preview_stats = (
            pipeline_module.StarunPostProcessor._pixel_distribution_stats(
                processor,
                preview,
            )
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=1122.0 / 65535.0),
                {"bg_std": 2.0 / 65535.0},
                baseline_stats,
                preview_stats,
                starless_recomposition_planned=True,
            )
        )

        cand_b = candidates[1]
        self.assertAlmostEqual(baseline_stats["p50"], 1122.0 / 65535.0, places=7)
        self.assertFalse(baseline_stats["is_nearly_white"])
        self.assertEqual(
            baseline_stats["pixel_domain"]["normalization_scale"],
            65535.0,
        )
        self.assertEqual(cand_b["method"], "linked_mtf")
        self.assertGreaterEqual(cand_b["params"]["mtf_shadows"], 0.0)
        self.assertLess(cand_b["params"]["mtf_shadows"], 1.0)
        self.assertGreater(cand_b["params"]["mtf_midtones"], 0.0)
        self.assertLess(cand_b["params"]["mtf_midtones"], 1.0)
        self.assertAlmostEqual(
            cand_b["params"]["source_background"],
            1122.0 / 65535.0,
            places=7,
        )
        self.assertLess(
            adaptation["starless_recomposition_candidate"]["mtf"][0],
            1.0,
        )

    def test_stage7_noise_floor_linked_mtf_keeps_ngc7000_visible_without_shadow_clip(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )
        processor._active_target_type = lambda: "emission_nebula_widefield"
        processor.pipeline_policy = {"policy_name": "emission_nebula_widefield"}

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.018697155639529228),
                {"bg_std": 0.000036705416277982295},
                {
                    "min": 0.018372751772403717,
                    "p01": 0.018583066761493683,
                    "p50": 0.018811671063303947,
                    "p99": 0.019665954653173688,
                    "max": 0.025324970483779907,
                },
                {"p50": 0.25297608971595764, "p99": 0.5141442459821708},
                starless_recomposition_planned=True,
            )
        )

        cand_b = candidates[1]
        calibration = adaptation["preview_calibration"]["candidate_b"]
        self.assertEqual(cand_b["method"], "linked_mtf")
        self.assertGreater(cand_b["params"]["mtf_shadows"], 0.0)
        self.assertLess(
            cand_b["params"]["mtf_shadows"],
            0.018372751772403717,
        )
        self.assertGreaterEqual(calibration["target_p50"], 0.17)
        self.assertLessEqual(calibration["target_p50"], 0.22)
        self.assertGreater(calibration["predicted_p99"], 0.30)
        self.assertAlmostEqual(
            calibration["predicted_p50"],
            calibration["target_p50"],
            places=7,
        )

    def test_stage7_linked_mtf_candidate_executes_one_siril_transform(self):
        processor = self._new_processor()
        candidate = {
            "name": "cand_b",
            "method": "linked_mtf",
            "params": {
                "mtf_shadows": 0.0,
                "mtf_midtones": 0.052345678,
                "mtf_highlights": 1.0,
            },
        }

        ok, used = (
            pipeline_module.StarunPostProcessor._execute_stage7_stretch_candidate(
                processor,
                candidate,
            )
        )

        self.assertTrue(ok)
        self.assertEqual(used, "zero-shadow linked MTF")
        self.assertEqual(
            processor.cmd_calls,
            [("mtf", "0.000000", "0.052346", "1.000000")],
        )

    def test_stage7_all_production_asinh_paths_fix_rgbblend_without_human(self):
        processor = self._new_processor()
        processor._apply_stage7_bright_nebula_hdr_masked = (
            lambda _params, _starmask: None
        )
        candidates = (
            {
                "method": "asinh",
                "params": {"asinh_stretch": 2.2, "asinh_offset": 0.002},
            },
            {
                "method": "asinh_ghs",
                "params": {
                    "asinh_stretch": 2.1,
                    "asinh_offset": 0.002,
                    "ghs_shadowsclip": -2.1,
                    "ghs_stretchamount": 1.05,
                },
            },
            {
                "method": "bright_nebula_hdr_masked",
                "params": {"asinh_stretch": 2.0, "asinh_offset": 0.001},
            },
        )
        for candidate in candidates:
            with self.subTest(method=candidate["method"]):
                processor.cmd_calls.clear()
                ok, _used = (
                    pipeline_module.StarunPostProcessor._execute_stage7_stretch_candidate(
                        processor,
                        candidate,
                    )
                )
                self.assertTrue(ok)
                asinh_call = next(
                    call for call in processor.cmd_calls if call[0] == "asinh"
                )
                self.assertEqual(asinh_call[-1], "-clipmode=rgbblend")
                self.assertNotIn("-human", asinh_call)

    def test_stage7_statistical_mtf_reference_maps_median_to_target(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        image = np.linspace(0.001, 0.020, 3 * 32 * 32, dtype=np.float32)
        image = image.reshape(3, 32, 32)

        reference = metrics_module.build_statistical_mtf_reference(
            image,
            0.18,
            blackpoint_sigma=5.0,
        )

        self.assertEqual(reference["status"], "available")
        self.assertEqual(reference["role"], "reference_only")
        self.assertFalse(reference["final_candidate"])
        self.assertEqual(
            reference["equivalence_scope"],
            "linked_rgb_no_curves_no_hdr_no_normalize",
        )
        self.assertLess(reference["blackpoint"], reference["source_p50"])
        self.assertAlmostEqual(reference["predicted_p50"], 0.18, places=7)
        self.assertEqual(reference["sample_layout"], "spatial_rgb_triplets")
        self.assertEqual(
            reference["blackpoint_method"],
            "lower_half_recentered_mad",
        )
        self.assertEqual(
            reference["active_scale_estimator"],
            "lower_half_recentered_mad",
        )
        self.assertEqual(
            reference["sample_count"],
            reference["sample_pixel_count"] * 3,
        )
        self.assertAlmostEqual(
            reference["normal_gaussian_sigma_ratio"],
            0.5916931999771552,
            places=12,
        )
        self.assertAlmostEqual(
            reference["normal_gaussian_equivalent_blackpoint_sigma"],
            2.958465999885776,
            places=12,
        )
        estimators = reference["scale_estimators"]
        self.assertEqual(
            set(estimators),
            {
                "full_mad",
                "one_sided_global_center_mad",
                "lower_half_recentered_mad",
            },
        )
        self.assertAlmostEqual(
            estimators["full_mad"]["normal_gaussian_sigma_ratio"],
            0.9999985036407106,
            places=12,
        )
        self.assertFalse(
            reference["zscale_interval_reference"][
                "participates_in_selection"
            ]
        )

    def test_stage7_statistical_mtf_reports_gaussian_scale_bias(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        rng = np.random.default_rng(731)
        plane = rng.normal(0.10, 0.01, size=(256, 256)).astype(np.float32)
        image = np.stack([plane, plane, plane], axis=0)

        reference = metrics_module.build_statistical_mtf_reference(
            image,
            0.18,
            blackpoint_sigma=5.0,
        )

        self.assertEqual(reference["status"], "available")
        estimators = reference["scale_estimators"]
        full_sigma = estimators["full_mad"]["robust_sigma"]
        one_sided_sigma = estimators["one_sided_global_center_mad"][
            "robust_sigma"
        ]
        recentered_sigma = estimators["lower_half_recentered_mad"][
            "robust_sigma"
        ]
        self.assertAlmostEqual(full_sigma, 0.01, delta=0.0005)
        self.assertAlmostEqual(one_sided_sigma, 0.01, delta=0.0005)
        self.assertAlmostEqual(recentered_sigma, 0.0059169, delta=0.0005)
        self.assertGreater(full_sigma / recentered_sigma, 1.55)

    def test_stage7_multiscale_contrast_reference_is_report_only(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        axis = np.linspace(0.0, 1.0, 96, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(axis, axis)
        broad = 0.05 + 0.16 * grid_x + 0.04 * grid_y
        detail = (
            0.004 * np.sin(2.0 * np.pi * grid_x * 24.0)
            + 0.006 * np.sin(2.0 * np.pi * grid_y * 6.0)
        )
        source_gray = (broad + detail).astype(np.float32)
        candidate_gray = (broad + 2.0 * detail).astype(np.float32)
        baseline = np.stack(
            [source_gray * 1.02, source_gray, source_gray * 0.98],
            axis=0,
        )
        candidate = np.stack(
            [candidate_gray * 1.02, candidate_gray, candidate_gray * 0.98],
            axis=0,
        )

        reference = metrics_module.assess_multiscale_contrast_reference(
            baseline,
            candidate,
        )

        self.assertEqual(
            reference["schema"],
            "starun.stage7-multiscale-contrast.v1",
        )
        self.assertEqual(reference["status"], "available")
        self.assertEqual(reference["role"], "report_only")
        self.assertFalse(reference["enforced"])
        self.assertFalse(reference["participates_in_selection"])
        self.assertFalse(reference["mas_equivalent"])
        self.assertEqual(reference["requested_radii"], [1, 2, 4, 8, 16])
        self.assertEqual(len(reference["scales"]), 5)
        self.assertGreater(
            np.median(
                [item["absolute_rms_gain"] for item in reference["scales"]]
            ),
            1.20,
        )
        self.assertTrue(
            all(
                item["sign_reversal_is_ringing_proxy_only"]
                for item in reference["scales"]
            )
        )

    def test_stage7_multiscale_grid_matches_canonical_downsampling(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        values = np.arange(3 * 257 * 193, dtype=np.uint32)
        chw = np.asarray(values % 65536, dtype=np.uint16).reshape(3, 257, 193)

        analysis_grid, sampling = metrics_module._stage7_rgb_analysis_grid(
            chw,
            max_side=64,
        )
        canonical_grid = metrics_module._stage7_rgb_float_image(
            chw,
            max_side=64,
        )

        np.testing.assert_array_equal(analysis_grid, canonical_grid)
        self.assertEqual(sampling["source_layout"], "chw_rgb")
        self.assertEqual(sampling["analysis_stride"], 5)
        self.assertEqual(
            sampling["analysis_grid_shape"],
            [52, 39],
        )

    def test_stage7_statistical_mtf_reference_uses_frozen_background_mask(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        low = np.linspace(0.004, 0.016, 32 * 16, dtype=np.float32).reshape(
            32,
            16,
        )
        high = np.linspace(0.20, 0.40, 32 * 16, dtype=np.float32).reshape(
            32,
            16,
        )
        plane = np.concatenate([low, high], axis=1)
        image = np.stack([plane, plane * 1.02, plane * 0.98], axis=0)
        background_mask = np.zeros((32, 32), dtype=np.float32)
        background_mask[:, :16] = 1.0

        full_reference = metrics_module.build_statistical_mtf_reference(
            image,
            0.18,
            max_samples=96,
        )
        masked_reference = metrics_module.build_statistical_mtf_reference(
            image,
            0.18,
            max_samples=96,
            reference_mask=background_mask,
        )

        self.assertEqual(full_reference["status"], "available")
        self.assertEqual(masked_reference["status"], "available")
        self.assertEqual(
            masked_reference["reference_region"],
            "frozen_background_mask",
        )
        self.assertIsNone(masked_reference["reference_mask_fallback_reason"])
        self.assertLess(
            masked_reference["source_p50"],
            full_reference["source_p50"],
        )
        self.assertEqual(
            masked_reference["sample_count"],
            masked_reference["sample_pixel_count"] * 3,
        )
        self.assertEqual(len(masked_reference["sample_channel_medians"]), 3)
        self.assertAlmostEqual(masked_reference["predicted_p50"], 0.18, places=7)

    def test_stage7_closed_form_mtf_matches_statistical_stretch_equation(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        source_median = 0.02
        target_median = 0.18
        midpoint = metrics_module.solve_linked_mtf_midpoint(
            source_median,
            target_median,
        )

        for sample in (0.005, 0.02, 0.20, 0.80):
            numerator = (
                (source_median - 1.0) * target_median * sample
            )
            denominator = source_median * (
                target_median + sample - 1.0
            ) - target_median * sample
            statistical_stretch_value = numerator / denominator
            linked_mtf_value = metrics_module.linked_mtf_sample(
                sample,
                0.0,
                midpoint,
            )
            self.assertAlmostEqual(
                linked_mtf_value,
                statistical_stretch_value,
                places=12,
            )

    def test_stage7_rec709_vector_color_reference_is_report_only(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        luminance = np.linspace(
            0.01,
            0.15,
            32 * 32,
            dtype=np.float32,
        ).reshape(32, 32)
        baseline = np.stack(
            [luminance, luminance * 0.70, luminance * 0.40],
            axis=0,
        )
        vector_scaled = baseline * 3.0
        channel_distorted = vector_scaled.copy()
        channel_distorted[0] *= 1.20
        channel_distorted[2] *= 0.70

        vector_report = metrics_module.assess_rec709_vector_color_reference(
            baseline,
            vector_scaled,
        )
        distorted_report = metrics_module.assess_rec709_vector_color_reference(
            baseline,
            channel_distorted,
        )
        hwc_report = metrics_module.assess_rec709_vector_color_reference(
            np.moveaxis(baseline, 0, -1),
            np.moveaxis(vector_scaled, 0, -1),
        )

        self.assertEqual(vector_report["status"], "available")
        self.assertEqual(vector_report["role"], "report_only")
        self.assertFalse(vector_report["enforced"])
        self.assertFalse(vector_report["participates_in_selection"])
        self.assertAlmostEqual(
            vector_report["metrics"]["chromaticity_l1_half_p95"],
            0.0,
            places=7,
        )
        self.assertGreater(
            distorted_report["metrics"]["chromaticity_l1_half_p95"],
            0.03,
        )
        self.assertEqual(hwc_report["source_layout"], "hwc_rgb")
        self.assertAlmostEqual(
            hwc_report["metrics"]["chromaticity_l1_half_p95"],
            0.0,
            places=7,
        )

    def test_stage7_closed_form_mtf_reference_rejects_measured_p50_drift(self):
        metrics_module = sys.modules["stage7_stretch_metrics"]
        source_p50 = 0.02
        shadows = 0.005
        target_p50 = 0.18
        normalized_source = (source_p50 - shadows) / (1.0 - shadows)
        midpoint = metrics_module.solve_linked_mtf_midpoint(
            normalized_source,
            target_p50,
        )
        params = {
            "mtf_shadows": shadows,
            "mtf_midtones": round(midpoint, 9),
            "mtf_highlights": 1.0,
            "source_background": source_p50,
            "target_background": target_p50,
        }

        accepted = metrics_module.assess_closed_form_mtf_conformance(
            params,
            target_p50,
        )
        rejected = metrics_module.assess_closed_form_mtf_conformance(
            params,
            target_p50 + 0.03,
        )

        self.assertTrue(accepted["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertTrue(
            any(
                issue.startswith("closed_form_mtf_p50_error")
                for issue in rejected["issues"]
            )
        )

    def test_stage7_preview_calibration_can_be_disabled(self):
        processor = self._new_processor()
        processor.cfg.stage7_preview_calibration_enabled = False
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.0004478424),
                {"bg_std": 0.0000242532},
                {
                    "p01": 0.0002327696,
                    "p50": 0.0004245928,
                    "p99": 0.0013253181,
                    "max": 0.0273016952,
                },
                {"p50": 0.124821, "p99": 0.800000},
            )
        )

        self.assertNotIn("preview_calibration", adaptation)
        self.assertEqual(candidates[0]["params"]["asinh_stretch"], 2.4)
        self.assertEqual(candidates[1]["params"]["asinh_stretch"], 2.2)

    def test_stage7_compact_stretch_keeps_default_for_normal_background(self):
        processor = self._new_processor()
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertEqual(adaptation["mode"], "default_compact")
        self.assertEqual(
            candidates[0]["params"],
            {"asinh_stretch": 2.2, "asinh_offset": 0.002},
        )
        self.assertEqual(candidates[1]["params"]["asinh_offset"], 0.002)

    def test_stage7_compact_stretch_restores_bright_nebula_target_profile(self):
        processor = self._new_processor()
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage7_stretch": {
                "candidate_mode": ["bright_nebula_hdr_masked", "asinh_core_protect"]
            },
        }
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertEqual(adaptation["target_aware"]["name"], "bright_core_protect")
        self.assertAlmostEqual(
            adaptation["target_aware"]["cand_b_p50_multiplier"],
            0.78,
        )
        self.assertEqual(candidates[0]["method"], "bright_nebula_hdr_masked")
        self.assertLess(candidates[0]["params"]["asinh_stretch"], 2.2)
        self.assertAlmostEqual(candidates[0]["params"]["core_protection"], 0.72)
        self.assertEqual(candidates[0]["params"]["star_mask_expand"], 4)
        self.assertAlmostEqual(
            candidates[0]["params"]["star_faint_suppression"],
            0.85,
        )
        self.assertAlmostEqual(
            candidates[0]["params"]["star_detail_suppression"],
            0.18,
        )
        self.assertEqual(
            pipeline_module.PipelineConfig().stage7_bright_nebula_star_growth_ratio_max,
            1.50,
        )
        self.assertLessEqual(candidates[1]["params"]["ghs_stretchamount"], 1.0)

    def test_stage7_compact_stretch_reveals_resolved_lagoon_trifid_field(self):
        processor = self._new_processor()
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage7_stretch": {
                "candidate_mode": ["bright_nebula_hdr_masked", "asinh_core_protect"]
            },
        }
        processor.target_profile = {
            "target_type": "bright_emission_reflection_nebula",
            "secondary_labels": [
                "bright_core",
                "large_nebulosity",
                "emission_red",
                "reflection_blue",
            ],
            "composite_targets": [
                {"name": "Lagoon Nebula"},
                {"name": "Trifid Nebula"},
            ],
        }
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        target = adaptation["target_aware"]
        self.assertEqual(target["name"], "bright_core_composite_reveal")
        self.assertAlmostEqual(target["cand_a_p50_multiplier"], 0.94)
        self.assertAlmostEqual(target["cand_b_p50_multiplier"], 0.92)
        self.assertAlmostEqual(target["cand_a_pixel_params"]["core_protection"], 0.80)
        self.assertLessEqual(candidates[1]["params"]["ghs_stretchamount"], 1.0)

    def test_stage7_compact_stretch_uses_asinh_only_for_star_preserve_targets(self):
        processor = self._new_processor()
        processor._active_target_type = lambda: "open_cluster"
        processor.pipeline_policy = {
            "policy_name": "open_cluster_color_preserve",
            "stage7_stretch": {"candidate_mode": ["star_color_preserving_stretch"]},
        }
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertEqual(adaptation["target_aware"]["name"], "star_colour_preserve")
        self.assertEqual([item["method"] for item in candidates], ["asinh", "asinh"])
        self.assertLess(candidates[1]["params"]["asinh_stretch"], 2.1)

    def test_stage7_cluster_with_nebulosity_lowers_field_without_changing_primary(self):
        processor = self._new_processor()
        processor._active_target_type = lambda: "open_cluster"
        processor.target_profile = {
            "primary_target": {"type": "open_cluster", "frozen": True},
            "secondary_labels": ["large_nebulosity", "emission_red"],
        }
        processor.pipeline_policy = {
            "policy_name": "open_cluster_color_preserve",
            "stage7_stretch": {
                "candidate_mode": ["star_color_preserving_stretch"],
                "star_preserve_with_nebulosity": True,
            },
        }

        profile = (
            pipeline_module.StarunPostProcessor._stage7_target_stretch_profile(
                processor
            )
        )

        self.assertEqual(profile["target_type"], "open_cluster")
        self.assertEqual(profile["name"], "star_colour_preserve")
        self.assertEqual(
            profile["secondary_context_overlay"],
            "stellar_primary_with_nebulosity",
        )
        self.assertAlmostEqual(profile["cand_a_p50_multiplier"], 0.68)
        self.assertAlmostEqual(profile["cand_b_p50_multiplier"], 0.62)
        self.assertEqual(profile["cand_b_method"], "asinh")

    def test_stage7_widefield_profile_separates_plain_and_faint_signal_fields(self):
        processor = self._new_processor()
        processor._active_target_type = lambda: "emission_nebula_widefield"
        processor.pipeline_policy = {
            "policy_name": "emission_nebula_widefield",
            "stage7_stretch": {"candidate_mode": []},
        }

        processor.target_profile = {"secondary_labels": []}
        separated = (
            pipeline_module.StarunPostProcessor._stage7_target_stretch_profile(
                processor
            )
        )
        self.assertEqual(separated["name"], "widefield_subject_separation")
        self.assertAlmostEqual(separated["cand_a_p50_multiplier"], 0.88)
        self.assertAlmostEqual(separated["cand_b_p50_multiplier"], 0.82)

        processor.target_profile = {
            "secondary_labels": ["faint_outer_cloud"],
        }
        faint_signal = (
            pipeline_module.StarunPostProcessor._stage7_target_stretch_profile(
                processor
            )
        )
        self.assertEqual(faint_signal["name"], "widefield_faint_signal")
        self.assertAlmostEqual(faint_signal["cand_a_p50_multiplier"], 0.98)
        self.assertAlmostEqual(faint_signal["cand_b_p50_multiplier"], 0.96)

    def test_stage7_cluster_visibility_uses_stellar_retention_contract(self):
        processor = pipeline_module.StarunPostProcessor()

        def gate(profile: str):
            return processor._stage7_candidate_visibility_gate(
                {"safe_preview_visibility_score": 0.20},
                {"name": profile},
                {
                    "metrics": {
                        "visibility": {
                            "available": True,
                            "ratio": 0.19,
                            "ranking_ratio": 0.19,
                        }
                    }
                },
            )

        cluster = gate("star_colour_preserve")
        diffuse = gate("widefield_nebulosity")
        self.assertTrue(cluster["accepted"], cluster)
        self.assertEqual(
            cluster["metrics"]["relative_contract"],
            "stellar_subject",
        )
        self.assertAlmostEqual(
            cluster["metrics"]["preview_visibility_retention_minimum"],
            0.18,
        )
        self.assertFalse(diffuse["accepted"], diffuse)

    def test_stage7_target_aware_stretch_can_be_disabled(self):
        processor = self._new_processor()
        processor.cfg.stage7_target_aware_stretch_enabled = False
        processor._active_target_type = lambda: "dark_nebula_low_contrast"
        processor._stage7_baseline_background_stats = types.MethodType(
            pipeline_module.StarunPostProcessor._stage7_baseline_background_stats,
            processor,
        )

        candidates, adaptation = (
            pipeline_module.StarunPostProcessor._stage7_compact_stretch_candidates(
                processor,
                pipeline_module.QualityMetrics(bg_median=0.020),
                {},
            )
        )

        self.assertFalse(adaptation["target_aware"]["enabled"])
        self.assertEqual(candidates[0]["method"], "asinh")
        self.assertEqual(candidates[0]["params"]["asinh_stretch"], 2.2)

    def test_stage7_records_post_stretch_feature_summary(self):
        processor = self._new_processor()
        processor.feature_measurements.extend(
            [
                pipeline_module.ImageFeatures(bg_median=0.2),
                pipeline_module.ImageFeatures(bg_median=0.2),
            ]
        )
        processor._run_stage7_stretching_candidates = lambda: (
            True,
            False,
            ["local candidates"],
            "asinh",
        )

        stage7_stretching(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("stage7_features=", message)
        self.assertIn("bg_median=0.2000", message)

    def test_stage7_marks_saved_quality_candidate_as_accepted_for_stage8(self):
        processor = self._new_processor()
        processor._run_stage7_stretching_candidates = lambda: (
            True,
            False,
            ["quality_ok=true"],
            "Asinh",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertTrue(processor._stage7_stretch_accepted)
        self.assertEqual(processor._stage7_stretch_output, "stage7_stretched")

    def test_stage7_target_bypass_rejection_preserves_star_provenance(self):
        processor = self._new_processor()
        processor._star_separation_state = (
            pipeline_module.StarSeparationState.TARGET_BYPASS.value
        )
        processor._star_preserve_target_bypass = True
        processor._stage6_passthrough_source = "stage6_passthrough"
        processor._run_stage7_stretching_candidates = lambda: (
            False,
            True,
            ["all target-bypass stretch candidates rejected"],
            "",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertEqual(
            processor._star_separation_state,
            pipeline_module.StarSeparationState.TARGET_BYPASS.value,
        )
        self.assertFalse(processor._stage7_stretch_accepted)
        self.assertEqual(
            processor._stage8_handoff["reason_code"],
            "stage7_stretch_not_accepted_target_bypass",
        )
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "stage7_stretch_not_accepted_target_bypass",
        )

    def test_stage7_legacy_hdr_eligibility_is_forced_to_review_only(self):
        processor = self._new_processor()
        processor._stage6_galaxy_roi_diagnostics = {
            "status": "ready",
            "available": True,
            "seed_position_px": {"x": 31, "y": 47},
        }
        source_pixels = processor.image_pixels.copy()
        processor._star_separation_state = (
            pipeline_module.StarSeparationState.REJECTED.value
        )
        processor._bright_core_with_stars_fallback = {
            "schema": "starun.bright-core-with-stars-fallback.v1",
            "eligible": True,
            "accepted": False,
            "status": "eligible",
            "source_stem": "stage6_input",
            "rejected_pair_id": "m42-pair",
        }

        processor._run_stage7_stretching_candidates = lambda: self.fail(
            "formal with-stars HDR candidates must be unreachable"
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertFalse(processor._stage7_stretch_accepted)
        self.assertEqual(
            processor._stage7_review_source,
            "stage7_review_with_stars",
        )
        self.assertFalse(processor._bright_core_with_stars_fallback["eligible"])
        self.assertFalse(processor._bright_core_with_stars_fallback["accepted"])
        self.assertEqual(
            processor._bright_core_with_stars_fallback["status"],
            "rejected_to_review",
        )
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "bright_core_starless_rejected_after_recovery",
        )
        report = processor.stage_json_reports["stage7_stretch_quality.json"]
        self.assertEqual(report["status"], "review_only")
        self.assertEqual(
            report["reason_code"],
            "bright_core_starless_rejected_after_recovery",
        )
        self.assertEqual(report["source_stem"], "stage6_input")
        self.assertEqual(report["review_output"], "stage7_review_with_stars")
        self.assertEqual(report["attempts"], [])
        self.assertIsNone(report["selected"])
        self.assertEqual(report["galaxy_roi"]["status"], "ready")
        self.assertEqual(
            report["galaxy_roi"]["seed_position_px"],
            {"x": 31, "y": 47},
        )
        self.assertEqual(
            report["display_rendition_contract"]["name"],
            "linked_review_visibility_v2",
        )
        self.assertNotIn(("autostretch", "-linked"), processor.cmd_calls)
        np.testing.assert_array_equal(
            processor.saved_image_pixels["stage7_review_with_stars"],
            source_pixels,
        )
        candidates_report = processor.stage_json_reports[
            "stretch_candidates_report.json"
        ]
        self.assertEqual(candidates_report["delivery_mode"], "with_stars_review_only")
        self.assertEqual(candidates_report["candidates"], [])

    def test_stage7_legacy_hdr_status_stays_review_only(self):
        processor = self._new_processor()
        processor._star_separation_state = (
            pipeline_module.StarSeparationState.REJECTED.value
        )
        processor._bright_core_with_stars_fallback = {
            "schema": "starun.bright-core-with-stars-fallback.v1",
            "eligible": True,
            "accepted": False,
            "status": "eligible",
            "source_stem": "stage6_input",
        }
        processor._run_stage7_stretching_candidates = lambda: (
            False,
            True,
            ["obsolete formal candidate path should not run"],
            "",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertFalse(processor._bright_core_with_stars_fallback["accepted"])
        self.assertEqual(
            processor._bright_core_with_stars_fallback["status"],
            "rejected_to_review",
        )
        self.assertEqual(
            processor._stage7_review_source,
            "stage7_review_with_stars",
        )
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "bright_core_starless_rejected_after_recovery",
        )

    def test_stage7_marks_validated_rescue_as_accepted_and_ok(self):
        processor = self._new_processor()
        processor._stage7_stretch_validated_rescue = True
        processor._stage7_stretch_fallback_reason = "validated_chroma_rescue"
        processor._run_stage7_stretching_candidates = lambda: (
            True,
            True,
            ["stage7 background chroma rescue accepted"],
            "background_chroma_rescue",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertTrue(processor._stage7_stretch_accepted)
        self.assertEqual(processor._stage7_stretch_output, "stage7_stretched")
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "validated_chroma_rescue",
        )

    def test_stage7_forced_delivery_is_reclassified_to_with_stars_review(self):
        processor = self._new_processor()

        def forced_delivery():
            processor._stage7_stretch_forced_delivery = True
            processor._stage7_forced_delivery_reasons = [
                "background_chroma_noise_score"
            ]
            processor._stage7_stretch_fallback_reason = (
                "forced_quality_delivery"
            )
            return (
                True,
                True,
                ["technically safe appearance reject"],
                "display90_linked_lut",
            )

        processor._run_stage7_stretching_candidates = forced_delivery

        pipeline_module.run_stage7_stretching(processor)

        self.assertFalse(processor._stage7_stretch_accepted)
        self.assertIsNone(processor._stage7_stretch_output)
        self.assertEqual(processor._stage7_review_source, "stage7_review_with_stars")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(
            processor._stage8_handoff["processing_policy"],
            "skip",
        )
        self.assertTrue(processor._stage8_handoff["restricted_downstream"])
        metadata = processor.result_metadata[-1]
        self.assertEqual(metadata["reason_code"], "stage7_stretch_not_accepted")
        self.assertIn(
            "stage7_stretch_not_accepted",
            processor._stage_review_reasons(7),
        )
        message = processor.results[-1][3]
        self.assertIn("stage6_star_separation_state=accepted", message)
        self.assertIn("stage7_stretch_state=rejected", message)
        self.assertIn("evaluated but none accepted", message)
        self.assertNotIn("star_separation_state=rejected", message)
        report = processor.stage_json_reports["stage7_stretch_quality.json"]
        self.assertEqual(report["upstream_star_separation_state"], "accepted")
        self.assertEqual(report["stage7_stretch_state"], "rejected")
        self.assertEqual(
            report["starless_candidate_execution"],
            "evaluated_not_accepted",
        )

    def test_stage7_tool_failure_reports_upstream_failure_and_skips_stretch(self):
        processor = self._new_processor()
        processor._star_separation_state = (
            pipeline_module.StarSeparationState.TOOL_FAILED.value
        )
        processor._run_stage7_stretching_candidates = lambda: self.fail(
            "Stage7 candidates must not run after a Stage6 tool failure"
        )

        pipeline_module.run_stage7_stretching(processor)

        message = processor.results[-1][3]
        self.assertIn("stage6_star_separation_state=tool_failed", message)
        self.assertIn("stage7_stretch_state=skipped", message)
        self.assertIn("starless-only stretch candidates skipped", message)
        report = processor.stage_json_reports["stage7_stretch_quality.json"]
        self.assertEqual(report["upstream_star_separation_state"], "tool_failed")
        self.assertEqual(report["stage7_stretch_state"], "skipped")
        self.assertEqual(report["delivery_class"], "review_only")

    def test_stage7_all_core_unsafe_candidates_revoke_pair_and_use_with_stars_review(self):
        processor = self._new_processor()
        processor.saved_image_pixels["stage6_input"] = processor.image_pixels.copy()
        processor._selected_syqon_pair_id = "unsafe-pair"
        processor._selected_syqon_attempt_id = "unsafe-attempt"
        processor._stage6_pair_handoff = {"pair_id": "unsafe-pair"}
        processor.starless_file = processor.process_dir / "starless.fit"
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starless_file.write_bytes(b"unsafe")
        processor.starmask_file.write_bytes(b"unsafe")

        def reject_core_candidates():
            processor._stage7_destructive_core_rejected = True
            processor._stage7_revoked_pair_id = "unsafe-pair"
            processor._stage7_bright_core_integrity_rejected_reasons = [
                "local_core_colored_plateau_component_ratio"
            ]
            return False, True, ["all candidates core-unsafe"], ""

        processor._run_stage7_stretching_candidates = reject_core_candidates

        pipeline_module.run_stage7_stretching(processor)

        self.assertEqual(
            processor._star_separation_state,
            pipeline_module.StarSeparationState.REJECTED.value,
        )
        self.assertEqual(processor._stage6_passthrough_source, "stage6_input")
        self.assertEqual(processor._stage7_review_source, "stage7_review_with_stars")
        self.assertIsNone(processor.starless_file)
        self.assertIsNone(processor.starmask_file)
        self.assertIsNone(processor._stage6_pair_handoff)
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "stage7_bright_core_integrity_rejected",
        )

    def test_stage7_validated_fallback_reason_distinguishes_rescue_modes(self):
        processor = pipeline_module.StarunPostProcessor()

        self.assertEqual(
            processor._stage7_validated_fallback_reason(
                {
                    "explicit_fallback": True,
                    "method": "asinh",
                    "feedback": {"mode": "post_transform_p50_calibration"},
                }
            ),
            "validated_brightness_feedback",
        )
        self.assertEqual(
            processor._stage7_validated_fallback_reason(
                {"explicit_fallback": True, "method": "adaptive_quantile"}
            ),
            "validated_quantile_fallback",
        )
        self.assertEqual(
            processor._stage7_validated_fallback_reason(
                {
                    "explicit_fallback": True,
                    "method": "background_chroma_rescue",
                }
            ),
            "validated_chroma_rescue",
        )

    def test_stage7_marks_review_only_candidate_as_degraded_not_accepted(self):
        processor = self._new_processor()

        def review_only_stretch():
            processor._stage7_review_source = "stage7_cand_a"
            return False, False, ["selected safe review source"], ""

        processor._run_stage7_stretching_candidates = review_only_stretch

        pipeline_module.run_stage7_stretching(processor)

        self.assertFalse(processor._stage7_stretch_accepted)
        self.assertIsNone(processor._stage7_stretch_output)
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(
            processor._stage7_review_source,
            "stage7_review_with_stars",
        )
        self.assertNotIn(("autostretch", "-linked"), processor.cmd_calls)
        self.assertIn(
            "starless-only stretch candidates evaluated but none accepted",
            processor.results[-1][3],
        )

    def test_stage7_chroma_rescue_only_allows_exclusive_chroma_rejection(self):
        processor = pipeline_module.StarunPostProcessor()
        chroma_only = {
            "status": "ok",
            "stem": "stage7_cand_a",
            "target_local_quality": {"accepted": True},
            "diagnostics": [
                "background_chroma_noise_score 0.431>0.340",
                "background_chroma_load_growth 3.210>1.350",
            ],
        }

        self.assertTrue(processor._stage7_attempt_allows_chroma_rescue(chroma_only))
        self.assertFalse(
            processor._stage7_attempt_allows_chroma_rescue(
                {
                    **chroma_only,
                    "diagnostics": [
                        *chroma_only["diagnostics"],
                        "background_mottling_score 0.600>0.450",
                    ],
                }
            )
        )

        self.assertFalse(
            processor._stage7_attempt_allows_chroma_rescue(
                {
                    **chroma_only,
                    "diagnostics": [
                        *chroma_only["diagnostics"],
                        "preview_target_p50_ratio_above_max 3.833>1.500",
                    ],
                }
            )
        )
        self.assertFalse(
            processor._stage7_attempt_allows_chroma_rescue(
                {**chroma_only, "method": "adaptive_quantile"}
            )
        )
        processor.cfg.stage7_chroma_rescue_enabled = False
        self.assertFalse(
            processor._stage7_attempt_allows_chroma_rescue(chroma_only)
        )

    def test_stage7_expected_tone_map_fails_closed_for_non_mapping_contract(self):
        processor = pipeline_module.StarunPostProcessor()

        expected, identity = processor._stage7_expected_tone_map(
            "stage7_cand_b",
            np.full((3, 8, 8), 0.1, dtype=np.float32),
        )

        self.assertIsNone(expected)
        self.assertEqual(identity["status"], "unavailable")
        self.assertIn("not a mapping", identity["reason"])

    def test_stage7_chroma_rescue_uses_three_strength_levels(self):
        processor = pipeline_module.StarunPostProcessor()

        self.assertEqual(
            processor._stage7_chroma_rescue_strengths(),
            [0.10, 0.20, 0.35],
        )

    def test_stage7_chroma_rescue_attempt_limit_truncates_safe_ladder(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.stage7_chroma_rescue_max_attempts = 1
        self.assertEqual(processor._stage7_chroma_rescue_strengths(), [0.10])
        processor.cfg.stage7_chroma_rescue_max_attempts = 0
        self.assertEqual(processor._stage7_chroma_rescue_strengths(), [])

    def test_stage7_candidate_selection_prefers_best_post_rescue_quality(self):
        processor = pipeline_module.StarunPostProcessor()
        gate_limits = {
            "chroma_noise_score_max": 0.34,
            "background_mottling_score_max": 0.45,
            "chroma_load_growth_max": 1.35,
        }

        def attempt(name: str, chroma: float, risk: float) -> dict[str, Any]:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": False,
                "diagnostics": [
                    f"background_chroma_noise_score {chroma:.3f}>0.340"
                ],
                "risk_score": risk,
                "background_quality_gate": {
                    "metrics": {
                        "chroma_noise_score": chroma,
                        "background_mottling_score": 0.10,
                        "chroma_load_growth": 1.10,
                    },
                    "limits": gate_limits,
                },
            }

        candidates = [
            attempt("cand_b", 0.751, 10.0),
            attempt("chroma_rescue_1", 0.515, 8.0),
            attempt("chroma_rescue_2", 0.380, 5.0),
        ]

        selected = min(
            candidates,
            key=processor._stage7_review_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "chroma_rescue_2")

    def test_stage7_candidate_ranking_contract_is_fixed(self):
        processor = pipeline_module.StarunPostProcessor()
        attempt = {
            "name": "cand_a",
            "stem": "stage7_cand_a",
            "status": "ok",
            "allowed_as_final": True,
            "diagnostics": [],
            "risk_score": 0.25,
        }

        key_before = processor._stage7_candidate_selection_key(attempt)
        attempt["multiscale_contrast_reference"] = {
            "status": "available",
            "participates_in_selection": False,
            "scales": [{"absolute_rms_gain": 999.0}],
        }
        key_with_multiscale_report = processor._stage7_candidate_selection_key(
            attempt
        )
        processor.pipeline_policy = {
            "stage7_stretch": {
                "scoring": {
                    "core_blowout_weight": 999.0,
                    "bg_noise_weight": 0.0,
                }
            }
        }
        key_after = processor._stage7_candidate_selection_key(attempt)

        self.assertEqual(key_before, key_with_multiscale_report)
        self.assertEqual(key_before, key_after)
        self.assertEqual(
            stage6_services_module.STAGE7_CANDIDATE_RANKING_POLICY,
            "hard_gate_continuous_quality_v7",
        )

    def test_stage7_strict_target_prefers_unsaturated_safe_candidate(self):
        processor = pipeline_module.StarunPostProcessor()

        def attempt(name: str, saturated: bool) -> dict[str, Any]:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "technical_safe": True,
                "diagnostics": [],
                "advisories": [],
                "risk_score": 0.1,
                "presentation_score": {"score": 0.8},
                "preview_target_attainment": {
                    "stretch_saturated": saturated,
                },
                "target_local_quality": {
                    "strict_target_evidence": {"strict": True},
                    "quality_gates": {},
                },
            }

        selected = min(
            [attempt("saturated", True), attempt("unsaturated", False)],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "unsaturated")

    def test_stage7_strict_forced_delivery_prefers_unsaturated_safe_candidate(self):
        def attempt(name: str, saturated: bool) -> dict[str, Any]:
            return {
                "name": name,
                "status": "ok",
                "stem": f"stage7_{name}",
                "target_local_quality": {
                    "strict_target_evidence": {"strict": True},
                },
                "preview_target_attainment": {
                    "stretch_saturated": saturated,
                },
                "presentation_score": {"score": 0.5},
                "background_quality_gate": {"metrics": {}, "limits": {}},
                "color_vector_gate": {"metrics": {}, "limits": {}},
                "diagnostics": ["background_chroma_noise_score"],
            }

        selected = min(
            [attempt("saturated", True), attempt("unsaturated", False)],
            key=pipeline_module.StarunPostProcessor._stage7_forced_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "unsaturated")

    def test_stage7_all_core_unsafe_candidates_trigger_pair_rejection(self):
        candidates = [
            {
                "target_local_quality": {
                    "quality_gates": {
                        "local_core_clip_ratio": {"hard_failed": True}
                    }
                }
            },
            {
                "target_local_quality": {
                    "quality_gates": {
                        "local_core_parity_phase_span": {"hard_failed": True}
                    }
                }
            },
        ]

        rejected, reasons = (
            stage6_services_module._stage7_all_saved_candidates_fail_core_gates(
                candidates,
                strict_target=True,
            )
        )

        self.assertTrue(rejected)
        self.assertEqual(
            reasons,
            ["local_core_clip_ratio", "local_core_parity_phase_span"],
        )
        candidates.append(
            {"target_local_quality": {"quality_gates": {}}}
        )
        rejected, _reasons = (
            stage6_services_module._stage7_all_saved_candidates_fail_core_gates(
                candidates,
                strict_target=True,
            )
        )
        self.assertFalse(rejected)

    def test_stage7_candidate_selection_uses_preview_perceptual_retention(self):
        processor = pipeline_module.StarunPostProcessor()

        def attempt(name: str, saturation: float, microcontrast: float) -> dict[str, Any]:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "diagnostics": [],
                "risk_score": 1.0,
                "preview_target_attainment": {
                    "attainment_ratio": 1.0,
                    "minimum_ratio": 0.55,
                    "maximum_ratio": 1.50,
                },
                "preview_retention": {
                    "metrics": {
                        "visibility": {"available": True, "ranking_ratio": 0.8},
                        "object_signal": {"available": True, "ranking_ratio": 0.8},
                        "saturation_p95": {
                            "available": True,
                            "ranking_ratio": saturation,
                        },
                        "microcontrast": {
                            "available": True,
                            "ranking_ratio": microcontrast,
                        },
                    }
                },
            }

        weaker_color = attempt("cand_a", 0.30, 0.90)
        stronger_color = attempt("cand_b", 0.80, 0.40)
        selected = min(
            [weaker_color, stronger_color],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "cand_b")

    def test_stage7_candidate_selection_keeps_hard_gate_for_final_output(self):
        processor = pipeline_module.StarunPostProcessor()
        rejected_low_risk = {
            "name": "rejected_low_risk",
            "stem": "stage7_rejected_low_risk",
            "status": "ok",
            "allowed_as_final": False,
            "diagnostics": ["background_chroma_noise_score 0.350>0.340"],
            "risk_score": 1.0,
        }
        accepted_higher_risk = {
            "name": "accepted_higher_risk",
            "stem": "stage7_accepted_higher_risk",
            "status": "ok",
            "allowed_as_final": True,
            "diagnostics": [],
            "risk_score": 4.0,
        }

        selected = min(
            [rejected_low_risk, accepted_higher_risk],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "accepted_higher_risk")

    def test_stage7_candidate_selection_honors_low_absolute_chroma_exemption(self):
        processor = pipeline_module.StarunPostProcessor()
        limits = {
            "chroma_noise_score_max": 0.34,
            "background_mottling_score_max": 0.45,
            "chroma_load_growth_max": 1.35,
            "chroma_load_low_absolute_max": 0.05,
        }
        exempted_primary = {
            "name": "cand_a",
            "stem": "stage7_cand_a",
            "status": "ok",
            "allowed_as_final": True,
            "diagnostics": [],
            "risk_score": 0.054,
            "background_quality_gate": {
                "metrics": {
                    "chroma_noise_score": 0.13,
                    "background_mottling_score": 0.06,
                    "chroma_load": 0.0091,
                    "chroma_load_growth": 1.93,
                    "chroma_load_growth_low_absolute_exempted": True,
                },
                "limits": limits,
            },
        }
        feedback = {
            "name": "cand_b_feedback_1",
            "stem": "stage7_cand_b_feedback_1",
            "status": "ok",
            "allowed_as_final": True,
            "diagnostics": [],
            "risk_score": 0.218,
            "background_quality_gate": {
                "metrics": {
                    "chroma_noise_score": 0.14,
                    "background_mottling_score": 0.07,
                    "chroma_load": 0.020,
                    "chroma_load_growth": 1.33,
                    "chroma_load_growth_low_absolute_exempted": False,
                },
                "limits": limits,
            },
        }

        selected = min(
            [feedback, exempted_primary],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "cand_a")

    def test_stage7_ic434_exempted_chroma_ranks_by_subject_retention(self):
        processor = pipeline_module.StarunPostProcessor()

        def attempt(name: str, *, visibility: float, signal: float, span: float, load: float):
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": True,
                "diagnostics": [],
                "advisories": [],
                "risk_score": 1.0,
                "preview_target_attainment": {
                    "attainment_ratio": 1.0,
                    "minimum_ratio": 0.90,
                    "hard_minimum_ratio": 0.80,
                    "maximum_ratio": 1.50,
                    "hard_maximum_ratio": 2.25,
                },
                "preview_retention": {
                    "metrics": {
                        "visibility": {"available": True, "ranking_ratio": visibility},
                        "object_signal": {"available": True, "ranking_ratio": signal},
                        "subject_span": {"available": True, "ranking_ratio": span},
                        "saturation_p95": {"available": True, "ranking_ratio": 0.7},
                        "microcontrast": {"available": True, "ranking_ratio": 0.7},
                    }
                },
                "background_quality_gate": {
                    "metrics": {
                        "chroma_noise_score": 0.12,
                        "background_mottling_score": 0.08,
                        "chroma_load": load,
                        "chroma_load_growth": 1.80,
                        "chroma_load_growth_low_absolute_exempted": True,
                        "chroma_load_low_absolute_effective_max": 0.06,
                    },
                    "limits": {
                        "chroma_noise_score_max": 0.34,
                        "background_mottling_score_max": 0.45,
                        "chroma_load_growth_max": 1.37,
                        "chroma_load_signal_excluded_max": 0.06,
                    },
                },
            }

        cand_a = attempt(
            "cand_a", visibility=0.66, signal=0.62, span=0.58, load=0.020
        )
        cand_b = attempt(
            "cand_b", visibility=0.82, signal=0.84, span=0.86, load=0.055
        )

        selected = min(
            [cand_a, cand_b],
            key=processor._stage7_candidate_selection_key,
        )
        self.assertEqual(selected["name"], "cand_b")

    def test_stage7_m8_quality_vector_prefers_cand_a_over_quantile(self):
        processor = pipeline_module.StarunPostProcessor()
        limits = {
            "chroma_noise_score_max": 0.34,
            "background_mottling_score_max": 0.45,
            "chroma_load_growth_max": 1.35,
            "chroma_load_low_absolute_max": 0.05,
            "chroma_load_low_absolute_effective_max": 0.0505,
        }

        def attempt(
            name: str,
            method: str,
            *,
            chroma: float,
            mottling: float,
            load: float,
            growth: float,
            ratio: float,
            risk: float,
        ) -> dict[str, Any]:
            return {
                "name": name,
                "method": method,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": False,
                "diagnostics": [
                    f"background_chroma_load_growth {growth:.3f}>1.350"
                ],
                "risk_score": risk,
                "preview_target_attainment": {
                    "attainment_ratio": ratio,
                    "minimum_ratio": 0.55,
                    "maximum_ratio": 1.50,
                },
                "background_quality_gate": {
                    "metrics": {
                        "chroma_noise_score": chroma,
                        "background_mottling_score": mottling,
                        "chroma_load": load,
                        "chroma_load_growth": growth,
                        "chroma_load_growth_low_absolute_exempted": False,
                    },
                    "limits": limits,
                },
            }

        cand_a = attempt(
            "cand_a",
            "bright_nebula_hdr_masked",
            chroma=0.01908455597003922,
            mottling=0.031012796495016048,
            load=0.05020237532146765,
            growth=1.850384273583177,
            ratio=1.0783284313766188,
            risk=5.044224199175751,
        )
        quantile = attempt(
            "cand_quantile",
            "adaptive_quantile",
            chroma=0.027829980551090393,
            mottling=0.031160389350340323,
            load=0.15822902378627346,
            growth=5.832084545078043,
            ratio=0.9999382599634722,
            risk=5.021536350767645,
        )

        selected = min(
            [quantile, cand_a],
            key=processor._stage7_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "cand_a")
        self.assertTrue(processor._stage7_attempt_allows_chroma_rescue(cand_a))
        self.assertFalse(processor._stage7_attempt_allows_chroma_rescue(quantile))

    def test_stage7_review_selection_prefers_quality_before_preview_ratio(self):
        processor = pipeline_module.StarunPostProcessor()

        def review_attempt(
            name: str,
            ratio: float,
            diagnostics: list[str],
            *,
            local_ok: bool = True,
        ) -> dict[str, Any]:
            return {
                "name": name,
                "stem": f"stage7_{name}",
                "status": "ok",
                "allowed_as_final": False,
                "diagnostics": diagnostics,
                "risk_score": 1.0,
                "pixel_stats": {
                    "p50": ratio * 0.10,
                    "p99": 0.80,
                    "dynamic_range": 0.79,
                },
                "preview_target_attainment": {
                    "attainment_ratio": ratio,
                },
                "target_local_quality": {"accepted": local_ok},
            }

        candidate_a = review_attempt(
            "cand_a",
            1.19,
            ["background_chroma_noise_score 0.431>0.340"],
        )
        candidate_b = review_attempt(
            "cand_b",
            3.83,
            [
                "background_chroma_noise_score 0.401>0.340",
                "preview_target_p50_ratio_above_max 3.830>1.500",
            ],
        )
        unsafe_lower_ratio = review_attempt(
            "unsafe_core",
            1.05,
            ["background_chroma_noise_score 0.410>0.340"],
            local_ok=False,
        )

        safe_candidates = [
            attempt
            for attempt in (candidate_b, unsafe_lower_ratio, candidate_a)
            if processor._stage7_review_candidate_is_safe(attempt)
        ]
        selected = min(
            safe_candidates,
            key=processor._stage7_review_candidate_selection_key,
        )

        self.assertEqual(selected["name"], "cand_a")
        self.assertNotIn(unsafe_lower_ratio, safe_candidates)

    def test_stage7_chroma_rescue_preserves_luminance_and_signal_region(self):
        processor = pipeline_module.StarunPostProcessor()
        rgb = np.empty((3, 8, 8), dtype=np.float32)
        rgb[0] = 0.34
        rgb[1] = 0.22
        rgb[2] = 0.30
        luma = (
            0.2126 * rgb[0]
            + 0.7152 * rgb[1]
            + 0.0722 * rgb[2]
        ).astype(np.float32)
        background_mask = np.ones((8, 8), dtype=np.float32)
        background_mask[3:5, 3:5] = 0.0
        processor._stage8_generate_starless_masks = lambda _image: {
            "rgb": rgb.copy(),
            "gray": luma.copy(),
            "background_mask": background_mask,
            "coverage": {
                "core": 4.0 / 64.0,
                "nebula": 0.0,
                "faint_nebula": 0.0,
            },
        }
        processor._stage8_restore_rgb_like = (
            lambda source, rescued_rgb: rescued_rgb.astype(source.dtype, copy=False)
        )

        rescued, metadata = processor._stage7_background_chroma_rescue_pixels(
            rgb,
            strength=0.55,
        )
        rescued_luma = (
            0.2126 * rescued[0]
            + 0.7152 * rescued[1]
            + 0.0722 * rescued[2]
        )
        before_chroma = np.std(rgb[:, 0, 0] - luma[0, 0])
        after_chroma = np.std(rescued[:, 0, 0] - rescued_luma[0, 0])

        self.assertTrue(np.allclose(rescued_luma, luma, atol=1e-6))
        self.assertLess(after_chroma, before_chroma * 0.50)
        self.assertTrue(np.allclose(rescued[:, 3:5, 3:5], rgb[:, 3:5, 3:5]))
        self.assertTrue(metadata["luminance_preserved"])

    def test_stage7_background_gate_rejects_case_candidates_a_and_b(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.003310588,
            "background_mottling_score": 0.000616090,
            "bg_std": 0.000019856,
            "bg_median": 0.000494266,
        }
        candidate_a = {
            "chroma_noise_score": 0.430704,
            "background_mottling_score": 0.078502,
            "bg_std": 0.00253955,
            "bg_median": 0.03274113,
        }
        candidate_b = {
            "chroma_noise_score": 0.795766,
            "background_mottling_score": 0.20,
            "bg_std": 0.00941930,
            "bg_median": 0.19506431,
        }

        gate_a = processor._stage7_stretch_background_gate(baseline, candidate_a)
        gate_b = processor._stage7_stretch_background_gate(baseline, candidate_b)

        self.assertTrue(gate_a["accepted"])
        self.assertFalse(gate_b["accepted"])
        self.assertTrue(
            any(
                "background_chroma_noise_score" in advisory
                for advisory in gate_a["advisories"]
            )
        )
        self.assertTrue(
            any(
                "background_chroma_load_growth" in advisory
                for advisory in gate_a["advisories"]
            )
        )
        self.assertTrue(
            any("background_chroma_noise_score" in issue for issue in gate_b["issues"])
        )

    def test_background_quality_weight_excludes_galaxy_signal_mask(self):
        background = np.ones((32, 32), dtype=np.float32)
        galaxy_signal = np.zeros((32, 32), dtype=np.float32)
        galaxy_signal[8:24, 8:24] = 1.0

        weight = pipeline_module.stage8_pixels._stage8_exclusive_background_weight(
            {
                "background_mask": background,
                "galaxy_signal_mask": galaxy_signal,
            },
            background,
        )

        self.assertEqual(float(np.max(weight[8:24, 8:24])), 0.0)
        self.assertEqual(float(np.min(weight[:6, :6])), 1.0)

    def test_stage7_background_gate_prefers_direct_chroma_load_metric(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.02,
            "background_chroma_load": 0.80,
            "background_mottling_score": 0.01,
            "bg_std": 0.001,
            "bg_median": 0.002,
        }
        candidate = {
            "chroma_noise_score": 0.03,
            "background_chroma_load": 0.60,
            "background_mottling_score": 0.02,
            "bg_std": 0.005,
            "bg_median": 0.05,
        }

        gate = processor._stage7_stretch_background_gate(baseline, candidate)

        self.assertTrue(gate["accepted"], gate)
        self.assertAlmostEqual(gate["metrics"]["chroma_load"], 0.60)
        self.assertAlmostEqual(gate["metrics"]["chroma_load_growth"], 0.75)

    def test_stage7_uncalibrated_smooth_background_cast_requires_review(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.color_calibration_report = {
            "method": "PRESERVE_INPUT",
            "physical_color": {
                "accepted": False,
                "method": "PRESERVE_INPUT",
            },
        }
        selected = {
            "background_quality_gate": {
                "metrics": {
                    "chroma_load": 0.189,
                    "signal_exclusion_applied": True,
                    "background_red_mean": 0.041,
                    "background_green_mean": 0.058,
                    "background_blue_mean": 0.040,
                    "background_green_excess": 0.31,
                }
            }
        }

        gate = processor._stage7_uncalibrated_background_color_review_gate(
            selected
        )

        self.assertTrue(gate["applicable"])
        self.assertTrue(gate["requires_review"])
        self.assertEqual(gate["status"], "review_required")
        self.assertEqual(
            gate["reason_code"],
            "uncalibrated_background_chroma_load_exceeded",
        )
        self.assertAlmostEqual(gate["value"], 0.189)
        self.assertAlmostEqual(gate["limit"], 0.12)
        self.assertFalse(gate["global_white_balance_applied"])
        self.assertTrue(gate["global_white_balance_prohibited"])

    def test_stage7_background_cast_review_gate_skips_accepted_physical_color(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.color_calibration_report = {
            "method": "SPCC",
            "physical_color": {"accepted": True, "method": "SPCC"},
        }
        selected = {
            "background_quality_gate": {
                "metrics": {
                    "chroma_load": 0.191,
                    "signal_exclusion_applied": True,
                }
            }
        }

        gate = processor._stage7_uncalibrated_background_color_review_gate(
            selected
        )

        self.assertFalse(gate["applicable"])
        self.assertFalse(gate["requires_review"])
        self.assertEqual(gate["status"], "not_applicable")

    def test_stage7_failed_physical_method_name_does_not_bypass_cast_gate(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.color_calibration_report = {
            "method": "SPCC",
            "physical_color": {"method": "SPCC"},
        }
        selected = {
            "background_quality_gate": {
                "metrics": {
                    "chroma_load": 0.189,
                    "signal_exclusion_applied": True,
                }
            }
        }

        gate = processor._stage7_uncalibrated_background_color_review_gate(
            selected
        )

        self.assertTrue(gate["applicable"])
        self.assertTrue(gate["requires_review"])
        self.assertEqual(
            gate["reason_code"],
            "uncalibrated_background_chroma_load_exceeded",
        )

    def test_stage7_uncalibrated_background_cast_within_limit_can_continue(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._channel_semantics = "broadband_rgb_osc"
        processor.color_calibration_report = {
            "method": "PRESERVE_INPUT",
            "physical_color": {"accepted": False},
        }
        selected = {
            "background_quality_gate": {
                "metrics": {
                    "chroma_load": 0.119,
                    "signal_exclusion_applied": True,
                }
            }
        }

        gate = processor._stage7_uncalibrated_background_color_review_gate(
            selected
        )

        self.assertTrue(gate["applicable"])
        self.assertFalse(gate["requires_review"])
        self.assertEqual(gate["status"], "ok")

    def test_stage7_background_metrics_can_reuse_frozen_source_coordinates(self):
        processor = self._new_processor()
        candidate = np.full((3, 32, 32), 0.02, dtype=np.float32)
        candidate[0, :, :16] = 0.20
        candidate[1, :, :16] = 0.05
        candidate[2, :, :16] = 0.05
        frozen_mask = np.zeros((32, 32), dtype=np.float32)
        frozen_mask[:, :16] = 1.0

        candidate_local = (
            pipeline_module.StarunPostProcessor._background_quality_metrics(
                processor,
                candidate,
            )
        )
        frozen = pipeline_module.StarunPostProcessor._background_quality_metrics(
            processor,
            candidate,
            {"background_mask": frozen_mask},
        )

        self.assertGreater(
            frozen["background_chroma_load"],
            candidate_local["background_chroma_load"] + 0.25,
        )

    def test_stage7_diffuse_visibility_gate_warns_then_rejects_near_black_candidate(self):
        processor = pipeline_module.StarunPostProcessor()
        target = {"name": "widefield_nebulosity"}

        advisory = processor._stage7_candidate_visibility_gate(
            {"safe_preview_visibility_score": 0.060},
            target,
            {},
        )
        rejected = processor._stage7_candidate_visibility_gate(
            {"safe_preview_visibility_score": 0.040},
            target,
            {},
        )
        accepted = processor._stage7_candidate_visibility_gate(
            {"safe_preview_visibility_score": 0.080},
            target,
            {},
        )

        self.assertTrue(advisory["accepted"])
        self.assertTrue(advisory["advisories"])
        self.assertFalse(rejected["accepted"])
        self.assertIn("visibility_score_below_minimum", rejected["issues"][0])
        self.assertTrue(accepted["accepted"])

    def test_stage7_preview_visibility_retention_has_60_40_contract(self):
        processor = pipeline_module.StarunPostProcessor()
        target = {"name": "generic_balanced"}

        def gate(ratio: float):
            return processor._stage7_candidate_visibility_gate(
                {"safe_preview_visibility_score": 0.20},
                target,
                {
                    "metrics": {
                        "visibility": {
                            "available": True,
                            "ratio": ratio,
                            "ranking_ratio": min(ratio, 1.0),
                        }
                    }
                },
            )

        self.assertEqual(gate(0.60)["status"], "ok")
        self.assertEqual(gate(0.40)["status"], "advisory")
        rejected = gate(0.399)
        self.assertFalse(rejected["accepted"])
        self.assertIn(
            "preview_visibility_retention_below_minimum",
            rejected["issues"][0],
        )

    def test_stage7_seestar198_visibility_rejects_a_and_accepts_b(self):
        processor = pipeline_module.StarunPostProcessor()
        target = {"name": "generic_balanced"}

        def gate(ratio: float):
            return processor._stage7_candidate_visibility_gate(
                {"safe_preview_visibility_score": 0.60 * ratio},
                target,
                {
                    "metrics": {
                        "visibility": {
                            "available": True,
                            "ratio": ratio,
                            "ranking_ratio": min(ratio, 1.0),
                        }
                    }
                },
            )

        self.assertFalse(gate(0.038)["accepted"])
        self.assertTrue(gate(0.708)["accepted"])

    def test_stage7_ngc7000_uses_visible_chroma_only_b_as_rescue_parent(self):
        processor = pipeline_module.StarunPostProcessor()
        target = {"name": "widefield_nebulosity"}
        visible_b = processor._stage7_candidate_visibility_gate(
            {"safe_preview_visibility_score": 0.55},
            target,
            {
                "metrics": {
                    "visibility": {
                        "available": True,
                        "ratio": 1.06,
                        "ranking_ratio": 1.0,
                    }
                }
            },
        )
        dark_a = processor._stage7_candidate_visibility_gate(
            {"safe_preview_visibility_score": 0.05},
            target,
            {
                "metrics": {
                    "visibility": {
                        "available": True,
                        "ratio": 0.097,
                        "ranking_ratio": 0.097,
                    }
                }
            },
        )
        cand_b = {
            "status": "ok",
            "stem": "stage7_cand_b",
            "method": "linked_mtf",
            "target_local_quality": {"accepted": True},
            "diagnostics": [
                "background_chroma_load_growth 42.959>1.370",
            ],
        }

        self.assertFalse(dark_a["accepted"])
        self.assertTrue(visible_b["accepted"])
        self.assertTrue(processor._stage7_attempt_allows_chroma_rescue(cand_b))

    def test_stage7_visibility_contract_warns_rescue_style_final_path(self):
        processor = pipeline_module.StarunPostProcessor()

        quality_ok, issues, gate = (
            processor._stage7_apply_candidate_visibility_gate(
                True,
                ["prior_advisory"],
                {"safe_preview_visibility_score": 0.060},
                {"name": "dark_nebula_separation"},
                {},
            )
        )

        self.assertTrue(quality_ok)
        self.assertTrue(gate["accepted"])
        self.assertEqual(issues[0], "prior_advisory")
        self.assertTrue(
            any(
                advisory.startswith("visibility_score_below_minimum")
                for advisory in gate["advisories"]
            )
        )

    def test_stage7_background_gate_accepts_low_absolute_load_after_signal_exclusion(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.0004317982075008331,
            "background_chroma_load": 0.003073794089676055,
            "background_mottling_score": 0.0007385105315430943,
            "bg_std": 0.00015994571731425822,
            "bg_median": 0.018843578174710274,
        }
        candidate = {
            "chroma_noise_score": 0.03353840518828569,
            "background_chroma_load": 0.052487816404259584,
            "background_mottling_score": 0.045088194926023534,
            "bg_std": 0.01180883590131998,
            "bg_median": 0.15154215693473816,
        }

        without_exclusion = processor._stage7_stretch_background_gate(
            baseline,
            candidate,
        )
        with_exclusion = processor._stage7_stretch_background_gate(
            baseline,
            {**candidate, "signal_exclusion_applied": True},
        )

        self.assertFalse(without_exclusion["accepted"])
        self.assertTrue(with_exclusion["accepted"], with_exclusion)
        self.assertTrue(
            with_exclusion["metrics"][
                "chroma_load_growth_signal_excluded_exempted"
            ]
        )
        self.assertAlmostEqual(
            with_exclusion["metrics"]["chroma_load_low_absolute_effective_max"],
            0.06,
        )

    def test_stage7_background_gate_checks_mottling_and_accepts_safe_candidate(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.0033,
            "background_mottling_score": 0.001,
            "bg_std": 0.00002,
            "bg_median": 0.0005,
        }
        safe = {
            "chroma_noise_score": 0.20,
            "background_mottling_score": 0.20,
            "bg_std": 0.003,
            "bg_median": 0.05,
        }
        mottled = {**safe, "background_mottling_score": 0.60}

        self.assertTrue(
            processor._stage7_stretch_background_gate(baseline, safe)["accepted"]
        )
        mottled_gate = processor._stage7_stretch_background_gate(baseline, mottled)
        self.assertTrue(mottled_gate["accepted"])
        self.assertTrue(
            any(
                "background_mottling_score" in advisory
                for advisory in mottled_gate["advisories"]
            )
        )
        rejected_gate = processor._stage7_stretch_background_gate(
            baseline,
            {**safe, "background_mottling_score": 0.70},
        )
        self.assertFalse(rejected_gate["accepted"])
        self.assertTrue(
            any(
                "background_mottling_score" in issue
                for issue in rejected_gate["issues"]
            )
        )

    def test_stage7_background_gate_exempts_low_absolute_load_in_extreme_low_background(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.0012268481441424228,
            "background_mottling_score": 0.0004914217773451431,
            "bg_std": 0.000014195245057635475,
            "bg_median": 0.0005103948060423136,
        }
        candidate_a = {
            "chroma_noise_score": 0.15973878325894475,
            "background_mottling_score": 0.06247950174535314,
            "bg_std": 0.001812034985050559,
            "bg_median": 0.03351139277219772,
        }
        candidate_b = {
            "chroma_noise_score": 0.4499879052243117,
            "background_mottling_score": 0.1006566743036724,
            "bg_std": 0.007080338895320892,
            "bg_median": 0.18866348266601562,
        }

        gate_a = processor._stage7_stretch_background_gate(baseline, candidate_a)
        gate_b = processor._stage7_stretch_background_gate(baseline, candidate_b)

        self.assertTrue(gate_a["accepted"])
        self.assertAlmostEqual(gate_a["metrics"]["chroma_load"], 0.04766700815594566)
        self.assertTrue(
            gate_a["metrics"]["chroma_load_growth_low_absolute_exempted"]
        )
        self.assertTrue(gate_a["metrics"]["extreme_low_background"])
        self.assertTrue(gate_b["accepted"])
        self.assertTrue(
            any(
                "background_chroma_noise_score" in advisory
                for advisory in gate_b["advisories"]
            )
        )
        rejected = processor._stage7_stretch_background_gate(
            baseline,
            {**candidate_b, "chroma_noise_score": 0.52},
        )
        self.assertFalse(rejected["accepted"])
        self.assertTrue(
            any(
                "background_chroma_noise_score" in issue
                for issue in rejected["issues"]
            )
        )

    def test_stage7_background_gate_accepts_m8_boundary_with_numeric_tolerance(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.00037741565392934717,
            "background_chroma_load": 0.027130783609748937,
            "background_mottling_score": 0.0008186877948901383,
            "bg_std": 0.00007896655006334186,
            "bg_median": 0.0023227420169860125,
        }
        m8_cand_a = {
            "chroma_noise_score": 0.01908455597003922,
            "background_chroma_load": 0.05020237532146765,
            "background_mottling_score": 0.031012796495016048,
            "bg_std": 0.00396227091550827,
            "bg_median": 0.06262941658496857,
        }

        gate = processor._stage7_stretch_background_gate(baseline, m8_cand_a)

        self.assertTrue(gate["accepted"], gate)
        self.assertTrue(
            gate["metrics"]["chroma_load_growth_low_absolute_exempted"]
        )
        self.assertAlmostEqual(
            gate["limits"]["chroma_load_low_absolute_effective_max"],
            0.0505,
        )

        clearly_high = {
            **m8_cand_a,
            "background_chroma_load": 0.055,
        }
        advisory = processor._stage7_stretch_background_gate(
            baseline,
            clearly_high,
        )
        self.assertTrue(advisory["accepted"])
        self.assertFalse(
            advisory["metrics"]["chroma_load_growth_low_absolute_exempted"]
        )
        self.assertTrue(advisory["advisories"])

        rejected = processor._stage7_stretch_background_gate(
            baseline,
            {**m8_cand_a, "background_chroma_load": 0.060},
        )
        self.assertFalse(rejected["accepted"])

    def test_stage7_background_gate_still_rejects_high_absolute_load_growth(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.10,
            "background_mottling_score": 0.01,
            "bg_std": 0.002,
            "bg_median": 0.05,
        }
        candidate = {
            "chroma_noise_score": 0.30,
            "background_mottling_score": 0.10,
            "bg_std": 0.004,
            "bg_median": 0.03,
        }

        gate = processor._stage7_stretch_background_gate(baseline, candidate)

        self.assertFalse(gate["accepted"])
        self.assertFalse(gate["metrics"]["chroma_load_growth_low_absolute_exempted"])
        self.assertTrue(
            any("background_chroma_load_growth" in issue for issue in gate["issues"])
        )

    def test_stage7_background_gate_does_not_exempt_normal_background_growth(self):
        processor = pipeline_module.StarunPostProcessor()
        baseline = {
            "chroma_noise_score": 0.10,
            "background_mottling_score": 0.01,
            "bg_std": 0.002,
            "bg_median": 0.05,
        }
        candidate = {
            "chroma_noise_score": 0.20,
            "background_mottling_score": 0.10,
            "bg_std": 0.003,
            "bg_median": 0.05,
        }

        gate = processor._stage7_stretch_background_gate(baseline, candidate)

        self.assertLessEqual(
            gate["metrics"]["chroma_load"],
            gate["limits"]["chroma_load_low_absolute_max"],
        )
        self.assertFalse(gate["metrics"]["extreme_low_background"])
        self.assertTrue(gate["accepted"])
        self.assertTrue(
            any(
                "background_chroma_load_growth" in advisory
                for advisory in gate["advisories"]
            )
        )
        rejected = processor._stage7_stretch_background_gate(
            baseline,
            {**candidate, "background_chroma_load": 0.050},
        )
        self.assertFalse(rejected["accepted"])
        self.assertTrue(
            any(
                "background_chroma_load_growth" in issue
                for issue in rejected["issues"]
            )
        )

    def test_stage7_pixel_repair_accepts_significant_chroma_reduction(self):
        cfg = pipeline_module.PipelineConfig()
        assessment = stage6_star_separation_module._stage7_chroma_repair_acceptance(
            cfg,
            {"chroma_noise_score": 0.003310588},
            {"chroma_noise_score": 0.001822228},
            residual_not_worse=True,
            halo_not_worse=True,
        )

        self.assertTrue(assessment["accepted"])
        self.assertGreater(assessment["reduction_ratio"], 0.40)

    def test_stage7_pixel_repair_rejects_chroma_gain_when_halo_worsens(self):
        cfg = pipeline_module.PipelineConfig()
        assessment = stage6_star_separation_module._stage7_chroma_repair_acceptance(
            cfg,
            {"chroma_noise_score": 0.003310588},
            {"chroma_noise_score": 0.001822228},
            residual_not_worse=True,
            halo_not_worse=False,
        )

        self.assertFalse(assessment["accepted"])

    def test_stage7_does_not_accept_degraded_candidate_for_stage8(self):
        processor = self._new_processor()
        processor._run_stage7_stretching_candidates = lambda: (
            True,
            True,
            ["quality_ok=false"],
            "Asinh",
        )

        pipeline_module.run_stage7_stretching(processor)

        self.assertFalse(processor._stage7_stretch_accepted)
        self.assertIsNone(processor._stage7_stretch_output)

    def test_stage7_dynamic_range_gate_accepts_peak_well_above_extreme_background(self):
        cfg = pipeline_module.PipelineConfig()

        assessment = stage7_quality_module.stage7_dynamic_range_assessment(
            cfg,
            dynamic_range_ratio=0.09246493544624469,
            peak_signal=0.0051499465480446815,
            background_level=0.000503482879139483,
        )

        self.assertFalse(assessment["collapsed"])
        self.assertGreater(assessment["peak_background_ratio"], 10.0)

    def test_stage7_dynamic_range_gate_rejects_flat_low_signal_output(self):
        cfg = pipeline_module.PipelineConfig()

        assessment = stage7_quality_module.stage7_dynamic_range_assessment(
            cfg,
            dynamic_range_ratio=0.09,
            peak_signal=0.001,
            background_level=0.0005,
        )

        self.assertTrue(assessment["collapsed"])
        self.assertTrue(assessment["hard_failed"])
        self.assertFalse(assessment["advisory"])
        self.assertEqual(assessment["peak_background_ratio"], 2.0)

    def test_stage7_quality_gate_keeps_two_x_abnormality_advisory_only(self):
        cfg = pipeline_module.PipelineConfig()

        upper_advisory = stage7_quality_module.stage7_upper_quality_gate(
            cfg,
            value=0.90,
            accepted_limit=0.45,
        )
        upper_hard = stage7_quality_module.stage7_upper_quality_gate(
            cfg,
            value=0.901,
            accepted_limit=0.45,
        )
        lower_advisory = stage7_quality_module.stage7_lower_quality_gate(
            cfg,
            value=0.175,
            accepted_limit=0.35,
        )
        lower_hard = stage7_quality_module.stage7_lower_quality_gate(
            cfg,
            value=0.174,
            accepted_limit=0.35,
        )

        self.assertEqual(upper_advisory["status"], "advisory")
        self.assertEqual(lower_advisory["status"], "advisory")
        self.assertTrue(upper_hard["hard_failed"])
        self.assertTrue(lower_hard["hard_failed"])

    def test_stage7_dynamic_range_gate_keeps_moderate_collapse_advisory_only(self):
        cfg = pipeline_module.PipelineConfig()

        assessment = stage7_quality_module.stage7_dynamic_range_assessment(
            cfg,
            dynamic_range_ratio=0.30,
            peak_signal=0.004,
            background_level=0.0012,
        )

        self.assertTrue(assessment["collapsed"])
        self.assertTrue(assessment["advisory"])
        self.assertFalse(assessment["hard_failed"])

    def test_stage7_dynamic_collapse_skips_same_model_parameter_retries(self):
        self.assertEqual(
            stage6_star_separation_module._syqon_quality_failure_codes(
                ["dynamic_range_collapse", "compact_halo_residue"]
            ),
            ["DYNAMIC_RANGE_COLLAPSE", "HALO"],
        )

    def test_stage7_repair_rolls_back_when_compact_halo_improves_but_global_halo_worsens(self):
        before = {
            "status": "poor",
            "derived": {
                "residual_star_score": 0.10,
                "global_residual_star_score": 0.10,
                "compact_residual_star_score": 0.10,
                "halo_residue_score": 0.12,
                "global_halo_residue_score": 0.12,
                "compact_halo_residue_score": 0.20,
                "galaxy_disk_halo_residue_score": 0.08,
                "black_hole_score": 0.01,
                "starless_dynamic_range_ratio": 0.70,
            },
        }
        after = {
            "status": "poor",
            "derived": {
                "residual_star_score": 0.10,
                "global_residual_star_score": 0.10,
                "compact_residual_star_score": 0.10,
                "halo_residue_score": 0.12,
                "global_halo_residue_score": 0.123,
                "compact_halo_residue_score": 0.19,
                "galaxy_disk_halo_residue_score": 0.08,
                "black_hole_score": 0.01,
                "starless_dynamic_range_ratio": 0.70,
            },
        }

        result = stage6_star_separation_module._stage7_repair_non_regression(
            before,
            after,
        )

        self.assertFalse(result["accepted"])
        self.assertIn("global_halo_residue_score", result["violations"])
        self.assertTrue(
            stage6_star_separation_module._stage7_trigger_improvement(
                before,
                after,
                ["compact_halo_residue"],
            )["accepted"]
        )

    def test_stage7_calibrated_dynamic_range_does_not_trigger_refinement(self):
        processor = pipeline_module.StarunPostProcessor()
        quality = {
            "derived": {
                "residual_star_score": 0.0,
                "halo_residue_score": 0.03,
                "black_hole_score": 0.0,
                "starless_dynamic_range_ratio": 0.09246493544624469,
                "starless_peak_signal": 0.0051499465480446815,
                "dynamic_range_collapse": False,
            }
        }

        self.assertEqual(processor._stage7_repair_triggers(quality), [])

    def test_stage6_galaxy_roi_detects_halo_hidden_on_bright_disk(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._active_target_type = lambda: "large_galaxy"
        source, starless, starmask = self._synthetic_galaxy_starless_layers(
            disk_halo_amplitude=0.020,
        )

        scores = stage7_quality_module.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )

        self.assertLess(
            scores["global_halo_residue_score"],
            processor.cfg.stage7_halo_residue_score_max,
        )
        self.assertEqual(scores["galaxy_disk_halo_evidence_available"], 1.0)
        self.assertGreater(
            scores["galaxy_disk_halo_residue_score"],
            processor.cfg.stage7_halo_residue_score_max,
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        processor.stretched_name = "source"
        processor.starmask_file = Path(temp_dir.name) / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._read_image_by_stem = lambda stem: {
            "source": source,
            "starless": starless,
            "starmask": starmask,
        }.get(stem)
        quality = processor._stage7_quality_assessment(
            "galaxy-disk-halo",
            tool_label="synthetic",
            source_stem="source",
        )
        self.assertEqual(quality["status"], "ok")
        self.assertFalse(quality["issues"], quality)
        self.assertTrue(
            any(
                advisory.startswith("galaxy_disk_halo_residue ")
                for advisory in quality["advisories"]
            ),
            quality,
        )
        self.assertEqual(
            quality["derived"]["halo_residue_score"],
            quality["derived"]["galaxy_disk_halo_residue_score"],
        )

    def test_stage6_single_galaxy_disk_halo_point_cannot_hard_reject_alone(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._active_target_type = lambda: "large_galaxy"
        source, starless, starmask = self._synthetic_galaxy_starless_layers(
            disk_halo_amplitude=0.030,
        )
        scores = stage7_quality_module.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )
        self.assertEqual(scores["galaxy_disk_halo_corroborated_local_count"], 1)
        self.assertGreater(
            scores["galaxy_disk_halo_residue_score"],
            processor.cfg.stage7_large_galaxy_halo_residue_score_max * 2.0,
        )

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        processor.stretched_name = "source"
        processor.starmask_file = Path(temp_dir.name) / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._read_image_by_stem = lambda stem: {
            "source": source,
            "starless": starless,
            "starmask": starmask,
        }.get(stem)

        quality = processor._stage7_quality_assessment(
            "galaxy-disk-single-point",
            tool_label="synthetic",
            source_stem="source",
        )

        self.assertEqual(quality["status"], "ok")
        self.assertFalse(quality["issues"], quality)
        gate = quality["quality_gates"]["galaxy_disk_halo_residue"]
        self.assertEqual(gate["status"], "advisory")
        self.assertEqual(
            gate["reason_code"],
            "single_local_galaxy_halo_evidence",
        )
        failure = (
            stage6_star_separation_module._stage6_quality_hard_failure_summary(
                processor,
                quality,
            )
        )
        self.assertFalse(failure["hard_failed"], failure)
        self.assertNotIn("HALO", failure["failure_codes"])
        self.assertNotIn(
            "halo_residue",
            stage7_quality_module.stage7_repair_triggers(processor, quality),
        )
        handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
            processor,
            quality,
            [],
            separation_accepted=True,
        )
        self.assertEqual(handoff["processing_policy"], "full")
        self.assertEqual(handoff["reason_code"], "")
        self.assertTrue(
            any(
                advisory.startswith("galaxy_disk_halo_residue ")
                for advisory in handoff["advisories"]
            ),
            handoff,
        )
        self.assertEqual(
            handoff["suppressed_advisories"],
            handoff["advisories"],
        )
        self.assertLessEqual(
            handoff["metrics"]["effective_halo_residue_score"],
            processor._stage7_effective_halo_threshold(),
        )
        processor.process_dir = Path(temp_dir.name)
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor._stage7_selected_quality = quality
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")
        stage9_reason = processor._stage9_bad_starless_reason()
        self.assertNotIn("stage7_halo_residue_score", stage9_reason)
        self.assertTrue(
            any(
                advisory.startswith("galaxy_disk_halo_residue ")
                for advisory in processor._stage9_starless_advisories
            ),
            processor._stage9_starless_advisories,
        )

    def test_stage6_galaxy_roi_rejects_one_sided_disk_structure_as_halo(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._active_target_type = lambda: "large_galaxy"
        source, starless, starmask = self._synthetic_galaxy_starless_layers()
        yy, xx = np.mgrid[:256, :256]
        one_sided = (
            0.030
            * np.exp(-((xx - 158) ** 2 + (yy - 145) ** 2) / 45.0)
            * (xx >= 158)
        ).astype(np.float32)
        starless = np.clip(starless + one_sided[None, :, :], 0.0, 1.0)

        scores = stage7_quality_module.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )

        self.assertGreater(scores["galaxy_disk_halo_raw_local_count"], 0)
        self.assertEqual(
            scores["galaxy_disk_halo_corroborated_local_count"],
            0,
        )
        self.assertEqual(scores["galaxy_disk_halo_evidence_available"], 0.0)
        self.assertLess(
            scores["galaxy_disk_halo_residue_score"],
            processor.cfg.stage7_large_galaxy_halo_residue_score_max,
        )

    def test_stage6_galaxy_roi_rejects_removed_bright_core(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._active_target_type = lambda: "large_galaxy"
        source, starless, starmask = self._synthetic_galaxy_starless_layers(
            core_damage=True,
        )

        scores = stage7_quality_module.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )
        quality = {
            "derived": {
                "residual_star_score": 0.0,
                "halo_residue_score": 0.0,
                "compact_halo_residue_score": 0.0,
                "black_hole_score": 0.0,
                "starless_dynamic_range_ratio": 1.0,
                "starless_peak_signal": 1.0,
                **scores,
            }
        }

        self.assertLess(
            scores["galaxy_core_preservation_ratio"],
            processor.cfg.stage7_galaxy_core_preservation_ratio_min,
        )
        self.assertLess(
            scores["galaxy_core_contrast_ratio"],
            processor.cfg.stage7_galaxy_core_contrast_ratio_min,
        )
        self.assertIn("galaxy_core_damage", processor._stage7_repair_triggers(quality))

    def test_stage6_bright_nebula_advisory_halo_triggers_pixel_repair(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        quality = {
            "status": "ok",
            "derived": {
                "halo_residue_score": 0.488,
                "compact_halo_residue_score": 0.493,
            },
        }

        trigger = (
            stage6_star_separation_module._stage7_starless_pixel_repair_trigger(
                processor,
                quality,
            )
        )

        self.assertTrue(trigger["triggered"])
        self.assertEqual(trigger["reason"], "bright_nebula_halo_advisory")
        self.assertTrue(trigger["within_target_limit"])
        self.assertAlmostEqual(trigger["measured_halo_score"], 0.493)

    def test_stage6_stage8_handoff_uses_three_level_bright_nebula_gate(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"

        cases = (
            (0.3500, 0.3400, "full"),
            (0.4880, 0.4930, "limited"),
            (0.6000, 0.5900, "limited"),
            (0.6001, 0.5900, "limited"),
            (0.3000, 0.6001, "limited"),
            (1.2001, 0.5900, "limited"),
            (0.3000, 1.2001, "limited"),
        )
        for global_halo, compact_halo, expected_policy in cases:
            with self.subTest(
                global_halo=global_halo,
                compact_halo=compact_halo,
            ):
                quality = {
                    "status": "ok",
                    "derived": {
                        "halo_residue_score": global_halo,
                        "compact_halo_residue_score": compact_halo,
                        "residual_star_score": 0.10,
                        "starless_noise_gain": 1.0,
                    },
                }
                handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
                    processor,
                    quality,
                    [],
                    separation_accepted=True,
                )
                self.assertEqual(handoff["processing_policy"], expected_policy)

        contract_invalid_handoff = (
            stage6_star_separation_module._stage8_handoff_from_stage6(
                processor,
                {
                    "status": "poor",
                    "derived": {
                        "halo_residue_score": 1.2001,
                        "compact_halo_residue_score": 0.59,
                    },
                },
                [],
                separation_accepted=False,
            )
        )
        self.assertEqual(
            contract_invalid_handoff["processing_policy"],
            "skip",
        )

        m42_handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
            processor,
            {
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.488,
                    "compact_halo_residue_score": 0.493,
                    "residual_star_score": 0.10,
                    "starless_noise_gain": 1.0,
                },
            },
            [],
            separation_accepted=True,
        )
        self.assertEqual(
            m42_handoff["reason_text"],
            "bright_nebula_halo_advisory: 0.488 > 0.350, accepted_limit=0.600",
        )
        self.assertEqual(m42_handoff["reason_code"], "bright_nebula_halo_advisory")

        repaired_handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
            processor,
            {
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.445,
                    "compact_halo_residue_score": 0.445,
                    "residual_star_score": 0.056,
                    "starless_noise_gain": 0.625,
                },
            },
            [
                {
                    "accepted": True,
                    "acceptance_path": "residual_or_halo",
                    "trigger": {
                        "reason": "bright_nebula_halo_advisory",
                        "halo_residue_score": 0.488,
                        "compact_halo_residue_score": 0.493,
                    },
                }
            ],
            separation_accepted=True,
        )
        self.assertEqual(
            repaired_handoff["reason_text"],
            "bright_nebula_halo_advisory: 0.488 > 0.350, accepted_limit=0.600",
        )
        self.assertAlmostEqual(
            repaired_handoff["metrics"]["halo_residue_score"],
            0.445,
        )
        self.assertAlmostEqual(
            repaired_handoff["metrics"]["trigger_effective_halo_residue_score"],
            0.493,
        )
        self.assertAlmostEqual(
            m42_handoff["metrics"]["effective_halo_residue_score"],
            0.493,
        )

    def test_stage6_safe_bright_nebula_halo_does_not_trigger_pixel_repair(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        quality = {
            "status": "ok",
            "derived": {
                "halo_residue_score": 0.32,
                "compact_halo_residue_score": 0.34,
            },
        }

        trigger = (
            stage6_star_separation_module._stage7_starless_pixel_repair_trigger(
                processor,
                quality,
            )
        )

        self.assertFalse(trigger["triggered"])
        self.assertEqual(trigger["reason"], "")

    def test_cleanup_archives_only_lightweight_text_diagnostics(self):
        import zipfile

        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.debug_mode = False

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.work_dir = Path(td.name)
        processor.process_dir = processor.work_dir / "process"
        review_dir = (
            processor.process_dir / "review_bundles" / "stage5_linear_cleanup"
        )
        review_dir.mkdir(parents=True)
        (processor.process_dir / "stage5_linear_report.json").write_text(
            '{"status":"ok"}', encoding="utf-8"
        )
        (processor.process_dir / "stage.log").write_text("diagnostic", encoding="utf-8")
        (review_dir / "preview.png").write_bytes(b"png")
        (review_dir / "review.json").write_text(
            json.dumps({"previews": {"preview": str(review_dir / "preview.png")}}),
            encoding="utf-8",
        )
        ui_preview = processor.process_dir / "ui_preview" / "latest.png"
        ui_preview.parent.mkdir()
        ui_preview.write_bytes(b"ui")
        (processor.process_dir / "large.fit").write_bytes(b"fits")

        processor.cleanup()

        archive_path = processor.work_dir / "starun_diagnostics.zip"
        self.assertTrue(archive_path.exists())
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
        self.assertIn("manifest.json", names)
        self.assertIn("process/stage5_linear_report.json", names)
        self.assertIn(
            "process/review_bundles/stage5_linear_cleanup/review.json",
            names,
        )
        self.assertNotIn(
            "process/review_bundles/stage5_linear_cleanup/preview.png",
            names,
        )
        self.assertNotIn("process/ui_preview/latest.png", names)
        self.assertNotIn("process/large.fit", names)
        self.assertFalse((review_dir / "preview.png").exists())
        self.assertTrue(ui_preview.is_file())

    def test_checkpoint_mode_preserves_failure_site_when_preflight_fails(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.checkpoint_mode = True
        processor.cfg.debug_mode = False

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.work_dir = Path(td.name)
        processor.process_dir = processor.work_dir / "process"
        processor.process_dir.mkdir()
        evidence = processor.process_dir / "stage8_candidate.fit"
        evidence.write_bytes(b"failure-evidence")
        processor._checkpoint_compaction_preflight = (  # type: ignore[method-assign]
            lambda: (False, "final delivery SHA-256 verification failed")
        )

        processor.cleanup()

        self.assertTrue(evidence.is_file())
        self.assertFalse(processor._checkpoint_retention_report["applied"])
        self.assertEqual(
            processor._checkpoint_retention_report["reason"],
            "final delivery SHA-256 verification failed",
        )
        retention_path = processor.work_dir / "checkpoint-retention.json"
        self.assertTrue(retention_path.is_file())
        persisted = json.loads(retention_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "preserved")

    def test_stage7_retry_cleanup_preserves_all_best_snapshot_layers(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        retained = {
            "starless_best_syqon_refine_1.fit",
            "starmask_best_syqon_refine_1.fit",
            "starmask_raw_best_syqon_refine_1.fit",
        }
        removable = {
            "starless.fit",
            "starmask.fit",
            "starmask_raw.fit",
        }
        for name in retained | removable:
            (processor.process_dir / name).write_bytes(name.encode("utf-8"))

        processor._clear_star_separation_outputs()

        self.assertTrue(all((processor.process_dir / name).exists() for name in retained))
        self.assertTrue(all(not (processor.process_dir / name).exists() for name in removable))

    def test_stage7_restore_rejects_incomplete_snapshot_without_partial_restore(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        current_starless = processor.process_dir / "starless.fit"
        current_starmask = processor.process_dir / "starmask_clean.fit"
        current_raw = processor.process_dir / "starmask_raw.fit"
        current_starless.write_bytes(b"current-starless")
        current_starmask.write_bytes(b"current-starmask")
        current_raw.write_bytes(b"current-raw")
        (processor.process_dir / "starless_best.fit").write_bytes(b"best-starless")
        (processor.process_dir / "starmask_best.fit").write_bytes(b"best-starmask")
        processor.starless_file = current_starless
        processor.starmask_file = current_starmask

        with self.assertRaisesRegex(FileNotFoundError, "Stage7 snapshot is incomplete"):
            processor._stage7_restore_snapshot(
                {
                    "starless": "starless_best",
                    "starmask": "starmask_best",
                    "starmask_raw": "starmask_raw_missing",
                    "starmask_kind": "clean",
                }
            )

        self.assertEqual(current_starless.read_bytes(), b"current-starless")
        self.assertEqual(current_starmask.read_bytes(), b"current-starmask")
        self.assertEqual(current_raw.read_bytes(), b"current-raw")
        self.assertEqual(processor.starless_file, current_starless)
        self.assertEqual(processor.starmask_file, current_starmask)

    def test_stage7_restore_replaces_all_layers_from_one_snapshot(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        current_starless = processor.process_dir / "starless.fit"
        current_starmask = processor.process_dir / "starmask_clean.fit"
        current_raw = processor.process_dir / "starmask_raw.fit"
        for path in (current_starless, current_starmask, current_raw):
            path.write_bytes(b"retry-2")
        (processor.process_dir / "starless_best.fit").write_bytes(b"retry-1-starless")
        (processor.process_dir / "starmask_best.fit").write_bytes(b"retry-1-starmask")
        (processor.process_dir / "starmask_raw_best.fit").write_bytes(b"retry-1-raw")

        processor._stage7_restore_snapshot(
            {
                "starless": "starless_best",
                "starmask": "starmask_best",
                "starmask_raw": "starmask_raw_best",
                "starmask_kind": "clean",
            }
        )

        self.assertEqual(current_starless.read_bytes(), b"retry-1-starless")
        self.assertEqual(current_starmask.read_bytes(), b"retry-1-starmask")
        self.assertEqual(current_raw.read_bytes(), b"retry-1-raw")
        self.assertEqual(processor.starless_file, current_starless)
        self.assertEqual(processor.starmask_file, current_starmask)

    def test_stage7_transform_loss_gate_is_non_overridable(self):
        processor = pipeline_module.StarunPostProcessor()
        report = {
            "status": "available",
            "global": {
                "newly_hard_clipped_ratio": 0.0006,
                "newly_zeroed_ratio": 0.0,
                "unexpected_newly_zeroed_ratio": 0.0,
            },
        }

        rejected = processor._stage7_transform_loss_gate(report)
        self.assertFalse(rejected["accepted"])
        self.assertTrue(rejected["technical_gate"])
        self.assertIn("transform_new_hard_clip_ratio", rejected["issues"][0])

        report["global"]["newly_hard_clipped_ratio"] = 0.0002
        advisory = processor._stage7_transform_loss_gate(report)
        self.assertTrue(advisory["accepted"])
        self.assertEqual(advisory["status"], "advisory")

        unavailable = processor._stage7_transform_loss_gate(
            {"status": "unavailable"}
        )
        self.assertFalse(unavailable["accepted"])
        self.assertEqual(unavailable["issues"], ["transform_loss_unavailable"])

    def test_stage7_color_vector_gate_uses_channel_semantics(self):
        processor = pipeline_module.StarunPostProcessor()
        report = {
            "status": "available",
            "metrics": {"chromaticity_l1_half_p95": 0.09},
        }

        processor._channel_semantics = "broadband_rgb"
        broadband = processor._stage7_color_vector_gate(report)
        self.assertFalse(broadband["accepted"])

        processor._channel_semantics = "narrowband_composite"
        narrowband = processor._stage7_color_vector_gate(report)
        self.assertTrue(narrowband["accepted"])
        self.assertEqual(narrowband["status"], "ok")

    def test_stage7_adaptive_chroma_rescue_reaches_m8_style_excess(self):
        processor = pipeline_module.StarunPostProcessor()
        attempt = {
            "background_quality_gate": {
                "metrics": {
                    "chroma_load": 0.113,
                    "chroma_load_low_absolute_effective_max": 0.060,
                },
                "limits": {"chroma_load_signal_excluded_max": 0.060},
            }
        }

        strengths = processor._stage7_chroma_rescue_strengths(attempt)

        self.assertEqual(len(strengths), 3)
        self.assertGreater(max(strengths), 0.35)
        self.assertTrue(any(abs(value - 0.4956) < 0.001 for value in strengths))
