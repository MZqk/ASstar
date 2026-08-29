"""Focused tests for the Stage 8 large-galaxy disk and seam gate."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class Stage8LargeGalaxySeamTests(PipelinePluginFallbackTestBase):
    def _large_galaxy_case(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "broadband_rgb_osc"
        processor._active_target_type = lambda: "large_galaxy"
        processor._frozen_primary_target = {
            "type": "large_galaxy",
            "confidence": 0.95,
            "method": "target_profiler",
        }
        processor._stage7_halo_residue_score = lambda: 0.0
        processor._stage7_effective_halo_threshold = lambda: 0.35
        height, width = 240, 320
        yy, xx = np.indices((height, width))
        signal = np.exp(
            -(((xx - 160) / 52.0) ** 2 + ((yy - 120) / 30.0) ** 2)
        ).astype(np.float32)
        image = np.full((3, height, width), 0.03, dtype=np.float32)
        image += np.asarray([0.34, 0.22, 0.15], dtype=np.float32)[
            :, None, None
        ] * signal[None]
        return processor, image

    def test_elliptical_disk_and_outer_weights_are_continuous_and_monotonic(self):
        processor, image = self._large_galaxy_case()
        processor.cfg.stage8_large_galaxy_structure_scale_max = 0.75
        generic_masks = processor._stage8_generate_starless_masks(image)

        masks, report = (
            pipeline_module.stage8_pixels.stage8_large_galaxy_structure_masks(
                processor,
                generic_masks,
            )
        )

        self.assertIsNotNone(masks, report)
        self.assertEqual(report["route"], "large_galaxy_elliptical_soft_v1")
        routed, routed_report = (
            pipeline_module.stage8_pixels.stage8_target_structure_masks(
                processor,
                generic_masks,
            )
        )
        self.assertIsNotNone(routed, routed_report)
        self.assertEqual(routed_report["structure_scale_max"], 0.50)
        geometry = report["geometry"]
        yy, xx = np.indices(image.shape[1:])
        dx = xx - geometry["center_x"]
        dy = yy - geometry["center_y"]
        major = np.asarray(geometry["major_axis_vector"])
        minor = np.asarray(geometry["minor_axis_vector"])
        rho = np.sqrt(
            np.square((dx * major[0] + dy * major[1]) / geometry["major_radius"])
            + np.square(
                (dx * minor[0] + dy * minor[1]) / geometry["minor_radius"]
            )
        )
        disk = masks["enhancement_subject_weight"]
        outer = masks["enhancement_outer_weight"]
        self.assertTrue(np.allclose(disk[rho <= 0.72], 1.0, atol=1e-6))
        self.assertTrue(np.allclose(disk[rho >= 1.15], 0.0, atol=1e-6))
        self.assertTrue(np.all(outer <= disk + 1e-7))
        self.assertTrue(np.allclose(outer[rho <= 0.30], 0.0, atol=1e-6))
        ordered = np.argsort(rho.reshape(-1))
        self.assertLessEqual(
            float(np.max(np.diff(disk.reshape(-1)[ordered]))),
            1e-6,
        )
        self.assertTrue(
            np.allclose(masks["background_mask"], 1.0 - disk, atol=1e-7)
        )

    def test_large_galaxy_operations_use_disk_and_preserve_outside(self):
        processor, image = self._large_galaxy_case()

        enhanced, diagnostics, _messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.10, "unsharp_amount": 0.10},
                label="test",
            )
        )

        self.assertEqual(
            diagnostics["mask_route"],
            "large_galaxy_elliptical_soft_v1",
        )
        operations = diagnostics["local_adjustment_engine"]["operations"]
        self.assertTrue(operations)
        self.assertEqual({operation["mask"] for operation in operations}, {"nebula"})
        masks = processor._stage8_generate_starless_masks(image)
        routed, _ = (
            pipeline_module.stage8_pixels.stage8_target_structure_masks(
                processor,
                masks,
            )
        )
        outside = routed["enhancement_subject_weight"] <= 0.0
        self.assertTrue(
            np.allclose(enhanced[:, outside], image[:, outside], atol=1e-7)
        )
        self.assertGreater(float(np.max(np.abs(enhanced - image))), 0.0)

    def test_quality_assessment_rejects_sub_float32_ulp_outside_delta(self):
        processor, baseline = self._large_galaxy_case()
        baseline = baseline.astype(np.float64)
        generic_masks = processor._stage8_generate_starless_masks(baseline)
        routed, route_report = (
            pipeline_module.stage8_pixels.stage8_target_structure_masks(
                processor,
                generic_masks,
            )
        )
        self.assertIsNotNone(routed, route_report)
        outside = np.argwhere(
            np.asarray(routed["enhancement_support_weight"]) <= 0.0
        )
        self.assertGreater(len(outside), 0)
        candidate = baseline.copy()
        y, x = (int(value) for value in outside[0])
        candidate[0, y, x] += np.float64(1e-12)
        processor._read_image_by_stem = (
            lambda stem: baseline
            if stem == "stage8_input_starless"
            else candidate
        )

        quality = processor._stage8_quality_assessment()

        identity = quality["outside_target_identity"]
        self.assertEqual(identity["status"], "hard_failed")
        self.assertFalse(identity["accepted"])
        self.assertEqual(identity["reason"], "outside_target_pixels_changed")
        self.assertNotEqual(
            identity["baseline_outside_pixel_sha256"],
            identity["candidate_outside_pixel_sha256"],
        )
        self.assertGreater(identity["max_abs_change"], 0.0)
        self.assertIn(
            "outside_target_pixel_identity_gate_failed="
            "outside_target_pixels_changed",
            quality["issues"],
        )

    def test_outside_target_identity_missing_or_shape_tampered_fails_closed(self):
        processor, baseline = self._large_galaxy_case()
        missing = (
            pipeline_module.stage8_pixels.stage8_outside_target_identity_report(
                baseline,
                baseline.copy(),
                {},
                target_type="large_galaxy",
            )
        )
        tampered = (
            pipeline_module.stage8_pixels.stage8_outside_target_identity_report(
                baseline,
                baseline.copy(),
                {"enhancement_support_weight": np.zeros((2, 2))},
                target_type="large_galaxy",
            )
        )

        self.assertFalse(missing["accepted"])
        self.assertEqual(missing["reason"], "target_structure_support_unavailable")
        self.assertFalse(tampered["accepted"])
        self.assertEqual(
            tampered["reason"],
            "target_structure_support_shape_mismatch",
        )

    def test_large_galaxy_fit_failure_is_lossless_passthrough(self):
        processor, image = self._large_galaxy_case()
        with patch.object(
            pipeline_module.stage8_pixels.stage7_quality,
            "stage7_galaxy_structure_masks",
            return_value={
                "available": False,
                "status": "unavailable",
                "reason": "synthetic_fit_failure",
            },
        ):
            enhanced, diagnostics, messages = (
                processor._apply_stage8_masked_pixel_enhancement(
                    image,
                    {"saturation": 0.10, "unsharp_amount": 0.10},
                    label="test",
                )
            )

        self.assertTrue(np.array_equal(enhanced, image))
        self.assertEqual(
            diagnostics["processing_scope"]["mode"],
            "large_galaxy_passthrough",
        )
        self.assertEqual(diagnostics["local_adjustment_engine"]["status"], "passthrough")
        self.assertTrue(any("synthetic_fit_failure" in item for item in messages))

    def test_large_galaxy_limited_route_is_lossless_passthrough(self):
        processor, image = self._large_galaxy_case()
        processor._stage8_handoff = {"processing_policy": "limited"}

        enhanced, diagnostics, _messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.05, "unsharp_amount": 0.10},
                label="test",
            )
        )

        self.assertTrue(np.array_equal(enhanced, image))
        self.assertEqual(
            diagnostics["processing_scope"]["reason"],
            "large_galaxy_limited_safe_passthrough",
        )
        self.assertEqual(diagnostics["structure_execution"]["scale"], 0.0)

    def test_limited_advisory_can_enter_independent_safe_passthrough_preflight(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        processor._stage8_handoff = {
            "processing_policy": "limited",
            "restricted_downstream": True,
            "quality_status": "ok",
            "reason_code": "stage6_quality_advisory",
            "reasons": [
                {
                    "code": "stage6_quality_advisory",
                    "source_stage": 6,
                    "advisories": [
                        "galaxy_disk_halo_residue 0.713>0.480",
                    ],
                }
            ],
            "metrics": {
                "residual_star_score": 0.0,
                "residual_star_hard_limit": 0.90,
                "starless_noise_gain": 0.474,
                "starless_noise_gain_hard_limit": 2.50,
                "effective_halo_residue_score": 0.713,
                "halo_residue_hard_limit": 0.96,
            },
        }
        guard = {
            "status": "ok",
            "processing_policy": "limited",
            "reason_code": "stage6_quality_advisory",
            "hard_reasons": [],
            "subject_reasons": [],
        }

        report = runtime._stage8_limited_safe_passthrough_eligibility(
            processor,
            stage8_guard_report=guard,
            final_source="stage8_enhanced",
            final_quality="ok",
            user_processing_mode="auto",
            external_override=False,
        )

        self.assertTrue(report["accepted"], report)
        self.assertTrue(report["checks"]["upstream_hard_metrics_clear"])
        self.assertTrue(report["checks"]["review_requirement_free"])

    def test_limited_hard_failure_or_review_never_becomes_formal_passthrough(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        guard = {
            "status": "ok",
            "processing_policy": "limited",
            "reason_code": "stage6_quality_hard_failed_retained",
            "hard_reasons": [],
            "subject_reasons": [],
        }
        for scenario in ("hard_metric", "review_requirement"):
            with self.subTest(scenario=scenario):
                processor = self._new_processor()
                processor._stage8_handoff = {
                    "processing_policy": "limited",
                    "restricted_downstream": True,
                    "quality_status": "ok",
                    "reason_code": (
                        "stage6_quality_hard_failed_retained"
                        if scenario == "hard_metric"
                        else "stage6_quality_advisory"
                    ),
                    "reasons": [
                        {
                            "code": (
                                "stage6_quality_hard_failed_retained"
                                if scenario == "hard_metric"
                                else "stage6_quality_advisory"
                            ),
                            "source_stage": 6,
                        }
                    ],
                    "metrics": {
                        "residual_star_score": 0.0,
                        "residual_star_hard_limit": 0.90,
                        "starless_noise_gain": 0.474,
                        "starless_noise_gain_hard_limit": 2.50,
                        "effective_halo_residue_score": (
                            1.10 if scenario == "hard_metric" else 0.713
                        ),
                        "halo_residue_hard_limit": 0.96,
                    },
                }
                scenario_guard = dict(guard)
                scenario_guard["reason_code"] = processor._stage8_handoff[
                    "reason_code"
                ]
                if scenario == "review_requirement":
                    processor._require_review(6, "starmask_cleanup_borderline")

                report = runtime._stage8_limited_safe_passthrough_eligibility(
                    processor,
                    stage8_guard_report=scenario_guard,
                    final_source="stage8_enhanced",
                    final_quality="ok",
                    user_processing_mode="auto",
                    external_override=False,
                )

                self.assertFalse(report["accepted"], report)
                expected = (
                    "upstream_hard_metrics_clear"
                    if scenario == "hard_metric"
                    else "review_requirement_free"
                )
                self.assertIn(expected, report["issues"])

    def test_limited_verified_noop_requires_exact_final_pixel_identity(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        baseline = processor.image_pixels.copy()
        processor.saved_image_pixels["stage8_input_starless"] = baseline.copy()
        processor.saved_image_pixels["stage8_enhanced"] = baseline.copy()
        processor._save_stage_output("stage8_input_starless")
        processor._save_stage_output("stage8_enhanced")
        processor._stage8_safe_passthrough_color_only_preflight = {
            "schema": "starun.stage8-safe-passthrough-preflight.v1",
            "status": "accepted",
            "accepted": True,
            "source_mode": "limited_safe_passthrough",
        }
        processor._stage8_quality_assessment = lambda **_kwargs: {
            "status": "ok",
            "issues": [],
            "outside_target_identity": {
                "status": "ok",
                "accepted": True,
                "exact_pixel_identity": True,
            },
        }
        processor._stage8_generate_starless_masks = lambda image: {
            "star_halo_guard_mask": np.zeros(image.shape[-2:], dtype=np.float32)
        }
        accepted_gate = {
            "status": "ok",
            "accepted": True,
            "issues": [],
        }
        skipped_color = {
            "status": "skipped_ineligible",
            "accepted": False,
        }
        with patch.object(
            runtime.spatial_background_lineage,
            "assess_final_spatial_background",
            return_value=accepted_gate,
        ), patch.object(
            runtime.star_halo_guard,
            "assess_candidate",
            return_value=accepted_gate,
        ):
            accepted = runtime._stage8_safe_passthrough_final_validation(
                processor,
                subject_chroma_report=skipped_color,
                palette_report=skipped_color,
            )
            processor.saved_image_pixels["stage8_enhanced"] = (
                baseline + np.float32(0.001)
            )
            rejected = runtime._stage8_safe_passthrough_final_validation(
                processor,
                subject_chroma_report=skipped_color,
                palette_report=skipped_color,
            )

        self.assertTrue(accepted["accepted"], accepted)
        self.assertEqual(
            accepted["checks"]["color"]["mode"],
            "verified_pixel_identity",
        )
        self.assertFalse(rejected["accepted"], rejected)
        self.assertIn("color", rejected["issues"])

    def test_post_cumulative_verified_color_noop_requires_audited_terminals(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        baseline = processor.image_pixels.copy()
        processor.saved_image_pixels["stage8_input_starless"] = baseline.copy()
        processor.saved_image_pixels["stage8_enhanced"] = baseline.copy()
        processor._save_stage_output("stage8_input_starless")
        processor._save_stage_output("stage8_enhanced")
        processor._stage8_safe_passthrough_color_only_preflight = {
            "schema": "starun.stage8-safe-passthrough-preflight.v1",
            "status": "accepted",
            "accepted": True,
            "source_mode": "post_cumulative_structure_rollback",
        }
        processor._stage8_quality_assessment = lambda **_kwargs: {
            "status": "ok",
            "issues": [],
            "outside_target_identity": {
                "status": "ok",
                "accepted": True,
                "exact_pixel_identity": True,
            },
        }
        processor._stage8_generate_starless_masks = lambda image: {
            "star_halo_guard_mask": np.zeros(image.shape[-2:], dtype=np.float32)
        }
        accepted_gate = {"status": "ok", "accepted": True, "issues": []}
        subject_noop = {
            "schema": "starun.stage8-subject-chroma.v1",
            "status": "skipped_ineligible",
            "accepted": False,
            "feeds_main_pipeline": False,
            "eligibility": {"eligible": False},
            "transaction": {
                "baseline_saved": False,
                "candidate_saved": False,
                "rollback_performed": False,
            },
        }
        palette_noop = {
            "schema": "starun.stage8-dualband-palette.v2",
            "status": "rejected_by_palette_quality_gate",
            "accepted": False,
            "feeds_main_pipeline": False,
            "eligibility": {"eligible": True},
            "transaction": {
                "baseline_saved": True,
                "candidate_saved": False,
                "rollback_performed": True,
                "rollback_ok": True,
            },
        }

        with patch.object(
            runtime.spatial_background_lineage,
            "assess_final_spatial_background",
            return_value=accepted_gate,
        ), patch.object(
            runtime.star_halo_guard,
            "assess_candidate",
            return_value=accepted_gate,
        ):
            accepted = runtime._stage8_safe_passthrough_final_validation(
                processor,
                subject_chroma_report=subject_noop,
                palette_report=palette_noop,
            )
            bad_schema = dict(palette_noop)
            bad_schema["schema"] = "tampered"
            rejected_schema = runtime._stage8_safe_passthrough_final_validation(
                processor,
                subject_chroma_report=subject_noop,
                palette_report=bad_schema,
            )
            bad_rollback = dict(palette_noop)
            bad_rollback["transaction"] = dict(
                palette_noop["transaction"], rollback_ok=False
            )
            rejected_rollback = runtime._stage8_safe_passthrough_final_validation(
                processor,
                subject_chroma_report=subject_noop,
                palette_report=bad_rollback,
            )
            processor.saved_image_pixels["stage8_enhanced"] = (
                baseline + np.float32(0.001)
            )
            rejected_pixels = runtime._stage8_safe_passthrough_final_validation(
                processor,
                subject_chroma_report=subject_noop,
                palette_report=palette_noop,
            )

        self.assertTrue(accepted["accepted"], accepted)
        self.assertEqual(
            accepted["checks"]["color"]["mode"],
            "verified_color_noop_after_structure_rollback",
        )
        self.assertEqual(
            accepted["checks"]["color"]["nonmutating_terminal_evidence"]
            ["palette"]["terminal_mode"],
            "quality_rejected_and_rolled_back",
        )
        for report in (rejected_schema, rejected_rollback, rejected_pixels):
            self.assertFalse(report["accepted"], report)
            self.assertIn("color", report["issues"])

    def test_safe_passthrough_final_rejects_missing_outside_identity_evidence(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        baseline = processor.image_pixels.copy()
        processor.saved_image_pixels["stage8_input_starless"] = baseline.copy()
        processor.saved_image_pixels["stage8_enhanced"] = baseline.copy()
        processor._save_stage_output("stage8_input_starless")
        processor._save_stage_output("stage8_enhanced")
        processor._stage8_safe_passthrough_color_only_preflight = {
            "status": "accepted",
            "accepted": True,
            "source_mode": "limited_safe_passthrough",
        }
        processor._stage8_quality_assessment = lambda **_kwargs: {
            "status": "ok",
            "issues": [],
        }
        processor._stage8_generate_starless_masks = lambda image: {
            "star_halo_guard_mask": np.zeros(image.shape[-2:], dtype=np.float32)
        }
        accepted_gate = {"status": "ok", "accepted": True, "issues": []}
        skipped_color = {"status": "skipped_ineligible", "accepted": False}

        with patch.object(
            runtime.spatial_background_lineage,
            "assess_final_spatial_background",
            return_value=accepted_gate,
        ), patch.object(
            runtime.star_halo_guard,
            "assess_candidate",
            return_value=accepted_gate,
        ), patch.object(
            runtime,
            "_stage8_mutation_union_outside_identity",
            return_value={
                "schema": "starun.stage8-outside-target-identity.v1",
                "status": "unavailable",
                "available": False,
                "applicable": True,
                "accepted": False,
                "reason": "mutation_union_support_unavailable",
            },
        ):
            report = runtime._stage8_safe_passthrough_final_validation(
                processor,
                subject_chroma_report=skipped_color,
                palette_report=skipped_color,
            )

        self.assertFalse(report["accepted"], report)
        self.assertIn("outside_target_pixel_identity", report["issues"])

    def test_mutation_union_allows_frozen_color_roi_but_rejects_pixels_outside_it(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        processor._active_target_type = lambda: "large_galaxy"
        baseline = np.full((3, 32, 40), 0.10, dtype=np.float32)
        subject = np.zeros((32, 40), dtype=np.float32)
        subject[8:24, 10:30] = 1.0
        background = np.ones((32, 40), dtype=np.float32)
        background[subject > 0.0] = 0.0
        processor._stage8_generate_starless_masks = lambda _image: {
            "background_mask": background,
            "subject_mask": subject,
        }
        accepted_color = {"status": "accepted", "accepted": True}
        skipped_palette = {"status": "skipped", "accepted": False}

        inside_candidate = baseline.copy()
        inside_candidate[0, 12, 16] += np.float32(0.002)
        inside = runtime._stage8_mutation_union_outside_identity(
            processor,
            baseline,
            inside_candidate,
            subject_chroma_report=accepted_color,
            palette_report=skipped_palette,
            include_structure=False,
        )

        outside_candidate = inside_candidate.copy()
        outside_candidate[1, 2, 3] += np.float32(0.001)
        outside = runtime._stage8_mutation_union_outside_identity(
            processor,
            baseline,
            outside_candidate,
            subject_chroma_report=accepted_color,
            palette_report=skipped_palette,
            include_structure=False,
        )

        self.assertTrue(inside["accepted"], inside)
        self.assertEqual(
            inside["support_sources"],
            ["frozen_subject_chroma_roi"],
        )
        self.assertFalse(outside["accepted"], outside)
        self.assertEqual(outside["reason"], "outside_target_pixels_changed")
        self.assertGreater(outside["max_abs_change"], 0.0)

    def test_palette_mutation_union_uses_same_frozen_guarded_roi(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        processor._active_target_type = lambda: "emission_nebula_widefield"
        processor.cfg.stage8_dualband_palette_strength = 0.85
        baseline = np.full((3, 32, 40), 0.10, dtype=np.float32)
        shape = baseline.shape[-2:]
        nebula = np.zeros(shape, dtype=np.float32)
        nebula[6:26, 8:32] = 1.0
        core = np.zeros(shape, dtype=np.float32)
        core[14:18, 18:22] = 1.0
        stars = np.zeros(shape, dtype=np.float32)
        stars[9:11, 11:13] = 1.0
        halo = np.zeros(shape, dtype=np.float32)
        halo[8:12, 10:14] = 1.0
        background = np.ones(shape, dtype=np.float32)
        background[nebula > 0.0] = 0.0
        generated_background = np.zeros(shape, dtype=np.float32)
        processor._stage8_generate_starless_masks = lambda _image: {
            "background_mask": generated_background,
            "core_mask": np.zeros(shape, dtype=np.float32),
            "nebula_mask": np.ones(shape, dtype=np.float32),
            "faint_nebula_mask": np.zeros(shape, dtype=np.float32),
            "star_mask": np.zeros(shape, dtype=np.float32),
            "star_halo_guard_mask": np.zeros(shape, dtype=np.float32),
        }
        processor._stage7_frozen_rendition_masks = {
            "background_mask": background,
            "core_mask": core,
            "nebula_mask": nebula,
            "faint_nebula_mask": np.zeros(shape, dtype=np.float32),
            "star_mask": stars,
            "star_halo_guard_mask": halo,
            "subject_mask": nebula,
        }
        skipped_color = {"status": "skipped", "accepted": False}
        accepted_palette = {"status": "accepted", "accepted": True}

        inside_candidate = baseline.copy()
        inside_candidate[0, 20, 25] += np.float32(0.002)
        inside = runtime._stage8_mutation_union_outside_identity(
            processor,
            baseline,
            inside_candidate,
            subject_chroma_report=skipped_color,
            palette_report=accepted_palette,
            include_structure=False,
        )

        guarded_candidate = baseline.copy()
        guarded_candidate[0, 15, 19] += np.float32(0.002)
        guarded = runtime._stage8_mutation_union_outside_identity(
            processor,
            baseline,
            guarded_candidate,
            subject_chroma_report=skipped_color,
            palette_report=accepted_palette,
            include_structure=False,
        )

        self.assertTrue(inside["accepted"], inside)
        self.assertFalse(guarded["accepted"], guarded)
        self.assertIn("core_mask", inside["frozen_mask_keys"])
        self.assertEqual(
            inside["support_sources"],
            ["dualband_palette_subject_roi"],
        )

    def test_seam_gate_accepts_smooth_delta_and_hard_rejects_step_edge(self):
        processor, image = self._large_galaxy_case()
        masks = processor._stage8_generate_starless_masks(image)
        routed, report = (
            pipeline_module.stage8_pixels.stage8_large_galaxy_structure_masks(
                processor,
                masks,
            )
        )
        self.assertIsNotNone(routed, report)
        disk = routed["enhancement_subject_weight"]
        smooth_candidate = image + 0.001 * disk[None]
        hard_candidate = image + 0.08 * (disk > 0.50)[None]

        smooth = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            smooth_candidate,
        )
        hard = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            hard_candidate,
        )

        self.assertEqual(smooth["status"], "ok", smooth)
        self.assertTrue(smooth["accepted"])
        self.assertEqual(hard["status"], "hard_failed", hard)
        self.assertFalse(hard["accepted"])
        self.assertTrue(hard["seam_channels"]["luma"])
        self.assertGreaterEqual(hard["sample_counts"]["boundary"], 64)
        self.assertGreaterEqual(hard["sample_counts"]["interior"], 64)

    def test_task_config_cannot_relax_seam_hard_limits(self):
        processor, image = self._large_galaxy_case()
        processor.cfg.stage8_subject_boundary_luma_residual_max = 1.0
        processor.cfg.stage8_subject_boundary_chroma_residual_max = 1.0
        processor.cfg.stage8_subject_boundary_residual_ratio_max = 99.0
        masks = processor._stage8_generate_starless_masks(image)
        routed, route_report = (
            pipeline_module.stage8_pixels.stage8_large_galaxy_structure_masks(
                processor,
                masks,
            )
        )
        self.assertIsNotNone(routed, route_report)
        disk = routed["enhancement_subject_weight"]
        candidate = image + 0.08 * (disk > 0.50)[None]

        report = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            candidate,
        )

        self.assertEqual(
            report["thresholds"]["boundary_luma_residual_p95_max"],
            0.0035,
        )
        self.assertEqual(
            report["thresholds"]["boundary_chroma_residual_p95_max"],
            0.0020,
        )
        self.assertEqual(
            report["thresholds"]["boundary_to_interior_ratio_max"],
            1.60,
        )
        self.assertFalse(report["accepted"], report)

    def test_emission_nebula_uses_the_same_transactional_seam_gate(self):
        processor, image = self._large_galaxy_case()
        processor._active_target_type = lambda: "emission_nebula_widefield"
        masks = processor._stage8_generate_starless_masks(image)
        subject = np.asarray(masks["nebula_mask"], dtype=np.float32)
        hard_candidate = image + 0.08 * (subject > 0.50)[None]

        report = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            hard_candidate,
        )

        self.assertTrue(report["applicable"], report)
        self.assertEqual(
            report["mask_route"],
            "emission_nebula_target_soft_v1",
        )
        self.assertEqual(report["status"], "hard_failed", report)
        self.assertFalse(report["accepted"])

    def test_boundary_ratio_is_a_hard_limit_even_when_absolute_residual_is_small(self):
        processor, image = self._large_galaxy_case()
        masks = processor._stage8_generate_starless_masks(image)
        routed, route_report = (
            pipeline_module.stage8_pixels.stage8_large_galaxy_structure_masks(
                processor,
                masks,
            )
        )
        self.assertIsNotNone(routed, route_report)
        disk = routed["enhancement_subject_weight"]
        candidate = image + 0.001 * (disk > 0.50)[None]

        report = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            candidate,
        )

        metrics = report["metrics"]
        self.assertLess(
            metrics["boundary_luma_residual_p95"],
            report["thresholds"]["boundary_luma_residual_p95_max"],
        )
        self.assertGreater(
            metrics["boundary_luma_to_interior_ratio"],
            report["thresholds"]["boundary_to_interior_ratio_max"],
        )
        self.assertEqual(report["status"], "hard_failed", report)
        self.assertFalse(report["accepted"])

    def test_boundary_retry_uses_analytic_cap_and_preserves_outside(self):
        processor, image = self._large_galaxy_case()
        masks = processor._stage8_generate_starless_masks(image)
        routed, route_report = (
            pipeline_module.stage8_pixels.stage8_large_galaxy_structure_masks(
                processor,
                masks,
            )
        )
        self.assertIsNotNone(routed, route_report)
        disk = routed["enhancement_subject_weight"]
        yy, xx = np.indices(disk.shape)
        texture = ((xx + yy) % 2).astype(np.float32)
        boundary = (disk > 0.05) & (disk < 0.80)
        interior = disk >= 0.80
        delta = np.zeros(disk.shape, dtype=np.float32)
        delta[boundary] = 0.006 * texture[boundary]
        delta[interior] = 0.003 * texture[interior]
        candidate = image + delta[None]
        before = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            candidate,
        )

        retry, retry_report = (
            pipeline_module.stage8_pixels.stage8_subject_boundary_retry_candidate(
                processor,
                image,
                candidate,
                seam_report=before,
            )
        )
        after = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            retry,
        )

        self.assertEqual(before["schema"], "starun.stage8-subject-boundary-seam.v2")
        self.assertEqual(before["status"], "hard_failed", before)
        self.assertTrue(retry_report["accepted"], retry_report)
        self.assertEqual(retry_report["retained_delta"], 1.0)
        self.assertLessEqual(retry_report["boundary_scale"], 1.0)
        self.assertEqual(retry_report["interior_scale"], 1.0)
        self.assertGreater(retry_report["analytic_boundary_cap"], 0.0)
        self.assertTrue(after["accepted"], after)
        target_masks, _ = (
            pipeline_module.stage8_pixels.stage8_target_structure_masks(
                processor,
                masks,
            )
        )
        outside = target_masks["enhancement_subject_weight"] <= 0.05
        self.assertTrue(np.array_equal(retry[:, outside], image[:, outside]))

    def test_emission_seam_builds_guarded_mask_at_full_resolution(self):
        processor, _image = self._large_galaxy_case()
        processor._active_target_type = lambda: "emission_nebula_widefield"
        height, width = 1200, 1280
        yy, xx = np.indices((height, width), dtype=np.float32)
        signal = np.exp(
            -(((xx - 640.0) / 260.0) ** 2 + ((yy - 600.0) / 220.0) ** 2)
        ).astype(np.float32)
        image = np.full((3, height, width), 0.03, dtype=np.float32)
        image += 0.18 * signal[None]
        observed_shapes = []

        def full_resolution_masks(data):
            observed_shapes.append(tuple(np.asarray(data).shape))
            self.assertEqual(tuple(np.asarray(data).shape), image.shape)
            subject = np.clip(signal, 0.0, 1.0)
            return {
                "gray": np.mean(np.asarray(data), axis=0),
                "core_mask": np.zeros_like(subject),
                "nebula_mask": subject,
                "faint_nebula_mask": np.zeros_like(subject),
                "background_mask": 1.0 - subject,
                "star_halo_guard_mask": np.zeros_like(subject),
                "coverage": {"nebula": float(np.mean(subject > 0.12))},
            }

        processor._stage8_generate_starless_masks = full_resolution_masks
        candidate = image + 0.001 * signal[None]

        report = pipeline_module.stage8_pixels.stage8_subject_boundary_seam_report(
            processor,
            image,
            candidate,
        )

        self.assertEqual(observed_shapes, [image.shape])
        self.assertTrue(report["available"], report)
        self.assertEqual(
            report["geometry"]["source"],
            "full_resolution_target_aware_guarded_mask",
        )
        self.assertEqual(report["geometry"]["analysis_stride"], 2)

    def test_conservative_rerun_applies_unified_35_percent_structure_scale(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor.cfg.stage8_masked_unsharp_amount_max = 0.12
        image = np.full((3, 96, 128), 0.05, dtype=np.float32)
        captured_plans = []
        processor.cmd_with_check = lambda *_args, **_kwargs: True
        processor.siril.get_image_pixeldata = lambda preview=False: image.copy()
        processor._set_current_image_pixeldata = lambda *_args, **_kwargs: None
        processor._save_stage_output = lambda _stem: True
        processor._stage8_quality_assessment = lambda: {"status": "ok"}

        def apply(candidate, plan, *, label, plugin_candidate=None):
            captured_plans.append(dict(plan))
            return candidate.copy(), {"saturation_execution": {}}, [label]

        processor._apply_stage8_masked_pixel_enhancement = apply

        result = pipeline_module.stage8_pixels.stage8_conservative_rerun(
            processor,
            0.20,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(captured_plans[0]["structure_scale"], 0.35)
        self.assertEqual(captured_plans[0]["saturation"], 0.10)

    def test_conservative_rerun_uses_adaptive_boundary_projection_for_seam(self):
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor._active_target_type = lambda: "emission_nebula_widefield"
        height, width = 96, 128
        baseline = np.full((3, height, width), 0.10, dtype=np.float32)
        candidate = baseline.copy()
        candidate[:, 16:80, 20:108] += 0.020
        subject = np.zeros((height, width), dtype=np.float32)
        subject[16:80, 20:108] = 0.40
        subject[32:64, 44:84] = 0.90
        seam = {
            "schema": "starun.stage8-subject-boundary-seam.v2",
            "status": "hard_failed",
            "seam_detected": True,
            "mask_route": "generic_nebula_threshold_v1",
            "metrics": {
                "boundary_luma_residual_p95": 0.00686,
            },
            "thresholds": {
                "boundary_luma_residual_p95_max": 0.0035,
                "boundary_chroma_residual_p95_max": 0.0020,
                "boundary_to_interior_ratio_max": 1.60,
            },
        }
        assessments = iter(
            [
                {
                    "status": "poor",
                    "issues": ["subject_boundary_mask_seam"],
                    "subject_boundary_seam": seam,
                },
                {"status": "ok", "issues": []},
            ]
        )
        pixels = iter([baseline.copy(), candidate.copy()])
        captured = []
        processor._stage8_quality_assessment = lambda: next(assessments)
        processor.cmd_with_check = lambda *_args, **_kwargs: True
        processor.siril.get_image_pixeldata = lambda preview=False: next(pixels)
        processor._stage8_generate_starless_masks = lambda _data: {
            "nebula_mask": subject,
            "coverage": {"nebula": float(np.mean(subject > 0.05))},
        }
        processor._set_current_image_pixeldata = (
            lambda data, **_kwargs: captured.append(np.asarray(data).copy())
        )
        processor._save_stage_output = lambda _stem: True

        result = pipeline_module.stage8_pixels.stage8_conservative_rerun(
            processor,
            0.20,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "analytic_boundary_retained_delta_search")
        self.assertEqual(result["selected_retained_delta"], 1.0)
        self.assertAlmostEqual(
            result["retry_projection"]["boundary_scale"],
            0.5,
            places=2,
        )
        self.assertEqual(result["retry_projection"]["interior_scale"], 1.0)
        retry = captured[-1]
        self.assertTrue(np.array_equal(retry[:, subject <= 0.05], baseline[:, subject <= 0.05]))
        boundary_delta = np.median(retry[:, subject == 0.40] - baseline[:, subject == 0.40])
        interior_delta = np.median(retry[:, subject == 0.90] - baseline[:, subject == 0.90])
        self.assertGreater(interior_delta, boundary_delta)

    def test_seam_retry_failure_rolls_back_to_stage8_input(self):
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
        seam_failure = {
            "status": "poor",
            "issues": ["subject_boundary_mask_seam"],
        }
        processor._stage8_quality_assessment = lambda: seam_failure
        processor._apply_stage8_color_correction_from_quality = lambda _quality: None
        processor._stage8_needs_conservative_rerun = lambda _quality: True
        processor._stage8_conservative_rerun = lambda _sat: {
            "status": "poor",
            "safe_saturation": 0.10,
            "assessment": seam_failure,
        }
        processor._stage8_enhancement_quality_report = lambda: {
            "status": "ok",
            "issues": [],
        }
        rollback_calls = []
        processor._rollback_stage8_to_input = (
            lambda: rollback_calls.append("stage8_input_starless") or True
        )

        stage8_nebula_enhancement(processor)

        self.assertEqual(rollback_calls, ["stage8_input_starless"])
        self.assertEqual(processor._stage8_final_source, "stage8_input_starless")
        self.assertTrue(processor._stage8_fallback_used)
        self.assertEqual(processor.results[-1][1], "degraded")

    def test_seam_retry_success_is_retained_as_degraded_fallback(self):
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
        seam_failure = {
            "status": "poor",
            "issues": ["subject_boundary_mask_seam"],
        }
        accepted = {"status": "ok", "issues": []}
        processor._stage8_quality_assessment = lambda: seam_failure
        processor._apply_stage8_color_correction_from_quality = lambda _quality: None
        processor._stage8_needs_conservative_rerun = lambda _quality: True
        processor._stage8_conservative_rerun = lambda _sat: {
            "status": "ok",
            "safe_saturation": 0.10,
            "assessment": accepted,
        }
        processor._stage8_enhancement_quality_report = lambda: {
            "status": "ok",
            "issues": [],
        }
        rollback_calls = []
        processor._rollback_stage8_to_input = (
            lambda: rollback_calls.append("stage8_input_starless") or True
        )

        stage8_nebula_enhancement(processor)

        self.assertEqual(rollback_calls, [])
        self.assertEqual(processor._stage8_final_source, "stage8_enhanced")
        self.assertEqual(processor._stage8_final_quality, "ok")
        self.assertTrue(processor._stage8_fallback_used)
        self.assertTrue(processor._stage8_handoff["restricted_downstream"])
        self.assertFalse(processor._stage8_handoff["formal_eligible"])
        self.assertEqual(processor.results[-1][1], "degraded")

    def test_frozen_sky_visible_noise_limit_cannot_be_relaxed(self):
        processor = self._new_processor()
        processor.cfg.stage8_frozen_sky_visible_noise_growth_max = 1.50
        image = np.full((3, 32, 32), 0.05, dtype=np.float32)

        report = (
            pipeline_module.stage8_pixels.stage8_frozen_sky_visible_noise_report(
                processor,
                image,
                image,
            )
        )

        self.assertEqual(report["growth_max"], 1.10)
        self.assertEqual(report["status"], "unavailable")
        self.assertFalse(report["accepted"])
        self.assertEqual(
            report["reason"],
            "stage3_frozen_sky_lineage_unverified",
        )

    def test_final_cumulative_qa_rejects_two_locally_safe_noise_deltas(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        baseline = processor.image_pixels.copy().astype(np.float32)
        support = np.ones(baseline.shape[-2:], dtype=bool)
        (processor.process_dir / "stage3_spatial_background_lineage.json").write_text(
            "{}",
            encoding="utf-8",
        )
        lineage = {
            "accepted": True,
            "support_mask": support,
            "support_sha256": "a" * 64,
        }
        yy, xx = np.indices(baseline.shape[-2:])
        pattern = np.where((xx + yy) % 2 == 0, 1.0, -1.0).astype(
            np.float32
        )
        noise_report = (
            pipeline_module.stage8_pixels.stage8_frozen_sky_visible_noise_report
        )
        selected = None
        with patch.object(
            pipeline_module.stage8_pixels.spatial_background_lineage,
            "load_lineage",
            return_value=lineage,
        ):
            for amplitude in np.geomspace(1.0e-6, 0.01, 300):
                first = (
                    baseline + pattern[None] * np.float32(amplitude)
                ).astype(np.float32)
                cumulative = (
                    baseline + pattern[None] * np.float32(2.0 * amplitude)
                ).astype(np.float32)
                first_gate = noise_report(processor, baseline, first)
                second_gate = noise_report(processor, first, cumulative)
                cumulative_gate = noise_report(
                    processor,
                    baseline,
                    cumulative,
                )
                if (
                    first_gate.get("accepted") is True
                    and second_gate.get("accepted") is True
                    and cumulative_gate.get("accepted") is False
                ):
                    selected = (
                        cumulative,
                        first_gate,
                        second_gate,
                        cumulative_gate,
                    )
                    break
        self.assertIsNotNone(selected)
        cumulative, first_gate, second_gate, cumulative_gate = selected
        self.assertLessEqual(first_gate["growth"], 1.10)
        self.assertLessEqual(second_gate["growth"], 1.10)
        self.assertGreater(cumulative_gate["growth"], 1.10)

        processor.image_pixels = baseline.copy()
        processor._save_stage_output("stage8_input_starless")
        processor.image_pixels = cumulative.copy()
        processor._save_stage_output("stage8_enhanced")
        processor._stage8_final_source = "stage8_enhanced"
        processor._stage8_final_quality = "ok"
        zeros = np.zeros(baseline.shape[-2:], dtype=np.float32)
        processor._stage8_generate_starless_masks = lambda _image: {
            "background_mask": np.ones_like(zeros),
            "subject_mask": np.ones_like(zeros),
            "star_halo_guard_mask": zeros,
        }
        processor._stage8_quality_assessment = lambda: {
            "status": "poor",
            "issues": ["frozen_sky_visible_noise_gate_failed"],
            "subject_boundary_seam": {
                "status": "ok",
                "available": True,
                "accepted": True,
            },
            "frozen_sky_visible_noise": cumulative_gate,
            "outside_target_identity": {
                "status": "not_applicable",
                "available": True,
                "accepted": True,
            },
        }
        accepted_gate = {"status": "ok", "accepted": True, "issues": []}
        local_finish = {
            "status": "accepted",
            "accepted": True,
            "accepted_steps": ["revela", "subject_curves"],
        }
        local_chroma = {"status": "accepted", "accepted": True}
        local_palette = {"status": "skipped", "accepted": False}

        with patch.object(
            runtime.spatial_background_lineage,
            "assess_final_spatial_background",
            return_value=accepted_gate,
        ), patch.object(
            runtime.star_halo_guard,
            "assess_candidate",
            return_value=accepted_gate,
        ):
            report = runtime._stage8_enforce_final_cumulative_validation(
                processor,
                subject_chroma_report=local_chroma,
                palette_report=local_palette,
                starless_finish_report=local_finish,
            )

        self.assertFalse(report["accepted"], report)
        self.assertIn("frozen_sky_visible_noise", report["issues"])
        self.assertTrue(report["rollback"]["accepted"], report)
        self.assertEqual(processor._stage8_final_source, "stage8_input_starless")
        self.assertEqual(
            processor._stage8_final_quality,
            "final_cumulative_qa_rejected",
        )
        self.assertTrue(
            np.array_equal(
                processor.saved_image_pixels["stage8_enhanced"],
                baseline,
            )
        )
        self.assertIn(
            "stage8_final_cumulative_qa_rejected",
            processor._stage_review_reasons(8),
        )

    def test_formal_stage8_handoff_requires_fresh_cumulative_report(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        processor.image_pixels = processor.image_pixels.astype(np.float32)
        processor._save_stage_output("stage8_input_starless")
        processor._save_stage_output("stage8_enhanced")
        processor._stage8_input_source = "stage8_input_starless"

        for cumulative in (
            {},
            {
                "schema": "starun.stage8-final-cumulative-quality.v1",
                "status": "accepted",
                "accepted": True,
                "fresh_evaluation": False,
                "issues": [],
            },
        ):
            processor._stage8_final_cumulative_quality_report = cumulative
            handoff = runtime._set_stage8_handoff(
                processor,
                source_stem="stage8_enhanced",
                passthrough=False,
                restricted_downstream=False,
                final_quality="ok",
                processing_route="structure_enhanced",
                formal_eligible=True,
            )
            self.assertFalse(handoff["formal_eligible"], handoff)
            self.assertTrue(handoff["restricted_downstream"], handoff)
            self.assertFalse(
                handoff["final_cumulative_quality_verified"],
                handoff,
            )

        processor._stage8_final_cumulative_quality_report = {
            "schema": "starun.stage8-final-cumulative-quality.v1",
            "status": "accepted",
            "accepted": True,
            "fresh_evaluation": True,
            "issues": [],
        }
        handoff = runtime._set_stage8_handoff(
            processor,
            source_stem="stage8_enhanced",
            passthrough=False,
            restricted_downstream=False,
            final_quality="ok",
            processing_route="structure_enhanced",
            formal_eligible=True,
        )
        self.assertTrue(handoff["formal_eligible"], handoff)
        self.assertFalse(handoff["restricted_downstream"], handoff)
        self.assertTrue(handoff["final_cumulative_quality_verified"], handoff)
        self.assertTrue(
            handoff["handoff_integrity"]["writer_self_verified"],
            handoff,
        )
        self.assertTrue(
            runtime.verify_stage8_handoff_integrity(handoff)["accepted"],
            handoff,
        )
        self.assertEqual(
            processor.stage_json_reports["stage8_handoff.json"]
            ["handoff_integrity"]["canonical_sha256"],
            handoff["handoff_integrity"]["canonical_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
