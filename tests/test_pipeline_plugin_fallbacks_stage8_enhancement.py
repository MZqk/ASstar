"""Pipeline/plugin fallback tests for stage8 enhancement."""

from contextlib import nullcontext

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage8EnhancementTests(PipelinePluginFallbackTestBase):
    def _subject_chroma_processor(self):
        processor = self._new_processor()
        processor.cfg.stage8_target_aware_chroma_enabled = True
        processor.cfg.stage8_nebula_saturation_enabled = True
        processor._channel_semantics = "broadband_rgb_osc"
        processor._stage7_stretch_accepted = True
        processor._stage8_final_quality = "ok"
        processor._star_preserve_target_bypass = False
        processor._stage7_target_stretch_profile = lambda: {
            "name": "generic_balanced"
        }
        height, width = 96, 128
        subject = np.zeros((height, width), dtype=np.float32)
        subject[20:76, 24:104] = 1.0
        background = 1.0 - subject
        image = np.full((3, height, width), 0.08, dtype=np.float32)
        image[:, subject > 0.5] = np.asarray(
            (0.42, 0.25, 0.14),
            dtype=np.float32,
        )[:, None]
        processor.image_pixels = image.copy()
        processor.saved_image_pixels["stage8_enhanced"] = image.copy()
        (processor.process_dir / "stage8_enhanced.fit").write_bytes(b"mock")
        masks = {
            "subject_mask": subject,
            "nebula_mask": subject,
            "faint_nebula_mask": np.zeros_like(subject),
            "core_mask": np.zeros_like(subject),
            "background_mask": background,
            "star_mask": np.zeros_like(subject),
        }
        processor._stage7_frozen_rendition_masks = {
            key: np.array(value, copy=True) for key, value in masks.items()
        }
        processor._stage8_generate_starless_masks = lambda _image: {
            key: np.array(value, copy=True) for key, value in masks.items()
        }
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        processor._set_current_image_pixeldata = lambda image, **_kwargs: setattr(
            processor,
            "image_pixels",
            np.asarray(image).copy(),
        )
        return processor

    def _star_preserve_nebulosity_processor(self):
        processor = self._new_processor()
        processor._star_preserve_target_bypass = True
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor.target_profile = {
            "primary_target": {
                "name": "NGC6910",
                "type": "open_cluster",
                "frozen": True,
            },
            "secondary_labels": [
                "bright_core",
                "large_nebulosity",
                "emission_red",
            ],
        }
        height, width = 96, 128
        y, x = np.mgrid[:height, :width]
        diffuse = 0.04 + 0.12 * np.exp(
            -(((x - 66) / 28.0) ** 2 + ((y - 48) / 20.0) ** 2)
        )
        processor.image_pixels = np.stack(
            (diffuse * 1.20, diffuse * 0.92, diffuse * 0.75),
            axis=0,
        ).astype(np.float32)
        for row, col in (
            (20, 25),
            (35, 55),
            (45, 82),
            (60, 42),
            (70, 95),
            (28, 106),
        ):
            processor.image_pixels[:, row - 1, col - 1] = (0.80, 0.75, 0.70)
        processor.saved_image_pixels["stage7_stretched"] = (
            processor.image_pixels.copy()
        )
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        processor.siril.set_image_pixeldata = lambda image: setattr(
            processor,
            "image_pixels",
            np.asarray(image).copy(),
        )
        return processor

    def test_stage8_star_preserve_nebulosity_overlay_restores_star_pixels(self):
        processor = self._star_preserve_nebulosity_processor()
        source = processor.image_pixels.copy()

        output, report = (
            pipeline_module.stage8_pixels.stage8_star_preserve_nebulosity_overlay(
                processor,
                source,
            )
        )

        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(
            report["metrics"]["protected_star_max_abs_change"],
            0.0,
        )
        self.assertGreater(report["metrics"]["changed_pixel_ratio"], 0.01)
        self.assertGreater(float(np.max(np.abs(output - source))), 0.0)

    def test_stage8_routes_stellar_primary_with_nebulosity_to_bounded_overlay(self):
        processor = self._star_preserve_nebulosity_processor()

        stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        self.assertEqual(report["mode"], "star_preserve_secondary_nebulosity")
        self.assertTrue(report["secondary_context"]["primary_policy_unchanged"])
        self.assertTrue(report["secondary_nebulosity_overlay"]["accepted"])
        self.assertEqual(
            processor._stage8_final_quality,
            "star_preserve_secondary_nebulosity",
        )
        self.assertFalse(processor._stage8_handoff["passthrough"])
        palette_report = processor.stage_json_reports[
            "stage8_palette_report.json"
        ]
        self.assertEqual(
            palette_report["schema"],
            "starun.stage8-dualband-palette.v2",
        )
        self.assertFalse(palette_report["accepted"])
        self.assertEqual(report["dualband_palette"], palette_report)
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "completed",
        )

    def test_stage8_nebulosity_overlay_failure_preserves_stellar_route(self):
        processor = self._star_preserve_nebulosity_processor()
        processor.siril.get_image_pixeldata = lambda preview=False: None

        stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        self.assertEqual(report["mode"], "star_preserve_target_bypass")
        self.assertEqual(
            report["secondary_nebulosity_overlay"]["status"],
            "failed_safe_passthrough",
        )
        self.assertEqual(processor._stage8_final_quality, "star_preserve_bypass")
        self.assertTrue(processor._stage8_handoff["passthrough"])

    def test_stage8_subject_chroma_transaction_is_bounded_and_audited(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._subject_chroma_processor()
        baseline = processor.image_pixels.copy()
        report = runtime._stage8_run_subject_chroma(
            processor,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="broadband_rgb_osc",
            processing_policy="full",
            user_processing_mode="auto",
            external_override=False,
            requested_saturation_budget=0.40,
            effective_saturation_budget=0.40,
            generic_saturation_suppressed=True,
        )

        self.assertEqual(report["schema"], "starun.stage8-subject-chroma.v1")
        self.assertTrue(report["accepted"], report)
        self.assertTrue(report["feeds_main_pipeline"])
        self.assertEqual(report["requested_saturation_budget"], 0.40)
        self.assertEqual(report["effective_saturation_budget"], 0.40)
        self.assertEqual(
            report["generic_saturation_execution"]["reason"],
            "reserved_for_stage8_target_aware_subject_chroma",
        )
        candidate = processor.saved_image_pixels[
            "stage8_subject_chroma_candidate"
        ]
        baseline_luma = (
            0.2126 * baseline[0] + 0.7152 * baseline[1] + 0.0722 * baseline[2]
        )
        candidate_luma = (
            0.2126 * candidate[0]
            + 0.7152 * candidate[1]
            + 0.0722 * candidate[2]
        )
        np.testing.assert_allclose(candidate_luma, baseline_luma, atol=1e-6)
        background = report["candidate"]["background_unchanged"]
        self.assertTrue(background)
        self.assertEqual(
            processor._stage8_saturation_execution["method"],
            "masked_subject_chroma_rendition",
        )
        self.assertTrue(
            processor._stage8_saturation_execution[
                "generic_saturation_suppressed"
            ]
        )

    def test_stage8_subject_chroma_requires_reserved_budget(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._subject_chroma_processor()
        report = runtime._stage8_run_subject_chroma(
            processor,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="broadband_rgb_osc",
            processing_policy="full",
            user_processing_mode="auto",
            external_override=False,
            requested_saturation_budget=0.40,
            effective_saturation_budget=0.40,
            generic_saturation_suppressed=False,
        )

        self.assertFalse(report["accepted"])
        self.assertEqual(report["status"], "skipped_ineligible")
        self.assertIn(
            "generic_saturation_budget_not_reserved",
            report["eligibility"]["issues"],
        )
        self.assertNotIn(
            "stage8_pre_subject_chroma",
            processor.saved_image_pixels,
        )

    def test_stage8_starless_finish_projects_plugins_to_linked_subject_luma(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._subject_chroma_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        baseline = processor.image_pixels.copy()
        masks = processor._stage8_generate_starless_masks(baseline)
        core = np.zeros(baseline.shape[-2:], dtype=np.float32)
        core[42:48, 60:68] = 1.0
        masks["core_hard_mask"] = core
        processor._stage8_generate_starless_masks = lambda _image: {
            key: np.array(value, copy=True) for key, value in masks.items()
        }

        def run_plugin(step_key, _candidates):
            if step_key == "细节/结构增强2":
                processor.image_pixels = np.clip(
                    processor.image_pixels
                    * np.asarray((1.10, 1.03, 0.97), dtype=np.float32)[:, None, None],
                    0.0,
                    1.0,
                )
                return "VeraLux Revela"
            if step_key == "最终微调颜色":
                processor.image_pixels = np.clip(
                    processor.image_pixels
                    * np.asarray((1.04, 1.01, 0.98), dtype=np.float32)[:, None, None],
                    0.0,
                    1.0,
                )
                return "VeraLux Curves"
            return None

        processor._run_first_available_command = run_plugin
        report = runtime._stage8_run_starless_finish(
            processor,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="broadband_rgb_osc",
            processing_policy="full",
            user_processing_mode="auto",
            external_override=False,
            vectra_route_selected=False,
            effective_saturation_budget=0.40,
        )

        self.assertEqual(report["schema"], "starun.stage8-starless-finish.v1")
        self.assertEqual(report["accepted_steps"], ["revela", "subject_curves"])
        candidate = processor.saved_image_pixels["stage8_enhanced"]
        background = masks["background_mask"] >= 0.5
        np.testing.assert_array_equal(candidate[:, background], baseline[:, background])
        np.testing.assert_array_equal(candidate[:, core >= 0.5], baseline[:, core >= 0.5])
        subject = (masks["subject_mask"] > 0.5) & (core < 0.5)
        baseline_ratio = baseline[0, subject] / np.maximum(baseline[1, subject], 1e-6)
        candidate_ratio = candidate[0, subject] / np.maximum(candidate[1, subject], 1e-6)
        np.testing.assert_allclose(candidate_ratio, baseline_ratio, atol=2e-6)

    def test_stage8_vectra_is_exclusive_and_failure_has_no_subject_fallback(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._subject_chroma_processor()
        processor.cfg.workflow_plugin_probe_enabled = True

        def run_plugin(step_key, _candidates):
            if step_key != "调色2（可选）":
                return None
            luma = (
                0.2126 * processor.image_pixels[0]
                + 0.7152 * processor.image_pixels[1]
                + 0.0722 * processor.image_pixels[2]
            )
            processor.image_pixels = np.clip(
                luma[None, ...]
                + (processor.image_pixels - luma[None, ...]) * 1.12,
                0.0,
                1.0,
            ).astype(np.float32)
            return "VeraLux Vectra"

        processor._run_first_available_command = run_plugin
        finish = runtime._stage8_run_starless_finish(
            processor,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="broadband_rgb_osc",
            processing_policy="full",
            user_processing_mode="auto",
            external_override=False,
            vectra_route_selected=True,
            effective_saturation_budget=0.40,
        )
        self.assertTrue(processor._stage8_vectra_applied, finish)
        self.assertEqual(
            processor._stage8_saturation_execution["method"],
            "vectra_exclusive_color_route",
        )
        subject = runtime._stage8_run_subject_chroma(
            processor,
            [],
            base_stem="stage8_enhanced",
            channel_semantics="broadband_rgb_osc",
            processing_policy="full",
            user_processing_mode="auto",
            external_override=False,
            requested_saturation_budget=0.40,
            effective_saturation_budget=0.40,
            generic_saturation_suppressed=True,
            vectra_route_selected=True,
        )
        self.assertFalse(subject["accepted"])
        self.assertIn(
            "vectra_exclusive_color_route",
            subject["eligibility"]["issues"],
        )
        self.assertNotIn("stage8_subject_chroma_candidate", processor.saved_image_pixels)

    def test_stage8_starless_finish_skips_restricted_policy_without_probe(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        for policy in ("limited", "background_only", "skip", "preserve"):
            with self.subTest(policy=policy):
                processor = self._subject_chroma_processor()
                processor._run_first_available_command = lambda *_args: self.fail(
                    "restricted Stage8 route must not probe Starless plugins"
                )
                report = runtime._stage8_run_starless_finish(
                    processor,
                    [],
                    base_stem="stage8_enhanced",
                    channel_semantics="broadband_rgb_osc",
                    processing_policy=policy,
                    user_processing_mode="auto",
                    external_override=False,
                    vectra_route_selected=False,
                    effective_saturation_budget=0.40,
                )
                self.assertEqual(report["status"], "skipped_ineligible")
                self.assertIn(
                    "processing_policy_not_full",
                    report["eligibility"]["issues"],
                )

    def test_stage8_safe_passthrough_never_reenters_starless_finish_plugins(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._subject_chroma_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor._stage8_final_quality = "poor"
        processor._stage8_safe_passthrough_color_only_preflight = {
            "schema": "starun.stage8-safe-passthrough-preflight.v1",
            "status": "accepted",
            "accepted": True,
        }
        processor._run_first_available_command = lambda *_args: self.fail(
            "safe color-only passthrough must not probe Starless plugins"
        )

        report = runtime._stage8_run_starless_finish(
            processor,
            [],
            base_stem="stage8_input_starless",
            channel_semantics="broadband_rgb_osc",
            processing_policy="full",
            user_processing_mode="auto",
            external_override=False,
            vectra_route_selected=False,
            effective_saturation_budget=0.40,
        )

        self.assertEqual(
            report["status"],
            "skipped_safe_passthrough_color_only",
        )
        self.assertFalse(report["accepted"])
        self.assertEqual(report["steps"], [])
        self.assertFalse(processor._stage8_vectra_applied)

    def test_stage8_starless_finish_rollback_failure_requires_review(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._subject_chroma_processor()
        processor.cfg.workflow_plugin_probe_enabled = True
        processor._run_first_available_command = lambda *_args: "VeraLux Revela"
        with patch.object(
            runtime,
            "project_linked_luminance_candidate",
            side_effect=RuntimeError("forced projection failure"),
        ), patch.object(
            runtime,
            "_stage8_restore_finish_step",
            return_value=(False, "forced rollback failure"),
        ):
            report = runtime._stage8_run_starless_finish(
                processor,
                [],
                base_stem="stage8_enhanced",
                channel_semantics="broadband_rgb_osc",
                processing_policy="full",
                user_processing_mode="auto",
                external_override=False,
                vectra_route_selected=False,
                effective_saturation_budget=0.40,
            )

        self.assertEqual(report["status"], "failed_rollback_failed")
        self.assertEqual(report["steps"][0]["status"], "failed_rollback_failed")
        self.assertIn(
            "stage8_starless_finish_rollback_failed",
            processor._stage_review_reasons(8),
        )

    def test_stage8_starless_finish_step_unavailable_failure_and_rejection(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        for scenario in (
            "unavailable",
            "command_failure",
            "quality_rejection",
            "candidate_save_failure",
        ):
            with self.subTest(scenario=scenario):
                processor = self._subject_chroma_processor()
                processor.cfg.workflow_plugin_probe_enabled = True

                def run_plugin(*_args):
                    if scenario == "unavailable":
                        return None
                    if scenario == "command_failure":
                        raise RuntimeError("forced command failure")
                    processor.image_pixels = np.clip(
                        processor.image_pixels * 1.03,
                        0.0,
                        1.0,
                    )
                    return "VeraLux Revela"

                processor._run_first_available_command = run_plugin
                original_save = processor._save_stage_output
                if scenario == "candidate_save_failure":
                    processor._save_stage_output = lambda stem: (
                        False
                        if stem == "stage8_revela_candidate"
                        else original_save(stem)
                    )
                gate_patch = (
                    patch.object(
                        runtime,
                        "assess_finish_candidate",
                        return_value={
                            "accepted": False,
                            "status": "rejected",
                            "issues": ["forced_rejection"],
                        },
                    )
                    if scenario == "quality_rejection"
                    else nullcontext()
                )
                with gate_patch:
                    step = runtime._stage8_run_starless_finish_step(
                        processor,
                        [],
                        step_id="revela",
                        step_key="细节/结构增强2",
                        command_candidates=[("Revela", ("revela",))],
                        base_stem="stage8_enhanced",
                        mode="structure",
                    )

                expected = {
                    "unavailable": "skipped_unavailable",
                    "command_failure": "failed_rolled_back",
                    "quality_rejection": "rejected_rolled_back",
                    "candidate_save_failure": "failed_rolled_back",
                }[scenario]
                self.assertEqual(step["status"], expected)
                if scenario != "unavailable":
                    self.assertTrue(step["transaction"]["rollback_ok"])

    def test_stage8_high_risk_generic_saturation_does_not_double_consume(self):
        processor = self._new_processor()
        processor.cfg.stage8_target_aware_chroma_enabled = True
        processor.cfg.stage8_nebula_saturation_enabled = True
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor.saved_image_pixels["stage7_stretched"] = (
            processor.image_pixels.copy()
        )
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"mock")
        processor._stage7_halo_residue_score = lambda: 1.0
        processor._stage7_effective_halo_threshold = lambda: 0.35

        stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports[
            "stage8_subject_chroma_report.json"
        ]
        self.assertEqual(report["status"], "skipped_ineligible")
        self.assertIn(
            "generic_saturation_budget_not_reserved",
            report["eligibility"]["issues"],
        )
        runtime_execution = report["generic_saturation_execution"][
            "runtime_execution"
        ]
        self.assertTrue(runtime_execution.get("applied", False))
        self.assertNotIn(
            "stage8_subject_chroma_candidate",
            processor.saved_image_pixels,
        )

    def test_stage8_subject_chroma_rollback_failure_is_review_only(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        for failure_mode in ("quality_reject", "exception"):
            with self.subTest(failure_mode=failure_mode):
                processor = self._subject_chroma_processor()
                original_save = processor._save_stage_output

                def save(stem):
                    if stem == "stage8_enhanced":
                        return False
                    return original_save(stem)

                processor._save_stage_output = save
                if failure_mode == "quality_reject":
                    patcher = patch.object(
                        runtime,
                        "assess_subject_chroma_candidate",
                        return_value={
                            "accepted": False,
                            "status": "rejected",
                            "issues": ["forced_reject"],
                        },
                    )
                else:
                    patcher = patch.object(
                        runtime,
                        "apply_subject_chroma_rendition",
                        side_effect=RuntimeError("forced failure"),
                    )
                with patcher:
                    report = runtime._stage8_run_subject_chroma(
                        processor,
                        [],
                        base_stem="stage8_enhanced",
                        channel_semantics="broadband_rgb_osc",
                        processing_policy="full",
                        user_processing_mode="auto",
                        external_override=False,
                        requested_saturation_budget=0.40,
                        effective_saturation_budget=0.40,
                        generic_saturation_suppressed=True,
                    )

                self.assertIn("rollback_failed", report["status"])
                self.assertEqual(
                    processor._stage8_final_quality,
                    "subject_chroma_rollback_failed",
                )
                self.assertEqual(
                    processor._stage8_final_source,
                    "stage8_pre_subject_chroma",
                )
                self.assertTrue(processor._stage8_fallback_used)
                self.assertIn(
                    "stage8_subject_chroma_rollback_failed",
                    processor._stage_review_reasons(8),
                )

    def test_stage8_palette_never_runs_on_background_or_preserve_policy(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        for policy in ("background_only", "preserve"):
            with self.subTest(policy=policy):
                processor = self._dualband_palette_processor(
                    requested_palette="auto"
                )
                processor.saved_image_pixels["stage8_enhanced"] = (
                    processor.image_pixels.copy()
                )
                processor.cmd_calls.clear()
                report = runtime._stage8_run_dualband_palette(
                    processor,
                    [],
                    base_stem="stage8_enhanced",
                    channel_semantics="narrowband_composite",
                    processing_policy=policy,
                    external_override=False,
                )

                self.assertEqual(report["status"], "skipped_ineligible")
                self.assertIn(
                    "stage8_policy_not_full",
                    report["eligibility"]["issues"],
                )
                self.assertNotIn(
                    "stage8_pre_palette",
                    processor.saved_image_pixels,
                )
                self.assertFalse(
                    any(call[:1] == ("load",) for call in processor.cmd_calls)
                )

    def test_stage8_palette_rollback_failure_is_review_only(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._dualband_palette_processor(requested_palette="auto")
        processor.saved_image_pixels["stage8_enhanced"] = (
            processor.image_pixels.copy()
        )
        original_save = processor._save_stage_output

        def save(stem):
            if stem in {"stage8_palette_selected", "stage8_enhanced"}:
                return False
            return original_save(stem)

        processor._save_stage_output = save

        def palette_candidate(image, *, palette, **_kwargs):
            return np.asarray(image).copy(), {
                "accepted": True,
                "palette": palette,
                "synthetic_sii": palette != "HOO",
                "warnings": [],
                "metrics": {
                    "subject_background_chroma_separation_gain": 0.02,
                    "subject_saturation_p50_gain": 0.01,
                    "luminance_drift_p95": 0.0,
                    "clip_growth": 0.0,
                },
            }

        with patch.object(
            runtime,
            "build_dualband_palette_candidate",
            side_effect=palette_candidate,
        ):
            report = runtime._stage8_run_dualband_palette(
                processor,
                [],
                base_stem="stage8_enhanced",
                channel_semantics="narrowband_composite",
                processing_policy="full",
                external_override=False,
            )

        self.assertEqual(report["status"], "failed_rollback_failed")
        self.assertEqual(report["reason_code"], "stage8_palette_rollback_failed")
        self.assertEqual(processor._stage8_final_source, "stage8_pre_palette")
        self.assertEqual(processor._stage8_final_quality, "palette_rollback_failed")
        self.assertTrue(processor._stage8_fallback_used)
        self.assertIn(
            "stage8_palette_rollback_failed",
            processor._stage_review_reasons(8),
        )

    def test_stage8_mixed_nebula_saturation_is_bounded_above_galaxy_route(self):
        galaxy_name, galaxy_bands = (
            pipeline_module.stage8_pixels.stage8_broadband_hue_saturation_bands(
                "large_galaxy",
                0.10,
            )
        )
        mixed_name, mixed_bands = (
            pipeline_module.stage8_pixels.stage8_broadband_hue_saturation_bands(
                "bright_emission_reflection_nebula",
                0.10,
            )
        )

        self.assertEqual(galaxy_name, "galaxy")
        self.assertEqual(mixed_name, "mixed_nebula")
        self.assertGreater(
            max(band["amount"] for band in mixed_bands),
            max(band["amount"] for band in galaxy_bands),
        )
        self.assertLessEqual(
            max(band["amount"] for band in mixed_bands),
            0.05,
        )

    def test_stage8_resolved_mixed_composite_uses_larger_masked_color_budget(self):
        standard_name, standard_bands = (
            pipeline_module.stage8_pixels.stage8_broadband_hue_saturation_bands(
                "bright_emission_reflection_nebula",
                0.14,
            )
        )
        composite_name, composite_bands = (
            pipeline_module.stage8_pixels.stage8_broadband_hue_saturation_bands(
                "bright_emission_reflection_nebula",
                0.14,
                mixed_composite=True,
            )
        )
        processor = self._new_processor()
        processor.target_profile = {
            "primary_target": {
                "name": "Lagoon Nebula",
                "type": "bright_emission_reflection_nebula",
                "confidence": 0.98,
            },
            "secondary_labels": ["emission_red", "reflection_blue"],
            "composite_targets": [
                {"name": "Lagoon Nebula"},
                {"name": "Trifid Nebula"},
            ],
        }
        context = (
            pipeline_module.stage8_pixels.stage8_mixed_nebula_composite_context(
                processor
            )
        )

        self.assertEqual(standard_name, composite_name)
        self.assertTrue(context["eligible"])
        self.assertGreater(
            max(band["amount"] for band in composite_bands),
            max(band["amount"] for band in standard_bands),
        )
        self.assertLessEqual(
            max(band["amount"] for band in composite_bands),
            0.08,
        )

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
        palette_report = processor.stage_json_reports[
            "stage8_palette_report.json"
        ]
        self.assertFalse(palette_report["accepted"])
        self.assertFalse(palette_report["feeds_main_pipeline"])
        self.assertIn(
            "stage8_policy_not_full",
            palette_report["eligibility"]["issues"],
        )
        self.assertEqual(report["dualband_palette"], palette_report)

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

    def test_stage8_manual_ohs_palette_still_requires_positive_chroma_gain(self):
        processor = self._dualband_palette_processor(requested_palette="OHS")

        stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_palette_report.json"]
        self.assertFalse(report["accepted"], report)
        self.assertEqual(report["requested_palette"], "OHS")
        self.assertEqual(report["automatic_palette"], "SHO")
        self.assertEqual(report["planned_palette"], "OHS")
        self.assertEqual(report["selection_mode"], "explicit_user_palette")
        self.assertTrue(report["manual_override"])
        self.assertFalse(report["feeds_main_pipeline"])
        self.assertTrue(report["color_parent"]["degraded_pcc_applied"])
        self.assertTrue(report["color_parent"]["requires_review"])
        self.assertEqual(report["status"], "rejected_by_palette_quality_gate")
        self.assertIn("auto_palette_subject_chroma_gain_unmet", report["issues"])
        self.assertFalse(processor.cfg.optional_color_transform_enabled)
        self.assertEqual(
            processor.result_metadata[-1]["details"]["dualband_palette"],
            report,
        )

    def test_stage8_auto_palette_rolls_back_when_all_candidates_lose_chroma(self):
        processor = self._dualband_palette_processor(requested_palette="auto")

        stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_palette_report.json"]
        self.assertFalse(report["accepted"], report)
        self.assertEqual(report["requested_palette"], "auto")
        self.assertEqual(report["automatic_palette"], "SHO")
        self.assertEqual(report["selection_mode"], "automatic_target_mapping")
        self.assertFalse(report["manual_override"])
        self.assertEqual(report["candidate_count"], 6)
        self.assertEqual(report["status"], "rejected_by_palette_quality_gate")
        self.assertTrue(report["transaction"]["rollback_performed"])

    def test_stage8_auto_palette_selects_largest_positive_chroma_gain(self):
        processor = self._dualband_palette_processor(requested_palette="auto")
        gains = {
            "SHO": 0.01,
            "HOO": 0.02,
            "HSO": 0.03,
            "OSH": 0.06,
            "OHS": 0.04,
            "HOS": 0.05,
        }

        def build_candidate(image, *, palette, **_kwargs):
            gain = gains[palette]
            return np.array(image, copy=True), {
                "schema": "starun.stage8-dualband-palette.v2",
                "accepted": True,
                "status": "accepted",
                "palette": palette,
                "synthetic_sii": True,
                "metrics": {
                    "subject_background_chroma_separation_gain": gain,
                    "subject_saturation_p50_gain": gain * 0.8,
                    "luminance_drift_p95": 0.001,
                    "clip_growth": 0.0,
                },
                "issues": [],
                "warnings": [],
            }

        with patch(
            "stages.stage8_nebula_enhancement.build_dualband_palette_candidate",
            side_effect=build_candidate,
        ) as builder:
            stage8_nebula_enhancement(processor)

        report = processor.stage_json_reports["stage8_palette_report.json"]
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["palette"], "OSH")
        self.assertEqual(report["candidate_count"], 6)
        self.assertEqual(
            report["selection_execution_mode"],
            "automatic_candidate_competition",
        )
        self.assertEqual(builder.call_count, 6)

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
        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        self.assertIn("target_aware_subject_chroma", report["substeps"])
        self.assertTrue(report["substeps"]["target_aware_subject_chroma"])

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
        self.assertEqual(processor.results[-1][1], "degraded")
        report = processor.stage_json_reports["stage8_enhancement_report.json"]
        quality = processor.stage_json_reports["stage8_quality.json"]
        self.assertEqual(report["status"], "conservative_skipped")
        self.assertEqual(report["final_quality"], "conservative_skipped")
        self.assertEqual(quality["initial"]["status"], "conservative_skipped")
        self.assertEqual(quality["final"]["final_quality"], "conservative_skipped")
        handoff = processor.stage_json_reports["stage8_handoff.json"]
        self.assertFalse(handoff["formal_eligible"])
        self.assertTrue(handoff["restricted_downstream"])
        self.assertEqual(handoff["processing_route"], "review_only")
        self.assertIn(
            "stage8_input_guard_skip",
            processor._stage_review_reasons(8),
        )

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
        color_report = processor.stage_json_reports[
            "stage8_color_quality_report.json"
        ]
        self.assertTrue(color_report["used_for_gate"])
        self.assertIn("guard_lineage", color_report)
        self.assertIn("component_anomalies", color_report)
        self.assertIn("weakened_retry", color_report)
        self.assertIn("final_pixel_identity", color_report)
        handoff = processor.stage_json_reports["stage8_handoff.json"]
        self.assertEqual(
            handoff["color_gate"]["report"],
            "stage8_color_quality_report.json",
        )
        self.assertTrue(handoff["color_gate"]["used_for_gate"])
        palette_report = processor.stage_json_reports[
            "stage8_palette_report.json"
        ]
        self.assertFalse(palette_report["accepted"])
        self.assertIn(
            "stage8_policy_not_full",
            palette_report["eligibility"]["issues"],
        )
        self.assertEqual(report["dualband_palette"], palette_report)

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

    def test_stage8_input_guard_keeps_full_route_for_single_local_galaxy_halo(self):
        image = np.full((3, 32, 32), 0.05, dtype=np.float32)
        raw_local_halo = 1.376
        probe = SimpleNamespace(
            _stage8_handoff={
                "processing_policy": "full",
                "suppressed_advisories": [
                    "galaxy_disk_halo_residue 1.376>0.480"
                ],
            },
            _stage7_selected_quality={
                "status": "ok",
                "advisories": ["galaxy_disk_halo_residue 1.376>0.480"],
                "quality_gates": {
                    "galaxy_disk_halo_residue": {
                        "status": "advisory",
                        "advisory": True,
                        "hard_failed": False,
                        "reason_code": "single_local_galaxy_halo_evidence",
                        "value": raw_local_halo,
                        "accepted_limit": 0.48,
                        "hard_limit": 0.96,
                    },
                },
                "derived": {
                    "residual_star_score": 0.10,
                    "halo_residue_score": raw_local_halo,
                    "global_halo_residue_score": 0.0556,
                    "compact_halo_residue_score": 0.1974,
                    "galaxy_disk_halo_corroborated_local_count": 1,
                    "galaxy_disk_halo_evidence_available": 1.0,
                    "galaxy_disk_halo_residue_score": raw_local_halo,
                    "starless_noise_gain": 1.0,
                },
            },
            _stage7_starless_skipped=False,
            cfg=SimpleNamespace(
                stage8_processing_mode="auto",
                stage8_masked_enhancement_enabled=True,
                stage7_residual_star_score_max=0.45,
                stage7_halo_residue_score_max=0.35,
                stage7_starless_noise_gain_max=1.25,
                stage8_mask_signal_coverage_min=0.002,
            ),
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: image.copy()
            ),
            _stage7_halo_residue_score=lambda: raw_local_halo,
            _stage7_effective_halo_threshold=lambda: 0.48,
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

        self.assertEqual(report["processing_policy"], "full")
        self.assertFalse(report["background_only"])
        self.assertEqual(report["subject_reasons"], [])
        self.assertAlmostEqual(report["derived"]["halo_residue_score"], 0.1974)
        self.assertAlmostEqual(
            report["derived"]["raw_halo_residue_score"], raw_local_halo
        )
        self.assertTrue(
            report["derived"]["single_local_galaxy_halo_override_active"]
        )

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
        palette_report = processor.stage_json_reports[
            "stage8_palette_report.json"
        ]
        self.assertFalse(palette_report["accepted"])
        self.assertIn(
            "stage8_structural_quality_not_ok",
            palette_report["eligibility"]["issues"],
        )
        self.assertEqual(report["dualband_palette"], palette_report)

    def test_stage8_builtin_saturation_fallback_is_reported_as_internal_processing(self):
        processor = self._new_processor()

        stage8_nebula_enhancement(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("内置 Starless satu", message)
        self.assertIn("内置 Starless unsharp", message)
        self.assertNotIn("插件未命中", message)
        palette_report = processor.stage_json_reports[
            "stage8_palette_report.json"
        ]
        self.assertEqual(palette_report["status"], "skipped_ineligible")
        self.assertFalse(palette_report["accepted"])
        self.assertIn(
            "channel_semantics_not_narrowband_composite",
            palette_report["eligibility"]["issues"],
        )

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

    def test_stage8_limited_advisory_verified_noop_gets_formal_v3_handoff(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor.cfg.stage8_limited_saturation_max = 0.05
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor._save_stage_output("stage7_stretched")
        processor._stage8_handoff = {
            "processing_policy": "limited",
            "restricted_downstream": True,
            "quality_status": "ok",
            "reason_code": "stage6_quality_advisory",
            "reason_text": "stage6_quality_advisory: halo_residue 0.713>0.480",
            "reasons": [
                {
                    "code": "stage6_quality_advisory",
                    "source_stage": 6,
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
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._stage8_input_enhancement_guard = lambda: {
            "status": "ok",
            "skip_enhancement": False,
            "processing_policy": "limited",
            "reason_code": "stage6_quality_advisory",
            "reason_text": processor._stage8_handoff["reason_text"],
            "reason_details": processor._stage8_handoff["reasons"],
            "hard_reasons": [],
            "subject_reasons": [],
            "advisories": [processor._stage8_handoff["reason_text"]],
        }
        processor._apply_stage8_builtin_enhancement = (
            lambda _plan, *, label: [f"{label} verified noop"]
        )
        processor._stage8_quality_assessment = lambda: {
            "status": "ok",
            "issues": [],
        }
        processor._stage8_enhancement_quality_report = lambda: {
            "status": "ok",
            "issues": [],
            "advisories": [],
        }
        processor._stage8_star_halo_guard_report = {
            "schema": "starun.stage6-star-halo-guard.v1",
            "status": "ok",
            "reason_code": "stage6_star_halo_guard_ready",
            "artifact": "stage6_star_halo_guard.fit",
            "artifact_sha256": "a" * 64,
        }
        processor._stage8_star_halo_guard_verified = True
        (processor.process_dir / "stage3_spatial_background_lineage.json").write_text(
            "{}",
            encoding="utf-8",
        )

        eligibility_capture = {}

        def accepted_preflight(_pipeline, *, source_mode, eligibility):
            eligibility_capture.update(eligibility)
            report = {
                "schema": "starun.stage8-safe-passthrough-preflight.v1",
                "status": "accepted",
                "accepted": True,
                "source_mode": source_mode,
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
            processor._stage8_safe_passthrough_color_only_preflight = report
            return report

        def accepted_final(_pipeline, **_kwargs):
            return {
                "schema": "starun.stage8-safe-passthrough-final.v1",
                "status": "accepted",
                "accepted": True,
                "checks": {
                    "color": {"accepted": True},
                    "background_seam_clip_presentation": {"status": "ok"},
                    "spatial_background": {"accepted": True},
                    "star_halo": {"accepted": True},
                    "artifact": {"accepted": True},
                },
            }

        def accepted_color_report(_pipeline, *, final_source, **_kwargs):
            identity = runtime._stage8_source_identity(
                processor,
                final_source,
            )
            return {
                "schema": "starun.color-quality-report.v1",
                "status": "reported",
                "used_for_gate": True,
                "issues": [],
                "guard_lineage": {"verified": True},
                "final_pixel_identity": identity,
                "contract": {},
            }

        def accepted_cumulative(_pipeline, **_kwargs):
            report = {
                "schema": "starun.stage8-final-cumulative-quality.v1",
                "status": "accepted",
                "accepted": True,
                "fresh_evaluation": True,
                "issues": [],
            }
            _pipeline._stage8_final_cumulative_quality_report = dict(report)
            return report

        with patch.object(
            runtime.spatial_background_lineage,
            "load_lineage",
            return_value={
                "schema": "starun.stage3-spatial-background-lineage.v2",
                "status": "accepted",
                "accepted": True,
                "support_sha256": "b" * 64,
                "issues": [],
            },
        ), patch.object(
            runtime,
            "_stage8_safe_passthrough_preflight",
            side_effect=accepted_preflight,
        ), patch.object(
            runtime,
            "_stage8_safe_passthrough_final_validation",
            side_effect=accepted_final,
        ), patch.object(
            runtime,
            "_write_stage8_color_quality_report",
            side_effect=accepted_color_report,
        ), patch.object(
            runtime,
            "_stage8_enforce_final_cumulative_validation",
            side_effect=accepted_cumulative,
        ):
            stage8_nebula_enhancement(processor)

        handoff = processor._stage8_handoff
        self.assertTrue(eligibility_capture["accepted"], eligibility_capture)
        self.assertEqual(
            handoff["processing_route"],
            "safe_passthrough_color_only",
        )
        self.assertTrue(handoff["formal_eligible"], handoff)
        self.assertFalse(handoff["restricted_downstream"], handoff)
        self.assertTrue(handoff["passthrough"], handoff)
        self.assertEqual(
            handoff["outcome_reason_code"],
            "stage8_limited_safe_passthrough_accepted",
        )

    def test_stage8_cumulative_structure_rejection_retries_color_only(self):
        runtime = sys.modules["stages.stage8_nebula_enhancement"]
        processor = self._new_processor()
        processor.cfg.stage8_masked_enhancement_enabled = True
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor._save_stage_output("stage7_stretched")
        processor._stage8_handoff = {
            "processing_policy": "full",
            "restricted_downstream": False,
            "quality_status": "ok",
            "reason_code": "",
            "reasons": [],
        }
        processor._stage8_input_enhancement_guard = lambda: {
            "status": "ok",
            "skip_enhancement": False,
            "processing_policy": "full",
            "reason_code": "",
            "reason_text": "",
            "reason_details": [],
            "hard_reasons": [],
            "subject_reasons": [],
            "advisories": [],
        }
        processor._apply_stage8_builtin_enhancement = (
            lambda _plan, *, label: [f"{label} structure candidate"]
        )
        processor._stage8_quality_assessment = lambda **_kwargs: {
            "status": "ok",
            "issues": [],
        }
        processor._stage8_enhancement_quality_report = lambda: {
            "status": "ok",
            "issues": [],
            "advisories": [],
        }
        processor._stage8_star_halo_guard_report = {
            "schema": "starun.stage6-star-halo-guard.v1",
            "status": "ok",
            "reason_code": "stage6_star_halo_guard_ready",
            "artifact": "stage6_star_halo_guard.fit",
            "artifact_sha256": "a" * 64,
        }
        processor._stage8_star_halo_guard_verified = True
        (processor.process_dir / "stage3_spatial_background_lineage.json").write_text(
            "{}",
            encoding="utf-8",
        )
        cumulative_calls = []
        preflight_modes = []

        def cumulative(_pipeline, **kwargs):
            cumulative_calls.append(dict(kwargs))
            if len(cumulative_calls) == 1:
                self.assertTrue(kwargs["defer_review_on_exact_rollback"])
                baseline = np.array(
                    _pipeline.saved_image_pixels["stage8_input_starless"],
                    copy=True,
                )
                _pipeline.image_pixels = baseline
                _pipeline.saved_image_pixels["stage8_enhanced"] = baseline
                _pipeline._stage8_final_source = "stage8_input_starless"
                _pipeline._stage8_final_quality = (
                    "final_cumulative_qa_rejected"
                )
                report = {
                    "schema": "starun.stage8-final-cumulative-quality.v1",
                    "status": "rejected",
                    "accepted": False,
                    "fresh_evaluation": True,
                    "issues": ["outside_target_pixel_identity"],
                    "rollback": {"status": "restored", "accepted": True},
                    "review_deferred_for_safe_passthrough": True,
                    "reason_code": "stage8_final_cumulative_qa_rejected",
                }
                _pipeline._stage8_final_cumulative_quality_report = dict(
                    report
                )
                return report
            report = {
                "schema": "starun.stage8-final-cumulative-quality.v1",
                "status": "accepted",
                "accepted": True,
                "fresh_evaluation": True,
                "issues": [],
                "reason_code": "accepted",
            }
            _pipeline._stage8_final_cumulative_quality_report = dict(report)
            return report

        def accepted_preflight(_pipeline, *, source_mode, eligibility=None):
            _ = eligibility
            preflight_modes.append(source_mode)
            report = {
                "schema": "starun.stage8-safe-passthrough-preflight.v1",
                "status": "accepted",
                "accepted": True,
                "source_mode": source_mode,
                "checks": {"exact_structure_rollback": {"accepted": True}},
                "issues": [],
            }
            _pipeline._stage8_safe_passthrough_color_only_preflight = report
            return report

        def accepted_subject(_pipeline, _messages, **kwargs):
            report = {
                "schema": "starun.stage8-subject-chroma.v1",
                "status": "accepted",
                "accepted": True,
                "source": f"{kwargs['base_stem']}.fit",
                "output": "stage8_enhanced.fit",
                "factor": {"factor": 1.02},
            }
            _pipeline._stage8_subject_chroma_report = report
            return report

        def accepted_color_report(_pipeline, *, final_source, **_kwargs):
            return {
                "schema": "starun.color-quality-report.v1",
                "status": "reported",
                "used_for_gate": True,
                "issues": [],
                "guard_lineage": {"verified": True},
                "final_pixel_identity": runtime._stage8_source_identity(
                    _pipeline,
                    final_source,
                ),
                "contract": {},
            }

        with patch.object(
            runtime.spatial_background_lineage,
            "load_lineage",
            return_value={
                "schema": "starun.stage3-spatial-background-lineage.v2",
                "status": "accepted",
                "accepted": True,
                "support_sha256": "b" * 64,
                "issues": [],
            },
        ), patch.object(
            runtime,
            "_stage8_safe_passthrough_preflight",
            side_effect=accepted_preflight,
        ), patch.object(
            runtime,
            "_stage8_safe_passthrough_final_validation",
            return_value={
                "schema": "starun.stage8-safe-passthrough-final.v1",
                "status": "accepted",
                "accepted": True,
                "checks": {"color": {"accepted": True}},
                "issues": [],
            },
        ), patch.object(
            runtime,
            "_stage8_run_subject_chroma",
            side_effect=accepted_subject,
        ), patch.object(
            runtime,
            "_stage8_enforce_final_cumulative_validation",
            side_effect=cumulative,
        ), patch.object(
            runtime,
            "_write_stage8_color_quality_report",
            side_effect=accepted_color_report,
        ):
            stage8_nebula_enhancement(processor)

        handoff = processor._stage8_handoff
        self.assertEqual(len(cumulative_calls), 2)
        self.assertEqual(
            preflight_modes,
            ["post_cumulative_structure_rollback"],
        )
        self.assertEqual(
            handoff["processing_route"],
            "safe_passthrough_color_only",
        )
        self.assertTrue(handoff["formal_eligible"], handoff)
        self.assertFalse(handoff["restricted_downstream"], handoff)
        self.assertTrue(handoff["structure_cumulative_retry"]["accepted"])
        self.assertEqual(
            handoff["outcome_reason_code"],
            "stage8_structure_cumulative_rollback_"
            "safe_passthrough_accepted",
        )
        self.assertFalse(processor._stage_review_reasons(8))

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

    def test_stage8_narrowband_zero_saturation_keeps_structure_steps(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "narrowband_composite"
        processor._active_target_type = lambda: "emission_nebula_widefield"
        processor._stage7_halo_residue_score = lambda: 0.0
        processor._stage7_effective_halo_threshold = lambda: 0.35
        image = np.full((3, 96, 96), 0.04, dtype=np.float32)
        yy, xx = np.indices((96, 96))
        signal = np.exp(
            -(((xx - 48) / 23.0) ** 2 + ((yy - 48) / 17.0) ** 2)
        ).astype(np.float32)
        image += np.asarray([0.18, 0.11, 0.09], dtype=np.float32)[
            :, None, None
        ] * signal[None]

        enhanced, diagnostics, _messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.0, "unsharp_amount": 0.0},
                label="test",
            )
        )

        structure = diagnostics["structure_execution"]
        self.assertGreater(structure["faint_nebula_boost"], 0.0)
        self.assertGreater(structure["nebula_contrast"], 0.0)
        self.assertTrue(structure["independent_from_saturation"])
        self.assertEqual(diagnostics["saturation_execution"]["passes"], 0)
        self.assertGreater(float(np.max(np.abs(enhanced - image))), 0.0)

    def test_stage8_limited_pixels_are_core_excluded_and_structure_disabled(self):
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
        self.assertEqual(float(np.max(delta[weak_signal > 0.0])), 0.0)
        self.assertEqual(diagnostics["structure_execution"]["scale"], 0.0)
        self.assertEqual(
            diagnostics["structure_execution"]["faint_nebula_boost"],
            0.0,
        )
        self.assertEqual(
            diagnostics["structure_execution"]["nebula_contrast"],
            0.0,
        )
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

    def test_stage8_resolved_mixed_composite_uses_budgeted_masked_chroma_recovery(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "broadband_rgb_osc"
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        processor._frozen_primary_target = {
            "type": "bright_emission_reflection_nebula",
            "confidence": 0.98,
            "method": "catalog_name_wcs_composite_match",
        }
        processor.target_profile = {
            "primary_target": {
                **processor._frozen_primary_target,
                "name": "Lagoon Nebula",
            },
            "secondary_labels": ["emission_red", "reflection_blue"],
            "composite_targets": [
                {"name": "Lagoon Nebula"},
                {"name": "Trifid Nebula"},
            ],
        }
        processor._stage7_halo_residue_score = lambda: 0.10
        processor._stage7_effective_halo_threshold = lambda: 0.35
        height, width = 120, 160
        yy, xx = np.indices((height, width))
        signal = np.exp(
            -(((xx - 80) / 38.0) ** 2 + ((yy - 60) / 28.0) ** 2)
        ).astype(np.float32)
        image = np.full((3, height, width), 0.035, dtype=np.float32)
        image += np.asarray([0.30, 0.14, 0.22], dtype=np.float32)[
            :, None, None
        ] * signal[None]

        _enhanced, diagnostics, _messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.14, "unsharp_amount": 0.0},
                label="test",
            )
        )

        operations = diagnostics["local_adjustment_engine"]["operations"]
        operation_types = [operation["type"] for operation in operations]
        self.assertEqual(operation_types.count("saturation"), 1)
        self.assertEqual(operation_types.count("hue_selective_saturation"), 1)
        broad = next(
            operation
            for operation in operations
            if operation["type"] == "saturation"
        )
        self.assertLessEqual(broad["effective_amount_peak"], 0.115)
        self.assertEqual(
            diagnostics["local_adjustment_engine"]["metrics"]["clip_growth"],
            0.0,
        )
        self.assertEqual(
            diagnostics["local_adjustment_engine"]["metrics"][
                "outside_mask_changed_ratio"
            ],
            0.0,
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

    def test_stage8_generic_saturation_reports_measured_amount(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._stage8_handoff = {"processing_policy": "full"}
        processor._channel_semantics = "broadband_rgb_osc"
        processor._active_target_type = lambda: "generic_low_snr_safe"
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

        _enhanced, diagnostics, _messages = (
            processor._apply_stage8_masked_pixel_enhancement(
                image,
                {"saturation": 0.10, "unsharp_amount": 0.0},
                label="test",
            )
        )

        saturation_execution = diagnostics["saturation_execution"]
        self.assertTrue(saturation_execution["applied"])
        self.assertEqual(saturation_execution["effect_status"], "effective")
        self.assertGreater(saturation_execution["applied_amount"], 0.0)
        self.assertGreater(
            saturation_execution["effective_amount_mean"],
            0.0,
        )

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
