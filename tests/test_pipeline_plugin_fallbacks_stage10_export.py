"""Pipeline/plugin fallback tests for stage10 export."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403
from managed_output import _read_managed_display_png


class PipelinePluginFallbackStage10ExportTests(PipelinePluginFallbackTestBase):
    def test_stage10_catalog_failure_withholds_review_even_with_generic_peaks(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage10_test_use_real_catalog_visibility = True
        processor.cfg.stage10_final_denoise_enabled = False
        processor.cfg.stage10_final_saturation_enabled = False
        processor.cfg.stage10_managed_output_enabled = False
        processor._require_review(3, "stage3_background_review_required")
        image = np.full((3, 32, 32), 0.14, dtype=np.float32)
        for y, x in (
            (3, 15),
            (4, 24),
            (8, 2),
            (10, 29),
            (15, 8),
            (16, 24),
            (21, 3),
            (23, 29),
            (28, 9),
            (29, 20),
        ):
            image[:, y, x] = 0.95
        processor.image_pixels = image
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        processor._set_current_image_pixeldata = (
            lambda pixels, **_kwargs: setattr(
                processor,
                "image_pixels",
                np.array(pixels, copy=True),
            )
        )

        stage10_export(processor)

        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "required_stars_catalog_visibility_failed",
        )
        self.assertEqual(processor.results[-1][1], "failed")
        self.assertFalse(
            any(
                call[0] in {"savetif", "savepng"}
                for call in processor.cmd_calls
            )
        )
        audit = processor.stage_json_reports[
            "stage10_pre_export_visibility.json"
        ]["audit"]
        self.assertTrue(
            audit["checks"]["star_visibility"][
                "compact_peak_diagnostic"
            ]["passed"]
        )
        self.assertFalse(
            audit["checks"]["star_visibility"]["passed"]
        )

    def test_stage10_legacy_accepted_hdr_state_has_no_preserve_exception(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage9_stars_required = False
        processor._stage9_stars_applied = False
        processor._stage9_output_contains_stars = True
        processor._stage9_star_delivery_contract_accepted = True
        processor._bright_core_with_stars_fallback = {
            "eligible": True,
            "accepted": True,
            "status": "accepted",
            "output_stem": "stage7_with_stars_hdr",
        }
        processor._final_quality_report = lambda _stem: {
            "schema": "starun.final-quality.v2",
            "severity": "normal",
            "final_quality": "ok",
            "status": "ok",
            "needs_conservative_rerun": False,
            "issues": [],
        }

        stage10_export(processor)

        self.assertTrue(any(call[0] == "satu" for call in processor.cmd_calls))
        self.assertFalse(processor.script_calls)
        self.assertIn(("savetif", "result_processed", "-astro"), processor.cmd_calls)
        self.assertFalse(processor._final_output_review_only)
        denoise = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertNotIn("bright_core_with_stars_hdr_preserve", denoise)
        self.assertNotIn("skipped_by_bright_core_with_stars_hdr", denoise)
        self.assertEqual(
            processor._stage10_quality_repair_report["reason"],
            "final_quality_is_not_hard_reject",
        )

    def test_stage10_review_primary_png_matches_managed_display(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.output_format = "png"
        processor.cfg.force_review_only_output = True
        processor.cfg.stage10_managed_output_enabled = True
        yy, xx = np.mgrid[:96, :128]
        nebula = np.exp(
            -0.5
            * (
                ((xx - 66.0) / 23.0) ** 2
                + ((yy - 46.0) / 18.0) ** 2
            )
        )
        processor.image_pixels = np.stack(
            (
                0.02 + 0.36 * nebula,
                0.018 + 0.22 * nebula,
                0.021 + 0.30 * nebula,
            )
        ).astype(np.float32)
        for y_pos, x_pos in ((8, 11), (20, 70), (44, 21), (72, 99), (86, 48)):
            processor.image_pixels[:, y_pos, x_pos] = (0.95, 0.88, 0.82)
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        processor._set_current_image_pixeldata = (
            lambda image, **_kwargs: setattr(
                processor,
                "image_pixels",
                np.array(image, copy=True),
            )
        )

        stage10_export(processor)

        primary = processor.work_dir / "result_review.png"
        managed = processor.work_dir / "result_review_display_srgb.png"
        self.assertTrue(primary.is_file())
        self.assertTrue(managed.is_file())
        self.assertEqual(primary.read_bytes(), managed.read_bytes())
        export_report = processor.stage_json_reports["stage10_export_report.json"]
        self.assertEqual(
            export_report["review_display"]["pixel_identity"],
            "byte_identical_copy",
        )

    def test_stage2_review_preserves_already_visible_stage10_pixels(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.output_format = "png"
        processor.cfg.stage10_managed_output_enabled = True
        processor._require_review(2, "stage2_view_review_required")
        processor._display_rendition_contract = {
            "schema": "starun.display-rendition-contract.v1",
            "status": "ready",
            "applicable": True,
            "observer_only": True,
            "name": "linked_review_bright_v1",
            "reason": "stage2_view_review_required",
            "luminance": {
                "white_percentile": 0.995,
                "white_point": 0.50,
                "gamma": 0.50,
            },
            "rgb_mapping": {
                "linked_channels": True,
                "gamut_policy": "shared_per_pixel_scale",
            },
        }
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        yy, xx = np.mgrid[:96, :128]
        nebula = np.exp(
            -0.5
            * (
                ((xx - 66.0) / 24.0) ** 2
                + ((yy - 47.0) / 19.0) ** 2
            )
        )
        source = np.stack(
            (
                0.14 + 0.28 * nebula,
                0.13 + 0.20 * nebula,
                0.12 + 0.24 * nebula,
            )
        ).astype(np.float32)
        for y_pos, x_pos in (
            (8, 11),
            (20, 70),
            (44, 21),
            (72, 99),
            (86, 48),
        ):
            source[:, y_pos, x_pos] = (0.95, 0.88, 0.82)
        processor.image_pixels = source.copy()
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        processor._set_current_image_pixeldata = (
            lambda image, **_kwargs: setattr(
                processor,
                "image_pixels",
                np.array(image, copy=True),
            )
        )

        stage10_export(processor)

        contract = processor.stage_json_reports[
            "display_rendition_contract.json"
        ]
        self.assertEqual(contract["mode"], "preserve")
        self.assertEqual(contract["source_visibility"]["exposure_state"], "acceptable")
        self.assertFalse(contract["rgb_mapping"]["derivative_pixels_changed"])
        self.assertNotIn(("autostretch", "-linked"), processor.cmd_calls)
        managed = _read_managed_display_png(
            processor.work_dir / "result_review_display_srgb.png"
        )
        np.testing.assert_allclose(
            managed,
            np.flip(source, axis=1),
            atol=1.0 / 65535.0,
            rtol=0.0,
        )
        self.assertEqual(
            (processor.work_dir / "result_review.png").read_bytes(),
            (processor.work_dir / "result_review_display_srgb.png").read_bytes(),
        )

    def test_stage10_script_failure_with_aberration_fallback_is_ok(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/SCUNet_Denoise.py",
            }
        )
        processor.script_fail_steps.add("最终降噪")
        processor.script_fail_steps.add("最终降噪回退")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=SASP Aberration API", message)
        self.assertIn("fallback_status=success", message)
        self.assertIn("final_denoise_effective=SASP Aberration API", message)
        self.assertIn("effective_status=success", message)
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "denoiser_chain_to_aberration",
        )

    def test_stage10_narrowband_skips_global_saturation(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._channel_semantics = "narrowband_composite"

        stage10_export(processor)

        self.assertFalse(any(call[0] == "satu" for call in processor.cmd_calls))
        self.assertIn(
            "Stage10 global color adjustment skipped by channel semantics "
            "(narrowband_composite)",
            processor.results[-1][3],
        )

    def test_stage10_color_dominant_input_uses_chroma_plan_with_full_model(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.cfg.stage10_final_denoise_strength = 0.31
        pixels = np.full((3, 32, 32), 0.05, dtype=np.float32)
        processor.siril.get_image_pixeldata = lambda preview=False: pixels
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.431,
            "bg_std": 0.003,
            "background_mottling_score": 0.144,
        }
        processor._set_current_image_pixeldata = lambda _image, **_kwargs: None

        stage10_export(processor)

        denoise_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Denoise.py"
        ]
        self.assertTrue(denoise_calls)
        args = denoise_calls[0][2]
        mode_index = args.index("-denoising_mode") + 1
        self.assertEqual(args[mode_index], "full")
        strength_index = args.index("-denoise_strength") + 1
        self.assertEqual(args[strength_index], "0.31")
        report = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(report["selected_mode"], "chroma")
        self.assertEqual(report["effective_mode"], "chroma")
        self.assertEqual(report["requested_strength"], 0.31)
        self.assertTrue(report["star_protection"]["applied"])

    def test_stage10_missing_validated_star_mask_skips_denoise_fail_closed(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor._stage9_star_reference_catalog = {
            "status": "unavailable",
            "reason": "mock catalog rejection",
        }

        stage10_export(processor)

        self.assertFalse(
            any(step == "最终降噪" for step, _name, _args in processor.script_calls)
        )
        report = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(report["effective_status"], "skipped_safe")
        self.assertTrue(report["skipped_by_star_protection_guard"])
        self.assertEqual(
            report["star_protection"]["reason"],
            "mock catalog rejection",
        )

    def test_stage10_applies_star_protection_after_successful_backend(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        original = np.full((3, 32, 32), 0.8, dtype=np.float32)
        denoised = np.full((3, 32, 32), 0.2, dtype=np.float32)
        state = {"pixels": original.copy()}
        processor.siril.get_image_pixeldata = (
            lambda preview=False: state["pixels"].copy()
        )
        processor._set_current_image_pixeldata = (
            lambda image, **_kwargs: state.__setitem__(
                "pixels",
                np.array(image, copy=True),
            )
        )

        def run_backend(*_args, **_kwargs):
            state["pixels"] = denoised.copy()
            return "CosmicClarity Denoise script"

        processor._run_plugin_script_by_path = run_backend

        stage10_export(processor)

        np.testing.assert_array_equal(
            state["pixels"][:, 8, 9],
            original[:, 8, 9],
        )
        np.testing.assert_array_equal(
            state["pixels"][:, 0, 0],
            denoised[:, 0, 0],
        )
        report = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(report["effective_status"], "success")
        self.assertTrue(report["star_protection"]["applied"])

    def test_stage10_rolls_back_when_star_protection_merge_fails(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        original = np.full((3, 32, 32), 0.8, dtype=np.float32)
        denoised = np.full((3, 32, 32), 0.2, dtype=np.float32)
        state = {"pixels": original.copy()}
        processor.siril.get_image_pixeldata = (
            lambda preview=False: state["pixels"].copy()
        )

        def set_pixels(image, *, label):
            if label == "Stage10 star-protected denoise merge":
                raise RuntimeError("mock protected merge failure")
            state["pixels"] = np.array(image, copy=True)

        processor._set_current_image_pixeldata = set_pixels

        def run_backend(*_args, **_kwargs):
            state["pixels"] = denoised.copy()
            return "CosmicClarity Denoise script"

        processor._run_plugin_script_by_path = run_backend

        stage10_export(processor)

        np.testing.assert_array_equal(state["pixels"], original)
        report = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(report["effective_status"], "rolled_back_safe")
        self.assertEqual(report["effective_mode"], "skipped")
        self.assertEqual(
            report["star_protection"]["rollback_status"],
            "success",
        )

    def test_stage10_low_noise_input_skips_expensive_denoiser(self):
        processor = self._new_processor()
        (processor.process_dir / "stage9_remixed.fit").write_bytes(b"mock")
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        pixels = np.full((3, 32, 32), 0.05, dtype=np.float32)
        processor.siril.get_image_pixeldata = lambda preview=False: pixels
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.058,
            "bg_std": 0.00084,
            "background_mottling_score": 0.031,
        }

        stage10_export(processor)

        self.assertFalse(
            any(step == "最终降噪" for step, _name, _args in processor.script_calls)
        )
        report = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(report["selected_mode"], "skip")
        self.assertEqual(report["effective_status"], "skipped_safe")
        self.assertTrue(report["skipped_by_low_noise_guard"])
        self.assertFalse(report["skipped_by_review_only"])
        self.assertFalse(report["skipped_by_duplicate_guard"])
        self.assertIn("Stage10 low-noise guard", processor.results[-1][3])
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["components"]["denoise"]["reason_code"],
            "auto_low_noise",
        )

    def test_stage10_script_failure_prefers_scunet_command_fallback(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.cfg.stage10_final_denoise_strength = 0.33
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_fail_steps.add("最终降噪")
        processor.available_commands.add("siril_scunet_denoise")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=Siril-SCUNet Denoise", message)
        self.assertIn("fallback_status=success", message)
        self.assertNotIn("fallback_component=SASP Aberration API", message)
        self.assertIn(("siril_scunet_denoise", "0.33"), processor.cmd_calls)
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["components"]["denoise"]["method"],
            "Siril-SCUNet Denoise",
        )

    def test_stage10_uses_in_process_cosmic_clarity_by_default(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("final_denoise_primary=CosmicClarity Denoise in-process script", message)
        self.assertIn("final_denoise_effective=CosmicClarity Denoise script", message)
        self.assertNotIn("fallback_component=", message)
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])

    def test_stage10_preserve_mode_keeps_stage9_pixels_and_can_export_normally(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_processing_mode = "preserve"
        original = processor.siril.get_image_pixeldata(preview=False)

        stage10_export(processor)

        np.testing.assert_array_equal(
            processor.siril.get_image_pixeldata(preview=False),
            original,
        )
        self.assertFalse(processor.script_calls)
        self.assertFalse(any(call[0] == "satu" for call in processor.cmd_calls))
        self.assertIn(("savetif", "result_processed", "-astro"), processor.cmd_calls)
        self.assertFalse(processor._final_output_review_only)
        metadata = processor.result_metadata[-1]
        self.assertEqual(metadata["details"]["processing_mode"], "preserve")
        self.assertEqual(
            metadata["components"]["denoise"]["reason_code"],
            "preserve_mode",
        )

    def test_stage10_scunet_only_never_attempts_cosmic(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_denoise_backend_policy = "scunet_only"
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.available_commands.add("siril_scunet_denoise")
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.50,
            "bg_std": 0.03,
            "background_mottling_score": 0.60,
        }

        stage10_export(processor)

        self.assertFalse(processor.script_calls)
        self.assertIn(("siril_scunet_denoise", "0.28"), processor.cmd_calls)
        plan = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertEqual(plan["backend_policy"], "scunet_only")

    def test_stage10_cosmic_only_never_attempts_scunet_or_aberration(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_denoise_backend_policy = "cosmic_only"
        processor.cfg.aberration_api_enabled = True
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_fail_steps.add("最终降噪")
        processor.cli_fail_steps.add("最终降噪")
        processor.available_commands.add("siril_scunet_denoise")
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.50,
            "bg_std": 0.03,
            "background_mottling_score": 0.60,
        }

        stage10_export(processor)

        self.assertFalse(
            any(call[0] == "siril_scunet_denoise" for call in processor.cmd_calls)
        )
        self.assertFalse(processor.aberration_calls)
        self.assertEqual(
            processor.stage_json_reports["stage10_denoise_plan.json"][
                "backend_policy"
            ],
            "cosmic_only",
        )

    def test_stage10_task_switches_skip_denoise_saturation_and_quality_repair(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_final_denoise_enabled = False
        processor.cfg.stage10_final_saturation_enabled = False
        processor.cfg.stage10_quality_repair_enabled = False
        processor._final_quality_report = lambda _stem: self._stage10_quality_noise_report(
            chroma=1.20,
            hard=True,
        )
        with patch.object(
            stage10_export_module,
            "_attempt_stage10_quality_repair",
        ) as repair:
            stage10_export(processor)

        repair.assert_not_called()
        self.assertFalse(processor.script_calls)
        self.assertFalse(any(call[0] == "satu" for call in processor.cmd_calls))
        repair_record = processor.stage_json_reports[
            "final_quality_report.json"
        ]["repair"]
        self.assertEqual(repair_record["status"], "disabled")

    def test_stage10_preserve_review_rolls_back_and_exports_review_only(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_failure_action = "preserve_review"
        processor.cfg.stage10_denoise_backend_policy = "cosmic_only"
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.50,
            "bg_std": 0.03,
            "background_mottling_score": 0.60,
        }

        stage10_export(processor)

        self.assertIn(("load", "stage9_remixed"), processor.cmd_calls)
        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertTrue(
            processor.result_metadata[-1]["details"][
                "preserve_review_triggered"
            ]
        )

    def test_stage10_stop_writes_diagnostics_and_generates_no_final_export(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_failure_action = "stop"
        processor.cfg.stage10_denoise_backend_policy = "cosmic_only"
        processor._background_quality_metrics = lambda _image: {
            "chroma_noise_score": 0.50,
            "bg_std": 0.03,
            "background_mottling_score": 0.60,
        }

        with self.assertRaisesRegex(RuntimeError, "用户严格停止"):
            stage10_export(processor)

        self.assertIn("stage10_denoise_plan.json", processor.stage_json_reports)
        self.assertIn("stage10_failure_policy.json", processor.stage_json_reports)
        self.assertEqual(processor.results[-1][1], "failed")
        self.assertFalse(
            any(call[0] in {"savetif", "savepng"} for call in processor.cmd_calls)
        )

    def test_stage10_applies_saturation_after_final_denoise(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        events: list[str] = []
        original_plugin_runner = processor._run_plugin_script_by_path
        original_command = processor.cmd_with_check

        def run_plugin(*args: Any, **kwargs: Any):
            if len(args) >= 2 and args[1] == "CosmicClarity Denoise":
                events.append("denoise")
            return original_plugin_runner(*args, **kwargs)

        def command(*args: Any, quiet: bool = False) -> bool:
            if args and args[0] == "satu":
                events.append("saturation")
            return original_command(*args, quiet=quiet)

        processor._run_plugin_script_by_path = run_plugin
        processor.cmd_with_check = command

        stage10_export(processor)

        self.assertIn("denoise", events)
        self.assertIn("saturation", events)
        self.assertLess(events.index("denoise"), events.index("saturation"))
        report = processor.stage_json_reports[
            "stage10_color_rebalance_report.json"
        ]
        self.assertEqual(report["status"], "reported")
        self.assertFalse(report["used_for_gate"])
        self.assertEqual(
            report["operation_order"][2],
            "budgeted_saturation_rebalance",
        )
        self.assertGreater(report["decision"]["applied_saturation"], 0.0)
        self.assertEqual(
            report["decision"]["applied_saturation"],
            report["decision"]["effective_saturation"],
        )
        self.assertEqual(
            [entry["operation"] for entry in report["cross_stage_ledger"][-2:]],
            [
                "final_denoise_color_delta",
                "post_denoise_budgeted_saturation",
            ],
        )

    def test_stage10_export_filename_fallback_is_structured(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage5_denoise_applied = True
        processor._stage8_final_quality = "ok"
        processor._stage8_fallback_used = False
        processor._result_output_basename = lambda: "primary_result"
        processor.main_output_fit_basename_template = "primary_final"
        command_calls: list[tuple[Any, ...]] = []

        def command(*args: Any, quiet: bool = False) -> bool:
            _ = quiet
            command_calls.append(args)
            if args[:2] == ("savetif", "primary_result"):
                raise RuntimeError("mock primary TIFF failure")
            return True

        processor.cmd_with_check = command

        stage10_export(processor)

        report = processor.stage_json_reports["stage10_export_report.json"]
        self.assertTrue(report["fallback_used"])
        self.assertEqual(report["fallback_formats"], ["tif"])
        self.assertEqual(report["outputs"]["tif"]["status"], "fallback")
        self.assertIn(("savetif", "result_processed", "-astro"), command_calls)
        metadata = processor.result_metadata[-1]
        self.assertTrue(metadata["fallback_used"])
        self.assertEqual(metadata["reason_code"], "final_export_fallback")
        self.assertTrue(metadata["components"]["export"]["fallback_used"])

    def test_stage10_in_process_path_does_not_depend_on_cli_connection(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.cli_failure_errors["最终降噪"] = (
            "CosmicClarity_Denoise.py: subprocess exited with code 1; "
            "output_tail=Error: Failed to connect to Siril"
        )

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertNotIn("CLI Siril 连接失败", message)
        self.assertIn("primary_status=success", message)

    def test_stage10_uses_native_cosmic_clarity_without_classic_executable(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/CosmicClarity_Native.py",
            }
        )
        processor.classic_cc_args = None

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        classic_calls = [
            call
            for call in processor.script_calls
            if call[1] == "CosmicClarity_Denoise.py"
        ]
        self.assertFalse(classic_calls)
        self.assertIn("CosmicClarity classic 路径未启用，已选择 Native Denoise", message)
        self.assertIn("final_denoise_primary=CosmicClarity Native Denoise cli-subprocess", message)
        self.assertIn("primary_status=success", message)
        self.assertNotIn("fallback_component=CosmicClarity Native Denoise", message)
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])

    def test_stage10_script_failure_without_scunet_falls_back_to_aberration(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_fail_steps.add("最终降噪")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Siril-SCUNet Denoise 回退不可用", message)
        self.assertIn("fallback_component=SASP Aberration API", message)
        self.assertNotIn("fallback_component=in-process CosmicClarity Denoise script (CosmicClarity_Denoise.py)", message)

    def test_stage10_skips_unparameterized_scunet_gui_script(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["最终降噪"] = "SASP Aberration API (CPU)"
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/SCUNet_Denoise.py",
            }
        )
        processor.script_fail_steps.add("最终降噪")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("fallback_component=SASP Aberration API", message)
        self.assertIn(
            "SCUNet_Denoise.py is interactive and has no explicit headless "
            "strength contract; skipped",
            message,
        )
        scunet_calls = [
            call for call in processor.script_calls if call[1] == "SCUNet_Denoise.py"
        ]
        self.assertFalse(scunet_calls)

    def test_stage10_degraded_when_final_denoise_skipped_even_exports_succeed(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = False

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertIn("最终降噪未执行（script/scunet unavailable, Aberration API disabled）", message)

    def test_final_quality_recognizes_stage8_conservative_skip(self):
        probe = SimpleNamespace(
            _stage8_final_quality="conservative_skipped",
            _stage8_fallback_used=True,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.35,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "emission_nebula_widefield",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertFalse(report["strict_gate"])
        self.assertTrue(report["metrics"]["stage8_conservative_skipped"])
        self.assertEqual(report["final_quality"], "ok")

    def test_final_quality_reports_uncalibrated_background_cast_review_gate(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage7_background_color_review_required=True,
            _stage7_background_color_review_gate={
                "status": "review_required",
                "requires_review": True,
                "value": 0.191,
                "limit": 0.12,
                "signal_exclusion_applied": True,
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.01,
                "background_mottling_score": 0.02,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "large_galaxy",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertEqual(report["final_quality"], "poor")
        self.assertIn(
            "uncalibrated_background_chroma_load 0.191>0.120",
            report["issues"],
        )
        self.assertTrue(
            report["metrics"][
                "uncalibrated_background_color_review_required"
            ]
        )

    def test_final_quality_warns_for_moderate_uncalibrated_background_cast(self):
        image = np.zeros((3, 64, 64), dtype=np.float32)
        image[0] = 0.041
        image[1] = 0.058
        image[2] = 0.040
        yy, xx = np.mgrid[:64, :64]
        signal = 0.30 * np.exp(
            -(((xx - 32) / 8.0) ** 2 + ((yy - 32) / 7.0) ** 2)
        )
        image += signal[None, :, :]
        stage8_pixels = pipeline_module.stage8_pixels
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage7_background_color_review_required=False,
            _stage7_background_color_review_gate={
                "applicable": True,
                "status": "ok",
                "requires_review": False,
                "value": 0.10,
                "limit": 0.12,
                "signal_exclusion_applied": True,
            },
            _read_image_by_stem=lambda _stem: image,
            _background_quality_metrics=(
                lambda candidate, masks=None: stage8_pixels.background_quality_metrics(
                    probe,
                    candidate,
                    masks,
                )
            ),
            _stage8_soften_mask=(
                lambda mask, passes=3: stage8_pixels.stage8_soften_mask(
                    probe,
                    mask,
                    passes=passes,
                )
            ),
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = stage8_pixels.final_quality_report(probe)

        final_metrics = report["metrics"][
            "final_signal_excluded_background_metrics"
        ]
        self.assertGreater(final_metrics["background_chroma_load"], 0.12)
        self.assertLessEqual(final_metrics["background_chroma_load"], 0.18)
        self.assertFalse(report["strict_gate"])
        self.assertEqual(report["final_quality"], "ok")
        self.assertFalse(
            report["metrics"][
                "uncalibrated_background_color_review_required"
            ]
        )
        self.assertTrue(report["advisories"])
        self.assertEqual(
            report["metrics"]["final_background_color_quality_gate"]["status"],
            "advisory",
        )

    def test_final_quality_final_cast_measurement_unavailable_fails_closed(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage7_background_color_review_required=False,
            _stage7_background_color_review_gate={
                "applicable": True,
                "status": "ok",
                "requires_review": False,
                "value": 0.10,
                "limit": 0.12,
                "signal_exclusion_applied": True,
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.01,
                "background_mottling_score": 0.02,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertIn(
            "uncalibrated_final_signal_excluded_background_chroma_load_"
            "unavailable",
            report["issues"],
        )
        self.assertNotIn(
            "uncalibrated_background_chroma_load 0.100>0.120",
            report["issues"],
        )

    def test_final_quality_uses_large_galaxy_patch_variance_limit(self):
        probe = SimpleNamespace(
            cfg=SimpleNamespace(
                stage10_large_galaxy_local_patch_variance_max=0.00032,
            ),
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0003015513502759859,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "large_galaxy_core_protect",
            _active_target_type=lambda: "large_galaxy",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertFalse(report["strict_gate"])
        self.assertEqual(report["final_quality"], "ok")
        self.assertEqual(
            report["metrics"]["local_patch_variance_max"],
            0.00032,
        )

    def test_final_quality_keeps_generic_patch_variance_as_soft_advisory(self):
        probe = SimpleNamespace(
            cfg=SimpleNamespace(
                stage10_large_galaxy_local_patch_variance_max=0.00032,
            ),
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.000221,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertEqual(report["schema"], "starun.final-quality.v2")
        self.assertEqual(report["severity"], "soft_warning")
        self.assertEqual(report["final_quality"], "ok")
        self.assertFalse(report["needs_conservative_rerun"])
        self.assertEqual(
            report["metrics"]["local_patch_variance_max"],
            0.00022,
        )
        self.assertFalse(report["issues"])
        self.assertTrue(
            any(
                "local_patch_variance 0.000221>0.000220" in warning
                for warning in report["warnings"]
            )
        )

    def test_final_quality_single_chroma_anomaly_is_soft_warning(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image, _masks=None: {
                "chroma_noise_score": 0.43,
                "background_mottling_score": 0.10,
                "local_patch_variance": 0.00001,
                "local_texture_residual_p90": 0.001,
                "local_texture_residual_outlier_score": 0.2,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.10,
                "bg_dirty_score": 0.0,
                "bg_std": 0.003,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertEqual(report["severity"], "soft_warning")
        self.assertEqual(report["final_quality"], "ok")
        self.assertFalse(report["hard_issues"])
        self.assertTrue(
            any("background_chroma_noise_score" in item for item in report["warnings"])
        )

    def test_final_quality_extreme_chroma_is_hard_reject(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image, _masks=None: {
                "chroma_noise_score": 0.91,
                "background_mottling_score": 0.10,
                "local_patch_variance": 0.00001,
                "local_texture_residual_p90": 0.001,
                "local_texture_residual_outlier_score": 0.2,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.10,
                "bg_dirty_score": 0.0,
                "bg_std": 0.003,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "large_galaxy",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertEqual(report["severity"], "hard_reject")
        self.assertTrue(report["needs_conservative_rerun"])
        self.assertTrue(
            any(
                item.startswith("background_chroma_noise_extreme")
                for item in report["hard_issues"]
            )
        )

    def test_final_quality_non_finite_image_is_hard_reject(self):
        image = np.zeros((3, 16, 16), dtype=np.float32)
        image[0, 3, 4] = np.nan
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: image,
            _background_quality_metrics=lambda _image, _masks=None: {
                "chroma_noise_score": 0.10,
                "background_mottling_score": 0.10,
                "local_patch_variance": 0.00001,
                "local_texture_residual_p90": 0.001,
                "local_texture_residual_outlier_score": 0.2,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.10,
                "bg_dirty_score": 0.0,
                "bg_std": 0.003,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertEqual(report["severity"], "hard_reject")
        self.assertIn("final_image_non_finite_pixels", report["hard_issues"])

    def test_final_quality_combined_noise_growth_is_hard_reject(self):
        metric_sets = {
            "stage9_remixed": {
                "chroma_noise_score": 0.30,
                "background_mottling_score": 0.40,
                "local_patch_variance": 0.00001,
                "local_texture_residual_p90": 0.001,
                "local_texture_residual_outlier_score": 0.2,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.30,
                "bg_dirty_score": 0.0,
                "bg_std": 0.003,
            },
            "stage10_final": {
                "chroma_noise_score": 0.60,
                "background_mottling_score": 0.70,
                "local_patch_variance": 0.00001,
                "local_texture_residual_p90": 0.001,
                "local_texture_residual_outlier_score": 0.2,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.30,
                "bg_dirty_score": 0.0,
                "bg_std": 0.003,
            },
        }
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage10_quality_baseline_stem="stage9_remixed",
            _read_image_by_stem=lambda stem: stem,
            _background_quality_metrics=(
                lambda image, _masks=None: dict(metric_sets[str(image)])
            ),
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(
            probe,
            "stage10_final",
        )

        self.assertEqual(report["severity"], "hard_reject")
        self.assertTrue(
            any(
                item.startswith("background_noise_combined_growth")
                for item in report["hard_issues"]
            )
        )
        self.assertGreater(
            report["metrics"]["noise_growth"]["chroma"]["ratio"],
            1.5,
        )

    def test_final_quality_extreme_residual_texture_is_hard_reject(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image, _masks=None: {
                "chroma_noise_score": 0.10,
                "background_mottling_score": 0.10,
                "local_patch_variance": 0.00001,
                "local_texture_residual_p90": 0.010,
                "local_texture_residual_outlier_score": 4.1,
                "local_texture_affected_patch_ratio": 0.36,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.10,
                "bg_dirty_score": 0.0,
                "bg_std": 0.003,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertEqual(report["severity"], "hard_reject")
        self.assertTrue(
            any(
                item.startswith("local_texture_residual_extreme")
                for item in report["hard_issues"]
            )
        )

    def test_final_quality_representative_batch_regression(self):
        def evaluate(
            metrics: dict[str, float],
            *,
            target_type: str,
            stars_applied: bool,
        ) -> dict[str, Any]:
            probe = SimpleNamespace(
                _stage8_final_quality="ok",
                _stage8_fallback_used=False,
                _stage9_bypassed_bad_starless=False,
                _stage9_stars_required=True,
                _stage9_stars_applied=stars_applied,
                _stage9_stars_application_mode=(
                    "screen" if stars_applied else "no_starmask"
                ),
                _read_image_by_stem=lambda _stem: object(),
                _background_quality_metrics=(
                    lambda _image, _masks=None: dict(metrics)
                ),
                _stage7_halo_residue_score=lambda: 0.0,
                _active_policy_name=lambda: "batch_regression",
                _active_target_type=lambda: target_type,
            )
            return pipeline_module.stage8_pixels.final_quality_report(probe)

        ngc2237 = evaluate(
            {
                "chroma_noise_score": 0.134063,
                "background_mottling_score": 0.344227,
                "local_patch_variance": 0.000916735,
                "local_texture_residual_p90": 0.0,
                "local_texture_residual_outlier_score": 0.0,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.464253,
                "bg_dirty_score": 0.01745,
                "bg_std": 0.002083,
            },
            target_type="emission_nebula_widefield",
            stars_applied=True,
        )
        m33 = evaluate(
            {
                "chroma_noise_score": 1.386789,
                "background_mottling_score": 0.126777,
                "local_patch_variance": 0.000815428,
                "local_texture_residual_p90": 0.0,
                "local_texture_residual_outlier_score": 0.0,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.369035,
                "bg_dirty_score": 0.0,
                "bg_std": 0.018685,
            },
            target_type="large_galaxy",
            stars_applied=False,
        )
        dwarf = evaluate(
            {
                "chroma_noise_score": 0.387028,
                "background_mottling_score": 0.126752,
                "local_patch_variance": 0.001351624,
                "local_texture_residual_p90": 0.0,
                "local_texture_residual_outlier_score": 0.0,
                "local_texture_affected_patch_ratio": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.285231,
                "bg_dirty_score": 0.0,
                "bg_std": 0.027041,
            },
            target_type="dark_nebula_low_contrast",
            stars_applied=False,
        )

        self.assertEqual(ngc2237["severity"], "soft_warning")
        self.assertEqual(ngc2237["final_quality"], "ok")
        self.assertEqual(m33["severity"], "hard_reject")
        self.assertIn("stage9_required_stars_not_applied", m33["primary_issues"])
        self.assertTrue(
            any(
                item.startswith("background_chroma_noise_extreme")
                for item in m33["hard_issues"]
            )
        )
        self.assertEqual(dwarf["severity"], "hard_reject")
        self.assertEqual(
            dwarf["primary_issues"],
            ["stage9_required_stars_not_applied"],
        )
        self.assertFalse(
            any(
                item.startswith("background_chroma_noise_extreme")
                for item in dwarf["hard_issues"]
            )
        )
        self.assertTrue(
            any("background_chroma_noise_score" in item for item in dwarf["warnings"])
        )

    def test_final_quality_keeps_strict_patch_limit_for_large_galaxy(self):
        probe = SimpleNamespace(
            cfg=SimpleNamespace(
                stage10_large_galaxy_local_patch_variance_max=0.00032,
            ),
            _stage8_final_quality="ok",
            _stage8_fallback_used=True,
            _stage9_bypassed_bad_starless=False,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.00017,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "large_galaxy_core_protect",
            _active_target_type=lambda: "large_galaxy",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertEqual(report["final_quality"], "poor")
        self.assertEqual(
            report["metrics"]["local_patch_variance_max"],
            0.00016,
        )

    def test_final_quality_rejects_hidden_compact_stage7_halo(self):
        probe = SimpleNamespace(
            _stage8_final_quality="conservative_skipped",
            _stage8_fallback_used=True,
            _stage9_bypassed_bad_starless=False,
            _stage7_selected_quality={
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.427,
                    "global_halo_residue_score": 0.427,
                    "compact_halo_residue_score": 0.857,
                },
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.427,
            _stage7_effective_halo_threshold=lambda: 0.60,
            _active_policy_name=lambda: "bright_emission_reflection_nebula",
            _active_target_type=lambda: "bright_emission_reflection_nebula",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertEqual(report["final_quality"], "poor")
        self.assertIn(
            "stage7_compact_halo_residue_score 0.857>0.600",
            report["issues"],
        )
        self.assertEqual(
            report["metrics"]["stage7_compact_halo_residue_score"],
            0.857,
        )

    def test_final_quality_accepts_near_limit_compact_halo_after_safe_star_remix(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_stars_application_mode="screen",
            _stage9_starmask_stretch_failed=False,
            _stage9_selected_remix_quality={
                "metrics": {
                    "chromatic_star_addition_ratio": 0.00001,
                    "local_color_risk_score": 0.66,
                },
                "limits": {"chromatic_star_addition_ratio": 0.003},
            },
            _stage7_selected_quality={
                "status": "ok",
                "derived": {
                    "halo_residue_score": 0.343,
                    "global_halo_residue_score": 0.343,
                    "compact_halo_residue_score": 0.637,
                    "compact_residual_star_score": 0.001,
                    "compact_residual_coverage": 0.0001,
                },
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.01,
                "background_mottling_score": 0.03,
                "local_patch_variance": 0.000001,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.06,
                "bg_dirty_score": 0.04,
                "bg_std": 0.004,
            },
            _stage7_halo_residue_score=lambda: 0.343,
            _stage7_effective_halo_threshold=lambda: 0.60,
            _active_policy_name=lambda: "bright_nebula_hdr_conservative",
            _active_target_type=lambda: "bright_emission_reflection_nebula",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertFalse(report["strict_gate"])
        self.assertEqual(report["final_quality"], "ok")
        self.assertTrue(
            report["metrics"]["stage7_compact_halo_raw_limit_exceeded"]
        )
        self.assertTrue(
            report["metrics"]["stage7_compact_halo_target_aware_exempted"]
        )

        probe._stage9_remix_formally_accepted = False
        probe._stage9_review_candidate_selected = True
        review_report = pipeline_module.stage8_pixels.final_quality_report(
            probe
        )
        self.assertFalse(
            review_report["metrics"][
                "stage7_compact_halo_target_aware_exempted"
            ]
        )
        self.assertEqual(review_report["final_quality"], "poor")
        self.assertTrue(
            review_report["metrics"]["stage9_review_candidate_selected"]
        )

    def test_final_quality_rejects_selected_stage9_chromatic_artifacts(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_stars_application_mode="screen",
            _stage9_selected_remix_quality={
                "metrics": {"chromatic_star_addition_ratio": 0.010},
                "limits": {"chromatic_star_addition_ratio": 0.003},
            },
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "emission_nebula_widefield",
            _active_target_type=lambda: "emission_nebula_widefield",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertEqual(report["final_quality"], "poor")
        self.assertIn(
            "stage9_chromatic_star_addition_ratio 0.010000>0.003000",
            report["issues"],
        )
        self.assertEqual(
            report["metrics"]["stage9_chromatic_star_addition_ratio"],
            0.010,
        )

        probe._stage9_selected_remix_quality = {
            "metrics": {"chromatic_star_addition_ratio": 0.004},
            "limits": {"chromatic_star_addition_ratio": 0.003},
        }
        advisory_report = pipeline_module.stage8_pixels.final_quality_report(probe)
        self.assertEqual(advisory_report["final_quality"], "ok")
        self.assertEqual(
            advisory_report["metrics"][
                "stage9_chromatic_star_addition_quality_gate"
            ]["status"],
            "advisory",
        )
        self.assertTrue(advisory_report["advisories"])

    def test_final_quality_requires_review_after_unsafe_stage9_bypass(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=True,
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertEqual(report["final_quality"], "poor")
        self.assertTrue(report["needs_conservative_rerun"])

    def test_final_quality_requires_review_when_required_stars_not_applied(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=False,
            _stage9_stars_application_mode="no_starmask",
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertIn("stage9_required_stars_not_applied", report["issues"])
        self.assertTrue(report["metrics"]["stage9_stars_required"])
        self.assertFalse(report["metrics"]["stage9_stars_applied"])
        self.assertTrue(report["needs_conservative_rerun"])

    def test_final_quality_requires_review_after_starmask_stretch_failure(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_starmask_stretch_failed=True,
            _stage9_stars_application_mode="screen",
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertIn("stage9_starmask_stretch_failed", report["issues"])
        self.assertTrue(report["metrics"]["stage9_starmask_stretch_failed"])
        self.assertTrue(report["needs_conservative_rerun"])

    def test_final_quality_reports_starmask_preparation_separately(self):
        probe = SimpleNamespace(
            _stage8_final_quality="ok",
            _stage8_fallback_used=False,
            _stage9_bypassed_bad_starless=False,
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_starmask_preparation_failed=True,
            _stage9_starmask_preparation_failure_reason="reference unavailable",
            _stage9_starmask_stretch_failed=False,
            _stage9_stars_application_mode="starmask_preparation_failed",
            _read_image_by_stem=lambda _stem: object(),
            _background_quality_metrics=lambda _image: {
                "chroma_noise_score": 0.0,
                "background_mottling_score": 0.0,
                "local_patch_variance": 0.0,
                "core_clip_score": 0.0,
                "starless_artifact_score": 0.0,
                "bg_dirty_score": 0.0,
                "bg_std": 0.0,
            },
            _stage7_halo_residue_score=lambda: 0.0,
            _active_policy_name=lambda: "generic_low_snr_safe",
            _active_target_type=lambda: "generic_low_snr_safe",
        )

        report = pipeline_module.stage8_pixels.final_quality_report(probe)

        self.assertTrue(report["strict_gate"])
        self.assertIn("stage9_starmask_preparation_failed", report["issues"])
        self.assertTrue(report["metrics"]["stage9_starmask_preparation_failed"])
        self.assertFalse(report["metrics"]["stage9_starmask_stretch_failed"])
        self.assertEqual(
            report["metrics"]["stage9_starmask_preparation_failure_reason"],
            "reference unavailable",
        )

    def test_stage10_degraded_when_script_and_aberration_unavailable(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.aberration_api_enabled = True
        processor.aberration_errors["最终降噪"] = "import failed: No module named 'PyQt6'"
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_fail_steps.add("最终降噪")

        stage10_export(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "degraded")
        self.assertTrue(message.strip())
        self.assertIn("最终降噪脚本与 Aberration API 均不可用", message)
        self.assertIn("Aberration API 不可用", message)

    def test_stage10_skips_scunet_after_primary_timeout(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.classic_cc_args = None
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Denoise.py",
                "processing/CosmicClarity_Native.py",
                "processing/SCUNet_Denoise.py",
            }
        )
        processor.cli_fail_steps.add("最终降噪")
        processor.cli_failure_errors["最终降噪"] = (
            "CosmicClarity_Native.py: subprocess timeout after 180s"
        )

        stage10_export(processor)

        scunet_calls = [
            call for call in processor.script_calls if call[1] == "SCUNet_Denoise.py"
        ]
        self.assertFalse(scunet_calls)
        self.assertIn("skipped after primary denoiser timeout", processor.results[-1][3])

    def test_stage10_uses_linear_resume_output_suffixes(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage1_input_mode = "linear_resume"

        stage10_export(processor)

        linear_base = "result_processed_linear"
        self.assertIn(("savetif", linear_base, "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", linear_base), processor.cmd_calls)
        self.assertIn(("save", "result_final_linear"), processor.cmd_calls)
        self.assertEqual(processor.main_output_basename_template, linear_base)

    def test_stage10_nonpreferred_source_recovery_forces_review_only_output(self):
        processor = self._new_processor()
        processor._stage9_final_source = "preferred_stage9_output"
        self._stage10_final_input(processor)
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")

        stage10_export(processor)

        self.assertIn(("load", "stage9_remixed"), processor.cmd_calls)
        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertFalse(
            any(step == "最终降噪" for step, _name, _args in processor.script_calls)
        )
        self.assertTrue(processor._final_output_review_only)
        metadata = processor.result_metadata[-1]
        self.assertEqual(
            metadata["reason_code"],
            "final_source_recovery_review_required",
        )
        self.assertTrue(metadata["details"]["final_source_review_required"])

    def test_stage10_unavailable_source_lineage_forces_review_only_output(self):
        processor = self._new_processor()

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        metadata = processor.result_metadata[-1]
        self.assertEqual(
            metadata["reason_code"],
            "final_source_unavailable_review_required",
        )
        self.assertTrue(metadata["details"]["final_source_review_required"])

    def test_stage10_quality_repair_accepts_single_improving_candidate(self):
        original = np.full((3, 16, 16), 0.20, dtype=np.float32)
        candidate = original.copy()
        candidate[:, :8, :8] = 0.19
        state = {"pixels": original.copy()}
        saved: dict[str, np.ndarray] = {}
        initial = self._stage10_quality_noise_report(chroma=1.20, hard=True)
        improved = self._stage10_quality_noise_report(chroma=0.50, hard=False)

        def save(stem: str) -> bool:
            saved[stem] = state["pixels"].copy()
            return True

        def command(*args: Any, **_kwargs: Any) -> bool:
            if args and args[0] == "load":
                state["pixels"] = saved[str(args[1])].copy()
            return True

        probe = SimpleNamespace(
            _stage10_quality_frozen_background_masks={"background_mask": np.ones((16, 16))},
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: state["pixels"].copy()
            ),
            _set_current_image_pixeldata=(
                lambda image, **_kwargs: state.__setitem__(
                    "pixels",
                    np.array(image, copy=True),
                )
            ),
            _save_stage_output=save,
            cmd_with_check=command,
            _final_quality_report=lambda _stem: dict(improved),
        )

        with patch.object(
            stage10_export_module,
            "_stage10_quality_repair_candidate",
            return_value=(candidate, {"mode": "mock"}),
        ) as candidate_builder, patch.object(
            stage10_export_module,
            "_stage10_quality_repair_structure_metrics",
            return_value={
                "signal_luminance_correlation": 0.999,
                "signal_flux_ratio": 1.0,
                "core_clip_growth": 0.0,
            },
        ):
            report = stage10_export_module._attempt_stage10_quality_repair(
                probe,
                initial,
                source_trusted=True,
            )

        candidate_builder.assert_called_once()
        self.assertEqual(report["repair"]["status"], "accepted")
        np.testing.assert_array_equal(saved["stage10_final"], candidate)
        self.assertIn("stage10_pre_quality_repair", saved)
        self.assertIn("stage10_quality_repair_candidate", saved)

    def test_stage10_quality_repair_rolls_back_rejected_candidate(self):
        original = np.full((3, 16, 16), 0.20, dtype=np.float32)
        candidate = np.full_like(original, 0.10)
        state = {"pixels": original.copy()}
        saved: dict[str, np.ndarray] = {}
        initial = self._stage10_quality_noise_report(chroma=1.20, hard=True)
        not_improved = self._stage10_quality_noise_report(chroma=1.10, hard=True)

        def save(stem: str) -> bool:
            saved[stem] = state["pixels"].copy()
            return True

        def command(*args: Any, **_kwargs: Any) -> bool:
            if args and args[0] == "load":
                state["pixels"] = saved[str(args[1])].copy()
            return True

        probe = SimpleNamespace(
            _stage10_quality_frozen_background_masks={"background_mask": np.ones((16, 16))},
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: state["pixels"].copy()
            ),
            _set_current_image_pixeldata=(
                lambda image, **_kwargs: state.__setitem__(
                    "pixels",
                    np.array(image, copy=True),
                )
            ),
            _save_stage_output=save,
            cmd_with_check=command,
            _final_quality_report=lambda _stem: dict(not_improved),
        )

        with patch.object(
            stage10_export_module,
            "_stage10_quality_repair_candidate",
            return_value=(candidate, {"mode": "mock"}),
        ), patch.object(
            stage10_export_module,
            "_stage10_quality_repair_structure_metrics",
            return_value={
                "signal_luminance_correlation": 0.999,
                "signal_flux_ratio": 1.0,
                "core_clip_growth": 0.0,
            },
        ):
            report = stage10_export_module._attempt_stage10_quality_repair(
                probe,
                initial,
                source_trusted=True,
            )

        self.assertEqual(report["repair"]["status"], "rolled_back")
        np.testing.assert_array_equal(state["pixels"], original)
        np.testing.assert_array_equal(saved["stage10_final"], original)

    def test_stage10_quality_repair_rolls_back_when_final_save_fails(self):
        original = np.full((3, 16, 16), 0.20, dtype=np.float32)
        candidate = np.full_like(original, 0.19)
        state = {"pixels": original.copy()}
        saved: dict[str, np.ndarray] = {}
        final_save_calls = 0
        initial = self._stage10_quality_noise_report(chroma=1.20, hard=True)
        improved = self._stage10_quality_noise_report(chroma=0.50, hard=False)

        def save(stem: str) -> bool:
            nonlocal final_save_calls
            if stem == "stage10_final":
                final_save_calls += 1
                if final_save_calls == 1:
                    return False
            saved[stem] = state["pixels"].copy()
            return True

        def command(*args: Any, **_kwargs: Any) -> bool:
            if args and args[0] == "load":
                state["pixels"] = saved[str(args[1])].copy()
            return True

        probe = SimpleNamespace(
            _stage10_quality_frozen_background_masks={"background_mask": np.ones((16, 16))},
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: state["pixels"].copy()
            ),
            _set_current_image_pixeldata=(
                lambda image, **_kwargs: state.__setitem__(
                    "pixels",
                    np.array(image, copy=True),
                )
            ),
            _save_stage_output=save,
            cmd_with_check=command,
            _final_quality_report=lambda _stem: dict(improved),
        )

        with patch.object(
            stage10_export_module,
            "_stage10_quality_repair_candidate",
            return_value=(candidate, {"mode": "mock"}),
        ), patch.object(
            stage10_export_module,
            "_stage10_quality_repair_structure_metrics",
            return_value={
                "signal_luminance_correlation": 0.999,
                "signal_flux_ratio": 1.0,
                "core_clip_growth": 0.0,
            },
        ):
            report = stage10_export_module._attempt_stage10_quality_repair(
                probe,
                initial,
                source_trusted=True,
            )

        self.assertEqual(report["repair"]["status"], "rolled_back")
        self.assertEqual(final_save_calls, 2)
        np.testing.assert_array_equal(state["pixels"], original)
        np.testing.assert_array_equal(saved["stage10_final"], original)

    def test_stage10_quality_repair_skips_untrusted_review_source(self):
        initial = self._stage10_quality_noise_report(chroma=1.20, hard=True)
        probe = SimpleNamespace()

        with patch.object(
            stage10_export_module,
            "_stage10_quality_repair_candidate",
        ) as candidate_builder:
            report = stage10_export_module._attempt_stage10_quality_repair(
                probe,
                initial,
                source_trusted=False,
            )

        candidate_builder.assert_not_called()
        self.assertFalse(report["repair"]["attempted"])
        self.assertEqual(report["repair"]["status"], "skipped")

    def test_stage10_quality_chroma_repair_preserves_signal_and_luminance(self):
        yy, xx = np.mgrid[:32, :32]
        checker = ((xx + yy) % 2).astype(np.float32)
        original = np.empty((3, 32, 32), dtype=np.float32)
        original[0] = 0.18 + 0.04 * checker
        original[1] = 0.18
        original[2] = 0.18 - 0.03 * checker
        signal = np.zeros((32, 32), dtype=np.float32)
        signal[12:20, 12:20] = 1.0
        background = 1.0 - signal
        masks = {
            "background_mask": background,
            "core_mask": signal,
            "nebula_mask": signal,
            "faint_nebula_mask": np.zeros_like(signal),
        }
        probe = SimpleNamespace(
            _stage10_quality_frozen_background_masks=masks,
        )
        report = self._stage10_quality_noise_report(chroma=1.20, hard=True)
        report["metrics"].update(
            {
                "local_patch_variance": 0.0,
                "local_patch_variance_max": 0.00022,
                "bg_std": 0.004,
            }
        )

        with patch.object(
            stage10_export_module,
            "_build_stage10_star_protection_mask",
            return_value=(
                np.zeros((32, 32), dtype=np.float32),
                {"status": "ready", "reason": "mock"},
            ),
        ):
            candidate, metadata = (
                stage10_export_module._stage10_quality_repair_candidate(
                    probe,
                    original,
                    report,
                )
            )

        original_luma = (
            0.2126 * original[0] + 0.7152 * original[1] + 0.0722 * original[2]
        )
        candidate_luma = (
            0.2126 * candidate[0]
            + 0.7152 * candidate[1]
            + 0.0722 * candidate[2]
        )
        np.testing.assert_allclose(candidate[:, 12:20, 12:20], original[:, 12:20, 12:20])
        np.testing.assert_allclose(candidate_luma, original_luma, atol=1e-6)
        self.assertTrue(metadata["chroma_repair"])
        self.assertFalse(metadata["texture_repair"])

    def test_stage10_checkpoint_save_failure_fails_closed_to_review_output(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        quality_calls: list[str] = []
        processor._final_quality_report = lambda stem: (
            quality_calls.append(stem)
            or {
                "final_quality": "ok",
                "status": "ok",
                "needs_conservative_rerun": False,
                "issues": [],
            }
        )
        processor._save_stage_output = lambda stem: stem != "stage10_final"

        stage10_export(processor)

        self.assertEqual(quality_calls, [])
        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        metadata = processor.result_metadata[-1]
        self.assertEqual(
            metadata["reason_code"],
            "final_quality_checkpoint_unavailable",
        )
        self.assertEqual(
            metadata["details"]["final_quality_gate_status"],
            "checkpoint_unavailable",
        )

    def test_stage10_soft_final_quality_warning_keeps_normal_output_names(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        soft_report = self._stage10_quality_noise_report(
            chroma=0.50,
            hard=False,
        )
        soft_report["warnings"] = [
            "background_chroma_noise_score 0.500>0.420"
        ]
        soft_report["advisories"] = list(soft_report["warnings"])
        processor._final_quality_report = lambda _stem: dict(soft_report)

        stage10_export(processor)

        self.assertIn(("savetif", "result_processed", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_final"), processor.cmd_calls)
        self.assertFalse(processor._final_output_review_only)
        self.assertEqual(
            processor.result_metadata[-1]["details"]["final_quality_severity"],
            "soft_warning",
        )
        self.assertEqual(
            processor.result_metadata[-1]["details"]["final_quality_warning_count"],
            1,
        )

    def test_stage10_stage7_appearance_forced_delivery_is_review_only(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_quality_repair_enabled = False
        processor._stage7_stretch_forced_delivery = True
        processor._stage7_forced_delivery_reasons = [
            "background_chroma_noise_score"
        ]
        processor._stage7_background_color_review_required = True
        processor._final_quality_report = lambda _stem: (
            self._stage10_quality_noise_report(chroma=1.20, hard=True)
        )

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        metadata = processor.result_metadata[-1]
        self.assertEqual(
            metadata["details"]["final_quality_gate_status"],
            "review_required",
        )
        self.assertTrue(metadata["details"]["stage7_forced_delivery"])

    def test_stage10_stage7_forced_delivery_never_overrides_core_damage(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.stage10_quality_repair_enabled = False
        processor._stage7_stretch_forced_delivery = True
        hard_report = self._stage10_quality_noise_report(chroma=1.20, hard=True)
        hard_report["issues"] = ["core_clip_score 0.0200>0.0120"]
        hard_report["hard_issues"] = list(hard_report["issues"])
        processor._final_quality_report = lambda _stem: dict(hard_report)

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertEqual(
            processor.result_metadata[-1]["details"][
                "final_quality_gate_status"
            ],
            "review_required",
        )

    def test_stage10_quality_gate_exception_fails_closed_to_review_output(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)

        def unavailable_quality_gate(_stem: str) -> dict[str, Any]:
            raise RuntimeError("mock final quality read failure")

        processor._final_quality_report = unavailable_quality_gate

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        metadata = processor.result_metadata[-1]
        self.assertEqual(metadata["reason_code"], "final_quality_gate_unavailable")
        self.assertEqual(
            metadata["details"]["final_quality_gate_status"],
            "unavailable",
        )
        self.assertIn(
            "mock final quality read failure",
            metadata["details"]["final_quality_gate_error"],
        )

    def test_stage10_inconsistent_quality_report_fails_closed(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._final_quality_report = lambda _stem: {
            "final_quality": "poor",
            "status": "ok",
            "needs_conservative_rerun": False,
            "issues": [],
        }

        stage10_export(processor)

        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "final_quality_requires_review",
        )

    def test_stage10_withholds_normal_names_when_quality_requires_rerun(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._final_quality_report = lambda _stem: {
            "final_quality": "poor",
            "status": "needs_conservative_rerun",
            "needs_conservative_rerun": True,
            "issues": ["background_chroma_noise_score 0.431>0.340"],
        }

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", "result_review"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertFalse(
            any(
                "result_processed" in str(item) or "result_final" in str(item)
                for call in processor.cmd_calls
                for item in call
            )
        )
        self.assertTrue(processor._final_output_review_only)
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIn("review_only_output=true", processor.results[-1][3])

    def test_stage10_stage9_bypass_uses_linear_review_only_names(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage1_input_mode = "linear_resume"
        processor._stage9_bypassed_bad_starless = True
        processor._require_review(9, "with_stars_review_fallback")
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")

        stage10_export(processor)

        self.assertIn(
            ("savetif", "result_review_linear", "-astro"),
            processor.cmd_calls,
        )
        self.assertIn(("savepng", "result_review_linear"), processor.cmd_calls)
        self.assertIn(
            ("save", "result_review_linear_final"),
            processor.cmd_calls,
        )
        self.assertTrue(processor._final_output_review_only)
        self.assertFalse(
            any(step == "最终降噪" for step, _name, _args in processor.script_calls)
        )
        denoise_plan = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertTrue(denoise_plan["skipped_by_review_only"])
        self.assertFalse(denoise_plan["skipped_by_duplicate_guard"])

    def test_stage10_missing_required_star_remix_forces_review_only_output(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage9_bypassed_bad_starless = False
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = False
        processor._stage9_output_contains_stars = True
        processor._stage9_stars_application_mode = "no_starmask"

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertNotIn(
            ("savetif", "result_processed", "-astro"),
            processor.cmd_calls,
        )
        self.assertNotIn(("save", "result_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertIn("stage9_stars_applied=false", processor.results[-1][3])

    def test_stage10_partial_stage9_psf_evidence_forces_review_only_output(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_output_contains_stars = True
        processor._stage9_psf_review_required = True
        processor._require_review(
            9,
            "stage9_psf_subgroup_evidence_insufficient",
        )

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertIn(
            "stage9_psf_review_required=true",
            processor.results[-1][3],
        )

    def test_stage10_bounded_stage9_review_candidate_skips_denoise_and_color(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        (processor.process_dir / "stage9_review_with_stars.fit").write_bytes(
            b"mock"
        )
        processor._stage9_final_source = "stage9_review_with_stars"
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_output_contains_stars = True
        processor._stage9_psf_review_required = True
        processor._stage9_review_candidate_selected = True
        processor._stage9_remix_formally_accepted = False
        processor._require_review(9, "best_failed_candidate_review")
        processor.cfg.final_saturation = 0.15

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertNotIn(
            ("savetif", "result_processed", "-astro"),
            processor.cmd_calls,
        )
        self.assertNotIn(("save", "result_final"), processor.cmd_calls)
        self.assertFalse(
            any(call and call[0] == "satu" for call in processor.cmd_calls)
        )
        self.assertFalse(
            any(step == "最终降噪" for step, _name, _args in processor.script_calls)
        )
        self.assertTrue(processor._final_output_review_only)
        self.assertIn(
            "stage9_review_candidate_selected=true",
            processor.results[-1][3],
        )

    def test_stage10_stage8_starmask_fallback_is_review_only(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        (processor.process_dir / "stage9_review_with_stars.fit").write_bytes(
            b"mock"
        )
        processor._stage9_final_source = "stage9_review_with_stars"
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_output_contains_stars = True
        processor._stage9_psf_review_required = False
        processor._stage9_review_candidate_selected = False
        processor._stage9_remix_formally_accepted = False
        processor._stage9_stars_application_mode = (
            "screen_minimal_review_fallback"
        )
        processor._require_review(9, "stage8_starmask_review_fallback")
        processor.cfg.final_saturation = 0.15

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertNotIn(
            ("savetif", "result_processed", "-astro"),
            processor.cmd_calls,
        )
        self.assertNotIn(("save", "result_final"), processor.cmd_calls)
        self.assertFalse(
            any(call and call[0] == "satu" for call in processor.cmd_calls)
        )
        self.assertFalse(
            any(
                step == "最终降噪"
                for step, _name, _args in processor.script_calls
            )
        )
        self.assertTrue(processor._final_output_review_only)
        self.assertIn(
            "stage8_starmask_review_fallback",
            processor.results[-1][3],
        )
        self.assertNotIn(
            "stage9_stars_applied=false",
            processor.results[-1][3],
        )
        denoise_plan = processor.stage_json_reports["stage10_denoise_plan.json"]
        self.assertTrue(denoise_plan["skipped_by_review_only"])

    def test_stage10_withholds_all_outputs_when_required_stars_are_absent(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = False
        processor._stage9_output_contains_stars = False
        processor._stage9_output_withheld = True

        stage10_export(processor)

        self.assertFalse(
            any(
                command[0] in {"savetif", "savepng"}
                or (
                    command[0] == "save"
                    and any("result_" in str(value) for value in command[1:])
                )
                for command in processor.cmd_calls
            )
        )
        self.assertEqual(processor.results[-1][1], "failed")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "required_stars_output_withheld",
        )

    def test_stage10_starmask_borderline_forces_review_only_output(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage6_starmask_borderline_review_required = True
        processor._require_review(6, "starmask_cleanup_borderline")

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "starmask_diffuse_residual_borderline",
        )
        self.assertIn(
            "stage6_starmask_diffuse_residual_borderline=true",
            processor.results[-1][3],
        )

    def test_stage10_retained_failed_syqon_pair_forces_review_only_output(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage6_quality_hard_failed_retained = True

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "stage6_quality_hard_failed_retained",
        )
        self.assertTrue(
            processor.result_metadata[-1]["details"][
                "stage6_quality_hard_failed_retained"
            ]
        )
        self.assertIn(
            "stage6_quality_hard_failed_retained=true",
            processor.results[-1][3],
        )

    def test_stage10_color_review_forces_review_only_names(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage4_color_review_required = True
        processor._require_review(4, "color_calibration_review_required")
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", "result_review"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertIn("stage4_color_review_required=true", processor.results[-1][3])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "stage4_color_review_required",
        )

    def test_stage10_stage2_and_stage3_registry_reviews_force_review_only_names(self):
        cases = (
            (
                2,
                "stage2_view_review_required",
            ),
            (
                3,
                "stage3_background_review_required",
            ),
        )
        for stage, reason_code in cases:
            with self.subTest(stage=stage):
                processor = self._new_processor()
                self._stage10_final_input(processor)
                processor._require_review(stage, reason_code)
                processor._stage9_stars_required = True
                processor._stage9_stars_applied = True

                stage10_export(processor)

                self.assertIn(
                    ("savetif", "result_review", "-astro"),
                    processor.cmd_calls,
                )
                self.assertNotIn(
                    ("savetif", "result_processed", "-astro"),
                    processor.cmd_calls,
                )
                self.assertTrue(processor._final_output_review_only)
                self.assertEqual(
                    processor.result_metadata[-1]["reason_code"],
                    reason_code,
                )

    def test_stage10_ignores_unregistered_legacy_background_flag(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._background_review_required = True
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True

        stage10_export(processor)

        self.assertIn(
            ("savetif", "result_processed", "-astro"),
            processor.cmd_calls,
        )
        self.assertEqual(processor._stage_review_reasons(3), [])

    def test_stage10_uncalibrated_background_cast_forces_review_only_names(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage4_color_review_required = False
        processor._stage7_background_color_review_required = True
        processor._require_review(
            7,
            "uncalibrated_background_color_review_required",
        )
        processor._stage7_background_color_review_gate = {
            "status": "review_required",
            "requires_review": True,
            "value": 0.189,
            "limit": 0.12,
            "global_white_balance_applied": False,
        }
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", "result_review"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertTrue(processor._final_output_review_only)
        self.assertIn(
            "stage7_uncalibrated_background_color_review_required=true",
            processor.results[-1][3],
        )
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "uncalibrated_background_color_review_required",
        )
        self.assertTrue(
            processor.result_metadata[-1]["details"][
                "stage7_background_color_review_required"
            ]
        )

    def test_stage10_explicit_review_only_setting_withholds_normal_names(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor.cfg.force_review_only_output = True
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True

        stage10_export(processor)

        self.assertIn(("savetif", "result_review", "-astro"), processor.cmd_calls)
        self.assertIn(("savepng", "result_review"), processor.cmd_calls)
        self.assertIn(("save", "result_review_final"), processor.cmd_calls)
        self.assertFalse(
            any(
                "result_processed" in str(item) or "result_final" in str(item)
                for call in processor.cmd_calls
                for item in call
            )
        )
        self.assertIn("force_review_only_output=true", processor.results[-1][3])

    def test_stage10_applied_required_star_remix_keeps_normal_output_names(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage9_bypassed_bad_starless = False
        processor._stage9_stars_required = True
        processor._stage9_stars_applied = True
        processor._stage9_stars_application_mode = "screen"

        stage10_export(processor)

        self.assertIn(
            ("savetif", "result_processed", "-astro"),
            processor.cmd_calls,
        )
        self.assertIn(("save", "result_final"), processor.cmd_calls)
        self.assertFalse(processor._final_output_review_only)

    def test_stage10_saves_fits_before_preview_autostretch_png(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)

        stage10_export(processor)

        commands = [call[0] for call in processor.cmd_calls]
        self.assertLess(commands.index("save"), commands.index("autostretch"))
        self.assertLess(commands.index("autostretch"), commands.index("savepng"))
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("PNG preview stretch applied", message)

    def test_stage10_skips_second_preview_stretch_for_accepted_stage7_output(self):
        processor = self._new_processor()
        self._stage10_final_input(processor)
        processor._stage7_stretch_accepted = True

        stage10_export(processor)

        commands = [call[0] for call in processor.cmd_calls]
        self.assertNotIn("autostretch", commands)
        self.assertIn("savepng", commands)
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("second autostretch skipped", message)
