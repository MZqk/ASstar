"""Pipeline/plugin fallback tests for stage5 linear."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage5LinearTests(PipelinePluginFallbackTestBase):
    def test_stage5_low_noise_guard_skips_enabled_denoise_candidates(self):
        processor = self._new_processor()
        processor.cfg.stage5_deconvolution_enabled = False
        processor.image_pixels = np.full_like(processor.image_pixels, 0.1)

        stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["denoise"]["reason_code"], "auto_low_noise")
        self.assertEqual(report["components"]["denoise"]["status"], "skipped")
        self.assertFalse(any(call[0] == "denoise" for call in processor.cmd_calls))

    def test_stage5_manual_denoise_disable_is_not_auto_low_noise(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.cfg.stage5_deconvolution_enabled = False
        processor.cfg.auto_tune_enabled = True
        processor.auto_tune_result = object()
        processor._task_manual_override_fields = ("denoise_enabled",)

        stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["denoise"]["reason_code"], "user_disabled")
        self.assertEqual(report["components"]["denoise"]["status"], "skipped")

    def test_stage5_forced_denoise_disable_is_user_disabled(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.cfg.stage5_deconvolution_enabled = False
        processor.cfg.auto_tune_enabled = True
        processor.auto_tune_result = object()
        processor._force_denoise_enabled = False

        stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["denoise"]["reason_code"], "user_disabled")
        self.assertEqual(report["components"]["denoise"]["status"], "skipped")

    def test_stage5_keeps_builtin_denoise_primary_with_legacy_plugins_available(self):
        processor = self._new_processor()
        processor.cfg.aberration_api_enabled = True
        processor.aberration_labels["矫正星点"] = "SASP Aberration API (CPU)"
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Denoise.py",
            }
        )
        processor.script_labels["锐化"] = "CosmicClarity Sharpen script (CosmicClarity_Sharpen.py)"
        processor.script_labels["初步降噪"] = "CosmicClarity Denoise script (CosmicClarity_Denoise.py)"

        stage5_linear_denoise(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 siril_builtin denoise candidate accepted", message)
        self.assertNotIn("矫正星点", processor.aberration_calls)
        sharpen_calls = [args for step, _name, args in processor.script_calls if step == "锐化"]
        self.assertFalse(sharpen_calls)

    def test_stage5_does_not_reintroduce_legacy_global_sharpen_for_local_model(self):
        processor = self._new_processor()
        processor.cfg.aberration_api_enabled = False
        processor.local_aberration_model = Path("/tmp/model_v2_0_1.onnx")
        processor.aberration_labels["矫正星点"] = "SASP Aberration API (CPU) [model_v2_0_1.onnx]"
        processor.available_scripts.add("processing/CosmicClarity_Sharpen.py")
        processor.script_fail_steps.add("锐化")
        processor.command_labels["锐化"] = "Unsharp fallback"

        stage5_linear_denoise(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 siril_builtin denoise candidate accepted", message)
        self.assertNotIn("矫正星点", processor.aberration_calls)

    def test_stage5_falls_back_to_internal_denoise_when_scripts_unavailable(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = True
        processor.command_labels["锐化"] = "Unsharp fallback"

        stage5_linear_denoise(processor)

        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 siril_builtin denoise candidate accepted", message)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("denoise", cmds)

    def test_stage5_runs_deconvolution_before_linear_denoise(self):
        processor = self._new_processor()
        processor.siril.get_image_stars = self._stage5_psf_stars

        stage5_linear_denoise(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertLess(cmds.index("findstar"), cmds.index("denoise"))
        self.assertLess(cmds.index("makepsf"), cmds.index("denoise"))
        self.assertLess(cmds.index("rl"), cmds.index("denoise"))
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["final_linear_source"], "stage5_linear")
        self.assertEqual(report["denoise"]["input"], "stage5_deconv")
        self.assertTrue(report["deconvolution"]["runs_before_denoise"])
        self.assertEqual(report["deconvolution"]["psf_quality"]["status"], "available")
        self.assertEqual(
            report["deconvolution"]["local_star_guard"]["mode"],
            "enforced",
        )
        self.assertTrue(
            report["deconvolution"]["local_star_guard"]["enforced"]
        )
        self.assertTrue(
            report["deconvolution"]["local_star_guard"]["accepted"]
        )

    def test_stage5_prefers_graxpert_object_deconvolution_when_model_exists(self):
        processor = self._new_processor()
        processor.siril.get_image_stars = self._stage5_psf_stars
        processor.cfg.denoise_enabled = False
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": "",
            },
            clear=False,
        ):
            os.environ.pop("STARUN_GRAXPERT_GPU", None)
            stage5_linear_denoise(processor)

        graxpert_call = next(
            args
            for step, _name, args in processor.script_calls
            if step == "Stage5 GraXpert反卷积"
        )
        self.assertIn("-deconv_obj", graxpert_call)
        self.assertIn("1.0.1", graxpert_call)
        self.assertIn("-gpu", graxpert_call)
        self.assertNotIn("-nogpu", graxpert_call)
        self.assertNotIn("rl", [str(call[0]) for call in processor.cmd_calls])
        self.assertNotIn("denoise", [str(call[0]) for call in processor.cmd_calls])
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "graxpert_object")
        self.assertEqual(report["denoise"]["input"], "stage5_graxpert_deconv")
        self.assertEqual(report["components"]["deconvolution"]["status"], "applied")
        self.assertEqual(report["components"]["denoise"]["status"], "skipped")
        self.assertEqual(
            report["deconvolution"]["graxpert"]["hardware_acceleration"],
            "auto",
        )
        self.assertEqual(
            report["components"]["denoise"]["reason_code"],
            "config_disabled",
        )
        self.assertEqual(
            processor.result_metadata[-1]["components"],
            report["components"],
        )
        self.assertEqual(report["final_linear_source"], "stage5_linear")
        self.assertEqual(report["deconvolution"]["psf_quality"]["status"], "not_run")
        self.assertNotIn("stage5_psf_quality.json", processor.stage_json_reports)
        self.assertIn(
            report["deconvolution"]["local_star_guard"]["status"],
            {"available", "unavailable"},
        )
        self.assertEqual(
            report["deconvolution"]["local_star_guard"]["mode"],
            "enforced",
        )
        self.assertTrue(
            report["deconvolution"]["local_star_guard"][
                "participates_in_acceptance"
            ]
        )

    def test_stage5_graxpert_local_guard_retries_once_at_lower_strength(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")
        original_runner = processor._run_plugin_script_by_path

        def ring_primary(step_key, label, script_path, *, args=()):
            result = original_runner(
                step_key,
                label,
                script_path,
                args=args,
            )
            if step_key == "Stage5 GraXpert反卷积":
                strength = args[args.index("-strength") + 1]
                if strength == "0.30":
                    yy, xx = np.mgrid[
                        : processor.image_pixels.shape[-2],
                        : processor.image_pixels.shape[-1],
                    ]
                    for star in self._stage5_psf_stars():
                        radius = np.sqrt(
                            (xx - float(star.xpos)) ** 2
                            + (yy - float(star.ypos)) ** 2
                        )
                        ring = (radius >= 3.0) & (radius <= 4.5)
                        processor.image_pixels[:, ring] += 0.08
            return result

        processor._run_plugin_script_by_path = ring_primary
        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": "",
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        attempts = report["deconvolution"]["local_star_guard_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]["accepted"])
        self.assertTrue(attempts[1]["accepted"])
        self.assertEqual(report["deconvolution"]["method"], "graxpert_object")
        self.assertEqual(
            report["deconvolution"]["graxpert"]["accepted_strength"],
            0.25,
        )
        self.assertEqual(
            len(
                [
                    call
                    for call in processor.script_calls
                    if call[0] == "Stage5 GraXpert反卷积"
                ]
            ),
            2,
        )

    def test_stage5_unavailable_local_measurement_rolls_back_without_retry(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.siril.get_image_stars = lambda: []
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": "",
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        guard = report["deconvolution"]["local_star_guard"]
        self.assertEqual(guard["status"], "unavailable")
        self.assertFalse(guard["accepted"])
        self.assertTrue(guard["rollback_required"])
        self.assertEqual(report["deconvolution"]["method"], "none")
        self.assertEqual(
            len(
                [
                    call
                    for call in processor.script_calls
                    if call[0] == "Stage5 GraXpert反卷积"
                ]
            ),
            1,
        )

    def test_stage5_graxpert_cpu_compatibility_disables_hardware_acceleration(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": "",
                "STARUN_GRAXPERT_GPU": "0",
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        graxpert_call = next(
            args
            for step, _name, args in processor.script_calls
            if step == "Stage5 GraXpert反卷积"
        )
        self.assertIn("-nogpu", graxpert_call)
        self.assertNotIn("-gpu", graxpert_call)
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(
            report["deconvolution"]["graxpert"]["hardware_acceleration"],
            "cpu",
        )

    def test_stage5_graxpert_failure_reloads_baseline_then_falls_back_to_rl(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        processor.script_fail_steps.add("Stage5 GraXpert反卷积")
        model = (
            processor.work_dir
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"mock onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": "",
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        self.assertIn("rl", [str(call[0]) for call in processor.cmd_calls])
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "siril_rl")
        self.assertTrue(report["deconvolution"]["graxpert"]["attempted"])

    def test_stage5_links_user_provided_graxpert_model_into_isolated_home(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        external_model = (
            processor.work_dir
            / "user-models"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        external_model.parent.mkdir(parents=True)
        external_model.write_bytes(b"user supplied onnx")
        isolated_home = processor.work_dir / "isolated-home"

        with patch.dict(
            os.environ,
            {
                "HOME": str(isolated_home),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": str(external_model.parent.parent),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        linked_model = (
            isolated_home
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        self.assertTrue(linked_model.is_symlink())
        self.assertEqual(linked_model.resolve(), external_model.resolve())
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "graxpert_object")
        self.assertEqual(
            report["deconvolution"]["graxpert"]["source"], "user_provided"
        )
        self.assertEqual(
            report["deconvolution"]["graxpert"]["resolved_model_path"],
            str(linked_model),
        )

    def test_stage5_accepts_identical_model_already_linked_from_another_source(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        configured_model = (
            processor.work_dir
            / "bundled-models"
            / "1.0.1"
            / "model.onnx"
        )
        configured_model.parent.mkdir(parents=True)
        configured_model.write_bytes(b"identical object deconvolution model")
        existing_model = (
            processor.work_dir
            / "graxpert-app-models"
            / "1.0.1"
            / "model.onnx"
        )
        existing_model.parent.mkdir(parents=True)
        existing_model.write_bytes(configured_model.read_bytes())
        isolated_home = processor.work_dir / "isolated-home"
        isolated_model = (
            isolated_home
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        isolated_model.parent.mkdir(parents=True)
        isolated_model.symlink_to(existing_model)

        with patch.dict(
            os.environ,
            {
                "HOME": str(isolated_home),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": str(configured_model),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "graxpert_object")
        self.assertEqual(report["deconvolution"]["graxpert"]["reason"], "")
        self.assertEqual(
            report["deconvolution"]["graxpert"]["resolved_model_path"],
            str(isolated_model),
        )
        self.assertNotIn("rl", [str(call[0]) for call in processor.cmd_calls])

    def test_stage5_rejects_different_model_in_same_isolated_version(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        configured_model = (
            processor.work_dir
            / "bundled-models"
            / "1.0.1"
            / "model.onnx"
        )
        configured_model.parent.mkdir(parents=True)
        configured_model.write_bytes(b"new object deconvolution model")
        isolated_home = processor.work_dir / "isolated-home"
        isolated_model = (
            isolated_home
            / "Library"
            / "Application Support"
            / "GraXpert"
            / "deconvolution-object-ai-models"
            / "1.0.1"
            / "model.onnx"
        )
        isolated_model.parent.mkdir(parents=True)
        isolated_model.write_bytes(b"different model with same version")

        with patch.dict(
            os.environ,
            {
                "HOME": str(isolated_home),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": str(configured_model),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "siril_rl")
        self.assertEqual(
            report["deconvolution"]["graxpert"]["reason"],
            "model_version_conflicts_with_isolated_home",
        )

    def test_stage5_invalid_user_model_path_falls_back_to_rl(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        missing_model = processor.work_dir / "missing-model"

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir / "isolated-home"),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": str(missing_model),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        graxpert = report["deconvolution"]["graxpert"]
        self.assertEqual(report["deconvolution"]["method"], "siril_rl")
        self.assertEqual(graxpert["reason"], "configured_model_not_found_or_invalid")
        self.assertEqual(graxpert["configured_path"], str(missing_model))

    def test_stage5_accepts_user_model_without_semantic_version_directory(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/GraXpert-AI.py")
        model = processor.work_dir / "user-model" / "custom-object-model.onnx"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"user supplied onnx")

        with patch.dict(
            os.environ,
            {
                "HOME": str(processor.work_dir / "isolated-home"),
                "STARUN_GRAXPERT_OBJECT_MODEL_PATH": str(model),
            },
            clear=False,
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        graxpert = report["deconvolution"]["graxpert"]
        self.assertEqual(report["deconvolution"]["method"], "graxpert_object")
        linked_model = Path(graxpert["resolved_model_path"])
        self.assertTrue(linked_model.is_symlink())
        self.assertEqual(linked_model.resolve(), model.resolve())
        self.assertRegex(linked_model.parent.name, r"^user-[0-9a-f]{12}$")

    def test_stage5_rl_failure_reloads_input_before_denoise(self):
        processor = self._new_processor()
        processor.fail_commands.add("rl")

        stage5_linear_denoise(processor)

        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertLess(cmds.index("rl"), cmds.index("denoise"))
        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertFalse(report["deconvolution"]["applied"])
        self.assertEqual(report["denoise"]["input"], "stage5_input_linear")

    def test_stage5_rl_local_star_rejection_rolls_back_without_retry(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        original_cmd = processor.cmd_with_check

        def ring_after_rl(*args, quiet=False):
            result = original_cmd(*args, quiet=quiet)
            if args and args[0] == "rl":
                yy, xx = np.mgrid[
                    : processor.image_pixels.shape[-2],
                    : processor.image_pixels.shape[-1],
                ]
                for star in self._stage5_psf_stars():
                    radius = np.sqrt(
                        (xx - float(star.xpos)) ** 2
                        + (yy - float(star.ypos)) ** 2
                    )
                    ring = (radius >= 3.0) & (radius <= 4.5)
                    processor.image_pixels[:, ring] += 0.08
            return result

        processor.cmd_with_check = ring_after_rl

        stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["deconvolution"]["method"], "none")
        self.assertEqual(
            report["components"]["deconvolution"]["status"],
            "rolled_back",
        )
        self.assertFalse(
            report["deconvolution"]["local_star_guard"]["accepted"]
        )
        self.assertEqual(
            [call[0] for call in processor.cmd_calls].count("rl"),
            1,
        )
        np.testing.assert_array_equal(
            processor.image_pixels,
            processor.saved_image_pixels["stage5_input_linear"],
        )

    def test_stage5_double_restore_failure_stops_later_candidates(self):
        processor = self._new_processor()
        original_cmd = processor.cmd_with_check

        def ring_after_rl(*args, quiet=False):
            result = original_cmd(*args, quiet=quiet)
            if args and args[0] == "rl":
                yy, xx = np.mgrid[
                    : processor.image_pixels.shape[-2],
                    : processor.image_pixels.shape[-1],
                ]
                for star in self._stage5_psf_stars():
                    radius = np.sqrt(
                        (xx - float(star.xpos)) ** 2
                        + (yy - float(star.ypos)) ** 2
                    )
                    processor.image_pixels[
                        :,
                        (radius >= 3.0) & (radius <= 4.5),
                    ] += 0.08
            return result

        processor.cmd_with_check = ring_after_rl
        stage5_module = sys.modules["stages.stage5_linear_denoise"]
        with patch.object(
            stage5_module,
            "_stage5_restore_denoise_baseline",
            return_value={
                "required": True,
                "completed": False,
                "method": "failed",
                "checkpoint_error": "mock checkpoint failure",
                "pixel_error": "mock frozen pixel failure",
            },
        ):
            stage5_linear_denoise(processor)

        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertFalse(report["deconvolution"]["integrity_ok"])
        self.assertEqual(
            report["components"]["deconvolution"]["status"],
            "failed",
        )
        self.assertFalse(any(call[0] == "denoise" for call in processor.cmd_calls))
        self.assertIn(
            "deconvolution_rollback_failed",
            processor._stage_review_reasons(5),
        )

    def test_stage5_all_deconvolution_methods_failed_marks_stage_degraded(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.fail_commands.add("rl")

        stage5_linear_denoise(processor)

        _name, status, _duration, message = processor.results[-1]
        report = processor.stage_json_reports["stage5_linear_report.json"]
        metadata = processor.result_metadata[-1]
        self.assertEqual(status, "degraded")
        self.assertEqual(report["status"], "degraded")
        self.assertEqual(report["components"]["deconvolution"]["status"], "failed")
        self.assertEqual(report["reason_code"], "deconvolution_unavailable")
        self.assertEqual(metadata["reason_code"], "deconvolution_unavailable")
        self.assertIn("all enabled deconvolution methods were unavailable", message)

    def test_stage5_rollback_without_denoise_marks_stage_degraded(self):
        processor = self._new_processor()
        processor.cfg.denoise_enabled = False
        processor.adaptive_measurements.extend(
            [
                {
                    "dirty_background_score": 0.10,
                    "chroma_noise_score": 0.01,
                    "bg_std": 0.00010,
                },
                {
                    "dirty_background_score": 0.18,
                    "chroma_noise_score": 0.02,
                    "bg_std": 0.00013,
                },
            ]
        )

        stage5_linear_denoise(processor)

        _name, status, _duration, message = processor.results[-1]
        report = processor.stage_json_reports["stage5_linear_report.json"]
        metadata = processor.result_metadata[-1]
        self.assertEqual(status, "degraded")
        self.assertEqual(report["components"]["deconvolution"]["status"], "rolled_back")
        self.assertEqual(report["components"]["denoise"]["status"], "skipped")
        self.assertEqual(
            report["reason_code"],
            "deconvolution_rollback_without_denoise",
        )
        self.assertTrue(metadata["fallback_used"])
        self.assertIn("deconvolution was rolled back and denoise was not applied", message)

    def test_stage5_skips_classic_cosmic_clarity_without_executable(self):
        processor = self._new_processor()
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Denoise.py",
            }
        )
        processor.classic_cc_args = None

        stage5_linear_denoise(processor)

        classic_calls = [
            call
            for call in processor.script_calls
            if call[1] in {"CosmicClarity_Sharpen.py", "CosmicClarity_Denoise.py"}
        ]
        self.assertFalse(classic_calls)
        cmds = [str(call[0]) for call in processor.cmd_calls]
        self.assertIn("denoise", cmds)

    def test_stage5_prefers_builtin_denoise_when_cosmic_native_is_available(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": False,
                "denoise_mode": "chroma_first",
            },
        }
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Denoise.py",
                "processing/CosmicClarity_Native.py",
            }
        )
        processor.classic_cc_args = None

        stage5_linear_denoise(processor)

        native_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Native.py"
        ]
        self.assertFalse(native_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 siril_builtin denoise candidate accepted", message)

    def test_stage5_skips_global_sharpen_when_background_policy_protects_background(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "dark_nebula_low_contrast",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": True,
                "denoise_mode": "chroma_first",
                "sharpen_mode": "minimal",
            },
        }
        processor.adaptive_measurements.extend(
            [
                {
                    "dirty_background_score": 0.02,
                    "chroma_noise_score": 0.01,
                    "gradient_score": 0.01,
                    "bg_std": 0.0001,
                },
                {
                    "dirty_background_score": 0.02,
                    "chroma_noise_score": 0.01,
                    "gradient_score": 0.01,
                    "bg_std": 0.0001,
                },
            ]
        )
        processor.available_scripts.update(
            {
                "processing/CosmicClarity_Sharpen.py",
                "processing/CosmicClarity_Native.py",
            }
        )

        stage5_linear_denoise(processor)

        sharpen_calls = [call for call in processor.script_calls if call[0] == "锐化"]
        self.assertFalse(sharpen_calls)
        _name, _status, _dur, message = processor.results[-1]
        self.assertIn("Stage5 siril_builtin denoise candidate accepted", message)

    def test_stage5_uses_builtin_denoise_for_chroma_first_policy(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": True,
                "denoise_mode": "chroma_first",
            },
        }
        processor.available_scripts.add("processing/CosmicClarity_Denoise.py")
        processor.script_labels["初步降噪"] = "CosmicClarity Denoise script (CosmicClarity_Denoise.py)"

        stage5_linear_denoise(processor)

        denoise_calls = [args for step, _name, args in processor.script_calls if step == "初步降噪"]
        self.assertFalse(denoise_calls)
        self.assertIn(("denoise", "-mod=0.35", "-indep"), processor.cmd_calls)

    def test_stage5_multiscale_candidate_uses_common_transaction(self):
        processor = self._new_processor()
        processor.cfg.stage5_multiscale_denoise_enabled = True

        stage5_linear_denoise(processor)

        transaction = processor.stage_json_reports["stage5_denoise_attempts.json"]
        self.assertEqual(transaction["baseline"]["status"], "ready")
        self.assertTrue(transaction["integrity_ok"])
        self.assertEqual(transaction["accepted_method"], "deterministic_multiscale")
        self.assertEqual(transaction["attempts"][0]["status"], "accepted")
        self.assertFalse(any(call[0] == "denoise" for call in processor.cmd_calls))

    def test_stage5_builtin_consumes_denoise_mod_and_safety_max(self):
        processor = self._new_processor()
        processor.cfg.denoise_mod = 0.48
        processor.cfg.denoise_safety_max = 0.32

        stage5_linear_denoise(processor)

        self.assertIn(("denoise", "-mod=0.32", "-indep"), processor.cmd_calls)
        report = processor.stage_json_reports["stage5_linear_report.json"]
        self.assertEqual(report["denoise"]["siril_builtin_mod"], 0.32)
        self.assertEqual(
            report["denoise"]["siril_builtin_mod_source"],
            "denoise_mod clamped by denoise_safety_max",
        )

    def test_stage5_rejected_builtin_candidate_restores_shared_baseline(self):
        processor = self._new_processor()
        processor.cfg.stage5_multiscale_denoise_enabled = True
        original_cmd = processor.cmd_with_check

        def aggressive_builtin(*args: Any, quiet: bool = False) -> bool:
            result = original_cmd(*args, quiet=quiet)
            if args and args[0] == "denoise":
                processor.image_pixels = np.full_like(
                    processor.image_pixels,
                    0.05,
                )
            return result

        processor.cmd_with_check = aggressive_builtin
        rejected_multiscale = {
            "schema": "starun.multiscale-denoise-candidate.v1",
            "status": "rejected",
            "accepted": False,
            "issues": ["signal_detail_retention"],
            "transaction": {"pixels_mutated": False},
        }
        stage5_module = sys.modules["stages.stage5_linear_denoise"]

        with patch.object(
            stage5_module,
            "_run_multiscale_linear_denoise",
            return_value=(False, rejected_multiscale),
        ):
            stage5_linear_denoise(processor)

        baseline = processor.saved_image_pixels["stage5_pre_denoise"]
        np.testing.assert_array_equal(processor.image_pixels, baseline)
        transaction = processor.stage_json_reports["stage5_denoise_attempts.json"]
        builtin = next(
            attempt
            for attempt in transaction["attempts"]
            if attempt.get("method") == "siril_builtin"
        )
        self.assertEqual(builtin["status"], "rejected")
        self.assertFalse(builtin["accepted"])
        self.assertTrue(builtin["transaction"]["rollback_completed"])
        self.assertIn(
            "signal_detail_retention",
            builtin["quality_gate"]["issues"],
        )
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "all_candidates_rejected_safe_passthrough",
        )

    def test_stage5_partial_builtin_failure_rolls_back_before_next_backend(self):
        processor = self._new_processor()
        original_cmd = processor.cmd_with_check

        def partially_mutating_failure(*args: Any, quiet: bool = False) -> bool:
            if args and args[0] == "denoise":
                processor.cmd_calls.append(args)
                processor.image_pixels[:] = 0.25
                raise processor.module.CommandError("mock partial denoise failure")
            return original_cmd(*args, quiet=quiet)

        processor.cmd_with_check = partially_mutating_failure

        stage5_linear_denoise(processor)

        baseline = processor.saved_image_pixels["stage5_pre_denoise"]
        np.testing.assert_array_equal(processor.image_pixels, baseline)
        transaction = processor.stage_json_reports["stage5_denoise_attempts.json"]
        builtin = next(
            attempt
            for attempt in transaction["attempts"]
            if attempt.get("method") == "siril_builtin"
        )
        self.assertEqual(builtin["status"], "failed")
        self.assertTrue(builtin["transaction"]["rollback_completed"])
        baseline_loads = [
            call
            for call in processor.cmd_calls
            if call == ("load", "stage5_pre_denoise")
        ]
        self.assertGreaterEqual(len(baseline_loads), 2)

    def test_stage5_prohibits_denoise_without_immutable_baseline(self):
        processor = self._new_processor()
        original_save = processor._save_stage_output

        def fail_baseline_save(stem: str) -> bool:
            if stem == "stage5_pre_denoise":
                return False
            return original_save(stem)

        processor._save_stage_output = fail_baseline_save

        stage5_linear_denoise(processor)

        self.assertFalse(any(call[0] == "denoise" for call in processor.cmd_calls))
        transaction = processor.stage_json_reports["stage5_denoise_attempts.json"]
        self.assertEqual(transaction["baseline"]["status"], "prohibited")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "immutable_baseline_unavailable",
        )

    def test_stage5_rolls_back_when_background_chroma_gets_worse(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "generic_low_snr_safe",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": False,
            },
        }
        processor.cfg.denoise_enabled = True
        processor.cfg.denoise_mod = 0.35
        processor.adaptive_measurements.extend(
            [
                {
                    "dirty_background_score": 0.25,
                    "chroma_noise_score": 0.08,
                    "bg_std": 0.00010,
                },
                {
                    "dirty_background_score": 0.34,
                    "chroma_noise_score": 0.12,
                    "bg_std": 0.00013,
                },
            ]
        )

        stage5_linear_denoise(processor)

        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        self.assertIn(("denoise", "-mod=0.35", "-indep"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 background guard dropped siril_rl result", message)
        self.assertTrue(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "component_fallback_used",
        )

    def test_stage5_deconvolution_background_std_growth_boundary(self):
        before = {
            "bg_std": 0.001,
            "chroma_noise_score": 0.01,
            "dirty_background_score": 0.01,
        }

        accepted, _reason = stage5_linear_denoise_module._stage5_background_worsened(
            before,
            {**before, "bg_std": 0.00138},
        )
        rejected, reason = stage5_linear_denoise_module._stage5_background_worsened(
            before,
            {**before, "bg_std": 0.001381},
        )

        self.assertFalse(accepted)
        self.assertTrue(rejected)
        self.assertIn("bg_std_growth=1.381", reason)

    def test_stage5_rolls_back_when_chroma_becomes_more_visible_than_luma_noise(self):
        processor = self._new_processor()
        processor.pipeline_policy = {
            "policy_name": "bright_nebula_hdr_conservative",
            "stage5_linear": {
                "protect_background": True,
                "avoid_global_sharpen": False,
                "denoise_mode": "chroma_first",
            },
        }
        processor.cfg.denoise_enabled = True
        processor.cfg.denoise_mod = 0.35
        processor.adaptive_measurements.extend(
            [
                {
                    "dirty_background_score": 0.0051,
                    "chroma_noise_score": 0.01346,
                    "bg_std": 0.000055,
                },
                {
                    "dirty_background_score": 0.0048,
                    "chroma_noise_score": 0.01388,
                    "bg_std": 0.000033,
                },
            ]
        )

        stage5_linear_denoise(processor)

        self.assertIn(("load", "stage5_input_linear"), processor.cmd_calls)
        self.assertIn(("denoise", "-mod=0.35", "-indep"), processor.cmd_calls)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("Stage5 background guard dropped siril_rl result", message)
        self.assertIn("chroma_bg_ratio_growth", message)

    def test_stage3_graxpert_runtime_error_uses_normal_background_backup(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(workflow_plugin_probe_enabled=False)
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.input_profile = {
                    "state": "linear",
                    "safe_for_linear_steps": True,
                    "source": "test_fixture",
                }
                self.siril = Stage3SampleSiril()
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
                        "gradient_score": 0.03,
                        "dirty_background_score": 0.16,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                if args and (
                    args[0] in ("gxp", "graxpert")
                    or (args[0] == "pyscript" and "-bge" in args)
                ):
                    raise RuntimeError(
                        "GraXpert-AI.py Error: too many indices for array: "
                        "array is 2-dimensional, but 3 were indexed"
                    )
                return True

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return complete_stage3_preservation()

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
        candidates = [
            (
                "GraXpert-AI BGE CPU",
                (
                    "pyscript",
                    "GraXpert-AI.py",
                    "-bge",
                    "-model",
                    "model_v2_0_1",
                    "-correction",
                    "subtraction",
                    "-keep_bg",
                    "-nogpu",
                ),
                "graxpert",
            ),
            (
                "ADBE",
                ("pyscript", "AutoBGE.py", "-npoints", "120"),
                "plugin",
            ),
        ]
        accepted_pixel_gate = {
            "status": "accepted",
            "accepted": True,
            "severity": "normal",
            "warnings": [],
            "hard_issues": [],
            "issues": [],
        }
        accepted_validation_gate = {
            "status": "accepted",
            "accepted": True,
            "severity": "normal",
            "warnings": [],
            "hard_issues": [],
            "issues": [],
        }
        with (
            patch.object(
                stage3_module,
                "_stage3_background_candidate_chain",
                return_value=(candidates, [], "test_integrated_backends"),
            ),
            patch.object(
                stage3_module,
                "_stage3_candidate_pixel_gate",
                return_value=(True, accepted_pixel_gate),
            ),
            patch.object(
                stage3_module,
                "assess_single_background_validation",
                return_value=(True, accepted_validation_gate),
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(
            background_attempts[:2],
            [
                candidates[0][1],
                candidates[1][1],
            ],
        )
        self.assertFalse(
            any(call and call[0] in {"gxp", "graxpert", "adbe"} for call in processor.cmd_calls)
        )
        self.assertEqual(processor.workflow_command_used["背景提取插件链"], "ADBE")
        self.assertTrue(processor.report["graxpert_runtime_error"])
        self.assertTrue(processor.report["fallback_triggered_by_graxpert_error"])
        self.assertTrue(processor.report["backup_used"])
        self.assertEqual(
            processor.report["backup_reason"],
            "graxpert_runtime_error_external_selected",
        )
        self.assertFalse(processor.report["fallback_used"])
        self.assertIsNone(processor.report["fallback_reason"])
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertFalse(processor.report["review_required"])
        graxpert_statuses = [
            record["status"]
            for record in processor.report["attempts"]
            if record.get("source") == "graxpert"
        ]
        self.assertEqual(
            graxpert_statuses,
            ["graxpert_runtime_error"],
        )

    def test_stage3_graxpert_success_without_image_change_is_runtime_error(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.fingerprints = iter(("unchanged", "unchanged"))
                self.cmd_calls: list[tuple[Any, ...]] = []

            def _validate_plugin_script_prerequisites(self, _script_path: Path):
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
            '"/mock/siril-scripts/processing/GraXpert-AI.py"',
            "-bge",
        )

        ok, reason = stage3_module._stage3_try_background_command(
            processor,
            "GraXpert",
            command,
            "graxpert",
        )

        self.assertFalse(ok)
        self.assertEqual(processor.cmd_calls, [command])
        self.assertEqual(
            reason,
            "graxpert_runtime_error: command returned success but image did not change",
        )

    def test_stage3_primary_graxpert_candidate_uses_locked_subtraction_contract(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(log=FakeLogger())
        script = Path("/mock/siril-scripts/processing/GraXpert-AI.py")

        def model_ready(target):
            target._stage3_graxpert_provenance = {}
            return True

        with (
            patch.object(stage3_module, "_stage3_find_script", return_value=script),
            patch.object(
                stage3_module,
                "_stage3_ensure_graxpert_bge_model",
                side_effect=model_ready,
            ),
            patch.object(
                stage3_module,
                "_stage3_sha256",
                return_value=stage3_module.STAGE3_GRAXPERT_SCRIPT_SHA256,
            ),
        ):
            candidates = stage3_module._stage3_graxpert_candidates(processor)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], "GraXpert-AI BGE CPU")
        self.assertEqual(
            candidates[0][1],
            (
                "pyscript",
                '"/mock/siril-scripts/processing/GraXpert-AI.py"',
                "-bge",
                "-model",
                "model_v2_0_1",
                "-correction",
                "subtraction",
                "-keep_bg",
                "-nogpu",
            ),
        )

    def test_stage3_candidate_chain_puts_builtins_before_graxpert(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(log=FakeLogger())
        primary = (
            "GraXpert-AI BGE CPU",
            ("pyscript", "GraXpert-AI.py", "-bge", "-nogpu"),
            "graxpert",
        )
        plugin_backup = ("ADBE", ("adbe",), "plugin")

        with patch.object(
            stage3_module,
            "_stage3_theoretical_plugin_candidates",
            return_value=[primary, plugin_backup],
        ):
            ordered, builtin_labels, _reason = (
                stage3_module._stage3_background_candidate_chain(
                    processor,
                    rbf_attempts=[("rbf", ("subsky", "-rbf", "-existing"), "builtin")],
                    poly_attempt=[("poly", ("subsky", "1", "-existing"), "builtin")],
                    poly_first=True,
                )
            )

        self.assertEqual(
            [label for label, _command, _source in ordered],
            ["poly", "rbf", "GraXpert-AI BGE CPU", "ADBE"],
        )
        self.assertEqual(builtin_labels, ["poly", "rbf"])

    def test_stage3_graxpert_missing_onnx_is_rejected_before_execution(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake:
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cmd_calls: list[tuple[Any, ...]] = []

            def _validate_plugin_script_prerequisites(self, script_path: Path):
                self.validated_script = script_path
                return False, "missing python modules: onnx"

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                return True

        processor = Stage3Fake()
        command = (
            "pyscript",
            '"/mock/siril scripts/processing/GraXpert-AI.py"',
            "-bge",
        )

        ok, reason = stage3_module._stage3_try_background_command(
            processor,
            "GraXpert",
            command,
            "graxpert",
        )

        self.assertFalse(ok)
        self.assertFalse(processor.cmd_calls)
        self.assertEqual(
            processor.validated_script,
            Path("/mock/siril scripts/processing/GraXpert-AI.py"),
        )
        self.assertEqual(
            reason,
            "graxpert_runtime_error: prerequisites unavailable: "
            "missing python modules: onnx",
        )

    def test_graxpert_prerequisites_include_script_import_names(self):
        modules = pipeline_module.StarunPostProcessor._SCRIPT_PREREQUISITE_MODULES[
            "GraXpert-AI.py"
        ]

        self.assertIn("onnx", modules)
        self.assertIn("appdirs", modules)
        self.assertNotIn("platformdirs", modules)
