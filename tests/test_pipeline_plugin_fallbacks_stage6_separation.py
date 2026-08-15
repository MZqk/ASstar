"""Pipeline/plugin fallback tests for stage6 separation."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage6SeparationTests(PipelinePluginFallbackTestBase):
    def test_stage6_repair_score_growth_gate_honors_configured_limit(self):
        cases = (
            ("default_strict", 0.0, 0.0, True),
            ("below_limit", 0.10, 0.05, True),
            ("at_limit", 0.10, 0.10, True),
            ("over_limit", 0.10, 0.1001, False),
        )
        for label, limit, growth, expected in cases:
            with self.subTest(label=label):
                audit = stage6_star_separation_module._stage6_repair_acceptance(
                    score_before=1.0,
                    score_after=1.0 + growth,
                    configured_max_score_growth=limit,
                    non_regression_passed=True,
                    trigger_improved=True,
                    chroma_improved=False,
                )

                self.assertEqual(audit["accepted"], expected)
                self.assertEqual(audit["score_gate_passed"], expected)
                self.assertAlmostEqual(audit["score_before"], 1.0)
                self.assertAlmostEqual(audit["score_after"], 1.0 + growth)
                self.assertAlmostEqual(audit["score_growth"], growth)
                self.assertAlmostEqual(audit["score_growth_max"], limit)
                self.assertEqual(
                    audit["gate_conclusion"],
                    "accepted" if expected else "rejected",
                )

    def test_stage6_repair_score_growth_does_not_bypass_other_gates(self):
        audit = stage6_star_separation_module._stage6_repair_acceptance(
            score_before=1.0,
            score_after=1.05,
            configured_max_score_growth=0.20,
            non_regression_passed=False,
            trigger_improved=False,
            chroma_improved=False,
        )

        self.assertTrue(audit["score_gate_passed"])
        self.assertFalse(audit["non_regression_gate_passed"])
        self.assertFalse(audit["improvement_gate_passed"])
        self.assertFalse(audit["accepted"])

    def test_pyqt6_headless_stub_includes_sasp_stage8_widget_imports(self):
        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "PyQt6" or name.startswith("PyQt6.")
        }
        try:
            processor = self._new_processor()
            installed = pipeline_module.StarunPostProcessor._install_pyqt6_headless_stub(
                processor
            )
            self.assertTrue(installed)
            from PyQt6.QtGui import QDoubleValidator, QIntValidator, QPainter
            from PyQt6.QtCore import QCoreApplication, QPoint, QUrl, pyqtProperty
            from PyQt6.QtQuickWidgets import QQuickWidget

            self.assertIsNotNone(QIntValidator)
            self.assertIsNotNone(QDoubleValidator)
            self.assertIsNotNone(QPainter)
            self.assertIsNotNone(QCoreApplication)
            self.assertIsNotNone(QPoint)
            self.assertIsNotNone(QUrl)
            self.assertIsNotNone(pyqtProperty)
            self.assertIsNotNone(QQuickWidget)
        finally:
            for name in list(sys.modules):
                if name == "PyQt6" or name.startswith("PyQt6."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)

    def test_sasp_stage8_widget_shim_avoids_widget_package_side_effects(self):
        if not hasattr(pipeline_module.np, "array"):
            self.skipTest("real numpy is not available in this test interpreter")
        wheel_dir = REPO_ROOT / "resources" / "siril_plugins" / "downloads"
        wheels = sorted(wheel_dir.glob("setiastrosuitepro-*.whl"))
        if not wheels:
            self.skipTest("setiastrosuitepro wheel not bundled")

        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "setiastro.saspro.widgets"
            or name.startswith("setiastro.saspro.widgets.")
        }
        try:
            processor = self._new_processor()
            pipeline_module.StarunPostProcessor._install_pyqt6_headless_stub(processor)
            pipeline_module.StarunPostProcessor._install_sasp_stage8_widget_import_shims(
                processor,
                wheels[-1],
            )

            self.assertIn("setiastro.saspro.widgets.wavelet_utils", sys.modules)
            widgets_pkg = sys.modules.get("setiastro.saspro.widgets")
            self.assertIsNotNone(widgets_pkg)
            self.assertFalse(getattr(widgets_pkg, "__file__", None))
            wavelet = sys.modules["setiastro.saspro.widgets.wavelet_utils"]
            for name in ("atrous_decompose", "rgb_to_lab", "lab_to_rgb"):
                self.assertTrue(callable(getattr(wavelet, name, None)), name)
        finally:
            for name in list(sys.modules):
                if name == "setiastro.saspro.widgets" or name.startswith(
                    "setiastro.saspro.widgets."
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)

    def test_sasp_stage8_widget_shim_uses_numpy_for_numba_wheel_cache_error(self):
        wheel_dir = REPO_ROOT / "resources" / "siril_plugins" / "downloads"
        wheels = sorted(wheel_dir.glob("setiastrosuitepro-*.whl"))
        if not wheels:
            self.skipTest("setiastrosuitepro wheel not bundled")

        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "setiastro.saspro.widgets"
            or name.startswith("setiastro.saspro.widgets.")
        }
        real_import = __import__

        def import_with_numba_cache_failure(
            name,
            globals=None,
            locals=None,
            fromlist=(),
            level=0,
        ):
            if name == "setiastro.saspro.legacy.numba_utils":
                raise RuntimeError(
                    "cannot cache function 'rgb_to_xyz_numba': no locator "
                    "available for file '/runtime/setiastrosuitepro.whl/numba_utils.py'"
                )
            return real_import(name, globals, locals, fromlist, level)

        try:
            processor = self._new_processor()
            pipeline_module.StarunPostProcessor._install_pyqt6_headless_stub(
                processor
            )
            with patch(
                "builtins.__import__",
                side_effect=import_with_numba_cache_failure,
            ):
                pipeline_module.StarunPostProcessor._install_sasp_stage8_widget_import_shims(
                    processor,
                    wheels[-1],
                )

            wavelet = sys.modules["setiastro.saspro.widgets.wavelet_utils"]
            self.assertFalse(wavelet._HAVE_NUMBA)
            rgb = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
            lab = wavelet.rgb_to_lab(rgb)
            restored = wavelet.lab_to_rgb(lab)
            self.assertEqual(lab.shape, rgb.shape)
            self.assertTrue(np.isfinite(restored).all())
            self.assertTrue(np.allclose(restored, rgb, atol=1e-4))
        finally:
            pipeline_module.sasp_runner._clear_sasp_stage8_import_modules()
            sys.modules.update(saved)

    def test_sasp_stage8_widget_shim_does_not_hide_other_runtime_errors(self):
        wheel_dir = REPO_ROOT / "resources" / "siril_plugins" / "downloads"
        wheels = sorted(wheel_dir.glob("setiastrosuitepro-*.whl"))
        if not wheels:
            self.skipTest("setiastrosuitepro wheel not bundled")

        real_import = __import__

        def import_with_unexpected_runtime_error(
            name,
            globals=None,
            locals=None,
            fromlist=(),
            level=0,
        ):
            if name == "setiastro.saspro.legacy.numba_utils":
                raise RuntimeError("unexpected Numba initialization failure")
            return real_import(name, globals, locals, fromlist, level)

        processor = self._new_processor()
        pipeline_module.StarunPostProcessor._install_pyqt6_headless_stub(processor)
        with (
            patch(
                "builtins.__import__",
                side_effect=import_with_unexpected_runtime_error,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "unexpected Numba initialization failure",
            ),
        ):
            pipeline_module.StarunPostProcessor._install_sasp_stage8_widget_import_shims(
                processor,
                wheels[-1],
            )

        for module_name in pipeline_module.sasp_runner._SASP_STAGE8_IMPORT_MODULES:
            self.assertNotIn(module_name, sys.modules)

    def test_sasp_stage8_loader_stops_when_widget_shim_fails(self):
        def fail_widget_shim(_wheel_path):
            raise RuntimeError("unexpected Numba initialization failure")

        processor = SimpleNamespace(
            _sasp_stage8_module=None,
            _sasp_stage8_module_error=None,
            _find_latest_sasp_wheel=lambda: Path("/mock/setiastrosuitepro.whl"),
            _install_pyqt6_headless_stub=lambda: True,
            _install_sasp_stage8_widget_import_shims=fail_widget_shim,
            _short_text=lambda value: str(value),
            cfg=SimpleNamespace(debug_mode=False),
            log=FakeLogger(),
        )

        with patch.object(
            pipeline_module.sasp_runner.importlib,
            "import_module",
        ) as import_module:
            loaded = pipeline_module.sasp_runner.load_sasp_stage8_module(processor)

        self.assertIsNone(loaded)
        import_module.assert_not_called()
        self.assertEqual(
            processor._sasp_stage8_module_error,
            "widget shim failed: unexpected Numba initialization failure",
        )

    def test_stage6_allows_sasp_fallback_when_probe_disabled(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.command_labels["去星"] = "SASP Dark Star"

        used = processor._run_first_available_command(
            "去星",
            [("SASP Dark Star", ("sasp_dark_star",))],
            allow_when_probe_disabled=True,
        )

        self.assertEqual(used, "SASP Dark Star")
        self.assertEqual(processor.workflow_command_used.get("去星"), "SASP Dark Star")

    def test_cli_subprocess_treats_syqon_callback_output_error_as_failure(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)
        script_path = Path(td.name) / "Starless.py"
        script_path.write_text("# mock\n", encoding="utf-8")
        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Could not send output to Siril: Timeout while receiving data\n"
                    "Processing complete!\n"
                ),
            )

        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "去星",
                    "SyQon Starless",
                    script_path,
                )

        self.assertIsNone(used)
        self.assertIn(
            "Could not send output to Siril",
            processor._last_plugin_script_error or "",
        )

    def test_cleanup_removes_runtime_star_layers_and_sasp_exchange_files(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.debug_mode = False
        processor.stretched_name = "stage7_stretched"

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.work_dir = Path(td.name)
        processor.process_dir = processor.work_dir / "process"
        processor.process_dir.mkdir()
        runtime_files = {
            "starless.fit",
            "starmask.fit",
            "starmask_raw.fit",
            "starmask_clean.fit",
            "starmask_external_raw.fit",
            "starmask_stretched.fit",
            "stage8_limited_candidate.fit",
        }
        for name in runtime_files | {"temporary.fit"}:
            (processor.process_dir / name).write_bytes(name.encode("utf-8"))
        (processor.process_dir / "final_quality_report.json").write_text(
            '{"status":"ok"}', encoding="utf-8"
        )
        for name in ("sasp_starless_input.fit", "sasp_starmask_input.fit"):
            (processor.work_dir / name).write_bytes(name.encode("utf-8"))

        processor.cleanup()

        self.assertTrue(
            all(not (processor.process_dir / name).exists() for name in runtime_files)
        )
        self.assertFalse((processor.process_dir / "temporary.fit").exists())
        self.assertTrue((processor.process_dir / "final_quality_report.json").is_file())
        self.assertFalse((processor.work_dir / "sasp_starless_input.fit").exists())
        self.assertFalse((processor.work_dir / "sasp_starmask_input.fit").exists())

    def test_stage6_halo_threshold_is_target_aware_for_diffuse_emission(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_large_galaxy_halo_residue_score_max = 0.48
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60

        processor._active_target_type = lambda: "galaxy"
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.35)
        processor._active_target_type = lambda: "large_galaxy"
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.48)
        processor._active_target_type = lambda: "emission_nebula_widefield"
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.45)
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.60)

        processor._active_target_type = lambda: "dark_nebula_low_contrast"
        processor.target_profile = {
            "secondary_labels": [
                "large_nebulosity",
                "faint_outer_cloud",
                "emission_red",
            ],
            "features": {},
        }
        self.assertEqual(processor._stage7_effective_halo_threshold(), 0.45)

    def test_stage6_mixed_nebula_uses_compact_halo_evidence(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._active_target_type = lambda: "dark_nebula_low_contrast"
        processor.target_profile = {
            "secondary_labels": [
                "large_nebulosity",
                "faint_outer_cloud",
                "emission_red",
            ],
            "features": {},
        }
        height = width = 192
        yy, xx = np.mgrid[:height, :width]
        background = np.full((height, width), 0.012, dtype=np.float32)
        diffuse = (
            0.070
            * np.exp(
                -(
                    ((xx - 92.0) / 48.0) ** 2
                    + ((yy - 88.0) / 34.0) ** 2
                )
            )
        ).astype(np.float32)
        source_gray = background + diffuse
        starmask_gray = np.zeros_like(source_gray)
        for cy, cx, amplitude in (
            (38, 42, 0.35),
            (58, 142, 0.24),
            (126, 54, 0.20),
            (142, 136, 0.28),
        ):
            radius2 = (yy - cy) ** 2 + (xx - cx) ** 2
            star = amplitude * np.exp(-radius2 / 10.0)
            source_gray += star
            starmask_gray += star
        source = np.repeat(source_gray[None, :, :], 3, axis=0)
        starless = np.repeat((background + diffuse)[None, :, :], 3, axis=0)
        starmask = np.repeat(starmask_gray[None, :, :], 3, axis=0)

        scores = stage7_quality_module.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )

        self.assertEqual(scores["diffuse_nebula_context"], 1.0)
        self.assertGreater(scores["diffuse_nebula_protection_coverage"], 0.0)
        self.assertLess(
            scores["global_halo_residue_score"],
            processor._stage7_effective_halo_threshold(),
        )

        residual_starless = np.repeat(
            (background + diffuse + 0.25 * starmask_gray)[None, :, :],
            3,
            axis=0,
        )
        residual_scores = stage7_quality_module.stage7_starless_artifact_scores(
            processor,
            source,
            residual_starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(residual_starless),
        )
        self.assertGreater(
            residual_scores["global_halo_residue_score"],
            processor._stage7_effective_halo_threshold(),
        )

    def test_stage6_galaxy_roi_does_not_treat_bulge_and_arms_as_halo(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._active_target_type = lambda: "large_galaxy"
        source, starless, starmask = self._synthetic_galaxy_starless_layers()

        scores = stage7_quality_module.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )

        self.assertEqual(scores["galaxy_roi_available"], 1.0)
        self.assertLess(scores["galaxy_roi_disk_coverage"], 0.30)
        self.assertLess(
            scores["galaxy_disk_halo_residue_score"],
            processor.cfg.stage7_halo_residue_score_max,
        )
        self.assertLess(
            scores["global_halo_residue_score"],
            processor.cfg.stage7_halo_residue_score_max,
        )
        self.assertGreater(
            scores["galaxy_core_preservation_ratio"],
            processor.cfg.stage7_galaxy_core_preservation_ratio_min,
        )
        self.assertGreater(
            scores["galaxy_core_contrast_ratio"],
            processor.cfg.stage7_galaxy_core_contrast_ratio_min,
        )
        self.assertLess(
            scores["starmask_contamination"],
            processor.cfg.stage7_starmask_contamination_max,
        )

    def test_stage6_galaxy_pixel_repair_preserves_fitted_disk_exactly(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._active_target_type = lambda: "large_galaxy"
        source, starless, starmask = self._synthetic_galaxy_starless_layers()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        processor.process_dir = Path(temp_dir.name)
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        processor._read_image_by_stem = lambda stem: {
            "source": source,
            "starmask": starmask,
        }.get(stem)
        processor.cmd_with_check = lambda *_args, **_kwargs: None
        processor.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: starless,
        )
        captured: dict[str, np.ndarray] = {}
        processor._set_current_image_pixeldata = (
            lambda data, *, label: captured.setdefault("output", np.asarray(data))
        )
        processor._save_stage_output = lambda _stem: True

        result = stage7_repair_module.apply_stage7_starless_pixel_repair(
            processor,
            source_stem="source",
            label="galaxy-protection-regression",
        )
        protection_mask = stage7_repair_module._stage7_galaxy_protection_mask(
            processor,
            source,
        )

        self.assertEqual(result["status"], "applied", result)
        self.assertIsNotNone(protection_mask)
        self.assertGreater(
            result["metrics"]["galaxy_structure_protection_coverage"],
            0.05,
        )
        np.testing.assert_array_equal(
            captured["output"][:, protection_mask],
            starless[:, protection_mask],
        )

    def test_stage6_candidate_selection_prefers_ok_before_soft_score(self):
        processor = SimpleNamespace(
            _stage7_quality_score=lambda quality: float(quality["score"]),
        )
        poor_low_score = {"status": "poor", "score": 0.01}
        ok_higher_score = {"status": "ok", "score": 0.40}

        self.assertLess(
            stage6_star_separation_module._stage7_quality_selection_key(
                processor,
                ok_higher_score,
            ),
            stage6_star_separation_module._stage7_quality_selection_key(
                processor,
                poor_low_score,
            ),
        )

    def test_stage6_starmask_borderline_limits_handoff_and_star_intensity(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._active_target_type = lambda: "generic_low_snr_safe"
        quality = {
            "status": "ok",
            "issues": [],
            "derived": {
                "halo_residue_score": 0.10,
                "compact_halo_residue_score": 0.10,
                "residual_star_score": 0.10,
                "starless_noise_gain": 1.0,
                "starmask_diffuse_residual_ratio": 0.12,
                "starmask_diffuse_residual_ratio_max": 0.08,
                "starmask_diffuse_uncertainty_abs": 0.0005,
                "starmask_diffuse_advisory_multiplier": 2.0,
                "starmask_diffuse_effective_hard_limit": 0.16,
                "starmask_cleanup_borderline": True,
            },
        }

        handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
            processor,
            quality,
            [],
            separation_accepted=True,
        )
        remix = processor._stage7_update_star_remix_from_quality(quality)

        self.assertEqual(handoff["processing_policy"], "limited")
        self.assertEqual(
            handoff["reason_code"],
            "starmask_diffuse_residual_borderline",
        )
        self.assertAlmostEqual(remix["intensity_scale"], 0.70)
        self.assertIn("advisory band", remix["reason"])

    def test_stage6_quality_advisory_continues_with_limited_handoff(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._active_target_type = lambda: "generic_low_snr_safe"
        quality = {
            "status": "ok",
            "issues": [],
            "advisories": ["residual_stars 0.700>0.450"],
            "derived": {
                "halo_residue_score": 0.10,
                "compact_halo_residue_score": 0.10,
                "residual_star_score": 0.70,
                "starless_noise_gain": 1.0,
                "starmask_cleanup_borderline": False,
            },
        }

        handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
            processor,
            quality,
            [],
            separation_accepted=True,
        )

        self.assertEqual(handoff["processing_policy"], "limited")
        self.assertEqual(handoff["reason_code"], "stage6_quality_advisory")
        self.assertNotEqual(handoff["processing_policy"], "skip")
        self.assertIn("residual_stars", handoff["reason_text"])

    def test_stage6_contract_valid_quality_failure_is_retained_for_limited_review(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._active_target_type = lambda: "generic_low_snr_safe"
        processor._stage6_quality_hard_failed_retained = True
        processor._stage6_quality_failure_codes = ["RESIDUAL_GLOBAL", "HALO"]
        processor._selected_syqon_attempt_id = "attempt-1"
        processor._selected_syqon_pair_id = "pair-1"
        handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
            processor,
            {
                "status": "poor",
                "derived": {
                    "halo_residue_score": 0.50,
                    "compact_halo_residue_score": 0.50,
                    "residual_star_score": 0.70,
                    "starless_noise_gain": 1.0,
                },
            },
            [],
            separation_accepted=True,
        )

        self.assertEqual(handoff["processing_policy"], "limited")
        self.assertTrue(handoff["restricted_downstream"])
        self.assertEqual(
            handoff["reason_code"],
            "stage6_quality_hard_failed_retained",
        )
        self.assertEqual(handoff["reasons"][0]["star_intensity_cap"], 0.70)
        self.assertEqual(handoff["attempt_id"], "attempt-1")
        self.assertEqual(handoff["pair_id"], "pair-1")

    def test_stage6_quality_over_two_x_is_retained_limited_when_contract_valid(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._active_target_type = lambda: "generic_low_snr_safe"
        quality = {
            "status": "ok",
            "issues": [],
            "derived": {
                "halo_residue_score": 0.10,
                "compact_halo_residue_score": 0.10,
                "residual_star_score": 0.901,
                "starless_noise_gain": 1.0,
                "starmask_cleanup_borderline": False,
            },
        }

        handoff = stage6_star_separation_module._stage8_handoff_from_stage6(
            processor,
            quality,
            [],
            separation_accepted=True,
        )

        self.assertEqual(handoff["processing_policy"], "limited")
        self.assertEqual(
            handoff["reason_code"],
            "stage6_quality_hard_failed_retained",
        )
        self.assertEqual(handoff["reasons"][0]["hard_metrics"], ["residual"])
        self.assertEqual(handoff["reasons"][0]["star_intensity_cap"], 0.70)

    def test_stage6_poor_quality_still_triggers_pixel_repair(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._active_target_type = lambda: "galaxy"
        quality = {
            "status": "poor",
            "derived": {
                "halo_residue_score": 0.10,
                "compact_halo_residue_score": 0.08,
            },
        }

        trigger = (
            stage6_star_separation_module._stage7_starless_pixel_repair_trigger(
                processor,
                quality,
            )
        )

        self.assertTrue(trigger["triggered"])
        self.assertEqual(trigger["reason"], "quality_status=poor")

    def test_stage6_compact_halo_triggers_refinement_and_candidate_penalty(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_bright_nebula_halo_residue_score_max = 0.60
        processor._active_target_type = lambda: "bright_emission_reflection_nebula"

        safe = {
            "derived": {
                "residual_star_score": 0.03,
                "halo_residue_score": 0.49,
                "compact_halo_residue_score": 0.58,
                "black_hole_score": 0.0,
                "starmask_contamination": 0.0,
                "starless_noise_gain": 1.0,
                "starless_dynamic_range_ratio": 1.0,
                "starless_peak_signal": 1.0,
                "starmask_coverage_ratio": 1.0,
                "starmask_width_ratio": 1.0,
            }
        }
        compact_halo = {
            "derived": {
                **safe["derived"],
                "compact_halo_residue_score": 0.89,
            }
        }

        self.assertEqual(processor._stage7_repair_triggers(safe), [])
        self.assertIn(
            "compact_halo_residue",
            processor._stage7_repair_triggers(compact_halo),
        )
        self.assertGreater(
            processor._stage7_quality_score(compact_halo),
            processor._stage7_quality_score(safe),
        )

    def test_stage6_compact_halo_measurement_stays_local_in_dense_star_field(self):
        processor = pipeline_module.StarunPostProcessor()
        processor._active_target_type = lambda: "galaxy"
        height = width = 256
        yy, xx = np.mgrid[:height, :width]
        background = 0.015 + xx.astype(np.float32) / width * 0.018
        source_gray = background.copy()
        starmask_gray = np.zeros_like(background)
        for cy in range(12, height, 20):
            for cx in range(12, width, 20):
                radius2 = (yy - cy) ** 2 + (xx - cx) ** 2
                star = 0.22 * np.exp(-radius2 / 5.0)
                source_gray += star
                starmask_gray += star
        source = np.repeat(source_gray[None, :, :], 3, axis=0)
        starless = np.repeat(background[None, :, :], 3, axis=0)
        starmask = np.repeat(starmask_gray[None, :, :], 3, axis=0)

        scores = pipeline_module.stage7_quality.stage7_starless_artifact_scores(
            processor,
            source,
            starless,
            starmask,
            pipeline_module.measure_image_features(source),
            pipeline_module.measure_image_features(starless),
        )

        self.assertEqual(scores["galaxy_roi_available"], 0.0)
        self.assertLess(scores["compact_halo_mask_coverage"], 0.50)
        self.assertLess(scores["compact_halo_residue_score"], 0.60)

    def test_stage6_syqon_variant_collects_script_outputs(self):
        processor = self._new_processor()
        processor.siril_plugin_dir = processor.work_dir / "siril_plugins"
        model_dir = processor.siril_plugin_dir / "syqon_starless"
        model_dir.mkdir(parents=True)
        model_bytes = b"verified-zenith-model"
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        (model_dir / "zenith.pt").write_bytes(model_bytes)
        (model_dir / "zenith.pt.sha256").write_text(
            model_hash + "  zenith.pt\n",
            encoding="utf-8",
        )
        processor.available_scripts.add("SyQon/Starless.py")
        processor.syqon_output_mode = "both"
        fits.PrimaryHDU(
            data=np.full((3, 32, 32), 0.20, dtype=np.float32)
        ).writeto(
            processor.process_dir / f"{processor.stretched_name}.fit"
        )
        processor._clear_star_separation_outputs = (  # type: ignore[method-assign]
            lambda: pipeline_module.syqon_starless.clear_star_separation_outputs(processor)
        )
        processor._stage7_prepare_starmask = lambda: None  # type: ignore[method-assign]
        script = processor._find_plugin_script(("SyQon/Starless.py",))
        self.assertIsNotNone(script)

        with (
            patch.dict(
                os.environ,
                {pipeline_module.syqon_starless.ENV_SYQON_MODEL_DIR_KEY: ""},
                clear=False,
            ),
            patch.object(
                pipeline_module.syqon_starless,
                "SYQON_ZENITH_SHA256",
                model_hash,
            ),
            patch.object(
                pipeline_module.syqon_starless,
                "verify_syqon_supply_chain",
                return_value=(
                    {
                        "upstream": {"commit": "test-fixture"},
                        "script": {"sha256": "test-fixture"},
                        "patch": {"sha256": "test-fixture"},
                        "model": {
                            "name": "Zenith",
                            "sha256": model_hash,
                        },
                    },
                    None,
                ),
            ),
        ):
            used = pipeline_module.syqon_starless.stage7_try_syqon_variant(
                processor,
                script,
                attempt_name="initial",
                profile=pipeline_module.syqon_starless.SYQON_BASELINE_PROFILE,
            )

        self.assertIsNotNone(used)
        self.assertIn("SyQon Starless initial", processor.workflow_command_used["去星"])
        syqon_calls = [
            args
            for step, script_name, args in processor.script_calls
            if step == "去星" and script_name == "Starless.py"
        ]
        self.assertTrue(syqon_calls)
        self.assertIn("--tile-size", syqon_calls[0])
        self.assertIn("--overlap", syqon_calls[0])
        self.assertIn("--input-file", syqon_calls[0])
        self.assertIn("--starless-output", syqon_calls[0])
        self.assertIn("--starmask-output", syqon_calls[0])
        self.assertIn("--manifest-output", syqon_calls[0])
        self.assertIn("--stretch-method", syqon_calls[0])
        self.assertIn("--target-median", syqon_calls[0])
        self.assertIn("--stat-bp-sigma", syqon_calls[0])
        self.assertIn("--mask-method", syqon_calls[0])
        self.assertIn("--unlinked-stretch", syqon_calls[0])
        self.assertIn("--black-clip", syqon_calls[0])
        self.assertIn("--no-amp", syqon_calls[0])
        self.assertNotIn("--no_gpu", syqon_calls[0])
        self.assertTrue((processor.process_dir / "starless.fit").exists())
        self.assertTrue((processor.process_dir / "starmask_raw.fit").exists())
        exchange = processor.stage_json_reports["stage6_syqon_exchange.json"]
        self.assertEqual(exchange["schema"], "starun.syqon-pixel-exchange.v2")
        self.assertEqual(exchange["worker"]["requested"]["target_median"], 0.15)
        self.assertTrue(exchange["pair_id"])
        self.assertEqual(
            exchange["stop_reason"],
            "CONTRACT_VALID_PAIR_COMMITTED",
        )
        self.assertEqual(
            exchange["shadow_metrics"]["transform_roundtrip"]["status"],
            "shadow",
        )
        self.assertEqual(
            exchange["shadow_metrics"]["tiling"]["coverage"]["status"],
            "shadow",
        )

    def test_stage6_syqon_exchange_canary_accepts_stable_linear_scale(self):
        source = np.linspace(0.005, 0.21, 4096, dtype=np.float32).reshape(
            1,
            64,
            64,
        )
        starless = np.clip(source * 0.91, 0.0, 1.0)

        report = pipeline_module.syqon_starless.assess_syqon_exchange_pixels(
            source,
            starless,
        )

        self.assertEqual(report["status"], "accepted")
        self.assertTrue(report["accepted"])
        self.assertAlmostEqual(report["metrics"]["median_ratio"], 0.91, places=5)

    def test_stage6_syqon_exchange_canary_rejects_gross_median_scale_jump(self):
        source = np.full((3, 64, 64), 0.023, dtype=np.float32)
        starless = np.full((3, 64, 64), 0.278, dtype=np.float32)

        report = pipeline_module.syqon_starless.assess_syqon_exchange_pixels(
            source,
            starless,
        )

        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["accepted"])
        self.assertGreater(report["metrics"]["median_ratio"], 12.0)
        self.assertTrue(
            any(
                "median_scale_ratio_out_of_bounds" in issue
                for issue in report["issues"]
            )
        )

    def test_stage6_syqon_model_dir_prefers_explicit_verified_bundle(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        model_dir = Path(td.name) / "explicit-models"
        model_dir.mkdir()
        model_bytes = b"explicit-zenith"
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        (model_dir / "zenith.pt").write_bytes(model_bytes)
        (model_dir / "zenith.pt.sha256").write_text(
            hashlib.sha256(model_bytes).hexdigest() + "  zenith.pt\n",
            encoding="utf-8",
        )
        processor = SimpleNamespace(siril_plugin_dir=Path(td.name) / "missing")

        with (
            patch.dict(
                os.environ,
                {
                    pipeline_module.syqon_starless.ENV_SYQON_MODEL_DIR_KEY: str(
                        model_dir
                    )
                },
                clear=False,
            ),
            patch.object(
                pipeline_module.syqon_starless,
                "SYQON_ZENITH_SHA256",
                model_hash,
            ),
        ):
            resolved, error = (
                pipeline_module.syqon_starless.resolve_syqon_model_dir(processor)
            )

        self.assertEqual(resolved, model_dir.resolve())
        self.assertIsNone(error)

    def test_stage6_syqon_supply_chain_matches_locked_project_assets(self):
        plugin_root = REPO_ROOT / "resources" / "siril_plugins"
        processor = SimpleNamespace(siril_plugin_dir=plugin_root)
        script = plugin_root / "vendor" / "siril-scripts" / "SyQon" / "Starless.py"
        model_dir = plugin_root / "syqon_starless"

        assets, error = pipeline_module.syqon_starless.verify_syqon_supply_chain(
            processor,
            script,
            model_dir,
        )

        self.assertIsNone(error)
        self.assertIsNotNone(assets)
        assert assets is not None
        self.assertEqual(
            assets["upstream"]["commit"],
            pipeline_module.syqon_starless.SYQON_UPSTREAM_COMMIT,
        )
        self.assertEqual(
            assets["upstream"]["sha256"],
            pipeline_module.syqon_starless.SYQON_UPSTREAM_STARLESS_SHA256,
        )
        self.assertEqual(
            assets["model"]["sha256"],
            pipeline_module.syqon_starless.SYQON_ZENITH_SHA256,
        )

    def test_stage6_syqon_bad_checksum_skips_plugin_without_download_path(self):
        processor = self._new_processor()
        processor.siril_plugin_dir = processor.work_dir / "siril_plugins"
        model_dir = processor.siril_plugin_dir / "syqon_starless"
        model_dir.mkdir(parents=True)
        (model_dir / "zenith.pt").write_bytes(b"corrupted")
        (model_dir / "zenith.pt.sha256").write_text(
            "0" * 64 + "  zenith.pt\n",
            encoding="utf-8",
        )
        processor.available_scripts.add("SyQon/Starless.py")
        script = processor._find_plugin_script(("SyQon/Starless.py",))
        self.assertIsNotNone(script)

        with patch.dict(
            os.environ,
            {pipeline_module.syqon_starless.ENV_SYQON_MODEL_DIR_KEY: ""},
            clear=False,
        ):
            used = pipeline_module.syqon_starless.stage7_try_syqon_variant(
                processor,
                script,
                attempt_name="initial",
                profile=pipeline_module.syqon_starless.SYQON_BASELINE_PROFILE,
            )

        self.assertIsNone(used)
        self.assertFalse(processor.script_calls)
        self.assertIn("项目锁定摘要不一致", processor._last_plugin_script_error or "")

    def test_stage6_syqon_script_candidates_prefer_current_layout(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        plugin_root = Path(td.name) / "siril_plugins"
        scripts_root = plugin_root / "vendor" / "siril-scripts"
        (scripts_root / "processing").mkdir(parents=True)
        (scripts_root / "SyQon").mkdir()
        current_script = scripts_root / "SyQon" / "Starless.py"
        current_script.write_text("# current SyQon layout\n", encoding="utf-8")
        legacy_script = scripts_root / "processing" / "SyQon-Starless.py"
        legacy_script.write_text("# legacy SyQon layout\n", encoding="utf-8")
        processor = SimpleNamespace(siril_plugin_dir=plugin_root)
        processor._resolve_siril_scripts_root = lambda: (
            stage_support_module.plugin_runner.resolve_siril_scripts_root(processor)
        )

        script = stage_support_module.plugin_runner.find_plugin_script(
            processor,
            stage6_star_separation_module.SYQON_SCRIPT_CANDIDATES
        )

        self.assertEqual(script, current_script)
        current_script.unlink()
        self.assertIsNone(
            stage_support_module.plugin_runner.find_plugin_script(
                processor,
                stage6_star_separation_module.SYQON_SCRIPT_CANDIDATES,
            )
        )

    def test_stage6_unaccepted_purge_removes_all_starless_artifacts(self):
        processor = self._new_processor()
        removable = {
            "starless.fit",
            "starless_best_initial.fit",
            "stage6_starless.fit",
            "stage6_starless_repaired.fit",
            "starmask.fit",
            "starmask_raw_best_initial.fit",
        }
        survivors = {
            "stage6_input.fit",
            "stage6_passthrough.fit",
            "stage6_starless_quality.json",
        }
        for name in removable | survivors:
            (processor.process_dir / name).write_bytes(name.encode("utf-8"))
        lineage_root = processor.process_dir / ".stage6_syqon" / "raw-attempt"
        lineage_root.mkdir(parents=True)
        (lineage_root / "starless.fit").write_bytes(b"invalid-pair")
        (processor.process_dir / "stage6_syqon_selected.json").write_text(
            "{}",
            encoding="utf-8",
        )
        processor._selected_syqon_pair_id = "invalid-pair"
        processor._selected_syqon_attempt_id = "raw-attempt"
        for name in ("sasp_starless_input.fit", "sasp_starmask_input.fit"):
            (processor.work_dir / name).write_bytes(name.encode("utf-8"))

        removed = pipeline_module.syqon_starless.purge_unaccepted_star_separation_outputs(
            processor
        )

        self.assertTrue(removable.issubset(set(removed)))
        self.assertTrue(all(not (processor.process_dir / name).exists() for name in removable))
        self.assertTrue(all((processor.process_dir / name).exists() for name in survivors))
        self.assertFalse((processor.process_dir / ".stage6_syqon").exists())
        self.assertFalse((processor.process_dir / "stage6_syqon_selected.json").exists())
        self.assertIsNone(processor._selected_syqon_pair_id)
        self.assertIsNone(processor._selected_syqon_attempt_id)
        self.assertFalse((processor.work_dir / "sasp_starless_input.fit").exists())
        self.assertFalse((processor.work_dir / "sasp_starmask_input.fit").exists())

    def test_stage6_cpu_recovery_is_an_explicit_profile(self):
        processor = self._new_processor()

        args, _timeout, note = processor._syqon_starless_cli_options(
            profile=pipeline_module.syqon_starless.SYQON_CPU_RECOVERY_PROFILE,
        )

        self.assertIn("--no_gpu", args)
        self.assertIn("zenith_cpu_recovery", note)

    def test_syqon_baseline_cli_is_explicit_zenith_fp32_contract(self):
        processor = self._new_processor()

        args, _timeout, note = pipeline_module.syqon_starless.syqon_starless_cli_options(
            processor,
            profile=pipeline_module.syqon_starless.SYQON_BASELINE_PROFILE,
        )

        self.assertIn("--no-amp", args)
        self.assertIn("--unlinked-stretch", args)
        self.assertIn("--black-clip", args)
        self.assertIn("model=Zenith", note)
        self.assertIn("precision=FP32", note)

    def test_current_syqon_script_keeps_offline_prerequisite_checks(self):
        modules = pipeline_module.StarunPostProcessor._SCRIPT_PREREQUISITE_MODULES[
            "Starless.py"
        ]

        self.assertEqual(
            modules,
            ("PyQt6", "PySide6", "astropy", "scipy"),
        )
        self.assertNotIn(
            "SyQon-Starless.py",
            pipeline_module.StarunPostProcessor._SCRIPT_PREREQUISITE_MODULES,
        )
