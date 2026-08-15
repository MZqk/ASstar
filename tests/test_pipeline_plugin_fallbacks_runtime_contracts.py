"""Pipeline/plugin fallback tests for runtime contracts."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackRuntimeContractTests(PipelinePluginFallbackTestBase):
    def test_stage_review_bundle_is_local_only(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        payload = {
            "status": "ready",
            "stage": "stage3_background_extraction",
            "visual_review": {
                "status": "not_requested",
                "advisor_mode": "not_requested",
            },
            "candidates": [
                {
                    "selection_status": "selected",
                    "visual_acceptance_status": "not_requested",
                }
            ],
        }

        with patch.object(
            stage_support_module.review_bundle,
            "create_stage_review_bundle",
            return_value=payload,
        ):
            result = processor._create_stage_review_bundle(
                "stage3_background_extraction",
                "before",
                "after",
            )

        self.assertEqual(result["visual_review"]["status"], "not_requested")
        self.assertEqual(result["visual_review"]["advisor_mode"], "not_requested")

    def test_stage11_entrypoint_is_removed(self):
        processor = pipeline_module.StarunPostProcessor()
        self.assertFalse(hasattr(processor, "stage11_ai_postprocess"))

    def test_stage_preview_publishes_only_accepted_artifact_pixels(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        process_dir = work_dir / "process"
        process_dir.mkdir()
        (process_dir / "stage1_prepared.fit").write_bytes(b"accepted")
        processor = pipeline_module.StarunPostProcessor()
        processor.work_dir = work_dir
        processor.process_dir = process_dir
        processor.log = FakeLogger()
        processor.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: np.array(
                [[0.0, 0.25], [0.5, 0.75]],
                dtype=np.float32,
            )
        )
        commands = []
        processor.cmd_with_check = lambda *args: commands.append(args)

        processor._publish_stage_preview(1, "前期准备", "ok")

        preview_path = process_dir / "ui_preview" / "latest.png"
        self.assertTrue(preview_path.is_file())
        self.assertIn(("load", "stage1_prepared"), commands)
        self.assertTrue(
            any(
                "[PIPELINE_PREVIEW]" in message and '"status":"ready"' in message
                for level, message in processor.log.events
                if level == "info"
            )
        )

    def test_failed_or_skipped_stage_does_not_publish_preview(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.process_dir = Path("/tmp/unused-preview-test")
        processor.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: (_ for _ in ()).throw(
                AssertionError("failed/skipped stage must not decode")
            )
        )

        processor._publish_stage_preview(3, "背景提取", "failed")
        processor._publish_stage_preview(4, "校色", "skipped")

        self.assertFalse(processor.log.events)

    def test_unexpected_preview_error_cannot_change_stage_result(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.results = []
        processor.log = FakeLogger()
        processor._publish_stage_preview = lambda *_args: (_ for _ in ()).throw(
            AttributeError("mock preview failure")
        )

        processor._record_stage("阶段 2: 裁切", "ok", 0.2, "accepted")

        self.assertEqual(processor.results[-1].status, "ok")
        self.assertTrue(
            any(
                "预览观察链路异常" in message
                for level, message in processor.log.events
                if level == "warn"
            )
        )

    def test_record_stage_emits_gui_state_from_structured_fields_only(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.results = []
        processor.log = FakeLogger()
        processor._publish_stage_preview = lambda *_args: None

        processor._record_stage(
            "阶段 9: 星点处理与合成",
            "ok",
            1.2,
            "fallback_used=true; controlled Screen remix",
        )
        processor._record_stage(
            "阶段 10: 最终降噪与导出",
            "ok",
            0.5,
            "final denoise skipped because input is already low-noise",
        )
        processor._record_stage(
            "阶段 6: 去星与 Halo 修复",
            "ok",
            0.8,
            "primary tool failed; alternate accepted",
            fallback_used=True,
            reason_code="alternate_star_separation",
        )
        processor._record_stage(
            "阶段 8: Starless 深加工",
            "ok",
            0.4,
            "guard retained the accepted Stage 7 source",
            execution="safe_passthrough",
            reason_code="bright_nebula_halo_advisory",
        )

        events = [
            message
            for level, message in processor.log.events
            if level == "info" and "[PIPELINE_STAGE_RESULT]" in message
        ]
        detail_events = [
            json.loads(message.split("[PIPELINE_STAGE_DETAIL] ", 1)[1])
            for level, message in processor.log.events
            if level == "info" and "[PIPELINE_STAGE_DETAIL]" in message
        ]
        self.assertIn("stage=9 status=ok", events[0])
        self.assertIn("stage=10 status=ok", events[1])
        self.assertIn("stage=6 status=degraded", events[2])
        self.assertIn("stage=8 status=ok", events[3])
        self.assertEqual(detail_events[0]["display_status"], "ok")
        self.assertEqual(detail_events[1]["display_status"], "ok")
        self.assertTrue(detail_events[2]["fallback_used"])
        self.assertEqual(detail_events[2]["display_status"], "ok_with_fallback")
        self.assertEqual(detail_events[3]["execution"], "safe_passthrough")
        self.assertEqual(
            detail_events[3]["display_status"],
            "ok_safe_passthrough",
        )
        self.assertEqual(processor.results[0].status, "ok")
        self.assertEqual(processor.results[1].status, "ok")

    def test_stage_outputs_write_only_the_requested_canonical_name(self):
        calls: list[tuple[str, str]] = []
        log = FakeLogger()

        saved = pipeline_module.save_stage_output(
            lambda *args: calls.append(tuple(str(item) for item in args)),
            log,
            "stage6_starless",
        )

        self.assertTrue(saved)
        self.assertIn(("save", "stage6_starless"), calls)
        self.assertNotIn(("save", "stage7_starless"), calls)

        calls.clear()
        saved = pipeline_module.save_stage_output(
            lambda *args: calls.append(tuple(str(item) for item in args)),
            log,
            "stage7_stretched",
        )

        self.assertTrue(saved)
        self.assertIn(("save", "stage7_stretched"), calls)
        self.assertNotIn(("save", "stage6_stretched"), calls)

    def test_stage_json_writes_only_the_requested_canonical_name(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name)
        log = FakeLogger()

        pipeline_module.write_stage_json(
            process_dir,
            log,
            "stage7_stretch_quality.json",
            {"stage": "stage7_stretch"},
        )

        self.assertTrue((process_dir / "stage7_stretch_quality.json").exists())

        pipeline_module.write_stage_json(
            process_dir,
            log,
            "stage6_starless_quality.json",
            {"stage": "stage6_starless"},
        )

        self.assertTrue((process_dir / "stage6_starless_quality.json").exists())
        self.assertFalse((process_dir / "stage7_quality.json").exists())

    def test_legacy_stage_method_aliases_are_removed(self):
        processor = pipeline_module.StarunPostProcessor()
        for legacy_name in (
            "stage6_stretching",
            "stage7_star_separation",
            "stage6_5_pre_starless_gate",
            "stage2_5_target_profiler",
        ):
            self.assertFalse(hasattr(processor, legacy_name), legacy_name)

    def test_debug_stage_save_writes_quality_metrics(self):
        import json

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name)
        processor = pipeline_module.StarunPostProcessor()
        processor.cfg.debug_mode = True
        processor.log = FakeLogger()
        processor.process_dir = process_dir
        processor.siril = SimpleNamespace(
            cmd=lambda *_args: None,
            get_image_pixeldata=lambda preview=False: object(),
        )

        metric_globals = processor._write_debug_quality_metrics.__globals__
        with patch.dict(
            metric_globals,
            {
                "measure_quality_metrics": lambda _image: pipeline_module.QualityMetrics(
                    bg_median=0.123,
                ),
                "measure_image_features": lambda _image: pipeline_module.ImageFeatures(
                    edge_black_ratio=0.045,
                ),
            },
        ):
            self.assertTrue(processor._save_stage_output("stage_debug_probe"))

        metrics_path = process_dir / "stage_debug_probe_quality_metrics.json"
        self.assertTrue(metrics_path.exists())
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "starun.stage_quality.v1")
        self.assertEqual(payload["stem"], "stage_debug_probe")
        self.assertIn("bg_median", payload["metrics"])
        self.assertIn("edge_black_ratio", payload["features"])

        jsonl_path = process_dir / "stage_quality_metrics.jsonl"
        self.assertTrue(jsonl_path.exists())
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["stem"], "stage_debug_probe")
        self.assertTrue(
            any(
                "[STAGE_QUALITY_METRICS]" in message
                for level, message in processor.log.events
                if level == "info"
            )
        )

    def test_cmd_with_check_treats_closed_connection_as_fatal_without_retry(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        calls: list[tuple[Any, ...]] = []

        def dead_cmd(*args: Any) -> None:
            calls.append(args)
            raise pipeline_module.SirilConnectionError("connection closed by Siril")

        processor.siril = SimpleNamespace(cmd=dead_cmd)

        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.cmd_with_check("spcc")

        self.assertTrue(processor._siril_process_terminated)
        self.assertEqual(calls, [("spcc",)])
        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.cmd_with_check("pcc")
        self.assertEqual(calls, [("spcc",)])

    def test_direct_siril_api_connection_death_uses_same_fatal_fuse(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._siril_ever_connected = True
        calls: list[str] = []

        def dead_pixel_read(*_args: Any, **_kwargs: Any):
            calls.append("get_image_pixeldata")
            raise pipeline_module.SirilConnectionError("broken pipe")

        processor.siril = pipeline_module._FatalSirilInterfaceProxy(
            processor,
            SimpleNamespace(get_image_pixeldata=dead_pixel_read),
        )

        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.siril.get_image_pixeldata(preview=False)
        with self.assertRaises(pipeline_module.SirilNativeProcessTerminated):
            processor.siril.get_image_pixeldata(preview=False)

        self.assertEqual(calls, ["get_image_pixeldata"])

    def test_auto_tune_lifts_low_signal_emission_nebula_without_boosting_stars(self):
        cfg = pipeline_module.PipelineConfig()
        tuned, result = pipeline_module.auto_tune_config(
            cfg,
            pipeline_module.TargetType.EMISSION_NEBULA,
            pipeline_module.ImageFeatures(
                bg_median=0.0020,
                bg_std=0.0001,
                red_dominance=1.02,
                blue_dominance=1.01,
                star_density=0.00008,
                object_area_ratio=0.0002,
                diffuse_ratio=0.0,
                core_brightness_ratio=0.19,
            ),
        )

        self.assertGreaterEqual(tuned.nebula_saturation, 0.30)
        self.assertGreaterEqual(tuned.final_saturation, 0.12)
        self.assertLessEqual(tuned.star_intensity, 0.95)
        self.assertGreaterEqual(tuned.asinh_stretch, 1.85)
        self.assertIn("low-signal emission nebula", " ".join(result.notes))

    def test_detect_target_type_recognizes_ic434_path_as_nebula(self):
        target_type = pipeline_module.detect_target_type(
            Path("/Users/mz/SeeStar/IC 434_sub/process/working.fit")
        )

        self.assertEqual(target_type, pipeline_module.TargetType.EMISSION_NEBULA)

    def test_cosmic_clarity_native_uses_device_auto_by_default(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/CosmicClarity_Native.py")

        with patch.dict(os.environ, {}, clear=False):
            processor._run_cosmic_clarity_native_denoise_fallback("最终降噪")

        native_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Native.py"
        ]
        self.assertTrue(native_calls)
        self.assertNotIn("--cpu", native_calls[-1][2])

    def test_cosmic_clarity_native_can_force_cpu(self):
        processor = self._new_processor()
        processor.available_scripts.add("processing/CosmicClarity_Native.py")

        with patch.dict(os.environ, {"STARUN_COSMIC_NATIVE_GPU": "0"}, clear=False):
            processor._run_cosmic_clarity_native_denoise_fallback("最终降噪")

        native_calls = [
            call for call in processor.script_calls if call[1] == "CosmicClarity_Native.py"
        ]
        self.assertTrue(native_calls)
        self.assertIn("--cpu", native_calls[-1][2])

    def test_pipeline_status_stage2_view_review_is_review_required(self):
        probe = SimpleNamespace(
            results=[],
            _stage2_view_review_required=True,
            _stage9_stars_required=False,
            _stage9_stars_applied=False,
        )

        status = pipeline_module.StarunPostProcessor._pipeline_result_status(
            probe
        )

        self.assertEqual(status, "review_required")

    def test_pipeline_status_color_review_is_review_required(self):
        probe = SimpleNamespace(
            results=[],
            _stage4_color_review_required=True,
            _stage9_stars_required=False,
            _stage9_stars_applied=False,
        )

        status = pipeline_module.StarunPostProcessor._pipeline_result_status(
            probe
        )

        self.assertEqual(status, "review_required")

    def test_pipeline_status_stage9_review_candidate_is_review_required(self):
        probe = SimpleNamespace(
            results=[],
            _stage9_stars_required=True,
            _stage9_stars_applied=True,
            _stage9_review_candidate_selected=True,
            _stage9_remix_formally_accepted=False,
        )

        status = pipeline_module.StarunPostProcessor._pipeline_result_status(
            probe
        )

        self.assertEqual(status, "review_required")

    def test_result_output_basename_uses_template_when_headers_are_complete(self):
        processor = self._new_processor()
        processor.header_metadata.update(
            {
                "OBJECT": "NGC7000",
                "STACKCNT": 120,
                "EXPTIME": 10.0,
                "DATE-OBS": "2026-07-15T12:00:00",
            }
        )

        base_name = processor._result_output_basename()

        self.assertEqual(base_name, pipeline_module.RESULT_BASENAME_TEMPLATE)
        self.assertEqual(
            processor.main_output_fit_basename_template,
            pipeline_module.RESULT_BASENAME_TEMPLATE + "_final",
        )

    def test_result_output_basename_uses_partial_metadata_when_stack_count_missing(self):
        processor = self._new_processor()
        processor.header_metadata.update(
            {
                "OBJECT": "M 42",
                "EXPTIME": 60.0,
                "DATE-OBS": "2026-02-16T14:02:34.608000",
            }
        )

        base_name = processor._result_output_basename()

        self.assertEqual(base_name, "M_42_60sec_20260216_140234_processed")
        self.assertEqual(
            processor.main_output_fit_basename_template,
            "M_42_60sec_20260216_140234_processed_final",
        )
        self.assertNotIn("$", base_name)
        self.assertTrue(
            any(
                "通用结果名覆盖" in message
                for level, message in processor.log.events
                if level == "warn"
            )
        )

    def test_result_output_basename_keeps_generic_fallback_without_identity(self):
        processor = self._new_processor()

        base_name = processor._result_output_basename()

        self.assertEqual(base_name, "result_processed")
        self.assertEqual(processor.main_output_fit_basename_template, "result_final")

    def test_cli_subprocess_failure_records_output_tail(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)
        script_path = Path(td.name) / "Starless.py"
        script_path.write_text("# mock\n", encoding="utf-8")

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            return SimpleNamespace(
                returncode=1,
                stdout="\n".join(f"line{i}" for i in range(1, 16)) + "\n",
            )

        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )
        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "去星",
                    "SyQon Starless",
                    script_path,
                )

        self.assertIsNone(used)
        self.assertIn("output_tail=", processor._last_plugin_script_error or "")
        self.assertIn("line4", processor._last_plugin_script_error or "")
        self.assertIn("line15", processor._last_plugin_script_error or "")
        self.assertNotIn("output_tail=line1", processor._last_plugin_script_error or "")

    def test_review_route_artifacts_are_available_as_stage_previews(self):
        processor = pipeline_module.StarunPostProcessor()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)

        expected = {
            6: "stage6_passthrough.fit",
            7: "stage7_review_with_stars.fit",
            8: "stage8_review_with_stars.fit",
        }
        for stage, filename in expected.items():
            self.assertIn(
                processor.process_dir / filename,
                processor._stage_preview_candidates(stage),
            )

    def test_project_env_allowlist_includes_acceleration_and_quality_gates(self):
        allowed = sys.modules["processor_runtime"].PROJECT_ENV_ALLOWED_KEYS
        self.assertIn("STARUN_GRAXPERT_GPU", allowed)
        self.assertIn(
            "STARUN_STAGE7_STARLESS_REPAIR_CHROMA_REDUCTION_MIN",
            allowed,
        )
        self.assertIn(
            "STARUN_STAGE7_STARLESS_REPAIR_CHROMA_DELTA_MIN",
            allowed,
        )
        self.assertIn(
            "STARUN_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX",
            allowed,
        )
        self.assertIn(
            "STARUN_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_TOLERANCE",
            allowed,
        )
        self.assertIn("STARUN_STAGE7_CHROMA_RESCUE_ENABLE", allowed)
        self.assertIn("STARUN_STAGE7_GALAXY_ROI_HALO_GATE_ENABLE", allowed)
        self.assertIn(
            "STARUN_STAGE7_LARGE_GALAXY_HALO_RESIDUE_SCORE_MAX",
            allowed,
        )
        self.assertIn(
            "STARUN_STAGE7_GALAXY_CORE_PRESERVATION_RATIO_MIN",
            allowed,
        )
        self.assertIn(
            "STARUN_STAGE7_GALAXY_CORE_CONTRAST_RATIO_MIN",
            allowed,
        )
        self.assertIn("STARUN_STAGE8_LIMITED_SATURATION_MAX", allowed)
        self.assertIn("STARUN_STAGE8_LIMITED_CORE_EXCLUSION_EXPAND", allowed)
        self.assertIn(
            "STARUN_STAGE8_LIMITED_HALO_TEXTURE_GROWTH_MAX",
            allowed,
        )
        self.assertIn(
            "STARUN_STAGE8_LIMITED_HALO_TEXTURE_DELTA_MAX",
            allowed,
        )
        self.assertIn("STARUN_STAGE9_COMPACT_STARMASK_ENABLE", allowed)
        self.assertIn(
            "STARUN_STAGE9_STARMASK_PRE_STRETCH_COMPACT_ENABLE",
            allowed,
        )

    def test_runtime_env_migrates_legacy_compaction_only_when_new_key_missing(
        self,
    ):
        legacy_key = "STARUN_STAGE9_COMPACT_STARMASK_ENABLE"
        new_key = "STARUN_STAGE9_STARMASK_PRE_STRETCH_COMPACT_ENABLE"
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor._project_env_explicit_keys = frozenset({legacy_key})

        with patch.dict(os.environ, {legacy_key: "1"}, clear=True):
            processor._apply_runtime_env_overrides()

        self.assertTrue(processor.cfg.stage9_compact_starmask_enabled)
        self.assertTrue(
            processor.cfg.stage9_starmask_pre_stretch_compact_enabled
        )

        processor.cfg.stage9_starmask_pre_stretch_compact_enabled = True
        processor._project_env_explicit_keys = frozenset(
            {legacy_key, new_key}
        )
        with patch.dict(
            os.environ,
            {legacy_key: "1", new_key: "0"},
            clear=True,
        ):
            processor._apply_runtime_env_overrides()

        self.assertTrue(processor.cfg.stage9_compact_starmask_enabled)
        self.assertFalse(
            processor.cfg.stage9_starmask_pre_stretch_compact_enabled
        )

    def test_debug_cleanup_keeps_process_evidence_but_removes_exchange_copies(self):
        import zipfile

        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.debug_mode = True

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.work_dir = Path(td.name)
        processor.process_dir = processor.work_dir / "process"
        review_dir = (
            processor.process_dir / "review_bundles" / "stage9_star_remixing"
        )
        review_dir.mkdir(parents=True)
        (processor.process_dir / "starless.fit").write_bytes(b"science")
        (review_dir / "before.png").write_bytes(b"png")
        (review_dir / "review.json").write_text("{}", encoding="utf-8")
        for name in ("sasp_starless_input.fit", "sasp_starmask_input.fit"):
            (processor.work_dir / name).write_bytes(b"exchange")

        processor.cleanup()

        self.assertTrue((processor.process_dir / "starless.fit").is_file())
        self.assertTrue((review_dir / "before.png").is_file())
        self.assertFalse((processor.work_dir / "sasp_starless_input.fit").exists())
        self.assertFalse((processor.work_dir / "sasp_starmask_input.fit").exists())
        with zipfile.ZipFile(processor.work_dir / "starun_diagnostics.zip") as archive:
            self.assertNotIn(
                "process/review_bundles/stage9_star_remixing/before.png",
                set(archive.namelist()),
            )

    def test_checkpoint_mode_overrides_debug_retention_matrix(self):
        cases = (
            (False, False, False, False),
            (False, True, True, True),
            (True, False, True, False),
            (True, True, True, False),
        )
        for checkpoint_mode, debug_mode, keep_stage1, keep_scratch in cases:
            with self.subTest(
                checkpoint_mode=checkpoint_mode,
                debug_mode=debug_mode,
            ), tempfile.TemporaryDirectory() as td:
                processor = pipeline_module.StarunPostProcessor()
                processor.log = FakeLogger()
                processor.cfg.checkpoint_mode = checkpoint_mode
                processor.cfg.debug_mode = debug_mode
                processor.work_dir = Path(td)
                processor.process_dir = processor.work_dir / "process"
                processor.process_dir.mkdir()
                processor._checkpoint_compaction_preflight = (  # type: ignore[method-assign]
                    lambda: (True, "verified final delivery")
                )

                stage1 = processor.process_dir / "stage1_prepared.fit"
                scratch = processor.process_dir / "stage8_candidate.fit"
                report = processor.process_dir / "stage8_report.json"
                runtime_log = processor.process_dir / "pipeline.log"
                ui_preview = processor.process_dir / "ui_preview" / "latest.png"
                ui_preview.parent.mkdir()
                for path in (stage1, scratch):
                    path.write_bytes(b"fits")
                report.write_text("{}", encoding="utf-8")
                runtime_log.write_text("log", encoding="utf-8")
                ui_preview.write_bytes(b"preview")

                processor.cleanup()

                self.assertEqual(stage1.exists(), keep_stage1)
                self.assertEqual(scratch.exists(), keep_scratch)
                if checkpoint_mode:
                    self.assertTrue(report.is_file())
                    self.assertTrue(runtime_log.is_file())
                    self.assertTrue(ui_preview.is_file())
                    self.assertTrue(
                        (processor.work_dir / "starun_diagnostics.zip").is_file()
                    )
                    self.assertTrue(processor._checkpoint_retention_report["applied"])

    def test_checkpoint_mode_task_run_keeps_only_task_level_checkpoints(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.checkpoint_mode = True
        processor.cfg.debug_mode = True

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.work_dir = Path(td.name)
        processor.process_dir = processor.work_dir / "process"
        processor.process_dir.mkdir()
        task_checkpoint = (
            processor.work_dir
            / "checkpoints"
            / "stage5"
            / "stage5_linear.fit"
        )
        task_checkpoint.parent.mkdir(parents=True)
        task_checkpoint.write_bytes(b"formal-checkpoint")
        process_checkpoints = [
            processor.process_dir / "stage1_prepared.fit",
            processor.process_dir / "stage2_corrected.fit",
            processor.process_dir / "stage5_linear.fit",
        ]
        for path in process_checkpoints:
            path.write_bytes(b"process-copy")
        processor._task_run_manifest_payload = {
            "task_directory": str(processor.work_dir),
        }
        processor._checkpoint_compaction_preflight = (  # type: ignore[method-assign]
            lambda: (True, "verified formal checkpoints")
        )

        processor.cleanup()

        self.assertTrue(task_checkpoint.is_file())
        self.assertTrue(all(not path.exists() for path in process_checkpoints))
        self.assertTrue(processor._checkpoint_retention_report["task_managed"])
        self.assertEqual(
            processor._checkpoint_retention_report["task_checkpoint_stages"],
            [1, 2, 5],
        )

    def test_warning_cleanup_keeps_only_compact_problem_stage_review(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.cfg.debug_mode = False
        processor.results = [
            pipeline_module.StageResult(
                "阶段 9: 星点层处理与重新合成",
                "degraded",
                reason_code="review_required",
            )
        ]

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.work_dir = Path(td.name)
        processor.process_dir = processor.work_dir / "process"
        review_dir = (
            processor.process_dir / "review_bundles" / "stage9_star_remixing"
        )
        review_dir.mkdir(parents=True)
        for name in (
            "before.png",
            "after.png",
            "difference.png",
            "signed_difference.png",
            "review.png",
        ):
            (review_dir / name).write_bytes(name.encode("utf-8"))
        (review_dir / "review.json").write_text("{}", encoding="utf-8")

        processor.cleanup()

        self.assertEqual(
            {path.name for path in review_dir.glob("*.png")},
            {"review.png"},
        )
        report = json.loads((review_dir / "review.json").read_text(encoding="utf-8"))
        self.assertEqual(
            report["retention"]["mode"],
            "compact_problem_evidence",
        )

    def test_runtime_env_clamps_stage2_center_protect_area_ratio(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()

        with patch.dict(
            os.environ,
            {"STARUN_STAGE2_CENTER_PROTECT_AREA_RATIO": "0.82"},
            clear=False,
        ):
            processor._apply_runtime_env_overrides()
        self.assertEqual(processor.cfg.stage2_center_protect_area_ratio, 0.82)

        with patch.dict(
            os.environ,
            {"STARUN_STAGE2_CENTER_PROTECT_AREA_RATIO": "1.20"},
            clear=False,
        ):
            processor._apply_runtime_env_overrides()
        self.assertEqual(processor.cfg.stage2_center_protect_area_ratio, 0.95)

    def test_runtime_env_routes_and_clamps_stage4_auto_reference(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()

        with patch.dict(
            os.environ,
            {
                "STARUN_STAGE4_OFFLINE_FALLBACK_MODE": "preserve",
                "STARUN_STAGE4_AUTO_REFERENCE_GLOBAL_WHITE_ENABLE": "1",
                "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_SAMPLE_TARGET": "999",
                "STARUN_STAGE4_AUTO_REFERENCE_BACKGROUND_SAMPLE_MIN": "1",
                "STARUN_STAGE4_AUTO_REFERENCE_HOLDOUT_RATIO": "0.99",
                "STARUN_STAGE4_AUTO_REFERENCE_GAIN_LIMIT": "4.0",
                "STARUN_STAGE4_AUTO_REFERENCE_GRADIENT_GROWTH_MAX": "0.2",
                "STARUN_STAGE4_AUTO_REFERENCE_TARGET_CHROMA_DRIFT_MAX": "3.0",
                "STARUN_STAGE4_SPCC_ONLINE_UNVERIFIED_TIMEOUT_SEC": "999",
            },
            clear=True,
        ):
            processor._apply_runtime_env_overrides()

        self.assertEqual(processor.cfg.stage4_offline_fallback_mode, "preserve")
        self.assertTrue(processor.cfg.stage4_auto_reference_global_white_enabled)
        self.assertEqual(
            processor.cfg.stage4_auto_reference_background_sample_target,
            64,
        )
        self.assertEqual(
            processor.cfg.stage4_auto_reference_background_sample_min,
            16,
        )
        self.assertEqual(processor.cfg.stage4_auto_reference_holdout_ratio, 0.40)
        self.assertEqual(processor.cfg.stage4_auto_reference_gain_limit, 1.20)
        self.assertEqual(
            processor.cfg.stage4_auto_reference_gradient_growth_max,
            1.0,
        )
        self.assertEqual(
            processor.cfg.stage4_auto_reference_target_chroma_drift_max,
            0.50,
        )
        self.assertEqual(
            processor.cfg.stage4_spcc_online_unverified_timeout_sec,
            180,
        )

    def test_runtime_env_uses_canonical_stage5_denoise_mod(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()

        with patch.dict(
            os.environ,
            {"STARUN_DENOISE_MOD": "0.31"},
            clear=True,
        ):
            processor._apply_runtime_env_overrides()

        self.assertEqual(processor.cfg.denoise_mod, 0.31)
        self.assertFalse(hasattr(processor.cfg, "stage5_builtin_denoise_mod"))

    def test_removed_stage5_denoise_env_alias_is_ignored(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        original = processor.cfg.denoise_mod

        with patch.dict(
            os.environ,
            {"STARUN_STAGE5_BUILTIN_DENOISE_MOD": "0.29"},
            clear=True,
        ):
            processor._apply_runtime_env_overrides()

        self.assertEqual(processor.cfg.denoise_mod, original)

    def test_runtime_env_applies_adaptive_star_and_target_local_controls(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        overrides = {
            "STARUN_STAGE9_STARMASK_ADAPTIVE_STRETCH_ENABLE": "0",
            "STARUN_STAGE9_SOURCE_STAR_DETAIL_PERCENTILE": "98.2",
            "STARUN_STAGE9_SOURCE_COMPONENT_DENSITY_MAX": "2300",
            "STARUN_STAGE9_SOURCE_SINGLE_PIXEL_RATIO_MAX": "0.36",
            "STARUN_STAGE9_STARMASK_ASINH_STRETCH_MAX": "640",
            "STARUN_STAGE9_STARMASK_FAINT_TARGET": "0.19",
            "STARUN_STAGE9_STARMASK_MID_TARGET": "0.48",
            "STARUN_STAGE9_STARMASK_BRIGHT_TARGET": "0.73",
            "STARUN_STAGE9_STARMASK_PEAK_TARGET": "0.79",
            "STARUN_STAGE9_STARMASK_CHROMA_REGULARIZATION_ENABLE": "0",
            "STARUN_STAGE9_STARMASK_FAINT_CHROMA_MAX": "0.32",
            "STARUN_STAGE9_STARMASK_BRIGHT_CHROMA_MAX": "0.58",
            "STARUN_STAGE9_STARMASK_PREDICTED_CHANGE_RATIO_MAX": "0.27",
            "STARUN_STAGE9_STAR_REFERENCE_SIGMA": "5.5",
            "STARUN_STAGE9_COMPACT_WEAK_STAR_RETENTION_MIN": "0.83",
            "STARUN_STAGE9_MIXED_STAR_PEAK_RATIO_MIN": "4.5",
            "STARUN_STAGE9_MIXED_STAR_WEAK_COUNT_MIN": "24",
            "STARUN_STAGE9_MIXED_STAR_BRIGHT_COUNT_MIN": "4",
            "STARUN_STAGE9_WEAK_STAR_RECOVERY_RATIO_MIN": "0.72",
            "STARUN_STAGE9_STAR_RECOVERY_RATIO_MIN": "0.78",
            "STARUN_STAGE9_WEAK_STAR_SCREEN_INTENSITY_MIN": "0.96",
            "STARUN_STAGE9_STAR_SUPPORT_RATIO_MAX": "0.11",
            "STARUN_STAGE9_UNMATCHED_CHANGED_RATIO_MAX": "0.008",
            "STARUN_STAGE9_CHROMATIC_ADDITION_PEAK_MIN": "0.025",
            "STARUN_STAGE9_CHROMATIC_ADDITION_SATURATION_MIN": "0.74",
            "STARUN_STAGE9_CHROMATIC_ADDITION_RATIO_MAX": "0.0025",
            "STARUN_STAGE9_STAR_POSITIVE_DELTA_WINDOW_RECOVERY_RATIO_MIN": "0.76",
            "STARUN_STAGE9_STAR_WING_RECOVERY_RATIO_MIN": "0.66",
            "STARUN_STAGE9_RESIDUAL_DARK_HOLE_RATIO_MAX": "0.14",
            "STARUN_STAGE9_HOLLOW_STRUCTURE_DELTA_MIN": "0.06",
            "STARUN_STAGE9_NEW_HOLLOW_STRUCTURE_AREA_MAX": "48",
            "STARUN_STAGE9_LOCAL_COMPONENT_PEAK_MIN": "0.012",
            "STARUN_STAGE9_LOCAL_COMPONENT_AREA_MAX": "320",
            "STARUN_STAGE9_LOCAL_COMPONENT_ASPECT_RATIO_MAX": "4.0",
            "STARUN_STAGE9_LOCAL_COMPONENT_FILL_RATIO_MIN": "0.12",
            "STARUN_STAGE9_LOCAL_SINGLE_PIXEL_RATIO_MAX": "0.18",
            "STARUN_STAGE9_LOCAL_CYAN_BLUE_PEAK_MIN": "0.015",
            "STARUN_STAGE9_LOCAL_CYAN_BLUE_SATURATION_MIN": "0.55",
            "STARUN_STAGE9_LOCAL_CYAN_BLUE_COMPONENT_AREA_MAX": "72",
            "STARUN_STAGE9_CORE_PERCENTILE": "92",
            "STARUN_STAGE9_CORE_COLOR_JUMP_MIN": "0.11",
            "STARUN_STAGE9_CORE_COLOR_JUMP_COMPONENT_AREA_MAX": "80",
            "STARUN_STAGE10_FINAL_DENOISE_STRENGTH": "0.31",
            "STARUN_STAGE10_STAR_PROTECTION_COVERAGE_MAX": "0.42",
            "STARUN_STAGE10_LARGE_GALAXY_LOCAL_PATCH_VARIANCE_MAX": "0.00036",
            "STARUN_STAGE10_STAGE9_LOCAL_COLOR_RISK_STRENGTH": "0.8",
            "STARUN_FORCE_REVIEW_ONLY_OUTPUT": "1",
            "STARUN_STAGE7_TARGET_LOCAL_METRICS_ENABLE": "0",
            "STARUN_STAGE7_LOCAL_CORE_CLIP_RATIO_MAX": "0.08",
            "STARUN_STAGE7_LOCAL_FAINT_SNR_MIN": "0.31",
            "STARUN_STAGE7_LOCAL_DARK_SEPARATION_MIN": "0.002",
            "STARUN_STAGE7_STRETCH_FEEDBACK_RETRY_MAX": "4",
            "STARUN_STAGE7_STARLESS_STRUCTURE_GATE_ENABLE": "0",
            "STARUN_STAGE7_STARLESS_MASKED_RANK_DRIFT_P95_MAX": "0.16",
            "STARUN_STAGE7_STARLESS_HALO_DETAIL_GROWTH_RATIO_MAX": "1.45",
            "STARUN_STAGE7_STARLESS_HALO_DETAIL_DELTA_MIN": "0.008",
            "STARUN_STAGE7_QUANTILE_FALLBACK_ENABLE": "0",
            "STARUN_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_MAX": "0.045",
            "STARUN_STAGE7_STRETCH_CHROMA_LOAD_LOW_ABSOLUTE_TOLERANCE": "0.0008",
            "STARUN_STAGE7_PREVIEW_TARGET_P50_MAX_RATIO": "1.42",
            "STARUN_STAGE7_BRIGHT_NEBULA_STAR_MASK_EXPAND": "12",
            "STARUN_STAGE7_BRIGHT_NEBULA_STAR_FAINT_SUPPRESSION": "1.4",
            "STARUN_STAGE7_BRIGHT_NEBULA_STAR_DETAIL_SUPPRESSION": "0.72",
            "STARUN_STAGE7_STARLESS_PEAK_BACKGROUND_RATIO_MIN": "5.0",
            "STARUN_STAGE7_STARMASK_DIFFUSE_RESIDUAL_RATIO_MAX": "0.07",
            "STARUN_STAGE7_GALAXY_ROI_HALO_GATE_ENABLE": "0",
            "STARUN_STAGE7_GALAXY_CORE_PRESERVATION_RATIO_MIN": "1.20",
            "STARUN_STAGE7_GALAXY_CORE_CONTRAST_RATIO_MIN": "0.20",
        }

        with patch.dict(os.environ, overrides, clear=False):
            processor._apply_runtime_env_overrides()

        self.assertFalse(processor.cfg.stage9_starmask_adaptive_stretch_enabled)
        self.assertEqual(processor.cfg.stage9_source_star_detail_percentile, 98.2)
        self.assertEqual(processor.cfg.stage9_source_component_density_max, 2300.0)
        self.assertEqual(processor.cfg.stage9_source_single_pixel_ratio_max, 0.36)
        self.assertEqual(processor.cfg.stage9_starmask_asinh_stretch_max, 640.0)
        self.assertEqual(processor.cfg.stage9_starmask_faint_target, 0.19)
        self.assertEqual(processor.cfg.stage9_starmask_mid_target, 0.48)
        self.assertEqual(processor.cfg.stage9_starmask_bright_target, 0.73)
        self.assertEqual(processor.cfg.stage9_starmask_peak_target, 0.79)
        self.assertFalse(processor.cfg.stage9_starmask_chroma_regularization_enabled)
        self.assertEqual(processor.cfg.stage9_starmask_faint_chroma_max, 0.32)
        self.assertEqual(processor.cfg.stage9_starmask_bright_chroma_max, 0.58)
        self.assertEqual(
            processor.cfg.stage9_starmask_predicted_change_ratio_max,
            0.27,
        )
        self.assertEqual(processor.cfg.stage9_star_reference_sigma, 5.5)
        self.assertEqual(
            processor.cfg.stage9_compact_weak_star_retention_min,
            0.83,
        )
        self.assertEqual(processor.cfg.stage9_mixed_star_peak_ratio_min, 4.5)
        self.assertEqual(processor.cfg.stage9_mixed_star_weak_count_min, 24)
        self.assertEqual(processor.cfg.stage9_mixed_star_bright_count_min, 4)
        self.assertEqual(processor.cfg.stage9_weak_star_recovery_ratio_min, 0.72)
        self.assertEqual(processor.cfg.stage9_star_recovery_ratio_min, 0.78)
        self.assertEqual(processor.cfg.stage9_weak_star_screen_intensity_min, 0.96)
        self.assertEqual(processor.cfg.stage9_star_support_ratio_max, 0.11)
        self.assertEqual(processor.cfg.stage9_unmatched_changed_ratio_max, 0.008)
        self.assertEqual(processor.cfg.stage9_chromatic_addition_peak_min, 0.025)
        self.assertEqual(
            processor.cfg.stage9_chromatic_addition_saturation_min,
            0.74,
        )
        self.assertEqual(processor.cfg.stage9_chromatic_addition_ratio_max, 0.0025)
        self.assertEqual(
            processor.cfg.stage9_star_positive_delta_window_recovery_ratio_min,
            0.76,
        )
        self.assertEqual(processor.cfg.stage9_star_wing_recovery_ratio_min, 0.66)
        self.assertEqual(processor.cfg.stage9_residual_dark_hole_ratio_max, 0.14)
        self.assertEqual(processor.cfg.stage9_hollow_structure_delta_min, 0.06)
        self.assertEqual(processor.cfg.stage9_new_hollow_structure_area_max, 48.0)
        self.assertEqual(processor.cfg.stage9_local_component_peak_min, 0.012)
        self.assertEqual(processor.cfg.stage9_local_component_area_max, 320.0)
        self.assertEqual(processor.cfg.stage9_local_component_aspect_ratio_max, 4.0)
        self.assertEqual(processor.cfg.stage9_local_component_fill_ratio_min, 0.12)
        self.assertEqual(processor.cfg.stage9_local_single_pixel_ratio_max, 0.18)
        self.assertEqual(processor.cfg.stage9_local_cyan_blue_peak_min, 0.015)
        self.assertEqual(processor.cfg.stage9_local_cyan_blue_saturation_min, 0.55)
        self.assertEqual(
            processor.cfg.stage9_local_cyan_blue_component_area_max,
            72.0,
        )
        self.assertEqual(processor.cfg.stage9_core_percentile, 92.0)
        self.assertEqual(processor.cfg.stage9_core_color_jump_min, 0.11)
        self.assertEqual(
            processor.cfg.stage9_core_color_jump_component_area_max,
            80.0,
        )
        self.assertEqual(processor.cfg.stage10_final_denoise_strength, 0.31)
        self.assertEqual(
            processor.cfg.stage10_star_protection_coverage_max,
            0.42,
        )
        self.assertEqual(
            processor.cfg.stage10_large_galaxy_local_patch_variance_max,
            0.00036,
        )
        self.assertEqual(
            processor.cfg.stage10_stage9_local_color_risk_strength,
            0.8,
        )
        self.assertTrue(processor.cfg.force_review_only_output)
        self.assertFalse(processor.cfg.stage7_target_local_metrics_enabled)
        self.assertEqual(processor.cfg.stage7_local_core_clip_ratio_max, 0.08)
        self.assertEqual(processor.cfg.stage7_local_faint_snr_min, 0.31)
        self.assertEqual(processor.cfg.stage7_local_dark_separation_min, 0.002)
        self.assertEqual(processor.cfg.stage7_stretch_feedback_retry_max, 1)
        self.assertFalse(processor.cfg.stage7_starless_structure_gate_enabled)
        self.assertEqual(
            processor.cfg.stage7_starless_masked_rank_drift_p95_max,
            0.16,
        )
        self.assertEqual(
            processor.cfg.stage7_starless_halo_detail_growth_ratio_max,
            1.45,
        )
        self.assertEqual(processor.cfg.stage7_starless_halo_detail_delta_min, 0.008)
        self.assertFalse(processor.cfg.stage7_quantile_fallback_enabled)
        self.assertEqual(
            processor.cfg.stage7_stretch_chroma_load_low_absolute_max,
            0.045,
        )
        self.assertEqual(
            processor.cfg.stage7_stretch_chroma_load_low_absolute_tolerance,
            0.0008,
        )
        self.assertEqual(
            processor.cfg.stage7_preview_target_p50_max_ratio,
            1.42,
        )
        self.assertEqual(
            processor.cfg.stage7_bright_nebula_star_mask_expand,
            8,
        )
        self.assertEqual(
            processor.cfg.stage7_bright_nebula_star_faint_suppression,
            1.0,
        )
        self.assertEqual(
            processor.cfg.stage7_bright_nebula_star_detail_suppression,
            0.60,
        )
        self.assertEqual(
            processor.cfg.stage7_starmask_diffuse_residual_ratio_max,
            0.07,
        )
        self.assertEqual(
            processor.cfg.stage7_starless_peak_background_ratio_min,
            5.0,
        )
        self.assertFalse(processor.cfg.stage7_galaxy_roi_halo_gate_enabled)
        self.assertEqual(
            processor.cfg.stage7_galaxy_core_preservation_ratio_min,
            0.95,
        )
        self.assertEqual(
            processor.cfg.stage7_galaxy_core_contrast_ratio_min,
            0.30,
        )

    def test_cosmic_clarity_wrapper_uses_stable_python_when_siril_env_is_boolean(self):
        wrapper = REPO_ROOT / "resources" / "siril_plugins" / "bin" / "CosmicClarity"

        proc = subprocess.run(
            [str(wrapper), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "SIRIL_PYTHON_CLI": "1",
                "STARUN_SIRIL_PYTHON_CLI": sys.executable,
            },
        )

        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("Bundled Cosmic Clarity classic wrapper", proc.stdout)

    def test_legacy_cosmic_clarity_wrapper_is_rejected(self):
        processor = pipeline_module.StarunPostProcessor()
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        plugin_root = Path(td.name) / "siril_plugins"
        bin_dir = plugin_root / "bin"
        bin_dir.mkdir(parents=True)
        wrapper = bin_dir / "CosmicClarity"
        wrapper.write_text(
            "#!/bin/sh\n"
            '"exec" "${SIRIL_PYTHON_CLI:-python3}" "$0" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        processor.siril_plugin_dir = plugin_root

        reason = processor._classic_cosmic_clarity_candidate_error(wrapper)

        self.assertIsNotNone(reason)
        self.assertIn("boolean SIRIL_PYTHON_CLI", reason or "")

    def test_run_linear_resume_without_manifest_is_rejected(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        linear_path = work_dir / "result_linear.fit"
        linear_path.write_bytes(b"linear-fit")

        calls: list[str] = []

        processor.connect = lambda: setattr(processor, "work_dir", work_dir)  # type: ignore[method-assign]
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: None,
        )

        def _prepare_linear_resume() -> None:
            calls.append("prepare_linear_resume")
            processor._stage1_input_mode = "linear_resume"
            processor.linear_intermediate_path = linear_path

        processor._prepare_linear_resume_input = _prepare_linear_resume  # type: ignore[method-assign]
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")  # type: ignore[method-assign]
        processor.stage1_preparation = lambda: calls.append("stage1")  # type: ignore[method-assign]
        processor.stage2_view_correction = lambda: calls.append("stage2")  # type: ignore[method-assign]
        processor.stage3_background_extraction = lambda: calls.append("stage3")  # type: ignore[method-assign]
        processor.stage4_color_calibration = lambda: calls.append("stage4")  # type: ignore[method-assign]
        processor.stage5_linear_denoise = lambda: calls.append("stage5")  # type: ignore[method-assign]
        processor.stage6_star_separation = lambda: calls.append("stage6")  # type: ignore[method-assign]
        processor.stage7_stretching = lambda: calls.append("stage7")  # type: ignore[method-assign]
        processor.stage8_nebula_enhancement = lambda: calls.append("stage8")  # type: ignore[method-assign]
        processor.stage9_star_remixing = lambda: calls.append("stage9")  # type: ignore[method-assign]
        processor.stage10_export = lambda: calls.append("stage10")  # type: ignore[method-assign]
        processor.cleanup = lambda: calls.append("cleanup")  # type: ignore[method-assign]

        with patch.dict(
            os.environ,
            {pipeline_module.ENV_INPUT_MODE_KEY: pipeline_module.INPUT_MODE_LINEAR_RESUME},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "task-run manifest"):
                processor.run()

    def test_run_stage2_resume_without_manifest_is_rejected(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        stage2_path = work_dir / pipeline_module.STAGE2_CORRECTED_INPUT_NAME
        stage2_path.write_bytes(b"stage2-fit")

        calls: list[str] = []

        processor.connect = lambda: setattr(processor, "work_dir", work_dir)  # type: ignore[method-assign]
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: None,
        )

        def _prepare_stage2_corrected_resume() -> None:
            calls.append("prepare_stage2_corrected_resume")
            processor._stage1_input_mode = "stage2_corrected_resume"
            processor.source_file = stage2_path

        processor._prepare_stage2_corrected_resume_input = _prepare_stage2_corrected_resume  # type: ignore[method-assign]
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")  # type: ignore[method-assign]
        processor.stage1_preparation = lambda: calls.append("stage1")  # type: ignore[method-assign]
        processor.stage2_view_correction = lambda: calls.append("stage2")  # type: ignore[method-assign]
        processor.stage3_background_extraction = lambda: calls.append("stage3")  # type: ignore[method-assign]
        processor.stage4_color_calibration = lambda: calls.append("stage4")  # type: ignore[method-assign]
        processor.stage5_linear_denoise = lambda: calls.append("stage5")  # type: ignore[method-assign]
        processor.stage6_star_separation = lambda: calls.append("stage6")  # type: ignore[method-assign]
        processor.stage7_stretching = lambda: calls.append("stage7")  # type: ignore[method-assign]
        processor.stage8_nebula_enhancement = lambda: calls.append("stage8")  # type: ignore[method-assign]
        processor.stage9_star_remixing = lambda: calls.append("stage9")  # type: ignore[method-assign]
        processor.stage10_export = lambda: calls.append("stage10")  # type: ignore[method-assign]
        processor.cleanup = lambda: calls.append("cleanup")  # type: ignore[method-assign]

        with patch.dict(
            os.environ,
            {pipeline_module.ENV_INPUT_MODE_KEY: pipeline_module.INPUT_MODE_STAGE2_CORRECTED_RESUME},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "task-run manifest"):
                processor.run()

    def test_run_reraises_generic_stage_failure_for_siril_and_gui(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.initial_cfg.checkpoint_mode = True
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        work_dir = Path(td.name)
        calls: list[str] = []

        def connect() -> None:
            processor.work_dir = work_dir
            processor._siril_ever_connected = True

        processor.connect = connect
        processor.siril = SimpleNamespace(
            cmd=lambda *_args, **_kwargs: None,
            disconnect=lambda: calls.append("disconnect"),
        )
        processor.stage1_preparation = lambda: calls.append("stage1")
        processor._auto_tune_for_current_input = lambda: calls.append("auto_tune")

        def fail_stage2() -> None:
            calls.append("stage2")
            raise RuntimeError("simulated stage2 failure")

        processor.stage2_view_correction = fail_stage2
        processor.stage3_background_extraction = lambda: calls.append("stage3")

        with patch.dict(
            os.environ,
            {pipeline_module.ENV_INPUT_MODE_KEY: pipeline_module.INPUT_MODE_AUTO},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated stage2 failure"):
                processor.run()

        self.assertEqual(calls, ["stage1", "auto_tune", "stage2", "disconnect"])
        self.assertTrue(
            any(
                level == "error" and "程序中断: simulated stage2 failure" in message
                for level, message in processor.log.events
            )
        )
        retention = json.loads(
            (work_dir / "checkpoint-retention.json").read_text(encoding="utf-8")
        )
        self.assertFalse(retention["applied"])
        self.assertEqual(retention["status"], "preserved")
        self.assertEqual(retention["reason"], "simulated stage2 failure")

    def test_plugin_script_prereq_check_skips_runtime_execution_when_modules_missing(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        def _unexpected_cmd(*_args, **_kwargs):
            raise AssertionError("pyscript should be skipped before runtime call")

        processor.cmd_with_check = _unexpected_cmd  # type: ignore[method-assign]
        script_path = Path("/tmp/CosmicClarity_Denoise.py")

        used = processor._run_plugin_script_by_path(
            "最终降噪",
            "CosmicClarity Denoise",
            script_path,
            args=("-denoising_mode", "luminance"),
        )

        self.assertIsNone(used)
        self.assertIsNotNone(processor._last_plugin_script_error)
        self.assertIn("missing python modules", processor._last_plugin_script_error)

    def test_cli_subprocess_ignores_boolean_siril_python_cli_env(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "CosmicClarity_Denoise.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")

        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        captured_cmd: dict[str, list[str]] = {}
        captured_env: dict[str, str] = {}

        def _fake_run(cmd: list[str], **kwargs: Any):
            captured_cmd["value"] = list(cmd)
            proc_env = kwargs.get("env") or {}
            if isinstance(proc_env, dict):
                captured_env.update({str(k): str(v) for k, v in proc_env.items()})
            return SimpleNamespace(returncode=0, stdout="")

        contaminated_env = {
            "SIRIL_PYTHON_CLI": "1",
            "STARUN_SIRIL_PYTHON_CLI": "1",
            "QT_PLUGIN_PATH": "/app/PySide6/Qt/plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/app/PySide6/Qt/plugins/platforms",
            "QML2_IMPORT_PATH": "/app/PySide6/Qt/qml",
            "QT_QPA_PLATFORM": "cocoa",
        }
        with patch.dict(os.environ, contaminated_env, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "最终降噪",
                    "CosmicClarity Denoise",
                    script_path,
                    args=("-denoising_mode", "luminance"),
                )

        self.assertIsNotNone(used)
        self.assertIn("value", captured_cmd)
        self.assertNotEqual(captured_cmd["value"][0], "1")
        self.assertIn(str(script_path), captured_cmd["value"])
        self.assertEqual(captured_env.get("STARUN_SIRILPY_TIMEOUT_SEC"), "300")
        self.assertNotEqual(captured_env.get("SIRIL_PYTHON_CLI"), "1")
        self.assertEqual(captured_env.get("QT_QPA_PLATFORM"), "offscreen")
        self.assertNotIn("QT_PLUGIN_PATH", captured_env)
        self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", captured_env)
        self.assertNotIn("QML2_IMPORT_PATH", captured_env)

    def test_cli_subprocess_prefers_stable_starun_python_cli_env(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "Starless.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")
        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        captured_cmd: dict[str, list[str]] = {}

        def _fake_run(cmd: list[str], **_kwargs: Any):
            captured_cmd["value"] = list(cmd)
            return SimpleNamespace(returncode=0, stdout="")

        with patch.dict(
            os.environ,
            {
                "STARUN_SIRIL_PYTHON_CLI": sys.executable,
                "SIRIL_PYTHON_CLI": "/usr/bin/false",
            },
            clear=False,
        ):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "去星",
                    "SyQon Starless",
                    script_path,
                )

        self.assertIsNotNone(used)
        self.assertEqual(captured_cmd["value"][0], sys.executable)

    def test_cli_subprocess_releases_parent_siril_connection(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "Starless.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")

        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        events: list[str] = []

        class _ConnectedSiril:
            connected = True

            def disconnect(self) -> None:
                events.append("disconnect")
                self.connected = False

            def connect(self) -> bool:
                events.append("connect")
                self.connected = True
                return True

        processor.siril = _ConnectedSiril()

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            events.append("run")
            return SimpleNamespace(returncode=0, stdout="")

        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "去星",
                    "SyQon Starless",
                    script_path,
                    args=("--tile-size", "512"),
                )

        self.assertIsNotNone(used)
        self.assertEqual(events, ["disconnect", "run", "connect"])
        self.assertTrue(processor.siril.connected)

    def test_cli_subprocess_file_mode_keeps_parent_siril_connection(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)
        script_path = Path(td.name) / "Starless.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")
        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        events: list[str] = []

        class _ConnectedSiril:
            connected = True

            def disconnect(self) -> None:
                events.append("disconnect")
                self.connected = False

            def connect(self) -> bool:
                events.append("connect")
                self.connected = True
                return True

        processor.siril = _ConnectedSiril()

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            events.append("run")
            return SimpleNamespace(returncode=0, stdout="")

        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(pipeline_module.subprocess, "run", _fake_run):
                used = processor._run_plugin_script_cli_subprocess(
                    "去星",
                    "SyQon Starless file worker",
                    script_path,
                    uses_siril_connection=False,
                    verify_image_change=False,
                )

        self.assertIsNotNone(used)
        self.assertEqual(events, ["run"])
        self.assertTrue(processor.siril.connected)

    def test_cli_subprocess_heartbeat_stays_local_while_parent_siril_is_disconnected(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "Starless.py"
        script_path.write_text("print('mock')\n", encoding="utf-8")
        processor._validate_plugin_script_prerequisites = (  # type: ignore[method-assign]
            lambda _path, _python_executable=None: (True, "")
        )

        live_messages: list[str] = []
        connection_events: list[str] = []

        class SirilConnectionError(Exception):
            pass

        class _ConnectedSiril:
            connected = True

            def disconnect(self) -> None:
                connection_events.append("disconnect")
                self.connected = False

            def connect(self) -> bool:
                connection_events.append("connect")
                self.connected = True
                return True

            def log(self, line: str) -> None:
                if not self.connected:
                    raise SirilConnectionError(
                        "Error in _send_command(): [Errno 9] Bad file descriptor"
                    )
                live_messages.append(line)

        processor.siril = pipeline_module._FatalSirilInterfaceProxy(
            processor,
            _ConnectedSiril(),
        )
        processor._siril_ever_connected = True
        processor.log = pipeline_module.PipelineLogger("DEBUG")
        log_path = Path(td.name) / "pipeline.log"
        processor.log.set_file_path(log_path)
        processor.log.set_sink(processor.siril.log)

        class _ImmediateFirstWaitEvent:
            def __init__(self) -> None:
                self._first_wait = True
                self._stopped = False

            def wait(self, _timeout: float) -> bool:
                if self._stopped:
                    return True
                if self._first_wait:
                    self._first_wait = False
                    return False
                return True

            def set(self) -> None:
                self._stopped = True

        class _SynchronousThread:
            def __init__(self, *, target: Any, daemon: bool) -> None:
                _ = daemon
                self._target = target

            def start(self) -> None:
                self._target()

            def join(self, timeout: float | None = None) -> None:
                _ = timeout

        def _fake_run(_cmd: list[str], **_kwargs: Any):
            return SimpleNamespace(returncode=0, stdout="")

        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            with patch.object(
                pipeline_module.plugin_runner.threading,
                "Event",
                _ImmediateFirstWaitEvent,
            ):
                with patch.object(
                    pipeline_module.plugin_runner.threading,
                    "Thread",
                    _SynchronousThread,
                ):
                    with patch.object(pipeline_module.subprocess, "run", _fake_run):
                        used = processor._run_plugin_script_cli_subprocess(
                            "去星",
                            "SyQon Starless",
                            script_path,
                            args=("--tile-size", "512"),
                        )

        self.assertIsNotNone(used)
        self.assertEqual(connection_events, ["disconnect", "connect"])
        self.assertFalse(processor._siril_process_terminated)
        self.assertIn("CLI 子进程仍在运行", log_path.read_text(encoding="utf-8"))
        self.assertFalse(any("CLI 子进程仍在运行" in line for line in live_messages))
        self.assertTrue(any("CLI 子进程成功" in line for line in live_messages))

    def test_cli_subprocess_timeout_applies_when_child_produces_no_output(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.workflow_command_used = {}

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        processor.process_dir = Path(td.name)
        processor.work_dir = Path(td.name)

        script_path = Path(td.name) / "sleeping_plugin.py"
        script_path.write_text(
            "import time\n"
            "time.sleep(2)\n",
            encoding="utf-8",
        )

        started = time.monotonic()
        with patch.dict(os.environ, {"SIRIL_PYTHON_CLI": sys.executable}, clear=False):
            used = processor._run_plugin_script_cli_subprocess(
                "测试脚本",
                "Sleeping Plugin",
                script_path,
                timeout_sec=1,
            )
        elapsed = time.monotonic() - started

        self.assertIsNone(used)
        self.assertLess(elapsed, 1.8)
        self.assertIn("subprocess timeout", processor._last_plugin_script_error or "")

    def test_final_denoise_cli_timeout_tracks_sirilpy_timeout_with_cap(self):
        processor = pipeline_module.StarunPostProcessor()

        with patch.dict(os.environ, {"STARUN_SIRILPY_TIMEOUT_SEC": "300"}, clear=False):
            self.assertEqual(processor._final_denoise_cli_timeout_sec(), 300)
        with patch.dict(os.environ, {"STARUN_SIRILPY_TIMEOUT_SEC": "999"}, clear=False):
            self.assertEqual(processor._final_denoise_cli_timeout_sec(), 300)
        with patch.dict(os.environ, {"STARUN_SIRILPY_TIMEOUT_SEC": "bad"}, clear=False):
            self.assertEqual(processor._final_denoise_cli_timeout_sec(), 300)

    def test_stage_diff_note_separates_pixel_and_header_changes(self):
        from astropy.io import fits

        processor = pipeline_module.StarunPostProcessor()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name) / "process"
        process_dir.mkdir(parents=True, exist_ok=True)
        processor.process_dir = process_dir

        previous_path = process_dir / "stage4_colorbalanced.fit"
        current_path = process_dir / "stage5_denoised.fit"
        pixels = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)

        previous_hdu = fits.PrimaryHDU(pixels)
        previous_hdu.header["DATE"] = "2026-08-08T10:00:00"
        previous_hdu.writeto(previous_path)
        current_hdu = fits.PrimaryHDU(pixels.copy())
        current_hdu.header["DATE"] = "2026-08-08T10:00:00"
        current_hdu.writeto(current_path)

        same_note = processor._stage_diff_note("stage5_denoised", "stage4_colorbalanced")
        self.assertIsNotNone(same_note)
        self.assertIn("像素内容一致", same_note)
        self.assertIn("FITS header 一致", same_note)

        header_only_hdu = fits.PrimaryHDU(pixels.copy())
        header_only_hdu.header["DATE"] = "2026-08-08T10:00:01"
        for index in range(40):
            header_only_hdu.header.add_history(f"header-only audit entry {index:02d}")
        header_only_hdu.writeto(current_path, overwrite=True)
        header_note = processor._stage_diff_note(
            "stage5_denoised",
            "stage4_colorbalanced",
        )
        self.assertIsNotNone(header_note)
        self.assertIn("像素内容一致", header_note)
        self.assertIn("FITS header 有变化", header_note)
        self.assertNotIn("像素内容有变化", header_note)

        changed_pixels = pixels.copy()
        changed_pixels[1, 3, 4] += np.float32(0.25)
        changed_hdu = fits.PrimaryHDU(changed_pixels)
        changed_hdu.header["DATE"] = "2026-08-08T10:00:01"
        changed_hdu.writeto(current_path, overwrite=True)
        diff_note = processor._stage_diff_note("stage5_denoised", "stage4_colorbalanced")
        self.assertIsNotNone(diff_note)
        self.assertIn("像素内容有变化", diff_note)

    def test_stage_diff_note_does_not_treat_unparsed_container_sha_as_pixels(self):
        processor = pipeline_module.StarunPostProcessor()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        process_dir = Path(td.name) / "process"
        process_dir.mkdir(parents=True, exist_ok=True)
        processor.process_dir = process_dir
        (process_dir / "before.fit").write_bytes(b"not-a-fits-before")
        (process_dir / "after.fit").write_bytes(b"not-a-fits-after")

        note = processor._stage_diff_note("after", "before")

        self.assertIsNotNone(note)
        self.assertIn("像素内容无法判定", note)
        self.assertIn("不作为像素变化依据", note)
        self.assertNotIn("像素内容有变化", note)

    def test_stage_result_display_status_uses_structured_fields_only(self):
        message_only_result = pipeline_module.StageResult(
            "阶段 X",
            "ok",
            message="fallback: failed_component=A; fallback_component=B; fallback_status=success",
        )
        skipped_message_result = pipeline_module.StageResult(
            "阶段 Y",
            "ok",
            message="SPCC skipped on Light_ preprocess mode",
        )
        fallback_result = pipeline_module.StageResult(
            "阶段 A",
            "ok",
            fallback_used=True,
        )
        skipped_result = pipeline_module.StageResult(
            "阶段 B",
            "ok",
            execution="skipped",
        )
        passthrough_result = pipeline_module.StageResult(
            "阶段 C",
            "ok",
            execution="safe_passthrough",
        )
        degraded_result = pipeline_module.StageResult("阶段 Z", "degraded")

        self.assertEqual(message_only_result.display_status, "ok")
        self.assertEqual(skipped_message_result.display_status, "ok")
        self.assertEqual(fallback_result.display_status, "ok_with_fallback")
        self.assertEqual(skipped_result.display_status, "ok_skipped_optional")
        self.assertEqual(passthrough_result.display_status, "ok_safe_passthrough")
        self.assertEqual(degraded_result.display_status, "degraded")

    def test_ai_plan_parser_is_removed(self):
        processor = pipeline_module.StarunPostProcessor()
        self.assertFalse(hasattr(processor, "_extract_first_json_object"))

    def test_processing_software_identity_fingerprints_stage3_sources(self):
        processor = pipeline_module.StarunPostProcessor()

        identity = processor._processing_software_identity()
        stage3_identity = identity["stage_algorithms"]["stage3_background"]

        self.assertEqual(identity["schema"], "starun.software-identity.v1")
        self.assertEqual(stage3_identity["algorithm_contract_version"], "1.2.0")
        self.assertEqual(len(stage3_identity["source_sha256"]), 64)
        self.assertEqual(
            set(stage3_identity["source_files"]),
            {
                "stage3_contract.py",
                "background_sampling.py",
                "stages/stage3_background_extraction.py",
            },
        )

    def test_autobge_prerequisites_check_cv2_import_name(self):
        modules = pipeline_module.StarunPostProcessor._SCRIPT_PREREQUISITE_MODULES[
            "AutoBGE.py"
        ]

        self.assertIn("cv2", modules)
        self.assertNotIn("opencv-python", modules)
