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
        generic_masks = processor._stage8_generate_starless_masks(image)

        masks, report = (
            pipeline_module.stage8_pixels.stage8_large_galaxy_structure_masks(
                processor,
                generic_masks,
            )
        )

        self.assertIsNotNone(masks, report)
        self.assertEqual(report["route"], "large_galaxy_elliptical_soft_v1")
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
            pipeline_module.stage8_pixels.stage8_large_galaxy_structure_masks(
                processor,
                masks,
            )
        )
        outside = routed["enhancement_subject_weight"] <= 0.0
        self.assertTrue(
            np.allclose(enhanced[:, outside], image[:, outside], atol=1e-7)
        )
        self.assertGreater(float(np.max(np.abs(enhanced - image))), 0.0)

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
        self.assertEqual(processor.results[-1][1], "degraded")


if __name__ == "__main__":
    unittest.main()
