"""Stage 9 spatial-scale and scaled-catalog fallback tests."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage9ScaleTests(PipelinePluginFallbackTestBase):
    @staticmethod
    def _scaled_catalog_fixture(*, status: str = "ok", reason: str = ""):
        if status != "ok":
            return {"status": status, "reason": reason}
        return {
            "status": "ok",
            "reference_source": "starmask_only",
            "reference_degraded": True,
            "component_count": 4,
            "_peak_y": np.asarray([20, 35, 50, 65], dtype=np.int32),
            "_peak_x": np.asarray([20, 40, 60, 80], dtype=np.int32),
            "_source_fwhm_px": np.full(4, 4.0, dtype=np.float32),
        }

    def _prepare_scaled_catalog_with(self, side_effect):
        processor = self._new_processor()
        processor.siril.get_image_pixeldata = (
            lambda preview=False: processor.image_pixels.copy()
        )
        (processor.process_dir / "stage5_linear.fit").touch()
        processor._stage9_psf_review_required = False
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        stage5_reference = {
            "stars": [
                {
                    "geometry_valid": True,
                    "saturated": False,
                    "fwhm_geometry": 4.0,
                }
                for _ in range(4)
            ]
        }
        with (
            patch.object(
                stage9_module,
                "_prepare_stage9_matched_domain_context",
                return_value={
                    "available": False,
                    "report": {"reason": "test matched pair unavailable"},
                },
            ),
            patch.object(
                stage9_module,
                "_stage9_stage5_star_reference_report",
                return_value=stage5_reference,
            ),
            patch.object(
                stage9_module.stage9_quality,
                "build_star_reference_catalog",
                side_effect=side_effect,
            ),
        ):
            summary = stage9_module._prepare_stage9_star_reference(
                processor,
                "starmask",
                [],
            )
        return processor, summary

    def test_rejection_preserves_ready_scale_and_uses_strict_catalog(self):
        processor, summary = self._prepare_scaled_catalog_with(
            [
                self._scaled_catalog_fixture(),
                self._scaled_catalog_fixture(
                    status="unavailable",
                    reason="scaled source contamination",
                ),
                self._scaled_catalog_fixture(),
            ]
        )

        self.assertEqual(processor._stage9_spatial_scale["status"], "ready")
        self.assertEqual(summary["status"], "ok")
        self.assertTrue(processor._stage9_star_reference_degraded)
        validation = processor._stage9_scaled_catalog_validation
        self.assertEqual(validation["status"], "degraded")
        self.assertEqual(validation["selected_route"], "strict_starmask_only")

    def test_all_catalog_failures_keep_scale_but_discard_bootstrap(self):
        processor, summary = self._prepare_scaled_catalog_with(
            [
                self._scaled_catalog_fixture(),
                self._scaled_catalog_fixture(
                    status="unavailable",
                    reason="scaled source contamination",
                ),
                self._scaled_catalog_fixture(
                    status="unavailable",
                    reason="strict catalog contamination",
                ),
            ]
        )

        self.assertEqual(processor._stage9_spatial_scale["status"], "ready")
        self.assertEqual(summary["status"], "unavailable")
        self.assertEqual(
            processor._stage9_star_reference_catalog["status"],
            "unavailable",
        )
        validation = processor._stage9_scaled_catalog_validation
        self.assertEqual(validation["status"], "unavailable")
        self.assertEqual(validation["selected_route"], "none")

    def test_scaled_strict_catalog_can_recover_after_bootstrap_failure(self):
        unavailable = self._scaled_catalog_fixture(
            status="unavailable",
            reason="catalog contamination",
        )
        processor, summary = self._prepare_scaled_catalog_with(
            [
                unavailable,
                unavailable,
                unavailable,
                self._scaled_catalog_fixture(),
            ]
        )

        self.assertEqual(processor._stage9_spatial_scale["status"], "ready")
        self.assertEqual(summary["status"], "ok")
        validation = processor._stage9_scaled_catalog_validation
        self.assertEqual(validation["status"], "degraded")
        self.assertEqual(validation["selected_route"], "strict_starmask_only")
