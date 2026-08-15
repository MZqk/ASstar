"""Pipeline/plugin fallback tests for stage2 preparation."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage2PreparationTests(PipelinePluginFallbackTestBase):
    def test_stage2_applies_adaptive_edge_crop_when_edge_black_remains_high(self):
        processor = self._new_processor()
        processor.feature_measurements.extend(
            [
                pipeline_module.ImageFeatures(edge_black_ratio=0.19),
                pipeline_module.ImageFeatures(edge_black_ratio=0.05),
                pipeline_module.ImageFeatures(edge_black_ratio=0.05),
            ]
        )
        stage2_module = sys.modules["stages.stage2_view_correction"]

        with (
            patch.object(
                stage2_module,
                "_detect_auto_edge_crop",
                side_effect=[
                    ((10, 10, 980, 980), "initial edge crop"),
                    ((8, 8, 964, 964), "adaptive edge crop"),
                ],
            ),
            patch.object(stage2_module, "_edge_color_artifact_crop", return_value=""),
        ):
            stage2_view_correction(processor)

        crop_calls = [call for call in processor.cmd_calls if call[0] == "crop"]
        self.assertGreaterEqual(len(crop_calls), 2)
        _name, status, _dur, message = processor.results[-1]
        self.assertEqual(status, "ok")
        self.assertIn("adaptive edge crop", message)

    def test_stage2_color_artifact_crop_limit_is_a_fixed_invariant(self):
        stage2_module = sys.modules["stages.stage2_view_correction"]

        self.assertFalse(
            hasattr(
                pipeline_module.PipelineConfig(),
                "stage2_color_artifact_max_crop",
            )
        )
        self.assertEqual(
            stage2_module.STAGE2_COLOR_ARTIFACT_MIN_RETAINED_RATIO,
            0.90,
        )

    def test_stage2_cropper_is_not_a_plugin_prerequisite(self):
        self.assertNotIn(
            "Autocrop.py",
            pipeline_module.StarunPostProcessor._SCRIPT_PREREQUISITE_MODULES,
        )
