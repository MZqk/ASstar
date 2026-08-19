"""Stage 9 formal candidate ordering and persisted-output acceptance tests."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage9AcceptanceTests(PipelinePluginFallbackTestBase):
    def test_formal_ranking_ignores_lower_chromatic_addition(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        base = {
            "accepted": True,
            "metrics": {
                "catalog_star_visibility_ratio_all": 0.95,
                "catalog_star_visibility_ratio_weak": 0.92,
                "catalog_star_visibility_ratio_bright": 1.0,
                "weak_star_recovery_ratio": 0.90,
                "star_recovery_ratio": 0.92,
                "star_positive_delta_window_recovery_ratio": 0.91,
                "star_wing_recovery_ratio": 0.80,
                "highlight_clip_growth": 0.001,
                "bright_pixel_growth": 0.002,
            },
            "psf_closure": {
                "groups": {"all": {"status": "ok", "fwhm_ratio_median": 1.0}}
            },
            "reference_fidelity": {"support_rgb_mae": 0.01},
        }
        low_chroma = copy.deepcopy(base)
        low_chroma["metrics"]["chromatic_star_addition_ratio"] = 0.00001
        high_chroma = copy.deepcopy(base)
        high_chroma["metrics"]["chromatic_star_addition_ratio"] = 0.002

        low_score = stage9_module._stage9_formal_candidate_score(
            low_chroma, support_mode="normal"
        )
        high_score = stage9_module._stage9_formal_candidate_score(
            high_chroma, support_mode="normal"
        )
        self.assertEqual(low_score, high_score)

        lower_visibility = copy.deepcopy(base)
        lower_visibility["metrics"]["catalog_star_visibility_ratio_weak"] = 0.75
        self.assertLess(
            low_score,
            stage9_module._stage9_formal_candidate_score(
                lower_visibility, support_mode="normal"
            ),
        )

    def test_persisted_validation_rejects_frame_mismatch(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        active = np.zeros((3, 12, 16), dtype=np.float32)
        active[:, 5, 7] = 0.5
        persisted = active.copy()
        persisted[:, 2, 3] = 0.1
        state = {"pixels": active}

        class PersistedPipeline:
            cfg = SimpleNamespace(stage9_star_color_post_validation_enabled=False)
            siril = SimpleNamespace(
                get_image_pixeldata=lambda preview=False: state["pixels"].copy()
            )
            _stage9_star_color_reference_samples = None
            _stage9_last_star_layer = None

            @staticmethod
            def cmd_with_check(command, stem):
                self.assertEqual(command, "load")
                if stem == "stage9_remixed":
                    state["pixels"] = persisted

            @staticmethod
            def _stage9_assess_current_remix(_source, *, attempt, formula):
                return {
                    "attempt": attempt,
                    "formula": formula,
                    "status": "ok",
                    "accepted": True,
                    "catalog_visibility": {
                        "available": True,
                        "passed": True,
                        "coordinate_contract": {"validated": True},
                        "groups": {
                            name: {"passed": True}
                            for name in ("all", "weak", "bright")
                        },
                    },
                }

        report = stage9_module._validate_stage9_persisted_output(
            PersistedPipeline(),
            "stage8_enhanced",
            {"attempt": "screen_primary", "formula": "screen"},
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["pixels_match"])
        self.assertIn("persisted_frame_pixels_mismatch", report["failures"])
        self.assertNotEqual(
            report["active_pixel_hash"], report["persisted_pixel_hash"]
        )
