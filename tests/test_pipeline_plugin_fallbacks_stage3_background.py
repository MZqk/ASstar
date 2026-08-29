"""Pipeline/plugin fallback tests for stage3 background."""

import copy
import math

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


def _accepted_stage3_pixel_gate(*_args: Any, **kwargs: Any):
    profile = str(kwargs.get("gate_profile") or "output_first")
    return True, {
        "status": "accepted",
        "accepted": True,
        "severity": "normal",
        "warnings": [],
        "hard_issues": [],
        "issues": [],
        "profile": profile,
        "effective_thresholds": {"profile": profile},
    }


def _stage3_recovery_sample_reports(*, height: int = 512, width: int = 640):
    points = [
        (
            (cell_x + fraction) / 4.0 * (width - 1),
            (cell_y + fraction) / 4.0 * (height - 1),
        )
        for cell_y in range(4)
        for cell_x in range(4)
        for fraction in (0.30, 0.70)
    ]
    sources = [
        "regular_grid" if index % 2 == 0 else "dark_patch_refinement"
        for index in range(len(points))
    ]
    shared = {
        "valid_mask": "applied",
        "saturation_map": "applied",
        "star_catalog": "applied",
    }
    masks = {
        "source_mask_fraction": 0.34,
        "usable_sky_fraction": 0.60,
    }
    mask_evidence = {
        "applied_to_sampling": True,
        "usable_sky_fraction": 0.60,
    }
    thresholds = {
        "brightness_quantile_max": 0.70,
        "texture_quantile_max": 0.55,
    }
    full_sky_support = np.ones((height, width), dtype=bool)
    base = {
        "status": "insufficient_safe_coverage",
        "sample_count": 0,
        "selected_candidate_count": 9,
        "safe_candidate_count": 9,
        "minimum_count": 12,
        "coverage": {
            "quadrants": 4,
            "grid_cells": 7,
            "available_grid_cells": 7,
            "x_span_ratio": 0.90,
            "y_span_ratio": 0.65,
        },
        "masks": masks,
        "mask_evidence": mask_evidence,
        "shared_scene_support": shared,
        "thresholds": thresholds,
        "selected_samples": [],
        "rejection_counts": {"shared_catalog_star": 190},
        "candidate_independent_sky_support": {
            "status": "available",
            "pixel_count": height * width,
            "coverage": 1.0,
        },
        "_candidate_independent_sky_support_mask": full_sky_support,
    }
    refined = {
        **base,
        "status": "ready",
        "sample_count": len(points),
        "selected_candidate_count": len(points),
        "safe_candidate_count": 83,
        "coverage": {
            "quadrants": 4,
            "grid_cells": 16,
            "available_grid_cells": 16,
            "x_span_ratio": 0.90,
            "y_span_ratio": 0.84,
        },
        "selected_candidate_sources": {
            "regular_grid": 16,
            "dark_patch_refinement": 16,
        },
        "selected_samples": [
            {
                "point": list(point),
                "source": source,
                "grid_cell": [0, 0],
            }
            for point, source in zip(points, sources)
        ],
    }
    return points, base, refined


def _stage3_dense_recovery_sample_reports(*, height: int = 512, width: int = 640):
    points, base, refined = _stage3_recovery_sample_reports(
        height=height,
        width=width,
    )
    base = copy.deepcopy(base)
    refined = copy.deepcopy(refined)
    base.update(
        candidate_count=240,
        base_candidate_count=240,
        rejection_counts={"shared_catalog_star": 235, "exclusion_masked": 5},
    )
    base["masks"] = {
        "source_mask_fraction": 0.42805,
        "usable_sky_fraction": 0.5670248,
    }
    base["mask_evidence"] = {
        "applied_to_sampling": True,
        "usable_sky_fraction": 0.3637058,
        "strict_unmasked_sky_fraction": 0.3637058,
        "nonstellar_sky_fraction": 0.5670248,
        "layers": {
            "scene_support_stars": {
                "available": True,
                "applied": True,
                "pixel_fraction": 0.4592845,
                "method": "scene_support_catalog_2_5x_fwhm",
            }
        },
    }
    refined.update(
        masks=copy.deepcopy(base["masks"]),
        mask_evidence={
            **copy.deepcopy(base["mask_evidence"]),
            "usable_sky_fraction": 0.5670248,
            "effective_usable_sky_fraction": 0.5670248,
            "effective_definition": (
                "nonstellar_sky_with_catalog_points_masked_in_sample_statistics"
            ),
            "masked_catalog_statistics": {
                "schema_version": "starun.stage3-dense-star-sampling.v1",
                "minimum_patch_support_fraction": 0.80,
                "support_mask_sha256": "support-sha",
                "catalog_mask_sha256": "catalog-sha",
                "siril_recalculate": False,
            },
        },
        base_candidate_count=400,
        candidate_count=400,
        masked_catalog_statistics=True,
        dense_star_masked_sampling={
            "schema_version": "starun.stage3-dense-star-sampling.v1",
            "status": "ready",
            "minimum_patch_support_fraction": 0.80,
            "selected_support_fraction_min": 0.81,
            "siril_recalculate": False,
        },
        _masked_pixel_support_mask=np.ones((height, width), dtype=bool),
    )
    refined["selected_samples"] = [
        {
            "point": list(point),
            "source": source,
            "grid_cell": [
                min(3, int(point[0] * 4 / width)),
                min(3, int(point[1] * 4 / height)),
            ],
            "sample_size": 25,
            "masked_support_count": 510,
            "masked_support_fraction": 0.816,
            "shared_star_fraction": 0.12,
            "compact_source_fraction": 0.064,
            "point_source_mask_fraction": 0.184,
            "hard_exclusion_fraction": 0.0,
            "channel_count": 1,
            "channel_medians": [0.055],
            "native_luminance_mean": 0.055,
            "native_luminance_min": 0.040,
            "native_luminance_max": 0.070,
        }
        for point, source in zip(
            points,
            (
                "regular_grid" if index % 2 == 0 else "dark_patch_refinement"
                for index in range(len(points))
            ),
        )
    ]
    return points, base, refined


class PipelinePluginFallbackStage3BackgroundTests(PipelinePluginFallbackTestBase):
    def test_stage3_graxpert_model_requires_canonical_path_and_exact_sha(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        payload = b"locked-graxpert-bge-model"
        payload_sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plugin_root = root / "plugins"
            runtime_root = root / "graxpert-home"
            canonical = (
                plugin_root
                / "graxpert"
                / "bge-ai-models"
                / "model_v2_0_1"
                / "model.onnx"
            )
            canonical.parent.mkdir(parents=True)
            processor = SimpleNamespace(
                log=FakeLogger(),
                siril_plugin_dir=plugin_root,
            )

            self.assertFalse(
                stage3_module._stage3_ensure_graxpert_bge_model(processor)
            )
            (plugin_root / "model_v2_0_1.onnx").write_bytes(payload)
            self.assertFalse(
                stage3_module._stage3_ensure_graxpert_bge_model(processor)
            )
            canonical.write_bytes(b"wrong")
            self.assertFalse(
                stage3_module._stage3_ensure_graxpert_bge_model(processor)
            )

            canonical.write_bytes(payload)
            with (
                patch.object(
                    stage3_module,
                    "STAGE3_GRAXPERT_BGE_MODEL_SHA256",
                    payload_sha,
                ),
                patch.object(
                    stage3_module.os.path,
                    "expanduser",
                    return_value=str(runtime_root),
                ),
            ):
                self.assertTrue(
                    stage3_module._stage3_ensure_graxpert_bge_model(processor)
                )
            installed = (
                runtime_root
                / "bge-ai-models"
                / "model_v2_0_1"
                / "model.onnx"
            )
            self.assertEqual(installed.read_bytes(), payload)
            self.assertEqual(
                processor._stage3_graxpert_provenance["correction"],
                "subtraction",
            )
            self.assertEqual(
                processor._stage3_graxpert_provenance["compute"],
                "cpu",
            )

    def test_stage3_graxpert_background_model_is_preserved_and_finalized(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        with tempfile.TemporaryDirectory() as td:
            process_dir = Path(td)
            current = process_dir / "stage3_bg_input.fit"
            emitted = process_dir / "stage3_bg_input_bg.fit"
            emitted.write_bytes(b"background-model")
            processor = SimpleNamespace(
                process_dir=process_dir,
                log=FakeLogger(),
                siril=SimpleNamespace(
                    get_image_filename=lambda: str(current),
                ),
                _stage3_graxpert_provenance={
                    "background_model_artifact": None,
                },
            )

            candidate_path = stage3_module._stage3_capture_graxpert_background_model(
                processor,
                label="GraXpert-AI BGE CPU",
            )
            self.assertIsNotNone(candidate_path)
            self.assertTrue(Path(candidate_path).is_file())
            selected = {"background_model_artifact": candidate_path}
            final_path = stage3_module._stage3_finalize_graxpert_background_model(
                processor,
                selected,
            )
            self.assertEqual(
                Path(str(final_path)).name,
                "stage3_background_model.fit",
            )
            self.assertTrue(Path(str(final_path)).is_file())
            self.assertEqual(
                processor._stage3_graxpert_provenance[
                    "background_model_artifact"
                ],
                final_path,
            )

    def test_stage3_outcome_reason_priority_is_deterministic(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        defaults = {
            "policy_abort_candidate_search": False,
            "failure_action": "auto_fallback",
            "final_output_validation_rejected": False,
            "bg_ok": True,
            "stage_saved": True,
            "pattern_review_required": False,
            "compound_selected_degraded": False,
            "selected_gate_warnings": [],
            "background_backup_used": False,
            "profile_fallback_used": False,
            "fallback_warning": False,
            "review_required": False,
        }
        cases = (
            (
                {
                    "policy_abort_candidate_search": True,
                    "failure_action": "stop",
                    "final_output_validation_rejected": True,
                },
                "failure_policy_stop",
            ),
            (
                {
                    "policy_abort_candidate_search": True,
                    "failure_action": "preserve_review",
                },
                "failure_policy_preserve_review",
            ),
            (
                {"final_output_validation_rejected": True},
                "final_output_validation_rejected",
            ),
            ({"stage_saved": False}, "stage3_output_save_failed"),
            ({"bg_ok": False}, "no_background_candidate_accepted"),
            (
                {"pattern_review_required": True},
                "mixed_gradient_pattern_noise_review",
            ),
            (
                {"compound_selected_degraded": True},
                "compound_poly_residual_rbf_degraded_review",
            ),
            (
                {"selected_gate_warnings": ["limited improvement"]},
                "background_accepted_with_soft_warnings",
            ),
            (
                {"background_backup_used": True},
                "background_backup_accepted",
            ),
            (
                {"profile_fallback_used": True},
                "target_profiler_fallback",
            ),
            (
                {"fallback_warning": True},
                "background_improvement_limited",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                values = {**defaults, **overrides}
                self.assertEqual(
                    stage3_module._stage3_outcome_reason_code(**values),
                    expected,
                )

    def test_stage3_outer_halo_keeps_better_scored_polynomial_fidelity(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        polynomial = {
            "label": "subsky-poly-existing",
            "score": 0.0828,
            "preservation": {
                "target_flux_retention_ratio": 0.9999,
                "target_morphology_correlation": 0.999999,
            },
        }
        rbf = {
            "label": "subsky-rbf-existing-2",
            "score": 0.0881,
            "preservation": {
                "target_flux_retention_ratio": 0.9963,
                "target_morphology_correlation": 0.999993,
            },
        }

        selected, report = stage3_module._stage3_outer_halo_selection_override(
            polynomial,
            rbf,
            {"protect_outer_halo": True},
        )

        self.assertIs(selected, polynomial)
        self.assertTrue(report["applied"])
        self.assertEqual(
            report["reason_code"],
            "outer_halo_low_order_fidelity_preferred",
        )

    def test_stage3_outer_halo_allows_rbf_with_better_aggregate_score(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        polynomial = {
            "label": "subsky-poly-existing",
            "score": 0.12,
            "preservation": {"target_flux_retention_ratio": 1.0},
        }
        rbf = {
            "label": "subsky-rbf-existing-2",
            "score": 0.08,
            "preservation": {"target_flux_retention_ratio": 0.997},
        }

        selected, report = stage3_module._stage3_outer_halo_selection_override(
            polynomial,
            rbf,
            {"protect_outer_halo": True},
        )

        self.assertIs(selected, rbf)
        self.assertFalse(report["applied"])

    def test_stage3_outer_halo_keeps_materially_cleaner_safe_rbf(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        polynomial = {
            "label": "subsky-poly-existing",
            "score": 0.0828,
            "validation": {"robust_span": 3.58e-5},
            "preservation": {
                "target_flux_retention_ratio": 0.9999,
                "target_morphology_correlation": 0.999999,
                "target_centroid_shift_fraction": 1.2e-5,
            },
        }
        rbf = {
            "label": "subsky-rbf-existing-2",
            "score": 0.0881,
            "validation": {"robust_span": 1.11e-5},
            "preservation": {
                "target_flux_retention_ratio": 0.9963,
                "target_morphology_correlation": 0.999993,
                "target_centroid_shift_fraction": 0.00022,
            },
        }

        selected, report = stage3_module._stage3_outer_halo_selection_override(
            polynomial,
            rbf,
            {"protect_outer_halo": True},
        )

        self.assertIs(selected, rbf)
        self.assertFalse(report["applied"])
        self.assertTrue(report["preserved_statistical_selection"])
        self.assertEqual(
            report["reason_code"],
            "rbf_material_background_gain_within_outer_halo_fidelity_budget",
        )

    def test_background_chroma_noise_ignores_smooth_colour_bias(self):
        processor = pipeline_module.StarunPostProcessor()
        smooth_red = np.full((3, 64, 64), 0.04, dtype=np.float32)
        smooth_red[0] = 0.16

        metrics = processor._background_quality_metrics(smooth_red)

        self.assertLess(metrics["chroma_noise_score"], 0.05)
        self.assertGreater(metrics["background_chroma_load"], 0.50)

    def test_background_chroma_noise_detects_high_frequency_colour_variation(self):
        processor = pipeline_module.StarunPostProcessor()
        yy, xx = np.mgrid[:64, :64]
        checker = ((xx + yy) % 2).astype(np.float32)
        noisy = np.full((3, 64, 64), 0.08, dtype=np.float32)
        noisy[0] += checker * 0.10
        noisy[2] += (1.0 - checker) * 0.10

        metrics = processor._background_quality_metrics(noisy)

        self.assertGreater(metrics["chroma_noise_score"], 0.34)

    def test_pipeline_status_uncalibrated_background_cast_is_review_required(self):
        probe = pipeline_module.StarunPostProcessor()
        probe.results = []
        probe._require_review(
            7,
            "uncalibrated_background_color_review_required",
        )

        status = probe._pipeline_result_status()

        self.assertEqual(status, "review_required")

    def test_plugin_fingerprint_uses_preview_and_bounded_sample(self):
        preview_calls: list[bool] = []
        image = np.zeros((3, 512, 512), dtype=np.float32)
        processor = SimpleNamespace(
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: (
                    preview_calls.append(preview) or image
                )
            ),
            log=FakeLogger(),
        )

        before = pipeline_module.plugin_runner.current_image_fingerprint(processor)
        image[:] = 1.0
        after = pipeline_module.plugin_runner.current_image_fingerprint(processor)

        self.assertEqual(preview_calls, [True, True])
        self.assertNotEqual(before, after)

    def test_stage3_quality_diagnostics_are_not_a_compatibility_gate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        preservation = {
            "available": True,
            "star_retention_ratio": 0.82,
            "nebula_mean_change_ratio": 0.14,
            "before_star_count": 100,
            "after_star_count": 82,
        }
        gate_msg = stage3_module._stage3_quality_diagnostic_message(
            pipeline_module.ImageFeatures(bg_std=0.02, bg_median=0.08, object_area_ratio=0.20),
            pipeline_module.ImageFeatures(bg_std=0.02, bg_median=0.08, object_area_ratio=0.20),
            preservation,
        )

        self.assertIn("held-out sky and pixel gates own acceptance", gate_msg)
        self.assertIn("star_retention_ratio=0.82000", gate_msg)

    def test_stage3_background_score_prefers_cleaner_low_gradient_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        before = {
            "bg_std": 0.00010,
            "gradient_score": 0.10,
            "dirty_background_score": 0.42,
            "red_dominance": 1.00,
            "blue_dominance": 1.00,
            "green_cast": 1.00,
        }
        dirty_candidate = {
            "bg_std": 0.00011,
            "gradient_score": 0.09,
            "dirty_background_score": 0.38,
            "chroma_noise_score": 0.12,
            "red_dominance": 1.32,
            "blue_dominance": 0.92,
            "green_cast": 1.18,
        }
        cleaner_candidate = {
            "bg_std": 0.00010,
            "gradient_score": 0.04,
            "dirty_background_score": 0.20,
            "chroma_noise_score": 0.05,
            "red_dominance": 1.02,
            "blue_dominance": 1.01,
            "green_cast": 0.99,
        }

        dirty_score = stage3_module._stage3_background_score_components(
            before, dirty_candidate
        )["total"]
        cleaner_score = stage3_module._stage3_background_score_components(
            before, cleaner_candidate
        )["total"]

        self.assertLess(cleaner_score, dirty_score)
        self.assertFalse(stage3_module._stage3_candidate_sufficient(before, dirty_candidate, dirty_score))
        self.assertTrue(stage3_module._stage3_candidate_sufficient(before, cleaner_candidate, cleaner_score))

    def test_stage3_background_score_archives_components_and_weights(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        before = {
            "bg_std": 0.01,
            "red_dominance": 1.0,
            "blue_dominance": 1.0,
            "green_cast": 1.0,
        }
        after = {
            "bg_std": 0.011,
            "dirty_background_score": 0.20,
            "gradient_score": 0.04,
            "chroma_noise_score": 0.03,
            "red_dominance": 1.02,
            "blue_dominance": 0.99,
            "green_cast": 1.01,
        }

        report = stage3_module._stage3_background_score_components(before, after)

        self.assertAlmostEqual(
            report["total"],
            report["total"],
        )
        self.assertEqual(
            set(report["components"]),
            {
                "dirty_background_score",
                "gradient_score",
                "chroma_noise_score",
                "bg_std_growth",
                "color_shift",
            },
        )
        self.assertAlmostEqual(
            report["total"],
            sum(report["weighted_components"].values()),
        )

    def test_stage3_statistical_selection_is_runtime_authoritative(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        current = {
            "label": "runtime-score-winner",
            "source": "builtin",
            "score": 0.10,
            "preservation_penalty": 0.0,
            "directional_pattern_penalty": 0.0,
            "validation": {
                "status": "ready",
                "robust_span": 0.08,
                "patch_median_uncertainty": 0.01,
            },
            "validation_gate": {
                "accepted": True,
                "span_improvement": 0.02,
                "sampling_uncertainty_3sigma": 0.01,
            },
        }
        statistical = {
            "label": "statistical-shadow-winner",
            "source": "builtin",
            "score": 0.16,
            "preservation_penalty": 0.05,
            "directional_pattern_penalty": 0.0,
            "validation": {
                "status": "ready",
                "robust_span": 0.02,
                "patch_median_uncertainty": 0.01,
            },
            "validation_gate": {
                "accepted": True,
                "span_improvement": 0.08,
                "sampling_uncertainty_3sigma": 0.01,
            },
        }

        report = stage3_module._stage3_select_candidate(
            [current, statistical],
            current,
        )

        self.assertEqual(report["current_runtime_candidate"], current["label"])
        self.assertEqual(
            report["recommended_candidate"],
            statistical["label"],
        )
        self.assertTrue(report["selection_would_change"])
        self.assertTrue(report["runtime_selection_affected"])

    def test_stage3_output_first_flux_penalty_is_symmetric(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        loss = stage3_module._stage3_preservation_penalty(
            {"available": True, "target_flux_retention_ratio": 0.5},
            gate_profile="output_first",
        )
        growth = stage3_module._stage3_preservation_penalty(
            {"available": True, "target_flux_retention_ratio": 1.5},
            gate_profile="output_first",
        )

        self.assertAlmostEqual(loss, growth)

    def test_stage3_batch_regression_m8a_selects_best_soft_warning_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        candidates = [
            {
                "label": "M8-A subsky-rbf-existing-1",
                "source": "builtin",
                "score": 0.30,
                "sufficient": False,
                "severity": "soft_warning",
                "gate_warnings": ["improvement below sampling uncertainty"],
                "validation": {
                    "status": "ready",
                    "robust_span": 0.000008489204242,
                    "patch_median_uncertainty": 0.000001644499422,
                },
                "validation_gate": {"accepted": True},
                "preservation": {
                    "target_flux_retention_ratio": 0.9969786552,
                    "target_morphology_correlation": 0.9999974168,
                    "target_centroid_shift_fraction": 0.0001479180,
                    "target_change_residual_significance": 0.0475881849,
                },
            },
            {
                "label": "M8-A DBE",
                "source": "plugin",
                "score": 0.28,
                "sufficient": False,
                "severity": "soft_warning",
                "gate_warnings": ["improvement below sampling uncertainty"],
                "validation": {
                    "status": "ready",
                    "robust_span": 0.000010031891356,
                    "patch_median_uncertainty": 0.000001644499423,
                },
                "validation_gate": {"accepted": True},
                "preservation": {
                    "target_flux_retention_ratio": 1.1922317502,
                    "target_morphology_correlation": 0.9996902601,
                    "target_centroid_shift_fraction": 0.0007474976,
                    "target_change_residual_significance": 8.4440638041,
                },
            },
        ]

        report = stage3_module._stage3_select_candidate(
            candidates,
            candidates[1],
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(
            report["recommended_candidate"],
            "M8-A subsky-rbf-existing-1",
        )
        self.assertTrue(report["selection_would_change"])

    def test_stage3_verified_background_color_normalization_clears_only_generic_warning(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        before = {
            "bg_std": 0.00021,
            "gradient_score": 0.13,
            "dirty_background_score": 0.09,
            "chroma_noise_score": 0.15,
            "red_dominance": 2.75,
            "blue_dominance": 0.89,
            "green_cast": 0.55,
            "color_balance_score": 0.07,
        }
        accepted_gate = {"accepted": True, "severity": "normal"}
        candidate = {
            "source": "builtin",
            "score": 0.76,
            "gate_warnings": [
                "candidate does not meet clean-output sufficiency thresholds"
            ],
            "hard_gate_metrics_available": True,
            "after_adaptive": {
                "bg_std": 0.00016,
                "gradient_score": 0.018,
                "dirty_background_score": 0.014,
                "chroma_noise_score": 0.015,
                "red_dominance": 1.28,
                "blue_dominance": 0.97,
                "green_cast": 0.89,
                "color_balance_score": 0.83,
            },
            "background_score_components": {
                "components": {"color_shift": 1.47},
                "weighted_components": {"color_shift": 0.66},
            },
            "preservation": {
                "target_flux_retention_ratio": 0.98,
                "target_morphology_correlation": 0.9998,
                "target_centroid_shift_fraction": 0.006,
                "target_change_residual_significance": 0.34,
            },
            "pixel_integrity_gate": accepted_gate,
            "target_fidelity_gate": accepted_gate,
            "validation_gate": accepted_gate,
            "pattern_quality_gate": accepted_gate,
        }
        final_output = {
            "accepted": True,
            "severity": "normal",
            "pixel_integrity_gate": accepted_gate,
        }

        report = stage3_module._stage3_verified_background_color_normalization(
            before,
            candidate,
            final_output,
            gate_profile="output_first",
        )

        self.assertTrue(report["applied"], report)
        self.assertEqual(
            report["reason_code"],
            "verified_background_color_normalization",
        )
        candidate.update(
            label="subsky-rbf-existing-2",
            sufficient=False,
            validation={
                "status": "ready",
                "robust_span": 0.00015,
                "patch_median_uncertainty": 0.00003,
            },
            directional_pattern_penalty=0.0,
        )
        candidate_evidence = (
            stage3_module._stage3_color_normalization_candidate_evidence(
                candidate
            )
        )
        self.assertTrue(candidate_evidence["eligible"], candidate_evidence)
        adbe = {
            "label": "ADBE",
            "source": "plugin",
            "score": 0.34,
            "sufficient": True,
            "severity": "normal",
            "gate_warnings": [],
            "validation": {
                "status": "ready",
                "robust_span": 0.00020,
                "patch_median_uncertainty": 0.00003,
            },
            "validation_gate": accepted_gate,
            "preservation": {
                "target_flux_retention_ratio": 1.04,
                "target_morphology_correlation": 0.89,
                "target_centroid_shift_fraction": 0.049,
                "target_change_residual_significance": 1.31,
            },
        }
        selection = stage3_module._stage3_select_candidate(
            [candidate, adbe],
            adbe,
        )
        self.assertEqual(
            selection["recommended_candidate"],
            "subsky-rbf-existing-2",
        )
        selected_row = next(
            row
            for row in selection["candidates"]
            if row["label"] == "subsky-rbf-existing-2"
        )
        self.assertTrue(selected_row["verified_color_normalization_candidate"])

        candidate["preservation"] = {
            **candidate["preservation"],
            "target_morphology_correlation": 0.90,
        }
        rejected = stage3_module._stage3_verified_background_color_normalization(
            before,
            candidate,
            final_output,
            gate_profile="output_first",
        )
        self.assertFalse(rejected["applied"], rejected)
        self.assertIn(
            "target_signal_preservation_evidence_failed",
            rejected["issues"],
        )
    def test_stage3_pixel_integrity_gate_hard_rejects_invalid_outputs(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        baseline = np.ones((8, 10), dtype=np.float32)

        for candidate, expected_issue in (
            (baseline.copy(), "did not change"),
            (np.ones((7, 10), dtype=np.float32), "dimensions changed"),
            (np.full((8, 10), np.nan, dtype=np.float32), "non-finite"),
        ):
            with self.subTest(expected_issue=expected_issue):
                accepted, gate = stage3_module._stage3_candidate_pixel_gate(
                    baseline,
                    candidate,
                )
                self.assertFalse(accepted)
                self.assertEqual(gate["severity"], "hard_rejected")
                self.assertTrue(
                    any(expected_issue in issue for issue in gate["hard_issues"])
                )

    def test_stage3_candidate_sufficient_uses_policy_std_growth_limit(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        before = {
            "bg_std": 0.00010,
            "gradient_score": 0.08,
            "dirty_background_score": 0.28,
            "red_dominance": 1.00,
            "blue_dominance": 1.00,
            "green_cast": 1.00,
        }
        candidate = {
            "bg_std": 0.000107,
            "gradient_score": 0.02,
            "dirty_background_score": 0.12,
            "chroma_noise_score": 0.02,
            "red_dominance": 1.01,
            "blue_dominance": 1.00,
            "green_cast": 0.99,
        }

        self.assertTrue(stage3_module._stage3_candidate_sufficient(before, candidate, 0.12))
        self.assertFalse(
            stage3_module._stage3_candidate_sufficient(
                before,
                candidate,
                0.12,
                {"max_bg_std_growth": 1.03},
                "strict",
            )
        )

    def test_stage3_large_emission_nebula_prefers_poly_first(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.48},
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.18},
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "bright_emission_reflection_nebula",
                    "object_stats": {
                        "object_area_ratio": 0.16,
                        "nebulosity_area_ratio": 0.42,
                    },
                },
                {},
            )
        )
        self.assertTrue(
            stage3_module._stage3_prefers_poly_first(
                {
                    "target_type": "large_galaxy",
                    "object_stats": {"object_area_ratio": 0.48},
                },
                {},
            )
        )

    def test_stage3_faint_nebula_signal_protects_generic_profile(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        profile = {
            "target_type": "generic_low_snr_safe",
            "object_stats": {
                "nebulosity_area_ratio": 0.12,
                "faint_structure_score": 0.45,
            },
        }
        protect, context = stage3_module._stage3_should_exhaust_builtin_search(
            profile,
            {},
            {},
        )

        self.assertTrue(stage3_module._stage3_prefers_poly_first(profile, {}))
        self.assertTrue(protect)
        self.assertTrue(context["faint_nebula_protection"])
        self.assertEqual(context["protection_reason"], "faint_nebula_signal")

    def test_stage3_faint_structure_increases_nebula_preservation_penalty(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        preservation = {
            "available": True,
            "nebula_mean_change_ratio": 0.08,
            "star_retention_ratio": 1.0,
        }
        base_penalty = stage3_module._stage3_preservation_penalty(
            preservation,
            diffuse_context={"faint_structure_score": 0.40},
        )
        strong_penalty = stage3_module._stage3_preservation_penalty(
            preservation,
            diffuse_context={
                "faint_nebula_protection": True,
                "faint_structure_score": 0.90,
            },
        )

        self.assertGreater(strong_penalty, base_penalty)
        self.assertLessEqual(
            stage3_module._stage3_nebula_preservation_weight(
                {"faint_nebula_protection": True, "faint_structure_score": 1.0}
            ),
            2.5,
        )

    def test_stage3_theoretical_chain_falls_back_until_candidate_is_sufficient(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(
                    workflow_plugin_probe_enabled=False,
                    stage3_gate_profile="strict",
                )
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
                self.try_calls: list[tuple[str, ...]] = []
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
                        "bg_std": 0.00011,
                        "gradient_score": 0.11,
                        "dirty_background_score": 0.39,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.10,
                        "dirty_background_score": 0.37,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
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
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
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
        graxpert_command = (
            "pyscript",
            "GraXpert-AI.py",
            "-bge",
            "-model",
            "model_v2_0_1",
            "-correction",
            "subtraction",
            "-keep_bg",
            "-nogpu",
        )
        with (
            patch.object(
                stage3_module,
                "_stage3_theoretical_plugin_candidates",
                return_value=[
                    ("GraXpert-AI BGE CPU", graxpert_command, "graxpert")
                ],
            ),
            patch.object(
                stage3_module,
                "_stage3_candidate_pixel_gate",
                side_effect=_accepted_stage3_pixel_gate,
            ),
            patch.object(
                stage3_module,
                "assess_single_background_validation",
                return_value=(
                    True,
                    {
                        "status": "accepted",
                        "accepted": True,
                        "severity": "normal",
                        "warnings": [],
                        "hard_issues": [],
                        "issues": [],
                    },
                ),
            ),
            patch.object(
                stage3_module,
                "_stage3_write_spatial_background_lineage",
                return_value={
                    "status": "accepted",
                    "accepted": True,
                    "review_required": False,
                    "issues": [],
                },
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(
            background_attempts[:3],
            [
                ("subsky", "-rbf", "-existing"),
                ("subsky", "1", "-existing"),
                graxpert_command,
            ],
        )
        self.assertIn(
            ("load", "stage3_candidate_graxpert_ai_bge_cpu"),
            processor.cmd_calls,
        )
        self.assertEqual(
            processor.workflow_command_used["GraXpert 背景提取"],
            "GraXpert-AI BGE CPU",
        )
        self.assertTrue(processor.report["graxpert_attempted"])
        self.assertTrue(processor.report["backup_used"])
        self.assertEqual(
            processor.report["backup_reason"],
            "builtin_and_compound_not_clean_graxpert_selected",
        )
        self.assertFalse(processor.report["fallback_used"])
        self.assertEqual(processor.report["fallback_reasons"], [])
        self.assertTrue(processor.report["subsky_existing_enforced"])
        self.assertTrue(processor.siril.set_calls)
        self.assertTrue(processor.siril.set_calls[0]["recalculate"])
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage3_compound_target_guard_excludes_protected_structure(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        eligible = stage3_module._stage3_compound_target_guard(
            {"target_type": "open_cluster"},
            {"diffuse": False},
            {},
            {"requires_review": False},
        )
        dark = stage3_module._stage3_compound_target_guard(
            {"target_type": "dark_nebula_low_contrast"},
            {"diffuse": False},
            {"protect_dark_structure": True},
            {"requires_review": False},
        )
        diffuse = stage3_module._stage3_compound_target_guard(
            {"target_type": "emission_nebula_widefield"},
            {"diffuse": True, "emission_diffuse": True},
            {"protect_nebulosity": True},
            {"requires_review": False},
        )

        self.assertTrue(eligible["eligible"])
        self.assertFalse(dark["eligible"])
        self.assertIn("dark_nebula_structure", dark["reasons"])
        self.assertFalse(diffuse["eligible"])
        self.assertIn("large_or_diffuse_nebula_signal", diffuse["reasons"])

    def test_stage3_compound_config_cannot_relax_safety_contract(self):
        cfg = pipeline_module.PipelineConfig()
        cfg.stage3_compound_min_sample_count = 10
        cfg.stage3_compound_fit_min_count = 6
        cfg.stage3_compound_validation_min_count = 2
        cfg.stage3_compound_score_abs_improvement_min = 0.01
        cfg.stage3_compound_score_rel_improvement_min = 0.02
        cfg.stage3_compound_validation_improvement_min = 0.03
        cfg.stage3_compound_zero_point_abs_max = 0.05
        cfg.stage3_compound_zero_point_rel_max = 0.50

        clamped = pipeline_module.clamp_config(cfg)

        self.assertEqual(clamped.stage3_compound_min_sample_count, 12)
        self.assertEqual(clamped.stage3_compound_fit_min_count, 8)
        self.assertEqual(clamped.stage3_compound_validation_min_count, 4)
        self.assertEqual(
            clamped.stage3_compound_score_abs_improvement_min,
            0.03,
        )
        self.assertEqual(
            clamped.stage3_compound_score_rel_improvement_min,
            0.10,
        )
        self.assertEqual(
            clamped.stage3_compound_validation_improvement_min,
            0.10,
        )
        self.assertEqual(clamped.stage3_compound_zero_point_abs_max, 0.01)
        self.assertEqual(clamped.stage3_compound_zero_point_rel_max, 0.15)

    def test_stage3_compound_runs_before_plugins_and_reuses_safe_rbf(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()

        stage3_module.run_stage3_background_extraction(processor)

        subsky_calls = [
            call for call in processor.cmd_calls if call and call[0] == "subsky"
        ]
        self.assertEqual(len(subsky_calls), 4)
        self.assertEqual(subsky_calls[0], subsky_calls[-1])
        self.assertEqual(subsky_calls[1], ("subsky", "1", "-existing"))
        self.assertEqual(subsky_calls[2], ("subsky", "1", "-existing"))
        self.assertNotIn(("gxp",), processor.cmd_calls)
        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "accepted",
        )
        self.assertEqual(
            processor.report["model_used"],
            "subsky-poly-residual-rbf",
        )
        self.assertIsNone(processor.report["fallback_reason"])
        self.assertEqual(
            processor.report["schema_version"],
            "starun.stage3-background-quality.v9",
        )
        self.assertEqual(processor.report["algorithm_contract_version"], "1.7.0")
        self.assertIn("spatial_coverage", processor.report["decision_thresholds"])
        self.assertEqual(processor.report["selection"]["status"], "ready")
        self.assertTrue(
            any(
                "background_score_components" in attempt
                for attempt in processor.report["attempts"]
                if attempt.get("status") == "accepted"
            )
        )
        split = processor.report["safe_samples"]["fit_validation_split"]
        validation_points = {
            tuple(point) for point in split["validation_points"]
        }
        self.assertGreaterEqual(split["fit_count"], 24)
        self.assertGreaterEqual(split["validation_count"], 8)
        self.assertEqual(
            split["fit_count"] + split["validation_count"],
            split["sample_count"],
        )
        self.assertTrue(processor.siril.set_calls)
        self.assertTrue(
            all(call["count"] == split["fit_count"] for call in processor.siril.set_calls)
        )
        self.assertTrue(
            all(
                validation_points.isdisjoint(set(call["points"]))
                for call in processor.siril.set_calls
            )
        )
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertTrue(processor.report["backup_used"])

    def test_stage3_diffuse_sample_recovery_uses_strict_polynomial_only(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(external_success=True)
        processor.cfg.stage3_gate_profile = "output_first"
        processor.target_profile = {
            "target_type": "emission_nebula_widefield",
            "object_stats": {
                "object_area_ratio": 0.30,
                "nebulosity_area_ratio": 0.42,
            },
        }
        processor.pipeline_policy["stage3_background"] = {
            "protect_nebulosity": True,
            "reject_samples_on_nebula": True,
        }
        processor.metrics["polynomial"].update(
            gradient_score=0.02,
            dirty_background_score=0.12,
            chroma_noise_score=0.02,
        )
        points, base_report, refined_report = _stage3_recovery_sample_reports()

        with (
            patch.object(
                stage3_module,
                "build_safe_background_samples",
                side_effect=[([], base_report), (points, refined_report)],
            ),
            patch.object(
                stage3_module,
                "analyze_directional_pattern_noise",
                return_value={
                    "status": "ok",
                    "detected": False,
                    "pattern_score": 0.05,
                    "walking_noise_score": 0.04,
                },
            ),
            patch.object(
                stage3_module,
                "_stage3_theoretical_plugin_candidates",
                return_value=[("mock-graxpert", ("gxp",), "graxpert")],
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        subsky_calls = [
            call for call in processor.cmd_calls if call and call[0] == "subsky"
        ]
        self.assertEqual(subsky_calls, [("subsky", "1", "-existing")])
        self.assertFalse(any(call[0] == "gxp" for call in processor.cmd_calls))
        self.assertEqual(processor.report["configured_gate_profile"], "output_first")
        self.assertEqual(processor.report["effective_gate_profile"], "strict")
        self.assertEqual(
            processor.report["builtin_order_reason"],
            "conservative_sample_recovery_polynomial_degree_1_only",
        )
        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "not_required",
        )
        self.assertEqual(
            processor.report["safe_samples"]["recovery"]["status"],
            "applied",
        )
        self.assertTrue(
            processor.report["safe_samples"]["fit_validation_split"]
            ["regular_validation_ready"]
        )
        self.assertEqual(
            processor.report["reason_code"],
            "stage3_safe_sample_recovery_applied",
        )
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage3_dense_star_recovery_uses_masked_bg_samples(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(external_success=True)
        processor.cfg.stage3_gate_profile = "output_first"
        processor.target_profile = {
            "target_type": "emission_nebula_widefield",
            "object_stats": {
                "object_area_ratio": 0.30,
                "nebulosity_area_ratio": 0.42,
            },
        }
        processor.pipeline_policy["stage3_background"] = {
            "protect_nebulosity": True,
            "reject_samples_on_nebula": True,
        }
        processor.metrics["polynomial"].update(
            gradient_score=0.02,
            dirty_background_score=0.12,
            chroma_noise_score=0.02,
        )
        points, base_report, refined_report = (
            _stage3_dense_recovery_sample_reports()
        )

        with (
            patch.object(
                stage3_module,
                "build_safe_background_samples",
                side_effect=[([], base_report), (points, refined_report)],
            ) as sample_builder,
            patch.object(
                stage3_module,
                "analyze_directional_pattern_noise",
                return_value={
                    "status": "ok",
                    "detected": False,
                    "pattern_score": 0.05,
                    "walking_noise_score": 0.04,
                },
            ),
            patch.object(
                stage3_module,
                "_stage3_theoretical_plugin_candidates",
                return_value=[("mock-graxpert", ("gxp",), "graxpert")],
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(sample_builder.call_count, 2)
        self.assertFalse(
            sample_builder.call_args_list[0].kwargs["candidate_refinement"]
        )
        self.assertTrue(
            sample_builder.call_args_list[1].kwargs[
                "masked_catalog_statistics"
            ]
        )
        self.assertEqual(
            [call for call in processor.cmd_calls if call[0] == "subsky"],
            [("subsky", "1", "-existing")],
        )
        self.assertTrue(processor.siril.set_calls)
        self.assertFalse(processor.siril.set_calls[0]["recalculate"])
        recovery = processor.report["safe_samples"]["recovery"]
        self.assertEqual(recovery["status"], "applied")
        self.assertEqual(
            recovery["reason_code"],
            "stage3_dense_star_masked_sampling_applied",
        )
        self.assertEqual(recovery["recovery_mode"], "masked_catalog_statistics")
        self.assertAlmostEqual(
            recovery["strict_unmasked_sky_fraction"],
            0.3637058,
        )
        self.assertAlmostEqual(recovery["nonstellar_sky_fraction"], 0.5670248)
        self.assertEqual(
            processor.report["reason_code"],
            "stage3_safe_sample_recovery_applied",
        )
        self.assertTrue(
            processor.report["safe_samples"]["fit_validation_split"][
                "regular_validation_ready"
            ]
        )
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage3_rgb_recovery_projects_single_subsky_to_neutral_axis(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(external_success=True)
        processor.cfg.stage3_gate_profile = "output_first"
        processor.target_profile = {
            "target_type": "emission_nebula_widefield",
            "object_stats": {
                "object_area_ratio": 0.30,
                "nebulosity_area_ratio": 0.42,
            },
        }
        processor.pipeline_policy["stage3_background"] = {
            "protect_nebulosity": True,
            "reject_samples_on_nebula": True,
        }
        baseline_luma = processor.images["baseline"].copy()
        baseline_rgb = np.stack(
            (
                baseline_luma + np.float32(0.030),
                baseline_luma,
                baseline_luma - np.float32(0.015),
            ),
            axis=0,
        ).astype(np.float32)
        height, width = baseline_luma.shape
        y, x = np.mgrid[:height, :width]
        common_correction = (
            -0.048 * (x / (width - 1) - 0.5)
            - 0.020 * (y / (height - 1) - 0.5)
        )
        raw_proposal = baseline_rgb.astype(np.float64)
        raw_proposal += common_correction[None, :, :]
        raw_proposal += np.asarray((0.020, -0.012, 0.008))[:, None, None]
        processor.images["baseline"] = baseline_rgb
        processor.images["polynomial"] = raw_proposal.astype(np.float32)
        for state in ("single_rbf", "compound", "plugin"):
            mono = processor.images[state]
            processor.images[state] = np.stack((mono, mono, mono), axis=0)
        processor.metrics["polynomial"].update(
            gradient_score=0.02,
            dirty_background_score=0.12,
            chroma_noise_score=0.02,
        )
        points, base_report, refined_report = _stage3_recovery_sample_reports()

        with (
            patch.object(
                stage3_module,
                "build_safe_background_samples",
                side_effect=[([], base_report), (points, refined_report)],
            ),
            patch.object(
                stage3_module,
                "analyze_directional_pattern_noise",
                return_value={
                    "status": "ok",
                    "detected": False,
                    "pattern_score": 0.05,
                    "walking_noise_score": 0.04,
                },
            ),
            patch.object(
                stage3_module,
                "_stage3_theoretical_plugin_candidates",
                return_value=[("mock-graxpert", ("gxp",), "graxpert")],
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        subsky_calls = [
            call for call in processor.cmd_calls if call and call[0] == "subsky"
        ]
        self.assertEqual(subsky_calls, [("subsky", "1", "-existing")])
        self.assertIn(
            "stage3_candidate_neutral_axis_poly1",
            processor.saved_images,
        )
        output = processor.saved_images["stage3_bgremoved"]
        tolerance = processor.report["neutral_axis_projection"]["invariants"][
            "opponent_tolerance"
        ]
        np.testing.assert_allclose(
            output[0] - output[1],
            baseline_rgb[0] - baseline_rgb[1],
            rtol=0.0,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            output[2] - output[1],
            baseline_rgb[2] - baseline_rgb[1],
            rtol=0.0,
            atol=tolerance,
        )
        self.assertGreaterEqual(processor.siril.image_lock_entries, 1)
        self.assertEqual(processor.report["model_used"], "neutral-axis-poly1")
        self.assertEqual(
            processor.report["neutral_axis_projection"]["schema"],
            "starun.stage3-neutral-axis-projection.v1",
        )
        self.assertTrue(
            processor.report["final_output_validation"]
            ["neutral_axis_persistence"]["accepted"]
        )
        self.assertEqual(
            processor.report["reason_code"],
            "stage3_safe_sample_recovery_applied",
        )
        self.assertEqual(processor.results[-1][1], "ok")

    def test_stage3_diffuse_recovery_strict_warning_rolls_back_without_backup(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(external_success=True)
        processor.cfg.stage3_gate_profile = "output_first"
        processor.target_profile = {
            "target_type": "emission_nebula_widefield",
            "object_stats": {
                "object_area_ratio": 0.30,
                "nebulosity_area_ratio": 0.42,
            },
        }
        processor.pipeline_policy["stage3_background"] = {
            "protect_nebulosity": True,
        }
        points, base_report, refined_report = _stage3_recovery_sample_reports()

        with (
            patch.object(
                stage3_module,
                "build_safe_background_samples",
                side_effect=[([], base_report), (points, refined_report)],
            ),
            patch.object(
                stage3_module,
                "analyze_directional_pattern_noise",
                return_value={
                    "status": "ok",
                    "detected": False,
                    "pattern_score": 0.05,
                    "walking_noise_score": 0.04,
                },
            ),
            patch.object(
                stage3_module,
                "_stage3_theoretical_plugin_candidates",
                return_value=[("mock-graxpert", ("gxp",), "graxpert")],
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        subsky_calls = [
            call for call in processor.cmd_calls if call and call[0] == "subsky"
        ]
        self.assertEqual(subsky_calls, [("subsky", "1", "-existing")])
        self.assertFalse(any(call[0] == "gxp" for call in processor.cmd_calls))
        self.assertIsNone(processor.report["model_used"])
        self.assertTrue(processor.report["review_required"])
        self.assertEqual(
            processor.report["reason_code"],
            "stage3_conservative_recovery_candidate_rejected",
        )
        self.assertEqual(processor.state, "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")

    def test_stage3_diffuse_recovery_eligibility_is_fail_closed(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        _points, base_template, _refined = _stage3_recovery_sample_reports()

        for case in (
            "linearity_unknown",
            "usable_sky_below_minimum",
            "star_catalog_untrusted",
            "directional_pattern_detected",
            "graxpert_only",
        ):
            with self.subTest(case=case):
                processor = Stage3CompoundFake(external_success=True)
                processor.target_profile = {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {
                        "object_area_ratio": 0.30,
                        "nebulosity_area_ratio": 0.42,
                    },
                }
                processor.pipeline_policy["stage3_background"] = {
                    "protect_nebulosity": True,
                }
                base_report = copy.deepcopy(base_template)
                pattern_detected = False
                if case == "linearity_unknown":
                    processor.input_profile = {
                        "state": "unknown",
                        "safe_for_linear_steps": False,
                    }
                elif case == "usable_sky_below_minimum":
                    base_report["mask_evidence"]["usable_sky_fraction"] = 0.49
                elif case == "star_catalog_untrusted":
                    base_report["shared_scene_support"]["star_catalog"] = (
                        "unavailable"
                    )
                elif case == "directional_pattern_detected":
                    pattern_detected = True
                elif case == "graxpert_only":
                    processor.cfg.stage3_backend_policy = "graxpert_only"

                with (
                    patch.object(
                        stage3_module,
                        "build_safe_background_samples",
                        return_value=([], base_report),
                    ) as sample_builder,
                    patch.object(
                        stage3_module,
                        "analyze_directional_pattern_noise",
                        return_value={
                            "status": "ok",
                            "detected": pattern_detected,
                            "pattern_score": 0.70 if pattern_detected else 0.05,
                            "walking_noise_score": 0.04,
                        },
                    ),
                ):
                    stage3_module.run_stage3_background_extraction(processor)

                self.assertEqual(sample_builder.call_count, 1)
                self.assertFalse(
                    any(call[0] == "subsky" for call in processor.cmd_calls)
                )
                recovery = processor.report["safe_samples"]["recovery"]
                self.assertEqual(recovery["status"], "ineligible")
                self.assertEqual(
                    recovery["reason_code"],
                    "stage3_safe_sample_recovery_ineligible",
                )
                self.assertEqual(
                    processor.report["reason_code"],
                    "insufficient_source_masked_true_sky_support",
                )
                self.assertTrue(processor.report["review_required"])

    def test_stage3_compound_soft_profile_records_missing_hard_gate_evidence(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.cfg.stage3_gate_profile = "output_first"
        for state in ("single_rbf", "polynomial"):
            processor.metrics[state].update(
                gradient_score=0.90,
                dirty_background_score=0.90,
                chroma_noise_score=0.40,
            )
        processor._stage3_measure_features = lambda _label: (
            None
            if processor.state == "compound"
            else SimpleNamespace(state=processor.state)
        )

        stage3_module.run_stage3_background_extraction(processor)

        compound = next(
            attempt
            for attempt in processor.report["attempts"]
            if attempt.get("source") == "compound"
        )
        self.assertFalse(compound["hard_gate_metrics_available"])
        self.assertEqual(compound["status"], "accepted_with_warnings")
        self.assertIn(
            "compound hard-gate metrics unavailable",
            compound["gate_warnings"],
        )

    def test_stage3_candidate_attempt_limit_counts_compound_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.cfg.stage3_candidate_attempt_limit = 2

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(len(processor.report["attempts"]), 2)
        self.assertEqual(
            processor.report["compound_fallback"]["reason"],
            "candidate_attempt_limit_reached",
        )
        self.assertFalse(
            any(
                attempt.get("source") in {"compound", "graxpert", "plugin"}
                for attempt in processor.report["attempts"]
            )
        )

    def test_stage3_final_saved_output_hard_gate_rolls_back_profiled_run(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.input_profile = {
            "state": "linear",
            "safe_for_linear_steps": True,
            "confidence": 1.0,
            "source": "test",
        }
        original_save = processor._save_stage_output
        tampered = False

        def tamper_first_final_save(stem: str) -> bool:
            nonlocal tampered
            saved = original_save(stem)
            if stem == "stage3_bgremoved" and not tampered:
                tampered = True
                processor.images[processor.state] = processor.images["baseline"].copy()
            return saved

        processor._save_stage_output = tamper_first_final_save
        with patch.object(
            stage3_module,
            "assess_target_fidelity",
            return_value=(True, {"status": "accepted", "accepted": True}),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        final_validation = processor.report["final_output_validation"]
        self.assertEqual(final_validation["status"], "rejected")
        self.assertTrue(final_validation["rollback"]["completed"])
        self.assertTrue(final_validation["rollback"]["output_saved"])
        self.assertIsNone(processor.report["model_used"])
        self.assertEqual(
            processor.report["attempted_selected_model"],
            "subsky-poly-residual-rbf",
        )
        self.assertEqual(processor.state, "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "final_output_validation_rejected",
        )

    def test_stage3_final_saved_output_soft_warning_remains_accepted(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        baseline_image = np.zeros((32, 40), dtype=np.float32)
        candidate_image = np.full((32, 40), 0.01, dtype=np.float32)
        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_gate_profile="output_first"),
            siril=SimpleNamespace(
                get_image_pixeldata=lambda preview=False: candidate_image.copy()
            ),
        )
        baseline_validation = {
            "status": "ready",
            "robust_span": 0.010,
            "patch_mad_median": 0.004,
            "patch_median_uncertainty": 0.001,
        }
        candidate_validation = {
            "status": "ready",
            "robust_span": 0.011,
            "patch_mad_median": 0.004,
            "patch_median_uncertainty": 0.001,
        }

        with patch.object(
            stage3_module,
            "measure_background_validation",
            return_value=candidate_validation,
        ):
            report = stage3_module._stage3_final_output_validation(
                processor,
                baseline_image=baseline_image,
                baseline_validation=baseline_validation,
                validation_points=[(8.0, 8.0)] * 4,
                patch_radius=4,
                minimum_count=4,
                enforced=True,
            )

        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["severity"], "soft_warning")
        self.assertEqual(
            report["validation_gate"]["severity"],
            "soft_warning",
        )

    def test_stage3_selected_soft_warning_does_not_require_review(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.cfg.stage3_gate_profile = "output_first"
        soft_final = {
            "status": "accepted",
            "accepted": True,
            "enforced": True,
            "severity": "soft_warning",
            "validation_gate": {
                "status": "accepted_with_warnings",
                "accepted": True,
                "severity": "soft_warning",
                "warnings": [
                    "held-out span improvement is below sampling uncertainty"
                ],
                "issues": [
                    "held-out span improvement is below sampling uncertainty"
                ],
                "hard_issues": [],
            },
            "pixel_integrity_gate": {
                "status": "accepted",
                "accepted": True,
                "warnings": [],
                "hard_issues": [],
            },
        }

        with (
            patch.object(
                stage3_module,
                "_stage3_final_output_validation",
                return_value=soft_final,
            ),
            patch.object(
                stage3_module,
                "_stage3_verified_background_color_normalization",
                return_value={"applied": False},
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.results[-1][1], "ok")
        self.assertFalse(processor.report["review_required"])
        self.assertEqual(processor.report["quality"], "ok")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "background_accepted_with_soft_warnings",
        )
        self.assertEqual(processor._stage_review_reasons(3), [])

    def test_stage3_verified_noop_audit_allows_only_below_three_sigma(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        accepted_gate = {"accepted": True}
        baseline_validation = {
            "status": "ready",
            "robust_span": 0.040,
            "patch_mad_median": 0.002,
            "patch_median_uncertainty": 0.001,
            "patch_radius": 3,
        }
        candidate_validation = {
            "status": "ready",
            "robust_span": 0.039,
            "patch_mad_median": 0.002,
            "patch_median_uncertainty": 0.001,
            "patch_radius": 3,
        }
        attempt = {
            "label": "bounded-poly",
            "status": "rejected",
            "candidate_stem": "stage3_candidate_bounded_poly",
            "candidate_checkpoint": {
                "status": "accepted",
                "accepted": True,
            },
            "hard_gate_metrics_available": True,
            "pixel_integrity_gate": accepted_gate,
            "target_fidelity_gate": accepted_gate,
            "pattern_quality_gate": accepted_gate,
            "directional_gradient_gate": accepted_gate,
            "color_shift_gate": accepted_gate,
            "validation": candidate_validation,
            "validation_gate": {
                "accepted": False,
                "baseline_span": 0.040,
                "candidate_span": 0.039,
                "span_improvement": 0.001,
                "sampling_uncertainty_3sigma": 0.006,
                "material_improvement": False,
                "span_not_worse": True,
                "background_rms_not_worse": True,
                "hard_issues": [
                    "held-out span improvement is below sampling uncertainty"
                ],
            },
            "gate_warnings": [],
        }
        process = {
            "linear_input": {"confirmed": True},
            "true_sky_support": {"supported": True},
            "hard_block_reasons": [],
        }
        pattern = {"status": "ok", "detected": False}
        route = {"requires_review": False}

        accepted = stage3_module._stage3_verified_noop_candidate_audit(
            process,
            pattern,
            route,
            [attempt],
            baseline_validation=baseline_validation,
        )
        target_rejected = copy.deepcopy(attempt)
        target_rejected["target_fidelity_gate"] = {"accepted": False}
        rejected = stage3_module._stage3_verified_noop_candidate_audit(
            process,
            pattern,
            route,
            [target_rejected],
            baseline_validation=baseline_validation,
        )
        sky_limited = copy.deepcopy(process)
        sky_limited["true_sky_support"]["supported"] = False
        sky_rejected = stage3_module._stage3_verified_noop_candidate_audit(
            sky_limited,
            pattern,
            route,
            [attempt],
            baseline_validation=baseline_validation,
        )
        missing_validation = copy.deepcopy(attempt)
        missing_validation.pop("validation_gate")
        incomplete_rejected = (
            stage3_module._stage3_verified_noop_candidate_audit(
                process,
                pattern,
                route,
                [attempt, missing_validation],
                baseline_validation=baseline_validation,
            )
        )
        save_failed = copy.deepcopy(attempt)
        save_failed.update(
            status="candidate_save_failed",
            candidate_stem=None,
            candidate_checkpoint={"status": "rejected", "accepted": False},
            failure_reason="candidate checkpoint save failed",
        )
        save_rejected = stage3_module._stage3_verified_noop_candidate_audit(
            process,
            pattern,
            route,
            [attempt, save_failed],
            baseline_validation=baseline_validation,
        )

        self.assertTrue(accepted["eligible"], accepted)
        self.assertFalse(rejected["eligible"], rejected)
        self.assertEqual(rejected["candidate_blockers"], ["bounded-poly"])
        self.assertFalse(sky_rejected["eligible"], sky_rejected)
        self.assertFalse(incomplete_rejected["eligible"], incomplete_rejected)
        self.assertEqual(incomplete_rejected["assessed_candidate_count"], 2)
        self.assertFalse(save_rejected["eligible"], save_rejected)
        self.assertFalse(
            save_rejected["candidates"][1]["technical_checks"]
            ["candidate_checkpoint_saved"]
        )

        accepted_candidate = copy.deepcopy(attempt)
        accepted_candidate.update(status="accepted", candidate_stem="accepted")
        accepted_candidate["validation_gate"]["accepted"] = True
        accepted_rejected = stage3_module._stage3_verified_noop_candidate_audit(
            process,
            pattern,
            route,
            [accepted_candidate],
            baseline_validation=baseline_validation,
        )
        self.assertFalse(accepted_rejected["eligible"], accepted_rejected)

        for field in (
            "span_improvement",
            "sampling_uncertainty_3sigma",
        ):
            missing = copy.deepcopy(attempt)
            missing["validation_gate"].pop(field)
            missing_result = stage3_module._stage3_verified_noop_candidate_audit(
                process,
                pattern,
                route,
                [missing],
                baseline_validation=baseline_validation,
            )
            self.assertFalse(missing_result["eligible"], missing_result)

        nonfinite = copy.deepcopy(attempt)
        nonfinite["validation_gate"]["span_improvement"] = math.nan
        nonfinite_result = stage3_module._stage3_verified_noop_candidate_audit(
            process,
            pattern,
            route,
            [nonfinite],
            baseline_validation=baseline_validation,
        )
        self.assertFalse(nonfinite_result["eligible"], nonfinite_result)

        contradictory = copy.deepcopy(attempt)
        contradictory["validation_gate"]["sampling_uncertainty_3sigma"] = 0.1
        contradiction_result = (
            stage3_module._stage3_verified_noop_candidate_audit(
                process,
                pattern,
                route,
                [contradictory],
                baseline_validation=baseline_validation,
            )
        )
        self.assertFalse(contradiction_result["eligible"], contradiction_result)

    def test_stage3_restored_noop_pixel_gate_requires_exact_identity(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        baseline = np.full((3, 32, 40), 0.125, dtype=np.float32)
        exact_ok, exact_report = (
            stage3_module._stage3_restored_noop_pixel_gate(
                baseline,
                baseline.copy(),
                gate_profile="strict",
            )
        )
        mutated = baseline.copy()
        mutated[0, 0, 0] = np.nextafter(
            mutated[0, 0, 0],
            np.float32(1.0),
        )
        mutated_ok, mutated_report = (
            stage3_module._stage3_restored_noop_pixel_gate(
                baseline,
                mutated,
                gate_profile="strict",
            )
        )

        self.assertTrue(exact_ok, exact_report)
        self.assertTrue(exact_report["pixel_exact"])
        self.assertFalse(mutated_ok, mutated_report)
        self.assertFalse(mutated_report["pixel_exact"])

    def test_stage3_verified_noop_absolute_background_gate_is_fail_closed(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        ready = {
            "status": "ready",
            "sample_count": 8,
            "expected_count": 8,
            "minimum": 0.010,
            "p10": 0.012,
            "median": 0.018,
            "p90": 0.024,
            "maximum": 0.030,
            "robust_span": 0.012,
            "supported_pixel_count": 800,
            "low_clip_count": 0,
            "high_clip_count": 0,
        }

        accepted = stage3_module._stage3_absolute_background_gate(ready)
        self.assertTrue(accepted["accepted"], accepted)

        for field, value in (
            ("minimum", 0.0),
            ("maximum", 1.0),
            ("low_clip_count", 1),
            ("high_clip_count", 1),
        ):
            with self.subTest(field=field):
                rejected_payload = dict(ready)
                rejected_payload[field] = value
                rejected = stage3_module._stage3_absolute_background_gate(
                    rejected_payload
                )
                self.assertFalse(rejected["accepted"], rejected)

        missing = dict(ready)
        missing.pop("p10")
        self.assertFalse(
            stage3_module._stage3_absolute_background_gate(missing)["accepted"]
        )
        nonfinite = dict(ready)
        nonfinite["median"] = math.nan
        self.assertFalse(
            stage3_module._stage3_absolute_background_gate(nonfinite)["accepted"]
        )

    def test_stage3_verified_noop_restores_exact_baseline_and_stays_formal(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.cfg.stage3_gate_profile = "output_first"
        candidate_audit = {
            "schema": "starun.stage3-verified-noop-candidate-audit.v1",
            "status": "eligible",
            "eligible": True,
            "candidates": [{"label": "subsky-poly-existing"}],
        }
        noop_report = {
            "schema": "starun.stage3-verified-noop.v1",
            "status": "accepted",
            "accepted": True,
            "checks": {
                "pixel_exact": True,
                "true_sky_support": True,
                "absolute_background": True,
                "target_fidelity": True,
                "directional_pattern": True,
                "directional_gradient": True,
            },
        }

        with (
            patch.object(
                stage3_module,
                "_stage3_try_background_command",
                return_value=(False, "fixture_candidate_rejected"),
            ),
            patch.object(
                stage3_module,
                "_stage3_verified_noop_candidate_audit",
                return_value=candidate_audit,
            ),
            patch.object(
                stage3_module,
                "_stage3_verify_restored_noop",
                return_value=noop_report,
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.state, "baseline")
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "verified_noop_below_sampling_uncertainty",
        )
        self.assertEqual(
            processor.result_metadata[-1]["execution"],
            "skipped",
        )
        self.assertTrue(processor.report["verified_noop"]["accepted"])
        self.assertFalse(processor.report["review_required"])

    def test_stage3_verified_noop_rejects_persisted_output_pixel_mutation(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        baseline = processor.images["baseline"].copy()
        processor._save_stage_output("stage3_bgremoved")
        mutated = processor.saved_images["stage3_bgremoved"].copy()
        mutated[0, 0] = np.nextafter(mutated[0, 0], np.float32(1.0))
        processor.saved_images["stage3_bgremoved"] = mutated

        report = stage3_module._stage3_verify_persisted_noop_output(
            processor,
            baseline_image=baseline,
            output_stem="stage3_bgremoved",
        )

        self.assertFalse(report["accepted"], report)
        self.assertFalse(report["checks"]["pixels_exact"])
        self.assertNotEqual(
            report["baseline_pixel_sha256"],
            report["persisted_pixel_sha256"],
        )

    def test_stage3_pre_candidate_preserve_is_review_only_not_verified_noop(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        processor.cfg.stage3_processing_mode = "preserve"

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.state, "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "stage3_passthrough_requires_verified_noop",
        )
        self.assertTrue(processor.report["review_required"])
        self.assertFalse(
            processor.report["spatial_background_lineage"]["accepted"]
        )
        self.assertIn(
            "stage3_passthrough_requires_verified_noop",
            processor._stage_review_reasons(3),
        )

    def test_stage3_selected_correction_save_failure_is_review_required(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        original_save = processor._save_stage_output

        def fail_final_save(stem: str) -> bool:
            if stem == "stage3_bgremoved":
                return False
            return original_save(stem)

        processor._save_stage_output = fail_final_save
        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertTrue(processor.report["review_required"])
        self.assertEqual(
            processor.report["reason_code"],
            "stage3_output_save_failed",
        )
        self.assertIn(
            "stage3_output_save_failed",
            processor._stage_review_reasons(3),
        )

    def test_stage3_formal_lineage_binds_input_output_support_and_reference_plane(
        self,
    ):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            height, width = 96, 128
            yy, xx = np.mgrid[:height, :width]
            mono = (
                0.05
                + 0.02 * xx / (width - 1)
                + 0.01 * yy / (height - 1)
            ).astype(np.float32)
            image = np.stack((mono, mono * 0.98, mono * 1.02))
            fits.PrimaryHDU(image).writeto(root / "stage3_bg_input.fit")
            fits.PrimaryHDU(image).writeto(root / "stage3_bgremoved.fit")
            points = [
                (
                    (cell_x + 0.5) / 4.0 * (width - 1),
                    (cell_y + 0.5) / 4.0 * (height - 1),
                )
                for cell_y in range(4)
                for cell_x in range(4)
            ]
            reports = {}

            def write_report(name, payload):
                reports[name] = payload
                (root / name).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            pipeline = SimpleNamespace(
                process_dir=root,
                work_dir=root / "test-run",
                _write_stage_json=write_report,
            )

            report = stage3_module._stage3_write_spatial_background_lineage(
                pipeline,
                baseline_image=image,
                fit_points=points[:12],
                validation_points=points[12:],
                patch_radius=3,
                support_mask=np.ones((height, width), dtype=bool),
                projection={},
                review_required=False,
                processing_route="verified_noop",
            )

            self.assertTrue(report["accepted"], report)
            self.assertEqual(report["processing_route"], "verified_noop")
            self.assertEqual(len(report["stage3_input_pixel_sha256"]), 64)
            self.assertEqual(len(report["stage3_output_pixel_sha256"]), 64)
            self.assertEqual(len(report["support_sha256"]), 64)
            self.assertEqual(len(report["reference_plane"]["sha256"]), 64)
            self.assertIn("luma", report["reference_plane"]["components"])
            self.assertEqual(
                report["support_kind"],
                "candidate_independent_full_sky_mask",
            )
            self.assertEqual(report["support_pixel_count"], height * width)
            self.assertEqual(report["support_coverage"], 1.0)
            self.assertLess(
                report["sample_patch_support_pixel_count"],
                report["support_pixel_count"],
            )

            self.assertEqual(
                reports["stage3_spatial_background_lineage.json"],
                report,
            )
            loaded = stage3_module.spatial_background_lineage.load_lineage(root)
            self.assertTrue(loaded["accepted"], loaded)
            with fits.open(
                root / "stage3_bg_input.fit",
                mode="update",
                memmap=False,
            ) as hdul:
                hdul[0].data[0, 0, 0] += np.float32(0.001)
                hdul.flush()
            tampered = stage3_module.spatial_background_lineage.load_lineage(root)
            self.assertFalse(tampered["accepted"])
            self.assertIn("input SHA mismatch", " ".join(tampered["issues"]))

    def test_stage3_formal_lineage_rejects_missing_full_sky_support(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((3, 32, 40), 0.05, dtype=np.float32)
            fits.PrimaryHDU(image).writeto(root / "stage3_bg_input.fit")
            fits.PrimaryHDU(image).writeto(root / "stage3_bgremoved.fit")
            points = [(4.0, 4.0), (20.0, 4.0), (35.0, 4.0), (4.0, 26.0)]
            pipeline = SimpleNamespace(
                process_dir=root,
                work_dir=root / "test-run",
                _write_stage_json=lambda _name, _payload: None,
            )

            rejected = stage3_module._stage3_write_spatial_background_lineage(
                pipeline,
                baseline_image=image,
                fit_points=points[:3],
                validation_points=points[3:],
                patch_radius=2,
                support_mask=None,
                projection={},
                review_required=False,
                processing_route="verified_noop",
            )

            self.assertFalse(rejected["accepted"], rejected)
            self.assertIn("candidate-independent sky support", rejected["issues"][0])

    def test_stage3_verified_noop_lineage_rejects_output_pixel_mutation(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            height, width = 96, 128
            yy, xx = np.mgrid[:height, :width]
            mono = (
                0.05
                + 0.02 * xx / (width - 1)
                + 0.01 * yy / (height - 1)
            ).astype(np.float32)
            image = np.stack((mono, mono * 0.98, mono * 1.02))
            mutated = image.copy()
            mutated[0, 0, 0] = np.nextafter(
                mutated[0, 0, 0],
                np.float32(1.0),
            )
            fits.PrimaryHDU(image).writeto(root / "stage3_bg_input.fit")
            fits.PrimaryHDU(mutated).writeto(root / "stage3_bgremoved.fit")
            points = [
                (
                    (cell_x + 0.5) / 4.0 * (width - 1),
                    (cell_y + 0.5) / 4.0 * (height - 1),
                )
                for cell_y in range(4)
                for cell_x in range(4)
            ]
            pipeline = SimpleNamespace(
                process_dir=root,
                work_dir=root / "test-run",
                _write_stage_json=lambda _name, _payload: None,
            )

            rejected = stage3_module._stage3_write_spatial_background_lineage(
                pipeline,
                baseline_image=image,
                fit_points=points[:12],
                validation_points=points[12:],
                patch_radius=3,
                support_mask=np.ones((height, width), dtype=bool),
                projection={},
                review_required=False,
                processing_route="verified_noop",
            )

            self.assertFalse(rejected["accepted"], rejected)
            self.assertIn("pixel identity mismatch", rejected["issues"][0])

    def test_stage3_final_consistency_warning_requires_review_without_rollback(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        original_save = processor._save_stage_output
        original_adaptive = processor._adaptive_features_current
        final_saved = False

        def mark_final_save(stem: str) -> bool:
            nonlocal final_saved
            saved = original_save(stem)
            if stem == "stage3_bgremoved":
                final_saved = True
            return saved

        def final_growth_metrics():
            metrics = original_adaptive()
            if final_saved:
                metrics["bg_std"] = 0.002
            return metrics

        processor._save_stage_output = mark_final_save
        processor._adaptive_features_current = final_growth_metrics
        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(processor.report["quality"], "review_required")
        self.assertEqual(processor.state, "compound")
        self.assertEqual(
            processor.report["final_output_validation"]["status"],
            "accepted",
        )

    def test_stage3_compound_validation_rejection_continues_to_plugin(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(
            compound_mode="validation_rejected",
            external_success=True,
        )
        graxpert_command = (
            "pyscript",
            "GraXpert-AI.py",
            "-bge",
            "-model",
            "model_v2_0_1",
            "-correction",
            "subtraction",
            "-keep_bg",
            "-nogpu",
        )
        with patch.object(
            stage3_module,
            "_stage3_theoretical_plugin_candidates",
            return_value=[("GraXpert-AI BGE CPU", graxpert_command, "graxpert")],
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "validation_rejected",
        )
        self.assertIn(graxpert_command, processor.cmd_calls)
        compound_rbf_index = max(
            index
            for index, call in enumerate(processor.cmd_calls)
            if call and call[0] == "subsky" and "-rbf" in call
        )
        self.assertLess(
            compound_rbf_index,
            processor.cmd_calls.index(graxpert_command),
        )
        self.assertEqual(
            processor.report["model_used"],
            "GraXpert-AI BGE CPU",
        )
        self.assertEqual(processor.results[-1][1], "ok")
        self.assertIn(
            {"context": "evaluated:subsky-poly-residual-rbf", "status": "restored"},
            processor.report["rollback_events"],
        )

    def test_stage3_single_hard_rejection_blocks_compound_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(external_success=True)

        original_pixel_gate = stage3_module._stage3_candidate_pixel_gate

        def pixel_gate(before: Any, after: Any, *, gate_profile: str):
            if processor.state == "polynomial":
                return False, {
                    "accepted": False,
                    "severity": "hard_reject",
                    "hard_issues": ["mock protected-signal rejection"],
                }
            return original_pixel_gate(before, after, gate_profile=gate_profile)

        with patch.object(
            stage3_module,
            "_stage3_candidate_pixel_gate",
            side_effect=pixel_gate,
        ):
            stage3_module.run_stage3_background_extraction(processor)

        compound = processor.report["compound_fallback"]
        self.assertEqual(compound["status"], "not_triggered")
        self.assertIn(
            "single_stage_hard_gate_rejection_present",
            compound["eligibility_issues"],
        )
        self.assertNotIn(
            "stage3_compound_poly_intermediate",
            processor.saved_states,
        )
        self.assertFalse(
            any(call and call[0] in {"gxp", "graxpert"} for call in processor.cmd_calls)
        )

    def test_stage3_compound_rollback_failure_invalidates_candidate(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake()
        original_cmd = processor.cmd_with_check

        def fail_compound_rollback(*args: Any, quiet: bool = False):
            if (
                tuple(args) == ("load", "stage3_bg_input")
                and "stage3_candidate_subsky_poly_residual_rbf"
                in processor.saved_states
            ):
                raise pipeline_module.CommandError("mock compound rollback failure")
            return original_cmd(*args, quiet=quiet)

        processor.cmd_with_check = fail_compound_rollback
        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "rollback_failed",
        )
        self.assertNotEqual(
            processor.report["model_used"],
            "subsky-poly-residual-rbf",
        )
        self.assertIn(
            {
                "context": "evaluated:subsky-poly-residual-rbf",
                "status": "failed",
                "reason": "mock compound rollback failure",
            },
            processor.report["rollback_events"],
        )

    def test_stage3_insufficient_compound_is_degraded_review_output(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3CompoundFake(compound_mode="insufficient")

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(
            processor.report["compound_fallback"]["status"],
            "accepted_degraded",
        )
        self.assertEqual(
            processor.report["model_used"],
            "subsky-poly-residual-rbf",
        )
        self.assertEqual(processor.report["quality"], "review_required")
        self.assertTrue(processor._background_review_required)
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertFalse(processor.result_metadata[-1]["fallback_used"])
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "compound_poly_residual_rbf_degraded_review",
        )

    def test_stage3_all_candidates_rejected_restores_baseline(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(gate_ok=False)

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [("rejected", ("subsky", "1"), "builtin")],
                ["rejected"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertEqual(processor.report["quality"], "review_required")
        self.assertTrue(processor.report["review_required"])
        self.assertEqual(processor.report["attempts"][0]["status"], "rejected")
        self.assertIn(
            {"context": "rejected:rejected", "status": "restored"},
            processor.report["rollback_events"],
        )

    def test_stage3_subsky_existing_fails_closed_without_safe_samples(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(gate_ok=True)
        processor.siril.set_image_bgsamples = lambda *_args, **_kwargs: False

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [
                    (
                        "safe-samples",
                        ("subsky", "-rbf", "-existing"),
                        "builtin",
                    )
                ],
                ["safe-samples"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertNotIn(
            ("subsky", "-rbf", "-existing"),
            processor.cmd_calls,
        )
        self.assertEqual(
            processor.report["attempts"][0]["status"],
            "safe_sample_install_failed",
        )
        self.assertTrue(processor.report["subsky_existing_enforced"])

    def test_stage3_safe_sample_install_rejects_collapsed_coverage(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": []}

        def set_samples(points, **_kwargs):
            state["samples"] = list(points)
            return True

        processor = SimpleNamespace(
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: state["samples"][:-1],
            ),
        )
        points = [(float(index), float(index)) for index in range(16)]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
        )

        self.assertFalse(accepted)
        self.assertEqual(report["status"], "failed")
        self.assertIn("collapsed audited sample coverage", report["reason"])
        self.assertEqual(state["samples"], [])

    def test_stage3_safe_sample_install_accepts_siril_recalculated_subset(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": []}

        def set_samples(points, **_kwargs):
            state["samples"] = list(points)
            return True

        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_safe_sample_min_count=16),
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: state["samples"][:-1],
            ),
        )
        points = [
            (float(x), float(y))
            for y in (20, 80, 140, 200)
            for x in (20, 85, 150, 215, 280)
        ]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
        )

        self.assertTrue(accepted, report)
        self.assertEqual(report["requested_count"], 20)
        self.assertEqual(report["observed_count"], 19)
        self.assertEqual(report["siril_rejected_count"], 1)
        self.assertGreaterEqual(report["observed_coverage"]["grid_cells"], 8)

    def test_stage3_masked_bg_samples_disable_recalculation_and_roundtrip(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": [], "recalculate": None}

        def set_samples(samples, **kwargs):
            state["samples"] = list(samples)
            state["recalculate"] = kwargs.get("recalculate")
            return True

        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_safe_sample_min_count=8),
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: list(state["samples"]),
            ),
        )
        points = [
            (float(x), float(y))
            for y in (20, 100, 180)
            for x in (20, 100, 180, 260)
        ][:10]
        records = [
            {
                "point": list(point),
                "sample_size": 25,
                "channel_count": 3,
                "channel_medians": [0.04, 0.05, 0.06],
                "native_luminance_mean": 0.05,
                "native_luminance_min": 0.03,
                "native_luminance_max": 0.07,
            }
            for point in points
        ]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
            minimum_count=8,
            sample_contract="dense_star_masked_fit",
            sample_records=records,
            masked_statistics=True,
        )

        self.assertTrue(accepted, report)
        self.assertFalse(state["recalculate"])
        self.assertTrue(report["roundtrip_verified"])
        self.assertEqual(
            report["statistics_mode"],
            "masked_native_channel_bg_sample",
        )
        self.assertIsNotNone(report["statistics_sha256"])

    def test_stage3_masked_bg_sample_statistic_drift_fails_closed(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": []}

        def set_samples(samples, **_kwargs):
            state["samples"] = list(samples)
            return True

        def drifted_samples():
            samples = list(state["samples"])
            if samples:
                samples[0].median = (0.20, 0.05, 0.06)
            return samples

        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_safe_sample_min_count=8),
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=drifted_samples,
            ),
        )
        points = [
            (float(x), float(y))
            for y in (20, 100, 180)
            for x in (20, 100, 180, 260)
        ][:10]
        records = [
            {
                "point": list(point),
                "sample_size": 25,
                "channel_count": 3,
                "channel_medians": [0.04, 0.05, 0.06],
                "native_luminance_mean": 0.05,
                "native_luminance_min": 0.03,
                "native_luminance_max": 0.07,
            }
            for point in points
        ]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
            minimum_count=8,
            sample_records=records,
            masked_statistics=True,
        )

        self.assertFalse(accepted)
        self.assertEqual(
            report["reason_code"],
            "stage3_dense_star_bg_sample_roundtrip_failed",
        )
        self.assertEqual(state["samples"], [])

    def test_stage3_masked_bg_samples_pad_sirilpy_144_transport(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": [], "transmitted_count": 0}

        def set_samples(samples, **_kwargs):
            state["transmitted_count"] = len(samples)
            state["samples"] = list(samples[:-1])
            return True

        set_samples.__module__ = "sirilpy.connection"
        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_safe_sample_min_count=8),
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: list(state["samples"]),
            ),
        )
        points = [
            (float(x), float(y))
            for y in (20, 100, 180)
            for x in (20, 100, 180, 260)
        ][:10]
        records = [
            {
                "point": list(point),
                "sample_size": 25,
                "channel_count": 3,
                "channel_medians": [0.04, 0.05, 0.06],
                "native_luminance_mean": 0.05,
                "native_luminance_min": 0.03,
                "native_luminance_max": 0.07,
            }
            for point in points
        ]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
            minimum_count=8,
            sample_records=records,
            masked_statistics=True,
        )

        self.assertTrue(accepted, report)
        self.assertEqual(state["transmitted_count"], len(points) + 1)
        self.assertTrue(report["transport_padding_applied"])
        self.assertEqual(report["transmitted_count"], len(points) + 1)
        self.assertEqual(report["observed_count"], len(points))

    def test_stage3_compound_fit_uses_its_own_sample_minimum(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        state = {"samples": []}

        def set_samples(points, **_kwargs):
            state["samples"] = list(points)
            return True

        processor = SimpleNamespace(
            cfg=SimpleNamespace(stage3_safe_sample_min_count=12),
            log=FakeLogger(),
            siril=SimpleNamespace(
                clear_image_bgsamples=lambda: state.update(samples=[]),
                set_image_bgsamples=set_samples,
                get_image_bgsamples=lambda: state["samples"][:-1],
            ),
        )
        points = [
            (float(x), float(y))
            for y in (20, 100, 180)
            for x in (20, 100, 180, 260)
        ][:10]

        accepted, report = stage3_module._stage3_install_safe_background_samples(
            processor,
            points,
            minimum_count=8,
            sample_contract="compound_fit",
        )

        self.assertTrue(accepted, report)
        self.assertEqual(report["observed_count"], 9)
        self.assertEqual(report["minimum_count"], 8)
        self.assertEqual(report["sample_contract"], "compound_fit")

    def test_stage3_pattern_noise_route_preserves_baseline_and_requires_review(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(gate_ok=True)
        processor.siril = Stage3SampleSiril()
        y = np.arange(processor.siril.image.shape[0], dtype=np.float64)[:, None]
        processor.siril.image += 0.025 * np.sin(2 * np.pi * y / 8)
        processor._adaptive_features_current = lambda: {
            "bg_std": 0.0001,
            "gradient_score": 0.03,
            "dirty_background_score": 0.08,
            "chroma_noise_score": 0.02,
            "red_dominance": 1.0,
            "blue_dominance": 1.0,
            "green_cast": 1.0,
        }

        stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertTrue(processor._background_review_required)
        self.assertEqual(
            processor.report["noise_route"]["route"],
            "mixed_gradient_and_pattern_noise",
        )
        self.assertEqual(
            processor.result_metadata[-1]["reason_code"],
            "no_background_candidate_accepted",
        )

    def test_stage3_selected_candidate_load_failure_restores_baseline(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = Stage3TransactionFake(
            gate_ok=True,
            fail_selected_load=True,
        )

        with patch.object(
            stage3_module,
            "_stage3_background_candidate_chain",
            return_value=(
                [("accepted", ("subsky", "1"), "builtin")],
                ["accepted"],
                "test",
            ),
        ):
            stage3_module.run_stage3_background_extraction(processor)

        self.assertEqual(processor.saved_sources["stage3_bgremoved"], "baseline")
        self.assertEqual(processor.results[-1][1], "degraded")
        self.assertIsNone(processor.report["model_used"])
        self.assertIn(
            {
                "context": "selected_load_failed:accepted",
                "status": "restored",
            },
            processor.report["rollback_events"],
        )

    def test_stage3_candidate_chain_respects_backend_and_plugin_controls(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        primary = (
            "GraXpert-AI BGE CPU",
            ("pyscript", "GraXpert-AI.py", "-bge", "-nogpu"),
            "graxpert",
        )
        plugin_backup = ("ADBE", ("adbe",), "plugin")
        scenarios = (
            ("auto_chain", True, 0, ["poly", "rbf", primary[0], "ADBE"]),
            ("builtin_only", True, 0, ["poly", "rbf"]),
            ("graxpert_only", True, 0, [primary[0]]),
            ("auto_chain", False, 0, ["poly", "rbf"]),
            ("auto_chain", True, 2, ["poly", "rbf"]),
        )

        for backend, plugins_enabled, limit, expected in scenarios:
            with self.subTest(
                backend=backend,
                plugins_enabled=plugins_enabled,
                limit=limit,
            ):
                processor = SimpleNamespace(
                    log=FakeLogger(),
                    cfg=SimpleNamespace(
                        stage3_backend_policy=backend,
                        stage3_plugin_fallback_enabled=plugins_enabled,
                        stage3_candidate_attempt_limit=limit,
                    ),
                )
                with patch.object(
                    stage3_module,
                    "_stage3_theoretical_plugin_candidates",
                    return_value=[primary, plugin_backup],
                ):
                    ordered, _builtin_labels, _reason = (
                        stage3_module._stage3_background_candidate_chain(
                            processor,
                            rbf_attempts=[
                                (
                                    "rbf",
                                    ("subsky", "-rbf", "-existing"),
                                    "builtin",
                                )
                            ],
                            poly_attempt=[
                                ("poly", ("subsky", "1", "-existing"), "builtin")
                            ],
                            poly_first=True,
                        )
                    )
                self.assertEqual(
                    [label for label, _command, _source in ordered],
                    expected,
                )

    def test_stage3_theoretical_plugins_require_integrated_scripts(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(log=FakeLogger())

        with (
            patch.object(stage3_module, "_stage3_graxpert_candidates", return_value=[]),
            patch.object(stage3_module, "_stage3_find_script", return_value=None),
        ):
            candidates = stage3_module._stage3_theoretical_plugin_candidates(processor)

        self.assertEqual(candidates, [])

    def test_stage3_autobge_success_without_image_change_is_rejected(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.fingerprints = iter(("unchanged", "unchanged"))
                self.cmd_calls: list[tuple[Any, ...]] = []

            def _validate_plugin_script_prerequisites(self, script_path: Path):
                self.validated_script = script_path
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
            '"/mock/siril scripts/processing/AutoBGE.py"',
        )

        ok, reason = stage3_module._stage3_try_background_command(
            processor,
            "AutoBGE",
            command,
            "plugin",
        )

        self.assertFalse(ok)
        self.assertEqual(
            processor.validated_script,
            Path("/mock/siril scripts/processing/AutoBGE.py"),
        )
        self.assertEqual(processor.cmd_calls, [command])
        self.assertEqual(
            reason,
            "plugin_runtime_error: command returned success but image did not change",
        )

    def test_stage3_evaluates_all_builtin_candidates_before_selecting(self):
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
                self.try_calls: list[tuple[str, ...]] = []
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
                        "gradient_score": 0.04,
                        "dirty_background_score": 0.18,
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
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return True

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

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
        with patch.object(
            stage3_module,
            "_stage3_candidate_pixel_gate",
            side_effect=_accepted_stage3_pixel_gate,
        ):
            stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertEqual(
            background_attempts,
            [
                ("subsky", "-rbf", "-existing"),
                ("subsky", "1", "-existing"),
            ],
        )
        self.assertIn(
            ("load", "stage3_candidate_subsky_rbf_existing_1"),
            processor.cmd_calls,
        )
        self.assertNotIn("GraXpert 背景提取", processor.workflow_command_used)

    def test_stage3_large_emission_nebula_tries_poly_before_rbf(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]

        class Stage3Fake(ReviewRegistryTestDouble):
            def __init__(self) -> None:
                self.log = FakeLogger()
                self.cfg = SimpleNamespace(
                    workflow_plugin_probe_enabled=False,
                )
                self.input_profile = {
                    "state": "linear",
                    "safe_for_linear_steps": True,
                    "source": "test_fixture",
                }
                self.target_profile = {
                    "target_type": "emission_nebula_widefield",
                    "object_stats": {"object_area_ratio": 0.46},
                }
                self.pipeline_policy = {
                    "policy_name": "test",
                    "stage3_background": {"protect_nebulosity": True},
                }
                self.siril = Stage3SampleSiril()
                self.try_calls: list[tuple[str, ...]] = []
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
                        "object_area_ratio": 0.46,
                        "nebulosity_area_ratio": 0.42,
                        "red_dominance": 1.00,
                        "blue_dominance": 1.00,
                        "green_cast": 1.00,
                    },
                    {
                        "bg_std": 0.00011,
                        "gradient_score": 0.11,
                        "dirty_background_score": 0.39,
                        "chroma_noise_score": 0.10,
                        "red_dominance": 1.03,
                        "blue_dominance": 1.02,
                        "green_cast": 0.98,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.02,
                        "dirty_background_score": 0.14,
                        "chroma_noise_score": 0.03,
                        "red_dominance": 1.01,
                        "blue_dominance": 1.01,
                        "green_cast": 0.99,
                    },
                    {
                        "bg_std": 0.00010,
                        "gradient_score": 0.02,
                        "dirty_background_score": 0.14,
                        "chroma_noise_score": 0.04,
                    },
                ]

            def cmd_with_check(self, *args: Any, quiet: bool = False) -> bool:
                _ = quiet
                self.cmd_calls.append(args)
                if args and args[0] not in ("save", "load", "subsky"):
                    raise RuntimeError(f"mock plugin unavailable: {args[0]}")
                return True

            def _try_cmd(self, *args: str) -> bool:
                self.try_calls.append(tuple(args))
                return args and args[0] == "subsky"

            def _stage3_subsky_rbf_candidates(self):
                return [("subsky", "-rbf")]

            def _stage3_measure_features(self, _label: str):
                return None

            def _stage3_signal_preservation_metrics(self, _before: Any, _after: Any):
                return {"available": False}

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
        with patch.object(
            stage3_module,
            "_stage3_candidate_pixel_gate",
            side_effect=_accepted_stage3_pixel_gate,
        ):
            stage3_module.run_stage3_background_extraction(processor)
        background_attempts = [
            tuple(call)
            for call in processor.cmd_calls
            if call and call[0] not in ("save", "load")
        ]

        self.assertLess(
            background_attempts.index(("subsky", "1", "-existing")),
            background_attempts.index(("subsky", "-rbf", "-existing")),
        )
        self.assertIn(
            ("load", "stage3_candidate_subsky_rbf_existing_1"),
            processor.cmd_calls,
        )
        self.assertEqual(
            processor.report["builtin_order_reason"],
            "diffuse_signal_safe_samples_poly_before_rbf",
        )
        self.assertEqual(
            processor.report["builtin_search_mode"],
            "safe_samples_with_diffuse_signal_protection",
        )

    def test_stage3_decision_skips_clean_background(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.02,
                "dirty_background_score": 0.08,
                "chroma_noise_score": 0.01,
            },
            diffuse_context={"diffuse": False},
        )

        self.assertEqual(decision["decision"], "review_required")
        self.assertEqual(decision["source"], "process_evidence")

    def test_stage3_decision_requires_review_for_diffuse_signal(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={"target_type": "emission_nebula_widefield"},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.20,
                "dirty_background_score": 0.40,
                "chroma_noise_score": 0.08,
            },
            diffuse_context={
                "diffuse": True,
                "emission_diffuse": True,
            },
        )

        self.assertEqual(decision["decision"], "review_required")

    def test_stage3_decision_skips_low_dirty_gradient_when_diffuse_signal_is_protected(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={"target_type": "emission_nebula_widefield"},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.13,
                "dirty_background_score": 0.088,
                "chroma_noise_score": 0.15,
            },
            diffuse_context={
                "diffuse": True,
                "emission_diffuse": True,
                "pixel_signal_protection": True,
            },
        )

        self.assertEqual(decision["decision"], "review_required")
        self.assertEqual(decision["source"], "process_evidence")
        self.assertEqual(decision["confidence"], 0.0)

    def test_stage3_decision_applies_high_confidence_offline_gradient(self):
        stage3_module = sys.modules["stages.stage3_background_extraction"]
        processor = SimpleNamespace(
            cfg=pipeline_module.PipelineConfig(),
            target_profile={},
        )

        decision = stage3_module._stage3_background_decision(
            processor,
            {
                "gradient_score": 0.14,
                "dirty_background_score": 0.32,
                "chroma_noise_score": 0.04,
            },
            diffuse_context={"diffuse": False},
        )

        self.assertEqual(decision["decision"], "review_required")
        self.assertEqual(decision["source"], "process_evidence")

    def test_stage3_dynamic_rbf_candidates_expand_for_noisy_complex_fields(self):
        processor = pipeline_module.StarunPostProcessor()
        processor.log = FakeLogger()
        processor.siril = SimpleNamespace(get_image_pixeldata=lambda preview=False: object())

        with patch.object(
            sys.modules["stage_support"],
            "measure_image_features",
            return_value=pipeline_module.ImageFeatures(
                bg_std=0.070,
                star_density=0.006,
                object_area_ratio=0.42,
            ),
        ):
            candidates = processor._stage3_subsky_rbf_candidates()

        self.assertGreaterEqual(len(candidates), 4)
        command_text = [" ".join(cmd) for cmd in candidates]
        self.assertTrue(any("-smooth=1.000" in text or "-smooth=1.200" in text for text in command_text))
        self.assertTrue(any("-tolerance=0.800" in text or "-tolerance=0.700" in text for text in command_text))
        self.assertTrue(all("-existing" in command for command in candidates))
