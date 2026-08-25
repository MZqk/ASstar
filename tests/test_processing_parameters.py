#!/usr/bin/env python3
"""Tests for the shared Stage 1-10 processing-parameter contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.models import PipelineConfig
from pipeline.task_plan import build_resume_fingerprints
from pipeline import run_manifest
from pipeline.task_workspace import (
    WorkspaceError,
    begin_task_run,
    build_source_record,
    ensure_task_workspace,
)
from gui.task_intake import stage_config_from_processing_settings
from pipeline.processing_parameters import (
    GATE_PROFILE_DEFAULT,
    GATE_PROFILE_PARAMETER_SPECS,
    GATE_PROFILE_RELAXED,
    GATE_PROFILE_UNLIMITED,
    LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4,
    LEGACY_PROCESSING_PARAMETERS_SCHEMA_V5,
    PROCESSING_PARAMETERS_SCHEMA,
    PROCESSING_GATE_PARAMETER_SPECS,
    PROCESSING_PARAMETER_SPECS,
    SPECS_BY_STAGE,
    SPECS_BY_FIELD,
    apply_processing_parameters_to_config,
    default_processing_parameters,
    effective_parameter_value,
    gate_profile_requires_review,
    normalize_processing_parameters,
    processing_gate_profile_audit,
    reset_stage_parameters,
)


class ProcessingParameterContractTests(unittest.TestCase):
    def test_current_registry_and_failure_actions(self) -> None:
        payload = default_processing_parameters()
        self.assertEqual(payload["schema"], PROCESSING_PARAMETERS_SCHEMA)
        for stage in range(2, 10):
            spec = SPECS_BY_FIELD[f"stage{stage}_failure_action"]
            self.assertEqual(spec.stage, stage)
            self.assertEqual(spec.section, "fallback")
            self.assertEqual(
                {value for _label, value in spec.choices},
                {"auto_fallback", "preserve_review", "stop"},
            )
        self.assertEqual(SPECS_BY_FIELD["stage10_failure_action"].stage, 10)
        self.assertNotIn("crop_margin", SPECS_BY_FIELD)
        self.assertEqual(
            SPECS_BY_FIELD["stage2_base_crop_margin"].depends_on,
            (("stage2_base_crop_enabled", (True,)),),
        )
        self.assertTrue(
            SPECS_BY_FIELD["stage2_field_rotation_detection_enabled"].default
        )
        self.assertEqual(
            SPECS_BY_FIELD["stage2_field_rotation_noise_ratio_min"].default,
            1.35,
        )
        self.assertEqual(
            SPECS_BY_FIELD["stage2_field_rotation_chroma_ratio_min"].depends_on,
            (("stage2_field_rotation_detection_enabled", (True,)),),
        )
        self.assertEqual(
            SPECS_BY_FIELD["stage5_deconv_bg_std_growth_max"].strictness,
            "lower_is_stricter",
        )
        palette_spec = SPECS_BY_FIELD["stage8_dualband_palette_selection"]
        self.assertEqual(palette_spec.default, "auto")
        self.assertEqual(
            [value for _label, value in palette_spec.choices],
            ["auto", "HSO", "SHO", "OSH", "OHS", "HOS", "HOO"],
        )
        self.assertEqual(
            palette_spec.depends_on,
            (
                ("stage8_processing_mode", ("auto",)),
                ("stage8_dualband_palette_enabled", (True,)),
            ),
        )
        offline_color = SPECS_BY_FIELD["stage4_offline_fallback_mode"]
        self.assertEqual(offline_color.level, "recommended")
        self.assertEqual(offline_color.default, "auto_local_reference")
        self.assertEqual(
            {value for _label, value in offline_color.choices},
            {"auto_local_reference", "preserve"},
        )
        self.assertTrue(
            SPECS_BY_FIELD[
                "stage4_auto_reference_global_white_enabled"
            ].default
        )

    def test_stage5_auto_denoise_is_enabled_by_default(self) -> None:
        payload = default_processing_parameters()

        self.assertTrue(PipelineConfig().denoise_enabled)
        self.assertTrue(SPECS_BY_FIELD["denoise_enabled"].default)
        self.assertTrue(effective_parameter_value(payload, "denoise_enabled"))

    def test_explicit_stage5_denoise_disable_is_a_manual_override(self) -> None:
        cfg = PipelineConfig()
        payload = default_processing_parameters()
        payload["stages"]["5"]["overrides"]["denoise_enabled"] = False

        _normalized, _adjustments, fields = apply_processing_parameters_to_config(
            cfg,
            payload,
        )

        self.assertFalse(cfg.denoise_enabled)
        self.assertIn("denoise_enabled", fields)

    def test_v1_through_v3_are_rejected_without_migration(self) -> None:
        for schema in (
            "starun.processing-parameters.v1",
            "starun.processing-parameters.v2",
            "starun.processing-parameters.v3",
        ):
            with self.subTest(schema=schema):
                payload = default_processing_parameters()
                payload["schema"] = schema
                with self.assertRaisesRegex(ValueError, "v4"):
                    normalize_processing_parameters(payload)

    def test_v4_explicit_compact_override_migrates_to_pre_stretch_compaction(
        self,
    ) -> None:
        payload = default_processing_parameters()
        payload["schema"] = LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4
        payload["stages"]["9"]["overrides"][
            "stage9_compact_starmask_enabled"
        ] = True

        normalized, adjustments = normalize_processing_parameters(payload)

        self.assertEqual(normalized["schema"], PROCESSING_PARAMETERS_SCHEMA)
        self.assertTrue(
            normalized["stages"]["9"]["overrides"][
                "stage9_compact_starmask_enabled"
            ]
        )
        self.assertTrue(
            normalized["stages"]["9"]["overrides"][
                "stage9_starmask_pre_stretch_compact_enabled"
            ]
        )
        self.assertIn(
            "v4_explicit_compact_starmask_migration",
            {str(item.get("reason") or "") for item in adjustments},
        )

        cfg = PipelineConfig()
        apply_processing_parameters_to_config(cfg, payload)
        self.assertTrue(cfg.stage9_compact_starmask_enabled)
        self.assertTrue(cfg.stage9_starmask_pre_stretch_compact_enabled)

    def test_v4_implicit_and_v5_explicit_support_keep_new_default_off(
        self,
    ) -> None:
        legacy = default_processing_parameters()
        legacy["schema"] = LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4
        legacy_normalized, _adjustments = normalize_processing_parameters(
            legacy
        )
        self.assertNotIn(
            "stage9_starmask_pre_stretch_compact_enabled",
            legacy_normalized["stages"]["9"]["overrides"],
        )

        current = default_processing_parameters()
        current["stages"]["9"]["overrides"][
            "stage9_compact_starmask_enabled"
        ] = True
        cfg = PipelineConfig()
        apply_processing_parameters_to_config(cfg, current)
        self.assertTrue(cfg.stage9_compact_starmask_enabled)
        self.assertFalse(cfg.stage9_starmask_pre_stretch_compact_enabled)

    def test_v4_v5_stage3_legacy_fields_migrate_but_v6_rejects(self) -> None:
        for schema in (
            LEGACY_PROCESSING_PARAMETERS_SCHEMA_V4,
            LEGACY_PROCESSING_PARAMETERS_SCHEMA_V5,
        ):
            with self.subTest(schema=schema):
                payload = default_processing_parameters()
                payload["schema"] = schema
                payload["stages"]["3"]["overrides"].update(
                    {
                        "bg_quality_gate_enabled": False,
                        "stage3_apply_confidence_min": 0.60,
                    }
                )
                normalized, adjustments = normalize_processing_parameters(payload)
                self.assertEqual(normalized["schema"], PROCESSING_PARAMETERS_SCHEMA)
                self.assertEqual(normalized["stages"]["3"]["overrides"], {})
                self.assertEqual(
                    {item["field"] for item in adjustments},
                    {"bg_quality_gate_enabled", "stage3_apply_confidence_min"},
                )

        current = default_processing_parameters()
        current["stages"]["3"]["overrides"][
            "bg_quality_gate_enabled"
        ] = False
        with self.assertRaisesRegex(ValueError, "未知参数"):
            normalize_processing_parameters(current)

    def test_v6_contract_includes_general_stage1_stage9_and_stage10_controls(self) -> None:
        payload = default_processing_parameters()
        self.assertEqual(payload["schema"], PROCESSING_PARAMETERS_SCHEMA)
        self.assertEqual(set(payload["stages"]), {str(i) for i in range(1, 11)})
        self.assertEqual(payload["stages"]["1"]["mode"], "auto")
        self.assertEqual(
            payload["general"],
            {
                "output_formats": ["tif", "png", "fit"],
                "review_only": False,
                "compute_mode": "auto",
                "auto_tune_enabled": True,
                "max_retries": 2,
                "retry_delay": 1.0,
                "review_bundle_enabled": True,
                "managed_output_enabled": True,
                "checkpoint_mode": False,
            },
        )
        payload["general"].update({"max_retries": 8, "retry_delay": -3.0})
        payload["stages"]["1"]["overrides"][
            "stage1_register_fail_ratio_max"
        ] = 0.8
        payload["stages"]["9"]["overrides"].update(
            {
                "stage9_psf_recovery_target_min": 0.2,
                "stage9_psf_recovery_target_max": 2.0,
            }
        )
        payload["stages"]["10"].update(
            {
                "mode": "preserve",
                "overrides": {
                    "stage10_denoise_backend_policy": "scunet_only",
                    "final_saturation": -1.0,
                },
            }
        )

        normalized, adjustments = normalize_processing_parameters(payload)

        self.assertEqual(normalized["general"]["max_retries"], 3)
        self.assertEqual(normalized["general"]["retry_delay"], 0.0)
        self.assertEqual(
            normalized["stages"]["1"]["overrides"][
                "stage1_register_fail_ratio_max"
            ],
            0.50,
        )
        self.assertEqual(
            normalized["stages"]["9"]["overrides"][
                "stage9_psf_recovery_target_min"
            ],
            0.50,
        )
        self.assertEqual(
            normalized["stages"]["9"]["overrides"][
                "stage9_psf_recovery_target_max"
            ],
            1.50,
        )
        self.assertEqual(normalized["stages"]["10"]["mode"], "preserve")
        self.assertEqual(
            normalized["stages"]["10"]["overrides"]["final_saturation"],
            0.0,
        )
        self.assertEqual(len(adjustments), 6)

    def test_palette_selection_normalizes_in_current_contract(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["8"]["overrides"][
            "stage8_dualband_palette_selection"
        ] = "  ohs  "

        normalized, adjustments = normalize_processing_parameters(payload)

        self.assertEqual(
            normalized["stages"]["8"]["overrides"][
                "stage8_dualband_palette_selection"
            ],
            "OHS",
        )
        self.assertEqual(adjustments, [])
        cfg = PipelineConfig()
        _normalized, _adjustments, fields = apply_processing_parameters_to_config(
            cfg,
            normalized,
        )
        self.assertEqual(cfg.stage8_dualband_palette_selection, "OHS")
        self.assertIn("stage8_dualband_palette_selection", fields)

        invalid = default_processing_parameters()
        invalid["stages"]["8"]["overrides"][
            "stage8_dualband_palette_selection"
        ] = "XYZ"
        with self.assertRaisesRegex(ValueError, "选项不受支持"):
            normalize_processing_parameters(invalid)

    def test_current_operational_ranges_and_enums_are_fail_closed(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["2"]["overrides"].update(
            {
                "stage2_level_artifact_window": 80,
                "stage2_base_crop_margin": 0.5,
            }
        )
        payload["stages"]["9"]["overrides"].update(
            {
                "stage9_fallback_retry_max": 99,
                "stage9_fallback_intensity_floor": 0.1,
            }
        )

        normalized, adjustments = normalize_processing_parameters(payload)

        self.assertEqual(
            normalized["stages"]["2"]["overrides"][
                "stage2_level_artifact_window"
            ],
            81,
        )
        self.assertEqual(
            normalized["stages"]["2"]["overrides"][
                "stage2_base_crop_margin"
            ],
            0.06,
        )
        self.assertEqual(
            normalized["stages"]["9"]["overrides"][
                "stage9_fallback_retry_max"
            ],
            3,
        )
        self.assertEqual(
            normalized["stages"]["9"]["overrides"][
                "stage9_fallback_intensity_floor"
            ],
            0.40,
        )
        self.assertEqual(len(adjustments), 4)

        invalid = default_processing_parameters()
        invalid["stages"]["8"]["overrides"]["stage8_failure_action"] = (
            "force_accept"
        )
        with self.assertRaisesRegex(ValueError, "选项不受支持"):
            normalize_processing_parameters(invalid)

    def test_registry_covers_every_configurable_stage_and_both_levels(self) -> None:
        self.assertEqual(set(SPECS_BY_STAGE), set(range(1, 11)))
        self.assertGreaterEqual(len(PROCESSING_PARAMETER_SPECS), 250)
        self.assertEqual(
            len(PROCESSING_PARAMETER_SPECS),
            len({spec.field for spec in PROCESSING_PARAMETER_SPECS}),
        )
        for stage, specs in SPECS_BY_STAGE.items():
            if stage == 1:
                continue
            self.assertTrue(
                any(spec.level == "recommended" for spec in specs),
                f"Stage {stage} has no recommended parameters",
            )
        self.assertTrue(PROCESSING_GATE_PARAMETER_SPECS)
        self.assertTrue(
            all(spec.level == "expert" for spec in PROCESSING_GATE_PARAMETER_SPECS)
        )
        self.assertEqual(
            {spec.stage for spec in PROCESSING_GATE_PARAMETER_SPECS},
            set(range(2, 10)),
        )
        self.assertEqual(
            {
                stage: sum(
                    spec.stage == stage
                    for spec in PROCESSING_GATE_PARAMETER_SPECS
                )
                for stage in range(2, 10)
            },
            {2: 4, 3: 18, 4: 20, 5: 4, 6: 32, 7: 51, 8: 17, 9: 84},
        )
        gate_fields = {spec.field for spec in PROCESSING_GATE_PARAMETER_SPECS}
        self.assertTrue(
            {
                "stage2_edge_black_target",
                "stage3_safe_sample_target_count",
                "stage4_auto_reference_background_sample_target",
                "stage4_auto_reference_star_min_objects",
                "stage4_auto_reference_target_chroma_drift_max",
                "stage5_multiscale_detail_retention_min",
                "stage7_residual_star_score_max",
                "stage7_9_quality_advisory_multiplier",
                "stage7_stretch_chroma_noise_score_max",
                "stage8_texture_artifact_growth_max",
                "stage8_subject_boundary_luma_residual_max",
                "stage8_subject_boundary_chroma_residual_max",
                "stage8_subject_boundary_residual_ratio_max",
                "stage9_quality_gate_enabled",
                "stage9_core_color_jump_min",
                "stage9_unscreen_denominator_floor",
                "stage9_unscreen_reliable_support_min",
                "stage9_unscreen_roundtrip_relative_improvement_min",
                "stage9_psf_selective_wing_enabled",
                "stage9_psf_selective_wing_target_ratio",
                "stage9_psf_selective_wing_strength_max",
            }.issubset(gate_fields)
        )
        self.assertNotIn("stage4_pcc_quality_gate_enabled", gate_fields)
        self.assertTrue(
            {
                "stage4_pcc_channel_gain_ratio_max",
                "stage4_pcc_emission_balance_gain_ratio_max",
                "stage4_pcc_clip_growth_max",
                "stage4_pcc_star_temperature_ratio_min",
                "stage4_pcc_star_temperature_ratio_max",
                "stage4_pcc_background_color_delta_max",
                "stage4_pcc_target_color_drift_max",
                "stage4_pcc_emission_target_color_drift_max",
            }.isdisjoint(gate_fields)
        )
        self.assertFalse(
            {
                "stage4_local_star_wb_min_pixels",
                "stage4_local_star_mask_radius",
                "stage4_local_star_mask_coverage_max",
            }
            & gate_fields
        )
        self.assertTrue(
            {
                "bg_quality_gate_enabled",
                "stage3_conditional_decision_enabled",
                "stage3_deterministic_auto_apply_enabled",
                "stage3_apply_confidence_min",
            }.isdisjoint(gate_fields)
        )
        self.assertTrue(
            {
                "bg_std_worsen_ratio_max",
                "stage3_gradient_skip_max",
                "stage9_fallback_intensity_cap",
            }.isdisjoint(gate_fields)
        )

    def test_canonical_defaults_match_pipeline_config(self) -> None:
        cfg = PipelineConfig()
        for spec in PROCESSING_PARAMETER_SPECS:
            self.assertTrue(hasattr(cfg, spec.field), spec.field)
            self.assertEqual(spec.default, getattr(cfg, spec.field), spec.field)
            if spec.kind == "bool":
                self.assertIs(type(spec.default), bool, spec.field)
            elif spec.kind == "int":
                self.assertIs(type(spec.default), int, spec.field)
            elif spec.kind == "float":
                self.assertIsInstance(spec.default, (int, float), spec.field)
                self.assertIsNot(type(spec.default), bool, spec.field)
            else:
                self.assertIsInstance(spec.default, str, spec.field)
        self.assertEqual(cfg.denoise_mod, 0.35)
        self.assertEqual(cfg.stage4_spcc_timeout_sec, 300)
        self.assertEqual(cfg.stage4_spcc_online_unverified_timeout_sec, 300)
        self.assertEqual(cfg.stage4_pcc_timeout_sec, 180)
        self.assertEqual(cfg.stage3_gate_profile, "output_first")
        safe_stage9_defaults = {
            "star_intensity": 1.05,
            "stage9_compact_starmask_enabled": True,
            "stage9_starmask_pre_stretch_compact_enabled": False,
            "stage9_psf_fwhm_ratio_min": 0.93,
            "stage9_psf_review_fwhm_ratio_max": 1.65,
            "stage9_psf_support_radius_max": 6,
            "stage9_psf_support_retry_pixels": 2,
            "stage9_psf_selective_wing_enabled": True,
            "stage9_psf_selective_wing_target_ratio": 1.08,
            "stage9_psf_selective_wing_strength_max": 1.15,
            "stage9_targeted_recovery_enabled": True,
            "stage9_targeted_recovery_retry_max": 3,
            "stage9_source_autostretch_wing_reference_enabled": True,
            "stage9_source_autostretch_wing_floor_fraction": 0.05,
            "stage9_source_autostretch_wing_target_ratio": 1.03,
            "stage9_source_autostretch_wing_radius_max": 10,
            "stage9_weak_star_screen_intensity_min": 0.55,
        }
        for field, expected in safe_stage9_defaults.items():
            self.assertEqual(getattr(cfg, field), expected, field)
            self.assertEqual(SPECS_BY_FIELD[field].default, expected, field)
        self.assertEqual(
            SPECS_BY_FIELD["stage3_gate_profile"].choices,
            (
                ("输出优先", "output_first"),
                ("平衡", "balanced"),
                ("严格保真", "strict"),
            ),
        )
        payload = default_processing_parameters()
        self.assertEqual(payload["stages"]["2"]["mode"], "auto")
        self.assertEqual(payload["stages"]["9"]["overrides"], {})
        for field, expected in safe_stage9_defaults.items():
            self.assertEqual(
                effective_parameter_value(payload, field),
                expected,
                field,
            )
        frozen_without_gate_profile = default_processing_parameters()
        applied = PipelineConfig()
        apply_processing_parameters_to_config(
            applied,
            frozen_without_gate_profile,
        )
        self.assertEqual(applied.stage3_gate_profile, "output_first")

    def test_task_gate_profiles_scale_static_defaults_and_expert_wins(self) -> None:
        payload = default_processing_parameters()
        payload["gate_profile"] = GATE_PROFILE_RELAXED
        cfg = PipelineConfig()
        cfg.stage8_mask_signal_coverage_min = 0.001
        cfg.stage8_texture_artifact_growth_max = 1.10

        apply_processing_parameters_to_config(cfg, payload)

        self.assertAlmostEqual(
            cfg.stage8_mask_signal_coverage_min,
            PipelineConfig().stage8_mask_signal_coverage_min / 3.0,
        )
        self.assertAlmostEqual(
            cfg.stage8_texture_artifact_growth_max,
            1.0
            + (PipelineConfig().stage8_texture_artifact_growth_max - 1.0) * 3.0,
        )
        self.assertAlmostEqual(
            cfg.stage7_diffuse_visibility_score_min,
            PipelineConfig().stage7_diffuse_visibility_score_min / 3.0,
        )
        self.assertEqual(
            cfg.stage7_highlight_clip_ratio_max,
            PipelineConfig().stage7_highlight_clip_ratio_max,
        )
        self.assertEqual(
            cfg.stage7_stretch_chroma_noise_score_max,
            PipelineConfig().stage7_stretch_chroma_noise_score_max,
        )
        self.assertEqual(
            cfg.stage7_transform_new_hard_clip_ratio_max,
            PipelineConfig().stage7_transform_new_hard_clip_ratio_max,
        )
        self.assertEqual(cfg.stage9_mixed_star_weak_count_min, 6)
        self.assertEqual(cfg.stage7_starless_repair_max_score_growth, 0.0)
        self.assertAlmostEqual(cfg.stage9_psf_fwhm_ratio_min, 0.93 / 3.0)
        self.assertEqual(cfg.stage9_psf_review_fwhm_ratio_max, 1.65)
        self.assertAlmostEqual(
            cfg.stage9_weak_star_screen_intensity_min,
            0.55 / 3.0,
        )

        payload["gate_profile"] = GATE_PROFILE_UNLIMITED
        payload["stages"]["5"]["overrides"][
            "stage5_multiscale_detail_retention_min"
        ] = 0.90
        payload["stages"]["9"]["overrides"][
            "stage9_psf_review_fwhm_ratio_max"
        ] = 1.40
        apply_processing_parameters_to_config(cfg, payload)
        audit = processing_gate_profile_audit(payload)
        records = {record["field"]: record for record in audit["fields"]}

        self.assertEqual(cfg.stage5_multiscale_detail_retention_min, 0.90)
        self.assertEqual(cfg.stage9_changed_pixel_ratio_max, 1.0)
        self.assertAlmostEqual(cfg.stage9_psf_fwhm_ratio_min, 0.93 / 10.0)
        self.assertEqual(cfg.stage9_psf_review_fwhm_ratio_max, 1.40)
        self.assertAlmostEqual(
            cfg.stage9_weak_star_screen_intensity_min,
            0.55 / 10.0,
        )
        self.assertGreater(
            cfg.stage9_highlight_clip_ratio_max,
            SPECS_BY_FIELD["stage9_highlight_clip_ratio_max"].maximum,
        )
        self.assertEqual(
            records["stage5_multiscale_detail_retention_min"]["source"],
            "expert_override",
        )
        self.assertEqual(
            records["stage9_psf_fwhm_ratio_min"]["static_baseline"],
            0.93,
        )
        self.assertEqual(
            records["stage9_psf_fwhm_ratio_min"]["source"],
            "gate_profile",
        )
        self.assertNotIn("stage9_psf_review_fwhm_ratio_max", records)
        self.assertEqual(
            records["stage9_weak_star_screen_intensity_min"]["static_baseline"],
            0.55,
        )
        self.assertTrue(
            records["stage9_changed_pixel_ratio_max"]["physical_clamped"]
        )
        self.assertTrue(gate_profile_requires_review(GATE_PROFILE_UNLIMITED))

    def test_gate_profile_registry_is_explicit_and_numeric_only(self) -> None:
        self.assertGreater(len(GATE_PROFILE_PARAMETER_SPECS), 0)
        self.assertTrue(
            all(spec.kind in {"int", "float"} for spec in GATE_PROFILE_PARAMETER_SPECS)
        )
        self.assertEqual(
            SPECS_BY_FIELD["stage7_starmask_coverage_min_ratio"].profile_scaling,
            "lower",
        )
        self.assertEqual(
            SPECS_BY_FIELD["stage7_starmask_background_floor_percentile"].profile_scaling,
            "none",
        )
        self.assertEqual(
            SPECS_BY_FIELD["stage9_quality_gate_enabled"].profile_scaling,
            "none",
        )

    def test_stage4_advisory_diagnostics_ignore_gate_profiles(self) -> None:
        diagnostic_fields = (
            "stage4_pcc_channel_gain_ratio_max",
            "stage4_pcc_emission_balance_gain_ratio_max",
            "stage4_pcc_clip_growth_max",
            "stage4_pcc_star_temperature_ratio_min",
            "stage4_pcc_star_temperature_ratio_max",
            "stage4_pcc_background_color_delta_max",
            "stage4_pcc_target_color_drift_max",
            "stage4_pcc_emission_target_color_drift_max",
        )
        defaults = PipelineConfig()
        for profile in (GATE_PROFILE_RELAXED, GATE_PROFILE_UNLIMITED):
            payload = default_processing_parameters()
            payload["gate_profile"] = profile
            cfg = PipelineConfig()
            apply_processing_parameters_to_config(cfg, payload)
            for field in diagnostic_fields:
                self.assertEqual(getattr(cfg, field), getattr(defaults, field))
                self.assertEqual(SPECS_BY_FIELD[field].section, "diagnostics")
                self.assertEqual(SPECS_BY_FIELD[field].profile_scaling, "none")

        payload = default_processing_parameters()
        payload["gate_profile"] = GATE_PROFILE_UNLIMITED
        payload["stages"]["4"]["overrides"][
            "stage4_pcc_clip_growth_max"
        ] = 0.010
        cfg = PipelineConfig()
        apply_processing_parameters_to_config(cfg, payload)
        self.assertEqual(cfg.stage4_pcc_clip_growth_max, 0.010)

    def test_unknown_fields_and_schema_are_rejected(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["5"]["overrides"]["stage5_quality_gate"] = 1
        with self.assertRaisesRegex(ValueError, "未知参数"):
            normalize_processing_parameters(payload)
        payload = default_processing_parameters()
        payload["schema"] = "starun.processing-parameters.v99"
        with self.assertRaisesRegex(ValueError, "schema"):
            normalize_processing_parameters(payload)

    def test_legacy_stage2_engine_policy_is_rejected(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["2"]["overrides"][
            "stage2_crop_engine_policy"
        ] = "autocrop_only"
        self.assertNotIn("stage2_crop_engine_policy", SPECS_BY_FIELD)
        with self.assertRaisesRegex(ValueError, "未知参数"):
            normalize_processing_parameters(payload)

    def test_spcc_enums_reject_arbitrary_command_text(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["4"]["overrides"]["stage4_spcc_osc_sensor"] = (
            'Sony IMX585" -catalog=remote'
        )
        with self.assertRaisesRegex(ValueError, "选项不受支持"):
            normalize_processing_parameters(payload)

    def test_numeric_values_are_clamped_and_psf_kernel_is_odd(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["4"]["overrides"].update(
            {
                "stage4_spcc_narrowband_r_wavelength_nm": 1200,
                "stage4_spcc_narrowband_r_bandwidth_nm": 0.2,
            }
        )
        payload["stages"]["5"]["overrides"]["stage5_rl_psf_kernel_size"] = 40

        normalized, adjustments = normalize_processing_parameters(payload)

        self.assertEqual(
            normalized["stages"]["4"]["overrides"][
                "stage4_spcc_narrowband_r_wavelength_nm"
            ],
            900.0,
        )
        self.assertEqual(
            normalized["stages"]["4"]["overrides"][
                "stage4_spcc_narrowband_r_bandwidth_nm"
            ],
            1.0,
        )
        self.assertEqual(
            normalized["stages"]["5"]["overrides"]["stage5_rl_psf_kernel_size"],
            41,
        )
        self.assertEqual(len(adjustments), 3)

    def test_gate_thresholds_are_clamped_and_serialized_as_expert_overrides(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["2"]["overrides"]["stage2_edge_black_target"] = 0.50
        payload["stages"]["8"]["overrides"][
            "stage8_texture_artifact_growth_max"
        ] = 9.0
        payload["stages"]["9"]["overrides"][
            "stage9_local_component_area_max"
        ] = 9000

        normalized, adjustments = normalize_processing_parameters(payload)

        self.assertEqual(
            normalized["stages"]["2"]["overrides"]["stage2_edge_black_target"],
            0.18,
        )
        self.assertEqual(
            normalized["stages"]["8"]["overrides"][
                "stage8_texture_artifact_growth_max"
            ],
            2.20,
        )
        self.assertEqual(
            normalized["stages"]["9"]["overrides"][
                "stage9_local_component_area_max"
            ],
            4096,
        )
        self.assertEqual(len(adjustments), 3)

    def test_non_finite_numeric_values_are_rejected(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["7"]["overrides"]["asinh_stretch"] = float("nan")
        with self.assertRaisesRegex(ValueError, "有限数值"):
            normalize_processing_parameters(payload)

    def test_model_override_requires_a_readable_local_file(self) -> None:
        payload = default_processing_parameters()
        payload["stages"]["5"]["overrides"]["graxpert_object_model_path"] = (
            "/does/not/exist/model.onnx"
        )
        with self.assertRaisesRegex(ValueError, "可读的本地文件"):
            normalize_processing_parameters(payload, validate_paths=True)
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = Path(temporary_dir) / "model.onnx"
            model.write_bytes(b"onnx")
            payload["stages"]["5"]["overrides"][
                "graxpert_object_model_path"
            ] = str(model)
            normalized, _adjustments = normalize_processing_parameters(
                payload,
                validate_paths=True,
            )
        self.assertEqual(
            normalized["stages"]["5"]["overrides"][
                "graxpert_object_model_path"
            ],
            str(model.resolve()),
        )

    def test_explicit_values_override_auto_tuned_config(self) -> None:
        cfg = PipelineConfig()
        cfg.denoise_mod = 0.22
        cfg.asinh_stretch = 1.8
        cfg.star_intensity = 0.88
        cfg.stage8_texture_artifact_growth_max = 1.10
        payload = default_processing_parameters()
        payload["stages"]["2"]["mode"] = "preserve"
        payload["stages"]["5"]["overrides"].update(
            {
                "denoise_enabled": True,
                "denoise_mod": 0.50,
                "stage5_deconvolution_mode": "rl",
            }
        )
        payload["stages"]["7"]["overrides"]["asinh_stretch"] = 3.25
        payload["stages"]["9"]["overrides"]["star_intensity"] = 0.95
        payload["stages"]["8"]["overrides"][
            "stage8_texture_artifact_growth_max"
        ] = 1.80

        _normalized, _adjustments, fields = apply_processing_parameters_to_config(
            cfg, payload
        )

        self.assertEqual(cfg.stage2_processing_mode, "preserve")
        self.assertTrue(cfg.denoise_enabled)
        self.assertEqual(cfg.denoise_mod, 0.50)
        self.assertTrue(cfg.stage5_deconvolution_enabled)
        self.assertFalse(cfg.stage5_graxpert_deconvolution_enabled)
        self.assertEqual(cfg.asinh_stretch, 3.25)
        self.assertEqual(cfg.star_intensity, 0.95)
        self.assertEqual(cfg.stage8_texture_artifact_growth_max, 1.80)
        self.assertIn("asinh_stretch", fields)
        self.assertIn("star_intensity", fields)
        self.assertIn("stage8_texture_artifact_growth_max", fields)

    def test_stage_reset_preserves_general_and_other_stages(self) -> None:
        payload = default_processing_parameters(
            general={
                "output_formats": ["fit"],
                "compute_mode": "cpu",
                "max_retries": 3,
            }
        )
        payload["stages"]["1"]["overrides"][
            "stage1_register_fail_ratio_max"
        ] = 0.20
        payload["stages"]["4"]["mode"] = "preserve"
        payload["stages"]["5"]["overrides"]["denoise_mod"] = 0.45
        payload["stages"]["10"]["mode"] = "preserve"
        payload["gate_profile"] = GATE_PROFILE_RELAXED

        reset = reset_stage_parameters(payload, (4,))

        self.assertEqual(reset["general"]["output_formats"], ["fit"])
        self.assertEqual(reset["general"]["compute_mode"], "cpu")
        self.assertEqual(reset["general"]["max_retries"], 3)
        self.assertEqual(
            reset["stages"]["1"]["overrides"][
                "stage1_register_fail_ratio_max"
            ],
            0.20,
        )
        self.assertEqual(reset["stages"]["4"]["mode"], "auto")
        self.assertEqual(reset["stages"]["5"]["overrides"]["denoise_mod"], 0.45)
        self.assertEqual(reset["gate_profile"], GATE_PROFILE_RELAXED)
        self.assertEqual(
            reset_stage_parameters(payload)["gate_profile"],
            GATE_PROFILE_DEFAULT,
        )
        reset_all = reset_stage_parameters(payload)
        self.assertTrue(
            all(
                entry == {"mode": "auto", "overrides": {}}
                for entry in reset_all["stages"].values()
            )
        )
        self.assertEqual(reset_all["general"]["max_retries"], 3)

    def test_resume_fingerprints_follow_stage_boundaries(self) -> None:
        base = default_processing_parameters()

        self.assertEqual(
            stage_config_from_processing_settings(base)[2]["boundary_correction"],
            "native_crop_v5",
        )
        self.assertEqual(
            stage_config_from_processing_settings(base)[2]["gate_profile"],
            GATE_PROFILE_DEFAULT,
        )

        def fingerprints(payload):
            return build_resume_fingerprints(
                input_fingerprint="source-fingerprint",
                stage_config=stage_config_from_processing_settings(payload),
            )

        baseline = fingerprints(base)
        stage2 = default_processing_parameters()
        stage2["stages"]["2"]["overrides"]["stage2_edge_black_target"] = 0.08
        stage3 = default_processing_parameters()
        stage3["stages"]["3"]["overrides"][
            "stage3_compound_score_rel_improvement_min"
        ] = 0.20
        stage7 = default_processing_parameters()
        stage7["stages"]["7"]["overrides"][
            "stage7_stretch_chroma_noise_score_max"
        ] = 0.40
        stage8_palette = default_processing_parameters()
        stage8_palette["stages"]["8"]["overrides"][
            "stage8_dualband_palette_selection"
        ] = "OHS"
        relaxed = default_processing_parameters()
        relaxed["gate_profile"] = GATE_PROFILE_RELAXED

        after_stage2 = fingerprints(stage2)
        after_stage3 = fingerprints(stage3)
        after_stage7 = fingerprints(stage7)
        after_stage8_palette = fingerprints(stage8_palette)
        after_relaxed = fingerprints(relaxed)

        self.assertNotEqual(
            baseline["stage2"]["fingerprint"],
            after_stage2["stage2"]["fingerprint"],
        )
        self.assertEqual(
            baseline["stage2"]["fingerprint"],
            after_stage3["stage2"]["fingerprint"],
        )
        self.assertNotEqual(
            baseline["stage5"]["fingerprint"],
            after_stage3["stage5"]["fingerprint"],
        )
        self.assertEqual(baseline, after_stage7)
        self.assertEqual(baseline, after_stage8_palette)
        self.assertNotEqual(
            baseline["stage2"]["fingerprint"],
            after_relaxed["stage2"]["fingerprint"],
        )
        self.assertNotEqual(
            baseline["stage5"]["fingerprint"],
            after_relaxed["stage5"]["fingerprint"],
        )

        for stage, field, value in (
            (3, "stage3_compound_validation_improvement_min", 0.20),
            (4, "stage4_pcc_clip_growth_max", 0.010),
            (5, "stage5_multiscale_detail_retention_min", 0.90),
        ):
            changed = default_processing_parameters()
            changed["stages"][str(stage)]["overrides"][field] = value
            current = fingerprints(changed)
            self.assertEqual(baseline["stage2"], current["stage2"])
            self.assertNotEqual(baseline["stage5"], current["stage5"])

        for stage, field, value in (
            (6, "stage7_residual_star_score_max", 0.60),
            (7, "stage7_stretch_chroma_noise_score_max", 0.40),
            (8, "stage8_texture_artifact_growth_max", 1.50),
            (
                9,
                "stage9_starmask_pre_stretch_compact_enabled",
                True,
            ),
        ):
            changed = default_processing_parameters()
            changed["stages"][str(stage)]["overrides"][field] = value
            self.assertEqual(baseline, fingerprints(changed))

        output_changed = default_processing_parameters(
            general={"output_formats": ["fit"]}
        )
        self.assertEqual(baseline, fingerprints(output_changed))

        delivery_changed = default_processing_parameters(
            general={
                "review_only": True,
                "max_retries": 0,
                "retry_delay": 8.0,
                "review_bundle_enabled": False,
                "managed_output_enabled": False,
            }
        )
        self.assertEqual(baseline, fingerprints(delivery_changed))

        stage10_changed = default_processing_parameters()
        stage10_changed["stages"]["10"]["overrides"][
            "stage10_final_denoise_strength"
        ] = 0.40
        self.assertEqual(baseline, fingerprints(stage10_changed))

        compute_changed = default_processing_parameters(
            general={"compute_mode": "cpu"}
        )
        after_compute = fingerprints(compute_changed)
        self.assertEqual(baseline["stage1"], after_compute["stage1"])
        self.assertEqual(baseline["stage2"], after_compute["stage2"])
        self.assertNotEqual(baseline["stage5"], after_compute["stage5"])

        auto_tune_changed = default_processing_parameters(
            general={"auto_tune_enabled": False}
        )
        after_auto_tune = fingerprints(auto_tune_changed)
        self.assertEqual(baseline["stage1"], after_auto_tune["stage1"])
        self.assertNotEqual(baseline["stage2"], after_auto_tune["stage2"])
        self.assertNotEqual(baseline["stage5"], after_auto_tune["stage5"])

        stage1_changed = default_processing_parameters()
        stage1_changed["stages"]["1"]["overrides"][
            "stage1_register_fail_ratio_max"
        ] = 0.20
        after_stage1 = fingerprints(stage1_changed)
        for boundary in ("stage1", "stage2", "stage5"):
            self.assertNotEqual(baseline[boundary], after_stage1[boundary])

    def test_signed_run_manifest_freezes_processing_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.fit"
            source.write_bytes(b"source")
            source_record = build_source_record(
                source_kind="master_file",
                selected_path=source,
                files=(source,),
            )
            workspace = ensure_task_workspace(
                source_record=source_record,
                selected_path=source,
            )
            payload = default_processing_parameters(
                general={"output_formats": ["fit"], "compute_mode": "cpu"}
            )
            payload["stages"]["7"]["overrides"]["asinh_stretch"] = 3.1
            payload["stages"]["8"]["overrides"][
                "stage8_dualband_palette_selection"
            ] = "OHS"
            payload["stages"]["9"]["overrides"][
                "stage9_highlight_clip_growth_max"
            ] = 0.01
            payload["stages"]["9"]["overrides"][
                "stage9_starmask_pre_stretch_compact_enabled"
            ] = True
            run = begin_task_run(
                workspace=workspace,
                source_record=source_record,
                run_id="processing-parameters-test",
                processing_parameters=payload,
            )
            frozen = run_manifest.load_json(run.manifest_path)

        unsigned = dict(frozen)
        claimed_hash = unsigned.pop("manifest_hash")
        self.assertEqual(
            claimed_hash,
            run_manifest.canonical_payload_hash(unsigned),
        )
        self.assertEqual(
            frozen["processing_parameters"]["stages"]["7"]["overrides"][
                "asinh_stretch"
            ],
            3.1,
        )
        self.assertEqual(
            frozen["processing_parameters"]["general"]["compute_mode"],
            "cpu",
        )
        self.assertEqual(
            frozen["processing_parameters"]["stages"]["8"]["overrides"][
                "stage8_dualband_palette_selection"
            ],
            "OHS",
        )
        self.assertEqual(
            frozen["processing_parameters"]["stages"]["9"]["overrides"][
                "stage9_highlight_clip_growth_max"
            ],
            0.01,
        )
        self.assertTrue(
            frozen["processing_parameters"]["stages"]["9"]["overrides"][
                "stage9_starmask_pre_stretch_compact_enabled"
            ]
        )
        self.assertEqual(
            frozen["processing_gate_profile"]["profile"],
            GATE_PROFILE_DEFAULT,
        )
        self.assertEqual(
            frozen["processing_gate_profile"]["managed_field_count"],
            len(GATE_PROFILE_PARAMETER_SPECS),
        )

    def test_invalid_processing_parameters_fail_before_run_directory_creation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.fit"
            source.write_bytes(b"source")
            source_record = build_source_record(
                source_kind="master_file",
                selected_path=source,
                files=(source,),
            )
            workspace = ensure_task_workspace(
                source_record=source_record,
                selected_path=source,
            )
            payload = default_processing_parameters()
            payload["stages"]["7"]["overrides"]["unknown_parameter"] = 1

            with self.assertRaisesRegex(WorkspaceError, "处理参数无效"):
                begin_task_run(
                    workspace=workspace,
                    source_record=source_record,
                    run_id="invalid-processing-parameters",
                    processing_parameters=payload,
                )

            self.assertFalse(
                (workspace.runs_dir / "invalid-processing-parameters").exists()
            )


if __name__ == "__main__":
    unittest.main()
