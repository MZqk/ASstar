"""Stage 9 PSF, catalog, and authenticated same-source recovery tests."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class PipelinePluginFallbackStage9RecoveryTests(PipelinePluginFallbackTestBase):
    def test_stage9_targeted_psf_recovery_only_targets_small_bright_group(self):
        processor = self._new_processor()
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        quality = self._psf_quality(
            "screen_unscreen_normal_primary",
            {"all": 0.866, "weak": 0.968, "bright": 0.804},
        )

        groups = stage9_module._stage9_psf_recovery_target_groups(
            processor,
            quality,
        )

        self.assertEqual(groups, ("bright",))
        self.assertNotIn("weak", groups)

    def test_stage9_ngc2237_fixed_metrics_use_independent_unscreen_supports_and_stop(self):
        processor = self._new_processor()
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.stage9_targeted_recovery_enabled = True
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")
        shape = (4, 4)
        normal_layer = np.full((3, *shape), 0.20, dtype=np.float32)
        strict_layer = np.full((3, *shape), 0.16, dtype=np.float32)
        prepared_unscreens = []

        def preflight(pipeline, *_args, **_kwargs):
            report = {
                "schema": "starun.stage9-starmask-support-preflight.v2",
                "status": "ready",
                "route": "dual_competition",
                "reason_code": "stage9_support_preflight_boundary_dual",
                "planned_candidates": ["normal", "strict_compact"],
                "skipped_candidates": [],
                "support_masks_equivalent": False,
                "candidates": {
                    "normal": {"status": "ok", "usable": True},
                    "strict_compact": {"status": "ok", "usable": True},
                },
                "_calibrations": {
                    "normal": {"status": "ok", "support_mode": "normal"},
                    "strict_compact": {
                        "status": "ok",
                        "support_mode": "strict_compact",
                    },
                },
            }
            pipeline._stage9_starmask_support_preflight = (
                stage9_module.stage9_quality.public_starmask_support_preflight(
                    report
                )
            )
            return report

        def prepare_support(pipeline, *_args, **kwargs):
            support_mode = (
                "strict_compact" if kwargs.get("strict_support") else "normal"
            )
            layer = strict_layer if support_mode == "strict_compact" else normal_layer
            mask = np.ones(shape, dtype=bool)
            pipeline._stage9_starmask_calibration = {
                "status": "ok",
                "support_mode": support_mode,
            }
            pipeline._stage9_last_star_layer = layer.copy()
            pipeline._stage9_last_star_overlay_mask = mask.copy()
            pipeline._stage9_last_weak_overlay_mask = mask.copy()
            pipeline._stage9_last_bright_overlay_mask = np.zeros(shape, dtype=bool)
            return str(kwargs["output_name"])

        def prepare_unscreen(
            pipeline,
            trusted_starmask_name,
            _messages,
            *,
            output_name,
            support_mode,
        ):
            prepared_unscreens.append(
                (support_mode, trusted_starmask_name, output_name)
            )
            layer = (
                np.full((3, *shape), 0.31, dtype=np.float32)
                if support_mode == "normal"
                else np.full((3, *shape), 0.27, dtype=np.float32)
            )
            mask = np.ones(shape, dtype=bool)
            pipeline._stage9_last_star_layer = layer.copy()
            pipeline._stage9_last_star_overlay_mask = mask.copy()
            pipeline._stage9_last_weak_overlay_mask = mask.copy()
            pipeline._stage9_last_bright_overlay_mask = np.zeros(shape, dtype=bool)
            return {
                "available": True,
                "report": {
                    "schema": "starun.stage9-unscreen-reference.v1",
                    "status": "ready",
                    "available": True,
                    "reason_code": "stage9_unscreen_reference_ready",
                },
                "starmask": output_name,
                "support_mode": support_mode,
                "support_starmask": trusted_starmask_name,
                "trusted_stars": (
                    normal_layer
                    if support_mode == "normal"
                    else strict_layer
                ),
                "unscreen_stars": layer,
                "stars": layer,
                "support_mask": mask,
                "weak_mask": mask,
                "bright_mask": np.zeros(shape, dtype=bool),
            }

        def assess(_source_stem, *, attempt, formula):
            if attempt == "screen_primary":
                ratios = {
                    "all": 0.733799397945404,
                    "weak": 0.7559289336204529,
                    "bright": 0.6859943866729736,
                }
                chroma = 0.004643959673283182
            elif attempt in {
                "screen_compact_primary",
                "screen_unscreen_strict_compact_primary",
            }:
                ratios = {
                    "all": 1.1180340051651,
                    "weak": 1.1435438394546509,
                    "bright": 0.8997353911399841,
                }
                chroma = 0.000007721634124603768
            else:
                ratios = {
                    "all": 1.0377490520477295,
                    "weak": 1.0444660186767578,
                    "bright": 1.0206207036972046,
                }
                chroma = 0.000007721634124603768
            quality = self._psf_quality(attempt, ratios)
            quality["formula"] = formula
            quality["metrics"]["chromatic_star_addition_ratio"] = chroma
            return quality

        processor._stage9_assess_current_remix = assess

        with (
            patch.object(
                stage9_module,
                "_stage9_starmask_support_preflight",
                side_effect=preflight,
            ),
            patch.object(
                stage9_module,
                "_prepare_stage9_starmask_for_pixel_remix",
                side_effect=prepare_support,
            ),
            patch.object(
                stage9_module,
                "_prepare_stage9_unscreen_candidate",
                side_effect=prepare_unscreen,
            ),
            patch.object(
                stage9_module,
                "_stage9_reference_fidelity",
                return_value={"status": "ok", "support_rgb_mae": 0.04},
            ),
        ):
            stage9_star_remixing(processor)

        self.assertEqual(
            prepared_unscreens,
            [
                (
                    "normal",
                    "starmask_stretched",
                    "starmask_unscreen_normal",
                ),
                (
                    "strict_compact",
                    "starmask_stretched_compact_primary",
                    "starmask_unscreen_strict_compact",
                ),
            ],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(len(report["attempts"]), 4)
        self.assertEqual(
            report["selected"]["attempt"],
            "screen_unscreen_normal_primary",
        )
        self.assertEqual(report["selected"]["support_mode"], "normal")
        self.assertEqual(
            report["selected"]["recovery_kind"],
            "unscreen_amplitude_recovery",
        )
        self.assertEqual(
            report["selected"]["support_starmask"],
            "starmask_stretched",
        )
        self.assertFalse(report["delivery_fallback_used"])
        self.assertFalse(report["fallback_used"])
        self.assertAlmostEqual(
            report["selected"]["metrics"]["chromatic_star_addition_ratio"],
            0.000007721634124603768,
        )
        self.assertAlmostEqual(
            report["selected"]["psf_closure"]["groups"]["all"][
                "fwhm_ratio_median"
            ],
            1.0377490520477295,
        )
        self.assertTrue(report["candidate_recovery_used"])
        self.assertFalse(
            any(
                "source_presence" in str(attempt.get("attempt") or "")
                or "selective" in str(attempt.get("attempt") or "")
                for attempt in report["attempts"]
            )
        )

    def test_pipeline_status_stage9_psf_review_is_review_required(self):
        probe = pipeline_module.ProcessorRuntimeMixin()
        probe.results = []
        probe._require_review(9, "stage9_psf_subgroup_evidence_insufficient")

        status = probe._pipeline_result_status()

        self.assertEqual(status, "review_required")

    def test_stage9_accepted_soft_large_psf_uses_contraction_contract(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        quality = self._psf_quality(
            "accepted_soft_large",
            {"all": 1.06, "weak": 1.07, "bright": 1.01},
        )
        quality["accepted"] = True
        quality["status"] = "ok"
        quality["issues"] = []

        self.assertEqual(
            stage9_module._stage9_psf_contraction_target_groups(
                processor,
                quality,
            ),
            ("weak",),
        )
        self.assertTrue(
            stage9_module._stage9_is_psf_large_only_failure(
                processor,
                quality,
            )
        )

    def test_stage9_psf_contraction_rejects_any_non_psf_regression(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        parent = {
            "metrics": {
                "weak_star_recovery_ratio": 0.82,
                "catalog_star_visibility_ratio_weak": 0.90,
                "highlight_clip_growth": 0.001,
                "background_mottling_growth": 1.01,
            }
        }
        candidate = copy.deepcopy(parent)
        candidate["metrics"]["weak_star_recovery_ratio"] = 0.81
        candidate["metrics"]["highlight_clip_growth"] = 0.0011

        report = stage9_module._stage9_psf_contraction_nonregression(
            parent,
            candidate,
        )

        self.assertFalse(report["accepted"])
        self.assertIn(
            "psf_contraction_regressed:weak_star_recovery_ratio",
            report["issues"],
        )
        self.assertIn(
            "psf_contraction_regressed:highlight_clip_growth",
            report["issues"],
        )

    def _authenticated_sep_recovery_fixture(self, processor):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor._stage9_star_reference_catalog = {
            "status": "ok",
            "_component_ids": np.asarray([1, 2]),
            "_peak_y": np.asarray([10.0, 22.0]),
            "_peak_x": np.asarray([12.0, 25.0]),
            "_source_peak_y": np.asarray([10.0, 22.0]),
            "_source_peak_x": np.asarray([12.0, 25.0]),
        }
        original = np.full((3, 32, 32), 0.10, dtype=np.float32)
        before = original.copy()
        original[:, 9:12, 11:14] = 0.65
        persisted = before.copy()
        source_path = processor.process_dir / "stage6_input.fit"
        source_path.write_bytes(b"authenticated-stage6-input")
        source_sha256 = stage9_module.run_manifest.sha256_file(source_path)
        processor._stage6_pair_handoff = {
            "files": {
                "stage6_input": {
                    "path": source_path.name,
                    "sha256": source_sha256,
                    "shape": list(original.shape),
                }
            }
        }
        processor._stage9_matched_domain_context = {
            "available": True,
            "original_display": original,
            "report": {
                "pair_handoff": {
                    "accepted": True,
                    "paths": {"stage6_input": str(source_path)},
                },
                "matched_domain_transfer": {
                    "method": "closed_form_linked_mtf"
                },
            },
        }
        record = {
            "id": "O000001",
            "x": 12.0,
            "y": 10.0,
            "flux": 500.0,
            "peak": 0.65,
            "a": 0.85,
            "b": 0.85,
            "theta": 0.0,
            "npix": 9,
            "flag": 0,
            "fwhm_px": 2.0,
            "axis_ratio": 1.0,
        }
        pixel_sha256 = stage9_module._stage9_pixel_hash(original)
        evidence = {
            "status": "rejected",
            "failed_gates": ["source_recovery_ratio"],
            "match_radius_px": 3.0,
            "sources": {
                "O": {
                    "source_role": "O",
                    "source_name": (
                        "verified_stage6_input_in_stage7_matched_domain"
                    ),
                    "pixel_sha256": pixel_sha256,
                    "source_shape": list(original.shape),
                }
            },
            "catalogs": {
                "O": {
                    "schema": "starun.stage9-sep-catalog.v1",
                    "status": "ok",
                    "source_role": "O",
                    "pixel_sha256": pixel_sha256,
                    "source_shape": list(original.shape),
                    "coordinate_domain": (
                        "siril_pixel_buffer_bottom_up"
                    ),
                    "valid_count": 1,
                    "records_sha256": (
                        stage9_module.stage9_quality._stage9_sep_payload_hash(
                            [record]
                        )
                    ),
                    "records": [record],
                }
            },
            "matches": {"O_C": {"matches": []}},
        }
        binding = stage9_module._stage9_bind_sep_o_source_evidence(
            processor,
            original,
            evidence,
        )
        self.assertTrue(binding["accepted"], binding)
        return original, before, persisted, evidence

    def test_stage9_sep_recovery_uses_only_authenticated_o_coordinates(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        original, before, persisted, evidence = (
            self._authenticated_sep_recovery_fixture(processor)
        )

        candidate, report = stage9_module._stage9_build_same_source_sep_recovery(
            processor,
            original=original,
            before=before,
            persisted=persisted,
            sep_evidence=evidence,
        )

        self.assertIsNotNone(candidate, report)
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["selected_o_ids"], ["O000001"])
        self.assertEqual(report["new_coordinate_count"], 0)
        self.assertEqual(report["selected_sources"][0]["o_x"], 12.0)
        self.assertEqual(report["selected_sources"][0]["o_y"], 10.0)
        self.assertEqual(report["outside_support_max_abs_change"], 0.0)
        self.assertGreater(float(candidate[:, 10, 12].max()), 0.10)
        np.testing.assert_array_equal(candidate[:, 2, 2], persisted[:, 2, 2])

    def test_stage9_sep_recovery_rejects_missing_or_mismatched_o_digest(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        for mutation in ("missing", "mismatch"):
            with self.subTest(mutation=mutation):
                processor = self._new_processor()
                original, before, persisted, evidence = (
                    self._authenticated_sep_recovery_fixture(processor)
                )
                if mutation == "missing":
                    evidence["catalogs"]["O"].pop("records_sha256")
                else:
                    evidence["catalogs"]["O"]["records_sha256"] = "f" * 64

                candidate, report = (
                    stage9_module._stage9_build_same_source_sep_recovery(
                        processor,
                        original=original,
                        before=before,
                        persisted=persisted,
                        sep_evidence=evidence,
                    )
                )

                self.assertIsNone(candidate)
                self.assertFalse(report["accepted"])
                self.assertEqual(
                    report["reason"],
                    (
                        "sep_o_catalog_records_digest_missing"
                        if mutation == "missing"
                        else "sep_o_catalog_records_digest_mismatch"
                    ),
                )

    def test_stage9_sep_recovery_rejects_source_and_coordinate_tampering(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        cases = (
            ("pixel", "sep_o_source_pixel_sha256_mismatch"),
            ("artifact", "sep_o_source_artifact_mismatch"),
            ("coordinate", "sep_evidence_report_digest_mismatch"),
        )
        for mutation, reason in cases:
            with self.subTest(mutation=mutation):
                processor = self._new_processor()
                original, before, persisted, evidence = (
                    self._authenticated_sep_recovery_fixture(processor)
                )
                if mutation == "pixel":
                    evidence["catalogs"]["O"]["pixel_sha256"] = "a" * 64
                elif mutation == "artifact":
                    evidence["sources"]["O"]["source_artifact"][
                        "sha256"
                    ] = "b" * 64
                else:
                    record = evidence["catalogs"]["O"]["records"][0]
                    record["x"] = 12.25
                    evidence["catalogs"]["O"]["records_sha256"] = (
                        stage9_module.stage9_quality._stage9_sep_payload_hash(
                            [record]
                        )
                    )

                candidate, report = (
                    stage9_module._stage9_build_same_source_sep_recovery(
                        processor,
                        original=original,
                        before=before,
                        persisted=persisted,
                        sep_evidence=evidence,
                    )
                )

                self.assertIsNone(candidate)
                self.assertFalse(report["accepted"])
                self.assertEqual(report["reason"], reason)

    def test_stage9_sep_recovery_refuses_nonunique_gate_failure(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor = self._new_processor()
        image = np.full((3, 16, 16), 0.10, dtype=np.float32)
        candidate, report = stage9_module._stage9_build_same_source_sep_recovery(
            processor,
            original=image,
            before=image,
            persisted=image,
            sep_evidence={
                "failed_gates": [
                    "source_recovery_ratio",
                    "unmatched_ratio",
                ]
            },
        )

        self.assertIsNone(candidate)
        self.assertFalse(report["accepted"])
        self.assertEqual(
            report["reason"],
            "sep_recovery_requires_unique_source_recovery_failure",
        )

    def test_stage9_remix_base_identity_rejects_between_candidate_mutation(self):
        processor = self._new_processor()
        base_path = processor.process_dir / "stage8_enhanced.fit"
        base_path.write_bytes(b"immutable-stage8-base")

        locked = processor._stage9_verify_remix_base_identity("stage8_enhanced")
        self.assertEqual(locked["status"], "verified")
        base_path.write_bytes(b"mutated-stage8-base")
        rejected = processor._stage9_verify_remix_base_identity("stage8_enhanced")

        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("changed between candidates", rejected["reason"])

    def test_stage9_hard_halo_risk_still_bypasses_after_accepted_stretch(self):
        processor = self._new_processor()
        processor.cfg.stage7_residual_star_score_max = 0.28
        processor.cfg.stage7_halo_residue_score_max = 0.35
        processor.cfg.stage7_starless_noise_gain_max = 2.2
        processor.cfg.stage7_starless_dynamic_range_min_ratio = 0.55
        processor.cfg.stage7_starless_peak_signal_min = 0.006
        processor.stretched_name = "stage7_stretched"
        processor._stage7_stretch_accepted = True
        processor._stage7_stretch_output = "stage7_stretched"
        processor._stage7_selected_quality = {
            "status": "poor",
            "issues": [
                "starless_dynamic_range_collapse 0.117<0.550, "
                "peak=0.00551<0.00600"
            ],
            "derived": {
                "residual_star_score": 0.0,
                "halo_residue_score": 0.71,
                "starless_noise_gain": 0.78,
                "starless_dynamic_range_ratio": 0.117,
                "starless_peak_signal": 0.00551,
            },
        }
        (processor.process_dir / "stage7_stretched.fit").write_bytes(b"accepted")
        processor._stage9_bad_starless_reason = types.MethodType(
            pipeline_module.StarunPostProcessor._stage9_bad_starless_reason,
            processor,
        )
        processor._stage7_effective_halo_threshold = lambda: (
            processor.cfg.stage7_halo_residue_score_max
        )

        reason = processor._stage9_bad_starless_reason()

        self.assertIn("stage7_halo_residue_score", reason)
        self.assertNotIn("stage7_starless_dynamic_range", reason)

    def test_stage9_uses_descending_fallback_intensity_ladder(self):
        processor = self._new_processor()
        processor.cfg.workflow_plugin_probe_enabled = False
        processor.cfg.star_intensity = 1.05
        processor.cfg.stage9_fallback_intensity_cap = 1.0
        processor.starmask_file = processor.process_dir / "starmask.fit"
        processor.starmask_file.write_bytes(b"mock")

        def assess(_source_stem, *, attempt, formula):
            accepted = attempt == "screen_fallback_040"
            return {
                "attempt": attempt,
                "formula": formula,
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
                "issues": [] if accepted else ["changed_pixel_ratio"],
                "metrics": {},
            }

        processor._stage9_assess_current_remix = assess

        stage9_star_remixing(processor)

        self.assertEqual(
            [call[2] for call in processor.previous_stage_remix_calls],
            [1.05, 0.75, 0.55, 0.40],
        )
        report = processor.stage_json_reports["stage9_remix_quality.json"]
        self.assertEqual(report["selected"]["attempt"], "screen_fallback_040")

    def test_stage9_retry_limit_and_floor_bound_the_intensity_ladder(self):
        processor = self._new_processor()
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        processor.cfg.stage9_fallback_intensity_cap = 0.95
        processor.cfg.stage9_fallback_retry_max = 3
        processor.cfg.stage9_fallback_intensity_floor = 0.55

        candidates = stage9_module._stage9_remix_intensity_candidates(
            processor,
            primary_intensity=1.05,
            remix_scale=1.0,
        )

        self.assertEqual(
            candidates,
            [("primary", 1.05), ("fallback_075", 0.75), ("fallback_055", 0.55)],
        )
        processor.cfg.stage9_fallback_retry_max = 1
        self.assertEqual(
            stage9_module._stage9_remix_intensity_candidates(
                processor,
                primary_intensity=1.05,
                remix_scale=1.0,
            ),
            [("primary", 1.05), ("fallback_075", 0.75)],
        )

    def test_stage9_handoff_v3_rejects_legal_route_swaps_by_canonical_digest(self):
        stage9_module = sys.modules["stages.stage9_star_remixing"]
        scenarios = (
            ("safe_passthrough_color_only", "structure_enhanced"),
            ("structure_enhanced", "safe_passthrough_color_only"),
        )
        for original_route, swapped_route in scenarios:
            with self.subTest(
                original_route=original_route,
                swapped_route=swapped_route,
            ):
                processor = self._new_processor()
                handoff = processor._make_stage8_handoff(
                    "stage8_enhanced",
                    processing_route=original_route,
                )
                if original_route == "safe_passthrough_color_only":
                    artifact = dict(handoff["source_artifact"])
                    preflight = {
                        "accepted": True,
                        "source_mode": "structure_rollback",
                        "checks": {
                            name: {"accepted": True}
                            for name in (
                                "exact_structure_rollback",
                                "stage7_presentation_reference",
                                "spatial_background",
                                "subject_boundary_seam",
                                "star_halo",
                                "clipping",
                            )
                        },
                    }
                    final_validation = {
                        "accepted": True,
                        "checks": {
                            "color": {"accepted": True},
                            "background_seam_clip_presentation": {
                                "status": "ok"
                            },
                            "spatial_background": {"accepted": True},
                            "star_halo": {"accepted": True},
                            "artifact": {"accepted": True},
                        },
                    }
                    handoff.update(
                        safe_passthrough_color_only={
                            "preflight": preflight,
                            "final_validation": final_validation,
                            "color_gate_verified": True,
                        },
                        color_gate={
                            "used_for_gate": True,
                            "status": "reported",
                            "guard_lineage_verified": True,
                            "final_pixel_identity": {
                                "sha256": artifact["sha256"],
                                "pixel_sha256": artifact["pixel_sha256"],
                            },
                        },
                        star_halo_guard={"status": "ok", "verified": True},
                    )
                    handoff = seal_stage8_handoff(handoff)
                original = stage9_module._stage9_verify_stage8_handoff(
                    processor,
                    "stage8_enhanced",
                    handoff,
                )
                self.assertTrue(original["verified"], original)
                handoff["processing_route"] = swapped_route

                report = stage9_module._stage9_verify_stage8_handoff(
                    processor,
                    "stage8_enhanced",
                    handoff,
                )

                self.assertFalse(report["verified"], report)
                self.assertTrue(report["review_only"], report)
                self.assertIn(
                    "stage8_handoff_canonical_digest_mismatch",
                    report["issues"],
                )
                self.assertIn(
                    "stage8_handoff_route_evidence_summary_mismatch",
                    report["issues"],
                )
