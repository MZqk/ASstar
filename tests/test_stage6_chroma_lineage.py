"""Stage 6 subject-chroma lineage contract tests."""

from tests.pipeline_plugin_fallbacks_support import *  # noqa: F401,F403


class Stage6SubjectChromaLineageTests(unittest.TestCase):
    @staticmethod
    def _fixture(*, layout: str = "chw"):
        height = width = 256
        yy, xx = np.mgrid[:height, :width]
        signal = np.exp(
            -(
                ((xx - width * 0.50) / 82.0) ** 2
                + ((yy - height * 0.48) / 74.0) ** 2
            )
        ).astype(np.float32)
        source = np.stack(
            (
                0.018 + 0.19 * signal,
                0.018 + 0.075 * signal,
                0.018 + 0.042 * signal,
            ),
            axis=0,
        ).astype(np.float32)
        records = [
            {
                "id": "S3000001",
                "x": 34.0,
                "y": 42.0,
                "fwhm_px": 3.0,
                "valid_fraction": 1.0,
                "saturated": False,
            }
        ]
        shared = {
            "status": "available",
            "manifest": {
                "schema": pipeline_module.scene_support.SCENE_SUPPORT_SCHEMA,
                "status": "available",
                "components": {
                    "valid_mask": {"status": "available"},
                    "saturation_map": {"status": "available"},
                    "star_catalog": {
                        "status": "available",
                        "valid_count": len(records),
                        "records": records,
                    },
                },
            },
            "valid_mask": np.ones((height, width), dtype=np.uint8),
            "saturation_map": np.zeros((height, width), dtype=np.uint8),
        }
        if layout == "hwc":
            source = np.moveaxis(source, 0, -1)
        return source, shared

    def _assess(self, source, output, shared):
        return stage7_quality_module.assess_stage6_subject_chroma_lineage(
            source,
            output,
            shared,
        )

    def test_chw_float32_identity_preserves_subject_chroma(self):
        source, shared = self._fixture()

        report = self._assess(source, source.copy(), shared)

        self.assertEqual(report["status"], "ok", report)
        self.assertTrue(report["accepted"])
        self.assertAlmostEqual(report["metrics"]["saturation_p50_retention"], 1.0)
        self.assertAlmostEqual(report["metrics"]["opponent_energy_retention"], 1.0)

    def test_hwc_brightness_change_keeps_chroma_direction(self):
        source, shared = self._fixture(layout="hwc")
        output = np.clip(0.82 * source + 0.003, 0.0, 1.0).astype(np.float32)

        report = self._assess(source, output, shared)

        self.assertEqual(report["status"], "ok", report)
        self.assertEqual(report["source_layout"]["channel_layout"], "hwc")
        self.assertGreaterEqual(
            report["metrics"]["opponent_direction_correlation"],
            0.99,
        )

    def test_severe_desaturation_is_a_hard_failure(self):
        source, shared = self._fixture()
        luminance = (
            0.2126 * source[0] + 0.7152 * source[1] + 0.0722 * source[2]
        )
        output = np.repeat(luminance[None, :, :], 3, axis=0).astype(np.float32)

        report = self._assess(source, output, shared)

        self.assertEqual(report["status"], "hard_failed", report)
        self.assertEqual(report["reason_code"], "stage6_subject_chroma_collapse")
        self.assertIn("opponent_energy_retention", report["failed_metrics"])
        self.assertIn("saturation_p50_retention", report["failed_metrics"])

    def test_faint_chroma_on_linear_pedestal_is_not_misclassified_as_colorless(self):
        source, shared = self._fixture()
        signal = np.clip((source[0] - 0.018) / 0.19, 0.0, 1.0)
        source = np.stack(
            (
                0.087 + 0.0040 * signal,
                0.087 + 0.0022 * signal,
                0.087 + 0.0012 * signal,
            ),
            axis=0,
        ).astype(np.float32)
        luminance = (
            0.2126 * source[0] + 0.7152 * source[1] + 0.0722 * source[2]
        )
        output = np.repeat(luminance[None, :, :], 3, axis=0).astype(np.float32)

        report = self._assess(source, output, shared)

        self.assertTrue(report["applicable"], report)
        self.assertEqual(report["status"], "hard_failed", report)
        self.assertEqual(report["reason_code"], "stage6_subject_chroma_collapse")
        self.assertIn("opponent_energy_retention", report["failed_metrics"])

    def test_opponent_direction_reversal_is_rejected(self):
        source, shared = self._fixture()
        output = source[[2, 1, 0], :, :].copy()

        report = self._assess(source, output, shared)

        self.assertEqual(report["status"], "hard_failed", report)
        self.assertIn("opponent_direction_correlation", report["failed_metrics"])

    def test_mono_and_low_chroma_sources_are_not_applicable(self):
        source, shared = self._fixture()
        mono = np.mean(source, axis=0).astype(np.float32)
        mono_report = self._assess(mono, mono.copy(), shared)
        low_chroma = np.repeat(mono[None, :, :], 3, axis=0)
        low_chroma_report = self._assess(low_chroma, low_chroma.copy(), shared)

        self.assertEqual(mono_report["status"], "not_applicable")
        self.assertEqual(low_chroma_report["status"], "not_applicable")
        self.assertFalse(mono_report["hard_failed"])
        self.assertFalse(low_chroma_report["hard_failed"])

    def test_missing_scene_evidence_fails_closed_for_rgb(self):
        source, _shared = self._fixture()

        report = self._assess(source, source.copy(), {})

        self.assertEqual(report["status"], "unverified", report)
        self.assertTrue(report["hard_failed"])
        self.assertEqual(
            report["reason_code"],
            "stage6_subject_chroma_lineage_unverified",
        )

    def test_support_is_candidate_independent(self):
        source, shared = self._fixture()
        identity = self._assess(source, source.copy(), shared)
        desaturated = np.repeat(
            np.mean(source, axis=0, keepdims=True),
            3,
            axis=0,
        ).astype(np.float32)
        collapsed = self._assess(source, desaturated, shared)

        self.assertEqual(identity["coverage"], collapsed["coverage"])
        self.assertEqual(identity["star_exclusion"], collapsed["star_exclusion"])
        self.assertTrue(identity["candidate_independent_support"])
        self.assertTrue(collapsed["candidate_independent_support"])

    def test_failure_codes_are_definite_and_not_retainable(self):
        quality = {
            "status": "poor",
            "subject_chroma_lineage": {
                "hard_failed": True,
                "reason_code": "stage6_subject_chroma_collapse",
            },
            "derived": {},
        }

        processor = pipeline_module.StarunPostProcessor()
        triggers = stage7_quality_module.stage7_repair_triggers(processor, quality)
        codes = stage6_star_separation_module._syqon_quality_failure_codes(triggers)

        self.assertIn("SUBJECT_CHROMA_COLLAPSE", codes)
        self.assertIn(
            "SUBJECT_CHROMA_COLLAPSE",
            stage6_star_separation_module._STAGE6_DEFINITE_QUALITY_REJECTION_CODES,
        )
        self.assertFalse(
            stage6_star_separation_module._stage6_can_retain_hard_failed_pair(
                {
                    "hard_failed": True,
                    "destructive_core_failure": False,
                    "failure_codes": codes,
                },
                pair_valid=True,
            )
        )

    def test_chroma_retry_requires_full_quality_pass(self):
        safe = {
            "status": "ok",
            "subject_chroma_lineage": {
                "accepted": True,
                "hard_failed": False,
            },
        }
        ordinary_failure = copy.deepcopy(safe)
        ordinary_failure["status"] = "poor"
        chroma_failure = copy.deepcopy(safe)
        chroma_failure["subject_chroma_lineage"]["hard_failed"] = True

        self.assertTrue(
            stage6_star_separation_module._stage6_chroma_retry_passed(safe)
        )
        self.assertFalse(
            stage6_star_separation_module._stage6_chroma_retry_passed(
                ordinary_failure
            )
        )
        self.assertFalse(
            stage6_star_separation_module._stage6_chroma_retry_passed(
                chroma_failure
            )
        )

    def test_chroma_retry_plan_is_single_and_fail_closed(self):
        collapsed = {
            "subject_chroma_lineage": {
                "hard_failed": True,
                "reason_code": "stage6_subject_chroma_collapse",
            }
        }
        unverified = {
            "subject_chroma_lineage": {
                "hard_failed": True,
                "reason_code": "stage6_subject_chroma_lineage_unverified",
            }
        }

        ready = stage6_star_separation_module._stage6_chroma_retry_plan(
            collapsed,
            retry_max=3,
            syqon_available=True,
            failure_action="auto_fallback",
        )
        preserve = stage6_star_separation_module._stage6_chroma_retry_plan(
            collapsed,
            retry_max=3,
            syqon_available=True,
            failure_action="preserve_review",
        )
        unavailable = stage6_star_separation_module._stage6_chroma_retry_plan(
            unverified,
            retry_max=3,
            syqon_available=True,
            failure_action="auto_fallback",
        )

        self.assertTrue(ready["should_attempt"])
        self.assertEqual(ready["attempt_limit"], 1)
        self.assertFalse(preserve["should_attempt"])
        self.assertEqual(preserve["status"], "blocked_by_failure_action")
        self.assertFalse(unavailable["should_attempt"])
        self.assertEqual(
            unavailable["status"],
            "direct_reject_unverified_lineage",
        )


if __name__ == "__main__":
    unittest.main()
