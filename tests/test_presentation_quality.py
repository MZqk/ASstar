from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


from models import PipelineConfig  # noqa: E402
from presentation_quality import (  # noqa: E402
    build_presentation_quality_report,
    verify_stage7_presentation_reference,
)
from stage7_stretch_metrics import (  # noqa: E402
    canonical_json_sha256,
    stage7_pixel_sha256,
)


class PresentationQualityTests(unittest.TestCase):
    @staticmethod
    def _reference(size: int = 64) -> np.ndarray:
        yy, xx = np.indices((size, size), dtype=np.float32)
        checker = ((xx.astype(np.int32) + yy.astype(np.int32)) % 2) * 2.0 - 1.0
        radial = np.exp(
            -((xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2)
            / (2.0 * (size / 7.0) ** 2)
        )
        base = 0.08 + 0.16 * radial + 0.006 * checker * radial
        return np.stack(
            (
                base + 0.050 * radial,
                base + 0.020 * radial,
                base + 0.035 * radial,
            ),
            axis=0,
        ).astype(np.float32)

    @staticmethod
    def _masks(size: int = 64) -> dict[str, np.ndarray]:
        yy, xx = np.indices((size, size))
        subject = (
            (xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2
            <= (size / 4.0) ** 2
        )
        return {
            "subject_mask": subject.astype(np.float32),
            "background_mask": (~subject).astype(np.float32),
            "star_mask": np.zeros((size, size), dtype=np.float32),
            "star_halo_guard_mask": np.zeros(
                (size, size),
                dtype=np.float32,
            ),
        }

    @staticmethod
    def _science(*, accepted: bool = True) -> dict:
        return {
            "schema": "starun.final-quality.v4",
            "status": "ok" if accepted else "needs_conservative_rerun",
            "final_quality": "ok" if accepted else "poor",
            "needs_conservative_rerun": not accepted,
            "issues": [] if accepted else ["background_noise"],
            "spatial_background_gradient": {
                "status": "ok" if accepted else "rejected",
                "accepted": accepted,
            },
        }

    @staticmethod
    def _psf(ratio: float = 1.0) -> dict:
        return {
            "psf_closure": {
                "groups": {
                    name: {
                        "status": "ok",
                        "fwhm_ratio_median": ratio,
                    }
                    for name in ("all", "weak", "bright")
                }
            }
        }

    def _report(
        self,
        candidate: np.ndarray,
        *,
        science: bool = True,
        psf: float = 1.0,
        masks: dict[str, np.ndarray] | None = None,
    ) -> dict:
        reference = self._reference()
        return build_presentation_quality_report(
            reference,
            candidate,
            masks if masks is not None else self._masks(),
            PipelineConfig(),
            target_type="emission_nebula_widefield",
            stage9_quality=self._psf(psf),
            stars_required=True,
            stars_not_required_verified=False,
            scientific_report=self._science(accepted=science),
        )

    def test_identical_internal_reference_passes_all_presentation_gates(self):
        report = self._report(self._reference())

        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["accepted"])
        self.assertFalse(report["external_reference_used"])
        self.assertEqual(
            report["reference"],
            "stage7_presentation_reference",
        )

    def test_color_collapse_fails_closed(self):
        reference = self._reference()
        gray = np.repeat(
            np.mean(reference, axis=0, keepdims=True),
            3,
            axis=0,
        )
        report = self._report(gray)

        self.assertFalse(report["accepted"])
        self.assertEqual(report["gates"]["color"]["status"], "rejected")
        self.assertIn("color", report["issues"])

    def test_task_config_cannot_relax_low_chroma_not_applicable_evidence(self):
        reference = self._reference()
        gray = np.mean(reference, axis=0)
        low_color = np.stack(
            (gray + 0.005, gray, gray),
            axis=0,
        ).astype(np.float32)
        cfg = PipelineConfig()
        cfg.stage7_low_chroma_source_saturation_max = 0.10
        cfg.stage7_low_chroma_source_opponent_rms_max = 0.01

        report = build_presentation_quality_report(
            low_color,
            low_color,
            self._masks(),
            cfg,
            target_type="emission_nebula_widefield",
            stage9_quality=self._psf(),
            stars_required=True,
            stars_not_required_verified=False,
            scientific_report=self._science(),
        )

        color = report["gates"]["color"]
        self.assertEqual(
            color["limits"]["low_chroma_source_saturation_max"],
            0.02,
        )
        self.assertEqual(
            color["limits"]["low_chroma_source_opponent_rms_max"],
            0.0001,
        )
        self.assertTrue(color["applicable"], color)
        self.assertNotEqual(
            color.get("reason"),
            "stage7_reference_chroma_below_measurement_floor",
        )

    def test_science_only_pass_is_not_formal_presentation_acceptance(self):
        report = self._report(self._reference(), psf=1.08)

        self.assertTrue(report["gates"]["scientific"]["accepted"])
        self.assertTrue(
            report["gates"]["stars"]["hard_science_gate_passed"]
        )
        self.assertFalse(
            report["gates"]["stars"]["soft_presentation_target_passed"]
        )
        self.assertFalse(report["accepted"])

    def test_required_psf_group_missing_fails_closed(self):
        reference = self._reference()
        incomplete_psf = self._psf()
        incomplete_psf["psf_closure"]["groups"].pop("weak")

        report = build_presentation_quality_report(
            reference,
            reference,
            self._masks(),
            PipelineConfig(),
            target_type="emission_nebula_widefield",
            stage9_quality=incomplete_psf,
            stars_required=True,
            stars_not_required_verified=False,
            scientific_report=self._science(),
        )

        stars = report["gates"]["stars"]
        self.assertFalse(stars["accepted"])
        self.assertFalse(stars["groups_complete"])
        self.assertEqual(stars["missing_groups"], ["weak"])
        self.assertEqual(stars["issues"], ["stage9_psf_groups_incomplete"])
        self.assertFalse(report["accepted"])

    def test_presentation_only_pass_cannot_override_science_failure(self):
        report = self._report(self._reference(), science=False)

        self.assertFalse(report["gates"]["scientific"]["accepted"])
        self.assertFalse(report["accepted"])

    def test_verified_star_preserve_route_makes_psf_not_applicable(self):
        reference = self._reference()
        report = build_presentation_quality_report(
            reference,
            reference,
            self._masks(),
            PipelineConfig(),
            target_type="open_cluster",
            profile_name="star_colour_preserve",
            stage9_quality=None,
            stars_required=False,
            stars_not_required_verified=True,
            scientific_report=self._science(),
        )

        self.assertTrue(report["gates"]["stars"]["accepted"])
        self.assertEqual(
            report["gates"]["stars"]["status"],
            "not_applicable",
        )
        self.assertTrue(report["accepted"])

    def test_task_configuration_cannot_weaken_locked_presentation_gates(self):
        cfg = PipelineConfig()
        cfg.stage10_presentation_color_p50_retention_min = 0.01
        cfg.stage10_presentation_visibility_retention_min = 0.01
        cfg.stage10_presentation_microdetail_growth_max = 3.0
        cfg.stage10_presentation_generic_brightness_retention_min = 0.01
        cfg.stage9_psf_recovery_target_min = 0.50
        cfg.stage9_psf_recovery_target_max = 1.50
        reference = self._reference()

        report = build_presentation_quality_report(
            reference,
            reference,
            self._masks(),
            cfg,
            target_type="emission_nebula_widefield",
            stage9_quality=self._psf(),
            stars_required=True,
            stars_not_required_verified=False,
            scientific_report=self._science(),
        )

        self.assertEqual(
            report["gates"]["color"]["limits"][
                "saturation_p50_retention_min"
            ],
            0.35,
        )
        self.assertEqual(report["gates"]["visibility"]["limit"], 0.60)
        self.assertEqual(
            report["gates"]["microdetail"]["growth"]["limit"],
            1.60,
        )
        self.assertEqual(
            report["gates"]["brightness"]["floors"][
                "subject_p50_retention"
            ],
            0.60,
        )
        self.assertEqual(report["gates"]["stars"]["limits"]["soft_min"], 0.97)
        self.assertEqual(report["gates"]["stars"]["limits"]["soft_max"], 1.05)

    def test_star_remix_does_not_fabricate_microdetail_growth(self):
        reference = self._reference()
        candidate = np.array(reference, copy=True)
        masks = self._masks()
        yy, xx = np.indices(reference.shape[-2:])
        subject = masks["subject_mask"] > 0.25
        star_support = subject & (xx < reference.shape[-1] // 2 - 2)
        star_texture = (
            ((xx.astype(np.int32) + yy.astype(np.int32)) % 2) * 2.0 - 1.0
        ).astype(np.float32)
        candidate[:, star_support] = np.clip(
            candidate[:, star_support] + 0.25 * star_texture[star_support],
            0.0,
            1.0,
        )
        masks["star_mask"][star_support] = 1.0
        masks["star_halo_guard_mask"][
            subject & (xx < reference.shape[-1] // 2 + 2)
        ] = 1.0

        report = self._report(candidate, masks=masks)

        detail = report["gates"]["microdetail"]
        self.assertTrue(detail["accepted"], detail)
        self.assertLessEqual(detail["metrics"]["retention"], 1.60)
        self.assertGreater(
            report["metrics"][
                "overall_rendition_microdetail_retention_diagnostic"
            ],
            1.60,
        )
        self.assertTrue(
            detail["support"]["shared_between_reference_and_candidate"]
        )

    def test_nonstellar_over_sharpening_still_fails_microdetail_gate(self):
        reference = self._reference()
        masks = self._masks()
        yy, xx = np.indices(reference.shape[-2:])
        checker = (
            ((xx.astype(np.int32) + yy.astype(np.int32)) % 2) * 2.0 - 1.0
        ).astype(np.float32)
        subject = masks["subject_mask"] > 0.25
        candidate = np.array(reference, copy=True)
        candidate[:, subject] = np.clip(
            candidate[:, subject] + 0.08 * checker[subject],
            0.0,
            1.0,
        )

        report = self._report(candidate, masks=masks)

        detail = report["gates"]["microdetail"]
        self.assertEqual(detail["status"], "rejected")
        self.assertFalse(detail["growth"]["accepted"])
        self.assertGreater(detail["metrics"]["retention"], 1.60)
        self.assertIn("microdetail", report["issues"])

    def test_microdetail_support_ignores_background_overlap_and_is_stable(self):
        reference = self._reference()
        masks_a = self._masks()
        masks_b = self._masks()
        subject = masks_a["subject_mask"] > 0.25
        masks_a["background_mask"][subject] = 0.0
        masks_b["background_mask"][subject] = 1.0

        report_a = self._report(reference, masks=masks_a)
        report_b = self._report(reference, masks=masks_b)
        support_a = report_a["gates"]["microdetail"]["support"]
        support_b = report_b["gates"]["microdetail"]["support"]

        self.assertEqual(support_a["count"], support_b["count"])
        self.assertEqual(support_a["sha256"], support_b["sha256"])
        self.assertEqual(
            support_a["mask_provenance"]["digest_sha256"],
            support_b["mask_provenance"]["digest_sha256"],
        )
        self.assertNotIn(
            "background_mask",
            support_a["mask_provenance"]["consumed_masks"],
        )
        self.assertTrue(support_a["shared_between_reference_and_candidate"])

    def test_microdetail_evidence_missing_or_shape_tampered_fails_closed(self):
        reference = self._reference()
        missing = self._masks()
        missing.pop("star_mask")
        missing_halo = self._masks()
        missing_halo.pop("star_halo_guard_mask")
        malformed = self._masks()
        malformed["star_halo_guard_mask"] = np.zeros(
            (reference.shape[-2] - 1, reference.shape[-1]),
            dtype=np.float32,
        )

        for masks, reason in (
            (missing, "star_mask evidence is unavailable"),
            (missing_halo, "star_halo_guard_mask evidence is unavailable"),
            (malformed, "star_halo_guard_mask shape mismatch"),
        ):
            with self.subTest(reason=reason):
                report = self._report(reference, masks=masks)
                detail = report["gates"]["microdetail"]
                self.assertEqual(detail["status"], "unavailable")
                self.assertFalse(detail["accepted"])
                self.assertIn(reason, detail["reason"])
                self.assertIn("microdetail", report["issues"])

    def test_frozen_reference_report_tamper_fails_closed(self):
        pixels = self._reference()
        pixel_sha = stage7_pixel_sha256(pixels)
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "stage7_presentation_reference.fit"
            artifact_path.write_bytes(b"reference-container")
            artifact = {
                "stem": "stage7_presentation_reference",
                "file": artifact_path.name,
                "container_sha256": hashlib.sha256(
                    b"reference-container"
                ).hexdigest(),
                "pixel_sha256": pixel_sha,
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
            }
            selected = {
                "stem": "stage7_cand_b",
                "pixel_sha256": pixel_sha,
            }
            formal = {
                "stem": "stage7_stretched",
                "pixel_sha256": pixel_sha,
            }
            report = {
                "schema": "starun.stage7-presentation-reference.v1",
                "status": "ready",
                "accepted": True,
                "linear_source": {"stem": "stage6_starless"},
                "selected_candidate": selected,
                "matched_domain": {"schema": "v4", "status": "active"},
                "source_artifact": formal,
                "artifact": artifact,
            }
            binding = {
                "linear_source": report["linear_source"],
                "selected_candidate": selected,
                "matched_domain": report["matched_domain"],
                "formal_source_artifact": formal,
            }
            report["source_binding_sha256"] = canonical_json_sha256(binding)
            report["report_sha256"] = canonical_json_sha256(report)

            verified = verify_stage7_presentation_reference(
                report,
                pixels,
                artifact_path,
            )
            self.assertTrue(verified["accepted"])

            report["matched_domain"]["status"] = "tampered"
            with self.assertRaisesRegex(ValueError, "report digest mismatch"):
                verify_stage7_presentation_reference(
                    report,
                    pixels,
                    artifact_path,
                )


if __name__ == "__main__":
    unittest.main()
