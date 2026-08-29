"""Focused Stage 3 to Stage 7/10 spatial-background lineage tests."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import spatial_background_lineage as lineage  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class SpatialBackgroundLineageTests(unittest.TestCase):
    @staticmethod
    def _image(*, rg_slope: float = 0.0, luma_slope: float = 0.0):
        height, width = 80, 96
        yy, xx = np.mgrid[:height, :width]
        xn = xx / (width - 1) - 0.5
        yn = yy / (height - 1) - 0.5
        luma = 0.20 + luma_slope * xn
        rg = rg_slope * xn - 0.4 * rg_slope * yn
        green = luma - 0.2126 * rg
        return np.stack((green + rg, green, green)).astype(np.float32)

    @staticmethod
    def _points(width: int, height: int):
        return [
            (
                (cell_x + 0.5) / 4.0 * (width - 1),
                (cell_y + 0.5) / 4.0 * (height - 1),
            )
            for cell_y in range(4)
            for cell_x in range(4)
        ]

    def _write_lineage(
        self,
        root: Path,
        reference: np.ndarray,
        *,
        display_reference: np.ndarray | None = None,
    ):
        height, width = reference.shape[1:]
        support = np.ones((height, width), dtype=np.uint8)
        points = self._points(width, height)
        support_path = root / "stage3_spatial_background_support.fit"
        input_path = root / "stage3_bg_input.fit"
        output_path = root / "stage3_bgremoved.fit"
        fits.PrimaryHDU(support).writeto(support_path)
        fits.PrimaryHDU(reference).writeto(input_path)
        fits.PrimaryHDU(reference).writeto(output_path)
        metrics = lineage.measure_spatial_background_planes(
            reference,
            support.astype(bool),
            points,
            patch_radius=2,
        )
        reference_plane = {
            name: {
                "coefficients": list(component.get("coefficients") or []),
                "slope_span": component.get("slope_span"),
                "slope_significance_sigma": component.get(
                    "slope_significance_sigma"
                ),
            }
            for name, component in metrics.items()
        }
        payload = lineage.seal_lineage({
            "schema": lineage.LINEAGE_SCHEMA,
            "status": "accepted",
            "accepted": True,
            "review_required": False,
            "run_id": "spatial-lineage-fixture",
            "processing_route": "verified_noop",
            "image_shape": [height, width],
            "channel_layout": "rgb_chw",
            "support_artifact": support_path.name,
            "support_kind": "candidate_independent_full_sky_mask",
            "support_pixel_count": int(np.count_nonzero(support)),
            "support_coverage": 1.0,
            "sample_patch_support_pixel_count": int(
                np.count_nonzero(
                    lineage.build_sample_patch_support(
                        support.shape,
                        points,
                        support.astype(bool),
                        patch_radius=2,
                    )[0]
                )
            ),
            "sample_patch_min_support_pixel_count": 25,
            "support_sha256": _sha256(support_path),
            "stage3_input_sha256": _sha256(input_path),
            "stage3_input_pixel_sha256": lineage._array_sha256(reference),
            "stage3_output_sha256": _sha256(output_path),
            "stage3_output_pixel_sha256": lineage._array_sha256(reference),
            "fit_points": [list(point) for point in points[:12]],
            "validation_points": [list(point) for point in points[12:]],
            "patch_radius": 2,
            "reference_metrics": metrics,
            "reference_plane": {
                "coordinate_system": "normalized_image_xy",
                "components": reference_plane,
                "sha256": lineage._json_sha256(reference_plane),
            },
            "projection_schema": None,
            "projection_reason_code": None,
            "selected_components": [],
            "unresolved_components": [],
        })
        (root / "stage3_spatial_background_lineage.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        expected = reference if display_reference is None else display_reference
        candidate_path = root / "stage7_candidate.fit"
        fits.PrimaryHDU(expected).writeto(candidate_path)
        spatial = lineage.assess_stage7_spatial_chroma(
            root,
            expected,
            expected,
            transform_identity={
                "status": "ok",
                "method": "synthetic_authenticated_tone",
                "digest": "synthetic",
            },
        )
        stage7_reference = lineage.build_stage7_display_reference(
            root,
            {
                "name": "synthetic",
                "file": candidate_path.name,
                "spatial_chroma_quality": spatial,
            },
            {
                "status": "active",
                "schema": "starun.stage7-matched-domain-transfer.v3",
                "method": "synthetic_authenticated_tone",
                "chain_contract": {"sha256": "synthetic"},
            },
        )
        (root / lineage.STAGE7_REFERENCE_NAME).write_text(
            json.dumps(stage7_reference),
            encoding="utf-8",
        )

    def test_stage7_accepts_theoretical_match_and_rejects_extra_chroma(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self._image()
            self._write_lineage(root, reference)
            expected = self._image(rg_slope=0.002)

            accepted = lineage.assess_stage7_spatial_chroma(
                root,
                expected,
                expected,
            )
            rejected = lineage.assess_stage7_spatial_chroma(
                root,
                self._image(rg_slope=0.008),
                expected,
            )

            self.assertTrue(accepted["accepted"], accepted)
            self.assertFalse(rejected["accepted"], rejected)
            self.assertTrue(
                any("R-G" in issue for issue in rejected["issues"]),
                rejected,
            )

    def test_stage7_star_preserve_reference_does_not_require_replay_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self._image()
            self._write_lineage(root, reference)
            candidate_path = root / "stage7_star_preserve.fit"
            fits.PrimaryHDU(reference).writeto(candidate_path)
            spatial = lineage.assess_stage7_spatial_chroma(
                root,
                reference,
                reference,
                transform_identity={
                    "status": "ok",
                    "method": "iterative_masked_mtf",
                },
            )
            selected = {
                "name": "cand_a",
                "file": candidate_path.name,
                "spatial_chroma_quality": spatial,
            }
            unavailable_transfer = {
                "status": "unavailable",
                "method": "iterative_masked_mtf",
            }

            required = lineage.build_stage7_display_reference(
                root,
                selected,
                unavailable_transfer,
            )
            star_preserve = lineage.build_stage7_display_reference(
                root,
                selected,
                unavailable_transfer,
                stars_required=False,
            )

            self.assertFalse(required["accepted"], required)
            self.assertTrue(star_preserve["accepted"], star_preserve)
            self.assertEqual(
                star_preserve["matched_domain_transfer"]["status"],
                "not_required",
            )
            self.assertEqual(
                star_preserve["matched_domain_transfer"]["reason_code"],
                "stars_not_required",
            )

    def test_final_gate_rejects_significant_gradient_and_sha_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self._image()
            self._write_lineage(root, reference)

            accepted = lineage.assess_final_spatial_background(root, reference)
            rejected = lineage.assess_final_spatial_background(
                root,
                self._image(rg_slope=0.006),
            )
            self.assertTrue(accepted["accepted"], accepted)
            self.assertFalse(rejected["accepted"], rejected)

            support = root / "stage3_spatial_background_support.fit"
            support.write_bytes(support.read_bytes() + b"tamper")
            tampered = lineage.assess_final_spatial_background(root, reference)
            self.assertFalse(tampered["accepted"])
            self.assertIn("SHA mismatch", " ".join(tampered["lineage"]["issues"]))

    def test_formal_lineage_rejects_missing_or_tampered_contract_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self._image()
            self._write_lineage(root, reference)
            report_path = root / "stage3_spatial_background_lineage.json"
            original = json.loads(report_path.read_text(encoding="utf-8"))

            for field in (
                "run_id",
                "processing_route",
                "review_required",
                "fit_points",
                "validation_points",
                "support_kind",
                "support_coverage",
                "sample_patch_support_pixel_count",
                "sample_patch_min_support_pixel_count",
                "stage3_input_pixel_sha256",
                "stage3_output_sha256",
                "stage3_output_pixel_sha256",
                "reference_plane",
                "chain_digest",
            ):
                with self.subTest(missing=field):
                    mutated = dict(original)
                    mutated.pop(field)
                    if field != "chain_digest":
                        mutated = lineage.seal_lineage(mutated)
                    report_path.write_text(json.dumps(mutated), encoding="utf-8")
                    loaded = lineage.load_lineage(root)
                    self.assertFalse(loaded["accepted"], loaded)

            tampered_cases = []
            bad_route = dict(original)
            bad_route["processing_route"] = "review_only"
            tampered_cases.append(lineage.seal_lineage(bad_route))
            review = dict(original)
            review["review_required"] = True
            tampered_cases.append(lineage.seal_lineage(review))
            output_sha = dict(original)
            output_sha["stage3_output_sha256"] = "0" * 64
            tampered_cases.append(lineage.seal_lineage(output_sha))
            reference_plane = json.loads(json.dumps(original))
            reference_plane["reference_plane"]["components"]["luma"][
                "slope_span"
            ] = 0.25
            tampered_cases.append(lineage.seal_lineage(reference_plane))
            chain_only = json.loads(json.dumps(original))
            chain_only["run_id"] = "tampered-without-resealing"
            tampered_cases.append(chain_only)

            for index, mutated in enumerate(tampered_cases):
                with self.subTest(tampered=index):
                    report_path.write_text(json.dumps(mutated), encoding="utf-8")
                    loaded = lineage.load_lineage(root)
                    self.assertFalse(loaded["accepted"], loaded)

    def test_legacy_lineage_is_readable_but_never_formal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self._image()
            self._write_lineage(root, reference)
            report_path = root / "stage3_spatial_background_lineage.json"
            legacy = json.loads(report_path.read_text(encoding="utf-8"))
            legacy["schema"] = "starun.stage3-spatial-background-lineage.v1"
            legacy.pop("chain_digest", None)
            report_path.write_text(json.dumps(legacy), encoding="utf-8")

            loaded = lineage.load_lineage(root)

            self.assertEqual(loaded["status"], "legacy_nonformal")
            self.assertFalse(loaded["accepted"])
            self.assertFalse(loaded["formal_eligible"])
            self.assertEqual(loaded["run_id"], legacy["run_id"])

    def test_final_gate_compares_against_the_display_domain_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = np.random.default_rng(42)
            linear = self._image(rg_slope=0.00001, luma_slope=0.00001)
            displayed = self._image(rg_slope=0.00012, luma_slope=0.00012)
            displayed += generator.normal(
                0.0,
                0.004,
                size=displayed.shape,
            ).astype(np.float32)
            displayed = np.clip(displayed, 0.0, 1.0)
            self._write_lineage(
                root,
                linear,
                display_reference=displayed,
            )

            accepted = lineage.assess_final_spatial_background(root, displayed)
            rejected = lineage.assess_final_spatial_background(
                root,
                self._image(rg_slope=0.010, luma_slope=0.010),
            )

            self.assertTrue(accepted["accepted"], accepted)
            self.assertFalse(rejected["accepted"], rejected)
            self.assertLessEqual(
                accepted["components"]["R-G"]["slope_growth"],
                1.25,
            )

    def test_final_gate_accepts_unchanged_significant_display_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            linear = self._image(rg_slope=0.00001, luma_slope=0.00001)
            displayed = self._image(rg_slope=0.006, luma_slope=0.006)
            self._write_lineage(
                root,
                linear,
                display_reference=displayed,
            )

            accepted = lineage.assess_final_spatial_background(root, displayed)

            self.assertTrue(accepted["accepted"], accepted)
            self.assertTrue(accepted["components"]["R-G"]["significant"])
            self.assertTrue(
                accepted["components"]["R-G"]["reference_significant"]
            )
            self.assertFalse(
                accepted["components"]["R-G"]["newly_significant"]
            )

    def test_final_gate_rejects_tampered_stage7_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self._image()
            self._write_lineage(root, reference)
            candidate = root / "stage7_candidate.fit"
            with fits.open(candidate, mode="update", memmap=False) as hdul:
                hdul[0].data[0, 0, 0] += np.float32(0.01)
                hdul.flush()

            report = lineage.assess_final_spatial_background(root, reference)

            self.assertFalse(report["accepted"])
            self.assertIn(
                "pixel SHA mismatch",
                " ".join(report["stage7_display_reference"]["issues"]),
            )


if __name__ == "__main__":
    unittest.main()
