#!/usr/bin/env python3
"""Independent SEP catalog and persisted O/B/C cross-match tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from models import PipelineConfig  # noqa: E402
import stage9_quality  # noqa: E402


_SEP_DTYPE = np.dtype(
    [
        ("x", "f8"),
        ("y", "f8"),
        ("flux", "f8"),
        ("peak", "f8"),
        ("a", "f8"),
        ("b", "f8"),
        ("theta", "f8"),
        ("npix", "i4"),
        ("flag", "i4"),
    ]
)


class _FakeBackground:
    globalback = 0.01
    globalrms = 0.02

    def __init__(self, image, *, bw, bh, fw, fh):
        self.image = np.asarray(image)
        self.parameters = (bw, bh, fw, fh)

    def back(self):
        return np.zeros_like(self.image)


class _FakeSep:
    __version__ = "1.4.1-test"
    OBJ_MERGED = 1
    OBJ_TRUNC = 2
    OBJ_DOVERFLOW = 4
    OBJ_SINGU = 8
    Background = _FakeBackground

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=_SEP_DTYPE)
        self.extract_kwargs = None

    def extract(self, image, threshold, **kwargs):
        self.extract_kwargs = {
            "shape": tuple(np.asarray(image).shape),
            "threshold": threshold,
            **kwargs,
        }
        return self.rows.copy()


def _record(role: str, index: int, x: float, y: float, flux: float) -> dict:
    return {
        "id": f"{role}{index:06d}",
        "x": float(x),
        "y": float(y),
        "flux": float(flux),
        "peak": float(min(1.0, flux / 1000.0)),
        "a": 1.7,
        "b": 1.7,
        "theta": 0.0,
        "npix": 12,
        "flag": 0,
        "fwhm_px": 4.003,
        "axis_ratio": 1.0,
    }


def _catalog(role: str, coordinates: list[tuple[float, float]], fluxes=None) -> dict:
    values = fluxes or [100.0 + index for index in range(len(coordinates))]
    records = [
        _record(role, index + 1, x, y, values[index])
        for index, (x, y) in enumerate(coordinates)
    ]
    return {
        "schema": "starun.stage9-sep-catalog.v1",
        "status": "ok",
        "source_role": role,
        "valid_count": len(records),
        "records_sha256": stage9_quality._stage9_sep_payload_hash(records),
        "records": records,
    }


class Stage9SepCrossmatchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PipelineConfig()
        self.scale = {"status": "ready", "fwhm_median_px": 4.0}

    def test_independent_catalog_uses_fixed_sep_contract_and_stable_records(self):
        rows = [
            (12, 8, 90, 0.7, 1.70, 1.65, 0.1, 14, 0),
            (4, 5, 80, 0.6, 1.70, 1.70, 0.0, 12, _FakeSep.OBJ_MERGED),
            (6, 6, 70, 0.5, 1.70, 1.70, 0.0, 12, _FakeSep.OBJ_TRUNC),
            (7, 7, 60, 0.4, 2.00, 0.70, 0.0, 12, 0),
            (9, 9, 50, 0.3, 4.50, 4.50, 0.0, 30, 0),
        ]
        fake_sep = _FakeSep(rows)
        image = np.zeros((3, 24, 32), dtype=np.float32)
        catalog = stage9_quality.build_independent_sep_catalog(
            image,
            self.cfg,
            role="O",
            pixel_sha256="a" * 64,
            spatial_scale=self.scale,
            sep_module=fake_sep,
        )

        self.assertEqual(catalog["status"], "ok")
        self.assertEqual(catalog["valid_count"], 2)
        self.assertEqual([row["id"] for row in catalog["records"]], ["O000001", "O000002"])
        self.assertEqual([row["y"] for row in catalog["records"]], [5.0, 8.0])
        self.assertFalse(catalog["starmask_prefiltered"])
        self.assertEqual(fake_sep.extract_kwargs["threshold"], 5.0)
        self.assertEqual(fake_sep.extract_kwargs["minarea"], 3)
        self.assertEqual(fake_sep.extract_kwargs["deblend_nthresh"], 32)
        self.assertTrue(fake_sep.extract_kwargs["clean"])
        self.assertEqual(catalog["rejection_counts"]["rejected_flag"], 1)
        self.assertEqual(catalog["rejection_counts"]["axis_ratio"], 1)
        self.assertEqual(catalog["rejection_counts"]["fwhm_scale"], 1)
        self.assertEqual(
            catalog["records_sha256"],
            stage9_quality._stage9_sep_payload_hash(catalog["records"]),
        )

    def test_missing_sep_is_unavailable_not_exception(self):
        catalog = stage9_quality.build_independent_sep_catalog(
            np.zeros((20, 20), dtype=np.float32),
            self.cfg,
            role="C",
            pixel_sha256="b" * 64,
            spatial_scale=self.scale,
            sep_module=None,
        )
        self.assertEqual(catalog["status"], "unavailable")
        self.assertIn("SEP runtime dependency", catalog["reason"])

    def test_nonfinite_pixels_and_shape_mismatch_are_unavailable(self):
        image = np.zeros((20, 20), dtype=np.float32)
        image[3, 4] = np.nan
        catalog = stage9_quality.build_independent_sep_catalog(
            image,
            self.cfg,
            role="O",
            pixel_sha256="a" * 64,
            spatial_scale=self.scale,
            sep_module=_FakeSep([]),
        )
        report = stage9_quality.assess_independent_sep_crossmatch(
            np.zeros((3, 20, 20), dtype=np.float32),
            np.zeros((3, 20, 20), dtype=np.float32),
            np.zeros((3, 19, 20), dtype=np.float32),
            self.cfg,
            original_pixel_sha256="a" * 64,
            before_pixel_sha256="b" * 64,
            after_pixel_sha256="c" * 64,
            spatial_scale=self.scale,
            sep_module=_FakeSep([]),
        )

        self.assertEqual(catalog["status"], "unavailable")
        self.assertIn("non-finite", catalog["reason"])
        self.assertEqual(report["status"], "unavailable")
        self.assertIn("shapes", report["reason"])

    def test_match_is_deterministic_and_one_to_one(self):
        source = _catalog("C", [(0.0, 0.0), (0.2, 0.0), (5.0, 5.0)])
        target = _catalog("O", [(0.1, 0.0), (0.3, 0.0), (5.1, 5.0)])
        first = stage9_quality._stage9_sep_match_catalogs(
            source, target, radius_px=1.0
        )
        second = stage9_quality._stage9_sep_match_catalogs(
            source, target, radius_px=1.0
        )

        self.assertEqual(first, second)
        self.assertEqual(first["match_count"], 3)
        self.assertEqual(len({row["source_id"] for row in first["matches"]}), 3)
        self.assertEqual(len({row["target_id"] for row in first["matches"]}), 3)
        self.assertEqual(
            first["matches_sha256"],
            stage9_quality._stage9_sep_payload_hash(first["matches"]),
        )

    def _assess_with_catalogs(self, catalogs):
        def fake_catalog(_image, _cfg, *, role, **_kwargs):
            return catalogs[role]

        with patch.object(
            stage9_quality,
            "build_independent_sep_catalog",
            side_effect=fake_catalog,
        ):
            return stage9_quality.assess_independent_sep_crossmatch(
                np.zeros((3, 64, 64), dtype=np.float32),
                np.zeros((3, 64, 64), dtype=np.float32),
                np.zeros((3, 64, 64), dtype=np.float32),
                self.cfg,
                original_pixel_sha256="a" * 64,
                before_pixel_sha256="b" * 64,
                after_pixel_sha256="c" * 64,
                spatial_scale=self.scale,
                sep_module=SimpleNamespace(),
            )

    def test_formal_bright_c_subset_accepts_same_source_catalog(self):
        coordinates = [
            (float(x), float(y))
            for y in range(8, 64, 8)
            for x in range(8, 64, 8)
        ]
        catalogs = {
            role: _catalog(role, coordinates)
            for role in ("O", "B", "C")
        }
        report = self._assess_with_catalogs(catalogs)

        self.assertEqual(report["status"], "ok", report)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["formal_set"]["selected_count"], 16)
        self.assertEqual(report["formal_set"]["crossmatch"]["source_match_ratio"], 1.0)
        self.assertEqual(report["match_radius_px"], 3.0)

    def test_new_bright_c_sources_reject_formal_gate(self):
        coordinates = [
            (float(x), float(y))
            for y in range(8, 64, 8)
            for x in range(8, 64, 8)
        ]
        c_coordinates = list(coordinates)
        c_coordinates[:5] = [(200.0 + index * 10.0, 200.0) for index in range(5)]
        bright_fluxes = [10000.0 - index for index in range(5)] + [100.0 + index for index in range(len(coordinates) - 5)]
        catalogs = {
            "O": _catalog("O", coordinates),
            "B": _catalog("B", coordinates),
            "C": _catalog("C", c_coordinates, bright_fluxes),
        }
        report = self._assess_with_catalogs(catalogs)

        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["accepted"])
        self.assertIn("source_match_ratio", report["failed_gates"])
        self.assertIn("unmatched_ratio", report["failed_gates"])

    def test_candidate_only_provenance_cannot_hide_missing_source_stars(self):
        original_coordinates = [
            (float((index % 20) * 10), float((index // 20) * 10))
            for index in range(160)
        ]
        candidate_coordinates = original_coordinates[:32]
        catalogs = {
            "O": _catalog("O", original_coordinates),
            "B": _catalog("B", candidate_coordinates),
            "C": _catalog("C", candidate_coordinates),
        }

        report = self._assess_with_catalogs(catalogs)

        self.assertEqual(report["status"], "rejected")
        self.assertEqual(
            report["formal_set"]["crossmatch"]["source_match_ratio"],
            1.0,
        )
        self.assertAlmostEqual(
            report["formal_set"]["source_recovery"]["source_match_ratio"],
            0.20,
        )
        self.assertIn("source_recovery_ratio", report["failed_gates"])

    def test_config_cannot_relax_fixed_sep_catalog_or_crossmatch_gates(self):
        self.cfg.stage9_sep_axis_ratio_min = 0.10
        self.cfg.stage9_sep_fwhm_ratio_min = 0.10
        self.cfg.stage9_sep_fwhm_ratio_max = 9.00
        self.cfg.stage9_sep_match_radius_min_px = 16.0
        self.cfg.stage9_sep_match_radius_max_px = 32.0
        self.cfg.stage9_sep_match_radius_fwhm = 4.0
        self.cfg.stage9_sep_high_confidence_fraction = 0.0
        self.cfg.stage9_sep_source_match_ratio_min = 0.0
        self.cfg.stage9_sep_unmatched_ratio_max = 1.0
        self.cfg.stage9_sep_source_recovery_ratio_min = 0.0
        self.cfg.stage9_sep_separation_p50_max_px = 32.0
        self.cfg.stage9_sep_separation_p95_max_px = 32.0

        resolved = stage9_quality._stage9_sep_config(self.cfg)
        self.assertEqual(resolved["axis_ratio_min"], 0.50)
        self.assertEqual(resolved["fwhm_ratio_min"], 0.50)
        self.assertEqual(resolved["fwhm_ratio_max"], 2.20)

        original_coordinates = [
            (float((index % 20) * 10), float((index // 20) * 10))
            for index in range(160)
        ]
        candidate_coordinates = original_coordinates[:32]
        report = self._assess_with_catalogs(
            {
                "O": _catalog("O", original_coordinates),
                "B": _catalog("B", candidate_coordinates),
                "C": _catalog("C", candidate_coordinates),
            }
        )

        self.assertEqual(report["status"], "rejected", report)
        self.assertEqual(report["match_radius_px"], 3.0)
        self.assertEqual(report["formal_set"]["selected_count"], 16)
        self.assertEqual(report["failed_gates"], ["source_recovery_ratio"])
        gates = report["gates"]
        self.assertEqual(gates["source_match_ratio"]["minimum"], 0.75)
        self.assertEqual(gates["unmatched_ratio"]["maximum"], 0.25)
        self.assertEqual(gates["source_recovery_ratio"]["minimum"], 0.30)
        self.assertEqual(gates["distance_p50_px"]["maximum"], 0.75)
        self.assertEqual(gates["distance_p95_px"]["maximum"], 1.50)

    def test_independent_source_presence_restores_only_positive_o_minus_b_pixels(self):
        original = np.full((3, 32, 32), 0.10, dtype=np.float32)
        starless = original.copy()
        original[:, 14:18, 14:18] = 0.40
        stabilized = np.zeros_like(original)
        support = np.zeros((32, 32), dtype=bool)
        source_catalog = _catalog("O", [(15.5, 15.5)], [500.0])

        with patch.object(
            stage9_quality,
            "build_independent_sep_catalog",
            return_value=source_catalog,
        ):
            candidate, recovered_support, report = (
                stage9_quality.build_independent_source_presence_candidate(
                    original,
                    starless,
                    stabilized,
                    support,
                    self.cfg,
                    spatial_scale=self.scale,
                )
            )

        self.assertEqual(report["status"], "ready", report)
        self.assertTrue(report["changed"])
        self.assertGreater(report["added_support_pixel_count"], 0)
        self.assertTrue(recovered_support[15, 15])
        self.assertGreater(float(candidate[:, 15, 15].max()), 0.0)
        self.assertEqual(float(candidate[:, 2, 2].max()), 0.0)

    def test_formal_ratio_and_distance_boundaries_are_inclusive(self):
        coordinates = [(float(index * 10), 20.0) for index in range(40)]
        c_coordinates = list(coordinates)
        c_coordinates[6:12] = [
            (coordinates[index][0] + 1.5, coordinates[index][1])
            for index in range(6, 12)
        ]
        c_coordinates[12:16] = [
            (500.0 + index * 10.0, 500.0)
            for index in range(4)
        ]
        fluxes = [10000.0 - index for index in range(16)] + [
            100.0 + index for index in range(24)
        ]
        catalogs = {
            "O": _catalog("O", coordinates),
            "B": _catalog("B", coordinates),
            "C": _catalog("C", c_coordinates, fluxes),
        }
        report = self._assess_with_catalogs(catalogs)

        formal = report["formal_set"]
        self.assertTrue(report["accepted"], report)
        self.assertEqual(formal["crossmatch"]["source_match_ratio"], 0.75)
        self.assertEqual(formal["unmatched_ratio"], 0.25)
        self.assertEqual(formal["crossmatch"]["distance_p50_px"], 0.75)
        self.assertEqual(formal["crossmatch"]["distance_p95_px"], 1.5)

    def test_insufficient_o_sources_is_unavailable(self):
        coordinates = [(float(index), 10.0) for index in range(40)]
        catalogs = {
            "O": _catalog("O", coordinates[:31]),
            "B": _catalog("B", coordinates),
            "C": _catalog("C", coordinates),
        }
        report = self._assess_with_catalogs(catalogs)
        self.assertEqual(report["status"], "unavailable")
        self.assertFalse(report["accepted"])

    def test_v10_reader_requires_sep_artifact_and_v9_requires_review(self):
        v10 = stage9_quality.interpret_stage9_remix_quality_report(
            {
                "schema": "starun.stage9-remix-quality.v10",
                "formal_accepted": True,
                "persisted_output_validation": {
                    "accepted": True,
                    "sep_crossmatch_accepted": True,
                },
                "sep_crossmatch": {
                    "schema": "starun.stage9-sep-crossmatch.v1",
                    "accepted": True,
                    "artifact_sha256": "d" * 64,
                },
            }
        )
        v9 = stage9_quality.interpret_stage9_remix_quality_report(
            {
                "schema": "starun.stage9-remix-quality.v9",
                "formal_accepted": True,
                "persisted_output_validation": {"accepted": True},
            }
        )
        self.assertTrue(v10["formal_accepted"])
        self.assertFalse(v9["formal_accepted"])
        self.assertEqual(v9["reason_code"], "stage9_v9_sep_crossmatch_unavailable")


if __name__ == "__main__":
    unittest.main()
