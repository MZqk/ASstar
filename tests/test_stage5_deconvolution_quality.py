from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy.exceptions" not in sys.modules:
    package = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")

    class SirilError(Exception):
        pass

    class CommandError(SirilError):
        pass

    exceptions.SirilError = SirilError
    exceptions.CommandError = CommandError
    package.exceptions = exceptions
    sys.modules["sirilpy"] = package
    sys.modules["sirilpy.exceptions"] = exceptions

from sirilpy.exceptions import SirilError  # noqa: E402
import stage5_deconvolution_quality as quality  # noqa: E402
from stages import stage5_linear_denoise  # noqa: E402


def _star(
    x: float,
    y: float,
    *,
    fwhm_x: float = 2.0,
    fwhm_y: float = 2.2,
    saturated: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        xpos=x,
        ypos=y,
        fwhmx=fwhm_x,
        fwhmy=fwhm_y,
        A=0.5,
        B=0.01,
        rmse=0.001,
        sat=1.0,
        has_saturated=saturated,
    )


def _catalog_and_gaussian_image() -> tuple[list[dict], np.ndarray]:
    coordinates = (
        (30, 30),
        (70, 30),
        (110, 30),
        (30, 80),
        (70, 80),
        (110, 80),
    )
    report, catalog = quality.build_psf_quality_report(
        [_star(x, y) for x, y in coordinates],
        (130, 145),
    )
    assert report["decision"] == "proceed"
    yy, xx = np.mgrid[:130, :145]
    image = np.full((3, 130, 145), 0.01, dtype=np.float32)
    for star in catalog:
        sigma = float(star["fwhm_geometry"]) / 2.355
        gaussian = 0.30 * np.exp(
            -(
                (xx - float(star["x"])) ** 2
                + (yy - float(star["y"])) ** 2
            )
            / (2.0 * sigma**2)
        )
        image += gaussian[None, :, :].astype(np.float32)
    return catalog, image


class Stage5PSFQualityTests(unittest.TestCase):
    def test_only_structural_catalog_failures_skip_rl(self) -> None:
        cases = {
            "empty_star_catalog": [],
            "all_stars_saturated": [
                _star(30, 30, saturated=True),
                _star(60, 60, saturated=True),
            ],
            "no_finite_positive_fwhm_coordinates": [
                _star(30, 30, fwhm_x=0.0),
                _star(60, 60, fwhm_y=float("nan")),
            ],
        }
        for reason, stars in cases.items():
            with self.subTest(reason=reason):
                report, _catalog = quality.build_psf_quality_report(
                    stars,
                    (100, 100),
                )
                self.assertTrue(report["hard_skip_rl"])
                self.assertEqual(report["decision"], "skip_rl")
                self.assertEqual(report["reason_code"], reason)

        shadow_only, _catalog = quality.build_psf_quality_report(
            [_star(25, 25), _star(75, 75)],
            (100, 100),
        )
        self.assertEqual(shadow_only["decision"], "proceed")
        self.assertTrue(shadow_only["shadow_thresholds"]["would_warn"])

    def test_catalog_records_geometry_isolation_overlap_and_exclusions(self) -> None:
        target = np.zeros((120, 120), dtype=bool)
        target[48:63, 48:63] = True
        report, catalog = quality.build_psf_quality_report(
            [
                _star(5, 5),
                _star(50, 50),
                _star(53, 50),
                _star(95, 95),
            ],
            (120, 120),
            target_structure_mask=target,
        )

        self.assertEqual(report["status"], "available")
        self.assertIn("edge_distance_lt_5_fwhm", catalog[0]["exclusion_reasons"])
        self.assertIn(
            "target_structure_overlap_gt_0_25",
            catalog[1]["exclusion_reasons"],
        )
        self.assertIn(
            "nearest_neighbor_lt_6_fwhm",
            catalog[1]["exclusion_reasons"],
        )
        self.assertIn("median", report["summary"]["fwhm_geometry"])
        self.assertIn("mad", report["summary"]["ellipticity"])

    def test_local_guard_accepts_unchanged_or_lightly_sharpened_gaussian_stars(self) -> None:
        catalog, baseline = _catalog_and_gaussian_image()

        unchanged = quality.assess_local_star_guard(
            baseline,
            baseline.copy(),
            catalog,
            method="graxpert_object",
        )
        lightly_sharpened = baseline.copy()
        for star in catalog:
            y = int(round(float(star["y"])))
            x = int(round(float(star["x"])))
            lightly_sharpened[:, y, x] *= 1.05
        light = quality.assess_local_star_guard(
            baseline,
            lightly_sharpened,
            catalog,
            method="siril_rl",
        )

        self.assertEqual(unchanged["status"], "available")
        self.assertFalse(unchanged["would_rollback"])
        self.assertEqual(light["status"], "available")
        self.assertFalse(light["would_rollback"], light)

    def test_local_guard_detects_bright_ring_dark_ring_and_core_anomaly(self) -> None:
        catalog, baseline = _catalog_and_gaussian_image()
        yy, xx = np.mgrid[: baseline.shape[1], : baseline.shape[2]]

        def modified(kind: str) -> np.ndarray:
            candidate = baseline.copy()
            for star in catalog:
                radius = np.sqrt(
                    (xx - float(star["x"])) ** 2
                    + (yy - float(star["y"])) ** 2
                )
                fwhm = float(star["fwhm_geometry"])
                ring = (radius >= 1.4 * fwhm) & (radius <= 2.1 * fwhm)
                core = radius <= 0.75 * fwhm
                if kind == "bright_ring":
                    candidate[:, ring] += 0.08
                elif kind == "dark_ring":
                    candidate[:, ring] -= 0.08
                else:
                    candidate[:, core] = 0.01 + 3.0 * (
                        candidate[:, core] - 0.01
                    )
            return candidate

        expected = {
            "bright_ring": "positive_ring_residual_p95",
            "dark_ring": "negative_ring_residual_p95",
            "core": "core_peak_ratio_p95",
        }
        for kind, reason in expected.items():
            with self.subTest(kind=kind):
                report = quality.assess_local_star_guard(
                    baseline,
                    modified(kind),
                    catalog,
                    method="siril_rl",
                )
                self.assertTrue(report["would_rollback"], report)
                self.assertIn(reason, report["would_rollback_reasons"])
                self.assertTrue(report["enforced"])
                self.assertTrue(report["participates_in_acceptance"])
                self.assertFalse(report["accepted"])
                self.assertTrue(report["rollback_required"])

    def test_local_guard_is_unavailable_when_catalog_evidence_is_insufficient(self) -> None:
        catalog, baseline = _catalog_and_gaussian_image()
        for star in catalog[2:]:
            star["eligible_for_local_guard"] = False

        report = quality.assess_local_star_guard(
            baseline,
            baseline,
            catalog,
            method="graxpert_object",
        )

        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["eligible_star_count"], 2)
        self.assertTrue(report["would_rollback"])
        self.assertFalse(report["accepted"])
        self.assertTrue(report["rollback_required"])

    def test_local_guard_excludes_low_signal_stars_without_ratio_explosion(self) -> None:
        catalog, baseline = _catalog_and_gaussian_image()
        star = catalog[0]
        yy, xx = np.mgrid[: baseline.shape[1], : baseline.shape[2]]
        radius = np.sqrt(
            (xx - float(star["x"])) ** 2
            + (yy - float(star["y"])) ** 2
        )
        low_signal = baseline.copy()
        low_signal[:, radius <= 5.0 * float(star["fwhm_geometry"])] = 0.01

        report = quality.assess_local_star_guard(
            low_signal,
            low_signal.copy(),
            catalog,
            method="graxpert_object",
        )

        self.assertEqual(report["status"], "available", report)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["excluded_low_signal_star_count"], 1)
        self.assertEqual(report["evaluated_star_count"], 5)
        excluded = report["excluded_low_signal_stars"]
        self.assertEqual(len(excluded), 1)
        self.assertNotIn(
            "core_peak_ratio",
            excluded[0]["signals"]["rec709"],
        )

    def test_all_near_zero_star_samples_fail_closed_without_large_ratios(self) -> None:
        catalog, baseline = _catalog_and_gaussian_image()
        near_zero = np.full_like(baseline, 1e-9)

        report = quality.assess_local_star_guard(
            near_zero,
            near_zero.copy(),
            catalog,
            method="siril_rl",
        )

        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["excluded_low_signal_star_count"], 6)
        self.assertTrue(report["rollback_required"])
        self.assertNotIn("aggregates", report)


class _RLFake:
    def __init__(self, stars: object = None, *, expose_api: bool = True) -> None:
        self.cfg = SimpleNamespace(
            stage5_deconvolution_enabled=True,
            stage5_rl_maxstars=200,
            stage5_rl_psf_kernel_size=33,
            stage5_rl_iters=8,
            stage5_rl_alpha=3000.0,
            stage5_rl_gdstep=0.0005,
            stage5_rl_stop=0.001,
        )
        self.events: list[tuple] = []
        self.log = SimpleNamespace(
            info=lambda *_args: None,
            warn=lambda *_args: None,
        )
        if expose_api:
            if isinstance(stars, BaseException):
                def get_image_stars():
                    raise stars
            else:
                def get_image_stars():
                    return stars
            self.siril = SimpleNamespace(get_image_stars=get_image_stars)
        else:
            self.siril = SimpleNamespace()
        self._stage5_input_linear_pixels = np.zeros((3, 100, 100), dtype=np.float32)
        self._stage5_target_structure_mask = None

    def cmd_with_check(self, *args):
        self.events.append(("cmd", *args))

    def _write_stage_json(self, name, report):
        self.events.append(("json", name, report.get("decision")))

    def _save_stage_output(self, stem):
        self.events.append(("save", stem))
        return True

    @staticmethod
    def _short_text(value, _limit):
        return str(value)


class Stage5RLOrderingTests(unittest.TestCase):
    def test_structural_failure_writes_report_and_restores_before_skipping(self) -> None:
        for stars in (
            [],
            [_star(30, 30, saturated=True)],
            [_star(30, 30, fwhm_x=0.0)],
        ):
            with self.subTest(stars=stars):
                pipeline = _RLFake(stars)
                applied = stage5_linear_denoise._run_stage5_rl_deconvolution(
                    pipeline,
                    [],
                )
                commands = [event[1:] for event in pipeline.events if event[0] == "cmd"]
                self.assertFalse(applied)
                self.assertNotIn(("makepsf",), [command[:1] for command in commands])
                self.assertNotIn(("rl",), [command[:1] for command in commands])
                self.assertEqual(commands[-1], ("load", "stage5_input_linear"))
                self.assertTrue(pipeline._stage5_psf_quality_report["hard_skip_rl"])

    def test_api_unavailable_or_exception_proceeds_unverified(self) -> None:
        for pipeline in (
            _RLFake(expose_api=False),
            _RLFake(SirilError("api failed")),
        ):
            with self.subTest(expose_api=hasattr(pipeline.siril, "get_image_stars")):
                applied = stage5_linear_denoise._run_stage5_rl_deconvolution(
                    pipeline,
                    [],
                )
                commands = [event[1] for event in pipeline.events if event[0] == "cmd"]
                self.assertTrue(applied)
                self.assertIn("makepsf", commands)
                self.assertIn("rl", commands)
                self.assertEqual(
                    pipeline._stage5_psf_quality_report["decision"],
                    "proceed_unverified",
                )

    def test_valid_report_is_written_before_makepsf_and_rl_without_reload(self) -> None:
        pipeline = _RLFake(
            [_star(30, 30), _star(70, 30), _star(30, 70), _star(70, 70)]
        )

        applied = stage5_linear_denoise._run_stage5_rl_deconvolution(
            pipeline,
            [],
        )

        self.assertTrue(applied)
        findstar_index = next(
            index
            for index, event in enumerate(pipeline.events)
            if event[:2] == ("cmd", "findstar")
        )
        report_index = next(
            index
            for index, event in enumerate(pipeline.events)
            if event[:2] == ("json", "stage5_psf_quality.json")
        )
        makepsf_index = next(
            index
            for index, event in enumerate(pipeline.events)
            if event[:2] == ("cmd", "makepsf")
        )
        rl_index = next(
            index
            for index, event in enumerate(pipeline.events)
            if event[:2] == ("cmd", "rl")
        )
        self.assertLess(findstar_index, report_index)
        self.assertLess(report_index, makepsf_index)
        self.assertLess(makepsf_index, rl_index)
        self.assertFalse(
            any(
                event[:2] == ("cmd", "load")
                for event in pipeline.events[findstar_index + 1 : makepsf_index]
            )
        )
        findstar = pipeline.events[findstar_index]
        self.assertIn("-out=stage5_psf_stars.csv", findstar)


if __name__ == "__main__":
    unittest.main()
