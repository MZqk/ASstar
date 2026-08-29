"""Stage 7 common-root colour, noise, replay, and reference contracts."""
from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


if "sirilpy" not in sys.modules:
    fake_sirilpy = types.ModuleType("sirilpy")
    fake_exceptions = types.ModuleType("sirilpy.exceptions")
    fake_enums = types.ModuleType("sirilpy.enums")

    class _SirilError(Exception):
        pass

    fake_sirilpy.SirilInterface = object
    fake_exceptions.SirilError = _SirilError
    fake_exceptions.CommandError = _SirilError
    fake_exceptions.DataError = _SirilError
    fake_exceptions.SirilConnectionError = _SirilError
    fake_enums.CommandStatus = type(
        "CommandStatus",
        (),
        {"CMD_GENERIC_ERROR": 1, "CMD_THREAD_RUNNING": 2},
    )
    sys.modules["sirilpy"] = fake_sirilpy
    sys.modules["sirilpy.exceptions"] = fake_exceptions
    sys.modules["sirilpy.enums"] = fake_enums

import stage7_stretch_metrics as metrics  # noqa: E402
from stage6_services import (  # noqa: E402
    STAGE7_CANDIDATE_RANKING_POLICY,
    Stage6ServiceMixin,
    _stage7_matched_domain_transfer_contract,
)
from stages import stage7_stretching  # noqa: E402
from stages import stage9_star_remixing  # noqa: E402


def _rendition(
    subject_saturation: float,
    non_background_saturation: float,
    *,
    opponent_rms: float = 0.05,
    mask_source: str = "stage6_frozen_roi",
) -> dict:
    return {
        "schema": metrics.RENDITION_METRICS_SCHEMA,
        "status": "available",
        "mask_source": mask_source,
        "metrics": {
            "saturation_median": subject_saturation,
            "saturation_p95": subject_saturation * 1.5,
            "opponent_rms": opponent_rms,
            "non_background_saturation_median": non_background_saturation,
            "non_background_saturation_p95": non_background_saturation * 1.5,
        },
    }


class Stage7CommonRootContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimpleNamespace()

    def _chroma_gate(
        self,
        ratio: float,
        *,
        non_background_ratio: float | None = None,
        profile: str = "generic_balanced",
        source_saturation: float = 0.20,
        source_opponent_rms: float = 0.05,
        semantics: str = "broadband_rgb_osc",
        cfg: object | None = None,
    ) -> dict:
        source_non_background = 0.18
        return metrics.assess_subject_chroma_retention(
            _rendition(
                source_saturation * ratio,
                source_non_background
                * (
                    ratio
                    if non_background_ratio is None
                    else non_background_ratio
                ),
                opponent_rms=max(source_opponent_rms * ratio, 0.0),
            ),
            _rendition(
                source_saturation,
                source_non_background,
                opponent_rms=source_opponent_rms,
            ),
            cfg or self.cfg,
            profile_name=profile,
            channel_semantics=semantics,
        )

    def test_five_target_vectors_select_the_expected_candidate(self) -> None:
        vectors = {
            "M31": {
                "expected": "cand_quantile",
                "candidates": {
                    "cand_b": (0.999, 0.484, False),
                    "cand_display90": (0.084, 0.020, True),
                    "cand_quantile": (0.82, 0.70, True),
                },
            },
            "NGC6888": {
                "expected": "cand_b",
                "candidates": {
                    "cand_b": (0.790, 0.784, True),
                    "cand_display90": (0.534, 0.530, True),
                },
            },
            "NGC7000": {
                "expected": "cand_b",
                "candidates": {
                    "cand_b": (0.919, 0.677, True),
                    "cand_display90": (0.077, 0.029, True),
                },
            },
            "NGC6910": {
                "expected": "cand_a",
                "profile": "star_colour_preserve",
                "candidates": {
                    "cand_a": (0.278, None, True),
                    "cand_b": (0.417, None, False),
                },
            },
            "M8": {
                "expected": "cand_b",
                "candidates": {
                    "cand_b": (0.894, 0.675, True),
                    "cand_display70": (0.386, 0.250, True),
                },
            },
        }
        for target, vector in vectors.items():
            with self.subTest(target=target):
                attempts = []
                for index, (name, values) in enumerate(
                    vector["candidates"].items()
                ):
                    ratio, non_background_ratio, brightness_ok = values
                    gate = self._chroma_gate(
                        ratio,
                        non_background_ratio=non_background_ratio,
                        profile=vector.get("profile", "generic_balanced"),
                    )
                    allowed = bool(gate["accepted"] and brightness_ok)
                    attempts.append(
                        {
                            "name": name,
                            "status": "ok",
                            "stem": f"stage7_{name}",
                            "allowed_as_final": allowed,
                            "technical_safe": True,
                            "target_local_quality": {},
                            "subject_brightness_selection": {
                                "formal_floor_passed": brightness_ok,
                                "ranking": {"goal_count": 1, "utility": 0.8},
                            },
                            "presentation_score": {
                                "policy": STAGE7_CANDIDATE_RANKING_POLICY,
                                "score": 1.0 - index * 0.05,
                            },
                            "subject_chroma_retention": gate,
                        }
                    )
                accepted = [
                    attempt
                    for attempt in attempts
                    if attempt["allowed_as_final"]
                ]
                selected = min(
                    accepted,
                    key=Stage6ServiceMixin._stage7_candidate_selection_key,
                )
                self.assertEqual(selected["name"], vector["expected"])

    def test_star_preserve_uses_subject_only_advisory_band(self) -> None:
        report = self._chroma_gate(
            0.278,
            profile="star_colour_preserve",
        )
        self.assertTrue(report["accepted"])
        self.assertTrue(report["advisory"])
        self.assertEqual(
            report["quality_gates"]["non_background"]["status"],
            "not_applicable",
        )

    def test_low_chroma_na_requires_both_measurement_floors(self) -> None:
        accepted = self._chroma_gate(
            0.05,
            source_saturation=0.019,
            source_opponent_rms=0.00009,
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["status"], "not_applicable")

        rejected = self._chroma_gate(
            0.05,
            source_saturation=0.019,
            source_opponent_rms=0.00011,
        )
        self.assertFalse(rejected["accepted"])
        self.assertNotEqual(rejected["status"], "not_applicable")

    def test_chroma_gate_fails_closed_for_route_or_frozen_roi_tamper(self) -> None:
        missing_route = self._chroma_gate(1.0, semantics="unknown")
        self.assertEqual(
            missing_route["reason_code"],
            "rgb_auto_route_evidence_missing",
        )
        tampered = metrics.assess_subject_chroma_retention(
            _rendition(0.2, 0.18, mask_source="candidate_rebuilt_roi"),
            _rendition(0.2, 0.18),
            self.cfg,
            profile_name="generic_balanced",
            channel_semantics="broadband_rgb_osc",
        )
        self.assertFalse(tampered["accepted"])
        self.assertEqual(
            tampered["reason_code"],
            "frozen_roi_measurement_unavailable",
        )

    def test_visible_noise_nominal_advisory_and_hard_thresholds(self) -> None:
        nominal = metrics.stage7_visible_noise_quality_gate(0.025, self.cfg)
        advisory = metrics.stage7_visible_noise_quality_gate(0.030, self.cfg)
        hard = metrics.stage7_visible_noise_quality_gate(0.0376, self.cfg)
        self.assertEqual(nominal["status"], "ok")
        self.assertEqual(advisory["status"], "advisory")
        self.assertFalse(advisory["hard_failed"])
        self.assertEqual(hard["status"], "rejected")
        self.assertTrue(hard["hard_failed"])

    def test_config_cannot_weaken_chroma_noise_or_na_gates(self) -> None:
        weakened = SimpleNamespace(
            stage7_subject_chroma_retention_nominal=0.05,
            stage7_subject_chroma_retention_hard_min=0.05,
            stage7_non_background_chroma_retention_nominal=0.05,
            stage7_non_background_chroma_retention_hard_min=0.05,
            stage7_star_preserve_chroma_retention_nominal=0.05,
            stage7_star_preserve_chroma_retention_hard_min=0.05,
            stage7_low_chroma_source_saturation_max=1.0,
            stage7_low_chroma_source_opponent_rms_max=1.0,
            stage7_visible_noise_score_max=1.0,
            stage7_visible_noise_score_hard_max=1.0,
        )
        diffuse = self._chroma_gate(
            0.54,
            non_background_ratio=0.29,
            cfg=weakened,
        )
        self.assertFalse(diffuse["accepted"])
        self.assertEqual(diffuse["limits"]["subject_hard_min"], 0.55)
        self.assertEqual(
            diffuse["limits"]["non_background_hard_min"],
            0.30,
        )
        stellar = self._chroma_gate(
            0.22,
            profile="star_colour_preserve",
            cfg=weakened,
        )
        self.assertFalse(stellar["accepted"])
        self.assertEqual(stellar["limits"]["subject_hard_min"], 0.23)

        not_na = self._chroma_gate(
            0.05,
            source_saturation=0.021,
            source_opponent_rms=0.00011,
            cfg=weakened,
        )
        self.assertFalse(not_na["low_chroma_not_applicable"])
        self.assertFalse(not_na["accepted"])
        self.assertEqual(
            not_na["limits"]["low_chroma_source_saturation_max"],
            0.02,
        )
        self.assertEqual(
            not_na["limits"]["low_chroma_source_opponent_rms_max"],
            0.0001,
        )

        noise = metrics.stage7_visible_noise_quality_gate(0.0376, weakened)
        self.assertTrue(noise["hard_failed"])
        self.assertEqual(noise["accepted_limit"], 0.025)
        self.assertEqual(noise["hard_limit"], 0.0375)

    def _adaptive_contract(self) -> tuple[np.ndarray, dict, dict]:
        ramp = np.linspace(0.003, 0.42, 32 * 32, dtype=np.float32).reshape(
            32,
            32,
        )
        source = np.stack((ramp, ramp * 0.91, ramp * 0.78), axis=0)
        adaptation = {
            "preview_calibration": {
                "candidate_a": {
                    "target_p50": 0.19,
                    "target_p99": 0.82,
                }
            }
        }
        calibration = metrics.calibrate_adaptive_quantile_stretch(
            source,
            adaptation,
            self.cfg,
            source_stem="stage6_starless",
        )
        selected = {
            "name": "cand_quantile",
            "method": "adaptive_quantile",
            "params": {"calibration": calibration},
        }
        transfer = _stage7_matched_domain_transfer_contract(selected, {})
        return source, calibration, transfer

    def test_adaptive_quantile_v4_authenticates_all_bindings(self) -> None:
        source, calibration, transfer = self._adaptive_contract()
        self.assertEqual(
            calibration["schema"],
            metrics.ADAPTIVE_QUANTILE_CALIBRATION_SCHEMA,
        )
        self.assertEqual(
            transfer["schema"],
            metrics.STAGE7_MATCHED_DOMAIN_TRANSFER_SCHEMA_V4,
        )
        validated = (
            metrics.validate_adaptive_quantile_matched_domain_transfer(
                transfer
            )
        )
        replayed = metrics.apply_adaptive_quantile_stretch(
            source,
            validated["calibration"]["calibration"],
        )
        self.assertTrue(np.all(np.isfinite(replayed)))
        self.assertGreater(float(np.median(replayed)), float(np.median(source)))
        for path in (
            ("calibration", "input_anchors"),
            ("source_binding", "source_stem"),
            ("winner_binding", "selected_candidate_id"),
            ("chain_contract", "sha256"),
        ):
            with self.subTest(path=path):
                forged = copy.deepcopy(transfer)
                container = forged[path[0]]
                if path[1] == "input_anchors":
                    container[path[1]][2] += 0.001
                else:
                    container[path[1]] = "tampered"
                with self.assertRaises(ValueError):
                    metrics.validate_adaptive_quantile_matched_domain_transfer(
                        forged
                    )

        forged_strategy = copy.deepcopy(transfer)
        forged_strategy["calibration"]["strategy_contract"]["mode"] = (
            "unbounded_reference_curve"
        )
        with self.assertRaises(ValueError):
            metrics.validate_adaptive_quantile_matched_domain_transfer(
                forged_strategy
            )

    def test_adaptive_quantile_rejects_the_m31_steep_shadow_curve(self) -> None:
        with self.assertRaisesRegex(ValueError, "derivative exceeds"):
            metrics._adaptive_quantile_curve_contract(
                [0.0, 1.0, 50.0, 100.0],
                [0.0, 0.025704, 0.025840, 1.0],
                [0.0, 0.018, 0.204, 0.78],
            )

    def test_bounded_m31_like_search_repairs_brightness_without_gate_relief(
        self,
    ) -> None:
        rng = np.random.default_rng(12)
        height, width = 128, 160
        yy, xx = np.mgrid[:height, :width]
        rho = (
            ((xx - width * 0.52) / (width * 0.24)) ** 2
            + ((yy - height * 0.49) / (height * 0.18)) ** 2
        )
        galaxy = 0.0015 * np.exp(-rho * 1.8)
        pedestal = 0.02583 + rng.normal(
            0.0,
            8e-6,
            (height, width),
        )
        source = np.stack(
            (
                pedestal + 1.10 * galaxy,
                pedestal + galaxy,
                pedestal + 0.82 * galaxy,
            )
        ).astype(np.float32)
        subject = (rho < 1.1).astype(np.float32)
        background = (rho > 1.5).astype(np.float32)
        frozen_masks = {
            "subject_mask": subject,
            "galaxy_signal_mask": subject,
            "background_mask": background,
        }
        linked_preview_pixels = metrics.apply_linked_mtf(
            source,
            0.02555,
            0.0007,
            1.0,
        )
        linked_preview = metrics.measure_frozen_rendition_metrics(
            linked_preview_pixels,
            frozen_masks,
        )
        base_candidate = metrics.apply_linked_mtf(
            source,
            0.0250,
            0.0036,
            1.0,
        )
        base_brightness = metrics.subject_brightness_selection(
            metrics.measure_frozen_rendition_metrics(
                base_candidate,
                frozen_masks,
            ),
            linked_preview,
            profile_name="galaxy_core_halo_balance",
        )
        self.assertFalse(base_brightness["formal_floor_passed"])

        adaptation = {
            "preview_calibration": {
                "candidate_a": {
                    "target_p50": 0.20,
                    "target_p99": 0.69,
                },
                "candidate_b": {
                    "mtf": [0.0250, 0.0036, 1.0],
                },
            },
            "conditional_source_profile": {
                "background_median": 0.02583,
            },
            "target_aware": {"name": "galaxy_core_halo_balance"},
        }
        calibration = metrics.calibrate_adaptive_quantile_stretch(
            source,
            adaptation,
            self.cfg,
            source_stem="stage6_starless",
            frozen_masks=frozen_masks,
            linked_preview=linked_preview,
            profile_name="galaxy_core_halo_balance",
            channel_semantics="broadband_rgb_osc",
        )
        self.assertEqual(calibration["status"], "ok", calibration)
        strategy = calibration["strategy_contract"]
        self.assertEqual(
            strategy["mode"],
            "frozen_roi_bounded_mtf_search",
        )
        selected = strategy["selected"]
        self.assertTrue(selected["accepted"])
        self.assertGreaterEqual(selected["subject_p50_retention"], 0.58)
        self.assertGreaterEqual(selected["subject_lift_retention"], 0.50)
        self.assertLessEqual(selected["visible_noise_score"], 0.025)
        self.assertLessEqual(selected["background_chroma_load"], 0.06)
        self.assertLessEqual(selected["color_vector_p95"], 0.08)
        nominal_attempts = [
            attempt
            for attempt in strategy["attempts"]
            if attempt["accepted"]
            and attempt["visible_noise_status"] == "ok"
        ]
        self.assertTrue(nominal_attempts)
        conservative_attempts = [
            attempt
            for attempt in nominal_attempts
            if attempt["background_above_preview"] is False
        ]
        presentation_pool = conservative_attempts or nominal_attempts
        self.assertAlmostEqual(
            selected["background_presentation_distance"],
            min(
                attempt["background_presentation_distance"]
                for attempt in presentation_pool
            ),
        )
        self.assertLessEqual(
            calibration["curve_contract"]["maximum_derivative"],
            metrics.ADAPTIVE_QUANTILE_MAX_DERIVATIVE,
        )

        repaired = metrics.apply_adaptive_quantile_stretch(
            source,
            calibration,
        )
        repaired_rendition = metrics.measure_frozen_rendition_metrics(
            repaired,
            frozen_masks,
        )
        repaired_brightness = metrics.subject_brightness_selection(
            repaired_rendition,
            linked_preview,
            profile_name="galaxy_core_halo_balance",
        )
        repaired_chroma = metrics.assess_subject_chroma_retention(
            repaired_rendition,
            linked_preview,
            self.cfg,
            profile_name="galaxy_core_halo_balance",
            channel_semantics="broadband_rgb_osc",
        )
        repaired_noise = metrics.assess_frozen_background_luma_noise_growth(
            source,
            repaired,
            frozen_masks,
            self.cfg,
            expected_candidate=repaired,
        )
        self.assertTrue(repaired_brightness["formal_floor_passed"])
        self.assertTrue(repaired_chroma["accepted"])
        self.assertTrue(repaired_noise["accepted"])

    def test_stage9_replays_only_the_authenticated_adaptive_v4_curve(self) -> None:
        source, calibration, transfer = self._adaptive_contract()
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = SimpleNamespace(
                _stage7_matched_domain_transfer=transfer,
                _stage7_closed_form_mtf_reference=None,
                process_dir=Path(temporary),
            )
            resolved = (
                stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                    pipeline
                )
            )

        self.assertEqual(resolved["status"], "ready", resolved)
        self.assertEqual(
            resolved["method"],
            "linked_piecewise_linear_quantile_curve",
        )
        expected = metrics.apply_adaptive_quantile_stretch(source, calibration)
        replayed = stage9_star_remixing._stage9_apply_matched_domain_transfer(
            source,
            resolved,
        )
        np.testing.assert_array_equal(replayed, expected)

        forged = copy.deepcopy(transfer)
        forged["winner_binding"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            rejected = (
                stage9_star_remixing._stage9_resolve_matched_domain_transfer(
                    SimpleNamespace(
                        _stage7_matched_domain_transfer=forged,
                        _stage7_closed_form_mtf_reference=None,
                        process_dir=Path(temporary),
                    )
                )
            )
        self.assertEqual(rejected["status"], "unavailable")
        self.assertEqual(
            rejected["reason_code"],
            "stage9_adaptive_quantile_transfer_invalid",
        )


class _ReferenceProcessor(Stage6ServiceMixin):
    def __init__(self, root: Path, pixels: np.ndarray) -> None:
        self.process_dir = root
        self.pixels = np.asarray(pixels, dtype=np.float32)
        self.saved: dict[str, np.ndarray] = {}
        self.reports: dict[str, dict] = {}
        self.siril = SimpleNamespace(
            get_image_pixeldata=lambda preview=False: self.pixels.copy()
        )
        source = root / "stage6_starless.fit"
        source.write_bytes(b"stage6-source")
        self._stage7_source_binding = {
            "stem": "stage6_starless",
            "file": source.name,
            "container_sha256": self._sha256_file(source),
            "pixel_sha256": metrics.stage7_pixel_sha256(self.pixels),
            "shape": list(self.pixels.shape),
            "dtype": str(self.pixels.dtype),
        }

    def _save_stage_output(self, stem: str) -> bool:
        self.saved[stem] = self.pixels.copy()
        (self.process_dir / f"{stem}.fit").write_bytes(
            b"reference-" + stem.encode("ascii")
        )
        return True

    def _read_image_by_stem(self, stem: str):
        value = self.saved.get(stem)
        return None if value is None else value.copy()

    def _write_stage_json(self, filename: str, payload: dict) -> None:
        self.reports[filename] = copy.deepcopy(payload)

    def _sha256_file(self, path: Path):
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _short_text(value, limit: int = 180) -> str:
        return str(value)[:limit]


class Stage7PresentationReferenceTests(unittest.TestCase):
    def test_reference_binds_container_pixels_source_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.full((3, 24, 24), 0.12, dtype=np.float32)
            pixels[0, 8:16, 8:16] = 0.24
            processor = _ReferenceProcessor(root, pixels)
            selected_path = root / "stage7_cand_a.fit"
            selected_path.write_bytes(b"selected")
            pixel_sha = metrics.stage7_pixel_sha256(pixels)
            selected = {
                "name": "cand_a",
                "stem": "stage7_cand_a",
                "method": "iterative_masked_mtf",
                "adaptation": {
                    "target_aware": {"name": "star_colour_preserve"}
                },
                "artifact_identity": {
                    "container_sha256": hashlib.sha256(b"selected").hexdigest(),
                    "pixel_sha256": pixel_sha,
                },
            }
            report = processor._stage7_freeze_presentation_reference(
                selected,
                source_stem="stage6_starless",
                matched_domain_transfer={"status": "unavailable"},
            )
            self.assertTrue(report["accepted"])
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["artifact"]["pixel_sha256"], pixel_sha)
            self.assertEqual(
                report["selected_candidate"]["container_sha256"],
                hashlib.sha256(b"selected").hexdigest(),
            )
            self.assertTrue(
                (root / "stage7_presentation_reference.fit").is_file()
            )
            self.assertIn(
                "stage7_presentation_reference.json",
                processor.reports,
            )

            formal_path = root / "stage7_stretched.fit"
            formal_path.write_bytes(b"formal-stage7")
            bound = (
                stage7_stretching._bind_stage7_formal_presentation_reference(
                    processor,
                    "stage7_stretched",
                )
            )
            self.assertTrue(bound["accepted"])
            self.assertEqual(
                bound["source_artifact"]["file"],
                formal_path.name,
            )
            binding_payload = {
                "linear_source": bound["linear_source"],
                "selected_candidate": bound["selected_candidate"],
                "matched_domain": bound["matched_domain"],
                "formal_source_artifact": bound["source_artifact"],
            }
            self.assertEqual(
                bound["source_binding_sha256"],
                metrics.canonical_json_sha256(binding_payload),
            )
            unsigned = dict(bound)
            report_sha = unsigned.pop("report_sha256")
            self.assertEqual(
                report_sha,
                metrics.canonical_json_sha256(unsigned),
            )

    def test_reference_fails_closed_on_selected_pixel_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.full((3, 24, 24), 0.12, dtype=np.float32)
            processor = _ReferenceProcessor(root, pixels)
            selected_path = root / "stage7_cand_a.fit"
            selected_path.write_bytes(b"selected")
            selected = {
                "name": "cand_a",
                "stem": "stage7_cand_a",
                "method": "iterative_masked_mtf",
                "adaptation": {
                    "target_aware": {"name": "star_colour_preserve"}
                },
                "artifact_identity": {
                    "container_sha256": hashlib.sha256(b"selected").hexdigest(),
                    "pixel_sha256": "0" * 64,
                },
            }
            report = processor._stage7_freeze_presentation_reference(
                selected,
                source_stem="stage6_starless",
                matched_domain_transfer={"status": "unavailable"},
            )
            self.assertFalse(report["accepted"])
            self.assertEqual(report["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
