#!/usr/bin/env python3
"""Tests for the independent external-reference QA tool."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "script" / "five_target_reference_qa.py"
)
SPEC = importlib.util.spec_from_file_location("five_target_reference_qa", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def _synthetic_reference(size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(20260828)
    yy, xx = np.indices((size, size), dtype=np.float32)
    background = 0.035 + 0.004 * np.sin(xx / 37.0) * np.cos(yy / 43.0)
    nebula = 0.19 * np.exp(
        -(((xx - size * 0.54) / (size * 0.24)) ** 2)
        -(((yy - size * 0.51) / (size * 0.18)) ** 2)
    )
    dust = 0.035 * np.sin((xx + yy) / 13.0) * (nebula / max(nebula.max(), 1e-6))
    rgb = np.stack(
        (
            background + nebula * 1.25 + dust,
            background + nebula * 0.72 + dust * 0.5,
            background + nebula * 0.48,
        ),
        axis=2,
    )
    # Distinct, non-circular knots make SIFT correspondence deterministic;
    # a field of only symmetric Gaussian stars is intentionally ambiguous.
    for _ in range(48):
        center_x = float(rng.uniform(24, size - 24))
        center_y = float(rng.uniform(24, size - 24))
        sigma_x = float(rng.uniform(2.0, 7.0))
        sigma_y = float(rng.uniform(1.0, 3.0))
        angle = float(rng.uniform(0.0, np.pi))
        local_x = (xx - center_x) * np.cos(angle) + (yy - center_y) * np.sin(angle)
        local_y = -(xx - center_x) * np.sin(angle) + (yy - center_y) * np.cos(angle)
        knot = float(rng.uniform(0.025, 0.075)) * np.exp(
            -(local_x**2 / (2.0 * sigma_x**2) + local_y**2 / (2.0 * sigma_y**2))
        )
        rgb += knot[:, :, None] * rng.uniform(0.7, 1.2, size=3)[None, None, :]
    for _ in range(320):
        x = int(rng.integers(12, size - 12))
        y = int(rng.integers(12, size - 12))
        amplitude = float(rng.uniform(0.20, 0.75))
        sigma = float(rng.uniform(0.8, 1.5))
        star = amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)
        )
        colour = rng.uniform(0.82, 1.12, size=3)
        rgb += star[:, :, None] * colour[None, None, :]
    rgb += rng.normal(0.0, 0.002, rgb.shape).astype(np.float32)
    return np.clip(rgb, 0.0, 0.97)


def _baseline_from(reference: np.ndarray) -> np.ndarray:
    gray = np.sum(reference * np.array([0.2126, 0.7152, 0.0722]), axis=2)
    muted = np.stack(
        [gray * 0.78 + reference[:, :, channel] * 0.22 for channel in range(3)],
        axis=2,
    )
    muted = gaussian_filter(muted, sigma=(1.25, 1.25, 0.0))
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    mottling = 0.018 * np.sin(xx / 18.0) * np.sin(yy / 23.0)
    rng = np.random.default_rng(17)
    muted = 0.62 * muted + 0.025
    muted += mottling[:, :, None]
    muted += rng.normal(0.0, 0.012, muted.shape).astype(np.float32)
    return np.clip(muted, 0.0, 0.97)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _sha256(path: Path) -> str:
    return qa._sha256(path)


def _write_pipeline_result(
    run_root: Path,
    artifact: Path,
    *,
    formal: bool,
) -> None:
    sha256 = _sha256(artifact)
    payload = {
        "schema": "starun.pipeline-result.v2",
        "status": "partial_success",
        "review_required": False,
        "review_requirements": [],
        "outputs": {
            artifact.name: {
                "path": artifact.name,
                "sha256": sha256,
                "size": artifact.stat().st_size,
            }
        },
    }
    if formal:
        payload.update(
            delivery_eligible=True,
            delivery_gates={
                "schema": "starun.final-delivery-gates.v1",
                "legacy_delivery_contract": False,
                "scientific": {"accepted": True},
                "presentation": {"accepted": True},
                "artifacts": {
                    "accepted": True,
                    "formal_count": 1,
                    "formal_outputs": [artifact.name],
                },
                "review": {"accepted": True},
                "formal_delivery_accepted": True,
            },
        )
    (run_root / "pipeline-result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _visual_review(optimized: Path, *, passed: bool = True) -> dict[str, object]:
    return {
        "optimized_sha256": _sha256(optimized),
        **{name: passed for name in qa.VISUAL_REVIEW_CHECKS},
    }


def _write_five_target_manifest(root: Path) -> Path:
    entries = []
    reference_root = root / "references"
    reference_root.mkdir(exist_ok=True)
    for index, (target, profile) in enumerate(qa.EXPECTED_TARGET_PROFILES.items()):
        baseline_root = root / f"baseline-{target}"
        optimized_root = root / f"optimized-{target}"
        baseline_root.mkdir(exist_ok=True)
        optimized_root.mkdir(exist_ok=True)
        baseline = baseline_root / f"{target}-baseline.png"
        optimized = optimized_root / f"{target}-optimized.png"
        reference = reference_root / f"{target}-reference.png"
        _write_rgb(
            baseline,
            np.full((32, 32, 3), (25 + index) / 255.0, dtype=np.float32),
        )
        _write_rgb(
            optimized,
            np.full((32, 32, 3), (75 + index) / 255.0, dtype=np.float32),
        )
        _write_rgb(
            reference,
            np.full((32, 32, 3), (125 + index) / 255.0, dtype=np.float32),
        )
        _write_pipeline_result(baseline_root, baseline, formal=False)
        _write_pipeline_result(optimized_root, optimized, formal=True)
        entries.append(
            {
                "target": target,
                "profile": profile,
                "baseline": str(baseline),
                "optimized": str(optimized),
                "reference": str(reference),
                "baseline_run_root": str(baseline_root),
                "optimized_run_root": str(optimized_root),
                "visual_artifact_review": _visual_review(optimized),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": qa.MANIFEST_SCHEMA, "entries": entries}),
        encoding="utf-8",
    )
    return manifest


class FiveTargetReferenceQATests(unittest.TestCase):
    def test_dense_field_registration_uses_mutual_ratio_matches(self) -> None:
        try:
            cv2 = qa._load_cv2()
        except qa.QAError as exc:
            self.skipTest(str(exc))
        reference = _synthetic_reference(768)
        source = np.roll(np.roll(reference, 9, axis=0), -13, axis=1)
        registered, valid, report = qa._register_to_reference(
            source,
            reference,
            cv2,
            source_shape=source.shape,
            reference_shape=reference.shape,
        )
        self.assertEqual(report["match_policy"], "mutual_bidirectional_ratio")
        self.assertFalse(report["fallback_used"])
        self.assertGreaterEqual(report["inlier_ratio"], qa.MIN_REGISTRATION_INLIER_RATIO)
        self.assertGreaterEqual(report["overlap_ratio"], qa.MIN_REGISTRATION_OVERLAP)
        self.assertEqual(registered.shape, reference.shape)
        self.assertEqual(valid.shape, reference.shape[:2])

    def test_missing_sift_fails_closed(self) -> None:
        with patch.object(qa, "_load_cv2", side_effect=qa.QAError("missing SIFT")):
            with self.assertRaisesRegex(qa.QAError, "missing SIFT"):
                qa._load_cv2()

    def test_manifest_requires_all_three_image_roles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": qa.MANIFEST_SCHEMA,
                        "entries": [
                            {
                                "target": "M31",
                                "baseline": "missing.png",
                                "optimized": "missing.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(qa.QAError):
                qa.load_manifest(manifest)

    def test_synthetic_optimization_is_measured_without_reference_copy(self) -> None:
        try:
            cv2 = qa._load_cv2()
        except qa.QAError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_dir = root / "inputs"
            input_dir.mkdir()
            reference = _synthetic_reference()
            baseline = _baseline_from(reference)
            yy, xx = np.indices(reference.shape[:2], dtype=np.float32)
            independent_residual = (
                0.00035 * np.sin(xx / 29.0) * np.cos(yy / 31.0)
            )
            optimized = np.clip(
                reference * 0.997 + independent_residual[:, :, None],
                0.0,
                0.97,
            )
            paths = {
                "baseline": input_dir / "baseline.png",
                "optimized": input_dir / "optimized.png",
                "reference": input_dir / "reference.png",
            }
            _write_rgb(paths["baseline"], baseline)
            _write_rgb(paths["optimized"], optimized)
            _write_rgb(paths["reference"], reference)
            self.assertNotEqual(
                _sha256(paths["optimized"]), _sha256(paths["reference"])
            )
            result = qa.evaluate_entry(
                {
                    **paths,
                    "target": "synthetic-nebula",
                    "profile": "emission_nebula",
                    "visual_artifact_review": {"passed": True},
                    "baseline_provenance": {},
                    "optimized_provenance": {},
                },
                cv2,
            )

            self.assertEqual(result["status"], "accepted")
            self.assertGreaterEqual(
                result["improved_dimension_count"], 4
            )
            self.assertTrue(result["structure_correlation_guard_passed"])

    def test_five_target_run_writes_identity_bound_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = _write_five_target_manifest(root)
            output_dir = root / "reports"

            def accepted_stub(entry, _cv2):
                return {
                    "target": entry["target"],
                    "status": "accepted",
                    "accepted": True,
                    "quantitative_accepted": True,
                    "visual_artifact_review_passed": True,
                    "improved_dimension_count": 5,
                    "dimensions": {
                        "color": {"improved": True, "optimized_reference_ratio": 1.0},
                        "detail": {
                            "improved": True,
                            "structure_correlation_change": 0.1,
                        },
                        "stars": {"improved": True, "optimized_reference_ratio": 1.0},
                        "noise": {
                            "improved": True,
                            "high_frequency": {"optimized_reference_ratio": 1.0},
                            "low_frequency": {"optimized_reference_ratio": 1.0},
                        },
                        "contrast": {"improved": True},
                    },
                }

            with patch.object(qa, "_load_cv2", return_value=object()), patch.object(
                qa, "evaluate_entry", side_effect=accepted_stub
            ):
                report = qa.run_qa(manifest_path, output_dir)

            self.assertTrue(report["accepted"])
            self.assertTrue(report["coverage_gate"]["accepted"])
            self.assertTrue(report["path_isolation_gate"]["accepted"])
            self.assertTrue(report["external_reference_used"])
            self.assertFalse(report["production_feedback"])
            self.assertFalse(report["production_pixels_written"])
            self.assertEqual(report["summary"]["accepted_count"], 5)
            self.assertEqual(len(report["artifacts"]["contact_sheet_sha256"]), 64)
            for name in (
                "five_target_reference_qa.json",
                "five_target_reference_qa.md",
                "five_target_reference_qa.jpg",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_manifest_rejects_reference_role_alias_and_equal_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_five_target_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            first = payload["entries"][0]
            first["optimized"] = first["reference"]
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(qa.QAError, "multiple roles"):
                qa.load_manifest(manifest)

            manifest = _write_five_target_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            first = payload["entries"][0]
            optimized = Path(first["optimized"])
            reference = Path(first["reference"])
            optimized.write_bytes(reference.read_bytes())
            _write_pipeline_result(Path(first["optimized_run_root"]), optimized, formal=True)
            first["visual_artifact_review"] = _visual_review(optimized)
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(qa.QAError, "equal the reference"):
                qa.load_manifest(manifest)

    def test_manifest_requires_five_locked_profiles_and_formal_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_five_target_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["entries"].pop()
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(qa.QAError, "coverage mismatch"):
                qa.load_manifest(manifest)

            manifest = _write_five_target_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["entries"][0]["profile"] = "generic"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(qa.QAError, "requires profile"):
                qa.load_manifest(manifest)

            manifest = _write_five_target_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            first = payload["entries"][0]
            optimized = Path(first["optimized"])
            _write_pipeline_result(Path(first["optimized_run_root"]), optimized, formal=False)
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(qa.QAError, "formal delivery"):
                qa.load_manifest(manifest)

    def test_manifest_requires_strict_false_review_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_five_target_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            first = payload["entries"][0]
            result_path = (
                Path(first["optimized_run_root"]) / "pipeline-result.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["review_required"] = "false"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            with self.assertRaisesRegex(qa.QAError, "formal delivery"):
                qa.load_manifest(manifest)

    def test_nonregression_has_no_unlocked_absolute_slack(self) -> None:
        self.assertTrue(qa._nonregressed(0.01, 0.011))
        self.assertFalse(qa._nonregressed(0.01, 0.011001))
        self.assertFalse(qa._nonregressed(0.0, 1.0e-6))

    def test_low_frequency_metric_retains_directional_gradient(self) -> None:
        size = 256
        yy, xx = np.indices((size, size), dtype=np.float32)
        gray = 0.05 + 0.10 * xx / (size - 1) + 0.02 * yy / (size - 1)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        background = np.zeros((size, size), dtype=bool)
        background[:, : size // 2] = True
        subject = np.zeros_like(background)
        subject[:, size // 2 :] = True
        metrics = qa._image_metrics(
            image,
            {"valid": np.ones_like(background), "background": background, "subject": subject},
        )
        self.assertGreater(
            metrics["background_low_frequency_residual_sigma"],
            metrics["background_low_frequency_detrended_sigma"] * 10.0,
        )

    def test_visual_review_is_per_check_and_sha_bound(self) -> None:
        payload = {
            "optimized_sha256": "a" * 64,
            **{name: True for name in qa.VISUAL_REVIEW_CHECKS},
        }
        payload["no_new_halo_or_dark_rim"] = False
        review = qa._visual_review_payload(payload, "a" * 64)
        self.assertFalse(review["passed"])
        with self.assertRaisesRegex(qa.QAError, "SHA"):
            qa._visual_review_payload(payload, "b" * 64)

    def test_report_directory_cannot_be_inside_production_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_five_target_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            forbidden = Path(payload["entries"][0]["optimized_run_root"]) / "qa"
            with self.assertRaisesRegex(qa.QAError, "outside every production run"):
                qa.run_qa(manifest, forbidden)

    def test_visual_artifact_gate_does_not_auto_promote(self) -> None:
        accepted, status = qa._acceptance_status(True, False)

        self.assertFalse(accepted)
        self.assertEqual(status, "review_required")


if __name__ == "__main__":
    unittest.main()
