from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import star_halo_guard  # noqa: E402


class Stage6StarHaloGuardTests(unittest.TestCase):
    @staticmethod
    def _fixture(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.indices((size, size), dtype=np.float32)
        distance = np.sqrt((yy - 64.0) ** 2 + (xx - 64.0) ** 2)
        star = np.exp(-0.5 * (distance / 2.0) ** 2).astype(np.float32)
        halo = (0.04 * np.exp(-0.5 * (distance / 8.0) ** 2)).astype(np.float32)
        starless = np.repeat((0.08 + halo)[None, ...], 3, axis=0)
        starmask = np.repeat(star[None, ...], 3, axis=0)
        return starless, starmask

    def test_builds_adaptive_guard_for_compact_star(self) -> None:
        starless, starmask = self._fixture()
        guard, report = star_halo_guard.build_star_halo_guard(starless, starmask)

        self.assertEqual(guard.shape, (128, 128))
        self.assertGreater(float(guard[64, 64]), 0.5)
        self.assertGreater(report["component_count"], 0)
        self.assertGreater(report["coverage"], 0.0)

    def test_asymmetric_blue_nebula_outside_guard_is_preserved(self) -> None:
        starless, starmask = self._fixture()
        starless[2, 20:48, 88:120] += 0.12
        guard, report = star_halo_guard.build_star_halo_guard(starless, starmask)

        self.assertLess(float(np.max(guard[20:48, 100:120])), 0.05)
        self.assertEqual(report["hard_anomaly_count"], 0)

    def test_persist_verify_and_sha_tamper_rejection(self) -> None:
        starless, starmask = self._fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            starless_path = root / "stage6_starless.fit"
            starmask_path = root / "starmask.fit"
            fits.PrimaryHDU(starless).writeto(starless_path)
            fits.PrimaryHDU(starmask).writeto(starmask_path)
            pipeline = types.SimpleNamespace(process_dir=root, _run_id="run-1")

            report = star_halo_guard.persist_stage6_guard(
                pipeline,
                starless_path=starless_path,
                starmask_path=starmask_path,
            )
            guard, verified = star_halo_guard.verify_stage6_guard(
                pipeline,
                (128, 128),
            )
            self.assertIsNotNone(guard)
            self.assertEqual(verified["artifact_sha256"], report["artifact_sha256"])

            artifact = root / star_halo_guard.ARTIFACT_NAME
            artifact.write_bytes(artifact.read_bytes() + b"tampered")
            guard, rejected = star_halo_guard.verify_stage6_guard(
                pipeline,
                (128, 128),
            )
            self.assertIsNone(guard)
            self.assertEqual(
                rejected["reason_code"],
                "stage6_star_halo_guard_lineage_unverified",
            )

    def test_source_lineage_tamper_is_rejected(self) -> None:
        starless, starmask = self._fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            starless_path = root / "stage6_starless.fit"
            starmask_path = root / "starmask.fit"
            fits.PrimaryHDU(starless).writeto(starless_path)
            fits.PrimaryHDU(starmask).writeto(starmask_path)
            pipeline = types.SimpleNamespace(
                process_dir=root,
                _run_id="run-1",
                starmask_file=starmask_path,
            )
            star_halo_guard.persist_stage6_guard(
                pipeline,
                starless_path=starless_path,
                starmask_path=starmask_path,
            )
            starless_path.write_bytes(starless_path.read_bytes() + b"tampered")

            guard, rejected = star_halo_guard.verify_stage6_guard(
                pipeline,
                (128, 128),
            )

            self.assertIsNone(guard)
            self.assertIn("starless source SHA256 mismatch", rejected["error"])

    def test_guard_deducts_only_protected_support(self) -> None:
        starless, starmask = self._fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            starless_path = root / "stage6_starless.fit"
            starmask_path = root / "starmask.fit"
            fits.PrimaryHDU(starless).writeto(starless_path)
            fits.PrimaryHDU(starmask).writeto(starmask_path)
            pipeline = types.SimpleNamespace(
                process_dir=root,
                _run_id="run-1",
                _stage8_handoff={"star_halo_guard": {"status": "ok"}},
            )
            star_halo_guard.persist_stage6_guard(
                pipeline,
                starless_path=starless_path,
                starmask_path=starmask_path,
            )
            subject = np.ones((128, 128), dtype=np.float32)
            masks = star_halo_guard.apply_guard_to_masks(
                pipeline,
                {"gray": starless[0], "subject_mask": subject},
            )

            self.assertLess(float(masks["subject_mask"][64, 64]), 0.1)
            self.assertAlmostEqual(float(masks["subject_mask"][10, 10]), 1.0)
            self.assertTrue(pipeline._stage8_star_halo_guard_verified)

    def test_color_gate_rejects_local_chroma_and_half_delta_can_pass(self) -> None:
        baseline = np.full((3, 64, 64), 0.10, dtype=np.float32)
        guard = np.zeros((64, 64), dtype=np.float32)
        guard[20:44, 20:44] = 1.0
        candidate = baseline.copy()
        candidate[0, 20:44, 20:44] += 0.010

        rejected = star_halo_guard.assess_candidate(
            baseline,
            candidate,
            guard,
            mode="color",
        )
        weakened = star_halo_guard.assess_candidate(
            baseline,
            baseline + 0.5 * (candidate - baseline),
            guard,
            mode="color",
        )

        self.assertFalse(rejected["accepted"])
        self.assertTrue(weakened["accepted"])


if __name__ == "__main__":
    unittest.main()
