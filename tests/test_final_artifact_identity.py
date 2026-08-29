from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import final_artifact_identity  # noqa: E402
import display_rendition  # noqa: E402
import managed_output  # noqa: E402
from stage8_starless_finish import (  # noqa: E402
    canonical_decoded_pixel_sha256,
    persisted_fits_decoded_pixel_sha256,
    pixel_sha256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_icc_profile() -> bytes:
    profile = bytearray(128)
    profile[:4] = (128).to_bytes(4, "big")
    profile[36:40] = b"acsp"
    return bytes(profile)


class FinalArtifactIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temporary_root = Path(self.temporary.name)

    @staticmethod
    def _source_pixels() -> np.ndarray:
        y_grid, x_grid = np.indices((18, 24), dtype=np.float32)
        x_grid /= np.float32(23.0)
        y_grid /= np.float32(17.0)
        return np.ascontiguousarray(
            np.stack(
                (
                    np.float32(0.08) + np.float32(0.52) * x_grid,
                    np.float32(0.09) + np.float32(0.44) * y_grid,
                    np.float32(0.07)
                    + np.float32(0.30) * (x_grid + y_grid) / np.float32(2.0),
                )
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _write_report(fixture: dict[str, Any]) -> None:
        report_path = fixture["process_dir"] / "managed_output_report.json"
        report_path.write_text(
            json.dumps(fixture["managed_report"], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _make_fixture(self, name: str) -> dict[str, Any]:
        root = self.temporary_root / name
        process = root / "process"
        process.mkdir(parents=True)
        basename = "M31_60sec_20260519_194350_processed"
        source = self._source_pixels()

        stage10_path = process / "stage10_final.fit"
        formal_fit_path = root / f"{basename}.fit"
        fits.PrimaryHDU(source).writeto(stage10_path)
        fits.PrimaryHDU(source).writeto(formal_fit_path)

        outputs: dict[str, dict[str, Any]] = {
            formal_fit_path.name: {"sha256": _sha256(formal_fit_path)},
        }
        source_pixel_sha = canonical_decoded_pixel_sha256(source)
        expected_derivative = np.ascontiguousarray(
            managed_output.canonical_managed_derivative_pixels(source),
            dtype=np.float32,
        )
        derivative_pixel_sha = pixel_sha256(expected_derivative)

        managed_artifacts = []
        for role, name_token, suffix in (
            ("display", "display", "png"),
            ("editable", "edit", "tif"),
        ):
            artifact_path = root / f"{basename}_{name_token}_srgb.{suffix}"
            if role == "display":
                managed_output.write_managed_display_png(artifact_path, source)
            else:
                managed_output.write_managed_edit_tiff(
                    artifact_path,
                    source,
                    icc_profile=_test_icc_profile(),
                )
            artifact_sha = _sha256(artifact_path)
            outputs[artifact_path.name] = {"sha256": artifact_sha}
            managed_artifacts.append(
                {
                    "role": role,
                    "name": artifact_path.name,
                    "status": "written",
                    "sha256": artifact_sha,
                    "pixel_chain": {
                        "schema": "starun.managed-output-pixel-chain.v1",
                        "accepted": True,
                        "source_pixel_sha256": source_pixel_sha,
                        "source_pixel_sha256_method": (
                            final_artifact_identity.MANAGED_SOURCE_METHOD
                        ),
                        "expected_pixel_sha256": derivative_pixel_sha,
                        "decoded_pixel_sha256": derivative_pixel_sha,
                        "decoded_pixel_sha256_method": (
                            final_artifact_identity.MANAGED_DERIVATIVE_METHOD
                        ),
                    },
                }
            )

        fixture = {
            "root": root,
            "process_dir": process,
            "basename": basename,
            "source": source,
            "outputs": outputs,
            "managed_report": {
                "schema": "starun.managed-output.v2",
                "status": "ready",
                "ready": True,
                "source_pixels": {
                    "checkpoint": "stage10_final.fit",
                    "pixel_sha256": source_pixel_sha,
                    "pixel_sha256_method": (
                        final_artifact_identity.MANAGED_SOURCE_METHOD
                    ),
                },
                "artifacts": managed_artifacts,
                "issues": [],
            },
        }
        self._write_report(fixture)
        return fixture

    def test_decoded_pixel_identity_ignores_icc_profile_image_extension(self):
        pixels = self._source_pixels()
        path = self.temporary_root / "stage8_with_icc.fit"
        fits.HDUList(
            [
                fits.PrimaryHDU(pixels),
                fits.ImageHDU(
                    np.frombuffer(_test_icc_profile(), dtype=np.uint8),
                    name="ICCProfile",
                ),
            ]
        ).writeto(path)

        self.assertEqual(
            persisted_fits_decoded_pixel_sha256(path),
            canonical_decoded_pixel_sha256(pixels),
        )
        np.testing.assert_array_equal(
            final_artifact_identity._decoded_stage10_pixels(path),
            pixels,
        )

    def test_decoded_pixel_identity_rejects_second_science_image(self):
        pixels = self._source_pixels()
        path = self.temporary_root / "stage8_ambiguous.fit"
        fits.HDUList(
            [
                fits.PrimaryHDU(pixels),
                fits.ImageHDU(pixels.copy(), name="SECOND_SCIENCE"),
            ]
        ).writeto(path)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            persisted_fits_decoded_pixel_sha256(path)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            final_artifact_identity._decoded_stage10_pixels(path)

    @staticmethod
    def _verify(
        fixture: dict[str, Any],
        *,
        output_format: str = "all",
        formal_basenames: list[str] | None = None,
    ) -> dict[str, Any]:
        return final_artifact_identity.verify_formal_artifacts(
            work_dir=fixture["root"],
            process_dir=fixture["process_dir"],
            outputs=fixture["outputs"],
            output_format=output_format,
            formal_basenames=(
                formal_basenames
                if formal_basenames is not None
                else [fixture["basename"]]
            ),
        )

    @staticmethod
    def _managed_artifact(
        fixture: dict[str, Any],
        role: str,
    ) -> dict[str, Any]:
        return next(
            artifact
            for artifact in fixture["managed_report"]["artifacts"]
            if artifact["role"] == role
        )

    def test_valid_stage10_and_root_fit_pass(self) -> None:
        fixture = self._make_fixture("valid-fit")

        report = self._verify(fixture, output_format="fit")

        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["issues"], [])
        self.assertEqual(
            report["formal_outputs"],
            [f"{fixture['basename']}.fit"],
        )
        self.assertEqual(len(report["scientific"]), 1)
        self.assertTrue(report["scientific"][0]["accepted"])

    def test_valid_stage10_fit_and_requested_managed_outputs_pass(self) -> None:
        fixture = self._make_fixture("valid-all")

        report = self._verify(fixture, output_format="all")

        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["issues"], [])
        self.assertEqual(
            report["formal_outputs"],
            sorted(
                (
                    f"{fixture['basename']}.fit",
                    f"{fixture['basename']}_display_srgb.png",
                    f"{fixture['basename']}_edit_srgb.tif",
                )
            ),
        )
        self.assertEqual(
            {item["role"] for item in report["managed"]},
            {"display", "editable"},
        )
        self.assertTrue(all(item["accepted"] for item in report["managed"]))

    def test_compacted_stage10_uses_exact_formal_fit_as_source_anchor(self) -> None:
        fixture = self._make_fixture("compacted-stage10")
        (fixture["process_dir"] / "stage10_final.fit").unlink()

        report = self._verify(fixture, output_format="all")

        self.assertTrue(report["accepted"], report)
        self.assertTrue(report["source"]["compacted_source"])
        self.assertEqual(
            report["source"]["role"],
            "compacted_scientific_archive_anchor",
        )
        self.assertEqual(
            report["source"]["artifact"],
            f"{fixture['basename']}.fit",
        )

    def test_compacted_stage10_anchor_tampering_fails_closed(self) -> None:
        fixture = self._make_fixture("compacted-stage10-tampered")
        (fixture["process_dir"] / "stage10_final.fit").unlink()
        fixture["managed_report"]["source_pixels"]["pixel_sha256"] = "0" * 64
        self._write_report(fixture)

        report = self._verify(fixture, output_format="all")

        self.assertFalse(report["accepted"], report)
        self.assertTrue(
            any(
                issue.startswith("stage10_source_identity_unavailable:")
                for issue in report["issues"]
            ),
            report,
        )

    def test_source_bound_display_rendition_is_replayed_for_identity(self) -> None:
        fixture = self._make_fixture("display-rendition")
        artifact = self._managed_artifact(fixture, "display")
        artifact_path = fixture["root"] / artifact["name"]
        source_display = np.flip(fixture["source"], axis=1)
        contract = display_rendition.build_linked_review_contract(
            source_display,
            reason="test_underexposed_source",
            source_stem="stage10_final",
        )
        rendered = display_rendition.apply_review_contract(
            source_display,
            contract,
        )
        managed_output.write_managed_display_png(
            artifact_path,
            np.flip(rendered, axis=1),
        )
        decoded = managed_output.read_managed_display_png(artifact_path)
        derivative_sha = pixel_sha256(
            np.ascontiguousarray(decoded, dtype=np.float32)
        )
        artifact["display_transform"] = contract
        artifact["sha256"] = _sha256(artifact_path)
        artifact["pixel_chain"]["expected_pixel_sha256"] = derivative_sha
        artifact["pixel_chain"]["decoded_pixel_sha256"] = derivative_sha
        fixture["outputs"][artifact_path.name]["sha256"] = _sha256(
            artifact_path
        )
        self._write_report(fixture)

        report = self._verify(fixture)

        self.assertTrue(report["accepted"], report)
        self.assertTrue(
            next(
                item
                for item in report["managed"]
                if item["role"] == "display"
            )["accepted"]
        )

        artifact["display_transform"]["luminance"]["gamma"] = 0.8
        self._write_report(fixture)
        tampered = self._verify(fixture)
        self.assertFalse(tampered["accepted"], tampered)
        self.assertIn(
            "managed_artifact_decoded_pixel_sha_mismatch",
            next(
                item
                for item in tampered["managed"]
                if item["role"] == "display"
            )["issues"],
        )

    def test_decodable_root_fit_with_different_pixels_is_rejected(self) -> None:
        fixture = self._make_fixture("different-fit-pixels")
        formal_path = fixture["root"] / f"{fixture['basename']}.fit"
        changed = fixture["source"].copy()
        changed[0, 4, 7] += np.float32(0.025)
        formal_path.unlink()
        fits.PrimaryHDU(changed).writeto(formal_path)
        fixture["outputs"][formal_path.name]["sha256"] = _sha256(formal_path)

        report = self._verify(fixture, output_format="fit")

        self.assertFalse(report["accepted"])
        self.assertIn(
            "stage10_decoded_pixel_sha_mismatch",
            report["scientific"][0]["issues"],
        )
        self.assertIn(
            f"scientific_fit_unverified:{formal_path.name}",
            report["issues"],
        )
        self.assertIn(
            "no_stage10_pixel_bound_scientific_fits",
            report["issues"],
        )

    def test_undecodable_root_fit_is_rejected(self) -> None:
        fixture = self._make_fixture("undecodable-fit")
        formal_path = fixture["root"] / f"{fixture['basename']}.fit"
        formal_path.write_bytes(b"not a FITS container")
        fixture["outputs"][formal_path.name]["sha256"] = _sha256(formal_path)

        report = self._verify(fixture, output_format="fit")

        self.assertFalse(report["accepted"])
        self.assertTrue(
            any(
                issue.startswith("formal_fits_decode_failed:")
                for issue in report["scientific"][0]["issues"]
            ),
            report,
        )
        self.assertIn(
            "no_stage10_pixel_bound_scientific_fits",
            report["issues"],
        )

    def test_managed_file_pixel_tampering_is_rejected_for_png_and_tiff(
        self,
    ) -> None:
        for role in ("display", "editable"):
            with self.subTest(role=role):
                fixture = self._make_fixture(f"actual-{role}-tamper")
                artifact = self._managed_artifact(fixture, role)
                artifact_path = fixture["root"] / artifact["name"]
                changed = fixture["source"].copy()
                changed[1, 3, 5] += np.float32(0.04)
                if role == "display":
                    managed_output.write_managed_display_png(
                        artifact_path,
                        changed,
                    )
                else:
                    managed_output.write_managed_edit_tiff(
                        artifact_path,
                        changed,
                        icc_profile=_test_icc_profile(),
                    )
                changed_file_sha = _sha256(artifact_path)
                fixture["outputs"][artifact["name"]]["sha256"] = (
                    changed_file_sha
                )
                artifact["sha256"] = changed_file_sha
                self._write_report(fixture)

                report = self._verify(fixture)

                self.assertFalse(report["accepted"])
                managed_item = next(
                    item for item in report["managed"] if item["role"] == role
                )
                self.assertIn(
                    "managed_artifact_decoded_pixel_sha_mismatch",
                    managed_item["issues"],
                )

    def test_managed_report_pixel_chain_tampering_is_rejected_for_png_and_tiff(
        self,
    ) -> None:
        for role in ("display", "editable"):
            with self.subTest(role=role):
                fixture = self._make_fixture(f"report-{role}-tamper")
                artifact = self._managed_artifact(fixture, role)
                artifact["pixel_chain"]["decoded_pixel_sha256"] = "0" * 64
                self._write_report(fixture)

                report = self._verify(fixture)

                self.assertFalse(report["accepted"])
                managed_item = next(
                    item for item in report["managed"] if item["role"] == role
                )
                self.assertIn(
                    "managed_artifact_decoded_pixel_sha_mismatch",
                    managed_item["issues"],
                )

    def test_unsafe_or_nonformal_basename_is_rejected(self) -> None:
        cases = (
            ("unsafe", ["../unsafe"], "formal_output_basenames_unavailable"),
            (
                "nonformal",
                ["different_formal_result"],
                "no_stage10_pixel_bound_scientific_fits",
            ),
        )
        for name, basenames, expected_issue in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture(f"basename-{name}")

                report = self._verify(
                    fixture,
                    output_format="fit",
                    formal_basenames=basenames,
                )

                self.assertFalse(report["accepted"])
                self.assertIn(expected_issue, report["issues"])
                self.assertEqual(report["formal_outputs"], [])

    def test_missing_or_duplicate_requested_managed_role_is_rejected(
        self,
    ) -> None:
        cases = (("missing", "editable"), ("duplicate", "display"))
        for case, role in cases:
            with self.subTest(case=case, role=role):
                fixture = self._make_fixture(f"role-{case}")
                artifact = self._managed_artifact(fixture, role)
                if case == "missing":
                    fixture["managed_report"]["artifacts"].remove(artifact)
                else:
                    fixture["managed_report"]["artifacts"].append(
                        copy.deepcopy(artifact)
                    )
                self._write_report(fixture)

                report = self._verify(fixture)

                self.assertFalse(report["accepted"])
                self.assertIn(
                    f"managed_output_role_cardinality_invalid:{role}",
                    report["issues"],
                )
                if case == "missing":
                    self.assertIn(
                        f"managed_output_roles_unverified:{role}",
                        report["issues"],
                    )


if __name__ == "__main__":
    unittest.main()
