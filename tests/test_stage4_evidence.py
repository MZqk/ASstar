#!/usr/bin/env python3
"""Tests for observer-only Stage 4 solver and Header evidence."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import stage4_evidence as evidence  # noqa: E402
from gui import runtime_capabilities  # noqa: E402


def _card(key: str, value=None) -> str:
    if key == "END":
        return "END".ljust(80)
    if isinstance(value, str):
        rendered = "'" + value.replace("'", "''") + "'"
    elif value is True:
        rendered = "T"
    elif value is False:
        rendered = "F"
    else:
        rendered = str(value)
    return f"{key:<8}= {rendered:>20}".ljust(80)[:80]


def _write_fits(path: Path, extra_cards=()) -> None:
    cards = [
        _card("SIMPLE", True),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", 32),
        _card("NAXIS2", 24),
        *extra_cards,
        _card("END"),
    ]
    header = "".join(cards).encode("ascii")
    header += b" " * ((2880 - len(header) % 2880) % 2880)
    path.write_bytes(header)


class Stage4EvidenceTests(unittest.TestCase):
    def test_fits_snapshot_preserves_duplicate_filter_cards_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.fts"
            _write_fits(
                path,
                (
                    _card("FILTER", "Ha/OIII"),
                    _card("FILTER", "IRCUT"),
                    _card("RA", "05 35 17.3"),
                    _card("DEC", "-05 23 28"),
                ),
            )

            first = evidence.read_fits_header_snapshot(
                path,
                role="stage4_input",
                relation="test",
            )
            second = evidence.read_fits_header_snapshot(
                path,
                role="stage4_input",
                relation="test",
            )

        self.assertEqual(first["status"], "available")
        filters = [card for card in first["cards"] if card["key"] == "FILTER"]
        self.assertEqual([card["occurrence"] for card in filters], [1, 2])
        self.assertEqual(filters[0]["value"], "Ha/OIII")
        self.assertEqual(first["dimensions"], {"naxis1": 32, "naxis2": 24})
        self.assertEqual(first["cards_sha256"], second["cards_sha256"])
        self.assertEqual(first["header_sha256"], second["header_sha256"])
        self.assertEqual(first["conflicts"][0]["key"], "FILTER")

    def test_xisf_identity_uses_converted_stage1_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "master.xisf"
            source.write_bytes(b"xisf-test")
            _write_fits(root / "stage1_prepared.fit", (_card("FILTER", "LP"),))
            _write_fits(root / "stage3_bgremoved.fit", (_card("FILTER", "LP"),))
            source_sha = evidence.file_sha256(source)
            manifest = {
                "schema": "starun.task-run.v1",
                "manifest_hash": "a" * 64,
                "source": {
                    "kind": "master_file",
                    "read_only": True,
                    "fingerprint": "b" * 64,
                    "file_count": 1,
                    "files": [
                        {
                            "path": str(source),
                            "size": source.stat().st_size,
                            "sha256": source_sha,
                        }
                    ],
                },
            }
            report = evidence.build_filter_header_evidence(
                process_dir=root,
                source_file=source,
                task_run_manifest=manifest,
                stage4_metadata={"FILTER": "LP"},
                filter_selection={"selected_filter_headers": []},
                channel_mapping={
                    "schema": "starun.narrowband-channel-mapping.v1",
                    "mapping": "osc_hoo_rgb",
                    "confidence": 0.86,
                    "evidence": "authoritative_filter_field_hint",
                },
                explicit_filter_hint="",
                device_geometry_report={"schema": "starun.device-geometry.v1"},
                header_guided_enabled=True,
            )

        original = report["snapshots"]["original_source"]
        converted = report["snapshots"]["stage1_prepared"]
        self.assertEqual(original["status"], "not_applicable")
        self.assertEqual(original["sha256"], source_sha)
        self.assertEqual(original["reason"], "original_xisf_header_not_parsed")
        self.assertEqual(converted["status"], "available")
        self.assertEqual(
            converted["relation"],
            "siril_converted_fits_observation",
        )

    def test_solver_inventory_never_promotes_unintegrated_backends(self):
        runtime_manifest = {
            "schema": "starun.runtime-capabilities.v1",
            "status": "ready",
            "capabilities": {
                "siril": {
                    "available": True,
                    "selected_path": "/App/Contents/Resources/siril-cli",
                    "launch_probe": {"version": "1.4.1"},
                    "candidates": [
                        {
                            "path": "/App/Contents/Resources/siril-cli",
                            "within_resources_root": True,
                        }
                    ],
                }
            },
        }
        report = evidence.build_solver_capabilities(
            runtime_decision={
                "status": "ready",
                "source": "runtime_capabilities_manifest",
                "commands": {"platesolve": True},
            },
            runtime_manifest=runtime_manifest,
            configured=True,
            catalogs=(
                {
                    "id": "gaia",
                    "kind": "online_catalog",
                    "order": 0,
                    "available": True,
                },
            ),
            processing_mode="auto",
        )
        report = evidence.finalize_solver_capabilities(
            report,
            attempts=(
                {
                    "label": "catalog:gaia",
                    "command": "platesolve -catalog=gaia",
                    "status": "ok",
                },
            ),
            platesolve_attempted=True,
            platesolve_ok=True,
            skip_reason="",
        )

        self.assertEqual(
            [backend["id"] for backend in report["backends"]],
            list(evidence.SOLVER_BACKEND_IDS),
        )
        self.assertEqual(report["selection"]["backend"], "siril_platesolve")
        for backend in report["backends"][1:]:
            self.assertEqual(backend["runtime_status"], "not_probed")
            self.assertFalse(backend["eligible"])
            self.assertFalse(backend["attempted"])

    def test_candidate_wcs_is_retained_when_final_file_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_fits(root / "stage1_prepared.fit")
            _write_fits(root / "stage3_bgremoved.fit", (_card("FILTER", "IRCUT"),))
            report = evidence.build_filter_header_evidence(
                process_dir=root,
                source_file=None,
                task_run_manifest=None,
                stage4_metadata={"FILTER": "IRCUT"},
                filter_selection={"selected_filter_headers": []},
                channel_mapping={},
                explicit_filter_hint="",
                device_geometry_report={"schema": "starun.device-geometry.v1"},
                header_guided_enabled=True,
            )
            ps_path = root / "stage4_psolved.fit"
            _write_fits(
                ps_path,
                (
                    _card("FILTER", "IRCUT"),
                    _card("CRVAL1", 83.82),
                    _card("CRVAL2", -5.39),
                    _card("SECPIX", 1.8),
                ),
            )
            evidence.capture_solver_candidate(
                report,
                ps_path,
                platesolve_ok=True,
                output_saved=True,
            )
            _write_fits(ps_path, (_card("FILTER", "IRCUT"),))
            report = evidence.finalize_filter_header_evidence(
                report,
                final_path=ps_path,
                final_output_saved=True,
                processing_mode="auto",
                platesolve_attempted=True,
                platesolve_ok=False,
                device_geometry_report={"schema": "starun.device-geometry.v1"},
            )

        candidate_added = {
            item["key"]
            for item in report["header_diff"]["solver_candidate"]["added"]
        }
        final_added = report["header_diff"]["final_psolved"]["added"]
        self.assertIn("SECPIX", candidate_added)
        self.assertEqual(final_added, [])
        self.assertEqual(
            report["snapshots"]["final_psolved"]["relation"],
            "solver_output_validation_rejected",
        )
        self.assertTrue(
            report["snapshots"]["final_psolved"]["rollback_observed"]
        )

    def test_artifact_file_hash_matches_persisted_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = evidence.build_solver_capabilities(
                runtime_decision={
                    "status": "ready",
                    "commands": {"platesolve": True},
                },
                runtime_manifest=None,
                configured=True,
                catalogs=(),
                processing_mode="preserve",
            )
            payload = evidence.finalize_solver_capabilities(
                payload,
                attempts=(),
                platesolve_attempted=False,
                platesolve_ok=False,
                skip_reason="user_preserve",
            )
            reference = evidence.write_evidence_artifact(
                root,
                evidence.SOLVER_CAPABILITIES_NAME,
                payload,
            )
            artifact = root / evidence.SOLVER_CAPABILITIES_NAME
            loaded = json.loads(artifact.read_text(encoding="utf-8"))
            persisted_sha256 = evidence.file_sha256(artifact)

        self.assertEqual(loaded, payload)
        self.assertEqual(reference["sha256"], persisted_sha256)

    def test_schema_validation_rejects_tampered_attempt_digest(self):
        payload = evidence.build_solver_capabilities(
            runtime_decision={
                "status": "ready",
                "commands": {"platesolve": True},
            },
            runtime_manifest=None,
            configured=True,
            catalogs=(),
            processing_mode="preserve",
        )
        payload = evidence.finalize_solver_capabilities(
            payload,
            attempts=(),
            platesolve_attempted=False,
            platesolve_ok=False,
            skip_reason="user_preserve",
        )
        payload["attempts_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            evidence.validate_solver_capabilities(payload)

    def test_gui_runtime_inventory_has_fixed_order_without_external_probes(self):
        siril = {
            "available": True,
            "selected_path": "/bundle/siril-cli",
            "candidates": [
                {
                    "path": "/bundle/siril-cli",
                    "within_resources_root": True,
                }
            ],
        }
        backends = runtime_capabilities._stage4_plate_solver_backends(siril)
        self.assertEqual(
            [backend["id"] for backend in backends],
            list(evidence.SOLVER_BACKEND_IDS),
        )
        self.assertTrue(backends[0]["eligible"])
        self.assertTrue(
            all(backend["runtime_status"] == "not_probed" for backend in backends[1:])
        )
        report = evidence.build_solver_capabilities(
            runtime_decision={
                "status": "ready",
                "commands": {"platesolve": True},
            },
            runtime_manifest={
                "schema": "starun.runtime-capabilities.v1",
                "status": "ready",
                "capabilities": {
                    "siril": siril,
                    "stage4_plate_solver_backends": backends,
                },
            },
            configured=True,
            catalogs=(),
            processing_mode="auto",
        )
        self.assertEqual(
            report["backend_inventory_source"],
            "runtime_capabilities_manifest",
        )


if __name__ == "__main__":
    unittest.main()
