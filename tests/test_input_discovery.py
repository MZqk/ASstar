#!/usr/bin/env python3
"""Regression tests for explicit file, recursive Light, and legacy intake."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from input_discovery import (  # noqa: E402
    DiscoveryTrust,
    InputKind,
    discover_input,
    discover_light_groups,
)
import run_manifest  # noqa: E402


def _fits_card(keyword: str, value: object) -> str:
    if isinstance(value, str):
        rendered = f"'{value}'"
    elif isinstance(value, bool):
        rendered = "T" if value else "F"
    else:
        rendered = str(value)
    return f"{keyword:<8}= {rendered:<20}".ljust(80)[:80]


def _write_fits(path: Path, **metadata: object) -> None:
    cards = [_fits_card("SIMPLE", True)]
    cards.extend(_fits_card(key, value) for key, value in metadata.items())
    cards.append("END".ljust(80))
    header = "".join(cards).encode("ascii")
    header += b" " * ((2880 - len(header) % 2880) % 2880)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header)


class InputDiscoveryTests(unittest.TestCase):
    def test_explicit_xisf_is_master_and_never_guessed_as_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "NGC6910.xisf"
            source.write_bytes(b"XISF0100-test")

            result = discover_input(source)

        self.assertEqual(result.kind, InputKind.MASTER_FILE)
        self.assertEqual(result.trust, DiscoveryTrust.RECOGNIZED)
        self.assertIsNone(result.resume_after_stage)
        self.assertIn("Stage 1", result.summary)

    def test_rendered_image_is_review_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "preview.png"
            source.write_bytes(b"png")

            result = discover_input(source)

        self.assertEqual(result.kind, InputKind.REVIEW_FILE)
        self.assertEqual(result.trust, DiscoveryTrust.REVIEW_REQUIRED)
        self.assertTrue(result.accepted)

    def test_recursive_lights_group_by_target_filter_camera_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Capture"
            common = {
                "OBJECT": "NGC 6910",
                "INSTRUME": "Seestar S50",
                "NAXIS1": 1080,
                "NAXIS2": 1920,
                "XBINNING": 1,
                "YBINNING": 1,
            }
            _write_fits(
                root / "NGC6910" / "Light" / "night1" / "Light_001.fit",
                **common,
                FILTER="Seestar LP",
            )
            _write_fits(
                root / "NGC6910" / "Light" / "night2" / "Light_002.fits",
                **common,
                FILTER="Seestar LP",
            )
            _write_fits(
                root / "NGC6910" / "Lights" / "Light_003.fit",
                **common,
                FILTER="No filter",
            )
            _write_fits(
                root / "NGC6910" / "Dark" / "Dark_001.fit",
                **common,
                FILTER="Seestar LP",
            )
            _write_fits(
                root / "NGC6910" / "Flat" / "Light_misnamed.fit",
                **common,
                FILTER="Seestar LP",
            )

            groups = discover_light_groups(root)
            result = discover_input(root)

        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(group.files) for group in groups), [1, 2])
        self.assertEqual({group.target for group in groups}, {"NGC 6910"})
        self.assertEqual(
            {group.filter_name for group in groups},
            {"No filter", "Seestar LP"},
        )
        self.assertEqual({group.camera for group in groups}, {"Seestar S50"})
        self.assertEqual({group.geometry for group in groups}, {"1080x1920@1x1"})
        self.assertEqual(result.kind, InputKind.LIGHT_DIRECTORY)
        self.assertEqual(len(result.light_groups), 2)
        self.assertIn("2 个独立叠加任务", result.summary)

    def test_directory_with_master_requires_explicit_file_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "NGC6910_master.fit", OBJECT="NGC 6910")

            result = discover_input(root)

        self.assertEqual(result.kind, InputKind.UNSUPPORTED)
        self.assertFalse(result.accepted)
        self.assertIn("直接拖入", result.errors[0])

    def test_legacy_result_name_does_not_select_stage5(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fits(root / "result_linear.fit", OBJECT="NGC 6910")

            result = discover_input(root)

        self.assertEqual(result.kind, InputKind.LEGACY_DIRECTORY)
        self.assertIsNone(result.resume_after_stage)
        self.assertIn("不会根据", result.warnings[0])

    def test_signed_legacy_checkpoint_is_migratable_but_still_starts_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "result_linear.fit"
            checkpoint.write_bytes(b"trusted-legacy-linear")
            plan = {
                "schema": "seestar.processing-plan.v1",
                "run_id": "legacy-run",
            }
            plan["plan_hash"] = run_manifest.canonical_payload_hash(plan)
            run_manifest.atomic_write_json(root / "processing-plan.json", plan)
            result_manifest = {
                "schema": "seestar.pipeline-result.v1",
                "status": "success",
                "plan_hash": plan["plan_hash"],
                "checkpoints": {
                    "result_linear": {
                        **run_manifest.file_record(checkpoint, base_dir=root),
                        "state": "linear",
                    }
                },
            }
            result_manifest["manifest_hash"] = (
                run_manifest.canonical_payload_hash(result_manifest)
            )
            run_manifest.atomic_write_json(
                root / "pipeline-result.json",
                result_manifest,
            )

            discovery = discover_input(root)

        self.assertEqual(discovery.kind, InputKind.LEGACY_DIRECTORY)
        self.assertEqual(discovery.trust, DiscoveryTrust.VERIFIED)
        self.assertTrue(discovery.accepted)
        self.assertIsNone(discovery.resume_after_stage)
        self.assertEqual(discovery.master_file, checkpoint.resolve())
        self.assertIn("Stage 1", discovery.summary)


if __name__ == "__main__":
    unittest.main()
