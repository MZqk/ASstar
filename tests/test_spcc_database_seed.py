#!/usr/bin/env python3
"""Fixed Siril SPCC database seed integrity and runtime sync tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gui.common import (
    siril_spcc_database_root_from_home,
    sync_siril_spcc_database_seed,
    verify_siril_spcc_database_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_ROOT = REPO_ROOT / "resources" / "siril_spcc_database"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SirilSpccDatabaseSeedTests(unittest.TestCase):
    def test_bundled_manifest_pins_required_entries_and_checksums(self):
        manifest = json.loads((SEED_ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["schema"],
            "seestar.siril-spcc-database-seed.v1",
        )
        self.assertEqual(
            manifest["source"]["commit"],
            "3426f0939d53d4d3a1b4c8e620a6faf8212bda32",
        )
        self.assertEqual(manifest["license"]["spdx"], "GPL-3.0-only")
        labels = {entry["label"] for entry in manifest["files"]}
        self.assertTrue(
            {
                "Sony IMX585",
                "ZWO Seestar LP",
                "No filter",
                "Average Spiral Galaxy",
                "Star, type G2(v)",
                "GNU General Public License v3",
            }.issubset(labels)
        )
        for entry in manifest["files"]:
            source = SEED_ROOT / entry["path"]
            self.assertTrue(source.is_file(), source)
            self.assertEqual(_sha256(source), entry["sha256"], source)

    def test_sync_is_idempotent_repairs_managed_files_and_preserves_extras(self):
        with tempfile.TemporaryDirectory() as td:
            runtime_home = Path(td) / "runtime_home"
            target_root = siril_spcc_database_root_from_home(runtime_home)

            first = sync_siril_spcc_database_seed(SEED_ROOT, runtime_home)

            self.assertEqual(first["target_root"], target_root)
            self.assertEqual(first["managed_files"], 8)
            self.assertEqual(len(first["copied_files"]), 8)
            ready, detail = verify_siril_spcc_database_seed(SEED_ROOT, runtime_home)
            self.assertTrue(ready, detail)

            managed = target_root / "osc_sensors" / "Sony_IMX585.json"
            managed.write_text("corrupted", encoding="utf-8")
            extra = target_root / "osc_sensors" / "User_Custom.json"
            extra.write_text("[]\n", encoding="utf-8")

            repaired = sync_siril_spcc_database_seed(SEED_ROOT, runtime_home)

            self.assertEqual(
                repaired["copied_files"],
                ["osc_sensors/Sony_IMX585.json"],
            )
            self.assertEqual(
                _sha256(managed),
                _sha256(SEED_ROOT / "osc_sensors" / "Sony_IMX585.json"),
            )
            self.assertTrue(extra.is_file())
            unchanged = sync_siril_spcc_database_seed(SEED_ROOT, runtime_home)
            self.assertEqual(unchanged["copied_files"], [])

    def test_verify_fails_before_runtime_seed_is_copied(self):
        with tempfile.TemporaryDirectory() as td:
            ready, detail = verify_siril_spcc_database_seed(
                SEED_ROOT,
                Path(td) / "runtime_home",
            )

        self.assertFalse(ready)
        self.assertIn("runtime SPCC 数据文件缺失", detail)


if __name__ == "__main__":
    unittest.main()
