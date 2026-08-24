from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import scene_support  # noqa: E402


class _Background:
    globalback = 0.01
    globalrms = 0.002

    def __init__(self, image, **_kwargs):
        self._shape = np.asarray(image).shape

    def back(self):
        return np.full(self._shape, self.globalback, dtype=np.float32)


class _Sep:
    __version__ = "test"
    OBJ_MERGED = 1
    OBJ_TRUNC = 2
    OBJ_DOVERFLOW = 4
    OBJ_SINGU = 8
    Background = _Background

    @staticmethod
    def extract(_image, _threshold, **_kwargs):
        dtype = [
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
        return np.asarray(
            [
                (30.0, 20.0, 9.0, 0.8, 1.8, 1.5, 0.1, 8, 0),
                (15.0, 10.0, 5.0, 0.6, 1.7, 1.4, 0.2, 7, 1),
                (40.0, 25.0, 8.0, 0.7, 1.8, 1.5, 0.1, 8, 2),
            ],
            dtype=dtype,
        )


def _image() -> np.ndarray:
    yy, xx = np.mgrid[:64, :80]
    image = np.stack(
        [
            0.05 + xx / 1000.0,
            0.06 + yy / 1000.0,
            0.07 + (xx + yy) / 2000.0,
        ]
    ).astype(np.float32)
    image[:, :2, :] = 0.0
    image[:, -2:, :] = 0.0
    image[:, :, :2] = 0.0
    image[:, :, -2:] = 0.0
    image[:, 32, 40] = 0.0
    image[0, 20, 30] = 1.0
    return image


class SceneSupportTests(unittest.TestCase):
    def test_valid_mask_excludes_edge_fill_but_keeps_internal_zero(self) -> None:
        valid, report = scene_support.build_valid_mask(_image())
        self.assertEqual(report["status"], "available")
        self.assertEqual(valid.dtype, np.uint8)
        self.assertEqual(int(valid[0, 0]), 0)
        self.assertEqual(int(valid[32, 40]), 1)

        constant = np.full((1, 32, 48), 0.1, dtype=np.float32)
        constant_valid, constant_report = scene_support.build_valid_mask(constant)
        self.assertTrue(np.all(constant_valid == 1))
        self.assertTrue(constant_report["constant_image_preserved"])

    def test_saturation_map_preserves_channel_bits(self) -> None:
        image = np.zeros((3, 8, 9), dtype=np.float32)
        image[0, 2, 3] = 1.0
        image[1, 2, 3] = 1.0
        image[2, 4, 5] = 1.0
        saturation, report = scene_support.build_saturation_map(image)
        self.assertEqual(report["status"], "available")
        self.assertEqual(int(saturation[2, 3]), 3)
        self.assertEqual(int(saturation[4, 5]), 4)

    def test_catalog_is_stable_and_rejects_disallowed_flags(self) -> None:
        image = _image()
        valid, _report = scene_support.build_valid_mask(image)
        saturation, _report = scene_support.build_saturation_map(image)
        catalog = scene_support.extract_sep_catalog(
            image,
            valid,
            saturation,
            sep_module=_Sep,
        )
        self.assertEqual(catalog["status"], "available")
        self.assertEqual([row["id"] for row in catalog["records"]], ["S3000001", "S3000002"])
        self.assertEqual([row["y"] for row in catalog["records"]], [10.0, 20.0])
        self.assertEqual(
            catalog["records_sha256"],
            scene_support.canonical_json_sha256(catalog["records"]),
        )
        self.assertTrue(catalog["records"][1]["saturated"])

    def test_bundle_is_deterministic_validated_and_read_only(self) -> None:
        image = _image()
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_source = Path(first_dir) / "stage3_bg_input.fit"
            second_source = Path(second_dir) / "stage3_bg_input.fit"
            first_source.write_bytes(b"scene-support-source")
            second_source.write_bytes(b"scene-support-source")
            first = scene_support.build_scene_support(
                image,
                Path(first_dir),
                source_path=first_source,
                sep_module=_Sep,
            )
            second = scene_support.build_scene_support(
                image,
                Path(second_dir),
                source_path=second_source,
                sep_module=_Sep,
            )
            self.assertEqual(first["arrays"]["file_sha256"], second["arrays"]["file_sha256"])
            self.assertEqual(
                first["components"]["star_catalog"]["records_sha256"],
                second["components"]["star_catalog"]["records_sha256"],
            )
            loaded = scene_support.load_scene_support(
                Path(first_dir), expected_shape=image.shape
            )
            self.assertEqual(loaded["status"], "available")
            self.assertFalse(loaded["valid_mask"].flags.writeable)
            with self.assertRaises(ValueError):
                loaded["valid_mask"][0, 0] = 1

            manifest_path = Path(first_dir) / scene_support.SCENE_SUPPORT_JSON
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["components"]["valid_mask"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            rejected = scene_support.load_scene_support(Path(first_dir))
            self.assertEqual(rejected["status"], "unavailable")

    def test_missing_sep_only_makes_catalog_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = scene_support.build_scene_support(
                _image(),
                Path(directory),
                sep_module=None,
            )
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["components"]["valid_mask"]["status"], "available")
            self.assertEqual(manifest["components"]["saturation_map"]["status"], "available")
            self.assertEqual(manifest["components"]["star_catalog"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
