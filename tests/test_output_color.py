from __future__ import annotations

import binascii
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from output_color import (  # noqa: E402
    build_output_color_manifest,
    inspect_output_artifact,
)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(tag)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", checksum)


def _write_png(path: Path, *, srgb: bool) -> None:
    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)),
    ]
    if srgb:
        chunks.append(_png_chunk(b"sRGB", b"\x00"))
    chunks.extend(
        (
            _png_chunk(b"IDAT", zlib.compress(b"\x00\x10\x20\x30")),
            _png_chunk(b"IEND", b""),
        )
    )
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def _write_tiff_with_icc(path: Path) -> None:
    bits_offset = 38
    icc_offset = 44
    header = b"II" + struct.pack("<HI", 42, 8)
    entries = (
        struct.pack("<H", 2)
        + struct.pack("<HHI", 258, 3, 3)
        + struct.pack("<I", bits_offset)
        + struct.pack("<HHI", 34675, 1, 12)
        + struct.pack("<I", icc_offset)
        + struct.pack("<I", 0)
    )
    payload = struct.pack("<HHH", 16, 16, 16) + b"ICC_PROFILE!"
    path.write_bytes(header + entries + payload)


def _write_fits(path: Path) -> None:
    cards = [
        "SIMPLE  =                    T".ljust(80),
        "BITPIX  =                  -32".ljust(80),
        "NAXIS   =                    3".ljust(80),
        "END".ljust(80),
    ]
    header = "".join(cards).encode("ascii")
    path.write_bytes(header.ljust(2880, b" "))


class OutputColorAuditTests(unittest.TestCase):
    def test_png_profile_detection_distinguishes_tagged_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tagged = root / "tagged.png"
            untagged = root / "untagged.png"
            _write_png(tagged, srgb=True)
            _write_png(untagged, srgb=False)

            self.assertTrue(
                inspect_output_artifact(tagged)["display_profile_verified"]
            )
            self.assertFalse(
                inspect_output_artifact(untagged)["display_profile_verified"]
            )

    def test_manifest_audits_scientific_editable_and_display_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_png(root / "result_processed.png", srgb=True)
            _write_tiff_with_icc(root / "result_processed.tif")
            _write_fits(root / "result_final.fit")

            manifest = build_output_color_manifest(
                work_dir=root,
                base_filename="result_processed",
                fit_filename="result_final",
                fallback_base="result_processed",
                fallback_fit_base="result_final",
                output_format="all",
                channel_semantics="broadband_rgb_osc",
                review_only=False,
            )

        self.assertEqual(manifest["mode"], "report_only")
        self.assertFalse(manifest["rewrote_outputs"])
        self.assertEqual(manifest["summary"]["artifact_count"], 3)
        self.assertTrue(
            manifest["summary"]["ready_for_future_managed_export"]
        )
        self.assertEqual(
            manifest["desired_contract"]["fits"]["color_transform"],
            "none",
        )


if __name__ == "__main__":
    unittest.main()
