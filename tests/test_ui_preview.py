#!/usr/bin/env python3
"""Regression tests for raw Stage 0 and per-stage preview generation."""

from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from astropy.io import fits

from pipeline.ui_preview import (
    _linked_display_stretch,
    _rgb_raw_float,
    build_linked_display_curve_contract,
    write_display_preview,
    write_raw_fits_preview,
    write_raw_preview,
)


def _read_png16_rgb(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("not a PNG file")
    offset = 8
    width = height = 0
    idat = bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        tag = payload[offset + 4:offset + 8]
        chunk = payload[offset + 8:offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, *_rest = struct.unpack(
                ">IIBBBBB", chunk
            )
            if (bit_depth, color_type) != (16, 2):
                raise AssertionError("expected 16-bit RGB PNG")
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"IEND":
            break
    rows = zlib.decompress(bytes(idat))
    stride = 1 + width * 3 * 2
    decoded = np.empty((height, width, 3), dtype=np.uint16)
    for row_index in range(height):
        row = rows[row_index * stride:(row_index + 1) * stride]
        if row[0] != 0:
            raise AssertionError("preview writer should use PNG filter 0")
        decoded[row_index] = np.frombuffer(row[1:], dtype=">u2").reshape(width, 3)
    return decoded


class RawPreviewTests(unittest.TestCase):
    def test_linked_display_curve_contract_preserves_legacy_pixels_exactly(self):
        rng = np.random.default_rng(73)
        source = np.clip(
            rng.lognormal(-7.0, 1.0, size=(3, 777, 999)),
            0.0,
            1.0,
        ).astype(np.float32)

        luminance = (
            source[0] * np.float32(0.2126)
            + source[1] * np.float32(0.7152)
            + source[2] * np.float32(0.0722)
        )
        sample = luminance.ravel()
        if sample.size > 500_000:
            sample = sample[::max(1, sample.size // 500_000)]
        sample = sample[np.isfinite(sample)]
        black, median, white = np.percentile(sample, (0.2, 50.0, 99.8))
        span = float(white - black)
        normalized = np.clip((source - float(black)) / span, 0.0, 1.0)
        normalized_luma = np.clip(
            (luminance - float(black)) / span,
            0.0,
            1.0,
        )
        median_normalized = float(
            np.clip((median - black) / span, 1e-6, 0.999999)
        )
        gamma = float(
            np.clip(np.log(0.18) / np.log(median_normalized), 0.20, 1.00)
        )
        stretched_luma = np.power(normalized_luma, gamma).astype(
            np.float32,
            copy=False,
        )
        gain = np.divide(
            stretched_luma,
            normalized_luma,
            out=np.zeros_like(stretched_luma),
            where=normalized_luma > 1e-7,
        )
        legacy = np.clip(
            normalized * gain[np.newaxis, :, :],
            0.0,
            1.0,
        )

        rebuilt = _linked_display_stretch(source)
        contract = build_linked_display_curve_contract(source)

        np.testing.assert_array_equal(rebuilt, legacy)
        self.assertEqual(contract["status"], "ok")
        self.assertEqual(contract["sampling"]["preview_max_side"], 1600)
        self.assertEqual(contract["percentiles"]["black"], 0.2)
        self.assertEqual(contract["percentiles"]["white"], 99.8)

    def test_float_preview_preserves_raw_levels_without_normalization(self):
        source = np.array(
            [[0.10, 0.20], [0.40, 0.80]],
            dtype=np.float32,
        )

        converted = _rgb_raw_float(source, max_side=1600)

        np.testing.assert_allclose(converted[0], np.flip(source, axis=0))
        self.assertAlmostEqual(float(converted.min()), 0.10, places=6)
        self.assertAlmostEqual(float(converted.max()), 0.80, places=6)

    def test_png_is_atomic_16_bit_rgb_and_keeps_native_uint16_levels(self):
        source = np.array(
            [[0, 32768], [65535, 16384]],
            dtype=np.uint16,
        )
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "latest.png"

            write_raw_preview(source, target)

            decoded = _read_png16_rgb(target)
            self.assertEqual(decoded.shape, (2, 2, 3))
            np.testing.assert_array_equal(
                decoded[:, :, 0],
                np.flip(source, axis=0),
            )
            self.assertFalse(target.with_name("latest.png.tmp").exists())

    def test_stage0_uses_first_readable_fits_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            invalid = root / "Light_000.fit"
            valid = root / "Light_001.fit"
            invalid.write_bytes(b"not fits")
            fits.writeto(
                valid,
                np.array([[0, 1024], [2048, 4096]], dtype=np.uint16),
            )
            target = root / "stage0.png"

            selected = write_raw_fits_preview([invalid, valid], target)

            self.assertEqual(selected, valid)
            self.assertTrue(target.is_file())
            self.assertEqual(_read_png16_rgb(target).shape, (2, 2, 3))

    def test_display_stretch_lifts_linear_preview_without_mutating_input(self):
        source = np.linspace(0.001, 0.05, 64 * 64, dtype=np.float32).reshape(64, 64)
        original = source.copy()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "display.png"

            write_display_preview(source, target, apply_stretch=True)

            decoded = _read_png16_rgb(target).astype(np.float32) / 65535.0
        np.testing.assert_array_equal(source, original)
        self.assertGreater(float(np.median(decoded[:, :, 0])), 0.12)
        self.assertFalse(target.with_name("display.png.tmp").exists())


if __name__ == "__main__":
    unittest.main()
