#!/usr/bin/env python3
"""Tests for the on-demand, runtime-only Gaia catalogue installer."""

from __future__ import annotations

import bz2
import hashlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from gui import gaia_catalog


class _Response(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class GaiaCatalogTests(unittest.TestCase):
    def test_catalogue_path_is_runtime_home_not_project(self):
        runtime_home = Path("/tmp/seestar-runtime-home")
        path = gaia_catalog.gaia_catalog_path(runtime_home)
        self.assertEqual(
            path,
            runtime_home
            / ".local/share/siril"
            / gaia_catalog.GAIA_CATALOG_FILENAME,
        )

    def test_download_verifies_and_atomically_installs_catalogue(self):
        uncompressed = (b"Gaia-DR3-Teff\n" * 256) + b"complete"
        archive = bz2.compress(uncompressed)
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            gaia_catalog,
            "GAIA_CATALOG_SIZE_BYTES",
            len(uncompressed),
        ), patch.object(
            gaia_catalog,
            "GAIA_CATALOG_ARCHIVE_SHA256",
            hashlib.sha256(archive).hexdigest(),
        ), patch.object(
            gaia_catalog,
            "GAIA_CATALOG_SHA256",
            hashlib.sha256(uncompressed).hexdigest(),
        ), patch.object(
            gaia_catalog,
            "GAIA_CATALOG_DOWNLOAD_HEADROOM_BYTES",
            0,
        ):
            destination = gaia_catalog.download_gaia_catalog(
                Path(temp_dir),
                stop_event=threading.Event(),
                progress=messages.append,
                opener=lambda *_args, **_kwargs: _Response(archive),
            )

            self.assertEqual(destination.read_bytes(), uncompressed)
            self.assertTrue(
                gaia_catalog.gaia_catalog_status(Path(temp_dir))["available"]
            )
            self.assertFalse(
                destination.with_name(
                    f".{gaia_catalog.GAIA_CATALOG_ARCHIVE_FILENAME}.part"
                ).exists()
            )
            self.assertTrue(any("安装完成" in message for message in messages))

    def test_cancel_never_publishes_partial_catalogue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_event = threading.Event()
            stop_event.set()
            with self.assertRaises(gaia_catalog.GaiaCatalogCancelled):
                gaia_catalog.download_gaia_catalog(
                    Path(temp_dir),
                    stop_event=stop_event,
                    progress=lambda _message: None,
                    opener=lambda *_args, **_kwargs: _Response(b""),
                )
            self.assertFalse(
                gaia_catalog.gaia_catalog_path(Path(temp_dir)).exists()
            )


if __name__ == "__main__":
    unittest.main()
