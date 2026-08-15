#!/usr/bin/env python3
"""Regression tests for the stdlib-only SyQon offline model helpers."""

from __future__ import annotations

import ast
import io
import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[1]
SYQON_SCRIPT = (
    REPO_ROOT
    / "resources"
    / "siril_plugins"
    / "vendor"
    / "siril-scripts"
    / "SyQon"
    / "Starless.py"
)
SYQON_PATCHER = (
    REPO_ROOT
    / "resources"
    / "siril_plugins"
    / "patches"
    / "apply_syqon_offline_model_patch.py"
)


def _load_patcher():
    spec = importlib.util.spec_from_file_location(
        "syqon_offline_model_patch_test",
        SYQON_PATCHER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load patcher: {SYQON_PATCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_offline_helpers() -> dict[str, object]:
    tree = ast.parse(SYQON_SCRIPT.read_text(encoding="utf-8"))
    helper_names = {
        "syqon_network_downloads_allowed",
        "resolve_zenith_model_dir",
        "download_file",
        "verify_shasum",
        "validate_local_zenith_model",
    }
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    if {node.name for node in helper_nodes} != helper_names:
        raise AssertionError("SyQon offline model helpers are incomplete")
    namespace: dict[str, object] = {
        "os": os,
        "sys": sys,
        "urllib": __import__("urllib"),
        "hashlib": hashlib,
        "Path": Path,
        "__file__": str(SYQON_SCRIPT),
        "ENV_SYQON_MODEL_DIR_KEY": "STARUN_SYQON_MODEL_DIR",
        "ENV_NETWORK_MODE_KEY": "STARUN_NETWORK_MODE",
        "ENV_TRUE_VALUES": frozenset({"1", "true", "yes", "on"}),
    }
    exec(
        compile(
            ast.Module(body=helper_nodes, type_ignores=[]),
            str(SYQON_SCRIPT),
            "exec",
        ),
        namespace,
    )
    return namespace


def _load_pixel_exchange_helpers() -> dict[str, object]:
    tree = ast.parse(SYQON_SCRIPT.read_text(encoding="utf-8"))
    helper_names = {
        "_load_fits_file_input",
        "prepare_image_for_inference",
        "restore_image_dtype",
        "_make_safe_header",
        "_write_fits_file_output",
    }
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    if {node.name for node in helper_nodes} != helper_names:
        raise AssertionError("SyQon pixel exchange helpers are incomplete")
    namespace: dict[str, object] = {
        "os": os,
        "np": np,
        "fits": fits,
        "Path": Path,
        "Tuple": tuple,
    }
    exec(
        compile(
            ast.Module(body=helper_nodes, type_ignores=[]),
            str(SYQON_SCRIPT),
            "exec",
        ),
        namespace,
    )
    return namespace


def _load_file_mode_runner(namespace: dict[str, object]) -> None:
    tree = ast.parse(SYQON_SCRIPT.read_text(encoding="utf-8"))
    runner = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_file_mode"
        ),
        None,
    )
    if runner is None:
        raise AssertionError("SyQon pure file mode runner is missing")
    exec(
        compile(
            ast.Module(body=[runner], type_ignores=[]),
            str(SYQON_SCRIPT),
            "exec",
        ),
        namespace,
    )


class SyqonOfflineModelTests(unittest.TestCase):
    def test_inference_engine_forwards_every_file_mode_parameter(self):
        tree = ast.parse(SYQON_SCRIPT.read_text(encoding="utf-8"))
        engine = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "InferenceEngine"
        )
        process_async = next(
            node
            for node in engine.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_async"
        )
        process_call = next(
            node
            for node in ast.walk(process_async)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "process_image"
        )
        forwarded = {keyword.arg for keyword in process_call.keywords}
        self.assertTrue(
            {
                "tile",
                "overlap",
                "generate_mask",
                "mask_method",
                "use_amp",
                "use_gpu",
                "stretch_method",
                "mtf_target",
                "ihs_target",
                "linked_stretch",
                "stat_bp_sigma",
                "no_black_clip",
            }.issubset(forwarded)
        )

    def test_runtime_patcher_reapplies_fix_after_upstream_refresh(self):
        patcher = _load_patcher()
        patched_source = SYQON_SCRIPT.read_text(encoding="utf-8")
        upstream_source = patched_source
        for old, new, _label in reversed(patcher.REPLACEMENTS):
            self.assertEqual(upstream_source.count(new), 1)
            upstream_source = upstream_source.replace(new, old, 1)
        self.assertNotIn(patcher.PATCH_SENTINEL, upstream_source)
        self.assertEqual(
            hashlib.sha256(upstream_source.encode("utf-8")).hexdigest(),
            patcher.UPSTREAM_STARLESS_SHA256,
        )

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "Starless.py"
            target.write_text(upstream_source, encoding="utf-8")

            self.assertTrue(patcher.apply_patch(target))
            self.assertEqual(target.read_text(encoding="utf-8"), patched_source)
            self.assertFalse(patcher.apply_patch(target))

    def test_explicit_local_bundle_is_resolved_and_verified_offline(self):
        helpers = _load_offline_helpers()
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            model_bytes = b"local-zenith"
            (model_dir / "zenith.pt").write_bytes(model_bytes)
            (model_dir / "zenith.pt.sha256").write_text(
                hashlib.sha256(model_bytes).hexdigest() + "  zenith.pt\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "STARUN_SYQON_MODEL_DIR": str(model_dir),
                    "STARUN_NETWORK_MODE": "0",
                },
                clear=False,
            ):
                resolved, source = helpers["resolve_zenith_model_dir"](
                    model_dir / "runtime"
                )
                valid, message = helpers["validate_local_zenith_model"](resolved)
                network_allowed = helpers["syqon_network_downloads_allowed"]()

        self.assertEqual(resolved, model_dir.resolve())
        self.assertEqual(source, "STARUN_SYQON_MODEL_DIR")
        self.assertTrue(valid, message)
        self.assertFalse(network_allowed)

    def test_local_bundle_checksum_mismatch_is_rejected(self):
        helpers = _load_offline_helpers()
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "zenith.pt").write_bytes(b"corrupted")
            (model_dir / "zenith.pt.sha256").write_text(
                "0" * 64 + "  zenith.pt\n",
                encoding="utf-8",
            )

            valid, message = helpers["validate_local_zenith_model"](model_dir)

        self.assertFalse(valid)
        self.assertIn("SHA256 verification failed", message)

    def test_offline_download_guard_never_opens_url(self):
        helpers = _load_offline_helpers()
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(
                os.environ,
                {"STARUN_NETWORK_MODE": "0"},
                clear=False,
            ),
            patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("network access attempted"),
            ) as urlopen,
        ):
            downloaded = helpers["download_file"](
                "https://siril.syqon.it/zenith.pt",
                Path(td) / "zenith.pt",
                "Zenith model",
                silent=True,
            )

        self.assertFalse(downloaded)
        urlopen.assert_not_called()

    def test_uint16_input_is_returned_as_canonical_float32_without_requantizing(self):
        helpers = _load_pixel_exchange_helpers()
        source = np.array([0, 32768, 65535], dtype=np.uint16)

        prepared, original_dtype, scale = helpers[
            "prepare_image_for_inference"
        ](source)
        model_output = np.clip(prepared * 0.8, 0.0, 1.0)
        restored = helpers["restore_image_dtype"](
            model_output,
            original_dtype,
            scale,
        )

        self.assertEqual(restored.dtype, np.float32)
        np.testing.assert_allclose(restored, model_output, rtol=0.0, atol=1e-7)
        self.assertGreater(float(restored[1]), 0.39)
        self.assertLess(float(restored[1]), 0.41)

    def test_canonical_header_removes_source_integer_scaling_cards(self):
        helpers = _load_pixel_exchange_helpers()
        source_header = fits.Header()
        source_header["BITPIX"] = 16
        source_header["BSCALE"] = 2.0
        source_header["BZERO"] = 32768.0
        source_header["BLANK"] = -32768
        source_header["DATAMIN"] = 0
        source_header["DATAMAX"] = 65535
        source_header["BG-PTS"] = "legacy"
        source_header["OBJECT"] = "M17"

        safe_text = helpers["_make_safe_header"](
            source_header.tostring(sep="\n"),
            "starless",
        )
        safe_header = fits.Header.fromstring(safe_text, sep="\n")

        for keyword in (
            "BITPIX",
            "BSCALE",
            "BZERO",
            "BLANK",
            "DATAMIN",
            "DATAMAX",
            "BG-PTS",
        ):
            self.assertNotIn(keyword, safe_header)
        self.assertEqual(safe_header["FILTER"], "starless")
        self.assertEqual(safe_header["OBJECT"], "M17")

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "starless.fit"
            pixels = np.array([[0.023, 0.278]], dtype=np.float32)
            fits.PrimaryHDU(data=pixels, header=safe_header).writeto(output_path)
            with fits.open(output_path, do_not_scale_image_data=False) as hdul:
                roundtrip = np.asarray(hdul[0].data)
                output_header = hdul[0].header

        self.assertEqual(roundtrip.dtype, np.dtype(">f4"))
        np.testing.assert_allclose(roundtrip, pixels, rtol=0.0, atol=1e-7)
        self.assertEqual(output_header["BITPIX"], -32)
        self.assertNotIn("BSCALE", output_header)
        self.assertNotIn("BZERO", output_header)

    def test_pure_file_helpers_roundtrip_without_mutating_input(self):
        helpers = _load_pixel_exchange_helpers()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "linear.fit"
            starless_path = root / "starless.fit"
            starmask_path = root / "starmask.fit"
            source_header = fits.Header()
            source_header["OBJECT"] = "M17"
            source_header["BG-PTS"] = "legacy"
            source_pixels = np.array(
                [[0, 32768, 65535], [1024, 16384, 49152]],
                dtype=np.uint16,
            )
            fits.PrimaryHDU(
                data=source_pixels,
                header=source_header,
            ).writeto(input_path)
            input_before = input_path.read_bytes()

            (
                resolved_input,
                prepared,
                _original_dtype,
                _scale_factor,
                header_text,
            ) = helpers["_load_fits_file_input"](input_path)
            helpers["_write_fits_file_output"](
                starless_path,
                prepared * np.float32(0.8),
                header_text,
                "starless",
            )
            helpers["_write_fits_file_output"](
                starmask_path,
                prepared * np.float32(0.2),
                header_text,
                "starmask",
            )

            self.assertEqual(resolved_input, input_path.resolve())
            self.assertEqual(input_path.read_bytes(), input_before)
            for output_path, filter_name in (
                (starless_path, "starless"),
                (starmask_path, "starmask"),
            ):
                with fits.open(output_path, do_not_scale_image_data=False) as hdul:
                    output_pixels = np.asarray(hdul[0].data)
                    output_header = hdul[0].header
                self.assertEqual(output_pixels.dtype, np.dtype(">f4"))
                self.assertTrue(np.isfinite(output_pixels).all())
                self.assertEqual(output_header["BITPIX"], -32)
                self.assertEqual(output_header["FILTER"], filter_name)
                self.assertEqual(output_header["OBJECT"], "M17")
                self.assertNotIn("BSCALE", output_header)
                self.assertNotIn("BZERO", output_header)
                self.assertNotIn("BG-PTS", output_header)

    def test_file_mode_inference_error_returns_nonzero(self):
        class FailingInferenceEngine:
            def __init__(self, **_kwargs):
                pass

            def process_async(self, *, error_callback, **_kwargs):
                error_callback("injected inference failure")

            @staticmethod
            def is_processing():
                return False

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            namespace: dict[str, object] = {
                "Path": Path,
                "sys": sys,
                "ZENITH_MODEL_PATH": root / "zenith.pt",
                "resolve_zenith_model_dir": lambda _runtime: (root, "test bundle"),
                "validate_local_zenith_model": lambda _model_dir: (True, "verified"),
                "_starun_roundtrip_shadow": lambda *_args, **_kwargs: {
                    "status": "shadow"
                },
                "_write_file_mode_manifest": lambda *_args, **_kwargs: None,
                "STARUN_LAST_INFERENCE_DIAGNOSTICS": {},
                "_load_fits_file_input": lambda input_path: (
                    input_path,
                    np.zeros((2, 2), dtype=np.float32),
                    np.dtype("float32"),
                    1.0,
                    "",
                ),
                "InferenceEngine": FailingInferenceEngine,
            }
            _load_file_mode_runner(namespace)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = namespace["_run_file_mode"](
                    SimpleNamespace(
                        input_file=str(root / "input.fit"),
                        starless_output=str(root / "starless.fit"),
                        starmask_output=str(root / "starmask.fit"),
                        no_gpu=True,
                        tile_size=512,
                        overlap=64,
                        stretch_method="statistical",
                        target_median=0.15,
                        linked_stretch=False,
                        stat_bp_sigma=5.0,
                        no_black_clip=False,
                        mask_method="subtraction",
                        use_amp=False,
                        manifest_output=str(root / "manifest.json"),
                    )
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("injected inference failure", stderr.getvalue())

    def test_main_dispatches_file_mode_before_constructing_siril_interface(self):
        source = SYQON_SCRIPT.read_text(encoding="utf-8")
        self.assertLess(
            source.index("return _run_file_mode(args)"),
            source.index("siril = s.SirilInterface()"),
        )
        self.assertIn("raise SystemExit(main())", source)


if __name__ == "__main__":
    unittest.main()
