#!/usr/bin/env python3
"""Geometry and identity-fusion tests for the patched Zenith tiler."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import amp
except ImportError:  # pragma: no cover - the packaged runtime includes torch
    torch = None
    nn = None
    F = None
    amp = None


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


def _load_tiling_helpers() -> dict[str, object]:
    tree = ast.parse(SYQON_SCRIPT.read_text(encoding="utf-8"))
    helper_names = {
        "_weight_1d",
        "_edge_key",
        "_build_weight_cache",
        "_tile_positions",
        "_starun_pad_zenith_tensor",
        "tile_inference_torch",
    }
    nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in helper_names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "STARUN_LAST_INFERENCE_DIAGNOSTICS"
            for target in node.targets
        ):
            nodes.append(node)
    if {
        node.name for node in nodes if isinstance(node, ast.FunctionDef)
    } != helper_names:
        raise AssertionError("patched Zenith tiling helpers are incomplete")
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "amp": amp,
        "np": np,
        "Callable": Callable,
        "Optional": Optional,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SYQON_SCRIPT), "exec"),
        namespace,
    )
    return namespace


@unittest.skipIf(torch is None, "torch is unavailable")
class SyqonTilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helpers = _load_tiling_helpers()
        cls.device = torch.device("cpu")
        cls.model = nn.Identity().to(cls.device)

    def _assert_identity(self, height: int, width: int) -> dict[str, object]:
        count = height * width
        source = torch.linspace(
            0.0,
            1.0,
            steps=max(count, 2),
            dtype=torch.float32,
            device=self.device,
        )[:count].reshape(1, 1, height, width)
        output = self.helpers["tile_inference_torch"](
            self.model,
            source,
            512,
            64,
            self.device,
            False,
        )
        self.assertEqual(tuple(output.shape), tuple(source.shape))
        self.assertTrue(bool(torch.isfinite(output).all()))
        max_abs = float(torch.max(torch.abs(output - source)).cpu())
        self.assertLessEqual(max_abs, 2e-6)
        diagnostics = dict(
            self.helpers["STARUN_LAST_INFERENCE_DIAGNOSTICS"]
        )
        self.assertGreater(float(diagnostics["coverage_min"]), 0.0)
        self.assertEqual(diagnostics["crop_shape"], [height, width])
        self.assertEqual(diagnostics["original_shape"], [height, width])
        padded_height, padded_width = diagnostics["padded_shape"]
        self.assertEqual(int(padded_height) % 16, 0)
        self.assertEqual(int(padded_width) % 16, 0)
        return diagnostics

    def test_square_boundary_sizes_preserve_identity_and_expected_grid(self) -> None:
        expected = {
            511: ("full_frame", 1),
            512: ("full_frame", 1),
            513: ("tiled", 4),
            960: ("tiled", 4),
            961: ("tiled", 9),
        }
        for size, (mode, tile_count) in expected.items():
            with self.subTest(size=size):
                diagnostics = self._assert_identity(size, size)
                self.assertEqual(diagnostics["mode"], mode)
                self.assertEqual(diagnostics["grid"]["tiles"], tile_count)

    def test_rectangular_and_non_aligned_sizes_have_no_coverage_holes(self) -> None:
        for height, width in (
            (517, 773),
            (400, 8000),
            (8000, 400),
            (529, 777),
        ):
            with self.subTest(shape=(height, width)):
                diagnostics = self._assert_identity(height, width)
                self.assertEqual(diagnostics["mode"], "tiled")
                self.assertGreater(diagnostics["grid"]["tiles"], 1)

    def test_single_pixel_axis_uses_edge_padding_and_restores_shape(self) -> None:
        for height, width in ((1, 513), (513, 1)):
            with self.subTest(shape=(height, width)):
                diagnostics = self._assert_identity(height, width)
                self.assertEqual(diagnostics["padding_mode"], "replicate")


if __name__ == "__main__":
    unittest.main()
