#!/usr/bin/env python3
"""Compare selected Starun pipeline modules in source and native layouts."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


MODULES = (
    "stage3_contract",
    "background_sampling",
    "stage4_auto_reference",
    "local_adjustments",
    "stage9_quality",
)
WORKER_TIMEOUT_SECONDS = 120


def _array_summary(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "finite_count": int(finite.size),
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
    }


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return {"array": _array_summary(value)}
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _payload_digest(value: Any) -> str:
    canonical = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _exercise_modules(mode: str, module_root: Path, source_root: Path) -> dict[str, Any]:
    import numpy as np

    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(module_root))
    loaded = {name: importlib.import_module(name) for name in MODULES}

    module_files: dict[str, str] = {}
    for name, module in loaded.items():
        path = Path(module.__file__).resolve()
        module_files[name] = path.name
        if mode == "native":
            if path.parent != module_root.resolve() or path.suffix != ".so":
                raise RuntimeError(f"{name} did not resolve to the native overlay: {path}")
        elif path.parent != source_root.resolve() or path.suffix != ".py":
            raise RuntimeError(f"{name} did not resolve to source: {path}")

    stage3_contract = loaded["stage3_contract"]
    background_sampling = loaded["background_sampling"]
    stage4_auto_reference = loaded["stage4_auto_reference"]
    local_adjustments = loaded["local_adjustments"]
    stage9_quality = loaded["stage9_quality"]

    y, x = np.mgrid[0:48, 0:64]
    base = 0.045 + (x / 63.0) * 0.018 + (y / 47.0) * 0.009
    walking = 0.0025 * np.sin((x + 2.0 * y) / 4.5)
    image = np.stack(
        (
            base + walking + 0.004,
            base + 0.6 * walking,
            base - 0.4 * walking + 0.002,
        ),
        axis=2,
    ).astype(np.float32)
    image[12:15, 20:23, :] += 0.35
    image[31:34, 45:48, :] += 0.22
    image = np.clip(image, 0.0, 1.0)

    background_report = background_sampling.analyze_directional_pattern_noise(
        image,
        detection_threshold=0.55,
        walking_threshold=0.50,
        max_side=128,
    )
    auto_candidate, auto_report = stage4_auto_reference.evaluate_auto_local_reference(
        image,
        config=None,
        channel_kind="broadband_rgb_osc",
        linear=True,
    )
    masks = local_adjustments.build_local_masks(image)
    monotonic = local_adjustments.apply_monotonic_curve(
        np.transpose(image, (2, 0, 1)),
        ((0.0, 0.0), (0.35, 0.31), (0.70, 0.76), (1.0, 1.0)),
        np.asarray(masks["masks"]["subject"], dtype=np.float32),
        opacity=0.35,
    )
    stars = np.zeros_like(image)
    stars[12:15, 20:23, :] = np.array([0.32, 0.24, 0.18], dtype=np.float32)
    stars[31:34, 45:48, :] = np.array([0.18, 0.22, 0.30], dtype=np.float32)
    blended = stage9_quality.screen_blend(image, stars, 0.72)
    recovered = stage9_quality.unscreen_layer(blended, image)

    signatures = {
        "stage3_gate_thresholds": str(inspect.signature(stage3_contract.stage3_gate_thresholds)),
        "analyze_directional_pattern_noise": str(
            inspect.signature(background_sampling.analyze_directional_pattern_noise)
        ),
        "evaluate_auto_local_reference": str(
            inspect.signature(stage4_auto_reference.evaluate_auto_local_reference)
        ),
        "apply_monotonic_curve": str(inspect.signature(local_adjustments.apply_monotonic_curve)),
        "screen_blend": str(inspect.signature(stage9_quality.screen_blend)),
    }
    results = {
        "stage3_contract": {
            "thresholds": stage3_contract.stage3_gate_thresholds("output_first"),
            "manifest": stage3_contract.stage3_static_contract_manifest(),
        },
        "background_sampling_digest": _payload_digest(background_report),
        "stage4_auto_reference": {
            "candidate": _array_summary(auto_candidate) if auto_candidate is not None else None,
            "report_digest": _payload_digest(auto_report),
        },
        "local_adjustments": {
            "masks_digest": _payload_digest(masks),
            "monotonic": _array_summary(monotonic),
        },
        "stage9_quality": {
            "blended": _array_summary(blended),
            "recovered": _array_summary(recovered),
            "scale_radius": stage9_quality.stage9_scale_radius(
                4.0,
                None,
                fwhm_px=2.5,
                rounding="ceil",
                minimum=1,
            ),
        },
        "signatures": signatures,
    }
    return {"mode": mode, "module_files": module_files, "results": _jsonable(results)}


def _run_worker(
    python: Path,
    *,
    mode: str,
    module_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [
                str(python),
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--module-root",
                str(module_root),
                "--source-dir",
                str(source_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{mode} verification worker exceeded {WORKER_TIMEOUT_SECONDS} seconds"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} verification worker failed ({completed.returncode}):\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return json.loads(completed.stdout)


def verify(python: Path, source_root: Path, native_root: Path) -> None:
    source_payload = _run_worker(
        python,
        mode="source",
        module_root=source_root,
        source_root=source_root,
    )
    native_payload = _run_worker(
        python,
        mode="native",
        module_root=native_root,
        source_root=source_root,
    )
    source_results = json.dumps(source_payload["results"], ensure_ascii=False, indent=2, sort_keys=True)
    native_results = json.dumps(native_payload["results"], ensure_ascii=False, indent=2, sort_keys=True)
    if source_results != native_results:
        diff = "\n".join(
            difflib.unified_diff(
                source_results.splitlines(),
                native_results.splitlines(),
                fromfile="source",
                tofile="native",
                lineterm="",
            )
        )
        raise RuntimeError(f"Representative source/native fixture equivalence failed:\n{diff}")
    print("native_import_ok")
    print("representative_source_native_equivalence_ok")
    for name in MODULES:
        print(f"{name}={native_payload['module_files'][name]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--worker", choices=("source", "native"))
    parser.add_argument("--module-root", type=Path)
    parser.add_argument("--expected-modules", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_dir.expanduser().resolve()
    if args.worker:
        if args.module_root is None:
            raise SystemExit("--module-root is required with --worker")
        payload = _exercise_modules(args.worker, args.module_root.expanduser().resolve(), source_root)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.python is None or args.native_dir is None:
        raise SystemExit("--python and --native-dir are required")
    expected_modules = tuple(args.expected_modules or ())
    if expected_modules != MODULES:
        raise SystemExit(
            "Build/verifier native module list mismatch: "
            f"build={expected_modules!r}, verifier={MODULES!r}"
        )
    # Keep a venv interpreter's symlink path intact.  Resolving the Siril seed
    # python symlink would switch sys.prefix to the bare bundled interpreter
    # and hide the seed venv's NumPy installation.
    python = Path(os.path.abspath(args.python.expanduser()))
    verify(
        python,
        source_root,
        args.native_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
