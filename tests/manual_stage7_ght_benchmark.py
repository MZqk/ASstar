#!/usr/bin/env python3
"""Run one isolated Siril GHT candidate for Stage 7 real-image benchmarking.

This tool is intentionally outside the production candidate builder. It never
chains Asinh, AutoGHS, inverse GHT, black-point shifting, or a second GHT.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from image_metrics import measure_quality_metrics  # noqa: E402
from stage7_pixel_domain import canonicalize_stage7_pixels_01  # noqa: E402
import stage7_stretch_metrics  # noqa: E402


ALLOWED_TARGET_TYPES = frozenset(
    {
        "bright_emission_reflection_nebula",
        "large_galaxy",
        "small_galaxy",
    }
)
OUTPUT_FITS_NAME = "stage7_ght_candidate.fit"
OUTPUT_REPORT_NAME = "stage7_ght_benchmark_report.json"


def _finite(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def validate_ght_parameters(
    *,
    D: Any,
    B: Any,
    LP: Any,
    SP: Any,
    HP: Any,
) -> Dict[str, float]:
    values = {
        "D": _finite(D, "D"),
        "B": _finite(B, "B"),
        "LP": _finite(LP, "LP"),
        "SP": _finite(SP, "SP"),
        "HP": _finite(HP, "HP"),
    }
    if not 0.0 < values["D"] <= 10.0:
        raise ValueError("D must satisfy 0 < D <= 10")
    if not -5.0 <= values["B"] <= 15.0:
        raise ValueError("B must satisfy -5 <= B <= 15")
    if not 0.0 <= values["LP"] <= values["SP"] <= values["HP"] <= 1.0:
        raise ValueError("LP/SP/HP must satisfy 0 <= LP <= SP <= HP <= 1")
    return values


def build_ght_command(parameters: Dict[str, float]) -> list[str]:
    return [
        "ght",
        *(
            f"-{name}={parameters[name]:.12g}"
            for name in ("D", "B", "LP", "SP", "HP")
        ),
        "-even",
        "-clipmode=rgbblend",
    ]


def _quote_siril_path(path: Path) -> str:
    return '"' + str(path).replace('"', '\\"') + '"'


def build_siril_script(
    source: Path,
    output_dir: Path,
    parameters: Dict[str, float],
) -> list[str]:
    command = build_ght_command(parameters)
    output_stem = output_dir / Path(OUTPUT_FITS_NAME).stem
    return [
        "requires 1.4.0",
        f"cd {_quote_siril_path(output_dir)}",
        f"load {_quote_siril_path(source)}",
        " ".join(command),
        f"save {_quote_siril_path(output_stem)}",
        "close",
    ]


def validate_inputs(
    *,
    source: Path,
    cand_a: Path,
    target_type: str,
    siril_cli: Path,
    output_dir: Path,
) -> None:
    if source.name != "stage6_starless.fit":
        raise ValueError("source must be the immutable stage6_starless.fit")
    if cand_a.name != "stage7_cand_a.fit":
        raise ValueError("cand-a must be the current stage7_cand_a.fit")
    if not source.is_file() or not cand_a.is_file():
        raise ValueError("source and cand-a FITS files must exist")
    if source.resolve() == cand_a.resolve():
        raise ValueError("source and cand-a must be different files")
    if target_type not in ALLOWED_TARGET_TYPES:
        raise ValueError(
            "target-type must be bright_emission_reflection_nebula, "
            "large_galaxy, or small_galaxy"
        )
    if not siril_cli.is_file() or not os.access(siril_cli, os.X_OK):
        raise ValueError(f"siril-cli is not executable: {siril_cli}")
    for protected in (source.resolve(), cand_a.resolve()):
        if (output_dir / OUTPUT_FITS_NAME).resolve() == protected:
            raise ValueError("benchmark output must not overwrite either input")


def _read_fits(path: Path) -> np.ndarray:
    try:
        from astropy.io import fits
    except ImportError as error:
        raise RuntimeError("astropy is required to read benchmark FITS files") from error
    data = np.asarray(fits.getdata(path, memmap=False))
    data = np.squeeze(data)
    pixels, _domain = canonicalize_stage7_pixels_01(data)
    return pixels


def _frozen_background_mask(source: np.ndarray) -> np.ndarray:
    rgb = stage7_stretch_metrics._stage7_rgb_float_fullres(source)
    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return luma <= float(np.quantile(luma, 0.30))


def _candidate_report(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    name: str,
    method: str,
    parameters: Dict[str, float],
    target_type: str,
    background_mask: np.ndarray,
) -> Dict[str, Any]:
    return {
        "name": name,
        "quality_metrics": asdict(measure_quality_metrics(candidate)),
        "transform_loss": stage7_stretch_metrics.assess_transform_loss(
            source,
            candidate,
            method=method,
            params=parameters,
            background_mask=background_mask,
        ),
        "color_vector_reference": (
            stage7_stretch_metrics.assess_rec709_vector_color_reference(
                source,
                candidate,
            )
        ),
        "target_local_quality": stage7_stretch_metrics.assess_target_local_stretch(
            source,
            candidate,
            target_type,
            SimpleNamespace(stage7_target_local_metrics_enabled=True),
        ),
    }


def _quality_deltas(
    current: Dict[str, Any],
    ght: Dict[str, Any],
) -> Dict[str, float]:
    current_metrics = current.get("quality_metrics") or {}
    ght_metrics = ght.get("quality_metrics") or {}
    deltas: Dict[str, float] = {}
    for key in sorted(set(current_metrics) & set(ght_metrics)):
        left = current_metrics.get(key)
        right = ght_metrics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[key] = float(right) - float(left)
    return deltas


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    source_path = Path(args.source).expanduser().resolve()
    cand_a_path = Path(args.cand_a).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    siril_cli = Path(args.siril_cli).expanduser().resolve()
    parameters = validate_ght_parameters(
        D=args.D,
        B=args.B,
        LP=args.LP,
        SP=args.SP,
        HP=args.HP,
    )
    validate_inputs(
        source=source_path,
        cand_a=cand_a_path,
        target_type=args.target_type,
        siril_cli=siril_cli,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FITS_NAME
    report_path = output_dir / OUTPUT_REPORT_NAME
    script_path = output_dir / ".stage7_ght_benchmark.ssf"
    output_path.unlink(missing_ok=True)
    script_lines = build_siril_script(source_path, output_dir, parameters)
    script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    command = [str(siril_cli), "-d", str(output_dir), "-s", str(script_path)]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=300,
            env=os.environ.copy(),
        )
    finally:
        script_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        tail = " | ".join((completed.stdout or "").splitlines()[-12:])
        raise RuntimeError(
            f"siril-cli GHT benchmark failed with exit={completed.returncode}: {tail}"
        )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError("siril-cli completed without stage7_ght_candidate.fit")

    source = _read_fits(source_path)
    current_pixels = _read_fits(cand_a_path)
    ght_pixels = _read_fits(output_path)
    background_mask = _frozen_background_mask(source)
    current = _candidate_report(
        source,
        current_pixels,
        name="current_cand_a",
        method="current_cand_a_unknown_transform",
        parameters={},
        target_type=args.target_type,
        background_mask=background_mask,
    )
    ght = _candidate_report(
        source,
        ght_pixels,
        name="stage7_ght_candidate",
        method="ght",
        parameters=parameters,
        target_type=args.target_type,
        background_mask=background_mask,
    )
    ght_semantics = {
        "schema": stage7_stretch_metrics.STRETCH_SEMANTICS_SCHEMA,
        "status": "available",
        "engine": "siril",
        "method": "ght",
        "minimum_siril_version": (
            stage7_stretch_metrics.SIRIL_MINIMUM_VERSION_CONTRACT
        ),
        "bundled_reference_version": (
            stage7_stretch_metrics.SIRIL_BUNDLED_REFERENCE_VERSION
        ),
        "clip_mode": "rgbblend",
        "human_weighted": False,
        "steps": [
            {
                "command": "ght",
                "argv": build_ght_command(parameters)[1:],
                "full_argv": build_ght_command(parameters),
                "even_channels": True,
                "clip_mode": "rgbblend",
            }
        ],
        "forbidden_chain_steps": [
            "asinh",
            "autoghs",
            "inverse",
            "blackpoint_shift",
            "second_ght",
        ],
    }
    ght["transform_semantics"] = ght_semantics
    report: Dict[str, Any] = {
        "schema": "starun.stage7-ght-benchmark.v1",
        "status": "completed",
        "production_candidate": False,
        "promotion_status": "requires_real_full_chain",
        "target_type": args.target_type,
        "inputs": {
            "immutable_source": str(source_path),
            "current_cand_a": str(cand_a_path),
        },
        "output": str(output_path),
        "parameters": parameters,
        "execution": {
            "runner": "independent_siril_cli",
            "cli": str(siril_cli),
            "siril_script": script_lines,
            "ght_command": build_ght_command(parameters),
            "ght_command_count": 1,
            "source_reloaded_between_transforms": False,
            "stdout_tail": (completed.stdout or "").splitlines()[-12:],
        },
        "background_roi": {
            "method": "source_fixed_rec709_bottom_quantile",
            "quantile": 0.30,
            "coverage": float(np.mean(background_mask)),
        },
        "candidates": {
            "current_cand_a": current,
            "ght": ght,
        },
        "comparison": {
            "quality_metric_delta_ght_minus_cand_a": _quality_deltas(current, ght),
            "selection_performed": False,
            "claim_of_superiority": False,
        },
    }
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="immutable stage6_starless.fit")
    parser.add_argument("--cand-a", required=True, help="current stage7_cand_a.fit")
    parser.add_argument("--target-type", required=True, choices=sorted(ALLOWED_TARGET_TYPES))
    parser.add_argument("--D", required=True, type=float)
    parser.add_argument("--B", required=True, type=float)
    parser.add_argument("--LP", required=True, type=float)
    parser.add_argument("--SP", required=True, type=float)
    parser.add_argument("--HP", required=True, type=float)
    parser.add_argument("--siril-cli", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
