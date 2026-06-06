from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_SYQON_GPU_KEY = "SEESTAR_SYQON_GPU"
ENV_SYQON_TIMEOUT_KEY = "SEESTAR_SYQON_TIMEOUT_SEC"


def _clamp_int(value: object, min_value: int, max_value: int) -> int:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        ivalue = min_value
    return max(min_value, min(max_value, ivalue))


def syqon_starless_cli_options(
    pipeline: object,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    axiom: bool = False,
) -> Tuple[Tuple[str, ...], int, str]:
    use_gpu = True
    gpu_raw = os.getenv(ENV_SYQON_GPU_KEY)
    if gpu_raw is not None:
        lowered = gpu_raw.strip().lower()
        if lowered in ENV_TRUE_VALUES:
            use_gpu = True
        elif lowered in ENV_FALSE_VALUES:
            use_gpu = False
        else:
            pipeline.log.warn(
                f"{ENV_SYQON_GPU_KEY} has invalid value; defaulting to GPU enabled"
            )

    accel_available = False
    accel_name = ""
    if use_gpu:
        try:
            torch_mod = importlib.import_module("torch")
            if torch_mod.cuda.is_available():
                accel_available = True
                accel_name = "CUDA"
            elif (
                hasattr(torch_mod, "backends")
                and hasattr(torch_mod.backends, "mps")
                and torch_mod.backends.mps.is_available()
            ):
                accel_available = True
                accel_name = "MPS"
            elif hasattr(torch_mod, "xpu") and torch_mod.xpu.is_available():
                accel_available = True
                accel_name = "XPU"
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            pipeline.log.warn(
                "SyQon GPU backend probe failed; script will decide device: "
                f"{pipeline._short_text(e, 120)}"
            )

    timeout_sec = 900 if (not use_gpu or accel_available) else 180
    timeout_raw = os.getenv(ENV_SYQON_TIMEOUT_KEY)
    if timeout_raw is not None:
        try:
            timeout_sec = int(float(timeout_raw.strip()))
        except (TypeError, ValueError):
            pipeline.log.warn(
                f"{ENV_SYQON_TIMEOUT_KEY} has invalid value; using {timeout_sec}s"
            )
    timeout_sec = max(60, min(1800, timeout_sec))

    tile_size = _clamp_int(tile_size, 512, 1024)
    overlap = _clamp_int(overlap, 64, 128)
    if overlap >= tile_size:
        overlap = max(64, min(128, tile_size // 4))

    args: List[str] = ["--tile-size", str(tile_size), "--overlap", str(overlap)]
    if not use_gpu:
        args.append("--no_gpu")
    if axiom:
        args.append("--axiom")
    if not use_gpu:
        device_note = "CPU forced"
    elif accel_available:
        device_note = f"{accel_name} enabled"
    else:
        device_note = "GPU requested but no torch backend detected; CPU fallback expected"
    if axiom:
        device_note += "; Axiom v2"
    device_note += f"; tile={tile_size}; overlap={overlap}"
    return tuple(args), timeout_sec, device_note


def collect_star_separation_outputs(
    pipeline: object,
) -> Tuple[Optional[Path], Optional[Path]]:
    if not pipeline.process_dir:
        return None, None

    starless_candidates: List[Path] = []
    starmask_candidates: List[Path] = []
    if pipeline.stretched_name:
        starless_candidates.extend(
            [
                pipeline.process_dir / f"starless_{pipeline.stretched_name}.fit",
                pipeline.process_dir / f"starless_{pipeline.stretched_name}.fits",
            ]
        )
        starmask_candidates.extend(
            [
                pipeline.process_dir / f"starmask_{pipeline.stretched_name}.fit",
                pipeline.process_dir / f"starmask_{pipeline.stretched_name}.fits",
                pipeline.process_dir / f"{pipeline.stretched_name}_starmask.fit",
                pipeline.process_dir / f"{pipeline.stretched_name}_starmask.fits",
            ]
        )
    starless_candidates.extend(
        [
            pipeline.process_dir / "starless.fit",
            pipeline.process_dir / "starless.fits",
        ]
    )
    starmask_candidates.extend(
        [
            pipeline.process_dir / "starmask.fit",
            pipeline.process_dir / "starmask.fits",
        ]
    )

    def _first_existing(paths: List[Path]) -> Optional[Path]:
        for path in paths:
            if path.is_file():
                return path
        return None

    starless_src = _first_existing(starless_candidates)
    starmask_src = _first_existing(starmask_candidates)

    if starless_src is None:
        for ext in ("fit", "fits"):
            fallback = sorted(pipeline.process_dir.glob(f"starless_*.{ext}"))
            if fallback:
                starless_src = fallback[0]
                break
    if starmask_src is None:
        for ext in ("fit", "fits"):
            fallback = sorted(pipeline.process_dir.glob(f"starmask_*.{ext}"))
            if fallback:
                starmask_src = fallback[0]
                break

    return starless_src, starmask_src


def clear_star_separation_outputs(pipeline: object) -> None:
    if not pipeline.process_dir:
        return
    patterns = (
        "starless.fit",
        "starless.fits",
        "starmask.fit",
        "starmask.fits",
        "starless_*.fit",
        "starless_*.fits",
        "starmask_*.fit",
        "starmask_*.fits",
        "*_starmask.fit",
        "*_starmask.fits",
        "*_stars.fit",
        "*_stars.fits",
    )
    seen: set[Path] = set()
    for pattern in patterns:
        for path in pipeline.process_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            if path.stem.startswith(("starless_ai_best_", "starmask_ai_best_")):
                continue
            seen.add(path)
            pipeline._safe_unlink(path)


def syqon_axiom_model_available(pipeline: object) -> bool:
    candidates: List[Path] = []
    if pipeline.siril_plugin_dir:
        candidates.extend(
            [
                pipeline.siril_plugin_dir / "syqon_starless" / "Siril_axiomv2.pt",
                pipeline.siril_plugin_dir
                / "vendor"
                / "syqon_starless"
                / "Siril_axiomv2.pt",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return True
    return False


def stage7_try_syqon_variant(
    pipeline: object,
    syqon_script: Path,
    *,
    attempt_name: str,
    tile_size: int,
    overlap: int,
    axiom: bool,
) -> Optional[str]:
    pipeline._clear_star_separation_outputs()
    pipeline.starless_file = None
    pipeline.starmask_file = None
    pipeline.cmd_with_check("load", pipeline.stretched_name)
    syqon_args, syqon_timeout, syqon_device_note = pipeline._syqon_starless_cli_options(
        tile_size=tile_size,
        overlap=overlap,
        axiom=axiom,
    )
    used = pipeline._run_plugin_script_cli_subprocess(
        "去星",
        f"SyQon Starless {attempt_name} ({syqon_device_note})",
        syqon_script,
        args=syqon_args,
        timeout_sec=syqon_timeout,
    )
    if not used:
        return None
    starless_src, starmask_src = pipeline._collect_star_separation_outputs()
    if starless_src is None:
        pipeline._last_plugin_script_error = "SyQon 脚本未生成 starless 产物"
        return None

    target_starless = pipeline.process_dir / "starless.fit"
    if starless_src != target_starless:
        shutil.copy2(starless_src, target_starless)
    pipeline.starless_file = target_starless
    if starmask_src is not None:
        target_starmask = pipeline.process_dir / "starmask.fit"
        if starmask_src != target_starmask:
            shutil.copy2(starmask_src, target_starmask)
        pipeline.starmask_file = target_starmask
    pipeline._stage7_prepare_starmask()
    return used


def stage7_try_syqon_with_source(
    pipeline: object,
    syqon_script: Path,
    *,
    source_stem: str,
    attempt_name: str,
    tile_size: int,
    overlap: int,
    axiom: bool,
) -> Optional[str]:
    previous_stretched_name = pipeline.stretched_name
    try:
        pipeline.stretched_name = source_stem
        return pipeline._stage7_try_syqon_variant(
            syqon_script,
            attempt_name=attempt_name,
            tile_size=tile_size,
            overlap=overlap,
            axiom=axiom,
        )
    finally:
        pipeline.stretched_name = previous_stretched_name
