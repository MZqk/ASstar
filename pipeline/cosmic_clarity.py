"""CosmicClarity classic and native fallback helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_COSMIC_NATIVE_GPU_KEY = "SEESTAR_COSMIC_NATIVE_GPU"
ENV_COSMIC_CLASSIC_GPU_KEY = "SEESTAR_COSMIC_CLASSIC_GPU"
ENV_COSMIC_CLASSIC_ENABLE_KEY = "SEESTAR_COSMIC_CLASSIC_ENABLE"


def classic_cosmic_clarity_args(
    pipeline, config_name: str, label: str
) -> Optional[Tuple[str, ...]]:
    enabled_raw = os.getenv(ENV_COSMIC_CLASSIC_ENABLE_KEY, "").strip().lower()
    if enabled_raw not in ENV_TRUE_VALUES:
        if enabled_raw and enabled_raw not in ENV_FALSE_VALUES:
            pipeline.log.warn(
                f"{ENV_COSMIC_CLASSIC_ENABLE_KEY} has invalid value; "
                "defaulting to Native CosmicClarity"
            )
        pipeline.log.info(
            f"{label} classic 路径默认跳过；使用 Native CosmicClarity "
            f"（如需调试 classic，设置 {ENV_COSMIC_CLASSIC_ENABLE_KEY}=1）"
        )
        return None

    env_path = os.getenv("SEESTAR_COSMIC_CLARITY_EXECUTABLE", "").strip()
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    config_path = (
        Path.home()
        / "Library"
        / "Application Support"
        / "org.siril.Siril"
        / "siril"
        / config_name
    )
    if config_path.is_file():
        try:
            configured = config_path.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, UnicodeError, IndexError):
            configured = ""
        if configured:
            candidates.append(Path(configured).expanduser())

    candidates.extend(pipeline._classic_cosmic_clarity_auto_candidates())

    for candidate in candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        validation_error = pipeline._classic_cosmic_clarity_candidate_error(candidate)
        if validation_error:
            pipeline.log.warn(
                f"{label} classic executable 无效，跳过: {candidate} ({validation_error})"
            )
            continue
        pipeline._persist_classic_cosmic_clarity_config(config_path, candidate, label)
        return ("-executable", str(candidate))

    if env_path:
        pipeline.log.warn(
            f"{label} classic executable 无效，跳过 classic 脚本: {env_path}"
        )
    else:
        pipeline.log.info(
            f"{label} classic executable 未配置，classic 脚本不运行；将使用 Native/SCUNet 链路"
        )
    return None


def classic_cosmic_clarity_candidate_error(pipeline, candidate: Path) -> Optional[str]:
    """Return a reason when a known bundled wrapper is not safe to execute."""
    if candidate.name != "CosmicClarity" or not pipeline.siril_plugin_dir:
        return None
    try:
        candidate.resolve().relative_to((pipeline.siril_plugin_dir / "bin").resolve())
    except (OSError, ValueError):
        return None

    try:
        header = candidate.read_text(encoding="utf-8", errors="replace")[:512]
    except OSError as e:
        return f"cannot read wrapper: {e}"
    if '"exec" "${SIRIL_PYTHON_CLI:-python3}" "$0" "$@"' in header:
        return "legacy wrapper does not guard boolean SIRIL_PYTHON_CLI"
    if "SEESTAR_SIRIL_PYTHON_CLI" not in header:
        return "wrapper missing stable python env fallback"
    return None


def classic_cosmic_clarity_auto_candidates(pipeline) -> List[Path]:
    names = (
        "CosmicClarity",
        "Cosmic Clarity",
        "cosmicclarity",
        "cosmic_clarity",
    )
    roots: List[Path] = []
    roots.extend(Path("/Applications").glob("*Cosmic*Clarity*.app/Contents/MacOS"))
    roots.extend((Path.home() / "Applications").glob("*Cosmic*Clarity*.app/Contents/MacOS"))
    if pipeline.siril_plugin_dir:
        roots.append(pipeline.siril_plugin_dir / "cosmic_clarity")
        roots.append(pipeline.siril_plugin_dir / "bin")
    roots.extend(
        [
            Path("/usr/local/bin"),
            Path("/opt/homebrew/bin"),
        ]
    )

    candidates: List[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for name in names:
            candidates.append(root / name)
    return candidates


def persist_classic_cosmic_clarity_config(
    pipeline,
    config_path: Path,
    executable: Path,
    label: str,
) -> None:
    try:
        current = ""
        if config_path.is_file():
            current = config_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if current == str(executable):
            return
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(str(executable) + "\n", encoding="utf-8")
        pipeline.log.info(
            f"{label} classic executable 已自动配置: {executable}"
        )
    except (OSError, UnicodeError, IndexError) as e:
        pipeline.log.warn(
            f"{label} classic executable 配置写入失败，仍继续使用本次发现路径: {e}"
        )


def classic_cosmic_clarity_device_args(pipeline) -> Tuple[Tuple[str, ...], str]:
    use_gpu = True
    gpu_raw = os.getenv(ENV_COSMIC_CLASSIC_GPU_KEY)
    if gpu_raw is not None:
        lowered = gpu_raw.strip().lower()
        if lowered in ENV_TRUE_VALUES:
            use_gpu = True
        elif lowered in ENV_FALSE_VALUES:
            use_gpu = False
        else:
            pipeline.log.warn(
                f"{ENV_COSMIC_CLASSIC_GPU_KEY} has invalid value; defaulting to GPU/device auto"
            )

    if use_gpu:
        return tuple(), "GPU/device auto"
    return ("-no_gpu",), "CPU forced"


def cosmic_clarity_native_sharpen_cli_options(pipeline) -> Tuple[Tuple[str, ...], str]:
    native_args, device_note = pipeline._cosmic_clarity_native_denoise_cli_options()
    cpu_forced = "--cpu" in native_args
    args: List[str] = [
        "--mode",
        "sharpen",
        "--sharpening-mode",
        "Both",
        "--stellar-amount",
        "0.35",
        "--nonstellar-amount",
        "0.35",
        "--nonstellar-psf",
        "3.0",
    ]
    if cpu_forced:
        args.append("--cpu")
    return tuple(args), device_note


def run_cosmic_clarity_native_sharpen_fallback(pipeline, step_key: str) -> Optional[str]:
    native_script = pipeline._find_plugin_script(("processing/CosmicClarity_Native.py",))
    if native_script is None:
        pipeline._last_plugin_script_error = "CosmicClarity_Native.py script missing"
        return None
    native_args, device_note = pipeline._cosmic_clarity_native_sharpen_cli_options()
    return pipeline._run_plugin_script_cli_subprocess(
        step_key,
        f"CosmicClarity Native Sharpen ({device_note})",
        native_script,
        args=native_args,
        timeout_sec=pipeline._final_denoise_cli_timeout_sec(),
        verify_image_change=True,
    )


def final_denoise_cli_timeout_sec(pipeline) -> int:
    raw_timeout = str(os.getenv("SEESTAR_SIRILPY_TIMEOUT_SEC", "120")).strip()
    try:
        sirilpy_timeout = int(float(raw_timeout))
    except (TypeError, ValueError):
        sirilpy_timeout = 120
    return max(60, min(300, sirilpy_timeout + 60))


def cosmic_clarity_native_denoise_cli_options(pipeline) -> Tuple[Tuple[str, ...], str]:
    use_gpu = True
    gpu_raw = os.getenv(ENV_COSMIC_NATIVE_GPU_KEY)
    if gpu_raw is not None:
        lowered = gpu_raw.strip().lower()
        if lowered in ENV_TRUE_VALUES:
            use_gpu = True
        elif lowered in ENV_FALSE_VALUES:
            use_gpu = False
        else:
            pipeline.log.warn(
                f"{ENV_COSMIC_NATIVE_GPU_KEY} has invalid value; defaulting to GPU/device auto"
            )

    mode = str(
        getattr(pipeline, "_cosmic_clarity_native_denoise_mode_override", "luminance")
        or "luminance"
    ).strip().lower()
    if mode not in {"full", "luminance", "separate"}:
        pipeline.log.warn(
            f"Invalid CosmicClarity Native denoise mode '{mode}', defaulting to luminance"
        )
        mode = "luminance"
    strength = str(
        getattr(pipeline, "_cosmic_clarity_native_denoise_strength_override", "0.5")
        or "0.5"
    )

    args: List[str] = [
        "--mode",
        "denoise",
        "--denoise-mode",
        mode,
        "--denoise-strength",
        strength,
    ]
    if not use_gpu:
        args.append("--cpu")
        return tuple(args), "CPU forced"
    return tuple(args), "GPU/device auto"


def run_cosmic_clarity_native_denoise_fallback(pipeline, step_key: str) -> Optional[str]:
    native_script = pipeline._find_plugin_script(("processing/CosmicClarity_Native.py",))
    if native_script is None:
        pipeline._last_plugin_script_error = "CosmicClarity_Native.py script missing"
        return None
    native_args, device_note = pipeline._cosmic_clarity_native_denoise_cli_options()
    return pipeline._run_plugin_script_cli_subprocess(
        step_key,
        f"CosmicClarity Native Denoise ({device_note})",
        native_script,
        args=native_args,
        timeout_sec=pipeline._final_denoise_cli_timeout_sec(),
        verify_image_change=True,
    )
