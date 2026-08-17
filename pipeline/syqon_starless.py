from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from stage7_pixel_domain import canonicalize_stage7_pixels_01

try:
    from sirilpy.exceptions import CommandError, SirilError
except ImportError:
    CommandError = RuntimeError
    SirilError = RuntimeError


ENV_SYQON_TIMEOUT_KEY = "STARUN_SYQON_TIMEOUT_SEC"
ENV_SYQON_MODEL_DIR_KEY = "STARUN_SYQON_MODEL_DIR"
SYQON_MODEL_BUNDLE_REL = Path("syqon_starless")
SYQON_EXCHANGE_SCHEMA = "starun.syqon-pixel-exchange.v2"
SYQON_SELECTED_SCHEMA = "starun.syqon-selected-pair.v1"
SYQON_ATTEMPT_SCHEMA = "starun.syqon-attempt.v1"
STAGE6_PAIR_HANDOFF_SCHEMA = "starun.stage6-pair-handoff.v1"
SYQON_MEDIAN_RATIO_MIN = 0.25
SYQON_MEDIAN_RATIO_MAX = 4.0
SYQON_MEDIAN_SIGNAL_FLOOR = 1e-5
SYQON_TILE_VALUES = (512, 1024)
SYQON_OVERLAP_VALUES = (64, 96, 128)
SYQON_STRETCH_METHODS = frozenset({"statistical", "mtf", "ihs", "none"})
SYQON_MASK_METHODS = frozenset({"subtraction", "descreen"})
SYQON_ZENITH_SHA256 = (
    "1246d011455271133f7b09ba39408313204daa8bacabca127e6af11b608d2cd2"
)
SYQON_UPSTREAM_COMMIT = "4cc9e204f9ddfd6d03cc4283aac76c82d4d19167"
SYQON_UPSTREAM_STARLESS_SHA256 = (
    "d36818f24a6927b245ab66fc7c00eaaa3b330a47406b61f0e9beb0764e06ab11"
)


@dataclass(frozen=True)
class SyQonAttemptProfile:
    """Code-owned Zenith file-mode parameters for one reproducible attempt."""

    profile_id: str
    tile_size: int = 512
    overlap: int = 64
    use_gpu: bool = True
    use_amp: bool = False
    stretch_method: str = "statistical"
    target_median: float = 0.15
    linked_stretch: bool = False
    stat_bp_sigma: float = 5.0
    no_black_clip: bool = False
    mask_method: str = "subtraction"

    def normalized(self) -> "SyQonAttemptProfile":
        tile_size = int(self.tile_size)
        overlap = int(self.overlap)
        stretch_method = str(self.stretch_method).strip().lower()
        mask_method = str(self.mask_method).strip().lower()
        if tile_size not in SYQON_TILE_VALUES:
            raise ValueError(
                f"unsupported SyQon tile_size={tile_size}; allowed={SYQON_TILE_VALUES}"
            )
        if overlap not in SYQON_OVERLAP_VALUES or overlap >= tile_size:
            raise ValueError(
                "unsupported SyQon overlap="
                f"{overlap}; allowed={SYQON_OVERLAP_VALUES}, overlap<tile"
            )
        if tile_size % 16 or overlap % 16:
            raise ValueError("SyQon tile_size and overlap must be divisible by 16")
        if stretch_method not in SYQON_STRETCH_METHODS:
            raise ValueError(f"unsupported SyQon stretch method: {stretch_method}")
        if mask_method not in SYQON_MASK_METHODS:
            raise ValueError(f"unsupported SyQon mask method: {mask_method}")
        target_median = float(self.target_median)
        if not np.isfinite(target_median) or not 0.01 <= target_median <= 0.50:
            raise ValueError("SyQon target_median must be finite and inside 0.01..0.50")
        stat_bp_sigma = float(self.stat_bp_sigma)
        if not np.isfinite(stat_bp_sigma) or not 0.0 <= stat_bp_sigma <= 10.0:
            raise ValueError("SyQon stat_bp_sigma must be finite and inside 0..10")
        return replace(
            self,
            profile_id=str(self.profile_id).strip() or "zenith_attempt",
            tile_size=tile_size,
            overlap=overlap,
            stretch_method=stretch_method,
            target_median=target_median,
            stat_bp_sigma=stat_bp_sigma,
            mask_method=mask_method,
        )

    def manifest(self) -> Dict[str, Any]:
        payload = asdict(self.normalized())
        payload["model"] = "zenith"
        payload["precision"] = "amp" if payload["use_amp"] else "fp32"
        return payload


SYQON_BASELINE_PROFILE = SyQonAttemptProfile(profile_id="zenith_baseline")
SYQON_CPU_RECOVERY_PROFILE = replace(
    SYQON_BASELINE_PROFILE,
    profile_id="zenith_cpu_recovery",
    use_gpu=False,
)
SYQON_BRIGHT_CORE_RECOVERY_PROFILE = replace(
    SYQON_BASELINE_PROFILE,
    profile_id="zenith_bright_core_ihs_recovery",
    tile_size=512,
    overlap=64,
    use_amp=False,
    stretch_method="ihs",
    target_median=0.15,
    mask_method="subtraction",
)


@lru_cache(maxsize=8)
def _sha256_for_file_stat(path_text: str, size: int, mtime_ns: int) -> str:
    """Hash a stable file snapshot once per pipeline process."""
    _ = (size, mtime_ns)
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_syqon_model_dir(
    pipeline: object,
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve and verify the read-only Zenith bundle used by Stage 6."""
    configured = os.getenv(ENV_SYQON_MODEL_DIR_KEY, "").strip()
    if configured:
        model_dir = Path(configured).expanduser()
        source = ENV_SYQON_MODEL_DIR_KEY
    else:
        plugin_dir = getattr(pipeline, "siril_plugin_dir", None)
        if not plugin_dir:
            return None, "未配置 Siril 插件目录，无法定位离线 Zenith 模型"
        model_dir = Path(plugin_dir) / SYQON_MODEL_BUNDLE_REL
        source = "项目离线资源"

    try:
        model_dir = model_dir.resolve()
    except OSError as error:
        return None, f"{source}路径解析失败: {error}"

    model_file = model_dir / "zenith.pt"
    checksum_file = model_dir / "zenith.pt.sha256"
    if not model_file.is_file():
        return None, f"{source}缺少 Zenith 模型: {model_file}"
    if not checksum_file.is_file():
        return None, f"{source}缺少 Zenith 校验文件: {checksum_file}"

    try:
        checksum_parts = checksum_file.read_text(encoding="utf-8").split()
        expected = checksum_parts[0].lower() if checksum_parts else ""
    except OSError as error:
        return None, f"无法读取 Zenith 校验文件: {error}"
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        return None, f"Zenith 校验文件格式无效: {checksum_file}"
    if expected != SYQON_ZENITH_SHA256:
        return None, (
            "Zenith 资产清单与项目锁定摘要不一致: "
            f"expected={SYQON_ZENITH_SHA256}, sidecar={expected}"
        )

    try:
        stat = model_file.stat()
        actual = _sha256_for_file_stat(
            str(model_file),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
    except OSError as error:
        return None, f"无法校验 Zenith 模型: {error}"
    if actual.lower() != expected:
        return None, (
            "Zenith 模型 SHA-256 校验失败: "
            f"expected={expected}, actual={actual.lower()}"
        )
    return model_dir, None


def syqon_starless_cli_options(
    pipeline: object,
    *,
    profile: SyQonAttemptProfile = SYQON_BASELINE_PROFILE,
) -> Tuple[Tuple[str, ...], int, str]:
    profile = profile.normalized()
    use_gpu = profile.use_gpu

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
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
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

    args: List[str] = [
        "--tile-size",
        str(profile.tile_size),
        "--overlap",
        str(profile.overlap),
        "--stretch-method",
        profile.stretch_method,
        "--target-median",
        f"{profile.target_median:.6g}",
        "--stat-bp-sigma",
        f"{profile.stat_bp_sigma:.6g}",
        "--mask-method",
        profile.mask_method,
        "--linked-stretch" if profile.linked_stretch else "--unlinked-stretch",
        "--no-black-clip" if profile.no_black_clip else "--black-clip",
        "--use-amp" if profile.use_amp else "--no-amp",
    ]
    if not use_gpu:
        args.append("--no_gpu")
    if not use_gpu:
        device_note = "CPU forced"
    elif accel_available:
        device_note = f"{accel_name} enabled"
    else:
        device_note = "GPU requested but no torch backend detected; CPU fallback expected"
    device_note += (
        f"; model=Zenith; profile={profile.profile_id}; "
        f"tile={profile.tile_size}; overlap={profile.overlap}; "
        f"stretch={profile.stretch_method}:{profile.target_median:.3f}; "
        f"precision={'AMP' if profile.use_amp else 'FP32'}"
    )
    return tuple(args), timeout_sec, device_note


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
            if path.stem.startswith(
                ("starless_best_", "starmask_best_", "starmask_raw_best_")
            ):
                continue
            seen.add(path)
            pipeline._safe_unlink(path)


def purge_unaccepted_star_separation_outputs(pipeline: object) -> List[str]:
    """Remove every Starless/starmask artifact before a non-accepted handoff."""
    removed: List[str] = []
    seen: set[Path] = set()
    process_dir = getattr(pipeline, "process_dir", None)
    if process_dir:
        patterns = (
            "starless.fit",
            "starless.fits",
            "starless_*.fit",
            "starless_*.fits",
            "stage6_starless.fit",
            "stage6_starless.fits",
            "stage6_starless_*.fit",
            "stage6_starless_*.fits",
            "starmask.fit",
            "starmask.fits",
            "starmask_*.fit",
            "starmask_*.fits",
            "*_starmask.fit",
            "*_starmask.fits",
            "*_stars.fit",
            "*_stars.fits",
        )
        for pattern in patterns:
            for path in process_dir.glob(pattern):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                try:
                    path.unlink()
                except OSError as error:
                    log = getattr(pipeline, "log", None)
                    warn = getattr(log, "warn", None)
                    if callable(warn):
                        warn(
                            "Stage6 could not remove rejected artifact "
                            f"{path.name}: {error}"
                        )
                else:
                    removed.append(path.name)

    work_dir = getattr(pipeline, "work_dir", None)
    if work_dir:
        for name in ("sasp_starless_input.fit", "sasp_starmask_input.fit"):
            path = work_dir / name
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                path.unlink()
            except OSError as error:
                log = getattr(pipeline, "log", None)
                warn = getattr(log, "warn", None)
                if callable(warn):
                    warn(
                        "Stage6 could not remove rejected exchange artifact "
                        f"{path.name}: {error}"
                    )
            else:
                removed.append(path.name)

    if process_dir:
        pointer_path = Path(process_dir) / "stage6_syqon_selected.json"
        if pointer_path.is_file():
            try:
                pointer_path.unlink()
            except OSError as error:
                log = getattr(pipeline, "log", None)
                warn = getattr(log, "warn", None)
                if callable(warn):
                    warn(f"Stage6 could not remove rejected pair pointer: {error}")
            else:
                removed.append(pointer_path.name)
        lineage_root = Path(process_dir) / ".stage6_syqon"
        if lineage_root.is_dir():
            try:
                shutil.rmtree(lineage_root)
            except OSError as error:
                log = getattr(pipeline, "log", None)
                warn = getattr(log, "warn", None)
                if callable(warn):
                    warn(f"Stage6 could not purge rejected pair lineage: {error}")
            else:
                removed.append(lineage_root.name)

    pipeline.starless_file = None
    pipeline.starmask_file = None
    if hasattr(pipeline, "_selected_syqon_pair_id"):
        pipeline._selected_syqon_pair_id = None
    if hasattr(pipeline, "_selected_syqon_attempt_id"):
        pipeline._selected_syqon_attempt_id = None
    if hasattr(pipeline, "_stage6_pair_handoff"):
        pipeline._stage6_pair_handoff = None
    if hasattr(pipeline, "sasp_starless_exchange"):
        pipeline.sasp_starless_exchange = None
    if hasattr(pipeline, "sasp_starmask_exchange"):
        pipeline.sasp_starmask_exchange = None
    return removed


def assess_syqon_exchange_pixels(
    source_data: Any,
    output_data: Any,
) -> Dict[str, Any]:
    """Fail closed on a gross scale/domain change at the SyQon boundary."""
    report: Dict[str, Any] = {
        "schema": SYQON_EXCHANGE_SCHEMA,
        "status": "rejected",
        "accepted": False,
        "issues": [],
        "limits": {
            "median_ratio_min": SYQON_MEDIAN_RATIO_MIN,
            "median_ratio_max": SYQON_MEDIAN_RATIO_MAX,
            "median_signal_floor": SYQON_MEDIAN_SIGNAL_FLOOR,
        },
    }
    try:
        source, source_domain = canonicalize_stage7_pixels_01(source_data)
        output, output_domain = canonicalize_stage7_pixels_01(output_data)
    except (TypeError, ValueError) as error:
        report["issues"] = [f"canonical_domain_error: {error}"]
        return report

    report["source_pixel_domain"] = source_domain
    report["output_pixel_domain"] = output_domain
    if source.shape != output.shape:
        report["issues"] = [
            f"shape_mismatch: source={source.shape}, output={output.shape}"
        ]
        return report

    percentiles = (1.0, 50.0, 99.0)
    source_values = np.percentile(source, percentiles)
    output_values = np.percentile(output, percentiles)
    source_stats = {
        "p01": float(source_values[0]),
        "p50": float(source_values[1]),
        "p99": float(source_values[2]),
    }
    output_stats = {
        "p01": float(output_values[0]),
        "p50": float(output_values[1]),
        "p99": float(output_values[2]),
    }
    report["source_stats"] = source_stats
    report["output_stats"] = output_stats

    source_median = source_stats["p50"]
    output_median = output_stats["p50"]
    median_delta = output_median - source_median
    absolute_drift_limit = max(0.05, source_median * 3.0)
    median_ratio: Optional[float] = None
    issues: List[str] = []
    if source_median > SYQON_MEDIAN_SIGNAL_FLOOR:
        median_ratio = output_median / source_median
        if not (
            SYQON_MEDIAN_RATIO_MIN
            <= median_ratio
            <= SYQON_MEDIAN_RATIO_MAX
        ):
            issues.append(
                "median_scale_ratio_out_of_bounds: "
                f"ratio={median_ratio:.6f}"
            )
        if abs(median_delta) > absolute_drift_limit:
            issues.append(
                "median_absolute_drift_out_of_bounds: "
                f"delta={median_delta:.6f}"
            )

    report["metrics"] = {
        "median_ratio": median_ratio,
        "median_delta": median_delta,
        "absolute_drift_limit": absolute_drift_limit,
    }
    report["issues"] = issues
    report["accepted"] = not issues
    report["status"] = "accepted" if not issues else "rejected"
    return report


def _read_current_pixeldata(pipeline: object) -> Tuple[Optional[np.ndarray], str]:
    siril = getattr(pipeline, "siril", None)
    getter = getattr(siril, "get_image_pixeldata", None)
    if not callable(getter):
        return None, "Siril pixel getter unavailable"
    try:
        try:
            pixels = getter(preview=False)
        except TypeError:
            pixels = getter()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return None, f"Siril pixel read failed: {error}"
    if pixels is None:
        return None, "Siril returned an empty pixel buffer"
    try:
        return np.array(pixels, copy=True), ""
    except (TypeError, ValueError) as error:
        return None, f"Siril pixel buffer conversion failed: {error}"


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_for_file_stat(
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _read_fits_pixels(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
        if not hdul or hdul[0].data is None:
            raise ValueError(f"FITS primary image is empty: {path}")
        return np.array(hdul[0].data, copy=True)


def _pixel_file_manifest(path: Path, pixels: np.ndarray) -> Dict[str, Any]:
    canonical, domain = canonicalize_stage7_pixels_01(pixels)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
        "shape": [int(value) for value in canonical.shape],
        "dtype": str(np.asarray(pixels).dtype),
        "canonical_dtype": str(canonical.dtype),
        "pixel_domain": domain,
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def _runtime_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "numpy": np.__version__,
    }
    for module_name in ("torch", "astropy", "scipy"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except (ImportError, OSError, RuntimeError):
            versions[module_name] = "unavailable"
    return versions


def verify_syqon_supply_chain(
    pipeline: object,
    syqon_script: Path,
    model_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Verify the pinned upstream/patch/script/model lock before execution."""
    plugin_root = Path(getattr(pipeline, "siril_plugin_dir", "") or "")
    checksum_path = plugin_root / "asset-checksums.sha256"
    patch_path = plugin_root / "patches" / "apply_syqon_offline_model_patch.py"
    try:
        entries: Dict[str, str] = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                entries[parts[1]] = parts[0].lower()
    except OSError as error:
        return None, f"SyQon asset lock manifest unavailable: {error}"

    expected_raw = entries.get("siril-scripts/upstream/SyQon/Starless.py")
    if expected_raw != SYQON_UPSTREAM_STARLESS_SHA256:
        return None, "SyQon pinned upstream Starless.py digest is missing or changed"
    required = {
        "script": (
            syqon_script,
            "vendor/siril-scripts/SyQon/Starless.py",
        ),
        "patch": (
            patch_path,
            "patches/apply_syqon_offline_model_patch.py",
        ),
        "model": (
            model_dir / "zenith.pt",
            "syqon_starless/zenith.pt",
        ),
    }
    assets: Dict[str, Any] = {
        "upstream": {
            "commit": SYQON_UPSTREAM_COMMIT,
            "path": "SyQon/Starless.py",
            "sha256": SYQON_UPSTREAM_STARLESS_SHA256,
        }
    }
    for name, (path, relative) in required.items():
        expected = entries.get(relative)
        if not expected or len(expected) != 64:
            return None, f"SyQon asset lock missing: {relative}"
        if not path.is_file():
            return None, f"SyQon locked asset missing: {path}"
        try:
            actual = _sha256_file(path)
        except OSError as error:
            return None, f"SyQon locked asset unreadable: {path}: {error}"
        if actual != expected:
            return None, (
                f"SyQon locked asset digest mismatch: {relative}; "
                f"expected={expected}, actual={actual}"
            )
        assets[name] = {
            "path": str(path),
            "sha256": actual,
        }
    assets["model"].update(name="Zenith", output_semantics="residual")
    return assets, None


def _closure_shadow_metrics(
    source_pixels: np.ndarray,
    starless_pixels: np.ndarray,
    starmask_pixels: np.ndarray,
) -> Dict[str, Any]:
    try:
        source, _ = canonicalize_stage7_pixels_01(source_pixels)
        starless, _ = canonicalize_stage7_pixels_01(starless_pixels)
        starmask, _ = canonicalize_stage7_pixels_01(starmask_pixels)
    except (TypeError, ValueError) as error:
        return {"status": "unavailable", "reason": str(error)}
    if source.shape != starless.shape or source.shape != starmask.shape:
        return {
            "status": "unavailable",
            "reason": (
                "shape mismatch: "
                f"source={source.shape}, starless={starless.shape}, "
                f"starmask={starmask.shape}"
            ),
        }
    absolute_error = np.abs(source - (starless + starmask))
    return {
        "status": "shadow",
        "domain": "canonical_linear_0..1",
        "mae": float(np.mean(absolute_error)),
        "p99_abs": float(np.percentile(absolute_error, 99.0)),
        "max_abs": float(np.max(absolute_error)),
        "starless_above_source_ratio": float(np.mean(starless > source + 2e-6)),
    }


def _tiling_shadow_metrics(worker_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Expose geometry/coverage diagnostics without adding model inference."""
    actual = worker_manifest.get("actual") or {}
    if not isinstance(actual, dict):
        return {"status": "unavailable", "reason": "worker geometry missing"}
    grid = actual.get("grid") or {}
    if not isinstance(grid, dict):
        grid = {}
    mode = str(actual.get("mode") or "unknown")
    try:
        coverage_min = float(actual["coverage_min"])
        coverage_max = float(actual["coverage_max"])
    except (KeyError, TypeError, ValueError):
        return {"status": "unavailable", "reason": "worker coverage missing"}
    if not np.isfinite(coverage_min) or not np.isfinite(coverage_max):
        return {"status": "unavailable", "reason": "worker coverage is non-finite"}

    y_positions = [int(value) for value in grid.get("y_positions", [])]
    x_positions = [int(value) for value in grid.get("x_positions", [])]
    has_internal_boundary = len(y_positions) > 1 or len(x_positions) > 1
    boundary_band = (
        {
            "status": "shadow",
            "basis": "global blend-coverage envelope over all pixels, including seams",
            "seam_y_positions": y_positions[1:],
            "seam_x_positions": x_positions[1:],
            "coverage_min": coverage_min,
            "coverage_max": coverage_max,
        }
        if has_internal_boundary
        else {
            "status": "unavailable",
            "reason": "single tile/full-frame execution has no internal boundary band",
        }
    )
    return {
        "status": "shadow",
        "mode": mode,
        "coverage": {
            "status": "shadow",
            "minimum": coverage_min,
            "maximum": coverage_max,
            "has_gap": coverage_min <= 0.0,
        },
        "boundary_band": boundary_band,
        "tile_full_comparison": {
            "status": "unavailable",
            "reason": (
                "paired full-frame inference is intentionally not run in production"
                if mode == "tiled"
                else "execution used one full-frame path; no paired tiled result"
            ),
        },
    }


def _load_worker_manifest(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        return {"status": "unavailable", "reason": str(error)}
    if not isinstance(payload, dict):
        return {"status": "unavailable", "reason": "worker manifest is not an object"}
    return payload


def _validate_worker_manifest(
    payload: Dict[str, Any],
    profile: SyQonAttemptProfile,
    *,
    expected_spatial_shape: Tuple[int, int],
) -> List[str]:
    """Validate that file-mode honored every code-owned Zenith parameter."""
    issues: List[str] = []
    if payload.get("schema") != "starun.syqon-worker.v1":
        issues.append("worker manifest schema mismatch")
    if payload.get("status") != "accepted":
        issues.append("worker manifest did not report accepted status")
    if str(payload.get("model") or "").lower() != "zenith":
        issues.append("worker manifest selected a non-Zenith model")

    requested = payload.get("requested")
    if not isinstance(requested, dict):
        issues.append("worker requested parameter manifest missing")
        requested = {}
    expected_requested = {
        "tile_size": profile.tile_size,
        "overlap": profile.overlap,
        "use_gpu": profile.use_gpu,
        "use_amp": profile.use_amp,
        "stretch_method": profile.stretch_method,
        "target_median": profile.target_median,
        "linked_stretch": profile.linked_stretch,
        "stat_bp_sigma": profile.stat_bp_sigma,
        "no_black_clip": profile.no_black_clip,
        "mask_method": profile.mask_method,
    }
    for name, expected in expected_requested.items():
        actual = requested.get(name)
        if isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) <= 1e-9
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            issues.append(
                f"worker requested parameter mismatch: {name}={actual!r}, "
                f"expected={expected!r}"
            )

    actual = payload.get("actual")
    if not isinstance(actual, dict):
        issues.append("worker actual execution manifest missing")
        actual = {}
    if str(actual.get("model") or "").lower() != "zenith":
        issues.append("worker actual model is not Zenith")
    if actual.get("tile_size") != profile.tile_size:
        issues.append("worker actual tile_size mismatch")
    if actual.get("overlap") != profile.overlap:
        issues.append("worker actual overlap mismatch")
    if bool(actual.get("actual_amp", False)) and not profile.use_amp:
        issues.append("worker enabled AMP although the profile disabled it")
    if str(actual.get("mask_method") or "") != profile.mask_method:
        issues.append("worker actual mask method mismatch")
    stretch = actual.get("stretch")
    if not isinstance(stretch, dict):
        issues.append("worker actual stretch manifest missing")
        stretch = {}
    for name, expected in (
        ("method", profile.stretch_method),
        ("linked", profile.linked_stretch),
        ("no_black_clip", profile.no_black_clip),
    ):
        if stretch.get(name) != expected:
            issues.append(f"worker actual stretch {name} mismatch")
    for name, expected in (
        ("target_median", profile.target_median),
        ("stat_bp_sigma", profile.stat_bp_sigma),
    ):
        try:
            matches = abs(float(stretch.get(name)) - expected) <= 1e-9
        except (TypeError, ValueError):
            matches = False
        if not matches:
            issues.append(f"worker actual stretch {name} mismatch")

    try:
        coverage_min = float(actual.get("coverage_min"))
    except (TypeError, ValueError):
        coverage_min = 0.0
    if not np.isfinite(coverage_min) or coverage_min <= 0.0:
        issues.append("worker actual coverage has a gap")
    try:
        coverage_max = float(actual.get("coverage_max"))
    except (TypeError, ValueError):
        coverage_max = float("nan")
    if not np.isfinite(coverage_max) or coverage_max < coverage_min:
        issues.append("worker actual coverage maximum is invalid")
    crop_shape = actual.get("crop_shape")
    if crop_shape != [int(value) for value in expected_spatial_shape]:
        issues.append(
            "worker crop shape mismatch: "
            f"actual={crop_shape!r}, expected={list(expected_spatial_shape)!r}"
        )
    for name in ("padding", "padded_shape", "grid"):
        if name not in actual:
            issues.append(f"worker actual geometry missing: {name}")
    shadow = payload.get("shadow_metrics")
    if not isinstance(shadow, dict) or "transform_roundtrip" not in shadow:
        issues.append("worker transform roundtrip shadow metric missing")
    return issues


def _spatial_shape(pixels: np.ndarray) -> Tuple[int, int]:
    shape = tuple(int(value) for value in np.asarray(pixels).shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[0] < min(shape[1:]):
        return shape[1], shape[2]
    if len(shape) == 3 and shape[2] in (1, 3, 4):
        return shape[0], shape[1]
    raise ValueError(f"unsupported SyQon image shape: {shape}")


def _attempt_failure_code(reason: str) -> str:
    lowered = str(reason).lower()
    if "sha" in lowered or "model" in lowered and "integrity" in lowered:
        return "MODEL_INTEGRITY"
    if "starmask" in lowered or "pair" in lowered:
        return "PAIR_MISMATCH"
    if "shape" in lowered or "domain" in lowered or "nan" in lowered or "inf" in lowered:
        return "PIXEL_DOMAIN"
    if "out of memory" in lowered or "oom" in lowered:
        return "OOM_DEVICE"
    if "torch" in lowered or "cuda" in lowered or "mps" in lowered:
        return "RUNTIME_INCOMPATIBLE"
    return "SCRIPT_CONTRACT"


def record_syqon_derived_generation(
    pipeline: object,
    *,
    generation: str,
    details: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Commit a clean/repaired child pair without mutating its raw parent."""
    process_dir = getattr(pipeline, "process_dir", None)
    starless_file = getattr(pipeline, "starless_file", None)
    starmask_file = getattr(pipeline, "starmask_file", None)
    parent_pair_id = getattr(pipeline, "_selected_syqon_pair_id", None)
    if (
        not process_dir
        or not parent_pair_id
        or not isinstance(starless_file, Path)
        or not starless_file.is_file()
        or not isinstance(starmask_file, Path)
        or not starmask_file.is_file()
    ):
        return None
    generation = str(generation).strip().lower()
    if generation not in {"clean", "repaired"}:
        raise ValueError(f"unsupported SyQon derived generation: {generation}")

    try:
        starless_pixels = _read_fits_pixels(starless_file)
        starmask_pixels = _read_fits_pixels(starmask_file)
        starless_manifest = _pixel_file_manifest(starless_file, starless_pixels)
        starmask_manifest = _pixel_file_manifest(starmask_file, starmask_pixels)
    except (OSError, TypeError, ValueError) as error:
        log = getattr(pipeline, "log", None)
        warn = getattr(log, "warn", None)
        if callable(warn):
            warn(f"SyQon {generation} generation not committed: {error}")
        return None
    if starless_manifest["shape"] != starmask_manifest["shape"]:
        return None

    parent_report = dict(
        getattr(pipeline, "_last_syqon_exchange_report", {}) or {}
    )

    pair_seed = "|".join(
        (
            str(parent_pair_id),
            starless_manifest["sha256"],
            starmask_manifest["sha256"],
            generation,
        )
    )
    pair_id = hashlib.sha256(pair_seed.encode("utf-8")).hexdigest()
    attempt_id = f"{generation}-{uuid.uuid4().hex[:12]}"
    attempts_root = Path(process_dir) / ".stage6_syqon"
    attempts_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{attempt_id}.", suffix=".tmp", dir=attempts_root)
    )
    try:
        _atomic_copy(starless_file, temp_dir / "starless.fit")
        _atomic_copy(starmask_file, temp_dir / "starmask.fit")
        starless_manifest["path"] = "starless.fit"
        starmask_manifest["path"] = "starmask.fit"
        manifest = {
            "schema": SYQON_ATTEMPT_SCHEMA,
            "attempt_id": attempt_id,
            "pair_id": pair_id,
            "parent_pair_id": parent_pair_id,
            "generation": generation,
            "stop_reason": "DERIVED_GENERATION_COMMITTED",
            "files": {
                "source": dict(
                    (parent_report.get("files") or {}).get("source") or {}
                ),
                "starless": starless_manifest,
                f"starmask_{generation}": starmask_manifest,
            },
            "profile": dict(parent_report.get("profile") or {}),
            "assets": dict(parent_report.get("assets") or {}),
            "runtime": dict(parent_report.get("runtime") or {}),
            "worker": dict(parent_report.get("worker") or {}),
            "shadow_metrics": dict(parent_report.get("shadow_metrics") or {}),
            "details": dict(details or {}),
        }
        _atomic_write_json(temp_dir / "attempt-manifest.json", manifest)
        final_dir = attempts_root / attempt_id
        os.replace(temp_dir, final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    selected_payload = {
        "schema": SYQON_SELECTED_SCHEMA,
        "attempt_id": attempt_id,
        "pair_id": pair_id,
        "parent_pair_id": parent_pair_id,
        "generation": generation,
        "stop_reason": "DERIVED_GENERATION_COMMITTED",
        "attempt_manifest": str(
            (final_dir / "attempt-manifest.json").relative_to(process_dir)
        ),
        "starless": str((final_dir / "starless.fit").relative_to(process_dir)),
        f"starmask_{generation}": str(
            (final_dir / "starmask.fit").relative_to(process_dir)
        ),
    }
    _atomic_write_json(Path(process_dir) / "stage6_syqon_selected.json", selected_payload)
    pipeline._selected_syqon_pair_id = pair_id
    pipeline._selected_syqon_attempt_id = attempt_id
    report = parent_report
    if report:
        report_files = dict(report.get("files") or {})
        report_files[f"starless_{generation}"] = starless_manifest
        report_files[f"starmask_{generation}"] = starmask_manifest
        generation_records = list(report.get("generations") or [])
        generation_records.append(
            {
                "attempt_id": attempt_id,
                "pair_id": pair_id,
                "parent_pair_id": parent_pair_id,
                "generation": generation,
                "stop_reason": "DERIVED_GENERATION_COMMITTED",
                "attempt_manifest": selected_payload["attempt_manifest"],
                "starless_sha256": starless_manifest["sha256"],
                "starmask_sha256": starmask_manifest["sha256"],
            }
        )
        report.update(
            {
                "pair_id": pair_id,
                "parent_pair_id": parent_pair_id,
                "generation": generation,
                "stop_reason": "DERIVED_GENERATION_COMMITTED",
                "files": report_files,
                "generations": generation_records,
                "selected_pointer": selected_payload,
            }
        )
        _write_syqon_exchange_report(pipeline, report)
    return selected_payload


def verify_selected_syqon_pair(
    pipeline: object,
    *,
    expected_pair_id: Optional[str] = None,
    verify_canonical_aliases: bool = True,
) -> Dict[str, Any]:
    """Verify the immutable selected generation before downstream consumption."""
    process_dir = getattr(pipeline, "process_dir", None)
    expected_pair_id = expected_pair_id or getattr(
        pipeline,
        "_selected_syqon_pair_id",
        None,
    )
    if not process_dir or not expected_pair_id:
        return {"status": "not_syqon", "accepted": None}
    pointer_path = Path(process_dir) / "stage6_syqon_selected.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict):
            raise ValueError("selected pointer is not an object")
        if pointer.get("pair_id") != expected_pair_id:
            raise ValueError("selected pointer pair_id does not match pipeline state")
        manifest_rel = str(pointer.get("attempt_manifest") or "")
        manifest_path = (Path(process_dir) / manifest_rel).resolve()
        manifest_path.relative_to(Path(process_dir).resolve())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("pair_id") != expected_pair_id:
            raise ValueError("attempt manifest pair_id mismatch")
        files = manifest.get("files") or {}
        if not isinstance(files, dict):
            raise ValueError("attempt manifest files are invalid")
        verified: Dict[str, Any] = {}
        verified_paths: Dict[str, str] = {}
        for name, file_manifest in files.items():
            if name == "source":
                # Derived generations retain the parent source manifest without
                # copying that file into the child directory.  The new Stage6
                # handoff independently hashes the exact stage6_input.
                continue
            if not isinstance(file_manifest, dict):
                raise ValueError(f"invalid file manifest: {name}")
            file_name = str(file_manifest.get("path") or "")
            file_path = (manifest_path.parent / file_name).resolve()
            file_path.relative_to(Path(process_dir).resolve())
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            actual_sha256 = _sha256_file(file_path)
            if actual_sha256 != file_manifest.get("sha256"):
                raise ValueError(f"hash mismatch for {name}")
            verified[name] = actual_sha256
            verified_paths[name] = str(file_path)

        if verify_canonical_aliases:
            canonical_starless = getattr(pipeline, "starless_file", None)
            expected_starless = files.get("starless")
            if isinstance(canonical_starless, Path) and canonical_starless.is_file():
                if not isinstance(expected_starless, dict):
                    raise ValueError("selected generation has no starless manifest")
                if _sha256_file(canonical_starless) != expected_starless.get("sha256"):
                    raise ValueError("canonical starless does not match selected pair")
            canonical_starmask = getattr(pipeline, "starmask_file", None)
            if isinstance(canonical_starmask, Path) and canonical_starmask.is_file():
                starmask_entries = [
                    item
                    for name, item in files.items()
                    if str(name).startswith("starmask_") and isinstance(item, dict)
                ]
                if len(starmask_entries) != 1:
                    raise ValueError("selected generation starmask manifest is ambiguous")
                if _sha256_file(canonical_starmask) != starmask_entries[0].get("sha256"):
                    raise ValueError("canonical starmask does not match selected pair")
    except (OSError, TypeError, ValueError) as error:
        return {
            "status": "rejected",
            "accepted": False,
            "failure_code": "PAIR_MISMATCH",
            "reason": str(error),
        }
    return {
        "status": "accepted",
        "accepted": True,
        "pair_id": expected_pair_id,
        "attempt_id": pointer.get("attempt_id"),
        "generation": pointer.get("generation"),
        "verified_files": verified,
        "verified_paths": verified_paths,
        "canonical_aliases_verified": bool(verify_canonical_aliases),
    }


def record_stage6_pair_handoff(
    pipeline: object,
    *,
    source_path: Path,
    starless_path: Path,
) -> Dict[str, Any]:
    """Freeze the exact Stage 6 pair consumed by later matched-domain stages."""
    process_dir_value = getattr(pipeline, "process_dir", None)
    if process_dir_value is None:
        return {
            "schema": STAGE6_PAIR_HANDOFF_SCHEMA,
            "status": "unavailable",
            "accepted": False,
            "reason": "process directory unavailable",
        }
    process_dir = Path(process_dir_value).resolve()
    try:
        role_paths = {
            "stage6_input": Path(source_path).resolve(),
            "stage6_starless": Path(starless_path).resolve(),
        }
        manifests: Dict[str, Dict[str, Any]] = {}
        for role, path in role_paths.items():
            path.relative_to(process_dir)
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest = _pixel_file_manifest(path, _read_fits_pixels(path))
            manifest["path"] = str(path.relative_to(process_dir))
            manifests[role] = manifest
        if manifests["stage6_input"]["shape"] != manifests["stage6_starless"]["shape"]:
            raise ValueError("Stage6 handoff pair shape mismatch")

        pair_id = getattr(pipeline, "_selected_syqon_pair_id", None)
        payload = {
            "schema": STAGE6_PAIR_HANDOFF_SCHEMA,
            "status": "accepted",
            "accepted": True,
            "pair_id": pair_id,
            "attempt_id": getattr(pipeline, "_selected_syqon_attempt_id", None),
            "files": manifests,
        }
        _atomic_write_json(process_dir / "stage6_pair_handoff.json", payload)
        setattr(pipeline, "_stage6_pair_handoff", payload)
        return payload
    except (OSError, TypeError, ValueError) as error:
        payload = {
            "schema": STAGE6_PAIR_HANDOFF_SCHEMA,
            "status": "rejected",
            "accepted": False,
            "reason": str(error),
        }
        setattr(pipeline, "_stage6_pair_handoff", payload)
        return payload


def verify_stage6_pair_handoff(pipeline: object) -> Dict[str, Any]:
    """Verify explicit Stage 6 artifacts without consulting mutable live aliases."""
    process_dir_value = getattr(pipeline, "process_dir", None)
    if process_dir_value is None:
        return {
            "schema": STAGE6_PAIR_HANDOFF_SCHEMA,
            "status": "unavailable",
            "accepted": False,
            "reason_code": "stage9_stage6_pair_handoff_unavailable",
            "reason": "process directory unavailable",
        }
    process_dir = Path(process_dir_value).resolve()
    handoff_path = process_dir / "stage6_pair_handoff.json"
    try:
        payload = json.loads(handoff_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Stage6 pair handoff is not an object")
        if payload.get("schema") != STAGE6_PAIR_HANDOFF_SCHEMA:
            raise ValueError("Stage6 pair handoff schema mismatch")
        if payload.get("accepted") is not True:
            raise ValueError(str(payload.get("reason") or "Stage6 pair not accepted"))
        files = payload.get("files") or {}
        if not isinstance(files, dict):
            raise ValueError("Stage6 pair handoff files are invalid")

        verified_paths: Dict[str, str] = {}
        verified_domains: Dict[str, Any] = {}
        verified_shapes: Dict[str, Any] = {}
        for role in ("stage6_input", "stage6_starless"):
            manifest = files.get(role)
            if not isinstance(manifest, dict):
                raise ValueError(f"Stage6 pair handoff role missing: {role}")
            relative_path = Path(str(manifest.get("path") or ""))
            path = (process_dir / relative_path).resolve()
            path.relative_to(process_dir)
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256_file(path) != manifest.get("sha256"):
                raise ValueError(f"Stage6 pair handoff hash mismatch: {role}")
            pixels = _read_fits_pixels(path)
            canonical, domain = canonicalize_stage7_pixels_01(pixels)
            shape = [int(value) for value in canonical.shape]
            if shape != manifest.get("shape"):
                raise ValueError(f"Stage6 pair handoff shape mismatch: {role}")
            verified_paths[role] = str(path)
            verified_domains[role] = domain
            verified_shapes[role] = shape
        if verified_shapes["stage6_input"] != verified_shapes["stage6_starless"]:
            raise ValueError("Stage6 pair handoff role shapes differ")

        pair_id = payload.get("pair_id")
        generation_verification: Dict[str, Any]
        if pair_id:
            generation_verification = verify_selected_syqon_pair(
                pipeline,
                expected_pair_id=str(pair_id),
                verify_canonical_aliases=False,
            )
            if generation_verification.get("accepted") is not True:
                raise ValueError(
                    "selected SyQon generation mismatch: "
                    f"{generation_verification.get('reason') or generation_verification.get('status')}"
                )
        else:
            generation_verification = {
                "status": "not_syqon",
                "accepted": None,
            }
        report = {
            "schema": STAGE6_PAIR_HANDOFF_SCHEMA,
            "status": "accepted",
            "accepted": True,
            "reason_code": "stage9_stage6_pair_verified",
            "pair_id": pair_id,
            "paths": verified_paths,
            "pixel_domains": verified_domains,
            "shapes": verified_shapes,
            "generation_verification": generation_verification,
        }
        setattr(pipeline, "_stage6_pair_handoff", payload)
        return report
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        return {
            "schema": STAGE6_PAIR_HANDOFF_SCHEMA,
            "status": "rejected" if handoff_path.exists() else "unavailable",
            "accepted": False,
            "reason_code": (
                "stage9_stage6_pair_mismatch"
                if handoff_path.exists()
                else "stage9_stage6_pair_handoff_unavailable"
            ),
            "reason": str(error),
        }


def _write_syqon_exchange_report(pipeline: object, report: Dict[str, Any]) -> None:
    payload = dict(report)
    if not payload.get("stop_reason"):
        failure_code = str(payload.get("failure_code") or "").strip()
        status = str(payload.get("status") or "unknown").strip().upper()
        payload["stop_reason"] = failure_code or status
    pipeline._last_syqon_exchange_report = payload
    writer = getattr(pipeline, "_write_stage_json", None)
    if not callable(writer):
        return
    try:
        writer("stage6_syqon_exchange.json", payload)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        log = getattr(pipeline, "log", None)
        warn = getattr(log, "warn", None)
        if callable(warn):
            warn(f"SyQon exchange report write failed: {error}")


def stage7_try_syqon_variant(
    pipeline: object,
    syqon_script: Path,
    *,
    attempt_name: str,
    profile: SyQonAttemptProfile = SYQON_BASELINE_PROFILE,
) -> Optional[str]:
    try:
        profile = profile.normalized()
    except (TypeError, ValueError) as error:
        pipeline._last_plugin_script_error = f"SyQon profile contract failed: {error}"
        pipeline.log.warn(pipeline._last_plugin_script_error)
        return None

    model_dir, model_error = resolve_syqon_model_dir(pipeline)
    if model_dir is None:
        pipeline._last_plugin_script_error = (
            "SyQon 离线模型预检失败: " + (model_error or "未知错误")
        )
        pipeline.log.warn(pipeline._last_plugin_script_error)
        _write_syqon_exchange_report(
            pipeline,
            {
                "schema": SYQON_EXCHANGE_SCHEMA,
                "status": "rejected",
                "accepted": False,
                "failure_code": "MODEL_INTEGRITY",
                "reason": pipeline._last_plugin_script_error,
                "profile": profile.manifest(),
            },
        )
        return None
    os.environ[ENV_SYQON_MODEL_DIR_KEY] = str(model_dir)
    pipeline.log.info(f"SyQon 使用已校验的本地 Zenith 模型: {model_dir}")

    assets, supply_error = verify_syqon_supply_chain(
        pipeline,
        syqon_script,
        model_dir,
    )
    if assets is None:
        pipeline._last_plugin_script_error = supply_error or "SyQon asset lock failed"
        pipeline.log.warn(pipeline._last_plugin_script_error)
        _write_syqon_exchange_report(
            pipeline,
            {
                "schema": SYQON_EXCHANGE_SCHEMA,
                "status": "rejected",
                "accepted": False,
                "failure_code": "SCRIPT_CONTRACT",
                "reason": pipeline._last_plugin_script_error,
                "profile": profile.manifest(),
            },
        )
        return None

    if not getattr(pipeline, "process_dir", None):
        pipeline._last_plugin_script_error = "SyQon Stage6 process directory is unavailable"
        pipeline.log.warn(pipeline._last_plugin_script_error)
        return None

    pipeline._clear_star_separation_outputs()
    pipeline.starless_file = None
    pipeline.starmask_file = None
    pipeline.cmd_with_check("load", pipeline.stretched_name)
    source_pixels, source_read_error = _read_current_pixeldata(pipeline)
    source_file: Optional[Path] = None
    if pipeline.process_dir and pipeline.stretched_name:
        for suffix in (".fit", ".fits", ".fts"):
            candidate = pipeline.process_dir / f"{pipeline.stretched_name}{suffix}"
            if candidate.is_file():
                source_file = candidate
                break
    if source_file is None:
        pipeline._last_plugin_script_error = (
            "SyQon 文件输入不存在: "
            f"{pipeline.stretched_name or 'unknown source'}"
        )
        pipeline.log.warn(pipeline._last_plugin_script_error)
        return None

    safe_attempt_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(attempt_name)
    ).strip("_") or "attempt"
    attempt_id = f"{safe_attempt_name[:48]}-{uuid.uuid4().hex[:12]}"
    attempts_root = pipeline.process_dir / ".stage6_syqon"
    attempts_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{attempt_id}.", suffix=".tmp", dir=attempts_root)
    )
    target_starless = temp_dir / "starless.fit"
    target_starmask = temp_dir / "starmask.fit"
    worker_manifest_path = temp_dir / "worker-manifest.json"
    syqon_args, syqon_timeout, syqon_device_note = pipeline._syqon_starless_cli_options(
        profile=profile,
    )
    syqon_file_args = (
        *syqon_args,
        "--input-file",
        str(source_file),
        "--starless-output",
        str(target_starless),
        "--starmask-output",
        str(target_starmask),
        "--manifest-output",
        str(worker_manifest_path),
    )
    used = pipeline._run_plugin_script_cli_subprocess(
        "去星",
        f"SyQon Starless {attempt_name} ({syqon_device_note})",
        syqon_script,
        args=syqon_file_args,
        timeout_sec=syqon_timeout,
        verify_image_change=False,
        uses_siril_connection=False,
    )
    if not used:
        reason = str(
            getattr(pipeline, "_last_plugin_script_error", "SyQon worker failed")
        )
        _write_syqon_exchange_report(
            pipeline,
            {
                "schema": SYQON_EXCHANGE_SCHEMA,
                "status": "rejected",
                "accepted": False,
                "attempt_id": attempt_id,
                "failure_code": _attempt_failure_code(reason),
                "reason": reason,
                "profile": profile.manifest(),
            },
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    if not target_starless.is_file() or not target_starmask.is_file():
        missing = [
            path.name
            for path in (target_starless, target_starmask)
            if not path.is_file()
        ]
        pipeline._last_plugin_script_error = (
            "SyQon 成对产物不完整: " + ", ".join(missing)
        )
        _write_syqon_exchange_report(
            pipeline,
            {
                "schema": SYQON_EXCHANGE_SCHEMA,
                "status": "rejected",
                "accepted": False,
                "attempt_id": attempt_id,
                "failure_code": "PAIR_MISMATCH",
                "reason": pipeline._last_plugin_script_error,
                "profile": profile.manifest(),
            },
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    try:
        source_file_pixels = _read_fits_pixels(source_file)
        output_pixels = _read_fits_pixels(target_starless)
        starmask_pixels = _read_fits_pixels(target_starmask)
    except (OSError, TypeError, ValueError) as error:
        pipeline._last_plugin_script_error = f"SyQon pixel contract read failed: {error}"
        _write_syqon_exchange_report(
            pipeline,
            {
                "schema": SYQON_EXCHANGE_SCHEMA,
                "status": "rejected",
                "accepted": False,
                "attempt_id": attempt_id,
                "failure_code": "PIXEL_DOMAIN",
                "reason": pipeline._last_plugin_script_error,
                "profile": profile.manifest(),
            },
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    exchange_report = assess_syqon_exchange_pixels(
        source_file_pixels,
        output_pixels,
    )
    if source_pixels is None:
        source_pixels = source_file_pixels
        source_read_error = ""
    exchange_report.update(
        {
            "attempt_id": attempt_id,
            "parent_pair_id": getattr(pipeline, "_selected_syqon_pair_id", None),
            "generation": "raw",
            "profile": profile.manifest(),
            "requested_device": "gpu" if profile.use_gpu else "cpu",
            "requested_precision": "amp" if profile.use_amp else "fp32",
        }
    )

    if np.asarray(starmask_pixels).shape != np.asarray(source_file_pixels).shape:
        exchange_report["status"] = "rejected"
        exchange_report["accepted"] = False
        exchange_report.setdefault("issues", []).append(
            "starmask_shape_mismatch: "
            f"source={np.asarray(source_file_pixels).shape}, "
            f"starmask={np.asarray(starmask_pixels).shape}"
        )

    if exchange_report.get("status") == "rejected":
        issue_text = ", ".join(
            str(issue) for issue in exchange_report.get("issues", [])
        ) or "unknown pixel exchange violation"
        pipeline._last_plugin_script_error = (
            "SyQon 数值交换哨兵拒绝产物: " + issue_text
        )
        pipeline.log.warn(pipeline._last_plugin_script_error)
        clear_star_separation_outputs(pipeline)
        pipeline.starless_file = None
        pipeline.starmask_file = None
        exchange_report["failure_code"] = _attempt_failure_code(issue_text)
        rollback_error = ""
        try:
            pipeline.cmd_with_check("load", pipeline.stretched_name)
        except (CommandError, SirilError, OSError, RuntimeError) as error:
            rollback_error = str(error)
            pipeline.log.warn(f"SyQon 数值交换拒绝后的源图回滚失败: {error}")
        exchange_report["rollback"] = {
            "source_stem": str(pipeline.stretched_name),
            "status": "failed" if rollback_error else "restored",
            "error": rollback_error or None,
        }
        _write_syqon_exchange_report(pipeline, exchange_report)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    worker_manifest = _load_worker_manifest(worker_manifest_path)
    if worker_manifest.get("status") == "unavailable":
        pipeline._last_plugin_script_error = (
            "SyQon worker manifest unavailable: "
            f"{worker_manifest.get('reason', 'unknown')}"
        )
        exchange_report.update(
            {
                "status": "rejected",
                "accepted": False,
                "failure_code": "SCRIPT_CONTRACT",
                "reason": pipeline._last_plugin_script_error,
            }
        )
        _write_syqon_exchange_report(pipeline, exchange_report)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    try:
        worker_manifest_issues = _validate_worker_manifest(
            worker_manifest,
            profile,
            expected_spatial_shape=_spatial_shape(source_file_pixels),
        )
    except (TypeError, ValueError) as error:
        worker_manifest_issues = [str(error)]
    if worker_manifest_issues:
        pipeline._last_plugin_script_error = (
            "SyQon worker parameter/geometry contract failed: "
            + "; ".join(worker_manifest_issues)
        )
        exchange_report.update(
            {
                "status": "rejected",
                "accepted": False,
                "failure_code": "SCRIPT_CONTRACT",
                "reason": pipeline._last_plugin_script_error,
                "worker": worker_manifest,
            }
        )
        _write_syqon_exchange_report(pipeline, exchange_report)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    try:
        source_manifest = _pixel_file_manifest(source_file, source_file_pixels)
        starless_manifest = _pixel_file_manifest(target_starless, output_pixels)
        starmask_manifest = _pixel_file_manifest(target_starmask, starmask_pixels)
    except (OSError, TypeError, ValueError) as error:
        pipeline._last_plugin_script_error = f"SyQon file manifest rejected: {error}"
        exchange_report.update(
            {
                "status": "rejected",
                "accepted": False,
                "failure_code": "PIXEL_DOMAIN",
                "reason": pipeline._last_plugin_script_error,
                "worker": worker_manifest,
            }
        )
        _write_syqon_exchange_report(pipeline, exchange_report)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    pair_seed = "|".join(
        (
            source_manifest["sha256"],
            starless_manifest["sha256"],
            starmask_manifest["sha256"],
            profile.profile_id,
        )
    )
    pair_id = hashlib.sha256(pair_seed.encode("utf-8")).hexdigest()
    assert assets is not None
    closure_shadow = _closure_shadow_metrics(
        source_file_pixels,
        output_pixels,
        starmask_pixels,
    )
    worker_shadow = worker_manifest.get("shadow_metrics") or {}
    if not isinstance(worker_shadow, dict):
        worker_shadow = {}
    for file_manifest, name in (
        (source_manifest, source_file.name),
        (starless_manifest, "starless.fit"),
        (starmask_manifest, "starmask.fit"),
    ):
        file_manifest["path"] = name

    attempt_manifest: Dict[str, Any] = {
        "schema": SYQON_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "pair_id": pair_id,
        "parent_pair_id": getattr(pipeline, "_selected_syqon_pair_id", None),
        "generation": "raw",
        "stop_reason": "CONTRACT_VALID_PAIR_COMMITTED",
        "profile": profile.manifest(),
        "files": {
            "source": source_manifest,
            "starless": starless_manifest,
            "starmask_raw": starmask_manifest,
        },
        "assets": assets,
        "runtime": _runtime_versions(),
        "worker": worker_manifest,
        "shadow_metrics": {
            "transform_roundtrip": worker_shadow.get(
                "transform_roundtrip",
                {"status": "unavailable", "reason": "worker metric missing"},
            ),
            "tiling": _tiling_shadow_metrics(worker_manifest),
            "decomposition_closure": closure_shadow,
        },
    }
    _atomic_write_json(temp_dir / "attempt-manifest.json", attempt_manifest)
    final_dir = attempts_root / attempt_id
    os.replace(temp_dir, final_dir)

    canonical_starless = pipeline.process_dir / "starless.fit"
    canonical_starmask = pipeline.process_dir / "starmask_raw.fit"
    _atomic_copy(final_dir / "starless.fit", canonical_starless)
    _atomic_copy(final_dir / "starmask.fit", canonical_starmask)
    selected_payload = {
        "schema": SYQON_SELECTED_SCHEMA,
        "attempt_id": attempt_id,
        "pair_id": pair_id,
        "generation": "raw",
        "stop_reason": "CONTRACT_VALID_PAIR_COMMITTED",
        "attempt_manifest": str(
            (final_dir / "attempt-manifest.json").relative_to(pipeline.process_dir)
        ),
        "starless": str((final_dir / "starless.fit").relative_to(pipeline.process_dir)),
        "starmask_raw": str((final_dir / "starmask.fit").relative_to(pipeline.process_dir)),
    }
    _atomic_write_json(
        pipeline.process_dir / "stage6_syqon_selected.json",
        selected_payload,
    )
    pipeline._selected_syqon_pair_id = pair_id
    pipeline._selected_syqon_attempt_id = attempt_id
    pipeline.starless_file = canonical_starless
    pipeline.starmask_file = canonical_starmask

    exchange_report.update(
        {
            "status": "accepted",
            "accepted": True,
            "pair_id": pair_id,
            "failure_code": None,
            "stop_reason": "CONTRACT_VALID_PAIR_COMMITTED",
            "files": attempt_manifest["files"],
            "assets": assets,
            "runtime": attempt_manifest["runtime"],
            "worker": worker_manifest,
            "shadow_metrics": attempt_manifest["shadow_metrics"],
            "generations": [
                {
                    "attempt_id": attempt_id,
                    "pair_id": pair_id,
                    "parent_pair_id": attempt_manifest["parent_pair_id"],
                    "generation": "raw",
                    "stop_reason": "CONTRACT_VALID_PAIR_COMMITTED",
                    "attempt_manifest": selected_payload["attempt_manifest"],
                    "starless_sha256": starless_manifest["sha256"],
                    "starmask_sha256": starmask_manifest["sha256"],
                }
            ],
            "selected_pointer": selected_payload,
        }
    )
    _write_syqon_exchange_report(pipeline, exchange_report)
    metrics = dict(exchange_report.get("metrics") or {})
    pipeline.log.info(
        "SyQon 数值交换哨兵通过: "
        f"pair={pair_id[:12]}, "
        f"median_ratio={float(metrics.get('median_ratio') or 0.0):.4f}, "
        f"median_delta={float(metrics.get('median_delta') or 0.0):.4f}"
    )
    pipeline._stage7_prepare_starmask()
    return used


def stage7_try_syqon_with_source(
    pipeline: object,
    syqon_script: Path,
    *,
    source_stem: str,
    attempt_name: str,
    profile: SyQonAttemptProfile = SYQON_BASELINE_PROFILE,
) -> Optional[str]:
    previous_stretched_name = pipeline.stretched_name
    try:
        pipeline.stretched_name = source_stem
        return pipeline._stage7_try_syqon_variant(
            syqon_script,
            attempt_name=attempt_name,
            profile=profile,
        )
    finally:
        pipeline.stretched_name = previous_stretched_name
