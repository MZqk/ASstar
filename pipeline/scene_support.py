"""Immutable Stage 3 scene support shared by Stages 3, 6, and 7."""

from __future__ import annotations

from collections import deque
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional
import zipfile

import numpy as np

try:
    import sep as sep_library
except (ImportError, OSError):  # pragma: no cover - bundled runtime owns SEP
    sep_library = None


SCENE_SUPPORT_SCHEMA = "starun.stage3-scene-support.v1"
SCENE_SUPPORT_JSON = "stage3_scene_support.json"
SCENE_SUPPORT_ARRAYS = "stage3_scene_support.npz"
COORDINATE_DOMAIN = "siril_pixel_buffer_bottom_up"
LIMITATIONS = [
    "same_coordinate_domain_not_same_photometry",
    "not_registration_footprint",
    "not_scientific_photometry",
    "not_stage6_starmask",
]
_SEP_FILTER_KERNEL = np.asarray(
    ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
    dtype=np.float32,
) / 16.0
_DEFAULT_SEP = object()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: np.ndarray) -> str:
    source = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(source.dtype.str.encode("ascii", "strict"))
    digest.update(json.dumps(list(source.shape), separators=(",", ":")).encode("ascii"))
    digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def _channels_first(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim == 2:
        return source[None, :, :]
    if source.ndim != 3:
        raise ValueError(f"scene support expects 2-D or 3-D pixels, got {source.shape}")
    if source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
        return source
    if source.shape[-1] in (1, 3, 4):
        return np.moveaxis(source, -1, 0)
    raise ValueError(f"scene support cannot determine channel axis for {source.shape}")


def _normalized_channels(image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    source = _channels_first(image)
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("scene support pixels are not numeric")
    report: Dict[str, Any] = {
        "source_dtype": str(source.dtype),
        "normalization": "identity",
        "normalization_scale": 1.0,
    }
    if np.issubdtype(source.dtype, np.integer):
        scale = float(np.iinfo(source.dtype).max)
        report.update(
            normalization="integer_dtype_max",
            normalization_scale=scale,
            verified_white_level=float(np.iinfo(source.dtype).max),
        )
        return source.astype(np.float32) / max(scale, 1.0), report
    normalized = source.astype(np.float32, copy=False)
    finite = normalized[np.isfinite(normalized)]
    if finite.size and float(np.min(finite)) >= -1.0e-6 and float(np.max(finite)) <= 1.000001:
        report["verified_white_level"] = 1.0
        report["normalization"] = "verified_normalized_float"
    else:
        report["verified_white_level"] = None
        report["normalization"] = "unverified_float_scale"
    return normalized, report


def _luminance(channels: np.ndarray) -> np.ndarray:
    if channels.shape[0] == 1:
        return np.asarray(channels[0], dtype=np.float32)
    return np.asarray(
        0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2],
        dtype=np.float32,
    )


def _edge_connected(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    connected = np.zeros_like(source, dtype=bool)
    pending: deque[tuple[int, int]] = deque()
    for x in range(width):
        if source[0, x]:
            pending.append((0, x))
        if height > 1 and source[height - 1, x]:
            pending.append((height - 1, x))
    for y in range(1, max(1, height - 1)):
        if source[y, 0]:
            pending.append((y, 0))
        if width > 1 and source[y, width - 1]:
            pending.append((y, width - 1))
    while pending:
        y, x = pending.popleft()
        if connected[y, x] or not source[y, x]:
            continue
        connected[y, x] = True
        if y:
            pending.append((y - 1, x))
        if y + 1 < height:
            pending.append((y + 1, x))
        if x:
            pending.append((y, x - 1))
        if x + 1 < width:
            pending.append((y, x + 1))
    return connected


def build_valid_mask(image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    channels, domain = _normalized_channels(image)
    finite = np.all(np.isfinite(channels), axis=0)
    valid = finite.copy()
    gray = _luminance(channels)
    finite_values = gray[finite]
    report: Dict[str, Any] = {
        "status": "available",
        "method": "all_channels_finite_plus_edge_connected_repeated_floor",
        "finite_fraction": float(np.mean(finite)),
        "pixel_domain": domain,
        "edge_fill_fraction": 0.0,
    }
    if finite_values.size:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
        dynamic = high - low
        if dynamic > max(abs(high), 1.0) * 1.0e-8:
            tolerance = float(
                max(dynamic * 1.0e-7, float(np.finfo(np.float32).eps) * 8.0)
            )
            repeated_floor = finite & (np.abs(gray - low) <= tolerance)
            repeated_fraction = float(np.mean(repeated_floor))
            report.update(
                floor_value=low,
                floor_tolerance=tolerance,
                repeated_floor_fraction=repeated_fraction,
            )
            if repeated_fraction >= 0.002:
                edge_fill = _edge_connected(repeated_floor)
                if not bool(np.all(edge_fill)):
                    valid &= ~edge_fill
                    report["edge_fill_fraction"] = float(np.mean(edge_fill))
        else:
            report["constant_image_preserved"] = True
    if not bool(np.any(valid)):
        report.update(status="unavailable", reason="no_valid_pixels")
    report["valid_fraction"] = float(np.mean(valid))
    return np.ascontiguousarray(valid, dtype=np.uint8), report


def build_saturation_map(image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    source = _channels_first(image)
    channels, domain = _normalized_channels(source)
    height, width = channels.shape[1:]
    saturation = np.zeros((height, width), dtype=np.uint8)
    white_level = domain.get("verified_white_level")
    report: Dict[str, Any] = {
        "status": "available",
        "method": "verified_native_white_level_channel_bitset",
        "pixel_domain": domain,
        "channel_count": int(min(channels.shape[0], 3)),
        "bit_contract": {"mono_or_r": 0, "g": 1, "b": 2},
    }
    if white_level is None:
        report.update(status="unavailable", reason="source_white_level_unverified")
        return saturation, report
    if np.issubdtype(source.dtype, np.integer):
        threshold = float(white_level)
        comparison = source[:3].astype(np.float64) >= threshold
    else:
        threshold = 1.0
        comparison = channels[:3] >= threshold
    for index in range(min(int(comparison.shape[0]), 3)):
        saturation[comparison[index] & np.isfinite(channels[index])] |= np.uint8(1 << index)
    report.update(
        threshold=threshold,
        saturated_fraction=float(np.mean(saturation > 0)),
        saturated_pixel_count=int(np.count_nonzero(saturation)),
    )
    return np.ascontiguousarray(saturation), report


def _aperture_fraction(mask: np.ndarray, x: float, y: float, radius: float) -> float:
    height, width = mask.shape
    r = max(1, int(math.ceil(radius)))
    x0 = max(0, int(math.floor(x)) - r)
    x1 = min(width, int(math.floor(x)) + r + 1)
    y0 = max(0, int(math.floor(y)) - r)
    y1 = min(height, int(math.floor(y)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    aperture = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    if not bool(np.any(aperture)):
        return 0.0
    return float(np.mean(np.asarray(mask[y0:y1, x0:x1])[aperture] > 0))


def extract_sep_objects_from_plane(
    plane: np.ndarray,
    sep_module: Any,
) -> tuple[Any, Any]:
    """Run the one frozen SEP detection profile shared by Stage 3 and Stage 9."""
    prepared = np.ascontiguousarray(np.asarray(plane), dtype=np.float32)
    if prepared.ndim != 2 or prepared.size == 0 or not np.all(np.isfinite(prepared)):
        raise ValueError("SEP plane must be a finite non-empty 2-D array")
    if sep_module is None:
        raise RuntimeError("SEP runtime dependency is unavailable")
    background = sep_module.Background(prepared, bw=64, bh=64, fw=3, fh=3)
    residual = np.ascontiguousarray(prepared - background.back(), dtype=np.float32)
    objects = sep_module.extract(
        residual,
        5.0,
        err=float(background.globalrms),
        minarea=3,
        filter_kernel=_SEP_FILTER_KERNEL,
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        clean_param=1.0,
    )
    return background, objects


def extract_sep_catalog(
    image: np.ndarray,
    valid_mask: np.ndarray,
    saturation_map: np.ndarray,
    *,
    sep_module: Any = _DEFAULT_SEP,
) -> Dict[str, Any]:
    selected_sep = sep_library if sep_module is _DEFAULT_SEP else sep_module
    parameters = {
        "threshold_sigma": 5.0,
        "background_mesh": [64, 64],
        "background_filter": [3, 3],
        "filter_kernel": "gaussian_3x3_1_2_1",
        "minarea": 3,
        "deblend_nthresh": 32,
        "deblend_cont": 0.005,
        "clean": True,
        "clean_param": 1.0,
        "axis_ratio_min": 0.5,
        "fwhm_anchor_ratio": [0.5, 2.2],
        "allowed_flags": ["MERGED"],
        "rejected_flags": ["TRUNC", "DOVERFLOW", "SINGU"],
    }
    base: Dict[str, Any] = {
        "status": "unavailable",
        "detector": "SEP",
        "sep_version": (
            str(getattr(selected_sep, "__version__", "unknown"))
            if selected_sep is not None
            else None
        ),
        "parameters": parameters,
        "membership_frozen": True,
        "coordinate_domain": COORDINATE_DOMAIN,
        "records": [],
        "records_sha256": canonical_json_sha256([]),
    }
    try:
        if selected_sep is None:
            raise RuntimeError("SEP runtime dependency is unavailable")
        channels, _domain = _normalized_channels(image)
        plane = np.ascontiguousarray(_luminance(channels), dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        saturation = np.asarray(saturation_map, dtype=np.uint8)
        if valid.shape != plane.shape or saturation.shape != plane.shape:
            raise ValueError("scene support component shapes do not match")
        finite_valid = valid & np.isfinite(plane)
        if int(np.count_nonzero(finite_valid)) < 256:
            raise ValueError("insufficient valid pixels for SEP")
        prepared = plane.copy()
        prepared[~finite_valid] = float(np.median(prepared[finite_valid]))
        background, objects = extract_sep_objects_from_plane(
            prepared,
            selected_sep,
        )
        rejected_flags = int(
            getattr(selected_sep, "OBJ_TRUNC", 0)
            | getattr(selected_sep, "OBJ_DOVERFLOW", 0)
            | getattr(selected_sep, "OBJ_SINGU", 0)
        )
        provisional = []
        rejection_counts = {
            "invalid_numeric": 0,
            "rejected_flag": 0,
            "axis_ratio": 0,
            "invalid_region": 0,
            "fwhm_scale": 0,
        }
        for source_index, obj in enumerate(objects):
            values = {
                "x": float(obj["x"]),
                "y": float(obj["y"]),
                "flux": float(obj["flux"]),
                "peak": float(obj["peak"]),
                "a": float(obj["a"]),
                "b": float(obj["b"]),
                "theta": float(obj["theta"]),
                "npix": int(obj["npix"]),
                "flag": int(obj["flag"]),
            }
            numeric = tuple(values[name] for name in ("x", "y", "flux", "peak", "a", "b", "theta"))
            if not all(math.isfinite(value) for value in numeric) or values["a"] <= 0 or values["b"] <= 0:
                rejection_counts["invalid_numeric"] += 1
                continue
            if values["flag"] & rejected_flags:
                rejection_counts["rejected_flag"] += 1
                continue
            axis_ratio = min(values["a"], values["b"]) / max(values["a"], values["b"])
            if axis_ratio < 0.5:
                rejection_counts["axis_ratio"] += 1
                continue
            fwhm = 2.354820045 * math.sqrt(values["a"] * values["b"])
            valid_fraction = _aperture_fraction(valid, values["x"], values["y"], max(2.0, fwhm))
            if valid_fraction < 0.5:
                rejection_counts["invalid_region"] += 1
                continue
            provisional.append(
                {
                    **values,
                    "source_index": source_index,
                    "fwhm_px": fwhm,
                    "axis_ratio": axis_ratio,
                    "valid_fraction": valid_fraction,
                    "saturation_fraction": _aperture_fraction(
                        saturation,
                        values["x"],
                        values["y"],
                        max(2.0, fwhm),
                    ),
                }
            )
        fwhm_values = np.asarray([row["fwhm_px"] for row in provisional], dtype=np.float64)
        anchor = None
        if fwhm_values.size:
            median = float(np.median(fwhm_values))
            mad = float(np.median(np.abs(fwhm_values - median)))
            if mad > 0:
                clipped = fwhm_values[np.abs(fwhm_values - median) <= 4.0 * 1.4826 * mad]
                if clipped.size:
                    median = float(np.median(clipped))
            anchor = median if math.isfinite(median) and median > 0 else None
        accepted = []
        for row in provisional:
            if anchor is not None and not 0.5 * anchor <= row["fwhm_px"] <= 2.2 * anchor:
                rejection_counts["fwhm_scale"] += 1
                continue
            row["saturated"] = bool(row["saturation_fraction"] > 0.0)
            accepted.append(row)
        accepted.sort(key=lambda row: (row["y"], row["x"], -row["flux"], row["source_index"]))
        records = []
        for index, row in enumerate(accepted, start=1):
            record = {"id": f"S3{index:06d}"}
            record.update({key: value for key, value in row.items() if key != "source_index"})
            records.append(record)
        base.update(
            status="available",
            detected_count=int(len(objects)),
            valid_count=len(records),
            rejected_count=int(len(objects) - len(records)),
            rejection_counts=rejection_counts,
            frozen_fwhm_px=anchor,
            background={
                "global_mean": float(background.globalback),
                "global_rms": float(background.globalrms),
            },
            records=records,
            records_sha256=canonical_json_sha256(records),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, FloatingPointError) as error:
        base.update(reason=str(error), reason_code="scene_support_sep_unavailable")
    return base


def _array_descriptor(array: np.ndarray) -> Dict[str, Any]:
    canonical = np.ascontiguousarray(array)
    return {
        "shape": [int(value) for value in canonical.shape],
        "dtype": str(canonical.dtype),
        "order": "C",
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _validate_catalog_records(catalog: Mapping[str, Any], shape: tuple[int, int]) -> None:
    records = catalog.get("records")
    if not isinstance(records, list):
        raise ValueError("scene support catalog records are invalid")
    claimed_hash = catalog.get("records_sha256")
    if not _valid_sha256(claimed_hash) or claimed_hash != canonical_json_sha256(records):
        raise ValueError("scene support catalog hash mismatch")
    height, width = shape
    sort_keys = []
    required_numeric = (
        "x",
        "y",
        "flux",
        "peak",
        "a",
        "b",
        "theta",
        "fwhm_px",
        "axis_ratio",
        "saturation_fraction",
        "valid_fraction",
    )
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping) or record.get("id") != f"S3{index:06d}":
            raise ValueError("scene support catalog stable ID contract mismatch")
        try:
            numeric = {name: float(record[name]) for name in required_numeric}
            npix = record["npix"]
            flag = record["flag"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("scene support catalog record is malformed") from error
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("scene support catalog record is nonfinite")
        if not (
            0.0 <= numeric["x"] < width
            and 0.0 <= numeric["y"] < height
            and numeric["a"] > 0.0
            and numeric["b"] > 0.0
            and numeric["fwhm_px"] > 0.0
            and 0.5 <= numeric["axis_ratio"] <= 1.0
            and 0.0 <= numeric["saturation_fraction"] <= 1.0
            and 0.0 <= numeric["valid_fraction"] <= 1.0
        ):
            raise ValueError("scene support catalog record range is invalid")
        if (
            isinstance(npix, bool)
            or not isinstance(npix, int)
            or isinstance(flag, bool)
            or not isinstance(flag, int)
            or not isinstance(record.get("saturated"), bool)
        ):
            raise ValueError("scene support catalog record type is invalid")
        sort_keys.append((numeric["y"], numeric["x"], -numeric["flux"]))
    if sort_keys != sorted(sort_keys):
        raise ValueError("scene support catalog stable ordering mismatch")


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, mode="w") as archive:
            for name in sorted(arrays):
                buffer = BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.ascontiguousarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def unavailable_scene_support(reason: str, *, reason_code: str) -> Dict[str, Any]:
    return {
        "schema": SCENE_SUPPORT_SCHEMA,
        "status": "unavailable",
        "reason_code": reason_code,
        "reason": str(reason),
        "source": {"role": "stage3_bg_input"},
        "components": {},
        "limitations": list(LIMITATIONS),
    }


def write_unavailable_scene_support(
    process_dir: Path,
    reason: str,
    *,
    reason_code: str,
) -> Dict[str, Any]:
    manifest = unavailable_scene_support(reason, reason_code=reason_code)
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    _write_json(Path(process_dir) / SCENE_SUPPORT_JSON, manifest)
    return manifest


def build_scene_support(
    image: np.ndarray,
    process_dir: Path,
    *,
    source_path: Optional[Path] = None,
    sep_module: Any = _DEFAULT_SEP,
) -> Dict[str, Any]:
    source = np.asarray(image)
    source_file_sha = (
        _file_sha256(source_path) if source_path is not None and source_path.is_file() else None
    )
    component_errors: Dict[str, str] = {}
    try:
        valid_mask, valid_report = build_valid_mask(source)
    except (RuntimeError, TypeError, ValueError, FloatingPointError) as error:
        shape = _channels_first(source).shape[1:]
        valid_mask = np.zeros(shape, dtype=np.uint8)
        valid_report = {"status": "unavailable", "reason": str(error)}
        component_errors["valid_mask"] = str(error)
    try:
        saturation_map, saturation_report = build_saturation_map(source)
    except (RuntimeError, TypeError, ValueError, FloatingPointError) as error:
        saturation_map = np.zeros(valid_mask.shape, dtype=np.uint8)
        saturation_report = {"status": "unavailable", "reason": str(error)}
        component_errors["saturation_map"] = str(error)
    catalog = extract_sep_catalog(
        source,
        valid_mask,
        saturation_map,
        sep_module=sep_module,
    )
    if catalog.get("status") != "available":
        component_errors["star_catalog"] = str(catalog.get("reason") or "unavailable")
    arrays_path = Path(process_dir) / SCENE_SUPPORT_ARRAYS
    _write_deterministic_npz(
        arrays_path,
        {"saturation_map": saturation_map, "valid_mask": valid_mask},
    )
    components = {
        "valid_mask": {
            **valid_report,
            **_array_descriptor(valid_mask),
            "allowed_values": [0, 1],
        },
        "saturation_map": {
            **saturation_report,
            **_array_descriptor(saturation_map),
            "allowed_values": list(range(8)),
        },
        "star_catalog": catalog,
    }
    available_count = sum(
        str(component.get("status") or "") == "available"
        for component in components.values()
    )
    status = "available" if available_count == 3 else "partial" if available_count else "unavailable"
    manifest: Dict[str, Any] = {
        "schema": SCENE_SUPPORT_SCHEMA,
        "status": status,
        "source": {
            "role": "stage3_bg_input",
            "artifact": source_path.name if source_path is not None else "stage3_bg_input.fit",
            "file_sha256": source_file_sha,
            "pixel_sha256": pixel_sha256(source),
            "shape": [int(value) for value in source.shape],
            "dtype": str(source.dtype),
            "coordinate_domain": COORDINATE_DOMAIN,
        },
        "arrays": {
            "artifact": SCENE_SUPPORT_ARRAYS,
            "file_sha256": _file_sha256(arrays_path),
        },
        "components": components,
        "component_errors": component_errors,
        "limitations": list(LIMITATIONS),
    }
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    _write_json(Path(process_dir) / SCENE_SUPPORT_JSON, manifest)
    return manifest


def load_scene_support(
    process_dir: Path,
    *,
    expected_shape: Optional[tuple[int, ...]] = None,
) -> Dict[str, Any]:
    manifest_path = Path(process_dir) / SCENE_SUPPORT_JSON
    arrays_path = Path(process_dir) / SCENE_SUPPORT_ARRAYS
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != SCENE_SUPPORT_SCHEMA:
            raise ValueError("scene support schema is invalid")
        stored_payload_hash = manifest.get("manifest_payload_sha256")
        payload_without_hash = dict(manifest)
        payload_without_hash.pop("manifest_payload_sha256", None)
        if not _valid_sha256(stored_payload_hash) or canonical_json_sha256(payload_without_hash) != stored_payload_hash:
            raise ValueError("scene support manifest payload hash mismatch")
        source_shape = tuple(int(value) for value in (manifest.get("source") or {}).get("shape") or [])
        source = manifest.get("source") or {}
        if source.get("role") != "stage3_bg_input":
            raise ValueError("scene support source role is invalid")
        if source.get("artifact") != "stage3_bg_input.fit":
            raise ValueError("scene support source artifact is invalid")
        if source.get("coordinate_domain") != COORDINATE_DOMAIN:
            raise ValueError("scene support coordinate domain mismatch")
        if not _valid_sha256(source.get("file_sha256")):
            raise ValueError("scene support source file hash is invalid")
        local_source_path = Path(process_dir) / "stage3_bg_input.fit"
        if local_source_path.is_file() and _file_sha256(local_source_path) != source.get(
            "file_sha256"
        ):
            raise ValueError("scene support source file hash mismatch")
        if not _valid_sha256(source.get("pixel_sha256")):
            raise ValueError("scene support source pixel hash is invalid")
        try:
            np.dtype(str(source.get("dtype") or ""))
        except TypeError as error:
            raise ValueError("scene support source dtype is invalid") from error
        if len(source_shape) == 2:
            expected_spatial_shape = source_shape
        elif len(source_shape) == 3 and source_shape[0] in (1, 3, 4):
            expected_spatial_shape = source_shape[1:]
        elif len(source_shape) == 3 and source_shape[-1] in (1, 3, 4):
            expected_spatial_shape = source_shape[:2]
        else:
            raise ValueError("scene support source shape is invalid")
        if expected_shape is not None and source_shape != tuple(int(value) for value in expected_shape):
            raise ValueError("scene support source shape mismatch")
        arrays = manifest.get("arrays") or {}
        if arrays.get("artifact") != SCENE_SUPPORT_ARRAYS:
            raise ValueError("scene support arrays artifact is invalid")
        if not _valid_sha256(arrays.get("file_sha256")):
            raise ValueError("scene support arrays file hash is invalid")
        if not arrays_path.is_file() or _file_sha256(arrays_path) != arrays.get("file_sha256"):
            raise ValueError("scene support arrays file hash mismatch")
        with np.load(arrays_path, allow_pickle=False) as archive:
            valid_mask = np.ascontiguousarray(archive["valid_mask"])
            saturation_map = np.ascontiguousarray(archive["saturation_map"])
        for name, array in (("valid_mask", valid_mask), ("saturation_map", saturation_map)):
            descriptor = (manifest.get("components") or {}).get(name) or {}
            expected_allowed = [0, 1] if name == "valid_mask" else list(range(8))
            if descriptor.get("allowed_values") != expected_allowed:
                raise ValueError(f"scene support {name} allowed values mismatch")
            if descriptor.get("shape") != [int(value) for value in array.shape]:
                raise ValueError(f"scene support {name} shape mismatch")
            if tuple(array.shape) != tuple(expected_spatial_shape):
                raise ValueError(f"scene support {name} source shape mismatch")
            if descriptor.get("dtype") != str(array.dtype) or array.dtype != np.uint8:
                raise ValueError(f"scene support {name} dtype mismatch")
            if descriptor.get("order") != "C" or not array.flags.c_contiguous:
                raise ValueError(f"scene support {name} order mismatch")
            if hashlib.sha256(array.tobytes(order="C")).hexdigest() != descriptor.get("sha256"):
                raise ValueError(f"scene support {name} byte hash mismatch")
        if not bool(np.all((valid_mask == 0) | (valid_mask == 1))):
            raise ValueError("scene support valid_mask value range is invalid")
        if not bool(np.all(saturation_map <= 7)):
            raise ValueError("scene support saturation_map value range is invalid")
        components = manifest.get("components") or {}
        catalog = components.get("star_catalog") or {}
        _validate_catalog_records(catalog, tuple(expected_spatial_shape))
        component_statuses = [
            str((components.get(name) or {}).get("status") or "")
            for name in ("valid_mask", "saturation_map", "star_catalog")
        ]
        if any(status not in {"available", "unavailable"} for status in component_statuses):
            raise ValueError("scene support component status is invalid")
        available_count = component_statuses.count("available")
        expected_status = (
            "available"
            if available_count == 3
            else "partial"
            if available_count
            else "unavailable"
        )
        if manifest.get("status") != expected_status:
            raise ValueError("scene support root status is inconsistent")
        valid_mask.flags.writeable = False
        saturation_map.flags.writeable = False
        return {
            "status": str(manifest.get("status") or "unavailable"),
            "manifest": manifest,
            "valid_mask": valid_mask,
            "saturation_map": saturation_map,
            "manifest_path": manifest_path,
            "arrays_path": arrays_path,
        }
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        return {
            "status": "unavailable",
            "manifest": unavailable_scene_support(
                str(error), reason_code="scene_support_validation_failed"
            ),
            "valid_mask": None,
            "saturation_map": None,
            "manifest_path": manifest_path,
            "arrays_path": arrays_path,
        }


def scene_support_summary(value: Any) -> Dict[str, Any]:
    manifest = value.get("manifest") if isinstance(value, dict) and "manifest" in value else value
    if not isinstance(manifest, Mapping):
        manifest = unavailable_scene_support(
            "scene support is not available", reason_code="scene_support_unavailable"
        )
    components = manifest.get("components") if isinstance(manifest.get("components"), Mapping) else {}
    manifest_path = value.get("manifest_path") if isinstance(value, dict) else None
    manifest_file_sha256 = (
        _file_sha256(Path(manifest_path))
        if manifest_path is not None and Path(manifest_path).is_file()
        else None
    )
    return {
        "schema": SCENE_SUPPORT_SCHEMA,
        "status": str(manifest.get("status") or "unavailable"),
        "source_file_sha256": (manifest.get("source") or {}).get("file_sha256"),
        "source_pixel_sha256": (manifest.get("source") or {}).get("pixel_sha256"),
        "coordinate_domain": (manifest.get("source") or {}).get("coordinate_domain"),
        "manifest_artifact": SCENE_SUPPORT_JSON,
        "manifest_sha256": manifest_file_sha256,
        "arrays_artifact": SCENE_SUPPORT_ARRAYS,
        "arrays_sha256": (manifest.get("arrays") or {}).get("file_sha256"),
        "component_status": {
            name: str((components.get(name) or {}).get("status") or "unavailable")
            for name in ("star_catalog", "saturation_map", "valid_mask")
        },
        "star_count": int((components.get("star_catalog") or {}).get("valid_count", 0) or 0),
        "reason_code": manifest.get("reason_code"),
    }


def catalog_aperture_mask(
    value: Any,
    shape: tuple[int, int],
    *,
    radius_scale: float = 2.5,
) -> Optional[np.ndarray]:
    manifest = value.get("manifest") if isinstance(value, dict) and "manifest" in value else value
    if not isinstance(manifest, Mapping):
        return None
    catalog = ((manifest.get("components") or {}).get("star_catalog") or {})
    if catalog.get("status") != "available":
        return None
    height, width = (int(shape[0]), int(shape[1]))
    mask = np.zeros((height, width), dtype=np.float32)
    accepted = 0
    for record in catalog.get("records") or []:
        if not isinstance(record, Mapping):
            continue
        try:
            x = float(record.get("x"))
            y = float(record.get("y"))
            fwhm = float(record.get("fwhm_px"))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x, y, fwhm)) or fwhm <= 0:
            continue
        radius = max(3.0, min(32.0, float(radius_scale) * fwhm))
        extent = int(math.ceil(radius))
        x0 = max(0, int(math.floor(x)) - extent)
        x1 = min(width, int(math.floor(x)) + extent + 1)
        y0 = max(0, int(math.floor(y)) - extent)
        y1 = min(height, int(math.floor(y)) + extent + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = 1.0
        accepted += 1
    if not accepted:
        return None
    mask.flags.writeable = False
    return mask


__all__ = [
    "COORDINATE_DOMAIN",
    "SCENE_SUPPORT_ARRAYS",
    "SCENE_SUPPORT_JSON",
    "SCENE_SUPPORT_SCHEMA",
    "build_scene_support",
    "build_saturation_map",
    "build_valid_mask",
    "canonical_json_sha256",
    "catalog_aperture_mask",
    "extract_sep_catalog",
    "extract_sep_objects_from_plane",
    "load_scene_support",
    "pixel_sha256",
    "scene_support_summary",
    "unavailable_scene_support",
    "write_unavailable_scene_support",
]
