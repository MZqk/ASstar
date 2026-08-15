"""Deterministic user-input discovery for files, Light trees, and tasks.

No stage is inferred from a filename.  External masters always enter through
Stage 1; only a signed product task may expose a formal resume boundary.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


INPUT_DISCOVERY_SCHEMA = "starun.input-discovery.v1"
TASK_MANIFEST_NAME = "task-manifest.json"
MASTER_SUFFIXES = frozenset({".fit", ".fits", ".fts", ".xisf"})
FITS_LIGHT_SUFFIXES = frozenset({".fit", ".fits", ".fts"})
REVIEW_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})

_LIGHT_DIRECTORY_NAMES = frozenset({"light", "lights"})
_CALIBRATION_NAMES = frozenset(
    {
        "bias",
        "biases",
        "dark",
        "darks",
        "darkflat",
        "darkflats",
        "dark-flat",
        "dark-flats",
        "flat",
        "flats",
    }
)
_MANAGED_DIRECTORY_NAMES = frozenset(
    {
        "checkpoints",
        "process",
        "results",
        "review_bundles",
        "runs",
        "starun",
    }
)
_RETIRED_PROCESSING_MARKERS = (
    "pipeline-result.json",
    "processing-plan.json",
    "result_linear.fit",
    "stage2_corrected.fit",
)


class InputKind(str, Enum):
    MASTER_FILE = "master_file"
    LIGHT_DIRECTORY = "light_directory"
    PRODUCT_TASK = "product_task"
    REVIEW_FILE = "review_file"
    UNSUPPORTED = "unsupported"


class DiscoveryTrust(str, Enum):
    VERIFIED = "verified"
    RECOGNIZED = "recognized"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class LightGroup:
    """One independently stackable set of Light frames."""

    key: str
    target: str
    filter_name: str
    camera: str
    geometry: str
    files: Tuple[Path, ...]
    total_bytes: int

    @property
    def display_label(self) -> str:
        components = [self.target, self.filter_name, self.camera, self.geometry]
        return " · ".join(value for value in components if value != "unknown")

    def to_dict(self, *, source_root: Optional[Path] = None) -> Dict[str, Any]:
        paths = []
        for path in self.files:
            if source_root is None:
                paths.append(str(path))
                continue
            try:
                paths.append(path.relative_to(source_root).as_posix())
            except ValueError:
                paths.append(str(path))
        return {
            "key": self.key,
            "target": self.target,
            "filter": self.filter_name,
            "camera": self.camera,
            "geometry": self.geometry,
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
            "files": paths,
        }


@dataclass(frozen=True)
class InputDiscovery:
    """A complete, UI-ready intake conclusion."""

    selected_path: Path
    kind: InputKind
    trust: DiscoveryTrust
    summary: str
    source_root: Optional[Path] = None
    master_file: Optional[Path] = None
    light_groups: Tuple[LightGroup, ...] = ()
    task_directory: Optional[Path] = None
    resume_after_stage: Optional[int] = None
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        regular_input = self.kind in {
            InputKind.MASTER_FILE,
            InputKind.LIGHT_DIRECTORY,
            InputKind.PRODUCT_TASK,
            InputKind.REVIEW_FILE,
        }
        return regular_input and not self.errors

    @property
    def creates_independent_task(self) -> bool:
        return self.kind in {
            InputKind.MASTER_FILE,
            InputKind.LIGHT_DIRECTORY,
            InputKind.REVIEW_FILE,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": INPUT_DISCOVERY_SCHEMA,
            "selected_path": str(self.selected_path),
            "kind": self.kind.value,
            "trust": self.trust.value,
            "summary": self.summary,
            "source_root": str(self.source_root) if self.source_root else None,
            "master_file": str(self.master_file) if self.master_file else None,
            "light_groups": [
                group.to_dict(source_root=self.source_root)
                for group in self.light_groups
            ],
            "task_directory": (
                str(self.task_directory) if self.task_directory else None
            ),
            "resume_after_stage": self.resume_after_stage,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SourceHeaderSummary:
    """Small, UI-safe summary of one selected source file's primary header."""

    source_path: Path
    status: str
    device_name: str = ""
    filter_name: str = ""
    exposure: str = ""
    details: Tuple[Tuple[str, str], ...] = ()
    message: str = ""
    header_field_count: int = 0


def _normalized_name(value: Any, fallback: str = "unknown") -> str:
    text = " ".join(str(value or "").replace("\x00", "").split()).strip()
    return text or fallback


def _parse_fits_value(raw: str) -> Any:
    value = raw.strip()
    if value.startswith("'"):
        chars = []
        index = 1
        while index < len(value):
            if value[index] == "'":
                if index + 1 < len(value) and value[index + 1] == "'":
                    chars.append("'")
                    index += 2
                    continue
                break
            chars.append(value[index])
            index += 1
        return "".join(chars).strip()
    value = value.split("/", 1)[0].strip()
    if value in {"T", "F"}:
        return value == "T"
    try:
        return float(value.replace("D", "E")) if any(
            token in value.upper() for token in (".", "E", "D")
        ) else int(value)
    except ValueError:
        return value


def read_fits_group_metadata(path: Path) -> Dict[str, Any]:
    """Read bounded primary FITS cards without loading image pixels."""

    metadata: Dict[str, Any] = {}
    try:
        with path.open("rb") as handle:
            for _ in range(128):
                block = handle.read(2880)
                if not block:
                    break
                for offset in range(0, len(block), 80):
                    card = block[offset : offset + 80]
                    if len(card) < 80:
                        break
                    text = card.decode("ascii", errors="replace")
                    keyword = text[:8].strip().upper()
                    if keyword == "END":
                        return metadata
                    if text[8:10] != "= " or not keyword:
                        continue
                    metadata[keyword] = _parse_fits_value(text[10:])
    except OSError:
        return {}
    return metadata


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is not None and (
            not isinstance(value, str) or _normalized_name(value, "")
        ):
            return value
    return None


def _metadata_names(metadata: Mapping[str, Any], *keys: str) -> Tuple[str, ...]:
    values = []
    seen = set()
    for key in keys:
        value = _normalized_name(metadata.get(key), "")
        identity = value.casefold()
        if not value or identity in seen:
            continue
        seen.add(identity)
        values.append(value)
    return tuple(values)


def _finite_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace("D", "E"))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number_text(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return _normalized_name(value, "")
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def _measurement_text(value: Any, unit: str) -> str:
    text = _number_text(value)
    if not text:
        return ""
    return f"{text} {unit}" if _finite_number(value) is not None else text


def _positive_integer(value: Any) -> Optional[int]:
    number = _finite_number(value)
    if number is None or number <= 0 or not number.is_integer():
        return None
    return int(number)


def _fits_data_format(value: Any) -> str:
    bitpix = _finite_number(value)
    if bitpix is None or not bitpix.is_integer():
        return _normalized_name(value, "")
    formats = {
        8: "8-bit 整数",
        16: "16-bit 整数",
        32: "32-bit 整数",
        64: "64-bit 整数",
        -32: "32-bit 浮点",
        -64: "64-bit 浮点",
    }
    integer = int(bitpix)
    return formats.get(integer, f"BITPIX={integer}")


def inspect_source_header(path: Path) -> SourceHeaderSummary:
    """Summarize a selected FITS primary header for non-blocking UI display."""

    source_path = path.expanduser()
    suffix = source_path.suffix.lower()
    if suffix not in FITS_LIGHT_SUFFIXES:
        kind = "XISF" if suffix == ".xisf" else "非 FITS"
        return SourceHeaderSummary(
            source_path=source_path,
            status="unsupported",
            message=(
                f"当前仅扫描 FITS 主 Header；{kind} 元数据未读取，"
                "不影响后续处理。"
            ),
        )
    if not source_path.is_file():
        return SourceHeaderSummary(
            source_path=source_path,
            status="unavailable",
            message="源文件不可读，未扫描 Header；不影响重新选择输入。",
        )

    metadata = read_fits_group_metadata(source_path)
    if not metadata or "SIMPLE" not in metadata:
        return SourceHeaderSummary(
            source_path=source_path,
            status="unavailable",
            message="未找到可识别的 FITS 主 Header；可继续处理并在 Stage 1 复核。",
        )

    devices = _metadata_names(
        metadata,
        "TELESCOP",
        "INSTRUME",
        "CAMERA",
        "DETECTOR",
    )
    filters = _metadata_names(metadata, "FILTER", "FILTER1", "FILTER2")
    exposure_value = _metadata_value(metadata, "EXPTIME", "EXPOSURE", "EXP_TIME")
    stack_count = _positive_integer(
        _metadata_value(metadata, "STACKCNT", "NCOMBINE", "NFRAMES", "FRAMES")
    )
    exposure = _measurement_text(exposure_value, "秒")
    if exposure and stack_count and stack_count > 1 and _finite_number(
        exposure_value
    ) is not None:
        exposure += "/帧"

    details = []

    def add_detail(label: str, value: Any) -> None:
        text = _normalized_name(value, "")
        if text:
            details.append((label, text))

    add_detail(
        "目标",
        _metadata_value(metadata, "OBJECT", "OBJNAME", "TARGET"),
    )
    add_detail(
        "拍摄时间",
        _metadata_value(metadata, "DATE-OBS", "DATEOBS", "DATE"),
    )
    if stack_count is not None:
        add_detail("叠加帧数", stack_count)

    width = _positive_integer(metadata.get("NAXIS1"))
    height = _positive_integer(metadata.get("NAXIS2"))
    channels = _positive_integer(metadata.get("NAXIS3"))
    if width is not None and height is not None:
        dimensions = f"{width} × {height}"
        if channels is not None and channels > 1:
            dimensions += f" × {channels}"
        add_detail("图像尺寸", dimensions)

    add_detail("数据格式", _fits_data_format(metadata.get("BITPIX")))
    add_detail(
        "图像类型",
        _metadata_value(metadata, "IMAGETYP", "FRAMETYP", "FRAME"),
    )

    xbin = _positive_integer(
        _metadata_value(metadata, "XBINNING", "XBIN", "BINNING")
    )
    ybin = _positive_integer(_metadata_value(metadata, "YBINNING", "YBIN"))
    if xbin is not None:
        add_detail("Binning", f"{xbin} × {ybin or xbin}")

    gain = _metadata_value(metadata, "GAIN", "EGAIN", "CCDGAIN")
    if gain is not None:
        add_detail("增益", _number_text(gain))

    temperature = _metadata_value(
        metadata,
        "CCD-TEMP",
        "CCDTEMP",
        "SENSORT",
        "SENSOR_T",
    )
    if temperature is not None:
        add_detail("传感器温度", _measurement_text(temperature, "℃"))

    focal_length = _metadata_value(
        metadata,
        "FOCALLEN",
        "FOCAL",
        "FOCLEN",
    )
    if focal_length is not None:
        add_detail("焦距", _measurement_text(focal_length, "mm"))

    x_pixel = _metadata_value(metadata, "XPIXSZ", "PIXSIZE")
    y_pixel = _metadata_value(metadata, "YPIXSZ")
    if x_pixel is not None:
        x_pixel_text = _number_text(x_pixel)
        y_pixel_text = _number_text(y_pixel)
        pixel_size = x_pixel_text
        if y_pixel_text and y_pixel_text != x_pixel_text:
            pixel_size += f" × {y_pixel_text}"
        add_detail("像元尺寸", f"{pixel_size} μm")

    ra = _metadata_value(metadata, "OBJCTRA", "RA", "CRVAL1")
    dec = _metadata_value(metadata, "OBJCTDEC", "DEC", "CRVAL2")
    coordinate_parts = []
    if ra is not None:
        coordinate_parts.append(f"RA {_normalized_name(ra, '')}")
    if dec is not None:
        coordinate_parts.append(f"Dec {_normalized_name(dec, '')}")
    if coordinate_parts:
        add_detail("中心坐标", " · ".join(coordinate_parts))

    add_detail(
        "Bayer 阵列",
        _metadata_value(metadata, "BAYERPAT", "BAYERPATTERN"),
    )
    add_detail(
        "创建软件",
        _metadata_value(metadata, "CREATOR", "SWCREATE", "PROGRAM"),
    )

    return SourceHeaderSummary(
        source_path=source_path,
        status="ok",
        device_name=" · ".join(devices),
        filter_name=" · ".join(filters),
        exposure=exposure,
        details=tuple(details),
        message=(
            f"已读取 {len(metadata)} 个 FITS 主 Header 字段；未读取图像像素。"
        ),
        header_field_count=len(metadata),
    )


def _relative_parts(path: Path, source_root: Path) -> Tuple[str, ...]:
    try:
        return path.relative_to(source_root).parts
    except ValueError:
        return path.parts


def _has_named_part(parts: Iterable[str], names: frozenset[str]) -> bool:
    return any(str(part).strip().lower() in names for part in parts)


def _is_light_candidate(path: Path, source_root: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in FITS_LIGHT_SUFFIXES:
        return False
    parts = _relative_parts(path, source_root)
    if any(str(part).startswith(".") for part in parts):
        return False
    if _has_named_part(parts[:-1], _CALIBRATION_NAMES):
        return False
    if _has_named_part(parts[:-1], _MANAGED_DIRECTORY_NAMES):
        return False
    stem = path.stem.strip().lower()
    if any(
        stem.startswith(prefix)
        for prefix in ("bias_", "dark_", "darkflat_", "dark-flat_", "flat_")
    ):
        return False
    return stem.startswith("light_") or _has_named_part(
        parts[:-1],
        _LIGHT_DIRECTORY_NAMES,
    )


def _path_target_hint(path: Path, source_root: Path) -> str:
    parts = list(_relative_parts(path.parent, source_root))
    for index, part in enumerate(parts):
        if part.strip().lower() in _LIGHT_DIRECTORY_NAMES and index > 0:
            return _normalized_name(parts[index - 1])
    if source_root.name.strip().lower() not in _LIGHT_DIRECTORY_NAMES:
        return _normalized_name(source_root.name)
    return _normalized_name(source_root.parent.name)


def _filename_filter_hint(path: Path) -> str:
    compact = re.sub(r"[^a-z0-9]+", " ", path.stem.lower())
    patterns = (
        (r"\b(?:lp|light pollution)\b", "Seestar LP"),
        (r"\b(?:ha ?oiii|dual ?narrowband|duoband)\b", "dual narrowband"),
        (r"\b(?:no ?filter|clear)\b", "no filter"),
    )
    for pattern, label in patterns:
        if re.search(pattern, compact):
            return label
    return "unknown"


def _first_metadata_value(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _normalized_name(metadata.get(key), "")
        if value:
            return value
    return "unknown"


def _group_identity(path: Path, source_root: Path) -> Tuple[str, str, str, str]:
    metadata = read_fits_group_metadata(path)
    target = _first_metadata_value(metadata, "OBJECT", "OBJNAME", "TARGET")
    if target == "unknown":
        target = _path_target_hint(path, source_root)
    filter_name = _first_metadata_value(metadata, "FILTER", "FILTER1", "FILTER2")
    if filter_name == "unknown":
        filter_name = _filename_filter_hint(path)
    camera = _first_metadata_value(metadata, "INSTRUME", "CAMERA", "DETECTOR")
    width = metadata.get("NAXIS1")
    height = metadata.get("NAXIS2")
    xbin = metadata.get("XBINNING", metadata.get("XBIN", 1))
    ybin = metadata.get("YBINNING", metadata.get("YBIN", xbin))
    try:
        geometry = f"{int(width)}x{int(height)}@{int(xbin)}x{int(ybin)}"
    except (TypeError, ValueError):
        geometry = "unknown"
    return tuple(
        _normalized_name(item) for item in (target, filter_name, camera, geometry)
    )  # type: ignore[return-value]


def discover_light_groups(
    source_root: Path,
    *,
    max_files: int = 10000,
) -> Tuple[LightGroup, ...]:
    """Recursively find Lights and group them by physical compatibility."""

    root = source_root.expanduser().resolve()
    if not root.is_dir():
        return ()
    candidates = []
    for path in root.rglob("*"):
        if _is_light_candidate(path, root):
            candidates.append(path.resolve())
            if len(candidates) > max_files:
                raise ValueError(
                    f"Light 文件超过安全扫描上限 {max_files}，请缩小输入目录"
                )

    grouped: Dict[Tuple[str, str, str, str], list[Path]] = {}
    for path in sorted(candidates, key=lambda item: item.as_posix().lower()):
        grouped.setdefault(_group_identity(path, root), []).append(path)

    groups = []
    for identity, files in sorted(grouped.items(), key=lambda item: item[0]):
        canonical = "\x1f".join(part.casefold() for part in identity)
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        total_bytes = 0
        for path in files:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
        groups.append(
            LightGroup(
                key=key,
                target=identity[0],
                filter_name=identity[1],
                camera=identity[2],
                geometry=identity[3],
                files=tuple(files),
                total_bytes=total_bytes,
            )
        )
    return tuple(groups)


def _looks_like_retired_processing_directory(path: Path) -> bool:
    return any(
        (path / name).exists() for name in _RETIRED_PROCESSING_MARKERS
    ) or any(
        candidate.is_file() for candidate in path.glob("stage[0-9]*_*.fit")
    )


def discover_input(
    selected_path: Path,
    *,
    current_resume_fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> InputDiscovery:
    """Classify one explicit drag/drop selection without heuristic stage guesses."""

    expanded = selected_path.expanduser()
    try:
        path = expanded.resolve()
    except OSError:
        path = expanded
    if not path.exists():
        return InputDiscovery(
            selected_path=path,
            kind=InputKind.UNSUPPORTED,
            trust=DiscoveryTrust.REVIEW_REQUIRED,
            summary="输入不存在",
            errors=(f"路径不存在：{path}",),
        )

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in MASTER_SUFFIXES:
            return InputDiscovery(
                selected_path=path,
                source_root=path.parent,
                kind=InputKind.MASTER_FILE,
                trust=DiscoveryTrust.RECOGNIZED,
                summary=f"已识别母版 {path.name}，将从 Stage 1 导入并校验",
                master_file=path,
                resume_after_stage=None,
            )
        if suffix in REVIEW_SUFFIXES:
            return InputDiscovery(
                selected_path=path,
                source_root=path.parent,
                kind=InputKind.REVIEW_FILE,
                trust=DiscoveryTrust.REVIEW_REQUIRED,
                summary=f"已识别预览图 {path.name}，仅生成复核任务",
                master_file=path,
                warnings=("TIFF/PNG/JPEG 不能作为线性处理断点。",),
            )
        return InputDiscovery(
            selected_path=path,
            kind=InputKind.UNSUPPORTED,
            trust=DiscoveryTrust.REVIEW_REQUIRED,
            summary="不支持的文件类型",
            errors=(f"不支持 {suffix or '无扩展名'} 文件",),
        )

    task_manifest = path / TASK_MANIFEST_NAME
    if task_manifest.is_file():
        try:
            try:
                from .task_workspace import inspect_task_workspace
            except ImportError:
                from task_workspace import inspect_task_workspace

            inspection = inspect_task_workspace(
                path,
                current_resume_fingerprints=current_resume_fingerprints,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            inspection = {
                "recognized": False,
                "verified": False,
                "detail": f"任务清单校验失败：{error}",
            }
        recognized = bool(inspection.get("recognized"))
        verified = bool(inspection.get("verified"))
        resume_stage = inspection.get("resume_after_stage") if verified else None
        return InputDiscovery(
            selected_path=path,
            source_root=path,
            kind=InputKind.PRODUCT_TASK,
            trust=(
                DiscoveryTrust.VERIFIED
                if verified
                else DiscoveryTrust.REVIEW_REQUIRED
            ),
            summary=(
                f"产品任务已验证，将从 Stage {int(resume_stage) + 1} 继续"
                if resume_stage
                else "已识别产品任务，但没有可验证的兼容断点"
            ),
            task_directory=path,
            resume_after_stage=int(resume_stage) if resume_stage else None,
            warnings=(
                ()
                if verified
                else (str(inspection.get("detail") or "任务需要复核"),)
            ),
            errors=(
                ()
                if recognized
                else (str(inspection.get("detail") or "任务清单无效"),)
            ),
            details=dict(inspection),
        )

    if _looks_like_retired_processing_directory(path):
        return InputDiscovery(
            selected_path=path,
            source_root=path,
            kind=InputKind.UNSUPPORTED,
            trust=DiscoveryTrust.REVIEW_REQUIRED,
            summary="旧版处理目录不再支持续跑",
            errors=(
                "请重新选择原始母版文件，并作为新任务从 Stage 1 导入。",
            ),
            details={"reason": "retired_processing_directory"},
        )

    try:
        groups = discover_light_groups(path)
    except ValueError as error:
        return InputDiscovery(
            selected_path=path,
            source_root=path,
            kind=InputKind.UNSUPPORTED,
            trust=DiscoveryTrust.REVIEW_REQUIRED,
            summary="Light 目录扫描未完成",
            errors=(str(error),),
        )
    if groups:
        frame_count = sum(len(group.files) for group in groups)
        return InputDiscovery(
            selected_path=path,
            source_root=path,
            kind=InputKind.LIGHT_DIRECTORY,
            trust=DiscoveryTrust.RECOGNIZED,
            summary=(
                f"发现 {frame_count} 个 Light，分为 {len(groups)} 个独立叠加任务"
            ),
            light_groups=groups,
        )

    master_files = sorted(
        candidate.name
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in MASTER_SUFFIXES
    )
    warning = (
        f"目录中有 {len(master_files)} 个母版；请直接拖入要处理的具体文件。"
        if master_files
        else "目录中未发现递归 Light、产品任务清单或可处理输入。"
    )
    return InputDiscovery(
        selected_path=path,
        source_root=path,
        kind=InputKind.UNSUPPORTED,
        trust=DiscoveryTrust.REVIEW_REQUIRED,
        summary="无法确定目录动作",
        errors=(warning,),
        details={"top_level_master_files": master_files},
    )


__all__ = [
    "DiscoveryTrust",
    "FITS_LIGHT_SUFFIXES",
    "INPUT_DISCOVERY_SCHEMA",
    "InputDiscovery",
    "InputKind",
    "LightGroup",
    "MASTER_SUFFIXES",
    "REVIEW_SUFFIXES",
    "SourceHeaderSummary",
    "TASK_MANIFEST_NAME",
    "discover_input",
    "discover_light_groups",
    "inspect_source_header",
    "read_fits_group_metadata",
]
