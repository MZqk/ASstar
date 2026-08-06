"""Deterministic user-input discovery for files, Light trees, and tasks.

No stage is inferred from a filename.  External masters always enter through
Stage 1; only a signed product task may expose a formal resume boundary.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from . import run_manifest
except ImportError:
    import run_manifest


INPUT_DISCOVERY_SCHEMA = "seestar.input-discovery.v1"
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
        "seestarsuperimpose",
    }
)
_LEGACY_MARKERS = (
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
    LEGACY_DIRECTORY = "legacy_directory"
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
        verified_legacy = (
            self.kind == InputKind.LEGACY_DIRECTORY
            and self.trust == DiscoveryTrust.VERIFIED
            and self.master_file is not None
        )
        return (regular_input or verified_legacy) and not self.errors

    @property
    def creates_independent_task(self) -> bool:
        return self.kind in {
            InputKind.MASTER_FILE,
            InputKind.LIGHT_DIRECTORY,
            InputKind.REVIEW_FILE,
            InputKind.LEGACY_DIRECTORY,
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
    """Read only primary FITS cards needed for Light grouping."""

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


def _looks_like_legacy_directory(path: Path) -> bool:
    return any((path / name).exists() for name in _LEGACY_MARKERS) or any(
        candidate.is_file() for candidate in path.glob("stage[0-9]*_*.fit")
    )


def inspect_legacy_directory(path: Path) -> Dict[str, Any]:
    """Verify a legacy checkpoint without treating its filename as provenance."""

    root = path.expanduser().resolve()
    result: Dict[str, Any] = {
        "recognized": _looks_like_legacy_directory(root),
        "verified": False,
        "directory": str(root),
        "resume_after_stage": None,
    }
    manifest_path = root / "pipeline-result.json"
    manifest = run_manifest.load_json(manifest_path)
    if manifest is None or str(manifest.get("schema") or "") != (
        "seestar.pipeline-result.v1"
    ):
        result["detail"] = "pipeline-result.json 缺失或 schema 不受支持"
        return result
    claimed_manifest_hash = str(manifest.get("manifest_hash") or "")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_hash", None)
    if not claimed_manifest_hash or claimed_manifest_hash != (
        run_manifest.canonical_payload_hash(unsigned_manifest)
    ):
        result["detail"] = "pipeline-result.json 哈希无效"
        return result

    plan_hash = str(manifest.get("plan_hash") or "")
    plan = run_manifest.load_json(root / "processing-plan.json")
    if plan is None or str(plan.get("schema") or "") != (
        "seestar.processing-plan.v1"
    ):
        result["detail"] = "processing-plan.json 缺失或 schema 不受支持"
        return result
    claimed_plan_hash = str(plan.get("plan_hash") or "")
    unsigned_plan = dict(plan)
    unsigned_plan.pop("plan_hash", None)
    if (
        not plan_hash
        or not claimed_plan_hash
        or claimed_plan_hash != plan_hash
        or claimed_plan_hash != run_manifest.canonical_payload_hash(unsigned_plan)
    ):
        result["detail"] = "旧版处理计划与结果清单不匹配"
        return result

    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        result["detail"] = "pipeline-result.json 没有断点来源记录"
        return result
    rejections: Dict[str, str] = {}
    for checkpoint_name, stage_number in (
        ("result_linear", 5),
        ("stage2_corrected", 2),
        ("stage1_prepared", 1),
    ):
        record = checkpoints.get(checkpoint_name)
        if not isinstance(record, Mapping):
            rejections[checkpoint_name] = "missing"
            continue
        relative_path = Path(str(record.get("path") or ""))
        if relative_path.is_absolute() or not relative_path.as_posix():
            rejections[checkpoint_name] = "checkpoint path is not relative"
            continue
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            rejections[checkpoint_name] = "checkpoint path escapes legacy directory"
            continue
        verification = run_manifest.verify_resume_provenance(
            work_dir=root,
            input_path=candidate,
            checkpoint_name=checkpoint_name,
        )
        if not verification.get("verified"):
            rejections[checkpoint_name] = str(
                verification.get("detail") or "checkpoint verification failed"
            )
            continue
        result.update(
            {
                "verified": True,
                "checkpoint": checkpoint_name,
                "legacy_stage": stage_number,
                "checkpoint_path": str(candidate),
                "manifest_hash": claimed_manifest_hash,
                "plan_hash": plan_hash,
                "detail": (
                    f"已验证旧版 {checkpoint_name}；"
                    "迁移后作为外部母版从 Stage 1 安全导入"
                ),
                "rejections": rejections,
            }
        )
        return result
    result["detail"] = "没有通过清单、线性状态和 SHA-256 校验的旧版断点"
    result["rejections"] = rejections
    return result


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

    if _looks_like_legacy_directory(path):
        inspection = inspect_legacy_directory(path)
        verified = bool(inspection.get("verified"))
        checkpoint_path = inspection.get("checkpoint_path")
        return InputDiscovery(
            selected_path=path,
            source_root=path,
            kind=InputKind.LEGACY_DIRECTORY,
            trust=(
                DiscoveryTrust.VERIFIED
                if verified
                else DiscoveryTrust.REVIEW_REQUIRED
            ),
            summary=(
                "旧版目录断点已验证，将只读迁移并从 Stage 1 安全导入"
                if verified
                else "检测到旧版处理目录，但无法安全迁移"
            ),
            master_file=(
                Path(str(checkpoint_path)) if verified and checkpoint_path else None
            ),
            warnings=(
                "不会根据 stage/result 文件名自动推断起点。",
                "旧目录保持只读；旧 Stage 5 不会被冒充为当前契约断点。",
            ),
            errors=(
                ()
                if verified
                else (str(inspection.get("detail") or "旧版清单无法验证"),)
            ),
            details=dict(inspection),
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
    "TASK_MANIFEST_NAME",
    "discover_input",
    "discover_light_groups",
    "inspect_legacy_directory",
    "read_fits_group_metadata",
]
