from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_RESOURCE_REL = Path("pipeline") / "seestar_Superimpose.py"
SIRIL_PLUGIN_RESOURCE_REL = Path("resources") / "siril_plugins"
APP_RUNTIME_HOME_REL = Path("Library/Application Support/SeestarSuperimpose/runtime_home")
AI_ENV_RESOURCE_REL = Path("ai.env")
DEFAULT_ENV_RESOURCE_REL = Path("default.env")
AI_ENV_OVERRIDE_NAME = ".seestar_ai.env"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    if is_frozen():
        exe_path = Path(sys.executable).resolve()
        return exe_path.parent.parent / "Resources"
    return project_root() / "resources"


def default_pipeline_path(resources: Path) -> Path:
    if is_frozen():
        return resources / PIPELINE_RESOURCE_REL
    return project_root() / PIPELINE_RESOURCE_REL


def default_siril_plugin_dir(resources: Path) -> Path:
    if is_frozen():
        return resources / "siril_plugins"
    return project_root() / SIRIL_PLUGIN_RESOURCE_REL


def resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_runtime_home() -> Path:
    return Path.home() / APP_RUNTIME_HOME_REL


def siril_state_root_from_home(runtime_home: Path) -> Path:
    return runtime_home / "Library/Application Support/org.siril.Siril/siril"


def shell_quote_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')
