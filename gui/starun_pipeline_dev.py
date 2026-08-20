#!/usr/bin/env python3
"""Run the source pipeline headlessly with system Siril and an existing seed."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence

try:
    from .pipeline_worker import (
        INPUT_MODE_AUTO,
        PipelineWorker,
    )
    from .starun_gui_dev import (
        DEFAULT_SIRIL_APP,
        DEFAULT_SIRIL_SEED,
        PROJECT_RESOURCES,
        _absolute_path,
        prepare_resource_overlay,
    )
    from .siril_runtime import verify_siril_offline_seed_venv
except ImportError:  # Direct execution: python gui/starun_pipeline_dev.py
    from pipeline_worker import (  # type: ignore[no-redef]
        INPUT_MODE_AUTO,
        PipelineWorker,
    )
    from starun_gui_dev import (  # type: ignore[no-redef]
        DEFAULT_SIRIL_APP,
        DEFAULT_SIRIL_SEED,
        PROJECT_RESOURCES,
        _absolute_path,
        prepare_resource_overlay,
    )
    from siril_runtime import (  # type: ignore[no-redef]
        verify_siril_offline_seed_venv,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_HOME = (
    Path.home() / "Library/Application Support/Starun/runtime_home"
)
INPUT_MODES = (INPUT_MODE_AUTO,)


def validate_work_dir(work_dir: Path, input_mode: str) -> None:
    if not work_dir.is_dir():
        raise ValueError(f"工作目录不存在：{work_dir}")
    if input_mode != INPUT_MODE_AUTO:
        raise ValueError("开发入口仅接受 auto；续跑必须由验签产品任务发起")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "不启动 GUI，直接使用源码 PipelineWorker 和系统 siril-cli "
            "验证核心流水线输出。"
        )
    )
    parser.add_argument(
        "--work-dir",
        type=_absolute_path,
        required=True,
        help="流水线输入和输出目录",
    )
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default=INPUT_MODE_AUTO,
        help="处理入口模式（默认：auto）",
    )
    parser.add_argument(
        "--siril-app",
        type=_absolute_path,
        default=DEFAULT_SIRIL_APP,
        help=f"系统 Siril.app（默认：{DEFAULT_SIRIL_APP}）",
    )
    parser.add_argument(
        "--siril-seed",
        type=_absolute_path,
        default=DEFAULT_SIRIL_SEED,
        help=f"现有 Siril seed（默认：{DEFAULT_SIRIL_SEED}）",
    )
    parser.add_argument(
        "--runtime-home",
        type=_absolute_path,
        default=DEFAULT_RUNTIME_HOME,
        help=f"隔离 runtime HOME（默认：{DEFAULT_RUNTIME_HOME}）",
    )
    parser.add_argument(
        "--offline-resource-root",
        type=_absolute_path,
        default=None,
        help="可选 runner 离线资源包根目录；必须包含 siril_plugins/",
    )
    parser.add_argument(
        "--offline",
        dest="network",
        action="store_false",
        default=True,
        help="显式禁用互联网并要求使用本地 Gaia 星表",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="保留更多 stage 中间文件",
    )
    return parser


def resolve_siril_plugin_dir(offline_resource_root: Path | None) -> Path:
    if offline_resource_root is None:
        return PROJECT_RESOURCES / "siril_plugins"
    plugin_dir = offline_resource_root / "siril_plugins"
    if not plugin_dir.is_dir():
        raise ValueError(f"离线资源包缺少 siril_plugins：{plugin_dir}")
    return plugin_dir


def run_pipeline(args: argparse.Namespace) -> int:
    validate_work_dir(args.work_dir, args.input_mode)
    siril_plugin_dir = resolve_siril_plugin_dir(args.offline_resource_root)
    runtime_venv = (
        args.runtime_home
        / "Library/Application Support/org.siril.Siril/siril/venv"
    )
    ok, detail = verify_siril_offline_seed_venv(runtime_venv)
    if not ok:
        raise RuntimeError(f"隔离 Siril runtime 未就绪：{detail}")

    result: dict[str, object] = {}

    def on_done(
        status: str,
        exit_code: int,
        had_errors: bool,
        cli_used: str,
    ) -> None:
        result.update(
            status=status,
            exit_code=exit_code,
            had_errors=had_errors,
            cli_used=cli_used,
        )

    with tempfile.TemporaryDirectory(prefix="starun_core_resources_") as td:
        resources = prepare_resource_overlay(
            Path(td),
            project_resources=PROJECT_RESOURCES,
            siril_app=args.siril_app,
            siril_seed=args.siril_seed,
        )
        worker = PipelineWorker(
            work_dir=args.work_dir,
            config_template=PROJECT_RESOURCES / "config.1.4.ini.template",
            pipeline_path=PROJECT_ROOT / "pipeline/starun.py",
            siril_plugin_dir=siril_plugin_dir,
            resources=resources,
            runtime_home=args.runtime_home,
            siril_candidates=[
                args.siril_app / "Contents/MacOS/siril-cli",
            ],
            input_mode=args.input_mode,
            debug_mode=args.debug,
            network_mode=args.network,
        )
        worker.log.connect(lambda text: sys.stdout.write(text))
        worker.state.connect(
            lambda state: print(f"[CORE_STATE] {state}", flush=True)
        )
        worker.done.connect(on_done)
        worker.run()

    print(
        "[CORE_RESULT] "
        + " ".join(f"{key}={value}" for key, value in result.items()),
        flush=True,
    )
    return (
        0
        if result.get("status") in {"Completed", "CompletedWithWarning"}
        and result.get("exit_code") == 0
        else 1
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_pipeline(args)
    except Exception as exc:
        print(f"[CORE_ERROR] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
