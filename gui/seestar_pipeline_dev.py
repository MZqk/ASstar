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
        INPUT_MODE_LINEAR_RESUME,
        INPUT_MODE_STAGE2_CORRECTED_RESUME,
        PipelineWorker,
    )
    from .seestar_gui_dev import (
        DEFAULT_SIRIL_APP,
        DEFAULT_SIRIL_SEED,
        PROJECT_RESOURCES,
        _absolute_path,
        prepare_resource_overlay,
    )
    from .siril_runtime import verify_siril_offline_seed_venv
except ImportError:  # Direct execution: python gui/seestar_pipeline_dev.py
    from pipeline_worker import (  # type: ignore[no-redef]
        INPUT_MODE_AUTO,
        INPUT_MODE_LINEAR_RESUME,
        INPUT_MODE_STAGE2_CORRECTED_RESUME,
        PipelineWorker,
    )
    from seestar_gui_dev import (  # type: ignore[no-redef]
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
    Path.home() / "Library/Application Support/SeestarSuperimpose/runtime_home"
)
INPUT_MODES = (
    INPUT_MODE_AUTO,
    INPUT_MODE_STAGE2_CORRECTED_RESUME,
    INPUT_MODE_LINEAR_RESUME,
)


def validate_work_dir(work_dir: Path, input_mode: str) -> None:
    if not work_dir.is_dir():
        raise ValueError(f"工作目录不存在：{work_dir}")
    if input_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
        candidates = (
            work_dir / "stage2_corrected.fit",
            work_dir / "process/stage2_corrected.fit",
        )
        if not any(path.is_file() for path in candidates):
            raise ValueError(
                "从裁切后继续需要 stage2_corrected.fit："
                + " 或 ".join(str(path) for path in candidates)
            )
    elif input_mode == INPUT_MODE_LINEAR_RESUME:
        checkpoint = work_dir / "result_linear.fit"
        if not checkpoint.is_file():
            raise ValueError(f"从线性结果继续需要：{checkpoint}")


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
        "--network",
        action="store_true",
        help="允许在线 Gaia 等网络请求；默认严格离线",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="保留更多 stage 中间文件",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    validate_work_dir(args.work_dir, args.input_mode)
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

    with tempfile.TemporaryDirectory(prefix="seestar_core_resources_") as td:
        resources = prepare_resource_overlay(
            Path(td),
            project_resources=PROJECT_RESOURCES,
            siril_app=args.siril_app,
            siril_seed=args.siril_seed,
        )
        worker = PipelineWorker(
            work_dir=args.work_dir,
            config_template=PROJECT_RESOURCES / "config.1.4.ini.template",
            pipeline_path=PROJECT_ROOT / "pipeline/seestar_Superimpose.py",
            siril_plugin_dir=PROJECT_RESOURCES / "siril_plugins",
            resources=resources,
            runtime_home=args.runtime_home,
            siril_candidates=[
                args.siril_app / "Contents/MacOS/siril-cli",
            ],
            input_mode=args.input_mode,
            debug_mode=args.debug,
            network_mode=args.network,
            ai_stage_enabled=False,
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
