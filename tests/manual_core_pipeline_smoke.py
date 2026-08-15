#!/usr/bin/env python3
"""Interactive, packaging-free smoke test for the source core pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_manifest  # noqa: E402
import task_plan  # noqa: E402


DEFAULT_WORK_DIR = Path.home() / "SeeStar/sirildev"
DEFAULT_SIRIL_APP = Path("/Applications/Siril.app")
DEFAULT_SIRIL_SEED = (
    Path.home() / "Library/Application Support/org.siril.Siril/siril"
)
DEFAULT_RUNTIME_HOME = (
    Path.home() / "Library/Application Support/Starun/runtime_home"
)
MODE_LABELS = {
    "auto": "从原始输入验证完整 Stage 1-10",
}
MENU_MODES = tuple(MODE_LABELS)


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="人工选择入口，直接验证当前仓库核心流水线源码，不打包 GUI。"
    )
    parser.add_argument(
        "--mode",
        choices=MENU_MODES,
        default="auto",
        help="处理入口（仅支持 auto；续跑由验签产品任务发起）",
    )
    parser.add_argument(
        "--work-dir",
        type=_absolute_path,
        default=DEFAULT_WORK_DIR,
        help=f"输入/输出目录（默认：{DEFAULT_WORK_DIR}）",
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
        "--offline",
        dest="network",
        action="store_false",
        default=True,
        help="禁用在线 Gaia 等请求，仅验证离线回退链路",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="保留 Stage 中间文件（默认）",
    )
    debug_group.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="清理大部分 Stage 中间文件",
    )
    parser.set_defaults(debug=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将执行的命令，不启动 Siril",
    )
    return parser


def build_launcher_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "gui.starun_pipeline_dev",
        "--work-dir",
        str(args.work_dir),
        "--input-mode",
        mode,
        "--siril-app",
        str(args.siril_app),
        "--siril-seed",
        str(args.siril_seed),
        "--runtime-home",
        str(args.runtime_home),
    ]
    if not args.network:
        command.append("--offline")
    if args.debug:
        command.append("--debug")
    return command


def format_command(command: Sequence[str]) -> str:
    return " \\\n  ".join(
        f'"{part}"' if any(char.isspace() for char in part) else part
        for part in command
    )


def verify_result_manifest(
    work_dir: Path,
    *,
    run_started_at: float | None = None,
) -> tuple[bool, list[str]]:
    manifest_path = work_dir / "pipeline-result.json"
    details: list[str] = []
    if not manifest_path.is_file():
        return False, [f"结果清单不存在：{manifest_path}"]
    if run_started_at is not None:
        try:
            if manifest_path.stat().st_mtime < run_started_at - 1.0:
                return False, ["pipeline-result.json 不是本轮生成，拒绝复用旧结果"]
        except OSError as error:
            return False, [f"无法读取结果清单时间：{error}"]

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return False, [f"结果清单不可读：{error}"]
    if not isinstance(payload, dict):
        return False, ["结果清单根节点不是 JSON object"]

    claimed_hash = str(payload.get("manifest_hash") or "")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("manifest_hash", None)
    actual_hash = run_manifest.canonical_payload_hash(unsigned_payload)
    if not claimed_hash or claimed_hash != actual_hash:
        return False, ["pipeline-result.json manifest_hash 校验失败"]

    plan_path = work_dir / "processing-plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return False, [f"processing-plan.json 不可读：{error}"]
    plan_verification = task_plan.verify_processing_plan(plan)
    if not plan_verification.get("verified"):
        return False, [
            "processing-plan.json 校验失败："
            + str(plan_verification.get("detail") or "unknown error")
        ]
    if str(payload.get("plan_hash") or "") != str(plan.get("plan_hash") or ""):
        return False, ["pipeline-result.json 的 plan_hash 引用不匹配"]
    if str(payload.get("run_id") or "") != str(plan.get("run_id") or ""):
        return False, ["pipeline-result.json 与 processing-plan.json 的 run_id 不匹配"]

    outputs = payload.get("outputs") or {}
    if not isinstance(outputs, dict) or not outputs:
        return False, ["结果清单没有登记输出文件"]
    failed_outputs: list[str] = []
    for name, record in outputs.items():
        if not isinstance(record, dict):
            failed_outputs.append(str(name))
            continue
        relative_path = Path(str(record.get("path") or name))
        output_path = relative_path if relative_path.is_absolute() else work_dir / relative_path
        if (
            not output_path.is_file()
            or run_manifest.sha256_file(output_path) != record.get("sha256")
        ):
            failed_outputs.append(str(name))
    if failed_outputs:
        return False, ["输出缺失或 SHA-256 不一致：" + ", ".join(failed_outputs)]

    status = str(payload.get("status") or "unknown")
    details.append(f"pipeline_status={status}")
    details.append(f"verified_outputs={len(outputs)}")
    details.append(f"manifest_hash={claimed_hash}")
    details.append(f"plan_hash={plan['plan_hash']}")
    if status == "failed":
        return False, details + ["流水线全局状态为 failed"]

    final_quality_path = work_dir / "process/final_quality_report.json"
    if final_quality_path.is_file():
        try:
            final_quality = json.loads(final_quality_path.read_text(encoding="utf-8"))
            details.append(
                "final_quality="
                + str(final_quality.get("final_quality") or final_quality.get("status") or "unknown")
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            details.append("final_quality=unreadable")
    return True, details


def run_smoke(args: argparse.Namespace) -> int:
    mode = args.mode
    command = build_launcher_command(args, mode)
    print("\n核心源码验证配置：")
    print(f"  范围：{MODE_LABELS[mode]}")
    print(f"  工作目录：{args.work_dir}")
    print(f"  网络：{'ON' if args.network else 'OFF'}")
    print(f"  Debug：{'ON' if args.debug else 'OFF'}")
    print("  GUI：不启动")
    print("\n执行命令：")
    print(format_command(command))
    if args.dry_run:
        return 0
    if not args.work_dir.is_dir():
        print(f"[CORE_SMOKE_FAIL] 工作目录不存在：{args.work_dir}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = args.work_dir / f"core_pipeline_smoke_{stamp}.log"
    run_started_at = time.time()
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("无法捕获 launcher 输出")
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
            exit_code = process.wait()
    except (OSError, RuntimeError) as error:
        print(f"[CORE_SMOKE_FAIL] 启动失败：{error}", file=sys.stderr)
        return 2

    if exit_code != 0:
        print(f"[CORE_SMOKE_FAIL] launcher_exit_code={exit_code} log={log_path}")
        return exit_code

    verified, details = verify_result_manifest(
        args.work_dir,
        run_started_at=run_started_at,
    )
    marker = "CORE_SMOKE_PASS" if verified else "CORE_SMOKE_FAIL"
    print(f"[{marker}] " + " ".join(details) + f" log={log_path}")
    return 0 if verified else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_smoke(args)
    except RuntimeError as error:
        print(f"[CORE_SMOKE_FAIL] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
