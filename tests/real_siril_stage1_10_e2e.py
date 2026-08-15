#!/usr/bin/env python3
"""Run the real offline Siril Stage 1-10 regression on a prepared macOS host.

This is intentionally not named ``test_*.py``: the ordinary unit suite must not
pretend that a mocked run covers Siril, the offline model assets, or final file
export.  CI invokes this program on the dedicated ``starun-e2e`` runner.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.runtime_capabilities import (  # noqa: E402
    GAIA_ASTRO_EXPECTED_SIZE_BYTES,
    GAIA_ASTRO_FILENAME,
    GAIA_XP_EXPECTED_CHUNKS,
    GAIA_XP_FILE_PREFIX,
    GAIA_XP_MIN_CHUNK_BYTES,
    runtime_catalog_paths,
)
from pipeline import run_manifest  # noqa: E402
from tests import manual_core_pipeline_smoke as core_smoke  # noqa: E402


EXPECTED_STAGES = tuple(range(1, 11))
SUCCESS_STATUSES = {"success", "partial_success", "review_required"}
DEFAULT_TIMEOUT_SECONDS = 3 * 60 * 60


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _env_path(name: str, default: Path | None = None) -> Path | None:
    raw = os.getenv(name, "").strip()
    if raw:
        return _absolute_path(raw)
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在已准备的 macOS runner 上执行真实 Siril、离线 SyQon、"
            "Stage 1-10 与最终导出回归。"
        )
    )
    parser.add_argument(
        "--input",
        type=_absolute_path,
        default=_env_path("STARUN_REAL_E2E_INPUT"),
        help="已审阅的真实线性 FITS/XISF 母版",
    )
    parser.add_argument(
        "--work-dir",
        type=_absolute_path,
        default=_env_path("STARUN_REAL_E2E_WORK_DIR"),
        help="必须为空或尚不存在；运行现场会完整保留",
    )
    parser.add_argument(
        "--siril-app",
        type=_absolute_path,
        default=_env_path(
            "STARUN_SIRIL_APP",
            Path("/Applications/Siril.app"),
        ),
    )
    parser.add_argument(
        "--siril-seed",
        type=_absolute_path,
        default=_env_path(
            "STARUN_SIRIL_SEED",
            Path.home() / "Library/Application Support/org.siril.Siril/siril",
        ),
    )
    parser.add_argument(
        "--runtime-home",
        type=_absolute_path,
        default=_env_path("STARUN_RUNTIME_HOME"),
        help="必须已包含完整离线 Gaia astro 与 48 个 XP 分块",
    )
    parser.add_argument(
        "--offline-resource-root",
        type=_absolute_path,
        default=_env_path("STARUN_OFFLINE_RESOURCE_ROOT", REPO_ROOT / "resources"),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    return parser


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.is_file() else 0
    except OSError:
        return 0


def validate_environment(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if args.input is None or not args.input.is_file():
        errors.append(f"真实输入不存在：{args.input}")
    if args.work_dir is None:
        errors.append("缺少 --work-dir/STARUN_REAL_E2E_WORK_DIR")
    elif args.work_dir.exists() and not args.work_dir.is_dir():
        errors.append(f"工作目录路径不是目录：{args.work_dir}")
    elif args.work_dir.exists() and any(args.work_dir.iterdir()):
        errors.append(f"工作目录不是空目录：{args.work_dir}")

    siril_cli = args.siril_app / "Contents/MacOS/siril-cli"
    if not siril_cli.is_file() or not os.access(siril_cli, os.X_OK):
        errors.append(f"真实 Siril CLI 不可执行：{siril_cli}")
    if not args.siril_seed.is_dir() or not (args.siril_seed / ".python_module").exists():
        errors.append(f"Siril Python seed 不完整：{args.siril_seed}")
    if args.runtime_home is None:
        errors.append("缺少 --runtime-home/STARUN_RUNTIME_HOME")
    else:
        astro_path, xp_root = runtime_catalog_paths(args.runtime_home)
        astro_size = _file_size(astro_path)
        if astro_size != GAIA_ASTRO_EXPECTED_SIZE_BYTES:
            errors.append(
                "离线 Gaia astro 不完整："
                f"{astro_path} size={astro_size} expected={GAIA_ASTRO_EXPECTED_SIZE_BYTES}"
            )
        missing_xp = []
        for index in range(GAIA_XP_EXPECTED_CHUNKS):
            chunk = xp_root / f"{GAIA_XP_FILE_PREFIX}{index}.dat"
            if _file_size(chunk) < GAIA_XP_MIN_CHUNK_BYTES:
                missing_xp.append(chunk.name)
        if missing_xp:
            errors.append(
                "离线 Gaia XP 分块缺失或过小：" + ", ".join(missing_xp)
            )

    resource_root = args.offline_resource_root
    required_resources = (
        resource_root / "siril_plugins/syqon_starless/zenith.pt",
        resource_root / "siril_plugins/syqon_starless/zenith.pt.sha256",
        resource_root / "siril_plugins/cosmic_clarity/deep_denoise_color_AI4.pth",
        resource_root / "siril_plugins/cosmic_clarity/deep_denoise_mono_AI4.pth",
    )
    for required in required_resources:
        if not required.is_file() or _file_size(required) <= 0:
            errors.append(f"离线插件资源缺失：{required}")
    return errors


def build_core_smoke_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(REPO_ROOT / "tests/manual_core_pipeline_smoke.py"),
        "--mode",
        "auto",
        "--work-dir",
        str(args.work_dir),
        "--siril-app",
        str(args.siril_app),
        "--siril-seed",
        str(args.siril_seed),
        "--runtime-home",
        str(args.runtime_home),
        "--offline",
        "--debug",
    ]


def _stage_number(name: object) -> int | None:
    match = re.match(r"^(?:阶段|Stage)\s*(\d+)\s*[:·]", str(name or "").strip())
    return int(match.group(1)) if match else None


def verify_e2e_artifacts(work_dir: Path) -> tuple[bool, list[str]]:
    verified, details = core_smoke.verify_result_manifest(work_dir)
    if not verified:
        return False, details

    result = run_manifest.load_json(work_dir / "pipeline-result.json") or {}
    errors: list[str] = []
    status = str(result.get("status") or "")
    if status not in SUCCESS_STATUSES:
        errors.append(f"流水线状态不可交付：{status or 'missing'}")

    actual_steps = result.get("actual_steps")
    if not isinstance(actual_steps, list):
        errors.append("pipeline-result.json 缺少 actual_steps")
        actual_steps = []
    observed = {
        number
        for number in (_stage_number(step.get("name")) for step in actual_steps if isinstance(step, dict))
        if number is not None
    }
    missing_stages = sorted(set(EXPECTED_STAGES) - observed)
    if missing_stages:
        errors.append("未覆盖完整 Stage 1-10：" + ", ".join(map(str, missing_stages)))
    failed_stages = [
        str(step.get("name") or "unknown")
        for step in actual_steps
        if isinstance(step, dict) and str(step.get("status") or "").lower() == "failed"
    ]
    if failed_stages:
        errors.append("存在失败阶段：" + ", ".join(failed_stages))

    final_quality = run_manifest.load_json(
        work_dir / "process/final_quality_report.json"
    )
    if not isinstance(final_quality, dict):
        errors.append("缺少可解析的 process/final_quality_report.json")

    syqon_exchange = run_manifest.load_json(
        work_dir / "process/stage6_syqon_exchange.json"
    )
    if not isinstance(syqon_exchange, dict) or not bool(
        syqon_exchange.get("accepted")
    ) or str(syqon_exchange.get("status") or "").lower() != "accepted":
        errors.append("真实离线 SyQon 交换未验收")

    if errors:
        return False, details + errors
    return True, details + [
        "verified_stages=1-10",
        "offline_syqon=accepted",
        f"gaia_astro={GAIA_ASTRO_FILENAME}",
    ]


def _terminate_process_group(process: subprocess.Popen[object]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def run_e2e(args: argparse.Namespace) -> int:
    errors = validate_environment(args)
    if errors:
        for error in errors:
            print(f"[REAL_SIRIL_E2E_FAIL] {error}", file=sys.stderr)
        return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    input_copy = args.work_dir / args.input.name
    shutil.copy2(args.input, input_copy)
    log_path = args.work_dir / "real_siril_stage1_10_e2e.log"
    astro_path, xp_root = runtime_catalog_paths(args.runtime_home)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "QT_QPA_PLATFORM": "offscreen",
            "STARUN_NETWORK_MODE": "0",
            "STARUN_OFFLINE_RESOURCE_ROOT": str(args.offline_resource_root),
            "STARUN_GAIA_ASTRO_CATALOG": str(astro_path),
            "STARUN_GAIA_PHOTO_CATALOG": str(xp_root),
        }
    )
    command = build_core_smoke_command(args)
    print("[REAL_SIRIL_E2E] " + core_smoke.format_command(command))
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=max(60, int(args.timeout_seconds)))
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            print(
                f"[REAL_SIRIL_E2E_FAIL] timeout after {args.timeout_seconds}s; "
                f"log={log_path}",
                file=sys.stderr,
            )
            return 2

    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
    if tail:
        print("\n".join(tail))
    if exit_code != 0:
        print(
            f"[REAL_SIRIL_E2E_FAIL] core smoke exit={exit_code}; log={log_path}",
            file=sys.stderr,
        )
        return exit_code

    verified, verification_details = verify_e2e_artifacts(args.work_dir)
    marker = "REAL_SIRIL_E2E_PASS" if verified else "REAL_SIRIL_E2E_FAIL"
    print(f"[{marker}] " + " ".join(verification_details) + f" log={log_path}")
    return 0 if verified else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run_e2e(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
