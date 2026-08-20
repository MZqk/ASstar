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
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


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
from pipeline import outcome, run_manifest  # noqa: E402
from tests import manual_core_pipeline_smoke as core_smoke  # noqa: E402


EXPECTED_STAGES = tuple(range(1, 11))
FINAL_QUALITY_SCHEMA = "starun.final-quality.v2"
SYQON_EXCHANGE_SCHEMA = "starun.syqon-pixel-exchange.v2"
SYQON_SELECTED_SCHEMA = "starun.syqon-selected-pair.v1"
SYQON_COMMIT_STOP_REASONS = frozenset(
    {"CONTRACT_VALID_PAIR_COMMITTED", "DERIVED_GENERATION_COMMITTED"}
)
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
        "--offline-resource-root",
        str(args.offline_resource_root),
        "--offline",
        "--debug",
    ]


def _blocking_issue_labels(value: object) -> tuple[list[str], bool]:
    if not isinstance(value, list):
        return [], False
    labels: list[str] = []
    valid = True
    for item in value:
        if not isinstance(item, Mapping):
            valid = False
            continue
        raw_severity = item.get("severity")
        if not isinstance(raw_severity, str) or not raw_severity.strip():
            valid = False
            continue
        severity = raw_severity.strip().lower()
        if severity not in outcome.ISSUE_SEVERITIES:
            valid = False
            continue
        if severity in {"error", "fatal"}:
            labels.append(str(item.get("code") or item.get("message") or severity))
    return labels, valid


def _manifest_output_entries(
    result: Mapping[str, object],
    work_dir: Path,
) -> tuple[list[tuple[str, Path, Mapping[str, object]]], list[str]]:
    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping):
        return [], ["pipeline-result.json outputs 不是对象"]
    entries: list[tuple[str, Path, Mapping[str, object]]] = []
    errors: list[str] = []
    resolved_root = work_dir.resolve()
    for key, record in outputs.items():
        if not isinstance(record, Mapping):
            errors.append(f"输出记录不是对象：{key}")
            continue
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            errors.append(f"输出记录缺少相对 path：{key}")
            continue
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            errors.append(f"输出 path 不得为绝对路径：{raw_path}")
            continue
        output_path = (resolved_root / relative_path).resolve()
        try:
            output_path.relative_to(resolved_root)
        except ValueError:
            errors.append(f"输出 path 逃逸工作目录：{raw_path}")
            continue
        entries.append((relative_path.name.lower(), output_path, record))
    return entries, errors


def _validate_syqon_exchange(work_dir: Path, report: object) -> list[str]:
    if not isinstance(report, Mapping):
        return ["缺少可解析的 process/stage6_syqon_exchange.json"]

    errors: list[str] = []
    if str(report.get("schema") or "") != SYQON_EXCHANGE_SCHEMA:
        errors.append("SyQon exchange 不是 starun.syqon-pixel-exchange.v2")
    if report.get("accepted") is not True or str(
        report.get("status") or ""
    ).lower() != "accepted":
        errors.append("真实离线 SyQon 交换未验收")

    attempt_id = str(report.get("attempt_id") or "").strip()
    selected_attempt_id = str(report.get("selected_attempt_id") or "").strip()
    pair_id = str(report.get("pair_id") or "").strip()
    stop_reason = str(report.get("stop_reason") or "").strip()
    generation = str(report.get("generation") or "").strip()
    if not attempt_id or selected_attempt_id != attempt_id:
        errors.append("SyQon exchange attempt provenance 不完整")
    if not pair_id:
        errors.append("SyQon exchange pair_id 缺失")
    if stop_reason not in SYQON_COMMIT_STOP_REASONS:
        errors.append("SyQon exchange stop_reason 不是已提交 generation")

    for field in ("files", "assets", "runtime", "worker"):
        value = report.get(field)
        if not isinstance(value, Mapping) or not value:
            errors.append(f"SyQon exchange {field} provenance 不完整")

    attempts = report.get("attempts")
    if not isinstance(attempts, list) or not any(
        isinstance(item, Mapping)
        and str(item.get("attempt_id") or "") == attempt_id
        and item.get("accepted") is True
        and str(item.get("status") or "").lower() == "accepted"
        for item in attempts or []
    ):
        errors.append("SyQon exchange 缺少已验收 attempt 记录")

    generations = report.get("generations")
    pointer = report.get("selected_pointer")
    if not isinstance(pointer, Mapping):
        errors.append("SyQon exchange 缺少 selected_pointer")
        pointer = {}
    pointer_attempt_id = str(pointer.get("attempt_id") or "").strip()
    if str(pointer.get("schema") or "") != SYQON_SELECTED_SCHEMA:
        errors.append("SyQon selected_pointer schema 无效")
    if not pointer_attempt_id or str(pointer.get("pair_id") or "") != pair_id:
        errors.append("SyQon selected_pointer pair provenance 不匹配")
    if str(pointer.get("stop_reason") or "") != stop_reason:
        errors.append("SyQon selected_pointer stop_reason 不匹配")
    if str(pointer.get("generation") or "") != generation:
        errors.append("SyQon selected_pointer generation 不匹配")
    if not isinstance(generations, list) or not any(
        isinstance(item, Mapping)
        and str(item.get("attempt_id") or "") == pointer_attempt_id
        and str(item.get("pair_id") or "") == pair_id
        and str(item.get("stop_reason") or "") == stop_reason
        for item in generations or []
    ):
        errors.append("SyQon exchange generations 与 selected_pointer 不匹配")

    selected_report = run_manifest.load_json(
        work_dir / "process/stage6_syqon_selected.json"
    )
    if selected_report != dict(pointer):
        errors.append("SyQon selected pair 文件与 exchange pointer 不匹配")
    return errors


def verify_e2e_artifacts(work_dir: Path) -> tuple[bool, list[str]]:
    verified, details = core_smoke.verify_result_manifest(work_dir)
    if not verified:
        return False, details

    result = run_manifest.load_json(work_dir / "pipeline-result.json") or {}
    errors: list[str] = []
    schema = str(result.get("schema") or "")
    if schema != outcome.PIPELINE_RESULT_SCHEMA_V2:
        errors.append(
            "pipeline-result.json 必须使用正式 v2 schema："
            f"{schema or 'missing'}"
        )

    status = str(result.get("status") or "")
    if status != "success":
        errors.append(f"流水线状态不是正式 success：{status or 'missing'}")
    if str(result.get("failure_reason") or "").strip():
        errors.append("流水线仍登记 failure_reason")

    for field in (
        "review_required",
        "had_errors",
        "had_fatal_errors",
        "had_degradations",
        "had_fallbacks",
    ):
        if result.get(field) is not False:
            errors.append(f"流水线正式验收要求 {field}=false")

    review_requirements = result.get("review_requirements")
    if not isinstance(review_requirements, list) or review_requirements:
        errors.append("流水线仍有 review_requirements")
    result_errors = result.get("errors")
    if not isinstance(result_errors, list) or result_errors:
        errors.append("流水线仍有 errors")
    result_issue_labels, result_issues_valid = _blocking_issue_labels(
        result.get("issues")
    )
    if not result_issues_valid:
        errors.append("流水线 issues 不是规范对象数组")
    if result_issue_labels:
        errors.append("流水线仍有 error/fatal issues：" + ", ".join(result_issue_labels))

    actual_steps = result.get("actual_steps")
    if not isinstance(actual_steps, list):
        errors.append("pipeline-result.json 缺少 actual_steps")
        actual_steps = []
    if len(actual_steps) != len(EXPECTED_STAGES):
        errors.append(
            "未覆盖完整 Stage 1-10："
            f"actual_steps={len(actual_steps)}"
        )
    observed_stages: list[int | None] = []
    for step in actual_steps:
        raw_stage = step.get("stage") if isinstance(step, Mapping) else None
        observed_stages.append(raw_stage if type(raw_stage) is int else None)
    if observed_stages != list(EXPECTED_STAGES):
        errors.append(
            "Stage 顺序必须严格为 1-10："
            + ", ".join(map(str, observed_stages))
        )
    for expected_stage, step in zip(EXPECTED_STAGES, actual_steps):
        if not isinstance(step, Mapping):
            errors.append(f"Stage {expected_stage} 记录不是对象")
            continue
        stage_label = f"Stage {expected_stage}"
        step_status = str(step.get("status") or "").strip().lower()
        if step_status != "ok":
            errors.append(f"{stage_label} status={step_status or 'missing'}")
        execution = str(step.get("execution") or "").strip().lower()
        if execution != "completed":
            errors.append(f"{stage_label} execution={execution or 'missing'}")
        for field in ("fallback_used", "upstream_passthrough", "review_required"):
            if step.get(field) is not False:
                errors.append(f"{stage_label} 正式验收要求 {field}=false")
        review_reasons = step.get("review_reasons")
        if not isinstance(review_reasons, list) or review_reasons:
            errors.append(f"{stage_label} 仍有 review_reasons")
        issue_labels, issues_valid = _blocking_issue_labels(step.get("issues"))
        if not issues_valid:
            errors.append(f"{stage_label} issues 不是规范对象数组")
        if issue_labels:
            errors.append(
                f"{stage_label} 仍有 error/fatal issues：" + ", ".join(issue_labels)
            )

    if all(isinstance(step, Mapping) for step in actual_steps) and isinstance(
        review_requirements,
        list,
    ):
        try:
            derived = outcome.summarize_outcome(
                actual_steps,
                review_requirements,
                failure_reason=result.get("failure_reason"),
            )
        except (AttributeError, TypeError, ValueError) as error:
            errors.append(f"无法从阶段事实推导流水线状态：{error}")
        else:
            if derived.get("status") != "success":
                errors.append(
                    "阶段事实推导结果不是 success："
                    + str(derived.get("status") or "missing")
                )

    color_calibration = result.get("color_calibration")
    if not isinstance(color_calibration, Mapping):
        errors.append("pipeline-result.json 缺少 color_calibration")
    else:
        for field in ("requires_review", "stage7_forced_delivery"):
            if color_calibration.get(field) is not False:
                errors.append(f"色彩交付要求 {field}=false")

    star_separation = result.get("star_separation")
    if not isinstance(star_separation, Mapping):
        errors.append("pipeline-result.json 缺少 star_separation")
    else:
        if str(star_separation.get("state") or "") != "accepted":
            errors.append("星点分离状态不是 accepted")
        for field in (
            "stars_required",
            "stars_applied",
            "output_contains_stars",
            "remix_formally_accepted",
            "delivery_contract_accepted",
        ):
            if star_separation.get(field) is not True:
                errors.append(f"正式星点交付要求 {field}=true")
        for field in (
            "output_withheld",
            "starmask_borderline_review_required",
            "psf_review_required",
            "review_candidate_selected",
        ):
            if star_separation.get(field) is not False:
                errors.append(f"正式星点交付要求 {field}=false")
        if str(star_separation.get("final_source") or "") != "stage9_remixed":
            errors.append("正式星点交付 final_source 不是 stage9_remixed")

    output_entries, output_errors = _manifest_output_entries(result, work_dir)
    errors.extend(output_errors)
    review_outputs = [
        name for name, _path, _record in output_entries
        if name.startswith("result_review")
    ]
    if review_outputs:
        errors.append("结果清单包含 review-only 输出：" + ", ".join(review_outputs))
    formal_fit_outputs = [
        (name, path, record)
        for name, path, record in output_entries
        if name.endswith("_final.fit") and not name.startswith("result_review")
    ]
    if not formal_fit_outputs:
        errors.append("结果清单缺少非复核的 *_final.fit 正式产物")
    for name, path, record in formal_fit_outputs:
        try:
            actual_size = path.stat().st_size
        except OSError:
            actual_size = 0
        recorded_size = record.get("size")
        if actual_size <= 0:
            errors.append(f"正式 FIT 产物为空：{name}")
        if type(recorded_size) is not int or recorded_size != actual_size:
            errors.append(f"正式 FIT size 记录不匹配：{name}")

    final_quality = run_manifest.load_json(
        work_dir / "process/final_quality_report.json"
    )
    if not isinstance(final_quality, dict):
        errors.append("缺少可解析的 process/final_quality_report.json")
    else:
        if str(final_quality.get("schema") or "") != FINAL_QUALITY_SCHEMA:
            errors.append("最终质量报告不是 starun.final-quality.v2")
        if str(final_quality.get("status") or "") != "ok":
            errors.append("最终质量报告 status 不是 ok")
        if str(final_quality.get("final_quality") or "") != "ok":
            errors.append("最终质量报告 final_quality 不是 ok")
        if final_quality.get("needs_conservative_rerun") is not False:
            errors.append("最终质量报告仍要求 conservative rerun")
        if final_quality.get("issues") != []:
            errors.append("最终质量报告仍有 hard issues")
        if (
            "hard_issues" in final_quality
            and final_quality.get("hard_issues") != []
        ):
            errors.append("最终质量报告 hard_issues 非空")

    syqon_exchange = run_manifest.load_json(
        work_dir / "process/stage6_syqon_exchange.json"
    )
    errors.extend(_validate_syqon_exchange(work_dir, syqon_exchange))

    if errors:
        return False, details + errors
    return True, details + [
        "verified_stages=1-10",
        "offline_syqon=accepted",
        "formal_delivery=accepted",
        "final_quality=ok",
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
