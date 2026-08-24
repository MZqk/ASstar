"""Plugin command and CLI fallback helpers for the Starun pipeline."""

from __future__ import annotations

import hashlib
import importlib
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from sirilpy.exceptions import CommandError, DataError, SirilError


ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_INHERITED_QT_ENV_KEYS = (
    "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "QT_QPA_PLATFORM",
    "QML2_IMPORT_PATH",
    "QML_IMPORT_PATH",
    "QTWEBENGINEPROCESS_PATH",
    "QTWEBENGINE_RESOURCES_PATH",
    "QTWEBENGINE_LOCALES_PATH",
    "QT_API",
)


def isolated_plugin_subprocess_env(source: dict[str, str]) -> dict[str, str]:
    """Return a headless Qt environment isolated from the frozen GUI."""
    env = source.copy()
    for key in _INHERITED_QT_ENV_KEYS:
        env.pop(key, None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def run_first_available_command(
    pipeline,
    step_key: str,
    candidates,
    allow_when_probe_disabled: bool = False,
):
    """
    在多个候选命令中按顺序执行第一个可用命令。
    candidates: List[(label, Tuple[arg1, arg2, ...])]
    """
    if (
        not pipeline.cfg.workflow_plugin_probe_enabled
        and not allow_when_probe_disabled
    ):
        return None
    for label, args in candidates:
        if pipeline._try_cmd(*args):
            pipeline.workflow_command_used[step_key] = label
            pipeline.log.info(f"{step_key} 使用命令: {label}")
            return label
    return None


def quote_siril_arg(pipeline, value: Path | str) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def resolve_siril_scripts_root(pipeline) -> Optional[Path]:
    if not pipeline.siril_plugin_dir:
        return None
    candidates = [
        pipeline.siril_plugin_dir / "vendor" / "siril-scripts",
        pipeline.siril_plugin_dir / "vendor" / "siril-scripts" / "siril-scripts",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        if (root / "processing").is_dir():
            return root
    return None


def find_plugin_script(pipeline, relative_candidates: Tuple[str, ...]) -> Optional[Path]:
    scripts_root = pipeline._resolve_siril_scripts_root()
    if scripts_root is None:
        return None
    for rel in relative_candidates:
        candidate = scripts_root / rel
        if candidate.is_file():
            return candidate
    return None


def is_python_module_available(pipeline, module_name: str) -> bool:
    existing = sys.modules.get(module_name)
    if existing is not None:
        # Treat in-memory PyQt6 stubs as unavailable for third-party scripts.
        if module_name == "PyQt6" and not getattr(existing, "__file__", None):
            return False
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def validate_plugin_script_prerequisites(
    pipeline,
    script_path: Path,
    python_executable: Optional[str] = None,
) -> Tuple[bool, str]:
    required_modules = pipeline._SCRIPT_PREREQUISITE_MODULES.get(script_path.name, ())
    if not required_modules:
        return True, ""

    if python_executable:
        probe = (
            "import importlib.util, sys\n"
            "missing=[m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]\n"
            "print(','.join(missing))\n"
            "raise SystemExit(1 if missing else 0)\n"
        )
        try:
            proc = subprocess.run(
                [python_executable, "-c", probe, *required_modules],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
            return False, (
                "python module prerequisite probe failed: "
                f"{pipeline._short_text(e, 160)}"
            )
        if proc.returncode == 0:
            return True, ""
        missing_text = (proc.stdout or "").strip()
        if not missing_text:
            missing_text = ", ".join(required_modules)
        return False, f"missing python modules: {missing_text}"

    missing = [
        module_name
        for module_name in required_modules
        if not pipeline._is_python_module_available(module_name)
    ]
    if not missing:
        return True, ""
    return False, f"missing python modules: {', '.join(missing)}"


def run_plugin_script_by_path(
    pipeline,
    step_key: str,
    label: str,
    script_path: Path,
    *,
    args: Tuple[str, ...] = (),
) -> Optional[str]:
    pipeline._last_plugin_script_error = None
    prerequisites_ok, prerequisites_reason = (
        pipeline._validate_plugin_script_prerequisites(script_path)
    )
    if not prerequisites_ok:
        pipeline._last_plugin_script_error = (
            f"{script_path.name}: {prerequisites_reason}"
        )
        pipeline.log.warn(
            f"{step_key} 脚本跳过 ({script_path.name}): {prerequisites_reason}"
        )
        return None

    cmd = ("pyscript", pipeline._quote_siril_arg(script_path), *args)
    before_fingerprint = pipeline._current_image_fingerprint()
    try:
        pipeline.cmd_with_check(*cmd, quiet=True)
        after_fingerprint = pipeline._current_image_fingerprint()
        if (
            before_fingerprint
            and after_fingerprint
            and before_fingerprint == after_fingerprint
        ):
            pipeline._last_plugin_script_error = (
                f"{script_path.name}: command returned success but image did not change"
            )
            pipeline.log.warn(
                f"{step_key} 脚本未产生图像变化，按失败处理 ({script_path.name})"
            )
            return None
        used_label = f"{label} script ({script_path.name})"
        pipeline.workflow_command_used[step_key] = used_label
        pipeline.log.info(f"{step_key} 使用脚本: {script_path.name}")
        return used_label
    except (CommandError, SirilError, DataError) as e:
        pipeline._last_plugin_script_error = (
            f"{script_path.name}: {pipeline._short_text(e, 160)}"
        )
        pipeline.log.warn(f"{step_key} 脚本执行失败 ({script_path.name}): {e}")
        return None


def current_image_fingerprint(pipeline) -> Optional[str]:
    try:
        # Plugin no-op detection does not need a second full-resolution FITS
        # transfer. Siril's preview is much smaller and a distributed sample
        # still detects processing changes deterministically.
        try:
            image_data = pipeline.siril.get_image_pixeldata(preview=True)
        except (TypeError, ValueError):
            image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            return None
        arr = np.asarray(image_data)
        digest = hashlib.sha256()
        digest.update(str(arr.shape).encode("ascii", errors="ignore"))
        digest.update(str(arr.dtype).encode("ascii", errors="ignore"))
        flat = arr.reshape(-1)
        sample_count = min(flat.size, 65536)
        if sample_count:
            if flat.size > sample_count:
                indices = np.linspace(
                    0, flat.size - 1, num=sample_count, dtype=np.int64
                )
                sample = flat[indices]
            else:
                sample = flat
            digest.update(np.ascontiguousarray(sample).view(np.uint8))
        return digest.hexdigest()
    except (AttributeError, TypeError, ValueError, IndexError, FloatingPointError) as e:
        pipeline.log.debug(f"图像指纹采样跳过: {e}")
        return None


def plugin_output_failure_reason(pipeline, script_name: str, output_text: str) -> Optional[str]:
    lowered = output_text.lower()
    failure_markers = (
        "modulenotfounderror:",
        "traceback (most recent call last):",
        "cosmic clarity process failed",
        "cosmic clarity sharpening process failed",
        "could not send output to siril:",
        "error:",
    )
    if not any(marker in lowered for marker in failure_markers):
        return None

    important_lines: List[str] = []
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        line_lower = line.lower()
        if (
            "modulenotfounderror:" in line_lower
            or "cosmic clarity" in line_lower and "failed" in line_lower
            or "could not send output to siril:" in line_lower
            or line_lower.startswith("error:")
            or "traceback (most recent call last):" in line_lower
        ):
            important_lines.append(line)
    if not important_lines:
        important_lines = [
            line.strip()
            for line in output_text.splitlines()
            if line.strip()
        ][-4:]
    return f"{script_name}: " + pipeline._short_text("; ".join(important_lines), 260)


def fallback_summary(
    pipeline,
    failed_component: str,
    failure_reason: str,
    fallback_component: str,
    fallback_succeeded: bool,
) -> str:
    outcome = "success" if fallback_succeeded else "failed"
    reason = pipeline._short_text(failure_reason, 360)
    return (
        f"fallback: failed_component={failed_component}; "
        f"reason={reason}; fallback_component={fallback_component}; "
        f"fallback_status={outcome}"
    )


def is_classic_cc_not_configured(pipeline, reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return (
        text == "cosmicclarity executable not configured"
        or text == "cosmic clarity executable not configured"
        or text == "cosmicclarity classic disabled; using native path"
        or text == "cosmic clarity classic disabled; using native path"
    )


def is_siril_connection_failure(pipeline, value: object) -> bool:
    text = str(value or "").lower()
    return (
        "failed to connect to siril" in text
        or "sirilconnectionerror" in text
        or "connection refused" in text
        or "connection reset" in text
    )


def subprocess_output_tail(pipeline, output_text: str, max_lines: int = 12) -> str:
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-max_lines:])


def _run_subprocess_with_group_timeout(
    cmd: List[str],
    *,
    timeout: int,
    cwd: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run a plugin in its own POSIX process group and clean up descendants."""
    start_new_session = os.name == "posix"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=start_new_session,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if start_new_session:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            if start_new_session:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.communicate()
        raise exc
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout)


def run_plugin_script_cli_subprocess(
    pipeline,
    step_key: str,
    label: str,
    script_path: Path,
    *,
    args: Tuple[str, ...] = (),
    timeout_sec: int = 1800,
    verify_image_change: bool = False,
    uses_siril_connection: bool = True,
) -> Optional[str]:
    """
    以外部 Python 子进程调用脚本 CLI 模式（不走 Siril pyscript GUI 路径）。
    """
    pipeline._last_plugin_script_error = None
    python_cli = ""
    for python_cli_env_key in (
        "STARUN_SIRIL_PYTHON_CLI",
        "SIRIL_PYTHON_CLI",
    ):
        raw_python_cli = os.getenv(python_cli_env_key, "").strip()
        if not raw_python_cli:
            continue
        lowered = raw_python_cli.lower()
        if lowered in ENV_TRUE_VALUES or lowered in ENV_FALSE_VALUES:
            continue
        candidate_path = Path(raw_python_cli).expanduser()
        if candidate_path.exists() and candidate_path.is_file():
            python_cli = str(candidate_path)
            break
        resolved = shutil.which(raw_python_cli)
        if resolved:
            python_cli = resolved
            break
        pipeline.log.warn(
            f"{python_cli_env_key} is not executable; "
            f"ignore value: {raw_python_cli}"
        )

    if not python_cli:
        sys_exec = (sys.executable or "").strip()
        if sys_exec and Path(sys_exec).exists():
            python_cli = sys_exec
        else:
            python_cli = shutil.which("python3") or shutil.which("python") or ""

    if python_cli:
        os.environ["SIRIL_PYTHON_CLI"] = python_cli
        os.environ["STARUN_SIRIL_PYTHON_CLI"] = python_cli

    if not python_cli:
        pipeline._last_plugin_script_error = (
            f"{script_path.name}: no valid python executable found for CLI subprocess"
        )
        pipeline.log.warn(f"{step_key} CLI 子进程失败: 无可用 Python 解释器")
        return None

    prerequisites_ok, prerequisites_reason = (
        pipeline._validate_plugin_script_prerequisites(script_path, python_cli)
    )
    if not prerequisites_ok:
        pipeline._last_plugin_script_error = (
            f"{script_path.name}: {prerequisites_reason}"
        )
        pipeline.log.warn(
            f"{step_key} CLI 子进程跳过 ({script_path.name}): {prerequisites_reason}"
        )
        return None

    cmd = [python_cli, str(script_path), *args]
    before_fingerprint = pipeline._current_image_fingerprint() if verify_image_change else None

    env = isolated_plugin_subprocess_env(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    raw_timeout = str(env.get("STARUN_SIRILPY_TIMEOUT_SEC", "")).strip()
    if not raw_timeout:
        env["STARUN_SIRILPY_TIMEOUT_SEC"] = "300"
    env["SIRIL_PYTHON_CLI"] = python_cli
    env["STARUN_SIRIL_PYTHON_CLI"] = python_cli
    cwd = str(pipeline.process_dir or pipeline.work_dir or Path.cwd())

    pipeline.log.info(f"{step_key} 使用 CLI 子进程: {script_path.name}")
    pipeline.log.debug(f"{step_key} CLI 命令: {' '.join(cmd)}")
    parent_was_connected = bool(
        uses_siril_connection and getattr(pipeline.siril, "connected", False)
    )
    log_sink_paused = False
    if parent_was_connected:
        try:
            set_log_sink = getattr(pipeline.log, "set_sink", None)
            if callable(set_log_sink):
                # The parent Siril connection is intentionally unavailable while
                # the child owns it. Keep heartbeat/status logs file-only until
                # the parent reconnects, otherwise the Siril sink marks the
                # expected disconnect as a native-process failure.
                set_log_sink(None)
                log_sink_paused = True
            pipeline.siril.disconnect()
            pipeline.log.debug(
                f"{step_key} CLI 子进程前已临时释放 Siril 连接"
            )
        except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
            if log_sink_paused and bool(
                getattr(pipeline.siril, "connected", False)
            ):
                pipeline.log.set_sink(pipeline.siril.log)
            pipeline._last_plugin_script_error = (
                f"{script_path.name}: failed to release parent Siril "
                f"connection before CLI subprocess: {pipeline._short_text(e, 160)}"
            )
            pipeline.log.warn(
                f"{step_key} CLI 子进程启动前释放 Siril 连接失败: {e}"
            )
            return None

    stop_heartbeat = threading.Event()
    heartbeat_interval = 30.0
    timeout_value = max(1, int(timeout_sec))

    def _heartbeat() -> None:
        started = time.monotonic()
        while not stop_heartbeat.wait(heartbeat_interval):
            elapsed = int(time.monotonic() - started)
            pipeline.log.info(
                f"{step_key} CLI 子进程仍在运行: {script_path.name} "
                f"(elapsed={elapsed}s, timeout={timeout_value}s)"
            )

    reconnect_failed = False
    try:
        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            proc = _run_subprocess_with_group_timeout(
                cmd,
                timeout=timeout_value,
                cwd=cwd,
                env=env,
            )
        except subprocess.TimeoutExpired:
            pipeline._last_plugin_script_error = (
                f"{script_path.name}: subprocess timeout after {timeout_sec}s"
            )
            pipeline.log.warn(
                f"{step_key} CLI 子进程超时 ({script_path.name}): {timeout_sec}s"
            )
            return None
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            pipeline._last_plugin_script_error = (
                f"{script_path.name}: subprocess error: {pipeline._short_text(e, 160)}"
            )
            pipeline.log.warn(f"{step_key} CLI 子进程异常 ({script_path.name}): {e}")
            return None
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1.0)
    finally:
        if parent_was_connected:
            try:
                pipeline.siril.connect()
                if log_sink_paused:
                    pipeline.log.set_sink(pipeline.siril.log)
                pipeline.log.debug(
                    f"{step_key} CLI 子进程后已恢复 Siril 连接"
                )
            except (CommandError, DataError, SirilError, OSError, RuntimeError) as e:
                reconnect_failed = True
                pipeline._last_plugin_script_error = (
                    f"{script_path.name}: failed to reconnect parent Siril "
                    f"after CLI subprocess: {pipeline._short_text(e, 160)}"
                )
                pipeline.log.warn(
                    f"{step_key} CLI 子进程后恢复 Siril 连接失败: {e}"
                )

    if reconnect_failed:
        return None

    output_text = proc.stdout or ""
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if line:
            pipeline.log.info(f"[{script_path.name}] {line}")

    output_failure = pipeline._plugin_output_failure_reason(
        script_path.name,
        output_text,
    )
    if output_failure:
        pipeline._last_plugin_script_error = output_failure
        pipeline.log.warn(
            f"{step_key} CLI 子进程输出包含失败信息 ({script_path.name}): "
            f"{pipeline._short_text(output_failure, 220)}"
        )
        return None

    if proc.returncode != 0:
        output_tail = pipeline._subprocess_output_tail(output_text)
        tail_suffix = f"; output_tail={output_tail}" if output_tail else ""
        pipeline._last_plugin_script_error = (
            f"{script_path.name}: subprocess exited with code {proc.returncode}"
            f"{tail_suffix}"
        )
        pipeline.log.warn(
            f"{step_key} CLI 子进程失败 ({script_path.name}): exit={proc.returncode}"
        )
        return None

    if verify_image_change:
        after_fingerprint = pipeline._current_image_fingerprint()
        if (
            before_fingerprint
            and after_fingerprint
            and before_fingerprint == after_fingerprint
        ):
            pipeline._last_plugin_script_error = (
                f"{script_path.name}: subprocess returned success but image did not change"
            )
            pipeline.log.warn(
                f"{step_key} CLI 子进程未产生图像变化，按失败处理 ({script_path.name})"
            )
            return None

    used_label = f"{label} cli-subprocess ({script_path.name})"
    pipeline.workflow_command_used[step_key] = used_label
    pipeline.log.info(f"{step_key} CLI 子进程成功: {script_path.name}")
    return used_label
