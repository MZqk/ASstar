#!/usr/bin/env python3
"""Pipeline worker for the Seestar Superimpose GUI."""

from __future__ import annotations

import os
import platform
import queue
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

try:
    from .common import *
except ImportError:
    from common import *  # type: ignore[no-redef]

try:
    from .constants import FITS_SUFFIXES
    from .disk_preflight import format_bytes, safe_file_size
except ImportError:
    from constants import FITS_SUFFIXES  # type: ignore[no-redef]
    from disk_preflight import format_bytes, safe_file_size  # type: ignore[no-redef]


BOOTSTRAP_TIMEOUT_ENV = "SEESTAR_BOOTSTRAP_TIMEOUT_SEC"
DEFAULT_BOOTSTRAP_TIMEOUT_SEC = 300
MIN_BOOTSTRAP_TIMEOUT_SEC = 60
MAX_BOOTSTRAP_TIMEOUT_SEC = 3600
BOOTSTRAP_TIMEOUT_SEC_PER_GIB = 120
BYTES_PER_GIB = 1024 * 1024 * 1024
TEMP_CLEANUP_TIMEOUT_ENV = "SEESTAR_TEMP_CLEANUP_TIMEOUT_SEC"
DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC = 30
MIN_TEMP_CLEANUP_TIMEOUT_SEC = 1
MAX_TEMP_CLEANUP_TIMEOUT_SEC = 300


class PipelineWorker(QThread):
    """Runs Siril processing in a worker thread via siril-cli subprocess."""

    log = Signal(str)
    state = Signal(str)
    done = Signal(str, int, bool, str)

    def __init__(
        self,
        work_dir: Path,
        config_template: Path,
        pipeline_path: Path,
        siril_plugin_dir: Path,
        resources: Path,
        runtime_home: Path,
        siril_candidates: list[Path],
        input_mode: str = INPUT_MODE_AUTO,
        debug_mode: bool = False,
        network_mode: bool = True,
        ai_stage_enabled: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.work_dir = work_dir
        self.config_template = config_template
        self.pipeline_path = pipeline_path
        self.siril_plugin_dir = siril_plugin_dir
        self.resources = resources
        self.runtime_home = runtime_home
        self.siril_candidates = siril_candidates
        self.input_mode = (
            input_mode
            if input_mode in {
                INPUT_MODE_AUTO,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }
            else INPUT_MODE_AUTO
        )
        self.debug_mode = bool(debug_mode)
        self.network_mode = bool(network_mode)
        self.ai_stage_enabled = bool(ai_stage_enabled)

        self._stop_event = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._active_mode = "python"
        self._run_had_errors = False
        self._last_output_ts = 0.0
        self._pyscript_seen_at: float | None = None
        self._pipeline_output_seen = False
        self._python_env_issue = False
        self._python_env_repair_attempted = False
        self._spcc_seen_in_run = False
        self._spcc_cli_crash_detected = False
        self._spcc_crash_retry_attempted = False
        self._force_disable_spcc_for_retry = False
        self._recent_process_output: deque[str] = deque(maxlen=80)
        self._last_spcc_command = ""
        self._ai_env_sources: list[str] = []
        self._ai_env_applied_keys: list[str] = []
        self._ai_env_warnings: list[str] = []
        self._runtime_plugin_dir: Path | None = None
        self._temp_cleanup_timeout_sec = DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC

    def stop(self) -> None:
        self._stop_event.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_event(self, msg: str) -> None:
        self.log.emit(f"[{self._timestamp()}] {msg}\n")

    def _ai_env_candidates(self) -> list[Path]:
        return [
            self.resources / DEFAULT_ENV_RESOURCE_REL,
            self.resources / AI_ENV_RESOURCE_REL,
            self.runtime_home / AI_ENV_OVERRIDE_NAME,
            self.work_dir / AI_ENV_OVERRIDE_NAME,
        ]

    def _load_ai_env_overrides(self) -> tuple[dict[str, str], list[str], list[str]]:
        merged: dict[str, str] = {}
        sources: list[str] = []
        warnings: list[str] = []

        for path in self._ai_env_candidates():
            if not path.exists() or not path.is_file():
                continue
            parsed, parse_warnings = parse_ai_env_file(path)
            sources.append(str(path))
            merged.update(parsed)
            warnings.extend(parse_warnings)

        return merged, sources, warnings

    def _inspect_output_for_errors(self, text: str) -> None:
        lowered = text.lower()
        stripped = text.strip()
        if stripped:
            self._recent_process_output.append(stripped)
        error_markers = (
            "script execution failed",
            "failed to install python module",
            "python not ready yet",
            "python validation failed",
            "python version check failed",
            "failed to initialize python virtual environment",
            "unable to spawn python",
            "error in line",
            "unknown error",
            "exiting batch processing",
        )
        python_env_markers = (
            "failed to install python module",
            "python not ready yet",
            "unable to install or update the siril python module",
            "failed to execute pip",
            "pip command failed",
            "python validation failed",
            "python version check failed",
            "failed to initialize python virtual environment",
            "error finding venv python path",
            "unable to spawn python",
            "failed to create python connection",
            "failed to execute python script",
        )
        if any(marker in lowered for marker in error_markers):
            self._run_had_errors = True
        if any(marker in lowered for marker in python_env_markers):
            self._python_env_issue = True
            # Treat Python environment problems as hard pipeline errors even when
            # siril-cli exits 0, so caller can retry/reset/fallback correctly.
            self._run_had_errors = True
        if (
            "running command: spcc" in lowered
            or "running command spcc" in lowered
            or "input command:spcc" in lowered
            or "input command: spcc" in lowered
        ):
            self._spcc_seen_in_run = True
            if "spcc" in lowered:
                self._last_spcc_command = stripped
        if (
            ("running command: pyscript" in lowered or "running command pyscript" in lowered)
            and self._pyscript_seen_at is None
        ):
            self._pyscript_seen_at = time.time()
        if self._pyscript_seen_at is not None and stripped:
            # Any non-empty output after entering pyscript means the pipeline is alive.
            if "running command: pyscript" not in lowered and "running command pyscript" not in lowered:
                self._pipeline_output_seen = True
        pipeline_markers = ("stage 1", "阶段 1", "[info]")
        if any(marker in lowered for marker in pipeline_markers):
            self._pipeline_output_seen = True

    def _append_spcc_crash_diagnostics(self, exit_code: int) -> None:
        self._append_event(
            "SPCC 崩溃诊断: siril-cli 在执行 SPCC 后以 "
            f"{exit_code} 退出，通常表示 Siril 原生测光/SPCC 代码段错误，"
            "Python 侧不会产生 CommandError。"
        )
        if self._last_spcc_command:
            self._append_event(f"SPCC 崩溃前命令标记: {self._last_spcc_command}")
        if self._recent_process_output:
            self._append_event("SPCC 崩溃前最后输出（最多 25 行）:")
            for line in list(self._recent_process_output)[-25:]:
                self._append_event(f"  {line}")

    def _siril_venv_dir(self) -> Path:
        return self._siril_state_root() / "venv"

    def _siril_state_root(self) -> Path:
        return siril_state_root_from_home(self.runtime_home)

    def _restore_offline_siril_seed(self) -> bool:
        state_root = self._siril_state_root()
        venv_dir = state_root / "venv"
        module_dir = state_root / ".python_module"
        seed_root = self.resources / "SirilPythonSeed"
        seed_venv = seed_root / "venv"
        seed_module = seed_root / ".python_module"

        if not seed_venv.exists() or not seed_module.exists():
            self._append_event(
                f"应用资源中缺少离线 Siril seed：{seed_root}"
            )
            return False

        try:
            state_root.mkdir(parents=True, exist_ok=True)
            if not venv_dir.exists():
                shutil.copytree(seed_venv, venv_dir, symlinks=True)
                self._append_event(f"已从离线 seed 恢复 Siril venv：{venv_dir}")
            if not module_dir.exists():
                shutil.copytree(seed_module, module_dir, symlinks=True)
                self._append_event(
                    f"已从离线 seed 恢复 Siril Python 模块：{module_dir}"
                )
            elif not (module_dir / "sirilpy").exists():
                shutil.rmtree(module_dir, ignore_errors=True)
                shutil.copytree(seed_module, module_dir, symlinks=True)
                self._append_event(
                    f"已重新从离线 seed 恢复 Siril Python 模块：{module_dir}"
                )

            # Rewrite venv interpreter links/config to current bundled Siril path.
            py_bin = (
                self.resources
                / "Siril.app"
                / "Contents"
                / "Frameworks"
                / "Python.framework"
                / "Versions"
                / "3.12"
                / "bin"
                / "python3.12"
            )
            if not py_bin.exists():
                self._append_event(f"内置 Siril Python 缺失：{py_bin}")
                return False

            bin_dir = venv_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            for name in ("python3.12", "python3", "python"):
                dst = bin_dir / name
                if dst.exists() or dst.is_symlink():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                dst.symlink_to(py_bin)

            cfg = venv_dir / "pyvenv.cfg"
            cfg_lines: list[str] = []
            if cfg.exists():
                cfg_lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
            siril_python = self.resources / "Siril.app" / "Contents" / "MacOS" / "python3"
            replacements = {
                "home": str(py_bin.parent),
                "executable": str(py_bin),
                "command": f"{siril_python} -m venv {venv_dir}",
            }
            seen: set[str] = set()
            out_lines: list[str] = []
            for line in cfg_lines:
                if "=" not in line:
                    out_lines.append(line)
                    continue
                key, _value = line.split("=", 1)
                k = key.strip()
                if k in replacements:
                    out_lines.append(f"{k} = {replacements[k]}")
                    seen.add(k)
                else:
                    out_lines.append(line)
            for k, v in replacements.items():
                if k not in seen:
                    out_lines.append(f"{k} = {v}")
            cfg.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

            site_dir = resolve_venv_site_packages(venv_dir)
            repaired = repair_site_packages_from_pip_vendor(site_dir)
            if repaired:
                self._append_event(
                    "已从 pip vendor 修补 Siril venv 依赖："
                    + ", ".join(repaired)
                )
            ok, detail = verify_siril_offline_seed_venv(venv_dir)
            if not ok:
                self._append_event(f"离线 Siril seed 校验失败：{detail}")
                return False
            self._append_event(f"离线 Siril seed 校验通过：{detail}")

            return True
        except Exception as e:
            self._append_event(f"恢复离线 Siril seed 失败：{e}")
            return False

    def _process_identity(self, pid: int) -> tuple[str, str] | None:
        try:
            cp = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "lstart=,command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return None
            fields = cp.stdout.strip().split(maxsplit=5)
            if len(fields) != 6:
                return None
            return " ".join(fields[:5]), fields[5]
        except Exception:
            return None

    def _collect_processes(
        self,
        needle_parts: tuple[str, ...],
    ) -> list[tuple[int, str, str]]:
        matches: list[tuple[int, str, str]] = []
        try:
            cp = subprocess.run(
                ["/bin/ps", "-ax", "-o", "pid=,lstart=,command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return matches
            for line in cp.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                fields = line.split(maxsplit=6)
                if len(fields) != 7:
                    continue
                pid_txt = fields[0]
                start_time = " ".join(fields[1:6])
                command = fields[6]
                if not pid_txt.isdigit():
                    continue
                pid = int(pid_txt)
                if pid <= 0 or pid == os.getpid():
                    continue
                if all(part in command for part in needle_parts):
                    matches.append((pid, start_time, command))
        except Exception:
            return matches
        return matches

    def _terminate_processes(self, procs: list[tuple[int, str, str]]) -> int:
        if not procs:
            return 0
        terminated = 0
        unique_procs = {pid: (start_time, command) for pid, start_time, command in procs}
        for pid, expected_identity in unique_procs.items():
            if self._process_identity(pid) != expected_identity:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                terminated += 1
            except Exception:
                continue
        time.sleep(1.0)
        for pid, expected_identity in unique_procs.items():
            if self._process_identity(pid) != expected_identity:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                continue
        return terminated

    def _reset_siril_python_venv(self) -> bool:
        venv_dir = self._siril_venv_dir()
        python_token = str(venv_dir / "bin/python")
        pip_token = str(venv_dir / "bin/pip")
        venv_target = str(venv_dir)

        # Terminate stale venv python/pip processes first.
        stale = self._collect_processes((python_token,))
        stale += self._collect_processes((pip_token,))
        stale += self._collect_processes(("python3 -m venv", venv_target))
        if stale:
            terminated = self._terminate_processes(stale)
            self._append_event(
                f"已终止 {terminated} 个残留的 Siril Python/venv 进程。"
            )

        if not venv_dir.exists():
            self._append_event(
                "Siril Python venv 不存在，开始恢复离线 seed。"
            )
            return self._restore_offline_siril_seed()

        try:
            shutil.rmtree(venv_dir)
            self._append_event(f"已删除 Siril Python venv：{venv_dir}")
            return self._restore_offline_siril_seed()
        except Exception as e:
            self._append_event(f"删除 Siril Python venv 失败：{e}")
            return False

    def _reader(self, stream, out_queue: queue.Queue[str | None]) -> None:
        try:
            for line in iter(stream.readline, ""):
                out_queue.put(line)
        finally:
            out_queue.put(None)

    def _prepare_runtime_files(self, temp_dir: Path) -> tuple[Path, Path, Path]:
        run_ssf = temp_dir / "run_job_embedded.ssf"
        run_ini = temp_dir / "config.1.4.ini"
        run_py = temp_dir / self.pipeline_path.name
        pipeline_dir = self.pipeline_path.parent
        stage11_module_path = self.pipeline_path.with_name("stage11_ai_postprocess.py")
        run_stage11_module = temp_dir / stage11_module_path.name
        cpu_limit = compute_siril_cpu_limit()

        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"未找到流水线脚本：{self.pipeline_path}")
        for py_file in pipeline_dir.glob("*.py"):
            shutil.copy2(py_file, temp_dir / py_file.name)
        stages_dir = pipeline_dir / "stages"
        if stages_dir.exists() and stages_dir.is_dir():
            shutil.copytree(stages_dir, temp_dir / "stages", dirs_exist_ok=True)
        if not stage11_module_path.exists():
            raise FileNotFoundError(f"未找到 Stage11 模块脚本：{stage11_module_path}")
        if not run_stage11_module.exists():
            shutil.copy2(stage11_module_path, run_stage11_module)

        self._runtime_plugin_dir = None
        if self.siril_plugin_dir.exists() and self.siril_plugin_dir.is_dir():
            plugin_dst = temp_dir / "siril_plugins"
            shutil.copytree(self.siril_plugin_dir, plugin_dst, dirs_exist_ok=True)
            if apply_siril_runtime_patches(plugin_dst):
                self._append_event("已应用 GraXpert-AI 运行时兼容补丁")
            self._runtime_plugin_dir = plugin_dst

        ssf_lines = [
            "requires 1.4.0",
        ]
        if cpu_limit is not None:
            ssf_lines.append(f"setcpu {cpu_limit}")
        ssf_lines.extend(
            [
                f'cd "{shell_quote_path(self.work_dir)}"',
                f'pyscript "{shell_quote_path(run_py)}"',
                "close",
            ]
        )
        run_ssf.write_text("\n".join(ssf_lines) + "\n", encoding="utf-8")

        template_text = self.config_template.read_text(
            encoding="utf-8", errors="replace"
        )
        patched = normalize_siril_config_template(template_text)
        run_ini.write_text(patched, encoding="utf-8")
        return run_ssf, run_ini, run_py

    def _build_env(self, siril_cli: Path) -> dict[str, str]:
        env = scrub_python_env(os.environ.copy())
        ai_env, ai_sources, ai_warnings = self._load_ai_env_overrides()
        applied_keys: list[str] = []
        for key, value in ai_env.items():
            if not env.get(key):
                env[key] = value
                applied_keys.append(key)
        self._ai_env_sources = ai_sources
        self._ai_env_applied_keys = sorted(applied_keys)
        self._ai_env_warnings = ai_warnings

        # Finder-launched apps may lack UTF-8 locale vars.
        env["HOME"] = str(self.runtime_home)
        env["LANG"] = "en_US.UTF-8"
        env["LC_ALL"] = "en_US.UTF-8"
        env["LC_CTYPE"] = "en_US.UTF-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault(
            BOOTSTRAP_TIMEOUT_ENV,
            str(DEFAULT_BOOTSTRAP_TIMEOUT_SEC),
        )
        env.setdefault(
            TEMP_CLEANUP_TIMEOUT_ENV,
            str(DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC),
        )
        try:
            cleanup_timeout = int(round(float(env[TEMP_CLEANUP_TIMEOUT_ENV])))
        except (OverflowError, TypeError, ValueError):
            cleanup_timeout = DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC
            self._append_event(
                f"{TEMP_CLEANUP_TIMEOUT_ENV}={env[TEMP_CLEANUP_TIMEOUT_ENV]!r} 无效，"
                f"使用默认值 {DEFAULT_TEMP_CLEANUP_TIMEOUT_SEC}s。"
            )
        self._temp_cleanup_timeout_sec = max(
            MIN_TEMP_CLEANUP_TIMEOUT_SEC,
            min(MAX_TEMP_CLEANUP_TIMEOUT_SEC, cleanup_timeout),
        )
        env.setdefault("SEESTAR_SIRILPY_TIMEOUT_SEC", "120")
        env["PIP_NO_INDEX"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        pip_find_links: list[str] = []
        bundled_downloads = self.resources / "siril_plugins" / "downloads"
        if bundled_downloads.is_dir():
            pip_find_links.append(str(bundled_downloads))
        if self._runtime_plugin_dir:
            runtime_downloads = self._runtime_plugin_dir / "downloads"
            if runtime_downloads.is_dir():
                pip_find_links.append(str(runtime_downloads))
        if pip_find_links:
            env["PIP_FIND_LINKS"] = " ".join(dict.fromkeys(pip_find_links))
        bundled_py = (
            self.resources
            / "Siril.app"
            / "Contents"
            / "Frameworks"
            / "Python.framework"
            / "Versions"
            / "3.12"
            / "bin"
            / "python3.12"
        )
        if bundled_py.exists():
            env["SIRIL_PYTHON_CLI"] = str(bundled_py)
            env["SEESTAR_SIRIL_PYTHON_CLI"] = str(bundled_py)

        bundled_siril_cli = self.resources / "Siril.app" / "Contents" / "MacOS" / "siril-cli"
        if siril_cli == bundled_siril_cli:
            relocated = self.resources / "Siril.app" / "Contents" / "Resources"
            env["SIRIL_RELOCATED_RES_DIR"] = str(relocated)

        if self._runtime_plugin_dir and self._runtime_plugin_dir.exists():
            env["SEESTAR_SIRIL_PLUGIN_DIR"] = str(self._runtime_plugin_dir)
            classic_wrapper = self._runtime_plugin_dir / "bin" / "CosmicClarity"
            if classic_wrapper.is_file() and os.access(classic_wrapper, os.X_OK):
                env.setdefault("SEESTAR_COSMIC_CLARITY_EXECUTABLE", str(classic_wrapper))
            scripts_dir = resolve_siril_scripts_root(self._runtime_plugin_dir)
            if scripts_dir is not None:
                env["SIRIL_SCRIPTS_DIR"] = str(scripts_dir)
                env["SIRIL_SCRIPTS_PATH"] = str(scripts_dir)

        env["SEESTAR_DEBUG_MODE"] = "1" if self.debug_mode else "0"
        env["SEESTAR_INPUT_MODE"] = self.input_mode
        # GUI toggle is the highest-priority control for optional stage11 execution.
        env["SEESTAR_AI_ENABLED"] = "1" if self.ai_stage_enabled else "0"
        if self._force_disable_spcc_for_retry:
            env["SEESTAR_SPCC_ENABLE"] = "0"
        return env

    def _bootstrap_input_size_bytes(self) -> int:
        if not self.work_dir.is_dir():
            return 0
        total = 0
        try:
            candidates = self.work_dir.iterdir()
            for path in candidates:
                if path.is_file() and path.suffix.lower() in FITS_SUFFIXES:
                    total += safe_file_size(path)
        except OSError:
            return total
        return total

    def _bootstrap_timeout_sec(
        self,
        env: dict[str, str] | None = None,
    ) -> tuple[int, int, int]:
        source_env = env if env is not None else os.environ
        raw_value = source_env.get(
            BOOTSTRAP_TIMEOUT_ENV,
            str(DEFAULT_BOOTSTRAP_TIMEOUT_SEC),
        )
        try:
            base_timeout = int(round(float(raw_value)))
        except (OverflowError, TypeError, ValueError):
            base_timeout = DEFAULT_BOOTSTRAP_TIMEOUT_SEC
            self._append_event(
                f"{BOOTSTRAP_TIMEOUT_ENV}={raw_value!r} 无效，"
                f"使用默认值 {DEFAULT_BOOTSTRAP_TIMEOUT_SEC}s。"
            )
        base_timeout = max(
            MIN_BOOTSTRAP_TIMEOUT_SEC,
            min(MAX_BOOTSTRAP_TIMEOUT_SEC, base_timeout),
        )

        input_bytes = self._bootstrap_input_size_bytes()
        adaptive_extra = (
            input_bytes * BOOTSTRAP_TIMEOUT_SEC_PER_GIB + BYTES_PER_GIB - 1
        ) // BYTES_PER_GIB
        effective_timeout = min(
            MAX_BOOTSTRAP_TIMEOUT_SEC,
            base_timeout + adaptive_extra,
        )
        return effective_timeout, base_timeout, input_bytes

    def _cleanup_temp_dir(
        self,
        temp_dir: Path,
        timeout_sec: float | None = None,
    ) -> bool:
        timeout = (
            float(self._temp_cleanup_timeout_sec)
            if timeout_sec is None
            else max(0.0, float(timeout_sec))
        )
        errors: list[Exception] = []
        removed = threading.Event()

        def remove() -> None:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                removed.set()
            except Exception as exc:
                errors.append(exc)
            else:
                removed.set()

        cleanup_thread = threading.Thread(
            target=remove,
            name="seestar-temp-cleanup",
            daemon=True,
        )
        cleanup_thread.start()
        cleanup_thread.join(timeout)

        if cleanup_thread.is_alive():
            self._append_event(
                f"临时目录清理超过 {timeout:g}s，已转为后台清理：{temp_dir}"
            )
            return False
        if errors:
            self._append_event(f"临时目录清理失败，已保留：{temp_dir} ({errors[0]})")
            return False
        if not removed.is_set():
            self._append_event(f"临时目录清理异常中止，已保留：{temp_dir}")
            return False
        return True

    def _run_once(self, siril_cli: Path, run_ssf: Path, run_ini: Path) -> tuple[bool, int]:
        self._run_had_errors = False
        self._python_env_issue = False
        self._pyscript_seen_at = None
        self._pipeline_output_seen = False
        self._spcc_seen_in_run = False
        self._spcc_cli_crash_detected = False
        self._recent_process_output.clear()
        self._last_spcc_command = ""
        self._last_output_ts = time.time()

        cmd = build_siril_cli_command(
            siril_cli=siril_cli,
            work_dir=self.work_dir,
            run_ini=run_ini,
            run_ssf=run_ssf,
            offline_mode=not self.network_mode,
        )
        self._append_event(f"开始启动进程（{self._active_mode}），使用 {siril_cli}")
        self._append_event("命令：" + " ".join(cmd))
        proc_env = self._build_env(siril_cli)
        self._append_event(f"Siril 运行时主目录：{proc_env.get('HOME', '')}")
        self._append_event(
            "Siril Python CLI："
            + proc_env.get("SIRIL_PYTHON_CLI", "<未设置>")
        )
        self._append_event(
            f"调试模式: {'ON' if self.debug_mode else 'OFF'}"
        )
        self._append_event(
            f"联网模式: {'ON' if self.network_mode else 'OFF'}"
        )
        self._append_event(f"输入模式: {self.input_mode}")
        self._append_event(
            f"AI 阶段开关: {'ON' if self.ai_stage_enabled else 'OFF'} "
            "(controls stage11)"
        )
        bootstrap_timeout_sec, bootstrap_base_sec, bootstrap_input_bytes = (
            self._bootstrap_timeout_sec(proc_env)
        )
        self._append_event(
            "pyscript bootstrap 超时: "
            f"{bootstrap_timeout_sec}s "
            f"(base={bootstrap_base_sec}s, FITS={format_bytes(bootstrap_input_bytes)}, "
            f"rate={BOOTSTRAP_TIMEOUT_SEC_PER_GIB}s/GiB)"
        )
        if self._ai_env_sources:
            self._append_event(
                "AI 环境配置来源: " + ", ".join(self._ai_env_sources)
            )
            if self._ai_env_applied_keys:
                self._append_event(
                    "AI 环境已注入键: " + ", ".join(self._ai_env_applied_keys)
                )
        for warning in self._ai_env_warnings:
            self._append_event(f"AI 环境配置警告: {warning}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env,
            )
        except Exception as e:
            self._append_event(f"启动进程失败：{e}")
            self._run_had_errors = True
            return False, -1

        if self._proc.stdout is None:
            self._append_event("无法捕获进程输出。")
            self._run_had_errors = True
            return False, -1

        out_queue: queue.Queue[str | None] = queue.Queue()
        reader_t = threading.Thread(target=self._reader, args=(self._proc.stdout, out_queue), daemon=True)
        reader_t.start()

        bootstrap_timeout = False
        reader_done = False

        while True:
            drained = False
            while True:
                try:
                    item = out_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if item is None:
                    reader_done = True
                    break
                self._last_output_ts = time.time()
                self._inspect_output_for_errors(item)
                self.log.emit(item)

            proc_ret = self._proc.poll()

            if self._stop_event.is_set() and proc_ret is None:
                self._append_event("已请求停止...")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._append_event("进程未能正常退出，正在强制结束。")
                    self._proc.kill()

            now = time.time()
            if (
                proc_ret is None
                and self._active_mode == "python"
                and self._pyscript_seen_at
                and not self._pipeline_output_seen
                and now - self._pyscript_seen_at > bootstrap_timeout_sec
            ):
                bootstrap_timeout = True
                self._run_had_errors = True
                self._python_env_issue = True
                self._append_event(
                    "pyscript 启动超时"
                    f"（>{bootstrap_timeout_sec}s）：Siril Python 环境疑似卡住。"
                )
                self._append_event(
                    "提示：关闭 Siril，删除 "
                    f"'{self._siril_venv_dir()}' 后重试。"
                )
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

            proc_ret = self._proc.poll()
            if proc_ret is not None and reader_done and out_queue.empty():
                break

            if not drained:
                time.sleep(0.1)

        # Drain remaining output after process exit.
        while True:
            try:
                item = out_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            self._inspect_output_for_errors(item)
            self.log.emit(item)

        try:
            self._proc.stdout.close()
        except Exception:
            pass
        reader_t.join(timeout=1)

        exit_code = self._proc.returncode if self._proc.returncode is not None else -1
        self._proc = None

        if bootstrap_timeout:
            return False, exit_code

        if exit_code == -11 and self._spcc_seen_in_run:
            self._spcc_cli_crash_detected = True
            self._run_had_errors = True
            self._append_spcc_crash_diagnostics(exit_code)

        if (
            self._active_mode == "python"
            and self._pyscript_seen_at is not None
            and not self._pipeline_output_seen
            and not self._stop_event.is_set()
        ):
            if not self._run_had_errors:
                self._append_event(
                    "pyscript 启动后未检测到流水线阶段输出，本次运行按失败处理。"
                )
            self._run_had_errors = True

        success = exit_code == 0 and not self._run_had_errors and not self._stop_event.is_set()
        return success, exit_code

    def run(self) -> None:
        self.state.emit("Running")
        run_status = "Failed"
        exit_code = -1
        cli_used = ""

        temp_dir = Path(tempfile.mkdtemp(prefix="seestar_superimpose_embedded_"))
        try:
            run_ssf, run_ini, _run_py = self._prepare_runtime_files(temp_dir)

            for attempt, siril_cli in enumerate(self.siril_candidates, start=1):
                if self._stop_event.is_set():
                    run_status = "Stopped"
                    break

                cli_used = str(siril_cli)
                success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                if self._stop_event.is_set():
                    run_status = "Stopped"
                    break

                if success:
                    run_status = "Completed"
                    break

                if self._spcc_cli_crash_detected and not self._spcc_crash_retry_attempted:
                    self._spcc_crash_retry_attempted = True
                    self._force_disable_spcc_for_retry = True
                    self._append_event(
                        "检测到 Siril 在 SPCC 测光阶段崩溃（退出码 -11）。"
                        "正在禁用 SPCC 并重试完整流水线，Stage 4 将改走 PCC/本地校色回退。"
                    )
                    success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                    if self._stop_event.is_set():
                        run_status = "Stopped"
                        break
                    if success:
                        run_status = "Completed"
                        break

                if self._python_env_issue and not self._python_env_repair_attempted:
                    self._python_env_repair_attempted = True
                    self._append_event(
                        "检测到 Siril Python 环境异常。"
                        "正在重置 Siril Python venv，并重试一次..."
                    )
                    if self._reset_siril_python_venv():
                        success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                        if self._stop_event.is_set():
                            run_status = "Stopped"
                            break
                        if success:
                            run_status = "Completed"
                            break
                    else:
                        self._append_event(
                            "自动重置 venv 失败。"
                        )

                if attempt < len(self.siril_candidates):
                    self._append_event(
                        "主流水线失败或卡住，"
                        "正在使用备用 Siril 运行时重试完整内置流水线..."
                    )
            else:
                run_status = "Failed"

        except Exception as e:
            self._append_event(f"Worker 内部错误：{e}")
            self._run_had_errors = True
            run_status = "Failed"
            exit_code = -1
        finally:
            self._cleanup_temp_dir(temp_dir)

        if self._stop_event.is_set() and run_status != "Stopped":
            run_status = "Stopped"

        self.done.emit(run_status, exit_code, self._run_had_errors, cli_used)
