#!/usr/bin/env python3
"""Cancellable runtime bootstrap worker for the Starun GUI."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from PySide6.QtCore import QThread, Signal


class BootstrapCancelled(RuntimeError):
    """Raised when the user cancels runtime preparation."""


class BootstrapError(RuntimeError):
    """Runtime preparation failure with a user-facing dialog title."""

    def __init__(self, title: str, detail: str) -> None:
        super().__init__(detail)
        self.title = title


BootstrapRunner = Callable[
    [threading.Event, Callable[[str], None]],
    object,
]


class BootstrapWorker(QThread):
    """Runs the pre-pipeline runtime preparation without blocking the UI."""

    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, runner: BootstrapRunner, parent=None) -> None:
        super().__init__(parent)
        self._runner = runner
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            result = self._runner(self._stop_event, self.progress.emit)
            if self._stop_event.is_set():
                raise BootstrapCancelled()
        except BootstrapCancelled:
            self.cancelled.emit()
        except BootstrapError as exc:
            self.failed.emit(exc.title, str(exc))
        except Exception as exc:
            self.failed.emit("准备运行环境失败", str(exc))
        else:
            self.succeeded.emit(result)


def _terminate_process_group(
    proc: subprocess.Popen[Any],
    *,
    grace_sec: float = 2.0,
) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            proc.terminate()
        except OSError:
            pass

    deadline = time.monotonic() + max(0.0, grace_sec)
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def run_cancellable_process(
    args: Sequence[str],
    *,
    stop_event: threading.Event,
    cwd: str | None = None,
    capture_output: bool = False,
    text: bool = False,
    check: bool = False,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    """A subprocess.run-compatible subset that responds to bootstrap cancel."""

    proc = subprocess.Popen(
        list(args),
        cwd=cwd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        env=env,
        start_new_session=True,
    )
    started_at = time.monotonic()
    stdout: Any = None
    stderr: Any = None

    while True:
        if stop_event.is_set():
            _terminate_process_group(proc)
            try:
                proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            raise BootstrapCancelled()

        if timeout is not None and time.monotonic() - started_at >= timeout:
            _terminate_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                stdout, stderr = None, None
            raise subprocess.TimeoutExpired(
                list(args),
                timeout,
                output=stdout,
                stderr=stderr,
            )

        try:
            stdout, stderr = proc.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            continue

    completed = subprocess.CompletedProcess(
        list(args),
        proc.returncode,
        stdout,
        stderr,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed
