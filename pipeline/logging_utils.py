"""Lightweight logging helpers for the Seestar pipeline."""
from __future__ import annotations

import time
import textwrap
from pathlib import Path
from typing import Callable, Optional, Union


class PipelineLogger:
    """轻量级结构化日志，不依赖 pyscript 的 stdout 管道。"""

    _LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
    _MAX_PENDING_LINES = 256
    _SINK_CHUNK_CHARS = 220

    def __init__(self, min_level: str = "INFO"):
        self.min_level = self._LEVELS.get(min_level, 1)
        self._stage_start = None
        self._stage_name = None
        self._sink: Optional[Callable[[str], object]] = None
        self._file_path: Optional[Path] = None
        self._pending: list[tuple[str, str]] = []

    @staticmethod
    def _reraise_control_flow(exc: BaseException) -> None:
        if not isinstance(exc, Exception):
            raise exc

    def set_file_path(self, path: Union[str, Path]) -> None:
        """启用本地日志，并写入连接前暂存的日志。"""
        self._file_path = Path(path)
        pending = self._pending
        self._pending = []
        for _tag, line in pending:
            self._write_file(line)

    def set_sink(self, sink: Optional[Callable[[str], object]]) -> None:
        """设置实时日志接口；传入 ``None`` 可关闭实时转发。"""
        self._sink = sink

    def _write_file(self, line: str) -> None:
        if self._file_path is None:
            return
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with self._file_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        except BaseException as exc:
            self._reraise_control_flow(exc)
            # 日志失败不能中断图像处理；实时 sink 仍可继续使用。

    @classmethod
    def _sink_chunks(cls, line: str):
        """限制单条 Siril 日志长度，避免超过接口的消息上限。"""
        logical_lines = line.splitlines() or [""]
        for logical_line in logical_lines:
            chunks = textwrap.wrap(
                logical_line,
                width=cls._SINK_CHUNK_CHARS,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            yield from (chunks or [""])

    def _write_sink(self, line: str) -> None:
        if self._sink is None:
            return
        try:
            for chunk in self._sink_chunks(line):
                self._sink(chunk)
        except BaseException as exc:
            self._reraise_control_flow(exc)
            # Siril 日志接口失效后熔断，后续只写本地文件。
            self._sink = None
            ts = time.strftime("%H:%M:%S")
            self._write_file(
                f"[{ts}] [WARN] 实时日志接口已关闭: "
                f"{type(exc).__name__}: {exc}"
            )

    def _emit(self, tag, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [{tag}] {msg}"
        if self._file_path is None:
            self._pending.append((tag, line))
            if len(self._pending) > self._MAX_PENDING_LINES:
                del self._pending[:-self._MAX_PENDING_LINES]
        else:
            self._write_file(line)

        # DEBUG 可能非常密集，只落盘，避免给 Siril 通信通道制造压力。
        if tag != "DEBUG":
            self._write_sink(line)

    def debug(self, msg):
        if self.min_level <= 0:
            self._emit("DEBUG", msg)

    def info(self, msg):
        if self.min_level <= 1:
            self._emit("INFO", msg)

    def warn(self, msg):
        if self.min_level <= 2:
            self._emit("WARN", msg)

    def error(self, msg):
        if self.min_level <= 3:
            self._emit("ERROR", msg)

    def stage_start(self, name):
        self._stage_name = name
        self._stage_start = time.time()
        self.info("=" * 50)
        self.info(name)
        self.info("=" * 50)

    def stage_end(self, name=None):
        elapsed = time.time() - self._stage_start if self._stage_start else 0
        display = name or self._stage_name or "未知阶段"
        self.info(f"✓ {display} 完成 ({elapsed:.1f}s)")
        self._stage_start = None
        return elapsed
