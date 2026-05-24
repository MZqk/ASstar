"""Lightweight logging helpers for the Seestar pipeline."""
from __future__ import annotations

import time


class PipelineLogger:
    """轻量级结构化日志"""

    _LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}

    def __init__(self, min_level: str = "INFO"):
        self.min_level = self._LEVELS.get(min_level, 1)
        self._stage_start = None
        self._stage_name = None

    def _emit(self, tag, msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{tag}] {msg}")

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
