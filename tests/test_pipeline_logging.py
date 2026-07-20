#!/usr/bin/env python3
"""Regression tests for non-blocking pipeline logging."""

from __future__ import annotations

import builtins
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from logging_utils import PipelineLogger


class PipelineLoggerTests(unittest.TestCase):
    def test_logging_never_writes_to_stdout(self) -> None:
        logger = PipelineLogger("DEBUG")
        with patch.object(
            builtins,
            "print",
            side_effect=AssertionError("stdout must not be used"),
        ):
            logger.debug("debug")
            logger.info("info")

    def test_debug_is_file_only_and_info_reaches_sink(self) -> None:
        received: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "pipeline.log"
            logger = PipelineLogger("DEBUG")
            logger.set_file_path(log_path)
            logger.set_sink(received.append)

            logger.debug("debug detail")
            logger.info("visible status")

            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("[DEBUG] debug detail", contents)
            self.assertIn("[INFO] visible status", contents)
            self.assertFalse(any("debug detail" in line for line in received))
            self.assertTrue(any("visible status" in line for line in received))

    def test_preconnection_logs_are_flushed_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "pipeline.log"
            logger = PipelineLogger()
            logger.info("before connection")

            logger.set_file_path(log_path)

            self.assertIn(
                "[INFO] before connection",
                log_path.read_text(encoding="utf-8"),
            )

    def test_broken_sink_is_disabled_without_interrupting_pipeline(self) -> None:
        calls = 0

        def broken_sink(_line: str) -> None:
            nonlocal calls
            calls += 1
            raise BrokenPipeError("closed")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "pipeline.log"
            logger = PipelineLogger()
            logger.set_file_path(log_path)
            logger.set_sink(broken_sink)

            logger.info("first")
            logger.info("second")

            contents = log_path.read_text(encoding="utf-8")
            self.assertEqual(calls, 1)
            self.assertIn("[INFO] first", contents)
            self.assertIn("实时日志接口已关闭", contents)
            self.assertIn("[INFO] second", contents)


if __name__ == "__main__":
    unittest.main()
