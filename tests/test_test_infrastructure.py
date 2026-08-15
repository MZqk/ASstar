#!/usr/bin/env python3
"""Repository-level guardrails for the test infrastructure itself."""
from __future__ import annotations

import configparser
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestInfrastructureTests(unittest.TestCase):
    def test_dev_requirements_declare_runner_and_coverage_plugin(self) -> None:
        requirements = (
            REPO_ROOT / "requirements-dev.txt"
        ).read_text(encoding="utf-8")
        declared = {
            re.split(r"[<>=!~\[]", line.strip(), maxsplit=1)[0].lower()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("pytest", declared)
        self.assertIn("pytest-cov", declared)

    def test_coverage_config_has_enforced_branch_floor(self) -> None:
        config = configparser.ConfigParser()
        config.read(REPO_ROOT / ".coveragerc", encoding="utf-8")
        self.assertTrue(config.getboolean("run", "branch"))
        self.assertGreaterEqual(config.getfloat("report", "fail_under"), 65.0)

    def test_ci_runs_unit_coverage_and_real_siril_contract(self) -> None:
        workflow = (
            REPO_ROOT / ".github/workflows/tests.yml"
        ).read_text(encoding="utf-8")
        for required in (
            "--cov-fail-under=65",
            "real-siril-stage1-10:",
            "tests/real_siril_stage1_10_e2e.py",
            "runs-on: [self-hosted, macOS, ARM64, starun-e2e]",
        ):
            self.assertIn(required, workflow)

    def test_fallback_tests_remain_split_at_stage_boundaries(self) -> None:
        wrapper = REPO_ROOT / "tests/test_pipeline_plugin_fallbacks.py"
        modules = sorted(
            path
            for path in (REPO_ROOT / "tests").glob(
                "test_pipeline_plugin_fallbacks_*.py"
            )
            if path.name != "test_pipeline_plugin_fallbacks.py"
        )
        self.assertGreaterEqual(len(modules), 8)
        self.assertLessEqual(len(wrapper.read_text(encoding="utf-8").splitlines()), 100)
        offenders = {
            path.name: len(path.read_text(encoding="utf-8").splitlines())
            for path in modules
            if len(path.read_text(encoding="utf-8").splitlines()) > 4000
        }
        self.assertEqual(offenders, {})


if __name__ == "__main__":
    unittest.main()
