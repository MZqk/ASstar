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

    def test_real_siril_job_uses_prepared_external_offline_resources(self) -> None:
        workflow = (
            REPO_ROOT / ".github/workflows/tests.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "STARUN_OFFLINE_RESOURCE_ROOT: "
            "${{ vars.STARUN_OFFLINE_RESOURCE_ROOT }}",
            workflow,
        )
        self.assertNotIn(
            "STARUN_OFFLINE_RESOURCE_ROOT: ${{ github.workspace }}/resources",
            workflow,
        )
        self.assertNotIn("lfs: true", workflow)
        self.assertNotIn("download_siril_plugins.sh", workflow)
        self.assertIn('test -n "${STARUN_OFFLINE_RESOURCE_ROOT:-}"', workflow)
        self.assertIn('case "${STARUN_OFFLINE_RESOURCE_ROOT}" in', workflow)
        self.assertIn(
            "STARUN_OFFLINE_RESOURCE_ROOT must be an absolute path",
            workflow,
        )
        self.assertIn(
            'test -d "${STARUN_OFFLINE_RESOURCE_ROOT}/siril_plugins"',
            workflow,
        )
        self.assertLess(
            workflow.index("Validate prepared offline resource root"),
            workflow.index("Run real Siril and offline plugin regression"),
        )

    def test_versioned_real_e2e_docs_match_runner_contract(self) -> None:
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        markdown_negations = {
            line.strip()
            for line in gitignore.splitlines()
            if line.strip().startswith("!")
            and line.strip().endswith(".md")
        }
        self.assertEqual(
            markdown_negations,
            {
                "!/README.md",
                "!/INTEGRATION_README.md",
                "!/resources/siril_plugins/README.md",
            },
        )

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        integration = (REPO_ROOT / "INTEGRATION_README.md").read_text(
            encoding="utf-8"
        )
        plugin_readme = (
            REPO_ROOT / "resources/siril_plugins/README.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((readme, integration, plugin_readme))
        self.assertNotIn("仓库 LFS 插件模型", combined)
        self.assertNotIn("SyQon LFS 模型", combined)
        self.assertIn("STARUN_OFFLINE_RESOURCE_ROOT", readme)
        self.assertIn("不等于进程退出成功", readme)
        self.assertIn(
            "输入、Siril/runtime、Gaia、SyQon 和 CosmicClarity 资源全部准备",
            readme,
        )
        self.assertIn("starun.pipeline-result.v2", integration)
        self.assertIn("STARUN_REAL_E2E_READY=true", integration)
        self.assertIn("actions/setup-python", integration)
        self.assertIn("starun.processing-parameters.v5", combined)
        self.assertIn("历史 v4", combined)
        self.assertNotIn(
            "唯一支持的 `starun.processing-parameters.v4`",
            combined,
        )
        self.assertNotIn("新运行只接受 `starun.processing-parameters.v4`", combined)
        self.assertNotIn("处理参数只接受 v4 当前字段", combined)
        self.assertNotIn("检查仅 v4 验收", combined)
        self.assertIn(
            "不会下载 CosmicClarity mono/color denoise 模型",
            plugin_readme,
        )

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
