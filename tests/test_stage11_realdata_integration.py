#!/usr/bin/env python3
"""Real-data integration test for isolated Stage11 execution."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MAIN = REPO_ROOT / "pipeline" / "seestar_Superimpose.py"
PIPELINE_STAGE11 = REPO_ROOT / "pipeline" / "stage11_ai_postprocess.py"

REQUIRED_STAGE11_ENV = (
    "STAGE11_TEST_INPUT_FILE",
    "STAGE11_TEST_AI_ENV_FILE",
    "STAGE11_TEST_SIRIL_CLI",
    "STAGE11_TEST_SIRIL_PYTHON_CLI",
    "STAGE11_TEST_RUNTIME_HOME",
)


def parse_simple_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out

    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        out[key] = value
    return out


def tail_lines(text: str, n: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


class Stage11RealDataIntegrationTest(unittest.TestCase):
    def test_stage11_realdata_not_skipped(self):
        configured = {
            key: os.getenv(key, "").strip() for key in REQUIRED_STAGE11_ENV
        }
        missing = [key for key, value in configured.items() if not value]
        if missing:
            self.skipTest(
                "requires explicit external Stage11 test configuration: "
                + ", ".join(missing)
            )

        input_fit = Path(configured["STAGE11_TEST_INPUT_FILE"]).expanduser().resolve()
        self.assertTrue(input_fit.exists(), f"input FIT not found: {input_fit}")

        siril_cli = Path(configured["STAGE11_TEST_SIRIL_CLI"]).expanduser().resolve()
        self.assertTrue(siril_cli.exists(), f"siril-cli not found: {siril_cli}")

        ai_env_file = Path(configured["STAGE11_TEST_AI_ENV_FILE"]).expanduser().resolve()
        ai_env = parse_simple_env_file(ai_env_file)
        required_ai_keys = (
            "SEESTAR_AI_ENABLED",
            "SEESTAR_AI_ENDPOINT",
            "SEESTAR_AI_MODEL",
            "SEESTAR_AI_API_KEY",
        )
        missing_keys = [k for k in required_ai_keys if not ai_env.get(k, "").strip()]
        self.assertFalse(
            missing_keys,
            f"missing required AI keys in {ai_env_file}: {', '.join(missing_keys)}",
        )

        with tempfile.TemporaryDirectory(prefix="stage11-realdata-") as td:
            tmp_dir = Path(td)
            run_py = tmp_dir / PIPELINE_MAIN.name
            run_stage11_py = tmp_dir / PIPELINE_STAGE11.name
            runner_py = tmp_dir / "stage11_realdata_runner.py"
            run_ini = tmp_dir / "config.1.4.ini"
            run_ssf = tmp_dir / "run_stage11.ssf"

            shutil.copy2(PIPELINE_MAIN, run_py)
            shutil.copy2(PIPELINE_STAGE11, run_stage11_py)

            run_ini.write_text(
                "[core]\n",
                encoding="utf-8",
            )

            runner_py.write_text(
                textwrap.dedent(
                    """
                    #!/usr/bin/env python3
                    import copy
                    import os
                    from pathlib import Path
                    from seestar_Superimpose import PipelineConfig, SeestarPostProcessor

                    target = Path(os.environ["STAGE11_TEST_INPUT_FILE"]).expanduser().resolve()
                    proc = SeestarPostProcessor(PipelineConfig())
                    proc.connect()
                    proc.cfg = copy.deepcopy(proc.initial_cfg)
                    proc._apply_ai_env_overrides()
                    proc.work_dir = target.parent
                    proc.process_dir = proc.work_dir / "process"
                    proc.process_dir.mkdir(exist_ok=True)
                    proc.cmd_with_check("cd", f'"{proc.work_dir}"')
                    proc.cmd_with_check("load", target.stem)
                    proc.stage11_ai_postprocess()
                    if proc.results:
                        result = proc.results[-1]
                        print(
                            f"STAGE11_RESULT status={result.status} "
                            f"message={result.message}"
                        )
                    else:
                        print("STAGE11_RESULT status=unknown message=no-result-recorded")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            run_ssf.write_text(
                "\n".join(
                    [
                        "requires 1.4.0",
                        f'cd "{input_fit.parent}"',
                        f'pyscript "{runner_py}"',
                        "close",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(ai_env)
            env["STAGE11_TEST_INPUT_FILE"] = str(input_fit)
            env["HOME"] = str(
                Path(configured["STAGE11_TEST_RUNTIME_HOME"]).expanduser().resolve()
            )
            bundled_py = Path(
                configured["STAGE11_TEST_SIRIL_PYTHON_CLI"]
            ).expanduser().resolve()
            self.assertTrue(bundled_py.exists(), f"Siril Python not found: {bundled_py}")
            env["SIRIL_PYTHON_CLI"] = str(bundled_py)

            cmd = [
                str(siril_cli),
                "--offline",
                "-d",
                str(input_fit.parent),
                "-i",
                str(run_ini),
                "-s",
                str(run_ssf),
            ]
            cp = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
            )

            output = (cp.stdout or "") + "\n" + (cp.stderr or "")
            self.assertEqual(
                cp.returncode,
                0,
                msg=(
                    "siril-cli returned non-zero.\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"tail:\n{tail_lines(output)}"
                ),
            )

            match = re.search(
                r"STAGE11_RESULT status=([a-zA-Z_]+) message=(.*)",
                output,
            )
            self.assertIsNotNone(
                match,
                msg=f"missing STAGE11_RESULT marker.\nTail:\n{tail_lines(output)}",
            )
            status = match.group(1).strip().lower()
            message = match.group(2).strip()

            self.assertIn(
                status,
                {"ok", "degraded"},
                msg=(
                    "stage11 should run on real data and must not be skipped/failed.\n"
                    f"status={status}, message={message}\n"
                    f"tail:\n{tail_lines(output)}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
