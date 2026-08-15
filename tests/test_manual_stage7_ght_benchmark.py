from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "manual_stage7_ght_benchmark.py"
SPEC = importlib.util.spec_from_file_location("manual_stage7_ght_benchmark", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class ManualStage7GHTBenchmarkTests(unittest.TestCase):
    def test_parameter_boundaries_and_order_are_validated(self) -> None:
        valid = benchmark.validate_ght_parameters(D=10, B=-5, LP=0, SP=0.5, HP=1)
        self.assertEqual(valid["D"], 10.0)

        invalid_sets = (
            {"D": 0, "B": 0, "LP": 0, "SP": 0.5, "HP": 1},
            {"D": 10.1, "B": 0, "LP": 0, "SP": 0.5, "HP": 1},
            {"D": 1, "B": -5.1, "LP": 0, "SP": 0.5, "HP": 1},
            {"D": 1, "B": 15.1, "LP": 0, "SP": 0.5, "HP": 1},
            {"D": 1, "B": 0, "LP": 0.6, "SP": 0.5, "HP": 1},
            {"D": 1, "B": 0, "LP": 0, "SP": 0.8, "HP": 0.7},
        )
        for values in invalid_sets:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    benchmark.validate_ght_parameters(**values)

    def test_script_contains_exactly_one_unlinked_ght_command(self) -> None:
        parameters = benchmark.validate_ght_parameters(
            D=3.2,
            B=1.0,
            LP=0.001,
            SP=0.15,
            HP=0.98,
        )
        lines = benchmark.build_siril_script(
            Path("/tmp/stage6_starless.fit"),
            Path("/tmp/benchmark"),
            parameters,
        )
        commands = [line for line in lines if line.startswith("ght ")]

        self.assertEqual(len(commands), 1)
        self.assertIn("-D=3.2", commands[0])
        self.assertIn("-B=1", commands[0])
        self.assertIn("-LP=0.001", commands[0])
        self.assertIn("-even", commands[0])
        self.assertIn("-clipmode=rgbblend", commands[0])
        lowered = "\n".join(lines).lower()
        self.assertNotIn("asinh", lowered)
        self.assertNotIn("autoghs", lowered)
        self.assertNotIn("inverse", lowered)
        self.assertNotIn("bp=", lowered)

    def test_only_supported_high_dynamic_targets_are_allowed(self) -> None:
        parser = benchmark.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--source", "stage6_starless.fit",
                    "--cand-a", "stage7_cand_a.fit",
                    "--target-type", "open_cluster_color_preserve",
                    "--D", "1",
                    "--B", "0",
                    "--LP", "0",
                    "--SP", "0.5",
                    "--HP", "1",
                    "--siril-cli", "/tmp/siril-cli",
                    "--output-dir", "/tmp/out",
                ]
            )


if __name__ == "__main__":
    unittest.main()
