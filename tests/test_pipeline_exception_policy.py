#!/usr/bin/env python3
"""Static checks for pipeline exception-handling boundaries."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"


class PipelineExceptionPolicyTests(unittest.TestCase):
    def test_only_explicit_pipeline_boundaries_keep_exception_catch_all(self) -> None:
        catch_all_handlers: list[tuple[Path, ast.ExceptHandler, str]] = []

        for path in sorted(PIPELINE_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent

            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.ExceptHandler)
                    and isinstance(node.type, ast.Name)
                    and node.type.id == "Exception"
                ):
                    continue
                owner = node
                while owner in parents and not isinstance(
                    owner, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    owner = parents[owner]
                owner_name = owner.name if isinstance(
                    owner, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) else "<module>"
                catch_all_handlers.append((path, node, owner_name))

        boundaries = Counter(
            (path.name, owner_name) for path, _handler, owner_name in catch_all_handlers
        )
        self.assertEqual(
            boundaries,
            Counter(
                {
                    ("seestar_Superimpose.py", "guarded"): 1,
                    ("seestar_Superimpose.py", "cmd_with_check"): 1,
                    ("seestar_Superimpose.py", "_record_stage"): 2,
                    ("seestar_Superimpose.py", "run"): 1,
                }
            ),
        )

        run_handler = next(
            handler
            for path, handler, owner_name in catch_all_handlers
            if path.name == "seestar_Superimpose.py" and owner_name == "run"
        )
        self.assertEqual(run_handler.name, "e")
        self.assertIn("traceback.format_exc()", ast.unparse(run_handler))

    def test_siril_import_fallbacks_do_not_alias_to_exception(self) -> None:
        offenders = []
        for path in sorted(PIPELINE_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not isinstance(node.value, ast.Name) or node.value.id != "Exception":
                    continue
                names = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
                if names:
                    offenders.append((path, node.lineno, names))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
