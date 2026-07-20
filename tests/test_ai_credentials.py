#!/usr/bin/env python3
"""AI credential packaging and Keychain bootstrap tests."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ai_credentials = _load_module(
    "ai_credentials",
    REPO_ROOT / "gui" / "ai_credentials.py",
)
packager = _load_module(
    "package_ai_credentials_test_module",
    REPO_ROOT / "build" / "package_ai_credentials.py",
)


class _MemoryKeychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, account: str) -> str | None:
        return self.values.get(account)

    def set(self, account: str, secret: str) -> None:
        self.values[account] = secret

    def delete(self, account: str) -> bool:
        return self.values.pop(account, None) is not None


class AiCredentialsTests(unittest.TestCase):
    def test_bootstrap_round_trip_does_not_contain_plaintext_key(self):
        secret = "test-provider-key-not-real"
        payload = ai_credentials.create_bootstrap_payload(
            {"SEESTAR_AI_API_KEY": secret}
        )

        self.assertNotIn(secret.encode("utf-8"), payload)
        self.assertEqual(
            ai_credentials.decode_bootstrap_payload(payload),
            {"SEESTAR_AI_API_KEY": secret},
        )

    def test_packager_sanitizes_env_and_verifies_bootstrap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "ai.env"
            sanitized = root / "packaged" / "ai.env"
            bootstrap = root / "packaged" / "ai-trial.bootstrap"
            binary = root / "app-binary"
            secret = "test-provider-key-not-real"
            source.write_text(
                "SEESTAR_AI_ENDPOINT=https://example.test/v1\n"
                "SEESTAR_AI_MODEL=test-model\n"
                f"SEESTAR_AI_API_KEY={secret}\n",
                encoding="utf-8",
            )
            os.chmod(source, 0o600)
            binary.write_bytes(b"fake binary without credentials")

            self.assertEqual(
                packager.package_credentials(source, sanitized, bootstrap),
                0,
            )
            sanitized_text = sanitized.read_text(encoding="utf-8")
            self.assertIn("SEESTAR_AI_API_KEY=", sanitized_text)
            self.assertIn("is stored in macOS Keychain", sanitized_text)
            self.assertNotIn(secret, sanitized_text)
            self.assertNotIn(secret.encode("utf-8"), bootstrap.read_bytes())
            self.assertEqual(
                packager.verify_credentials(
                    source,
                    sanitized,
                    bootstrap,
                    [sanitized, bootstrap, binary],
                ),
                0,
            )

    def test_developer_bootstrap_imports_missing_key_once(self):
        with tempfile.TemporaryDirectory() as td:
            resources = Path(td)
            secret = "test-provider-key-not-real"
            (resources / ai_credentials.AI_BOOTSTRAP_RESOURCE_NAME).write_bytes(
                ai_credentials.create_bootstrap_payload(
                    {"SEESTAR_AI_API_KEY": secret}
                )
            )
            store = _MemoryKeychain()

            first = ai_credentials.ensure_developer_credentials(
                resources,
                store=store,
            )
            (resources / ai_credentials.AI_BOOTSTRAP_RESOURCE_NAME).unlink()
            second = ai_credentials.ensure_developer_credentials(
                resources,
                store=store,
            )

            self.assertEqual(first["SEESTAR_AI_API_KEY"], secret)
            self.assertEqual(second["SEESTAR_AI_API_KEY"], secret)

    def test_changed_bootstrap_rotates_developer_keychain_value(self):
        with tempfile.TemporaryDirectory() as td:
            resources = Path(td)
            bootstrap = resources / ai_credentials.AI_BOOTSTRAP_RESOURCE_NAME
            store = _MemoryKeychain()
            bootstrap.write_bytes(
                ai_credentials.create_bootstrap_payload(
                    {"SEESTAR_AI_API_KEY": "trial-key-v1"}
                )
            )
            ai_credentials.ensure_developer_credentials(resources, store=store)
            bootstrap.write_bytes(
                ai_credentials.create_bootstrap_payload(
                    {"SEESTAR_AI_API_KEY": "trial-key-v2"}
                )
            )

            rotated = ai_credentials.ensure_developer_credentials(
                resources,
                store=store,
            )

            self.assertEqual(rotated["SEESTAR_AI_API_KEY"], "trial-key-v2")


if __name__ == "__main__":
    unittest.main()
