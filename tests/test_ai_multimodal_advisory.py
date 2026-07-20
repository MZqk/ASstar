#!/usr/bin/env python3
"""Tests for optional multimodal advisor payloads and text fallback."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

if "sirilpy" not in sys.modules:
    sirilpy = types.ModuleType("sirilpy")
    exceptions = types.ModuleType("sirilpy.exceptions")
    enums = types.ModuleType("sirilpy.enums")

    class _SirilError(Exception):
        pass

    class _SirilConnectionError(_SirilError):
        pass

    class _SirilInterface:
        pass

    class _CommandStatus:
        CMD_GENERIC_ERROR = 1
        CMD_THREAD_RUNNING = 2

    exceptions.CommandError = _SirilError
    exceptions.DataError = _SirilError
    exceptions.SirilError = _SirilError
    exceptions.SirilConnectionError = _SirilConnectionError
    sirilpy.SirilInterface = _SirilInterface
    enums.CommandStatus = _CommandStatus
    sirilpy.exceptions = exceptions
    sys.modules["sirilpy"] = sirilpy
    sys.modules["sirilpy.exceptions"] = exceptions
    sys.modules["sirilpy.enums"] = enums

import ai_advisory  # noqa: E402


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)


class _Pipeline:
    def __init__(self, *, reject_images: bool = False) -> None:
        self.cfg = SimpleNamespace(
            ai_post_enabled=True,
            ai_endpoint="https://example.invalid/v1/chat/completions",
            ai_model="kimi-vision",
            ai_api_key="test-key",
            ai_timeout_sec=30,
            ai_prompt="",
            ai_advisor_mode="multimodal",
        )
        self.log = _Logger()
        self._ai_stage_circuit_breaker: dict[str, str] = {}
        self.payloads: list[dict[str, Any]] = []
        self.reject_images = reject_images

    def _build_ai_chat_endpoint_candidates(self, endpoint: str) -> list[str]:
        return [endpoint]

    def _post_json_with_auth(
        self, _endpoint: str, payload: dict[str, Any], _key: str, _timeout: int
    ) -> dict[str, Any]:
        self.payloads.append(payload)
        content = payload["messages"][1]["content"]
        if self.reject_images and isinstance(content, list):
            raise RuntimeError("image_url unsupported")
        return {"choices": [{"message": {"content": '{"plan":{"value":1}}'}}]}

    def _extract_chat_content(self, response: dict[str, Any]) -> str:
        return str(response["choices"][0]["message"]["content"])

    def _extract_first_json_object(self, content: str) -> dict[str, Any]:
        return json.loads(content)

    def _extract_stage_advisory_from_text(
        self, _stage_name: str, _content: str
    ) -> None:
        return None

    def _write_ai_raw_response(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class MultimodalAdvisorTests(unittest.TestCase):
    def test_multimodal_request_contains_image_url(self) -> None:
        pipeline = _Pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            preview = Path(tmpdir) / "preview.png"
            preview.write_bytes(b"\x89PNG\r\n\x1a\nmock")
            result = ai_advisory.request_stage_ai_advisory(
                pipeline,
                "test_stage",
                '{"plan":{"value":1}}',
                {"metric": 1},
                image_paths=[("Current image", preview)],
            )

        self.assertEqual(result["plan"]["value"], 1)
        user_content = pipeline.payloads[0]["messages"][1]["content"]
        self.assertIsInstance(user_content, list)
        image_parts = [item for item in user_content if item.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_multimodal_request_falls_back_to_text(self) -> None:
        pipeline = _Pipeline(reject_images=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            preview = Path(tmpdir) / "preview.png"
            preview.write_bytes(b"\x89PNG\r\n\x1a\nmock")
            result = ai_advisory.request_stage_ai_advisory(
                pipeline,
                "test_stage",
                '{"plan":{"value":1}}',
                {"metric": 1},
                image_paths=[("Current image", preview)],
            )

        self.assertEqual(result["plan"]["value"], 1)
        request_modes = [
            "multimodal"
            if isinstance(payload["messages"][1]["content"], list)
            else "text"
            for payload in pipeline.payloads
        ]
        self.assertIn("multimodal", request_modes)
        self.assertIn("text", request_modes)
        self.assertTrue(any("text-only advisor" in item for item in pipeline.log.warnings))


if __name__ == "__main__":
    unittest.main()
