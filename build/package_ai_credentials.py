#!/usr/bin/env python3
"""Sanitize AI env resources and package release credentials for Keychain import."""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = PROJECT_ROOT / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from ai_credentials import (  # noqa: E402
    AI_SECRET_ENV_KEYS,
    create_bootstrap_payload,
    decode_bootstrap_payload,
)


_ENV_ASSIGNMENT = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<equals>\s*=)(?P<value>.*)$"
)


def _normalized_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    comment_index = value.find(" #")
    if comment_index >= 0:
        value = value[:comment_index].rstrip()
    return value


def parse_secret_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        match = _ENV_ASSIGNMENT.match(raw_line)
        if match is None:
            continue
        key = match.group("key")
        if key in AI_SECRET_ENV_KEYS:
            values[key] = _normalized_env_value(match.group("value"))
    return values


def sanitize_env_text(text: str) -> str:
    output: list[str] = []
    for raw_line in text.splitlines():
        match = _ENV_ASSIGNMENT.match(raw_line)
        if match is None or match.group("key") not in AI_SECRET_ENV_KEYS:
            output.append(raw_line)
            continue
        output.append(f"# {match.group('key')} is stored in macOS Keychain")
        output.append(
            f"{match.group('prefix')}{match.group('key')}{match.group('equals')}"
        )
    suffix = "\n" if text.endswith("\n") or output else ""
    return "\n".join(output) + suffix


def _require_private_source(path: Path, secrets: dict[str, str]) -> None:
    if not any(value.strip() for value in secrets.values()):
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(
            f"AI 凭据源文件权限必须为 0600: {path} "
            f"(当前 {mode:04o}，请先执行 chmod 600)"
        )


def package_credentials(source: Path, sanitized_env: Path, bootstrap: Path) -> int:
    text = source.read_text(encoding="utf-8", errors="strict")
    secrets = parse_secret_values(text)
    _require_private_source(source, secrets)

    sanitized_env.parent.mkdir(parents=True, exist_ok=True)
    sanitized_env.write_text(sanitize_env_text(text), encoding="utf-8")

    payload = create_bootstrap_payload(secrets)
    if payload:
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.write_bytes(payload)
        os.chmod(bootstrap, 0o600)
    elif bootstrap.exists():
        bootstrap.unlink()
    print(
        "Packaged sanitized AI config; "
        f"Keychain bootstrap entries={sum(bool(value.strip()) for value in secrets.values())}"
    )
    return 0


def verify_credentials(
    source: Path,
    sanitized_env: Path,
    bootstrap: Path,
    scan_paths: list[Path],
) -> int:
    source_values = parse_secret_values(
        source.read_text(encoding="utf-8", errors="strict")
    )
    expected = {
        key: value.strip()
        for key, value in source_values.items()
        if value.strip()
    }
    packaged_values = parse_secret_values(
        sanitized_env.read_text(encoding="utf-8", errors="strict")
    )
    leaked_fields = [key for key, value in packaged_values.items() if value.strip()]
    if leaked_fields:
        raise RuntimeError(
            "打包后的 ai.env 仍包含非空凭据字段: " + ", ".join(leaked_fields)
        )

    decoded = decode_bootstrap_payload(bootstrap.read_bytes()) if bootstrap.is_file() else {}
    if decoded != expected:
        raise RuntimeError("Keychain bootstrap 与构建凭据源不一致")

    secret_bytes = [value.encode("utf-8") for value in expected.values()]
    for path in scan_paths:
        if not path.is_file():
            raise RuntimeError(f"凭据扫描目标不存在: {path}")
        content = path.read_bytes()
        if any(secret in content for secret in secret_bytes):
            raise RuntimeError(f"打包产物中检测到明文 AI Key: {path}")
    print(
        "Verified packaged AI credentials: plaintext fields empty, "
        f"bootstrap entries={len(expected)}, scan targets={len(scan_paths)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--source", type=Path, required=True)
    package_parser.add_argument("--sanitized-env", type=Path, required=True)
    package_parser.add_argument("--bootstrap", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--source", type=Path, required=True)
    verify_parser.add_argument("--sanitized-env", type=Path, required=True)
    verify_parser.add_argument("--bootstrap", type=Path, required=True)
    verify_parser.add_argument("--scan", type=Path, action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "package":
            return package_credentials(args.source, args.sanitized_env, args.bootstrap)
        return verify_credentials(
            args.source,
            args.sanitized_env,
            args.bootstrap,
            list(args.scan),
        )
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        print(f"AI credential packaging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
