#!/usr/bin/env python3
"""macOS Keychain storage and release bootstrap helpers for AI credentials.

The bundled bootstrap is deliberately an obfuscation layer, not a security
boundary: the application must be able to unwrap it without a server-held
secret.  Its purpose is to keep the provider key out of plaintext resources
and import it into the user's login Keychain on first use.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping


AI_BOOTSTRAP_RESOURCE_NAME = "ai-trial.bootstrap"
AI_KEYCHAIN_SERVICE = "StarunC.SeestarSuperimpose.ai"
AI_PROVIDER_DEVELOPER = "developer"
AI_PROVIDER_CUSTOM = "custom"
AI_PROVIDER_MODES = frozenset({AI_PROVIDER_DEVELOPER, AI_PROVIDER_CUSTOM})

AI_SECRET_ENV_KEYS = (
    "SEESTAR_AI_API_KEY",
    "SEESTAR_AI_ARTISTIC_API_KEY",
)
AI_KEYCHAIN_ACCOUNTS = {
    "SEESTAR_AI_API_KEY": "developer.default.api-key",
    "SEESTAR_AI_ARTISTIC_API_KEY": "developer.default.artistic-api-key",
}
CUSTOM_API_KEY_ACCOUNT = "user.custom.api-key"
DEVELOPER_BOOTSTRAP_DIGEST_ACCOUNT = "developer.bootstrap.sha256"

_BOOTSTRAP_FORMAT = 1
_BOOTSTRAP_ITERATIONS = 200_000
_ITEM_NOT_FOUND = -25300


class AiCredentialError(RuntimeError):
    """Raised when bootstrap or Keychain credential handling fails."""


def _bootstrap_passphrase() -> str:
    # This material is intentionally recoverable from the client.  Splitting it
    # only avoids leaving a useful wrapping password as a single literal.
    fragments = ("Star", "unC", "Seestar", "Superimpose", "trial", "macOS14")
    return hashlib.sha256("|".join(fragments).encode("utf-8")).hexdigest()


def _openssl_path() -> str:
    preferred = Path("/usr/bin/openssl")
    if preferred.is_file():
        return str(preferred)
    located = shutil.which("openssl")
    if located:
        return located
    raise AiCredentialError("未找到系统 openssl，无法处理 AI 凭据导入载荷")


def _openssl_crypt(payload: bytes, *, decrypt: bool) -> bytes:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "SEESTAR_AI_BOOTSTRAP_WRAP": _bootstrap_passphrase(),
    }
    command = [
        _openssl_path(),
        "enc",
        "-aes-256-cbc",
        "-pbkdf2",
        "-iter",
        str(_BOOTSTRAP_ITERATIONS),
        "-md",
        "sha256",
        "-pass",
        "env:SEESTAR_AI_BOOTSTRAP_WRAP",
    ]
    if decrypt:
        command.append("-d")
    completed = subprocess.run(
        command,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AiCredentialError(f"AI 凭据导入载荷处理失败: {detail or completed.returncode}")
    return completed.stdout


def create_bootstrap_payload(secrets: Mapping[str, str]) -> bytes:
    filtered = {
        key: str(secrets.get(key, "")).strip()
        for key in AI_SECRET_ENV_KEYS
        if str(secrets.get(key, "")).strip()
    }
    if not filtered:
        return b""
    plaintext = json.dumps(
        {"version": _BOOTSTRAP_FORMAT, "secrets": filtered},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    ciphertext = _openssl_crypt(plaintext, decrypt=False)
    envelope = {
        "format": _BOOTSTRAP_FORMAT,
        "cipher": "aes-256-cbc-pbkdf2",
        "iterations": _BOOTSTRAP_ITERATIONS,
        "payload": base64.b64encode(ciphertext).decode("ascii"),
    }
    return (json.dumps(envelope, separators=(",", ":")) + "\n").encode("utf-8")


def decode_bootstrap_payload(payload: bytes) -> dict[str, str]:
    try:
        envelope = json.loads(payload.decode("utf-8"))
        if int(envelope.get("format", 0)) != _BOOTSTRAP_FORMAT:
            raise ValueError("unsupported format")
        ciphertext = base64.b64decode(envelope["payload"], validate=True)
        decoded = json.loads(_openssl_crypt(ciphertext, decrypt=True).decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise AiCredentialError("AI 凭据导入载荷格式无效") from exc
    secrets = decoded.get("secrets")
    if not isinstance(secrets, dict):
        raise AiCredentialError("AI 凭据导入载荷缺少 secrets")
    return {
        key: str(secrets.get(key, "")).strip()
        for key in AI_SECRET_ENV_KEYS
        if str(secrets.get(key, "")).strip()
    }


class MacOSKeychainStore:
    """Small generic-password wrapper around macOS Security.framework."""

    def __init__(self, service: str = AI_KEYCHAIN_SERVICE) -> None:
        if sys.platform != "darwin":
            raise AiCredentialError("AI Keychain 配置仅支持 macOS")
        self.service = service
        security_path = ctypes.util.find_library("Security") or (
            "/System/Library/Frameworks/Security.framework/Security"
        )
        core_foundation_path = ctypes.util.find_library("CoreFoundation") or (
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        try:
            self._security = ctypes.CDLL(security_path)
            self._core_foundation = ctypes.CDLL(core_foundation_path)
        except OSError as exc:
            raise AiCredentialError(f"无法加载 macOS Keychain framework: {exc}") from exc
        self._configure_functions()
        self._keychain = ctypes.c_void_p()
        status = int(self._security.SecKeychainCopyDefault(ctypes.byref(self._keychain)))
        if status != 0 or not self._keychain.value:
            raise AiCredentialError(
                f"无法打开默认 macOS Keychain (OSStatus {status})"
            )

    def _configure_functions(self) -> None:
        uint32_pointer = ctypes.POINTER(ctypes.c_uint32)
        void_pointer_pointer = ctypes.POINTER(ctypes.c_void_p)
        self._security.SecKeychainCopyDefault.argtypes = [void_pointer_pointer]
        self._security.SecKeychainCopyDefault.restype = ctypes.c_int32
        self._security.SecKeychainFindGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            uint32_pointer,
            void_pointer_pointer,
            void_pointer_pointer,
        ]
        self._security.SecKeychainFindGenericPassword.restype = ctypes.c_int32
        self._security.SecKeychainAddGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            void_pointer_pointer,
        ]
        self._security.SecKeychainAddGenericPassword.restype = ctypes.c_int32
        self._security.SecKeychainItemModifyAttributesAndData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._security.SecKeychainItemModifyAttributesAndData.restype = ctypes.c_int32
        self._security.SecKeychainItemDelete.argtypes = [ctypes.c_void_p]
        self._security.SecKeychainItemDelete.restype = ctypes.c_int32
        self._security.SecKeychainItemFreeContent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._security.SecKeychainItemFreeContent.restype = ctypes.c_int32
        self._core_foundation.CFRelease.argtypes = [ctypes.c_void_p]
        self._core_foundation.CFRelease.restype = None

    def __del__(self) -> None:
        keychain = getattr(self, "_keychain", None)
        core_foundation = getattr(self, "_core_foundation", None)
        if keychain is not None and keychain.value and core_foundation is not None:
            try:
                core_foundation.CFRelease(keychain)
            except (AttributeError, OSError):
                pass
            keychain.value = None

    def _encoded_identity(self, account: str) -> tuple[bytes, bytes]:
        return self.service.encode("utf-8"), account.encode("utf-8")

    def _find_item(self, account: str) -> tuple[int, ctypes.c_void_p]:
        service, encoded_account = self._encoded_identity(account)
        length = ctypes.c_uint32()
        data = ctypes.c_void_p()
        item_ref = ctypes.c_void_p()
        status = int(
            self._security.SecKeychainFindGenericPassword(
                self._keychain,
                len(service),
                service,
                len(encoded_account),
                encoded_account,
                ctypes.byref(length),
                ctypes.byref(data),
                ctypes.byref(item_ref),
            )
        )
        if data.value:
            self._security.SecKeychainItemFreeContent(None, data)
        return status, item_ref

    def get(self, account: str) -> str | None:
        service, encoded_account = self._encoded_identity(account)
        length = ctypes.c_uint32()
        data = ctypes.c_void_p()
        item_ref = ctypes.c_void_p()
        status = int(
            self._security.SecKeychainFindGenericPassword(
                self._keychain,
                len(service),
                service,
                len(encoded_account),
                encoded_account,
                ctypes.byref(length),
                ctypes.byref(data),
                ctypes.byref(item_ref),
            )
        )
        if status == _ITEM_NOT_FOUND:
            return None
        if status != 0:
            raise AiCredentialError(f"读取 macOS Keychain 失败 (OSStatus {status})")
        try:
            raw = ctypes.string_at(data, length.value) if data.value else b""
            return raw.decode("utf-8")
        except UnicodeError as exc:
            raise AiCredentialError("macOS Keychain 中的 AI Key 不是有效 UTF-8") from exc
        finally:
            if data.value:
                self._security.SecKeychainItemFreeContent(None, data)
            if item_ref.value:
                self._core_foundation.CFRelease(item_ref)

    def set(self, account: str, secret: str) -> None:
        normalized = str(secret).strip()
        if not normalized:
            raise AiCredentialError("不能把空的 AI Key 写入 macOS Keychain")
        secret_bytes = normalized.encode("utf-8")
        secret_buffer = ctypes.create_string_buffer(secret_bytes)
        status, item_ref = self._find_item(account)
        try:
            if status == 0:
                update_status = int(
                    self._security.SecKeychainItemModifyAttributesAndData(
                        item_ref,
                        None,
                        len(secret_bytes),
                        ctypes.cast(secret_buffer, ctypes.c_void_p),
                    )
                )
                if update_status != 0:
                    raise AiCredentialError(
                        f"更新 macOS Keychain 失败 (OSStatus {update_status})"
                    )
                return
            if status != _ITEM_NOT_FOUND:
                raise AiCredentialError(f"查询 macOS Keychain 失败 (OSStatus {status})")
            service, encoded_account = self._encoded_identity(account)
            add_status = int(
                self._security.SecKeychainAddGenericPassword(
                    self._keychain,
                    len(service),
                    service,
                    len(encoded_account),
                    encoded_account,
                    len(secret_bytes),
                    ctypes.cast(secret_buffer, ctypes.c_void_p),
                    None,
                )
            )
            if add_status != 0:
                raise AiCredentialError(
                    f"写入 macOS Keychain 失败 (OSStatus {add_status})"
                )
        finally:
            if item_ref.value:
                self._core_foundation.CFRelease(item_ref)

    def delete(self, account: str) -> bool:
        status, item_ref = self._find_item(account)
        try:
            if status == _ITEM_NOT_FOUND:
                return False
            if status != 0:
                raise AiCredentialError(f"查询 macOS Keychain 失败 (OSStatus {status})")
            delete_status = int(self._security.SecKeychainItemDelete(item_ref))
            if delete_status != 0:
                raise AiCredentialError(
                    f"删除 macOS Keychain 项失败 (OSStatus {delete_status})"
                )
            return True
        finally:
            if item_ref.value:
                self._core_foundation.CFRelease(item_ref)


def ensure_developer_credentials(
    resources: Path,
    *,
    fallback_secrets: Mapping[str, str] | None = None,
    store: MacOSKeychainStore | None = None,
) -> dict[str, str]:
    keychain = store or MacOSKeychainStore()
    bootstrap_path = resources / AI_BOOTSTRAP_RESOURCE_NAME
    if bootstrap_path.is_file():
        bootstrap_payload = bootstrap_path.read_bytes()
        bootstrap_digest = hashlib.sha256(bootstrap_payload).hexdigest()
        stored_digest = keychain.get(DEVELOPER_BOOTSTRAP_DIGEST_ACCOUNT)
        if stored_digest != bootstrap_digest:
            import_values = decode_bootstrap_payload(bootstrap_payload)
            for env_key, account in AI_KEYCHAIN_ACCOUNTS.items():
                value = import_values.get(env_key, "").strip()
                if value:
                    keychain.set(account, value)
                else:
                    keychain.delete(account)
            keychain.set(DEVELOPER_BOOTSTRAP_DIGEST_ACCOUNT, bootstrap_digest)

    resolved: dict[str, str] = {}
    for env_key, account in AI_KEYCHAIN_ACCOUNTS.items():
        current = keychain.get(account)
        if current:
            resolved[env_key] = current
            continue
        if not fallback_secrets:
            continue
        fallback = str(fallback_secrets.get(env_key, "")).strip()
        if fallback:
            keychain.set(account, fallback)
            resolved[env_key] = fallback
    return resolved


def get_custom_api_key(*, store: MacOSKeychainStore | None = None) -> str | None:
    return (store or MacOSKeychainStore()).get(CUSTOM_API_KEY_ACCOUNT)


def set_custom_api_key(
    secret: str,
    *,
    store: MacOSKeychainStore | None = None,
) -> None:
    (store or MacOSKeychainStore()).set(CUSTOM_API_KEY_ACCOUNT, secret)


def delete_custom_api_key(*, store: MacOSKeychainStore | None = None) -> bool:
    return (store or MacOSKeychainStore()).delete(CUSTOM_API_KEY_ACCOUNT)
