#!/usr/bin/env python3
"""Completely isolated experimental AI artistic-derivative output."""
from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np

from image_metrics import _to_rgb_float_image
from sirilpy.exceptions import CommandError, DataError, SirilError


MAX_RESPONSE_BYTES = 64 * 1024 * 1024
DEFAULT_ARTISTIC_PROMPT = (
    "Create a polished artistic derivative of this deep-sky astrophotography image. "
    "Keep the source composition recognizable, preserve natural star placement, and "
    "use tasteful color, depth, and contrast. This is explicitly an artistic derivative, "
    "not scientific data. Do not add text, labels, borders, signatures, or watermarks."
)


def build_image_edit_endpoint_candidates(endpoint: str) -> List[str]:
    raw = endpoint.strip().rstrip("/")
    if not raw:
        return []
    parsed = urllib_parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return [raw]
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        return [raw + "/v1/images/edits"]
    if path.endswith("/v1"):
        return [raw + "/images/edits"]
    return [raw]


def _multipart_body(
    fields: Dict[str, str],
    *,
    image_path: Path,
) -> Tuple[bytes, str]:
    boundary = "----SeestarArtistic" + uuid.uuid4().hex
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                (
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                ).encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: image/png\r\n\r\n",
            image_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(parts), boundary


def _read_limited(response: Any) -> bytes:
    payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeError("AI artistic response exceeded 64 MiB limit")
    return payload


def _image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    raise RuntimeError("AI artistic endpoint returned unsupported image bytes")


def _extract_image_bytes(
    response_obj: Dict[str, Any],
    *,
    timeout_sec: int,
) -> Tuple[bytes, Dict[str, Any]]:
    data = response_obj.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError("AI artistic response missing data[0]")
    first = data[0]
    encoded = first.get("b64_json")
    if isinstance(encoded, str) and encoded.strip():
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise RuntimeError("AI artistic response contains invalid base64 image") from error
        if len(image_bytes) > MAX_RESPONSE_BYTES:
            raise RuntimeError("AI artistic image exceeded 64 MiB limit")
        return image_bytes, {"transport": "b64_json", "revised_prompt": first.get("revised_prompt")}

    result_url = first.get("url")
    if isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
        try:
            with urllib_request.urlopen(result_url, timeout=timeout_sec) as response:
                image_bytes = _read_limited(response)
        except (urllib_error.URLError, OSError) as error:
            raise RuntimeError(f"AI artistic result download failed: {error}") from error
        return image_bytes, {"transport": "url", "revised_prompt": first.get("revised_prompt")}
    raise RuntimeError("AI artistic response has neither b64_json nor a result URL")


def request_artistic_derivative(
    endpoint: str,
    model: str,
    api_key: str,
    prompt: str,
    image_path: Path,
    timeout_sec: int,
) -> Tuple[bytes, Dict[str, Any]]:
    endpoints = build_image_edit_endpoint_candidates(endpoint)
    if not endpoints:
        raise RuntimeError("SEESTAR_AI_ARTISTIC_ENDPOINT is empty")
    body, boundary = _multipart_body(
        {"model": model, "prompt": prompt},
        image_path=image_path,
    )
    errors: List[str] = []
    for endpoint_url in endpoints:
        request = urllib_request.Request(endpoint_url, data=body, method="POST")
        request.add_header("Authorization", f"Bearer {api_key}")
        request.add_header("Accept", "application/json")
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urllib_request.urlopen(request, timeout=timeout_sec) as response:
                raw = _read_limited(response)
            response_obj = json.loads(raw.decode("utf-8"))
            if not isinstance(response_obj, dict):
                raise RuntimeError("AI artistic endpoint returned invalid JSON object")
            image_bytes, metadata = _extract_image_bytes(
                response_obj,
                timeout_sec=timeout_sec,
            )
            parsed_endpoint = urllib_parse.urlparse(endpoint_url)
            metadata["endpoint"] = urllib_parse.urlunparse(
                parsed_endpoint._replace(query="", fragment="")
            )
            return image_bytes, metadata
        except urllib_error.HTTPError as error:
            detail = ""
            try:
                detail = error.read(512).decode("utf-8", errors="replace").strip()
            except (OSError, UnicodeError):
                pass
            errors.append(f"{endpoint_url}: HTTP {error.code} {detail}")
        except (
            urllib_error.URLError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            errors.append(f"{endpoint_url}: {error}")
    raise RuntimeError("AI artistic request failed: " + " | ".join(errors[-3:]))


def _display_preview(image_data: Any) -> np.ndarray:
    rgb = _to_rgb_float_image(np.asarray(image_data), max_side=2048)
    gray = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    high = max(float(np.quantile(gray, 0.995)), 1e-6)
    return np.flip(np.sqrt(np.clip(rgb / high, 0.0, 1.0)), axis=1)


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def run_ai_artistic_derivative(
    owner: Any,
    *,
    write_png_rgb16_func: Callable[[Path, Any], None],
    request_func: Callable[
        [str, str, str, str, Path, int], Tuple[bytes, Dict[str, Any]]
    ] = request_artistic_derivative,
) -> Optional[Path]:
    """Generate an isolated derivative without importing it back into Siril."""
    owner.ai_artistic_output_generated = False
    owner.ai_artistic_output_path = None
    if not bool(getattr(owner.cfg, "ai_artistic_derivative_enabled", False)):
        return None

    output_dir = Path(owner.work_dir) / "ai_artistic_derivative"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "artistic_report.json"
    required = {
        "SEESTAR_AI_ARTISTIC_ENDPOINT": str(
            getattr(owner.cfg, "ai_artistic_endpoint", "") or ""
        ).strip(),
        "SEESTAR_AI_ARTISTIC_MODEL": str(
            getattr(owner.cfg, "ai_artistic_model", "") or ""
        ).strip(),
        "SEESTAR_AI_ARTISTIC_API_KEY": str(
            getattr(owner.cfg, "ai_artistic_api_key", "") or ""
        ).strip(),
    }
    missing = [name for name, value in required.items() if not value]
    base_report: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": "ai_artistic_derivative",
        "scientific_pipeline_input": "process/stage10_final.fit",
        "uploaded_representation": "display_stretched_preview_max_side_2048",
        "isolated": True,
        "reimported_into_siril": False,
        "affects_pipeline_status": False,
        "disclaimer": "Artistic derivative; not scientific or calibrated astronomy data.",
        "model": required["SEESTAR_AI_ARTISTIC_MODEL"] or None,
    }
    if missing:
        report = {**base_report, "status": "skipped", "missing_config": missing}
        _write_report(report_path, report)
        owner.log.warn("[AI-Artistic] skipped; missing isolated config: " + ", ".join(missing))
        return None

    source_fit = Path(owner.process_dir) / "stage10_final.fit"
    source_preview = output_dir / "source_preview.png"
    prompt = str(getattr(owner.cfg, "ai_artistic_prompt", "") or "").strip()
    prompt = prompt or DEFAULT_ARTISTIC_PROMPT
    timeout_sec = max(
        30,
        min(600, int(getattr(owner.cfg, "ai_artistic_timeout_sec", 180))),
    )
    try:
        if not source_fit.is_file():
            raise RuntimeError(f"canonical Stage10 source missing: {source_fit}")
        owner.cmd_with_check("load", "stage10_final")
        image_data = owner.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("Siril returned empty Stage10 image pixels")
        write_png_rgb16_func(source_preview, _display_preview(image_data))

        owner.log.info(
            "[AI-Artistic] isolated image-edit request started "
            f"model={required['SEESTAR_AI_ARTISTIC_MODEL']}"
        )
        image_bytes, response_metadata = request_func(
            required["SEESTAR_AI_ARTISTIC_ENDPOINT"],
            required["SEESTAR_AI_ARTISTIC_MODEL"],
            required["SEESTAR_AI_ARTISTIC_API_KEY"],
            prompt,
            source_preview,
            timeout_sec,
        )
        extension = _image_extension(image_bytes)
        output_path = output_dir / f"result_artistic_derivative{extension}"
        for stale_extension in (".png", ".jpg", ".webp"):
            stale_path = output_dir / f"result_artistic_derivative{stale_extension}"
            if stale_path != output_path and stale_path.exists():
                stale_path.unlink()
        temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
        temp_output.write_bytes(image_bytes)
        os.replace(temp_output, output_path)
        report = {
            **base_report,
            "status": "ok",
            "source_preview": str(source_preview),
            "output_path": str(output_path),
            "response": response_metadata,
            "prompt": prompt,
        }
        _write_report(report_path, report)
        owner.ai_artistic_output_generated = True
        owner.ai_artistic_output_path = output_path
        owner.log.info(f"[AI-Artistic] isolated derivative written: {output_path}")
        return output_path
    except (
        CommandError,
        DataError,
        SirilError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        report = {
            **base_report,
            "status": "failed",
            "source_preview": str(source_preview),
            "error": str(error)[:500],
        }
        try:
            _write_report(report_path, report)
        except OSError as report_error:
            owner.log.warn(f"[AI-Artistic] report write failed: {report_error}")
        owner.log.warn(f"[AI-Artistic] isolated experiment failed: {error}")
        return None
