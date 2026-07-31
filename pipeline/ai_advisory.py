from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from image_metrics import format_feature_summary
from models import TargetType
from review_bundle import image_data_to_data_url, image_path_to_data_url
from sirilpy.exceptions import CommandError, DataError, SirilError


DEFAULT_AI_PROMPT = (
    "Conservative deep-sky astrophotography enhancement only. "
    "Preserve astronomical realism, faint structures, and natural star colors. "
    "Do not invent objects, do not oversaturate, do not clip black background, "
    "do not increase star size, and avoid halos or artificial sharpening artifacts."
)
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
VISUAL_ACCEPTANCE_STAGE_KEYS = frozenset(
    {
        "stage3_background_extraction",
        "stage6_star_separation",
        "stage7_stretching",
        "stage8_nebula_enhancement",
        "stage11_ai_postprocess",
    }
)


def network_mode_enabled() -> bool:
    """Return whether this run explicitly opted in to outbound network access."""
    return (
        os.getenv("SEESTAR_NETWORK_MODE", "0").strip().lower()
        in ENV_TRUE_VALUES
    )


def _selection_payload(
    obj: Dict[str, Any],
    *envelope_keys: str,
) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    for key in envelope_keys:
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return obj


def _selection_rationale(pipeline: object, payload: Dict[str, Any]) -> str:
    return pipeline._short_text(
        str(
            payload.get(
                "rationale",
                payload.get("summary", payload.get("reason", "")),
            )
        ),
        180,
    )


def _clamp_float(value: object, min_value: float, max_value: float) -> float:
    try:
        fvalue = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        fvalue = min_value
    return max(min_value, min(max_value, fvalue))


def _clamp_int(value: object, min_value: int, max_value: int) -> int:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        ivalue = min_value
    return max(min_value, min(max_value, ivalue))


def post_json_with_auth(
    endpoint: str,
    payload: Dict[str, Any],
    api_key: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    if not network_mode_enabled():
        raise RuntimeError(
            "outbound AI request blocked: SEESTAR_NETWORK_MODE is disabled"
        )
    if not endpoint.lower().startswith(("http://", "https://")):
        raise ValueError("SEESTAR_AI_ENDPOINT must be an absolute http(s) URL")

    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read()
    except urllib_error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace").strip()
        except (OSError, UnicodeError):
            detail = ""
        if len(detail) > 280:
            detail = detail[:280] + "..."
        raise RuntimeError(f"AI endpoint returned HTTP {e.code}: {detail}") from e
    except urllib_error.URLError as e:
        raise RuntimeError(f"AI endpoint request failed: {e}") from e

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError("AI endpoint returned non-JSON body") from e


def build_ai_chat_endpoint_candidates(endpoint: str) -> List[str]:
    raw = endpoint.strip().rstrip("/")
    if not raw:
        return []
    parsed = urllib_parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return [raw]

    path = parsed.path.rstrip("/")
    candidates: List[str] = []
    append_raw = True

    if path in {"", "/"}:
        candidates.append(raw + "/v1/chat/completions")
        append_raw = False
    elif path.endswith("/v1"):
        candidates.append(raw + "/chat/completions")
        append_raw = False
    elif path.endswith("/v1/chat"):
        candidates.append(raw + "/completions")
        append_raw = False
    elif path.endswith("/chat/completions"):
        candidates.append(raw)
        append_raw = False

    if append_raw:
        candidates.append(raw)

    deduped: List[str] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def advisor_mode(pipeline: object) -> str:
    mode = str(getattr(pipeline.cfg, "ai_advisor_mode", "text") or "text").strip().lower()
    return mode if mode in {"text", "multimodal"} else "text"


def _current_image_data_url(pipeline: object) -> Optional[str]:
    try:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            return None
        return image_data_to_data_url(image_data)
    except (
        CommandError,
        DataError,
        SirilError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        AttributeError,
    ) as error:
        pipeline.log.warn(f"[AI] multimodal preview unavailable; using text advisor: {error}")
        return None


def _multimodal_user_content(
    prompt: str,
    image_data_urls: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, data_url in image_data_urls:
        content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            }
        )
    return content


def extract_chat_content(pipeline: object, response_obj: Dict[str, Any]) -> str:
    choices = response_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("AI chat response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("AI chat response choices[0] is invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("AI chat response missing message field")
    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content:
            return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            txt = item.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
        merged = "\n".join(parts).strip()
        if merged:
            return merged

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        reasoning_content = reasoning_content.strip()
        if reasoning_content:
            pipeline.log.warn(
                "[AI] Chat response content is empty, fallback to reasoning_content"
            )
            return reasoning_content
    raise RuntimeError("AI chat response content is empty")


def extract_adjustments_from_text(pipeline: object, text: str) -> Optional[Dict[str, Any]]:
    """Compatibility parser that accepts only a candidate identifier.

    Numeric parameters in free-form model text are intentionally ignored.
    """
    match = re.search(
        r"[\"']?selected_candidate_id[\"']?\s*[:=：]\s*[\"']?([a-z0-9_-]+)",
        text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    summary = "fallback: parsed candidate id from non-json AI response"
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line:
        summary = pipeline._short_text(first_line, max_len=96)
    return {
        "summary": summary,
        "selected_candidate_id": match.group(1).strip().lower(),
    }


def extract_first_json_object(pipeline: object, text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise RuntimeError("AI plan content is empty")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

    parsed_from_text = extract_adjustments_from_text(pipeline, text)
    if parsed_from_text is not None:
        pipeline.log.warn(
            "[AI] Plan is not strict JSON; extracted candidate id from plain text fallback"
        )
        return parsed_from_text

    raise RuntimeError("AI plan is not valid JSON object")


def extract_stage_advisory_from_text(
    pipeline: object,
    stage_name: str,
    text: str,
) -> Optional[Dict[str, Any]]:
    raw_text = text or ""
    lowered = raw_text.lower()
    first_line = raw_text.splitlines()[0].strip().lower() if raw_text else ""
    if "required output json schema" in lowered or "observations json" in lowered:
        return None

    def first_useful_line() -> str:
        for line in raw_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith("the user wants"):
                continue
            return cleaned
        return first_line or "parsed from non-json ai text"

    if stage_name == "stage6_stretch_plan":
        candidate_id = ""
        if (
            "autostretch" in lowered
            or "auto stretch" in lowered
            or "automatic stretch" in lowered
            or "自动拉伸" in raw_text
        ):
            candidate_id = "autostretch"
        elif "asinh_ghs" in lowered or "asinh+ghs" in lowered:
            candidate_id = "asinh_ghs"
        elif "ghs" in lowered or "autoghs" in lowered or "generalized hyperbolic" in lowered:
            candidate_id = "ghs"
        elif "asinh" in lowered or "arcsinh" in lowered or "反双曲" in raw_text:
            candidate_id = "asinh"
        if not candidate_id:
            return None
        return {
            "stage6_stretch_selection": {
                "rationale": pipeline._short_text(first_useful_line(), 140),
                "selected_candidate_id": candidate_id,
            }
        }
    if stage_name == "stage7_starless_plan":
        if "syqon_axiom_standard" in lowered or "axiom" in lowered:
            candidate_id = "syqon_axiom_standard"
        elif (
            "syqon_large_context" in lowered
            or "large context" in lowered
            or "larger tile" in lowered
            or "1024" in lowered
        ):
            candidate_id = "syqon_large_context"
        elif (
            "syqon_standard" in lowered
            or "standard" in lowered
            or "512" in lowered
        ):
            candidate_id = "syqon_standard"
        else:
            return None
        return {
            "stage7_starless_selection": {
                "rationale": pipeline._short_text(first_useful_line(), 140),
                "selected_candidate_id": candidate_id,
            }
        }
    if stage_name == "stage7_quality":
        residual = any(
            token in lowered
            for token in (
                "residual star",
                "residual_stars",
                "remaining stars",
                "leftover stars",
            )
        ) or "残星" in raw_text
        missing = any(
            token in lowered
            for token in ("missing stars", "starmask missing", "under mask", "too sparse")
        ) or "缺星" in raw_text
        too_wide = any(
            token in lowered
            for token in ("too wide", "over-wide", "bloated mask", "mask is wide")
        ) or "过宽" in raw_text
        verdict = "poor" if residual or missing or too_wide else "ok"
        action = "retry_syqon" if verdict == "poor" else "accept"
        if "fallback" in lowered or "sasp" in lowered:
            action = "fallback_sasp"
        if "degrade" in lowered:
            action = "degrade"
        if "conservative" in lowered:
            star_remix_candidate_id = "conservative"
        elif "reduced" in lowered:
            star_remix_candidate_id = "reduced"
        else:
            star_remix_candidate_id = "full" if verdict == "ok" else "reduced"
        if "guarded" in lowered:
            residual_candidate_id = "guarded"
        elif residual or "light" in lowered:
            residual_candidate_id = "light"
        else:
            residual_candidate_id = "off"
        quality: Dict[str, Any] = {
            "verdict": verdict,
            "summary": pipeline._short_text(first_useful_line(), 160),
            "residual_stars": residual,
            "starmask_missing": missing,
            "starmask_too_wide": too_wide,
            "recommended_action": action,
            "star_remix_candidate_id": star_remix_candidate_id,
            "residual_suppression_candidate_id": residual_candidate_id,
        }
        return {"stage7_quality": quality}
    if stage_name == "stage8_processing_plan":
        if "detail_preserving" in lowered or "detail preserving" in lowered:
            candidate_id = "detail_preserving"
        elif "conservative" in lowered:
            candidate_id = "conservative"
        elif "preserve" in lowered or "no enhancement" in lowered:
            candidate_id = "preserve"
        elif "balanced" in lowered:
            candidate_id = "balanced"
        else:
            return None
        return {
            "stage8_processing_selection": {
                "rationale": pipeline._short_text(first_useful_line(), 140),
                "selected_candidate_id": candidate_id,
            }
        }
    return None


def request_stage_ai_advisory(
    pipeline: object,
    stage_name: str,
    schema_text: str,
    observations: Dict[str, Any],
    *,
    max_tokens: int = 700,
    image_paths: Optional[List[Tuple[str, Path]]] = None,
    allow_text_fallback: bool = True,
) -> Dict[str, Any]:
    breaker_reason = pipeline._ai_stage_circuit_breaker.get(stage_name)
    if breaker_reason:
        raise RuntimeError(f"{stage_name} AI advisory circuit open: {breaker_reason}")

    endpoint = pipeline.cfg.ai_endpoint.strip()
    model = pipeline.cfg.ai_model.strip()
    api_key = pipeline.cfg.ai_api_key.strip()
    timeout_sec = int(pipeline.cfg.ai_timeout_sec)
    endpoint_candidates = pipeline._build_ai_chat_endpoint_candidates(endpoint)
    if not endpoint_candidates:
        raise RuntimeError("SEESTAR_AI_ENDPOINT is empty")

    system_prompt = (
        "You are a conservative astronomical image-processing quality advisor. "
        "Return a strict JSON object as the final answer. "
        "If you reason internally, put only the final JSON object on the last line. "
        "Never suggest destructive edits. "
        "Preserve astronomical realism, faint structures, natural star size, "
        "and natural star color."
    )
    user_prompt = (
        f"Stage: {stage_name}\n"
        f"Model prompt context: {(pipeline.cfg.ai_prompt or DEFAULT_AI_PROMPT).strip()}\n"
        "Observations JSON:\n"
        f"{json.dumps(observations, ensure_ascii=False, sort_keys=True)}\n"
        "Required output JSON schema:\n"
        f"{schema_text}\n"
        "Final answer must be one minified JSON object only. No markdown."
    )
    text_payload_base = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    payload_variants: List[Tuple[str, Dict[str, Any]]] = []
    if advisor_mode(pipeline) == "multimodal":
        image_data_urls: List[Tuple[str, str]] = []
        for label, path in image_paths or []:
            try:
                image_data_urls.append((label, image_path_to_data_url(Path(path))))
            except OSError as error:
                pipeline.log.warn(f"[AI] visual evidence unavailable ({path}): {error}")
        if image_paths is None:
            current_url = _current_image_data_url(pipeline)
            if current_url:
                image_data_urls.append(("Current stage image", current_url))
        if image_data_urls:
            vision_payload = copy.deepcopy(text_payload_base)
            vision_payload["messages"][1]["content"] = _multimodal_user_content(
                user_prompt,
                image_data_urls,
            )
            payload_variants.append(("multimodal", vision_payload))
    if allow_text_fallback:
        payload_variants.append(("text", text_payload_base))
    elif not payload_variants:
        raise RuntimeError(f"{stage_name} multimodal evidence unavailable")

    temperatures = (1.0,) if "kimi" in model.lower() else (0.1, 1.0)
    attempt_errors: List[str] = []
    parse_failures: List[str] = []
    for endpoint_url in endpoint_candidates:
        for request_mode, payload_base in payload_variants:
            if request_mode == "text" and payload_variants[0][0] == "multimodal":
                pipeline.log.warn(f"[AI] {stage_name} falling back to text-only advisor")
            for temperature in temperatures:
                for json_mode in (True, False):
                    payload = copy.deepcopy(payload_base)
                    payload["temperature"] = temperature
                    if json_mode:
                        payload["response_format"] = {"type": "json_object"}
                    try:
                        response_obj = pipeline._post_json_with_auth(
                            endpoint_url, payload, api_key, timeout_sec
                        )
                        content = pipeline._extract_chat_content(response_obj)
                        pipeline._write_ai_raw_response(
                            stage_name,
                            endpoint_url=endpoint_url,
                            temperature=temperature,
                            json_mode=json_mode,
                            response_obj=response_obj,
                            content=content,
                        )
                        try:
                            plan_obj = pipeline._extract_first_json_object(content)
                        except (
                            RuntimeError,
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ):
                            parse_failures.append(
                                f"endpoint={endpoint_url},mode={request_mode},"
                                f"temperature={temperature},json_mode={json_mode}"
                            )
                            plan_obj = pipeline._extract_stage_advisory_from_text(
                                stage_name, content
                            )
                            if plan_obj is None:
                                raise
                            pipeline.log.warn(
                                f"[AI] {stage_name} parsed advisory from non-JSON text"
                            )
                        if temperature != 0.1:
                            pipeline.log.warn(
                                f"[AI] {stage_name} advisory used temperature fallback={temperature}"
                            )
                        return plan_obj
                    except (OSError, RuntimeError, TypeError, ValueError) as e:
                        if "response_obj" in locals() or "content" in locals():
                            try:
                                pipeline._write_ai_raw_response(
                                    stage_name,
                                    endpoint_url=endpoint_url,
                                    temperature=temperature,
                                    json_mode=json_mode,
                                    response_obj=locals().get("response_obj"),
                                    content=locals().get("content"),
                                    error_text=str(e),
                                )
                            except (OSError, RuntimeError, TypeError, ValueError):
                                pass
                        attempt_errors.append(
                            "endpoint="
                            f"{endpoint_url},mode={request_mode},temperature={temperature},"
                            f"json_mode={json_mode},error={e}"
                        )
                        err_text = str(e).lower()
                        if json_mode and (
                            "response_format" in err_text
                            or "json_object" in err_text
                            or "unsupported" in err_text
                        ):
                            continue
                        if temperature == 0.1 and "only 1 is allowed for this model" in err_text:
                            pipeline.log.warn(
                                f"[AI] {stage_name} model requires temperature=1, retrying"
                            )
                            break
                        pipeline.log.warn(
                            f"[AI] {stage_name} advisory failed "
                            f"(endpoint={endpoint_url}, mode={request_mode}, "
                            f"temperature={temperature}, json_mode={json_mode}): {e}"
                        )
                    finally:
                        if "response_obj" in locals():
                            del response_obj
                        if "content" in locals():
                            del content
    if parse_failures:
        reason = "json parse failed"
        pipeline._ai_stage_circuit_breaker[stage_name] = reason
        pipeline.log.warn(
            f"[AI] {stage_name} advisory circuit opened for this run: {reason}"
        )
    if attempt_errors:
        raise RuntimeError(" | ".join(attempt_errors[-3:]))
    raise RuntimeError(f"{stage_name} AI advisory failed")


def request_visual_acceptance(
    pipeline: object,
    stage_key: str,
    review_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Ask a vision model to review the selected candidate without blocking the pipeline."""
    if stage_key not in VISUAL_ACCEPTANCE_STAGE_KEYS:
        return None
    if advisor_mode(pipeline) != "multimodal":
        return None
    if not bool(getattr(pipeline.cfg, "ai_post_enabled", False)):
        return None
    if not (
        pipeline.cfg.ai_endpoint.strip()
        and pipeline.cfg.ai_model.strip()
        and pipeline.cfg.ai_api_key.strip()
    ):
        return None

    previews = review_payload.get("previews") or {}
    image_paths: List[Tuple[str, Path]] = []
    for key, label in (
        ("before_preview", "Before processing"),
        ("after_preview", "Selected candidate after processing"),
        ("signed_luminance_difference", "Signed luminance difference: red=increased, blue=decreased"),
    ):
        path_text = previews.get(key)
        if path_text:
            image_paths.append((label, Path(str(path_text))))
    if len(image_paths) < 2:
        raise RuntimeError("visual acceptance requires before and after previews")

    candidates = review_payload.get("candidates") or []
    observations = {
        "stage": stage_key,
        "context": review_payload.get("context") or {},
        "metrics": review_payload.get("metrics") or {},
        "candidates": [
            {
                key: candidate.get(key)
                for key in (
                    "id",
                    "name",
                    "label",
                    "attempt",
                    "method",
                    "status",
                    "quality_ok",
                    "risk_score",
                    "selection_status",
                )
                if candidate.get(key) is not None
            }
            for candidate in candidates
        ],
    }
    schema = (
        "{\n"
        '  "visual_acceptance": {\n'
        '    "verdict": "accept|review_required|reject",\n'
        '    "confidence": 0.0,\n'
        '    "summary": "short visual assessment",\n'
        '    "issues": ["visible issue"],\n'
        '    "recommended_parameter_ranges": {"parameter": "safe range"}\n'
        "  }\n"
        "}"
    )
    raw = request_stage_ai_advisory(
        pipeline,
        f"{stage_key}_visual_acceptance",
        schema,
        observations,
        max_tokens=700,
        image_paths=image_paths,
        allow_text_fallback=False,
    )
    result = raw.get("visual_acceptance") if isinstance(raw, dict) else None
    if not isinstance(result, dict):
        result = raw if isinstance(raw, dict) else {}
    verdict = str(result.get("verdict") or "review_required").strip().lower()
    aliases = {
        "ok": "accept",
        "accepted": "accept",
        "pass": "accept",
        "review": "review_required",
        "warning": "review_required",
        "rejected": "reject",
        "fail": "reject",
    }
    verdict = aliases.get(verdict, verdict)
    if verdict not in {"accept", "review_required", "reject"}:
        verdict = "review_required"
    issues = result.get("issues")
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    ranges = result.get("recommended_parameter_ranges")
    if not isinstance(ranges, dict):
        ranges = {}
    return {
        "verdict": verdict,
        "confidence": _clamp_float(result.get("confidence", 0.0), 0.0, 1.0),
        "summary": pipeline._short_text(str(result.get("summary") or ""), 240),
        "issues": [pipeline._short_text(str(item), 180) for item in issues[:6]],
        "recommended_parameter_ranges": {
            pipeline._short_text(str(key), 80): pipeline._short_text(str(value), 120)
            for key, value in list(ranges.items())[:12]
        },
    }


def _stage6_stretch_candidate_presets(
    pipeline: object,
) -> Dict[str, Dict[str, Any]]:
    """Build the code-owned compatibility stretch candidate set."""
    params = {
        "asinh_stretch": _clamp_float(
            pipeline.cfg.asinh_stretch, 1.6, 3.6
        ),
        "asinh_offset": _clamp_float(
            pipeline.cfg.asinh_offset, 0.0005, 0.006
        ),
        "ghs_shadowsclip": _clamp_float(
            pipeline.cfg.ghs_shadowsclip, -3.6, -1.8
        ),
        "ghs_stretchamount": _clamp_float(
            pipeline.cfg.ghs_stretchamount, 1.0, 2.8
        ),
    }
    return {
        candidate_id: {
            "candidate_id": candidate_id,
            "method": candidate_id,
            "params": dict(params),
        }
        for candidate_id in ("asinh", "asinh_ghs", "ghs", "autostretch")
    }


def normalize_stage6_ai_plan(
    pipeline: object,
    obj: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve an AI response to one code-owned stretch candidate."""
    plan = _selection_payload(
        obj,
        "stage6_stretch_selection",
        "stage6_stretch_plan",
    )
    candidate_id = str(plan.get("selected_candidate_id", "")).strip().lower()
    candidate = _stage6_stretch_candidate_presets(pipeline).get(candidate_id)
    if candidate is None:
        return None
    return {
        **candidate,
        "selected_candidate_id": candidate_id,
        "summary": _selection_rationale(pipeline, plan),
        "confidence": _clamp_float(plan.get("confidence", 0.0), 0.0, 1.0),
        "selection_contract": "candidate_id_only",
    }


def request_stage6_stretch_plan(
    pipeline: object,
    baseline_features: Optional[object],
    baseline_quality: Optional[object],
) -> Optional[Dict[str, Any]]:
    candidates = _stage6_stretch_candidate_presets(pipeline)
    observations = {
        "target_type": (
            getattr(getattr(pipeline, "auto_tune_result", None), "target_type", TargetType.UNKNOWN).name
        ),
        "features": asdict(baseline_features) if baseline_features else None,
        "quality_metrics": asdict(baseline_quality) if baseline_quality else None,
        "allowed_candidates": list(candidates.values()),
        "constraints": {
            "bg_median_min": pipeline.cfg.stage6_bg_median_min,
            "black_pixel_ratio_max": pipeline.cfg.stage6_black_pixel_ratio_max,
            "highlight_clip_ratio_max": pipeline.cfg.stage6_highlight_clip_ratio_max,
            "star_growth_ratio_max": pipeline.cfg.stage6_star_growth_ratio_max,
            "model_output": "selected_candidate_id only; parameters are code-owned",
        },
    }
    schema = (
        "{\n"
        '  "stage6_stretch_selection": {\n'
        '    "selected_candidate_id": "asinh|asinh_ghs|ghs|autostretch",\n'
        '    "confidence": 0.0,\n'
        '    "rationale": "short reason"\n'
        "  }\n"
        "}"
    )
    try:
        raw_plan = pipeline._request_stage_ai_advisory(
            "stage6_stretch_plan", schema, observations
        )
        plan = pipeline._normalize_stage6_ai_plan(raw_plan)
        if plan is None:
            raise RuntimeError("stage6 AI selection has unknown candidate id")
        return plan
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.warn(f"[AI] stage6 stretch advisory unavailable: {e}")
        return None


def normalize_stage7_stretch_selection(
    pipeline: object,
    obj: Dict[str, Any],
    allowed_candidate_ids: List[str],
) -> Optional[Dict[str, Any]]:
    """Validate a model selection against the post-gate candidate allow-list."""
    selection = _selection_payload(obj, "stage7_stretch_selection")
    candidate_id = str(
        selection.get("selected_candidate_id", "")
    ).strip().lower()
    allowed = {
        str(item).strip().lower()
        for item in allowed_candidate_ids
        if str(item).strip()
    }
    if not candidate_id or candidate_id not in allowed:
        return None
    return {
        "selected_candidate_id": candidate_id,
        "confidence": _clamp_float(
            selection.get("confidence", 0.0), 0.0, 1.0
        ),
        "rationale": _selection_rationale(pipeline, selection),
        "allowed_candidate_ids": sorted(allowed),
        "selection_contract": "candidate_id_only_after_hard_gates",
    }


def request_stage7_stretch_selection(
    pipeline: object,
    accepted_attempts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ask AI to choose only among locally generated, hard-gate-passing IDs."""
    if not pipeline._ai_stage_advisory_enabled("ai_stage6_enabled"):
        return None
    selectable = [
        attempt
        for attempt in accepted_attempts
        if not bool(attempt.get("explicit_fallback"))
        and str(attempt.get("name") or "").strip()
    ]
    allowed_ids = [str(attempt["name"]).strip() for attempt in selectable]
    if not allowed_ids:
        return None

    observations = {
        "target_type": (
            pipeline._active_target_type()
            if hasattr(pipeline, "_active_target_type")
            else "unknown"
        ),
        "allowed_candidates": [
            {
                "candidate_id": str(attempt["name"]),
                "method": str(attempt.get("method") or ""),
                "risk_score": float(attempt.get("risk_score", 0.0) or 0.0),
                "quality_metrics": attempt.get("metrics"),
                "pixel_stats": attempt.get("pixel_stats"),
                "preview_target_attainment": attempt.get(
                    "preview_target_attainment"
                ),
                "target_local_quality": attempt.get("target_local_quality"),
                "background_quality_gate": attempt.get(
                    "background_quality_gate"
                ),
            }
            for attempt in selectable
        ],
        "constraints": {
            "allowed_candidate_ids": allowed_ids,
            "all_candidates_passed_hard_gates": True,
            "model_must_not_return_parameters": True,
        },
    }
    schema = (
        "{\n"
        '  "stage7_stretch_selection": {\n'
        f'    "selected_candidate_id": "one of: {"|".join(allowed_ids)}",\n'
        '    "confidence": 0.0,\n'
        '    "rationale": "short reason"\n'
        "  }\n"
        "}"
    )
    try:
        raw = pipeline._request_stage_ai_advisory(
            "stage7_stretch_selection",
            schema,
            observations,
        )
        selection = normalize_stage7_stretch_selection(
            pipeline,
            raw,
            allowed_ids,
        )
        if selection is None:
            raise RuntimeError(
                "stage7 AI selection is not in the allowed candidate ids"
            )
        return selection
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        pipeline.log.warn(
            f"[AI] stage7 stretch selection unavailable: {error}"
        )
        return None


def stage7_starless_candidate_presets(
    pipeline: object,
) -> Dict[str, Dict[str, Any]]:
    candidates = {
        "syqon_standard": {
            "candidate_id": "syqon_standard",
            "tile_size": 512,
            "overlap": 64,
            "use_axiom": False,
        },
        "syqon_large_context": {
            "candidate_id": "syqon_large_context",
            "tile_size": 1024,
            "overlap": 128,
            "use_axiom": False,
        },
    }
    if pipeline._syqon_axiom_model_available():
        candidates["syqon_axiom_standard"] = {
            "candidate_id": "syqon_axiom_standard",
            "tile_size": 512,
            "overlap": 64,
            "use_axiom": True,
        }
    return candidates


def normalize_stage7_starless_plan(
    pipeline: object,
    obj: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    plan = _selection_payload(
        obj,
        "stage7_starless_selection",
        "stage7_starless_plan",
    )
    candidate_id = str(plan.get("selected_candidate_id", "")).strip().lower()
    candidate = stage7_starless_candidate_presets(pipeline).get(candidate_id)
    if candidate is None:
        return None
    return {
        **candidate,
        "selected_candidate_id": candidate_id,
        "summary": _selection_rationale(pipeline, plan),
        "confidence": _clamp_float(plan.get("confidence", 0.0), 0.0, 1.0),
        "selection_contract": "candidate_id_only",
    }


def request_stage7_starless_plan(pipeline: object) -> Optional[Dict[str, Any]]:
    if not pipeline._ai_stage_advisory_enabled("ai_stage7_enabled"):
        return None
    features = pipeline._measure_current_features()
    quality = pipeline._measure_current_quality()
    candidates = stage7_starless_candidate_presets(pipeline)
    observations = {
        "features": asdict(features) if features else None,
        "quality_metrics": asdict(quality) if quality else None,
        "allowed_candidates": list(candidates.values()),
        "constraints": {
            "allowed_candidate_ids": list(candidates),
            "model_must_not_return_parameters": True,
        },
    }
    schema = (
        "{\n"
        '  "stage7_starless_selection": {\n'
        f'    "selected_candidate_id": "one of: {"|".join(candidates)}",\n'
        '    "confidence": 0.0,\n'
        '    "rationale": "short reason"\n'
        "  }\n"
        "}"
    )
    try:
        raw = pipeline._request_stage_ai_advisory(
            "stage7_starless_plan", schema, observations
        )
        plan = pipeline._normalize_stage7_starless_plan(raw)
        if plan is None:
            raise RuntimeError(
                "stage7 starless AI selection has unknown candidate id"
            )
        return plan
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.warn(f"[AI] stage7 starless plan unavailable: {e}")
        return None


def normalize_stage7_ai_quality(pipeline: object, obj: Dict[str, Any]) -> Dict[str, Any]:
    quality = obj.get("stage7_quality") if isinstance(obj, dict) else None
    if not isinstance(quality, dict):
        quality = obj if isinstance(obj, dict) else {}

    verdict = str(quality.get("verdict", quality.get("status", ""))).strip().lower()
    if verdict not in {"ok", "poor", "bad", "degraded"}:
        verdict = "ok"

    def as_bool(name: str) -> bool:
        value = quality.get(name, False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ENV_TRUE_VALUES
        return bool(value)

    action = str(quality.get("recommended_action", "accept")).strip().lower()
    if action not in {"accept", "retry_syqon", "fallback_sasp", "degrade"}:
        action = "accept" if verdict == "ok" else "retry_syqon"

    star_remix_candidates = {
        "full": 1.0,
        "reduced": 0.75,
        "conservative": 0.50,
    }
    residual_suppression_candidates = {
        "off": 0.0,
        "light": 0.08,
        "guarded": 0.16,
    }
    default_star_remix_id = (
        "full"
        if verdict == "ok"
        and not as_bool("starmask_missing")
        and not as_bool("starmask_too_wide")
        else "reduced"
    )
    default_residual_id = "light" if as_bool("residual_stars") else "off"
    star_remix_id = str(
        quality.get("star_remix_candidate_id", default_star_remix_id)
    ).strip().lower()
    residual_id = str(
        quality.get(
            "residual_suppression_candidate_id",
            default_residual_id,
        )
    ).strip().lower()
    if star_remix_id not in star_remix_candidates:
        star_remix_id = default_star_remix_id
    if residual_id not in residual_suppression_candidates:
        residual_id = default_residual_id

    return {
        "verdict": verdict,
        "residual_stars": as_bool("residual_stars"),
        "starmask_missing": as_bool("starmask_missing"),
        "starmask_too_wide": as_bool("starmask_too_wide"),
        "recommended_action": action,
        "star_remix_candidate_id": star_remix_id,
        "stage9_star_intensity_scale": star_remix_candidates[
            star_remix_id
        ],
        "residual_suppression_candidate_id": residual_id,
        "residual_suppression_strength": residual_suppression_candidates[
            residual_id
        ],
        "summary": pipeline._short_text(str(quality.get("summary", "")), 180),
        "selection_contract": "candidate_id_only",
    }


def request_stage7_quality_ai(
    pipeline: object,
    observations: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not pipeline._ai_stage_advisory_enabled("ai_stage7_enabled"):
        return None
    schema = (
        "{\n"
        '  "stage7_quality": {\n'
        '    "verdict": "ok|poor",\n'
        '    "summary": "short reason",\n'
        '    "residual_stars": false,\n'
        '    "starmask_missing": false,\n'
        '    "starmask_too_wide": false,\n'
        '    "star_remix_candidate_id": "full|reduced|conservative",\n'
        '    "residual_suppression_candidate_id": "off|light|guarded",\n'
        '    "recommended_action": "accept|retry_syqon|fallback_sasp|degrade"\n'
        "  }\n"
        "}"
    )
    try:
        vision_paths: Optional[List[Tuple[str, Path]]] = None
        if advisor_mode(pipeline) == "multimodal":
            try:
                pipeline.cmd_with_check("load", "starless")
            except (
                CommandError,
                DataError,
                SirilError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                AttributeError,
            ) as error:
                pipeline.log.warn(
                    f"[AI] stage7 starless preview load failed; using text advisor: {error}"
                )
                vision_paths = []
        raw = pipeline._request_stage_ai_advisory(
            "stage7_quality", schema, observations, image_paths=vision_paths
        )
        return pipeline._normalize_stage7_ai_quality(raw)
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.warn(f"[AI] stage7 quality advisory unavailable: {e}")
        return None


def stage8_processing_candidate_presets(
    pipeline: object,
) -> Dict[str, Dict[str, Any]]:
    saturation = _clamp_float(
        pipeline.cfg.nebula_saturation, 0.0, 0.65
    )
    bg_factor = _clamp_int(pipeline.cfg.nebula_bg_factor, 0, 3)
    return {
        "preserve": {
            "candidate_id": "preserve",
            "saturation": 0.0,
            "bg_factor": 0,
            "unsharp_radius": 0.0,
            "unsharp_amount": 0.0,
            "apply_after_plugins": False,
        },
        "conservative": {
            "candidate_id": "conservative",
            "saturation": min(saturation, 0.08),
            "bg_factor": min(bg_factor, 1),
            "unsharp_radius": 0.55,
            "unsharp_amount": 0.18,
            "apply_after_plugins": True,
        },
        "balanced": {
            "candidate_id": "balanced",
            "saturation": saturation,
            "bg_factor": bg_factor,
            "unsharp_radius": 0.8,
            "unsharp_amount": 0.35,
            "apply_after_plugins": True,
        },
        "detail_preserving": {
            "candidate_id": "detail_preserving",
            "saturation": min(saturation, 0.12),
            "bg_factor": min(bg_factor, 1),
            "unsharp_radius": 0.65,
            "unsharp_amount": 0.25,
            "apply_after_plugins": True,
        },
    }


def normalize_stage8_processing_plan(
    pipeline: object,
    obj: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    plan = _selection_payload(
        obj,
        "stage8_processing_selection",
        "stage8_processing_plan",
    )
    candidate_id = str(plan.get("selected_candidate_id", "")).strip().lower()
    candidate = stage8_processing_candidate_presets(pipeline).get(candidate_id)
    if candidate is None:
        return None
    return {
        **candidate,
        "selected_candidate_id": candidate_id,
        "summary": _selection_rationale(pipeline, plan),
        "confidence": _clamp_float(plan.get("confidence", 0.0), 0.0, 1.0),
        "selection_contract": "candidate_id_only",
    }


def request_stage8_processing_plan(pipeline: object) -> Optional[Dict[str, Any]]:
    if not pipeline._ai_stage_advisory_enabled("ai_stage8_enabled"):
        return None
    features = pipeline._measure_current_features()
    quality = pipeline._measure_current_quality()
    candidates = stage8_processing_candidate_presets(pipeline)
    observations = {
        "features": asdict(features) if features else None,
        "quality_metrics": asdict(quality) if quality else None,
        "allowed_candidates": list(candidates.values()),
        "constraints": {
            "allowed_candidate_ids": list(candidates),
            "model_must_not_return_parameters": True,
        },
    }
    schema = (
        "{\n"
        '  "stage8_processing_selection": {\n'
        f'    "selected_candidate_id": "one of: {"|".join(candidates)}",\n'
        '    "confidence": 0.0,\n'
        '    "rationale": "short reason"\n'
        "  }\n"
        "}"
    )
    try:
        raw = pipeline._request_stage_ai_advisory(
            "stage8_processing_plan", schema, observations
        )
        plan = pipeline._normalize_stage8_processing_plan(raw)
        if plan is None:
            raise RuntimeError(
                "stage8 AI selection has unknown candidate id"
            )
        return plan
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.warn(f"[AI] stage8 processing plan unavailable: {e}")
        return None


def normalize_stage8_ai_quality(pipeline: object, obj: Dict[str, Any]) -> Dict[str, Any]:
    quality = obj.get("stage8_quality") if isinstance(obj, dict) else None
    if not isinstance(quality, dict):
        quality = obj if isinstance(obj, dict) else {}
    verdict = str(quality.get("verdict", quality.get("status", ""))).strip().lower()
    if verdict not in {"ok", "poor", "bad", "degraded"}:
        verdict = "ok"

    def as_bool(name: str) -> bool:
        value = quality.get(name, False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ENV_TRUE_VALUES
        return bool(value)

    action = str(quality.get("recommended_action", "accept")).strip().lower()
    if action not in {"accept", "blue_guard", "conservative_rerun", "rollback"}:
        action = "accept" if verdict == "ok" else "conservative_rerun"
    blue_guard_candidates = {
        "strict": 0.07,
        "balanced": 0.10,
        "permissive": 0.14,
    }
    default_blue_guard_id = (
        "strict" if as_bool("blue_bias") else "balanced"
    )
    blue_guard_id = str(
        quality.get(
            "blue_guard_candidate_id",
            default_blue_guard_id,
        )
    ).strip().lower()
    if blue_guard_id not in blue_guard_candidates:
        blue_guard_id = default_blue_guard_id
    return {
        "verdict": verdict,
        "oversaturated": as_bool("oversaturated"),
        "blue_bias": as_bool("blue_bias"),
        "microcontrast_overdone": as_bool("microcontrast_overdone"),
        "recommended_action": action,
        "blue_guard_candidate_id": blue_guard_id,
        "target_blue_excess": blue_guard_candidates[blue_guard_id],
        "summary": pipeline._short_text(str(quality.get("summary", "")), 180),
        "selection_contract": "candidate_id_only",
    }


def request_stage8_quality_ai(
    pipeline: object,
    observations: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not pipeline._ai_stage_advisory_enabled("ai_stage8_enabled"):
        return None
    schema = (
        "{\n"
        '  "stage8_quality": {\n'
        '    "verdict": "ok|poor",\n'
        '    "summary": "short reason",\n'
        '    "oversaturated": false,\n'
        '    "blue_bias": false,\n'
        '    "microcontrast_overdone": false,\n'
        '    "blue_guard_candidate_id": "strict|balanced|permissive",\n'
        '    "recommended_action": "accept|blue_guard|conservative_rerun|rollback"\n'
        "  }\n"
        "}"
    )
    try:
        vision_paths: Optional[List[Tuple[str, Path]]] = None
        if advisor_mode(pipeline) == "multimodal":
            try:
                pipeline.cmd_with_check("load", "stage8_enhanced")
            except (
                CommandError,
                DataError,
                SirilError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                AttributeError,
            ) as error:
                pipeline.log.warn(
                    f"[AI] stage8 candidate preview load failed; using text advisor: {error}"
                )
                vision_paths = []
        raw = pipeline._request_stage_ai_advisory(
            "stage8_quality", schema, observations, image_paths=vision_paths
        )
        return pipeline._normalize_stage8_ai_quality(raw)
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        pipeline.log.warn(f"[AI] stage8 quality advisory unavailable: {e}")
        return None


def _normalize_adjustment_values(
    pipeline: object,
    adjustments: Dict[str, Any],
) -> Dict[str, float]:
    def pick(name: str, default: float) -> float:
        value = adjustments.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    return {
        "background_protection": _clamp_float(
            pick("background_protection", 0.85), 0.60, 0.98
        ),
        "global_contrast_delta": _clamp_float(
            pick("global_contrast_delta", 0.04), -0.10, 0.12
        ),
        "global_saturation_delta": _clamp_float(
            pick("global_saturation_delta", 0.03), -0.10, 0.12
        ),
        "red_balance_delta": _clamp_float(
            pick("red_balance_delta", 0.0), -0.08, 0.08
        ),
        "blue_balance_delta": _clamp_float(
            pick("blue_balance_delta", 0.0), -0.08, 0.08
        ),
        "denoise_strength": _clamp_float(
            pick("denoise_strength", 0.06), 0.0, 0.20
        ),
        "detail_boost": _clamp_float(
            pick("detail_boost", 0.03), 0.0, 0.12
        ),
        "blend_strength": _clamp_float(
            pick("blend_strength", pipeline.cfg.ai_strength), 0.05, 0.20
        ),
    }


def stage11_adjustment_candidate_presets(
    pipeline: object,
) -> Dict[str, Dict[str, float]]:
    configured_blend = _clamp_float(
        pipeline.cfg.ai_strength, 0.05, 0.20
    )
    return {
        "preserve": {
            "background_protection": 0.95,
            "global_contrast_delta": 0.0,
            "global_saturation_delta": 0.0,
            "red_balance_delta": 0.0,
            "blue_balance_delta": 0.0,
            "denoise_strength": 0.0,
            "detail_boost": 0.0,
            "blend_strength": 0.05,
        },
        "conservative": {
            "background_protection": 0.92,
            "global_contrast_delta": 0.025,
            "global_saturation_delta": 0.015,
            "red_balance_delta": 0.0,
            "blue_balance_delta": 0.0,
            "denoise_strength": 0.04,
            "detail_boost": 0.015,
            "blend_strength": min(configured_blend, 0.10),
        },
        "balanced": {
            "background_protection": 0.88,
            "global_contrast_delta": 0.04,
            "global_saturation_delta": 0.03,
            "red_balance_delta": 0.0,
            "blue_balance_delta": 0.0,
            "denoise_strength": 0.06,
            "detail_boost": 0.03,
            "blend_strength": configured_blend,
        },
        "detail_safe": {
            "background_protection": 0.92,
            "global_contrast_delta": 0.03,
            "global_saturation_delta": 0.015,
            "red_balance_delta": 0.0,
            "blue_balance_delta": 0.0,
            "denoise_strength": 0.035,
            "detail_boost": 0.04,
            "blend_strength": min(configured_blend, 0.10),
        },
    }


def normalize_ai_adjustments(
    pipeline: object,
    obj: Dict[str, Any],
) -> Dict[str, float]:
    """Map an AI-selected preset ID to immutable, code-owned adjustments."""
    selection = _selection_payload(
        obj,
        "stage11_adjustment_selection",
    )
    candidate_id = str(
        selection.get("selected_candidate_id", "")
    ).strip().lower()
    candidate = stage11_adjustment_candidate_presets(pipeline).get(
        candidate_id
    )
    if candidate is None:
        raise ValueError("Stage11 AI response has unknown candidate id")
    pipeline._ai_selected_candidate_id = candidate_id
    return _normalize_adjustment_values(pipeline, candidate)


def stage11_feature_based_fallback_adjustments(
    pipeline: object,
    source_features: object,
) -> Tuple[Dict[str, float], str]:
    bg = float(source_features.bg_median)
    red = float(source_features.red_dominance)
    blue = float(source_features.blue_dominance)
    star_density = float(source_features.star_density)
    core = float(source_features.core_brightness_ratio)

    adjustments = {
        "background_protection": 0.88,
        "global_contrast_delta": 0.04,
        "global_saturation_delta": 0.03,
        "red_balance_delta": 0.0,
        "blue_balance_delta": 0.0,
        "denoise_strength": 0.06,
        "detail_boost": 0.03,
        "blend_strength": pipeline.cfg.ai_strength,
    }
    notes: List[str] = []

    if bg > 0.16:
        adjustments.update(
            {
                "background_protection": 0.94,
                "global_contrast_delta": 0.025,
                "global_saturation_delta": 0.02,
                "denoise_strength": 0.045,
                "detail_boost": 0.02,
                "blend_strength": min(float(pipeline.cfg.ai_strength), 0.10),
            }
        )
        notes.append("bright background protected")
    elif bg < 0.035:
        adjustments.update(
            {
                "background_protection": 0.82,
                "global_contrast_delta": 0.055,
                "global_saturation_delta": 0.035,
                "denoise_strength": 0.055,
                "detail_boost": 0.035,
                "blend_strength": min(max(float(pipeline.cfg.ai_strength), 0.12), 0.16),
            }
        )
        notes.append("dark background lifted conservatively")

    if core > 0.08:
        adjustments["global_contrast_delta"] = min(
            adjustments["global_contrast_delta"], 0.035
        )
        adjustments["detail_boost"] = min(adjustments["detail_boost"], 0.025)
        notes.append("bright core protected")

    if star_density > 0.003:
        adjustments["detail_boost"] = min(adjustments["detail_boost"], 0.02)
        adjustments["denoise_strength"] = min(adjustments["denoise_strength"], 0.05)
        notes.append("dense star field protected")

    if blue > red + 0.08:
        adjustments["blue_balance_delta"] = -0.025
        adjustments["global_saturation_delta"] = min(
            adjustments["global_saturation_delta"], 0.02
        )
        notes.append("blue dominance reduced")
    elif red > blue + 0.12:
        adjustments["red_balance_delta"] = -0.015
        notes.append("red dominance restrained")
    elif red > blue + 0.03:
        adjustments["red_balance_delta"] = 0.008
        notes.append("warm nebula signal preserved")

    normalized = _normalize_adjustment_values(pipeline, adjustments)
    pipeline._ai_selected_candidate_id = "feature_based_fallback"
    reason = "; ".join(notes) if notes else "balanced conservative fallback"
    return normalized, f"fallback: feature-based conservative adjustments ({reason})"


def request_ai_adjustments(
    pipeline: object,
    source_features: object,
) -> Tuple[Dict[str, float], str]:
    pipeline._ai_plan_parse_fallback = False
    pipeline._ai_plan_parse_fallback_reason = None

    endpoint = pipeline.cfg.ai_endpoint.strip()
    model = pipeline.cfg.ai_model.strip()
    api_key = pipeline.cfg.ai_api_key.strip()
    prompt = (pipeline.cfg.ai_prompt or DEFAULT_AI_PROMPT).strip()
    timeout_sec = int(pipeline.cfg.ai_timeout_sec)
    endpoint_candidates = pipeline._build_ai_chat_endpoint_candidates(endpoint)
    if not endpoint_candidates:
        raise RuntimeError("SEESTAR_AI_ENDPOINT is empty")

    system_prompt = (
        "You are an expert astronomical image post-processing advisor. "
        "Return strict JSON only. Select exactly one supplied candidate id. "
        "Never return or modify numeric parameters. Keep edits conservative and "
        "scientifically realistic. Do not hallucinate objects."
    )
    candidates = stage11_adjustment_candidate_presets(pipeline)
    user_prompt = (
        "Select one code-owned local postprocess candidate for this deep-sky image.\n"
        f"Model prompt context: {prompt}\n"
        f"Image features: {format_feature_summary(source_features)}\n"
        "Allowed candidates (read-only):\n"
        f"{json.dumps(candidates, ensure_ascii=False, sort_keys=True)}\n"
        "Output JSON schema:\n"
        "{\n"
        '  "stage11_adjustment_selection": {\n'
        f'    "selected_candidate_id": "one of: {"|".join(candidates)}",\n'
        '    "confidence": 0.0,\n'
        '    "rationale": "short reason"\n'
        "  }\n"
        "}\n"
        "JSON only, no markdown."
    )
    text_payload_base = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 600,
    }
    payload_variants: List[Tuple[str, Dict[str, Any]]] = []
    if advisor_mode(pipeline) == "multimodal":
        current_url = _current_image_data_url(pipeline)
        if current_url:
            vision_payload = copy.deepcopy(text_payload_base)
            vision_payload["messages"][1]["content"] = _multimodal_user_content(
                user_prompt,
                [("Current final-stage image", current_url)],
            )
            payload_variants.append(("multimodal", vision_payload))
    payload_variants.append(("text", text_payload_base))

    temperatures = (1.0,) if "kimi" in model.lower() else (0.1, 1.0)
    last_error: Optional[Exception] = None
    attempt_errors: List[str] = []
    successful_response_count = 0
    for endpoint_url in endpoint_candidates:
        for request_mode, payload_base in payload_variants:
            if request_mode == "text" and payload_variants[0][0] == "multimodal":
                pipeline.log.warn("[AI] Stage11 falling back to text-only advisor")
            for temperature in temperatures:
                for json_mode in (True, False):
                    payload = copy.deepcopy(payload_base)
                    payload["temperature"] = temperature
                    if json_mode:
                        payload["response_format"] = {"type": "json_object"}
                    try:
                        response_obj = pipeline._post_json_with_auth(
                            endpoint_url, payload, api_key, timeout_sec
                        )
                        successful_response_count += 1
                        content = pipeline._extract_chat_content(response_obj)
                        pipeline._write_ai_raw_response(
                            "stage11_adjustment_plan",
                            endpoint_url=endpoint_url,
                            temperature=temperature,
                            json_mode=json_mode,
                            response_obj=response_obj,
                            content=content,
                        )
                        plan_obj = pipeline._extract_first_json_object(content)
                        adjustments = pipeline._normalize_ai_adjustments(plan_obj)
                        selection = _selection_payload(
                            plan_obj,
                            "stage11_adjustment_selection",
                        )
                        summary = _selection_rationale(pipeline, selection)
                        selected_id = str(
                            selection.get("selected_candidate_id", "")
                        ).strip()
                        if selected_id:
                            summary = (
                                f"candidate={selected_id}"
                                + (f"; {summary}" if summary else "")
                            )
                        pipeline._ai_plan_parse_fallback = False
                        pipeline._ai_plan_parse_fallback_reason = None
                        if temperature != 0.1:
                            pipeline.log.warn(
                                f"[AI] Plan request used temperature fallback={temperature}"
                            )
                        return adjustments, summary
                    except (OSError, RuntimeError, TypeError, ValueError) as e:
                        if "response_obj" in locals() or "content" in locals():
                            try:
                                pipeline._write_ai_raw_response(
                                    "stage11_adjustment_plan",
                                    endpoint_url=endpoint_url,
                                    temperature=temperature,
                                    json_mode=json_mode,
                                    response_obj=locals().get("response_obj"),
                                    content=locals().get("content"),
                                    error_text=str(e),
                                )
                            except (OSError, RuntimeError, TypeError, ValueError):
                                pass
                        last_error = e
                        attempt_errors.append(
                            "endpoint="
                            f"{endpoint_url},mode={request_mode},temperature={temperature},"
                            f"json_mode={json_mode},error={e}"
                        )
                        err_text = str(e).lower()
                        if json_mode and (
                            "response_format" in err_text
                            or "json_object" in err_text
                            or "unsupported" in err_text
                        ):
                            continue
                        if temperature == 0.1 and "only 1 is allowed for this model" in err_text:
                            pipeline.log.warn(
                                "[AI] Model requires temperature=1, retrying with fallback"
                            )
                            break
                        if temperature == 1.0 and "only 1 is allowed for this model" in err_text:
                            break
                        pipeline.log.warn(
                            "[AI] Plan request failed "
                            f"(endpoint={endpoint_url}, mode={request_mode}, "
                            f"temperature={temperature}, json_mode={json_mode}): {e}"
                        )
                    finally:
                        if "response_obj" in locals():
                            del response_obj
                        if "content" in locals():
                            del content

    if successful_response_count > 0:
        pipeline.log.warn(
            "[AI] Model response received but JSON plan parsing failed; "
            "fallback to feature-based conservative adjustments"
        )
        pipeline._ai_plan_parse_fallback = True
        pipeline._ai_plan_parse_fallback_reason = (
            "ai plan json parse failed; used feature-based conservative fallback"
        )
        return pipeline._stage11_feature_based_fallback_adjustments(source_features)

    if attempt_errors:
        detail = " | ".join(attempt_errors[-4:])
        raise RuntimeError(
            f"AI suggestion request failed: {detail}"
        )
    raise RuntimeError(f"AI suggestion request failed: {last_error}")
