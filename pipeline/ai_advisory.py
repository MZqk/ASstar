from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from image_metrics import format_feature_summary
from models import TargetType


DEFAULT_AI_PROMPT = (
    "Conservative deep-sky astrophotography enhancement only. "
    "Preserve astronomical realism, faint structures, and natural star colors. "
    "Do not invent objects, do not oversaturate, do not clip black background, "
    "do not increase star size, and avoid halos or artificial sharpening artifacts."
)
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _clamp_float(value: object, min_value: float, max_value: float) -> float:
    try:
        fvalue = float(value)  # type: ignore[arg-type]
    except Exception:
        fvalue = min_value
    return max(min_value, min(max_value, fvalue))


def _clamp_int(value: object, min_value: int, max_value: int) -> int:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except Exception:
        ivalue = min_value
    return max(min_value, min(max_value, ivalue))


def post_json_with_auth(
    endpoint: str,
    payload: Dict[str, Any],
    api_key: str,
    timeout_sec: int,
) -> Dict[str, Any]:
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
        except Exception:
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
    keys = (
        "background_protection",
        "global_contrast_delta",
        "global_saturation_delta",
        "red_balance_delta",
        "blue_balance_delta",
        "denoise_strength",
        "detail_boost",
    )

    adjustments: Dict[str, float] = {}
    for key in keys:
        pattern = (
            rf"[\"']?{re.escape(key)}[\"']?"
            r"\s*[:=：]\s*(-?\d+(?:\.\d+)?)"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            adjustments[key] = float(match.group(1))
        except Exception:
            continue

    if not adjustments:
        return None

    summary = "fallback: parsed non-json ai plan text"
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line:
        summary = pipeline._short_text(first_line, max_len=96)
    return {
        "summary": summary,
        "adjustments": adjustments,
    }


def extract_first_json_object(pipeline: object, text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise RuntimeError("AI plan content is empty")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
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
                    except Exception:
                        pass
                    break

    parsed_from_text = extract_adjustments_from_text(pipeline, text)
    if parsed_from_text is not None:
        pipeline.log.warn(
            "[AI] Plan is not strict JSON; extracted adjustments from plain text fallback"
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

    def pick_float_any(names: Tuple[str, ...]) -> Optional[float]:
        for name in names:
            patterns = [
                rf"{re.escape(name)}\s*[:=：]\s*(-?\d+(?:\.\d+)?)",
                rf"{re.escape(name.replace('_', ' '))}\s*[:=：]\s*(-?\d+(?:\.\d+)?)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except Exception:
                        return None
        return None

    if stage_name == "stage6_stretch_plan":
        method = ""
        if (
            "autostretch" in lowered
            or "auto stretch" in lowered
            or "automatic stretch" in lowered
            or "自动拉伸" in raw_text
        ):
            method = "autostretch"
        elif "ghs" in lowered or "autoghs" in lowered or "generalized hyperbolic" in lowered:
            method = "ghs"
        elif "asinh" in lowered or "arcsinh" in lowered or "反双曲" in raw_text:
            method = "asinh"
        if not method:
            return None

        def pick(name: str) -> Optional[float]:
            return pick_float_any((name,))

        params = {
            "asinh_stretch": pick("asinh_stretch") or pipeline.cfg.asinh_stretch,
            "asinh_offset": pick("asinh_offset") or pipeline.cfg.asinh_offset,
            "ghs_shadowsclip": pick("ghs_shadowsclip") or pipeline.cfg.ghs_shadowsclip,
            "ghs_stretchamount": pick("ghs_stretchamount") or pipeline.cfg.ghs_stretchamount,
        }
        return {
            "stage6_stretch_plan": {
                "summary": pipeline._short_text(first_useful_line(), 140),
                "method": method,
                "params": params,
            }
        }
    if stage_name == "stage7_starless_plan":
        def pick_int(name: str, default: int) -> int:
            match = re.search(
                rf"{re.escape(name)}\s*[:=：]\s*(\d+)",
                text,
                flags=re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    rf"{re.escape(name.replace('_', ' '))}\s*[:=：]\s*(\d+)",
                    text,
                    flags=re.IGNORECASE,
                )
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    pass
            return default

        tile_size = pick_int("tile_size", 512)
        overlap = pick_int("overlap", 64)
        pair_match = re.search(
            r"(?:tile|tile_size)?\s*(512|1024)\s*[/,x×]\s*(?:overlap)?\s*(64|96|128)",
            text,
            flags=re.IGNORECASE,
        )
        if pair_match:
            tile_size = int(pair_match.group(1))
            overlap = int(pair_match.group(2))
        elif "tile_size" not in lowered and "tile size" not in lowered and "overlap" not in lowered:
            if "larger overlap" in lowered or "increase overlap" in lowered or "more overlap" in lowered:
                tile_size, overlap = 512, 96
            elif "larger tile" in lowered or "increase tile" in lowered or "1024" in lowered:
                tile_size, overlap = 1024, 64
            elif "residual" in lowered or "残星" in raw_text:
                tile_size, overlap = 512, 96
            else:
                return None
        return {
            "stage7_starless_plan": {
                "summary": pipeline._short_text(first_useful_line(), 140),
                "tile_size": tile_size,
                "overlap": overlap,
                "use_axiom": "axiom" in lowered,
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
        ai_scale = pick_float_any(
            (
                "stage9_star_intensity_scale",
                "star_intensity_scale",
                "intensity_scale",
                "scale",
            )
        )
        suppression = pick_float_any(
            (
                "residual_suppression_strength",
                "suppression_strength",
                "residual suppression",
                "suppression",
            )
        )
        quality: Dict[str, Any] = {
            "verdict": verdict,
            "summary": pipeline._short_text(first_useful_line(), 160),
            "residual_stars": residual,
            "starmask_missing": missing,
            "starmask_too_wide": too_wide,
            "recommended_action": action,
        }
        if ai_scale is not None:
            quality["stage9_star_intensity_scale"] = ai_scale
        if suppression is not None:
            quality["residual_suppression_strength"] = suppression
        elif residual:
            quality["residual_suppression_strength"] = 0.12
        return {"stage7_quality": quality}
    if stage_name == "stage8_processing_plan":
        def pick_float(name: str, default: float) -> float:
            match = re.search(
                rf"{re.escape(name)}\s*[:=：]\s*(-?\d+(?:\.\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    rf"{re.escape(name.replace('_', ' '))}\s*[:=：]\s*(-?\d+(?:\.\d+)?)",
                    text,
                    flags=re.IGNORECASE,
                )
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    pass
            return default

        if not any(
            token in lowered
            for token in ("saturation", "unsharp", "bg_factor", "bg factor")
        ):
            return None
        return {
            "stage8_processing_plan": {
                "summary": pipeline._short_text(text.splitlines()[0] if text else "parsed from text", 140),
                "saturation": pick_float("saturation", pipeline.cfg.nebula_saturation),
                "bg_factor": int(round(pick_float("bg_factor", float(pipeline.cfg.nebula_bg_factor)))),
                "unsharp_radius": pick_float("unsharp_radius", 0.8),
                "unsharp_amount": pick_float("unsharp_amount", 0.35),
                "apply_after_plugins": not any(
                    token in lowered
                    for token in (
                        "keep plugin",
                        "no after plugin",
                        "apply_after_plugins=false",
                        "apply after plugins false",
                    )
                ),
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
    payload_base = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }

    temperatures = (1.0,) if "kimi" in model.lower() else (0.1, 1.0)
    attempt_errors: List[str] = []
    parse_failures: List[str] = []
    for endpoint_url in endpoint_candidates:
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
                    except Exception:
                        parse_failures.append(
                            f"endpoint={endpoint_url},temperature={temperature},json_mode={json_mode}"
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
                except Exception as e:
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
                        except Exception:
                            pass
                    attempt_errors.append(
                        "endpoint="
                        f"{endpoint_url},temperature={temperature},"
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
                        f"(endpoint={endpoint_url}, temperature={temperature}, "
                        f"json_mode={json_mode}): {e}"
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


def normalize_stage6_ai_plan(pipeline: object, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    plan = obj.get("stage6_stretch_plan") if isinstance(obj, dict) else None
    if not isinstance(plan, dict):
        plan = obj if isinstance(obj, dict) else {}
    method = str(plan.get("method", "")).strip().lower().replace("-", "_")
    aliases = {
        "auto": "autostretch",
        "auto_stretch": "autostretch",
        "auto stretch": "autostretch",
        "asinh_stretch": "asinh",
        "autoghs": "ghs",
        "asinh+ghs": "asinh_ghs",
        "asinh ghs": "asinh_ghs",
        "asinh_then_ghs": "asinh_ghs",
    }
    method = aliases.get(method, method)
    if method not in {"asinh", "asinh_ghs", "ghs", "autostretch"}:
        return None

    params_obj = plan.get("params")
    params = params_obj if isinstance(params_obj, dict) else plan
    return {
        "method": method,
        "summary": str(plan.get("summary", plan.get("reason", ""))).strip(),
        "params": {
            "asinh_stretch": _clamp_float(
                params.get("asinh_stretch", pipeline.cfg.asinh_stretch), 1.6, 3.6
            ),
            "asinh_offset": _clamp_float(
                params.get("asinh_offset", pipeline.cfg.asinh_offset), 0.0005, 0.006
            ),
            "ghs_shadowsclip": _clamp_float(
                params.get("ghs_shadowsclip", pipeline.cfg.ghs_shadowsclip), -3.6, -1.8
            ),
            "ghs_stretchamount": _clamp_float(
                params.get("ghs_stretchamount", pipeline.cfg.ghs_stretchamount), 1.0, 2.8
            ),
        },
    }


def request_stage6_stretch_plan(
    pipeline: object,
    baseline_features: Optional[object],
    baseline_quality: Optional[object],
) -> Optional[Dict[str, Any]]:
    observations = {
        "target_type": (
            getattr(getattr(pipeline, "auto_tune_result", None), "target_type", TargetType.UNKNOWN).name
        ),
        "features": asdict(baseline_features) if baseline_features else None,
        "quality_metrics": asdict(baseline_quality) if baseline_quality else None,
        "current_params": {
            "asinh_stretch": pipeline.cfg.asinh_stretch,
            "asinh_offset": pipeline.cfg.asinh_offset,
            "ghs_shadowsclip": pipeline.cfg.ghs_shadowsclip,
            "ghs_stretchamount": pipeline.cfg.ghs_stretchamount,
        },
        "constraints": {
            "bg_median_min": pipeline.cfg.stage6_bg_median_min,
            "black_pixel_ratio_max": pipeline.cfg.stage6_black_pixel_ratio_max,
            "highlight_clip_ratio_max": pipeline.cfg.stage6_highlight_clip_ratio_max,
            "star_growth_ratio_max": pipeline.cfg.stage6_star_growth_ratio_max,
        },
    }
    schema = (
        "{\n"
        '  "stage6_stretch_plan": {\n'
        '    "summary": "short reason",\n'
        '    "method": "asinh|asinh_ghs|ghs|autostretch",\n'
        '    "params": {\n'
        '      "asinh_stretch": 1.6,\n'
        '      "asinh_offset": 0.0005,\n'
        '      "ghs_shadowsclip": -3.6,\n'
        '      "ghs_stretchamount": 1.0\n'
        "    }\n"
        "  }\n"
        "}"
    )
    try:
        raw_plan = pipeline._request_stage_ai_advisory(
            "stage6_stretch_plan", schema, observations
        )
        plan = pipeline._normalize_stage6_ai_plan(raw_plan)
        if plan is None:
            raise RuntimeError("stage6 AI plan has invalid method")
        return plan
    except Exception as e:
        pipeline.log.warn(f"[AI] stage6 stretch advisory unavailable: {e}")
        return None


def normalize_stage7_starless_plan(pipeline: object, obj: Dict[str, Any]) -> Dict[str, Any]:
    plan = obj.get("stage7_starless_plan") if isinstance(obj, dict) else None
    if not isinstance(plan, dict):
        plan = obj if isinstance(obj, dict) else {}

    tile_size = _clamp_int(plan.get("tile_size", 512), 512, 1024)
    overlap = _clamp_int(plan.get("overlap", 64), 64, 128)
    if overlap >= tile_size:
        overlap = max(64, min(128, tile_size // 4))

    use_axiom = bool(plan.get("use_axiom", False))
    if use_axiom and not pipeline._syqon_axiom_model_available():
        use_axiom = False

    return {
        "summary": pipeline._short_text(str(plan.get("summary", "")), 180),
        "tile_size": tile_size,
        "overlap": overlap,
        "use_axiom": use_axiom,
    }


def request_stage7_starless_plan(pipeline: object) -> Optional[Dict[str, Any]]:
    if not pipeline._ai_stage_advisory_enabled("ai_stage7_enabled"):
        return None
    features = pipeline._measure_current_features()
    quality = pipeline._measure_current_quality()
    observations = {
        "features": asdict(features) if features else None,
        "quality_metrics": asdict(quality) if quality else None,
        "current_syqon_defaults": {
            "tile_size": 512,
            "overlap": 64,
            "use_axiom": False,
            "axiom_available": pipeline._syqon_axiom_model_available(),
        },
        "constraints": {
            "tile_size": "512..1024; do not use 256 as an optimization path",
            "overlap": "64..128 and lower than tile_size; do not use 32 as an optimization path",
            "use_axiom": "only true when axiom_available is true",
        },
    }
    schema = (
        "{\n"
        '  "stage7_starless_plan": {\n'
        '    "summary": "short reason",\n'
        '    "tile_size": 512,\n'
        '    "overlap": 64,\n'
        '    "use_axiom": false\n'
        "  }\n"
        "}"
    )
    try:
        raw = pipeline._request_stage_ai_advisory(
            "stage7_starless_plan", schema, observations
        )
        return pipeline._normalize_stage7_starless_plan(raw)
    except Exception as e:
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

    def opt_float(name: str, lo: float, hi: float) -> Optional[float]:
        if name not in quality:
            return None
        try:
            return _clamp_float(quality.get(name), lo, hi)
        except Exception:
            return None

    return {
        "verdict": verdict,
        "residual_stars": as_bool("residual_stars"),
        "starmask_missing": as_bool("starmask_missing"),
        "starmask_too_wide": as_bool("starmask_too_wide"),
        "recommended_action": action,
        "stage9_star_intensity_scale": opt_float(
            "stage9_star_intensity_scale", 0.35, 1.0
        ),
        "residual_suppression_strength": opt_float(
            "residual_suppression_strength", 0.0, 0.25
        ),
        "summary": pipeline._short_text(str(quality.get("summary", "")), 180),
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
        '    "stage9_star_intensity_scale": 0.75,\n'
        '    "residual_suppression_strength": 0.08,\n'
        '    "recommended_action": "accept|retry_syqon|fallback_sasp|degrade"\n'
        "  }\n"
        "}"
    )
    try:
        raw = pipeline._request_stage_ai_advisory(
            "stage7_quality", schema, observations
        )
        return pipeline._normalize_stage7_ai_quality(raw)
    except Exception as e:
        pipeline.log.warn(f"[AI] stage7 quality advisory unavailable: {e}")
        return None


def normalize_stage8_processing_plan(pipeline: object, obj: Dict[str, Any]) -> Dict[str, Any]:
    plan = obj.get("stage8_processing_plan") if isinstance(obj, dict) else None
    if not isinstance(plan, dict):
        plan = obj if isinstance(obj, dict) else {}

    return {
        "summary": pipeline._short_text(str(plan.get("summary", "")), 180),
        "saturation": _clamp_float(
            plan.get("saturation", pipeline.cfg.nebula_saturation), 0.0, 0.65
        ),
        "bg_factor": _clamp_int(
            plan.get("bg_factor", pipeline.cfg.nebula_bg_factor), 0, 3
        ),
        "unsharp_radius": _clamp_float(
            plan.get("unsharp_radius", 0.8), 0.0, 1.2
        ),
        "unsharp_amount": _clamp_float(
            plan.get("unsharp_amount", 0.35), 0.0, 0.60
        ),
        "apply_after_plugins": bool(plan.get("apply_after_plugins", True)),
    }


def request_stage8_processing_plan(pipeline: object) -> Optional[Dict[str, Any]]:
    if not pipeline._ai_stage_advisory_enabled("ai_stage8_enabled"):
        return None
    features = pipeline._measure_current_features()
    quality = pipeline._measure_current_quality()
    observations = {
        "features": asdict(features) if features else None,
        "quality_metrics": asdict(quality) if quality else None,
        "current_params": {
            "saturation": pipeline.cfg.nebula_saturation,
            "bg_factor": pipeline.cfg.nebula_bg_factor,
            "unsharp_radius": 0.8,
            "unsharp_amount": 0.35,
        },
        "constraints": {
            "saturation": "0.0..0.65",
            "bg_factor": "0..3 integer",
            "unsharp_radius": "0.0..1.2, 0 disables unsharp",
            "unsharp_amount": "0.0..0.60, 0 disables unsharp",
            "apply_after_plugins": "true by default; use false only when plugin output should be kept untouched",
        },
    }
    schema = (
        "{\n"
        '  "stage8_processing_plan": {\n'
        '    "summary": "short reason",\n'
        '    "saturation": 0.20,\n'
        '    "bg_factor": 1,\n'
        '    "unsharp_radius": 0.8,\n'
        '    "unsharp_amount": 0.35,\n'
        '    "apply_after_plugins": true\n'
        "  }\n"
        "}"
    )
    try:
        raw = pipeline._request_stage_ai_advisory(
            "stage8_processing_plan", schema, observations
        )
        return pipeline._normalize_stage8_processing_plan(raw)
    except Exception as e:
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
    target_blue_excess = quality.get("target_blue_excess", None)
    if target_blue_excess is not None:
        try:
            target_blue_excess = _clamp_float(target_blue_excess, 0.05, 0.16)
        except Exception:
            target_blue_excess = None
    return {
        "verdict": verdict,
        "oversaturated": as_bool("oversaturated"),
        "blue_bias": as_bool("blue_bias"),
        "microcontrast_overdone": as_bool("microcontrast_overdone"),
        "recommended_action": action,
        "target_blue_excess": target_blue_excess,
        "summary": pipeline._short_text(str(quality.get("summary", "")), 180),
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
        '    "target_blue_excess": 0.10,\n'
        '    "recommended_action": "accept|blue_guard|conservative_rerun|rollback"\n'
        "  }\n"
        "}"
    )
    try:
        raw = pipeline._request_stage_ai_advisory(
            "stage8_quality", schema, observations
        )
        return pipeline._normalize_stage8_ai_quality(raw)
    except Exception as e:
        pipeline.log.warn(f"[AI] stage8 quality advisory unavailable: {e}")
        return None


def normalize_ai_adjustments(pipeline: object, obj: Dict[str, Any]) -> Dict[str, float]:
    adjustments = obj.get("adjustments")
    if not isinstance(adjustments, dict):
        adjustments = {}

    def pick(name: str, default: float) -> float:
        value = adjustments.get(name, default)
        try:
            return float(value)
        except Exception:
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

    normalized = pipeline._normalize_ai_adjustments({"adjustments": adjustments})
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
        "Return strict JSON only. Keep edits conservative and scientifically realistic. "
        "Do not hallucinate objects. Keep star colors natural and avoid over-saturation."
    )
    user_prompt = (
        "Generate local postprocess suggestions for deep-sky image.\n"
        f"Model prompt context: {prompt}\n"
        f"Image features: {format_feature_summary(source_features)}\n"
        "Output JSON schema:\n"
        "{\n"
        '  "summary": "short rationale",\n'
        '  "adjustments": {\n'
        '    "background_protection": 0.60..0.98,\n'
        '    "global_contrast_delta": -0.10..0.12,\n'
        '    "global_saturation_delta": -0.10..0.12,\n'
        '    "red_balance_delta": -0.08..0.08,\n'
        '    "blue_balance_delta": -0.08..0.08,\n'
        '    "denoise_strength": 0.00..0.20,\n'
        '    "detail_boost": 0.00..0.12,\n'
        '    "blend_strength": 0.05..0.20\n'
        "  }\n"
        "}\n"
        "JSON only, no markdown."
    )
    payload_base = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 600,
    }

    temperatures = (1.0,) if "kimi" in model.lower() else (0.1, 1.0)
    last_error: Optional[Exception] = None
    attempt_errors: List[str] = []
    successful_response_count = 0
    for endpoint_url in endpoint_candidates:
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
                    summary = str(plan_obj.get("summary", "")).strip()
                    pipeline._ai_plan_parse_fallback = False
                    pipeline._ai_plan_parse_fallback_reason = None
                    if temperature != 0.1:
                        pipeline.log.warn(
                            f"[AI] Plan request used temperature fallback={temperature}"
                        )
                    return adjustments, summary
                except Exception as e:
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
                        except Exception:
                            pass
                    last_error = e
                    attempt_errors.append(
                        "endpoint="
                        f"{endpoint_url},temperature={temperature},"
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
                        f"(endpoint={endpoint_url}, temperature={temperature}, "
                        f"json_mode={json_mode}): {e}"
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
