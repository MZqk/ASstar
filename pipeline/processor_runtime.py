"""Service mixins for SeestarPostProcessor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ai_advisory
import cosmic_clarity
import plugin_runner
import sasp_runner
import scunet_denoise
import syqon_starless
import stage7_quality
import stage7_repair
import stage8_pixels
from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    format_feature_summary,
    measure_quality_metrics,
)
from models import ImageFeatures, QualityMetrics, Stage6StretchStrategy, StageResult, TargetType
from save_utils import save_stage_output, write_ai_raw_response, write_stage_json

try:
    from sirilpy.exceptions import CommandError, DataError, SirilError
except Exception:
    CommandError = Exception
    DataError = Exception
    SirilError = Exception

try:
    from image_feature_analyzer import analyze_image as analyze_adaptive_image
    from policy_selector import DEFAULT_POLICY, policy_for_profile
    from stretch_candidate_evaluator import (
        build_candidate_spec,
        candidate_modes,
        choose_best as choose_best_stretch_candidate,
        score_candidate as score_stretch_candidate,
    )
    from target_profiler import build_target_profile
except Exception:
    analyze_adaptive_image = None
    DEFAULT_POLICY = {
        "policy_name": "generic_low_snr_safe",
        "stage6_stretch": {"fallback_candidate": "asinh_core_protect"},
    }
    policy_for_profile = None
    build_candidate_spec = None
    candidate_modes = None
    choose_best_stretch_candidate = None
    score_stretch_candidate = None
    build_target_profile = None

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ENV_DEBUG_MODE_KEY = "SEESTAR_DEBUG_MODE"
ENV_INPUT_MODE_KEY = "SEESTAR_INPUT_MODE"
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "result_linear_resume"
RESULT_BASENAME_TEMPLATE = (
    "$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec"
    "_$DATE-OBS:dm12$_processed"
)

def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))

class ProcessorRuntimeMixin:
    def _result_output_basename(self) -> str:
        base_filename = RESULT_BASENAME_TEMPLATE
        if self._stage1_input_mode == "linear_resume":
            base_filename += "_linear"
        self.main_output_basename_template = base_filename
        return base_filename


    def _parse_env_bool(self, raw_value: str, fallback: bool) -> bool:
        value = raw_value.strip().lower()
        if value in ENV_TRUE_VALUES:
            return True
        if value in ENV_FALSE_VALUES:
            return False
        return fallback


    def _sync_logger_level(self):
        self.log.min_level = self.log._LEVELS.get(
            "DEBUG" if self.cfg.debug_mode else "INFO",
            1,
        )


    def _save_stage_output(self, stem: str) -> bool:
        return save_stage_output(self.cmd_with_check, self.log, stem)


    def _sha256_file(self, path: Path) -> Optional[str]:
        if not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None


    def _stage_diff_note(self, current_stem: str, previous_stem: str) -> Optional[str]:
        if not self.process_dir:
            return None
        current_path = self.process_dir / f"{current_stem}.fit"
        previous_path = self.process_dir / f"{previous_stem}.fit"
        if not current_path.exists() or not previous_path.exists():
            return None

        current_hash = self._sha256_file(current_path)
        previous_hash = self._sha256_file(previous_path)
        if not current_hash or not previous_hash:
            return None

        if current_hash == previous_hash:
            return (
                f"阶段对比: {current_stem}.fit 与 {previous_stem}.fit 内容一致 "
                f"(sha256={current_hash[:12]})"
            )
        return (
            f"阶段对比: {current_stem}.fit 与 {previous_stem}.fit 内容有变化 "
            f"({previous_hash[:8]} -> {current_hash[:8]})"
        )


    def _feature_summary_note(self, label: str) -> Optional[str]:
        feat = self._measure_current_features()
        if feat is None:
            return None
        return f"{label}: {format_feature_summary(feat)}"


    def _apply_runtime_env_overrides(self):
        input_mode_raw = os.getenv(ENV_INPUT_MODE_KEY)
        if input_mode_raw is not None:
            normalized = input_mode_raw.strip().lower()
            if normalized in {INPUT_MODE_AUTO, INPUT_MODE_LINEAR_RESUME}:
                self.input_mode = normalized
            else:
                self.log.warn(
                    f"{ENV_INPUT_MODE_KEY} has invalid value; keeping current setting"
                )

        debug_raw = os.getenv(ENV_DEBUG_MODE_KEY)
        if debug_raw is not None:
            parsed = self._parse_env_bool(debug_raw, self.cfg.debug_mode)
            self.cfg.debug_mode = parsed
            if debug_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    f"{ENV_DEBUG_MODE_KEY} has invalid value; keeping current setting"
                )
            self._sync_logger_level()

        plugin_probe_raw = os.getenv("SEESTAR_WORKFLOW_PLUGIN_PROBE")
        if plugin_probe_raw is not None:
            parsed = self._parse_env_bool(
                plugin_probe_raw,
                self.cfg.workflow_plugin_probe_enabled,
            )
            self.cfg.workflow_plugin_probe_enabled = parsed
            if plugin_probe_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_WORKFLOW_PLUGIN_PROBE has invalid value; "
                    "keeping current setting"
                )

        spcc_enable_raw = os.getenv("SEESTAR_SPCC_ENABLE")
        if spcc_enable_raw is not None:
            parsed = self._parse_env_bool(
                spcc_enable_raw,
                self.cfg.spcc_enabled,
            )
            self.cfg.spcc_enabled = parsed
            if spcc_enable_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_SPCC_ENABLE has invalid value; keeping current setting"
                )

        aberration_api_raw = os.getenv("SEESTAR_ABERRATION_API_ENABLE")
        if aberration_api_raw is not None:
            parsed = self._parse_env_bool(
                aberration_api_raw,
                getattr(self.cfg, "aberration_api_enabled", False),
            )
            self.cfg.aberration_api_enabled = parsed
            if aberration_api_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_ABERRATION_API_ENABLE has invalid value; keeping current setting"
                )

        denoise_enable_raw = os.getenv("SEESTAR_DENOISE_ENABLE")
        if denoise_enable_raw is not None:
            parsed = self._parse_env_bool(
                denoise_enable_raw,
                self.cfg.denoise_enabled,
            )
            self.cfg.denoise_enabled = parsed
            if denoise_enable_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_DENOISE_ENABLE has invalid value; keeping current setting"
                )

        denoise_force_raw = os.getenv("SEESTAR_DENOISE_FORCE")
        if denoise_force_raw is not None:
            parsed = self._parse_env_bool(
                denoise_force_raw,
                self.cfg.denoise_enabled,
            )
            self._force_denoise_enabled = parsed
            if denoise_force_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_DENOISE_FORCE has invalid value; keeping current setting"
                )

        enabled_raw = os.getenv("SEESTAR_AI_ENABLED")
        if enabled_raw is not None:
            parsed = self._parse_env_bool(enabled_raw, self.cfg.ai_post_enabled)
            self.cfg.ai_post_enabled = parsed
            if enabled_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_AI_ENABLED has invalid value; keeping current setting"
                )

        for env_key, attr_name in (
            ("SEESTAR_AI_STAGE6_ENABLE", "ai_stage6_enabled"),
            ("SEESTAR_AI_STAGE7_ENABLE", "ai_stage7_enabled"),
            ("SEESTAR_AI_STAGE8_ENABLE", "ai_stage8_enabled"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            parsed = self._parse_env_bool(raw_value, getattr(self.cfg, attr_name))
            setattr(self.cfg, attr_name, parsed)
            if raw_value.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(f"{env_key} has invalid value; keeping current setting")

        endpoint = os.getenv("SEESTAR_AI_ENDPOINT")
        if endpoint is not None:
            self.cfg.ai_endpoint = endpoint.strip()

        model = os.getenv("SEESTAR_AI_MODEL")
        if model is not None:
            self.cfg.ai_model = model.strip()

        api_key = os.getenv("SEESTAR_AI_API_KEY")
        if api_key is not None:
            self.cfg.ai_api_key = api_key.strip()

        prompt = os.getenv("SEESTAR_AI_PROMPT")
        if prompt is not None:
            self.cfg.ai_prompt = prompt.strip()

        timeout_raw = os.getenv("SEESTAR_AI_TIMEOUT_SEC")
        if timeout_raw is not None:
            try:
                self.cfg.ai_timeout_sec = int(timeout_raw.strip())
            except ValueError:
                self.log.warn(
                    f"Invalid SEESTAR_AI_TIMEOUT_SEC={timeout_raw!r}; using current value"
                )

        strength_raw = os.getenv("SEESTAR_AI_STRENGTH")
        if strength_raw is not None:
            try:
                self.cfg.ai_strength = float(strength_raw.strip())
            except ValueError:
                self.log.warn(
                    f"Invalid SEESTAR_AI_STRENGTH={strength_raw!r}; using current value"
                )

        stage7_retry_raw = os.getenv("SEESTAR_STAGE7_QUALITY_RETRY_MAX")
        if stage7_retry_raw is not None:
            try:
                self.cfg.stage7_quality_retry_max = int(stage7_retry_raw.strip())
            except ValueError:
                self.log.warn(
                    "Invalid SEESTAR_STAGE7_QUALITY_RETRY_MAX="
                    f"{stage7_retry_raw!r}; using current value"
                )

        stage7_skip_raw = os.getenv("SEESTAR_STAGE7_SKIP_UNREADY_STARLESS")
        if stage7_skip_raw is not None:
            parsed = self._parse_env_bool(
                stage7_skip_raw,
                self.cfg.stage7_skip_unready_starless,
            )
            self.cfg.stage7_skip_unready_starless = parsed
            if stage7_skip_raw.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(
                    "SEESTAR_STAGE7_SKIP_UNREADY_STARLESS has invalid value; keeping current setting"
                )

        for env_key, attr_name in (
            ("SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH", "stage7_soft_starless_asinh_stretch"),
            ("SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX", "stage7_bright_nebula_halo_residue_score_max"),
            ("SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH", "stage7_starless_repair_strength"),
            ("SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH", "stage7_starless_halo_repair_strength"),
            ("SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH", "stage7_starless_chroma_denoise_strength"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            try:
                setattr(self.cfg, attr_name, float(raw_value.strip()))
            except ValueError:
                self.log.warn(f"Invalid {env_key}={raw_value!r}; using current value")

        for env_key, attr_name in (
            ("SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE", "stage7_starless_pixel_repair_enabled"),
            ("SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR", "stage8_force_conservative_after_stage7_repair"),
        ):
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue
            parsed = self._parse_env_bool(raw_value, getattr(self.cfg, attr_name))
            setattr(self.cfg, attr_name, parsed)
            if raw_value.strip().lower() not in (ENV_TRUE_VALUES | ENV_FALSE_VALUES):
                self.log.warn(f"{env_key} has invalid value; keeping current setting")

        old_timeout = self.cfg.ai_timeout_sec
        old_strength = self.cfg.ai_strength
        old_stage7_retry = self.cfg.stage7_quality_retry_max
        old_stage7_soft = self.cfg.stage7_soft_starless_asinh_stretch
        old_stage7_bright_halo = self.cfg.stage7_bright_nebula_halo_residue_score_max
        old_stage7_repair = self.cfg.stage7_starless_repair_strength
        old_stage7_halo = self.cfg.stage7_starless_halo_repair_strength
        old_stage7_chroma = self.cfg.stage7_starless_chroma_denoise_strength
        self.cfg.ai_timeout_sec = _clamp_int(self.cfg.ai_timeout_sec, 15, 300)
        self.cfg.ai_strength = _clamp_float(self.cfg.ai_strength, 0.05, 0.25)
        self.cfg.stage7_quality_retry_max = _clamp_int(
            self.cfg.stage7_quality_retry_max, 0, 3
        )
        self.cfg.stage7_soft_starless_asinh_stretch = _clamp_float(
            self.cfg.stage7_soft_starless_asinh_stretch,
            1.05,
            self.cfg.stage7_ultra_conservative_asinh_stretch,
        )
        self.cfg.stage7_bright_nebula_halo_residue_score_max = _clamp_float(
            self.cfg.stage7_bright_nebula_halo_residue_score_max,
            self.cfg.stage7_halo_residue_score_max,
            1.20,
        )
        self.cfg.stage7_starless_repair_strength = _clamp_float(
            self.cfg.stage7_starless_repair_strength,
            0.0,
            0.85,
        )
        self.cfg.stage7_starless_halo_repair_strength = _clamp_float(
            self.cfg.stage7_starless_halo_repair_strength,
            0.0,
            0.90,
        )
        self.cfg.stage7_starless_chroma_denoise_strength = _clamp_float(
            self.cfg.stage7_starless_chroma_denoise_strength,
            0.0,
            0.90,
        )
        if old_timeout != self.cfg.ai_timeout_sec:
            self.log.warn(
                f"AI timeout clamped: {old_timeout} -> {self.cfg.ai_timeout_sec}"
            )
        if old_strength != self.cfg.ai_strength:
            self.log.warn(
                f"AI strength clamped: {old_strength} -> {self.cfg.ai_strength}"
            )
        if old_stage7_retry != self.cfg.stage7_quality_retry_max:
            self.log.warn(
                "Stage7 quality retry max clamped: "
                f"{old_stage7_retry} -> {self.cfg.stage7_quality_retry_max}"
            )
        for label, old_value, new_value in (
            ("Stage7 soft starless asinh stretch", old_stage7_soft, self.cfg.stage7_soft_starless_asinh_stretch),
            ("Stage7 bright-nebula halo threshold", old_stage7_bright_halo, self.cfg.stage7_bright_nebula_halo_residue_score_max),
            ("Stage7 starless repair strength", old_stage7_repair, self.cfg.stage7_starless_repair_strength),
            ("Stage7 starless halo repair strength", old_stage7_halo, self.cfg.stage7_starless_halo_repair_strength),
            ("Stage7 starless chroma denoise strength", old_stage7_chroma, self.cfg.stage7_starless_chroma_denoise_strength),
        ):
            if old_value != new_value:
                self.log.warn(f"{label} clamped: {old_value} -> {new_value}")


    def _apply_ai_env_overrides(self):
        """Compatibility wrapper for isolated Stage11 runners."""
        self._apply_runtime_env_overrides()


    def _safe_unlink(self, path: Path):
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            self.log.debug(f"Unable to remove temp file {path.name}: {e}")


    def _measure_current_features(self) -> Optional[ImageFeatures]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            return measure_image_features(image_data)
        except Exception as e:
            self.log.warn(f"[AI] Failed to measure image features: {e}")
            return None


    def _measure_current_quality(self) -> Optional[QualityMetrics]:
        try:
            image_data = self.siril.get_image_pixeldata(preview=False)
            return measure_quality_metrics(image_data)
        except Exception as e:
            self.log.warn(f"[AI] Failed to measure image quality metrics: {e}")
            return None


    def _read_image_by_stem(self, stem: str) -> Optional[np.ndarray]:
        try:
            self.cmd_with_check("load", stem)
            image_data = self.siril.get_image_pixeldata(preview=False)
            if image_data is None:
                return None
            return np.asarray(image_data)
        except Exception as e:
            self.log.warn(f"读取图像失败 ({stem}): {e}")
            return None


    def _set_current_image_pixeldata(self, image_data: np.ndarray, *, label: str) -> None:
        lock_factory = getattr(self.siril, "image_lock", None)
        if callable(lock_factory):
            with lock_factory():
                self.siril.set_image_pixeldata(image_data)
            return
        self.log.warn(f"{label}: image_lock unavailable, writing pixels without thread lock")
        self.siril.set_image_pixeldata(image_data)


    def _read_fits_header_metadata(self, *candidates: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        paths: List[Path] = []
        for candidate in candidates:
            if not candidate:
                continue
            path = candidate if isinstance(candidate, Path) else Path(str(candidate))
            if path.suffix.lower() not in {".fit", ".fits"}:
                path = path.with_suffix(".fit")
            if not path.is_absolute() and self.process_dir:
                path = self.process_dir / path
            paths.append(path)
        for path in paths:
            try:
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    while True:
                        block = handle.read(2880)
                        if not block:
                            break
                        for offset in range(0, len(block), 80):
                            card = block[offset:offset + 80].decode("ascii", errors="ignore")
                            key = card[:8].strip()
                            if key == "END":
                                metadata["_header_source"] = str(path)
                                return metadata
                            if not key or card[8:10] != "= ":
                                continue
                            raw = card[10:80].split("/", 1)[0].strip()
                            if raw.startswith("'") and "'" in raw[1:]:
                                value: Any = raw[1:raw.find("'", 1)]
                            else:
                                try:
                                    value = int(raw)
                                except ValueError:
                                    try:
                                        value = float(raw)
                                    except ValueError:
                                        value = raw
                            metadata[key] = value
            except OSError as e:
                self.log.debug(f"Unable to read FITS header {path}: {e}")
        return metadata


    def _write_stage_json(self, filename: str, payload: Dict[str, Any]) -> None:
        write_stage_json(self.process_dir, self.log, filename, payload)


    def _write_ai_raw_response(
        self,
        stage_name: str,
        *,
        endpoint_url: str,
        temperature: float,
        json_mode: bool,
        response_obj: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
        error_text: Optional[str] = None,
    ) -> None:
        self._ai_raw_response_counter = write_ai_raw_response(
            self.process_dir,
            self.log,
            self._ai_raw_response_counter,
            self._short_text,
            stage_name,
            endpoint_url=endpoint_url,
            temperature=temperature,
            json_mode=json_mode,
            response_obj=response_obj,
            content=content,
            error_text=error_text,
        )

