from __future__ import annotations

from typing import Optional


def _clamp_float(value: object, min_value: float, max_value: float) -> float:
    try:
        fvalue = float(value)  # type: ignore[arg-type]
    except Exception:
        fvalue = min_value
    return max(min_value, min(max_value, fvalue))


def run_siril_scunet_denoise_fallback(
    pipeline: object,
    step_key: str,
    strength: float,
    *,
    command_error_types: tuple[type[BaseException], ...],
    recoverable_error_types: tuple[type[BaseException], ...],
) -> Optional[str]:
    """
    Run the Siril-SCUNet fallback chain.

    The script path is preferred because it keeps behavior aligned with the
    plugin package; command aliases are kept as compatibility fallbacks.
    """
    pipeline._last_scunet_fallback_error = None
    script_reason: Optional[str] = None
    scunet_script = pipeline._find_plugin_script(("processing/SCUNet_Denoise.py",))
    if scunet_script is not None:
        used = pipeline._run_plugin_script_cli_subprocess(
            step_key,
            "Siril-SCUNet Denoise",
            scunet_script,
            args=(),
            timeout_sec=300,
        )
        if used:
            return used
        script_reason = (
            getattr(pipeline, "_last_plugin_script_error", None)
            or f"{scunet_script.name}: execution failed"
        )
    else:
        script_reason = "SCUNet_Denoise.py 脚本缺失"

    strength_text = f"{_clamp_float(strength, 0.0, 1.0):.2f}"
    candidates = [
        ("Siril-SCUNet Denoise", ("siril_scunet_denoise",)),
        ("Siril-SCUNet Denoise", ("siril_scunet_denoise", strength_text)),
        ("Siril-SCUNet Denoise", ("scunet_denoise",)),
        ("Siril-SCUNet Denoise", ("scunet_denoise", strength_text)),
        ("Siril-SCUNet Denoise", ("siril_scunet",)),
        ("Siril-SCUNet Denoise", ("scunet",)),
        ("Siril-SCUNet Denoise", ("scunet", f"-strength={strength_text}")),
    ]

    command_reason: Optional[str] = None
    for index, (label, args) in enumerate(candidates):
        try:
            pipeline.cmd_with_check(*args, quiet=True)
            pipeline.workflow_command_used[step_key] = label
            pipeline.log.info(f"{step_key} 使用命令: {label}")
            return label
        except command_error_types as e:
            text = str(e).strip()
            lowered = text.lower()
            if (
                index == 0
                and ("command not found" in lowered or "unknown command" in lowered)
            ):
                command_reason = "当前 Siril 构建未暴露 SCUNet 命令（unknown command）"
                pipeline.log.debug(
                    f"{step_key} 跳过其余 SCUNet 命令别名探测: {command_reason}"
                )
                break
            command_reason = pipeline._short_text(text, 160)
            continue
        except recoverable_error_types as e:
            command_reason = pipeline._short_text(e, 160)
            continue

    reasons = [reason for reason in (script_reason, command_reason) if reason]
    if reasons:
        pipeline._last_scunet_fallback_error = "; ".join(reasons)
    return None
