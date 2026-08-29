"""Qt-free presentation models for one verified Starun task run.

The GUI may use history status as a progress hint, but terminal delivery state is
derived only from a hash-verified run bundle.  This module deliberately performs
no filesystem access so it is also usable from tests and non-Qt consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class RunOutcome(str, Enum):
    """Run states rendered by the result workbench."""

    PREPARING = "preparing"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    REVIEW_REQUIRED = "review_required"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    VERIFICATION_FAILED = "verification_failed"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class VerifiedOutput:
    """One result-manifest output whose on-disk SHA-256 matched."""

    name: str
    path: Path
    sha256: str
    size: int
    kind: str

    def __fspath__(self) -> str:
        return str(self.path)


@dataclass(frozen=True, slots=True)
class VerifiedRunBundle:
    """Read-only evidence collected for a verified run-manifest identity.

    ``plan`` and ``result`` are exposed only after their own integrity and
    identity checks.  ``lineage_verified`` additionally binds their run id and
    plan hash.  A bundle can therefore represent an incomplete or damaged run
    without making it eligible for delivery.
    """

    run_root: Path
    run_manifest: Mapping[str, Any]
    plan: Mapping[str, Any] | None
    result: Mapping[str, Any] | None
    verified_outputs: tuple[VerifiedOutput, ...]
    verified_png: Path | None
    plan_verified: bool = False
    result_verified: bool = False
    lineage_verified: bool = False
    integrity_errors: tuple[str, ...] = ()
    verification_issues: tuple[str, ...] = ()

    @property
    def processing_plan(self) -> Mapping[str, Any] | None:
        """Compatibility alias used by the task inspector."""

        return self.plan

    @property
    def pipeline_result(self) -> Mapping[str, Any] | None:
        """Compatibility alias used by the task inspector."""

        return self.result

    @property
    def verification_errors(self) -> tuple[str, ...]:
        """All reasons the bundle is not a complete delivery chain."""

        return tuple(dict.fromkeys((*self.integrity_errors, *self.verification_issues)))


@dataclass(frozen=True, slots=True)
class RunPresentation:
    """Stable, Qt-free fields consumed by the result workbench."""

    status: RunOutcome
    tone: str
    title: str
    summary: str
    delivery_eligible: bool
    output_kind: str
    verified_outputs: tuple[VerifiedOutput, ...]
    preview_path: Path | None
    review_requirements: tuple[Mapping[str, Any], ...]
    issues: tuple[Mapping[str, Any], ...]
    integrity_error: str | None
    formal_output_names: tuple[str, ...] = ()

    @property
    def outcome(self) -> RunOutcome:
        """Compatibility alias for callers that name the state ``outcome``."""

        return self.status

    @property
    def deliverable(self) -> bool:
        return self.delivery_eligible

    @property
    def download_enabled(self) -> bool:
        return self.delivery_eligible

    @property
    def reasons(self) -> tuple[str, ...]:
        values: list[str] = []
        for requirement in self.review_requirements:
            code = str(requirement.get("code") or "").strip()
            if code:
                values.append(code)
        for issue in self.issues:
            message = str(issue.get("message") or issue.get("code") or "").strip()
            if message:
                values.append(message)
        return tuple(dict.fromkeys(values))


_NON_ELEVATING_FALLBACKS = frozenset(
    {
        RunOutcome.PREPARING,
        RunOutcome.RUNNING,
        RunOutcome.STOPPED,
        RunOutcome.INTERRUPTED,
        RunOutcome.FAILED,
    }
)
_OUTPUT_REQUIRED_OUTCOMES = frozenset(
    {
        RunOutcome.SUCCESS,
        RunOutcome.PARTIAL_SUCCESS,
        RunOutcome.REVIEW_REQUIRED,
    }
)


def _coerce_outcome(value: Any) -> RunOutcome | None:
    normalized = str(getattr(value, "value", value) or "").strip().lower()
    if not normalized:
        return None
    try:
        return RunOutcome(normalized)
    except ValueError:
        return None


def _mapping_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(record) for record in value if isinstance(record, Mapping))


def formal_output_allowlist(
    result: Mapping[str, Any] | None,
) -> frozenset[str] | None:
    """Return the exact formal artifact names, or ``None`` for no valid list."""

    if not isinstance(result, Mapping):
        return None
    delivery_gates = result.get("delivery_gates")
    if (
        not isinstance(delivery_gates, Mapping)
        or str(delivery_gates.get("schema") or "")
        != "starun.final-delivery-gates.v1"
    ):
        return None
    artifact_gate = delivery_gates.get("artifacts")
    if not isinstance(artifact_gate, Mapping):
        return None
    raw_names = artifact_gate.get("formal_outputs")
    if not isinstance(raw_names, (list, tuple)):
        return None
    names: list[str] = []
    for raw_name in raw_names:
        if not isinstance(raw_name, str):
            return None
        name = raw_name.strip()
        if not name or name != raw_name or Path(name).name != name:
            return None
        names.append(name)
    if len(names) != len(set(names)):
        return None
    return frozenset(names)


def _plan_requires_review(plan: Mapping[str, Any] | None) -> bool:
    if not isinstance(plan, Mapping):
        return False
    route = plan.get("route")
    output = plan.get("output")
    return bool(
        (isinstance(route, Mapping) and route.get("review_only", False))
        or (isinstance(output, Mapping) and output.get("review_only", False))
    )


def _integrity_issue(message: str) -> Mapping[str, Any]:
    return {
        "stage": 0,
        "component": "run_verification",
        "severity": "fatal",
        "code": "run_bundle_verification_failed",
        "recovered": False,
        "message": str(message),
    }


def _failure_summary(result: Mapping[str, Any] | None) -> str:
    if not isinstance(result, Mapping):
        return "处理未完成，且没有可验证的流水线结果。"
    reason = str(result.get("failure_reason") or "").strip()
    if reason:
        return reason
    errors = result.get("errors")
    if isinstance(errors, (list, tuple)):
        message = next((str(value).strip() for value in errors if str(value).strip()), "")
        if message:
            return message
    return "处理未能生成可交付结果，请查看阶段详情和日志。"


def _base_copy(
    *,
    status: RunOutcome,
    result: Mapping[str, Any] | None,
    review_count: int,
) -> tuple[str, str, str]:
    if status is RunOutcome.PREPARING:
        return "info", "正在准备任务", "正在校验输入、运行环境与处理计划。"
    if status is RunOutcome.RUNNING:
        return "info", "正在处理图像", "流水线正在运行，结果尚未形成正式交付链。"
    if status is RunOutcome.SUCCESS:
        return "success", "处理完成", "处理计划、运行身份与输出文件均已验证，可以安全下载。"
    if status is RunOutcome.PARTIAL_SUCCESS:
        return (
            "warning",
            "降级完成，正式结果可用",
            "正式结果已通过完整性校验；处理包含降级或安全回退，下载时会保留披露信息。",
        )
    if status is RunOutcome.REVIEW_REQUIRED:
        detail = f"共有 {review_count} 项需要人工确认。" if review_count else "存在需要人工确认的处理条件。"
        return "warning", "处理完成，仅供复核", f"结果完整性已验证；{detail}"
    if status is RunOutcome.STOPPED:
        return "neutral", "处理已中止", "本次运行已由用户中止，没有正式交付结果。"
    if status is RunOutcome.INTERRUPTED:
        return "warning", "处理异常中断", "应用或处理进程未正常结束，请检查日志后重试。"
    if status is RunOutcome.FAILED:
        return "error", "处理失败", _failure_summary(result)
    return "error", "结果无法验证", "运行产物的身份或完整性校验失败，不能交付。"


def build_run_presentation(
    bundle: VerifiedRunBundle | None,
    fallback_status: str | RunOutcome | None = None,
) -> RunPresentation:
    """Build fail-closed result-page state from verified evidence.

    ``fallback_status`` is intentionally restricted to progress/failure hints.
    In particular, a history value of ``success`` can never grant delivery or
    replace a missing/invalid processing plan, signed result, or output hash.
    """

    fallback = _coerce_outcome(fallback_status)
    if bundle is None:
        status = (
            fallback
            if fallback in _NON_ELEVATING_FALLBACKS
            else RunOutcome.VERIFICATION_FAILED
        )
        tone, title, summary = _base_copy(status=status, result=None, review_count=0)
        missing = (
            ()
            if status in _NON_ELEVATING_FALLBACKS
            else (_integrity_issue("没有可验证的运行包"),)
        )
        return RunPresentation(
            status=status,
            tone=tone,
            title=title,
            summary=summary,
            delivery_eligible=False,
            output_kind="none",
            verified_outputs=(),
            preview_path=None,
            review_requirements=(),
            issues=missing,
            integrity_error=("没有可验证的运行包" if missing else None),
            formal_output_names=(),
        )

    result = bundle.result if bundle.result_verified else None
    result_status = _coerce_outcome(result.get("status") if result else None)
    signed_user_interrupted = bool(
        result_status is RunOutcome.FAILED
        and str(result.get("failure_reason") or "").strip().casefold()
        == "user interrupted"
    ) if result is not None else False
    plan_requires_review = _plan_requires_review(
        bundle.plan if bundle.plan_verified else None
    )
    formal_allowlist = formal_output_allowlist(result)
    presentation_verified_outputs = tuple(
        replace(output, kind="auxiliary")
        if output.kind == "formal"
        and (
            formal_allowlist is None
            or output.name not in formal_allowlist
        )
        else output
        for output in bundle.verified_outputs
    )
    presentation_outputs = tuple(
        output
        for output in presentation_verified_outputs
        if output.kind in {"formal", "review"}
    )
    formal_outputs = tuple(
        output for output in presentation_outputs if output.kind == "formal"
    )
    verified_formal_names = {output.name for output in formal_outputs}
    trusted_identity = bool(
        bundle.plan_verified
        and bundle.result_verified
        and bundle.lineage_verified
        and not bundle.integrity_errors
    )
    complete_trusted_chain = bool(trusted_identity and presentation_outputs)

    if signed_user_interrupted:
        status = RunOutcome.STOPPED
    elif plan_requires_review and result_status in _OUTPUT_REQUIRED_OUTCOMES:
        status = (
            RunOutcome.REVIEW_REQUIRED
            if complete_trusted_chain
            else RunOutcome.VERIFICATION_FAILED
        )
    elif result_status in {RunOutcome.SUCCESS, RunOutcome.PARTIAL_SUCCESS}:
        status = (
            result_status
            if trusted_identity and formal_outputs
            else RunOutcome.VERIFICATION_FAILED
        )
    elif result_status is RunOutcome.REVIEW_REQUIRED:
        status = (
            result_status
            if complete_trusted_chain
            else RunOutcome.VERIFICATION_FAILED
        )
    elif result_status is RunOutcome.FAILED:
        status = RunOutcome.FAILED
    elif fallback in _NON_ELEVATING_FALLBACKS:
        status = fallback
    else:
        status = RunOutcome.VERIFICATION_FAILED

    reviews = _mapping_records(result.get("review_requirements") if result else None)
    delivery_gates = (
        result.get("delivery_gates")
        if isinstance(result, Mapping)
        else None
    )
    delivery_contract_present = bool(
        isinstance(delivery_gates, Mapping)
        and str(delivery_gates.get("schema") or "")
        == "starun.final-delivery-gates.v1"
    )
    scientific_gate = (
        delivery_gates.get("scientific")
        if delivery_contract_present
        and isinstance(delivery_gates.get("scientific"), Mapping)
        else {}
    )
    presentation_gate = (
        delivery_gates.get("presentation")
        if delivery_contract_present
        and isinstance(delivery_gates.get("presentation"), Mapping)
        else {}
    )
    artifact_gate = (
        delivery_gates.get("artifacts")
        if delivery_contract_present
        and isinstance(delivery_gates.get("artifacts"), Mapping)
        else {}
    )
    review_gate = (
        delivery_gates.get("review")
        if delivery_contract_present
        and isinstance(delivery_gates.get("review"), Mapping)
        else {}
    )
    legacy_delivery_contract = bool(
        not delivery_contract_present
        or formal_allowlist is None
        or delivery_gates.get("legacy_delivery_contract") is not False
    )
    reported_formal_count = artifact_gate.get("formal_count")
    formal_count_matches = bool(
        isinstance(reported_formal_count, int)
        and not isinstance(reported_formal_count, bool)
        and formal_allowlist is not None
        and reported_formal_count == len(formal_allowlist)
    )
    formal_outputs_complete = bool(
        formal_allowlist
        and verified_formal_names == set(formal_allowlist)
    )
    delivery_contract_accepted = bool(
        delivery_contract_present
        and not legacy_delivery_contract
        and delivery_gates.get("formal_delivery_accepted") is True
        and scientific_gate.get("accepted") is True
        and presentation_gate.get("accepted") is True
        and artifact_gate.get("accepted") is True
        and review_gate.get("accepted") is True
        and formal_count_matches
        and formal_outputs_complete
    )
    if (
        result_status in {RunOutcome.SUCCESS, RunOutcome.PARTIAL_SUCCESS}
        and not delivery_contract_accepted
    ):
        reviews = (
            *reviews,
            {
                "stage": 10,
                "code": (
                    "legacy_delivery_contract"
                    if legacy_delivery_contract
                    else "delivery_gates_rejected"
                ),
                "message": (
                    "历史结果缺少科学、表现双门或正式产物名单，"
                    "不能参加本轮正式验收。"
                    if legacy_delivery_contract
                    else "科学、表现、复核或正式产物身份门尚未全部通过。"
                ),
            },
        )
    if plan_requires_review and status is RunOutcome.REVIEW_REQUIRED:
        reviews = (
            *reviews,
            {
                "stage": 0,
                "code": "frozen_plan_review_only",
                "message": "冻结处理计划要求本次结果仅供复核。",
            },
        )
    result_issues = _mapping_records(result.get("issues") if result else None)
    verification_messages = (
        bundle.integrity_errors
        if result is None and status in _NON_ELEVATING_FALLBACKS
        else bundle.verification_errors
    )
    if status is RunOutcome.VERIFICATION_FAILED and not verification_messages:
        verification_messages = ("处理计划、结果或输出文件未形成完整的可信链",)
    verification_issues = tuple(_integrity_issue(message) for message in verification_messages)
    issues = (*result_issues, *verification_issues)

    review_flag = bool(result.get("review_required", False)) if result else False
    delivery_eligible = bool(
        status in {RunOutcome.SUCCESS, RunOutcome.PARTIAL_SUCCESS}
        and complete_trusted_chain
        and formal_outputs
        and not reviews
        and not review_flag
        and delivery_contract_accepted
    )
    if delivery_eligible:
        output_kind = "formal"
    elif presentation_outputs:
        output_kind = "review"
    else:
        output_kind = "none"
    tone, title, summary = _base_copy(
        status=status,
        result=result,
        review_count=len(reviews),
    )
    if (
        status in {RunOutcome.SUCCESS, RunOutcome.PARTIAL_SUCCESS}
        and complete_trusted_chain
        and not delivery_eligible
        and (reviews or review_flag)
    ):
        tone = "warning"
        if status is RunOutcome.PARTIAL_SUCCESS:
            title = "降级完成，仅供复核"
            summary = (
                "结果完整性已验证并包含降级或安全回退；"
                "仍有需要人工确认的条件，不能作为正式结果。"
            )
        else:
            title = "处理完成，仅供复核"
            summary = (
                "结果完整性已验证；存在需要人工确认的处理条件，"
                "不能作为正式结果。"
            )
    integrity_error = (
        "；".join(verification_messages)
        if verification_messages
        else None
    )
    presentation_paths = {output.path for output in presentation_outputs}
    preview_path = (
        bundle.verified_png
        if bundle.verified_png in presentation_paths
        else None
    )
    return RunPresentation(
        status=status,
        tone=tone,
        title=title,
        summary=summary,
        delivery_eligible=delivery_eligible,
        output_kind=output_kind,
        verified_outputs=presentation_verified_outputs,
        preview_path=preview_path,
        review_requirements=reviews,
        issues=issues,
        integrity_error=integrity_error,
        formal_output_names=tuple(sorted(formal_allowlist or ())),
    )


__all__ = [
    "RunOutcome",
    "RunPresentation",
    "VerifiedOutput",
    "VerifiedRunBundle",
    "build_run_presentation",
    "formal_output_allowlist",
]
