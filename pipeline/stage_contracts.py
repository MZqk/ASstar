"""Canonical product-stage, checkpoint, and artifact contracts.

This module deliberately separates stable product terminology from legacy
runtime labels and aliases.  New task plans use the canonical names here;
legacy names are read-only compatibility inputs and are never canonical
outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple


PIPELINE_CONTRACT_SCHEMA = "seestar.pipeline-stage-contract.v1"
PIPELINE_CONTRACT_VERSION = "1.0.0"
PRODUCT_STAGE_NUMBERS = tuple(range(1, 11))
FORMAL_RESUME_STAGES = (1, 2, 5)


class StagePhase(str, Enum):
    """Product-facing processing domains."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class StageContract:
    """Stable identity and artifact contract for one formal stage."""

    number: int
    key: str
    title: str
    phase: StagePhase
    primary_artifact: str
    runtime_label: str
    formal_resume_checkpoint: bool = False
    legacy_read_aliases: Tuple[str, ...] = ()
    product_stage: bool = True

    @property
    def display_label(self) -> str:
        return f"Stage {self.number} · {self.title}"

    @property
    def artifact_prefix(self) -> str:
        return f"stage{self.number}_"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.number,
            "key": self.key,
            "title": self.title,
            "display_label": self.display_label,
            "phase": self.phase.value,
            "primary_artifact": self.primary_artifact,
            "artifact_prefix": self.artifact_prefix,
            "formal_resume_checkpoint": self.formal_resume_checkpoint,
            "legacy_read_aliases": list(self.legacy_read_aliases),
            "product_stage": self.product_stage,
        }


_STAGE_CONTRACTS = (
    StageContract(
        1,
        "input_preparation",
        "输入准备",
        StagePhase.LINEAR,
        "stage1_prepared.fit",
        "阶段 1: 前期准备",
        formal_resume_checkpoint=True,
    ),
    StageContract(
        2,
        "boundary_correction",
        "边界校正",
        StagePhase.LINEAR,
        "stage2_corrected.fit",
        "阶段 2: 裁切",
        formal_resume_checkpoint=True,
    ),
    StageContract(
        3,
        "background_processing",
        "背景处理",
        StagePhase.LINEAR,
        "stage3_bgremoved.fit",
        "阶段 3: 背景提取",
    ),
    StageContract(
        4,
        "color_calibration",
        "图像解析与色彩校准",
        StagePhase.LINEAR,
        "stage4_color.fit",
        "阶段 4: 图像解析 + 色彩校准",
        legacy_read_aliases=("stage4_colorbalanced.fit",),
    ),
    StageContract(
        5,
        "linear_cleanup",
        "线性反卷积与降噪",
        StagePhase.LINEAR,
        "stage5_linear.fit",
        "阶段 5: 线性反卷积 / 轻降噪",
        formal_resume_checkpoint=True,
        legacy_read_aliases=("stage5_denoised.fit", "result_linear.fit"),
    ),
    StageContract(
        6,
        "linear_star_separation",
        "线性去星与星点层准备",
        StagePhase.LINEAR,
        "stage6_starless.fit",
        "阶段 6: 去星与星点层准备",
        legacy_read_aliases=("stage7_starless.fit",),
    ),
    StageContract(
        7,
        "subject_stretch",
        "主体拉伸",
        StagePhase.NONLINEAR,
        "stage7_stretched.fit",
        "阶段 7: 主体拉伸",
    ),
    StageContract(
        8,
        "starless_enhancement",
        "Starless 增强",
        StagePhase.NONLINEAR,
        "stage8_enhanced.fit",
        "阶段 8: Starless 深加工",
        legacy_read_aliases=("starless_enhanced.fit",),
    ),
    StageContract(
        9,
        "star_remixing",
        "星点处理与合成",
        StagePhase.NONLINEAR,
        "stage9_remixed.fit",
        "阶段 9: 星点处理与合成",
    ),
    StageContract(
        10,
        "final_export",
        "最终降噪与导出",
        StagePhase.NONLINEAR,
        "stage10_final.fit",
        "阶段 10: 最终降噪与导出",
    ),
    StageContract(
        11,
        "ai_postprocess",
        "AI 后期美化",
        StagePhase.OPTIONAL,
        "stage11_ai_output.fit",
        "阶段 11: AI 后期美化",
        product_stage=False,
    ),
)

STAGE_CONTRACTS_BY_NUMBER = {
    contract.number: contract for contract in _STAGE_CONTRACTS
}

RESULT_ARTIFACT_FAMILIES = {
    "linear": "result_linear.fit",
    "processed_png": "result_processed.png",
    "processed_tiff": "result_processed.tif",
    "final_fits": "result_final.fit",
    "review_prefix": "result_review",
}


def stage_contract(stage_number: int) -> StageContract:
    """Return the canonical contract for a stage number."""

    try:
        return STAGE_CONTRACTS_BY_NUMBER[int(stage_number)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unknown pipeline stage: {stage_number!r}") from error


def product_stage_contracts() -> Tuple[StageContract, ...]:
    """Return the ordered Stage 1-10 product contracts."""

    return tuple(stage_contract(number) for number in PRODUCT_STAGE_NUMBERS)


def formal_resume_contracts() -> Tuple[StageContract, ...]:
    """Return the ordered cross-run resume boundaries."""

    return tuple(stage_contract(number) for number in FORMAL_RESUME_STAGES)


def pipeline_contract_manifest() -> Dict[str, Any]:
    """Return the serializable contract embedded in frozen task plans."""

    return {
        "schema": PIPELINE_CONTRACT_SCHEMA,
        "version": PIPELINE_CONTRACT_VERSION,
        "product_stages": list(PRODUCT_STAGE_NUMBERS),
        "linear_stages": [1, 2, 3, 4, 5, 6],
        "nonlinear_stages": [7, 8, 9, 10],
        "formal_resume_stages": list(FORMAL_RESUME_STAGES),
        "stage_artifact_pattern": "stageN_<semantic>.<extension>",
        "result_artifact_pattern": "result_<delivery>.<extension>",
        "stages": [contract.to_dict() for contract in _STAGE_CONTRACTS],
        "result_artifacts": dict(RESULT_ARTIFACT_FAMILIES),
    }


def _validate_contract_registry() -> None:
    numbers = tuple(contract.number for contract in _STAGE_CONTRACTS)
    if numbers != tuple(range(1, 12)):
        raise RuntimeError(f"pipeline stage contracts are not contiguous: {numbers}")
    primary_artifacts = set()
    for contract in _STAGE_CONTRACTS:
        if not contract.primary_artifact.startswith(contract.artifact_prefix):
            raise RuntimeError(
                f"Stage {contract.number} artifact violates stage prefix: "
                f"{contract.primary_artifact}"
            )
        if contract.primary_artifact in primary_artifacts:
            raise RuntimeError(
                f"duplicate primary stage artifact: {contract.primary_artifact}"
            )
        primary_artifacts.add(contract.primary_artifact)
    actual_resume = tuple(
        contract.number
        for contract in _STAGE_CONTRACTS
        if contract.formal_resume_checkpoint
    )
    if actual_resume != FORMAL_RESUME_STAGES:
        raise RuntimeError(
            "formal resume checkpoint registry does not match "
            f"{FORMAL_RESUME_STAGES}: {actual_resume}"
        )
    if any(
        not artifact.startswith("result_")
        for artifact in RESULT_ARTIFACT_FAMILIES.values()
    ):
        raise RuntimeError("delivery artifacts must use the result_ namespace")


_validate_contract_registry()


__all__ = [
    "FORMAL_RESUME_STAGES",
    "PIPELINE_CONTRACT_SCHEMA",
    "PIPELINE_CONTRACT_VERSION",
    "PRODUCT_STAGE_NUMBERS",
    "RESULT_ARTIFACT_FAMILIES",
    "STAGE_CONTRACTS_BY_NUMBER",
    "StageContract",
    "StagePhase",
    "formal_resume_contracts",
    "pipeline_contract_manifest",
    "product_stage_contracts",
    "stage_contract",
]
