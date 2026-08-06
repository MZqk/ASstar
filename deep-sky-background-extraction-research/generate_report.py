#!/usr/bin/env python3
"""Generate the Stage 3 background-extraction research report."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
OUTLINE_PATH = ROOT / "outline.yaml"
FIELDS_PATH = ROOT / "fields.yaml"
REPORT_PATH = ROOT / "report.md"

SUMMARY_FIELDS = ("standard_status", "evidence_level")
INTERNAL_KEYS = {"_source_file", "uncertain"}

# Required compatibility aliases from the deep-research report contract, plus
# the categories used by this report.
CATEGORY_MAPPING = {
    "基本信息": ["basic_info", "基本信息"],
    "技术特性": ["technical_features", "technical_characteristics", "技术特性"],
    "性能指标": ["performance_metrics", "performance", "性能指标"],
    "里程碑意义": ["milestone_significance", "milestones", "里程碑意义"],
    "商业信息": ["business_info", "commercial_info", "商业信息"],
    "竞争与生态": ["competition_ecosystem", "competition", "竞争与生态"],
    "历史沿革": ["history", "历史沿革"],
    "市场定位": ["market_positioning", "market", "市场定位"],
    "结论": ["conclusion", "结论"],
    "技术审计": ["technical_audit", "技术审计"],
    "判定与安全性": ["decision_and_safety", "判定与安全性"],
    "项目落地": ["project_delivery", "项目落地"],
}

FIELD_LABELS = {
    "item_name": "调研项",
    "executive_conclusion": "核心结论",
    "standard_status": "标准状态",
    "evidence_level": "证据强度与局限",
    "metric_definition": "指标与公式",
    "threshold_provenance": "阈值来源",
    "scale_and_state_sensitivity": "尺度与输入状态敏感性",
    "contamination_classification": "污染类型区分",
    "need_correction_criteria": "是否需要校正",
    "auto_apply_safety_criteria": "自动执行安全条件",
    "sky_coverage_and_sampling": "天空覆盖与采样",
    "residual_and_preservation_tests": "残差与目标保真验证",
    "high_risk_cases": "高风险场景",
    "project_assessment": "当前项目评估",
    "recommended_gate": "建议判定门",
    "calibration_protocol": "标定方案",
    "source_references": "直接证据来源",
    "open_questions": "待确认问题",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def normalized_name(value: Any) -> str:
    return re.sub(r"[\s_—–-]+", "", str(value or "")).lower()


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data["_source_file"] = path.name
            results.append(data)
    return results


def order_results(
    results: list[dict[str, Any]], outline: dict[str, Any]
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    used: set[int] = set()
    for spec in outline.get("items", []):
        wanted = normalized_name(spec.get("name")) if isinstance(spec, dict) else ""
        match = next(
            (
                item
                for item in results
                if normalized_name(item.get("item_name")) == wanted
                or wanted in normalized_name(item.get("item_name"))
                or normalized_name(item.get("item_name")) in wanted
            ),
            None,
        )
        if match is not None and id(match) not in used:
            ordered.append(match)
            used.add(id(match))
    ordered.extend(item for item in results if id(item) not in used)
    return ordered


def categories(fields: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    raw = fields.get("field_categories") or fields.get("categories") or []
    return [
        (
            str(category.get("category") or "其他信息"),
            [field for field in category.get("fields", []) if isinstance(field, dict)],
        )
        for category in raw
        if isinstance(category, dict)
    ]


def walk_for_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        if field_name in value:
            return value[field_name]
        for child in value.values():
            found = walk_for_field(child, field_name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = walk_for_field(child, field_name)
            if found is not None:
                return found
    return None


def get_field(data: dict[str, Any], category: str, field_name: str) -> Any:
    if field_name in data:
        return data[field_name]
    aliases = CATEGORY_MAPPING.get(category, [category])
    for alias in aliases:
        section = data.get(alias)
        if isinstance(section, dict) and field_name in section:
            return section[field_name]
    return walk_for_field(data, field_name)


def contains_uncertain(value: Any) -> bool:
    if isinstance(value, str):
        return "[不确定]" in value
    if isinstance(value, dict):
        return any(contains_uncertain(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_uncertain(item) for item in value)
    return False


def is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def inline(value: Any, limit: int = 120) -> str:
    if isinstance(value, list):
        text = "；".join(str(item) for item in value if not isinstance(item, (dict, list)))
    elif isinstance(value, dict):
        text = "；".join(f"{key}: {item}" for key, item in value.items())
    else:
        text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_value(value: Any) -> str:
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if is_empty(item) or contains_uncertain(item):
                continue
            if isinstance(item, dict):
                parts = [f"**{key}**: {inline(child, 240)}" for key, child in item.items()]
                lines.append("- " + " | ".join(parts))
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)
    if isinstance(value, dict):
        return "\n".join(
            f"- **{key}**: {render_value(child)}"
            for key, child in value.items()
            if key not in INTERNAL_KEYS and not is_empty(child) and not contains_uncertain(child)
        )
    text = str(value)
    return "> " + text.replace("\n", "\n> ") if len(text) > 100 else text


def synthesis() -> str:
    return """## 综合结论

当前 Stage 3 的 `0.045 / 0.16 / 0.08 / 0.18 / 0.75` 是项目内部工程门禁，不是行业标准，也没有在仓库中找到真实图像标注、注入真值、盲评、ROC、阈值敏感性或跨设备分层标定依据。评分公式在 2026-05-01 引入，五个门禁配置及三态决策在 2026-07-31 的两个相邻提交中加入；源码自身也把依据写为 `project internal engineering gate`。

代表性的官方软件与科研资料没有采用统一的“文件需要背景提取”标量阈值。共同做法是先明确输入仍处于线性状态，再区分加性天空梯度、乘性暗角/平场误差、方向性图样噪声和真实大尺度天体信号；随后检查源掩膜、真实天空覆盖、模型尺度与复杂度、背景/RMS 图、残差结构及目标通量保真。[Siril](https://siril.readthedocs.io/en/latest/processing/background.html)要求样点避开对象，并对填满画面的星云降低模型复杂度；[SExtractor](https://sextractor.readthedocs.io/en/stable/Background.html)明确警告网格过小会把扩展目标通量吸收到背景图；[Photutils](https://photutils.readthedocs.io/en/2.3.0/api/photutils.background.Background2D.html)把源掩膜、覆盖掩膜、每网格有效像素数和背景 RMS 作为一等数据。

[STScI DrizzlePac](https://hst-docs.stsci.edu/drizzpac/chapter-6-reprocessing-with-the-drizzlepac-package/6-3-running-astrodrizzle)指出窄带流水线通常关闭天空扣除，扩展源占满视场且没有真实天空像素时几乎肯定应关闭；低表面亮度研究进一步表明，未掩膜弱源、恒星扩展 PSF 和复杂局部模型会造成背景高估与扩展结构过扣。[Watkins 等（2024）](https://academic.oup.com/mnras/article/528/3/4289/7603727)给出的面亮度和掩膜深度数值只适用于其 HSC/LSST 类实验，不能换算成 Stage 3 通用阈值。

### 当前门禁的实际含义

| 条件 | 当前动作 | 说明 |
|---|---|---|
| `gradient ≤ 0.045` 且 `dirty ≤ 0.16` | `skip` | 固定置信度 `0.90` |
| `gradient ≥ 0.08` 且 `dirty ≥ 0.18` | 可确定性 `apply` | 置信度从 `0.75` 映射到最高 `0.95` |
| 其他组合 | `review_required` | 保留基线 |
| 弥散/大尺度信号风险 | 通常复核或保护性跳过 | 默认禁止自动 DBE |

这些 `confidence` 是手工规则的输出，不是经过概率校准的正确率。`stage3_apply_confidence_min=0.75` 约束外部 apply 建议；确定性路径则以两项 apply 门为资格条件，并自行从 `0.75` 起算。

### M31 样本的尺度问题

指定运行的 `gradient_score=0.039167`、`dirty_background_score=0.067757`，所以按当前规则跳过。其 `bg_std≈0.0001179`，使 `max(6×bg_std, 0.01)` 取固定下限 `0.01`；该分数因此不是六倍背景噪声归一化的显著性。它对应的四边中位数峰谷约 `0.000392`，约为背景中位数的 `1.55%`、单像素 `bg_std` 的 `3.32` 倍，但边带中位数的抽样不确定度未估计，不能将其称作 `3.32σ` 检验。`dirty` 的三个加权贡献中约 `77%` 来自色度项。由此可知当前分数对线性数值尺度、输入编码、裁切和通道语义敏感。

### 推荐替代逻辑

1. 先判定“是否存在需要校正的加性低频背景结构”：要求线性输入、污染分类明确、充足且空间分布良好的真实天空样本，并在多尺度/轻微裁切下保持稳定。
2. 再判定“候选是否安全自动应用”：候选必须在空间分块留出样本上优于 no-op 基线，降低残差低频趋势，同时通过星点、颜色、噪声、星云/星系外晕通量和形态保真门。
3. 乘性暗角回到 flat/calibration；banding 与 walking noise 走传感器、抖动和叠加诊断；无真实天空、IFN、低表面亮度、扩展目标占满视场、窄带和类别不明一律复核或保留基线。
4. 新门先以 shadow mode 记录建议而不改像素，用分层标注集和注入真值标定。自动 apply 优先优化高精确率和低误扣率，而不是覆盖率；所有公式、掩膜、数据集和阈值都要版本化。
"""


def build_report() -> str:
    outline = load_yaml(OUTLINE_PATH)
    field_defs = load_yaml(FIELDS_PATH)
    output_dir = Path(outline.get("execution", {}).get("output_dir") or ROOT / "results")
    results = order_results(load_results(output_dir), outline)
    category_defs = categories(field_defs)
    known_fields = {
        str(field.get("name")) for _, fields in category_defs for field in fields
    }

    lines = [
        f"# {outline.get('topic', '深度调研报告')}",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "调研口径：不限年代；优先项目源码与提交历史、天文机构/软件官方文档和同行评审论文。标为不确定的值不进入正文。",
        "",
        synthesis().rstrip(),
        "",
        "## 目录",
        "",
    ]

    for index, data in enumerate(results, start=1):
        name = str(data.get("item_name") or data.get("_source_file") or f"调研项 {index}")
        summaries = []
        for field_name in SUMMARY_FIELDS:
            value = get_field(data, "", field_name)
            if not is_empty(value) and not contains_uncertain(value):
                summaries.append(f"{FIELD_LABELS[field_name]}：{inline(value)}")
        suffix = " — " + "；".join(summaries) if summaries else ""
        lines.append(f"{index}. [{name}](#item-{index}){suffix}")

    for index, data in enumerate(results, start=1):
        name = str(data.get("item_name") or data.get("_source_file") or f"调研项 {index}")
        uncertain = {
            str(field_name).strip()
            for field_name in data.get("uncertain", [])
            if isinstance(field_name, str)
        }
        lines.extend(["", f'<a id="item-{index}"></a>', f"## {index}. {name}", ""])
        for category_name, field_specs in category_defs:
            rendered: list[str] = []
            for field_spec in field_specs:
                field_name = str(field_spec.get("name") or "")
                value = get_field(data, category_name, field_name)
                if (
                    not field_name
                    or field_name in uncertain
                    or is_empty(value)
                    or contains_uncertain(value)
                ):
                    continue
                rendered.extend(
                    [f"#### {FIELD_LABELS.get(field_name, field_name)}", "", render_value(value), ""]
                )
            if rendered:
                lines.extend([f"### {category_name}", "", *rendered])

        extras = {
            key: value
            for key, value in data.items()
            if key not in INTERNAL_KEYS
            and key not in known_fields
            and not is_empty(value)
            and not contains_uncertain(value)
        }
        if extras:
            lines.extend(["### 其他信息", ""])
            for key, value in extras.items():
                lines.extend([f"#### {key}", "", render_value(value), ""])

        if uncertain:
            lines.extend(["### 已排除的未确认字段", ""])
            lines.extend(f"- {FIELD_LABELS.get(field_name, field_name)}" for field_name in sorted(uncertain))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
