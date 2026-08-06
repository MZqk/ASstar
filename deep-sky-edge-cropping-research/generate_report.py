#!/usr/bin/env python3
"""Generate the deep-sky edge-cropping research report from validated JSON results."""

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
RESULTS_DIR = ROOT / "results"
REPORT_PATH = ROOT / "report.md"

SUMMARY_FIELDS = ("recommended_pipeline_stage", "acceptance_and_fallback")

# Bidirectional category aliases. The first eight groups are retained for
# compatibility with the deep-research report contract; the final five are
# the categories used by this research.
CATEGORY_MAPPING = {
    "基本信息": ["basic_info", "基本信息"],
    "技术特性": ["technical_features", "technical_characteristics", "技术特性"],
    "性能指标": ["performance_metrics", "performance", "性能指标"],
    "里程碑意义": ["milestone_significance", "milestones", "里程碑意义"],
    "商业信息": ["business_info", "commercial_info", "商业信息"],
    "竞争与生态": ["competition_ecosystem", "competition", "竞争与生态"],
    "历史沿革": ["history", "历史沿革"],
    "市场定位": ["market_positioning", "market", "市场定位"],
    "研究对象": ["research_object", "研究对象"],
    "检测与算法": ["detection_algorithms", "检测与算法"],
    "决策与保护": ["decision_protection", "决策与保护"],
    "数据完整性与影响": ["data_integrity_impact", "数据完整性与影响"],
    "证据与项目映射": ["evidence_project_mapping", "证据与项目映射"],
}

FIELD_LABELS = {
    "name": "调研项名称",
    "scope_and_question": "范围与问题",
    "pixel_state_taxonomy": "像素状态分类",
    "defect_causes": "缺陷成因",
    "validity_evidence_source": "有效性证据与优先级",
    "required_inputs": "所需输入",
    "metrics_and_thresholds": "指标与阈值",
    "algorithm_steps": "算法步骤",
    "geometric_mask_vs_reliable_mask": "几何掩膜与可靠掩膜",
    "internal_hole_policy": "内部空洞策略",
    "interpolation_and_support_radius": "插值与支持域",
    "recommended_pipeline_stage": "推荐处理阶段",
    "crop_geometry_and_framing_mode": "裁切几何与构图模式",
    "guard_band_rule": "安全保护带规则",
    "science_roi_constraint": "科学目标 ROI 约束",
    "information_loss_and_stop_conditions": "信息损失与停止条件",
    "acceptance_and_fallback": "验收与回退",
    "false_positive_and_failure_modes": "误判与失败模式",
    "cfa_and_channel_constraints": "CFA 与通道约束",
    "wcs_and_auxiliary_propagation": "WCS 与辅助平面传播",
    "downstream_impact": "下游影响",
    "qa_and_validation": "质量验证",
    "crop_provenance": "裁切溯源",
    "software_and_literature_practice": "软件与文献实践",
    "evidence_sources": "直接证据来源",
    "evidence_strength_and_conflicts": "证据强度与冲突",
    "current_project_fit": "当前项目适配情况",
    "implementation_recommendations": "实施建议",
}

INTERNAL_KEYS = {"_source_file", "uncertain"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value if isinstance(value, dict) else {}


def load_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value["_source_file"] = path.name
            results.append(value)
    return results


def normalized_name(value: Any) -> str:
    return re.sub(r"[\s_—–-]+", "", str(value or "")).lower()


def order_results(results: list[dict[str, Any]], outline: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {normalized_name(item.get("name")): item for item in results}
    ordered: list[dict[str, Any]] = []
    used: set[int] = set()
    for spec in outline.get("items", []):
        if not isinstance(spec, dict):
            continue
        wanted = normalized_name(spec.get("name"))
        match = by_name.get(wanted)
        if match is None:
            match = next(
                (
                    item
                    for item in results
                    if wanted in normalized_name(item.get("name"))
                    or normalized_name(item.get("name")) in wanted
                ),
                None,
            )
        if match is not None and id(match) not in used:
            ordered.append(match)
            used.add(id(match))
    ordered.extend(item for item in results if id(item) not in used)
    return ordered


def category_aliases(category: str) -> list[str]:
    aliases = list(CATEGORY_MAPPING.get(category, [category]))
    for canonical, candidate_aliases in CATEGORY_MAPPING.items():
        if category in candidate_aliases and canonical not in aliases:
            aliases.append(canonical)
    return aliases


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
    for alias in category_aliases(category):
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


def format_inline(value: Any, limit: int = 92) -> str:
    if isinstance(value, list):
        text = "；".join(str(item) for item in value if not isinstance(item, (dict, list)))
    elif isinstance(value, dict):
        text = "；".join(f"{key}: {item}" for key, item in value.items())
    else:
        text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def format_value(value: Any, depth: int = 0) -> str:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if key in INTERNAL_KEYS or is_empty(item) or contains_uncertain(item):
                continue
            if isinstance(item, (dict, list)):
                rendered = format_value(item, depth + 1)
                lines.append(f"- **{key}**:")
                lines.extend("  " + line for line in rendered.splitlines())
            else:
                lines.append(f"- **{key}**: {item}")
        return "\n".join(lines)

    if isinstance(value, list):
        lines = []
        for item in value:
            if is_empty(item) or contains_uncertain(item):
                continue
            if isinstance(item, dict):
                parts = []
                for key, child in item.items():
                    if key in INTERNAL_KEYS or is_empty(child) or contains_uncertain(child):
                        continue
                    parts.append(f"**{key}**: {format_inline(child, 220)}")
                if parts:
                    lines.append("- " + " | ".join(parts))
            elif isinstance(item, list):
                rendered = format_value(item, depth + 1)
                lines.extend(rendered.splitlines())
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)

    text = str(value)
    if len(text) > 100:
        return "> " + text.replace("\n", "\n> ")
    return text


def defined_fields(fields: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    categories: list[tuple[str, list[dict[str, Any]]]] = []
    for category in fields.get("categories", []):
        if not isinstance(category, dict):
            continue
        name = str(category.get("category") or "其他信息")
        specs = [item for item in category.get("fields", []) if isinstance(item, dict)]
        categories.append((name, specs))
    return categories


def collect_extra_fields(data: dict[str, Any], known: set[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    category_keys = {
        alias
        for canonical, aliases in CATEGORY_MAPPING.items()
        for alias in (canonical, *aliases)
    }
    for key, value in data.items():
        if key in INTERNAL_KEYS or key in known or key in category_keys:
            continue
        if not is_empty(value) and not contains_uncertain(value):
            extras[key] = value
    return extras


def executive_summary() -> str:
    return """## 执行摘要

深空图像边缘裁切不应被理解为“找到黑色像素后切掉”，而应被建模为可靠支持域判定：先利用配准变换、footprint、覆盖计数、权重和数据质量掩膜判断像素是否具有足够观测支持；只有缺少这些证据时，才退化到最终图像的近黑比例、亮度台阶、色偏和连通性检测。[Siril 的注册文档](https://siril.readthedocs.io/en/stable/preprocessing/registration.html)明确提供 `current/min/max/cog` framing，并说明 `min` 是所有图像的共同区域；[reproject](https://reproject.readthedocs.io/en/stable/footprints.html)则把 footprint 作为重投影的标准输出，甚至可以表达分数覆盖。

推荐把边缘处理分成两层：`geometric_mask` 表示像素是否落入输入视场，`reliable_mask` 再叠加最低覆盖、DQ/rejection、插值核腐蚀和下游支持域。矩形交付链路从可靠掩膜中求受目标 ROI 约束的最大安全矩形；内部空洞、坏列、低覆盖岛和马赛克非矩形边界继续用 mask/weight 表达，不应为了一个内部洞牺牲大面积视场。Drizzle 的权重和噪声还受 `pixfrac`、尺度、几何与抖动方式影响，因此不存在跨数据集通用的固定像素保护带。[STScI DrizzlePac](https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise)提供了这一点的直接依据。

裁切必须作为原子数据操作处理：SCI、WCS、mask、uncertainty、weight、coverage、context 和 rejection plane 同步裁切并校验；裁后还需为测光孔径、背景环、PSF、卷积、反卷积和神经网络保留各自安全边界。[Astropy CCDData](https://docs.astropy.org/en/stable/nddata/ccddata.html)展示了 mask/flags/uncertainty 与图像数据并存且裁切时保持 WCS 更新的数据模型；[Photutils](https://photutils.readthedocs.io/en/latest/api/photutils.aperture.decode_aperture_flags.html)则显式区分 no-overlap、partial-overlap、masked、non-finite 和 too-few-pixels 等测量失败状态。

### 推荐决策逻辑

1. 在注册或叠加阶段保存每帧变换及输出 `coverage/weight/DQ/rejection`；外部母版优先读取已有扩展或旁车。
2. 构造 `geometric_mask`，再按最低覆盖、异常标志、插值支持域和下游算子半径腐蚀为 `reliable_mask`。
3. 没有辅助证据时才启动像素 fallback，并要求近黑、亮度台阶、色偏、跨尺度一致性等至少两类软证据共同支持。
4. 根据产品目的选择 `common/min`、`reference/current` 或 `union/max + mask`；多通道合成使用所需通道可靠区的交集。
5. 在原始网格上求 ROI 约束的候选矩形，先模拟面积、角视场、目标/参考星和 WCS 损失，再决定是否提交。
6. 提交后同步裁切所有辅助平面、更新并验证 WCS；内部洞继续保留为 mask。
7. 输出 `science-safe / processing-safe / display-only / degraded / manual-review` 分类、剩余风险和完整 provenance。

### 对当前项目的结论

当前 Stage 2 位于背景建模之前，阶段位置正确，也已经具备四边非对称检测、迭代复检和累计裁切报告；Siril 官方同样警告场旋、dithering 或不完整重叠形成的黑边会污染自动背景模型，要求先清理边缘。[Siril 背景提取文档](https://siril.readthedocs.io/en/latest/processing/background.html)

主要缺口是 Stage 1 没有保留逐像素 coverage/weight，Stage 2 因而把 RGB 亮度扫描同时当作几何有效性和视觉伪影判据；同时缺少统一总裁切预算、目标 ROI、失败候选回滚、辅助平面/WCS 事务和续跑旁车恢复。建议按 P0/P1/P2 推进：P0 先修预算、回滚、配置接线、非有限值状态和 provenance；P1 再从 Stage 1 引入 coverage/weight 与可靠掩膜；P2 才扩展 common/reference/union+mask 以及分级产品。具体证据和实施项见后文“当前项目 Stage 2 差距与实施建议”。
"""


def build_report() -> str:
    outline = load_yaml(OUTLINE_PATH)
    fields = load_yaml(FIELDS_PATH)
    results = order_results(load_results(), outline)
    categories = defined_fields(fields)
    known = {str(spec.get("name")) for _, specs in categories for spec in specs}

    lines = [
        f"# {outline.get('topic', fields.get('topic', '深度调研报告'))}",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "调研口径：不限年代，优先软件/天文机构官方文档、同行评审论文及可验证实现。所有明确标为不确定的字段和值均不进入本报告正文，原始标记保留在 `results/*.json`。",
        "",
        executive_summary().rstrip(),
        "",
        "## 目录",
        "",
    ]

    for index, data in enumerate(results, start=1):
        name = str(data.get("name") or data.get("_source_file") or f"调研项 {index}")
        summary_parts = []
        for field_name in SUMMARY_FIELDS:
            value = get_field(data, "", field_name)
            if is_empty(value) or contains_uncertain(value):
                continue
            summary_parts.append(f"{FIELD_LABELS.get(field_name, field_name)}：{format_inline(value)}")
        suffix = " — " + "；".join(summary_parts) if summary_parts else ""
        lines.append(f"{index}. [{name}](#item-{index}){suffix}")

    for index, data in enumerate(results, start=1):
        name = str(data.get("name") or data.get("_source_file") or f"调研项 {index}")
        uncertain_names = {
            str(item).strip()
            for item in data.get("uncertain", [])
            if isinstance(item, str)
        }
        lines.extend(["", f'<a id="item-{index}"></a>', f"## {index}. {name}", ""])

        for category, specs in categories:
            rendered_fields: list[str] = []
            for spec in specs:
                field_name = str(spec.get("name") or "")
                if not field_name or field_name in uncertain_names:
                    continue
                value = get_field(data, category, field_name)
                if is_empty(value) or contains_uncertain(value):
                    continue
                if field_name == "name":
                    continue
                label = FIELD_LABELS.get(field_name, field_name)
                rendered = format_value(value)
                if rendered:
                    rendered_fields.extend([f"#### {label} (`{field_name}`)", "", rendered, ""])
            if rendered_fields:
                lines.extend([f"### {category}", "", *rendered_fields])

        extras = collect_extra_fields(data, known)
        if extras:
            lines.extend(["### 其他信息", "", format_value(extras), ""])

    lines.extend(
        [
            "",
            "## 调研文件",
            "",
            "- `outline.yaml`：调研范围、条目与执行配置。",
            "- `fields.yaml`：字段定义与不确定项口径。",
            "- `results/*.json`：逐项结构化结果及完整不确定标记。",
            "- `generate_report.py`：本报告的可复现生成脚本。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"generated {REPORT_PATH} ({len(report)} characters)")


if __name__ == "__main__":
    main()
