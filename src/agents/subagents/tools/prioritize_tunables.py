"""LLM-only complete variable prioritization; no inferred scores."""
from __future__ import annotations
from src.agents.errors import AgentUnavailableError, unavailable, fail_closed, failure_from
from src.agents.reasoning import configured, redact

import json
import logging
import re
from copy import deepcopy
from typing import Any

from src.models.tunable import OptimizationIntent, PrioritizationResult, TunableReport, TunableVariable

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
你是化工流程优化专家，负责评估设计变量对优化目标的重要性。

评分规则：
- priority_score 取值范围 0.1（几乎无影响）~ 1.0（核心驱动因子）
- 直接影响目标函数（如能耗、产量、成本）的变量评分高
- 影响路径长（需经过多个中间环节才影响目标）的变量评分低
- 与约束条件强相关的变量适当提高评分
- priority_reason 用中文简短说明（20字以内）

只返回 JSON，不要任何解释文字，格式严格如下：
{
  "prioritization": [
    {"aspen_path": "...", "priority_score": 0.9, "priority_reason": "..."},
    ...
  ],
  "ranking_notes": "一句话总结排序依据"
}
"""


def _build_user_prompt(
    variables: list[TunableVariable],
    intent: OptimizationIntent,
) -> str:
    """构建发给 LLM 的 user prompt，只包含 high/medium 变量。"""
    lines = ["优化目标："]
    for g in intent.goals:
        direction = "最小化" if g.direction == "min" else "最大化"
        name = g.name or g.metric
        lines.append(f"- {direction} {name}")
    if intent.hard_constraints:
        lines.append("硬约束：")
        for c in intent.hard_constraints:
            lines.append(f"- {c.name or c.metric} {c.direction} {c.target_value or ''}")

    lines.append(f"\n设计变量候选（共 {len(variables)} 个）：")
    for i, v in enumerate(variables, 1):
        role = v.semantic_role or "未知角色"
        val = f"{v.current_value:.4g}" if v.current_value is not None else "N/A"
        bounds = (
            f"[{v.suggested_lower}, {v.suggested_upper}]"
            if v.suggested_lower is not None and v.suggested_upper is not None
            else "边界未知"
        )
        lines.append(
            f"{i}. {role} | {v.aspen_path} | 当前值={val} | 范围={bounds} | confidence={v.confidence}"
        )

    lines.append("\n请返回上述变量的优先级 JSON。")
    return "\n".join(lines)


@fail_closed("prioritize_tunables")
def prioritize_tunables_impl(report, intent, llm_config=None):
    import math
    from src.agents.lead.llm_client import chat
    cfg = configured(llm_config)
    variables = report.tunable_variables
    if not variables:
        raise unavailable("agent_output_invalid", stage="prioritize_input")
    data = json.loads(chat(cfg, system=_SYSTEM_PROMPT, user=_build_user_prompt(variables, intent)))
    data = redact(data, cfg)
    rows = data["prioritization"]
    paths = {v.aspen_path for v in variables}
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows) or len(rows) != len(paths) or {r["aspen_path"] for r in rows} != paths:
        raise unavailable("agent_output_invalid", stage="prioritize_coverage")
    if not isinstance(data.get("ranking_notes"), str) or not data["ranking_notes"].strip():
        raise unavailable("llm_schema_invalid", stage="prioritize_output")
    by_path = {r["aspen_path"]: r for r in rows}
    ranked = []
    for variable in variables:
        row = by_path[variable.aspen_path]
        score, reason = row["priority_score"], row["priority_reason"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1 or not isinstance(reason, str) or not reason.strip():
            raise unavailable("llm_schema_invalid", stage="prioritize_output")
        item = deepcopy(variable)
        item.priority_score, item.priority_reason = score, reason
        ranked.append(item)
    ranked.sort(key=lambda v: v.priority_score, reverse=True)
    return PrioritizationResult(ranked_variables=ranked, ranking_notes=data["ranking_notes"], warnings=[], source="llm")
