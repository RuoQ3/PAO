"""Fail-closed boundary recommendations with hard-bound validation."""
from __future__ import annotations
from src.agents.errors import AgentUnavailableError, unavailable, fail_closed, failure_from
from src.agents.reasoning import configured, redact

import logging
from dataclasses import dataclass

from src.agents.subagents.boundary_advisor.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.agents.subagents.boundary_advisor.tools import (
    VarMeta,
    bounds_from_k,
    build_variables_block,
)

# 模块级 import：支持测试 monkeypatch
from src.agents.lead.llm_client import chat, is_configured, load_llm_config  # noqa: E402

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRecommendation:
    """单个变量的边界推荐结果。"""
    name: str
    lower: float
    upper: float
    k_lo: float
    k_hi: float
    source: str          # verified LLM
    reason: str


@dataclass
class BoundaryReport:
    """boundary_advisor 的完整输出。"""
    recommendations: list[BoundaryRecommendation]
    used_llm: bool
    global_notes: str
    warnings: list[str]

    def as_bounds_dict(self) -> dict[str, tuple[float, float]]:
        """返回 {name: (lower, upper)},便于直接覆盖 param_bounds。"""
        return {r.name: (r.lower, r.upper) for r in self.recommendations}


# ---------------------------------------------------------------------------
# 纯规则层（不依赖 LLM）
# ---------------------------------------------------------------------------

def recommend_boundaries(variables: list[VarMeta]) -> BoundaryReport:
    """Legacy entry uses the same verified provider path."""
    return recommend_boundaries_agent(variables)


@fail_closed("boundary_advisor")
def recommend_boundaries_agent(variables, context="", model=None, provider=None, llm_config=None):
    import math
    cfg = configured(llm_config or load_llm_config(provider=provider, model=model))
    if not variables:
        raise unavailable("agent_output_invalid", stage="boundary_input")
    raw = chat(cfg, system=SYSTEM_PROMPT, user=USER_TEMPLATE.format(
        context=context or "(no context)", variables_block=build_variables_block(variables)))
    import json
    parsed = redact(json.loads(raw), cfg)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("variables"), list):
        raise unavailable("llm_schema_invalid", stage="boundary_output")
    rows = parsed["variables"]
    if not all(isinstance(r, dict) for r in rows) or not isinstance(parsed.get("global_notes"), str):
        raise unavailable("llm_schema_invalid", stage="boundary_output")
    if len(rows) != len(variables) or {r.get("name") for r in rows} != {v.name for v in variables}:
        raise unavailable("agent_output_invalid", stage="boundary_coverage")
    by_name = {r["name"]: r for r in rows}
    recs = []
    for v in variables:
        r = by_name[v.name]
        lo, hi = r["k_lo"], r["k_hi"]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (lo, hi)) or not isinstance(r.get("reason"), str) or not r["reason"].strip():
            raise unavailable("llm_schema_invalid", stage="boundary_output")
        if v.initial_value is None or v.initial_value <= 0:
            raise unavailable("agent_output_invalid", stage="boundary_evidence")
        bounds = bounds_from_k(v, lo, hi)
        if bounds is None or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise unavailable("agent_output_invalid", stage="boundary_validation")
        recs.append(BoundaryRecommendation(v.name, *bounds, lo, hi, "llm", r["reason"]))
    return BoundaryReport(recs, True, parsed["global_notes"], [])


def format_boundary_report(report: BoundaryReport) -> str:
    """把 BoundaryReport 渲染成可读文本,供日志/CLI 展示。"""
    lines = ["=== BoundaryAdvisor 边界推荐报告 ===", ""]
    lines.append(f"使用大模型：{'是' if report.used_llm else '否'}")
    if report.global_notes:
        lines.append(f"整体判断：{report.global_notes}")
    lines.append("")
    lines.append(f"{'变量':<28}{'下界':>14}{'上界':>14}  来源     说明")
    for r in report.recommendations:
        short = r.name.replace("\\", "/").split("/")[-1][:26]
        lines.append(
            f"{short:<28}{r.lower:>14.4g}{r.upper:>14.4g}  {r.source:<8} {r.reason}"
        )
    if report.warnings:
        lines.append("")
        lines.append("警告：")
        for w in report.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
