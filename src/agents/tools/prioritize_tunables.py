"""
prioritize_tunables.py — LLM 变量优先级排序工具

职责
----
根据用户的优化意图（OptimizationIntent）和已发现的设计变量列表（TunableReport），
调用 LLM 为每个变量评估对目标函数的重要性，产出 PrioritizationResult。

降级策略
--------
LLM 未配置 / 调用失败 / 返回无法解析的 JSON 时，自动按 confidence 映射 priority_score：
  high → 0.8, medium → 0.5, low → 0.2
不抛异常，不阻断 onboarding 流程。

Token 控制
----------
只把 confidence=high/medium 的变量发给 LLM（通常 5-20 个），
confidence=low 的变量数量巨大（可能几千个），直接赋 priority_score=0.2，
不参与 LLM 分析，以避免 prompt 过长。
"""
from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

from src.models.tunable import OptimizationIntent, PrioritizationResult, TunableReport, TunableVariable

_log = logging.getLogger(__name__)

# confidence → 降级 priority_score 映射
_FALLBACK_SCORES: dict[str, float] = {
    "high":   0.8,
    "medium": 0.5,
    "low":    0.2,
}

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


def _fallback_result(
    report: TunableReport,
    reason: str,
) -> PrioritizationResult:
    """按 confidence 映射 priority_score，返回降级结果。"""
    ranked = []
    for v in report.tunable_variables:
        v2 = deepcopy(v)
        v2.priority_score = _FALLBACK_SCORES.get(v.confidence, 0.2)
        v2.priority_reason = ""
        ranked.append(v2)
    ranked.sort(key=lambda v: v.priority_score, reverse=True)
    return PrioritizationResult(
        ranked_variables=ranked,
        ranking_notes="",
        warnings=[reason],
        source="fallback",
    )


def _parse_llm_response(
    raw: str,
    valid_paths: set[str],
    all_vars: list[TunableVariable],
) -> tuple[dict[str, tuple[float, str]], str]:
    """
    解析 LLM 返回的 JSON，校验 aspen_path，返回：
    ({aspen_path: (score, reason)}, ranking_notes)
    """
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()

    data = json.loads(text)
    scores: dict[str, tuple[float, str]] = {}
    for item in data.get("prioritization", []):
        path = item.get("aspen_path", "")
        if path not in valid_paths:
            continue   # 防止 LLM 幻觉出不存在的路径
        score = float(item.get("priority_score", 0.5))
        score = max(0.0, min(1.0, score))   # 钳位到 [0, 1]
        reason = str(item.get("priority_reason", ""))[:60]
        scores[path] = (score, reason)

    return scores, str(data.get("ranking_notes", ""))


def prioritize_tunables_impl(
    report: TunableReport,
    intent: OptimizationIntent,
    llm_config: Any = None,
) -> PrioritizationResult:
    """
    对 TunableReport 中的设计变量按优化目标重要性排序。

    Parameters
    ----------
    report:
        discover_tunables_tool 产出的变量报告。
    intent:
        用户优化意图（目标函数 + 约束）。
    llm_config:
        LLMConfig 实例；None 时尝试从环境变量加载；
        未配置 / 调用失败时自动降级。

    Returns
    -------
    PrioritizationResult
        ranked_variables 已按 priority_score 降序排列。
    """
    from src.agents.llm_client import chat, is_configured, load_llm_config

    all_vars = report.tunable_variables
    if not all_vars:
        return PrioritizationResult(
            ranked_variables=[],
            ranking_notes="",
            warnings=["无设计变量，跳过优先级排序"],
            source="fallback",
        )

    # 只把 high/medium 变量发给 LLM，low 直接降级赋分
    hm_vars = [v for v in all_vars if v.confidence in ("high", "medium")]
    low_vars = [v for v in all_vars if v.confidence not in ("high", "medium")]

    # 尝试加载 LLM 配置
    try:
        cfg = llm_config if llm_config is not None else load_llm_config()
    except Exception as exc:
        return _fallback_result(report, f"LLM 配置加载失败：{exc}")

    if not is_configured(cfg):
        return _fallback_result(report, "LLM 未配置（无 API key），已按 confidence 降级排序")

    if not hm_vars:
        return _fallback_result(report, "无 high/medium 变量可发给 LLM，已按 confidence 降级排序")

    # 调用 LLM
    user_prompt = _build_user_prompt(hm_vars, intent)
    _log.info("prioritize_tunables: 发送 %d 个变量给 LLM 分析（%s）", len(hm_vars), cfg.model)

    try:
        raw = chat(cfg, system=_SYSTEM_PROMPT, user=user_prompt)
    except Exception as exc:
        _log.warning("prioritize_tunables: LLM 调用失败，降级排序：%s", exc)
        return _fallback_result(report, f"LLM 调用失败：{exc}")

    # 解析 JSON
    valid_paths = {v.aspen_path for v in hm_vars}
    try:
        scores, ranking_notes = _parse_llm_response(raw, valid_paths, hm_vars)
    except Exception as exc:
        _log.warning("prioritize_tunables: LLM 返回无法解析，降级排序：%s | raw=%s", exc, raw[:200])
        return _fallback_result(report, f"LLM 返回格式无效：{exc}")

    # 将 LLM 分数写回变量（未被 LLM 返回的 hm 变量用 confidence 降级）
    ranked: list[TunableVariable] = []
    for v in hm_vars:
        v2 = deepcopy(v)
        if v.aspen_path in scores:
            v2.priority_score, v2.priority_reason = scores[v.aspen_path]
        else:
            v2.priority_score = _FALLBACK_SCORES.get(v.confidence, 0.5)
            v2.priority_reason = ""
        ranked.append(v2)

    # low 变量统一赋低分
    for v in low_vars:
        v2 = deepcopy(v)
        v2.priority_score = 0.2
        v2.priority_reason = ""
        ranked.append(v2)

    ranked.sort(key=lambda v: v.priority_score, reverse=True)

    _log.info(
        "prioritize_tunables: 完成，top3=%s",
        [f"{v.aspen_path}({v.priority_score:.2f})" for v in ranked[:3]],
    )

    warnings: list[str] = []
    missing = [p for p in valid_paths if p not in scores]
    if missing:
        warnings.append(f"LLM 未返回 {len(missing)} 个变量的评分，已按 confidence 补全")

    return PrioritizationResult(
        ranked_variables=ranked,
        ranking_notes=ranking_notes,
        warnings=warnings,
        source="llm",
    )
