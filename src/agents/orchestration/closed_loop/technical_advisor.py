"""Read-only technical-advisor bridge for the closed optimization loop.

The bridge deliberately returns a small, JSON-safe summary.  It never returns
an Aspen action and it treats all advice as untrusted evidence for the primary
decision agent and the existing ``ActionPlan`` validator.
"""
from __future__ import annotations
from src.agents.errors import AgentUnavailableError, unavailable, fail_closed, failure_from
from src.agents.reasoning import configured, redact

from dataclasses import asdict, is_dataclass
from typing import Any

from src.agents.evidence.models import ProcessContext
from src.agents.subagents.technical_advisor import TechnicalAdvisor


TRIGGERS = {
    "initial_planning",
    "consecutive_failures",
    "constraint_bottleneck",
    "stagnation",
    "pareto_tradeoff_change",
    "user_question",
}


def advisor_trigger(state: dict, snapshot: dict, settings: Any, context: dict | None = None) -> str | None:
    """Return the highest-priority reason to consult the technical advisor."""
    context = context or {}
    if context.get("technical_advisor_question") or context.get("user_question"):
        return "user_question"
    if not state.get("observations"):
        return "initial_planning"
    tail = state.get("observations", [])[-max(1, int(getattr(settings, "failure_trigger", 3))):]
    if len(tail) >= max(1, int(getattr(settings, "failure_trigger", 3))) and all(not r.get("success") for r in tail):
        return "consecutive_failures"
    report = snapshot.get("analysis_report") or {}
    constraints = report.get("constraints") or {}
    if any(float(v.get("violation_rate", 0) or 0) > 0.5 for v in constraints.values() if isinstance(v, dict)):
        return "constraint_bottleneck"
    if int(snapshot.get("stagnation_count", 0) or 0) > 0:
        return "stagnation"
    pareto = report.get("pareto") or {}
    previous = (state.get("decisions") or [])[-1].get("before_analysis", {}).get("pareto", {}) if state.get("decisions") else {}
    if pareto.get("front_size", 0) and pareto.get("tradeoff_changed"):
        return "pareto_tradeoff_change"
    if previous and pareto.get("front_size") != previous.get("front_size"):
        return "pareto_tradeoff_change"
    return None


def _process_context(context: Any, snapshot: dict) -> ProcessContext:
    if isinstance(context, ProcessContext):
        return context
    raw = dict(context or {}) if isinstance(context, dict) else {}
    variables = raw.get("variables") or {}
    if not isinstance(variables, dict):
        variables = {
            str(item.get("name") or item.get("aspen_path")): dict(item)
            for item in (raw.get("design_variables") or [])
            if isinstance(item, dict) and (item.get("name") or item.get("aspen_path"))
        }
    return ProcessContext(
        model_id=str(raw.get("model_id") or raw.get("model") or "closed-loop"),
        equipment=list(raw.get("equipment") or []),
        streams=list(raw.get("streams") or []),
        topology=dict(raw.get("topology") or {}),
        variables=variables,
        objectives=list(raw.get("objectives") or []),
        constraints=list(raw.get("constraints") or []),
        current_values=dict(raw.get("current_values") or {}),
        diagnostics=list(raw.get("diagnostics") or []),
    )


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return {}


def consult(snapshot: dict, state: dict, context: dict | None, *, advisor: Any = None,
            settings: Any = None, trigger: str | None = None, max_chars: int = 2400) -> dict:
    """Call a read-only advisor and validate its evidence/probe references."""
    trigger = trigger or advisor_trigger(state, snapshot, settings or type("S", (), {"failure_trigger": 3})(), context)
    if trigger is None:
        return {"available": False, "trigger": None, "summary": None, "hypotheses": [], "ima_sources": []}
    known = {str(r.get("case_id")) for r in state.get("observations", [])}
    hard = set(snapshot.get("hard_bounds") or {})
    question = (context or {}).get("technical_advisor_question") if isinstance(context, dict) else None
    question = question or {
        "initial_planning": "请针对首次规划给出可验证的设备或物流优化假设和小步探测建议。",
        "consecutive_failures": "请诊断连续仿真失败，给出可验证假设、证据 case ID 和安全探测变量。",
        "constraint_bottleneck": "请解释当前约束瓶颈并建议受控探测变量。",
        "stagnation": "请解释优化停滞的可能设备或物流原因，并建议信息增益实验。",
        "pareto_tradeoff_change": "请解释 Pareto 多目标权衡变化，并建议验证该权衡的实验。",
        "user_question": "请回答用户关于当前闭环技术问题，并给出可验证假设。",
    }.get(trigger, "请为当前优化闭环提供可验证的技术假设。")
    try:
        result = (advisor or TechnicalAdvisor(None)).advise(question, _process_context(context, snapshot))
        raw = _as_dict(result)
        if raw.get("degraded"):
            raise unavailable("subagent_unavailable", stage="technical_advisor_legacy")
        advice = raw.get("advice") if isinstance(raw.get("advice"), dict) else raw
        if not isinstance(advice.get("conclusion"), str) or not advice["conclusion"].strip():
            raise unavailable("agent_output_invalid", stage="technical_advisor_summary")
        evidence_ids = set()
        for item in (raw.get("simulation_evidence") or []) + (advice.get("evidence") or []):
            item = _as_dict(item)
            ref = item.get("source_ref")
            if ref in known:
                evidence_ids.add(ref)
        # Adapter implementations may provide an explicit structured contract.
        hypotheses = raw.get("hypotheses") or advice.get("hypotheses") or []
        if not hypotheses:
            hypotheses = [{
                "id": f"ta-{trigger}",
                "statement": str(advice.get("conclusion") or "技术顾问未形成明确结论"),
                "evidence_case_ids": sorted(evidence_ids),
                "probe_variables": [],
                "risks": list(advice.get("risks") or []),
            }]
        clean = []
        for i, h in enumerate(hypotheses[:8]):
            h = _as_dict(h)
            if not isinstance(h.get("statement", h.get("hypothesis")), str):
                raise unavailable("agent_output_invalid", stage="technical_hypothesis")
            if not set(h.get("evidence_case_ids", [])) <= known or not set(h.get("probe_variables", h.get("suggested_probe_variables", []))) <= hard:
                raise unavailable("agent_output_invalid", stage="technical_references")
            refs = [x for x in h.get("evidence_case_ids", []) if x in known]
            probes = [x for x in h.get("probe_variables", h.get("suggested_probe_variables", [])) if x in hard]
            if not probes:
                suggestions = advice.get("suggested_probe_variables") or advice.get("suggested_experiments") or []
                probes = [x for x in suggestions if isinstance(x, str) and x in hard]
            clean.append({
                "id": str(h.get("id") or f"ta-{trigger}-{i}"),
                "statement": str(h.get("statement") or h.get("hypothesis") or advice.get("conclusion") or "" )[:800],
                "evidence_case_ids": refs,
                "probe_variables": probes,
                "risks": [str(x)[:300] for x in (h.get("risks") or advice.get("risks") or [])[:8]],
            })
        sources = []
        ima = raw.get("ima_answer") or {}
        for source in (ima.get("sources") or [])[:8]:
            source = _as_dict(source)
            sources.append({k: str(source.get(k))[:300] for k in ("source", "title", "chapter", "page", "quote") if source.get(k) is not None})
        summary = str(advice.get("conclusion") or "")[:max_chars]
        return {"available": True, "trigger": trigger,
                "summary": summary, "hypotheses": clean, "ima_sources": sources,
                "risks": [str(x)[:300] for x in (advice.get("risks") or [])[:8]],
                "query": str(raw.get("query") or "")[:500]}
    except Exception as exc:
        raise failure_from(exc, "closed_loop_technical_advisor") from None
