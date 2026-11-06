"""Persistent conversation orchestration for the Coordinator.

This layer plans and combines advisor work.  It deliberately has no Aspen,
COM, database write, or simulation execution capability.  Mutating requests
become :class:`UserAction` previews and must be handled by a later approval
flow.
"""
from __future__ import annotations

import inspect
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.agents.contracts.models import (
    ActionKind,
    AdviceUncertainty,
    EvidenceBundle,
    EvidenceItem,
    EvidenceSource,
    ImaAnswer,
    ProcessContext,
    TechnicalAdvice,
    UserAction,
)
from src.agents.process_context import ProcessContextBuilder
from src.agents.technical_advisor import TechnicalAdvisor
from src.models.coordinator import IntentClassification, IntentStep, IntentType


@dataclass(frozen=True)
class RouteSpec:
    required_context: tuple[str, ...]
    consultants: tuple[str, ...]
    mode: str = "answer"


ROUTE_TABLE: dict[IntentType, RouteSpec] = {
    IntentType.EQUIPMENT_EXPLANATION: RouteSpec(("process_context",), ("technical_advisor", "ima")),
    IntentType.STREAM_TRACE: RouteSpec(("process_context",), ("technical_advisor", "ima")),
    IntentType.DIAGNOSIS: RouteSpec(("process_context",), ("process_advisor", "technical_advisor", "simulation")),
    IntentType.DIAGNOSE_RESULTS: RouteSpec(("process_context",), ("process_advisor", "technical_advisor", "simulation")),
    IntentType.CASE_COMPARISON: RouteSpec(("process_context",), ("technical_advisor", "process_advisor", "optimization_analysis")),
    IntentType.BOUNDARY_ADVICE: RouteSpec(("process_context",), ("technical_advisor", "boundary_advisor")),
    IntentType.RECOMMEND_VAR_BOUNDS: RouteSpec(("process_context",), ("technical_advisor", "boundary_advisor")),
    IntentType.OPTIMIZATION_PLANNING: RouteSpec(("process_context",), ("technical_advisor", "process_advisor", "optimization_analysis")),
    IntentType.PARAMETER_MODIFICATION: RouteSpec(("process_context",), ("technical_advisor", "action_gateway"), "action"),
    IntentType.SINGLE_SIMULATION: RouteSpec(("process_context",), ("technical_advisor", "action_gateway"), "action"),
    IntentType.OPTIMIZATION_RUN: RouteSpec(("process_context",), ("process_advisor", "optimization_analysis", "action_gateway"), "action"),
    IntentType.RUN_OPTIMIZATION: RouteSpec(("process_context",), ("process_advisor", "optimization_analysis", "action_gateway"), "action"),
    IntentType.GENERAL_CHAT: RouteSpec((), ("general_chat",)),
}


@dataclass
class OrchestrationResult:
    classification: IntentClassification
    route: list[RouteSpec] = field(default_factory=list)
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)
    answer: str = ""
    answer_sections: dict[str, Any] = field(default_factory=dict)
    actions: list[UserAction] = field(default_factory=list)
    clarifications: list[str] = field(default_factory=list)
    call_trace: list[str] = field(default_factory=list)
    task_results: dict[str, Any] = field(default_factory=dict)


def _intent_alias(step: IntentStep) -> IntentType:
    """Map legacy intent names to the persistent task vocabulary."""
    aliases = {
        IntentType.DIAGNOSE_RESULTS: IntentType.DIAGNOSIS,
        IntentType.RECOMMEND_VAR_BOUNDS: IntentType.BOUNDARY_ADVICE,
        IntentType.RUN_OPTIMIZATION: IntentType.OPTIMIZATION_RUN,
    }
    return aliases.get(step.intent_type, step.intent_type)


def _call(fn: Callable[..., Any], *, question: str, context: ProcessContext | None, step: IntentStep) -> Any:
    """Call injected mocks and adapters with a small, forgiving protocol."""
    try:
        sig = inspect.signature(fn)
        kwargs = {}
        for name in sig.parameters:
            if name in {"question", "user_text", "text"}:
                kwargs[name] = question
            elif name in {"context", "process_context"}:
                kwargs[name] = context
            elif name in {"step", "intent"}:
                kwargs[name] = step
        if kwargs:
            return fn(**kwargs)
    except (TypeError, ValueError):
        pass
    for args in ((question, context), (question,), ()):
        try:
            return fn(*args)
        except TypeError:
            continue
    return None


def _as_evidence(value: Any, source: EvidenceSource, prefix: str) -> tuple[list[EvidenceItem], list[TechnicalAdvice], list[str]]:
    facts: list[EvidenceItem] = []
    advice: list[TechnicalAdvice] = []
    warnings: list[str] = []
    if value is None:
        return facts, advice, warnings
    if isinstance(value, ImaAnswer):
        if value.answer:
            facts.append(EvidenceItem(f"{prefix}.answer", EvidenceSource.IMA, value.answer, value.source or value.request_id, confidence=value.confidence))
        facts.extend(value.evidence)
        return facts, advice, warnings
    if isinstance(value, TechnicalAdvice):
        advice.append(value)
        facts.extend(value.evidence)
        return facts, advice, warnings
    # Read-only specialist agents return an explicit evidence-first contract.
    if hasattr(value, "evidence"):
        facts.extend(x for x in getattr(value, "evidence", []) if isinstance(x, EvidenceItem))
        advice.extend(x for x in getattr(value, "technical_advice", []) if isinstance(x, TechnicalAdvice))
        warnings.extend(str(x) for x in getattr(value, "missing", []) if x)
        warnings.extend(str(x) for x in getattr(value, "uncertainty", []) if x)
        return facts, advice, warnings
    if isinstance(value, EvidenceItem):
        facts.append(value)
        return facts, advice, warnings
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            f, a, w = _as_evidence(item, source, f"{prefix}.{i}")
            facts.extend(f); advice.extend(a); warnings.extend(w)
        return facts, advice, warnings
    if isinstance(value, Mapping):
        if "advice" in value:
            return _as_evidence(value["advice"], source, prefix)
        claim = str(value.get("claim", value.get("answer", value.get("report", value))))
        facts.append(EvidenceItem(f"{prefix}.{len(facts)}", source, claim, value=value))
        return facts, advice, warnings
    facts.append(EvidenceItem(f"{prefix}.0", source, str(value), value=value))
    return facts, advice, warnings


def merge_evidence(*, context: ProcessContext | None, consultant_results: Mapping[str, Any]) -> EvidenceBundle:
    """Merge all sources while retaining provenance and recording conflicts."""
    bundle = EvidenceBundle(bundle_id=str(uuid.uuid4()), complete=True)
    if context is not None:
        bundle.local_process_facts.extend(context.sources)
        bundle.simulation_evidence.extend(context.history_evidence)
        bundle.conflicts.extend(context.diagnostics)
    claims: dict[str, list[EvidenceItem]] = {}
    for name, result in consultant_results.items():
        source = EvidenceSource.IMA if name == "ima" else EvidenceSource.SIMULATION if name in {"simulation", "optimization_analysis"} else EvidenceSource.LOCAL_PROCESS
        facts, advice, warnings = _as_evidence(result, source, name)
        if source == EvidenceSource.IMA and isinstance(result, ImaAnswer):
            bundle.ima.append(result)
        bundle.technical_advice.extend(advice)
        for fact in facts:
            if fact.source in {EvidenceSource.SIMULATION, EvidenceSource.SIMULATION_DB}:
                bundle.simulation_evidence.append(fact)
            else:
                # Keep IMA citations, NodeDB snapshots and derived optimizer
                # facts in the local packet while preserving each source tag.
                bundle.local_process_facts.append(fact)
        for fact in facts:
            key = re.sub(r"\s+", " ", fact.claim.lower()).strip()
            if key:
                claims.setdefault(key, []).append(fact)
        bundle.missing.extend(warnings)
    for key, items in claims.items():
        values = {repr(item.value) for item in items if item.value is not None}
        if len(values) > 1:
            bundle.conflicts.append("证据冲突：" + key + "；来源=" + ", ".join(sorted({str(x.source) for x in items})))
    bundle.complete = not bool(bundle.missing)
    return bundle


def _make_action(intent: IntentType, step: IntentStep, question: str, evidence: EvidenceBundle) -> UserAction:
    kind = {
        IntentType.PARAMETER_MODIFICATION: ActionKind.SET_PARAMETER,
        IntentType.SINGLE_SIMULATION: ActionKind.SIMULATE_ONCE,
        IntentType.OPTIMIZATION_RUN: ActionKind.OPTIMIZE,
        IntentType.RUN_OPTIMIZATION: ActionKind.OPTIMIZE,
    }[intent]
    target = step.raw_slots.get("target") or step.raw_slots.get("parameter") or step.raw_slots.get("path") or ""
    if not target:
        target = "待用户指定目标"
    new_value = step.raw_slots.get("value") or step.raw_slots.get("new_value")
    risks = ["需要用户确认后才可提交受控网关", "Coordinator 不会直接执行 Aspen 操作"]
    return UserAction(action_type=kind, target=target, new_value=new_value, reason=question, risks=risks, confirmation_status="required", evidence_ids=[x.id for x in evidence.local_process_facts[:5]])


def _format_answer(bundle: EvidenceBundle, actions: list[UserAction], clarifications: list[str]) -> tuple[str, dict[str, Any]]:
    facts = [x.claim for x in bundle.local_process_facts[:8]]
    simulation = [x.claim for x in bundle.simulation_evidence[:8]]
    conclusions = [a.conclusion for a in bundle.technical_advice if a.conclusion]
    uncertainty = "高" if bundle.conflicts or bundle.missing else "中"
    sections = {
        "结论": conclusions[0] if conclusions else ("需要补充信息后才能可靠回答。" if clarifications else "已汇总可用证据。"),
        "事实": facts,
        "技术依据": [a.mechanism for a in bundle.technical_advice if a.mechanism],
        "仿真证据": simulation,
        "假设": [a for adv in bundle.technical_advice for a in adv.assumptions],
        "不确定性": uncertainty,
        "下一步": clarifications or (["审批后再提交动作"] if actions else ["如需更高置信度，请补充流程上下文或历史工况"]),
        "证据来源": sorted({str(x.source) for x in bundle.local_process_facts + bundle.simulation_evidence}),
        "冲突": list(bundle.conflicts),
    }
    text = "\n".join(f"{k}：{v}" for k, v in sections.items())
    return text, sections


def orchestrate_turn(
    user_text: str,
    *,
    classification: IntentClassification | None = None,
    process_context: ProcessContext | None = None,
    consultants: Mapping[str, Callable[..., Any]] | None = None,
    session: Any = None,
    model: str | None = None,
    provider: str | None = None,
    llm_config: Any = None,
    dynamic_plan: Any = None,
) -> OrchestrationResult:
    """Route one conversational turn to one or more read-only consultants."""
    from src.agents.coordinator.intent import classify_intent_agent

    if dynamic_plan is not None:
        from src.agents.conversation.dynamic import ALLOWED_AGENTS

        classification = IntentClassification(
            steps=[IntentStep(IntentType.GENERAL_CHAT, reason=f"dynamic plan: {dynamic_plan.intent}")],
            notes=f"dynamic intent={dynamic_plan.intent}",
            used_llm=bool(dynamic_plan.used_llm),
        )
        registry = dict(consultants or {})
        routes: list[RouteSpec] = []
        task_results: dict[str, Any] = {}
        trace: list[str] = []
        clarifications: list[str] = []
        if dynamic_plan.needs_clarification:
            clarifications.append(dynamic_plan.clarification or "请补充最少必要的设备、物流、参数或目标信息。")
        for task in ([] if dynamic_plan.needs_clarification else dynamic_plan.tasks):
            agent_name = getattr(task, "agent", None)
            task_id = str(getattr(task, "id", ""))
            # This check is deliberately repeated at the execution boundary;
            # the planner's output is data, never authority to import/call code.
            if agent_name not in ALLOWED_AGENTS:
                trace.append(f"{task_id}:{agent_name or 'unknown'}:rejected_unknown_agent")
                clarifications.append(f"已拒绝未授权的 Agent：{agent_name!r}。")
                continue
            if any(dep not in task_results or task_results[dep] is None for dep in task.depends_on):
                trace.append(f"{task_id}:{agent_name}:skipped_dependency")
                task_results[task_id] = None
                continue
            fn = registry.get(agent_name)
            if fn is None:
                trace.append(f"{task_id}:{agent_name}:unavailable")
                task_results[task_id] = None
                clarifications.append(f"{agent_name} 当前不可用。")
                continue
            route = RouteSpec(("process_context",), (agent_name,))
            routes.append(route)
            result = _call_planned_task(
                fn, question=user_text, context=process_context, task=task,
                entities=getattr(task, "entities", {}) or getattr(dynamic_plan, "entities", {}),
            )
            task_results[task_id] = result
            trace.append(f"{task_id}:{agent_name}:local_agent")
        consultant_results = {
            f"task:{task_id}": result for task_id, result in task_results.items() if result is not None
        }
        # The specialist agents return scoped evidence. Avoid merging the whole
        # ProcessContext here, which could leak unrelated equipment variables.
        bundle = merge_evidence(context=None, consultant_results=consultant_results)
        if process_context is None:
            bundle.missing.append("缺少 ProcessContext；未凭空填写设备、路径或参数。")
        if clarifications:
            bundle.missing.extend(clarifications)
        bundle.complete = not bool(bundle.missing)
        answer, sections = _format_answer(bundle, [], clarifications)
        return OrchestrationResult(
            classification=classification, route=routes, evidence=bundle,
            answer=answer, answer_sections=sections, clarifications=clarifications,
            call_trace=trace, task_results=task_results,
        )

    classification = classification or classify_intent_agent(user_text, model=model, provider=provider, llm_config=llm_config)
    registry = dict(consultants or {})
    if "technical_advisor" not in registry and process_context is not None:
        registry["technical_advisor"] = TechnicalAdvisor().advise
    routes: list[RouteSpec] = []
    all_results: dict[str, Any] = {}
    actions: list[UserAction] = []
    clarifications: list[str] = []
    trace: list[str] = []
    for step in classification.steps:
        intent = _intent_alias(step)
        if intent == IntentType.CLARIFY_NEEDED:
            clarifications.append(step.reason or "请具体说明要回答的问题或要预览的动作。")
            continue
        route = ROUTE_TABLE.get(intent, RouteSpec(("process_context",), ("technical_advisor",)))
        routes.append(route)
        if route.mode == "action":
            actions.append(_make_action(intent, step, user_text, EvidenceBundle()))
            if not (step.raw_slots.get("target") or step.raw_slots.get("parameter") or step.raw_slots.get("path")):
                clarifications.append("请提供要修改/仿真的 Aspen 参数路径或目标名称，以及期望值或运行范围。")
        for name in route.consultants:
            fn = registry.get(name)
            if fn is None:
                if name == "ima":
                    continue
                trace.append(f"{intent.value}:{name}:unavailable")
                continue
            result = _call(fn, question=user_text, context=process_context, step=step)
            all_results[f"{name}:{len(trace)}"] = result
            trace.append(f"{intent.value}:{name}")
    bundle = merge_evidence(context=process_context, consultant_results=all_results)
    if clarifications:
        bundle.missing.extend(clarifications)
        bundle.complete = False
    # Attach evidence ids to actions after the bundle is known.
    for action in actions:
        action.evidence_ids = [x.id for x in bundle.local_process_facts[:5]]
    answer, sections = _format_answer(bundle, actions, clarifications)
    if not process_context and any("process_context" in r.required_context for r in routes):
        bundle.missing.append("缺少 ProcessContext；未凭空填写设备、路径或参数。")
        bundle.complete = False
        clarifications.append("请提供可读取的流程上下文（设备/物流/配置或历史结果），再继续分析。")
        answer, sections = _format_answer(bundle, actions, clarifications)
    return OrchestrationResult(classification, routes, bundle, answer, sections, actions, clarifications, trace)


def _call_planned_task(
    fn: Callable[..., Any], *, question: str, context: ProcessContext | None,
    task: Any, entities: dict[str, list[str]],
) -> Any:
    """Invoke a registered read-only agent without exposing arbitrary kwargs."""
    try:
        sig = inspect.signature(fn)
        available = {
            "question": question, "user_text": question, "text": question,
            "context": context, "process_context": context,
            "task": task, "entities": entities,
        }
        kwargs = {name: available[name] for name in sig.parameters if name in available}
        if kwargs:
            return fn(**kwargs)
    except (TypeError, ValueError):
        pass
    for args in ((question, context, task), (question, context), (question,), ()):
        try:
            return fn(*args)
        except TypeError:
            continue
    return None


run_orchestrator = orchestrate_turn
