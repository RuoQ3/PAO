"""LLM planning and deterministic fallback for read-only case conversations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from src.agents.contracts.models import ProcessContext
from src.agents.llm_client import chat, is_configured, load_llm_config


ALLOWED_AGENTS = frozenset({
    "CaseOverviewAgent", "EquipmentAgent", "TopologyAgent",
    "ParameterImpactAgent", "SimulationDiagnosticAgent", "OptimizationAgent",
    "TechnicalAdvisorAgent",
})
ALLOWED_SOURCES = frozenset({
    "yaml", "aspen_model", "nodedb", "simulation_db",
    "optimization_results", "ima",
})
ALLOWED_INTENTS = frozenset({
    "case_overview", "equipment_parameters", "equipment_explanation",
    "topology_analysis", "parameter_impact", "result_reliability_diagnosis",
    "simulation_diagnosis", "optimization_analysis", "case_comparison",
    "technical_question", "comprehensive_analysis", "equipment_comparison", "clarification",
})

PLANNER_SYSTEM_PROMPT = """你是 PAO 案例对话的总控规划器。你只负责理解问题并返回 JSON 任务计划，不回答技术问题，不执行工具。
只能引用 user payload 中 entities.allowed 的规范实体名。不得创造设备、物流、变量、目标或约束。
只可使用 allowed_agents 中的 Agent；任务 sources 只能从 allowed_sources 选取。
开放式案例概览应调用 CaseOverviewAgent；拓扑问题调用 TopologyAgent；参数影响调用 ParameterImpactAgent；结果可信度问题应拆成模拟/历史与优化可行性任务；历史比较调用 OptimizationAgent。
需要时将多个问题拆成多个任务，并用 depends_on 表达依赖。优先用 history 解析“它/这个塔/刚才的结果”。
无法确定对象且现有上下文也不能消歧时才要求澄清。计划必须严格符合 JSON schema，不要代码围栏或附加文字。

JSON schema:
{
  "intent": "one allowed intent",
  "entities": {"equipment": [], "streams": [], "variables": [], "objectives": [], "constraints": []},
  "tasks": [{"id": "task_1", "agent": "allowed agent", "objective": "read-only task", "sources": [], "depends_on": [], "entities": {"equipment": []}}],
  "needs_clarification": false,
  "clarification": "",
  "confidence": 0.0
}
"""

PLANNER_SYSTEM_PROMPT += """
Understand the original natural expression, not a question menu. Internal intent is
only a routing hint. For a unit's role, inspect EquipmentAgent and TopologyAgent;
inspect linked objectives, stream composition and history when available. Model
type alone does not prove process duty. Comparison requires separate unit facts
and verified relationships. Tasks may include entities (a subset of plan entities).
Do not silently choose one referent from multiple prior equipment candidates.
Use case equipment/streams/variables/objectives/constraints as the allowed names.
For comparisons, create separate one-equipment EquipmentAgent tasks and a
TopologyAgent task for verifiable relation evidence.
"""


def scope_plan_tasks(plan: "DynamicPlan") -> "DynamicPlan":
    """Assign explicit inputs and isolate equipment specialists before execution."""
    tasks = []
    replacements: dict[str, list[str]] = {}
    for task in plan.tasks:
        scope = {key: list(values) for key, values in plan.entities.items()}
        scope.update(task.entities)
        equipment = scope.get("equipment", [])
        units = equipment if task.agent in {"EquipmentAgent", "ParameterImpactAgent", "TechnicalAdvisorAgent"} and equipment else [None]
        ids = []
        for index, unit in enumerate(units):
            task_scope = {key: list(values) for key, values in scope.items()}
            if unit:
                task_scope["equipment"] = [unit]
            task_id = task.id if len(units) == 1 else f"{task.id}.{index + 1}"
            deps = [expanded for dep in task.depends_on for expanded in replacements.get(dep, [dep])]
            tasks.append(replace(task, id=task_id, entities=task_scope, depends_on=deps))
            ids.append(task_id)
        replacements[task.id] = ids
    plan.tasks = tasks
    return plan


@dataclass
class DynamicTask:
    id: str
    agent: str
    objective: str
    sources: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: str = "planned"
    result: Any = None
    entities: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DynamicPlan:
    intent: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    tasks: list[DynamicTask] = field(default_factory=list)
    needs_clarification: bool = False
    clarification: str = ""
    confidence: float = 0.0
    used_llm: bool = False
    planner_mode: str = "local_rules"
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "tasks": [t.__dict__.copy() for t in self.tasks],
            "needs_clarification": self.needs_clarification,
            "clarification": self.clarification,
            "confidence": self.confidence,
            "used_llm": self.used_llm,
            "planner_mode": self.planner_mode,
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class DynamicAnswer:
    plan: DynamicPlan
    answer: str
    handled: bool = True
    evidence_ids: list[str] = field(default_factory=list)


def _contains(text: str, *words: str) -> bool:
    low = text.casefold()
    return any(w.casefold() in low for w in words)


def _known_entities(context: ProcessContext) -> dict[str, list[str]]:
    return {
        "equipment": list(dict.fromkeys(str(x.get("name")) for x in context.equipment if x.get("name"))),
        "streams": list(dict.fromkeys(str(x.get("name")) for x in context.streams if x.get("name"))),
        "variables": list(dict.fromkeys(str(v.get("name") or key) for key, v in context.variables.items())),
        "objectives": list(dict.fromkeys(str(x.get("name")) for x in context.objectives if x.get("name"))),
        "constraints": list(dict.fromkeys(str(x.get("name") or x.get("stream") or x.get("variable")) for x in context.constraints if x.get("name") or x.get("stream") or x.get("variable"))),
    }


def _entities(question: str, context: ProcessContext, current_equipment: str | None) -> dict[str, list[str]]:
    q = question.casefold()
    known = _known_entities(context)
    found = {key: [value for value in values if value.casefold() in q] for key, values in known.items()}
    # Chinese parameter aliases may select existing variables, but an
    # equipment-scoped question may only select variables tied to that block.
    equipment_set = set(found["equipment"])
    aliases = {
        "回流": ("reflux", "rr"), "流量": ("flow", "rate"),
        "压力": ("press",), "温度": ("temp", "temperature"),
    }
    for alias, needles in aliases.items():
        if alias not in question:
            continue
        for key, var in context.variables.items():
            name = str(var.get("name") or key)
            related = set(str(x) for x in var.get("related_equipment", []))
            if any(needle in name.casefold() for needle in needles) and (not equipment_set or related.intersection(equipment_set)):
                found["variables"].append(name)
    if not found["equipment"] and current_equipment and _contains(
        question, "它", "这个塔", "这个设备", "该设备", "前面的设备", "前面的塔", "这个参数"
    ):
        if current_equipment in known["equipment"]:
            found["equipment"] = [current_equipment]
    return {key: list(dict.fromkeys(values)) for key, values in found.items()}


def _task(task_id: str, agent: str, objective: str, sources: list[str], depends_on: list[str] | None = None) -> DynamicTask:
    return DynamicTask(task_id, agent, objective, sources, depends_on or [])


def _local_plan(question: str, context: ProcessContext, *, current_equipment: str | None = None) -> DynamicPlan:
    q = question.strip()
    entities = _entities(q, context, current_equipment)
    eq = entities["equipment"]
    if not q:
        return DynamicPlan("clarification", entities, needs_clarification=True, clarification="请描述要分析的案例问题。", confidence=1.0)

    if _contains(q, "有哪些设备", "有什么设备", "设备清单", "设备有哪些", "哪些设备") or (
        _contains(q, "介绍", "概览") and _contains(q, "案例", "流程", "工艺")
    ):
        tasks = [_task("task_1", "CaseOverviewAgent", "读取案例设备、类型、设计变量、目标和约束", ["yaml", "nodedb", "aspen_model"])]
        intent = "case_overview"
        if _contains(q, "整个流程", "上下游", "物流", "连接", "关系"):
            tasks.append(_task("task_2", "TopologyAgent", "检查案例物流边和设备上下游", ["nodedb", "aspen_model"], ["task_1"]))
            intent = "comprehensive_analysis"
        return DynamicPlan(intent, entities, tasks, confidence=.96)

    if _contains(q, "不可信", "不可靠", "可信", "结果差", "为什么这次", "收敛", "失败", "结果异常"):
        tasks = [
            _task("task_1", "CaseOverviewAgent", "读取目标与约束定义，确定评价标准", ["yaml"]),
            _task("task_2", "SimulationDiagnosticAgent", "检查仿真状态、目标读取、约束和历史样本", ["simulation_db", "nodedb"], ["task_1"]),
            _task("task_3", "OptimizationAgent", "比较可行性和多目标历史结果", ["yaml", "simulation_db", "optimization_results"], ["task_2"]),
        ]
        return DynamicPlan("result_reliability_diagnosis", entities, tasks, confidence=.94)

    if len(eq) > 1 and _contains(q, "区别", "不一样", "比较", "关联", "联系", "流程"):
        tasks = [
            _task("task_1", "EquipmentAgent", "分别读取所提设备的类型、参数和配置关联", ["aspen_model", "nodedb", "yaml", "simulation_db"]),
            _task("task_2", "TopologyAgent", "查询设备之间可验证的 stream 连接边", ["nodedb", "aspen_model"], ["task_1"]),
        ]
        return DynamicPlan("equipment_comparison", entities, tasks, confidence=.9)

    if _contains(q, "比较", "对比", "历史结果", "之前", "刚才那个结果", "历史趋势", "趋势"):
        tasks = [
            _task("task_1", "SimulationDiagnosticAgent", "读取当前可用结果与历史记录", ["simulation_db"]),
            _task("task_2", "OptimizationAgent", "按目标、状态和可行性比较历史结果", ["yaml", "simulation_db", "optimization_results"], ["task_1"]),
        ]
        return DynamicPlan("case_comparison", entities, tasks, confidence=.91)

    if _contains(q, "上游", "下游", "上下游", "什么关系", "前面的设备", "物流", "连接", "经过哪些设备", "拓扑"):
        if not eq and not entities["streams"] and not current_equipment:
            return DynamicPlan("clarification", entities, needs_clarification=True, clarification="请给出要追踪的设备或物流名称。", confidence=.72)
        return DynamicPlan("topology_analysis", entities, [_task("task_1", "TopologyAgent", "从 NodeDB/Aspen 拓扑快照查找设备和物流边", ["nodedb", "aspen_model"])], confidence=.9)

    if _contains(q, "值得调整", "最值得", "调整哪些", "参数优先级", "调哪些", "优化变量"):
        tasks = [
            _task("task_1", "ParameterImpactAgent", "按设备关联、目标路径和历史样本筛选设计变量", ["yaml", "aspen_model", "simulation_db"]),
            _task("task_2", "OptimizationAgent", "结合可行性、Pareto 权衡和历史覆盖给出实验优先级", ["yaml", "simulation_db", "optimization_results"], ["task_1"]),
        ]
        return DynamicPlan("optimization_analysis", entities, tasks, confidence=.9)

    if _contains(q, "影响哪些目标", "影响目标", "参数变化", "参数会影响", "影响", "敏感性", "相关性", "敏感"):
        tasks = [_task("task_1", "ParameterImpactAgent", "检查参数与设备、目标和历史数据的关联，区分相关与因果", ["yaml", "aspen_model", "simulation_db", "ima"])]
        if _contains(q, "历史", "趋势", "相关性", "敏感性"):
            tasks.append(_task("task_2", "OptimizationAgent", "检查历史样本、相关性与多目标结果", ["simulation_db", "optimization_results"], ["task_1"]))
        return DynamicPlan("parameter_impact", entities, tasks, confidence=.89)

    if _contains(q, "有点奇怪", "感觉不对", "看起来不对", "帮我看看", "多产", "增加产品", "目标产品"):
        tasks = [
            _task("task_1", "SimulationDiagnosticAgent", "检查最近仿真收敛、目标读取和约束观察", ["simulation_db", "yaml"]),
            _task("task_2", "ParameterImpactAgent", "检查配置变量与目标路径及历史样本的关系，区分相关与因果", ["yaml", "aspen_model", "simulation_db"], ["task_1"]),
            _task("task_3", "OptimizationAgent", "读取可行性、多目标权衡与可执行的下一步实验", ["yaml", "simulation_db", "optimization_results"], ["task_2"]),
        ]
        return DynamicPlan("comprehensive_analysis", entities, tasks, confidence=.76)

    if eq and _contains(q, "参数", "变量", "设备", "什么", "类型", "介绍", "关键", "作用", "干什么", "做什么", "功能", "角色"):
        tasks = [_task("task_1", "EquipmentAgent", "读取指定设备的模型、配置参数、目标路径和数据边界", ["aspen_model", "nodedb", "yaml", "simulation_db"])]
        explanation = _contains(q, "作用", "干什么", "做什么", "功能", "角色", "流程")
        if explanation:
            tasks.append(_task("task_2", "TopologyAgent", "检查可验证的进出物流和邻接设备，不从类型推断工艺角色", ["nodedb", "aspen_model"]))
            tasks.append(_task("task_3", "ParameterImpactAgent", "检查设备关联目标路径及可用历史观察", ["yaml", "simulation_db"]))
        intent = "equipment_explanation" if explanation else "equipment_parameters"
        return DynamicPlan(intent, entities, tasks, confidence=.95)

    if entities["streams"]:
        return DynamicPlan("topology_analysis", entities, [_task("task_1", "TopologyAgent", "读取指定物流的拓扑端点", ["nodedb", "aspen_model"])], confidence=.85)

    # Unknown natural-language questions still get a useful local inspection
    # before asking a question. Conceptual requests can also consult IMA.
    if _contains(q, "原理", "机理", "什么意思", "是什么", "如何", "怎么"):
        return DynamicPlan("technical_question", entities, [_task("task_1", "TechnicalAdvisorAgent", "依据已提供的案例证据和可用 IMA 引用回答；证据不足时说明边界", ["yaml", "aspen_model", "simulation_db", "ima"])], confidence=.7)
    if _contains(q, "案例", "流程", "结果", "分析", "看看", "建议"):
        tasks = [
            _task("task_1", "CaseOverviewAgent", "读取可用案例定义和设备", ["yaml", "nodedb", "aspen_model"]),
            _task("task_2", "SimulationDiagnosticAgent", "检查可用历史仿真状态", ["simulation_db"], ["task_1"]),
        ]
        return DynamicPlan("comprehensive_analysis", entities, tasks, confidence=.72)

    # Detect device-like identifiers that are absent from the selected case.
    mentioned = re.findall(r"(?<![\w])(?:[A-Z]{1,5}\d{2,}[A-Z0-9_]*)(?![\w])", q, flags=re.I)
    known = {x.casefold() for x in _known_entities(context)["equipment"] + _known_entities(context)["streams"]}
    unknown = [x for x in mentioned if x.casefold() not in known]
    if unknown:
        prompt = f"当前案例实体中没有找到 {', '.join(dict.fromkeys(unknown))}。请确认名称或切换到对应案例。"
        return DynamicPlan("clarification", entities, needs_clarification=True, clarification=prompt, confidence=.8)
    return DynamicPlan(
        "clarification", entities, needs_clarification=True,
        clarification="我还不能从当前案例证据确定你指的对象；请补充设备、物流、参数或目标名称。",
        confidence=.45,
    )


def build_dynamic_plan(question: str, context: ProcessContext, *, current_equipment: str | None = None) -> DynamicPlan:
    """Build the deterministic local plan; retained as the no-LLM API."""
    return _local_plan(question, context, current_equipment=current_equipment)


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        lo, hi = cleaned.find("{"), cleaned.rfind("}")
        if lo < 0 or hi <= lo:
            return None
        cleaned = cleaned[lo:hi + 1]
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _validate_plan(raw: dict[str, Any], question: str, context: ProcessContext, *, current_equipment: str | None) -> DynamicPlan | None:
    required = {"intent", "entities", "tasks", "needs_clarification", "confidence"}
    if not required.issubset(raw) or set(raw) - (required | {"clarification"}):
        return None
    intent = raw.get("intent")
    if intent not in ALLOWED_INTENTS:
        return None
    known = _known_entities(context)
    entity_raw = raw.get("entities")
    if not isinstance(entity_raw, dict) or set(entity_raw) - set(known):
        return None
    entities: dict[str, list[str]] = {}
    canonical = {key: {value.casefold(): value for value in values} for key, values in known.items()}
    for key in known:
        values = entity_raw.get(key, [])
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            return None
        normalized = []
        for value in values:
            name = canonical[key].get(value.strip().casefold())
            if name is None:
                return None
            normalized.append(name)
        entities[key] = list(dict.fromkeys(normalized))
    if entities["equipment"] and entities["variables"]:
        related_by_name = {
            str(value.get("name") or key): set(str(x) for x in value.get("related_equipment", []))
            for key, value in context.variables.items()
        }
        for variable in entities["variables"]:
            if not related_by_name.get(variable, set()).intersection(entities["equipment"]):
                return None
    # Deterministic entity matching supplements model extraction. Pronouns
    # only resolve to an entity that was established by a prior case turn.
    direct = _entities(question, context, current_equipment)
    for key, values in direct.items():
        entities[key] = list(dict.fromkeys(entities.get(key, []) + values))

    tasks_raw = raw.get("tasks")
    clarify = raw.get("needs_clarification")
    confidence = raw.get("confidence")
    if not isinstance(clarify, bool) or not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        return None
    if not isinstance(tasks_raw, list) or len(tasks_raw) > 8:
        return None
    tasks: list[DynamicTask] = []
    ids: set[str] = set()
    for item in tasks_raw:
        if not isinstance(item, dict) or not {"id", "agent", "objective", "sources", "depends_on"}.issubset(item) or set(item) - {"id", "agent", "objective", "sources", "depends_on", "entities"}:
            return None
        task_id, agent, objective = item["id"], item["agent"], item["objective"]
        sources, deps = item["sources"], item["depends_on"]
        if not all(isinstance(x, str) for x in (task_id, agent, objective)) or not task_id or len(task_id) > 40 or len(objective) > 500:
            return None
        if agent not in ALLOWED_AGENTS or task_id in ids:
            return None
        if not isinstance(sources, list) or any(x not in ALLOWED_SOURCES for x in sources):
            return None
        if not isinstance(deps, list) or any(not isinstance(x, str) for x in deps):
            return None
        ids.add(task_id)
        scope = item.get("entities", {})
        if not isinstance(scope, dict) or set(scope) - set(known):
            return None
        for key, values in scope.items():
            if not isinstance(values, list) or any(not isinstance(v, str) or v.strip().casefold() not in canonical[key] for v in values):
                return None
            normalized_scope = [canonical[key][v.strip().casefold()] for v in values]
            if any(value not in entities[key] for value in normalized_scope):
                return None
            scope[key] = list(dict.fromkeys(normalized_scope))
        tasks.append(DynamicTask(task_id, agent, objective.strip(), list(dict.fromkeys(sources)), list(dict.fromkeys(deps)), entities=scope))
    if clarify and not str(raw.get("clarification", "")).strip():
        return None
    if clarify and tasks:
        return None
    if not clarify and not tasks:
        return None
    if any(dep not in ids or dep == task.id for task in tasks for dep in task.depends_on):
        return None
    # Reject cycles before execution.
    resolved: set[str] = set()
    for task in tasks:
        if any(dep not in resolved for dep in task.depends_on):
            return None
        resolved.add(task.id)
    return DynamicPlan(
        intent=intent, entities=entities, tasks=tasks,
        needs_clarification=clarify,
        clarification=str(raw.get("clarification", ""))[:500],
        confidence=float(confidence), used_llm=True, planner_mode="llm",
    )


def plan_conversation_turn(
    question: str,
    context: ProcessContext,
    *,
    current_equipment: str | None = None,
    history: list[dict[str, str]] | None = None,
    llm_config: Any = None,
    chat_fn: Callable[..., str] | None = None,
) -> DynamicPlan:
    """Ask the coordinator LLM for a schema-checked plan, then fail closed."""
    cfg = llm_config if llm_config is not None else load_llm_config()
    if not is_configured(cfg):
        fallback = _local_plan(question, context, current_equipment=current_equipment)
        fallback.fallback_reason = f"LLM 未配置（{cfg.api_key_env} 未设置）"
        return fallback

    history_rows = []
    for item in (history or [])[-8:]:
        history_rows.append({
            "user": str(item.get("user", ""))[-500:],
            "assistant": str(item.get("assistant", ""))[-800:],
        })
    payload = {
        "user_question": question,
        "recent_history": history_rows,
        "current_equipment_from_prior_turn": current_equipment,
        "case": {
            "model_id": context.model_id,
            "equipment": _known_entities(context)["equipment"],
            "streams": _known_entities(context)["streams"],
            "variables": _known_entities(context)["variables"],
            "objectives": _known_entities(context)["objectives"],
            "constraints": _known_entities(context)["constraints"],
            "topology_has_stream_edges": bool(context.topology.get("edges")),
            "history_sample_count": len(context.history_evidence),
        },
        "allowed_agents": sorted(ALLOWED_AGENTS),
        "allowed_sources": sorted(ALLOWED_SOURCES),
        "allowed_intents": sorted(ALLOWED_INTENTS),
    }
    try:
        response = (chat_fn or chat)(
            cfg,
            system=PLANNER_SYSTEM_PROMPT,
            user=json.dumps(payload, ensure_ascii=False, default=str),
        )
    except Exception as exc:
        fallback = _local_plan(question, context, current_equipment=current_equipment)
        fallback.fallback_reason = f"LLM planning call failed ({type(exc).__name__}): {str(exc)[:160]}"
        return fallback
    parsed = _parse_json_object(response)
    if parsed is None:
        fallback = _local_plan(question, context, current_equipment=current_equipment)
        fallback.fallback_reason = "LLM 规划返回非法 JSON"
        return fallback
    plan = _validate_plan(parsed, question, context, current_equipment=current_equipment)
    if plan is None:
        fallback = _local_plan(question, context, current_equipment=current_equipment)
        fallback.fallback_reason = "LLM 计划未通过 schema、Agent 白名单或案例实体校验"
        return fallback
    return plan


def _source_labels(context: ProcessContext) -> list[str]:
    labels = [str(x.source.value) for x in context.sources]
    if context.history_evidence:
        labels.append("simulation_db")
    if context.equipment or context.streams:
        labels.append("aspen_model")
    if context.equipment_parameters:
        labels.append("nodedb")
    return list(dict.fromkeys(labels))


def _overview(context: ProcessContext) -> str:
    types: dict[str, int] = {}
    for item in context.equipment:
        kind = str(item.get("block_type") or item.get("category") or "unknown")
        types[kind] = types.get(kind, 0) + 1
    equipment = ", ".join(str(x.get("name")) for x in context.equipment) or "未读取到设备"
    variables = ", ".join(str(v.get("name") or k) for k, v in context.variables.items()) or "无设计变量记录"
    objectives = ", ".join(str(x.get("name")) for x in context.objectives if x.get("name")) or "无目标函数记录"
    constraints = ", ".join(str(x.get("name") or x.get("stream") or x.get("variable")) for x in context.constraints) or "无约束记录"
    kinds = ", ".join(f"{name} {count} 个" for name, count in types.items()) or "类型未读取"
    return f"设备共 {len(context.equipment)} 个（{kinds}）：{equipment}。设计变量：{variables}。目标：{objectives}。约束：{constraints}。"


def _local_answer(question: str, plan: DynamicPlan, context: ProcessContext) -> str:
    if plan.needs_clarification:
        return f"结论：{plan.clarification}\n证据：当前案例 ProcessContext\n不确定性：高\n下一步：请确认对象名称。"
    sections: list[str] = []
    sources = _source_labels(context)
    uncertainty = list(context.diagnostics)
    if plan.intent in {"case_overview", "comprehensive_analysis"}:
        sections.append(_overview(context))
    if plan.intent == "topology_analysis":
        edges = context.topology.get("edges") or []
        if edges:
            sections.append("拓扑边：" + "; ".join(f"{e.get('id') or e.get('label')}: {e.get('source')} → {e.get('target')}" for e in edges))
        else:
            sections.append("NodeDB 当前没有可用的 stream 边，无法可靠追踪上游和下游。")
            uncertainty.append("拓扑缺少 stream 边")
    if plan.intent in {"equipment_parameters", "equipment_explanation"}:
        for name in plan.entities.get("equipment", []):
            equipment = next((x for x in context.equipment if x.get("name") == name), None)
            if equipment is None:
                sections.append(f"当前案例没有设备 {name} 的记录。")
                continue
            variables = [v.get("name") or k for k, v in context.variables.items() if name in v.get("related_equipment", [])]
            sections.append(f"{name} 的 Aspen 类型为 {equipment.get('block_type') or '未记录'}，路径 {equipment.get('aspen_path') or '未记录'}。")
            sections.append(f"YAML 关联设计变量：{', '.join(map(str, variables)) if variables else '当前配置未记录该设备的可调参数'}。")
            params = context.equipment_parameters.get(name, [])
            if params:
                sample = params[:8]
                sections.append("NodeDB catalog 输入快照：" + "; ".join(f"{p['name']}={p['value']} {p.get('unit','')}".strip() for p in sample) + "。这些是扫描快照值，不代表本次仿真当前值。")
                sources.append("nodedb")
            else:
                sections.append("没有可用的 NodeDB 输入节点快照参数。")
    if plan.intent == "parameter_impact":
        equipment = plan.entities.get("equipment", [])
        variables = [(k, v) for k, v in context.variables.items() if not equipment or set(equipment).intersection(v.get("related_equipment", []))]
        if equipment and not variables:
            sections.append(f"YAML 中没有记录与 {', '.join(equipment)} 关联的可调设计变量，也没有证据支持这些设备参数对目标的确定影响。")
        elif variables:
            sections.append("关联设计变量：" + "; ".join(f"{v.get('name') or k}（设备 {', '.join(v.get('related_equipment', [])) or '未标注'}，YAML 初始值 {v.get('initial_value', '未记录')}）" for k, v in variables) + "。YAML 初始值不是仿真当前值。")
        else:
            sections.append("当前配置可用设计变量：" + ", ".join(str(v.get("name") or k) for k, v in context.variables.items()))
        sections.append("参数与目标的因果影响尚未由受控实验验证；历史样本不足时不能据此声称趋势。")
        uncertainty.append("参数因果关系未验证")
    if plan.intent == "result_reliability_diagnosis":
        if not context.history_evidence:
            sections.append("SimulationDB 没有历史工况，无法核验收敛、目标读取、约束满足或历史趋势。")
            uncertainty.append("SimulationDB 为空")
        else:
            sections.append(f"SimulationDB 中可读取 {len(context.history_evidence)} 条历史记录；需结合每条工况的详细状态判断本次结果。")
    if plan.intent in {"case_comparison", "optimization_analysis"}:
        sections.append(f"可用历史工况 {len(context.history_evidence)} 条；仅在目标值和状态字段齐全时才能比较 Pareto/可行性。")
    if not sections:
        sections.append("当前案例证据不足以形成可靠结论。")
    return (
        "结论：" + " ".join(sections) +
        "\n证据：" + (", ".join(dict.fromkeys(sources)) or "暂无可用来源") +
        "\n不确定性：" + ("；".join(dict.fromkeys(uncertainty)) if uncertainty else "本地规则回答，未调用 LLM") +
        "\n下一步：确认相关设备/参数路径，并通过受控小步长实验验证影响。"
    )


def answer_dynamic(question: str, context: ProcessContext, *, current_equipment: str | None = None) -> DynamicAnswer:
    """Compatibility helper for callers that explicitly want local rules."""
    plan = _local_plan(question, context, current_equipment=current_equipment)
    return DynamicAnswer(plan, _local_answer(question, plan, context))
