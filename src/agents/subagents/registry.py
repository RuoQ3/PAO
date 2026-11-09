from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from typing import Any

from src.agents.orchestration.models import Evidence, TaskResult, TaskSpec


@dataclass(frozen=True)
class SubagentDefinition:
    name: str
    description: str
    tool_allowlist: tuple[str, ...] = ()
    max_steps: int = 8
    budget: int = 4
    allow_resident: bool = True
    allow_subtasks: bool = False
    input_context: tuple[str, ...] = ("process_context",)
    evidence_sources: tuple[str, ...] = ("ProcessContext",)

    def __post_init__(self) -> None:
        if not self.name or self.max_steps < 1 or self.budget < 1:
            raise ValueError("invalid subagent definition limits")
        if self.allow_subtasks:
            raise ValueError("PAO subagents cannot create subagents by default")


def _general(task: TaskSpec, **_: Any) -> TaskResult:
    return TaskResult(task_id=task.task_id, agent=task.agent, summary=task.prompt,
                      facts=[{"claim": task.prompt, "kind": "direct_answer"}],
                      evidence=[Evidence("user", "prompt", "User request", task.prompt, confidence=1.0)])


def _context_agent(task: TaskSpec, *, context: Any = None, **_: Any) -> TaskResult:
    """Read-only specialist adapter used until domain-specific LLM policies are added."""
    if context is None:
        return TaskResult(task.task_id, task.agent, summary="No project context is available.",
                          uncertainties=["ProcessContext was not supplied"])
    variables = getattr(context, "variables", {}) or {}
    objectives = getattr(context, "objectives", []) or []
    constraints = getattr(context, "constraints", []) or []
    equipment = list(getattr(context, "equipment", []) or [])
    streams = list(getattr(context, "streams", []) or [])
    allow = set(_.get("tool_allowlist", ()) or ())
    if task.tool_allowlist and not allow.issuperset(task.tool_allowlist):
        return TaskResult.failed(task, "subagent tool allowlist was not enforced")
    facts = [{"kind": "context_summary", "variables": len(variables),
              "objectives": len(objectives), "constraints": len(constraints)}]
    # A follow-up may name a block (for example T0301) without repeating the
    # word "equipment". Treat that as an inventory/detail request so the
    # read-only adapter returns topology evidence instead of a generic status.
    prompt_lower = task.prompt.casefold()
    def mentioned(name: Any) -> bool:
        value = str(name or "").casefold()
        if not value:
            return False
        # Numeric stream IDs must not match digits embedded in another ID or
        # in a parameter value. Equipment IDs are handled the same way.
        return re.search(r"(?<![a-z0-9_-])" + re.escape(value) + r"(?![a-z0-9_-])", prompt_lower) is not None

    named_equipment = [x for x in equipment if mentioned(x.get("name"))]
    named_streams = [x for x in streams if mentioned(x.get("name"))]
    if named_equipment or named_streams:
        task.task_type = "project_query"
    if (task.task_type in {"project_research", "project_query", "analysis", "parameter_impact", "equipment_inventory"}
            or any(word in task.prompt for word in ("\u8bbe\u5907", "\u88c5\u7f6e", "\u7269\u6d41", "\u9879\u76ee\u4e0a\u4e0b\u6587"))
            or "equipment" in task.prompt.casefold()
            or named_equipment or named_streams):
        facts.append({"kind": "equipment_inventory", "count": len(equipment), "items": equipment})
        facts.append({"kind": "stream_inventory", "count": len(streams), "items": streams})
        names = ", ".join(str(x.get("name") or x.get("label")) for x in equipment) or "未发现设备快照"
        summary = f"项目上下文包含 {len(equipment)} 个设备、{len(streams)} 条物流：{names}。"
    else:
        summary = f"{task.agent} reviewed the available project context."
    if named_equipment:
        for item in named_equipment:
            name = str(item.get("name") or item.get("label") or "")
            related_streams = [s for s in streams
                               if s.get("upstream") == name or s.get("downstream") == name]
            parameters = list(getattr(context, "equipment_parameters", {}).get(name, []) or [])
            related_variables = [v for v in variables.values()
                                 if name in (v.get("related_equipment") or [])]
            facts.append({"kind": "equipment_detail", "name": name,
                          "equipment": item, "related_streams": related_streams,
                          "parameters": parameters, "variables": related_variables})
            facts.append({"kind": "equipment_role", "name": name,
                          "block_type": item.get("block_type"),
                          "role": item.get("role") or _role_for_block(item.get("block_type")),
                          "inference_method": item.get("inference_method") or "block_type_mapping",
                          "confidence": item.get("confidence"),
                          "evidence_ids": list(item.get("evidence_ids") or []),
                          "basis": "block_type and read-only flowsheet topology",
                          "kind_of_claim": "inferred_role"})
        cards = []
        for item in named_equipment:
            name = str(item.get("name") or item.get("label") or "")
            typ = str(item.get("block_type") or "unknown")
            related = [s for s in streams
                       if s.get("upstream") == name or s.get("downstream") == name]
            incoming = [str(s.get("name")) for s in related if s.get("downstream") == name]
            outgoing = [str(s.get("name")) for s in related if s.get("upstream") == name]
            cards.append(f"{name}（{typ}）：进料 {', '.join(incoming) or '未知'}；出料 {', '.join(outgoing) or '未知'}；作用：{_role_for_block(typ)}")
        summary = "；".join(cards)
    # Expand a named RADFRAC card with the observed feed/outlet split.  This
    # makes follow-up questions about how a tower works answerable from the
    # read-only .bkp snapshot instead of returning only its block type.
    if named_equipment:
        for item in named_equipment:
            name = str(item.get("name") or item.get("label") or "")
            if str(item.get("block_type") or "").upper() != "RADFRAC":
                continue
            incoming = [s for s in streams if s.get("downstream") == name]
            outgoing = [s for s in streams if s.get("upstream") == name]
            params = [p for p in variables.values()
                      if name in (p.get("related_equipment") or [])]
            feed = incoming[0] if incoming else {}
            comps = sorted(list(feed.get("components") or []),
                           key=lambda c: float(c.get("mole_fraction", 0.0) or 0.0),
                           reverse=True)
            main = comps[0] if comps else None
            enriched, depleted = [], []
            if main:
                target = str(main.get("name"))
                base = float(main.get("mole_fraction", 0.0) or 0.0)
                for outlet in outgoing:
                    value = next((float(c.get("mole_fraction"))
                                  for c in outlet.get("components") or []
                                  if str(c.get("name", "")).casefold() == target.casefold()), None)
                    if value is not None:
                        (enriched if value > base else depleted).append((outlet, value))
            detail = (f"{name} 是 RADFRAC 精馏塔：进料 {', '.join(str(s.get('name')) for s in incoming) or '未知'} "
                      "在塔内按汽液平衡逐级接触，回流与再沸提供分离驱动力。")
            if main:
                detail += f" 进料主组分为 {main.get('name')}（{float(main.get('mole_fraction', 0.0)):.6g} mol fraction）。"
            if enriched:
                out, value = max(enriched, key=lambda x: x[1])
                detail += f" 快照显示 {out.get('name')} 中 {main.get('name')} 富集至 {value:.6g}，作为主组分富集的塔底方向物流。"
            if depleted:
                out, value = min(depleted, key=lambda x: x[1])
                detail += f" {out.get('name')} 中该组分降至 {value:.6g}，带走较易挥发的伴随组分并送往 {out.get('downstream') or '下游'}。"
            if params:
                bits = [f"{p.get('name')}={p.get('initial_value')}" for p in params if p.get('initial_value') is not None]
                if bits:
                    detail += " 当前配置参数：" + "、".join(bits) + "。"
            summary += " " + detail
            facts.append({"kind": "equipment_operation", "name": name,
                          "block_type": "RADFRAC", "incoming": incoming,
                          "outgoing": outgoing, "dominant_feed_component": main,
                          "enriched_outlets": [{"stream": s.get("name"), "mole_fraction": v} for s, v in enriched],
                          "depleted_outlets": [{"stream": s.get("name"), "mole_fraction": v} for s, v in depleted],
                          "parameters": params,
                          "basis": "bkp_index stream composition snapshot and YAML parameters"})

    if named_streams:
        for item in named_streams:
            name = str(item.get("name") or "")
            components = list(item.get("components") or [])
            nonzero = [c for c in components if abs(float(c.get("mole_fraction", 0.0) or 0.0)) > 1e-12]
            facts.append({"kind": "stream_detail", "name": name,
                          "stream": item, "components": components,
                          "composition_basis": item.get("composition_basis"),
                          "kind_of_claim": "observed_fact",
                          "evidence_ids": list(item.get("evidence_ids") or [])})
            facts.append({"kind": "stream_composition", "name": name,
                          "components": nonzero,
                          "basis": item.get("composition_basis", "topology only")})
        cards = []
        for item in named_streams:
            name = str(item.get("name") or "")
            components = [c for c in (item.get("components") or [])
                          if abs(float(c.get("mole_fraction", 0.0) or 0.0)) > 1e-12]
            if components:
                comp_text = ", ".join(
                    f"{c['name']} ({float(c.get('mole_fraction', 0.0)):.6g} mol fraction)"
                    for c in components
                )
                cards.append(f"流股{name}包含：{comp_text}。")
            else:
                cards.append(f"流股{name}的组成数据未加载；目前只有拓扑信息。")
        summary = "；".join(cards)
        if any(token in task.prompt for token in (chr(20998)+chr(26512), chr(20316)+chr(29992), chr(24847)+chr(20041), chr(24433)+chr(21709), chr(35780)+chr(20272), chr(21028)+chr(26029), chr(25152)+chr(20197), chr(28982)+chr(21518), "analysis", "analyze", "impact")):
            for item in named_streams:
                components = [c for c in (item.get("components") or [])
                              if abs(float(c.get("mole_fraction", 0.0) or 0.0)) > 1e-12]
                ranked = sorted(components, key=lambda c: float(c.get("mole_fraction", 0.0) or 0.0), reverse=True)
                upstream = item.get("upstream") or "未知"
                downstream = item.get("downstream") or "流程末端"
                if ranked:
                    lead = ranked[0]
                    impurity = [c for c in ranked[1:] if float(c.get("mole_fraction", 0.0) or 0.0) >= 1e-3]
                    impurity_text = "、".join(str(c["name"]) for c in impurity) or "其余组分均为低含量"
                    summary += (f" 分析：该流股由 {upstream} 送往 {downstream}，"
                                f"主组分为 {lead['name']}（{float(lead.get('mole_fraction', 0.0)):.6g} mol fraction）；"
                                f"需要关注的伴随组分为 {impurity_text}。"
                                "这表明后续分离的主要任务是围绕主组分纯化并处理伴随组分。")
                    facts.append({"kind": "stream_analysis", "name": item.get("name"),
                                  "upstream": upstream, "downstream": downstream,
                                  "dominant_component": lead, "notable_components": impurity,
                                  "conclusion": "主组分占优，后续分离重点是纯化主组分并去除伴随组分"})
    evidence = [Evidence("tool", "ProcessContext", "Read-only project context loaded",
                         {"variables": len(variables), "objectives": len(objectives),
                          "constraints": len(constraints)}, confidence=1.0)]
    return TaskResult(task.task_id, task.agent,
                      summary=summary,
                      facts=facts, evidence=evidence,
                      uncertainties=list(getattr(context, "diagnostics", []) or []))


def _role_for_block(block_type: Any) -> str:
    """Return a conservative process role from an Aspen block type."""
    roles = {
        "PUMP": "泵送并提升物流压力",
        "MIXER": "混合进料物流",
        "RCSTR": "连续搅拌釜式反应器，进行反应",
        "RSTOIC": "化学计量反应器，按设定反应计量转化",
        "FLASH2": "两相闪蒸，按汽液平衡分离物流",
        "HEATER": "加热物流并调整其热状态",
        "RADFRAC": "精馏塔，通过汽液平衡和回流实现组分分离",
    }
    return roles.get(str(block_type or "").upper(), "块作用需结合 Aspen 模型参数进一步确认")


def _boundary(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    """Adapter for the deterministic boundary advisor."""
    if context is None:
        return _context_agent(task, context=None, **kwargs)
    try:
        from src.agents.subagents.boundary_advisor import VarMeta, recommend_boundaries
        variables = []
        for name, raw in (getattr(context, "variables", {}) or {}).items():
            variables.append(VarMeta(name=str(name), initial_value=raw.get("initial_value"),
                                     unit=str(raw.get("unit", "")),
                                     semantic_role=str(raw.get("semantic_role", "")),
                                     var_type=str(raw.get("type", "continuous")),
                                     lower_global=raw.get("lower_bound"), upper_global=raw.get("upper_bound")))
        report = recommend_boundaries(variables)
        return TaskResult(task.task_id, task.agent, summary="Boundary advisor completed a deterministic review; hard bounds were not modified.",
                          facts=[{"kind": "role_behavior", "role": "boundary_advisor", "compatibility_adapter": True, "write_performed": False},
                                 {"kind": "soft_boundary_recommendation", "value": report.to_dict() if hasattr(report, "to_dict") else str(report)}],
                          evidence=[Evidence("model", "boundary_advisor", "Boundary recommendation generated", report.to_dict() if hasattr(report, "to_dict") else str(report), confidence=1.0)],
                          risks=list(getattr(report, "warnings", []) or []))
    except Exception as exc:
        return TaskResult.failed(task, f"boundary advisor: {exc}")


def _optimization(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    result = _context_agent(task, context=context, **kwargs)
    result.summary = "Optimization advisor reviewed objectives, constraints and available history."
    result.facts.append({"kind": "role_behavior", "role": "optimization_advisor", "compatibility_adapter": True, "write_performed": False, "simulation_performed": False})
    result.evidence.append(Evidence("subagent", "optimization_advisor", "Optimization role review", task.prompt, confidence=1.0))
    result.recommended_actions.append({"action": "review_feasibility_and_budget", "requires_confirmation": True})
    return result


def _role_adapter(role: str, task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    """Compatibility adapter with explicit role behavior and provenance."""
    result = _context_agent(task, context=context, **kwargs)
    result.facts.append({"kind": "role_behavior", "role": role,
                         "compatibility_adapter": True, "write_performed": False,
                         "simulation_performed": False})
    result.evidence.append(Evidence("subagent", role, f"{role} read-only role review", task.prompt, confidence=1.0))
    # Keep concrete inventory answers visible to the user.  The compatibility
    # role label is useful metadata, but must not replace a verified equipment
    # or stream listing with a generic status sentence.
    if not any(item.get("kind") == "equipment_inventory" for item in result.facts):
        result.summary = f"{role} completed a bounded read-only review."
    return result


def _research(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    # The classifier may select project_researcher while emitting a generic
    # task_type.  When a flowsheet is available, force the read-only adapter
    # through its inventory path so the role cannot collapse to a status line.
    if context is not None and (getattr(context, "equipment", None) or getattr(context, "streams", None)):
        task.task_type = "project_query"
    result = _role_adapter("project_researcher", task, context=context, **kwargs)
    # A project-level research request should still expose the verified
    # inventory when a flowsheet snapshot was supplied.  The generic role
    # sentence is only appropriate when the context is genuinely empty.
    if context is not None and (getattr(context, "equipment", None) or getattr(context, "streams", None)):
        if any(item.get("kind") == "equipment_inventory" for item in result.facts):
            result.summary = next(
                (str(item.get("summary")) for item in result.facts if item.get("kind") == "project_overview" and item.get("summary")),
                result.summary,
            )
    return result


def _process(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    result = _context_agent(task, context=context, **kwargs)
    result.facts.append({"kind": "role_behavior", "role": "process_advisor", "compatibility_adapter": True, "write_performed": False, "simulation_performed": False})
    result.evidence.append(Evidence("subagent", "process_advisor", "Process role review", task.prompt, confidence=1.0))
    return result


def _kinetics(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    result = _role_adapter("kinetics_advisor", task, context=context, **kwargs)
    result.facts.append({"kind": "kinetics_analysis", "fit_requested": False})
    return result


def _risk(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    result = _role_adapter("risk_reviewer", task, context=context, **kwargs)
    result.risks.append("Any action requires fresh validation and human confirmation.")
    return result


def _code(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    result = _role_adapter("code_engineer", task, context=context, **kwargs)
    result.summary = "code_engineer prepared a code analysis and proposal; no files were changed."
    result.uncertainties.append("Proposal only; ActionGateway approval is required for writes.")
    return result


def _report(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    result = _role_adapter("report_writer", task, context=context, **kwargs)
    result.summary = "report_writer drafted a report from available Evidence; no files were written."
    return result


class SubagentRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., TaskResult]] = {}
        self._definitions: dict[str, SubagentDefinition] = {}

    def register(self, name: str, handler: Callable[..., TaskResult], *, definition: SubagentDefinition | None = None, replace: bool = False) -> None:
        if name in self._handlers and not replace:
            raise ValueError(f"subagent already registered: {name}")
        self._handlers[name] = handler
        self._definitions[name] = definition or SubagentDefinition(name=name, description=name)

    def get(self, name: str) -> Callable[..., TaskResult]:
        return self._handlers[name]

    def definition(self, name: str) -> SubagentDefinition:
        return self._definitions[name]

    def definitions(self) -> Mapping[str, SubagentDefinition]:
        return dict(self._definitions)

    def as_mapping(self) -> Mapping[str, Callable[..., TaskResult]]:
        return dict(self._handlers)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))


def default_registry() -> SubagentRegistry:
    registry = SubagentRegistry()
    definitions = {
        "project_researcher": SubagentDefinition("project_researcher", "Research source, configuration, tests and historical results.", ("read_files", "read_yaml", "read_node_db", "read_simulation_db", "read_checkpoint"), 8, 6, True, False, ("yaml", "files", "node_db", "simulation_db", "history"), ("yaml", "files", "NodeDB", "SimulationDB", "checkpoint")),
        "process_advisor": SubagentDefinition("process_advisor", "Review process, equipment and streams from read-only snapshots.", ("read_process_context", "read_node_db", "read_simulation_db"), 8, 6, True, False, ("process_context", "node_db", "simulation_db"), ("ProcessContext", "NodeDB", "SimulationDB")),
        "optimization_advisor": SubagentDefinition("optimization_advisor", "Review objectives, Pareto evidence, convergence and budget.", ("read_yaml", "read_simulation_db", "read_checkpoint"), 10, 8, True, False, ("yaml", "simulation_db", "history"), ("yaml", "SimulationDB", "checkpoint")),
        "boundary_advisor": SubagentDefinition("boundary_advisor", "Recommend soft search regions while preserving hard bounds.", ("read_yaml", "read_process_context"), 6, 4, False, False, ("yaml", "process_context"), ("yaml", "ProcessContext")),
        "kinetics_advisor": SubagentDefinition("kinetics_advisor", "Review kinetics fits and parameter evidence without writing or simulating.", ("read_files", "read_simulation_db"), 8, 6, True, False, ("files", "simulation_db"), ("files", "SimulationDB")),
        "risk_reviewer": SubagentDefinition("risk_reviewer", "Review constraints, failure modes and action impact.", ("read_process_context", "read_checkpoint"), 6, 4, True, False, ("process_context", "history"), ("ProcessContext", "checkpoint")),
        "code_engineer": SubagentDefinition("code_engineer", "Analyze code and produce modification proposals without writing.", ("read_files",), 8, 5, False, False, ("files",), ("files", "user")),
        "report_writer": SubagentDefinition("report_writer", "Draft reports from validated Evidence only.", ("read_evidence", "read_files"), 8, 5, False, False, ("files", "history"), ("Evidence", "files")),
        "general_agent": SubagentDefinition("general_agent", "Handle general questions without execution tools.", (), 4, 2, True, False, ("none",), ("user",)),
    }
    handlers = {
        "general_agent": _general,
        "boundary_advisor": _boundary,
        "optimization_advisor": _optimization,
        "project_researcher": _research,
        "process_advisor": _process,
        "kinetics_advisor": _kinetics,
        "risk_reviewer": _risk,
        "code_engineer": _code,
        "report_writer": _report,
    }
    for name, handler in handlers.items():
        registry.register(name, handler, definition=definitions[name])
    return registry
