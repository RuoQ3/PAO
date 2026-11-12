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


def _reasoning_agent(task: TaskSpec, *, context: Any = None, llm_config=None, llm_call=None, definition=None, **kwargs) -> TaskResult:
    from src.agents.reasoning import reason, snapshot
    context = snapshot(context)
    context["conversation"] = task.metadata.get("conversation", [])
    from .tool_runtime import ToolSession
    tools = kwargs.get("tools") or default_tools_for_roles()
    tool_session = ToolSession(tools, task, context, max_tool_calls=kwargs.get("max_tool_calls", 32),
        audit=kwargs.get("tool_audit"), cfg=llm_config, stopped=kwargs.get("stopped", lambda: False))
    data, evidence = reason(task.prompt, context=context, role=task.agent,
                            llm_config=llm_config, llm_call=llm_call,
                            tool_session=tool_session, role_contract={"responsibility": definition.description,
                                           "allowed_tools": task.tool_allowlist,
                                           "evidence_sources": definition.evidence_sources} if definition else None)
    return TaskResult(task.task_id, task.agent, summary=data["text"],
                      facts=[{"kind": "role_behavior", "role": task.agent, "write_performed": False, "simulation_performed": False}],
                      evidence=[Evidence("tool", "ProcessContext", "Read-only evidence supplied to reasoning", evidence)],
                      uncertainties=data["uncertainties"], blocks=data.get("blocks", []),
                      recommended_actions=[p.to_dict() for p in tool_session.proposals], audit=list(tool_session.audit))


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


def default_registry(*, llm_config=None, llm_call=None, tools=None) -> SubagentRegistry:
    registry = SubagentRegistry()
    definitions = {
        "project_researcher": SubagentDefinition("project_researcher", "Research source, configuration, tests and historical results.", ("read_files", "read_yaml", "read_node_db", "read_simulation_db", "read_checkpoint"), 8, 6, True, False, ("yaml", "files", "node_db", "simulation_db", "history"), ("yaml", "files", "NodeDB", "SimulationDB", "checkpoint")),
        "process_advisor": SubagentDefinition("process_advisor", "Review process, equipment and streams from read-only snapshots.", ("read_process_context", "read_node_db", "read_simulation_db"), 8, 6, True, False, ("process_context", "node_db", "simulation_db"), ("ProcessContext", "NodeDB", "SimulationDB")),
        "optimization_advisor": SubagentDefinition("optimization_advisor", "Review objectives, Pareto evidence, convergence and budget.", ("read_yaml", "read_simulation_db", "read_checkpoint", "plan_optimization"), 10, 8, True, False, ("yaml", "simulation_db", "history"), ("yaml", "SimulationDB", "checkpoint")),
        "boundary_advisor": SubagentDefinition("boundary_advisor", "Recommend soft search regions while preserving hard bounds.", ("read_yaml", "read_process_context"), 6, 4, False, False, ("yaml", "process_context"), ("yaml", "ProcessContext")),
        "kinetics_advisor": SubagentDefinition("kinetics_advisor", "Review kinetics fits and parameter evidence without writing or simulating.", ("read_files", "read_simulation_db"), 8, 6, True, False, ("files", "simulation_db"), ("files", "SimulationDB")),
        "risk_reviewer": SubagentDefinition("risk_reviewer", "Review constraints, failure modes and action impact.", ("read_process_context", "read_checkpoint"), 6, 4, True, False, ("process_context", "history"), ("ProcessContext", "checkpoint")),
        "code_engineer": SubagentDefinition("code_engineer", "Analyze code and produce modification proposals without writing.", ("read_files",), 8, 5, False, False, ("files",), ("files", "user")),
        "report_writer": SubagentDefinition("report_writer", "Draft reports from validated Evidence only.", ("read_evidence", "read_files"), 8, 5, False, False, ("files", "history"), ("Evidence", "files")),
        "general_agent": SubagentDefinition("general_agent", "Handle general questions without execution tools.", (), 4, 2, True, False, ("none",), ("user",)),
    }
    from functools import partial
    from .tool_runtime import default_tools
    registry.tools = tools or default_tools(definitions)
    for name, definition in definitions.items():
        registry.register(name, partial(_reasoning_agent, llm_config=llm_config, llm_call=llm_call, definition=definition, tools=registry.tools), definition=definition)
    return registry


def default_tools_for_roles():
    return default_registry().tools
