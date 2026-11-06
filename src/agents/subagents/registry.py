from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.agents.orchestration.models import Evidence, TaskResult, TaskSpec


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
    facts = [{"kind": "context_summary", "variables": len(variables),
              "objectives": len(objectives), "constraints": len(constraints)}]
    evidence = [Evidence("tool", "ProcessContext", "Read-only project context loaded",
                         {"variables": len(variables), "objectives": len(objectives),
                          "constraints": len(constraints)}, confidence=1.0)]
    return TaskResult(task.task_id, task.agent,
                      summary=f"{task.agent} reviewed the available project context.",
                      facts=facts, evidence=evidence,
                      uncertainties=list(getattr(context, "diagnostics", []) or []))


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
        return TaskResult(task.task_id, task.agent, summary="Boundary advisor completed a deterministic review.",
                          facts=[report.to_dict() if hasattr(report, "to_dict") else str(report)],
                          evidence=[Evidence("model", "boundary_advisor", "Boundary recommendation generated", report.to_dict() if hasattr(report, "to_dict") else str(report), confidence=1.0)],
                          risks=list(getattr(report, "warnings", []) or []))
    except Exception as exc:
        return TaskResult.failed(task, f"boundary advisor: {exc}")


def _optimization(task: TaskSpec, *, context: Any = None, **kwargs: Any) -> TaskResult:
    result = _context_agent(task, context=context, **kwargs)
    result.summary = "Optimization advisor reviewed objectives, constraints and available history."
    result.recommended_actions.append({"action": "review_feasibility_and_budget", "requires_confirmation": True})
    return result


class SubagentRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., TaskResult]] = {}

    def register(self, name: str, handler: Callable[..., TaskResult], *, replace: bool = False) -> None:
        if name in self._handlers and not replace:
            raise ValueError(f"subagent already registered: {name}")
        self._handlers[name] = handler

    def get(self, name: str) -> Callable[..., TaskResult]:
        return self._handlers[name]

    def as_mapping(self) -> Mapping[str, Callable[..., TaskResult]]:
        return dict(self._handlers)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))


def default_registry() -> SubagentRegistry:
    registry = SubagentRegistry()
    registry.register("general_agent", _general)
    registry.register("boundary_advisor", _boundary)
    registry.register("optimization_advisor", _optimization)
    for name in ("project_researcher", "process_advisor", "kinetics_advisor",
                 "risk_reviewer", "code_engineer", "report_writer"):
        registry.register(name, _context_agent)
    return registry
