"""Stage 5 controlled role registry tests with fake read-only context."""
from types import SimpleNamespace

from src.agents.orchestration.models import IntentSchema, SubagentRequest, TaskStatus
from src.agents.orchestration.runtime import OrchestrationRuntime, TaskPlanner
from src.agents.subagents.registry import SubagentDefinition, default_registry


ROLES = {
    "project_researcher", "process_advisor", "optimization_advisor",
    "boundary_advisor", "kinetics_advisor", "risk_reviewer",
    "code_engineer", "report_writer", "general_agent",
}


def _context():
    return SimpleNamespace(
        variables={"rr": {"initial_value": 2.0, "lower_bound": 1.0, "upper_bound": 4.0}},
        objectives=[{"name": "purity"}], constraints=[{"name": "pressure"}],
        equipment=[{"name": "T1"}], streams=[{"name": "S1"}],
        history_evidence=[{"case_id": "c1"}], diagnostics=[],
    )


def test_default_registry_registers_all_roles_with_metadata():
    registry = default_registry()
    assert set(registry.names()) == ROLES
    for name in ROLES:
        definition = registry.definition(name)
        assert isinstance(definition, SubagentDefinition)
        assert definition.description
        assert definition.max_steps > 0 and definition.budget > 0
        assert definition.allow_subtasks is False
        assert definition.input_context
        assert definition.evidence_sources


def test_each_role_returns_structured_read_only_task_result():
    registry = default_registry()
    for name in sorted(ROLES):
        from src.agents.orchestration.models import TaskSpec
        result = registry.get(name)(TaskSpec(agent=name, task_type="project_query", prompt="review"), context=_context(), tool_allowlist=())
        assert result.agent == name
        assert result.status is TaskStatus.COMPLETED
        assert result.evidence
        assert all(item.source != "AspenCOM" for item in result.evidence)
        if name != "general_agent":
            assert any(isinstance(item, dict) and item.get("kind") == "role_behavior" for item in result.facts)


def test_code_engineer_and_boundary_are_proposal_only():
    registry = default_registry()
    from src.agents.orchestration.models import TaskSpec
    code = registry.get("code_engineer")(TaskSpec(agent="code_engineer", prompt="change code"), context=_context(), tool_allowlist=())
    boundary = registry.get("boundary_advisor")(TaskSpec(agent="boundary_advisor", prompt="review bounds"), context=_context(), tool_allowlist=())
    assert any(f.get("write_performed") is False for f in code.facts)
    assert "hard bounds" in boundary.summary


def test_runtime_applies_role_limits_and_disables_subtasks():
    registry = default_registry()
    intent = IntentSchema(
        intent="project_query", task_type="project_query", project_scope="current_project",
        context_required=["process_context"],
        subagents=[SubagentRequest("process_advisor", "inspect")],
        action_type="read", confidence=0.9,
    )
    task = TaskPlanner().plan(intent, "inspect") [0]
    task.max_steps = 99
    task.budget = 99
    task.allow_subtasks = True
    runtime = OrchestrationRuntime(registry.as_mapping(), role_definitions=registry.definitions())
    normalized = runtime._normalize([task])[0]
    definition = registry.definition("process_advisor")
    assert normalized.max_steps <= definition.max_steps
    assert normalized.budget <= definition.budget
    assert normalized.allow_subtasks is False


def test_role_handler_failure_is_structured_and_runtime_survives():
    registry = default_registry()
    result = registry.get("process_advisor")(
        __import__("src.agents.orchestration.models", fromlist=["TaskSpec"]).TaskSpec(agent="process_advisor", prompt="inspect"),
        context=None, tool_allowlist=(),
    )
    assert result.status is TaskStatus.COMPLETED
    assert result.uncertainties
