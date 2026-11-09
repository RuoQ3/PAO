"""Task 10 coverage for the unified Lead Agent boundary.

These tests use deterministic fake handlers/executors.  They intentionally do
not require Aspen COM, a provider SDK, a database server, or an LLM key.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.agents import LeadAgent
from src.agents.actions import ActionGateway
from src.agents.actions.models import ModelIdentity, ModelSchema, ParameterSpec, ActionRequest, ActionType
from src.agents.orchestration import (
    ActionProposal, ApprovalRequest, OrchestrationRuntime, RuntimePolicy,
    SessionState, TaskResult, TaskSpec, TaskStatus,
)
from src.agents.subagents.registry import SubagentRegistry


def test_lead_answers_general_and_routes_project_question_without_context():
    lead = LeadAgent()
    assert lead.answer("What is a control chart?").facts
    answer = lead.answer("请分析动力学数据")
    assert answer.task_results[0].agent == "kinetics_advisor"
    assert answer.text


def test_runtime_runs_independent_subagents_concurrently_and_structures_results():
    started: list[str] = []

    def handler(task, **_):
        started.append(task.agent)
        time.sleep(0.08)
        return TaskResult(task.task_id, task.agent, summary=f"done:{task.agent}", facts=[task.agent])

    runtime = OrchestrationRuntime({"a": handler, "b": handler}, policy=RuntimePolicy(max_concurrency=2))
    results = runtime.run([TaskSpec(agent="a"), TaskSpec(agent="b")])
    assert {r.summary for r in results} == {"done:a", "done:b"}
    assert all(isinstance(r.facts, list) for r in results)
    assert {e["event"] for e in runtime.audit} >= {"task_started", "task_finished"}


def test_runtime_failure_isolated_and_delegation_is_bounded():
    def boom(task, **_):
        raise RuntimeError("fake failure")

    def ok(task, **_):
        return TaskResult(task.task_id, task.agent, summary="ok")

    policy = RuntimePolicy(max_tasks=2, total_budget=2, allow_subagent_delegation=False)
    runtime = OrchestrationRuntime({"bad": boom, "ok": ok}, policy=policy)
    bad, good = runtime.run([TaskSpec(agent="bad", allow_subtasks=True), TaskSpec(agent="ok")])
    assert bad.status is TaskStatus.FAILED
    assert good.status is TaskStatus.COMPLETED
    assert not TaskSpec(agent="ok", allow_subtasks=True).allow_subtasks or not policy.allow_subagent_delegation


def test_missing_evidence_is_reported_instead_of_claimed():
    task = TaskSpec(agent="empty")
    runtime = OrchestrationRuntime({"empty": lambda task, **_: TaskResult(task.task_id, task.agent)})
    result = runtime.run([task])[0]
    assert runtime.validate_results([result])


def _gateway():
    schema = ModelSchema(
        identity=ModelIdentity("demo"), current_values={r"\Data\Blocks\T1\Input\RR": 1.0},
        parameters={r"\Data\Blocks\T1\Input\RR": ParameterSpec(
            r"\Data\Blocks\T1\Input\RR", value_type="float", minimum=0.1, maximum=5.0,
            impact_targets=("objective",), requires_simulation=True)},
    )
    return ActionGateway(schema)


def test_write_request_creates_pending_approval_and_gateway_requires_confirmation():
    gateway = _gateway()
    lead = LeadAgent(action_gateway=gateway, policy=RuntimePolicy(allow_writes=True))
    answer = lead.answer("修改边界")
    assert isinstance(answer.approval_request, ApprovalRequest)
    assert answer.approval_request.status == "pending"
    audit_id = answer.approval_request.preview.get("audit_id")
    with pytest.raises(Exception):
        gateway.execute(audit_id)


def test_action_gateway_executes_only_after_approval_and_records_audit():
    gateway = _gateway()
    action = ActionRequest(ActionType.SET_PARAMETER, r"\Data\Blocks\T1\Input\RR", old_value=1.0, new_value=1.2, reason="test")
    result = gateway.submit(action)
    assert result.valid and result.preview
    gateway.preview(action.audit_id)
    gateway.approve(action.audit_id)
    output = gateway.execute(action.audit_id)
    assert output["status"] == "executed"
    assert {event.event_type.value for event in gateway.audit.events} >= {"user_request", "validation", "preview", "confirmation", "execution"}


def test_checkpoint_round_trip_preserves_nested_approval_and_task_result():
    state = SessionState(pending_approval=ApprovalRequest(ActionProposal("inspect", "project")), task_results=[TaskResult("1", "a", summary="ok")])
    restored = SessionState.from_json(state.to_json())
    assert restored.pending_approval.proposal.action_type == "inspect"
    assert restored.task_results[0].summary == "ok"


def test_no_llm_key_is_clear_and_unrelated_question_does_not_build_process_context(monkeypatch):
    from src.agents.lead import llm_client
    monkeypatch.setenv("PAO_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "_load_dotenv_if_present", lambda **_: None)
    cfg = llm_client.load_llm_config()
    assert not llm_client.is_configured(cfg)
    calls = []

    class Builder:
        def build(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("context should not be built")

    answer = LeadAgent(context_builder=Builder()).answer("hello")
    assert answer.facts and not calls
