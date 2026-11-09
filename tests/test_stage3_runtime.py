"""Stage 3 bounded orchestration tests with fake Subagents."""
from __future__ import annotations

import asyncio
import time

from src.agents.orchestration import OrchestrationRuntime, RuntimePolicy, TaskResult, TaskSpec, TaskStatus
from src.agents.orchestration.models import Evidence
from src.agents import SessionState


def test_retry_is_bounded_and_audit_records_attempts():
    calls = []

    def flaky(task, **kwargs):
        calls.append(kwargs["attempt"])
        if len(calls) == 1:
            raise RuntimeError("transient")
        return TaskResult(task.task_id, task.agent, summary="recovered")

    runtime = OrchestrationRuntime({"flaky": flaky}, policy=RuntimePolicy(max_retries=2, total_budget=2))
    result = runtime.run([TaskSpec(agent="flaky", retry_attempts=2, budget=2)])[0]
    assert result.status is TaskStatus.COMPLETED
    assert calls == [1, 2]
    assert any(e["event"] == "task_retry" for e in runtime.audit)


def test_timeout_is_structured_and_does_not_block_other_task():
    def slow(task, **kwargs):
        time.sleep(0.2)
        return TaskResult(task.task_id, task.agent, summary="late")

    def fast(task, **kwargs):
        return TaskResult(task.task_id, task.agent, summary="ok")

    runtime = OrchestrationRuntime({"slow": slow, "fast": fast}, policy=RuntimePolicy(max_concurrency=2, total_budget=2))
    slow_result, fast_result = runtime.run([
        TaskSpec(agent="slow", timeout_seconds=0.02), TaskSpec(agent="fast")
    ])
    assert slow_result.status is TaskStatus.TIMEOUT
    assert fast_result.status is TaskStatus.COMPLETED


def test_cancel_before_dispatch_is_structured():
    task = TaskSpec(agent="worker")
    runtime = OrchestrationRuntime({"worker": lambda task, **_: TaskResult(task.task_id, task.agent, summary="bad")})
    runtime.cancel(task.task_id)
    result = runtime.run([task])[0]
    assert result.status is TaskStatus.CANCELLED
    assert any(e["event"] == "task_cancel_requested" for e in runtime.audit)


def test_recursion_and_duplicate_signature_are_rejected():
    def worker(task, **kwargs):
        return TaskResult(task.task_id, task.agent, summary="should not run")

    policy = RuntimePolicy(allow_subagent_delegation=True, max_recursion_depth=1, total_budget=4)
    runtime = OrchestrationRuntime({"worker": worker}, policy=policy)
    deep = TaskSpec(agent="worker", prompt="deep", recursion_depth=2)
    one = TaskSpec(agent="worker", prompt="same")
    two = TaskSpec(agent="worker", prompt="same")
    results = runtime.run([deep, one, two])
    assert results[0].status is TaskStatus.CANCELLED
    assert results[2].status is TaskStatus.CANCELLED
    assert any(e["event"] == "task_loop_detected" for e in runtime.audit)


def test_resident_state_and_tool_allowlist_are_passed_to_subagent():
    seen = []

    def worker(task, **kwargs):
        seen.append(kwargs)
        return TaskResult(task.task_id, task.agent, summary="ok")

    runtime = OrchestrationRuntime({"worker": worker}, policy=RuntimePolicy(total_budget=2))
    task = TaskSpec(agent="worker", tool_allowlist=["read_node_db"])
    runtime.run([task])
    assert seen[0]["tool_allowlist"] == ("read_node_db",)
    assert runtime.resident_state()["worker"]["status"] == "completed"


def test_conflicting_evidence_is_reported():
    a = TaskResult("a", "a", summary="a", evidence=[Evidence("tool", "same", "value", 1)])
    b = TaskResult("b", "b", summary="b", evidence=[Evidence("tool", "same", "value", 2)])
    issues = OrchestrationRuntime.validate_results([a, b])
    assert any("conflicting evidence" in item for item in issues)


def test_runtime_checkpoint_restores_resident_results_and_audit():
    calls = []

    def worker(task, **kwargs):
        calls.append(kwargs.get("resident_state"))
        return TaskResult(task.task_id, task.agent, summary="persisted")

    from pathlib import Path
    checkpoint = Path("runtime_stage3_checkpoint.db")
    task = TaskSpec(agent="worker", prompt="persist", task_type="query")
    try:
        first = OrchestrationRuntime({"worker": worker}, policy=RuntimePolicy(total_budget=2), checkpoint_path=checkpoint, session_id="s1")
        assert first.run([task])[0].summary == "persisted"
        second = OrchestrationRuntime({"worker": worker}, policy=RuntimePolicy(total_budget=2), checkpoint_path=checkpoint, resume=True)
        assert second.resident_state()["worker"]["summary"] == "persisted"
        assert second.run([task])[0].summary == "persisted"
        assert len(calls) == 1
    finally:
        checkpoint.unlink(missing_ok=True)


def test_async_runtime_adapter_and_result_summary():
    def worker(task, **kwargs):
        return TaskResult(task.task_id, task.agent, summary="ok")

    async def run():
        runtime = OrchestrationRuntime({"worker": worker}, policy=RuntimePolicy(total_budget=2))
        result = await runtime.arun([TaskSpec(agent="worker")])
        return runtime.summarize_results(result)

    summary = asyncio.run(run())
    assert summary["completed"] == 1 and summary["failed"] == 0


def test_session_state_round_trip_keeps_resident_runtime_state():
    state = SessionState(runtime_state={"process_advisor": {"summary": "prior"}})
    restored = SessionState.from_json(state.to_json())
    assert restored.runtime_state["process_advisor"]["summary"] == "prior"
