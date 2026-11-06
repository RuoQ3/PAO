"""Bounded, failure-isolating execution runtime for Subagents."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .models import RuntimePolicy, TaskResult, TaskSpec, TaskStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrchestrationRuntime:
    """Dispatches registered handlers without making domain decisions."""

    def __init__(self, registry: Mapping[str, Callable[..., TaskResult]] | None = None,
                 *, policy: RuntimePolicy | None = None, audit: list[dict[str, Any]] | None = None):
        self.registry = dict(registry or {})
        self.policy = policy or RuntimePolicy()
        self.audit = audit if audit is not None else []
        self._cancelled: set[str] = set()

    def cancel(self, task_id: str) -> None:
        self._cancelled.add(task_id)

    def run(self, tasks: list[TaskSpec], *, context: Any = None) -> list[TaskResult]:
        tasks = self._normalize(tasks)
        results: dict[str, TaskResult] = {}
        remaining_budget = self.policy.total_budget
        pending = list(tasks)
        while pending:
            ready = [t for t in pending if all(dep in results for dep in t.dependencies)]
            if not ready:
                for task in pending:
                    results[task.task_id] = TaskResult.failed(task, "dependency cycle or missing dependency")
                break
            pending = [t for t in pending if t not in ready]
            allowed = ready[: max(0, remaining_budget)]
            for task in ready[len(allowed):]:
                results[task.task_id] = TaskResult.failed(task, "runtime budget exhausted", status=TaskStatus.CANCELLED)
            if not allowed:
                continue
            with ThreadPoolExecutor(max_workers=self.policy.max_concurrency) as pool:
                futures = {pool.submit(self._execute, task, context): task for task in allowed}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result(timeout=task.timeout_seconds)
                    except TimeoutError:
                        result = TaskResult.failed(task, "task timeout")
                    except Exception as exc:  # isolate one specialist failure
                        result = TaskResult.failed(task, f"{type(exc).__name__}: {exc}")
                    results[task.task_id] = result
                    remaining_budget -= max(1, min(task.budget, self.policy.max_steps_per_task))
        return [results[t.task_id] for t in tasks]

    def _normalize(self, tasks: list[TaskSpec]) -> list[TaskSpec]:
        if len(tasks) > self.policy.max_tasks:
            tasks = tasks[: self.policy.max_tasks]
        for task in tasks:
            task.max_steps = min(max(1, task.max_steps), self.policy.max_steps_per_task)
            task.budget = min(max(1, task.budget), task.max_steps)
            task.allow_subtasks = task.allow_subtasks and self.policy.allow_subagent_delegation
        return tasks

    def _execute(self, task: TaskSpec, context: Any) -> TaskResult:
        started = _now()
        self.audit.append({"event": "task_started", "task_id": task.task_id, "agent": task.agent, "at": started})
        if task.task_id in self._cancelled:
            result = TaskResult.failed(task, "task cancelled", status=TaskStatus.CANCELLED)
        elif task.agent not in self.registry:
            result = TaskResult.failed(task, f"unknown subagent: {task.agent}")
        else:
            handler = self.registry[task.agent]
            try:
                result = handler(task, context=context, tool_allowlist=tuple(task.tool_allowlist), max_steps=task.max_steps)
                if not isinstance(result, TaskResult):
                    raise TypeError("subagent must return TaskResult")
            except Exception as exc:
                result = TaskResult.failed(task, f"{type(exc).__name__}: {exc}")
        self.audit.append({"event": "task_finished", "task_id": task.task_id, "status": result.status.value, "at": _now(), "errors": list(result.errors)})
        return result

    @staticmethod
    def validate_results(results: list[TaskResult]) -> list[str]:
        issues: list[str] = []
        for result in results:
            if result.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
                issues.append(f"{result.task_id}: {result.status.value}")
            if result.status == TaskStatus.COMPLETED and not result.summary and not result.facts and not result.evidence:
                issues.append(f"{result.task_id}: completed result has no facts, evidence, or summary")
            for item in result.evidence:
                if not item.source or not item.source_ref or not item.claim:
                    issues.append(f"{result.task_id}: incomplete evidence")
        return issues
