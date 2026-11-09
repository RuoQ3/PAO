"""Bounded, auditable execution runtime for PAO Subagents."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping

from .models import IntentSchema, RuntimePolicy, TaskResult, TaskSpec, TaskStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskPlanner:
    """Convert a validated IntentSchema into bounded TaskSpec objects."""

    def __init__(self, *, policy: RuntimePolicy | None = None):
        self.policy = policy or RuntimePolicy()

    def plan(self, intent: IntentSchema, prompt: str) -> list[TaskSpec]:
        planned: list[TaskSpec] = []
        previous: str | None = None
        for request in intent.subagents[: self.policy.max_tasks]:
            dependencies = [] if request.parallel or previous is None else [previous]
            task = TaskSpec(
                task_type=intent.task_type,
                prompt=f"{prompt}\n\nTask: {request.task}" if request.task else prompt,
                agent=request.name,
                dependencies=dependencies,
                tool_allowlist=self._tools_for(request.name, intent),
                max_steps=self.policy.max_steps_per_task,
                budget=min(self.policy.max_steps_per_task, max(1, self.policy.total_budget // max(1, len(intent.subagents)))),
                allow_subtasks=self.policy.allow_subagent_delegation,
                metadata={"priority": request.priority, "parallel": request.parallel},
            )
            planned.append(task)
            previous = task.task_id
        if not planned:
            planned.append(TaskSpec(task_type=intent.task_type, prompt=prompt, agent="general_agent", budget=1))
        return planned

    @staticmethod
    def _tools_for(agent: str, intent: IntentSchema) -> list[str]:
        if intent.action_type != "read":
            return ["read_context"]
        return {
            "project_researcher": ["read_files", "read_yaml", "read_node_db", "read_simulation_db"],
            "process_advisor": ["read_process_context", "read_node_db", "read_simulation_db"],
            "optimization_advisor": ["read_yaml", "read_simulation_db", "read_checkpoint"],
            "boundary_advisor": ["read_yaml", "read_process_context"],
            "kinetics_advisor": ["read_files", "read_simulation_db"],
            "risk_reviewer": ["read_process_context", "read_checkpoint"],
            "code_engineer": ["read_files"],
            "report_writer": ["read_evidence", "read_files"],
        }.get(agent, ["read_context"])


class OrchestrationRuntime:
    """Dispatch registered handlers with bounded concurrency and budgets."""

    def __init__(self, registry: Mapping[str, Callable[..., TaskResult]] | None = None,
                 *, policy: RuntimePolicy | None = None,
                 audit: list[dict[str, Any]] | None = None,
                 role_definitions: Mapping[str, Any] | None = None,
                 resident: Mapping[str, Any] | None = None,
                 checkpoint_path: str | Path | None = None,
                 resume: bool = False,
                 session_id: str | None = None):
        self.registry = dict(registry or {})
        self.role_definitions = dict(role_definitions or {})
        self.policy = policy or RuntimePolicy()
        self.audit = audit if audit is not None else []
        self._cancelled: set[str] = set()
        self._resident: dict[str, Any] = dict(resident or {})
        self._lock = threading.Lock()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.session_id = session_id or hashlib.sha256(str(self.checkpoint_path or id(self)).encode()).hexdigest()[:16]
        self._checkpoint_results: dict[str, TaskResult] = {}
        if resume and self.checkpoint_path:
            self._restore_checkpoint()

    def cancel(self, task_id: str) -> None:
        self._cancelled.add(task_id)
        self._record("task_cancel_requested", task_id=task_id)

    def resident_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._resident)

    async def arun(self, tasks: list[TaskSpec], *, context: Any = None) -> list[TaskResult]:
        """Async adapter for callers whose event loop must remain responsive."""
        return await asyncio.to_thread(self.run, tasks, context=context)

    def save_checkpoint(self, tasks: list[TaskSpec], results: list[TaskResult], *, status: str = "completed") -> None:
        """Persist serializable runtime state; no COM or live context is stored."""
        if self.checkpoint_path is None:
            return
        payload = {
            "version": 1, "session_id": self.session_id, "status": status,
            "tasks": [task.to_dict() for task in tasks],
            "results": [result.to_dict() for result in results],
            "resident": self.resident_state(), "audit": list(self.audit),
            "budget": self.policy.total_budget,
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.checkpoint_path)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS runtime_checkpoint (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            with db:
                db.execute("INSERT OR REPLACE INTO runtime_checkpoint VALUES (1, ?)",
                           (json.dumps(payload, ensure_ascii=False, allow_nan=False),))
        finally:
            db.close()

    def checkpoint_state(self) -> dict[str, Any] | None:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return None
        db = sqlite3.connect(self.checkpoint_path)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS runtime_checkpoint (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            row = db.execute("SELECT payload FROM runtime_checkpoint WHERE id=1").fetchone()
            return json.loads(row[0]) if row else None
        finally:
            db.close()

    def _restore_checkpoint(self) -> None:
        payload = self.checkpoint_state()
        if not payload:
            return
        self.session_id = str(payload.get("session_id") or self.session_id)
        self._resident.update(payload.get("resident") or {})
        self._checkpoint_results = {
            item["task_id"]: TaskResult.from_dict(item)
            for item in payload.get("results", []) if isinstance(item, dict) and item.get("task_id")
        }
        self.audit.extend(payload.get("audit") or [])

    def run(self, tasks: list[TaskSpec], *, context: Any = None) -> list[TaskResult]:
        tasks = self._normalize(tasks)
        results: dict[str, TaskResult] = dict(self._checkpoint_results)
        tasks = [task for task in tasks if task.task_id not in results]
        if not tasks:
            return list(self._checkpoint_results.values())
        remaining_budget = self.policy.total_budget
        pending = list(tasks)
        while pending:
            ready = [t for t in pending if all(dep in results for dep in t.dependencies)]
            if not ready:
                for task in pending:
                    results[task.task_id] = TaskResult.failed(task, "dependency cycle or missing dependency")
                    self._record("task_rejected", task_id=task.task_id, reason="dependency cycle or missing dependency")
                break
            pending = [t for t in pending if t not in ready]
            allowed: list[TaskSpec] = []
            for task in ready:
                cost = self._cost(task)
                if remaining_budget >= cost:
                    allowed.append(task)
                    remaining_budget -= cost
                else:
                    results[task.task_id] = TaskResult.failed(task, "runtime budget exhausted", status=TaskStatus.CANCELLED)
                    self._record("task_rejected", task_id=task.task_id, reason="runtime budget exhausted")
            if not allowed:
                continue
            pool = ThreadPoolExecutor(max_workers=min(self.policy.max_concurrency, len(allowed)))
            futures = {pool.submit(self._execute_with_retry, task, context): task for task in allowed}
            deadlines = {future: time.monotonic() + max(0.001, task.timeout_seconds) * self._attempts(task)
                         for future, task in futures.items()}
            unfinished = set(futures)
            try:
                while unfinished:
                    wait_for = min(max(0.0, deadlines[f] - time.monotonic()) for f in unfinished)
                    done, _ = wait(unfinished, timeout=wait_for, return_when=FIRST_COMPLETED)
                    if not done:
                        expired = {f for f in unfinished if deadlines[f] <= time.monotonic()}
                        for future in expired:
                            task = futures[future]
                            future.cancel()
                            results[task.task_id] = TaskResult.failed(task, "task timeout", status=TaskStatus.TIMEOUT)
                            self._record("task_timeout", task_id=task.task_id)
                        unfinished -= expired
                        continue
                    for future in done:
                        task = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = TaskResult.failed(task, f"{type(exc).__name__}: {exc}")
                        results[task.task_id] = result
                        self._record("task_finished", task_id=task.task_id, agent=task.agent,
                                     status=result.status.value, errors=list(result.errors))
                    unfinished -= done
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        ordered = [results[t.task_id] for t in tasks]
        self._checkpoint_results.update(results)
        self.save_checkpoint(tasks, ordered, status="completed")
        return ordered

    def _normalize(self, tasks: list[TaskSpec]) -> list[TaskSpec]:
        normalized = list(tasks[: self.policy.max_tasks])
        signatures: set[str] = set()
        for task in normalized:
            definition = self.role_definitions.get(task.agent)
            if definition is not None:
                task.max_steps = min(task.max_steps, int(getattr(definition, "max_steps", task.max_steps)))
                task.budget = min(task.budget, int(getattr(definition, "budget", task.budget)))
                task.allow_subtasks = bool(task.allow_subtasks and getattr(definition, "allow_subtasks", False))
                allowed = set(getattr(definition, "tool_allowlist", ()) or ())
                task.tool_allowlist = [tool for tool in task.tool_allowlist if tool in allowed]
            task.max_steps = min(max(1, int(task.max_steps)), self.policy.max_steps_per_task)
            task.budget = min(max(1, int(task.budget)), task.max_steps)
            task.retry_attempts = min(max(0, int(task.retry_attempts)), self.policy.max_retries)
            task.recursion_depth = max(0, int(task.recursion_depth))
            task.allow_subtasks = bool(task.allow_subtasks) and self.policy.allow_subagent_delegation and task.recursion_depth < self.policy.max_recursion_depth
            signature = self._signature(task)
            if signature in signatures:
                self._record("task_loop_detected", task_id=task.task_id, agent=task.agent)
                task.metadata = {**task.metadata, "loop_detected": True}
            signatures.add(signature)
        return normalized

    def _execute_with_retry(self, task: TaskSpec, context: Any) -> TaskResult:
        if task.metadata.get("loop_detected"):
            return TaskResult.failed(task, "duplicate task signature detected", status=TaskStatus.CANCELLED)
        last: TaskResult | None = None
        for attempt in range(self._attempts(task)):
            if task.task_id in self._cancelled:
                return TaskResult.failed(task, "task cancelled", status=TaskStatus.CANCELLED)
            self._record("task_started", task_id=task.task_id, agent=task.agent, attempt=attempt + 1)
            result = self._execute(task, context, attempt=attempt + 1)
            last = result
            if result.status in {TaskStatus.COMPLETED, TaskStatus.PARTIAL}:
                return result
            if attempt + 1 < self._attempts(task):
                self._record("task_retry", task_id=task.task_id, attempt=attempt + 2, reason=list(result.errors))
        return last or TaskResult.failed(task, "task did not execute")

    def _execute(self, task: TaskSpec, context: Any, *, attempt: int) -> TaskResult:
        if task.task_id in self._cancelled:
            return TaskResult.failed(task, "task cancelled", status=TaskStatus.CANCELLED)
        if task.recursion_depth > self.policy.max_recursion_depth:
            return TaskResult.failed(task, "maximum recursion depth exceeded", status=TaskStatus.CANCELLED)
        handler = self.registry.get(task.agent)
        if handler is None:
            return TaskResult.failed(task, f"unknown subagent: {task.agent}")
        try:
            result = handler(task, context=context, tool_allowlist=tuple(task.tool_allowlist),
                            max_steps=task.max_steps, max_tool_calls=self.policy.max_tool_calls_per_task,
                            resident_state=self._resident.get(task.agent), attempt=attempt,
                            is_tool_allowed=lambda name: self.is_tool_allowed(task, name))
            if not isinstance(result, TaskResult):
                raise TypeError("subagent must return TaskResult")
            with self._lock:
                self._resident[task.agent] = {"task_id": task.task_id, "status": result.status.value,
                                              "summary": result.summary[:1000], "updated_at": _now()}
            return result
        except Exception as exc:
            return TaskResult.failed(task, f"{type(exc).__name__}: {exc}")

    def _attempts(self, task: TaskSpec) -> int:
        return 1 + min(max(0, task.retry_attempts), self.policy.max_retries)

    @staticmethod
    def _cost(task: TaskSpec) -> int:
        return max(1, min(int(task.budget), int(task.max_steps)))

    @staticmethod
    def _signature(task: TaskSpec) -> str:
        raw = "|".join((task.agent, task.task_type, task.prompt.strip(), ",".join(sorted(task.dependencies))))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _record(self, event: str, **payload: Any) -> None:
        self.audit.append({"event": event, "at": _now(), **payload})

    @staticmethod
    def rank_results(results: list[TaskResult]) -> list[TaskResult]:
        """Put usable, evidence-backed results before failures and empty output."""
        return sorted(results, key=lambda result: (
            result.status not in {TaskStatus.COMPLETED, TaskStatus.PARTIAL},
            not bool(result.evidence), not bool(result.summary), result.agent,
        ))

    @classmethod
    def summarize_results(cls, results: list[TaskResult]) -> dict[str, Any]:
        ranked = cls.rank_results(results)
        return {
            "results": [item.to_dict() for item in ranked],
            "issues": cls.validate_results(ranked),
            "completed": sum(item.status is TaskStatus.COMPLETED for item in ranked),
            "failed": sum(item.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT} for item in ranked),
        }

    @staticmethod
    def is_tool_allowed(task: TaskSpec, tool_name: str) -> bool:
        """Check the task allowlist; an empty list means no optional tools."""
        return isinstance(tool_name, str) and tool_name in set(task.tool_allowlist)

    @staticmethod
    def validate_results(results: list[TaskResult]) -> list[str]:
        issues: list[str] = []
        claims: dict[str, Any] = {}
        for result in results:
            if result.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}:
                issues.append(f"{result.task_id}: {result.status.value}")
            if result.status == TaskStatus.COMPLETED and not result.summary and not result.facts and not result.evidence:
                issues.append(f"{result.task_id}: completed result has no facts, evidence, or summary")
            for item in result.evidence:
                if not item.source or not item.source_ref or not item.claim:
                    issues.append(f"{result.task_id}: incomplete evidence")
                key = f"{item.source_ref}:{item.claim}"
                if key in claims and claims[key] != item.value:
                    issues.append(f"conflicting evidence: {key}")
                else:
                    claims[key] = item.value
        return issues
