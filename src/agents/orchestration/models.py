"""Stable, JSON-safe contracts shared by Lead Agent, Runtime and Subagents."""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import Enum
from typing import Any
import json
import uuid


class TaskStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class ExecutionMode(str, Enum):
    INSTANT = "instant"
    BRIEF = "brief"
    STANDARD = "standard"
    DEEP = "deep"


def _json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


class JsonModel:
    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kwargs)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]):
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.__name__} requires an object")
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        data = dict(raw)
        if cls is TaskResult:
            data["status"] = TaskStatus(data.get("status", TaskStatus.COMPLETED))
            data["evidence"] = [Evidence.from_dict(x) if isinstance(x, dict) else x for x in data.get("evidence", [])]
        elif cls is RuntimePolicy:
            data["mode"] = ExecutionMode(data.get("mode", ExecutionMode.STANDARD))
        elif cls is SessionState:
            data["mode"] = ExecutionMode(data.get("mode", ExecutionMode.STANDARD))
            data["task_results"] = [TaskResult.from_dict(x) if isinstance(x, dict) else x for x in data.get("task_results", [])]
        elif cls is Answer:
            data["evidence"] = [Evidence.from_dict(x) if isinstance(x, dict) else x for x in data.get("evidence", [])]
            data["task_results"] = [TaskResult.from_dict(x) if isinstance(x, dict) else x for x in data.get("task_results", [])]
        return cls(**data)

    @classmethod
    def from_json(cls, raw: str):
        return cls.from_dict(json.loads(raw))


@dataclass
class Evidence(JsonModel):
    source: str
    source_ref: str
    claim: str
    value: Any = None
    unit: str | None = None
    confidence: float | None = None


@dataclass
class TaskSpec(JsonModel):
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_type: str = "general_question"
    prompt: str = ""
    agent: str = "general_agent"
    dependencies: list[str] = field(default_factory=list)
    tool_allowlist: list[str] = field(default_factory=list)
    max_steps: int = 8
    timeout_seconds: float = 60.0
    budget: int = 8
    allow_subtasks: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult(JsonModel):
    task_id: str
    agent: str
    status: TaskStatus = TaskStatus.COMPLETED
    summary: str = ""
    facts: list[Any] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommended_actions: list[Any] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    needs_user_confirmation: bool = False
    errors: list[str] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def failed(cls, task: TaskSpec, error: str, *, status: TaskStatus = TaskStatus.FAILED) -> "TaskResult":
        return cls(task_id=task.task_id, agent=task.agent, status=status, errors=[error])


@dataclass
class RuntimePolicy(JsonModel):
    mode: ExecutionMode = ExecutionMode.STANDARD
    max_tasks: int = 8
    max_concurrency: int = 4
    max_steps_per_task: int = 8
    total_budget: int = 64
    allow_writes: bool = False
    require_risk_review: bool = True
    allow_subagent_delegation: bool = False

    def __post_init__(self) -> None:
        self.max_tasks = max(1, int(self.max_tasks))
        self.max_concurrency = max(1, int(self.max_concurrency))
        self.max_steps_per_task = max(1, int(self.max_steps_per_task))
        self.total_budget = max(1, int(self.total_budget))
        if self.mode == ExecutionMode.INSTANT:
            self.max_tasks = 1
            self.max_concurrency = 1
            self.allow_writes = False
            self.allow_subagent_delegation = False
        elif self.mode == ExecutionMode.BRIEF:
            self.max_tasks = min(self.max_tasks, 2)
            self.max_concurrency = min(self.max_concurrency, 2)


@dataclass
class ActionProposal(JsonModel):
    action_type: str
    target: str
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    requires_confirmation: bool = True


@dataclass
class ApprovalRequest(JsonModel):
    proposal: ActionProposal
    preview: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    status: str = "pending"


@dataclass
class Answer(JsonModel):
    text: str
    facts: list[Any] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    recommendations: list[Any] = field(default_factory=list)
    approval_request: ApprovalRequest | None = None
    task_results: list[TaskResult] = field(default_factory=list)


@dataclass
class SessionState(JsonModel):
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mode: ExecutionMode = ExecutionMode.STANDARD
    history: list[dict[str, str]] = field(default_factory=list)
    context_refs: dict[str, str] = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: ApprovalRequest | None = None
    budget_used: int = 0

