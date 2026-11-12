"""Stable, JSON-safe contracts shared by Lead Agent, Runtime and Subagents."""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import Enum
from typing import Any
import json
import math
import uuid
from src.agents.answer_content import AnswerBlock, blocks_from, legacy_blocks


class TaskStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


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
        raw = dict(raw)
        if cls is SessionState and (raw.pop("fallback_reason", None) is not None or raw.get("classification_source") not in {None, "llm", "llm_repaired"} or raw.get("agent_unavailable") or
                                   (raw.get("pending_approval") and raw.get("classification_source") not in {"llm", "llm_repaired"})):
            raw.update(agent_unavailable=True, classification_source="agent_unavailable", pending_approval=None, intent=None)
            raw["runtime_state"] = {}
            raw["task_results"] = []
            raw["audit_events"] = [e for e in raw.get("audit_events", []) if e.get("event") in
                                   {"agent_unavailable", "action_executed", "answer_created"}]
        if cls is Classification and (raw.pop("fallback_reason", None) is not None or raw.get("classification_source") not in {"llm", "llm_repaired", None}):
            from src.agents.errors import unavailable
            raise unavailable("agent_output_invalid", stage="classification_restore")
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        data = dict(raw)
        if cls is TaskResult:
            data["status"] = TaskStatus(data.get("status", TaskStatus.COMPLETED))
            data["evidence"] = [Evidence.from_dict(x) if isinstance(x, dict) else x for x in data.get("evidence", [])]
        elif cls is IntentSchema:
            data["subagents"] = [SubagentRequest.from_dict(x) if isinstance(x, dict) else x for x in data.get("subagents", [])]
        elif cls is Classification:
            raw_intent = data.get("intent")
            if isinstance(raw_intent, dict):
                data["intent"] = IntentSchema.from_dict(raw_intent)
        elif cls is RuntimePolicy:
            data["mode"] = ExecutionMode(data.get("mode", ExecutionMode.STANDARD))
        elif cls is SessionState:
            task = data.get("pending_task")
            if isinstance(task, dict):
                data["pending_task"] = PendingTask.from_dict(task)
            # Approvals are process-local gateway capabilities, never restored
            # from a JSON/checkpoint. A restored task must produce a new preview.
            if data.get("pending_task") is not None:
                old_approval = data.get("pending_approval")
                if old_approval:
                    preview = old_approval.get("preview", {}) if isinstance(old_approval, dict) else old_approval.preview
                    if preview.get("audit_id"):
                        data.setdefault("runtime_state", {}).setdefault("revoked_approval_ids", []).append(preview["audit_id"])
                data["pending_approval"] = None
                if data["pending_task"].phase == "awaiting_approval":
                    data["pending_task"].phase = "ready"
                    data["pending_task"].audit.append({"event": "approval_invalidated_on_restore"})
                if data.get("agent_unavailable") or data.get("internal_error"):
                    data["pending_task"].phase = "failed"
            data["mode"] = ExecutionMode(data.get("mode", ExecutionMode.STANDARD))
            data["task_results"] = [TaskResult.from_dict(x) if isinstance(x, dict) else x for x in data.get("task_results", [])]
            intent = data.get("intent")
            if isinstance(intent, dict):
                data["intent"] = IntentSchema.from_dict(intent)
            pending = data.get("pending_approval")
            if isinstance(pending, dict):
                proposal = pending.get("proposal")
                if isinstance(proposal, dict):
                    pending = dict(pending)
                    pending["proposal"] = ActionProposal.from_dict(proposal)
                data["pending_approval"] = ApprovalRequest.from_dict(pending)
        elif cls is Answer:
            data["evidence"] = [Evidence.from_dict(x) if isinstance(x, dict) else x for x in data.get("evidence", [])]
            data["task_results"] = [TaskResult.from_dict(x) if isinstance(x, dict) else x for x in data.get("task_results", [])]
            approval = data.get("approval_request")
            if isinstance(approval, dict):
                approval = dict(approval)
                proposal = approval.get("proposal")
                if isinstance(proposal, dict):
                    approval["proposal"] = ActionProposal.from_dict(proposal)
                data["approval_request"] = ApprovalRequest.from_dict(approval)
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


INTENTS = frozenset({
    "chat", "project_query", "analysis", "modify_request", "simulation_request",
    "optimization_request", "parameter_impact", "report_request", "code_request", "general_question",
})
PROJECT_SCOPES = frozenset({"none", "current_project", "current_case", "current_session", "external"})
ACTION_TYPES = frozenset({"none", "read", "modify_file", "set_parameter", "simulate_once", "optimize", "modify_structure"})
PRIORITIES = frozenset({"low", "normal", "high"})
CONTEXT_REQUIREMENTS_ORDER = ("process_context", "yaml", "node_db", "simulation_db", "history", "files", "none")
CONTEXT_REQUIREMENTS = frozenset(CONTEXT_REQUIREMENTS_ORDER)


def validate_context_requirements(context: Any, *, intent: str, project_scope: str) -> None:
    if not isinstance(context, list) or not all(isinstance(value, str) for value in context):
        raise TypeError("context_required must be a list of strings")
    unknown = set(context) - CONTEXT_REQUIREMENTS
    if unknown:
        raise ValueError(f"invalid context_required: {sorted(unknown)}; allowed values: {sorted(CONTEXT_REQUIREMENTS)}. Equipment, stream and component identifiers belong in question/focus/analysis slots.")
    if "none" in context and len(context) > 1:
        raise ValueError("context_required='none' cannot be combined with other context sources")
    current_scope = project_scope in {"current_project", "current_case", "current_session"}
    if current_scope and (not context or context == ["none"]):
        raise ValueError("context_required cannot be empty or none when project_scope is " + project_scope)
    if (intent in {"project_query", "analysis", "parameter_impact"} or (intent == "general_question" and current_scope)) and "process_context" not in context:
        raise ValueError(f"{intent} requires context_required to include process_context")


@dataclass
class SubagentRequest(JsonModel):
    name: str
    task: str
    priority: str = "normal"
    parallel: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("subagent name must be a non-empty string")
        if not isinstance(self.task, str):
            raise TypeError("subagent task must be a string")
        if self.priority not in PRIORITIES:
            raise ValueError(f"invalid subagent priority: {self.priority!r}")
        if type(self.parallel) is not bool:
            raise TypeError("subagent parallel must be boolean")


@dataclass
class IntentSchema(JsonModel):
    """Strict, provider-neutral intent contract emitted by the classifier."""

    intent: str
    task_type: str
    project_scope: str
    context_required: list[str] = field(default_factory=list)
    subagents: list[SubagentRequest] = field(default_factory=list)
    action_type: str = "none"
    requires_confirmation: bool = False
    confidence: float = 0.0
    missing_information: list[str] = field(default_factory=list)
    reason: str = ""
    # Structured slots for parameter impact questions.  Kept optional so
    # existing provider payloads and checkpoints remain compatible.
    analysis: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"invalid intent: {self.intent!r}")
        if not isinstance(self.task_type, str) or not self.task_type:
            raise ValueError("task_type must be a non-empty string")
        if self.project_scope not in PROJECT_SCOPES:
            raise ValueError(f"invalid project_scope: {self.project_scope!r}")
        validate_context_requirements(self.context_required, intent=self.intent, project_scope=self.project_scope)
        if not isinstance(self.context_required, list) or not all(isinstance(x, str) for x in self.context_required):
            raise TypeError("context_required must be a list of strings")
        unknown_context = set(self.context_required) - CONTEXT_REQUIREMENTS
        if unknown_context:
            raise ValueError(f"invalid context_required: {sorted(unknown_context)}")
        if "none" in self.context_required and len(self.context_required) > 1:
            raise ValueError("context_required='none' cannot be combined with other context sources")
        if not isinstance(self.subagents, list) or not all(isinstance(x, SubagentRequest) for x in self.subagents):
            raise TypeError("subagents must be a list of SubagentRequest")
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"invalid action_type: {self.action_type!r}")
        if type(self.requires_confirmation) is not bool:
            raise TypeError("requires_confirmation must be boolean")
        if (not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool)
                or not math.isfinite(float(self.confidence)) or not 0 <= self.confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.missing_information, list) or not all(isinstance(x, str) for x in self.missing_information):
            raise TypeError("missing_information must be a list of strings")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if not isinstance(self.analysis, dict):
            raise TypeError("analysis must be a mapping")
        if self.intent == "parameter_impact":
            required_slots = {"equipment", "target_stream", "target_component", "metric", "candidate_parameters"}
            if required_slots - self.analysis.keys():
                raise ValueError("analysis missing required slots: " + ", ".join(sorted(required_slots - self.analysis.keys())))
            if not isinstance(self.analysis["candidate_parameters"], list) or not all(isinstance(v, str) for v in self.analysis["candidate_parameters"]):
                raise TypeError("analysis.candidate_parameters must be a list of strings")
        elif self.analysis:
            raise ValueError("analysis must be {} for ordinary intents")
        if self.action_type in {"modify_file", "set_parameter", "simulate_once", "optimize", "modify_structure"} and not self.requires_confirmation:
            raise ValueError("mutating, simulation and optimization actions require confirmation")
        if self.action_type in {"none", "read"} and self.requires_confirmation:
            raise ValueError("read-only and no-op intents cannot require confirmation")
        expected_action = {
            "simulation_request": "simulate_once",
            "optimization_request": "optimize",
            "parameter_impact": "simulate_once",
        }.get(self.intent)
        if expected_action is not None and self.action_type != expected_action:
            raise ValueError(f"{self.intent} must use action_type={expected_action}")
        if self.intent == "modify_request" and self.action_type in {"none", "read"}:
            raise ValueError("modify_request must declare a mutating action_type")
        if self.missing_information and self.action_type not in {"none", "read"} and self.requires_confirmation is not True:
            raise ValueError("incomplete action requests must remain pending and explicit")


@dataclass
class Classification(JsonModel):
    """Classifier output with verified provider provenance."""

    intent: IntentSchema
    classification_source: str = "llm"
    retry_count: int = 0

    def __post_init__(self) -> None:
        if type(self.retry_count) is not int or self.retry_count not in (0, 1):
            raise ValueError("retry_count must be 0 or 1")
        if self.classification_source not in {"llm", "llm_repaired"}:
            raise ValueError("classification_source must be llm or llm_repaired")


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
    # A bare TaskSpec consumes one budget unit. Callers that need a larger
    # allowance must set it explicitly; this keeps small compatibility tasks
    # from exhausting a runtime merely because max_steps is larger.
    budget: int = 1
    allow_subtasks: bool = False
    retry_attempts: int = 0
    recursion_depth: int = 0
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
    blocks: list[AnswerBlock] = field(default_factory=list)

    def __post_init__(self):
        self.blocks = blocks_from(self.blocks)

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
    max_retries: int = 1
    max_recursion_depth: int = 1
    max_tool_calls_per_task: int = 32

    def __post_init__(self) -> None:
        self.max_tasks = max(1, int(self.max_tasks))
        self.max_concurrency = max(1, int(self.max_concurrency))
        self.max_steps_per_task = max(1, int(self.max_steps_per_task))
        self.total_budget = max(1, int(self.total_budget))
        self.max_retries = max(0, int(self.max_retries))
        self.max_recursion_depth = max(0, int(self.max_recursion_depth))
        self.max_tool_calls_per_task = max(1, int(self.max_tool_calls_per_task))
        if self.mode == ExecutionMode.INSTANT:
            self.max_tasks = 1
            self.max_concurrency = 1
            self.allow_writes = False
            self.allow_subagent_delegation = False
            self.max_retries = 0
            self.max_recursion_depth = 0
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

    # Full evidence for structured consumers and persisted audit, not terminal prose.
    details: dict[str, Any] = field(default_factory=dict)
    blocks: list[AnswerBlock] = field(default_factory=list)
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Answer requires a standalone summary and supported schema version")
        self.blocks = blocks_from(self.blocks)

    @property
    def content(self) -> list[AnswerBlock]:
        """Summary followed by optional supporting blocks, for every UI."""
        return [*legacy_blocks(self.text), *self.blocks]


@dataclass
class PendingTask(JsonModel):
    """Serializable task continuation; focus/history are not execution state."""

    original_goal: str
    verified_intent: IntentSchema
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    waiting_fields: list[str] = field(default_factory=list)
    candidates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    field_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_digest: str = ""
    planning_blockers: list[dict[str, str]] = field(default_factory=list)
    phase: str = "validating"
    audit: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.verified_intent, dict):
            self.verified_intent = IntentSchema.from_dict(self.verified_intent)
        if not isinstance(self.verified_intent, IntentSchema) or not self.original_goal:
            raise ValueError("pending task requires a goal and verified intent")
        if self.phase not in {"validating", "clarifying", "ready", "awaiting_approval", "completed", "cancelled", "superseded", "failed"}:
            raise ValueError("invalid task phase")

    @property
    def active(self) -> bool:
        return self.phase in {"validating", "clarifying", "ready", "awaiting_approval"}


@dataclass
class SessionState(JsonModel):
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mode: ExecutionMode = ExecutionMode.STANDARD
    history: list[dict[str, str]] = field(default_factory=list)
    context_refs: dict[str, str] = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: ApprovalRequest | None = None
    pending_task: PendingTask | None = None
    internal_error: dict[str, Any] | None = None
    budget_used: int = 0
    intent: IntentSchema | None = None
    classification_source: str | None = None
    classification_retry_count: int = 0
    agent_unavailable: bool = False
    unavailable_error: dict[str, Any] | None = None
    context_summary: dict[str, Any] = field(default_factory=dict)
    context_sources: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_ref: str | None = None
    runtime_state: dict[str, Any] = field(default_factory=dict)

