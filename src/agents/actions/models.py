from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

class _Enum(str, Enum):
    def __str__(self): return self.value

class ActionType(_Enum):
    INSPECT = "inspect"
    SET_PARAMETER = "set_parameter"
    RUN_SIMULATION = "run_simulation"
    RUN_OPTIMIZATION = "run_optimization"
    SET_SEARCH_REGION = "set_search_region"
    MODIFY_STRUCTURE = "modify_structure"

class ApprovalStatus(_Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    FAILED = "failed"

class ExecutionMode(_Enum):
    PREVIEW = "preview"
    MANUAL = "manual"
    AUTOMATIC = "automatic"

class AuditEventType(_Enum):
    USER_REQUEST = "user_request"
    AGENT_SUGGESTION = "agent_suggestion"
    VALIDATION = "validation"
    PREVIEW = "preview"
    CONFIRMATION = "confirmation"
    EXECUTION = "execution"
    ERROR = "error"

@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    model_version: str | None = None

@dataclass(frozen=True)
class ParameterSpec:
    path: str
    value_type: str = "float"  # int, float, bool, str
    writable: bool = True
    unit: str | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    integer: bool = False
    impact_targets: tuple[str, ...] = ()
    requires_simulation: bool = True
    dependencies: tuple[str, ...] = ()
    allowed_values: tuple[Any, ...] = ()
    derived: bool = False
    derived_from: tuple[str, ...] = ()

@dataclass
class ModelSchema:
    identity: ModelIdentity
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    current_values: dict[str, Any] = field(default_factory=dict)
    writable: bool = True
    permissions: set[str] = field(default_factory=lambda: {"inspect", "set_parameter", "run_simulation", "run_optimization", "set_search_region"})
    targets: tuple[str, ...] = ()
    # Cross variable rules, e.g. {FEED_STAGE: {"lt": NSTAGE}}.
    dependencies: dict[str, dict[str, str]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    derived_variables: set[str] = field(default_factory=set)

@dataclass
class ActionRequest:
    action_type: ActionType | str
    target: str
    old_value: Any = None
    new_value: Any = None
    unit: str | None = None
    reason: str = ""
    evidence: list[Any] = field(default_factory=list)
    risk: list[str] = field(default_factory=list)
    requires_confirmation: bool = True
    execution_mode: ExecutionMode | str = ExecutionMode.MANUAL
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str | None = None
    model_version: str | None = None
    # Optional per-action override.  ``None`` keeps the schema default.
    run_simulation: bool | None = None

    def __post_init__(self):
        self.action_type = ActionType(self.action_type)
        self.execution_mode = ExecutionMode(self.execution_mode)
        if not isinstance(self.target, str): raise TypeError("target must be a string")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["action_type"] = self.action_type.value
        out["execution_mode"] = self.execution_mode.value
        return out

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kwargs)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActionRequest":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - allowed
        if unknown: raise ValueError(f"Unknown action fields: {sorted(unknown)}")
        return cls(**dict(raw))

    @classmethod
    def from_json(cls, raw: str) -> "ActionRequest":
        return cls.from_dict(json.loads(raw))

@dataclass(frozen=True)
class ActionPreview:
    audit_id: str
    action_type: str
    target: str
    old_value: Any
    new_value: Any
    unit: str | None
    impact_targets: tuple[str, ...]
    constraint_risks: tuple[str, ...]
    will_run_simulation: bool
    approval_required: bool
    confirmation_prompt: str = ""
    value_type: str | None = None
    writable: bool = True
    backup_required: bool = True

    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    audit_id: str
    event_type: AuditEventType | str
    timestamp: str = field(default_factory=_now)
    actor: str = "system"
    status: str | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self): object.__setattr__(self, "event_type", AuditEventType(self.event_type))
    def to_dict(self):
        d = asdict(self); d["event_type"] = self.event_type.value; return d
    @classmethod
    def from_dict(cls, raw): return cls(**dict(raw))
