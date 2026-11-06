"""SDK-neutral conversation and evidence models.

All models are deliberately plain dataclasses.  ``to_dict``/``from_dict``
provide a stable JSON boundary for checkpoints, HTTP adapters, and tests.
Unknown fields are rejected on input so a producer cannot silently bypass the
contract.  ``None`` means unknown or not available; it is never an assertion
that a value is zero or false.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Literal, TypeVar


class _StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - useful for logs
        return self.value


class EvidenceSource(_StrEnum):
    IMA = "ima"
    LOCAL_PROCESS = "local_process"
    SIMULATION = "simulation"
    TECHNICAL_ADVICE = "technical_advice"
    USER = "user"
    SYSTEM = "system"
    # ProcessContext provenance labels.  Kept as first-class values so
    # callers can distinguish configuration, Aspen snapshots and history.
    ASPEN_MODEL = "aspen_model"
    YAML = "yaml"
    SIMULATION_DB = "simulation_db"
    USER_INPUT = "user_input"
    NODEDB = "nodedb"
    OPTIMIZATION_RESULTS = "optimization_results"


class AdviceUncertainty(_StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ActionKind(_StrEnum):
    READ = "read"
    SET_PARAMETER = "set_parameter"
    SIMULATE_ONCE = "simulate_once"
    OPTIMIZE = "optimize"
    MODIFY_STRUCTURE = "modify_structure"


@dataclass
class EvidenceItem:
    """A claim or observation with explicit provenance and uncertainty.

    ``id`` is stable within one conversation/checkpoint. ``source_ref`` points
    to an external record (case id, checkpoint key, file, or IMA request id).
    ``supports`` is intentionally optional: evidence may contradict a claim.
    """

    id: str
    source: EvidenceSource
    claim: str
    source_ref: str | None = None
    value: Any = None
    unit: str | None = None
    confidence: float | None = None
    uncertainty: AdviceUncertainty = AdviceUncertainty.UNKNOWN
    supports: bool | None = None
    observed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImaAnswer:
    """An answer returned by an external IMA adapter.

    Lifecycle: created per request, merged into an ``EvidenceBundle``, then
    retained as an immutable audit item.  ``answer`` can be absent when the
    adapter reports an error or a partial response.
    """

    answer: str | None = None
    source: str = ""
    confidence: float | None = None
    assumptions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    request_id: str | None = None
    received_at: str | None = None


@dataclass
class ProcessContext:
    """Read-only process snapshot supplied to advisors and the orchestrator.

    The builder may source fields from ``ChatSessionState``, Aspen topology or
    manifest readers, YAML configuration, ``ProcessCase`` history, and
    ``SimulationDB``.  It is a snapshot: updates create a new context and do
    not mutate the originating objects.
    """

    model_id: str
    model_version: str | None = None
    equipment: list[dict[str, Any]] = field(default_factory=list)
    streams: list[dict[str, Any]] = field(default_factory=list)
    topology: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    objectives: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    current_values: dict[str, Any] = field(default_factory=dict)
    # Cached Aspen Input nodes grouped by block. These are catalog snapshot
    # values, not values from a running simulation.
    equipment_parameters: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    history_evidence: list[EvidenceItem] = field(default_factory=list)
    sources: list[EvidenceItem] = field(default_factory=list)
    session_id: str | None = None
    generated_at: str | None = None
    status: str = "ok"
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class TechnicalAdvice:
    """Technical conclusion with mechanisms, evidence, and next checks.

    This is advisory output only.  It does not authorize a write or a run;
    authorization is represented separately by ``UserAction``.
    """

    conclusion: str
    mechanism: str | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    suggested_checks: list[str] = field(default_factory=list)
    suggested_experiments: list[str] = field(default_factory=list)
    uncertainty: AdviceUncertainty = AdviceUncertainty.UNKNOWN
    confidence: float | None = None
    assumptions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    """The orchestrator's evidence packet for one response or action.

    ``conflicts`` records unresolved disagreements instead of selecting a
    winner silently.  ``complete`` is false when required evidence is missing.
    """

    ima: list[ImaAnswer] = field(default_factory=list)
    local_process_facts: list[EvidenceItem] = field(default_factory=list)
    simulation_evidence: list[EvidenceItem] = field(default_factory=list)
    technical_advice: list[TechnicalAdvice] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    complete: bool = False
    bundle_id: str | None = None


@dataclass
class UserAction:
    """A proposed or confirmed action sent to the controlled gateway.

    ``confirmation_status`` is ``required`` until a human confirms any
    operation that can change a model or spend simulation/optimization budget.
    The gateway must re-check this status and the action scope.
    """

    action_type: ActionKind
    target: str
    old_value: Any = None
    new_value: Any = None
    reason: str = ""
    risks: list[str] = field(default_factory=list)
    confirmation_status: Literal["not_required", "required", "confirmed", "rejected"] = "required"
    evidence_ids: list[str] = field(default_factory=list)
    action_id: str | None = None


T = TypeVar("T", bound="JsonModel")


class JsonModel:
    """Mixin implementing strict, recursively JSON-safe serialization."""

    _enum_fields: ClassVar[dict[str, type[Enum]]] = {}

    def to_dict(self) -> dict[str, Any]:
        return _encode(asdict(self))

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kwargs)

    @classmethod
    def from_dict(cls: type[T], raw: dict[str, Any]) -> T:
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.__name__} requires an object")
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        data = dict(raw)
        for name, enum_type in cls._enum_fields.items():
            if data.get(name) is not None and not isinstance(data[name], enum_type):
                data[name] = enum_type(data[name])
        nested = {
            EvidenceItem: {},
            ImaAnswer: {"evidence": EvidenceItem},
            ProcessContext: {"history_evidence": EvidenceItem, "sources": EvidenceItem},
            TechnicalAdvice: {"evidence": EvidenceItem},
            EvidenceBundle: {
                "ima": ImaAnswer,
                "local_process_facts": EvidenceItem,
                "simulation_evidence": EvidenceItem,
                "technical_advice": TechnicalAdvice,
            },
            UserAction: {},
        }.get(cls, {})
        for name, child_type in nested.items():
            values = data.get(name)
            if isinstance(values, list):
                data[name] = [child_type.from_dict(v) if isinstance(v, dict) else v for v in values]
        return cls(**data)

    @classmethod
    def from_json(cls: type[T], raw: str) -> T:
        return cls.from_dict(json.loads(raw))


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


# Attach the mixin behavior without making the public dataclass inheritance
# hierarchy cumbersome for type checkers and downstream serializers.
for _model in (EvidenceItem, ImaAnswer, ProcessContext, TechnicalAdvice, EvidenceBundle, UserAction):
    _model.to_dict = JsonModel.to_dict  # type: ignore[attr-defined]
    _model.to_json = JsonModel.to_json  # type: ignore[attr-defined]
    _model.from_dict = classmethod(JsonModel.from_dict.__func__)  # type: ignore[attr-defined]
    _model.from_json = classmethod(JsonModel.from_json.__func__)  # type: ignore[attr-defined]

EvidenceItem._enum_fields = {"source": EvidenceSource, "uncertainty": AdviceUncertainty}
ImaAnswer._enum_fields = {}
ProcessContext._enum_fields = {}
TechnicalAdvice._enum_fields = {"uncertainty": AdviceUncertainty}
EvidenceBundle._enum_fields = {}
UserAction._enum_fields = {"action_type": ActionKind}
