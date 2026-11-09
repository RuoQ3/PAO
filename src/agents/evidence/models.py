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
    # Canonical graph vocabulary.  ``equipment`` and ``streams`` remain the
    # compatibility views used by existing agents; these appended fields keep
    # positional construction of the historical dataclass compatible.
    equipment_nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    compositions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[dict[str, Any]] = field(default_factory=list)

    # The graph helpers intentionally operate on the normalized fields but
    # fall back to legacy fields so contexts created by older clients remain
    # fully usable.
    def _nodes(self) -> list[dict[str, Any]]:
        return list(self.equipment_nodes or self.equipment or [])

    def _streams(self) -> list[dict[str, Any]]:
        if self.streams:
            return list(self.streams)
        # Permit hand-authored canonical contexts that provide only ``edges``.
        # The synthesized records deliberately retain the edge dictionary so
        # callers can still inspect arbitrary metadata.
        return [dict(edge, id=str(edge.get("id") or edge.get("stream_id") or edge.get("name")),
                     name=str(edge.get("name") or edge.get("id") or edge.get("stream_id")),
                     source=edge.get("source"), target=edge.get("target"),
                     source_node=edge.get("source"), downstream=edge.get("target"),
                     upstream=edge.get("source")) for edge in self.edges]

    @staticmethod
    def _same_id(item: dict[str, Any], value: str) -> bool:
        needle = str(value).casefold()
        return any(str(item.get(key, "")).casefold() == needle
                   for key in ("id", "name", "label", "aspen_path"))

    def get_equipment(self, equipment_id: str) -> dict[str, Any] | None:
        return next((item for item in self._nodes() if self._same_id(item, equipment_id)), None)

    def get_stream(self, stream_id: str) -> dict[str, Any] | None:
        return next((item for item in self._streams() if self._same_id(item, stream_id)), None)

    def _stream_source(self, stream: dict[str, Any]) -> str | None:
        return stream.get("source") or stream.get("upstream")

    def _stream_target(self, stream: dict[str, Any]) -> str | None:
        return stream.get("target") or stream.get("downstream")

    def downstream(self, equipment_id: str) -> list[str]:
        """Return equipment IDs reached through outgoing streams."""
        node = self.get_equipment(equipment_id)
        if node is None:
            return []
        name = str(node.get("id") or node.get("name"))
        result: list[str] = []
        for stream in self._streams():
            if str(self._stream_source(stream) or "").casefold() != name.casefold():
                continue
            target = self._stream_target(stream)
            if target and target not in result:
                result.append(str(target))
        return result

    def upstream(self, equipment_id: str) -> list[str]:
        """Return equipment IDs feeding the requested equipment."""
        node = self.get_equipment(equipment_id)
        if node is None:
            return []
        name = str(node.get("id") or node.get("name"))
        result: list[str] = []
        for stream in self._streams():
            if str(self._stream_target(stream) or "").casefold() != name.casefold():
                continue
            source = self._stream_source(stream)
            if source and source not in result:
                result.append(str(source))
        return result

    def trace_stream(self, stream_id: str) -> dict[str, Any] | None:
        stream = self.get_stream(stream_id)
        if stream is None:
            return None
        return {"stream": stream, "source": self._stream_source(stream),
                "target": self._stream_target(stream),
                "source_node": self.get_equipment(str(self._stream_source(stream))) if self._stream_source(stream) else None,
                "target_node": self.get_equipment(str(self._stream_target(stream))) if self._stream_target(stream) else None}

    def trace_path(self, start_id: str, end_id: str) -> list[dict[str, Any]]:
        """Find a shortest equipment/stream path without guessing missing links."""
        start = self.get_equipment(start_id)
        end = self.get_equipment(end_id)
        if start is None or end is None:
            return []
        start_name = str(start.get("id") or start.get("name"))
        end_name = str(end.get("id") or end.get("name"))
        queue: list[tuple[str, list[dict[str, Any]]]] = [(start_name, [start])]
        visited = {start_name.casefold()}
        while queue:
            current, path = queue.pop(0)
            for stream in self._streams():
                if str(self._stream_source(stream) or "").casefold() != current.casefold():
                    continue
                target = self._stream_target(stream)
                if not target:
                    continue
                next_path = path + [stream]
                if str(target).casefold() == end_name.casefold():
                    target_node = self.get_equipment(str(target))
                    return next_path + ([target_node] if target_node else [])
                key = str(target).casefold()
                if key not in visited:
                    visited.add(key)
                    target_node = self.get_equipment(str(target))
                    queue.append((str(target), next_path + ([target_node] if target_node else [])))
        return []

    def terminal_streams(self) -> list[dict[str, Any]]:
        return [s for s in self._streams() if not self._stream_target(s)]

    def boundary_nodes(self) -> list[dict[str, Any]]:
        # A boundary node has an open side (feed/product/vent/unresolved
        # connection), including a completely isolated equipment record.
        result = []
        for node in self._nodes():
            name = str(node.get("id") or node.get("name") or "")
            incoming = [s for s in self._streams() if str(self._stream_target(s) or "").casefold() == name.casefold()]
            outgoing = [s for s in self._streams() if str(self._stream_source(s) or "").casefold() == name.casefold()]
            open_incoming = any(not self._stream_source(s) for s in incoming)
            open_outgoing = any(not self._stream_target(s) for s in outgoing)
            if not incoming or not outgoing or open_incoming or open_outgoing:
                result.append(node)
        return result

    def compare_compositions(self, input_stream: str, output_stream: str) -> dict[str, Any]:
        left = self.get_stream(input_stream) or {}
        right = self.get_stream(output_stream) or {}
        def values(stream: dict[str, Any]) -> dict[str, float]:
            raw = stream.get("composition") or stream.get("components") or []
            if isinstance(raw, dict):
                raw = [{"name": k, "value": v} for k, v in raw.items()]
            result = {}
            for item in raw if isinstance(raw, list) else []:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("component") or item.get("id")
                value = item.get("mole_fraction", item.get("mass_fraction", item.get("value")))
                try:
                    if name is not None and value is not None:
                        result[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
            return result
        a, b = values(left), values(right)
        changes = []
        for name in sorted(set(a) | set(b), key=str.casefold):
            before, after = a.get(name), b.get(name)
            delta = None if before is None or after is None else after - before
            changes.append({"component": name, "input": before, "output": after,
                            "delta": delta,
                            "classification": ("unknown" if delta is None else
                                "enriched" if delta > 1e-12 else
                                "reduced" if delta < -1e-12 else "unchanged")})
        return {"input_stream": input_stream, "output_stream": output_stream,
                "changes": changes, "observed": bool(a or b),
                "evidence_ids": list((left.get("evidence_ids") or []) + (right.get("evidence_ids") or []))}

    def summarize_process(self) -> dict[str, Any]:
        nodes = self._nodes()
        return {
            "model_id": self.model_id,
            "entry_streams": [s for s in self._streams() if not self._stream_source(s)],
            "major_stages": [n for n in nodes if n.get("role") not in {None, "unknown"}],
            "separation_or_reaction_nodes": [n for n in nodes if str(n.get("block_type", "")).upper() in {"RCSTR", "RSTOIC", "FLASH2", "RADFRAC", "SEP"}],
            "branches": [s for s in self._streams() if sum(1 for x in self._streams() if self._stream_source(x) == self._stream_source(s)) > 1],
            "recirculation": [s for s in self._streams() if self._is_cycle_stream(s)],
            "terminal_streams": self.terminal_streams(),
            "confirmed_facts": list(self.facts),
            "inferences": list(self.inferences),
            "missing_information": list(self.diagnostics),
        }

    def _is_cycle_stream(self, stream: dict[str, Any]) -> bool:
        source = self._stream_source(stream)
        target = self._stream_target(stream)
        if not source or not target:
            return False
        seen = {str(target).casefold()}
        queue = [str(target)]
        while queue:
            current = queue.pop(0)
            if current.casefold() == str(source).casefold():
                return True
            for candidate in self._streams():
                if str(self._stream_source(candidate) or "").casefold() == current.casefold():
                    nxt = self._stream_target(candidate)
                    if nxt and str(nxt).casefold() not in seen:
                        seen.add(str(nxt).casefold())
                        queue.append(str(nxt))
        return False


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
