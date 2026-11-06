"""Serializable state for the incremental conversation CLI.

The conversation layer only stores references and text metadata.  It never stores
Aspen COM objects, open database handles, credentials, or other runtime objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


_SENSITIVE_PARTS = ("password", "secret", "token", "api_key", "access_key", "credential", "authorization")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(part in str(key).lower() for part in _SENSITIVE_PARTS) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


@dataclass
class PendingAction:
    """An action proposed by an upper orchestrator and waiting for approval."""

    action_id: str
    description: str
    payload: dict[str, Any] = field(default_factory=dict)
    audit_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PendingAction":
        return cls(
            action_id=str(value.get("action_id") or uuid.uuid4().hex),
            description=str(value.get("description", "")),
            payload=dict(value.get("payload") or {}),
            audit_id=value.get("audit_id"),
            created_at=str(value.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ConversationState:
    """Cross-turn state used by the CLI conversation adapter.

    All paths are strings so this object remains JSON serializable and can be
    restored in a process that has no Aspen installation.  ``yaml_path`` and
    ``simulation_db_path`` are also exposed under the legacy-friendly aliases
    ``case_config_path`` and ``db_path``.
    """

    aspen_file: str | None = None
    yaml_path: str | None = None
    simulation_db_path: str | None = None
    node_db_path: str | None = None
    user_goal: str = ""
    context_summary: str = ""
    last_response: str = ""
    pending_actions: list[PendingAction] = field(default_factory=list)
    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_count: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    last_checkpoint: dict[str, Any] | None = None
    schema_version: int = 1
    current_equipment: str | None = None
    equipment_focus: list[str] = field(default_factory=list)
    current_objective: str | None = None
    current_experiment: str | None = None
    last_analysis: dict[str, Any] | None = None

    @property
    def case_config_path(self) -> str | None:
        return self.yaml_path

    @case_config_path.setter
    def case_config_path(self, value: str | None) -> None:
        self.yaml_path = value

    @property
    def db_path(self) -> str | None:
        return self.simulation_db_path

    @db_path.setter
    def db_path(self, value: str | None) -> None:
        self.simulation_db_path = value

    def add_pending_action(self, description: str, payload: dict[str, Any] | None = None,
                           *, action_id: str | None = None, audit_id: str | None = None) -> PendingAction:
        action = PendingAction(
            action_id=action_id or uuid.uuid4().hex,
            description=description,
            payload=dict(payload or {}),
            audit_id=audit_id or self.audit_id,
        )
        self.pending_actions.append(action)
        return action

    def checkpoint(self) -> None:
        """Capture restorable metadata before a command-level mutation."""
        self.last_checkpoint = {
            "aspen_file": self.aspen_file,
            "yaml_path": self.yaml_path,
            "simulation_db_path": self.simulation_db_path,
            "node_db_path": self.node_db_path,
            "user_goal": self.user_goal,
            "context_summary": self.context_summary,
            "last_response": self.last_response,
        }

    def rollback(self) -> bool:
        if not self.last_checkpoint:
            return False
        for key, value in self.last_checkpoint.items():
            setattr(self, key, value)
        self.last_checkpoint = None
        return True

    def find_pending(self, action_id: str | None = None) -> PendingAction | None:
        if not self.pending_actions:
            return None
        if action_id is None:
            return self.pending_actions[0]
        return next((a for a in self.pending_actions if a.action_id == action_id), None)

    def remove_pending(self, action_id: str | None = None) -> PendingAction | None:
        action = self.find_pending(action_id)
        if action is not None:
            self.pending_actions.remove(action)
        return action

    def record_turn(self, user_text: str, response: str = "") -> None:
        self.turn_count += 1
        self.last_response = response
        execution_path = ""
        if isinstance(self.last_analysis, dict):
            path = self.last_analysis.get("execution_path")
            if isinstance(path, dict):
                execution_path = "; ".join(f"{key}={value}" for key, value in path.items() if value)
            elif path:
                execution_path = str(path)
        self.history.append({"user": user_text, "assistant": response, "execution_path": execution_path})
        # Keep persistence bounded while retaining enough context for /history.
        if len(self.history) > 100:
            del self.history[:-100]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pending_actions"] = [_redact(a.to_dict()) for a in self.pending_actions]
        return _redact(result)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationState":
        # Explicit field selection prevents arbitrary JSON from becoming runtime state.
        state = cls(
            aspen_file=value.get("aspen_file"),
            yaml_path=value.get("yaml_path", value.get("case_config_path")),
            simulation_db_path=value.get("simulation_db_path", value.get("db_path")),
            node_db_path=value.get("node_db_path"),
            user_goal=str(value.get("user_goal", "")),
            context_summary=str(value.get("context_summary", "")),
            last_response=str(value.get("last_response", "")),
            audit_id=str(value.get("audit_id") or uuid.uuid4().hex),
            turn_count=int(value.get("turn_count", 0)),
            history=[dict(x) for x in (value.get("history") or []) if isinstance(x, dict)],
            last_checkpoint=dict(value["last_checkpoint"]) if isinstance(value.get("last_checkpoint"), dict) else None,
            schema_version=int(value.get("schema_version", 1)),
            current_equipment=value.get("current_equipment"),
            equipment_focus=[str(x) for x in value.get("equipment_focus", [])],
            current_objective=value.get("current_objective"),
            current_experiment=value.get("current_experiment"),
            last_analysis=dict(value["last_analysis"]) if isinstance(value.get("last_analysis"), dict) else None,
        )
        state.pending_actions = [PendingAction.from_dict(x) for x in (value.get("pending_actions") or []) if isinstance(x, dict)]
        return state

    def as_display_dict(self) -> dict[str, str]:
        return {
            "aspen_file": self.aspen_file or "(未设置)",
            "yaml_path": self.yaml_path or "(未设置)",
            "simulation_db_path": self.simulation_db_path or "(未设置)",
            "node_db_path": self.node_db_path or "(未设置)",
            "user_goal": self.user_goal or "(未设置)",
            "audit_id": self.audit_id,
            "turn_count": str(self.turn_count),
            "pending_actions": str(len(self.pending_actions)),
        }

    def to_context_summary(self) -> str:
        paths = [
            f"Aspen: {self.aspen_file or '未设置'}",
            f"YAML: {self.yaml_path or '未设置'}",
            f"SimulationDB: {self.simulation_db_path or '未设置'}",
            f"NodeDB: {self.node_db_path or '未设置'}",
        ]
        extras = [f"用户目标: {self.user_goal}" if self.user_goal else ""]
        if self.context_summary:
            extras.append(f"上下文摘要: {self.context_summary}")
        return "\n".join(paths + [x for x in extras if x])


# Names useful to callers migrating from the old coordinator model.
SessionState = ConversationState
ConversationSessionState = ConversationState
