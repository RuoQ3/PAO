"""Internal closed-loop service used by :class:`LeadAgent`.

The service is deliberately not a user-facing agent.  It owns the bridge to
the existing checkpointed optimization workflow; routing and approval remain
the responsibility of LeadAgent and ActionGateway.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable
from src.agents.errors import AgentUnavailableError, unavailable


class ClosedLoopService:
    """Small dependency-injection boundary around the closed-loop workflow."""

    def __init__(self, runner: Callable[..., Any] | None = None):
        self._runner = runner

    def run(self, *, driver: Any = None, config: Any = None, approved: bool = False,
            checkpoint_path: str | Path | None = None, checkpoint_metadata: dict[str, Any] | None = None,
            resume: bool = False, **kwargs: Any) -> Any:
        """Disabled legacy execution entry. Checkpoint read/write utilities remain."""
        raise PermissionError("Legacy closed-loop execution disabled: approved/_internal booleans are not Gateway authorization")

    @staticmethod
    def _budget(config: Any) -> dict[str, Any]:
        settings = getattr(config, "agent_loop", None) or {}
        return {"max_evaluations": settings.get("max_evaluations") if isinstance(settings, dict) else None,
                "checkpoint": getattr(config, "agent_checkpoint_path", None)}

    @staticmethod
    def _summary(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return {key: result.get(key) for key in ("status", "result", "used", "termination_reason") if key in result}
        return {"type": type(result).__name__}

    @staticmethod
    def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
        """Persist only JSON-safe orchestration metadata; never driver/COM objects."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(target)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS closed_loop_agent_checkpoint (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("INSERT OR REPLACE INTO closed_loop_agent_checkpoint VALUES (1, ?)",
                       (json.dumps(payload, ensure_ascii=False, default=str),))
            db.commit()
        finally:
            db.close()

    @staticmethod
    def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
        target = Path(path)
        if not target.exists():
            return None
        db = sqlite3.connect(target)
        try:
            row = db.execute("SELECT payload FROM closed_loop_agent_checkpoint WHERE id=1").fetchone()
            return json.loads(row[0]) if row else None
        finally:
            db.close()
