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


class ClosedLoopService:
    """Small dependency-injection boundary around the closed-loop workflow."""

    def __init__(self, runner: Callable[..., Any] | None = None):
        self._runner = runner

    def run(self, *, driver: Any = None, config: Any = None, approved: bool = False,
            checkpoint_path: str | Path | None = None, checkpoint_metadata: dict[str, Any] | None = None,
            resume: bool = False, **kwargs: Any) -> Any:
        """Run the internal loop after LeadAgent has obtained approval.

        A runner can be injected by tests or applications.  The default is the
        legacy workflow adapter, which keeps ProcessDecisionAgent private to
        the closed-loop implementation.
        """
        if not approved and not kwargs.pop("_internal", False):
            raise PermissionError("closed-loop execution requires ActionGateway confirmation")
        runner = self._runner
        injected = runner is not None
        if runner is None:
            from src.workflows.agent_entry import run_configured_agent
            runner = run_configured_agent
        if config is None:
            raise ValueError("closed-loop config is required")
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path, {
                "status": "running", "metadata": checkpoint_metadata or {},
                "budget": self._budget(config), "resume": bool(resume),
            })
        if injected:
            result = runner(driver, config, **kwargs)
        else:
            result = runner(driver, config, _internal=True, **kwargs)
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path, {
                "status": "completed", "metadata": checkpoint_metadata or {},
                "budget": self._budget(config), "result_summary": self._summary(result),
            })
        return result

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

