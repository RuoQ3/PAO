from __future__ import annotations
from typing import Any, Protocol

class ExecutorProtocol(Protocol):
    def backup(self, action: Any) -> Any: ...
    def execute(self, action: Any, *, preview: Any) -> dict[str, Any]: ...
    def rollback(self, token: Any) -> None: ...

class ExecutionError(RuntimeError): pass

class MockExecutor:
    """Safe test executor; stores values in memory and never exposes SDK objects."""
    def __init__(self): self.calls: list[dict[str, Any]] = []; self.backups: list[dict[str, Any]] = []
    def backup(self, action):
        token = {"audit_id": action.audit_id, "old_value": action.old_value}
        self.backups.append(token); return token
    def execute(self, action, *, preview):
        if action.action_type.value == "modify_structure": raise ExecutionError("structural modifications disabled")
        result = {"audit_id": action.audit_id, "status": "simulated", "simulated": True, "real_execution": False, "action_type": action.action_type.value, "target": action.target, "new_value": action.new_value}
        self.calls.append(result); return result
    def rollback(self, token):
        if token not in self.backups: raise ExecutionError("unknown backup token")
        return None
