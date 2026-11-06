"""Parameter action parsing and a COM-isolated Aspen executor.

The gateway only receives plain values.  This module is the sole place where
an AspenDriver may be called for a parameter write; callers never receive COM
objects.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import ActionRequest, ActionType, ModelSchema


def parse_parameter_request(text: str, schema: ModelSchema, *, reason: str = "user request") -> ActionRequest:
    """Convert common Chinese/English parameter requests into an action.

    Supported forms include ``把 T1 的回流比改成 0.85`` and
    ``set \\Data\\Blocks\\T1\\Input\\REFLUX = 0.85``.  Ambiguous or unknown
    names are rejected instead of guessed.
    """
    m = re.search(r"(?:改(?:成|为)|设置为|set\s+[^=]+?=)\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
    if not m:
        raise ValueError("无法解析参数新值")
    value = float(m.group(1))
    if value.is_integer() and any(k in text for k in ("级", "stage", "整数")):
        value = int(value)
    path = None
    text_lower = text.lower()
    for candidate in schema.parameters:
        short = candidate.rsplit("\\", 1)[-1].lower()
        configured = schema.aliases.get(candidate, "")
        aliases = configured if isinstance(configured, (list, tuple, set)) else [configured]
        if candidate.lower() in text_lower or (short and short in text_lower) or any(a and str(a).lower() in text_lower for a in aliases):
            path = candidate; break
    if path is None:
        # Match equipment plus a configured alias, e.g. T1 + 回流比.
        for alias, candidate in schema.aliases.items():
            if isinstance(candidate, str) and alias.lower() in text_lower and candidate in schema.parameters:
                path = candidate; break
    if path is None:
        raise ValueError("参数名称不在白名单中")
    return ActionRequest(ActionType.SET_PARAMETER, path, new_value=value, reason=reason,
                         model_id=schema.identity.model_id)


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BackupToken:
    source: str
    backup: str
    fingerprint: str


class AspenParameterExecutor:
    """Controlled adapter: all Aspen access is delegated to ``driver``."""
    def __init__(self, driver: Any, *, run_once: Callable[[ActionRequest], Any] | None = None,
                 backup_dir: str | Path | None = None):
        self.driver = driver
        self.run_once = run_once
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self._values: dict[str, Any] = {}
        self._tokens: dict[str, BackupToken] = {}

    def refresh_schema(self, schema: ModelSchema) -> None:
        for path, spec in schema.parameters.items():
            try:
                schema.current_values[path] = self.driver.get_value(path)
            except Exception:
                # Validation will reject unavailable current values.
                schema.current_values.pop(path, None)

    def backup(self, action: ActionRequest) -> BackupToken:
        source = getattr(self.driver, "filepath", None)
        if source is None:
            raise RuntimeError("无法获取 Aspen 当前文件路径，拒绝写入")
        source = str(Path(source).resolve())
        src = Path(source)
        if not src.exists():
            raise RuntimeError(f"Aspen 文件不存在: {source}")
        target_dir = self.backup_dir or src.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{src.stem}.{action.audit_id}.bak{src.suffix}"
        shutil.copy2(src, target)
        token = BackupToken(source, str(target), _fingerprint(target))
        self._values[action.target] = self.driver.get_value(action.target)
        self._tokens[action.audit_id] = token
        return {"source": token.source, "backup": token.backup, "fingerprint": token.fingerprint,
                "path": action.target, "value": self.driver.get_value(action.target)}

    def execute(self, action: ActionRequest, *, preview: Any) -> dict[str, Any]:
        if action.action_type != ActionType.SET_PARAMETER:
            raise RuntimeError("受控 Aspen 执行器只支持白名单参数修改")
        before = self.driver.get_value(action.target)
        self.driver.set_value(action.target, action.new_value)
        after = self.driver.get_value(action.target)
        if after != action.new_value:
            raise RuntimeError(f"写入校验失败: requested={action.new_value!r}, actual={after!r}")
        result: dict[str, Any] = {"audit_id": action.audit_id, "status": "executed",
                                  "target": action.target, "old_value": before,
                                  "new_value": after, "backup": self._tokens.get(action.audit_id).__dict__ if action.audit_id in self._tokens else None}
        if preview.will_run_simulation and self.run_once:
            result["simulation"] = self.run_once(action)
        return result

    def rollback(self, token: Any) -> None:
        backup = Path(token["backup"] if isinstance(token, dict) else token.backup)
        source = Path(token["source"] if isinstance(token, dict) else token.source)
        if not backup.exists():
            raise RuntimeError("备份文件不存在，无法回滚")
        shutil.copy2(backup, source)
        if isinstance(token, dict) and token.get("path") in self._values:
            self.driver.set_value(token["path"], self._values[token["path"]])
        # A real live COM document should be reopened by the owner after file rollback.


__all__ = ["parse_parameter_request", "AspenParameterExecutor", "BackupToken"]


def compare_results(before: Any, after: Any) -> dict[str, Any]:
    """Build a JSON-safe before/after delta for audit and UI rendering."""
    def obj_map(value: Any) -> dict[str, Any]:
        if value is None: return {}
        if isinstance(value, dict): return value
        items = getattr(value, "objectives", None)
        if items is None: return {}
        return {getattr(x, "name", str(i)): getattr(x, "value", None) for i, x in enumerate(items)}
    b, a = obj_map(before), obj_map(after)
    keys = sorted(set(b) | set(a))
    return {"before": b, "after": a,
            "delta": {k: (a.get(k) - b[k] if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)) else None) for k in keys}}
