"""JSON persistence for conversation metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import ConversationState


class SessionPersistenceError(RuntimeError):
    """Raised when a session cannot be saved or restored."""


class JsonSessionStore:
    """Persist only JSON metadata; no COM/database objects are serialized."""

    def save(self, state: ConversationState, path: str | Path) -> Path:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            raise SessionPersistenceError(f"无法保存会话：{exc}") from exc
        return target

    def load(self, path: str | Path) -> ConversationState:
        target = Path(path)
        try:
            data: Any = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("会话文件根节点必须是对象")
            return ConversationState.from_dict(data)
        except FileNotFoundError:
            raise SessionPersistenceError(f"会话文件不存在：{target}") from None
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise SessionPersistenceError(f"会话恢复失败：{exc}") from exc


def save_session(state: ConversationState, path: str | Path) -> Path:
    return JsonSessionStore().save(state, path)


def load_session(path: str | Path) -> ConversationState:
    return JsonSessionStore().load(path)
