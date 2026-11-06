from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.orchestration.models import ExecutionMode, SessionState


@dataclass
class ConversationSession:
    state: SessionState
    lead: Any

    def submit(self, text: str):
        self.state.history.append({"role": "user", "content": text})
        answer = self.lead.answer(text, self.state)
        self.state.history.append({"role": "assistant", "content": answer.text})
        return answer

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()

