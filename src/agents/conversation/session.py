"""Small adapter suitable for embedding in the existing ``--chat`` loop."""
from __future__ import annotations

from collections.abc import Callable

from .commands import CommandResult, handle_command, parse_input
from .state import ConversationState, PendingAction


class ConversationSession:
    """Route slash commands locally and forward ordinary text unchanged.

    ``orchestrator`` is intentionally injected.  This layer does not import or
    invoke the real coordinator, Aspen, an LLM, or a database writer.
    """

    def __init__(self, state: ConversationState, orchestrator: Callable[[str, ConversationState], str] | None = None):
        self.state = state
        self.orchestrator = orchestrator

    def submit(self, text: str | None) -> CommandResult:
        parsed = parse_input(text)
        result = handle_command(parsed, self.state)
        if result.forwarded is not None and self.orchestrator is not None:
            response = self.orchestrator(result.forwarded, self.state)
            self.state.record_turn(result.forwarded, response)
            return CommandResult(output=response, forwarded=result.forwarded)
        if result.forwarded is not None:
            self.state.record_turn(result.forwarded, "")
        return result

    def propose_action(self, description: str, payload: dict | None = None, *, audit_id: str | None = None) -> PendingAction:
        return self.state.add_pending_action(description, payload, audit_id=audit_id)

