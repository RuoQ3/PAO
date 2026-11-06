"""Minimal dependency-free protocols for the planned call graph."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .models import EvidenceBundle, ImaAnswer, ProcessContext, TechnicalAdvice, UserAction


class ImaAdapter(Protocol):
    """Read-only boundary to an external IMA service; no SDK types leak out."""

    def ask(self, question: str, *, context: ProcessContext | None = None) -> ImaAnswer: ...


class ProcessContextBuilder(Protocol):
    """Builds a snapshot from local files, history, and optional simulation data."""

    def build(self, *, session: Any, model_id: str | None = None) -> ProcessContext: ...


class TechnicalAdvisor(Protocol):
    """Produces traceable advice from context and an evidence packet."""

    def advise(self, context: ProcessContext, evidence: EvidenceBundle) -> TechnicalAdvice: ...


class ConversationOrchestrator(Protocol):
    """Coordinates evidence and returns a response plus optional action proposal."""

    def handle(
        self,
        question: str,
        *,
        context: ProcessContext,
        evidence: EvidenceBundle,
    ) -> tuple[str, UserAction | None]: ...


class ActionGateway(Protocol):
    """Controlled execution boundary; implementations must enforce policy."""

    def preview(self, action: UserAction, context: ProcessContext) -> dict[str, Any]: ...

    def execute(self, action: UserAction, context: ProcessContext) -> dict[str, Any]: ...
