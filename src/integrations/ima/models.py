"""Transport-independent IMA request and response models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.evidence.models import ImaAnswer as ContractImaAnswer


@dataclass(frozen=True)
class ImaSource:
    """A retained citation from IMA (document/chapter/page/snippet)."""
    source: str
    title: str | None = None
    chapter: str | None = None
    page: int | None = None
    snippet: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "title": self.title, "chapter": self.chapter,
            "page": self.page, "snippet": self.snippet, "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ImaRequest:
    query: str
    context: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    knowledge_base: str | None = None
    request_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "query": self.query, "context": self.context, "model": self.model,
            "knowledge_base": self.knowledge_base, "request_id": self.request_id,
        }


@dataclass(frozen=True)
class ImaAnswer:
    """Normalized answer used by this integration; no SDK objects escape."""
    answer: str | None = None
    source: str = "ima"
    confidence: float | None = None
    assumptions: tuple[str, ...] = ()
    sources: tuple[ImaSource, ...] = ()
    request_id: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_contract(self) -> ContractImaAnswer:
        """Project to the phase-0 public contract while retaining citations."""
        from src.agents.evidence.models import EvidenceItem, EvidenceSource
        evidence = [EvidenceItem(
            id=f"ima-source-{i}", source=EvidenceSource.IMA,
            claim=s.snippet or s.title or s.source, source_ref=s.source,
            confidence=s.confidence, metadata=s.to_dict(),
        ) for i, s in enumerate(self.sources)]
        return ContractImaAnswer(
            answer=self.answer, source=self.source, confidence=self.confidence,
            assumptions=list(self.assumptions), evidence=evidence,
            request_id=self.request_id,
        )
