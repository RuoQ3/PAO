"""High-level IMA client with normalization, limits, retries, and redaction."""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Protocol

from .config import ImaConfig, load_ima_config
from .errors import ImaAnswerError, ImaError, ImaNotConfiguredError
from .models import ImaAnswer, ImaRequest, ImaSource
from .transport import HttpImaTransport, ImaTransport, McpImaTransport, MockImaTransport

log = logging.getLogger(__name__)


class ImaClient(Protocol):
    def health(self) -> dict[str, Any]: ...
    def capabilities(self) -> dict[str, Any]: ...
    def query(self, request: ImaRequest) -> ImaAnswer: ...
    def query_with_sources(self, request: ImaRequest) -> ImaAnswer: ...
    def close(self) -> None: ...


class ImaServiceClient:
    def __init__(self, transport: ImaTransport, *, config: ImaConfig | None = None):
        self.transport, self.config = transport, config or ImaConfig()
        log.debug("IMA client initialized: %s", self.config.redacted())

    @classmethod
    def from_env(cls, config: ImaConfig | None = None) -> "ImaServiceClient":
        cfg = config or load_ima_config()
        if not cfg.configured: raise ImaNotConfiguredError("IMA is not configured: set PAO_IMA_BASE_URL")
        if cfg.transport == "http":
            transport: ImaTransport = HttpImaTransport(cfg.base_url or "", token=cfg.auth_token, timeout=cfg.timeout_seconds, auth_header=cfg.auth_header)
        elif cfg.transport == "mock": transport = MockImaTransport()
        else: raise ImaNotConfiguredError(f"Unsupported or unregistered IMA transport: {cfg.transport}")
        return cls(transport, config=cfg)

    def health(self) -> dict[str, Any]: return self.transport.health()
    def capabilities(self) -> dict[str, Any]: return self.transport.capabilities()
    def query(self, request: ImaRequest) -> ImaAnswer: return self._query(request, with_sources=False)
    def query_with_sources(self, request: ImaRequest) -> ImaAnswer: return self._query(request, with_sources=True)

    def _query(self, request: ImaRequest, *, with_sources: bool) -> ImaAnswer:
        if not request.query.strip(): raise ImaAnswerError("IMA query cannot be empty")
        encoded = json.dumps(request.context, ensure_ascii=False).encode("utf-8")
        if len(encoded) > self.config.max_context_bytes:
            raise ImaAnswerError(f"IMA context exceeds {self.config.max_context_bytes} bytes")
        req = ImaRequest(request.query, request.context, request.model or self.config.model,
                         request.knowledge_base or self.config.knowledge_base,
                         request.request_id or str(uuid.uuid4()))
        last: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                payload = self.transport.request(req)
                return self._parse(payload, req.request_id, with_sources)
            except ImaAnswerError: raise
            except ImaError as exc:
                last = exc
                if attempt >= self.config.retries: raise
                time.sleep(min(0.25 * (2 ** attempt), 2.0))
        raise ImaAnswerError("IMA request failed") from last

    @staticmethod
    def _parse(payload: dict[str, Any], request_id: str | None, with_sources: bool) -> ImaAnswer:
        if not isinstance(payload, dict): raise ImaAnswerError("IMA response must be an object")
        answer = payload.get("answer", payload.get("text"))
        if not isinstance(answer, str) or not answer.strip(): raise ImaAnswerError("IMA returned an empty answer")
        raw_sources = payload.get("sources", []) if with_sources else payload.get("sources", [])
        if not isinstance(raw_sources, list): raise ImaAnswerError("IMA sources must be a list")
        sources = []
        for item in raw_sources:
            if not isinstance(item, dict) or not item.get("source"):
                raise ImaAnswerError("IMA source has invalid format")
            sources.append(ImaSource(source=str(item["source"]), title=item.get("title"), chapter=item.get("chapter"), page=item.get("page"), snippet=item.get("snippet", item.get("quote")), confidence=item.get("confidence"), metadata={k: v for k, v in item.items() if k not in {"source", "title", "chapter", "page", "snippet", "quote", "confidence"}}))
        confidence = payload.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ImaAnswerError("IMA confidence must be between 0 and 1")
        assumptions = payload.get("assumptions", [])
        if not isinstance(assumptions, list) or not all(isinstance(x, str) for x in assumptions): raise ImaAnswerError("IMA assumptions must be strings")
        return ImaAnswer(answer=answer, source=str(payload.get("source", "ima")), confidence=confidence, assumptions=tuple(assumptions), sources=tuple(sources), request_id=payload.get("request_id", request_id), raw_metadata={"provider": payload.get("provider")} if payload.get("provider") else {})

    def close(self) -> None: self.transport.close()


class MockImaClient(ImaServiceClient):
    """Convenience client for tests and local development."""

    def __init__(self, response: dict[str, Any] | None = None, *, config: ImaConfig | None = None):
        super().__init__(MockImaTransport(response), config=config or ImaConfig(transport="mock"))
