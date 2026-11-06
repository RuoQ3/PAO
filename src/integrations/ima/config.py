"""Environment-only IMA configuration (never part of checkpoint business data)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ImaConfig:
    transport: str = "mock"
    base_url: str | None = None
    auth_token: str | None = None
    timeout_seconds: float = 10.0
    retries: int = 2
    model: str | None = None
    knowledge_base: str | None = None
    max_context_bytes: int = 32_000
    auth_header: str = "Authorization"

    @property
    def configured(self) -> bool:
        return self.transport == "mock" or (self.transport == "http" and bool(self.base_url))

    def redacted(self) -> dict[str, object]:
        return {
            "transport": self.transport, "base_url": self.base_url,
            "auth_configured": bool(self.auth_token), "timeout_seconds": self.timeout_seconds,
            "retries": self.retries, "model": self.model,
            "knowledge_base": self.knowledge_base, "max_context_bytes": self.max_context_bytes,
        }


def _float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default


def _int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default


def load_ima_config() -> ImaConfig:
    return ImaConfig(
        # Missing configuration is explicit; opt in to the local stub with
        # PAO_IMA_TRANSPORT=mock.
        transport=os.getenv("PAO_IMA_TRANSPORT", "none").strip().lower(),
        base_url=os.getenv("PAO_IMA_BASE_URL") or None,
        auth_token=os.getenv("PAO_IMA_AUTH_TOKEN") or None,
        timeout_seconds=max(0.1, _float("PAO_IMA_TIMEOUT_SECONDS", 10.0)),
        retries=max(0, _int("PAO_IMA_RETRIES", 2)),
        model=os.getenv("PAO_IMA_MODEL") or None,
        knowledge_base=os.getenv("PAO_IMA_KNOWLEDGE_BASE") or None,
        max_context_bytes=max(1024, _int("PAO_IMA_MAX_CONTEXT_BYTES", 32_000)),
        auth_header=os.getenv("PAO_IMA_AUTH_HEADER", "Authorization"),
    )
