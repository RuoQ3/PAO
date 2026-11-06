"""Replaceable IMA transports. HTTP uses only the Python standard library."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from .errors import ImaAnswerError, ImaAuthError, ImaServiceUnavailableError, ImaTimeoutError
from .models import ImaRequest


class ImaTransport(Protocol):
    def health(self) -> dict[str, Any]: ...
    def capabilities(self) -> dict[str, Any]: ...
    def request(self, request: ImaRequest) -> dict[str, Any]: ...
    def close(self) -> None: ...


class MockImaTransport:
    def __init__(self, response: dict[str, Any] | None = None, *, error: Exception | None = None):
        self.response = response or {"answer": "mock answer", "source": "mock", "sources": []}
        self.error = error
        self.closed = False

    def health(self) -> dict[str, Any]: return {"status": "ok", "transport": "mock"}
    def capabilities(self) -> dict[str, Any]: return {"query": True, "sources": True}
    def request(self, request: ImaRequest) -> dict[str, Any]:
        if self.error: raise self.error
        return dict(self.response)
    def close(self) -> None: self.closed = True


class HttpImaTransport:
    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 10,
                 auth_header: str = "Authorization"):
        self.base_url = base_url.rstrip("/")
        self.token, self.timeout, self.auth_header = token, timeout, auth_header

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                try:
                    return json.loads(response.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ImaAnswerError("IMA response is not valid JSON") from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403): raise ImaAuthError("IMA authentication rejected") from exc
            raise ImaServiceUnavailableError(f"IMA HTTP error {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError) or isinstance(exc, TimeoutError):
                raise ImaTimeoutError("IMA request timed out") from exc
            raise ImaServiceUnavailableError("IMA endpoint unavailable") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.token: headers[self.auth_header] = f"Bearer {self.token}"
        return headers

    def health(self) -> dict[str, Any]: return self._get("/health")
    def capabilities(self) -> dict[str, Any]: return self._get("/capabilities")
    def request(self, request: ImaRequest) -> dict[str, Any]:
        body = json.dumps(request.to_payload()).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/query", data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                try:
                    return json.loads(response.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ImaAnswerError("IMA response is not valid JSON") from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403): raise ImaAuthError("IMA authentication rejected") from exc
            raise ImaServiceUnavailableError(f"IMA HTTP error {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError) or isinstance(exc, TimeoutError):
                raise ImaTimeoutError("IMA request timed out") from exc
            raise ImaServiceUnavailableError("IMA endpoint unavailable") from exc
    def close(self) -> None: pass


class McpImaTransport:
    """MCP hook; inject a callable/adapter when the team's MCP protocol is known."""
    def __init__(self, caller: Any): self.caller = caller
    def health(self) -> dict[str, Any]: return self.caller("health", {})
    def capabilities(self) -> dict[str, Any]: return self.caller("capabilities", {})
    def request(self, request: ImaRequest) -> dict[str, Any]: return self.caller("query", request.to_payload())
    def close(self) -> None: return None
