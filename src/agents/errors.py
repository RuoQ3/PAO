"""Shared fail-closed errors for every Agent boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
import uuid
import functools
import logging
import json
import inspect


@dataclass
class AgentUnavailableError(RuntimeError):
    """A request cannot be answered by a verified Agent pipeline."""

    code: str
    user_message: str = "Agent unavailable; the request was stopped."
    stage: str = "unknown"
    retry_count: int = 0
    cause_type: str | None = None
    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.user_message)

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "user_message": self.user_message,
                "stage": self.stage, "retry_count": self.retry_count,
                "cause_type": self.cause_type, "audit_id": self.audit_id}


class AgentInternalError(AgentUnavailableError):
    """Protocol/routing defect, distinct from an unavailable component."""


def _failure_message(code: str, stage: str) -> str:
    reasons = {
        "llm_low_confidence": "现有证据不足以形成可靠回答，请明确问题范围或补充已保存的执行记录",
        "llm_timeout": "LLM 请求超时",
        "llm_request_failed": "LLM 服务请求失败",
        "llm_provider_unavailable": "LLM 服务依赖不可用",
        "llm_invalid_json": "LLM 返回的 JSON 协议无效",
        "llm_schema_invalid": "LLM 返回的字段格式无效",
        "llm_repair_failed": "LLM 协议修复未通过",
    }
    reason = reasons.get(code, "Agent 无法完成本次请求")
    return f"{reason}（{code}，阶段 {stage}）。本次请求已停止；已有结果保留在审计中。请修正后使用新会话重试。"


def internal_error(code: str, *, stage: str) -> AgentInternalError:
    return AgentInternalError(code=code, stage=stage,
        user_message=_failure_message(code, stage))


def unavailable(code: str, *, stage: str, retry_count: int = 0,
                cause: BaseException | None = None,
                user_message: str | None = None) -> AgentUnavailableError:
    protocol_error = code in {"agent_output_invalid", "llm_invalid_json", "llm_schema_invalid", "llm_repair_failed"}
    error_type = AgentInternalError if protocol_error else AgentUnavailableError
    return error_type(
        code=code,
        user_message=user_message or _failure_message(code, stage),
        stage=stage, retry_count=retry_count,
        cause_type=type(cause).__name__ if cause else None,
    )


def failure_from(exc: Exception, stage: str, retry_count: int = 0) -> AgentUnavailableError:
    if isinstance(exc, AgentUnavailableError):
        return exc
    code = ("llm_timeout" if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
            else "llm_invalid_json" if isinstance(exc, json.JSONDecodeError)
            else "llm_schema_invalid" if isinstance(exc, (ValueError, TypeError, KeyError))
            else "llm_provider_unavailable" if isinstance(exc, ImportError)
            else "llm_request_failed")
    return unavailable(code, stage=stage, retry_count=retry_count, cause=exc)


def fail_closed(stage: str):
    """Normalize failures without putting provider exception text in public errors."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                error = failure_from(exc, stage)
                logging.getLogger(__name__).error("%s: %s", "agent_internal_error" if isinstance(error, AgentInternalError) else "agent_unavailable", error.to_dict())
                raise error from None
        return wrapped
    return decorate


def mark_unavailable(session, error: AgentUnavailableError) -> None:
    if session.pending_task is not None and session.pending_task.phase != "failed":
        session.pending_task.phase = "failed"
        session.pending_task.audit.append({"event": "task_failed", "code": error.code})
    if isinstance(error, AgentInternalError):
        session.internal_error = error.to_dict()
        session.pending_approval = None
        session.runtime_state.pop("active_plan", None)
        if not any(e.get("audit_id") == error.audit_id for e in session.audit_events):
            session.audit_events.append({"event": "agent_internal_error", **error.to_dict()})
        return
    session.agent_unavailable = True
    session.classification_source = "agent_unavailable"
    session.unavailable_error = error.to_dict()
    session.pending_approval = None
    session.runtime_state.pop("active_plan", None)
    if not any(e.get("audit_id") == error.audit_id for e in session.audit_events):
        session.audit_events.append({"event": "agent_unavailable", **error.to_dict()})


def ensure_session_available(session) -> None:
    if session.internal_error:
        raise AgentInternalError(**session.internal_error)
    if session.agent_unavailable or session.classification_source not in {None, "llm", "llm_repaired"}:
        raise AgentUnavailableError(**(session.unavailable_error or {
            "code": "agent_output_invalid", "stage": "session_restore",
            "user_message": "Agent unavailable: restart the session after fixing LLM configuration."}))


def session_boundary(fn):
    """Persist one terminal failure consistently across public session APIs."""
    signature = inspect.signature(fn)
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        session = bound.arguments.get("session") or getattr(self, "state", None)
        try:
            if session is not None:
                ensure_session_available(session)
            if getattr(self, "use_llm", True) is False:
                raise unavailable("llm_not_configured", stage="llm_disabled")
            return fn(self, *args, **kwargs)
        except AgentUnavailableError as error:
            if session is not None:
                owner = getattr(self, "lead", self)
                invalidate = getattr(owner, "invalidate_approval", None)
                if invalidate is not None:
                    invalidate(session)
                mark_unavailable(session, error)
                getattr(self, "_pending_closed_loops", {}).pop(session.session_id, None)
            raise error from None
    return wrapped
