"""Structured intent classification for the single PAO LeadAgent entry."""
from __future__ import annotations

import json
import re
import logging
from typing import Any, Callable

from src.agents.orchestration.models import (
    Classification,
    IntentSchema,
    SubagentRequest,
)
from src.agents.errors import AgentInternalError, AgentUnavailableError, unavailable, failure_from, mark_unavailable, ensure_session_available
from src.agents.reasoning import configured, redact
from . import llm_client
from .intent_response_adapter import IntentResponseAdapter


class IntentClassifier:
    """Classify every turn through one strictly validated LLM path."""

    def __init__(self, *, llm_config: Any = None, llm_call: Callable[..., str] | None = None,
                 confidence_threshold: float = 0.55, logger: Any = None, enabled: bool = True):
        self.llm_config = llm_config
        self.llm_call = llm_call or llm_client.chat_json
        self.response_adapter = IntentResponseAdapter()
        if type(confidence_threshold) not in (int, float) or not .55 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [.55, 1]")
        self.confidence_threshold = float(confidence_threshold)
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = enabled

    def classify(self, question: str, *, session: Any = None) -> Classification:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("请输入非空问题。")
        if session is not None:
            ensure_session_available(session)
        attempts = []
        try:
            return self._classify(question, session=session, attempts=attempts)
        except AgentUnavailableError as error:
            if session is not None:
                mark_unavailable(session, error)
                session.classification_retry_count = error.retry_count
                session.audit_events.append({"event": "intent_classification_trace", "retry_count": error.retry_count,
                    "failure_code": error.code, "audit_id": error.audit_id,
                    "attempts": self._public_attempts(attempts)})
            self.logger.error("%s: %s", "agent_internal_error" if isinstance(error, AgentInternalError) else "agent_unavailable", error.to_dict())
            raise

    @staticmethod
    def _public_attempts(attempts):
        return [{"attempt": a["attempt"], "error": {
            k: v for k, v in a.get("error", {}).items() if k in {"code", "cause_type"}}}
            for a in attempts]

    def _classify(self, question: str, *, session: Any = None, attempts: list) -> Classification:

        def finish(result, retries=0):
            result.retry_count = retries
            if session is not None:
                session.intent = result.intent
                session.classification_source = result.classification_source
                session.classification_retry_count = retries
                session.audit_events.append({
                    "event": "intent_classification_trace", "classification_source": result.classification_source,
                    "retry_count": retries, "attempts": self._public_attempts(attempts),
                })
            self.logger.debug("intent classification trace: %s", {
                "source": result.classification_source,
                "retry_count": retries, "attempts": attempts}, extra={"pao_file_only": True})
            return result

        if not isinstance(question, str) or not question.strip():
            raise unavailable("agent_output_invalid", stage="intent_input", user_message="请输入非空问题。")
        cfg = configured(self.llm_config, enabled=self.enabled)
        focus = ((getattr(session, "runtime_state", {}) or {}).get("conversation_focus") or {})
        request = json.dumps({"question": question, "conversation_focus": focus}, ensure_ascii=False)
        for attempt in range(2):
            trace = {"attempt": attempt, "request": redact(request, cfg)}
            attempts.append(trace)
            try:
                raw = self.llm_call(cfg, system=self._system_prompt(), user=request)
            except Exception as exc:
                trace["error"] = {"code": self._reason_code(exc), "cause_type": type(exc).__name__}
                self.logger.error("intent provider request failed (attempt=%s, type=%s)", attempt, type(exc).__name__)
                error = failure_from(exc, "intent_request", attempt)
                error.retry_count = attempt
                raise error from None
            trace["raw_response"] = redact(raw, cfg)
            self.logger.debug("intent raw response attempt=%s: %r", attempt, redact(raw, cfg), extra={"pao_file_only": True})
            try:
                intent = self.response_adapter.parse(redact(raw, cfg), validator=self._validate_provider_payload)
            except (ValueError, TypeError, KeyError) as exc:
                error = {"code": self._reason_code(exc), "cause_type": type(exc).__name__, "message": redact(str(exc), cfg)}
                trace["error"] = error
                trace["validation_error"] = error
                self.logger.debug("intent validation failed attempt=%s: %s", attempt, error, extra={"pao_file_only": True})
                if attempt == 1:
                    raise unavailable("llm_repair_failed",
                                      stage="intent_validation", retry_count=1, cause=exc) from None
                request = self.response_adapter.build_repair_request(
                    question=redact(question, cfg), focus=redact(focus, cfg), raw=redact(raw, cfg), error=error)
                self.logger.debug("intent repair request: %s", redact(request, cfg), extra={"pao_file_only": True})
                continue
            if intent.confidence < self.confidence_threshold and not intent.missing_information:
                trace["error"] = {"code": "low_confidence", "message": "confidence below threshold"}
                raise unavailable("llm_low_confidence", stage="intent_validation", retry_count=attempt)
            return finish(Classification(intent=intent,
                classification_source="llm_repaired" if attempt else "llm"), attempt)
        raise AssertionError("unreachable classification state")

    def _validate_provider_payload(self, payload):
        return self.response_adapter.validate_provider_payload(payload)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "For process feed/product/section purpose questions choose analysis and process_advisor. "
            "Current project questions require process_context. context_required only allows process_context, yaml, node_db, simulation_db, history, files, none. "
            "analysis must be an object; ordinary intents use {}. "
            "Classify the PAO user message. Return one JSON object only, with exactly these fields: "
            "intent, task_type, project_scope, context_required, subagents, action_type, "
            "requires_confirmation, confidence, missing_information, reason, analysis. "
            "analysis MUST be an object (never string, array or null); ordinary intents return analysis: {}. "
            "context_required and missing_information are arrays of strings; requires_confirmation is boolean; "
            "confidence is a finite number between 0 and 1; reason is a string. "
            "subagents is a list of {name,task,priority,parallel}. "
            "Allowed intent: chat, project_query, analysis, parameter_impact, modify_request, simulation_request, "
            "optimization_request, report_request, code_request, general_question. "
            "Allowed project_scope: none, current_project, current_case, current_session, external. "
            "Allowed action_type: none, read, modify_file, set_parameter, simulate_once, optimize, modify_structure. "
            "Any modify, simulation, optimization, parameter impact or structure action requires_confirmation=true. "
            "For parameter_impact, put slots in analysis: equipment, target_stream, target_component, metric, candidate_parameters. "
            "All five analysis keys are required for parameter_impact; candidate_parameters is an array of strings. "
            "candidate_parameters is OPTIONAL experimental input: use [] for automatic evidence-based selection. "
            "Never list candidate_parameters in missing_information merely because the user did not name parameters. "
            "Populate it only when the user explicitly restricts the parameter scope. "
            "For unresolved scalar slots use null and list their key in missing_information; never invent values. "
            "Input may be a repair_intent_json request containing an invalid response and validation errors. "
            "Return only a corrected COMPLETE JSON object, no explanation and no Markdown. "
            'Example JSON for a greeting: {"intent":"chat","task_type":"chat",'
            '"project_scope":"none","context_required":[],"subagents":[],"action_type":"none",'
            '"requires_confirmation":false,"confidence":0.95,"missing_information":[], '
            '"reason":"The user greets the assistant.","analysis":{}}'
        )

    @staticmethod
    def _reason_code(exc: Exception) -> str:
        if str(exc) == "empty_model_result":
            return "empty_model_result"
        if isinstance(exc, json.JSONDecodeError):
            return "invalid_json"
        if isinstance(exc, (ValueError, TypeError)):
            return "schema_validation_failed"
        if isinstance(exc, TimeoutError):
            return "request_timeout"
        return "llm_api_failure"

