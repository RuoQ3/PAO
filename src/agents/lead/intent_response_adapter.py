"""Strict, provider-independent intent response boundary; no I/O or retries."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from src.agents.orchestration.models import (
    CONTEXT_REQUIREMENTS,
    CONTEXT_REQUIREMENTS_ORDER,
    IntentSchema,
    validate_context_requirements,
)


class IntentResponseAdapter:
    """Extract and validate responses, and construct explicit repair requests."""

    def parse(self, raw: str, *, validator: Callable | None = None) -> IntentSchema:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("empty_model_result")
        payload = (validator or self.validate_provider_payload)(self.parse_json_response(raw))
        return IntentSchema.from_dict(payload)

    @staticmethod
    def build_repair_request(*, question: str, focus: dict[str, Any],
                             raw: Any, error: dict[str, str]) -> str:
        allowed = list(CONTEXT_REQUIREMENTS_ORDER)
        instruction = (
            "Return only the corrected complete JSON object. No explanation or Markdown. Do not explain. "
            "Keep the original user intent; do not invent missing slots or bypass confirmation. "
            "Do not put equipment, stream or component names in context_required. "
            "For current project questions and parameter_impact, replace context_required with [\"process_context\"]. "
            "For parameter_impact, equipment, target_stream, target_component, metric, candidate_parameters "
            "must all exist inside analysis. For optimization_request preserve explicitly requested algorithm/method, "
            "equipment, component, metric, target and execution inside analysis; other ordinary intents use analysis: {}. "
            "For optimization_request, omit unspecified execution, optimizer_parameters and random_seed; "
            "execution and optimizer_parameters must be JSON objects when provided. "
            "For parameter_impact set action_type to simulate_once and requires_confirmation to true. "
            "Never return an empty context_required for current_project, current_case, or current_session. "
            "The value none must appear alone."
            " Use only registered subagent names listed in the system role catalogue. "
            "Project read/analysis requests require at least one registered subagent; "
            "repair missing or unknown routes yourself, without a host-selected fallback."
        )
        # Explain the actual invalid identifiers without changing the payload
        # or hard-coding a particular equipment/stream/component name.
        try:
            invalid = IntentResponseAdapter.parse_json_response(raw) if isinstance(raw, str) else {}
        except (ValueError, TypeError):
            invalid = {}
        context = invalid.get("context_required")
        error_text = str(error.get("message", ""))
        if (isinstance(context, list) and any(isinstance(value, str) and value not in CONTEXT_REQUIREMENTS for value in context)) or any(
            token in error_text for token in ("equipment", "stream:", "component:")
        ):
            instruction += ' Replace context_required with ["process_context"].'
            error_values = re.findall(r"equipment:([A-Za-z0-9_-]+)", error_text)
            for value in [*(context or []), *error_values]:
                if isinstance(value, str) and ":" in value:
                    identifier = value.split(":", 1)[1]
                    instruction += f" Keep {identifier} in the question/focus/analysis slot, not in context_required."
        return json.dumps({
            "request_type": "repair_intent_json", "original_question": question,
            "conversation_focus": focus, "invalid_response": raw,
            "validation_error": error,
            "allowed_context_required": allowed,
            "repair_examples": {
                "project_query": ["process_context"],
                "equipment_analysis": ["process_context"],
                "parameter_impact": ["process_context"],
                "no_context_chat": ["none"],
            },
            "instruction": instruction,
        }, ensure_ascii=False)

    @staticmethod
    def parse_json_response(raw: str) -> dict[str, Any]:
        """Parse a provider response while tolerating common JSON fences.

        The contract remains strict after extraction: the decoded value must
        be an object and IntentSchema still rejects unknown fields and invalid
        enum/confirmation combinations.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError("intent response must be a JSON object")
        return value

    @staticmethod
    def validate_routes(payload, *, allowed_roles):
        """Reject dispatch-invalid classification while one LLM repair remains."""
        routes = payload["subagents"]
        names = [route.get("name") for route in routes]
        if any(not isinstance(name, str) or name not in allowed_roles for name in names):
            raise ValueError("subagents contains unregistered role; allowed names: " + ", ".join(sorted(allowed_roles)))
        direct_general = (payload["intent"] in {"chat", "general_question"}
                          and payload["project_scope"] in {"none", "external"})
        if payload["action_type"] in {"none", "read"} and not direct_general and not routes:
            raise ValueError("subagents must include at least one registered role for project read/analysis requests")

    @staticmethod
    def validate_provider_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate the wire contract, normalizing documented optional null slots."""
        required = {"intent", "task_type", "project_scope", "context_required", "subagents",
                    "action_type", "requires_confirmation", "confidence", "missing_information", "reason", "analysis"}
        missing = required - payload.keys()
        if missing:
            raise ValueError("missing required fields: " + ", ".join(sorted(missing)))
        for key in ("intent", "task_type", "project_scope", "action_type", "reason"):
            if not isinstance(payload[key], str):
                raise TypeError(f"{key} must be a string")
        for key in ("context_required", "missing_information"):
            if not isinstance(payload[key], list) or not all(isinstance(x, str) for x in payload[key]):
                raise TypeError(f"{key} must be a list of strings")
        context = payload["context_required"]
        scope = payload["project_scope"]
        intent_name = payload["intent"]
        validate_context_requirements(context, intent=intent_name, project_scope=scope)
        if not isinstance(payload["subagents"], list) or not all(isinstance(x, dict) for x in payload["subagents"]):
            raise TypeError("subagents must be a list of objects")
        analysis = payload["analysis"]
        if not isinstance(analysis, dict):
            raise TypeError("analysis must be an object, not " + type(analysis).__name__)
        if payload["intent"] == "parameter_impact":
            slots = {"equipment", "target_stream", "target_component", "metric", "candidate_parameters"}
            absent = slots - analysis.keys()
            if absent:
                raise ValueError("analysis missing required slots: " + ", ".join(sorted(absent)))
            for key in slots - {"candidate_parameters"}:
                value = analysis[key]
                if value is not None and not isinstance(value, str):
                    raise TypeError(f"analysis.{key} must be a string or null")
                if not value and key not in payload["missing_information"]:
                    raise ValueError(f"missing_information must include unresolved slot: {key}")
            if not isinstance(analysis["candidate_parameters"], list) or not all(isinstance(x, str) for x in analysis["candidate_parameters"]):
                raise TypeError("analysis.candidate_parameters must be a list of strings")
        elif payload["intent"] == "optimization_request":
            from src.agents.orchestration.models import validate_optimization_analysis
            payload = {**payload, "analysis": validate_optimization_analysis(analysis)}
        elif analysis:
            raise ValueError("analysis must be {} for ordinary intents")
        return payload

