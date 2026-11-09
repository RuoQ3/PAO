"""Strict, provider-independent intent response boundary; no I/O or retries."""
from __future__ import annotations

import json
import re
from typing import Any

from src.agents.orchestration.models import IntentSchema


class IntentResponseAdapter:
    """Extract and validate responses, and construct explicit repair requests."""

    def parse(self, raw: str) -> IntentSchema:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("empty_model_result")
        payload = self.validate_provider_payload(self.parse_json_response(raw))
        return IntentSchema.from_dict(payload)

    @staticmethod
    def build_repair_request(*, question: str, focus: dict[str, Any],
                             raw: Any, error: dict[str, str]) -> str:
        return json.dumps({
            "request_type": "repair_intent_json", "original_question": question,
            "conversation_focus": focus, "invalid_response": raw,
            "validation_error": error,
            "instruction": "Return only the corrected complete JSON object. No explanation or Markdown. "
                           "Keep the original user intent; do not invent missing slots or bypass confirmation.",
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
    def validate_provider_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate the wire contract without coercing invalid provider fields."""
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
        elif analysis:
            raise ValueError("analysis must be {} for ordinary intents")
        return payload

