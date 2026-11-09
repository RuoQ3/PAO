"""Stage 2 protocol tests using fake LLM responses and no external services."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents import IntentSchema, LeadAgent, SessionState, SubagentRequest
from src.agents.lead.intent import IntentClassifier
from src.agents.orchestration.models import Classification


def _cfg():
    return SimpleNamespace(api_key="fake", provider="deepseek", model="fake-model")


def _payload(**updates):
    payload = {
        "intent": "project_query",
        "task_type": "equipment_query",
        "project_scope": "current_project",
        "context_required": ["process_context"],
        "subagents": [{"name": "process_advisor", "task": "inspect context", "priority": "normal", "parallel": True}],
        "action_type": "read",
        "requires_confirmation": False,
        "confidence": 0.95,
        "missing_information": [],
        "reason": "The user asks about the current process.",
        "analysis": {},
    }
    payload.update(updates)
    return payload


def test_fake_llm_returns_strict_intent_and_source():
    result = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: __import__("json").dumps(_payload())).classify("Which equipment is in this case?")
    assert isinstance(result, Classification)
    assert result.classification_source == "llm"
    assert result.intent.subagents[0].name == "process_advisor"


def test_json_fenced_provider_response_is_parsed_but_schema_remains_strict():
    raw = "```json\n" + __import__("json").dumps(_payload()) + "\n```"
    result = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: raw).classify("Which equipment is in this case?")
    assert result.classification_source == "llm"


def test_provider_scalar_fields_require_repair_instead_of_silent_normalization():
    payload = _payload(
        intent="general_question",
        task_type="clarification",
        project_scope="none",
        context_required=False,
        subagents=[],
        action_type="none",
        confidence=0.9,
        missing_information="",
    )
    result = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: __import__("json").dumps(payload)).classify("hello")
    assert result.classification_source == "rule_fallback"
    assert result.retry_count == 1
    assert result.fallback_reason == "schema_validation_failed"


@pytest.mark.parametrize("raw,reason", [("not json", "invalid_json"), ("", "empty_model_result")])
def test_invalid_llm_output_is_explicit_rule_fallback(raw, reason):
    result = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: raw).classify("Why did the tower fail?")
    assert result.classification_source == "rule_fallback"
    assert result.fallback_reason == reason
    assert result.intent.project_scope != "none"


def test_missing_key_is_the_only_key_gate_for_rule_fallback(monkeypatch):
    monkeypatch.setattr("src.agents.lead.intent.llm_client.is_configured", lambda cfg: False)
    result = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: (_ for _ in ()).throw(AssertionError())).classify("hello")
    assert result.classification_source == "rule_fallback"
    assert result.fallback_reason == "llm_key_missing"


def test_api_failure_is_explicit_fallback_and_is_recorded():
    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    result = IntentClassifier(llm_config=_cfg(), llm_call=fail).classify("为什么这个塔一直跑不出来？")
    assert result.classification_source == "rule_fallback"
    assert result.fallback_reason == "llm_api_failure"
    assert result.intent.project_scope == "current_project"


def test_low_confidence_falls_back_and_preserves_implicit_project_context():
    result = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: __import__("json").dumps(_payload(confidence=0.1))).classify("Why did it fail?")
    assert result.classification_source == "rule_fallback"
    assert result.fallback_reason == "low_confidence"
    assert result.intent.project_scope == "current_project"
    assert result.intent.context_required == ["process_context"]


def test_schema_rejects_unknown_fields_and_inconsistent_confirmation():
    with pytest.raises(ValueError, match="Unknown IntentSchema fields"):
        IntentSchema.from_dict(_payload(extra="reject"))
    with pytest.raises(ValueError, match="require confirmation"):
        IntentSchema.from_dict(_payload(action_type="optimize", requires_confirmation=False))


def test_session_round_trip_preserves_intent_and_fallback_metadata():
    classifier = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: "bad")
    classification = classifier.classify("继续刚才的实验")
    state = SessionState(intent=classification.intent, classification_source=classification.classification_source,
                         fallback_reason=classification.fallback_reason)
    restored = SessionState.from_json(state.to_json())
    assert restored.intent.intent == "project_query"
    assert restored.classification_source == "rule_fallback"


def test_lead_always_records_structured_classification_and_routes_implicit_project_question():
    builder_calls = []

    class Builder:
        def build(self, **kwargs):
            builder_calls.append(kwargs)
            return SimpleNamespace(variables={}, objectives=[], constraints=[], equipment=[], streams=[], diagnostics=[])

    state = SessionState()
    answer = LeadAgent(context_builder=Builder()).answer("刚才那个参数改大之后有什么影响？", state)
    assert state.classification_source == "rule_fallback"
    assert state.intent.project_scope != "none"
    assert builder_calls
    assert answer.task_results
    assert answer.task_results[0].agent == "process_advisor"
