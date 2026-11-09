import json

import pytest

from src.agents.lead.intent import IntentClassifier
from src.agents.orchestration.models import SessionState
from test_stage2_intent import _cfg, _payload


def run_responses(responses, question='请分析 T0301 参数影响', session=None):
    calls = []
    iterator = iter(responses)
    def provider(cfg, **kwargs):
        calls.append(kwargs)
        return next(iterator)
    result = IntentClassifier(llm_config=_cfg(), llm_call=provider).classify(question, session=session)
    return result, calls


def impact_payload():
    return _payload(intent='parameter_impact', action_type='simulate_once', requires_confirmation=True,
                    analysis={'equipment': 'T0301', 'target_stream': '0319', 'target_component': 'ADN',
                              'metric': 'mole_fraction', 'candidate_parameters': ['T0301_BF']})


def test_legal_deepseek_slots_and_context_are_preserved_without_retry():
    state = SessionState(runtime_state={'conversation_focus': {'equipment': 'T0301'}})
    result, calls = run_responses([json.dumps(impact_payload())], session=state)
    assert result.classification_source == 'llm' and result.retry_count == 0
    assert result.intent.analysis['target_stream'] == '0319'
    assert result.intent.requires_confirmation is True
    assert json.loads(calls[0]['user'])['conversation_focus']['equipment'] == 'T0301'
    assert len(calls) == 1


@pytest.mark.parametrize('invalid', ['bad', [], None])
def test_analysis_type_requires_one_repair_and_audits_raw_response(invalid):
    state = SessionState()
    bad = json.dumps(_payload(analysis=invalid))
    good = json.dumps(_payload())
    result, calls = run_responses([bad, good], session=state)
    assert result.classification_source == 'llm_repaired'
    assert result.retry_count == state.classification_retry_count == 1
    repair = json.loads(calls[1]['user'])
    assert repair['request_type'] == 'repair_intent_json'
    assert 'analysis must be an object' in repair['validation_error']['message']
    assert repair['invalid_response'] == bad
    trace = state.audit_events[-1]
    assert trace['attempts'][0]['raw_response'] == bad
    assert trace['attempts'][1]['raw_response'] == good
    restored = SessionState.from_json(state.to_json())
    assert restored.classification_source == 'llm_repaired'
    assert restored.classification_retry_count == 1


@pytest.mark.parametrize('bad,reason', [
    ('not JSON', 'invalid_json'),
    (json.dumps(_payload(intent='illegal')), 'schema_validation_failed'),
    (json.dumps(_payload(action_type='optimize', requires_confirmation=False)), 'schema_validation_failed'),
    (json.dumps(_payload(analysis=None)), 'schema_validation_failed'),
    ('[]', 'schema_validation_failed'),
])
def test_invalid_response_is_repaired_once_or_falls_back_with_reason(bad, reason):
    result, calls = run_responses([bad, json.dumps(_payload())])
    assert result.classification_source == 'llm_repaired' and len(calls) == 2
    assert json.loads(calls[1]['user'])['validation_error']['code'] == reason
    state = SessionState(runtime_state={'conversation_focus': {'equipment': 'T0301', 'stream': '0319', 'component': 'ADN', 'metric': 'mole_fraction'}})
    result, calls = run_responses([bad, bad], session=state)
    assert result.classification_source == 'rule_fallback'
    assert result.retry_count == 1 and len(calls) == 2
    assert result.fallback_reason == reason
    assert result.intent.analysis['target_component'] == 'ADN'
    assert result.intent.requires_confirmation is True


def test_missing_analysis_and_missing_impact_slots_are_not_filled_silently():
    missing = _payload()
    del missing['analysis']
    result, calls = run_responses([json.dumps(missing), json.dumps(impact_payload())])
    assert result.classification_source == 'llm_repaired'
    assert 'analysis' in json.loads(calls[1]['user'])['validation_error']['message']
    incomplete = impact_payload()
    del incomplete['analysis']['metric']
    result, calls = run_responses([json.dumps(incomplete)]*2)
    assert result.classification_source == 'rule_fallback'
    assert 'metric' in json.loads(calls[1]['user'])['validation_error']['message']


def test_missing_key_has_zero_provider_calls(monkeypatch):
    monkeypatch.setattr('src.agents.lead.intent.llm_client.is_configured', lambda cfg: False)
    result, calls = run_responses([])
    assert result.fallback_reason == 'llm_key_missing'
    assert result.retry_count == 0 and not calls


def test_repair_provider_failure_is_distinguished():
    calls = []
    def provider(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return 'bad JSON'
        raise TimeoutError('repair request timed out')
    result = IntentClassifier(llm_config=_cfg(), llm_call=provider).classify('分析 T0301 参数')
    assert result.retry_count == 1 and len(calls) == 2
    assert result.fallback_reason == 'request_timeout'


def test_repaired_action_still_requires_approval():
    from src.agents.lead import ConversationSession, LeadAgent
    payload = _payload(intent='simulation_request', action_type='simulate_once', requires_confirmation=True)
    responses = iter(['invalid JSON', json.dumps(payload)])
    classifier = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **kw: next(responses))
    session = ConversationSession(SessionState(), LeadAgent(intent_classifier=classifier))
    answer = session.submit('运行仿真')
    assert session.state.classification_source == 'llm_repaired'
    assert answer.approval_request is not None
    assert session.state.pending_approval is not None
    assert session.state.pending_approval.status == 'pending'
