"""Adapter and real urllib request construction, with no external API calls."""
from agent_fakes import fake_kwargs, CFG, FakeTechnicalAdvisor
from src.agents.errors import AgentUnavailableError
import io
import json

import pytest

from src.agents.lead import llm_client
from src.agents.lead.intent import IntentClassifier
from src.agents.lead.intent_response_adapter import IntentResponseAdapter
from src.agents.orchestration.models import SessionState
from test_stage2_intent import _payload
from test_intent_repair import impact_payload


def config():
    return llm_client.LLMConfig(provider='deepseek', model='deepseek-flash',
                                api_key_env='DEEPSEEK_API_KEY', api_key='test-only')


def transport(monkeypatch, responses):
    calls = []
    responses = iter(responses)
    def urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': next(responses)}}]}).encode())
    monkeypatch.setattr(llm_client.urllib.request, 'urlopen', urlopen)
    return calls


def test_standalone_adapter_preserves_slots_and_parses_fences():
    result = IntentResponseAdapter().parse('```json\n' + json.dumps(impact_payload()) + '\n```')
    assert result.analysis == impact_payload()['analysis']
    assert result.requires_confirmation is True


@pytest.mark.parametrize('payload, message', [
    (_payload(analysis=None), 'analysis must be an object'),
    (_payload(analysis=[]), 'analysis must be an object'),
    (_payload(analysis='bad'), 'analysis must be an object'),
    (_payload(intent='illegal'), 'intent'),
    (_payload(action_type='optimize', requires_confirmation=False), 'confirmation'),
])
def test_standalone_adapter_rejects_wire_and_business_errors(payload, message):
    with pytest.raises((ValueError, TypeError), match=message):
        IntentResponseAdapter().parse(json.dumps(payload))


def test_standalone_repair_request_preserves_diagnostics_and_context():
    error = {'code': 'invalid_json', 'message': 'Unexpected token'}
    request = json.loads(IntentResponseAdapter.build_repair_request(
        question='分析塔参数', focus={'equipment': 'T0301'}, raw='broken', error=error))
    assert request['validation_error'] == error
    assert request['invalid_response'] == 'broken'
    assert request['original_question'] == '分析塔参数'
    assert request['conversation_focus'] == {'equipment': 'T0301'}
    assert 'No explanation or Markdown' in request['instruction']


@pytest.mark.parametrize('bad', ['not JSON', '   ', None, json.dumps(_payload(analysis=[]))])
def test_native_json_mode_on_initial_and_repair_requests(monkeypatch, bad):
    calls = transport(monkeypatch, [bad, json.dumps(_payload())])
    state = SessionState(runtime_state={'conversation_focus': {'equipment': 'T0301'}})
    result = IntentClassifier(llm_config=config()).classify('设备有什么作用？', session=state)
    assert result.classification_source == 'llm_repaired'
    assert result.retry_count == 1
    assert len(calls) == 2
    for call in calls:
        assert call['response_format'] == {'type': 'json_object'}
        assert 'JSON' in call['messages'][0]['content']
        assert 'Example JSON' in call['messages'][0]['content']
    repair = json.loads(calls[1]['messages'][-1]['content'])
    assert repair['request_type'] == 'repair_intent_json'
    assert repair['conversation_focus'] == {'equipment': 'T0301'}
    assert state.audit_events[-1]['attempts'][0]['error']['cause_type']


def test_native_json_mode_valid_response_needs_no_retry(monkeypatch):
    calls = transport(monkeypatch, [json.dumps(impact_payload())])
    result = IntentClassifier(llm_config=config()).classify('分析 T0301 参数影响')
    assert result.classification_source == 'llm'
    assert result.retry_count == 0 and len(calls) == 1
    assert calls[0]['response_format'] == {'type': 'json_object'}
    assert result.intent.requires_confirmation is True


def test_native_json_mode_stops_after_failed_repair(monkeypatch):
    calls = transport(monkeypatch, ['{', '{'])
    with pytest.raises(AgentUnavailableError) as error:
        IntentClassifier(llm_config=config()).classify('分析 T0301 参数影响')
    assert error.value.code == 'llm_repair_failed'
    assert error.value.retry_count == 1 and len(calls) == 2


def test_normal_chat_does_not_enable_json_mode(monkeypatch):
    calls = transport(monkeypatch, ['普通回答'])
    assert llm_client.chat(config(), 'Answer naturally', '你好') == '普通回答'
    assert 'response_format' not in calls[0]


def test_other_provider_retains_existing_text_transport(monkeypatch):
    class Model:
        def invoke(self, messages):
            assert len(messages) == 2
            return type('Reply', (), {'content': json.dumps(_payload())})()
    monkeypatch.setattr(llm_client, 'build_chat_model', lambda cfg: Model())
    cfg = llm_client.LLMConfig(provider='anthropic', model='test', api_key_env='TEST', api_key='test')
    result = IntentClassifier(llm_config=cfg).classify('设备有什么作用？')
    assert result.classification_source == 'llm'
