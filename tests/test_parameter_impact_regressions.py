from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.actions import ActionGateway, ModelIdentity, ModelSchema
from src.agents.lead import ConversationSession, LeadAgent
from src.agents.orchestration.models import ActionProposal, Answer, ApprovalRequest, SessionState
from src.agents.orchestration.parameter_impact import (
    AspenSandboxRunner, FakeAspenSandboxRunner, ParameterImpactPlan,
    SensitivityAnalyzer, design_perturbations, make_simulation_plan,
)
from src.models.process_case import ProcessCase, CaseStatus
from src.models.stream import ComponentFlow, StreamResult, StreamType, StreamConvergenceStatus

PATH = r'\Data\Blocks\T0301\Input\B:F'
OUTPUT = r'\Data\Streams\0319\Output\STR_MAIN'


def impact():
    return ParameterImpactPlan('T0301', '0319', 'ADN', 'mole_fraction', [
        dict(name='BF', aspen_path=PATH, baseline=.6, lower=.599964, upper=.600036, unit='-')
    ], expected_outputs=[OUTPUT], output_spec={
        'metric': 'mole_fraction', 'stream_id': '0319', 'component': 'ADN',
        'extractor': 'stream_component_mole_fraction', 'aspen_paths': [OUTPUT],
    })


def simulation(tmp_path):
    model = tmp_path / 'model.bkp'
    model.write_bytes(b'original')
    plan = make_simulation_plan(impact(), base_model=str(model))
    plan.approved = True
    return plan


def test_clipped_steps_are_deduplicated_before_budget_check():
    p = impact()
    p.max_cases = 3
    cases = design_perturbations(p)
    assert [c['inputs'][PATH] for c in cases] == [.6, .599964, .600036]


@pytest.mark.parametrize('output', [
    {'success': False, 'error': 'T0302 ERRORS'},
    {'success': True, 'outputs': {}},
    {'success': True, 'outputs': {'mole_fraction': float('nan')}},
])
def test_baseline_failure_stops_and_cleans_up(tmp_path, output):
    plan = simulation(tmp_path)
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        return output
    results = AspenSandboxRunner(evaluate, allowed_paths=[PATH]).run(plan)
    assert len(calls) == len(results) == 1
    assert '基准预检失败' in results[0].errors[-1]
    assert not Path(plan.temporary_model).exists()
    assert Path(plan.base_model).read_bytes() == b'original'


def stream(name, fraction):
    return StreamResult(name, StreamType.MATERIAL, StreamConvergenceStatus.SUCCESS,
                        components=[ComponentFlow('ADN', mole_frac=fraction)])


def test_real_process_case_extracts_exact_target_and_preserves_identity(tmp_path):
    plan = simulation(tmp_path)
    raw = ProcessCase(status=CaseStatus.SUCCESS, streams={
        '0320': stream('0320', .99), '0319': stream('0319', .001)})
    original_id = raw.case_id
    result = AspenSandboxRunner._coerce(raw, plan.parameter_changes[0], Path('copy.bkp'), 0, plan=plan)
    assert result.case_id == original_id
    assert result.impact_outputs['mole_fraction'] == .001
    raw = ProcessCase(status=CaseStatus.SUCCESS, streams={'0320': stream('0320', .99)})
    result = AspenSandboxRunner._coerce(raw, plan.parameter_changes[0], Path('copy.bkp'), 0, plan=plan)
    assert 'mole_fraction' not in result.impact_outputs
    assert any('目标指标不可用' in x for x in result.warnings)


def test_real_failure_preserves_aspen_diagnostics(tmp_path):
    plan = simulation(tmp_path)
    raw = ProcessCase(status=CaseStatus.SIM_FAILED,
                      sim_result=SimpleNamespace(error='T0302 ERRORS: convergence failed', warnings=['detail']))
    result = AspenSandboxRunner._coerce(raw, plan.parameter_changes[0], Path('copy.bkp'), 0, plan=plan)
    assert 'T0302' in result.errors[0]
    assert 'detail' in result.warnings


def case(x, z, y, name='case'):
    return ProcessCase(case_id=name, status=CaseStatus.SUCCESS, converged=True,
                       impact_inputs={'x': x, 'z': z}, impact_outputs={'m': y})


def test_ofat_statistics_exclude_other_factors_and_duplicates():
    results = [case(1, 1, 10, 'baseline'), case(.9, 1, 9), case(1.1, 1, 11),
               case(.9, 1, 9), case(1, .9, -1000), case(1, 1.1, 1000)]
    report = SensitivityAnalyzer().analyze(results, metric='m', parameters=['x', 'z'])
    x = next(r for r in report['parameter_rank'] if r['parameter'] == 'x')
    assert x['local_slope'] == pytest.approx(10)
    assert x['metric_change'] == 2
    assert x['sample_count'] == 3
    assert x['confidence'] == 'not_estimated'
    results[0].converged = False
    assert SensitivityAnalyzer().analyze(results, metric='m', parameters=['x'])['parameter_rank'] == []


def approved_session(tmp_path, evaluator):
    plan = simulation(tmp_path)
    gateway = ActionGateway(ModelSchema(ModelIdentity('model'), targets=('model',)),
                            simulation_runner=FakeAspenSandboxRunner(evaluator))
    proposal = ActionProposal('run_simulation', 'model', reason='test impact', payload={
        'simulation_plan': plan.to_dict(), 'parameter_impact_plan': impact().to_dict()})
    preview = gateway.preview_proposal(proposal)
    assert preview['valid']
    state = SessionState(pending_approval=ApprovalRequest(proposal, preview=preview))
    return ConversationSession(state, LeadAgent(action_gateway=gateway, use_llm=False))


@pytest.mark.parametrize('success', [True, False])
def test_confirmation_returns_answer_and_history_without_reexecution(tmp_path, success):
    calls = []
    def evaluate(inputs):
        calls.append(inputs)
        return {'success': success, 'outputs': {'mole_fraction': inputs[PATH]}, 'error': '' if success else 'T0302 ERRORS'}
    session = approved_session(tmp_path, evaluate)
    answer = session.submit('确认')
    assert isinstance(answer, Answer)
    assert answer.recommendations == []
    assert session.state.history[-1]['content'] == answer.text
    assert session.state.pending_approval is None
    assert len(calls) == (3 if success else 1)
    assert 'T0302' in answer.text if not success else '局部敏感性' in answer.text
    count = len(calls)
    session.submit('确认')
    assert len(calls) == count


def test_assistant_description_does_not_set_user_metric():
    lead = SimpleNamespace(answer=lambda *args: Answer('ADN 富集，mole fraction 0.9'))
    session = ConversationSession(SessionState(), lead)
    session.submit('T0301 有哪些物流？')
    assert 'metric' not in session.state.runtime_state['conversation_focus']


def test_gateway_exception_returns_failure_answer(tmp_path):
    session = approved_session(tmp_path, lambda inputs: {})
    session.lead.action_gateway.execute = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('database unavailable'))
    answer = session.submit('确认')
    assert isinstance(answer, Answer)
    assert 'database unavailable' in answer.text
    assert session.state.pending_approval is None


def test_real_runner_reads_composition_as_subtree_and_closes_driver(tmp_path, monkeypatch):
    import src.workflows.run_case as workflow
    plan = simulation(tmp_path)
    events = []
    driver = SimpleNamespace(connect=lambda: events.append('connect'),
                             open=lambda path: events.append('open'),
                             disconnect=lambda: events.append('disconnect'),
                             close=lambda: events.append('close'))
    def run_case(driver_arg, inputs, cfg):
        assert driver_arg is driver
        assert cfg.output_paths == []
        assert cfg.extract_streams == ['0319']
        assert cfg.check_status_paths is None  # retain full-flow checks
        return ProcessCase(status=CaseStatus.SUCCESS, streams={'0319': stream('0319', .01)})
    monkeypatch.setattr(workflow, 'run_case', run_case)
    results = AspenSandboxRunner(driver_factory=lambda path: driver, allowed_paths=[PATH]).run(plan)
    assert len(results) == 3
    assert all(c.impact_outputs['mole_fraction'] == .01 for c in results)
    assert events == ['connect', 'open', 'disconnect', 'close'] * 3


def test_cli_confirmation_failure_is_printed_and_loop_survives(tmp_path, monkeypatch, capsys):
    import logging
    # main configures console streams on import; avoid touching pytest's capture.
    with capsys.disabled():
        import src.main as cli
    import src.agents.lead as lead_module
    import src.agents.context as context_module
    session = approved_session(tmp_path, lambda inputs: {'success': False, 'error': 'T0302 ERRORS'})
    monkeypatch.setattr(lead_module, 'ConversationSession', lambda *args: session)
    builder = SimpleNamespace(build=lambda: SimpleNamespace(), close=lambda: None)
    monkeypatch.setattr(context_module, 'ProcessContextBuilder', lambda **kwargs: builder)
    turns = iter(['确认', 'hello', 'exit'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(turns))
    assert cli._run_unified_chat_loop(config_path=tmp_path/'model.bkp', yaml_path=None,
        db_path=tmp_path/'simulation.db', node_db_path=None,
        initial_intent='', log=logging.getLogger('test')) == 0
    output = capsys.readouterr().out
    assert 'T0302' in output
    assert '未执行 2 个' in output
    assert '已退出' in output
    assert '??/??' not in output


def test_analysis_exception_is_visible_in_answer():
    result = {'simulation_cases': [], 'sensitivity': {'error': 'analysis broke'}}
    result['answer'] = LeadAgent._render_parameter_impact_result({}, result)
    answer = ConversationSession._action_answer(result)
    assert 'analysis broke' in answer.text


def test_nonfinite_metrics_are_not_counted_as_effective():
    report = SensitivityAnalyzer().analyze([
        case(1, 1, 10, 'baseline'), case(.9, 1, float('inf')), case(1.1, 1, float('nan')),
    ], metric='m', parameters=['x'])
    assert report['effective_case_count'] == 1
    assert report['missing_output_count'] == 2
    assert report['parameter_rank'] == []
