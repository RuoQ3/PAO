from execution_fakes import run_simulation, simulation_schema
import ast
import logging
import threading
from pathlib import Path

import pytest

from src.agents.orchestration.parameter_impact import AspenSandboxRunner
from src.agents.lead import LeadAgent, ConversationSession
from src.aspen_driver.runner import _decode_history_text, _read_history_diagnostics
from src.utils.logger import setup_logging, LogContext
from test_parameter_impact_regressions import simulation, PATH


@pytest.fixture
def logging_state():
    root = logging.getLogger()
    handlers, filters, level = root.handlers[:], root.filters[:], root.level
    noisy = {n: logging.getLogger(n).level for n in ['skopt', 'sklearn', 'matplotlib', 'PIL', 'urllib3', 'httpx']}
    root.handlers = []
    root.filters = []
    try:
        yield
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers, root.filters = handlers, filters
        root.setLevel(level)
        for name, value in noisy.items():
            logging.getLogger(name).setLevel(value)


def test_conversation_console_and_utf8_file_are_independent(tmp_path, capsys, logging_state):
    path = tmp_path/'run.log'
    setup_logging('INFO', log_file=path, console_level='WARNING', file_level='DEBUG', user_console=True)
    log = logging.getLogger('src.workflows.run_case')
    with LogContext(case_id='test'):
        log.debug('输入校验 参数 缺少')
        log.warning('T0302 block_status mutation_snapshot 内部诊断')
        log.error('内部错误详情')
        log.warning('用户提示：请检查日志', extra={'pao_user': True})
    terminal = capsys.readouterr().err
    assert '用户提示' in terminal
    assert '内部' not in terminal and '输入校验' not in terminal
    contents = path.read_text(encoding='utf-8')
    assert '输入校验 参数 缺少' in contents
    assert 'mutation_snapshot' in contents and '内部错误详情' in contents
    assert contents.count('[case_id=test]') == 4
    assert '\ufffd' not in contents


def test_explicit_console_diagnostics_and_legacy_behavior(tmp_path, capsys, logging_state):
    setup_logging('INFO', log_file=tmp_path/'run.log', console_level='DEBUG', file_level='ERROR')
    logging.getLogger('src.workflows.run_case').debug('调试详情')
    assert '调试详情' in capsys.readouterr().err
    assert '调试详情' not in (tmp_path/'run.log').read_text()
    setup_logging('WARNING')
    logging.getLogger('src.workflows.run_case').warning('兼容原有日志')
    assert '兼容原有日志' in capsys.readouterr().err


@pytest.mark.parametrize('success', [True, False])
def test_progress_counts_and_baseline_stop(tmp_path, success):
    plan = simulation(tmp_path)
    events = []
    runner = AspenSandboxRunner(lambda **kwargs: {'success': success, 'outputs': {'mole_fraction': .1},
                                'evidence': {'run_id': kwargs['run_id'], 'fresh': True,
                                             'target_verified': success, 'target_status': 'success' if success else 'errors',
                                             'failure_kind': None if success else 'global_model'}},
                                allowed_paths=[PATH], progress_callback=events.append)
    results = run_simulation(runner, plan)
    assert events[0]['status'] == 'started'
    assert events[0]['completed'] == 0
    assert all(e['total'] == 3 for e in events)
    assert events[-1]['status'] == ('finished' if success else 'stopped')
    assert events[-1]['completed'] == len(results) == (3 if success else 1)
    assert [e['completed'] for e in events] == sorted(e['completed'] for e in events)


def test_heartbeat_keeps_execution_on_owned_unit_and_stops(tmp_path):
    plan = simulation(tmp_path)
    plan.parameter_changes = plan.parameter_changes[:1]
    beat = threading.Event()
    events = []
    owner = threading.get_ident()
    def progress(event):
        events.append(event)
        if event['status'] == 'running':
            beat.set()
    def evaluate(**kwargs):
        assert threading.get_ident() != owner
        assert beat.wait(2)
        return {'success': True, 'outputs': {'mole_fraction': .1}, 'evidence': {
            'run_id': kwargs['run_id'], 'fresh': True, 'target_verified': True, 'target_status': 'success'}}
    run_simulation(AspenSandboxRunner(evaluate, allowed_paths=[PATH], progress_callback=progress,
                       progress_interval=.01), plan)
    assert any(e['status'] == 'running' for e in events)
    assert events[-1]['status'] == 'finished'
    assert not any(t.name == 'pao-progress' for t in threading.enumerate())


def test_callback_error_does_not_change_results(tmp_path):
    def broken(event):
        raise RuntimeError('presentation failed')
    results = run_simulation(AspenSandboxRunner(lambda **kwargs: {'success': True, 'outputs': {'mole_fraction': .1},
                                'evidence': {'run_id': kwargs['run_id'], 'fresh': True,
                                             'target_verified': True, 'target_status': 'success'}},
                                 allowed_paths=[PATH], progress_callback=broken), simulation(tmp_path))
    assert len(results) == 3 and all(c.converged for c in results)


def test_answer_is_bounded_but_retains_full_diagnostics():
    detail = 'T0302 收敛失败\n' + 'mutation_snapshot input verification\n' * 100
    result = {'simulation_cases': [{'converged': False, 'errors': [detail], 'warnings': []}],
              'sensitivity': {'effective_case_count': 0, 'limitations': ['基准无效']}}
    from src.agents.orchestration.models import Answer
    result['answer'] = Answer('T0302 收敛失败，没有有效指标，无法判断参数影响。详情可查看，完整诊断保留在日志。',
                              uncertainties=[detail], details=result.copy()).to_dict()
    answer = ConversationSession._action_answer(result)
    assert 'T0302' in answer.text and '日志' in answer.text
    assert 'mutation_snapshot' not in answer.text
    assert detail in answer.uncertainties
    assert len(answer.text) < 500


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-8-sig', 'utf-16', 'gb18030'])
def test_aspen_history_chinese_decoding(tmp_path, encoding):
    text = 'SEVERE ERROR\n确认：塔底参数缺少\n'
    assert _decode_history_text(text.encode(encoding)) == text
    (tmp_path/'model.his').write_bytes(text.encode(encoding))
    diagnostics = _read_history_diagnostics(tmp_path/'model.bkp')
    assert any('塔底参数缺少' in item for item in diagnostics)


def test_python_sources_are_utf8_without_known_mojibake():
    markers = ['\ufffd', '纭\ue13f\ue17b', '濉斿簳', '鍙傛暟', '缂哄皯']
    for path in Path('src').rglob('*.py'):
        source = path.read_text(encoding='utf-8-sig')
        strings = [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        for text in [source, *strings]:
            assert not any(marker in text for marker in markers), path
            assert not any('\ue000' <= c <= '\uf8ff' for c in text), path


def test_cli_utf8_output_even_with_legacy_pythonioencoding():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        'import src.main as m; m._print_backtest_progress(dict(status="running", completed=0, total=9, elapsed_seconds=15))'],
        env={**os.environ, 'PYTHONIOENCODING': 'gbk'}, capture_output=True, check=True)
    assert '回测进度 0/9：运行中' in result.stdout.decode('utf-8')


def test_repair_trace_only_in_file_log(tmp_path, capsys, logging_state):
    import json
    from src.agents.lead.intent import IntentClassifier
    from test_stage2_intent import _cfg, _payload
    path = tmp_path/'intent.log'
    setup_logging('INFO', log_file=path, console_level='WARNING', file_level='DEBUG', user_console=True)
    responses = iter([json.dumps(_payload(analysis='invalid-analysis')), json.dumps(_payload())])
    result = IntentClassifier(llm_config=_cfg(), llm_call=lambda *a, **k: next(responses)).classify('查看项目')
    assert result.classification_source == 'llm_repaired'
    assert capsys.readouterr().err == ''
    detail = path.read_text(encoding='utf-8')
    for marker in ['invalid-analysis', 'analysis must be an object', 'repair_intent_json', 'llm_repaired']:
        assert marker in detail
