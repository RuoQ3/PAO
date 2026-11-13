"""Conversation choices reach the approved runtime identity; no Aspen/network."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import yaml

from agent_fakes import CFG, payload
from src.agents.actions import ActionGateway
from src.agents.actions.authorization import ExecutionBoundaryError, file_digest
from src.agents.actions.plans import build_optimization_plan, validate_optimization_plan
from src.agents.context import ProcessContextBuilder
from src.agents.errors import AgentInternalError
from src.agents.lead import ConversationSession, LeadAgent
from src.agents.lead.continuation import new_task, validate_task, merge_turn
from src.agents.orchestration.models import IntentSchema, SessionState
from src.optimization.requests import prepare_request, OptimizationRequestError
from src.optimization.targets import parse_purity_target

P = r"\Data\Blocks\T0301\Input\RR"
O = r"\Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN"


@pytest.mark.parametrize("value", ["0.9999以上", "99.99%以上", "不低于 99.99%", ">=0.9999",
    "≥ 0.9999", "至少0.9999", "达到99.99%", "纯度为99.99%", "９９．９９％", ".9999", .9999, "9.999e-1"])
def test_purity_lower_bounds(value):
    assert parse_purity_target(value) == .9999


@pytest.mark.parametrize("value", ["0.9999以下", "不超过99.99%", "<0.9999", "0.9到0.9999",
    "99.99", "NaN", "Infinity", True, [], "纯度为99.99%，预算200次", "约99.99%", "99.99%%", "-0.1"])
def test_purity_parser_never_extracts_or_reverses_ambiguous_targets(value):
    with pytest.raises(ValueError):
        parse_purity_target(value)


@pytest.fixture
def configuration(tmp_path):
    model = tmp_path / "fake.bkp"
    model.write_bytes(b"offline test only")
    raw = {"simulator": {"filepath": str(model)},
           "design_variables": [{"name": "RR", "aspen_path": P, "type": "continuous", "lower_bound": 1., "upper_bound": 3., "unit": "1"},
                                {"name": "stage", "aspen_path": P + "_STAGE", "type": "integer", "lower_bound": 10, "upper_bound": 20, "unit": "1"}],
           "output_paths": [O],
           "objectives": [{"name": "flow", "aspen_path": O, "minimize": False}, {"name": "duty", "aspen_path": O, "minimize": True}],
           "constraints": [{"name": "purity_min", "aspen_path": O, "operator": ">=", "threshold": .999}],
           "optimizer": {"type": "pareto_bayesian", "surrogate_model": "qEHVI", "n_initial_points": 2, "n_iterations": 2, "random_seed": 7},
           "trust_region": {"enabled": True}, "feasibility_filter": {"enabled": True},
           "execution": {"max_evaluations": 12, "max_seconds": 60, "max_failures": 3}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path, raw


def selection():
    return {"algorithm": "GA", "equipment": "T0301", "component": "ADN", "metric": "purity", "target": "99.99%",
            "optimizer_parameters": {"pop_size": 4, "generations": 3}}


def logged_optimization_response():
    """The optional null slots returned in the user's 18:43:30 run."""
    return {"intent": "optimization_request", "task_type": "optimization",
        "project_scope": "current_project", "context_required": ["process_context", "yaml", "simulation_db"],
        "subagents": [
            {"name": "optimization_advisor", "task": "Review genetic algorithm optimization setup.", "priority": "high", "parallel": True},
            {"name": "process_advisor", "task": "Review T0301 process context.", "priority": "normal", "parallel": True},
            {"name": "risk_reviewer", "task": "Review operational limits.", "priority": "low", "parallel": True}],
        "action_type": "optimize", "requires_confirmation": True, "confidence": .95,
        "missing_information": ["target_stream"], "reason": "Genetic algorithm optimization of T0301 ADN purity.",
        "analysis": {"algorithm": "GA", "equipment": "T0301", "component": "ADN", "target_stream": None,
            "metric": "purity", "target": "99.99%", "execution": None, "optimizer_parameters": None, "random_seed": None}}


@pytest.mark.parametrize("slots", [(), ("execution",), ("optimizer_parameters",), ("random_seed",),
                                     ("execution", "optimizer_parameters", "random_seed")])
def test_optional_null_slots_normalize_without_repair_and_preserve_case(configuration, slots):
    from src.agents.lead.intent import IntentClassifier
    path, raw = configuration
    raw["execution"]["max_evaluations"] = 80
    response = logged_optimization_response()
    for key in ("execution", "optimizer_parameters", "random_seed"):
        if key not in slots:
            del response["analysis"][key]
    original = deepcopy(response)
    calls = []
    def provider(cfg, *, system, user):
        calls.append(user)
        return json.dumps(response)
    result = IntentClassifier(llm_config=CFG, llm_call=provider).classify("遗传算法优化 T0301，ADN 纯度 99.99%")
    assert len(calls) == 1 and result.retry_count == 0 and result.classification_source == "llm"
    assert response == original
    for key in ("execution", "optimizer_parameters", "random_seed"):
        assert key not in result.intent.analysis
    restored = IntentSchema.from_dict(original)
    assert restored.analysis == result.intent.analysis
    assert IntentSchema.from_json(restored.to_json()).analysis == result.intent.analysis
    # Also cover callers passing the raw response straight to plan preparation.
    request = prepare_request(original["analysis"], {"configuration": raw})
    assert request["algorithm"] == "GA"
    assert request["overrides"]["optimizer"]["random_seed"] == 7
    assert request["overrides"]["execution"] == raw["execution"]
    assert request["overrides"]["constraints"][0]["threshold"] == .9999


@pytest.mark.parametrize("field,bad", [("execution", []), ("execution", ""), ("execution", False),
                                      ("optimizer_parameters", []), ("optimizer_parameters", ""),
                                      ("optimizer_parameters", False), ("random_seed", "7"), ("random_seed", True)])
def test_optional_null_compatibility_still_rejects_invalid_types(field, bad):
    from src.agents.lead.intent_response_adapter import IntentResponseAdapter
    response = logged_optimization_response()
    response["analysis"][field] = bad
    with pytest.raises(TypeError):
        IntentResponseAdapter().parse(json.dumps(response))
    with pytest.raises(TypeError):
        IntentSchema.from_dict(response)


def test_logged_null_response_without_budget_continues_clarification(configuration):
    path, raw = configuration
    raw.pop("execution")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    calls = []
    def provider(cfg, *, system, user):
        calls.append(system)
        assert "Classify the PAO user message" in system
        return json.dumps(logged_optimization_response())
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True)))
    answer = chat.submit("在这个项目里，我想使用遗传算法对T0301塔进行优化，以达到ADN的纯度有99.99%。")
    assert len(calls) == 1 and chat.state.classification_retry_count == 0
    assert "max_evaluations" in answer.text
    assert chat.state.pending_task.phase == "clarifying"
    assert chat.state.pending_task.parameters["algorithm"] == "GA"
    assert not chat.state.internal_error and not chat.state.agent_unavailable


def test_logged_null_response_can_reach_ga_preview(configuration):
    pytest.importorskip("pymoo")
    path, raw = configuration
    raw["execution"]["max_evaluations"] = 80
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    response = logged_optimization_response()
    request = prepare_request(response["analysis"], {"configuration": raw})
    _, schema = build_optimization_plan(path, overrides=request["overrides"])
    gateway = ActionGateway(schema)
    classifications = []
    def provider(cfg, *, system, user):
        if "Classify the PAO user message" in system:
            classifications.append(user)
            return json.dumps(response)
        evidence = json.loads(user)["evidence"]
        if not evidence.get("tool_results"):
            return json.dumps({"tool_calls": [{"name": "plan_optimization", "arguments": {}, "required": True}]})
        return json.dumps({"text": "GA 纯度优化计划待审核。", "confidence": .9, "uncertainties": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True), action_gateway=gateway))
    answer = chat.submit("在这个项目里，我想使用遗传算法对T0301塔进行优化，以达到ADN的纯度有99.99%。")
    assert answer.approval_request and len(classifications) == 1
    assert chat.state.classification_retry_count == 0 and not chat.state.internal_error
    plan = answer.approval_request.proposal.payload["optimization_plan"]
    assert plan["algorithm"]["actual_algorithm"] == "GA"
    assert plan["configuration"]["optimizer"]["random_seed"] == 7
    assert plan["configuration"]["constraints"][0]["threshold"] == .9999


def test_request_overrides_yaml_and_binds_target_budget_and_algorithm(configuration):
    path, raw = configuration
    before = file_digest(path)
    request = prepare_request(selection(), {"configuration": raw})
    plan, schema = build_optimization_plan(path, overrides=request["overrides"])
    assert plan["algorithm"]["algorithm"] == plan["algorithm"]["actual_algorithm"] == "GA"
    assert plan["configuration"]["constraints"][0]["threshold"] == .9999
    assert plan["configuration"]["objectives"][0]["aspen_path"] == O
    assert plan["configuration"]["objectives"][0]["minimize"] is False
    assert len(plan["configuration"]["objectives"]) == 1
    assert plan["configuration"]["trust_region"] == {"enabled": False}
    assert plan["algorithm"]["parameters"]["pop_size"] == 4
    assert plan["limits"]["max_evaluations"] == 12
    assert validate_optimization_plan(plan, schema)
    assert file_digest(path) == before and raw["constraints"][0]["threshold"] == .999
    changed = deepcopy(plan)
    changed["configuration_overrides"]["constraints"][0]["threshold"] = .95
    with pytest.raises(ExecutionBoundaryError, match="changed"):
        validate_optimization_plan(changed, schema)


def test_invalid_unsupported_and_ambiguous_selections_are_explained(configuration):
    _, raw = configuration
    for update, message in [({"algorithm": "GWO"}, "尚未接入"), ({"target": "99.99"}, "百分数"),
                            ({"algorithm": "PSO"}, "不兼容"), ({"component": "OTHER"}, "无法唯一匹配")]:
        with pytest.raises(OptimizationRequestError, match=message):
            prepare_request({**selection(), **update}, {"configuration": raw})
    ambiguous = deepcopy(raw)
    ambiguous["constraints"].append({**raw["constraints"][0], "name": "another"})
    with pytest.raises(OptimizationRequestError, match="无法唯一匹配"):
        prepare_request(selection(), {"configuration": ambiguous})


def test_full_conversation_preserves_algorithm_and_only_previews(configuration):
    pytest.importorskip("pymoo")
    path, raw = configuration
    # Gateway schema is independent of the selected algorithm.
    request = prepare_request(selection(), {"configuration": raw})
    _, schema = build_optimization_plan(path, overrides=request["overrides"])
    gateway = ActionGateway(schema)
    calls = []
    def provider(cfg, *, system, user):
        calls.append((system, json.loads(user)))
        if "Classify the PAO user message" in system:
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = selection()
            intent["missing_information"] = ["decision variable bounds", "process constraints", "optimization budget"]
            return json.dumps(intent)
        evidence = json.loads(user)["evidence"]
        if not evidence.get("tool_results"):
            assert evidence["optimization_request"]["algorithm"] == "GA"
            return json.dumps({"tool_calls": [{"name": "plan_optimization", "arguments": {}, "required": True}]})
        return json.dumps({"text": "遗传算法 GA，ADN 纯度目标 99.99%，请审核。", "confidence": .9, "uncertainties": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True), action_gateway=gateway))
    answer = chat.submit("使用遗传算法优化 T0301，让 ADN 纯度达到 99.99%。")
    assert answer.approval_request
    plan = answer.approval_request.proposal.payload["optimization_plan"]
    assert plan["algorithm"]["algorithm"] == "GA"
    assert plan["configuration"]["constraints"][0]["threshold"] == .9999
    assert gateway.status(answer.approval_request.preview["audit_id"]).value == "pending"
    assert not any("task_fields" in system for system, _ in calls)


def test_missing_budget_resumes_with_same_goal_and_algorithm(configuration):
    path, raw = configuration
    raw.pop("execution")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    seen = []
    def provider(cfg, *, system, user):
        data = json.loads(user)
        seen.append(system)
        if "Classify the PAO user message" in system:
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = selection()
            intent["missing_information"] = ["optimization budget"]
            return json.dumps(intent)
        if "interpret_turn" in system:
            return json.dumps({"kind": "answer", "updates": {"max_evaluations": 12, "max_seconds": 60, "max_failures": 3}})
        return json.dumps({"text": "已保留 GA 与纯度目标，尚未执行。", "confidence": .9, "uncertainties": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True)))
    first = chat.submit("遗传算法优化 ADN 纯度到99.99%")
    task_id = chat.state.pending_task.task_id
    assert "max_evaluations" in first.text and not chat.state.internal_error
    # The same task survives JSON serialization and uses structured replies.
    chat.state = SessionState.from_json(chat.state.to_json())
    second = chat.submit("12次，60秒，最多失败3次")
    assert chat.state.pending_task.task_id == task_id
    assert chat.state.pending_task.parameters["algorithm"] == "GA"
    assert not chat.state.internal_error and not chat.state.agent_unavailable
    assert "task_fields" not in " ".join(seen)


@pytest.mark.parametrize("bad", [
    {"fields": {"optimization_budget": {}}, "parameters": {}},
    {"fields": {"optimization budget": {"type": "number", "multiple": False, "required": True, "candidate_refs": []}}, "parameters": {"equipment": "T0301"}},
])
def test_generic_field_protocol_repairs_inconsistent_keys_and_audits(bad):
    intent = payload("analysis")
    intent["missing_information"] = ["optimization budget"]
    good = {"fields": {"optimization budget": {"type": "number", "multiple": False, "required": True, "candidate_refs": []}}, "parameters": {"optimization budget": None}}
    responses = iter([bad, good])
    requests = []
    def provider(cfg, *, system, user):
        requests.append(json.loads(user))
        return json.dumps(next(responses))
    audit = []
    task = new_task(SimpleNamespace(llm_config=CFG, llm_call=provider), "分析", IntentSchema.from_dict(intent), {}, audit=audit)
    assert task.required_fields == ["optimization budget"]
    assert audit[0]["code"] == "task_fields_incomplete"
    assert "validation_error" in requests[1]


def test_generic_field_protocol_stops_after_one_failed_repair():
    intent = payload("analysis")
    intent["missing_information"] = ["budget"]
    calls = []
    def provider(*args, **kw):
        calls.append(kw)
        return '{"fields": {}, "parameters": {}}'
    audit = []
    with pytest.raises(AgentInternalError) as error:
        new_task(SimpleNamespace(llm_config=CFG, llm_call=provider), "分析", IntentSchema.from_dict(intent), {}, audit=audit)
    assert error.value.code == "task_fields_incomplete" and error.value.retry_count == 1
    assert len(calls) == len(audit) == 2 and audit[-1]["missing_fields"] == ["budget"]


def test_real_ga_mixed_integer_constraints():
    pytest.importorskip("pymoo")
    from src.optimization.problem import OptimizationProblem, Variable, Objective, Constraint, AlgorithmSpec, Observation
    from src.optimization.registry import create_optimizer
    from src.optimization.population import resolve_parameters
    from src.optimization.execution import run_optimizer
    p = OptimizationProblem((Variable("rr", P, 1., 3.), Variable("stage", P + "_STAGE", 10, 20, "integer")),
        (Objective("purity", False),), (Constraint("minimum", {"operator": ">="}, 1.),), max_evaluations=24)
    opt = create_optimizer(p, AlgorithmSpec("GA", "pymoo", resolve_parameters("GA", {"pop_size": 8, "generations": 3}), 12))
    calls = []
    def evaluate(candidate):
        calls.append(candidate)
        x, stage = candidate.x
        purity = .99995 - .0001 * (x - 2)**2 - .000001 * (stage - 15)**2
        residual = .9999 - purity
        return Observation(candidate.id, "feasible" if residual <= 0 else "infeasible", {"purity": purity}, {"minimum": residual})
    result = run_optimizer(opt, evaluate, remaining=lambda: 24-len(calls))
    assert len(calls) == 24 and opt.generation == 3
    assert all(float(c.x[1]).is_integer() and 10 <= c.x[1] <= 20 and 1 <= c.x[0] <= 3 for c in calls)
    assert result.algorithm_identity["actual_algorithm"] == "GA"
    best = result.to_summary()["best_feasible"]
    assert best is not None and best["observation"]["residuals"]["minimum"] <= 0


def test_cli_budget_reply_uses_overlay_for_lazy_gateway(configuration, tmp_path, monkeypatch, capsys):
    pytest.importorskip("pymoo")
    from src import main
    from src.agents.lead import llm_client
    import logging
    path, raw = configuration
    raw.pop("execution")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    def provider(cfg, *, system, user):
        if "Classify the PAO user message" in system:
            if json.loads(user)["question"] == "Hello":
                return json.dumps(payload("chat"))
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = selection()
            return json.dumps(intent)
        if "interpret_turn" in system:
            return json.dumps({"kind": "answer", "updates": {"max_evaluations": 12, "max_seconds": 60, "max_failures": 3}})
        evidence = json.loads(user)["evidence"]
        if not evidence.get("tool_results"):
            return json.dumps({"tool_calls": [{"name": "plan_optimization", "arguments": {}, "required": True}]})
        return json.dumps({"text": "计划已生成，尚未执行。", "confidence": .9, "uncertainties": []})
    monkeypatch.setattr(llm_client, "load_llm_config", lambda: CFG)
    monkeypatch.setattr(llm_client, "chat_json", provider)
    turns = iter(["12次，60秒，最多失败3次", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(turns))
    assert main._run_unified_chat_loop(config_path=path, yaml_path=path, db_path=tmp_path / "sim.db",
        node_db_path=None, initial_intent="遗传算法优化T0301，ADN纯度99.99%", log=logging.getLogger("test-selection")) == 0
    audits = list(tmp_path.glob("conversation-*.jsonl"))
    records = [json.loads(line) for line in audits[0].read_text(encoding="utf-8").splitlines()]
    answers = [row["answer"] for row in records if row["event"] == "answer_created"]
    approval = answers[-1]["approval_request"]
    assert approval["preview"]["valid"] is True
    plan = approval["proposal"]["payload"]["optimization_plan"]
    assert plan["algorithm"]["algorithm"] == "GA" and plan["limits"]["max_evaluations"] == 12
    assert "execution" not in yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "GA" in capsys.readouterr().out


@pytest.mark.parametrize("purity", [.99995, .99985])
def test_approved_ga_overlay_reaches_runtime_and_evaluation_ledger(configuration, tmp_path, monkeypatch, purity):
    pytest.importorskip("pymoo")
    import sys
    from types import ModuleType
    from src.agents.actions import ActionRequest
    from src.agents.actions.authorization import evaluated_case, check_run
    from src.database.simulation_db import SimulationDB
    from src.models.process_case import ProcessCase, CaseStatus, ObjectiveValue, ConstraintValue
    path, raw = configuration
    request = prepare_request(selection(), {"configuration": raw})
    plan, schema = build_optimization_plan(path, overrides=request["overrides"])
    seen = []
    class Driver:
        simulated = True
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def open(self, copied):
            assert copied != tmp_path / "fake.bkp"
        def disconnect(self): pass
    module = ModuleType("src.aspen_driver.driver")
    module.AspenDriver = Driver
    monkeypatch.setitem(sys.modules, "src.aspen_driver.driver", module)
    def evaluate(driver, inputs, config, **kwargs):
        check_run()
        seen.append(inputs)
        return ProcessCase(status=CaseStatus.SUCCESS if purity >= .9999 else CaseStatus.INFEASIBLE, design_vars=inputs,
            objectives=[ObjectiveValue("requested_purity", purity, "1", False)],
            constraints=[ConstraintValue("purity_min", .9999-purity)])
    monkeypatch.setattr("src.workflows.run_case.run_case", evaluated_case(evaluate))
    with SimulationDB(tmp_path / "execution.db") as db:
        gateway = ActionGateway(schema, simulation_db=db)
        action = ActionRequest("run_optimization", "model", reason="offline overlay integration test", payload={"optimization_plan": plan})
        valid = gateway.submit(action)
        assert valid.valid, valid.errors
        gateway.preview(action.audit_id)
        gateway.approve(action.audit_id)
        output = gateway.execute(action.audit_id)
        assert output["status"] == "simulated" and not output["real_execution"]
        assert output["algorithm"]["actual_algorithm"] == "GA"
        assert len(seen) == output["evaluations_used"] == 12
        assert all(float(row[P + "_STAGE"]).is_integer() for row in seen)
        assert len(db.get_execution_evaluations(action.audit_id)) == 12
        if purity >= .9999:
            assert output["summary"]["best_feasible"]["observation"]["objectives"]["requested_purity"] >= .9999
        else:
            assert output["summary"]["best_feasible"] is None
            assert output["summary"]["status_counts"]["infeasible"] == 12
            assert output["summary"]["stop_reason"] == "generations_completed"


@pytest.mark.parametrize("bad", ["garbled", '{"fields":[],"parameters":{}}',
                                    '{"fields":{"budget":{"type":"number"}},"parameters":{}}'])
def test_generic_protocol_repairs_json_and_spec_errors(bad):
    intent = payload("analysis")
    intent["missing_information"] = ["budget"]
    good = {"fields": {"budget": {"type": "number", "multiple": False, "required": True, "candidate_refs": []}}, "parameters": {"budget": None}}
    responses = iter([bad, json.dumps(good)])
    task = new_task(SimpleNamespace(llm_config=CFG, llm_call=lambda *a, **kw: next(responses)), "分析", IntentSchema.from_dict(intent), {})
    assert task.required_fields == ["budget"]


def unsupported_metric_response():
    # Same invalid evidence shape as the run: source scalar + model-derived
    # comparison presented as a verified six-field metric tuple.
    return {"text": "需要核对纯度基准，尚未执行。", "confidence": .9, "uncertainties": [],
        "blocks": [{"kind": "metrics", "evidence_refs": ["/constraints/0/threshold"],
            "metrics": [{"label": "purity", "value": .993085932, "unit": "mole fraction",
                         "baseline": "mass purity 0.9999", "delta": -.006814068, "scope": "static snapshot"}]}]}


def test_budget_correction_transcript_repairs_advice_and_preserves_task(configuration):
    pytest.importorskip("pymoo")
    path, raw = configuration
    raw.pop("execution")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    selected = {"algorithm": "GA", "equipment": "T0301", "component": "ADN", "metric": "purity", "target": "0.9999",
                "execution": {"max_evaluations": 200, "max_seconds": 600, "max_failures": 10}}
    request = prepare_request(selected, {"configuration": raw})
    _, schema = build_optimization_plan(path, overrides=request["overrides"])
    gateway = ActionGateway(schema)
    decisions = iter([("answer", {"max_evaluations": 10}), ("answer", {"max_seconds": 600}),
                      ("answer", {"max_failures": 10}), ("correction", {"max_evaluations": 200}),
                      ("correction", {"max_evaluations": 100})])
    calls, repairs = [], []
    def provider(cfg, *, system, user):
        data = json.loads(user)
        calls.append(system)
        if "Classify the PAO user message" in system:
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = {k: v for k, v in selected.items() if k != "execution"}
            assert "stream" not in data["conversation_focus"]  # .9999 is not stream 9999
            return json.dumps(intent)
        if "interpret_turn" in system:
            kind, updates = next(decisions)
            return json.dumps({"kind": kind, "updates": updates})
        evidence = data["evidence"]
        schedule = evidence["optimization_request"]["evaluation_schedule"]
        assert schedule["planned_evaluations"] == 200 and schedule["generations_include_initial_population"]
        assert evidence["configuration"]["constraints"][0]["threshold"] == .9999
        assert evidence["source_configuration"]["constraints"][0]["threshold"] == .999
        if "invalid_response" in data:
            repairs.append(data)
            assert "metric_tuple_not_in_evidence" in data["validation_error"]["reason"]
            return json.dumps({"text": "纯度基准需要核对，可达性尚无仿真证据。", "confidence": .9,
                "uncertainties": ["尚未执行 Aspen"], "blocks": [{"kind": "limitation", "text": "可达性未知。",
                    "evidence_refs": ["/constraints/0/aspen_path"]}]})
        if evidence["optimization_request"]["overrides"]["execution"]["max_evaluations"] == 200:
            return json.dumps(unsupported_metric_response())
        if not evidence.get("tool_results"):
            return json.dumps({"tool_calls": [{"name": "plan_optimization", "arguments": {}, "required": True}]})
        return json.dumps({"text": "已生成待审核 GA 计划，未执行。", "confidence": .9, "uncertainties": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True), action_gateway=gateway))
    first = chat.submit("使用遗传算法优化T0301，使ADN纯度达到0.9999以上。")
    task_id = chat.state.pending_task.task_id
    assert "max_evaluations" in first.text
    assert "max_seconds" in chat.submit("10次").text
    assert "max_failures" in chat.submit("600秒").text
    assert "不足一代" in chat.submit("10次").text
    corrected = chat.submit("那就总评估预算为200次")
    assert "基准" in corrected.text and len(repairs) == 1
    assert not corrected.approval_request and not chat.state.internal_error
    assert chat.state.pending_task.parameters["max_evaluations"] == 200
    assert chat.state.pending_task.task_id == task_id
    assert any(e["event"] == "agent_output_validation_failed" and e["reason"] == "metric_tuple_not_in_evidence"
               for e in chat.state.audit_events)
    chat.state = SessionState.from_json(chat.state.to_json())
    answer = chat.submit("改为100次")
    assert answer.approval_request
    plan = answer.approval_request.proposal.payload["optimization_plan"]
    assert plan["limits"] == {"max_evaluations": 100, "max_seconds": 600, "max_failures": 10}
    assert plan["algorithm"]["actual_algorithm"] == "GA"
    assert chat.state.pending_task.task_id == task_id


def test_repeated_invalid_advice_keeps_task_and_allows_retry(configuration):
    pytest.importorskip("pymoo")
    path, raw = configuration
    request = prepare_request(selection(), {"configuration": raw})
    _, schema = build_optimization_plan(path, overrides=request["overrides"])
    gateway = ActionGateway(schema)
    invalid_calls = []
    def provider(cfg, *, system, user):
        if "Classify the PAO user message" in system:
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = selection()
            return json.dumps(intent)
        if "interpret_turn" in system:
            return json.dumps({"kind": "answer", "updates": {}})
        data = json.loads(user)
        if len(invalid_calls) < 2:
            invalid_calls.append(data)
            return json.dumps(unsupported_metric_response())
        if not data["evidence"].get("tool_results"):
            return json.dumps({"tool_calls": [{"name": "plan_optimization", "arguments": {}, "required": True}]})
        return json.dumps({"text": "GA 计划待审核。", "confidence": .9, "uncertainties": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True), action_gateway=gateway))
    rejected = chat.submit("使用遗传算法优化T0301纯度至0.9999")
    assert "修复后仍未通过" in rejected.text and len(invalid_calls) == 2
    assert chat.state.pending_task.phase == "clarifying"
    assert not chat.state.internal_error and not chat.state.agent_unavailable
    assert not chat.state.pending_approval and not chat.state.task_results
    failure = next(e for e in chat.state.audit_events if e["event"] == "optimization_explanation_rejected")
    assert failure["retry_count"] == 1
    answer = chat.submit("请重新生成计划")
    assert answer.approval_request and not chat.state.pending_task.planning_blockers
    assert chat.state.pending_task.phase == "awaiting_approval"
    previous_id = answer.approval_request.preview["audit_id"]
    chat.state = SessionState.from_json(chat.state.to_json())
    assert chat.state.pending_approval is None
    restored_preview = chat.submit("确认")
    assert "恢复的审批已失效" in restored_preview.text
    assert restored_preview.approval_request.preview["audit_id"] != previous_id
    assert not any(e.get("event") == "action_executed" for e in chat.state.audit_events)
    paused = chat.submit("先不执行，我要修改预算")
    assert "旧审批已撤销" in paused.text
    assert chat.state.pending_approval is None


def test_repair_cannot_execute_tools_or_submit_invalid_plan(configuration):
    pytest.importorskip("pymoo")
    path, raw = configuration
    def provider(cfg, *, system, user):
        if "Classify the PAO user message" in system:
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = selection()
            return json.dumps(intent)
        if "invalid_response" in json.loads(user):
            return json.dumps({"tool_calls": [{"name": "plan_optimization", "arguments": {}, "required": True}]})
        return json.dumps(unsupported_metric_response())
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True)))
    answer = chat.submit("遗传算法优化T0301纯度")
    assert "未生成审批" in answer.text
    assert not any(e["event"] == "tool_call" for e in chat.state.audit_events)
    assert not chat.state.pending_approval


def test_logged_lower_bound_transcript_preserves_generations_and_accepts_percent_restatement(configuration):
    pytest.importorskip("pymoo")
    path, raw = configuration
    raw.pop("execution")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    selected = {"algorithm": "GA", "equipment": "T0301", "component": "ADN", "metric": "purity",
                "target": "0.9999以上"}
    expected = {**selected, "optimizer_parameters": {"generations": 200},
                "execution": {"max_evaluations": 200, "max_seconds": 600, "max_failures": 50}}
    request = prepare_request(expected, {"configuration": raw})
    _, schema = build_optimization_plan(path, overrides=request["overrides"])
    decisions = iter([{"generations": 200}, {"max_evaluations": 200}, {"max_seconds": 600},
                      {"max_failures": 50}, {"target": "99.99%"}, {"target": "99.99%"}])
    seen = []
    def provider(cfg, *, system, user):
        data = json.loads(user)
        if "Classify the PAO user message" in system:
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = selected
            return json.dumps(intent)
        if "interpret_turn" in system:
            return json.dumps({"kind": "answer", "updates": next(decisions)})
        evidence = data["evidence"]
        seen.append(evidence["optimization_request"])
        if not evidence.get("tool_results"):
            return json.dumps({"tool_calls": [{"name": "plan_optimization", "arguments": {}, "required": True}]})
        return json.dumps({"text": "GA 优化计划待审核。", "confidence": .9, "uncertainties": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True), action_gateway=ActionGateway(schema)))
    original = path.read_bytes()
    assert "max_evaluations" in chat.submit("我想使用遗传算法对我的T0301进行优化，能够使得最终产品ADN的纯度达到0.9999以上。").text
    task_id = chat.state.pending_task.task_id
    generations = chat.submit("200代")
    assert "已记录 200 代" in generations.text and "max_evaluations" in generations.text
    assert chat.state.pending_task.parameters["max_evaluations"] is None
    assert "max_seconds" in chat.submit("200次").text
    assert "max_failures" in chat.submit("600秒").text
    for message in ("50次", "纯度为99.99%", "纯度为99.99%"):
        answer = chat.submit(message)
        assert answer.approval_request, answer.text
        assert "计划评估 4000 次" in answer.text and "最多支持 10 个完整种群" in answer.text
        plan = answer.approval_request.proposal.payload["optimization_plan"]
        assert plan["limits"] == expected["execution"]
        assert plan["algorithm"]["actual_algorithm"] == "GA"
        assert plan["algorithm"]["parameters"]["generations"] == 200
        assert chat.state.pending_task.task_id == task_id
        assert not chat.state.internal_error
        chat.state = SessionState.from_json(chat.state.to_json())
    assert all(r["overrides"]["constraints"][0]["threshold"] == .9999 for r in seen)
    assert all(r["evaluation_schedule"]["planned_evaluations"] == 4000 for r in seen)
    assert path.read_bytes() == original


def test_invalid_initial_target_is_repairable_as_answer_after_checkpoint(configuration):
    path, raw = configuration
    selected = {**selection(), "target": "0.9999以下"}
    intent = payload("optimization_request", role="optimization_advisor", action="optimize")
    intent["analysis"] = selected
    observed = []
    def provider(cfg, *, system, user):
        if "Classify the PAO user message" in system:
            return json.dumps(intent)
        data = json.loads(user)
        if "interpret_turn" in system:
            observed.append(data["task"])
            return json.dumps({"kind": "answer", "updates": {"target": "99.99%"}})
        observed.append(data["evidence"]["optimization_request"])
        return json.dumps({"text": "目标已修复，尚未执行。", "confidence": .9, "uncertainties": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True)))
    first = chat.submit("遗传算法优化T0301，ADN纯度0.9999以下")
    assert "当前支持纯度下限" in first.text
    assert chat.state.pending_task.waiting_fields == ["target"]
    chat.state = SessionState.from_json(chat.state.to_json())
    answer = chat.submit("纯度为99.99%")
    assert "目标已修复" in answer.text and not chat.state.internal_error
    assert observed[0]["waiting_fields"] == ["target"]
    assert observed[0]["planning_blockers"][0]["field"] == "target"
    assert observed[1]["overrides"]["constraints"][0]["threshold"] == .9999
    assert chat.state.pending_task.parameters["algorithm"] == "GA"
    assert chat.state.pending_task.parameters["max_evaluations"] == 12


def test_answer_cannot_silently_change_valid_target_or_algorithm(configuration):
    _, raw = configuration
    intent = payload("optimization_request", role="optimization_advisor", action="optimize")
    intent["analysis"] = selection()
    task = new_task(None, "GA纯度99.99%", IntentSchema.from_dict(intent), {"configuration": raw})
    assert validate_task(task, {"configuration": raw})
    assert not merge_turn(task, {"kind": "answer", "updates": {"target": "99%"}}, "纯度99%")
    assert not merge_turn(task, {"kind": "answer", "updates": {"algorithm": "BO"}}, "BO")
    assert merge_turn(task, {"kind": "correction", "updates": {"target": "99%"}}, "改为99%")


def test_live_provider_population_alias_normalizes_before_task_and_planning(configuration):
    _, raw = configuration
    intent = logged_optimization_response()
    intent["analysis"]["optimizer_parameters"] = {"population_size": 4, "generations": 2,
                                                  "include_initial_generation": True}
    parsed = IntentSchema.from_dict(intent)
    assert parsed.analysis["optimizer_parameters"] == {"pop_size": 4, "generations": 2}
    task = new_task(None, "每代4个，2代", parsed, {"configuration": raw})
    assert task.parameters["pop_size"] == 4
    request = prepare_request(parsed.analysis, {"configuration": raw})
    assert request["evaluation_schedule"]["planned_evaluations"] == 8
    assert "population_size" in intent["analysis"]["optimizer_parameters"]  # no mutation
    intent["analysis"]["optimizer_parameters"]["pop_size"] = 8
    with pytest.raises(ValueError, match="conflicting"):
        IntentSchema.from_dict(intent)
    intent["analysis"]["optimizer_parameters"] = {"generations": 2, "include_initial_generation": False}
    with pytest.raises(ValueError, match="include the initial"):
        IntentSchema.from_dict(intent)


def test_algorithm_name_does_not_become_component_focus():
    chat = ConversationSession(SessionState(), None)
    chat._update_focus_from_turn("使用GA优化T0301，使ADN质量分数达到0.9999")
    assert chat.state.runtime_state["conversation_focus"]["component"] == "ADN"


def test_live_chinese_metric_and_stream_label_preserve_mass_basis(configuration):
    _, raw = configuration
    intent = logged_optimization_response()
    intent["analysis"].update(metric="质量分数", target_stream="产品物流ADN",
                             optimizer_parameters={"pop_size": 4, "generations": 2})
    parsed = IntentSchema.from_dict(intent)
    assert parsed.analysis["metric"] == "mass_fraction"
    assert parsed.analysis["target_stream"] == "ADN"
    request = prepare_request(parsed.analysis, {"configuration": raw})
    assert request["overrides"]["objectives"][0]["aspen_path"] == O


def test_explicit_empty_extraction_lists_skip_optional_tree_walks(configuration):
    from src.utils.file_io import _build_run_config
    _, raw = configuration
    raw["extraction"] = {"blocks": [], "streams": [], "check_status_paths": [r"\Data\Blocks\T0301"]}
    config = _build_run_config(raw)
    assert config.extract_blocks == [] and config.extract_streams == []
    assert config.check_status_paths == [r"\Data\Blocks\T0301"]
    raw["extraction"] = {}
    config = _build_run_config(raw)
    assert config.extract_blocks is None and config.extract_streams is None


def test_invalid_citation_repair_receives_exact_rejected_pointer(configuration):
    path, _ = configuration
    repairs = []
    def provider(cfg, *, system, user):
        data = json.loads(user)
        if "Classify the PAO user message" in system:
            intent = payload("optimization_request", role="optimization_advisor", action="optimize")
            intent["analysis"] = selection()
            return json.dumps(intent)
        if "invalid_response" in data:
            repairs.append(data["validation_error"])
            return json.dumps({"text": "配置已核对，尚未执行。", "confidence": .9, "uncertainties": [],
                               "blocks": [{"kind": "paragraph", "text": "有效配置。", "evidence_refs": ["/configuration"]}]})
        return json.dumps({"text": "配置已核对。", "confidence": .9, "uncertainties": [],
                           "blocks": [{"kind": "paragraph", "text": "GA配置。", "evidence_refs": ["/configuration/missing_path"]}]})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True)))
    answer = chat.submit("GA优化T0301纯度99.99%")
    assert "已核对" in answer.text
    assert repairs[0]["invalid_reference"] == "/configuration/missing_path"


def test_execution_analysis_repair_never_repeats_execution():
    from src.agents.reasoning import reason
    calls = []
    evidence = {"result": {"summary": {"stop_reason": "generations_completed"}, "resources": {},
                           "execution_evidence": {"evidence_digest": "saved"}}}
    def provider(cfg, *, system, user):
        data = json.loads(user)
        calls.append(data)
        if "invalid_response" not in data:
            return json.dumps({"text": "执行已完成。", "uncertainties": [], "confidence": .9,
                               "blocks": [{"kind": "paragraph", "text": "已完成。",
                                           "evidence_refs": ["/result/not_a_real_key"]}]})
        assert data["validation_error"]["invalid_references"] == ["/result/not_a_real_key"]
        return json.dumps({"text": "执行已完成，未重复运行。", "uncertainties": [], "confidence": .9,
                           "blocks": [{"kind": "paragraph", "text": "已完成。",
                                       "evidence_refs": ["/result/summary"]}]})
    answer, _ = reason("解释已保存结果", context=evidence, role="action_result_analysis", llm_config=CFG, llm_call=provider)
    assert len(calls) == 2 and "未重复运行" in answer["text"]
