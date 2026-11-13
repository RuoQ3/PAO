"""Third-batch external behavior; conftest blocks all network and Aspen COM."""
import json
import math
import threading
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from agent_fakes import CFG, payload
from src.agents.actions import ActionGateway
from src.agents.actions.plans import build_optimization_plan
from src.agents.context import ProcessContextBuilder
from src.agents.context.versions import file_version
from src.agents.errors import AgentUnavailableError
from src.agents.lead import ConversationSession, LeadAgent
from src.agents.orchestration.models import SessionState, TaskSpec
from src.agents.reasoning import reason
from src.agents.subagents.onboarding_agent.project import ProjectOnboarding
from src.agents.subagents.registry import default_registry
from src.agents.subagents.tool_runtime import ToolDefinition, ToolRegistry, ToolSession
from src.database.node_db import NodeDB
from src.models.node_catalog import CatalogEntry, CatalogScan
from src.reporting.answer_terminal import TerminalRenderer

P = r"\Data\Blocks\R1\Input\TEMP"
O = r"\Data\Streams\S1\Output\MASSFRAC\MIXED\A"


def final(text="有证据的测试回答"):
    return json.dumps({"text": text, "uncertainties": [], "confidence": .9})


def print_example(text):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding))


def calls(name, arguments=None, required=True):
    return json.dumps({"tool_calls": [{"name": name, "arguments": arguments or {}, "required": required}]})


def tool_task(role="process_advisor", names=("measure",), **kw):
    return TaskSpec(agent=role, tool_allowlist=list(names), budget=3, **kw)


def test_actual_dispatch_result_reaches_followup_reasoning_and_audit():
    executed, seen = [], []
    tools = ToolRegistry()
    def handler(context, count):
        executed.append(count)
        return {"observed": 37, "unit": "C", "source": "fake://selected"}
    tools.register(ToolDefinition("measure", "test read", handler, ("process_advisor",), {"count": int}, ("count",)))
    session = ToolSession(tools, tool_task(), {"session_id": "s"})
    def provider(*a, **kw):
        evidence = json.loads(kw["user"])["evidence"]
        seen.append(evidence)
        if len(seen) == 1:
            return calls("measure", {"count": 2})
        assert evidence["tool_results"][0]["value"]["observed"] == 37
        return final("静态证据记录温度 37 C。")
    answer, evidence = reason("测量", role="process_advisor", llm_config=CFG, llm_call=provider, tool_session=session)
    assert executed == [2] and "37" in answer["text"]
    assert evidence["tool_results"][0]["evidence_id"] == session.audit[0]["call_id"]
    assert session.audit[0]["status"] == "completed" and session.audit[0]["elapsed_seconds"] >= 0
    assert evidence["tool_results"][0]["handler_version"]["sha256"]


@pytest.mark.parametrize("role,name,args", [("process_advisor", "shell", {}),
    ("code_engineer", "measure", {}), ("process_advisor", "measure", {"bad": 2}),
    ("process_advisor", "measure", {"count": True})])
def test_reject_unregistered_role_and_invalid_parameters(role, name, args):
    executed = []
    tools = ToolRegistry()
    tools.register(ToolDefinition("measure", "x", lambda *a, **k: executed.append(1),
        ("process_advisor",), {"count": int}, ("count",)))
    session = ToolSession(tools, tool_task(role), {})
    with pytest.raises((PermissionError, ValueError)):
        session.invoke(name, args)
    assert not executed and session.audit[0]["status"] == "rejected"


def test_call_step_and_wall_time_budgets_discard_late_results():
    tools = ToolRegistry()
    executed = []
    tools.register(ToolDefinition("measure", "x", lambda c: executed.append(1) or 1, ("process_advisor",), max_calls=1))
    session = ToolSession(tools, tool_task(), {}, max_tool_calls=1)
    session.invoke("measure", {})
    with pytest.raises(PermissionError):
        session.invoke("measure", {})
    assert len(executed) == 1
    session = ToolSession(tools, tool_task(max_steps=1), {})
    with pytest.raises(AgentUnavailableError) as error:
        reason("x", role="process_advisor", llm_config=CFG,
               llm_call=lambda *a, **k: calls("measure"), tool_session=session)
    assert error.value.code == "tool_step_budget_exhausted"
    release = threading.Event()
    tools = ToolRegistry()
    tools.register(ToolDefinition("measure", "slow", lambda c: release.wait(1), ("process_advisor",), timeout_seconds=.01))
    session = ToolSession(tools, tool_task(timeout_seconds=.03), {})
    try:
        with pytest.raises(TimeoutError):
            session.invoke("measure", {})
        assert not session.results and session.audit[-1]["worker_termination"] == "unknown"
    finally:
        release.set()


def test_required_failure_stops_optional_failure_is_visible():
    for required in (True, False):
        tools = ToolRegistry()
        tools.register(ToolDefinition("measure", "missing", None, ("process_advisor",)))
        session = ToolSession(tools, tool_task(), {})
        seen = []
        def provider(*a, **kw):
            seen.append(json.loads(kw["user"]))
            return calls("measure", required=required) if len(seen) == 1 else final()
        if required:
            with pytest.raises(AgentUnavailableError) as error:
                reason("x", role="process_advisor", llm_config=CFG, llm_call=provider, tool_session=session)
            assert error.value.code == "required_tool_unavailable" and len(seen) == 1
        else:
            answer, evidence = reason("x", role="process_advisor", llm_config=CFG, llm_call=provider, tool_session=session)
            assert "measure" in answer["uncertainties"][-1]
            assert evidence["tool_results"][0]["status"] == "unavailable"


def test_source_update_and_attachment_update_reject_stale_results(tmp_path):
    attachment = tmp_path / "notes.txt"
    attachment.write_text("old", encoding="utf-8")
    version = file_version(attachment)
    context = {"attachments": [{"path": str(attachment), "version": version, "status": "available", "content": "old"}]}
    tools = default_registry().tools
    session = ToolSession(tools, tool_task("code_engineer", ("read_files",)), context)
    attachment.write_text("new", encoding="utf-8")
    with pytest.raises(LookupError):
        session.invoke("read_files", {"index": 0})
    assert not session.results


def test_capabilities_track_handler_sources_and_role_allowlists():
    registry = default_registry()
    empty = {x["id"]: x for x in registry.tools.catalogue({})}
    assert not empty["read_yaml"]["available"] and empty["execute_code"]["handler"] is None
    assert not empty["scan_aspen_nodes"]["available"] and not empty["query_knowledge"]["available"]
    ready = registry.tools.catalogue({"configuration": {"x": 1}}, role="boundary_advisor",
        allowlist=registry.definition("boundary_advisor").tool_allowlist)
    assert next(x for x in ready if x["id"] == "read_yaml")["available"]
    assert all("boundary_advisor" in x["roles"] for x in ready)
    assert not any(x["side_effect"] for x in registry.tools.catalogue({}, role="code_engineer"))


def values():
    return {"product": {"name": "A 产品", "stream": "S1", "component": "A"},
        "objectives": [{"name": "产品质量分数", "type": "aspen_path", "aspen_path": O,
            "minimize": False, "metric": "mass_fraction", "basis": "mass", "unit": "1", "stream": "S1", "component": "A"}],
        "constraints": [{"name": "质量要求", "aspen_path": O, "operator": ">=", "threshold": .9,
            "metric": "mass_fraction", "basis": "mass", "unit": "1", "stream": "S1", "component": "A"}],
        "design_variables": [{"name": "反应器温度", "aspen_path": P, "type": "continuous", "lower_bound": 20,
            "upper_bound": 80, "unit": "C", "initial_value": 40}],
        "optimizer": {"type": "bayesian", "surrogate_model": "random", "n_initial_points": 2,
            "n_iterations": 1, "acquisition_function": "EI"},
        "execution": {"max_evaluations": 8, "max_seconds": 120, "max_failures": 3, "timeout": 20}}


def update_response(message, updates=None, conflicts=None, resolved=None):
    return json.dumps({"updates": {k: {"value": v, "origin": "user", "quote": message, "evidence_ref": ""}
        for k, v in (updates or {}).items()}, "conflicts": conflicts or [], "resolved_conflicts": resolved or [],
        "question": "请核对当前目标、质量基准、单位、边界和总预算。"}, ensure_ascii=False)


@pytest.fixture
def project(tmp_path):
    model = tmp_path / "model.apw"
    model.write_bytes(b"fake binary Aspen model")
    node_path = tmp_path / "nodes.db"
    with NodeDB(node_path) as db:
        db.save_catalog_scan(CatalogScan("selected", str(model), file_version(model)["sha256"], "fake", 1, 1, 2, 6, "2026-11-11"))
        db.save_catalog_entries([CatalogEntry("selected", path, path.split("\\")[-1], path.rsplit("\\", 1)[0], 4,
            name, block, "MIXER" if block else "MATERIAL", stream, True, False, 2, unit, 0, 40, "", "2026-11-11")
            for path, name, block, stream, unit in [(P, "TEMP", "R1", "", "C"), (O, "MASSFRAC", "", "S1", "1")]])
    queue, seen = [], []
    def provider(*a, **kw):
        data = json.loads(kw["user"])
        seen.append(data)
        return queue.pop(0) if queue else update_response(data["message"])
    session = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider))
    session.submit('/project new ' + str(model))
    session.submit('/project node-db ' + str(node_path))
    return session, queue, seen, model, node_path


def populate(project):
    session, queue, *_ = project
    statement = "产品 A，S1 质量分数最大化，质量下限 0.9；温度 20 至 80 C；明确选择 random，DOE 2 加搜索 1，总评估 8 次，总时限 120 秒，失败 3 次，单次 20 秒。"
    queue.append(update_response(statement, values()))
    answer = session.submit(statement)
    assert not ProjectOnboarding(session).issues(ProjectOnboarding(session).draft)
    return answer


def test_no_yaml_project_correction_confirmation_save_and_roundtrip(project, tmp_path):
    session, queue, seen, model, _ = project
    answer = populate(project)
    rendered = TerminalRenderer().render(answer)
    assert "质量" in rendered and "80" in rendered and "120" in rendered
    assert any(x["abs_path"] == O for fact in seen[1]["evidence"]["facts"]
               if fact.get("kind") == "static_node_catalog" for x in fact["nodes"])
    correction = "质量下限改为 0.999，温度上限改为 70 C，总预算改为 9 次。"
    changed = values()
    changed["constraints"][0]["threshold"] = .999
    changed["design_variables"][0]["upper_bound"] = 70
    changed["execution"]["max_evaluations"] = 9
    queue.append(update_response(correction, {k: changed[k] for k in ("constraints", "design_variables", "execution")}))
    corrected_answer = session.submit(correction)
    session.submit('/project confirm')
    assert session.state.pending_approval is None
    destination = tmp_path / "generated.yaml"
    session.submit('/project save ' + str(destination))
    from src.utils.file_io import load_optimize_config
    parsed, selected_model, _ = load_optimize_config(destination)
    plan, schema = build_optimization_plan(destination)
    assert selected_model == model and parsed.param_bounds[P] == (20, 70)
    raw = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert raw["constraints"] == changed["constraints"] and raw["objectives"] == changed["objectives"]
    assert raw["execution"]["max_evaluations"] == 9 and schema.parameters[P].unit == "C"
    assert plan["algorithm"]["surrogate_model"] == "random"
    audit = json.loads(destination.with_name(destination.name + '.project.json').read_text(encoding="utf-8"))
    assert audit["draft"]["history"][-1]["event"] == "business_configuration_confirmed"
    assert not session.state.pending_approval and model.read_bytes() == b"fake binary Aspen model"
    original = destination.read_bytes()
    session.submit('/project save ' + str(destination))
    assert destination.read_bytes() == original
    session.submit('/project use')
    assert ProjectOnboarding(session).draft.status == "active"
    print("\n模拟示例：无 YAML 建立项目 → 自然语言纠正 → confirm → save → use")
    print_example(rendered)
    print_example(TerminalRenderer().render(corrected_answer))
    print(f"保存并实际解析通过：{destination}；未运行 Aspen。")


def test_conflicts_missing_basis_unsupported_algorithm_and_bounds_are_not_filled(project):
    session, queue, *_ = project
    with pytest.raises(ValueError):
        session.submit('/project confirm')
    populate(project)
    statement = "描述下限 0.9，但阈值写成 0.999，尚未决定。"
    queue.append(update_response(statement, conflicts=[{"id": "threshold", "message": statement}]))
    session.submit(statement)
    with pytest.raises(ValueError, match="冲突"):
        session.submit('/project confirm')
    fix = "确认质量下限为 0.9，解决这个冲突。"
    queue.append(update_response(fix, resolved=["threshold"]))
    session.submit(fix)
    service = ProjectOnboarding(session)
    draft = service.draft
    for field, child, value, match in [("optimizer", "surrogate_model", "PSO", "未实现"),
                                      ("execution", "max_evaluations", 1, "总预算")]:
        invalid = deepcopy(draft)
        invalid.fields[field]["value"][child] = value
        with pytest.raises(ValueError, match=match):
            service.validate(invalid)
    invalid = deepcopy(draft)
    invalid.fields["objectives"]["value"][0]["basis"] = "mole"
    with pytest.raises(ValueError, match="基准冲突"):
        service.validate(invalid)
    invalid = deepcopy(draft)
    invalid.fields["design_variables"]["value"][0].pop("unit")
    with pytest.raises(ValueError):
        service.validate(invalid)


def test_restore_switch_and_source_updates_never_reuse_authority(project, tmp_path):
    session, queue, _, model, _ = project
    populate(project)
    session.submit('/project confirm')
    checkpoint = tmp_path / "draft.json"
    session.submit('/project checkpoint ' + str(checkpoint))
    old_project = ProjectOnboarding(session).draft.project_id
    restored = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=lambda *a, **k: final()))
    restored.submit('/project resume ' + str(checkpoint))
    draft = ProjectOnboarding(restored).draft
    assert draft.status == "draft" and all(not f["confirmed"] for f in draft.fields.values())
    assert restored.state.pending_approval is None
    model.write_bytes(b"changed model")
    restored.submit('/project show')
    assert "source-change" in {c["id"] for c in ProjectOnboarding(restored).draft.conflicts}
    session.state.task_results.append("old evidence")
    session.submit('/project model ' + str(model))
    draft = ProjectOnboarding(session).draft
    assert draft.project_id != old_project and not draft.fields and not session.state.task_results
    assert session.state.pending_approval is None and session.lead.context_builder is None


def test_undo_revokes_confirmation_and_failure_has_no_substitute(project):
    session, queue, *_ = project
    populate(project)
    session.submit('/project confirm')
    statement = "总评估改为 10 次。"
    execution = values()["execution"] | {"max_evaluations": 10}
    queue.append(update_response(statement, {"execution": execution}))
    session.submit(statement)
    session.submit('/project undo')
    draft = ProjectOnboarding(session).draft
    assert draft.fields["execution"]["value"]["max_evaluations"] == 8 and draft.status == "draft"
    session.lead.llm_call = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("offline"))
    with pytest.raises(AgentUnavailableError):
        session.submit("继续澄清")
    assert session.state.agent_unavailable and not session.state.pending_approval
    assert ProjectOnboarding(session).draft.fields == draft.fields


@pytest.mark.parametrize("explicit_optimization", [False, True])
def test_effectful_tool_only_previews_via_gateway_and_rejects_direct_execution(project, tmp_path, explicit_optimization):
    session, _, _, _, _ = project
    populate(project)
    session.submit('/project confirm')
    path = tmp_path / "case.yaml"
    session.submit('/project save ' + str(path))
    _, schema = build_optimization_plan(path)
    gateway = ActionGateway(schema)
    seen = []
    def provider(*a, **kw):
        if "Classify the PAO user message" in kw["system"]:
            return json.dumps(payload(kind="optimization_request" if explicit_optimization else "project_query",
                role="optimization_advisor", action="optimize" if explicit_optimization else "read"))
        evidence = json.loads(kw["user"])["evidence"]
        seen.append(evidence)
        return final("已形成待审核计划。") if evidence.get("tool_results") else calls("plan_optimization")
    lead = LeadAgent(context_builder=ProcessContextBuilder(yaml_path=path, read_only=True),
                     action_gateway=gateway, llm_config=CFG, llm_call=provider)
    chat = ConversationSession(SessionState(), lead)
    answer = chat.submit("请检查配置并生成待审核计划")
    assert answer.approval_request and seen[-1]["tool_results"][0]["value"]["executed"] is False
    audit_id = answer.approval_request.preview["audit_id"]
    assert gateway.status(audit_id).value == "pending"
    with pytest.raises(Exception, match="approved"):
        gateway.execute(audit_id)
    lead.invalidate_approval(chat.state)
    assert gateway.status(audit_id).value == "rejected"
    from src.agents.subagents.tools.run_case import _impl_run_case
    from src.agents.subagents.tools.discover_tunables import _scan_aspen_file
    with pytest.raises(PermissionError):
        _impl_run_case(str(path), None, 0)
    with pytest.raises(PermissionError, match="Gateway"):
        _scan_aspen_file(str(project[3]), str(tmp_path / "new.db"))


def test_kinetics_tool_runs_real_numerical_kernel_with_explicit_identity(tmp_path):
    attachment = tmp_path / "rates.csv"
    rows = "temperature_K,rate_constant\n" + "\n".join(f"{t},{1e6 * math.exp(-50000 / (8.314462618 * t))}" for t in (320, 350, 400, 450))
    attachment.write_text(rows, encoding="utf-8")
    context = {"attachments": [{"path": str(attachment), "status": "available", "content": rows, "version": file_version(attachment)}]}
    registry = default_registry()
    tool = ToolSession(registry.tools, tool_task("kinetics_advisor", ("fit_arrhenius_linear",)), context)
    result = tool.invoke("fit_arrhenius_linear", {"index": 0, "rate_unit": "1/s"})["value"]
    assert result["method"] == "linearized_least_squares" and result["Ea"] == pytest.approx(50000)
    assert result["k0"] == pytest.approx(1e6) and result["source_version"]["sha256"]


def test_cli_no_yaml_new_and_existing_paths_stay_compatible(monkeypatch, tmp_path):
    from src import main
    seen = []
    monkeypatch.setattr(main, "_setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(main, "_run_unified_chat_loop", lambda **kw: seen.append(kw) or 0)
    assert main.main(["--new-project"]) == 0 and seen[-1]["new_project"]
    assert main.main([str(tmp_path / "existing.yaml"), "--conversation"]) == 0
    assert seen[-1]["yaml_path"] == (tmp_path / "existing.yaml").resolve()


def test_config_validation_uses_actual_backend_and_never_loads_custom_code(project, tmp_path):
    session, *_ = project
    populate(project)
    session.submit('/project confirm')
    path = tmp_path / "validate.yaml"
    session.submit('/project save ' + str(path))
    context = ProcessContextBuilder(yaml_path=path, read_only=True).build(session=session.state)
    tool = ToolSession(default_registry().tools, tool_task("project_researcher", ("validate_config",)), context)
    result = tool.invoke("validate_config", {})["value"]
    assert result["valid"] and not result["aspen_opened"] and result["limits"]["max_evaluations"] == 8
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["objectives"][0].update(type="custom_module", module="must_not_import.py")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    context = ProcessContextBuilder(yaml_path=path, read_only=True).build(session=session.state)
    tool = ToolSession(default_registry().tools, tool_task("project_researcher", ("validate_config",)), context)
    with pytest.raises(PermissionError, match="custom module"):
        tool.invoke("validate_config", {})


def test_missing_llm_disables_onboarding_before_interpretation(tmp_path):
    from types import SimpleNamespace
    seen = []
    session = ConversationSession(SessionState(), LeadAgent(llm_config=SimpleNamespace(provider="deepseek", api_key=None),
        llm_call=lambda *a, **k: seen.append(k)))
    with pytest.raises(AgentUnavailableError):
        session.submit('/project new')
    assert seen == [] and session.state.agent_unavailable and session.state.pending_approval is None


def test_model_change_revokes_real_pending_gateway_plan(project, tmp_path):
    session, *_ = project
    populate(project)
    session.submit('/project confirm')
    path = tmp_path / "config.yaml"
    session.submit('/project save ' + str(path))
    plan, schema = build_optimization_plan(path)
    gateway = ActionGateway(schema)
    from src.agents.orchestration.models import ApprovalRequest, ActionProposal
    proposal = ActionProposal("run_optimization", "model", "test", payload={"optimization_plan": plan})
    preview = gateway.preview_proposal(proposal)
    gateway.approve(preview["audit_id"])
    session.lead.action_gateway = gateway
    session.state.pending_approval = ApprovalRequest(proposal, preview=preview)
    session.submit('/project model ' + str(project[3]))
    assert gateway.status(preview["audit_id"]).value == "rejected"
    with pytest.raises(Exception, match="approved"):
        gateway.execute(preview["audit_id"])


def test_current_source_hard_bounds_preserved_and_read_plan_is_non_executable(project, tmp_path):
    session, *_ = project
    populate(project)
    service = ProjectOnboarding(session)
    session.state.runtime_state["project_static_evidence"]["variables"] = {"t": values()["design_variables"][0]}
    draft = service.draft
    draft.fields["design_variables"]["value"][0]["upper_bound"] = 81
    with pytest.raises(ValueError, match="硬边界"):
        service.validate(draft)
    answer = session.submit('/project read-plan')
    assert not answer.details["read_plan"]["executable"] and session.state.pending_approval is None
    assert "COM" in TerminalRenderer().render(answer)


def test_tool_audit_does_not_keep_credentials():
    tools = ToolRegistry()
    tools.register(ToolDefinition("measure", "echo", lambda c, credentials: {"password": "private"},
        ("process_advisor",), {"credentials": str}, ("credentials",)))
    session = ToolSession(tools, tool_task(), {}, cfg=CFG)
    packet = session.invoke("measure", {"credentials": "private"})
    assert "private" not in json.dumps(session.audit) and packet["value"]["password"] == "[REDACTED]"


def test_conversation_checkpoint_restores_project_without_business_or_execution_confirmation(project, tmp_path):
    session, *_ = project
    populate(project)
    session.submit('/project confirm')
    path = tmp_path / "conversation.db"
    session.save_checkpoint(path)
    restored = ConversationSession.from_checkpoint(path, LeadAgent(llm_config=CFG, llm_call=lambda *a, **k: final()))
    draft = ProjectOnboarding(restored).draft
    assert draft.status == "draft" and draft.confirmed_digest is None
    assert not restored.state.pending_approval and all(not x["confirmed"] for x in draft.fields.values())


def test_existing_project_specialist_uses_actual_registered_snapshot_tool(project, tmp_path):
    session, *_ = project
    populate(project)
    session.submit('/project confirm')
    path = tmp_path / "existing.yaml"
    session.submit('/project save ' + str(path))
    seen = []
    def provider(*a, **kw):
        if "Classify the PAO user message" in kw["system"]:
            return json.dumps(payload(role="process_advisor"))
        evidence = json.loads(kw["user"])["evidence"]
        seen.append(evidence)
        if not evidence.get("tool_results"):
            return calls("read_node_db")
        params = evidence["tool_results"][0]["value"]["equipment_parameters"]["R1"]
        assert params[0]["aspen_path"] == P and params[0]["unit"] == "C"
        return final("R1 温度静态节点单位为 C，目录样本为 40；不是当前实时测量。")
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider,
        context_builder=ProcessContextBuilder(yaml_path=path, read_only=True)))
    answer = chat.submit("R1 温度节点的单位和已有证据是什么？")
    assert "40" in answer.text and answer.approval_request is None
    audit = next(x for x in chat.state.audit_events if x["event"] == "tool_call")
    assert audit["status"] == "completed" and audit["role"] == "process_advisor"
    assert seen[-1]["tool_results"][0]["source_versions"]["node_db"]["sha256"]
    print("\n模拟示例：已有项目专业问题 → process_advisor → read_node_db → 带版本证据 → Agent 回答")
    print_example(TerminalRenderer().render(answer))
    print(f"工具：{audit['tool']}；状态：{audit['status']}；证据：{audit['evidence_refs'][0]}；未运行 Aspen。")


def test_cli_end_to_end_builds_and_saves_without_yaml_or_execution(project, tmp_path, monkeypatch, capsys):
    from src import main
    from src.agents.lead import llm_client
    model, nodes = project[3:]
    statement = "完整要求：产品 A，S1 质量分数最大化，质量下限 0.9，温度 20 到 80 C，random，DOE 2，搜索 1，总评估 8，总时限 120 秒，最多失败 3 次，单次 20 秒。"
    def provider(*a, **kw):
        if "Classify the PAO user message" in kw["system"]:
            return json.dumps(payload(kind="chat"))
        data = json.loads(kw["user"])
        return update_response(data["message"], values() if data["message"] == statement else {})
    monkeypatch.setattr(llm_client, "load_llm_config", lambda: CFG)
    monkeypatch.setattr(llm_client, "chat_json", provider)
    destination = tmp_path / "cli.yaml"
    turns = iter(['/project confirm', '/project save ' + str(destination), 'exit'])
    monkeypatch.setattr("builtins.input", lambda prompt: next(turns))
    import logging
    assert main._run_unified_chat_loop(config_path=model, yaml_path=None, db_path=tmp_path / "sim.db",
        node_db_path=nodes, initial_intent=statement, log=logging.getLogger("batch3"), new_project=True) == 0
    assert destination.exists() and not (tmp_path / "sim.db").exists()
    assert yaml.safe_load(destination.read_text(encoding="utf-8"))["execution"]["max_evaluations"] == 8
    assert "保存" in capsys.readouterr().out


@pytest.mark.parametrize("simple_parser", [False, True])
def test_project_dotenv_loads_by_default_without_opt_in(tmp_path, monkeypatch, simple_parser):
    """Only a temporary fake dotenv is selected; no repository secret reads."""
    from src.agents.lead import llm_client
    import builtins
    fake_env = tmp_path / "fake.env"
    fake_env.write_text('PAO_LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=fake-dotenv-key\nPAO_LLM_MODEL=fake-dotenv-model\n', encoding="utf-8-sig")
    monkeypatch.setattr(llm_client, "_PROJECT_ENV_FILE", fake_env)
    monkeypatch.delenv("PAO_LOAD_DOTENV", raising=False)
    monkeypatch.setenv("PAO_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "stale-test-value")
    original_import = builtins.__import__
    if simple_parser:
        def without_dotenv(name, *args, **kwargs):
            if name == "dotenv":
                raise ImportError("test optional dependency absent")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", without_dotenv)
    # Track direct os.environ assignments made by either loader for cleanup.
    monkeypatch.setenv("PAO_LLM_MODEL", "old-test-model")
    cfg = llm_client.load_llm_config()
    assert cfg.provider == "deepseek" and cfg.model == "fake-dotenv-model"
    assert cfg.api_key == "fake-dotenv-key" and llm_client.is_configured(cfg)
    assert "fake-dotenv-key" not in cfg.redacted()


def test_dotenv_opt_out_never_touches_file_and_missing_key_fails_closed(monkeypatch):
    from src.agents.lead import llm_client
    from src.agents.reasoning import configured
    monkeypatch.setenv("PAO_LOAD_DOTENV", "0")
    monkeypatch.setenv("PAO_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    def forbidden(**kwargs):
        raise AssertionError("dotenv must not be read")
    monkeypatch.setattr(llm_client, "_load_dotenv_if_present", forbidden)
    cfg = llm_client.load_llm_config()
    assert not llm_client.is_configured(cfg)
    with pytest.raises(AgentUnavailableError):
        configured(cfg)


def test_dotenv_config_keeps_explicit_provider_and_model_overrides(tmp_path, monkeypatch):
    from src.agents.lead import llm_client
    fake_env = tmp_path / "fake.env"
    fake_env.write_text("PAO_LLM_PROVIDER=deepseek\nPAO_LLM_MODEL=default-test-model\nDEEPSEEK_API_KEY=fake-key\n", encoding="utf-8")
    monkeypatch.setattr(llm_client, "_PROJECT_ENV_FILE", fake_env)
    monkeypatch.delenv("PAO_LOAD_DOTENV", raising=False)
    for name in ("PAO_LLM_PROVIDER", "PAO_LLM_MODEL", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-explicit-key")
    cfg = llm_client.load_llm_config(provider="openai", model="explicit-test-model")
    assert cfg.provider == "openai" and cfg.model == "explicit-test-model"
    assert cfg.api_key == "fake-explicit-key"


@pytest.mark.parametrize("bad_routes", [[], [{"name": "onboarding_agent", "task": "inspect", "priority": "normal", "parallel": False}],
    [{"name": "technical_advisor", "task": "inspect", "priority": "normal", "parallel": False}]])
def test_project_contents_missing_or_unknown_route_is_repaired_by_llm(bad_routes):
    seen, built = [], []
    class Builder:
        def build(self, **kwargs):
            built.append(True)
            return {"equipment": [{"id": "R1", "type": "reactor"}], "status": "partial"}
    def provider(*a, **kw):
        seen.append(kw)
        if "Classify the PAO user message" in kw["system"]:
            classified = payload(role="process_advisor")
            if len(seen) == 1:
                classified["subagents"] = bad_routes
                return json.dumps(classified)
            repair = json.loads(kw["user"])
            assert repair["request_type"] == "repair_intent_json"
            assert "subagents" in repair["validation_error"]["message"]
            assert not built
            return json.dumps(classified)
        evidence = json.loads(kw["user"])["evidence"]
        assert evidence["equipment"][0]["id"] == "R1"
        return final("当前静态资料包含 R1 反应器，来源不完整。")
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider, context_builder=Builder()))
    answer = chat.submit("你能查看本项目的内容是什么吗？")
    assert answer.task_results[0].agent == "process_advisor" and "R1" in answer.text
    assert chat.state.classification_source == "llm_repaired" and len(seen) == 3
    assert chat.state.pending_approval is None


def test_invalid_role_repair_still_fails_closed_before_context_or_dispatch():
    seen, built = [], []
    class Builder:
        def build(self, **kwargs):
            built.append(True)
            raise AssertionError("invalid route must not load context")
    def provider(*a, **kw):
        seen.append(kw)
        return json.dumps(payload() | {"subagents": []})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider, context_builder=Builder()))
    with pytest.raises(AgentUnavailableError) as error:
        chat.submit("你能查看本项目的内容是什么吗？")
    assert error.value.code == "llm_repair_failed" and len(seen) == 2
    assert not built and not chat.state.task_results and chat.state.pending_approval is None


def test_classifier_prompt_uses_actual_registered_roles_and_keeps_general_chat_route_free():
    from src.agents.lead.intent import IntentClassifier
    from src.agents.subagents.registry import SubagentDefinition
    seen = []
    def provider(*a, **kw):
        seen.append(kw)
        return json.dumps(payload(kind="chat"))
    classifier = IntentClassifier(llm_config=CFG, llm_call=provider,
        role_definitions={"local_reader": SubagentDefinition("local_reader", "Read the selected project facts")})
    answer = classifier.classify("你好")
    assert answer.intent.subagents == [] and answer.retry_count == 0
    assert '"local_reader": "Read the selected project facts"' in seen[0]["system"]


def test_project_researcher_accepts_named_citations_and_preserves_source_snapshot():
    """Replay the failing provider envelope through the public conversation API."""
    context = {"status": "partial", "equipment": [{"id": "R1", "block_type": "reactor"}],
               "streams": [{"id": "S1", "composition": {"A": .8}}],
               "sources": [{"source_ref": "fake://selected-model"}],
               "dependency_versions": {"model": {"path": "fake.apw", "sha256": "fake-version"}}}
    citations = {"case_ref": "/sources/0/source_ref", "model_path": "/dependency_versions/model/path",
                 "status": "/status", "reactor": "/equipment/0",
                 "products": ["/streams/0/composition"], "nested": {"equipment_type": "/equipment/0/block_type"}}
    class Builder:
        def build(self, **kwargs):
            return deepcopy(context)
    seen = []
    def provider(*a, **kw):
        seen.append(kw)
        if "Classify the PAO user message" in kw["system"]:
            return json.dumps(payload(role="project_researcher"))
        assert "optional top-level evidence" in kw["system"]
        evidence = json.loads(kw["user"])["evidence"]
        assert evidence["equipment"][0]["id"] == "R1"
        return json.dumps({"text": "当前静态资料包含 R1 反应器，来源不完整。", "uncertainties": ["未读取实时模型"],
            "confidence": .74, "blocks": [
                {"kind": "paragraph", "text": "R1 为反应器。", "evidence_refs": ["/equipment/0/block_type"]},
                {"kind": "table", "columns": ["物流", "A 分数"], "rows": [["S1", .8]],
                 "evidence_refs": ["/streams/0/composition"]},
                {"kind": "bullets", "items": ["当前只有静态来源。"]},
                {"kind": "limitation", "text": "未执行 Aspen。"},
                {"kind": "next_steps", "epistemic": "hypothesis", "items": ["进一步核实目标基准。"]}],
            "evidence": citations})
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider, context_builder=Builder()))
    answer = chat.submit("你能查看本项目的内容是什么吗？")
    assert len(seen) == 2 and answer.task_results[0].agent == "project_researcher"
    assert "R1" in TerminalRenderer().render(answer) and len(answer.blocks) == 5
    assert answer.details["reasoning_evidence"]["answer_citations"] == citations
    assert answer.task_results[0].evidence[0].value["answer_citations"] == citations
    assert answer.details["reasoning_evidence"]["dependency_versions"] == context["dependency_versions"]
    assert "answer_citations" not in context  # Provider references never modify source facts.
    assert answer.approval_request is None and chat.state.pending_approval is None
    assert not any(e["event"] in {"tool_call", "agent_unavailable"} for e in chat.state.audit_events)
    restored = type(answer).from_dict(json.loads(json.dumps(answer.to_dict())))
    assert restored.details["reasoning_evidence"]["answer_citations"] == citations


@pytest.mark.parametrize("patch,reason_code", [
    ({"evidence": {"source": "/missing"}}, "blocks_or_evidence_invalid"),
    ({"evidence": {"source": "/equipment/01"}}, "blocks_or_evidence_invalid"),
    ({"evidence": {"source": "/equipment/-1"}}, "blocks_or_evidence_invalid"),
    ({"evidence": {"source": "invented engineering fact"}}, "blocks_or_evidence_invalid"),
    ({"evidence": {"measurement": 42}}, "blocks_or_evidence_invalid"),
    ({"evidence": {"source": ["/equipment/0", {"nested": "/missing"}]}}, "blocks_or_evidence_invalid"),
    ({"evidence": ["/equipment/0"]}, "blocks_or_evidence_invalid"),
    ({"blocks": [{"kind": "paragraph", "text": "R1", "evidence_refs": ["/missing"]}]}, "block_evidence_reference_invalid"),
    ({"private-fake-key": "sensitive-test-value"}, "answer_fields"),
    ({"text": ""}, "text_or_uncertainties_type"),
    ({"confidence": True}, "confidence_type_or_range"),
])
def test_invalid_specialist_output_stops_with_safe_schema_diagnostic(patch, reason_code, caplog):
    tools = ToolRegistry()
    task = tool_task()
    task.metadata["session_id"] = "test-session"
    session = ToolSession(tools, task, {}, cfg=CFG)
    response = json.loads(final()) | patch
    with pytest.raises(AgentUnavailableError) as caught:
        reason("inspect", context={"equipment": [{"id": "R1"}]}, role="process_advisor", llm_config=CFG,
               llm_call=lambda *a, **k: json.dumps(response), tool_session=session)
    assert caught.value.code == "agent_output_invalid" and reason_code in caught.value.user_message
    record = next(e for e in session.audit if e["event"] == "agent_output_validation_failed")
    assert record["reason"] == reason_code and record["response_type"] == "dict"
    assert record["role"] == "process_advisor" and record["task_id"] == task.task_id
    assert record["session_id"] == "test-session"
    diagnostic = json.dumps(session.audit) + caplog.text + str(caught.value)
    assert "private-fake-key" not in diagnostic and "sensitive-test-value" not in diagnostic
    assert "invented engineering fact" not in diagnostic


def test_named_citations_resolve_escaped_json_pointer_tokens_without_certifying_metrics():
    context = {"a/b": {"~name": "selected source"}}
    data, evidence = reason("source", context=context, llm_config=CFG,
        llm_call=lambda *a, **k: json.dumps(json.loads(final()) | {"evidence": {"source": "/a~1b/~0name"}}))
    assert evidence["answer_citations"] == data["evidence"] == {"source": "/a~1b/~0name"}
    response = json.loads(final()) | {"evidence": {"source": "/a~1b/~0name"},
        "blocks": [{"kind": "metrics", "evidence_refs": ["/a~1b/~0name"],
                    "metrics": [{"label": "purity", "value": .99, "unit": "1", "baseline": None,
                                 "delta": None, "scope": "current"}]}]}
    with pytest.raises(AgentUnavailableError):
        reason("source", context=context, llm_config=CFG, llm_call=lambda *a, **k: json.dumps(response))


def test_invalid_citation_fails_closed_at_conversation_boundary():
    class Builder:
        def build(self, **kwargs):
            return {"status": "partial", "equipment": [{"id": "R1"}]}
    responses = iter([json.dumps(payload(role="project_researcher")),
                      json.dumps(json.loads(final()) | {"evidence": {"source": "/missing"}}),
                      json.dumps(json.loads(final()) | {"evidence": {"source": "/missing"}})])
    seen = []
    def provider(*a, **kw):
        seen.append(kw)
        return next(responses)
    chat = ConversationSession(SessionState(), LeadAgent(llm_config=CFG, llm_call=provider, context_builder=Builder()))
    with pytest.raises(AgentUnavailableError) as caught:
        chat.submit("查看项目")
    assert caught.value.code == "agent_output_invalid" and len(seen) == 3
    assert chat.state.internal_error["code"] == "agent_output_invalid"
    assert not chat.state.task_results and not chat.last_answer
    assert chat.state.pending_approval is None
    assert any(e["event"] == "agent_output_validation_failed" and e["reason"] == "blocks_or_evidence_invalid"
               for e in chat.state.audit_events)
    with pytest.raises(AgentUnavailableError):
        chat.submit("继续")
    assert len(seen) == 3  # No further retry or implicit authorization after failed repair.

def test_project_population_protocol_repairs_once_without_authorizing(project):
    session, queue, seen, *_ = project
    message = '使用GA，只建立草稿。'
    bad = json.loads(update_response(message, {'optimizer': {'type': 'ga', 'backend': 'pymoo'}}))
    bad['updates']['population'] = bad['updates'].pop('optimizer')
    queue.extend([json.dumps(bad), update_response(message, {'optimizer': {'type': 'ga', 'backend': 'pymoo'}})])
    session.submit(message)
    draft = ProjectOnboarding(session).draft
    assert draft.fields['optimizer']['value']['type'] == 'ga'
    assert 'population' not in draft.fields and session.state.pending_approval is None
    assert any(e['event'] == 'project_protocol_repair' for e in session.state.audit_events)
    assert 'invalid_response' in seen[-1]


def test_constraint_correction_cannot_drop_existing_residual_scale(project):
    session, queue, *_ = project
    populate(project)
    onboarding = ProjectOnboarding(session)
    draft = onboarding.draft
    draft.fields["constraints"]["value"][0]["scale"] = 1.0
    onboarding._store(draft)
    message = "change threshold only"
    changed = values()["constraints"]
    changed[0]["threshold"] = .95
    bad = json.loads(update_response(message, {"constraints": changed}))
    with pytest.raises(ValueError, match="residual scale"):
        onboarding._apply_turn(bad, draft, message, {})
    assert onboarding.draft.fields["constraints"]["value"][0]["scale"] == 1.0
    changed[0]["scale"] = 1.0
    good = json.loads(update_response(message, {"constraints": changed}))
    onboarding._apply_turn(good, draft, message, {})
    assert onboarding.draft.fields["constraints"]["value"][0]["threshold"] == .95
