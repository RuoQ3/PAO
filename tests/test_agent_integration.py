from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.closed_loop.advisor import ProcessDecisionAgent
from src.models.process_case import ProcessCase, ObjectiveValue, ConstraintValue, CaseStatus
from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig, optimize_pareto_case


def cfg(tmp_path):
    return ParetoOptimizeCaseConfig(
        param_bounds={"x": (0, 1)}, objective_names=["a", "b"], n_initial=1,
        n_iterations=5, reference_values={"x": 0.5}, surrogate_model="random", random_seed=2,
        db_path=tmp_path / "results.db", agent_loop={"enabled": True, "max_evaluations": 5},
        agent_checkpoint_path=str(tmp_path / "agent.db"),
    )


def fake_run(driver, design_vars, run_config, iteration=0, tags=None, run_id=None):
    x = design_vars["x"]
    return ProcessCase(iteration=iteration, design_vars=dict(design_vars), status=CaseStatus.SUCCESS,
                       objectives=[ObjectiveValue("a", 1 + x), ObjectiveValue("b", 2 - x)],
                       constraints=[ConstraintValue("c", -1)], tags=tags or [])


def test_existing_optimizer_entry_routes_to_loop_and_report(tmp_path, monkeypatch):
    import src.workflows.agent_optimize as loop
    from src.agents.closed_loop.advisor import fallback_plan
    monkeypatch.setattr(loop, "run_case", fake_run)
    monkeypatch.setattr(ProcessDecisionAgent, "propose", lambda self, snap: (fallback_plan(snap, 2), "test"))
    config = cfg(tmp_path)
    result = optimize_pareto_case(None, config)
    assert result.n_total == 5 and result.n_success == 5
    assert result.first_front
    assert "completed_verified" in result.early_stop_reason
    assert (tmp_path / "agent.report.json").exists()
    assert (tmp_path / "agent.report.md").exists()
    config.agent_resume = True
    monkeypatch.setattr(loop, "run_case", lambda *a, **kw: pytest.fail("terminal resume must not run"))
    assert optimize_pareto_case(None, config).n_total == 5


def test_skopt_uses_outside_region_history_without_rejecting(caplog):
    from src.optimization.surrogate import SurrogateConfig, SurrogateOptimizer
    optimizer = SurrogateOptimizer([(0, 1)], SurrogateConfig(model="GP", n_initial_min=2, random_seed=2))
    for x in (0.05, 0.2, 0.8, 0.95):
        optimizer.tell([x], (x - 0.6) ** 2, is_success=True)
    value = optimizer.ask(active_bounds=[(0.4, 0.7)])[0]
    assert 0.4 <= value <= 0.7
    assert len(optimizer._skopt.Xi) == 4
    assert not any("失败" in record.message for record in caplog.records)


def test_incremental_session_keeps_all_observations_across_regions():
    from src.optimization.search_session import ParetoSearchSession
    config = ParetoOptimizeCaseConfig(param_bounds={"x": (0, 1)}, objective_names=["a", "b"],
                                      surrogate_model="GP", n_initial=1, n_initial_min=2, random_seed=5)
    session = ParetoSearchSession(config)
    for x in (0.1, 0.5, 0.9):
        session.tell(fake_run(None, {"x": x}, None), {"x": x})
    assert 0.3 <= session.ask({"x": (0.3, 0.6)})["x"] <= 0.6
    assert 0.7 <= session.ask({"x": (0.7, 0.8)})["x"] <= 0.8
    assert len(session.optimizer._observations) == 3


def test_llm_json_proposal_is_used_and_malformed_response_is_rejected(monkeypatch):
    from src.agents import llm_client
    monkeypatch.setattr(llm_client, "is_configured", lambda cfg: True)
    monkeypatch.setattr(llm_client, "chat", lambda *a, **kw: '{"action":"probe","candidate":{"x":0.6}}')
    agent = ProcessDecisionAgent(llm_config=SimpleNamespace())
    plan, source = agent.propose({"recent_observations": []})
    assert source == "llm" and plan.candidate == {"x": 0.6}
    monkeypatch.setattr(llm_client, "chat", lambda *a, **kw: '```invalid JSON```')
    with pytest.raises(ValueError):
        agent.propose({})


def test_graph_preserves_confirmation_and_skips_final_manual_loop(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    import src.agents.graph as graph
    monkeypatch.setattr(graph, "onboarding_node", lambda s: {})
    monkeypatch.setattr(graph, "write_feasibility_node", lambda s: {})
    monkeypatch.setattr(graph, "human_confirm_node", lambda s: {"current_phase": "optimizing"})
    monkeypatch.setattr(graph, "optimization_node", lambda s: {"iteration": 1})
    monkeypatch.setattr(graph, "analysis_node", lambda s: {"analysis_report": "ok"})
    app = graph.build_graph(MemorySaver())
    thread = {"configurable": {"thread_id": "test"}}
    app.invoke(graph.PAOGraphState(agent_loop={"enabled": True}), thread)
    assert app.get_state(thread).next == ("human_confirm",)
    result = app.invoke(None, thread)
    assert not app.get_state(thread).next
    assert result["current_phase"] == "done"


def test_graph_optimization_forwards_agent_settings(tmp_path, monkeypatch):
    import src.agents.graph as graph
    from src.agents.tools import _common
    config = cfg(tmp_path)
    seen = []
    class Driver:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def open(self, path): pass
    monkeypatch.setattr(_common, "_import_pareto_deps", lambda: None)
    monkeypatch.setattr(_common, "_load_optimize_config", lambda p: (config, Path("mock.bkp"), {}))
    monkeypatch.setattr(_common, "_AspenDriver", Driver)
    monkeypatch.setattr(_common, "_resolve_config_path", lambda p: Path(p))
    def optimize(**kw):
        seen.append(kw["config"])
        return SimpleNamespace(early_stop_reason="evaluation_budget:completed_verified")
    monkeypatch.setattr(_common, "_optimize_pareto_fn", optimize)
    from src.agents.tools import optimize_pareto
    monkeypatch.setattr(optimize_pareto, "_fmt_pareto_result_summary", lambda *a: "ok")
    result = graph.optimization_node(graph.PAOGraphState(
        config_yaml_path=str(tmp_path / "cfg.yaml"), agent_loop={"enabled": True, "max_evaluations": 6},
        agent_checkpoint_path=str(tmp_path / "custom.db"), agent_resume=True,
    ))
    assert seen[0].agent_loop["max_evaluations"] == 6
    assert seen[0].agent_resume
    assert result["termination_reason"].endswith("completed_verified")


def test_config_fingerprints_custom_objectives_and_model():
    from src.utils.file_io import load_optimize_config
    config, _, _ = load_optimize_config("cases/demo_case_2/pareto_config_epsd_aligned.yaml")
    assert config.agent_context["model_sha256"]
    assert any(p.endswith("epsd_objectives.py") for p in config.agent_context["objective_module_hashes"])


def test_optional_botorch_constrained_fit_includes_infeasible(monkeypatch):
    pytest.importorskip("botorch")
    from src.optimization.botorch_backend import BoTorchMOOptimizer
    from src.optimization.surrogate import SurrogateConfig
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
    optimizer = BoTorchMOOptimizer([(0, 1)], 2, set(), SurrogateConfig(model="qEHVI"))
    for x, margin in [(0.1, -0.2), (0.3, -0.1), (0.5, 0.1), (0.7, 0.2)]:
        optimizer.tell([x], 0, is_success=margin >= 0, y_vec=[x + 1, 2 - x], c_vec={"quality": margin})
    captured = []
    def partition(*args, **kwargs):
        captured.append(kwargs["Y"])
        return NondominatedPartitioning(*args, **kwargs)
    monkeypatch.setattr("botorch.utils.multi_objective.box_decompositions.non_dominated.NondominatedPartitioning", partition)
    monkeypatch.setattr(optimizer, "_optimize_and_round", lambda *a: [0.55])
    assert optimizer._ask_botorch([(0.4, 0.6)]) == [0.55]
    assert captured[0].shape[0] == 2
