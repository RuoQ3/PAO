from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.closed_loop.analysis import (
    build_analysis_report,
    compare_analysis_reports,
)
from src.agents.closed_loop.contracts import ActionPlan, LoopConfig, validate_plan
from src.agents.closed_loop.journal import SessionJournal
from src.models.process_case import (
    CaseStatus,
    ConstraintValue,
    ObjectiveValue,
    ProcessCase,
)
from src.models.simulation_result import RunStatus, SimulationResult
from src.workflows.agent_optimize import optimize_agent_case
from src.workflows.optimize_pareto_case import (
    ParetoOptimizeCaseConfig,
    _extract_all_objectives,
    _extract_constraint_margins,
)


def config():
    return ParetoOptimizeCaseConfig(
        param_bounds={"x": (0.0, 1.0)}, objective_names=["cost", "energy"],
        reference_values={"x": 0.5}, n_initial=1, n_iterations=10,
        surrogate_model="random", random_seed=4,
    )


def evaluation(driver, actual, run_config, iteration, tags):
    x = actual["x"]
    violation = abs(x - 0.5) - 0.2
    sim = SimulationResult(status=RunStatus.SUCCESS, success=True,
                           requested_inputs=dict(actual), actual_inputs=dict(actual))
    return ProcessCase(
        status=CaseStatus.SUCCESS if violation <= 0 else CaseStatus.INFEASIBLE,
        iteration=iteration, design_vars=dict(actual), sim_result=sim, tags=list(tags),
        objectives=[ObjectiveValue("cost", 1 + x), ObjectiveValue("energy", 2 - x)],
        constraints=[ConstraintValue("quality", violation)],
    )


class ScriptedAgent:
    def __init__(self, callback=None):
        self.snapshots = []
        self.callback = callback

    def propose(self, snapshot):
        self.snapshots.append(deepcopy(snapshot))
        if self.callback:
            return self.callback(snapshot), "test"
        return ActionPlan(
            evidence_case_ids=[r["case_id"] for r in snapshot["recent_observations"][-1:]],
            batch_size=snapshot["batch_size_limit"],
        ), "test"


def run(tmp_path, cfg=None, **kwargs):
    args = dict(settings=LoopConfig(max_evaluations=8, batch_size=2),
                checkpoint_path=tmp_path / "checkpoint.db", context={"model": "test-v1"},
                knowledge=[], agent=ScriptedAgent(), evaluator=evaluation)
    args.update(kwargs)
    return optimize_agent_case(None, cfg or config(), **args)


def test_constraint_violation_enters_botorch_but_not_front(monkeypatch):
    import src.optimization.botorch_backend as backend
    monkeypatch.setattr(backend, "_select_device", lambda: None)
    monkeypatch.setattr(backend, "_check_botorch", lambda: False)
    case = evaluation(None, {"x": 0.9}, None, 0, [])
    cfg = config()
    assert not case.success and case.simulation_valid
    assert _extract_all_objectives(case, cfg) is None
    y = _extract_all_objectives(case, cfg, allow_infeasible=True)
    optimizer = backend.BoTorchMOOptimizer([(0, 1)], 2, set(), SimpleNamespace())
    optimizer.tell([0.9], 0, is_success=False, y_vec=y, c_vec=_extract_constraint_margins(case))
    assert optimizer._train_C[0][0] < 0
    assert optimizer._train_feasible == [False]
    from src.optimization.pareto import compute_pareto
    assert not compute_pareto([case], cfg.objective_names).first_front


def test_constraints_reject_changed_schema_and_nonfinite(monkeypatch):
    import src.optimization.botorch_backend as backend
    monkeypatch.setattr(backend, "_select_device", lambda: None)
    optimizer = backend.BoTorchMOOptimizer([(0, 1)], 2, set(), SimpleNamespace())
    optimizer.tell([0.2], 0, is_success=True, y_vec=[1, 2], c_vec={"a": 0.1})
    for c in ({"b": 0.2}, {"a": float("nan")}):
        with pytest.raises(ValueError):
            optimizer.tell([0.3], 0, is_success=False, y_vec=[1, 2], c_vec=c)
    assert len(optimizer._train_X) == 1


def test_end_to_end_budget_feedback_and_verification(tmp_path):
    agent = ScriptedAgent()
    cfg = config()
    cfg.db_path = tmp_path / "simulation.db"
    original = deepcopy(cfg.param_bounds)
    state = run(tmp_path, cfg, agent=agent)
    assert state["used"] == len(state["observations"]) == 8
    assert state["result"] == "completed_verified"
    assert len(state["verification_ids"]) == 1
    assert state["observations"][-1]["initialization"] == "reset"
    assert agent.snapshots[1]["previous_actions"][0]["outcome"]["case_ids"]
    assert cfg.param_bounds == original
    from src.database.simulation_db import SimulationDB
    db = SimulationDB(cfg.db_path)
    assert len(db.query_cases(session_id=cfg.session_id, limit=100)) == 8
    db.close()
    # The checkpoint is authoritative: a completed resume also rebuilds a lost projection.
    Path(cfg.db_path).unlink()
    resumed = run(tmp_path, cfg, resume=True)
    assert resumed["used"] == 8
    db = SimulationDB(cfg.db_path)
    assert len(db.query_cases(session_id=cfg.session_id, limit=100)) == 8
    db.close()


def test_goal_requires_one_feasible_point_and_fresh_recheck(tmp_path):
    settings = LoopConfig(max_evaluations=8, objective_targets={"cost": 1.5, "energy": 1.5})
    state = run(tmp_path, settings=settings)
    assert state["result"] == "target_verified"
    assert state["used"] == 2
    assert state["observations"][0]["optimizer_inputs"] == state["observations"][1]["optimizer_inputs"]


def test_failed_recheck_never_claims_goal(tmp_path):
    def drifting(*args):
        case = evaluation(*args)
        if "verification" in args[-1]:
            case.objectives[0].value += 20
        return case
    state = run(tmp_path, evaluator=drifting,
                settings=LoopConfig(max_evaluations=8, objective_targets={"cost": 1.5}))
    assert state["result"] == "verification_failed"


def test_invalid_agent_plan_falls_back_and_records_reason(tmp_path):
    agent = ScriptedAgent(lambda snap: ActionPlan(action="set_region", search_region={"x": [-3, 5]}))
    state = run(tmp_path, agent=agent)
    assert all(d["source"] == "rules:rejected_proposal" for d in state["decisions"])
    assert state["decisions"][0]["rejection"]
    assert all(0 <= r["optimizer_inputs"]["x"] <= 1 for r in state["observations"])


def test_resume_replays_evidence_and_does_not_repeat_unknown_trial(tmp_path):
    captured = []
    def interrupt(driver, actual, rc, iteration, tags):
        captured.append(dict(actual))
        if iteration == 2:
            raise KeyboardInterrupt()
        return evaluation(driver, actual, rc, iteration, tags)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, evaluator=interrupt)
    with SessionJournal(tmp_path / "checkpoint.db") as journal:
        checkpoint = journal.load()
        assert checkpoint["used"] == 3 and checkpoint["pending"]
        previous_ids = [r["case_id"] for r in checkpoint["observations"]]
    resumed_agent = ScriptedAgent()
    state = run(tmp_path, resume=True, agent=resumed_agent)
    assert state["used"] == 8
    assert [r["case_id"] for r in state["observations"][:2]] == previous_ids
    assert "outcome unknown" in state["observations"][2]["notes"]
    assert state["observations"][3]["initialization"] == "reset"
    assert resumed_agent.snapshots[0]["recent_observations"]
    assert run(tmp_path, resume=True)["used"] == 8  # completed resume is a read


def test_checkpoint_identity_and_overwrite_protection(tmp_path):
    run(tmp_path)
    with pytest.raises(ValueError, match="Checkpoint exists"):
        run(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, resume=True, context={"model": "test-v2"})
    with pytest.raises(ValueError, match="No checkpoint"):
        run(tmp_path / "other", resume=True)


def test_journal_blocks_second_writer(tmp_path):
    with SessionJournal(tmp_path / "test.db"):
        with pytest.raises(RuntimeError, match="Another controller"):
            with SessionJournal(tmp_path / "test.db"):
                pass


def test_previous_initialization_is_reset_after_failure(tmp_path):
    used = []
    def evaluator(driver, actual, rc, iteration, tags):
        used.append(rc.reinit)
        case = evaluation(driver, actual, rc, iteration, tags)
        if iteration == 1:
            case.status = CaseStatus.SIM_FAILED
        return case
    def plan(snapshot):
        return ActionPlan(initialization="previous", batch_size=2,
                          evidence_case_ids=[r["case_id"] for r in snapshot["recent_observations"][-1:]])
    run(tmp_path, evaluator=evaluator, agent=ScriptedAgent(plan))
    assert used[:3] == [True, False, True]


def test_derived_inputs_replayed_separately_and_actual_duplicates_removed(tmp_path):
    cfg = config()
    cfg.param_bounds = {"x": (0, 1), "frac": (0, 1)}
    cfg.fixed_vars = {"nstage": 4}
    cfg.reference_values = {"x": 0.5, "frac": 0.5}
    cfg.derived_var_specs = [{"frac_path": "frac", "target_path": "feed_stage", "depends_on": "nstage", "frac_lo": 1}]
    state = run(tmp_path, cfg)
    for row in state["observations"]:
        assert "frac" in row["optimizer_inputs"]
        assert "frac" not in row["design_vars"]
        assert isinstance(row["design_vars"]["feed_stage"], int)


def test_no_feasible_solution_never_verified(tmp_path):
    def failing(*args):
        c = evaluation(*args)
        c.status = CaseStatus.SIM_FAILED
        return c
    state = run(tmp_path, evaluator=failing)
    assert state["result"] == "no_feasible_solution"
    assert not state["verification_ids"]
    assert state["used"] <= 8


@pytest.mark.parametrize("patch", [
    {"max_evaluations": 0}, {"batch_size": -1}, {"max_step_fraction": float("nan")},
    {"verification_repeats": 0}, {"max_evaluations": 1}, {"batch_size": 1.5},
])
def test_invalid_budget_rejected(patch):
    with pytest.raises(ValueError):
        LoopConfig.from_dict(patch)


def test_step_limit_and_bounds_preserved(tmp_path):
    def plan(s):
        return ActionPlan(action="probe", candidate={"x": 1.0}, batch_size=1,
                          evidence_case_ids=[r["case_id"] for r in s["recent_observations"][-1:]])
    state = run(tmp_path, agent=ScriptedAgent(plan), settings=LoopConfig(max_evaluations=6, batch_size=1, max_step_fraction=0.05))
    xs = [r["optimizer_inputs"]["x"] for r in state["observations"] if "verification" not in r["tags"]]
    assert xs[0] == 0.5
    assert all(b - a <= 0.050000001 for a, b in zip(xs, xs[1:]))


def test_evidence_and_unknown_actions_rejected():
    state = {"hard_bounds": {"x": [0, 1]}, "observations": [{"case_id": "known"}]}
    with pytest.raises(ValueError, match="Unknown evidence"):
        validate_plan(ActionPlan(evidence_case_ids=["invented"]), state, LoopConfig(), set(), set())
    with pytest.raises(TypeError):
        ActionPlan.parse({"action": "continue", "constraints": []})
    with pytest.raises(ValueError):
        validate_plan(ActionPlan(action="write_aspen"), state, LoopConfig(), set(), set())


def test_cli_dry_run_does_not_import_com_or_mutate_config(tmp_path):
    from src.main import main
    source = Path("cases/demo_case_2/pareto_config_epsd_aligned.yaml")
    before = source.read_bytes()
    assert main([str(source), "--agent", "--dry-run", "--db", str(tmp_path / "sim.db")]) == 0
    assert source.read_bytes() == before
    assert not (tmp_path / "sim.db").exists()


def test_search_region_is_independent_from_hard_bounds(tmp_path):
    import yaml

    from src.utils.file_io import load_optimize_config
    raw = yaml.safe_load(Path("cases/demo_case_2/pareto_config_epsd_aligned.yaml").read_text())
    raw["search_region"] = {"T1_RR": [0.7, 1.0]}
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(raw))
    cfg, _, _ = load_optimize_config(file)
    path = next(p for p in cfg.param_bounds if "T1" in p and "BASIS_RR" in p)
    assert cfg.param_bounds[path] == (0.416, 2.08)
    assert cfg.search_region[path] == (0.7, 1.0)


def test_analysis_report_contains_structured_evidence_and_reliability():
    cases = [evaluation(None, {"x": x}, None, i, []) for i, x in enumerate((0.4, 0.5, 0.6, 0.9))]
    report = build_analysis_report(
        cases,
        objective_names=["cost", "energy"],
        param_paths=["x"],
        optimizer_inputs=[case.design_vars for case in cases],
        recent_window=3,
    )

    assert report["report_version"] == 1
    assert report["data_quality"]["n_total"] == 4
    assert report["data_quality"]["n_recent"] == 3
    assert report["metrics"]["feasible_rate"] < 1.0
    assert report["pareto"]["front_size"] >= 1
    assert report["constraints"]["quality"]["violation_rate"] > 0
    sensitivity = report["sensitivity"]
    assert sensitivity["available"]
    assert sensitivity["ranked_variables"][0]["path"] == "x"
    assert sensitivity["ranked_variables"][0]["reliable"]


def test_analysis_report_marks_insufficient_sensitivity_evidence():
    cases = [evaluation(None, {"x": 0.5}, None, 0, [])]
    report = build_analysis_report(
        cases,
        objective_names=["cost", "energy"],
        param_paths=["x"],
        optimizer_inputs=[{"x": 0.5}],
    )
    assert not report["data_quality"]["sufficient_for_sensitivity"]
    assert report["sensitivity"]["available"]
    assert not report["sensitivity"]["ranked_variables"][0]["reliable"]
    assert report["sensitivity"]["warnings"]


def test_action_effect_comparison_is_explicitly_non_causal():
    cases_before = [evaluation(None, {"x": 0.4}, None, 0, [])]
    cases_after = cases_before + [evaluation(None, {"x": 0.5}, None, 1, [])]
    before = build_analysis_report(
        cases_before,
        objective_names=["cost", "energy"],
        param_paths=["x"],
        optimizer_inputs=[{"x": 0.4}],
    )
    after = build_analysis_report(
        cases_after,
        objective_names=["cost", "energy"],
        param_paths=["x"],
        optimizer_inputs=[{"x": 0.4}, {"x": 0.5}],
    )
    effect = compare_analysis_reports(before, after, expected_effect="feasibility")
    assert "feasible_rate_delta" in effect
    assert effect["interpretation"].startswith("Observed association")


def test_closed_loop_persists_analysis_before_and_after_action(tmp_path):
    agent = ScriptedAgent()
    state = run(tmp_path, agent=agent)
    assert agent.snapshots[0]["analysis_report"]["report_version"] == 1
    decision = state["decisions"][0]
    assert decision["before_analysis"]["data_quality"]["n_total"] == 0
    assert "after_analysis" in decision["outcome"]
    assert "analysis_effect" in decision["outcome"]
    assert decision["outcome"]["analysis_effect"]["interpretation"].startswith(
        "Observed association"
    )

    with SessionJournal(tmp_path / "checkpoint.db") as journal:
        payload = journal.load()
    assert payload["decisions"][0]["before_analysis"]["report_version"] == 1
