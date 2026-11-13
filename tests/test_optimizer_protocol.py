"""Fourth batch offline acceptance. Real pymoo tests skip only if not installed."""
from dataclasses import replace
import json
from types import ModuleType
import sys

import pytest
import yaml

from src.optimization.problem import (OptimizationProblem, Variable, Objective, Constraint,
                                      AlgorithmSpec, Candidate, Observation)
from src.optimization.registry import (create_optimizer, capabilities, validate_combination,
                                       algorithm_catalogue, from_config, identity)
from src.optimization.population import PymooOptimizer, resolve_parameters
from src.optimization.execution import run_optimizer, observe_case, UnifiedResult
from src.models.process_case import ProcessCase, ObjectiveValue, ConstraintValue, CaseStatus

P = r"\Data\Blocks\B\Input\TEMP"
O = r"\Data\Blocks\B\Output\DUTY"


def problem(multi=False, constrained=False, integer=False, budget=17):
    return OptimizationProblem((Variable("temperature", P, -2, 2, "integer" if integer else "continuous", "C"),),
        (Objective("cost", False, "USD"), Objective("energy", True, "kW")) if multi else (Objective("cost", False, "USD"),),
        constraints=(Constraint("limit", {"operator": "<=", "threshold": 0, "unit": "C"}, 2),) if constrained else (),
        max_evaluations=budget, authorization_scope={P: (-2, 2)}, identity={"model_hash": "analytic"})


@pytest.mark.parametrize("algorithm,multi,integer,constrained", [
    ("PSO", True, False, False), ("PSO", False, True, False), ("PSO", False, False, True),
    ("NSGA-II", False, False, False), ("NSGA-II", True, True, False),
    ("qEHVI", True, True, False), ("BO", True, False, False)])
def test_combinations_rejected_before_backend(algorithm, multi, integer, constrained):
    p = problem(multi, constrained, integer)
    with pytest.raises(ValueError, match="unsupported"):
        validate_combination(p, AlgorithmSpec(algorithm, capabilities(algorithm)["backend"]), require_available=False)


@pytest.mark.parametrize("algorithm", ["GWO", "NSGA-III", "unknown"])
def test_unconnected_never_substitutes(algorithm):
    assert capabilities(algorithm)["status"] == "not_connected"
    with pytest.raises(ValueError, match="not connected"):
        create_optimizer(problem(), AlgorithmSpec(algorithm, "pymoo"))


def test_identity_and_directions_and_validation():
    p = problem(True, True)
    assert p.internal_objectives({"cost": 4, "energy": 3}) == [-4, 3]
    assert p.objectives[0].minimize is False
    assert p.inputs([1]) == {P: 1}
    for value in (3, float("nan"), True):
        with pytest.raises(ValueError):
            p.inputs([value])
    with pytest.raises(ValueError, match="degenerate"):
        replace(p, variables=(Variable("t", P, 1, 1),))
    with pytest.raises(ValueError, match="scale"):
        replace(p, constraints=(Constraint("c", {}, 0),))
    o = Observation("one", "infeasible", {"cost": 1, "energy": 2}, {"limit": 3})
    o.validate(p)
    with pytest.raises(ValueError, match="disagrees"):
        replace(o, status="feasible").validate(p)


@pytest.mark.parametrize("algo,raw", [("PSO", {"pop_size": 1}), ("PSO", {"generations": True}),
    ("PSO", {"w": 1.5}), ("PSO", {"c1": float("nan")}), ("PSO", {"c2": 5}),
    ("NSGA-II", {"mutation_eta": 0}), ("NSGA-II", {"crossover_probability": 2}),
    ("NSGA-II", {"mutation_probability": -1}), ("NSGA-II", {"repair": "round"})])
def test_population_parameters_fail_closed(algo, raw):
    with pytest.raises(ValueError):
        resolve_parameters(algo, raw)


def test_missing_dependency_does_not_call_any_other_algorithm(monkeypatch):
    import src.optimization.registry as registry
    original = registry.dependency_versions
    monkeypatch.setattr(registry, "dependency_versions", lambda b: {"pymoo": None} if b == "pymoo" else original(b))
    for algorithm in ("PSO", "NSGA-II"):
        with pytest.raises(RuntimeError, match="backend unavailable"):
            create_optimizer(problem(algorithm == "NSGA-II"), AlgorithmSpec(algorithm, "pymoo", resolve_parameters(algorithm, {})))


def test_random_resume_continues_rng_and_budget_and_checks_identity():
    p = problem(budget=5)
    spec = AlgorithmSpec("random", "python", seed=42)
    a = create_optimizer(p, spec)
    for _ in range(2):
        c = a.ask(5)[0]
        a.tell([Observation(c.id, "feasible", {"cost": c.x[0]}, evidence_id="e")])
    snapshot = json.loads(json.dumps(a.state_dict()))
    b = create_optimizer(p, spec)
    b.load_state_dict(snapshot)
    expected = a.ask(3)[0]
    assert b.ask(3)[0] == expected
    b.tell([Observation(expected.id, "feasible", {"cost": expected.x[0]})])
    calls = []
    result = run_optimizer(b, lambda c: calls.append(c) or Observation(c.id, "feasible", {"cost": c.x[0]}), remaining=lambda: 50)
    assert len(calls) == 2 and result.evaluations_used == 5 and len(result.records) == 5
    assert result.stop_reason == "budget_exhausted"
    for changed in (replace(p, identity={"model_hash": "changed"}), replace(p, max_evaluations=6)):
        with pytest.raises(ValueError, match="mismatch"):
            create_optimizer(changed, spec).load_state_dict(snapshot)
    a.stop("test")
    with pytest.raises(ValueError, match="pending"):
        create_optimizer(p, spec).load_state_dict(a.state_dict())


def test_real_skopt_protocol_and_legacy_bridge():
    from src.workflows.optimize_case import OptimizeCaseConfig
    from src.optimization.bo_adapter import make_legacy_bo
    config = OptimizeCaseConfig({P: (-2, 2)}, objective_name="cost", minimize=False,
                                n_initial=2, n_iterations=4, n_initial_min=2, random_seed=8)
    p, spec = from_config(config)
    a = create_optimizer(p, spec)
    for _ in range(3):
        c = a.ask(4)[0]
        a.tell([Observation(c.id, "feasible", {"cost": -c.x[0]**2})])
    assert a.capabilities()["id"] == "BO"
    assert len(a.history) == 3
    bridge = make_legacy_bo(config)
    bridge.tell([.2], 1, is_success=True)
    assert -2 <= bridge.ask()[0] <= 2
    with pytest.raises(ValueError, match="resume disabled"):
        bridge.load_state_dict(bridge.state_dict())


def test_real_parego_public_bridge():
    from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig
    from src.optimization.bo_adapter import make_legacy_bo
    config = ParetoOptimizeCaseConfig({P: (-2, 2)}, ["a", "b"], n_initial_min=2, random_seed=9)
    bridge = make_legacy_bo(config)
    bridge.tell([-.5], [.25, 2.25], is_success=True)
    bridge.tell([.5], [.25, .25], is_success=True)
    assert -1 <= bridge.ask(active_bounds=[(-1, 1)])[0] <= 1
    assert bridge.spec.algorithm == "ParEGO"


@pytest.mark.parametrize("status", [CaseStatus.INFEASIBLE, CaseStatus.SIM_FAILED, CaseStatus.OBJECTIVE_ERROR,
                                      CaseStatus.CONSTRAINT_ERROR, CaseStatus.SUCCESS])
def test_evidence_observation_policy(status):
    p = problem(constrained=True)
    case = ProcessCase(status=status, objectives=[ObjectiveValue("cost", 4, "USD", False)],
        constraints=[ConstraintValue("limit", 1 if status == CaseStatus.INFEASIBLE else -1)])
    if status == CaseStatus.OBJECTIVE_ERROR:
        case.objectives[0].value = float("nan")
    if status == CaseStatus.CONSTRAINT_ERROR:
        case.constraints = []
    observation = observe_case(case, Candidate("a", (0,), 0), p)
    observation.validate(p)
    expected = {CaseStatus.INFEASIBLE: "infeasible", CaseStatus.SIM_FAILED: "simulation_failed",
                CaseStatus.OBJECTIVE_ERROR: "metrics_unavailable", CaseStatus.CONSTRAINT_ERROR: "metrics_unavailable",
                CaseStatus.SUCCESS: "feasible"}[status]
    assert observation.status == expected
    case.impact_outputs["_lifecycle"] = {"execution_state": "unknown"}
    assert observe_case(case, Candidate("a", (0,), 0), p).status == "unknown"


class PopulationSchedule:
    """Scheduler test double only; it does not prove a library algorithm works."""
    def __init__(self, p):
        self.problem = p
        self.spec = AlgorithmSpec("NSGA-II", "pymoo", {"pop_size": 4, "generations": 3})
        self.stop_reason = None
        self.history = []
        self.told = 0
    def stop(self, reason): self.stop_reason = reason
    def ask(self, remaining):
        if remaining < 4:
            self.stop("insufficient_budget_for_population")
            return []
        return [Candidate(f"{self.told}:{i}", (0.,), self.told) for i in range(4)]
    def tell(self, observations): self.told += 1


def test_partial_budget_duplicates_and_invalid_population_are_not_fabricated():
    a = PopulationSchedule(problem(True, budget=6))
    calls = []
    result = run_optimizer(a, lambda c: calls.append(c) or Observation(c.id, "feasible", {"cost": 0, "energy": 1}), remaining=lambda: 6-len(calls))
    assert len(calls) == 4 and a.told == 1
    assert result.stop_reason == "insufficient_budget_for_population"
    assert result.evaluations_used == 4 and len(result.records) == 4
    b = PopulationSchedule(problem(True, budget=6))
    failed = run_optimizer(b, lambda c: Observation(c.id, "simulation_failed"), remaining=lambda: 6)
    assert failed.evaluations_used == 1 and b.told == 0
    assert failed.stop_reason == "invalid_population_observation"
    assert failed.to_summary()["no_feasible_solution"]


def test_faults_unknown_and_authorization_cannot_become_fitness_or_retries():
    from src.agents.actions.authorization import ExecutionBoundaryError
    a = create_optimizer(problem(), AlgorithmSpec("random", "python", seed=1))
    def fail(c): raise ExecutionBoundaryError("revoked")
    with pytest.raises(ExecutionBoundaryError):
        run_optimizer(a, fail, remaining=lambda: 10)
    b = create_optimizer(problem(), AlgorithmSpec("random", "python", seed=1))
    result = run_optimizer(b, lambda c: Observation(c.id, "unknown"), remaining=lambda: 10)
    assert result.evaluations_used == 1 and result.stop_reason == "execution_unknown"
    assert result.to_summary()["best_feasible"] is None
    c = PopulationSchedule(problem(True))
    def fault(*args): raise RuntimeError("backend broke")
    c.ask = fault
    assert "optimizer_fault" in run_optimizer(c, lambda c: pytest.fail("ran"), remaining=lambda: 10).stop_reason


def write_config(tmp_path, kind="pso", budget=6):
    model = tmp_path / "fake.bkp"
    model.write_bytes(b"fake analytic test only")
    raw = dict(simulator={"filepath": str(model)},
        design_variables=[dict(name="T", aspen_path=P, type="continuous", lower_bound=-2, upper_bound=2)],
        objectives=[dict(name="cost", type="aspen_path", aspen_path=O, minimize=False, unit="USD")],
        optimizer=dict(type=kind, backend="pymoo", parameters=dict(pop_size=4, generations=3), random_seed=7),
        execution=dict(max_evaluations=budget, max_failures=6, max_seconds=30))
    if kind == "nsga2":
        raw["objectives"].append(dict(name="energy", type="aspen_path", aspen_path=O, minimize=True, unit="kW"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path, raw


def test_configuration_approval_identity_and_population_authorization(tmp_path):
    from src.agents.actions.plans import build_optimization_plan, validate_optimization_plan
    from src.agents.actions.authorization import ExecutionBoundaryError
    from src.utils.file_io import load_optimize_config
    from src.workflows.optimize_population import optimize_population
    path, raw = write_config(tmp_path)
    config, _, _ = load_optimize_config(path)
    plan, schema = build_optimization_plan(path)
    assert plan["algorithm"]["algorithm"] == "PSO"
    assert plan["algorithm"]["parameters"]["w"] == .9
    assert plan["algorithm"]["problem"]["max_evaluations"] == 6
    assert plan["algorithm"]["versions"]["pymoo"] == capabilities("PSO")["versions"]["pymoo"]
    with pytest.raises(ExecutionBoundaryError):
        optimize_population(object(), config)
    raw["optimizer"]["parameters"]["w"] = .7
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExecutionBoundaryError, match="changed"):
        validate_optimization_plan(plan, schema)


def test_gateway_population_uses_existing_ledger_and_evidence(tmp_path, monkeypatch):
    """Fake candidate scheduler, real Gateway/ledger/run_case reservation boundary."""
    from src.agents.actions import ActionGateway, ActionRequest
    from src.agents.actions.plans import build_optimization_plan
    from src.agents.actions.authorization import evaluated_case, check_run
    from src.database.simulation_db import SimulationDB
    path, raw = write_config(tmp_path, "nsga2", budget=6)
    plan, schema = build_optimization_plan(path)
    calls = []
    class Driver:
        simulated = True
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def open(self, copy):
            assert copy != tmp_path / "fake.bkp" and copy.read_bytes() == b"fake analytic test only"
        def disconnect(self): pass
    fake_module = ModuleType("src.aspen_driver.driver")
    fake_module.AspenDriver = Driver
    monkeypatch.setitem(sys.modules, "src.aspen_driver.driver", fake_module)
    def evaluate(driver, inputs, config, **kwargs):
        check_run()
        calls.append(inputs)
        return ProcessCase(status=CaseStatus.SUCCESS, design_vars=inputs,
            objectives=[ObjectiveValue("cost", 1, "USD", False), ObjectiveValue("energy", 2, "kW", True)])
    monkeypatch.setattr("src.workflows.run_case.run_case", evaluated_case(evaluate))
    # Explicit scheduler injection proves the shared execution contract only.
    monkeypatch.setattr("src.agents.actions.plans.preflight_optimizer", lambda c: None)
    monkeypatch.setattr("src.workflows.optimize_population.create_optimizer", lambda p, s: PopulationSchedule(p))
    with SimulationDB(tmp_path / "sim.db") as db:
        gateway = ActionGateway(schema, simulation_db=db)
        action = ActionRequest("run_optimization", "model", reason="offline candidate scheduling test",
                               payload={"optimization_plan": plan})
        validation = gateway.submit(action)
        assert validation.valid, validation.errors
        gateway.preview(action.audit_id)
        gateway.approve(action.audit_id)
        output = gateway.execute(action.audit_id)
        assert len(calls) == output["evaluations_used"] == 4
        assert output["summary"]["stop_reason"] == "insufficient_budget_for_population"
        assert len(db.get_execution_evaluations(action.audit_id)) == 4
        assert output["execution_evidence"]["optimization"]["evaluations_used"] == 4
        assert all(c["execution_algorithm"] == plan["algorithm"] for c in output["simulation_cases"])
        assert db.get_action_audit(action.audit_id)["result"]["algorithm"] == plan["algorithm"]
        assert output["simulated"] and not output["real_execution"]
        with pytest.raises(PermissionError):
            gateway.execute(action.audit_id)
    assert (tmp_path / "fake.bkp").read_bytes() == b"fake analytic test only"


def test_population_library_adapter_pre_budget_fault_and_empty_stop_without_substitution():
    """Boundary unit tests only; adapter construction is bypassed explicitly."""
    a = PymooOptimizer.__new__(PymooOptimizer)
    a.problem = problem(True)
    a.spec = AlgorithmSpec("NSGA-II", "pymoo", resolve_parameters("NSGA-II", {"pop_size": 4}))
    a.parameters = a.spec.parameters
    a.stop_reason, a.pending, a.history, a.generation = None, [], [], 0
    class Backend:
        def ask(self): raise RuntimeError("library failed")
    a._algorithm = Backend()
    assert a.ask(3) == [] and a.stop_reason == "insufficient_budget_for_population"
    a.stop_reason = None
    with pytest.raises(RuntimeError, match="pymoo ask failed"):
        a.ask(4)
    assert "ask_failed" in a.stop_reason
    class Empty:
        def ask(self): return []
    a._algorithm, a.stop_reason = Empty(), None
    assert a.ask(4) == [] and a.stop_reason == "candidate_space_exhausted"
    with pytest.raises(ValueError, match="dynamic"):
        a.ask(4, [(-1, 1)])


def test_agent_can_query_registry_without_selected_yaml():
    from src.agents.subagents.registry import default_registry
    from src.agents.subagents.tool_runtime import _list_optimizers
    r = default_registry()
    assert "list_optimizers" in r.definition("optimization_advisor").tool_allowlist
    tool = next(t for t in r.tools.catalogue({}, role="optimization_advisor") if t["id"] == "list_optimizers")
    assert tool["available"] and not tool["side_effect"]
    assert _list_optimizers({})["selection_required"]


def test_real_botorch_constrained_nehvi_preserves_acquisition_identity(monkeypatch):
    pytest.importorskip("botorch")
    from src.optimization.botorch_backend import BoTorchMOOptimizer
    from src.optimization.surrogate import SurrogateConfig
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    opt = BoTorchMOOptimizer([(0, 1)], 2, set(), SurrogateConfig(model="NEHVI"))
    for x, margin in [(.1, -.2), (.3, -.1), (.5, .1), (.7, .2)]:
        opt.tell([x], 0, is_success=margin >= 0, y_vec=[x+1, 2-x], c_vec={"limit": margin})
    # Fit and acquisition construction are real. Acquisition maximization is
    # outside this regression's scope; inspect what reaches the optimizer.
    def inspect(acqf, *args):
        assert isinstance(acqf, qLogNoisyExpectedHypervolumeImprovement)
        return [.55]
    monkeypatch.setattr(opt, "_optimize_and_round", inspect)
    assert opt.ask() == [.55]
    from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig, MultiObjectiveBayesianOptimizer
    cfg = ParetoOptimizeCaseConfig({P: (-2, 2)}, ["a", "b"], surrogate_model="NEHVI")
    with pytest.raises(ValueError, match="resume unsupported"):
        MultiObjectiveBayesianOptimizer([(-2, 2)], cfg, [P]).restore_rng({"outer": []})


def test_candidate_exhaustion_and_out_of_bounds_stop_without_evaluation():
    a = PopulationSchedule(problem(True))
    a.ask = lambda remaining: [Candidate("bad", (100.,), 0)]
    result = run_optimizer(a, lambda c: pytest.fail("illegal candidate evaluated"), remaining=lambda: 10)
    assert result.evaluations_used == 0 and result.stop_reason.startswith("illegal_candidate")
    b = PopulationSchedule(problem(True))
    b.ask = lambda remaining: []
    assert run_optimizer(b, lambda c: pytest.fail("empty candidate evaluated"), remaining=lambda: 10).stop_reason == "candidate_space_exhausted"


def test_bo_rejects_explicit_parameter_backend_identity_mismatch():
    with pytest.raises(ValueError, match="identity invalid"):
        create_optimizer(problem(), AlgorithmSpec("BO", "skopt", {"surrogate_model": "random"}))


def test_report_feasible_only_mixed_directions_and_registry():
    p = problem(True, True)
    result = UnifiedResult(identity(p, AlgorithmSpec("NSGA-II", "pymoo")))
    for i, (values, residual) in enumerate([({"cost": 4, "energy": 2}, -1),
                                           ({"cost": 3, "energy": 3}, -1),
                                           ({"cost": 100, "energy": 0}, 1)]):
        o = Observation(str(i), "infeasible" if residual > 0 else "feasible", values, {"limit": residual}, str(i))
        result.records.append({"observation": o.__dict__})
    summary = result.to_summary()
    assert [r["observation"]["evidence_id"] for r in summary["feasible_pareto"]] == ["0"]
    assert summary["best_feasible"] is None
    rows = {r["id"]: r for r in algorithm_catalogue()}
    assert rows["PSO"]["objectives"] == "single"
    assert rows["NSGA-II"]["resume"] == "disabled"
    assert rows["GA"]["backend"] == "pymoo" and rows["GA"]["objectives"] == "single"
    assert rows["GWO"]["status"] == rows["NSGA-III"]["status"] == "not_connected"


@pytest.mark.parametrize("algorithm", ["PSO", "NSGA-II", "GA"])
def test_real_pymoo_analytic_generation_flow(algorithm):
    pymoo = pytest.importorskip("pymoo", reason="offline environment has no pymoo distribution; real algorithm NOT verified")
    assert pymoo.__version__ == "0.6.1.5"
    p = problem(algorithm == "NSGA-II", algorithm == "NSGA-II", budget=13)
    params = resolve_parameters(algorithm, {"pop_size": 4, "generations": 3})
    opt = create_optimizer(p, AlgorithmSpec(algorithm, "pymoo", params, 42))
    calls = []
    def analytic(c):
        x = c.x[0]
        calls.append(c)
        values = {"cost": -x*x}
        residuals = {}
        if algorithm == "NSGA-II":
            values["energy"] = (x-1)**2
            residuals["limit"] = x-.5
        return Observation(c.id, "feasible" if not residuals or x <= .5 else "infeasible", values, residuals, c.id)
    result = run_optimizer(opt, analytic, remaining=lambda: 13-len(calls))
    assert 8 <= len(calls) <= 12
    assert opt.generation == 3 and result.stop_reason == "generations_completed"
    assert all(-2 <= c.x[0] <= 2 for c in calls)
    assert len({c.generation for c in calls}) == 3
    assert len(result.records) == len(calls)
    assert all(r["observation"]["objectives"]["cost"] <= 0 for r in result.records)
    assert opt.state_dict()["resumable"] is False
    with pytest.raises(ValueError, match="resume disabled"):
        opt.load_state_dict(opt.state_dict())
    with pytest.raises(RuntimeError, match="callbacks disabled"):
        opt._problem.evaluate([[0.]])
