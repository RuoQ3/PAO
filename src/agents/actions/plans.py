"""Static plans and validation. Never opens an Aspen document."""
from __future__ import annotations

import math
from pathlib import Path

from .authorization import ExecutionBoundaryError, digest, file_digest, validate_inputs
from .models import ModelIdentity, ModelSchema, ParameterSpec


def simulation_scope(plan, schema):
    plan.validate_execution()
    if not plan.executable or not plan.base_model or not Path(plan.base_model).is_file():
        raise ValueError("simulation plan requires an available base model")
    if not schema.model_path or Path(schema.model_path).resolve() != Path(plan.base_model).resolve():
        raise ValueError("simulation model does not match ModelSchema model_path")
    if not plan.parameter_changes or not plan.expected_outputs:
        raise ValueError("simulation plan requires explicit cases, inputs and expected outputs")
    from .validation import _SAFE_PATH
    if any(not _SAFE_PATH.match(p) or "\\output\\" not in p.casefold() for p in plan.expected_outputs):
        raise ValueError("simulation expected outputs must be explicit Aspen Output paths")
    if len(plan.parameter_changes) > plan.max_evaluations:
        raise ValueError("planned cases exceed total evaluation budget")
    scope = {}
    for p in plan.adaptive_policy.get("parameters", []):
        path = p.get("aspen_path") or p.get("path")
        scope[path] = (p.get("lower"), p.get("upper"))
    for change in plan.parameter_changes:
        inputs = change.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("each case requires explicit non-empty inputs")
        for path, value in inputs.items():
            if not plan.adaptive_policy.get("parameters"):
                lo, hi = scope.get(path, (value, value))
                scope[path] = (min(lo, value), max(hi, value))
    validate_scope(scope, schema)
    ids = [c.get("case_id") for c in plan.parameter_changes]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("simulation case IDs must be present and unique")
    for change in plan.parameter_changes:
        validate_inputs(change["inputs"], schema, scope)
    return scope


def validate_scope(scope, schema):
    if not scope:
        raise ValueError("execution scope is empty")
    for path, interval in scope.items():
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in interval)
                or interval[0] > interval[1]):
            raise ValueError(f"invalid approved bounds: {path}")
        # Validate endpoints independently of candidate-level dependencies.
        from copy import deepcopy
        standalone = deepcopy(schema)
        standalone.dependencies = {}
        validate_inputs({path: interval[0]}, standalone, scope)
        validate_inputs({path: interval[1]}, standalone, scope)


def build_optimization_plan(config_path):
    """Bind the complete YAML and model contents, separately from iteration count."""
    import yaml
    from src.utils.file_io import load_optimize_config
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    objective_modules = {str(Path(obj["module"]).resolve()): file_digest(obj["module"])
                         for obj in raw.get("objectives", []) if obj.get("type") == "custom_module"}
    config, model, _ = load_optimize_config(path)
    if getattr(config, "agent_loop", {}).get("enabled"):
        raise ValueError("legacy Agent closed-loop execution is disabled; no approved envelope adapter")
    if config.derived_var_specs or config.run_config.recycle_warmstart is not None:
        raise ValueError("derived variables/recycle warmstart need an explicit physical write envelope; execution disabled")
    limits = raw.get("execution") or {}
    for key in ("max_evaluations", "max_failures"):
        if type(limits.get(key)) is not int or limits[key] < 1:
            raise ValueError(f"execution.{key} must be an explicit positive integer")
    if type(limits.get("max_seconds")) not in (int, float) or not math.isfinite(limits["max_seconds"]) or limits["max_seconds"] <= 0:
        raise ValueError("execution.max_seconds must be explicit, finite and positive")
    schema = ModelSchema(ModelIdentity(str(model.resolve())), model_path=str(model.resolve()),
                         config_path=str(path), targets=("model",))
    schema.dependencies = raw.get("var_dependencies") or {}
    scope = {}
    for dv in raw.get("design_variables", []):
        p = dv["aspen_path"]
        integer = dv.get("type") == "integer"
        fixed = dv.get("type") == "fixed"
        lo = dv.get("initial_value") if fixed else dv.get("lower_bound")
        hi = lo if fixed else dv.get("upper_bound")
        schema.parameters[p] = ParameterSpec(p, integer=integer, value_type="int" if integer else "float",
                                             minimum=lo, maximum=hi, unit=dv.get("unit"))
        scope[p] = (lo, hi)
    validate_scope(scope, schema)
    plan = {"config_path": str(path), "config_hash": file_digest(path),
            "model_path": str(model.resolve()), "model_hash": file_digest(model),
            "objective_module_hashes": objective_modules,
            "configuration": raw, "scope": scope, "limits": limits,
            "algorithm": {"type": raw.get("optimizer", {}).get("type", "bayesian"),
                          "surrogate_model": config.surrogate_model,
                          "acquisition": config.acquisition, "xi": config.xi, "kappa": config.kappa,
                          "n_initial": config.n_initial, "n_initial_min": config.n_initial_min,
                          "n_iterations": config.n_iterations, "random_seed": config.random_seed,
                          "scalarization": getattr(config, "scalarization", None)},
            "policy": {"all_phases_share_budget": True, "automatic_replay": False,
                       "feasibility_search": raw.get("feasibility_search"),
                       "sensitivity_probe": raw.get("sensitivity_probe"),
                       "retries": "each attempt consumes budget", "verification": "each run consumes budget"}}
    return plan, schema


def validate_optimization_plan(plan, schema):
    current, _ = build_optimization_plan(plan["config_path"])
    if digest(current) != digest(plan):
        raise ExecutionBoundaryError("optimization model/configuration/algorithm changed; preview and approve again")
    validate_scope(plan["scope"], schema)
    return plan["scope"]


def preflight_optimizer(config):
    """Initialize the requested backend before any model open, DOE or probe."""
    from src.optimization.surrogate import SurrogateConfig, make_surrogate_optimizer
    from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig, _MultiObjectiveBayesianOptimizer
    ff = config.feasibility_filter
    if ff.enabled and ff.model != "random":
        from src.optimization.feasibility import FeasibilityClassifier, _HAS_SKLEARN
        if not _HAS_SKLEARN:
            raise RuntimeError("feasibility filter backend unavailable: scikit-learn required")
        FeasibilityClassifier(ff)._build_model()
    if isinstance(config, ParetoOptimizeCaseConfig):
        from src.workflows.optimize_pareto_case import _validate_config
        _validate_config(config)
        return _MultiObjectiveBayesianOptimizer(list(config.param_bounds.values()), config, list(config.param_bounds))
    from src.workflows.optimize_case import _validate_config
    _validate_config(config)
    if len(config.run_config.objective_fns) != 1:
        raise ValueError("single-objective optimizer requires exactly one objective")
    return make_surrogate_optimizer(list(config.param_bounds.values()), SurrogateConfig(
        model=config.surrogate_model, acquisition=config.acquisition, xi=config.xi,
        kappa=config.kappa, n_initial_min=config.n_initial_min, random_seed=config.random_seed),
        {i for i, p in enumerate(config.param_bounds) if p in config.integer_var_paths})


def execute_optimization(plan):
    from src.utils.file_io import load_optimize_config
    config, model, kwargs = load_optimize_config(plan["config_path"])
    from dataclasses import asdict
    from .authorization import require_authorization
    grant = require_authorization()
    if grant.ledger is not None:
        config.db_path = grant.ledger._path
    if getattr(grant, "session_id", ""):
        config.session_id = grant.session_id
    grant.algorithm = plan["algorithm"]
    grant.expected_config = digest(asdict(config))
    grant.config_object = config
    preflight_optimizer(config)
    grant.check()
    # Every authorized optimization owns an isolated copy, never the source.
    import shutil
    import tempfile
    from src.aspen_driver.driver import AspenDriver
    from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig, optimize_pareto_case
    from src.workflows.optimize_case import optimize_case
    from src.aspen_driver.lifecycle import ExecutionUnit
    directory = Path(tempfile.mkdtemp(prefix="pao_optimization_"))
    copied = directory / model.name
    unit = ExecutionUnit(str(copied))
    try:
        shutil.copy2(model, copied)
        if file_digest(copied) != plan["model_hash"]:
            raise ExecutionBoundaryError("model changed while copying; new approval required")
    except Exception as primary:
        errors = []
        try:
            shutil.rmtree(directory)
        except OSError as cleanup:
            errors.append(str(cleanup))
        primary.resources = {"temporary_directory": str(directory), "cleanup_errors": errors,
            "files_state": "retained" if directory.exists() else "removed"}
        raise
    grant.execution_model = copied.resolve()
    grant.model_hash = plan["model_hash"]
    def execute():
        driver = AspenDriver(**kwargs)
        unit.driver = driver
        driver.stop_requested = unit.stop_requested
        driver.__enter__()
        driver.open(copied)
        result = optimize_pareto_case(driver, config) if isinstance(config, ParetoOptimizeCaseConfig) else optimize_case(driver, config)
        return {"status": "execution_completed", "simulated": bool(getattr(driver, "simulated", False)),
                "algorithm": plan["algorithm"], "summary": result.to_summary()}
    try:
        output = unit.invoke(execute, timeout=max(.001, grant.deadline - __import__("time").monotonic()), cleanup_timeout=5)
    except ExecutionBoundaryError as primary:
        grant.audit.record(grant.action_id, "execution", status="stopped", payload={"resources": unit.snapshot()})
        if unit.cleanup_state == "completed" and unit.execution_state != "unknown":
            try:
                shutil.rmtree(directory)
            except OSError as cleanup:
                unit.cleanup_errors.append(str(cleanup))
        primary.resources = {"unit": unit.snapshot(), "temporary_directory": str(directory),
                             "files_state": "retained" if directory.exists() else "removed"}
        raise
    except Exception as exc:
        output = {"status": "outcome_unknown" if unit.timed_out or unit.execution_state == "unknown" else "failed",
                  "error": str(exc), "algorithm": plan["algorithm"]}
    errors = []
    if unit.execution_state == "unknown" or unit.timed_out:
        output["status"] = "outcome_unknown"
    if not unit.timed_out and unit.execution_state != "unknown" and unit.cleanup_state == "completed":
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            errors.append(str(exc))
    if output.get("simulated"):
        output["status"] = "simulated"
    output["resources"] = {"unit": unit.snapshot(), "temporary_directory": str(directory),
        "files_state": "retained" if directory.exists() else "removed", "cleanup_errors": errors}
    return output


def single_simulation_plan(context, base_model):
    """One explicit configured operating point; never infers missing inputs."""
    from src.agents.orchestration.parameter_impact import SimulationPlan
    inputs = {}
    for row in (getattr(context, "variables", {}) or {}).values():
        path = row.get("aspen_path") or row.get("path")
        value = row.get("initial_value", row.get("baseline", row.get("value")))
        if not path or value is None:
            raise ValueError("单次仿真缺少配置的 Aspen 输入路径或明确基准值")
        inputs[path] = value
    outputs = []
    metric = None
    unit = None
    direction = None
    for row in (getattr(context, "objectives", []) or []):
        if isinstance(row, dict) and row.get("aspen_path"):
            outputs.append(row["aspen_path"])
            metric = metric or row.get("name") or row["aspen_path"]
            if len(outputs) == 1:
                unit = row.get("unit")
                direction = row.get("direction") or ("minimize" if row.get("minimize") is True else
                            "maximize" if row.get("minimize") is False else None)
    if not inputs or not outputs or not base_model:
        raise ValueError("单次仿真缺少模型文件、明确输入基准值或标量输出路径；请补全配置后重新生成计划")
    return SimulationPlan(base_model=str(base_model), parameter_changes=[{"case_id": "single", "inputs": inputs}],
                          expected_outputs=outputs, max_evaluations=1, reason="Run one configured operating point",
                          output_spec={"metric": metric, "extractor": "scalar", "aspen_paths": [outputs[0]],
                                       "unit": unit, "direction": direction},
                          adaptive_policy={"max_failures": 1, "max_seconds": 300,
                                           "max_reference_trials": 0, "max_shrinks": 0})
