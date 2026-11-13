"""Static configuration translation for population algorithms."""
from __future__ import annotations

from .problem import OptimizationProblem, Variable, Objective, Constraint, AlgorithmSpec
from .population import resolve_parameters
from .registry import validate_combination


def population_config(raw, run_config):
    from src.workflows.optimize_population import PopulationOptimizeConfig
    options = raw["optimizer"]
    if set(options) - {"type", "backend", "parameters", "random_seed"}:
        raise ValueError("population optimizer accepts only type/backend/parameters/random_seed; BO strategies cannot be applied")
    algorithm = {"pso": "PSO", "nsga2": "NSGA-II", "ga": "GA"}[options["type"]]
    if options.get("backend", "pymoo") != "pymoo":
        raise ValueError("population backend must be explicitly pymoo")
    for key in ("trust_region", "boundary_refine", "feasibility_filter", "feasibility_search", "sensitivity_probe",
                "search_region", "agent_loop", "var_dependencies", "early_stopping", "preflight"):
        value = raw.get(key)
        if value and (not isinstance(value, dict) or value.get("enabled", True)):
            raise ValueError(f"{key} unsupported for population optimization; no BO policy substitution")
    if run_config.recycle_warmstart is not None:
        raise ValueError("population recycle warmstart unsupported")
    variables, fixed, scope = [], {}, {}
    for dv in raw.get("design_variables", []):
        kind = dv.get("type", "continuous")
        if kind not in {"continuous", "integer", "fixed"}:
            raise ValueError("population supports continuous/integer and explicit fixed variables only")
        path = dv["aspen_path"]
        if path in scope:
            raise ValueError("duplicate variable path")
        if kind == "fixed":
            value = dv.get("initial_value")
            fixed[path] = value
            scope[path] = (value, value)
        else:
            lo, hi = dv["lower_bound"], dv["upper_bound"]
            scope[path] = (lo, hi)
            variables.append(Variable(dv.get("id", dv.get("name", path)), path, lo, hi, kind, dv.get("unit", "")))
    objectives = tuple(Objective(o["name"], o.get("minimize", True), o.get("unit", "")) for o in raw.get("objectives", []))
    constraints = []
    for c in raw.get("constraints", []):
        # Preserve the existing residual convention, but reject strict/equality
        # operators rather than quietly changing their semantics or tolerance.
        if c.get("operator") not in {"<=", ">="}:
            raise ValueError("population constraint operator must be <= or >=; equality/strict tolerances unsupported")
        if "scale" not in c:
            raise ValueError("population constraints require explicit engineering residual scale")
        constraints.append(Constraint(c["name"], dict(c), c["scale"]))
    limits = raw.get("execution") or {}
    problem = OptimizationProblem(tuple(variables), objectives, tuple(constraints), fixed=fixed,
        authorization_scope=scope, max_evaluations=limits.get("max_evaluations", 30),
        identity={"objective_definitions": raw.get("objectives", []), "simulator": raw.get("simulator", {})})
    parameters = resolve_parameters(algorithm, options.get("parameters", {}))
    spec = AlgorithmSpec(algorithm, "pymoo", parameters, options.get("random_seed"))
    validate_combination(problem, spec, require_available=False)
    return PopulationOptimizeConfig(problem, spec, run_config,
        {v.path: (v.lower, v.upper) for v in variables}, fixed, [], set(),
        parameters["pop_size"], parameters["pop_size"] * parameters["generations"])
