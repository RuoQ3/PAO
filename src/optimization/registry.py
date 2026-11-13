"""Single source for algorithm identities, dependencies and supported combinations."""
from __future__ import annotations

from dataclasses import asdict
from importlib import metadata, util

from .problem import AlgorithmSpec


def dependency_versions(backend):
    packages = {"skopt": ("scikit-optimize", "numpy", "scipy", "scikit-learn"),
                "botorch": ("botorch", "torch", "gpytorch", "numpy"),
                "pymoo": ("pymoo", "numpy", "scipy"), "python": ()}
    if backend not in packages:
        raise ValueError(f"unsupported backend: {backend}")
    import platform
    versions = {"python": platform.python_version()}
    for name in packages[backend]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capabilities(algorithm):
    if algorithm not in {"BO", "ParEGO", "qEHVI", "NEHVI", "random", "PSO", "NSGA-II", "GA"}:
        return dict(id=algorithm, available=False, status="not_connected", dependency="not implemented")
    backend = ("pymoo" if algorithm in {"PSO", "NSGA-II", "GA"} else
               "botorch" if algorithm in {"qEHVI", "NEHVI"} else
               "python" if algorithm == "random" else "skopt")
    modules = {"skopt": ("skopt",), "python": (), "botorch": ("torch", "botorch", "gpytorch"), "pymoo": ("pymoo",)}
    versions = dependency_versions(backend)
    ready = all(v is not None for v in versions.values()) and all(util.find_spec(m) is not None for m in modules[backend])
    # Pin the target API; actual imports and license are checked at preflight.
    if backend == "pymoo":
        ready = ready and versions.get("pymoo") == "0.6.1.5"
    return dict(id=algorithm, backend=backend, available=ready,
        status="connected" if ready else "unavailable", dependency=backend, versions=versions,
        objectives="single" if algorithm in {"BO", "PSO", "GA"} else "single_or_multi" if algorithm == "random" else "multi",
        variable_types=["continuous", "integer"] if algorithm == "GA" else ["continuous"] if backend in {"pymoo", "botorch"} else ["continuous", "integer"],
        constraints="none" if algorithm == "PSO" else "feasibility_first_scaled_residuals" if algorithm in {"NSGA-II", "GA"} else
                    "feasible_observations_only" if algorithm in {"BO", "ParEGO"} else "constraint_surrogate" if backend == "botorch" else "record_only",
        evaluation="population_sequential" if backend == "pymoo" else "point_sequential",
        partial_batch=False, resume="exact_json" if algorithm == "random" else "disabled",
        dynamic_region=backend != "pymoo",
        failure_policy="stop_batch_without_synthetic_fitness" if backend == "pymoo" else "typed_skip_invalid; legacy_internal_penalty",
        dependency_verification="runtime_import_and_license_check_required" if backend == "pymoo" else "runtime_preflight_required",
        limitations=(["dependency and license require local distribution verification",
                      "derived/dynamic region unsupported" if algorithm == "GA" else "integer/derived/dynamic region unsupported", "resume disabled"]
                     if backend == "pymoo" else ["backend faults stop; no algorithm substitution"]))


def algorithm_catalogue():
    rows = [capabilities(n) for n in ("BO", "ParEGO", "qEHVI", "NEHVI", "random", "PSO", "NSGA-II", "GA", "GWO", "NSGA-III")]
    # Preserve the user-facing surrogate selections of earlier configuration files.
    for name in ("GP", "RF", "ET", "GBRT"):
        rows.append({**capabilities("BO"), "id": name, "strategy": "BO or ParEGO", "surrogate": name})
    return rows


def validate_combination(problem, spec, *, require_available=True):
    cap = capabilities(spec.algorithm)
    if cap["status"] == "not_connected":
        raise ValueError(f"algorithm not connected: {spec.algorithm}; no substitution")
    if spec.backend != cap["backend"]:
        raise ValueError("algorithm/backend identity mismatch")
    n = len(problem.objectives)
    if cap["objectives"] == "single" and n != 1 or cap["objectives"] == "multi" and n < 2:
        raise ValueError("unsupported number of objectives")
    if any(v.kind not in cap["variable_types"] for v in problem.variables):
        raise ValueError("unsupported variable type for selected algorithm")
    if problem.constraints and cap["constraints"] == "none":
        raise ValueError("constraints unsupported for selected algorithm")
    if problem.derived:
        raise ValueError("derived variables require an approved physical write envelope")
    if require_available and not cap["available"]:
        raise RuntimeError(f"backend unavailable for {spec.algorithm}: {cap['dependency']} {cap['versions']}")
    return cap


def identity(problem, spec):
    import json
    import hashlib
    from pathlib import Path
    directory = Path(__file__).parent
    files = [directory / name for name in ("registry.py", "population.py", "bo_adapter.py", "problem.py", "execution.py",
                                          "configuration.py", "surrogate.py", "botorch_backend.py")]
    files += [directory.parent / "workflows" / name for name in ("optimize_case.py", "optimize_pareto_case.py", "optimize_population.py")]
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return json.loads(json.dumps({**asdict(spec), "requested_algorithm": spec.algorithm, "actual_algorithm": spec.algorithm,
            "requested_backend": spec.backend, "actual_backend": spec.backend,
            "versions": dependency_versions(spec.backend), "implementation_hashes": implementation,
            "problem_hash": problem.fingerprint, "problem": asdict(problem),
            "capabilities": capabilities(spec.algorithm), "execution_policy": {
                "partial_population": "stop_before_generation", "parallel_aspen": False,
                "duplicates": "backend_strategy_only; repeated population members consume budget",
                "invalid_observation": "stop_population; no synthetic fitness" if spec.backend == "pymoo" else
                    "legacy BO explicit internal penalty; typed protocol skips invalid targets",
                "bounds": "reject_illegal_candidates", "resume_execution": "disabled",
                "backend_policies": ({"sampling": "FloatRandomSampling", "adaptive": False, "perturb_best": False,
                    "boundary_handling": "pymoo PSO native bounded position/velocity strategy"} if spec.algorithm == "PSO" else
                    {"sampling": "FloatRandomSampling", "crossover": "SBX", "mutation": "PM", "mutation_individual_probability": 1.0,
                     "eliminate_duplicates": True, "boundary_handling": "pymoo SBX/PM native bounded operators",
                     "integer_repair": "numpy.rint in sampling and SBX/PM" if spec.algorithm == "GA" else "unsupported"}
                    if spec.algorithm in {"NSGA-II", "GA"} else {})}}))


def create_optimizer(problem, spec, *, legacy_config=None):
    validate_combination(problem, spec)
    if legacy_config is not None:
        expected_problem, expected_spec = from_config(legacy_config)
        if expected_problem.fingerprint != problem.fingerprint or expected_spec != spec:
            raise ValueError("legacy strategy configuration differs from requested problem/algorithm identity")
    if spec.backend == "pymoo":
        from .population import PymooOptimizer
        return PymooOptimizer(problem, spec)
    from .bo_adapter import BOAdapter
    return BOAdapter(problem, spec, legacy_config=legacy_config)


def from_config(config):
    """Resolved identity for existing Python config objects and population configs."""
    if hasattr(config, "problem"):
        return config.problem, config.algorithm_spec
    from .problem import OptimizationProblem, Variable, Objective, Constraint
    objectives = getattr(config, "objective_definitions", None)
    names = getattr(config, "objective_names", [getattr(config, "objective_name", "")])
    if objectives is None:
        objectives = [dict(name=n, minimize=getattr(config, "minimize", True), unit="") for n in names]
    model = config.surrogate_model
    algorithm = model if model in {"qEHVI", "NEHVI", "random"} else "ParEGO" if len(names) > 1 else "BO"
    backend = "botorch" if algorithm in {"qEHVI", "NEHVI"} else "python" if model == "random" else "skopt"
    definitions = {v.get("aspen_path"): v for v in getattr(config, "problem_identity", {}).get("design_variables", [])}
    problem = OptimizationProblem(tuple(Variable(definitions.get(p, {}).get("id", definitions.get(p, {}).get("name", p)),
        p, *b, "integer" if p in config.integer_var_paths else "continuous", definitions.get(p, {}).get("unit", ""))
        for p, b in config.param_bounds.items()), tuple(Objective(o["name"], o.get("minimize", True), o.get("unit", "")) for o in objectives),
        constraints=tuple(Constraint(c["name"], c, c.get("scale", 1.0)) for c in getattr(config, "constraint_definitions", [])),
        fixed=config.fixed_vars, derived=tuple(config.derived_var_specs),
        max_evaluations=getattr(config, "total_evaluation_budget", config.n_iterations),
        authorization_scope={**config.param_bounds, **{p: (v, v) for p, v in config.fixed_vars.items()}},
        identity=getattr(config, "problem_identity", {}))
    params = {k: getattr(config, k) for k in ("acquisition", "xi", "kappa", "n_initial_min", "n_initial", "n_iterations")}
    params["surrogate_model"] = model
    if hasattr(config, "scalarization"):
        params["scalarization"] = config.scalarization
    return problem, AlgorithmSpec(algorithm, backend, params, config.random_seed)
