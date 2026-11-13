"""Public BO bridge, preserving legacy strategies and their point API."""
from __future__ import annotations

from dataclasses import asdict
import random

from .problem import Candidate
from .registry import capabilities, identity


class BOAdapter:
    def __init__(self, problem, spec, *, legacy_config=None):
        from .registry import validate_combination
        validate_combination(problem, spec)
        if spec.algorithm == "BO":
            allowed = {"surrogate_model", "acquisition", "xi", "kappa", "n_initial_min", "n_initial", "n_iterations"}
            if set(spec.parameters) - allowed or spec.parameters.get("surrogate_model", "GP") not in {"GP", "RF", "ET", "GBRT"}:
                raise ValueError("BO parameter/model identity invalid; no backend substitution")
        if spec.algorithm == "random" and spec.parameters.get("surrogate_model", "random") != "random":
            raise ValueError("random parameter/model identity invalid")
        self.problem, self.spec = problem, spec
        self.stop_reason = None
        self.pending = []
        self.history = []
        self._rng = random.Random(spec.seed)
        self._backend = None
        if spec.algorithm != "random":
            bounds = [(v.lower, v.upper) for v in problem.variables]
            if len(problem.objectives) > 1:
                if legacy_config is None:
                    raise ValueError("ParEGO/BoTorch require explicit legacy strategy configuration")
                from src.workflows.optimize_pareto_case import MultiObjectiveBayesianOptimizer
                self._backend = MultiObjectiveBayesianOptimizer(bounds, legacy_config, [v.path for v in problem.variables])
            else:
                from .surrogate import SurrogateConfig, make_surrogate_optimizer
                p = spec.parameters
                self._backend = make_surrogate_optimizer(bounds, SurrogateConfig(
                    model=p.get("surrogate_model", "GP"), acquisition=p.get("acquisition", "EI"),
                    xi=p.get("xi", .01), kappa=p.get("kappa", 1.96),
                    n_initial_min=p.get("n_initial_min", 3), random_seed=spec.seed),
                    {i for i, v in enumerate(problem.variables) if v.kind == "integer"})

    def capabilities(self):
        return capabilities(self.spec.algorithm)

    def stop(self, reason):
        self.stop_reason = reason

    def ask(self, remaining, region=None):
        if self.pending:
            raise RuntimeError("feedback required")
        if self.stop_reason:
            return []
        if remaining < 1:
            self.stop("budget_exhausted")
            return []
        if region is not None and (len(region) != len(self.problem.variables) or any(
                not v.lower <= lo <= hi <= v.upper for v, (lo, hi) in zip(self.problem.variables, region))):
            raise ValueError("search region outside hard bounds")
        try:
            if self._backend is None:
                bounds = region or [(v.lower, v.upper) for v in self.problem.variables]
                import math
                x = [float(self._rng.randint(math.ceil(lo), math.floor(hi))) if v.kind == "integer" else self._rng.uniform(lo, hi)
                     for v, (lo, hi) in zip(self.problem.variables, bounds)]
            elif len(self.problem.objectives) > 1:
                if region is not None:
                    self._backend.set_effective_bounds(region)
                x = self._backend.ask()
            else:
                x = self._backend.ask(active_bounds=region)
            self.problem.inputs(x)
            self.pending = [Candidate(str(len(self.history)), tuple(float(v) for v in x), len(self.history))]
            return list(self.pending)
        except Exception as exc:
            self.stop(f"ask_failed: {exc}")
            raise

    def tell(self, observations):
        if not self.pending or [o.candidate_id for o in observations] != [c.id for c in self.pending]:
            raise ValueError("feedback differs from pending candidates")
        o, c = observations[0], self.pending[0]
        o.validate(self.problem)
        self.history.append({"candidate": asdict(c), "observation": asdict(o)})
        try:
            # Unified policy: invalid outputs are never synthetic GP targets.
            # ParEGO/BO learn only feasible observations; constrained BoTorch may
            # also learn trustworthy infeasible values and engineering margins.
            valid = o.status == "feasible" or (self.spec.backend == "botorch" and o.status == "infeasible")
            if self._backend is not None and valid:
                y = self.problem.internal_objectives(o.objectives)
                if len(y) > 1:
                    self._backend.tell(list(c.x), y, is_success=o.status == "feasible",
                        c_vec={k: -v for k, v in o.residuals.items()} or None)
                else:
                    self._backend.tell(list(c.x), y[0], is_success=True)
            if o.status in {"unknown", "timeout"}:
                self.stop("execution_unknown")
            self.pending = []
        except Exception as exc:
            self.stop(f"tell_failed: {exc}")
            raise

    def state_dict(self):
        return dict(format=1, identity=identity(self.problem, self.spec), history=self.history,
                    pending=[asdict(c) for c in self.pending], rng=self._rng.getstate(),
                    stop_reason=self.stop_reason, resumable=self.spec.algorithm == "random")

    def load_state_dict(self, state):
        if self.spec.algorithm != "random":
            raise ValueError("BO backend exact resume disabled; observation replay is not exact continuation")
        if state.get("format") != 1 or state.get("identity") != identity(self.problem, self.spec):
            raise ValueError("checkpoint problem/configuration/backend/dependency mismatch")
        if state.get("pending") or any(r["observation"]["status"] in {"unknown", "timeout"} for r in state["history"]):
            raise ValueError("pending/unknown evaluations cannot be automatically replayed")
        if len(state["history"]) > self.problem.max_evaluations:
            raise ValueError("checkpoint exceeds total evaluation budget")
        from .problem import Observation
        for row in state["history"]:
            self.problem.inputs(row["candidate"]["x"])
            Observation(**row["observation"]).validate(self.problem)
        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, (list, tuple)) else x
        self._rng.setstate(tuples(state["rng"]))
        self.history = list(state["history"])
        self.stop_reason = state["stop_reason"]


def make_legacy_bo(config):
    """Existing workflows retain DOE, filtering, penalty and trust-region policy.

    This bridge intentionally exposes their historical ask/tell semantics. New
    protocol users use BOAdapter's typed candidate/observation methods instead.
    """
    from .registry import from_config, validate_combination
    problem, spec = from_config(config)
    validate_combination(problem, spec)
    if len(problem.objectives) > 1:
        from src.workflows.optimize_pareto_case import MultiObjectiveBayesianOptimizer
        backend = MultiObjectiveBayesianOptimizer(list(config.param_bounds.values()), config, list(config.param_bounds))
        return LegacyBOBridge(problem, spec, backend)
    from .surrogate import SurrogateConfig, make_surrogate_optimizer
    backend = make_surrogate_optimizer(list(config.param_bounds.values()), SurrogateConfig(
        model=config.surrogate_model, acquisition=config.acquisition, xi=config.xi, kappa=config.kappa,
        n_initial_min=config.n_initial_min, random_seed=config.random_seed),
        {i for i, v in enumerate(problem.variables) if v.kind == "integer"})
    return LegacyBOBridge(problem, spec, backend)


class LegacyBOBridge:
    """Public migration API; historical BO policies remain workflow-owned."""
    def __init__(self, problem, spec, backend):
        self.problem, self.spec, self.backend = problem, spec, backend
        self.stop_reason = None

    def ask(self, active_bounds=None):
        if len(self.problem.objectives) > 1:
            if active_bounds is not None:
                self.backend.set_effective_bounds(active_bounds)
            return self.backend.ask()
        return self.backend.ask(active_bounds=active_bounds)

    def tell(self, x, y, **kwargs):
        return self.backend.tell(x, y, **kwargs)

    def set_effective_bounds(self, bounds):
        return self.backend.set_effective_bounds(bounds)

    def capabilities(self):
        return capabilities(self.spec.algorithm)

    def state_dict(self):
        return dict(identity=identity(self.problem, self.spec), resumable=False,
                    reason="legacy workflow exact continuation unsupported")

    def load_state_dict(self, state):
        raise ValueError("legacy BO resume disabled")
