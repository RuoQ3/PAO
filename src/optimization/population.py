"""pymoo ask/tell adapters. Library objective evaluation is deliberately disabled."""
from __future__ import annotations

from dataclasses import asdict

from .problem import Candidate, finite
from .registry import capabilities, identity, validate_combination


def resolve_parameters(algorithm, raw):
    if algorithm not in {"PSO", "NSGA-II", "GA"}:
        raise ValueError("unknown population algorithm")
    # Explicit PAO draft defaults, pending verification against the local target
    # distribution. Adaptive coefficients and perturb_best are disabled.
    common = dict(pop_size=20, generations=10)
    defaults = (dict(w=0.9, c1=2.0, c2=2.0, max_velocity_rate=0.2) if algorithm == "PSO" else
                dict(crossover_probability=0.9, crossover_eta=15.0, mutation_eta=20.0, mutation_probability=0.1))
    allowed = {**common, **defaults}
    if not isinstance(raw, dict) or set(raw) - set(allowed):
        raise ValueError("unknown population algorithm parameters")
    params = {**allowed, **raw}
    if any(type(params[k]) is not int or params[k] < (2 if k == "pop_size" else 1) for k in common):
        raise ValueError("pop_size >= 2 and generations >= 1 required")
    for key in defaults:
        v = params[key]
        if not finite(v) or v <= 0:
            raise ValueError(f"{key} must be finite positive")
        if key in {"w", "max_velocity_rate", "crossover_probability", "mutation_probability"} and v > 1:
            raise ValueError(f"{key} must be <= 1")
        if key in {"c1", "c2"} and v > 4:
            raise ValueError(f"{key} must be <= 4")
    return params


class PymooOptimizer:
    def __init__(self, problem, spec):
        validate_combination(problem, spec)
        self.problem, self.spec = problem, spec
        self.parameters = resolve_parameters(spec.algorithm, spec.parameters)
        if self.parameters != spec.parameters:
            raise ValueError("resolve all defaults into the approval identity before initialization")
        self.stop_reason = None
        self.generation = 0
        self.pending = []
        self.history = []
        self._population = None
        try:
            from importlib.metadata import distribution
            dist = distribution("pymoo")
            license_text = dist.metadata.get("License") or dist.metadata.get("License-Expression") or ""
            for f in dist.files or []:
                if "dist-info" in str(f) and "LICENSE" in str(f).upper():
                    license_text += dist.locate_file(f).read_text(encoding="utf-8")
            if "Apache" not in license_text or "2.0" not in license_text:
                raise RuntimeError("pymoo Apache-2.0 license could not be verified in installed distribution")
            import numpy as np
            from pymoo.core.problem import Problem
            from pymoo.core.evaluator import Evaluator
            from pymoo.operators.sampling.rnd import FloatRandomSampling
            class ExternalOnlyProblem(Problem):
                def _evaluate(self, *args, **kwargs):
                    raise RuntimeError("pymoo objective callbacks disabled; use controlled execution")
            self._problem = ExternalOnlyProblem(n_var=len(problem.variables), n_obj=len(problem.objectives),
                n_ieq_constr=len(problem.constraints), xl=np.array([v.lower for v in problem.variables]),
                xu=np.array([v.upper for v in problem.variables]))
            p = self.parameters
            if spec.algorithm == "PSO":
                from pymoo.algorithms.soo.nonconvex.pso import PSO
                self._algorithm = PSO(pop_size=p["pop_size"], w=p["w"], c1=p["c1"], c2=p["c2"],
                    adaptive=False, perturb_best=False, max_velocity_rate=p["max_velocity_rate"],
                    sampling=FloatRandomSampling())
            else:
                from pymoo.algorithms.moo.nsga2 import NSGA2
                from pymoo.algorithms.soo.nonconvex.ga import GA
                from pymoo.operators.crossover.sbx import SBX
                from pymoo.operators.mutation.pm import PM
                from pymoo.core.repair import Repair
                integer_indices = [i for i, v in enumerate(problem.variables) if v.kind == "integer"]
                class IntegerRepair(Repair):
                    def _do(self, problem, X, **kwargs):
                        X[:, integer_indices] = np.rint(X[:, integer_indices])
                        return X
                repair = IntegerRepair() if integer_indices else None
                factory = GA if spec.algorithm == "GA" else NSGA2
                self._algorithm = factory(pop_size=p["pop_size"],
                    sampling=FloatRandomSampling(),
                    repair=repair,
                    crossover=SBX(prob=p["crossover_probability"], eta=p["crossover_eta"], repair=repair),
                    mutation=PM(prob=1.0, prob_var=p["mutation_probability"], eta=p["mutation_eta"], repair=repair),
                    eliminate_duplicates=True)
            # F/G are set only after controlled evaluation; no evaluator.eval call.
            self._algorithm.setup(self._problem, termination=("n_gen", p["generations"]), seed=spec.seed,
                                  verbose=False, evaluator=Evaluator())
        except Exception as exc:
            self.stop_reason = f"initialization_failed: {exc}"
            raise RuntimeError(f"{spec.algorithm}/pymoo initialization failed: {exc}") from exc

    def capabilities(self):
        return capabilities(self.spec.algorithm)

    def stop(self, reason):
        self.stop_reason = reason

    def ask(self, remaining, region=None):
        if region is not None:
            raise ValueError("population dynamic search regions unsupported")
        if self.pending:
            raise RuntimeError("feedback required before next generation")
        if self.stop_reason:
            return []
        if self.generation >= self.parameters["generations"]:
            self.stop("generations_completed")
            return []
        if remaining < self.parameters["pop_size"]:
            self.stop("insufficient_budget_for_population")
            return []
        try:
            self._population = self._algorithm.ask()
            if self._population is None or len(self._population) == 0:
                self.stop("candidate_space_exhausted")
                return []
            if len(self._population) > self.parameters["pop_size"]:
                raise ValueError("backend generated an unexpected population size")
            if len(self._population) < self.parameters["pop_size"]:
                self.stop("candidate_space_exhausted")
                return []
            xs = self._population.get("X").tolist()
            for x in xs:
                self.problem.inputs(x)  # reject illegal outputs, never public-layer repair
            self.pending = [Candidate(f"g{self.generation}:i{i}", tuple(x), self.generation) for i, x in enumerate(xs)]
            return list(self.pending)
        except Exception as exc:
            self.stop(f"ask_failed: {exc}")
            raise RuntimeError(f"{self.spec.algorithm}/pymoo ask failed: {exc}") from exc

    def tell(self, observations):
        if [o.candidate_id for o in observations] != [c.id for c in self.pending] or not self.pending:
            raise ValueError("feedback must match the complete pending population in order")
        for o in observations:
            o.validate(self.problem)
        self.history.extend({"candidate": asdict(c), "observation": asdict(o)} for c, o in zip(self.pending, observations))
        if any(o.status not in {"feasible", "infeasible"} for o in observations):
            self.stop("invalid_population_observation")
            self.pending = []
            return
        try:
            import numpy as np
            self._population.set("F", np.array([self.problem.internal_objectives(o.objectives) for o in observations]))
            if self.problem.constraints:
                self._population.set("G", np.array([[o.residuals[c.name] / c.scale for c in self.problem.constraints] for o in observations]))
            for individual in self._population:
                individual.evaluated.update(["F", "G", "H"])
            self._algorithm.tell(infills=self._population)
            self.generation += 1
            self.pending = []
        except Exception as exc:
            self.stop(f"tell_failed: {exc}")
            raise RuntimeError(f"{self.spec.algorithm}/pymoo tell failed: {exc}") from exc

    def state_dict(self):
        return dict(format=1, identity=identity(self.problem, self.spec), generation=self.generation,
                    pending=[asdict(c) for c in self.pending], history=self.history, stop_reason=self.stop_reason,
                    resumable=False, reason="backend state serialization has not been validated; diagnostic snapshot only")

    def load_state_dict(self, state):
        raise ValueError("pymoo resume disabled; never reinitialize as continuation")
