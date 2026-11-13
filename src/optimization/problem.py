"""Optimizer-neutral engineering problem. No simulator or backend callbacks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Protocol

from src.agents.actions.authorization import digest


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class Variable:
    id: str
    path: str
    lower: float
    upper: float
    kind: str = "continuous"
    unit: str = ""


@dataclass(frozen=True)
class Objective:
    name: str
    minimize: bool = True
    unit: str = ""
    validity: str = "converged_finite_complete"


@dataclass(frozen=True)
class Constraint:
    name: str
    definition: dict
    scale: float = 1.0
    convention: str = "residual_le_zero"


@dataclass(frozen=True)
class OptimizationProblem:
    variables: tuple[Variable, ...]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()
    fixed: dict = field(default_factory=dict)
    derived: tuple[dict, ...] = ()
    authorization_scope: dict = field(default_factory=dict)
    max_evaluations: int = 30
    identity: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.variables or not self.objectives:
            raise ValueError("problem needs variables and objectives")
        if type(self.max_evaluations) is not int or self.max_evaluations < 1:
            raise ValueError("total evaluation budget must be positive integer")
        for names in ([v.id for v in self.variables], [v.path.casefold() for v in self.variables],
                      [o.name for o in self.objectives], [c.name for c in self.constraints]):
            if any(not n for n in names) or len(names) != len(set(names)):
                raise ValueError("problem identifiers must be nonempty and unique")
        if set(p.casefold() for p in self.fixed) & {v.path.casefold() for v in self.variables}:
            raise ValueError("fixed and search variables overlap")
        if any(not finite(x) for x in self.fixed.values()):
            raise ValueError("fixed variables must be finite")
        for v in self.variables:
            if v.kind not in {"continuous", "integer"}:
                raise ValueError("unsupported variable type")
            if not finite(v.lower) or not finite(v.upper) or v.lower >= v.upper:
                raise ValueError("degenerate bounds; declare a fixed variable explicitly")
            if v.kind == "integer" and any(not float(x).is_integer() for x in (v.lower, v.upper)):
                raise ValueError("integer bounds required")
        if any(type(o.minimize) is not bool or o.validity != "converged_finite_complete" for o in self.objectives):
            raise ValueError("invalid objective direction/validity requirement")
        if any(not finite(c.scale) or c.scale <= 0 or c.convention != "residual_le_zero" for c in self.constraints):
            raise ValueError("constraint scale must be finite positive, convention residual_le_zero")
        if self.authorization_scope:
            for path, lo, hi in [(v.path, v.lower, v.upper) for v in self.variables] + [(p, v, v) for p, v in self.fixed.items()]:
                scope = self.authorization_scope.get(path)
                if (not isinstance(scope, (tuple, list)) or len(scope) != 2 or any(not finite(s) for s in scope)
                        or not scope[0] <= lo <= hi <= scope[1]):
                    raise ValueError("problem bounds outside authorized scope")

    @property
    def fingerprint(self):
        return digest(asdict(self))

    def inputs(self, x):
        if len(x) != len(self.variables):
            raise ValueError("candidate dimension changed")
        for v, value in zip(self.variables, x):
            if not finite(value) or not v.lower <= value <= v.upper:
                raise ValueError(f"illegal candidate: {v.id}")
            if v.kind == "integer" and not float(value).is_integer():
                raise ValueError(f"noninteger candidate: {v.id}")
        return {**self.fixed, **{v.path: value for v, value in zip(self.variables, x)}}

    def internal_objectives(self, raw):
        if set(raw) != {o.name for o in self.objectives} or any(not finite(v) for v in raw.values()):
            raise ValueError("incomplete/nonfinite objective vector")
        return [float(raw[o.name]) * (1 if o.minimize else -1) for o in self.objectives]


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm: str
    backend: str
    parameters: dict = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self):
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError("seed must be null or a uint32 integer")


@dataclass(frozen=True)
class Candidate:
    id: str
    x: tuple[float, ...]
    generation: int


@dataclass(frozen=True)
class Observation:
    candidate_id: str
    status: str  # feasible, infeasible, simulation_failed, metrics_unavailable, unknown, timeout
    objectives: dict = field(default_factory=dict)  # raw engineering values only
    residuals: dict = field(default_factory=dict)  # engineering residual <= 0, before scaling
    evidence_id: str | None = None

    def validate(self, problem):
        if self.status not in {"feasible", "infeasible", "simulation_failed", "metrics_unavailable", "unknown", "timeout"}:
            raise ValueError("invalid observation status")
        if self.status in {"feasible", "infeasible"}:
            problem.internal_objectives(self.objectives)
            if set(self.residuals) != {c.name for c in problem.constraints} or any(not finite(v) for v in self.residuals.values()):
                raise ValueError("incomplete/nonfinite constraint residuals")
            feasible = all(v <= 0 for v in self.residuals.values())
            if feasible != (self.status == "feasible"):
                raise ValueError("feasibility status disagrees with residuals")


class Optimizer(Protocol):
    """Candidates only; initialization occurs in the registered factory.

    ask returns an indivisible batch unless capabilities explicitly say otherwise.
    state_dict/load_state_dict must refuse unsupported restoration, never restart.
    """
    problem: OptimizationProblem
    spec: AlgorithmSpec
    stop_reason: str | None

    def ask(self, remaining: int, region: list | None = None) -> list[Candidate]: ...
    def tell(self, observations: list[Observation]) -> None: ...
    def stop(self, reason: str) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def capabilities(self) -> dict: ...
