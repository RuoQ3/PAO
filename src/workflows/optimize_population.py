"""Population workflow under the same ActionGateway envelope as BO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.actions.authorization import authorized_workflow, require_authorization
from src.optimization.problem import OptimizationProblem, AlgorithmSpec
from src.optimization.registry import create_optimizer
from src.optimization.execution import run_optimizer, observe_case


@dataclass
class PopulationOptimizeConfig:
    problem: OptimizationProblem
    algorithm_spec: AlgorithmSpec
    run_config: Any
    param_bounds: dict
    fixed_vars: dict
    derived_var_specs: list
    integer_var_paths: set
    n_initial: int
    n_iterations: int
    db_path: Any = None
    session_id: str = ""


@authorized_workflow
def optimize_population(driver, config):
    from .run_case import run_case
    grant = require_authorization()
    optimizer = create_optimizer(config.problem, config.algorithm_spec)
    cases = []
    def evaluate(candidate):
        case = run_case(driver, config.problem.inputs(candidate.x), config.run_config,
                        iteration=grant.used, tags=[config.algorithm_spec.algorithm, "optimize"])
        cases.append(case)
        return observe_case(case, candidate, config.problem)
    result = run_optimizer(optimizer, evaluate, remaining=lambda: grant.max_evaluations - grant.used,
                           used=lambda: grant.used)
    result.cases = cases
    return result
