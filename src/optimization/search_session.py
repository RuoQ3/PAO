"""Incremental ask/tell adapter shared with the existing Pareto optimizer."""
from __future__ import annotations

from src.workflows.optimize_pareto_case import (
    _MultiObjectiveBayesianOptimizer, _extract_all_objectives, _extract_constraint_margins,
)


class ParetoSearchSession:
    def __init__(self, config):
        self.config = config
        self.paths = list(config.param_bounds)
        self.optimizer = _MultiObjectiveBayesianOptimizer(
            list(config.param_bounds.values()), config, self.paths,
        )

    def ask(self, region: dict) -> dict:
        self.optimizer.set_effective_bounds([tuple(region[p]) for p in self.paths])
        return dict(zip(self.paths, self.optimizer.ask()))

    def tell(self, case, optimizer_inputs: dict):
        self.optimizer.tell(
            [optimizer_inputs[p] for p in self.paths],
            _extract_all_objectives(case, self.config, allow_infeasible=True),
            is_success=case.success,
            c_vec=_extract_constraint_margins(case),
        )

    def rng_state(self):
        backend = self.optimizer._botorch_opt
        return {"outer": self.optimizer._rng.getstate(),
                "backend": backend._rng.getstate() if backend else None}

    def restore_rng(self, state):
        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, list) else x
        if not state:
            return
        self.optimizer._rng.setstate(tuples(state["outer"]))
        if self.optimizer._botorch_opt and state.get("backend"):
            self.optimizer._botorch_opt._rng.setstate(tuples(state["backend"]))
