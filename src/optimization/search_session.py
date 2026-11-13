"""Incremental ask/tell adapter shared with the existing Pareto optimizer."""
from __future__ import annotations

from src.workflows.optimize_pareto_case import (
    MultiObjectiveBayesianOptimizer, _extract_all_objectives, _extract_constraint_margins,
)


class ParetoSearchSession:
    def __init__(self, config):
        self.config = config
        self.paths = list(config.param_bounds)
        # Legacy closed-loop execution remains disabled at its production entry.
        # The mathematical session is retained for offline checkpoint analysis.
        self.optimizer = MultiObjectiveBayesianOptimizer(list(config.param_bounds.values()), config, self.paths)

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
        return self.optimizer.rng_state()

    def restore_rng(self, state):
        self.optimizer.restore_rng(state)
