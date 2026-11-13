"""Shared candidate scheduling; real evaluation always uses authorized run_case."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .problem import Observation, finite
from .registry import identity


def observe_case(case, candidate, problem):
    lifecycle = case.impact_outputs.get("_lifecycle", {})
    status = ("unknown" if lifecycle.get("execution_state") == "unknown" else
              "simulation_failed" if not case.simulation_valid else "feasible")
    raw, residuals = {}, {}
    if status == "feasible":
        for obj in problem.objectives:
            value = case.get_objective(obj.name)
            if (value is None or not value.available or not finite(value.value)
                    or value.minimize != obj.minimize or value.unit != obj.unit):
                status = "metrics_unavailable"
                break
            raw[obj.name] = value.value
        for c in problem.constraints:
            cv = next((x for x in case.constraints if x.name == c.name), None)
            if cv is None or not cv.available or not finite(cv.value) or cv.satisfied != (cv.value <= 0):
                status = "metrics_unavailable"
                break
            residuals[c.name] = cv.value
        if status == "feasible" and any(v > 0 for v in residuals.values()):
            status = "infeasible"
    return Observation(candidate.id, status, raw, residuals, case.case_id)


@dataclass
class UnifiedResult:
    algorithm_identity: dict
    records: list = field(default_factory=list)
    stop_reason: str | None = None
    evaluations_used: int = 0
    cases: list = field(default_factory=list, repr=False)
    diagnostics: dict = field(default_factory=dict)

    def to_summary(self):
        good = [r for r in self.records if r["observation"]["status"] == "feasible"]
        objectives = self.algorithm_identity["problem"]["objectives"]
        def values(r):
            return [r["observation"]["objectives"][o["name"]] * (1 if o["minimize"] else -1) for o in objectives]
        front = [r for r in good if not any(all(a <= b for a, b in zip(values(s), values(r))) and
                 any(a < b for a, b in zip(values(s), values(r))) for s in good)]
        return dict(algorithm=self.algorithm_identity, evaluations_used=self.evaluations_used,
            stop_reason=self.stop_reason, records=self.records,
            best_feasible=min(good, key=values) if good and len(objectives) == 1 else None,
            feasible_pareto=front if len(objectives) > 1 else [], no_feasible_solution=not good,
            optimizer_snapshot=self.diagnostics,
            status_counts={s: sum(r["observation"]["status"] == s for r in self.records) for s in
                ("feasible", "infeasible", "simulation_failed", "metrics_unavailable", "timeout", "unknown")},
            limitations=self.algorithm_identity["capabilities"]["limitations"])


def run_optimizer(optimizer, evaluator, *, remaining, used=lambda: 0):
    """Host-owned evaluator only. Does not grant permission to use any simulator.

    Simulations call this with a run_case closure; analytical tests use pure
    evaluators. Budget ownership stays in the Gateway ledger, not the library.
    Duplicate population members are evaluated as submitted (including repeats
    between generations); no public resampling or fabricated cached feedback.
    """
    result = UnifiedResult(identity(optimizer.problem, optimizer.spec))
    for row in getattr(optimizer, "history", []):
        result.records.append({**row, "inputs": optimizer.problem.inputs(row["candidate"]["x"])})
    attempts = len(result.records)
    while not optimizer.stop_reason:
        budget = min(remaining(), optimizer.problem.max_evaluations - attempts)
        try:
            batch = optimizer.ask(budget)
        except Exception as exc:
            optimizer.stop(f"optimizer_fault: {exc}")
            break
        if not batch:
            if not optimizer.stop_reason:
                optimizer.stop("candidate_space_exhausted")
            break
        if len(batch) > budget:
            optimizer.stop("backend_batch_exceeds_budget")
            break
        observations = []
        for c in batch:
            try:
                optimizer.problem.inputs(c.x)
            except ValueError as exc:
                optimizer.stop(f"illegal_candidate: {exc}")
                break
            if remaining() < 1:
                optimizer.stop("budget_exhausted_mid_population")
                break
            # Any exception may represent a reserved, started, unknown run.
            # Never retry automatically; the durable ledger is authoritative.
            try:
                o = evaluator(c)
            except Exception as exc:
                from src.agents.actions.authorization import ExecutionBoundaryError
                if isinstance(exc, ExecutionBoundaryError):
                    raise
                attempts += 1
                result.records.append(dict(candidate=asdict(c), inputs=optimizer.problem.inputs(c.x),
                    observation=asdict(Observation(c.id, "unknown")), error=str(exc)))
                optimizer.stop(f"evaluation_outcome_unknown: {exc}")
                break
            attempts += 1
            try:
                if o.candidate_id != c.id:
                    raise ValueError("evaluation evidence belongs to another candidate")
                o.validate(optimizer.problem)
            except (TypeError, ValueError, AttributeError) as exc:
                result.records.append(dict(candidate=asdict(c), inputs=optimizer.problem.inputs(c.x),
                    observation=asdict(Observation(c.id, "metrics_unavailable")), error=str(exc)))
                optimizer.stop("invalid_observation_evidence")
                break
            observations.append(o)
            adapted = ({"F_minimization": optimizer.problem.internal_objectives(o.objectives),
                        "G_dimensionless": {k.name: o.residuals[k.name] / k.scale for k in optimizer.problem.constraints}}
                       if o.status in {"feasible", "infeasible"} else None)
            result.records.append(dict(candidate=asdict(c), inputs=optimizer.problem.inputs(c.x),
                                       observation=asdict(o), algorithm_transform=adapted))
            if o.status in {"unknown", "timeout"}:
                optimizer.stop("execution_unknown")
                break
            if optimizer.spec.backend == "pymoo" and o.status not in {"feasible", "infeasible"}:
                optimizer.stop("invalid_population_observation")
                break
        if len(observations) == len(batch) and not optimizer.stop_reason:
            try:
                optimizer.tell(observations)
            except Exception as exc:
                optimizer.stop(f"optimizer_fault: {exc}")
        elif not optimizer.stop_reason:
            optimizer.stop("incomplete_population")
    result.stop_reason = optimizer.stop_reason
    result.evaluations_used = max(attempts, used())
    if hasattr(optimizer, "state_dict"):
        result.diagnostics = optimizer.state_dict()
    return result
