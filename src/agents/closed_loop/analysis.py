"""Deterministic evidence analysis for the closed-loop decision Agent.

The LLM is deliberately kept out of this module.  Every function here turns
the observed :class:`ProcessCase` records into finite, JSON-serialisable
evidence that can be inspected, persisted and replayed during checkpoint
resume.  The decision Agent interprets the report; it does not recompute the
statistics from an unbounded history of raw rows.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from src.models.process_case import CaseStatus, ProcessCase
from src.optimization.convergence_kb import match_failure_patterns
from src.optimization.metrics import rank_variables, sensitivity_analysis
from src.optimization.pareto import compute_pareto

ANALYSIS_REPORT_VERSION = 1
DEFAULT_RECENT_WINDOW = 12
MIN_SENSITIVITY_SAMPLES = 3


def _finite(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing/non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _status_value(case: ProcessCase) -> str:
    status = case.status
    return status.value if isinstance(status, CaseStatus) else str(status)


def _is_simulation_valid(case: ProcessCase) -> bool:
    return bool(case.simulation_valid)


def _is_success(case: ProcessCase) -> bool:
    return bool(case.success)


def _safe_case_id(case: ProcessCase) -> str:
    return str(case.case_id)


def _error_text(case: ProcessCase) -> str:
    parts: list[str] = []
    if case.sim_result is not None and case.sim_result.error:
        parts.append(str(case.sim_result.error))
    if case.notes:
        parts.append(str(case.notes))
    return " | ".join(p for p in parts if p).strip()


def _objective_values(
    cases: Sequence[ProcessCase],
    name: str,
) -> list[tuple[ProcessCase, float, bool]]:
    values: list[tuple[ProcessCase, float, bool]] = []
    for case in cases:
        objective = case.get_objective(name)
        if objective is None or not objective.available:
            continue
        value = _finite(objective.value)
        if value is not None:
            values.append((case, value, bool(objective.minimize)))
    return values


def _objective_summary(cases: Sequence[ProcessCase], name: str) -> dict[str, Any]:
    values = _objective_values(cases, name)
    if not values:
        return {
            "n_available": 0,
            "direction": None,
            "first_value": None,
            "latest_value": None,
            "best_value": None,
            "best_case_id": None,
            "latest_case_id": None,
            "improvement_from_first": None,
            "best_improvement_from_first": None,
        }

    minimize = values[0][2]
    oriented = [value if minimize else -value for _, value, _ in values]
    best_index = min(range(len(values)), key=oriented.__getitem__)
    first_value = values[0][1]
    latest_value = values[-1][1]
    first_oriented = oriented[0]
    latest_oriented = oriented[-1]
    best_oriented = oriented[best_index]
    return {
        "n_available": len(values),
        "direction": "minimize" if minimize else "maximize",
        "first_value": first_value,
        "latest_value": latest_value,
        "best_value": values[best_index][1],
        "best_case_id": _safe_case_id(values[best_index][0]),
        "latest_case_id": _safe_case_id(values[-1][0]),
        # Positive means an improvement in the declared objective direction.
        "improvement_from_first": first_oriented - latest_oriented,
        "best_improvement_from_first": first_oriented - best_oriented,
    }


def _constraint_names(cases: Sequence[ProcessCase]) -> list[str]:
    names: set[str] = set()
    for case in cases:
        names.update(str(item.name) for item in case.constraints)
    return sorted(names)


def _constraint_summary(cases: Sequence[ProcessCase], name: str) -> dict[str, Any]:
    values: list[tuple[ProcessCase, float, bool]] = []
    for case in cases:
        constraint = next((item for item in case.constraints if item.name == name), None)
        if constraint is None or not constraint.available:
            continue
        value = _finite(constraint.value)
        if value is None:
            continue
        satisfied = bool(constraint.satisfied) if constraint.satisfied is not None else value <= 0
        values.append((case, value, satisfied))

    if not values:
        return {
            "n_available": 0,
            "n_satisfied": 0,
            "violation_rate": None,
            "max_violation": None,
            "min_margin": None,
            "worst_case_id": None,
            "first_value": None,
            "latest_value": None,
            "improvement_from_first": None,
        }

    worst_index = max(range(len(values)), key=lambda i: values[i][1])
    first_value = values[0][1]
    latest_value = values[-1][1]
    return {
        "n_available": len(values),
        "n_satisfied": sum(item[2] for item in values),
        "violation_rate": sum(not item[2] for item in values) / len(values),
        "max_violation": max(item[1] for item in values),
        # Constraint convention in ProcessCase is value <= 0; positive margin
        # therefore means distance from violation.
        "min_margin": min(-item[1] for item in values),
        "worst_case_id": _safe_case_id(values[worst_index][0]),
        "first_value": first_value,
        "latest_value": latest_value,
        # Positive means the normalized constraint value moved toward <= 0.
        "improvement_from_first": first_value - latest_value,
    }


def _failure_summary(cases: Sequence[ProcessCase]) -> dict[str, Any]:
    errors = [_error_text(case) for case in cases]
    errors = [error for error in errors if error]
    diagnoses = match_failure_patterns(errors)
    pattern_counts = Counter(item.pattern_id for item in diagnoses)
    severities = Counter(item.severity for item in diagnoses)
    examples: list[str] = []
    for error in errors:
        if error not in examples:
            examples.append(error[:500])
        if len(examples) >= 3:
            break
    return {
        "n_error_records": len(errors),
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "severity_counts": dict(sorted(severities.items())),
        "diagnoses": [
            {
                "pattern_id": item.pattern_id,
                "description": item.description,
                "matched_keywords": list(item.matched_keywords),
                "severity": item.severity,
                "fixes": list(item.fixes[:3]),
            }
            for item in diagnoses[:8]
        ],
        "examples": examples,
    }


def _data_quality_summary(
    cases: Sequence[ProcessCase],
    recent: Sequence[ProcessCase],
    objective_names: Sequence[str],
) -> dict[str, Any]:
    status_counts = Counter(_status_value(case) for case in cases)
    n_total = len(cases)
    n_valid = sum(_is_simulation_valid(case) for case in cases)
    n_success = sum(_is_success(case) for case in cases)
    n_recent = len(recent)
    n_recent_valid = sum(_is_simulation_valid(case) for case in recent)
    n_recent_success = sum(_is_success(case) for case in recent)
    n_constraints = sum(bool(case.constraints) for case in cases)
    n_objectives = sum(
        all(case.get_objective(name) is not None and case.get_objective(name).available
            for name in objective_names)
        for case in cases
    )
    return {
        "n_total": n_total,
        "n_recent": n_recent,
        "n_recent_simulation_valid": n_recent_valid,
        "n_recent_success": n_recent_success,
        "n_simulation_valid": n_valid,
        "n_success": n_success,
        "n_infeasible": status_counts.get(CaseStatus.INFEASIBLE.value, 0),
        "n_sim_failed": status_counts.get(CaseStatus.SIM_FAILED.value, 0),
        "n_objective_error": status_counts.get(CaseStatus.OBJECTIVE_ERROR.value, 0),
        "n_constraint_error": status_counts.get(CaseStatus.CONSTRAINT_ERROR.value, 0),
        "n_with_all_objectives": n_objectives,
        "n_with_constraints": n_constraints,
        "status_counts": dict(sorted(status_counts.items())),
        "sufficient_for_sensitivity": n_valid >= MIN_SENSITIVITY_SAMPLES,
        "sensitivity_min_samples": MIN_SENSITIVITY_SAMPLES,
    }


def _convergence_summary(cases: Sequence[ProcessCase], recent: Sequence[ProcessCase]) -> dict[str, Any]:
    def rate(items: Sequence[ProcessCase], predicate) -> float | None:
        return sum(predicate(case) for case in items) / len(items) if items else None

    return {
        "all_rate": rate(cases, _is_simulation_valid),
        "recent_rate": rate(recent, _is_simulation_valid),
        "all_failure_rate": rate(cases, lambda case: not _is_simulation_valid(case)),
        "recent_failure_rate": rate(recent, lambda case: not _is_simulation_valid(case)),
        "consecutive_failures": _consecutive_failures(cases),
    }


def _consecutive_failures(cases: Sequence[ProcessCase]) -> int:
    count = 0
    for case in reversed(cases):
        if _is_simulation_valid(case):
            break
        count += 1
    return count


def _sensitivity_summary(
    cases: Sequence[ProcessCase],
    optimizer_inputs: Sequence[Mapping[str, Any]],
    param_paths: Sequence[str],
    objective_names: Sequence[str],
    method: str,
) -> dict[str, Any]:
    if not param_paths or not objective_names:
        return {
            "available": False,
            "method": method,
            "n_samples": 0,
            "ranked_variables": [],
            "warnings": ["缺少设计变量或目标名称，跳过敏感性分析。"],
        }

    # ProcessCase.design_vars contains actual Aspen inputs.  The optimizer may
    # also contain derived variables, so use optimizer_inputs for this report.
    analysis_cases: list[ProcessCase] = []
    for index, case in enumerate(cases):
        values = optimizer_inputs[index] if index < len(optimizer_inputs) else case.design_vars
        analysis_cases.append(replace(case, design_vars=dict(values)))

    try:
        result = sensitivity_analysis(
            cases=analysis_cases,
            param_paths=list(param_paths),
            objective_names=list(objective_names),
            method=method,  # type: ignore[arg-type]
            include_infeasible=True,
        )
        ranked = []
        for path, score in rank_variables(result):
            ranked.append({
                "path": path,
                "score": score,
                "reliable": bool(result.is_reliable(path)),
                "effective_samples": result.min_effective(path),
            })
        return {
            "available": result.n_samples > 0,
            "method": result.method,
            "n_samples": result.n_samples,
            "min_required_samples": result.min_required_samples,
            "ranked_variables": ranked,
            "warnings": list(result.warnings),
            "scores": result.to_summary().get("scores", {}),
            "effective_samples": result.to_summary().get("effective_samples", {}),
        }
    except Exception as exc:  # noqa: BLE001 — analysis must never break the Aspen loop
        return {
            "available": False,
            "method": method,
            "n_samples": 0,
            "ranked_variables": [],
            "warnings": [f"敏感性分析失败，已隔离：{type(exc).__name__}: {exc}"],
        }


def _pareto_summary(
    cases: Sequence[ProcessCase],
    objective_names: Sequence[str],
    hypervolume: float | None,
    hv_margin: float,
    include_infeasible: bool = False,
) -> dict[str, Any]:
    if not cases or not objective_names:
        return {
            "n_evaluated": 0,
            "n_excluded": len(cases),
            "n_fronts": 0,
            "front_size": 0,
            "hypervolume": hypervolume,
            "front_case_ids": [],
            "reference_point": None,
        }
    try:
        result = compute_pareto(
            list(cases), list(objective_names), compute_hv=hypervolume is None,
            hv_margin=hv_margin, include_infeasible=include_infeasible,
        )
        front = result.first_front
        return {
            "n_evaluated": result.n_evaluated,
            "n_excluded": len(result.excluded_cases),
            "n_fronts": result.n_fronts,
            "front_size": len(front.cases) if front else 0,
            "hypervolume": hypervolume if hypervolume is not None else result.hypervolume,
            "front_case_ids": [_safe_case_id(case) for case in front.cases] if front else [],
            "reference_point": result.reference_point,
        }
    except Exception as exc:  # noqa: BLE001 — malformed historical data must not stop a run
        return {
            "n_evaluated": 0,
            "n_excluded": len(cases),
            "n_fronts": 0,
            "front_size": 0,
            "hypervolume": hypervolume,
            "front_case_ids": [],
            "reference_point": None,
            "error": f"Pareto 分析失败，已隔离：{type(exc).__name__}: {exc}",
        }


def _metric_snapshot(
    data_quality: Mapping[str, Any],
    convergence: Mapping[str, Any],
    constraints: Mapping[str, Mapping[str, Any]],
    pareto: Mapping[str, Any],
) -> dict[str, Any]:
    constraint_rates = [
        value["violation_rate"] for value in constraints.values()
        if value.get("violation_rate") is not None
    ]
    return {
        "hypervolume": pareto.get("hypervolume"),
        "feasible_rate": (
            data_quality.get("n_recent_success", 0) / data_quality["n_recent"]
            if data_quality.get("n_recent") else 0.0
        ),
        "convergence_rate": convergence.get("recent_rate"),
        "feasible_count": data_quality.get("n_success", 0),
        "constraint_violation_rate": (
            sum(constraint_rates) / len(constraint_rates) if constraint_rates else None
        ),
    }


@dataclass(frozen=True)
class AnalysisReport:
    """Versioned, JSON-ready report consumed by the decision Agent."""

    report_version: int
    window_size: int
    data_quality: dict[str, Any]
    convergence: dict[str, Any]
    objectives: dict[str, Any]
    constraints: dict[str, Any]
    pareto: dict[str, Any]
    sensitivity: dict[str, Any]
    failures: dict[str, Any]
    metrics: dict[str, Any]
    action_effect: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "window_size": self.window_size,
            "data_quality": self.data_quality,
            "convergence": self.convergence,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "pareto": self.pareto,
            "sensitivity": self.sensitivity,
            "failures": self.failures,
            "metrics": self.metrics,
            "action_effect": self.action_effect,
        }


def build_analysis_report(
    cases: Sequence[ProcessCase],
    *,
    objective_names: Sequence[str],
    param_paths: Sequence[str] = (),
    optimizer_inputs: Sequence[Mapping[str, Any]] = (),
    recent_window: int = DEFAULT_RECENT_WINDOW,
    hypervolume: float | None = None,
    hv_margin: float = 0.1,
    sensitivity_method: str = "spearman",
    include_infeasible: bool = False,
) -> dict[str, Any]:
    """Build all deterministic evidence used for one Agent decision.

    The function is intentionally side-effect free.  It accepts the in-memory
    checkpoint observations, so a resumed session produces the same report
    without querying Aspen or depending on a live COM object.
    """
    cases = list(cases)
    objective_names = [str(name) for name in objective_names]
    param_paths = [str(path) for path in param_paths]
    recent_window = max(1, int(recent_window))
    recent = cases[-recent_window:]
    metrics_hv = _finite(hypervolume) if hypervolume is not None else None

    data_quality = _data_quality_summary(cases, recent, objective_names)
    convergence = _convergence_summary(cases, recent)
    objective_report = {
        name: _objective_summary(cases, name) for name in objective_names
    }
    constraint_report = {
        name: _constraint_summary(cases, name) for name in _constraint_names(cases)
    }
    pareto = _pareto_summary(
        cases, objective_names, metrics_hv, hv_margin,
        include_infeasible=include_infeasible,
    )
    sensitivity = _sensitivity_summary(
        cases, list(optimizer_inputs), param_paths, objective_names, sensitivity_method,
    )
    failures = _failure_summary(cases)
    metrics = _metric_snapshot(data_quality, convergence, constraint_report, pareto)
    return AnalysisReport(
        report_version=ANALYSIS_REPORT_VERSION,
        window_size=recent_window,
        data_quality=data_quality,
        convergence=convergence,
        objectives=objective_report,
        constraints=constraint_report,
        pareto=pareto,
        sensitivity=sensitivity,
        failures=failures,
        metrics=metrics,
    ).to_dict()


def compare_analysis_reports(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
    *,
    expected_effect: str | None = None,
) -> dict[str, Any]:
    """Compare reports before/after one Agent action.

    Deltas are evidence of association only; the result never claims causal
    attribution.  ``None`` is returned for metrics that are not comparable.
    """
    before = before or {}
    after_metrics = after.get("metrics", {})
    before_metrics = before.get("metrics", {}) if isinstance(before, Mapping) else {}

    def delta(name: str) -> float | None:
        left = _finite(before_metrics.get(name))
        right = _finite(after_metrics.get(name))
        return right - left if left is not None and right is not None else None

    objective_delta: dict[str, float | None] = {}
    before_objectives = before.get("objectives", {}) if isinstance(before, Mapping) else {}
    after_objectives = after.get("objectives", {})
    for name, current in after_objectives.items():
        previous = before_objectives.get(name, {}) if isinstance(before_objectives, Mapping) else {}
        before_latest = _finite(previous.get("latest_value"))
        after_latest = _finite(current.get("latest_value"))
        if before_latest is None or after_latest is None:
            objective_delta[name] = None
            continue
        direction = current.get("direction")
        objective_delta[name] = (
            before_latest - after_latest
            if direction == "minimize" else after_latest - before_latest
        )

    expected_delta = {
        "objective": delta("hypervolume"),
        "feasibility": delta("feasible_rate"),
        "convergence": delta("convergence_rate"),
        "information": None,
    }.get(expected_effect or "")
    improved = None if expected_delta is None else expected_delta > 1e-10
    return {
        "expected_effect": expected_effect,
        "hypervolume_delta": delta("hypervolume"),
        "feasible_rate_delta": delta("feasible_rate"),
        "convergence_rate_delta": delta("convergence_rate"),
        "feasible_count_delta": delta("feasible_count"),
        "constraint_violation_rate_delta": delta("constraint_violation_rate"),
        "objective_deltas": objective_delta,
        "expected_metric_improved": improved if expected_effect else None,
        "evidence_case_count": after.get("data_quality", {}).get("n_total", 0),
        "interpretation": "Observed association only; not proof of causal attribution",
    }


def build_analysis_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    objective_names: Sequence[str],
    param_paths: Sequence[str] = (),
    recent_window: int = DEFAULT_RECENT_WINDOW,
    sensitivity_method: str = "spearman",
) -> dict[str, Any]:
    """Rebuild cases from checkpoint-like rows for standalone tools/tests."""
    from src.agents.closed_loop.journal import decode_case

    rows = list(rows)
    cases = [decode_case(dict(row)) for row in rows]
    optimizer_inputs = [dict(row.get("optimizer_inputs") or row.get("design_vars") or {}) for row in rows]
    if not param_paths:
        keys: set[str] = set()
        for values in optimizer_inputs:
            keys.update(str(key) for key in values)
        param_paths = sorted(keys)
    return build_analysis_report(
        cases,
        objective_names=objective_names,
        param_paths=param_paths,
        optimizer_inputs=optimizer_inputs,
        recent_window=recent_window,
        sensitivity_method=sensitivity_method,
    )
