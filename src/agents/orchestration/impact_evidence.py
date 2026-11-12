"""Target-specific evidence and bounded OFAT scheduling (no Aspen dependencies)."""
from __future__ import annotations

import math
import time
import uuid
from copy import deepcopy


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def assessment(case):
    return case.impact_outputs.get("_assessment", {})


def usable(case, metric):
    a = assessment(case)
    return (a.get("metric") == metric and a.get("target_validity") == "valid"
            and a.get("fresh") is True and finite(case.impact_outputs.get(metric)))


def assess(case, evidence, metric, run_id):
    """Never infer freshness from a finite value or a newly assigned timestamp."""
    value = case.impact_outputs.get(metric)
    fresh = evidence.get("run_id") == run_id and evidence.get("fresh") is True
    target = evidence.get("target_status", "unknown")
    verified = evidence.get("target_verified") is True
    valid = fresh and finite(value) and target in {"success", "warnings"} and verified
    failure = evidence.get("failure_kind")
    if evidence.get("execution_state") == "not_executed":
        case.converged = None
    if failure not in {"global_model", "target_extraction", "local_nonconvergence", "execution_unknown"}:
        failure = ("local_nonconvergence" if case.converged is False else
                   "target_extraction" if not finite(value) else "unverified")
    a = {"run_id": run_id, "metric": metric, "run_success": case.converged,
         "execution_state": evidence.get("execution_state", "unknown" if case.converged is None else "attempted"),
         "block_statuses": evidence.get("block_statuses", {}), "target_status": target,
         "metric_present": metric in case.impact_outputs, "metric_finite": finite(value),
         "fresh": fresh, "source": case.result_source, "provenance": deepcopy(evidence),
         "target_validity": "valid" if valid else "invalid" if target == "errors" else "unverified" if target == "unknown" else "invalid" if not finite(value) else "unverified",
         "failure_kind": None if valid else failure, "reasons": list(case.errors),
         "inputs": dict(case.impact_inputs)}
    if not valid:
        a["reasons"].extend(x for ok, x in [(fresh, "本次运行来源未确认"),
            (finite(value), "目标指标缺失或非有限"), (verified, "目标有效性未验证")] if not ok)
    case.impact_outputs["_assessment"] = a


def in_scope(change, plan):
    params = plan.adaptive_policy.get("parameters", [])
    if not params:
        return any(change["inputs"] == c["inputs"] for c in plan.parameter_changes)
    inputs = change["inputs"]
    if set(inputs) != {p["aspen_path"] for p in params}:
        return False
    return all(finite(inputs[p["aspen_path"]]) and p["lower"] <= inputs[p["aspen_path"]] <= p["upper"]
               and (not p.get("integer") or float(inputs[p["aspen_path"]]).is_integer()) for p in params)


def new_plan(plan, changes, reason):
    from .parameter_impact import SimulationPlan
    raw = deepcopy(plan.to_dict())
    raw.update(plan_id=uuid.uuid4().hex, approved=False, requires_confirmation=True,
               temporary_model=None, parameter_changes=changes, reason=reason,
               max_evaluations=max(plan.max_evaluations, len(changes)))
    # Make any requested boundary expansion explicit in the new approval artifact.
    params = raw["adaptive_policy"].get("parameters", [])
    known = {p["aspen_path"] for p in params}
    for change in changes:
        if set(change["inputs"]) != known:
            raw["executable"] = False  # New parameters need metadata/authorization first.
        for p in params:
            value = change["inputs"].get(p["aspen_path"])
            if finite(value):
                p["lower"] = min(p["lower"], value)
                p["upper"] = max(p["upper"], value)
    return SimulationPlan.from_dict(raw).to_dict()


def neighborhood(plan, reference):
    """Round-robin small two-sided steps before larger steps, including integers."""
    policy = plan.adaptive_policy
    inputs = reference.impact_inputs
    candidates = []
    for step in policy.get("relative_steps", [.05, .10]):
        for p in policy.get("parameters", []):
            path = p["aspen_path"]
            base = inputs[path]
            delta = max(1, round(abs(base) * step)) if p.get("integer") else (abs(base) or p["upper"] - p["lower"]) * step
            for sign in (-1, 1):
                value = max(p["lower"], min(p["upper"], base + sign * delta))
                if p.get("integer"):
                    value = max(math.ceil(p["lower"]), min(math.floor(p["upper"]), round(value)))
                if value != base:
                    candidates.append({"case_id": uuid.uuid4().hex, "inputs": {**inputs, path: value},
                                       "changed_parameter": path, "shrink": 0})
    return candidates


def execute_design(plan, invoke, persist=None):
    policy = plan.adaptive_policy
    metric = (plan.output_spec or {}).get("metric")
    queue = deepcopy(plan.parameter_changes)
    results, seen = [], set()
    reference = None
    failures = 0
    started = time.monotonic()
    stop = "design_complete"
    pending = None
    while queue:
        if len(results) >= plan.max_evaluations:
            stop = "case_budget"; break
        if time.monotonic() - started >= policy.get("max_seconds", plan.timeout_seconds * plan.max_evaluations):
            stop = "time_budget"; break
        if failures >= policy.get("max_failures", 4):
            stop = "failure_budget"; break
        if reference is None and len(results) >= 1 + policy.get("max_reference_trials", 3):
            stop = "reference_search_budget"; break
        change = queue.pop(0)
        key = tuple(sorted(change["inputs"].items()))
        if key in seen:
            continue
        if not in_scope(change, plan):
            stop = "requires_reapproval"
            pending = new_plan(plan, [change, *queue], "Requested design exceeds approved parameter scope")
            break
        seen.add(key)
        remaining = policy.get("max_seconds", plan.timeout_seconds * plan.max_evaluations) - (time.monotonic() - started)
        if remaining <= 0:
            queue.insert(0, change)
            seen.remove(key)
            stop = "time_budget"; break
        case = invoke(change, len(results), min(plan.timeout_seconds, remaining))
        a = assessment(case)
        a["design"] = deepcopy(change)
        a["reference_case_id"] = reference.case_id if reference else None
        results.append(case)
        if not usable(case, metric) or not case.converged:
            failures += 1
        if usable(case, metric) and reference is None:
            reference = case
            a["is_reference"] = True
            a["reference_changed"] = len(results) != 1
            if len(results) != 1:
                queue = neighborhood(plan, reference)
        if persist:
            persist(case)
        lifecycle = case.impact_outputs.get("_lifecycle", {})
        if lifecycle.get("execution_state") == "unknown":
            stop = "execution_unknown"; break
        if lifecycle.get("cleanup_state") == "incomplete":
            stop = "cleanup_incomplete"; break
        kind = a.get("failure_kind")
        if kind in {"global_model", "target_extraction", "unverified", "execution_unknown"}:
            stop = kind; break
        if reference is not None and not usable(case, metric) and policy:
            path = change.get("changed_parameter")
            p = next((p for p in policy.get("parameters", []) if p["aspen_path"] == path), None)
            depth = change.get("shrink", 0)
            if p and depth < policy.get("max_shrinks", 2):
                value = (change["inputs"][path] + reference.impact_inputs[path]) / 2
                if p.get("integer"):
                    value = round(value)
                if value != reference.impact_inputs[path]:
                    queue.append({"case_id": uuid.uuid4().hex,
                        "inputs": {**reference.impact_inputs, path: value}, "changed_parameter": path, "shrink": depth + 1})
        if reference is None and not policy:
            stop = "reference_search_not_approved"; break
    if queue and stop in {"case_budget", "failure_budget", "time_budget", "reference_search_budget"}:
        pending = new_plan(plan, queue, "Additional valid evidence requires a newly approved execution budget")
    return results, {"stop_reason": stop, "planned_case_count": len(plan.parameter_changes),
        "actual_case_count": len(results), "valid_metric_count": sum(usable(c, metric) for c in results),
        "failed_case_count": sum(assessment(c).get("run_success") is False for c in results),
        "unknown_case_count": sum(assessment(c).get("run_success") is None for c in results),
        "not_executed_case_count": sum(assessment(c).get("execution_state") == "not_executed" for c in results),
        "invalid_metric_count": sum(not usable(c, metric) for c in results),
        "skipped_case_count": sum(tuple(sorted(c["inputs"].items())) not in seen for c in plan.parameter_changes),
        "reference_case_id": reference.case_id if reference else None,
        "reference_changed": bool(reference and assessment(reference).get("reference_changed")),
        "reapproval_plan": pending,
        "added_case_count": sum(not any(c.impact_inputs == x["inputs"] for x in plan.parameter_changes) for c in results),
        "count_scope": "planned=original design; actual=reserved attempts with records; added=actual points outside original design; skipped=original points not attempted",
        "skipped_cases": [{"case_id": c["case_id"], "reason": stop} for c in plan.parameter_changes
                          if tuple(sorted(c["inputs"].items())) not in seen]}


def analyze(results, metric, parameters, parameter_specs=None):
    valid = [c for c in results if usable(c, metric)]
    ref = next((c for c in valid if assessment(c).get("is_reference")), None)
    # Imported data must explicitly identify the reference; never replace it silently.
    if ref is None:
        ref = next((c for c in valid if c.case_id == "baseline"), None)
    report = {"metric": metric, "baseline": ref.impact_outputs[metric] if ref else None,
        "reference_case_id": ref.case_id if ref else None,
        "reference_changed": bool(ref and assessment(ref).get("reference_changed")),
        "sample_count": len(valid), "effective_case_count": len(valid),
        "failed_case_count": sum(c.converged is False for c in results),
        "unknown_case_count": sum(c.converged is None for c in results),
        "missing_output_count": sum(not finite(c.impact_outputs.get(metric)) for c in results),
        "convergence_rate": sum(c.converged is True for c in results) / max(1, len(results)),
        "parameter_rank": [], "parameters": [], "limitations": [], "conclusion_type": "insufficient_data"}
    specs = {p["aspen_path"]: p for p in parameter_specs or []}
    for parameter in parameters:
        rows = [c for c in results if parameter in c.impact_inputs and (not ref or
            set(c.impact_inputs) == set(ref.impact_inputs) and all(v == c.impact_inputs[k]
                for k, v in ref.impact_inputs.items() if k != parameter))]
        pairs = {}
        for c in rows:
            if usable(c, metric) and finite(c.impact_inputs[parameter]):
                pairs.setdefault(float(c.impact_inputs[parameter]), []).append(c.impact_outputs[metric])
        # Conflicting repeats cannot be averaged into invented evidence.
        conflicting = any(len(set(ys)) > 1 for ys in pairs.values())
        values = sorted(pairs)
        failed = [{"value": c.impact_inputs[parameter], "failure_kind": assessment(c).get("failure_kind"),
                   "reasons": assessment(c).get("reasons", [])} for c in rows if not usable(c, metric)]
        row = {"parameter": parameter, "evidence_strength": "insufficient_data", "reason": "缺少有效参考点或不同取值",
               "parameter_unit": specs.get(parameter, {}).get("unit"),
               "observations": [{"value": x, "metric_values": pairs[x]} for x in values],
               "sample_count": len(values), "unique_value_count": len(values), "valid_values": values,
               "failed_values": failed, "success_rate": sum(c.converged is True for c in rows) / len(rows) if rows else None,
               "valid_metric_rate": sum(usable(c, metric) for c in rows) / len(rows) if rows else None,
               "feasible_interval_hint": [min(values), max(values)] if values else None,
               "interval_warning": "仅为已验证离散点包络，不证明区间内可行", "confidence": "not_estimated"}
        if ref and parameter in ref.impact_inputs and not conflicting:
            x0, y0 = ref.impact_inputs[parameter], ref.impact_outputs[metric]
            row.update(reference_value=x0, reference_metric=y0,
                       comparisons=[{"value": x, "metric_value": pairs[x][0], "delta_metric": pairs[x][0] - y0}
                                    for x in values])
            left, right = [x for x in values if x < x0], [x for x in values if x > x0]
            if left and right:
                lo, hi = max(left), min(right)
                slope = (pairs[hi][0] - pairs[lo][0]) / (hi - lo)
                row.update(evidence_strength="local_direction", reason="有效参考点及双侧单因素扰动",
                    local_slope=slope, direction="positive" if slope > 0 else "negative" if slope < 0 else "flat",
                    perturbation_range=[lo, hi], metric_change=abs(pairs[hi][0] - pairs[lo][0]))
                spec = specs.get(parameter)
                scale = spec["upper"] - spec["lower"] if spec else None
                if scale and scale > 0:
                    row["normalized_coverage"] = [(x0 - lo) / scale, (hi - x0) / scale]
                    row["sensitivity"] = abs(slope) * scale / max(abs(y0), 1e-12)
            elif left or right:
                row.update(evidence_strength="exploratory_one_sided", reason="仅有单侧有效扰动，不能给出双侧局部斜率",
                    one_sided_changes=[{"value": x, "delta_metric": pairs[x][0] - y0} for x in values if x != x0])
        elif conflicting:
            row["reason"] = "重复取值结果冲突，需检查可重复性"
        report["parameters"].append(row)
    rows = report["parameters"]
    coverage = [r.get("normalized_coverage") for r in rows]
    comparable = bool(rows) and all(r["evidence_strength"] == "local_direction" for r in rows) and all(coverage)
    if comparable:
        comparable = all(math.isclose(c[i], coverage[0][i], rel_tol=.1, abs_tol=1e-12) for c in coverage for i in (0, 1))
    if comparable:
        report["parameter_rank"] = sorted(rows, key=lambda r: r["sensitivity"], reverse=True)
        report["conclusion_type"] = "exploratory_ranking"
    elif any(r["evidence_strength"] == "local_direction" for r in rows):
        report["conclusion_type"] = "local_direction"
    elif any(r["evidence_strength"] == "exploratory_one_sided" for r in rows):
        report["conclusion_type"] = "exploratory_one_sided"
    if not comparable:
        report["limitations"].append("各参数双侧数据覆盖或归一化取值范围不可比较，不给出排序。")
    report["limitations"].append("有限样本仅支持局部方向或探索性排序，不代表统计显著性、全局重要性或连续可行区间。")
    report["uncertainty"] = list(report["limitations"])
    return report


def execution_evidence(result, plan):
    """One serializable fact packet used by interpretation and persisted audit.

    Counts deliberately overlap: convergence, target validity and constraints
    are independent. All reference changes are absolute in the native unit.
    """
    from src.agents.actions.authorization import digest
    cases = deepcopy(result.get("simulation_cases", []))
    spec = deepcopy(plan.output_spec or {})
    reference = next((c for c in cases if c["case_id"] == result.get("reference_case_id")), None)
    metric = spec.get("metric")
    baseline = reference["impact_outputs"].get(metric) if reference else None
    observations = []
    for c in cases:
        a = c["impact_outputs"].get("_assessment", {})
        value = c["impact_outputs"].get(metric) if a.get("target_validity") == "valid" else None
        observations.append({"case_id": c["case_id"], "run_id": c.get("run_id"),
            "source": c["impact_outputs"].get("_source"), "assessment": a,
            "lifecycle": c["impact_outputs"].get("_lifecycle"),
            "actual_inputs": a.get("provenance", {}).get("actual_inputs", c.get("impact_inputs")),
            "constraints": c.get("constraints", []),
            "objectives": c.get("objectives", []),
            "metric": {"label": metric, "value": value, "unit": spec.get("unit") or "unknown",
                "baseline": baseline, "delta": value - baseline if finite(value) and finite(baseline) else None,
                "scope": "local tested cases", "reference_case_id": reference["case_id"] if reference else None}})
    packet = {"version": 2, "execution_id": result["audit_id"], "batch_id": plan.plan_id,
        "output_spec": spec, "counts": {k: result.get(k) for k in (
            "planned_case_count", "actual_case_count", "added_case_count", "skipped_case_count",
            "valid_metric_count", "invalid_metric_count", "failed_case_count", "unknown_case_count", "not_executed_case_count")},
        "skipped_cases": result.get("skipped_cases", []), "observations": observations,
        "constraints": "per observation; no implicit product constraints",
        "scope": "finite local observations; no causal or global importance claim",
        "evaluations": deepcopy(result.get("evaluations", [])),
        "resources": deepcopy(result.get("resources", {})),
        "statistics": deepcopy(result.get("sensitivity", {}))}
    packet["evidence_digest"] = digest(packet)
    return packet
