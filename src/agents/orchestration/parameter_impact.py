"""Controlled, read-only-by-default Aspen parameter impact analysis.

This module deliberately keeps planning, validation, execution and analysis
separate.  A plan is data only; an ``AspenSandboxRunner`` refuses to execute
until its plan is explicitly approved by the caller (normally ActionGateway).
The module has no dependency on a particular case, block or component name.
"""
from __future__ import annotations

import json
import math
import logging
import threading
import re
import shutil
import statistics
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.models.process_case import CaseStatus, ProcessCase


_SAFE_PATH = re.compile(r"^\\Data\\(?:Blocks|Streams)\\[A-Za-z0-9_ .:-]+\\(?:Input|Output)(?:\\[A-Za-z0-9_ .:-]+)+$")
_STRUCTURAL_WORDS = {"connections", "number of stages", "nstage", "flowsheet", "structure", "topology", "property method"}


@dataclass
class OutputSpec:
    """Logical impact metric mapped to one or more Aspen output nodes."""
    metric: str
    stream_id: str | None = None
    component: str | None = None
    aspen_paths: list[str] = field(default_factory=list)
    extractor: str = "scalar"
    unit: str | None = None
    basis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json({"metric": self.metric, "stream_id": self.stream_id,
                      "component": self.component, "aspen_paths": self.aspen_paths,
                      "extractor": self.extractor, "unit": self.unit, "basis": self.basis})


def resolve_output_spec(context: Any, *, metric: str, stream_id: str | None,
                        component: str | None) -> OutputSpec:
    """Resolve a logical metric without ever treating it as an Aspen path."""
    metric = str(metric).casefold()
    stream = context.get_stream(stream_id) if stream_id and hasattr(context, "get_stream") else None
    base = str((stream or {}).get("aspen_path") or (rf"\Data\Streams\{stream_id}" if stream_id else ""))
    candidates = []
    for item in (getattr(context, "streams", []) or []):
        if stream_id and str(item.get("id") or item.get("name")) == str(stream_id):
            candidates.extend(item.get("output_paths") or item.get("aspen_output_paths") or [])
    basis = metric if metric in {"mole_fraction", "mass_fraction"} else (stream or {}).get("composition_basis")
    # Prefer a declared scalar output with the exact requested basis/entity.
    # This avoids a full STR_MAIN traversal and any mass/mole substitution.
    if basis in {"mass_fraction", "mole_fraction"} and stream_id and component:
        token = "MASSFRAC" if basis == "mass_fraction" else "MOLEFRAC"
        exact = rf"\Data\Streams\{stream_id}\Output\{token}\MIXED\{component}"
        declared = list((getattr(context, "configuration", {}) or {}).get("output_paths", []))
        declared.extend(str(r.get("aspen_path") or "") for r in (getattr(context, "objectives", []) or []))
        if any(str(p).casefold() == exact.casefold() for p in declared):
            return OutputSpec(metric, stream_id, component, [exact], "scalar",
                              "kg/kg" if basis == "mass_fraction" else "mol/mol", basis)
    if metric == "purity" and basis not in {"mole_fraction", "mass_fraction"}:
        return OutputSpec(metric, stream_id, component, [], "unresolved_basis", None)
    if metric in {"mole_fraction", "mass_fraction", "purity", "recovery"} and base:
        paths = [str(x) for x in candidates if "output" in str(x).casefold()]
        if not paths:
            paths = [base + r"\Output\STR_MAIN"]
        extractor = "stream_component_" + (basis if metric == "purity" else metric)
        return OutputSpec(metric, stream_id, component, paths, extractor,
            "kg/kg" if basis == "mass_fraction" else "mol/mol" if basis == "mole_fraction" else None, basis)
    if metric in {"energy", "duty"}:
        equipment = getattr(context, "get_equipment", lambda _x: None)(stream_id) if stream_id else None
        path = str((equipment or {}).get("aspen_path") or (rf"\Data\Blocks\{stream_id}" if stream_id else ""))
        return OutputSpec(metric, stream_id, component, [path + r"\Output\DUTY"] if path else [], "scalar", None)
    return OutputSpec(metric, stream_id, component, [], "scalar", None)


def extract_output_spec(raw_outputs: Mapping[str, Any], spec: OutputSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Extract a logical metric from a complete, already-authorized result."""
    s = spec if isinstance(spec, Mapping) else spec.to_dict()
    if s.get("extractor") == "scalar":
        paths = s.get("aspen_paths") or []
        value = raw_outputs.get(paths[0]) if len(paths) == 1 else None
        value = value.get("value") if isinstance(value, Mapping) else getattr(value, "value", value)
        return {str(s["metric"]): value} if _finite(value) else {}
    stream_id, component = s.get("stream_id"), s.get("component")
    stream_data = raw_outputs.get(stream_id, raw_outputs) if stream_id else raw_outputs
    # Runners may return a complete snapshot under ``streams`` or use Aspen
    # output paths as keys.  Resolve the logical stream before extraction.
    if isinstance(stream_data, Mapping) and "streams" in stream_data and isinstance(stream_data["streams"], Mapping):
        stream_data = stream_data["streams"].get(stream_id, {}) if stream_id else stream_data["streams"]
    if not isinstance(stream_data, Mapping):
        return {}
    comps = stream_data.get("components") or stream_data.get("composition") or stream_data.get("mole_fractions") or {}
    if isinstance(comps, Mapping):
        comp = comps.get(component)
    elif isinstance(comps, (list, tuple)):
        comp = next((x for x in comps if isinstance(x, Mapping) and str(x.get("component", x.get("name"))) == str(component)), None)
    else:
        comp = None
    basis = s.get("basis") or ("mass_fraction" if s.get("metric") == "mass_fraction" else "mole_fraction" if s.get("metric") == "mole_fraction" else None)
    value = comp.get(basis, comp.get("mass_frac" if basis == "mass_fraction" else "mole_frac")) if isinstance(comp, Mapping) and basis else None
    result: dict[str, Any] = {}
    if value is not None:
        result[str(s["metric"])] = value
    if stream_data.get("flow") is not None or stream_data.get("total_mole_flow") is not None:
        result["flow"] = stream_data.get("flow", stream_data.get("total_mole_flow"))
    component_flow = None
    if isinstance(comp, Mapping):
        component_flow = comp.get("flow", comp.get("molar_flow", comp.get("component_flow")))
    if str(s.get("metric")) == "recovery" or str(s.get("extractor", "")).endswith("recovery"):
        # Recovery needs both total flow and a component flow (or an explicit
        # feed/denominator supplied by the runner).  Mole fraction alone is
        # insufficient and must produce a warning at coercion time.
        denominator = stream_data.get("feed_component_flow") or stream_data.get("input_component_flow")
        if component_flow is not None and denominator not in (None, 0):
            result["recovery"] = float(component_flow) / float(denominator)
    return result


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


@dataclass
class ParameterImpactPlan:
    equipment_id: str
    target_stream: str | None
    target_component: str | None
    metric: str
    parameters: list[dict[str, Any]] = field(default_factory=list)
    perturbation_policy: dict[str, Any] = field(default_factory=lambda: {"type": "local", "relative_steps": [0.05, 0.10]})
    max_cases: int = 20
    bounds: dict[str, dict[str, float]] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    requires_confirmation: bool = True
    diagnostics: list[str] = field(default_factory=list)
    output_spec: dict[str, Any] = field(default_factory=dict)
    selection: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.equipment_id, str) or not self.equipment_id.strip():
            raise ValueError("equipment_id is required")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("metric is required")
        if int(self.max_cases) < 1:
            raise ValueError("max_cases must be positive")
        self.max_cases = int(self.max_cases)

    def to_dict(self) -> dict[str, Any]:
        return _json({
            "equipment_id": self.equipment_id, "target_stream": self.target_stream,
            "target_component": self.target_component, "metric": self.metric,
            "parameters": self.parameters, "perturbation_policy": self.perturbation_policy,
            "max_cases": self.max_cases, "bounds": self.bounds,
            "expected_outputs": self.expected_outputs,
            "requires_confirmation": self.requires_confirmation,
            "diagnostics": self.diagnostics, "output_spec": self.output_spec,
            "selection": self.selection, "blockers": self.blockers,
        })

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ParameterImpactPlan":
        allowed = {"equipment_id", "target_stream", "target_component", "metric", "parameters", "perturbation_policy", "max_cases", "bounds", "expected_outputs", "requires_confirmation", "diagnostics", "output_spec", "selection", "blockers"}
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f"Unknown ParameterImpactPlan fields: {sorted(extra)}")
        return cls(**dict(raw))


@dataclass
class SimulationPlan:
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    base_model: str = ""
    temporary_model: str | None = None
    parameter_changes: list[dict[str, Any]] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    max_evaluations: int = 0
    timeout_seconds: float = 300.0
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    requires_confirmation: bool = True
    approved: bool = False
    output_spec: dict[str, Any] | None = None
    executable: bool = True
    adaptive_policy: dict[str, Any] = field(default_factory=dict)

    def validate_execution(self) -> None:
        if type(self.max_evaluations) is not int or self.max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer")
        if not _finite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for key in ("max_failures", "max_reference_trials", "max_shrinks"):
            value = self.adaptive_policy.get(key, 1)
            if type(value) is not int or value < (1 if key == "max_failures" else 0):
                raise ValueError(f"invalid adaptive budget: {key}")
        seconds = self.adaptive_policy.get("max_seconds", self.timeout_seconds * self.max_evaluations)
        if not _finite(seconds) or seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        parameters = self.adaptive_policy.get("parameters", [])
        if parameters:
            errors = validate_parameter_impact_plan(ParameterImpactPlan("scope", None, None, "metric", parameters))
            if errors:
                raise ValueError("invalid approved parameter scope: " + "; ".join(errors))
        for step in self.adaptive_policy.get("relative_steps", [.05, .10]):
            if not _finite(step) or step <= 0:
                raise ValueError("invalid adaptive step")

    def to_dict(self) -> dict[str, Any]:
        return _json({"plan_id": self.plan_id, "base_model": self.base_model,
                      "temporary_model": self.temporary_model, "parameter_changes": self.parameter_changes,
                      "expected_outputs": self.expected_outputs, "max_evaluations": self.max_evaluations,
                      "timeout_seconds": self.timeout_seconds, "reason": self.reason,
                      "evidence_ids": self.evidence_ids, "requires_confirmation": self.requires_confirmation,
                      "approved": self.approved, "output_spec": self.output_spec, "executable": self.executable,
                      "adaptive_policy": self.adaptive_policy})

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SimulationPlan":
        allowed = {"plan_id", "base_model", "temporary_model", "parameter_changes", "expected_outputs", "max_evaluations", "timeout_seconds", "reason", "evidence_ids", "requires_confirmation", "approved", "output_spec", "executable", "adaptive_policy"}
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f"Unknown SimulationPlan fields: {sorted(extra)}")
        return cls(**dict(raw))


def _parameter_name(item: Mapping[str, Any]) -> str:
    return str(item.get("aspen_path") or item.get("path") or item.get("name") or "")


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return int(n) if n.is_integer() else n


def parameter_catalog(context: Any, equipment_id: str) -> list[dict[str, Any]]:
    """Merge raw evidence before normalizing; explicit YAML variables authorize scope.

    Cached Input leaves need an explicit authorization/tunable flag. A negative
    authorization in any source wins. Aliases never create extra experiments.
    """
    grouped = {}
    records = []
    for name, value in (getattr(context, "variables", {}) or {}).items():
        if isinstance(value, Mapping):
            records.append((dict(value), name, True))
    records.extend((dict(row), row.get("name", ""), False)
        for row in (getattr(context, "equipment_parameters", {}) or {}).get(equipment_id, [])
        if isinstance(row, Mapping))
    for row, alias, configured in records:
        path = str(row.get("aspen_path") or row.get("path") or "")
        related = row.get("related_equipment") or []
        # The path is definitive even when related_equipment was omitted.
        if not path.casefold().startswith(f"\\data\\blocks\\{equipment_id}\\".casefold()):
            continue
        if related and equipment_id not in related:
            continue
        key = path.casefold()
        entry = grouped.setdefault(key, {"aspen_path": path, "aliases": [], "records": [], "configured": False})
        entry["configured"] |= configured
        entry["records"].append(row)
        entry["aliases"] = list(dict.fromkeys([*entry["aliases"], path, path.rsplit("\\", 1)[-1],
                                              *[v for v in (alias, row.get("name")) if v]]))
    current = getattr(context, "current_values", {}) or {}
    result = []
    for group in grouped.values():
        rows, path = group["records"], group["aspen_path"]
        def first(*keys, default=None):
            return next((row[k] for row in rows for k in keys if row.get(k) not in (None, "")), default)
        cv = current.get(path)
        baseline = first("baseline", "initial_value", "value", default=cv.get("value") if isinstance(cv, Mapping) else cv)
        denied = any(row.get("authorized") is False or row.get("tunable") is False for row in rows)
        authorized = not denied and (group["configured"] or any(row.get("authorized") is True or row.get("tunable") is True for row in rows))
        kind = str(first("value_type", "type", default="continuous")).casefold()
        integer = kind in {"int", "integer"} or any(row.get("integer") is True for row in rows)
        result.append({"name": str(first("name", default=group["aliases"][-1])), "aspen_path": path,
            "aliases": group["aliases"], "baseline": _number(baseline),
            "lower": _number(first("lower", "lower_bound", "minimum")),
            "upper": _number(first("upper", "upper_bound", "maximum")),
            "unit": first("unit", default=""), "value_type": kind, "integer": integer,
            "authorized": authorized, "tunable": authorized,
            "authorization_source": "configured_variable" if group["configured"] else "explicit_case_flag",
            "impact_targets": first("impact_targets", "related_metrics", default=[]),
            "evidence": rows,
            "evidence_ids": list(dict.fromkeys(str(e) for row in rows for e in row.get("evidence_ids", [])))})
    return result


def parameter_issues(parameter: dict[str, Any]) -> list[dict[str, str]]:
    """Typed prerequisites; these are evidence gaps, not selection questions."""
    p, issues = parameter, []
    def add(field, message):
        issues.append({"field": field, "message": f"{p['aspen_path']}: {message}"})
    path = p["aspen_path"]
    if not p["authorized"] or not _SAFE_PATH.match(path) or "\\input\\" not in path.casefold() or any(w in path.casefold() for w in _STRUCTURAL_WORDS):
        add("parameter_authorization", "缺少授权的非结构性 Aspen Input 参数")
    if not _finite(p["baseline"]):
        add("parameter_baseline", "缺少有限基准值")
    if not _finite(p["lower"]) or not _finite(p["upper"]):
        add("parameter_bounds", "缺少有限上下界")
    elif p["lower"] >= p["upper"] or (_finite(p["baseline"]) and not p["lower"] <= p["baseline"] <= p["upper"]):
        add("parameter_bounds", "边界无有效变化空间或基准值越界")
    if not str(p["unit"]).strip():
        add("parameter_unit", "缺少单位（无量纲请显式指定）")
    if p["value_type"] not in {"continuous", "float", "double", "real", "int", "integer"}:
        add("parameter_type", "不支持的离散类型；需要连续或整数参数")
    if p["integer"] and _finite(p["baseline"]) and float(p["baseline"]) != int(p["baseline"]):
        add("parameter_type", "整数参数的基准值必须为整数")
    return issues


def select_impact_parameters(plan: ParameterImpactPlan, *, decisions: list[dict[str, Any]],
                             restricted: bool = False) -> None:
    """Apply Agent relevance ranking, then the existing numerical budget checks."""
    by_path = {p["aspen_path"]: p for p in plan.parameters}
    plan.parameters = []
    rejected_issues = []
    for decision in decisions:
        p = by_path[decision["path"]]
        issues = parameter_issues(p)
        if restricted and not decision["include"] and not issues:
            issues.append({"field": "parameter_relevance", "message": f"{p['aspen_path']}: {decision['reason']}"})
        if not issues and decision["include"]:
            trial = ParameterImpactPlan(plan.equipment_id, plan.target_stream, plan.target_component,
                plan.metric, [*plan.parameters, p], max_cases=plan.max_cases)
            try:
                cases = design_perturbations(trial)
                if not any(c["changed_parameter"] == p["aspen_path"] for c in cases):
                    issues.append({"field": "parameter_bounds", "message": f"{p['aspen_path']}: 没有可行扰动工况"})
            except ValueError as exc:
                issues.append({"field": "simulation_budget", "message": str(exc)})
        chosen = decision["include"] and not issues
        plan.selection.append({"path": p["aspen_path"], "aliases": p["aliases"], "selected": chosen,
            "reason": decision["reason"], "exclusions": issues})
        if chosen:
            plan.parameters.append(p)
        else:
            rejected_issues.extend(issues)
    if not plan.parameters and by_path:
        plan.blockers.extend(rejected_issues or [{"field": "parameter_relevance",
            "message": "当前 case 证据不足以支持指标相关的安全参数选择"}])
    elif restricted and rejected_issues:
        # Do not silently narrow an explicitly requested experiment.
        plan.blockers.extend(rejected_issues)
    plan.bounds = {p["aspen_path"]: {"lower": p["lower"], "upper": p["upper"]} for p in plan.parameters}


def build_parameter_impact_plan(
    context: Any,
    *,
    equipment_id: str,
    target_stream: str | None,
    target_component: str | None,
    metric: str,
    candidate_parameters: Iterable[str] | None = None,
    max_cases: int = 20,
) -> ParameterImpactPlan:
    """Build a plan from ProcessContext snapshots without parsing BKP again."""
    selected = set(candidate_parameters or ())
    catalog = parameter_catalog(context, equipment_id)
    parameters = [p for p in catalog if not selected or selected.intersection(p["aliases"])]
    output_spec = resolve_output_spec(context, metric=metric, stream_id=target_stream, component=target_component)
    return ParameterImpactPlan(equipment_id, target_stream, target_component, metric, parameters,
        max_cases=max_cases, bounds={p["aspen_path"]: {"lower": p["lower"], "upper": p["upper"]}
            for p in parameters if p["lower"] is not None and p["upper"] is not None},
        expected_outputs=list(output_spec.aspen_paths), output_spec=output_spec.to_dict())


def validate_parameter_impact_plan(plan: ParameterImpactPlan, *, budget: int | None = None,
                                   allowed_paths: Iterable[str] | None = None) -> list[str]:
    errors: list[str] = []
    if plan.output_spec.get("extractor") == "unresolved_basis":
        errors.append("purity 缺少质量分数或摩尔分数基准；请明确指标基准")
    if not plan.parameters:
        errors.append("no parameters selected for perturbation")
    if budget is not None and plan.max_cases > int(budget):
        errors.append("plan exceeds total simulation budget")
    allow = set(allowed_paths or ())
    for item in plan.parameters:
        path = _parameter_name(item)
        if item.get("authorized") is False or item.get("tunable") is False:
            errors.append(f"parameter is not an authorized tunable Input: {path}")
        if not path or not _SAFE_PATH.match(path):
            errors.append(f"unsafe Aspen path: {path}")
        if allow and path not in allow:
            errors.append(f"path is not authorized: {path}")
        if "\\input\\" not in path.casefold():
            errors.append(f"parameter must reference an Aspen Input node: {path}")
        if any(word in path.casefold() for word in _STRUCTURAL_WORDS):
            errors.append(f"structural parameter is not allowed: {path}")
        baseline, lo, hi = item.get("baseline"), item.get("lower"), item.get("upper")
        if any(x is None for x in (baseline, lo, hi)):
            errors.append(f"missing baseline/bounds: {path}")
            continue
        try:
            if not math.isfinite(float(baseline)) or not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(lo) > float(hi) or not float(lo) <= float(baseline) <= float(hi):
                errors.append(f"invalid bounds: {path}")
        except (TypeError, ValueError):
            errors.append(f"non-numeric bounds: {path}")
        if item.get("integer") and _finite(baseline) and float(baseline) != int(float(baseline)):
            errors.append(f"integer baseline required: {path}")
        if not str(item.get("unit") or "").strip():
            errors.append(f"unit is required: {path}")
    return errors


def design_perturbations(plan: ParameterImpactPlan, *, relative_steps: Iterable[float] | None = None) -> list[dict[str, Any]]:
    """Create a bounded baseline plus one-factor local neighborhood."""
    errors = validate_parameter_impact_plan(plan)
    if errors:
        raise ValueError("invalid parameter impact plan: " + "; ".join(errors))
    steps = [abs(float(x)) for x in (relative_steps or plan.perturbation_policy.get("relative_steps", [0.05, 0.10]))]
    steps = [x for x in steps if x > 0 and math.isfinite(x)]
    cases: list[dict[str, Any]] = [{"case_id": "baseline", "inputs": {p["aspen_path"]: p["baseline"] for p in plan.parameters}, "changed_parameter": None, "delta": 0.0}]
    for p in plan.parameters:
        base, lo, hi = float(p["baseline"]), float(p["lower"]), float(p["upper"])
        if p.get("integer"):
            integer_values = [int(base)]
            for value in (int(base) - 1, int(base) + 1):
                if math.ceil(lo) <= value <= math.floor(hi):
                    integer_values.append(value)
            for value in dict.fromkeys(integer_values):
                if value == p["baseline"]:
                    continue
                inputs = {q["aspen_path"]: q["baseline"] for q in plan.parameters}
                inputs[p["aspen_path"]] = value
                cases.append({"case_id": uuid.uuid4().hex, "inputs": inputs,
                              "changed_parameter": p["aspen_path"], "delta": value - p["baseline"]})
            continue
        for step in steps:
            for sign in (-1.0, 1.0):
                value = base * (1.0 + sign * step) if base else base + sign * step * max(abs(hi - lo), 1.0)
                value = max(lo, min(hi, value))
                if p.get("integer"):
                    value = int(round(value))
                    value = max(math.ceil(lo), min(math.floor(hi), value))
                if value == p["baseline"]:
                    continue
                inputs = {q["aspen_path"]: q["baseline"] for q in plan.parameters}
                inputs[p["aspen_path"]] = value
                cases.append({"case_id": uuid.uuid4().hex, "inputs": inputs,
                              "changed_parameter": p["aspen_path"], "delta": value - p["baseline"]})
    # Clipping different step sizes can produce identical inputs. Count the
    # distinct design points before applying the evaluation budget.
    unique = {}
    for case in cases:
        unique.setdefault(tuple(sorted(case["inputs"].items())), case)
    cases = list(unique.values())
    if len(cases) > plan.max_cases:
        raise ValueError(f"perturbation design requires {len(cases)} cases, exceeds max_cases={plan.max_cases}")
    return cases


def make_simulation_plan(impact: ParameterImpactPlan, *, base_model: str,
                         evidence_ids: Iterable[str] = (), timeout_seconds: float = 300.0) -> SimulationPlan:
    changes = design_perturbations(impact)
    model_ok = bool(base_model) and Path(str(base_model)).is_file()
    return SimulationPlan(base_model=str(base_model), parameter_changes=changes,
                          expected_outputs=list(impact.expected_outputs), max_evaluations=len(changes),
                          timeout_seconds=float(timeout_seconds), reason=f"Controlled local parameter impact analysis for {impact.equipment_id}",
                          evidence_ids=list(evidence_ids), requires_confirmation=True,
                          output_spec=dict(impact.output_spec or {"metric": impact.metric}),
                          adaptive_policy={"parameters": [dict(p) for p in impact.parameters],
                              "max_failures": impact.perturbation_policy.get("max_failures", 4),
                              "max_reference_trials": impact.perturbation_policy.get("max_reference_trials", 3),
                              "max_seconds": impact.perturbation_policy.get("max_seconds", timeout_seconds * len(changes)),
                              "max_shrinks": impact.perturbation_policy.get("max_shrinks", 2),
                              "relative_steps": impact.perturbation_policy.get("relative_steps", [0.05, 0.10])},
                          executable=bool(impact.expected_outputs) and model_ok and not impact.blockers)


class AspenSandboxRunner:
    """Execute approved plans against a temporary model copy.

    ``run_case`` is injected in tests and deployments that already own an
    Aspen driver.  A real runner can inject a callable which opens the copied
    BKP, runs one case and returns a ProcessCase or a serializable mapping.
    """
    def __init__(self, run_case: Callable[..., Any] | None = None, *, driver_factory: Callable[..., Any] | None = None,
                 allowed_paths: Iterable[str] | None = None, require_path_whitelist: bool = True,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None,
                 progress_interval: float = 15.0, cleanup_timeout: float = 5.0):
        self.run_case_fn = run_case
        self.driver_factory = driver_factory
        self.allowed_paths = set(allowed_paths or ())
        self.require_path_whitelist = require_path_whitelist
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval
        if not _finite(cleanup_timeout) or cleanup_timeout <= 0:
            raise ValueError("cleanup_timeout must be finite and positive")
        self.cleanup_timeout = cleanup_timeout
        self._quarantined = False

    def _notify(self, status: str, completed: int, total: int, started: float) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback({"status": status, "completed": completed, "total": total,
                                    "elapsed_seconds": max(0.0, time.monotonic() - started)})
        except Exception:
            # A presentation failure must not invalidate an Aspen case.
            logging.getLogger(__name__).exception("回测进度回调失败")

    def _invoke_with_progress(self, model, inputs, plan, completed, started):
        # ExecutionUnit owns the invocation thread; AspenDriver separately
        # owns its COM STA. Heartbeat stops when the bounded wait returns.
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(self.progress_interval):
                self._notify("running", completed, len(plan.parameter_changes), started)
        thread = None
        if self.progress_callback is not None and self.progress_interval > 0:
            thread = threading.Thread(target=heartbeat, name="pao-progress", daemon=True)
            thread.start()
        try:
            from src.aspen_driver.lifecycle import ExecutionUnit
            unit = ExecutionUnit(str(model))
            self._unit = unit
            try:
                return unit.invoke(lambda: self._invoke(model, inputs, plan),
                    timeout=self._case_timeout, cleanup_timeout=self.cleanup_timeout,
                    notify=lambda status: self._notify(status, completed, len(plan.parameter_changes), started))
            finally:
                if unit.timed_out or unit.execution_state == "unknown" or unit.cleanup_state != "completed":
                    self._quarantined = True
        finally:
            stop.set()
            if thread is not None:
                thread.join()

    def run(self, plan: SimulationPlan) -> list[ProcessCase]:
        from src.agents.actions.authorization import require_authorization
        grant = require_authorization()
        if self._quarantined:
            raise RuntimeError("execution unit is quarantined; create a new runner after inspecting retained resources")
        if self.run_case_fn is None and self.driver_factory is None:
            raise RuntimeError("Aspen execution capability unavailable: no driver_factory or run_case configured")
        self.last_report = {}
        if not plan.approved:
            raise PermissionError("simulation plan requires ActionGateway approval")
        if not plan.executable:
            raise RuntimeError("simulation plan is not executable: no authorized Aspen output path")
        plan.validate_execution()
        if not plan.base_model or not Path(plan.base_model).is_file():
            self.last_report = {"stop_reason": "global_model", "actual_case_count": 0,
                "skipped_case_count": len(plan.parameter_changes), "diagnostics": ["base Aspen model is unavailable"]}
            return []
        if plan.max_evaluations and len(plan.parameter_changes) > plan.max_evaluations:
            from .impact_evidence import new_plan
            self.last_report = {"stop_reason": "requires_reapproval", "actual_case_count": 0,
                "skipped_case_count": len(plan.parameter_changes),
                "reapproval_plan": new_plan(plan, plan.parameter_changes, "Design exceeds approved case budget")}
            return []
        if self.require_path_whitelist and not self.allowed_paths:
            raise PermissionError("Aspen path whitelist is required")
        if self.require_path_whitelist:
            bad_outputs = [p for p in plan.expected_outputs
                           if not _SAFE_PATH.match(str(p)) or "\\output\\" not in str(p).casefold()]
            if bad_outputs:
                raise PermissionError("unauthorized Aspen output path: " + ", ".join(map(str, bad_outputs)))
        try:
            temp_root = Path(tempfile.mkdtemp(prefix="pao_aspen_"))
        except PermissionError:
            # Some locked-down Windows hosts deny the process temp root.  A
            # sibling directory is still isolated from the original BKP and
            # remains inside the caller's authorized workspace.
            temp_root = Path(plan.base_model).resolve().parent / f".pao_aspen_{uuid.uuid4().hex}"
            temp_root.mkdir(parents=False, exist_ok=False)
        temp_model = temp_root / Path(plan.base_model).name
        from src.agents.actions.authorization import file_digest, ExecutionBoundaryError
        snapshot = temp_root / ("snapshot" + temp_model.suffix)
        try:
            shutil.copy2(plan.base_model, temp_model)
            if file_digest(temp_model) != getattr(plan, "_approved_model_hash", None):
                raise ExecutionBoundaryError("model changed while creating batch snapshot; approve again")
            shutil.copy2(temp_model, snapshot)
        except Exception as exc:
            cleanup_errors = []
            try:
                shutil.rmtree(temp_root)
            except OSError as cleanup:
                cleanup_errors.append(str(cleanup))
            self.last_report = {"primary_error": str(exc), "resources": {"temporary_directory": str(temp_root),
                "files_state": "retained" if cleanup_errors else "removed", "cleanup_errors": cleanup_errors}}
            grant.audit.record(grant.action_id, "execution", status="failed", payload=self.last_report)
            raise
        plan.temporary_model = str(temp_model)
        self._batch_source = {"source_path": str(Path(plan.base_model).resolve()),
                              "sha256": file_digest(snapshot), "execution_id": grant.action_id,
                              "batch_id": plan.plan_id, "schema_version": grant.schema.identity.model_version}
        from src.agents.actions.authorization import schema_digest, digest
        self._batch_source.update(schema_digest=schema_digest(grant.schema), plan_digest=digest(plan.to_dict()),
            config_digest=file_digest(grant.schema.config_path) if grant.schema.config_path else None)
        grant.execution_model = temp_model.resolve()
        results: list[ProcessCase] = []
        started = time.monotonic()
        total = len(plan.parameter_changes)
        try:
            from .impact_evidence import execute_design, assess, usable
            def invoke(change, i, timeout):
                inputs = dict(change.get("inputs") or {})
                if any(p not in self.allowed_paths for p in inputs) and self.require_path_whitelist:
                    raise PermissionError("unauthorized Aspen input path")
                self._notify("started", i, total, started)
                # Restore the approved model before every run; no failed-run warm start.
                shutil.copy2(snapshot, temp_model)
                from src.agents.actions.authorization import file_digest, ExecutionBoundaryError
                if file_digest(temp_model) != getattr(plan, "_approved_model_hash", None):
                    raise ExecutionBoundaryError("model changed while copying; preview and approve again")
                self._run_id = uuid.uuid4().hex
                self._case_timeout = timeout
                self._evidence = {}
                self._unit = None
                failures_before = grant.failures
                try:
                    with grant.evaluation(inputs):
                        raw = self._invoke_with_progress(temp_model, inputs, plan, i, started)
                    case = self._coerce(raw, change, temp_model, i, plan=plan)
                    evidence = raw.get("evidence", {}) if isinstance(raw, Mapping) else self._evidence
                except PermissionError:
                    raise
                except Exception as exc:
                    case = self._failed(change, f"Aspen run failed: {type(exc).__name__}: {exc}", model_path=temp_model)
                    evidence = {"failure_kind": "global_model"}
                unit = self._unit
                lifecycle = unit.snapshot() if unit else {}
                case.impact_outputs["_lifecycle"] = lifecycle
                if unit and (unit.timed_out or unit.execution_state == "unknown" or unit.cleanup_state != "completed"):
                    grant.execution_unknown = True
                    case.warnings.extend(unit.cleanup_errors)
                    if unit.timed_out or unit.execution_state == "unknown":
                        evidence = {**evidence, "failure_kind": "execution_unknown", "fresh": False}
                        case.converged = None
                case.run_id = self._run_id
                case.session_id = getattr(grant, "session_id", "")
                case.impact_outputs["_source"] = dict(self._batch_source)
                # A design ID is unique within this approved execution, including baseline.
                case.case_id = f"{plan.plan_id}:{change['case_id']}"
                assess(case, evidence, (plan.output_spec or {}).get("metric"), self._run_id)
                if not usable(case, (plan.output_spec or {}).get("metric")) and grant.failures == failures_before:
                    grant.failures += 1
                self._notify("completed" if usable(case, (plan.output_spec or {}).get("metric")) else "failed", i + 1, total, started)
                return case
            results, self.last_report = execute_design(plan, invoke, getattr(self, "case_callback", None))
        finally:
            # The temporary copy and all driver/COM resources are owned by this
            # scope; no COM object is returned or placed in a checkpoint.
            cleanup_errors = []
            if not self._quarantined:
                try:
                    shutil.rmtree(temp_root)
                except OSError as exc:
                    cleanup_errors.append(str(exc))
            self.last_report["resources"] = {"temporary_directory": str(temp_root),
                "files_state": "retained" if self._quarantined or cleanup_errors else "removed",
                "cleanup_errors": cleanup_errors, "source": self._batch_source}
            if self._quarantined or cleanup_errors:
                self._notify("cleanup_incomplete", len(results), total, started)
        if len(results) == total and not self._quarantined and not cleanup_errors:
            self._notify("finished", len(results), total, started)
        else:
            self._notify("stopped", len(results), total, started)
        return results

    def _invoke(self, model: Path, inputs: dict[str, Any], plan: SimulationPlan) -> Any:
        if self.run_case_fn is not None:
            return self.run_case_fn(model_path=str(model), inputs=inputs, expected_outputs=plan.expected_outputs, timeout=self._case_timeout, run_id=self._run_id)
        if self.driver_factory is None:
            raise RuntimeError("Aspen is unavailable: inject run_case or driver_factory")
        driver = self.driver_factory(str(model))
        unit = self._unit
        unit.driver = driver
        driver.stop_requested = unit.stop_requested
        try:
            driver.connect(); driver.open(str(model))
            from src.workflows.run_case import RunCaseConfig, run_case
            # Composition is a subtree, not a scalar Value node. Let the
            # stream extractor read it, while retaining full-flow convergence checks.
            spec = plan.output_spec or {}
            stream_id = spec.get("stream_id")
            composition = str(spec.get("extractor", "")).startswith("stream_component_")
            cfg = RunCaseConfig(
                output_paths=[] if composition else list(plan.expected_outputs),
                extract_streams=[stream_id] if composition and stream_id else None,
                timeout=self._case_timeout,
            )
            invocation_started = time.monotonic()
            case = run_case(driver, inputs, cfg)
            sim = case.sim_result
            statuses = [s.to_dict() for s in getattr(sim, "block_statuses", [])]
            target_status = next((s for s in statuses if s["name"] == stream_id), {})
            flags = target_status.get("status_flags", [])
            target = "errors" if "ERRORS" in flags else "unknown" if any(f in flags for f in ("INCOMPAT", "INACCESS", "NO_RESULTS", "UNKNOWN")) else "success" if "SUCCESS" in flags else "warnings" if "WARNINGS" in flags else "unknown"
            if spec.get("extractor") == "scalar" and sim and sim.success:
                target = "success"  # scalar output validated by Runner status/readback checks
            fresh = bool(sim and sim.success and sim.result_time >= invocation_started
                and sim.source_filepath and Path(sim.source_filepath).resolve() == model.resolve()
                and sim.mutation_snapshot == driver.mutation_count
                and sim.actual_inputs == inputs and sim.input_verifications
                and all(v.match for v in sim.input_verifications))
            status = getattr(getattr(sim, "status", None), "value", "unknown")
            self._evidence = {"run_id": self._run_id, "fresh": fresh,
                "result_time": getattr(sim, "result_time", None),
                "mutation_snapshot": getattr(sim, "mutation_snapshot", None),
                "actual_inputs": getattr(sim, "actual_inputs", {}),
                "model_path": str(model),
                "target_status": target, "target_verified": fresh and target in {"success", "warnings"},
                "block_statuses": statuses,
                "failure_kind": "global_model" if status in {"run_failed", "write_failed", "status_unavailable", "inaccess", "incompat"} else
                    "local_nonconvergence" if status in {"errors", "no_results"} else None}
            from src.agents.actions.authorization import _evaluation
            frame = _evaluation.get()
            self._evidence["execution_state"] = "attempted" if frame and frame["run_started"] else "not_executed"
            return case
        finally:
            # ExecutionUnit owns bounded close, including connect/open failures.
            pass

    @staticmethod
    def _coerce(raw: Any, change: Mapping[str, Any], model: Path, iteration: int, *, plan: SimulationPlan | None = None) -> ProcessCase:
        if isinstance(raw, ProcessCase):
            raw.design_vars = dict(change.get("inputs") or raw.design_vars)
            raw.source_filepath = model
            raw.impact_inputs = dict(change.get("inputs") or {})
            raw.model_path = str(model)
            raw.timestamp = datetime.now(timezone.utc).isoformat()
            raw.converged = raw.simulation_valid
            raw.result_source = raw.result_source or "aspen_sandbox"
            # The plan, never an arbitrary first stream, owns the target.
            spec = (plan.output_spec if plan else None) or {}
            if spec.get("metric"):
                raw.impact_outputs.pop(spec["metric"], None)
            stream_data = {k: v.to_dict() for k, v in raw.streams.items()}
            raw.impact_outputs["streams"] = stream_data
            if spec:
                raw.impact_outputs.update(extract_output_spec(
                    (getattr(raw.sim_result, "outputs", {}) or {}) if spec.get("extractor") == "scalar"
                    else {"streams": stream_data}, spec))
                metric = spec.get("metric")
                if not _finite(raw.impact_outputs.get(metric)):
                    raw.warnings.append(f"目标指标不可用：{spec.get('stream_id')}/{spec.get('component')}/{metric}")
            raw.iteration = iteration
            sim = raw.sim_result
            raw.errors = list(raw.errors or ([sim.error] if sim and sim.error else
                                           [raw.notes] if raw.notes and not raw.converged else []))
            raw.warnings = list(raw.warnings or []) + list(getattr(sim, "warnings", []) or [])
            return raw
        data = dict(raw or {}) if isinstance(raw, Mapping) else {}
        converged = bool(data.get("converged", data.get("success", False)))
        case = ProcessCase(case_id=str(data.get("case_id") or change.get("case_id") or uuid.uuid4().hex), iteration=iteration,
                           status=CaseStatus.SUCCESS if converged else CaseStatus.SIM_FAILED,
                           design_vars=dict(change.get("inputs") or {}), notes=str(data.get("error") or ""), source_filepath=model)
        case.impact_inputs = dict(change.get("inputs") or {})
        case.impact_outputs = dict(data.get("outputs") or {})
        # The runner may return a complete stream snapshot under
        # ``raw_outputs``.  Extract logical metrics only after the whitelist
        # read, so values such as ``0325.MGN.mole_fraction`` never become
        # Aspen paths.
        spec = (plan.output_spec if plan else None) or {}
        extractor = spec.get("extractor") if isinstance(spec, Mapping) else None
        raw_outputs = data.get("raw_outputs") or data.get("stream_results") or {}
        if extractor and isinstance(raw_outputs, Mapping):
            extracted = extract_output_spec(raw_outputs, spec)
            case.impact_outputs.update({k: v for k, v in extracted.items() if k not in case.impact_outputs})
            if extractor.endswith("recovery") and "recovery" not in extracted:
                case.warnings.append("缺少流量或组分流量，禁止计算 recovery")
        if data.get("target_stream") is not None:
            case.impact_outputs.setdefault("target_stream", data.get("target_stream"))
        if data.get("target_component") is not None:
            case.impact_outputs.setdefault("target_component", data.get("target_component"))
        case.converged = converged
        case.errors = list(data.get("errors") or ([] if converged else [str(data.get("error") or "simulation failed")]))
        case.warnings = list(data.get("warnings") or [])
        if extractor and extractor.endswith("recovery") and "recovery" not in case.impact_outputs:
            warning = "缺少流量或组分流量信息，禁止计算 recovery"
            if warning not in case.warnings:
                case.warnings.append(warning)
        case.result_source = str(data.get("source") or "aspen_sandbox")
        case.model_path = str(model)
        case.timestamp = datetime.now(timezone.utc).isoformat()
        return case

    @staticmethod
    def _failed(change: Mapping[str, Any], error: str, *, model_path: Path | None = None) -> ProcessCase:
        case = ProcessCase(case_id=str(change.get("case_id") or uuid.uuid4().hex), status=CaseStatus.SIM_FAILED,
                           design_vars=dict(change.get("inputs") or {}), notes=error)
        case.impact_inputs = dict(change.get("inputs") or {}); case.impact_outputs = {}
        case.converged = False; case.errors = [error]; case.warnings = []
        case.result_source = "aspen_sandbox"; case.model_path = str(model_path) if model_path else None
        case.timestamp = datetime.now(timezone.utc).isoformat()
        return case


class FakeAspenSandboxRunner(AspenSandboxRunner):
    """Deterministic fake runner; it never claims to be a real Aspen result."""
    simulated = True
    def __init__(self, evaluator: Callable[[dict[str, Any]], Mapping[str, Any]]):
        def fake(**kw):
            data = dict(evaluator(dict(kw["inputs"])), source="fake_aspen")
            success = data.get("converged", data.get("success", bool(data.get("outputs"))))
            data["success"] = success
            data.setdefault("evidence", {"run_id": kw["run_id"], "fresh": True,
                "target_status": "success" if success else "errors", "target_verified": bool(success),
                "block_statuses": data.get("block_statuses", {}), "failure_kind": data.get("failure_kind")})
            return data
        super().__init__(run_case=fake, require_path_whitelist=False)



def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class SensitivityAnalyzer:
    """Analyze each one-factor neighborhood against its valid baseline."""
    def analyze(self, results, *, metric, parameters, parameter_specs=None):
        from .impact_evidence import analyze
        return analyze(results, metric, parameters, parameter_specs)


def _slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x, y = list(xs), list(ys)
    if len(x) < 2: return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    den = sum((v - mx) ** 2 for v in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den if den else 0.0


def _pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    x, y = list(xs), list(ys)
    if len(x) < 2: return None
    mx, my = statistics.mean(x), statistics.mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / den if den else 0.0


def _spearman(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    x, y = list(xs), list(ys)
    if len(x) < 2: return None
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i]); r = [0.0] * len(values)
        for rank, index in enumerate(order): r[index] = float(rank)
        return r
    return _pearson(ranks(x), ranks(y))


__all__ = ["OutputSpec", "resolve_output_spec", "extract_output_spec", "ParameterImpactPlan", "SimulationPlan", "build_parameter_impact_plan", "validate_parameter_impact_plan", "design_perturbations", "make_simulation_plan", "AspenSandboxRunner", "FakeAspenSandboxRunner", "SensitivityAnalyzer"]
