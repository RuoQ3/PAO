"""Controlled, read-only-by-default Aspen parameter impact analysis.

This module deliberately keeps planning, validation, execution and analysis
separate.  A plan is data only; an ``AspenSandboxRunner`` refuses to execute
until its plan is explicitly approved by the caller (normally ActionGateway).
The module has no dependency on a particular case, block or component name.
"""
from __future__ import annotations

import json
import math
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

    def to_dict(self) -> dict[str, Any]:
        return _json({"metric": self.metric, "stream_id": self.stream_id,
                      "component": self.component, "aspen_paths": self.aspen_paths,
                      "extractor": self.extractor, "unit": self.unit})


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
    if metric in {"mole_fraction", "purity", "recovery"} and base:
        paths = [str(x) for x in candidates if "output" in str(x).casefold()]
        if not paths:
            paths = [base + r"\Output\STR_MAIN"]
        extractor = "stream_component_mole_fraction" if metric == "mole_fraction" else ("stream_component_purity" if metric == "purity" else "stream_component_recovery")
        return OutputSpec(metric, stream_id, component, paths, extractor, "mol/mol" if metric != "recovery" else None)
    if metric in {"energy", "duty"}:
        equipment = getattr(context, "get_equipment", lambda _x: None)(stream_id) if stream_id else None
        path = str((equipment or {}).get("aspen_path") or (rf"\Data\Blocks\{stream_id}" if stream_id else ""))
        return OutputSpec(metric, stream_id, component, [path + r"\Output\DUTY"] if path else [], "scalar", None)
    return OutputSpec(metric, stream_id, component, [], "scalar", None)


def extract_output_spec(raw_outputs: Mapping[str, Any], spec: OutputSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Extract a logical metric from a complete, already-authorized result."""
    s = spec if isinstance(spec, Mapping) else spec.to_dict()
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
    value = comp.get("mole_fraction", comp.get("mole_frac", comp.get("value"))) if isinstance(comp, Mapping) else comp
    result: dict[str, Any] = {}
    if value is not None:
        result["mole_fraction"] = value
        result["purity"] = value
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
        })

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ParameterImpactPlan":
        allowed = {"equipment_id", "target_stream", "target_component", "metric", "parameters", "perturbation_policy", "max_cases", "bounds", "expected_outputs", "requires_confirmation", "diagnostics", "output_spec"}
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

    def to_dict(self) -> dict[str, Any]:
        return _json({"plan_id": self.plan_id, "base_model": self.base_model,
                      "temporary_model": self.temporary_model, "parameter_changes": self.parameter_changes,
                      "expected_outputs": self.expected_outputs, "max_evaluations": self.max_evaluations,
                      "timeout_seconds": self.timeout_seconds, "reason": self.reason,
                      "evidence_ids": self.evidence_ids, "requires_confirmation": self.requires_confirmation,
                      "approved": self.approved, "output_spec": self.output_spec, "executable": self.executable})

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SimulationPlan":
        allowed = {"plan_id", "base_model", "temporary_model", "parameter_changes", "expected_outputs", "max_evaluations", "timeout_seconds", "reason", "evidence_ids", "requires_confirmation", "approved", "output_spec", "executable"}
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
    selected = {str(x) for x in candidate_parameters or ()}
    raw: list[dict[str, Any]] = []
    # Preserve the documented source precedence: explicit process variables
    # override equipment/catalog snapshots, while still filling missing
    # metadata from lower-priority records.
    for name, value in (getattr(context, "variables", {}) or {}).items():
        item = dict(value) if isinstance(value, Mapping) else {"value": value}
        item.setdefault("name", name)
        related = {str(x) for x in item.get("related_equipment", []) or ()}
        if not related or equipment_id in related:
            raw.append(item)
    grouped = getattr(context, "equipment_parameters", {}) or {}
    raw.extend(dict(x) for x in grouped.get(equipment_id, []) if isinstance(x, Mapping))
    current = getattr(context, "current_values", {}) or {}
    params: list[dict[str, Any]] = []
    bounds: dict[str, dict[str, float]] = {}
    diagnostics: list[str] = []
    seen: set[str] = set()
    by_path: dict[str, dict[str, Any]] = {}
    for item in raw:
        path = _parameter_name(item)
        if not path or (selected and path not in selected and str(item.get("name")) not in selected):
            continue
        if path in by_path:
            # Catalog and YAML can describe the same node.  Preserve the
            # richer record and fill missing bounds/unit/type fields.
            merged = by_path[path]
            for key, value in item.items():
                if merged.get(key) in (None, "") and value not in (None, ""):
                    merged[key] = value
            continue
        baseline = item.get("baseline", item.get("initial_value", item.get("value")))
        if baseline is None and path in current:
            cv = current[path]
            baseline = cv.get("value") if isinstance(cv, Mapping) else cv
        lo = item.get("lower", item.get("lower_bound", item.get("minimum")))
        hi = item.get("upper", item.get("upper_bound", item.get("maximum")))
        b, l, h = _number(baseline), _number(lo), _number(hi)
        entry = {"name": str(item.get("name") or path.rsplit("\\", 1)[-1]), "aspen_path": path,
                 "baseline": b, "lower": l, "upper": h, "unit": item.get("unit", ""),
                 "source": item.get("source", "context"), "confidence": item.get("confidence", 1.0),
                 "value_type": item.get("value_type", item.get("type", "float")),
                 "integer": bool(item.get("integer", False) or str(item.get("type", "")).casefold() in {"int", "integer"} or str(item.get("value_type", "")).casefold() == "int"),
                 "evidence_ids": list(item.get("evidence_ids") or [f"{item.get('source', 'context')}:{path}"])}
        entry["authorized"] = bool(item.get("authorized", item.get("tunable", True)))
        entry["tunable"] = bool(item.get("tunable", item.get("authorized", True)))
        by_path[path] = entry
        params.append(entry)
        if not entry["authorized"] or not entry["tunable"]:
            diagnostics.append(f"parameter is not an authorized tunable Input: {path}")
        if b is None or l is None or h is None:
            diagnostics.append(f"{path}: baseline and lower/upper bounds are required")
        else:
            bounds[path] = {"lower": float(l), "upper": float(h)}
    if not params:
        diagnostics.append(f"No authorized tunable parameters found for equipment {equipment_id}")
    output_spec = resolve_output_spec(context, metric=metric, stream_id=target_stream, component=target_component)
    expected = list(output_spec.aspen_paths)
    if not expected:
        diagnostics.append(f"逻辑指标 {metric} 没有足够的 Aspen Output 路径；计划不可执行")
    diagnostics = list(diagnostics)
    if not expected:
        diagnostics.append(f"逻辑指标 {metric} 没有足够的 Aspen Output 路径；计划不可执行")
    return ParameterImpactPlan(equipment_id, target_stream, target_component, metric, params,
                               max_cases=max_cases, bounds=bounds, expected_outputs=expected,
                               diagnostics=diagnostics, output_spec=output_spec.to_dict())


def validate_parameter_impact_plan(plan: ParameterImpactPlan, *, budget: int | None = None,
                                   allowed_paths: Iterable[str] | None = None) -> list[str]:
    errors: list[str] = []
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
        if item.get("integer") and float(baseline) != int(float(baseline)):
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
                          output_spec=dict(impact.output_spec or {}),
                          executable=bool(impact.expected_outputs) and model_ok)


class AspenSandboxRunner:
    """Execute approved plans against a temporary model copy.

    ``run_case`` is injected in tests and deployments that already own an
    Aspen driver.  A real runner can inject a callable which opens the copied
    BKP, runs one case and returns a ProcessCase or a serializable mapping.
    """
    def __init__(self, run_case: Callable[..., Any] | None = None, *, driver_factory: Callable[..., Any] | None = None,
                 allowed_paths: Iterable[str] | None = None, require_path_whitelist: bool = True):
        self.run_case_fn = run_case
        self.driver_factory = driver_factory
        self.allowed_paths = set(allowed_paths or ())
        self.require_path_whitelist = require_path_whitelist

    def run(self, plan: SimulationPlan) -> list[ProcessCase]:
        if plan.requires_confirmation and not plan.approved:
            raise PermissionError("simulation plan requires ActionGateway approval")
        if not plan.executable:
            raise RuntimeError("simulation plan is not executable: no authorized Aspen output path")
        if not plan.base_model or not Path(plan.base_model).is_file():
            return [self._failed(c, "base Aspen model is unavailable") for c in plan.parameter_changes]
        if plan.max_evaluations and len(plan.parameter_changes) > plan.max_evaluations:
            raise ValueError("simulation plan exceeds max_evaluations")
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
        try:
            shutil.copy2(plan.base_model, temp_model)
        except PermissionError:
            shutil.rmtree(temp_root, ignore_errors=True)
            temp_root = Path(plan.base_model).resolve().parent / f".pao_aspen_{uuid.uuid4().hex}"
            temp_root.mkdir(parents=False, exist_ok=False)
            temp_model = temp_root / Path(plan.base_model).name
            shutil.copy2(plan.base_model, temp_model)
        plan.temporary_model = str(temp_model)
        results: list[ProcessCase] = []
        try:
            for i, change in enumerate(plan.parameter_changes):
                inputs = dict(change.get("inputs") or {})
                unauthorized = [p for p in inputs if self.allowed_paths and p not in self.allowed_paths]
                if unauthorized:
                    results.append(self._failed(change, "unauthorized Aspen path: " + ", ".join(unauthorized), model_path=temp_model))
                    continue
                try:
                    raw = self._invoke(temp_model, inputs, plan)
                    results.append(self._coerce(raw, change, temp_model, i, plan=plan))
                except Exception as exc:
                    results.append(self._failed(change, f"Aspen run failed: {type(exc).__name__}: {exc}", model_path=temp_model))
        finally:
            # The temporary copy and all driver/COM resources are owned by this
            # scope; no COM object is returned or placed in a checkpoint.
            shutil.rmtree(temp_root, ignore_errors=True)
        return results

    def _invoke(self, model: Path, inputs: dict[str, Any], plan: SimulationPlan) -> Any:
        if self.run_case_fn is not None:
            return self.run_case_fn(model_path=str(model), inputs=inputs, expected_outputs=plan.expected_outputs, timeout=plan.timeout_seconds)
        if self.driver_factory is None:
            raise RuntimeError("Aspen is unavailable: inject run_case or driver_factory")
        driver = self.driver_factory(str(model))
        try:
            driver.connect(); driver.open(str(model))
            from src.workflows.run_case import RunCaseConfig, run_case
            cfg = RunCaseConfig(output_paths=list(plan.expected_outputs), timeout=plan.timeout_seconds)
            return run_case(driver, inputs, cfg)
        finally:
            for name in ("disconnect", "close"):
                fn = getattr(driver, name, None)
                if callable(fn):
                    try: fn()
                    except Exception: pass

    @staticmethod
    def _coerce(raw: Any, change: Mapping[str, Any], model: Path, iteration: int, *, plan: SimulationPlan | None = None) -> ProcessCase:
        if isinstance(raw, ProcessCase):
            raw.design_vars = dict(change.get("inputs") or raw.design_vars)
            raw.source_filepath = model
            raw.impact_inputs = dict(change.get("inputs") or {})
            raw.model_path = str(model)
            raw.timestamp = datetime.now(timezone.utc).isoformat()
            raw.converged = raw.status in (CaseStatus.SUCCESS, CaseStatus.WARNINGS)
            raw.result_source = raw.result_source or "aspen_sandbox"
            # A ProcessCase from the legacy run_case path has its stream
            # snapshot available; expose a minimal metric map for analysis.
            if not raw.impact_outputs:
                stream_data = {k: v.to_dict() for k, v in raw.streams.items()}
                raw.impact_outputs = {"streams": stream_data}
                if raw.streams:
                    stream = raw.streams.get(getattr(raw, "target_stream", None)) or next(iter(raw.streams.values()))
                    components = getattr(stream, "components", []) or []
                    raw.impact_outputs.setdefault("flow", getattr(stream, "total_mole_flow", None))
                    target_component = getattr(raw, "target_component", None)
                    if target_component:
                        for component in components:
                            if getattr(component, "component", None) == target_component:
                                raw.impact_outputs.setdefault("mole_fraction", getattr(component, "mole_frac", None))
            raw.errors = list(raw.errors or ([raw.notes] if raw.notes and not raw.converged else []))
            raw.warnings = list(raw.warnings or [])
            return raw
        data = dict(raw or {}) if isinstance(raw, Mapping) else {}
        converged = bool(data.get("converged", data.get("success", bool(data.get("outputs")))))
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
    def __init__(self, evaluator: Callable[[dict[str, Any]], Mapping[str, Any]]):
        super().__init__(run_case=lambda **kw: dict(evaluator(dict(kw["inputs"])), source="fake_aspen"),
                         require_path_whitelist=False)


class SensitivityAnalyzer:
    """Read-only local slope and correlation analysis for sandbox results."""
    def analyze(self, results: list[ProcessCase], *, metric: str, parameters: list[str]) -> dict[str, Any]:
        def out(case: ProcessCase) -> Any:
            payload = getattr(case, "impact_outputs", {}) or {}
            value: Any = payload.get(metric)
            if value is None and "." in metric:
                cur: Any = payload
                for part in metric.split("."):
                    if isinstance(cur, Mapping): cur = cur.get(part)
                    else: cur = None; break
                value = cur
            if value is None and payload.get("metric") == metric: value = payload.get("value")
            return value
        valid = [(c, out(c)) for c in results if getattr(c, "converged", c.status in (CaseStatus.SUCCESS, CaseStatus.WARNINGS)) and out(c) is not None]
        baseline_case = next((c for c in results if getattr(c, "impact_inputs", {}).get("_baseline") or c.case_id == "baseline"), results[0] if results else None)
        baseline = out(baseline_case) if baseline_case else None
        report: dict[str, Any] = {"metric": metric, "baseline": baseline, "parameter_rank": [], "limitations": [],
                                  "sample_count": len(valid), "effective_case_count": len(valid),
                                  "missing_output_count": len(results) - len(valid),
                                  "convergence_rate": len([c for c in results if getattr(c, "converged", False)]) / max(1, len(results)),
                                  "conclusion_type": "insufficient_data"}
        if len(valid) <= 1:
            report["limitations"].append("无法根据当前单点结果判断参数显著性。")
        elif len(valid) == 2:
            report["conclusion_type"] = "directional_only"
        elif len(valid) <= 4:
            report["conclusion_type"] = "preliminary_local_association"
        else:
            report["conclusion_type"] = "local_sensitivity_ranking"
        for parameter in parameters:
            pairs = []
            for c, y in valid:
                x = (getattr(c, "impact_inputs", {}) or c.design_vars).get(parameter)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(float(x)) and math.isfinite(float(y)):
                    pairs.append((float(x), float(y)))
            if not pairs:
                continue
            xs, ys = zip(*pairs)
            slope = _slope(xs, ys)
            corr = _pearson(xs, ys)
            rank_corr = _spearman(xs, ys)
            x_range = max(xs) - min(xs) if len(xs) > 1 else 0.0
            y_range = max(ys) - min(ys) if len(ys) > 1 else 0.0
            sensitivity = abs(slope) * x_range / max(abs(float(baseline or statistics.mean(ys))), 1e-12)
            report["parameter_rank"].append({"parameter": parameter, "sensitivity": sensitivity,
                "direction": "positive" if slope > 0 else "negative" if slope < 0 else "flat",
                "perturbation_range": [min(xs), max(xs)], "metric_change": y_range,
                "local_slope": slope, "pearson": corr, "spearman": rank_corr,
                "sample_count": len(pairs), "convergence_rate": len(pairs) / max(1, len(results)),
                "confidence": "high" if len(pairs) >= 5 else "medium" if len(pairs) >= 3 else "low"})
            if len(pairs) < 2:
                report["limitations"].append(f"{parameter} 有效样本不足")
        report["parameter_rank"].sort(key=lambda x: x["sensitivity"], reverse=True)
        if not any((getattr(c, "impact_outputs", {}) or {}).get("flow") is not None for c in results):
            report["limitations"].append("未计算回收率，因为缺少流量数据。")
        report["uncertainty"] = report["limitations"][:]
        report["limitations"].append("局部回测不能证明全局因果关系；范围不足时不能声称显著影响。")
        return report


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
