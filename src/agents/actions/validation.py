from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from .models import ActionRequest, ActionType, ModelSchema, ActionPreview

class ActionValidationError(ValueError): pass

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preview: ActionPreview | None = None

_SAFE_PATH = __import__('re').compile(r"^\\(?:[A-Za-z0-9_ .-]+\\)*[A-Za-z0-9_ .-]+$")

def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))

def validate_action(action: ActionRequest, schema: ModelSchema, *, actor: str = "agent") -> ValidationResult:
    errors: list[str] = []; warnings: list[str] = []
    try: kind = ActionType(action.action_type)
    except Exception: errors.append("unsupported action type"); kind = None
    if kind == ActionType.MODIFY_STRUCTURE:
        errors.append("structural modification actions are disabled")
    if kind and kind.value not in schema.permissions:
        errors.append(f"permission denied for {kind.value}")
    if action.model_id and action.model_id != schema.identity.model_id:
        errors.append("model identity mismatch")
    if action.model_version and schema.identity.model_version and action.model_version != schema.identity.model_version:
        errors.append("model version mismatch")
    if not action.reason.strip(): errors.append("reason is required")
    if not action.audit_id.strip(): errors.append("audit_id is required")
    impact: tuple[str, ...] = (); risks: list[str] = []
    old = action.old_value; new = action.new_value; unit = action.unit
    will_run = kind in {ActionType.RUN_SIMULATION, ActionType.RUN_OPTIMIZATION}
    if kind == ActionType.INSPECT:
        if action.target and action.target not in schema.current_values and action.target not in schema.targets:
            errors.append("inspect target does not exist")
        will_run = False
    elif kind == ActionType.SET_PARAMETER:
        if not _SAFE_PATH.match(action.target): errors.append("invalid or unsafe parameter path")
        spec = schema.parameters.get(action.target)
        if spec is None: errors.append("parameter path does not exist")
        else:
            impact = spec.impact_targets
            will_run = spec.requires_simulation
            if spec.derived or action.target in schema.derived_variables:
                errors.append("derived variables are read-only; modify their source parameter")
            if not schema.writable or not spec.writable: errors.append("parameter is not writable")
            if action.target not in schema.current_values: errors.append("current value is unavailable")
            elif old is not None and old != schema.current_values[action.target]: errors.append("old_value does not match current model value")
            old = schema.current_values.get(action.target) if old is None else old
            if unit and spec.unit and unit != spec.unit: errors.append("unit mismatch")
            if new is None: errors.append("new_value is required")
            else:
                expected = spec.value_type
                ok = ((expected == "int" and type(new) is int) or
                      (expected == "float" and type(new) in (int, float) and not isinstance(new, bool)) or
                      (expected == "bool" and type(new) is bool) or
                      (expected == "str" and type(new) is str))
                if not ok: errors.append(f"new_value must have type {expected}")
                if spec.integer and (type(new) is not int and not (isinstance(new, float) and new.is_integer())): errors.append("integer constraint violated")
                if isinstance(new, (int, float)) and not isinstance(new, bool):
                    if not math.isfinite(float(new)): errors.append("new_value must be finite")
                    if spec.minimum is not None and new < spec.minimum: errors.append("hard lower bound violated")
                    if spec.maximum is not None and new > spec.maximum: errors.append("hard upper bound violated")
                if spec.allowed_values and new not in spec.allowed_values: errors.append("value is not in allowed set")
                # Dependencies may be paths or simple comparison rules.  They
                # are checked against the prospective model state so a change
                # cannot silently violate a configured relationship.
                prospective = dict(schema.current_values)
                prospective[action.target] = new
                for dep in spec.dependencies:
                    if dep not in prospective: risks.append(f"dependency unavailable: {dep}")
                for dep, rule in getattr(schema, "dependencies", {}).get(action.target, {}).items() if hasattr(schema, "dependencies") else ():
                    if dep not in prospective:
                        errors.append(f"dependency unavailable: {dep}")
                    elif rule == "lt" and not prospective[action.target] < prospective[dep]:
                        errors.append(f"dependency violated: {action.target} < {dep}")
                    elif rule == "le" and not prospective[action.target] <= prospective[dep]:
                        errors.append(f"dependency violated: {action.target} <= {dep}")
    elif kind == ActionType.SET_SEARCH_REGION:
        if not isinstance(new, dict) or not new: errors.append("search region must be a non-empty mapping")
        else:
            for path, interval in new.items():
                spec = schema.parameters.get(path)
                if spec is None: errors.append(f"region parameter does not exist: {path}"); continue
                if not isinstance(interval, (list, tuple)) or len(interval) != 2 or not all(_finite(x) for x in interval): errors.append(f"invalid interval: {path}"); continue
                lo, hi = interval
                if lo > hi: errors.append(f"reversed interval: {path}")
                if spec.minimum is not None and lo < spec.minimum: errors.append(f"lower bound violated: {path}")
                if spec.maximum is not None and hi > spec.maximum: errors.append(f"upper bound violated: {path}")
                if spec.integer and math.ceil(lo) > math.floor(hi): errors.append(f"no integer in interval: {path}")
            will_run = False
    elif kind in {ActionType.RUN_SIMULATION, ActionType.RUN_OPTIMIZATION}:
        if kind == ActionType.RUN_SIMULATION and isinstance(action.payload.get("simulation_plan"), dict):
            try:
                from src.agents.orchestration.parameter_impact import SimulationPlan
                plan = SimulationPlan.from_dict(action.payload["simulation_plan"])
                plan.validate_execution()
                from .plans import simulation_scope
                simulation_scope(plan, schema)
            except (ValueError, TypeError, KeyError, OSError, PermissionError) as exc:
                errors.append(f"invalid simulation plan: {exc}")
        elif kind == ActionType.RUN_OPTIMIZATION and isinstance(action.payload.get("optimization_plan"), dict):
            try:
                from .plans import validate_optimization_plan
                validate_optimization_plan(action.payload["optimization_plan"], schema)
            except (ValueError, TypeError, KeyError, OSError, PermissionError) as exc:
                errors.append(f"invalid optimization plan: {exc}")
        else:
            errors.append("complete execution plan required: base model, explicit inputs/scope, outputs, total budget and limits")
        if action.target not in {schema.identity.model_id, "model", ""}: errors.append("run target does not match model")
        will_run = True
    if action.execution_mode.value == "automatic" and action.requires_confirmation: warnings.append("automatic mode still requires explicit confirmation")
    if action.run_simulation is not None and kind == ActionType.SET_PARAMETER:
        will_run = bool(action.run_simulation)
    spec = schema.parameters.get(action.target) if kind == ActionType.SET_PARAMETER else None
    preview = ActionPreview(
        action.audit_id, kind.value if kind else str(action.action_type), action.target,
        old, new, unit, impact, tuple(risks), will_run, action.requires_confirmation,
        confirmation_prompt=(f"确认将 {action.target} 从 {old!r} 修改为 {new!r}"
                             + (f" {unit}" if unit else "")
                             + ("，并运行一次仿真？" if will_run else "？")),
        value_type=spec.value_type if spec else None,
        writable=bool(spec.writable) if spec else True,
        backup_required=kind == ActionType.SET_PARAMETER,
        execution_plan=__import__('copy').deepcopy(action.payload),
    ) if not errors else None
    return ValidationResult(not errors, errors, warnings, preview)
