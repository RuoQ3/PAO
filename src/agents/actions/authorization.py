"""Execution scopes issued by ActionGateway, shared by every Aspen workflow.

This is an application boundary, not a Python code sandbox. Nothing serialized
in a plan (including ``approved``) is a credential.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
import hashlib
import json
import math
import re
from pathlib import Path
import threading
import time


class ExecutionBoundaryError(PermissionError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, default=str).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema_digest(schema):
    raw = asdict(schema)
    raw["permissions"] = sorted(schema.permissions)
    raw["derived_variables"] = sorted(schema.derived_variables)
    return digest(raw)


def validate_inputs(inputs, schema, scope):
    if not isinstance(inputs, dict):
        raise ExecutionBoundaryError("inputs must be a mapping")
    if not schema.writable:
        raise ExecutionBoundaryError("model is read-only")
    for path, value in inputs.items():
        if (not isinstance(path, str) or not re.fullmatch(r"\\(?:[A-Za-z0-9_ .:-]+\\)*[A-Za-z0-9_ .:-]+", path)
                or "\\input\\" not in path.casefold()):
            raise ExecutionBoundaryError(f"execution requires an explicit Aspen Input path: {path}")
        spec = schema.parameters.get(path)
        if spec is None or not spec.writable or spec.derived or path in schema.derived_variables:
            raise ExecutionBoundaryError(f"parameter is not writable: {path}")
        if path not in scope:
            raise ExecutionBoundaryError(f"parameter outside approved scope: {path}")
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ExecutionBoundaryError(f"parameter must be finite numeric: {path}")
        if spec.value_type not in {"int", "float"}:
            raise ExecutionBoundaryError(f"unsupported execution parameter type: {path}")
        if (spec.integer or spec.value_type == "int") and not float(value).is_integer():
            raise ExecutionBoundaryError(f"integer parameter required: {path}")
        if spec.minimum is None or spec.maximum is None:
            raise ExecutionBoundaryError(f"model hard bounds unavailable: {path}")
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (spec.minimum, spec.maximum))
                or spec.minimum > spec.maximum):
            raise ExecutionBoundaryError(f"model hard bounds invalid: {path}")
        lo, hi = scope[path]
        if not spec.minimum <= value <= spec.maximum:
            raise ExecutionBoundaryError(f"model hard bound violated: {path}")
        if not lo <= value <= hi:
            raise ExecutionBoundaryError(f"approved bound violated: {path}")
        if spec.allowed_values and value not in spec.allowed_values:
            raise ExecutionBoundaryError(f"value outside allowed set: {path}")
    prospective = {**schema.current_values, **inputs}
    for path, rules in schema.dependencies.items():
        for op, other in rules.items():
            if path not in inputs and other not in inputs:
                continue
            if path not in prospective or other not in prospective or op not in {"lt", "le"}:
                raise ExecutionBoundaryError("dependency cannot be validated")
            if not (prospective[path] < prospective[other] if op == "lt" else prospective[path] <= prospective[other]):
                raise ExecutionBoundaryError(f"dependency violated: {path} {op} {other}")


_active = ContextVar("pao_gateway_authorization", default=None)
_evaluation = ContextVar("pao_current_evaluation", default=None)


def require_authorization():
    grant = _active.get()
    if grant is None:
        raise ExecutionBoundaryError("real execution requires ActionGateway authorization; preview and approve a complete plan")
    grant.check()
    return grant


class _Grant:
    def __init__(self, *, action_id, schema, scope, max_evaluations, max_seconds,
                 max_failures, check, audit, ledger=None):
        self.action_id = action_id
        self.schema = deepcopy(schema)
        self.scope = deepcopy(scope)
        self.max_evaluations = max_evaluations
        self.max_failures = max_failures
        self.deadline = time.monotonic() + max_seconds
        self._check = check
        self.audit = audit
        self.ledger = ledger
        self.used = 0
        self.started_runs = 0
        self.failures = 0
        self.closed = False
        self.lock = threading.Lock()
        self.allowed_points = None
        self.algorithm = None
        self.expected_config = None
        self.config_object = None

    def check(self):
        if getattr(self, "execution_unknown", False):
            raise ExecutionBoundaryError("execution state unknown; dispatch stopped")
        if self.closed:
            raise ExecutionBoundaryError("authorization is closed")
        self._check()
        if self.config_object is not None and digest(asdict(self.config_object)) != self.expected_config:
            raise ExecutionBoundaryError("active algorithm/configuration changed; new approval required")
        if time.monotonic() >= self.deadline:
            raise ExecutionBoundaryError("authorized time budget exhausted")

    @contextmanager
    def evaluation(self, inputs):
        self.check()
        if _evaluation.get() is not None:
            raise ExecutionBoundaryError("nested evaluation reservations are disabled")
        validate_inputs(inputs, self.schema, self.scope)
        if self.allowed_points is not None and inputs not in self.allowed_points:
            raise ExecutionBoundaryError("candidate differs from approved simulation cases")
        with self.lock:
            if self.used >= self.max_evaluations or self.failures >= self.max_failures:
                raise ExecutionBoundaryError("authorized evaluation/failure budget exhausted")
            # Commit before any write/run. Unknown and failed attempts never refund.
            if self.ledger:
                try:
                    self.ledger.reserve_execution(self.action_id, self.max_evaluations, inputs, self.algorithm)
                except Exception as exc:
                    raise ExecutionBoundaryError(f"could not durably reserve execution budget: {exc}") from exc
            self.used += 1
            self.audit.record(self.action_id, "execution", status="evaluation_reserved",
                              payload={"evaluation": self.used, "inputs": inputs})
        frame = {"grant": self, "inputs": deepcopy(inputs), "run_started": False}
        token = _evaluation.set(frame)
        try:
            yield
        except BaseException:
            self.failures += 1
            raise
        finally:
            _evaluation.reset(token)


@contextmanager
def _activate(grant):
    token = _active.set(grant)
    try:
        yield grant
    finally:
        grant.closed = True
        _active.reset(token)


def authorized_workflow(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        grant = require_authorization()
        config = kwargs.get("config", args[1] if len(args) > 1 else None)
        if grant.expected_config is None or digest(asdict(config)) != grant.expected_config:
            raise ExecutionBoundaryError("workflow configuration differs from the approved execution object")
        return fn(*args, **kwargs)
    return wrapped


def evaluated_case(fn):
    @wraps(fn)
    def wrapped(driver, design_vars, *args, **kwargs):
        grant = require_authorization()
        if getattr(driver, "needs_recovery", False):
            raise ExecutionBoundaryError("driver state unknown; further evaluations and automatic retries stopped")
        current = _evaluation.get()
        if current is not None:
            if current["inputs"] != design_vars:
                raise ExecutionBoundaryError("nested run changed approved inputs")
            return fn(driver, design_vars, *args, **kwargs)
        with grant.evaluation(design_vars):
            case = fn(driver, design_vars, *args, **kwargs)
            if getattr(driver, "needs_recovery", False):
                grant.execution_unknown = True
                case.impact_outputs["_lifecycle"] = {"execution_state": "unknown", "cleanup_state": "not_started"}
            config = args[0] if args else kwargs.get("config")
            case.session_id = (getattr(grant, "session_id", "") or getattr(config, "session_id", "")
                               or getattr(case, "session_id", ""))
            case.impact_outputs["_source"] = {"execution_id": grant.action_id,
                "model_path": str(getattr(grant, "execution_model", "")),
                "model_hash": getattr(grant, "model_hash", None)}
            # Converged but infeasible candidates are valid optimizer feedback,
            # not failed simulations. Otherwise a strict purity target exhausts
            # max_failures before GA can use constraint residuals to improve.
            if (not case.simulation_valid or not case.objectives_available
                    or case.has_constraints and not case.constraints_available):
                grant.failures += 1
            case.execution_algorithm = deepcopy(grant.algorithm)
            if grant.ledger:
                try:
                    grant.ledger.save_case(case.to_dict())
                    grant.ledger.finish_execution_evaluation(grant.action_id, grant.used, case)
                except Exception as exc:
                    raise ExecutionBoundaryError(f"execution evidence could not be persisted; stopped: {exc}") from exc
            return case
    return wrapped


def check_write(path, value):
    grant = require_authorization()
    frame = _evaluation.get()
    if frame is None or frame["inputs"].get(path) != value or path not in frame["inputs"]:
        raise ExecutionBoundaryError("write requires an approved, reserved evaluation")
    validate_inputs(frame["inputs"], grant.schema, grant.scope)


def check_run(*, started=True):
    grant = require_authorization()
    frame = _evaluation.get()
    if frame is None or frame["run_started"]:
        raise ExecutionBoundaryError("each model run requires a new evaluation reservation")
    frame["run_started"] = True
    if started:
        grant.started_runs += 1
    return grant


def check_model_open(path):
    """An active execution may only open its registered disposable model."""
    grant = _active.get()
    if grant is not None:
        grant.check()
        if Path(path).resolve() != getattr(grant, "execution_model", None):
            raise ExecutionBoundaryError("driver model differs from approved disposable copy")
