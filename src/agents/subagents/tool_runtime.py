"""Host-owned tool catalogue and bounded dispatch. No COM, shell or code tools."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import queue
import threading
import time
import uuid
from typing import Any, Callable

from src.agents.errors import unavailable
from src.agents.reasoning import redact, snapshot


def bounded_call(call, seconds):
    """Bound waiting, not thread termination. Timed out work is never reused."""
    if seconds <= 0:
        raise TimeoutError("task deadline exhausted")
    result = queue.Queue(maxsize=1)
    def run():
        try:
            result.put((True, call()))
        except Exception as exc:
            result.put((False, exc))
    threading.Thread(target=run, daemon=True, name="pao-readonly-tool").start()
    try:
        ok, value = result.get(timeout=seconds)
    except queue.Empty:
        raise TimeoutError("waiting deadline exceeded; worker termination is not guaranteed") from None
    if not ok:
        raise value
    return value


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    handler: Callable | None
    roles: tuple[str, ...]
    parameters: dict = field(default_factory=dict)  # exact argument name -> Python type
    required: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    available: Callable = lambda context: True
    evidence_sources: tuple[str, ...] = ()
    side_effect: bool = False
    mode: str = "read"  # effectful capabilities expose ONLY plan builders
    max_calls: int = 4
    timeout_seconds: float = 5
    output_protocol: str = "versioned_evidence_v1"

    def __post_init__(self):
        if self.mode not in {"read", "plan", "unavailable"} or (self.side_effect and self.mode != "plan"):
            raise ValueError("effectful tools may only expose plan builders")
        if self.max_calls < 1 or self.timeout_seconds <= 0:
            raise ValueError("invalid tool limits")


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, definition):
        if definition.name in self._tools:
            raise ValueError("duplicate tool identifier")
        self._tools[definition.name] = definition

    def catalogue(self, context, *, role=None, allowlist=None):
        rows = []
        for d in self._tools.values():
            if role and (role not in d.roles or allowlist is not None and d.name not in allowlist):
                continue
            try:
                ready = d.mode != "unavailable" and d.handler is not None and bool(d.available(context))
                reason = None if ready else "handler not connected or selected dependency/source unavailable"
            except Exception as exc:
                ready, reason = False, type(exc).__name__
            rows.append(dict(id=d.name, description=d.description,
                handler=f"{getattr(d.handler, '__module__', '')}.{getattr(d.handler, '__qualname__', type(d.handler).__name__)}" if d.handler else None,
                input={k: v.__name__ for k, v in d.parameters.items()}, required=list(d.required),
                output=d.output_protocol, dependencies=list(d.dependencies), available=ready,
                status=d.mode if ready else "unavailable", evidence_sources=list(d.evidence_sources),
                unavailable_reason=reason,
                side_effect=d.side_effect, requires_approval=d.side_effect,
                roles=list(d.roles), max_calls=d.max_calls, timeout_seconds=d.timeout_seconds))
        return rows


class ToolSession:
    def __init__(self, registry, task, context, *, max_tool_calls=32, audit=None, cfg=None, stopped=lambda: False):
        if (not math.isfinite(task.timeout_seconds) or task.timeout_seconds <= 0
                or task.budget < 1 or max_tool_calls < 1):
            raise ValueError("invalid tool task budget")
        self.registry, self.task, self.context = registry, task, snapshot(context)
        self.audit = audit if audit is not None else []
        self.cfg, self.stopped = cfg, stopped
        self.deadline = time.monotonic() + task.timeout_seconds
        self.limit = min(task.budget, max_tool_calls)
        self.calls = 0
        self.counts = {}
        self.results = []
        self.proposals = []

    def check(self):
        if self.stopped() or time.monotonic() >= self.deadline:
            raise TimeoutError("tool task stopped or timed out")

    def invoke(self, name, arguments):
        started = time.monotonic()
        record = dict(event="tool_call", call_id=uuid.uuid4().hex, task_id=self.task.task_id,
            session_id=self.task.metadata.get("session_id"), role=self.task.agent, tool=name,
            arguments={}, status="rejected", evidence_refs=[], error_class=None)
        try:
            self.check()
            if name not in self.registry._tools:
                raise PermissionError("unregistered tool")
            d = self.registry._tools[name]
            if name not in self.task.tool_allowlist or self.task.agent not in d.roles:
                raise PermissionError("role/task tool permission denied")
            if not isinstance(arguments, dict) or set(arguments) - set(d.parameters) or set(d.required) - set(arguments):
                raise ValueError("invalid tool arguments")
            if any(type(v) is not d.parameters[k] for k, v in arguments.items()):
                raise ValueError("invalid tool argument type")
            # Unknown/invalid arguments never enter an audit, even if they contain credentials.
            record["arguments"] = redact(arguments, self.cfg)
            if self.calls >= self.limit or self.counts.get(name, 0) >= d.max_calls:
                raise PermissionError("tool call budget exhausted")
            self.calls += 1
            self.counts[name] = self.counts.get(name, 0) + 1
            if d.mode == "unavailable" or d.handler is None:
                raise LookupError("tool dependency/source unavailable")
            # The lower boundary rechecks role, source freshness and scope too.
            value = bounded_call(lambda: self._dispatch(d, arguments),
                                 min(d.timeout_seconds, self.deadline - time.monotonic()))
            self.check()
            self._check_versions()
            from src.agents.orchestration.models import ActionProposal
            if d.side_effect:
                if not isinstance(value, ActionProposal) or not value.requires_confirmation:
                    raise PermissionError("effect tool must return a concrete approval proposal")
                self.proposals.append(value)
                value = {"status": "plan_only", "proposal": value.to_dict(), "executed": False}
            version = hashlib.sha256(json.dumps(self.context, sort_keys=True, default=str).encode()).hexdigest()
            from src.agents.context.versions import file_version
            import inspect
            handler = getattr(d.handler, "func", d.handler)
            packet = dict(tool=name, status="available", value=redact(value, self.cfg),
                evidence_id=record["call_id"], source_versions=self.context.get("dependency_versions", {}),
                handler_version=file_version(inspect.getsourcefile(handler)) if inspect.isfunction(handler) else None,
                snapshot_digest=version, sources=deepcopy(self.context.get("sources", [])))
            self.results.append(packet)
            record.update(status="completed", evidence_refs=[packet["evidence_id"]])
            return packet
        except Exception as exc:
            record.update(status="timeout" if isinstance(exc, TimeoutError) else "rejected" if isinstance(exc, (PermissionError, ValueError)) else "unavailable",
                          error_class=type(exc).__name__)
            if isinstance(exc, TimeoutError):
                record.update(wait_timed_out=True, worker_termination="unknown", late_results_discarded=True)
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.audit.append(record)

    def _dispatch(self, definition, arguments):
        self.check()
        if self.task.agent not in definition.roles or definition.name not in self.task.tool_allowlist:
            raise PermissionError("tool authorization missing")
        if not definition.available(self.context):
            raise LookupError("tool dependency/source unavailable")
        self._check_versions()
        return definition.handler(deepcopy(self.context), **arguments)

    def _check_versions(self):
        from src.agents.context.versions import file_version
        def walk(value):
            if isinstance(value, dict):
                if "path" in value and "status" in value and ("sha256" in value or value["status"] == "unavailable"):
                    if file_version(value["path"]) != value:
                        raise LookupError("source version changed; reload context")
                else:
                    for child in value.values():
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(self.context.get("dependency_versions", {}))
        for item in self.context.get("attachments", self.context.get("files", [])):
            walk(item.get("version", {}))


def _select(context, fields):
    return {key: deepcopy(context[key]) for key in fields if key in context}


def _attachment(context, index):
    items = context.get("attachments", context.get("files", []))
    if index < 0 or index >= len(items):
        raise ValueError("attachment outside user-selected data scope")
    item = items[index]
    if item.get("status") != "available":
        raise LookupError("attachment unavailable or truncated")
    return item


def _plan_optimization(context):
    from src.agents.actions.plans import build_optimization_plan
    from src.agents.orchestration.models import ActionProposal
    version = context.get("dependency_versions", {}).get("yaml")
    if not version or version.get("status") != "available":
        raise LookupError("selected optimization YAML unavailable")
    # No arbitrary Python/custom-module import from source content.
    _safe_configuration(context)
    request = context.get("optimization_request") or {}
    plan, _ = build_optimization_plan(version["path"], **(
        {"overrides": request["overrides"]} if request.get("overrides") else {}))
    if request and plan["algorithm"]["algorithm"] != request["algorithm"]:
        raise ValueError("planned algorithm differs from the conversation selection")
    return ActionProposal("run_optimization", "model", "Review selected project optimization plan",
        payload={"optimization_plan": plan, "session_id": context.get("session_id"),
                 "conversation_selection": request},
        risks=["All Aspen evaluations consume the shared total budget"], requires_confirmation=True)


def _safe_configuration(context):
    """Re-read selected bytes, never load custom Python from an agent request."""
    import yaml
    from pathlib import Path
    from src.agents.context.versions import file_version
    version = context.get("dependency_versions", {}).get("yaml")
    if not version or file_version(version["path"]) != version:
        raise LookupError("selected configuration unavailable or changed")
    raw = yaml.safe_load(Path(version["path"]).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(x.get("type") == "custom_module" for x in raw.get("objectives", [])):
        raise PermissionError("custom module loading is not exposed to production agents")
    return version["path"]


def _validate_configuration(context):
    from src.agents.actions.plans import build_optimization_plan, preflight_optimizer
    from src.utils.file_io import load_optimize_config
    path = _safe_configuration(context)
    request = context.get("optimization_request") or {}
    options = {"overrides": request["overrides"]} if request.get("overrides") else {}
    config, _, _ = load_optimize_config(path, **options)
    try:
        plan, _ = build_optimization_plan(path, **options)
    except ValueError as exc:
        # A readable legacy configuration may lack an execution envelope.
        # Report that diagnostic without turning a read-only check into a
        # protocol failure or inventing permission to run.
        return {"valid": False, "execution_ready": False,
                "diagnostics": [str(exc)], "aspen_opened": False}
    preflight_optimizer(config)
    return {"valid": True, "algorithm": plan["algorithm"], "limits": plan["limits"],
            "scope": plan["scope"], "configuration": plan["configuration"], "aspen_opened": False}


def _list_optimizers(context):
    from src.optimization.registry import algorithm_catalogue
    return {"algorithms": algorithm_catalogue(), "selection_required": True, "execution_performed": False}


def _configuration_ready(context):
    try:
        path = _safe_configuration(context)
        request = context.get("optimization_request") or {}
        options = {"overrides": request["overrides"]} if request.get("overrides") else {}
        from src.utils.file_io import load_optimize_config
        from src.agents.actions.plans import build_optimization_plan, preflight_optimizer
        config, _, _ = load_optimize_config(path, **options)
        build_optimization_plan(path, **options)
        preflight_optimizer(config)
        return True
    except Exception:
        return False


def _fit_linear(context, index, rate_unit):
    """Explicit linearized least squares only; reuse the existing numerical kernel."""
    import csv
    import io
    from src.agents.subagents.kinetics_advisor.fitting import _linear_regression, _R
    item = _attachment(context, index)
    if not rate_unit.strip() or len(rate_unit) > 80:
        raise ValueError("rate constant unit must be explicit")
    rows = list(csv.DictReader(io.StringIO(item["content"])))
    if not 3 <= len(rows) <= 10000 or set(rows[0]) != {"temperature_K", "rate_constant"}:
        raise ValueError("need 3..10000 rows with temperature_K,rate_constant")
    temps, rates = [], []
    for row in rows:
        t, k = float(row["temperature_K"]), float(row["rate_constant"])
        if not math.isfinite(t) or not math.isfinite(k) or t <= 0 or k <= 0:
            raise ValueError("temperature and rate constants must be positive finite values")
        temps.append(t)
        rates.append(k)
    a, b, r2 = _linear_regression([1 / t for t in temps], [math.log(k) for k in rates])
    result = {"method": "linearized_least_squares", "Ea": -b * _R, "Ea_unit": "J/mol",
              "k0": math.exp(a), "k0_unit": rate_unit, "r_squared": r2, "n_points": len(rows),
              "temperature_unit": "K", "temperature_range": [min(temps), max(temps)],
              "source": item.get("path"), "source_version": item.get("version"),
              "limitation": "Numerical fit only; physical validity requires Agent review. No nonlinear refinement."}
    if any(not math.isfinite(result[k]) for k in ("Ea", "k0", "r_squared")):
        raise ValueError("non-finite fit")
    return result


def default_tools(definitions):
    """Role declarations are the single authority for role/tool associations."""
    from functools import partial
    registry = ToolRegistry()
    fields = {
        "read_process_context": ("equipment", "streams", "topology", "compositions", "variables", "conflicts", "diagnostics", "facts"),
        "read_yaml": ("configuration", "variables", "objectives", "constraints", "conflicts"),
        "read_node_db": ("equipment_parameters", "current_values"),
        "read_simulation_db": ("history_evidence",),
        "read_evidence": ("facts", "sources", "history_evidence"),
    }
    for name, keys in fields.items():
        roles = tuple(d.name for d in definitions.values() if name in d.tool_allowlist)
        registry.register(ToolDefinition(name, "Query authorized snapshot: " + ", ".join(keys),
            partial(_select, fields=keys), roles,
            available=lambda c, keys=keys: any(c.get(k) for k in keys), evidence_sources=keys,
            dependencies=("selected versioned ProcessContext",)))
    registry.register(ToolDefinition("read_files", "Read a user-selected UTF-8 attachment by index", _attachment,
        tuple(d.name for d in definitions.values() if "read_files" in d.tool_allowlist),
        parameters={"index": int}, required=("index",), dependencies=("complete selected attachment",),
        available=lambda c: any(x.get("status") == "available" for x in c.get("attachments", c.get("files", []))), evidence_sources=("attachments",)))
    registry.register(ToolDefinition("plan_optimization", "Build a concrete optimization proposal; never execute", _plan_optimization,
        tuple(d.name for d in definitions.values() if "plan_optimization" in d.tool_allowlist),
        dependencies=("selected YAML", "model", "execution limits"), evidence_sources=("yaml", "model"),
        available=_configuration_ready,
        side_effect=True, mode="plan"))
    registry.register(ToolDefinition("validate_config", "Validate selected configuration with actual parser, bounds and backend preflight; no Aspen",
        _validate_configuration, tuple(d.name for d in definitions.values() if "validate_config" in d.tool_allowlist),
        dependencies=("selected YAML", "installed optimizer backend"), evidence_sources=("yaml", "model"),
        available=lambda c: bool(c.get("configuration") and c.get("dependency_versions", {}).get("yaml"))))
    registry.register(ToolDefinition("list_optimizers", "Query actual optimizer support; suggestions are not user selections",
        _list_optimizers, tuple(d.name for d in definitions.values() if "list_optimizers" in d.tool_allowlist),
        dependencies=("installed optimizer metadata",), evidence_sources=("optimizer registry",)))
    registry.register(ToolDefinition("fit_arrhenius_linear", "Fit selected CSV using explicit linearized least squares, without nonlinear fallback",
        _fit_linear, tuple(d.name for d in definitions.values() if "fit_arrhenius_linear" in d.tool_allowlist),
        parameters={"index": int, "rate_unit": str}, required=("index", "rate_unit"),
        dependencies=("complete selected CSV: temperature_K,rate_constant",), evidence_sources=("attachments",),
        available=lambda c: any(x.get("status") == "available" and x.get("path", "").lower().endswith(".csv")
                                for x in c.get("attachments", c.get("files", [])))))
    for name, description in {
        "read_checkpoint": "Checkpoint evidence adapter is not connected to production tools",
        "query_knowledge": "External knowledge transport is not connected to the production tool registry",
        "scan_aspen_nodes": "COM discovery has no approved Gateway adapter; supply a static node catalogue",
        "execute_code": "Production code_engineer can only read selected code and propose changes",
    }.items():
        registry.register(ToolDefinition(name, description, None, (), mode="unavailable", available=lambda c: False))
    return registry
