"""Host-owned tool catalogue and bounded dispatch. No COM, shell or code tools."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
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
            ready = d.handler is not None and bool(d.available(context))
            rows.append(dict(id=d.name, description=d.description,
                handler=f"{getattr(d.handler, '__module__', '')}.{getattr(d.handler, '__qualname__', type(d.handler).__name__)}" if d.handler else None,
                input={k: v.__name__ for k, v in d.parameters.items()}, required=list(d.required),
                output=d.output_protocol, dependencies=list(d.dependencies), available=ready,
                status=d.mode if ready else "unavailable", evidence_sources=list(d.evidence_sources),
                side_effect=d.side_effect, requires_approval=d.side_effect,
                roles=list(d.roles), max_calls=d.max_calls, timeout_seconds=d.timeout_seconds))
        return rows


class ToolSession:
    def __init__(self, registry, task, context, *, max_tool_calls=32, audit=None, cfg=None, stopped=lambda: False):
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
            if d.handler is None or not d.available(self.context):
                raise LookupError("tool dependency/source unavailable")
            # The lower boundary rechecks role, source freshness and scope too.
            value = bounded_call(lambda: self._dispatch(d, arguments),
                                 min(d.timeout_seconds, self.deadline - time.monotonic()))
            self.check()
            from src.agents.orchestration.models import ActionProposal
            if d.side_effect:
                if not isinstance(value, ActionProposal) or not value.requires_confirmation:
                    raise PermissionError("effect tool must return a concrete approval proposal")
                self.proposals.append(value)
                value = {"status": "plan_only", "proposal": value.to_dict(), "executed": False}
            version = hashlib.sha256(json.dumps(self.context, sort_keys=True, default=str).encode()).hexdigest()
            packet = dict(tool=name, status="available", value=redact(value, self.cfg),
                evidence_id=record["call_id"], source_versions=self.context.get("dependency_versions", {}),
                snapshot_digest=version, sources=deepcopy(self.context.get("sources", [])))
            self.results.append(packet)
            record.update(status="completed", evidence_refs=[packet["evidence_id"]])
            return packet
        except Exception as exc:
            record.update(status="timeout" if isinstance(exc, TimeoutError) else "rejected" if isinstance(exc, (PermissionError, ValueError)) else "unavailable",
                          error_class=type(exc).__name__)
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.audit.append(record)

    def _dispatch(self, definition, arguments):
        self.check()
        if self.task.agent not in definition.roles or definition.name not in self.task.tool_allowlist:
            raise PermissionError("tool authorization missing")
        from src.agents.context.versions import file_version
        for version in self.context.get("dependency_versions", {}).values():
            if isinstance(version, dict) and "path" in version and file_version(version["path"]) != version:
                raise LookupError("source version changed; reload context")
        return definition.handler(deepcopy(self.context), **arguments)


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
    if any(x.get("type") == "custom_module" for x in context.get("configuration", {}).get("objectives", [])):
        raise PermissionError("custom module planning is not exposed to production agents")
    plan, _ = build_optimization_plan(version["path"])
    return ActionProposal("run_optimization", "model", "Review selected project optimization plan",
        payload={"optimization_plan": plan, "session_id": context.get("session_id")},
        risks=["All Aspen evaluations consume the shared total budget"], requires_confirmation=True)


def default_tools(definitions):
    """Role declarations are the single authority for role/tool associations."""
    from functools import partial
    registry = ToolRegistry()
    fields = {
        "read_process_context": ("equipment", "streams", "topology", "compositions", "variables", "conflicts", "diagnostics"),
        "read_yaml": ("configuration", "variables", "objectives", "constraints", "conflicts"),
        "read_node_db": ("equipment_parameters", "current_values"),
        "read_simulation_db": ("history_evidence",),
        "read_checkpoint": ("history_evidence",),
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
        available=lambda c: bool(c.get("attachments", c.get("files"))), evidence_sources=("attachments",)))
    registry.register(ToolDefinition("plan_optimization", "Build a concrete optimization proposal; never execute", _plan_optimization,
        tuple(d.name for d in definitions.values() if "plan_optimization" in d.tool_allowlist),
        dependencies=("selected YAML", "model", "execution limits"), evidence_sources=("yaml", "model"),
        available=lambda c: bool(c.get("configuration") and c.get("dependency_versions", {}).get("yaml")),
        side_effect=True, mode="plan"))
    return registry
