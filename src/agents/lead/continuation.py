"""Evidence-checked task continuation, independent of intent and field names.

The LLM interprets turns and declares task fields. It cannot certify its own
candidates: candidates are resolved from JSON pointers into the current case.
Domain adapters may supply derived candidates with explicit evidence.
"""
from __future__ import annotations

import json
import math
import hashlib
from copy import deepcopy

from src.agents.errors import failure_from, internal_error
from src.agents.orchestration.models import Answer, PendingTask
from src.agents.reasoning import configured, redact, snapshot


def provider_json(lead, operation, payload, contract):
    from .llm_client import chat_json
    cfg = configured(lead.llm_config or getattr(lead.intent_classifier, "llm_config", None))
    try:
        raw = (lead.llm_call or chat_json)(cfg,
            system=f"PAO task continuation: {operation}. {contract} Return strict JSON only. Never execute actions.",
            user=json.dumps(redact(payload, cfg), ensure_ascii=False, default=str))
    except Exception as exc:
        raise failure_from(exc, operation) from None
    try:
        data = json.loads(redact(raw, cfg))
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (ValueError, TypeError):
        raise internal_error("continuation_protocol_invalid", stage=operation) from None


def interpret_turn(lead, task, text):
    data = provider_json(lead, "interpret_turn", {"task": task.to_dict(), "message": text},
        'Interpret the message relative to the pending original goal and waiting_fields. '
        'Return exactly {"kind":"answer|correction|new_task|cancel|unclear", "updates":{field:value}}. '
        'Short replies usually answer waiting_fields, not a new task. Explicit changes to known values '
        'are corrections. New unrelated requests are new_task; do not merge them. Ambiguous replies '
        'are unclear. Updates must reflect only what this user message says; never infer missing values '
        'from focus or fill all candidates. A correction may clear a value with null.')
    if (set(data) != {"kind", "updates"} or data["kind"] not in
            {"answer", "correction", "new_task", "cancel", "unclear"} or not isinstance(data["updates"], dict)):
        raise internal_error("continuation_protocol_invalid", stage="interpret_turn")
    if set(data["updates"]) - set(task.field_specs):
        raise internal_error("unknown_task_field", stage="interpret_turn")
    if data["kind"] in {"new_task", "cancel", "unclear"} and data["updates"]:
        raise internal_error("conflicting_turn_decision", stage="interpret_turn")
    return data


def pointer(document, path):
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("evidence must use a JSON pointer")
    value = document
    for part in path[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def new_task(lead, goal, intent, context):
    task = PendingTask(original_goal=goal, verified_intent=deepcopy(intent))
    if intent.intent == "parameter_impact":
        task.required_fields = ["equipment", "target_stream", "target_component", "metric"]
        task.field_specs = {name: {"type": "choice", "multiple": False} for name in task.required_fields}
        task.field_specs["candidate_parameters"] = {"type": "choice", "multiple": True}
        task.parameters = deepcopy(intent.analysis)
    elif intent.missing_information:
        data = provider_json(lead, "task_fields", {"goal": goal, "intent": intent.to_dict(), "case": snapshot(context)},
            'Describe fields needed to complete this task, including already known parameters. Return exactly '
            '{"fields":{field:{"type":"choice|text|number", "multiple":boolean, "required":boolean, '
            '"candidate_refs":["/JSON/pointer/to/scalar/in/case"]}}, "parameters":{field:value}}. '
            'Use stable field names from missing_information. All case entities and constrained values must '
            'use choice with candidate_refs to actual scalar evidence in the supplied case. '
            'text/number are only unconstrained user preferences, never case identifiers. '
            'Extract known parameters only from the user goal; unresolved fields are null. Do not invent evidence.')
        if set(data) != {"fields", "parameters"} or not isinstance(data["fields"], dict) or not isinstance(data["parameters"], dict):
            raise internal_error("task_fields_invalid", stage="task_fields")
        task.field_specs, task.parameters = data["fields"], data["parameters"]
        if not set(intent.missing_information) <= set(task.field_specs) or set(task.parameters) - set(task.field_specs):
            raise internal_error("task_fields_incomplete", stage="task_fields")
        for name, spec in task.field_specs.items():
            if (not isinstance(spec, dict) or set(spec) != {"type", "multiple", "required", "candidate_refs"}
                    or spec["type"] not in {"choice", "text", "number"}
                    or type(spec["multiple"]) is not bool or type(spec["required"]) is not bool
                    or not isinstance(spec["candidate_refs"], list)
                    or not all(isinstance(p, str) for p in spec["candidate_refs"])):
                raise internal_error("task_fields_invalid", stage="task_fields")
            if spec["required"] or name in intent.missing_information:
                task.required_fields.append(name)
    for name, value in task.parameters.items():
        if value is not None and value != []:
            task.parameter_sources[name] = {"source": "user_goal", "text": goal, "validated_by": "intent_classifier"}
    task.audit.append({"event": "task_started", "goal": goal})
    return task


def _candidate(value, path, evidence):
    return {"value": value, "evidence": [{"path": path, "value": deepcopy(evidence)}]}


def impact_candidates(task, document):
    """Parameter-impact field adapter; every identifier comes from this case."""
    result = {name: [] for name in task.field_specs}
    equipment = task.parameters.get("equipment")
    stream = task.parameters.get("target_stream")
    nodes_key = "equipment_nodes" if document.get("equipment_nodes") else "equipment"
    streams_key = "streams" if document.get("streams") else "edges"
    for index, node in enumerate(document.get(nodes_key, [])):
        value = node.get("id") or node.get("name")
        if value:
            result["equipment"].append(_candidate(value, f"/{nodes_key}/{index}", node))
    for index, row in enumerate(document.get(streams_key, [])):
        value = row.get("id") or row.get("name") or row.get("stream_id")
        source = row.get("source") or row.get("source_node") or row.get("upstream")
        path = f"/{streams_key}/{index}"
        if value and (not equipment or source == equipment):
            result["target_stream"].append(_candidate(value, path, row))
        if stream and value != stream:
            continue
        components = row.get("components") or row.get("composition") or row.get("mole_fractions") or {}
        if isinstance(components, dict):
            for name in components:
                result["target_component"].append(_candidate(name, path, row))
        elif isinstance(components, list):
            for comp in components:
                name = comp.get("component") or comp.get("name") or comp.get("id") if isinstance(comp, dict) else comp
                if name:
                    result["target_component"].append(_candidate(name, path, row))
        # These are extractor capabilities backed by an actual stream record,
        # not fabricated measurements; planning validates output prerequisites.
        composition_evidence = components or document.get("compositions", {}).get(value)
        if value and composition_evidence:
            for metric in ("mole_fraction", "purity"):
                result["metric"].append(_candidate(metric, path, row))
            if row.get("feed_component_flow") or row.get("input_component_flow"):
                result["metric"].append(_candidate("recovery", path, row))
    for stream_id, components in document.get("compositions", {}).items():
        if stream and stream_id != stream:
            continue
        for comp in components:
            name = comp.get("component") or comp.get("name") or comp.get("component_id")
            if name:
                result["target_component"].append(_candidate(name,
                    "/compositions/" + stream_id.replace("~", "~0").replace("/", "~1"), components))
    from types import SimpleNamespace
    from src.agents.orchestration.parameter_impact import parameter_catalog
    for parameter in parameter_catalog(SimpleNamespace(**document), equipment):
        candidate = _candidate(parameter["aspen_path"], "case_parameter_catalog", parameter["evidence"])
        candidate["aliases"] = parameter["aliases"]
        result["candidate_parameters"].append(candidate)
    return result


def validate_task(task, context):
    document = snapshot(context)
    stable = {k: v for k, v in document.items() if k not in {"generated_at", "session_id"}}
    task.evidence_digest = hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    if task.verified_intent.intent == "parameter_impact":
        # Also migrate persisted tasks created by the old required-slot policy.
        task.required_fields = [f for f in task.required_fields if f != "candidate_parameters"]
        task.parameters.setdefault("candidate_parameters", [])
        task.candidates = impact_candidates(task, document)
    else:
        task.candidates = {}
        for name, spec in task.field_specs.items():
            candidates = []
            for path in spec.get("candidate_refs", []):
                try:
                    value = pointer(document, path)
                except (KeyError, IndexError, TypeError, ValueError):
                    continue  # missing/stale evidence is a normal clarification
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    candidates.append(_candidate(value, path, value))
            task.candidates[name] = candidates
    missing = []
    for name, spec in task.field_specs.items():
        value = task.parameters.get(name)
        empty = value is None or value == "" or value == []
        valid = name not in task.required_fields if empty else True
        evidence = []
        if not empty:
            values = value if spec["multiple"] and isinstance(value, list) else [value]
            valid = isinstance(value, list) if spec["multiple"] else not isinstance(value, (list, dict))
            for item in values:
                if spec["type"] == "choice":
                    matches = [c for c in task.candidates.get(name, []) if type(c["value"]) is type(item) and (c["value"] == item or item in c.get("aliases", []))]
                    valid = valid and bool(matches)
                    evidence.extend(e for c in matches for e in c["evidence"])
                elif spec["type"] == "number":
                    valid = valid and type(item) in (int, float) and math.isfinite(item)
                elif spec["type"] == "resource_path":
                    from src.agents.context.resources import resolve_model_file
                    resource = resolve_model_file(item)
                    valid = valid and isinstance(item, str) and resource.path is not None
                    evidence.append(resource.to_dict())
                else:
                    valid = valid and isinstance(item, str) and bool(item.strip())
        if not valid:
            missing.append(name)
        if name in task.parameter_sources:
            task.parameter_sources[name].update(valid=valid, evidence=evidence)
    task.missing_fields = missing
    task.waiting_fields = missing[:1]
    task.phase = "clarifying" if missing else "ready"
    task.audit.append({"event": "task_validated", "missing_fields": list(missing),
                       "model_id": document.get("model_id"), "model_version": document.get("model_version")})
    return not missing


def clarification(task, *, prefix=""):
    fields = task.waiting_fields or task.missing_fields
    lines = []
    for name in fields:
        values = list(dict.fromkeys(str(c["value"]) for c in task.candidates.get(name, [])))
        lines.append(f"请明确 {name}" + (f"（候选：{', '.join(values)}）" if values else "（当前证据不足，请补充信息或更新 case 证据）") + "。")
    return Answer(text=prefix + "\n".join(lines),
                  facts=[{"kind": "task_clarification", "task_id": task.task_id, "waiting_fields": fields,
                          "candidates": {f: task.candidates.get(f, []) for f in fields}}])


def merge_turn(task, decision, text):
    task.audit.append({"event": "turn_interpreted", "kind": decision["kind"], "message": text})
    updates = decision["updates"]
    # An answer cannot silently overwrite an already established parameter.
    if decision["kind"] == "answer" and any(
        (name not in task.missing_fields or task.parameter_sources.get(name, {}).get("source") == "user_goal")
        and task.parameters.get(name) not in (None, [], value)
        for name, value in updates.items()
    ):
        task.audit.append({"event": "conflicting_answer"})
        return False
    if not updates:
        return False
    for name, value in updates.items():
        task.parameters[name] = value
        task.parameter_sources[name] = {"source": "user_reply", "text": text, "kind": decision["kind"]}
    return True


def task_prompt(task):
    return task.original_goal + "\nVerified task parameters: " + json.dumps(task.parameters, ensure_ascii=False)
