"""Evidence-checked task continuation, independent of intent and field names.

The LLM interprets turns and declares task fields. It cannot certify its own
candidates: candidates are resolved from JSON pointers into the current case.
Domain adapters may supply derived candidates with explicit evidence.
"""
from __future__ import annotations

import json
import math
import hashlib
import logging
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
    logging.getLogger(__name__).debug("continuation response operation=%s: %s", operation,
        redact(raw, cfg), extra={"pao_file_only": True})
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
        'are corrections. Requests to explain a concept, including budget calculation, are new_task even when a budget field is waiting; do not label them unclear or cancel merely because the message says 算了 before 先解释. New unrelated requests are new_task; do not merge them. Ambiguous replies '
        'are unclear. Updates must reflect only what this user message says; never infer missing values '
        'from focus or fill all candidates. A correction may clear a value with null. '
        'For optimization, generations counts 代/generations; max_evaluations counts individual '
        'evaluations/次, never convert generations into evaluations. Preserve percent signs and '
        'A standalone explicit generation count like 200代 supplies generations even while waiting for max_evaluations; it is answer, not unclear. '
        'lower-bound qualifiers in target. A reply supplying a field reported invalid in '
        'planning_blockers repairs that field even when its old value came from the original goal. '
        'Explicit purity statements such as 纯度为99.99% update target, even after an earlier rejection.')
    if (set(data) != {"kind", "updates"} or data["kind"] not in
            {"answer", "correction", "new_task", "cancel", "unclear"} or not isinstance(data["updates"], dict)):
        raise internal_error("continuation_protocol_invalid", stage="interpret_turn")
    # Normalize an unambiguous numeric format after the real LLM interpretation.
    # Units bind the field, independent of which budget slot is currently asked.
    if task.verified_intent.intent == "optimization_request" and data["kind"] in {"answer", "unclear"}:
        import re
        import unicodedata
        match = re.fullmatch(r"\s*(\d+)\s*(代|generations?|秒|s)\s*[。.]?\s*", unicodedata.normalize("NFKC", text), re.I)
        if match:
            field = "generations" if match[2].casefold() not in {"秒", "s"} else "max_seconds"
            data = {"kind": "answer", "updates": {field: int(match[1])}}
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


def new_task(lead, goal, intent, context, *, audit=None):
    task = PendingTask(original_goal=goal, verified_intent=deepcopy(intent))
    if intent.intent == "parameter_impact":
        task.required_fields = ["equipment", "target_stream", "target_component", "metric"]
        task.field_specs = {name: {"type": "choice", "multiple": False} for name in task.required_fields}
        task.field_specs["candidate_parameters"] = {"type": "choice", "multiple": True}
        task.parameters = deepcopy(intent.analysis)
    elif intent.intent == "optimization_request":
        # Classification describes the request; YAML supplies process facts.
        # Do not turn speculative missing_information prose into required slots.
        analysis = intent.analysis
        task.field_specs = {"algorithm": {"type": "text", "multiple": False}}
        task.parameters = {"algorithm": analysis.get("algorithm") or analysis.get("method")}
        for name in ("equipment", "component", "target_component", "target_stream", "metric", "target"):
            task.field_specs[name] = {"type": "text", "multiple": False}
            value = analysis.get(name)
            task.parameters[name] = str(value) if value is not None else None
        population_options = {**(snapshot(context).get("configuration", {}).get("optimizer", {}).get("parameters") or {}),
                              **analysis.get("optimizer_parameters", {})}
        for name in ("pop_size", "generations"):
            task.field_specs[name] = {"type": "number", "multiple": False}
            task.parameters[name] = population_options.get(name)
            if task.parameters[name] is not None and name not in analysis.get("optimizer_parameters", {}):
                task.parameter_sources[name] = {"source": "case_configuration"}
        limits = {**(snapshot(context).get("configuration", {}).get("execution") or {}),
                  **(analysis.get("execution") or {})}
        for name in ("max_evaluations", "max_seconds", "max_failures"):
            task.field_specs[name] = {"type": "number", "multiple": False}
            task.required_fields.append(name)
            task.parameters[name] = limits.get(name)
            if limits.get(name) is not None and name not in (analysis.get("execution") or {}):
                task.parameter_sources[name] = {"source": "case_configuration"}
    elif intent.missing_information:
        payload = {"goal": goal, "intent": intent.to_dict(), "case": snapshot(context)}
        contract = (
            'Describe fields needed to complete this task, including already known parameters. Return exactly '
            '{"fields":{field:{"type":"choice|text|number", "multiple":boolean, "required":boolean, '
            '"candidate_refs":["/JSON/pointer/to/scalar/in/case"]}}, "parameters":{field:value}}. '
            'Use stable field names from missing_information. All case entities and constrained values must '
            'use choice with candidate_refs to actual scalar evidence in the supplied case. '
            'text/number are only unconstrained user preferences, never case identifiers. '
            'Extract known parameters only from the user goal; unresolved fields are null. Do not invent evidence. '
            'Every missing_information string must be used verbatim as a fields key, including spaces. '
            'Every parameters key must also be declared in fields. Do not rename keys.')
        for attempt in range(2):
            try:
                data = provider_json(lead, "task_fields", payload, contract)
                code, diagnostics = _field_errors(data, intent.missing_information)
            except Exception as exc:
                if getattr(exc, "code", None) != "continuation_protocol_invalid":
                    raise
                data, code, diagnostics = {}, "task_fields_invalid", {"reason": "invalid_json_object"}
            if code is None:
                break
            record = {"event": "task_fields_validation_failed", "attempt": attempt,
                      "code": code, **diagnostics}
            cfg = lead.llm_config or getattr(lead.intent_classifier, "llm_config", None)
            record = redact(record, cfg)
            logging.getLogger(__name__).warning("task fields validation: %s", record,
                extra={"pao_file_only": True})
            # Persist even when task construction ultimately fails.
            if audit is not None:
                audit.append(record)
            if attempt:
                error = internal_error(code, stage="task_fields")
                error.retry_count = attempt
                raise error
            payload = {**payload, "invalid_response": data, "validation_error": diagnostics,
                       "instruction": "Repair the complete response using the exact required keys; do not invent values."}
        task.field_specs, task.parameters = data["fields"], data["parameters"]
        for name, spec in task.field_specs.items():
            if spec["required"] or name in intent.missing_information:
                task.required_fields.append(name)
    for name, value in task.parameters.items():
        if value is not None and value != []:
            task.parameter_sources.setdefault(name, {"source": "user_goal", "text": goal, "validated_by": "intent_classifier"})
    task.audit.append({"event": "task_started", "goal": goal})
    return task


def _field_errors(data, required):
    if set(data) != {"fields", "parameters"} or not isinstance(data.get("fields"), dict) or not isinstance(data.get("parameters"), dict):
        return "task_fields_invalid", {"reason": "fields_and_parameters_objects_required"}
    fields, parameters = data["fields"], data["parameters"]
    missing, extra = sorted(set(required) - set(fields)), sorted(set(parameters) - set(fields))
    if missing or extra:
        return "task_fields_incomplete", {"missing_fields": missing, "undeclared_parameters": extra}
    for spec in fields.values():
        if (not isinstance(spec, dict) or set(spec) != {"type", "multiple", "required", "candidate_refs"}
                or spec["type"] not in {"choice", "text", "number"}
                or type(spec["multiple"]) is not bool or type(spec["required"]) is not bool
                or not isinstance(spec["candidate_refs"], list)
                or not all(isinstance(p, str) for p in spec["candidate_refs"])):
            return "task_fields_invalid", {"reason": "invalid_field_specification"}
    return None, {}


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
        # Targets can be downstream products; they need not leave the varied
        # equipment directly. Every candidate remains backed by a case stream.
        if value:
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
            for metric in ("mole_fraction", "mass_fraction", "purity"):
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
    # Recompute semantic diagnostics on every turn, including restored tasks.
    task.planning_blockers = [b for b in task.planning_blockers if b.get("source") != "task_validation"]
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
                    if task.verified_intent.intent == "optimization_request":
                        valid = valid and item > 0 and (name == "max_seconds" or type(item) is int)
                        if name == "pop_size":
                            valid = valid and item >= 2
                        if not valid:
                            task.planning_blockers.append({"field": name, "source": "task_validation",
                                "message": f"{name} 无效：次数和种群必须为正整数（种群至少2），时限必须为有限正数；请提供有效值。"})
                elif spec["type"] == "resource_path":
                    from src.agents.context.resources import resolve_model_file
                    resource = resolve_model_file(item)
                    valid = valid and isinstance(item, str) and resource.path is not None
                    evidence.append(resource.to_dict())
                else:
                    valid = valid and isinstance(item, str) and bool(item.strip())
            if task.verified_intent.intent == "optimization_request" and name == "target" and valid:
                from src.optimization.targets import parse_purity_target
                try:
                    parse_purity_target(value)
                except ValueError as exc:
                    valid = False
                    task.planning_blockers.append({"field": name, "message": str(exc), "source": "task_validation"})
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
        if task.verified_intent.intent == "optimization_request":
            diagnostics = [b["message"] for b in task.planning_blockers if b["field"] == name]
            if diagnostics:
                lines.extend(diagnostics)
                continue
            labels = {"max_evaluations": "总评估次数（正整数）", "max_seconds": "总时限（秒）",
                      "max_failures": "最大失败次数（正整数）", "pop_size": "每代种群数量",
                      "generations": "种群代数", "target": "纯度目标（如 99.99% 或 0.9999）"}
            if name == "max_evaluations" and task.parameters.get("generations") is not None:
                lines.append(f"已记录 {task.parameters['generations']} 代；代数与总评估次数分别设置，每代评估一个种群。")
            lines.append(f"请补充 {labels.get(name, name)}（{name}）。")
            continue
        values = list(dict.fromkeys(str(c["value"]) for c in task.candidates.get(name, [])))
        lines.append(f"请明确 {name}" + (f"（候选：{', '.join(values)}）" if values else "（当前证据不足，请补充信息或更新 case 证据）") + "。")
    return Answer(text=prefix + "\n".join(lines),
                  facts=[{"kind": "task_clarification", "task_id": task.task_id, "waiting_fields": fields,
                          "candidates": {f: task.candidates.get(f, []) for f in fields}}])


def merge_turn(task, decision, text):
    task.audit.append({"event": "turn_interpreted", "kind": decision["kind"], "message": text})
    updates = decision["updates"]
    def repairs_or_restates(name, value):
        if task.verified_intent.intent != "optimization_request":
            return False
        if name in task.missing_fields and task.parameter_sources.get(name, {}).get("valid") is False:
            return True
        if name == "target":
            from src.optimization.targets import parse_purity_target
            try:
                return parse_purity_target(task.parameters.get(name)) == parse_purity_target(value)
            except ValueError:
                pass
        return False

    # Invalid values can be repaired; valid established values still require
    # an explicit correction (equivalent purity notation is also harmless).
    if decision["kind"] == "answer" and any(
        (name not in task.missing_fields or task.parameter_sources.get(name, {}).get("source") == "user_goal")
        and task.parameters.get(name) not in (None, [], value)
        and not repairs_or_restates(name, value)
        for name, value in updates.items()
    ):
        task.audit.append({"event": "conflicting_answer"})
        return False
    if not updates:
        return False
    for name, value in updates.items():
        task.parameters[name] = value
        task.parameter_sources[name] = {"source": "user_reply", "text": text, "kind": decision["kind"]}
    task.planning_blockers = [b for b in task.planning_blockers
                              if b.get("source") != "optimization_request" or b["field"] not in updates]
    return True


def task_prompt(task):
    if task.verified_intent.intent == "optimization_request":
        return ("生成并解释当前优化计划。以下已验证参数是本次完整有效要求；更正前的用户目标只在审计历史中保留，不是当前要求。"
                "只说明这些参数，不推断未运行的变量贡献或可达性。\nVerified task parameters (current): "
                + json.dumps(task.parameters, ensure_ascii=False))
    return (task.original_goal + "\nVerified task parameters (current, supersede earlier goal values after corrections): "
            + json.dumps(task.parameters, ensure_ascii=False))
