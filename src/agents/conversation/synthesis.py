"""Evidence constrained final response generation for case conversations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.agents.llm_client import chat, is_configured, load_llm_config


SYNTHESIS_SYSTEM_PROMPT = """你是 PAO 的证据约束回答器。只允许使用输入 evidence 中的事实，不得补造设备、参数、物流、值、趋势、因果关系或外部资料。
每个结论和分析句都必须引用一个或多个已有 evidence id。输出必须是 JSON，不要 Markdown 代码围栏。
区分：configuration/model fact、statistical correlation、engineering inference、causal conclusion、unverified hypothesis。
YAML initial_value 只能称配置初始值，不能称仿真当前值；NodeDB catalog sample 只能称扫描快照值；没有 stream edge 时明确说无法可靠追踪；历史为空时不得声称趋势；IMA 仅在证据列出 ima 来源时可引用。
格式：{"conclusion":[{"text":"...","evidence_ids":["E1"],"kind":"fact|inference|hypothesis"}],"analysis":[...同结构...],"uncertainty":[...同结构...],"next_steps":[...同结构...]}
无法由 evidence 支持的内容不要输出。next_steps 可基于 evidence 的缺口提出检查或实验，但仍需引用指出缺口的 evidence。"""

SYNTHESIS_SYSTEM_PROMPT += """
The input includes task_evidence, mapping each specialist task to its evidence IDs.
Keep claims scoped: facts about one equipment must cite evidence that identifies that
same equipment. For comparisons, state each unit separately. Shared type claims may
cite both records. Do not use a constraint or parameter from one unit to describe
another. Keep the user's intent and answer naturally; these fields are internal only.
"""


@dataclass
class SynthesisResult:
    answer: str
    mode: str
    fallback_reason: str | None = None
    used_evidence_ids: list[str] | None = None
    diagnostics: dict[str, Any] | None = None


def _parse_json(raw: str) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.I | re.S)
    if match:
        cleaned = match.group(1)
    else:
        lo, hi = cleaned.find("{"), cleaned.rfind("}")
        if lo < 0 or hi <= lo:
            return None
        cleaned = cleaned[lo:hi + 1]
    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _claims(raw: Any, evidence_ids: set[str]) -> list[dict[str, Any]] | None:
    if not isinstance(raw, list) or len(raw) > 12:
        return None
    clean = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"text", "evidence_ids", "kind"}:
            return None
        text, refs, kind = item["text"], item["evidence_ids"], item["kind"]
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            return None
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in evidence_ids for ref in refs):
            return None
        if kind not in {"fact", "inference", "hypothesis"}:
            return None
        clean.append({"text": text.strip(), "evidence_ids": list(dict.fromkeys(refs)), "kind": kind})
    return clean


def _claim_supported(claim: dict[str, Any], evidence_by_id: dict[str, Any]) -> bool:
    cited = [evidence_by_id[ref] for ref in claim["evidence_ids"]]
    corpus = " ".join(
        json.dumps({"claim": item.claim, "value": item.value, "metadata": item.metadata},
                   ensure_ascii=False, default=str)
        for item in cited
    )
    text = claim["text"]
    # Check identifiers and explicit measurements, but do not interpret
    # ordinary counts such as “两个设备” as engineering measurements.
    identifiers = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z]\d{4,}\b", text))
    measurements = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?\s*(%|kPa|MPa|bar|℃|K|kg/h|kmol/h)(?![A-Za-z])", text, re.I)
    if any(token.casefold() not in corpus.casefold() for token in identifiers):
        return False
    for number, unit in re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(%|kPa|MPa|bar|℃|K|kg/h|kmol/h)(?![A-Za-z])", text, re.I):
        canonical_unit = unit.casefold()
        candidates = re.findall(r"(-?\d+(?:\.\d+)?)\s*(%|kPa|MPa|bar|℃|K|kg/h|kmol/h)(?![A-Za-z])", corpus, re.I)
        if not any(float(number) == float(observed) and canonical_unit == observed_unit.casefold()
                   for observed, observed_unit in candidates):
            # Evidence values may store the number and unit in separate fields.
            if not any(item.unit and item.unit.casefold() == canonical_unit
                       and isinstance(item.value, (int, float)) and float(item.value) == float(number)
                       for item in cited):
                return False
    equipment_ids = set(re.findall(r"(?<![A-Za-z0-9])[A-Z]\d{4,}(?![A-Za-z0-9])", text))
    scoped_citations: dict[str, list[Any]] = {}
    for equipment_id in equipment_ids:
        matching = [item for item in cited if equipment_id.casefold() in _equipment_scope(item)]
        if not matching:
            return False
        scoped_citations[equipment_id.casefold()] = matching
    # A claim about one unit cannot cite another unit's configuration as support.
    if len(equipment_ids) == 1:
        target = next(iter(equipment_ids)).casefold()
        if any(_equipment_scope(item) and target not in _equipment_scope(item) for item in cited):
            return False
    elif len(equipment_ids) > 1:
        # Keep per-unit comparisons atomic. Combined claims are allowed only for
        # a shared model type or an explicit unknown topology relation.
        shared_type = bool(re.search(r"\b(RadFrac|Heater|HeatX|Flash2|Mixer|Pump|Valve|RStoic|RCSTR|RPlug)\b", text, re.I))
        topology_unknown = any(
            item.metadata.get("layer") == "topology_missing"
            and len({x.casefold() for x in equipment_ids}.intersection(_equipment_scope(item))) == len(equipment_ids)
            for item in cited
        ) and any(word in text for word in ("无法确认", "无法可靠", "缺少", "没有连接边"))
        if not shared_type and not topology_unknown:
            return False
    # A comparison sentence citing multiple equipment records should not
    # attribute a parameter/value pair to the wrong unit.
    for parameter, value in re.findall(r"\b([A-Z][A-Z0-9_]+)\s*[=:]\s*(-?\d+(?:\.\d+)?)", text):
        scoped = cited
        if equipment_ids:
            scoped = [item for item in cited if any(
                equipment_id.casefold() in item.id.casefold()
                or equipment_id.casefold() == str(item.metadata.get("equipment", "")).casefold()
                or equipment_id.casefold() in [str(x).casefold() for x in item.metadata.get("related_equipment", [])]
                for equipment_id in equipment_ids
            )]
        if not any(parameter.casefold() in json.dumps({"claim": item.claim, "value": item.value}, ensure_ascii=False, default=str).casefold()
                   and value.casefold() in json.dumps({"claim": item.claim, "value": item.value}, ensure_ascii=False, default=str).casefold()
                   for item in scoped):
            return False
    # Reject positive causal/trend assertions unless a cited item actually
    # contains that evidence. Negated boundary statements remain valid.
    strong_claims = ("导致", "造成", "因此使", "会提高", "会降低", "呈上升趋势", "呈下降趋势", "显著影响", "存在因果", "存在相关性")
    positive_claim = any(marker in text for marker in strong_claims)
    explicit_negation = any(marker in text for marker in ("不能证明", "不支持", "无法确认", "未建立", "不能据此", "不代表"))
    if positive_claim and not explicit_negation and not any(marker in corpus for marker in strong_claims):
        return False
    return True


def _equipment_scope(item: Any) -> set[str]:
    metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
    values = []
    for key in ("equipment_scope", "equipment", "related_equipment", "equipment_filter"):
        value = metadata.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(x) for x in value)
    serialized = " ".join((str(getattr(item, "id", "")), str(getattr(item, "claim", "")),
                           str(getattr(item, "source_ref", "")), json.dumps(getattr(item, "value", None), ensure_ascii=False, default=str)))
    values.extend(re.findall(r"(?<![A-Za-z0-9])[A-Z]\d{4,}(?![A-Za-z0-9])", serialized))
    return {value.casefold() for value in values if value}


def _validate_llm_payload(parsed: dict[str, Any] | None, evidence: list[Any], allowed_ids: set[str] | None = None) -> tuple[dict[str, list[dict[str, Any]]] | None, str]:
    if parsed is None:
        return None, "parse_error: response is not valid JSON object"
    expected = {"conclusion", "analysis", "uncertainty", "next_steps"}
    if not {"conclusion"}.issubset(parsed) or set(parsed) - expected:
        return None, "schema_error: expected conclusion and optional analysis/uncertainty/next_steps"
    allowed = {item.id for item in evidence} if allowed_ids is None else set(allowed_ids)
    sections: dict[str, list[dict[str, Any]]] = {}
    invalid_refs = 0
    unsupported = 0
    by_id = {item.id: item for item in evidence if item.id in allowed}
    for key in expected:
        raw = parsed.get(key, [])
        if not isinstance(raw, list):
            return None, f"schema_error: {key} must be an array"
        valid: list[dict[str, Any]] = []
        for row in raw:
            # Preserve sound claims when another claim in the same response
            # contains a fabricated citation or unsupported concrete fact.
            if not isinstance(row, dict) or set(row) != {"text", "evidence_ids", "kind"}:
                unsupported += 1
                continue
            refs = row.get("evidence_ids")
            if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in allowed for ref in refs):
                invalid_refs += 1
                continue
            clean = _claims([row], allowed)
            if not clean or not _claim_supported(clean[0], by_id):
                unsupported += 1
                continue
            valid.extend(clean)
        sections[key] = valid
    if invalid_refs or unsupported:
        return sections, f"claim_validation: dropped {invalid_refs} invalid-citation and {unsupported} unsupported claims"
    if not sections["conclusion"]:
        return None, "schema_error: no valid conclusion claims"
    return sections, ""


def _local_answer(plan: Any, orchestration: Any, path: dict[str, str]) -> SynthesisResult:
    evidence = []
    seen = set()
    agent_results = []
    sources = set()
    for value in orchestration.task_results.values():
        agent_results.append(value)
        for item in getattr(value, "evidence", []):
            if item.id in seen:
                continue
            seen.add(item.id)
            evidence.append(item)
            sources.add(str(item.source.value))
    task_by_agent: dict[str, list[Any]] = {}
    for task_id, value in orchestration.task_results.items():
        task_by_agent.setdefault(task_id.split(".")[0], []).append(value)
    findings: list[str] = []
    missing: list[str] = []
    uncertainty: list[str] = []
    next_steps: list[str] = []
    for value in orchestration.task_results.values():
        findings.extend(str(x) for x in getattr(value, "findings", []) if x)
        missing.extend(str(x) for x in getattr(value, "missing", []) if x)
        uncertainty.extend(str(x) for x in getattr(value, "uncertainty", []) if x)
        next_steps.extend(str(x) for x in getattr(value, "next_steps", []) if x)
    missing.extend(orchestration.clarifications)
    if not evidence and plan.needs_clarification:
        conclusion = plan.clarification
    else:
        conclusion = "当前案例没有足够证据回答这个问题。"

    def evidence_for(evidence_id: str) -> Any:
        return next((item for item in evidence if item.id == evidence_id), None)

    def agent_named(name: str) -> Any:
        return next((item for item in agent_results if getattr(item, "agent", "") == name), None)

    conclusion_refs: list[str] = []
    intent = plan.intent
    if evidence:
        if intent in {"case_overview", "comprehensive_analysis"}:
            count_fact = evidence_for("aspen.case.equipment_count")
            yaml_fact = evidence_for("yaml.summary")
            facts = []
            if count_fact:
                count = (count_fact.value or {}).get("equipment_count", "未知")
                facts.append(f"匹配案例 catalog 记录 {count} 台设备")
                conclusion_refs.append(count_fact.id)
            if yaml_fact:
                value = yaml_fact.value or {}
                facts.append(f"YAML 记录 {len(value.get('design_variables', []))} 个设计变量、{len(value.get('objectives', []))} 个目标和 {len(value.get('constraints', []))} 个约束")
                conclusion_refs.append(yaml_fact.id)
            if facts:
                conclusion = "；".join(facts) + "。设备名称和配置条目见下方证据。"
        elif intent in {"equipment_parameters", "equipment_explanation"}:
            name = (plan.entities.get("equipment") or [""])[0]
            equipment_fact = evidence_for(f"aspen.equipment.{name}")
            no_vars = evidence_for(f"yaml.no_variables.{name}")
            params = [item for item in evidence if item.id.startswith(f"nodedb.parameter.{name}.")]
            if equipment_fact:
                kind = (equipment_fact.value or {}).get("block_type") or "未记录"
                conclusion = f"{name} 的 Aspen catalog 类型为 {kind}。"
                conclusion_refs.append(equipment_fact.id)
            if no_vars:
                conclusion += f"当前 YAML 没有记录与 {name} 关联的可调设计变量。"
                conclusion_refs.append(no_vars.id)
            if params:
                conclusion += f"NodeDB 提供 {len(params)} 个输入节点扫描快照值。"
                conclusion_refs.append(params[0].id)
            if intent == "equipment_explanation":
                objective_refs = [item for item in evidence if item.id.startswith(f"yaml.equipment_objective.{name}.")]
                objective_names = [item.value.get("name") for item in objective_refs if isinstance(item.value, dict) and item.value.get("name")]
                if objective_names:
                    conclusion += f"案例配置将该设备的输出路径用于目标 {', '.join(objective_names)}；路径关联本身不证明因果作用。"
                    conclusion_refs.extend(item.id for item in objective_refs)
                role_boundary = evidence_for(f"model.role_boundary.{name}")
                topology_gap = evidence_for("nodedb.topology.edges_missing")
                if role_boundary and topology_gap:
                    conclusion += "由于缺少可验证的进出物流边，当前不能确认其实际分离对象或流程连接关系。"
                    conclusion_refs.extend([role_boundary.id, topology_gap.id])
        elif intent == "equipment_comparison":
            facts = []
            for name in plan.entities.get("equipment", []):
                equipment_fact = evidence_for(f"aspen.equipment.{name}")
                variables = [item for item in evidence if item.id.startswith("yaml.variable.") and name.casefold() in _equipment_scope(item)]
                no_variables = evidence_for(f"yaml.no_variables.{name}")
                if equipment_fact:
                    facts.append(f"{name} 的 catalog 类型为 {(equipment_fact.value or {}).get('block_type') or '未记录'}")
                    conclusion_refs.append(equipment_fact.id)
                if variables:
                    facts.append(f"配置记录 {len(variables)} 个与 {name} 关联的设计变量")
                    conclusion_refs.extend(item.id for item in variables)
                elif no_variables:
                    facts.append(f"YAML 未记录与 {name} 关联的设计变量")
                    conclusion_refs.append(no_variables.id)
            if facts:
                conclusion = "；".join(facts) + "。类型相同不能证明工艺任务相同。"
                topology_gap = evidence_for("nodedb.topology.edges_missing")
                if topology_gap:
                    conclusion += "当前没有 stream 连接边，无法确认两者的流程连接关系。"
                    conclusion_refs.append(topology_gap.id)
        elif intent == "parameter_impact":
            no_vars = next((item for item in evidence if item.id.startswith("yaml.no_variables.")), None)
            variables = [item for item in evidence if item.id.startswith("yaml.variable.")]
            objectives = [item for item in evidence if item.id.startswith("yaml.objective.")]
            if no_vars:
                conclusion = no_vars.claim + "；因此当前证据不能确认该设备参数对目标的影响。"
                conclusion_refs = [no_vars.id] + [item.id for item in objectives[:2]]
            elif variables:
                conclusion = f"已读取 {len(variables)} 个相关 YAML 设计变量和 {len(objectives)} 个目标定义；这建立配置关联，不构成因果结论。"
                conclusion_refs = [item.id for item in variables[:2] + objectives[:2]]
        elif intent in {"result_reliability_diagnosis", "simulation_diagnosis"}:
            summary = evidence_for("simulation.diagnostic.summary") or evidence_for("simulation.history.empty")
            diagnostic = agent_named("SimulationDiagnosticAgent")
            if summary:
                conclusion = summary.claim + "。"
                conclusion_refs = [summary.id]
            elif diagnostic and diagnostic.findings:
                conclusion = diagnostic.findings[0]
                conclusion_refs = [x.id for x in diagnostic.evidence[:2]]
        elif intent == "optimization_analysis":
            comparison = evidence_for("optimization.latest_comparison")
            priority = [item for item in evidence if item.id.startswith("optimization.parameter_priority.")]
            pareto = evidence_for("optimization.pareto.front")
            summary = evidence_for("optimization.history.summary") or evidence_for("simulation.history.empty")
            if priority:
                conclusion = "当前实验候选排序依据显式边界、目标路径关联和历史样本数；不代表参数影响大小。"
                conclusion_refs = [item.id for item in priority[:3]]
            elif pareto:
                conclusion = pareto.claim + "。"
                conclusion_refs = [pareto.id]
            elif comparison:
                conclusion = comparison.claim
                conclusion_refs = [comparison.id]
            elif summary:
                conclusion = summary.claim + "。"
                conclusion_refs = [summary.id]
        elif intent == "case_comparison":
            comparison = evidence_for("optimization.latest_comparison")
            summary = evidence_for("optimization.history.summary") or evidence_for("simulation.history.empty")
            if comparison:
                conclusion = comparison.claim
                conclusion_refs = [comparison.id]
            elif summary:
                conclusion = summary.claim + "。"
                conclusion_refs = [summary.id]
        elif intent == "topology_analysis":
            topology = evidence_for("nodedb.topology.edges_missing")
            if topology:
                conclusion = "NodeDB 当前没有 stream 连接边，无法可靠追踪上下游。"
                conclusion_refs = [topology.id]
        if not conclusion_refs:
            conclusion = "已按当前案例范围完成只读检索；下列条目逐项保留了来源和引用。"
            conclusion_refs = [item.id for item in evidence[:2]]

    conclusion_text = conclusion + (" [" + ", ".join(dict.fromkeys(conclusion_refs)) + "]" if conclusion_refs else "")
    lines = [f"结论：{conclusion_text}"]
    lines.append("证据：")
    if evidence:
        for item in evidence[:30]:
            ref = f"；引用 {item.source_ref}" if item.source_ref else ""
            lines.append(f"- [{item.id}] {item.claim}（{item.source.value}{ref}）")
    else:
        lines.append("- 暂无可用证据。")
    if findings:
        analysis_lines = []
        for task in plan.tasks:
            result = orchestration.task_results.get(task.id)
            if result is None:
                continue
            refs = [item.id for item in getattr(result, "evidence", [])]
            for finding in getattr(result, "findings", []):
                named = set(re.findall(r"(?<![A-Za-z0-9])[A-Z]\d{4,}(?![A-Za-z0-9])", finding))
                if named:
                    refs = [item.id for item in getattr(result, "evidence", [])
                            if not _equipment_scope(item) or named.intersection(_equipment_scope(item))]
                elif len(task.entities.get("equipment", [])) == 1:
                    unit = task.entities["equipment"][0].casefold()
                    refs = [item.id for item in getattr(result, "evidence", [])
                            if not _equipment_scope(item) or unit in _equipment_scope(item)]
                citation = " [" + ", ".join(refs) + "]" if refs else ""
                analysis_lines.append(f"- {finding}{citation}")
        if analysis_lines:
            lines.append("分析：\n" + "\n".join(dict.fromkeys(analysis_lines)))
    else:
        lines.append("分析：未形成超出证据本身的推断。")
    uncertainties = list(dict.fromkeys(missing + uncertainty))
    if not uncertainties:
        uncertainties.extend(["规划或回答采用本地规则，未调用 LLM。"])
    uncertainty_lines = []
    for task in plan.tasks:
        result = orchestration.task_results.get(task.id)
        if result is None:
            continue
        refs = [item.id for item in getattr(result, "evidence", [])]
        for warning in list(getattr(result, "missing", [])) + list(getattr(result, "uncertainty", [])):
            citation = " [" + ", ".join(refs[:3]) + "]" if refs else ""
            uncertainty_lines.append(f"{warning}{citation}")
    if not uncertainty_lines:
        uncertainty_lines.extend(uncertainties)
    lines.append("不确定性：" + ("；".join(dict.fromkeys(uncertainty_lines)) if uncertainty_lines else "结论仅限当前列出的案例数据。"))
    suggestions = list(dict.fromkeys(next_steps))
    if not suggestions and uncertainties:
        suggestions.append("补齐列出的缺失数据后再判断。")
    next_lines = []
    for task in plan.tasks:
        result = orchestration.task_results.get(task.id)
        if result is None:
            continue
        refs = [item.id for item in getattr(result, "evidence", [])]
        for suggestion in getattr(result, "next_steps", []):
            citation = " [" + ", ".join(refs[:3]) + "]" if refs else ""
            next_lines.append(f"- {suggestion}{citation}")
    if not next_lines and suggestions:
        refs = [item.id for item in evidence[:3]]
        citation = " [" + ", ".join(refs) + "]" if refs else ""
        next_lines = [f"- {suggestion}{citation}" for suggestion in suggestions]
    lines.append("下一步：\n" + "\n".join(dict.fromkeys(next_lines)) if next_lines else "下一步：无额外建议。")
    lines.append("来源：" + (", ".join(sorted(sources)) if sources else "暂无可用来源"))
    lines.append("执行路径：" + "; ".join(f"{key}={value}" for key, value in path.items()))
    return SynthesisResult("\n".join(lines), "local_rules", path.get("fallback"), [x.id for x in evidence])


def _render_llm(parsed: dict[str, Any], evidence: list[Any], path: dict[str, str], *, validated: bool = False) -> SynthesisResult | None:
    if validated:
        sections = {key: parsed.get(key, []) for key in ("conclusion", "analysis", "uncertainty", "next_steps")}
        if not isinstance(sections["conclusion"], list) or not sections["conclusion"]:
            return None
    else:
        sections, error = _validate_llm_payload(parsed, evidence)
        if error or sections is None:
            return None
    lines = []
    used: list[str] = []
    for key, title in (("conclusion", "结论"), ("analysis", "分析"), ("uncertainty", "不确定性"), ("next_steps", "下一步")):
        claims = sections[key] or []
        if not claims:
            continue
        rendered = []
        for item in claims:
            refs = item["evidence_ids"]
            used.extend(refs)
            citation = ", ".join(refs)
            rendered.append(f"- {item['text']} [{citation}] ({item['kind']})")
        lines.append(f"{title}：\n" + "\n".join(rendered))
    selected = [item for item in evidence if item.id in set(used)]
    sources = sorted({str(item.source.value) for item in selected})
    lines.append("证据：")
    for item in selected[:30]:
        ref = f"；引用 {item.source_ref}" if item.source_ref else ""
        lines.append(f"- [{item.id}] {item.claim}（{item.source.value}{ref}）")
    lines.append("来源：" + (", ".join(sources) if sources else "暂无可用来源"))
    lines.append("执行路径：" + "; ".join(f"{key}={value}" for key, value in path.items()))
    return SynthesisResult("\n".join(lines), "llm", used_evidence_ids=list(dict.fromkeys(used)))


def synthesize_answer(
    question: str,
    plan: Any,
    orchestration: Any,
    *,
    llm_config: Any = None,
    chat_fn: Callable[..., str] | None = None,
    path: dict[str, str] | None = None,
) -> SynthesisResult:
    """Synthesize from specialist evidence, or render a local cited fallback."""
    path = dict(path or {})
    evidence = []
    seen = set()
    for value in orchestration.task_results.values():
        for item in getattr(value, "evidence", []):
            if item.id not in seen:
                seen.add(item.id)
                evidence.append(item)
    if not evidence or plan.needs_clarification or plan.planner_mode != "llm":
        path.setdefault("synthesizer", "local_rules")
        if plan.planner_mode != "llm":
            path.setdefault("fallback", plan.fallback_reason or "planner=local_rules")
        return _local_answer(plan, orchestration, path)

    cfg = llm_config if llm_config is not None else load_llm_config()
    if not is_configured(cfg):
        path["synthesizer"] = "local_rules"
        path["fallback"] = "LLM 合成器未配置"
        local = _local_answer(plan, orchestration, path)
        return SynthesisResult(local.answer, local.mode, path["fallback"], local.used_evidence_ids)

    # Round-robin selection prevents the first equipment from exhausting the
    # evidence budget. Validate against precisely this transmitted packet.
    groups = [list(getattr(result, "evidence", [])) for result in orchestration.task_results.values()]
    selected = []
    selected_ids: set[str] = set()
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group) and group[index].id not in selected_ids and len(selected) < 100:
                selected.append(group[index])
                selected_ids.add(group[index].id)
    evidence = selected
    compact_evidence = []
    for item in evidence:
        compact_evidence.append({
            "id": item.id, "source": item.source.value, "source_ref": item.source_ref,
            "claim": item.claim, "value": item.value, "unit": item.unit,
            "confidence": item.confidence, "metadata": item.metadata,
        })
    findings = []
    missing = []
    for value in orchestration.task_results.values():
        findings.extend(str(x) for x in getattr(value, "findings", []) if x)
        missing.extend(str(x) for x in getattr(value, "missing", []) if x)
    task_evidence = {}
    for task_id, result in orchestration.task_results.items():
        task_evidence[task_id] = [item.id for item in getattr(result, "evidence", []) if item.id in selected_ids]
    payload = {
        "question": question,
        "plan": {"intent": plan.intent, "entities": plan.entities, "tasks": [t.__dict__ for t in plan.tasks]},
        "agent_findings": list(dict.fromkeys(findings)),
        "missing_data": list(dict.fromkeys(missing + orchestration.clarifications)),
        "evidence": compact_evidence,
        "task_evidence": task_evidence,
    }
    caller = chat_fn or chat
    last_error = ""
    diagnostics: dict[str, Any] = {"attempts": 0, "errors": [], "provided_evidence_ids": sorted(selected_ids)}
    retained = None
    for attempt in range(2):
        diagnostics["attempts"] += 1
        retry_payload = dict(payload)
        if attempt:
            retry_payload["repair_instruction"] = (
                f"上次输出未通过校验：{last_error}。请只输出合法 JSON；保留可支持的分析，"
                "每条只引用 evidence 列出的 ID，实体和测量值必须由被引证据支持。"
            )
        try:
            raw = caller(cfg, system=SYNTHESIS_SYSTEM_PROMPT,
                         user=json.dumps(retry_payload, ensure_ascii=False, default=str))
        except Exception as exc:
            last_error = f"call_error:{type(exc).__name__}: {str(exc)[:160]}"
            diagnostics["errors"].append(last_error)
            break
        parsed = _parse_json(raw)
        checked, last_error = _validate_llm_payload(parsed, evidence, selected_ids)
        if last_error:
            diagnostics["errors"].append(last_error)
        if checked is not None and checked.get("conclusion"):
            # Render only validated claims. Invalid incidental claims do not
            # discard supported analysis or force a whole-answer fallback.
            normalized = dict(parsed)
            normalized.update(checked)
            rendered = _render_llm(normalized, evidence, {**path, "synthesizer": "llm", "fallback": "none"}, validated=True)
            if rendered is not None:
                rendered.diagnostics = diagnostics
                if not last_error:
                    return rendered
                retained = rendered
                diagnostics["partial_response"] = True
        if not last_error:
            last_error = "schema_error: no supported conclusion"
    if retained is not None:
        retained.diagnostics = diagnostics
        diagnostics["partial_response"] = True
        return retained
    path["synthesizer"] = "local_rules"
    path["fallback"] = f"LLM 合成降级：{last_error or 'schema_error'}"
    local = _local_answer(plan, orchestration, path)
    return SynthesisResult(local.answer, local.mode, path["fallback"], local.used_evidence_ids, diagnostics)
