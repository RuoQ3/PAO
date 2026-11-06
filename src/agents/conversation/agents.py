"""Read-only specialist agents used by the general conversation coordinator."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from statistics import correlation
from typing import Any

from src.agents.contracts.models import EvidenceItem, EvidenceSource, ProcessContext
from src.agents.technical_advisor import TechnicalAdvisor


@dataclass
class AgentResult:
    agent: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "evidence": [item.to_dict() for item in self.evidence],
            "findings": list(self.findings),
            "missing": list(self.missing),
            "uncertainty": list(self.uncertainty),
            "next_steps": list(self.next_steps),
        }


def _ev(
    evidence_id: str,
    source: EvidenceSource,
    claim: str,
    source_ref: str | None,
    value: Any,
    *,
    confidence: float = 0.95,
    metadata: dict[str, Any] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id, source=source, claim=claim, source_ref=source_ref,
        value=value, confidence=confidence, metadata=dict(metadata or {}),
    )


def _entity_scope(entities: dict[str, list[str]] | None, key: str, question: str, context: ProcessContext) -> list[str]:
    allowed = [str(x.get("name")) for x in context.equipment if x.get("name")] if key == "equipment" else [str(x.get("name")) for x in context.streams if x.get("name")]
    supplied = list((entities or {}).get(key, []))
    if not supplied:
        low = question.casefold()
        supplied = [name for name in allowed if name.casefold() in low]
    allowed_by_case = {name.casefold(): name for name in allowed}
    return list(dict.fromkeys(allowed_by_case[x.casefold()] for x in supplied if x.casefold() in allowed_by_case))


def _evidence_ref(context: ProcessContext, source: EvidenceSource) -> str:
    item = next((x for x in context.sources if x.source == source and x.source_ref), None)
    return item.source_ref if item else context.model_id


def _path_mentions_equipment(path: str, equipment: str) -> bool:
    parts = [x.casefold() for x in re.split(r"[\\/]+", path) if x]
    return len(parts) >= 3 and parts[0] == "data" and parts[1] == "blocks" and parts[2] == equipment.casefold()


def _yaml_variable(context: ProcessContext, key: str, value: dict[str, Any]) -> EvidenceItem:
    name = str(value.get("name") or key)
    path = str(value.get("aspen_path") or value.get("path") or "")
    return _ev(
        f"yaml.variable.{name}", EvidenceSource.YAML,
        f"YAML 定义设计变量 {name}" + (f"，路径 {path}" if path else ""),
        _evidence_ref(context, EvidenceSource.YAML), value,
        metadata={"layer": "configuration", "related_equipment": list(value.get("related_equipment", []))},
    )


def _history_row(item: EvidenceItem) -> dict[str, Any]:
    value = item.value if isinstance(item.value, dict) else {}
    return dict(value)


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default
    return default


def _objectives(row: dict[str, Any]) -> list[dict[str, Any]]:
    data = row.get("objectives", row.get("objectives_json", []))
    parsed = _json_value(data, [])
    if isinstance(parsed, dict):
        return [{"name": key, "value": value, "available": value is not None} for key, value in parsed.items()]
    return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []


def _constraints(row: dict[str, Any]) -> list[dict[str, Any]]:
    data = row.get("constraints", row.get("constraints_json", []))
    parsed = _json_value(data, [])
    return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []


def _design_vars(row: dict[str, Any]) -> dict[str, Any]:
    raw = _json_value(row.get("design_vars", {}), {})
    return raw if isinstance(raw, dict) else {}


def _numeric(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


class CaseOverviewAgent:
    name = "CaseOverviewAgent"

    def run(self, question: str, context: ProcessContext, task: Any = None, *, entities: dict[str, list[str]] | None = None) -> AgentResult:
        result = AgentResult(self.name)
        task_text = str(getattr(task, "objective", ""))
        equipment_requested = any(word in question + task_text for word in ("设备", "设备类型", "设备清单", "流程"))
        variables_requested = any(word in question + task_text for word in ("变量", "设计变量", "参数"))
        objectives_requested = any(word in question + task_text for word in ("目标", "优化"))
        constraints_requested = any(word in question + task_text for word in ("约束", "可行"))
        if equipment_requested:
            for equipment in context.equipment:
                name = str(equipment.get("name") or "")
                if not name:
                    continue
                result.evidence.append(_ev(
                    f"aspen.equipment.{name}", EvidenceSource.ASPEN_MODEL,
                    f"Aspen/NodeDB catalog 记录设备 {name}，类型 {equipment.get('block_type') or '未记录'}",
                    str(equipment.get("aspen_path") or context.model_id), equipment,
                    metadata={"layer": "model_structure", "retrieved_from": "nodedb"},
                ))
        for item in context.sources:
            if item.source == EvidenceSource.YAML:
                if (item.id == "yaml.summary"
                        or variables_requested and item.id.startswith("yaml.variable.")
                        or objectives_requested and item.id.startswith("yaml.objective.")
                        or constraints_requested and item.id.startswith("yaml.constraint.")):
                    result.evidence.append(item)
        requested_data = []
        if equipment_requested:
            result.evidence.append(_ev(
                "aspen.case.equipment_count", EvidenceSource.ASPEN_MODEL,
                f"当前匹配的 Aspen/NodeDB catalog 包含 {len(context.equipment)} 个设备节点",
                _evidence_ref(context, EvidenceSource.ASPEN_MODEL),
                {"equipment_count": len(context.equipment), "equipment": [x.get("name") for x in context.equipment]},
                metadata={"layer": "case_overview", "retrieved_from": "nodedb"},
            ))
            requested_data.append(f"{len(context.equipment)} 个设备")
        if variables_requested:
            requested_data.append(f"{len(context.variables)} 个 YAML 设计变量")
        if objectives_requested:
            requested_data.append(f"{len(context.objectives)} 个目标")
        if constraints_requested:
            requested_data.append(f"{len(context.constraints)} 个约束")
        if requested_data:
            result.findings.append("读取到" + "、".join(requested_data) + "。")
        else:
            result.findings.append(f"YAML 记录 {len(context.objectives)} 个目标和 {len(context.constraints)} 个约束。")
        if equipment_requested and not context.equipment:
            result.evidence.append(_ev(
                "nodedb.catalog.empty", EvidenceSource.NODEDB,
                "当前案例没有从 NodeDB catalog 读取到设备节点",
                _evidence_ref(context, EvidenceSource.NODEDB), {"equipment_count": 0},
                metadata={"layer": "case_overview", "complete": False},
            ))
            result.missing.append("当前 NodeDB/Aspen catalog 未提供设备节点。")
        if objectives_requested and not context.objectives:
            result.missing.append("YAML 未记录目标函数。")
        if constraints_requested and not context.constraints:
            result.missing.append("YAML 未记录约束。")
        return result


class EquipmentAgent:
    name = "EquipmentAgent"

    def __init__(self, ima_client: Any = None):
        self.ima_client = ima_client

    @staticmethod
    def _rank_parameter(item: dict[str, Any]) -> tuple[int, str]:
        name = str(item.get("name") or "")
        upper = name.upper()
        exact = {"B:F", "D:F", "BASIS_RR", "FEED_STAGE", "NSTAGE", "N_STAGES", "PRES", "PRES1", "TEMP"}
        rank = 0 if upper in exact else 1 if any(x in upper for x in ("STAGE", "FLOW", "PRES", "TEMP", "DUTY", "RATIO", "RR")) else 2
        return rank, name.casefold()

    def run(self, question: str, context: ProcessContext, task: Any = None, *, entities: dict[str, list[str]] | None = None) -> AgentResult:
        result = AgentResult(self.name)
        equipment_names = _entity_scope(entities, "equipment", question, context)
        if not equipment_names:
            result.missing.append("问题没有解析到当前案例中的设备。")
            return result
        for name in equipment_names:
            equipment = next(x for x in context.equipment if str(x.get("name")) == name)
            path = str(equipment.get("aspen_path") or rf"\Data\Blocks\{name}")
            result.evidence.append(_ev(
                f"aspen.equipment.{name}", EvidenceSource.ASPEN_MODEL,
                f"设备 {name} 的 Aspen catalog 类型为 {equipment.get('block_type') or '未记录'}，路径 {path}",
                path, equipment, metadata={"layer": "model_structure"},
            ))
            variables = [
                (key, var) for key, var in context.variables.items()
                if name in [str(x) for x in var.get("related_equipment", [])]
            ]
            if variables:
                for key, variable in variables:
                    result.evidence.append(_yaml_variable(context, key, variable))
            else:
                result.evidence.append(_ev(
                    f"yaml.no_variables.{name}", EvidenceSource.YAML,
                    f"当前 YAML 中没有记录与设备 {name} 关联的 design_variables",
                    _evidence_ref(context, EvidenceSource.YAML), [],
                    metadata={"layer": "configuration", "equipment_filter": name, "count": 0},
                ))
                result.findings.append(f"当前配置未记录 {name} 的可调设计变量。")

            candidates = sorted(context.equipment_parameters.get(name, []), key=self._rank_parameter)
            if candidates:
                result.uncertainty.append(
                    f"{name} 的 NodeDB 参数值来自 catalog 扫描快照，不代表实时 Aspen 状态或本次仿真当前值。"
                )
                for index, item in enumerate(candidates[:12]):
                    param_name = str(item.get("name") or "")
                    path_ref = str(item.get("aspen_path") or "")
                    value = item.get("value")
                    result.evidence.append(_ev(
                        f"nodedb.parameter.{name}.{index}", EvidenceSource.NODEDB,
                        f"NodeDB catalog 快照记录 {name} 输入节点 {param_name}={value}" + (f" {item.get('unit')}" if item.get("unit") else ""),
                        f"{item.get('catalog_id') or context.model_id}:{path_ref}", item,
                        metadata={"layer": "model_input_snapshot", "value_stage": "catalog_snapshot", "equipment": name},
                    ))
            else:
                result.missing.append(f"NodeDB 没有 {name} 的可用 Input 节点值快照。")

            values = [
                (node_path, value) for node_path, value in context.current_values.items()
                if _path_mentions_equipment(node_path, name)
            ]
            for index, (node_path, value) in enumerate(values[:12]):
                result.evidence.append(_ev(
                    f"aspen.case_value.{name}.{index}", EvidenceSource.ASPEN_MODEL,
                    f"NodeDB 工况节点读数 {node_path}={value}", node_path, value,
                    metadata={"layer": "case_node_value"},
                ))

            related_constraints = []
            for constraint in context.constraints:
                path_or_var = str(constraint.get("aspen_path") or constraint.get("path") or "")
                variable_name = str(constraint.get("variable") or "")
                if _path_mentions_equipment(path_or_var, name) or variable_name in {str(k) for k, _ in variables}:
                    related_constraints.append(constraint)
            if related_constraints:
                for index, constraint in enumerate(related_constraints):
                    result.evidence.append(_ev(
                        f"yaml.constraint.{name}.{index}", EvidenceSource.YAML,
                        f"YAML 记录与 {name} 路径直接关联的约束 {constraint.get('name') or constraint}",
                        _evidence_ref(context, EvidenceSource.YAML), constraint,
                        metadata={"layer": "configuration", "equipment_filter": name},
                    ))
            else:
                result.evidence.append(_ev(
                    f"yaml.no_constraint.{name}", EvidenceSource.YAML,
                    f"YAML 中没有通过 Aspen 路径或 design_variable 直接关联到 {name} 的约束",
                    _evidence_ref(context, EvidenceSource.YAML), [],
                    metadata={"layer": "configuration", "equipment_filter": name, "count": 0},
                ))
                result.findings.append(f"没有找到通过 Aspen 路径或 design_variable 直接关联到 {name} 的约束。")

            for objective in context.objectives:
                objective_path = str(objective.get("aspen_path") or "")
                if _path_mentions_equipment(objective_path, name):
                    result.evidence.append(_ev(
                        f"yaml.equipment_objective.{name}.{objective.get('name')}", EvidenceSource.YAML,
                        f"YAML 目标 {objective.get('name')} 直接读取 {name} 路径 {objective_path}；路径关联不证明因果影响",
                        _evidence_ref(context, EvidenceSource.YAML), objective,
                        metadata={"equipment": name, "layer": "configuration", "causal": False},
                    ))
            if _mentions_explanation(question):
                result.evidence.append(_ev(
                    f"model.role_boundary.{name}", EvidenceSource.ASPEN_MODEL,
                    f"{name} 的 catalog 类型 {equipment.get('block_type')} 仅确定模型单元类型，不能单独确定具体分离任务、目标产品或工艺角色",
                    path, {"equipment": name, "role_verified": False},
                    metadata={"equipment": name, "layer": "knowledge_boundary"},
                ))
                result.uncertainty.append(f"{name} 的具体工艺角色还需进出物流、组分和目标产品证据核对。")

            # Existing technical advisor is used only for model-specific
            # equipment explanations. Its external answer remains tagged IMA.
            if self.ima_client is not None and _mentions_explanation(question):
                scoped_question = f"请仅解释设备 {name} 的模型类型和可能作用，具体工艺任务需案例证据验证。"
                try:
                    scoped_context = replace(
                        context, equipment=[equipment],
                        variables={key: var for key, var in variables},
                        constraints=related_constraints,
                        objectives=[obj for obj in context.objectives if _path_mentions_equipment(str(obj.get("aspen_path") or ""), name)],
                        streams=[], topology={"nodes": [equipment], "edges": []},
                        current_values={p: v for p, v in context.current_values.items() if _path_mentions_equipment(p, name)},
                        equipment_parameters={name: context.equipment_parameters.get(name, [])},
                        sources=[], history_evidence=[],
                    )
                    advice = TechnicalAdvisor(self.ima_client).advise(scoped_question, scoped_context)
                    if advice.ima_answer is not None and advice.ima_answer.answer and advice.ima_answer.sources:
                        result.evidence.append(_ev(
                            f"ima.answer.{name}", EvidenceSource.IMA,
                            advice.ima_answer.answer, advice.ima_answer.request_id or advice.ima_answer.source,
                            advice.ima_answer.answer,
                            confidence=advice.ima_answer.confidence or 0.5,
                            metadata={"layer": "external_reference", "assumptions": advice.ima_answer.assumptions},
                        ))
                        for index, citation in enumerate(advice.ima_answer.sources):
                            result.evidence.append(_ev(
                                f"ima.citation.{name}.{index}", EvidenceSource.IMA,
                                citation.snippet or citation.title or citation.source,
                                citation.source, citation.to_dict(),
                                confidence=citation.confidence or 0.7,
                                metadata={"layer": "citation", **citation.to_dict()},
                            ))
                except Exception as exc:
                    result.uncertainty.append(f"IMA 不可用：{type(exc).__name__}")
            if _mentions_explanation(question) and not any(item.source == EvidenceSource.IMA for item in result.evidence):
                result.uncertainty.append(f"{name} 未取得 IMA 外部机理资料。")
        for item in result.evidence:
            scoped = [name for name in equipment_names if name.casefold() in (
                item.id + " " + str(item.value) + " " + item.claim
            ).casefold()]
            item.metadata.setdefault("equipment_scope", scoped)
        return result


def _mentions_explanation(question: str) -> bool:
    return any(term in question for term in ("是什么设备", "什么设备", "设备类型", "作用", "干什么", "做什么", "功能", "角色", "介绍设备", "what is", "what equipment"))


class TopologyAgent:
    name = "TopologyAgent"

    def run(self, question: str, context: ProcessContext, task: Any = None, *, entities: dict[str, list[str]] | None = None) -> AgentResult:
        result = AgentResult(self.name)
        equipment_names = set(_entity_scope(entities, "equipment", question, context))
        stream_names = set(_entity_scope(entities, "streams", question, context))
        edges = list(context.topology.get("edges") or [])
        if not edges:
            ref = _evidence_ref(context, EvidenceSource.ASPEN_MODEL)
            result.evidence.append(_ev(
                "nodedb.topology.edges_missing", EvidenceSource.NODEDB,
                "当前 NodeDB catalog/ProcessContext 没有可用的 stream 连接边",
                ref, {"edge_count": 0, "equipment_filter": sorted(equipment_names), "stream_filter": sorted(stream_names)},
                confidence=0.98, metadata={"layer": "topology_missing", "complete": False,
                                           "equipment_filter": sorted(equipment_names)},
            ))
            result.missing.append("NodeDB 中没有 stream 边，无法可靠追踪上下游；不会根据设备类型推测连接关系。")
            result.uncertainty.append("拓扑缺少可验证的 stream source/target 边")
            return result
        selected = []
        for edge in edges:
            edge_id = str(edge.get("id") or edge.get("label") or "")
            endpoints = {str(edge.get("source") or ""), str(edge.get("target") or "")}
            if (not equipment_names and not stream_names) or edge_id in stream_names or endpoints.intersection(equipment_names):
                selected.append(edge)
        for index, edge in enumerate(selected):
            result.evidence.append(_ev(
                f"nodedb.topology.edge.{index}", EvidenceSource.NODEDB,
                f"拓扑边 {edge.get('id') or edge.get('label')}: {edge.get('source')} → {edge.get('target')}",
                str(edge.get("source_ref") or context.model_id), edge,
                metadata={"layer": "topology_edge"},
            ))
        if not selected:
            result.missing.append("拓扑快照没有与指定设备/物流相交的边。")
        return result


class ParameterImpactAgent:
    name = "ParameterImpactAgent"

    def run(self, question: str, context: ProcessContext, task: Any = None, *, entities: dict[str, list[str]] | None = None) -> AgentResult:
        result = AgentResult(self.name)
        target_equipment = _entity_scope(entities, "equipment", question, context)
        requested_variables = set((entities or {}).get("variables", []))
        variables = [
            (key, value) for key, value in context.variables.items()
            if (not target_equipment or set(target_equipment).intersection(value.get("related_equipment", [])))
            and (not requested_variables or str(value.get("name") or key) in requested_variables or key in requested_variables)
        ]
        if target_equipment and not variables:
            for equipment in target_equipment:
                result.evidence.append(_ev(
                    f"yaml.no_variables.{equipment}", EvidenceSource.YAML,
                    f"YAML 没有定义关联设备 {equipment} 的可调设计变量；不能将其他设备变量归给它",
                    _evidence_ref(context, EvidenceSource.YAML), [],
                    metadata={"layer": "configuration", "equipment_filter": equipment, "count": 0},
                ))
            result.findings.append("没有发现该设备的 YAML 可调变量，无法从当前配置定量归因其对目标的影响。")
            if context.history_evidence:
                for equipment in target_equipment:
                    prefix = f"\\Data\\Blocks\\{equipment}\\".casefold()
                    matching_samples = []
                    for item in context.history_evidence:
                        row = _history_row(item)
                        if any(str(path).casefold().startswith(prefix) for path in _design_vars(row)):
                            matching_samples.append(str(row.get("case_id") or item.source_ref))
                    result.evidence.append(_ev(
                        f"simulation.no_parameter_samples.{equipment}", EvidenceSource.SIMULATION_DB,
                        f"检查 {len(context.history_evidence)} 条历史记录，未找到设备 {equipment} 的 design_vars 样本",
                        "SimulationDB design_vars", {"equipment": equipment, "history_count": len(context.history_evidence), "matching_cases": matching_samples},
                        metadata={"layer": "history_scope", "causal": False},
                    ))
                result.missing.append("历史记录没有该设备的设计变量样本，无法估计其参数与目标的相关性。")
            else:
                result.evidence.append(_ev(
                    "simulation.history.empty", EvidenceSource.SIMULATION_DB,
                    "当前 SimulationDB 查询返回 0 条历史工况",
                    "SimulationDB query_cases(limit=100)", {"history_count": 0},
                    metadata={"layer": "history_summary", "complete": False},
                ))
                result.missing.append("SimulationDB 为空；没有历史相关性或敏感性证据。")
            result.next_steps.append(
                f"若需验证 {', '.join(target_equipment)} 参数影响，先在 YAML 明确登记这些设备的设计变量和 Aspen 路径，再做受控小步长扰动并记录目标与约束余量。"
            )
        for key, variable in variables:
            result.evidence.append(_yaml_variable(context, key, variable))
            path = str(variable.get("aspen_path") or variable.get("path") or "")
            linked_objectives = []
            for objective in context.objectives:
                objective_path = str(objective.get("aspen_path") or objective.get("path") or "")
                if path and objective_path and _path_mentions_equipment(path, _path_equipment(path)) and _path_equipment(path) == _path_equipment(objective_path):
                    linked_objectives.append(objective)
            for objective in linked_objectives:
                result.evidence.append(_ev(
                    f"yaml.objective.link.{key}.{objective.get('name')}", EvidenceSource.YAML,
                    f"设计变量 {variable.get('name') or key} 与目标 {objective.get('name')} 的 Aspen 路径都位于设备 {_path_equipment(path)} 下",
                    _evidence_ref(context, EvidenceSource.YAML), {"variable_path": path, "objective_path": objective.get("aspen_path")},
                    metadata={"association": "shared_aspen_equipment_path", "causal": False},
                ))

        for index, objective in enumerate(context.objectives):
            result.evidence.append(_ev(
                f"yaml.objective.{objective.get('name') or index}", EvidenceSource.YAML,
                f"YAML 目标 {objective.get('name') or index}，方向={'最小化' if objective.get('minimize', True) else '最大化'}",
                _evidence_ref(context, EvidenceSource.YAML), objective,
                metadata={"layer": "objective_definition"},
            ))

        correlations = self._history_correlations(context, variables)
        if correlations:
            result.evidence.extend(correlations)
            result.findings.append("历史样本关联按统计相关性呈现，不能解释为因果结论。")
        elif context.history_evidence and variables:
            result.missing.append("历史记录没有足够的成功样本同时包含这些变量和目标值，无法估计相关性。")
        elif not context.history_evidence and not target_equipment:
            result.evidence.append(_ev(
                "simulation.history.empty", EvidenceSource.SIMULATION_DB,
                "当前 SimulationDB 查询返回 0 条历史工况，无法计算参数与目标的历史相关性",
                "SimulationDB query_cases(limit=100)", {"history_count": 0},
                metadata={"layer": "history_summary", "complete": False},
            ))
            result.missing.append("SimulationDB 为空；没有历史相关性或敏感性证据。")
        result.findings.extend([
            "工程推断：当前任务没有额外经过验证的工程推断。",
            "因果结论：当前证据没有受控参数扰动，不能给出因果结论。",
            "未验证假设：若要评估影响，应在约束边界内对单个变量做小步长扰动并记录结果。",
        ])
        if not context.objectives:
            result.missing.append("YAML 没有目标定义，无法判断参数与哪些目标相关。")
        return result

    @staticmethod
    def _history_correlations(context: ProcessContext, variables: list[tuple[str, dict[str, Any]]]) -> list[EvidenceItem]:
        if not variables:
            return []
        rows = [
            _history_row(item) for item in context.history_evidence
            if str(_history_row(item).get("status", "")).casefold() in {"success", "warnings"}
        ]
        output: list[EvidenceItem] = []
        for key, variable in variables:
            path = str(variable.get("aspen_path") or variable.get("path") or "")
            name = str(variable.get("name") or key)
            paired: dict[str, list[tuple[float, float]]] = {}
            for row in rows:
                x = _numeric(_design_vars(row).get(path)) if path else None
                if x is None:
                    x = _numeric(_design_vars(row).get(name))
                if x is None:
                    continue
                for objective in _objectives(row):
                    y = _numeric(objective.get("value")) if objective.get("available", True) else None
                    if y is not None and objective.get("name"):
                        paired.setdefault(str(objective["name"]), []).append((x, y))
            for objective_name, pairs in paired.items():
                if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
                    continue
                try:
                    value = correlation([x for x, _ in pairs], [y for _, y in pairs])
                except (ValueError, ArithmeticError):
                    continue
                output.append(_ev(
                    f"simulation.correlation.{name}.{objective_name}", EvidenceSource.SIMULATION_DB,
                    f"历史样本中 {name} 与 {objective_name} 的 Pearson 相关系数为 {value:.4f}（n={len(pairs)}）",
                    "SimulationDB successful paired cases", {"r": value, "n": len(pairs), "variable": name, "objective": objective_name},
                    confidence=min(0.95, 0.5 + len(pairs) / 100),
                    metadata={"kind": "correlation", "causal": False, "sample_count": len(pairs)},
                ))
        return output


def _path_equipment(path: str) -> str:
    parts = [x for x in re.split(r"[\\/]+", path) if x]
    return parts[2] if len(parts) >= 3 and parts[0].casefold() == "data" and parts[1].casefold() == "blocks" else ""


class SimulationDiagnosticAgent:
    name = "SimulationDiagnosticAgent"

    def run(self, question: str, context: ProcessContext, task: Any = None, *, entities: dict[str, list[str]] | None = None) -> AgentResult:
        result = AgentResult(self.name)
        if not context.history_evidence:
            result.evidence.append(_ev(
                "simulation.history.empty", EvidenceSource.SIMULATION_DB,
                "当前 SimulationDB 查询返回 0 条历史工况",
                "SimulationDB query_cases(limit=100)", {"history_count": 0},
                metadata={"layer": "history_summary", "complete": False},
            ))
            result.missing.append("SimulationDB 没有历史工况，无法核验本次收敛状态、目标读取或历史趋势。")
            result.uncertainty.append("当前会话没有绑定具体 case_id；无法确认所谓本次结果对应哪条记录。")
            return result
        rows = [_history_row(item) for item in context.history_evidence]
        status_counts: dict[str, int] = {}
        for row in rows[:12]:
            status = str(row.get("status") or "unknown")
            result.evidence.append(_ev(
                f"simulation.case.{row.get('case_id') or len(result.evidence)}", EvidenceSource.SIMULATION_DB,
                f"SimulationDB 工况 {row.get('case_id') or '未知'} 状态 {status}",
                str(row.get("case_id") or "SimulationDB"), row,
                metadata={"layer": "simulation_observation", "status": status},
            ))
        for row in rows:
            status = str(row.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        result.findings.append("历史状态计数：" + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
        sim_failures = sum(count for status, count in status_counts.items() if status.casefold() in {"sim_failed", "simulation_failed", "failed"})
        objective_errors = sum(count for status, count in status_counts.items() if "objective" in status.casefold() or "read" in status.casefold())
        constraint_errors = sum(count for status, count in status_counts.items() if status.casefold() in {"infeasible", "constraint_error"})
        result.evidence.append(_ev(
            "simulation.diagnostic.summary", EvidenceSource.SIMULATION_DB,
            f"SimulationDB 汇总 {len(rows)} 条历史记录；状态计数 {status_counts}",
            "SimulationDB query_cases(limit=100)",
            {"count": len(rows), "status_counts": status_counts,
             "objectives_available": sum(bool(row.get("objectives_available")) for row in rows),
             "constraints_available": sum(bool(row.get("constraints_available")) for row in rows)},
            metadata={"layer": "history_summary", "causal": False},
        ))
        result.findings.append(f"历史分类：仿真失败 {sim_failures} 条、目标读取异常 {objective_errors} 条、不可行/约束异常 {constraint_errors} 条。该分类不能单独确定根因。")
        if not any(row.get("objectives_available") for row in rows):
            result.missing.append("历史工况没有可用目标值标记，无法核验目标读取成功率。")
        for index, constraint in enumerate(context.constraints):
            result.evidence.append(_ev(
                f"yaml.constraint.{constraint.get('name') or index}", EvidenceSource.YAML,
                f"YAML 约束定义 {constraint.get('name') or index}",
                _evidence_ref(context, EvidenceSource.YAML), constraint,
                metadata={"layer": "constraint_definition"},
            ))
        constraint_samples = [c for row in rows for c in _constraints(row) if c.get("available", True)]
        if constraint_samples:
            satisfied = [c.get("satisfied") for c in constraint_samples if isinstance(c.get("satisfied"), bool)]
            result.findings.append(f"SimulationDB 可读到 {len(constraint_samples)} 条约束观察；其中 {sum(satisfied)} 条标记满足 / {len(satisfied)} 条有满足状态。")
        elif context.constraints:
            result.missing.append("有 YAML 约束定义，但历史约束观察为空。")
        result.uncertainty.append("历史数据库状态不能代替当前 Aspen 运行日志；未从本地数据确认失败根因。")
        return result


class OptimizationAgent:
    name = "OptimizationAgent"

    def run(self, question: str, context: ProcessContext, task: Any = None, *, entities: dict[str, list[str]] | None = None) -> AgentResult:
        result = AgentResult(self.name)
        target_equipment = _entity_scope(entities, "equipment", question, context)
        variables = [
            (key, variable) for key, variable in context.variables.items()
            if not target_equipment or set(target_equipment).intersection(variable.get("related_equipment", []))
        ]
        if not context.objectives:
            result.missing.append("YAML 未定义优化目标。")
        for index, objective in enumerate(context.objectives):
            name = str(objective.get("name") or index)
            result.evidence.append(_ev(
                f"yaml.optimization.objective.{name}", EvidenceSource.YAML,
                f"目标定义 {name}，{'最小化' if objective.get('minimize', True) else '最大化'}",
                _evidence_ref(context, EvidenceSource.YAML), objective,
                metadata={"layer": "objective_definition"},
            ))

        if variables and any(term in question for term in ("值得", "优先", "调整", "建议")):
            ranked = []
            successful_rows = [
                _history_row(item) for item in context.history_evidence
                if str(_history_row(item).get("status", "")).casefold() in {"success", "warnings"}
            ]
            for key, variable in variables:
                name = str(variable.get("name") or key)
                path = str(variable.get("aspen_path") or variable.get("path") or "")
                lower = variable.get("lower_bound") is not None
                upper = variable.get("upper_bound") is not None
                samples = 0
                for row in successful_rows:
                    row_vars = _design_vars(row)
                    if path and path in row_vars and _numeric(row_vars[path]) is not None:
                        samples += 1
                    elif name in row_vars and _numeric(row_vars[name]) is not None:
                        samples += 1
                direct_objectives = [
                    objective for objective in context.objectives
                    if _path_equipment(path) and _path_equipment(path) == _path_equipment(str(objective.get("aspen_path") or objective.get("path") or ""))
                ]
                # Transparent experiment-readiness score; it is not a
                # sensitivity score or a prediction of improvement.
                score = int(lower) + int(upper) + int(bool(direct_objectives))
                ranked.append((score, samples, name, direct_objectives, lower, upper))
            ranked.sort(key=lambda x: (-x[0], x[1], x[2].casefold()))
            labels = []
            for rank, (score, samples, name, direct, lower, upper) in enumerate(ranked, 1):
                payload = {
                    "rank": rank, "readiness_score": score,
                    "has_lower_bound": lower, "has_upper_bound": upper,
                    "history_sample_count": samples,
                    "direct_objective_paths": [x.get("name") for x in direct],
                    "variable": name,
                }
                result.evidence.append(_ev(
                    f"optimization.parameter_priority.{name}", EvidenceSource.OPTIMIZATION_RESULTS,
                    f"实验准备度候选排序 #{rank}: {name}，边界/目标路径准备分={score}，历史样本数={samples}；该分数不表示参数敏感性或改善幅度",
                    "YAML design_variables + objective paths + SimulationDB sample count", payload,
                    metadata={"method": "transparent_experiment_readiness", "causal": False},
                ))
                labels.append(f"#{rank} {name}（准备分 {score}，历史样本 {samples}）")
            result.findings.append("实验候选排序（仅根据显式边界、目标路径同设备关联和历史样本数，不代表影响大小）：" + "；".join(labels))

        if not context.history_evidence:
            result.evidence.append(_ev(
                "simulation.history.empty", EvidenceSource.SIMULATION_DB,
                "当前 SimulationDB 查询返回 0 条历史工况",
                "SimulationDB query_cases(limit=100)", {"history_count": 0},
                metadata={"layer": "optimization_history_summary", "complete": False},
            ))
            result.missing.append("SimulationDB 为空，无法比较历史目标、可行性或 Pareto 前沿。")
            return result
        rows = [_history_row(item) for item in context.history_evidence]
        rows.sort(key=lambda row: (int(row.get("iteration") or 0), str(row.get("created_at") or "")), reverse=True)
        usable = [row for row in rows if str(row.get("status", "")).casefold() in {"success", "warnings"} and row.get("objectives_available")]
        result.findings.append(f"历史记录 {len(rows)} 条，可比较的成功/警告且目标可用工况 {len(usable)} 条。")
        result.evidence.append(_ev(
            "optimization.history.summary", EvidenceSource.SIMULATION_DB,
            f"SimulationDB 共 {len(rows)} 条记录，其中 {len(usable)} 条为成功/警告且目标可用工况",
            "SimulationDB query_cases(limit=100)",
            {"history_count": len(rows), "usable_count": len(usable)},
            metadata={"layer": "optimization_history_summary"},
        ))
        for index, row in enumerate(usable[:12]):
            result.evidence.append(_ev(
                f"optimization.history.{row.get('case_id') or index}", EvidenceSource.SIMULATION_DB,
                f"历史工况 {row.get('case_id')}：status={row.get('status')} feasible={row.get('feasible')} objectives={_objectives(row)}",
                str(row.get("case_id") or "SimulationDB"), row,
                metadata={"layer": "optimization_history"},
            ))
        if len(usable) >= 2:
            current, prior = usable[0], usable[1]
            current_obj = {str(x.get("name")): _numeric(x.get("value")) for x in _objectives(current) if x.get("available", True)}
            prior_obj = {str(x.get("name")): _numeric(x.get("value")) for x in _objectives(prior) if x.get("available", True)}
            deltas = {}
            for name, value in current_obj.items():
                old = prior_obj.get(name)
                if value is not None and old is not None:
                    deltas[name] = {"latest": value, "previous": old, "delta": value - old}
            if deltas:
                result.evidence.append(_ev(
                    "optimization.latest_comparison", EvidenceSource.OPTIMIZATION_RESULTS,
                    f"SimulationDB 最新可用工况 {current.get('case_id')} 与上一可用工况 {prior.get('case_id')} 的目标差值 {deltas}",
                    f"{current.get('case_id')} vs {prior.get('case_id')}", deltas,
                    metadata={"method": "latest_two_database_records", "causal": False},
                ))
                result.findings.append("最新数据库记录与前一条可用记录只表示数值差异；不证明参数变化造成差异。")
        result.uncertainty.append("数据库最新记录不一定是当前 Aspen 实时运行结果。")
        front = self._pareto_front(usable, context.objectives)
        if front:
            refs = [str(row.get("case_id")) for row in front]
            result.evidence.append(_ev(
                "optimization.pareto.front", EvidenceSource.OPTIMIZATION_RESULTS,
                f"按 YAML 目标方向对 {len(usable)} 条成功且目标可用历史工况进行本地非支配筛选，得到 {len(front)} 个样本的观察前沿",
                "SimulationDB history + YAML objective directions",
                {"case_ids": refs, "sample_count": len(usable), "front_size": len(front)},
                confidence=0.95,
                metadata={"method": "non_dominated_filter", "causal": False, "underlying_source": "simulation_db"},
            ))
            result.findings.append("Pareto 前沿是当前历史样本上的离散非支配集合，不保证代表全局前沿。")
        else:
            result.missing.append("没有足够的多目标成功样本可计算 Pareto 前沿。")
        if len(usable) < 3:
            result.uncertainty.append("可用历史样本少于 3，趋势和优先级判断不稳定。")
        infeasible = [row for row in rows if row.get("feasible") is False or str(row.get("status", "")).casefold() == "infeasible"]
        if infeasible:
            result.findings.append(f"历史记录中有 {len(infeasible)} 条标记为不可行；建议查看相应约束残差。")
        result.next_steps.append("下一轮实验优先选择有清晰 YAML 边界、且历史覆盖不足的设计变量；逐次记录目标读取和约束余量。")
        return result

    @staticmethod
    def _pareto_front(rows: list[dict[str, Any]], objectives: list[dict[str, Any]]) -> list[dict[str, Any]]:
        specs = [(str(x.get("name") or ""), bool(x.get("minimize", True))) for x in objectives if x.get("name")]
        if len(specs) < 2:
            return []
        vectors: list[tuple[dict[str, Any], list[float]]] = []
        for row in rows:
            values = {str(x.get("name")): _numeric(x.get("value")) for x in _objectives(row) if x.get("available", True)}
            if all(values.get(name) is not None for name, _ in specs):
                vectors.append((row, [float(values[name]) * (1 if minimize else -1) for name, minimize in specs]))
        front = []
        for index, (row, vector) in enumerate(vectors):
            dominated = False
            for other_index, (_, other) in enumerate(vectors):
                if index == other_index:
                    continue
                if all(a <= b for a, b in zip(other, vector)) and any(a < b for a, b in zip(other, vector)):
                    dominated = True
                    break
            if not dominated:
                front.append(row)
        return front


class TechnicalAdvisorAgent:
    name = "TechnicalAdvisorAgent"

    def __init__(self, ima_client: Any = None):
        self.ima_client = ima_client

    def run(self, question: str, context: ProcessContext, task: Any = None, *, entities: dict[str, list[str]] | None = None) -> AgentResult:
        result = AgentResult(self.name)
        advice = TechnicalAdvisor(self.ima_client).advise(question, context)
        result.evidence.extend(advice.local_facts)
        result.evidence.extend(advice.simulation_evidence)
        if advice.ima_answer is not None:
            result.evidence.append(_ev(
                "ima.technical_answer", EvidenceSource.IMA, advice.ima_answer.answer or "IMA returned no answer",
                advice.ima_answer.request_id or advice.ima_answer.source, advice.ima_answer.answer,
                confidence=advice.ima_answer.confidence or 0.5,
                metadata={"assumptions": advice.ima_answer.assumptions},
            ))
            for index, citation in enumerate(advice.ima_answer.sources):
                result.evidence.append(_ev(
                    f"ima.technical_citation.{index}", EvidenceSource.IMA,
                    citation.snippet or citation.title or citation.source,
                    citation.source, citation.to_dict(), confidence=citation.confidence or 0.7,
                    metadata={"layer": "citation", **citation.to_dict()},
                ))
        result.findings.append(advice.advice.conclusion)
        result.uncertainty.extend(advice.advice.conflicts)
        result.next_steps.extend(advice.advice.suggested_checks)
        result.next_steps.extend(advice.advice.suggested_experiments)
        result.missing.extend(item.prompt for item in advice.intent.clarifications)
        return result


class LocalCaseAgents:
    """Concrete, allow-listed read-only agent registry."""

    def __init__(self, ima_client: Any = None):
        self.case_overview = CaseOverviewAgent()
        self.equipment = EquipmentAgent(ima_client)
        self.topology = TopologyAgent()
        self.parameter_impact = ParameterImpactAgent()
        self.simulation_diagnostic = SimulationDiagnosticAgent()
        self.optimization = OptimizationAgent()
        self.technical_advisor = TechnicalAdvisorAgent(ima_client)

    def registry(self) -> dict[str, Any]:
        agents = (
            self.case_overview, self.equipment, self.topology,
            self.parameter_impact, self.simulation_diagnostic,
            self.optimization, self.technical_advisor,
        )
        return {agent.name: agent.run for agent in agents}
