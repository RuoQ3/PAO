from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # PyYAML is optional in lightweight agent deployments.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only without dependency
    yaml = None
import json

from src.agents.contracts.models import EvidenceItem, EvidenceSource, ProcessContext
from src.database.node_db import NodeDB
from src.database.simulation_db import SimulationDB


class ProcessContextBuildError(ValueError):
    """Raised only for malformed required inputs; missing data is diagnostic."""


@dataclass
class _Selection:
    kind: str
    item: dict[str, Any]


class ProcessContextBuilder:
    """Build a read-only, provenance preserving process snapshot.

    ``topology`` is an optional persisted topology dictionary in the format
    returned by :func:`src.aspen_driver.topology.read_process_topology`.
    The builder never imports or opens COM/Aspen; NodeDB is used as a cached
    model snapshot.  ``session`` may be a mapping or an object with matching
    attributes (``session_id``, ``case_id``, ``user_input``).
    """

    def __init__(
        self,
        *,
        yaml_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        node_db: NodeDB | str | Path | None = None,
        simulation_db: SimulationDB | str | Path | None = None,
        topology: dict[str, Any] | None = None,
        model_id: str | None = None,
        catalog_id: str | None = None,
        aspen_file: str | Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.yaml_path = Path(yaml_path) if yaml_path else None
        self.config = dict(config or {})
        self.aspen_file = Path(aspen_file) if aspen_file else None
        self.read_only = read_only
        self.node_db = node_db if isinstance(node_db, NodeDB) else (NodeDB(node_db, read_only=read_only) if node_db else None)
        self.simulation_db = simulation_db if isinstance(simulation_db, SimulationDB) else (SimulationDB(simulation_db, read_only=read_only) if simulation_db else None)
        self.topology = topology or {}
        self.model_id = model_id
        self.catalog_id = catalog_id
        self._owns_node = not isinstance(node_db, NodeDB) and self.node_db is not None
        self._owns_sim = not isinstance(simulation_db, SimulationDB) and self.simulation_db is not None
        self.catalog_warning: str | None = None

    def close(self) -> None:
        if self._owns_node and self.node_db:
            self.node_db.close()
        if self._owns_sim and self.simulation_db:
            self.simulation_db.close()

    def __enter__(self) -> "ProcessContextBuilder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _load_config(self) -> dict[str, Any]:
        if self.config:
            return self.config
        if not self.yaml_path:
            return {}
        if not self.yaml_path.exists():
            raise ProcessContextBuildError(f"YAML 配置不存在: {self.yaml_path}")
        text = self.yaml_path.read_text(encoding="utf-8")
        if yaml is not None:
            raw = yaml.safe_load(text) or {}
        else:
            # JSON is a strict subset of YAML and is sufficient as a
            # dependency-free interchange format for generated configs.
            try:
                raw = json.loads(text) or {}
            except json.JSONDecodeError as exc:
                raise ProcessContextBuildError("读取 YAML 需要 PyYAML，或请提供 JSON 子集配置") from exc
        if not isinstance(raw, dict):
            raise ProcessContextBuildError("YAML 根节点必须是 mapping")
        return raw

    @staticmethod
    def _session_value(session: Any, name: str, default: Any = None) -> Any:
        if isinstance(session, dict):
            return session.get(name, default)
        return getattr(session, name, default)

    @staticmethod
    def _evidence(eid: str, source: EvidenceSource, claim: str, ref: str | None = None, value: Any = None, unit: str | None = None) -> EvidenceItem:
        return EvidenceItem(id=eid, source=source, claim=claim, source_ref=ref, value=value, unit=unit)

    def _configured_aspen_file(self, cfg: dict[str, Any] | None = None) -> Path | None:
        if self.aspen_file is not None:
            target = self.aspen_file
        else:
            simulator = (cfg or {}).get("simulator", {}) or {}
            raw = simulator.get("filepath") if isinstance(simulator, dict) else None
            if not raw:
                return None
            target = Path(str(raw)).expanduser()
            if not target.is_absolute():
                cwd_candidate = Path.cwd() / target
                yaml_candidate = (self.yaml_path.parent / target) if self.yaml_path else cwd_candidate
                target = cwd_candidate if cwd_candidate.exists() or not yaml_candidate.exists() else yaml_candidate
        try:
            return target.resolve(strict=False)
        except OSError:
            return target.absolute()

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        try:
            return str(Path(path).expanduser().resolve(strict=False)).replace("/", "\\").casefold()
        except OSError:
            return str(path).replace("/", "\\").casefold()

    def _catalog_entries(self, cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.node_db:
            return []
        catalog = self.catalog_id
        if not catalog:
            scans = self.node_db.list_catalog_scans()
            expected = self._configured_aspen_file(cfg)
            if expected is not None:
                normalized = self._normalize_path(expected)
                scan = next((item for item in scans if self._normalize_path(
                    str(item.get("aspen_file_path") or "")
                ) == normalized), None)
                if scan is None:
                    self.catalog_warning = f"NodeDB 没有与 Aspen 文件 {expected} 匹配的 catalog；未使用其他模型的节点数据。"
            else:
                scan = scans[0] if scans else None
            catalog = scan.get("catalog_id") if scan else None
        if not catalog:
            return []
        self.catalog_id = catalog
        return self.node_db.get_catalog_entries(catalog)

    @staticmethod
    def _infer_topology(entries: list[dict[str, Any]]) -> dict[str, Any]:
        blocks: dict[str, dict[str, Any]] = {}
        streams: set[str] = set()
        for e in entries:
            b = e.get("block_name") or ""
            if b:
                blocks.setdefault(b, {"id": b, "label": b, "block_type": e.get("block_type", "")})
            s = e.get("stream_name") or ""
            if s:
                streams.add(s)
        return {"nodes": list(blocks.values()), "edges": [], "streams": [{"name": s} for s in sorted(streams)]}

    def _model_parts(self, cfg: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[EvidenceItem]]:
        entries = self._catalog_entries(cfg)
        self._catalog_entries_cache = entries
        topo = self.topology or self._infer_topology(entries)
        nodes = list(topo.get("nodes") or [])
        edges = list(topo.get("edges") or [])
        equipment: list[dict[str, Any]] = []
        aliases = (cfg or {}).get("equipment_aliases", {}) or {}
        for n in nodes:
            name = str(n.get("id") or n.get("name") or n.get("label") or "")
            if not name:
                continue
            equipment.append({"name": name, "aspen_path": rf"\Data\Blocks\{name}", "block_type": n.get("block_type", ""), "category": n.get("category", "unknown"), "alias": aliases.get(name), "source": "aspen_model"})
        stream_names = {str(e.get("id") or e.get("label") or "") for e in edges if e.get("id") or e.get("label")}
        stream_names.update(str(s.get("name") or s.get("id") or "") for s in (topo.get("streams") or []) if s.get("name") or s.get("id"))
        stream_items = []
        for name in sorted(x for x in stream_names if x):
            edge = next((e for e in edges if str(e.get("id") or e.get("label")) == name), {})
            stream_items.append({"name": name, "aspen_path": rf"\Data\Streams\{name}", "alias": (cfg or {}).get("stream_aliases", {}).get(name), "source": "aspen_model", "upstream": edge.get("source"), "downstream": edge.get("target")})
        evidence = []
        if nodes or stream_items:
            evidence.append(self._evidence("model.topology", EvidenceSource.ASPEN_MODEL, f"读取到 {len(equipment)} 个设备和 {len(stream_items)} 条物流", self.catalog_id or "topology", {"equipment": len(equipment), "streams": len(stream_items)}))
        else:
            evidence.append(self._evidence("model.missing", EvidenceSource.ASPEN_MODEL, "未找到拓扑快照；模型可能未打开或尚未导出", self.catalog_id))
        return equipment, stream_items, {"nodes": nodes, "edges": edges}, evidence

    @staticmethod
    def _variable_target(item: dict[str, Any]) -> str:
        return str(item.get("aspen_path") or item.get("target_path") or item.get("name") or "")

    def _yaml_parts(self, cfg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[EvidenceItem]]:
        variables: dict[str, dict[str, Any]] = {}
        evidence: list[EvidenceItem] = []
        for raw in cfg.get("design_variables", []) or []:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("name") or raw.get("aspen_path") or raw.get("target_path") or "")
            if not key:
                continue
            d = dict(raw); d["source"] = "yaml"; d["path"] = self._variable_target(raw)
            d["related_equipment"] = self._paths_to_names(d["path"], "Blocks")
            d["related_streams"] = self._paths_to_names(d["path"], "Streams")
            variables[key] = d
            evidence.append(self._evidence(f"yaml.variable.{key}", EvidenceSource.YAML, f"配置设计变量 {key}", str(self.yaml_path) if self.yaml_path else "config", d))
        objectives = [dict(x, source="yaml") for x in (cfg.get("objectives", []) or []) if isinstance(x, dict)]
        raw_constraints = list(cfg.get("constraints", []) or [])
        raw_constraints.extend(dict(x, hard_boundary=True) for x in (cfg.get("hard_constraints", []) or []) if isinstance(x, dict))
        constraints = [dict(x, source="yaml") for x in raw_constraints if isinstance(x, dict)]
        for i, obj in enumerate(objectives):
            evidence.append(self._evidence(f"yaml.objective.{i}", EvidenceSource.YAML, f"配置目标 {obj.get('name', i)}", str(self.yaml_path) if self.yaml_path else "config", obj))
        for i, con in enumerate(constraints):
            con["related_streams"] = sorted(set(self._paths_to_names(str(con.get("aspen_path") or con.get("path") or con.get("variable") or con.get("stream") or ""), "Streams") + ([str(con["stream"])] if con.get("stream") else [])))
            evidence.append(self._evidence(f"yaml.constraint.{i}", EvidenceSource.YAML, f"配置约束 {con.get('name', i)}", str(self.yaml_path) if self.yaml_path else "config", con))
        evidence.append(self._evidence(
            "yaml.summary", EvidenceSource.YAML,
            f"YAML 配置记录 {len(variables)} 个设计变量、{len(objectives)} 个目标和 {len(constraints)} 个约束",
            str(self.yaml_path) if self.yaml_path else "config",
            {"design_variables": list(variables), "objectives": [x.get("name") for x in objectives], "constraints": [x.get("name") for x in constraints]},
        ))
        return variables, objectives, constraints, evidence

    @staticmethod
    def _equipment_parameter_parts(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Keep sampled Aspen Input leaves separate from YAML tunable variables."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            block = str(entry.get("block_name") or "")
            path = str(entry.get("abs_path") or "")
            if not block or not path.casefold().startswith(f"\\data\\blocks\\{block}\\input\\".casefold()):
                continue
            if not entry.get("is_leaf") or entry.get("sample_value") is None or entry.get("sample_error"):
                continue
            grouped.setdefault(block, []).append({
                "name": str(entry.get("name") or path.rsplit("\\", 1)[-1]),
                "aspen_path": path,
                "value": entry.get("sample_value"),
                "unit": str(entry.get("unit_string") or ""),
                "captured_at": entry.get("cached_at"),
                "catalog_id": entry.get("catalog_id"),
                "source": "nodedb",
                "value_stage": "catalog_snapshot",
            })
        for items in grouped.values():
            items.sort(key=lambda x: str(x.get("name", "")).casefold())
        return grouped

    @staticmethod
    def _paths_to_names(path: str, kind: str) -> list[str]:
        m = re.search(rf"\\Data\\{kind}\\([^\\]+)", path, flags=re.I)
        return [m.group(1)] if m else []

    def build(self, *, session: Any = None, model_id: str | None = None) -> ProcessContext:
        cfg = self._load_config()
        equipment, streams, topology, sources = self._model_parts(cfg)
        equipment_parameters = self._equipment_parameter_parts(self._catalog_entries_cache)
        variables, objectives, constraints, yaml_evidence = self._yaml_parts(cfg)
        sources.extend(yaml_evidence)
        session_id = self._session_value(session, "session_id")
        case_id = self._session_value(session, "case_id")
        user_input = self._session_value(session, "user_input")
        if user_input:
            sources.append(self._evidence("user.input", EvidenceSource.USER_INPUT, "用户输入已纳入上下文", str(session_id or "session"), user_input))
        current_values: dict[str, Any] = {}
        diagnostics: list[str] = []
        if self.catalog_warning:
            diagnostics.append(self.catalog_warning)
        if self.node_db and case_id:
            for row in self.node_db.get_node_values(str(case_id), include_errors=True):
                current_values[row["path"]] = {"value": row.get("value"), "unit": row.get("unit", ""), "error": row.get("error"), "source": "aspen_model"}
        elif self.node_db and not case_id:
            diagnostics.append("未提供 case_id，NodeDB 当前值未绑定到具体工况")
        history: list[EvidenceItem] = []
        if self.simulation_db:
            rows = self.simulation_db.query_cases(session_id=session_id, limit=100, newest_first=True) if session_id else self.simulation_db.query_cases(limit=100, newest_first=True)
            if not rows:
                diagnostics.append("SimulationDB 历史数据为空")
            for i, row in enumerate(rows):
                history.append(self._evidence(f"simulation.case.{i}", EvidenceSource.SIMULATION_DB, f"历史工况 {row.get('case_id')} 状态 {row.get('status')}", row.get("case_id"), row))
        if not equipment and not streams:
            diagnostics.append("未找到设备/物流节点；请确认 NodeDB catalog 或只读拓扑快照可用")
        for kind, items in (("设备", equipment), ("物流", streams)):
            counts: dict[str, int] = {}
            for item in items:
                name = str(item.get("name", ""))
                counts[name] = counts.get(name, 0) + 1
            duplicates = sorted(name for name, count in counts.items() if name and count > 1)
            if duplicates:
                diagnostics.append(f"{kind}名称重复: {', '.join(duplicates)}；请使用 Aspen 路径选择")
        if (equipment or streams) and not topology.get("edges"):
            diagnostics.append("拓扑未包含 stream 边，无法可靠追踪上下游；当前仅提供节点清单")
        for key, var in variables.items():
            if not var.get("unit") and var.get("type") != "derived":
                diagnostics.append(f"变量 {key} 缺少单位")
        status = "ok" if not diagnostics else "partial"
        return ProcessContext(model_id=model_id or self.model_id or str(cfg.get("model_id") or (self.yaml_path.stem if self.yaml_path else "unknown")), model_version=cfg.get("model_version"), equipment=equipment, streams=streams, topology=topology, variables=variables, objectives=objectives, constraints=constraints, current_values=current_values, equipment_parameters=equipment_parameters, history_evidence=history, sources=sources, session_id=session_id, generated_at=datetime.now(timezone.utc).isoformat(), status=status, diagnostics=diagnostics)

    @staticmethod
    def _match(value: str, query: str) -> bool:
        return value.lower() == query.lower() or query.lower() in value.lower()

    def select_equipment(self, context: ProcessContext, selector: str) -> list[dict[str, Any]]:
        return [x for x in context.equipment if self._match(str(x.get("name", "")), selector) or self._match(str(x.get("aspen_path", "")), selector) or self._match(str(x.get("alias", "")), selector)]

    find_equipment = select_equipment

    def select_streams(self, context: ProcessContext, selector: str) -> list[dict[str, Any]]:
        return [x for x in context.streams if self._match(str(x.get("name", "")), selector) or self._match(str(x.get("aspen_path", "")), selector) or self._match(str(x.get("alias", "")), selector)]

    find_streams = select_streams

    def related_streams(self, context: ProcessContext, equipment: str, *, recursive: bool = True) -> list[dict[str, Any]]:
        selected = {str(x.get("name")) for x in self.select_equipment(context, equipment)}
        edges = context.topology.get("edges", [])
        seen = set(selected); changed = True
        while recursive and changed:
            changed = False
            for e in edges:
                if e.get("source") in seen or e.get("target") in seen:
                    for n in (e.get("source"), e.get("target")):
                        if n and n not in seen:
                            seen.add(n); changed = True
        return [s for s in context.streams if s.get("upstream") in seen or s.get("downstream") in seen]

    def related_equipment(self, context: ProcessContext, stream: str) -> list[dict[str, Any]]:
        selected = self.select_streams(context, stream)
        names = {n for s in selected for n in (s.get("upstream"), s.get("downstream")) if n}
        return [e for e in context.equipment if e.get("name") in names]

    def related_variables(self, context: ProcessContext, *, equipment: str | None = None, stream: str | None = None) -> list[dict[str, Any]]:
        names = set()
        if equipment:
            names.update(str(x.get("name")) for x in self.select_equipment(context, equipment))
        if stream:
            names.add(str(stream))
        result = []
        for var in context.variables.values():
            if names.intersection(set(var.get("related_equipment", [])) | set(var.get("related_streams", []))):
                result.append(var)
        return result

    def subgraph(self, context: ProcessContext, *, equipment: str | None = None, stream: str | None = None) -> ProcessContext:
        eq = self.select_equipment(context, equipment) if equipment else context.equipment
        st = self.related_streams(context, equipment) if equipment else (self.select_streams(context, stream) if stream else context.streams)
        names = {x.get("name") for x in eq}
        names.update(x.get("name") for x in st)
        edges = [e for e in context.topology.get("edges", []) if e.get("id") in names or e.get("source") in names or e.get("target") in names]
        vars_ = {k: v for k, v in context.variables.items() if names.intersection(set(v.get("related_equipment", [])) | set(v.get("related_streams", [])))}
        return ProcessContext(model_id=context.model_id, model_version=context.model_version, equipment=eq, streams=st, topology={"nodes": [n for n in context.topology.get("nodes", []) if n.get("id") in {e.get("name") for e in eq}], "edges": edges}, variables=vars_, objectives=context.objectives, constraints=[c for c in context.constraints if names.intersection(set(c.get("related_streams", [])))], current_values={k: v for k, v in context.current_values.items() if any(k.startswith((e.get("aspen_path") or "")) for e in eq + st)}, equipment_parameters={k: v for k, v in context.equipment_parameters.items() if k in {e.get("name") for e in eq}}, history_evidence=context.history_evidence, sources=context.sources, session_id=context.session_id, generated_at=context.generated_at, status=context.status, diagnostics=context.diagnostics)
