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

from src.agents.evidence.models import EvidenceItem, EvidenceSource, ProcessContext
from src.database.node_db import NodeDB
from src.database.simulation_db import SimulationDB
from .resources import resolve_model_file, session_model_file


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
        self._explicit_catalog_id = catalog_id
        self._owns_node = not isinstance(node_db, NodeDB) and self.node_db is not None
        self._owns_sim = not isinstance(simulation_db, SimulationDB) and self.simulation_db is not None
        self.catalog_warning: str | None = None
        self._catalog_entries_cache: list[dict[str, Any]] = []

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
        return self.resolve_model_resource(config=cfg).candidate_path

    def resolve_model_resource(self, *, session=None, config=None):
        override = session_model_file(session)
        cfg = self._load_config() if config is None else config
        simulator = cfg.get("simulator", {}) or {}
        raw = override or self.aspen_file or (simulator.get("filepath") if isinstance(simulator, dict) else None)
        source = "session.model_file" if override else ("aspen_file" if self.aspen_file else "simulator.filepath")
        return resolve_model_file(raw, yaml_path=self.yaml_path, source=source,
                                  source_ref=str(self.yaml_path) if self.yaml_path else None)

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        try:
            return str(Path(path).expanduser().resolve(strict=False)).replace("/", "\\").casefold()
        except OSError:
            return str(path).replace("/", "\\").casefold()

    def _catalog_entries(self, cfg: dict[str, Any] | None = None, resource=None) -> list[dict[str, Any]]:
        if not self.node_db:
            return []
        catalog = self._explicit_catalog_id
        if not catalog:
            scans = self.node_db.list_catalog_scans()
            expected = resource.candidate_path if resource else self._configured_aspen_file(cfg)
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
            self.catalog_id = None
            return []
        scan = self.node_db.get_catalog_scan(catalog)
        expected = resource.path if resource else self._configured_aspen_file(cfg)
        if expected and scan:
            import hashlib
            stamp = str(scan.get("aspen_file_hash") or "")
            if len(stamp) in {32, 64}:
                actual = hashlib.md5(Path(expected).read_bytes()).hexdigest() if len(stamp) == 32 else hashlib.sha256(Path(expected).read_bytes()).hexdigest()
                if actual != stamp:
                    self.catalog_id = None
                    self.catalog_warning = f"NodeDB catalog {catalog} 与当前模型摘要冲突，未加载过期节点"
                    return []
            else:
                self.catalog_warning = f"NodeDB catalog {catalog} 未提供可验证的模型版本，仅为历史静态快照"
        if scan and resource and resource.candidate_path and self._normalize_path(scan["aspen_file_path"]) != self._normalize_path(resource.candidate_path):
            self.catalog_warning = "显式 catalog 指向其他模型，未载入"
            self.catalog_id = None
            return []
        self.catalog_id = catalog
        return self.node_db.get_catalog_entries(catalog)

    def _config_node_db(self, cfg: dict[str, Any]) -> None:
        """Attach the catalog database declared by the YAML configuration.

        The conversation entry point intentionally does not know PAO's
        extraction settings.  Resolving this here keeps ``--conversation``
        useful with an existing YAML file while preserving the builder's
        read-only contract.
        """
        if self.node_db is not None:
            return
        extraction = cfg.get("extraction", {}) or {}
        raw = extraction.get("catalog_db") if isinstance(extraction, dict) else None
        if not raw:
            return
        target = Path(str(raw)).expanduser()
        if not target.is_absolute():
            candidates = [Path.cwd() / target]
            if self.yaml_path:
                candidates.insert(0, self.yaml_path.parent / target)
            # PAO configs commonly store paths relative to the repository
            # root (for example ``cases/demo_case/output/node.db``), while
            # small ad-hoc configs use paths relative to the YAML directory.
            # Prefer an existing candidate; otherwise retain the repository
            # relative spelling for a useful diagnostic.
            target = next((item for item in candidates if item.exists()), Path.cwd() / Path(str(raw)))
        if not target.exists():
            self.catalog_warning = f"NodeDB 文件不存在: {target}"
            return
        self.node_db = NodeDB(target, read_only=self.read_only)
        self._owns_node = True

    def _bkp_model_parts(self, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[EvidenceItem]]:
        """Read the portable block/connection index embedded in a .bkp file.

        Aspen's text backup format contains a small flowsheet index before
        the binary-like data sections.  Reading that index gives the Agent a
        useful, provenance-labelled inventory when Aspen COM or a catalog DB
        is unavailable; values are never treated as live simulation results.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeError):
            return [], [], {"nodes": [], "edges": []}, []
        blocks: dict[str, dict[str, Any]] = {}
        # The first flowsheet index lists: name, display label, Built-In, type.
        pat = re.compile(r"(?m)^>VERSION[^\r\n]*\r?\n([^\r\n]+)\r?\n([^\r\n]+)\r?\nBuilt-In\r?\n([A-Za-z][A-Za-z0-9_-]*)")
        for match in pat.finditer(text):
            name = match.group(1).strip().strip('"')
            if name and name not in blocks:
                blocks[name] = {"id": name, "label": name, "block_type": match.group(3).upper()}
        edges: list[dict[str, Any]] = []
        stream_links: dict[str, dict[str, str]] = {}
        block_pat = re.compile(r"BLOCK\s+BLKID\s*=\s*\"?([^\"\s]+)\"?\s+BLKTYPE\s*=\s*\"?([^\"\s]+)\"?(.*?)(?=\\\s*\\\s*BLOCK\b|$)", re.I | re.S)
        for match in block_pat.finditer(text):
            name, typ, body = match.group(1), match.group(2), match.group(3)
            blocks.setdefault(name, {"id": name, "label": name, "block_type": typ.upper()})
            blocks[name]["block_type"] = typ.upper()
            for direction in ("IN", "OUT"):
                section = re.search(rf"\b{direction}\s*=\s*\((.*?)\)", body, re.I | re.S)
                if not section:
                    continue
                # Streams are followed by Aspen substream identifiers (M0-1,
                # M1-2, ...).  This avoids mistaking those identifiers for
                # stream names.
                names = re.findall(r"([A-Za-z0-9][A-Za-z0-9_-]*)\s+M[0-9-]+", section.group(1))
                for stream in names:
                    stream_links.setdefault(stream, {})["source" if direction == "OUT" else "target"] = name
        # Aspen stores stream composition vectors in STRM_XY.  Keep the
        # component names and nonzero mole fractions in the read-only snapshot
        # so a follow-up such as "流股0318包含哪些物料" can be answered from
        # the .bkp without opening Aspen COM.  This is a snapshot value, not a
        # newly simulated result.
        component_names: list[str] = []
        comp_match = re.search(r"IDSET\s+ID_\w+\s+\(COMPONENTS\)\s*\((.*?)\)", text, re.I | re.S)
        if comp_match:
            component_names = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", comp_match.group(1))
        stream_compositions: dict[str, list[dict[str, Any]]] = {}
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?"
        for match in re.finditer(r"DSET\s+STREAM\s+MATERIAL\s+([A-Za-z0-9_-]+)\s+STRM_XY\s+@\w+\s*\((.*?)\)", text, re.I | re.S):
            stream = match.group(1)
            values = [float(x.replace("D", "E").replace("d", "e"))
                      for x in re.findall(number, match.group(2))]
            if not component_names or len(values) < len(component_names):
                continue
            # STRM_XY begins with the mole-fraction vector.  Do not infer a
            # mass basis from the following Aspen payload because its exact
            # layout depends on the selected property scope.
            n = len(component_names)
            mole = values[:n]
            stream_compositions[stream] = [
                {"name": component, "mole_fraction": mole[i]}
                for i, component in enumerate(component_names)
                if abs(mole[i]) > 1e-12
            ]
        streams = [{"name": stream, "aspen_path": rf"\Data\Streams\{stream}", "source": "bkp_index",
                    "upstream": link.get("source"), "downstream": link.get("target"),
                    **({"components": stream_compositions[stream],
                        "composition_basis": "mole_fraction (STRM_XY snapshot)"}
                       if stream in stream_compositions else {})}
                   for stream, link in sorted(stream_links.items())]
        for stream, link in sorted(stream_links.items()):
            if link.get("source") or link.get("target"):
                edges.append({"id": stream, "source": link.get("source"), "target": link.get("target")})
        equipment = [{"name": name, "aspen_path": rf"\Data\Blocks\{name}",
                      "block_type": item.get("block_type", ""), "category": "unknown",
                      "alias": None, "source": "bkp_index"}
                     for name, item in blocks.items()]
        evidence = [self._evidence("bkp.index", EvidenceSource.ASPEN_MODEL,
                                    f"从 .bkp 文本索引读取到 {len(equipment)} 个设备和 {len(streams)} 条物流",
                                    str(path), {"equipment": len(equipment), "streams": len(streams)})]
        return equipment, streams, {"nodes": list(blocks.values()), "edges": edges}, evidence

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

    def _model_parts(self, cfg: dict[str, Any] | None = None, resource=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[EvidenceItem]]:
        cfg = cfg or {}
        self._config_node_db(cfg)
        entries = self._catalog_entries(cfg, resource)
        self._catalog_entries_cache = entries
        topo = self.topology or self._infer_topology(entries)
        nodes = list(topo.get("nodes") or [])
        edges = list(topo.get("edges") or [])
        equipment: list[dict[str, Any]] = []
        aliases = cfg.get("equipment_aliases", {}) or {}
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
        bkp_path = resource.path if resource else self.resolve_model_resource(config=cfg).path
        if bkp_path and bkp_path.exists() and bkp_path.suffix.casefold() == ".bkp":
            # BKP is the primary portable topology snapshot.  Keep it even
            # when a NodeDB catalog is also present so normalization can
            # retain both observations and report disagreements explicitly.
            bkp_equipment, bkp_streams, bkp_topology, bkp_evidence = self._bkp_model_parts(bkp_path)
            if bkp_equipment or bkp_streams:
                equipment = bkp_equipment + equipment
                stream_items = bkp_streams + stream_items
                nodes = list(bkp_topology.get("nodes", [])) + nodes
                edges = list(bkp_topology.get("edges", [])) + edges
                evidence = bkp_evidence + evidence
        return equipment, stream_items, {"nodes": nodes, "edges": edges}, evidence

    def _simulation_model_parts(self, session: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[EvidenceItem]]:
        """Read the latest persisted flowsheet snapshot, when available.

        SimulationDB is an evidence source, not a live Aspen interface.  Its
        blocks/streams JSON is accepted in both mapping and list forms and is
        merged later by the same conflict-preserving normalizer.
        """
        if self.simulation_db is None:
            return [], [], {"nodes": [], "edges": []}, []
        session_id = self._session_value(session, "session_id")
        case_id = self._session_value(session, "case_id")
        row = self.simulation_db.get_case(str(case_id)) if case_id else None
        if row is not None and session_id and row.get("session_id") != session_id:
            return [], [], {"nodes": [], "edges": []}, [self._evidence(
                "simulation.session_conflict", EvidenceSource.SYSTEM,
                "请求的历史工况不属于当前会话，未载入", str(case_id))]
        if row is None:
            summaries = self.simulation_db.query_cases(session_id=session_id, limit=1, newest_first=True) if session_id else self.simulation_db.query_cases(limit=1, newest_first=True)
            if summaries:
                row = self.simulation_db.get_case(str(summaries[0].get("case_id")))
        if not row:
            return [], [], {"nodes": [], "edges": []}, []
        def records(raw: Any, *, key_name: str) -> list[dict[str, Any]]:
            if isinstance(raw, dict):
                result = []
                for key, value in raw.items():
                    result.append(dict(value, id=str(value.get("id") or key)) if isinstance(value, dict) else {"id": str(key), key_name: value})
                return result
            return [dict(x) for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
        equipment = records(row.get("blocks"), key_name="block")
        for item in equipment:
            item.setdefault("name", item.get("id"))
            item.setdefault("source", "simulation_db")
        streams = records(row.get("streams"), key_name="stream")
        for item in streams:
            item.setdefault("name", item.get("id"))
            item.setdefault("source", "simulation_db")
        evidence = [self._evidence("simulation.model", EvidenceSource.SIMULATION_DB,
                                    "SimulationDB flowsheet snapshot loaded", str(row.get("case_id")),
                                    {"equipment": len(equipment), "streams": len(streams)})]
        return equipment, streams, {"nodes": equipment, "streams": streams, "edges": streams}, evidence

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
                "authorized": bool(entry.get("is_enterable", True)),
                "tunable": bool(entry.get("is_enterable", True)),
                "confidence": 1.0,
                "type": "float",
            })
        for items in grouped.values():
            items.sort(key=lambda x: str(x.get("name", "")).casefold())
        return grouped

    @staticmethod
    def _paths_to_names(path: str, kind: str) -> list[str]:
        m = re.search(rf"\\Data\\{kind}\\([^\\]+)", path, flags=re.I)
        return [m.group(1)] if m else []

    _ROLE_MAP = {
        "PUMP": "pump",
        "MIXER": "mixer",
        "RCSTR": "continuous_reactor",
        "RSTOIC": "stoichiometric_reactor",
        "FLASH2": "vapor_liquid_separator",
        "RADFRAC": "fractionation_column",
        "HEATER": "heater",
        "COOLER": "cooler",
        "HEATX": "heat_exchanger",
        "SEP": "separator",
    }
    _ROLE_DESCRIPTIONS = {
        "PUMP": "transport_or_pressurization",
        "MIXER": "mixing",
        "RCSTR": "continuous_reaction",
        "RSTOIC": "stoichiometric_reaction",
        "FLASH2": "vapor_liquid_equilibrium_separation",
        "RADFRAC": "fractionation_or_stage_separation",
        "HEATER": "heating",
        "COOLER": "cooling",
        "HEATX": "heat_exchange",
        "SEP": "separation",
    }

    @classmethod
    def _infer_role(cls, block_type: Any) -> tuple[str, float]:
        kind = str(block_type or "").upper().strip()
        if kind in cls._ROLE_MAP:
            return cls._ROLE_MAP[kind], 0.95
        return "unknown", 0.0

    @staticmethod
    def _number(value: Any) -> float | int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip().replace(",", "")
        if not text or text.casefold() in {"nan", "na", "n/a", "none", "null", "*"}:
            return None
        try:
            converted = float(text.replace("D", "E").replace("d", "e"))
            return int(converted) if converted.is_integer() else converted
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_composition(cls, raw: Any, *, evidence_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        diagnostics: list[str] = []
        if raw is None:
            return [], diagnostics
        if isinstance(raw, dict):
            # Accept both ``{component: fraction}`` and a vector payload such
            # as ``{components: [...], values: [...]}`` without guessing a
            # basis when the vector lengths do not agree.
            if isinstance(raw.get("components"), (list, tuple)) and isinstance(raw.get("values"), (list, tuple)):
                names, values = list(raw["components"]), list(raw["values"])
                if len(names) != len(values):
                    diagnostics.append(f"composition {evidence_id} component/value count mismatch ({len(names)} != {len(values)})")
                raw = [{"name": name, "mole_fraction": values[i] if i < len(values) else None}
                       for i, name in enumerate(names)]
            else:
                raw = [{"name": name, "mole_fraction": value} for name, value in raw.items()]
        if not isinstance(raw, (list, tuple)):
            diagnostics.append(f"composition {evidence_id} is not a list or mapping")
            return [], diagnostics
        result: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if isinstance(item, str):
                name, value = item, None
            elif isinstance(item, dict):
                name = item.get("name") or item.get("component") or item.get("id")
                value = item.get("mole_fraction", item.get("mole", item.get("mass_fraction", item.get("value"))))
            else:
                diagnostics.append(f"composition {evidence_id}[{index}] is not an object")
                continue
            if not name:
                diagnostics.append(f"composition {evidence_id}[{index}] has no component name")
                continue
            parsed = cls._number(value)
            entry = dict(item) if isinstance(item, dict) else {"name": name}
            entry["name"] = str(name)
            if parsed is not None:
                if "mole_fraction" in entry or "mole" in entry:
                    entry["mole_fraction"] = parsed
            result.append(entry)
        return result, diagnostics

    @classmethod
    def _normalize_graph(cls, equipment: list[dict[str, Any]], streams: list[dict[str, Any]], topology: dict[str, Any], cfg: dict[str, Any], sources: list[EvidenceItem]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """Normalize all supported snapshots into the stable graph contract."""
        diagnostics: list[str] = []
        conflicts: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        inferences: list[dict[str, Any]] = []
        evidence_ids = [item.id for item in sources]

        # In-memory configurations may use any of these neutral names.  The
        # aliases are deliberately structural and do not encode case meaning.
        raw_nodes: list[dict[str, Any]] = list(equipment)
        for key in ("equipment_nodes", "equipment", "blocks"):
            value = cfg.get(key)
            if isinstance(value, list):
                raw_nodes.extend(x for x in value if isinstance(x, dict))
        raw_nodes.extend(x for x in topology.get("nodes", []) if isinstance(x, dict))
        # ``_model_parts`` exposes both its compatibility lists and the raw
        # topology.  Identical records are one observation, while records with
        # differing values are retained for conflict reporting.
        unique_nodes: list[dict[str, Any]] = []
        for candidate in raw_nodes:
            if any(candidate == previous for previous in unique_nodes):
                continue
            unique_nodes.append(candidate)
        raw_nodes = unique_nodes
        node_map: dict[str, dict[str, Any]] = {}
        for raw in raw_nodes:
            name = str(raw.get("id") or raw.get("name") or raw.get("label") or "").strip()
            if not name:
                diagnostics.append("equipment entry is missing id/name")
                continue
            key = str(raw.get("id") or name).casefold()
            item = dict(raw)
            item["id"] = name
            item["name"] = str(item.get("name") or item.get("label") or name)
            item.setdefault("kind", "equipment")
            item.setdefault("block_type", str(item.get("type") or "").upper())
            item["block_type"] = str(item.get("block_type") or "").upper()
            item.setdefault("aspen_path", rf"\\Data\\Blocks\\{name}")
            role, confidence = cls._infer_role(item.get("block_type"))
            item.setdefault("role", role)
            item.setdefault("role_description", cls._ROLE_DESCRIPTIONS.get(item.get("block_type"), "unknown"))
            item.setdefault("confidence", confidence)
            item.setdefault("provenance_source", item.get("source") or "config")
            item.setdefault("source", item.get("provenance_source") or "config")
            item.setdefault("source_ref", item.get("source"))
            item.setdefault("upstream_streams", [])
            item.setdefault("downstream_streams", [])
            item.setdefault("parameters", [])
            item.setdefault("inference_method", "block_type_mapping")
            item.setdefault("evidence_ids", evidence_ids[:1])
            if key in node_map:
                existing = node_map[key]
                conflict_count = len(conflicts)
                for field in ("block_type", "aspen_path", "role"):
                    old, new = existing.get(field), item.get(field)
                    old_cmp = str(old).replace("/", "\\").replace("\\\\", "\\") if field == "aspen_path" else old
                    new_cmp = str(new).replace("/", "\\").replace("\\\\", "\\") if field == "aspen_path" else new
                    if old_cmp not in (None, "", "unknown") and new_cmp not in (None, "", "unknown") and old_cmp != new_cmp:
                        conflicts.append({"kind": "equipment", "id": name, "field": field, "values": [{"value": old, "source": existing.get("provenance_source")}, {"value": new, "source": item.get("provenance_source")}]})
                for field, value in item.items():
                    if existing.get(field) in (None, "", [], {}):
                        existing[field] = value
                if len(conflicts) > conflict_count:
                    diagnostics.append(f"duplicate equipment merged with conflicting values: {name}")
            else:
                node_map[key] = item
            node = node_map[key]
            inferences.append({"kind": "inferred_role", "target_id": name, "role": role,
                               "inference_method": "block_type_mapping", "confidence": confidence,
                               "evidence_ids": list(node.get("evidence_ids") or evidence_ids[:1]),
                               "note": ("block_type is not mapped; role remains unknown"
                                        if role == "unknown" else None)})
            facts.append({"kind": "equipment_observed", "target_id": name, "evidence_ids": list(node.get("evidence_ids") or evidence_ids[:1])})

        raw_streams: list[dict[str, Any]] = list(streams)
        for key in ("streams", "edges"):
            value = cfg.get(key)
            if isinstance(value, list):
                raw_streams.extend(x for x in value if isinstance(x, dict))
        raw_streams.extend(x for x in topology.get("streams", []) if isinstance(x, dict))
        raw_streams.extend(x for x in topology.get("edges", []) if isinstance(x, dict))
        unique_streams: list[dict[str, Any]] = []
        for candidate in raw_streams:
            if any(candidate == previous for previous in unique_streams):
                continue
            unique_streams.append(candidate)
        raw_streams = unique_streams
        stream_map: dict[str, dict[str, Any]] = {}
        compositions: dict[str, list[dict[str, Any]]] = {}
        for raw in raw_streams:
            sid = str(raw.get("id") or raw.get("name") or raw.get("label") or "").strip()
            if not sid:
                diagnostics.append("stream entry is missing id/name")
                continue
            item = dict(raw)
            item["id"] = sid
            item["name"] = str(item.get("name") or item.get("label") or sid)
            # Legacy fields carry endpoints.  Canonical endpoint fields are
            # source_node/target; source remains the historical provenance
            # value when supplied by existing Aspen readers.
            raw_provenance = item.get("provenance_source")
            raw_source = item.get("source")
            item.setdefault("endpoint_fields_present", {
                "source": any(k in raw for k in ("source_node", "upstream", "source_equipment")) or
                          ("source" in raw and (raw_source is None or str(raw_source).casefold() in node_map)),
                "target": any(k in raw for k in ("target_node", "target", "downstream", "target_equipment")),
            })
            endpoint_values = {item.get("upstream"), item.get("source_equipment"), item.get("source_node")}
            if raw_provenance is None and raw_source and raw_source not in endpoint_values and str(raw_source).casefold() not in node_map:
                raw_provenance = raw_source
            source_node = item.get("source_node") or item.get("upstream") or item.get("source_equipment")
            target_node = item.get("target_node") or item.get("target") or item.get("downstream") or item.get("target_equipment")
            if source_node is None and item.get("source") and str(item.get("source")).casefold() in node_map:
                source_node = item.get("source")
                item["provenance_source"] = item.get("provenance_source") or "topology"
            item["source_node"] = source_node
            item["target"] = target_node
            # ``source`` is the canonical upstream endpoint.  Existing
            # provenance labels remain available under ``provenance_source``
            # and ``source_ref`` instead of being confused with topology.
            item["provenance_source"] = raw_provenance or item.get("provenance_source") or "config"
            item["source_ref"] = item.get("source_ref") or item["provenance_source"]
            item["source"] = source_node
            item["upstream"] = source_node
            item["downstream"] = target_node
            item.setdefault("aspen_path", rf"\\Data\\Streams\\{sid}")
            # Explicit stream semantics are authoritative observations.  For
            # an otherwise untyped edge we only infer generic boundary
            # semantics from topology; no case-specific names are used.
            explicit_stream_type = item.get("stream_type") or item.get("kind")
            if explicit_stream_type:
                item["stream_type"] = str(explicit_stream_type)
            elif item.get("is_energy") or str(item.get("phase") or "").casefold() == "energy":
                item["stream_type"] = "energy"
            elif source_node is None and target_node is not None:
                item["stream_type"] = "feed"
            elif source_node is not None and target_node is None:
                item["stream_type"] = "vent" if item.get("is_vent") else "product"
            elif source_node is None and target_node is None:
                item["stream_type"] = "unresolved_boundary"
            else:
                item["stream_type"] = "material"
            item.setdefault("temperature", None)
            item.setdefault("pressure", None)
            item.setdefault("flow", None)
            item.setdefault("phase", None)
            item.setdefault("confidence", 0.9 if source_node or target_node else 0.4)
            item.setdefault("evidence_ids", evidence_ids[:1])
            for field in ("temperature", "pressure", "flow"):
                if field in item and item[field] is not None:
                    parsed = cls._number(item[field])
                    if parsed is None:
                        diagnostics.append(f"stream {sid} has unparseable {field}: {item[field]!r}")
                    else:
                        item[field] = parsed
            composition_raw = item.get("composition", item.get("components"))
            composition, comp_diag = cls._normalize_composition(composition_raw, evidence_id=sid)
            diagnostics.extend(comp_diag)
            if composition:
                item["composition"] = composition
                item["components"] = composition
                compositions[sid] = composition
            key = sid.casefold()
            if key in stream_map:
                existing = stream_map[key]
                conflict_count = len(conflicts)
                for field in ("source_node", "target", "temperature", "pressure", "flow", "composition"):
                    old, new = existing.get(field), item.get(field)
                    if old not in (None, "", [], {}) and new not in (None, "", [], {}) and old != new:
                        conflicts.append({"kind": "stream", "id": sid, "field": field, "values": [{"value": old, "source": existing.get("provenance_source")}, {"value": new, "source": item.get("provenance_source")}]})
                if len(conflicts) > conflict_count:
                    diagnostics.append(f"duplicate stream merged with conflicting values: {sid}")
                for field, value in item.items():
                    if existing.get(field) in (None, "", [], {}):
                        existing[field] = value
            else:
                stream_map[key] = item
            facts.append({"kind": "stream_observed", "target_id": sid, "source_node": source_node, "target_node": target_node, "evidence_ids": list(item.get("evidence_ids") or evidence_ids[:1])})

        normalized_nodes = list(node_map.values())
        normalized_streams = list(stream_map.values())
        node_names = {str(n["id"]).casefold() for n in normalized_nodes}
        edges = []
        for stream in normalized_streams:
            source_node, target_node = stream.get("source_node"), stream.get("target")
            if source_node and str(source_node).casefold() not in node_names:
                diagnostics.append(f"stream {stream['id']} references unknown source {source_node}")
            if target_node and str(target_node).casefold() not in node_names:
                diagnostics.append(f"stream {stream['id']} references unknown target {target_node}")
            if not source_node and not target_node:
                diagnostics.append(f"stream {stream['id']} has no known upstream or downstream equipment")
            elif not source_node or not target_node:
                diagnostics.append(f"stream {stream['id']} is an open boundary stream")
            if not stream.get("composition"):
                diagnostics.append(f"stream {stream['id']} has no composition observation")
            if source_node or target_node:
                edges.append({"id": stream["id"], "source": source_node, "target": target_node, "stream_id": stream["id"]})
        for node in normalized_nodes:
            name = str(node["id"])
            node["upstream_streams"] = [s["id"] for s in normalized_streams if str(s.get("target") or "").casefold() == name.casefold()]
            node["downstream_streams"] = [s["id"] for s in normalized_streams if str(s.get("source_node") or "").casefold() == name.casefold()]
            if not node["upstream_streams"] and not node["downstream_streams"]:
                diagnostics.append(f"equipment {name} has no connected streams")
        for stream in normalized_streams:
            stream["source"] = stream.get("source_node")
            stream.setdefault("provenance_source", "unknown")
            if stream.get("stream_type") == "material" and ProcessContext(model_id="cycle-check", streams=normalized_streams)._is_cycle_stream(stream):
                stream["stream_type"] = "recycle"
        canonical_topology = {"nodes": normalized_nodes, "edges": edges,
                              "streams": [{"id": s["id"], "source": s.get("source_node"), "target": s.get("target")} for s in normalized_streams]}
        return normalized_nodes, normalized_streams, canonical_topology, compositions, conflicts, facts, inferences, diagnostics

    def build(self, *, session: Any = None, model_id: str | None = None) -> ProcessContext:
        from .versions import dependencies
        from src.agents.actions.authorization import digest
        cfg = self._load_config()
        resource = self.resolve_model_resource(session=session, config=cfg)
        self.catalog_warning = None
        equipment, streams, topology, sources = self._model_parts(cfg, resource)
        sources.append(self._evidence("case.model_file", EvidenceSource.SYSTEM,
                                      resource.error or "已验证 case 模型文件可读", resource.source_ref,
                                      resource.to_dict()))
        equipment_parameters = self._equipment_parameter_parts(self._catalog_entries_cache)
        variables, objectives, constraints, yaml_evidence = self._yaml_parts(cfg)
        configured_parameters = cfg.get("equipment_parameters", {})
        if isinstance(configured_parameters, dict):
            for equipment_id, raw_parameters in configured_parameters.items():
                if isinstance(raw_parameters, dict):
                    raw_parameters = [{"name": name, "value": value} for name, value in raw_parameters.items()]
                if isinstance(raw_parameters, list):
                    equipment_parameters.setdefault(str(equipment_id), []).extend(
                        dict(item, source="config") for item in raw_parameters if isinstance(item, dict)
                    )
        # A YAML-only session has no Aspen/NodeDB topology by design.  Still
        # expose equipment names that are explicitly present in configured
        # variable paths, so read-only questions such as "有哪些设备" can be
        # answered without opening Aspen COM.  These are configuration-derived
        # names and are labelled accordingly, never presented as live model
        # verification.
        if not equipment:
            names: dict[str, dict[str, Any]] = {}
            for var in variables.values():
                for name in var.get("related_equipment", []) or []:
                    names.setdefault(str(name), {
                        "name": str(name),
                        "aspen_path": rf"\Data\Blocks\{name}",
                        "block_type": "",
                        "category": "unknown",
                        "alias": None,
                        "source": "yaml_path",
                    })
            equipment = list(names.values())
            if equipment:
                sources.append(self._evidence(
                    "yaml.equipment_inferred", EvidenceSource.YAML,
                    f"从 YAML 变量路径推断出 {len(equipment)} 个设备名称（未进行 Aspen 实时验证）",
                    str(self.yaml_path) if self.yaml_path else "config",
                    {"equipment": [x["name"] for x in equipment]},
                ))
        sim_equipment, sim_streams, sim_topology, sim_sources = self._simulation_model_parts(session)
        sources.extend(sim_sources)
        if sim_equipment or sim_streams:
            equipment.extend(sim_equipment)
            streams.extend(sim_streams)
            topology = {"nodes": list(topology.get("nodes", [])) + list(sim_topology.get("nodes", [])),
                        "streams": list(topology.get("streams", [])) + list(sim_topology.get("streams", [])),
                        "edges": list(topology.get("edges", [])) + list(sim_topology.get("edges", []))}
        sources.extend(yaml_evidence)
        # Normalize after YAML-only equipment discovery so every input mode
        # (BKP, YAML, NodeDB, SimulationDB or an in-memory mapping) has the
        # same graph vocabulary.  Existing ``equipment``/``streams`` values
        # are retained as compatibility views of these normalized records.
        (equipment, streams, topology, compositions, conflicts, facts,
         inferences, graph_diagnostics) = self._normalize_graph(
            equipment, streams, topology, cfg, sources)
        session_id = self._session_value(session, "session_id")
        case_id = self._session_value(session, "case_id")
        user_input = self._session_value(session, "user_input")
        if user_input:
            sources.append(self._evidence("user.input", EvidenceSource.USER_INPUT, "用户输入已纳入上下文", str(session_id or "session"), user_input))
        current_values: dict[str, Any] = {}
        if isinstance(cfg.get("current_values"), dict):
            current_values.update(dict(cfg["current_values"]))
        diagnostics: list[str] = []
        diagnostics.extend(graph_diagnostics)
        if self.catalog_warning:
            diagnostics.append(self.catalog_warning)
        if resource.path and resource.path.suffix.casefold() == ".apw":
            diagnostics.append("APW 不支持静态正文解析；仅提供文件版本及已有 NodeDB/拓扑快照，未运行验证")
        if self.node_db and case_id:
            for row in self.node_db.get_node_values(str(case_id), include_errors=True):
                current_values[row["path"]] = {"value": row.get("value"), "unit": row.get("unit", ""), "error": row.get("error"), "source": "historical_node_snapshot", "case_id": case_id}
        elif self.node_db and not case_id:
            diagnostics.append("未提供 case_id，NodeDB 当前值未绑定到具体工况")
        history: list[EvidenceItem] = []
        facts.append({"kind": "static_node_catalog", "catalog_id": self.catalog_id,
                      "evidence_ids": [s.id for s in sources if s.source == EvidenceSource.NODEDB],
                      "nodes": [dict(x) for x in self._catalog_entries_cache]})
        if self.simulation_db:
            rows = self.simulation_db.query_cases(session_id=session_id, limit=100, newest_first=True) if session_id else self.simulation_db.query_cases(limit=100, newest_first=True)
            if not rows:
                diagnostics.append("SimulationDB 历史数据为空")
            for i, row in enumerate(rows):
                stored = self.simulation_db.get_case(row["case_id"]) or row
                history.append(self._evidence(f"simulation.case.{row['case_id']}", EvidenceSource.SIMULATION_DB, f"历史工况 {row.get('case_id')} 状态 {row.get('status')}，不是当前实时值", row.get("case_id"), stored))
            if len(rows) == 100:
                diagnostics.append("历史上下文仅选取精确会话范围内最近 100 条工况；其余记录未读取，可按工况 ID 查询")
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
        # Keep source conflicts explicit and add evidence-backed composition
        # observations.  A delta is a fact; a chemical interpretation remains
        # an inference and is never asserted here.
        for stream in streams:
            if stream.get("composition"):
                facts.append({"kind": "composition_observed", "stream_id": stream.get("id"),
                              "composition": stream.get("composition"),
                              "evidence_ids": list(stream.get("evidence_ids") or [])})
        # Compare only observed inlet/outlet vectors.  The resulting facts
        # describe enrichment/reduction; role labels such as reaction or
        # separation stay explicitly probabilistic and evidence-linked.
        for node in equipment:
            node_id = str(node.get("id") or node.get("name") or "")
            incoming = [s for s in streams if str(s.get("target") or "").casefold() == node_id.casefold() and s.get("composition")]
            outgoing = [s for s in streams if str(s.get("source_node") or "").casefold() == node_id.casefold() and s.get("composition")]
            if not incoming or not outgoing:
                continue
            for before in incoming:
                for after in outgoing:
                    comparison = ProcessContext(
                        model_id="composition-comparison", streams=[before, after]
                    ).compare_compositions(str(before.get("id")), str(after.get("id")))
                    changes = [x for x in comparison["changes"] if x.get("classification") in {"enriched", "reduced"}]
                    if not changes:
                        continue
                    evidence_ids = list(dict.fromkeys(list(before.get("evidence_ids") or []) + list(after.get("evidence_ids") or [])))
                    facts.append({"kind": "composition_change", "equipment_id": node_id,
                                  "input_stream": before.get("id"), "output_stream": after.get("id"),
                                  "changes": changes, "evidence_ids": evidence_ids})
                    block_type = str(node.get("block_type") or "").upper()
                    possible = {"RCSTR": "possible_reaction", "RSTOIC": "possible_reaction",
                                "FLASH2": "possible_separation", "RADFRAC": "possible_separation",
                                "SEP": "possible_separation", "MIXER": "possible_mixing"}.get(block_type)
                    if possible:
                        inferences.append({"kind": possible, "target_id": node_id,
                                           "input_stream": before.get("id"), "output_stream": after.get("id"),
                                           "confidence": 0.45, "inference_method": "block_type_plus_composition_delta",
                                           "evidence_ids": evidence_ids,
                                           "note": "Composition change supports a possible role; it does not establish a chemical identity or mechanism."})
            if len(incoming) > 1 and len(outgoing) >= 1:
                inferences.append({"kind": "possible_mixing_or_dilution", "target_id": node_id,
                                   "confidence": 0.35, "inference_method": "multiple_inlets_and_composition_observation",
                                   "evidence_ids": list(dict.fromkeys(sum((list(s.get("evidence_ids") or []) for s in incoming + outgoing), [])))})
        for stream in streams:
            if context_cycle := ProcessContext(model_id="cycle-check", streams=streams)._is_cycle_stream(stream):
                inferences.append({"kind": "recycle", "target_id": stream.get("id"),
                                   "confidence": 0.9, "inference_method": "topology_cycle", "evidence_ids": list(stream.get("evidence_ids") or [])})
        for conflict in conflicts:
            diagnostics.append(f"conflicting values retained for {conflict.get('kind')} {conflict.get('id')} field {conflict.get('field')}")
        return ProcessContext(
            model_id=model_id or self.model_id or str(cfg.get("model_id") or (self.yaml_path.stem if self.yaml_path else "unknown")),
            model_version=cfg.get("model_version"), equipment=equipment, streams=streams,
            topology=topology, equipment_nodes=equipment, edges=list(topology.get("edges", [])),
            compositions=compositions, variables=variables, objectives=objectives,
            constraints=constraints, current_values=current_values,
            equipment_parameters=equipment_parameters, history_evidence=history,
            sources=sources, session_id=session_id,
            generated_at=datetime.now(timezone.utc).isoformat(), status=status,
            diagnostics=diagnostics, conflicts=conflicts, facts=facts,
            inferences=inferences,
            configuration=dict(cfg),
            dependency_versions={**dependencies({"yaml": self.yaml_path, "model": resource.path,
                "node_db": getattr(self.node_db, "_path", None),
                "simulation_db": getattr(self.simulation_db, "_path", None)}),
                "configuration_digest": digest(cfg), "topology_digest": digest(self.topology)},
        )

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
        selected_ids = {str(e.get("id") or e.get("name")) for e in eq}
        selected_stream_ids = {str(s.get("id") or s.get("name")) for s in st}
        return ProcessContext(model_id=context.model_id, model_version=context.model_version,
            equipment=eq, streams=st,
            topology={"nodes": [n for n in context.topology.get("nodes", []) if str(n.get("id") or n.get("name")) in selected_ids], "edges": edges},
            equipment_nodes=eq, edges=edges,
            compositions={k: v for k, v in context.compositions.items() if k in selected_stream_ids},
            variables=vars_, objectives=context.objectives,
            constraints=[c for c in context.constraints if names.intersection(set(c.get("related_streams", [])))],
            current_values={k: v for k, v in context.current_values.items() if any(k.startswith((e.get("aspen_path") or "")) for e in eq + st)},
            equipment_parameters={k: v for k, v in context.equipment_parameters.items() if k in selected_ids},
            history_evidence=context.history_evidence, sources=context.sources,
            session_id=context.session_id, generated_at=context.generated_at,
            status=context.status, diagnostics=context.diagnostics,
            conflicts=context.conflicts, facts=context.facts, inferences=context.inferences,
            attachments=context.attachments, dependency_versions=context.dependency_versions,
            configuration=context.configuration)

    # Explicit builder-level wrappers make the read-only API convenient for
    # callers that do not want to depend on ProcessContext implementation
    # details, while the context methods remain the canonical behavior.
    @staticmethod
    def get_equipment(context: ProcessContext, equipment_id: str) -> dict[str, Any] | None:
        return context.get_equipment(equipment_id)

    @staticmethod
    def get_stream(context: ProcessContext, stream_id: str) -> dict[str, Any] | None:
        return context.get_stream(stream_id)

    @staticmethod
    def downstream(context: ProcessContext, equipment_id: str) -> list[str]:
        return context.downstream(equipment_id)

    @staticmethod
    def upstream(context: ProcessContext, equipment_id: str) -> list[str]:
        return context.upstream(equipment_id)

    @staticmethod
    def trace_stream(context: ProcessContext, stream_id: str) -> dict[str, Any] | None:
        return context.trace_stream(stream_id)

    @staticmethod
    def trace_path(context: ProcessContext, start_id: str, end_id: str) -> list[dict[str, Any]]:
        return context.trace_path(start_id, end_id)

    @staticmethod
    def terminal_streams(context: ProcessContext) -> list[dict[str, Any]]:
        return context.terminal_streams()

    @staticmethod
    def boundary_nodes(context: ProcessContext) -> list[dict[str, Any]]:
        return context.boundary_nodes()

    @staticmethod
    def summarize_process(context: ProcessContext) -> dict[str, Any]:
        return context.summarize_process()

    @staticmethod
    def compare_compositions(context: ProcessContext, input_stream: str, output_stream: str) -> dict[str, Any]:
        return context.compare_compositions(input_stream, output_stream)
