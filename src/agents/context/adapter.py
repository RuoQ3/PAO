"""Configuration driven, read-only entry point for process cases.

The adapter is intentionally a small boundary around :class:`ProcessContextBuilder`.
It resolves case-local paths and selects already persisted snapshots; it never
opens Aspen, writes a model, or runs a simulation.  A caller therefore only
needs to replace a case configuration and its input files to obtain the same
``ProcessContext`` contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

try:  # PyYAML is optional in lightweight deployments.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from src.agents.evidence.models import EvidenceItem, EvidenceSource, ProcessContext

from .builder import ProcessContextBuilder, ProcessContextBuildError


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        value = yaml.safe_load(text) or {}
    else:
        value = json.loads(text) or {}
    if not isinstance(value, dict):
        raise ProcessContextBuildError("case configuration root must be a mapping")
    return value


def _path(value: Any, base: Path, *, legacy_candidates: bool = False) -> Path | None:
    if value in (None, ""):
        return None
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate
    local = base / candidate
    if legacy_candidates and not local.exists():
        root = Path.cwd() / candidate
        if root.exists():
            return root
    return local


@dataclass(frozen=True)
class CaseInputConfig:
    """Stable input contract for one read-only case.

    Paths are retained as resolved (but non-strict) paths.  Nonexistent files
    are deliberately not rejected here: :meth:`ProcessCaseAdapter.validate`
    reports them as diagnostics so a partial context can still expose the
    available sources.
    """

    model_id: str | None = None
    aspen_file: Path | None = None
    yaml_path: Path | None = None
    node_db_path: Path | None = None
    simulation_db_path: Path | None = None
    topology_path: Path | None = None
    read_only: bool = True
    aliases: dict[str, str] = field(default_factory=dict)
    alias_groups: dict[str, dict[str, str]] = field(default_factory=dict)
    source_options: dict[str, bool] = field(default_factory=dict)
    strict: bool = False
    config_path: Path | None = None
    memory: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: str | Path = ".",
                     config_path: str | Path | None = None) -> "CaseInputConfig":
        root = dict(raw)
        case = root.get("case") if isinstance(root.get("case"), Mapping) else root
        case = dict(case or {})
        options = dict(root.get("options") or {}) if isinstance(root.get("options"), Mapping) else {}
        source_options = dict(root.get("sources") or {}) if isinstance(root.get("sources"), Mapping) else {}
        aliases_raw = root.get("aliases") or {}
        aliases: dict[str, str] = {}
        alias_groups: dict[str, dict[str, str]] = {"equipment": {}, "streams": {}}
        if isinstance(aliases_raw, Mapping):
            # Keep the public field flat while accepting the useful
            # ``aliases: {equipment: {}, streams: {}}`` spelling.
            for key, value in aliases_raw.items():
                if isinstance(value, Mapping):
                    group = "streams" if str(key).casefold() in {"stream", "streams"} else "equipment"
                    values = {str(k): str(v) for k, v in value.items()}
                    alias_groups[group].update(values)
                    aliases.update(values)
                elif value is not None:
                    aliases[str(key)] = str(value)
        base = Path(base_dir).expanduser()
        if not base.is_absolute():
            base = (Path.cwd() / base).resolve(strict=False)
        cfg_path = Path(config_path).expanduser() if config_path else None
        if cfg_path is not None and not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve(strict=False)
        memory = root.get("memory", root.get("config", {}))
        if not isinstance(memory, Mapping):
            memory = {}
        # A top-level graph mapping is also a valid in-memory case.  Do not
        # copy wrapper metadata into the builder's process configuration.
        if memory == {}:
            memory = {k: v for k, v in root.items()
                      if k not in {"case", "sources", "options", "aliases", "memory", "config"}}
        return cls(
            model_id=(str(case.get("model_id")) if case.get("model_id") is not None else
                      (str(case.get("id")) if case.get("id") is not None else None)),
            aspen_file=_path(case.get("aspen_file", case.get("aspen_path", (root.get("simulator") or {}).get("filepath"))), base, legacy_candidates=True),
            yaml_path=_path(case.get("yaml_path", case.get("yaml_file")), base),
            node_db_path=_path(case.get("node_db_path", case.get("node_db", (root.get("extraction") or {}).get("catalog_db"))), base, legacy_candidates=True),
            simulation_db_path=_path(case.get("simulation_db_path", case.get("simulation_db")), base, legacy_candidates=True),
            topology_path=_path(case.get("topology_path", case.get("topology_file")), base, legacy_candidates=True),
            read_only=bool(options.get("read_only", case.get("read_only", True))),
            aliases=aliases,
            alias_groups=alias_groups,
            source_options={str(k): bool(v) for k, v in source_options.items()},
            strict=bool(options.get("strict", False)), config_path=cfg_path,
            memory=dict(memory),
        )


class ProcessCaseAdapter:
    """Select case sources and build one cached, provenance preserving context."""

    def __init__(self, config: CaseInputConfig, *, raw_config: Mapping[str, Any] | None = None) -> None:
        self.config = config
        self.raw_config = dict(raw_config or {})
        self._diagnostics: list[str] = []
        self._context: ProcessContext | None = None
        self._source_records: list[dict[str, Any]] = []
        self._loaded_yaml: dict[str, Any] = {}
        self._topology: dict[str, Any] = {}
        self._prepare()

    @classmethod
    def from_config(cls, config_path: str | Path | Mapping[str, Any]) -> "ProcessCaseAdapter":
        if isinstance(config_path, Mapping):
            return cls(CaseInputConfig.from_mapping(config_path), raw_config=config_path)
        path = Path(config_path).expanduser()
        if not path.is_file():
            config = CaseInputConfig.from_mapping({}, base_dir=path.parent, config_path=path)
            adapter = cls(config, raw_config={})
            adapter._diagnostics.insert(0, f"配置文件不存在: {path}")
            return adapter
        try:
            raw = _read_mapping(path)
        except Exception as exc:  # retain a diagnostic instead of fabricating data
            config = CaseInputConfig.from_mapping({}, base_dir=path.parent, config_path=path)
            adapter = cls(config, raw_config={})
            adapter._diagnostics.insert(0, f"配置文件读取失败: {path}: {exc}")
            return adapter
        input_config = CaseInputConfig.from_mapping(raw, base_dir=path.parent, config_path=path)
        # Legacy PAO YAMLs are themselves model/config payloads (they have no
        # ``case:`` wrapper).  Treat that file as the YAML source while still
        # resolving its simulator and extraction paths through this boundary.
        if not isinstance(raw.get("case"), Mapping) and input_config.yaml_path is None:
            input_config = replace(input_config, yaml_path=path.resolve(strict=False), memory={})
        return cls(input_config, raw_config=raw)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "ProcessCaseAdapter":
        """Alias for integrations that name the boundary by its file input."""
        return cls.from_config(config_path)

    @classmethod
    def from_inputs(cls, *, config: Mapping[str, Any] | None = None, **kwargs: Any) -> "ProcessCaseAdapter":
        """Convenience constructor for tests and callers with memory inputs."""
        data = dict(config or {})
        data.update(kwargs)
        return cls(CaseInputConfig.from_mapping(data), raw_config=data)

    def _prepare(self) -> None:
        cfg = self.config
        for label, path, optional in (
            ("Aspen/BKP", cfg.aspen_file, False),
            ("YAML", cfg.yaml_path, False),
            ("NodeDB", cfg.node_db_path, True),
            ("SimulationDB", cfg.simulation_db_path, True),
            ("topology", cfg.topology_path, True),
        ):
            if path is None:
                continue
            exists = path.is_file()
            self._source_records.append({"source": label.casefold().replace("/", "_"), "source_ref": str(path), "available": exists})
            if not exists:
                self._diagnostics.append(f"{label} 文件不存在: {path}" + (" (optional)" if optional else ""))
        if cfg.yaml_path and cfg.yaml_path.is_file() and cfg.source_options.get("use_yaml", True):
            try:
                self._loaded_yaml = _read_mapping(cfg.yaml_path)
            except Exception as exc:
                self._diagnostics.append(f"YAML 配置读取失败: {cfg.yaml_path}: {exc}")
        if cfg.topology_path and cfg.topology_path.is_file():
            try:
                raw = json.loads(cfg.topology_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._topology = raw
                else:
                    self._diagnostics.append(f"拓扑文件根节点不是 mapping: {cfg.topology_path}")
            except Exception as exc:
                self._diagnostics.append(f"拓扑文件读取失败: {cfg.topology_path}: {exc}")

    def validate(self) -> list[str]:
        """Return diagnostics without opening or modifying any source."""
        return list(dict.fromkeys(self._diagnostics))

    def _builder_config(self) -> dict[str, Any]:
        # A separate yaml_file is the model/config payload.  The wrapper
        # metadata remains in CaseInputConfig and is never mixed into design
        # optimization settings.
        payload = dict(self._loaded_yaml or self.config.memory or {})
        if self.config.aspen_file and self.config.source_options.get("use_bkp_index", True):
            simulator = dict(payload.get("simulator") or {})
            simulator["filepath"] = str(self.config.aspen_file)
            payload["simulator"] = simulator
        if self.config.aliases:
            payload.setdefault("equipment_aliases", {})
            payload.setdefault("stream_aliases", {})
            # The flat alias map is safe for either selector; structured
            # aliases were flattened at the public boundary intentionally.
            payload["equipment_aliases"].update(self.config.alias_groups.get("equipment") or self.config.aliases)
            payload["stream_aliases"].update(self.config.alias_groups.get("streams") or self.config.aliases)
        payload.setdefault("model_id", self.config.model_id)
        return payload

    def load(self, *, session: Any = None) -> ProcessContext:
        """Load a read-only ProcessContext; repeated calls share the snapshot."""
        if self._context is not None:
            return self._context
        cfg = self.config
        node_db = cfg.node_db_path if cfg.node_db_path and cfg.node_db_path.is_file() and cfg.source_options.get("use_node_db", True) else None
        sim_db = cfg.simulation_db_path if cfg.simulation_db_path and cfg.simulation_db_path.is_file() and cfg.source_options.get("use_simulation_db", True) else None
        bkp = cfg.aspen_file if cfg.aspen_file and cfg.source_options.get("use_bkp_index", True) else None
        yaml_path = (cfg.yaml_path if cfg.yaml_path and cfg.yaml_path.is_file()
                     and cfg.source_options.get("use_yaml", True) else None)
        if not cfg.read_only:
            self._diagnostics.append("ProcessCaseAdapter enforces read_only=True; no source is opened for writing")
        builder = ProcessContextBuilder(
            yaml_path=yaml_path, config=self._builder_config(), node_db=node_db,
            simulation_db=sim_db, topology=self._topology, model_id=cfg.model_id,
            aspen_file=bkp, read_only=True,
        )
        try:
            context = builder.build(session=session, model_id=cfg.model_id)
        finally:
            builder.close()
        diagnostics = list(dict.fromkeys(list(context.diagnostics) + self.validate()))
        # An adapter owns the final source audit trail.  Builder evidence is
        # retained and wrapper paths are represented as explicit records.
        context_sources = list(context.sources)
        source_kinds = {
            "aspen/bkp": EvidenceSource.ASPEN_MODEL,
            "yaml": EvidenceSource.YAML,
            "nodedb": EvidenceSource.NODEDB,
            "simulationdb": EvidenceSource.SIMULATION_DB,
            "topology": EvidenceSource.ASPEN_MODEL,
        }
        for record in self._source_records:
            ref = record.get("source_ref")
            if not ref or any(item.source_ref == ref for item in context_sources):
                continue
            source_name = str(record.get("source") or "")
            context_sources.append(EvidenceItem(
                id=f"adapter.source.{len(context_sources)}",
                source=source_kinds.get(source_name, EvidenceSource.SYSTEM),
                claim=f"case source {'available' if record.get('available') else 'unavailable'}: {source_name}",
                source_ref=str(ref), value={"available": bool(record.get("available"))},
            ))
        self._context = ProcessContext(**{**context.__dict__, "diagnostics": diagnostics, "sources": context_sources})
        self._source_records.extend({
            "id": item.id, "source": item.source.value, "source_ref": item.source_ref,
            "claim": item.claim, "available": True,
        } for item in self._context.sources)
        return self._context

    build = load

    def sources(self) -> list[dict[str, Any]]:
        """Return source references, including unavailable optional inputs."""
        if self._context is None:
            # This only records paths and does not load files.
            return list(self._source_records)
        return list(self._source_records)


__all__ = ["CaseInputConfig", "ProcessCaseAdapter"]
