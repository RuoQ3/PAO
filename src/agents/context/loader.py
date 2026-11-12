"""Minimal, provenance preserving context loading for LeadAgent turns."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.agents.evidence.models import EvidenceItem, EvidenceSource, ProcessContext


@dataclass
class LoadedContext:
    """Bounded context packet passed to Runtime/Subagents."""
    context: Any = None
    summary: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    requested: list[str] = field(default_factory=list)


class ContextLoader:
    """Load only the context sources requested by a validated intent.

    The supplied builder remains read-only.  A full ``ProcessContext`` is
    projected to the requested portions before it reaches a subagent.
    """

    ALLOWED = frozenset({"process_context", "yaml", "node_db", "simulation_db", "history", "files", "none"})

    def __init__(self, builder: Any = None, *, max_file_bytes: int = 32_000) -> None:
        self.builder = builder
        self.max_file_bytes = max_file_bytes

    def load(self, required: Iterable[str], *, session: Any = None, files: Iterable[str] | None = None) -> LoadedContext:
        requested = list(dict.fromkeys(str(x) for x in required))
        if not requested:
            requested = ["none"]
        unknown = set(requested) - self.ALLOWED
        if unknown:
            raise ValueError(f"unsupported context sources: {sorted(unknown)}")
        if "none" in requested:
            return LoadedContext(requested=["none"], summary={"loaded": []})

        full = None
        needs_process = bool(set(requested) & {"process_context", "yaml", "node_db", "simulation_db", "history"})
        if needs_process and self.builder is not None:
            try:
                full = self.builder.build(session=session)
            except TypeError:
                full = self.builder.build(session=session, model_id=None)

        sources: list[dict[str, Any]] = []
        if full is not None:
            sources = self._source_refs(full)
        projected = self._project(full, requested)
        if "files" in requested:
            projected = self._attach_files(projected, files or self._session_files(session), sources)
        summary = self._summary(projected, requested)
        return LoadedContext(context=projected, summary=summary, sources=sources, requested=requested)

    @staticmethod
    def _source_refs(context: Any) -> list[dict[str, Any]]:
        refs = []
        for item in list(getattr(context, "sources", []) or []) + list(getattr(context, "history_evidence", []) or []):
            if isinstance(item, EvidenceItem):
                refs.append({"id": item.id, "source": item.source.value, "source_ref": item.source_ref, "claim": item.claim})
            elif isinstance(item, dict):
                refs.append({"source": item.get("source"), "source_ref": item.get("source_ref"), "claim": item.get("claim")})
        return refs

    @staticmethod
    def _project(context: Any, requested: list[str]) -> Any:
        if context is None:
            return None
        if requested == ["process_context"]:
            return context
        # Preserve a ProcessContext-shaped object while blanking unrequested
        # large fields. This keeps existing adapters compatible and avoids
        # passing complete YAML/history/NodeDB content to every role.
        if not isinstance(context, ProcessContext):
            return context
        keep_process = "process_context" in requested
        keep_yaml = "yaml" in requested
        keep_node = "node_db" in requested
        keep_sim = "simulation_db" in requested or "history" in requested
        return ProcessContext(
            model_id=context.model_id, model_version=context.model_version,
            equipment=list(context.equipment) if keep_process or keep_node else [],
            streams=list(context.streams) if keep_process or keep_node else [],
            topology=dict(context.topology) if keep_process or keep_node else {},
            equipment_nodes=list(context.equipment_nodes) if keep_process or keep_node else [],
            edges=list(context.edges) if keep_process or keep_node else [],
            compositions=dict(context.compositions) if keep_process or keep_node else {},
            variables=dict(context.variables) if keep_yaml or keep_process else {},
            objectives=list(context.objectives) if keep_yaml or keep_process else [],
            constraints=list(context.constraints) if keep_yaml or keep_process else [],
            current_values=dict(context.current_values) if keep_node else {},
            equipment_parameters=dict(context.equipment_parameters) if keep_node else {},
            history_evidence=list(context.history_evidence) if keep_sim else [],
            sources=list(context.sources), session_id=context.session_id,
            generated_at=context.generated_at, status=context.status,
            diagnostics=list(context.diagnostics), conflicts=list(context.conflicts),
            facts=list(context.facts), inferences=list(context.inferences),
            attachments=list(context.attachments), dependency_versions=dict(context.dependency_versions),
            configuration=dict(context.configuration) if keep_yaml or keep_process else {},
        )

    def _attach_files(self, context: Any, files: Iterable[str], sources: list[dict[str, Any]]) -> Any:
        from .versions import file_version
        from dataclasses import replace
        selected = []
        remaining = self.max_file_bytes
        for raw in files:
            path = Path(str(raw)).expanduser()
            record = {"path": str(path.resolve()), "status": "unavailable"}
            # Explicit selection grants content access, except secrets. Never
            # recursively collect siblings, home directories or project trees.
            sensitive = path.name.lower().startswith(".env") or path.name.lower() in {"credentials", "id_rsa", "id_ed25519"} or path.suffix.lower() in {".pem", ".key", ".p12"}
            sensitive |= any(part.lower() in {".aws", ".ssh"} for part in path.parts)
            if sensitive:
                record["reason"] = "secret file excluded"
                selected.append(record)
                continue
            try:
                with path.open("rb") as stream:
                    payload = stream.read(remaining + 1)
                truncated = len(payload) > remaining
                payload = payload[:remaining]
                # Incomplete UTF-8 tail is explicitly accounted for below.
                import codecs
                text = codecs.getincrementaldecoder("utf-8-sig")("strict").decode(payload, final=not truncated)
                record.update(status="partial" if truncated else "available", content=text,
                    version=file_version(path), encoding="utf-8-sig", truncated=truncated,
                    byte_range=[0, len(payload)], content_sha256=__import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
                    line_range=[1, len(text.splitlines())],
                    selection="prefix within shared attachment byte budget")
                remaining -= len(payload)
            except (OSError, UnicodeError) as exc:
                record["reason"] = str(exc)
            selected.append(record)
            sources.append({"source": "file", "source_ref": str(path.resolve()),
                            "claim": "User-selected file snapshot", **record})
        if isinstance(context, ProcessContext):
            diagnostics = list(context.diagnostics) + [f"attachment {r['path']}: {r['status']} {r.get('reason', '')}" for r in selected]
            return replace(context, diagnostics=diagnostics, attachments=selected)
        return {"context": context, "files": selected}

    @staticmethod
    def _session_files(session: Any) -> list[str]:
        if session is None:
            return []
        value = session.get("files", session.get("context_refs", {}).get("files")) if isinstance(session, dict) else getattr(session, "context_refs", {}).get("files")
        return [value] if isinstance(value, str) else list(value or [])

    @staticmethod
    def _summary(context: Any, requested: list[str]) -> dict[str, Any]:
        if context is None:
            return {"loaded": requested, "available": False}
        return {
            "loaded": requested,
            "model_id": getattr(context, "model_id", None),
            "equipment": len(getattr(context, "equipment", []) or []),
            "equipment_nodes": len(getattr(context, "equipment_nodes", []) or getattr(context, "equipment", []) or []),
            "streams": len(getattr(context, "streams", []) or []),
            "edges": len(getattr(context, "edges", []) or getattr(context, "topology", {}).get("edges", []) or []),
            "variables": len(getattr(context, "variables", {}) or {}),
            "history": len(getattr(context, "history_evidence", []) or []),
            "facts": len(getattr(context, "facts", []) or []),
            "inferences": len(getattr(context, "inferences", []) or []),
            "conflicts": len(getattr(context, "conflicts", []) or []),
            "diagnostics": list(getattr(context, "diagnostics", []) or []),
        }
