"""Read-only case resource resolution shared by context and planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseResource:
    path: Path | None
    candidate_path: Path | None
    source: str
    source_ref: str | None
    raw_path: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def resolve_model_file(raw: Any, *, yaml_path=None, base_dir=None,
                       source="aspen_file", source_ref=None) -> CaseResource:
    """Prefer an existing CWD path, then a config-local path, as the builder did.

    Only ``path`` is usable for planning. ``candidate_path`` is diagnostic and
    may name a missing file. Validation opens the file read-only, never Aspen.
    """
    if raw is None or raw == "":
        return CaseResource(None, None, source, source_ref, None, "case 未提供 BKP/APW 路径")
    target = None
    try:
        target = Path(str(raw)).expanduser()
        if not target.is_absolute():
            cwd = Path.cwd() / target
            base = Path(base_dir) if base_dir else (Path(yaml_path).parent if yaml_path else Path.cwd())
            local = base / target
            target = cwd if cwd.exists() or not local.exists() else local
        target = target.resolve(strict=False)
        if target.suffix.casefold() not in {".bkp", ".apw"}:
            error = f"模型文件必须是 BKP/APW: {target}"
        elif not target.is_file():
            error = f"模型文件不存在或不是文件: {target}"
        else:
            with target.open("rb") as handle:
                handle.read(1)
            return CaseResource(target, target, source, source_ref, str(raw))
    except (OSError, ValueError) as exc:
        error = f"模型文件不可访问: {raw} ({type(exc).__name__})"
    return CaseResource(None, target, source, source_ref, str(raw), error)


def session_model_file(session):
    refs = session.get("context_refs", {}) if isinstance(session, dict) else getattr(session, "context_refs", {})
    return (refs or {}).get("model_file")


def resolve_case_resource(builder, *, session=None) -> CaseResource:
    """One planning entry point; providers own configuration interpretation."""
    # ContextLoader is a transparent wrapper, not a separate configuration.
    from .loader import ContextLoader
    if isinstance(builder, ContextLoader):
        builder = builder.builder
    if hasattr(builder, "resolve_model_resource"):
        return builder.resolve_model_resource(session=session)
    override = session_model_file(session)
    # Compatibility for lightweight providers implementing build + aspen_file.
    return resolve_model_file(override or getattr(builder, "aspen_file", None),
                              source="session.model_file" if override else "aspen_file")
