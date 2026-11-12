"""Dependency identities for read-only context caches (including SQLite WAL)."""
from __future__ import annotations

import hashlib
from pathlib import Path

BUILD_VERSION = "context-2"


def file_version(path):
    if path is None:
        return None
    target = Path(path)
    try:
        # Stream hashing bounds memory, and detects equal-size/mtime edits.
        h = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return {"path": str(target.resolve()), "sha256": h.hexdigest(), "status": "available"}
    except OSError as exc:
        return {"path": str(target.resolve()), "status": "unavailable", "reason": str(exc)}


def dependencies(paths):
    result = {"build_version": BUILD_VERSION}
    result["build_rules"] = {name: file_version(Path(__file__).parent / name)
                             for name in ("builder.py", "adapter.py", "loader.py", "versions.py")}
    for name, path in paths.items():
        result[name] = file_version(path)
        if path is not None and Path(path).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            result[name + ":wal"] = file_version(str(path) + "-wal")
    return result
