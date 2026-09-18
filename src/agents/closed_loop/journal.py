"""Atomic checkpoints with a process lock (released by the OS after a crash)."""
from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path

from src.models.process_case import CaseStatus, ConstraintValue, ObjectiveValue, ProcessCase
from src.models.simulation_result import RunStatus, SimulationResult


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encode_case(case: ProcessCase, x: dict, initialization: str) -> dict:
    row = case.to_dict()
    # Keep evidence needed for decisions and replay, without serializing COM objects.
    for key in ("blocks", "streams", "semantic_blocks"):
        row.pop(key, None)
    row["optimizer_inputs"] = dict(x)
    row["initialization"] = initialization
    sim = case.sim_result
    row["actual_inputs"] = dict(sim.actual_inputs) if sim else {}
    row["block_statuses"] = [b.to_dict() for b in sim.block_statuses] if sim else []
    row["output_values"] = {
        p: {"value": v.value, "unit": v.unit} for p, v in sim.outputs.items()
    } if sim else {}
    return clean_json(row)


def decode_case(row: dict) -> ProcessCase:
    sim_raw = row.get("sim_result")
    sim = None
    if sim_raw:
        status = RunStatus(sim_raw["status"])
        sim = SimulationResult(
            status=status, success=status.is_convergent,
            requested_inputs=row["design_vars"], actual_inputs=row.get("actual_inputs", {}),
            error=sim_raw.get("error"), warnings=sim_raw.get("warnings", []),
            run_time=sim_raw.get("run_time", 0.0),
        )
    return ProcessCase(
        case_id=row["case_id"], iteration=row["iteration"], status=CaseStatus(row["status"]),
        design_vars=row["design_vars"], sim_result=sim,
        objectives=[ObjectiveValue(**{k: o[k] for k in ("name", "value", "unit", "minimize", "error")
                                     if k in o}) for o in row["objectives"]],
        constraints=[ConstraintValue(**{k: c[k] for k in ("name", "value", "satisfied", "error")
                                       if k in c}) for c in row["constraints"]],
        tags=row.get("tags", []), notes=row.get("notes", ""), run_id=row.get("run_id"),
    )


class SessionJournal:
    """One controller per journal; every action/result is committed before proceeding."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = open(str(self.path) + ".lock", "a+b")
        self.lock.seek(0, os.SEEK_END)
        if self.lock.tell() == 0:
            self.lock.write(b"0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("Another controller is already using this checkpoint") from None
        try:
            self.db = sqlite3.connect(self.path)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        except Exception:
            self.lock.close()
            raise
        return self

    def load(self):
        row = self.db.execute("SELECT payload FROM checkpoint WHERE id=1").fetchone()
        return json.loads(row[0]) if row else None

    def save(self, state: dict):
        payload = json.dumps(clean_json(state), ensure_ascii=False, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO checkpoint VALUES (1, ?)", (payload,))

    def __exit__(self, *exc):
        self.db.close()
        self.lock.close()
