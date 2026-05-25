"""
simulation_db.py — SQLite 持久化层，存储 ProcessCase 仿真记录。

将 run_case() / param_scan() 产出的 ProcessCase 落盘，供 agent 检索历史
仿真数据，无需重跑 Aspen Plus。

Schema（3 张表）
-----------------
cases      — 主表，每行一个 ProcessCase，重型 JSON 列存 blocks/streams。
objectives — 每行一个目标函数值，支持按值过滤/排序（JOIN cases）。
tags       — 每行一个标签，支持 EXISTS 子查询过滤。

objectives 和 tags 与 cases 的 JSON 列冗余存储：
  - JSON 列（objectives_json / tags_json）供 get_case() 完整重建 dict；
  - 子表供 query_cases / query_by_objective 做高效 SQL 过滤，不解析 JSON。

用法
----
    from pathlib import Path
    from src.database.simulation_db import SimulationDB

    with SimulationDB("cases/demo_case/output/simulation.db") as db:
        db.save_case(case.to_dict())
        rows = db.query_cases(status="success", limit=20)
        top = db.query_by_objective("ADN_FRAC", order_desc=True, limit=5)

线程安全
--------
连接以 check_same_thread=False 打开。并发写入须由调用方加锁；并发读取在
WAL 模式下安全。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS cases (
    case_id               TEXT    PRIMARY KEY,
    iteration             INTEGER NOT NULL,
    status                TEXT    NOT NULL,
    simulation_valid      INTEGER NOT NULL,
    success               INTEGER NOT NULL,
    feasible              INTEGER,
    has_constraints       INTEGER NOT NULL,
    objectives_available  INTEGER NOT NULL,
    constraints_available INTEGER NOT NULL,
    run_time              REAL    NOT NULL DEFAULT 0.0,
    source_filepath       TEXT,
    run_id                TEXT,
    notes                 TEXT    NOT NULL DEFAULT '',
    design_vars           TEXT    NOT NULL DEFAULT '{}',
    objectives_json       TEXT    NOT NULL DEFAULT '[]',
    constraints_json      TEXT    NOT NULL DEFAULT '[]',
    tags_json             TEXT    NOT NULL DEFAULT '[]',
    sim_result            TEXT,
    blocks                TEXT    NOT NULL DEFAULT '{}',
    streams               TEXT    NOT NULL DEFAULT '{}',
    created_at            TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_status    ON cases (status);
CREATE INDEX IF NOT EXISTS idx_cases_iteration ON cases (iteration);
CREATE INDEX IF NOT EXISTS idx_cases_success   ON cases (success);

CREATE TABLE IF NOT EXISTS objectives (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id   TEXT    NOT NULL REFERENCES cases (case_id) ON DELETE CASCADE,
    name      TEXT    NOT NULL,
    value     REAL,
    unit      TEXT    NOT NULL DEFAULT '',
    minimize  INTEGER NOT NULL DEFAULT 1,
    available INTEGER NOT NULL DEFAULT 0,
    error     TEXT
);

CREATE INDEX IF NOT EXISTS idx_objectives_case_id    ON objectives (case_id);
CREATE INDEX IF NOT EXISTS idx_objectives_name       ON objectives (name);
CREATE INDEX IF NOT EXISTS idx_objectives_name_value ON objectives (name, value);

CREATE TABLE IF NOT EXISTS tags (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT    NOT NULL REFERENCES cases (case_id) ON DELETE CASCADE,
    tag     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tags_case_id ON tags (case_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag     ON tags (tag);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_case_tag ON tags (case_id, tag);
"""

# query_cases / query_by_objective 返回的摘要列（不含 blocks/streams/sim_result）
_SUMMARY_COLS = (
    "case_id", "iteration", "status",
    "simulation_valid", "success", "feasible",
    "has_constraints", "objectives_available", "constraints_available",
    "run_time", "source_filepath", "run_id", "notes",
    "design_vars", "objectives_json", "constraints_json", "tags_json",
)


class SimulationDB:
    """
    SQLite 持久化层，存储 ProcessCase 仿真记录。

    Parameters
    ----------
    db_path:
        SQLite 文件路径，如 ``Path("cases/demo_case/output/simulation.db")``。
        父目录不存在时自动创建。
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # 上下文管理器
    # ------------------------------------------------------------------ #

    def __enter__(self) -> SimulationDB:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def save_case(self, case_dict: dict[str, Any]) -> None:
        """
        持久化单个 ProcessCase。

        接受 ``ProcessCase.to_dict()`` 的返回值。若 ``case_id`` 已存在则
        替换（INSERT OR REPLACE），ON DELETE CASCADE 自动清理旧子行。

        Parameters
        ----------
        case_dict:
            ``ProcessCase.to_dict()`` 的输出，必须含 ``case_id`` 键。

        Raises
        ------
        KeyError
            ``case_dict`` 缺少 ``case_id`` 字段。
        sqlite3.Error
            数据库写入失败，事务已回滚。
        """
        with self._conn:
            self._insert_case(case_dict)

    def save_cases(self, case_dicts: list[dict[str, Any]]) -> None:
        """
        批量持久化 ProcessCase，所有插入共享一个事务。

        空列表为空操作，不开启事务。

        Parameters
        ----------
        case_dicts:
            ``ProcessCase.to_dict()`` 输出的列表。

        Raises
        ------
        sqlite3.Error
            任意一条写入失败，整批回滚。
        """
        if not case_dicts:
            return
        with self._conn:
            for d in case_dicts:
                self._insert_case(d)

    def _insert_case(self, d: dict[str, Any]) -> None:
        """在当前事务内插入一条 case（不开启新事务）。"""
        case_id = d["case_id"]
        simulation_valid = bool(d.get("simulation_valid", False))

        # 仿真未收敛时，blocks/streams 快照不可信，拒绝写入非空数据
        if not simulation_valid:
            if d.get("blocks") or d.get("streams"):
                raise ValueError(
                    f"case '{case_id}'：simulation_valid=False 但 blocks/streams 非空，"
                    "拒绝写入不可信的仿真快照。请在上游将失败工况的 blocks/streams 清空后再入库。"
                )

        feasible = d.get("feasible")
        feasible_int = None if feasible is None else int(bool(feasible))

        self._conn.execute(
            """
            INSERT OR REPLACE INTO cases (
                case_id, iteration, status,
                simulation_valid, success, feasible,
                has_constraints, objectives_available, constraints_available,
                run_time, source_filepath, run_id, notes,
                design_vars, objectives_json, constraints_json, tags_json,
                sim_result, blocks, streams, created_at
            ) VALUES (
                ?,?,?,  ?,?,?,  ?,?,?,  ?,?,?,?,  ?,?,?,?,  ?,?,?,  ?
            )
            """,
            (
                case_id,
                d.get("iteration", 0),
                d.get("status", "pending"),
                int(bool(simulation_valid)),
                int(bool(d.get("success", False))),
                feasible_int,
                int(bool(d.get("has_constraints", False))),
                int(bool(d.get("objectives_available", False))),
                int(bool(d.get("constraints_available", False))),
                float(d.get("run_time", 0.0)),
                d.get("source_filepath"),
                d.get("run_id"),
                d.get("notes", ""),
                json.dumps(d.get("design_vars", {}), ensure_ascii=False, default=str),
                json.dumps(d.get("objectives", []),  ensure_ascii=False, default=str),
                json.dumps(d.get("constraints", []), ensure_ascii=False, default=str),
                json.dumps(d.get("tags", []),        ensure_ascii=False),
                json.dumps(d.get("sim_result"),      ensure_ascii=False, default=str)
                    if d.get("sim_result") is not None else None,
                json.dumps(d.get("blocks", {}),      ensure_ascii=False, default=str),
                json.dumps(d.get("streams", {}),     ensure_ascii=False, default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        # INSERT OR REPLACE 已通过 CASCADE 删除旧子行，直接插入新子行
        for obj in d.get("objectives", []):
            self._conn.execute(
                """
                INSERT INTO objectives (case_id, name, value, unit, minimize, available, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    obj.get("name", ""),
                    obj.get("value"),
                    obj.get("unit", ""),
                    int(bool(obj.get("minimize", True))),
                    int(bool(obj.get("available", False))),
                    obj.get("error"),
                ),
            )

        for tag in d.get("tags", []):
            self._conn.execute(
                "INSERT OR IGNORE INTO tags (case_id, tag) VALUES (?, ?)",
                (case_id, tag),
            )

    # ------------------------------------------------------------------ #
    # 点查询
    # ------------------------------------------------------------------ #

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """
        按 UUID 取完整 ProcessCase 记录（含 blocks / streams）。

        JSON 列自动解码为 Python 对象。

        Returns
        -------
        dict | None
            与 ``ProcessCase.to_dict()`` 结构一致的字典，或 ``None``。
        """
        row = self._conn.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_full_dict(row)

    # ------------------------------------------------------------------ #
    # 过滤查询
    # ------------------------------------------------------------------ #

    def query_cases(
        self,
        *,
        status: str | None = None,
        tags: list[str] | None = None,
        iteration_min: int | None = None,
        iteration_max: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        按条件查询摘要行（不含 blocks / streams / sim_result）。

        所有参数可选，可任意组合。结果按 ``iteration ASC, created_at ASC`` 排序。

        Parameters
        ----------
        status:
            按 CaseStatus 值过滤，如 ``"success"``、``"sim_failed"``。
        tags:
            列表中每个 tag 都必须存在（AND 语义）。空列表视为不过滤。
        iteration_min:
            iteration 下界（含）。
        iteration_max:
            iteration 上界（含）。
        limit:
            最多返回行数，``None`` 不限。
        offset:
            跳过前 N 行，用于分页，默认 0。

        Returns
        -------
        list[dict]
            每行含摘要字段，JSON 列已解码。
        """
        cols = ", ".join(f"cases.{c}" for c in _SUMMARY_COLS)
        sql = f"SELECT {cols} FROM cases"
        conditions: list[str] = []
        params: list[Any] = []

        if status is not None:
            conditions.append("cases.status = ?")
            params.append(status)
        if iteration_min is not None:
            conditions.append("cases.iteration >= ?")
            params.append(iteration_min)
        if iteration_max is not None:
            conditions.append("cases.iteration <= ?")
            params.append(iteration_max)
        for tag in (tags or []):
            conditions.append(
                "EXISTS (SELECT 1 FROM tags t WHERE t.case_id = cases.case_id AND t.tag = ?)"
            )
            params.append(tag)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY cases.iteration ASC, cases.created_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
            if offset:
                sql += " OFFSET ?"
                params.append(offset)
        elif offset:
            # SQLite 要求 OFFSET 必须跟在 LIMIT 后，用 -1 表示无上限
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_summary_dict(r) for r in rows]

    def query_by_objective(
        self,
        objective_name: str,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        order_desc: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        按目标函数值过滤/排序，返回摘要行。

        只返回该目标函数值不为 NULL（available=1）的工况。

        Parameters
        ----------
        objective_name:
            目标函数名称，如 ``"ADN_FRAC"``，大小写敏感。
        min_value:
            目标值下界（含）。
        max_value:
            目标值上界（含）。
        order_desc:
            ``True``（默认）降序（最大化目标用）；``False`` 升序（最小化目标用）。
        limit:
            最多返回行数。

        Returns
        -------
        list[dict]
            摘要字段 + ``objective_value: float``。
        """
        cols = ", ".join(f"cases.{c}" for c in _SUMMARY_COLS)
        sql = (
            f"SELECT {cols}, o.value AS objective_value "
            "FROM cases "
            "JOIN objectives o ON o.case_id = cases.case_id "
            "  AND o.name = ? AND o.available = 1 AND o.value IS NOT NULL"
        )
        params: list[Any] = [objective_name]

        if min_value is not None:
            sql += " AND o.value >= ?"
            params.append(min_value)
        if max_value is not None:
            sql += " AND o.value <= ?"
            params.append(max_value)

        direction = "DESC" if order_desc else "ASC"
        sql += f" ORDER BY o.value {direction}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            d = self._row_to_summary_dict(r)
            d["objective_value"] = float(r["objective_value"])
            result.append(d)
        return result

    # ------------------------------------------------------------------ #
    # 聚合
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """返回 cases 表的总行数。"""
        return self._conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """关闭 SQLite 连接，可重复调用。"""
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_json(val: str | None, default: Any) -> Any:
        if val is None:
            return default
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError) as exc:
            _log.warning("JSON 列解码失败，返回默认值 %r：%s", default, exc)
            return default

    def _row_to_summary_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """将摘要列的 Row 转换为 dict，JSON 列解码。"""
        d: dict[str, Any] = {}
        for col in _SUMMARY_COLS:
            d[col] = row[col]

        # bool 列还原
        d["simulation_valid"] = bool(d["simulation_valid"])
        d["success"]          = bool(d["success"])
        d["feasible"]         = None if d["feasible"] is None else bool(d["feasible"])
        d["has_constraints"]        = bool(d["has_constraints"])
        d["objectives_available"]   = bool(d["objectives_available"])
        d["constraints_available"]  = bool(d["constraints_available"])

        # JSON 列解码，并用 summary() 的 key 名对外暴露
        d["design_vars"]  = self._decode_json(d.pop("design_vars"),       {})
        d["objectives"]   = self._decode_json(d.pop("objectives_json"),   [])
        d["constraints"]  = self._decode_json(d.pop("constraints_json"),  [])
        d["tags"]         = self._decode_json(d.pop("tags_json"),         [])
        return d

    def _row_to_full_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """将完整 Row（含 blocks/streams/sim_result）转换为 dict。"""
        keys = row.keys()
        d: dict[str, Any] = {k: row[k] for k in keys}

        # bool 列还原
        d["simulation_valid"] = bool(d["simulation_valid"])
        d["success"]          = bool(d["success"])
        d["feasible"]         = None if d["feasible"] is None else bool(d["feasible"])
        d["has_constraints"]        = bool(d["has_constraints"])
        d["objectives_available"]   = bool(d["objectives_available"])
        d["constraints_available"]  = bool(d["constraints_available"])

        # JSON 列解码，key 名与 ProcessCase.to_dict() 一致
        d["design_vars"]   = self._decode_json(d.pop("design_vars"),       {})
        d["objectives"]    = self._decode_json(d.pop("objectives_json"),   [])
        d["constraints"]   = self._decode_json(d.pop("constraints_json"),  [])
        d["tags"]          = self._decode_json(d.pop("tags_json"),         [])
        d["sim_result"]    = self._decode_json(d.get("sim_result"),        None)
        d["blocks"]        = self._decode_json(d.get("blocks"),            {})
        d["streams"]       = self._decode_json(d.get("streams"),           {})
        return d
