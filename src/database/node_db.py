"""
node_db.py — SQLite 持久化层，存储 Aspen Plus 树节点原始数据。

与 simulation_db.py 的分工
--------------------------
simulation_db  存工况级聚合结果（ProcessCase → blocks/streams JSON 块）。
node_db        存节点级原始数据：TreeExporter 产出的 TreeValueRecord 列表、
               AspenNode.info() 产出的 NodeInfo 元数据、以及读取失败记录。

两者通过 case_id（TEXT）逻辑关联，无跨文件外键约束。

Schema（3 张表）
-----------------
node_values   — 原始节点值，每行一个 TreeValueRecord。
node_metadata — 节点元数据缓存，按 path 键控，跨 case 复用。
node_errors   — 失败节点索引（node_values 中 error 非 NULL 的冗余副本），
                供 agent 快速诊断哪些路径反复失败。

用法
----
    from pathlib import Path
    from src.database.node_db import NodeDB

    with NodeDB("cases/demo_case/output/node.db") as db:
        db.save_node_values_bulk(
            case_id=case.case_id,
            exports=block_records,
            source_prefix="block",
        )
        rows = db.get_node_values(case_id, source="block:T0301")
        failures = db.get_recurring_failures(min_case_count=2)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..aspen_driver.exporter import TreeValueRecord
    from ..aspen_driver.node import NodeInfo

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS node_values (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    path       TEXT    NOT NULL,
    rel_path   TEXT    NOT NULL,
    value      TEXT,
    unit       TEXT    NOT NULL DEFAULT '',
    value_type INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    created_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nv_case_id     ON node_values (case_id);
CREATE INDEX IF NOT EXISTS idx_nv_case_source ON node_values (case_id, source);
CREATE INDEX IF NOT EXISTS idx_nv_path        ON node_values (path);

CREATE TABLE IF NOT EXISTS node_metadata (
    path         TEXT    PRIMARY KEY,
    name         TEXT    NOT NULL,
    value        TEXT,
    unit_string  TEXT    NOT NULL DEFAULT '',
    value_type   INTEGER NOT NULL DEFAULT 0,
    dimension    INTEGER NOT NULL DEFAULT 0,
    is_output    INTEGER NOT NULL DEFAULT 0,
    is_enterable INTEGER NOT NULL DEFAULT 0,
    record_type  TEXT    NOT NULL DEFAULT '',
    has_children INTEGER NOT NULL DEFAULT 0,
    children     TEXT    NOT NULL DEFAULT '[]',
    cached_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS node_errors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    path       TEXT    NOT NULL,
    rel_path   TEXT    NOT NULL,
    error      TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ne_case_id ON node_errors (case_id);
CREATE INDEX IF NOT EXISTS idx_ne_path    ON node_errors (path);
"""


class NodeDB:
    """
    SQLite 持久化层，存储 Aspen Plus 树节点原始数据。

    Parameters
    ----------
    db_path:
        SQLite 文件路径，如 ``Path("cases/demo_case/output/node.db")``。
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

    def __enter__(self) -> NodeDB:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """关闭 SQLite 连接，可重复调用。"""
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 写入：节点值
    # ------------------------------------------------------------------ #

    def save_node_values(
        self,
        case_id: str,
        source: str,
        records: list[TreeValueRecord],
    ) -> None:
        """
        持久化一批 TreeValueRecord，属于同一 (case_id, source) 对。

        写入前先删除该 (case_id, source) 的旧记录，保证同一工况同一来源
        只保留最新一次导出的快照，不产生重复节点。
        error 非 None 的记录同时写入 node_errors 表，供快速诊断。
        所有操作共享一个事务。

        Parameters
        ----------
        case_id:
            与 simulation.db cases 表的逻辑关联键。
        source:
            导出来源标签，如 ``"block:T0301"`` 或 ``"stream:ADN"``。
        records:
            ``TreeExporter.export_block_outputs()`` 或
            ``export_stream_table()`` 对单个 block/stream 的输出。
            空列表表示该 source 当前无可保存节点，仍会清理旧快照。
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "DELETE FROM node_values WHERE case_id = ? AND source = ?",
                (case_id, source),
            )
            self._conn.execute(
                "DELETE FROM node_errors WHERE case_id = ? AND source = ?",
                (case_id, source),
            )
            for r in records:
                self._insert_record(case_id, source, r, now)

    def save_node_values_bulk(
        self,
        case_id: str,
        exports: dict[str, list[TreeValueRecord]],
        source_prefix: str = "block",
    ) -> None:
        """
        持久化 ``export_block_outputs()`` 或 ``export_stream_table()`` 的完整输出。

        每个 source 写入前先清理旧记录，所有操作共享一个事务。

        Parameters
        ----------
        case_id:
            与 simulation.db cases 表的逻辑关联键。
        exports:
            ``{name: [TreeValueRecord, ...]}``，如
            ``{"T0301": [...], "T0302": [...]}``。
        source_prefix:
            拼接 source 标签的前缀。``"block"`` → ``"block:T0301"``；
            ``"stream"`` → ``"stream:ADN"``。
        """
        if not exports:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            for name, records in exports.items():
                source = f"{source_prefix}:{name}"
                self._conn.execute(
                    "DELETE FROM node_values WHERE case_id = ? AND source = ?",
                    (case_id, source),
                )
                self._conn.execute(
                    "DELETE FROM node_errors WHERE case_id = ? AND source = ?",
                    (case_id, source),
                )
                for r in records:
                    self._insert_record(case_id, source, r, now)

    def _insert_record(
        self,
        case_id: str,
        source: str,
        r: TreeValueRecord,
        now: str,
    ) -> None:
        """在当前事务内插入单条 TreeValueRecord（不开启新事务）。"""
        value_json = (
            None if r.error is not None
            else json.dumps(r.value, default=str)
        )
        self._conn.execute(
            """
            INSERT INTO node_values
                (case_id, source, path, rel_path, value, unit, value_type, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (case_id, source, r.path, r.rel_path,
             value_json, r.unit, r.value_type, r.error, now),
        )
        if r.error is not None:
            self._conn.execute(
                """
                INSERT INTO node_errors
                    (case_id, source, path, rel_path, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (case_id, source, r.path, r.rel_path, r.error, now),
            )

    # ------------------------------------------------------------------ #
    # 写入：元数据缓存
    # ------------------------------------------------------------------ #

    def cache_node_metadata(self, info: NodeInfo) -> None:
        """
        将 NodeInfo 快照写入元数据缓存（INSERT OR REPLACE，按 path 键控）。

        Parameters
        ----------
        info:
            ``AspenNode.info()`` 的返回值。
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO node_metadata
                    (path, name, value, unit_string, value_type, dimension,
                     is_output, is_enterable, record_type, has_children, children, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    info.path, info.name,
                    json.dumps(info.value, default=str),
                    info.unit_string, info.value_type, info.dimension,
                    int(info.is_output), int(info.is_enterable),
                    info.record_type, int(info.has_children),
                    json.dumps(info.children), now,
                ),
            )

    def cache_node_metadata_bulk(self, infos: list[NodeInfo]) -> None:
        """
        批量写入 NodeInfo 快照，单事务。

        Parameters
        ----------
        infos:
            NodeInfo 列表。
        """
        if not infos:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            for info in infos:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO node_metadata
                        (path, name, value, unit_string, value_type, dimension,
                         is_output, is_enterable, record_type, has_children, children, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        info.path, info.name,
                        json.dumps(info.value, default=str),
                        info.unit_string, info.value_type, info.dimension,
                        int(info.is_output), int(info.is_enterable),
                        info.record_type, int(info.has_children),
                        json.dumps(info.children), now,
                    ),
                )

    # ------------------------------------------------------------------ #
    # 查询：节点值
    # ------------------------------------------------------------------ #

    def get_node_values(
        self,
        case_id: str,
        *,
        source: str | None = None,
        include_errors: bool = True,
    ) -> list[dict[str, Any]]:
        """
        返回某工况的所有节点值记录，可按 source 过滤。

        Parameters
        ----------
        case_id:
            目标工况 ID。
        source:
            若指定，只返回该 source 的记录，如 ``"block:T0301"``。
        include_errors:
            ``False`` 时排除 error 非 NULL 的记录。

        Returns
        -------
        list[dict]
            每行含 ``id, case_id, source, path, rel_path, value,
            unit, value_type, error, created_at``。
            ``value`` 已从 JSON 解码为原始 Python 类型。
        """
        sql = "SELECT * FROM node_values WHERE case_id = ?"
        params: list[Any] = [case_id]
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        if not include_errors:
            sql += " AND error IS NULL"
        sql += " ORDER BY id ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_node_value_dict(r) for r in rows]

    def get_node_values_by_path_pattern(
        self,
        pattern: str,
        *,
        case_id: str | None = None,
        source: str | None = None,
        include_errors: bool = True,
    ) -> list[dict[str, Any]]:
        """
        按 SQL LIKE 模式匹配 path，返回节点值记录。

        Parameters
        ----------
        pattern:
            SQL LIKE 模式，如 ``"%TEMP%"`` 或 ``r"%\\T0301\\%"``。
        case_id:
            若指定，只在该工况内搜索。
        source:
            若指定，只返回该 source 的记录，如 ``"block:T0301"``。
        include_errors:
            ``False`` 时排除 error 非 NULL 的记录。

        Returns
        -------
        list[dict]
            与 ``get_node_values`` 相同的结构。
        """
        sql = "SELECT * FROM node_values WHERE path LIKE ?"
        params: list[Any] = [pattern]
        if case_id is not None:
            sql += " AND case_id = ?"
            params.append(case_id)
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        if not include_errors:
            sql += " AND error IS NULL"
        sql += " ORDER BY case_id ASC, id ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_node_value_dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # 查询：失败记录
    # ------------------------------------------------------------------ #

    def get_error_records(
        self,
        case_id: str,
        *,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        返回某工况的所有失败节点记录（来自 node_errors 表）。

        Parameters
        ----------
        case_id:
            目标工况 ID。
        source:
            若指定，只返回该 source 的失败记录。

        Returns
        -------
        list[dict]
            每行含 ``id, case_id, source, path, rel_path, error, created_at``。
        """
        sql = "SELECT * FROM node_errors WHERE case_id = ?"
        params: list[Any] = [case_id]
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY id ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_recurring_failures(
        self,
        min_case_count: int = 2,
        *,
        source_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        返回在至少 ``min_case_count`` 个不同工况中失败的路径。

        供 agent 学习哪些路径结构性损坏，应从未来导出中排除。

        Parameters
        ----------
        min_case_count:
            路径必须在至少这么多个不同工况中失败才会被返回。
        source_prefix:
            若指定，只统计 source 以该前缀开头的失败记录，
            如 ``"block"`` 只看 block 相关失败。

        Returns
        -------
        list[dict]
            每行含 ``path, fail_count, sources, last_error``，
            按 ``fail_count DESC`` 排序。
            ``sources``：出现该失败的 source 标签去重列表（JSON 解码后）。
            ``last_error``：最近一次失败的错误信息。
        """
        sql = """
            SELECT
                path,
                COUNT(DISTINCT case_id)  AS fail_count,
                GROUP_CONCAT(DISTINCT source) AS sources_concat,
                MAX(error)               AS last_error
            FROM node_errors
            WHERE (? IS NULL OR source LIKE ? || ':%')
            GROUP BY path
            HAVING fail_count >= ?
            ORDER BY fail_count DESC
        """
        rows = self._conn.execute(
            sql, (source_prefix, source_prefix, min_case_count)
        ).fetchall()
        result = []
        for r in rows:
            sources_raw = r["sources_concat"] or ""
            sources = sorted(set(s for s in sources_raw.split(",") if s))
            result.append({
                "path":       r["path"],
                "fail_count": r["fail_count"],
                "sources":    sources,
                "last_error": r["last_error"],
            })
        return result

    # ------------------------------------------------------------------ #
    # 查询：元数据缓存
    # ------------------------------------------------------------------ #

    def get_node_metadata(self, path: str) -> dict[str, Any] | None:
        """
        返回指定路径的缓存 NodeInfo，不存在时返回 None。

        Parameters
        ----------
        path:
            Aspen 树绝对路径。

        Returns
        -------
        dict | None
            字段与 NodeInfo 一致，bool 列已还原，children 已解码为 list[str]。
        """
        row = self._conn.execute(
            "SELECT * FROM node_metadata WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_metadata_dict(row)

    def get_node_metadata_bulk(
        self,
        paths: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        批量返回多个路径的缓存元数据。

        Parameters
        ----------
        paths:
            Aspen 树绝对路径列表。

        Returns
        -------
        ``{path: metadata_dict}``，缺失的路径不出现在结果中。
        """
        if not paths:
            return {}
        placeholders = ",".join("?" * len(paths))
        rows = self._conn.execute(
            f"SELECT * FROM node_metadata WHERE path IN ({placeholders})",
            paths,
        ).fetchall()
        return {r["path"]: self._row_to_metadata_dict(r) for r in rows}

    # ------------------------------------------------------------------ #
    # 聚合与维护
    # ------------------------------------------------------------------ #

    def count_node_values(self, case_id: str | None = None) -> int:
        """
        返回 node_values 表的行数，可按 case_id 过滤。

        Parameters
        ----------
        case_id:
            若指定，只统计该工况的行数。
        """
        if case_id is None:
            return self._conn.execute(
                "SELECT COUNT(*) FROM node_values"
            ).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM node_values WHERE case_id = ?", (case_id,)
        ).fetchone()[0]

    def count_cached_paths(self) -> int:
        """返回 node_metadata 表中缓存的路径数。"""
        return self._conn.execute(
            "SELECT COUNT(*) FROM node_metadata"
        ).fetchone()[0]

    def delete_case(self, case_id: str) -> int:
        """
        删除某工况的所有 node_values 和 node_errors 记录。

        不删除 node_metadata（元数据按 path 键控，跨 case 复用）。

        Parameters
        ----------
        case_id:
            目标工况 ID。

        Returns
        -------
        int
            删除的 node_values 行数。
        """
        with self._conn:
            deleted = self._conn.execute(
                "DELETE FROM node_values WHERE case_id = ?", (case_id,)
            ).rowcount
            self._conn.execute(
                "DELETE FROM node_errors WHERE case_id = ?", (case_id,)
            )
        return deleted

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _decode_json(self, val: str | None, default: Any) -> Any:
        if val is None:
            return default
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError) as exc:
            _log.warning("JSON 列解码失败，返回默认值 %r：%s", default, exc)
            return default

    def _row_to_node_value_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["value"] = self._decode_json(d.get("value"), None)
        return d

    def _row_to_metadata_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["value"]        = self._decode_json(d.get("value"), None)
        d["children"]     = self._decode_json(d.get("children"), [])
        d["is_output"]    = bool(d["is_output"])
        d["is_enterable"] = bool(d["is_enterable"])
        d["has_children"] = bool(d["has_children"])
        return d