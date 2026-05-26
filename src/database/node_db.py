"""
node_db.py — SQLite 持久化层，存储 Aspen Plus 树节点原始数据。

与 simulation_db.py 的分工
--------------------------
simulation_db  存工况级聚合结果（ProcessCase → blocks/streams JSON 块）。
node_db        存节点级原始数据：TreeExporter 产出的 TreeValueRecord 列表、
               AspenNode.info() 产出的 NodeInfo 元数据、以及读取失败记录。
               同时存储 catalog scan 结果和 read manifest。

两者通过 case_id（TEXT）逻辑关联，无跨文件外键约束。

Schema（5 张表）
-----------------
node_values        — 原始节点值，每行一个 TreeValueRecord。
node_metadata      — 节点元数据缓存，按 path 键控，跨 case 复用。
node_errors        — 失败节点索引（node_values 中 error 非 NULL 的冗余副本），
                     供 agent 快速诊断哪些路径反复失败。
node_catalog       — catalog scan 产出的节点发现记录（含 block_type、unit_string 等）。
catalog_scans      — 每次 catalog scan 的元数据（绑定到 Aspen 文件 hash）。
read_manifests     — manifest 构建结果元数据。
read_manifest_items — manifest 中每条读取项（路径 → 语义字段映射）。

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
    from ..models.node_catalog import CatalogEntry, CatalogScan
    from ..models.read_manifest import ReadManifest, ReadManifestItem

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

-- catalog scan 元数据（每次扫描一行）
CREATE TABLE IF NOT EXISTS catalog_scans (
    catalog_id        TEXT    PRIMARY KEY,
    aspen_file_path   TEXT    NOT NULL,
    aspen_file_hash   TEXT    NOT NULL DEFAULT '',
    aspen_version     TEXT    NOT NULL DEFAULT '',
    n_blocks          INTEGER NOT NULL DEFAULT 0,
    n_streams         INTEGER NOT NULL DEFAULT 0,
    n_entries         INTEGER NOT NULL DEFAULT 0,
    scan_depth        INTEGER NOT NULL DEFAULT 5,
    created_at        TEXT    NOT NULL,
    notes             TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cs_file_hash ON catalog_scans (aspen_file_hash);

-- catalog 节点发现记录（每个节点一行）
CREATE TABLE IF NOT EXISTS node_catalog (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_id   TEXT    NOT NULL,
    abs_path     TEXT    NOT NULL,
    rel_path     TEXT    NOT NULL DEFAULT '',
    parent_path  TEXT    NOT NULL DEFAULT '',
    depth        INTEGER NOT NULL DEFAULT 0,
    name         TEXT    NOT NULL DEFAULT '',
    block_name   TEXT    NOT NULL DEFAULT '',
    block_type   TEXT    NOT NULL DEFAULT '',
    stream_name  TEXT    NOT NULL DEFAULT '',
    is_leaf      INTEGER NOT NULL DEFAULT 1,
    has_children INTEGER NOT NULL DEFAULT 0,
    value_type   INTEGER NOT NULL DEFAULT 0,
    unit_string  TEXT    NOT NULL DEFAULT '',
    dimension    INTEGER NOT NULL DEFAULT 0,
    sample_value TEXT,
    sample_error TEXT    NOT NULL DEFAULT '',
    cached_at    TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_nc_catalog_path ON node_catalog (catalog_id, abs_path);
CREATE INDEX IF NOT EXISTS idx_nc_catalog_id   ON node_catalog (catalog_id);
CREATE INDEX IF NOT EXISTS idx_nc_block_name   ON node_catalog (catalog_id, block_name);
CREATE INDEX IF NOT EXISTS idx_nc_block_type   ON node_catalog (catalog_id, block_type);

-- read manifest 元数据
CREATE TABLE IF NOT EXISTS read_manifests (
    manifest_id      TEXT    PRIMARY KEY,
    catalog_id       TEXT    NOT NULL,
    objective_names  TEXT    NOT NULL DEFAULT '[]',
    is_valid         INTEGER NOT NULL DEFAULT 1,
    error            TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rm_catalog_id ON read_manifests (catalog_id);

-- read manifest 条目（每个语义字段一行）
CREATE TABLE IF NOT EXISTS read_manifest_items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_id    TEXT    NOT NULL,
    source_type    TEXT    NOT NULL DEFAULT 'block',
    source_name    TEXT    NOT NULL DEFAULT '',
    equipment_type TEXT    NOT NULL DEFAULT '',
    semantic_field TEXT    NOT NULL DEFAULT '',
    abs_path       TEXT    NOT NULL DEFAULT '',
    rel_path       TEXT    NOT NULL DEFAULT '',
    unit_string    TEXT    NOT NULL DEFAULT '',
    value_type     INTEGER NOT NULL DEFAULT 0,
    required       INTEGER NOT NULL DEFAULT 1,
    confidence     REAL    NOT NULL DEFAULT 1.0,
    rule_id        TEXT    NOT NULL DEFAULT '',
    error          TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rmi_manifest_id ON read_manifest_items (manifest_id);
CREATE INDEX IF NOT EXISTS idx_rmi_source      ON read_manifest_items (manifest_id, source_name);
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

    # ------------------------------------------------------------------ #
    # 写入：catalog scan
    # ------------------------------------------------------------------ #

    def save_catalog_scan(self, scan: "CatalogScan") -> None:
        """
        持久化一次 catalog scan 的元数据（INSERT OR REPLACE）。

        Parameters
        ----------
        scan:
            CatalogScan 实例，catalog_id 为主键。
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO catalog_scans
                    (catalog_id, aspen_file_path, aspen_file_hash, aspen_version,
                     n_blocks, n_streams, n_entries, scan_depth, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan.catalog_id, scan.aspen_file_path, scan.aspen_file_hash,
                    scan.aspen_version, scan.n_blocks, scan.n_streams,
                    scan.n_entries, scan.scan_depth, scan.created_at, scan.notes,
                ),
            )

    def save_catalog_entries(self, entries: list["CatalogEntry"]) -> None:
        """
        批量持久化 catalog 节点发现记录，单事务。

        同一 (catalog_id, abs_path) 的记录会被替换（INSERT OR REPLACE）。

        Parameters
        ----------
        entries:
            CatalogEntry 列表，通常来自 CatalogScanner.scan()。
        """
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            for e in entries:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO node_catalog
                        (catalog_id, abs_path, rel_path, parent_path, depth, name,
                         block_name, block_type, stream_name, is_leaf, has_children,
                         value_type, unit_string, dimension, sample_value, sample_error,
                         cached_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        e.catalog_id, e.abs_path, e.rel_path, e.parent_path,
                        e.depth, e.name, e.block_name, e.block_type, e.stream_name,
                        int(e.is_leaf), int(e.has_children),
                        e.value_type, e.unit_string, e.dimension,
                        json.dumps(e.sample_value, default=str) if e.sample_value is not None else None,
                        e.sample_error,
                        e.cached_at or now,
                    ),
                )

    # ------------------------------------------------------------------ #
    # 查询：catalog
    # ------------------------------------------------------------------ #

    def get_catalog_scan(self, catalog_id: str) -> dict[str, Any] | None:
        """返回指定 catalog_id 的扫描元数据，不存在时返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM catalog_scans WHERE catalog_id = ?", (catalog_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_latest_catalog_scan(self, aspen_file_hash: str) -> dict[str, Any] | None:
        """
        按文件 hash 查找最新的 catalog scan 元数据。

        用于判断当前 Aspen 文件是否已有可复用的 catalog。
        """
        row = self._conn.execute(
            """
            SELECT * FROM catalog_scans
            WHERE aspen_file_hash = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (aspen_file_hash,),
        ).fetchone()
        return dict(row) if row else None

    def get_catalog_entries(
        self,
        catalog_id: str,
        *,
        block_name: str | None = None,
        block_type: str | None = None,
        is_leaf: bool | None = None,
        path_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        查询 catalog 节点记录，支持多维过滤。

        Parameters
        ----------
        catalog_id:
            目标 catalog ID。
        block_name:
            若指定，只返回该 block 的节点。
        block_type:
            若指定，只返回该 block 类型的节点（如 "RADFRAC"）。
        is_leaf:
            True 只返回叶节点；False 只返回非叶节点；None 不过滤。
        path_pattern:
            SQL LIKE 模式，匹配 abs_path（如 "%REB_DUTY%"）。

        Returns
        -------
        list[dict]，每行含 node_catalog 表的全部字段，sample_value 已 JSON 解码。
        """
        sql = "SELECT * FROM node_catalog WHERE catalog_id = ?"
        params: list[Any] = [catalog_id]
        if block_name is not None:
            sql += " AND block_name = ?"
            params.append(block_name)
        if block_type is not None:
            sql += " AND block_type = ?"
            params.append(block_type)
        if is_leaf is not None:
            sql += " AND is_leaf = ?"
            params.append(int(is_leaf))
        if path_pattern is not None:
            sql += " AND abs_path LIKE ?"
            params.append(path_pattern)
        sql += " ORDER BY depth ASC, abs_path ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_catalog_entry_dict(r) for r in rows]

    def get_catalog_block_types(self, catalog_id: str) -> dict[str, str]:
        """
        返回 catalog 中所有 block 的 {block_name: block_type} 映射。

        用于 ManifestBuilder 识别 block 类型。
        """
        rows = self._conn.execute(
            """
            SELECT DISTINCT block_name, block_type
            FROM node_catalog
            WHERE catalog_id = ? AND block_name != ''
            ORDER BY block_name
            """,
            (catalog_id,),
        ).fetchall()
        return {r["block_name"]: r["block_type"] for r in rows}

    def count_catalog_entries(self, catalog_id: str) -> int:
        """返回指定 catalog 的节点总数。"""
        return self._conn.execute(
            "SELECT COUNT(*) FROM node_catalog WHERE catalog_id = ?", (catalog_id,)
        ).fetchone()[0]

    # ------------------------------------------------------------------ #
    # 写入：read manifest
    # ------------------------------------------------------------------ #

    def save_manifest(self, manifest: "ReadManifest") -> None:
        """
        持久化 ReadManifest 及其所有 items，单事务。

        同一 manifest_id 的旧记录会被先删除再重写。

        Parameters
        ----------
        manifest:
            ReadManifest 实例，manifest_id 为主键。
        """
        now = manifest.created_at or datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "DELETE FROM read_manifest_items WHERE manifest_id = ?",
                (manifest.manifest_id,),
            )
            self._conn.execute(
                "DELETE FROM read_manifests WHERE manifest_id = ?",
                (manifest.manifest_id,),
            )
            self._conn.execute(
                """
                INSERT INTO read_manifests
                    (manifest_id, catalog_id, objective_names, is_valid, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.manifest_id, manifest.catalog_id,
                    json.dumps(manifest.objective_names),
                    int(manifest.is_valid), manifest.error, now,
                ),
            )
            for item in manifest.items:
                self._conn.execute(
                    """
                    INSERT INTO read_manifest_items
                        (manifest_id, source_type, source_name, equipment_type,
                         semantic_field, abs_path, rel_path, unit_string, value_type,
                         required, confidence, rule_id, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.manifest_id, item.source_type, item.source_name,
                        item.equipment_type, item.semantic_field,
                        item.abs_path, item.rel_path, item.unit_string,
                        item.value_type, int(item.required), item.confidence,
                        item.rule_id, item.error,
                    ),
                )

    # ------------------------------------------------------------------ #
    # 查询：read manifest
    # ------------------------------------------------------------------ #

    def get_manifest(self, manifest_id: str) -> dict[str, Any] | None:
        """返回指定 manifest_id 的元数据，不存在时返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM read_manifests WHERE manifest_id = ?", (manifest_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["objective_names"] = self._decode_json(d.get("objective_names"), [])
        d["is_valid"] = bool(d["is_valid"])
        return d

    def get_latest_manifest(self, catalog_id: str) -> dict[str, Any] | None:
        """
        返回指定 catalog_id 的最新 manifest 元数据（按 created_at 降序）。

        用于 run_case 中 manifest_id="auto" 时自动查找。
        """
        row = self._conn.execute(
            """
            SELECT * FROM read_manifests
            WHERE catalog_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (catalog_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["objective_names"] = self._decode_json(d.get("objective_names"), [])
        d["is_valid"] = bool(d["is_valid"])
        return d

    def get_manifest_items(
        self,
        manifest_id: str,
        *,
        source_name: str | None = None,
        required_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        返回 manifest 的所有 items，可按 source_name 和 required 过滤。

        Parameters
        ----------
        manifest_id:
            目标 manifest ID。
        source_name:
            若指定，只返回该 block/stream 的 items。
        required_only:
            True 时只返回 required=1 的 items。

        Returns
        -------
        list[dict]，每行含 read_manifest_items 表的全部字段。
        """
        sql = "SELECT * FROM read_manifest_items WHERE manifest_id = ?"
        params: list[Any] = [manifest_id]
        if source_name is not None:
            sql += " AND source_name = ?"
            params.append(source_name)
        if required_only:
            sql += " AND required = 1"
        sql += " ORDER BY source_name ASC, semantic_field ASC"
        rows = self._conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["required"] = bool(d["required"])
            result.append(d)
        return result

    # ------------------------------------------------------------------ #
    # 内部工具（续）
    # ------------------------------------------------------------------ #

    def _row_to_catalog_entry_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["sample_value"] = self._decode_json(d.get("sample_value"), None)
        d["is_leaf"]      = bool(d["is_leaf"])
        d["has_children"] = bool(d["has_children"])
        return d