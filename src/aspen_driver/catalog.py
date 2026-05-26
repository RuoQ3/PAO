"""
catalog.py — Aspen Plus 节点目录扫描器。

职责：对任意 Aspen case 的 block/stream 树做发现扫描，
将节点路径、类型、单位、深度等元数据写入 NodeDB.node_catalog 表。

核心原则
--------
- 不多线程访问同一 Aspen COM 实例（COM STA 线程模型限制）。
- 扫描失败不中断全部 catalog（strict=False 时记录失败节点）。
- 扫描结果绑定到 Aspen 文件 hash，支持跨 case 复用。
- 扫描深度可配置，发现模式建议 max_depth=6~8。

典型用法
--------
    from src.aspen_driver.catalog import CatalogScanner

    scanner = CatalogScanner(driver, node_db)
    scan = scanner.scan(
        aspen_file_path=str(driver.filepath),
        max_depth=6,
        strict=False,
    )
    print(f"扫描完成：{scan.n_entries} 个节点，{scan.n_blocks} 个 block")
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import AspenNodeError
from .node import AspenNode
from ..models.node_catalog import CatalogEntry, CatalogScan

if TYPE_CHECKING:
    from .driver import AspenDriver
    from ..database.node_db import NodeDB

_log = logging.getLogger(__name__)

_BLOCKS_PATH  = r"\Data\Blocks"
_STREAMS_PATH = r"\Data\Streams"


class CatalogScanner:
    """
    对 Aspen Plus 树做发现扫描，产出 CatalogScan + CatalogEntry 列表。

    Parameters
    ----------
    driver:
        已连接并打开仿真文件的 AspenDriver 实例。
    node_db:
        NodeDB 实例，扫描结果写入此数据库。
    """

    def __init__(self, driver: "AspenDriver", node_db: "NodeDB") -> None:
        self._driver  = driver
        self._node_db = node_db

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def scan(
        self,
        aspen_file_path: str | None = None,
        *,
        max_depth: int = 6,
        strict: bool = False,
        block_names: list[str] | None = None,
        stream_names: list[str] | None = None,
        include_streams: bool = True,
        catalog_id: str | None = None,
    ) -> CatalogScan:
        """
        执行 catalog 扫描，将结果写入 NodeDB。

        Parameters
        ----------
        aspen_file_path:
            Aspen 文件路径，用于计算 hash 和记录元数据。
            None 时从 driver.filepath 获取。
        max_depth:
            扫描深度上限（相对于 block/stream 根节点），默认 6。
            建议发现模式使用 6~8，runtime 模式使用 manifest 直接路径。
        strict:
            True：任意节点扫描失败时抛出异常，中断扫描。
            False（默认）：记录失败节点，继续扫描其他节点。
        block_names:
            指定扫描的 block 列表；None 表示自动枚举所有 block。
        stream_names:
            指定扫描的 stream 列表；None 表示自动枚举所有 stream。
        include_streams:
            True（默认）：同时扫描 stream 节点。
        catalog_id:
            指定 catalog_id；None 时自动生成 UUID。

        Returns
        -------
        CatalogScan
            扫描元数据，已写入 NodeDB。
        """
        file_path = aspen_file_path or (str(self._driver.filepath) if self._driver.filepath else "")
        file_hash = _compute_file_hash(file_path) if file_path else ""
        cid       = catalog_id or str(uuid.uuid4())
        now       = datetime.now(timezone.utc).isoformat()

        entries: list[CatalogEntry] = []
        fail_notes: list[str] = []

        # 扫描 blocks
        scanned_blocks = self._scan_blocks(
            cid, max_depth, strict, block_names, entries, fail_notes
        )

        # 扫描 streams
        scanned_streams: list[str] = []
        if include_streams:
            scanned_streams = self._scan_streams(
                cid, max_depth, strict, stream_names, entries, fail_notes
            )

        notes = ""
        if fail_notes:
            notes = f"扫描失败节点 {len(fail_notes)} 个：" + "；".join(fail_notes[:5])
            if len(fail_notes) > 5:
                notes += f"（共 {len(fail_notes)} 个，仅显示前 5 个）"

        scan = CatalogScan(
            catalog_id=cid,
            aspen_file_path=file_path,
            aspen_file_hash=file_hash,
            aspen_version=self._get_aspen_version(),
            n_blocks=len(scanned_blocks),
            n_streams=len(scanned_streams),
            n_entries=len(entries),
            scan_depth=max_depth,
            created_at=now,
            notes=notes,
        )

        self._node_db.save_catalog_scan(scan)
        self._node_db.save_catalog_entries(entries)

        _log.info(
            "catalog scan 完成：catalog_id=%s，%d blocks，%d streams，%d 节点，%d 失败",
            cid, scan.n_blocks, scan.n_streams, scan.n_entries, len(fail_notes),
        )
        return scan

    # ------------------------------------------------------------------ #
    # 内部：block 扫描
    # ------------------------------------------------------------------ #

    def _scan_blocks(
        self,
        catalog_id: str,
        max_depth: int,
        strict: bool,
        block_names: list[str] | None,
        entries: list[CatalogEntry],
        fail_notes: list[str],
    ) -> list[str]:
        """枚举并扫描所有（或指定）block，返回成功扫描的 block 名称列表。"""
        if not self._driver.node_exists(_BLOCKS_PATH):
            _log.warning("Blocks 根节点不存在：%s", _BLOCKS_PATH)
            return []

        if block_names is None:
            try:
                parent = AspenNode(self._driver, _BLOCKS_PATH)
                block_names = parent.child_names()
            except AspenNodeError as exc:
                msg = f"枚举 block 列表失败：{exc}"
                if strict:
                    raise
                _log.warning(msg)
                fail_notes.append(msg)
                return []

        scanned: list[str] = []
        for bname in block_names:
            block_root = f"{_BLOCKS_PATH}\\{bname}"
            block_type = self._get_block_type(block_root)
            try:
                self._scan_subtree(
                    catalog_id=catalog_id,
                    root_path=block_root,
                    max_depth=max_depth,
                    strict=strict,
                    block_name=bname,
                    block_type=block_type,
                    stream_name="",
                    entries=entries,
                    fail_notes=fail_notes,
                )
                scanned.append(bname)
            except AspenNodeError as exc:
                msg = f"block '{bname}' 扫描失败：{exc}"
                if strict:
                    raise
                _log.warning(msg)
                fail_notes.append(msg)
        return scanned

    # ------------------------------------------------------------------ #
    # 内部：stream 扫描
    # ------------------------------------------------------------------ #

    def _scan_streams(
        self,
        catalog_id: str,
        max_depth: int,
        strict: bool,
        stream_names: list[str] | None,
        entries: list[CatalogEntry],
        fail_notes: list[str],
    ) -> list[str]:
        """枚举并扫描所有（或指定）stream，返回成功扫描的 stream 名称列表。"""
        if not self._driver.node_exists(_STREAMS_PATH):
            _log.warning("Streams 根节点不存在：%s", _STREAMS_PATH)
            return []

        if stream_names is None:
            try:
                parent = AspenNode(self._driver, _STREAMS_PATH)
                stream_names = parent.child_names()
            except AspenNodeError as exc:
                msg = f"枚举 stream 列表失败：{exc}"
                if strict:
                    raise
                _log.warning(msg)
                fail_notes.append(msg)
                return []

        scanned: list[str] = []
        for sname in stream_names:
            stream_root = f"{_STREAMS_PATH}\\{sname}"
            try:
                self._scan_subtree(
                    catalog_id=catalog_id,
                    root_path=stream_root,
                    max_depth=max_depth,
                    strict=strict,
                    block_name="",
                    block_type="MATERIAL",
                    stream_name=sname,
                    entries=entries,
                    fail_notes=fail_notes,
                )
                scanned.append(sname)
            except AspenNodeError as exc:
                msg = f"stream '{sname}' 扫描失败：{exc}"
                if strict:
                    raise
                _log.warning(msg)
                fail_notes.append(msg)
        return scanned

    # ------------------------------------------------------------------ #
    # 内部：子树递归扫描
    # ------------------------------------------------------------------ #

    def _scan_subtree(
        self,
        catalog_id: str,
        root_path: str,
        max_depth: int,
        strict: bool,
        block_name: str,
        block_type: str,
        stream_name: str,
        entries: list[CatalogEntry],
        fail_notes: list[str],
        current_path: str | None = None,
        depth: int = 0,
    ) -> None:
        """递归扫描子树，将每个节点写入 entries 列表。"""
        if current_path is None:
            current_path = root_path

        if not self._driver.node_exists(current_path):
            return

        node = AspenNode(self._driver, current_path)
        rel_path = current_path[len(root_path):].lstrip("\\")
        parent_path = current_path.rsplit("\\", 1)[0] if "\\" in current_path else ""
        name = current_path.rsplit("\\", 1)[-1]
        now  = datetime.now(timezone.utc).isoformat()

        # 读取子节点列表
        try:
            children = node.child_names()
        except AspenNodeError as exc:
            msg = f"枚举 '{current_path}' 子节点失败：{exc}"
            if strict:
                raise
            fail_notes.append(msg)
            children = []

        is_leaf      = len(children) == 0
        has_children = len(children) > 0

        # 读取节点元数据
        value_type  = 0
        unit_string = ""
        dimension   = 0
        sample_value: Any = None
        sample_error = ""

        try:
            value_type = int(node.value_type)
        except Exception:
            pass

        try:
            unit_string = node.get_unit()
        except Exception:
            pass

        try:
            dimension = node.dimension
        except Exception:
            pass

        if is_leaf:
            try:
                sample_value = node.value
            except Exception as exc:
                sample_error = str(exc)

        entries.append(CatalogEntry(
            catalog_id=catalog_id,
            abs_path=current_path,
            rel_path=rel_path,
            parent_path=parent_path,
            depth=depth,
            name=name,
            block_name=block_name,
            block_type=block_type,
            stream_name=stream_name,
            is_leaf=is_leaf,
            has_children=has_children,
            value_type=value_type,
            unit_string=unit_string,
            dimension=dimension,
            sample_value=sample_value,
            sample_error=sample_error,
            cached_at=now,
        ))

        # 递归子节点
        if has_children:
            if depth >= max_depth:
                msg = (
                    f"达到 max_depth={max_depth}，节点 '{current_path}' 仍有子节点，"
                    "未继续展开。增大 max_depth 可获取更深层节点。"
                )
                if strict:
                    raise AspenNodeError(msg)
                _log.debug(msg)
                return

            for child_name in children:
                child_path = f"{current_path}\\{child_name}"
                try:
                    self._scan_subtree(
                        catalog_id=catalog_id,
                        root_path=root_path,
                        max_depth=max_depth,
                        strict=strict,
                        block_name=block_name,
                        block_type=block_type,
                        stream_name=stream_name,
                        entries=entries,
                        fail_notes=fail_notes,
                        current_path=child_path,
                        depth=depth + 1,
                    )
                except AspenNodeError as exc:
                    msg = f"扫描 '{child_path}' 失败：{exc}"
                    if strict:
                        raise
                    fail_notes.append(msg)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _get_block_type(self, block_root: str) -> str:
        """尝试从 HAP_RECORDTYPE 读取 block 类型，失败时返回 ""。"""
        try:
            node = AspenNode(self._driver, block_root)
            hap = self._driver.hap_constants
            if hap is None:
                return ""
            com_node = node._raw_com_node()
            rt = com_node.AttributeValue(hap.get("HAP_RECORDTYPE", -1))
            return str(rt) if rt else ""
        except Exception:
            return ""

    def _get_aspen_version(self) -> str:
        """尝试读取 Aspen Plus 版本字符串，失败时返回 ""。"""
        try:
            app = self._driver._app
            if app is None:
                return ""
            ver = getattr(app, "Version", None) or getattr(app, "VersionNumber", None)
            return str(ver) if ver else ""
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# 模块级工具
# ---------------------------------------------------------------------------

def _compute_file_hash(file_path: str, chunk_size: int = 65536) -> str:
    """
    计算文件的 MD5 摘要（十六进制字符串）。

    文件不存在或读取失败时返回 ""，不抛出异常。
    """
    try:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        _log.debug("计算文件 hash 失败（%s）：%s", file_path, exc)
        return ""
