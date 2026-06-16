"""
topology.py — 从 Aspen Plus COM 读取流程拓扑（Block + Stream 连接关系）

职责
----
提供两种读取方式：

1. read_process_topology(driver)
   在 Aspen 文件已打开的状态下直接读取 Stream 的上下游 Block 连接。
   block_type 通过尝试读取 HAP_RECORDTYPE 属性获取；失败时从 catalog entries 补充。

2. topology_from_catalog_entries(entries)
   从已有的 catalog entries（NodeDB 扫描结果）提取拓扑，不需要 Aspen 连接。
   适合 onboarding 已完成后的快速重建。

返回格式兼容 vue-flow：
    {
        "nodes": [{"id": "T0301", "block_type": "RADFRAC",
                   "label": "T0301", "category": "distillation"}, ...],
        "edges": [{"id": "0318", "source": "T0301",
                   "target": "T0302", "label": "0318"}, ...]
    }

Feed streams（无上游 Block）和 Product streams（无下游 Block）的处理：
  source / target 设为 None，前端负责渲染为外部进/出节点。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.aspen_driver.driver import AspenDriver

_log = logging.getLogger(__name__)

_BLOCKS_ROOT  = r"\Data\Blocks"
_STREAMS_ROOT = r"\Data\Streams"

_UPBLK_SUFFIX = r"\Input\STRM_UPBLK"
_DNBLK_SUFFIX = r"\Input\STRM_DNBLK"

_BLOCK_CATEGORY: dict[str, str] = {
    "RADFRAC":  "distillation",
    "EXTRACT":  "distillation",
    "FLASH2":   "flash",
    "FLASH3":   "flash",
    "HEATX":    "heatex",
    "HEATER":   "heatex",
    "COOLER":   "heatex",
    "PUMP":     "pump",
    "COMPR":    "compressor",
    "MCOMPR":   "compressor",
    "RSTOIC":   "reactor",
    "REQUIL":   "reactor",
    "RGIBBS":   "reactor",
    "RPLUG":    "reactor",
    "RCSTR":    "reactor",
    "MIXER":    "mixer",
    "FSPLIT":   "splitter",
    "SSPLIT":   "splitter",
    "SEP":      "separator",
    "SEP2":     "separator",
}


def _read_str(driver: "AspenDriver", path: str) -> str:
    """读取节点字符串值，失败时返回空字符串。"""
    try:
        val = driver.get_value(path)
        return str(val).strip() if val is not None else ""
    except Exception:
        return ""


def read_process_topology(driver: "AspenDriver") -> dict:
    """
    从已打开的 Aspen 仿真文件读取流程拓扑。
    block_type 从 HAP_RECORDTYPE 读取（需要 hap_constants）；
    失败时赋空字符串，由前端按 "unknown" 类别渲染。
    """
    from src.aspen_driver.node import AspenNode

    nodes: list[dict] = []
    edges: list[dict] = []

    # ── 枚举 Blocks ───────────────────────────────────────────────────────────
    try:
        block_names: list[str] = AspenNode(driver, _BLOCKS_ROOT).child_names()
    except Exception as exc:
        _log.warning("read_process_topology: 无法枚举 Blocks：%s", exc)
        return {"nodes": [], "edges": []}

    for bname in block_names:
        btype = _get_block_type_from_driver(driver, bname)
        nodes.append({
            "id":         bname,
            "label":      bname,
            "block_type": btype,
            "category":   _BLOCK_CATEGORY.get(btype, "unknown"),
        })

    block_set = {n["id"] for n in nodes}

    # ── 枚举 Streams，读上下游 Block ──────────────────────────────────────────
    try:
        stream_names: list[str] = AspenNode(driver, _STREAMS_ROOT).child_names()
    except Exception as exc:
        _log.warning("read_process_topology: 无法枚举 Streams：%s", exc)
        return {"nodes": nodes, "edges": []}

    for sname in stream_names:
        up   = _read_str(driver, f"{_STREAMS_ROOT}\\{sname}{_UPBLK_SUFFIX}")
        down = _read_str(driver, f"{_STREAMS_ROOT}\\{sname}{_DNBLK_SUFFIX}")

        has_up   = bool(up   and up   in block_set)
        has_down = bool(down and down in block_set)

        if not has_up and not has_down:
            continue

        edges.append({
            "id":     sname,
            "label":  sname,
            "source": up   if has_up   else None,
            "target": down if has_down else None,
        })

    _log.info("read_process_topology: %d blocks, %d edges", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges}


def _get_block_type_from_driver(driver: "AspenDriver", block_name: str) -> str:
    """
    从 Aspen COM 读取 block 的设备类型。
    优先使用 hap_constants（EnsureDispatch 成功时可用），
    否则尝试读取根节点的 Value（部分版本会返回类型字符串）。
    """
    from src.aspen_driver.node import AspenNode
    path = f"{_BLOCKS_ROOT}\\{block_name}"
    try:
        node = AspenNode(driver, path)
        # hap_constants 可用时通过 info() 获取 record_type
        if driver.hap_constants:
            info = node.info(driver.hap_constants)
            return (info.record_type or "").upper()
        # 降级：直接读节点 Value
        val = driver.get_value(path)
        return str(val or "").upper()
    except Exception:
        return ""


def topology_from_catalog_entries(entries: list[dict]) -> dict:
    """
    从 catalog entries（NodeDB 扫描结果）提取拓扑，不需要 Aspen 连接。

    catalog 已有 block_name / block_type / stream_name 字段，
    但不含 stream 连接信息，所以此函数只能产出节点列表，无法生成边。
    用于 onboarding 后的快速节点展示（连接关系需要 read_process_topology 补充）。

    Parameters
    ----------
    entries : list[dict]
        NodeDB.get_catalog_entries() 返回的字典列表。

    Returns
    -------
    dict  {"nodes": [...], "edges": []}
    """
    seen_blocks: dict[str, str] = {}   # block_name → block_type
    for e in entries:
        bname = e.get("block_name", "")
        btype = (e.get("block_type") or "").upper()
        if bname and bname not in seen_blocks:
            seen_blocks[bname] = btype

    nodes = [
        {
            "id":         bname,
            "label":      bname,
            "block_type": btype,
            "category":   _BLOCK_CATEGORY.get(btype, "unknown"),
        }
        for bname, btype in seen_blocks.items()
    ]
    return {"nodes": nodes, "edges": []}

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.aspen_driver.driver import AspenDriver

_log = logging.getLogger(__name__)

_BLOCKS_ROOT  = r"\Data\Blocks"
_STREAMS_ROOT = r"\Data\Streams"

# 读取 stream 上下游 block 的 Aspen 路径后缀
_UPBLK_SUFFIX = r"\Input\STRM_UPBLK"
_DNBLK_SUFFIX = r"\Input\STRM_DNBLK"

# block_type → 前端渲染类别（用于颜色/图标区分）
_BLOCK_CATEGORY: dict[str, str] = {
    "RADFRAC":  "distillation",
    "EXTRACT":  "distillation",
    "FLASH2":   "flash",
    "FLASH3":   "flash",
    "HEATX":    "heatex",
    "HEATER":   "heatex",
    "COOLER":   "heatex",
    "PUMP":     "pump",
    "COMPR":    "compressor",
    "MCOMPR":   "compressor",
    "RSTOIC":   "reactor",
    "REQUIL":   "reactor",
    "RGIBBS":   "reactor",
    "RPLUG":    "reactor",
    "RCSTR":    "reactor",
    "MIXER":    "mixer",
    "FSPLIT":   "splitter",
    "SSPLIT":   "splitter",
    "SEP":      "separator",
    "SEP2":     "separator",
}


def _get_block_type(driver: "AspenDriver", block_name: str) -> str:
    """读取 block 的设备类型字符串（如 RADFRAC）。"""
    from src.aspen_driver.node import AspenNode
    path = f"{_BLOCKS_ROOT}\\{block_name}"
    try:
        node = AspenNode(driver, path)
        # info() 返回 NodeInfo dataclass，其中 record_type = HAP_RECORDTYPE
        info = node.info(driver.hap_constants)
        return (info.record_type or "").upper()
    except Exception:
        # hap_constants 不可用时降级：直接读节点 Value（某些版本可读到类型字符串）
        try:
            val = driver.get_value(path)
            return str(val or "").upper()
        except Exception:
            return ""


def read_process_topology(driver: "AspenDriver") -> dict:
    """
    从已打开的 Aspen 仿真文件读取流程拓扑。

    Parameters
    ----------
    driver:
        已连接并打开仿真文件的 AspenDriver 实例。

    Returns
    -------
    dict
        {"nodes": [...], "edges": [...]}
        节点和边格式见模块文档。失败时返回空拓扑。
    """
    from src.aspen_driver.node import AspenNode
    from src.aspen_driver.errors import AspenNodeError

    nodes: list[dict] = []
    edges: list[dict] = []

    # ── 1. 枚举所有 Block ─────────────────────────────────────────────────────
    try:
        blocks_node = AspenNode(driver, _BLOCKS_ROOT)
        block_names = blocks_node.child_names()
    except Exception as exc:
        _log.warning("read_process_topology: 无法枚举 Blocks，返回空拓扑：%s", exc)
        return {"nodes": [], "edges": []}

    for bname in block_names:
        btype = _get_block_type(driver, bname)
        category = _BLOCK_CATEGORY.get(btype, "unknown")
        nodes.append({
            "id":         bname,
            "label":      bname,
            "block_type": btype,
            "category":   category,
        })

    block_set = {n["id"] for n in nodes}

    # ── 2. 枚举所有 Stream，读取上下游连接 ───────────────────────────────────
    try:
        streams_node = AspenNode(driver, _STREAMS_ROOT)
        stream_names = streams_node.child_names()
    except Exception as exc:
        _log.warning("read_process_topology: 无法枚举 Streams：%s", exc)
        return {"nodes": nodes, "edges": []}

    for sname in stream_names:
        up_block   = _read_str_value(driver, f"{_STREAMS_ROOT}\\{sname}{_UPBLK_SUFFIX}")
        down_block = _read_str_value(driver, f"{_STREAMS_ROOT}\\{sname}{_DNBLK_SUFFIX}")

        # 过滤无效连接：上下游 block 都不在已知 block 列表内则跳过
        # （部分内部辅助流股不会出现在 Block 树中）
        has_up   = up_block   and up_block   in block_set
        has_down = down_block and down_block in block_set

        if not has_up and not has_down:
            continue   # 完全孤立的流股，不展示

        edges.append({
            "id":     sname,
            "label":  sname,
            "source": up_block   if has_up   else None,
            "target": down_block if has_down else None,
        })

    _log.info(
        "read_process_topology: %d 个 block，%d 条 stream 边",
        len(nodes), len(edges),
    )
    return {"nodes": nodes, "edges": edges}


def _read_str_value(driver: "AspenDriver", path: str) -> str:
    """读取节点字符串值，失败时返回空字符串。"""
    try:
        val = driver.get_value(path)
        return str(val).strip() if val is not None else ""
    except Exception:
        return ""
