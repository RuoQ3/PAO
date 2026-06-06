"""
query_node_db.py — query_node_db_tool 实现。

功能：查询 NodeDB 中的 Aspen Plus 树节点原始数据、catalog 扫描结果和 read
      manifest，供 agent 在无需重新连接 Aspen 的情况下检索节点值、诊断读取
      失败、理解工艺结构。
不依赖 Aspen COM，可在任意环境中安全调用。

支持五种查询模式：
  node_values      — 按 case_id（+可选 source）返回节点值列表
  path_search      — 按 SQL LIKE 模式跨工况搜索路径
  recurring_errors — 返回反复失败的路径，供 agent 学习结构性损坏路径
  catalog          — 查询 catalog scan 结果（按 block_name/block_type/路径过滤）
  manifest         — 查询 read manifest 的语义字段映射
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_log = logging.getLogger(__name__)

# 单次查询的硬上限（防止上下文爆炸）
_MAX_LIMIT: int = 200

# 合法的查询模式集合
_VALID_MODES = frozenset(
    {"node_values", "path_search", "recurring_errors", "catalog", "manifest"}
)


# ---------------------------------------------------------------------------
# 数据库路径解析（与 query_simulation_db 风格一致）
# ---------------------------------------------------------------------------

def _resolve_db_path(db_path: str) -> Path:
    """
    解析数据库路径，优先级：
    1. 绝对路径直接使用
    2. 相对于当前工作目录
    3. 相对于项目根目录（src/agents/tools/ 往上三层）
    """
    p = Path(db_path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"数据库文件不存在：{p}")
        return p

    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd.resolve()

    project_root = Path(__file__).parent.parent.parent.parent
    from_root = project_root / p
    if from_root.exists():
        return from_root.resolve()

    raise FileNotFoundError(
        f"数据库文件不存在：{db_path!r}\n"
        f"  已尝试：\n"
        f"    {from_cwd}\n"
        f"    {from_root}"
    )


# ---------------------------------------------------------------------------
# 结果格式化辅助函数
# ---------------------------------------------------------------------------

def _fmt_node_value_row(row: dict, index: int) -> str:
    """格式化单条节点值记录为单行显示。"""
    path = row.get("path", "?")
    value = row.get("value")
    unit = row.get("unit", "")
    error = row.get("error")
    source = row.get("source", "?")

    if error:
        return f"  [{index:3d}] [错误] {path}\n        来源: {source}  错误: {error}"

    try:
        val_str = f"{float(value):.6g}" if isinstance(value, (int, float)) else str(value)
    except (TypeError, ValueError):
        val_str = str(value)

    unit_str = f"  [{unit}]" if unit else ""
    return f"  [{index:3d}] {path}\n        值: {val_str}{unit_str}  来源: {source}"


def _fmt_node_values_report(
    db_path: str,
    rows: list[dict],
    case_id: str,
    source: str | None,
    include_errors: bool,
) -> str:
    """格式化 node_values 查询报告。"""
    lines: list[str] = ["=== query_node_db 节点值查询报告 ===", ""]
    lines.append(f"数据库  : {db_path}")
    lines.append(f"case_id : {case_id}")
    if source:
        lines.append(f"source  : {source}")
    lines.append(f"返回记录: {len(rows)} 条（{'含' if include_errors else '不含'}失败节点）")
    lines.append("")

    if not rows:
        lines.append("（无匹配节点记录）")
        return "\n".join(lines)

    n_ok = sum(1 for r in rows if not r.get("error"))
    n_err = len(rows) - n_ok
    lines.append("【统计】")
    lines.append(f"  正常节点 : {n_ok}")
    lines.append(f"  失败节点 : {n_err}")
    lines.append("")

    lines.append("【节点列表】")
    for i, row in enumerate(rows, 1):
        lines.append(_fmt_node_value_row(row, i))
    lines.append("")

    if n_err > 0:
        lines.append(f"【注意】{n_err} 个节点读取失败，可使用 mode='recurring_errors' 查看反复失败路径。")

    return "\n".join(lines)


def _fmt_path_search_report(
    db_path: str,
    rows: list[dict],
    pattern: str,
    case_id: str | None,
) -> str:
    """格式化路径搜索报告。"""
    lines: list[str] = ["=== query_node_db 路径搜索报告 ===", ""]
    lines.append(f"数据库    : {db_path}")
    lines.append(f"搜索模式  : {pattern!r}")
    if case_id:
        lines.append(f"case_id   : {case_id}")
    lines.append(f"匹配记录  : {len(rows)} 条")
    lines.append("")

    if not rows:
        lines.append("（无匹配路径）")
        return "\n".join(lines)

    # 按 case_id 分组统计
    cases_seen: dict[str, int] = {}
    for r in rows:
        cid = r.get("case_id", "?")
        cases_seen[cid] = cases_seen.get(cid, 0) + 1
    lines.append("【工况分布】")
    for cid, cnt in sorted(cases_seen.items()):
        lines.append(f"  {cid} : {cnt} 条")
    lines.append("")

    lines.append("【匹配节点】")
    for i, row in enumerate(rows, 1):
        lines.append(_fmt_node_value_row(row, i))
    lines.append("")

    return "\n".join(lines)


def _fmt_recurring_errors_report(
    db_path: str,
    rows: list[dict],
    min_case_count: int,
    source_prefix: str | None,
) -> str:
    """格式化反复失败路径报告。"""
    lines: list[str] = ["=== query_node_db 反复失败路径报告 ===", ""]
    lines.append(f"数据库      : {db_path}")
    lines.append(f"最小失败工况数: {min_case_count}")
    if source_prefix:
        lines.append(f"来源前缀    : {source_prefix!r}")
    lines.append(f"问题路径数  : {len(rows)}")
    lines.append("")

    if not rows:
        lines.append("（无符合条件的反复失败路径）")
        return "\n".join(lines)

    lines.append("【反复失败路径（按失败工况数降序）】")
    for i, row in enumerate(rows, 1):
        path = row.get("path", "?")
        fail_count = row.get("fail_count", 0)
        sources = row.get("sources", [])
        last_error = row.get("last_error", "")
        sources_str = ", ".join(sources[:5])
        if len(sources) > 5:
            sources_str += f" ... (共 {len(sources)} 个)"
        lines.append(f"  [{i:3d}] 失败工况数={fail_count}  {path}")
        lines.append(f"        来源: {sources_str}")
        lines.append(f"        最近错误: {last_error}")
        lines.append("")

    lines.append("【建议】")
    lines.append("  以上路径在多个工况中均读取失败，建议：")
    lines.append("  1. 检查 Aspen 模型中该路径是否真实存在（可能是版本差异）。")
    lines.append("  2. 在 read manifest 规则中降低该路径的优先级或标记为非必须。")
    lines.append("  3. 如确认为结构性损坏，考虑从未来 catalog scan 中排除。")

    return "\n".join(lines)


def _fmt_catalog_entry_row(row: dict, index: int) -> str:
    """格式化单条 catalog 节点记录。"""
    abs_path = row.get("abs_path", "?")
    block_name = row.get("block_name", "")
    block_type = row.get("block_type", "")
    is_leaf = row.get("is_leaf", True)
    unit_string = row.get("unit_string", "")
    sample_value = row.get("sample_value")
    sample_error = row.get("sample_error", "")

    leaf_str = "叶节点" if is_leaf else "中间节点"
    block_str = f"  block={block_name}({block_type})" if block_name else ""
    unit_str = f"  [{unit_string}]" if unit_string else ""

    if sample_error:
        val_str = f"  样本=[错误: {sample_error}]"
    elif sample_value is not None:
        try:
            val_str = f"  样本={float(sample_value):.4g}{unit_str}"
        except (TypeError, ValueError):
            val_str = f"  样本={sample_value}{unit_str}"
    else:
        val_str = ""

    return f"  [{index:3d}] [{leaf_str}]{block_str}  {abs_path}{val_str}"


def _fmt_catalog_report(
    db_path: str,
    scan: dict | None,
    rows: list[dict],
    block_name: str | None,
    block_type: str | None,
    path_pattern: str | None,
    selection_note: str | None = None,
) -> str:
    """格式化 catalog 查询报告。"""
    lines: list[str] = ["=== query_node_db catalog 查询报告 ===", ""]
    lines.append(f"数据库    : {db_path}")
    if selection_note:
        lines.append(f"⚠ 选择说明: {selection_note}")

    if scan:
        lines.append(f"catalog_id: {scan.get('catalog_id', '?')}")
        lines.append(f"Aspen 文件: {scan.get('aspen_file_path', '?')}")
        lines.append(f"文件 hash : {scan.get('aspen_file_hash', '?')[:16]}...")
        lines.append(
            f"扫描统计  : {scan.get('n_blocks', 0)} blocks, "
            f"{scan.get('n_streams', 0)} streams, "
            f"{scan.get('n_entries', 0)} 节点"
        )
        lines.append(f"扫描深度  : {scan.get('scan_depth', '?')}")
        lines.append(f"创建时间  : {scan.get('created_at', '?')}")
    lines.append("")

    filter_parts: list[str] = []
    if block_name:
        filter_parts.append(f"block_name={block_name!r}")
    if block_type:
        filter_parts.append(f"block_type={block_type!r}")
    if path_pattern:
        filter_parts.append(f"path LIKE {path_pattern!r}")
    filter_str = ", ".join(filter_parts) if filter_parts else "全量"
    lines.append(f"过滤条件  : {filter_str}")
    lines.append(f"返回节点  : {len(rows)} 条")
    lines.append("")

    if not rows:
        lines.append("（无匹配节点）")
        return "\n".join(lines)

    # 按 block_type 统计
    type_counts: dict[str, int] = {}
    for r in rows:
        bt = r.get("block_type", "") or "(无)"
        type_counts[bt] = type_counts.get(bt, 0) + 1
    lines.append("【block_type 分布】")
    for bt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {bt:<20} : {cnt} 个节点")
    lines.append("")

    lines.append("【节点列表】")
    for i, row in enumerate(rows, 1):
        lines.append(_fmt_catalog_entry_row(row, i))
    lines.append("")

    return "\n".join(lines)


def _fmt_manifest_report(
    db_path: str,
    manifest: dict | None,
    items: list[dict],
    source_name: str | None,
    required_only: bool,
    selection_note: str | None = None,
) -> str:
    """格式化 manifest 查询报告。"""
    lines: list[str] = ["=== query_node_db manifest 查询报告 ===", ""]
    lines.append(f"数据库      : {db_path}")
    if selection_note:
        lines.append(f"⚠ 选择说明  : {selection_note}")

    if manifest is None:
        lines.append("（manifest 不存在）")
        return "\n".join(lines)

    is_valid = manifest.get("is_valid", False)
    valid_str = "有效 ✓" if is_valid else "无效 ✗"
    lines.append(f"manifest_id : {manifest.get('manifest_id', '?')}")
    lines.append(f"catalog_id  : {manifest.get('catalog_id', '?')}")
    lines.append(f"状态        : {valid_str}")
    lines.append(f"目标函数    : {', '.join(manifest.get('objective_names', []))}")
    lines.append(f"rules_hash  : {manifest.get('rules_hash', '')[:16]}{'...' if manifest.get('rules_hash') else ''}")
    lines.append(f"创建时间    : {manifest.get('created_at', '?')}")
    if not is_valid and manifest.get("error"):
        lines.append(f"错误信息    : {manifest['error']}")
    lines.append("")

    filter_parts: list[str] = []
    if source_name:
        filter_parts.append(f"source_name={source_name!r}")
    if required_only:
        filter_parts.append("required=True")
    filter_str = ", ".join(filter_parts) if filter_parts else "全量"
    lines.append(f"过滤条件    : {filter_str}")
    lines.append(f"返回条目    : {len(items)}")
    lines.append("")

    if not items:
        lines.append("（无匹配条目）")
        return "\n".join(lines)

    # 按 source_name 分组
    by_source: dict[str, list[dict]] = {}
    for item in items:
        sn = item.get("source_name", "?")
        by_source.setdefault(sn, []).append(item)

    n_required = sum(1 for it in items if it.get("required"))
    n_with_error = sum(1 for it in items if it.get("error"))
    lines.append("【统计】")
    lines.append(f"  必须字段   : {n_required} / {len(items)}")
    lines.append(f"  构建失败   : {n_with_error}")
    lines.append("")

    lines.append("【语义字段映射（按 source 分组）】")
    for sn, group in sorted(by_source.items()):
        eq_type = group[0].get("equipment_type", "")
        lines.append(f"  ── {sn}  ({eq_type}) ──")
        for item in group:
            field = item.get("semantic_field", "?")
            abs_path = item.get("abs_path", "?")
            unit = item.get("unit_string", "")
            conf = item.get("confidence", 1.0)
            required = item.get("required", False)
            err = item.get("error", "")
            req_str = "[必须]" if required else "[可选]"
            unit_str = f"  [{unit}]" if unit else ""
            if err:
                lines.append(f"    {req_str} {field:<25} → [构建失败: {err}]")
            else:
                lines.append(
                    f"    {req_str} {field:<25} → {abs_path}{unit_str}"
                    f"  conf={conf:.2f}"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 参数校验辅助
# ---------------------------------------------------------------------------

def _validate_limit(mode: str, limit: int) -> str | None:
    """校验 limit，返回错误字符串（None 表示合法）。"""
    if limit <= 0:
        return (
            f"错误：limit={limit} 不合法，必须在 1 到 {_MAX_LIMIT} 之间。"
            "请缩小过滤条件后再查询。"
        )
    if limit > _MAX_LIMIT:
        return (
            f"错误：limit={limit} 超过硬上限 {_MAX_LIMIT}。"
            "请缩小过滤条件或分页（offset）查询。"
        )
    return None


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def _impl_query_node_db(
    db_path: str,
    mode: str,
    # --- node_values ---
    case_id: str | None,
    source: str | None,
    include_errors: bool,
    # --- path_search ---
    path_pattern: str | None,
    # --- recurring_errors ---
    min_case_count: int,
    source_prefix: str | None,
    # --- catalog ---
    catalog_id: str | None,
    block_name: str | None,
    block_type: str | None,
    catalog_path_pattern: str | None,
    is_leaf: bool | None,
    # --- manifest ---
    manifest_id: str | None,
    manifest_source_name: str | None,
    required_only: bool,
    # --- 通用 ---
    limit: int,
    offset: int,
) -> str:
    """query_node_db_tool 的核心实现，出错时返回 '错误：' 字符串。"""
    # 0. 严格校验 mode
    if mode not in _VALID_MODES:
        return (
            f"错误：mode={mode!r} 不是合法值。"
            f"可选值：{sorted(_VALID_MODES)}。"
        )

    # 0b. limit 硬上限校验（recurring_errors 结果集有限，不受限）
    limit_err = _validate_limit(mode, limit)
    if limit_err:
        return limit_err

    # 1. 解析数据库路径
    try:
        resolved_path = _resolve_db_path(db_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    # 2. 打开数据库
    try:
        from src.database.node_db import NodeDB
    except ImportError as exc:
        return f"错误：无法导入 NodeDB — {exc}"

    try:
        db = NodeDB(resolved_path)
    except Exception as exc:
        return f"错误：无法打开数据库 [{type(exc).__name__}] — {exc}"

    db_path_str = str(resolved_path)

    try:
        # ---------- 模式：node_values ----------
        if mode == "node_values":
            if not case_id or not case_id.strip():
                return "错误：mode='node_values' 时必须提供 case_id 参数。"

            try:
                rows = db.get_node_values(
                    case_id.strip(),
                    source=source or None,
                    include_errors=include_errors,
                )
            except Exception as exc:
                return f"错误：get_node_values 执行失败 [{type(exc).__name__}] — {exc}"

            # 手动分页（NodeDB.get_node_values 不带 limit/offset）
            off = max(0, offset)
            rows = rows[off: off + limit]
            return _fmt_node_values_report(
                db_path_str, rows, case_id.strip(),
                source or None, include_errors,
            )

        # ---------- 模式：path_search ----------
        if mode == "path_search":
            if not path_pattern or not path_pattern.strip():
                return "错误：mode='path_search' 时必须提供 path_pattern 参数（SQL LIKE 模式，如 '%REB_DUTY%'）。"

            try:
                rows = db.get_node_values_by_path_pattern(
                    path_pattern.strip(),
                    case_id=case_id.strip() if case_id and case_id.strip() else None,
                    source=source or None,
                    include_errors=include_errors,
                )
            except Exception as exc:
                return f"错误：get_node_values_by_path_pattern 执行失败 [{type(exc).__name__}] — {exc}"

            off = max(0, offset)
            rows = rows[off: off + limit]
            return _fmt_path_search_report(
                db_path_str, rows, path_pattern.strip(),
                case_id.strip() if case_id and case_id.strip() else None,
            )

        # ---------- 模式：recurring_errors ----------
        if mode == "recurring_errors":
            min_cnt = max(1, min_case_count)
            try:
                rows = db.get_recurring_failures(
                    min_case_count=min_cnt,
                    source_prefix=source_prefix or None,
                    limit=limit,
                )
            except Exception as exc:
                return f"错误：get_recurring_failures 执行失败 [{type(exc).__name__}] — {exc}"

            return _fmt_recurring_errors_report(
                db_path_str, rows, min_cnt, source_prefix or None,
            )

        # ---------- 模式：catalog ----------
        if mode == "catalog":
            # 若未提供 catalog_id，使用公开接口取全库最新，并在报告中注明
            catalog_selection_note: str | None = None
            if not catalog_id or not catalog_id.strip():
                latest_scan = db.get_latest_catalog_scan_any()
                if latest_scan is None:
                    return "错误：数据库中不存在任何 catalog scan，请先执行 catalog scan。"
                resolved_catalog_id = latest_scan["catalog_id"]
                catalog_selection_note = (
                    f"未提供 catalog_id，已自动使用全库最新 catalog"
                    f"（catalog_id={resolved_catalog_id}，"
                    f"Aspen 文件={latest_scan.get('aspen_file_path', '?')}）。"
                    "请确认此文件/hash 属于当前任务，必要时显式传入 catalog_id 参数。"
                )
                _log.info(
                    "query_node_db: catalog_id 未指定，自动使用全库最新 catalog_id=%s（aspen_file=%s）",
                    resolved_catalog_id, latest_scan.get("aspen_file_path", ""),
                )
            else:
                resolved_catalog_id = catalog_id.strip()

            scan = db.get_catalog_scan(resolved_catalog_id)
            if scan is None:
                return f"错误：catalog_id={resolved_catalog_id!r} 不存在。"

            try:
                rows = db.get_catalog_entries(
                    resolved_catalog_id,
                    block_name=block_name or None,
                    block_type=block_type or None,
                    is_leaf=is_leaf,
                    path_pattern=catalog_path_pattern or None,
                )
            except Exception as exc:
                return f"错误：get_catalog_entries 执行失败 [{type(exc).__name__}] — {exc}"

            off = max(0, offset)
            rows = rows[off: off + limit]
            return _fmt_catalog_report(
                db_path_str, scan, rows,
                block_name or None, block_type or None, catalog_path_pattern or None,
                selection_note=catalog_selection_note,
            )

        # ---------- 模式：manifest ----------
        if mode == "manifest":
            manifest_selection_note: str | None = None
            if not manifest_id or not manifest_id.strip():
                # 需要 catalog_id 来限定范围，避免跨 Aspen 文件误选 manifest
                if not catalog_id or not catalog_id.strip():
                    # 退路：用公开接口取全库最新 catalog，再取该 catalog 下最新 manifest
                    latest_scan = db.get_latest_catalog_scan_any()
                    if latest_scan is None:
                        return (
                            "错误：数据库中不存在任何 catalog scan 或 manifest。"
                            "请提供 manifest_id（或先执行 catalog scan 和 manifest 构建）。"
                        )
                    inferred_catalog_id = latest_scan["catalog_id"]
                    manifest_selection_note = (
                        f"未提供 manifest_id 和 catalog_id，已自动推断使用全库最新"
                        f" catalog（catalog_id={inferred_catalog_id}，"
                        f"Aspen 文件={latest_scan.get('aspen_file_path', '?')}）"
                        "下的最新 manifest。请确认此 manifest 属于当前任务；"
                        "必要时显式传入 manifest_id 或 catalog_id 参数。"
                    )
                    _log.info(
                        "query_node_db: manifest_id 和 catalog_id 均未指定，"
                        "自动使用全库最新 catalog_id=%s 下的最新 manifest。",
                        inferred_catalog_id,
                    )
                else:
                    inferred_catalog_id = catalog_id.strip()
                    manifest_selection_note = (
                        f"未提供 manifest_id，已自动使用 catalog_id={inferred_catalog_id}"
                        " 下的最新 manifest。如有多个 manifest 版本，请显式传入 manifest_id。"
                    )

                latest_manifest = db.get_latest_manifest_by_catalog(inferred_catalog_id)
                if latest_manifest is None:
                    return (
                        f"错误：catalog_id={inferred_catalog_id!r} 下不存在任何 manifest。"
                        "请执行 manifest 构建，或直接提供 manifest_id 参数。"
                    )
                resolved_manifest_id = latest_manifest["manifest_id"]
                _log.info(
                    "query_node_db: 自动选取 manifest_id=%s（catalog_id=%s）",
                    resolved_manifest_id, inferred_catalog_id,
                )
            else:
                resolved_manifest_id = manifest_id.strip()

            manifest = db.get_manifest(resolved_manifest_id)
            if manifest is None:
                return f"错误：manifest_id={resolved_manifest_id!r} 不存在。"

            try:
                items = db.get_manifest_items(
                    resolved_manifest_id,
                    source_name=manifest_source_name or None,
                    required_only=required_only,
                )
            except Exception as exc:
                return f"错误：get_manifest_items 执行失败 [{type(exc).__name__}] — {exc}"

            off = max(0, offset)
            items = items[off: off + limit]
            return _fmt_manifest_report(
                db_path_str, manifest, items,
                manifest_source_name or None, required_only,
                selection_note=manifest_selection_note,
            )

        # 不应到达此处（已在开头校验）
        return f"错误：内部错误，未处理的 mode={mode!r}。"

    finally:
        db.close()


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def query_node_db_tool(
    db_path: str,
    mode: str = "node_values",
    case_id: str = "",
    source: str = "",
    include_errors: bool = True,
    path_pattern: str = "",
    min_case_count: int = 2,
    source_prefix: str = "",
    catalog_id: str = "",
    block_name: str = "",
    block_type: str = "",
    catalog_path_pattern: str = "",
    is_leaf: str = "",
    manifest_id: str = "",
    manifest_source_name: str = "",
    required_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """查询 NodeDB 中的 Aspen Plus 树节点原始数据、catalog 和 read manifest，无需重新连接 Aspen。

    支持五种查询模式（通过 mode 参数切换）：

    **mode='node_values'**（默认）：返回指定工况的节点值列表。
      必填：case_id。可选：source（如 "block:T0301"）、include_errors、limit、offset。

    **mode='path_search'**：按路径模式跨工况搜索节点，适合诊断特定路径在哪些工况中出现。
      必填：path_pattern（SQL LIKE，如 "%REB_DUTY%"、"%\\\\T0301\\\\%"）。
      可选：case_id（限定单个工况）、source、include_errors、limit、offset。

    **mode='recurring_errors'**：返回在多个工况中反复失败的路径，供 agent 学习结构性损坏路径。
      可选：min_case_count（默认 2）、source_prefix（如 "block" 或 "stream"）、limit、offset。
      同样受 limit 限制（默认 50，最大 200）。

    **mode='catalog'**：查询 catalog scan 结果，了解 Aspen 模型结构（block、路径、单位等）。
      可选：catalog_id（不填自动使用最新）、block_name、block_type（如 "RADFRAC"）、
           catalog_path_pattern（如 "%TEMP%"）、is_leaf（"true"/"false"/""）、limit、offset。

    **mode='manifest'**：查询 read manifest 的语义字段映射，了解每个语义字段对应哪条 Aspen 路径。
      可选：manifest_id（不填自动使用最新）、manifest_source_name（如 "T0301"）、
           required_only（只看必须字段）、limit、offset。

    Args:
        db_path: NodeDB SQLite 文件路径（相对于项目根目录或绝对路径）。
            典型路径：``cases/demo_case/output/node.db``。
        mode: 查询模式，可选 ``"node_values"``（默认）、``"path_search"``、
            ``"recurring_errors"``、``"catalog"``、``"manifest"``。
            非法值直接返回错误，不会静默降级。
        case_id: 目标工况 UUID（mode='node_values' 必填；mode='path_search' 可选）。
        source: 来源过滤（mode='node_values'/'path_search' 可选）。
            格式：``"block:T0301"`` 或 ``"stream:ADN"``。不传时不过滤。
        include_errors: 是否包含读取失败的节点记录（默认 True）。
            mode='node_values'/'path_search' 有效。
        path_pattern: SQL LIKE 路径搜索模式（mode='path_search' 必填）。
            示例：``"%REB_DUTY%"``（含 REB_DUTY 的路径）。
        min_case_count: 失败路径的最小工况数阈值（mode='recurring_errors' 有效）。默认 2。
        source_prefix: 来源前缀过滤（mode='recurring_errors' 有效）。
            示例：``"block"`` 只统计 block 来源的失败。
        catalog_id: catalog scan ID（mode='catalog' 有效）。
            不传时自动使用最新一次 catalog scan。
        block_name: 按 block 名过滤（mode='catalog' 有效）。
            示例：``"T0301"``。
        block_type: 按 block 类型过滤（mode='catalog' 有效）。
            示例：``"RADFRAC"``、``"HEATX"``。
        catalog_path_pattern: 按路径 SQL LIKE 过滤（mode='catalog' 有效）。
            示例：``"%TEMP%"``、``"%REB_DUTY%"``。
        is_leaf: 叶节点过滤（mode='catalog' 有效）。
            ``"true"`` 只返回叶节点；``"false"`` 只返回中间节点；``""`` 不过滤。
            其他任何字符串均返回错误（不静默降级）。
        manifest_id: manifest ID（mode='manifest' 有效）。
            不传时自动使用同 catalog 下的最新 manifest（若 catalog_id 也未传则用全库最新 catalog）。
        manifest_source_name: 按 block/stream 名过滤（mode='manifest' 有效）。
            示例：``"T0301"``。
        required_only: 只返回 required=True 的字段（mode='manifest' 有效）。默认 False。
        limit: 最多返回条数（所有模式均受限，默认 50，最大 200）。
        offset: 跳过前 N 条（分页）。默认 0。

    Returns:
        格式化的查询报告文本。出错时返回以 "错误：" 开头的描述字符串。
    """
    # is_leaf 字符串严格校验：只允许 "" / "true" / "false"
    is_leaf_stripped = is_leaf.strip().lower()
    if is_leaf_stripped == "true":
        is_leaf_val: bool | None = True
    elif is_leaf_stripped == "false":
        is_leaf_val = False
    elif is_leaf_stripped == "":
        is_leaf_val = None
    else:
        return (
            f"错误：is_leaf={is_leaf!r} 不合法，只接受 \"true\"、\"false\" 或 \"\"（不过滤）。"
        )

    return _impl_query_node_db(
        db_path=db_path,
        mode=mode,
        case_id=case_id.strip() or None,
        source=source.strip() or None,
        include_errors=include_errors,
        path_pattern=path_pattern.strip() or None,
        min_case_count=min_case_count,
        source_prefix=source_prefix.strip() or None,
        catalog_id=catalog_id.strip() or None,
        block_name=block_name.strip() or None,
        block_type=block_type.strip() or None,
        catalog_path_pattern=catalog_path_pattern.strip() or None,
        is_leaf=is_leaf_val,
        manifest_id=manifest_id.strip() or None,
        manifest_source_name=manifest_source_name.strip() or None,
        required_only=required_only,
        limit=limit,
        offset=offset,
    )
