"""
query_simulation_db.py — query_simulation_db_tool 实现。

功能：查询 SimulationDB 历史仿真记录，支持按状态、标签、迭代区间、
      目标函数值范围等多条件过滤，返回 agent 可读的格式化报告。
不依赖 Aspen COM，可在任意环境中安全调用。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_log = logging.getLogger(__name__)

# 单次查询的硬上限：防止一次返回过多条目导致 LLM 上下文爆炸。
# 用户如需超过此限制，请缩小过滤条件后再分页查询。
_MAX_LIMIT: int = 200

# 合法的查询模式集合，用于早期校验。
_VALID_MODES = frozenset({"query", "by_objective", "get_case"})


# ---------------------------------------------------------------------------
# 数据库路径解析
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

    # src/agents/tools/_xxx.py → 项目根
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

def _fmt_design_vars_short(design_vars: dict, max_vars: int = 4) -> str:
    """将设计变量字典格式化为单行短字符串，至多显示 max_vars 个变量。"""
    if not design_vars:
        return "（无）"
    items = list(design_vars.items())[:max_vars]
    parts = []
    for path, val in items:
        # 取 Aspen 树路径最后一段作为短名
        short = path.split("\\")[-1] if "\\" in path else path
        try:
            parts.append(f"{short}={float(val):.4g}")
        except (TypeError, ValueError):
            parts.append(f"{short}={val}")
    suffix = f"  (共 {len(design_vars)} 个变量)" if len(design_vars) > max_vars else ""
    return "  ".join(parts) + suffix


def _fmt_objectives_short(objectives: list[dict]) -> str:
    """将目标函数列表格式化为单行字符串。"""
    if not objectives:
        return "（无）"
    parts = []
    for obj in objectives:
        name = obj.get("name", "?")
        if obj.get("available"):
            val = obj.get("value")
            unit = obj.get("unit", "")
            try:
                val_str = f"{float(val):.4g}"
            except (TypeError, ValueError):
                val_str = str(val)
            parts.append(f"{name}={val_str}{(' ' + unit) if unit else ''}")
        else:
            err = obj.get("error", "不可用")
            parts.append(f"{name}=[{err}]")
    return "  ".join(parts)


def _fmt_case_row(row: dict, index: int, show_design_vars: bool = True) -> str:
    """将单个摘要行格式化为一行记录（用于列表展示）。"""
    case_id = row.get("case_id", "?")
    iteration = row.get("iteration", "?")
    status = row.get("status", "?")
    success = "✓" if row.get("success") else "✗"
    sim_valid = "✓" if row.get("simulation_valid") else "✗"
    run_time = row.get("run_time", 0.0)
    tags = row.get("tags", [])
    tags_str = f"  [{', '.join(tags)}]" if tags else ""

    lines = [
        f"  [{index:3d}] case_id={case_id}  iter={iteration}  status={status}"
        f"  成功={success}  收敛={sim_valid}  耗时={run_time:.1f}s{tags_str}"
    ]

    obj_str = _fmt_objectives_short(row.get("objectives", []))
    lines.append(f"        目标值: {obj_str}")

    if show_design_vars:
        dv_str = _fmt_design_vars_short(row.get("design_vars", {}))
        lines.append(f"        设计变量: {dv_str}")

    return "\n".join(lines)


def _fmt_objective_row(row: dict, index: int) -> str:
    """将 query_by_objective 结果行格式化为两行（含 objective_value 及可行性标志）。"""
    case_id = row.get("case_id", "?")
    iteration = row.get("iteration", "?")
    status = row.get("status", "?")
    run_time = row.get("run_time", 0.0)
    success = "✓" if row.get("success") else "✗"
    sim_valid = "✓" if row.get("simulation_valid") else "✗"
    feasible = row.get("feasible")
    feasible_str = "✓" if feasible is True else ("✗" if feasible is False else "—")
    obj_val = row.get("objective_value")
    try:
        obj_val_str = f"{float(obj_val):.6g}"
    except (TypeError, ValueError):
        obj_val_str = str(obj_val)

    dv_str = _fmt_design_vars_short(row.get("design_vars", {}))
    return (
        f"  [{index:3d}] {obj_val_str:<14}  case_id={case_id}  iter={iteration}"
        f"  status={status}  成功={success}  收敛={sim_valid}  可行={feasible_str}"
        f"  耗时={run_time:.1f}s\n"
        f"        设计变量: {dv_str}"
    )


def _fmt_db_stats(db_path: str, total: int, rows: list[dict], mode: str) -> str:
    """格式化数据库统计摘要（查询报告顶部）。"""
    lines = ["=== query_simulation_db 查询报告 ===", ""]
    lines.append(f"数据库  : {db_path}")
    lines.append(f"总记录数: {total}")
    lines.append(f"查询结果: {len(rows)} 条（{mode}）")
    return "\n".join(lines)


def _fmt_query_cases_report(
    db_path: str,
    total: int,
    rows: list[dict],
    query_desc: str,
) -> str:
    """将 query_cases 结果格式化为完整报告。"""
    lines: list[str] = []
    lines.append(_fmt_db_stats(db_path, total, rows, query_desc))
    lines.append("")

    if not rows:
        lines.append("（无匹配记录）")
        return "\n".join(lines)

    # 统计摘要
    n_success = sum(1 for r in rows if r.get("success"))
    n_sim_failed = sum(1 for r in rows if r.get("status") == "sim_failed")
    n_infeasible = sum(1 for r in rows if r.get("status") == "infeasible")
    lines.append("【结果统计】")
    lines.append(f"  成功工况     : {n_success} / {len(rows)}")
    lines.append(f"  仿真失败     : {n_sim_failed}")
    lines.append(f"  不可行（约束）: {n_infeasible}")
    lines.append("")

    lines.append("【记录列表】")
    for i, row in enumerate(rows, 1):
        lines.append(_fmt_case_row(row, i, show_design_vars=True))
        lines.append("")

    lines.append("【综合结论】")
    if n_success == 0:
        lines.append("  无成功工况，建议检查仿真配置或放宽约束。")
    else:
        success_rate = n_success / len(rows) * 100
        lines.append(f"  成功率 {success_rate:.1f}%，共 {n_success} 个有效工况。")

    # 机器可读区块：列出本次查询返回的所有 case_id。
    # 此区块供 RealToolRunner.get_failed_case_ids 解析，不影响人类可读报告内容。
    lines.append("")
    lines.append("[CASE_IDS]")
    for row in rows:
        cid = row.get("case_id", "")
        if cid:
            lines.append(cid)
    lines.append("[/CASE_IDS]")

    return "\n".join(lines)


def _fmt_query_objective_report(
    db_path: str,
    total: int,
    rows: list[dict],
    objective_name: str,
    order_desc: bool,
    success_only: bool = True,
) -> str:
    """将 query_by_objective 结果格式化为完整报告。"""
    direction = "降序（最大化）" if order_desc else "升序（最小化）"
    scope = "仅成功工况" if success_only else "含失败/不可行工况"
    query_desc = f"按目标函数 {objective_name!r} {direction} 排序（{scope}）"

    lines: list[str] = []
    lines.append(_fmt_db_stats(db_path, total, rows, query_desc))
    lines.append("")

    if not rows:
        hint = (
            f"（未找到目标函数 {objective_name!r} 的成功工况记录。"
            "如需查看失败/不可行工况，请传入 include_unsuccessful=True。）"
            if success_only
            else f"（未找到目标函数 {objective_name!r} 的有效记录）"
        )
        lines.append(hint)
        return "\n".join(lines)

    # 值域统计
    vals = [r["objective_value"] for r in rows]
    lines.append(f"【{objective_name} 值域（success_only={success_only}）】")
    lines.append(f"  最小值: {min(vals):.6g}")
    lines.append(f"  最大值: {max(vals):.6g}")
    if len(vals) > 1:
        avg = sum(vals) / len(vals)
        lines.append(f"  均值  : {avg:.6g}")
    lines.append("")

    lines.append(f"【{objective_name} 排序结果】")
    for i, row in enumerate(rows, 1):
        lines.append(_fmt_objective_row(row, i))
        lines.append("")

    return "\n".join(lines)


def _fmt_get_case_report(db_path: str, row: dict) -> str:
    """将 get_case 完整记录格式化为详细报告。"""
    lines: list[str] = []
    lines.append("=== query_simulation_db 单条记录详情 ===")
    lines.append("")
    lines.append(f"数据库  : {db_path}")
    lines.append("")

    lines.append("【运行状态】")
    lines.append(f"  case_id          : {row.get('case_id', '?')}")
    lines.append(f"  session_id       : {row.get('session_id', '')}")
    lines.append(f"  iteration        : {row.get('iteration', '?')}")
    lines.append(f"  status           : {row.get('status', '?')}")
    lines.append(f"  成功（可采纳）   : {'是' if row.get('success') else '否'}")
    lines.append(f"  仿真收敛         : {'是' if row.get('simulation_valid') else '否'}")
    feasible = row.get("feasible")
    lines.append(f"  可行（约束）     : {'是' if feasible else ('否' if feasible is False else '无约束')}")
    lines.append(f"  运行耗时         : {row.get('run_time', 0.0):.1f} s")
    lines.append(f"  来源文件         : {row.get('source_filepath') or '—'}")
    lines.append(f"  run_id           : {row.get('run_id') or '—'}")
    lines.append(f"  创建时间         : {row.get('created_at', '?')}")
    tags = row.get("tags", [])
    if tags:
        lines.append(f"  标签             : {', '.join(tags)}")
    lines.append("")

    lines.append("【设计变量】")
    design_vars = row.get("design_vars", {})
    if design_vars:
        for path, val in design_vars.items():
            short = path.split("\\")[-1] if "\\" in path else path
            try:
                lines.append(f"  {short:<30} = {float(val):.6g}")
            except (TypeError, ValueError):
                lines.append(f"  {short:<30} = {val}")
    else:
        lines.append("  （无）")
    lines.append("")

    lines.append("【目标函数】")
    objectives = row.get("objectives", [])
    if objectives:
        for obj in objectives:
            name = obj.get("name", "?")
            direction = "最小化↓" if obj.get("minimize") else "最大化↑"
            if obj.get("available"):
                val = obj.get("value")
                unit = obj.get("unit", "")
                try:
                    val_str = f"{float(val):.6g}"
                except (TypeError, ValueError):
                    val_str = str(val)
                lines.append(f"  {name:<20} = {val_str} {unit}  [{direction}]")
            else:
                err = obj.get("error", "不可用")
                lines.append(f"  {name:<20} = [不可用]  错误：{err}")
    else:
        lines.append("  （无目标函数）")
    lines.append("")

    lines.append("【约束条件】")
    constraints = row.get("constraints", [])
    if constraints:
        for con in constraints:
            name = con.get("name", "?")
            if con.get("available"):
                val = con.get("value")
                satisfied = con.get("satisfied")
                sat_str = "满足" if satisfied else "违反"
                try:
                    val_str = f"{float(val):.6g}"
                except (TypeError, ValueError):
                    val_str = str(val)
                lines.append(f"  {name:<25} = {val_str}  [{sat_str}]")
            else:
                err = con.get("error", "不可用")
                lines.append(f"  {name:<25} = [不可用]  错误：{err}")
    else:
        lines.append("  （无约束）")
    lines.append("")

    notes = row.get("notes", "")
    if notes:
        lines.append("【运行注记】")
        for note_line in notes.split("\n")[:10]:
            lines.append(f"  {note_line}")
        lines.append("")

    # blocks / streams 存在时给出摘要，不全量展开（避免超长输出）
    blocks = row.get("blocks", {})
    streams = row.get("streams", {})
    semantic_blocks = row.get("semantic_blocks", {})
    if blocks or streams or semantic_blocks:
        lines.append("【仿真快照摘要】")
        lines.append(f"  blocks 数量        : {len(blocks)}")
        lines.append(f"  streams 数量       : {len(streams)}")
        lines.append(f"  semantic_blocks 数 : {len(semantic_blocks)}")
        if blocks:
            sample_keys = list(blocks.keys())[:5]
            lines.append(f"  blocks 样例        : {', '.join(sample_keys)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 参数解析辅助
# ---------------------------------------------------------------------------

def _parse_tags_str(tags_str: str | None) -> list[str]:
    """将逗号分隔的 tags 字符串解析为列表（空字符串返回空列表）。"""
    if not tags_str or not tags_str.strip():
        return []
    return [t.strip() for t in tags_str.split(",") if t.strip()]


def _parse_optional_int(val: int | None, name: str) -> tuple[int | None, str | None]:
    """校验可选整数参数，返回 (值, 错误消息)。"""
    if val is None:
        return None, None
    if not isinstance(val, int) or val < 0:
        return None, f"参数 {name} 必须为非负整数，实际值：{val!r}"
    return val, None


def _parse_optional_float(val: float | None, name: str) -> tuple[float | None, str | None]:
    """校验可选浮点参数，返回 (值, 错误消息)。"""
    if val is None:
        return None, None
    try:
        return float(val), None
    except (TypeError, ValueError):
        return None, f"参数 {name} 必须为数值，实际值：{val!r}"


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def _impl_query_simulation_db(
    db_path: str,
    mode: str,
    status: str | None,
    session_id: str | None,
    tags_str: str | None,
    iteration_min: int | None,
    iteration_max: int | None,
    objective_name: str | None,
    objective_min: float | None,
    objective_max: float | None,
    order_desc: bool,
    limit: int,
    offset: int,
    case_id: str | None,
    include_unsuccessful: bool = False,
) -> str:
    """query_simulation_db_tool 的核心实现，出错时返回 '错误：' 字符串。"""
    # 0. 严格校验 mode（早期拒绝，避免静默退回默认行为）
    if mode not in _VALID_MODES:
        return (
            f"错误：mode={mode!r} 不是合法值。"
            f"可选值：{sorted(_VALID_MODES)}。"
        )

    # 0b. limit 硬上限校验（get_case 模式不受限）
    if mode != "get_case":
        if limit <= 0:
            return (
                f"错误：limit={limit} 不合法，必须在 1 到 {_MAX_LIMIT} 之间。"
                "请缩小过滤条件后再查询，或传入具体 limit 值（建议 ≤50）。"
            )
        if limit > _MAX_LIMIT:
            return (
                f"错误：limit={limit} 超过硬上限 {_MAX_LIMIT}。"
                "请缩小过滤条件或分页（offset）查询。"
            )

    # 1. 解析数据库路径
    try:
        resolved_path = _resolve_db_path(db_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    # 2. 打开数据库
    try:
        from src.database.simulation_db import SimulationDB
    except ImportError as exc:
        return f"错误：无法导入 SimulationDB — {exc}"

    try:
        db = SimulationDB(resolved_path)
    except Exception as exc:
        return f"错误：无法打开数据库 [{type(exc).__name__}] — {exc}"

    try:
        total = db.count()
        db_path_str = str(resolved_path)

        # ---------- 模式：get_case（按 case_id 精确查询） ----------
        if mode == "get_case":
            if not case_id or not case_id.strip():
                return "错误：mode='get_case' 时必须提供 case_id 参数。"
            row = db.get_case(case_id.strip())
            if row is None:
                return f"查询结果：数据库中不存在 case_id='{case_id}'。\n数据库总记录数：{total}"
            return _fmt_get_case_report(db_path_str, row)

        # ---------- 模式：by_objective（按目标函数值排序） ----------
        if mode == "by_objective":
            if not objective_name or not objective_name.strip():
                return "错误：mode='by_objective' 时必须提供 objective_name 参数。"

            obj_min, err = _parse_optional_float(objective_min, "objective_min")
            if err:
                return f"错误：{err}"
            obj_max, err = _parse_optional_float(objective_max, "objective_max")
            if err:
                return f"错误：{err}"

            success_only = not include_unsuccessful
            try:
                rows = db.query_by_objective(
                    objective_name.strip(),
                    min_value=obj_min,
                    max_value=obj_max,
                    order_desc=order_desc,
                    limit=limit,
                    success_only=success_only,
                )
            except Exception as exc:
                return f"错误：query_by_objective 执行失败 [{type(exc).__name__}] — {exc}"

            return _fmt_query_objective_report(
                db_path_str, total, rows, objective_name.strip(), order_desc,
                success_only=success_only,
            )

        # ---------- 默认模式：query（多条件过滤） ----------
        iter_min, err = _parse_optional_int(iteration_min, "iteration_min")
        if err:
            return f"错误：{err}"
        iter_max, err = _parse_optional_int(iteration_max, "iteration_max")
        if err:
            return f"错误：{err}"

        tags = _parse_tags_str(tags_str)
        # limit 已在开头通过硬上限校验，此处直接使用
        off = max(0, offset)

        # 构造查询描述（用于报告标题）
        parts: list[str] = []
        if status:
            parts.append(f"status={status!r}")
        if session_id:
            parts.append(f"session_id={session_id!r}")
        if tags:
            parts.append(f"tags={tags}")
        if iter_min is not None:
            parts.append(f"iteration>={iter_min}")
        if iter_max is not None:
            parts.append(f"iteration<={iter_max}")
        query_desc = (", ".join(parts) or "全量查询") + f"  limit={limit}"

        try:
            rows = db.query_cases(
                status=status or None,
                session_id=session_id or None,
                tags=tags if tags else None,
                iteration_min=iter_min,
                iteration_max=iter_max,
                limit=limit,
                offset=off,
            )
        except Exception as exc:
            return f"错误：query_cases 执行失败 [{type(exc).__name__}] — {exc}"

        return _fmt_query_cases_report(db_path_str, total, rows, query_desc)

    finally:
        db.close()


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def query_simulation_db_tool(
    db_path: str,
    mode: str = "query",
    status: str = "",
    session_id: str = "",
    tags: str = "",
    iteration_min: int = -1,
    iteration_max: int = -1,
    objective_name: str = "",
    objective_min: float = float("nan"),
    objective_max: float = float("nan"),
    order_desc: bool = True,
    limit: int = 20,
    offset: int = 0,
    case_id: str = "",
    include_unsuccessful: bool = False,
) -> str:
    """查询 SimulationDB 历史仿真记录，无需重跑 Aspen Plus。

    支持三种查询模式（通过 mode 参数切换）：

    **mode='query'**（默认）：多条件过滤，按迭代轮次升序返回摘要列表。
      可组合：status、session_id、tags、iteration_min/max、limit、offset。

    **mode='by_objective'**：按目标函数值排序，找最优/最差工况。
      必填：objective_name。可选：objective_min/max、order_desc、limit、include_unsuccessful。
      默认仅返回 success=True 的工况（过滤不可行/失败点），可通过 include_unsuccessful=True 查看全量。

    **mode='get_case'**：按 case_id 精确查询，返回含 blocks/streams 的完整记录。
      必填：case_id。

    Args:
        db_path: SQLite 数据库文件路径（相对于项目根目录或绝对路径）。
            典型路径：``cases/demo_case/output/simulation.db``。
        mode: 查询模式，可选 ``"query"``（默认）、``"by_objective"``、``"get_case"``。
            非法值直接返回错误，不会静默降级。
        status: 按运行状态过滤（mode='query' 有效）。
            常见值：``"success"``、``"sim_failed"``、``"infeasible"``、
            ``"objective_error"``、``"pending"``。不传时不过滤。
        session_id: 按优化 session 过滤（mode='query' 有效）。不传时不过滤。
        tags: 逗号分隔的标签列表（mode='query' 有效），AND 语义。
            示例：``"agent_run_case"``，``"phase0,bayesian"``。不传时不过滤。
        iteration_min: 迭代轮次下界（含，mode='query' 有效）。-1 表示不限。
        iteration_max: 迭代轮次上界（含，mode='query' 有效）。-1 表示不限。
        objective_name: 目标函数名称（mode='by_objective' 必填），大小写敏感。
            示例：``"TAC"``、``"ADN_FRAC"``。
        objective_min: 目标函数值下界（含，mode='by_objective' 有效）。不传时不限。
        objective_max: 目标函数值上界（含，mode='by_objective' 有效）。不传时不限。
        order_desc: 排序方向（mode='by_objective' 有效）。
            ``True``（默认）降序；``False`` 升序（最小化目标用此选项）。
        limit: 最多返回条数（mode='query' 和 'by_objective' 有效）。
            默认 20，最大 200（硬上限）。<=0 或超过硬上限时返回错误。
        offset: 跳过前 N 条（分页，mode='query' 有效）。默认 0。
        case_id: 精确查询的 case UUID（mode='get_case' 必填）。
        include_unsuccessful: 仅 mode='by_objective' 有效。
            ``False``（默认）只返回 success=True 的工况，避免不可行/失败点混入"最优"排名。
            ``True`` 时包含全部工况（含 infeasible / sim_failed）。

    Returns:
        格式化的查询报告文本，包含统计摘要和记录列表。
        出错时返回以 "错误：" 开头的描述字符串。
    """
    import math

    # 将哨兵值还原为 None（LangChain 工具层不接受 Optional 参数，用哨兵代替）
    status_val = status.strip() or None
    session_id_val = session_id.strip() or None
    tags_val = tags.strip() or None
    objective_name_val = objective_name.strip() or None
    case_id_val = case_id.strip() or None
    iter_min_val = None if iteration_min < 0 else iteration_min
    iter_max_val = None if iteration_max < 0 else iteration_max
    obj_min_val = None if math.isnan(objective_min) else objective_min
    obj_max_val = None if math.isnan(objective_max) else objective_max

    return _impl_query_simulation_db(
        db_path=db_path,
        mode=mode,
        status=status_val,
        session_id=session_id_val,
        tags_str=tags_val,
        iteration_min=iter_min_val,
        iteration_max=iter_max_val,
        objective_name=objective_name_val,
        objective_min=obj_min_val,
        objective_max=obj_max_val,
        order_desc=order_desc,
        limit=limit,
        offset=offset,
        case_id=case_id_val,
        include_unsuccessful=include_unsuccessful,
    )
