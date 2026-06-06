"""
summarize_pareto.py — summarize_pareto_tool 实现。

功能：从已有 SimulationDB 读取历史工况，计算 Pareto 前沿和超体积指标，
      可选计算设计变量敏感性分析，返回结构化摘要报告。

不依赖 Aspen COM，可在任意环境中安全调用。
与 optimize_pareto_tool 的区别：
  optimize_pareto_tool  — 驱动完整优化循环，需要 Aspen COM
  summarize_pareto_tool — 事后分析历史数据，纯数据库查询
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_log = logging.getLogger(__name__)

# 单次查询工况总数的硬上限，防止超大数据集导致 Pareto 计算时间过长
_MAX_CASES: int = 1000


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
# ProcessCase 重建（从 DB dict）
# ---------------------------------------------------------------------------

def _dict_to_process_case(row: dict[str, Any]) -> Any:
    """
    从 SimulationDB 摘要行重建 ProcessCase，仅填充 Pareto 计算所需字段。

    不填充 blocks / streams / sim_result / semantic_blocks，
    避免不必要的内存开销（Pareto 计算只使用目标函数值和设计变量）。

    Parameters
    ----------
    row:
        SimulationDB.query_cases() 返回的单行 dict。

    Returns
    -------
    ProcessCase 实例，status 已映射为对应 CaseStatus 枚举。
    """
    from src.models.process_case import (
        CaseStatus, ObjectiveValue, ConstraintValue, ProcessCase,
    )

    # 映射 status 字符串到 CaseStatus 枚举，未知值降级为 SIM_FAILED
    status_str = row.get("status", "sim_failed")
    try:
        status = CaseStatus(status_str)
    except ValueError:
        status = CaseStatus.SIM_FAILED

    # 重建目标函数列表
    objectives: list[ObjectiveValue] = []
    for obj_dict in (row.get("objectives") or []):
        objectives.append(ObjectiveValue(
            name=obj_dict.get("name", "?"),
            value=obj_dict.get("value"),
            unit=obj_dict.get("unit", ""),
            minimize=bool(obj_dict.get("minimize", True)),
            error=obj_dict.get("error"),
        ))

    # 重建约束列表
    constraints: list[ConstraintValue] = []
    for con_dict in (row.get("constraints") or []):
        constraints.append(ConstraintValue(
            name=con_dict.get("name", "?"),
            value=con_dict.get("value"),
            satisfied=con_dict.get("satisfied"),
            error=con_dict.get("error"),
        ))

    return ProcessCase(
        case_id=row.get("case_id", ""),
        iteration=row.get("iteration", 0),
        status=status,
        design_vars=dict(row.get("design_vars") or {}),
        objectives=objectives,
        constraints=constraints,
        tags=list(row.get("tags") or []),
        notes=row.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# 格式化辅助函数
# ---------------------------------------------------------------------------

def _fmt_value(value: Any) -> str:
    """将数值格式化为紧凑字符串。"""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_path_tail(path: str, tail: int = 3) -> str:
    """截取 Aspen 路径末尾 N 段，用于紧凑显示。"""
    if not path:
        return "<empty>"
    parts = [p for p in path.replace("/", "\\").split("\\") if p]
    if len(parts) > tail:
        parts = parts[-tail:]
    return "\\".join(parts) if parts else path


def _fmt_pareto_front_section(
    pareto_result: Any,
    objective_names: list[str],
    include_all_fronts: bool,
    max_front_display: int,
) -> list[str]:
    """格式化 Pareto 前沿区段，返回行列表。"""
    lines: list[str] = []
    first_front = pareto_result.first_front
    n_fronts = pareto_result.n_fronts

    if first_front is None or not first_front.cases:
        lines.append("【Pareto 前沿】")
        lines.append("  (无有效 Pareto 前沿，可能所有工况均失败)")
        lines.append("")
        return lines

    front_size = len(first_front.cases)
    lines.append(f"【Pareto 前沿（第一前沿，共 {front_size} 个解，按拥挤距离降序）】")

    triples = list(zip(
        first_front.cases,
        first_front.objective_vectors,
        first_front.crowding_distances,
    ))
    # 边界解（cd=inf）排前面；非边界按拥挤距离降序
    triples.sort(key=lambda t: t[2] if t[2] != math.inf else 1e18, reverse=True)

    display = triples[:max_front_display]
    for i, (case, vec, cd) in enumerate(display, 1):
        obj_str = "  ".join(f"{n}={_fmt_value(v)}" for n, v in zip(objective_names, vec))
        # 显示前 4 个设计变量
        dv_items = list(case.design_vars.items())[:4]
        dv_str = "  ".join(f"{_fmt_path_tail(k)}={_fmt_value(v)}" for k, v in dv_items)
        cd_str = "∞" if cd == math.inf else f"{cd:.3f}"
        line = f"  [{i:3d}] {obj_str}"
        if dv_str:
            line += f"  |  {dv_str}"
        line += f"  |  cd={cd_str}"
        lines.append(line)

    if front_size > max_front_display:
        lines.append(f"  ... 共 {front_size} 个解，仅显示前 {max_front_display} 个")
    lines.append("")

    # 其他前沿摘要
    if n_fronts > 1:
        if include_all_fronts:
            lines.append("【其他 Pareto 层】")
            for front in pareto_result.fronts[1:]:
                lines.append(f"  第 {front.rank + 1} 层: {len(front.cases)} 个解")
        else:
            other_sizes = [len(f.cases) for f in pareto_result.fronts[1:]]
            summary = "  ".join(f"第{r+2}层:{s}个" for r, s in enumerate(other_sizes[:5]))
            if len(other_sizes) > 5:
                summary += f"  ... 共{n_fronts}层"
            lines.append(f"  其他 Pareto 层（共 {n_fronts - 1} 层）: {summary}")
        lines.append("")

    return lines


def _fmt_hv_section(pareto_result: Any, objective_names: list[str]) -> list[str]:
    """格式化超体积区段。"""
    lines: list[str] = ["【超体积（HV）】"]
    hv = pareto_result.hypervolume
    ref_pt = pareto_result.reference_point

    if hv is not None:
        lines.append(f"  最终 HV : {hv:.6g}")
    else:
        lines.append("  最终 HV : N/A（有效工况不足，无法计算）")

    if ref_pt is not None:
        ref_str = "[" + ", ".join(_fmt_value(v) for v in ref_pt) + "]"
        lines.append(f"  参考点  : {ref_str}（原始值，已还原到各目标原始方向）")
        lines.append("  参考说明: 由最小化方向自动推断后还原（最大化目标已取反再还原）")

    lines.append("")
    return lines


def _fmt_sensitivity_section(
    cases: list[Any],
    objective_names: list[str],
    method: str,
) -> list[str]:
    """格式化变量敏感性分析区段。"""
    lines: list[str] = ["【设计变量敏感性分析】"]

    # 收集参与分析的工况（仅 simulation_valid=True 的工况有可信设计变量）
    valid_cases = [c for c in cases if c.simulation_valid]
    if not valid_cases:
        lines.append("  (无 simulation_valid=True 的工况，无法进行敏感性分析)")
        lines.append("")
        return lines

    # 提取设计变量路径：取所有 simulation_valid 工况的 design_vars 键的 union，
    # 同时统计每个变量的覆盖率（出现在几个工况中），用于标注证据不足的变量。
    param_path_counts: dict[str, int] = {}
    for case in valid_cases:
        for path in (case.design_vars or {}):
            param_path_counts[path] = param_path_counts.get(path, 0) + 1
    param_paths = list(param_path_counts.keys())

    if not param_paths:
        lines.append("  (工况中无设计变量记录，跳过敏感性分析)")
        lines.append("")
        return lines

    try:
        from src.optimization.metrics import sensitivity_analysis, rank_variables
    except ImportError as exc:
        lines.append(f"  (无法导入 metrics 模块，跳过敏感性分析：{exc})")
        lines.append("")
        return lines

    try:
        sens_result = sensitivity_analysis(
            cases=valid_cases,
            param_paths=param_paths,
            objective_names=objective_names,
            method=method,
        )
    except Exception as exc:
        lines.append(f"  (敏感性分析失败：{exc})")
        lines.append("")
        return lines

    ranked = rank_variables(sens_result)

    lines.append(f"  分析样本数 : {sens_result.n_samples}")
    lines.append(f"  变量总数   : {len(param_paths)}")
    lines.append(f"  方法       : {sens_result.method}")
    if sens_result.warnings:
        for w in sens_result.warnings[:3]:
            lines.append(f"  注意: {w}")
    lines.append("")
    lines.append("  变量重要性排序（高 → 低）：")

    n_valid = len(valid_cases)
    for param_path, score in ranked:
        short_name = _fmt_path_tail(param_path)
        reliable = sens_result.is_reliable(param_path)
        reliable_str = "可靠" if reliable else "证据不足"

        # 覆盖率：该变量出现在几个工况中（针对 union 后可能存在的稀疏变量）
        coverage = param_path_counts.get(param_path, 0)
        cov_str = f"覆盖{coverage}/{n_valid}" if coverage < n_valid else ""

        # 各目标维度的分数
        per_obj_parts = []
        for obj_name in objective_names:
            try:
                obj_score = sens_result.score(param_path, obj_name)
                per_obj_parts.append(f"{obj_name}={obj_score:.2f}")
            except (KeyError, IndexError):
                pass
        per_obj_str = ", ".join(per_obj_parts)

        line = f"    {short_name:<20} : {score:.3f}  [{reliable_str}]"
        if cov_str:
            line += f"  [{cov_str}]"
        if per_obj_str:
            line += f"  ({per_obj_str})"
        lines.append(line)

    # 低敏感性变量建议
    low_sens = [(p, s) for p, s in ranked if s < 0.10 and sens_result.is_reliable(p)]
    if low_sens:
        names = [_fmt_path_tail(p) for p, _ in low_sens]
        lines.append("")
        lines.append(f"  建议固定的低敏感性变量（score < 0.10）: {', '.join(names)}")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 主报告格式化函数
# ---------------------------------------------------------------------------

def _fmt_summarize_pareto_report(
    db_path: str,
    pareto_result: Any,
    cases: list[Any],
    objective_names: list[str],
    query_desc: str,
    include_sensitivity: bool,
    sensitivity_method: str,
    include_all_fronts: bool,
    max_front_display: int,
    include_infeasible: bool = False,
) -> str:
    """将 ParetoResult 和原始工况列表格式化为完整的摘要报告。"""
    lines: list[str] = ["=== summarize_pareto 历史 Pareto 分析报告 ===", ""]

    lines.append(f"数据库   : {db_path}")
    lines.append(f"查询条件 : {query_desc}")
    if include_infeasible:
        lines.append("模式     : 含不可行解（include_infeasible=True，仅用于诊断）")
    else:
        lines.append("模式     : 正式 Pareto（已排除 infeasible 工况，仅含 success=True）")
    lines.append("")

    # ---- 数据概况 ----
    n_total = len(cases)
    n_evaluated = pareto_result.n_evaluated
    n_excluded = len(pareto_result.excluded_cases)
    n_success = sum(1 for c in cases if c.success)
    n_failed = sum(1 for c in cases if c.status.value == "sim_failed")
    n_infeasible = sum(1 for c in cases if c.status.value == "infeasible")
    obj_desc = "  ".join(
        f"{n}({'最小化' if _get_minimize(cases, n) else '最大化'})"
        for n in objective_names
    )
    if include_infeasible:
        pareto_desc = "success=True + infeasible 且目标可用"
        excluded_desc = "仿真失败/目标不可用"
    else:
        pareto_desc = "success=True"
        excluded_desc = "失败/不可行/目标不可用"

    lines.append("【数据概况】")
    lines.append(f"  查询工况总数  : {n_total}")
    lines.append(f"  参与 Pareto   : {n_evaluated}（{pareto_desc}）")
    lines.append(f"  排除工况      : {n_excluded}（{excluded_desc}）")
    lines.append(f"    仿真失败     : {n_failed}")
    if not include_infeasible:
        lines.append(f"    不可行       : {n_infeasible}")
    if n_total > 0:
        lines.append(f"  成功率        : {n_success / n_total * 100:.1f}%")
    lines.append(f"  目标函数      : {obj_desc}")
    lines.append("")

    # ---- Pareto 前沿 ----
    lines.extend(_fmt_pareto_front_section(
        pareto_result, objective_names, include_all_fronts, max_front_display,
    ))

    # ---- 超体积 ----
    lines.extend(_fmt_hv_section(pareto_result, objective_names))

    # ---- 敏感性分析 ----
    if include_sensitivity:
        lines.extend(_fmt_sensitivity_section(cases, objective_names, sensitivity_method))

    # ---- 综合结论 ----
    lines.append("【综合结论】")
    first_front = pareto_result.first_front
    front_size = len(first_front.cases) if first_front else 0
    hv = pareto_result.hypervolume

    if n_evaluated == 0:
        lines.append("  [失败] 无有效工况参与 Pareto 计算，无法生成前沿。")
        lines.append("  建议：检查目标函数名称是否正确，或放宽查询过滤条件。")
    elif front_size == 0:
        lines.append("  [部分完成] 有有效工况但 Pareto 前沿计算失败。")
    else:
        hv_str = f"HV = {hv:.6g}" if hv is not None else "HV 未计算"
        lines.append(
            f"  [完成] 第一 Pareto 前沿 {front_size} 个解，"
            f"{hv_str}，共 {pareto_result.n_fronts} 层前沿。"
        )
        if n_total > 0 and n_success / n_total < 0.5:
            lines.append(
                f"  注意：成功率仅 {n_success / n_total * 100:.1f}%，"
                "建议检查仿真配置或放宽约束。"
            )
    lines.append("")
    return "\n".join(lines)


def _get_minimize(cases: list[Any], objective_name: str) -> bool:
    """从工况列表中查找目标函数的最小化方向，默认返回 True。"""
    for case in cases:
        for obj in (case.objectives or []):
            if obj.name == objective_name:
                return obj.minimize
    return True


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def _impl_summarize_pareto(
    db_path: str,
    objective_names_str: str,
    session_id: str | None,
    iteration_min: int | None,
    iteration_max: int | None,
    tags_str: str | None,
    include_sensitivity: bool,
    sensitivity_method: str,
    include_all_fronts: bool,
    max_front_display: int,
    include_infeasible: bool,
) -> str:
    """summarize_pareto_tool 的核心实现，出错时返回 '错误：' 字符串。"""
    # 1. 参数校验
    if not objective_names_str or not objective_names_str.strip():
        return "错误：objective_names 不能为空，请提供目标函数名称（逗号分隔）。"

    objective_names = [n.strip() for n in objective_names_str.split(",") if n.strip()]
    if not objective_names:
        return "错误：objective_names 解析后为空列表，请检查格式（逗号分隔的名称）。"

    # 重复目标名会导致维度崩塌，直接拒绝
    if len(objective_names) != len(set(objective_names)):
        duplicates = [n for n in objective_names if objective_names.count(n) > 1]
        return (
            f"错误：objective_names 中存在重复目标名：{list(set(duplicates))}。"
            "Pareto 计算要求每个目标名唯一，请去除重复项。"
        )

    if max_front_display <= 0 or max_front_display > 200:
        return f"错误：max_front_display={max_front_display} 不合法，必须在 1 到 200 之间。"

    if sensitivity_method not in ("spearman", "variance"):
        return (
            f"错误：sensitivity_method={sensitivity_method!r} 不合法，"
            "可选值：'spearman'、'variance'。"
        )

    # 2. 解析数据库路径
    try:
        resolved_path = _resolve_db_path(db_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    # 3. 导入数据库
    try:
        from src.database.simulation_db import SimulationDB
    except ImportError as exc:
        return f"错误：无法导入 SimulationDB — {exc}"

    try:
        db = SimulationDB(resolved_path)
    except Exception as exc:
        return f"错误：无法打开数据库 [{type(exc).__name__}] — {exc}"

    db_path_str = str(resolved_path)

    # 4. 查询工况
    try:
        tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
        rows = db.query_cases(
            session_id=session_id or None,
            iteration_min=iteration_min,
            iteration_max=iteration_max,
            tags=tags_list,
            limit=_MAX_CASES,
            offset=0,
        )
    except Exception as exc:
        return f"错误：query_cases 执行失败 [{type(exc).__name__}] — {exc}"
    finally:
        db.close()

    if not rows:
        return (
            "查询结果为空：数据库中没有符合条件的工况记录。\n"
            "请检查查询过滤条件（session_id、iteration 范围、tags）或确认数据库路径正确。"
        )

    # 5. 重建 ProcessCase 对象
    try:
        cases = [_dict_to_process_case(row) for row in rows]
    except Exception as exc:
        _log.exception("重建 ProcessCase 时出现意外错误")
        return f"错误：重建 ProcessCase 失败 [{type(exc).__name__}] — {exc}"

    # 5b. 检查目标名是否存在于查询结果中（完全不存在时报错，避免 agent 误判为"无有效工况"）
    names_found: set[str] = set()
    for row in rows:
        for obj_dict in (row.get("objectives") or []):
            names_found.add(obj_dict.get("name", ""))
    missing = [n for n in objective_names if n not in names_found]
    if missing:
        available = sorted(names_found - {""})
        return (
            f"错误：目标名 {missing} 在数据库工况中未找到。"
            f"数据库中存在的目标名：{available}。"
            "请检查 objective_names 参数的拼写和大小写。"
        )

    # 6. 计算 Pareto 前沿
    try:
        from src.optimization.pareto import compute_pareto
    except ImportError as exc:
        return f"错误：无法导入 compute_pareto — {exc}"

    try:
        pareto_result = compute_pareto(
            cases=cases,
            objective_names=objective_names,
            compute_hv=True,
            include_infeasible=include_infeasible,
        )
    except ValueError as exc:
        return f"错误：Pareto 计算失败 — {exc}"
    except Exception as exc:
        _log.exception("Pareto 计算时出现意外错误")
        return f"错误：Pareto 计算时出现意外错误 [{type(exc).__name__}] — {exc}"

    # 7. 构建查询描述
    query_parts: list[str] = []
    if session_id:
        query_parts.append(f"session={session_id!r}")
    if iteration_min is not None:
        query_parts.append(f"iter>={iteration_min}")
    if iteration_max is not None:
        query_parts.append(f"iter<={iteration_max}")
    if tags_list:
        query_parts.append(f"tags={tags_list}")
    if len(rows) >= _MAX_CASES:
        query_parts.append(f"[已截断至 {_MAX_CASES} 条]")
    query_desc = "、".join(query_parts) if query_parts else "全库"

    # 8. 格式化报告
    try:
        report = _fmt_summarize_pareto_report(
            db_path=db_path_str,
            pareto_result=pareto_result,
            cases=cases,
            objective_names=objective_names,
            query_desc=query_desc,
            include_sensitivity=include_sensitivity,
            sensitivity_method=sensitivity_method,
            include_all_fronts=include_all_fronts,
            max_front_display=max_front_display,
            include_infeasible=include_infeasible,
        )
    except Exception as exc:
        _log.exception("格式化 summarize_pareto 报告时出现意外错误")
        return f"错误：格式化报告时出现意外错误 — {exc}"

    _log.info(
        "summarize_pareto_tool: 完成 objectives=%s n_cases=%d "
        "n_evaluated=%d front_size=%d HV=%s",
        objective_names,
        len(cases),
        pareto_result.n_evaluated,
        len(pareto_result.first_front.cases) if pareto_result.first_front else 0,
        f"{pareto_result.hypervolume:.4g}" if pareto_result.hypervolume is not None else "N/A",
    )
    return report


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def summarize_pareto_tool(
    db_path: str,
    objective_names: str,
    session_id: str = "",
    iteration_min: int = -1,
    iteration_max: int = -1,
    tags: str = "",
    include_sensitivity: bool = True,
    sensitivity_method: str = "spearman",
    include_all_fronts: bool = False,
    max_front_display: int = 15,
    include_infeasible: bool = False,
) -> str:
    """从 SimulationDB 读取历史工况，计算 Pareto 前沿和超体积，返回摘要报告。

    不需要连接 Aspen Plus，可在任意环境中调用。适用于：
      - 优化完成后的结果汇总
      - 部分运行后的中间检查
      - 多次运行结果的联合分析

    与 optimize_pareto_tool 的区别：
      optimize_pareto_tool  — 驱动完整优化循环，需要 Aspen COM
      summarize_pareto_tool — 事后分析历史数据，不需要 Aspen COM

    Args:
        db_path: SimulationDB SQLite 文件路径（相对于项目根目录或绝对路径）。
            典型路径：``cases/demo_case/output/simulation.db``。
        objective_names: 目标函数名称，逗号分隔（必填）。
            名称须与数据库中的 ObjectiveValue.name 完全一致，区分大小写。
            顺序决定超体积计算的维度顺序。
            示例：``"TAC"``（单目标仅看分布）、``"TAC,ADN_FRAC"``（双目标）。
        session_id: 仅分析指定 session 的工况（可选）。
            不传时分析数据库中所有工况，适合联合分析多次运行的结果。
        iteration_min: 迭代轮次下界（含，-1 表示不限）。
        iteration_max: 迭代轮次上界（含，-1 表示不限）。
        tags: 逗号分隔的标签过滤（AND 语义，空字符串不过滤）。
            示例：``"bayesian"``、``"phase1,exploitation"``。
        include_sensitivity: 是否计算设计变量敏感性分析（默认 True）。
            分析样本不足时会在报告中给出警告，不会报错。
        sensitivity_method: 敏感性分析方法（默认 "spearman"）。
            ``"spearman"``：秩相关，适合样本数 >= 5。
            ``"variance"``：方差贡献估计，适合样本数 >= 20。
        include_all_fronts: 是否在报告中展示所有 Pareto 层（默认 False）。
            False 时只展示第一前沿，其他层以摘要行显示。
        max_front_display: 第一前沿最多展示的解数（默认 15，最大 200）。
        include_infeasible: 是否将 infeasible 工况纳入 Pareto 计算（默认 False）。
            False（默认）：仅 success=True 的工况参与前沿计算，适合正式结果汇总。
            True：将 infeasible 工况也纳入，适合诊断模式（报告会注明此模式）。

    Returns:
        格式化的 Pareto 分析报告，包含数据概况、前沿解集、超体积指标、
        设计变量敏感性和综合结论。出错时返回以 "错误：" 开头的字符串。
    """
    return _impl_summarize_pareto(
        db_path=db_path,
        objective_names_str=objective_names,
        session_id=session_id.strip() or None,
        iteration_min=None if iteration_min < 0 else iteration_min,
        iteration_max=None if iteration_max < 0 else iteration_max,
        tags_str=tags.strip() or None,
        include_sensitivity=include_sensitivity,
        sensitivity_method=sensitivity_method,
        include_all_fronts=include_all_fronts,
        max_front_display=max_front_display,
        include_infeasible=include_infeasible,
    )