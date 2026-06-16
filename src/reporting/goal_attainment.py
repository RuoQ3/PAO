"""
goal_attainment.py — 目标达成度评估模块（H2）。

职责
----
对照用户提供的 OptimizationIntent（goals + hard_constraints）与优化产出
（SimulationDB 中的 Pareto 前沿），评估"是否/多大程度达成了初始目标"，
生成可并入 summary_report 的达成度 Markdown 章节。

报告逻辑
--------
1. 对 hard_constraints：从 Pareto 前沿工况的 constraints 字段读取 satisfied 判定
2. 对 goals：从 Pareto 前沿工况的 objectives 字段取最优值，可选与外部基线比较
3. 顶部给一句总判定："X/Y 约束满足；主目标最优值 Z"

约束评估契约
-----------
ProcessCase.ConstraintValue 已将约束标准化为 value <= 0（满足），satisfied 是
优化层写入 DB 时的可信判定，不能绕过。evaluate_constraints() 优先从
constraints_json 读取 satisfied；只有约束名未匹配时才降级为"数据缺失"。

基线契约
--------
establish_baseline() 只返回调用方显式传入的外部基线字典，不从 DB 拉随机点。
在优化前单跑初始参数是获取真实基线的唯一正确方式，当前阶段不支持则
improvement 严格为 None，报告如实说明。

设计约束
--------
- 纯读 SimulationDB，不调用 Aspen COM
- 数据缺失时降级（satisfied=None / improvement=None），不伪装成功
- _load_pareto_best 返回结构化结果，降级时报告标注来源
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# H2-1  GoalAttainment dataclass
# ---------------------------------------------------------------------------

@dataclass
class GoalAttainment:
    """
    单个目标/约束的达成度记录。

    Attributes
    ----------
    metric:
        目标/约束的指标类型字符串（来自 GoalSpec.metric）。
    name:
        目标/约束在优化配置中使用的名称。
    direction:
        优化方向："min" 或 "max"（目标）；约束不使用此字段做判定。
    target_value:
        约束阈值（原始物理值，仅供展示）；纯目标时为 None。
    achieved_value:
        Pareto 前沿最优点的实际目标值；约束时为标准化约束值（value <= 0=满足）。
    baseline_value:
        外部传入基线的对应值；未提供时为 None。
    satisfied:
        约束是否满足（来自 ConstraintValue.satisfied）；目标型为 None。
        True=满足, False=未满足, None=数据缺失无法判断。
    improvement:
        相对 baseline 的改善幅度（%）；baseline 缺失时为 None。
    note:
        补充说明（匹配方式、数据缺失原因等）。
    """
    metric: str
    name: str
    direction: str
    target_value: float | None = None
    achieved_value: float | None = None
    baseline_value: float | None = None
    satisfied: bool | None = None
    improvement: float | None = None
    note: str = ""


# ---------------------------------------------------------------------------
# 结构化 Pareto 加载结果
# ---------------------------------------------------------------------------

class _ParetoRows(NamedTuple):
    rows: list[dict[str, Any]]
    source: str   # "pareto_front" | "all_success" | "empty"
    warning: str  # 非空时表示发生了降级，应在报告中标注


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _open_db(db_path: str) -> Any:
    from src.database.simulation_db import SimulationDB
    return SimulationDB(db_path)


def _load_pareto_best(db_path: str, session_id: str | None) -> _ParetoRows:
    """
    从数据库加载 Pareto 第一前沿工况列表（结构化返回）。

    Returns
    -------
    _ParetoRows
        .rows   — 工况 dict 列表
        .source — "pareto_front" / "all_success" / "empty"
        .warning — 非空时说明发生了降级
    """
    with _open_db(db_path) as db:
        all_rows = db.query_cases(session_id=session_id)

    if not all_rows:
        return _ParetoRows(rows=[], source="empty", warning="")

    try:
        from src.agents.tools.summarize_pareto import _dict_to_process_case
        from src.optimization.pareto import compute_pareto

        cases = [_dict_to_process_case(r) for r in all_rows]

        obj_names: list[str] = []
        for r in all_rows:
            objs = r.get("objectives") or []
            if objs:
                obj_names = [o.get("name", "") for o in objs if o.get("name")]
                break

        if not obj_names:
            success_rows = [r for r in all_rows if r.get("status") == "success"]
            return _ParetoRows(
                rows=success_rows,
                source="all_success",
                warning="未检测到目标函数名称，已降级为所有成功工况",
            )

        pareto_result = compute_pareto(cases, obj_names, compute_hv=False)
        if pareto_result is None or pareto_result.first_front is None:
            success_rows = [r for r in all_rows if r.get("status") == "success"]
            return _ParetoRows(
                rows=success_rows,
                source="all_success",
                warning="Pareto 前沿计算失败，已降级为所有成功工况",
            )

        front_ids = {c.case_id for c in pareto_result.first_front.cases}
        front_rows = [r for r in all_rows if r.get("case_id") in front_ids]
        return _ParetoRows(rows=front_rows, source="pareto_front", warning="")

    except Exception as exc:
        _log.warning("Pareto 前沿计算失败，回退到 success 工况：%s", exc)
        success_rows = [r for r in all_rows if r.get("status") == "success"]
        return _ParetoRows(
            rows=success_rows,
            source="all_success",
            warning=f"Pareto 计算异常（{exc}），已降级为所有成功工况",
        )


def _find_best_objective_value(
    rows: list[dict[str, Any]],
    target_name: str,
    direction: str,
) -> tuple[float | None, str]:
    """
    在工况列表的 objectives 中按名称精确匹配，取最优值。

    Returns (value, note)
    """
    values: list[float] = []
    for row in rows:
        objs = row.get("objectives") or []
        for obj in objs:
            if obj.get("name", "").upper() == target_name.upper():
                v = obj.get("value")
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

    if not values:
        return None, f"objectives 中未找到名称 '{target_name}'"

    best = min(values) if direction == "min" else max(values)
    return best, f"精确匹配 objectives['{target_name}']，共 {len(values)} 个有效点"


def _find_best_objective_by_metric(
    rows: list[dict[str, Any]],
    metric: str,
    direction: str,
) -> tuple[float | None, str]:
    """按 metric 关键词模糊匹配 objectives 列名。"""
    metric_keywords: dict[str, list[str]] = {
        "TAC":       ["tac", "annualized_cost", "annual_cost"],
        "emissions": ["emission", "co2", "carbon"],
        "flow":      ["flow", "massflow", "moleflow"],
        "yield":     ["yield", "recovery"],
        "purity":    ["purity", "massfrac", "molefrac"],
        "custom":    [],
    }
    keywords = metric_keywords.get(metric, [metric.lower()])
    if not keywords:
        return None, f"metric='{metric}' 无关键词规则，无法模糊匹配"

    for row in rows:
        objs = row.get("objectives") or []
        for obj in objs:
            name_lower = obj.get("name", "").lower()
            if any(kw in name_lower for kw in keywords):
                matched = obj.get("name", "")
                val, note = _find_best_objective_value(rows, matched, direction)
                if val is not None:
                    return val, f"metric='{metric}' 模糊匹配到 objectives['{matched}']；{note}"

    return None, f"objectives 中无匹配 metric='{metric}' 的列"


def _resolve_objective_value(
    rows: list[dict[str, Any]],
    goal_name: str | None,
    metric: str,
    direction: str,
) -> tuple[float | None, str]:
    """先精确匹配名称，再按 metric 模糊匹配。"""
    if goal_name:
        val, note = _find_best_objective_value(rows, goal_name, direction)
        if val is not None:
            return val, note
    return _find_best_objective_by_metric(rows, metric, direction)


def _compute_improvement(
    achieved: float | None,
    baseline: float | None,
    direction: str,
) -> float | None:
    """
    计算相对 baseline 的改善幅度（%，正值=向优化方向改善）。
    baseline=0 时无法计算，返回 None。
    """
    if achieved is None or baseline is None:
        return None
    if baseline == 0:
        return None
    if direction == "min":
        return (baseline - achieved) / abs(baseline) * 100
    return (achieved - baseline) / abs(baseline) * 100


# ---------------------------------------------------------------------------
# H2-2  evaluate_constraints — 从 DB constraints 字段读取 satisfied
# ---------------------------------------------------------------------------

def evaluate_constraints(
    intent: Any,
    db_path: str,
    session_id: str | None = None,
) -> list[GoalAttainment]:
    """
    评估 intent.hard_constraints 中每条约束是否满足。

    核心原则
    --------
    优先读 ProcessCase.constraints（constraints_json）中的 satisfied 字段——
    这是优化层在仿真后按 value <= 0 标准计算的可信判定。

    匹配策略：intent.hard_constraints[i].name 对 DB constraints[j].name 精确匹配
    （大小写不敏感）。匹配到时直接使用 satisfied；未匹配到时 satisfied=None 并说明原因。

    不在 objectives 中模糊搜索约束值——purity/yield 等约束已经标准化入 constraints，
    不应绕过 satisfied 逻辑重新做阈值比较。
    """
    constraints_spec = getattr(intent, "hard_constraints", []) or []
    if not constraints_spec:
        return []

    pareto = _load_pareto_best(db_path, session_id)
    rows = pareto.rows

    results: list[GoalAttainment] = []
    for spec in constraints_spec:
        metric    = getattr(spec, "metric", "custom")
        direction = getattr(spec, "direction", "max")
        target    = getattr(spec, "target_value", None)
        name      = getattr(spec, "name", None) or metric

        # 从 Pareto 前沿工况的 constraints 字段收集判定结果
        # 每个工况的约束可能有多个点，取"最优"语义：
        #   - 若任意一个工况满足该约束，则报告满足
        #   - 只有全部工况均违反时，才报告未满足
        #   - 全部缺失时 None
        satisfied_votes: list[bool] = []
        achieved_vals: list[float] = []
        matched_count = 0

        for row in rows:
            row_constraints = row.get("constraints") or []
            for c in row_constraints:
                c_name = c.get("name", "")
                if c_name.upper() == name.upper():
                    matched_count += 1
                    s = c.get("satisfied")
                    if s is not None:
                        satisfied_votes.append(bool(s))
                    v = c.get("value")
                    if v is not None:
                        try:
                            achieved_vals.append(float(v))
                        except (TypeError, ValueError):
                            pass
                    break  # 每个工况只取一次

        if matched_count == 0:
            # 名称未匹配到任何工况的约束列表
            note = (
                f"DB constraints 中未找到名称 '{name}'；"
                f"（来源：{pareto.source}，共 {len(rows)} 个工况）"
            )
            if pareto.warning:
                note += f"；{pareto.warning}"
            results.append(GoalAttainment(
                metric=metric, name=name, direction=direction,
                target_value=target, achieved_value=None,
                satisfied=None, note=note,
            ))
            continue

        # 综合判定：Pareto 前沿中只要有一个工况满足该约束，报告满足
        if satisfied_votes:
            satisfied = any(satisfied_votes)
        else:
            satisfied = None

        # 展示用：取标准化约束值（<= 0 为满足）的最小值（最"满足"的点）
        achieved = min(achieved_vals) if achieved_vals else None

        note = (
            f"从 DB constraints 字段读取，匹配 {matched_count}/{len(rows)} 个前沿工况"
        )
        if pareto.source != "pareto_front":
            note += f"；⚠ {pareto.warning}"
        if satisfied is None:
            note += "；satisfied 字段全部缺失"

        results.append(GoalAttainment(
            metric=metric, name=name, direction=direction,
            target_value=target,
            achieved_value=achieved,
            satisfied=satisfied,
            note=note,
        ))

    return results


# ---------------------------------------------------------------------------
# H2-3  evaluate_objectives
# ---------------------------------------------------------------------------

def evaluate_objectives(
    intent: Any,
    db_path: str,
    session_id: str | None = None,
    baseline: dict[str, float] | None = None,
) -> list[GoalAttainment]:
    """
    评估 intent.goals 中每条目标的达成情况。

    Parameters
    ----------
    baseline:
        外部传入的基线值 {目标名: 值}，来自优化前初始单跑结果。
        为 None 或键缺失时 improvement=None，报告如实说明"无真实基线"。
    """
    goals = getattr(intent, "goals", []) or []
    if not goals:
        return []

    pareto = _load_pareto_best(db_path, session_id)
    rows = pareto.rows

    results: list[GoalAttainment] = []
    for spec in goals:
        metric    = getattr(spec, "metric", "custom")
        direction = getattr(spec, "direction", "min")
        target    = getattr(spec, "target_value", None)
        name      = getattr(spec, "name", None) or metric

        achieved, match_note = _resolve_objective_value(rows, name, metric, direction)

        base_val: float | None = None
        if baseline:
            base_val = baseline.get(name)
            if base_val is None:
                base_val = baseline.get(metric)

        improvement = _compute_improvement(achieved, base_val, direction)

        if base_val is None:
            base_note = "无真实基线（未提供优化前初始单跑结果），改善幅度不可计算"
        elif improvement is None:
            base_note = "基线值为 0，改善百分比无意义"
        else:
            base_note = f"改善 {improvement:+.1f}%（相对基线 {base_val:.4g}）"

        note = match_note
        if pareto.source != "pareto_front":
            note += f"；⚠ {pareto.warning}"
        note += f"；{base_note}"

        results.append(GoalAttainment(
            metric=metric, name=name, direction=direction,
            target_value=target,
            achieved_value=achieved,
            baseline_value=base_val,
            improvement=improvement,
            note=note,
        ))

    return results


# ---------------------------------------------------------------------------
# H2-4  establish_baseline
# ---------------------------------------------------------------------------

def establish_baseline(
    baseline_dict: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    返回外部传入的基线值字典。

    真实基线 = 优化前用初始设计变量跑一次 run_case 得到的目标值。
    当前阶段不支持自动跑初始点，调用方负责在优化前获取并传入。
    未传入时严格返回 {}，不从 DB 拉随机点替代。

    Parameters
    ----------
    baseline_dict:
        {目标名: 目标值} 的字典，来自外部（如 run_phase1_e2e.py 中的初始单跑）。
        None 表示本次优化没有真实基线。

    Returns
    -------
    dict[str, float]
        原样返回 baseline_dict（None 时返回 {}）。
    """
    if not baseline_dict:
        _log.info("未提供真实基线，改善幅度将标记为 N/A（无法量化）")
        return {}
    return dict(baseline_dict)


# ---------------------------------------------------------------------------
# H2-5  generate_goal_attainment_section
# ---------------------------------------------------------------------------

def _fmt_val(v: float | None, precision: str = ".4g") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), precision)
    except (TypeError, ValueError):
        return str(v)


def _satisfied_icon(satisfied: bool | None) -> str:
    if satisfied is True:
        return "✅ 满足"
    if satisfied is False:
        return "❌ 未满足"
    return "❓ 数据缺失"


def generate_goal_attainment_section(
    intent: Any,
    db_path: str,
    config_path: str | None = None,
    session_id: str | None = None,
    baseline: dict[str, float] | None = None,
) -> str:
    """
    生成目标达成度 Markdown 章节（并入报告第 0 章）。

    Parameters
    ----------
    intent:
        OptimizationIntent 对象；为 None 时返回降级说明。
    db_path:
        SimulationDB 路径。
    config_path:
        预留参数（当前未使用）。
    session_id:
        会话 ID 过滤。
    baseline:
        外部传入的基线值字典 {目标名: 值}，来自优化前初始单跑。
        None 时报告标注"无真实基线，改善幅度不可计算"。
    """
    if intent is None:
        return "> ℹ 未提供优化意图，无法评估达成度。"

    lines: list[str] = []

    # 检查 Pareto 数据来源（用于顶部警告）
    pareto_info = _load_pareto_best(db_path, session_id)

    # 评估约束与目标
    constraint_attainments: list[GoalAttainment] = []
    try:
        constraint_attainments = evaluate_constraints(intent, db_path, session_id)
    except Exception as exc:
        _log.warning("约束达成度评估失败：%s", exc)

    objective_attainments: list[GoalAttainment] = []
    try:
        objective_attainments = evaluate_objectives(
            intent, db_path, session_id, establish_baseline(baseline)
        )
    except Exception as exc:
        _log.warning("目标达成度评估失败：%s", exc)

    # 数据来源标注
    if pareto_info.source == "pareto_front":
        source_note = f"基于 Pareto 第一前沿（{len(pareto_info.rows)} 个解）"
    elif pareto_info.source == "all_success":
        source_note = f"⚠ {pareto_info.warning}（{len(pareto_info.rows)} 个工况）"
    else:
        source_note = "⚠ 数据库为空，无法评估"

    lines.append(f"> *数据来源：{source_note}*")
    lines.append("")

    # 总判定句
    n_constraints = len(constraint_attainments)
    n_satisfied   = sum(1 for a in constraint_attainments if a.satisfied is True)
    n_failed      = sum(1 for a in constraint_attainments if a.satisfied is False)
    n_unknown_c   = sum(1 for a in constraint_attainments if a.satisfied is None)

    if n_constraints > 0:
        sat_str = f"{n_satisfied}/{n_constraints} 约束满足"
        if n_failed > 0:
            sat_str += f"，{n_failed} 条未满足"
        if n_unknown_c > 0:
            sat_str += f"，{n_unknown_c} 条数据缺失"
    else:
        sat_str = "无硬约束"

    has_baseline = any(a.baseline_value is not None for a in objective_attainments)
    main_imp: float | None = next(
        (a.improvement for a in objective_attainments if a.improvement is not None),
        None,
    )
    if main_imp is not None:
        imp_str = f"主目标改善 {main_imp:+.1f}%"
    elif has_baseline:
        imp_str = "主目标改善幅度无法计算（基线值为 0）"
    else:
        imp_str = "无真实基线，改善幅度不可计算"

    lines.append(f"> **总判定：{sat_str}；{imp_str}**")
    lines.append("")

    # 约束达成表
    if constraint_attainments:
        lines.append("### 硬约束达成情况")
        lines.append("")
        lines.append("| 约束名 | 阈值（原始） | 标准化约束值 | 状态 |")
        lines.append("|--------|------------|-------------|------|")
        for a in constraint_attainments:
            # target_value 是原始物理阈值（如 purity >= 0.9），展示给用户
            # achieved_value 是标准化约束值（<=0 为满足），展示真实判定依据
            lines.append(
                f"| {a.name} | {_fmt_val(a.target_value)} | "
                f"{_fmt_val(a.achieved_value)} (≤0=满足) | {_satisfied_icon(a.satisfied)} |"
            )
        lines.append("")
    else:
        lines.append("*无硬约束，跳过约束达成表。*")
        lines.append("")

    # 目标改善表
    if objective_attainments:
        lines.append("### 优化目标达成情况")
        lines.append("")
        if has_baseline:
            lines.append("| 目标名 | 方向 | 最优值 | 基线值 | 改善幅度 |")
            lines.append("|--------|------|--------|--------|---------|")
        else:
            lines.append("| 目标名 | 方向 | 最优值 | 基线 |")
            lines.append("|--------|------|--------|------|")

        for a in objective_attainments:
            dir_label = "最小化 ↓" if a.direction == "min" else "最大化 ↑"
            if has_baseline:
                imp_cell = f"{a.improvement:+.1f}%" if a.improvement is not None else "N/A"
                lines.append(
                    f"| {a.name} | {dir_label} | {_fmt_val(a.achieved_value)} | "
                    f"{_fmt_val(a.baseline_value)} | {imp_cell} |"
                )
            else:
                lines.append(
                    f"| {a.name} | {dir_label} | {_fmt_val(a.achieved_value)} | "
                    f"无真实基线 |"
                )
        lines.append("")
    else:
        lines.append("*未检测到优化目标数据。*")
        lines.append("")

    # 详细备注（折叠）
    all_attainments = constraint_attainments + objective_attainments
    notes_with_content = [a for a in all_attainments if a.note]
    if notes_with_content:
        lines.append("<details>")
        lines.append("<summary>详细匹配说明</summary>")
        lines.append("")
        for a in notes_with_content:
            lines.append(f"- **{a.name}**：{a.note}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)
