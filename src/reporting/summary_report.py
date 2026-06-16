"""
summary_report.py — 调优结果综合分析报告（阶段 C1）。

职责
----
调优结束后，从 SimulationDB 读取 Pareto 前沿和全部历史工况数据，
生成 Markdown 格式的综合分析报告，供用户决策和存档。

报告结构（5 个章节）
-------------------
1. 优化总览    — 迭代次数、成功率、Pareto 前沿大小、超体积
2. TAC 分解    — 第一前沿工况的设备费用 vs 操作费用构成
3. 排放分析    — 第一前沿工况的排放分项（蒸汽/电力/冷却水）
4. 设计变量重要性 — Spearman 相关系数排序
5. 失败诊断摘要 — 近 N 个失败工况的类型统计与高危参数区域

设计约束
--------
- 所有函数均不依赖 Aspen COM；空数据库时返回"无数据"说明而非报错
- 导入 tac.py / emissions.py 的路径与现有代码一致，不修改其接口
- 与 summarize_pareto_tool 共用 _dict_to_process_case 工具，不重复实现
- 不依赖 config_path 参数（report 层不读配置文件）
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _fmt_value(value: Any, precision: str = ".4g") -> str:
    """将数值格式化为紧凑字符串，None 显示为 N/A。"""
    if value is None:
        return "N/A"
    try:
        return format(float(value), precision)
    except (TypeError, ValueError):
        return str(value)


def _fmt_path_tail(path: str, tail: int = 3) -> str:
    """截取 Aspen 路径末尾 N 段，用于紧凑显示。"""
    if not path:
        return "<empty>"
    parts = [p for p in path.replace("/", "\\").split("\\") if p]
    return "\\".join(parts[-tail:]) if len(parts) > tail else "\\".join(parts)


def _open_db(db_path: str) -> Any:
    """打开 SimulationDB，路径解析与 summarize_pareto_tool 保持一致。"""
    from src.database.simulation_db import SimulationDB
    return SimulationDB(db_path)


def _compute_signed_spearman(
    cases: list[Any],
    param_paths: list[str],
    objective_name: str,
) -> dict[str, float]:
    """
    计算各设计变量对指定目标的带符号 Spearman ρ。

    sensitivity_analysis() 内部对 ρ 取了 abs()，不能用于判断方向。
    本函数独立复现 Spearman 计算，保留符号，返回 {param_path: ρ}。
    样本对不足（< 3）时该变量不进入结果字典。
    """
    # 借用 metrics.py 内部的 _rank / _pearson 计算，不重复实现秩算法
    try:
        from src.optimization.metrics import _rank, _pearson  # type: ignore[attr-defined]
    except ImportError:
        return {}

    valid_cases = [c for c in cases if c.simulation_valid]
    if not valid_cases:
        return {}

    result: dict[str, float] = {}
    for path in param_paths:
        pairs: list[tuple[float, float]] = []
        for c in valid_cases:
            x_raw = (c.design_vars or {}).get(path)
            ov = c.get_objective(objective_name)
            if ov is None or not ov.available or ov.value is None:
                continue
            try:
                x = float(x_raw)
                y = float(ov.value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                pairs.append((x, y))

        if len(pairs) < 3:
            continue

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n = len(xs)

        rx = _rank(xs)
        ry = _rank(ys)

        has_ties = len(set(xs)) < n or len(set(ys)) < n
        if has_ties:
            rho = _pearson(rx, ry)
        else:
            d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
            denom = n * (n * n - 1)
            rho = 0.0 if denom == 0 else 1.0 - 6.0 * d_sq_sum / denom

        result[path] = rho

    return result


def _extract_diagnose_suggestions(diag_report: str) -> list[str]:
    """
    从 _impl_diagnose_case 返回的报告文本中提取"【诊断建议】"段落的建议条目。

    Returns
    -------
    list[str]
        建议条目列表（不含序号前缀），未找到建议段时返回空列表。
    """
    lines = diag_report.splitlines()
    in_section = False
    suggestions: list[str] = []
    for line in lines:
        if "【诊断建议】" in line:
            in_section = True
            continue
        if in_section:
            # 空行或下一个【...】区段标志结束
            if line.startswith("【") or (line.strip() == "" and suggestions):
                break
            stripped = line.strip()
            # 格式：  1. 建议内容…
            if stripped and stripped[0].isdigit() and ". " in stripped:
                text = stripped.split(". ", 1)[1].strip()
                if text:
                    suggestions.append(text)
            elif stripped:
                # 不符合编号格式的行也收入（容错）
                suggestions.append(stripped)
    return suggestions


def _load_success_cases_full(db_path: str, session_id: str | None) -> list[dict[str, Any]]:
    """
    从数据库加载所有 success=True 的完整工况记录（含 blocks/streams）。

    先用 query_cases 取摘要行拿到 case_id 列表，
    再用 get_case 逐条取完整记录，避免 query_cases 只返回摘要列的问题。
    """
    with _open_db(db_path) as db:
        summary_rows = db.query_cases(status="success", session_id=session_id)
        full_rows = []
        for row in summary_rows:
            full = db.get_case(row["case_id"])
            if full:
                full_rows.append(full)
    return full_rows


def _load_all_cases_summary(db_path: str, session_id: str | None) -> list[dict[str, Any]]:
    """加载所有工况的摘要行（不含 blocks/streams）。"""
    with _open_db(db_path) as db:
        rows = db.query_cases(session_id=session_id)
    return rows


def _build_process_cases(rows: list[dict[str, Any]]) -> list[Any]:
    """
    将 SimulationDB 摘要行批量重建为 ProcessCase 对象。
    复用 summarize_pareto_tool 中的 _dict_to_process_case，避免重复实现。
    blocks/streams 不填充（轻量重建），适合 Pareto 计算和变量重要性分析。
    """
    from src.agents.tools.summarize_pareto import _dict_to_process_case
    return [_dict_to_process_case(r) for r in rows]


def _build_process_cases_with_blocks(rows: list[dict[str, Any]]) -> list[Any]:
    """
    将完整工况 dict（含 blocks 快照）重建为 ProcessCase 对象，供 TAC/排放计算使用。

    blocks 字段格式与 BlockResult.to_dict() 一致：
    {block_name: {block_type, convergence, outputs: [{name, value, unit, ...}], ...}}

    重建逻辑：
    - 从 block_type 字符串映射到 BlockType 枚举，未知类型降级为 UNKNOWN
    - convergence 字符串映射到 BlockConvergenceStatus，默认 SUCCESS
    - outputs 列表还原为 BlockOutput 对象列表
    """
    from src.agents.tools.summarize_pareto import _dict_to_process_case
    from src.models.block import BlockConvergenceStatus, BlockOutput, BlockResult, BlockType

    results = []
    for row in rows:
        # 先重建基础 ProcessCase（含目标函数、设计变量等）
        pc = _dict_to_process_case(row)

        # 重建 blocks
        blocks_raw = row.get("blocks") or {}
        if isinstance(blocks_raw, dict) and blocks_raw:
            rebuilt: dict[str, BlockResult] = {}
            for bname, bdata in blocks_raw.items():
                if not isinstance(bdata, dict):
                    continue
                btype_str = bdata.get("block_type", "UNKNOWN")
                try:
                    btype = BlockType(btype_str)
                except ValueError:
                    btype = BlockType.UNKNOWN

                conv_str = bdata.get("convergence", "success")
                try:
                    conv = BlockConvergenceStatus(conv_str)
                except ValueError:
                    conv = BlockConvergenceStatus.SUCCESS

                outputs: list[BlockOutput] = []
                for out in (bdata.get("outputs") or []):
                    outputs.append(BlockOutput(
                        path=out.get("path", ""),
                        name=out.get("name", ""),
                        value=out.get("value"),
                        unit=out.get("unit", ""),
                        value_type=out.get("value_type", 0),
                    ))

                rebuilt[bname] = BlockResult(
                    name=bname,
                    block_type=btype,
                    convergence=conv,
                    outputs=outputs,
                    notes=bdata.get("notes", ""),
                )
            pc.blocks = rebuilt

        results.append(pc)
    return results


def _compute_pareto_first_front(
    cases: list[Any],
    objective_names: list[str] | None,
) -> Any | None:
    """
    计算第一 Pareto 前沿，失败时返回 None。

    objective_names=None 时自动从 cases 中检测。
    """
    if not cases:
        return None
    try:
        from src.optimization.pareto import compute_pareto

        if objective_names is None:
            # 从第一个有目标函数的工况中获取名称
            obj_names: list[str] = []
            for c in cases:
                if c.objectives:
                    obj_names = [o.name for o in c.objectives]
                    break
            if not obj_names:
                return None
            objective_names = obj_names

        result = compute_pareto(cases, objective_names, compute_hv=True)
        return result
    except Exception as exc:
        _log.warning("Pareto 计算失败：%s", exc)
        return None


def _detect_objective_names(rows: list[dict[str, Any]]) -> list[str]:
    """从摘要行中自动检测目标函数名称。"""
    for row in rows:
        objs = row.get("objectives") or []
        if objs:
            return [o.get("name", "") for o in objs if o.get("name")]
    return []


# ---------------------------------------------------------------------------
# C1-1  TAC 分解
# ---------------------------------------------------------------------------

def generate_tac_breakdown(
    db_path: str,
    session_id: str | None = None,
) -> str:
    """
    生成第一 Pareto 前沿工况的 TAC（总年化成本）分解表。

    对第一前沿中每个工况调用 tac.py 计算，分项列出设备资本费用（CAPEX）
    和年度操作费用（OPEX），输出 Markdown 表格。

    Parameters
    ----------
    db_path:
        SimulationDB 路径。
    session_id:
        会话过滤；None 时不过滤（使用全部数据）。

    Returns
    -------
    str
        Markdown 格式的 TAC 分解表，或"无数据"说明。
    """
    try:
        rows = _load_success_cases_full(db_path, session_id)
    except Exception as exc:
        return f"> ⚠ 无法读取数据库：{exc}"

    if not rows:
        return "> 无数据：数据库中无 success 工况，无法生成 TAC 分解。"

    obj_names = _detect_objective_names(rows)
    cases = _build_process_cases_with_blocks(rows)
    pareto_result = _compute_pareto_first_front(cases, obj_names if obj_names else None)

    # 取第一前沿工况；Pareto 计算失败时降级为全部 success 工况（最多 5 个）
    if pareto_result is not None and pareto_result.first_front is not None:
        front_cases = pareto_result.first_front.cases[:10]
        label = "第一 Pareto 前沿"
    else:
        front_cases = cases[:5]
        label = "成功工况（Pareto 前沿不可用，取前 5 个）"

    if not front_cases:
        return "> 无数据：无法确定前沿工况。"

    try:
        from src.economics.tac import TACConfig, calculate_tac
    except ImportError as exc:
        return f"> ⚠ 无法导入 tac.py：{exc}"

    # skip_missing=True：允许展示 partial 结果，但明确标记；
    # allow_partial_objective=False：仍遵守"不把低估值当作可比较合计"的原则。
    tac_cfg = TACConfig(skip_missing=True, allow_partial_objective=False)

    lines: list[str] = []
    lines.append(f"**{label}（共 {len(front_cases)} 个工况）**\n")

    # 表头
    lines.append("| 工况 | Block | 类型 | CAPEX ($) | OPEX ($/yr) | TAC ($/yr) | 说明 |")
    lines.append("|------|-------|------|----------:|------------:|-----------:|------|")

    any_data = False
    for case in front_cases:
        tac_result = calculate_tac(case, tac_cfg)
        short_id = case.case_id[:8]

        if not tac_result.equipment_costs:
            lines.append(
                f"| {short_id} | — | — | — | — | — | blocks 数据不可用 |"
            )
            continue

        any_data = True
        for ec in tac_result.equipment_costs:
            capex_str = _fmt_value(ec.capex, ".0f") if ec.capex is not None else "N/A"
            opex_str  = _fmt_value(ec.opex_annual, ".0f") if ec.opex_annual is not None else "N/A"
            tac_str   = _fmt_value(ec.tac, ".0f") if ec.tac is not None else "N/A"
            notes_str = ec.notes[:40].replace("|", "｜") if ec.notes else ""
            lines.append(
                f"| {short_id} | {ec.block_name} | {ec.block_type} "
                f"| {capex_str} | {opex_str} | {tac_str} | {notes_str} |"
            )

        # 工况合计行：有 skipped_blocks 时标记 ⚠ PARTIAL，不用于排序
        if tac_result.skipped_blocks:
            skip_note = f"⚠ PARTIAL（跳过: {', '.join(tac_result.skipped_blocks)}）不可用于排序"
            total_str = "N/A（PARTIAL）"
        else:
            skip_note = ""
            total_str = _fmt_value(tac_result.total_tac, ".0f") if tac_result.total_tac is not None else "N/A"
        lines.append(
            f"| {short_id} | **合计** | — | — | — | **{total_str}** | {skip_note} |"
        )

    if not any_data:
        return "> 无数据：前沿工况中未找到可计算的 block 数据（blocks 快照为空）。"

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C1-2  排放分析
# ---------------------------------------------------------------------------

def generate_emissions_summary(
    db_path: str,
    session_id: str | None = None,
) -> str:
    """
    生成第一 Pareto 前沿工况的排放分析表。

    按蒸汽 / 电力 / 冷却水分项输出，并对比最优点与最差点
    （以总排放量为基准）。

    Returns
    -------
    str
        Markdown 格式的排放分析表，或"无数据"说明。
    """
    try:
        rows = _load_success_cases_full(db_path, session_id)
    except Exception as exc:
        return f"> ⚠ 无法读取数据库：{exc}"

    if not rows:
        return "> 无数据：数据库中无 success 工况，无法生成排放分析。"

    obj_names = _detect_objective_names(rows)
    cases = _build_process_cases_with_blocks(rows)
    pareto_result = _compute_pareto_first_front(cases, obj_names if obj_names else None)

    if pareto_result is not None and pareto_result.first_front is not None:
        front_cases = pareto_result.first_front.cases[:10]
        label = "第一 Pareto 前沿"
    else:
        front_cases = cases[:5]
        label = "成功工况（Pareto 前沿不可用，取前 5 个）"

    if not front_cases:
        return "> 无数据：无法确定前沿工况。"

    try:
        from src.economics.emissions import EmissionsConfig, calculate_emissions
    except ImportError as exc:
        return f"> ⚠ 无法导入 emissions.py：{exc}"

    # skip_missing=True：允许展示 partial 结果，但明确标记；
    # allow_partial_objective=False：不把缺设备的工况合计纳入最优/最差比较。
    em_cfg = EmissionsConfig(skip_missing=True, allow_partial_objective=False)

    # block 类型分类（与 emissions.py 内部分类保持一致）
    _ELEC_TYPES = {"PUMP", "COMPR", "MCOMPR"}

    lines: list[str] = []
    lines.append(f"**{label}（共 {len(front_cases)} 个工况）**\n")
    lines.append("| 工况 | 蒸汽/热源排放 (t CO₂/yr) | 电力排放 (t CO₂/yr) | Scope 1 (t CO₂/yr) | 合计 (t CO₂/yr) | 完整性 |")
    lines.append("|------|-------------------------:|--------------------:|-------------------:|----------------:|--------|")

    # 只把无 skipped_blocks/streams 的完整结果纳入对比
    complete_totals: list[tuple[str, float]] = []

    for case in front_cases:
        em_result = calculate_emissions(case, em_cfg)
        short_id = case.case_id[:8]

        # blocks={} 且 semantic_blocks={} 时，calculate_emissions 会得到 total=0.0
        # 且 skipped_blocks=[]，但这不是"无排放"而是"无设备快照，无法计算"。
        # 必须单独检测并标记为 PARTIAL，绝不能加入 complete_totals。
        no_device_data = (not case.blocks) and (not case.semantic_blocks)
        is_partial = no_device_data or bool(em_result.skipped_blocks or em_result.skipped_streams)

        # 按 block 类型粗分：电力类 vs 蒸汽/热源类
        steam_total = 0.0
        elec_total  = 0.0
        for ee in em_result.equipment_emissions:
            if ee.scope2_annual is None:
                continue
            if ee.block_type in _ELEC_TYPES:
                elec_total += ee.scope2_annual
            else:
                steam_total += ee.scope2_annual

        scope1 = em_result.total_scope1_annual
        total  = em_result.total_annual

        if total is not None and not is_partial:
            complete_totals.append((short_id, total))

        if no_device_data:
            completeness  = "⚠ PARTIAL（无设备快照）"
            total_display = "—（无法计算）"
        elif is_partial:
            completeness  = "⚠ PARTIAL"
            total_display = f"~~{_fmt_value(total, '.2f')}~~（不可比较）"
        else:
            completeness  = "✓ 完整"
            total_display = f"**{_fmt_value(total, '.2f')}**"
        lines.append(
            f"| {short_id} "
            f"| {_fmt_value(steam_total, '.2f')} "
            f"| {_fmt_value(elec_total, '.2f')} "
            f"| {_fmt_value(scope1, '.2f')} "
            f"| {total_display} "
            f"| {completeness} |"
        )

    # 最优/最差对比：仅使用 complete_totals（无 skipped 的完整结果）
    if len(complete_totals) >= 2:
        sorted_em = sorted(complete_totals, key=lambda x: x[1])
        best_id, best_val = sorted_em[0]
        worst_id, worst_val = sorted_em[-1]
        delta = worst_val - best_val
        lines.append("")
        lines.append(
            f"> **对比（仅完整工况）**：最优点 `{best_id}` = {best_val:.2f} t CO₂/yr；"
            f"最差点 `{worst_id}` = {worst_val:.2f} t CO₂/yr；"
            f"差值 = {delta:.2f} t CO₂/yr"
        )
    elif complete_totals:
        lines.append("")
        lines.append("> 完整结果只有 1 个工况，无法进行最优/最差对比。")
    else:
        lines.append("")
        lines.append("> 所有工况均为 PARTIAL（存在 skipped blocks/streams），拒绝最优/最差排序。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C1-3  设计变量重要性
# ---------------------------------------------------------------------------

def generate_variable_importance(
    db_path: str,
    objective_name: str,
    session_id: str | None = None,
) -> str:
    """
    计算各设计变量与目标函数的 Spearman 相关系数，输出重要性排序表。

    Parameters
    ----------
    db_path:
        SimulationDB 路径。
    objective_name:
        目标函数名称（如 "TAC"、"REB_DUTY"），大小写敏感。
    session_id:
        会话过滤；None 时不过滤。

    Returns
    -------
    str
        Markdown 格式的变量重要性表 + 简短文字解读，或"无数据"说明。
    """
    try:
        rows = _load_all_cases_summary(db_path, session_id)
    except Exception as exc:
        return f"> ⚠ 无法读取数据库：{exc}"

    # 只保留 simulation_valid=True 的工况（目标值可信）
    valid_rows = [r for r in rows if r.get("simulation_valid")]
    if not valid_rows:
        return f"> 无数据：数据库中无 simulation_valid 工况，无法分析变量重要性（目标：{objective_name}）。"

    cases = _build_process_cases(valid_rows)

    try:
        from src.optimization.metrics import sensitivity_analysis, rank_variables
    except ImportError as exc:
        return f"> ⚠ 无法导入 metrics.py：{exc}"

    # 收集设计变量路径
    param_paths: list[str] = []
    seen: set[str] = set()
    for c in cases:
        for path in (c.design_vars or {}):
            if path not in seen:
                seen.add(path)
                param_paths.append(path)

    if not param_paths:
        return f"> 无数据：工况中无设计变量记录（目标：{objective_name}）。"

    try:
        sens = sensitivity_analysis(
            cases=cases,
            param_paths=param_paths,
            objective_names=[objective_name],
            method="spearman",
        )
    except Exception as exc:
        return f"> ⚠ 敏感性分析失败：{exc}"

    ranked = rank_variables(sens)

    # 独立计算带符号 Spearman ρ，用于方向判断。
    # sensitivity_analysis 内部对 ρ 取了 abs()，不能直接用 sens.score() 判断方向。
    signed_rho: dict[str, float] = _compute_signed_spearman(cases, param_paths, objective_name)

    lines: list[str] = []
    lines.append(f"**目标函数：{objective_name}  |  样本量：{sens.n_samples}  |  方法：{sens.method}**\n")
    lines.append("| 排名 | 变量（路径末尾）| Spearman ρ（带符号） | 方向 | 可靠性 |")
    lines.append("|------|----------------|:--------------------:|------|--------|")

    for rank_i, (path, _abs_score) in enumerate(ranked, 1):
        short = _fmt_path_tail(path)
        reliable = sens.is_reliable(path)
        reliable_str = "✓ 可靠" if reliable else "⚠ 证据不足"

        rho = signed_rho.get(path)
        if rho is None:
            rho_str = "N/A"
            direction = "—"
        else:
            rho_str = f"{rho:+.3f}"
            direction = "正相关 ↑" if rho >= 0 else "负相关 ↓"

        lines.append(
            f"| {rank_i} | `{short}` | {rho_str} | {direction} | {reliable_str} |"
        )

    # 警告
    if sens.warnings:
        lines.append("")
        for w in sens.warnings[:3]:
            lines.append(f"> ⚠ {w}")

    # 简短解读（使用绝对值排序的 top 变量，但显示带符号 ρ）
    # 样本不足时跳过解读，避免输出不可靠的 ρ 值误导用户
    if ranked and not sens.warnings:
        top_path, top_abs = ranked[0]
        top_short = _fmt_path_tail(top_path)
        top_rho = signed_rho.get(top_path)
        rho_display = f"{top_rho:+.3f}" if top_rho is not None else f"{top_abs:.3f}"
        direction_hint = ""
        if top_rho is not None:
            direction_hint = (
                "增大该变量会使目标升高。" if top_rho > 0
                else "增大该变量会使目标降低。" if top_rho < 0
                else ""
            )
        lines.append("")
        lines.append(
            f"> **解读**：`{top_short}` 对 `{objective_name}` 影响最大（ρ = {rho_display}）。"
            f"{direction_hint}"
            f" |ρ| > 0.3 的变量通常值得优先关注；|ρ| < 0.1 的变量可考虑固定以减少搜索维度。"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C1-4  失败诊断摘要
# ---------------------------------------------------------------------------

def generate_failure_summary(
    db_path: str,
    session_id: str | None = None,
    limit: int = 5,
) -> str:
    """
    归并最近 N 个失败工况的诊断结果，找出高危参数区域。

    Parameters
    ----------
    db_path:
        SimulationDB 路径。
    session_id:
        会话过滤；None 时不过滤。
    limit:
        分析的失败工况数量上限，默认 5。

    Returns
    -------
    str
        Markdown 格式的失败诊断摘要，或"无数据"说明。
    """
    try:
        with _open_db(db_path) as db:
            all_rows = db.query_cases(session_id=session_id)
    except Exception as exc:
        return f"> ⚠ 无法读取数据库：{exc}"

    # 失败状态包含：sim_failed / objective_error / infeasible / constraint_error
    _FAIL_STATUSES = {"sim_failed", "objective_error", "infeasible", "constraint_error"}
    failed_rows = [r for r in all_rows if r.get("status") in _FAIL_STATUSES]

    if not failed_rows:
        return "> 无数据：数据库中无失败工况记录。"

    # 统计各失败类型
    type_counts: dict[str, int] = {}
    for row in failed_rows:
        s = row.get("status", "unknown")
        type_counts[s] = type_counts.get(s, 0) + 1

    total_fail = len(failed_rows)
    total_cases = len(all_rows)
    fail_rate = total_fail / total_cases * 100 if total_cases > 0 else 0.0

    lines: list[str] = []
    lines.append(
        f"**失败工况统计：{total_fail}/{total_cases} ({fail_rate:.1f}%)**\n"
    )

    # 类型分布表
    lines.append("| 失败类型 | 数量 | 占比 |")
    lines.append("|----------|-----:|-----:|")
    _TYPE_DESC = {
        "sim_failed":        "仿真失败（引擎错误/超时）",
        "objective_error":   "目标函数计算失败",
        "infeasible":        "约束违反（不可行点）",
        "constraint_error":  "约束计算失败",
    }
    for status, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        desc = _TYPE_DESC.get(status, status)
        pct = count / total_fail * 100
        lines.append(f"| {desc} | {count} | {pct:.0f}% |")

    # 取最近 limit 个失败工况分析设计变量聚集区域
    recent_failed = failed_rows[-limit:]
    lines.append("")
    lines.append(f"**近 {len(recent_failed)} 个失败工况的设计变量分布：**\n")

    # 收集各变量在失败工况中的值
    var_values: dict[str, list[float]] = {}
    for row in recent_failed:
        dv = row.get("design_vars") or {}
        for path, val in dv.items():
            try:
                fval = float(val)
                if math.isfinite(fval):
                    var_values.setdefault(path, []).append(fval)
            except (TypeError, ValueError):
                pass

    if not var_values:
        lines.append("> 失败工况中无有效设计变量记录，无法分析高危区域。")
    else:
        lines.append("| 变量 | 最小值 | 最大值 | 均值 | 样本数 |")
        lines.append("|------|-------:|-------:|-----:|-------:|")
        for path in sorted(var_values):
            vals = var_values[path]
            short = _fmt_path_tail(path)
            lo = min(vals)
            hi = max(vals)
            mean = sum(vals) / len(vals)
            lines.append(
                f"| `{short}` | {lo:.4g} | {hi:.4g} | {mean:.4g} | {len(vals)} |"
            )
        lines.append("")
        lines.append(
            "> **提示**：上表显示失败工况中各设计变量的取值范围，"
            "集中在极端值（接近上下界）附近的变量可能是高危区域，"
            "建议适当收窄相应边界。"
        )

    # 对每个失败工况调用 diagnose_case_tool 归并诊断建议
    lines.append("")
    lines.append("**逐工况诊断建议：**\n")

    try:
        from src.agents.tools.diagnose_case import _impl_diagnose_case
        _diagnose_available = True
    except ImportError:
        _diagnose_available = False

    if not _diagnose_available:
        lines.append("> ⚠ 无法导入 diagnose_case_tool，跳过逐工况诊断。")
    else:
        for row in recent_failed:
            case_id = row.get("case_id", "")
            short_id = case_id[:8]
            status   = row.get("status", "?")
            iteration = row.get("iteration", "?")

            diag = _impl_diagnose_case(
                db_path=db_path,
                case_id=case_id,
                include_input_verification=False,
                include_block_details=False,
                include_failed_outputs=True,
                include_blocks_snapshot=False,
                max_items=5,
            )

            if diag.startswith("错误："):
                lines.append(f"- `{short_id}` (iter={iteration}, {status}): {diag}")
                continue

            # 从诊断报告中提取"【诊断建议】"段落
            suggestions = _extract_diagnose_suggestions(diag)
            if suggestions:
                lines.append(f"- `{short_id}` (iter={iteration}, {status}):")
                for s in suggestions[:3]:
                    lines.append(f"  - {s}")
            else:
                lines.append(
                    f"- `{short_id}` (iter={iteration}, {status}): 诊断工具未产出具体建议。"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C1-5  综合报告
# ---------------------------------------------------------------------------

def generate_summary_report(
    db_path: str,
    config_path: str | None = None,
    session_id: str | None = None,
    intent: Any | None = None,
    baseline: dict[str, float] | None = None,
) -> str:
    """
    生成完整的优化结果综合分析报告。

    串联目标达成总览（第 0 章）和 C1-1 ~ C1-4 及优化总览，输出完整 Markdown 报告（6 个章节）。

    Parameters
    ----------
    db_path:
        SimulationDB 路径。
    config_path:
        优化配置 YAML 路径（可选，用于在报告头部显示配置文件名）。
    session_id:
        会话过滤；None 时使用全部数据。
    intent:
        OptimizationIntent 对象（可选）；提供时生成第 0 章目标达成总览；
        缺失时该章节降级为"未提供优化意图"说明，不报错。
    baseline:
        外部传入的基线值 {目标名: 值}，来自优化前初始单跑结果（可选）。
        缺失时报告标注"无真实基线，改善幅度不可计算"，不报错。

    Returns
    -------
    str
        完整 Markdown 报告字符串。
    """
    lines: list[str] = []

    # 报告头部
    cfg_note = f"配置文件：`{Path(config_path).name}`  " if config_path else ""
    db_note  = f"数据库：`{Path(db_path).name}`"
    session_note = f"  会话：`{session_id}`" if session_id else ""
    lines.append(f"# PAO 优化结果综合分析报告")
    lines.append(f"> {cfg_note}{db_note}{session_note}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 第 0 章：目标达成总览（H2）
    # ------------------------------------------------------------------ #
    lines.append("## 0. 目标达成总览")
    lines.append("")
    try:
        from src.reporting.goal_attainment import generate_goal_attainment_section
        lines.append(generate_goal_attainment_section(
            intent=intent,
            db_path=db_path,
            config_path=config_path,
            session_id=session_id,
            baseline=baseline,
        ))
    except Exception as exc:
        _log.warning("目标达成总览生成失败：%s", exc)
        lines.append(f"> ⚠ 目标达成总览生成失败：{exc}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 第 1 章：优化总览
    # ------------------------------------------------------------------ #
    lines.append("## 1. 优化总览")
    lines.append("")

    try:
        all_rows = _load_all_cases_summary(db_path, session_id)
    except Exception as exc:
        lines.append(f"> ⚠ 无法读取数据库：{exc}")
        return "\n".join(lines)

    total_cases = len(all_rows)
    if total_cases == 0:
        lines.append("> 无数据：数据库中无任何工况记录。")
        for ch in ("2. TAC 分解", "3. 排放分析", "4. 设计变量重要性", "5. 失败诊断摘要"):
            lines.append(f"\n## {ch}\n\n> 无数据。")
        return "\n".join(lines)

    success_rows = [r for r in all_rows if r.get("status") == "success"]
    n_success = len(success_rows)
    success_rate = n_success / total_cases * 100

    # 计算 Pareto 前沿和超体积
    obj_names = _detect_objective_names(all_rows)
    cases_all = _build_process_cases(all_rows)
    pareto_result = _compute_pareto_first_front(cases_all, obj_names if obj_names else None)

    hv_str        = "N/A"
    front_size    = "N/A"
    n_fronts_str  = "N/A"
    obj_names_str = ", ".join(obj_names) if obj_names else "（未检测到）"

    if pareto_result is not None:
        n_fronts_str = str(pareto_result.n_fronts)
        if pareto_result.first_front is not None:
            front_size = str(len(pareto_result.first_front.cases))
        if pareto_result.hypervolume is not None:
            hv_str = f"{pareto_result.hypervolume:.6g}"

    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|---|")
    lines.append(f"| 总工况数 | {total_cases} |")
    lines.append(f"| 成功工况数 | {n_success} |")
    lines.append(f"| 成功率 | {success_rate:.1f}% |")
    lines.append(f"| 目标函数 | {obj_names_str} |")
    lines.append(f"| Pareto 前沿大小（第一前沿） | {front_size} |")
    lines.append(f"| Pareto 层数 | {n_fronts_str} |")
    lines.append(f"| 超体积（HV） | {hv_str} |")

    # 显示最大迭代编号
    if all_rows:
        max_iter = max(r.get("iteration", 0) for r in all_rows)
        lines.append(f"| 最大迭代编号 | {max_iter} |")

    lines.append("")

    # ------------------------------------------------------------------ #
    # 第 2 章：TAC 分解
    # ------------------------------------------------------------------ #
    lines.append("## 2. TAC 分解")
    lines.append("")
    lines.append(generate_tac_breakdown(db_path, session_id))
    lines.append("")

    # ------------------------------------------------------------------ #
    # 第 3 章：排放分析
    # ------------------------------------------------------------------ #
    lines.append("## 3. 排放分析")
    lines.append("")
    lines.append(generate_emissions_summary(db_path, session_id))
    lines.append("")

    # ------------------------------------------------------------------ #
    # 第 4 章：设计变量重要性
    # ------------------------------------------------------------------ #
    lines.append("## 4. 设计变量重要性")
    lines.append("")
    if obj_names:
        # 对每个目标分别输出重要性分析
        for obj_name in obj_names[:3]:   # 最多 3 个目标
            lines.append(f"### 4.{obj_names.index(obj_name) + 1} 目标：{obj_name}")
            lines.append("")
            lines.append(generate_variable_importance(db_path, obj_name, session_id))
            lines.append("")
    else:
        lines.append(generate_variable_importance(db_path, "", session_id))
        lines.append("")

    # ------------------------------------------------------------------ #
    # 第 5 章：失败诊断摘要
    # ------------------------------------------------------------------ #
    lines.append("## 5. 失败诊断摘要")
    lines.append("")
    lines.append(generate_failure_summary(db_path, session_id, limit=5))
    lines.append("")

    return "\n".join(lines)
