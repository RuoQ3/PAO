"""Standalone deterministic analysis skill for historical PAO cases.

This tool is useful to a general-purpose Agent or a human review workflow.
The running closed loop calls ``build_analysis_report`` directly from memory;
this database-backed wrapper gives the same evidence without Aspen COM.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from src.agents.closed_loop.analysis import build_analysis_report

from .summarize_pareto import _dict_to_process_case, _resolve_db_path

_MAX_CASES = 1000


def _parse_objective_names(raw: str) -> list[str]:
    names = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not names:
        raise ValueError("objective_names 不能为空，请提供逗号分隔的目标名称")
    if len(names) != len(set(names)):
        raise ValueError("objective_names 不能包含重复名称")
    return names


def _impl_analyze_closed_loop(
    db_path: str,
    objective_names: str,
    session_id: str | None = None,
    iteration_min: int | None = None,
    iteration_max: int | None = None,
    tags: str | None = None,
    sensitivity_method: str = "spearman",
    include_infeasible: bool = False,
) -> str:
    """Query a SimulationDB and return a JSON AnalysisReport."""
    try:
        names = _parse_objective_names(objective_names)
        if sensitivity_method not in ("spearman", "variance"):
            raise ValueError("sensitivity_method 必须为 spearman 或 variance")
        path = _resolve_db_path(db_path)
    except (FileNotFoundError, ValueError) as exc:
        return f"错误：{exc}"

    try:
        from src.database.simulation_db import SimulationDB

        tag_list = [item.strip() for item in (tags or "").split(",") if item.strip()]
        with SimulationDB(path) as db:
            rows = db.query_cases(
                session_id=session_id.strip() if session_id and session_id.strip() else None,
                iteration_min=iteration_min,
                iteration_max=iteration_max,
                tags=tag_list or None,
                limit=_MAX_CASES,
                offset=0,
            )
        if not rows:
            return "查询结果为空：没有符合条件的工况记录。"

        cases = [_dict_to_process_case(row) for row in rows]
        param_paths = sorted({
            str(path)
            for row in rows
            for path in (row.get("design_vars") or {})
        })
        report = build_analysis_report(
            cases,
            objective_names=names,
            param_paths=param_paths,
            optimizer_inputs=[row.get("design_vars") or {} for row in rows],
            sensitivity_method=sensitivity_method,
            include_infeasible=include_infeasible,
        )
        if include_infeasible:
            report["pareto"]["include_infeasible_requested"] = True
        report["query"] = {
            "db_path": str(path),
            "session_id": session_id or None,
            "iteration_min": iteration_min,
            "iteration_max": iteration_max,
            "tags": tag_list,
            "truncated": len(rows) >= _MAX_CASES,
        }
        return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as exc:  # noqa: BLE001 — a diagnostic tool must return a clear error
        return f"错误：分析数据库失败 [{type(exc).__name__}] — {exc}"


@tool
def analyze_closed_loop_tool(
    db_path: str,
    objective_names: str,
    session_id: str = "",
    iteration_min: int = -1,
    iteration_max: int = -1,
    tags: str = "",
    sensitivity_method: str = "spearman",
    include_infeasible: bool = False,
) -> str:
    """分析 PAO 历史工况并返回结构化 JSON evidence report。

    该工具不连接 Aspen Plus。报告包含数据质量、收敛率、约束违反、目标
    趋势、Pareto/HV、敏感性排序和失败模式，适合 Agent 在提出下一步动作
    前读取。默认 Pareto 只接受可行工况；显式设置 ``include_infeasible``
    可用于约束松弛诊断，但不应把该结果当作正式优化前沿。
    """
    return _impl_analyze_closed_loop(
        db_path=db_path,
        objective_names=objective_names,
        session_id=session_id,
        iteration_min=None if iteration_min < 0 else iteration_min,
        iteration_max=None if iteration_max < 0 else iteration_max,
        tags=tags,
        sensitivity_method=sensitivity_method,
        include_infeasible=include_infeasible,
    )
