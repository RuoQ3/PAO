"""
report.py — run_demo_case_workflow 报告组装层。

只从 DemoWorkflowState 读取已有数据，生成最终文本报告。
不调用任何 tools，不导入底层依赖，不读写数据库，不运行仿真。
不把失败、缺失、跳过伪装成成功。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.demo_workflow.state import DemoWorkflowState, WorkflowStep

# ---------------------------------------------------------------------------
# 内部格式化辅助
# ---------------------------------------------------------------------------

def _or_unset(value: object) -> str:
    """None 或空字符串显示为"未设置"。"""
    if value is None or value == "":
        return "未设置"
    return str(value)


def _or_none_str(value: object) -> str:
    """None 显示为"无"，其余转字符串。"""
    if value is None:
        return "无"
    return str(value)


def _list_or_none(lst: list) -> str:
    """空列表显示为"无"，否则逗号拼接。"""
    if not lst:
        return "无"
    return ", ".join(str(x) for x in lst)


def _fmt_step(step: "WorkflowStep | None", step_name: str) -> str:
    """将单个 WorkflowStep（或缺失）格式化为若干行文本。"""
    if step is None:
        return f"  [{step_name}] 未执行"

    status_tag = {
        "ok":      "[成功]",
        "error":   "[失败]",
        "skipped": "[已跳过]",
        "pending": "[等待中]",
    }.get(step.status, f"[{step.status}]")

    lines: list[str] = [f"  [{step_name}] {status_tag}"]

    if step.status == "skipped":
        reason = step.skipped_reason if step.skipped_reason else "未提供"
        lines.append(f"    跳过原因：{reason}")

    if step.status == "error":
        lines.append("    注意：以下为失败详情：")

    if step.report:
        # 缩进嵌入，限制行数避免报告过长
        report_lines = step.report.splitlines()
        for line in report_lines[:60]:
            lines.append(f"    {line}")
        if len(report_lines) > 60:
            lines.append(f"    ...（共 {len(report_lines)} 行，已截断）")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 章节 1：配置摘要
# ---------------------------------------------------------------------------

def _section_config_summary(state: "DemoWorkflowState") -> str:
    lines = ["【1. 配置摘要】"]
    lines.append(f"  配置路径（原始）   : {_or_unset(state.case_config_path)}")
    lines.append(f"  配置路径（已解析） : {_or_unset(state.resolved_config_path)}")
    lines.append(f"  optimizer_type    : {_or_unset(state.optimizer_type)}")
    lines.append(f"  objective_names   : {_list_or_none(state.objective_names)}")
    lines.append(f"  db_path           : {_or_none_str(state.db_path)}")
    lines.append(f"  node_db_path      : {_or_none_str(state.node_db_path)}")
    lines.append(f"  session_id        : {_or_none_str(state.session_id)}")
    lines.append(f"  diagnostic_case_ids: {_list_or_none(state.diagnostic_case_ids)}")
    lines.append(f"  case_id_source    : {_or_unset(state.case_id_source)}")
    lines.append(f"  aborted           : {'是' if state.aborted else '否'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 章节 2：校验结果
# ---------------------------------------------------------------------------

def _section_validate(state: "DemoWorkflowState") -> str:
    lines = ["【2. 校验结果】"]
    # 先展示 load_config 步骤（配置解析），再展示 validate_config（配置校验）
    load_step = state.get_step("load_config")
    lines.append(_fmt_step(load_step, "load_config"))
    step = state.get_step("validate_config")
    lines.append(_fmt_step(step, "validate_config"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 章节 3：运行/优化结果
# ---------------------------------------------------------------------------

def _section_run_result(state: "DemoWorkflowState") -> str:
    lines = ["【3. 运行/优化结果】"]
    if state.is_pareto_branch:
        lines.append("  分支：pareto_bayesian（多目标 Pareto 优化）")
        step = state.get_step("optimize_pareto")
        lines.append(_fmt_step(step, "optimize_pareto"))
        # run_case 在此分支不适用
        run_step = state.get_step("run_case")
        if run_step is None:
            lines.append("  [run_case] 未执行/不适用（pareto_bayesian 分支不运行单次仿真）")
        else:
            lines.append(_fmt_step(run_step, "run_case"))
    else:
        lines.append(f"  分支：{_or_unset(state.optimizer_type)}（单次/单目标运行）")
        step = state.get_step("run_case")
        lines.append(_fmt_step(step, "run_case"))
        # optimize_pareto 在此分支不适用
        lines.append("  [optimize_pareto] 不适用（当前分支不执行 Pareto 优化）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 章节 4：数据库查询结果
# ---------------------------------------------------------------------------

def _section_db_query(state: "DemoWorkflowState") -> str:
    lines = ["【4. 数据库查询结果】"]
    if state.db_path is None:
        lines.append("  db_path=None：当前分支未配置 SimulationDB 查询路径。")
        lines.append("  单次运行分支不保证写入数据库，以下仅展示步骤状态。")

    if state.db_path is not None:
        lines.append(f"  目标数据库：{state.db_path}")

    step = state.get_step("query_simulation_db")
    lines.append(_fmt_step(step, "query_simulation_db"))
    if step is not None and step.status == "error":
        lines.append("  注意：数据库查询失败，上方为错误详情，请勿以此判断工况写入成功。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 章节 5：诊断结论
# ---------------------------------------------------------------------------

def _section_diagnose(state: "DemoWorkflowState") -> str:
    lines = ["【5. 诊断结论】"]

    if not state.diagnostic_case_ids:
        lines.append("  无失败工况 case_id 可供诊断。")
        lines.append("  （诊断触发条件：pareto_bayesian 分支 + 数据库查询成功 + 存在失败工况）")
    else:
        ids_str = ", ".join(state.diagnostic_case_ids)
        lines.append(f"  诊断工况列表：{ids_str}")
        if state.case_id_source == "text_fallback":
            lines.append(
                "  注意：case_id 来源为文本解析 fallback（不确定），"
                "建议用 query_simulation_db_tool mode='get_case' 核实。"
            )
        elif state.case_id_source == "db_query":
            lines.append("  case_id 来源：DB 直接查询（可信）。")

    step_diagnose = state.get_step("diagnose_case")
    lines.append(_fmt_step(step_diagnose, "diagnose_case"))

    step_node = state.get_step("query_node_db")
    lines.append(_fmt_step(step_node, "query_node_db"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 章节 6：Pareto 总结
# ---------------------------------------------------------------------------

def _section_pareto_summary(state: "DemoWorkflowState") -> str:
    lines = ["【6. Pareto 总结】"]

    if not state.is_pareto_branch:
        lines.append("  不适用（当前分支非 pareto_bayesian）。")
        return "\n".join(lines)

    opt_step = state.get_step("optimize_pareto")
    sum_step = state.get_step("summarize_pareto")

    # optimize_pareto 失败时：除非 summarize_pareto 的 report 明确含"历史数据库"，
    # 否则一律显示"因优化失败跳过本次 Pareto 总结"，防止历史数据库结果被误认为本次结果。
    if opt_step is not None and opt_step.status == "error":
        if sum_step is not None and "历史数据库" in (sum_step.report or ""):
            lines.append(_fmt_step(sum_step, "summarize_pareto"))
        else:
            lines.append("  因优化步骤（optimize_pareto）失败，跳过本次 Pareto 总结。")
            lines.append("  不展示历史数据库 Pareto 结果，避免与本次运行结果混淆。")
        return "\n".join(lines)

    if sum_step is None:
        lines.append("  [summarize_pareto] 未执行")
    else:
        lines.append(_fmt_step(sum_step, "summarize_pareto"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 章节 7：下一步建议
# ---------------------------------------------------------------------------

def _section_next_actions(state: "DemoWorkflowState") -> str:
    lines = ["【7. 下一步建议】"]

    if state.next_actions:
        for i, action in enumerate(state.next_actions, 1):
            lines.append(f"  [{i}] {action}")
    else:
        # next_actions 为空仅在 determine_next_actions 未被调用时出现
        # （理论上不应发生；此处作保底，不输出固定套话）
        lines.append("  （暂无具体建议，请查看各步骤详情）")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 报告入口
# ---------------------------------------------------------------------------

def build_demo_workflow_report(state: "DemoWorkflowState") -> str:
    """基于 DemoWorkflowState 生成 run_demo_case_workflow 的完整文本报告。

    章节顺序固定：
      1. 配置摘要
      2. 校验结果
      3. 运行/优化结果
      4. 数据库查询结果
      5. 诊断结论
      6. Pareto 总结
      7. 下一步建议

    不调用任何 tool，不读写数据库，不访问 Aspen，不伪装失败为成功。
    """
    # 运行时才导入 state，避免循环依赖
    from src.agents.demo_workflow.state import DemoWorkflowState as _DemoWorkflowState  # noqa: F401

    sections = [
        "=== run_demo_case_workflow 报告 ===",
        "",
        _section_config_summary(state),
        "",
        _section_validate(state),
        "",
        _section_run_result(state),
        "",
        _section_db_query(state),
        "",
        _section_diagnose(state),
        "",
        _section_pareto_summary(state),
        "",
        _section_next_actions(state),
    ]
    return "\n".join(sections)
