"""
run_case.py — run_case_tool 实现。

功能：在 Aspen Plus 中执行一次单点工况评估，返回目标函数值和约束状态。
需要 Windows + Aspen Plus + pywin32 环境。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from . import _common

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 设计变量辅助函数
# ---------------------------------------------------------------------------

def _parse_design_vars_json(design_vars_json: str | None) -> tuple[dict, str | None]:
    """
    将 design_vars_json 字符串解析为 dict。
    成功时返回 (dict, None)；失败时返回 ({}, error_msg)。
    """
    if not design_vars_json or design_vars_json.strip() == "":
        return {}, None

    import json
    try:
        parsed = json.loads(design_vars_json)
    except json.JSONDecodeError as exc:
        return {}, f"design_vars_json 不是合法 JSON：{exc}"

    if not isinstance(parsed, dict):
        return {}, f"design_vars_json 应为 JSON 对象（字典），实际类型为 {type(parsed).__name__}。"

    result: dict = {}
    for k, v in parsed.items():
        try:
            result[str(k)] = float(v)
        except (TypeError, ValueError):
            return {}, (
                f"设计变量 {k!r} 的值 {v!r} 无法转为 float，"
                "design_vars_json 的值必须全部为数字。"
            )
    return result, None


def _build_initial_design_vars(opt_cfg: Any) -> dict:
    """
    从 OptimizeCaseConfig 中提取初始设计变量点（参数空间中间值）。
    仅用于配置验证和 smoke test，不是优化意义上的最优点。
    """
    result: dict = {}
    result.update(opt_cfg.fixed_vars)
    for path, (lo, hi) in opt_cfg.param_bounds.items():
        result[path] = (lo + hi) / 2.0
    return result


def _apply_derived_and_repair(opt_cfg: Any, design_vars: dict) -> dict:
    """
    对 design_vars 依次执行 repair_design_vars + apply_derived_vars。
    复现优化循环调用 run_case() 之前的预处理步骤。
    """
    from src.workflows.common import repair_design_vars, apply_derived_vars

    integer_paths = getattr(opt_cfg, "integer_var_paths", set())
    var_deps = getattr(opt_cfg, "var_dependencies", {})
    derived_specs = getattr(opt_cfg, "derived_var_specs", [])

    # 注意参数顺序：(design_vars, integer_paths, param_bounds, var_dependencies)
    repaired, _repair_notes = repair_design_vars(
        design_vars, integer_paths, opt_cfg.param_bounds, var_deps
    )
    # apply_derived_vars 返回 (expanded_vars, notes)，需要拆包
    final, _derived_notes = apply_derived_vars(repaired, derived_specs)
    return final


# ---------------------------------------------------------------------------
# 结果格式化
# ---------------------------------------------------------------------------

def _fmt_case_summary(case: Any) -> str:
    """将 ProcessCase 格式化为 agent 可读的运行报告。"""
    lines: list[str] = []
    lines.append("=== run_case 单次运行报告 ===")
    lines.append("")

    lines.append("【运行状态】")
    lines.append(f"  case_id   : {case.case_id}")
    lines.append(f"  iteration : {case.iteration}")
    lines.append(f"  status    : {case.status.value}")
    lines.append(f"  成功（可采纳）: {'是' if case.success else '否'}")
    lines.append(f"  仿真收敛  : {'是' if case.simulation_valid else '否'}")
    lines.append(f"  可行（约束）: {case.feasible if case.feasible is not None else '无约束'}")
    lines.append(f"  运行耗时  : {case.run_time:.1f} s")
    if case.tags:
        lines.append(f"  标签      : {', '.join(case.tags)}")
    lines.append("")

    lines.append("【设计变量（输入点）】")
    for path, val in case.design_vars.items():
        short = path.split("\\")[-1] if "\\" in path else path
        lines.append(f"  {short:<30} = {val}")
    lines.append("")

    lines.append("【目标函数】")
    if not case.objectives:
        lines.append("  （无目标函数）")
    else:
        for obj in case.objectives:
            direction = "最小化↓" if obj.minimize else "最大化↑"
            if obj.available:
                lines.append(f"  {obj.name:<20} = {obj.value:.6g} {obj.unit}  [{direction}]")
            else:
                lines.append(f"  {obj.name:<20} = [不可用]  错误：{obj.error}")
    lines.append("")

    lines.append("【约束条件】")
    if not case.constraints:
        lines.append("  （无约束）")
    else:
        for con in case.constraints:
            if con.available:
                satisfied_str = "满足" if con.satisfied else "违反"
                lines.append(
                    f"  {con.name:<25} = {con.value:.6g}  [{satisfied_str}]"
                    f"  (<=0 为满足)"
                )
            else:
                lines.append(f"  {con.name:<25} = [不可用]  错误：{con.error}")
    lines.append("")

    if not case.simulation_valid and case.sim_result is not None:
        lines.append("【仿真失败详情】")
        sr = case.sim_result
        lines.append(f"  引擎状态  : {sr.status.value}")
        if sr.error:
            lines.append(f"  错误信息  : {sr.error}")
        if getattr(sr, "warnings", None):
            for w in sr.warnings[:5]:
                lines.append(f"  警告      : {w}")
        lines.append("")

    if case.notes:
        lines.append("【运行注记（block/stream 提取失败等）】")
        for note_line in case.notes.split("\n")[:10]:
            lines.append(f"  {note_line}")
        lines.append("")

    lines.append("【综合结论】")
    if case.success:
        obj_vals = "  ".join(
            f"{o.name}={o.value:.4g}" for o in case.objectives if o.available
        )
        lines.append(f"  [成功] 工况有效，目标函数可采纳。{obj_vals}")
    elif case.status.value == "sim_failed":
        lines.append("  [仿真失败] Aspen 仿真未收敛或超时，本点不可用。")
        if case.sim_result and case.sim_result.error:
            lines.append(f"  根本原因：{case.sim_result.error}")
    elif case.status.value == "infeasible":
        violated = [
            f"{c.name}={c.value:.4g}" for c in case.constraints
            if c.available and c.satisfied is False
        ]
        lines.append(f"  [不可行] 约束违反：{', '.join(violated) or '详见上方'}")
    elif case.status.value == "objective_error":
        failed_objs = [o.name for o in case.objectives if not o.available]
        lines.append(f"  [目标错误] 目标函数计算失败：{', '.join(failed_objs)}")
    else:
        lines.append(f"  [其他] status={case.status.value}，请检查上方详情。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def _impl_run_case(
    config_path: str,
    design_vars_json: str | None,
    iteration: int,
) -> str:
    """run_case_tool 的核心实现，出错时返回 '错误：' 字符串。"""
    # 1. 解析配置路径
    try:
        yaml_path = _common._resolve_config_path(config_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    # 2. 按需导入运行时依赖
    err = _common._import_run_time_deps()
    if err:
        return err

    # 3. 加载 OptimizeCaseConfig
    try:
        opt_cfg, sim_filepath, driver_kwargs = _common._load_optimize_config(yaml_path)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        return f"错误：配置加载失败 [{type(exc).__name__}] — {exc}"
    except Exception as exc:
        return f"错误：配置加载时出现意外错误 — {exc}"

    # 4. 解析设计变量
    design_vars_override, parse_err = _parse_design_vars_json(design_vars_json)
    if parse_err:
        return f"错误：{parse_err}"

    if design_vars_override:
        base = _build_initial_design_vars(opt_cfg)
        base.update(design_vars_override)
        design_vars_raw = base
    else:
        design_vars_raw = _build_initial_design_vars(opt_cfg)

    # 5. apply derived + repair
    try:
        design_vars = _apply_derived_and_repair(opt_cfg, design_vars_raw)
    except Exception as exc:
        return f"错误：变量预处理失败（repair/apply_derived）— {exc}"

    # 6. 连接 Aspen 并运行
    try:
        with _common._AspenDriver(**driver_kwargs) as driver:
            driver.open(sim_filepath)
            case = _common._run_case_fn(
                driver=driver,
                design_vars=design_vars,
                config=opt_cfg.run_config,
                iteration=iteration,
                tags=["agent_run_case"],
            )
    except FileNotFoundError as exc:
        return f"错误：Aspen 仿真文件不存在 — {exc}"
    except Exception as exc:
        return f"错误：Aspen 连接或运行失败 [{type(exc).__name__}] — {exc}"

    # 7. 格式化报告
    try:
        report = _fmt_case_summary(case)
    except Exception as exc:
        _log.exception("格式化 run_case 报告时出现意外错误")
        return f"错误：格式化报告时出现意外错误 — {exc}"

    _log.info(
        "run_case_tool: 完成单次运行 case_id=%s status=%s run_time=%.1fs",
        case.case_id, case.status.value, case.run_time,
    )
    return report


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def run_case_tool(
    config_path: str,
    design_vars_json: str = "",
    iteration: int = 0,
) -> str:
    """在 Aspen Plus 中执行一次单点工况评估，返回目标函数值和约束状态。

    这是需要连接 Aspen COM 的工具，每次调用消耗一次 Aspen 仿真。
    适用场景：smoke test、单点评估、失败归因。

    Args:
        config_path: YAML 配置文件路径（相对于项目根目录或绝对路径）。
        design_vars_json: 设计变量 JSON 字符串（可选）。
            格式：{"Aspen 树路径": 数值, ...}
            不传时使用各设计变量的搜索空间中间值。
        iteration: 工况迭代编号（默认 0）。

    Returns:
        包含运行状态、目标值、约束状态和综合结论的报告文本。
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_run_case(config_path, design_vars_json or None, iteration)
