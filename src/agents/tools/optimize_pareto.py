"""
optimize_pareto.py — optimize_pareto_tool 实现。

功能：执行完整的多目标 Pareto 贝叶斯优化循环，返回 Pareto 前沿和超体积报告。
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
# 结果格式化辅助函数
# ---------------------------------------------------------------------------

def _fmt_pareto_front_lines(result: Any) -> list[str]:
    """格式化 Pareto 第一前沿的解集（至多显示 10 个点，按拥挤距离降序）。"""
    lines: list[str] = []
    front = result.first_front
    if front is None or not front.cases:
        lines.append("  （无有效 Pareto 前沿）")
        return lines

    obj_names = result.objective_names
    triples = list(zip(front.cases, front.objective_vectors, front.crowding_distances))
    triples.sort(key=lambda t: t[2] if t[2] != float("inf") else 1e18, reverse=True)

    display = triples[:10]
    for i, (case, vec, cd) in enumerate(display, 1):
        obj_str = "  ".join(f"{n}={v:.4g}" for n, v in zip(obj_names, vec))
        dv_str = "  ".join(
            f"{k.split(chr(92))[-1]}={v:.4f}"
            for k, v in list(case.design_vars.items())[:4]
        )
        cd_str = f"{cd:.3f}" if cd != float("inf") else "∞"
        lines.append(f"  [{i:2d}] {obj_str}  |  {dv_str}  |  cd={cd_str}")

    if len(triples) > 10:
        lines.append(f"  ... 共 {len(triples)} 个 Pareto 前沿解（仅显示前 10 个）")
    return lines


def _fmt_hv_trend(hv_history: list, n_show: int = 6) -> str:
    """格式化超体积收敛趋势（等间隔采样，末尾点始终出现）。"""
    valid = [(i, v) for i, v in enumerate(hv_history) if v is not None]
    if not valid:
        return "  无法计算（没有足够的成功样本）"

    if len(valid) <= n_show:
        samples = valid
    else:
        step = len(valid) / n_show
        samples = [valid[int(i * step)] for i in range(n_show)]
        samples.append(valid[-1])

    parts = [f"iter{idx}={hv:.4g}" for idx, hv in samples]
    return "  " + " → ".join(parts)


def _fmt_pareto_result_summary(result: Any, config_path: str) -> str:
    """将 ParetoOptimizeResult 格式化为 agent 可读的优化报告。"""
    lines: list[str] = []
    lines.append("=== optimize_pareto 优化报告 ===")
    lines.append(f"配置文件: {config_path}")
    lines.append(f"Session : {result.session_id}")
    lines.append("")

    lines.append("【运行统计】")
    lines.append(f"  总评估次数   : {result.n_total}（含 Phase 0: {result.n_phase0}）")
    lines.append(f"  成功工况     : {result.n_success}（{result.success_rate*100:.1f}%）")
    lines.append(f"  仿真失败     : {result.n_sim_failed}")
    lines.append(f"  目标计算失败 : {result.n_objective_error}")
    lines.append(f"  初始 DOE     : {result.n_initial}")
    lines.append(
        f"  总耗时       : {result.elapsed:.1f} s"
        f"  （均值 {result.elapsed/result.n_total:.1f} s/次）"
        if result.n_total > 0 else f"  总耗时       : {result.elapsed:.1f} s"
    )
    lines.append(f"  目标函数     : {', '.join(result.objective_names)}")
    lines.append("")

    lines.append("【Pareto 前沿（第一前沿，按拥挤距离降序）】")
    lines.extend(_fmt_pareto_front_lines(result))
    lines.append("")

    pr = result.pareto_result
    front_size = len(result.first_front.cases) if result.first_front else 0
    lines.append(f"  Pareto 层数  : {pr.n_fronts}（第一前沿 {front_size} 个解）")
    lines.append("")

    lines.append("【超体积（HV）】")
    hv_final = result.hypervolume
    if hv_final is not None:
        lines.append(f"  最终 HV      : {hv_final:.6g}")
    else:
        lines.append("  最终 HV      : N/A（成功样本不足）")
    if result.hv_reference_point:
        lines.append(f"  参考点       : {[f'{v:.4g}' for v in result.hv_reference_point]}")
    lines.append("  HV 收敛趋势  :")
    lines.append(_fmt_hv_trend(result.hv_history))
    lines.append("")

    lines.append("【综合结论】")
    if result.n_success == 0:
        lines.append("  [失败] 所有工况均失败，未找到任何有效 Pareto 解。")
        lines.append("  建议：检查 Aspen 仿真文件和约束配置。")
    elif result.first_front is None:
        lines.append("  [部分完成] 有成功工况但 Pareto 前沿计算失败。")
    else:
        lines.append(
            f"  [完成] 优化结束，Pareto 第一前沿 {front_size} 个解，"
            f"成功率 {result.success_rate*100:.1f}%。"
        )
        if hv_final is not None:
            lines.append(f"  最终超体积 HV = {hv_final:.6g}。")
        if result.n_total > 0 and result.success_rate < 0.5:
            lines.append(
                f"  注意：成功率仅 {result.success_rate*100:.1f}%，"
                "可考虑放宽约束阈值或增加 Phase 0 可行性搜索次数。"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def _impl_optimize_pareto(config_path: str, db_path: str | None) -> str:
    """optimize_pareto_tool 的核心实现，出错时返回 '错误：' 字符串。"""
    # 1. 解析配置路径
    try:
        yaml_path = _common._resolve_config_path(config_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    # 2. 按需导入依赖
    err = _common._import_pareto_deps()
    if err:
        return err

    # 3. 加载配置
    try:
        opt_cfg, sim_filepath, driver_kwargs = _common._load_optimize_config(yaml_path)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        return f"错误：配置加载失败 [{type(exc).__name__}] — {exc}"
    except Exception as exc:
        return f"错误：配置加载时出现意外错误 — {exc}"

    # 4. 校验是多目标配置
    try:
        from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig
        if not isinstance(opt_cfg, ParetoOptimizeCaseConfig):
            return (
                "错误：此工具仅支持 optimizer.type: pareto_bayesian 的多目标配置。"
                f"当前配置类型为 {type(opt_cfg).__name__}。"
                "如需单目标优化，请使用其他工具。"
            )
    except ImportError as exc:
        return f"错误：无法导入 ParetoOptimizeCaseConfig — {exc}"

    # 5. 确定数据库路径
    if db_path:
        opt_cfg.db_path = db_path
    elif opt_cfg.db_path is None:
        opt_cfg.db_path = yaml_path.parent / "output" / "simulation.db"

    # 6. 运行优化
    try:
        with _common._AspenDriver(**driver_kwargs) as driver:
            driver.open(sim_filepath)
            result = _common._optimize_pareto_fn(driver=driver, config=opt_cfg)
    except FileNotFoundError as exc:
        return f"错误：Aspen 仿真文件不存在 — {exc}"
    except Exception as exc:
        return f"错误：优化运行失败 [{type(exc).__name__}] — {exc}"

    # 7. 格式化报告
    try:
        report = _fmt_pareto_result_summary(result, config_path)
    except Exception as exc:
        _log.exception("格式化 optimize_pareto 报告时出现意外错误")
        return f"错误：格式化报告时出现意外错误 — {exc}"

    _log.info(
        "optimize_pareto_tool: 完成优化 session=%s n_total=%d n_success=%d HV=%s",
        result.session_id, result.n_total, result.n_success,
        f"{result.hypervolume:.4g}" if result.hypervolume is not None else "N/A",
    )
    return report


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def optimize_pareto_tool(
    config_path: str,
    db_path: str = "",
) -> str:
    """执行多目标 Pareto 贝叶斯优化完整循环，返回 Pareto 前沿和超体积报告。

    这是计算代价最高的工具，每次调用消耗数十到数百次 Aspen 仿真。
    仅支持 optimizer.type: pareto_bayesian 的多目标配置。
    结果自动写入 SQLite 数据库。

    Args:
        config_path: YAML 配置文件路径（相对于项目根目录或绝对路径）。
            配置文件中必须设置 optimizer.type: pareto_bayesian。
        db_path: 结果数据库路径（可选）。
            不传时使用配置文件同目录的 output/simulation.db。

    Returns:
        包含运行统计、Pareto 前沿、超体积收敛趋势和综合结论的报告文本。
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_optimize_pareto(config_path, db_path or None)
