"""
validate_config.py — validate_config_tool 实现。

功能：深度校验 PAO 配置，调用完整 Python 解析链，相当于不连接 Aspen 的 dry-run。
不依赖 Aspen COM。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ._common import _load_yaml_raw, _resolve_config_path

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aspen 文件检查
# ---------------------------------------------------------------------------

def _check_sim_file(cfg: dict[str, Any], yaml_path: Path) -> dict[str, Any]:
    """检查 simulator.filepath 指向的 Aspen 文件是否存在于磁盘。"""
    raw = cfg.get("simulator", {}).get("filepath", "")
    if not raw:
        return {"exists": False, "resolved": "", "raw": "", "note": "simulator.filepath 未配置"}

    p = Path(raw)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(yaml_path.parent / p)
        # src/agents/tools/validate_config.py → 项目根
        project_root = Path(__file__).parent.parent.parent.parent
        candidates.append(project_root / p)

    for candidate in candidates:
        if candidate.exists():
            return {
                "exists": True,
                "resolved": str(candidate.resolve()),
                "raw": raw,
                "note": f"文件存在（解析路径：{candidate.resolve()}）",
            }

    return {
        "exists": False,
        "resolved": str(candidates[0]),
        "raw": raw,
        "note": (
            f"文件不存在，已检查以下路径：\n"
            + "\n".join(f"  {c}" for c in candidates)
        ),
    }


# ---------------------------------------------------------------------------
# Python 解析链调用
# ---------------------------------------------------------------------------

def _run_python_parse(yaml_path: Path) -> dict[str, Any]:
    """调用 load_optimize_config() 执行完整的 Python 层解析，捕获所有已知异常。"""
    result: dict[str, Any] = {
        "success": False,
        "opt_type": None,
        "n_vars": None,
        "n_fixed": None,
        "n_objs": None,
        "n_cons": None,
        "n_initial": None,
        "n_iterations": None,
        "n_phase0_trials": None,
        "surrogate": None,
        "error_type": None,
        "error_msg": None,
    }
    try:
        from src.utils.file_io import load_optimize_config
        from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig

        opt_cfg, _sim_path, _driver_kwargs = load_optimize_config(yaml_path)

        is_pareto = isinstance(opt_cfg, ParetoOptimizeCaseConfig)
        result["success"] = True
        result["opt_type"] = "pareto" if is_pareto else "single"
        result["n_vars"] = len(opt_cfg.param_bounds)
        result["n_fixed"] = len(opt_cfg.fixed_vars)
        result["n_objs"] = len(opt_cfg.objective_names) if is_pareto else 1
        result["n_cons"] = len(opt_cfg.run_config.constraint_fns)
        result["n_initial"] = opt_cfg.n_initial
        result["n_iterations"] = opt_cfg.n_iterations
        result["surrogate"] = opt_cfg.surrogate_model
        if is_pareto:
            fs = getattr(opt_cfg, "feasibility_search", None)
            if fs is not None and getattr(fs, "enabled", False):
                result["n_phase0_trials"] = getattr(fs, "n_trials", None)

    except (FileNotFoundError, KeyError, ValueError, TypeError, ImportError) as exc:
        result["error_type"] = type(exc).__name__
        result["error_msg"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        result["error_type"] = type(exc).__name__
        result["error_msg"] = f"（意外错误）{exc}"

    return result


# ---------------------------------------------------------------------------
# 数值合理性检查
# ---------------------------------------------------------------------------

def _check_design_var_sanity(cfg: dict[str, Any]) -> list[str]:
    """对 design_variables 做数值合理性检查。所有 float() 转换包在 try/except 内。"""
    warnings: list[str] = []
    for dv in cfg.get("design_variables", []) or []:
        name = dv.get("name", dv.get("aspen_path", "?"))
        dv_type = dv.get("type", "continuous")

        if dv_type in ("continuous", "integer"):
            lo = dv.get("lower_bound")
            hi = dv.get("upper_bound")
            if lo is not None and hi is not None:
                try:
                    if float(lo) >= float(hi):
                        warnings.append(
                            f"⚠ 设计变量 {name!r}：lower_bound={lo} >= upper_bound={hi}，"
                            "搜索空间为空。"
                        )
                except (TypeError, ValueError):
                    warnings.append(
                        f"⚠ 设计变量 {name!r}：lower_bound={lo!r} 或 upper_bound={hi!r} "
                        "无法转为数字，请检查 YAML 字段类型。"
                    )
            if dv_type == "integer":
                for key, val in [("lower_bound", lo), ("upper_bound", hi)]:
                    if val is not None:
                        try:
                            if float(val) != int(float(val)):
                                warnings.append(
                                    f"⚠ 设计变量 {name!r}（integer）：{key}={val} 不是整数，"
                                    "将被 round 到最近整数。"
                                )
                        except (TypeError, ValueError):
                            pass

        elif dv_type == "derived":
            lo_f = dv.get("lo_frac")
            hi_f = dv.get("hi_frac")
            if lo_f is not None and hi_f is not None:
                try:
                    if float(lo_f) >= float(hi_f):
                        warnings.append(
                            f"⚠ derived 变量 {name!r}：lo_frac={lo_f} >= hi_frac={hi_f}，"
                            "搜索空间为空。"
                        )
                except (TypeError, ValueError):
                    warnings.append(
                        f"⚠ derived 变量 {name!r}：lo_frac={lo_f!r} 或 hi_frac={hi_f!r} "
                        "无法转为数字。"
                    )
            frac_lo = dv.get("frac_lo")
            if frac_lo is not None:
                try:
                    v = float(frac_lo)
                    if v < 1 or v != int(v):
                        warnings.append(
                            f"⚠ derived 变量 {name!r}：frac_lo={frac_lo} 应为正整数（>= 1），"
                            "表示进料板编号下界。"
                        )
                except (TypeError, ValueError):
                    pass
            if not dv.get("depends_on"):
                warnings.append(
                    f"⚠ derived 变量 {name!r}：缺少 depends_on 字段，"
                    "无法在运行时映射为实际 FEED_STAGE。"
                )
            if not dv.get("target_path"):
                warnings.append(f"⚠ derived 变量 {name!r}：缺少 target_path 字段。")

        elif dv_type == "fixed":
            if dv.get("initial_value") is None and dv.get("lower_bound") is None:
                warnings.append(
                    f"⚠ fixed 变量 {name!r}：未设置 initial_value，"
                    "也没有 lower_bound 作为 fallback，运行时将设为 None。"
                )

    return warnings


def _check_objective_sanity(cfg: dict[str, Any]) -> list[str]:
    """对 objectives 做配置合理性检查。"""
    warnings: list[str] = []
    for i, obj in enumerate(cfg.get("objectives", []) or [], 1):
        name = obj.get("name", f"objectives[{i}]")
        obj_type = obj.get("type", "aspen_path")

        if obj_type == "aspen_path" and not obj.get("aspen_path"):
            warnings.append(f"⚠ 目标函数 {name!r}（type=aspen_path）：缺少 aspen_path 字段。")

        if obj_type == "tac":
            af = obj.get("annualization_factor", 0.1)
            try:
                if float(af) <= 0 or float(af) > 1:
                    warnings.append(
                        f"⚠ 目标函数 {name!r}（TAC）：annualization_factor={af} 通常应在 (0, 1] 范围内"
                        "（例如 0.1 表示 10 年直线折旧）。"
                    )
            except (TypeError, ValueError):
                pass
            oh = obj.get("operating_hours", 8000)
            try:
                if float(oh) <= 0 or float(oh) > 8760:
                    warnings.append(
                        f"⚠ 目标函数 {name!r}（TAC）：operating_hours={oh} 应在 (0, 8760] 小时/年范围内。"
                    )
            except (TypeError, ValueError):
                pass

        if obj_type == "emissions":
            oh = obj.get("operating_hours", 8000)
            try:
                if float(oh) <= 0 or float(oh) > 8760:
                    warnings.append(
                        f"⚠ 目标函数 {name!r}（emissions）：operating_hours={oh} 超出合理范围。"
                    )
            except (TypeError, ValueError):
                pass

    return warnings


def _check_constraint_sanity(cfg: dict[str, Any]) -> list[str]:
    """对 constraints 做配置合理性检查。"""
    warnings: list[str] = []
    valid_ops = {"<=", "<", ">=", ">", "=="}
    for i, con in enumerate(cfg.get("constraints", []) or [], 1):
        name = con.get("name", f"constraints[{i}]")
        if not con.get("aspen_path"):
            warnings.append(f"⚠ 约束 {name!r}：缺少 aspen_path 字段。")
        op = con.get("operator", "<=")
        if op not in valid_ops:
            warnings.append(
                f"⚠ 约束 {name!r}：operator={op!r} 不合法，支持 {sorted(valid_ops)}。"
            )
    return warnings


def _check_optimizer_sanity(cfg: dict[str, Any]) -> list[str]:
    """对 optimizer 节做数值合理性检查。"""
    warnings: list[str] = []
    opt = cfg.get("optimizer") or {}

    n_init = opt.get("n_initial_points", 10)
    n_iter = opt.get("n_iterations", 30)
    try:
        if int(n_init) < 1:
            warnings.append(f"⚠ optimizer.n_initial_points={n_init}，应 >= 1。")
        if int(n_iter) < 1:
            warnings.append(f"⚠ optimizer.n_iterations={n_iter}，应 >= 1。")
    except (TypeError, ValueError):
        pass

    dvs = cfg.get("design_variables", []) or []
    n_searchable = sum(
        1 for dv in dvs
        if dv.get("type", "continuous") in ("continuous", "integer", "derived")
    )
    try:
        n_init_int = int(n_init)
        if n_searchable > 0 and n_init_int < n_searchable:
            warnings.append(
                f"⚠ optimizer.n_initial_points={n_init_int} 小于搜索空间维度 {n_searchable}，"
                "建议初始 DOE 点数 >= 搜索维度（经验规则：>= 2~3 倍维度）。"
            )
    except (TypeError, ValueError):
        pass

    hv_margin = opt.get("hv_margin")
    if hv_margin is not None:
        try:
            if float(hv_margin) < 0:
                warnings.append(f"⚠ optimizer.hv_margin={hv_margin}，应 >= 0。")
        except (TypeError, ValueError):
            pass

    return warnings


# ---------------------------------------------------------------------------
# 报告组装
# ---------------------------------------------------------------------------

def _build_validate_report(
    yaml_path: Path,
    cfg: dict[str, Any],
    sim_check: dict[str, Any],
    parse_result: dict[str, Any],
) -> str:
    """将各项检查结果组装为可读报告字符串。"""
    lines: list[str] = []
    lines.append("=== PAO 配置深度校验报告 ===")
    lines.append(f"配置文件: {yaml_path}")
    lines.append("")

    lines.append("【Python 解析（load_optimize_config）】")
    if parse_result["success"]:
        lines.append("  结果     : 解析成功")
        ot = "多目标 Pareto" if parse_result["opt_type"] == "pareto" else "单目标贝叶斯"
        lines.append(f"  优化类型 : {ot}")
        lines.append(f"  搜索维度 : {parse_result['n_vars']} 个变量"
                     f"（固定 {parse_result['n_fixed']} 个）")
        lines.append(f"  目标函数 : {parse_result['n_objs']} 个")
        lines.append(f"  约束函数 : {parse_result['n_cons']} 个")
        lines.append(f"  初始 DOE : {parse_result['n_initial']} 点")
        lines.append(f"  主优化总评估数 : {parse_result['n_iterations']} 次"
                     "（= 初始 DOE + BO 迭代，不含 Phase 0）")
        if parse_result["n_phase0_trials"] is not None:
            lines.append(f"  Phase 0 可行性搜索 : 最多 {parse_result['n_phase0_trials']} 次"
                         "（额外 Aspen 调用，计入总预算）")
        lines.append(f"  代理模型 : {parse_result['surrogate']}")
    else:
        lines.append(f"  结果     : 解析失败 [{parse_result['error_type']}]")
        lines.append(f"  错误信息 : {parse_result['error_msg']}")
    lines.append("")

    lines.append("【Aspen 仿真文件检查】")
    lines.append(f"  原始路径 : {sim_check['raw'] or '（未配置）'}")
    if sim_check["exists"]:
        lines.append(f"  状态     : [存在] {sim_check['note']}")
    else:
        lines.append("  状态     : [不存在]")
        lines.append(f"  详情     : {sim_check['note']}")
        lines.append("  说明     : Aspen 文件不存在不影响 Python 配置解析，"
                     "但运行优化前必须确保文件可访问。")
    lines.append("")

    all_warns: list[str] = []
    try:
        all_warns += _check_design_var_sanity(cfg)
        all_warns += _check_objective_sanity(cfg)
        all_warns += _check_constraint_sanity(cfg)
        all_warns += _check_optimizer_sanity(cfg)
    except Exception as exc:  # noqa: BLE001
        all_warns.append(f"⚠ 合理性检查遇到意外错误（{type(exc).__name__}: {exc}），部分检查已跳过。")

    lines.append("【数值合理性检查】")
    if all_warns:
        lines.append(f"  发现 {len(all_warns)} 条警告：")
        for w in all_warns:
            lines.append(f"  {w}")
    else:
        lines.append("  [OK] 未发现合理性问题")
    lines.append("")

    lines.append("【综合结论】")
    if not parse_result["success"]:
        lines.append("  [失败] Python 解析失败，配置不可用，需修复后重试。")
        lines.append(f"  根本原因：{parse_result['error_type']}: {parse_result['error_msg']}")
    elif not sim_check["exists"] and all_warns:
        lines.append("  [警告] Python 解析通过，但 Aspen 文件不存在，且存在合理性警告。")
        lines.append("  建议：修复警告并确认 Aspen 文件路径后再启动优化。")
    elif not sim_check["exists"]:
        lines.append("  [警告] Python 解析通过，但 Aspen 文件不存在。")
        lines.append("  建议：确认 simulator.filepath 路径正确，并保证 .bkp 文件可访问。")
    elif all_warns:
        lines.append("  [警告] Python 解析通过，Aspen 文件存在，但存在合理性警告。")
        lines.append("  建议：评估上述警告后决定是否继续。")
    else:
        lines.append("  [通过] Python dry-run 通过，Aspen 文件存在，未发现合理性问题。")
        lines.append("  注意：本工具未连接 Aspen，Aspen 树路径的有效性尚未验证。")
        lines.append("  可进入 run_case smoke test 或直接启动优化。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def _impl_validate_config(config_path: str) -> str:
    """validate_config_tool 的核心实现，出错时返回 '错误：' 字符串。"""
    try:
        yaml_path = _resolve_config_path(config_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    try:
        cfg = _load_yaml_raw(yaml_path)
    except Exception as exc:
        return f"错误：YAML 解析失败 — {exc}"

    if not isinstance(cfg, dict):
        return (
            f"错误：YAML 根节点应为字典，实际类型为 {type(cfg).__name__}。"
            f"路径：{yaml_path}"
        )

    sim_check    = _check_sim_file(cfg, yaml_path)
    parse_result = _run_python_parse(yaml_path)

    try:
        report = _build_validate_report(yaml_path, cfg, sim_check, parse_result)
    except Exception as exc:
        _log.exception("构建校验报告时出现意外错误")
        return f"错误：构建报告时出现意外错误 — {exc}"

    _log.info("validate_config_tool: 完成校验 %s（解析%s）",
              yaml_path.name, "成功" if parse_result["success"] else "失败")
    return report


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def validate_config_tool(config_path: str) -> str:
    """深度校验 PAO 优化配置，相当于不启动 Aspen 的 dry-run。

    与 load_case_config_tool 的区别：
      - load_case_config_tool：只读 YAML 原始字段，做语法级检查，速度快。
      - validate_config_tool ：调用完整 Python 解析链（load_optimize_config），
        构建真实的 OptimizeCaseConfig，暴露字段缺失、类型错误、数值非法等深层问题。
        同时检查 Aspen .bkp 文件是否存在于磁盘。

    Args:
        config_path: YAML 配置文件路径（相对于项目根目录或绝对路径）。

    Returns:
        包含以下各节的校验报告文本：
          - Python 解析结果（成功/失败，及搜索维度、目标数等摘要）
          - Aspen 仿真文件存在性检查
          - 数值合理性检查（设计变量范围、目标函数参数、约束算子等）
          - 综合结论（通过 / 警告 / 失败）
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_validate_config(config_path)
