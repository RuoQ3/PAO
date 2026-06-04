"""
load_config.py — load_case_config_tool 实现。

功能：加载并解析 PAO 优化配置 YAML，返回人类可读的配置摘要。
不依赖 Aspen COM，可在任意环境中安全调用。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool, BaseTool

from ._common import _load_yaml_raw, _resolve_config_path

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 各配置段的格式化函数
# ---------------------------------------------------------------------------

def _fmt_simulator(sim: dict) -> str:
    lines = ["【仿真器配置】"]
    filepath = sim.get("filepath", "（未配置）")
    lines.append(f"  Aspen 文件   : {filepath}")
    lines.append(f"  超时时间     : {sim.get('timeout', 300)} 秒")
    lines.append(f"  可见模式     : {sim.get('visible', False)}")
    lines.append(f"  抑制对话框   : {sim.get('suppress_dialogs', True)}")
    lines.append(f"  重初始化     : {sim.get('reinit', True)}")
    lines.append(f"  类型库要求   : {sim.get('require_type_library', False)}")
    return "\n".join(lines)


def _fmt_design_variables(dvs: list) -> str:
    if not dvs:
        return "【设计变量】\n  （未配置）"

    continuous, integer, derived, fixed = [], [], [], []
    for dv in dvs:
        dv_type = dv.get("type", "continuous")
        name = dv.get("name", dv.get("aspen_path", "?"))
        if dv_type == "continuous":
            lo = dv.get("lower_bound", "?")
            hi = dv.get("upper_bound", "?")
            init = dv.get("initial_value", "?")
            unit = dv.get("unit", "")
            continuous.append(f"    {name:<30} [{lo}, {hi}]  初始值={init}  {unit}")
        elif dv_type == "integer":
            lo = dv.get("lower_bound", "?")
            hi = dv.get("upper_bound", "?")
            init = dv.get("initial_value", "?")
            integer.append(f"    {name:<30} [{lo}, {hi}]  初始值={init}  整数")
        elif dv_type == "derived":
            lo_f = dv.get("lo_frac", "?")
            hi_f = dv.get("hi_frac", "?")
            target = dv.get("target_path", "?")
            depends = dv.get("depends_on", "?")
            derived.append(
                f"    {name:<30} frac=[{lo_f}, {hi_f}]  → {target}  依赖={depends}"
            )
        else:
            val = dv.get("initial_value", dv.get("lower_bound", "?"))
            path = dv.get("aspen_path", "?")
            fixed.append(f"    {name:<30} 固定值={val}  路径={path}")

    lines = [
        f"【设计变量】  总计 {len(dvs)} 个"
        f"（连续 {len(continuous)} / 整数 {len(integer)} / 派生 {len(derived)} / 固定 {len(fixed)}）"
    ]
    if continuous:
        lines.append(f"  连续变量（{len(continuous)} 个）：")
        lines.extend(continuous)
    if integer:
        lines.append(f"  整数变量（{len(integer)} 个）：")
        lines.extend(integer)
    if derived:
        lines.append(f"  派生变量（{len(derived)} 个）：")
        lines.extend(derived)
    if fixed:
        lines.append(f"  固定变量（{len(fixed)} 个）：")
        lines.extend(fixed)
    return "\n".join(lines)


def _fmt_objectives(objs: list) -> str:
    if not objs:
        return "【目标函数】\n  （未配置）"

    lines = [f"【目标函数】  共 {len(objs)} 个"]
    for i, obj in enumerate(objs, 1):
        name = obj.get("name", "?")
        obj_type = obj.get("type", "aspen_path")
        minimize = obj.get("minimize", True)
        unit = obj.get("unit", "")
        direction = "最小化 ↓" if minimize else "最大化 ↑"

        lines.append(f"  [{i}] {name}  方向={direction}  单位={unit}  类型={obj_type}")

        if obj_type == "aspen_path":
            path = obj.get("aspen_path", "?")
            lines.append(f"      Aspen 路径 : {path}")
        elif obj_type == "tac":
            af = obj.get("annualization_factor", 0.1)
            oh = obj.get("operating_hours", 8000)
            uc = obj.get("utility_cost", {})
            lines.append(f"      折旧系数   : {af}  年操作时长={oh}h")
            if uc:
                lines.append(
                    f"      公用工程价格: 蒸汽={uc.get('steam_price', 14.19)} $/GJ  "
                    f"冷却水={uc.get('cooling_water_price', 0.354)} $/GJ  "
                    f"电力={uc.get('electricity_price', 0.0775)} $/kWh"
                )
            ep = obj.get("equipment_params", {})
            if ep:
                lines.append(f"      CEPCI      : {ep.get('cepci_current', 800.0)}")
        elif obj_type == "emissions":
            oh = obj.get("operating_hours", 8000)
            ef = obj.get("emission_factors", {})
            lines.append(f"      年操作时长  : {oh}h")
            if ef:
                lines.append(
                    f"      排放因子    : 蒸汽={ef.get('steam_factor', 66.0)} kg/GJ  "
                    f"电力={ef.get('electricity_factor', 0.581)} kg/kWh"
                )
    return "\n".join(lines)


def _fmt_constraints(cons: list) -> str:
    if not cons:
        return "【约束条件】\n  （无约束）"

    lines = [f"【约束条件】  共 {len(cons)} 个"]
    for i, con in enumerate(cons, 1):
        name = con.get("name", "?")
        path = con.get("aspen_path", "?")
        op = con.get("operator", "<=")
        threshold = con.get("threshold", 0.0)
        desc = con.get("description", "")
        lines.append(f"  [{i}] {name}  路径={path}  {op} {threshold}")
        if desc:
            lines.append(f"      说明 : {desc}")
    return "\n".join(lines)


def _fmt_optimizer(opt: dict) -> str:
    if not opt:
        return "【优化器配置】\n  （未配置，使用默认值）"

    lines = ["【优化器配置】"]
    opt_type = opt.get("type", "bayesian")
    lines.append(f"  优化类型     : {opt_type}")

    if opt_type == "pareto_bayesian":
        lines.append("  ── 多目标 Pareto 贝叶斯优化 ──")
        lines.append(f"  标量化方法   : {opt.get('scalarization', 'weighted_sum')}")

    lines.append(f"  代理模型     : {opt.get('surrogate_model', 'GP')}")
    lines.append(f"  采集函数     : {opt.get('acquisition_function', 'EI')}")
    n_init = opt.get("n_initial_points", 10)
    n_iter = opt.get("n_iterations", 30)
    lines.append(f"  初始 DOE 点数: {n_init}")
    lines.append(f"  BO 迭代次数  : {n_iter}  （总评估次数 = {n_init + n_iter}）")
    if opt.get("random_seed") is not None:
        lines.append(f"  随机种子     : {opt['random_seed']}")
    if opt.get("reference_point"):
        lines.append(f"  参考点       : {opt['reference_point']}")
    return "\n".join(lines)


def _fmt_feasibility_search(fs: dict | None) -> str | None:
    if not fs or not fs.get("enabled", False):
        return None
    lines = ["【Phase 0 可行性搜索】  已启用"]
    lines.append(f"  随机试验次数 : {fs.get('n_trials', 20)}")
    lines.append(f"  提前停止阈值 : 找到 {fs.get('stop_after_feasible', 3)} 个可行点后停止")
    return "\n".join(lines)


def _fmt_var_dependencies(deps: dict | None) -> str | None:
    if not deps:
        return None
    lines = [f"【变量依赖约束】  共 {len(deps)} 条"]
    for var_path, rules in deps.items():
        for op, ref_path in rules.items():
            op_str = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}.get(op, op)
            lines.append(f"  {var_path}  {op_str}  {ref_path}")
    return "\n".join(lines)


def _fmt_extraction(ext: dict) -> str:
    if not ext:
        return "【数据提取配置】\n  （使用默认值）"

    lines = ["【数据提取配置】"]
    mode = ext.get("mode", "full")
    lines.append(f"  提取模式     : {mode}")
    lines.append(f"  Block 深度   : {ext.get('block_max_depth', 3)}")
    lines.append(f"  Stream 深度  : {ext.get('stream_max_depth', 3)}")
    blocks = ext.get("blocks") or []
    streams = ext.get("streams") or []
    if blocks:
        lines.append(f"  提取 Block   : {', '.join(str(b) for b in blocks)}")
    if streams:
        lines.append(f"  提取 Stream  : {', '.join(str(s) for s in streams)}")
    if mode == "manifest":
        lines.append(f"  Catalog DB   : {ext.get('catalog_db', '（未指定）')}")
        lines.append(f"  Manifest ID  : {ext.get('manifest_id', 'auto')}")
    return "\n".join(lines)


def _fmt_validation_warnings(cfg: dict) -> list[str]:
    """对配置做轻量级校验，返回警告列表。"""
    warnings: list[str] = []

    sim = cfg.get("simulator", {})
    if not sim.get("filepath"):
        warnings.append("⚠ simulator.filepath 未配置，运行时将报错。")

    dvs = cfg.get("design_variables", []) or []
    searchable = [
        dv for dv in dvs
        if dv.get("type", "continuous") in ("continuous", "integer", "derived")
    ]
    if not searchable:
        warnings.append("⚠ 没有可搜索的设计变量（continuous/integer/derived），无法构建优化配置。")

    objs = cfg.get("objectives", []) or []
    if not objs:
        warnings.append("⚠ objectives 未配置，至少需要一个目标函数。")

    opt = cfg.get("optimizer", {})
    opt_type = opt.get("type", "bayesian")

    if opt_type == "pareto_bayesian" and len(objs) < 2:
        warnings.append(
            f"⚠ optimizer.type=pareto_bayesian 需要至少 2 个目标函数，当前只有 {len(objs)} 个。"
        )

    if opt_type == "bayesian" and len(objs) > 1:
        warnings.append(
            f"⚠ optimizer.type=bayesian 只使用第一个目标函数（{objs[0].get('name', '?')}），"
            f"其余 {len(objs) - 1} 个目标被忽略。"
            "如需多目标优化，请设置 optimizer.type: pareto_bayesian。"
        )

    valid_surrogates = {"GP", "RF", "ET", "GBRT", "RANDOM"}
    sm = str(opt.get("surrogate_model", "GP")).upper()
    if sm not in valid_surrogates:
        warnings.append(
            f"⚠ optimizer.surrogate_model={sm!r} 不合法，"
            f"支持值：{sorted(valid_surrogates)}。"
        )

    valid_acq = {"EI", "UCB", "PI"}
    acq = str(opt.get("acquisition_function", "EI")).upper()
    if acq not in valid_acq:
        warnings.append(
            f"⚠ optimizer.acquisition_function={acq!r} 不合法，支持值：{sorted(valid_acq)}。"
        )

    ref = opt.get("reference_point")
    if ref is not None and opt_type == "pareto_bayesian":
        if len(ref) != len(objs):
            warnings.append(
                f"⚠ optimizer.reference_point 维度 {len(ref)} 与目标数 {len(objs)} 不匹配。"
            )

    return warnings


def _build_config_summary(cfg: dict, yaml_path: Path) -> str:
    """将 YAML 原始字典格式化为 agent 可读的配置摘要字符串。"""
    sections: list[str] = []

    sections.append("=== PAO 优化配置摘要 ===")
    sections.append(f"配置文件: {yaml_path}")

    sections.append(_fmt_simulator(cfg.get("simulator", {})))
    sections.append(_fmt_design_variables(cfg.get("design_variables") or []))
    sections.append(_fmt_objectives(cfg.get("objectives") or []))
    sections.append(_fmt_constraints(cfg.get("constraints") or []))
    sections.append(_fmt_optimizer(cfg.get("optimizer") or {}))

    fs_str = _fmt_feasibility_search(cfg.get("feasibility_search"))
    if fs_str:
        sections.append(fs_str)

    deps_str = _fmt_var_dependencies(cfg.get("var_dependencies"))
    if deps_str:
        sections.append(deps_str)

    sections.append(_fmt_extraction(cfg.get("extraction") or {}))

    out_paths = cfg.get("output_paths") or []
    if out_paths:
        sections.append(f"【output_paths】  共 {len(out_paths)} 条")
        for p in out_paths:
            sections.append(f"  {p}")

    warnings = _fmt_validation_warnings(cfg)
    if warnings:
        sections.append("【配置校验警告】")
        sections.extend(f"  {w}" for w in warnings)
    else:
        sections.append("【配置校验】  [OK] 未发现明显问题")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 核心实现（与 @tool 解耦，方便单元测试）
# ---------------------------------------------------------------------------

def _impl_load_config(config_path: str) -> str:
    """
    load_case_config_tool 的核心实现。

    出错时返回以 "错误：" 开头的字符串，不抛异常。
    """
    try:
        yaml_path = _resolve_config_path(config_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    try:
        cfg = _load_yaml_raw(yaml_path)
    except Exception as exc:
        return f"错误：YAML 解析失败 — {exc}"

    if not isinstance(cfg, dict):
        return f"错误：YAML 文件根节点应为字典，实际类型为 {type(cfg).__name__}。路径：{yaml_path}"

    try:
        summary = _build_config_summary(cfg, yaml_path)
    except Exception as exc:
        _log.exception("构建配置摘要时出现意外错误")
        return f"错误：构建摘要时出现意外错误 — {exc}"

    _log.info("load_case_config_tool: 已加载配置 %s", yaml_path.name)
    return summary


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def load_case_config_tool(config_path: str) -> str:
    """加载并解析 PAO 优化配置 YAML 文件，返回结构化的配置摘要。

    用途：在开始优化或分析任务前，先调用此工具了解当前配置的设计变量、
    目标函数、约束条件和优化器参数，为后续决策提供依据。

    Args:
        config_path: YAML 配置文件路径（相对于项目根目录或绝对路径）。
            例如：
              "cases/demo_case/pareto_config.yaml"
              "cases/demo_case_2/pareto_config.yaml"

    Returns:
        包含以下各节的配置摘要文本：
          - 仿真器配置（Aspen 文件路径、超时时间等）
          - 设计变量（类型、范围、初始值）
          - 目标函数（类型、方向、参数）
          - 约束条件（路径、算子、阈值）
          - 优化器配置（类型、代理模型、迭代次数）
          - 可行性搜索配置（若启用）
          - 变量依赖约束（若有）
          - 数据提取配置
          - 配置校验警告（若有问题）
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_load_config(config_path)


# 向后兼容别名
load_config_tool: BaseTool = load_case_config_tool
