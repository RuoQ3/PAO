"""
tools.py — PAO Agent 工具集合。

每个工具遵循 LangChain @tool 规范，供 LangGraph agent 节点调用。
工具职责：将 PAO 核心功能封装为 LLM 可推理的接口，返回结构化文本。

当前已实现：
  load_case_config_tool  — 加载并解析优化配置 YAML，返回人类可读的配置摘要。
                           不依赖 Aspen COM，可在任意环境中安全调用。
  validate_config_tool   — 深度校验配置：调用完整 Python 解析链，检查字段合法性、
                           数值合理性、Aspen 文件是否存在，相当于不连接 Aspen 的 dry-run。
                           不依赖 Aspen COM。
  run_case_tool          — 在 Aspen Plus 中执行一次单点工况评估，返回目标函数值和
                           约束状态。需要 Windows + Aspen Plus + pywin32 环境。
  optimize_pareto_tool   — 执行完整的多目标 Pareto 贝叶斯优化循环，返回 Pareto
                           前沿、超体积和运行统计。需要 Aspen 环境。

模块级导出：
  load_case_config_tool  — langchain_core.tools.BaseTool 实例（无需 Aspen）
  load_config_tool       — load_case_config_tool 的向后兼容别名
  validate_config_tool   — langchain_core.tools.BaseTool 实例（无需 Aspen）
  run_case_tool          — langchain_core.tools.BaseTool 实例（需要 Aspen COM）
  optimize_pareto_tool   — langchain_core.tools.BaseTool 实例（需要 Aspen COM）
  get_agent_tools()      — 返回所有 PAO agent 工具的列表，供 graph 统一注册。

调用方式（LangGraph node 内）：
    from src.agents.tools import get_agent_tools

    tools = get_agent_tools()
    tool_node = ToolNode(tools)
    model = ChatAnthropic(...).bind_tools(tools)

依赖：langchain-core>=1.4.0, langgraph>=1.2.0（见 requirements.txt agent 段）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool, BaseTool

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aspen 工具的可替换依赖引用（方便测试时 monkeypatch）
# 实际运行时由各 _import_*_deps() 在首次调用时按需 import 并赋值，
# 未安装 pywin32 的环境导入 tools 模块本身不会报错。
# 测试中通过 patch("src.agents.tools._xxx", ...) 等路径打桩。
# ---------------------------------------------------------------------------
_load_optimize_config: Any = None        # src.utils.file_io.load_optimize_config
_AspenDriver: Any = None                 # src.aspen_driver.driver.AspenDriver
_run_case_fn: Any = None                 # src.workflows.run_case.run_case
_optimize_pareto_fn: Any = None          # src.workflows.optimize_pareto_case.optimize_pareto_case


def _import_run_time_deps() -> str | None:
    """
    按需导入 run_case_tool 运行时依赖（load_optimize_config、AspenDriver、run_case）。
    成功时更新模块级引用并返回 None；失败时返回错误字符串。
    """
    global _load_optimize_config, _AspenDriver, _run_case_fn
    try:
        from src.utils.file_io import load_optimize_config
        _load_optimize_config = load_optimize_config
    except ImportError as exc:
        return f"错误：无法导入 load_optimize_config — {exc}"
    try:
        from src.aspen_driver.driver import AspenDriver
        _AspenDriver = AspenDriver
    except ImportError as exc:
        return f"错误：无法导入 AspenDriver（请确认在 Windows + pywin32 环境中运行）— {exc}"
    try:
        from src.workflows.run_case import run_case
        _run_case_fn = run_case
    except ImportError as exc:
        return f"错误：无法导入 run_case — {exc}"
    return None


def _import_pareto_deps() -> str | None:
    """
    按需导入 optimize_pareto_tool 运行时依赖。
    成功时更新模块级引用并返回 None；失败时返回错误字符串。
    """
    global _load_optimize_config, _AspenDriver, _optimize_pareto_fn
    try:
        from src.utils.file_io import load_optimize_config
        _load_optimize_config = load_optimize_config
    except ImportError as exc:
        return f"错误：无法导入 load_optimize_config — {exc}"
    try:
        from src.aspen_driver.driver import AspenDriver
        _AspenDriver = AspenDriver
    except ImportError as exc:
        return f"错误：无法导入 AspenDriver（请确认在 Windows + pywin32 环境中运行）— {exc}"
    try:
        from src.workflows.optimize_pareto_case import optimize_pareto_case
        _optimize_pareto_fn = optimize_pareto_case
    except ImportError as exc:
        return f"错误：无法导入 optimize_pareto_case — {exc}"
    return None


# ---------------------------------------------------------------------------
# load_case_config_tool 实现
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# load_config_tool
# ---------------------------------------------------------------------------

def _load_yaml_raw(yaml_path: Path) -> dict[str, Any]:
    """读取 YAML 文件，返回原始字典。"""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML 未安装，请执行：pip install pyyaml\n"
            f"原始错误：{exc}"
        ) from exc

    with yaml_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(config_path: str) -> Path:
    """
    解析配置路径，优先级：
    1. 绝对路径直接使用
    2. 相对于当前工作目录
    3. 相对于项目根目录（往上查找 src/ 同级目录）
    """
    p = Path(config_path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在：{p}")
        return p

    # 相对路径：先尝试 cwd
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd.resolve()

    # 再尝试从本文件位置推断项目根（src/agents/tools.py → 项目根）
    project_root = Path(__file__).parent.parent.parent
    from_root = project_root / p
    if from_root.exists():
        return from_root.resolve()

    raise FileNotFoundError(
        f"配置文件不存在：{config_path!r}\n"
        f"  已尝试：\n"
        f"    {from_cwd}\n"
        f"    {from_root}"
    )


# ── 各配置段的格式化函数 ────────────────────────────────────────────────────

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
    """
    对配置做轻量级校验，返回警告列表。
    不执行重型校验（不连接 Aspen，不解析 Python 目标函数）。
    """
    warnings: list[str] = []

    # 检查仿真文件字段
    sim = cfg.get("simulator", {})
    if not sim.get("filepath"):
        warnings.append("⚠ simulator.filepath 未配置，运行时将报错。")

    # 检查设计变量
    dvs = cfg.get("design_variables", []) or []
    searchable = [
        dv for dv in dvs
        if dv.get("type", "continuous") in ("continuous", "integer", "derived")
    ]
    if not searchable:
        warnings.append("⚠ 没有可搜索的设计变量（continuous/integer/derived），无法构建优化配置。")

    # 检查目标函数
    objs = cfg.get("objectives", []) or []
    if not objs:
        warnings.append("⚠ objectives 未配置，至少需要一个目标函数。")

    opt = cfg.get("optimizer", {})
    opt_type = opt.get("type", "bayesian")

    # 多目标检查
    if opt_type == "pareto_bayesian" and len(objs) < 2:
        warnings.append(
            f"⚠ optimizer.type=pareto_bayesian 需要至少 2 个目标函数，当前只有 {len(objs)} 个。"
        )

    # 单目标但配置了多个目标函数
    if opt_type == "bayesian" and len(objs) > 1:
        warnings.append(
            f"⚠ optimizer.type=bayesian 只使用第一个目标函数（{objs[0].get('name', '?')}），"
            f"其余 {len(objs) - 1} 个目标被忽略。"
            "如需多目标优化，请设置 optimizer.type: pareto_bayesian。"
        )

    # 代理模型校验
    valid_surrogates = {"GP", "RF", "ET", "GBRT", "RANDOM"}
    sm = str(opt.get("surrogate_model", "GP")).upper()
    if sm not in valid_surrogates:
        warnings.append(
            f"⚠ optimizer.surrogate_model={sm!r} 不合法，"
            f"支持值：{sorted(valid_surrogates)}。"
        )

    # 采集函数校验
    valid_acq = {"EI", "UCB", "PI"}
    acq = str(opt.get("acquisition_function", "EI")).upper()
    if acq not in valid_acq:
        warnings.append(
            f"⚠ optimizer.acquisition_function={acq!r} 不合法，支持值：{sorted(valid_acq)}。"
        )

    # 参考点维度检查（pareto 模式）
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

    # 标题
    sections.append(f"=== PAO 优化配置摘要 ===")
    sections.append(f"配置文件: {yaml_path}")

    # 各配置段
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

    # 输出路径
    out_paths = cfg.get("output_paths") or []
    if out_paths:
        sections.append(f"【output_paths】  共 {len(out_paths)} 条")
        for p in out_paths:
            sections.append(f"  {p}")

    # 校验警告
    warnings = _fmt_validation_warnings(cfg)
    if warnings:
        sections.append("【配置校验警告】")
        sections.extend(f"  {w}" for w in warnings)
    else:
        sections.append("【配置校验】  [OK] 未发现明显问题")

    return "\n\n".join(sections)


def _impl_load_config(config_path: str) -> str:
    """
    load_config_tool 的核心实现，与 @tool 装饰器解耦，方便单元测试。

    Parameters
    ----------
    config_path:
        YAML 配置文件路径（相对路径或绝对路径）。

    Returns
    -------
    str
        人类可读的配置摘要，供 LLM 理解和推理。
        出错时返回以 "错误：" 开头的错误描述字符串（不抛异常，让 agent 自行处理）。
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
# 模块级工具定义（真实 BaseTool，可直接传入 ToolNode / bind_tools）
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


# ---------------------------------------------------------------------------
# validate_config_tool 实现
# ---------------------------------------------------------------------------
# 与 load_case_config_tool 的区别：
#   load_case_config_tool — 只读 YAML 原始字典，做轻量级语义检查，速度快。
#   validate_config_tool  — 调用完整的 load_optimize_config() Python 解析链，
#                           构建真实的 OptimizeCaseConfig / ParetoOptimizeCaseConfig，
#                           暴露字段缺失、类型错误、数值非法、函数生成失败等深层问题。
#                           不启动 Aspen，但会检查 .bkp 文件是否存在于磁盘。
# ---------------------------------------------------------------------------

def _check_sim_file(cfg: dict[str, Any], yaml_path: Path) -> dict[str, Any]:
    """
    检查 simulator.filepath 指向的 Aspen 文件是否存在于磁盘。

    返回字典包含：
      exists      : bool
      resolved    : str  — 解析后的绝对路径（若能解析）
      raw         : str  — YAML 中原始值
      note        : str  — 附加说明（例如路径解析策略）
    """
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
        # 再尝试从 src/agents/tools.py 推断项目根
        project_root = Path(__file__).parent.parent.parent
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


def _run_python_parse(yaml_path: Path) -> dict[str, Any]:
    """
    调用 load_optimize_config() 执行完整的 Python 层解析。

    返回字典包含：
      success          : bool
      opt_type         : str | None  — "single" 或 "pareto"
      n_vars           : int | None  — 搜索空间维度（param_bounds 数量）
      n_fixed          : int | None  — fixed_vars 数量
      n_objs           : int | None  — 目标函数数量
      n_cons           : int | None  — 约束函数数量
      n_initial        : int | None
      n_iterations     : int | None  — 主优化总评估数（不含 Phase 0）
      n_phase0_trials  : int | None  — Phase 0 可行性搜索最大试验次数（未启用时为 None）
      surrogate        : str | None
      error_type       : str | None  — 异常类名（仅 success=False 时）
      error_msg        : str | None  — 异常消息（仅 success=False 时）
    """
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
        result["n_objs"] = (
            len(opt_cfg.objective_names) if is_pareto else 1
        )
        result["n_cons"] = len(opt_cfg.run_config.constraint_fns)
        result["n_initial"] = opt_cfg.n_initial
        result["n_iterations"] = opt_cfg.n_iterations
        result["surrogate"] = opt_cfg.surrogate_model
        # Phase 0 可行性搜索（仅 Pareto 模式且启用时有值）
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


def _check_design_var_sanity(cfg: dict[str, Any]) -> list[str]:
    """
    对 design_variables 做数值合理性检查，返回警告列表。

    所有 float() 转换都包在 try/except 内，保证此函数本身永远不抛异常，
    只产生警告字符串。

    检查项：
      - lower_bound >= upper_bound（搜索空间为空）
      - integer 变量的 lower/upper 不是整数
      - derived 变量的 lo_frac >= hi_frac
      - derived 变量的 frac_lo 不是正整数
      - derived 变量缺少 depends_on / target_path
      - fixed 变量没有 initial_value（也没有 lower_bound 作为 fallback）
    """
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
                warnings.append(
                    f"⚠ derived 变量 {name!r}：缺少 target_path 字段。"
                )

        elif dv_type == "fixed":
            if dv.get("initial_value") is None and dv.get("lower_bound") is None:
                warnings.append(
                    f"⚠ fixed 变量 {name!r}：未设置 initial_value，"
                    "也没有 lower_bound 作为 fallback，运行时将设为 None。"
                )

    return warnings


def _check_objective_sanity(cfg: dict[str, Any]) -> list[str]:
    """对 objectives 做配置合理性检查，返回警告列表。"""
    warnings: list[str] = []
    for i, obj in enumerate(cfg.get("objectives", []) or [], 1):
        name = obj.get("name", f"objectives[{i}]")
        obj_type = obj.get("type", "aspen_path")

        if obj_type == "aspen_path" and not obj.get("aspen_path"):
            warnings.append(
                f"⚠ 目标函数 {name!r}（type=aspen_path）：缺少 aspen_path 字段。"
            )

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
    """对 constraints 做配置合理性检查，返回警告列表。"""
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
    """对 optimizer 节做数值合理性检查，返回警告列表。"""
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

    # 搜索空间维度 vs 初始样本数的经验检查
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

    # ── 1. Python 解析结果 ────────────────────────────────────────────────
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

    # ── 2. Aspen 文件检查 ──────────────────────────────────────────────────
    lines.append("【Aspen 仿真文件检查】")
    lines.append(f"  原始路径 : {sim_check['raw'] or '（未配置）'}")
    if sim_check["exists"]:
        lines.append(f"  状态     : [存在] {sim_check['note']}")
    else:
        lines.append(f"  状态     : [不存在]")
        lines.append(f"  详情     : {sim_check['note']}")
        lines.append("  说明     : Aspen 文件不存在不影响 Python 配置解析，"
                     "但运行优化前必须确保文件可访问。")
    lines.append("")

    # ── 3. 数值合理性检查 ─────────────────────────────────────────────────
    # 解析失败时以 best-effort 模式运行，sanity check 内部异常不向上抛。
    # 注意：各 _check_*_sanity() 函数本身已做防御性 try/except；
    # 此处额外包裹是第二道保险，防止未预料的异常让报告构建失败。
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

    # ── 4. 综合结论 ────────────────────────────────────────────────────────
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


def _impl_validate_config(config_path: str) -> str:
    """
    validate_config_tool 的核心实现，与 @tool 装饰器解耦，方便单元测试。

    执行步骤：
      1. 解析配置文件路径
      2. 读取 YAML 原始字典（复用 _load_yaml_raw / _resolve_config_path）
      3. 检查 Aspen 仿真文件是否存在于磁盘
      4. 调用 load_optimize_config() 执行完整 Python 解析链
      5. 对设计变量、目标函数、约束、优化器做数值合理性检查
      6. 输出综合报告

    Parameters
    ----------
    config_path:
        YAML 配置文件路径（相对路径或绝对路径）。

    Returns
    -------
    str
        校验报告文本。出错时返回以 "错误：" 开头的字符串。
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
# 模块级工具定义
# ---------------------------------------------------------------------------

@tool
def validate_config_tool(config_path: str) -> str:
    """深度校验 PAO 优化配置，相当于不启动 Aspen 的 dry-run。

    与 load_case_config_tool 的区别：
      - load_case_config_tool：只读 YAML 原始字段，做语法级检查，速度快。
      - validate_config_tool ：调用完整 Python 解析链（load_optimize_config），
        构建真实的 OptimizeCaseConfig，暴露 load_case_config_tool 看不到的问题：
        字段缺失、类型错误、数值非法、目标/约束函数生成失败等。
        同时检查 Aspen .bkp 文件是否存在于磁盘。

    推荐工作流：先 load_case_config_tool 了解配置全貌，
    再 validate_config_tool 确认配置可以运行。

    Args:
        config_path: YAML 配置文件路径（相对于项目根目录或绝对路径）。
            例如：
              "cases/demo_case/pareto_config.yaml"
              "cases/demo_case_2/pareto_config.yaml"

    Returns:
        包含以下各节的校验报告文本：
          - Python 解析结果（成功/失败，及搜索维度、目标数等摘要）
          - Aspen 仿真文件存在性检查
          - 数值合理性检查（设计变量范围、目标函数参数、约束算子等）
          - 综合结论（通过 / 警告 / 失败）
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_validate_config(config_path)


# ---------------------------------------------------------------------------
# run_case_tool 实现
# ---------------------------------------------------------------------------
# 这是三个工具里唯一会真正连接 Aspen COM 的工具。
# 设计边界：
#   - 输入：config_path（必须） + design_vars_json（可选 JSON 字符串）
#           + iteration（可选整数）
#   - 行为：load_optimize_config → AspenDriver.open → run_case() × 1 次
#   - 输出：ProcessCase 摘要（状态、目标值、约束值、耗时、失败诊断）
#   - 错误：COM 连接失败、仿真失败、目标不可用均以结构化文本返回，不向上抛异常
#
# 与优化循环的区别：run_case_tool 只跑一次，不做代理模型拟合，
# 适合 smoke test、参数验证和单点评估场景。
# ---------------------------------------------------------------------------

def _parse_design_vars_json(design_vars_json: str | None) -> tuple[dict, str | None]:
    """
    将 design_vars_json 字符串解析为 dict。

    Returns
    -------
    (design_vars, error_msg)
        成功时 error_msg=None；失败时 design_vars={}，error_msg 为原因。
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

    # 确保所有 value 都是数字
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
    从 OptimizeCaseConfig / ParetoOptimizeCaseConfig 中提取初始设计变量点：
      - 连续/整数/derived 变量：取各维度的中间值 (lo + hi) / 2
      - fixed 变量：直接使用 fixed_vars 中的值

    这是在用户未指定 design_vars 时的合理默认值，
    不是优化意义上的"最优"，仅用于配置验证和 smoke test。
    """
    result: dict = {}

    # fixed 变量直接使用
    result.update(opt_cfg.fixed_vars)

    # 搜索变量取中间值
    for path, (lo, hi) in opt_cfg.param_bounds.items():
        result[path] = (lo + hi) / 2.0

    # derived 变量：frac 取中间值后，再通过 apply_derived_vars 展开
    # （这里先把 frac 中间值放进去，后续 repair/apply 阶段会处理）
    return result


def _apply_derived_and_repair(opt_cfg: Any, design_vars: dict) -> dict:
    """
    对 design_vars 依次执行：
      1. repair_design_vars：round/clamp/依赖约束修复
      2. apply_derived_vars：将 derived frac 变量展开为真实 Aspen 路径

    这复现了优化循环中调用 run_case 之前的预处理步骤，
    与 optimize_case.py / optimize_pareto_case.py 的调用模式一致。
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


def _fmt_case_summary(case: Any) -> str:
    """将 ProcessCase 格式化为 agent 可读的运行报告。"""
    lines: list[str] = []
    lines.append("=== run_case 单次运行报告 ===")
    lines.append("")

    # 基本信息
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

    # 设计变量
    lines.append("【设计变量（输入点）】")
    for path, val in case.design_vars.items():
        short = path.split("\\")[-1] if "\\" in path else path
        lines.append(f"  {short:<30} = {val}")
    lines.append("")

    # 目标函数
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

    # 约束
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

    # 仿真详情（失败时展示）
    if not case.simulation_valid and case.sim_result is not None:
        lines.append("【仿真失败详情】")
        sr = case.sim_result
        lines.append(f"  引擎状态  : {sr.status.value}")
        if sr.error:
            lines.append(f"  错误信息  : {sr.error}")
        if getattr(sr, "warnings", None):
            for w in sr.warnings[:5]:   # 最多显示 5 条
                lines.append(f"  警告      : {w}")
        lines.append("")

    # notes（含 block/stream 提取失败信息）
    if case.notes:
        lines.append("【运行注记（block/stream 提取失败等）】")
        for note_line in case.notes.split("\n")[:10]:  # 最多 10 行
            lines.append(f"  {note_line}")
        lines.append("")

    # 综合结论
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


def _impl_run_case(
    config_path: str,
    design_vars_json: str | None,
    iteration: int,
) -> str:
    """
    run_case_tool 的核心实现，与 @tool 装饰器解耦，方便单元测试。

    执行步骤：
      1. 解析 config_path，加载 OptimizeCaseConfig
      2. 解析 design_vars_json（可选），未提供时使用参数空间中间值
      3. apply derived 变量展开和 repair 修复
      4. AspenDriver.open → run_case() 一次
      5. 格式化 ProcessCase 摘要并返回

    Parameters
    ----------
    config_path:
        YAML 配置文件路径。
    design_vars_json:
        设计变量 JSON 字符串，格式 {"aspen_path": value, ...}。
        为空时使用各维度中间值。
    iteration:
        工况迭代编号（默认 0，用于标记）。

    Returns
    -------
    str
        运行报告文本。出错时返回以 "错误：" 开头的字符串。
    """
    # 1. 解析配置路径
    try:
        yaml_path = _resolve_config_path(config_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    # 2. 按需导入运行时依赖（首次调用时 import，后续直接使用模块级引用）
    # 测试时通过 patch("src.agents.tools._load_optimize_config", ...) 等打桩
    err = _import_run_time_deps()
    if err:
        return err

    # 3. 加载 OptimizeCaseConfig
    try:
        opt_cfg, sim_filepath, driver_kwargs = _load_optimize_config(yaml_path)
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
        with _AspenDriver(**driver_kwargs) as driver:
            driver.open(sim_filepath)
            case = _run_case_fn(
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

    # 6. 格式化报告
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


@tool
def run_case_tool(
    config_path: str,
    design_vars_json: str = "",
    iteration: int = 0,
) -> str:
    """在 Aspen Plus 中执行一次单点工况评估，返回目标函数值和约束状态。

    这是三个工具里唯一会真正连接 Aspen COM 的工具，每次调用消耗一次 Aspen 仿真。
    适用场景：
      - smoke test（验证配置能否成功运行一次）
      - 单点评估（验证某组参数的目标函数值）
      - 失败归因（在具体参数点观察仿真失败原因）

    注意：此工具需要 Aspen Plus 已安装的 Windows 环境。

    Args:
        config_path: YAML 配置文件路径（相对于项目根目录或绝对路径）。
            例如："cases/demo_case_2/pareto_config.yaml"

        design_vars_json: 设计变量 JSON 字符串（可选）。
            格式：{"Aspen 树路径": 数值, ...}
            例如：
              '{"\\\\Data\\\\Blocks\\\\T1\\\\Input\\\\BASIS_RR": 0.5}'
            不传或传空字符串时，使用各设计变量的搜索空间中间值作为默认点。

        iteration: 工况迭代编号（默认 0），用于标记和数据库分类。

    Returns:
        包含以下各节的运行报告文本：
          - 运行状态（case_id、status、success、feasible、run_time）
          - 设计变量（输入点）
          - 目标函数值（含单位和优化方向）
          - 约束状态（满足/违反）
          - 仿真失败详情（仅失败时）
          - 综合结论（成功/仿真失败/不可行/目标错误）
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_run_case(config_path, design_vars_json or None, iteration)


# ---------------------------------------------------------------------------
# optimize_pareto_tool 实现
# ---------------------------------------------------------------------------
# 调用完整的 optimize_pareto_case() 循环，执行多目标贝叶斯优化。
# 与 run_case_tool 的区别：
#   run_case_tool       — 单点评估，1 次 Aspen 调用，用于 smoke test。
#   optimize_pareto_tool — 完整优化循环，n_initial + n_iterations 次 Aspen 调用。
# ---------------------------------------------------------------------------

def _fmt_pareto_front_lines(result: Any) -> list[str]:
    """格式化 Pareto 第一前沿的解集（至多显示 10 个点）。"""
    lines: list[str] = []
    front = result.first_front
    if front is None or not front.cases:
        lines.append("  （无有效 Pareto 前沿）")
        return lines

    obj_names = result.objective_names
    # 按拥挤距离降序排列，优先展示多样性好的点
    triples = list(zip(front.cases, front.objective_vectors, front.crowding_distances))
    triples.sort(key=lambda t: t[2] if t[2] != float("inf") else 1e18, reverse=True)

    display = triples[:10]
    for i, (case, vec, cd) in enumerate(display, 1):
        obj_str = "  ".join(
            f"{n}={v:.4g}" for n, v in zip(obj_names, vec)
        )
        dv_str = "  ".join(
            f"{k.split(chr(92))[-1]}={v:.4f}"
            for k, v in list(case.design_vars.items())[:4]   # 最多显示 4 个变量
        )
        cd_str = f"{cd:.3f}" if cd != float("inf") else "∞"
        lines.append(f"  [{i:2d}] {obj_str}  |  {dv_str}  |  cd={cd_str}")

    if len(triples) > 10:
        lines.append(f"  ... 共 {len(triples)} 个 Pareto 前沿解（仅显示前 10 个）")
    return lines


def _fmt_hv_trend(hv_history: list, n_show: int = 6) -> str:
    """格式化超体积收敛趋势（取等间隔采样点）。"""
    valid = [(i, v) for i, v in enumerate(hv_history) if v is not None]
    if not valid:
        return "  无法计算（没有足够的成功样本）"

    # 等间隔采样至多 n_show 个点
    if len(valid) <= n_show:
        samples = valid
    else:
        step = len(valid) / n_show
        samples = [valid[int(i * step)] for i in range(n_show)]
        samples.append(valid[-1])   # 确保最后一个点总在

    parts = [f"iter{idx}={hv:.4g}" for idx, hv in samples]
    return "  " + " → ".join(parts)


def _fmt_pareto_result_summary(result: Any, config_path: str) -> str:
    """将 ParetoOptimizeResult 格式化为 agent 可读的优化报告。"""
    lines: list[str] = []
    lines.append("=== optimize_pareto 优化报告 ===")
    lines.append(f"配置文件: {config_path}")
    lines.append(f"Session : {result.session_id}")
    lines.append("")

    # ── 1. 运行统计 ────────────────────────────────────────────────────────
    lines.append("【运行统计】")
    lines.append(f"  总评估次数   : {result.n_total}（含 Phase 0: {result.n_phase0}）")
    lines.append(f"  成功工况     : {result.n_success}（{result.success_rate*100:.1f}%）")
    lines.append(f"  仿真失败     : {result.n_sim_failed}")
    lines.append(f"  目标计算失败 : {result.n_objective_error}")
    lines.append(f"  初始 DOE     : {result.n_initial}")
    lines.append(f"  总耗时       : {result.elapsed:.1f} s"
                 f"  （均值 {result.elapsed/result.n_total:.1f} s/次）"
                 if result.n_total > 0 else f"  总耗时       : {result.elapsed:.1f} s")
    lines.append(f"  目标函数     : {', '.join(result.objective_names)}")
    lines.append("")

    # ── 2. Pareto 前沿 ─────────────────────────────────────────────────────
    lines.append("【Pareto 前沿（第一前沿，按拥挤距离降序）】")
    lines.extend(_fmt_pareto_front_lines(result))
    lines.append("")

    # Pareto 层数
    pr = result.pareto_result
    lines.append(f"  Pareto 层数  : {pr.n_fronts}（第一前沿 {len(result.first_front.cases) if result.first_front else 0} 个解）")
    lines.append("")

    # ── 3. 超体积 ─────────────────────────────────────────────────────────
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

    # ── 4. 综合结论 ────────────────────────────────────────────────────────
    lines.append("【综合结论】")
    if result.n_success == 0:
        lines.append("  [失败] 所有工况均失败，未找到任何有效 Pareto 解。")
        lines.append("  建议：检查 Aspen 仿真文件和约束配置。")
    elif result.first_front is None:
        lines.append("  [部分完成] 有成功工况但 Pareto 前沿计算失败。")
    else:
        front_size = len(result.first_front.cases)
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


def _impl_optimize_pareto(config_path: str, db_path: str | None) -> str:
    """
    optimize_pareto_tool 的核心实现，与 @tool 装饰器解耦，方便单元测试。

    执行步骤：
      1. 解析 config_path，校验是多目标配置（pareto_bayesian）
      2. 按需导入 optimize_pareto_case / AspenDriver
      3. 若指定 db_path，覆盖配置中的 db_path
      4. AspenDriver.open → optimize_pareto_case() 完整循环
      5. 格式化 ParetoOptimizeResult 报告并返回

    Parameters
    ----------
    config_path:
        YAML 配置文件路径。必须配置了 optimizer.type: pareto_bayesian。
    db_path:
        结果数据库路径（可选）。传入时覆盖配置文件中的 db_path；
        不传时使用配置文件同目录的 output/simulation.db。

    Returns
    -------
    str
        优化报告文本。出错时返回以 "错误：" 开头的字符串。
    """
    # 1. 解析配置路径
    try:
        yaml_path = _resolve_config_path(config_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    # 2. 按需导入依赖
    err = _import_pareto_deps()
    if err:
        return err

    # 3. 加载配置
    try:
        opt_cfg, sim_filepath, driver_kwargs = _load_optimize_config(yaml_path)
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
        # 默认：与 YAML 同目录的 output/simulation.db
        opt_cfg.db_path = yaml_path.parent / "output" / "simulation.db"

    # 6. 运行优化
    try:
        with _AspenDriver(**driver_kwargs) as driver:
            driver.open(sim_filepath)
            result = _optimize_pareto_fn(driver=driver, config=opt_cfg)
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


@tool
def optimize_pareto_tool(
    config_path: str,
    db_path: str = "",
) -> str:
    """执行多目标 Pareto 贝叶斯优化完整循环，返回 Pareto 前沿和超体积报告。

    这是计算代价最高的工具——每次调用会执行完整的贝叶斯优化循环
    （初始 DOE + 多轮 BO 迭代），消耗数十到数百次 Aspen 仿真。
    调用前请先用 validate_config_tool 确认配置正确。

    仅支持 optimizer.type: pareto_bayesian 的多目标配置。
    结果自动写入 SQLite 数据库（可通过 db_path 指定路径）。

    Args:
        config_path: YAML 配置文件路径（相对于项目根目录或绝对路径）。
            例如："cases/demo_case_2/pareto_config.yaml"
            配置文件中必须设置 optimizer.type: pareto_bayesian。

        db_path: 结果数据库路径（可选）。
            不传或传空字符串时，使用配置文件同目录的 output/simulation.db。
            例如："cases/demo_case_2/output/run1.db"

    Returns:
        包含以下各节的优化报告文本：
          - 运行统计（总次数、成功率、耗时、Phase 0 次数）
          - Pareto 第一前沿（目标值、设计变量、拥挤距离，按拥挤距离降序）
          - 超体积（最终 HV、参考点、HV 收敛趋势）
          - 综合结论（完成/失败/低成功率警告）
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_optimize_pareto(config_path, db_path or None)


def get_agent_tools() -> list[BaseTool]:
    """返回所有 PAO agent 工具列表，供 graph 统一注册。

    用法：
        from src.agents.tools import get_agent_tools
        from langgraph.prebuilt import ToolNode

        tools = get_agent_tools()
        tool_node = ToolNode(tools)
        model = ChatAnthropic(...).bind_tools(tools)
    """
    return [load_case_config_tool, validate_config_tool, run_case_tool, optimize_pareto_tool]
