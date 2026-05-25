"""
file_io.py — YAML 配置加载与 OptimizeCaseConfig / RunCaseConfig 构建。

职责：
  1. 从 case_config.yaml 读取仿真、设计变量、目标函数、约束、提取配置
  2. 自动为每个 objectives 条目生成 ObjectiveFn（从 Aspen 树路径读值）
  3. 构建 RunCaseConfig 和 OptimizeCaseConfig，供 optimize_case() 直接使用

目标函数自动生成规则
---------------------
YAML 中每个 objective 条目包含：
  - name:       目标函数名称（ObjectiveValue.name）
  - aspen_path: Aspen 树路径，从 sim_result.outputs 字典中读取
  - minimize:   True/False
  - unit:       单位字符串（可选）

生成的 ObjectiveFn 从 ProcessCase.sim_result.outputs[aspen_path] 读取数值，
返回 ObjectiveValue。若路径不在 outputs 中，返回 error 字段非空的 ObjectiveValue。

典型用法
---------
    from src.utils.file_io import load_optimize_config

    opt_cfg, sim_filepath = load_optimize_config("cases/demo_case/case_config.yaml")
    with AspenDriver() as driver:
        driver.open(sim_filepath)
        result = optimize_case(driver, opt_cfg)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def load_optimize_config(
    yaml_path: str | Path,
) -> tuple[Any, Path, dict[str, Any]]:
    """
    从 case_config.yaml 加载并构建 OptimizeCaseConfig。

    Parameters
    ----------
    yaml_path:
        YAML 配置文件路径（绝对或相对于当前工作目录）。

    Returns
    -------
    (OptimizeCaseConfig, sim_filepath, driver_kwargs)
        OptimizeCaseConfig 可直接传给 optimize_case()。
        sim_filepath 是 Aspen 仿真文件的绝对路径，供调用方 driver.open() 使用。
        driver_kwargs 是 AspenDriver 的构造参数字典（visible、suppress_dialogs），
        供调用方 AspenDriver(**driver_kwargs) 使用。

    Raises
    ------
    FileNotFoundError
        YAML 文件不存在。
    KeyError / ValueError
        YAML 结构缺少必要字段或字段值非法。
    """
    import yaml  # PyYAML，运行时导入避免无 yaml 时模块级报错

    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{yaml_path}")

    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_filepath  = _parse_sim_filepath(cfg, yaml_path)
    driver_kwargs = _parse_driver_kwargs(cfg)
    run_cfg       = _build_run_config(cfg)
    opt_cfg       = _build_optimize_config(cfg, run_cfg)

    _log.info(
        "已加载配置：%s → %d 个设计变量，%d 个目标函数，n_initial=%d，n_iterations=%d。",
        yaml_path.name,
        len(opt_cfg.param_bounds),
        len(run_cfg.objective_fns),
        opt_cfg.n_initial,
        opt_cfg.n_iterations,
    )
    return opt_cfg, sim_filepath, driver_kwargs


# ---------------------------------------------------------------------------
# 仿真文件路径 / AspenDriver 参数解析
# ---------------------------------------------------------------------------

def _parse_driver_kwargs(cfg: dict) -> dict[str, Any]:
    """从 simulator 节提取 AspenDriver 构造参数。"""
    sim = cfg.get("simulator", {})
    return {
        "visible":          bool(sim.get("visible", False)),
        "suppress_dialogs": bool(sim.get("suppress_dialogs", True)),
    }

def _parse_sim_filepath(cfg: dict, yaml_path: Path) -> Path:
    """
    解析 simulator.filepath。

    解析优先级：
    1. 绝对路径 → 直接使用。
    2. 相对路径 → 先尝试相对于当前工作目录（用户从项目根运行时最常见），
       若文件不存在再尝试相对于 yaml 文件所在目录。
    """
    raw = cfg.get("simulator", {}).get("filepath")
    if not raw:
        raise KeyError("配置缺少 simulator.filepath 字段。")
    p = Path(raw)
    if p.is_absolute():
        return p
    # 相对路径：优先 cwd，其次 yaml 目录
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd.resolve()
    return (yaml_path.parent / p).resolve()


# ---------------------------------------------------------------------------
# RunCaseConfig 构建
# ---------------------------------------------------------------------------

def _build_run_config(cfg: dict) -> Any:
    """从 YAML 构建 RunCaseConfig。"""
    from ..workflows.run_case import RunCaseConfig

    sim   = cfg.get("simulator", {})
    ext   = cfg.get("extraction", {})
    objs  = cfg.get("objectives", []) or []
    cons  = cfg.get("constraints", []) or []

    objective_fns  = [_make_objective_fn(o) for o in objs]
    constraint_fns = [_make_constraint_fn(c) for c in cons]

    return RunCaseConfig(
        output_paths       = cfg.get("output_paths", []) or [],
        objective_fns      = objective_fns,
        constraint_fns     = constraint_fns,
        timeout            = float(sim.get("timeout", 300)),
        reinit             = bool(sim.get("reinit", True)),
        verify_inputs      = bool(sim.get("verify_inputs", True)),
        input_rtol         = float(sim.get("input_rtol", 1e-6)),
        check_status_paths = ext.get("check_status_paths") or None,
        extract_blocks     = ext.get("blocks") or None,
        extract_streams    = ext.get("streams") or None,
        block_max_depth    = int(ext.get("block_max_depth", 3)),
        stream_max_depth   = int(ext.get("stream_max_depth", 3)),
        stream_output_subtree = str(
            ext.get("stream_output_subtree", "Output\\STR_MAIN")
        ),
        strict_extraction  = bool(ext.get("strict_extraction", True)),
    )


# ---------------------------------------------------------------------------
# 设计变量解析（单目标和多目标共用）
# ---------------------------------------------------------------------------

def _parse_design_variables(cfg: dict) -> tuple[dict, dict]:
    """
    从 YAML 解析设计变量，返回 (param_bounds, fixed_vars)。

    type=continuous → param_bounds；type=integer/其他 → fixed_vars（固定为 initial_value）。
    """
    param_bounds: dict[str, tuple[float, float]] = {}
    fixed_vars:   dict[str, Any] = {}

    for dv in cfg.get("design_variables", []):
        path    = dv["aspen_path"]
        dv_type = dv.get("type", "continuous")
        if dv_type == "continuous":
            param_bounds[path] = (float(dv["lower_bound"]), float(dv["upper_bound"]))
        else:
            fixed_vars[path] = dv.get("initial_value", dv.get("lower_bound"))
            _log.debug(
                "设计变量 '%s'（type=%s）不支持贝叶斯优化，固定为初始值 %s。",
                dv.get("name", path), dv_type, fixed_vars[path],
            )

    return param_bounds, fixed_vars


# ---------------------------------------------------------------------------
# OptimizeCaseConfig 构建
# ---------------------------------------------------------------------------

def _build_optimize_config(cfg: dict, run_cfg: Any) -> Any:
    """
    从 YAML 构建优化配置。

    optimizer.type = "bayesian"（默认）→ OptimizeCaseConfig（单目标）
    optimizer.type = "pareto_bayesian"  → ParetoOptimizeCaseConfig（多目标）
    """
    opt_type = cfg.get("optimizer", {}).get("type", "bayesian")
    if opt_type == "pareto_bayesian":
        return _build_pareto_optimize_config(cfg, run_cfg)

    from ..workflows.optimize_case import OptimizeCaseConfig

    opt = cfg.get("optimizer", {})
    param_bounds, fixed_vars = _parse_design_variables(cfg)

    if not param_bounds:
        raise ValueError(
            "配置中没有 type=continuous 的设计变量，无法构建贝叶斯优化配置。"
        )

    objs = cfg.get("objectives", []) or []
    if not objs:
        raise ValueError("配置缺少 objectives 字段，至少需要一个目标函数。")
    primary = objs[0]
    if len(objs) > 1:
        _log.warning(
            "配置包含 %d 个目标函数，当前贝叶斯优化仅支持单目标，"
            "使用第一个目标 '%s'，其余忽略。如需多目标优化，请设置 optimizer.type: pareto_bayesian。",
            len(objs), primary["name"],
        )

    acq_raw = str(opt.get("acquisition_function", "EI")).upper()
    if acq_raw not in ("EI", "UCB", "PI"):
        _log.warning("acquisition_function '%s' 不合法，回退到 EI。", acq_raw)
        acq_raw = "EI"

    n_initial    = int(opt.get("n_initial_points", 10))
    n_bo         = int(opt.get("n_iterations", 30))
    n_iterations = n_initial + n_bo

    return OptimizeCaseConfig(
        param_bounds   = param_bounds,
        fixed_vars     = fixed_vars,
        run_config     = run_cfg,
        n_initial      = n_initial,
        n_iterations   = n_iterations,
        objective_name = primary["name"],
        minimize       = bool(primary.get("minimize", True)),
        acquisition    = acq_raw,  # type: ignore[arg-type]
        random_seed    = opt.get("random_seed"),
    )


def _build_pareto_optimize_config(cfg: dict, run_cfg: Any) -> Any:
    """从 YAML 构建 ParetoOptimizeCaseConfig（多目标贝叶斯优化）。"""
    from ..workflows.optimize_pareto_case import ParetoOptimizeCaseConfig

    opt = cfg.get("optimizer", {})
    param_bounds, fixed_vars = _parse_design_variables(cfg)

    if not param_bounds:
        raise ValueError(
            "配置中没有 type=continuous 的设计变量，无法构建多目标优化配置。"
        )

    objs = cfg.get("objectives", []) or []
    if len(objs) < 2:
        raise ValueError(
            f"pareto_bayesian 优化至少需要 2 个目标函数，当前只有 {len(objs)} 个。"
        )
    objective_names = [o["name"] for o in objs]

    scalarization = str(opt.get("scalarization", "weighted_sum"))
    if scalarization not in ("weighted_sum", "chebyshev"):
        _log.warning("scalarization '%s' 不合法，回退到 weighted_sum。", scalarization)
        scalarization = "weighted_sum"

    acq_raw = str(opt.get("acquisition_function", "EI")).upper()
    if acq_raw not in ("EI", "UCB", "PI"):
        _log.warning("acquisition_function '%s' 不合法，回退到 EI。", acq_raw)
        acq_raw = "EI"

    n_initial    = int(opt.get("n_initial_points", 10))
    n_bo         = int(opt.get("n_iterations", 30))
    n_iterations = n_initial + n_bo

    # 透传高级参数
    n_initial_min = int(opt.get("n_initial_min", 3))
    xi            = float(opt.get("xi", 0.01))
    kappa         = float(opt.get("kappa", 1.96))
    hv_margin     = float(opt.get("hv_margin", 0.1))
    tags          = list(opt.get("tags") or [])

    # reference_point：可选，需校验维度和有限性
    ref_raw = opt.get("reference_point")
    reference_point: list[float] | None = None
    if ref_raw is not None:
        import math
        try:
            reference_point = [float(v) for v in ref_raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"optimizer.reference_point 无法解析为浮点数列表：{exc}"
            ) from exc
        if len(reference_point) != len(objective_names):
            raise ValueError(
                f"optimizer.reference_point 维度 {len(reference_point)} 与目标数 "
                f"{len(objective_names)} 不一致。"
            )
        for i, v in enumerate(reference_point):
            if not math.isfinite(v):
                raise ValueError(
                    f"optimizer.reference_point[{i}]={v!r} 为非有限数（NaN/Inf）。"
                )

    _log.info(
        "已加载多目标配置：%d 个目标（%s），n_initial=%d，n_iterations=%d，scalarization=%s。",
        len(objective_names), objective_names, n_initial, n_iterations, scalarization,
    )

    return ParetoOptimizeCaseConfig(
        param_bounds    = param_bounds,
        fixed_vars      = fixed_vars,
        objective_names = objective_names,
        run_config      = run_cfg,
        n_initial       = n_initial,
        n_iterations    = n_iterations,
        n_initial_min   = n_initial_min,
        scalarization   = scalarization,  # type: ignore[arg-type]
        acquisition     = acq_raw,        # type: ignore[arg-type]
        xi              = xi,
        kappa           = kappa,
        reference_point = reference_point,
        hv_margin       = hv_margin,
        tags            = tags,
        random_seed     = opt.get("random_seed"),
    )


# ---------------------------------------------------------------------------
# 目标函数 / 约束函数自动生成
# ---------------------------------------------------------------------------

def _coerce_output_float(raw: Any, path: str) -> tuple[float | None, str | None]:
    """
    从 sim_result.outputs[path] 的值中提取 float。

    outputs 的值是 VariableResult，真实数值在 .value；
    也兼容直接存 float/int 的情况（测试 monkeypatch 常用）。

    Returns
    -------
    (value, error)
        成功时 error=None；失败时 value=None，error 为描述字符串。
    """
    # 解包 VariableResult
    if hasattr(raw, "value"):
        raw = raw.value
    if raw is None:
        return None, f"路径 '{path}' 的 VariableResult.value 为 None（Aspen 未输出该值）。"
    try:
        return float(raw), None
    except (TypeError, ValueError) as exc:
        return None, f"路径 '{path}' 的值 {raw!r} 无法转换为 float：{exc}"

def _make_objective_fn(obj_cfg: dict) -> Any:
    """
    从 YAML objective 条目生成 ObjectiveFn。

    type 字段决定目标函数类型：
      "aspen_path"（默认）：从 sim_result.outputs[aspen_path] 读取数值。
      "tac"：调用 make_tac_objective()，从 block 输出计算总年化成本。
      "emissions"：调用 make_emissions_objective()，从 block 输出计算 CO₂-eq 排放量。
    """
    obj_type = obj_cfg.get("type", "aspen_path")
    if obj_type == "tac":
        return _make_tac_fn(obj_cfg)
    if obj_type == "emissions":
        return _make_emissions_fn(obj_cfg)

    # 原有 aspen_path 逻辑
    from ..models.process_case import ObjectiveValue

    name      = obj_cfg["name"]
    path      = obj_cfg["aspen_path"]
    minimize  = bool(obj_cfg.get("minimize", True))
    unit      = str(obj_cfg.get("unit", ""))

    def objective_fn(case: Any) -> ObjectiveValue:
        outputs = {}
        if case.sim_result is not None:
            outputs = case.sim_result.outputs or {}

        raw = outputs.get(path)
        if raw is None:
            return ObjectiveValue(
                name=name, value=None, unit=unit, minimize=minimize,
                error=f"路径 '{path}' 不在 sim_result.outputs 中，"
                      "请确认 output_paths 已包含此路径。",
            )
        value, err = _coerce_output_float(raw, path)
        if err is not None:
            return ObjectiveValue(name=name, value=None, unit=unit, minimize=minimize, error=err)
        return ObjectiveValue(name=name, value=value, unit=unit, minimize=minimize)

    objective_fn.__name__ = name
    return objective_fn


def _make_tac_fn(obj_cfg: dict) -> Any:
    """
    从 YAML objective 条目（type: tac）构建 TAC 目标函数。

    YAML 参数（均可选，不写则使用 TACConfig 默认值）：
      annualization_factor, operating_hours, skip_missing, allow_partial_objective
      utility_cost.steam_price, utility_cost.cooling_water_price, utility_cost.electricity_price
      equipment_params.cepci_current
      output_key_map: {block_type: {semantic_key: output_node_name}}
        用于适配 Aspen block 输出键名与默认映射不一致的情况。
    """
    from ..economics.tac import (
        EquipmentCostParams, TACConfig, UtilityCost, make_tac_objective,
    )
    from ..models.process_case import ObjectiveValue

    _KNOWN_KEYS = {
        "name", "type", "minimize", "unit",
        "annualization_factor", "operating_hours", "skip_missing", "allow_partial_objective",
        "utility_cost", "equipment_params", "output_key_map", "block_design_params",
    }
    for key in obj_cfg:
        if key not in _KNOWN_KEYS:
            _log.warning(
                "tac 目标函数配置中存在未知字段 '%s'，已忽略。"
                "支持的字段：%s", key, sorted(_KNOWN_KEYS),
            )

    uc_raw     = obj_cfg.get("utility_cost") or {}
    ep_raw     = obj_cfg.get("equipment_params") or {}
    key_map_raw = obj_cfg.get("output_key_map")
    key_map    = dict(key_map_raw) if key_map_raw else {}

    # block_design_params: {block_name: {semantic_key: SI_value}}
    # 用于 Aspen Output 子树缺少节点或值无效时的 fallback（如 NSTAGE 在 Input 子树、DIAM=0）
    bdp_raw = obj_cfg.get("block_design_params")
    block_design_params: dict[str, dict[str, float]] = {}
    if bdp_raw:
        for blk_name, params in bdp_raw.items():
            if isinstance(params, dict):
                block_design_params[str(blk_name)] = {
                    str(k): float(v) for k, v in params.items()
                }

    tac_cfg = TACConfig(
        annualization_factor    = float(obj_cfg.get("annualization_factor", 0.1)),
        operating_hours         = float(obj_cfg.get("operating_hours", 8000.0)),
        skip_missing            = bool(obj_cfg.get("skip_missing", False)),
        allow_partial_objective = bool(obj_cfg.get("allow_partial_objective", False)),
        output_key_map          = key_map,
        block_design_params     = block_design_params,
        utility_cost = UtilityCost(
            steam_price         = float(uc_raw.get("steam_price", 14.19)),
            cooling_water_price = float(uc_raw.get("cooling_water_price", 0.354)),
            electricity_price   = float(uc_raw.get("electricity_price", 0.0775)),
        ),
        equipment_params = EquipmentCostParams(
            cepci_current = float(ep_raw.get("cepci_current", 800.0)),
        ),
    )
    _inner = make_tac_objective(tac_cfg)

    # 用 YAML 配置的 name/unit/minimize 覆盖内置函数返回的 ObjectiveValue 字段，
    # 保证 ProcessCase.objectives 里的名称与 ParetoOptimizeCaseConfig.objective_names 一致。
    yaml_name     = str(obj_cfg.get("name", "TAC"))
    yaml_unit     = str(obj_cfg.get("unit", "$/yr"))
    yaml_minimize = bool(obj_cfg.get("minimize", True))

    def tac_fn(case: Any) -> ObjectiveValue:
        result = _inner(case)
        return ObjectiveValue(
            name     = yaml_name,
            value    = result.value,
            unit     = yaml_unit,
            minimize = yaml_minimize,
            error    = result.error,
        )

    tac_fn.__name__ = yaml_name
    return tac_fn


def _make_emissions_fn(obj_cfg: dict) -> Any:
    """
    从 YAML objective 条目（type: emissions）构建排放量目标函数。

    YAML 参数（均可选，不写则使用 EmissionsConfig 默认值）：
      operating_hours, vent_streams, skip_missing, allow_partial_objective
      emission_factors.steam_factor, emission_factors.cooling_water_factor,
      emission_factors.electricity_factor
      ghg_components, missing_component_policy, zero_scope2_block_types,
      scope2_block_type_aliases, output_key_map
    """
    from ..economics.emissions import (
        EmissionFactors, EmissionsConfig, GWP100_DEFAULT, make_emissions_objective,
    )
    from ..models.process_case import ObjectiveValue

    ef_raw = obj_cfg.get("emission_factors") or {}

    # P2：透传扩展字段，并对用户写了但未支持的字段给 warning
    _KNOWN_KEYS = {
        "name", "type", "minimize", "unit",
        "operating_hours", "vent_streams", "skip_missing", "allow_partial_objective",
        "emission_factors", "ghg_components", "missing_component_policy",
        "zero_scope2_block_types", "scope2_block_type_aliases", "output_key_map",
    }
    for key in obj_cfg:
        if key not in _KNOWN_KEYS:
            _log.warning(
                "emissions 目标函数配置中存在未知字段 '%s'，已忽略。"
                "支持的字段：%s", key, sorted(_KNOWN_KEYS),
            )

    ghg_raw = obj_cfg.get("ghg_components")
    ghg_components = dict(ghg_raw) if ghg_raw else dict(GWP100_DEFAULT)

    zero_types_raw = obj_cfg.get("zero_scope2_block_types")
    zero_types = set(zero_types_raw) if zero_types_raw else set()

    aliases_raw = obj_cfg.get("scope2_block_type_aliases")
    aliases = dict(aliases_raw) if aliases_raw else {}

    key_map_raw = obj_cfg.get("output_key_map")
    key_map = dict(key_map_raw) if key_map_raw else {}

    missing_policy = str(obj_cfg.get("missing_component_policy", "zero"))
    if missing_policy not in ("zero", "error"):
        _log.warning(
            "missing_component_policy '%s' 不合法，回退到 'zero'。", missing_policy
        )
        missing_policy = "zero"

    em_cfg = EmissionsConfig(
        operating_hours             = float(obj_cfg.get("operating_hours", 8000.0)),
        vent_streams                = list(obj_cfg.get("vent_streams") or []),
        skip_missing                = bool(obj_cfg.get("skip_missing", False)),
        allow_partial_objective     = bool(obj_cfg.get("allow_partial_objective", False)),
        ghg_components              = ghg_components,
        missing_component_policy    = missing_policy,  # type: ignore[arg-type]
        zero_scope2_block_types     = zero_types,
        scope2_block_type_aliases   = aliases,
        output_key_map              = key_map,
        emission_factors = EmissionFactors(
            steam_factor         = float(ef_raw.get("steam_factor", 66.0)),
            cooling_water_factor = float(ef_raw.get("cooling_water_factor", 0.0)),
            electricity_factor   = float(ef_raw.get("electricity_factor", 0.581)),
        ),
    )
    _inner = make_emissions_objective(em_cfg)

    yaml_name     = str(obj_cfg.get("name", "EMISSIONS"))
    yaml_unit     = str(obj_cfg.get("unit", "tonne CO2-eq/yr"))
    yaml_minimize = bool(obj_cfg.get("minimize", True))

    def emissions_fn(case: Any) -> ObjectiveValue:
        result = _inner(case)
        return ObjectiveValue(
            name     = yaml_name,
            value    = result.value,
            unit     = yaml_unit,
            minimize = yaml_minimize,
            error    = result.error,
        )

    emissions_fn.__name__ = yaml_name
    return emissions_fn


def _make_constraint_fn(con_cfg: dict) -> Any:
    """
    从 YAML constraint 条目生成 ConstraintFn。

    约束形式：value <= 0 表示满足。
    YAML 中需提供 aspen_path、operator（"<="/"<"/">="/">"/"=="）和 threshold。
    生成的函数计算 (读取值 - threshold) 或 (threshold - 读取值)，标准化为 value <= 0。
    """
    from ..models.process_case import ConstraintValue

    name      = con_cfg["name"]
    path      = con_cfg["aspen_path"]
    operator  = str(con_cfg.get("operator", "<="))
    threshold = float(con_cfg.get("threshold", 0.0))

    def constraint_fn(case: Any) -> ConstraintValue:
        outputs = {}
        if case.sim_result is not None:
            outputs = case.sim_result.outputs or {}

        raw = outputs.get(path)
        if raw is None:
            return ConstraintValue(
                name=name, value=None,
                error=f"路径 '{path}' 不在 sim_result.outputs 中。",
            )
        v, err = _coerce_output_float(raw, path)
        if err is not None:
            return ConstraintValue(name=name, value=None, error=err)

        # 标准化为 value <= 0 形式
        if operator in ("<=", "<"):
            normalized = v - threshold
        elif operator in (">=", ">"):
            normalized = threshold - v
        elif operator == "==":
            normalized = abs(v - threshold)
        else:
            return ConstraintValue(
                name=name, value=None,
                error=f"不支持的约束运算符：{operator!r}，支持 <=/</>=/>/==。",
            )
        return ConstraintValue(name=name, value=normalized)

    constraint_fn.__name__ = name
    return constraint_fn
