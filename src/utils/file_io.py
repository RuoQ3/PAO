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
        "visible":              bool(sim.get("visible", False)),
        "suppress_dialogs":     bool(sim.get("suppress_dialogs", True)),
        "require_type_library": bool(sim.get("require_type_library", False)),
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
        extraction_mode    = str(ext.get("mode", "full")),
        catalog_db_path    = ext.get("catalog_db") or None,
        manifest_id        = str(ext.get("manifest_id", "auto")),
        semantic_rules_dir = str(ext.get("semantic_rules_dir", "configs/aspen_semantics")),
        build_manifest_if_missing = bool(ext.get("build_manifest_if_missing", True)),
        write_node_values  = bool(ext.get("write_node_values", True)),
        strict_manifest    = bool(ext.get("strict_manifest", True)),
        recycle_warmstart  = _parse_recycle_warmstart(sim),
    )


def _parse_recycle_warmstart(sim: dict) -> Any:
    """
    解析 simulator.recycle_warmstart 节，构建 RecycleWarmstartConfig。

    YAML 示例：
        simulator:
          recycle_warmstart:
            enabled: true
            mode: inherit          # fixed | inherit
            init_values:           # 固定初值 / inherit 模式的首次回退值
              \\Data\\Streams\\RECYCLE1\\Input\\TOTFLOW\\MIXED: 200.0
              \\Data\\Streams\\RECYCLE1\\Input\\TEMP\\MIXED: 350.0
            read_paths:            # inherit 模式：成功后从这些路径读取收敛值
              - \\Data\\Streams\\RECYCLE1\\Output\\TOTFLOW\\MIXED
              - \\Data\\Streams\\RECYCLE1\\Output\\TEMP\\MIXED

    未配置或 enabled=false 时返回 None，不影响现有工况行为。
    """
    rw_raw = sim.get("recycle_warmstart") or {}
    if not rw_raw.get("enabled", False):
        return None

    from ..workflows.run_case import RecycleWarmstartConfig

    mode = str(rw_raw.get("mode", "fixed"))
    if mode not in ("fixed", "inherit"):
        _log.warning(
            "recycle_warmstart.mode='%s' 不合法（仅支持 fixed/inherit），已回退为 fixed。",
            mode,
        )
        mode = "fixed"

    raw_init = rw_raw.get("init_values") or {}
    init_values = {str(k): float(v) for k, v in raw_init.items()}

    read_paths = [str(p) for p in (rw_raw.get("read_paths") or [])]

    _log.info(
        "recycle_warmstart 已启用：mode=%s，init_values=%d 个，read_paths=%d 个。",
        mode, len(init_values), len(read_paths),
    )
    return RecycleWarmstartConfig(
        mode=mode,
        init_values=init_values,
        read_paths=read_paths,
    )


# ---------------------------------------------------------------------------
# 设计变量解析（单目标和多目标共用）
# ---------------------------------------------------------------------------

def _parse_design_variables(cfg: dict) -> tuple[dict, dict, set, list[dict[str, Any]]]:
    """
    从 YAML 解析设计变量，返回 (param_bounds, fixed_vars, integer_var_paths)。

    type=continuous → param_bounds（连续搜索）
    type=integer    → param_bounds（整数搜索，round/clamp 在 repair 阶段处理）
    其他 type       → fixed_vars（固定为 initial_value）
    """
    param_bounds: dict[str, tuple[float, float]] = {}
    fixed_vars:   dict[str, Any] = {}
    integer_var_paths: set[str] = set()
    derived_var_specs: list[dict[str, Any]] = []

    for dv in cfg.get("design_variables", []):
        dv_type = dv.get("type", "continuous")
        if dv_type == "derived":
            frac_path = str(dv["name"])
            target_path = str(dv["target_path"])
            depends_on = str(dv["depends_on"])
            lo_frac = float(dv["lo_frac"])
            hi_frac = float(dv["hi_frac"])
            param_bounds[frac_path] = (lo_frac, hi_frac)
            derived_var_specs.append({
                "frac_path": frac_path,
                "target_path": target_path,
                "depends_on": depends_on,
                "frac_lo": int(dv.get("frac_lo", dv.get("lower_bound", 1))),
            })
            _log.debug(
                "derived design variable '%s': frac [%s, %s] -> %s depends_on=%s",
                frac_path, lo_frac, hi_frac, target_path, depends_on,
            )
            continue

        path    = dv["aspen_path"]
        dv_type = dv.get("type", "continuous")
        if dv_type == "continuous":
            param_bounds[path] = (float(dv["lower_bound"]), float(dv["upper_bound"]))
        elif dv_type == "integer":
            param_bounds[path] = (float(dv["lower_bound"]), float(dv["upper_bound"]))
            integer_var_paths.add(path)
            _log.debug(
                "设计变量 '%s'（type=integer）纳入 BO 搜索空间 [%s, %s]，将在 repair 阶段 round/clamp。",
                dv.get("name", path), dv["lower_bound"], dv["upper_bound"],
            )
        else:
            fixed_vars[path] = dv.get("initial_value", dv.get("lower_bound"))
            _log.debug(
                "设计变量 '%s'（type=%s）不支持贝叶斯优化，固定为初始值 %s。",
                dv.get("name", path), dv_type, fixed_vars[path],
            )

    return param_bounds, fixed_vars, integer_var_paths, derived_var_specs


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
    param_bounds, fixed_vars, integer_paths, derived_var_specs = _parse_design_variables(cfg)

    if not param_bounds:
        raise ValueError(
            "配置中没有 type=continuous 或 type=integer 的设计变量，无法构建贝叶斯优化配置。"
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

    surrogate_model = _parse_surrogate_model(opt)

    n_initial    = int(opt.get("n_initial_points", 10))
    n_bo         = int(opt.get("n_iterations", 30))
    n_iterations = n_initial + n_bo

    # 单目标也支持 feasibility_search（复用多目标解析逻辑）
    fs_raw = cfg.get("feasibility_search") or {}
    feasibility_search_single = None
    if fs_raw.get("enabled", False):
        from ..workflows.optimize_pareto_case import FeasibilitySearchConfig
        initial_point_raw: dict = {}
        for dv in cfg.get("design_variables", []):
            dv_type = dv.get("type", "continuous")
            if dv_type in ("continuous", "integer"):
                path = dv.get("aspen_path")
                iv   = dv.get("initial_value")
                if path and iv is not None:
                    try:
                        initial_point_raw[path] = float(iv)
                    except (TypeError, ValueError):
                        pass
            elif dv_type == "derived":
                frac_path = dv.get("name")
                lo_frac   = dv.get("lo_frac")
                hi_frac   = dv.get("hi_frac")
                iv        = dv.get("initial_value")
                if frac_path:
                    try:
                        if iv is not None:
                            initial_point_raw[str(frac_path)] = float(iv)
                        elif lo_frac is not None and hi_frac is not None:
                            mid = (float(lo_frac) + float(hi_frac)) / 2.0
                            initial_point_raw[str(frac_path)] = mid
                            _log.debug(
                                "Phase 0：derived 变量 '%s' 未设置 initial_value，"
                                "使用中点 %.3f（建议在 YAML 中填写反算值）。",
                                frac_path, mid,
                            )
                    except (TypeError, ValueError):
                        pass
        radii_raw = fs_raw.get("local_search_radii")
        local_radii = (
            [float(r) for r in radii_raw] if isinstance(radii_raw, list)
            else [0.2, 0.5]
        )
        feasibility_search_single = FeasibilitySearchConfig(
            enabled=True,
            n_trials=int(fs_raw.get("n_trials", 20)),
            stop_after_feasible=int(fs_raw.get("stop_after_feasible", 3)),
            abort_if_none_found=bool(fs_raw.get("abort_if_none_found", False)),  # 单目标默认不终止
            initial_point=initial_point_raw or None,
            local_search_radii=local_radii,
        )
        _log.info(
            "单目标 Phase 0 可行性搜索已启用：n_trials=%d，local_search_radii=%s。",
            feasibility_search_single.n_trials,
            feasibility_search_single.local_search_radii,
        )

    return OptimizeCaseConfig(
        param_bounds    = param_bounds,
        fixed_vars      = fixed_vars,
        run_config      = run_cfg,
        n_initial       = n_initial,
        n_iterations    = n_iterations,
        objective_name  = primary["name"],
        minimize        = bool(primary.get("minimize", True)),
        acquisition     = acq_raw,  # type: ignore[arg-type]
        surrogate_model = surrogate_model,  # type: ignore[arg-type]
        random_seed     = opt.get("random_seed"),
        integer_var_paths  = integer_paths,
        derived_var_specs  = derived_var_specs,
        feasibility_filter = _parse_feasibility_filter(opt.get("feasibility_filter")),
        early_stopping     = _parse_early_stopping(opt.get("early_stopping")),
        feasibility_search = feasibility_search_single,
    )


def _build_pareto_optimize_config(cfg: dict, run_cfg: Any) -> Any:
    """从 YAML 构建 ParetoOptimizeCaseConfig（多目标贝叶斯优化）。"""
    from ..workflows.optimize_pareto_case import ParetoOptimizeCaseConfig

    opt = cfg.get("optimizer", {})
    param_bounds, fixed_vars, integer_var_paths, derived_var_specs = _parse_design_variables(cfg)

    if not param_bounds:
        raise ValueError(
            "配置中没有 type=continuous 或 type=integer 的设计变量，无法构建多目标优化配置。"
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

    surrogate_model = _parse_surrogate_model(opt)

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
        "已加载多目标配置：%d 个目标（%s），%d 个连续变量，%d 个 integer 变量，"
        "n_initial=%d，n_iterations=%d，scalarization=%s。",
        len(objective_names), objective_names,
        len(param_bounds) - len(integer_var_paths) - len(derived_var_specs), len(integer_var_paths),
        n_initial, n_iterations, scalarization,
    )

    # 解析 var_dependencies（变量依赖约束，如 FEED_STAGE < NSTAGE）
    for spec in derived_var_specs:
        _log.info(
            "derived var: %s -> %s (depends_on=%s, frac_lo=%s)",
            spec["frac_path"], spec["target_path"], spec["depends_on"], spec["frac_lo"],
        )

    var_deps_raw = cfg.get("var_dependencies") or {}
    var_dependencies: dict[str, dict[str, str]] = {}
    for var_path, dep_rules in var_deps_raw.items():
        if isinstance(dep_rules, dict):
            var_dependencies[str(var_path)] = {str(k): str(v) for k, v in dep_rules.items()}

    # 解析 feasibility_search（Phase 0 可行性搜索）
    fs_raw = cfg.get("feasibility_search") or {}
    feasibility_search = None
    if fs_raw.get("enabled", False):
        from ..workflows.optimize_pareto_case import FeasibilitySearchConfig

        # 从设计变量的 initial_value 提取初始点（来自已收敛的 .bkp 文件）
        # continuous / integer 变量使用 aspen_path 为键，取 initial_value
        # derived 变量使用 name 为键（frac 路径），取 (lo_frac + hi_frac) / 2 作为初始 frac
        initial_point: dict | None = None
        initial_point_raw: dict = {}
        for dv in cfg.get("design_variables", []):
            dv_type = dv.get("type", "continuous")
            if dv_type in ("continuous", "integer"):
                path = dv.get("aspen_path")
                iv   = dv.get("initial_value")
                if path and iv is not None:
                    try:
                        initial_point_raw[path] = float(iv)
                    except (TypeError, ValueError):
                        pass
            elif dv_type == "derived":
                # derived 变量搜索的是 frac 值
                # 优先使用 initial_value（需在 YAML 中显式填写，如 FEED_STAGE/NSTAGE 的反算值）
                # 找不到 initial_value 时才退化为 (lo_frac + hi_frac) / 2 中点（不推荐，进料板可能偏差很大）
                frac_path = dv.get("name")
                lo_frac   = dv.get("lo_frac")
                hi_frac   = dv.get("hi_frac")
                iv        = dv.get("initial_value")
                if frac_path:
                    try:
                        if iv is not None:
                            initial_point_raw[str(frac_path)] = float(iv)
                        elif lo_frac is not None and hi_frac is not None:
                            mid = (float(lo_frac) + float(hi_frac)) / 2.0
                            initial_point_raw[str(frac_path)] = mid
                            _log.debug(
                                "Phase 0：derived 变量 '%s' 未设置 initial_value，"
                                "使用中点 %.3f 作为初始 frac（建议在 YAML 中反算填写）。",
                                frac_path, mid,
                            )
                    except (TypeError, ValueError):
                        pass
        if initial_point_raw:
            initial_point = initial_point_raw
            _log.info(
                "Phase 0：已从 initial_value 提取 %d 个变量的初始收敛解，"
                "将作为第一个候选点注入。",
                len(initial_point_raw),
            )

        # 解析自适应局部扩张半径
        radii_raw = fs_raw.get("local_search_radii")
        if radii_raw is None:
            local_search_radii = [0.2, 0.5]   # 默认：先 ±20%，再 ±50%
        elif isinstance(radii_raw, list):
            local_search_radii = [float(r) for r in radii_raw]
        else:
            local_search_radii = [0.2, 0.5]

        feasibility_search = FeasibilitySearchConfig(
            enabled=True,
            n_trials=int(fs_raw.get("n_trials", 20)),
            stop_after_feasible=int(fs_raw.get("stop_after_feasible", 3)),
            abort_if_none_found=bool(fs_raw.get("abort_if_none_found", True)),
            initial_point=initial_point,
            local_search_radii=local_search_radii,
        )
        _log.info(
            "Phase 0 可行性搜索已启用：n_trials=%d，stop_after_feasible=%d，"
            "abort_if_none_found=%s，local_search_radii=%s。",
            feasibility_search.n_trials,
            feasibility_search.stop_after_feasible,
            feasibility_search.abort_if_none_found,
            feasibility_search.local_search_radii,
        )

    # 解析 trust_region 配置
    tr_raw = cfg.get("trust_region") or {}
    trust_region_cfg = None
    if tr_raw.get("enabled", False):
        from ..optimization.trust_region import TrustRegionConfig
        trust_region_cfg = TrustRegionConfig(
            initial_radius=float(tr_raw.get("initial_radius", 0.5)),
            min_radius=float(tr_raw.get("min_radius", 0.05)),
            max_radius=float(tr_raw.get("max_radius", 1.0)),
            gamma_expand=float(tr_raw.get("gamma_expand", 1.5)),
            gamma_shrink=float(tr_raw.get("gamma_shrink", 0.7)),
            eta_success=float(tr_raw.get("eta_success", 0.05)),
            failure_tolerance=int(tr_raw.get("failure_tolerance", 5)),
        )
        _log.info(
            "Trust Region 已启用：initial_radius=%.2f, failure_tolerance=%d",
            trust_region_cfg.initial_radius, trust_region_cfg.failure_tolerance,
        )

    # 解析 reference_values（飞行前检查的参考工况，来自设计变量 initial_value）
    # continuous/integer 用 aspen_path 为键；derived 用 name(frac) 为键。
    # 这是任何能跑的 .bkp 都自带的"已知可行解"锚点。
    reference_values: dict[str, float] = {}
    for dv in cfg.get("design_variables", []):
        dv_type = dv.get("type", "continuous")
        iv = dv.get("initial_value")
        if iv is None:
            continue
        if dv_type in ("continuous", "integer"):
            path = dv.get("aspen_path")
            if path:
                try:
                    reference_values[str(path)] = float(iv)
                except (TypeError, ValueError):
                    pass
        elif dv_type == "derived":
            frac_path = dv.get("name")
            if frac_path:
                try:
                    reference_values[str(frac_path)] = float(iv)
                except (TypeError, ValueError):
                    pass

    preflight_cfg = _parse_preflight(cfg.get("preflight"))
    boundary_refine_cfg = _parse_boundary_refine(cfg.get("boundary_refine"))
    _br_interval = 20
    _br_raw = cfg.get("boundary_refine") or {}
    if isinstance(_br_raw, dict) and _br_raw.get("interval") is not None:
        try:
            _br_interval = max(1, int(_br_raw["interval"]))
        except (TypeError, ValueError):
            _br_interval = 20

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
        surrogate_model = surrogate_model,  # type: ignore[arg-type]
        xi              = xi,
        kappa           = kappa,
        reference_point = reference_point,
        hv_margin       = hv_margin,
        tags            = tags,
        random_seed     = opt.get("random_seed"),
        integer_var_paths  = integer_var_paths,
        derived_var_specs  = derived_var_specs,
        var_dependencies   = var_dependencies,
        feasibility_search = feasibility_search,
        feasibility_filter = _parse_feasibility_filter(opt.get("feasibility_filter")),
        early_stopping     = _parse_early_stopping(opt.get("early_stopping")),
        trust_region       = trust_region_cfg,
        sensitivity_probe  = _parse_sensitivity_probe(cfg.get("sensitivity_probe")),
        preflight          = preflight_cfg,
        reference_values   = reference_values,
        boundary_refine    = boundary_refine_cfg,
        boundary_refine_interval = _br_interval,
        # 从 YAML optimizer.session_id 恢复 API 会话 ID
        **( {"session_id": str(opt["session_id"])} if opt.get("session_id") else {} ),
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
    if obj_type == "custom_module":
        return _make_custom_module_fn(obj_cfg)

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
        "fallback_design",
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

    # fallback_design：语义更清晰的替代写法，支持 diameter/nstage 键名，
    # 并提供 source 和 conservative_factor 标注。
    # 与 block_design_params 合并，fallback_design 优先。
    fallback_source = "block_design_params"
    fallback_conservative_factor = 1.0
    fd_raw = obj_cfg.get("fallback_design")
    if fd_raw and isinstance(fd_raw, dict):
        fallback_source = str(fd_raw.get("source", "yaml_fallback"))
        fallback_conservative_factor = float(fd_raw.get("conservative_factor", 1.0))
        # column_diameter: {block_name: value_m}
        col_diam = fd_raw.get("column_diameter") or {}
        for blk_name, diam_val in col_diam.items():
            blk_key = str(blk_name)
            if blk_key not in block_design_params:
                block_design_params[blk_key] = {}
            block_design_params[blk_key]["diam"] = float(diam_val)
        # nstage: {block_name: value}（可选）
        nstage_map = fd_raw.get("nstage") or {}
        for blk_name, n_val in nstage_map.items():
            blk_key = str(blk_name)
            if blk_key not in block_design_params:
                block_design_params[blk_key] = {}
            block_design_params[blk_key]["nstage"] = float(n_val)

    tac_cfg = TACConfig(
        annualization_factor         = float(obj_cfg.get("annualization_factor", 0.1)),
        operating_hours              = float(obj_cfg.get("operating_hours", 8000.0)),
        skip_missing                 = bool(obj_cfg.get("skip_missing", False)),
        allow_partial_objective      = bool(obj_cfg.get("allow_partial_objective", False)),
        output_key_map               = key_map,
        block_design_params          = block_design_params,
        fallback_source              = fallback_source,
        fallback_conservative_factor = fallback_conservative_factor,
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


def _make_custom_module_fn(obj_cfg: dict) -> Any:
    """
    从 YAML objective 条目（type: custom_module）构建自定义目标函数。

    YAML 参数：
      module:   Python 文件路径（相对项目根目录或绝对路径），如
                "cases/demo_case_2/epsd_objectives.py"
      function: 工厂函数名，无参调用后返回 ObjectiveFn，如 "make_epsd_opex_objective"

    用法示例：
      - name: OPEX
        type: custom_module
        module: cases/demo_case_2/epsd_objectives.py
        function: make_epsd_opex_objective
        minimize: true
        unit: "RMB/yr"
    """
    import importlib.util

    module_path = obj_cfg["module"]
    func_name   = obj_cfg["function"]

    spec = importlib.util.spec_from_file_location("_custom_objective_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载自定义目标函数模块：{module_path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    factory = getattr(mod, func_name, None)
    if factory is None:
        raise AttributeError(
            f"模块 {module_path!r} 中未找到函数 {func_name!r}"
        )

    objective_fn = factory()

    # 支持在 YAML 中覆盖 name/minimize/unit（工厂函数内部的设置为默认值）
    yaml_name     = obj_cfg.get("name")
    yaml_minimize = obj_cfg.get("minimize")
    yaml_unit     = obj_cfg.get("unit")

    if yaml_name is not None or yaml_minimize is not None or yaml_unit is not None:
        # 包装一层，将 YAML 元数据覆盖到返回的 ObjectiveValue 上
        from ..models.process_case import ObjectiveValue

        inner_fn  = objective_fn
        ov_name   = yaml_name     if yaml_name     is not None else inner_fn.__name__
        ov_min    = bool(yaml_minimize) if yaml_minimize is not None else True
        ov_unit   = str(yaml_unit)      if yaml_unit     is not None else ""

        def wrapper_fn(case: Any) -> ObjectiveValue:
            result = inner_fn(case)
            return ObjectiveValue(
                name=ov_name,
                value=result.value,
                unit=ov_unit if ov_unit else result.unit,
                minimize=ov_min,
                error=result.error,
            )

        wrapper_fn.__name__ = ov_name
        return wrapper_fn

    return objective_fn


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


# ---------------------------------------------------------------------------
# 代理模型解析
# ---------------------------------------------------------------------------

_VALID_SURROGATE_UPPER = {"GP", "RF", "ET", "GBRT", "RANDOM", "QEHVI", "NEHVI"}


def _parse_surrogate_model(opt: dict) -> str:
    """
    从 optimizer 节解析 surrogate_model 字段。

    大小写不敏感；RANDOM → "random"；其余保持大写。
    非法值直接抛 ValueError，不静默回退。
    未配置时默认 "GP"。
    """
    raw = opt.get("surrogate_model", "GP")
    upper = str(raw).upper()
    if upper not in _VALID_SURROGATE_UPPER:
        raise ValueError(
            f"surrogate_model {raw!r} 不合法，"
            f"支持值：GP / RF / ET / GBRT / RANDOM / qEHVI / NEHVI（大小写不敏感）。"
        )
    if upper == "RANDOM":
        return "random"
    if upper == "QEHVI":
        return "qEHVI"
    return upper


# ---------------------------------------------------------------------------
# feasibility_filter 解析
# ---------------------------------------------------------------------------

_VALID_FF_MODELS: frozenset[str] = frozenset({"extra_trees", "random_forest", "random"})


def _parse_feasibility_filter(raw: dict[str, Any] | None) -> Any:
    """
    从 optimizer.feasibility_filter 节解析 FeasibilityConfig。

    如果节不存在或为空 dict，返回 FeasibilityConfig()（enabled=False）。
    任何字段非法时严格抛 ValueError，不静默回退默认值。

    支持字段
    --------
    enabled           : bool，严格接受 True/False/0/1/"true"/"false"（大小写不敏感），默认 False
    model             : str，允许 extra_trees / random_forest / random，默认 extra_trees
    min_samples       : int，>= 1，默认 10
    threshold         : float，[0.0, 1.0]，默认 0.5
    candidate_pool_size : int，>= 1，默认 200
    random_seed       : int | None，默认 None
    """
    from ..optimization.feasibility import FeasibilityConfig

    if raw is None or raw == {}:
        return FeasibilityConfig()

    # --- 类型检查：feasibility_filter 必须是 dict ---
    if not isinstance(raw, dict):
        raise ValueError(
            f"optimizer.feasibility_filter 必须是 dict（映射），"
            f"收到 {type(raw).__name__!r}：{raw!r}。"
        )

    # --- enabled（严格布尔解析）---
    enabled_raw = raw.get("enabled", False)
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    elif isinstance(enabled_raw, int) and enabled_raw in (0, 1):
        enabled = bool(enabled_raw)
    elif isinstance(enabled_raw, str):
        if enabled_raw.strip().lower() == "true":
            enabled = True
        elif enabled_raw.strip().lower() == "false":
            enabled = False
        else:
            raise ValueError(
                f"feasibility_filter.enabled={enabled_raw!r} 无法识别，"
                "请使用 true / false（YAML 布尔值）。"
            )
    else:
        raise ValueError(
            f"feasibility_filter.enabled={enabled_raw!r} 类型不合法"
            f"（{type(enabled_raw).__name__}），请使用 true 或 false。"
        )

    # --- model ---
    model_raw = str(raw.get("model", "extra_trees"))
    if model_raw not in _VALID_FF_MODELS:
        raise ValueError(
            f"feasibility_filter.model={model_raw!r} 不合法，"
            f"支持值：{sorted(_VALID_FF_MODELS)}。"
        )

    # --- min_samples ---
    try:
        min_samples = int(raw.get("min_samples", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"feasibility_filter.min_samples 无法转为整数：{exc}"
        ) from exc
    if min_samples < 1:
        raise ValueError(
            f"feasibility_filter.min_samples={min_samples} 必须 >= 1。"
        )

    # --- threshold ---
    try:
        threshold = float(raw.get("threshold", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"feasibility_filter.threshold 无法转为浮点数：{exc}"
        ) from exc
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"feasibility_filter.threshold={threshold} 必须在 [0.0, 1.0] 之间。"
        )

    # --- candidate_pool_size ---
    try:
        candidate_pool_size = int(raw.get("candidate_pool_size", 200))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"feasibility_filter.candidate_pool_size 无法转为整数：{exc}"
        ) from exc
    if candidate_pool_size < 1:
        raise ValueError(
            f"feasibility_filter.candidate_pool_size={candidate_pool_size} 必须 >= 1。"
        )

    # --- random_seed ---
    seed_raw = raw.get("random_seed")
    if seed_raw is None:
        random_seed = None
    else:
        try:
            random_seed = int(seed_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feasibility_filter.random_seed={seed_raw!r} 无法转为整数：{exc}"
            ) from exc

    return FeasibilityConfig(
        enabled=enabled,
        model=model_raw,
        min_samples=min_samples,
        threshold=threshold,
        candidate_pool_size=candidate_pool_size,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# early_stopping 解析
# ---------------------------------------------------------------------------

def _parse_early_stopping(raw: dict[str, Any] | None) -> Any:
    """
    从 optimizer.early_stopping 节解析 EarlyStoppingConfig。

    如果节不存在或为空 dict，返回 EarlyStoppingConfig()（enabled=False）。
    任何字段非法时严格抛 ValueError，不静默回退。
    """
    from ..workflows.common import EarlyStoppingConfig

    if raw is None or raw == {}:
        return EarlyStoppingConfig()

    if not isinstance(raw, dict):
        raise ValueError(
            f"optimizer.early_stopping 必须是 dict，收到 {type(raw).__name__!r}：{raw!r}。"
        )

    # 未知字段严格检查
    _ALLOWED_ES_KEYS = {
        "enabled", "min_iterations", "patience", "min_delta",
        "relative_delta", "max_duplicate_suggestions",
        "check_hypervolume", "check_first_front",
    }
    unknown = set(raw) - _ALLOWED_ES_KEYS
    if unknown:
        raise ValueError(
            f"optimizer.early_stopping 包含未知字段：{sorted(unknown)}。"
            f"允许的字段：{sorted(_ALLOWED_ES_KEYS)}。"
        )

    def _parse_bool(name: str, default: bool) -> bool:
        v = raw.get(name, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, int) and v in (0, 1):
            return bool(v)
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return v.strip().lower() == "true"
        raise ValueError(
            f"early_stopping.{name}={v!r} 无法识别，请使用 true 或 false。"
        )

    enabled = _parse_bool("enabled", False)

    try:
        min_iterations = int(raw.get("min_iterations", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"early_stopping.min_iterations 无法转为整数：{exc}") from exc

    try:
        patience = int(raw.get("patience", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"early_stopping.patience 无法转为整数：{exc}") from exc

    try:
        min_delta = float(raw.get("min_delta", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"early_stopping.min_delta 无法转为浮点数：{exc}") from exc

    rd_raw = raw.get("relative_delta")
    if rd_raw is None:
        relative_delta = None
    else:
        try:
            relative_delta = float(rd_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"early_stopping.relative_delta 无法转为浮点数：{exc}"
            ) from exc

    try:
        max_dup = int(raw.get("max_duplicate_suggestions", 3))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"early_stopping.max_duplicate_suggestions 无法转为整数：{exc}"
        ) from exc

    check_hv = _parse_bool("check_hypervolume", True)
    check_ff = _parse_bool("check_first_front", True)

    # EarlyStoppingConfig.__post_init__ 会做严格校验
    return EarlyStoppingConfig(
        enabled=enabled,
        min_iterations=min_iterations,
        patience=patience,
        min_delta=min_delta,
        relative_delta=relative_delta,
        max_duplicate_suggestions=max_dup,
        check_hypervolume=check_hv,
        check_first_front=check_ff,
    )


# ---------------------------------------------------------------------------
# sensitivity_probe 解析
# ---------------------------------------------------------------------------

def _parse_sensitivity_probe(raw: dict[str, Any] | None) -> Any:
    """
    从顶层 sensitivity_probe 节解析 SensitivityProbeConfig。

    如果节不存在或 enabled=false，返回 None（不启用探针）。

    支持字段
    --------
    enabled                : bool，默认 false
    n_perturbations        : int，每个变量扰动次数，默认 3
    perturbation_radius    : float (0,1]，扰动幅度占全局范围比例，默认 0.20
    min_doe_radius         : float (0,1]，高敏感变量 DOE 半径下限，默认 0.10
    correlation_threshold  : float [0,1]，强相关判断阈值，默认 0.70
    margin_weight          : float [0,1]，约束 margin 敏感度在综合敏感度中的权重，默认 0.5
                             0.0 = 纯收敛敏感度（旧行为）；1.0 = 纯约束 margin 敏感度
    thaw_hv_stall_patience : int，HV 停滞多少轮触发解冻，默认 10
    thaw_step_radius       : float，每步解冻半径扩大量，默认 0.10
    refreeze_fail_window   : int，失败率检测窗口，默认 5
    refreeze_fail_threshold: float，失败率超过此值则重冻，默认 0.60
    tags                   : list[str]，探针工况标签，默认 ["sensitivity_probe"]
    """
    if raw is None or raw == {}:
        return None

    if not isinstance(raw, dict):
        raise ValueError(
            f"sensitivity_probe 必须是 dict，收到 {type(raw).__name__!r}：{raw!r}。"
        )

    enabled_raw = raw.get("enabled", False)
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    elif isinstance(enabled_raw, int) and enabled_raw in (0, 1):
        enabled = bool(enabled_raw)
    elif isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() == "true"
    else:
        enabled = False

    if not enabled:
        return None

    from ..optimization.sensitivity_probe import SensitivityProbeConfig

    try:
        n_perturbations = int(raw.get("n_perturbations", 3))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.n_perturbations 无法转为整数：{exc}") from exc

    try:
        perturbation_radius = float(raw.get("perturbation_radius", 0.20))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.perturbation_radius 无法转为浮点数：{exc}") from exc

    try:
        min_doe_radius = float(raw.get("min_doe_radius", 0.10))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.min_doe_radius 无法转为浮点数：{exc}") from exc

    try:
        correlation_threshold = float(raw.get("correlation_threshold", 0.70))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.correlation_threshold 无法转为浮点数：{exc}") from exc

    try:
        margin_weight = float(raw.get("margin_weight", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.margin_weight 无法转为浮点数：{exc}") from exc

    try:
        thaw_hv_stall_patience = int(raw.get("thaw_hv_stall_patience", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.thaw_hv_stall_patience 无法转为整数：{exc}") from exc

    try:
        thaw_step_radius = float(raw.get("thaw_step_radius", 0.10))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.thaw_step_radius 无法转为浮点数：{exc}") from exc

    try:
        refreeze_fail_window = int(raw.get("refreeze_fail_window", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.refreeze_fail_window 无法转为整数：{exc}") from exc

    try:
        refreeze_fail_threshold = float(raw.get("refreeze_fail_threshold", 0.60))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensitivity_probe.refreeze_fail_threshold 无法转为浮点数：{exc}") from exc

    tags_raw = raw.get("tags")
    tags = list(tags_raw) if isinstance(tags_raw, list) else ["sensitivity_probe"]

    cfg = SensitivityProbeConfig(
        enabled=True,
        n_perturbations=n_perturbations,
        perturbation_radius=perturbation_radius,
        min_doe_radius=min_doe_radius,
        correlation_threshold=correlation_threshold,
        margin_weight=margin_weight,
        thaw_hv_stall_patience=thaw_hv_stall_patience,
        thaw_step_radius=thaw_step_radius,
        refreeze_fail_window=refreeze_fail_window,
        refreeze_fail_threshold=refreeze_fail_threshold,
        tags=tags,
    )
    _log.info(
        "敏感度探针已启用：n_perturbations=%d，perturbation_radius=%.2f，"
        "min_doe_radius=%.2f，margin_weight=%.2f，thaw_patience=%d。",
        cfg.n_perturbations, cfg.perturbation_radius,
        cfg.min_doe_radius, cfg.margin_weight, cfg.thaw_hv_stall_patience,
    )
    return cfg


# ---------------------------------------------------------------------------
# preflight 解析（飞行前检查）
# ---------------------------------------------------------------------------

def _parse_preflight(raw: dict[str, Any] | None) -> Any:
    """
    从顶层 preflight 节解析 PreflightConfig。

    节不存在或 enabled=false 时返回 None（不启用飞行前检查）。

    支持字段
    --------
    enabled                : bool，默认 false
    max_deviation_factor   : float>0 或省略，候选值相对初始收敛解的最大归一化偏离倍数。
                             建议 3~5。省略时不做偏离检查（仅做依赖/计算量检查）。
    check_var_dependencies : bool，默认 true，是否检查 var_dependencies。
    cost_proxy_groups      : list，每项 {name, var_paths:[...], max_ratio:float}，
                             用于拦截"计算量代理乘积放大过多"的地狱工况，默认空。
    """
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"preflight 必须是 dict，收到 {type(raw).__name__!r}：{raw!r}。")

    enabled_raw = raw.get("enabled", False)
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    elif isinstance(enabled_raw, int) and enabled_raw in (0, 1):
        enabled = bool(enabled_raw)
    elif isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() == "true"
    else:
        enabled = False
    if not enabled:
        return None

    from ..optimization.preflight import PreflightConfig, CostProxyGroup

    max_dev: float | None = None
    if raw.get("max_deviation_factor") is not None:
        try:
            max_dev = float(raw["max_deviation_factor"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"preflight.max_deviation_factor 无法转为浮点数：{exc}") from exc

    check_deps = bool(raw.get("check_var_dependencies", True))

    groups: list = []
    for g in (raw.get("cost_proxy_groups") or []):
        if not isinstance(g, dict):
            continue
        var_paths = [str(p) for p in (g.get("var_paths") or [])]
        if not var_paths:
            continue
        try:
            max_ratio = float(g.get("max_ratio", 3.0))
        except (TypeError, ValueError):
            max_ratio = 3.0
        groups.append(CostProxyGroup(
            name=str(g.get("name", "cost_proxy")),
            var_paths=var_paths,
            max_ratio=max_ratio,
        ))

    cfg = PreflightConfig(
        enabled=True,
        max_deviation_factor=max_dev,
        cost_proxy_groups=groups,
        check_var_dependencies=check_deps,
    )
    _log.info(
        "飞行前检查已启用：max_deviation_factor=%s，cost_proxy_groups=%d，check_var_dependencies=%s。",
        max_dev, len(groups), check_deps,
    )
    return cfg


# ---------------------------------------------------------------------------
# boundary_refine 解析（数据驱动边界收缩）
# ---------------------------------------------------------------------------

def _parse_boundary_refine(raw: dict[str, Any] | None) -> Any:
    """
    从顶层 boundary_refine 节解析 BoundaryRefineConfig。

    节不存在或 enabled=false 时返回 None（不启用数据驱动收缩）。

    支持字段
    --------
    enabled         : bool，默认 false
    min_feasible    : int，触发收缩的最少可行样本数，默认 15
    margin_frac     : float，可行点范围两侧裕量比例，默认 0.25
    max_shrink_frac : float，单次收缩后宽度相对原宽度的下限，默认 0.1
    only_shrink     : bool，是否只收不放(与原边界取交集)，默认 true
    interval        : int，每隔多少轮 BO 重估一次（由调用方读取，不入 config）
    """
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"boundary_refine 必须是 dict，收到 {type(raw).__name__!r}：{raw!r}。")

    enabled_raw = raw.get("enabled", False)
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    elif isinstance(enabled_raw, int) and enabled_raw in (0, 1):
        enabled = bool(enabled_raw)
    elif isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() == "true"
    else:
        enabled = False
    if not enabled:
        return None

    from ..optimization.boundary_refine import BoundaryRefineConfig

    def _f(name: str, default: float) -> float:
        try:
            return float(raw.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"boundary_refine.{name} 无法转为浮点数：{exc}") from exc

    def _i(name: str, default: int) -> int:
        try:
            return int(raw.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"boundary_refine.{name} 无法转为整数：{exc}") from exc

    cfg = BoundaryRefineConfig(
        enabled=True,
        min_feasible=_i("min_feasible", 15),
        margin_frac=_f("margin_frac", 0.25),
        max_shrink_frac=_f("max_shrink_frac", 0.1),
        only_shrink=bool(raw.get("only_shrink", True)),
    )
    _log.info(
        "数据驱动边界收缩已启用：min_feasible=%d，margin_frac=%.2f，max_shrink_frac=%.2f。",
        cfg.min_feasible, cfg.margin_frac, cfg.max_shrink_frac,
    )
    return cfg