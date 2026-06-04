"""
optimize_pareto_case.py — 多目标贝叶斯优化 workflow 层封装（ParEGO 随机标量化）。

职责：
  1. 接受多目标优化配置（设计变量边界、目标函数名称列表）
  2. 生成初始 DOE 样本（拉丁超立方采样）
  3. 每次迭代随机生成权重向量，将多目标标量化为单目标
  4. 拟合高斯过程代理模型，通过采集函数推荐下一个候选点
  5. 迭代运行 run_case() 直到达到最大迭代次数
  6. 返回 ParetoOptimizeResult（含所有 ProcessCase、Pareto 前沿和超体积历史）

层级关系
---------
optimize_pareto_case()（本文件）
  ├── run_case()（workflows/run_case.py）
  └── compute_pareto()（optimization/pareto.py）

多目标优化策略：ParEGO 随机标量化
-----------------------------------
每次贝叶斯优化迭代使用随机权重向量将多目标问题标量化为单目标：

  加权和：    scalarized(x) = Σ w_i · f̂_i(x)
  Chebyshev：scalarized(x) = max_i( w_i · f̂_i(x) )

其中 f̂_i 为归一化到 [0,1] 的目标值（基于当前观测范围），
权重 w ~ Dirichlet(1,...,1)（均匀分布在单纯形上），每次迭代重新采样。

随机权重使代理模型在不同迭代中关注 Pareto 前沿的不同区域，
逐步逼近完整 Pareto 前沿。每次迭代重新拟合 GP（基于当前标量化值），
保证代理模型与当前权重一致。

失败工况处理
-----------
仿真失败或目标不可用的工况不参与代理模型拟合，也不参与 Pareto 计算。
仍记录在 ParetoOptimizeResult.cases 中，供失败归因分析。

参考文献
--------
Knowles J., "ParEGO: A Hybrid Algorithm With On-Line Landscape Approximation
for Expensive Multiobjective Optimization Problems", IEEE TEVC 2006.
"""
from __future__ import annotations

import logging
import math
import random as _random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ..aspen_driver.driver import AspenDriver
from ..aspen_driver.errors import AspenConnectionError
from ..models.process_case import CaseStatus, ProcessCase
from ..optimization.pareto import ParetoResult, compute_pareto
from ..optimization.surrogate import SurrogateConfig, make_surrogate_optimizer
from .common import apply_derived_vars, repair_design_vars
from .run_case import RunCaseConfig, run_case

_log = logging.getLogger(__name__)

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Feasibility search 配置
# ---------------------------------------------------------------------------

@dataclass
class FeasibilitySearchConfig:
    """
    Phase 0 可行性搜索配置。

    Attributes
    ----------
    enabled:
        是否启用 Phase 0，默认 True。
    n_trials:
        Phase 0 最多运行的工况数，默认 20。
    stop_after_feasible:
        找到此数量的可行点后提前停止，默认 3。
        设为 0 表示不提前停止，跑完全部 n_trials。
    tags:
        Phase 0 工况的额外标签，默认 ["feasibility_search"]。
    """
    enabled: bool = True
    n_trials: int = 20
    stop_after_feasible: int = 3
    tags: list[str] = field(default_factory=lambda: ["feasibility_search"])


# ---------------------------------------------------------------------------
# 优化配置
# ---------------------------------------------------------------------------

@dataclass
class ParetoOptimizeCaseConfig:
    """
    optimize_pareto_case() 的配置参数。

    Attributes
    ----------
    param_bounds:
        设计变量的搜索边界 {Aspen 树路径: (下界, 上界)}。
        所有变量均为连续实数，下界必须严格小于上界。
    fixed_vars:
        固定不变的设计变量 {Aspen 树路径: 值}，每次运行均使用相同值。
    run_config:
        每次单次运行的配置，见 RunCaseConfig。
    objective_names:
        参与多目标优化的目标函数名称列表，至少 2 个。
        名称须与 run_config.objective_fns 中 ObjectiveValue.name 一致。
        minimize/maximize 方向由各 ObjectiveValue.minimize 字段决定。
    n_initial:
        初始 DOE 样本数（拉丁超立方采样），默认 10。
    n_iterations:
        总迭代次数（含初始 DOE），默认 30。必须 >= n_initial。
    n_initial_min:
        启用高斯过程代理模型所需的最少成功样本数，默认 3。
        不足时贝叶斯优化循环退化为随机采样。
    scalarization:
        标量化方法：
        "weighted_sum"（默认）：加权和 Σ w_i · f̂_i，适合凸 Pareto 前沿。
        "chebyshev"：Chebyshev 标量化 max_i(w_i · f̂_i)，对非凸前沿覆盖更均匀。
    acquisition:
        采集函数类型："EI"（默认）、"UCB"、"PI"。
    xi:
        EI/PI 采集函数的探索参数，默认 0.01。
    kappa:
        UCB 采集函数的探索参数，默认 1.96。
    reference_point:
        超体积计算的参考点（原始值，与目标方向一致）。
        None 时自动从数据推断（各维度最大值 × (1 + hv_margin)）。
    hv_margin:
        自动推断参考点时各维度的扩展比例，默认 0.1。
    tags:
        应用到所有工况的标签列表。
        初始 DOE 工况自动添加 "initial_doe"；贝叶斯优化工况自动添加 "bayesian_opt"。
    on_case_complete:
        每次工况完成后的回调函数，签名为 (case, index, total) -> None。
    db_path:
        SQLite 数据库路径，若指定则每次工况完成后自动持久化。None 不持久化。
    random_seed:
        随机种子，用于 LHS 采样和代理模型的可重复性。
    warm_start_cases:
        预热样本列表（如 Phase 1 DOE 结果）。这些工况不会被重新运行，
        但会在 Phase 2 开始前告知代理模型，使 GP 从已有数据出发，
        避免重复探索 Phase 1 已覆盖的区域。
        warm_start_cases 不计入 n_total / n_success 统计，也不触发回调。
    """
    param_bounds: dict[str, tuple[float, float]]
    objective_names: list[str]
    fixed_vars: dict[str, Any] = field(default_factory=dict)
    run_config: RunCaseConfig = field(default_factory=RunCaseConfig)
    n_initial: int = 10
    n_iterations: int = 30
    n_initial_min: int = 3
    scalarization: Literal["weighted_sum", "chebyshev"] = "weighted_sum"
    acquisition: Literal["EI", "UCB", "PI"] = "EI"
    xi: float = 0.01
    kappa: float = 1.96
    reference_point: list[float] | None = None
    hv_margin: float = 0.1
    tags: list[str] = field(default_factory=list)
    on_case_complete: Callable[[ProcessCase, int, int], None] | None = None
    db_path: Path | str | None = None
    random_seed: int | None = None
    surrogate_model: Literal["GP", "RF", "ET", "GBRT", "random"] = "GP"
    warm_start_cases: list[ProcessCase] = field(default_factory=list)
    # integer 变量路径集合（BO 提出连续值后 round/clamp）
    integer_var_paths: set[str] = field(default_factory=set)
    derived_var_specs: list[dict[str, Any]] = field(default_factory=list)
    # 变量依赖约束 {var_path: {"lt": other_path}} — 如 FEED_STAGE < NSTAGE
    var_dependencies: dict[str, dict[str, str]] = field(default_factory=dict)
    # Phase 0 可行性搜索配置（None 表示不启用）
    feasibility_search: FeasibilitySearchConfig | None = None
    # 每次优化运行的唯一 session_id，默认自动生成
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# 优化结果
# ---------------------------------------------------------------------------

@dataclass
class ParetoOptimizeResult:
    """
    optimize_pareto_case() 的返回值。

    Attributes
    ----------
    cases:
        所有工况的 ProcessCase 列表，顺序与迭代顺序一致。
    pareto_result:
        最终 Pareto 前沿计算结果（含所有层、超体积、拥挤距离）。
    param_bounds:
        本次优化的设计变量边界。
    fixed_vars:
        本次优化的固定变量。
    objective_names:
        多目标优化的目标函数名称列表。
    n_total:
        实际运行的总工况数。
    n_success:
        仿真收敛且所有目标函数均可用的工况数。
    n_sim_failed:
        仿真失败的工况数。
    n_objective_error:
        仿真收敛但目标函数计算失败的工况数。
    n_initial:
        初始 DOE 工况数。
    elapsed:
        总耗时（秒）。
    hv_history:
        每次迭代后的超体积历史列表，长度等于 n_total。
        None 表示截至该迭代成功样本不足，无法计算超体积。
        所有非 None 值均基于同一固定参考点（hv_reference_point），可直接比较。
    hv_reference_point:
        hv_history 使用的固定参考点（最小化方向的内部值）。
        由首批有效 DOE 样本确定，或由用户通过 config.reference_point 指定。
        None 表示整个优化过程中没有任何成功样本，超体积无法计算。
    """
    cases: list[ProcessCase]
    pareto_result: ParetoResult
    param_bounds: dict[str, tuple[float, float]]
    fixed_vars: dict[str, Any]
    objective_names: list[str]
    n_total: int
    n_success: int
    n_sim_failed: int
    n_objective_error: int
    n_initial: int
    elapsed: float
    hv_history: list[float | None]
    hv_reference_point: list[float] | None = None
    session_id: str = ""
    n_phase0: int = 0

    @property
    def first_front(self):
        """第一 Pareto 前沿（非支配集），无有效工况时为 None。"""
        return self.pareto_result.first_front

    @property
    def hypervolume(self) -> float | None:
        """最终超体积指标。"""
        return self.pareto_result.hypervolume

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_total if self.n_total > 0 else 0.0

    def to_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_sim_failed": self.n_sim_failed,
            "n_objective_error": self.n_objective_error,
            "n_initial": self.n_initial,
            "n_phase0": self.n_phase0,
            "success_rate": self.success_rate,
            "objective_names": self.objective_names,
            "hypervolume": self.hypervolume,
            "n_fronts": self.pareto_result.n_fronts,
            "first_front_size": len(self.first_front.cases) if self.first_front else 0,
            "elapsed": self.elapsed,
            "param_bounds": {k: list(v) for k, v in self.param_bounds.items()},
            "hv_reference_point": self.hv_reference_point,
            "pareto_reference_point": self.pareto_result.reference_point,
        }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def optimize_pareto_case(
    driver: AspenDriver,
    config: ParetoOptimizeCaseConfig,
    start_iteration: int = 0,
) -> ParetoOptimizeResult:
    """
    执行多目标贝叶斯优化循环，返回 ParetoOptimizeResult。

    单个工况的意外异常会被隔离为 SIM_FAILED，优化继续执行后续工况。
    若 driver 连接已断开（AspenConnectionError），则终止优化并返回已完成的结果。

    Parameters
    ----------
    driver:
        已连接并打开仿真文件的 AspenDriver 实例。
    config:
        优化配置，见 ParetoOptimizeCaseConfig。
    start_iteration:
        起始迭代编号，默认 0。

    Returns
    -------
    ParetoOptimizeResult
    """
    _validate_config(config)

    paths = list(config.param_bounds.keys())
    bounds = [config.param_bounds[p] for p in paths]
    n_total = config.n_iterations

    _log.info(
        "多目标贝叶斯优化开始：session_id=%s，%d 个设计变量（其中 %d 个 integer），"
        "%d 次初始 DOE，%d 次总迭代，目标=%s，标量化=%s。",
        config.session_id, len(paths), len(config.integer_var_paths),
        config.n_initial, n_total, config.objective_names, config.scalarization,
    )
    _log.info(
        "Surrogate model: %s, acquisition=%s, n_initial_min=%d",
        config.surrogate_model,
        config.acquisition,
        config.n_initial_min,
    )

    db = None
    if config.db_path is not None:
        from ..database.simulation_db import SimulationDB
        db = SimulationDB(config.db_path)

    cases: list[ProcessCase] = []
    case_xs: list[list[float]] = []
    hv_history: list[float | None] = []
    # 固定参考点：首批有效 DOE 样本确定后锁定，保证 hv_history 可比较
    _fixed_ref_point: list[float] | None = None
    t0 = time.monotonic()
    driver_dead = False

    # ------------------------------------------------------------------
    # Phase 0：可行性搜索（可选）
    # ------------------------------------------------------------------
    phase0_cases, driver_dead, phase0_xs = _feasibility_search(
        driver, config, paths, bounds, db, start_iteration
    )
    cases.extend(phase0_cases)
    case_xs.extend(phase0_xs)
    for _ in phase0_cases:
        _fixed_ref_point, hv = _compute_hv_fixed(cases, config, _fixed_ref_point)
        hv_history.append(hv)
    phase0_offset = len(phase0_cases)

    # ------------------------------------------------------------------
    # Phase 1：初始 DOE（拉丁超立方采样）
    # ------------------------------------------------------------------
    initial_points = _lhs_sample(bounds, config.n_initial, config.random_seed)

    for idx, point in enumerate(initial_points):
        if driver_dead:
            break

        design_vars_raw = {**config.fixed_vars, **dict(zip(paths, point))}
        design_vars_repaired, repair_notes = repair_design_vars(
            design_vars_raw,
            config.integer_var_paths,
            config.param_bounds,
            config.var_dependencies,
        )
        design_vars, derived_notes = apply_derived_vars(
            design_vars_repaired, config.derived_var_specs,
        )
        x_eval = [design_vars_repaired.get(p, point[i]) for i, p in enumerate(paths)]
        if repair_notes:
            _log.debug("Phase 1 repair [%d/%d]: %s", idx + 1, config.n_initial, repair_notes)
        if derived_notes:
            _log.debug("Phase 1 derived [%d/%d]: %s", idx + 1, config.n_initial, derived_notes)

        iteration = start_iteration + phase0_offset + idx
        tags = list(config.tags) + ["initial_doe", "pareto_opt"]

        _log.info(
            "初始 DOE [%d/%d]：%s",
            idx + 1, config.n_initial,
            {_short_var_name(k): round(v, 4) for k, v in design_vars_repaired.items()
             if k in config.param_bounds},
        )

        try:
            run_id = str(uuid.uuid4())
            case = run_case(
                driver=driver,
                design_vars=design_vars,
                config=config.run_config,
                iteration=iteration,
                tags=tags,
                run_id=run_id,
            )
        except AspenConnectionError as exc:
            _log.error("初始 DOE [%d/%d]：driver 连接断开，终止优化。原因：%s",
                       idx + 1, config.n_initial, exc)
            driver_dead = True
            case = ProcessCase(
                iteration=iteration, status=CaseStatus.SIM_FAILED,
                design_vars=design_vars, tags=tags,
                notes=f"driver 连接断开，优化终止：{exc}",
            )
            cases.append(case)
            case_xs.append(x_eval)
            _save_case(db, case, config.session_id)
            _fire_callback(config.on_case_complete, case, idx, n_total)
            _fixed_ref_point, hv = _compute_hv_fixed(cases, config, _fixed_ref_point)
            hv_history.append(hv)
            break
        except Exception as exc:
            _log.warning("初始 DOE [%d/%d]：run_case() 意外异常（已隔离）：%s",
                         idx + 1, config.n_initial, exc)
            case = ProcessCase(
                iteration=iteration, status=CaseStatus.SIM_FAILED,
                design_vars=design_vars, tags=tags,
                notes=f"run_case() 意外异常：{exc}",
            )

        cases.append(case)
        case_xs.append(x_eval)
        _save_case(db, case, config.session_id)
        _fire_callback(config.on_case_complete, case, idx, n_total)
        _fixed_ref_point, hv = _compute_hv_fixed(cases, config, _fixed_ref_point)
        hv_history.append(hv)
        _log.info("  → status=%s, success=%s, run_time=%.1fs",
                  case.status.value, case.success, case.run_time)
        if case.status == CaseStatus.OBJECTIVE_ERROR:
            for obj in (case.objectives or []):
                if getattr(obj, "error", None):
                    _log.info("    [%s] error: %s", obj.name, obj.error)

    # ------------------------------------------------------------------
    # Phase 2：贝叶斯优化循环
    # ------------------------------------------------------------------
    n_bo = n_total - config.n_initial

    if not driver_dead and n_bo > 0:
        optimizer = _MultiObjectiveBayesianOptimizer(bounds, config, paths)

        for c, x in zip(cases, case_xs):
            y_vec = _extract_all_objectives(c, config)
            optimizer.tell(x, y_vec, is_success=y_vec is not None)

        # warm_start_cases：注入 Phase 1 数据，不重新运行，不计入统计
        n_warm = 0
        for c in config.warm_start_cases:
            x = [c.design_vars.get(p) for p in paths]
            if None in x:
                continue
            y_vec = _extract_all_objectives(c, config)
            optimizer.tell(x, y_vec, is_success=y_vec is not None)
            n_warm += 1
        if n_warm > 0:
            _log.info("warm_start：已注入 %d 个 Phase 1 样本到代理模型。", n_warm)

        n_success_so_far = sum(
            1 for c in cases if _extract_all_objectives(c, config) is not None
        ) + sum(
            1 for c in config.warm_start_cases
            if _extract_all_objectives(c, config) is not None
        )
        if n_success_so_far < config.n_initial_min:
            _log.warning(
                "初始 DOE 成功样本数 %d < n_initial_min=%d，"
                "贝叶斯优化循环将以随机采样替代代理模型 %s。",
                n_success_so_far, config.n_initial_min, config.surrogate_model,
            )

        for bo_idx in range(n_bo):
            if driver_dead:
                break

            idx = config.n_initial + bo_idx
            iteration = start_iteration + phase0_offset + idx
            tags = list(config.tags) + ["bayesian_opt", "pareto_opt"]

            next_x = optimizer.ask()
            design_vars_raw = {**config.fixed_vars, **dict(zip(paths, next_x))}
            design_vars_repaired, repair_notes = repair_design_vars(
                design_vars_raw,
                config.integer_var_paths,
                config.param_bounds,
                config.var_dependencies,
            )
            design_vars, derived_notes = apply_derived_vars(
                design_vars_repaired, config.derived_var_specs,
            )
            x_eval = [design_vars_repaired.get(p, next_x[i]) for i, p in enumerate(paths)]
            if repair_notes:
                _log.debug("Phase 2 repair [%d/%d]: %s", idx + 1, n_total, repair_notes)
            if derived_notes:
                _log.debug("Phase 2 derived [%d/%d]: %s", idx + 1, n_total, derived_notes)

            _log.info(
                "贝叶斯优化 [%d/%d]：%s",
                idx + 1, n_total,
                {_short_var_name(k): round(v, 4) for k, v in design_vars_repaired.items()
                 if k in config.param_bounds},
            )

            try:
                run_id = str(uuid.uuid4())
                case = run_case(
                    driver=driver,
                    design_vars=design_vars,
                    config=config.run_config,
                    iteration=iteration,
                    tags=tags,
                    run_id=run_id,
                )
            except AspenConnectionError as exc:
                _log.error("贝叶斯优化 [%d/%d]：driver 连接断开，终止优化。原因：%s",
                           idx + 1, n_total, exc)
                driver_dead = True
                case = ProcessCase(
                    iteration=iteration, status=CaseStatus.SIM_FAILED,
                    design_vars=design_vars, tags=tags,
                    notes=f"driver 连接断开，优化终止：{exc}",
                )
                cases.append(case)
                case_xs.append(x_eval)
                _save_case(db, case, config.session_id)
                _fire_callback(config.on_case_complete, case, idx, n_total)
                _fixed_ref_point, hv = _compute_hv_fixed(cases, config, _fixed_ref_point)
                hv_history.append(hv)
                break
            except Exception as exc:
                _log.warning("贝叶斯优化 [%d/%d]：run_case() 意外异常（已隔离）：%s",
                             idx + 1, n_total, exc)
                case = ProcessCase(
                    iteration=iteration, status=CaseStatus.SIM_FAILED,
                    design_vars=design_vars, tags=tags,
                    notes=f"run_case() 意外异常：{exc}",
                )

            cases.append(case)
            case_xs.append(x_eval)
            _save_case(db, case, config.session_id)
            _fire_callback(config.on_case_complete, case, idx, n_total)

            y_vec = _extract_all_objectives(case, config)
            # 用修复后的实际运行点告知代理模型，保证 surrogate 学习正确的输入-输出关系。
            # next_x 是 BO 推荐的连续值，经 repair 后可能被 round/clamp，
            # 实际跑 Aspen 的是 design_vars，必须以此为准。
            optimizer.tell(x_eval, y_vec, is_success=y_vec is not None)
            _fixed_ref_point, hv = _compute_hv_fixed(cases, config, _fixed_ref_point)
            hv_history.append(hv)

            _log.info("  → status=%s, success=%s, run_time=%.1fs",
                      case.status.value, case.success, case.run_time)
            if case.status == CaseStatus.OBJECTIVE_ERROR:
                for obj in (case.objectives or []):
                    if getattr(obj, "error", None):
                        _log.info("    [%s] error: %s", obj.name, obj.error)

    # ------------------------------------------------------------------
    # 汇总结果
    # ------------------------------------------------------------------
    try:
        elapsed = time.monotonic() - t0

        n_success         = sum(1 for c in cases if _extract_all_objectives(c, config) is not None)
        n_sim_failed      = sum(1 for c in cases if c.status == CaseStatus.SIM_FAILED)
        n_objective_error = sum(1 for c in cases if c.status == CaseStatus.OBJECTIVE_ERROR)

        # 最终 Pareto 计算使用与 hv_history 相同的固定参考点，保证 HV 值一致。
        # _fixed_ref_point 是最小化方向的内部值，需还原为原始方向后传给 compute_pareto。
        final_ref_raw: list[float] | None = None
        if _fixed_ref_point is not None:
            sample = next((c for c in cases if c.success), None)
            if sample is not None:
                from ..optimization.pareto import _restore_reference_point
                final_ref_raw = _restore_reference_point(
                    _fixed_ref_point, sample, config.objective_names
                )

        pareto_result = compute_pareto(
            cases,
            config.objective_names,
            reference_point=final_ref_raw,
            hv_margin=config.hv_margin,
            compute_hv=True,
        )

        if driver_dead:
            _log.warning(
                "多目标优化因 driver 断开提前终止：已完成 %d/%d 个工况，%d 成功，耗时 %.1fs。",
                len(cases), n_total, n_success, elapsed,
            )
        else:
            _log.info(
                "多目标优化完成：%d/%d 成功，第一前沿 %d 个解，HV=%s，总耗时 %.1fs。",
                n_success, len(cases),
                len(pareto_result.first_front.cases) if pareto_result.first_front else 0,
                f"{pareto_result.hypervolume:.4g}" if pareto_result.hypervolume is not None else "N/A",
                elapsed,
            )

        if n_success == 0:
            _log_infeasible_diagnosis(cases)

        return ParetoOptimizeResult(
            cases=cases,
            pareto_result=pareto_result,
            param_bounds=config.param_bounds,
            fixed_vars=config.fixed_vars,
            objective_names=config.objective_names,
            n_total=len(cases),
            n_success=n_success,
            n_sim_failed=n_sim_failed,
            n_objective_error=n_objective_error,
            n_initial=config.n_initial,
            elapsed=elapsed,
            hv_history=hv_history,
            hv_reference_point=_fixed_ref_point,
            session_id=config.session_id,
            n_phase0=phase0_offset,
        )
    finally:
        if db is not None:
            db.close()


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

def _validate_config(config: ParetoOptimizeCaseConfig) -> None:
    if not config.param_bounds:
        raise ValueError("param_bounds 不能为空，至少需要一个设计变量。")

    for path, (lo, hi) in config.param_bounds.items():
        if lo >= hi:
            raise ValueError(
                f"param_bounds['{path}'] 的下界 {lo} >= 上界 {hi}，"
                "请确保下界严格小于上界。"
            )

    if len(config.objective_names) < 2:
        raise ValueError(
            f"objective_names 至少需要 2 个目标，收到 {len(config.objective_names)} 个。"
            "单目标优化请使用 optimize_case()。"
        )

    if config.n_initial < 1:
        raise ValueError(f"n_initial 必须 >= 1，收到：{config.n_initial}。")

    if config.n_iterations < config.n_initial:
        raise ValueError(
            f"n_iterations={config.n_iterations} 必须 >= n_initial={config.n_initial}。"
        )

    if config.scalarization not in ("weighted_sum", "chebyshev"):
        raise ValueError(
            f"scalarization 必须为 'weighted_sum' 或 'chebyshev'，收到：{config.scalarization!r}。"
        )

    if config.acquisition not in ("EI", "UCB", "PI"):
        raise ValueError(
            f"acquisition 必须为 'EI'、'UCB' 或 'PI'，收到：{config.acquisition!r}。"
        )

    _VALID_SURROGATE = {"GP", "RF", "ET", "GBRT", "random"}
    if config.surrogate_model not in _VALID_SURROGATE:
        raise ValueError(
            f"surrogate_model={config.surrogate_model!r} 不合法，"
            f"支持值：{sorted(_VALID_SURROGATE)}。"
        )

    if config.hv_margin < 0:
        raise ValueError(f"hv_margin 必须 >= 0，收到：{config.hv_margin}。")

    if config.reference_point is not None:
        n_obj = len(config.objective_names)
        if len(config.reference_point) != n_obj:
            raise ValueError(
                f"reference_point 维度 {len(config.reference_point)} 与 "
                f"objective_names 数量 {n_obj} 不一致。"
            )
        for i, v in enumerate(config.reference_point):
            if not math.isfinite(v):
                raise ValueError(
                    f"reference_point[{i}]={v!r} 为非有限数（NaN/Inf），"
                    "请提供有效的参考点。"
                )

    param_upper = {p.upper() for p in config.param_bounds}
    fixed_upper = {p.upper(): p for p in config.fixed_vars}
    conflicts = [
        (fixed_upper[u], next(p for p in config.param_bounds if p.upper() == u))
        for u in param_upper & set(fixed_upper)
    ]
    if conflicts:
        detail = "; ".join(f"fixed={f!r} vs param={s!r}" for f, s in conflicts)
        raise ValueError(
            f"fixed_vars 与 param_bounds 存在路径冲突（大小写不敏感）：{detail}。"
        )

    for path in config.integer_var_paths:
        if path not in config.param_bounds:
            raise ValueError(
                f"integer_var_paths 中的路径 '{path}' 不在 param_bounds 中，"
                "请确认路径拼写正确。"
            )
        lo, hi = config.param_bounds[path]
        if lo != int(lo) or hi != int(hi):
            raise ValueError(
                f"integer 变量 '{path}' 的边界 ({lo}, {hi}) 必须为整数值，"
                "请将 YAML 中的 lower_bound / upper_bound 改为整数（如 10, 50）。"
            )

    for spec in config.derived_var_specs:
        frac_path = str(spec.get("frac_path", ""))
        target_path = str(spec.get("target_path", ""))
        depends_on = str(spec.get("depends_on", ""))
        if frac_path not in config.param_bounds:
            raise ValueError(f"derived frac_path '{frac_path}' is not in param_bounds.")
        if target_path in config.param_bounds:
            raise ValueError(
                f"derived target_path '{target_path}' must not also be optimized."
            )
        if depends_on not in config.param_bounds and depends_on not in config.fixed_vars:
            raise ValueError(
                f"derived depends_on '{depends_on}' must be optimized or fixed."
            )
        frac_lo = int(spec.get("frac_lo", 1))
        lo, hi = config.param_bounds[frac_path]
        if lo < 0.0 or hi > 1.0 or lo >= hi:
            raise ValueError(
                f"derived frac bounds for '{frac_path}' must satisfy 0 <= lo < hi <= 1."
            )
        dep_bounds = config.param_bounds.get(depends_on)
        dep_min = dep_bounds[0] if dep_bounds is not None else float(config.fixed_vars[depends_on])
        if dep_min <= frac_lo:
            raise ValueError(
                f"derived frac_lo={frac_lo} must be smaller than minimum dependency "
                f"value {dep_min} for '{depends_on}'."
            )


# ---------------------------------------------------------------------------
# 拉丁超立方采样（与 optimize_case.py 保持一致）
# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _short_var_name(path: str) -> str:
    """
    从 Aspen 路径生成简短显示名，避免多变量名称覆盖。

    对于 Aspen 路径 \\Data\\Blocks\\T1\\Input\\BASIS_RR，
    返回 T1\\BASIS_RR（block 名 + 参数名）。
    对于其他路径，取最后两段。
    """
    parts = path.replace("/", "\\").split("\\")
    parts = [p for p in parts if p]  # 去除空段
    if len(parts) >= 2:
        # 尝试找到 Blocks 或 Streams 后的 block/stream 名
        for i, p in enumerate(parts):
            if p.upper() in ("BLOCKS", "STREAMS") and i + 1 < len(parts):
                block_name = parts[i + 1]
                last = parts[-1]
                return f"{block_name}\\{last}" if block_name != last else last
        return "\\".join(parts[-2:])
    return parts[-1] if parts else path


# ---------------------------------------------------------------------------
# Phase 0：可行性搜索
# ---------------------------------------------------------------------------

def _feasibility_search(
    driver: AspenDriver,
    config: "ParetoOptimizeCaseConfig",
    paths: list[str],
    bounds: list[tuple[float, float]],
    db: Any,
    start_iteration: int,
) -> tuple[list[ProcessCase], bool, list[list[float]]]:
    """
    Phase 0：可行性搜索。

    用 LHS 采样 n_trials 个点，找到 stop_after_feasible 个可行点后提前停止。
    返回 (all_cases, driver_dead)。
    """
    fs_cfg = config.feasibility_search
    if fs_cfg is None or not fs_cfg.enabled:
        return [], False, []

    n_trials   = fs_cfg.n_trials
    stop_after = fs_cfg.stop_after_feasible
    tags = list(config.tags) + list(fs_cfg.tags) + ["pareto_opt"]

    _log.info(
        "Phase 0 可行性搜索：最多 %d 次，找到 %d 个可行点后停止。",
        n_trials, stop_after,
    )

    points = _lhs_sample(bounds, n_trials, config.random_seed)
    cases: list[ProcessCase] = []
    case_xs: list[list[float]] = []
    n_feasible = 0
    driver_dead = False

    for idx, point in enumerate(points):
        if driver_dead:
            break
        if stop_after > 0 and n_feasible >= stop_after:
            _log.info("Phase 0：已找到 %d 个可行点，提前停止。", n_feasible)
            break

        design_vars_raw = {**config.fixed_vars, **dict(zip(paths, point))}
        design_vars_repaired, repair_notes = repair_design_vars(
            design_vars_raw,
            config.integer_var_paths,
            config.param_bounds,
            config.var_dependencies,
        )
        design_vars, derived_notes = apply_derived_vars(
            design_vars_repaired, config.derived_var_specs,
        )
        x_eval = [design_vars_repaired.get(p, point[i]) for i, p in enumerate(paths)]
        if repair_notes:
            _log.debug("Phase 0 repair [%d/%d]: %s", idx + 1, n_trials, repair_notes)
        if derived_notes:
            _log.debug("Phase 0 derived [%d/%d]: %s", idx + 1, n_trials, derived_notes)

        iteration = start_iteration + idx
        _log.info(
            "Phase 0 [%d/%d]：%s",
            idx + 1, n_trials,
            {_short_var_name(k): round(v, 4) for k, v in design_vars_repaired.items()
             if k in config.param_bounds},
        )

        try:
            run_id = str(uuid.uuid4())
            case = run_case(
                driver=driver,
                design_vars=design_vars,
                config=config.run_config,
                iteration=iteration,
                tags=tags,
                run_id=run_id,
            )
        except AspenConnectionError as exc:
            _log.error("Phase 0 [%d/%d]：driver 连接断开。原因：%s", idx + 1, n_trials, exc)
            driver_dead = True
            case = ProcessCase(
                iteration=iteration, status=CaseStatus.SIM_FAILED,
                design_vars=design_vars, tags=tags,
                notes=f"driver 连接断开：{exc}",
            )
            cases.append(case)
            case_xs.append(x_eval)
            _save_case(db, case, config.session_id)
            break
        except Exception as exc:
            _log.warning("Phase 0 [%d/%d]：run_case() 意外异常：%s", idx + 1, n_trials, exc)
            case = ProcessCase(
                iteration=iteration, status=CaseStatus.SIM_FAILED,
                design_vars=design_vars, tags=tags,
                notes=f"run_case() 意外异常：{exc}",
            )

        cases.append(case)
        case_xs.append(x_eval)
        _save_case(db, case, config.session_id)

        if case.feasible is True:
            n_feasible += 1
            _log.info("  → 可行点 #%d（status=%s）", n_feasible, case.status.value)
        else:
            _log.info(
                "  → 不可行（status=%s, feasible=%s）",
                case.status.value, case.feasible,
            )

    _log.info(
        "Phase 0 完成：运行 %d 次，找到 %d 个可行点。",
        len(cases), n_feasible,
    )
    return cases, driver_dead, case_xs


def _lhs_sample(
    bounds: list[tuple[float, float]],
    n: int,
    seed: int | None,
) -> list[list[float]]:
    d = len(bounds)
    if _HAS_NUMPY:
        rng = _np.random.default_rng(seed)
        samples = _np.zeros((n, d))
        for j, (lo, hi) in enumerate(bounds):
            perm = rng.permutation(n)
            u = (perm + rng.random(n)) / n
            samples[:, j] = lo + u * (hi - lo)
        return samples.tolist()

    rng = _random.Random(seed)
    cols: list[list[float]] = []
    for lo, hi in bounds:
        perm = list(range(n))
        rng.shuffle(perm)
        col = [lo + (perm[i] + rng.random()) / n * (hi - lo) for i in range(n)]
        cols.append(col)
    return [[cols[j][i] for j in range(d)] for i in range(n)]


# ---------------------------------------------------------------------------
# 多目标贝叶斯优化器（ParEGO 随机标量化）
# ---------------------------------------------------------------------------

class _MultiObjectiveBayesianOptimizer:
    """
    多目标贝叶斯优化器，基于 ParEGO 随机标量化策略。

    每次 ask() 时：
      1. 从 Dirichlet(1,...,1) 采样随机权重向量
      2. 对所有历史观测计算当前权重下的标量化值
      3. 用标量化值重新拟合 GP，通过采集函数推荐下一个候选点

    成功观测数 < n_initial_min 时退化为随机采样。
    skopt 不可用时始终随机采样。
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        config: ParetoOptimizeCaseConfig,
        paths: list[str],
    ) -> None:
        self._bounds = bounds
        self._n_obj = len(config.objective_names)
        self._n_initial_min = config.n_initial_min
        self._scalarization = config.scalarization
        self._acquisition = config.acquisition
        self._xi = config.xi
        self._kappa = config.kappa
        self._surrogate_model = config.surrogate_model
        self._rng = _random.Random(config.random_seed)
        # 将 integer_var_paths（Aspen 路径集合）转换为 bounds 的下标集合，
        # 供 make_surrogate_optimizer 构建混合整数搜索空间。
        self._integer_indices: set[int] = {
            i for i, p in enumerate(paths) if p in config.integer_var_paths
        }
        # 成功观测：(x, y_vec_min_direction)
        self._observations: list[tuple[list[float], list[float]]] = []
        # 失败观测：只存 x，penalty 在 ask() 时用当前权重动态计算
        self._failed_xs: list[list[float]] = []

    def tell(
        self,
        x: list[float],
        y_vec: list[float] | None,
        *,
        is_success: bool,
        penalty: float = 1e10,  # 保留参数签名兼容性，实际不使用
    ) -> None:
        """
        提交一次观测。

        成功样本存入 _observations 参与 GP 拟合。
        失败样本只存 x，penalty 在 ask() 时用当前权重动态计算，
        保证失败区域的惩罚信号与成功样本在同一标量化体系下，可复现。
        """
        if is_success and y_vec is not None:
            self._observations.append((list(x), list(y_vec)))
        else:
            self._failed_xs.append(list(x))

    def ask(self) -> list[float]:
        """推荐下一个候选点。成功观测不足 n_initial_min 时返回随机点。"""
        if len(self._observations) < self._n_initial_min:
            return [lo + self._rng.random() * (hi - lo) for lo, hi in self._bounds]

        weights = _dirichlet_sample(self._n_obj, self._rng)
        scalarized = [
            _scalarize(y_vec, weights, self._scalarization, self._observations)
            for _, y_vec in self._observations
        ]

        if scalarized:
            worst_scalar = max(scalarized)
            penalty = worst_scalar + max(abs(worst_scalar) * 0.1, 1.0)
        else:
            penalty = 1e10

        try:
            surrogate_cfg = SurrogateConfig(
                model=self._surrogate_model,
                acquisition=self._acquisition,
                xi=self._xi,
                kappa=self._kappa,
                n_initial_min=0,
                random_seed=self._rng.randint(0, 2 ** 31),
            )
            opt = make_surrogate_optimizer(self._bounds, surrogate_cfg, self._integer_indices)
            for (x, _), s in zip(self._observations, scalarized):
                opt.tell(x, s, is_success=True)
            for x in self._failed_xs:
                opt.tell(x, penalty, is_success=False)
            return opt.ask()
        except Exception as exc:
            _log.warning("代理模型多目标优化失败，回退到随机采样：%s", exc)
            return [lo + self._rng.random() * (hi - lo) for lo, hi in self._bounds]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _extract_all_objectives(
    case: ProcessCase,
    config: ParetoOptimizeCaseConfig,
) -> list[float] | None:
    """
    从 ProcessCase 提取所有目标值（统一转为最小化方向）。

    任意目标不可用、或值为 NaN/Inf 时返回 None，不参与代理模型拟合。
    """
    if not case.success:
        return None
    result: list[float] = []
    for name in config.objective_names:
        obj = case.get_objective(name)
        if obj is None or not obj.available:
            return None
        val = float(obj.value)  # type: ignore[arg-type]
        if not math.isfinite(val):
            return None
        result.append(val if obj.minimize else -val)
    return result


def _normalize_objectives(
    y_vec: list[float],
    observations: list[tuple[list[float], list[float]]],
) -> list[float]:
    """
    将目标向量归一化到 [0,1]（基于当前所有观测的范围）。

    某维度范围为 0 时（所有观测值相同），归一化值设为 0.0。
    """
    n_obj = len(y_vec)
    result: list[float] = []
    for i in range(n_obj):
        vals = [obs[1][i] for obs in observations]
        f_min = min(vals)
        f_max = max(vals)
        span = f_max - f_min
        result.append(0.0 if span < 1e-10 else (y_vec[i] - f_min) / span)
    return result


def _scalarize(
    y_vec: list[float],
    weights: list[float],
    method: str,
    observations: list[tuple[list[float], list[float]]],
) -> float:
    """
    将目标向量标量化为单个值（最小化方向）。

    先归一化再加权，避免不同量纲目标的尺度差异影响权重效果。
    """
    y_norm = _normalize_objectives(y_vec, observations)
    if method == "chebyshev":
        return max(w * y for w, y in zip(weights, y_norm))
    return sum(w * y for w, y in zip(weights, y_norm))


def _dirichlet_sample(n: int, rng: _random.Random) -> list[float]:
    """
    从 Dirichlet(1,...,1) 采样，即在 n 维单纯形上均匀采样权重向量。

    使用指数分布变换：x_i ~ Exp(1)，归一化后服从 Dirichlet(1,...,1)。
    """
    xs = [-math.log(rng.random() + 1e-300) for _ in range(n)]
    total = sum(xs)
    return [x / total for x in xs]


def _compute_hv_fixed(
    cases: list[ProcessCase],
    config: ParetoOptimizeCaseConfig,
    fixed_ref: list[float] | None,
) -> tuple[list[float] | None, float | None]:
    """
    计算当前 Pareto 前沿超体积，并维护固定参考点。

    首次有足够成功样本时，从数据推断参考点并锁定（或使用用户指定值）。
    后续所有迭代复用同一参考点，保证 hv_history 各值可直接比较。

    Returns
    -------
    (fixed_ref, hv):
        fixed_ref — 本次确定或沿用的固定参考点（最小化方向内部值）。
        hv        — 本次超体积值；样本不足时为 None。
    """
    from ..optimization.pareto import (
        _extract_objectives as _ext_obj,
        infer_reference_point,
        hypervolume,
        fast_non_dominated_sort,
    )

    try:
        # 提取所有成功样本的目标向量（最小化方向）
        vecs: list[list[float]] = []
        for c in cases:
            if not c.success:
                continue
            v = _ext_obj(c, config.objective_names)
            if v is not None:
                vecs.append(v)

        if len(vecs) < 2:
            return fixed_ref, None

        # 首次锁定参考点
        if fixed_ref is None:
            if config.reference_point is not None:
                # 用户指定值转换为最小化方向
                from ..optimization.pareto import _reference_point_to_min
                # 需要一个 sample_case 来判断 minimize 方向
                sample = next(c for c in cases if c.success)
                fixed_ref = _reference_point_to_min(
                    config.reference_point, sample, config.objective_names
                )
            else:
                fixed_ref = infer_reference_point(vecs, margin=config.hv_margin)
            _log.debug("超体积参考点已锁定：%s", [round(v, 4) for v in fixed_ref])

        # 计算第一前沿的超体积
        front_indices = fast_non_dominated_sort(vecs)
        first_front_vecs = [vecs[i] for i in front_indices[0]]
        hv = hypervolume(first_front_vecs, fixed_ref)
        return fixed_ref, hv

    except Exception as exc:
        _log.debug("超体积快照计算失败（已忽略）：%s", exc)
        return fixed_ref, None


def _log_infeasible_diagnosis(
    cases: list[ProcessCase],
    n_show: int = 5,
) -> None:
    """
    当 n_success=0 时，按约束违反程度排序输出最接近可行的工况。

    输出每个工况的：约束名/实际值/差距、设计变量值。
    仅处理 status=INFEASIBLE 且约束全部可用的工况。
    """
    infeasible = [
        c for c in cases
        if c.status == CaseStatus.INFEASIBLE and c.constraints_available
    ]
    if not infeasible:
        obj_err = [c for c in cases if c.status.value == "objective_error"]
        if obj_err:
            _log.info(
                "infeasible 诊断：无 INFEASIBLE 工况，但有 %d 个 OBJECTIVE_ERROR 工况。"
                "请检查 TAC/EMISSIONS 目标函数错误信息（见上方日志）。",
                len(obj_err),
            )
        return

    def total_violation(c: ProcessCase) -> float:
        return sum(max(0.0, cv.value) for cv in c.constraints if cv.value is not None)

    infeasible.sort(key=total_violation)
    _log.info(
        "=== infeasible 诊断：最接近可行的 %d/%d 个工况（按约束违反量升序）===",
        min(n_show, len(infeasible)), len(infeasible),
    )
    for rank, c in enumerate(infeasible[:n_show], 1):
        viol = total_violation(c)
        con_parts = []
        for cv in c.constraints:
            if cv.value is None:
                con_parts.append(f"{cv.name}=None")
            else:
                status_str = "✓" if cv.satisfied else f"违反+{cv.value:.4f}"
                con_parts.append(f"{cv.name}={status_str}")
        dv_short = {_short_var_name(k): round(v, 4) for k, v in c.design_vars.items()}
        _log.info(
            "  #%d iter=%d 总违反=%.4f | %s | vars=%s",
            rank, c.iteration, viol, " | ".join(con_parts), dv_short,
        )


def _save_case(db: Any, case: ProcessCase, session_id: str = "") -> None:
    if db is None:
        return
    try:
        d = case.to_dict()
        if session_id:
            d["session_id"] = session_id
        db.save_case(d)
    except Exception as exc:
        _log.warning("工况 '%s' 保存到数据库失败（已忽略）：%s", case.case_id, exc)


def _fire_callback(
    callback: Callable[[ProcessCase, int, int], None] | None,
    case: ProcessCase,
    idx: int,
    total: int,
) -> None:
    if callback is None:
        return
    try:
        callback(case, idx, total)
    except Exception as exc:
        _log.warning("on_case_complete 回调异常（已忽略）：%s", exc)
