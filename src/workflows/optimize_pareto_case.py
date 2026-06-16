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
from ..optimization.feasibility import FeasibilityClassifier, FeasibilityConfig
from ..optimization.pareto import ParetoResult, compute_pareto
from ..optimization.surrogate import SurrogateConfig, make_surrogate_optimizer
from .common import (
    EarlyStoppingConfig,
    apply_derived_vars,
    build_evaluated_set,
    feasibility_feature_names,
    fingerprint_design_vars,
    pick_first_unseen_candidate,
    repair_design_vars,
)
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
        Phase 0 最多运行的工况数，默认 20。此计数包含 initial_point（若提供）。
        局部扩张模式下，每个半径最多使用 n_trials 次，但总次数仍受此限制。
    stop_after_feasible:
        找到此数量的可行点后提前停止，默认 3。
        设为 0 表示不提前停止，跑完全部 n_trials。
    abort_if_none_found:
        True（默认）：n_trials 结束后仍未找到任何可行点，直接终止优化并抛出 RuntimeError。
        False：即使 Phase 0 全部失败，仍继续 Phase 1/2（代理模型降级为随机采样）。
    initial_point:
        从 YAML initial_value 提取的初始点 {Aspen路径: 值}。
        若不为 None，Phase 0 的第一个候选点强制使用此值，而非随机采样。
        此点来自用户上传的已收敛 .bkp 文件，大概率可行，可显著提升 Phase 0 成功率。
    local_search_radii:
        自适应局部扩张半径列表，默认 [0.2, 0.5]。
        需同时提供 initial_point 才会生效。
        Phase 0 先在 initial_point ± radius[0] 范围内 LHS 采样；
        若可行率为 0，自动扩展到 radius[1]，依此类推。
        所有半径都失败后，用全局 bounds 兜底一轮。
        设为 [] 跳过局部搜索，退化为原有全局随机采样。
    tags:
        Phase 0 工况的额外标签，默认 ["feasibility_search"]。
    """
    enabled: bool = True
    n_trials: int = 20
    stop_after_feasible: int = 3
    abort_if_none_found: bool = True
    initial_point: dict | None = None
    local_search_radii: list[float] = field(default_factory=lambda: [0.2, 0.5])
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
    surrogate_model: Literal["GP", "RF", "ET", "GBRT", "random", "qEHVI", "NEHVI"] = "GP"
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
    # BO 阶段候选池可行性过滤；enabled=False（默认）时完全不影响原有流程
    # 注意：与 feasibility_search（Phase 0 初始可行点搜索）是独立功能，可并存
    feasibility_filter: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    # 早停配置；enabled=False（默认）时完全不影响原有流程
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    # Trust Region 配置；None 表示不启用（使用全局 bounds 采样）
    trust_region: Any = None
    # 敏感度探针配置；None 表示不启用
    sensitivity_probe: Any = None
    # 飞行前检查配置；None 表示不启用(提交 Aspen 前拦截荒谬工况)
    preflight: Any = None
    # 飞行前检查的参考工况值 {Aspen路径: 值},通常是初始收敛解;
    # None/空 时偏离检查与计算量代理检查自动跳过(不影响依赖检查)
    reference_values: dict[str, float] = field(default_factory=dict)
    # 数据驱动边界收缩配置；None 表示不启用(用实际可行样本周期性收紧边界)
    boundary_refine: Any = None
    # 每隔多少轮 BO 触发一次 boundary_refine 重估,默认 20
    boundary_refine_interval: int = 20


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
    early_stopped: bool = False
    early_stop_reason: str | None = None
    completed_iterations: int = 0
    no_improvement_count: int = 0
    duplicate_skipped_iterations: int = 0
    no_unique_candidate_count: int = 0

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
            "early_stopped": self.early_stopped,
            "early_stop_reason": self.early_stop_reason,
            "completed_iterations": self.completed_iterations,
            "no_improvement_count": self.no_improvement_count,
            "duplicate_skipped_iterations": self.duplicate_skipped_iterations,
            "no_unique_candidate_count": self.no_unique_candidate_count,
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
    early_stopped = False
    early_stop_reason: str | None = None
    no_improvement_count = 0
    duplicate_skipped_iterations = 0
    no_unique_candidate_count = 0

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
    # Phase 0.5（可选）：敏感度探针
    # ------------------------------------------------------------------
    probe_result = None
    thaw_scheduler = None
    doe_bounds = bounds  # 默认使用全局 bounds，探针成功后覆盖

    if (
        config.sensitivity_probe is not None
        and config.sensitivity_probe.enabled
        and not driver_dead
    ):
        from ..optimization.sensitivity_probe import (
            SensitivityResult,
            ThawScheduler,
            adaptive_doe_bounds,
            run_sensitivity_probe,
        )

        # 找 Phase 0 找到的第一个可行点作为探针中心
        first_feasible = next((c for c in cases if c.success), None)
        if first_feasible is not None:
            # center_dict 以搜索空间路径（paths）为键构建。
            # 优先级：
            #   1. first_feasible.design_vars（展开后的 Aspen 路径，覆盖 continuous/integer）
            #   2. feasibility_search.initial_point（含 derived var 的 frac 初值）
            #   3. 全局 bounds 中点（兜底，不应依赖）
            _initial_point_fracs: dict[str, float] = {}
            if (
                config.feasibility_search is not None
                and config.feasibility_search.initial_point
            ):
                _initial_point_fracs = {
                    str(k): float(v)
                    for k, v in config.feasibility_search.initial_point.items()
                }

            center_dict: dict[str, float] = {}
            for i, p in enumerate(paths):
                if p in first_feasible.design_vars:
                    # continuous / integer：直接用 design_vars 里的已收敛值
                    center_dict[p] = float(first_feasible.design_vars[p])
                elif p in _initial_point_fracs:
                    # derived（frac 路径）：用 YAML initial_value 反算的 frac
                    center_dict[p] = _initial_point_fracs[p]
                else:
                    # 兜底：bounds 中点
                    center_dict[p] = (bounds[i][0] + bounds[i][1]) / 2.0
            integer_idx_set: set[int] = {
                i for i, p in enumerate(paths) if p in config.integer_var_paths
            }

            # 提取 center 点的约束 margin，供 run_sensitivity_probe 计算相对下降量。
            # margin = -ConstraintValue.value（value = threshold - actual，margin > 0 表示满足）
            _center_margins: dict[str, float] = {}
            if first_feasible.constraints:
                for _c in first_feasible.constraints:
                    if _c.available and _c.value is not None:
                        _center_margins[_c.name] = -_c.value

            probe_rng = _random.Random(
                (config.random_seed or 0) + 31337
            )

            def _probe_run_fn(
                candidate_vars: dict[str, float],
            ) -> "tuple[bool, dict[str, float]]":
                """将探针候选点发给 Aspen，返回 (收敛成功, 约束 margin 字典)。

                margin = actual_value - threshold = -ConstraintValue.value（>0 表示满足）。
                不收敛时返回 (False, {})；收敛但无约束时返回 (True, {})。
                热启动模式下由 warmup_fn 预先建立收敛状态，本函数直接在该状态上运行。
                """
                import uuid as _uuid
                import dataclasses as _dc

                _no_reinit_cfg = _dc.replace(config.run_config, reinit=False)

                def _run_once_noreinit(dvars: dict[str, float]) -> "ProcessCase":
                    _dv_rep, _ = repair_design_vars(
                        {**config.fixed_vars, **dvars},
                        config.integer_var_paths,
                        config.param_bounds,
                        config.var_dependencies,
                    )
                    _dv_full, _ = apply_derived_vars(
                        _dv_rep, config.derived_var_specs
                    )
                    return run_case(
                        driver=driver,
                        design_vars=_dv_full,
                        config=_no_reinit_cfg,
                        iteration=-1,
                        tags=list(config.sensitivity_probe.tags),
                        run_id=str(_uuid.uuid4()),
                    )

                try:
                    _probe_case = _run_once_noreinit(candidate_vars)
                    _save_case(db, _probe_case, config.session_id)
                    converged = bool(_probe_case.simulation_valid)
                    # 提取约束 margin：margin = -ConstraintValue.value（value=threshold-actual）
                    margins: dict[str, float] = {}
                    if converged and _probe_case.constraints:
                        for _c in _probe_case.constraints:
                            if _c.available and _c.value is not None:
                                margins[_c.name] = -_c.value
                    return converged, margins
                except Exception as exc:
                    _log.warning("敏感度探针 run_fn 异常：%s", exc)
                    return False, {}

            def _probe_warmup_fn(warmup_vars: dict[str, float]) -> bool:
                """热启动函数：用 center 点（reinit=False）预热 Aspen 内部状态。

                在每次扰动前调用，让 Aspen 从 center 的已收敛状态出发，
                解决孤岛问题（.bkp 的热启动依赖）。
                热启动结果不保存到 DB，不计入统计。
                """
                import uuid as _uuid
                import dataclasses as _dc

                _no_reinit_cfg = _dc.replace(config.run_config, reinit=False)
                try:
                    _dv_rep, _ = repair_design_vars(
                        {**config.fixed_vars, **warmup_vars},
                        config.integer_var_paths,
                        config.param_bounds,
                        config.var_dependencies,
                    )
                    _dv_full, _ = apply_derived_vars(
                        _dv_rep, config.derived_var_specs
                    )
                    _case = run_case(
                        driver=driver,
                        design_vars=_dv_full,
                        config=_no_reinit_cfg,
                        iteration=-1,
                        tags=["probe_warmup"],
                        run_id=str(_uuid.uuid4()),
                    )
                    return bool(_case.success)
                except Exception as exc:
                    _log.warning("探针热启动异常：%s", exc)
                    return False

            probe_result = run_sensitivity_probe(
                center=center_dict,
                bounds=bounds,
                paths=paths,
                config=config.sensitivity_probe,
                run_fn=_probe_run_fn,
                integer_indices=integer_idx_set,
                rng=probe_rng,
                warmup_fn=_probe_warmup_fn,
                center_margins=_center_margins if _center_margins else None,
            )

            doe_bounds = adaptive_doe_bounds(
                center=center_dict,
                global_bounds=bounds,
                paths=paths,
                probe_result=probe_result,
            )

            # 初始化解冻调度器（Phase 2 使用）
            thaw_scheduler = ThawScheduler(
                probe_result=probe_result,
                config=config.sensitivity_probe,
                global_bounds=bounds,
                paths=paths,
                center=center_dict,
            )

            _log.info(
                "Phase 0.5 探针完成：%d 次仿真，约束 %d 个，自适应 DOE bounds 已就绪。"
                "综合敏感度 top4：%s",
                probe_result.n_probes_run,
                len(probe_result.constraint_names),
                [(p.split("\\")[-1], round(probe_result.sensitivity[p], 2))
                 for p in probe_result.sensitivity_rank[:4]],
            )
            if probe_result.constraint_names:
                _margin_rank = sorted(
                    probe_result.margin_sensitivity,
                    key=lambda p: probe_result.margin_sensitivity[p],
                    reverse=True,
                )
                _log.info(
                    "  margin 敏感度 top4：%s",
                    [(p.split("\\")[-1], round(probe_result.margin_sensitivity[p], 2))
                     for p in _margin_rank[:4]],
                )
        else:
            _log.warning(
                "Phase 0.5：Phase 0 未找到可行点，跳过敏感度探针，使用全局 bounds DOE。"
            )

    # ------------------------------------------------------------------
    # Phase 1：初始 DOE（自适应宽度拉丁超立方采样）
    # ------------------------------------------------------------------
    initial_points = _lhs_sample(doe_bounds, config.n_initial, config.random_seed)

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

        # 飞行前检查：被拦截则直接判 infeasible,不提交 Aspen(零成本)
        _pf_reason = _preflight_blocked(design_vars, config)
        if _pf_reason is not None:
            _log.info("  → 飞行前拦截(infeasible,未提交 Aspen)：%s", _pf_reason)
            case = ProcessCase(
                iteration=iteration, status=CaseStatus.INFEASIBLE,
                design_vars=design_vars, tags=tags,
                notes=f"飞行前检查拦截：{_pf_reason}",
            )
            cases.append(case)
            case_xs.append(x_eval)
            _save_case(db, case, config.session_id)
            _fire_callback(config.on_case_complete, case, idx, n_total)
            _fixed_ref_point, hv = _compute_hv_fixed(cases, config, _fixed_ref_point)
            hv_history.append(hv)
            continue

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
        # COM 自愈：本点若触发超时/COM 异常,重建连接后再跑下一点
        if not _maybe_recover_driver(driver, f"初始 DOE [{idx + 1}/{config.n_initial}]"):
            driver_dead = True
            break
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
            _tell_optimizer(
                optimizer, x, y_vec, is_success=y_vec is not None,
                c_vec=_extract_constraint_margins(c),
            )

        # warm_start_cases：注入 Phase 1 数据，不重新运行，不计入统计
        n_warm = 0
        for c in config.warm_start_cases:
            x = [c.design_vars.get(p) for p in paths]
            if None in x:
                continue
            y_vec = _extract_all_objectives(c, config)
            _tell_optimizer(
                optimizer, x, y_vec, is_success=y_vec is not None,
                c_vec=_extract_constraint_margins(c),
            )
            n_warm += 1
        if n_warm > 0:
            _log.info("warm_start：已注入 %d 个 Phase 1 样本到代理模型。", n_warm)

        # ------------------------------------------------------------------
        # Trust Region 初始化（若启用）
        # ------------------------------------------------------------------
        tr = None
        if config.trust_region is not None:
            from ..optimization.trust_region import TrustRegion
            first_feasible = next((c for c in cases if c.success), None)
            if first_feasible is not None:
                center = [float(first_feasible.design_vars.get(p, (bounds[i][0] + bounds[i][1]) / 2))
                          for i, p in enumerate(paths)]
                tr = TrustRegion(center, config.trust_region.initial_radius, bounds)
                _log.info(
                    "Trust Region 初始化：r=%.4f，中心=%s",
                    tr.radius,
                    {_short_var_name(paths[i]): round(center[i], 4) for i in range(min(4, len(paths)))},
                )
            else:
                _log.warning("Trust Region 启用但 Phase 0/1 无可行点，跳过 TR 初始化（使用全局 bounds）。")

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

        # 早停状态（P1-2：从 DOE/warm_start 初始化基线）
        es = config.early_stopping
        no_improvement_count = 0
        consecutive_dup_count = 0
        duplicate_skipped_iterations = 0
        no_unique_candidate_count = 0
        # 从已有数据中取当前最佳 HV
        best_hv_so_far: float | None = next(
            (hv for hv in reversed(hv_history) if hv is not None), None
        )
        # 从已有数据中计算当前第一前沿 fingerprint
        prev_front_fps: set[tuple] = _current_front_fps(cases, config)

        # 数据驱动边界收缩：周期性用可行样本重估,与 bounds 取交集(只收不放)。
        # refined_overlay[path] = (lo, hi),None 表示尚未收缩。
        refined_overlay: dict[str, tuple[float, float]] = {}
        _br_cfg = getattr(config, "boundary_refine", None)
        _br_interval = max(1, int(getattr(config, "boundary_refine_interval", 20)))

        for bo_idx in range(n_bo):
            if driver_dead or early_stopped:
                break

            idx = config.n_initial + bo_idx

            # 数据驱动边界收缩：每 _br_interval 轮用可行样本重估一次 overlay
            if _br_cfg is not None and getattr(_br_cfg, "enabled", False) \
                    and bo_idx > 0 and bo_idx % _br_interval == 0:
                refined_overlay = _compute_refined_overlay(cases, paths, config, _br_cfg)

            def _apply_overlay(bnds: list[tuple[float, float]]) -> list[tuple[float, float]]:
                """把 refined_overlay 与给定 bounds 取交集(只收不放)。"""
                if not refined_overlay:
                    return bnds
                out: list[tuple[float, float]] = []
                for i, (lo, hi) in enumerate(bnds):
                    ov = refined_overlay.get(paths[i])
                    if ov is None:
                        out.append((lo, hi)); continue
                    nlo, nhi = max(lo, ov[0]), min(hi, ov[1])
                    out.append((nlo, nhi) if nlo < nhi else (lo, hi))
                return out

            # Trust Region + ThawScheduler：在本次迭代开始前确定有效 bounds
            if tr is not None and thaw_scheduler is not None:
                # 两者都启用：取 TR 局部 bounds 与 Thaw bounds 的交集（更紧的约束）
                tr_local = tr.compute_local_bounds()
                thaw_local = thaw_scheduler.effective_bounds()
                combined = [
                    (max(tr_lo, th_lo), min(tr_hi, th_hi))
                    for (tr_lo, tr_hi), (th_lo, th_hi) in zip(tr_local, thaw_local)
                ]
                # 退化保护：若交集为空则退回 TR bounds
                combined = [
                    (lo, hi) if lo < hi else tr_b
                    for (lo, hi), tr_b in zip(combined, tr_local)
                ]
                optimizer.set_effective_bounds(_apply_overlay(combined))
            elif tr is not None:
                optimizer.set_effective_bounds(_apply_overlay(tr.compute_local_bounds()))
            elif thaw_scheduler is not None:
                optimizer.set_effective_bounds(_apply_overlay(thaw_scheduler.effective_bounds()))
            elif refined_overlay:
                optimizer.set_effective_bounds(_apply_overlay(list(bounds)))
            iteration = start_iteration + phase0_offset + idx
            tags = list(config.tags) + ["bayesian_opt", "pareto_opt"]

            # 确定本次迭代候选池使用的 bounds
            # 优先级：thaw_scheduler(动态) > probe_result(静态 doe_bounds) > 全局 bounds
            # 再与数据驱动 refined_overlay 取交集(只收不放)
            current_doe_bounds = _apply_overlay(
                thaw_scheduler.effective_bounds()
                if thaw_scheduler is not None
                else list(doe_bounds)
            )

            # ----------------------------------------------------------
            # 候选点选取（含去重保护 + 可选可行性过滤）
            # ----------------------------------------------------------
            fc = config.feasibility_filter
            use_filter = fc.enabled and fc.candidate_pool_size > 1

            # P1-1：已评估集合包含 warm_start_cases
            evaluated_fps = build_evaluated_set(cases + list(config.warm_start_cases))

            base_seed = config.random_seed
            iter_seed = (
                None if base_seed is None
                else base_seed + idx * 1009 + consecutive_dup_count
            )

            if use_filter:
                # ---- 带可行性分类器的候选池路径 ----
                clf = FeasibilityClassifier(fc)
                all_training = cases + list(config.warm_start_cases)
                rows = _build_feasibility_rows(all_training)
                feature_names = feasibility_feature_names(paths, config.derived_var_specs)
                trained = clf.fit(rows, feature_names)

                picked: dict[str, Any] | None = None
                retry = 0
                max_retries = es.max_duplicate_suggestions if es.enabled else 3
                while retry <= max_retries:
                    retry_seed = (
                        None if iter_seed is None
                        else iter_seed + retry * 7919
                    )
                    raw_candidates = _generate_candidate_points(
                        optimizer, current_doe_bounds, fc.candidate_pool_size, retry_seed,
                    )
                    screen_inputs: list[dict[str, Any]] = []
                    full_candidates: list[dict[str, Any]] = []
                    for j, x_raw in enumerate(raw_candidates):
                        dv_raw = {**config.fixed_vars, **dict(zip(paths, x_raw))}
                        dv_rep, rep_notes = repair_design_vars(
                            dv_raw,
                            config.integer_var_paths,
                            config.param_bounds,
                            config.var_dependencies,
                        )
                        dv_full, der_notes = apply_derived_vars(
                            dv_rep, config.derived_var_specs,
                        )
                        x_ev = [dv_rep.get(p, x_raw[i]) for i, p in enumerate(paths)]
                        screen_inputs.append({
                            **{name: dv_full.get(name) for name in feature_names},
                            "__candidate_index": j,
                        })
                        full_candidates.append({
                            "design_vars":   dv_full,
                            "x_eval":        x_ev,
                            "repair_notes":  rep_notes,
                            "derived_notes": der_notes,
                        })
                    screened = clf.screen(screen_inputs, fallback_top_k=1)
                    picked = pick_first_unseen_candidate(
                        full_candidates, screened, evaluated_fps
                    )
                    if picked is not None:
                        break
                    retry += 1

                if picked is None:
                    # P0-2：全部重复，跳过本轮，不调用 run_case
                    consecutive_dup_count += 1
                    duplicate_skipped_iterations += 1
                    no_unique_candidate_count += 1
                    _log.warning(
                        "贝叶斯优化 [%d/%d]：多目标候选池连续 %d 次全部重复，跳过本轮。",
                        idx + 1, n_total, consecutive_dup_count,
                    )
                    if es.enabled and consecutive_dup_count >= es.max_duplicate_suggestions:
                        early_stopped = True
                        early_stop_reason = "no_unique_candidate"
                        _log.warning(
                            "Early stopping triggered: reason=%s, 连续 %d 次未找到新候选，iteration=%d",
                            early_stop_reason, consecutive_dup_count, idx + 1,
                        )
                    continue
                else:
                    consecutive_dup_count = 0

                prob = None
                picked_fp = fingerprint_design_vars(picked["design_vars"])
                for entry in screened:
                    idx2 = int(entry.get("__candidate_index", 0))
                    if fingerprint_design_vars(
                        full_candidates[idx2]["design_vars"]
                    ) == picked_fp:
                        prob = entry.get("_predicted_feasible")
                        break

                design_vars   = picked["design_vars"]
                x_eval        = picked["x_eval"]
                repair_notes  = picked["repair_notes"]
                derived_notes = picked["derived_notes"]

                _log.info(
                    "贝叶斯优化 [%d/%d] 可行性过滤：训练=%s，候选池=%d，"
                    "筛选后=%d，重试次数=%d，选中概率=%s",
                    idx + 1, n_total, trained,
                    len(raw_candidates), len(screened), retry,
                    f"{prob:.3f}" if prob is not None else "N/A",
                )
            else:
                # ---- 无可行性分类器路径：optimizer.ask() + 去重保护 ----
                picked_no_filter: dict[str, Any] | None = None
                retry = 0
                max_retries = es.max_duplicate_suggestions if es.enabled else 3
                while retry <= max_retries:
                    retry_seed = (
                        None if iter_seed is None
                        else iter_seed + retry * 7919
                    )
                    raw_xs = _generate_candidate_points(optimizer, current_doe_bounds, max(2, max_retries), retry_seed)
                    for x_raw in raw_xs:
                        dv_raw = {**config.fixed_vars, **dict(zip(paths, x_raw))}
                        dv_rep, rep_notes = repair_design_vars(
                            dv_raw,
                            config.integer_var_paths,
                            config.param_bounds,
                            config.var_dependencies,
                        )
                        dv_full, der_notes = apply_derived_vars(
                            dv_rep, config.derived_var_specs,
                        )
                        x_ev = [dv_rep.get(p, x_raw[i]) for i, p in enumerate(paths)]
                        fp = fingerprint_design_vars(dv_full)
                        if fp not in evaluated_fps:
                            picked_no_filter = {
                                "design_vars":   dv_full,
                                "x_eval":        x_ev,
                                "repair_notes":  rep_notes,
                                "derived_notes": der_notes,
                            }
                            break
                    if picked_no_filter is not None:
                        break
                    retry += 1

                if picked_no_filter is None:
                    consecutive_dup_count += 1
                    duplicate_skipped_iterations += 1
                    no_unique_candidate_count += 1
                    _log.warning(
                        "贝叶斯优化 [%d/%d]：多目标无过滤器模式连续 %d 次候选全部重复，跳过本轮。",
                        idx + 1, n_total, consecutive_dup_count,
                    )
                    if es.enabled and consecutive_dup_count >= es.max_duplicate_suggestions:
                        early_stopped = True
                        early_stop_reason = "no_unique_candidate"
                        _log.warning(
                            "Early stopping triggered: reason=%s, iteration=%d",
                            early_stop_reason, idx + 1,
                        )
                    continue
                else:
                    consecutive_dup_count = 0

                design_vars   = picked_no_filter["design_vars"]
                x_eval        = picked_no_filter["x_eval"]
                repair_notes  = picked_no_filter["repair_notes"]
                derived_notes = picked_no_filter["derived_notes"]
            if repair_notes:
                _log.debug("Phase 2 repair [%d/%d]: %s", idx + 1, n_total, repair_notes)
            if derived_notes:
                _log.debug("Phase 2 derived [%d/%d]: %s", idx + 1, n_total, derived_notes)

            _log.info(
                "贝叶斯优化 [%d/%d]：%s",
                idx + 1, n_total,
                {_short_var_name(p): round(x_eval[i], 4) for i, p in enumerate(paths)},
            )

            # 飞行前检查：被拦截则构造 infeasible 工况,跳过 Aspen 调用,
            # 但仍走后续 tell/hv/thaw/早停逻辑(infeasible 点对代理模型有训练价值)
            _pf_reason = _preflight_blocked(design_vars, config)

            try:
                if _pf_reason is not None:
                    _log.info("  → 飞行前拦截(infeasible,未提交 Aspen)：%s", _pf_reason)
                    case = ProcessCase(
                        iteration=iteration, status=CaseStatus.INFEASIBLE,
                        design_vars=design_vars, tags=tags,
                        notes=f"飞行前检查拦截：{_pf_reason}",
                    )
                else:
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
            _tell_optimizer(
                optimizer, x_eval, y_vec, is_success=y_vec is not None,
                c_vec=_extract_constraint_margins(case),
            )
            _fixed_ref_point, hv = _compute_hv_fixed(cases, config, _fixed_ref_point)
            hv_history.append(hv)

            # Trust Region 更新
            if tr is not None:
                prev_hv = next((h for h in reversed(hv_history[:-1]) if h is not None), None)
                action = tr.update(prev_hv, hv, config.trust_region)
                _log.info("  TR r=%.4f (%s)", tr.radius, action)
                # 若本轮产生了成功的可行点，将信任域中心移向该点
                if case.success:
                    new_center = [
                        float(case.design_vars.get(p, tr.center[i]))
                        for i, p in enumerate(paths)
                    ]
                    tr.move_center(new_center)

            # ThawScheduler 更新（敏感度三阶段解冻）
            if thaw_scheduler is not None:
                prev_hv_ts = next((h for h in reversed(hv_history[:-1]) if h is not None), None)
                hv_improved_ts = (
                    hv is not None
                    and prev_hv_ts is not None
                    and (hv - prev_hv_ts) / max(abs(prev_hv_ts), 1e-10)
                    > config.sensitivity_probe.thaw_hv_stall_patience * 0  # 任意正改进
                )
                # 更新中心（与 TR 同步）
                new_thaw_center: dict[str, float] | None = None
                if case.success:
                    new_thaw_center = {
                        p: float(case.design_vars.get(p, thaw_scheduler._center.get(p, bounds[i][0])))
                        for i, p in enumerate(paths)
                    }
                new_stage = thaw_scheduler.step(
                    hv_improved=hv_improved_ts,
                    case_success=bool(case.success),
                    new_center=new_thaw_center,
                )
                _log.debug("  ThawScheduler: stage=%s", new_stage.value)

            _log.info("  → status=%s, success=%s, run_time=%.1fs",
                      case.status.value, case.success, case.run_time)
            if case.status == CaseStatus.OBJECTIVE_ERROR:
                for obj in (case.objectives or []):
                    if getattr(obj, "error", None):
                        _log.info("    [%s] error: %s", obj.name, obj.error)

            # COM 自愈：本点若触发超时/COM 异常,重建连接后再跑下一点
            if not _maybe_recover_driver(driver, f"贝叶斯优化 [{idx + 1}/{n_total}]"):
                driver_dead = True
                break

            # ---- 多目标早停判断 ----
            if es.enabled and idx + 1 >= es.min_iterations:
                hv_improved = _check_hv_improvement(hv, best_hv_so_far, es)
                front_changed = _check_front_changed(cases, config, prev_front_fps)
                if hv_improved:
                    best_hv_so_far = hv
                if hv_improved or front_changed:
                    no_improvement_count = 0
                    prev_front_fps = _current_front_fps(cases, config)
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= es.patience:
                        early_stopped = True
                        early_stop_reason = (
                            "hypervolume_stagnation" if es.check_hypervolume
                            else "pareto_stagnation"
                        )
                        _log.warning(
                            "Early stopping triggered: reason=%s, iteration=%d, "
                            "patience=%d, no_improvement=%d, best_hv=%s",
                            early_stop_reason, idx + 1, es.patience,
                            no_improvement_count, best_hv_so_far,
                        )
            elif es.enabled and hv is not None and best_hv_so_far is None:
                best_hv_so_far = hv

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
        elif early_stopped:
            _log.warning(
                "多目标优化早停：reason=%s，已完成 %d 个工况，%d 成功，耗时 %.1fs。"
                " （在当前搜索策略下继续改进概率较低，不代表全局最优。）",
                early_stop_reason, len(cases), n_success, elapsed,
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
            early_stopped=early_stopped,
            early_stop_reason=early_stop_reason,
            completed_iterations=len(cases),
            no_improvement_count=no_improvement_count,
            duplicate_skipped_iterations=duplicate_skipped_iterations,
            no_unique_candidate_count=no_unique_candidate_count,
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

    _VALID_SURROGATE = {"GP", "RF", "ET", "GBRT", "random", "qEHVI", "NEHVI"}
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


def _local_bounds(
    bounds: list[tuple[float, float]],
    initial_point_vals: list[float],
    radius: float,
) -> list[tuple[float, float]]:
    """
    以 initial_point_vals 为中心、radius 比例构造局部搜索边界，裁剪不超出全局 bounds。

    对每一维：
      local_lo = max(global_lo, init_val * (1 - radius))
      local_hi = min(global_hi, init_val * (1 + radius))
    特殊情况：init_val == 0 时用 (global_hi - global_lo) * radius 作为绝对偏移，
              避免局部范围退化为单点。
    """
    local: list[tuple[float, float]] = []
    for (glo, ghi), val in zip(bounds, initial_point_vals):
        if val == 0.0:
            half = (ghi - glo) * radius
            lo = max(glo, -half)
            hi = min(ghi, half)
        else:
            lo = max(glo, val * (1.0 - radius))
            hi = min(ghi, val * (1.0 + radius))
        # 保证 lo < hi（极端情况下 init_val 在边界时可能相等）
        if lo >= hi:
            lo, hi = glo, ghi
        local.append((lo, hi))
    return local


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

    # 从 initial_point 提取初始值向量（按 paths 顺序）
    initial_point_vals: list[float] | None = None
    if fs_cfg.initial_point:
        try:
            initial_point_vals = [float(fs_cfg.initial_point[p]) for p in paths]
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("Phase 0：初始点提取失败，退化为全局随机采样：%s", exc)
            initial_point_vals = None

    # 构建采样轮次列表：(描述, 采样点列表)
    # 策略：初始点 → 局部 LHS（按 radii 依序扩张）→ 全局 LHS 兜底
    sampling_rounds: list[tuple[str, list[list[float]]]] = []

    if initial_point_vals is not None:
        # 第一轮：初始收敛解本身（1 个点）
        sampling_rounds.append(("初始收敛解", [initial_point_vals]))
        remaining = n_trials - 1

        radii = [r for r in (fs_cfg.local_search_radii or []) if 0 < r <= 1.0]
        if radii and remaining > 0:
            # 按半径分配采样次数：每个半径平均分配，最后一个取余
            n_per_radius = max(1, remaining // len(radii))
            for i, radius in enumerate(radii):
                n_this = remaining if i == len(radii) - 1 else n_per_radius
                n_this = min(n_this, remaining)
                if n_this <= 0:
                    break
                lb = _local_bounds(bounds, initial_point_vals, radius)
                pts = _lhs_sample(lb, n_this, config.random_seed)
                sampling_rounds.append((f"局部 LHS（±{int(radius*100)}%）", pts))
                remaining -= n_this
        else:
            # 无 radii 配置：剩余次数全部全局随机
            if remaining > 0:
                sampling_rounds.append(("全局 LHS", _lhs_sample(bounds, remaining, config.random_seed)))
    else:
        # 无初始点：全局 LHS
        sampling_rounds.append(("全局 LHS", _lhs_sample(bounds, n_trials, config.random_seed)))

    cases: list[ProcessCase] = []
    case_xs: list[list[float]] = []
    n_feasible = 0
    driver_dead = False
    global_iter = 0  # 跨轮次的绝对迭代编号（用于 iteration 字段）

    for round_name, round_points in sampling_rounds:
        if driver_dead:
            break
        if stop_after > 0 and n_feasible >= stop_after:
            break
        # 只有在新的半径轮次开始时（不是第一轮初始点），才在可行数 > 0 时停止扩张
        if round_name.startswith("局部 LHS") and n_feasible > 0:
            _log.info(
                "Phase 0：已找到 %d 个可行点（在更小范围内），不继续扩张半径。",
                n_feasible,
            )
            break

        if round_points:
            _log.info("Phase 0：开始 %s，共 %d 个候选点。", round_name, len(round_points))

        for point in round_points:
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
                _log.debug("Phase 0 repair [%d]: %s", global_iter + 1, repair_notes)
            if derived_notes:
                _log.debug("Phase 0 derived [%d]: %s", global_iter + 1, derived_notes)

            iteration = start_iteration + global_iter
            _log.info(
                "Phase 0 [%d/%d]：%s",
                global_iter + 1, n_trials,
                {_short_var_name(k): round(v, 4) for k, v in design_vars_repaired.items()
                 if k in config.param_bounds},
            )

            # 飞行前检查：拦截则记为 infeasible,不提交 Aspen
            _pf_reason = _preflight_blocked(design_vars, config)
            if _pf_reason is not None:
                _log.info("  → 飞行前拦截(infeasible,未提交 Aspen)：%s", _pf_reason)
                case = ProcessCase(
                    iteration=iteration, status=CaseStatus.INFEASIBLE,
                    design_vars=design_vars, tags=tags,
                    notes=f"飞行前检查拦截：{_pf_reason}",
                )
                cases.append(case)
                case_xs.append(x_eval)
                _save_case(db, case, config.session_id)
                global_iter += 1
                continue

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
                _log.error("Phase 0 [%d/%d]：driver 连接断开。原因：%s", global_iter + 1, n_trials, exc)
                driver_dead = True
                case = ProcessCase(
                    iteration=iteration, status=CaseStatus.SIM_FAILED,
                    design_vars=design_vars, tags=tags,
                    notes=f"driver 连接断开：{exc}",
                )
                cases.append(case)
                case_xs.append(x_eval)
                _save_case(db, case, config.session_id)
                global_iter += 1
                break
            except Exception as exc:
                _log.warning("Phase 0 [%d/%d]：run_case() 意外异常：%s", global_iter + 1, n_trials, exc)
                case = ProcessCase(
                    iteration=iteration, status=CaseStatus.SIM_FAILED,
                    design_vars=design_vars, tags=tags,
                    notes=f"run_case() 意外异常：{exc}",
                )

            cases.append(case)
            case_xs.append(x_eval)
            _save_case(db, case, config.session_id)
            global_iter += 1

            is_phase0_feasible = (
                case.feasible is True
                or (not case.has_constraints and case.success)
            )
            if is_phase0_feasible:
                n_feasible += 1
                _log.info("  → 可行点 #%d（status=%s）", n_feasible, case.status.value)
            else:
                _log.info(
                    "  → 不可行（status=%s, feasible=%s）",
                    case.status.value, case.feasible,
                )

            # COM 自愈：本点若触发超时/COM 异常,重建连接后再跑下一点
            if not _maybe_recover_driver(driver, f"Phase 0 [{global_iter}/{n_trials}]"):
                driver_dead = True
                break

    _log.info(
        "Phase 0 完成：运行 %d 次，找到 %d 个可行点。",
        len(cases), n_feasible,
    )

    # 失败门槛：若启用 abort_if_none_found 且全部未找到可行点，直接终止
    if n_feasible == 0 and fs_cfg.abort_if_none_found and not driver_dead:
        raise RuntimeError(
            f"Phase 0 可行性搜索在 {len(cases)} 次随机采样中未找到任何可行点，优化终止。\n"
            "建议：\n"
            "  1. 检查约束设置是否过严（如纯度 >= 0.999 在宽松搜索范围下难以满足）\n"
            "  2. 缩小设计变量搜索范围，使其更贴近已知可行域\n"
            "  3. 确认 initial_value 已在 YAML 中正确填写（用于注入初始收敛解）\n"
            "  4. 增大 feasibility_search.n_trials 给随机探索更多机会\n"
            "  5. 若希望即使 Phase 0 全部失败也继续运行，"
            "设置 feasibility_search.abort_if_none_found: false"
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
        self._integer_indices: set[int] = {
            i for i, p in enumerate(paths) if p in config.integer_var_paths
        }
        # 成功观测：(x, y_vec_min_direction)
        self._observations: list[tuple[list[float], list[float]]] = []
        # 失败观测：只存 x
        self._failed_xs: list[list[float]] = []
        # Trust Region 覆盖的局部 bounds（None = 使用全局 bounds）
        self._effective_bounds: list[tuple[float, float]] | None = None

        # BoTorch 路径：预先创建持久化优化器（避免每次 ask 重建）。
        # 若 BoTorch 未安装，make_surrogate_optimizer 会返回 skopt GP fallback；
        # 此时不能继续按 BoTorch 接口传 y_vec，应退回本类下方的 ParEGO/skopt 路径。
        self._is_botorch = config.surrogate_model in ("qEHVI", "NEHVI")
        if self._is_botorch:
            surrogate_cfg = SurrogateConfig(
                model=config.surrogate_model,
                acquisition=config.acquisition,
                xi=config.xi,
                kappa=config.kappa,
                n_initial_min=config.n_initial_min,
                random_seed=config.random_seed,
            )
            botorch_candidate = make_surrogate_optimizer(
                bounds, surrogate_cfg, self._integer_indices,
                n_objectives=self._n_obj,
            )
            if botorch_candidate.__class__.__name__ == "BoTorchMOOptimizer":
                self._botorch_opt = botorch_candidate
            else:
                self._botorch_opt = None
                self._is_botorch = False
                self._surrogate_model = "GP"
                _log.warning(
                    "%s 不可用，_MultiObjectiveBayesianOptimizer 已切换到 "
                    "ParEGO/skopt GP fallback。",
                    config.surrogate_model,
                )
        else:
            self._botorch_opt = None

    def tell(
        self,
        x: list[float],
        y_vec: list[float] | None,
        *,
        is_success: bool,
        penalty: float = 1e10,
        c_vec: "dict[str, float] | None" = None,
    ) -> None:
        """
        提交一次观测。

        skopt 路径：成功样本存入 _observations，失败样本存入 _failed_xs。
        BoTorch 路径：同时转发给 _botorch_opt.tell()，传入完整 y_vec 和 c_vec。

        Parameters
        ----------
        c_vec:
            约束 margin 字典 {约束名: margin}。
            margin = actual_value - threshold（>= 0 表示满足约束）。
            None 时该点不参与约束 GP 训练（目标 GP 不受影响）。
        """
        if is_success and y_vec is not None:
            self._observations.append((list(x), list(y_vec)))
        else:
            self._failed_xs.append(list(x))
        # BoTorch 路径同步：透传 c_vec 给约束感知后端
        if self._botorch_opt is not None:
            self._botorch_opt.tell(
                x, 0.0, is_success=is_success, y_vec=y_vec, c_vec=c_vec
            )

    def set_effective_bounds(self, bounds: list[tuple[float, float]]) -> None:
        """设置本次 ask() 使用的局部 bounds（Trust Region 调用）。"""
        self._effective_bounds = bounds
        if self._botorch_opt is not None:
            self._botorch_opt.set_effective_bounds(bounds)

    def ask(self) -> list[float]:
        """推荐下一个候选点。成功观测不足 n_initial_min 时返回随机点。"""
        active_bounds = self._effective_bounds if self._effective_bounds else self._bounds
        # 每次 ask 后清除临时局部 bounds（下次迭代由 Phase 2 循环重新设置）
        self._effective_bounds = None
        if self._botorch_opt is not None:
            self._botorch_opt._effective_bounds = None

        if len(self._observations) < self._n_initial_min:
            return [lo + self._rng.random() * (hi - lo) for lo, hi in active_bounds]

        # BoTorch 路径：qEHVI/qNEHVI
        if self._botorch_opt is not None:
            self._botorch_opt.set_effective_bounds(active_bounds)
            return self._botorch_opt.ask()

        # skopt 路径：ParEGO 随机标量化
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
            opt = make_surrogate_optimizer(active_bounds, surrogate_cfg, self._integer_indices)
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


def _extract_constraint_margins(case: "ProcessCase") -> "dict[str, float] | None":
    """
    从 ProcessCase 提取约束 margin 字典。

    margin = actual_value - threshold = -ConstraintValue.value
    （>= 0 表示约束满足，< 0 表示违反）。

    仅在仿真收敛（simulation_valid=True）且所有约束均可用时返回字典；
    否则返回 None，调用方传 c_vec=None 给 optimizer.tell()，
    使该点不参与约束 GP 训练（目标 GP 不受影响）。
    """
    if not case.simulation_valid:
        return None
    if not case.constraints:
        return None
    margins: dict[str, float] = {}
    for c in case.constraints:
        if not c.available or c.value is None:
            return None  # 任意约束不可用则放弃整个点
        margins[c.name] = -c.value  # value = threshold - actual → margin = -value
    return margins if margins else None



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


def _tell_optimizer(optimizer: Any, x: Any, y_vec: Any, *, is_success: bool, c_vec: Any = None) -> None:
    """向 optimizer.tell 提交观测,兼容不接受 c_vec 的旧/桩 optimizer。

    优先带 c_vec 调用(约束感知后端用得到);若该 optimizer.tell 不接受
    c_vec 关键字,回退到不带 c_vec 的调用。两种签名都能工作。
    """
    try:
        optimizer.tell(x, y_vec, is_success=is_success, c_vec=c_vec)
    except TypeError:
        optimizer.tell(x, y_vec, is_success=is_success)


def _compute_refined_overlay(
    cases: list[Any],
    paths: list[str],
    config: Any,
    br_cfg: Any,
) -> dict[str, tuple[float, float]]:
    """用已有可行样本重估边界 overlay,返回 {path: (lo, hi)}。

    失败/样本不足时返回空字典(不影响优化)。与具体工艺无关。
    """
    try:
        from ..optimization.boundary_refine import extract_feasible_points, refine_bounds
        feasible = extract_feasible_points(cases, paths)
        current = {p: config.param_bounds[p] for p in paths if p in config.param_bounds}
        _new_bounds, results = refine_bounds(feasible, current, br_cfg)
        overlay = {r.path: r.new_bounds for r in results if r.shrunk}
        if overlay:
            _log.info(
                "boundary_refine：已基于 %d 个可行样本收紧 %d 个变量边界。",
                len(feasible), len(overlay),
            )
        return overlay
    except Exception as exc:  # noqa: BLE001
        _log.warning("boundary_refine 重估异常(已忽略)：%s", exc)
        return {}


def _preflight_blocked(design_vars: dict[str, Any], config: Any) -> str | None:
    """对候选点做飞行前检查, 被拦截返回原因字符串, 通过返回 None。

    config.preflight 为 None / 未启用时永远放行。预检本身异常也放行(不拦优化)。
    与具体工艺无关。
    """
    pf = getattr(config, "preflight", None)
    if pf is None:
        return None
    try:
        from ..optimization.preflight import check_preflight
        passed, reason = check_preflight(
            design_vars=design_vars,
            reference_values=getattr(config, "reference_values", {}) or {},
            global_bounds=config.param_bounds,
            config=pf,
            var_dependencies=config.var_dependencies,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("飞行前检查异常(已放行该点)：%s", exc)
        return None
    return None if passed else reason


def _maybe_recover_driver(driver: Any, where: str) -> bool:
    """若 driver 标记了 needs_recovery, 尝试 COM 自愈重建连接。

    在每次 run_case 之后调用。超时/COM 崩溃后 driver.needs_recovery 为 True,
    此时重建连接,避免后续工况连锁失败(DISP_E_EXCEPTION 秒崩)。

    Parameters
    ----------
    driver:
        AspenDriver 实例(可能是测试桩,无 needs_recovery 属性时直接返回 True)。
    where:
        调用位置标签,用于日志。

    Returns
    -------
    bool
        True  无需恢复,或恢复成功 → 可继续。
        False 恢复失败 → 调用方应置 driver_dead 并终止本轮。
    """
    needs = getattr(driver, "needs_recovery", False)
    if not needs:
        return True
    recover = getattr(driver, "recover", None)
    if not callable(recover):
        # 测试桩或不支持自愈的 driver:清不掉标志,保守视为不可继续
        _log.warning("%s：driver 标记需恢复但不支持 recover(),无法自愈。", where)
        return False
    _log.warning("%s：仿真超时/COM 异常,尝试重建 Aspen 连接……", where)
    ok = bool(recover())
    if ok:
        _log.info("%s：Aspen 连接已自愈,继续后续工况。", where)
    else:
        _log.error("%s：Aspen 连接自愈失败,终止本轮优化。", where)
    return ok


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


# ---------------------------------------------------------------------------
# 可行性分类器辅助函数（BO 候选池筛选）
# ---------------------------------------------------------------------------

def _build_feasibility_rows(cases: list[ProcessCase]) -> list[dict[str, Any]]:
    """
    从历史工况列表构造可行性分类器训练数据。

    只使用 valid_for_classifier=True 且 design_vars 非空的工况。
    label = case.feasible_label（等同于 case.success）。

    训练数据应包含当前 cases 和 warm_start_cases（调用方负责合并后传入）。
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not case.valid_for_classifier:
            continue
        if not case.design_vars:
            continue
        rows.append({
            "case_id":     case.case_id,
            "design_vars": case.design_vars,
            "label":       case.feasible_label,
            "status":      case.status.value,
        })
    return rows


def _generate_candidate_points(
    optimizer: Any,
    bounds: list[tuple[float, float]],
    n_candidates: int,
    seed: int | None,
) -> list[list[float]]:
    """
    生成候选点池，用于可行性过滤。

    第一个候选来自 optimizer.ask()（保留 BO 推荐），其余用 LHS 随机填充。
    若 n_candidates <= 1，则只返回 optimizer.ask() 的一个候选。
    """
    first = optimizer.ask()
    if n_candidates <= 1:
        return [first]
    rest = _lhs_sample(bounds, n_candidates - 1, seed)
    return [first] + rest


# ---------------------------------------------------------------------------
# 多目标早停辅助函数
# ---------------------------------------------------------------------------

def _check_hv_improvement(
    hv: float | None,
    best_hv: float | None,
    es: Any,  # EarlyStoppingConfig
) -> bool:
    """判断当前 HV 是否相对历史最优有有效改善。"""
    if not es.check_hypervolume:
        return False
    if hv is None:
        return False
    if best_hv is None:
        return True   # 第一个有效 HV 算改善

    delta = hv - best_hv
    if delta <= 0:
        return False
    if delta < es.min_delta:
        return False
    if es.relative_delta is not None:
        ref = abs(best_hv) if best_hv != 0 else 1.0
        if delta / ref < es.relative_delta:
            return False
    return True


def _current_front_fps(
    cases: list[Any],
    config: Any,
) -> set[tuple]:
    """
    提取当前第一 Pareto 前沿中所有成功工况的 design_vars fingerprint 集合。
    """
    from ..optimization.pareto import fast_non_dominated_sort
    from ..optimization.pareto import _extract_objectives as _ext_obj

    vecs_with_cases: list[tuple[list[float], Any]] = []
    for c in cases:
        if not c.success:
            continue
        v = _ext_obj(c, config.objective_names)
        if v is not None:
            vecs_with_cases.append((v, c))

    if not vecs_with_cases:
        return set()

    vecs = [vc[0] for vc in vecs_with_cases]
    try:
        front_indices = fast_non_dominated_sort(vecs)
        first = front_indices[0] if front_indices else []
    except Exception:
        return set()

    fps: set[tuple] = set()
    for i in first:
        c = vecs_with_cases[i][1]
        if c.design_vars:
            fps.add(fingerprint_design_vars(c.design_vars))
    return fps


def _check_front_changed(
    cases: list[Any],
    config: Any,
    prev_fps: set[tuple],
) -> bool:
    """判断 Pareto 第一前沿是否相对上一轮发生了变化。"""
    if not config.early_stopping.check_first_front:
        return False
    current_fps = _current_front_fps(cases, config)
    return current_fps != prev_fps
