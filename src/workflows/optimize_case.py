"""
optimize_case.py — 贝叶斯优化 workflow 层封装。

职责：
  1. 接受优化配置（设计变量边界、目标函数名称、约束）
  2. 生成初始 DOE 样本（拉丁超立方采样）
  3. 拟合代理模型（高斯过程回归）
  4. 通过采集函数推荐下一个候选点
  5. 迭代运行 run_case() 直到达到最大迭代次数
  6. 返回 OptimizeResult（含所有 ProcessCase 和最优解）

层级关系
---------
optimize_case()（本文件）
  └── run_case()（workflows/run_case.py）
        ├── SimulationRunner.run_case()     → SimulationResult
        ├── TreeExporter                    → block/stream 原始记录
        ├── _extract_blocks/streams()       → BlockResult / StreamResult
        └── _compute_objectives/constraints → ObjectiveValue / ConstraintValue

优化流程
---------
Phase 1 — 初始 DOE（拉丁超立方采样）：
    生成 n_initial 个均匀分布的初始样本，顺序运行。
    至少需要 n_initial_min 个成功样本才能启用高斯过程代理模型。

Phase 2 — 贝叶斯优化循环（共 n_iterations - n_initial 次）：
    1. 用成功样本拟合高斯过程代理模型
    2. 最大化采集函数（EI/UCB/PI）得到下一个候选点
    3. 运行 run_case() 评估候选点
    4. 更新代理模型，重复直到达到 n_iterations

代理模型后端
-----------
默认使用 scikit-optimize（skopt）的高斯过程回归。
若未安装 skopt，自动回退到随机采样并记录 WARNING。
若未安装 numpy，LHS 采样退化为均匀随机采样。

失败工况处理
-----------
仿真失败（SIM_FAILED）或目标函数不可用（OBJECTIVE_ERROR）的工况：
  - 不参与代理模型拟合
  - 以惩罚值（当前最差观测值 × 1.1）告知优化器，引导其远离失败区域
  - 仍记录在 OptimizeResult.cases 中，供失败归因分析

典型用法
---------
    from src.aspen_driver.driver import AspenDriver
    from src.workflows.run_case import RunCaseConfig
    from src.workflows.optimize_case import OptimizeCaseConfig, optimize_case

    run_cfg = RunCaseConfig(
        objective_fns=[tac_objective],
        constraint_fns=[purity_constraint],
    )
    opt_cfg = OptimizeCaseConfig(
        param_bounds={
            r"\\Data\\Blocks\\T0301\\Input\\BASIS_RR": (1.0, 5.0),
            r"\\Data\\Blocks\\T0301\\Input\\B:F": (0.3, 0.8),
        },
        run_config=run_cfg,
        n_initial=10,
        n_iterations=30,
        objective_name="TAC",
        minimize=True,
    )
    with AspenDriver() as driver:
        driver.open("二级氢氰化工段.bkp")
        result = optimize_case(driver, opt_cfg)

    print(f"最优 TAC：{result.best_value:.4g}")
    print(f"最优参数：{result.best_case.design_vars}")
"""
from __future__ import annotations

import logging
import random as _random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ..aspen_driver.driver import AspenDriver
from ..aspen_driver.errors import AspenConnectionError
from ..models.process_case import CaseStatus, ProcessCase
from ..optimization.feasibility import FeasibilityClassifier, FeasibilityConfig
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
# 优化配置
# ---------------------------------------------------------------------------

@dataclass
class OptimizeCaseConfig:
    """
    optimize_case() 的配置参数。

    Attributes
    ----------
    param_bounds:
        设计变量的搜索边界 {Aspen 树路径: (下界, 上界)}。
        所有变量均为连续实数，下界必须严格小于上界。
    fixed_vars:
        固定不变的设计变量 {Aspen 树路径: 值}，每次运行均使用相同值。
        若与 param_bounds 存在相同路径，param_bounds 优先。
    run_config:
        每次单次运行的配置，见 RunCaseConfig。
    n_initial:
        初始 DOE 样本数（拉丁超立方采样），默认 10。
        建议设为设计变量维度的 5~10 倍。
    n_iterations:
        总迭代次数（含初始 DOE），默认 30。必须 >= n_initial。
    objective_name:
        优化目标函数名称，须与 run_config.objective_fns 中某个函数的
        ObjectiveValue.name 一致。
    minimize:
        True（默认）：最小化目标函数；False：最大化。
    acquisition:
        采集函数类型："EI"（默认）、"UCB"、"PI"。
    xi:
        EI/PI 采集函数的探索参数，默认 0.01。
    kappa:
        UCB 采集函数的探索参数，默认 1.96。
    n_initial_min:
        启用高斯过程代理模型所需的最少成功样本数，默认 3。
        不足时贝叶斯优化循环退化为随机采样。
    tags:
        应用到所有工况的标签列表。
        初始 DOE 工况自动添加 "initial_doe"；贝叶斯优化工况自动添加 "bayesian_opt"。
    on_case_complete:
        每次工况完成后的回调函数，签名为 (case, index, total) -> None。
        index 从 0 开始，total 为 n_iterations。
    db_path:
        SQLite 数据库路径，若指定则每次工况完成后自动持久化。None 不持久化。
    random_seed:
        随机种子，用于 LHS 采样和代理模型的可重复性。
    """
    param_bounds: dict[str, tuple[float, float]]
    fixed_vars: dict[str, Any] = field(default_factory=dict)
    run_config: RunCaseConfig = field(default_factory=RunCaseConfig)
    n_initial: int = 10
    n_iterations: int = 30
    objective_name: str = ""
    minimize: bool = True
    acquisition: Literal["EI", "UCB", "PI"] = "EI"
    xi: float = 0.01
    kappa: float = 1.96
    n_initial_min: int = 3
    surrogate_model: Literal["GP", "RF", "ET", "GBRT", "random"] = "GP"
    tags: list[str] = field(default_factory=list)
    on_case_complete: Callable[[ProcessCase, int, int], None] | None = None
    db_path: Path | str | None = None
    random_seed: int | None = None
    # type=integer 的设计变量路径集合；BO 提出连续值后 round/clamp 到整数
    integer_var_paths: set[str] = field(default_factory=set)
    derived_var_specs: list[dict[str, Any]] = field(default_factory=list)
    # 可行性分类器配置；enabled=False（默认）时完全不影响原有流程
    feasibility_filter: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    # 早停配置；enabled=False（默认）时完全不影响原有流程
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


# ---------------------------------------------------------------------------
# 优化结果
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    """
    optimize_case() 的返回值，包含所有工况结果和最优解。

    Attributes
    ----------
    cases:
        所有工况的 ProcessCase 列表，顺序与迭代顺序一致。
        前 n_initial 个为初始 DOE 工况，后续为贝叶斯优化工况。
    best_case:
        目标函数值最优的成功工况；无成功工况时为 None。
    param_bounds:
        本次优化的设计变量边界（来自 OptimizeCaseConfig.param_bounds）。
    fixed_vars:
        本次优化的固定变量（来自 OptimizeCaseConfig.fixed_vars）。
    objective_name:
        优化目标函数名称。
    minimize:
        True 表示最小化，False 表示最大化。
    n_total:
        实际运行的总工况数（driver 断开时可能少于 n_iterations）。
    n_success:
        仿真收敛且目标函数可用的工况数。
    n_sim_failed:
        仿真失败的工况数。
    n_objective_error:
        仿真收敛但目标函数计算失败的工况数。
    n_initial:
        初始 DOE 工况数（来自 OptimizeCaseConfig.n_initial）。
    elapsed:
        总耗时（秒）。
    """
    cases: list[ProcessCase]
    best_case: ProcessCase | None
    param_bounds: dict[str, tuple[float, float]]
    fixed_vars: dict[str, Any]
    objective_name: str
    minimize: bool
    n_total: int
    n_success: int
    n_sim_failed: int
    n_objective_error: int
    n_initial: int
    elapsed: float
    early_stopped: bool = False
    early_stop_reason: str | None = None
    completed_iterations: int = 0
    duplicate_skipped_iterations: int = 0
    no_unique_candidate_count: int = 0

    @property
    def best_value(self) -> float | None:
        """最优目标函数值；无成功工况时为 None。"""
        if self.best_case is None:
            return None
        obj = self.best_case.get_objective(self.objective_name)
        return float(obj.value) if obj and obj.available else None

    @property
    def success_rate(self) -> float:
        """成功率（0.0 ~ 1.0）。n_total=0 时返回 0.0。"""
        return self.n_success / self.n_total if self.n_total > 0 else 0.0

    @property
    def convergence_history(self) -> list[float | None]:
        """
        每次迭代后的最优目标值历史列表，长度等于 n_total。

        None 表示截至该迭代尚无成功样本。可用于绘制收敛曲线。
        """
        best_so_far: float | None = None
        history: list[float | None] = []
        for c in self.cases:
            if c.success:
                obj = c.get_objective(self.objective_name)
                if obj and obj.available:
                    y = float(obj.value)
                    if best_so_far is None:
                        best_so_far = y
                    elif self.minimize and y < best_so_far:
                        best_so_far = y
                    elif not self.minimize and y > best_so_far:
                        best_so_far = y
            history.append(best_so_far)
        return history

    def successful_cases(self) -> list[ProcessCase]:
        """返回所有当前优化目标（objective_name）可提取且 case.success=True 的工况。"""
        result = []
        for c in self.cases:
            if not c.success:
                continue
            obj = c.get_objective(self.objective_name)
            if obj is not None and obj.available:
                result.append(c)
        return result

    def to_summary(self) -> dict[str, Any]:
        """返回汇总字典，供日志和数据库记录。"""
        return {
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_sim_failed": self.n_sim_failed,
            "n_objective_error": self.n_objective_error,
            "n_initial": self.n_initial,
            "success_rate": self.success_rate,
            "best_value": self.best_value,
            "objective_name": self.objective_name,
            "minimize": self.minimize,
            "elapsed": self.elapsed,
            "param_bounds": {k: list(v) for k, v in self.param_bounds.items()},
            "early_stopped": self.early_stopped,
            "early_stop_reason": self.early_stop_reason,
            "completed_iterations": self.completed_iterations,
            "duplicate_skipped_iterations": self.duplicate_skipped_iterations,
            "no_unique_candidate_count": self.no_unique_candidate_count,
        }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def optimize_case(
    driver: AspenDriver,
    config: OptimizeCaseConfig,
    start_iteration: int = 0,
) -> OptimizeResult:
    """
    执行贝叶斯优化循环，返回 OptimizeResult。

    单个工况的意外异常会被隔离为 SIM_FAILED，优化继续执行后续工况。
    若 driver 连接已断开（AspenConnectionError），则终止优化并返回已完成的结果。

    Parameters
    ----------
    driver:
        已连接并打开仿真文件的 AspenDriver 实例。
    config:
        优化配置，见 OptimizeCaseConfig。
    start_iteration:
        起始迭代编号，默认 0。第 i 个工况的 iteration = start_iteration + i。

    Returns
    -------
    OptimizeResult
    """
    _validate_config(config)

    paths = list(config.param_bounds.keys())
    bounds = [config.param_bounds[p] for p in paths]
    # integer 变量在 bounds 中的下标集合，传给代理模型构建混合整数搜索空间
    integer_indices: set[int] = {
        i for i, p in enumerate(paths) if p in config.integer_var_paths
    }
    n_total = config.n_iterations

    _log.info(
        "贝叶斯优化开始：%d 个设计变量，%d 次初始 DOE，%d 次总迭代，目标=%s（%s）。",
        len(paths), config.n_initial, n_total,
        config.objective_name, "最小化" if config.minimize else "最大化",
    )

    db = None
    if config.db_path is not None:
        from ..database.simulation_db import SimulationDB
        db = SimulationDB(config.db_path)

    cases: list[ProcessCase] = []
    case_xs: list[list[float]] = []
    t0 = time.monotonic()
    driver_dead = False
    early_stopped = False
    early_stop_reason: str | None = None
    duplicate_skipped_iterations = 0
    no_unique_candidate_count = 0

    # ------------------------------------------------------------------
    # Phase 1：初始 DOE（拉丁超立方采样）
    # ------------------------------------------------------------------
    initial_points = _lhs_sample(bounds, config.n_initial, config.random_seed)

    for idx, point in enumerate(initial_points):
        if driver_dead:
            break

        design_vars_raw = {**config.fixed_vars, **dict(zip(paths, point))}
        design_vars_repaired, repair_notes = repair_design_vars(
            design_vars_raw, config.integer_var_paths, config.param_bounds, {},
        )
        design_vars, derived_notes = apply_derived_vars(
            design_vars_repaired, config.derived_var_specs,
        )
        x_eval = [design_vars_repaired.get(p, point[i]) for i, p in enumerate(paths)]
        if repair_notes:
            _log.debug("初始 DOE [%d/%d] repair: %s", idx + 1, config.n_initial, repair_notes)
        if derived_notes:
            _log.debug("initial DOE [%d/%d] derived: %s", idx + 1, config.n_initial, derived_notes)
        iteration = start_iteration + idx
        tags = list(config.tags) + ["initial_doe", "optimize"]

        _log.info(
            "初始 DOE [%d/%d]：%s",
            idx + 1, config.n_initial,
            {k.split("\\")[-1]: round(v, 4) for k, v in dict(zip(paths, point)).items()},
        )

        try:
            case = run_case(
                driver=driver,
                design_vars=design_vars,
                config=config.run_config,
                iteration=iteration,
                tags=tags,
            )
        except AspenConnectionError as exc:
            _log.error(
                "初始 DOE [%d/%d]：driver 连接断开，终止优化。原因：%s",
                idx + 1, config.n_initial, exc,
            )
            driver_dead = True
            case = ProcessCase(
                iteration=iteration, status=CaseStatus.SIM_FAILED,
                design_vars=design_vars, tags=tags,
                notes=f"driver 连接断开，优化终止：{exc}",
            )
            cases.append(case)
            case_xs.append(x_eval)
            _save_case(db, case)
            _fire_callback(config.on_case_complete, case, idx, n_total)
            _log.info("  → status=%s, success=%s, run_time=%.1fs",
                      case.status.value, case.success, case.run_time)
            break
        except Exception as exc:
            _log.warning(
                "初始 DOE [%d/%d]：run_case() 意外异常（已隔离）：%s",
                idx + 1, config.n_initial, exc,
            )
            case = ProcessCase(
                iteration=iteration, status=CaseStatus.SIM_FAILED,
                design_vars=design_vars, tags=tags,
                notes=f"run_case() 意外异常：{exc}",
            )

        cases.append(case)
        case_xs.append(x_eval)
        _save_case(db, case)
        _fire_callback(config.on_case_complete, case, idx, n_total)
        _log.info(
            "  → status=%s, success=%s, run_time=%.1fs",
            case.status.value, case.success, case.run_time,
        )

    # ------------------------------------------------------------------
    # Phase 2：贝叶斯优化循环
    # ------------------------------------------------------------------
    n_bo = n_total - config.n_initial

    if not driver_dead and n_bo > 0:
        optimizer = make_surrogate_optimizer(
            bounds,
            SurrogateConfig(
                model=config.surrogate_model,
                acquisition=config.acquisition,
                xi=config.xi,
                kappa=config.kappa,
                n_initial_min=config.n_initial_min,
                random_seed=config.random_seed,
            ),
            integer_indices,
        )

        # 用初始 DOE 的观测初始化优化器（成功样本用真实 y，失败样本用惩罚值）
        for c, x in zip(cases, case_xs):
            y = _extract_y(c, config)
            optimizer.tell(
                x,
                y if y is not None else _penalty_value(cases, config),
                is_success=y is not None,
            )

        n_success_so_far = sum(1 for c in cases if _extract_y(c, config) is not None)
        if n_success_so_far < config.n_initial_min:
            _log.warning(
                "初始 DOE 成功样本数 %d < n_initial_min=%d，"
                "贝叶斯优化循环将以随机采样替代高斯过程。",
                n_success_so_far, config.n_initial_min,
            )

        # 早停状态
        es = config.early_stopping
        no_improvement_count = 0
        consecutive_dup_count = 0
        duplicate_skipped_iterations = 0
        no_unique_candidate_count = 0
        # P1-2：从 DOE 结果初始化早停基线，避免 BO 第一轮差值被误判为改善
        best_y_so_far: float | None = min(
            (y for c in cases
             for y in [_extract_y(c, config)] if y is not None),
            default=None,
        )

        for bo_idx in range(n_bo):
            if driver_dead or early_stopped:
                break

            idx = config.n_initial + bo_idx
            iteration = start_iteration + idx
            tags = list(config.tags) + ["bayesian_opt", "optimize"]

            # ----------------------------------------------------------
            # 候选点选取（含去重保护 + 可选可行性过滤）
            # 去重是 BO workflow 的通用保护，不依赖 feasibility_filter。
            # ----------------------------------------------------------
            fc = config.feasibility_filter
            use_filter = fc.enabled and fc.candidate_pool_size > 1

            # 已评估 fingerprint 集合（每轮重建，保证最新）
            evaluated_fps = build_evaluated_set(cases)

            # 迭代相关 seed：避免每轮候选池固定
            base_seed = config.random_seed
            iter_seed = (
                None if base_seed is None
                else base_seed + idx * 1009 + consecutive_dup_count
            )

            if use_filter:
                # ---- 带可行性分类器的候选池路径 ----
                clf = FeasibilityClassifier(fc)
                rows = _build_feasibility_rows(cases)
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
                        optimizer, bounds, fc.candidate_pool_size, retry_seed,
                    )
                    screen_inputs: list[dict[str, Any]] = []
                    full_candidates: list[dict[str, Any]] = []
                    for j, x_raw in enumerate(raw_candidates):
                        dv_raw = {**config.fixed_vars, **dict(zip(paths, x_raw))}
                        dv_rep, rep_notes = repair_design_vars(
                            dv_raw, config.integer_var_paths, config.param_bounds, {},
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
                    # P0-2：候选池全部重复，绝不 fallback 运行重复点
                    consecutive_dup_count += 1
                    duplicate_skipped_iterations += 1
                    no_unique_candidate_count += 1
                    _log.warning(
                        "贝叶斯优化 [%d/%d]：候选池连续 %d 次全部重复，跳过本轮。",
                        idx + 1, n_total, consecutive_dup_count,
                    )
                    if es.enabled and consecutive_dup_count >= es.max_duplicate_suggestions:
                        early_stopped = True
                        early_stop_reason = "no_unique_candidate"
                        _log.warning(
                            "Early stopping triggered: reason=%s, 连续 %d 次未找到新候选，"
                            "iteration=%d",
                            early_stop_reason, consecutive_dup_count, idx + 1,
                        )
                    continue  # 直接跳过本轮，不调用 run_case
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
                    # 生成一批候选（optimizer.ask() 开头 + LHS 补充）
                    raw_xs = _generate_candidate_points(optimizer, bounds, max(2, max_retries), retry_seed)
                    for x_raw in raw_xs:
                        dv_raw = {**config.fixed_vars, **dict(zip(paths, x_raw))}
                        dv_rep, rep_notes = repair_design_vars(
                            dv_raw, config.integer_var_paths, config.param_bounds, {},
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
                    # P0-2：全部重复，跳过本轮
                    consecutive_dup_count += 1
                    duplicate_skipped_iterations += 1
                    no_unique_candidate_count += 1
                    _log.warning(
                        "贝叶斯优化 [%d/%d]：无过滤器模式候选连续 %d 次全部重复，跳过本轮。",
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
                _log.debug("贝叶斯优化 [%d/%d] repair: %s", idx + 1, n_total, repair_notes)

            if derived_notes:
                _log.debug("bayesian opt [%d/%d] derived: %s", idx + 1, n_total, derived_notes)

            _log.info(
                "贝叶斯优化 [%d/%d]：%s",
                idx + 1, n_total,
                {p.split("\\")[-1]: round(x_eval[i], 4) for i, p in enumerate(paths)},
            )

            try:
                case = run_case(
                    driver=driver,
                    design_vars=design_vars,
                    config=config.run_config,
                    iteration=iteration,
                    tags=tags,
                )
            except AspenConnectionError as exc:
                _log.error(
                    "贝叶斯优化 [%d/%d]：driver 连接断开，终止优化。原因：%s",
                    idx + 1, n_total, exc,
                )
                driver_dead = True
                case = ProcessCase(
                    iteration=iteration, status=CaseStatus.SIM_FAILED,
                    design_vars=design_vars, tags=tags,
                    notes=f"driver 连接断开，优化终止：{exc}",
                )
                cases.append(case)
                case_xs.append(x_eval)
                _save_case(db, case)
                _fire_callback(config.on_case_complete, case, idx, n_total)
                _log.info("  → status=%s, success=%s, run_time=%.1fs",
                          case.status.value, case.success, case.run_time)
                break
            except Exception as exc:
                _log.warning(
                    "贝叶斯优化 [%d/%d]：run_case() 意外异常（已隔离）：%s",
                    idx + 1, n_total, exc,
                )
                case = ProcessCase(
                    iteration=iteration, status=CaseStatus.SIM_FAILED,
                    design_vars=design_vars, tags=tags,
                    notes=f"run_case() 意外异常：{exc}",
                )

            cases.append(case)
            case_xs.append(x_eval)
            _save_case(db, case)
            _fire_callback(config.on_case_complete, case, idx, n_total)

            y = _extract_y(case, config)
            # 用修复后的实际运行点 tell，保证代理模型学习正确的输入-输出关系
            optimizer.tell(
                x_eval,
                y if y is not None else _penalty_value(cases, config),
                is_success=y is not None,
            )

            _log.info(
                "  → status=%s, success=%s, run_time=%.1fs",
                case.status.value, case.success, case.run_time,
            )

            # ---- 早停判断 ----
            if es.enabled and y is not None and idx + 1 >= es.min_iterations:
                improved = _check_improvement(y, best_y_so_far, config.minimize, es)
                if improved or best_y_so_far is None:
                    best_y_so_far = y
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= es.patience:
                        early_stopped = True
                        early_stop_reason = "objective_stagnation"
                        _log.warning(
                            "Early stopping triggered: reason=%s, "
                            "iteration=%d, patience=%d, no_improvement=%d, best_y=%s",
                            early_stop_reason, idx + 1, es.patience,
                            no_improvement_count, best_y_so_far,
                        )
            elif es.enabled and y is not None and best_y_so_far is None:
                best_y_so_far = y

    try:
        elapsed = time.monotonic() - t0
        best = _find_best_case(cases, config)

        n_success         = sum(1 for c in cases if _extract_y(c, config) is not None)
        n_sim_failed      = sum(1 for c in cases if c.status == CaseStatus.SIM_FAILED)
        n_objective_error = sum(1 for c in cases if c.status == CaseStatus.OBJECTIVE_ERROR)

        if driver_dead:
            _log.warning(
                "贝叶斯优化因 driver 断开提前终止：已完成 %d/%d 个工况，%d 成功，耗时 %.1fs。",
                len(cases), n_total, n_success, elapsed,
            )
        elif early_stopped:
            _log.warning(
                "贝叶斯优化早停：reason=%s，已完成 %d 个工况，%d 成功，耗时 %.1fs。"
                " （注：在当前搜索策略下继续改进概率较低，不代表全局最优。）",
                early_stop_reason, len(cases), n_success, elapsed,
            )
        else:
            best_str = (
                f"{best.get_objective(config.objective_name).value:.4g}"
                if best else "N/A"
            )
            _log.info(
                "贝叶斯优化完成：%d/%d 成功，最优 %s=%s，总耗时 %.1fs。",
                n_success, len(cases), config.objective_name, best_str, elapsed,
            )

        return OptimizeResult(
            cases=cases,
            best_case=best,
            param_bounds=config.param_bounds,
            fixed_vars=config.fixed_vars,
            objective_name=config.objective_name,
            minimize=config.minimize,
            n_total=len(cases),
            n_success=n_success,
            n_sim_failed=n_sim_failed,
            n_objective_error=n_objective_error,
            n_initial=config.n_initial,
            elapsed=elapsed,
            early_stopped=early_stopped,
            early_stop_reason=early_stop_reason,
            completed_iterations=len(cases),
            duplicate_skipped_iterations=duplicate_skipped_iterations,
            no_unique_candidate_count=no_unique_candidate_count,
        )
    finally:
        if db is not None:
            db.close()


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

def _validate_config(config: OptimizeCaseConfig) -> None:
    if not config.param_bounds:
        raise ValueError("param_bounds 不能为空，至少需要一个设计变量。")

    for path, (lo, hi) in config.param_bounds.items():
        if lo >= hi:
            raise ValueError(
                f"param_bounds['{path}'] 的下界 {lo} >= 上界 {hi}，"
                "请确保下界严格小于上界。"
            )

    if config.n_initial < 1:
        raise ValueError(f"n_initial 必须 >= 1，收到：{config.n_initial}。")

    if config.n_iterations < config.n_initial:
        raise ValueError(
            f"n_iterations={config.n_iterations} 必须 >= n_initial={config.n_initial}。"
        )

    if not config.objective_name:
        raise ValueError("objective_name 不能为空，请指定要优化的目标函数名称。")

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
            "请从 fixed_vars 中移除冲突路径。"
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
# 拉丁超立方采样
# ---------------------------------------------------------------------------

def _lhs_sample(
    bounds: list[tuple[float, float]],
    n: int,
    seed: int | None,
) -> list[list[float]]:
    """
    拉丁超立方采样，返回 n 个样本点。

    每个维度分成 n 个等间隔区间，每个区间内随机取一个点，
    并对各维度独立随机排列，保证样本在参数空间中均匀分布。

    若 numpy 不可用，退化为均匀随机采样（仍可用，但空间覆盖性较差）。
    """
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
    result: list[list[float]] = []
    # 每个维度独立生成分层样本，再随机排列
    cols: list[list[float]] = []
    for lo, hi in bounds:
        perm = list(range(n))
        rng.shuffle(perm)
        col = [lo + (perm[i] + rng.random()) / n * (hi - lo) for i in range(n)]
        cols.append(col)
    for i in range(n):
        result.append([cols[j][i] for j in range(d)])
    return result


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _extract_y(case: ProcessCase, config: OptimizeCaseConfig) -> float | None:
    """
    从 ProcessCase 提取目标函数值，供优化器使用。

    skopt 总是最小化，因此最大化目标取负值。
    工况失败或目标不可用时返回 None。
    """
    if not case.success:
        return None
    obj = case.get_objective(config.objective_name)
    if obj is None or not obj.available:
        return None
    y = float(obj.value)  # type: ignore[arg-type]
    return y if config.minimize else -y


def _penalty_value(cases: list[ProcessCase], config: OptimizeCaseConfig) -> float:
    """
    为失败工况生成惩罚值（当前最差观测值 × 1.1）。

    惩罚值告知优化器该区域不可行，引导其探索其他区域。
    无任何成功观测时返回 1e10。
    """
    ys: list[float] = []
    for c in cases:
        if c.success:
            obj = c.get_objective(config.objective_name)
            if obj and obj.available:
                y = float(obj.value)  # type: ignore[arg-type]
                ys.append(y if config.minimize else -y)
    if not ys:
        return 1e10
    worst = max(ys)
    return worst * 1.1 if worst > 0 else worst * 0.9


def _find_best_case(
    cases: list[ProcessCase],
    config: OptimizeCaseConfig,
) -> ProcessCase | None:
    """从所有工况中找到目标函数值最优的成功工况。"""
    best: ProcessCase | None = None
    best_y: float | None = None
    for c in cases:
        if not c.success:
            continue
        obj = c.get_objective(config.objective_name)
        if obj is None or not obj.available:
            continue
        y = float(obj.value)  # type: ignore[arg-type]
        if best_y is None:
            best, best_y = c, y
        elif config.minimize and y < best_y:
            best, best_y = c, y
        elif not config.minimize and y > best_y:
            best, best_y = c, y
    return best


def _save_case(db: Any, case: ProcessCase) -> None:
    """将工况保存到数据库（db 为 None 时跳过）。"""
    if db is None:
        return
    try:
        db.save_case(case.to_dict())
    except Exception as exc:
        _log.warning("工况 '%s' 保存到数据库失败（已忽略）：%s", case.case_id, exc)


def _fire_callback(
    callback: Callable[[ProcessCase, int, int], None] | None,
    case: ProcessCase,
    idx: int,
    total: int,
) -> None:
    """触发 on_case_complete 回调，异常已隔离。"""
    if callback is None:
        return
    try:
        callback(case, idx, total)
    except Exception as exc:
        _log.warning("on_case_complete 回调异常（已忽略）：%s", exc)


# ---------------------------------------------------------------------------
# 可行性分类器辅助函数
# ---------------------------------------------------------------------------

def _build_feasibility_rows(cases: list[ProcessCase]) -> list[dict[str, Any]]:
    """
    从历史工况中构造可行性分类器训练数据。

    只使用 valid_for_classifier=True 且 design_vars 非空的工况。
    label = case.feasible_label（等同于 case.success）。
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

    Parameters
    ----------
    optimizer:
        SurrogateOptimizer 实例。
    bounds:
        设计变量边界列表。
    n_candidates:
        候选点总数。
    seed:
        LHS 随机种子。

    Returns
    -------
    list[list[float]]
        至少包含 1 个候选点。
    """
    first = optimizer.ask()
    if n_candidates <= 1:
        return [first]

    rest = _lhs_sample(bounds, n_candidates - 1, seed)
    return [first] + rest


def _check_improvement(
    current_y: float,
    best_y: float | None,
    minimize: bool,
    es: Any,  # EarlyStoppingConfig
) -> bool:
    """
    判断 current_y 相对 best_y 是否构成有效改善。

    skopt 内部统一最小化，所以传入的 y 已是最小化方向值。
    minimize=True 时：current_y < best_y - delta 才算改善；
    minimize=False 时：skopt 存的是 -y，所以 current_y < best_y - delta 仍然适用。
    """
    if best_y is None:
        return True  # 第一个有效观测默认算改善

    delta = current_y - best_y           # 负数 = 下降 = 改善（最小化）
    if delta >= 0:
        return False                     # 没有下降

    abs_improvement = -delta
    if abs_improvement < es.min_delta:
        return False

    if es.relative_delta is not None:
        ref = abs(best_y) if best_y != 0 else 1.0
        if abs_improvement / ref < es.relative_delta:
            return False

    return True
