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
from .run_case import RunCaseConfig, run_case

_log = logging.getLogger(__name__)

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from skopt import Optimizer as _SkoptOptimizer
    from skopt.space import Real as _Real
    _HAS_SKOPT = True
except ImportError:
    _HAS_SKOPT = False


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
    tags: list[str] = field(default_factory=list)
    on_case_complete: Callable[[ProcessCase, int, int], None] | None = None
    db_path: Path | str | None = None
    random_seed: int | None = None


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
    t0 = time.monotonic()
    driver_dead = False

    # ------------------------------------------------------------------
    # Phase 1：初始 DOE（拉丁超立方采样）
    # ------------------------------------------------------------------
    initial_points = _lhs_sample(bounds, config.n_initial, config.random_seed)

    for idx, point in enumerate(initial_points):
        if driver_dead:
            break

        design_vars = {**config.fixed_vars, **dict(zip(paths, point))}
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
        optimizer = _BayesianOptimizer(bounds, config)

        # 用初始 DOE 的观测初始化优化器（成功样本用真实 y，失败样本用惩罚值）
        for c in cases:
            x = [c.design_vars.get(p) for p in paths]
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

        for bo_idx in range(n_bo):
            if driver_dead:
                break

            idx = config.n_initial + bo_idx
            iteration = start_iteration + idx
            tags = list(config.tags) + ["bayesian_opt", "optimize"]

            next_x = optimizer.ask()
            design_vars = {**config.fixed_vars, **dict(zip(paths, next_x))}

            _log.info(
                "贝叶斯优化 [%d/%d]：%s",
                idx + 1, n_total,
                {k.split("\\")[-1]: round(v, 4) for k, v in dict(zip(paths, next_x)).items()},
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
            _save_case(db, case)
            _fire_callback(config.on_case_complete, case, idx, n_total)

            y = _extract_y(case, config)
            optimizer.tell(
                next_x,
                y if y is not None else _penalty_value(cases, config),
                is_success=y is not None,
            )

            _log.info(
                "  → status=%s, success=%s, run_time=%.1fs",
                case.status.value, case.success, case.run_time,
            )

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
# 贝叶斯优化器封装
# ---------------------------------------------------------------------------

class _BayesianOptimizer:
    """
    贝叶斯优化器封装，支持 skopt 高斯过程和随机采样回退。

    skopt 不可用，或成功观测数 < n_initial_min 时，ask() 返回随机点。
    skopt 可用且观测充足时，ask() 通过采集函数推荐下一个候选点。
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        config: OptimizeCaseConfig,
    ) -> None:
        self._bounds = bounds
        self._n_initial_min = config.n_initial_min
        self._rng = _random.Random(config.random_seed)
        self._n_success = 0
        self._skopt: Any = None

        if _HAS_SKOPT:
            acq_func_kwargs: dict[str, Any] = {}
            if config.acquisition in ("EI", "PI"):
                acq_func_kwargs["xi"] = config.xi
            elif config.acquisition == "UCB":
                acq_func_kwargs["kappa"] = config.kappa

            try:
                self._skopt = _SkoptOptimizer(
                    dimensions=[_Real(lo, hi) for lo, hi in bounds],
                    base_estimator="GP",
                    acq_func=config.acquisition,
                    acq_func_kwargs=acq_func_kwargs,
                    random_state=config.random_seed,
                    n_initial_points=0,
                )
            except Exception as exc:
                _log.warning("skopt Optimizer 初始化失败，回退到随机采样：%s", exc)
                self._skopt = None
        else:
            _log.warning(
                "scikit-optimize 未安装，贝叶斯优化将退化为随机采样。"
                "安装方法：pip install scikit-optimize"
            )

    def tell(self, x: list[float], y: float, *, is_success: bool) -> None:
        """向优化器提交一次观测。is_success=True 时计入成功样本数。"""
        if is_success:
            self._n_success += 1
        if self._skopt is not None:
            try:
                self._skopt.tell(x, y)
            except Exception as exc:
                _log.warning("skopt.tell() 失败（已忽略）：%s", exc)

    def ask(self) -> list[float]:
        """推荐下一个候选点。成功观测不足 n_initial_min 时返回随机点。"""
        if self._skopt is not None and self._n_success >= self._n_initial_min:
            try:
                return self._skopt.ask()
            except Exception as exc:
                _log.warning("skopt.ask() 失败，回退到随机采样：%s", exc)
        return [lo + self._rng.random() * (hi - lo) for lo, hi in self._bounds]


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

