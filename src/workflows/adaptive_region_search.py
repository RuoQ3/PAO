"""
adaptive_region_search.py — 自适应区域搜索 workflow。

核心思路
--------
传统优化器将整个输入空间视为均匀可搜索的，但实际上大量区域不可行或远离
Pareto 前沿，导致大量仿真预算浪费在无效区域。

本模块将输入空间划分为若干超矩形区域，通过三个阶段逐步聚焦搜索：

  Phase 0 — 空间分块（Region Partition）
    对整数变量（理论板数、进料位置）按步长划分网格；
    对连续变量（回流比、压力、萃取剂流量）按分位数划分区间。
    每个区域是一个超矩形，初始优先级相同。

  Phase 1 — 区域探测 DOE
    对每个区域做小规模 LHS（n_doe_per_region 个点），记录：
      - 收敛率（convergence_rate）
      - 可行率（feasibility_rate）
      - 目标值统计（均值、最优值）
    同时从所有 DOE 数据中计算变量敏感性，识别低敏感性变量。

  Phase 2 — 优先级更新与精细搜索
    按以下公式更新每个区域的采样优先级：
      priority = convergence_rate × feasibility_rate
                 × (1 + β × gp_uncertainty)
                 × improvement_potential
    优先级低于 prune_threshold 的区域标记为 PRUNED，不再采样。
    对保留区域调用 optimize_pareto_case() 执行精细贝叶斯优化。

层级关系
--------
adaptive_region_search()（本文件）
  ├── param_scan()（workflows/param_scan.py）         → Phase 1 DOE
  ├── sensitivity_analysis()（optimization/metrics.py）→ 变量重要性
  └── optimize_pareto_case()（workflows/optimize_pareto_case.py）→ Phase 2

设计约定
--------
- 不依赖任何第三方库（numpy/skopt 为可选加速项）
- 整数变量通过 IntegerVar 标记，分块时按步长离散化
- 敏感性分析结果自动写入日志，不强制降维（由调用方决定是否固定变量）
- 所有工况统一记录到同一 SQLite 数据库，tag 区分阶段

典型用法
--------
    from src.aspen_driver.driver import AspenDriver
    from src.workflows.adaptive_region_search import (
        AdaptiveRegionConfig, VarSpec, adaptive_region_search,
    )

    config = AdaptiveRegionConfig(
        var_specs=[
            VarSpec("\\Data\\Blocks\\T01\\Input\\NSTAGE", 20, 60, n_splits=4, is_integer=True),
            VarSpec("\\Data\\Blocks\\T01\\Input\\BASIS_RR", 1.0, 4.0, n_splits=3),
            VarSpec("\\Data\\Blocks\\T01\\Input\\PRES1", 1.0, 5.0, n_splits=3),
        ],
        fixed_vars={},
        objective_names=["TAC", "EMISSIONS"],
        run_config=run_cfg,
        n_doe_per_region=6,
        n_bo_per_region=20,
        prune_threshold=0.05,
        sensitivity_threshold=0.1,
        db_path="output/adaptive.db",
    )

    with AspenDriver() as driver:
        driver.open("process.bkp")
        result = adaptive_region_search(driver, config)
"""
from __future__ import annotations

import logging
import math
import random as _random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from ..aspen_driver.driver import AspenDriver
from ..aspen_driver.errors import AspenConnectionError
from ..models.process_case import CaseStatus, ProcessCase
from ..optimization.metrics import SensitivityResult, rank_variables, sensitivity_analysis
from ..optimization.pareto import ParetoResult, compute_pareto
from .optimize_pareto_case import ParetoOptimizeCaseConfig, optimize_pareto_case
from .param_scan import ParamScanConfig, ScanResult, param_scan
from .run_case import RunCaseConfig

_log = logging.getLogger(__name__)

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# 变量规格
# ---------------------------------------------------------------------------

@dataclass
class VarSpec:
    """
    单个设计变量的规格描述，用于区域分块。

    Attributes
    ----------
    aspen_path:
        Aspen 树路径。
    lower_bound:
        搜索下界。
    upper_bound:
        搜索上界。
    n_splits:
        沿该维度的分块数，默认 3。
        整数变量：实际分块数 = min(n_splits, (upper-lower)//step + 1)。
        连续变量：分块数 = n_splits。
    is_integer:
        True 表示整数变量（理论板数、进料位置），分块时按 step 离散化。
    step:
        整数变量的步长，默认 1。连续变量忽略此参数。
    """
    aspen_path: str
    lower_bound: float
    upper_bound: float
    n_splits: int = 3
    is_integer: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                f"VarSpec '{self.aspen_path}'：lower_bound {self.lower_bound} "
                f">= upper_bound {self.upper_bound}。"
            )
        if self.n_splits < 1:
            raise ValueError(
                f"VarSpec '{self.aspen_path}'：n_splits 必须 >= 1，收到 {self.n_splits}。"
            )
        if self.is_integer and self.step < 1:
            raise ValueError(
                f"VarSpec '{self.aspen_path}'：整数变量的 step 必须 >= 1，收到 {self.step}。"
            )


# ---------------------------------------------------------------------------
# 区域状态与数据类
# ---------------------------------------------------------------------------

class RegionStatus(str, Enum):
    PENDING  = "pending"   # 尚未探测
    EXPLORED = "explored"  # Phase 1 DOE 已完成
    ACTIVE   = "active"    # Phase 2 精细搜索中
    PRUNED   = "pruned"    # 优先级过低，已剪枝


@dataclass
class Region:
    """
    输入空间中的一个超矩形区域。

    Attributes
    ----------
    region_id:
        区域唯一标识符（整数索引）。
    bounds:
        各变量的边界 {aspen_path: (lower, upper)}。
    status:
        区域状态（RegionStatus 枚举）。
    n_sampled:
        已采样的工况数。
    n_converged:
        仿真收敛的工况数。
    n_feasible:
        满足约束的工况数。
    best_objectives:
        该区域内最优目标值向量（最小化方向），None 表示尚无成功样本。
    gp_uncertainty:
        代理模型在该区域的平均预测不确定性（Phase 1 后估计）。
    priority:
        采样优先级（0 ~ 1），由 _update_priority() 动态更新。
    cases:
        该区域内所有工况的 ProcessCase 列表。
    """
    region_id: int
    bounds: dict[str, tuple[float, float]]
    status: RegionStatus = RegionStatus.PENDING
    n_sampled: int = 0
    n_converged: int = 0
    n_feasible: int = 0
    best_objectives: list[float] | None = None
    gp_uncertainty: float = 0.5
    priority: float = 1.0
    cases: list[ProcessCase] = field(default_factory=list)

    @property
    def convergence_rate(self) -> float:
        return self.n_converged / self.n_sampled if self.n_sampled > 0 else 0.0

    @property
    def feasibility_rate(self) -> float:
        return self.n_feasible / self.n_sampled if self.n_sampled > 0 else 0.0

    def to_summary(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "status": self.status.value,
            "n_sampled": self.n_sampled,
            "n_converged": self.n_converged,
            "n_feasible": self.n_feasible,
            "convergence_rate": round(self.convergence_rate, 3),
            "feasibility_rate": round(self.feasibility_rate, 3),
            "priority": round(self.priority, 4),
            "best_objectives": self.best_objectives,
            "bounds": {k.split("\\")[-1]: list(v) for k, v in self.bounds.items()},
        }


# ---------------------------------------------------------------------------
# 搜索配置
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveRegionConfig:
    """
    adaptive_region_search() 的配置参数。

    Attributes
    ----------
    var_specs:
        设计变量规格列表，定义搜索空间和分块策略。
    fixed_vars:
        固定不变的设计变量 {Aspen 路径: 值}，不参与分块和搜索。
    objective_names:
        参与多目标优化的目标函数名称列表（至少 2 个）。
    run_config:
        每次单次运行的配置（RunCaseConfig）。
    n_doe_per_region:
        Phase 1 每个区域的 DOE 样本数，默认 6。
    n_bo_per_region:
        Phase 2 每个保留区域的贝叶斯优化迭代数，默认 20。
    prune_threshold:
        优先级低于此值的区域被剪枝，默认 0.05。
    sensitivity_threshold:
        综合敏感性低于此值的变量在日志中标记为"建议固定"，默认 0.1。
        注意：本模块不自动固定变量，仅记录建议，由调用方决定。
    exploration_beta:
        优先级公式中的探索权重 β，控制不确定性对优先级的贡献，默认 0.5。
    sensitivity_method:
        敏感性分析方法："spearman"（默认）或 "variance"。
    scalarization:
        Phase 2 多目标标量化方法："chebyshev"（默认）或 "weighted_sum"。
    acquisition:
        Phase 2 采集函数："EI"（默认）、"UCB"、"PI"。
    tags:
        应用到所有工况的标签列表。
    on_case_complete:
        每次工况完成后的回调函数，签名为 (case, region_id, phase) -> None。
    db_path:
        SQLite 数据库路径，若指定则每次工况完成后自动持久化。
    random_seed:
        随机种子。
    """
    var_specs: list[VarSpec]
    objective_names: list[str]
    run_config: RunCaseConfig = field(default_factory=RunCaseConfig)
    fixed_vars: dict[str, Any] = field(default_factory=dict)
    n_doe_per_region: int = 6
    n_bo_per_region: int = 20
    prune_threshold: float = 0.05
    sensitivity_threshold: float = 0.1
    exploration_beta: float = 0.5
    sensitivity_method: Literal["spearman", "variance"] = "spearman"
    scalarization: Literal["chebyshev", "weighted_sum"] = "chebyshev"
    acquisition: Literal["EI", "UCB", "PI"] = "EI"
    tags: list[str] = field(default_factory=list)
    on_case_complete: Callable[[ProcessCase, int, str], None] | None = None
    db_path: Path | str | None = None
    random_seed: int | None = None


# ---------------------------------------------------------------------------
# 搜索结果
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveRegionResult:
    """
    adaptive_region_search() 的返回值。

    Attributes
    ----------
    all_cases:
        所有阶段的全部工况（Phase 1 DOE + Phase 2 贝叶斯优化）。
    regions:
        所有区域的 Region 列表（含剪枝区域）。
    pareto_result:
        基于所有成功工况计算的最终 Pareto 前沿。
    sensitivity_result:
        Phase 1 结束后的变量敏感性分析结果。
    n_regions_total:
        初始区域总数。
    n_regions_pruned:
        被剪枝的区域数。
    n_regions_explored:
        Phase 2 精细搜索的区域数。
    n_total:
        总工况数（Phase 1 + Phase 2）。
    n_success:
        成功工况数。
    elapsed:
        总耗时（秒）。
    """
    all_cases: list[ProcessCase]
    regions: list[Region]
    pareto_result: ParetoResult
    sensitivity_result: SensitivityResult | None
    n_regions_total: int
    n_regions_pruned: int
    n_regions_explored: int
    n_total: int
    n_success: int
    elapsed: float
    persistence_errors: list[str] = field(default_factory=list)

    @property
    def active_regions(self) -> list[Region]:
        return [r for r in self.regions if r.status != RegionStatus.PRUNED]

    @property
    def pruned_regions(self) -> list[Region]:
        return [r for r in self.regions if r.status == RegionStatus.PRUNED]

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_total if self.n_total > 0 else 0.0

    def to_summary(self) -> dict[str, Any]:
        return {
            "n_regions_total": self.n_regions_total,
            "n_regions_pruned": self.n_regions_pruned,
            "n_regions_explored": self.n_regions_explored,
            "n_total": self.n_total,
            "n_success": self.n_success,
            "success_rate": self.success_rate,
            "elapsed": self.elapsed,
            "hypervolume": (
                self.pareto_result.hypervolume
                if self.pareto_result else None
            ),
            "first_front_size": (
                len(self.pareto_result.first_front.cases)
                if self.pareto_result and self.pareto_result.first_front else 0
            ),
            "n_persistence_errors": len(self.persistence_errors),
            "persistence_errors": self.persistence_errors[:5],
        }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def adaptive_region_search(
    driver: AspenDriver,
    config: AdaptiveRegionConfig,
) -> AdaptiveRegionResult:
    """
    执行自适应区域搜索，返回 AdaptiveRegionResult。

    Parameters
    ----------
    driver:
        已连接并打开仿真文件的 AspenDriver 实例。
    config:
        搜索配置，见 AdaptiveRegionConfig。

    Returns
    -------
    AdaptiveRegionResult
    """
    _validate_config(config)
    t0 = time.monotonic()
    rng = _random.Random(config.random_seed)

    db = None
    if config.db_path is not None:
        from ..database.simulation_db import SimulationDB
        db = SimulationDB(config.db_path)

    # ------------------------------------------------------------------
    # Phase 0：空间分块
    # ------------------------------------------------------------------
    regions = _partition_space(config)
    _log.info(
        "自适应区域搜索开始：%d 个区域，%d 个变量，目标=%s。",
        len(regions), len(config.var_specs), config.objective_names,
    )

    all_cases: list[ProcessCase] = []
    sensitivity_result: SensitivityResult | None = None
    persistence_errors: list[str] = []

    # ------------------------------------------------------------------
    # Phase 1：区域探测 DOE
    # ------------------------------------------------------------------
    _log.info("Phase 1：区域探测 DOE（每区域 %d 个样本）。", config.n_doe_per_region)

    for region in regions:
        scan_cfg = _build_scan_config(region, config, rng, phase="doe")
        try:
            scan_result = param_scan(driver, scan_cfg)
        except AspenConnectionError:
            _log.error("Phase 1：driver 连接断开，终止搜索。")
            break
        except Exception as exc:
            _log.warning("Phase 1 区域 %d：param_scan 异常（已跳过）：%s", region.region_id, exc)
            continue

        _update_region_from_scan(region, scan_result, config)
        all_cases.extend(scan_result.cases)

        for case in scan_result.cases:
            _save_case(db, case, persistence_errors)
            if config.on_case_complete is not None:
                _fire_callback(config.on_case_complete, case, region.region_id, "doe")

        _log.info(
            "  区域 %d：收敛率=%.1f%%，可行率=%.1f%%，采样=%d。",
            region.region_id,
            region.convergence_rate * 100,
            region.feasibility_rate * 100,
            region.n_sampled,
        )

    # ------------------------------------------------------------------
    # Phase 1 后：敏感性分析 + 优先级更新
    # ------------------------------------------------------------------
    param_paths = [v.aspen_path for v in config.var_specs]

    # 使用 include_infeasible=True，让 metrics.py 自行处理样本不足和路径错误，
    # 充分利用收敛但约束违反的样本识别可行域边界和高敏感变量。
    sensitivity_result = sensitivity_analysis(
        cases=all_cases,
        param_paths=param_paths,
        objective_names=config.objective_names,
        method=config.sensitivity_method,
        include_infeasible=True,
    )
    ranked = rank_variables(sensitivity_result, threshold=config.sensitivity_threshold)
    _log.info(
        "敏感性排序：%s",
        [(p.split("\\")[-1], round(s, 3)) for p, s in ranked],
    )

    # 计算全局最优参考，用于 improvement_potential
    global_best = _compute_global_best(all_cases, config.objective_names)

    for region in regions:
        region.status = RegionStatus.EXPLORED
        _update_priority(region, global_best, config.exploration_beta, config.prune_threshold)

    n_pruned = sum(1 for r in regions if r.priority < config.prune_threshold)
    for region in regions:
        if region.priority < config.prune_threshold:
            region.status = RegionStatus.PRUNED
            _log.info(
                "  区域 %d 已剪枝（priority=%.4f < threshold=%.4f）。",
                region.region_id, region.priority, config.prune_threshold,
            )

    active_regions = [r for r in regions if r.status != RegionStatus.PRUNED]
    _log.info(
        "Phase 1 完成：%d 个区域保留，%d 个区域剪枝。",
        len(active_regions), n_pruned,
    )

    # ------------------------------------------------------------------
    # Phase 2：精细贝叶斯优化
    # ------------------------------------------------------------------
    _log.info(
        "Phase 2：对 %d 个保留区域执行贝叶斯优化（每区域 %d 次迭代）。",
        len(active_regions), config.n_bo_per_region,
    )

    # 按优先级降序处理
    active_regions.sort(key=lambda r: r.priority, reverse=True)

    for region in active_regions:
        region.status = RegionStatus.ACTIVE
        _log.info(
            "Phase 2 区域 %d（priority=%.4f）：启动贝叶斯优化。",
            region.region_id, region.priority,
        )

        bo_cfg = _build_bo_config(region, config, all_cases, rng)
        if bo_cfg is None:
            # 全整数区域，无连续搜索空间，已在 _build_bo_config 中记录日志
            continue

        try:
            bo_result = optimize_pareto_case(driver, bo_cfg)
        except AspenConnectionError:
            _log.error("Phase 2 区域 %d：driver 连接断开，终止搜索。", region.region_id)
            break
        except Exception as exc:
            _log.warning(
                "Phase 2 区域 %d：optimize_pareto_case 异常（已跳过）：%s",
                region.region_id, exc,
            )
            continue

        all_cases.extend(bo_result.cases)
        region.cases.extend(bo_result.cases)

        for case in bo_result.cases:
            _save_case(db, case, persistence_errors)
            if config.on_case_complete is not None:
                _fire_callback(config.on_case_complete, case, region.region_id, "bo")

        _log.info(
            "  区域 %d 完成：%d/%d 成功，HV=%s。",
            region.region_id,
            bo_result.n_success, bo_result.n_total,
            f"{bo_result.hypervolume:.4g}" if bo_result.hypervolume is not None else "N/A",
        )

    # ------------------------------------------------------------------
    # 汇总结果
    # ------------------------------------------------------------------
    try:
        elapsed = time.monotonic() - t0
        n_success = sum(1 for c in all_cases if c.success)
        n_explored = sum(1 for r in regions if r.status in (RegionStatus.ACTIVE, RegionStatus.EXPLORED))

        pareto_result = compute_pareto(
            all_cases,
            config.objective_names,
            compute_hv=True,
        )

        _log.info(
            "自适应区域搜索完成：%d 个工况，%d 成功，第一前沿 %d 个解，HV=%s，耗时 %.1fs。",
            len(all_cases), n_success,
            len(pareto_result.first_front.cases) if pareto_result.first_front else 0,
            f"{pareto_result.hypervolume:.4g}" if pareto_result.hypervolume is not None else "N/A",
            elapsed,
        )

        return AdaptiveRegionResult(
            all_cases=all_cases,
            regions=regions,
            pareto_result=pareto_result,
            sensitivity_result=sensitivity_result,
            n_regions_total=len(regions),
            n_regions_pruned=n_pruned,
            n_regions_explored=n_explored,
            n_total=len(all_cases),
            n_success=n_success,
            elapsed=elapsed,
            persistence_errors=persistence_errors,
        )
    finally:
        if db is not None:
            db.close()


# ---------------------------------------------------------------------------
# Phase 0：空间分块
# ---------------------------------------------------------------------------

def _partition_space(config: AdaptiveRegionConfig) -> list[Region]:
    """
    将输入空间划分为超矩形区域列表。

    整数变量：按 step 离散化后均匀分组，每组包含若干离散值。
    连续变量：按等间距分位数划分区间。
    所有变量的分块取笛卡尔积，生成全部区域。
    """
    # 每个变量的区间列表：[(lower, upper), ...]
    var_intervals: list[list[tuple[float, float]]] = []

    for spec in config.var_specs:
        if spec.is_integer:
            intervals = _integer_intervals(spec)
        else:
            intervals = _continuous_intervals(spec)
        var_intervals.append(intervals)

    # 笛卡尔积生成所有区域
    regions: list[Region] = []
    region_id = 0

    def _cartesian(lists: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
        if not lists:
            return [[]]
        result = []
        for item in lists[0]:
            for rest in _cartesian(lists[1:]):
                result.append([item] + rest)
        return result

    combos = _cartesian(var_intervals)
    for combo in combos:
        bounds = {
            config.var_specs[i].aspen_path: combo[i]
            for i in range(len(config.var_specs))
        }
        regions.append(Region(region_id=region_id, bounds=bounds))
        region_id += 1

    _log.info(
        "空间分块完成：%d 个变量 → %d 个区域。",
        len(config.var_specs), len(regions),
    )
    return regions


def _integer_intervals(spec: VarSpec) -> list[tuple[float, float]]:
    """整数变量：生成 n_splits 个等大小的离散值区间。"""
    lo = int(math.ceil(spec.lower_bound))
    hi = int(math.floor(spec.upper_bound))
    values = list(range(lo, hi + 1, spec.step))
    if not values:
        return [(spec.lower_bound, spec.upper_bound)]

    n = min(spec.n_splits, len(values))
    if n <= 1:
        return [(float(values[0]), float(values[-1]))]

    chunk = len(values) / n
    intervals: list[tuple[float, float]] = []
    for i in range(n):
        start_idx = int(i * chunk)
        end_idx = int((i + 1) * chunk) - 1
        end_idx = min(end_idx, len(values) - 1)
        intervals.append((float(values[start_idx]), float(values[end_idx])))
    return intervals


def _continuous_intervals(spec: VarSpec) -> list[tuple[float, float]]:
    """连续变量：生成 n_splits 个等宽区间。"""
    lo, hi = spec.lower_bound, spec.upper_bound
    width = (hi - lo) / spec.n_splits
    return [
        (lo + i * width, lo + (i + 1) * width)
        for i in range(spec.n_splits)
    ]


# ---------------------------------------------------------------------------
# Phase 1：构建 DOE 扫描配置
# ---------------------------------------------------------------------------

def _build_scan_config(
    region: Region,
    config: AdaptiveRegionConfig,
    rng: _random.Random,
    phase: str,
) -> ParamScanConfig:
    """
    为指定区域构建 LHS DOE 的 ParamScanConfig。

    在区域边界内生成 n_doe_per_region 个 LHS 样本点，
    整数变量四舍五入到最近的合法整数值。
    """
    n = config.n_doe_per_region
    paths = list(region.bounds.keys())
    bounds_list = [region.bounds[p] for p in paths]

    # LHS 采样
    points = _lhs_sample(bounds_list, n, rng)

    # 整数变量取整
    int_paths = {spec.aspen_path for spec in config.var_specs if spec.is_integer}
    int_steps = {spec.aspen_path: spec.step for spec in config.var_specs if spec.is_integer}
    for pt in points:
        for j, path in enumerate(paths):
            if path in int_paths:
                step = int_steps[path]
                pt[j] = float(round(pt[j] / step) * step)

    # 转为 scan_vars（zip 模式）
    scan_vars: dict[str, list[Any]] = {
        path: [points[i][j] for i in range(n)]
        for j, path in enumerate(paths)
    }

    tags = list(config.tags) + [f"region_{region.region_id}", f"phase1_{phase}"]

    return ParamScanConfig(
        scan_vars=scan_vars,
        fixed_vars=config.fixed_vars,
        run_config=config.run_config,
        mode="zip",
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Phase 1：更新区域统计
# ---------------------------------------------------------------------------

def _update_region_from_scan(
    region: Region,
    scan_result: ScanResult,
    config: AdaptiveRegionConfig,
) -> None:
    """从 ScanResult 更新区域的收敛率、可行率和最优目标值。"""
    region.cases.extend(scan_result.cases)
    region.n_sampled += scan_result.n_total
    region.n_converged += scan_result.n_simulation_valid

    for case in scan_result.cases:
        if case.success:
            region.n_feasible += 1
            obj_vec = _extract_objectives(case, config.objective_names)
            if obj_vec is not None:
                if region.best_objectives is None:
                    region.best_objectives = obj_vec
                else:
                    # 取各维度最小值（最小化方向）
                    region.best_objectives = [
                        min(a, b)
                        for a, b in zip(region.best_objectives, obj_vec)
                    ]


# ---------------------------------------------------------------------------
# 优先级更新
# ---------------------------------------------------------------------------

def _update_priority(
    region: Region,
    global_best: list[float] | None,
    beta: float,
    prune_threshold: float,
) -> None:
    """
    更新区域采样优先级。

    公式：
      priority = convergence_rate × feasibility_rate
                 × (1 + β × gp_uncertainty)
                 × improvement_potential

    improvement_potential（衰减函数）：
      normalized_distance = 区域最优与全局最优的归一化距离（越小越好）
      improvement_potential = 1 / (1 + normalized_distance)
      → 越接近全局最优的区域 priority 越高（正确方向）
      → 无全局最优或区域无成功样本时设为 1.0（保持探索）

    不可行但收敛区域：
      给予高于 prune_threshold 的保护优先级，避免边界诊断信息被剪枝。
    """
    cr = region.convergence_rate
    fr = region.feasibility_rate

    if region.n_sampled == 0:
        region.priority = 1.0
        return

    # 改进潜力：衰减函数，越接近全局最优 priority 越高
    improvement = 1.0
    if global_best is not None and region.best_objectives is not None:
        diffs = [
            max(0.0, region.best_objectives[i] - global_best[i])
            for i in range(len(global_best))
        ]
        refs = [abs(v) + 1e-6 for v in global_best]
        normalized_distance = sum(d / r for d, r in zip(diffs, refs)) / len(diffs)
        improvement = 1.0 / (1.0 + normalized_distance)

    region.priority = cr * fr * (1.0 + beta * region.gp_uncertainty) * improvement

    # 不可行但收敛：保护优先级高于剪枝阈值，保留边界诊断信息
    if region.n_feasible == 0 and region.n_converged > 0:
        region.priority = max(region.priority, prune_threshold + 0.01)


# ---------------------------------------------------------------------------
# Phase 2：构建贝叶斯优化配置
# ---------------------------------------------------------------------------

def _build_bo_config(
    region: Region,
    config: AdaptiveRegionConfig,
    all_cases: list[ProcessCase],
    rng: _random.Random,
) -> ParetoOptimizeCaseConfig | None:
    """
    为指定区域构建 Phase 2 贝叶斯优化配置。

    整数变量处理：
      optimize_pareto_case() 将所有变量视为连续实数，直接传入整数变量边界
      会导致 Aspen 收到非法值（如理论板数 1.37）。
      因此整数变量统一固定到 Phase 1 中该区域的最优整数值（无成功样本时取中点），
      移入 fixed_vars，不参与连续 BO 搜索。

    warm_start_cases 过滤：
      只保留所有固定整数变量与当前固定值一致的 Phase 1 样本。
      整数值不同的样本会把整数变量造成的目标差异误归因到连续变量，污染 GP。

    全整数区域：
      若所有变量均为整数，param_bounds 为空，无法构建连续 BO，返回 None。
      调用方应显式跳过，不应依赖 optimize_pareto_case() 抛出 ValueError。
    """
    int_specs = {spec.aspen_path: spec for spec in config.var_specs if spec.is_integer}
    fixed_vars = dict(config.fixed_vars)
    param_bounds: dict[str, tuple[float, float]] = {}

    for path, bounds in region.bounds.items():
        lo, hi = bounds
        if path in int_specs:
            spec = int_specs[path]
            valid_ints = [
                v for v in range(int(math.ceil(lo)), int(math.floor(hi)) + 1, spec.step)
            ]
            if not valid_ints:
                val = int(round((lo + hi) / 2))
            elif len(valid_ints) == 1:
                val = valid_ints[0]
            else:
                val = _best_integer_from_cases(
                    region.cases, path, valid_ints, config.objective_names
                )
            fixed_vars[path] = float(val)
        else:
            param_bounds[path] = bounds

    if not param_bounds:
        _log.info(
            "区域 %d：所有变量均为整数，Phase 2 无连续搜索空间，跳过 BO。",
            region.region_id,
        )
        return None

    # ------------------------------------------------------------------
    # warm_start_cases 过滤：只保留整数固定值与当前一致的 Phase 1 样本
    # ------------------------------------------------------------------
    int_fixed = {
        path: int(round(val))
        for path, val in fixed_vars.items()
        if path in int_specs
    }
    filtered_warm = [
        c for c in region.cases
        if all(
            c.design_vars.get(path) is not None
            and int(round(float(c.design_vars[path]))) == expected
            for path, expected in int_fixed.items()
        )
    ]

    tags = list(config.tags) + [f"region_{region.region_id}", "phase2_bo"]

    return ParetoOptimizeCaseConfig(
        param_bounds=param_bounds,
        objective_names=config.objective_names,
        fixed_vars=fixed_vars,
        run_config=config.run_config,
        n_initial=min(config.n_doe_per_region, config.n_bo_per_region // 2),
        n_iterations=config.n_bo_per_region,
        scalarization=config.scalarization,
        acquisition=config.acquisition,
        tags=tags,
        warm_start_cases=filtered_warm,
        random_seed=rng.randint(0, 2 ** 31),
    )


def _best_integer_from_cases(
    cases: list[ProcessCase],
    path: str,
    valid_ints: list[int],
    objective_names: list[str],
) -> int:
    """
    从 Phase 1 成功样本中找指定路径的最优整数值。

    最优定义：使第一个目标最小化的整数值（最小化方向）。
    无成功样本时返回 valid_ints 的中间值。
    """
    best_val: int | None = None
    best_obj: float = float("inf")

    for case in cases:
        if not case.success:
            continue
        raw = case.design_vars.get(path)
        if raw is None:
            continue
        int_val = int(round(float(raw)))
        if int_val not in valid_ints:
            continue
        obj = case.get_objective(objective_names[0])
        if obj is None or not obj.available or obj.value is None:
            continue
        obj_val = float(obj.value) if obj.minimize else -float(obj.value)
        if obj_val < best_obj:
            best_obj = obj_val
            best_val = int_val

    if best_val is None:
        best_val = valid_ints[len(valid_ints) // 2]
    return best_val


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _extract_objectives(
    case: ProcessCase,
    objective_names: list[str],
) -> list[float] | None:
    """提取工况的目标值向量（最小化方向），任意目标不可用时返回 None。"""
    if not case.success:
        return None
    result: list[float] = []
    for name in objective_names:
        obj = case.get_objective(name)
        if obj is None or not obj.available:
            return None
        val = float(obj.value)  # type: ignore[arg-type]
        if not math.isfinite(val):
            return None
        result.append(val if obj.minimize else -val)
    return result


def _compute_global_best(
    cases: list[ProcessCase],
    objective_names: list[str],
) -> list[float] | None:
    """计算所有成功工况中各目标的全局最优值（最小化方向）。"""
    best: list[float] | None = None
    for case in cases:
        vec = _extract_objectives(case, objective_names)
        if vec is None:
            continue
        if best is None:
            best = vec[:]
        else:
            best = [min(a, b) for a, b in zip(best, vec)]
    return best


def _lhs_sample(
    bounds: list[tuple[float, float]],
    n: int,
    rng: _random.Random,
) -> list[list[float]]:
    """拉丁超立方采样，与 optimize_pareto_case.py 保持一致。"""
    d = len(bounds)
    if _HAS_NUMPY:
        import numpy as np
        seed_val = rng.randint(0, 2 ** 31)
        np_rng = np.random.default_rng(seed_val)
        samples = np.zeros((n, d))
        for j, (lo, hi) in enumerate(bounds):
            perm = np_rng.permutation(n)
            u = (perm + np_rng.random(n)) / n
            samples[:, j] = lo + u * (hi - lo)
        return samples.tolist()

    cols: list[list[float]] = []
    for lo, hi in bounds:
        perm = list(range(n))
        rng.shuffle(perm)
        col = [lo + (perm[i] + rng.random()) / n * (hi - lo) for i in range(n)]
        cols.append(col)
    return [[cols[j][i] for j in range(d)] for i in range(n)]


def _save_case(db: Any, case: ProcessCase, persistence_errors: list[str]) -> None:
    if db is None:
        return
    try:
        db.save_case(case.to_dict())
    except Exception as exc:
        msg = f"工况 '{case.case_id}' 保存到数据库失败：{exc}"
        _log.warning(msg)
        persistence_errors.append(msg)


def _fire_callback(
    callback: Callable[[ProcessCase, int, str], None],
    case: ProcessCase,
    region_id: int,
    phase: str,
) -> None:
    try:
        callback(case, region_id, phase)
    except Exception as exc:
        _log.warning("on_case_complete 回调异常（已忽略）：%s", exc)


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

def _validate_config(config: AdaptiveRegionConfig) -> None:
    if not config.var_specs:
        raise ValueError("var_specs 不能为空，至少需要一个设计变量。")

    if len(config.objective_names) < 2:
        raise ValueError(
            f"objective_names 至少需要 2 个目标，收到 {len(config.objective_names)} 个。"
        )

    if config.n_doe_per_region < 2:
        raise ValueError(f"n_doe_per_region 必须 >= 2，收到 {config.n_doe_per_region}。")

    if config.n_bo_per_region < config.n_doe_per_region:
        raise ValueError(
            f"n_bo_per_region={config.n_bo_per_region} 必须 >= "
            f"n_doe_per_region={config.n_doe_per_region}。"
        )

    if not 0.0 <= config.prune_threshold <= 1.0:
        raise ValueError(
            f"prune_threshold 必须在 [0, 1] 范围内，收到 {config.prune_threshold}。"
        )

    # 检查 fixed_vars 与 var_specs 无路径冲突
    spec_paths_upper = {s.aspen_path.upper() for s in config.var_specs}
    fixed_upper = {p.upper(): p for p in config.fixed_vars}
    conflicts = [
        (fixed_upper[u], next(s.aspen_path for s in config.var_specs if s.aspen_path.upper() == u))
        for u in spec_paths_upper & set(fixed_upper)
    ]
    if conflicts:
        detail = "; ".join(f"fixed={f!r} vs spec={s!r}" for f, s in conflicts)
        raise ValueError(
            f"fixed_vars 与 var_specs 存在路径冲突（大小写不敏感）：{detail}。"
        )
