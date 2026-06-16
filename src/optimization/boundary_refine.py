"""
boundary_refine.py — 数据驱动的搜索边界收缩（自校准层）。

职责（纯统计、确定性、可复现,不含任何 LLM 或工艺先验）：
在优化跑出一批样本后,用实际的 success/infeasible/sim_failed 数据,
统计"可行点在每个变量上的实际取值范围",据此动态收紧搜索边界。

它是 boundary_advisor(冷启动给初始 k)的自校准层：
  - boundary_advisor 用物理常识给一个"大致靠谱"的初始边界。
  - boundary_refine 跑起来后用真实数据修正它——如果可行点都挤在初始边界的
    一小段里,就把边界收到那一段(加裕量);如果可行点贴着边界,说明可能还能
    往外探,保持或略放宽。

与 ThawScheduler 的分工：
  - ThawScheduler 管"高敏感变量的窄→宽解冻节奏"（基于 HV 停滞,事件驱动）。
  - boundary_refine 管"基于已有可行样本分布的边界重估"（基于数据统计,周期性）。
  两者互补：前者控探索时机,后者控边界位置。

设计原则：
  - 只在可行样本数足够(>= min_feasible)时才收缩,样本不足返回原边界。
  - 收缩后的边界总是原边界的子集(只收不放,放由 ThawScheduler 负责),
    避免与解冻机制打架。
  - 加入 margin 裕量,不把边界收到贴着已知可行点(留探索空间)。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRefineConfig:
    """
    数据驱动边界收缩配置。

    Attributes
    ----------
    enabled:
        是否启用,默认 False（不影响现有流程）。
    min_feasible:
        触发收缩所需的最少可行样本数,默认 15。不足时返回原边界。
    margin_frac:
        在可行点实际范围两侧各扩展的裕量,占该范围宽度的比例,默认 0.25。
        例:可行点落在 [0.05, 0.20],margin_frac=0.25 → 收缩后 [0.0125, 0.2375]。
        裕量保证不把边界贴死在已知可行点,保留局部探索空间。
    max_shrink_frac:
        单次收缩后的边界宽度相对原边界宽度的最小比例,默认 0.1。
        即一次最多把宽度收到原来的 10%,防止过度收缩到近乎单点。
    only_shrink:
        True（默认）：收缩后的边界必与原边界取交集（只收不放）。
        False：允许收缩边界略超出原边界（一般不需要,放宽交给 ThawScheduler）。
    """
    enabled: bool = False
    min_feasible: int = 15
    margin_frac: float = 0.25
    max_shrink_frac: float = 0.1
    only_shrink: bool = True


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------

@dataclass
class RefineResult:
    """单变量边界重估结果。"""
    path: str
    old_bounds: tuple[float, float]
    new_bounds: tuple[float, float]
    n_feasible_used: int
    shrunk: bool
    note: str


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def refine_bounds(
    feasible_points: list[dict[str, float]],
    current_bounds: dict[str, tuple[float, float]],
    config: BoundaryRefineConfig,
) -> tuple[dict[str, tuple[float, float]], list[RefineResult]]:
    """基于可行样本分布,为每个变量重估(收紧)搜索边界。

    Parameters
    ----------
    feasible_points:
        可行工况的设计变量列表 [{path: value, ...}, ...]。
        通常来自 SimulationDB 里 success=True 的样本。
    current_bounds:
        当前各变量搜索边界 {path: (lo, hi)}。
    config:
        收缩配置。

    Returns
    -------
    (new_bounds, results)
        new_bounds：{path: (lo, hi)},收缩后的边界(未触发收缩的变量保留原值)。
        results：每个变量的 RefineResult,供日志/诊断。
    """
    new_bounds: dict[str, tuple[float, float]] = dict(current_bounds)
    results: list[RefineResult] = []

    if not config.enabled:
        return new_bounds, results

    n_total = len(feasible_points)
    if n_total < config.min_feasible:
        _log.debug(
            "boundary_refine：可行样本 %d < min_feasible %d,跳过收缩。",
            n_total, config.min_feasible,
        )
        return new_bounds, results

    for path, (old_lo, old_hi) in current_bounds.items():
        vals = [
            float(p[path]) for p in feasible_points
            if path in p and _is_number(p[path])
        ]
        if len(vals) < config.min_feasible:
            results.append(RefineResult(
                path=path, old_bounds=(old_lo, old_hi), new_bounds=(old_lo, old_hi),
                n_feasible_used=len(vals), shrunk=False,
                note=f"该变量可行样本 {len(vals)} < {config.min_feasible},保留原边界",
            ))
            continue

        v_min, v_max = min(vals), max(vals)
        span = v_max - v_min
        # 退化:所有可行点该变量取值相同 → 用原边界宽度的一小部分做裕量
        if span <= 0:
            span = (old_hi - old_lo) * config.max_shrink_frac

        margin = span * config.margin_frac
        cand_lo = v_min - margin
        cand_hi = v_max + margin

        # only_shrink：与原边界取交集,只收不放
        if config.only_shrink:
            cand_lo = max(cand_lo, old_lo)
            cand_hi = min(cand_hi, old_hi)

        # 防过度收缩：保证新宽度 >= 原宽度 × max_shrink_frac
        old_width = old_hi - old_lo
        min_width = old_width * config.max_shrink_frac
        if (cand_hi - cand_lo) < min_width:
            center = (cand_lo + cand_hi) / 2.0
            cand_lo = center - min_width / 2.0
            cand_hi = center + min_width / 2.0
            if config.only_shrink:
                cand_lo = max(cand_lo, old_lo)
                cand_hi = min(cand_hi, old_hi)

        # 合法性兜底
        if cand_lo >= cand_hi:
            results.append(RefineResult(
                path=path, old_bounds=(old_lo, old_hi), new_bounds=(old_lo, old_hi),
                n_feasible_used=len(vals), shrunk=False,
                note="收缩后边界非法,保留原边界",
            ))
            continue

        shrunk = (cand_lo > old_lo + 1e-12) or (cand_hi < old_hi - 1e-12)
        new_bounds[path] = (cand_lo, cand_hi)
        results.append(RefineResult(
            path=path, old_bounds=(old_lo, old_hi), new_bounds=(cand_lo, cand_hi),
            n_feasible_used=len(vals), shrunk=shrunk,
            note=(
                f"可行点范围[{v_min:g},{v_max:g}],加 {config.margin_frac:.0%} 裕量"
                if shrunk else "可行点已铺满原边界,未收缩"
            ),
        ))

    n_shrunk = sum(1 for r in results if r.shrunk)
    if n_shrunk:
        _log.info(
            "boundary_refine：基于 %d 个可行样本,收紧了 %d/%d 个变量的边界。",
            n_total, n_shrunk, len(current_bounds),
        )
    return new_bounds, results


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def extract_feasible_points(
    cases: list[Any],
    paths: list[str],
) -> list[dict[str, float]]:
    """从 ProcessCase 列表抽取可行工况的设计变量字典。

    可行定义：case.success 为 True。design_vars 缺某变量时该点跳过该变量。
    """
    points: list[dict[str, float]] = []
    for c in cases:
        if not getattr(c, "success", False):
            continue
        dv = getattr(c, "design_vars", None) or {}
        point: dict[str, float] = {}
        for p in paths:
            if p in dv and _is_number(dv[p]):
                point[p] = float(dv[p])
        if point:
            points.append(point)
    return points
