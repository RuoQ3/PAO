"""
pareto.py — Pareto 前沿算法层。

职责
----
接收 ProcessCase 列表，计算多目标优化的 Pareto 前沿，提供：

  - 支配关系判断（dominates / is_dominated）
  - 非支配排序（fast_non_dominated_sort，NSGA-II 算法）
  - 拥挤距离计算（crowding_distance）
  - 超体积指标（hypervolume，WFG 递归算法，支持 2~4 维）
  - ParetoFront 数据类（前沿工况、超体积、拥挤距离、排名）

设计约定
--------
- 所有目标统一转换为"最小化"方向（最大化目标取负值），
  算法内部只处理最小化问题，结果中保留原始值。
- 只有 case.success=True 的工况参与 Pareto 计算；
  失败/不可行工况记录在 ParetoResult.excluded_cases 中。
- 目标名称列表由调用方通过 objective_names 参数指定，
  顺序决定超体积参考点的维度顺序。
- 不依赖任何第三方库，仅使用标准库 math / itertools / dataclasses。

超体积算法
----------
使用 WFG（Walking Fish Group）递归算法：
  - 2D：O(N log N)，精确
  - 3D：O(N² log N)，精确
  - 4D：O(N³ log N)，精确但在 N>200 时较慢
  - >4D：给出 UserWarning，仍可计算但性能下降显著

参考文献
--------
Daulton S. et al., "Differentiable Expected Hypervolume Improvement", NeurIPS 2020.
Emmerich M. et al., "Single- and multiobjective evolutionary optimization
  assisted by Gaussian random field metamodels", IEEE TEVC 2006.
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Sequence

from ..models.process_case import ProcessCase

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部校验辅助
# ---------------------------------------------------------------------------

def _validate_vectors(vectors: list[list[float]], caller: str) -> None:
    """
    校验目标向量列表的一致性。

    - 所有向量维度相同
    - 维度非零
    - 所有值为有限数（非 NaN/Inf）

    Raises
    ------
    ValueError:
        维度为 0、各向量维度不一致、或存在非有限值。
    """
    if not vectors:
        return
    n_obj = len(vectors[0])
    if n_obj == 0:
        raise ValueError(f"{caller}: 目标向量维度不能为 0")
    for i, v in enumerate(vectors):
        if len(v) != n_obj:
            raise ValueError(
                f"{caller}: 向量 {i} 维度 {len(v)} 与第一个向量维度 {n_obj} 不一致"
            )
        for j, val in enumerate(v):
            if not math.isfinite(val):
                raise ValueError(
                    f"{caller}: 向量 {i} 第 {j} 维值 {val!r} 为非有限数（NaN/Inf），"
                    "请在调用前过滤非有限值"
                )


# ---------------------------------------------------------------------------
# 目标向量提取
# ---------------------------------------------------------------------------

def _extract_objectives(
    case: ProcessCase,
    objective_names: list[str],
) -> list[float] | None:
    """
    从 ProcessCase 提取指定目标的值向量（统一转为最小化方向）。

    返回 None 的情形：
    - 目标名称不存在于 case.objectives
    - ObjectiveValue.available 为 False（value=None 或有 error）
    - 目标值为 NaN 或 Inf（非有限数）

    最大化目标（minimize=False）取负值，使所有目标统一为最小化。
    """
    result: list[float] = []
    for name in objective_names:
        ov = case.get_objective(name)
        if ov is None or not ov.available:
            return None
        assert ov.value is not None
        val = ov.value if ov.minimize else -ov.value
        if not math.isfinite(val):
            _log.debug(
                "工况 %s 目标 '%s' 值为非有限数 %r，排除在 Pareto 计算之外",
                case.case_id[:8], name, val,
            )
            return None
        result.append(val)
    return result


# ---------------------------------------------------------------------------
# 支配关系
# ---------------------------------------------------------------------------

def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """
    判断目标向量 a 是否支配 b（均为最小化方向）。

    a 支配 b 当且仅当：
      - 所有目标 a[i] <= b[i]
      - 至少一个目标 a[i] < b[i]

    Raises
    ------
    ValueError:
        a 和 b 长度不一致、长度为 0、或含有非有限值（NaN/Inf）。
    """
    if len(a) != len(b):
        raise ValueError(f"目标向量维度不一致：len(a)={len(a)}, len(b)={len(b)}")
    if len(a) == 0:
        raise ValueError("目标向量维度不能为 0")
    for i, (ai, bi) in enumerate(zip(a, b)):
        if not math.isfinite(ai):
            raise ValueError(f"a[{i}]={ai!r} 为非有限数（NaN/Inf），拒绝支配判断")
        if not math.isfinite(bi):
            raise ValueError(f"b[{i}]={bi!r} 为非有限数（NaN/Inf），拒绝支配判断")
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            at_least_one_better = True
    return at_least_one_better


# ---------------------------------------------------------------------------
# 快速非支配排序（NSGA-II）
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(
    objective_vectors: list[list[float]],
) -> list[list[int]]:
    """
    快速非支配排序，返回各 Pareto 层的索引列表。

    Parameters
    ----------
    objective_vectors:
        N 个目标向量（均为最小化方向），每个向量长度相同。

    Returns
    -------
    fronts:
        fronts[0] 为第一 Pareto 前沿（非支配集）的索引列表，
        fronts[1] 为第二层，依此类推。
        索引对应 objective_vectors 的位置。

    Raises
    ------
    ValueError:
        向量列表为空、维度为 0、或各向量维度不一致。

    算法复杂度：O(M * N²)，M 为目标数，N 为解的数量。
    """
    n = len(objective_vectors)
    if n == 0:
        return []
    _validate_vectors(objective_vectors, "fast_non_dominated_sort")

    # dominated_by[i] = 支配 i 的解的数量
    dominated_by: list[int] = [0] * n
    # dominates_set[i] = i 支配的解的索引集合
    dominates_set: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objective_vectors[i], objective_vectors[j]):
                dominates_set[i].append(j)
            elif dominates(objective_vectors[j], objective_vectors[i]):
                dominated_by[i] += 1

    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if dominated_by[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominates_set[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


# ---------------------------------------------------------------------------
# 拥挤距离
# ---------------------------------------------------------------------------

def crowding_distance(
    objective_vectors: list[list[float]],
    indices: list[int],
) -> dict[int, float]:
    """
    计算指定索引集合中各解的拥挤距离。

    Parameters
    ----------
    objective_vectors:
        全部解的目标向量列表（最小化方向）。
    indices:
        需要计算拥挤距离的解的索引列表（通常为同一 Pareto 层）。

    Returns
    -------
    distances:
        {索引: 拥挤距离}，边界解的距离为 inf。

    Raises
    ------
    ValueError:
        objective_vectors 维度不一致或为 0。

    拥挤距离 = 各目标维度上相邻解之间归一化距离之和。
    边界解（每个目标维度上的最小/最大值）距离设为 inf，
    确保边界解在选择时优先保留。
    """
    if objective_vectors:
        _validate_vectors(objective_vectors, "crowding_distance")
    distances: dict[int, float] = {i: 0.0 for i in indices}
    if len(indices) <= 2:
        for i in indices:
            distances[i] = math.inf
        return distances

    n_obj = len(objective_vectors[0])

    for m in range(n_obj):
        # 按第 m 个目标排序
        sorted_idx = sorted(indices, key=lambda i: objective_vectors[i][m])
        f_min = objective_vectors[sorted_idx[0]][m]
        f_max = objective_vectors[sorted_idx[-1]][m]
        span = f_max - f_min

        # 边界解设为 inf
        distances[sorted_idx[0]] = math.inf
        distances[sorted_idx[-1]] = math.inf

        if span == 0.0:
            continue

        for k in range(1, len(sorted_idx) - 1):
            prev_val = objective_vectors[sorted_idx[k - 1]][m]
            next_val = objective_vectors[sorted_idx[k + 1]][m]
            distances[sorted_idx[k]] += (next_val - prev_val) / span

    return distances


# ---------------------------------------------------------------------------
# 超体积（WFG 递归算法）
# ---------------------------------------------------------------------------

def hypervolume(
    pareto_points: list[list[float]],
    reference_point: list[float],
) -> float:
    """
    计算 Pareto 前沿相对于参考点的超体积指标（最小化方向）。

    Parameters
    ----------
    pareto_points:
        Pareto 前沿的目标向量列表（最小化方向，已过滤非支配解）。
        每个向量的所有分量必须严格小于 reference_point 对应分量，
        否则该点对超体积无贡献（自动过滤）。
    reference_point:
        参考点，通常取各目标的最差可接受值（或观测最大值 × 1.1）。
        维度必须与 pareto_points 中向量的维度一致。

    Returns
    -------
    float:
        超体积值。空前沿或所有点均不优于参考点时返回 0.0。

    Notes
    -----
    使用 WFG 递归算法（Emmerich 2006）。
    维度 > 4 时发出 UserWarning，计算仍可进行但性能下降显著。
    """
    if not pareto_points:
        return 0.0

    n_obj = len(reference_point)
    if n_obj == 0:
        raise ValueError("reference_point 维度不能为 0")
    for i, v in enumerate(reference_point):
        if not math.isfinite(v):
            raise ValueError(
                f"reference_point[{i}]={v!r} 为非有限数（NaN/Inf），拒绝计算超体积"
            )
    _validate_vectors(pareto_points, "hypervolume")
    if len(pareto_points[0]) != n_obj:
        raise ValueError(
            f"pareto_points 维度 {len(pareto_points[0])} 与 reference_point 维度 {n_obj} 不一致"
        )

    if n_obj > 4:
        warnings.warn(
            f"hypervolume: 目标维度 {n_obj} > 4，WFG 算法性能下降显著（O(N^(d-1) log N)）。"
            "建议将目标数控制在 4 以内，或使用近似算法。",
            UserWarning,
            stacklevel=2,
        )

    # 过滤掉不优于参考点的点（任一维度 >= 参考点则无贡献）
    valid = [p for p in pareto_points if all(p[i] < reference_point[i] for i in range(n_obj))]
    if not valid:
        return 0.0

    return _wfg_hypervolume(valid, reference_point)


def _wfg_hypervolume(points: list[list[float]], ref: list[float]) -> float:
    """WFG 递归超体积计算（内部函数，假设所有点均优于 ref）。"""
    if not points:
        return 0.0

    n_obj = len(ref)

    # 1D 基础情形
    if n_obj == 1:
        best = min(p[0] for p in points)
        return ref[0] - best

    # 按最后一个目标排序（升序）
    points_sorted = sorted(points, key=lambda p: p[-1])

    hv = 0.0
    prev_last = ref[-1]

    for i in range(len(points_sorted) - 1, -1, -1):
        p = points_sorted[i]
        slice_height = prev_last - p[-1]
        if slice_height <= 0:
            continue

        # 投影到前 n_obj-1 维，计算该切片的超体积
        proj_points = [q[:-1] for q in points_sorted[i:] if q[-1] <= p[-1] or q is p]
        # 实际上取所有 last_dim <= p[-1] 的点的投影（包含 p 本身）
        proj_points = [q[:-1] for q in points_sorted if q[-1] <= p[-1]]
        proj_ref = ref[:-1]

        # 对投影点做非支配过滤（去掉被支配的点，减少递归规模）
        proj_nd = _filter_dominated(proj_points)

        hv += slice_height * _wfg_hypervolume(proj_nd, proj_ref)
        prev_last = p[-1]

    return hv


def _filter_dominated(points: list[list[float]]) -> list[list[float]]:
    """过滤掉被支配的点，返回非支配集（最小化方向）。"""
    if len(points) <= 1:
        return list(points)
    result: list[list[float]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and dominates(q, p):
                dominated = True
                break
        if not dominated:
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# 参考点自动推断
# ---------------------------------------------------------------------------

def infer_reference_point(
    objective_vectors: list[list[float]],
    margin: float = 0.1,
) -> list[float]:
    """
    从目标向量集合自动推断超体积参考点。

    参考点 = 各维度最大值 × (1 + margin)（最小化方向）。
    margin 默认 0.1（10%），确保参考点严格劣于所有 Pareto 点。

    Parameters
    ----------
    objective_vectors:
        目标向量列表（最小化方向）。
    margin:
        各维度最大值的扩展比例，默认 0.1。

    Returns
    -------
    list[float]:
        推断的参考点。
    """
    if not objective_vectors:
        raise ValueError("objective_vectors 为空，无法推断参考点")
    _validate_vectors(objective_vectors, "infer_reference_point")
    n_obj = len(objective_vectors[0])
    ref = []
    for m in range(n_obj):
        max_val = max(v[m] for v in objective_vectors)
        # 处理负值（最大化目标取负后可能为负数）
        if max_val >= 0:
            ref.append(max_val * (1.0 + margin))
        else:
            ref.append(max_val * (1.0 - margin))
    return ref


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class ParetoFront:
    """
    单层 Pareto 前沿的计算结果。

    Attributes
    ----------
    cases:
        该层前沿的 ProcessCase 列表，按拥挤距离降序排列（多样性优先）。
    objective_names:
        目标函数名称列表，与 objective_vectors 的列顺序对应。
    objective_vectors:
        各工况的目标向量（原始值，未取负），与 cases 一一对应。
    crowding_distances:
        各工况的拥挤距离，与 cases 一一对应。
    rank:
        该前沿的 Pareto 层级（0 = 第一前沿，即非支配集）。
    hypervolume:
        超体积指标（仅第一前沿计算，其他层为 None）。
    reference_point:
        计算超体积时使用的参考点（原始值，未取负）。
        None 表示未计算超体积。
    """
    cases: list[ProcessCase]
    objective_names: list[str]
    objective_vectors: list[list[float]]    # 原始值（未取负）
    crowding_distances: list[float]
    rank: int
    hypervolume: float | None = None
    reference_point: list[float] | None = None


@dataclass
class ParetoResult:
    """
    compute_pareto() 的完整返回值。

    Attributes
    ----------
    fronts:
        所有 Pareto 层，fronts[0] 为第一前沿（非支配集）。
    excluded_cases:
        未参与计算的工况（success=False 或目标不可用）。
    objective_names:
        目标函数名称列表。
    n_evaluated:
        参与计算的工况总数（success=True 且目标可用）。
    hypervolume:
        第一前沿的超体积指标；未计算时为 None。
    reference_point:
        超体积计算使用的参考点（原始值）；未计算时为 None。
    """
    fronts: list[ParetoFront]
    excluded_cases: list[ProcessCase]
    objective_names: list[str]
    n_evaluated: int
    hypervolume: float | None = None
    reference_point: list[float] | None = None

    @property
    def first_front(self) -> ParetoFront | None:
        """第一 Pareto 前沿（非支配集），无有效工况时为 None。"""
        return self.fronts[0] if self.fronts else None

    @property
    def n_fronts(self) -> int:
        """Pareto 层数。"""
        return len(self.fronts)

    def summary(self) -> dict:
        """返回可序列化的摘要字典，供日志和数据库记录。"""
        return {
            "n_evaluated": self.n_evaluated,
            "n_excluded": len(self.excluded_cases),
            "n_fronts": self.n_fronts,
            "front_sizes": [len(f.cases) for f in self.fronts],
            "objective_names": self.objective_names,
            "hypervolume": self.hypervolume,
            "reference_point": self.reference_point,
        }


# ---------------------------------------------------------------------------
# 主计算函数
# ---------------------------------------------------------------------------

def compute_pareto(
    cases: list[ProcessCase],
    objective_names: list[str],
    reference_point: list[float] | None = None,
    compute_hv: bool = True,
    hv_margin: float = 0.1,
    include_infeasible: bool = False,
) -> ParetoResult:
    """
    计算 ProcessCase 列表的 Pareto 前沿。

    Parameters
    ----------
    cases:
        工况列表，通常来自 optimize_case() 或 param_scan() 的结果。
    objective_names:
        参与 Pareto 计算的目标函数名称列表（顺序决定超体积维度顺序）。
        名称须与 ProcessCase.objectives 中的 ObjectiveValue.name 一致。
    reference_point:
        超体积计算的参考点（原始值，与目标方向一致）。
        None 时自动从数据推断（各维度最大值 × (1 + hv_margin)）。
    compute_hv:
        True（默认）：计算第一前沿的超体积指标。
        False：跳过超体积计算（节省时间，适用于大规模数据）。
    hv_margin:
        自动推断参考点时各维度的扩展比例，默认 0.1。
    include_infeasible:
        False（默认）：只有 case.success=True 的工况参与计算。
          SIM_FAILED / INFEASIBLE / OBJECTIVE_ERROR / CONSTRAINT_ERROR
          均进入 excluded_cases，不污染 Pareto 前沿。
        True：允许 INFEASIBLE 工况（约束违反但目标可用）参与计算。
          用于不可行解诊断或约束松弛分析，不建议用于正式优化结果。
          SIM_FAILED / OBJECTIVE_ERROR 等目标不可用的工况仍被排除。

    Returns
    -------
    ParetoResult:
        包含所有 Pareto 层、超体积、拥挤距离等信息。

    Raises
    ------
    ValueError:
        objective_names 为空。
    """
    if not objective_names:
        raise ValueError("objective_names 不能为空")

    # 分离有效工况和排除工况
    valid_cases: list[ProcessCase] = []
    valid_vecs: list[list[float]] = []   # 最小化方向
    excluded: list[ProcessCase] = []

    for case in cases:
        # P0-1：先检查工况状态，失败/不可行工况默认排除
        if not case.success:
            if include_infeasible and case.status.value == "infeasible":
                pass  # 允许 INFEASIBLE 继续走目标提取
            else:
                excluded.append(case)
                _log.debug(
                    "工况 %s 状态 %s，排除在 Pareto 计算之外",
                    case.case_id[:8], case.status.value,
                )
                continue

        # P0-2：提取目标向量（含 NaN/Inf 过滤）
        vec = _extract_objectives(case, objective_names)
        if vec is None:
            excluded.append(case)
            _log.debug("工况 %s 目标不可用或含非有限值，排除在 Pareto 计算之外", case.case_id[:8])
        else:
            valid_cases.append(case)
            valid_vecs.append(vec)

    if not valid_cases:
        _log.warning("没有有效工况参与 Pareto 计算（共 %d 个工况全部被排除）", len(cases))
        return ParetoResult(
            fronts=[],
            excluded_cases=excluded,
            objective_names=objective_names,
            n_evaluated=0,
        )

    # 非支配排序
    front_indices = fast_non_dominated_sort(valid_vecs)

    # 构建各层 ParetoFront
    fronts: list[ParetoFront] = []
    for rank, idx_list in enumerate(front_indices):
        cd = crowding_distance(valid_vecs, idx_list)

        # 按拥挤距离降序排列（多样性优先）
        sorted_idx = sorted(idx_list, key=lambda i: cd[i], reverse=True)

        front_cases = [valid_cases[i] for i in sorted_idx]
        # 原始值（还原最大化目标的符号）
        front_vecs_raw = [_restore_objectives(valid_vecs[i], cases=valid_cases[i], names=objective_names)
                          for i in sorted_idx]
        front_cd = [cd[i] if not math.isinf(cd[i]) else float("inf") for i in sorted_idx]

        fronts.append(ParetoFront(
            cases=front_cases,
            objective_names=objective_names,
            objective_vectors=front_vecs_raw,
            crowding_distances=front_cd,
            rank=rank,
        ))

    # 计算第一前沿的超体积
    hv_value: float | None = None
    ref_point_raw: list[float] | None = None

    if compute_hv and fronts:
        first_vecs_min = [valid_vecs[i] for i in front_indices[0]]

        if reference_point is not None:
            n_obj = len(objective_names)
            if len(reference_point) != n_obj:
                raise ValueError(
                    f"reference_point 维度 {len(reference_point)} 与目标数 {n_obj} 不一致"
                )
            for i, v in enumerate(reference_point):
                if not math.isfinite(v):
                    raise ValueError(
                        f"reference_point[{i}]={v!r} 为非有限数（NaN/Inf），拒绝计算超体积"
                    )
            ref_min = _reference_point_to_min(reference_point, valid_cases[0], objective_names)
        else:
            ref_min = infer_reference_point(first_vecs_min, margin=hv_margin)

        hv_value = hypervolume(first_vecs_min, ref_min)
        # 还原参考点到原始方向
        ref_point_raw = _restore_reference_point(ref_min, valid_cases[0], objective_names)

        fronts[0] = ParetoFront(
            cases=fronts[0].cases,
            objective_names=fronts[0].objective_names,
            objective_vectors=fronts[0].objective_vectors,
            crowding_distances=fronts[0].crowding_distances,
            rank=0,
            hypervolume=hv_value,
            reference_point=ref_point_raw,
        )

    _log.info(
        "Pareto 计算完成：%d 个有效工况，%d 层前沿，第一前沿 %d 个解，HV=%s",
        len(valid_cases),
        len(fronts),
        len(fronts[0].cases) if fronts else 0,
        f"{hv_value:.4g}" if hv_value is not None else "未计算",
    )

    return ParetoResult(
        fronts=fronts,
        excluded_cases=excluded,
        objective_names=objective_names,
        n_evaluated=len(valid_cases),
        hypervolume=hv_value,
        reference_point=ref_point_raw,
    )


# ---------------------------------------------------------------------------
# 内部辅助：原始值还原
# ---------------------------------------------------------------------------

def _restore_objectives(
    vec_min: list[float],
    cases: ProcessCase,
    names: list[str],
) -> list[float]:
    """将最小化方向的目标向量还原为原始值（最大化目标取负还原）。"""
    result = []
    for i, name in enumerate(names):
        ov = cases.get_objective(name)
        if ov is not None and not ov.minimize:
            result.append(-vec_min[i])
        else:
            result.append(vec_min[i])
    return result


def _reference_point_to_min(
    ref_raw: list[float],
    sample_case: ProcessCase,
    names: list[str],
) -> list[float]:
    """将原始方向的参考点转换为最小化方向。"""
    result = []
    for i, name in enumerate(names):
        ov = sample_case.get_objective(name)
        if ov is not None and not ov.minimize:
            result.append(-ref_raw[i])
        else:
            result.append(ref_raw[i])
    return result


def _restore_reference_point(
    ref_min: list[float],
    sample_case: ProcessCase,
    names: list[str],
) -> list[float]:
    """将最小化方向的参考点还原为原始值。"""
    return _restore_objectives(ref_min, sample_case, names)
