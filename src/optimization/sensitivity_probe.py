"""
sensitivity_probe.py — 双维度变量敏感度探针（收敛 + 约束 margin）。

设计思路
--------
在 Phase 0 找到初始可行点（center）之后、Phase 1 DOE 开始之前，
对每个设计变量单独做 k 次随机扰动（其余变量固定在 center），
通过仿真成败和约束 margin 统计每个变量的两类敏感度：

  收敛敏感度（conv_sensitivity）
  --------------------------------
      conv_sensitivity_i = 1 - (收敛次数 / 总扰动次数)

  conv ≈ 0：该变量可大范围变动而不影响收敛。
  conv ≈ 1：该变量略微偏离 center 就导致 Aspen 不收敛。

  约束 margin 敏感度（margin_sensitivity）
  ----------------------------------------
  margin_j = actual_value_j - threshold_j  （>0 表示满足，= -ConstraintValue.value）

      margin_drop_i = mean over converged runs:
                          max(0, center_margin_j - perturbed_margin_j) / center_margin_j
                      averaged across all constraints j

  margin ≈ 0：该变量大范围变动不影响约束裕度。
  margin ≈ 1：该变量小幅偏离就让 margin 急剧收缩（可行域杠杆变量）。

  综合敏感度（sensitivity）
  ------------------------
  用于 DOE 半径计算和解冻排序：

      sensitivity_i = margin_weight * margin_sensitivity_i
                    + (1 - margin_weight) * conv_sensitivity_i

  margin_weight 默认 0.5。
  若 run_fn 返回的 margin dict 始终为空（无约束配置），则退化为纯收敛模式
  （margin_sensitivity 全为 0，与旧行为完全一致）。

run_fn 签名升级
--------------
旧：Callable[[dict[str, float]], bool]
新：Callable[[dict[str, float]], tuple[bool, dict[str, float]]]
    返回 (converged, {constraint_name: margin})
    margin = actual - threshold = -ConstraintValue.value
    若仿真不收敛，margin dict 可以为空 {}。

向后兼容
--------
若调用方传入旧式 run_fn（仅返回 bool），run_sensitivity_probe 会自动检测
并包装为新接口，保证现有单目标流程（optimize_case.py）无需修改。

基于敏感度的自适应 DOE 宽度
--------------------------
每个变量的局部搜索半径（占全局宽度的比例）：

    doe_radius_i = min_doe_radius + (1 - sensitivity_i) * (1 - min_doe_radius)

sensitivity=1 → doe_radius = min_doe_radius（最小）
sensitivity=0 → doe_radius = 1.0（全局范围）

强相关对检测
-----------
对变量对 (i, j)，计算联合失败率（基于收敛失败，与旧逻辑相同）：

    co_fail_rate_ij = P(i 失败 AND j 失败) / max(P(i 失败), P(j 失败))

三阶段解冻状态机
---------------
由 ThawScheduler 管理，供 Phase 2 BO 循环调用：

  阶段 1（FROZEN）：高敏感变量使用窄范围，低敏感变量全局。
  阶段 2（THAWING）：HV 停滞后触发，逐步扩大敏感变量范围。
  阶段 3（REFREEZE）：扩张后失败率升高，自动回收范围。
"""
from __future__ import annotations

import logging
import math
import random as _random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Union

_log = logging.getLogger(__name__)

# run_fn 可接受两种签名：
#   旧式（向后兼容）：dict -> bool
#   新式：dict -> tuple[bool, dict[str, float]]
_RunFnNew = Callable[[dict[str, float]], tuple[bool, dict[str, float]]]
_RunFnOld = Callable[[dict[str, float]], bool]
_RunFn = Union[_RunFnNew, _RunFnOld]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class SensitivityProbeConfig:
    """
    敏感度探针配置。

    Attributes
    ----------
    enabled:
        是否启用探针，默认 True。False 时跳过探针，全变量使用全局 bounds DOE。
    n_perturbations:
        每个变量的扰动次数，默认 3。每次只动一个变量，其余固定在 center。
        总额外仿真次数 ≈ n_vars × n_perturbations。
    perturbation_radius:
        扰动幅度，占全局范围的比例，默认 0.20（±20%）。
    min_doe_radius:
        高敏感变量（sensitivity=1）DOE 搜索半径下限，占全局宽度比例，默认 0.10。
    correlation_threshold:
        联合失败率超过此阈值视为强相关对，默认 0.70。
    margin_weight:
        综合敏感度中 margin_sensitivity 的权重，默认 0.5。
        0.0 → 纯收敛敏感度（旧行为）；1.0 → 纯约束 margin 敏感度。
        若无约束（margin dict 始终为空），此参数无实际影响。
    thaw_hv_stall_patience:
        Phase 2 中连续多少轮 HV 无改进后触发解冻（阶段 1→2），默认 10。
    thaw_step_radius:
        每次解冻时将敏感变量 DOE 半径扩大的步长（占全局宽度比例），默认 0.10。
    refreeze_fail_window:
        检测解冻后失败率时的滑动窗口大小（迭代次数），默认 5。
    refreeze_fail_threshold:
        窗口内失败率超过此值则触发回收（阶段 2→3/1），默认 0.60。
    tags:
        探针工况的额外标签，默认 ["sensitivity_probe"]。
    """
    enabled: bool = True
    n_perturbations: int = 3
    perturbation_radius: float = 0.20
    min_doe_radius: float = 0.10
    correlation_threshold: float = 0.70
    margin_weight: float = 0.5
    thaw_hv_stall_patience: int = 10
    thaw_step_radius: float = 0.10
    refreeze_fail_window: int = 5
    refreeze_fail_threshold: float = 0.60
    tags: list[str] = field(default_factory=lambda: ["sensitivity_probe"])

    def __post_init__(self) -> None:
        if not (0 < self.min_doe_radius <= 1.0):
            raise ValueError(
                f"min_doe_radius 必须在 (0, 1] 内，收到 {self.min_doe_radius}"
            )
        if not (0 < self.perturbation_radius <= 1.0):
            raise ValueError(
                f"perturbation_radius 必须在 (0, 1] 内，收到 {self.perturbation_radius}"
            )
        if not (0.0 <= self.margin_weight <= 1.0):
            raise ValueError(
                f"margin_weight 必须在 [0, 1] 内，收到 {self.margin_weight}"
            )


# ---------------------------------------------------------------------------
# 探针结果
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """
    run_sensitivity_probe() 的返回值。

    Attributes
    ----------
    sensitivity:
        各变量的综合敏感度，{变量路径: 0~1 的 float}。
        = margin_weight * margin_sensitivity + (1-margin_weight) * conv_sensitivity。
        用于 DOE 半径计算和解冻排序。
    conv_sensitivity:
        纯收敛敏感度，{变量路径: 0~1}。偏离 center 导致 Aspen 不收敛的概率。
    margin_sensitivity:
        约束 margin 敏感度，{变量路径: 0~1}。
        偏离 center 后所有约束 margin 平均相对下降量。
        若无约束（run_fn 从不返回 margin），全为 0。
    constraint_names:
        参与 margin 统计的约束名列表。无约束时为空列表。
    margin_matrix:
        {变量路径: {约束名: [各次扰动的 margin 值]}}。
        仅含收敛成功的扰动点，不收敛的点不入库。
        供调试和离线分析使用。
    correlated_pairs:
        强相关变量对列表，每项为 (path_i, path_j, co_fail_rate)。
        基于收敛失败率计算（与旧逻辑一致）。
    doe_radii:
        推荐的 DOE 搜索半径，{变量路径: 0~1 的 float}。
        已按 min_doe_radius 做了下限裁剪。
    n_probes_run:
        实际运行的探针仿真次数（含不收敛的点）。
    sensitivity_rank:
        变量按综合敏感度降序排列的路径列表（最敏感 → 最不敏感）。
    """
    sensitivity: dict[str, float]
    conv_sensitivity: dict[str, float]
    margin_sensitivity: dict[str, float]
    constraint_names: list[str]
    margin_matrix: dict[str, dict[str, list[float]]]
    correlated_pairs: list[tuple[str, str, float]]
    doe_radii: dict[str, float]
    n_probes_run: int
    sensitivity_rank: list[str]


# ---------------------------------------------------------------------------
# 核心探针函数
# ---------------------------------------------------------------------------

def run_sensitivity_probe(
    center: dict[str, float],
    bounds: list[tuple[float, float]],
    paths: list[str],
    config: SensitivityProbeConfig,
    run_fn: _RunFn,
    integer_indices: set[int] | None = None,
    rng: _random.Random | None = None,
    warmup_fn: Callable[[dict[str, float]], bool] | None = None,
    center_margins: dict[str, float] | None = None,
) -> SensitivityResult:
    """
    对每个设计变量执行单变量扰动探针，计算收敛敏感度和约束 margin 敏感度。

    Parameters
    ----------
    center:
        已知可行点的设计变量值，{路径: 值}。来自 Phase 0 找到的第一个可行点。
    bounds:
        各变量的全局搜索边界 [(lo, hi), ...]，顺序与 paths 一致。
    paths:
        设计变量路径列表，顺序与 bounds 一致。
    config:
        探针配置。
    run_fn:
        单次仿真调用。支持两种签名（自动检测）：
          新式：接受 {路径: 值}，返回 (converged: bool, margins: dict[str, float])
                margins = {约束名: actual - threshold}，>0 表示满足。
                不收敛时 margins 可以为空 {}。
          旧式（向后兼容）：接受 {路径: 值}，返回 bool。
        由调用方（optimize_pareto_case）注入，封装 run_case。
    integer_indices:
        整数维度的索引集合，探针时对这些维度 round 到整数。
    rng:
        随机数生成器，None 时使用默认 random。
    warmup_fn:
        热启动函数（可选）。每次扰动前调用，让 Aspen 从 center 的已收敛状态出发。
        None 时跳过热启动，直接用 run_fn 运行。
    center_margins:
        center 点的约束 margin 值，{约束名: margin}，用于计算相对 margin 下降。
        None 时从第一次收敛的扰动结果中推断（以所有收敛点 margin 最大值为参考）。
        显式传入可获得更准确的相对变化量。

    Returns
    -------
    SensitivityResult
    """
    if rng is None:
        rng = _random.Random()
    integer_indices = integer_indices or set()

    # 检测 run_fn 签名，统一包装为新式接口
    run_fn_new = _wrap_run_fn(run_fn)

    n_vars = len(paths)
    # fail_matrix[i][k] = True 表示第 i 个变量的第 k 次扰动不收敛
    fail_matrix: list[list[bool]] = [[] for _ in range(n_vars)]
    # margin_matrix[i] = {约束名: [各次收敛扰动的 margin 值]}
    margin_matrix_raw: list[dict[str, list[float]]] = [{} for _ in range(n_vars)]
    n_probes_run = 0

    _log.info(
        "敏感度探针开始：%d 个变量，每变量 %d 次扰动，扰动半径 %.0f%%，"
        "margin_weight=%.2f。%s",
        n_vars, config.n_perturbations, config.perturbation_radius * 100,
        config.margin_weight,
        " [热启动模式]" if warmup_fn else "",
    )

    for i, (path, (lo, hi)) in enumerate(zip(paths, bounds)):
        width = hi - lo
        center_val = center.get(path, (lo + hi) / 2.0)

        for k in range(config.n_perturbations):
            relative_scale = max(abs(center_val), width * 0.01)
            delta = (rng.random() * 2 - 1) * config.perturbation_radius * relative_scale
            perturbed_val = max(lo, min(hi, center_val + delta))
            if i in integer_indices:
                perturbed_val = float(
                    max(math.ceil(lo), min(math.floor(hi), round(perturbed_val)))
                )

            candidate = dict(center)
            candidate[path] = perturbed_val

            if warmup_fn is not None:
                try:
                    warmup_fn(center)
                except Exception as exc:
                    _log.warning(
                        "敏感度探针 [var=%s, k=%d]：warmup_fn 异常，跳过热启动。%s",
                        _short(path), k, exc,
                    )

            try:
                converged, margins = run_fn_new(candidate)
            except Exception as exc:
                _log.warning(
                    "敏感度探针 [var=%s, k=%d]：run_fn 异常，计为不收敛。%s",
                    _short(path), k, exc,
                )
                converged, margins = False, {}

            fail_matrix[i].append(not converged)
            if converged and margins:
                for cname, mval in margins.items():
                    margin_matrix_raw[i].setdefault(cname, []).append(mval)
            n_probes_run += 1

        fail_rate = sum(fail_matrix[i]) / max(len(fail_matrix[i]), 1)
        _log.debug(
            "  变量 %d/%d [%s]：扰动 %d 次，不收敛 %d 次，收敛失败率=%.2f",
            i + 1, n_vars, _short(path),
            config.n_perturbations, sum(fail_matrix[i]), fail_rate,
        )

    # ------------------------------------------------------------------
    # 收集所有约束名（取并集）
    # ------------------------------------------------------------------
    all_constraint_names: list[str] = []
    seen_cnames: set[str] = set()
    for mm in margin_matrix_raw:
        for cname in mm:
            if cname not in seen_cnames:
                all_constraint_names.append(cname)
                seen_cnames.add(cname)

    # ------------------------------------------------------------------
    # 推断 center_margins（若未显式提供）
    # 取所有收敛点中每个约束的最大 margin 作为参考（近似 center 处的值）
    # ------------------------------------------------------------------
    if center_margins is None:
        center_margins = {}
        for cname in all_constraint_names:
            all_vals = [
                v
                for mm in margin_matrix_raw
                for v in mm.get(cname, [])
            ]
            if all_vals:
                # 用所有收敛点的最大值作为 center margin 的保守估计
                center_margins[cname] = max(all_vals)

    # ------------------------------------------------------------------
    # 计算收敛敏感度
    # ------------------------------------------------------------------
    conv_sensitivity: dict[str, float] = {}
    for i, path in enumerate(paths):
        conv_sensitivity[path] = sum(fail_matrix[i]) / max(len(fail_matrix[i]), 1)

    # ------------------------------------------------------------------
    # 计算约束 margin 敏感度
    # margin_sensitivity_i = mean over constraints j of:
    #   mean over converged runs k of:
    #     max(0, center_margin_j - perturbed_margin_j) / max(center_margin_j, eps)
    # ------------------------------------------------------------------
    _EPS = 1e-9
    margin_sensitivity: dict[str, float] = {}
    for i, path in enumerate(paths):
        mm = margin_matrix_raw[i]
        if not mm or not all_constraint_names:
            margin_sensitivity[path] = 0.0
            continue

        per_constraint_drops: list[float] = []
        for cname in all_constraint_names:
            ref = center_margins.get(cname, 0.0)
            if ref <= _EPS:
                # center margin 本身接近 0（刚好在约束边界上），
                # 对此约束跳过相对计算以避免数值不稳定
                continue
            perturbed_vals = mm.get(cname, [])
            if not perturbed_vals:
                continue
            drops = [max(0.0, ref - v) / ref for v in perturbed_vals]
            per_constraint_drops.append(sum(drops) / len(drops))

        margin_sensitivity[path] = (
            sum(per_constraint_drops) / len(per_constraint_drops)
            if per_constraint_drops
            else 0.0
        )

    # ------------------------------------------------------------------
    # 综合敏感度
    # ------------------------------------------------------------------
    w = config.margin_weight
    sensitivity: dict[str, float] = {
        path: min(1.0, w * margin_sensitivity[path] + (1.0 - w) * conv_sensitivity[path])
        for path in paths
    }

    # ------------------------------------------------------------------
    # 强相关对（基于收敛失败，与旧逻辑一致）
    # ------------------------------------------------------------------
    correlated_pairs: list[tuple[str, str, float]] = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            fails_i = fail_matrix[i]
            fails_j = fail_matrix[j]
            n = min(len(fails_i), len(fails_j))
            if n == 0:
                continue
            co_fail = sum(1 for k in range(n) if fails_i[k] and fails_j[k])
            max_fail = max(sum(fails_i[:n]), sum(fails_j[:n]))
            if max_fail == 0:
                continue
            co_rate = co_fail / max_fail
            if co_rate >= config.correlation_threshold:
                correlated_pairs.append((paths[i], paths[j], co_rate))
                _log.info(
                    "  强相关对：[%s] ↔ [%s]，联合失败率=%.2f",
                    _short(paths[i]), _short(paths[j]), co_rate,
                )

    # ------------------------------------------------------------------
    # 计算推荐 DOE 半径
    # ------------------------------------------------------------------
    doe_radii: dict[str, float] = {}
    for path, s in sensitivity.items():
        radius = config.min_doe_radius + (1.0 - s) * (1.0 - config.min_doe_radius)
        doe_radii[path] = max(config.min_doe_radius, min(1.0, radius))

    sensitivity_rank = sorted(sensitivity, key=lambda p: sensitivity[p], reverse=True)

    # ------------------------------------------------------------------
    # 日志摘要
    # ------------------------------------------------------------------
    _log.info(
        "敏感度探针完成：运行 %d 次仿真，约束 %d 个（%s）。",
        n_probes_run, len(all_constraint_names), all_constraint_names,
    )
    _log.info(
        "  综合敏感度 top4：%s",
        [(_short(p), round(sensitivity[p], 2)) for p in sensitivity_rank[:4]],
    )
    if all_constraint_names:
        margin_top = sorted(
            margin_sensitivity, key=lambda p: margin_sensitivity[p], reverse=True
        )
        _log.info(
            "  margin 敏感度 top4：%s",
            [(_short(p), round(margin_sensitivity[p], 2)) for p in margin_top[:4]],
        )
    _log.info(
        "  conv 敏感度 top4：%s",
        [(_short(p), round(conv_sensitivity[p], 2))
         for p in sorted(conv_sensitivity, key=lambda p: conv_sensitivity[p], reverse=True)[:4]],
    )
    _log.info(
        "  DOE 半径（top4 敏感）：%s",
        [(_short(p), round(doe_radii[p], 2)) for p in sensitivity_rank[:4]],
    )
    if correlated_pairs:
        _log.info("  强相关对（共 %d 对）：%s", len(correlated_pairs), correlated_pairs)

    # 整理 margin_matrix 为路径键格式（供调试用）
    margin_matrix_out: dict[str, dict[str, list[float]]] = {
        paths[i]: margin_matrix_raw[i] for i in range(n_vars)
    }

    return SensitivityResult(
        sensitivity=sensitivity,
        conv_sensitivity=conv_sensitivity,
        margin_sensitivity=margin_sensitivity,
        constraint_names=all_constraint_names,
        margin_matrix=margin_matrix_out,
        correlated_pairs=correlated_pairs,
        doe_radii=doe_radii,
        n_probes_run=n_probes_run,
        sensitivity_rank=sensitivity_rank,
    )


# ---------------------------------------------------------------------------
# 向后兼容包装器
# ---------------------------------------------------------------------------

def _wrap_run_fn(run_fn: _RunFn) -> _RunFnNew:
    """
    将旧式 run_fn（返回 bool）包装为新式（返回 tuple[bool, dict]）。
    通过调用一次并检查返回类型来判断签名。
    """
    # 使用一个哨兵来缓存判断结果，避免重复检测
    _detected: list[str] = []  # ["new"] 或 ["old"]

    def _wrapped(candidate: dict[str, float]) -> tuple[bool, dict[str, float]]:
        result = run_fn(candidate)
        if not _detected:
            if isinstance(result, tuple) and len(result) == 2:
                _detected.append("new")
            else:
                _detected.append("old")
        if _detected[0] == "new":
            # 新式：已经是 (bool, dict)
            ok, margins = result  # type: ignore[misc]
            return bool(ok), dict(margins) if margins else {}
        else:
            # 旧式：只有 bool
            return bool(result), {}

    return _wrapped


# ---------------------------------------------------------------------------
# 自适应 DOE bounds 生成
# ---------------------------------------------------------------------------

def adaptive_doe_bounds(
    center: dict[str, float],
    global_bounds: list[tuple[float, float]],
    paths: list[str],
    probe_result: SensitivityResult,
) -> list[tuple[float, float]]:
    """
    根据敏感度探针结果，为每个变量计算 DOE 搜索边界。

    高敏感变量：局部范围（center ± doe_radius × width）
    低敏感变量：全局范围（接近原始 bounds）

    Returns
    -------
    list[tuple[float, float]]
        与 paths / global_bounds 等长的自适应边界列表。
    """
    adaptive: list[tuple[float, float]] = []
    for path, (glo, ghi) in zip(paths, global_bounds):
        width = ghi - glo
        radius = probe_result.doe_radii.get(path, 1.0)
        center_val = center.get(path, (glo + ghi) / 2.0)
        lo = max(glo, center_val - radius * width)
        hi = min(ghi, center_val + radius * width)
        if lo >= hi:
            lo, hi = glo, ghi
        adaptive.append((lo, hi))
    return adaptive


# ---------------------------------------------------------------------------
# 三阶段解冻状态机
# ---------------------------------------------------------------------------

class ThawStage(Enum):
    FROZEN   = "frozen"    # 阶段 1：敏感变量窄范围
    THAWING  = "thawing"   # 阶段 2：逐步扩大范围
    REFROZEN = "refrozen"  # 阶段 3：失败率升高，收回


class ThawScheduler:
    """
    管理 Phase 2 BO 循环中敏感变量的解冻/重冻调度。

    每次 BO 迭代结束后调用 step()，ThawScheduler 自动维护状态并
    更新各变量的当前有效 DOE 半径。

    Parameters
    ----------
    probe_result:
        run_sensitivity_probe() 的返回值。
    config:
        SensitivityProbeConfig 配置。
    global_bounds:
        各维度全局搜索边界。
    paths:
        设计变量路径列表，顺序与 global_bounds 一致。
    center:
        当前可行点中心（初始来自 Phase 0，后续随 BO 最优点移动）。
    """

    def __init__(
        self,
        probe_result: SensitivityResult,
        config: SensitivityProbeConfig,
        global_bounds: list[tuple[float, float]],
        paths: list[str],
        center: dict[str, float],
    ) -> None:
        self._probe = probe_result
        self._config = config
        self._global_bounds = global_bounds
        self._paths = paths
        self._center = dict(center)
        self._stage = ThawStage.FROZEN
        self._current_radii: dict[str, float] = dict(probe_result.doe_radii)
        self._hv_stall_count: int = 0
        self._recent_outcomes: list[bool] = []
        self._thaw_idx: int = 0

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def step(
        self,
        hv_improved: bool,
        case_success: bool,
        new_center: dict[str, float] | None = None,
    ) -> ThawStage:
        """
        每次 BO 迭代结束后调用，更新调度状态。

        Parameters
        ----------
        hv_improved:
            本次迭代 HV 是否有效改进（用于触发解冻）。
        case_success:
            本次仿真是否收敛成功（用于检测解冻后失败率）。
        new_center:
            若本次迭代找到更好的可行点，传入新中心点；否则为 None。

        Returns
        -------
        ThawStage
            更新后的阶段。
        """
        self._recent_outcomes.append(case_success)
        if len(self._recent_outcomes) > self._config.refreeze_fail_window:
            self._recent_outcomes.pop(0)

        if new_center is not None:
            self._center = dict(new_center)

        if self._stage == ThawStage.FROZEN:
            self._handle_frozen(hv_improved)
        elif self._stage == ThawStage.THAWING:
            self._handle_thawing()

        return self._stage

    def effective_bounds(self) -> list[tuple[float, float]]:
        """返回当前有效的 DOE 搜索边界（基于当前半径和中心）。"""
        result: list[tuple[float, float]] = []
        for path, (glo, ghi) in zip(self._paths, self._global_bounds):
            width = ghi - glo
            r = self._current_radii.get(path, 1.0)
            c = self._center.get(path, (glo + ghi) / 2.0)
            lo = max(glo, c - r * width)
            hi = min(ghi, c + r * width)
            if lo >= hi:
                lo, hi = glo, ghi
            result.append((lo, hi))
        return result

    def radii_summary(self) -> dict[str, float]:
        """返回当前各变量的有效半径字典（调试/日志用）。"""
        return dict(self._current_radii)

    @property
    def stage(self) -> ThawStage:
        return self._stage

    # ------------------------------------------------------------------
    # 内部状态机
    # ------------------------------------------------------------------

    def _handle_frozen(self, hv_improved: bool) -> None:
        if hv_improved:
            self._hv_stall_count = 0
        else:
            self._hv_stall_count += 1

        if self._hv_stall_count >= self._config.thaw_hv_stall_patience:
            self._hv_stall_count = 0
            self._stage = ThawStage.THAWING
            self._thaw_next_variable()
            _log.info(
                "ThawScheduler：HV 停滞 %d 轮，进入 THAWING 阶段。当前半径（高敏感）：%s",
                self._config.thaw_hv_stall_patience,
                {_short(k): round(v, 2) for k, v in self._current_radii.items()
                 if self._probe.sensitivity.get(k, 0) > 0.5},
            )

    def _handle_thawing(self) -> None:
        if len(self._recent_outcomes) < self._config.refreeze_fail_window:
            return

        fail_rate = 1.0 - sum(self._recent_outcomes) / len(self._recent_outcomes)
        if fail_rate > self._config.refreeze_fail_threshold:
            self._refreeze_last()
            self._stage = ThawStage.REFROZEN
            _log.info(
                "ThawScheduler：解冻后失败率 %.0f%% > 阈值 %.0f%%，触发 REFROZEN。",
                fail_rate * 100, self._config.refreeze_fail_threshold * 100,
            )
        elif self._thaw_idx < len(self._probe.sensitivity_rank):
            self._thaw_next_variable()

    def _thaw_next_variable(self) -> None:
        rank = self._probe.sensitivity_rank
        if self._thaw_idx >= len(rank):
            return
        path = rank[self._thaw_idx]
        old_r = self._current_radii.get(path, self._config.min_doe_radius)
        new_r = min(1.0, old_r + self._config.thaw_step_radius)
        self._current_radii[path] = new_r
        self._thaw_idx += 1
        _log.info(
            "ThawScheduler：解冻变量 [%s]（综合敏感度=%.2f，"
            "margin敏感度=%.2f，conv敏感度=%.2f）半径 %.2f → %.2f。",
            _short(path),
            self._probe.sensitivity.get(path, 0),
            self._probe.margin_sensitivity.get(path, 0),
            self._probe.conv_sensitivity.get(path, 0),
            old_r, new_r,
        )

    def _refreeze_last(self) -> None:
        if self._thaw_idx == 0:
            return
        self._thaw_idx -= 1
        path = self._probe.sensitivity_rank[self._thaw_idx]
        current_r = self._current_radii.get(path, self._config.min_doe_radius)
        restored_r = max(self._config.min_doe_radius, current_r - self._config.thaw_step_radius)
        self._current_radii[path] = restored_r
        _log.info(
            "ThawScheduler：重冻变量 [%s]，半径 %.2f → %.2f。",
            _short(path), current_r, restored_r,
        )


# ---------------------------------------------------------------------------
# 联动采样（强相关对）
# ---------------------------------------------------------------------------

def apply_correlated_sampling(
    point: list[float],
    paths: list[str],
    bounds: list[tuple[float, float]],
    correlated_pairs: list[tuple[str, str, float]],
    center: dict[str, float],
) -> list[float]:
    """
    对已生成的候选点，将强相关对中"从属变量"的偏移方向对齐到"主变量"。

    对每个强相关对 (path_i, path_j)：
      - 计算 path_i 的偏移方向（相对于 center）
      - 将 path_j 也调整为相同方向（保持 path_j 自己的偏移幅度不变）
      - 裁剪到 bounds

    Parameters
    ----------
    point:
        原始候选点（float 列表，顺序与 paths 一致）。
    paths, bounds, center:
        变量路径、边界、中心点。
    correlated_pairs:
        强相关对列表，来自 SensitivityResult.correlated_pairs。

    Returns
    -------
    list[float]
        调整后的候选点（副本）。
    """
    path_to_idx = {p: i for i, p in enumerate(paths)}
    result = list(point)

    for path_i, path_j, _ in correlated_pairs:
        idx_i = path_to_idx.get(path_i)
        idx_j = path_to_idx.get(path_j)
        if idx_i is None or idx_j is None:
            continue

        center_i = center.get(path_i, (bounds[idx_i][0] + bounds[idx_i][1]) / 2)
        center_j = center.get(path_j, (bounds[idx_j][0] + bounds[idx_j][1]) / 2)

        dir_i = math.copysign(1.0, result[idx_i] - center_i)
        offset_j = abs(result[idx_j] - center_j)
        new_j = center_j + dir_i * offset_j
        lo_j, hi_j = bounds[idx_j]
        result[idx_j] = max(lo_j, min(hi_j, new_j))

    return result


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _short(path: str) -> str:
    """截取路径中有区分度的片段用于日志显示。

    \\Data\\Blocks\\T1\\Input\\NSTAGE → T1/NSTAGE
    \\Data\\Streams\\S1\\Input\\TOTFLOW\\MIXED → S1/MIXED
    """
    parts = path.replace("\\", "/").split("/")
    _SKIP = {"Data", "Blocks", "Streams", "Input", "Output", "TOTFLOW", ""}
    meaningful = [p for p in parts if p not in _SKIP]
    if len(meaningful) >= 2:
        return "/".join(meaningful[-2:])
    if meaningful:
        return meaningful[-1]
    return path
