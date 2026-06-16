"""
trust_region.py — FuRBO 信任域实现（纯 Python，不依赖 BoTorch）。

设计
----
信任域以 L∞ 范数定义一个以当前最优点为中心的超立方体，
各维度宽度 = radius × (全局上界 - 全局下界)，裁剪到全局 bounds 内。

半径更新规则（简化 TuRBO/FuRBO）：
  - 超体积（HV）相对改进 > eta_success  → 扩张 × gamma_expand
  - HV 无改进                           → 收缩 × gamma_shrink，失败计数 +1
  - 连续失败 >= failure_tolerance        → 重置到初始半径

参考文献
--------
Eriksson D. et al., "Scalable Global Optimization via Local Bayesian Optimization",
NeurIPS 2019. (TuRBO)

Daulton S. et al., "Feasibility-Driven Trust Region Bayesian Optimization",
AutoML 2025 Methods Track. (FuRBO)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)


@dataclass
class TrustRegionConfig:
    """
    信任域配置参数。

    Attributes
    ----------
    initial_radius:
        初始归一化半径（0~1），0.5 表示以最优点为中心 ±50% 搜索空间宽度。
    min_radius:
        最小半径，防止信任域退化为单点。
    max_radius:
        最大半径，通常取 1.0（等于全局搜索空间）。
    gamma_expand:
        HV 改进时的扩张倍数，默认 1.5。
    gamma_shrink:
        无 HV 改进时的收缩倍数，默认 0.7。
    eta_success:
        HV 相对改进比阈值，超过此值视为"成功"，默认 0.05（5%）。
    failure_tolerance:
        连续失败次数达到此值后重置信任域到初始半径，默认 5。
    """
    initial_radius: float = 0.5
    min_radius: float = 0.05
    max_radius: float = 1.0
    gamma_expand: float = 1.5
    gamma_shrink: float = 0.7
    eta_success: float = 0.05
    failure_tolerance: int = 5

    def __post_init__(self) -> None:
        if not (0 < self.min_radius <= self.initial_radius <= self.max_radius <= 1.0):
            raise ValueError(
                f"要求 0 < min_radius <= initial_radius <= max_radius <= 1.0，"
                f"收到 min={self.min_radius}, init={self.initial_radius}, max={self.max_radius}"
            )
        if self.gamma_expand <= 1.0:
            raise ValueError(f"gamma_expand 必须 > 1.0，收到 {self.gamma_expand}")
        if not (0 < self.gamma_shrink < 1.0):
            raise ValueError(f"gamma_shrink 必须在 (0, 1) 内，收到 {self.gamma_shrink}")


class TrustRegion:
    """
    L∞ 信任域（超立方体约束）。

    以当前最优点 center 为中心，radius 控制各维度的搜索范围：
      local_lo_i = max(global_lo_i, center_i - radius × width_i)
      local_hi_i = min(global_hi_i, center_i + radius × width_i)

    其中 width_i = global_hi_i - global_lo_i。

    Parameters
    ----------
    center:
        当前最优点（设计变量向量），与 global_bounds 维度相同。
    radius:
        初始归一化半径（从 TrustRegionConfig.initial_radius 取）。
    global_bounds:
        各维度的全局搜索边界 [(lo, hi), ...]。
    """

    def __init__(
        self,
        center: list[float],
        radius: float,
        global_bounds: list[tuple[float, float]],
    ) -> None:
        if len(center) != len(global_bounds):
            raise ValueError(
                f"center 维度 {len(center)} 与 global_bounds 维度 {len(global_bounds)} 不一致"
            )
        self.center: list[float] = list(center)
        self.radius: float = float(radius)
        self.global_bounds: list[tuple[float, float]] = global_bounds
        self._fail_count: int = 0
        self._initial_radius: float = float(radius)

    # ------------------------------------------------------------------
    # 主要接口
    # ------------------------------------------------------------------

    def compute_local_bounds(self) -> list[tuple[float, float]]:
        """
        计算当前信任域对应的局部搜索边界，裁剪到全局 bounds。

        Returns
        -------
        list[tuple[float, float]]
            各维度的局部边界 [(lo, hi), ...]，维度与 global_bounds 相同。
        """
        local: list[tuple[float, float]] = []
        for (glo, ghi), c in zip(self.global_bounds, self.center):
            width = ghi - glo
            lo = max(glo, c - self.radius * width)
            hi = min(ghi, c + self.radius * width)
            # 退化保护：若裁剪后 lo >= hi，回退到全局 bounds
            if lo >= hi:
                lo, hi = glo, ghi
            local.append((lo, hi))
        return local

    def update(
        self,
        hv_prev: float | None,
        hv_current: float | None,
        config: TrustRegionConfig,
    ) -> str:
        """
        根据 HV 变化更新信任域半径。

        Parameters
        ----------
        hv_prev:
            上一次记录的超体积值，None 表示尚无有效 HV。
        hv_current:
            本次迭代后的超体积值，None 表示本次未能计算 HV。
        config:
            信任域配置参数。

        Returns
        -------
        str
            执行的动作："expand" / "shrink" / "reset" / "skip"。
        """
        if hv_prev is None or hv_current is None:
            return "skip"

        denom = max(abs(hv_prev), 1e-10)
        rel_improvement = (hv_current - hv_prev) / denom

        if rel_improvement > config.eta_success:
            self._fail_count = 0
            self.radius = min(config.max_radius, self.radius * config.gamma_expand)
            _log.debug("TrustRegion expand: r=%.4f (HV +%.2f%%)", self.radius, rel_improvement * 100)
            return "expand"
        else:
            self._fail_count += 1
            if self._fail_count >= config.failure_tolerance:
                self._fail_count = 0
                self.radius = self._initial_radius
                _log.info(
                    "TrustRegion reset: 连续 %d 次无 HV 改进，半径重置至 %.4f",
                    config.failure_tolerance, self.radius,
                )
                return "reset"
            self.radius = max(config.min_radius, self.radius * config.gamma_shrink)
            _log.debug(
                "TrustRegion shrink: r=%.4f (fail_count=%d, HV %.2f%%)",
                self.radius, self._fail_count, rel_improvement * 100,
            )
            return "shrink"

    def move_center(self, new_center: list[float]) -> None:
        """
        将信任域中心移动到新的最优点。

        通常在找到新的可行且改进的点后调用。
        不重置半径——半径由 update() 单独管理。
        """
        if len(new_center) != len(self.center):
            raise ValueError(
                f"new_center 维度 {len(new_center)} 与当前 center 维度 {len(self.center)} 不一致"
            )
        self.center = list(new_center)
        _log.debug("TrustRegion move_center: 已更新中心")

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def fail_count(self) -> int:
        """当前连续失败计数。"""
        return self._fail_count

    def summary(self) -> dict:
        """返回可用于日志记录的摘要字典。"""
        return {
            "radius": round(self.radius, 6),
            "fail_count": self._fail_count,
            "center_first4": [round(v, 4) for v in self.center[:4]],
        }
