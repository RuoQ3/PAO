"""
surrogate.py — 可配置代理模型层。

支持 GP / RF / ET / GBRT / random 五种代理模型，
统一通过 SurrogateOptimizer 接口暴露 tell / ask。

  model == "random"
      始终随机采样，不依赖 skopt。

  model in {"GP", "RF", "ET", "GBRT"}
      使用 skopt.Optimizer(base_estimator=model)。
      skopt 不可用、初始化失败、ask/tell 失败、
      或成功样本数 < n_initial_min 时，回退随机采样并写 WARNING 日志。
"""
from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass
from typing import Any, Literal

_log = logging.getLogger(__name__)

try:
    from skopt import Optimizer as _SkoptOptimizer
    from skopt.space import Integer as _Integer
    from skopt.space import Real as _Real
    _HAS_SKOPT = True
except ImportError:
    _SkoptOptimizer = None  # type: ignore[assignment,misc]
    _Real = None            # type: ignore[assignment,misc]
    _Integer = None         # type: ignore[assignment,misc]
    _HAS_SKOPT = False


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

_VALID_MODELS = {"GP", "RF", "ET", "GBRT", "random"}


@dataclass
class SurrogateConfig:
    """
    代理模型配置。

    Attributes
    ----------
    model:
        代理模型类型："GP"（默认）、"RF"、"ET"、"GBRT"、"random"。
        "random" 不依赖 skopt，始终随机采样。
    acquisition:
        采集函数："EI"（默认）、"UCB"、"PI"。
    xi:
        EI/PI 探索参数，默认 0.01。
    kappa:
        UCB 探索参数，默认 1.96。
    n_initial_min:
        启用代理模型所需的最少成功样本数，默认 3。
        不足时 ask() 退化为随机采样。
    random_seed:
        随机种子，用于随机采样和 skopt 的可重复性。
    """
    model: Literal["GP", "RF", "ET", "GBRT", "random"] = "GP"
    acquisition: Literal["EI", "UCB", "PI"] = "EI"
    xi: float = 0.01
    kappa: float = 1.96
    n_initial_min: int = 3
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.model not in _VALID_MODELS:
            raise ValueError(
                f"SurrogateConfig.model={self.model!r} 不合法，"
                f"支持值：{sorted(_VALID_MODELS)}。"
            )


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------

class SurrogateOptimizer:
    """
    代理模型优化器，支持 GP / RF / ET / GBRT / random。

    tell() 提交观测，ask() 推荐下一个候选点。

    支持混合整数搜索空间：integer_indices 中的维度使用 skopt.space.Integer，
    采集函数直接在整数格点上采样；回退随机采样时同样对整数维度取整。

    回退规则（按优先级）：
      1. model == "random"：始终随机采样。
      2. skopt 不可用：随机采样（初始化时写 WARNING）。
      3. skopt 初始化失败：随机采样（写 WARNING）。
      4. 成功样本数 < n_initial_min：随机采样。
      5. skopt.ask() 失败：随机采样（写 WARNING）。
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        config: SurrogateConfig,
        integer_indices: set[int] | None = None,
    ) -> None:
        self._bounds = bounds
        self._integer_indices: set[int] = integer_indices or set()
        self._n_initial_min = config.n_initial_min
        self._rng = _random.Random(config.random_seed)
        self._n_success = 0
        self._skopt: Any = None
        self._use_random = (config.model == "random")

        if not self._use_random:
            if _HAS_SKOPT:
                # skopt 使用 LCB（Lower Confidence Bound）表示置信界采集函数；
                # 对外 API 保留 UCB 名称，内部映射为 LCB。
                skopt_acq = "LCB" if config.acquisition == "UCB" else config.acquisition
                acq_kwargs: dict[str, Any] = {}
                if config.acquisition in ("EI", "PI"):
                    acq_kwargs["xi"] = config.xi
                elif config.acquisition == "UCB":
                    acq_kwargs["kappa"] = config.kappa

                # dimensions 构造与 Optimizer 初始化放在同一个 try 块：
                # _Real/_Integer 为 None（skopt 未正确安装）或 Optimizer 初始化失败时，
                # 统一回退到随机采样，不向上抛出异常。
                try:
                    if not callable(_Real):
                        raise RuntimeError(
                            "skopt.space.Real 不可调用，skopt 安装可能不完整。"
                        )
                    # 仅当存在整数维度时才依赖 _Integer；纯连续空间不检查，
                    # 避免只 mock _Real 的测试场景被意外 fallback。
                    if self._integer_indices and not callable(_Integer):
                        raise RuntimeError(
                            "搜索空间含整数维度，但 skopt.space.Integer 不可调用，"
                            "skopt 安装可能不完整。"
                        )

                    # 整数维度使用 skopt.space.Integer，连续维度使用 Real。
                    # integer 变量的 lo/hi 在 _validate_config 中已保证为整数值，
                    # int() 转换安全，不会出现 int(1.2)=1 低于原始下界的问题。
                    # Integer 维度的采集函数直接在整数格点上采样，
                    # 避免连续松弛带来的代理模型偏差。
                    dimensions = []
                    for i, (lo, hi) in enumerate(bounds):
                        if i in self._integer_indices:
                            dimensions.append(_Integer(int(lo), int(hi)))
                        else:
                            dimensions.append(_Real(lo, hi))

                    if self._integer_indices:
                        _log.debug(
                            "搜索空间：%d 个维度，其中 %d 个整数维度（索引 %s）。",
                            len(bounds), len(self._integer_indices),
                            sorted(self._integer_indices),
                        )

                    self._skopt = _SkoptOptimizer(
                        dimensions=dimensions,
                        base_estimator=config.model,
                        acq_func=skopt_acq,
                        acq_func_kwargs=acq_kwargs,
                        random_state=config.random_seed,
                        n_initial_points=0,
                    )
                except Exception as exc:
                    _log.warning(
                        "skopt Optimizer(base_estimator=%r) 初始化失败，回退到随机采样：%s",
                        config.model, exc,
                    )
            else:
                _log.warning(
                    "scikit-optimize 未安装，代理模型 %r 将退化为随机采样。"
                    "安装方法：pip install scikit-optimize",
                    config.model,
                )

    def tell(self, x: list[float], y: float, *, is_success: bool) -> None:
        """提交一次观测。is_success=True 时计入成功样本数。"""
        if is_success:
            self._n_success += 1
        if self._skopt is not None:
            try:
                self._skopt.tell(x, y)
            except Exception as exc:
                _log.warning("skopt.tell() 失败（已忽略）：%s", exc)

    def ask(self) -> list[float]:
        """
        推荐下一个候选点。

        成功观测数 < n_initial_min，或代理模型不可用时，返回随机点。
        整数维度由 skopt.Integer 或 _random_point() 保证返回整数值。
        """
        if (
            not self._use_random
            and self._skopt is not None
            and self._n_success >= self._n_initial_min
        ):
            try:
                return self._skopt.ask()
            except Exception as exc:
                _log.warning("skopt.ask() 失败，回退到随机采样：%s", exc)
        return self._random_point()

    def _random_point(self) -> list[float]:
        """在各维度边界内均匀采样随机点，整数维度严格返回 [ceil(lo), floor(hi)] 内的整数。"""
        import math
        point = []
        for i, (lo, hi) in enumerate(self._bounds):
            if i in self._integer_indices:
                int_lo = math.ceil(lo)
                int_hi = math.floor(hi)
                # 若 ceil(lo) > floor(hi)，说明边界区间内无整数，退化到 int_lo
                val = float(self._rng.randint(int_lo, max(int_lo, int_hi)))
            else:
                val = lo + self._rng.random() * (hi - lo)
            point.append(val)
        return point


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def make_surrogate_optimizer(
    bounds: list[tuple[float, float]],
    config: SurrogateConfig,
    integer_indices: set[int] | None = None,
) -> SurrogateOptimizer:
    """
    创建 SurrogateOptimizer 实例。

    Parameters
    ----------
    bounds:
        各维度的 (下界, 上界) 列表。
    config:
        代理模型配置，见 SurrogateConfig。
    integer_indices:
        整数维度的索引集合（对应 bounds 的下标）。
        这些维度使用 skopt.space.Integer，采集函数在整数格点上采样。
        None 或空集合表示全部为连续维度，行为与旧版本一致。

    Returns
    -------
    SurrogateOptimizer
    """
    return SurrogateOptimizer(bounds, config, integer_indices)
