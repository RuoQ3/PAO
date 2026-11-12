"""
surrogate.py — 可配置代理模型层。

支持 GP / RF / ET / GBRT / random 五种代理模型，
统一通过 SurrogateOptimizer 接口暴露 tell / ask。

  model == "random"
      始终随机采样，不依赖 skopt。

  model in {"GP", "RF", "ET", "GBRT"}
      使用 skopt.Optimizer(base_estimator=model)。
      skopt 不可用或初始化、ask/tell 失败时抛出可定位错误。
      成功样本数 < n_initial_min 的随机初始化是算法的一部分。
"""
from __future__ import annotations

import logging
import math
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

_VALID_MODELS = {"GP", "RF", "ET", "GBRT", "random", "qEHVI", "NEHVI"}


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
        if self.acquisition not in {"EI", "UCB", "PI"}:
            raise ValueError("unsupported acquisition function")
        if type(self.n_initial_min) is not int or self.n_initial_min < 1:
            raise ValueError("n_initial_min must be a positive integer")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (self.xi, self.kappa)):
            raise ValueError("xi/kappa must be finite and nonnegative")
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

    显式 random 和 BO 初始采样允许随机点。后端故障必须停止。
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        config: SurrogateConfig,
        integer_indices: set[int] | None = None,
    ) -> None:
        self._bounds = bounds
        self._config = config
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
                # 初始化故障必须传递给调用者，不能替换算法。
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
                    raise RuntimeError(f"backend initialization failed for {config.model}: {exc}") from exc
            else:
                raise RuntimeError(f"backend unavailable for {config.model}: install scikit-optimize")

    def tell(self, x: list[float], y: float, *, is_success: bool) -> None:
        """提交一次观测。is_success=True 时计入成功样本数。"""
        if is_success:
            self._n_success += 1
        if self._skopt is not None:
            try:
                self._skopt.tell(x, y)
            except Exception as exc:
                raise RuntimeError(f"{self._config.model} tell failed: {exc}") from exc

    def ask(self, active_bounds: list[tuple[float, float]] | None = None) -> list[float]:
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
                if active_bounds is not None:
                    # Train in immutable global coordinates; score only local candidates.
                    # Creating a new skopt space from local bounds rejects older observations.
                    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
                    pool = [self._random_point(active_bounds) for _ in range(256)]
                    X = self._skopt.space.transform(pool)
                    model = self._skopt.models[-1]
                    if self._config.acquisition == "UCB":
                        score = gaussian_lcb(X, model, kappa=self._config.kappa)
                    elif self._config.acquisition == "PI":
                        score = -gaussian_pi(X, model, y_opt=min(self._skopt.yi), xi=self._config.xi)
                    else:
                        score = -gaussian_ei(X, model, y_opt=min(self._skopt.yi), xi=self._config.xi)
                    return pool[int(score.argmin())]
                return self._skopt.ask()
            except Exception as exc:
                raise RuntimeError(f"{self._config.model} ask failed: {exc}") from exc
        return self._random_point(active_bounds)

    def _random_point(self, bounds=None) -> list[float]:
        """在各维度边界内均匀采样随机点，整数维度严格返回 [ceil(lo), floor(hi)] 内的整数。"""
        import math
        point = []
        for i, (lo, hi) in enumerate(bounds if bounds is not None else self._bounds):
            if i in self._integer_indices:
                int_lo = math.ceil(lo)
                int_hi = math.floor(hi)
                # 若 ceil(lo) > floor(hi)，说明边界区间内无整数，退化到 int_lo
                if int_lo > int_hi:
                    raise ValueError("integer search interval is empty")
                val = float(self._rng.randint(int_lo, int_hi))
            else:
                val = lo + self._rng.random() * (hi - lo)
            point.append(val)
        return point


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def make_surrogate_optimizer(
    bounds: list[tuple[float, float]],
    config: "SurrogateConfig",
    integer_indices: set[int] | None = None,
    n_objectives: int = 1,
) -> "SurrogateOptimizer":
    """
    创建代理模型优化器实例。

    model="qEHVI"/"NEHVI" 时返回 BoTorchMOOptimizer；
    其余 model 值返回 SurrogateOptimizer（skopt 后端）。
    BoTorch 不可用时明确抛错，不替换为 GP。

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
    n_objectives:
        目标函数数量，仅 BoTorch 路径使用（qEHVI/NEHVI）。

    Returns
    -------
    SurrogateOptimizer
    """
    if config.model in ("qEHVI", "NEHVI"):
        from .botorch_backend import BoTorchMOOptimizer, _check_botorch
        if not _check_botorch():
            raise RuntimeError(f"backend unavailable for {config.model}: install torch/botorch/gpytorch")
        if n_objectives < 2:
            raise ValueError(f"{config.model} requires multiple objectives")
        return BoTorchMOOptimizer(bounds, n_objectives, integer_indices or set(), config)
    return SurrogateOptimizer(bounds, config, integer_indices)
