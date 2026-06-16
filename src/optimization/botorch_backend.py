"""
botorch_backend.py — BoTorch qEHVI/qNEHVI 多目标贝叶斯优化后端。

接口与 SurrogateOptimizer 完全兼容：tell(x, y, *, is_success) / ask() → list[float]，
可在 _MultiObjectiveBayesianOptimizer 中无缝替换 skopt 路径。

算法
----
qEHVI（q-Expected Hypervolume Improvement，默认）：
  每次推荐最大化超体积期望增量的候选点，通过蒙特卡罗采样估算期望值。
  相比 ParEGO（随机标量化），主动关注 Pareto 前沿的稀疏区域，样本效率更高。

qNEHVI（Noisy qEHVI）：
  适用于仿真结果有噪声的场景。

约束感知（Constrained qEHVI / PoF）
------------------------------------
若 tell() 接收到约束 margin 向量（c_vec），后端将目标和约束打包进同一个
多输出 SingleTaskGP（目标在前 n_obj 维，约束接在后面），然后通过
constraints=list[Callable] 把可行概率乘进 HV 期望值里：

    acquisition(x) = E[qEHVI(x)] × ∏_j 1{c_j(x) >= 0}

其中每个 callable 从模型输出的第 n_obj+j 维提取约束预测值，
返回负值表示可行（BoTorch 0.16.x 的约定：output < 0 → feasible）。

约束数据不足（< n_initial_min）或训练/优化失败时自动降级为无约束版本。

设备选择
--------
自动检测 CUDA，有 GPU 则用，否则退回 CPU。

回退机制
--------
BoTorch 初始化或优化失败时，ask() 返回随机点并写 WARNING 日志，
不向上层抛异常，保证优化循环不中断。

参考文献
--------
Daulton S. et al., "Differentiable Expected Hypervolume Improvement
for Parallel Multi-Objective Bayesian Optimization", NeurIPS 2020.

Gelbart M. et al., "Bayesian Optimization with Unknown Constraints",
UAI 2014.
"""
from __future__ import annotations

import logging
import math
import random as _random
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 设备选择
# ---------------------------------------------------------------------------

def _select_device() -> Any:
    """返回 torch.device（CUDA 或 CPU），torch 不可用时返回 None。"""
    try:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log.debug("BoTorch 设备：%s", dev)
        return dev
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# BoTorch 可用性检查
# ---------------------------------------------------------------------------

def _check_botorch() -> bool:
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
        return True
    except ImportError:
        return False


def _import_qlog_ehvi() -> tuple[Any, Any]:
    """
    按版本自动选择数值稳定的 qLogEHVI / qLogNEHVI 实现。

    BoTorch 版本差异：
      >= 0.9.2（发布时命名为 qLog*）：
        botorch.acquisition.multi_objective.logei 模块，
        类名 qLogExpectedHypervolumeImprovement / qLogNoisyExpectedHypervolumeImprovement
      旧版本（< 0.9.2）或 logei 模块不存在：
        回退到 botorch.acquisition.multi_objective.monte_carlo，
        类名 qExpectedHypervolumeImprovement / qNoisyExpectedHypervolumeImprovement

    Returns
    -------
    (qLogEHVI_class, qLogNEHVI_class)
    """
    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        )
        return qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
    except ImportError:
        # 旧版本回退
        from botorch.acquisition.multi_objective.monte_carlo import (
            qExpectedHypervolumeImprovement,
            qNoisyExpectedHypervolumeImprovement,
        )
        _log.debug(
            "botorch.acquisition.multi_objective.logei 不可用，"
            "回退到 qExpectedHypervolumeImprovement（旧版本）。"
        )
        return qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class BoTorchMOOptimizer:
    """
    BoTorch qEHVI/qNEHVI 多目标优化后端，支持约束感知采集（PoF）。

    约束感知模式（Constrained qEHVI）
    ----------------------------------
    当 tell() 接收到非空的 c_vec 时，目标和约束打包进同一个多输出 GP：
      输出维度 0..n_obj-1  → 目标（最大化方向，已取负）
      输出维度 n_obj..end  → 约束 margin（>= 0 表示满足）
    然后通过 constraints=list[Callable] 把可行概率乘进采集函数。
    约束数据不足时自动降级为无约束模式，不影响已有流程。

    Parameters
    ----------
    bounds:
        各维度的全局搜索边界 [(lo, hi), ...]。
    n_objectives:
        目标函数数量（>= 2）。
    integer_indices:
        整数维度的索引集合。
    config:
        SurrogateConfig，读取 model / random_seed。
    n_initial_min:
        启用 qEHVI 所需的最少成功样本数，默认 3。
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        n_objectives: int,
        integer_indices: set[int],
        config: Any,
        n_initial_min: int = 3,
    ) -> None:
        self._bounds = bounds
        self._n_obj = n_objectives
        self._integer_indices = integer_indices
        self._model_type = getattr(config, "model", "qEHVI")
        self._random_seed = getattr(config, "random_seed", None)
        self._n_initial_min = n_initial_min
        self._device = _select_device()
        self._rng = _random.Random(self._random_seed)

        # 目标函数观测数据（最小化方向）
        self._train_X: list[list[float]] = []
        self._train_Y: list[list[float]] = []

        # 约束 margin 观测数据
        # _train_C[i] 与 _train_X[i] 一一对应：
        #   非空 list  → 该点有完整约束 margin，参与约束 GP 训练
        #   空 list [] → 无约束数据，跳过
        self._train_C: list[list[float]] = []
        # 约束名称列表（首次接收到非空 c_vec 时按排序确定，后续不变）
        self._constraint_names: list[str] = []

        # Trust Region 局部 bounds（由 set_effective_bounds 设置）
        self._effective_bounds: list[tuple[float, float]] | None = None

        if not _check_botorch():
            _log.warning("BoTorch 未安装，BoTorchMOOptimizer 将始终返回随机点。")

    # ------------------------------------------------------------------
    # tell / ask 接口
    # ------------------------------------------------------------------

    def tell(
        self,
        x: list[float],
        y_scalar: float,
        *,
        is_success: bool,
        y_vec: list[float] | None = None,
        c_vec: dict[str, float] | None = None,
    ) -> None:
        """
        提交一次观测。

        Parameters
        ----------
        x:
            设计变量值向量。
        y_scalar:
            标量化值（ParEGO 兼容，BoTorch 路径不使用）。
        is_success:
            True 表示仿真成功且所有目标均可用。
        y_vec:
            多目标值向量（最小化方向）。None 时不参与 GP 训练。
        c_vec:
            约束 margin 字典 {约束名: margin}。
            margin = actual_value - threshold（>= 0 满足约束）。
            None 或空字典时该点不参与约束 GP 训练。
        """
        if is_success and y_vec is not None:
            if len(y_vec) == self._n_obj:
                self._train_X.append(list(x))
                self._train_Y.append(list(y_vec))

                if c_vec:
                    # 首次确定约束名顺序（排序保证确定性）
                    if not self._constraint_names:
                        self._constraint_names = sorted(c_vec.keys())
                        _log.debug(
                            "约束 GP：确定约束顺序 %s（共 %d 个）。",
                            self._constraint_names, len(self._constraint_names),
                        )
                    margin_vec = [
                        c_vec.get(name, 0.0) for name in self._constraint_names
                    ]
                    self._train_C.append(margin_vec)
                else:
                    # 无约束数据：空列表占位，保持与 _train_X 索引对齐
                    self._train_C.append([])
            else:
                _log.warning(
                    "tell(): y_vec 维度 %d 与 n_objectives %d 不一致，忽略此点。",
                    len(y_vec), self._n_obj,
                )

    def ask(self) -> list[float]:
        """
        推荐下一个候选点。

        成功样本数 < n_initial_min 或 BoTorch 不可用时，返回随机点。
        失败时回退到随机采样并写 WARNING 日志。
        """
        active_bounds = self._effective_bounds if self._effective_bounds else self._bounds

        if len(self._train_X) < self._n_initial_min or not _check_botorch():
            return self._random_point(active_bounds)

        try:
            return self._ask_botorch(active_bounds)
        except Exception as exc:
            _log.warning("qEHVI ask() 失败，回退到随机采样：%s", exc)
            return self._random_point(active_bounds)

    def set_effective_bounds(self, bounds: list[tuple[float, float]]) -> None:
        """设置本次 ask() 使用的局部 bounds（Trust Region 调用）。"""
        self._effective_bounds = bounds

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _ask_botorch(self, active_bounds: list[tuple[float, float]]) -> list[float]:
        """
        用约束感知 qEHVI 找下一个候选点。

        有足够约束数据时：目标+约束打包进联合 GP，用 constrained qEHVI。
        约束数据不足或训练失败时：静默降级为无约束 qEHVI。
        """
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        device = self._device or torch.device("cpu")
        dtype = torch.float64

        train_X = torch.tensor(self._train_X, dtype=dtype, device=device)
        # BoTorch 约定：最大化，目标取负
        train_Y_max = -torch.tensor(self._train_Y, dtype=dtype, device=device)

        # ── 判断是否启用约束 GP ───────────────────────────────────────
        valid_c_indices = [i for i, cv in enumerate(self._train_C) if cv]
        n_valid_c = len(valid_c_indices)
        use_constraints = (
            bool(self._constraint_names)
            and n_valid_c >= self._n_initial_min
        )

        if use_constraints:
            try:
                result = self._ask_with_constraints(
                    train_X, train_Y_max, valid_c_indices,
                    active_bounds, device, dtype,
                )
                return result
            except Exception as exc:
                _log.warning(
                    "constrained qEHVI 失败，降级为无约束版本：%s", exc
                )
        elif self._constraint_names:
            _log.debug(
                "约束 GP：有效数据点 %d < n_initial_min %d，本轮使用无约束 qEHVI。",
                n_valid_c, self._n_initial_min,
            )

        # ── 无约束版本（与旧行为完全一致）───────────────────────────
        return self._ask_unconstrained(
            train_X, train_Y_max, active_bounds, device, dtype
        )

    def _ask_unconstrained(
        self,
        train_X: Any,
        train_Y_max: Any,
        active_bounds: list[tuple[float, float]],
        device: Any,
        dtype: Any,
    ) -> list[float]:
        """標準無約束 qLogEHVI / qLogNEHVI。"""
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        model = SingleTaskGP(
            train_X, train_Y_max,
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=Standardize(m=self._n_obj),
        ).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        ref_point = (train_Y_max.min(dim=0).values * 1.1).tolist()
        sampler = self._make_sampler(device, dtype)
        qLogEHVI, qLogNEHVI = _import_qlog_ehvi()

        if self._model_type == "NEHVI":
            acqf = qLogNEHVI(
                model=model,
                ref_point=ref_point,
                X_baseline=train_X,
                sampler=sampler,
            )
        else:
            from botorch.utils.multi_objective.box_decompositions.non_dominated import (
                NondominatedPartitioning,
            )
            ref_point_t = torch.tensor(ref_point, dtype=dtype, device=device)
            partitioning = NondominatedPartitioning(
                ref_point=ref_point_t, Y=train_Y_max,
            )
            acqf = qLogEHVI(
                model=model,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )

        return self._optimize_and_round(acqf, active_bounds, device, dtype)

    def _ask_with_constraints(
        self,
        train_X: Any,
        train_Y_max: Any,
        valid_c_indices: list[int],
        active_bounds: list[tuple[float, float]],
        device: Any,
        dtype: Any,
    ) -> list[float]:
        """
        约束感知 qLogEHVI。

        目标和约束 margin 打包进同一个多输出 SingleTaskGP：
          输出维度 0..n_obj-1  → 目标（已取负，最大化方向）
          输出维度 n_obj..end  → 约束 margin（>= 0 满足）

        constraints callable：返回 -margin，负值表示可行（BoTorch 约定）。
        """
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            NondominatedPartitioning,
        )
        from gpytorch.mlls import ExactMarginalLogLikelihood

        from botorch.acquisition.multi_objective.objective import (
            IdentityMCMultiOutputObjective,
        )

        n_constraints = len(self._constraint_names)
        n_outputs = self._n_obj + n_constraints

        c_data = [self._train_C[i] for i in valid_c_indices]
        X_joint = train_X[valid_c_indices]
        Y_obj   = train_Y_max[valid_c_indices]
        C_mat   = torch.tensor(c_data, dtype=dtype, device=device)
        Y_joint = torch.cat([Y_obj, C_mat], dim=-1)  # (n_valid, n_obj+n_c)

        joint_model = SingleTaskGP(
            X_joint, Y_joint,
            input_transform=Normalize(d=X_joint.shape[-1]),
            outcome_transform=Standardize(m=n_outputs),
        ).to(device)
        mll = ExactMarginalLogLikelihood(joint_model.likelihood, joint_model)
        fit_gpytorch_mll(mll)

        # constraints: list[Callable]，返回负值表示可行
        constraint_callables = []
        for j in range(n_constraints):
            output_idx = self._n_obj + j

            def _make_callable(idx: int):
                def _c(samples):   # (..., q, n_outputs)
                    return -samples[..., idx]   # -margin <= 0 → feasible
                return _c

            constraint_callables.append(_make_callable(output_idx))

        ref_point = (Y_obj.min(dim=0).values * 1.1).tolist()
        ref_point_t = torch.tensor(ref_point, dtype=dtype, device=device)
        # partitioning 只包含目标维度（n_obj 维），ref_point 也只有 n_obj 维。
        # train_Y_max 全集保持 Pareto 前沿完整（含无约束数据的点）。
        partitioning = NondominatedPartitioning(
            ref_point=ref_point_t, Y=train_Y_max,
        )
        sampler = self._make_sampler(device, dtype)
        qLogEHVI, _ = _import_qlog_ehvi()

        # objective 告诉 qLogEHVI 只把前 n_obj 维当目标（其余维是约束，不进 HV）
        objective = IdentityMCMultiOutputObjective(
            outcomes=list(range(self._n_obj)),
            num_outcomes=self._n_obj + n_constraints,
        )

        acqf = qLogEHVI(
            model=joint_model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraint_callables,
        )

        _log.info(
            "本轮使用 constrained qLogEHVI：%d 个约束（%s），有效样本 %d/%d 个。",
            n_constraints, self._constraint_names,
            len(valid_c_indices), len(self._train_X),
        )

        return self._optimize_and_round(acqf, active_bounds, device, dtype)

    def _optimize_and_round(
        self,
        acqf: Any,
        active_bounds: list[tuple[float, float]],
        device: Any,
        dtype: Any,
    ) -> list[float]:
        """运行 optimize_acqf 并对整数维度取整。"""
        import torch
        from botorch.optim import optimize_acqf

        bounds_t = torch.tensor(
            [[lo for lo, hi in active_bounds], [hi for lo, hi in active_bounds]],
            dtype=dtype,
            device=device,
        )

        candidate, _ = optimize_acqf(
            acqf,
            bounds=bounds_t,
            q=1,
            num_restarts=5,
            raw_samples=128,
            options={"seed": self._rng.randint(0, 2 ** 31)},
        )

        result = candidate.detach().squeeze(0).tolist()

        for i in self._integer_indices:
            lo, hi = active_bounds[i]
            result[i] = float(
                max(math.ceil(lo), min(math.floor(hi), round(result[i])))
            )

        return result

    def _make_sampler(self, device: Any, dtype: Any) -> Any:
        """构造 SobolQMCSampler。"""
        import torch
        from botorch.sampling.normal import SobolQMCNormalSampler
        return SobolQMCNormalSampler(
            sample_shape=torch.Size([128]),
            seed=self._rng.randint(0, 2 ** 31),
        )

    def _random_point(self, bounds: list[tuple[float, float]]) -> list[float]:
        """在 bounds 内均匀随机采样，整数维度取整。"""
        point = []
        for i, (lo, hi) in enumerate(bounds):
            if i in self._integer_indices:
                int_lo = math.ceil(lo)
                int_hi = math.floor(hi)
                point.append(float(self._rng.randint(int_lo, max(int_lo, int_hi))))
            else:
                point.append(lo + self._rng.random() * (hi - lo))
        return point

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def n_success(self) -> int:
        """当前成功观测数。"""
        return len(self._train_X)

    @property
    def n_constraints(self) -> int:
        """当前已确定的约束数量。"""
        return len(self._constraint_names)

    @property
    def constraint_names(self) -> list[str]:
        """约束名称列表（按首次 tell 时的排序顺序）。"""
        return list(self._constraint_names)
