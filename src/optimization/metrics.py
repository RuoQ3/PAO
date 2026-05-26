"""
metrics.py — 变量敏感性分析与重要性排序。

职责
----
从 DOE 样本（ProcessCase 列表）中计算各设计变量对目标函数的敏感性，
输出变量重要性排序，供 adaptive_region_search.py 决定：
  - 哪些变量参与后续搜索（高敏感性）
  - 哪些变量固定在名义值（低敏感性）
  - 沿哪些维度做更细的区域分块

支持的方法
----------
1. Spearman 秩相关（默认）
   - 不假设线性关系，对单调非线性关系有效
   - 计算成本极低，适合 DOE 样本数较少（10~50）的场景
   - 输出：每个变量对每个目标的 Spearman ρ（-1 ~ 1）

2. 方差贡献估计（Variance-based，简化版）
   - 将输入空间按变量分位数分组，计算组间目标方差 vs 总方差
   - 近似 Sobol 一阶指数，无需额外仿真
   - 适合样本数 >= 20 的场景

设计约定
--------
- 不依赖任何第三方库，仅使用标准库 math / statistics
- 所有函数接受 ProcessCase 列表，不直接操作 Aspen
- 敏感性指数统一归一化到 [0, 1]，便于跨目标比较
- 多目标场景下，默认取各目标敏感性的最大值作为变量的综合重要性

典型用法
--------
    from src.optimization.metrics import sensitivity_analysis, rank_variables

    result = sensitivity_analysis(
        cases=doe_cases,
        param_paths=list(config.param_bounds.keys()),
        objective_names=["TAC", "EMISSIONS"],
        method="spearman",
    )

    # 获取重要性排序（从高到低）
    ranked = rank_variables(result)
    important = [name for name, score in ranked if score >= 0.1]
    fixed     = [name for name, score in ranked if score <  0.1]
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

from ..models.process_case import ProcessCase

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """
    sensitivity_analysis() 的返回值。

    Attributes
    ----------
    param_paths:
        参与分析的设计变量路径列表（与输入顺序一致）。
    objective_names:
        参与分析的目标函数名称列表。
    scores:
        敏感性矩阵 scores[i][j] = 变量 i 对目标 j 的敏感性指数（0 ~ 1）。
        注意：有效样本对不足时该格为 1.0（保守/证据不足），而非 0.0（低敏感）。
        使用前应结合 effective_samples[i][j] 判断该格是否可信。
    composite_scores:
        综合敏感性 composite_scores[i] = max_j(scores[i][j])，
        用于变量重要性排序。
    effective_samples:
        有效样本对矩阵 effective_samples[i][j] = 变量 i 与目标 j 实际参与
        计算的样本对数量（成对删除后）。
        低于 min_required_samples 时该格证据不足，scores[i][j] 被保守设为 1.0，
        不应用于自动决策。
    min_required_samples:
        当前方法要求的最小有效样本对数量：
          Spearman → 3
          Variance → n_bins * 2（默认 8）
        is_reliable() 和 rank_variables() 均基于此阈值判断可靠性，
        而非固定值 3，避免 Variance 方法下的误判。
    method:
        使用的分析方法（"spearman" 或 "variance"）。
    n_samples:
        进入候选集的工况数（过滤后），不等于每个变量-目标对的实际有效样本数，
        后者见 effective_samples。
    warnings:
        分析过程中产生的警告信息列表（如样本数不足、路径缺失）。
    """
    param_paths: list[str]
    objective_names: list[str]
    scores: list[list[float]]
    composite_scores: list[float]
    effective_samples: list[list[int]]
    method: str
    n_samples: int
    min_required_samples: int = 3
    warnings: list[str] = field(default_factory=list)

    def score(self, param_path: str, objective_name: str) -> float | None:
        """返回指定变量对指定目标的敏感性指数，路径不存在时返回 None。"""
        try:
            i = self.param_paths.index(param_path)
            j = self.objective_names.index(objective_name)
            return self.scores[i][j]
        except ValueError:
            return None

    def composite(self, param_path: str) -> float | None:
        """返回指定变量的综合敏感性，路径不存在时返回 None。"""
        try:
            i = self.param_paths.index(param_path)
            return self.composite_scores[i]
        except ValueError:
            return None

    def min_effective(self, param_path: str) -> int | None:
        """
        返回指定变量在所有目标上的最小有效样本对数量。

        结合 min_required_samples 可区分：
          - 真实高敏感：score=1.0 且 min_effective >= min_required_samples
          - 证据不足：  score=1.0 但 min_effective < min_required_samples
        路径不存在时返回 None。
        """
        try:
            i = self.param_paths.index(param_path)
            return min(self.effective_samples[i])
        except ValueError:
            return None

    def is_reliable(self, param_path: str, min_samples: int | None = None) -> bool | None:
        """
        返回指定变量的敏感性结果是否可靠。

        可靠定义：所有目标的有效样本对均 >= 阈值。
        阈值默认使用 self.min_required_samples（由方法决定：
          Spearman=3，Variance=n_bins*2），也可通过 min_samples 覆盖。
        路径不存在时返回 None。
        上层 agent 在使用 score 做自动决策前应先检查此方法。
        """
        threshold = min_samples if min_samples is not None else self.min_required_samples
        m = self.min_effective(param_path)
        if m is None:
            return None
        return m >= threshold

    def to_summary(self) -> dict:
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "min_required_samples": self.min_required_samples,
            "n_params": len(self.param_paths),
            "n_objectives": len(self.objective_names),
            "warnings": self.warnings,
            "composite_scores": dict(zip(self.param_paths, self.composite_scores)),
            "scores": {
                path: dict(zip(self.objective_names, self.scores[i]))
                for i, path in enumerate(self.param_paths)
            },
            "effective_samples": {
                path: dict(zip(self.objective_names, self.effective_samples[i]))
                for i, path in enumerate(self.param_paths)
            },
        }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    cases: list[ProcessCase],
    param_paths: list[str],
    objective_names: list[str],
    method: Literal["spearman", "variance"] = "spearman",
    n_bins: int = 4,
    include_infeasible: bool = True,
) -> SensitivityResult:
    """
    从 DOE 样本中计算各设计变量对目标函数的敏感性。

    Parameters
    ----------
    cases:
        DOE 样本工况列表。
    param_paths:
        参与分析的设计变量 Aspen 路径列表，不能为空。
    objective_names:
        参与分析的目标函数名称列表，不能为空。
    method:
        分析方法：
        "spearman"（默认）：Spearman 秩相关，适合样本数 >= 5 的场景。
        "variance"：方差贡献估计，适合样本数 >= 20 的场景。
        非法值直接抛出 ValueError，不静默 fallback。
    n_bins:
        "variance" 方法的分组数，必须 >= 2，默认 4。
    include_infeasible:
        True（默认）：使用 simulation_valid=True 的工况（含 INFEASIBLE），
        在 Y 矩阵提取阶段按目标逐列过滤，某目标失败不影响其他目标列。
        False：仅使用 case.success=True 的工况（排除不可行点）。
        早期 DOE 阶段建议保持 True，以充分利用收敛但不可行的样本
        识别可行域边界和高敏感变量。

    Returns
    -------
    SensitivityResult

    Raises
    ------
    ValueError:
        param_paths 为空、objective_names 为空、method 非法、n_bins < 2、
        或某 param_path 在所有样本中均未读取到值（路径配置错误）。
    """
    # ------------------------------------------------------------------
    # 严格入口校验
    # ------------------------------------------------------------------
    if not param_paths:
        raise ValueError("param_paths 不能为空，至少需要一个设计变量路径。")
    if not objective_names:
        raise ValueError("objective_names 不能为空，至少需要一个目标函数名称。")
    if method not in ("spearman", "variance"):
        raise ValueError(
            f"method 必须为 'spearman' 或 'variance'，收到：{method!r}。"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins 必须 >= 2，收到：{n_bins}。")

    warns: list[str] = []

    # ------------------------------------------------------------------
    # 样本过滤：include_infeasible=True 时只按 simulation_valid 进入候选集，
    # 目标缺失在 Y 矩阵提取阶段逐列处理，不整体丢弃样本。
    # ------------------------------------------------------------------
    if include_infeasible:
        valid_cases = [c for c in cases if c.simulation_valid]
    else:
        valid_cases = [c for c in cases if c.success]

    n = len(valid_cases)
    n_obj = len(objective_names)
    n_par = len(param_paths)

    # ------------------------------------------------------------------
    # 计算最终方法和对应的最小有效样本对要求：
    #   variance 在 n < 20 时会自动切换为 spearman，min_req 随之变化。
    #   提前计算，确保所有 fallback 路径都使用正确的阈值。
    # ------------------------------------------------------------------
    effective_method = "spearman" if (method == "variance" and n < 20) else method
    min_req = 3 if effective_method == "spearman" else n_bins * 2

    # ------------------------------------------------------------------
    # n == 0：无任何候选样本，无法构建 X 矩阵，也无法校验路径，直接 fallback。
    # n > 0：先构建 X、校验路径，再决定是否 fallback——
    #   错误的 Aspen 树路径不应被"样本不足"掩盖成低敏感。
    # ------------------------------------------------------------------
    if n == 0:
        warns.append("有效样本数为 0，无法执行敏感性分析，所有变量将被视为同等重要。")
        _log.warning(warns[-1])
        return SensitivityResult(
            param_paths=param_paths,
            objective_names=objective_names,
            scores=[[1.0] * n_obj for _ in range(n_par)],
            composite_scores=[1.0] * n_par,
            effective_samples=[[0] * n_obj for _ in range(n_par)],
            method=effective_method,
            n_samples=0,
            min_required_samples=min_req,
            warnings=warns,
        )

    # ------------------------------------------------------------------
    # 提取输入矩阵 X[i][j]：第 i 个样本的第 j 个变量值
    # ------------------------------------------------------------------
    X: list[list[float | None]] = []
    for case in valid_cases:
        row: list[float | None] = []
        for path in param_paths:
            raw = case.design_vars.get(path)
            label = f"样本 '{case.case_id[:8]}' 变量 '{path.split(chr(92))[-1]}'"
            row.append(_coerce_finite_float(raw, label, warns))
        X.append(row)

    # ------------------------------------------------------------------
    # 路径可用性校验（在 n < 3 fallback 之前执行）：
    #   全部样本均缺失 → 配置错误，直接 ValueError
    #   大部分缺失（< 30%）→ warning，结果不可信
    # ------------------------------------------------------------------
    for j, path in enumerate(param_paths):
        n_available = sum(1 for row in X if row[j] is not None)
        if n_available == 0:
            raise ValueError(
                f"param_path '{path}' 在所有 {n} 个样本中均未读取到值。"
                "请检查 Aspen 树路径是否正确，或确认该变量已写入 design_vars。"
            )
        if n_available < n * 0.3:
            warns.append(
                f"变量 '{path.split(chr(92))[-1]}' 仅在 {n_available}/{n} 个样本中有值"
                f"（{n_available/n:.0%}），敏感性结果可能不可靠。"
            )

    # ------------------------------------------------------------------
    # 提取输出矩阵 Y[i][k]：第 i 个样本的第 k 个目标值（最小化方向）
    # 逐目标独立过滤，某目标失败不影响其他目标列
    # ------------------------------------------------------------------
    Y: list[list[float | None]] = []
    for case in valid_cases:
        row: list[float | None] = []
        for name in objective_names:
            obj = case.get_objective(name)
            if obj is not None and obj.available:
                label = f"样本 '{case.case_id[:8]}' 目标 '{name}'"
                raw_val = obj.value
                fv = _coerce_finite_float(raw_val, label, warns)
                if fv is not None:
                    row.append(fv if obj.minimize else -fv)
                else:
                    row.append(None)
            else:
                row.append(None)
        Y.append(row)

    # ------------------------------------------------------------------
    # 计算实际有效样本对数量（n < 3 时也需要，避免 fallback 掩盖真实计数）
    # ------------------------------------------------------------------
    eff: list[list[int]] = [
        [
            sum(1 for s in range(n) if X[s][i] is not None and Y[s][k] is not None)
            for k in range(n_obj)
        ]
        for i in range(n_par)
    ]

    # ------------------------------------------------------------------
    # n < 3：样本不足，score 保守设为 1.0，但 effective_samples 保留实际计数
    # ------------------------------------------------------------------
    if n < 3:
        warns.append(
            f"有效样本数 {n} < 3，敏感性分析结果不可靠，所有变量将被视为同等重要。"
        )
        _log.warning(warns[-1])
        return SensitivityResult(
            param_paths=param_paths,
            objective_names=objective_names,
            scores=[[1.0] * n_obj for _ in range(n_par)],
            composite_scores=[1.0] * n_par,
            effective_samples=eff,
            method=effective_method,
            n_samples=n,
            min_required_samples=min_req,
            warnings=warns,
        )

    if method == "variance" and n < 20:
        warns.append(
            f"variance 方法建议样本数 >= 20，当前 {n}，自动切换为 spearman。"
        )
        _log.warning(warns[-1])
        method = "spearman"

    if method == "spearman":
        scores, eff = _spearman_scores(X, Y, param_paths, objective_names, warns)
    else:
        scores, eff = _variance_scores(X, Y, param_paths, objective_names, n_bins, warns)

    composite = [max(scores[i]) for i in range(n_par)]

    _log.info(
        "敏感性分析完成（方法=%s，样本数=%d，含不可行=%s）：%s",
        method, n, include_infeasible,
        {p.split("\\")[-1]: round(s, 3) for p, s in zip(param_paths, composite)},
    )

    return SensitivityResult(
        param_paths=param_paths,
        objective_names=objective_names,
        scores=scores,
        composite_scores=composite,
        effective_samples=eff,
        method=method,
        min_required_samples=min_req,
        n_samples=n,
        warnings=warns,
    )


def rank_variables(
    result: SensitivityResult,
    threshold: float = 0.1,
) -> list[tuple[str, float]]:
    """
    按综合敏感性从高到低排序变量。

    Parameters
    ----------
    result:
        sensitivity_analysis() 的返回值。
    threshold:
        重要性阈值，低于此值的变量建议固定（仅用于日志提示，不过滤结果）。

    Returns
    -------
    list of (param_path, composite_score)，按 score 降序排列。

    Notes
    -----
    score=1.0 可能来自两种情况：
      1. 真实高敏感（is_reliable=True）
      2. 证据不足保守保留（is_reliable=False，min_effective < 3）
    日志中会对后者单独标注，避免上层 agent 误读。
    """
    ranked = sorted(
        zip(result.param_paths, result.composite_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    important = [(p, s) for p, s in ranked if s >= threshold]
    negligible = [(p, s) for p, s in ranked if s < threshold]

    if negligible:
        _log.info(
            "建议固定的低敏感性变量（score < %.2f）：%s",
            threshold,
            {p.split("\\")[-1]: round(s, 3) for p, s in negligible},
        )
    if important:
        _log.info(
            "建议保留的高敏感性变量（score >= %.2f）：%s",
            threshold,
            {p.split("\\")[-1]: round(s, 3) for p, s in important},
        )

    # 对证据不足导致的保守 1.0 单独标注，避免与真实高敏感混淆
    for path, score in ranked:
        if not result.is_reliable(path):
            m = result.min_effective(path)
            _log.info(
                "  注意：变量 '%s' score=%.3f 为保守值（最小有效样本对 %s < %d），"
                "非物理高敏感，不建议用于自动决策。",
                path.split("\\")[-1], score, m, result.min_required_samples,
            )

    return list(ranked)


# ---------------------------------------------------------------------------
# 有限数值强制转换辅助
# ---------------------------------------------------------------------------

def _coerce_finite_float(
    value: object,
    label: str,
    warns: list[str],
) -> float | None:
    """
    将 value 转换为有限浮点数。

    转换失败（非数值类型）或结果为 NaN/Inf 时返回 None 并追加警告。
    label 用于警告信息中定位问题来源（如变量路径或目标名称）。
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        warns.append(f"{label}：值 {value!r} 无法转换为浮点数，按缺失处理。")
        return None
    if not math.isfinite(f):
        warns.append(f"{label}：值 {f!r} 为非有限数（NaN/Inf），按缺失处理。")
        return None
    return f


# ---------------------------------------------------------------------------
# Spearman 秩相关
# ---------------------------------------------------------------------------

def _spearman_scores(
    X: list[list[float | None]],
    Y: list[list[float | None]],
    param_paths: list[str],
    objective_names: list[str],
    warns: list[str],
) -> tuple[list[list[float]], list[list[int]]]:
    """
    计算每个变量对每个目标的 Spearman 秩相关系数绝对值，归一化到 [0, 1]。

    某变量或目标存在 None 值时，跳过该样本对（成对删除）。
    有效样本对 < 3 时，该格保守设为 1.0（证据不足，非低敏感），并记录警告。

    Returns
    -------
    (scores, effective_samples)
    """
    n_par = len(param_paths)
    n_obj = len(objective_names)
    scores: list[list[float]] = [[0.0] * n_obj for _ in range(n_par)]
    eff: list[list[int]] = [[0] * n_obj for _ in range(n_par)]

    for i, path in enumerate(param_paths):
        for k, obj_name in enumerate(objective_names):
            pairs = [
                (X[s][i], Y[s][k])
                for s in range(len(X))
                if X[s][i] is not None and Y[s][k] is not None
            ]
            eff[i][k] = len(pairs)
            if len(pairs) < 3:
                warns.append(
                    f"变量 '{path.split(chr(92))[-1]}' vs 目标 '{obj_name}'："
                    f"有效样本对 {len(pairs)} < 3，证据不足，敏感性保守设为 1.0。"
                )
                scores[i][k] = 1.0  # 保守：证据不足 ≠ 低敏感
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            rho = _spearman_rho(xs, ys)
            scores[i][k] = abs(rho)

    return scores, eff


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """
    计算两个等长序列的 Spearman 秩相关系数。

    使用标准公式：ρ = 1 - 6Σd²/(n(n²-1))，其中 d 为秩差。
    存在并列秩时使用皮尔逊相关系数公式（更精确）。
    """
    n = len(xs)
    if n < 2:
        return 0.0

    rx = _rank(xs)
    ry = _rank(ys)

    # 检查是否有并列秩
    has_ties = len(set(xs)) < n or len(set(ys)) < n
    if has_ties:
        return _pearson(rx, ry)

    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0
    return 1.0 - 6.0 * d_sq_sum / denom


def _rank(xs: list[float]) -> list[float]:
    """
    计算序列的秩（1-based），并列值取平均秩。

    例：[3.0, 1.0, 3.0, 2.0] → [3.5, 1.0, 3.5, 2.0]
    """
    n = len(xs)
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    """计算皮尔逊相关系数，用于并列秩场景。"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return num / (sx * sy)


# ---------------------------------------------------------------------------
# 方差贡献估计
# ---------------------------------------------------------------------------

def _variance_scores(
    X: list[list[float | None]],
    Y: list[list[float | None]],
    param_paths: list[str],
    objective_names: list[str],
    n_bins: int,
    warns: list[str],
) -> tuple[list[list[float]], list[list[int]]]:
    """
    通过分组方差估计每个变量对每个目标的一阶敏感性指数。

    算法：
      1. 将变量 i 的值按分位数分为 n_bins 组
      2. 计算各组内目标 k 的均值
      3. 组间方差 / 总方差 ≈ 一阶 Sobol 指数

    归一化：各变量的得分除以最大值，映射到 [0, 1]。
    有效样本对不足时保守设为 1.0（证据不足，非低敏感）。

    Returns
    -------
    (scores, effective_samples)
    """
    n_par = len(param_paths)
    n_obj = len(objective_names)
    scores: list[list[float]] = [[0.0] * n_obj for _ in range(n_par)]
    eff: list[list[int]] = [[0] * n_obj for _ in range(n_par)]

    for i in range(n_par):
        for k in range(n_obj):
            pairs = [
                (X[s][i], Y[s][k])
                for s in range(len(X))
                if X[s][i] is not None and Y[s][k] is not None
            ]
            eff[i][k] = len(pairs)
            if len(pairs) < n_bins * 2:
                warns.append(
                    f"变量 '{param_paths[i].split(chr(92))[-1]}' vs 目标 "
                    f"'{objective_names[k]}'：有效样本对 {len(pairs)} 不足"
                    f"（需 >= {n_bins * 2}），证据不足，敏感性保守设为 1.0。"
                )
                scores[i][k] = 1.0  # 保守：证据不足 ≠ 低敏感
                continue

            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]

            total_var = _variance(ys)
            if total_var < 1e-12:
                continue

            # 按变量值分组
            sorted_pairs = sorted(zip(xs, ys), key=lambda t: t[0])
            bin_size = len(sorted_pairs) // n_bins
            group_means: list[float] = []
            group_weights: list[int] = []
            for b in range(n_bins):
                start = b * bin_size
                end = start + bin_size if b < n_bins - 1 else len(sorted_pairs)
                group_ys = [sorted_pairs[j][1] for j in range(start, end)]
                if group_ys:
                    group_means.append(sum(group_ys) / len(group_ys))
                    group_weights.append(len(group_ys))

            if not group_means:
                continue

            # 组间方差（加权）
            total_n = sum(group_weights)
            grand_mean = sum(m * w for m, w in zip(group_means, group_weights)) / total_n
            between_var = sum(
                w * (m - grand_mean) ** 2
                for m, w in zip(group_means, group_weights)
            ) / total_n

            scores[i][k] = min(between_var / total_var, 1.0)

    # 归一化：各目标列独立归一化，跳过证据不足（eff=0）的格子
    for k in range(n_obj):
        col = [scores[i][k] for i in range(n_par) if eff[i][k] >= n_bins * 2]
        max_val = max(col) if col else 0.0
        if max_val > 1e-12:
            for i in range(n_par):
                if eff[i][k] >= n_bins * 2:
                    scores[i][k] = scores[i][k] / max_val
                # eff[i][k] < n_bins*2 的格子已在上面设为 1.0，不参与归一化

    return scores, eff


def _variance(xs: list[float]) -> float:
    """计算样本方差（无偏，n-1 分母）。"""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / (n - 1)
