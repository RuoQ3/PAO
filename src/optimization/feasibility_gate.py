"""
feasibility_gate.py — 可行性分类器启用前的质量门控模块。

在实际运行 optimize_case / optimize_pareto_case 之前，用
FeasibilityEvaluationResult 判断分类器是否满足质量要求，
决定是否允许启用 feasibility_filter。

核心设计理念
------------
feasibility_filter 的最大风险是 False Negative：
实际可行的 Aspen 工况被分类器错误地过滤掉，导致优化器错失潜在最优点。
因此门控的核心指标是 recall 和 false_negative_rate，而非 accuracy。

架构定位
--------
本模块只做"评估结果 → 允许/拒绝"的决策，不连接 Aspen COM，
不修改 ProcessCase / SimulationDB / optimize_case 的运行逻辑。
后续可由 workflow 层或 agent 工具将 gate 决策接入 YAML 启用流程。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 门控配置
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityGateConfig:
    """
    质量门控配置：指定允许启用 feasibility_filter 的最低指标要求。

    Attributes
    ----------
    enabled:
        是否启用门控机制。False 时直接放行（allowed=True），不检查任何指标。
        注意：这是"门控关闭"，不是 feasibility_filter 关闭。
    min_samples:
        评估所用有效样本数的最低要求，默认 30。
    min_recall:
        最低 recall（召回率）要求，默认 0.85。
        recall 低 = 可行点被错杀多，是最重要的门控指标。
    max_false_negative_rate:
        测试集中实际可行样本里，被误判为不可行的最高比例，默认 0.10。
        计算公式：FN / (FN + TP)，即 1 - recall。
        单独作为硬阈值，因为这是优化器视角的直接损失。
    max_false_negative_count:
        允许的 false negative 绝对数量上限。None 表示不限制（依赖比例指标）。
        适合小样本场景：即使 rate 低，绝对数量也不能太多。
    min_precision:
        最低 precision 要求。None 表示不检查。
    min_f1:
        最低 F1 分数要求。None 表示不检查。
    require_no_warnings:
        True 时，如果评估结果有任何 warning（包括单类别、FN 警告等），
        则拒绝启用过滤器。
    allow_single_class:
        True 时，允许评估样本只有单一类别（但指标为 None，通常被其他检查拦截）。
        False 时（默认），单类别评估直接拒绝。
    """
    enabled: bool = True
    min_samples: int = 30
    min_recall: float = 0.85
    max_false_negative_rate: float = 0.10
    max_false_negative_count: int | None = None
    min_precision: float | None = None
    min_f1: float | None = None
    require_no_warnings: bool = False
    allow_single_class: bool = False

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError(
                f"FeasibilityGateConfig.min_samples={self.min_samples} 必须 >= 1。"
            )
        if not (0.0 <= self.min_recall <= 1.0):
            raise ValueError(
                f"FeasibilityGateConfig.min_recall={self.min_recall} 必须在 [0.0, 1.0] 之间。"
            )
        if not (0.0 <= self.max_false_negative_rate <= 1.0):
            raise ValueError(
                f"FeasibilityGateConfig.max_false_negative_rate={self.max_false_negative_rate}"
                " 必须在 [0.0, 1.0] 之间。"
            )
        if self.max_false_negative_count is not None and self.max_false_negative_count < 0:
            raise ValueError(
                f"FeasibilityGateConfig.max_false_negative_count="
                f"{self.max_false_negative_count} 必须 >= 0。"
            )
        if self.min_precision is not None and not (0.0 <= self.min_precision <= 1.0):
            raise ValueError(
                f"FeasibilityGateConfig.min_precision={self.min_precision}"
                " 必须在 [0.0, 1.0] 之间。"
            )
        if self.min_f1 is not None and not (0.0 <= self.min_f1 <= 1.0):
            raise ValueError(
                f"FeasibilityGateConfig.min_f1={self.min_f1} 必须在 [0.0, 1.0] 之间。"
            )


# ---------------------------------------------------------------------------
# 门控决策结果
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityGateDecision:
    """
    质量门控决策结果。

    Attributes
    ----------
    allowed:
        True 表示允许启用 feasibility_filter；False 表示拒绝。
    reason:
        一句话总结决策原因，供日志和 agent 工具展示。
    failed_checks:
        未通过的检查项描述列表（allowed=False 时非空）。
    passed_checks:
        通过的检查项描述列表。
    warnings:
        提示性信息（不阻止通过，但需要关注）。
    metrics:
        本次决策使用的关键指标快照，便于 agent 工具展示和记录。
        包含：recall, precision, f1, accuracy,
              false_negative_count, false_negative_rate,
              n_samples, n_test。
    """
    allowed: bool
    reason: str
    failed_checks: list[str] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典，供日志和 agent 工具消费。"""
        return {
            "allowed":       self.allowed,
            "reason":        self.reason,
            "failed_checks": self.failed_checks,
            "passed_checks": self.passed_checks,
            "warnings":      self.warnings,
            "metrics":       self.metrics,
        }

    def summary(self) -> str:
        """返回人类可读的单行摘要。"""
        status = "✓ ALLOWED" if self.allowed else "✗ REJECTED"
        fn_rate = self.metrics.get("false_negative_rate")
        fn_str = f" FN_rate={fn_rate:.3f}" if isinstance(fn_rate, float) else ""
        return (
            f"FeasibilityGate [{status}] {self.reason} |"
            f" recall={_fmt(self.metrics.get('recall'))}"
            f" precision={_fmt(self.metrics.get('precision'))}"
            f" f1={_fmt(self.metrics.get('f1'))}"
            + fn_str
        )


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else "N/A"


# ---------------------------------------------------------------------------
# 门控决策函数
# ---------------------------------------------------------------------------

def decide_feasibility_filter_gate(
    evaluation: Any,  # FeasibilityEvaluationResult，避免循环导入用 Any
    config: FeasibilityGateConfig | None = None,
) -> FeasibilityGateDecision:
    """
    根据离线评估结果判断是否允许启用 feasibility_filter。

    Parameters
    ----------
    evaluation:
        FeasibilityEvaluationResult 实例，来自
        evaluate_feasibility_classifier() 或 evaluate_feasibility_from_db()。
    config:
        门控配置，None 时使用默认值（相对严格）。

    Returns
    -------
    FeasibilityGateDecision
        allowed=True 表示可以启用过滤器；False 表示建议禁用。
    """
    if config is None:
        config = FeasibilityGateConfig()

    failed:  list[str] = []
    passed:  list[str] = []
    gate_warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # 0. 门控机制本身关闭
    # ------------------------------------------------------------------ #
    if not config.enabled:
        return FeasibilityGateDecision(
            allowed=True,
            reason="门控机制已关闭（FeasibilityGateConfig.enabled=False），不检查分类器质量。",
            passed_checks=["gate_disabled"],
            metrics=_build_metrics(evaluation, fn_rate=None),
        )

    # ------------------------------------------------------------------ #
    # 计算 false_negative_rate（越早越好，后续多个检查要用）
    # ------------------------------------------------------------------ #
    fn_rate: float | None = None
    fn_rate_note: str = ""
    cm = evaluation.confusion_matrix
    fn = evaluation.false_negative_count

    if cm is not None:
        # confusion_matrix 格式：[[TN, FP], [FN, TP]]
        tn, fp = cm[0][0], cm[0][1]
        fn_cm, tp = cm[1][0], cm[1][1]
        actual_positive_test = fn_cm + tp   # 测试集中实际可行样本数
        if actual_positive_test > 0:
            fn_rate = fn_cm / actual_positive_test
        else:
            fn_rate_note = "测试集中无实际可行样本（actual_positive=0），无法计算 false_negative_rate。"
            gate_warnings.append(f"⚠ {fn_rate_note}")
    else:
        fn_rate_note = "confusion_matrix 为 None，无法计算 false_negative_rate。"
        gate_warnings.append(f"⚠ {fn_rate_note}")

    metrics = _build_metrics(evaluation, fn_rate)

    # ------------------------------------------------------------------ #
    # 1. 样本数检查
    # ------------------------------------------------------------------ #
    if evaluation.n_samples < config.min_samples:
        failed.append(
            f"n_samples={evaluation.n_samples} < min_samples={config.min_samples}，"
            "历史样本不足，分类器评估不可靠。"
        )
    else:
        passed.append(f"n_samples={evaluation.n_samples} >= {config.min_samples}")

    # ------------------------------------------------------------------ #
    # 2. 评估指标可用性检查（任一核心指标为 None 视为不可用）
    # ------------------------------------------------------------------ #
    if any(
        v is None for v in (
            evaluation.accuracy, evaluation.recall,
            evaluation.precision, evaluation.f1,
        )
    ):
        none_fields = [
            name for name, v in (
                ("accuracy", evaluation.accuracy),
                ("recall", evaluation.recall),
                ("precision", evaluation.precision),
                ("f1", evaluation.f1),
            ) if v is None
        ]
        failed.append(
            f"评估指标 {none_fields} 为 None"
            "（可能原因：单类别样本、分类器训练失败或样本不足），"
            "无法验证分类器质量。"
        )
        # 指标不可用时，后续数值检查无意义，直接返回拒绝
        return FeasibilityGateDecision(
            allowed=False,
            reason=f"评估指标不可用，拒绝启用 feasibility_filter。失败项：{'; '.join(failed)}",
            failed_checks=failed,
            passed_checks=passed,
            warnings=gate_warnings + (evaluation.warnings or []),
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # 3. 单类别检查
    # ------------------------------------------------------------------ #
    if not config.allow_single_class and (
        evaluation.n_feasible == 0 or evaluation.n_infeasible == 0
    ):
        failed.append(
            f"样本只有单一类别（feasible={evaluation.n_feasible}，"
            f"infeasible={evaluation.n_infeasible}），无法评估二分类可靠性。"
        )
    elif evaluation.n_feasible > 0 and evaluation.n_infeasible > 0:
        passed.append(
            f"正负样本均有（feasible={evaluation.n_feasible}，"
            f"infeasible={evaluation.n_infeasible}）"
        )

    # ------------------------------------------------------------------ #
    # 4. recall 检查（核心：过低意味着可行点被大量错杀）
    # ------------------------------------------------------------------ #
    recall = evaluation.recall
    if recall < config.min_recall:
        failed.append(
            f"recall={recall:.3f} < min_recall={config.min_recall}，"
            "过滤器会错误丢弃过多实际可行的 Aspen 工况。"
        )
    else:
        passed.append(f"recall={recall:.3f} >= {config.min_recall}")

    # ------------------------------------------------------------------ #
    # 5. false_negative_rate 检查
    # ------------------------------------------------------------------ #
    if fn_rate is not None:
        if fn_rate > config.max_false_negative_rate:
            failed.append(
                f"false_negative_rate={fn_rate:.3f} > max_false_negative_rate="
                f"{config.max_false_negative_rate}，"
                "实际可行点被过滤器误判的比例过高，有损优化闭环。"
            )
        else:
            passed.append(
                f"false_negative_rate={fn_rate:.3f} <= {config.max_false_negative_rate}"
            )
    else:
        # confusion_matrix 缺失时只有 fn 绝对数，无法算 rate
        failed.append(
            f"无法计算 false_negative_rate（{fn_rate_note}），"
            "无法验证核心风险指标，拒绝启用。"
        )

    # ------------------------------------------------------------------ #
    # 6. max_false_negative_count 检查（绝对数量上限，可选）
    # ------------------------------------------------------------------ #
    if config.max_false_negative_count is not None:
        if fn > config.max_false_negative_count:
            failed.append(
                f"false_negative_count={fn} > max_false_negative_count="
                f"{config.max_false_negative_count}，"
                "绝对数量超限，错杀的可行点过多。"
            )
        else:
            passed.append(
                f"false_negative_count={fn} <= {config.max_false_negative_count}"
            )

    # ------------------------------------------------------------------ #
    # 7. precision 检查（可选）
    # ------------------------------------------------------------------ #
    if config.min_precision is not None:
        prec = evaluation.precision
        if prec is None or prec < config.min_precision:
            failed.append(
                f"precision={_fmt(prec)} < min_precision={config.min_precision}，"
                "预测为可行的工况中误报率过高，浪费 Aspen 仿真资源。"
            )
        else:
            passed.append(f"precision={prec:.3f} >= {config.min_precision}")

    # ------------------------------------------------------------------ #
    # 8. f1 检查（可选）
    # ------------------------------------------------------------------ #
    if config.min_f1 is not None:
        f1 = evaluation.f1
        if f1 is None or f1 < config.min_f1:
            failed.append(
                f"f1={_fmt(f1)} < min_f1={config.min_f1}，"
                "分类器综合性能不足。"
            )
        else:
            passed.append(f"f1={f1:.3f} >= {config.min_f1}")

    # ------------------------------------------------------------------ #
    # 9. require_no_warnings 检查（可选）
    # ------------------------------------------------------------------ #
    if config.require_no_warnings and evaluation.warnings:
        failed.append(
            f"evaluation.warnings 非空（共 {len(evaluation.warnings)} 条），"
            "门控配置要求评估零 warning。"
        )
    elif not evaluation.warnings:
        passed.append("evaluation 无 warning")

    # ------------------------------------------------------------------ #
    # 10. 最终判断
    # ------------------------------------------------------------------ #
    allowed = len(failed) == 0

    if allowed:
        fn_rate_str = f"{fn_rate:.3f}" if fn_rate is not None else "N/A"
        reason = (
            f"所有门控检查通过（n={evaluation.n_samples}，"
            f"recall={recall:.3f}，"
            f"fn_rate={fn_rate_str}），"
            "允许启用 feasibility_filter。"
        )
    else:
        reason = (
            f"{len(failed)} 项门控检查未通过，拒绝启用 feasibility_filter。"
            f" 核心问题：{failed[0]}"
        )

    decision = FeasibilityGateDecision(
        allowed=allowed,
        reason=reason,
        failed_checks=failed,
        passed_checks=passed,
        warnings=gate_warnings + (evaluation.warnings or []),
        metrics=metrics,
    )
    _log.info("质量门控决策：%s", decision.summary())
    return decision


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _build_metrics(evaluation: Any, fn_rate: float | None) -> dict[str, Any]:
    """从 evaluation 提取关键指标快照。"""
    return {
        "recall":               evaluation.recall,
        "precision":            evaluation.precision,
        "f1":                   evaluation.f1,
        "accuracy":             evaluation.accuracy,
        "false_negative_count": evaluation.false_negative_count,
        "false_negative_rate":  fn_rate,
        "n_samples":            evaluation.n_samples,
        "n_test":               evaluation.n_test,
        "n_feasible":           evaluation.n_feasible,
        "n_infeasible":         evaluation.n_infeasible,
    }
