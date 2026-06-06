"""
feasibility_evaluation.py — 可行性分类器离线评估模块。

用途：在不启动 Aspen COM 的前提下，对历史 ProcessCase 数据评估
FeasibilityClassifier 的分类性能和候选过滤效果，为继续开发 agent
之前提供"质量检查仪表盘"。

架构定位
---------
本模块只读取已有 ProcessCase 数据或 SimulationDB.export_feasibility_dataset()
的输出，不写入数据库，不修改 ProcessCase，不连接 Aspen COM。

关键指标
--------
  accuracy    — 整体准确率
  precision   — 预测可行中实际可行的比例
  recall      — 实际可行中被正确预测的比例（Sensitivity）
  f1          — 精度与召回率的调和平均
  false_negative_count — **最重要**：实际可行但被过滤器误判为不可行的样本数。
                          若此数过高，优化器会错杀有价值的 Aspen 工况。
  false_positive_count — 实际不可行但被误判为可行，导致多余仿真。

依赖
----
  sklearn（可选）：用于 train_test_split 和指标计算；
  不可用时 evaluate_feasibility_classifier 会抛 ImportError。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

_log = logging.getLogger(__name__)

# sklearn 懒加载
try:
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import (
        accuracy_score as _acc,
        precision_score as _prec,
        recall_score as _rec,
        f1_score as _f1,
        confusion_matrix as _cm,
    )
    _HAS_SKLEARN = True
except ImportError:
    _tts = None   # type: ignore[assignment]
    _acc = _prec = _rec = _f1 = _cm = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityEvaluationConfig:
    """
    可行性分类器离线评估配置。

    Attributes
    ----------
    test_size:
        测试集比例，(0, 1) 之间，默认 0.25。
    random_seed:
        随机种子，用于可复现的训练/测试划分和模型训练，默认 42。
    threshold:
        分类判决阈值，predict_proba >= threshold 视为可行，默认 0.5。
    min_samples:
        最少有效样本数，不足时抛 ValueError，默认 10。
    model:
        分类模型，支持 "extra_trees" / "random_forest"，默认 "extra_trees"。
    """
    test_size: float = 0.25
    random_seed: int | None = 42
    threshold: float = 0.5
    min_samples: int = 10
    model: str = "extra_trees"

    _VALID_MODELS: frozenset[str] = field(
        default=frozenset({"extra_trees", "random_forest", "random"}),
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(
                f"FeasibilityEvaluationConfig.test_size={self.test_size} 必须在 (0, 1) 之间。"
            )
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"FeasibilityEvaluationConfig.threshold={self.threshold} 必须在 [0.0, 1.0] 之间。"
            )
        if self.min_samples < 1:
            raise ValueError(
                f"FeasibilityEvaluationConfig.min_samples={self.min_samples} 必须 >= 1。"
            )
        if self.model not in self._VALID_MODELS:
            raise ValueError(
                f"FeasibilityEvaluationConfig.model={self.model!r} 不合法，"
                f"支持值：{sorted(self._VALID_MODELS)}。"
            )


# ---------------------------------------------------------------------------
# 评估结果
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityEvaluationResult:
    """
    可行性分类器评估结果。

    Attributes
    ----------
    n_samples:
        用于评估的有效样本总数。
    n_train:
        训练集样本数。
    n_test:
        测试集样本数。
    n_feasible:
        有效样本中正样本（feasible=True）数量。
    n_infeasible:
        有效样本中负样本数量。
    accuracy:
        测试集整体准确率（0.0~1.0）。None 表示无法计算。
    precision:
        精度：预测可行中实际可行的比例。None 表示无法计算。
    recall:
        召回率：实际可行中被正确预测的比例。None 表示无法计算。
    f1:
        F1 分数。None 表示无法计算。
    false_positive_count:
        实际不可行但被预测为可行的样本数（浪费仿真资源）。
    false_negative_count:
        **最重要**：实际可行但被预测为不可行的样本数。
        若此数过高，意味着分类器会错杀有价值的 Aspen 工况。
    confusion_matrix:
        混淆矩阵 [[TN, FP], [FN, TP]]（标准 sklearn 格式）。
        None 表示无法计算。
    threshold:
        评估使用的判决阈值。
    model:
        评估使用的模型名称。
    var_names:
        训练时使用的设计变量路径列表。
    warnings:
        评估过程中产生的警告信息列表（样本不足、单类别等）。
    """
    n_samples: int
    n_train: int
    n_test: int
    n_feasible: int
    n_infeasible: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_count: int
    false_negative_count: int
    confusion_matrix: list[list[int]] | None
    threshold: float
    model: str
    var_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典，方便日志和 agent 工具消费。"""
        return {
            "n_samples":            self.n_samples,
            "n_train":              self.n_train,
            "n_test":               self.n_test,
            "n_feasible":           self.n_feasible,
            "n_infeasible":         self.n_infeasible,
            "accuracy":             self.accuracy,
            "precision":            self.precision,
            "recall":               self.recall,
            "f1":                   self.f1,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "confusion_matrix":     self.confusion_matrix,
            "threshold":            self.threshold,
            "model":                self.model,
            "var_names":            self.var_names,
            "warnings":             self.warnings,
        }

    def summary(self) -> str:
        """返回人类可读的单行摘要，方便日志输出。"""
        fn_warn = (
            f"  ⚠ FALSE_NEGATIVE={self.false_negative_count}（过滤器可能丢弃可行点）"
            if self.false_negative_count > 0 else ""
        )
        return (
            f"FeasibilityEval: n={self.n_samples}(train={self.n_train}/test={self.n_test}) "
            f"pos={self.n_feasible}/neg={self.n_infeasible} | "
            f"acc={_fmt(self.accuracy)} prec={_fmt(self.precision)} "
            f"rec={_fmt(self.recall)} f1={_fmt(self.f1)} | "
            f"FP={self.false_positive_count} FN={self.false_negative_count}"
            + fn_warn
        )


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# 核心评估函数
# ---------------------------------------------------------------------------

def evaluate_feasibility_classifier(
    cases: Sequence[Any],  # Sequence[ProcessCase]，避免循环导入用 Any
    config: FeasibilityEvaluationConfig | None = None,
    var_names: list[str] | None = None,
) -> FeasibilityEvaluationResult:
    """
    从历史 ProcessCase 数据评估可行性分类器性能。

    Parameters
    ----------
    cases:
        ProcessCase 序列，通常来自 OptimizeResult.cases 或数据库查询结果。
    config:
        评估配置，None 时使用默认值。
    var_names:
        设计变量路径列表（可选）。若传入，严格使用这些变量作为特征列，
        缺失或无法转 float 的样本会被跳过并记录 warning；
        若为 None，则取所有有效样本 design_vars 的交集自动推断。

    Returns
    -------
    FeasibilityEvaluationResult

    Raises
    ------
    ImportError
        sklearn 不可用时抛出。
    ValueError
        配置参数非法，或有效样本数 < config.min_samples 时抛出。
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "evaluate_feasibility_classifier 需要 sklearn，"
            "请先安装：pip install scikit-learn"
        )

    if config is None:
        config = FeasibilityEvaluationConfig()

    warnings_list: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. 筛选有效样本
    # ------------------------------------------------------------------ #
    valid_cases = [
        c for c in cases
        if c.valid_for_classifier and c.design_vars
    ]

    n_total = len(valid_cases)
    n_feasible = sum(1 for c in valid_cases if c.feasible_label)
    n_infeasible = n_total - n_feasible

    # 样本数检查
    if n_total < config.min_samples:
        raise ValueError(
            f"有效样本数 {n_total} < min_samples={config.min_samples}，"
            "无法进行可靠的训练/测试评估。请增加仿真样本或降低 min_samples。"
        )

    # 单一类别检查
    if n_feasible == 0 or n_infeasible == 0:
        msg = (
            f"样本只有{'正' if n_infeasible == 0 else '负'}类 "
            f"（feasible={n_feasible}，infeasible={n_infeasible}），"
            "无法训练有效的二分类器。"
        )
        warnings_list.append(f"⚠ {msg}")
        _log.warning(msg)
        return FeasibilityEvaluationResult(
            n_samples=n_total,
            n_train=0,
            n_test=0,
            n_feasible=n_feasible,
            n_infeasible=n_infeasible,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            false_positive_count=0,
            false_negative_count=0,
            confusion_matrix=None,
            threshold=config.threshold,
            model=config.model,
            warnings=warnings_list,
        )

    # ------------------------------------------------------------------ #
    # 2. 确定特征列
    # ------------------------------------------------------------------ #
    if var_names is not None:
        # 外部指定：严格使用给定变量列表，与优化器实际使用的维度一致
        if not var_names:
            raise ValueError("var_names 不能为空列表。")
        used_var_names = list(var_names)
    else:
        # 自动推断：取所有有效样本 design_vars 的交集
        key_sets = [set(c.design_vars.keys()) for c in valid_cases]
        used_var_names = sorted(set.intersection(*key_sets))
        if not used_var_names:
            raise ValueError(
                "所有有效样本的 design_vars 没有公共键，无法构造特征向量。"
                "请检查各工况的 design_vars 字段是否一致，或通过 var_names 参数显式指定。"
            )

    # ------------------------------------------------------------------ #
    # 3. 构造特征矩阵和标签向量
    # ------------------------------------------------------------------ #
    X: list[list[float]] = []
    y: list[int] = []
    skipped = 0
    for c in valid_cases:
        try:
            row = [float(c.design_vars[v]) for v in used_var_names]
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        X.append(row)
        y.append(1 if c.feasible_label else 0)

    if skipped:
        warnings_list.append(
            f"⚠ 有 {skipped} 个样本因变量缺失或无法转 float 被跳过。"
        )

    n_valid = len(X)
    if n_valid < config.min_samples:
        raise ValueError(
            f"特征提取后有效样本数 {n_valid} < min_samples={config.min_samples}。"
        )

    # ------------------------------------------------------------------ #
    # 4. 训练/测试划分
    # ------------------------------------------------------------------ #
    # 确保测试集中正负类都有（stratify=y）
    try:
        X_tr, X_te, y_tr, y_te = _tts(
            X, y,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=y,
        )
    except ValueError as exc:
        # 分层失败时（测试集太小无法放置所有类别）退回不分层
        warnings_list.append(
            f"⚠ 分层划分失败（{exc}），退回不分层划分。"
        )
        X_tr, X_te, y_tr, y_te = _tts(
            X, y,
            test_size=config.test_size,
            random_state=config.random_seed,
        )

    n_train = len(X_tr)
    n_test  = len(X_te)

    # ------------------------------------------------------------------ #
    # 5. 训练模型
    # ------------------------------------------------------------------ #
    from .feasibility import FeasibilityConfig, FeasibilityClassifier

    clf_cfg = FeasibilityConfig(
        enabled=True,
        model=config.model,
        min_samples=1,          # 评估时放宽内部门槛，由此函数负责样本检查
        threshold=config.threshold,
        random_seed=config.random_seed,
    )
    clf = FeasibilityClassifier(clf_cfg)

    # 直接构造 rows 格式传给 fit
    rows = [
        {"case_id": f"train_{i}", "design_vars": dict(zip(used_var_names, x)), "label": bool(lb)}
        for i, (x, lb) in enumerate(zip(X_tr, y_tr))
    ]
    fit_ok = clf.fit(rows, used_var_names)
    if not fit_ok:
        warnings_list.append("⚠ 分类器训练失败（FeasibilityClassifier.fit 返回 False）。")
        return FeasibilityEvaluationResult(
            n_samples=n_valid,
            n_train=n_train,
            n_test=n_test,
            n_feasible=sum(y),
            n_infeasible=n_valid - sum(y),
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            false_positive_count=0,
            false_negative_count=0,
            confusion_matrix=None,
            threshold=config.threshold,
            model=config.model,
            var_names=var_names,
            warnings=warnings_list,
        )

    # ------------------------------------------------------------------ #
    # 6. 预测并计算指标
    # ------------------------------------------------------------------ #
    y_pred: list[int] = []
    for x_row in X_te:
        prob = clf.predict_proba(dict(zip(used_var_names, x_row)))
        y_pred.append(1 if prob >= config.threshold else 0)

    acc   = float(_acc(y_te, y_pred))
    prec  = float(_prec(y_te, y_pred, zero_division=0))
    rec   = float(_rec(y_te, y_pred, zero_division=0))
    f1    = float(_f1(y_te, y_pred, zero_division=0))
    cm    = _cm(y_te, y_pred, labels=[0, 1]).tolist()

    # cm 格式：[[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    if fn > 0:
        warnings_list.append(
            f"⚠ FALSE NEGATIVE={fn}：{fn} 个实际可行样本被预测为不可行，"
            "若在优化中启用此过滤器，可能丢失有价值的 Aspen 工况。"
        )

    result = FeasibilityEvaluationResult(
        n_samples=n_valid,
        n_train=n_train,
        n_test=n_test,
        n_feasible=sum(y),
        n_infeasible=n_valid - sum(y),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        false_positive_count=fp,
        false_negative_count=fn,
        confusion_matrix=cm,
        threshold=config.threshold,
        model=config.model,
        var_names=used_var_names,
        warnings=warnings_list,
    )

    _log.info("可行性分类器评估完成：%s", result.summary())
    return result


# ---------------------------------------------------------------------------
# 数据库入口（可选）
# ---------------------------------------------------------------------------

def evaluate_feasibility_from_db(
    db: Any,  # SimulationDB，避免循环导入用 Any
    var_names: list[str],
    config: FeasibilityEvaluationConfig | None = None,
) -> FeasibilityEvaluationResult:
    """
    从 SimulationDB 导出数据后评估可行性分类器。

    使用 db.export_feasibility_dataset() 的输出，构造最小
    ProcessCase-like 对象供 evaluate_feasibility_classifier 消费。

    Parameters
    ----------
    db:
        已打开的 SimulationDB 实例。
    var_names:
        设计变量路径列表，严格限定评估所用特征维度，
        与优化器实际搜索空间保持一致。
    config:
        评估配置，None 时使用默认值。

    Notes
    -----
    不修改 db 内容，不修改 export_feasibility_dataset() 的行为。
    var_names 会通过 evaluate_feasibility_classifier 的同名参数传递，
    确保评估特征与线上特征维度一致。
    """
    rows = db.export_feasibility_dataset()
    mock_cases = [_MockCase(r) for r in rows]
    return evaluate_feasibility_classifier(mock_cases, config, var_names=var_names)


class _MockCase:
    """
    export_feasibility_dataset() 行的轻量代理，满足 evaluate_feasibility_classifier
    所需的 valid_for_classifier / feasible_label / design_vars 接口。
    """
    __slots__ = ("design_vars", "feasible_label", "valid_for_classifier")

    def __init__(self, row: dict[str, Any]) -> None:
        self.design_vars: dict[str, Any] = row.get("design_vars") or {}

        # P2：label 必须明确存在，不允许 None 或缺失被静默当成负样本
        label_raw = row.get("label")
        if label_raw is None:
            raise ValueError(
                f"_MockCase：行 '{row.get('case_id', '?')}' 缺少 label 字段，"
                "不能被伪装成有效样本。请检查 export_feasibility_dataset() 的输出。"
            )
        self.feasible_label: bool = bool(label_raw)
        # export_feasibility_dataset 已做状态过滤，design_vars 非空则视为 valid
        self.valid_for_classifier: bool = bool(self.design_vars)
