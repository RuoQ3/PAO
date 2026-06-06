"""
feasibility.py — 可行性分类器模块。

在 Aspen Plus 运行前对候选设计变量点做二分类筛选，将预测为不可行的候选
提前剔除，减少无效仿真次数。

架构定位
---------
本模块独立于优化循环，不接入 optimize_case.py，不改数据库 schema，不依赖
Aspen driver。使用方自行调用 fit / predict_proba / screen 三个接口。

训练数据来源
-----------
SimulationDB.export_feasibility_dataset() 的返回格式：
    [{"case_id": "...", "design_vars": {...}, "label": bool, "status": "..."}]

其中 label=True 为可行正样本（对应 ProcessCase.success=True），
label=False 为负样本（sim_failed / infeasible / objective_error / constraint_error）。

支持模型
--------
  "extra_trees"   — sklearn ExtraTreesClassifier（默认）
  "random_forest" — sklearn RandomForestClassifier
  "random"        — 始终跳过训练，fit 返回 False，predict_proba 返回 1.0，
                    用于调试或禁用分类器场景

sklearn 不可用时，所有训练型模型均静默回退（fit 返回 False），不抛异常。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# sklearn 懒加载，不可用时不影响模块导入
try:
    from sklearn.ensemble import ExtraTreesClassifier as _ETC
    from sklearn.ensemble import RandomForestClassifier as _RFC
    _HAS_SKLEARN = True
except ImportError:
    _ETC = None   # type: ignore[assignment,misc]
    _RFC = None   # type: ignore[assignment,misc]
    _HAS_SKLEARN = False
    _log.debug("sklearn 不可用，FeasibilityClassifier 将保持未训练状态。")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityConfig:
    """
    可行性分类器配置。

    Attributes
    ----------
    enabled:
        是否启用可行性过滤；False 时 fit / predict_proba / screen 均走 fallback。
    model:
        使用的分类模型。MVP 支持 ``"extra_trees"``、``"random_forest"``、``"random"``。
    min_samples:
        训练所需的最少有效样本数（正负各有至少 1 个且总量 >= min_samples）。
    threshold:
        screen 时接受候选的最低可行概率；低于此值的候选被过滤。
    candidate_pool_size:
        优化器生成候选池大小（本模块保存配置，不直接使用）。
    random_seed:
        sklearn 模型的随机种子；None 表示不固定。
    """
    enabled: bool = False
    model: str = "extra_trees"
    min_samples: int = 10
    threshold: float = 0.5
    candidate_pool_size: int = 200
    random_seed: int | None = None


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------

class FeasibilityClassifier:
    """
    可行性二分类器，用于在 Aspen 仿真前筛选候选设计变量点。

    Parameters
    ----------
    config:
        FeasibilityConfig 实例。
    """

    def __init__(self, config: FeasibilityConfig) -> None:
        self._config = config
        self._model: Any = None          # 训练好的 sklearn 模型
        self._var_names: list[str] = []  # fit 时保存的变量顺序

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #

    @property
    def is_trained(self) -> bool:
        """True 当且仅当分类器已成功训练（_model 非 None）。"""
        return self._model is not None

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #

    def fit(
        self,
        rows: list[dict[str, Any]],
        var_names: list[str],
    ) -> bool:
        """
        用历史仿真数据训练可行性分类器。

        Parameters
        ----------
        rows:
            来自 SimulationDB.export_feasibility_dataset() 的记录列表。
            每行包含 ``design_vars`` (dict) 和 ``label`` (bool)。
        var_names:
            设计变量名列表，决定特征向量的顺序。

        Returns
        -------
        bool
            训练成功返回 True；以下情况返回 False（不抛异常）：
            - config.enabled=False
            - model="random"
            - sklearn 不可用
            - 有效样本数 < min_samples
            - 正负样本不全（单一类别无法训练二分类器）
            - 训练过程异常
        """
        # 每次调用 fit 都先清空旧模型，确保失败路径后 is_trained=False
        self._model = None
        self._var_names = []

        # 配置关闭或 random 模式直接跳过
        if not self._config.enabled:
            _log.debug("FeasibilityClassifier：enabled=False，跳过训练。")
            return False
        if self._config.model == "random":
            _log.debug("FeasibilityClassifier：model='random'，跳过训练。")
            return False
        if not _HAS_SKLEARN:
            _log.warning("FeasibilityClassifier：sklearn 不可用，跳过训练。")
            return False

        # 提取有效样本
        X: list[list[float]] = []
        y: list[int] = []
        for row in rows:
            dv = row.get("design_vars", {})
            # 必须包含所有 var_names
            try:
                feature = [float(dv[v]) for v in var_names]
            except (KeyError, TypeError, ValueError) as exc:
                _log.debug(
                    "FeasibilityClassifier.fit：跳过行 '%s'，变量缺失或无法转 float：%s",
                    row.get("case_id", "?"), exc,
                )
                continue
            X.append(feature)
            y.append(1 if bool(row.get("label", False)) else 0)

        # 样本数检查
        if len(X) < self._config.min_samples:
            _log.warning(
                "FeasibilityClassifier：有效样本数 %d < min_samples=%d，跳过训练。",
                len(X), self._config.min_samples,
            )
            return False

        # 正负样本检查（二分类器必须两类都有）
        unique_labels = set(y)
        if len(unique_labels) < 2:
            _log.warning(
                "FeasibilityClassifier：样本标签只有 %s，正负样本不全，跳过训练。",
                unique_labels,
            )
            return False

        # 构建并训练模型
        try:
            clf = self._build_model()
            clf.fit(X, y)
        except Exception as exc:
            _log.warning("FeasibilityClassifier：训练异常，跳过。%s", exc)
            return False

        self._model = clf
        self._var_names = list(var_names)
        _log.debug(
            "FeasibilityClassifier：训练完成，样本数=%d，正样本=%d，负样本=%d。",
            len(y), y.count(1), y.count(0),
        )
        return True

    def _build_model(self) -> Any:
        """根据 config.model 实例化 sklearn 分类器。"""
        seed = self._config.random_seed
        if self._config.model == "random_forest":
            return _RFC(random_state=seed)
        # 默认 extra_trees
        return _ETC(random_state=seed)

    # ------------------------------------------------------------------ #
    # 预测
    # ------------------------------------------------------------------ #

    def predict_proba(self, design_vars: dict[str, Any]) -> float:
        """
        预测单个设计变量点的可行概率。

        Parameters
        ----------
        design_vars:
            {变量名: 值} 字典。

        Returns
        -------
        float
            可行概率，clamp 到 [0.0, 1.0]。
            - 未训练时返回 1.0（不阻塞，回退为不过滤）。
            - 变量缺失或无法转 float 时返回 0.0。
            - 预测异常时返回 1.0 并记录 warning。
        """
        if not self.is_trained:
            return 1.0

        try:
            feature = [[float(design_vars[v]) for v in self._var_names]]
        except (KeyError, TypeError, ValueError) as exc:
            _log.debug(
                "FeasibilityClassifier.predict_proba：变量缺失或无法转 float，返回 0.0。%s", exc
            )
            return 0.0

        try:
            proba_array = self._model.predict_proba(feature)
            # proba_array shape: (1, n_classes)；找正类（label=1）的列
            classes = list(self._model.classes_)
            pos_idx = classes.index(1)
            prob = float(proba_array[0][pos_idx])
        except Exception as exc:
            _log.warning("FeasibilityClassifier.predict_proba：预测异常，返回 1.0。%s", exc)
            return 1.0

        return max(0.0, min(1.0, prob))

    # ------------------------------------------------------------------ #
    # 批量筛选
    # ------------------------------------------------------------------ #

    def screen(
        self,
        candidates: list[dict[str, Any]],
        fallback_top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """
        对候选设计变量点做批量可行性筛选。

        Parameters
        ----------
        candidates:
            候选点列表，每个元素是 {变量名: 值} 字典。
        fallback_top_k:
            当没有候选超过 threshold 时，返回概率最高的前 k 个。
            <= 0 时按 1 处理。

        Returns
        -------
        list[dict]
            筛选后的候选列表（新 dict，不修改原输入）。
            每个候选附加 ``"_predicted_feasible": float``（可行概率）。
            优先返回 probability >= threshold 的候选，按概率降序。
            无超过阈值的候选时，返回概率最高的 max(1, fallback_top_k) 个。
        """
        if not candidates:
            return []

        # 未训练时返回原列表的副本（不附加 _predicted_feasible）
        if not self.is_trained:
            return [dict(c) for c in candidates]

        k = max(1, fallback_top_k)
        threshold = self._config.threshold

        # 计算每个候选的可行概率，构造带分数的新 dict
        scored: list[dict[str, Any]] = []
        for cand in candidates:
            prob = self.predict_proba(cand)
            scored.append({**cand, "_predicted_feasible": prob})

        # 按概率降序
        scored.sort(key=lambda d: d["_predicted_feasible"], reverse=True)

        # 优先返回超过阈值的候选
        above = [d for d in scored if d["_predicted_feasible"] >= threshold]
        if above:
            return above

        # fallback：返回概率最高的 k 个
        return scored[:k]
