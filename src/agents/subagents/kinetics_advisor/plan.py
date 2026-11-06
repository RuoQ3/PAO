"""
plan.py — 拟合任务的显式规划（deerflow 式 plan 阶段）。

职责（纯函数，不含 LLM、不含数值拟合，只做"能不能拟合、该怎么拟合"的判断）：
  - build_fit_plan：从原始数据点生成 KineticsFitPlan，在真正执行 fitting.py
    之前，把数据量不足、温度范围过窄等问题拦在最前面。

设计原则
--------
把"判断数据是否够拟合"和"真正拟合"分成两个独立步骤，而不是让 fitting.py
在拟合失败时才发现问题——这样调用方（agent.py）可以在 can_fit=False 时
直接向用户提问，不必先跑一次注定失败或不可信的拟合。

这是 deerflow 风格 Plan-Execute-Reflect 的 Plan 阶段：规划器只读数据的
统计特征（点数、温度分布），不做任何回归计算，可独立单测。
"""
from __future__ import annotations

from src.models.kinetics import KineticsDataPoint, KineticsFitPlan

# 最低数据点数：Arrhenius 两参数（Ea, k0）拟合本质欠定，< 3 点不接受。
_MIN_POINTS = 3

# 温度范围过窄的判定阈值：(T_max - T_min) / T_mean < 此比例时告警。
_NARROW_SPAN_RATIO = 0.05


def build_fit_plan(
    data_points: list[KineticsDataPoint],
    data_source: str = "user_upload",
) -> KineticsFitPlan:
    """
    从原始数据点生成拟合计划，判断是否满足拟合的最低条件。

    Parameters
    ----------
    data_points:
        用户提供的 (温度, 速率常数) 观测点列表。
    data_source:
        数据来源说明，写入计划供报告展示。

    Returns
    -------
    KineticsFitPlan
        can_fit=False 时，调用方应拒绝拟合并向用户展示 rejection_reason，
        不应继续调用 fitting.fit_arrhenius。
    """
    n = len(data_points)
    plan = KineticsFitPlan(
        model_form="arrhenius_simple",
        data_source=data_source,
        n_data_points=n,
    )

    if n == 0:
        plan.can_fit = False
        plan.rejection_reason = "未提供任何数据点，无法拟合"
        return plan

    if n < _MIN_POINTS:
        plan.can_fit = False
        plan.rejection_reason = (
            f"数据点数 {n} < {_MIN_POINTS}，Arrhenius 两参数（Ea, k0）拟合本质欠定，"
            "请补充更多不同温度下的实验点后重试"
        )
        return plan

    temps = [p.temperature for p in data_points]
    if any(t <= 0 for t in temps):
        plan.can_fit = False
        plan.rejection_reason = "存在非正温度值（温度必须是绝对温度 K，且为正数），请检查数据"
        return plan

    rates = [p.rate_constant for p in data_points]
    if any(k <= 0 for k in rates):
        plan.can_fit = False
        plan.rejection_reason = "存在非正速率常数（k 必须为正数，Arrhenius 线性化要求取对数），请检查数据"
        return plan

    t_min, t_max = min(temps), max(temps)
    plan.temperature_span = (t_min, t_max)

    if t_max == t_min:
        plan.can_fit = False
        plan.rejection_reason = (
            f"所有数据点温度均为 {t_min:g} K，温度无变化时无法区分 Ea 对速率的影响，"
            "请补充不同温度下的实验点"
        )
        return plan

    # 数据量和温度分布均满足最低条件，可以拟合
    plan.can_fit = True
    plan.steps = ["数据校验", "线性化初值估计（ln k vs 1/T）", "非线性精修（curve_fit）", "残差与合理性诊断"]

    t_mean = sum(temps) / n
    span_ratio = (t_max - t_min) / t_mean if t_mean > 0 else 0.0
    if span_ratio < _NARROW_SPAN_RATIO:
        plan.warnings.append(
            f"温度范围较窄（{t_min:.1f}~{t_max:.1f} K，跨度占均值的 "
            f"{span_ratio * 100:.1f}%），拟合出的 Ea 外推到该范围之外时可靠性较低，"
            "建议补充更宽温度范围的实验点"
        )

    if n < 5:
        plan.warnings.append(
            f"数据点数仅 {n} 个（略高于最低要求 {_MIN_POINTS}），拟合结果的统计置信度有限，"
            "建议数据点数达到 5 个以上以获得更稳健的标准误估计"
        )

    return plan
