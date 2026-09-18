"""
kinetics.py — 反应动力学参数发现与拟合的数据结构。

职责
----
为 kinetics_advisor（动力学顾问 agent）定义中间数据模型：
  - KineticsVarMeta   : 单个动力学参数（Ea/k0）的元信息，供冷启动边界推荐使用
  - KineticsDataPoint : 单个 (温度, 速率常数) 实验观测点
  - KineticsFitPlan   : 拟合任务的显式计划（数据校验、模型选择），拟合前生成
  - KineticsFitResult : 确定性数值拟合的结果（Ea/k0 + 统计诊断）
  - KineticsRecommendation : 冷启动推荐与拟合结果的统一输出容器

设计原则
--------
- 纯 Python dataclass，不依赖任何外部库（含 numpy / scipy / langchain 等）
- 不导入 aspen_driver、database 等底层模块
- 所有字段均有类型标注，None 表示"未知/待填写"

与 tunable.py 的关系
--------------------
kinetics_advisor 是独立子领域（反应动力学），不复用 tunable.py 的
TunableVariable，因为动力学参数（Ea/k0）挂在 Aspen 的 \\Data\\Reactions\\
子树下，而不是某个 Block 之下——两者的发现路径和语义完全不同。
KineticsRecommendation 最终以 (lower, upper, initial_value) 的形式，
和其他设计变量一样进入 ConfigDraft.design_variables，仍需经过
write_feasibility_node + human_confirm_node 才能进入优化循环。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# K1-1 KineticsVarMeta — 单个动力学参数的元信息（冷启动边界推荐用）
# ---------------------------------------------------------------------------

@dataclass
class KineticsVarMeta:
    """
    单个动力学参数（活化能 Ea 或频率因子 k0）的输入元信息。

    Attributes
    ----------
    name:
        变量标识符（与推荐结果 name 对应，通常是 YAML name 或 aspen_path）。
    param_type:
        参数类型："activation_energy"（Ea）或 "pre_exponential_factor"（k0）。
    unit:
        物理单位，如 "kJ/mol" / "cal/mol"（Ea）或 "1/s" / "m3/kmol/s"（k0，
        与反应级数相关，未知填 ""）。
    current_value:
        当前 Aspen 文件中的参数值；无初值（新增反应）时为 None。
    reaction_order:
        反应级数（用于判断 k0 的量纲和合理数量级）；未知为 None。
    temperature_range:
        该反应涉及的操作温度范围 (T_lo, T_hi)，单位 K；用于判断 Ea 敏感度
        （温度范围窄时，Ea 对速率的影响更容易被误判，边界应更保守）；
        未知为 None。
    """
    name: str
    param_type: Literal["activation_energy", "pre_exponential_factor"]
    unit: str = ""
    current_value: float | None = None
    reaction_order: float | None = None
    temperature_range: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# K1-2 KineticsDataPoint — 单个实验观测点
# ---------------------------------------------------------------------------

@dataclass
class KineticsDataPoint:
    """
    单个 (温度, 速率常数) 实验观测点，用于 Arrhenius 拟合。

    Attributes
    ----------
    temperature:
        绝对温度，单位 K。必须为正数。
    rate_constant:
        该温度下的速率常数 k，单位由反应级数决定（用户需自行统一单位，
        本模块不做单位换算）。必须为正数（取对数线性化要求）。
    weight:
        该点在拟合中的权重，默认 1.0（等权重）。用于反映实验点的置信度差异
        （如重复测量次数、仪器精度），None 等价于 1.0。
    source_note:
        数据来源说明（如"实验批次 A，2026-03"），供报告展示，不参与计算。
    """
    temperature: float
    rate_constant: float
    weight: float | None = None
    source_note: str = ""


# ---------------------------------------------------------------------------
# K1-3 KineticsFitPlan — 拟合任务的显式计划
# ---------------------------------------------------------------------------

@dataclass
class KineticsFitPlan:
    """
    拟合任务的显式计划，在真正执行数值拟合前生成（deerflow 式 plan 阶段）。

    把"能不能拟合、该怎么拟合"的判断从拟合函数内部拆出来，使数据量不足、
    温度范围过窄等问题在规划阶段就能被发现并反馈给用户，而不是拟合完
    才发现结果不可信。

    Attributes
    ----------
    model_form:
        拟合使用的模型形式：
        - "arrhenius_simple" : k = k0 * exp(-Ea/RT)，直接对 (T, k) 数据拟合
    data_source:
        数据来源说明，如 "user_upload"。
    n_data_points:
        输入数据点数量。
    temperature_span:
        数据覆盖的温度范围 (T_min, T_max)，单位 K。
    can_fit:
        是否满足拟合的最低条件（数据点数 >= 3 且非全部同温）。
        False 时 fit_kinetics_params 应拒绝拟合并返回 rejection_reason。
    rejection_reason:
        can_fit=False 时的原因说明；can_fit=True 时为 ""。
    warnings:
        规划阶段发现的非致命问题（如温度范围过窄），不阻断拟合但需提示用户。
    steps:
        计划执行的步骤说明列表，供日志/报告展示，如
        ["数据校验", "线性化初值估计", "非线性精修", "残差诊断"]。
    """
    model_form: Literal["arrhenius_simple"] = "arrhenius_simple"
    data_source: str = "user_upload"
    n_data_points: int = 0
    temperature_span: tuple[float, float] | None = None
    can_fit: bool = False
    rejection_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# K1-4 KineticsFitResult — 确定性数值拟合结果
# ---------------------------------------------------------------------------

@dataclass
class KineticsFitResult:
    """
    Arrhenius 拟合的完整数值结果（由 fitting.py 产出，纯数值代码，无 LLM 参与）。

    Attributes
    ----------
    ea:
        拟合得到的活化能，单位 J/mol（内部统一用 SI，展示层换算为 kJ/mol）。
        拟合失败时为 None。
    ea_stderr:
        Ea 的拟合标准误（来自协方差矩阵），单位与 ea 一致；无法估计时为 None。
    k0:
        拟合得到的频率因子（与输入 rate_constant 单位一致）。拟合失败时为 None。
    k0_stderr:
        k0 的拟合标准误；无法估计时为 None。
    r_squared:
        线性化阶段 ln(k) vs 1/T 的决定系数，用于快速判断数据线性度；
        取值范围理论上 (-inf, 1.0]，越接近 1 越好。
    residuals:
        非线性拟合（若执行）在原始 k 空间的残差列表，与输入数据点一一对应。
    n_points:
        参与拟合的数据点数。
    fit_quality:
        拟合质量分级："good"（r_squared >= 0.95）/ "marginal"（0.8~0.95）/
        "poor"（< 0.8 或拟合失败）。用于驱动 agent 层是否需要向用户追加提问。
    warnings:
        拟合过程中发现的问题（如 Ea 超出化学反应典型范围 40~400 kJ/mol，
        数据点集中在窄温度区间等），供人工复核。
    method:
        实际使用的拟合方法："linear"（仅线性化，未做非线性精修）或
        "nonlinear"（线性化给初值 + curve_fit 精修）。
    """
    ea: float | None
    ea_stderr: float | None
    k0: float | None
    k0_stderr: float | None
    r_squared: float | None
    residuals: list[float] = field(default_factory=list)
    n_points: int = 0
    fit_quality: Literal["good", "marginal", "poor"] = "poor"
    warnings: list[str] = field(default_factory=list)
    method: Literal["linear", "nonlinear"] = "linear"


# ---------------------------------------------------------------------------
# K1-5 KineticsRecommendation — 统一输出容器
# ---------------------------------------------------------------------------

@dataclass
class KineticsRecommendation:
    """
    kinetics_advisor 两条入口（冷启动边界推荐 / 数据拟合）的统一输出。

    无论走哪条路径，最终都产出这个结构，供上层决定如何写入 ConfigDraft
    的 design_variables（initial_value / lower_bound / upper_bound）。
    该结果不会自动写入草案——写入前仍需经过 write_feasibility_node 的
    COM 试写验证和 human_confirm_node 的用户确认，这是硬约束。

    Attributes
    ----------
    name:
        变量标识符，与 KineticsVarMeta.name 或用户输入对应。
    param_type:
        "activation_energy" 或 "pre_exponential_factor"。
    source:
        推荐来源："fit"（数值拟合）/ "heuristic"（规则兜底）/ "llm"（LLM 冷启动推理）。
    initial_value:
        建议的初始值；fit 来源时为拟合结果本身。
    suggested_lower / suggested_upper:
        建议的搜索边界。
    confidence:
        置信度等级，含义与 TunableVariable.confidence 一致：
        - "high"   : 拟合质量好（fit_quality="good"）或规则边界可靠
        - "medium" : 拟合质量一般或规则估算
        - "low"    : 拟合失败/数据不足，仅凭物理常识给出保守范围
    reason:
        推荐依据的简短说明。
    fit_result:
        来源为 "fit" 时的完整拟合结果；其他来源为 None。
    warnings:
        需要用户关注的问题列表。
    """
    name: str
    param_type: Literal["activation_energy", "pre_exponential_factor"]
    source: Literal["fit", "heuristic", "llm"]
    initial_value: float | None
    suggested_lower: float | None
    suggested_upper: float | None
    confidence: Literal["high", "medium", "low"]
    reason: str
    fit_result: KineticsFitResult | None = None
    warnings: list[str] = field(default_factory=list)
