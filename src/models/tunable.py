"""
tunable.py — 变量发现与优化意图数据结构。

职责
----
为阶段 A（接入层）定义配置构建所需的中间数据模型：
  - TunableVariable   : 单个可调设计变量的描述（由 discover_tunables 工具产出）
  - ReadableTarget    : 单个可读输出节点的描述（目标函数或约束候选）
  - TunableReport     : 一次变量发现扫描的完整报告
  - GoalSpec          : 单个优化目标/约束的用户意图描述
  - OptimizationIntent: 用户对整个优化任务的意图描述
  - ConfigDraft       : 基于 TunableReport + OptimizationIntent 生成的配置草案

设计原则
--------
- 纯 Python dataclass，不依赖任何外部库（含 langchain / numpy 等）
- 不导入 aspen_driver、database、economics 等底层模块
- 所有字段均有类型标注，None 表示"未知/待填写"
- ConfigDraft.to_yaml_dict() 输出符合 pareto_config.yaml 顶层 schema

层级关系
--------
discover_tunables_tool → TunableReport
config_builder.build_config_draft(TunableReport, OptimizationIntent) → ConfigDraft
ConfigDraft.to_yaml_dict() → 符合 pareto_config.yaml 格式的字典；
    只有 continuous/integer 变量边界补齐后的完整草案，
    才可被 load_optimize_config / validate_config_tool 作为可执行优化配置解析
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# A1-1 TunableVariable — 单个可调设计变量
# ---------------------------------------------------------------------------

@dataclass
class TunableVariable:
    """
    一个可调设计变量的发现记录。

    Attributes
    ----------
    aspen_path:
        Aspen 树中的完整绝对路径，如 \\Data\\Blocks\\T0301\\Input\\BASIS_RR。
    semantic_role:
        语义角色名称，来自匹配的规则字段名，如 "reflux_ratio"、"feed_stage"。
        无规则匹配时为 ""。
    suggested_type:
        变量类型建议：continuous 或 integer。
    current_value:
        当前 Aspen 仿真文件中的参数值；扫描失败或无法读取时为 None。
    suggested_lower:
        建议下界；规则中有经验边界时填写，否则为 None（需用户手动填写）。
    suggested_upper:
        建议上界；规则中有经验边界时填写，否则为 None（需用户手动填写）。
    unit:
        物理单位字符串；无单位时为 "-"。
    confidence:
        置信度等级：
        - "high"   : 语义规则中有明确经验边界，可直接使用
        - "medium" : 规则匹配但边界是经验估算，建议用户确认
        - "low"    : 仅按路径模式推断，无规则边界数据，必须请用户填写
    reason:
        置信度评定的简短说明，供用户/agent 参考。
    """
    aspen_path: str
    semantic_role: str
    suggested_type: Literal["continuous", "integer"]
    current_value: float | None
    suggested_lower: float | None
    suggested_upper: float | None
    unit: str
    confidence: Literal["high", "medium", "low"]
    reason: str


# ---------------------------------------------------------------------------
# A1-2 ReadableTarget — 单个可读输出节点
# ---------------------------------------------------------------------------

@dataclass
class ReadableTarget:
    """
    一个可读输出节点的发现记录（目标函数或约束的候选来源）。

    Attributes
    ----------
    aspen_path:
        Aspen 树中的完整绝对路径。
    semantic_role:
        语义角色名称，如 "reboiler_duty"、"mass_frac_product"。
    candidate_use:
        候选用途：
        - "objective"  : 适合作为优化目标
        - "constraint" : 适合作为约束
        - "both"       : 两者均可
    unit:
        物理单位字符串。
    current_value:
        当前值；读取失败时为 None。
    """
    aspen_path: str
    semantic_role: str
    candidate_use: Literal["objective", "constraint", "both"]
    unit: str
    current_value: float | None


# ---------------------------------------------------------------------------
# A1-3 TunableReport — 变量发现报告
# ---------------------------------------------------------------------------

@dataclass
class TunableReport:
    """
    一次变量发现扫描的完整报告。

    由 discover_tunables_tool 调用 _scan_aspen_file + _build_tunable_variables
    + _build_readable_targets 后组装产出。

    Attributes
    ----------
    aspen_file:
        被扫描的 Aspen 文件路径（绝对路径）。
    aspen_file_hash:
        文件 MD5/SHA256 摘要，用于检测文件变更。
    tunable_variables:
        所有发现的可调设计变量列表（包括 confidence=low 的推断变量）。
    readable_targets:
        所有发现的可读输出节点列表。
    scan_warnings:
        扫描过程中遇到的非致命问题列表（节点读取失败、规则不匹配等）。
    semantic_coverage:
        语义覆盖率 = 被规则命中的节点数 / 总节点数（0.0~1.0）。
    """
    aspen_file: str
    aspen_file_hash: str
    tunable_variables: list[TunableVariable] = field(default_factory=list)
    readable_targets: list[ReadableTarget] = field(default_factory=list)
    scan_warnings: list[str] = field(default_factory=list)
    semantic_coverage: float = 0.0

    def get_high_confidence_vars(self) -> list[TunableVariable]:
        """返回置信度为 high 的设计变量列表。"""
        return [v for v in self.tunable_variables if v.confidence == "high"]

    def get_medium_confidence_vars(self) -> list[TunableVariable]:
        """返回置信度为 medium 的设计变量列表。"""
        return [v for v in self.tunable_variables if v.confidence == "medium"]

    def get_low_confidence_vars(self) -> list[TunableVariable]:
        """返回置信度为 low 的设计变量列表。"""
        return [v for v in self.tunable_variables if v.confidence == "low"]

    def get_targets_for_use(
        self, use: Literal["objective", "constraint", "both"]
    ) -> list[ReadableTarget]:
        """
        返回候选用途匹配的可读目标列表。

        "objective" 同时返回 candidate_use="objective" 和 "both" 的节点。
        "constraint" 同时返回 candidate_use="constraint" 和 "both" 的节点。
        "both" 只返回 candidate_use="both" 的节点。
        """
        if use == "both":
            return [t for t in self.readable_targets if t.candidate_use == "both"]
        return [t for t in self.readable_targets if t.candidate_use in (use, "both")]


# ---------------------------------------------------------------------------
# A1-4 GoalSpec — 单个目标/约束意图
# ---------------------------------------------------------------------------

@dataclass
class GoalSpec:
    """
    用户对单个优化目标或约束的意图描述。

    Attributes
    ----------
    metric:
        目标指标类型：
        - "TAC"      : 总年化成本（通过 tac.py 经济模型计算）
        - "emissions": CO₂ 排放量（通过 emissions.py 计算）
        - "purity"   : 产品纯度（质量分数或摩尔分数，作约束使用）
        - "yield"    : 产品产量
        - "flow"     : 质量/摩尔流量
        - "custom"   : 自定义，通过 custom_aspen_path 直接指定节点
    direction:
        优化方向：
        - "min" : 最小化
        - "max" : 最大化
    target_value:
        约束阈值，仅用于约束型目标（如 purity >= 0.9 时 target_value=0.9）；
        纯最大/最小化目标时为 None。
    custom_aspen_path:
        metric="custom" 时的 Aspen 树路径；其他 metric 时为 None。
    """
    metric: str
    direction: Literal["min", "max"]
    target_value: float | None = None
    custom_aspen_path: str | None = None


# ---------------------------------------------------------------------------
# A1-5 OptimizationIntent — 完整的优化意图
# ---------------------------------------------------------------------------

@dataclass
class OptimizationIntent:
    """
    用户对整个优化任务的意图描述。

    Attributes
    ----------
    goals:
        主要优化目标列表（多目标时为 Pareto 优化）。
    hard_constraints:
        硬约束列表（违反则判 infeasible）。
    n_initial:
        初始 DOE 采样点数；默认 20。
    n_iterations:
        贝叶斯优化迭代次数（不含初始 DOE）；默认 60。
    notes:
        用户额外备注或来源文本（如从 LLM 解析的原始意图字符串）。
    """
    goals: list[GoalSpec] = field(default_factory=list)
    hard_constraints: list[GoalSpec] = field(default_factory=list)
    n_initial: int = 20
    n_iterations: int = 60
    notes: str = ""


# ---------------------------------------------------------------------------
# A1-6 ConfigDraft — 配置草案
# ---------------------------------------------------------------------------

@dataclass
class ConfigDraft:
    """
    基于 TunableReport + OptimizationIntent 生成的优化配置草案。

    此对象可直接通过 to_yaml_dict() 转换为 pareto_config.yaml 格式的字典，
    经 validate_config_tool 校验后写入文件传给优化框架。

    Attributes
    ----------
    draft_id:
        草案唯一标识符，默认自动生成 UUID（短格式 8 字符）。
    aspen_file:
        关联的 Aspen 仿真文件路径。
    design_variables:
        设计变量列表，每项为符合 pareto_config.yaml design_variables 格式的字典。
        字段：name / aspen_path / type / lower_bound / upper_bound / initial_value / unit
    objectives:
        目标函数列表，每项符合 pareto_config.yaml objectives 格式。
        字段：name / type / aspen_path / minimize / unit
        （type=tac / emissions 时使用对应经济模型；type=aspen_path 时直接读节点）
    constraints:
        约束列表，每项符合 pareto_config.yaml constraints 格式。
        字段：name / aspen_path / operator / threshold
    optimizer:
        优化器配置字典，符合 pareto_config.yaml optimizer 格式。
    extraction:
        提取配置字典，符合 pareto_config.yaml extraction 格式。
    warnings:
        生成过程中发现的问题列表（如边界为 None、置信度不足等），
        需在进入优化前由用户确认。
    confidence_summary:
        对草案整体置信度的一句话总结，供 agent/前端展示。
    """
    draft_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    aspen_file: str = ""
    design_variables: list[dict[str, Any]] = field(default_factory=list)
    objectives: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    optimizer: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    confidence_summary: str = ""

    def to_yaml_dict(self) -> dict[str, Any]:
        """
        将草案序列化为符合 pareto_config.yaml schema 的字典。

        输出格式与 cases/demo_case/pareto_config.yaml 兼容。
        lower_bound / upper_bound 为 None 的变量保持 None 值（YAML 中显示为 null）。

        ⚠️  含 None 边界的草案是"未完成状态"：
        continuous / integer 变量的 lower_bound / upper_bound 为 None 时，
        load_optimize_config 会因无法构建搜索空间而报错，不能进入优化流程。
        必须先由 config_builder（A4）/ onboarding_agent（B1）通过用户交互补齐边界，
        再调用 validate_config_tool 校验通过后，才能传给优化框架。

        Returns
        -------
        dict
            包含 simulator / design_variables / objectives / constraints /
            extraction / optimizer 等顶级键的字典。
        """
        # 收集 output_paths（从目标函数和约束的 aspen_path 字段中提取）
        output_paths: list[str] = []
        for obj in self.objectives:
            p = obj.get("aspen_path")
            if p and p not in output_paths:
                output_paths.append(p)
        for con in self.constraints:
            p = con.get("aspen_path")
            if p and p not in output_paths:
                output_paths.append(p)

        result: dict[str, Any] = {
            "simulator": {
                "filepath": self.aspen_file,
                "visible": False,
                "suppress_dialogs": True,
                "require_type_library": True,
                "timeout": 300,
                "reinit": True,
                "verify_inputs": True,
                "input_rtol": 1.0e-6,
            },
            "design_variables": self.design_variables,
            "output_paths": output_paths,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "extraction": self.extraction,
            "optimizer": self.optimizer,
        }
        return result
