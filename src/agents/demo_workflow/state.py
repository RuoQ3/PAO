"""
state.py — demo_workflow 控制层状态结构定义。

只含纯 Python 数据类，不导入任何底层依赖（AspenDriver、SimulationDB、
NodeDB、src.workflows、src.aspen_driver、src.database、LangGraph）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

StepStatus = Literal["pending", "ok", "error", "skipped"]

# validate_config_tool 报告中表示 Python 解析链 fatal 失败的固定文本
_VALIDATE_FATAL_MARKERS: tuple[str, ...] = (
    "[失败] Python 解析失败",
    "解析失败 [",
)


# ---------------------------------------------------------------------------
# WorkflowStep — 单个编排步骤的执行记录
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """记录 workflow 中一个编排步骤的执行结果。

    Attributes:
        name:          步骤标识，与区段名一一对应（如 "load_config"、"run_case"）。
        status:        执行状态，取值 "pending" / "ok" / "error" / "skipped"。
        report:        该步骤 tool 的返回文本（或格式化后的摘要），默认空字符串。
        skipped_reason: status="skipped" 时说明跳过原因，其他状态下忽略此字段。
    """
    name: str
    status: StepStatus
    report: str = ""
    skipped_reason: str = ""

    @property
    def is_fatal(self) -> bool:
        """是否为阻断后续步骤的 fatal 错误（status="error"）。"""
        return self.status == "error"


# ---------------------------------------------------------------------------
# DemoWorkflowState — run_demo_case_workflow 的完整控制状态
# ---------------------------------------------------------------------------

@dataclass
class DemoWorkflowState:
    """run_demo_case_workflow 在执行过程中积累的控制层状态。

    设计约束：
    - 只存放 agent 控制层所需的字段，不存放 ProcessCase、SimulationResult
      或任何底层驱动、数据库对象。
    - 所有字段均为基本类型（str、bool、list[str]、None），方便序列化和测试。

    Attributes:
        case_config_path:    传入 workflow 的原始配置路径（用户输入，未解析）。
        resolved_config_path: 经 _resolve_config_path 解析后的绝对路径；
                              校验通过前为 None。
        optimizer_type:      从 YAML 读取的 optimizer.type（如 "pareto_bayesian"）；
                              读取失败时为空字符串。
        objective_names:     从 YAML 读取的目标函数名列表，用于 summarize_pareto_tool。
        db_path:             推断的 SimulationDB 路径（{config 目录}/output/simulation.db）；
                             仅 pareto_bayesian 分支使用，单次运行分支保持 None。
        node_db_path:        NodeDB 路径（{config 目录}/output/node.db），由调用方在
                             初始化后赋值；仅区段 6 的 query_node_db_tool 调用时使用，
                             不存在时该步骤标记为 skipped。
        session_id:          从 optimize_pareto_tool 返回文本提取的 session ID；
                             提取失败或非 pareto_bayesian 分支时为 None。
        diagnostic_case_ids: 区段 6 中用于诊断的失败工况 case_id 列表；
                             来源标注见 case_id_source 字段。
        case_id_source:      diagnostic_case_ids 的来源，
                             "db_query"（DB 直接查询，可信）或
                             "text_fallback"（文本解析 fallback，存在不确定性）或
                             ""（尚未填充）。
        steps:               已执行的步骤列表，按执行顺序追加。
        errors:              所有 status="error" 步骤的 report 摘要，用于快速判断。
        next_actions:        区段 8 生成的建议列表（自然语言字符串）。
        aborted:             workflow 是否已提前中止（配置错误或校验 fatal 时置 True）。
    """
    case_config_path: str

    resolved_config_path: str | None = None
    optimizer_type: str = ""
    objective_names: list[str] = field(default_factory=list)
    db_path: str | None = None
    node_db_path: str | None = None
    session_id: str | None = None
    diagnostic_case_ids: list[str] = field(default_factory=list)
    case_id_source: str = ""     # "db_query" | "text_fallback" | ""
    steps: list[WorkflowStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    aborted: bool = False

    # ------------------------------------------------------------------
    # 步骤管理
    # ------------------------------------------------------------------

    def add_step(
        self,
        name: str,
        status: StepStatus,
        report: str = "",
        skipped_reason: str = "",
    ) -> WorkflowStep:
        """追加一个执行步骤，status="error" 时自动将 report 记入 errors。

        Returns:
            新追加的 WorkflowStep 实例。
        """
        step = WorkflowStep(
            name=name,
            status=status,
            report=report,
            skipped_reason=skipped_reason,
        )
        self.steps.append(step)
        if status == "error" and report:
            self.errors.append(f"[{name}] {report}")
        return step

    def get_step(self, name: str) -> WorkflowStep | None:
        """按 name 查找最后一次出现的步骤（同名步骤重试时只返回最新）。

        Returns:
            找到时返回 WorkflowStep；未找到返回 None。
        """
        # 倒序遍历，返回最近一次同名步骤
        for step in reversed(self.steps):
            if step.name == name:
                return step
        return None

    def has_errors(self) -> bool:
        """是否存在任意 status="error" 的步骤。

        以 steps 列表为唯一数据源，不依赖 errors 缓存，
        确保即使 errors 列表被外部修改也不会产生误判。
        """
        return any(s.status == "error" for s in self.steps)

    # ------------------------------------------------------------------
    # 路径分支判断
    # ------------------------------------------------------------------

    @property
    def is_pareto_branch(self) -> bool:
        """是否走 pareto_bayesian 执行路径。"""
        return self.optimizer_type == "pareto_bayesian"

    # ------------------------------------------------------------------
    # validate_config_tool 输出解析
    # ------------------------------------------------------------------

    @staticmethod
    def is_validate_fatal(tool_output: str) -> bool:
        """判断 validate_config_tool 的输出是否为 fatal（需中止 workflow）。

        fatal 条件（满足任意一条）：
        1. 输出以 "错误：" 开头（tool 内部异常提前返回）
        2. 输出包含 "[失败] Python 解析失败"（load_optimize_config 失败）
        3. 输出包含 "解析失败 ["（【Python 解析】区段失败行）

        非 fatal 的警告类输出（含 "[警告]" 结论行）不触发中止。
        """
        if tool_output.startswith("错误："):
            return True
        return any(marker in tool_output for marker in _VALIDATE_FATAL_MARKERS)
