"""
coordinator.py — 顶层意图编排的数据结构。

职责
----
为 coordinator agent（意图识别 → 任务拆解 → 执行分发）定义中间数据模型：
  - IntentType         : 可路由到的子 agent/工具枚举
  - IntentStep         : 单个已分类的意图步骤（含从文本抽取的槎位）
  - IntentClassification : 一次意图分类的完整结果（可含多个步骤）
  - TaskSpec            : 单个可执行任务（已补全技术参数 + 前提校验结果）
  - Plan                : 一次任务拆解的完整结果
  - CoordinatorStep / ExecutionState : 执行过程的记录（复用 demo_workflow 的
    WorkflowStep 范式，但独立定义以避免跨子系统依赖）

设计原则
--------
- 纯 Python dataclass，不依赖任何外部库（含 langchain / numpy 等）
- 不导入 aspen_driver、database、任何具体子 agent 模块
- 所有字段均有类型标注，None 表示"未知/待填写"

层级关系
--------
用户自由文本 → IntentClassification（intent.py）
IntentClassification + 环境参数 → Plan（planner.py）
Plan → ExecutionState（dispatcher.py，依次执行 TaskSpec，产出 CoordinatorStep 列表）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# C1-1 IntentType — 可路由的子 agent/工具枚举
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    """
    Coordinator 可识别并路由的任务类型。

    每个取值对应一个明确的执行目标（子 agent 入口函数或 LangChain tool）：
      ONBOARD_NEW_CASE           -> src.agents.onboarding_agent.run_onboarding
      RECOMMEND_VAR_BOUNDS       -> src.agents.boundary_advisor.recommend_boundaries_agent
      FIT_KINETICS_PARAMS        -> src.agents.kinetics_advisor.fit_kinetics_params
      RECOMMEND_KINETICS_BOUNDS  -> src.agents.kinetics_advisor.recommend_kinetics_bounds_agent
      DIAGNOSE_RESULTS           -> src.agents.process_advisor.run_process_advisor_agent
      RUN_OPTIMIZATION           -> src.agents.tools.optimize_pareto.optimize_pareto_tool
      GENERAL_CHAT               -> 无子 agent，直接由 LLM 用化工/优化领域背景回答，
                                     不驱动任何工具、不产出文件路径。用户是在问概念性
                                     问题、讨论方案、或单纯聊天，而不是要执行一个具体动作。
      CLARIFY_NEEDED             -> 无执行目标，"看起来想做某件事但不确定是哪种任务"时
                                     使用；与 GENERAL_CHAT 的区别见 prompts.py 的分类说明。
    """
    ONBOARD_NEW_CASE = "onboard_new_case"
    RECOMMEND_VAR_BOUNDS = "recommend_var_bounds"
    FIT_KINETICS_PARAMS = "fit_kinetics_params"
    RECOMMEND_KINETICS_BOUNDS = "recommend_kinetics_bounds"
    DIAGNOSE_RESULTS = "diagnose_results"
    RUN_OPTIMIZATION = "run_optimization"
    GENERAL_CHAT = "general_chat"
    CLARIFY_NEEDED = "clarify_needed"


# ---------------------------------------------------------------------------
# C1-2 IntentStep — 单个已分类的意图步骤
# ---------------------------------------------------------------------------

@dataclass
class IntentStep:
    """
    意图分类产出的单个步骤。

    一次用户请求可能被拆解为多个步骤（如"先拟合动力学参数，再帮我推荐边界"），
    IntentStep 只负责"这一步该路由到哪里、从文本里抓到了什么槎位"，
    不做任何技术参数补全——那是 planner.py 的职责。

    Attributes
    ----------
    intent_type:
        路由目标。
    raw_slots:
        从用户文本中抽取的原始槎位，键为语义名（如 "context"、"aspen_file_hint"、
        "case_config_path"），值均为字符串（未做类型转换，留给 planner 处理）。
        LLM 抽取失败或字段未提及时，对应键不出现（不是空字符串）。
    confidence:
        本步骤分类的置信度：
        - "high"   : 用户明确说明了任务类型和目标对象
        - "medium" : 任务类型明确但目标对象需要 Planner 从环境参数中推断
        - "low"    : 分类基于模糊匹配，建议在 Dispatcher 执行前提示用户确认
    reason:
        分类依据的简短说明（供日志/报告展示）。
    """
    intent_type: IntentType
    raw_slots: dict[str, str] = field(default_factory=dict)
    confidence: Literal["high", "medium", "low"] = "medium"
    reason: str = ""


# ---------------------------------------------------------------------------
# C1-3 IntentClassification — 一次意图分类的完整结果
# ---------------------------------------------------------------------------

@dataclass
class IntentClassification:
    """
    一次用户请求的完整意图分类结果。

    Attributes
    ----------
    steps:
        已分类的步骤列表，按用户描述的执行顺序排列。
        单一意图的请求只含一个元素；复合请求含多个元素。
        分类完全失败时，steps 含一个 intent_type=CLARIFY_NEEDED 的步骤，
        不会返回空列表（空列表意味着"什么都不用做"，语义上是错误的）。
    notes:
        对整体请求的一句话总结（如"用户想先拟合再推荐边界，两步操作"）。
    used_llm:
        本次分类是否真的用上了大模型；False 表示走了规则兜底或直接降级。
    warnings:
        分类过程中的问题列表（如"LLM 返回的 JSON 缺少字段，已忽略该步骤"）。
    """
    steps: list[IntentStep] = field(default_factory=list)
    notes: str = ""
    used_llm: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# C1-4 TaskSpec — 单个可执行任务
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """
    Planner 产出的单个可执行任务：已补全技术参数、已做前提校验。

    Attributes
    ----------
    task_id:
        任务标识符，格式 "{序号}_{intent_type.value}"，如 "0_recommend_var_bounds"。
    intent_type:
        路由目标，与来源 IntentStep 一致。
    kwargs:
        已补全好的、可直接传给目标函数的完整参数字典（技术参数如 node_db_path
        已由 Planner 自动推断填入；意图文本中的槎位如 context 也在其中）。
        ready=False 时此字段可能不完整，Dispatcher 不会使用它执行。
    ready:
        是否满足执行前提。False 表示存在阻塞（如缺少已生成的配置 YAML），
        Dispatcher 会跳过该任务，不会尝试用不完整参数硬调目标函数。
    blocking_reason:
        ready=False 时的具体原因，供用户理解"为什么这一步没跑"；
        ready=True 时为空字符串。
    source_step:
        产出本任务的原始 IntentStep，保留供调试/报告追溯。
    waiting_on_gapfill:
        本任务当前 ready=False 是否是因为"缺配置文件，但可以先自动接入
        补上"这一类可修复的缺口（而非无法修复的缺失，如完全没有 aspen_file）。
        True 时，Dispatcher 会在其前插入一个 gap-fill 用的
        ONBOARD_NEW_CASE 任务，执行成功后动态重新解析并解锁本任务，
        不需要用户额外发一轮请求。见 planner.py 的 gap-fill 逻辑。
    is_gapfill:
        本任务本身是否是 Planner/Dispatcher 自动插入的 gap-fill 任务
        （而非用户意图分类直接产出的）。用于报告展示时区分"用户要的步骤"
        和"系统自动补的前置步骤"。
    """
    task_id: str
    intent_type: IntentType
    kwargs: dict[str, Any] = field(default_factory=dict)
    ready: bool = True
    blocking_reason: str = ""
    source_step: IntentStep | None = None
    waiting_on_gapfill: bool = False
    is_gapfill: bool = False


# ---------------------------------------------------------------------------
# C1-5 Plan — 一次任务拆解的完整结果
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """
    Planner 产出的完整执行计划。

    Attributes
    ----------
    tasks:
        任务列表，与 IntentClassification.steps 顺序一一对应。
    warnings:
        拆解过程中的问题列表（如"检测到 CLARIFY_NEEDED，未生成可执行任务"）。
    """
    tasks: list[TaskSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# C1-6 CoordinatorStep / ExecutionState — 执行记录
# ---------------------------------------------------------------------------

StepStatus = Literal["pending", "ok", "error", "skipped"]


@dataclass
class CoordinatorStep:
    """
    记录 Dispatcher 执行单个 TaskSpec 的结果。

    字段含义与 src.agents.demo_workflow.state.WorkflowStep 一致（同一套
    "name/status/report/skipped_reason" 范式），独立定义以避免 coordinator
    模块跨子系统依赖 demo_workflow 包。

    Attributes
    ----------
    task_id:
        对应的 TaskSpec.task_id。
    intent_type:
        对应的路由目标，供报告展示。
    status:
        执行状态："pending" / "ok" / "error" / "skipped"。
    report:
        该任务目标函数的返回文本（或格式化后的摘要）。
    skipped_reason:
        status="skipped" 时的原因（通常直接取自 TaskSpec.blocking_reason）。
    produced_paths:
        本任务执行后新产出的文件路径，键为语义名（如 "case_config_path"、
        "db_path"），供连续对话模式下 ChatSessionState 记忆并在后续轮次
        自动填空。绝大多数任务不产出路径（如拟合类任务只产出数值结果），
        此时为空字典——不是所有 IntentType 都会写文件。
    """
    task_id: str
    intent_type: IntentType
    status: StepStatus
    report: str = ""
    skipped_reason: str = ""
    produced_paths: dict[str, str] = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        return self.status == "error"


@dataclass
class ExecutionState:
    """
    dispatch_plan 的完整执行结果。

    Attributes
    ----------
    steps:
        已执行（或跳过）的步骤列表，按 Plan.tasks 顺序追加。
    errors:
        所有 status="error" 步骤的 report 摘要。
    has_clarify:
        本次是否存在需要用户澄清的意图（即分类阶段产出了 CLARIFY_NEEDED），
        供上层 CLI 决定是否要追问用户而不是直接报"执行完成"。
    """
    steps: list[CoordinatorStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    has_clarify: bool = False

    def add_step(
        self,
        task_id: str,
        intent_type: IntentType,
        status: StepStatus,
        report: str = "",
        skipped_reason: str = "",
        produced_paths: dict[str, str] | None = None,
    ) -> CoordinatorStep:
        """追加一个执行步骤，status="error" 时自动将 report 记入 errors。"""
        step = CoordinatorStep(
            task_id=task_id, intent_type=intent_type, status=status,
            report=report, skipped_reason=skipped_reason,
            produced_paths=dict(produced_paths) if produced_paths else {},
        )
        self.steps.append(step)
        if status == "error" and report:
            self.errors.append(f"[{task_id}] {report}")
        return step

    def has_errors(self) -> bool:
        return any(s.status == "error" for s in self.steps)


# ---------------------------------------------------------------------------
# C1-7 ChatSessionState — 连续对话模式的会话记忆
# ---------------------------------------------------------------------------

@dataclass
class ChatSessionState:
    """
    连续对话模式（--chat）下跨多轮请求保留的会话状态。

    只记忆"文件路径"这一类技术参数，不记忆任何数值结果（如拟合出的
    Ea/k0）——数值层面的串联仍需用户看完上一轮结果后自行决定下一轮参数，
    这是与一次性模式（run_coordinator）共享的设计取舍，见
    src.agents.coordinator.agent 模块 docstring。

    Attributes
    ----------
    aspen_file:
        会话级常量，--chat 启动时从 CLI 参数固定下来，不会被任何任务覆盖
        （用户在一次会话里只针对一个 Aspen 文件工作；换文件应重启会话）。
    case_config_path:
        最近一次产出（或用户在某一轮显式提供）的优化配置 YAML 路径。
        典型来源：ONBOARD_NEW_CASE 任务执行后落盘的草案文件。
    db_path:
        最近一次产出的结果数据库路径。典型来源：RUN_OPTIMIZATION 任务。
    data_csv:
        用户在某一轮显式提供的实验数据 CSV 路径（拟合任务本身不产出新的
        CSV，此字段只会被用户输入更新，不会被任务执行结果更新）。
    turn_count:
        已完成的对话轮次数，供 CLI 展示/调试。
    """
    aspen_file: str
    case_config_path: str | None = None
    db_path: str | None = None
    data_csv: str | None = None
    turn_count: int = 0

    def apply_produced_paths(self, produced_paths: dict[str, str]) -> None:
        """把某个已完成任务产出的路径合并进会话记忆（覆盖同名旧值）。"""
        if "case_config_path" in produced_paths:
            self.case_config_path = produced_paths["case_config_path"]
        if "db_path" in produced_paths:
            self.db_path = produced_paths["db_path"]

    def as_display_dict(self) -> dict[str, str]:
        """返回当前记忆的路径快照，供 CLI `state` 元命令展示。"""
        return {
            "aspen_file": self.aspen_file,
            "case_config_path": self.case_config_path or "(未记录)",
            "db_path": self.db_path or "(未记录)",
            "data_csv": self.data_csv or "(未记录)",
            "turn_count": str(self.turn_count),
        }

    def to_context_summary(self) -> str:
        """
        把当前会话记忆渲染成一段自然语言摘要，供意图分类的 LLM 理解
        “这个/这个工艺/这个案例”之类的指代词，以及判断哪些前提已经满足
        （不需要因为“缺文件”就要求用户澄清——那是 Planner/Dispatcher
        的职责，见 planner.py 的 gap-fill 逻辑）。

        不包含任何数值结果，只描述文件是否存在，与本类“只记路径不记数值”
        的设计原则一致。
        """
        lines = [
            f"- Aspen 仿真文件：已提供（{self.aspen_file}），"
            "用户提到“这个/这个工艺/这个案例”等指代词应理解为此文件"
        ]
        if self.case_config_path:
            lines.append(f"- 优化配置草案：已生成（{self.case_config_path}）")
        else:
            lines.append("- 优化配置草案：尚未生成（若需要，可先接入生成，不必让用户手动提供）")
        if self.db_path:
            lines.append(f"- 结果数据库：已存在（{self.db_path}）")
        else:
            lines.append("- 结果数据库：尚无历史优化结果")
        if self.data_csv:
            lines.append(f"- 实验数据 CSV：已提供（{self.data_csv}）")
        return "\n".join(lines)
