"""
hitl_protocol.py — H4-0 HITL 交互契约

职责
----
定义 ``graph.py`` 两个 HITL 节点（``human_confirm_node`` / ``human_decide_node``）
的暂停/恢复交互契约，与传输层（FastAPI / Vue）完全解耦。
FastAPI 层只是这套契约的 HTTP/SSE 搬运工，不应包含任何业务逻辑。

公开 API
--------
  HitlPrompt   — 暂停态载荷：图暂停时向客户端描述"需要用户做什么"
  HitlResponse — 恢复态载荷：客户端收集用户输入后传回图，让图继续执行
  FinalResult  — 图运行结束时的最终产物摘要（非暂停态，无需再调用 resume）
  SessionError — 会话查找/状态异常时抛出

  start_session(aspen_file, intent_text, ...) -> tuple[str, HitlPrompt | FinalResult]
      启动新会话：创建 session_id，运行 onboarding 直至第一次 HITL 暂停（或直接完成），
      返回 (session_id, 初始暂停态/最终结果)。

  resume_session(session_id, response) -> HitlPrompt | FinalResult
      恢复已暂停会话：将用户输入注入图，运行直至下一次暂停（或完成）。

暂停/恢复约定
-----------
- 图以 LangGraph ``MemorySaver`` checkpoint 持久化；session_id 即 thread_id。
- ``interrupt_before=["human_confirm", "human_decide"]`` — 图在这两个节点前暂停。
- ``human_confirm`` 暂停时（phase="confirming"）：
    - 客户端展示 ``HitlPrompt.pending_bounds`` 中的变量，让用户确认/修改边界
    - 用户提交后构造 ``HitlResponse(confirmed_bounds={...})``
    - 内部映射为 ``Command(update={"user_feedback": {"bounds": confirmed_bounds}})``
- ``human_decide`` 暂停时（phase="deciding"）：
    - 客户端展示 ``HitlPrompt.pareto_summary`` 和分析摘要，让用户三选一
    - 用户提交后构造 ``HitlResponse(decision="continue"|"adjust"|"done")``
    - adjust 时可附带 ``edited_intent``，内部一并注入 ``intent_text``
    - 内部映射为 ``Command(update={"user_decision": ..., "intent_text": ...})``

验收标准
--------
- ``HitlPrompt`` / ``HitlResponse`` / ``FinalResult`` 可从此模块独立导入
- ``start_session`` / ``resume_session`` 有 docstring 说明契约边界
- 单元测试 ``tests/agents/test_hitl_protocol.py`` 全绿
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Union

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# H4-0-1  HitlPrompt — 暂停态载荷
# ---------------------------------------------------------------------------

@dataclass
class HitlPrompt:
    """
    图暂停时向客户端发送的载荷，描述"当前需要用户做什么"。

    Attributes
    ----------
    phase:
        当前暂停阶段：
        - "confirming" : 在 human_confirm 节点前暂停，需用户确认/补填变量边界
        - "deciding"   : 在 human_decide 节点前暂停，需用户决策 continue/adjust/done
    questions:
        需要用户关注的问题列表（中文问句）。confirming 阶段来自接入向导的提示；
        deciding 阶段来自分析报告的关键结论。
    config_draft_summary:
        配置草案摘要字典，含 draft_id / aspen_file / 目标函数列表 / 约束数量等。
        前端用于展示"当前优化配置快照"。
    pareto_summary:
        Pareto 优化结果摘要字典，仅在 phase="deciding" 时非 None，含：
        ``db_path``, ``iteration``, ``analysis_report``（分析文本）。
        confirming 阶段为 None（尚未完成优化）。
    pending_bounds:
        待用户确认的变量列表，每项包含：
        ``aspen_path``, ``name``, ``suggested_lower``, ``suggested_upper``,
        ``current_value``, ``unit``, ``confidence``, ``reason``。
        confirming 阶段：confidence != "high" 或边界为 None 的变量；
        deciding 阶段：空列表（边界已在上轮确认）。
    options:
        用户可选择的操作列表。
        confirming：["confirm"]（更换意图须重新调用 start_session，图路由不支持 confirm→onboarding）
        deciding  ：["continue", "adjust", "done"]
    """
    phase: str
    questions: list[str]
    config_draft_summary: dict
    pareto_summary: dict | None
    pending_bounds: list[dict]
    options: list[str]
    topology: dict = field(default_factory=dict)   # 流程拓扑 {nodes, edges}


# ---------------------------------------------------------------------------
# H4-0-2  HitlResponse — 恢复态载荷
# ---------------------------------------------------------------------------

@dataclass
class HitlResponse:
    """
    客户端收集用户输入后构造的恢复载荷，传给 resume_session 恢复图执行。

    Attributes
    ----------
    confirmed_bounds:
        用户确认的变量边界映射，格式 ``{aspen_path: [lo, hi]}``。
        仅在 phase="confirming" 时有意义；deciding 阶段应为空 dict。
        空 dict 表示"接受所有建议边界（包括 None），不做修改"。
    excluded_paths:
        用户主动排除的设计变量路径列表。
        上下界均留空时前端将该变量路径放入此列表，后端据此从 design_variables 中删除该变量。
        空列表表示不排除任何变量。
    decision:
        用户决策，仅在 phase="deciding" 时有意义：
        - "continue" : 继续再次优化（回到 confirming 阶段）
        - "adjust"   : 重新调整意图（回到 onboarding）
        - "done"     : 接受当前结果，结束会话
        confirming 阶段应为 None。
    edited_intent:
        用户重新编写的优化意图文本，仅在 decision="adjust" 时有意义。
        None 表示不修改意图，仍用上次的 intent_text 重跑 onboarding。
    """
    confirmed_bounds: dict[str, list[float]] = field(default_factory=dict)
    excluded_paths: list[str] = field(default_factory=list)
    decision: str | None = None
    edited_intent: str | None = None


# ---------------------------------------------------------------------------
# H4-0-3  FinalResult — 最终产物摘要
# ---------------------------------------------------------------------------

@dataclass
class FinalResult:
    """
    图运行结束（current_phase="done"）时向客户端返回的最终产物摘要。
    返回 FinalResult 意味着会话已关闭，不应再调用 resume_session。

    Attributes
    ----------
    session_id:
        会话标识符。
    db_path:
        SimulationDB SQLite 文件路径，None 表示优化从未完成。
    messages:
        图执行过程中积累的所有消息（含各阶段进度）。
    termination_reason:
        终止原因：None 表示用户主动终止（done）；
        其他值为非正常终止原因，如 "optimization_failed" / "confirm_validation_failed"。
    analysis_report:
        最后一次 process advisor 分析报告全文，空字符串表示从未分析。
    iteration:
        完成的优化轮次数。
    config_yaml_path:
        最后一次写出的配置 YAML 路径，None 表示配置从未通过验证。
    intent:
        本轮使用的 OptimizationIntent（从 onboarding_result.intent 取得）。
        供 generate_summary_report 生成第 0 章目标达成总览。None 表示意图未知。
    """
    session_id: str
    db_path: str | None
    messages: list[str]
    termination_reason: str | None
    analysis_report: str
    iteration: int
    config_yaml_path: str | None = None
    intent: Any = None


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class SessionError(Exception):
    """会话查找失败或状态非法时抛出。"""


# ---------------------------------------------------------------------------
# 模块级会话状态（内部，不对外暴露）
# ---------------------------------------------------------------------------

# 会话配置表：{session_id -> LangGraph thread config}
_SESSION_CONFIGS: dict[str, dict] = {}
_SESSION_LOCK = threading.Lock()

# 惰性初始化：_app / _checkpointer
_app = None
_checkpointer = None
_APP_LOCK = threading.Lock()


_ALLOWED_MSGPACK_MODULES = [
    ("src.models.tunable",                    "ConfigDraft"),
    ("src.models.tunable",                    "TunableVariable"),
    ("src.models.tunable",                    "ReadableTarget"),
    ("src.models.tunable",                    "TunableReport"),
    ("src.models.tunable",                    "GoalSpec"),
    ("src.models.tunable",                    "OptimizationIntent"),
    ("src.models.tunable",                    "WriteCheckResult"),
    ("src.models.tunable",                    "WriteFeasibilityReport"),
    ("src.models.tunable",                    "PrioritizationResult"),
    ("src.agents.onboarding_agent.agent",     "OnboardingResult"),
]


def _get_app():
    """惰性初始化并返回 LangGraph 编译图（全局单例）。"""
    global _app, _checkpointer
    if _app is not None:
        return _app
    with _APP_LOCK:
        if _app is not None:
            return _app
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from src.agents.graph import build_graph
        serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
        _checkpointer = MemorySaver(serde=serde)
        _app = build_graph(checkpointer=_checkpointer)
        _log.info("hitl_protocol: LangGraph app 初始化完成（MemorySaver，已注册 %d 个自定义类型）",
                  len(_ALLOWED_MSGPACK_MODULES))
    return _app


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _build_config_draft_summary(draft: Any) -> dict:
    """从 ConfigDraft 构建紧凑摘要字典，用于前端展示。"""
    if draft is None:
        return {}
    return {
        "draft_id": getattr(draft, "draft_id", ""),
        "aspen_file": getattr(draft, "aspen_file", ""),
        "n_design_variables": len(getattr(draft, "design_variables", [])),
        "objectives": [
            {
                "name": o.get("name", ""),
                "minimize": o.get("minimize", True),
                "unit": o.get("unit", "-"),
            }
            for o in getattr(draft, "objectives", [])
        ],
        "n_constraints": len(getattr(draft, "constraints", [])),
        "confidence_summary": getattr(draft, "confidence_summary", ""),
        "warnings": getattr(draft, "warnings", []),   # 全量传递，前端按需过滤
    }


# semantic_role → 中文显示名
_ROLE_LABELS: dict[str, str] = {
    "basis_reflux_ratio":   "回流比",
    "reflux_ratio":         "回流比",
    "bottoms_to_feed_ratio":"塔底采出比",
    "feed_stage":           "进料板位置",
    "column_pressure":      "操作压力",
    "reboiler_duty":        "再沸器热负荷",
    "condenser_duty":       "冷凝器热负荷",
    "nstage":               "理论板数",
    "column_diameter":      "塔径",
    "distillate_rate":      "塔顶采出量",
    "bottoms_rate":         "塔底采出量",
    "condenser_temperature":"冷凝器温度",
    "reboiler_temperature": "再沸器温度",
    # Flash / 闪蒸
    "flash_pressure":       "闪蒸压力",
    "flash_temperature":    "闪蒸温度",
    "flash_vapor_fraction": "气化率",
    # Pump / 泵
    "pump_pressure":        "出口压力",
    "pump_efficiency":      "泵效率",
    # HeatX / 换热器
    "hot_side_temperature": "热侧出口温度",
    "cold_side_temperature":"冷侧出口温度",
    "heat_duty":            "换热量",
    # Compressor
    "compressor_pressure":  "出口压力",
    "compressor_efficiency":"压缩效率",
    # Reactor
    "conversion":           "转化率",
    "reactor_temperature":  "反应温度",
    "reactor_pressure":     "反应压力",
}


def _make_display_name(aspen_path: str, semantic_role: str, draft_name: str) -> str:
    """
    构建前端友好的变量显示名称，格式：设备名 · 参数中文名

    优先级：
    1. ConfigDraft 里用户/规则给的 name（通常格式 T0301_BF）
    2. semantic_role 映射的中文名 + 路径提取的设备名
    3. 最后两段路径（原 shortPath 逻辑）
    """
    # 从路径提取设备名：\Data\Blocks\T0301\Input\BASIS_RR → T0301
    parts = aspen_path.replace("\\", "/").split("/")
    device = ""
    for i, seg in enumerate(parts):
        if seg.upper() in ("BLOCKS", "STREAMS") and i + 1 < len(parts):
            device = parts[i + 1]
            break

    role_label = _ROLE_LABELS.get(semantic_role, "")

    if draft_name:
        # draft_name 通常是 T0301_BF 这样的标识符，拼上中文语义更清晰
        if role_label and device:
            return f"{device} · {role_label}"
        return draft_name

    if role_label and device:
        return f"{device} · {role_label}"

    # 降级：取路径最后两段
    meaningful = [p for p in parts if p and p.upper() not in ("DATA", "INPUT", "OUTPUT", "MIXED")]
    return " · ".join(meaningful[-2:]) if len(meaningful) >= 2 else (meaningful[-1] if meaningful else aspen_path)


def _build_pending_bounds(
    config_draft: Any,
    onboarding_result: Any,
) -> list[dict]:
    """
    构建待用户确认的变量列表，以 ``TunableReport.tunable_variables`` 为主源。

    设计说明
    --------
    ``build_config_draft`` 的策略是：confidence=low 或 bounds 不完整的变量**不**进入
    ``config_draft.design_variables``（无法构建搜索空间）。若只遍历 design_variables，
    最需要用户补边界的变量反而不会出现——前端展示空列表，用户提交也被忽略。

    正确做法：以 ``tunable_report.tunable_variables`` 为主源遍历所有候选变量；
    再用 ``config_draft.design_variables`` 反查"已纳入草案的变量"以获取当前生效边界。
    两者的数据源一致，保证：
    - 需要用户确认的变量（low/medium/bounds=None）一定出现在列表里；
    - 已纳入草案且为 high confidence 的变量也出现（供前端只读参考），但不强制要求用户修改。

    纳入规则
    --------
    - confidence == "low"           → 必须（无规则边界，前端强制填写）
    - confidence == "medium"        → 建议（经验估算，前端预填+提示确认）
    - confidence == "high" 且边界完整 → 参考（前端可展示但不需要用户操作）
    - confidence == "high" 且边界 None → 不应发生（语义规则有 high 但无边界），按 medium 处理

    Parameters
    ----------
    config_draft:
        ConfigDraft 实例，可为 None。仅用于反查已纳入草案的变量的生效边界。
    onboarding_result:
        OnboardingResult 实例，可为 None。
        主数据源：``onboarding_result.tunable_report.tunable_variables``。

    Returns
    -------
    list[dict]
        每项含 aspen_path / name / suggested_lower / suggested_upper /
        current_value / unit / confidence / reason / in_draft（是否已纳入草案）。
    """
    # 从 config_draft.design_variables 构建"已纳入草案"的变量查找表
    draft_bounds: dict[str, dict] = {}
    if config_draft is not None:
        for dv in getattr(config_draft, "design_variables", []):
            path = dv.get("aspen_path", "")
            if path:
                draft_bounds[path] = dv

    # 若无 tunable_report，退化为只遍历 design_variables（向后兼容）
    tunable_vars = []
    if onboarding_result is not None:
        report = getattr(onboarding_result, "tunable_report", None)
        if report is not None:
            tunable_vars = getattr(report, "tunable_variables", [])

    if not tunable_vars and config_draft is not None:
        # 退化路径：无 TunableReport，只从草案中取已有变量
        pending: list[dict] = []
        for dv in getattr(config_draft, "design_variables", []):
            path = dv.get("aspen_path", "")
            lo = dv.get("lower_bound")
            hi = dv.get("upper_bound")
            if lo is None or hi is None:
                pending.append({
                    "aspen_path": path,
                    "name": dv.get("name", ""),
                    "suggested_lower": lo,
                    "suggested_upper": hi,
                    "current_value": dv.get("initial_value"),
                    "unit": dv.get("unit", "-"),
                    "confidence": "unknown",
                    "reason": "无 TunableReport，无法确认置信度",
                    "in_draft": True,
                })
        return pending

    # 主路径：以 tunable_report.tunable_variables 为主源
    pending = []
    for tv in tunable_vars:
        path = getattr(tv, "aspen_path", "")
        confidence = getattr(tv, "confidence", "unknown")
        reason = getattr(tv, "reason", "")
        current_val = getattr(tv, "current_value", None)
        unit = getattr(tv, "unit", "-")

        # 从草案反查生效边界（已纳入草案的变量用草案边界；未纳入的用建议边界）
        in_draft = path in draft_bounds
        if in_draft:
            dv = draft_bounds[path]
            lo = dv.get("lower_bound") if dv.get("lower_bound") is not None else getattr(tv, "suggested_lower", None)
            hi = dv.get("upper_bound") if dv.get("upper_bound") is not None else getattr(tv, "suggested_upper", None)
            name = dv.get("name", "")
            unit = dv.get("unit", unit)
        else:
            lo = getattr(tv, "suggested_lower", None)
            hi = getattr(tv, "suggested_upper", None)
            name = ""

        # 纳入待确认列表：非 high confidence，或边界不完整
        # high confidence 且边界完整的变量也纳入（供前端只读参考），
        # 由前端通过 confidence 字段决定是否强制用户操作。
        priority_score  = getattr(tv, "priority_score", 0.5)
        priority_reason = getattr(tv, "priority_reason", "")
        semantic_role   = getattr(tv, "semantic_role", "")
        display_name    = _make_display_name(path, semantic_role, name)

        pending.append({
            "aspen_path":      path,
            "name":            name,
            "display_name":    display_name,
            "suggested_lower": lo,
            "suggested_upper": hi,
            "current_value":   current_val,
            "unit":            unit,
            "confidence":      confidence,
            "reason":          reason,
            "in_draft":        in_draft,
            "priority_score":  priority_score,
            "priority_reason": priority_reason,
        })

    return pending


def _build_hitl_prompt(state_values: dict, next_node: str) -> "HitlPrompt":
    """
    从图快照的状态字典构建 HitlPrompt。

    Parameters
    ----------
    state_values:
        ``app.get_state(config).values`` 返回的当前状态字典。
    next_node:
        下一个将要执行的节点名称，"human_confirm" 或 "human_decide"。

    Returns
    -------
    HitlPrompt
    """
    config_draft = state_values.get("config_draft")
    onboarding_result = state_values.get("onboarding_result")
    config_summary = _build_config_draft_summary(config_draft)

    if next_node == "human_confirm":
        # ── confirming 阶段：展示配置草案，让用户确认变量边界 ──────────────────
        questions: list[str] = []
        if onboarding_result is not None:
            questions = list(getattr(onboarding_result, "questions_for_user", []))

        pending_bounds = _build_pending_bounds(config_draft, onboarding_result)

        return HitlPrompt(
            phase="confirming",
            questions=questions,
            config_draft_summary=config_summary,
            pareto_summary=None,
            pending_bounds=pending_bounds,
            options=["confirm"],
            topology=state_values.get("process_topology") or {},
        )

    # ── deciding 阶段：展示 Pareto 分析，让用户决策 ─────────────────────────────
    db_path: str | None = state_values.get("db_path")
    iteration: int = state_values.get("iteration", 0)
    analysis_report: str = state_values.get("analysis_report", "")

    pareto_summary: dict | None = None
    if db_path:
        # 从 SessionEntry 读取实时进度数据（由 optimization_node 的 on_case_complete 写入）
        pareto_points: list[dict] = []
        hv_history: list = []
        try:
            from backend.session_store import get_entry  # noqa: PLC0415
            # session_id 存在于 state_values 中
            _sid = state_values.get("session_id", "")
            if _sid:
                _entry = get_entry(_sid)
                if _entry is not None:
                    with _entry.lock:
                        pareto_points = list(_entry.pareto_points)
                        hv_history = list(_entry.hv_history)
        except Exception:
            pass  # 无法读取进度数据时降级：不携带图表数据，不影响决策流程

        pareto_summary = {
            "db_path": db_path,
            "iteration": iteration,
            "analysis_report": analysis_report,
            "pareto_points": pareto_points,
            "hv_history": hv_history,
        }

    # 从分析报告中提取摘要性问题（取前 5 行非空文本行）
    deciding_questions: list[str] = []
    if analysis_report:
        lines = [ln.strip() for ln in analysis_report.splitlines() if ln.strip()]
        deciding_questions = lines[:5]

    return HitlPrompt(
        phase="deciding",
        questions=deciding_questions,
        config_draft_summary=config_summary,
        pareto_summary=pareto_summary,
        pending_bounds=[],
        options=["continue", "adjust", "done"],
    )


def _state_to_final_result(session_id: str, state_values: dict) -> "FinalResult":
    """将图的最终状态字典转换为 FinalResult。"""
    # 从 onboarding_result.intent 取 OptimizationIntent（用于 summary_report 第 0 章）
    intent = None
    ob_result = state_values.get("onboarding_result")
    if ob_result is not None:
        intent = getattr(ob_result, "intent", None)

    return FinalResult(
        session_id=session_id,
        db_path=state_values.get("db_path"),
        messages=list(state_values.get("messages", [])),
        termination_reason=state_values.get("termination_reason"),
        analysis_report=state_values.get("analysis_report", ""),
        iteration=state_values.get("iteration", 0),
        config_yaml_path=state_values.get("config_yaml_path"),
        intent=intent,
    )


# ---------------------------------------------------------------------------
# H4-0-3  公开 API：start_session
# ---------------------------------------------------------------------------

def start_session(
    aspen_file: str,
    intent_text: str,
    node_db_path: str = "",
    llm_config: Any = None,
    max_iterations: int = 5,
    max_confirm_retries: int = 3,
    session_id: str | None = None,
) -> tuple[str, Union[HitlPrompt, FinalResult]]:
    """
    启动新的 PAO 优化会话。

    流程
    ----
    1. 创建（或接受外部传入的）session_id，向 LangGraph checkpointer 注册。
    2. 以给定参数初始化 PAOGraphState，调用 ``app.invoke()``。
    3. 图执行 onboarding_node，随后在 human_confirm 节点前暂停。
    4. 若图直接完成（onboarding 失败等异常情况），返回 FinalResult。
    5. 否则构建并返回初始 HitlPrompt（phase="confirming"）。

    会话隔离
    --------
    每个 session_id 对应一个独立的 LangGraph thread，持久化于模块级
    ``MemorySaver``。只要进程不重启，会话可跨多次 ``resume_session`` 调用存活。

    Parameters
    ----------
    aspen_file:
        Aspen 仿真文件绝对路径（.bkp / .apw）。
    intent_text:
        用户自然语言优化意图；空字符串时使用默认意图（TAC 最小 + 排放最小）。
    node_db_path:
        节点目录数据库路径（.db）；空字符串时不使用 NodeDB。
    llm_config:
        LLMConfig 实例，None 表示不使用 LLM（降级到关键词匹配意图解析）。
    max_iterations:
        状态机允许的最大优化轮次（传递给 PAOGraphState.max_iterations）。
    max_confirm_retries:
        配置校验最大重试次数（传递给 PAOGraphState.max_confirm_retries）。
    session_id:
        外部预分配的会话标识符（可选）。调用方（如 backend.session_store）传入
        public session_id，确保 LangGraph thread_id 与后端会话 ID 一致，
        resume_session 才能用同一 ID 正确恢复图。
        None 时由本函数自动生成（8 字符 UUID 前缀）。

    Returns
    -------
    tuple[str, HitlPrompt | FinalResult]
        ``(session_id, HitlPrompt)``：图在 human_confirm 暂停，等待用户确认变量边界。
        ``(session_id, FinalResult)``：图意外提前完成（onboarding 失败等），
        FinalResult.termination_reason 说明原因。

    Raises
    ------
    Exception
        graph.py 或底层工具抛出的未捕获异常（日志记录后向上传播）。
    """
    from src.agents.graph import PAOGraphState
    import uuid

    # 使用外部传入的 session_id（如 backend 预分配的 public id），否则自动生成。
    # 用完整 UUID，不截断：thread_id / SimulationDB 工况 session_id / 查询三者须一致。
    if not session_id:
        session_id = str(uuid.uuid4())
    thread_config: dict = {"configurable": {"thread_id": session_id}}

    with _SESSION_LOCK:
        _SESSION_CONFIGS[session_id] = thread_config

    app = _get_app()

    init_state = PAOGraphState(
        session_id=session_id,
        aspen_file=aspen_file,
        intent_text=intent_text,
        node_db_path=node_db_path,
        llm_config=llm_config,
        max_iterations=max_iterations,
        max_confirm_retries=max_confirm_retries,
    )

    _log.info("start_session: session=%s aspen_file=%r", session_id, aspen_file)

    # 执行图直到第一次 HITL 暂停（或图完成）
    app.invoke(init_state, thread_config)

    # 读取当前图状态快照
    snapshot = app.get_state(thread_config)
    state_values: dict = snapshot.values
    next_nodes: tuple = snapshot.next

    _log.info(
        "start_session: session=%s 暂停/完成，next=%s current_phase=%s",
        session_id, next_nodes, state_values.get("current_phase"),
    )

    # 判断是否已完成（next 为空）
    if not next_nodes:
        return session_id, _state_to_final_result(session_id, state_values)

    # 正常情况：暂停在 human_confirm（即 confirming 阶段）
    next_node = next_nodes[0]
    return session_id, _build_hitl_prompt(state_values, next_node)


# ---------------------------------------------------------------------------
# H4-0-3  公开 API：resume_session
# ---------------------------------------------------------------------------

def resume_session(
    session_id: str,
    response: HitlResponse,
) -> Union[HitlPrompt, FinalResult]:
    """
    恢复已暂停的会话，注入用户输入，让图继续执行直至下一次暂停或完成。

    用法示例
    --------
    confirming 阶段（用户确认变量边界）::

        prompt, result = start_session("file.bkp", "最大化产率")
        # ... 前端展示 pending_bounds 给用户 ...
        response = HitlResponse(
            confirmed_bounds={
                r"\\Data\\Blocks\\T0301\\Input\\BASIS_RR": [1.0, 5.0],
            }
        )
        next_prompt = resume_session(session_id, response)

    deciding 阶段（用户决策继续/调整/结束）::

        response = HitlResponse(decision="done")
        final = resume_session(session_id, response)
        # final 是 FinalResult，可读 final.db_path / final.messages

    adjust 阶段（用户修改意图后重新 onboarding）::

        response = HitlResponse(
            decision="adjust",
            edited_intent="改为最小化能耗，约束产品纯度 > 99%",
        )
        new_prompt = resume_session(session_id, response)

    注入映射规则
    -----------
    - ``response.confirmed_bounds`` 非空  →
        ``Command(update={"user_feedback": {"bounds": confirmed_bounds}})``
    - ``response.decision`` 非空           →
        ``Command(update={"user_decision": decision})``
    - ``response.decision == "adjust"`` 且 ``edited_intent`` 非空 →
        同时更新 ``intent_text``

    Parameters
    ----------
    session_id:
        由 ``start_session`` 返回的会话标识符。
    response:
        用户在前端构造的恢复载荷。

    Returns
    -------
    HitlPrompt
        图在下一个 HITL 节点前再次暂停，返回新的待确认载荷。
    FinalResult
        图运行结束（用户选 done / 达到最大轮次 / 异常终止）。

    Raises
    ------
    SessionError
        session_id 未找到（未调用 start_session 或进程已重启）。
    ValueError
        ``response`` 的字段组合不合法（如 confirming 阶段传了 decision）。
    """
    from langgraph.types import Command

    # ── 查找会话配置 ──────────────────────────────────────────────────────────
    with _SESSION_LOCK:
        thread_config = _SESSION_CONFIGS.get(session_id)
    if thread_config is None:
        raise SessionError(
            f"session_id={session_id!r} 不存在。"
            "请先调用 start_session 创建会话，或检查进程是否已重启（MemorySaver 不跨进程持久化）。"
        )

    app = _get_app()

    # ── 确定当前暂停节点 ──────────────────────────────────────────────────────
    snapshot = app.get_state(thread_config)
    next_nodes: tuple = snapshot.next

    if not next_nodes:
        # 图已完成，不应再调用 resume_session
        _log.warning(
            "resume_session: session=%s 图已完成（next 为空），返回 FinalResult", session_id,
        )
        return _state_to_final_result(session_id, snapshot.values)

    current_node = next_nodes[0]

    # ── 构建 LangGraph Command ────────────────────────────────────────────────
    state_update: dict = {}

    if current_node == "human_confirm":
        # confirming 阶段：注入 user_feedback.bounds 和 excluded_paths。
        # human_confirm_node 会从 state.onboarding_result.tunable_report 中取报告，
        # 调用 apply_user_feedback(draft, feedback, tunable_report=...) 支持变量提升，
        # 此处只需传递边界字典和排除路径列表。
        state_update["user_feedback"] = {
            "bounds": dict(response.confirmed_bounds or {}),
            "excluded_paths": list(response.excluded_paths or []),
        }

    elif current_node == "human_decide":
        # deciding 阶段：注入 user_decision（+ 可选 intent_text）
        decision = response.decision or "done"
        if decision not in ("continue", "adjust", "done"):
            raise ValueError(
                f"decision={decision!r} 不合法，必须为 'continue' / 'adjust' / 'done'。"
            )
        state_update["user_decision"] = decision

        # adjust 时同时更新意图文本（若用户提供了修改后的意图）
        if decision == "adjust" and response.edited_intent:
            state_update["intent_text"] = response.edited_intent

    else:
        _log.warning(
            "resume_session: session=%s 暂停在未知节点 %r，按 human_decide 处理",
            session_id, current_node,
        )
        state_update["user_decision"] = response.decision or "done"

    _log.info(
        "resume_session: session=%s current_node=%r state_update_keys=%s",
        session_id, current_node, list(state_update.keys()),
    )

    # ── 恢复图执行 ────────────────────────────────────────────────────────────
    app.invoke(Command(update=state_update), thread_config)

    # ── 读取下一个状态快照 ────────────────────────────────────────────────────
    new_snapshot = app.get_state(thread_config)
    new_state_values: dict = new_snapshot.values
    new_next_nodes: tuple = new_snapshot.next

    _log.info(
        "resume_session: session=%s 执行后 next=%s current_phase=%s",
        session_id, new_next_nodes, new_state_values.get("current_phase"),
    )

    # 图已完成
    if not new_next_nodes:
        return _state_to_final_result(session_id, new_state_values)

    # 图再次暂停
    new_next_node = new_next_nodes[0]
    return _build_hitl_prompt(new_state_values, new_next_node)
