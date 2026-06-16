"""
graph.py — PAO LangGraph 协作状态机（B2 阶段实现）

架构概览：
  PAOGraphState  — 贯穿所有节点的共享状态（纯 Python dataclass，无底层对象）
  onboarding_node    — 扫描 Aspen + 构建配置草案
  human_confirm_node — HITL：等待用户反馈 → 校验草案（最多重试 max_confirm_retries 次）
  optimization_node  — 运行 Pareto 贝叶斯优化
  analysis_node      — 只读 process advisor 分析
  human_decide_node  — HITL：用户决策 continue/adjust/done
  done_node          — 生成最终摘要

图结构：
  START → onboarding → human_confirm → optimization → analysis
        → human_decide → (continue → human_confirm | adjust → onboarding | done → done_node) → END

HITL 模式（interrupt_before）：
  - 图必须以 MemorySaver（或其他 checkpointer）编译后使用；
    无 checkpointer 时 build_graph 会发出 UserWarning。
  - 图在 human_confirm 和 human_decide 节点执行前暂停。
  - 客户端通过 Command(update={...}) 注入输入后再次 invoke 恢复：
      confirm 反馈：app.invoke(Command(update={'user_feedback': {...}}), config)
      decide 决策：app.invoke(Command(update={'user_decision': 'continue'/'adjust'/'done'}), config)

工程约束：
  - graph.py 不导入 src.aspen_driver（optimize_pareto / process_advisor 在节点内懒加载）。
  - 所有 Aspen 仿真由工具层封装，节点只处理状态转换逻辑。
  - 校验超过 max_confirm_retries 次失败时流程终止，不允许以未通过验证的草案进入优化。
"""
from __future__ import annotations

import pathlib
import logging
import tempfile
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import yaml

# LangGraph
from langgraph.graph import END, START, StateGraph

# PAO 内部
from src.agents.onboarding_agent import OnboardingResult, apply_user_feedback, run_onboarding
from src.agents.tools.validate_config import validate_config_tool
from src.models.tunable import ConfigDraft

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B2-1  PAOGraphState
# ---------------------------------------------------------------------------

@dataclass
class PAOGraphState:
    """贯穿 PAO 状态机所有节点的共享状态（仅含基本 Python 类型）。"""
    # 完整 UUID，不截断：须与 SimulationDB 工况 session_id、查询用 ID 一致，
    # 否则 Pareto/报告按 session_id 过滤查回 0 行（“结果全为 0”）。
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aspen_file: str = ""
    node_db_path: str = ""
    intent_text: str = ""
    llm_config: Any = None
    current_phase: str = "onboarding"
    config_draft: ConfigDraft | None = None
    config_yaml_path: str | None = None
    db_path: str | None = None
    onboarding_result: OnboardingResult | None = None
    analysis_report: str = ""
    iteration: int = 0
    max_iterations: int = 5
    confirm_retries: int = 0
    max_confirm_retries: int = 3
    messages: list[str] = field(default_factory=list)
    termination_reason: str | None = None
    # HITL 输入字段（由客户端通过 Command(update=...) 注入）
    user_feedback: dict = field(default_factory=dict)  # confirm 节点的反馈
    user_decision: str = ""  # decide 节点的决策（continue/adjust/done）
    write_feasibility_report: Any = None   # WriteFeasibilityReport | None
    process_topology: dict = field(default_factory=dict)  # {nodes, edges}


# ---------------------------------------------------------------------------
# B2-2  onboarding_node
# ---------------------------------------------------------------------------

def onboarding_node(state: PAOGraphState) -> dict:
    """扫描 Aspen 文件，解析用户意图，生成配置草案，产出待用户确认的问题列表。"""
    result = run_onboarding(
        aspen_file_path=state.aspen_file,
        intent_text=state.intent_text,
        node_db_path=state.node_db_path,
        llm_config=state.llm_config,
    )

    new_msgs = list(state.messages)
    new_msgs.append(f"【接入向导】已完成配置草案生成（草案 ID：{result.config_draft.draft_id}）")
    for i, q in enumerate(result.questions_for_user, 1):
        new_msgs.append(f"  问题 {i}：{q}")
    if result.warnings:
        new_msgs.append(f"  ⚠ 警告 {len(result.warnings)} 条，详见 config_draft.warnings")

    return {
        "onboarding_result": result,
        "config_draft":      result.config_draft,
        "current_phase":     "confirming",
        "confirm_retries":   0,
        "messages":          new_msgs,
        "process_topology":  getattr(
            getattr(result, "tunable_report", None), "topology", {}
        ) or {},
    }


# ---------------------------------------------------------------------------
# B2-2b  write_feasibility_node
# ---------------------------------------------------------------------------

def write_feasibility_node(state: PAOGraphState) -> dict:
    """
    对 ConfigDraft 的每个设计变量执行 COM 层写入测试（不运行仿真）。

    在 onboarding_node 完成后、human_confirm_node 之前自动运行，
    让用户第一次看到确认页时草案已是"可写变量"的干净集合。

    - 全部通过：只追加一条成功日志，不修改草案。
    - 有不可写变量：自动移除，追加移除说明到 messages。
    - 剩余变量数为 0：终止流程（termination_reason=no_writable_variables）。
    - Aspen 文件打开失败：跳过验证，草案原样透传（不阻断流程）。
    """
    import dataclasses as _dc
    from src.agents.tools.verify_write_feasibility import verify_write_feasibility_impl

    draft = state.config_draft
    if draft is None or not draft.design_variables:
        return {}   # 无草案 / 无变量，透传不做任何修改

    new_msgs = list(state.messages)

    report = verify_write_feasibility_impl(state.aspen_file, draft.design_variables)

    # Aspen 打开失败时，report.warnings 里有说明，所有变量 writable=False
    # 若全部失败且是因为文件打开失败，给用户友好提示并透传草案（不移除变量）
    all_failed_due_to_open = (
        report.warnings
        and all(not r.writable for r in report.results)
        and any("打开 Aspen 文件失败" in w for w in report.warnings)
    )
    if all_failed_due_to_open:
        for w in report.warnings:
            new_msgs.append(f"【写入验证】⚠ {w}，已跳过验证，变量列表保持原样。")
        return {"messages": new_msgs, "write_feasibility_report": report}

    if not report.unwritable_paths:
        n = len(report.results)
        new_msgs.append(f"【写入验证】全部 {n} 个设计变量写入测试通过。")
        return {"messages": new_msgs, "write_feasibility_report": report}

    # 移除不可写变量
    bad = set(report.unwritable_paths)
    new_dvs = [dv for dv in draft.design_variables if dv["aspen_path"] not in bad]
    extra_warnings = [
        f"变量 {p} COM 写入测试失败，已从优化草案中移除"
        for p in report.unwritable_paths
    ]
    new_draft = _dc.replace(
        draft,
        design_variables=new_dvs,
        warnings=list(draft.warnings) + extra_warnings,
    )

    new_msgs.append(
        f"【写入验证】发现 {len(bad)} 个变量无法写入 Aspen，已自动移除："
    )
    for r in report.results:
        if not r.writable:
            new_msgs.append(f"  ✗ {r.aspen_path}：{r.error[:160]}")

    if not new_dvs:
        new_msgs.append(
            "  移除后草案中无可用设计变量，无法继续优化，流程终止。"
            "请检查 Aspen 仿真文件中各塔/设备的规定设置（COL-SPECS）后重新开始。"
        )
        return {
            "config_draft": new_draft,
            "write_feasibility_report": report,
            "messages": new_msgs,
            "current_phase": "done",
            "termination_reason": "no_writable_variables",
        }

    new_msgs.append(
        f"  剩余 {len(new_dvs)} 个可写变量，请在确认页核查后启动优化。"
    )
    return {
        "config_draft": new_draft,
        "write_feasibility_report": report,
        "messages": new_msgs,
    }


# ---------------------------------------------------------------------------
# B2-3  human_confirm_node
# ---------------------------------------------------------------------------

def human_confirm_node(state: PAOGraphState) -> dict:
    """HITL 节点：应用用户反馈，校验草案，写 YAML。

    使用 interrupt_before 模式（不调用 interrupt()）。
    客户端通过 Command(update={'user_feedback': {...}}) 传入反馈，
    然后图会执行此节点。

    retry 逻辑：
      - 校验失败时，更新 confirm_retries，并在 messages 中追加失败信息。
      - 客户端检查 messages 后再次调用 Command(update={'user_feedback': {...}}) 重试。
      - 超过 max_confirm_retries 次校验失败时：终止流程（current_phase="done"，
        termination_reason="confirm_validation_failed"），绝不进入 optimizing。
        这是硬约束：未通过 validate_config_tool 的草案不允许触发任何 Aspen 仿真。
    """
    draft = state.config_draft
    if draft is None:
        return {"current_phase": "done", "termination_reason": "no_config_draft"}

    new_msgs = list(state.messages)
    feedback = state.user_feedback or {}

    # 取 tunable_report（若有 onboarding_result）供 apply_user_feedback 提升变量
    tunable_report = None
    if state.onboarding_result is not None:
        tunable_report = state.onboarding_result.tunable_report

    # 应用用户反馈（支持将 in_draft=False 变量提升进 design_variables）
    draft = apply_user_feedback(draft, feedback, tunable_report=tunable_report)
    yaml_path = _write_draft_yaml(draft, state.session_id)

    # 校验
    validate_result = validate_config_tool.invoke({"config_path": yaml_path})
    fatal = "[失败]" in validate_result or validate_result.startswith("错误：")

    retries = state.confirm_retries
    if fatal:
        retries += 1
        new_msgs.append(
            f"【配置校验】第 {retries} 次失败（草案 {draft.draft_id}）：{validate_result[:200]}"
        )
        if retries >= state.max_confirm_retries:
            # 硬约束：校验未通过，不允许进入优化，直接终止
            new_msgs.append(
                f"  已达最大校验重试次数（{state.max_confirm_retries}），"
                "配置草案始终无法通过验证，流程终止。请修正配置后重新开始。"
            )
            return {
                "config_draft": draft, "config_yaml_path": yaml_path,
                "current_phase": "done",
                "termination_reason": "confirm_validation_failed",
                "confirm_retries": retries,
                "messages": new_msgs, "user_feedback": {},
            }
        # 未超限：回到 confirming，等待用户再次反馈（interrupt_before 会再次暂停）
        return {
            "config_draft": draft, "config_yaml_path": yaml_path,
            "current_phase": "confirming", "confirm_retries": retries,
            "messages": new_msgs, "user_feedback": {},
        }

    new_msgs.append(f"【配置校验】通过（草案 {draft.draft_id}）。")
    return {
        "config_draft": draft, "config_yaml_path": yaml_path,
        "current_phase": "optimizing", "confirm_retries": 0,
        "messages": new_msgs, "user_feedback": {},
    }


def _write_draft_yaml(draft: ConfigDraft, session_id: str) -> str:
    """将 ConfigDraft 序列化为临时 YAML 文件，返回路径。"""
    tmp_dir = pathlib.Path(tempfile.gettempdir()) / "pao_sessions" / session_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = tmp_dir / f"draft_{draft.draft_id}.yaml"
    yaml_dict = draft.to_yaml_dict()
    # 将 API session_id 注入 optimizer 节，确保 SimulationDB 记录的 session_id
    # 与 backend 会话 ID 一致，使 /report 端点能正确过滤查询数据
    if isinstance(yaml_dict.get("optimizer"), dict):
        yaml_dict["optimizer"]["session_id"] = session_id
    else:
        yaml_dict["session_id"] = session_id
    yaml_path.write_text(yaml.dump(yaml_dict, allow_unicode=True), encoding="utf-8")
    return str(yaml_path)


# ---------------------------------------------------------------------------
# B2-4  optimization_node
# ---------------------------------------------------------------------------

def optimization_node(state: PAOGraphState) -> dict:
    """调用 optimize_pareto_tool 运行多目标贝叶斯优化。

    直接调用底层 optimize_pareto_case 并注入 on_case_complete 回调，
    每完成一个仿真样本就将当前 Pareto 前沿和超体积历史写入 SessionEntry，
    供 SSE 流实时推送 progress 事件给前端。
    """
    from src.agents.tools._common import (  # noqa: PLC0415
        _import_pareto_deps, _load_optimize_config, _AspenDriver, _optimize_pareto_fn,
    )
    import src.agents.tools._common as _common  # noqa: PLC0415
    from src.agents.tools.optimize_pareto import _fmt_pareto_result_summary  # noqa: PLC0415

    config_path = state.config_yaml_path
    if not config_path:
        return {"current_phase": "done", "termination_reason": "no_config_yaml_path",
                "messages": state.messages + ["【优化】错误：config_yaml_path 为空。"]}

    new_msgs = list(state.messages)
    new_msgs.append(f"【优化】开始第 {state.iteration + 1} 轮（配置：{config_path}）")

    # 尝试导入底层依赖
    err = _import_pareto_deps()
    if err:
        # 依赖不可用（非 Windows 环境），回退到 tool.invoke 方式（无实时进度）
        from src.agents.tools.optimize_pareto import optimize_pareto_tool  # noqa: PLC0415
        result_text = optimize_pareto_tool.invoke({"config_path": config_path, "db_path": ""})
        failed = result_text.startswith("错误：") or "[失败]" in result_text
        db_path = state.db_path or str(pathlib.Path(config_path).parent / "output" / "simulation.db")
        if failed:
            new_msgs.append(f"【优化】失败：{result_text[:200]}")
            return {"current_phase": "done", "termination_reason": "optimization_failed",
                    "db_path": db_path, "messages": new_msgs}
        new_msgs.append(f"【优化】第 {state.iteration + 1} 轮完成，DB：{db_path}")
        new_msgs.append(result_text[:500])
        return {"current_phase": "analyzing", "db_path": db_path,
                "iteration": state.iteration + 1, "messages": new_msgs}

    # 解析配置路径
    from src.agents.tools._common import _resolve_config_path  # noqa: PLC0415
    try:
        yaml_path = _resolve_config_path(config_path)
    except FileNotFoundError as exc:
        new_msgs.append(f"【优化】失败：{exc}")
        return {"current_phase": "done", "termination_reason": "optimization_failed",
                "messages": new_msgs}

    # 加载配置
    try:
        opt_cfg, sim_filepath, driver_kwargs = _common._load_optimize_config(yaml_path)
    except Exception as exc:
        new_msgs.append(f"【优化】失败（配置加载）：{exc}")
        return {"current_phase": "done", "termination_reason": "optimization_failed",
                "messages": new_msgs}

    db_path = state.db_path or str(pathlib.Path(config_path).parent / "output" / "simulation.db")

    # 确定数据库路径
    if not getattr(opt_cfg, "db_path", None):
        opt_cfg.db_path = yaml_path.parent / "output" / "simulation.db"
    else:
        db_path = str(opt_cfg.db_path)

    # 将 opt_cfg.session_id 对齐到 LangGraph 会话 ID，
    # 确保工况写入数据库时用的 session_id 与后续查询时一致
    session_id = state.session_id
    opt_cfg.session_id = session_id

    # 构建实时进度回调：每个仿真样本完成后更新 SessionEntry
    # 用可变容器在闭包中积累 HV 历史（每次 on_case_complete 追加一个值）
    _hv_accumulator: list[float] = []

    def _on_case_complete(case, sample_idx: int, total: int) -> None:  # noqa: ANN001
        """将当前 Pareto 前沿和 HV 历史写入 SessionEntry，触发 SSE progress 推送。"""
        try:
            from backend.session_store import get_entry  # noqa: PLC0415
            from src.optimization.pareto import compute_pareto  # noqa: PLC0415
            from src.database.simulation_db import SimulationDB  # noqa: PLC0415

            entry = get_entry(session_id)
            if entry is None:
                return

            # 从数据库读取当前 session 的所有工况计算 Pareto 前沿
            db_file = opt_cfg.db_path
            if db_file is None:
                return

            db = SimulationDB(db_file)
            try:
                rows = db.query_cases(session_id=session_id, limit=2000, offset=0)
            finally:
                db.close()

            if not rows:
                return

            # 目标函数名必须以优化配置为准，避免数据库首个失败/部分目标工况误导图表字段。
            obj_names: list[str] = list(getattr(opt_cfg, "objective_names", []) or [])
            if not obj_names:
                return

            # 重建 ProcessCase，同时计算 Pareto 前沿和超体积
            from src.agents.tools.summarize_pareto import _dict_to_process_case  # noqa: PLC0415
            cases_list = [_dict_to_process_case(r) for r in rows]
            pareto_result = compute_pareto(
                cases=cases_list,
                objective_names=obj_names,
                compute_hv=True,    # 同时计算 HV，供 HvChart 使用
            )

            # 将第一前沿序列化为前端期望的格式 [{obj1: v1, obj2: v2, ...}]
            points: list[dict] = []
            if pareto_result.first_front and pareto_result.first_front.cases:
                for pc, vec in zip(
                    pareto_result.first_front.cases,
                    pareto_result.first_front.objective_vectors,
                ):
                    pt: dict = {}
                    for name, val in zip(obj_names, vec):
                        pt[str(name).lower()] = val
                    points.append(pt)

            # 将当前 HV 值追加到累积列表（None 表示样本不足，不追加）
            hv_val = pareto_result.hypervolume
            if hv_val is not None:
                _hv_accumulator.append(hv_val)

            with entry.lock:
                entry.pareto_points = points
                entry.hv_history = list(_hv_accumulator)
                entry.progress_version += 1

        except Exception as exc:
            # 提升到 warning：此回调静默失败会表现为"图不实时更新"（数据只在优化结束后才出现），
            # 难以排查。保留异常隔离（不影响优化主流程），但让失败在日志中可见。
            _log.warning("_on_case_complete 实时进度回调异常（已隔离，不影响优化）：%s", exc, exc_info=True)

    opt_cfg.on_case_complete = _on_case_complete

    # 运行优化
    try:
        with _common._AspenDriver(**driver_kwargs) as driver:
            driver.open(sim_filepath)
            result = _common._optimize_pareto_fn(driver=driver, config=opt_cfg)
    except FileNotFoundError as exc:
        new_msgs.append(f"【优化】失败：Aspen 仿真文件不存在 — {exc}")
        return {"current_phase": "done", "termination_reason": "optimization_failed",
                "db_path": db_path, "messages": new_msgs}
    except Exception as exc:
        new_msgs.append(f"【优化】失败：{exc}")
        return {"current_phase": "done", "termination_reason": "optimization_failed",
                "db_path": db_path, "messages": new_msgs}

    db_path = str(opt_cfg.db_path) if opt_cfg.db_path else db_path

    # 优化完成后将最终 Pareto 点和完整 HV 历史写入 SessionEntry
    try:
        from backend.session_store import get_entry  # noqa: PLC0415
        entry = get_entry(session_id)
        if entry is not None:
            hv_clean = [v for v in result.hv_history if v is not None]
            points: list[dict] = []
            first_front = getattr(result, "first_front", None)
            obj_names = list(getattr(result, "objective_names", []) or [])
            if first_front is not None and obj_names:
                for vec in getattr(first_front, "objective_vectors", []) or []:
                    pt = {
                        str(name).lower(): val
                        for name, val in zip(obj_names, vec)
                    }
                    points.append(pt)
            with entry.lock:
                entry.pareto_points = points
                entry.hv_history = hv_clean
                entry.progress_version += 1
    except Exception as exc:
        _log.debug("优化完成后写入 hv_history 失败（已忽略）：%s", exc)

    # 格式化报告文本
    try:
        result_text = _fmt_pareto_result_summary(result, config_path)
    except Exception as exc:
        result_text = f"（报告格式化失败：{exc}）"

    failed = "[失败]" in result_text
    if failed:
        new_msgs.append(f"【优化】失败：{result_text[:200]}")
        return {"current_phase": "done", "termination_reason": "optimization_failed",
                "db_path": db_path, "messages": new_msgs}

    new_msgs.append(f"【优化】第 {state.iteration + 1} 轮完成，DB：{db_path}")
    new_msgs.append(result_text[:500])
    return {"current_phase": "analyzing", "db_path": db_path,
            "iteration": state.iteration + 1, "messages": new_msgs}


# ---------------------------------------------------------------------------
# B2-5  analysis_node
# ---------------------------------------------------------------------------

def analysis_node(state: PAOGraphState) -> dict:
    """调用 process_advisor_agent 对优化结果做只读分析。"""
    from src.agents.process_advisor import run_process_advisor_agent  # noqa: PLC0415

    new_msgs = list(state.messages)
    new_msgs.append(f"【分析】开始第 {state.iteration} 轮分析…")

    try:
        report = run_process_advisor_agent(
            case_config_path=state.config_yaml_path or "",
            db_path=state.db_path,
            session_id=state.session_id,
            node_db_path=state.node_db_path or None,
            mode="db" if state.db_path else "config",
            llm_config=state.llm_config,
        )
    except Exception as exc:
        report = f"分析出现异常：{exc}"

    new_msgs.append(f"【分析摘要】\n{report[:300]}{'…' if len(report) > 300 else ''}")
    return {"analysis_report": report, "current_phase": "deciding", "messages": new_msgs}


# ---------------------------------------------------------------------------
# B2-6  human_decide_node
# ---------------------------------------------------------------------------

def human_decide_node(state: PAOGraphState) -> dict:
    """HITL 节点：根据 state.user_decision 路由到下一步。

    使用 interrupt_before 模式（不调用 interrupt()）。
    客户端通过 Command(update={'user_decision': 'continue'/'adjust'/'done'}) 传入决策。
    """
    decision: str = state.user_decision or "done"

    new_msgs = list(state.messages)
    new_msgs.append(f"【用户决策】第 {state.iteration} 轮：{decision!r}")

    if decision == "continue" and state.iteration < state.max_iterations:
        return {"current_phase": "confirming", "confirm_retries": 0,
                "messages": new_msgs, "user_decision": ""}
    if decision == "adjust":
        return {"current_phase": "onboarding", "messages": new_msgs, "user_decision": ""}
    if state.iteration >= state.max_iterations:
        new_msgs.append(f"  已达最大迭代轮次 {state.max_iterations}，自动终止。")
    return {"current_phase": "done", "messages": new_msgs, "user_decision": ""}


# ---------------------------------------------------------------------------
# B2-7  done_node
# ---------------------------------------------------------------------------

def done_node(state: PAOGraphState) -> dict:
    """终止节点：生成最终摘要。"""
    new_msgs = list(state.messages) + [
        "【优化结束】",
        f"  总轮次：{state.iteration}",
        f"  终止原因：{state.termination_reason or '用户主动终止'}",
        f"  结果数据库：{state.db_path or '未生成'}",
        f"  分析报告：{'已生成' if state.analysis_report else '无'}",
    ]
    return {"current_phase": "done", "messages": new_msgs}


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------

def _route_after_confirm(state: PAOGraphState) -> str:
    """human_confirm_node 之后的路由：

    - current_phase == "confirming"：校验失败未超限，self-loop 回到 human_confirm
      （interrupt_before 会再次暂停等待新的 user_feedback）
    - current_phase == "done"：校验超限终止，转到 done_node
    - 其他（"optimizing"）：校验通过，转到 optimization
    """
    if state.current_phase == "confirming":
        return "human_confirm"
    if state.current_phase == "done":
        return "done"
    return "optimization"


def _route_after_optimization(state: PAOGraphState) -> str:
    """optimization_node 之后的路由：失败时 done，成功时 analysis。"""
    return "done" if state.termination_reason == "optimization_failed" else "analysis"


def _route_after_decide(state: PAOGraphState) -> str:
    """human_decide_node 之后的路由：根据 current_phase 分发。"""
    phase = state.current_phase
    if phase == "confirming":
        return "human_confirm"
    if phase == "onboarding":
        return "onboarding"
    return "done"


# ---------------------------------------------------------------------------
# B2-8  build_graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    构建并编译 PAO 协作状态机。

    HITL 模式（interrupt_before）：
      图在 human_confirm 和 human_decide 节点前暂停，等待客户端注入输入：
        - confirm 反馈：Command(update={'user_feedback': {...}})
        - decide 决策：Command(update={'user_decision': 'continue'/'adjust'/'done'})
      然后图继续执行相应节点。

    Parameters
    ----------
    checkpointer:
        LangGraph checkpointer（如 MemorySaver）；HITL 功能必须提供。

    Returns
    -------
    CompiledGraph
    """
    if checkpointer is None:
        warnings.warn(
            "build_graph called without a checkpointer. "
            "HITL功能（interrupt_before暂停/恢复）需要持久化 checkpointer（如 MemorySaver）。"
            "无 checkpointer 时图可用于单步测试，但无法跨调用保留状态，"
            "不适合生产环境的 human_confirm / human_decide 交互流程。",
            UserWarning,
            stacklevel=2,
        )
    g = StateGraph(PAOGraphState)

    g.add_node("onboarding",         onboarding_node)
    g.add_node("write_feasibility",  write_feasibility_node)
    g.add_node("human_confirm",      human_confirm_node)
    g.add_node("optimization",       optimization_node)
    g.add_node("analysis",           analysis_node)
    g.add_node("human_decide",       human_decide_node)
    g.add_node("done",               done_node)

    # 固定边
    g.add_edge(START,               "onboarding")
    g.add_edge("onboarding",        "write_feasibility")   # 试写验证在确认前自动执行
    g.add_edge("analysis",          "human_decide")
    g.add_edge("done",              END)

    # write_feasibility → human_confirm（正常）或 done（无可写变量）
    g.add_conditional_edges(
        "write_feasibility",
        lambda s: "done" if s.current_phase == "done" else "human_confirm",
        {"human_confirm": "human_confirm", "done": "done"},
    )

    # optimization → analysis（成功）或 done（失败）
    g.add_conditional_edges(
        "optimization",
        _route_after_optimization,
        {"analysis": "analysis", "done": "done"},
    )

    # human_confirm → human_confirm（校验失败重试）/ optimization（校验通过）/ done（超限终止）
    # 与 interrupt_before 结合：self-loop 在 HITL 暂停模式下是安全的
    g.add_conditional_edges(
        "human_confirm",
        _route_after_confirm,
        {"human_confirm": "human_confirm", "optimization": "optimization", "done": "done"},
    )

    # human_decide → human_confirm / onboarding / done
    g.add_conditional_edges(
        "human_decide",
        _route_after_decide,
        {"human_confirm": "human_confirm", "onboarding": "onboarding", "done": "done"},
    )

    # interrupt_before：在 HITL 节点执行前暂停，等待客户端注入 user_feedback / user_decision
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_confirm", "human_decide"],
    )
