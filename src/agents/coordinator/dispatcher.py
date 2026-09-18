"""
dispatcher.py — Coordinator agent 的执行分发层。

职责：按 Plan 顺序执行每个 TaskSpec，把结果记录成 ExecutionState
（沿用 src.agents.demo_workflow.state 的 WorkflowStep 范式：
name/status/report/skipped_reason，逻辑一致但独立定义，见
src.models.coordinator.CoordinatorStep 的 docstring 说明）。

设计原则
--------
  - ready=False 的任务直接标记为 "skipped"，不会尝试用不完整参数执行——
    除非它是 gap-fill 可修复的缺口（见下）。
  - 单个任务执行失败（目标函数抛异常）只影响该任务的记录（status="error"），
    不阻断后续任务——与 demo_workflow 的隔离原则一致。
  - v1 范围限定：不做任务间自动数据传递（数值结果层面）。每个任务独立执行、
    独立产出报告，如果用户想要"用第一步的拟合结果去做第二步的推荐"，
    需要自己看完第一步的结果后决定第二步的参数（这是骨架阶段的已知取舍，
    写在 coordinator/agent.py 的模块 docstring 里）。
    但文件路径层面的传递是支持的：每个 _exec_* 返回 (report, produced_paths)，
    produced_paths 供连续对话模式（run_coordinator_turn）记忆并在后续轮次
    自动填空，用户不用在每一轮重复给同一个路径；同一次 dispatch_plan 调用
    内部也会把前面任务产出的路径合并进 env，供后面被阻塞的任务重建时使用
    （见下方 gap-fill 机制）。
  - 各路由目标的实际调用被封装成独立的 _exec_* 函数，异常在这一层被
    统一捕获转成 status="error"，调用目标函数本身不做 try/except
    （沿用"目标函数只管做事，异常处理是编排层职责"的分层原则）。

gap-fill 链式推理
-----------------
某个任务 ready=False 且 TaskSpec.waiting_on_gapfill=True 时（由 planner.py
判定：缺的是配置文件，但 aspen_file 已知，可以自动接入生成），dispatch_plan
会：
  1. 先尝试用当前 env 重建该任务（如果同一个 Plan 里更靠前的任务已经产出了
     所需路径——比如用户自己在同一句话里说了"先接入再优化"，接入步骤已跑完
     ——直接复用，不再重复接入）。
  2. 仍然缺失时，自动插入并执行一个 ONBOARD_NEW_CASE gap-fill 任务，把产出
     的 case_config_path 合并进 env，重建被阻塞的任务。
  3. 一次 dispatch_plan 调用最多插入一次 gap-fill 接入（多个任务共享同一次
     接入结果，不会重复触发多次 Aspen 扫描）。

安全刹车：RUN_OPTIMIZATION 永不自动使用本轮刚生成（无论是通过 gap-fill
自动插入，还是用户自己在同一句话里显式请求的接入步骤）的配置直接开始仿真
——只要配置是"本轮新产出"的，就必须停在确认点，把决定权交还给用户；
只有配置是更早轮次就已存在（用户已经有机会看过）时才会正常执行。这是
唯一的安全边界：其余任务（推荐边界、诊断）在配置刚生成时可以直接跑，
因为它们本身就是只读分析/推荐性质，不会像启动仿真那样有实际副作用。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.agents.coordinator.planner import build_gapfill_onboard_task, rebuild_task
from src.models.coordinator import ExecutionState, IntentType, Plan, TaskSpec

_log = logging.getLogger(__name__)

# _exec_* 函数的返回类型：(报告文本, 本次新产出的路径字典)。
# 键名与 ChatSessionState 的记忆字段对齐："case_config_path" / "db_path"。
# 绝大多数任务不产出路径，返回空字典即可。
_ExecResult = tuple[str, dict[str, str]]


# ---------------------------------------------------------------------------
# 各路由目标的执行封装
# ---------------------------------------------------------------------------
# 每个 _exec_* 函数签名一致：(kwargs: dict, llm_config) -> _ExecResult
# 异常直接向上抛出，由 dispatch_plan 统一捕获记录为 status="error"。

def _write_draft_yaml(draft, aspen_file_path: str) -> str:
    """把 ConfigDraft 落盘为 YAML，返回路径。

    落盘位置：{Aspen 文件所在目录}/output/coordinator_draft_{draft_id}.yaml。
    不涉及 graph.py._write_draft_yaml 里的 session_id 注入逻辑
    （那是给 backend SSE 会话过滤用的，Coordinator CLI 场景不需要）。
    """
    import yaml as _yaml

    out_dir = Path(aspen_file_path).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"coordinator_draft_{draft.draft_id}.yaml"
    yaml_path.write_text(_yaml.dump(draft.to_yaml_dict(), allow_unicode=True), encoding="utf-8")
    return str(yaml_path)


def _exec_onboard(kwargs: dict, llm_config) -> _ExecResult:
    from src.agents.onboarding_agent import run_onboarding

    result = run_onboarding(
        aspen_file_path=kwargs["aspen_file_path"],
        intent_text=kwargs.get("intent_text", ""),
        node_db_path=kwargs["node_db_path"],
        llm_config=llm_config,
    )
    draft_path = _write_draft_yaml(result.config_draft, kwargs["aspen_file_path"])

    lines = [f"配置草案已生成（draft_id={result.config_draft.draft_id}）并写入：{draft_path}"]
    for i, q in enumerate(result.questions_for_user, 1):
        lines.append(f"  问题 {i}：{q}")
    if result.warnings:
        lines.append(f"  警告 {len(result.warnings)} 条：{result.warnings}")
    return "\n".join(lines), {"case_config_path": draft_path}


def _exec_recommend_bounds(kwargs: dict, llm_config) -> _ExecResult:
    import yaml as _yaml

    from src.agents.boundary_advisor import VarMeta, format_boundary_report, recommend_boundaries_agent

    config_path = kwargs["config_path"]
    with open(config_path, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f) or {}

    variables: list[VarMeta] = []
    for dv in cfg.get("design_variables", []):
        name = dv.get("name") or dv.get("aspen_path")
        if not name:
            continue
        try:
            iv = float(dv["initial_value"]) if dv.get("initial_value") is not None else None
        except (TypeError, ValueError):
            iv = None
        variables.append(VarMeta(
            name=str(name), initial_value=iv, unit=str(dv.get("unit", "") or ""),
            var_type=dv.get("type", "continuous"),
            lower_global=dv.get("lower_bound"), upper_global=dv.get("upper_bound"),
        ))
    if not variables:
        raise ValueError(f"配置 {config_path} 中未读到任何 design_variables")

    report = recommend_boundaries_agent(variables, context=kwargs.get("context", ""), llm_config=llm_config)
    return format_boundary_report(report), {}


def _exec_fit_kinetics(kwargs: dict, llm_config) -> _ExecResult:
    from src.agents.kinetics_advisor import KineticsDataPoint, fit_kinetics_params
    from src.agents.kinetics_advisor.tools import format_fit_result

    points: list[KineticsDataPoint] = []
    with open(kwargs["data_csv"], encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(KineticsDataPoint(
                temperature=float(row["temperature"]),
                rate_constant=float(row["rate_constant"]),
            ))
    if not points:
        raise ValueError(f"CSV {kwargs['data_csv']} 中未读到任何数据点")

    outcome = fit_kinetics_params(
        points, var_name=kwargs.get("var_name", "kinetics_param"),
        context=kwargs.get("context", ""), llm_config=llm_config,
    )
    if not outcome.plan.can_fit:
        return f"拒绝拟合：{outcome.plan.rejection_reason}", {}
    lines = [format_fit_result(outcome.fit_result)]
    if outcome.llm_review:
        lines.append("")
        lines.append(outcome.llm_review)
    return "\n".join(lines), {}


def _exec_recommend_kinetics_bounds(kwargs: dict, llm_config) -> _ExecResult:
    import yaml as _yaml

    from src.agents.kinetics_advisor import KineticsVarMeta, recommend_kinetics_bounds_agent
    from src.agents.kinetics_advisor.tools import format_recommendation

    with open(kwargs["variables_yaml"], encoding="utf-8") as f:
        cfg = _yaml.safe_load(f) or {}

    variables: list[KineticsVarMeta] = []
    for item in cfg.get("variables", []):
        name = item.get("name")
        param_type = item.get("param_type")
        if not name or param_type not in ("activation_energy", "pre_exponential_factor"):
            continue
        t_range = item.get("temperature_range")
        variables.append(KineticsVarMeta(
            name=str(name), param_type=param_type,
            unit=str(item.get("unit", "") or ""),
            current_value=(float(item["current_value"]) if item.get("current_value") is not None else None),
            reaction_order=(float(item["reaction_order"]) if item.get("reaction_order") is not None else None),
            temperature_range=(tuple(t_range) if t_range else None),
        ))
    if not variables:
        raise ValueError(f"变量清单 {kwargs['variables_yaml']} 中未读到任何有效变量")

    context = kwargs.get("context") or str(cfg.get("context", "") or "")
    report = recommend_kinetics_bounds_agent(variables, context=context, llm_config=llm_config)
    return "\n\n".join(format_recommendation(r) for r in report.recommendations), {}


def _exec_diagnose(kwargs: dict, llm_config) -> _ExecResult:
    from src.agents.process_advisor import run_process_advisor_agent

    report = run_process_advisor_agent(
        case_config_path=kwargs["case_config_path"],
        db_path=kwargs.get("db_path"),
        mode=kwargs.get("mode", "db"),
        llm_config=llm_config,
    )
    return report, {}


def _exec_run_optimization(kwargs: dict, llm_config) -> _ExecResult:
    from src.agents.tools.optimize_pareto import optimize_pareto_tool

    db_path = kwargs.get("db_path", "") or ""
    report = optimize_pareto_tool.invoke({
        "config_path": kwargs["config_path"],
        "db_path": db_path,
    })
    # db_path 若由 Planner 推断得出（非空），记为本次产出，供后续轮次
    # （如 diagnose_results）自动复用，不用用户重复指定。
    produced = {"db_path": db_path} if db_path else {}
    return report, produced


def _skip_run_optimization_confirm(kwargs: dict) -> str:
    """RUN_OPTIMIZATION 安全刹车：配置是本轮新产出的，停在确认点不执行。"""
    return (
        f"配置刚生成（{kwargs.get('config_path', '未知路径')}），出于安全考虑不会自动开始仿真。"
        "请先查看草案里的变量边界，确认无误后重新描述需求（如“开始优化”）以启动这一轮已确认的优化。"
    )


def _exec_general_chat(kwargs: dict, llm_config) -> _ExecResult:
    """纯知识问答/闲聊：直接调用 LLM 回答，不驱动任何工具、不产出路径。

    与其他 _exec_* 不同，这里的 LLM 调用不是"意图分类"（那是 intent.py
    的职责），而是"真正回答用户的问题"——两次调用完全独立，用不同的
    prompt（见 prompts.GENERAL_CHAT_SYSTEM_PROMPT）。

    无 LLM key 时不能装作能回答，必须如实告知——回答问题本身就是这个
    任务的唯一价值，没有规则兜底可言（不像边界推荐/拟合还有确定性算法
    或规则可以退路）。
    """
    from src.agents.coordinator.prompts import GENERAL_CHAT_SYSTEM_PROMPT, GENERAL_CHAT_USER_TEMPLATE
    from src.agents.llm_client import chat, is_configured, load_llm_config

    user_text = kwargs.get("user_text", "")
    if not user_text.strip():
        return "（未收到问题内容，无法回答）", {}

    cfg = llm_config if llm_config is not None else load_llm_config()
    if not is_configured(cfg):
        return (
            f"未配置大模型 API key（请设置 {cfg.api_key_env}），暂时无法回答知识性问题。"
            "可以先配置好 .env 后再试。"
        ), {}

    user = GENERAL_CHAT_USER_TEMPLATE.format(user_text=user_text)
    context = kwargs.get("context", "")
    if context:
        user = f"（工艺背景：{context}）\n\n{user}"

    answer = chat(cfg, system=GENERAL_CHAT_SYSTEM_PROMPT, user=user)
    return answer.strip(), {}


_EXECUTORS = {
    IntentType.ONBOARD_NEW_CASE: _exec_onboard,
    IntentType.RECOMMEND_VAR_BOUNDS: _exec_recommend_bounds,
    IntentType.FIT_KINETICS_PARAMS: _exec_fit_kinetics,
    IntentType.RECOMMEND_KINETICS_BOUNDS: _exec_recommend_kinetics_bounds,
    IntentType.DIAGNOSE_RESULTS: _exec_diagnose,
    IntentType.RUN_OPTIMIZATION: _exec_run_optimization,
    IntentType.GENERAL_CHAT: _exec_general_chat,
}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _run_executor(task: TaskSpec, llm_config) -> tuple[str, dict[str, str], bool]:
    """执行单个任务的目标函数，返回 (report, produced_paths, ok)。

    ok=False 表示执行抛了异常，report 此时是错误信息文本，供调用方
    统一记为 status="error"。抽出来是因为 gap-fill 插入的 onboard 任务
    和普通任务共用同一套执行+异常捕获逻辑。
    """
    executor = _EXECUTORS.get(task.intent_type)
    if executor is None:
        return f"内部错误：未找到 {task.intent_type.value} 对应的执行器", {}, False
    try:
        report, produced_paths = executor(task.kwargs, llm_config)
        return report, produced_paths, True
    except Exception as exc:  # noqa: BLE001 — 单任务失败不得阻断后续任务
        _log.warning("coordinator：任务 %s 执行失败：%s", task.task_id, exc, exc_info=True)
        return str(exc), {}, False


def dispatch_plan(plan: Plan, llm_config=None, env: dict | None = None) -> ExecutionState:
    """
    按 Plan 顺序执行每个 TaskSpec，产出 ExecutionState。

    Parameters
    ----------
    plan:
        planner.build_plan 产出的执行计划。
    llm_config:
        统一注入给所有需要 LLM 的子 agent（boundary_advisor/kinetics_advisor/
        process_advisor/onboarding_agent），None 时各子 agent 自行从环境变量
        加载。传入同一个 LLMConfig 避免每个任务重复解析 .env。
    env:
        构造 Plan 时使用的原始 env 字典（见 planner.make_env），gap-fill
        需要它来判断/补全 aspen_file、并在自动接入成功后重建被阻塞的任务。
        None 时等价于空字典——传 None 意味着放弃 gap-fill 能力（无法确定
        aspen_file 从何而来），所有 waiting_on_gapfill=True 的任务会直接
        按普通 skipped 处理，不会尝试自动接入。

    Returns
    -------
    ExecutionState
        has_clarify=True 表示 Plan 中存在 CLARIFY_NEEDED 任务，
        调用方应向用户展示 blocking_reason 并追问，而不是简单报告"部分跳过"。
    """
    state = ExecutionState()
    live_env = dict(env) if env else {}
    # 本次 dispatch_plan 调用内，gap-fill 只允许触发一次接入（多个任务共享
    # 结果），避免复合请求里每个缺配置的任务都各自触发一次 Aspen 扫描。
    # 同一个值也用于"任何任务（不只是 gap-fill 插入的）产出了 case_config_path
    # 后，后续 waiting_on_gapfill 任务应直接复用它，不再重复触发接入"——
    # 比如用户自己说"先接入再优化"，第一步显式接入成功后，第二步不该再
    # 自动插入一次接入。
    gapfill_case_config_path: str | None = None
    gapfill_attempted = False
    # 记录本轮（这次 dispatch_plan 调用）新产出的 case_config_path 集合，
    # 供 RUN_OPTIMIZATION 的安全刹车判断"这份配置是不是刚生成的"。
    freshly_produced_config_paths: set[str] = set()

    def _record_produced(produced_paths: dict[str, str]) -> None:
        nonlocal gapfill_case_config_path
        if "case_config_path" in produced_paths:
            path = produced_paths["case_config_path"]
            live_env["case_config_path"] = path
            freshly_produced_config_paths.add(path)
            if gapfill_case_config_path is None:
                gapfill_case_config_path = path
        if "db_path" in produced_paths:
            live_env["db_path"] = produced_paths["db_path"]

    for task in plan.tasks:
        if task.intent_type == IntentType.CLARIFY_NEEDED:
            state.has_clarify = True
            state.add_step(
                task.task_id, task.intent_type, "skipped",
                skipped_reason=task.blocking_reason,
            )
            continue

        if not task.ready:
            if not task.waiting_on_gapfill:
                state.add_step(
                    task.task_id, task.intent_type, "skipped",
                    skipped_reason=task.blocking_reason,
                )
                continue

            # ── gap-fill：尝试先用已知路径重建，缺失时自动插入接入任务 ──────
            if gapfill_case_config_path is not None:
                rebuilt = rebuild_task(task, {**live_env, "case_config_path": gapfill_case_config_path})
            elif not gapfill_attempted:
                gapfill_attempted = True
                gapfill_task_id = f"{task.task_id}_gapfill_onboard"
                gapfill_task = build_gapfill_onboard_task(gapfill_task_id, live_env)
                if not gapfill_task.ready:
                    # aspen_file 理论上应已知（planner 才会标 waiting_on_gapfill=True），
                    # 但仍防御性处理：接入本身缺前提时，原任务照常标 skipped。
                    state.add_step(
                        task.task_id, task.intent_type, "skipped",
                        skipped_reason=task.blocking_reason,
                    )
                    continue
                report, produced_paths, ok = _run_executor(gapfill_task, llm_config)
                if not ok:
                    state.add_step(gapfill_task_id, gapfill_task.intent_type, "error", report=report)
                    state.add_step(
                        task.task_id, task.intent_type, "skipped",
                        skipped_reason=f"自动接入失败，无法补全所需配置：{report}",
                    )
                    continue
                state.add_step(
                    gapfill_task_id, gapfill_task.intent_type, "ok", report=report,
                    produced_paths=produced_paths,
                )
                if "case_config_path" not in produced_paths:
                    state.add_step(
                        task.task_id, task.intent_type, "skipped",
                        skipped_reason="自动接入未产出配置路径，无法继续",
                    )
                    continue
                _record_produced(produced_paths)
                rebuilt = rebuild_task(task, live_env)
            else:
                # 已经尝试过一次 gap-fill 但没能拿到路径（如接入失败），
                # 不再重复尝试，直接标记跳过。
                state.add_step(
                    task.task_id, task.intent_type, "skipped",
                    skipped_reason=task.blocking_reason,
                )
                continue

            if not rebuilt.ready:
                state.add_step(
                    task.task_id, task.intent_type, "skipped",
                    skipped_reason=rebuilt.blocking_reason or task.blocking_reason,
                )
                continue
            task = rebuilt

        # ── 安全刹车：RUN_OPTIMIZATION 不得使用本轮刚产出的配置直接开跑 ──────
        used_config = task.kwargs.get("config_path")
        if (
            task.intent_type == IntentType.RUN_OPTIMIZATION
            and used_config in freshly_produced_config_paths
        ):
            state.add_step(
                task.task_id, task.intent_type, "skipped",
                skipped_reason=_skip_run_optimization_confirm(task.kwargs),
            )
            continue

        report, produced_paths, ok = _run_executor(task, llm_config)
        if ok:
            state.add_step(
                task.task_id, task.intent_type, "ok", report=report,
                produced_paths=produced_paths,
            )
            _record_produced(produced_paths)
        else:
            state.add_step(task.task_id, task.intent_type, "error", report=report)

    return state
