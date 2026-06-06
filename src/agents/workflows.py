"""
workflows.py — run_demo_case_workflow 控制骨架。

把配置解析（workflow_helpers）、状态管理（state）、报告组装（workflow_report）
串联成完整编排，通过可注入 DemoWorkflowToolRunner 调用 tools。

禁止导入：
  src.agents.tools、src.aspen_driver、src.database、src.workflows、LangGraph
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.agents.state import DemoWorkflowState
from src.agents.workflow_helpers import prepare_demo_workflow_state
from src.agents.workflow_report import build_demo_workflow_report


# ---------------------------------------------------------------------------
# Tool Runner 协议
# ---------------------------------------------------------------------------

@runtime_checkable
class DemoWorkflowToolRunner(Protocol):
    """注入到 run_demo_case_workflow 的 tool 调用接口。

    所有方法均返回字符串报告；以 "错误：" 开头表示失败。
    不依赖具体实现，便于测试时注入 FakeRunner。
    """

    def validate_config(self, case_config_path: str) -> str: ...

    def run_case(self, case_config_path: str) -> str: ...

    def optimize_pareto(
        self,
        case_config_path: str,
        db_path: str | None = None,
    ) -> str: ...

    def query_simulation_db(
        self,
        db_path: str,
        mode: str,
        status: str | None = None,
        objective_name: str | None = None,
        limit: int = 10,
        case_id: str | None = None,
    ) -> str: ...

    def diagnose_case(self, db_path: str, case_id: str) -> str: ...

    def query_node_db(
        self,
        db_path: str,
        mode: str,
        case_id: str | None = None,
        limit: int = 20,
    ) -> str: ...

    def summarize_pareto(
        self,
        db_path: str,
        objective_names: list[str],
        include_infeasible: bool = False,
    ) -> str: ...

    def get_failed_case_ids(
        self,
        db_path: str,
        limit: int = 3,
    ) -> list[str]:
        """返回最近失败工况的 case_id 列表（最多 limit 个）。

        空列表表示无失败工况。不在此方法中做文本解析，
        由 runner 内部结构化返回 UUID 字符串列表。
        """
        ...


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _is_tool_error(report: str) -> bool:
    """判断 tool 返回值是否为失败。

    识别规则（允许前导空白）：
    - 以 "错误：" 开头（全角冒号）
    - 以 "错误:" 开头（半角冒号）
    """
    stripped = report.lstrip()
    return stripped.startswith("错误：") or stripped.startswith("错误:")


def _config_path(state: DemoWorkflowState) -> str:
    """优先返回已解析的绝对路径，否则返回原始输入路径。"""
    return state.resolved_config_path or state.case_config_path


def _skip_remaining(state: DemoWorkflowState, reason: str) -> None:
    """给配置校验失败后所有未执行的下游 step 统一标记 skipped。"""
    for name in ("run_case", "optimize_pareto",
                 "query_simulation_db", "diagnose_case",
                 "query_node_db", "summarize_pareto"):
        if state.get_step(name) is None:
            state.add_step(name, "skipped", skipped_reason=reason)


# ---------------------------------------------------------------------------
# 步骤实现
# ---------------------------------------------------------------------------

def _step_validate(state: DemoWorkflowState, runner: DemoWorkflowToolRunner) -> bool:
    """执行 validate_config 步骤。返回 True 表示可继续，False 表示中止。"""
    try:
        report = runner.validate_config(_config_path(state))
    except Exception as exc:
        report = f"错误：validate_config 调用抛出异常 [{type(exc).__name__}] — {exc}"

    if _is_tool_error(report) or DemoWorkflowState.is_validate_fatal(report):
        state.add_step("validate_config", "error", report=report)
        state.aborted = True
        _skip_remaining(state, "配置校验失败")
        return False
    state.add_step("validate_config", "ok", report=report)
    return True


def _step_run_or_optimize(
    state: DemoWorkflowState, runner: DemoWorkflowToolRunner
) -> None:
    """执行运行/优化分支，并给不适用的另一分支打 skipped。"""
    cfg = _config_path(state)
    if state.is_pareto_branch:
        try:
            report = runner.optimize_pareto(cfg, db_path=state.db_path)
        except Exception as exc:
            report = f"错误：optimize_pareto 调用抛出异常 [{type(exc).__name__}] — {exc}"
        status = "error" if _is_tool_error(report) else "ok"
        state.add_step("optimize_pareto", status, report=report)
        state.add_step(
            "run_case", "skipped",
            skipped_reason="pareto_bayesian 分支不执行单次仿真",
        )
    else:
        try:
            report = runner.run_case(cfg)
        except Exception as exc:
            report = f"错误：run_case 调用抛出异常 [{type(exc).__name__}] — {exc}"
        status = "error" if _is_tool_error(report) else "ok"
        state.add_step("run_case", status, report=report)
        state.add_step(
            "optimize_pareto", "skipped",
            skipped_reason="非 pareto_bayesian 分支不执行 Pareto 优化",
        )


def _step_query_simulation_db(
    state: DemoWorkflowState, runner: DemoWorkflowToolRunner
) -> None:
    """仅 pareto_bayesian 且 db_path 非 None 时查询 SimulationDB。"""
    if not state.is_pareto_branch or state.db_path is None:
        state.add_step(
            "query_simulation_db", "skipped",
            skipped_reason="当前分支无 SimulationDB 查询路径",
        )
        return

    try:
        report = runner.query_simulation_db(
            db_path=state.db_path,
            mode="query",
            status=None,
            limit=10,
        )
    except Exception as exc:
        report = f"错误：query_simulation_db 调用抛出异常 [{type(exc).__name__}] — {exc}"
    status = "error" if _is_tool_error(report) else "ok"
    state.add_step("query_simulation_db", status, report=report)


def _step_diagnose(state: DemoWorkflowState, runner: DemoWorkflowToolRunner) -> None:
    """诊断阶段：通过 runner.get_failed_case_ids 获取结构化失败 case_id，
    再调用 diagnose_case 和 query_node_db。

    调用条件（全部满足时才执行）：
    - state.is_pareto_branch is True
    - state.db_path is not None
    - query_simulation_db step 存在且 status == "ok"
    - runner 提供 get_failed_case_ids（用 hasattr 检测，非强制）

    不从任何自然语言 report 解析 case_id，不设置 case_id_source="text_fallback"。
    """
    # ── 前置条件检查 ──────────────────────────────────────────────────────────
    db_query_step = state.get_step("query_simulation_db")
    if (
        not state.is_pareto_branch
        or state.db_path is None
        or db_query_step is None
        or db_query_step.status != "ok"
    ):
        reason = "缺少可靠失败工况 case_id"
        state.add_step("diagnose_case", "skipped", skipped_reason=reason)
        state.add_step("query_node_db", "skipped", skipped_reason=reason)
        return

    if not hasattr(runner, "get_failed_case_ids"):
        reason = "runner 未提供结构化失败 case_id 接口"
        state.add_step("diagnose_case", "skipped", skipped_reason=reason)
        state.add_step("query_node_db", "skipped", skipped_reason=reason)
        return

    # ── 获取失败 case_id ──────────────────────────────────────────────────────
    try:
        raw_ids: list[str] = runner.get_failed_case_ids(
            db_path=state.db_path, limit=3
        )
    except Exception as exc:
        err_report = f"错误：结构化失败 case_id 查询异常 [{type(exc).__name__}] — {exc}"
        state.add_step("diagnose_case", "error", report=err_report)
        state.add_step(
            "query_node_db", "skipped",
            skipped_reason="失败 case_id 查询失败",
        )
        return

    # 过滤空字符串，最多保留 3 个
    case_ids = [cid for cid in (raw_ids or []) if cid and cid.strip()][:3]

    if not case_ids:
        state.diagnostic_case_ids = []
        state.case_id_source = ""
        reason = "未查询到失败工况 case_id"
        state.add_step("diagnose_case", "skipped", skipped_reason=reason)
        state.add_step("query_node_db", "skipped", skipped_reason=reason)
        return

    state.diagnostic_case_ids = case_ids
    state.case_id_source = "db_query"

    # ── 对每个 case_id 调用 diagnose_case ────────────────────────────────────
    diag_parts: list[str] = []
    any_diag_error = False
    for cid in case_ids:
        try:
            diag_report = runner.diagnose_case(
                db_path=state.db_path, case_id=cid
            )
        except Exception as exc:
            diag_report = (
                f"错误：diagnose_case({cid!r}) 调用抛出异常"
                f" [{type(exc).__name__}] — {exc}"
            )
        if _is_tool_error(diag_report):
            any_diag_error = True
        diag_parts.append(f"--- case_id: {cid} ---\n{diag_report}")

    combined_diag = "\n\n".join(diag_parts)
    diag_status = "error" if any_diag_error else "ok"
    state.add_step("diagnose_case", diag_status, report=combined_diag)

    # ── query_node_db：用第一个 case_id，优先 node_db_path ───────────────────
    node_db = state.node_db_path or state.db_path
    first_id = case_ids[0]
    try:
        node_report = runner.query_node_db(
            db_path=node_db,
            mode="node_values",
            case_id=first_id,
            limit=20,
        )
    except Exception as exc:
        node_report = (
            f"错误：query_node_db 调用抛出异常 [{type(exc).__name__}] — {exc}"
        )
    node_status = "error" if _is_tool_error(node_report) else "ok"
    state.add_step("query_node_db", node_status, report=node_report)


def _step_summarize_pareto(
    state: DemoWorkflowState, runner: DemoWorkflowToolRunner
) -> None:
    """仅 pareto_bayesian + db_path 非 None + optimize_pareto ok 时执行。"""
    opt_step = state.get_step("optimize_pareto")
    if (
        not state.is_pareto_branch
        or state.db_path is None
        or opt_step is None
        or opt_step.status != "ok"
    ):
        state.add_step(
            "summarize_pareto", "skipped",
            skipped_reason="优化失败或当前分支不适用",
        )
        return

    try:
        report = runner.summarize_pareto(
            db_path=state.db_path,
            objective_names=state.objective_names,
            include_infeasible=False,
        )
    except Exception as exc:
        report = f"错误：summarize_pareto 调用抛出异常 [{type(exc).__name__}] — {exc}"
    status = "error" if _is_tool_error(report) else "ok"
    state.add_step("summarize_pareto", status, report=report)


# ---------------------------------------------------------------------------
# 规则型 next_actions 生成器
# ---------------------------------------------------------------------------

def determine_next_actions(state: DemoWorkflowState) -> list[str]:
    """根据 state 中各步骤的执行结果，用规则推断下一步建议列表。

    规则优先级（从高到低）：
    1. load_config 失败 → 修复配置文件后重试
    2. validate_config 失败 → 修复校验问题（YAML / Python dry-run）
    3. optimize_pareto / run_case 失败 → 检查 Aspen 环境或工况设置
    4. query_simulation_db 失败 → 根据上游运行状态分支建议
    5. diagnose_case 成功（有失败工况被诊断出来）→ 根据诊断建议调整参数边界
    6. summarize_pareto 失败（optimize_pareto 成功但汇总失败）→ 手动查询 DB
    7. summarize_pareto 成功（Pareto 分支）→ 进入结果筛选与能耗/经济分析闭环
    7b. run_case 成功（单次分支）→ 确认收敛与关键输出，再决定是否转入优化
    fallback. 无任何规则触发 → 根据是否有错误给出通用提示

    多条规则可同时触发；每条规则只在其前置条件确实发生时追加建议，
    不追加未发生的情形的建议，也不重复追加同一条建议。
    """
    actions: list[str] = []

    def _step_status(name: str) -> str:
        """返回步骤的 status 字符串，未执行时返回 ''。"""
        step = state.get_step(name)
        return step.status if step is not None else ""

    # ── 规则 1：load_config 失败 ─────────────────────────────────────────────
    if _step_status("load_config") == "error":
        actions.append(
            "配置文件加载失败：请检查路径是否正确，以及 YAML 文件是否存在且可读。"
        )
        actions.append(
            "修复配置文件后，重新运行 workflow。"
        )
        return actions  # 后续步骤全部 skipped，无需继续推断

    # ── 规则 2：validate_config 失败 ────────────────────────────────────────
    if _step_status("validate_config") == "error":
        actions.append(
            "配置校验失败：请查看【2. 校验结果】中的 validate_config 错误详情，"
            "修复 YAML 格式、字段完整性或 Python dry-run 失败问题。"
        )
        actions.append(
            "修复后可重新运行 validate_config_tool 验证，再执行完整 workflow。"
        )
        return actions  # aborted，后续无实质步骤

    # ── 规则 3：optimize_pareto / run_case 失败 ──────────────────────────────
    opt_status = _step_status("optimize_pareto")
    run_status = _step_status("run_case")
    if opt_status == "error" or run_status == "error":
        failed_step = "optimize_pareto" if opt_status == "error" else "run_case"
        actions.append(
            f"{failed_step} 执行失败：请确认 Aspen Plus 已启动、许可证有效，"
            "以及设计变量边界和初始值在允许范围内。"
        )
        actions.append(
            "可先运行 preflight_full_aspen.py 检查环境，再重新触发 workflow。"
        )

    # ── 规则 4：query_simulation_db 失败 ────────────────────────────────────
    # 根据上游运行/优化状态分支，避免建议与已知失败状态矛盾
    if _step_status("query_simulation_db") == "error":
        upstream_failed = opt_status == "error" or run_status == "error"
        if upstream_failed:
            actions.append(
                "SimulationDB 查询失败，且上游运行/优化步骤已失败：数据库可能未写入，"
                "建议先修复运行/优化失败后重新运行，数据库查询将随之恢复正常。"
            )
        else:
            actions.append(
                "SimulationDB 查询失败：请确认 simulation.db 路径正确、"
                "SQLite 文件存在且可读，以及运行/优化步骤已成功写入数据库。"
            )

    # ── 规则 5：diagnose_case 成功（有失败工况被诊断）───────────────────────
    if _step_status("diagnose_case") == "ok" and state.diagnostic_case_ids:
        actions.append(
            f"发现 {len(state.diagnostic_case_ids)} 个失败工况已完成诊断："
            "请查看【5. 诊断结论】中的诊断报告，根据建议调整设计变量边界或初始值后重试。"
        )

    # ── 规则 6：summarize_pareto 失败（optimize_pareto 成功）────────────────
    if opt_status == "ok" and _step_status("summarize_pareto") == "error":
        actions.append(
            "Pareto 汇总失败但优化本身已完成：可直接用 query_simulation_db_tool "
            "mode='pareto' 手动读取 Pareto 前沿数据。"
        )

    # ── 规则 7：summarize_pareto 成功（Pareto 分支）─────────────────────────
    if _step_status("summarize_pareto") == "ok":
        actions.append(
            "Pareto 优化与汇总均已成功完成：请查看【6. Pareto 总结】中的前沿分布，"
            "筛选满足产品纯度、流量或能耗约束的候选操作点，进行能耗与经济性分析。"
        )
        actions.append(
            "确认当前 Pareto 前沿覆盖度后，可收窄设计变量范围进行局部精化，"
            "或调整目标权重后进行下一轮多目标优化。"
        )

    # ── 规则 7b：run_case 成功（单次非 Pareto 分支）─────────────────────────
    elif run_status == "ok" and not state.is_pareto_branch:
        actions.append(
            "单次仿真已成功完成：请确认关键输出（产品纯度、流量、再沸器负荷等）"
            "数值合理、单位正确，以及 Aspen 模块均收敛。"
        )
        actions.append(
            "确认结果无误后，可考虑转入多目标 Pareto 优化（optimizer.type: pareto_bayesian），"
            "进一步探索设计变量的权衡空间。"
        )

    # ── 无任何规则触发（例如全部 skipped 但无错误）──────────────────────────
    if not actions:
        if state.has_errors():
            actions.append(
                "存在失败步骤：请查看各步骤的错误详情，定位根本原因后重新运行。"
            )
        else:
            actions.append(
                "所有已执行步骤均通过，可根据结果决定是否调整约束条件或进入下一阶段分析。"
            )

    return actions


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------

def run_demo_case_workflow(
    case_config_path: str,
    tool_runner: DemoWorkflowToolRunner,
) -> str:
    """执行 demo case workflow 并返回完整文本报告。

    Args:
        case_config_path: YAML 配置文件路径（相对或绝对）。
        tool_runner:      实现 DemoWorkflowToolRunner 协议的 runner，
                          测试时传入 FakeRunner，生产时传入真实 tool 适配器。

    Returns:
        完整文本报告，包含 7 个固定章节。任何异常都转换为报告内容，
        函数本身不向调用方抛异常。
    """
    # 1. 配置解析 —— 失败时构造最小 state 后直接返回报告
    try:
        state = prepare_demo_workflow_state(case_config_path)
    except Exception as exc:
        state = DemoWorkflowState(case_config_path=case_config_path)
        state.aborted = True
        state.add_step("load_config", "error", report=str(exc))
        _skip_remaining(state, "配置解析失败")
        state.next_actions = determine_next_actions(state)
        return build_demo_workflow_report(state)

    state.add_step("load_config", "ok", report="配置解析完成")

    # 2. 配置校验
    if not _step_validate(state, tool_runner):
        state.next_actions = determine_next_actions(state)
        return build_demo_workflow_report(state)

    # 3. 运行/优化分支
    _step_run_or_optimize(state, tool_runner)

    # 4. 数据库查询
    _step_query_simulation_db(state, tool_runner)

    # 5. 诊断（本任务留 skipped，任务六扩展）
    _step_diagnose(state, tool_runner)

    # 6. Pareto 总结
    _step_summarize_pareto(state, tool_runner)

    state.next_actions = determine_next_actions(state)
    return build_demo_workflow_report(state)
