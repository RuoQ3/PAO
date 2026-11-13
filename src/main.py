"""
main.py — PAO 命令行入口。

用法
----
    python -m src.main cases/demo_case/case_config.yaml [选项]

选项
----
    --db PATH       结果数据库路径（默认：与 YAML 同目录的 output/simulation.db）
    --log LEVEL     日志级别（DEBUG/INFO/WARNING，默认 INFO）
    --dry-run       只加载配置并打印摘要，不运行 Aspen 仿真
    --conversation  启用 AI Agent 协作模式。行为按传入的文件类型分支：
                      - 传入已有的 YAML 配置：boundary_advisor 扫描已有 design_variables
                        推荐边界，终端交互确认/修改，再运行 Pareto 优化（原有行为不变）。
                      - 传入裸 Aspen 文件（.bkp/.apw，尚无配置）：走 Lead Agent 只读接入
                        意图识别 + 任务拆解，按 --intent 描述的需求路由到
                        onboarding_agent / boundary_advisor / kinetics_advisor /
                        process_advisor / optimize_pareto 中的一个或多个。
    --intent TEXT   自由文本描述。YAML 输入时是"优化意图"（如边界推荐的工艺背景）；
                    Aspen 文件输入时是"任务请求"（如"帮我接入这个新工艺" /
                    "帮我拟合这批实验数据的活化能"），由 Lead Agent 路由到 Subagent。
    --data CSV_PATH 实验数据 CSV 路径（Aspen 文件输入且意图涉及动力学拟合时使用）。
    # removed legacy --conversation: 连续对话模式（仅裸 Aspen 文件 + --conversation 时可用）：进入 REPL 循环，
                    可连续输入多轮指令；系统记住上一轮产出的配置/数据库路径，
                    后续提问不用重复给路径（但不记住数值结果，如拟合出的 Ea 数值）。

示例
----
    # 单目标贝叶斯优化（直接读 YAML 配置）
    python -m src.main cases/demo_case/case_config.yaml

    # 多目标 Pareto 优化（optimizer.type: pareto_bayesian）
    python -m src.main cases/demo_case/pareto_tac_emissions_config.yaml

    # AI Agent 协作模式：已有 YAML，boundary_advisor 推荐边界 + 终端 HITL
    python -m src.main cases/demo_case_2/pareto_config_epsd_aligned.yaml --conversation
    python -m src.main cases/demo_case_2/pareto_config_epsd_aligned.yaml --conversation \\
        --intent "最小化总年费用和碳排放，约束甘油纯度 > 99%"

    # AI Agent 协作模式：裸 Aspen 文件进入 Lead Agent + Subagent 路由。
    python -m src.main cases/demo_case/二级氢氰化工段.bkp --conversation \\
        --intent "帮我接入这个新工艺，分析可调变量"
    python -m src.main cases/demo_case/二级氢氰化工段.bkp --conversation \\
        --intent "帮我拟合这批实验数据的活化能" --data scripts/kinetics_example_data.csv

    # AI Agent 协作模式：连续对话（多轮，记住上一轮产出的配置路径）
    python -m src.main cases/demo_case/二级氢氰化工段.bkp --conversation
    PAO> 帮我接入这个工艺
    PAO> 帮我推荐一下边界    # 自动复用上一轮生成的配置路径
    PAO> exit

    python -m src.main cases/demo_case/case_config.yaml --db output/run1.db --log DEBUG
    python -m src.main cases/demo_case/case_config.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Windows 控制台默认 GBK 编码，Agent 模式的报告文本含 emoji/"±"等字符会
# 导致 UnicodeEncodeError；强制 stdout/stderr 走 UTF-8，不影响非 Windows 环境
# （reconfigure 是幂等操作）。与 scripts/run_kinetics_advisor.py 的处理一致。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stdin, "reconfigure"):
    # PowerShell may expose the console input stream with the legacy system
    # code page.  Agent prompts are sent to the LLM as UTF-8 JSON, so decode
    # interactive turns consistently before classification.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def _setup_logging(level: str, log_file: Path | None = None, *,
                   conversation: bool = False, console_level: str | None = None,
                   file_level: str | None = None) -> None:
    from src.utils.logger import setup_logging
    setup_logging(level, log_file=log_file,
                  console_level=console_level or ("WARNING" if conversation else level),
                  file_level=file_level or ("DEBUG" if conversation else level),
                  user_console=conversation and console_level is None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="PAO — Aspen Plus 贝叶斯优化框架",
    )
    parser.add_argument("config", nargs="?", help="已有 YAML 或 Aspen 模型路径；--new-project 可省略")
    parser.add_argument("--new-project", action="store_true", help="进入无需手写 YAML 的项目建立对话")
    parser.add_argument("--project-draft", default=None, metavar="PATH", help="恢复项目草稿 JSON；不恢复执行授权")
    parser.add_argument("--db",       default=None, help="结果数据库路径（.db）")
    parser.add_argument("--log",      default="INFO", help="日志级别（默认 INFO）")
    parser.add_argument("--console-log", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default=None, help="显式启用终端诊断及其级别；对话模式默认仅显示回答与进度")
    parser.add_argument("--file-log", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default=None, help="文件日志级别；对话模式默认 DEBUG，其他模式沿用 --log")
    parser.add_argument("--log-file", default=None, metavar="PATH",
                        help="日志文件路径（默认：与数据库同目录的 run.log）")
    parser.add_argument("--dry-run",  action="store_true", help="只打印配置摘要，不运行仿真")
    parser.add_argument("--checkpoint", default=None, help="Agent 闭环 checkpoint 数据库路径")
    parser.add_argument("--resume", action="store_true", help="恢复同配置的 Agent 闭环")
    parser.add_argument("--intent",   default="",
                        help="优化意图自然语言描述（--conversation 模式使用；传入 Aspen 文件时"
                             "作为 Lead Agent 的自由文本请求，路由到合适的 Subagent）")
    parser.add_argument("--data",     default=None, metavar="CSV_PATH",
                        help="实验数据 CSV 路径（--conversation 模式下传入 Aspen 文件且意图涉及"
                             "动力学参数拟合时使用，列：temperature,rate_constant）")
    parser.add_argument("--conversation", action="store_true",
                        help="启用统一只读工程对话：ProcessContext + IMA + 本地历史分析")
    parser.add_argument("--yaml", dest="yaml_path", default=None, metavar="PATH",
                        help="统一对话使用的 YAML 配置路径")
    parser.add_argument("--node-db", dest="node_db_path", default=None, metavar="PATH",
                        help="NodeDB 路径（可选）")
    parser.add_argument("--file", dest="file_paths", action="append", default=[], metavar="PATH",
                        help="明确选择供专业角色只读访问的 UTF-8 文件，可重复指定；不执行代码")
    args = parser.parse_args(argv)
    if args.new_project or args.project_draft:
        args.conversation = True
    if not args.config and not args.new_project and not args.project_draft:
        parser.error("需要已有配置路径，或使用 --new-project / --project-draft")
    return args


def _print_single_summary(log: logging.Logger, result: object, db_path: Path) -> None:
    """打印单目标优化结果摘要。"""
    summary = result.to_summary()  # type: ignore[union-attr]
    log.info("=" * 60)
    log.info("优化完成")
    log.info("  总工况数：%d", summary["n_total"])
    log.info("  成功工况：%d（成功率 %.1f%%）", summary["n_success"],
             summary["success_rate"] * 100)
    log.info("  仿真失败：%d", summary["n_sim_failed"])
    log.info("  目标错误：%d", summary["n_objective_error"])
    log.info("  总耗时：%.1f s", summary["elapsed"])
    if result.best_case is not None:  # type: ignore[union-attr]
        log.info("  最优 %s = %.6g", result.objective_name, result.best_value)  # type: ignore[union-attr]
        log.info("  最优参数：")
        for path, val in result.best_case.design_vars.items():  # type: ignore[union-attr]
            log.info("    %s = %s", path.split("\\")[-1], val)
    else:
        log.warning("  未找到有效最优解（所有工况均失败或目标不可用）。")
    log.info("  结果已保存至：%s", db_path)
    log.info("=" * 60)


def _print_pareto_summary(log: logging.Logger, result: object, db_path: Path) -> None:
    """打印多目标 Pareto 优化结果摘要。"""
    summary = result.to_summary()  # type: ignore[union-attr]
    log.info("=" * 60)
    log.info("多目标优化完成")
    log.info("  总工况数：%d", summary["n_total"])
    log.info("  成功工况：%d（成功率 %.1f%%）", summary["n_success"],
             summary["success_rate"] * 100)
    log.info("  仿真失败：%d", summary["n_sim_failed"])
    log.info("  目标错误：%d", summary["n_objective_error"])
    log.info("  总耗时：%.1f s", summary["elapsed"])
    log.info("  Pareto 层数：%d", summary["n_fronts"])
    log.info("  第一前沿解数：%d", summary["first_front_size"])
    hv = summary.get("hypervolume")
    log.info("  超体积（HV）：%s", f"{hv:.4g}" if hv is not None else "N/A")

    front = result.first_front  # type: ignore[union-attr]
    if front is None:
        log.warning("  未找到有效 Pareto 前沿（所有工况均失败或约束违反）。")
    else:
        obj_names = front.objective_names
        log.info("  第一前沿（按拥挤距离降序）：")
        for i, (case, vec, cd) in enumerate(
            zip(front.cases, front.objective_vectors, front.crowding_distances)
        ):
            obj_str = "  ".join(
                f"{n}={v:.4g}" for n, v in zip(obj_names, vec)
            )
            dv_str = "  ".join(
                f"{k.split(chr(92))[-1]}={v:.4f}"
                for k, v in case.design_vars.items()
            )
            cd_str = f"{cd:.3f}" if cd != float("inf") else "∞"
            log.info("    [%d] %s  |  %s  |  cd=%s", i + 1, obj_str, dv_str, cd_str)

    log.info("  结果已保存至：%s", db_path)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Agent 模式：boundary_advisor + 自动跑优化
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 60) -> None:
    print(char * width)


def _load_var_metas(yaml_path: Path) -> tuple[list, dict, set]:
    """
    从 YAML design_variables 构造 VarMeta 列表，与 run_boundary_advisor.py 逻辑完全一致。

    Returns
    -------
    (variables, var_types, integer_names)
    """
    import yaml as _yaml
    from src.agents.subagents.boundary_advisor import VarMeta

    with open(yaml_path, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)

    variables: list = []
    var_types: dict = {}
    integer_names: set = set()

    def _infer_role(name: str, path: str, unit: str, var_type: str) -> str:
        n = (name + " " + path).lower()
        u = unit.lower()
        if "pres" in n or "atm" in u or "bar" in u or "pa" in u:
            return "pressure"
        if "basis_rr" in n or "reflux" in n or "rr" in n:
            return "reflux_ratio"
        if "nstage" in n:
            return "nstage"
        if "feed" in n or "frac" in n:
            return "feed_stage"
        if "totflow" in n or "flow" in n or "sol" in n or "kmol" in u:
            return "flow"
        return ""

    for dv in cfg.get("design_variables", []):
        dv_type = dv.get("type", "continuous")
        name = dv.get("name") or dv.get("aspen_path")
        if not name:
            continue
        name = str(name)
        unit = str(dv.get("unit", "") or "")
        iv = dv.get("initial_value")
        try:
            iv_f = float(iv) if iv is not None else None
        except (TypeError, ValueError):
            iv_f = None
        lo = dv.get("lower_bound")
        hi = dv.get("upper_bound")
        if dv_type == "derived":
            lo = dv.get("lo_frac", lo)
            hi = dv.get("hi_frac", hi)
        try:
            lo_f = float(lo) if lo is not None else None
            hi_f = float(hi) if hi is not None else None
        except (TypeError, ValueError):
            lo_f = hi_f = None

        role = _infer_role(name, dv.get("aspen_path", ""), unit, dv_type)
        variables.append(VarMeta(
            name=name,
            initial_value=iv_f,
            unit=unit,
            semantic_role=role,
            var_type=dv_type,
            lower_global=lo_f,
            upper_global=hi_f,
        ))
        var_types[name] = dv_type
        if dv_type == "integer":
            integer_names.add(name)

    return variables, var_types, integer_names


def _print_backtest_progress(event: dict) -> None:
    """Render bounded progress independently of diagnostic logging."""
    status = event["status"]
    completed, total = event["completed"], event["total"]
    elapsed = event["elapsed_seconds"]
    labels = {"started": "开始执行", "running": "运行中", "completed": "工况完成",
              "failed": "工况失败", "stopped": "已停止派发后续工况；底层状态见执行证据",
              "timeout": "等待超时，尚未确认底层停止", "cleaning": "正在尝试清理",
              "cleanup_completed": "工作单元清理流程已返回；Aspen 进程状态见证据",
              "cleanup_incomplete": "清理未完成，资源状态需核查",
              "finished": "回测结束"}
    from src.agents.orchestration.models import Answer
    from src.reporting.answer_terminal import TerminalRenderer
    print(TerminalRenderer().render(Answer(
        f"回测进度 {completed}/{total}：{labels.get(status, status)}（已用时 {elapsed:.0f}s）")), flush=True)


def _run_unified_chat_loop(
    *, config_path: Path, yaml_path: Path | None, db_path: Path | None,
    node_db_path: Path | None, initial_intent: str, log: logging.Logger,
    new_project: bool = False, project_draft: str | None = None,
    file_paths: list[str] | None = None,
) -> int:
    """Run the unified Lead Agent conversation entry point."""
    from src.agents.lead import ConversationSession, LeadAgent
    from src.agents.lead.intent import IntentClassifier
    from src.agents.errors import AgentInternalError, AgentUnavailableError
    from src.agents.reasoning import configured
    try:
        llm_config = configured()
        IntentClassifier(llm_config=llm_config).classify("Hello")
    except AgentUnavailableError as exc:
        print(exc.user_message)
        log.error("%s: %s", "agent_internal_error" if isinstance(exc, AgentInternalError) else "agent_unavailable", exc.to_dict())
        return 2
    from src.agents.orchestration import RuntimePolicy, SessionState
    from src.agents.context import ProcessContextBuilder
    from src.agents.actions import ActionGateway
    from src.agents.actions.models import ModelIdentity, ModelSchema, ParameterSpec
    from src.agents.orchestration.parameter_impact import AspenSandboxRunner
    from src.database.simulation_db import SimulationDB

    aspen_file = config_path if config_path.suffix.lower() in (".bkp", ".apw") else None
    if yaml_path is None and config_path.suffix.lower() in (".yaml", ".yml"):
        yaml_path = config_path
    builder = ProcessContextBuilder(
        yaml_path=yaml_path, aspen_file=aspen_file,
        node_db=node_db_path if node_db_path and node_db_path.exists() else None,
        simulation_db=db_path if db_path and db_path.exists() else None, read_only=True,
    )
    # Build the gateway at the CLI boundary.  The context remains read-only;
    # Aspen is opened lazily by the runner only after preview + approval.
    sim_db = None
    def create_gateway(optimization_plan=None):
        nonlocal sim_db
        current_builder = session.lead.context_builder
        context = current_builder.build(session=session.state)
        selected_yaml = getattr(current_builder, "yaml_path", None) or session.state.context_refs.get("case_config")
        if selected_yaml:
            from src.agents.actions.plans import build_optimization_plan
            # Only complete optimization configurations expose optimization approval.
            if optimization_plan is not None or context.configuration.get("execution"):
                if optimization_plan is not None and Path(optimization_plan["config_path"]).resolve() != Path(selected_yaml).resolve():
                    raise ValueError("optimization proposal does not match selected project")
                _, schema = build_optimization_plan(selected_yaml, **(
                    {"overrides": optimization_plan["configuration_overrides"]}
                    if optimization_plan and optimization_plan.get("configuration_overrides") else {}))
                sim_db = SimulationDB(db_path)
                runner = AspenSandboxRunner(driver_factory=lambda _path: __import__("src.aspen_driver.driver", fromlist=["AspenDriver"]).AspenDriver(),
                    allowed_paths=set(schema.parameters), require_path_whitelist=True, progress_callback=_print_backtest_progress)
                return ActionGateway(schema, simulation_runner=runner, simulation_db=sim_db)
        params = {}
        for item in [
            p for values in (getattr(context, "equipment_parameters", {}) or {}).values()
            for p in (values or []) if isinstance(p, dict)
        ] + list((getattr(context, "variables", {}) or {}).values()):
            path = str(item.get("aspen_path") or item.get("path") or "")
            if "\\input\\" not in path.casefold():
                continue
            params[path] = ParameterSpec(path=path, value_type="int" if str(item.get("type", "")).casefold() in {"int", "integer"} else "float",
                                         unit=item.get("unit"), writable=item.get("authorized", item.get("tunable", path in {v.get("aspen_path") or v.get("path") for v in (getattr(context, "variables", {}) or {}).values()})) is True, minimum=item.get("lower", item.get("lower_bound")),
                                         maximum=item.get("upper", item.get("upper_bound")), integer=bool(item.get("integer")))
        schema = ModelSchema(ModelIdentity(str(getattr(context, "model_id", None) or "model")), parameters=params,
                             model_path=str(current_builder.resolve_model_resource(session=session.state).path or ""),
                             config_path=str(selected_yaml) if selected_yaml else None,
                             current_values=getattr(context, "current_values", {}) or {},
                             permissions={"inspect", "run_simulation"}, targets=("model", ""))
        sim_db = SimulationDB(db_path)
        runner = AspenSandboxRunner(driver_factory=lambda _path: __import__("src.aspen_driver.driver", fromlist=["AspenDriver"]).AspenDriver(),
                                    allowed_paths=set(params), require_path_whitelist=True,
                                    progress_callback=_print_backtest_progress)
        return ActionGateway(schema, simulation_runner=runner, simulation_db=sim_db)
    class LazyGateway:
        instance = None
        def preview_proposal(self, proposal):
            plan = proposal.payload.get("optimization_plan")
            if plan is not None:
                self.reset_project()
                self.instance = create_gateway(plan)
            elif self.instance is None:
                self.instance = create_gateway()
            return self.instance.preview_proposal(proposal)
        def reset_project(self):
            nonlocal sim_db
            self.instance = None
            if sim_db is not None:
                sim_db.close()
                sim_db = None
        def __getattr__(self, name):
            if self.instance is None:
                self.instance = create_gateway()
            return getattr(self.instance, name)
    gateway = LazyGateway()
    session = ConversationSession(SessionState(), LeadAgent(
        policy=RuntimePolicy(allow_writes=False), context_builder=builder,
        action_gateway=gateway, use_llm=True, llm_config=llm_config))
    if file_paths:
        session.state.context_refs["files"] = list(file_paths)
    from src.reporting.answer_terminal import TerminalRenderer
    from src.reporting.answer_export import markdown_answer
    from src.agents.orchestration.models import Answer
    renderer = TerminalRenderer()
    audit_path = (db_path.parent if db_path else config_path.parent) / f"conversation-{session.state.session_id}.jsonl"
    audit_cursor = 0

    def save_audit():
        nonlocal audit_cursor
        from src.agents.reasoning import redact
        events = session.state.audit_events[audit_cursor:]
        if events:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as stream:
                for event in events:
                    stream.write(json.dumps(redact(event, llm_config), ensure_ascii=False, default=str) + "\n")
            audit_cursor += len(events)
    print("\nPAO Lead Agent conversation")
    print("输入问题，exit 退出；/details 查看详情；/value p1 查看路径；/export 文件.json 或 .md 导出；/project new 建立项目；/capabilities 查看能力")

    def run_turn(text: str) -> bool:
        if text.strip().lower() in {"/exit", "exit", "quit"}:
            return True
        if not text.strip():
            print("请输入非空问题。")
            return False
        command, _, argument = text.strip().partition(" ")
        if command in {"/details", "/answer", "/value", "/export"}:
            answer = session.last_answer
            if answer is None:
                print("当前没有可查看的回答。")
                return False
            if command in {"/details", "/answer"}:
                # Detail inspection must not invalidate the current path IDs.
                print((TerminalRenderer() if command == "/details" else renderer).render(answer, details=command == "/details"))
            elif command == "/value":
                try:
                    print(renderer.render_value(argument.strip()))
                except KeyError:
                    print("路径编号不存在；输入 /answer 重新显示编号。")
            else:
                destination = Path(argument.strip().strip('"'))
                if not argument or destination.suffix.lower() not in {".json", ".md"}:
                    print("用法：/export 文件.json 或 /export 文件.md")
                    return False
                try:
                    # Never silently overwrite an existing artifact.
                    with destination.open("x", encoding="utf-8") as stream:
                        stream.write(markdown_answer(answer) if destination.suffix.lower() == ".md" else answer.to_json(indent=2))
                    print(TerminalRenderer().render(Answer(f"已导出：{destination}")))
                except OSError as exc:
                    print(TerminalRenderer().render(Answer(f"导出失败：{exc}")))
            return False
        if command.startswith("/") and command not in {"/project", "/capabilities"}:
            print("未知命令；可用 /answer、/details、/value、/export、/exit。")
            return False
        try:
            result = session.submit(text)
        except (ValueError, OSError) as exc:
            print(renderer.render(Answer(str(exc))))
            return False
        finally:
            save_audit()
        print(renderer.render(result))
        return False

    try:
        if project_draft:
            run_turn('/project resume ' + project_draft)
        elif new_project:
            selected = str(config_path) if config_path.suffix.lower() in {".bkp", ".apw"} else ""
            run_turn('/project from-config ' + str(yaml_path) if yaml_path else '/project new ' + selected)
            if node_db_path:
                run_turn('/project node-db ' + str(node_db_path))
        if initial_intent.strip() and run_turn(initial_intent):
            return 0
        while True:
            try:
                text = input("\nPAO> ")
            except (EOFError, KeyboardInterrupt):
                print("\n已退出")
                return 0
            if run_turn(text):
                print("已退出")
                return 0
    except AgentUnavailableError as exc:
        print(renderer.render(Answer(exc.user_message + f"\n审计记录：{audit_path}")))
        log.error("%s: %s", "agent_internal_error" if isinstance(exc, AgentInternalError) else "agent_unavailable", exc.to_dict())
        return 2
    finally:
        if sim_db is not None:
            sim_db.close()
        builder.close()
        current = getattr(session.lead, "context_builder", None)
        if current is not builder and hasattr(current, "close"):
            current.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # 确定数据库路径（需在 setup_logging 之前，以便确定日志文件位置）
    if args.db:
        db_path = Path(args.db)
    else:
        yaml_dir = Path(args.config).resolve().parent if args.config else Path.cwd()
        db_path = yaml_dir / "output" / "simulation.db"

    # 确定日志文件路径：--log-file 优先，否则默认放在数据库同目录
    if args.log_file:
        log_file: Path | None = Path(args.log_file)
    else:
        log_file = db_path.parent / "run.log"

    _setup_logging(args.log, log_file=log_file, conversation=args.conversation,
                   console_level=args.console_log, file_level=args.file_log)
    log = logging.getLogger(__name__)
    if args.conversation and log_file is not None:
        print(f"详细日志：{log_file.resolve()}", flush=True)

    # ── Agent 模式 + 裸 Aspen 文件（尚无配置 YAML）：走 Lead Agent 只读接入 ───
    # 必须在 load_optimize_config 之前分支：该函数假定输入是可解析的优化
    # 配置 YAML 加载器不接受 .bkp/.apw；裸模型由 Lead Agent 接入。
    # 识别要处理的场景（用户还没有配置，需要先接入/拟合/推荐边界）。
    config_path = Path(args.config) if args.config else Path.cwd()
    # Unified conversation is explicitly read-only.  It is available with a
    # YAML or Aspen path and never enters the optimization execution branch.
    if args.conversation:
        yaml_path = Path(args.yaml_path).resolve() if args.yaml_path else (config_path.resolve() if config_path.suffix.lower() in (".yaml", ".yml") else None)
        node_db_path = Path(args.node_db_path).resolve() if args.node_db_path else None
        return _run_unified_chat_loop(
            config_path=config_path.resolve(),
            yaml_path=yaml_path,
            db_path=db_path,
            node_db_path=node_db_path,
            initial_intent=args.intent,
            log=log,
            **({"new_project": args.new_project, "project_draft": args.project_draft} if args.new_project or args.project_draft else {}),
            **({"file_paths": args.file_paths} if args.file_paths else {}),
        )
    if config_path.suffix.lower() in (".bkp", ".apw") and not args.conversation:
        log.error("Aspen 文件必须通过唯一的 --conversation 入口进入 LeadAgent")
        return 1

    # 加载配置（agent 和普通模式都需要）
    from src.utils.file_io import load_optimize_config
    try:
        opt_cfg, sim_filepath, driver_kwargs = load_optimize_config(args.config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("配置加载失败：%s", exc)
        return 1

    opt_cfg.db_path = db_path
    if hasattr(opt_cfg, "agent_loop"):
        if args.checkpoint:
            opt_cfg.agent_checkpoint_path = str(Path(args.checkpoint).resolve())
        opt_cfg.agent_resume = args.resume or opt_cfg.agent_resume
        if opt_cfg.agent_loop.get("enabled"):
            from src.workflows.agent_entry import validate_agent_config
            try:
                validate_agent_config(opt_cfg)
            except (ValueError, TypeError) as exc:
                log.error("Agent 配置错误：%s", exc)
                return 1
    if args.resume and not getattr(opt_cfg, "agent_loop", {}).get("enabled"):
        log.error("--resume 仅支持 Agent 闭环")
        return 1
    if args.dry_run and not args.conversation:
        log.info("配置有效；--dry-run 不连接 Aspen、不调用 LLM、不写回 YAML。")
        return 0

    # ── Agent 协作模式：boundary_advisor 推荐边界 → 写回 YAML → 跑优化 ────────
    if getattr(opt_cfg, "agent_loop", {}).get("enabled") and not args.conversation:
        log.error("Agent 闭环必须通过唯一的 --conversation 入口进入 LeadAgent")
        return 1

    # The CLI uses the same preview/approval/execution boundary as conversation.
    from src.agents.actions import ActionGateway, ActionRequest
    from src.agents.actions.plans import build_optimization_plan, preflight_optimizer
    from src.database.simulation_db import SimulationDB
    try:
        plan, schema = build_optimization_plan(args.config)
        preflight_optimizer(opt_cfg)
        with SimulationDB(db_path) as execution_db:
            gateway = ActionGateway(schema, simulation_db=execution_db)
            action = ActionRequest("run_optimization", "model", reason="CLI configured optimization",
                                   payload={"optimization_plan": plan})
            validation = gateway.submit(action, refresh=False)
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))
            print(json.dumps(gateway.preview(action.audit_id).to_dict(), ensure_ascii=False, indent=2))
            try:
                confirmation = input("确认以上模型、算法、边界和总执行预算？输入 APPROVE 执行：")
            except (EOFError, KeyboardInterrupt):
                confirmation = ""
            if confirmation != "APPROVE":
                gateway.reject(action.audit_id)
                log.info("执行未批准；未打开 Aspen。")
                return 1
            gateway.approve(action.audit_id)
            output = gateway.execute(action.audit_id)
            log.info("执行结果：%s", json.dumps(output, ensure_ascii=False, default=str))
            return 0 if output.get("real_execution") else 1
    except Exception as exc:
        log.error("优化未完成：%s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
