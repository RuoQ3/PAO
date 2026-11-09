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
import logging
import sys
from pathlib import Path

# Windows 控制台默认 GBK 编码，Agent 模式的报告文本含 emoji/"±"等字符会
# 导致 UnicodeEncodeError；强制 stdout/stderr 走 UTF-8，不影响非 Windows 环境
# （reconfigure 是幂等操作）。与 scripts/run_kinetics_advisor.py 的处理一致。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    # PowerShell may expose the console input stream with the legacy system
    # code page.  Agent prompts are sent to the LLM as UTF-8 JSON, so decode
    # interactive turns consistently before classification.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def _setup_logging(level: str, log_file: Path | None = None) -> None:
    from src.utils.logger import setup_logging
    setup_logging(level, log_file=log_file)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="PAO — Aspen Plus 贝叶斯优化框架",
    )
    parser.add_argument("config", help="case_config.yaml 路径（--conversation 模式下作为 Aspen 文件目录提示）")
    parser.add_argument("--db",       default=None, help="结果数据库路径（.db）")
    parser.add_argument("--log",      default="INFO", help="日志级别（默认 INFO）")
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
    return parser.parse_args(argv)


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


def _run_unified_chat_loop(
    *, config_path: Path, yaml_path: Path | None, db_path: Path | None,
    node_db_path: Path | None, initial_intent: str, log: logging.Logger,
) -> int:
    """Run the unified Lead Agent conversation entry point."""
    from src.agents.lead import ConversationSession, LeadAgent
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
    context = builder.build()
    params = {}
    for item in list((getattr(context, "variables", {}) or {}).values()) + [
        p for values in (getattr(context, "equipment_parameters", {}) or {}).values()
        for p in (values or []) if isinstance(p, dict)
    ]:
        path = str(item.get("aspen_path") or item.get("path") or "")
        if "\\input\\" not in path.casefold():
            continue
        params[path] = ParameterSpec(path=path, value_type="int" if str(item.get("type", "")).casefold() in {"int", "integer"} else "float",
                                     unit=item.get("unit"), minimum=item.get("lower", item.get("lower_bound")),
                                     maximum=item.get("upper", item.get("upper_bound")), integer=bool(item.get("integer")))
    schema = ModelSchema(ModelIdentity(str(getattr(context, "model_id", None) or "model")), parameters=params,
                         current_values=getattr(context, "current_values", {}) or {},
                         permissions={"inspect", "run_simulation"}, targets=("model", ""))
    sim_db = SimulationDB(db_path)
    runner = AspenSandboxRunner(driver_factory=lambda _path: __import__("src.aspen_driver.driver", fromlist=["AspenDriver"]).AspenDriver(),
                                allowed_paths=set(params), require_path_whitelist=True)
    gateway = ActionGateway(schema, simulation_runner=runner, simulation_db=sim_db)
    session = ConversationSession(SessionState(), LeadAgent(
        policy=RuntimePolicy(allow_writes=False), context_builder=builder,
        action_gateway=gateway, use_llm=True))
    print("\nPAO Lead Agent conversation")
    print("输入问题，输入 exit 退出")

    def run_turn(text: str) -> bool:
        if text.strip().lower() in {"/exit", "exit", "quit"}:
            return True
        result = session.submit(text)
        print(result.text)
        if result.recommendations and result.approval_request is None:
            print("建议：")
            for recommendation in result.recommendations:
                print("-", recommendation)
        return False

    try:
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
    finally:
        sim_db.close()
        builder.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # 确定数据库路径（需在 setup_logging 之前，以便确定日志文件位置）
    if args.db:
        db_path = Path(args.db)
    else:
        yaml_dir = Path(args.config).resolve().parent
        db_path = yaml_dir / "output" / "simulation.db"

    # 确定日志文件路径：--log-file 优先，否则默认放在数据库同目录
    if args.log_file:
        log_file: Path | None = Path(args.log_file)
    else:
        log_file = db_path.parent / "run.log"

    _setup_logging(args.log, log_file=log_file)
    log = logging.getLogger(__name__)

    # ── Agent 模式 + 裸 Aspen 文件（尚无配置 YAML）：走 Lead Agent 只读接入 ───
    # 必须在 load_optimize_config 之前分支：该函数假定输入是可解析的优化
    # 配置 YAML 加载器不接受 .bkp/.apw；裸模型由 Lead Agent 接入。
    # 识别要处理的场景（用户还没有配置，需要先接入/拟合/推荐边界）。
    config_path = Path(args.config)
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

    # 判断优化类型
    from src.workflows.optimize_pareto_case import ParetoOptimizeCaseConfig
    is_pareto = isinstance(opt_cfg, ParetoOptimizeCaseConfig)

    # 打印配置摘要
    log.info("仿真文件：%s", sim_filepath)
    log.info("设计变量（%d 个）：", len(opt_cfg.param_bounds))
    for path, (lo, hi) in opt_cfg.param_bounds.items():
        log.info("  %s  [%.4g, %.4g]", path.split("\\")[-1], lo, hi)
    if opt_cfg.fixed_vars:
        log.info("固定变量（%d 个）：%s", len(opt_cfg.fixed_vars),
                 list(opt_cfg.fixed_vars.keys()))

    if is_pareto:
        log.info(
            "优化模式：多目标 Pareto（%s），目标=%s，初始 DOE=%d，总迭代=%d，"
            "标量化=%s，采集函数=%s",
            "pareto_bayesian",
            opt_cfg.objective_names,
            opt_cfg.n_initial,
            opt_cfg.n_iterations,
            opt_cfg.scalarization,
            opt_cfg.acquisition,
        )
        log.info("Surrogate model: %s", opt_cfg.surrogate_model)
    else:
        log.info(
            "优化目标：%s（%s），初始 DOE=%d，总迭代=%d，采集函数=%s",
            opt_cfg.objective_name,
            "最小化" if opt_cfg.minimize else "最大化",
            opt_cfg.n_initial,
            opt_cfg.n_iterations,
            opt_cfg.acquisition,
        )
        log.info("Surrogate model: %s", opt_cfg.surrogate_model)
    log.info("结果数据库：%s", db_path)

    if args.dry_run:
        log.info("--dry-run 模式，跳过仿真。")
        return 0

    # 运行优化
    from src.aspen_driver.driver import AspenDriver

    log.info("正在连接 Aspen Plus 并打开仿真文件……")
    try:
        with AspenDriver(**driver_kwargs) as driver:
            driver.open(sim_filepath)
            if is_pareto:
                from src.workflows.optimize_pareto_case import optimize_pareto_case
                log.info("仿真文件已打开，开始多目标贝叶斯优化……")
                result = optimize_pareto_case(driver, opt_cfg)
            else:
                from src.workflows.optimize_case import optimize_case
                log.info("仿真文件已打开，开始贝叶斯优化……")
                result = optimize_case(driver, opt_cfg)
    except Exception as exc:
        log.error("优化运行失败：%s", exc, exc_info=True)
        return 1

    if is_pareto:
        _print_pareto_summary(log, result, db_path)
    else:
        _print_single_summary(log, result, db_path)

    # 自动生成可视化报告（仅多目标 Pareto 优化）
    if is_pareto:
        try:
            from src.reporting.plot_pareto import generate_pareto_report
            generate_pareto_report(
                db_path,
                out_dir=db_path.parent,
                session_id=getattr(result, "session_id", None),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("可视化报告生成失败（不影响结果）：%s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
