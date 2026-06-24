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
    --agent         启用 AI Agent 协作模式：LLM 自动扫描 Aspen 变量、生成配置草案，
                    终端交互确认/修改边界，再运行 Pareto 优化（忽略 YAML 配置）
    --intent TEXT   优化意图自然语言描述（仅 --agent 模式使用），
                    如 "最小化 TAC 和 CO2 排放，约束产品纯度 > 99%"

示例
----
    # 单目标贝叶斯优化（直接读 YAML 配置）
    python -m src.main cases/demo_case/case_config.yaml

    # 多目标 Pareto 优化（optimizer.type: pareto_bayesian）
    python -m src.main cases/demo_case/pareto_tac_emissions_config.yaml

    # AI Agent 协作模式（LLM 扫描变量，终端 HITL 交互）
    python -m src.main cases/demo_case_2/pareto_config_epsd_aligned.yaml --agent
    python -m src.main cases/demo_case_2/pareto_config_epsd_aligned.yaml --agent \\
        --intent "最小化总年费用和碳排放，约束甘油纯度 > 99%"

    python -m src.main cases/demo_case/case_config.yaml --db output/run1.db --log DEBUG
    python -m src.main cases/demo_case/case_config.yaml --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(level: str, log_file: Path | None = None) -> None:
    from src.utils.logger import setup_logging
    setup_logging(level, log_file=log_file)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="PAO — Aspen Plus 贝叶斯优化框架",
    )
    parser.add_argument("config", help="case_config.yaml 路径（--agent 模式下作为 Aspen 文件目录提示）")
    parser.add_argument("--db",       default=None, help="结果数据库路径（.db）")
    parser.add_argument("--log",      default="INFO", help="日志级别（默认 INFO）")
    parser.add_argument("--log-file", default=None, metavar="PATH",
                        help="日志文件路径（默认：与数据库同目录的 run.log）")
    parser.add_argument("--dry-run",  action="store_true", help="只打印配置摘要，不运行仿真")
    parser.add_argument("--agent",    action="store_true",
                        help="启用 AI Agent 协作模式（LLM 扫描变量 + 终端 HITL 交互）")
    parser.add_argument("--intent",   default="",
                        help="优化意图自然语言描述（仅 --agent 模式使用）")
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
    from src.agents.boundary_advisor import VarMeta

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


def _run_agent_mode(
    yaml_path: Path,
    intent_text: str,
    log: logging.Logger,
    opt_cfg,
    sim_filepath: Path,
    driver_kwargs: dict,
    db_path: Path,
) -> int:
    """
    终端版 Agent 协作模式。

    流程
    ----
    1. 从 YAML 读取已有设计变量（16 个，不扫描 Aspen 全量节点）
    2. 调用 boundary_advisor（LLM）为每个变量推荐搜索边界
    3. 终端展示推荐报告，询问是否写回 YAML
    4. 写回 YAML 后自动运行 Pareto 优化
    5. 打印优化结果摘要

    完全复用 run_boundary_advisor.py 的逻辑，加上第 4/5 步的优化触发。
    """
    from src.agents.boundary_advisor import recommend_boundaries_agent, format_boundary_report
    from src.agents.boundary_advisor.tools import plan_yaml_edits, apply_yaml_edits

    # ── 1. 读取变量 ──────────────────────────────────────────────────────────
    _sep()
    print("🤖  PAO Agent 模式")
    print(f"    配置文件：{yaml_path}")
    print(f"    优化意图：{intent_text or '（未指定，LLM 自动判断工艺类型）'}")
    _sep()

    variables, var_types, integer_names = _load_var_metas(yaml_path)
    if not variables:
        log.error("未从 design_variables 读到任何变量，请检查 YAML 配置。")
        return 1
    print(f"\n已从 YAML 读取 {len(variables)} 个设计变量，正在调用 boundary_advisor 推荐搜索边界…\n")

    # ── 2. 调用 boundary_advisor ─────────────────────────────────────────────
    try:
        report = recommend_boundaries_agent(variables, context=intent_text)
    except Exception as exc:
        log.error("boundary_advisor 调用失败：%s", exc, exc_info=True)
        return 1

    # ── 3. 展示报告 ──────────────────────────────────────────────────────────
    _sep()
    print("【boundary_advisor 推荐报告】")
    _sep()
    print(format_boundary_report(report))

    # 规划写回改动
    yaml_text = yaml_path.read_text(encoding="utf-8")
    bounds_by_name = {r.name: (r.lower, r.upper) for r in report.recommendations}
    edits = plan_yaml_edits(yaml_text, bounds_by_name, var_types, integer_names)

    _sep()
    print("将要写回 YAML 的边界改动：")
    _sep()
    any_change = False
    for e in edits:
        loc_lo = f"L{e.line_lo}" if e.line_lo else "未找到"
        loc_hi = f"L{e.line_hi}" if e.line_hi else "未找到"
        print(f"  {e.name}")
        print(f"    {e.field_lo}: {e.old_lo} -> {e.new_lo}  ({loc_lo})")
        print(f"    {e.field_hi}: {e.old_hi} -> {e.new_hi}  ({loc_hi})")
        if e.line_lo or e.line_hi:
            any_change = True

    if not any_change:
        print("\n⚠ 未找到任何可写回的边界行（YAML 中未定义 lower_bound/upper_bound 字段）。")
        print("  将使用 YAML 中的原有边界直接运行优化。")
    else:
        # ── 询问是否写回 ──────────────────────────────────────────────────────
        try:
            ans = input("\n确认将推荐边界写回 YAML 并启动优化？(y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans in ("y", "yes"):
            backup = yaml_path.with_suffix(yaml_path.suffix + ".bak")
            backup.write_text(yaml_text, encoding="utf-8")
            new_text, skipped = apply_yaml_edits(yaml_text, edits)
            yaml_path.write_text(new_text, encoding="utf-8")
            print(f"\n✅ 边界已写回：{yaml_path.name}")
            print(f"   原文件已备份：{backup.name}")
            if skipped:
                print(f"   跳过（未找到对应行）：{skipped}")

            # 重新加载配置（边界已更新），并补回 db_path
            from src.utils.file_io import load_optimize_config
            try:
                opt_cfg, sim_filepath, driver_kwargs = load_optimize_config(str(yaml_path))
                opt_cfg.db_path = db_path
            except Exception as exc:
                log.error("重新加载配置失败：%s", exc)
                return 1
        else:
            print("  已取消写回，使用 YAML 原有边界直接运行优化。")

    # ── 4. 询问是否启动优化 ──────────────────────────────────────────────────
    try:
        ans2 = input("\n是否立即启动 Pareto 优化？(Y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans2 = "y"

    if ans2 in ("n", "no"):
        print("已跳过优化，程序退出。")
        return 0

    # ── 5. 运行 Pareto 优化 ──────────────────────────────────────────────────
    _sep()
    print("🚀 开始运行 Pareto 贝叶斯优化…")
    _sep()

    from src.aspen_driver.driver import AspenDriver
    from src.workflows.optimize_pareto_case import optimize_pareto_case

    log.info("正在连接 Aspen Plus 并打开仿真文件……")
    try:
        with AspenDriver(**driver_kwargs) as driver:
            driver.open(sim_filepath)
            log.info("仿真文件已打开，开始多目标贝叶斯优化……")
            result = optimize_pareto_case(driver, opt_cfg)
    except Exception as exc:
        log.error("优化运行失败：%s", exc, exc_info=True)
        return 1

    # ── 6. 打印结果摘要 ──────────────────────────────────────────────────────
    _print_pareto_summary(log, result, opt_cfg.db_path)

    # ── 7. 生成可视化图表 ─────────────────────────────────────────────────────
    try:
        from src.reporting.plot_pareto import generate_pareto_report
        generate_pareto_report(
            opt_cfg.db_path,
            out_dir=opt_cfg.db_path.parent,
            session_id=getattr(result, "session_id", None),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("可视化报告生成失败（不影响结果）：%s", exc)

    return 0


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

    # 加载配置（agent 和普通模式都需要）
    from src.utils.file_io import load_optimize_config
    try:
        opt_cfg, sim_filepath, driver_kwargs = load_optimize_config(args.config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("配置加载失败：%s", exc)
        return 1

    opt_cfg.db_path = db_path

    # ── Agent 协作模式：boundary_advisor 推荐边界 → 写回 YAML → 跑优化 ────────
    if args.agent:
        return _run_agent_mode(
            yaml_path=Path(args.config).resolve(),
            intent_text=args.intent,
            log=log,
            opt_cfg=opt_cfg,
            sim_filepath=sim_filepath,
            driver_kwargs=driver_kwargs,
            db_path=db_path,
        )

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
