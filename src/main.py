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

示例
----
    # 单目标贝叶斯优化
    python -m src.main cases/demo_case/case_config.yaml

    # 多目标 Pareto 优化（optimizer.type: pareto_bayesian）
    python -m src.main cases/demo_case/pareto_tac_emissions_config.yaml

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
    parser.add_argument("config", help="case_config.yaml 路径")
    parser.add_argument("--db",       default=None, help="结果数据库路径（.db）")
    parser.add_argument("--log",      default="INFO", help="日志级别（默认 INFO）")
    parser.add_argument("--log-file", default=None, metavar="PATH",
                        help="日志文件路径（默认：与数据库同目录的 run.log）")
    parser.add_argument("--dry-run",  action="store_true", help="只打印配置摘要，不运行仿真")
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

    # 加载配置
    from src.utils.file_io import load_optimize_config
    try:
        opt_cfg, sim_filepath, driver_kwargs = load_optimize_config(args.config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("配置加载失败：%s", exc)
        return 1

    opt_cfg.db_path = db_path

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
    else:
        log.info(
            "优化目标：%s（%s），初始 DOE=%d，总迭代=%d，采集函数=%s",
            opt_cfg.objective_name,
            "最小化" if opt_cfg.minimize else "最大化",
            opt_cfg.n_initial,
            opt_cfg.n_iterations,
            opt_cfg.acquisition,
        )
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
