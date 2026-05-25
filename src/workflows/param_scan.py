"""
param_scan.py — 参数扫描 workflow 层封装。

职责：
  1. 接受参数扫描配置（扫描变量的值列表）
  2. 生成参数组合（全因子网格或逐点对应）
  3. 依次调用 run_case() 执行每个参数组合
  4. 收集并返回 ScanResult（含所有 ProcessCase 和汇总统计）

层级关系
---------
param_scan()（本文件）
  └── run_case()（workflows/run_case.py）
        ├── SimulationRunner.run_case()     → SimulationResult
        ├── TreeExporter                    → block/stream 原始记录
        ├── _extract_blocks/streams()       → BlockResult / StreamResult
        └── _compute_objectives/constraints → ObjectiveValue / ConstraintValue

扫描模式
---------
"grid"（默认）：全因子笛卡尔积。
    scan_vars = {
        path_A: [1.0, 2.0, 3.0],   # 3 个值
        path_B: [0.3, 0.6, 0.9],   # 3 个值
    }
    → 3 × 3 = 9 个工况

"zip"：逐点对应，各变量值列表长度必须相同。
    scan_vars = {
        path_A: [1.0, 2.0, 3.0],
        path_B: [0.3, 0.6, 0.9],
    }
    → 3 个工况：(1.0, 0.3), (2.0, 0.6), (3.0, 0.9)

典型用法
---------
    from src.aspen_driver.driver import AspenDriver
    from src.workflows.run_case import RunCaseConfig
    from src.workflows.param_scan import ParamScanConfig, param_scan, linspace

    run_cfg = RunCaseConfig(
        output_paths=[...],
        objective_fns=[...],
        extract_blocks=["T0301"],
        extract_streams=["ADN"],
    )
    scan_cfg = ParamScanConfig(
        scan_vars={
            r"\\Data\\Blocks\\T0301\\Input\\BASIS_RR": linspace(1.0, 3.0, 5),
        },
        fixed_vars={
            r"\\Data\\Blocks\\T0301\\Input\\B:F": 0.6,
            r"\\Data\\Blocks\\T0301\\Input\\FEED_STAGE\\0318": 15,
        },
        run_config=run_cfg,
        tags=["sensitivity"],
    )
    with AspenDriver() as driver:
        driver.open("二级氢氰化工段.bkp")
        result = param_scan(driver, scan_cfg)

    print(f"成功率：{result.success_rate:.1%}")
    for case in result.successful_cases():
        print(case.summary())
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..aspen_driver.driver import AspenDriver
from ..aspen_driver.errors import AspenConnectionError
from ..models.process_case import CaseStatus, ProcessCase
from .run_case import RunCaseConfig, run_case

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 扫描配置
# ---------------------------------------------------------------------------

@dataclass
class ParamScanConfig:
    """
    param_scan() 的配置参数。

    Attributes
    ----------
    scan_vars:
        需要扫描的变量 {Aspen 树路径: 值列表}。
        "grid" 模式下各变量取笛卡尔积；"zip" 模式下逐点对应。
        值列表可用 linspace() / arange() 快速生成。
    fixed_vars:
        固定不变的设计变量 {Aspen 树路径: 值}，每次运行均使用相同值。
        与 scan_vars 合并后作为 run_case() 的 design_vars 传入。
        若 fixed_vars 与 scan_vars 存在相同路径，scan_vars 优先。
    run_config:
        每次单次运行的配置，见 RunCaseConfig。
        run_config.reinit=False 可复用上次收敛解作为初值（加速相邻点收敛）。
    mode:
        "grid"（默认）：全因子笛卡尔积，总工况数 = 各变量值列表长度之积。
        "zip"：逐点对应，各变量值列表长度必须相同，总工况数 = 列表长度。
    tags:
        应用到所有工况的标签列表，追加到每次 run_case() 的 tags 之后。
        自动添加 "param_scan" 标签，无需手动指定。
    on_case_complete:
        每次工况完成后的回调函数，签名为 (case, index, total) -> None。
        index 从 0 开始，total 为总工况数。
        可用于进度显示、实时保存结果或提前终止（回调内抛出异常会被捕获并记录）。
    max_cases:
        允许的最大工况数上限，默认 None（不限制）。
        若生成的参数组合数超过此值，param_scan() 在运行前抛出 ValueError。
        用于防止高维 DOE 意外生成过多工况（如 10 个变量各 10 个值 = 10^10 个组合）。
    """
    scan_vars: dict[str, list[Any]]
    fixed_vars: dict[str, Any] = field(default_factory=dict)
    run_config: RunCaseConfig = field(default_factory=RunCaseConfig)
    mode: Literal["grid", "zip"] = "grid"
    tags: list[str] = field(default_factory=list)
    on_case_complete: Callable[[ProcessCase, int, int], None] | None = None
    max_cases: int | None = None


# ---------------------------------------------------------------------------
# 扫描结果
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """
    param_scan() 的返回值，包含所有工况结果和汇总统计。

    Attributes
    ----------
    cases:
        所有工况的 ProcessCase 列表，顺序与参数组合顺序一致。
    scan_vars:
        本次扫描的变量路径和值列表（来自 ParamScanConfig.scan_vars）。
    fixed_vars:
        本次扫描的固定变量（来自 ParamScanConfig.fixed_vars）。
    n_total:
        总工况数。
    n_success:
        仿真收敛且目标函数可用的工况数（case.success=True）。
    n_sim_failed:
        仿真失败的工况数（status=SIM_FAILED）。
    n_objective_error:
        仿真收敛但目标函数计算失败的工况数（status=OBJECTIVE_ERROR）。
    n_simulation_valid:
        仿真本身收敛的工况数（含 SUCCESS/WARNINGS/INFEASIBLE/OBJECTIVE_ERROR/CONSTRAINT_ERROR）。
        无目标函数配置时，此值比 n_success 更能反映仿真质量。
    n_warnings:
        仿真收敛但有警告的工况数（status=WARNINGS）。
    n_infeasible:
        仿真收敛但约束违反的工况数（status=INFEASIBLE）。
    n_constraint_error:
        仿真收敛但约束计算失败的工况数（status=CONSTRAINT_ERROR）。
    elapsed:
        总耗时（秒），含所有工况的仿真时间和提取时间。
    """
    cases: list[ProcessCase]
    scan_vars: dict[str, list[Any]]
    fixed_vars: dict[str, Any]
    n_total: int
    n_success: int
    n_sim_failed: int
    n_objective_error: int
    n_simulation_valid: int
    n_warnings: int
    n_infeasible: int
    n_constraint_error: int
    elapsed: float

    @property
    def success_rate(self) -> float:
        """成功率（0.0 ~ 1.0）。n_total=0 时返回 0.0。"""
        return self.n_success / self.n_total if self.n_total > 0 else 0.0

    def successful_cases(self) -> list[ProcessCase]:
        """返回所有 success=True 的工况（仿真收敛且目标函数可用）。"""
        return [c for c in self.cases if c.success]

    def simulation_valid_cases(self) -> list[ProcessCase]:
        """返回所有仿真收敛的工况（含 INFEASIBLE / OBJECTIVE_ERROR）。"""
        return [c for c in self.cases if c.simulation_valid]

    def to_summary(self) -> dict[str, Any]:
        """返回汇总字典，供日志和数据库记录。"""
        return {
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_simulation_valid": self.n_simulation_valid,
            "n_sim_failed": self.n_sim_failed,
            "n_warnings": self.n_warnings,
            "n_infeasible": self.n_infeasible,
            "n_objective_error": self.n_objective_error,
            "n_constraint_error": self.n_constraint_error,
            "success_rate": self.success_rate,
            "elapsed": self.elapsed,
            "scan_vars": {k: len(v) for k, v in self.scan_vars.items()},
            "fixed_vars": list(self.fixed_vars.keys()),
        }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def param_scan(
    driver: AspenDriver,
    config: ParamScanConfig,
    start_iteration: int = 0,
) -> ScanResult:
    """
    执行参数扫描，返回所有工况的 ScanResult。

    单个工况的意外异常会被隔离为 SIM_FAILED，扫描继续执行后续工况。
    若 driver 连接已断开（AspenConnectionError），则终止扫描并返回已完成的结果。

    Parameters
    ----------
    driver:
        已连接并打开仿真文件的 AspenDriver 实例。
    config:
        扫描配置，见 ParamScanConfig。
    start_iteration:
        起始迭代编号，默认 0。
        第 i 个工况的 iteration = start_iteration + i（i 从 0 开始）。
        在优化循环中调用参数扫描时，可传入当前迭代编号以保持连续性。

    Returns
    -------
    ScanResult
    """
    _validate_config(config)

    combos = _generate_combinations(config)
    n_total = len(combos)

    _log.info(
        "参数扫描开始：共 %d 个工况，模式=%s，扫描变量=%s。",
        n_total,
        config.mode,
        list(config.scan_vars.keys()),
    )

    cases: list[ProcessCase] = []
    t0 = time.monotonic()
    driver_dead = False

    for idx, scan_point in enumerate(combos):
        design_vars = {**config.fixed_vars, **scan_point}
        iteration = start_iteration + idx
        tags = list(config.tags) + ["param_scan"]

        _log.info(
            "参数扫描 [%d/%d]：%s",
            idx + 1,
            n_total,
            {k.split("\\")[-1]: v for k, v in scan_point.items()},
        )

        try:
            case = run_case(
                driver=driver,
                design_vars=design_vars,
                config=config.run_config,
                iteration=iteration,
                tags=tags,
            )
        except AspenConnectionError as exc:
            # driver 连接已断开，不可恢复，终止扫描
            _log.error(
                "参数扫描 [%d/%d]：driver 连接断开，终止扫描。原因：%s",
                idx + 1, n_total, exc,
            )
            driver_dead = True
            # 为当前工况构造失败记录，保留已完成结果
            case = ProcessCase(
                iteration=iteration,
                status=CaseStatus.SIM_FAILED,
                design_vars=design_vars,
                tags=tags,
                notes=f"driver 连接断开，扫描终止：{exc}",
            )
            cases.append(case)
            break
        except Exception as exc:
            # 意外异常（代码 bug、COM 偶发错误等），隔离为 SIM_FAILED，继续扫描
            _log.warning(
                "参数扫描 [%d/%d]：run_case() 意外异常（已隔离）：%s",
                idx + 1, n_total, exc,
            )
            case = ProcessCase(
                iteration=iteration,
                status=CaseStatus.SIM_FAILED,
                design_vars=design_vars,
                tags=tags,
                notes=f"run_case() 意外异常：{exc}",
            )

        cases.append(case)

        if config.on_case_complete is not None:
            try:
                config.on_case_complete(case, idx, n_total)
            except Exception as cb_exc:
                _log.warning("on_case_complete 回调异常（已忽略）：%s", cb_exc)

        _log.info(
            "  → status=%s, success=%s, run_time=%.1fs",
            case.status.value,
            case.success,
            case.run_time,
        )

    elapsed = time.monotonic() - t0

    n_success          = sum(1 for c in cases if c.success)
    n_sim_failed       = sum(1 for c in cases if c.status == CaseStatus.SIM_FAILED)
    n_objective_error  = sum(1 for c in cases if c.status == CaseStatus.OBJECTIVE_ERROR)
    n_simulation_valid = sum(1 for c in cases if c.simulation_valid)
    n_warnings         = sum(1 for c in cases if c.status == CaseStatus.WARNINGS)
    n_infeasible       = sum(1 for c in cases if c.status == CaseStatus.INFEASIBLE)
    n_constraint_error = sum(1 for c in cases if c.status == CaseStatus.CONSTRAINT_ERROR)

    if driver_dead:
        _log.warning(
            "参数扫描因 driver 断开提前终止：已完成 %d/%d 个工况，%d 成功，耗时 %.1fs。",
            len(cases), n_total, n_success, elapsed,
        )
    else:
        _log.info(
            "参数扫描完成：%d/%d 成功（仿真有效 %d），%d 仿真失败，%d 目标错误，总耗时 %.1fs。",
            n_success, n_total, n_simulation_valid, n_sim_failed, n_objective_error, elapsed,
        )

    return ScanResult(
        cases=cases,
        scan_vars=config.scan_vars,
        fixed_vars=config.fixed_vars,
        n_total=n_total,
        n_success=n_success,
        n_sim_failed=n_sim_failed,
        n_objective_error=n_objective_error,
        n_simulation_valid=n_simulation_valid,
        n_warnings=n_warnings,
        n_infeasible=n_infeasible,
        n_constraint_error=n_constraint_error,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

def _validate_config(config: ParamScanConfig) -> None:
    """
    在扫描开始前校验 ParamScanConfig，发现问题立即抛出 ValueError。

    检查项：
    - scan_vars 非空
    - scan_vars 中每个变量的值列表非空
    - mode 为合法值（"grid" 或 "zip"）
    - fixed_vars 与 scan_vars 无路径冲突（大小写不敏感）
    - 生成的工况数不超过 max_cases（若已设置）
    """
    if not config.scan_vars:
        raise ValueError("scan_vars 不能为空，至少需要一个扫描变量。")

    empty_paths = [p for p, v in config.scan_vars.items() if not v]
    if empty_paths:
        raise ValueError(
            f"scan_vars 中以下路径的值列表为空，无法生成参数组合：{empty_paths}。"
        )

    if config.mode not in ("grid", "zip"):
        raise ValueError(
            f"mode 必须为 'grid' 或 'zip'，收到：{config.mode!r}。"
        )

    # 路径冲突检测：大小写不敏感比较
    scan_upper  = {p.upper() for p in config.scan_vars}
    fixed_upper = {p.upper(): p for p in config.fixed_vars}
    conflicts = [
        (fixed_upper[u], next(p for p in config.scan_vars if p.upper() == u))
        for u in scan_upper & set(fixed_upper)
    ]
    if conflicts:
        detail = "; ".join(f"fixed={f!r} vs scan={s!r}" for f, s in conflicts)
        raise ValueError(
            f"fixed_vars 与 scan_vars 存在路径冲突（大小写不敏感）：{detail}。"
            "请从 fixed_vars 中移除冲突路径，或统一路径大小写。"
        )

    if config.max_cases is not None:
        # 预估工况数（不实际生成组合，避免内存问题）
        if config.mode == "grid":
            n_estimated = 1
            for v in config.scan_vars.values():
                n_estimated *= len(v)
        else:
            n_estimated = len(next(iter(config.scan_vars.values())))
        if n_estimated > config.max_cases:
            raise ValueError(
                f"预估工况数 {n_estimated} 超过 max_cases={config.max_cases}。"
                "请减少扫描变量的取值数量，或提高 max_cases 上限。"
            )


# ---------------------------------------------------------------------------
# 参数组合生成
# ---------------------------------------------------------------------------

def _generate_combinations(config: ParamScanConfig) -> list[dict[str, Any]]:
    """
    根据 mode 生成参数组合列表。调用前须先通过 _validate_config()。

    "grid"：全因子笛卡尔积，顺序为最后一个变量变化最快（C 顺序）。
    "zip"：逐点对应，各变量值列表长度必须相同。
    """
    paths = list(config.scan_vars.keys())
    value_lists = [config.scan_vars[p] for p in paths]

    if config.mode == "zip":
        lengths = [len(v) for v in value_lists]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"zip 模式要求所有 scan_vars 的值列表长度相同，"
                f"当前长度：{dict(zip(paths, lengths))}。"
            )
        return [dict(zip(paths, combo)) for combo in zip(*value_lists)]

    # grid 模式：笛卡尔积
    return [
        dict(zip(paths, combo))
        for combo in itertools.product(*value_lists)
    ]


# ---------------------------------------------------------------------------
# 便捷工具函数
# ---------------------------------------------------------------------------

def linspace(start: float, stop: float, n: int) -> list[float]:
    """
    生成 n 个均匀分布的浮点数（含两端点），类似 numpy.linspace。

    用于快速构造 scan_vars 的值列表：
        scan_vars = {path: linspace(1.0, 3.0, 5)}
        # → [1.0, 1.5, 2.0, 2.5, 3.0]

    Parameters
    ----------
    start:
        起始值（含）。
    stop:
        终止值（含）。
    n:
        点数，必须 >= 2。
    """
    if n < 2:
        raise ValueError(f"linspace 的 n 必须 >= 2，收到：{n}。")
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def arange(start: float, stop: float, step: float) -> list[float]:
    """
    生成从 start 到 stop（含）步长为 step 的浮点数列表。

    与 numpy.arange 不同，此函数包含 stop 端点（若恰好落在网格上）。
    用于快速构造 scan_vars 的值列表：
        scan_vars = {path: arange(1.0, 3.0, 0.5)}
        # → [1.0, 1.5, 2.0, 2.5, 3.0]

    Parameters
    ----------
    start:
        起始值（含）。
    stop:
        终止值（含，若恰好落在网格上）。
    step:
        步长，必须为正数。
    """
    if step <= 0:
        raise ValueError(f"arange 的 step 必须为正数，收到：{step}。")
    result: list[float] = []
    i = 0
    while True:
        val = start + i * step
        if val > stop + step * 1e-9:
            break
        result.append(val)
        i += 1
    return result
