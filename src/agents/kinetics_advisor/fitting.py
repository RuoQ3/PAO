"""
fitting.py — Arrhenius 动力学参数确定性数值拟合。

职责（不含任何 LLM 调用，纯数值计算，便于单测）：
  - fit_arrhenius：从 (温度, 速率常数) 观测点拟合活化能 Ea 和频率因子 k0。

设计原则
--------
  - 数值方法固定两步：
    1. 线性化：对 (1/T, ln k) 做最小二乘线性回归，得到 Ea/k0 初值和 R²。
       这一步永远执行，不依赖 scipy，只用标准库（math）+ 简单矩阵运算。
    2. 非线性精修：用步骤1的初值喂给 scipy.optimize.curve_fit，对原始
       k = k0*exp(-Ea/RT) 做非线性最小二乘，得到更准确的参数和标准误。
       scipy 不可用或拟合不收敛时，静默回退到纯线性化结果（method="linear"），
       不抛异常给上层——拟合失败是正常业务结果，不是程序错误。
  - LLM 绝不参与任何数值计算；本模块产出的 KineticsFitResult 是唯一的
    数值真相来源，agent.py 的 LLM 层只负责解读这个结果是否物理合理。
  - 所有输入校验（数据点数、正数检查）在 plan.py 阶段完成；本模块假定
    调用方已保证输入合法，但仍做防御性检查，绝不产出 NaN/Inf 而不报警。

物理背景
--------
Arrhenius 方程：k = k0 * exp(-Ea / (R*T))，R = 8.314 J/(mol*K)。
线性化：ln(k) = ln(k0) - (Ea/R) * (1/T)，即 y = a + b*x，
其中 x = 1/T，y = ln(k)，b = -Ea/R，a = ln(k0)。
"""
from __future__ import annotations

import logging
import math

from src.models.kinetics import KineticsDataPoint, KineticsFitResult

_log = logging.getLogger(__name__)

_R = 8.314462618  # J/(mol*K)，气体常数

# 化学反应活化能的典型物理范围（kJ/mol），超出此范围需警告用户核查数据/单位。
_EA_TYPICAL_LO_KJ = 40.0
_EA_TYPICAL_HI_KJ = 400.0

try:
    from scipy.optimize import curve_fit as _curve_fit
    _HAS_SCIPY = True
except ImportError:
    _curve_fit = None  # type: ignore[assignment]
    _HAS_SCIPY = False
    _log.debug("scipy 不可用，fit_arrhenius 将只执行线性化拟合，不做非线性精修。")


# ---------------------------------------------------------------------------
# 线性化拟合（无第三方依赖）
# ---------------------------------------------------------------------------

def _linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """最小二乘线性回归 y = a + b*x，返回 (a, b, r_squared)。

    纯标准库实现，不依赖 numpy/scipy，保证线性化这一步在任何环境下都能跑。
    """
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx == 0.0:
        raise ValueError("所有 1/T 值相同（数据点温度全部相同），无法做线性回归")
    b = sxy / sxx
    a = mean_y - b * mean_x

    # R²
    y_pred = [a + b * xi for xi in x]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return a, b, r_squared


def _arrhenius_k(temperature, ea: float, k0: float):
    """k = k0 * exp(-Ea / (R*T))

    curve_fit 会以 numpy 数组调用本函数（temperature 是数组），
    必须用 numpy.exp 而非 math.exp，否则对数组求值时抛
    "only length-1 arrays can be converted to Python scalars"。
    单点调用（temperature 为 float）时 numpy.exp 同样正确工作。
    """
    import numpy as _np
    return k0 * _np.exp(-ea / (_R * temperature))


def _fit_quality_from_r2(r_squared: float | None) -> str:
    if r_squared is None:
        return "poor"
    if r_squared >= 0.95:
        return "good"
    if r_squared >= 0.80:
        return "marginal"
    return "poor"


# ---------------------------------------------------------------------------
# 主拟合函数
# ---------------------------------------------------------------------------

def fit_arrhenius(data_points: list[KineticsDataPoint]) -> KineticsFitResult:
    """
    从 (温度, 速率常数) 观测点拟合 Arrhenius 参数 (Ea, k0)。

    调用方须保证 data_points 已通过 plan.py 的最低数据量校验
    （>= 3 个点，非全部同温，T/k 均为正数）；本函数仍做防御性检查，
    校验失败时返回 fit_quality="poor" 且 ea/k0=None，不抛异常。

    Parameters
    ----------
    data_points:
        实验观测点列表，每项含 temperature（K）和 rate_constant。

    Returns
    -------
    KineticsFitResult
        method="nonlinear" 表示线性化+curve_fit 均成功；
        method="linear" 表示仅线性化成功（scipy 不可用或非线性拟合未收敛）。
    """
    warnings: list[str] = []

    # ── 防御性校验（plan.py 应已拦截，这里是最后一道防线）──────────────────
    n = len(data_points)
    if n < 3:
        return KineticsFitResult(
            ea=None, ea_stderr=None, k0=None, k0_stderr=None, r_squared=None,
            n_points=n, fit_quality="poor",
            warnings=[f"数据点数 {n} < 3，Arrhenius 两参数拟合欠定，拒绝拟合"],
        )

    temps = [p.temperature for p in data_points]
    ks = [p.rate_constant for p in data_points]

    if any(t <= 0 for t in temps):
        return KineticsFitResult(
            ea=None, ea_stderr=None, k0=None, k0_stderr=None, r_squared=None,
            n_points=n, fit_quality="poor",
            warnings=["存在非正温度值（K 必须为正），拒绝拟合"],
        )
    if any(k <= 0 for k in ks):
        return KineticsFitResult(
            ea=None, ea_stderr=None, k0=None, k0_stderr=None, r_squared=None,
            n_points=n, fit_quality="poor",
            warnings=["存在非正速率常数（k 必须为正，取对数线性化要求），拒绝拟合"],
        )

    # ── 步骤1：线性化 ln(k) = ln(k0) - (Ea/R)*(1/T) ─────────────────────────
    x = [1.0 / t for t in temps]
    y = [math.log(k) for k in ks]
    try:
        a, b, r_squared = _linear_regression(x, y)
    except ValueError as exc:
        return KineticsFitResult(
            ea=None, ea_stderr=None, k0=None, k0_stderr=None, r_squared=None,
            n_points=n, fit_quality="poor", warnings=[str(exc)],
        )

    ea_linear = -b * _R
    k0_linear = math.exp(a)

    if ea_linear <= 0:
        warnings.append(
            f"线性化拟合得到 Ea={ea_linear:.3g} J/mol（非正值），"
            "速率常数随温度升高反而下降，请核查数据是否有误（如温度/速率对应关系颠倒）"
        )

    ea_kj = ea_linear / 1000.0
    if ea_linear > 0 and not (_EA_TYPICAL_LO_KJ <= ea_kj <= _EA_TYPICAL_HI_KJ):
        warnings.append(
            f"拟合得到 Ea={ea_kj:.1f} kJ/mol，超出化学反应活化能典型范围 "
            f"[{_EA_TYPICAL_LO_KJ:.0f}, {_EA_TYPICAL_HI_KJ:.0f}] kJ/mol，"
            "请核查数据单位（是否误用 cal 而非 J，或速率常数单位不一致）"
        )

    t_span = max(temps) - min(temps)
    t_mean = sum(temps) / n
    if t_mean > 0 and t_span / t_mean < 0.05:
        warnings.append(
            f"数据温度范围较窄（{min(temps):.1f}~{max(temps):.1f} K），"
            "Ea 外推到该范围之外时可靠性较低，建议补充更宽温度范围的实验点"
        )

    result = KineticsFitResult(
        ea=ea_linear, ea_stderr=None, k0=k0_linear, k0_stderr=None,
        r_squared=r_squared, n_points=n,
        fit_quality=_fit_quality_from_r2(r_squared),
        warnings=list(warnings), method="linear",
    )

    if ea_linear <= 0:
        # 非线性精修的初值(k0*exp(-Ea/RT))在 Ea<=0 时数值上无意义，不再尝试。
        return result

    # ── 步骤2：非线性精修（scipy 可用且线性初值合理时才尝试）──────────────
    if not _HAS_SCIPY:
        result.warnings.append("scipy 不可用，仅返回线性化拟合结果，未做非线性精修")
        return result

    try:
        popt, pcov = _curve_fit(
            _arrhenius_k, temps, ks, p0=[ea_linear, k0_linear],
            maxfev=5000,
        )
        ea_nl, k0_nl = float(popt[0]), float(popt[1])
        stderr = [float(v) ** 0.5 if v >= 0 else None for v in
                  (pcov[0][0], pcov[1][1])]
        ea_stderr, k0_stderr = stderr[0], stderr[1]

        if not math.isfinite(ea_nl) or not math.isfinite(k0_nl) or ea_nl <= 0 or k0_nl <= 0:
            result.warnings.append("非线性精修结果非法（非有限数或非正），已回退线性化结果")
            return result

        residuals = [k - _arrhenius_k(t, ea_nl, k0_nl) for t, k in zip(temps, ks)]

        nl_warnings = list(warnings)
        ea_nl_kj = ea_nl / 1000.0
        if not (_EA_TYPICAL_LO_KJ <= ea_nl_kj <= _EA_TYPICAL_HI_KJ):
            nl_warnings.append(
                f"非线性精修后 Ea={ea_nl_kj:.1f} kJ/mol 仍超出典型范围，请核查数据"
            )

        return KineticsFitResult(
            ea=ea_nl, ea_stderr=ea_stderr, k0=k0_nl, k0_stderr=k0_stderr,
            r_squared=r_squared, residuals=residuals, n_points=n,
            fit_quality=_fit_quality_from_r2(r_squared),
            warnings=nl_warnings, method="nonlinear",
        )
    except Exception as exc:  # noqa: BLE001 — curve_fit 可能抛多种异常（不收敛/奇异矩阵等）
        _log.info("fit_arrhenius：非线性精修失败，回退线性化结果：%s", exc)
        result.warnings.append(f"非线性精修未收敛（{exc}），已回退线性化结果")
        return result
