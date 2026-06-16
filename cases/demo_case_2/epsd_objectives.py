"""
epsd_objectives.py — 精确复现 MATLAB simulationOPEX() / simulationCAPEX() 的目标函数。

对应文件：cases/demo_case_2/epsd/simulationOPEX.m
         cases/demo_case_2/epsd/simulationCAPEX.m

计算说明
---------
CAPEX（优化目标 1）：
    = CO₂ + SO₂ + NOₓ 年排放量（tonne/yr）
    基于三塔再沸器燃料消耗，使用固定排放因子。

OPEX（优化目标 2）：
    = TCC/3 + TOC
    TCC = 三塔设备成本 + 换热器 H 成本（改良 Guthrie 法，CEPCI=1431.7/280 折算）
    TOC = 蒸汽费 + 冷却水费 + 真空系统运行费 + 换热器 H 冷却水费

所需 Aspen 输出节点（须在 output_paths 中配置）：
    三塔：OUTPUT/REB_DUTY, OUTPUT/COND_DUTY, OUTPUT/TOP_TEMP, OUTPUT/BOTTOM_TEMP
    三塔：INPUT/NSTAGE, INPUT/PRES1
    三塔塔径：Subobjects/Column Internals/INT-1/Input/CA_SUMP_DIAM/INT-1
    换热器 H：OUTPUT/QCALC
    W3 流股：Output/TEMP_OUT/MIXED
"""
from __future__ import annotations

import math
import logging
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aspen 节点路径常量
# ---------------------------------------------------------------------------

_PATHS = {
    "T1_REB_DUTY":  r"\Data\Blocks\T1\Output\REB_DUTY",
    "T1_COND_DUTY": r"\Data\Blocks\T1\Output\COND_DUTY",
    "T1_BOT_TEMP":  r"\Data\Blocks\T1\Output\BOTTOM_TEMP",
    "T1_TOP_TEMP":  r"\Data\Blocks\T1\Output\TOP_TEMP",
    "T1_NSTAGE":    r"\Data\Blocks\T1\Input\NSTAGE",
    "T1_PRES":      r"\Data\Blocks\T1\Input\PRES1",
    # 塔径节点：2epsd.bkp 中塔径通过 Column Internals 计算后输出在 CA_DIAM6
    "T1_DIAM":      r"\Data\Blocks\T1\Output\CA_DIAM6",

    "T2_REB_DUTY":  r"\Data\Blocks\T2\Output\REB_DUTY",
    "T2_COND_DUTY": r"\Data\Blocks\T2\Output\COND_DUTY",
    "T2_BOT_TEMP":  r"\Data\Blocks\T2\Output\BOTTOM_TEMP",
    "T2_TOP_TEMP":  r"\Data\Blocks\T2\Output\TOP_TEMP",
    "T2_NSTAGE":    r"\Data\Blocks\T2\Input\NSTAGE",
    "T2_PRES":      r"\Data\Blocks\T2\Input\PRES1",
    "T2_DIAM":      r"\Data\Blocks\T2\Output\CA_DIAM6",

    "T3_REB_DUTY":  r"\Data\Blocks\T3\Output\REB_DUTY",
    "T3_COND_DUTY": r"\Data\Blocks\T3\Output\COND_DUTY",
    "T3_BOT_TEMP":  r"\Data\Blocks\T3\Output\BOTTOM_TEMP",
    "T3_TOP_TEMP":  r"\Data\Blocks\T3\Output\TOP_TEMP",
    "T3_NSTAGE":    r"\Data\Blocks\T3\Input\NSTAGE",
    "T3_PRES":      r"\Data\Blocks\T3\Input\PRES1",
    "T3_DIAM":      r"\Data\Blocks\T3\Output\CA_DIAM6",

    "H_QCALC":      r"\Data\Blocks\H\Output\QCALC",
    "W3_TEMP":      r"\Data\Streams\W3\Output\TEMP_OUT\MIXED",
}

# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _get(outputs: dict, key: str) -> float:
    """从 outputs 字典中取值，路径不存在或非数值时抛 ValueError。

    PAO 在 manifest 提取模式下，outputs 中的值可能是 VariableResult 对象
    （含 .value 属性），而非裸数值。统一先尝试提取 .value，再转 float。
    """
    path = _PATHS[key]
    raw = outputs.get(path)
    if raw is None:
        raise ValueError(f"节点 '{path}' 不在 outputs 中，请检查 output_paths 配置")
    # 兼容 VariableResult 包装类型（manifest 模式）
    if hasattr(raw, "value"):
        raw = raw.value
    if raw is None:
        raise ValueError(f"节点 '{path}' 的 VariableResult.value 为 None")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"节点 '{path}' 值 {raw!r} 无法转换为 float")
    if not math.isfinite(val):
        raise ValueError(f"节点 '{path}' 值 {val!r} 为非有限数")
    return val


def _steam_price_and_area(Qr: float, Tr: float) -> tuple[float, float]:
    """
    按再沸器底温 Tr（°C）决定蒸汽价格 Cs（元/GJ × hr × 3.6）和再沸器面积 Ar（m²）。
    对应 MATLAB simulationOPEX.m 第 18-64 行的分段逻辑。
    Qr 单位：GJ/hr（已完成 0.0041868 换算）。
    """
    if 0 < Tr < 160:
        Cs = 7.78
        Ar = Qr / ((160 - Tr) * 0.568)
    elif 160 <= Tr < 184:
        Cs = 8.22
        Ar = Qr / ((184 - Tr) * 0.568)
    elif 184 <= Tr < 254:
        Cs = 9.88
        Ar = Qr / ((254 - Tr) * 0.568)
    else:
        Cs = 15.0
        Ar = Qr / ((350 - Tr) * 0.568)
    return Cs, Ar


def _cooling_price_and_lmtd(Tc: float) -> tuple[float, float]:
    """
    按冷凝器顶温 Tc（°C）决定冷却水价格 Cw（元/GJ × hr × 3.6）和对数平均温差 deT。
    对应 MATLAB simulationOPEX.m 第 67-105 行。
    """
    if Tc < -10:
        Cw = 100000.0
        deT = ((Tc + 100) - (Tc + 90)) / math.log((Tc + 100) / (Tc + 90))
    elif -10 <= Tc < 15:
        Cw = 7.89
        deT = ((Tc + 20) - (Tc + 10)) / math.log((Tc + 20) / (Tc + 10))
    elif 15 <= Tc < 42:
        Cw = 4.43
        deT = ((Tc - 5) - (Tc - 15)) / math.log((Tc - 5) / (Tc - 15))
    else:
        Cw = 0.354
        deT = ((Tc - 32) - (Tc - 42)) / math.log((Tc - 32) / (Tc - 42))
    return Cw, deT


def _column_equipment_cost(
    N: float, D: float, P: float,
    Qr: float, Qc: float,
    Tr: float, Tc: float,
) -> tuple[float, float, float]:
    """
    单塔设备成本（改良 Guthrie 法，CEPCI=1431.7/280）。
    返回 (Cost_column, CostSteam_annual, CostWater_annual)。
    - Cost_column：塔设备一次性投资（元）= CostT + CostN + Costc + Costr
    - CostSteam_annual：蒸汽年运行费（元/yr）
    - CostWater_annual：冷却水年运行费（元/yr）
    """
    cepci = 1431.7 / 280

    H = (N / 0.75 - 3) * 0.6 + 6  # 塔有效高度，m

    # 压力系数
    if 0 < P < 3.4001:
        Fp = 1.0
    elif P <= 6.8:
        Fp = 1.05
    else:
        Fp = 1.15

    CostT = cepci * 937.636 * (D ** 1.066) * (H ** 0.802) * (2.18 + 3.67 * Fp)
    CostN = cepci * 97.243 * (D ** 1.55) * H * 2.7

    Cs, Ar = _steam_price_and_area(Qr, Tr)
    Cw, deT = _cooling_price_and_lmtd(Tc)

    Ac = Qc / (deT * 0.852)   # 冷凝器面积，m²
    Costc = cepci * 474.668 * (Ac ** 0.65) * 5.29
    Costr = cepci * 474.668 * (Ar ** 0.65) * 7.3525

    Cost_col = Costc + Costr + CostT + CostN

    # 年运行费：Qr/Qc 单位 GJ/hr，×8 小时/天 ×3.6（换算系数）×Cs/Cw（元/GJ）
    # MATLAB 中: CostSteam = Qr * 8 * 3.6 * Cs （仅乘以 8，非 8000 小时，因为 MATLAB 的 Cs 单位已含年）
    # 参考 simulationOPEX.m 第 107-109 行，实际是按天计算后 ×365≈8760 包含在 Cs 中？
    # 逐字复现 MATLAB：CostSteam += Qr*8*3.6*Cs （单次调用，返回年费用）
    CostSteam_annual = Qr * 8 * 3.6 * Cs
    CostWater_annual = Qc * 8 * 3.6 * Cw

    return Cost_col, CostSteam_annual, CostWater_annual


def _vacuum_cost(N: float, D: float, P: float) -> float:
    """
    真空系统成本（当操作压力 P < 1 atm 时）。
    对应 MATLAB simulationOPEX.m 第 188-222 行。
    P 单位：atm。返回一次性设备成本（元）。
    """
    if P <= 0 or P >= 1:
        return 0.0
    H = (N / 0.75 - 3) * 0.6 + 6
    V = 35.3 * H * math.pi * ((D / 2) ** 2)
    log_P760 = math.log(P * 760)
    WW = 5 + (0.0298 + 0.03088 * log_P760 - 0.0005733 * log_P760 ** 2) * (V ** 0.66)
    return 1690 * 1.8 * ((WW / P) ** 0.41)


def _heatx_cost_and_annual(Qch: float, Th: float) -> tuple[float, float]:
    """
    换热器 H 的设备成本和冷却水年费。
    对应 MATLAB simulationOPEX.m 第 170-186 行。
    Qch 单位：GJ/hr（已完成 0.0041868 换算，取正值）。
    Th：W3 流股出口温度（°C）。
    返回 (Costch1, CostWaterh1)。
    """
    cepci = 1431.7 / 280
    Cwh, deTh = _cooling_price_and_lmtd(Th)
    Ach = Qch / (deTh * 0.852)
    Costch = cepci * 474.668 * (Ach ** 0.65) * 5.29
    CostWaterh = Qch * 8 * 3.6 * Cwh
    return Costch, CostWaterh


# ---------------------------------------------------------------------------
# CAPEX 目标函数（排放量）
# ---------------------------------------------------------------------------

def make_epsd_capex_objective():
    """
    返回 ObjectiveFn：计算 CO₂+SO₂+NOₓ 年排放量（tonne/yr）。

    完全对应 MATLAB simulationCAPEX.m。
    只需要三塔的 REB_DUTY（再沸器热负荷）。
    """
    from src.models.process_case import ObjectiveValue

    name = "CAPEX"
    unit = "tonne/yr"

    def capex_fn(case: Any) -> ObjectiveValue:
        outputs: dict = {}
        if case.sim_result is not None:
            outputs = case.sim_result.outputs or {}

        try:
            # 单位换算：Aspen 默认 Gcal/hr → GJ/hr（×0.0041868×1000=4.1868 kJ → GJ 需 ×0.0041868）
            Qr1 = 0.0041868 * _get(outputs, "T1_REB_DUTY")
            Qr2 = 0.0041868 * _get(outputs, "T2_REB_DUTY")
            Qr3 = 0.0041868 * _get(outputs, "T3_REB_DUTY")
        except ValueError as exc:
            return ObjectiveValue(name=name, value=None, unit=unit, minimize=True, error=str(exc))

        Qr_total = Qr1 + Qr2 + Qr3
        denominator = 0.92 * 29307.6  # 燃料热值折算系数（kJ/yr → Gcal/yr 已含年操作）

        CO2 = 2.493  * Qr_total / denominator * 8000
        SO2 = 0.075  * Qr_total / denominator * 8000
        NOX = 0.0375 * Qr_total / denominator * 8000
        CAPEX = CO2 + SO2 + NOX

        if not math.isfinite(CAPEX) or CAPEX < 0:
            return ObjectiveValue(
                name=name, value=None, unit=unit, minimize=True,
                error=f"CAPEX 计算结果异常：{CAPEX!r}（Qr_total={Qr_total:.4f} GJ/hr）",
            )

        return ObjectiveValue(name=name, value=CAPEX, unit=unit, minimize=True)

    capex_fn.__name__ = name
    return capex_fn


# ---------------------------------------------------------------------------
# OPEX 目标函数（设备折旧 + 年运行费）
# ---------------------------------------------------------------------------

def make_epsd_opex_objective():
    """
    返回 ObjectiveFn：计算 OPEX = TCC/3 + TOC（改良 Guthrie 法，3 年折旧）。

    完全对应 MATLAB simulationOPEX.m。
    所需节点见模块顶部 _PATHS 字典，须在 output_paths 中全部配置。
    """
    from src.models.process_case import ObjectiveValue

    name = "OPEX"
    unit = "RMB/yr"

    def opex_fn(case: Any) -> ObjectiveValue:
        outputs: dict = {}
        if case.sim_result is not None:
            outputs = case.sim_result.outputs or {}

        try:
            # ---- 三塔热负荷（GJ/hr）----
            Qr1 = 0.0041868 * _get(outputs, "T1_REB_DUTY")
            Qr2 = 0.0041868 * _get(outputs, "T2_REB_DUTY")
            Qr3 = 0.0041868 * _get(outputs, "T3_REB_DUTY")
            # 冷凝器为负值（放热），取正值用于计算面积和费用
            Qc1 = -0.0041868 * _get(outputs, "T1_COND_DUTY")
            Qc2 = -0.0041868 * _get(outputs, "T2_COND_DUTY")
            Qc3 = -0.0041868 * _get(outputs, "T3_COND_DUTY")

            # ---- 温度（°C）----
            Tr1 = _get(outputs, "T1_BOT_TEMP")
            Tc1 = _get(outputs, "T1_TOP_TEMP")
            Tr2 = _get(outputs, "T2_BOT_TEMP")
            Tc2 = _get(outputs, "T2_TOP_TEMP")
            Tr3 = _get(outputs, "T3_BOT_TEMP")
            Tc3 = _get(outputs, "T3_TOP_TEMP")

            # ---- 设计参数 ----
            N1 = _get(outputs, "T1_NSTAGE")
            D1 = _get(outputs, "T1_DIAM")
            P1 = _get(outputs, "T1_PRES")

            N2 = _get(outputs, "T2_NSTAGE")
            D2 = _get(outputs, "T2_DIAM")
            P2 = _get(outputs, "T2_PRES")

            N3 = _get(outputs, "T3_NSTAGE")
            D3 = _get(outputs, "T3_DIAM")
            P3 = _get(outputs, "T3_PRES")

            # ---- 换热器 H ----
            Qch1 = -0.0041868 * _get(outputs, "H_QCALC")
            Th1  = _get(outputs, "W3_TEMP")

        except ValueError as exc:
            return ObjectiveValue(name=name, value=None, unit=unit, minimize=True, error=str(exc))

        # ---- 三塔设备成本 & 年运行费 ----
        Cost1, CostSteam1, CostWater1 = _column_equipment_cost(N1, D1, P1, Qr1, Qc1, Tr1, Tc1)
        Cost2, CostSteam2, CostWater2 = _column_equipment_cost(N2, D2, P2, Qr2, Qc2, Tr2, Tc2)
        Cost3, CostSteam3, CostWater3 = _column_equipment_cost(N3, D3, P3, Qr3, Qc3, Tr3, Tc3)

        # ---- 换热器 H ----
        Costch1, CostWaterh1 = _heatx_cost_and_annual(Qch1, Th1)

        # ---- 真空系统 ----
        Vacuum1 = _vacuum_cost(N1, D1, P1)
        Vacuum2 = _vacuum_cost(N2, D2, P2)
        Vacuum3 = _vacuum_cost(N3, D3, P3)
        Vacuum  = Vacuum1 + Vacuum2 + Vacuum3

        # ---- 汇总 ----
        TOC1 = CostSteam1 + CostWater1 + CostSteam2 + CostWater2 + CostSteam3 + CostWater3
        TCC  = Cost1 + Cost2 + Cost3 + Costch1
        TOC  = TOC1 + Vacuum + CostWaterh1
        OPEX = TCC / 3 + TOC

        if not math.isfinite(OPEX) or OPEX < 0:
            return ObjectiveValue(
                name=name, value=None, unit=unit, minimize=True,
                error=f"OPEX 计算结果异常：{OPEX!r}（TCC={TCC:.2f}, TOC={TOC:.2f}）",
            )

        _log.debug(
            "OPEX 计算：TCC=%.2f, TOC=%.2f → OPEX=%.2f %s",
            TCC, TOC, OPEX, unit,
        )
        return ObjectiveValue(name=name, value=OPEX, unit=unit, minimize=True)

    opex_fn.__name__ = name
    return opex_fn
