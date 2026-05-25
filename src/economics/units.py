"""
units.py — 经济计算专用单位归一化层。

职责
----
将 Aspen Plus 导出的 BlockOutput.unit 字符串归一化到 TAC 计算所需的基准单位：

  热负荷  → GJ/hr
  功率    → kW
  面积    → m²
  长度    → m
  体积    → m³

Aspen 的 UnitString 取决于 case 的单位集（SI、ENG、METCBAR 等），
同一物理量可能出现多种写法。本模块维护已知写法的映射表；
遇到未知单位字符串时返回 (None, error_msg)，调用方必须将其传播为
ObjectiveValue(error=...)，不得继续计算。

接口
----
每个 normalize_* 函数签名统一为：
    normalize_xxx(value, unit: str) -> tuple[float | None, str | None]
value 接受 float | int | str | None，内部统一转换为 float。
返回 (归一化后的值, 错误信息)。
  - 成功：(float, None)
  - 失败：(None, str)  ← 调用方必须处理
  - value 为 None：(None, "...")
  - value 为 NaN/Inf：(None, "...")
  - value 无法转换为 float：(None, "...")
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def coerce_finite_float(value: Any, quantity_name: str) -> tuple[float | None, str | None]:
    """
    将任意输入转换为有限 float。

    接受 float / int / 可转换为 float 的 str；拒绝 None、NaN、Inf 及无法转换的类型。
    统一在所有 normalize_* 函数入口调用，确保异常契约一致。
    """
    if value is None:
        return None, f"{quantity_name}值为 None"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, f"{quantity_name}值 {value!r} 无法转换为 float"
    if not math.isfinite(v):
        return None, f"{quantity_name}值 {v!r} 不是有限数（NaN/Inf），拒绝计算"
    return v, None


def _normalize_unit_key(unit: str) -> str:
    """
    将 Aspen UnitString 规范化为映射表查找键。

    处理以下变体（不区分大小写，已在调用方 .upper() 后处理）：
    - 前后空格：strip()
    - 上标 Unicode：² → 2，³ → 3
    - 幂次写法：^2 → 2，**2 → 2，^3 → 3，**3 → 3
    - 前缀 SQ（平方）：SQ M → M2，SQ FT → FT2
    - 前缀 CU（立方）：CU M → M3，CU FT → FT3
    - 多余空格压缩
    """
    s = unit.strip().upper()
    # Unicode 上标
    s = s.replace("²", "2").replace("³", "3")
    # 幂次写法
    s = s.replace("**3", "3").replace("**2", "2").replace("^3", "3").replace("^2", "2")
    # SQ/CU 前缀（带空格）
    if s.startswith("SQ "):
        s = s[3:].strip() + "2"
    elif s.startswith("CU "):
        s = s[3:].strip() + "3"
    # 压缩内部多余空格
    s = " ".join(s.split())
    return s


# ---------------------------------------------------------------------------
# 热负荷：目标单位 GJ/hr
# ---------------------------------------------------------------------------

# 各单位到 GJ/hr 的换算系数
_DUTY_TO_GJ_HR: dict[str, float] = {
    # SI
    "GJ/HR":      1.0,
    "GJ/H":       1.0,
    "GJ/HOUR":    1.0,
    "MW":         3.6,           # 1 MW = 3.6 GJ/hr
    "KW":         3.6e-3,        # 1 kW = 0.0036 GJ/hr
    "W":          3.6e-6,
    "KJ/HR":      1e-6,
    "KJ/H":       1e-6,
    "KJ/S":       3.6e-3,        # = kW
    "MJ/HR":      1e-3,
    "MJ/H":       1e-3,
    "GCAL/HR":    4.1868,        # 1 Gcal/hr = 4.1868 GJ/hr
    "GCAL/H":     4.1868,
    "MCAL/HR":    4.1868e-3,
    "MCAL/H":     4.1868e-3,
    "KCAL/HR":    4.1868e-6,
    "KCAL/H":     4.1868e-6,
    "CAL/S":      1.50528e-5,    # 1 cal/s = 3.6 kcal/hr
    # 英制
    "BTU/HR":     1.05506e-6,
    "BTU/H":      1.05506e-6,
    "MMBTU/HR":   1.05506,       # 1 MMBtu/hr = 1.05506 GJ/hr
    "MMBTU/H":    1.05506,
    "KBTU/HR":    1.05506e-3,
    "KBTU/H":     1.05506e-3,
    "HP":         2.68452e-3,    # 1 hp = 0.7457 kW = 2.685e-3 GJ/hr
    # Aspen 常见缩写变体
    "MMKCAL/HR":  4.1868,        # = Gcal/hr
    "MMKCAL/H":   4.1868,
}


def normalize_duty(value: Any, unit: str) -> tuple[float | None, str | None]:
    """
    将热负荷归一化到 GJ/hr。

    Aspen 冷凝器负荷通常为负值（放热），本函数保留符号，
    调用方按需取绝对值。
    """
    v, err = coerce_finite_float(value, "热负荷")
    if v is None:
        return None, err
    key = _normalize_unit_key(unit)
    factor = _DUTY_TO_GJ_HR.get(key)
    if factor is None:
        return None, f"未知热负荷单位 '{unit}'，无法换算到 GJ/hr"
    return v * factor, None


# ---------------------------------------------------------------------------
# 功率：目标单位 kW
# ---------------------------------------------------------------------------

_POWER_TO_KW: dict[str, float] = {
    "KW":      1.0,
    "W":       1e-3,
    "MW":      1e3,
    "GW":      1e6,
    "HP":      0.7457,           # 机械马力
    "BHP":     0.7457,           # 制动马力
    "KJ/HR":   1.0 / 3.6,
    "KJ/H":    1.0 / 3.6,
    "BTU/HR":  2.93071e-4,
    "BTU/H":   2.93071e-4,
    "MMBTU/HR": 293.071,
    "MMBTU/H":  293.071,
    "KCAL/HR": 1.163e-3,
    "KCAL/H":  1.163e-3,
    "GCAL/HR": 1163.0,
    "GCAL/H":  1163.0,
}


def normalize_power(value: Any, unit: str) -> tuple[float | None, str | None]:
    """将功率归一化到 kW。"""
    v, err = coerce_finite_float(value, "功率")
    if v is None:
        return None, err
    key = _normalize_unit_key(unit)
    factor = _POWER_TO_KW.get(key)
    if factor is None:
        return None, f"未知功率单位 '{unit}'，无法换算到 kW"
    return v * factor, None


# ---------------------------------------------------------------------------
# 面积：目标单位 m²
# ---------------------------------------------------------------------------

_AREA_TO_M2: dict[str, float] = {
    "M2":    1.0,
    "M**2":  1.0,
    "M^2":   1.0,
    "SQM":   1.0,
    "CM2":   1e-4,
    "CM**2": 1e-4,
    "FT2":   0.092903,
    "FT**2": 0.092903,
    "FT^2":  0.092903,
    "SQFT":  0.092903,
    "IN2":   6.4516e-4,
    "IN**2": 6.4516e-4,
}


def normalize_area(value: Any, unit: str) -> tuple[float | None, str | None]:
    """将面积归一化到 m²。"""
    v, err = coerce_finite_float(value, "面积")
    if v is None:
        return None, err
    key = _normalize_unit_key(unit)
    factor = _AREA_TO_M2.get(key)
    if factor is None:
        return None, f"未知面积单位 '{unit}'，无法换算到 m²"
    return v * factor, None


# ---------------------------------------------------------------------------
# 长度：目标单位 m
# ---------------------------------------------------------------------------

_LENGTH_TO_M: dict[str, float] = {
    "M":    1.0,
    "CM":   1e-2,
    "MM":   1e-3,
    "KM":   1e3,
    "FT":   0.3048,
    "IN":   0.0254,
    "INCH": 0.0254,
    "YD":   0.9144,
}


def normalize_length(value: Any, unit: str) -> tuple[float | None, str | None]:
    """将长度归一化到 m。"""
    v, err = coerce_finite_float(value, "长度")
    if v is None:
        return None, err
    key = _normalize_unit_key(unit)
    factor = _LENGTH_TO_M.get(key)
    if factor is None:
        return None, f"未知长度单位 '{unit}'，无法换算到 m"
    return v * factor, None


# ---------------------------------------------------------------------------
# 体积：目标单位 m³
# ---------------------------------------------------------------------------

_VOLUME_TO_M3: dict[str, float] = {
    "M3":    1.0,
    "M**3":  1.0,
    "M^3":   1.0,
    "CUM":   1.0,
    "L":     1e-3,
    "LITER": 1e-3,
    "LITRE": 1e-3,
    "ML":    1e-6,
    "CM3":   1e-6,
    "CM**3": 1e-6,
    "FT3":   0.0283168,
    "FT**3": 0.0283168,
    "FT^3":  0.0283168,
    "CUFT":  0.0283168,
    "GAL":   3.78541e-3,   # 美制加仑
    "USGAL": 3.78541e-3,
    "BBL":   0.158987,     # 石油桶
}


def normalize_volume(value: Any, unit: str) -> tuple[float | None, str | None]:
    """将体积归一化到 m³。"""
    v, err = coerce_finite_float(value, "体积")
    if v is None:
        return None, err
    key = _normalize_unit_key(unit)
    factor = _VOLUME_TO_M3.get(key)
    if factor is None:
        return None, f"未知体积单位 '{unit}'，无法换算到 m³"
    return v * factor, None


# ---------------------------------------------------------------------------
# 质量流量：目标单位 kg/hr
# ---------------------------------------------------------------------------

_MASS_FLOW_TO_KG_HR: dict[str, float] = {
    "KG/HR":      1.0,
    "KG/H":       1.0,
    "KG/HOUR":    1.0,
    "KG/S":       3600.0,
    "KG/MIN":     60.0,
    "G/HR":       1e-3,
    "G/H":        1e-3,
    "G/S":        3.6,
    "TONNE/HR":   1000.0,
    "TONNE/H":    1000.0,
    "T/HR":       1000.0,
    "T/H":        1000.0,
    "LB/HR":      0.453592,
    "LB/H":       0.453592,
    "LB/S":       1632.93,
    "KLBS/HR":    453.592,
}


def normalize_mass_flow(value: Any, unit: str) -> tuple[float | None, str | None]:
    """将质量流量归一化到 kg/hr。"""
    v, err = coerce_finite_float(value, "质量流量")
    if v is None:
        return None, err
    key = _normalize_unit_key(unit)
    factor = _MASS_FLOW_TO_KG_HR.get(key)
    if factor is None:
        return None, f"未知质量流量单位 '{unit}'，无法换算到 kg/hr"
    return v * factor, None
