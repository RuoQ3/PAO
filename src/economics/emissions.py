"""
emissions.py — 过程排放量计算模块（Scope 1 + Scope 2）。

职责
----
接收一次仿真工况（ProcessCase）的 block 和 stream 结果，计算：

  Scope 2（间接排放）：
    从 block 的公用工程消耗（蒸汽、电力、冷却水）推算 CO₂ 排放量。
    计算逻辑与 tac.py 对称：读取相同的 block 输出节点（duty、power），
    乘以排放因子而非单价。
    MIXER/FSPLIT/SSPLIT/VALVE/SEP/SEP2/FLASH2/FLASH3/DECANTER/EXTRACT 等
    无公用工程消耗的 block 明确返回 0.0，不进入 skipped_blocks。

  Scope 1（直接工艺排放）：
    从用户指定的排放流股（vent_streams）读取 GHG 组分的质量流量，
    乘以 GWP100 因子换算为 CO₂-eq。
    典型 GHG 组分：CO₂、CH₄、N₂O 等。

输入
----
- ProcessCase：包含 blocks 字典和 streams 字典
- EmissionsConfig：排放因子、GWP 因子、排放流股列表、操作小时数

输出
----
- EmissionsResult：各设备 Scope 2 排放、各流股 Scope 1 排放及汇总值（tonne CO₂-eq/yr）
- make_emissions_objective() 返回 ObjectiveFn，可直接注册到优化循环

与框架的接口关系
----------------
- 消费 src/models/block.py 中的 BlockResult
- 消费 src/models/stream.py 中的 StreamResult / ComponentFlow
- 消费 src/models/process_case.py 中的 ProcessCase / ObjectiveValue
- 消费 src/economics/units.py 中的 normalize_duty / normalize_power / coerce_finite_float
- 产出 ObjectiveFn 类型（Callable[[ProcessCase], ObjectiveValue]）
- 不依赖任何第三方库，仅使用标准库 math / dataclasses

排放因子参考
------------
- 蒸汽（天然气锅炉，η=0.85）：~66 kg CO₂/GJ
- 冷却水（循环泵电耗折算）：~0.5 kg CO₂/GJ（可忽略，默认 0）
- 中国电网（2023）：0.581 kg CO₂/kWh（生态环境部 2023 年发布值）
- GWP100（IPCC AR6）：CO₂=1, CH₄=27.9, N₂O=273

missing_component_policy 说明
------------------------------
"zero"（默认）：ghg_components 中的某个组分不存在于 stream.components，
  按 0 排放处理。适用于真实流程中 vent 流股不含某 GHG 的正常情况。
  注意：组分存在但 mass_flow=None、单位未知、NaN/Inf 时仍必须失败，
  不受此策略影响。

"error"：组分不存在于 stream 时视为数据缺失，触发 skip_missing 策略。
  适用于已知流股必须含某 GHG 的严格校验场景。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from ..models.block import BlockResult
from ..models.process_case import ObjectiveValue, ProcessCase
from ..models.stream import StreamResult
from .units import coerce_finite_float, normalize_duty, normalize_mass_flow, normalize_power

if TYPE_CHECKING:
    ObjectiveFn = Callable[[ProcessCase], ObjectiveValue]


# ---------------------------------------------------------------------------
# GWP100 默认值（IPCC AR6，2021）
# ---------------------------------------------------------------------------

GWP100_DEFAULT: dict[str, float] = {
    "CO2":  1.0,
    "CH4":  27.9,
    "N2O":  273.0,
}


# ---------------------------------------------------------------------------
# block 键名映射（与 tac.py 保持一致）
# ---------------------------------------------------------------------------

_DEFAULT_KEY_MAP: dict[str, dict[str, str]] = {
    "RADFRAC": {"diam": "DIAM", "nstage": "NSTAGE", "reb_duty": "REB_DUTY", "cond_duty": "COND_DUTY"},
    "DISTL":   {"diam": "DIAM", "nstage": "NSTAGE", "reb_duty": "REB_DUTY", "cond_duty": "COND_DUTY"},
    "HEATX":   {"area": "AREA", "duty": "DUTY"},
    "HEATER":  {"area": "AREA", "duty": "DUTY"},
    "PUMP":    {"power": "POWER"},
    "COMPR":   {"power": "POWER"},
    "MCOMPR":  {"power": "POWER"},
    "RCSTR":   {"vol": "VOL", "duty": "DUTY"},
    "RPLUG":   {"vol": "VOL", "duty": "DUTY"},
    "RSTOIC":  {"vol": "VOL", "duty": "DUTY"},
    "RYIELD":  {"vol": "VOL", "duty": "DUTY"},
    "REQUIL":  {"vol": "VOL", "duty": "DUTY"},
    "RGIBBS":  {"vol": "VOL", "duty": "DUTY"},
    "RBATCH":  {"vol": "VOL", "duty": "DUTY"},
}

_COLUMN_TYPES  = {"RADFRAC", "DISTL"}
_HEATX_TYPES   = {"HEATX", "HEATER"}
_PUMP_TYPES    = {"PUMP"}
_COMPR_TYPES   = {"COMPR", "MCOMPR"}
_REACTOR_TYPES = {"RCSTR", "RPLUG", "RSTOIC", "RYIELD", "REQUIL", "RGIBBS", "RBATCH"}

# 确定无外部公用工程消耗的 block 类型：Scope 2 = 0.0，不进入 skipped_blocks。
# 保守原则：只放物理上不可能有外部热/电输入的类型。
# FLASH2/FLASH3/DECANTER/EXTRACT/MHEATX/CRYSTALLIZER 等可能有热负荷，
# 不在此列——默认进入 skipped（未知），用户可通过 zero_scope2_block_types 或
# scope2_block_type_aliases 显式声明处理方式。
_ZERO_EMISSION_TYPES_DEFAULT = frozenset({
    "MIXER",    # 流股混合，无能量输入
    "FSPLIT",   # 流股分割，无能量输入
    "SSPLIT",   # 子流股分割，无能量输入
    "VALVE",    # 节流阀，无外部公用工程
    "PIPE",     # 管道，无外部公用工程
    "PIPELINE", # 管线，无外部公用工程
})


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class EmissionFactors:
    """
    公用工程排放因子。

    steam_factor:
        蒸汽排放因子（kg CO₂/GJ），对应天然气锅炉（η=0.85）。
        天然气低热值 ~50 MJ/kg，燃烧排放 ~56 kg CO₂/GJ，
        锅炉效率 0.85 → 蒸汽侧 ~66 kg CO₂/GJ。
    cooling_water_factor:
        冷却水排放因子（kg CO₂/GJ），循环泵电耗折算，通常可忽略，默认 0。
    electricity_factor:
        电力排放因子（kg CO₂/kWh）。
        默认值为中国电网 2023 年全国平均值（生态环境部发布）。
    """
    steam_factor: float = 66.0           # kg CO₂/GJ
    cooling_water_factor: float = 0.0    # kg CO₂/GJ（可忽略）
    electricity_factor: float = 0.581    # kg CO₂/kWh（中国电网 2023）
    region: str = "CN"
    year: int = 2023
    source: str = "生态环境部 2023 年全国电网平均排放因子；蒸汽因子基于天然气锅炉 η=0.85 估算"


@dataclass
class EmissionsConfig:
    """
    排放量计算的全局配置。

    vent_streams:
        Scope 1 排放流股名称列表（Aspen stream 名称）。
        空列表时跳过 Scope 1 计算，total_scope1 = 0。
    ghg_components:
        GHG 组分名称到 GWP100 因子的映射。
        组分名称必须与 Aspen 组分列表一致（大小写敏感）。
        默认使用 GWP100_DEFAULT（IPCC AR6）。
    missing_component_policy:
        控制 ghg_components 中的组分不存在于 stream.components 时的行为。
        "zero"（默认）：按 0 排放处理，不触发 skip_missing 策略。
        "error"：视为数据缺失，触发 skip_missing 策略。
        注意：组分存在但 mass_flow=None、单位未知、NaN/Inf 时始终视为数据错误。
    zero_scope2_block_types:
        用户追加的零排放 block 类型集合，合并到内置 _ZERO_EMISSION_TYPES_DEFAULT。
        用于声明用户确认无外部公用工程的 block 类型，如 SEP/SEP2/FLASH2 等。
        示例：zero_scope2_block_types={"SEP", "SEP2", "FLASH2"}
    scope2_block_type_aliases:
        block 类型别名映射 {自定义类型: 已知类型}，用于将未知 block 类型路由到
        已有计算函数。例如将 MHEATX 按 HEATX 处理：
        scope2_block_type_aliases={"MHEATX": "HEATX"}
        注意：output_key_map 不能覆盖类型分发，必须通过此字段声明。
    output_key_map:
        用户覆盖层，合并时用户配置优先于 _DEFAULT_KEY_MAP。
    skip_missing:
        True 时跳过无法计算的 block/stream，对剩余部分求和（可能低估）。
        False（默认）时任一 block/stream 被跳过则 total = None。
    allow_partial_objective:
        True 时允许 partial 排放量进入优化目标（需用户明确承担低估风险）。
    """
    operating_hours: float = 8000.0
    emission_factors: EmissionFactors = field(default_factory=EmissionFactors)
    vent_streams: list[str] = field(default_factory=list)
    ghg_components: dict[str, float] = field(default_factory=lambda: dict(GWP100_DEFAULT))
    missing_component_policy: Literal["zero", "error"] = "zero"
    zero_scope2_block_types: set[str] = field(default_factory=set)
    scope2_block_type_aliases: dict[str, str] = field(default_factory=dict)
    output_key_map: dict[str, dict[str, str]] = field(default_factory=dict)
    skip_missing: bool = False
    allow_partial_objective: bool = False


@dataclass
class EquipmentEmissions:
    """单台设备的 Scope 2 排放估算结果（tonne CO₂-eq/yr）。"""
    block_name: str
    block_type: str
    scope2_annual: float | None    # tonne CO₂-eq/yr
    notes: str = ""


@dataclass
class StreamEmissions:
    """单条排放流股的 Scope 1 排放估算结果（tonne CO₂-eq/yr）。"""
    stream_name: str
    component_emissions: dict[str, float | None]   # {组分名: tonne CO₂-eq/yr}
    scope1_annual: float | None                    # 该流股所有 GHG 组分之和
    has_missing_data: bool = False                 # 是否有组分数据缺失（供上层决定是否加入 skipped）
    notes: str = ""


@dataclass
class EmissionsResult:
    """整个工况的排放量汇总结果。"""
    equipment_emissions: list[EquipmentEmissions]
    stream_emissions: list[StreamEmissions]
    skipped_blocks: list[str]
    skipped_streams: list[str]
    total_scope2_annual: float | None    # tonne CO₂-eq/yr
    total_scope1_annual: float | None    # tonne CO₂-eq/yr
    total_annual: float | None           # scope1 + scope2
    operating_hours: float
    notes: str = ""


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _get_val(
    block: BlockResult,
    semantic_key: str,
    key_map: dict[str, str],
) -> tuple[Any, str, str | None]:
    """从 block.outputs 按语义键查找原始值和单位，返回 (raw_value, unit, fetch_error)。"""
    output_name = key_map.get(semantic_key)
    if output_name is None:
        return None, "", f"语义键 '{semantic_key}' 不在键名映射中"
    output = block.get_output(output_name)
    if output is None:
        return None, "", f"输出节点 '{output_name}' 不存在于 block '{block.name}'"
    if output.value is None:
        return None, output.unit, f"节点 '{output_name}'（block '{block.name}'）的值为 None"
    return output.value, output.unit, None


def _resolve_key_map(block_type_value: str, user_map: dict[str, dict[str, str]]) -> dict[str, str]:
    """合并默认键名映射与用户覆盖层，用户配置优先。"""
    default = _DEFAULT_KEY_MAP.get(block_type_value, {})
    user    = user_map.get(block_type_value, {})
    return {**default, **user}


def _duty_to_co2_kg_hr(duty_gj_hr: float, factor_kg_per_gj: float) -> float:
    """duty_gj_hr * factor_kg_per_gj → kg CO₂/hr"""
    return duty_gj_hr * factor_kg_per_gj


def _power_to_co2_kg_hr(power_kw: float, factor_kg_per_kwh: float) -> float:
    """power_kw * factor_kg_per_kwh → kg CO₂/hr"""
    return power_kw * factor_kg_per_kwh


def _kg_hr_to_tonne_yr(kg_hr: float, operating_hours: float) -> float:
    """kg/hr × 操作小时数 / 1000 → tonne/yr"""
    return kg_hr * operating_hours / 1000.0


# ---------------------------------------------------------------------------
# Scope 2：各设备类型排放计算
# ---------------------------------------------------------------------------

def _calc_column_scope2(block: BlockResult, config: EmissionsConfig) -> EquipmentEmissions:
    """
    精馏塔 Scope 2 排放。

    再沸器负荷 → 蒸汽消耗 → CO₂（steam_factor）。
    冷凝器负荷 → 冷却水消耗 → CO₂（cooling_water_factor）。

    reb_duty 缺失/单位错误：scope2=None（蒸汽排放是主要贡献，不可忽略）。
    cond_duty 缺失/单位错误：
      - cooling_water_factor == 0：按 0 处理，记录 note，不使 scope2=None。
      - cooling_water_factor != 0：scope2=None，防止低估排放。
    """
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ef        = config.emission_factors

    reb_raw, reb_unit, reb_ferr   = _get_val(block, "reb_duty",  key_map)
    cond_raw, cond_unit, cond_ferr = _get_val(block, "cond_duty", key_map)

    notes_parts: list[str] = []
    co2_kg_hr: float = 0.0
    reb_valid = True
    cond_valid = True

    if reb_raw is None:
        notes_parts.append(reb_ferr or "缺少再沸器负荷（reb_duty），蒸汽排放无法计算")
        reb_valid = False
    else:
        reb_norm, reb_err = normalize_duty(reb_raw, reb_unit)
        if reb_norm is None:
            notes_parts.append(f"再沸器负荷单位归一化失败：{reb_err}")
            reb_valid = False
        else:
            co2_kg_hr += _duty_to_co2_kg_hr(abs(reb_norm), ef.steam_factor)

    if cond_raw is None:
        if ef.cooling_water_factor == 0.0:
            notes_parts.append(cond_ferr or "缺少冷凝器负荷（cond_duty），冷却水排放因子为 0，按 0 处理")
        else:
            notes_parts.append(
                (cond_ferr or "缺少冷凝器负荷（cond_duty）") +
                f"，cooling_water_factor={ef.cooling_water_factor} != 0，scope2=None"
            )
            cond_valid = False
    else:
        cond_norm, cond_err = normalize_duty(cond_raw, cond_unit)
        if cond_norm is None:
            if ef.cooling_water_factor == 0.0:
                notes_parts.append(f"冷凝器负荷单位归一化失败（{cond_err}），冷却水排放因子为 0，按 0 处理")
            else:
                notes_parts.append(
                    f"冷凝器负荷单位归一化失败（{cond_err}）"
                    f"，cooling_water_factor={ef.cooling_water_factor} != 0，scope2=None"
                )
                cond_valid = False
        else:
            co2_kg_hr += _duty_to_co2_kg_hr(abs(cond_norm), ef.cooling_water_factor)

    scope2 = _kg_hr_to_tonne_yr(co2_kg_hr, config.operating_hours) if (reb_valid and cond_valid) else None

    return EquipmentEmissions(
        block_name=block.name,
        block_type=btype_val,
        scope2_annual=scope2,
        notes="; ".join(notes_parts),
    )


def _calc_heatx_scope2(block: BlockResult, config: EmissionsConfig) -> EquipmentEmissions:
    """
    换热器 / 加热器 Scope 2 排放。

    duty > 0 → 蒸汽加热；duty < 0 → 冷却水冷却；duty = 0 → 无公用工程排放。
    """
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ef        = config.emission_factors

    duty_raw, duty_unit, duty_ferr = _get_val(block, "duty", key_map)

    notes_parts: list[str] = []
    scope2: float | None = None

    if duty_raw is None:
        notes_parts.append(duty_ferr or "缺少热负荷（duty），无法计算 Scope 2 排放")
    else:
        duty_norm, duty_err = normalize_duty(duty_raw, duty_unit)
        if duty_norm is None:
            notes_parts.append(f"热负荷单位归一化失败：{duty_err}")
        elif duty_norm > 0:
            scope2 = _kg_hr_to_tonne_yr(
                _duty_to_co2_kg_hr(duty_norm, ef.steam_factor), config.operating_hours
            )
        elif duty_norm < 0:
            scope2 = _kg_hr_to_tonne_yr(
                _duty_to_co2_kg_hr(abs(duty_norm), ef.cooling_water_factor), config.operating_hours
            )
        else:
            scope2 = 0.0

    return EquipmentEmissions(
        block_name=block.name,
        block_type=btype_val,
        scope2_annual=scope2,
        notes="; ".join(notes_parts),
    )


def _calc_pump_scope2(block: BlockResult, config: EmissionsConfig) -> EquipmentEmissions:
    """泵 Scope 2 排放（电力）。"""
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ef        = config.emission_factors

    power_raw, power_unit, power_ferr = _get_val(block, "power", key_map)

    notes_parts: list[str] = []
    scope2: float | None = None

    if power_raw is None:
        notes_parts.append(power_ferr or "缺少轴功率（power），无法计算 Scope 2 排放")
    else:
        power_norm, power_err = normalize_power(power_raw, power_unit)
        if power_norm is None:
            notes_parts.append(f"轴功率单位归一化失败：{power_err}")
        else:
            scope2 = _kg_hr_to_tonne_yr(
                _power_to_co2_kg_hr(abs(power_norm), ef.electricity_factor), config.operating_hours
            )

    return EquipmentEmissions(
        block_name=block.name,
        block_type=btype_val,
        scope2_annual=scope2,
        notes="; ".join(notes_parts),
    )


def _calc_compr_scope2(block: BlockResult, config: EmissionsConfig) -> EquipmentEmissions:
    """压缩机 Scope 2 排放（电力）。"""
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ef        = config.emission_factors

    power_raw, power_unit, power_ferr = _get_val(block, "power", key_map)

    notes_parts: list[str] = []
    scope2: float | None = None

    if power_raw is None:
        notes_parts.append(power_ferr or "缺少轴功率（power），无法计算 Scope 2 排放")
    else:
        power_norm, power_err = normalize_power(power_raw, power_unit)
        if power_norm is None:
            notes_parts.append(f"轴功率单位归一化失败：{power_err}")
        else:
            scope2 = _kg_hr_to_tonne_yr(
                _power_to_co2_kg_hr(abs(power_norm), ef.electricity_factor), config.operating_hours
            )

    return EquipmentEmissions(
        block_name=block.name,
        block_type=btype_val,
        scope2_annual=scope2,
        notes="; ".join(notes_parts),
    )


def _calc_reactor_scope2(block: BlockResult, config: EmissionsConfig) -> EquipmentEmissions:
    """
    反应器 Scope 2 排放（热负荷）。

    duty > 0 → 蒸汽加热；duty < 0 → 冷却水冷却。
    """
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ef        = config.emission_factors

    duty_raw, duty_unit, duty_ferr = _get_val(block, "duty", key_map)

    notes_parts: list[str] = []
    scope2: float | None = None

    if duty_raw is None:
        notes_parts.append(duty_ferr or "缺少热负荷（duty），无法计算 Scope 2 排放")
    else:
        duty_norm, duty_err = normalize_duty(duty_raw, duty_unit)
        if duty_norm is None:
            notes_parts.append(f"热负荷单位归一化失败：{duty_err}")
        elif duty_norm > 0:
            scope2 = _kg_hr_to_tonne_yr(
                _duty_to_co2_kg_hr(duty_norm, ef.steam_factor), config.operating_hours
            )
        elif duty_norm < 0:
            scope2 = _kg_hr_to_tonne_yr(
                _duty_to_co2_kg_hr(abs(duty_norm), ef.cooling_water_factor), config.operating_hours
            )
        else:
            scope2 = 0.0

    return EquipmentEmissions(
        block_name=block.name,
        block_type=btype_val,
        scope2_annual=scope2,
        notes="; ".join(notes_parts),
    )


def _calc_equipment_scope2(block: BlockResult, config: EmissionsConfig) -> EquipmentEmissions:
    """
    根据 block_type 分发到对应 Scope 2 计算函数。

    分发优先级：
    1. scope2_block_type_aliases：将自定义类型路由到已知类型（递归一次）
       alias 路由时，将 resolved 类型的默认键名映射合并到 output_key_map，
       确保计算函数能正确查找输出节点。
    2. 已知计算类型（COLUMN/HEATX/PUMP/COMPR/REACTOR）
    3. 零排放类型（内置 + 用户追加）
    4. 未知类型 → skipped
    """
    btype_val = block.block_type.value

    # 1. 别名路由（只展开一层，防止循环）
    resolved = config.scope2_block_type_aliases.get(btype_val, btype_val)

    # 若有 alias，将 resolved 类型的默认 key_map 合并到 output_key_map，
    # 使计算函数能用 resolved 类型的键名查找 block.outputs
    effective_config = config
    if resolved != btype_val:
        resolved_default = _DEFAULT_KEY_MAP.get(resolved, {})
        if resolved_default:
            merged_key_map = {**{resolved: resolved_default}, **config.output_key_map}
            # 同时把 btype_val 映射到 resolved 的键名，供 _resolve_key_map 查找
            merged_key_map[btype_val] = {**resolved_default, **config.output_key_map.get(btype_val, {})}
            from dataclasses import replace
            effective_config = replace(config, output_key_map=merged_key_map)

    if resolved in _COLUMN_TYPES:
        return _calc_column_scope2(block, effective_config)
    if resolved in _HEATX_TYPES:
        return _calc_heatx_scope2(block, effective_config)
    if resolved in _PUMP_TYPES:
        return _calc_pump_scope2(block, effective_config)
    if resolved in _COMPR_TYPES:
        return _calc_compr_scope2(block, effective_config)
    if resolved in _REACTOR_TYPES:
        return _calc_reactor_scope2(block, effective_config)

    # 2. 零排放类型（内置 + 用户追加）
    zero_types = _ZERO_EMISSION_TYPES_DEFAULT | config.zero_scope2_block_types
    if resolved in zero_types:
        return EquipmentEmissions(
            block_name=block.name,
            block_type=btype_val,
            scope2_annual=0.0,
            notes="",
        )

    # 3. 未知类型：无法判断是否有公用工程消耗，进入 skipped_blocks
    return EquipmentEmissions(
        block_name=block.name,
        block_type=btype_val,
        scope2_annual=None,
        notes=f"未知 block 类型 '{btype_val}'，无法判断 Scope 2 排放，跳过（可通过 zero_scope2_block_types 或 scope2_block_type_aliases 声明处理方式）",
    )


# ---------------------------------------------------------------------------
# Scope 1：排放流股 GHG 组分计算
# ---------------------------------------------------------------------------

def _calc_stream_scope1(stream: StreamResult, config: EmissionsConfig) -> StreamEmissions:
    """
    计算单条排放流股的 Scope 1 排放。

    对 config.ghg_components 中的每个组分，从 stream.components 读取质量流量，
    乘以 GWP100 因子，换算为 tonne CO₂-eq/yr。

    组分不存在于 stream 时，行为由 missing_component_policy 控制：
    - "zero"：按 0 处理，不触发 any_missing（默认）
    - "error"：视为数据缺失，触发 any_missing

    组分存在但 mass_flow=None、单位未知、NaN/Inf 时始终视为数据错误，
    不受 missing_component_policy 影响。
    """
    notes_parts: list[str] = []
    comp_emissions: dict[str, float | None] = {}
    total_co2_kg_hr: float = 0.0
    any_missing = False

    for comp_name, gwp in config.ghg_components.items():
        comp = stream.get_component(comp_name)
        if comp is None:
            if config.missing_component_policy == "zero":
                comp_emissions[comp_name] = 0.0
                # 不记录 notes，不设 any_missing：组分不存在是正常情况
            else:
                notes_parts.append(
                    f"组分 '{comp_name}' 不存在于流股 '{stream.name}'"
                    "（missing_component_policy='error'）"
                )
                comp_emissions[comp_name] = None
                any_missing = True
            continue

        # 组分存在，但 mass_flow 字段本身有问题 → 始终失败，不受 policy 影响
        if comp.mass_flow is None:
            notes_parts.append(f"组分 '{comp_name}' 的质量流量为 None")
            comp_emissions[comp_name] = None
            any_missing = True
            continue

        mass_norm, mass_err = normalize_mass_flow(comp.mass_flow, comp.mass_flow_unit)
        if mass_norm is None:
            notes_parts.append(f"组分 '{comp_name}' 质量流量单位归一化失败：{mass_err}")
            comp_emissions[comp_name] = None
            any_missing = True
            continue

        co2_eq_kg_hr = mass_norm * gwp
        tonne_yr = _kg_hr_to_tonne_yr(co2_eq_kg_hr, config.operating_hours)
        comp_emissions[comp_name] = tonne_yr
        total_co2_kg_hr += co2_eq_kg_hr

    if any_missing and not config.skip_missing:
        scope1_annual: float | None = None
        notes_parts.append("skip_missing=False，部分组分数据错误，scope1_annual=None")
    else:
        # 对所有非 None 值求和（包含 policy=zero 时的 0.0）
        valid_vals = [v for v in comp_emissions.values() if v is not None]
        scope1_annual = _kg_hr_to_tonne_yr(total_co2_kg_hr, config.operating_hours) if valid_vals else None

    return StreamEmissions(
        stream_name=stream.name,
        component_emissions=comp_emissions,
        scope1_annual=scope1_annual,
        has_missing_data=any_missing,
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# 主计算函数
# ---------------------------------------------------------------------------

def calculate_emissions(case: ProcessCase, config: EmissionsConfig) -> EmissionsResult:
    """
    计算整个 ProcessCase 的排放量（Scope 1 + Scope 2）。

    Scope 2：遍历 case.blocks，对每个 converged 的 block 计算公用工程排放。
    Scope 1：遍历 config.vent_streams，对每个 converged 的 stream 计算 GHG 直排。

    total_annual 的 None 策略由 EmissionsConfig.skip_missing 控制：
    - False（默认）：任一 block/stream 被跳过 → total=None（保守策略）
    - True：跳过无法计算的部分，对剩余求和（可能低估，notes 中说明）
    """
    equipment_emissions: list[EquipmentEmissions] = []
    stream_emissions: list[StreamEmissions] = []
    skipped_blocks: list[str] = []
    skipped_streams: list[str] = []
    global_notes: list[str] = []

    # Scope 2
    for block_name, block in case.blocks.items():
        if not block.converged:
            skipped_blocks.append(block_name)
            global_notes.append(f"Block '{block_name}'({block.block_type.value}) 未收敛，跳过 Scope 2")
            continue
        ee = _calc_equipment_scope2(block, config)
        equipment_emissions.append(ee)
        if ee.scope2_annual is None:
            skipped_blocks.append(block_name)

    # Scope 1
    for stream_name in config.vent_streams:
        stream = case.streams.get(stream_name)
        if stream is None:
            skipped_streams.append(stream_name)
            global_notes.append(f"排放流股 '{stream_name}' 不存在于 case.streams，跳过 Scope 1")
            continue
        if not stream.converged:
            skipped_streams.append(stream_name)
            global_notes.append(f"排放流股 '{stream_name}' 未收敛，跳过 Scope 1")
            continue
        se = _calc_stream_scope1(stream, config)
        stream_emissions.append(se)
        if se.scope1_annual is None or se.has_missing_data:
            skipped_streams.append(stream_name)

    # 汇总 Scope 2
    if not config.skip_missing and skipped_blocks:
        total_scope2: float | None = None
        global_notes.append(
            f"skip_missing=False，以下 block 无法计算 Scope 2，total_scope2=None：{skipped_blocks}"
        )
    else:
        valid_s2 = [ee.scope2_annual for ee in equipment_emissions if ee.scope2_annual is not None]
        # skip_missing=True 时对有效设备求和；无任何设备时为 0.0
        total_scope2 = sum(valid_s2) if valid_s2 else 0.0
        if skipped_blocks and config.skip_missing:
            global_notes.append(f"skip_missing=True，以下 block 已跳过（Scope 2 可能低估）：{skipped_blocks}")

    # 汇总 Scope 1
    if not config.vent_streams:
        total_scope1: float | None = 0.0
    elif not config.skip_missing and skipped_streams:
        total_scope1 = None
        global_notes.append(
            f"skip_missing=False，以下流股无法计算 Scope 1，total_scope1=None：{skipped_streams}"
        )
    else:
        valid_s1 = [se.scope1_annual for se in stream_emissions if se.scope1_annual is not None]
        # skip_missing=True 时对有效流股求和；无任何流股时为 0.0
        total_scope1 = sum(valid_s1) if valid_s1 else 0.0
        if skipped_streams and config.skip_missing:
            global_notes.append(f"skip_missing=True，以下流股已跳过（Scope 1 可能低估）：{skipped_streams}")

    # 合并
    if total_scope2 is not None and total_scope1 is not None:
        total_annual: float | None = total_scope2 + total_scope1
    else:
        total_annual = None

    for ee in equipment_emissions:
        if ee.notes:
            global_notes.append(f"  [{ee.block_name}({ee.block_type})] {ee.notes}")
    for se in stream_emissions:
        if se.notes:
            global_notes.append(f"  [stream:{se.stream_name}] {se.notes}")

    return EmissionsResult(
        equipment_emissions=equipment_emissions,
        stream_emissions=stream_emissions,
        skipped_blocks=skipped_blocks,
        skipped_streams=skipped_streams,
        total_scope2_annual=total_scope2,
        total_scope1_annual=total_scope1,
        total_annual=total_annual,
        operating_hours=config.operating_hours,
        notes="\n".join(global_notes),
    )


# ---------------------------------------------------------------------------
# ObjectiveFn 工厂
# ---------------------------------------------------------------------------

def make_emissions_objective(config: EmissionsConfig) -> "ObjectiveFn":
    """
    返回一个符合 ObjectiveFn 协议的可调用对象（最小化总排放量）。

    partial 排放保护：skip_missing=True 且有 block/stream 被跳过时，
    默认仍返回 error，防止低估排放的工况被优化器当成更优解。
    需显式设置 EmissionsConfig.allow_partial_objective=True 才允许 partial 排放进入优化。
    """
    def emissions_objective(case: ProcessCase) -> ObjectiveValue:
        result = calculate_emissions(case, config)
        if result.total_annual is None:
            error_msg = result.notes if result.notes else "排放量计算失败，部分设备或流股数据缺失"
            return ObjectiveValue(
                name="EMISSIONS", value=None, unit="tonne CO2-eq/yr",
                minimize=True, error=error_msg,
            )
        skipped = result.skipped_blocks + result.skipped_streams
        if skipped and not config.allow_partial_objective:
            error_msg = (
                f"排放量可能低估（{len(skipped)} 个 block/stream 被跳过）：{skipped}。"
                "如需允许 partial 排放进入优化，请设置 EmissionsConfig.allow_partial_objective=True。"
            )
            return ObjectiveValue(
                name="EMISSIONS", value=None, unit="tonne CO2-eq/yr",
                minimize=True, error=error_msg,
            )
        if not math.isfinite(result.total_annual):
            return ObjectiveValue(
                name="EMISSIONS", value=None, unit="tonne CO2-eq/yr",
                minimize=True,
                error=f"排放量计算结果为非有限数 {result.total_annual!r}，拒绝作为优化目标",
            )
        return ObjectiveValue(
            name="EMISSIONS", value=result.total_annual,
            unit="tonne CO2-eq/yr", minimize=True,
        )

    emissions_objective.__name__ = "EMISSIONS"
    return emissions_objective
