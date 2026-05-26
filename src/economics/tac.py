"""
tac.py — 基于 Turton 方法的总年化成本（TAC）计算模块。

职责
----
接收一次仿真工况（ProcessCase）的 block 结果，按设备类型估算资本成本（CAPEX）
和年度公用工程成本（OPEX），汇总为 TAC（$/yr）。

输入优先级
----------
1. case.semantic_blocks（manifest runtime 模式产出的语义字段）
   - 直接按字段名读取 reboiler_duty / condenser_duty / column_diameter / nstage 等
   - 单位来自 SemanticField.unit，由 normalize_* 处理
2. case.blocks（full/debug 模式产出的 BlockResult）
   - 通过 _DEFAULT_KEY_MAP 按 BlockOutput.name 查找
3. TACConfig.block_design_params（fallback 设计参数）

输出
----
- TACResult：各设备 EquipmentCost 列表、跳过的 block 列表及汇总值
- make_tac_objective() 返回 ObjectiveFn，可直接注册到优化循环

与框架的接口关系
----------------
- 消费 src/models/block.py 中的 BlockResult / BlockOutput
- 消费 src/models/node_catalog.py 中的 SemanticBlock / SemanticField
- 消费 src/models/process_case.py 中的 ProcessCase / ObjectiveValue
- 消费 src/economics/units.py 中的 normalize_* 函数（单位归一化）
- 产出 ObjectiveFn 类型（Callable[[ProcessCase], ObjectiveValue]）
- 不依赖任何第三方库，仅使用标准库 math / dataclasses

Turton 方法参考
---------------
Turton R. et al., "Analysis, Synthesis, and Design of Chemical Processes",
4th ed., Prentice Hall, 2012. Table A.1 设备成本关联式。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ..models.block import BlockResult, BlockType
from ..models.node_catalog import SemanticBlock
from ..models.process_case import ObjectiveValue, ProcessCase
from .units import normalize_area, normalize_duty, normalize_length, normalize_power, normalize_volume

if TYPE_CHECKING:
    ObjectiveFn = Callable[[ProcessCase], ObjectiveValue]


# ---------------------------------------------------------------------------
# 默认键名映射
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

# 支持 TAC 计算的 block 类型集合
_COLUMN_TYPES  = {"RADFRAC", "DISTL"}
_HEATX_TYPES   = {"HEATX", "HEATER"}
_PUMP_TYPES    = {"PUMP"}
_COMPR_TYPES   = {"COMPR", "MCOMPR"}
_REACTOR_TYPES = {"RCSTR", "RPLUG", "RSTOIC", "RYIELD", "REQUIL", "RGIBBS", "RBATCH"}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class UtilityCost:
    """
    公用工程单价。

    默认值来自 Turton 2018 附录，适用于北美工厂估算。
    region/year/source 字段用于追溯价格来源，不同地区或年份应通过 TACConfig 覆盖。
    """
    steam_price: float = 14.19          # $/GJ，低压蒸汽（LP steam, ~5 barg）
    cooling_water_price: float = 0.354  # $/GJ
    chilled_water_price: float = 4.43   # $/GJ
    electricity_price: float = 0.0775   # $/kWh
    region: str = "demo"
    year: int = 2023
    currency: str = "USD"
    source: str = "Turton 2018 (demo values, not for production use)"


@dataclass
class EquipmentCostParams:
    """
    Turton Table A.1 设备成本关联式系数及安装因子。

    log10(Cp0) = K1 + K2*log10(A) + K3*(log10(A))^2
    CTM = Cp0 * (cepci_current / cepci_ref) * FBM

    K 系数的有效范围（A 的范围）见 Turton Table A.1 原表；
    超出范围时计算结果为外推估算，精度下降。
    """
    cepci_ref: float = 397.0       # 2001 年 CEPCI 基准值
    cepci_current: float = 800.0   # 当前 CEPCI 估算值（需定期更新）

    # (K1, K2, K3)，A = 特征尺寸（见各设备说明）
    column_k: tuple = (3.4974, 0.4485, 0.1074)   # 塔体，A = 体积 (m³)
    heatx_k: tuple  = (4.3247, -0.3030, 0.1634)  # 换热器，A = 面积 (m²)
    pump_k: tuple   = (3.3892, 0.0536, 0.1538)   # 泵，A = 轴功率 (kW)
    compr_k: tuple  = (2.2897, 1.3604, -0.1027)  # 压缩机，A = 轴功率 (kW)
    reactor_k: tuple = (3.4974, 0.4485, 0.1074)  # 反应器，A = 体积 (m³)，暂用塔体系数

    # 总模块因子 FBM（含材质、压力修正的综合安装因子）
    column_fbm: float  = 4.16
    heatx_fbm: float   = 3.17
    pump_fbm: float    = 3.30
    compr_fbm: float   = 6.10
    reactor_fbm: float = 4.16

    source: str = "Turton 2018 Table A.1 (demo values, not for production use)"


@dataclass
class TACConfig:
    """
    TAC 计算的全局配置。

    output_key_map 是用户覆盖层，合并时用户配置优先于 _DEFAULT_KEY_MAP。
    仅需覆盖与默认值不同的条目，其余条目自动继承默认值。

    block_design_params 是设计参数 fallback 层，格式为
    ``{block_name: {semantic_key: SI_value}}``，例如：
    ``{"T0301": {"diam": 2.5, "nstage": 30}}``。
    当 Aspen Output 子树中找不到对应节点（如 NSTAGE 在 Input 子树），
    或节点值无效（如 DIAM=0 且单位为空，表示未做 sizing），
    则从此处取固定设计参数值（已是 SI 单位：m / - / GJ/hr / m² / kW / m³）。
    fallback 触发时会在 EquipmentCost.notes 中记录，不静默替换。

    from_db 是预留接口，economic_db.py 实现后可按 region/year 加载价格。
    """
    annualization_factor: float = 0.1    # CAPEX 年化因子，对应 10 年直线折旧
    operating_hours: float = 8000.0      # 年操作小时数
    utility_cost: UtilityCost = field(default_factory=UtilityCost)
    equipment_params: EquipmentCostParams = field(default_factory=EquipmentCostParams)
    output_key_map: dict[str, dict[str, str]] = field(default_factory=dict)
    block_design_params: dict[str, dict[str, float]] = field(default_factory=dict)
    skip_missing: bool = False
    # skip_missing=True 时，partial TAC（有设备被跳过）默认仍返回 ObjectiveValue(error=...)，
    # 防止低估成本的工况被优化器当成更优解。
    # 显式设为 True 才允许 partial TAC 进入优化目标（需用户明确承担低估风险）。
    allow_partial_objective: bool = False

    @classmethod
    def from_db(cls, db_path: str, region: str, year: int) -> "TACConfig":
        raise NotImplementedError("economic_db.py 尚未实现，请手动构造 TACConfig")


@dataclass
class EquipmentCost:
    """单台设备的成本估算结果。"""
    block_name: str
    block_type: str
    capex: float | None          # 设备总模块成本 CTM（$）
    opex_annual: float | None    # 年度公用工程成本（$/yr）
    tac: float | None            # = capex * annualization_factor + opex_annual
    notes: str = ""


@dataclass
class TACResult:
    """整个工况的 TAC 汇总结果。"""
    equipment_costs: list[EquipmentCost]
    skipped_blocks: list[str]        # 未收敛/不支持/缺数据的 block 名称列表
    total_capex: float | None
    total_opex_annual: float | None
    total_tac: float | None
    annualization_factor: float
    notes: str = ""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _get_val(
    block: BlockResult,
    semantic_key: str,
    key_map: dict[str, str],
) -> tuple[Any, str, str | None]:
    """
    从 block.outputs 按语义键查找原始值和单位，返回 (raw_value, unit, fetch_error)。

    - 语义键不在 key_map 中：(None, "", "语义键 '{key}' 不在键名映射中")
    - 输出节点不存在：(None, "", "输出节点 '{name}' 不存在于 block '{block.name}'")
    - 节点值为 None：(None, unit, "节点 '{name}' 的值为 None")
    - 成功：(raw_value, unit, None)

    不在此处做 float 转换，原始值交给 normalize_* 处理，
    保留完整诊断信息供 agent 自动修复 Aspen 提取路径。
    """
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
    """
    合并默认键名映射与用户覆盖层，用户配置优先。

    返回该 block_type 的完整 {语义键: BlockOutput.name} 映射。
    """
    default = _DEFAULT_KEY_MAP.get(block_type_value, {})
    user    = user_map.get(block_type_value, {})
    return {**default, **user}


def _turton_cost(
    A: float,
    k: tuple,
    fbm: float,
    cepci_ref: float,
    cepci_current: float,
) -> float:
    """
    Turton 设备购置成本关联式。

    log10(Cp0) = K1 + K2*log10(A) + K3*(log10(A))^2
    CTM = Cp0 * (cepci_current / cepci_ref) * FBM

    A 必须是有限正数，否则抛出 ValueError。
    """
    if not math.isfinite(A) or A <= 0:
        raise ValueError(f"Turton 关联式要求特征尺寸 A 为有限正数，实际值 A={A}")
    log_a = math.log10(A)
    k1, k2, k3 = k
    log_cp0 = k1 + k2 * log_a + k3 * log_a ** 2
    cp0 = 10 ** log_cp0
    return cp0 * (cepci_current / cepci_ref) * fbm


def _duty_gj_hr_to_annual_cost(duty_gj_hr: float, price_per_gj: float, hours: float) -> float:
    """duty_gj_hr * hours * price_per_gj"""
    return duty_gj_hr * hours * price_per_gj


def _power_kw_to_annual_cost(power_kw: float, price_per_kwh: float, hours: float) -> float:
    """power_kw * hours * price_per_kwh"""
    return power_kw * hours * price_per_kwh


# ---------------------------------------------------------------------------
# 语义字段读取辅助
# ---------------------------------------------------------------------------

def _get_semantic_val(
    semantic_block: SemanticBlock | None,
    field_name: str,
) -> tuple[Any, str, str | None]:
    """
    从 SemanticBlock 读取语义字段，返回 (raw_value, unit, fetch_error)。

    - semantic_block 为 None：(None, "", "无 semantic_block")
    - 字段不存在：(None, "", "字段 '{field_name}' 不在 semantic_block 中")
    - 字段不可用（error 非空）：(None, "", error_msg)
    - 成功：(value, unit, None)
    """
    if semantic_block is None:
        return None, "", "无 semantic_block（非 manifest 模式）"
    sf = semantic_block.get(field_name)
    if sf is None:
        return None, "", f"字段 '{field_name}' 不在 semantic_block 中"
    if not sf.available:
        return None, sf.unit, (
            sf.error or f"字段 '{field_name}' 不可用（value=None）"
        )
    return sf.value, sf.unit, None


# ---------------------------------------------------------------------------
# 各设备类型成本计算
# ---------------------------------------------------------------------------

def _calc_column_cost(
    block: BlockResult,
    config: TACConfig,
    design_params: dict[str, float] | None = None,
    semantic_block: SemanticBlock | None = None,
) -> EquipmentCost:
    """
    精馏塔 CAPEX + OPEX。

    读取优先级（每个字段独立）：
    1. semantic_block（manifest runtime 模式）
    2. block.outputs（full 模式，通过 key_map 查找）
    3. design_params（fallback 设计参数）

    塔体积估算：V = π/4 * D² * N * tray_spacing（工程估算，tray_spacing=0.6m）。
    此公式将塔视为等截面圆柱，忽略封头和裙座，适用于概念设计阶段精度（±30%）。

    再沸器负荷（reb_duty）→ 低压蒸汽成本。
    冷凝器负荷（cond_duty）→ 冷却水成本；Aspen 中冷凝器负荷通常为负值（放热），
    取绝对值后计算，若绝对值极小（< 1e-6 GJ/hr）则视为零负荷。

    design_params:
        来自 TACConfig.block_design_params[block.name] 的设计参数 fallback，
        格式 ``{semantic_key: SI_value}``，如 ``{"diam": 2.5, "nstage": 30}``。
        当 Aspen Output 子树缺少节点或值无效时使用，触发时记录到 notes。
    semantic_block:
        manifest runtime 模式产出的 SemanticBlock，优先于 block.outputs 读取。
    """
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ep        = config.equipment_params
    uc        = config.utility_cost

    notes_parts: list[str] = []
    capex: float | None = None

    # --- 塔径 ---
    diam_raw, diam_unit, diam_ferr = _get_semantic_val(semantic_block, "column_diameter")
    if diam_raw is None and diam_ferr and semantic_block is not None:
        notes_parts.append(f"[semantic] column_diameter: {diam_ferr}，尝试 block.outputs fallback")
    if diam_raw is None:
        diam_raw, diam_unit, diam_ferr = _get_val(block, "diam", key_map)

    diam: float | None = None
    if diam_raw is None:
        if design_params and "diam" in design_params:
            diam = float(design_params["diam"])
            notes_parts.append(
                f"DIAM 不可用（{diam_ferr}），使用设计参数 diam={diam:.3f} m"
            )
        else:
            notes_parts.append(diam_ferr or "缺少塔径（diam），无法计算 CAPEX")
    else:
        diam_norm, diam_err = normalize_length(diam_raw, diam_unit)
        if diam_norm is None:
            if design_params and "diam" in design_params:
                diam = float(design_params["diam"])
                notes_parts.append(
                    f"塔径单位归一化失败（{diam_err}），使用设计参数 diam={diam:.3f} m"
                )
            else:
                notes_parts.append(f"塔径单位归一化失败：{diam_err}")
        else:
            diam = diam_norm

    # --- 板数 ---
    nstage_raw, _, nstage_ferr = _get_semantic_val(semantic_block, "nstage")
    if nstage_raw is None and nstage_ferr and semantic_block is not None:
        notes_parts.append(f"[semantic] nstage: {nstage_ferr}，尝试 block.outputs fallback")
    if nstage_raw is None:
        nstage_raw, _, nstage_ferr = _get_val(block, "nstage", key_map)

    nstage: float | None = None
    if nstage_raw is None:
        if design_params and "nstage" in design_params:
            n_val = float(design_params["nstage"])
            if n_val <= 0:
                notes_parts.append(f"设计参数 nstage={n_val!r} 必须为正数，拒绝计算 CAPEX")
            elif abs(n_val - round(n_val)) > 1e-9:
                notes_parts.append(f"设计参数 nstage={n_val!r} 不是整数，拒绝计算 CAPEX")
            else:
                nstage = float(round(n_val))
                notes_parts.append(
                    f"NSTAGE 不可用（{nstage_ferr}），使用设计参数 nstage={int(nstage)}"
                )
        else:
            notes_parts.append(nstage_ferr or "缺少板数（nstage），无法计算 CAPEX")
    else:
        from .units import coerce_finite_float
        n_val, n_err = coerce_finite_float(nstage_raw, "板数")
        if n_val is None:
            notes_parts.append(f"板数（nstage）数值非法：{n_err}")
        elif n_val <= 0:
            notes_parts.append(f"板数（nstage）必须为正数，实际值 {n_val!r}")
        elif abs(n_val - round(n_val)) > 1e-9:
            notes_parts.append(f"板数（nstage）{n_val!r} 不是整数，拒绝计算 CAPEX")
        else:
            nstage = float(round(n_val))

    if diam is not None and nstage is not None:
        tray_spacing = 0.6  # m，标准板间距工程估算值
        volume = (math.pi / 4.0) * diam ** 2 * nstage * tray_spacing
        try:
            capex = _turton_cost(volume, ep.column_k, ep.column_fbm, ep.cepci_ref, ep.cepci_current)
        except ValueError as exc:
            notes_parts.append(f"CAPEX 计算失败：{exc}")
            capex = None

    # --- 再沸器负荷 ---
    reb_raw, reb_unit, reb_ferr = _get_semantic_val(semantic_block, "reboiler_duty")
    if reb_raw is None and reb_ferr and semantic_block is not None:
        notes_parts.append(f"[semantic] reboiler_duty: {reb_ferr}，尝试 block.outputs fallback")
    if reb_raw is None:
        reb_raw, reb_unit, reb_ferr = _get_val(block, "reb_duty", key_map)

    # --- 冷凝器负荷 ---
    cond_raw, cond_unit, cond_ferr = _get_semantic_val(semantic_block, "condenser_duty")
    if cond_raw is None and cond_ferr and semantic_block is not None:
        notes_parts.append(f"[semantic] condenser_duty: {cond_ferr}，尝试 block.outputs fallback")
    if cond_raw is None:
        cond_raw, cond_unit, cond_ferr = _get_val(block, "cond_duty", key_map)

    opex: float = 0.0
    opex_valid = True

    if reb_raw is None:
        notes_parts.append(reb_ferr or "缺少再沸器负荷（reb_duty），蒸汽成本无法计算")
        opex_valid = False
    else:
        reb_norm, reb_err = normalize_duty(reb_raw, reb_unit)
        if reb_norm is None:
            notes_parts.append(f"再沸器负荷单位归一化失败：{reb_err}")
            opex_valid = False
        else:
            opex += _duty_gj_hr_to_annual_cost(abs(reb_norm), uc.steam_price, config.operating_hours)

    if cond_raw is None:
        notes_parts.append(cond_ferr or "缺少冷凝器负荷（cond_duty），冷却水成本无法计算")
        opex_valid = False
    else:
        cond_norm, cond_err = normalize_duty(cond_raw, cond_unit)
        if cond_norm is None:
            notes_parts.append(f"冷凝器负荷单位归一化失败：{cond_err}")
            opex_valid = False
        else:
            opex += _duty_gj_hr_to_annual_cost(abs(cond_norm), uc.cooling_water_price, config.operating_hours)

    opex_annual: float | None = opex if opex_valid else None

    tac: float | None = None
    if capex is not None and opex_annual is not None:
        tac = capex * config.annualization_factor + opex_annual

    return EquipmentCost(
        block_name=block.name,
        block_type=btype_val,
        capex=capex,
        opex_annual=opex_annual,
        tac=tac,
        notes="; ".join(notes_parts),
    )


def _calc_heatx_cost(block: BlockResult, config: TACConfig) -> EquipmentCost:
    """
    换热器 / 加热器 CAPEX + OPEX。

    duty > 0 → 加热，使用蒸汽；duty < 0 → 冷却，使用冷却水。
    duty = 0 时 OPEX 为 0（纯换热，无外部公用工程）。
    """
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ep        = config.equipment_params
    uc        = config.utility_cost

    area_raw, area_unit, area_ferr = _get_val(block, "area", key_map)
    duty_raw, duty_unit, duty_ferr = _get_val(block, "duty", key_map)

    notes_parts: list[str] = []
    capex: float | None = None

    if area_raw is None:
        notes_parts.append(area_ferr or "缺少换热面积（area），无法计算 CAPEX")
    else:
        area_norm, area_err = normalize_area(area_raw, area_unit)
        if area_norm is None:
            notes_parts.append(f"换热面积单位归一化失败：{area_err}")
        else:
            try:
                capex = _turton_cost(area_norm, ep.heatx_k, ep.heatx_fbm, ep.cepci_ref, ep.cepci_current)
            except ValueError as exc:
                notes_parts.append(f"CAPEX 计算失败：{exc}")
                capex = None

    opex_annual: float | None = None

    if duty_raw is None:
        notes_parts.append(duty_ferr or "缺少热负荷（duty），无法计算 OPEX")
    else:
        duty_norm, duty_err = normalize_duty(duty_raw, duty_unit)
        if duty_norm is None:
            notes_parts.append(f"热负荷单位归一化失败：{duty_err}")
        elif duty_norm > 0:
            opex_annual = _duty_gj_hr_to_annual_cost(duty_norm, uc.steam_price, config.operating_hours)
        elif duty_norm < 0:
            opex_annual = _duty_gj_hr_to_annual_cost(abs(duty_norm), uc.cooling_water_price, config.operating_hours)
        else:
            opex_annual = 0.0

    tac: float | None = None
    if capex is not None and opex_annual is not None:
        tac = capex * config.annualization_factor + opex_annual

    return EquipmentCost(
        block_name=block.name,
        block_type=btype_val,
        capex=capex,
        opex_annual=opex_annual,
        tac=tac,
        notes="; ".join(notes_parts),
    )


def _calc_pump_cost(block: BlockResult, config: TACConfig) -> EquipmentCost:
    """泵 CAPEX（功率关联式）+ OPEX（电力）。"""
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ep        = config.equipment_params
    uc        = config.utility_cost

    power_raw, power_unit, power_ferr = _get_val(block, "power", key_map)

    notes_parts: list[str] = []
    capex: float | None = None
    opex_annual: float | None = None

    if power_raw is None:
        notes_parts.append(power_ferr or "缺少轴功率（power），无法计算 CAPEX 和 OPEX")
    else:
        power_norm, power_err = normalize_power(power_raw, power_unit)
        if power_norm is None:
            notes_parts.append(f"轴功率单位归一化失败：{power_err}")
        else:
            try:
                capex = _turton_cost(power_norm, ep.pump_k, ep.pump_fbm, ep.cepci_ref, ep.cepci_current)
            except ValueError as exc:
                notes_parts.append(f"CAPEX 计算失败：{exc}")
                capex = None
            opex_annual = _power_kw_to_annual_cost(abs(power_norm), uc.electricity_price, config.operating_hours)

    tac: float | None = None
    if capex is not None and opex_annual is not None:
        tac = capex * config.annualization_factor + opex_annual

    return EquipmentCost(
        block_name=block.name,
        block_type=btype_val,
        capex=capex,
        opex_annual=opex_annual,
        tac=tac,
        notes="; ".join(notes_parts),
    )


def _calc_compr_cost(block: BlockResult, config: TACConfig) -> EquipmentCost:
    """压缩机 CAPEX（功率关联式）+ OPEX（电力）。"""
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ep        = config.equipment_params
    uc        = config.utility_cost

    power_raw, power_unit, power_ferr = _get_val(block, "power", key_map)

    notes_parts: list[str] = []
    capex: float | None = None
    opex_annual: float | None = None

    if power_raw is None:
        notes_parts.append(power_ferr or "缺少轴功率（power），无法计算 CAPEX 和 OPEX")
    else:
        power_norm, power_err = normalize_power(power_raw, power_unit)
        if power_norm is None:
            notes_parts.append(f"轴功率单位归一化失败：{power_err}")
        else:
            try:
                capex = _turton_cost(power_norm, ep.compr_k, ep.compr_fbm, ep.cepci_ref, ep.cepci_current)
            except ValueError as exc:
                notes_parts.append(f"CAPEX 计算失败：{exc}")
                capex = None
            opex_annual = _power_kw_to_annual_cost(abs(power_norm), uc.electricity_price, config.operating_hours)

    tac: float | None = None
    if capex is not None and opex_annual is not None:
        tac = capex * config.annualization_factor + opex_annual

    return EquipmentCost(
        block_name=block.name,
        block_type=btype_val,
        capex=capex,
        opex_annual=opex_annual,
        tac=tac,
        notes="; ".join(notes_parts),
    )


def _calc_reactor_cost(block: BlockResult, config: TACConfig) -> EquipmentCost:
    """
    反应器 CAPEX（体积关联式）+ OPEX（热负荷）。

    duty > 0 → 加热，使用蒸汽；duty < 0 → 冷却，使用冷却水。
    体积（vol）缺失时 CAPEX 为 None；热负荷（duty）缺失时 OPEX 为 None。
    """
    btype_val = block.block_type.value
    key_map   = _resolve_key_map(btype_val, config.output_key_map)
    ep        = config.equipment_params
    uc        = config.utility_cost

    vol_raw, vol_unit, vol_ferr   = _get_val(block, "vol",  key_map)
    duty_raw, duty_unit, duty_ferr = _get_val(block, "duty", key_map)

    notes_parts: list[str] = []
    capex: float | None = None
    opex_annual: float | None = None

    if vol_raw is None:
        notes_parts.append(vol_ferr or "缺少反应器体积（vol），无法计算 CAPEX")
    else:
        vol_norm, vol_err = normalize_volume(vol_raw, vol_unit)
        if vol_norm is None:
            notes_parts.append(f"反应器体积单位归一化失败：{vol_err}")
        else:
            try:
                capex = _turton_cost(vol_norm, ep.reactor_k, ep.reactor_fbm, ep.cepci_ref, ep.cepci_current)
            except ValueError as exc:
                notes_parts.append(f"CAPEX 计算失败：{exc}")
                capex = None

    if duty_raw is None:
        notes_parts.append(duty_ferr or "缺少热负荷（duty），无法计算 OPEX")
    else:
        duty_norm, duty_err = normalize_duty(duty_raw, duty_unit)
        if duty_norm is None:
            notes_parts.append(f"热负荷单位归一化失败：{duty_err}")
        elif duty_norm > 0:
            opex_annual = _duty_gj_hr_to_annual_cost(duty_norm, uc.steam_price, config.operating_hours)
        elif duty_norm < 0:
            opex_annual = _duty_gj_hr_to_annual_cost(abs(duty_norm), uc.cooling_water_price, config.operating_hours)
        else:
            opex_annual = 0.0

    tac: float | None = None
    if capex is not None and opex_annual is not None:
        tac = capex * config.annualization_factor + opex_annual

    return EquipmentCost(
        block_name=block.name,
        block_type=btype_val,
        capex=capex,
        opex_annual=opex_annual,
        tac=tac,
        notes="; ".join(notes_parts),
    )



# ---------------------------------------------------------------------------
# 分发函数
# ---------------------------------------------------------------------------

def _calc_equipment_cost(
    block: BlockResult,
    config: TACConfig,
    semantic_block: SemanticBlock | None = None,
) -> EquipmentCost:
    """
    根据 block_type 分发到对应计算函数。

    semantic_block 优先于 block.outputs 读取语义字段（manifest runtime 模式）。
    不支持的类型（如 FLASH2、MIXER 等）返回 capex=None, opex_annual=None，
    notes 中说明原因，不抛出异常，由上层 skip_missing 策略决定如何处理。
    """
    btype_val = block.block_type.value
    design_params = config.block_design_params.get(block.name)

    if btype_val in _COLUMN_TYPES:
        return _calc_column_cost(block, config, design_params, semantic_block)
    if btype_val in _HEATX_TYPES:
        return _calc_heatx_cost(block, config)
    if btype_val in _PUMP_TYPES:
        return _calc_pump_cost(block, config)
    if btype_val in _COMPR_TYPES:
        return _calc_compr_cost(block, config)
    if btype_val in _REACTOR_TYPES:
        return _calc_reactor_cost(block, config)

    return EquipmentCost(
        block_name=block.name,
        block_type=btype_val,
        capex=None,
        opex_annual=None,
        tac=None,
        notes=f"不支持的 block 类型 '{btype_val}'，跳过成本计算",
    )


# ---------------------------------------------------------------------------
# 主计算函数
# ---------------------------------------------------------------------------

def calculate_tac(case: ProcessCase, config: TACConfig) -> TACResult:
    """
    计算整个 ProcessCase 的 TAC。

    遍历 case.blocks，对每个 converged 的 block 调用 _calc_equipment_cost。
    若 case.semantic_blocks 非空，优先从语义字段读取（manifest runtime 模式）。
    未收敛的 block 直接进入 skipped_blocks，不影响其他 block 的计算。
    tac=None 的设备（不支持类型或关键数据缺失）也进入 skipped_blocks。

    total_tac 的 None 策略由 TACConfig.skip_missing 控制：
    - False（默认）：任一 block 被跳过 -> total_tac=None（保守策略，避免低估）
    - True：跳过无法计算的 block，对剩余设备求和（total_tac 可能低估，notes 中说明）
    """
    equipment_costs: list[EquipmentCost] = []
    skipped_blocks: list[str] = []
    global_notes: list[str] = []

    use_semantic = bool(case.semantic_blocks)
    if use_semantic:
        global_notes.append("使用 semantic_blocks（manifest runtime 模式）读取设备参数")

    for block_name, block in case.blocks.items():
        if not block.converged:
            skipped_blocks.append(block_name)
            global_notes.append(
                f"Block '{block_name}'({block.block_type.value}) 未收敛，跳过"
            )
            continue
        semantic_block = case.semantic_blocks.get(block_name) if use_semantic else None
        ec = _calc_equipment_cost(block, config, semantic_block)
        equipment_costs.append(ec)
        if ec.tac is None:
            skipped_blocks.append(block_name)

    capex_values = [ec.capex for ec in equipment_costs if ec.capex is not None]
    total_capex: float | None = sum(capex_values) if capex_values else None

    opex_values = [ec.opex_annual for ec in equipment_costs if ec.opex_annual is not None]
    total_opex: float | None = sum(opex_values) if opex_values else None

    if not config.skip_missing and skipped_blocks:
        total_tac: float | None = None
        global_notes.append(
            f"skip_missing=False，以下 block 无法计算 TAC，total_tac=None：{skipped_blocks}"
        )
    else:
        valid_tac = [ec.tac for ec in equipment_costs if ec.tac is not None]
        total_tac = sum(valid_tac) if valid_tac else None
        if skipped_blocks and config.skip_missing:
            global_notes.append(
                f"skip_missing=True，以下 block 已跳过（total_tac 可能低估）：{skipped_blocks}"
            )

    for ec in equipment_costs:
        if ec.notes:
            global_notes.append(f"  [{ec.block_name}({ec.block_type})] {ec.notes}")

    return TACResult(
        equipment_costs=equipment_costs,
        skipped_blocks=skipped_blocks,
        total_capex=total_capex,
        total_opex_annual=total_opex,
        total_tac=total_tac,
        annualization_factor=config.annualization_factor,
        notes="\n".join(global_notes),
    )


# ---------------------------------------------------------------------------
# ObjectiveFn 工厂
# ---------------------------------------------------------------------------

def make_tac_objective(config: TACConfig) -> "ObjectiveFn":
    """
    返回一个符合 ObjectiveFn 协议的可调用对象。

    返回函数的 __name__ 设为 "TAC"，与优化循环中 objective_name 配置对应。

    partial TAC 保护：skip_missing=True 且有设备被跳过时，默认仍返回 error，
    防止低估成本的工况被优化器当成更优解。
    需显式设置 TACConfig.allow_partial_objective=True 才允许 partial TAC 进入优化。
    """
    def tac_objective(case: ProcessCase) -> ObjectiveValue:
        result = calculate_tac(case, config)
        if result.total_tac is None:
            error_msg = result.notes if result.notes else "TAC 计算失败，部分设备数据缺失"
            return ObjectiveValue(name="TAC", value=None, unit="$/yr", minimize=True, error=error_msg)
        if result.skipped_blocks and not config.allow_partial_objective:
            error_msg = (
                f"TAC 可能低估（{len(result.skipped_blocks)} 个设备被跳过）："
                f"{result.skipped_blocks}。"
                "如需允许 partial TAC 进入优化，请设置 TACConfig.allow_partial_objective=True。"
            )
            return ObjectiveValue(name="TAC", value=None, unit="$/yr", minimize=True, error=error_msg)
        if not math.isfinite(result.total_tac):
            return ObjectiveValue(
                name="TAC", value=None, unit="$/yr", minimize=True,
                error=f"TAC 计算结果为非有限数 {result.total_tac!r}，拒绝作为优化目标",
            )
        return ObjectiveValue(name="TAC", value=result.total_tac, unit="$/yr", minimize=True)

    tac_objective.__name__ = "TAC"
    return tac_objective
