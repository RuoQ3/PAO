"""
preflight.py — 工况飞行前检查(Pre-flight Check)。

在把候选点提交给 Aspen 之前,做一道纯计算的预检,拦截"必然超时/不收敛"
的荒谬工况。预检不通过的点直接判为不可行,根本不调用 Aspen,代价几乎为零。

设计原则(系统化、与具体工艺无关)
--------------------------------
所有规则都不写死任何工艺数字(如"压力<0.5"),只用相对规则:

  规则 1：偏离已知可行解的幅度(核心,普适)
    候选值相对参考值(初始收敛解)的归一化偏离不得超过 max_deviation_factor。
    参考值来自工程师调好的初始可行工况——任何能跑的 .bkp 都自带这个锚点。
    这一条就能拦掉绝大多数地狱工况(如塔板数翻 3 倍、压力翻百倍)。

  规则 2：变量依赖一致性(普适)
    复用已有的 var_dependencies(如 进料板 < 总塔板数)。

  规则 3：计算量代理指标(可选,需变量角色信息)
    某些变量组合的乘积正比于 Aspen 单点计算量(如精馏塔的 塔板数 × 回流比)。
    乘积相对初始工况放大超过阈值时,该点大概率超时。
    此规则需要知道"哪些是塔板数、哪些是回流比",由 boundary_advisor agent
    或用户通过 cost_proxy_groups 提供;不提供时跳过此规则(不影响前两条)。

返回约定
--------
check_preflight() 返回 (passed: bool, reason: str)：
  passed=True  → 该点可提交 Aspen。
  passed=False → reason 说明被拦截的原因,调用方据此判 infeasible,不调 Aspen。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class CostProxyGroup:
    """
    计算量代理组：一组变量的乘积正比于 Aspen 单点计算量。

    典型:RADFRAC 塔的 {塔板数路径, 回流比路径},乘积越大单点越慢。

    Attributes
    ----------
    name:
        组名(日志用),如 "T1_cost"。
    var_paths:
        参与乘积的变量 Aspen 路径列表。
    max_ratio:
        当前乘积 / 初始乘积 的上限。超过则判为高计算量风险,拦截。
        默认 3.0(乘积放大 3 倍以上视为危险)。
    """
    name: str
    var_paths: list[str]
    max_ratio: float = 3.0


@dataclass
class PreflightConfig:
    """
    飞行前检查配置。

    Attributes
    ----------
    enabled:
        是否启用飞行前检查,默认 True。False 时 check_preflight 永远放行。
    max_deviation_factor:
        规则 1:候选值相对参考值的最大归一化偏离倍数,默认 None(不启用偏离检查)。
        设为正数 k 时:对每个变量,要求候选值落在
        [参考值 - k×scale, 参考值 + k×scale] 内,scale 取 max(|参考值|, 全局宽度×0.05)。
        参考值缺失的变量跳过此检查。
        建议值 3~5:既拦截荒谬工况,又给优化保留探索空间。
    cost_proxy_groups:
        规则 3:计算量代理组列表,默认空(不启用)。
        每组定义一组变量的乘积约束,需该组所有变量都有参考值才生效。
    check_var_dependencies:
        规则 2:是否检查 var_dependencies(如 进料板<总塔板),默认 True。
        实际依赖关系由调用方通过 var_dependencies 参数传入。
    """
    enabled: bool = True
    max_deviation_factor: float | None = None
    cost_proxy_groups: list[CostProxyGroup] = field(default_factory=list)
    check_var_dependencies: bool = True

    def __post_init__(self) -> None:
        if self.max_deviation_factor is not None and self.max_deviation_factor <= 0:
            raise ValueError(
                f"max_deviation_factor 必须为正数或 None,收到 {self.max_deviation_factor}"
            )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def check_preflight(
    design_vars: dict[str, Any],
    reference_values: dict[str, float],
    global_bounds: dict[str, tuple[float, float]],
    config: PreflightConfig,
    var_dependencies: dict[str, dict[str, str]] | None = None,
) -> tuple[bool, str]:
    """
    对一个候选点做飞行前检查。

    Parameters
    ----------
    design_vars:
        候选点的完整设计变量 {Aspen 路径: 值}(已含 derived/fixed 展开)。
    reference_values:
        参考工况值 {Aspen 路径: 值},通常是初始收敛解。
        某变量无参考值时,该变量的偏离检查被跳过(不拦截)。
    global_bounds:
        各变量全局搜索边界 {Aspen 路径: (lo, hi)},用于计算偏离 scale。
    config:
        飞行前检查配置。
    var_dependencies:
        变量依赖 {被约束路径: {"lt"/"le"/"gt"/"ge": 参照路径}}。

    Returns
    -------
    (passed, reason)
        passed=True  → 放行,reason 为空串。
        passed=False → 拦截,reason 说明原因。
    """
    if not config.enabled:
        return True, ""

    # ── 规则 2：变量依赖一致性 ───────────────────────────────────────────
    if config.check_var_dependencies and var_dependencies:
        ok, reason = _check_dependencies(design_vars, var_dependencies)
        if not ok:
            return False, reason

    # ── 规则 1：偏离已知可行解 ───────────────────────────────────────────
    if config.max_deviation_factor is not None:
        ok, reason = _check_deviation(
            design_vars, reference_values, global_bounds, config.max_deviation_factor
        )
        if not ok:
            return False, reason

    # ── 规则 3：计算量代理指标 ───────────────────────────────────────────
    for group in config.cost_proxy_groups:
        ok, reason = _check_cost_proxy(design_vars, reference_values, group)
        if not ok:
            return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# 各规则实现
# ---------------------------------------------------------------------------

def _check_dependencies(
    design_vars: dict[str, Any],
    var_dependencies: dict[str, dict[str, str]],
) -> tuple[bool, str]:
    """规则 2:检查 lt/le/gt/ge 依赖关系。"""
    _OPS = {
        "lt": (lambda a, b: a < b, "<"),
        "le": (lambda a, b: a <= b, "<="),
        "gt": (lambda a, b: a > b, ">"),
        "ge": (lambda a, b: a >= b, ">="),
    }
    for var_path, dep in var_dependencies.items():
        if var_path not in design_vars:
            continue
        for op_name, ref_path in dep.items():
            fn_pair = _OPS.get(op_name)
            if fn_pair is None or ref_path not in design_vars:
                continue
            fn, sym = fn_pair
            try:
                a = float(design_vars[var_path])
                b = float(design_vars[ref_path])
            except (TypeError, ValueError):
                continue
            if not fn(a, b):
                return False, (
                    f"变量依赖违反：{_short(var_path)}={a:g} 不满足 {sym} "
                    f"{_short(ref_path)}={b:g}"
                )
    return True, ""


def _check_deviation(
    design_vars: dict[str, Any],
    reference_values: dict[str, float],
    global_bounds: dict[str, tuple[float, float]],
    k: float,
) -> tuple[bool, str]:
    """规则 1:候选值相对参考值的归一化偏离不得超过 k。"""
    for path, ref in reference_values.items():
        if path not in design_vars:
            continue
        try:
            val = float(design_vars[path])
            ref_f = float(ref)
        except (TypeError, ValueError):
            continue

        # scale:用参考值的量级;参考值接近 0 时用全局宽度兜底,避免 scale=0
        bounds = global_bounds.get(path)
        width = (bounds[1] - bounds[0]) if bounds else abs(ref_f)
        scale = max(abs(ref_f), width * 0.05)
        if scale <= 0:
            continue

        deviation = abs(val - ref_f) / scale
        if deviation > k:
            return False, (
                f"偏离过大：{_short(path)}={val:g} 相对参考值 {ref_f:g} "
                f"偏离 {deviation:.1f}× > 上限 {k:g}×(scale={scale:g})"
            )
    return True, ""


def _check_cost_proxy(
    design_vars: dict[str, Any],
    reference_values: dict[str, float],
    group: CostProxyGroup,
) -> tuple[bool, str]:
    """规则 3:变量乘积相对初始乘积的放大倍数不得超过 max_ratio。"""
    # 需要组内所有变量都有当前值和参考值
    try:
        cur_prod = 1.0
        ref_prod = 1.0
        for p in group.var_paths:
            if p not in design_vars or p not in reference_values:
                return True, ""  # 信息不全,跳过此组(不拦截)
            cur_prod *= abs(float(design_vars[p]))
            ref_prod *= abs(float(reference_values[p]))
    except (TypeError, ValueError):
        return True, ""

    if ref_prod <= 0:
        return True, ""

    ratio = cur_prod / ref_prod
    if ratio > group.max_ratio:
        return False, (
            f"计算量代理超限[{group.name}]：当前乘积 {cur_prod:g} / 初始乘积 "
            f"{ref_prod:g} = {ratio:.1f}× > 上限 {group.max_ratio:g}×,"
            f"该点大概率仿真超时,已拦截"
        )
    return True, ""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _short(path: str) -> str:
    """截取 Aspen 路径末尾有区分度的片段用于日志。"""
    parts = str(path).replace("\\", "/").split("/")
    _SKIP = {"Data", "Blocks", "Streams", "Input", "Output", "TOTFLOW", ""}
    meaningful = [p for p in parts if p not in _SKIP]
    if len(meaningful) >= 2:
        return "/".join(meaningful[-2:])
    if meaningful:
        return meaningful[-1]
    return str(path)
