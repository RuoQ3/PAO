"""
共享工作流工具模块。

本模块存放多个工作流模块共用的辅助函数，避免循环导入。
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def repair_design_vars(
    design_vars: dict[str, Any],
    integer_paths: set[str],
    param_bounds: dict[str, tuple[float, float]],
    var_dependencies: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    """
    对设计变量进行 round/clamp/repair 处理。

    处理顺序：
    1. integer 变量 round 到最近整数
    2. 所有变量 clamp 到 [lower, upper]
    3. 按 var_dependencies 修复依赖约束（如 FEED_STAGE < NSTAGE）

    Parameters
    ----------
    design_vars:
        原始设计变量 {Aspen 路径: 值}。
    integer_paths:
        需要 round 的路径集合。
    param_bounds:
        各变量的 [lower, upper] 边界。
    var_dependencies:
        变量依赖规则 {var_path: {"lt": other_path, "le": other_path}}。
        "lt" 表示 var_path 的值必须严格小于 other_path 的值。
        "le" 表示 var_path 的值必须 <= other_path 的值。

    Returns
    -------
    (repaired_vars, repair_notes)
        repaired_vars: 修复后的设计变量字典（新对象，不修改原始输入）。
        repair_notes: 修复操作说明列表（空列表表示无需修复）。
    """
    repaired = dict(design_vars)
    notes: list[str] = []

    # Step 1: round integer vars
    for path in integer_paths:
        if path in repaired and repaired[path] is not None:
            orig = repaired[path]
            rounded = int(round(float(orig)))
            if rounded != orig:
                repaired[path] = rounded
                notes.append(f"round {path.split(chr(92))[-1]}: {orig!r} → {rounded}")

    # Step 2: clamp to bounds
    # integer 变量的 lo/hi 在 _validate_config 中已保证为整数值，int() 转换安全。
    for path, (lo, hi) in param_bounds.items():
        if path not in repaired or repaired[path] is None:
            continue
        val = float(repaired[path])
        if val < lo:
            new_val = int(lo) if path in integer_paths else lo
            repaired[path] = new_val
            notes.append(f"clamp {path.split(chr(92))[-1]}: {val!r} → {new_val} (lo={lo})")
        elif val > hi:
            new_val = int(hi) if path in integer_paths else hi
            repaired[path] = new_val
            notes.append(f"clamp {path.split(chr(92))[-1]}: {val!r} → {new_val} (hi={hi})")

    # Step 3: enforce var_dependencies
    for var_path, rules in var_dependencies.items():
        if var_path not in repaired or repaired[var_path] is None:
            continue
        for op, other_path in rules.items():
            if other_path not in repaired or repaired[other_path] is None:
                continue
            var_val   = repaired[var_path]
            other_val = repaired[other_path]
            try:
                var_f   = float(var_val)
                other_f = float(other_val)
            except (TypeError, ValueError):
                continue

            violated = (op == "lt" and not (var_f < other_f)) or \
                       (op == "le" and not (var_f <= other_f))
            if not violated:
                continue

            lo = param_bounds.get(var_path, (1.0, other_f))[0]
            if op == "lt":
                new_f = max(lo, other_f - 1.0)
            else:
                new_f = max(lo, other_f)
            new_val = int(new_f) if var_path in integer_paths else new_f
            repaired[var_path] = new_val
            notes.append(
                f"dep {var_path.split(chr(92))[-1]} {op} "
                f"{other_path.split(chr(92))[-1]}: {var_f!r} → {new_val}"
            )

    return repaired, notes


def apply_derived_vars(
    design_vars: dict[str, Any],
    derived_specs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """
    Expand virtual derived variables into real Aspen input variables.

    A derived spec maps a virtual optimizer variable, usually a continuous
    fraction, to a real Aspen tree node. The virtual variable is removed from
    the returned dict so callers can pass the result directly to run_case().
    """
    if not derived_specs:
        return dict(design_vars), []

    expanded = dict(design_vars)
    notes: list[str] = []

    for spec in derived_specs:
        frac_path = str(spec["frac_path"])
        target_path = str(spec["target_path"])
        depends_on = str(spec["depends_on"])
        frac_lo = int(spec.get("frac_lo", 1))

        if frac_path not in expanded:
            continue

        raw_frac = expanded.pop(frac_path)
        if raw_frac is None:
            raise ValueError(f"derived variable '{frac_path}' is None")
        if depends_on not in expanded or expanded[depends_on] is None:
            raise ValueError(
                f"derived variable '{frac_path}' requires dependency "
                f"'{depends_on}' in design_vars"
            )

        frac = float(raw_frac)
        nstage = int(round(float(expanded[depends_on])))
        upper = nstage - 1
        if upper < frac_lo:
            raise ValueError(
                f"derived variable '{frac_path}' has invalid bounds: "
                f"frac_lo={frac_lo}, dependency {depends_on}={nstage}"
            )

        mapped = int(round(frac_lo + frac * (nstage - frac_lo - 1)))
        clamped = max(frac_lo, min(upper, mapped))
        expanded[target_path] = clamped

        note = (
            f"derived {_short_name(frac_path)}={frac:.6g} + "
            f"{_short_name(depends_on)}={nstage} -> "
            f"{_short_name(target_path)}={clamped}"
        )
        if clamped != mapped:
            note += f" (clamped from {mapped})"
        notes.append(note)

    return expanded, notes


def _short_name(path: str) -> str:
    return path.rsplit("\\", 1)[-1]
