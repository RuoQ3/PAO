"""
共享工作流工具模块。

本模块存放多个工作流模块共用的辅助函数，避免循环导入。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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


def feasibility_feature_names(
    param_paths: list[str],
    derived_specs: list[dict[str, Any]],
) -> list[str]:
    """
    Return feature names that are present in persisted Aspen input records.

    Optimizer-space derived variables use a virtual ``frac_path`` such as
    ``T1_FEED_F1_FRAC``. ``apply_derived_vars`` removes that virtual key before
    calling Aspen and stores the real ``target_path`` in ProcessCase.design_vars.
    The feasibility classifier trains on ProcessCase.design_vars, so it must use
    the real target path for derived variables.
    """
    derived_targets = {
        str(spec["frac_path"]): str(spec["target_path"])
        for spec in derived_specs
        if "frac_path" in spec and "target_path" in spec
    }
    return [derived_targets.get(path, path) for path in param_paths]


def _short_name(path: str) -> str:
    return path.rsplit("\\", 1)[-1]


# ---------------------------------------------------------------------------
# 早停配置
# ---------------------------------------------------------------------------

@dataclass
class EarlyStoppingConfig:
    """
    贝叶斯优化早停配置。

    Attributes
    ----------
    enabled:
        是否启用早停。False（默认）时完全不影响旧流程。
    min_iterations:
        至少完成多少次总迭代（含 DOE）后才允许触发早停，默认 0。
    patience:
        连续多少轮无有效改善后停止，默认 10。
    min_delta:
        绝对改善阈值；改善量 < min_delta 不算有效改善，默认 0.0。
    relative_delta:
        相对改善阈值；改善比例 < relative_delta 不算有效改善。None 表示不检查，默认 None。
    max_duplicate_suggestions:
        候选池连续选到重复候选多少次后触发 early stop，默认 3。
    check_hypervolume:
        多目标是否用 hypervolume 改善作为判断依据，默认 True。
    check_first_front:
        多目标是否检查 Pareto 第一前沿变化，默认 True。
    """
    enabled: bool = False
    min_iterations: int = 0
    patience: int = 10
    min_delta: float = 0.0
    relative_delta: float | None = None
    max_duplicate_suggestions: int = 3
    check_hypervolume: bool = True
    check_first_front: bool = True

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(
                f"EarlyStoppingConfig.patience={self.patience} 必须 >= 1。"
            )
        if self.min_iterations < 0:
            raise ValueError(
                f"EarlyStoppingConfig.min_iterations={self.min_iterations} 必须 >= 0。"
            )
        if self.min_delta < 0.0:
            raise ValueError(
                f"EarlyStoppingConfig.min_delta={self.min_delta} 必须 >= 0。"
            )
        if self.relative_delta is not None and self.relative_delta < 0.0:
            raise ValueError(
                f"EarlyStoppingConfig.relative_delta={self.relative_delta} 必须 >= 0。"
            )
        if self.max_duplicate_suggestions < 1:
            raise ValueError(
                f"EarlyStoppingConfig.max_duplicate_suggestions="
                f"{self.max_duplicate_suggestions} 必须 >= 1。"
            )


# ---------------------------------------------------------------------------
# 候选去重工具
# ---------------------------------------------------------------------------

def fingerprint_design_vars(
    design_vars: dict[str, Any],
    ndigits: int = 8,
) -> tuple:
    """
    为设计变量 dict 生成稳定的去重 fingerprint。

    浮点值 round 到 ndigits 位，整数直接保留，其余 str() 转换。
    返回按键排序的 (key, value) 元组，可用作 set / frozenset 元素。

    Parameters
    ----------
    design_vars:
        ProcessCase.design_vars 或 full_candidates 中的 design_vars dict。
    ndigits:
        浮点值的保留精度，默认 8 位，足以区分 Aspen 参数空间中的不同点，
        同时过滤掉 repair/derive 引入的浮点噪声。
    """
    result: list[tuple[str, Any]] = []
    for k in sorted(design_vars.keys()):
        v = design_vars[k]
        if isinstance(v, float):
            v = round(v, ndigits)
        elif isinstance(v, int):
            pass  # 整数不需要 round
        else:
            v = str(v)
        result.append((k, v))
    return tuple(result)


def build_evaluated_set(
    cases: list[Any],  # list[ProcessCase]
) -> set[tuple]:
    """
    从历史 ProcessCase 列表中构造已评估过的 fingerprint 集合。

    只包含 design_vars 非空的工况（PENDING 或空 design_vars 跳过）。
    """
    seen: set[tuple] = set()
    for c in cases:
        if c.design_vars:
            seen.add(fingerprint_design_vars(c.design_vars))
    return seen


def pick_first_unseen_candidate(
    full_candidates: list[dict[str, Any]],
    screened: list[dict[str, Any]],
    evaluated_fps: set[tuple],
) -> dict[str, Any] | None:
    """
    从 screened 候选（按可行概率降序）中找第一个未评估的候选。

    若 screened 全部重复，则遍历 full_candidates 找第一个未评估的。
    若 full_candidates 也全部重复，返回 None。

    Parameters
    ----------
    full_candidates:
        repair/derive 后的完整候选信息列表。
    screened:
        clf.screen() 返回的列表（含 __candidate_index 和 _predicted_feasible）。
    evaluated_fps:
        已评估过的 fingerprint 集合。
    """
    # 优先从 screened 中选（保留可行概率排序）
    for entry in screened:
        idx = int(entry.get("__candidate_index", 0))
        cand = full_candidates[idx]
        fp = fingerprint_design_vars(cand["design_vars"])
        if fp not in evaluated_fps:
            return cand

    # screened 全部重复，遍历 full_candidates（按索引顺序）
    for cand in full_candidates:
        fp = fingerprint_design_vars(cand["design_vars"])
        if fp not in evaluated_fps:
            return cand

    return None  # 候选池全部重复
