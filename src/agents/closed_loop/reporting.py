"""Deterministic, human-readable explanations for a closed-loop run.

The checkpoint is the source of truth for the audit trail.  This module only
renders fields that are already persisted in that checkpoint: the declared
ActionPlan hypothesis, evidence references, deterministic before/after
analysis and observed Aspen outcomes.  It deliberately does not attempt to
reconstruct or claim an LLM's hidden chain of thought.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


EXPLANATORY_REPORT_VERSION = 2


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    number = _number(value)
    if number is not None and isinstance(value, (int, float)):
        if percent:
            return f"{number * 100:.2f}%"
        if abs(number) >= 1000 or (0 < abs(number) < 1e-4):
            return f"{number:.4e}"
        return f"{number:.6g}"
    return str(value)


def _cell(value: Any) -> str:
    """Make arbitrary checkpoint text safe for a Markdown table cell."""
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    if not rows:
        return ["暂无数据。"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return lines


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


def _status_counts(observations: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("status", "unknown")) for row in observations)


def _objective_values(row: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in row.get("objectives") or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("available", item.get("error") is None):
            values[str(item.get("name", "objective"))] = item.get("value")
    return values


def _constraint_values(row: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in row.get("constraints") or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("available", item.get("error") is None):
            values[str(item.get("name", "constraint"))] = {
                "value": item.get("value"),
                "satisfied": item.get("satisfied"),
            }
    return values


def _case_description(row: Mapping[str, Any], parameter_keys: Sequence[str] = ()) -> str:
    case_id = str(row.get("case_id", "unknown"))
    status = str(row.get("status", "unknown"))
    objectives = ", ".join(f"{name}={_fmt(value)}" for name, value in _objective_values(row).items())
    constraints = []
    for name, item in _constraint_values(row).items():
        constraints.append(f"{name}={_fmt(item.get('value'))} ({'满足' if item.get('satisfied') else '违反'})")
    details = "; ".join(part for part in (objectives, ", ".join(constraints)) if part)
    parameters = {
        str(key): (row.get("optimizer_inputs") or {}).get(key)
        for key in parameter_keys
        if key in (row.get("optimizer_inputs") or {})
    }
    if parameters:
        details = "; ".join(part for part in (details, f"输入={_json(parameters)}") if part)
    if not details:
        details = str((row.get("sim_result") or {}).get("error") or row.get("notes") or "无目标/约束摘要")
    return f"`{case_id}`：状态={status}；{details}"


def _compact_evidence(
    ids: Sequence[Any],
    cases: Mapping[str, Mapping[str, Any]],
    *,
    missing_label: str,
    parameter_keys: Sequence[str] = (),
) -> list[str]:
    lines: list[str] = []
    for raw_id in ids:
        case_id = str(raw_id)
        row = cases.get(case_id)
        lines.append(_case_description(row, parameter_keys) if row else f"`{case_id}`：{missing_label}")
    return lines


def _latest_persisted_analysis(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for decision in reversed(state.get("decisions") or []):
        outcome = decision.get("outcome") or {}
        report = outcome.get("after_analysis")
        if isinstance(report, Mapping):
            return report
    report = state.get("analysis_report")
    return report if isinstance(report, Mapping) else None


def _latest_analysis(
    state: Mapping[str, Any],
    objective_names: Sequence[str],
) -> Mapping[str, Any] | None:
    """Prefer a fresh all-observation summary, including verification cases.

    ``after_analysis`` is intentionally attached to the action that caused it;
    a later independent verification can therefore be absent from that action's
    historical snapshot.  Rebuilding here keeps the final Markdown summary in
    sync with the complete checkpoint while preserving the historical snapshots
    in ``.report.json``.
    """
    observations = [row for row in (state.get("observations") or []) if isinstance(row, Mapping)]
    if observations and objective_names:
        try:
            from src.agents.closed_loop.analysis import build_analysis_from_rows

            persisted = _latest_persisted_analysis(state) or {}
            report = build_analysis_from_rows(
                observations,
                objective_names=list(objective_names),
                param_paths=list((state.get("problem") or {}).get("hard_bounds") or {}),
                recent_window=max(1, int(persisted.get("window_size", 12))),
            )
            history = state.get("hv_history") or []
            if history and history[-1] is not None:
                report["metrics"]["hypervolume"] = history[-1]
                report["pareto"]["hypervolume"] = history[-1]
            return report
        except Exception:
            # A legacy or partially written checkpoint should still render the
            # persisted action-level evidence instead of failing report output.
            pass
    return _latest_persisted_analysis(state)


def _metric_rows(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[list[str]]:
    names = (
        ("hypervolume", "超体积（HV）"),
        ("feasible_rate", "近期可行率"),
        ("convergence_rate", "近期收敛率"),
        ("feasible_count", "累计可行工况数"),
        ("constraint_violation_rate", "约束违反率"),
    )
    rows = []
    for key, label in names:
        left, right = before.get(key), after.get(key)
        if left is None and right is None:
            continue
        delta = None
        if _number(left) is not None and _number(right) is not None:
            delta = _number(right) - _number(left)
        rows.append([label, _fmt(left, percent="rate" in key), _fmt(right, percent="rate" in key), _fmt(delta)])
    return rows


def _stop_explanation(reason: Any, state: Mapping[str, Any], settings: Any) -> str:
    reason = str(reason or "unknown")
    used = state.get("used", 0)
    maximum = _get(settings, "max_evaluations", "未知")
    explanations = {
        "target_observed": "已观察到满足目标阈值的可行工况，流程转入独立复验；只有复验通过才会报告 target_verified。",
        "evaluation_budget": f"达到总评估预算（{used}/{maximum}）；预算包含失败、预检查拦截和复验，因此没有继续探索。",
        "decision_budget": "达到主决策次数上限；系统停止继续向 Agent 请求动作，并对代表可行工况进行复验。",
        "stagnation": "连续若干个已计数动作未带来新的可行性、收敛性或 Pareto/HV 改善，达到停滞耐心上限。",
        "agent_stop": "主决策 Agent 明确提出 stop；程序仍按协议尝试对代表可行工况进行复验。",
        "driver_unavailable": "Aspen 驱动不可用或恢复失败，流程被安全终止，结果不能视为已复验。",
    }
    explanation = explanations.get(reason, "流程以未分类原因结束；请结合 checkpoint 中的 pending、决策和观测记录复核。")
    no_case_actions = sum(
        not (decision.get("outcome") or {}).get("case_ids")
        for decision in state.get("decisions") or []
    )
    if no_case_actions:
        explanation += f" 本次有 {no_case_actions} 个决策没有形成新的有效工况；这属于搜索流程事件，不能当作 Aspen 实验结果解读。"
    return explanation


def _render_final_evidence(analysis: Mapping[str, Any] | None) -> list[str]:
    lines = ["## 最终证据摘要", ""]
    if not analysis:
        lines.append("当前 checkpoint 没有可用的 `after_analysis`；只能依据观测状态和动作审计字段复核。")
        return lines + [""]

    quality = analysis.get("data_quality") or {}
    convergence = analysis.get("convergence") or {}
    metrics = analysis.get("metrics") or {}
    lines.append("这些指标由确定性分析代码从已保存的 ProcessCase 计算，不是 Agent 自己编造的结论。")
    lines.append("")
    lines.extend(_table(
        ["指标", "值"],
        [
            ["总观测数", quality.get("n_total")],
            ["仿真有效数", quality.get("n_simulation_valid")],
            ["可行工况数", quality.get("n_success")],
            ["近期收敛率", _fmt(convergence.get("recent_rate"), percent=True)],
            ["近期可行率", _fmt(metrics.get("feasible_rate"), percent=True)],
            ["最终超体积（HV）", metrics.get("hypervolume")],
        ],
    ))
    lines.append("")

    objectives = analysis.get("objectives") or {}
    if objectives:
        lines.append("### 目标变化")
        lines.extend(_table(
            ["目标", "方向", "可用样本", "首次值", "最新值", "最好值", "最好工况", "相对首次改善"],
            [
                [
                    name,
                    item.get("direction"),
                    item.get("n_available"),
                    item.get("first_value"),
                    item.get("latest_value"),
                    item.get("best_value"),
                    item.get("best_case_id"),
                    item.get("best_improvement_from_first"),
                ]
                for name, item in objectives.items()
            ],
        ))
        lines.append("")

    constraints = analysis.get("constraints") or {}
    if constraints:
        lines.append("### 约束证据")
        lines.extend(_table(
            ["约束", "可用样本", "违反率", "最大违反", "最小裕量", "最差工况"],
            [
                [
                    name,
                    item.get("n_available"),
                    _fmt(item.get("violation_rate"), percent=True),
                    item.get("max_violation"),
                    item.get("min_margin"),
                    item.get("worst_case_id"),
                ]
                for name, item in constraints.items()
            ],
        ))
        lines.append("")

    pareto = analysis.get("pareto") or {}
    lines.append("### Pareto 与敏感性")
    lines.extend(_table(
        ["证据", "值"],
        [
            ["Pareto 前沿工况数", pareto.get("front_size")],
            ["前沿工况 ID", ", ".join(map(str, pareto.get("front_case_ids") or [])) or "—"],
            ["超体积参考点", _json(pareto.get("reference_point")) if pareto.get("reference_point") is not None else "—"],
            ["敏感性样本数", (analysis.get("sensitivity") or {}).get("n_samples")],
        ],
    ))
    sensitivity = analysis.get("sensitivity") or {}
    ranked = sensitivity.get("ranked_variables") or []
    if ranked:
        lines.append("")
        lines.append("敏感性排序（仅作为相关性线索；`reliable=false` 表示样本不足或不稳定）：")
        lines.extend(_table(
            ["变量", "score", "有效样本", "可靠"],
            [[item.get("path"), item.get("score"), item.get("effective_samples"), item.get("reliable")] for item in ranked[:8]],
        ))
    warnings = list(sensitivity.get("warnings") or [])
    if warnings:
        lines.append("")
        lines.append("敏感性分析警告：" + "；".join(str(item) for item in warnings[:5]))

    failures = analysis.get("failures") or {}
    diagnoses = failures.get("diagnoses") or []
    if diagnoses or failures.get("examples"):
        lines.append("")
        lines.append("失败证据：")
        for diagnosis in diagnoses[:5]:
            lines.append(
                f"- `{diagnosis.get('pattern_id', 'unknown')}`（{diagnosis.get('severity', 'unknown')}）："
                f"{diagnosis.get('description', '未提供描述')}；建议：{', '.join(map(str, diagnosis.get('fixes') or []))}"
            )
        for example in (failures.get("examples") or [])[:3]:
            lines.append(f"- 原始错误样例：{example}")
    lines.append("")
    return lines


def _render_decision(
    index: int,
    decision: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    plan = decision.get("plan") or {}
    outcome = decision.get("outcome") or {}
    source = decision.get("source", "unknown")
    action = plan.get("action", "unknown")
    lines = [f"### 决策 {index}：{action}（来源：{source}）", ""]
    lines.append(f"**动作假设：** {plan.get('hypothesis') or '未提供可审计假设。'}")
    lines.append(f"**期望作用：** `{plan.get('expected_effect', '—')}`；批次大小={plan.get('batch_size', '—')}；初始化={plan.get('initialization', '—')}；重复运行={_fmt(plan.get('repeat'))}")

    evidence_ids = plan.get("evidence_case_ids") or []
    parameter_keys = sorted(set(plan.get("candidate") or {}) | set(plan.get("search_region") or {}))
    lines.append("")
    lines.append("**动作依据（Aspen 历史证据）：**")
    if evidence_ids:
        lines.extend(
            f"- {item}"
            for item in _compact_evidence(
                evidence_ids,
                cases,
                missing_label="checkpoint 中找不到该工况",
                parameter_keys=parameter_keys,
            )
        )
    else:
        lines.append("- 未引用已有工况（通常只允许发生在初始基准动作）。")
    knowledge_ids = plan.get("knowledge_ids") or []
    lines.append(f"**知识依据 ID：** {', '.join(map(str, knowledge_ids)) if knowledge_ids else '无'}")

    changes = {}
    if plan.get("search_region"):
        changes["search_region"] = plan["search_region"]
    if plan.get("candidate"):
        changes["candidate"] = plan["candidate"]
    lines.append("")
    lines.append("**计划参数变化：**")
    if changes:
        lines.append("```json")
        lines.append(json.dumps(changes, ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
    else:
        lines.append("- 本轮没有直接指定参数映射，由搜索会话在当前软区域内生成候选。")

    rejection = decision.get("rejection")
    execution_rejection = decision.get("execution_rejection")
    if rejection:
        lines.append(f"**协议处理：** Agent 提案被拒绝：`{rejection}`；随后使用了确定性规则动作。")
    if execution_rejection:
        lines.append(f"**执行处理：** `{execution_rejection}`。这表示没有找到可执行的新候选，不表示 Aspen 运行失败。")

    case_ids = [str(item) for item in outcome.get("case_ids") or []]
    lines.append("")
    lines.append("**本轮实际结果：**")
    if case_ids:
        lines.extend(
            f"- {_case_description(cases[case_id], parameter_keys)}"
            if case_id in cases
            else f"- `{case_id}`：checkpoint 中找不到该工况"
            for case_id in case_ids
        )
    else:
        lines.append("- 没有新的执行工况；因此本轮没有 Aspen 输出可以支持参数效果结论。")

    before, after = decision.get("before") or {}, outcome.get("after") or {}
    if before or after:
        lines.append("")
        lines.append("**动作前后指标：**")
        lines.extend(_table(["指标", "动作前", "动作后", "变化（后-前）"], _metric_rows(before, after)))
    effect = outcome.get("analysis_effect") or {}
    if effect:
        lines.append("")
        lines.append("**确定性分析差异：**")
        effect_rows = [
            ["超体积（HV）", effect.get("hypervolume_delta")],
            ["可行率", _fmt(effect.get("feasible_rate_delta"), percent=True)],
            ["收敛率", _fmt(effect.get("convergence_rate_delta"), percent=True)],
            ["可行工况数", effect.get("feasible_count_delta")],
            ["期望指标是否改善", effect.get("expected_metric_improved")],
        ]
        lines.extend(_table(["指标", "变化"], effect_rows))
        if effect.get("objective_deltas"):
            lines.append("目标方向归一化改善：" + _json(effect["objective_deltas"]))
        lines.append(f"解释边界：{effect.get('interpretation', 'Observed association only; not proof of causal attribution')}")
    elif outcome.get("interpretation"):
        lines.append("")
        lines.append(f"解释边界：{outcome['interpretation']}")

    if "stagnation_counted" in outcome:
        lines.append(
            f"**停滞计数：** {'计入' if outcome.get('stagnation_counted') else '未计入'}；"
            f"当前累计={outcome.get('stagnation_count', '—')}。"
        )
    lines.append("")
    return lines


def _parameter_metadata(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Map persisted optimizer keys to the names an engineer recognizes."""
    context = (state.get("problem") or {}).get("context") or {}
    metadata: dict[str, dict[str, Any]] = {}
    for item in context.get("design_variables") or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("aspen_path") or "")
        key = name if item.get("type") == "derived" else str(item.get("aspen_path") or name)
        if key:
            metadata[key] = dict(item)
    return metadata


def _parameter_label(key: Any, metadata: Mapping[str, Mapping[str, Any]]) -> str:
    raw = str(key)
    item = metadata.get(raw) or {}
    name = str(item.get("name") or raw)
    if name.endswith("_NSTAGE"):
        return name.replace("_NSTAGE", " 理论板数")
    if name.endswith("_RR"):
        return name.replace("_RR", " 回流比")
    if name.endswith("_PRES"):
        return name.replace("_PRES", " 操作压力")
    if name.startswith("SOL") and name.endswith("_FLOW"):
        return name.replace("SOL", "S").replace("_FLOW", " 溶剂流量")
    if "_FEED_" in name and name.endswith("_FRAC"):
        prefix, feed = name.split("_FEED_", 1)
        return f"{prefix} {feed.removesuffix('_FRAC')} 进料位置"
    return name.replace("_", " ")


def _same_value(left: Any, right: Any) -> bool:
    left_number, right_number = _number(left), _number(right)
    if left_number is None or right_number is None:
        return left == right
    return math.isclose(left_number, right_number, rel_tol=1e-8, abs_tol=1e-10)


def _reference_inputs(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    for case_id in reversed(plan.get("evidence_case_ids") or []):
        row = cases.get(str(case_id))
        if row and row.get("optimizer_inputs"):
            return row["optimizer_inputs"]
    return state.get("initial_point") or {}


def _planned_changes(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    limit: int = 6,
) -> list[str]:
    """Summarize only changed variables, never dump a full candidate vector."""
    reference = _reference_inputs(plan, state, cases)
    candidate = plan.get("candidate") or {}
    changes: list[str] = []
    for key, value in candidate.items():
        old = reference.get(key) if isinstance(reference, Mapping) else None
        if old is not None and _same_value(old, value):
            continue
        label = _parameter_label(key, metadata)
        if old is None:
            changes.append(f"{label}={_fmt(value)}")
        else:
            changes.append(f"{label} {_fmt(old)} → {_fmt(value)}")
    if not changes:
        for key, pair in (plan.get("search_region") or {}).items():
            label = _parameter_label(key, metadata)
            changes.append(f"{label} 搜索区 [{_fmt(pair[0])}, {_fmt(pair[1])}]")
    return changes[:limit]


def _row_conclusion(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", "unknown"))
    if row.get("success"):
        return "可行"
    if status == "infeasible":
        return "已收敛，但违反产品约束"
    if row.get("simulation_valid"):
        return "已收敛，但目标/约束结果不可用"
    return "仿真失败或结果未知"


def _decision_outcome(
    decision: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
) -> tuple[str, int, int, int]:
    ids = [str(item) for item in (decision.get("outcome") or {}).get("case_ids") or []]
    rows = [cases[item] for item in ids if item in cases]
    feasible = sum(bool(row.get("success")) for row in rows)
    infeasible = sum(str(row.get("status")) == "infeasible" for row in rows)
    failed = len(rows) - feasible - infeasible
    if not rows:
        return "未形成新的 Aspen 工况证据", 0, 0, 0
    if feasible and not infeasible and not failed:
        return f"获得 {feasible} 个可行工况", feasible, infeasible, failed
    if infeasible and not feasible and not failed:
        return f"{infeasible} 个工况收敛但违反约束", feasible, infeasible, failed
    return f"可行 {feasible}，约束违反 {infeasible}，失败/无效 {failed}", feasible, infeasible, failed


def _feasible_rows(observations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in observations if row.get("success")]


def _objective_from_row(row: Mapping[str, Any], name: str) -> float | None:
    value = _objective_values(row).get(name)
    return _number(value)


def _recommended_row(
    state: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    objective_names: Sequence[str],
    analysis: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    cases = {str(row.get("case_id")): row for row in observations}
    reference = cases.get(str(state.get("verification_reference")))
    if reference and reference.get("success"):
        return reference
    feasible = _feasible_rows(observations)
    if not feasible:
        return None
    directions = {
        name: ((analysis or {}).get("objectives") or {}).get(name, {}).get("direction", "minimize")
        for name in objective_names
    }
    ranges: dict[str, tuple[float, float]] = {}
    for name in objective_names:
        values = [v for row in feasible if (v := _objective_from_row(row, name)) is not None]
        if values:
            ranges[name] = (min(values), max(values))

    def score(row: Mapping[str, Any]) -> float:
        total = 0.0
        count = 0
        for name in objective_names:
            value = _objective_from_row(row, name)
            if value is None or name not in ranges:
                continue
            lo, hi = ranges[name]
            normalized = (value - lo) / max(hi - lo, 1e-12)
            if directions[name] != "minimize":
                normalized = 1.0 - normalized
            total += normalized
            count += 1
        return total / max(count, 1)

    return min(feasible, key=score)


def _constraint_label(name: str) -> str:
    replacements = {
        "D1_PF_purity": "D1 PF 纯度",
        "D2_NPA_purity": "D2 NPA 纯度",
        "D3_WATER_purity": "D3 WATER 纯度",
    }
    return replacements.get(name, name.replace("_purity", " 纯度").replace("_", " "))


def _objective_improvement(item: Mapping[str, Any]) -> str:
    first = _number(item.get("first_value"))
    best = _number(item.get("best_value"))
    if first is None or best is None or first == 0:
        return "—"
    improvement = (first - best) / abs(first) if item.get("direction") == "minimize" else (best - first) / abs(first)
    return _fmt(improvement, percent=True)


def _best_feasible_value(
    observations: Sequence[Mapping[str, Any]],
    name: str,
    direction: str | None,
) -> float | None:
    values = [
        value
        for row in observations
        if row.get("success")
        and (value := _objective_from_row(row, name)) is not None
    ]
    if not values:
        return None
    return min(values) if direction != "maximize" else max(values)


def _unique_input_count(rows: Sequence[Mapping[str, Any]]) -> int:
    fingerprints = {
        _json(row.get("optimizer_inputs") or {})
        for row in rows
        if row.get("optimizer_inputs") is not None
    }
    return len(fingerprints)


def _status_sentence(result: str, termination: str, targets: Mapping[str, Any]) -> str:
    if result == "target_verified":
        return "指定目标已在代表工况上达到，并通过独立复验。"
    if result == "completed_verified":
        if targets:
            return "代表工况通过独立复验，但报告不把它扩大解释为整个 Pareto 前沿或全局最优。"
        return "代表工况通过独立复验；当前配置未设置数值目标阈值，因此这不是‘目标已达成’的结论。"
    if result == "verification_failed":
        return "代表工况未通过独立复验，当前结果不能作为稳定操作点交付。"
    if result == "no_feasible_solution":
        return "本轮预算内没有找到可行工况，下一轮应优先扩大可行域探索。"
    if termination == "stagnation":
        return "搜索因停滞提前结束，说明当前搜索策略没有继续产生有效改进，不等于不存在更优解。"
    return "本轮未形成可交付的稳定结论。"


def _termination_label(reason: str) -> str:
    return {
        "target_observed": "已观察到目标工况",
        "evaluation_budget": "评估预算用尽",
        "decision_budget": "决策次数用尽",
        "stagnation": "搜索停滞",
        "agent_stop": "Agent 主动停止",
        "driver_unavailable": "Aspen 驱动不可用",
    }.get(reason, reason or "未分类原因")


def _analyst_report(
    state: Mapping[str, Any],
    *,
    settings: Any,
    objective_names: Sequence[str],
) -> str:
    observations = [row for row in (state.get("observations") or []) if isinstance(row, Mapping)]
    cases = {str(row.get("case_id")): row for row in observations if row.get("case_id") is not None}
    decisions = [item for item in (state.get("decisions") or []) if isinstance(item, Mapping)]
    metadata = _parameter_metadata(state)
    analysis = _latest_analysis(state, objective_names)
    analysis = analysis or {}
    result = str(state.get("result", "unverified"))
    termination = str(state.get("termination_reason") or "unknown")
    targets = _get(settings, "objective_targets", {}) or {}
    feasible = _feasible_rows(observations)
    verified_count = len(state.get("verification_ids") or [])
    max_evaluations = _get(settings, "max_evaluations", "—")
    recommended = _recommended_row(state, observations, objective_names, analysis)

    objective_text = "、".join(objective_names) if objective_names else "当前配置目标"
    status_text = "、".join(f"{key} {value} 次" for key, value in sorted(_status_counts(observations).items())) or "无可用观测"
    lines = [
        "# PAO 工艺优化分析报告",
        "",
        f"> 报告版本 {EXPLANATORY_REPORT_VERSION}。本报告面向工艺决策：只保留结论、证据和建议。完整运行审计记录见同名 `.report.json`。",
        "",
        "## 一、执行摘要",
        "",
        f"本轮优化以**最小化 {objective_text}**为目标，同时满足全部产品纯度约束。"
        f"共完成 {len(observations)} 个 Aspen 工况评估，预算使用 {state.get('used', len(observations))}/{max_evaluations}；"
        f"其中 {sum(bool(row.get('simulation_valid')) for row in observations)} 个工况完成收敛，{len(feasible)} 个工况满足全部约束"
        f"（{_unique_input_count(feasible)} 个不同操作点）。",
        "",
        f"**总体结论：** {_status_sentence(result, termination, targets)}",
        "",
        f"本轮停止原因为“{_termination_label(termination)}”：{_stop_explanation(termination, state, settings)}",
        "",
        "## 二、已经完成的工作",
        "",
    ]
    work_items = [
        f"完成 {len(observations)} 次 Aspen 工况评估，状态分布为：{status_text}。",
        f"围绕已有可行锚点进行了 {sum((d.get('plan') or {}).get('action') == 'probe' for d in decisions)} 次定向参数试验，重点涉及回流比、溶剂流量和分离段操作条件。",
        f"完成 {verified_count} 次独立复验；复验只证明代表工况在当前模型下具有重复性，不证明全局最优。",
    ]
    rejected = sum(bool(item.get("rejection")) for item in decisions)
    if rejected:
        work_items.append(f"有 {rejected} 次自动动作建议未满足执行条件，系统已切换到安全规则继续运行；这些事件不应被解释为工艺实验结论。")
    lines.extend(f"- {item}" for item in work_items)
    lines.extend(["", "## 三、结果与工程含义", ""])

    objective_report = analysis.get("objectives") or {}
    if objective_report:
        lines.append("### 目标改善")
        objective_rows = []
        for name, item in objective_report.items():
            feasible_best = _best_feasible_value(observations, name, item.get("direction"))
            first = _number(item.get("first_value"))
            if feasible_best is None or first is None or first == 0:
                feasible_improvement = "—"
            else:
                change = (first - feasible_best) / abs(first) if item.get("direction") == "minimize" else (feasible_best - first) / abs(first)
                feasible_improvement = _fmt(change, percent=True)
            history_best = _number(item.get("best_value"))
            best_row = cases.get(str(item.get("best_case_id")))
            history_note = history_best
            if history_best is not None and best_row is not None and not best_row.get("success"):
                history_note = f"{_fmt(history_best)}（不可行）"
            objective_rows.append([name, first, feasible_best, feasible_improvement, history_note])
        lines.extend(_table(
            ["目标", "首次值", "最佳可行值", "可行改善", "全历史最低值"],
            objective_rows,
        ))
        lines.append("")
        lines.append("‘全历史最低值’若标注为不可行，只能说明经济目标有潜力，不能作为工艺候选；正式建议只采用同时满足全部产品约束的工况。")
        lines.append("目标值的改善只能说明本轮试验观察到了更优结果；由于可行点数量有限，不能单独归因于某一个参数，更不能据此宣称全局最优。")
        lines.append("")

    constraints = analysis.get("constraints") or {}
    if constraints:
        lines.append("### 当前主要瓶颈")
        ordered = sorted(constraints.items(), key=lambda item: item[1].get("violation_rate") or 0, reverse=True)
        rows = []
        for name, item in ordered[:5]:
            rows.append([
                _constraint_label(name),
                _fmt(item.get("violation_rate"), percent=True),
                item.get("n_satisfied"),
                item.get("max_violation"),
                "约束风险高" if (item.get("violation_rate") or 0) > 0.5 else "需要继续确认",
            ])
        lines.extend(_table(["约束", "违反率", "满足次数", "最大违反", "判断"], rows))
        top = ordered[0]
        lines.append("")
        if (top[1].get("violation_rate") or 0) > 0:
            lines.append(
                f"当前最主要的可行域瓶颈是 {_constraint_label(top[0])}（违反率 {_fmt(top[1].get('violation_rate'), percent=True)}）。"
                "因此下一阶段应先扩大该约束附近的可行裕量，再继续压低经济目标；单纯追求更低成本可能会把工况推回不可行区域。"
            )
        else:
            lines.append("本轮已评估工况均满足产品约束，暂未观察到明确的约束瓶颈；下一阶段可以把重点转向经济目标改善。")
        lines.append("")

    if recommended:
        lines.append("### 建议保留的代表工况")
        rows = []
        for name in objective_names:
            rows.append([name, _objective_from_row(recommended, name)])
        lines.extend(_table(["指标", "代表工况值"], rows))
        changes = []
        initial = state.get("initial_point") or {}
        for key, value in (recommended.get("optimizer_inputs") or {}).items():
            old = initial.get(key)
            if old is not None and not _same_value(old, value):
                changes.append(f"{_parameter_label(key, metadata)} {_fmt(old)} → {_fmt(value)}")
        if changes:
            lines.append("")
            lines.append("相对初始点的主要变化：" + "；".join(changes[:8]) + "。")
        constraint_summary = []
        for name, item in _constraint_values(recommended).items():
            state_text = "满足" if item.get("satisfied") else "违反"
            constraint_summary.append(f"{_constraint_label(name)}{state_text}")
        if constraint_summary:
            lines.append("代表工况约束状态：" + "；".join(constraint_summary) + "。")
        lines.append("")

    probes = []
    for decision in decisions:
        plan = decision.get("plan") or {}
        if plan.get("action") != "probe":
            continue
        outcome = decision.get("outcome") or {}
        ids = outcome.get("case_ids") or []
        if not ids:
            continue
        conclusion, feasible_n, infeasible_n, failed_n = _decision_outcome(decision, cases)
        changes = _planned_changes(plan, state, cases, metadata, limit=3)
        if not changes:
            continue
        probes.append(["；".join(changes), conclusion, "有改善" if outcome.get("expected_metric_improved") else "未显示明确改善"])
    if probes:
        lines.append("### 关键定向试验")
        lines.extend(_table(["试验变量", "实际结果", "对目标的意义"], probes[-6:]))
        lines.append("")
        lines.append("这些试验采用了单变量或少变量扰动，适合定位边界；但对于同时改变多个参数的试验，只能作为组合效果观察，不能拆分成单变量因果结论。")
        lines.append("")

    lines.extend(["## 四、专业建议", ""])
    recommendations: list[str] = []
    if constraints:
        ordered = sorted(constraints.items(), key=lambda item: item[1].get("violation_rate") or 0, reverse=True)
        bottlenecks = "、".join(_constraint_label(name) for name, item in ordered[:2] if (item.get("violation_rate") or 0) > 0)
        if bottlenecks:
            recommendations.append(f"优先围绕 {bottlenecks} 做可行域修复实验；建议固定其他变量，每次只改变一个补偿变量，并记录纯度裕量而不只记录是否合格。")
    if recommended:
        recommendations.append("将上述代表工况作为下一轮搜索锚点，不要直接从不可行低成本点继续扩大扰动；先确认该锚点的三个产品纯度均留有足够裕量。")
    if termination == "stagnation" and state.get("used", 0) < (_number(max_evaluations) or state.get("used", 0)):
        recommendations.append("本轮在预算未用尽时因停滞停止，不能视为优化完成。下一轮应减少无效候选和协议拒绝，采用更窄的局部区域继续搜索。")
    sensitivity = analysis.get("sensitivity") or {}
    ranked = [item for item in sensitivity.get("ranked_variables") or [] if item.get("reliable")]
    if ranked:
        labels = [_parameter_label(item.get("path"), metadata) for item in ranked[:3]]
        recommendations.append("敏感性排序仅作为筛选线索；当前可优先考察 " + "、".join(labels) + "，但必须用新的单变量试验验证，不应直接当作物理定律。")
    if not targets:
        recommendations.append("如果项目有明确业务底线，建议在下一轮配置中补充 CAPEX/OPEX 目标阈值；当前报告只能判断相对改善，不能判断是否达到业务目标。")
    if not recommendations:
        recommendations.append("当前证据不足以给出稳定的参数方向，建议先增加可行工况样本，再进行目标优化。")
    lines.extend(f"{index}. {item}" for index, item in enumerate(recommendations, 1))
    lines.extend([
        "",
        "## 五、结论边界",
        "",
        "本报告基于已完成的 Aspen 仿真与约束计算。收敛不等于产品合格，单次试验后的指标变化是观察关联而不是因果证明；当前结果适合作为下一轮实验依据，不应直接作为全局最优或最终工艺定版。",
        "",
    ])
    return "\n".join(lines)


def render_explanatory_report(
    state: Mapping[str, Any],
    *,
    settings: Any = None,
    objective_names: Sequence[str] | None = None,
) -> str:
    """Render a concise analyst-facing report; raw audit data remains JSON."""
    problem = state.get("problem") or {}
    names = list(objective_names or problem.get("objectives") or [])
    return _analyst_report(state, settings=settings, objective_names=names)
