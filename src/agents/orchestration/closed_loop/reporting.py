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
            from src.agents.orchestration.closed_loop.analysis import build_analysis_from_rows

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
        "driver_unavailable": "Aspen 驱动不可用，已停止后续派发；底层停止及清理状态以资源记录为准，结果未复验。",
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
                f"{diagnosis.get('description', '')}（规则匹配标签，不是 Agent 诊断）"
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


def _analyst_report(state, *, settings, objective_names):
    """Render persisted facts and a separately versioned Agent analysis only."""
    from src.agents.actions.authorization import digest
    from src.agents.orchestration.models import Answer
    from src.reporting.answer_export import markdown_answer
    observations = [row for row in state.get("observations", []) if isinstance(row, Mapping)]
    analysis = _latest_analysis(state, objective_names)
    lines = ["# PAO 执行记录与分析", "",
             f"观测工况 {len(observations)}；停止原因：{state.get('termination_reason', 'unknown')}。", ""]
    lines.extend(_render_final_evidence(analysis))
    for row in observations:
        lines.append(_case_description(row))
        for obj in row.get("objectives") or []:
            direction = "minimize" if obj.get("minimize") is True else "maximize" if obj.get("minimize") is False else "unknown"
            lines.append(f"- {obj.get('name')}：{obj.get('value')} {obj.get('unit', 'unknown')}；方向 {direction}")
        lines.append(f"- 约束数量：{len(row.get('constraints') or [])}")
    agent = state.get("agent_analysis") or {}
    if agent.get("status") == "available" and agent.get("evidence_digest") == digest(observations) and agent.get("answer"):
        lines.extend(["", f"Agent 分析版本 {agent.get('version', 'unknown')}", markdown_answer(Answer.from_dict(agent["answer"]))])
    else:
        lines.extend(["", "分析不可用：缺少绑定当前观测版本的 Agent 分析。仅展示执行与数值记录，不生成专业建议。"])
    return "\n".join(lines) + "\n"


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
