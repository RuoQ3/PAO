"""
onboarding_agent/agent.py — B1 接入向导核心逻辑

公开接口：
  OnboardingResult   — 向导结果 dataclass
  run_onboarding     — 主入口：扫描 → 意图解析 → 配置草案 → 提问列表
  apply_user_feedback — 将用户反馈合并进草案
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.models.tunable import ConfigDraft, GoalSpec, OptimizationIntent, TunableReport

# 提到模块顶层以允许测试 monkeypatch
from src.agents.tools.discover_tunables import discover_tunables_tool
from src.agents.config_builder import (
    IntentParseError,
    build_config_draft,
    parse_intent_from_text,
)


# ---------------------------------------------------------------------------
# 内部：默认意图（TAC 最小 + 排放最小）
# ---------------------------------------------------------------------------

def _make_default_intent(notes: str = "") -> OptimizationIntent:
    """当用户未提供意图或 LLM 不可用时使用的保守默认意图。"""
    return OptimizationIntent(
        goals=[
            GoalSpec(metric="TAC", direction="min"),
            GoalSpec(metric="emissions", direction="min"),
        ],
        hard_constraints=[],
        n_initial=20,
        n_iterations=60,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# B1-1  OnboardingResult
# ---------------------------------------------------------------------------

@dataclass
class OnboardingResult:
    """
    run_onboarding 的产出，汇总了向导流程的全部结果。

    Attributes
    ----------
    config_draft:
        基于扫描结果与用户意图生成的配置草案。
    tunable_report:
        变量发现扫描报告（含 TunableVariable 和 ReadableTarget 列表）。
    questions_for_user:
        需要用户确认或补充的问题列表，每项是一个完整的中文问句。
        - confidence != "high" 的设计变量边界各生成一条
        - 目标函数映射有 warning 时各生成一条
        - 采样/迭代次数建议生成一条
    warnings:
        汇总所有需要用户关注的警告（来自扫描 + 配置构建）。
    """
    config_draft: ConfigDraft
    tunable_report: TunableReport
    questions_for_user: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 内部：生成 questions_for_user
# ---------------------------------------------------------------------------

def _build_questions(
    report: TunableReport,
    draft: ConfigDraft,
    intent: OptimizationIntent,
) -> list[str]:
    """根据扫描报告、配置草案、意图，生成需要用户确认的问题列表。"""
    questions: list[str] = []

    # 1. confidence != "high" 的设计变量边界
    for var in report.tunable_variables:
        if var.confidence == "high":
            continue
        lo = var.suggested_lower
        hi = var.suggested_upper
        if lo is not None and hi is not None:
            hint = f"建议 [{lo}, {hi}]"
        else:
            hint = "当前无建议边界，请根据工艺经验填写"
        questions.append(
            f"请确认 {var.aspen_path} 的合理范围（{hint}）"
            f"【置信度：{var.confidence}，原因：{var.reason}】"
        )

    # 2. 目标函数映射有 warning（从 draft.warnings 中提取目标相关的）
    for w in draft.warnings:
        if "目标" in w or "metric" in w or "objective" in w.lower():
            questions.append(f"目标配置需确认：{w}")

    # 3. 采样与迭代次数建议
    n_initial = intent.n_initial
    n_iter = intent.n_iterations
    questions.append(
        f"建议运行 {n_initial} 次初始采样 + {n_iter} 次优化迭代"
        f"（共 {n_initial + n_iter} 次 Aspen 仿真），是否接受？"
    )

    return questions


# ---------------------------------------------------------------------------
# B1-2  run_onboarding
# ---------------------------------------------------------------------------

def run_onboarding(
    aspen_file_path: str,
    intent_text: str,
    node_db_path: str,
    llm_config: Any = None,
) -> OnboardingResult:
    """
    接入向导主入口。

    流程：
      1. 调用 discover_tunables_tool 扫描 Aspen 文件，获得 TunableReport
      2. 若 intent_text 非空，调用 parse_intent_from_text 解析意图；
         LLM 不可用时使用默认意图
      3. 调用 build_config_draft 生成 ConfigDraft
      4. 生成 questions_for_user

    不启动优化，不调用 run_case / optimize_pareto。

    Parameters
    ----------
    aspen_file_path:
        Aspen 仿真文件路径（.bkp/.apw）。
    intent_text:
        用户自由文本描述的优化意图；空字符串时使用默认意图。
    node_db_path:
        节点目录数据库路径（NodeDB .db 文件）。
    llm_config:
        LLMConfig 实例；None 表示不使用 LLM。

    Returns
    -------
    OnboardingResult
    """
    # ── 1. 变量发现 ──────────────────────────────────────────────────────────
    raw_json = discover_tunables_tool.invoke({
        "aspen_file_path": aspen_file_path,
        "node_db_path": node_db_path,
        "max_depth": 6,
    })

    if raw_json.startswith("错误："):
        # 扫描完全失败：构造空报告，仍能继续（但所有变量 confidence=low）
        report = TunableReport(
            aspen_file=aspen_file_path,
            aspen_file_hash="",
            scan_warnings=[raw_json],
            semantic_coverage=0.0,
        )
    else:
        report = _parse_tunable_report(raw_json, aspen_file_path)

    # ── 2. 意图解析 ──────────────────────────────────────────────────────────
    if intent_text.strip():
        try:
            intent = parse_intent_from_text(intent_text, llm_config)
        except (IntentParseError, Exception):
            intent = _make_default_intent("LLM 解析失败，已使用默认意图")
    else:
        intent = _make_default_intent("未提供意图文本，已使用默认意图")

    # ── 3. 配置草案 ──────────────────────────────────────────────────────────
    draft = build_config_draft(report, intent, node_db_path=node_db_path)

    # ── 4. 生成问题列表 ──────────────────────────────────────────────────────
    questions = _build_questions(report, draft, intent)

    # 汇总 warnings（扫描 + 草案）
    all_warnings = list(report.scan_warnings) + list(draft.warnings)

    return OnboardingResult(
        config_draft=draft,
        tunable_report=report,
        questions_for_user=questions,
        warnings=all_warnings,
    )


def _parse_tunable_report(raw_json: str, aspen_file_path: str) -> TunableReport:
    """将 discover_tunables_tool 返回的 JSON 字符串反序列化为 TunableReport。"""
    from src.models.tunable import TunableVariable, ReadableTarget

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return TunableReport(
            aspen_file=aspen_file_path,
            aspen_file_hash="",
            scan_warnings=[f"discover_tunables_tool 返回非法 JSON：{raw_json[:120]}"],
            semantic_coverage=0.0,
        )

    tunable_vars = []
    for v in data.get("tunable_variables", []):
        try:
            tunable_vars.append(TunableVariable(
                aspen_path=v["aspen_path"],
                semantic_role=v.get("semantic_role", ""),
                suggested_type=v.get("suggested_type", "continuous"),
                current_value=v.get("current_value"),
                suggested_lower=v.get("suggested_lower"),
                suggested_upper=v.get("suggested_upper"),
                unit=v.get("unit", "-"),
                confidence=v.get("confidence", "low"),
                reason=v.get("reason", ""),
            ))
        except (KeyError, TypeError):
            pass

    readable_targets = []
    for t in data.get("readable_targets", []):
        try:
            readable_targets.append(ReadableTarget(
                aspen_path=t["aspen_path"],
                semantic_role=t.get("semantic_role", ""),
                candidate_use=t.get("candidate_use", "objective"),
                unit=t.get("unit", "-"),
                current_value=t.get("current_value"),
            ))
        except (KeyError, TypeError):
            pass

    return TunableReport(
        aspen_file=data.get("aspen_file", aspen_file_path),
        aspen_file_hash=data.get("aspen_file_hash", ""),
        tunable_variables=tunable_vars,
        readable_targets=readable_targets,
        scan_warnings=data.get("scan_warnings", []),
        semantic_coverage=float(data.get("semantic_coverage", 0.0)),
    )


# ---------------------------------------------------------------------------
# B1-3  apply_user_feedback
# ---------------------------------------------------------------------------

def apply_user_feedback(draft: ConfigDraft, feedback: dict) -> ConfigDraft:
    """
    将用户反馈合并进 ConfigDraft，返回更新后的新草案实例。

    Parameters
    ----------
    draft:
        当前配置草案。
    feedback:
        用户反馈字典，支持以下键：
        - "bounds": {aspen_path: [lo, hi]}  更新设计变量边界
        - "objectives": [{"name":..., "aspen_path":..., "minimize":...}]  替换目标函数列表
        - "constraints": [{"name":..., "aspen_path":..., "operator":..., "threshold":...}]
        - "n_initial": int  更新初始采样次数
        - "n_iterations": int  更新迭代次数

    Returns
    -------
    ConfigDraft
        新草案（对原草案做深拷贝后修改，不修改入参）。
    """
    updated = copy.deepcopy(draft)
    feedback_warnings: list[str] = []

    # ── bounds：更新设计变量边界 ──────────────────────────────────────────────
    bounds: dict = feedback.get("bounds", {}) or {}
    if bounds:
        known_paths = {dv.get("aspen_path", "") for dv in updated.design_variables}
        confirmed_paths: set[str] = set()

        for path, val in bounds.items():
            # 未知路径 → warning，不写入
            if path not in known_paths:
                feedback_warnings.append(
                    f"bounds 中的路径 {path!r} 不在当前设计变量列表中，已忽略。"
                )
                continue
            # val 必须是长度 ≥ 2 的序列
            try:
                lo, hi = val[0], val[1]
            except (TypeError, IndexError, KeyError):
                feedback_warnings.append(
                    f"bounds[{path!r}] 格式错误（期望 [lo, hi]，得到 {val!r}），已忽略。"
                )
                continue

            for dv in updated.design_variables:
                if dv.get("aspen_path", "") == path:
                    dv["lower_bound"] = lo
                    dv["upper_bound"] = hi
                    confirmed_paths.add(path)

        # 从 warnings 移除已被用户成功确认边界的条目
        updated.warnings = [
            w for w in updated.warnings
            if not any(p in w for p in confirmed_paths)
        ]

    # ── objectives：替换目标函数列表 ─────────────────────────────────────────
    if "objectives" in feedback and feedback["objectives"]:
        updated.objectives = list(feedback["objectives"])

    # ── constraints：替换约束列表 ────────────────────────────────────────────
    if "constraints" in feedback and feedback["constraints"] is not None:
        updated.constraints = list(feedback["constraints"])

    # ── n_initial / n_iterations：更新优化器参数（非法值写 warning，不抛异常）──
    updated.optimizer = dict(updated.optimizer)
    if "n_initial" in feedback:
        try:
            updated.optimizer["n_initial_points"] = int(feedback["n_initial"])
        except (TypeError, ValueError):
            feedback_warnings.append(
                f"n_initial 无法转换为整数（值：{feedback['n_initial']!r}），已忽略。"
            )
    if "n_iterations" in feedback:
        try:
            updated.optimizer["n_iterations"] = int(feedback["n_iterations"])
        except (TypeError, ValueError):
            feedback_warnings.append(
                f"n_iterations 无法转换为整数（值：{feedback['n_iterations']!r}），已忽略。"
            )

    if feedback_warnings:
        updated.warnings = updated.warnings + feedback_warnings

    # ── 结构校验：将发现的问题写回 warnings ──────────────────────────────────
    _validate_and_warn(updated)

    return updated


def _validate_and_warn(draft: ConfigDraft) -> None:
    """对 apply_user_feedback 更新后的草案做结构校验，将问题写入 draft.warnings（原地修改）。"""
    new_warnings: list[str] = []

    # 设计变量边界数值顺序
    for dv in draft.design_variables:
        path = dv.get("aspen_path", "?")
        lo = dv.get("lower_bound")
        hi = dv.get("upper_bound")
        if lo is not None and hi is not None:
            try:
                if float(lo) > float(hi):
                    new_warnings.append(
                        f"变量 {path}：lower_bound ({lo}) > upper_bound ({hi})，请检查边界。"
                    )
            except (TypeError, ValueError):
                new_warnings.append(f"变量 {path}：边界值无法转换为数字（{lo}, {hi}）。")

    # 优化器参数为正整数
    n_init = draft.optimizer.get("n_initial_points")
    n_iter = draft.optimizer.get("n_iterations")
    if n_init is not None and (not isinstance(n_init, int) or n_init <= 0):
        new_warnings.append(f"optimizer.n_initial_points 应为正整数，当前值：{n_init}。")
    if n_iter is not None and (not isinstance(n_iter, int) or n_iter <= 0):
        new_warnings.append(f"optimizer.n_iterations 应为正整数，当前值：{n_iter}。")

    # 目标函数必备字段：name、aspen_path 或 type
    for i, obj in enumerate(draft.objectives):
        if not obj.get("name"):
            new_warnings.append(f"objectives[{i}] 缺少 name 字段。")
        if obj.get("type") == "aspen_path" and not obj.get("aspen_path"):
            new_warnings.append(f"objectives[{i}] type=aspen_path 但 aspen_path 为空。")

    # 约束必备字段：aspen_path、operator、threshold
    _VALID_OPERATORS = {">=", "<=", ">", "<", "=="}
    for i, con in enumerate(draft.constraints):
        missing = [f for f in ("aspen_path", "operator", "threshold") if f not in con or con[f] is None]
        if missing:
            new_warnings.append(f"constraints[{i}] 缺少必要字段 {missing}。")
            continue
        op = con["operator"]
        if op not in _VALID_OPERATORS:
            new_warnings.append(
                f"constraints[{i}] operator {op!r} 不合法，支持 {sorted(_VALID_OPERATORS)}。"
            )
        try:
            float(con["threshold"])
        except (TypeError, ValueError):
            new_warnings.append(
                f"constraints[{i}] threshold {con['threshold']!r} 无法转换为数值。"
            )

    if new_warnings:
        draft.warnings = draft.warnings + new_warnings
