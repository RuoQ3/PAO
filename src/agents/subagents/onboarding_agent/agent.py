"""
onboarding_agent/agent.py — B1 接入向导核心逻辑

公开接口：
  OnboardingResult   — 向导结果 dataclass
  run_onboarding     — 主入口：严格意图解析 → 扫描 → 配置草案 → 提问列表
  apply_user_feedback — 将用户反馈合并进草案
"""
from __future__ import annotations
from src.agents.errors import unavailable, fail_closed, failure_from
from src.agents.reasoning import configured

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.models.tunable import ConfigDraft, GoalSpec, OptimizationIntent, TunableReport

# 提到模块顶层以允许测试 monkeypatch
from src.agents.subagents.tools.discover_tunables import discover_tunables_tool
from src.agents.subagents.config_builder import (
    IntentParseError,
    build_config_draft,
    parse_intent_from_text,
)


# ---------------------------------------------------------------------------
# 向导结果与输入完整性检查
# ---------------------------------------------------------------------------


@dataclass
class OnboardingResult:
    config_draft: ConfigDraft
    tunable_report: TunableReport
    questions_for_user: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    intent: OptimizationIntent | None = None
    prioritization_notes: str = ""


def _build_questions(
    report: TunableReport,
    draft: ConfigDraft,
    intent: OptimizationIntent,
) -> list[str]:
    """根据扫描报告、配置草案、意图，生成需要用户确认的问题列表。

    问题数量限制：
    - medium 置信度变量：每条单独列出（通常数量少）
    - low 置信度变量：超过 5 条时改为汇总（避免产生数千条消息）
    """
    questions: list[str] = []

    medium_vars = [v for v in report.tunable_variables if v.confidence == "medium"]
    low_vars    = [v for v in report.tunable_variables if v.confidence == "low"]

    # 1. medium 置信度变量：逐条列出（通常不多）
    for var in medium_vars:
        lo = var.suggested_lower
        hi = var.suggested_upper
        if lo is not None and hi is not None:
            hint = f"建议 [{lo}, {hi}]"
        else:
            hint = "当前无建议边界，请根据工艺经验填写"
        questions.append(
            f"请确认 {var.aspen_path} 的合理范围（{hint}）"
            f"【置信度：medium，原因：{var.reason}】"
        )

    # 2. low 置信度变量：超过 5 条则改为汇总说明
    _LOW_DETAIL_LIMIT = 5
    if len(low_vars) <= _LOW_DETAIL_LIMIT:
        for var in low_vars:
            questions.append(
                f"请确认 {var.aspen_path} 的合理范围（当前无建议边界，请根据工艺经验填写）"
                f"【置信度：low，原因：{var.reason}】"
            )
    else:
        questions.append(
            f"发现 {len(low_vars)} 个低置信度变量（无语义规则命中），"
            "在下方「变量边界确认」表中可逐一填写范围，"
            "留空则从本轮优化中排除。"
        )

    # 3. 目标函数映射有 warning
    for w in draft.warnings:
        if "目标" in w or "metric" in w or "objective" in w.lower():
            questions.append(f"目标配置需确认：{w}")

    # 4. 采样与迭代次数建议
    n_initial = intent.n_initial
    n_iter    = intent.n_iterations
    questions.append(
        f"建议运行 {n_initial} 次初始采样 + {n_iter} 次优化迭代"
        f"（共 {n_initial + n_iter} 次 Aspen 仿真），是否接受？"
    )

    return questions


# ---------------------------------------------------------------------------
# B1-2  run_onboarding
# ---------------------------------------------------------------------------

@fail_closed("onboarding")
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
         LLM 不可用时立即终止
      3. 调用 build_config_draft 生成 ConfigDraft
      4. 生成 questions_for_user

    不启动优化，不调用 run_case / optimize_pareto。

    Parameters
    ----------
    aspen_file_path:
        Aspen 仿真文件路径（.bkp/.apw）。
    intent_text:
        用户自由文本描述的优化意图；空字符串为输入错误。
    node_db_path:
        节点目录数据库路径（NodeDB .db 文件）。
    llm_config:
        LLMConfig 实例；None 表示加载 LLM 配置。

    Returns
    -------
    OnboardingResult
    """
    # Parse and validate the actual intent before touching project evidence.
    intent = parse_intent_from_text(intent_text, llm_config)
    # ── 1. 变量发现 ──────────────────────────────────────────────────────────
    raw_json = discover_tunables_tool.invoke({
        "aspen_file_path": aspen_file_path,
        "node_db_path": node_db_path,
        "max_depth": 6,
    })

    if raw_json.startswith("错误："):
        raise unavailable("agent_output_invalid", stage="onboarding_evidence")
    report = _parse_tunable_report(raw_json, aspen_file_path)
    from src.agents.subagents.tools.prioritize_tunables import prioritize_tunables_impl
    priority_result = prioritize_tunables_impl(report, intent, llm_config)
    report.tunable_variables = priority_result.ranked_variables
    prioritization_notes = priority_result.ranking_notes

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
        intent=intent,
        prioritization_notes=prioritization_notes,
    )


def _parse_tunable_report(raw_json: str, aspen_file_path: str) -> TunableReport:
    """将 discover_tunables_tool 返回的 JSON 字符串反序列化为 TunableReport。"""
    from src.models.tunable import TunableVariable, ReadableTarget

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise failure_from(exc, "onboarding_evidence") from None

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
        except (KeyError, TypeError) as exc:
            raise failure_from(exc, "onboarding_evidence") from None

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
        except (KeyError, TypeError) as exc:
            raise failure_from(exc, "onboarding_evidence") from None

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

def apply_user_feedback(
    draft: ConfigDraft,
    feedback: dict,
    tunable_report: TunableReport | None = None,
) -> ConfigDraft:
    """
    将用户反馈合并进 ConfigDraft，返回更新后的新草案实例。

    参数说明
    --------
    draft:
        当前配置草案（仅含已通过筛选进入 design_variables 的变量）。
    feedback:
        用户反馈字典，支持以下键：

        - ``"bounds"`` : ``{aspen_path: [lo, hi]}``
            更新已有设计变量的边界，**或**将 tunable_report 中已知但尚未入草案的
            变量（in_draft=False）提升进 design_variables。
            - 路径在 design_variables 中：直接更新 lower_bound / upper_bound。
            - 路径不在 design_variables 但存在于 tunable_report：
              复制该 TunableVariable，覆盖边界，通过 ``_map_tunable_to_design_var``
              提升为新的设计变量条目，追加进 design_variables。
              同时删除草案 warnings 中对应的边界缺失提示。
            - 路径既不在草案也不在 tunable_report：写 warning 后忽略，
              不允许任意路径进入优化 YAML（安全约束）。
        - ``"objectives"`` : 替换目标函数列表
        - ``"constraints"`` : 替换约束列表
        - ``"n_initial"`` : 更新初始采样次数
        - ``"n_iterations"`` : 更新迭代次数

    tunable_report:
        OnboardingResult.tunable_report；提供时支持将未入草案的变量提升进 draft。
        None 时退化为旧行为（只更新已有变量，未知路径一律忽略）。

    返回
    ----
    ConfigDraft
        新草案（深拷贝后修改，不修改入参）。
    """
    updated = copy.deepcopy(draft)
    feedback_warnings: list[str] = []

    # ── bounds：更新设计变量边界（含提升 in_draft=False 变量）────────────────
    bounds: dict = feedback.get("bounds", {}) or {}
    if bounds:
        known_paths = {dv.get("aspen_path", "") for dv in updated.design_variables}
        confirmed_paths: set[str] = set()

        # 构建 tunable_report 的路径查找表，用于提升 in_draft=False 变量
        tunable_map: dict[str, Any] = {}
        if tunable_report is not None:
            for tv in tunable_report.tunable_variables:
                tunable_map[getattr(tv, "aspen_path", "")] = tv

        for path, val in bounds.items():
            # val 必须是长度 ≥ 2 的序列（无论哪个分支都要校验）
            try:
                lo, hi = float(val[0]), float(val[1])
            except (TypeError, IndexError, KeyError, ValueError):
                feedback_warnings.append(
                    f"bounds[{path!r}] 格式错误（期望 [lo, hi]，得到 {val!r}），已忽略。"
                )
                continue

            if path in known_paths:
                # ── 分支 A：更新已有变量边界 ──────────────────────────────────
                for dv in updated.design_variables:
                    if dv.get("aspen_path", "") == path:
                        dv["lower_bound"] = lo
                        dv["upper_bound"] = hi
                        confirmed_paths.add(path)

            elif path in tunable_map:
                # ── 分支 B：提升 in_draft=False 的变量进 design_variables ─────
                import copy as _copy
                tv = _copy.copy(tunable_map[path])
                # 用用户提交的边界覆盖建议边界
                tv.suggested_lower = lo
                tv.suggested_upper = hi
                # confidence 提升为 medium（用户已明确确认边界）
                tv.confidence = "medium"

                from src.agents.subagents.config_builder import (
                    _map_tunable_to_design_var,
                    _extract_name_from_path,
                    _BLOCK_ROOT,
                    _STREAM_ROOT,
                )
                promote_warnings: list[str] = []
                new_dv = _map_tunable_to_design_var(tv, promote_warnings)
                updated.design_variables.append(new_dv)
                known_paths.add(path)  # 防止同路径重复提升
                confirmed_paths.add(path)
                feedback_warnings.extend(promote_warnings)
                feedback_warnings.append(
                    f"变量 {path!r} 已从候选列表提升进设计变量（用户确认边界 [{lo}, {hi}]）。"
                )

                # ── 同步更新 extraction，使 Aspen 状态检查覆盖提升变量 ──────────
                # extraction 可能是 dict 子类或普通 dict；用 dict() 确保可写
                ext: dict = dict(updated.extraction)
                blocks_list: list[str] = list(ext.get("blocks", []))
                streams_list: list[str] = list(ext.get("streams", []))
                check_paths_list: list[str] = list(ext.get("check_status_paths", []))

                block_name = _extract_name_from_path(path, _BLOCK_ROOT)
                if block_name and block_name not in blocks_list:
                    blocks_list.append(block_name)
                    check_entry = rf"\Data\Blocks\{block_name}"
                    if check_entry not in check_paths_list:
                        check_paths_list.append(check_entry)

                stream_name = _extract_name_from_path(path, _STREAM_ROOT)
                if stream_name and stream_name not in streams_list:
                    streams_list.append(stream_name)
                    check_entry = rf"\Data\Streams\{stream_name}"
                    if check_entry not in check_paths_list:
                        check_paths_list.append(check_entry)

                ext["blocks"] = blocks_list
                ext["streams"] = streams_list
                ext["check_status_paths"] = check_paths_list
                updated.extraction = ext

            else:
                # ── 分支 C：路径既不在草案也不在 tunable_report ───────────────
                # 严格拒绝——不允许任意路径进入优化 YAML
                feedback_warnings.append(
                    f"bounds 中的路径 {path!r} 既不在设计变量列表中，"
                    "也不在本次变量发现扫描结果中，已忽略。"
                    "请检查路径是否正确，或重新运行 discover_tunables。"
                )

        # 从 warnings 移除已被用户成功确认边界的条目
        updated.warnings = [
            w for w in updated.warnings
            if not any(p in w for p in confirmed_paths)
        ]

    # ── excluded_paths：从 design_variables 中删除用户主动排除的变量 ────────────
    excluded_paths: list[str] = feedback.get("excluded_paths") or []
    if excluded_paths:
        excluded_set = set(excluded_paths)
        before_count = len(updated.design_variables)
        updated.design_variables = [
            dv for dv in updated.design_variables
            if dv.get("aspen_path", "") not in excluded_set
        ]
        removed = before_count - len(updated.design_variables)
        if removed:
            feedback_warnings.append(
                f"已按用户要求排除 {removed} 个边界为空的设计变量：{excluded_paths}。"
            )

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
