"""
explainability.py — Agent 决策可解释性

核心功能：
---------
1. 记录完整的决策推理链
2. 生成自然语言解释
3. 评估决策置信度
4. 支持动态 HITL 触发

设计原则：
---------
- 推理过程可追溯：记录从观察 → 分析 → 决策的完整链条
- 证据可验证：所有引用的 case_id 和 knowledge_id 都可追溯
- 不确定性透明：明确标记置信度和不确定因素
- 人工介入友好：置信度低时主动请求人工确认

使用：
-----
    trace = DecisionTrace(decision_id="dec_001", action_plan=action.to_legacy_plan())
    trace.reasoning_steps.append("观察到收敛率下降...")
    trace.confidence = 0.65

    if trace.should_request_human():
        print(trace.to_natural_language())
        # 等待人工确认
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionTrace:
    """
    决策追踪记录。

    记录 Agent 做出决策的完整过程，支持事后审计和实时解释。

    Attributes:
        decision_id: 决策唯一标识
        action_plan: 最终决策的 ActionPlan（字典格式）
        reasoning_steps: 推理步骤列表（按时间顺序）
        evidence_summary: 使用的证据摘要
        knowledge_applied: 应用的工艺知识规则
        similar_cases: 参考的历史相似案例
        alternatives_considered: 考虑过但未选择的备选方案
        why_chosen: 为什么选择当前方案
        confidence: 决策置信度（0-1）
        uncertainty_reasons: 不确定性来源列表
    """
    decision_id: str
    action_plan: dict

    # 推理过程
    reasoning_steps: list[str] = field(default_factory=list)
    evidence_summary: list[dict] = field(default_factory=list)
    knowledge_applied: list[dict] = field(default_factory=list)
    similar_cases: list[dict] = field(default_factory=list)

    # 备选方案
    alternatives_considered: list[dict] = field(default_factory=list)
    why_chosen: str = ""

    # 置信度
    confidence: float = 1.0
    uncertainty_reasons: list[str] = field(default_factory=list)

    def to_natural_language(self) -> str:
        """
        生成自然语言解释。

        输出为 Markdown 格式，包含：
        - 决策概述
        - 推理过程
        - 使用的证据
        - 应用的知识
        - 参考的案例
        - 备选方案
        - 置信度评估
        """
        lines = [f"# 决策 {self.decision_id} 解释\n"]

        # ======== 1. 动作概述 ========
        action = self.action_plan.get("action", "unknown")
        hypothesis = self.action_plan.get("hypothesis", "无")
        lines.append(f"**动作类型：** {action}")
        lines.append(f"**假设：** {hypothesis}\n")

        # ======== 2. 推理过程 ========
        if self.reasoning_steps:
            lines.append("## 推理过程\n")
            for i, step in enumerate(self.reasoning_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        # ======== 3. 使用的证据 ========
        if self.evidence_summary:
            lines.append("## 使用的证据\n")
            for ev in self.evidence_summary:
                case_id = ev.get("case_id", "?")
                summary = ev.get("summary", "")
                lines.append(f"- **{case_id}**: {summary}")
            lines.append("")

        # ======== 4. 应用的知识 ========
        if self.knowledge_applied:
            lines.append("## 应用的工艺知识\n")
            for k in self.knowledge_applied:
                rule_id = k.get("id", "?")
                guidance = k.get("guidance", "")
                lines.append(f"- **{rule_id}**: {guidance}")
            lines.append("")

        # ======== 5. 参考的历史案例 ========
        if self.similar_cases:
            lines.append("## 参考的历史案例\n")
            for case in self.similar_cases:
                similarity = case.get("similarity", 0)
                result = case.get("final_result", "")
                actions_count = len(case.get("actions_taken", []))
                lines.append(
                    f"- 相似度 {similarity:.1%}，"
                    f"{actions_count} 步动作，"
                    f"最终结果：{result}"
                )
            lines.append("")

        # ======== 6. 备选方案 ========
        if self.alternatives_considered:
            lines.append("## 考虑过的其他方案\n")
            for alt in self.alternatives_considered:
                alt_action = alt.get("action", "?")
                reason = alt.get("not_chosen_reason", "")
                lines.append(f"- **{alt_action}**: {reason}")
            lines.append("")

        # ======== 7. 为什么选择当前方案 ========
        if self.why_chosen:
            lines.append(f"## 选择理由\n\n{self.why_chosen}\n")

        # ======== 8. 置信度评估 ========
        if self.confidence >= 0.8:
            confidence_emoji = "[高]"
        elif self.confidence >= 0.6:
            confidence_emoji = "[中]"
        else:
            confidence_emoji = "[低]"
        lines.append(f"## 决策置信度：{confidence_emoji} {self.confidence:.0%}\n")

        if self.uncertainty_reasons:
            lines.append("**不确定因素：**")
            for reason in self.uncertainty_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        # ======== 9. 建议 ========
        if self.should_request_human():
            lines.append("[警告] **建议：** 置信度较低，建议人工确认后再执行。")
        else:
            lines.append("[OK] **建议：** 置信度足够，可以直接执行。")

        return "\n".join(lines)

    def should_request_human(self, threshold: float = 0.6) -> bool:
        """
        是否应请求人工介入。

        判断依据：
        1. 置信度低于阈值
        2. 不确定因素过多（≥ 3 个）

        Args:
            threshold: 置信度阈值（默认 0.6）

        Returns:
            True 表示应请求人工介入
        """
        return self.confidence < threshold or len(self.uncertainty_reasons) >= 3

    def to_dict(self) -> dict:
        """转换为字典（用于持久化）"""
        return {
            "decision_id": self.decision_id,
            "action_plan": self.action_plan,
            "reasoning_steps": self.reasoning_steps,
            "evidence_summary": self.evidence_summary,
            "knowledge_applied": self.knowledge_applied,
            "similar_cases": self.similar_cases,
            "alternatives_considered": self.alternatives_considered,
            "why_chosen": self.why_chosen,
            "confidence": self.confidence,
            "uncertainty_reasons": self.uncertainty_reasons,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionTrace":
        """从字典恢复"""
        return cls(
            decision_id=data["decision_id"],
            action_plan=data["action_plan"],
            reasoning_steps=data.get("reasoning_steps", []),
            evidence_summary=data.get("evidence_summary", []),
            knowledge_applied=data.get("knowledge_applied", []),
            similar_cases=data.get("similar_cases", []),
            alternatives_considered=data.get("alternatives_considered", []),
            why_chosen=data.get("why_chosen", ""),
            confidence=data.get("confidence", 1.0),
            uncertainty_reasons=data.get("uncertainty_reasons", []),
        )


def build_trace_from_snapshot(
    decision_id: str,
    action_plan: dict,
    snapshot: dict,
) -> DecisionTrace:
    """
    从 snapshot 构建 DecisionTrace（自动提取推理步骤）。

    这是一个辅助函数，用于快速构建包含基本推理信息的 trace。

    Args:
        decision_id: 决策 ID
        action_plan: ActionPlan 字典
        snapshot: 优化状态快照

    Returns:
        DecisionTrace 实例
    """
    trace = DecisionTrace(decision_id=decision_id, action_plan=action_plan)

    # 自动提取关键指标作为推理步骤
    convergence = snapshot.get("recent_convergence_rate", 0.0)
    feasible = snapshot.get("recent_feasible_rate", 0.0)
    stagnation = snapshot.get("stagnation_count", 0)
    budget = snapshot.get("budget_remaining", 0)

    trace.reasoning_steps.append(
        f"当前状态：收敛率 {convergence:.1%}，可行率 {feasible:.1%}，"
        f"停滞批次 {stagnation}，剩余预算 {budget}"
    )

    # 提取证据
    for obs in snapshot.get("recent_observations", [])[-5:]:
        trace.evidence_summary.append({
            "case_id": obs.get("case_id", "?"),
            "summary": f"收敛={obs.get('converged', False)}, 可行={obs.get('success', False)}",
        })

    # 初始置信度评估（基于简单规则）
    if convergence >= 0.7 and feasible >= 0.6:
        trace.confidence = 0.9
    elif convergence >= 0.5 or feasible >= 0.4:
        trace.confidence = 0.7
    else:
        trace.confidence = 0.5
        trace.uncertainty_reasons.append("收敛率或可行率较低")

    if stagnation >= 3:
        trace.confidence *= 0.9
        trace.uncertainty_reasons.append(f"已停滞 {stagnation} 批次")

    if budget < 10:
        trace.uncertainty_reasons.append(f"剩余预算不足（仅剩 {budget}）")

    return trace
