"""
action_types.py — 类型安全的 ActionPlan 实现

设计原则：
- 每种 action 类型有独立的 dataclass，避免字段混淆
- 编译期类型检查，降低运行时错误
- LLM 生成时减少字段互斥冲突
- 提供向后兼容的 to_legacy_plan() 转换

使用：
    # 创建 continue 动作
    action = ContinueAction(
        hypothesis="继续收集目标改善证据",
        evidence_case_ids=["case_123"],
        batch_size=5
    )

    # 转换为旧格式（向后兼容）
    legacy_plan = action.to_legacy_plan()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseAction:
    """所有动作的基类，包含通用字段"""
    hypothesis: str
    evidence_case_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    expected_effect: Literal["feasibility", "convergence", "objective", "information"] = "feasibility"
    initialization: Literal["reset", "previous"] = "reset"


@dataclass
class ContinueAction(BaseAction):
    """
    继续当前搜索策略，不改变搜索区域。

    适用场景：
    - 当前策略有效，需要更多数据
    - 收敛率和可行率正常
    - 目标函数持续改善

    特点：
    - 不包含 search_region 字段（明确为空）
    - 不包含 candidate 字段（明确为空）
    - 批次大小可配置
    """
    action: Literal["continue"] = "continue"
    batch_size: int = 5

    def to_legacy_plan(self) -> dict:
        """转换为旧格式 ActionPlan（向后兼容）"""
        return {
            "action": "continue",
            "hypothesis": self.hypothesis,
            "evidence_case_ids": self.evidence_case_ids,
            "knowledge_ids": self.knowledge_ids,
            "search_region": {},  # 明确为空，防止字段混淆
            "candidate": {},      # 明确为空，防止字段混淆
            "batch_size": self.batch_size,
            "expected_effect": self.expected_effect,
            "initialization": self.initialization,
            "repeat": False,
        }


@dataclass
class SetRegionAction(BaseAction):
    """
    调整软搜索区域。

    适用场景：
    - 收敛率低，需要缩小搜索范围
    - 停滞严重，需要扩大探索范围
    - 需要移动搜索重心

    特点：
    - 包含 search_region 字段（变量名 → [下界, 上界]）
    - 不包含 candidate 字段（明确为空）
    - 边界必须在硬边界内
    """
    action: Literal["set_region"] = "set_region"
    search_region: dict[str, list[float]] = field(default_factory=dict)
    batch_size: int = 5

    def to_legacy_plan(self) -> dict:
        """转换为旧格式 ActionPlan（向后兼容）"""
        return {
            "action": "set_region",
            "hypothesis": self.hypothesis,
            "evidence_case_ids": self.evidence_case_ids,
            "knowledge_ids": self.knowledge_ids,
            "search_region": self.search_region,
            "candidate": {},  # 明确为空，防止字段混淆
            "batch_size": self.batch_size,
            "expected_effect": self.expected_effect,
            "initialization": self.initialization,
            "repeat": False,
        }


@dataclass
class ProbeAction(BaseAction):
    """
    探测特定候选点。

    适用场景：
    - 有明确假设需要验证（如边界点行为）
    - 需要确认某个参数组合的收敛性
    - 诊断失败原因

    特点：
    - 包含 candidate 字段（变量名 → 具体数值）
    - 不包含 search_region 字段（明确为空）
    - 批次大小固定为 1
    - 支持 repeat 重复验证
    """
    action: Literal["probe"] = "probe"
    candidate: dict[str, float] = field(default_factory=dict)
    repeat: bool = False

    def to_legacy_plan(self) -> dict:
        """转换为旧格式 ActionPlan（向后兼容）"""
        return {
            "action": "probe",
            "hypothesis": self.hypothesis,
            "evidence_case_ids": self.evidence_case_ids,
            "knowledge_ids": self.knowledge_ids,
            "search_region": {},  # 明确为空，防止字段混淆
            "candidate": self.candidate,
            "batch_size": 1,  # probe 始终批次为 1
            "expected_effect": self.expected_effect,
            "initialization": self.initialization,
            "repeat": self.repeat,
        }


@dataclass
class StopAction(BaseAction):
    """
    请求停止优化。

    适用场景：
    - 已达到目标阈值
    - 预算即将耗尽
    - 判断无进展可能

    特点：
    - 不包含任何搜索相关字段
    - 批次大小为 0
    """
    action: Literal["stop"] = "stop"

    def to_legacy_plan(self) -> dict:
        """转换为旧格式 ActionPlan（向后兼容）"""
        return {
            "action": "stop",
            "hypothesis": self.hypothesis,
            "evidence_case_ids": self.evidence_case_ids,
            "knowledge_ids": self.knowledge_ids,
            "search_region": {},
            "candidate": {},
            "batch_size": 0,
            "expected_effect": self.expected_effect,
            "initialization": self.initialization,
            "repeat": False,
        }


# 类型联合：所有合法的 Action 类型
Action = ContinueAction | SetRegionAction | ProbeAction | StopAction


def parse_action_from_dict(data: dict) -> Action:
    """
    从字典解析 Action（工厂函数）。

    Args:
        data: 包含 action 字段的字典

    Returns:
        对应的 Action 实例

    Raises:
        ValueError: action 类型未知或字段不合法

    Example:
        >>> data = {"action": "continue", "hypothesis": "测试", "batch_size": 3}
        >>> action = parse_action_from_dict(data)
        >>> isinstance(action, ContinueAction)
        True
    """
    from .contracts import ActionPlan
    plan = ActionPlan.parse(data)
    if plan.action != "set_region" and plan.search_region or plan.action != "probe" and plan.candidate:
        raise ValueError("Action parameters do not match action type")
    common = {k: data[k] for k in ("hypothesis", "evidence_case_ids", "knowledge_ids", "expected_effect", "initialization")}
    if plan.action == "continue":
        return ContinueAction(**common, batch_size=plan.batch_size)
    if plan.action == "set_region":
        return SetRegionAction(**common, batch_size=plan.batch_size, search_region=plan.search_region)
    if plan.action == "probe":
        if plan.batch_size != 1:
            raise ValueError("Probe requires batch_size 1")
        return ProbeAction(**common, candidate=plan.candidate, repeat=plan.repeat)
    return StopAction(**common)
