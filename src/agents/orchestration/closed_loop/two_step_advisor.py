"""
two_step_advisor.py — 两阶段决策流程

设计思路：
-----------
当前问题：LLM 一次性生成完整 ActionPlan 时，容易同时填写互斥字段（如 search_region 和 candidate），
         导致协议违反率高达 40%。

解决方案：将决策拆分为两个独立步骤：
         Step 1: 决定动作类型（continue/set_region/probe/stop）
         Step 2: 根据类型生成对应参数

优势：
-----
1. 减少字段混淆：每步的输出结构更简单
2. 提高成功率：LLM 协议违反率预期降低至 < 10%
3. 易于调试：可以单独观察每一步的决策
4. 灵活降级：任一步失败都可以回退到规则决策

使用：
-----
    agent = TwoStepDecisionAgent(llm_config=cfg)
    action, source = agent.propose(snapshot)

    if source == "llm:two_step":
        print("LLM 成功决策")
    else:
        print("回退到规则决策")
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal

from .action_types import (
    Action,
    ContinueAction,
    ProbeAction,
    SetRegionAction,
    StopAction,
)

_log = logging.getLogger(__name__)

# ============================================================================
# Step 1: 动作类型选择 Prompt
# ============================================================================

STEP1_SYSTEM_PROMPT = """你是 PAO 化工优化决策 Agent（第一阶段：选择动作类型）。

根据当前优化状态，选择下一步动作类型：

**动作类型说明：**

1. **continue** - 继续当前搜索策略
   - 适用：当前策略有效，收敛率和可行率正常，需要更多数据点
   - 不改变搜索区域

2. **set_region** - 调整搜索区域
   - 适用：需要扩大/缩小/移动搜索范围
   - 示例：收敛率低 → 缩小区域；停滞 → 扩大区域

3. **probe** - 探测特定候选点
   - 适用：有明确假设需要验证某个特定参数组合
   - 示例：测试边界点、验证失败原因

4. **stop** - 停止优化
   - 适用：已达目标、预算即将耗尽、或判断无进展可能

**决策依据：**
- recent_convergence_rate: 近期收敛率
- recent_feasible_rate: 近期可行率
- stagnation_count: 停滞批次数
- budget_remaining: 剩余预算
- has_feasible: 是否有可行解

**输出格式（严格 JSON）：**
{
  "action_type": "continue",
  "reasoning": "当前收敛率 85%，可行率 70%，策略有效，继续收集数据"
}

注意：只输出 JSON 对象，不要输出任何其他内容。
"""

# ============================================================================
# Step 2: 具体参数生成 Prompts（按动作类型区分）
# ============================================================================

STEP2_CONTINUE_PROMPT = """你已选择 continue（继续当前策略）。现在生成具体参数。

**输出格式（严格 JSON）：**
{
  "hypothesis": "本批实验的假设（为什么继续当前策略）",
  "evidence_case_ids": ["case_123", "case_456"],
  "knowledge_ids": ["rule_001"],
  "batch_size": 5,
  "expected_effect": "objective",
  "initialization": "reset"
}

**字段约束：**
- hypothesis: 必填，说明本批实验的目的
- evidence_case_ids: 引用的历史工况 ID（至少 1 个）
- knowledge_ids: 应用的知识规则 ID（可选）
- batch_size: 不超过配置上限（snapshot 中的 batch_size_limit）
- expected_effect: 只能是 feasibility/convergence/objective/information 之一
- initialization: 只能是 reset 或 previous

注意：只输出 JSON 对象，不要输出 Markdown 围栏或其他内容。
"""

STEP2_SET_REGION_PROMPT_TEMPLATE = """你已选择 set_region（调整搜索区域）。现在生成区域调整方案。

**当前边界信息：**
{bounds_info}

**输出格式（严格 JSON）：**
{{
  "hypothesis": "为什么调整搜索区（如：缩小范围提高收敛率）",
  "evidence_case_ids": ["case_123"],
  "knowledge_ids": ["rule_002"],
  "search_region": {{
    "T1_RR": [0.6, 1.2],
    "T1_NSTAGE": [35, 55]
  }},
  "batch_size": 5,
  "expected_effect": "convergence",
  "initialization": "reset"
}}

**字段约束：**
- search_region: 必填，键为变量名或 Aspen 路径，值为 [下界, 上界]
- 新边界必须在硬边界（hard_bounds）内
- 整数变量必须确保至少包含一个整数（如 [35, 55] 包含 35, 36, ..., 55）
- 不要填写 candidate 字段

注意：只输出 JSON 对象，不要输出 Markdown 围栏或其他内容。
"""

STEP2_PROBE_PROMPT_TEMPLATE = """你已选择 probe（探测特定点）。现在生成要探测的候选点。

**当前变量和边界：**
{bounds_info}

**输出格式（严格 JSON）：**
{{
  "hypothesis": "为什么探测这个点（如：验证边界点收敛性）",
  "evidence_case_ids": ["case_123"],
  "knowledge_ids": [],
  "candidate": {{
    "T1_RR": 0.85,
    "T1_NSTAGE": 45,
    "T1_FEED_STAGE_FRAC": 0.5
  }},
  "expected_effect": "information",
  "initialization": "reset",
  "repeat": false
}}

**字段约束：**
- candidate: 必填，必须包含所有优化变量，且仅包含这些变量
- 数值必须在硬边界（hard_bounds）内
- 整数变量必须是整数值（如 45，不能是 45.5）
- repeat: 是否重复验证（true/false）
- 不要填写 search_region 字段

注意：只输出 JSON 对象，不要输出 Markdown 围栏或其他内容。
"""

STEP2_STOP_PROMPT = """你已选择 stop（停止优化）。说明停止理由。

**输出格式（严格 JSON）：**
{
  "hypothesis": "停止理由（如：已达目标/预算耗尽/无进展可能）",
  "evidence_case_ids": ["case_xxx"],
  "knowledge_ids": []
}

注意：只输出 JSON 对象，不要输出 Markdown 围栏或其他内容。
"""


# ============================================================================
# TwoStepDecisionAgent
# ============================================================================

class TwoStepDecisionAgent:
    """
    两阶段决策 Agent。

    工作流程：
    1. Step 1: 分析状态 → 选择动作类型
    2. Step 2: 根据类型 → 生成具体参数
    3. 任一步失败 → 回退到规则决策

    Args:
        llm_config: LLM 配置（None 时直接使用规则降级）
    """

    def __init__(self, llm_config=None):
        self.llm_config = llm_config

    def propose(self, snapshot: dict) -> tuple[Action, str]:
        """
        两阶段生成决策。

        Args:
            snapshot: 当前优化状态快照（包含历史、边界、预算等）

        Returns:
            (Action, source) 元组
            - Action: 具体的动作对象（ContinueAction/SetRegionAction/ProbeAction/StopAction）
            - source: 决策来源（"llm:two_step" 或 "rules:fallback"）
        """
        from src.agents.lead.llm_client import is_configured, load_llm_config

        cfg = self.llm_config or load_llm_config()
        if not is_configured(cfg):
            _log.info("LLM 未配置，直接使用规则降级")
            return self._fallback(snapshot), "rules:fallback"

        try:
            # Step 1: 选择动作类型
            action_type = self._step1_select_action(cfg, snapshot)
            _log.info("Step 1 选择动作类型：%s", action_type)

            # Step 2: 生成具体参数
            action = self._step2_generate_params(cfg, snapshot, action_type)
            _log.info("Step 2 生成参数完成：%s", action.__class__.__name__)

            return action, "llm:two_step"

        except Exception as exc:
            _log.warning("Two-step 决策失败，回退到规则：%s", exc, exc_info=True)
            return self._fallback(snapshot), "rules:fallback"

    def _step1_select_action(
        self, cfg, snapshot: dict
    ) -> Literal["continue", "set_region", "probe", "stop"]:
        """
        Step 1: 选择动作类型。

        输入：轻量级状态摘要（只包含关键指标）
        输出：动作类型（continue/set_region/probe/stop）
        """
        from src.agents.lead.llm_client import chat

        # 构造轻量级输入（避免发送完整历史）
        step1_input = {
            "recent_convergence_rate": snapshot.get("recent_convergence_rate", 0.0),
            "recent_feasible_rate": snapshot.get("recent_feasible_rate", 0.0),
            "stagnation_count": snapshot.get("stagnation_count", 0),
            "budget_remaining": snapshot.get("budget_remaining", 0),
            "has_feasible": bool([
                r for r in snapshot.get("recent_observations", [])
                if r.get("success")
            ]),
            "total_evaluations": len(snapshot.get("observations", [])),
        }

        raw = chat(cfg, system=STEP1_SYSTEM_PROMPT, user=json.dumps(step1_input, ensure_ascii=False))
        _log.debug("Step 1 LLM 原始输出：%s", raw[:200])

        # 解析 JSON
        data = json.loads(self._extract_json(raw))
        action_type = data.get("action_type", "continue")
        reasoning = data.get("reasoning", "")

        if action_type not in {"continue", "set_region", "probe", "stop"}:
            raise ValueError(f"无效的 action_type: {action_type!r}")

        _log.debug("Step 1 推理：%s", reasoning)
        return action_type

    def _step2_generate_params(
        self, cfg, snapshot: dict, action_type: str
    ) -> Action:
        """
        Step 2: 生成具体参数。

        根据动作类型使用不同的 prompt 和验证规则。
        """
        from src.agents.lead.llm_client import chat

        # 选择对应的 prompt
        if action_type == "continue":
            prompt = STEP2_CONTINUE_PROMPT
        elif action_type == "set_region":
            bounds_info = self._format_bounds_info(snapshot)
            prompt = STEP2_SET_REGION_PROMPT_TEMPLATE.format(bounds_info=bounds_info)
        elif action_type == "probe":
            bounds_info = self._format_bounds_info(snapshot)
            prompt = STEP2_PROBE_PROMPT_TEMPLATE.format(bounds_info=bounds_info)
        else:  # stop
            prompt = STEP2_STOP_PROMPT

        # 调用 LLM（需要序列化 snapshot，将 set 转换为 list）
        snapshot_serializable = self._make_serializable(snapshot)
        raw = chat(cfg, system=prompt, user=json.dumps(snapshot_serializable, ensure_ascii=False))
        _log.debug("Step 2 LLM 原始输出：%s", raw[:200])

        # 解析并构造对应的 Action
        data = json.loads(self._extract_json(raw))

        # 根据类型构造 Action 对象
        if action_type == "continue":
            return ContinueAction(
                hypothesis=data["hypothesis"],
                evidence_case_ids=data.get("evidence_case_ids", []),
                knowledge_ids=data.get("knowledge_ids", []),
                batch_size=data.get("batch_size", 5),
                expected_effect=data.get("expected_effect", "objective"),
                initialization=data.get("initialization", "reset"),
            )
        elif action_type == "set_region":
            return SetRegionAction(
                hypothesis=data["hypothesis"],
                evidence_case_ids=data.get("evidence_case_ids", []),
                knowledge_ids=data.get("knowledge_ids", []),
                search_region=data.get("search_region", {}),
                batch_size=data.get("batch_size", 5),
                expected_effect=data.get("expected_effect", "convergence"),
                initialization=data.get("initialization", "reset"),
            )
        elif action_type == "probe":
            return ProbeAction(
                hypothesis=data["hypothesis"],
                evidence_case_ids=data.get("evidence_case_ids", []),
                knowledge_ids=data.get("knowledge_ids", []),
                candidate=data.get("candidate", {}),
                repeat=data.get("repeat", False),
                expected_effect=data.get("expected_effect", "information"),
                initialization=data.get("initialization", "reset"),
            )
        else:  # stop
            return StopAction(
                hypothesis=data["hypothesis"],
                evidence_case_ids=data.get("evidence_case_ids", []),
                knowledge_ids=data.get("knowledge_ids", []),
            )

    def _extract_json(self, text: str) -> str:
        """
        提取 JSON 对象（容错处理）。

        支持：
        - 纯 JSON 对象
        - Markdown 围栏包裹的 JSON
        - 带前后缀文本的 JSON
        """
        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("LLM 返回为空")

        # 移除 Markdown 围栏
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()

        # 提取第一个 JSON 对象
        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char == "{":
                try:
                    value, _ = decoder.raw_decode(cleaned[index:])
                    if isinstance(value, dict):
                        return json.dumps(value, ensure_ascii=False)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"未找到合法 JSON 对象：{cleaned[:240]!r}")

    def _make_serializable(self, obj):
        """
        将对象转换为 JSON 可序列化格式（处理 set 等类型）。
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj

    def _format_bounds_info(self, snapshot: dict) -> str:
        """格式化边界信息（供 prompt 使用）"""
        lines = ["硬边界（不可突破）："]
        hard_bounds = snapshot.get("hard_bounds", {})
        integer_paths = snapshot.get("integer_paths", set())

        for key, (lo, hi) in hard_bounds.items():
            is_int = key in integer_paths
            type_info = "（整数）" if is_int else ""
            lines.append(f"  {key}: [{lo}, {hi}]{type_info}")

        search_region = snapshot.get("search_region", {})
        if search_region:
            lines.append("\n当前软搜索区：")
            for key, (lo, hi) in search_region.items():
                lines.append(f"  {key}: [{lo}, {hi}]")

        anchor = snapshot.get("anchor", {})
        if anchor:
            lines.append("\n基准点（收敛且可行）：")
            for key, val in anchor.items():
                lines.append(f"  {key}: {val}")

        return "\n".join(lines)

    def _fallback(self, snapshot: dict) -> Action:
        """
        规则降级（复用现有 fallback_plan 逻辑）。

        当 LLM 不可用或失败时使用确定性规则生成决策。
        """
        from .advisor import fallback_plan

        legacy_plan = fallback_plan(snapshot, snapshot.get("batch_size_limit", 5))

        # 转换为新的 Action 类型
        if legacy_plan.action == "set_region":
            return SetRegionAction(
                hypothesis=legacy_plan.hypothesis,
                evidence_case_ids=legacy_plan.evidence_case_ids,
                knowledge_ids=legacy_plan.knowledge_ids,
                search_region=legacy_plan.search_region,
                batch_size=legacy_plan.batch_size,
                expected_effect=legacy_plan.expected_effect,
                initialization=legacy_plan.initialization,
            )
        else:  # continue
            return ContinueAction(
                hypothesis=legacy_plan.hypothesis,
                evidence_case_ids=legacy_plan.evidence_case_ids,
                knowledge_ids=legacy_plan.knowledge_ids,
                batch_size=legacy_plan.batch_size,
                expected_effect=legacy_plan.expected_effect,
                initialization=legacy_plan.initialization,
            )
