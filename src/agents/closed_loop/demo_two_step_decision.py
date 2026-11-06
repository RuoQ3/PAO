"""
demo_two_step_decision.py — 演示两阶段决策功能

这个脚本演示如何使用新的 Two-Step 决策 Agent。

运行方式：
    python -m src.agents.closed_loop.demo_two_step_decision
"""
from __future__ import annotations

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_log = logging.getLogger(__name__)


def create_demo_snapshot() -> dict:
    """创建演示用的优化状态快照"""
    return {
        # 关键指标
        "recent_convergence_rate": 0.75,
        "recent_feasible_rate": 0.60,
        "stagnation_count": 0,
        "budget_remaining": 20,
        "batch_size_limit": 5,

        # 变量边界
        "hard_bounds": {
            "T1_RR": [0.4, 2.0],
            "T1_NSTAGE": [30, 70],
            "T1_FEED_STAGE_FRAC": [0.3, 0.7],
        },
        "search_region": {
            "T1_RR": [0.7, 1.2],
            "T1_NSTAGE": [40, 60],
            "T1_FEED_STAGE_FRAC": [0.4, 0.6],
        },
        "anchor": {
            "T1_RR": 0.85,
            "T1_NSTAGE": 50,
            "T1_FEED_STAGE_FRAC": 0.5,
        },
        "integer_paths": {"T1_NSTAGE"},

        # 历史观测
        "recent_observations": [
            {"case_id": "case_001", "success": True, "converged": True},
            {"case_id": "case_002", "success": True, "converged": True},
            {"case_id": "case_003", "success": False, "converged": False},
        ],
        "observations": [{"case_id": f"case_{i:03d}"} for i in range(1, 16)],

        # 问题定义（用于案例检索）
        "problem_definition": {
            "objectives": ["CAPEX", "EMISSIONS"],
            "constraints": ["product_purity >= 0.999"],
            "design_variables": ["T1_RR", "T1_NSTAGE", "T1_FEED_STAGE_FRAC"],
        },
    }


def demo_basic_usage():
    """演示 1：基本使用"""
    print("\n" + "=" * 60)
    print("演示 1：基本使用 - Two-Step 决策")
    print("=" * 60)

    from src.agents.closed_loop.advisor import create_decision_agent

    # 创建 agent（使用规则降级，无需 LLM）
    agent = create_decision_agent(llm_config=None, use_two_step=True)

    # 准备状态快照
    snapshot = create_demo_snapshot()

    # 生成决策
    print("\n[状态] 当前状态：")
    print(f"  收敛率: {snapshot['recent_convergence_rate']:.0%}")
    print(f"  可行率: {snapshot['recent_feasible_rate']:.0%}")
    print(f"  停滞批次: {snapshot['stagnation_count']}")
    print(f"  剩余预算: {snapshot['budget_remaining']}")

    print("\n[Agent] Agent 决策中...")
    action, source = agent.propose(snapshot)

    print(f"\n[OK] 决策完成")
    print(f"  来源: {source}")
    print(f"  动作类型: {action.action}")
    print(f"  假设: {action.hypothesis}")

    # 转换为旧格式
    legacy_plan = action.to_legacy_plan()
    print(f"\n[计划] 旧格式 ActionPlan:")
    print(json.dumps(legacy_plan, ensure_ascii=False, indent=2))


def demo_with_explainability():
    """演示 2：决策可解释性"""
    print("\n" + "=" * 60)
    print("演示 2：决策可解释性")
    print("=" * 60)

    from src.agents.closed_loop.advisor import create_decision_agent
    from src.agents.closed_loop.explainability import build_trace_from_snapshot

    agent = create_decision_agent(llm_config=None, use_two_step=True)
    snapshot = create_demo_snapshot()

    # 生成决策
    action, source = agent.propose(snapshot)

    # 构建决策追踪
    trace = build_trace_from_snapshot(
        decision_id="demo_001",
        action_plan=action.to_legacy_plan(),
        snapshot=snapshot,
    )

    # 生成自然语言解释
    explanation = trace.to_natural_language()
    print("\n" + explanation)

    # 检查是否需要人工介入
    if trace.should_request_human():
        print("\n[警告]  置信度较低，建议人工确认")
    else:
        print("\n[OK] 置信度足够，可以直接执行")


def demo_case_memory():
    """演示 3：案例记忆"""
    print("\n" + "=" * 60)
    print("演示 3：案例记忆库")
    print("=" * 60)

    from src.agents.closed_loop.case_memory import CaseMemory, OptimizationCase
    from datetime import datetime

    # 创建内存案例库（不持久化）
    import tempfile
    import os
    temp_path = os.path.join(tempfile.gettempdir(), "demo_case_library.json")

    memory = CaseMemory(temp_path)

    # 添加演示案例
    demo_case = OptimizationCase(
        case_id="demo_case_001",
        problem={
            "objectives": ["CAPEX", "EMISSIONS"],
            "constraints": ["purity >= 0.999"],
            "design_variables": ["T1_RR", "T1_NSTAGE"],
        },
        actions=[
            {"action": "continue", "batch_size": 5},
            {"action": "set_region", "search_region": {"T1_RR": [0.6, 1.0]}},
            {"action": "continue", "batch_size": 5},
        ],
        final_result="target_verified",
        metrics={"hypervolume": 0.85, "convergence_rate": 0.90},
        timestamp=datetime.now().isoformat(),
        tags=["distillation", "multi_objective"],
    )

    memory.add_case(demo_case)
    print(f"[OK] 添加案例: {demo_case.case_id}")

    # 检索相似案例
    snapshot = create_demo_snapshot()
    similar = memory.retrieve_similar_cases(
        problem=snapshot["problem_definition"],
        top_k=3,
        min_similarity=0.3,
    )

    print(f"\n[搜索] 检索到 {len(similar)} 个相似案例:")
    for case, similarity in similar:
        print(f"  - {case.case_id}: 相似度 {similarity:.0%}")
        print(f"    最终结果: {case.final_result}")
        print(f"    动作数量: {len(case.actions)}")

    # 统计信息
    stats = memory.get_statistics()
    print(f"\n[状态] 案例库统计:")
    print(f"  总案例数: {stats['total_cases']}")
    print(f"  成功案例: {stats['successful_cases']}")
    print(f"  标签: {', '.join(stats['tags'])}")

    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)


def demo_action_types():
    """演示 4：类型安全的 Action"""
    print("\n" + "=" * 60)
    print("演示 4：类型安全的 Action 定义")
    print("=" * 60)

    from src.agents.closed_loop.action_types import (
        ContinueAction,
        SetRegionAction,
        ProbeAction,
        StopAction,
    )

    # 创建不同类型的动作
    actions = [
        ContinueAction(
            hypothesis="继续当前策略",
            evidence_case_ids=["case_001", "case_002"],
            batch_size=5,
        ),
        SetRegionAction(
            hypothesis="缩小搜索范围",
            search_region={"T1_RR": [0.7, 1.0]},
            batch_size=5,
        ),
        ProbeAction(
            hypothesis="验证边界点",
            candidate={"T1_RR": 0.8, "T1_NSTAGE": 45},
            repeat=False,
        ),
        StopAction(
            hypothesis="已达目标",
            evidence_case_ids=["case_final"],
        ),
    ]

    print("\n[计划] 四种动作类型演示:\n")
    for i, action in enumerate(actions, 1):
        print(f"{i}. {action.__class__.__name__}")
        print(f"   动作: {action.action}")
        print(f"   假设: {action.hypothesis}")

        # 转换为旧格式
        legacy = action.to_legacy_plan()

        # 验证字段互斥性
        if action.action == "continue":
            assert legacy["search_region"] == {}, "continue 不应有 search_region"
            assert legacy["candidate"] == {}, "continue 不应有 candidate"
            print("   [OK] 字段互斥性验证通过")

        elif action.action == "set_region":
            assert legacy["search_region"], "set_region 应有 search_region"
            assert legacy["candidate"] == {}, "set_region 不应有 candidate"
            print(f"   [OK] search_region: {legacy['search_region']}")

        elif action.action == "probe":
            assert legacy["candidate"], "probe 应有 candidate"
            assert legacy["search_region"] == {}, "probe 不应有 search_region"
            print(f"   [OK] candidate: {legacy['candidate']}")

        print()


def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("Two-Step 决策功能演示")
    print("=" * 60)

    try:
        demo_basic_usage()
        demo_with_explainability()
        demo_case_memory()
        demo_action_types()

        print("\n" + "=" * 60)
        print("[OK] 所有演示完成！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 查看使用文档: docs/agent_upgrade_guide.md")
        print("2. 运行测试: pytest tests/agents/closed_loop/")
        print("3. 在实际优化中启用: agent_loop.use_two_step_decision = true")

    except Exception as exc:
        _log.error("演示失败: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
