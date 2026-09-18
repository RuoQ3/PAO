"""
coordinator — 顶层意图编排 agent。

三层结构：意图识别（intent.py）→ 任务拆解（planner.py）→ 执行分发（dispatcher.py），
统一入口见 run_coordinator。把自由文本请求路由到项目已有的子 agent
（onboarding_agent / boundary_advisor / kinetics_advisor / process_advisor /
optimize_pareto_tool），不重新实现任何业务逻辑。

已知范围限定（见 agent.py 模块 docstring）：不做任务间数值结果自动传递，
不做 write_feasibility 验证。文件路径层面的传递在连续对话模式下支持
（run_coordinator_turn + ChatSessionState）。

公开接口：
  run_coordinator       一次性调用入口，串联三层，无跨轮记忆
  run_coordinator_turn  连续对话模式单轮入口，自动记忆/填空文件路径
  CoordinatorResult      完整输出（含三层中间产物）
  classify_intent / classify_intent_agent   意图识别（规则/LLM）
  build_plan             任务拆解
  dispatch_plan          执行分发
"""
from src.agents.coordinator.agent import CoordinatorResult, run_coordinator, run_coordinator_turn
from src.agents.coordinator.dispatcher import dispatch_plan
from src.agents.coordinator.intent import classify_intent, classify_intent_agent
from src.agents.coordinator.planner import build_plan
from src.models.coordinator import ChatSessionState

__all__ = [
    "run_coordinator",
    "run_coordinator_turn",
    "ChatSessionState",
    "CoordinatorResult",
    "classify_intent",
    "classify_intent_agent",
    "build_plan",
    "dispatch_plan",
]
