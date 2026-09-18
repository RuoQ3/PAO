"""
agent.py — Coordinator agent 顶层入口。

串联三层：
  intent.py     意图识别（自由文本 → 有序的 IntentStep 列表）
  planner.py    任务拆解（IntentStep + 环境参数 → 可执行的 TaskSpec 列表）
  dispatcher.py 执行分发（依次执行每个 TaskSpec，产出 ExecutionState）

自由对话支持
------------
不是所有请求都要路由到一个具体动作——GENERAL_CHAT 类型专门承接知识性
问答、方案讨论、闲聊（如"RCSTR 和 RPLUG 有什么区别"），直接由 LLM 用
化工/优化领域背景回答，不驱动任何工具、不产出文件路径。这条路由与
CLARIFY_NEEDED 的边界：前者是"有明确答案的问题"，后者是"用户像是想
执行某个动作但看不出是哪种"——分类 prompt（prompts.py）里有详细区分。

已知范围限定（v1，骨架阶段的取舍）
--------------------------------
不做任务间数值结果自动传递。复合请求（如"先拟合动力学参数，再基于结果推荐
边界"）会被拆成多个独立任务依次执行，但第一步的拟合数值不会自动喂给第二步
的输入——每个任务各自产出报告，数值层面的串联仍需人工看完结果后决定下一步
的参数。

文件路径层面的传递是支持的，且有两层：
  1. 会话跨轮记忆（连续对话模式）：run_coordinator_turn 配合 ChatSessionState
     记住上一轮任务产出的文件路径（如 onboarding 生成的配置 YAML），下一轮
     用户提问缺少路径时自动填空。同时把会话状态摘要注入意图分类的 LLM
     prompt（见 intent.classify_intent_agent 的 session_context 参数），
     使"这个/这个工艺"等指代词能被正确理解，而不会被误判为信息不足。
  2. 单次调用内的链式推理（gap-fill）：某个任务缺配置文件、但 aspen_file
     已知时，dispatcher.py 会自动在它前面插入一个 ONBOARD_NEW_CASE 任务，
     执行成功后动态解锁被阻塞的任务，不需要用户额外发一轮请求确认接入。
     唯一的例外是 RUN_OPTIMIZATION：即使配置已经自动生成，也不会自动
     开始仿真，而是停在确认点，把是否启动优化的决定权交还给用户
     （因为跑优化会真实驱动 Aspen 做多次仿真，且用的边界还没人工看过）。
一次性调用（run_coordinator）同样支持 gap-fill（单次调用内链式推理），
但不涉及跨轮记忆，每次调用都是全新的、无状态的。

不做 write_feasibility 验证。ONBOARD_NEW_CASE 路由只调用 run_onboarding
产出草案，不做 COM 试写验证（那是 graph.py 状态机独有的一步）；需要更严格
验证时应走完整的 graph.py/hitl_protocol 流程，而不是这个骨架入口。

安全边界
--------
本模块本身不驱动 Aspen COM、不写数据库——所有实际执行都委托给
dispatcher.py 里封装的现有子 agent 入口，Coordinator 层只做路由和编排。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.coordinator.dispatcher import dispatch_plan
from src.agents.coordinator.intent import classify_intent_agent
from src.agents.coordinator.planner import build_plan, make_env
from src.models.coordinator import ChatSessionState, ExecutionState, IntentClassification, Plan


@dataclass
class CoordinatorResult:
    """run_coordinator 的完整输出，保留三层的中间产物供调试/报告展示。"""
    classification: IntentClassification
    plan: Plan
    execution: ExecutionState


def run_coordinator(
    user_text: str,
    *,
    aspen_file: str | None = None,
    case_config_path: str | None = None,
    data_csv: str | None = None,
    kinetics_variables_yaml: str | None = None,
    db_path: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    llm_config=None,
) -> CoordinatorResult:
    """
    对一次用户请求执行完整的意图识别 → 任务拆解 → 执行分发流程。

    Parameters
    ----------
    user_text:
        用户的自由文本请求，如"帮我拟合这批实验数据的活化能"。
    aspen_file / case_config_path / data_csv / kinetics_variables_yaml / db_path:
        CLI/调用方显式传入的环境参数（文件路径），优先于从 user_text 中
        抽取的槎位。均为可选——具体某个路由目标是否需要它们由 planner.py
        的前提校验决定，缺失时对应任务会被标记 ready=False 并跳过，
        不会抛异常中断整体流程。
    model / provider / llm_config:
        统一的 LLM 配置，传给意图分类层和所有需要 LLM 的子 agent。

    Returns
    -------
    CoordinatorResult
        含三层的完整中间产物。execution.has_clarify=True 时，调用方应
        优先展示 CLARIFY_NEEDED 任务的 blocking_reason 并向用户追问，
        而不是简单报告"部分任务被跳过"。
    """
    classification = classify_intent_agent(
        user_text, model=model, provider=provider, llm_config=llm_config,
    )

    plan = build_plan(
        classification,
        user_text=user_text,
        aspen_file=aspen_file,
        case_config_path=case_config_path,
        data_csv=data_csv,
        kinetics_variables_yaml=kinetics_variables_yaml,
        db_path=db_path,
    )

    env = make_env(
        aspen_file=aspen_file, case_config_path=case_config_path,
        data_csv=data_csv, kinetics_variables_yaml=kinetics_variables_yaml,
        db_path=db_path,
    )
    execution = dispatch_plan(plan, llm_config=llm_config, env=env)

    return CoordinatorResult(classification=classification, plan=plan, execution=execution)


def run_coordinator_turn(
    session: ChatSessionState,
    user_text: str,
    *,
    data_csv: str | None = None,
    kinetics_variables_yaml: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    llm_config=None,
) -> CoordinatorResult:
    """
    连续对话模式（--chat）下的单轮请求处理。

    与 run_coordinator 的区别：
      1. 缺失的 case_config_path / db_path 会先用 session 里记住的上一轮
         产出路径填空，用户不用在每一轮重复给同一个路径。
      2. 这一轮如果显式传入了新路径（data_csv/kinetics_variables_yaml，
         或用户在 user_text 里明确提到的路径被 Planner 解析出来），
         以本轮为准；session 只在缺失时兜底，不会覆盖本轮的显式输入。
      3. 执行完成后，把本轮所有任务产出的路径（CoordinatorStep.produced_paths）
         合并进 session，供下一轮使用；session.turn_count 自增。

    Parameters
    ----------
    session:
        跨轮次保留的会话状态，调用方负责在多轮之间持有同一个实例。
    user_text:
        本轮用户的自由文本请求。
    data_csv / kinetics_variables_yaml:
        本轮显式提供的路径（CLI 参数），不参与会话记忆（拟合类任务本身
        不产出新路径，见 ChatSessionState 的字段说明）。
    model / provider / llm_config:
        同 run_coordinator。

    Returns
    -------
    CoordinatorResult
        与 run_coordinator 返回类型一致，调用方可用同一套报告展示逻辑。
    """
    classification = classify_intent_agent(
        user_text, session_context=session.to_context_summary(),
        model=model, provider=provider, llm_config=llm_config,
    )

    effective_data_csv = data_csv or session.data_csv
    plan = build_plan(
        classification,
        user_text=user_text,
        aspen_file=session.aspen_file,
        case_config_path=session.case_config_path,
        data_csv=effective_data_csv,
        kinetics_variables_yaml=kinetics_variables_yaml,
        db_path=session.db_path,
    )

    env = make_env(
        aspen_file=session.aspen_file, case_config_path=session.case_config_path,
        data_csv=effective_data_csv, kinetics_variables_yaml=kinetics_variables_yaml,
        db_path=session.db_path,
    )
    execution = dispatch_plan(plan, llm_config=llm_config, env=env)

    # 把本轮新产出的路径合并进会话记忆，供下一轮填空
    for step in execution.steps:
        if step.produced_paths:
            session.apply_produced_paths(step.produced_paths)
    if data_csv:
        session.data_csv = data_csv
    session.turn_count += 1

    return CoordinatorResult(classification=classification, plan=plan, execution=execution)
