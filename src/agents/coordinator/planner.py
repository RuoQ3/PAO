"""
planner.py — Coordinator agent 的任务拆解层。

职责：把 IntentClassification 的每个 IntentStep 转成一个可执行的 TaskSpec：
  - 补全技术参数（node_db_path/db_path 等用户不会在对话里提到的东西）。
  - 做前提校验：缺少必需的文件路径时，标 ready=False 并给出 blocking_reason，
    绝不用不完整参数硬凑一次执行。

设计原则
--------
  - 纯函数，不调用任何 LLM、不驱动 Aspen COM、不执行任何子 agent。
  - 路径校验只做"文件是否存在"这类只读检查（与 demo_workflow/helpers.py 的
    _resolve_config_path 同一范式），不解析 YAML 内容、不做语义校验——
    "配置里到底有没有 design_variables" 这类更深的校验留给 Dispatcher
    调用真实工具时暴露（工具本身已经有校验逻辑，不需要 Planner 重新实现一遍）。
  - CLI/环境传入的显式路径参数（env 字典）优先于从用户文本抽取的 raw_slots——
    后者是 LLM 从对话里猜的，前者是用户在命令行明确给定的，可信度更高。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.models.coordinator import IntentClassification, IntentStep, IntentType, Plan, TaskSpec

# 项目根目录（src/agents/coordinator/planner.py → 上四层）
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent


def _resolve_existing_path(raw: str | None) -> str | None:
    """把候选路径解析为一个已存在的绝对路径字符串；找不到返回 None。

    解析优先级：绝对路径 → 相对于当前工作目录 → 相对于项目根目录。
    """
    if not raw or not raw.strip():
        return None
    p = Path(raw.strip())
    if p.is_absolute():
        return str(p) if p.exists() else None
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return str(from_cwd.resolve())
    from_root = _PROJECT_ROOT / p
    if from_root.exists():
        return str(from_root.resolve())
    return None


def _derive_node_db_path(aspen_file_path: str) -> str:
    """从 Aspen 文件路径推断 node_db_path：{文件所在目录}/output/node.db。"""
    return str(Path(aspen_file_path).parent / "output" / "node.db")


def _derive_db_path(config_path: str) -> str:
    """从配置文件路径推断 SimulationDB 路径：{配置目录}/output/simulation.db。"""
    return str(Path(config_path).parent / "output" / "simulation.db")


def make_env(
    *,
    aspen_file: str | None = None,
    case_config_path: str | None = None,
    data_csv: str | None = None,
    kinetics_variables_yaml: str | None = None,
    db_path: str | None = None,
) -> dict:
    """构造 builder 函数使用的 env 字典（公开，供 agent.py 和 dispatcher.py 复用同一结构）。

    dispatcher.py 在 gap-fill 补上配置路径后需要用同样结构的 env 重新调用
    builder（见 rebuild_task），因此这个构造逻辑不能只是 build_plan 内部的
    私有细节，必须暴露成单一来源，避免两处各写一份字典结构走样。
    """
    return {
        "aspen_file": aspen_file,
        "case_config_path": case_config_path,
        "data_csv": data_csv,
        "kinetics_variables_yaml": kinetics_variables_yaml,
        "db_path": db_path,
    }


# ---------------------------------------------------------------------------
# 各路由目标的任务构造
# ---------------------------------------------------------------------------
# 每个 builder 签名一致：(step, env) -> (kwargs, ready, blocking_reason, waiting_on_gapfill)
#
# waiting_on_gapfill=True 表示：这个任务之所以 ready=False，纯粹是因为
# "还没有配置文件"，而配置文件本身是可以自动生成的（只要 aspen_file 已知）。
# Dispatcher 看到这个标记会自动在它前面插入一个 ONBOARD_NEW_CASE gap-fill
# 任务，执行成功后动态解锁本任务，不需要用户再发一轮请求。
#
# 不是所有"缺文件"都能这样兜底——比如 fit_kinetics_params 缺的是用户的
# 实验数据 CSV，系统无法自己生成一份数据出来，所以那些 builder 永远
# waiting_on_gapfill=False，缺了就是真的缺，只能让用户提供。

_Env = dict
_GAPFILL_TARGET = IntentType.ONBOARD_NEW_CASE


def _build_onboard_task(step: IntentStep, env: _Env) -> tuple[dict, bool, str, bool]:
    aspen_file = env.get("aspen_file") or step.raw_slots.get("aspen_file")
    resolved = _resolve_existing_path(aspen_file)
    if resolved is None:
        return (
            {}, False,
            "需要一个可用的 Aspen 仿真文件路径（.bkp/.apw），未在请求或命令行参数中找到有效文件",
            False,  # 没有 aspen_file 本身就是所有 gap-fill 的起点，无法再往前补
        )
    kwargs = {
        "aspen_file_path": resolved,
        "intent_text": step.raw_slots.get("context", ""),
        "node_db_path": _derive_node_db_path(resolved),
    }
    return kwargs, True, "", False


def _build_recommend_bounds_task(step: IntentStep, env: _Env) -> tuple[dict, bool, str, bool]:
    config_path = env.get("case_config_path") or step.raw_slots.get("case_config_path")
    resolved = _resolve_existing_path(config_path)
    if resolved is None:
        can_gapfill = _resolve_existing_path(env.get("aspen_file") or step.raw_slots.get("aspen_file")) is not None
        return (
            {}, False,
            "需要一份已有的优化配置 YAML（含 design_variables），未找到有效路径；"
            "如果还没有配置，将自动先接入生成草案" if can_gapfill else
            "需要一份已有的优化配置 YAML（含 design_variables），未找到有效路径；"
            "且当前没有可用的 Aspen 文件，无法自动生成，请先提供配置或 Aspen 文件",
            can_gapfill,
        )
    kwargs = {
        "config_path": resolved,
        "context": step.raw_slots.get("context", ""),
    }
    return kwargs, True, "", False


def _build_fit_kinetics_task(step: IntentStep, env: _Env) -> tuple[dict, bool, str, bool]:
    data_csv = env.get("data_csv") or step.raw_slots.get("data_csv")
    resolved = _resolve_existing_path(data_csv)
    if resolved is None:
        return (
            {}, False,
            "需要一份实验数据 CSV 文件（列：temperature,rate_constant），未找到有效路径",
            False,  # 实验数据只能由用户提供，无法自动生成
        )
    kwargs = {
        "data_csv": resolved,
        "var_name": step.raw_slots.get("var_name", "kinetics_param"),
        "context": step.raw_slots.get("context", ""),
    }
    return kwargs, True, "", False


def _build_recommend_kinetics_bounds_task(step: IntentStep, env: _Env) -> tuple[dict, bool, str, bool]:
    variables_yaml = env.get("kinetics_variables_yaml") or step.raw_slots.get("kinetics_variables_yaml")
    resolved = _resolve_existing_path(variables_yaml)
    if resolved is None:
        return (
            {}, False,
            "需要一份动力学参数元信息清单（YAML，格式见 "
            "scripts/run_kinetics_advisor.py bounds 子命令说明），未找到有效路径",
            False,  # 变量清单只能由用户提供，无法自动生成
        )
    kwargs = {
        "variables_yaml": resolved,
        "context": step.raw_slots.get("context", ""),
    }
    return kwargs, True, "", False


def _build_diagnose_task(step: IntentStep, env: _Env) -> tuple[dict, bool, str, bool]:
    config_path = env.get("case_config_path") or step.raw_slots.get("case_config_path")
    resolved = _resolve_existing_path(config_path)
    if resolved is None:
        can_gapfill = _resolve_existing_path(env.get("aspen_file") or step.raw_slots.get("aspen_file")) is not None
        return (
            {}, False,
            "需要一份优化配置 YAML 才能定位对应的结果数据库，未找到有效路径；"
            "将自动先接入生成草案" if can_gapfill else
            "需要一份优化配置 YAML 才能定位对应的结果数据库，未找到有效路径，"
            "且当前没有可用的 Aspen 文件，无法自动生成",
            can_gapfill,
        )
    kwargs = {
        "case_config_path": resolved,
        "db_path": env.get("db_path"),  # None 时 run_process_advisor_agent 自动从配置推断
        "mode": "db",
    }
    return kwargs, True, "", False


def _build_run_optimization_task(step: IntentStep, env: _Env) -> tuple[dict, bool, str, bool]:
    config_path = env.get("case_config_path") or step.raw_slots.get("case_config_path")
    resolved = _resolve_existing_path(config_path)
    if resolved is None:
        can_gapfill = _resolve_existing_path(env.get("aspen_file") or step.raw_slots.get("aspen_file")) is not None
        return (
            {}, False,
            "需要一份已通过校验的优化配置 YAML 才能启动优化，未找到有效路径；"
            "将自动先接入生成草案，草案生成后会停下来等待人工确认边界，不会自动开始仿真"
            if can_gapfill else
            "需要一份已通过校验的优化配置 YAML 才能启动优化，未找到有效路径，"
            "且当前没有可用的 Aspen 文件，无法自动生成",
            can_gapfill,
        )
    kwargs = {
        "config_path": resolved,
        "db_path": env.get("db_path") or _derive_db_path(resolved),
    }
    return kwargs, True, "", False


def _build_general_chat_task(step: IntentStep, env: _Env) -> tuple[dict, bool, str, bool]:
    """纯问答任务：不依赖任何文件路径，永远 ready=True。

    kwargs 只带 user_text（分类阶段抽不出结构化槎位，因为这是自由文本
    问答，不是任务参数）和可选的 context（若用户提到了工艺背景）。
    user_text 直接取自 IntentStep.raw_slots 的 "user_text" 槎位——由
    build_plan 在调用 builder 前统一注入（见 build_plan 里的特殊处理），
    而不是依赖 LLM 从分类阶段的 raw_slots 里抽取，因为原始问题文本必须
    完整保留，不能有信息损失。
    """
    return {
        "user_text": step.raw_slots.get("user_text", ""),
        "context": step.raw_slots.get("context", ""),
    }, True, "", False


_TASK_BUILDERS = {
    IntentType.ONBOARD_NEW_CASE: _build_onboard_task,
    IntentType.RECOMMEND_VAR_BOUNDS: _build_recommend_bounds_task,
    IntentType.FIT_KINETICS_PARAMS: _build_fit_kinetics_task,
    IntentType.RECOMMEND_KINETICS_BOUNDS: _build_recommend_kinetics_bounds_task,
    IntentType.DIAGNOSE_RESULTS: _build_diagnose_task,
    IntentType.RUN_OPTIMIZATION: _build_run_optimization_task,
    IntentType.GENERAL_CHAT: _build_general_chat_task,
}

# 哪些路由目标在缺 case_config_path 时可以被 gap-fill 自动补上（见上方注释）。
# fit_kinetics_params / recommend_kinetics_bounds 不在此列——它们缺的是用户
# 数据，不是可自动生成的配置。
_GAPFILL_ELIGIBLE_INTENTS = frozenset({
    IntentType.RECOMMEND_VAR_BOUNDS,
    IntentType.DIAGNOSE_RESULTS,
    IntentType.RUN_OPTIMIZATION,
})


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_plan(
    classification: IntentClassification,
    *,
    user_text: str = "",
    aspen_file: str | None = None,
    case_config_path: str | None = None,
    data_csv: str | None = None,
    kinetics_variables_yaml: str | None = None,
    db_path: str | None = None,
) -> Plan:
    """
    把意图分类结果转成可执行的 Plan。

    Parameters
    ----------
    classification:
        intent.py 产出的分类结果。
    user_text:
        本轮用户的原始请求文本。仅供 GENERAL_CHAT 类型的任务使用——
        LLM 分类阶段不会（也不应该）把整段问题原文塞进 raw_slots
        （容易被截断/改写导致答题失去上下文），因此原文由 Planner
        直接从这里注入，保证纯问答任务拿到的是完整原始文本。
    aspen_file / case_config_path / data_csv / kinetics_variables_yaml / db_path:
        CLI/调用方显式传入的环境参数，优先于从用户文本抽取的 raw_slots
        （后者由 LLM 猜测，可信度更低）。均为可选，缺失时尝试用
        IntentStep.raw_slots 中同名字段兜底。

    Returns
    -------
    Plan
        tasks 与 classification.steps 顺序一一对应；CLARIFY_NEEDED 步骤
        始终产出 ready=False 的任务，不会被 Dispatcher 执行。
        某个任务 ready=False 且 waiting_on_gapfill=True 时，Dispatcher
        会自动在它前面插入一个 ONBOARD_NEW_CASE 任务并动态解锁它，
        本函数本身不插入任何任务——插入时机依赖执行结果（接入是否成功），
        属于运行时行为，留给 dispatcher.py 处理。
    """
    env = make_env(
        aspen_file=aspen_file, case_config_path=case_config_path,
        data_csv=data_csv, kinetics_variables_yaml=kinetics_variables_yaml,
        db_path=db_path,
    )

    tasks: list[TaskSpec] = []
    warnings: list[str] = []

    for i, step in enumerate(classification.steps):
        task_id = f"{i}_{step.intent_type.value}"

        if step.intent_type == IntentType.CLARIFY_NEEDED:
            tasks.append(TaskSpec(
                task_id=task_id, intent_type=step.intent_type,
                kwargs={}, ready=False,
                blocking_reason=step.reason or "信息不足，无法确定具体任务，请补充说明需求",
                source_step=step,
            ))
            continue

        if step.intent_type == IntentType.GENERAL_CHAT and user_text:
            # 原始问题文本由 Planner 直接注入，不依赖 LLM 分类阶段的 raw_slots
            # 转述（见函数 docstring）。
            step = replace(step, raw_slots={**step.raw_slots, "user_text": user_text})

        builder = _TASK_BUILDERS.get(step.intent_type)
        if builder is None:
            # 理论上不会发生（IntentType 枚举与 _TASK_BUILDERS 应一一覆盖，
            # 缺失说明代码本身有 bug），但仍防御性处理，不让 Planner 崩溃。
            tasks.append(TaskSpec(
                task_id=task_id, intent_type=step.intent_type,
                kwargs={}, ready=False,
                blocking_reason=f"内部错误：未找到 {step.intent_type.value} 对应的任务构造器",
                source_step=step,
            ))
            warnings.append(f"任务类型 {step.intent_type.value} 缺少对应的 builder，请检查代码")
            continue

        kwargs, ready, blocking_reason, waiting_on_gapfill = builder(step, env)
        tasks.append(TaskSpec(
            task_id=task_id, intent_type=step.intent_type,
            kwargs=kwargs, ready=ready, blocking_reason=blocking_reason,
            source_step=step,
            waiting_on_gapfill=(not ready) and waiting_on_gapfill
                and step.intent_type in _GAPFILL_ELIGIBLE_INTENTS,
        ))

    return Plan(tasks=tasks, warnings=warnings)


def rebuild_task(task: TaskSpec, env: dict) -> TaskSpec:
    """
    用新的 env（通常是 gap-fill 补上 case_config_path 之后）重新构造一个任务。

    供 dispatcher.py 在自动插入的 ONBOARD_NEW_CASE 任务执行成功后调用，
    对被它阻塞的下游任务重新走一遍 builder，不需要重新分类/重新拆解整个
    Plan——只重建这一个任务。task.source_step 保留原始 IntentStep 不变。

    Parameters
    ----------
    task:
        原本 waiting_on_gapfill=True 的任务。
    env:
        已注入新路径（如 gap-fill 生成的 case_config_path）的 env 字典，
        应通过 make_env() 构造。

    Returns
    -------
    TaskSpec
        新任务，task_id 与原任务一致（不改变编号），waiting_on_gapfill
        重置为 False（重建后不应再触发二次 gap-fill，避免无限循环）。
    """
    builder = _TASK_BUILDERS[task.intent_type]
    step = task.source_step
    kwargs, ready, blocking_reason, _ = builder(step, env)
    return TaskSpec(
        task_id=task.task_id, intent_type=task.intent_type,
        kwargs=kwargs, ready=ready, blocking_reason=blocking_reason,
        source_step=step, waiting_on_gapfill=False,
    )


def build_gapfill_onboard_task(task_id: str, env: dict) -> TaskSpec:
    """
    构造一个自动插入的 ONBOARD_NEW_CASE gap-fill 任务。

    与用户直接请求接入不同，这个任务没有对应的 IntentStep（source_step=None），
    is_gapfill=True 标记它是系统自动补的，供报告展示时区分。

    Parameters
    ----------
    task_id:
        gap-fill 任务的标识符，由 dispatcher.py 生成（约定格式
        "{被阻塞任务 task_id}_gapfill_onboard"，保证在报告里能看出
        是为哪个任务补的前置步骤）。
    env:
        已包含 aspen_file 的 env 字典。

    Returns
    -------
    TaskSpec
        ready 由 _build_onboard_task 的校验结果决定（理论上应为 True，
        因为 dispatcher 只在 aspen_file 已知时才会调用这个函数；仍保留
        真实校验结果而不是硬编码 True，防止未来调用方传入不完整 env）。
    """
    kwargs, ready, blocking_reason, _ = _build_onboard_task(
        IntentStep(intent_type=_GAPFILL_TARGET), env,
    )
    return TaskSpec(
        task_id=task_id, intent_type=_GAPFILL_TARGET,
        kwargs=kwargs, ready=ready, blocking_reason=blocking_reason,
        source_step=None, is_gapfill=True,
    )
