"""
process_advisor_agent.py — 大模型 ProcessAdvisor agent（在只读证据之上做语言推理）。

分层：
  process_advisor.py        —— 只读证据收集器（规则型，调用 6 个安全工具）
  process_advisor_agent.py  —— 本模块；让 LLM 读取证据报告，输出风险判断、
                               下一步实验建议、是否需要 full smoke 等语言分析

安全边界（与 process_advisor 一致）：
  - 严格只读：不驱动 Aspen、不重跑仿真、不写数据库。
  - LLM 只读取证据文本做分析，不被授予任何写工具或 tool-calling 回路。
  - 缺 API key（未配置）时降级：直接返回只读证据报告 + 一行降级说明，
    绝不伪装成"LLM 已分析"。LLM 调用异常同样降级。

入口：
  run_process_advisor_agent(case_config_path, db_path=None, session_id=None, ...) -> str
"""
from __future__ import annotations

import logging
from typing import Literal

from src.agents.llm_client import LLMConfig, chat, is_configured, load_llm_config
from src.agents.process_advisor import ProcessAdvisorToolRunner, run_process_advisor

_log = logging.getLogger(__name__)

# LLM 系统提示：明确角色与只读边界，要求结构化中文输出。
_SYSTEM_PROMPT = """\
你是 PAO（Process Aspen Optimization）项目的只读流程优化顾问。
你会收到一份由只读工具收集的"证据报告"，内容包括：配置摘要、历史数据库状态、
Pareto 摘要、失败诊断、规则型下一步建议。

你的任务（严格基于证据，不臆测未给出的数据）：
1. 风险判断：指出当前配置 / 历史数据中值得注意的风险或异常。
2. 结果解读：若有 Pareto 前沿或失败工况，解释其含义。
3. 下一步实验建议：给出具体、可执行的下一步（如收窄某变量边界、调整目标权重、
   先做一次 full smoke 验证等）。
4. 是否需要 full smoke：明确给出"需要 / 暂不需要"及理由。

硬性约束：
- 你只做语言分析，不能也不会启动 Aspen、不能重跑仿真、不能写数据库。
- 证据报告中标注"无历史数据库"时，绝不臆造历史结果，应建议先运行优化产生数据。
- 证据不足时如实说明"证据不足"，不要编造数字。
- 用简洁中文输出，分小节，避免空话套话。"""

_USER_TEMPLATE = """\
以下是只读证据报告，请基于它输出你的分析：

{evidence}"""


def run_process_advisor_agent(
    case_config_path: str,
    db_path: str | None = None,
    session_id: str | None = None,
    node_db_path: str | None = None,
    mode: Literal["config", "db"] = "db",
    model: str | None = None,
    llm_config: LLMConfig | None = None,
    advisor_runner: ProcessAdvisorToolRunner | None = None,
) -> str:
    """收集只读证据并交大模型分析，返回"证据 + LLM 分析"合并报告。

    严格只读：内部仅调用 run_process_advisor（6 个安全工具）+ LLM 文本推理，
    任何环节都不驱动 Aspen、不写数据库。

    Args:
        case_config_path: YAML 配置路径。
        db_path:          SimulationDB 路径；None 时从配置推断。
        session_id:       仅分析指定 session 的工况；None=全库历史。
        node_db_path:     NodeDB 路径；None 时从配置推断。
        mode:             "db"（默认）| "config"。
        model:            覆盖 PAO_LLM_MODEL。
        llm_config:       直接注入 LLMConfig（测试用）；None 时从环境变量构造。
        advisor_runner:   注入证据收集层的 tool_runner（测试用）。

    Returns:
        合并报告：先是只读证据报告，再追加【LLM 顾问分析】小节。
        未配置 key 或 LLM 调用失败时，仅返回证据报告 + 降级说明，绝不伪装。
    """
    # 1. 只读证据收集（这一步本身就是完整可用的报告）
    evidence = run_process_advisor(
        case_config_path=case_config_path,
        db_path=db_path,
        node_db_path=node_db_path,
        mode=mode,
        session_id=session_id,
        tool_runner=advisor_runner,
    )

    cfg = llm_config if llm_config is not None else load_llm_config(model=model)

    # 2. 未配置 LLM → 降级返回证据报告（不伪装成已分析）
    if not is_configured(cfg):
        return (
            f"{evidence}\n\n"
            "【LLM 顾问分析】\n"
            "  [降级] 未配置大模型 API key，本次仅返回只读证据报告，未做语言模型分析。\n"
            f"  如需启用：请设置环境变量 {cfg.api_key_env}"
            "（以及可选 PAO_LLM_PROVIDER / PAO_LLM_MODEL）后重试。\n"
            f"  当前 LLM 配置：{cfg.redacted()}"
        )

    # 3. 调用大模型分析证据；失败同样降级
    try:
        analysis = chat(
            cfg,
            system=_SYSTEM_PROMPT,
            user=_USER_TEMPLATE.format(evidence=evidence),
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("LLM 分析失败，降级返回证据报告：%s", exc)
        return (
            f"{evidence}\n\n"
            "【LLM 顾问分析】\n"
            f"  [降级] 大模型调用失败，本次仅返回只读证据报告。\n"
            f"  失败原因：{exc}\n"
            f"  当前 LLM 配置：{cfg.redacted()}"
        )

    return (
        f"{evidence}\n\n"
        "【LLM 顾问分析】\n"
        f"  （provider={cfg.provider}, model={cfg.model}）\n\n"
        f"{analysis}"
    )
