"""
agent.py — ProcessAdvisor agent 核心逻辑。

分两层：
  run_process_advisor()       — 只读证据收集器（规则型，调用 6 个安全工具）
  run_process_advisor_agent() — LLM 层：读取证据报告，输出语言分析

安全边界：
  - 严格只读：不驱动 Aspen、不重跑仿真、不写数据库。
  - LLM 只读取证据文本做分析，不被授予任何写工具。
  - 缺 API key 或 LLM 调用失败时降级返回证据报告 + 说明，绝不伪装成"已分析"。

禁止导入（保持只读隔离）：
  src.aspen_driver、win32com、src.agents.tools.run_case、
  src.agents.tools.optimize_pareto
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from src.agents.process_advisor.tools import (
    ProcessAdvisorToolRunner,
    ReadOnlyToolRunner,
    _is_tool_error,
)
from src.agents.process_advisor.prompts import SYSTEM_PROMPT, USER_TEMPLATE

# 模块级 import：让 monkeypatch 可在测试中替换这些名字
from src.agents.llm_client import chat, is_configured, load_llm_config  # noqa: E402

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent


def _resolve_existing_db(db_path: str | None) -> Path | None:
    """把候选 db_path 解析为一个已存在的文件，找不到返回 None。"""
    if not db_path:
        return None
    p = Path(db_path)
    if p.is_absolute():
        return p if p.exists() else None
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd.resolve()
    from_root = _PROJECT_ROOT / p
    if from_root.exists():
        return from_root.resolve()
    return None


def _indent_block(text: str, prefix: str = "  ", max_lines: int = 80) -> str:
    """缩进嵌入一段子报告，超过 max_lines 截断。"""
    if not text:
        return f"{prefix}（空）"
    lines = text.splitlines()
    out = [f"{prefix}{ln}" for ln in lines[:max_lines]]
    if len(lines) > max_lines:
        out.append(f"{prefix}...（共 {len(lines)} 行，已截断）")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 规则型 next_action 生成器
# ---------------------------------------------------------------------------

def _determine_next_actions(
    *,
    config_parse_error: str | None,
    config_mode: bool,
    is_pareto: bool,
    eff_db_path: str | None,
    db_resolved: Path | None,
    db_query_ok: bool,
    pareto_ok: bool,
    diagnosed_ids: list[str],
    objective_names: list[str],
) -> list[str]:
    """根据只读体检结果，用规则推断下一步建议。"""
    actions: list[str] = []

    if config_parse_error is not None:
        actions.append("配置解析失败：请检查 YAML 路径与语法，修复后重新体检。")
        return actions

    if config_mode:
        actions.append(
            "当前为 mode='config' 纯配置体检：如需历史数据分析，"
            "请用 mode='db' 重新运行，并确认 simulation.db 已存在。"
        )
        return actions

    if eff_db_path is None or db_resolved is None:
        if not is_pareto:
            actions.append(
                "无历史数据库且当前非 pareto_bayesian 分支：如需多目标分析，"
                "请将 optimizer.type 设为 pareto_bayesian 并先运行一次优化。"
            )
        else:
            actions.append(
                "无历史数据库：尚未产生仿真记录，请先运行优化"
                "（python -m src.main <config>）生成 simulation.db 后再体检。"
            )
        return actions

    if not db_query_ok:
        actions.append(
            "历史数据库查询失败：请确认 simulation.db 未损坏、可读，"
            "必要时用 query_simulation_db_tool 手动复查。"
        )
        return actions

    if diagnosed_ids:
        actions.append(
            f"发现 {len(diagnosed_ids)} 个失败工况已诊断：请按【4. 失败诊断】中的建议"
            "收窄设计变量边界或修正初始值，再运行下一轮优化。"
        )

    if is_pareto and pareto_ok:
        actions.append(
            "Pareto 前沿已成功汇总：请在【3. Pareto 摘要】中筛选满足纯度/能耗约束的"
            "候选操作点，进行能耗与经济性分析。"
        )
        actions.append(
            "若前沿覆盖度足够，可收窄变量范围做局部精化，或调整目标权重后再优化。"
        )
    elif is_pareto and not pareto_ok:
        if len(objective_names) < 2:
            actions.append(
                "Pareto 摘要因目标不足跳过：请在配置中提供至少 2 个 objectives 后重试。"
            )
        else:
            actions.append(
                "Pareto 摘要失败：可用 query_simulation_db_tool mode='by_objective' "
                "手动检查目标分布，确认数据库中已有可行工况。"
            )
    elif not is_pareto:
        actions.append(
            "历史数据库可读但当前非 pareto_bayesian 分支：可用 query_simulation_db_tool "
            "查看单次/单目标运行记录，确认关键输出合理。"
        )

    if not actions:
        actions.append("历史数据库健康且无失败工况：可按需进入下一阶段分析或继续优化。")

    return actions


# ---------------------------------------------------------------------------
# 主入口：只读证据收集
# ---------------------------------------------------------------------------

def run_process_advisor(
    case_config_path: str,
    db_path: str | None = None,
    node_db_path: str | None = None,
    mode: Literal["config", "db"] = "db",
    session_id: str | None = None,
    tool_runner: ProcessAdvisorToolRunner | None = None,
) -> str:
    """对一个 case 配置 + 其历史数据库做只读体检，返回结构化顾问报告。

    严格只读：只调用 6 个安全工具，绝不驱动 Aspen、绝不重跑仿真、
    绝不写数据库。任何异常都转化为报告内容，函数本身不向调用方抛异常。

    Args:
        case_config_path: YAML 配置路径（相对或绝对）。
        db_path:          SimulationDB 路径；None 时从配置自动推断。
        node_db_path:     NodeDB 路径；None 时从配置 extraction.catalog_db 推断。
        mode:             "db"（默认）含历史数据库分析；"config" 仅做配置体检。
        session_id:       仅分析指定优化 session；None 时分析全库历史。
        tool_runner:      实现 ProcessAdvisorToolRunner 协议的 runner；
                          默认 ReadOnlyToolRunner。测试可注入 FakeRunner。

    Returns:
        含 5 个固定章节的文本报告。
    """
    if mode not in ("config", "db"):
        return (
            "=== ProcessAdvisor 只读体检报告 ===\n\n"
            f"错误：mode={mode!r} 不是合法值，仅支持 'config' 或 'db'。\n"
            "  - mode='db'    ：含历史数据库分析（默认）\n"
            "  - mode='config'：仅做配置体检，跳过数据库查询"
        )

    runner = tool_runner if tool_runner is not None else ReadOnlyToolRunner()

    sections: list[str] = ["=== ProcessAdvisor 只读体检报告 ===", ""]

    # ── 配置解析 ──────────────────────────────────────────────────────────
    from src.agents.demo_workflow.helpers import prepare_demo_workflow_state

    config_parse_error: str | None = None
    optimizer_type = ""
    objective_names: list[str] = []
    resolved_config = case_config_path
    inferred_db: str | None = None
    inferred_node_db: str | None = None
    try:
        state = prepare_demo_workflow_state(case_config_path)
        optimizer_type = state.optimizer_type
        objective_names = state.objective_names
        resolved_config = state.resolved_config_path or case_config_path
        inferred_db = state.db_path
        inferred_node_db = state.node_db_path
    except Exception as exc:  # noqa: BLE001
        config_parse_error = f"{type(exc).__name__}: {exc}"

    is_pareto = optimizer_type == "pareto_bayesian"
    eff_db_path = db_path if db_path is not None else inferred_db
    eff_node_db = node_db_path if node_db_path is not None else inferred_node_db

    # ── 章节 1：配置摘要 ──────────────────────────────────────────────────
    cfg_lines = ["【1. 配置摘要】"]
    cfg_lines.append(f"  配置路径（原始）   : {case_config_path}")
    cfg_lines.append(f"  配置路径（已解析） : {resolved_config}")
    cfg_lines.append(f"  分析模式 mode      : {mode}")
    cfg_lines.append(
        f"  session_id 口径    : {session_id if session_id else '全库历史（未指定 session）'}"
    )
    if config_parse_error is not None:
        cfg_lines.append(f"  [失败] 配置解析失败：{config_parse_error}")
        cfg_lines.append("  无法读取 optimizer_type / objective_names，后续章节受限。")
    else:
        cfg_lines.append(f"  optimizer_type    : {optimizer_type or '未设置'}")
        cfg_lines.append(
            f"  objective_names   : {', '.join(objective_names) if objective_names else '无'}"
        )
        cfg_lines.append(f"  推断 db_path      : {eff_db_path or '无（非 pareto 分支或未配置）'}")
        cfg_lines.append(f"  推断 node_db_path : {eff_node_db or '无'}")
        try:
            load_report = runner.load_config(resolved_config)
        except Exception as exc:  # noqa: BLE001
            load_report = f"错误：load_config 调用异常 [{type(exc).__name__}] — {exc}"
        cfg_lines.append("  ── load_config ──")
        cfg_lines.append(_indent_block(load_report, prefix="    ", max_lines=40))
        try:
            validate_report = runner.validate_config(resolved_config)
        except Exception as exc:  # noqa: BLE001
            validate_report = f"错误：validate_config 调用异常 [{type(exc).__name__}] — {exc}"
        cfg_lines.append("  ── validate_config ──")
        cfg_lines.append(_indent_block(validate_report, prefix="    ", max_lines=40))
    sections.append("\n".join(cfg_lines))
    sections.append("")

    db_resolved: Path | None = None
    db_query_ok = False
    pareto_ok = False
    diagnosed_ids: list[str] = []
    config_mode = (mode == "config")

    # ── 章节 2：历史数据库状态 ────────────────────────────────────────────
    db_lines = ["【2. 历史数据库状态】"]
    if config_mode:
        db_lines.append("  mode='config'：仅做配置体检，跳过历史数据库查询。")
    elif eff_db_path is None:
        db_lines.append("  无历史数据库：配置未推断出 SimulationDB 路径")
        db_lines.append("  （通常因 optimizer.type 非 pareto_bayesian，或未配置输出目录）。")
        db_lines.append("  本次不做任何数据库分析，结果不代表已有历史仿真数据。")
    else:
        db_resolved = _resolve_existing_db(eff_db_path)
        if db_resolved is None:
            db_lines.append(f"  无历史数据库：目标路径不存在 → {eff_db_path}")
            db_lines.append("  尚未运行过优化，或数据库文件被移动/删除。")
            db_lines.append("  本次不做任何数据库分析，绝不伪装为查询成功。")
        else:
            db_lines.append(f"  历史数据库已找到：{db_resolved}")
            try:
                q_report = runner.query_simulation_db(
                    db_path=str(db_resolved), mode="query", limit=10,
                    session_id=session_id,
                )
            except Exception as exc:  # noqa: BLE001
                q_report = f"错误：query_simulation_db 调用异常 [{type(exc).__name__}] — {exc}"
            db_query_ok = not _is_tool_error(q_report)
            if not db_query_ok:
                db_lines.append("  [失败] 历史工况查询失败，下方为错误详情：")
            db_lines.append(_indent_block(q_report, prefix="    "))
    sections.append("\n".join(db_lines))
    sections.append("")

    # ── 章节 3：Pareto 摘要 ────────────────────────────────────────────────
    p_lines = ["【3. Pareto 摘要】"]
    if config_mode:
        p_lines.append("  mode='config'：跳过 Pareto 摘要。")
    elif not is_pareto:
        p_lines.append("  不适用：当前 optimizer.type 非 pareto_bayesian，无多目标前沿可汇总。")
    elif db_resolved is None:
        p_lines.append("  无历史数据库：无法计算 Pareto 前沿。")
    elif len(objective_names) < 2:
        p_lines.append(
            f"  目标数不足：检测到 {len(objective_names)} 个目标，Pareto 摘要至少需要 2 个。"
        )
    else:
        try:
            sp_report = runner.summarize_pareto(
                db_path=str(db_resolved),
                objective_names=objective_names,
                include_infeasible=False,
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            sp_report = f"错误：summarize_pareto 调用异常 [{type(exc).__name__}] — {exc}"
        pareto_ok = not _is_tool_error(sp_report)
        if not pareto_ok:
            p_lines.append("  [失败] Pareto 摘要失败，下方为错误详情：")
        p_lines.append(_indent_block(sp_report, prefix="    "))
    sections.append("\n".join(p_lines))
    sections.append("")

    # ── 章节 4：失败诊断 ───────────────────────────────────────────────────
    d_lines = ["【4. 失败诊断】"]
    if config_mode:
        d_lines.append("  mode='config'：跳过失败诊断。")
    elif db_resolved is None:
        d_lines.append("  无历史数据库：无失败工况可诊断。")
    elif not db_query_ok:
        d_lines.append("  历史工况查询失败，跳过诊断（不从失败查询中猜 case_id）。")
    else:
        get_failed = getattr(runner, "get_failed_case_ids", None)
        if get_failed is None:
            d_lines.append("  runner 未提供结构化失败 case_id 接口，跳过诊断。")
        else:
            try:
                raw_ids = get_failed(
                    db_path=str(db_resolved), limit=3, session_id=session_id,
                )
                diagnosed_ids = [c for c in (raw_ids or []) if c and c.strip()][:3]
            except Exception as exc:  # noqa: BLE001
                d_lines.append(
                    f"  [失败] 失败 case_id 查询异常 [{type(exc).__name__}] — {exc}"
                )
                diagnosed_ids = []
            if not diagnosed_ids and "[失败]" not in "\n".join(d_lines):
                d_lines.append("  未发现失败工况（sim_failed），历史运行较健康。")
            for cid in diagnosed_ids:
                try:
                    diag = runner.diagnose_case(db_path=str(db_resolved), case_id=cid)
                except Exception as exc:  # noqa: BLE001
                    diag = f"错误：diagnose_case({cid}) 调用异常 [{type(exc).__name__}] — {exc}"
                d_lines.append(f"  ── case_id: {cid} ──")
                d_lines.append(_indent_block(diag, prefix="    ", max_lines=50))
            if diagnosed_ids and eff_node_db:
                node_resolved = _resolve_existing_db(eff_node_db)
                if node_resolved is not None:
                    try:
                        node_rep = runner.query_node_db(
                            db_path=str(node_resolved),
                            mode="node_values",
                            case_id=diagnosed_ids[0],
                            limit=20,
                        )
                    except Exception as exc:  # noqa: BLE001
                        node_rep = (
                            f"错误：query_node_db 调用异常 [{type(exc).__name__}] — {exc}"
                        )
                    d_lines.append(f"  ── node_db 节点值（case_id={diagnosed_ids[0]}）──")
                    d_lines.append(_indent_block(node_rep, prefix="    ", max_lines=40))
    sections.append("\n".join(d_lines))
    sections.append("")

    # ── 章节 5：下一步建议 ────────────────────────────────────────────────
    next_actions = _determine_next_actions(
        config_parse_error=config_parse_error,
        config_mode=config_mode,
        is_pareto=is_pareto,
        eff_db_path=eff_db_path,
        db_resolved=db_resolved,
        db_query_ok=db_query_ok,
        pareto_ok=pareto_ok,
        diagnosed_ids=diagnosed_ids,
        objective_names=objective_names,
    )
    na_lines = ["【5. 下一步建议】"]
    for i, action in enumerate(next_actions, 1):
        na_lines.append(f"  [{i}] {action}")
    sections.append("\n".join(na_lines))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM 层入口
# ---------------------------------------------------------------------------

def run_process_advisor_agent(
    case_config_path: str,
    db_path: str | None = None,
    session_id: str | None = None,
    node_db_path: str | None = None,
    mode: Literal["config", "db"] = "db",
    model: str | None = None,
    llm_config=None,
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
        未配置 key 或 LLM 调用失败时，仅返回证据报告 + 降级说明。
    """
    # chat / is_configured / load_llm_config 从模块级 import 引入，支持 monkeypatch

    evidence = run_process_advisor(
        case_config_path=case_config_path,
        db_path=db_path,
        node_db_path=node_db_path,
        mode=mode,
        session_id=session_id,
        tool_runner=advisor_runner,
    )

    cfg = llm_config if llm_config is not None else load_llm_config(model=model)

    if not is_configured(cfg):
        return (
            f"{evidence}\n\n"
            "【LLM 顾问分析】\n"
            "  [降级] 未配置大模型 API key，本次仅返回只读证据报告，未做语言模型分析。\n"
            f"  如需启用：请设置环境变量 {cfg.api_key_env}"
            "（以及可选 PAO_LLM_PROVIDER / PAO_LLM_MODEL）后重试。\n"
            f"  当前 LLM 配置：{cfg.redacted()}"
        )

    try:
        analysis = chat(
            cfg,
            system=SYSTEM_PROMPT,
            user=USER_TEMPLATE.format(evidence=evidence),
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
