"""Read-only evidence collection followed by verified Agent reasoning."""
from __future__ import annotations
from src.agents.errors import AgentUnavailableError, unavailable, fail_closed, failure_from
from src.agents.reasoning import configured, redact

import logging
from pathlib import Path
from typing import Literal

from src.agents.subagents.process_advisor.tools import (
    ProcessAdvisorToolRunner,
    ReadOnlyToolRunner,
    _is_tool_error,
)
from src.agents.subagents.process_advisor.prompts import SYSTEM_PROMPT, USER_TEMPLATE

# 模块级 import：让 monkeypatch 可在测试中替换这些名字
from src.agents.lead.llm_client import chat, is_configured, load_llm_config  # noqa: E402

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

def _collect_process_evidence(
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
    config_parse_error: str | None = None
    optimizer_type = ""
    objective_names: list[str] = []
    resolved_config = case_config_path
    inferred_db: str | None = None
    inferred_node_db: str | None = None
    try:
        import yaml
        from pathlib import Path
        resolved_config = str(Path(case_config_path).resolve())
        raw = yaml.safe_load(Path(resolved_config).read_text(encoding="utf-8")) or {}
        optimizer_type = str((raw.get("optimizer") or {}).get("type", ""))
        objective_names = [str(x.get("name", x)) if isinstance(x, dict) else str(x)
                           for x in (raw.get("objectives") or [])]
        base = Path(resolved_config).parent
        inferred_db = str(base / "output" / "simulation.db")
        inferred_node_db = str(base / "output" / "node.db")
    except AgentUnavailableError:
        raise
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
        except AgentUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            load_report = f"错误：load_config 调用异常 [{type(exc).__name__}] — {exc}"
        cfg_lines.append("  ── load_config ──")
        cfg_lines.append(_indent_block(load_report, prefix="    ", max_lines=40))
        try:
            validate_report = runner.validate_config(resolved_config)
        except AgentUnavailableError:
            raise
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
            except AgentUnavailableError:
                raise
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
        except AgentUnavailableError:
            raise
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
            except AgentUnavailableError:
                raise
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
                except AgentUnavailableError:
                    raise
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
                    except AgentUnavailableError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        node_rep = (
                            f"错误：query_node_db 调用异常 [{type(exc).__name__}] — {exc}"
                        )
                    d_lines.append(f"  ── node_db 节点值（case_id={diagnosed_ids[0]}）──")
                    d_lines.append(_indent_block(node_rep, prefix="    ", max_lines=40))
    sections.append("\n".join(d_lines))
    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM 层入口
# ---------------------------------------------------------------------------

@fail_closed("process_advisor")
def run_process_advisor_agent(case_config_path, db_path=None, session_id=None,
                              node_db_path=None, mode="db", model=None, llm_config=None,
                              advisor_runner=None, question="Analyze the current process and its evidence."):
    from src.agents.reasoning import reason
    cfg = configured(llm_config or load_llm_config(model=model))
    # A validated provider response precedes evidence access at this standalone entry.
    from src.agents.lead.intent import IntentClassifier
    IntentClassifier(llm_config=cfg, llm_call=chat).classify(question)
    evidence = _collect_process_evidence(case_config_path, db_path, node_db_path,
                                         mode, session_id, advisor_runner)
    data, _ = reason(question, context={"report_evidence": evidence}, role="process_advisor",
                     llm_config=cfg, llm_call=chat)
    return data["text"]


def run_process_advisor(case_config_path, db_path=None, node_db_path=None, mode="db",
                        session_id=None, tool_runner=None, **kwargs):
    return run_process_advisor_agent(case_config_path, db_path, session_id, node_db_path,
                                    mode, advisor_runner=tool_runner, **kwargs)
