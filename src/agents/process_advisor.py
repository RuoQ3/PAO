"""
process_advisor.py — 只读 ProcessAdvisorAgent。

定位：在不接触 Aspen、不重跑仿真、不写任何数据库的前提下，对一个 case
配置 + 其历史数据库做只读体检，产出一份结构化顾问报告。

与 run_demo_case_workflow 的区别：
  workflows.py        —— 会触发 run_case / optimize_pareto（需要 Aspen COM）
  process_advisor.py  —— 严格只读，只调用 6 个安全工具，永不驱动仿真

允许调用的安全工具（全部不依赖 Aspen COM）：
  load_config、validate_config、query_simulation_db、
  summarize_pareto、diagnose_case、query_node_db

禁止调用：
  run_case_tool、optimize_pareto_tool、AspenDriver、win32com

禁止导入（保持只读隔离，由测试 TestEngineeringBoundary 强制）：
  src.aspen_driver、win32com、src.agents.tools.run_case、
  src.agents.tools.optimize_pareto

报告固定 5 个章节：
  1. 配置摘要
  2. 历史数据库状态
  3. Pareto 摘要
  4. 失败诊断
  5. 下一步建议（next_action）

没有历史数据库时，明确报告"无历史数据库"，绝不伪装成查询成功。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

# 仅复用纯 Python 的配置解析层（不调用 tool、不碰 Aspen、不碰 DB）
from src.agents.workflow_helpers import prepare_demo_workflow_state

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 只读 Tool Runner 协议
# ---------------------------------------------------------------------------

@runtime_checkable
class ProcessAdvisorToolRunner(Protocol):
    """注入到 run_process_advisor 的只读 tool 调用接口。

    只声明 6 个安全（不依赖 Aspen COM）工具，刻意不包含
    run_case / optimize_pareto，从类型层面杜绝写操作。

    所有方法均返回字符串报告；以 "错误：" 开头表示失败。
    """

    def load_config(self, case_config_path: str) -> str: ...

    def validate_config(self, case_config_path: str) -> str: ...

    def query_simulation_db(
        self,
        db_path: str,
        mode: str = "query",
        status: str | None = None,
        objective_name: str | None = None,
        limit: int = 10,
        case_id: str | None = None,
        session_id: str | None = None,
    ) -> str: ...

    def summarize_pareto(
        self,
        db_path: str,
        objective_names: list[str],
        include_infeasible: bool = False,
        session_id: str | None = None,
    ) -> str: ...

    def diagnose_case(self, db_path: str, case_id: str) -> str: ...

    def query_node_db(
        self,
        db_path: str,
        mode: str = "node_values",
        case_id: str | None = None,
        limit: int = 20,
    ) -> str: ...

    def get_failed_case_ids(
        self,
        db_path: str,
        limit: int = 3,
        session_id: str | None = None,
    ) -> list[str]:
        """返回最近失败工况 case_id 列表，空列表表示无失败工况。"""
        ...


# ---------------------------------------------------------------------------
# 默认只读 Runner（真实工具适配）
# ---------------------------------------------------------------------------

# 失败工况机器可读区块：query_simulation_db_tool 在 mode='query' 时输出
#   [CASE_IDS]\n<uuid>\n...\n[/CASE_IDS]
# 只解析该协议区块，绝不从自然语言文本里猜 case_id。
import re as _re  # noqa: E402

_CASE_IDS_PATTERN = _re.compile(r"\[CASE_IDS\]\n(.*?)\n\[/CASE_IDS\]", _re.DOTALL)


def _is_tool_error(report: str) -> bool:
    """tool 返回值是否为失败（允许前导空白，全/半角冒号）。"""
    stripped = report.lstrip()
    return stripped.startswith("错误：") or stripped.startswith("错误:")


def _extract_case_ids(report: str, limit: int) -> list[str]:
    """从 [CASE_IDS]...[/CASE_IDS] 区块提取 case_id，区块不存在返回 []。"""
    m = _CASE_IDS_PATTERN.search(report)
    if not m:
        return []
    ids = [line.strip() for line in m.group(1).splitlines() if line.strip()]
    return ids[:limit]


class ReadOnlyToolRunner:
    """把 ProcessAdvisorToolRunner 协议适配到 6 个真实安全工具。

    刻意从各工具子模块直接导入 tool 对象，而不是 from src.agents.tools import，
    这样既避免在导入期连带引入 run_case / optimize_pareto（它们引用 Aspen），
    也让本模块在静态检查上不出现任何被禁工具的名字。

    不持有 AspenDriver / SimulationDB / NodeDB 实例，不做任何写操作。
    """

    @staticmethod
    def _invoke(tool, payload: dict) -> str:
        result = tool.invoke(payload)
        return "" if result is None else str(result)

    def load_config(self, case_config_path: str) -> str:
        from src.agents.tools.load_config import load_case_config_tool
        return self._invoke(load_case_config_tool, {"config_path": case_config_path})

    def validate_config(self, case_config_path: str) -> str:
        from src.agents.tools.validate_config import validate_config_tool
        return self._invoke(validate_config_tool, {"config_path": case_config_path})

    def query_simulation_db(
        self,
        db_path: str,
        mode: str = "query",
        status: str | None = None,
        objective_name: str | None = None,
        limit: int = 10,
        case_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        from src.agents.tools.query_simulation_db import query_simulation_db_tool
        payload: dict = {"db_path": db_path, "mode": mode, "limit": limit}
        if status is not None:
            payload["status"] = status
        if objective_name is not None:
            payload["objective_name"] = objective_name
        if case_id is not None:
            payload["case_id"] = case_id
        if session_id is not None:
            payload["session_id"] = session_id
        return self._invoke(query_simulation_db_tool, payload)

    def summarize_pareto(
        self,
        db_path: str,
        objective_names: list[str],
        include_infeasible: bool = False,
        session_id: str | None = None,
    ) -> str:
        from src.agents.tools.summarize_pareto import summarize_pareto_tool
        payload: dict = {
            "db_path": db_path,
            "objective_names": ",".join(objective_names),
            "include_infeasible": include_infeasible,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        return self._invoke(summarize_pareto_tool, payload)

    def diagnose_case(self, db_path: str, case_id: str) -> str:
        from src.agents.tools.diagnose_case import diagnose_case_tool
        return self._invoke(diagnose_case_tool, {"db_path": db_path, "case_id": case_id})

    def query_node_db(
        self,
        db_path: str,
        mode: str = "node_values",
        case_id: str | None = None,
        limit: int = 20,
    ) -> str:
        from src.agents.tools.query_node_db import query_node_db_tool
        payload: dict = {"db_path": db_path, "mode": mode, "limit": limit}
        if case_id is not None:
            payload["case_id"] = case_id
        return self._invoke(query_node_db_tool, payload)

    def get_failed_case_ids(
        self,
        db_path: str,
        limit: int = 3,
        session_id: str | None = None,
    ) -> list[str]:
        """查询失败工况，从 [CASE_IDS] 协议区块提取 case_id。

        query 返回 "错误：..." 时抛 RuntimeError，由上层记为诊断错误，
        绝不把查询失败伪装成"无失败工况"。
        """
        report = self.query_simulation_db(
            db_path=db_path, mode="query", status="sim_failed", limit=limit,
            session_id=session_id,
        )
        if _is_tool_error(report):
            raise RuntimeError(f"query_simulation_db 返回错误报告 — {report}")
        return _extract_case_ids(report, limit)


# ---------------------------------------------------------------------------
# 内部：数据库路径解析（只读，不创建文件）
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


def _resolve_existing_db(db_path: str | None) -> Path | None:
    """把候选 db_path 解析为一个已存在的文件，找不到返回 None。

    解析顺序：绝对路径 → CWD 相对 → 项目根相对。
    只检查存在性，绝不创建文件（SimulationDB() 构造会建库，这里刻意不调用）。
    """
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
    """根据只读体检结果，用规则推断下一步建议。

    本 agent 只读，建议中绝不出现"已运行成功"之类伪装；缺数据时
    明确建议先跑优化生成数据库。
    """
    actions: list[str] = []

    # 配置解析失败：最高优先级
    if config_parse_error is not None:
        actions.append("配置解析失败：请检查 YAML 路径与语法，修复后重新体检。")
        return actions

    if config_mode:
        actions.append(
            "当前为 mode='config' 纯配置体检：如需历史数据分析，"
            "请用 mode='db' 重新运行，并确认 simulation.db 已存在。"
        )
        return actions

    # 无数据库（路径未推断出，或文件不存在）
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

    # 有数据库但查询失败
    if not db_query_ok:
        actions.append(
            "历史数据库查询失败：请确认 simulation.db 未损坏、可读，"
            "必要时用 query_simulation_db_tool 手动复查。"
        )
        return actions

    # 有数据库且查询成功
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
# 入口函数
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
        db_path:          SimulationDB 路径；None 时从配置自动推断
                          （pareto_bayesian → {config_dir}/output/simulation.db）。
        node_db_path:     NodeDB 路径；None 时从配置 extraction.catalog_db 推断。
        mode:             "db"（默认）= 含历史数据库分析；"config" = 仅做配置体检，
                          跳过所有数据库查询章节。非法值返回报告级错误，不静默执行。
        session_id:       仅分析指定优化 session 的工况；None 时分析全库历史。
                          强烈建议为单次优化结果体检时传入，避免把历史累计数据
                          混入本轮分析（与 plot_pareto 的 session 口径一致）。
        tool_runner:      实现 ProcessAdvisorToolRunner 协议的 runner；
                          默认 ReadOnlyToolRunner（生产用真实工具）。测试可注入 FakeRunner。

    Returns:
        含 5 个固定章节的文本报告：
          1. 配置摘要  2. 历史数据库状态  3. Pareto 摘要
          4. 失败诊断  5. 下一步建议
    """
    # mode 严格校验：非法值直接返回报告级错误，不退回 db 路径静默执行
    if mode not in ("config", "db"):
        return (
            "=== ProcessAdvisor 只读体检报告 ===\n\n"
            f"错误：mode={mode!r} 不是合法值，仅支持 'config' 或 'db'。\n"
            "  - mode='db'    ：含历史数据库分析（默认）\n"
            "  - mode='config'：仅做配置体检，跳过数据库查询"
        )

    runner = tool_runner if tool_runner is not None else ReadOnlyToolRunner()

    sections: list[str] = ["=== ProcessAdvisor 只读体检报告 ===", ""]

    # ── 配置解析（纯 Python，不调用 tool）──────────────────────────────────
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

    # 调用方显式传入优先；否则用配置推断值
    eff_db_path = db_path if db_path is not None else inferred_db
    eff_node_db = node_db_path if node_db_path is not None else inferred_node_db

    # ── 章节 1：配置摘要 ───────────────────────────────────────────────────
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
        # load_config / validate_config 只读校验
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

    # 供章节 5（next_action）决策的状态标志
    db_resolved: Path | None = None
    db_query_ok = False
    pareto_ok = False
    diagnosed_ids: list[str] = []
    config_mode = (mode == "config")

    # ── 章节 2：历史数据库状态 ─────────────────────────────────────────────
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
        # 通过结构化接口获取失败 case_id（不解析自然语言）
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
            # 取第一个失败工况补充 node_db 视角（若有 node_db 路径）
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

    # ── 章节 5：下一步建议（next_action）──────────────────────────────────
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
