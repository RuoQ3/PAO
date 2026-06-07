"""
tools.py — ProcessAdvisor 允许调用的只读工具集。

包含：
  ProcessAdvisorToolRunner  — 只读 tool runner 协议（Protocol）
  ReadOnlyToolRunner        — 协议的默认实现（适配 6 个真实安全工具）

安全约束：本模块不导入 run_case / optimize_pareto / AspenDriver / win32com。
"""
from __future__ import annotations

import re as _re
import logging
from typing import Protocol, runtime_checkable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 失败工况 case_id 协议区块解析
# ---------------------------------------------------------------------------
# query_simulation_db_tool 在 mode='query' 时输出：
#   [CASE_IDS]\n<uuid>\n...\n[/CASE_IDS]
# 只解析该协议区块，绝不从自然语言文本里猜 case_id。

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
