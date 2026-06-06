"""
tool_runner.py — RealToolRunner：把 DemoWorkflowToolRunner 接口适配到真实 tools。

职责：
  - 把 workflow 需要的方法转发到已有 LangChain tools
  - 通过 .invoke(...) 调用，统一返回 str
  - tool 调用抛异常时直接向上传播，由 workflows.py 捕获并记录为 step error
  - 不持有 AspenDriver、SimulationDB、NodeDB 实例
  - 不在 runner 内实现任何 workflow 控制逻辑

禁止导入：
  src.database、src.workflows、src.aspen_driver、
  AspenDriver、SimulationRunner、SimulationDB、NodeDB
"""
from __future__ import annotations

import re

from src.agents.tools import (
    validate_config_tool,
    run_case_tool,
    optimize_pareto_tool,
    query_simulation_db_tool,
    diagnose_case_tool,
    query_node_db_tool,
    summarize_pareto_tool,
)

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

# 机器可读区块的正则：匹配 [CASE_IDS]\n...\n[/CASE_IDS]
_CASE_IDS_PATTERN = re.compile(
    r"\[CASE_IDS\]\n(.*?)\n\[/CASE_IDS\]",
    re.DOTALL,
)


def _invoke_tool(tool, payload: dict) -> str:
    """统一调用 tool.invoke，返回字符串结果。

    - None 结果转为空字符串
    - tool.invoke 抛异常时直接向上传播（由 workflows.py 捕获）
    """
    result = tool.invoke(payload)
    if result is None:
        return ""
    return str(result)


def _extract_case_ids_from_machine_block(report: str, limit: int = 3) -> list[str]:
    """从报告文本中提取 [CASE_IDS]...[/CASE_IDS] 区块内的 case_id 列表。

    只解析协议区块，不解析普通自然语言文本（如 case_id=xxx 的行）。
    区块不存在时返回空列表。
    """
    m = _CASE_IDS_PATTERN.search(report)
    if not m:
        return []
    raw = m.group(1)
    ids = [line.strip() for line in raw.splitlines() if line.strip()]
    return ids[:limit]


# ---------------------------------------------------------------------------
# RealToolRunner
# ---------------------------------------------------------------------------

class RealToolRunner:
    """把 DemoWorkflowToolRunner 协议适配到真实 LangChain tools。

    可直接传入 run_demo_case_workflow：

        run_demo_case_workflow(
            "cases/demo_case/pareto_config.yaml",
            RealToolRunner(),
        )
    """

    # ------------------------------------------------------------------
    # 配置校验
    # ------------------------------------------------------------------

    def validate_config(self, case_config_path: str) -> str:
        return _invoke_tool(validate_config_tool, {"config_path": case_config_path})

    # ------------------------------------------------------------------
    # 单次仿真
    # ------------------------------------------------------------------

    def run_case(self, case_config_path: str) -> str:
        return _invoke_tool(run_case_tool, {"config_path": case_config_path})

    # ------------------------------------------------------------------
    # Pareto 优化
    # ------------------------------------------------------------------

    def optimize_pareto(
        self,
        case_config_path: str,
        db_path: str | None = None,
    ) -> str:
        payload: dict = {"config_path": case_config_path}
        if db_path is not None:
            payload["db_path"] = db_path
        return _invoke_tool(optimize_pareto_tool, payload)

    # ------------------------------------------------------------------
    # 数据库查询
    # ------------------------------------------------------------------

    def query_simulation_db(
        self,
        db_path: str,
        mode: str,
        status: str | None = None,
        objective_name: str | None = None,
        limit: int = 10,
        case_id: str | None = None,
    ) -> str:
        payload: dict = {
            "db_path": db_path,
            "mode": mode,
            "limit": limit,
        }
        if status is not None:
            payload["status"] = status
        if objective_name is not None:
            payload["objective_name"] = objective_name
        if case_id is not None:
            payload["case_id"] = case_id
        return _invoke_tool(query_simulation_db_tool, payload)

    # ------------------------------------------------------------------
    # 结构化失败 case_id 获取
    # ------------------------------------------------------------------

    def get_failed_case_ids(self, db_path: str, limit: int = 3) -> list[str]:
        """查询失败工况，从机器可读区块 [CASE_IDS]...[/CASE_IDS] 提取 case_id。

        使用 mode="query" + status="sim_failed"，依赖
        query_simulation_db_tool 输出的协议区块，不解析自然语言。

        行为约定：
        - query_simulation_db() 抛异常 → 直接向上传播，由 workflows.py 记录为
          diagnose_case error，不伪装成"无失败工况"。
        - query_simulation_db() 返回 "错误：..." / "错误:..." → raise RuntimeError，
          同样交给 workflows.py 捕获，避免把查询失败伪装成正常无失败工况。
        - [CASE_IDS] 区块不存在 → 返回 []，这是协议上"无结构化 ID"的正常情况。
        """
        # 不捕获异常，让 workflows.py 记录为 step error
        report = self.query_simulation_db(
            db_path=db_path,
            mode="query",
            status="sim_failed",
            limit=limit,
        )
        # tool 返回错误报告时作为异常抛出，而非静默返回空列表
        stripped = report.lstrip()
        if stripped.startswith("错误：") or stripped.startswith("错误:"):
            raise RuntimeError(
                f"get_failed_case_ids: query_simulation_db 返回错误报告 — {report}"
            )
        return _extract_case_ids_from_machine_block(report, limit=limit)

    # ------------------------------------------------------------------
    # 失败诊断
    # ------------------------------------------------------------------

    def diagnose_case(self, db_path: str, case_id: str) -> str:
        return _invoke_tool(diagnose_case_tool, {
            "db_path": db_path,
            "case_id": case_id,
        })

    # ------------------------------------------------------------------
    # 节点数据库查询
    # ------------------------------------------------------------------

    def query_node_db(
        self,
        db_path: str,
        mode: str,
        case_id: str | None = None,
        limit: int = 20,
    ) -> str:
        payload: dict = {
            "db_path": db_path,
            "mode": mode,
            "limit": limit,
        }
        if case_id is not None:
            payload["case_id"] = case_id
        return _invoke_tool(query_node_db_tool, payload)

    # ------------------------------------------------------------------
    # Pareto 总结
    # ------------------------------------------------------------------

    def summarize_pareto(
        self,
        db_path: str,
        objective_names: list[str],
        include_infeasible: bool = False,
    ) -> str:
        return _invoke_tool(summarize_pareto_tool, {
            "db_path": db_path,
            "objective_names": ",".join(objective_names),
            "include_infeasible": include_infeasible,
        })
