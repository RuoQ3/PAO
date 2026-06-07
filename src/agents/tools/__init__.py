"""
src/agents/tools/__init__.py — PAO Agent 工具包。

将八个工具模块统一对外暴露，保持与旧 src.agents.tools 模块完全相同的公开接口。

目录结构：
  _common.py              — 共享工具（路径解析、YAML 读取、运行时依赖引用）
  load_config.py          — load_case_config_tool（不依赖 Aspen COM）
  validate_config.py      — validate_config_tool（不依赖 Aspen COM）
  run_case.py             — run_case_tool（需要 Aspen COM）
  optimize_pareto.py      — optimize_pareto_tool（需要 Aspen COM）
  query_simulation_db.py  — query_simulation_db_tool（不依赖 Aspen COM）
  query_node_db.py        — query_node_db_tool（不依赖 Aspen COM）
  diagnose_case.py        — diagnose_case_tool（不依赖 Aspen COM）
  summarize_pareto.py     — summarize_pareto_tool（不依赖 Aspen COM）

公开接口（与旧 tools.py 完全兼容）：
  load_case_config_tool        — BaseTool
  load_config_tool             — load_case_config_tool 的别名（向后兼容）
  validate_config_tool         — BaseTool
  run_case_tool                — BaseTool
  optimize_pareto_tool         — BaseTool
  query_simulation_db_tool     — BaseTool
  query_node_db_tool           — BaseTool
  diagnose_case_tool           — BaseTool
  summarize_pareto_tool        — BaseTool
  get_agent_tools()            — 返回所有工具列表

测试 patch 路径示例（新路径）：
  patch("src.agents.tools._common._load_optimize_config", ...)
  patch("src.agents.tools._common._AspenDriver", ...)
  patch("src.agents.tools._common._run_case_fn", ...)
  patch("src.agents.tools._common._optimize_pareto_fn", ...)
  patch("src.agents.tools._common._import_run_time_deps", ...)
  patch("src.agents.tools._common._import_pareto_deps", ...)
  patch("src.agents.tools._common._resolve_config_path", ...)
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

# 工具本体
from .load_config import (
    load_case_config_tool,
    load_config_tool,          # 向后兼容别名
    _impl_load_config,
    _fmt_validation_warnings,
    _build_config_summary,
)
from .validate_config import (
    validate_config_tool,
    _impl_validate_config,
    _check_sim_file,
    _check_design_var_sanity,
    _check_objective_sanity,
    _check_constraint_sanity,
    _check_optimizer_sanity,
    _run_python_parse,
)
from .run_case import (
    run_case_tool,
    _impl_run_case,
    _parse_design_vars_json,
    _build_initial_design_vars,
    _apply_derived_and_repair,
    _fmt_case_summary,
)
from .optimize_pareto import (
    optimize_pareto_tool,
    _impl_optimize_pareto,
    _fmt_pareto_front_lines,
    _fmt_hv_trend,
    _fmt_pareto_result_summary,
)
from .query_simulation_db import (
    query_simulation_db_tool,
    _impl_query_simulation_db,
    _resolve_db_path,
    _parse_tags_str,
    _fmt_case_row,
    _fmt_objective_row,
    _fmt_query_cases_report,
    _fmt_query_objective_report,
    _fmt_get_case_report,
)
from .query_node_db import (
    query_node_db_tool,
    _impl_query_node_db,
    _fmt_node_values_report,
    _fmt_path_search_report,
    _fmt_recurring_errors_report,
    _fmt_catalog_report,
    _fmt_manifest_report,
)
from .diagnose_case import (
    diagnose_case_tool,
    _impl_diagnose_case,
    _fmt_diagnose_report,
    _build_suggestions,
    _fmt_block_status_row,
    _fmt_input_verif_row,
    _fmt_failed_output_row,
    _fmt_block_snapshot_row,
)
from .summarize_pareto import (
    summarize_pareto_tool,
    _impl_summarize_pareto,
    _fmt_summarize_pareto_report,
    _dict_to_process_case,
    _fmt_pareto_front_section,
    _fmt_hv_section,
    _fmt_sensitivity_section,
)
from .discover_tunables import (
    discover_tunables_tool,
    discover_tunables_impl,
    _scan_aspen_file,
    _build_tunable_variables,
    _build_readable_targets,
    _compute_semantic_coverage,
    _serialize_report,
)

# 共享工具（供测试直接访问）
from ._common import (
    _load_yaml_raw,
    _resolve_config_path,
    _import_run_time_deps,
    _import_pareto_deps,
)


def get_agent_tools() -> list[BaseTool]:
    """返回所有 PAO agent 工具列表，供 graph 统一注册。

    用法：
        from src.agents.tools import get_agent_tools
        from langgraph.prebuilt import ToolNode

        tools = get_agent_tools()
        tool_node = ToolNode(tools)
        model = ChatAnthropic(...).bind_tools(tools)
    """
    return [
        load_case_config_tool,
        validate_config_tool,
        run_case_tool,
        optimize_pareto_tool,
        query_simulation_db_tool,
        query_node_db_tool,
        diagnose_case_tool,
        summarize_pareto_tool,
        discover_tunables_tool,
    ]


__all__ = [
    # 工具实例
    "load_case_config_tool",
    "load_config_tool",
    "validate_config_tool",
    "run_case_tool",
    "optimize_pareto_tool",
    "query_simulation_db_tool",
    "query_node_db_tool",
    "discover_tunables_tool",
    # 工具注册
    "get_agent_tools",
    # 核心实现（供测试）
    "_impl_load_config",
    "_impl_validate_config",
    "_impl_run_case",
    "_impl_optimize_pareto",
    "_impl_query_simulation_db",
    "_impl_query_node_db",
    # 格式化辅助（供测试）
    "_fmt_validation_warnings",
    "_fmt_case_summary",
    "_fmt_pareto_front_lines",
    "_fmt_hv_trend",
    "_fmt_pareto_result_summary",
    "_fmt_case_row",
    "_fmt_objective_row",
    "_fmt_query_cases_report",
    "_fmt_query_objective_report",
    "_fmt_get_case_report",
    "_fmt_node_values_report",
    "_fmt_path_search_report",
    "_fmt_recurring_errors_report",
    "_fmt_catalog_report",
    "_fmt_manifest_report",
    # diagnose_case_tool
    "diagnose_case_tool",
    "_impl_diagnose_case",
    "_fmt_diagnose_report",
    "_build_suggestions",
    "_fmt_block_status_row",
    "_fmt_input_verif_row",
    "_fmt_failed_output_row",
    "_fmt_block_snapshot_row",
    # summarize_pareto_tool
    "summarize_pareto_tool",
    "_impl_summarize_pareto",
    "_fmt_summarize_pareto_report",
    "_dict_to_process_case",
    "_fmt_pareto_front_section",
    "_fmt_hv_section",
    "_fmt_sensitivity_section",
    # 校验辅助（供测试）
    "_check_sim_file",
    "_check_design_var_sanity",
    "_check_objective_sanity",
    "_check_constraint_sanity",
    "_check_optimizer_sanity",
    "_run_python_parse",
    # 共享工具（供测试）
    "_load_yaml_raw",
    "_resolve_config_path",
    "_resolve_db_path",
    "_parse_design_vars_json",
    "_build_initial_design_vars",
    "_apply_derived_and_repair",
    "_parse_tags_str",
    "_import_run_time_deps",
    "_import_pareto_deps",
    # discover_tunables
    "discover_tunables_tool",
    "discover_tunables_impl",
    "_build_tunable_variables",
    "_build_readable_targets",
    "_compute_semantic_coverage",
    "_serialize_report",
]
