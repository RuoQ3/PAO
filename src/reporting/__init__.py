# src/reporting — 优化结果可视化与报告生成
from .plot_pareto import generate_pareto_report
from .summary_report import (
    generate_tac_breakdown,
    generate_emissions_summary,
    generate_variable_importance,
    generate_failure_summary,
    generate_summary_report,
)

__all__ = [
    "generate_pareto_report",
    "generate_tac_breakdown",
    "generate_emissions_summary",
    "generate_variable_importance",
    "generate_failure_summary",
    "generate_summary_report",
]
