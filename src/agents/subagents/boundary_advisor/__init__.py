"""
boundary_advisor — 搜索边界顾问 agent。

基于初始收敛解 + 变量元信息(路径/单位/初始值/语义角色),为每个设计变量
推理出合理的搜索倍数 k,生成自适应搜索边界。是系统化方案的"冷启动边界判断"层。

公开接口：
  recommend_boundaries        兼容入口（严格 LLM 校验）
  recommend_boundaries_agent  LLM 层，失败立即终止
  format_boundary_report      报告格式化
  VarMeta / BoundaryReport / BoundaryRecommendation  数据容器
"""
from src.agents.subagents.boundary_advisor.agent import (
    BoundaryRecommendation,
    BoundaryReport,
    format_boundary_report,
    recommend_boundaries,
    recommend_boundaries_agent,
)
from src.agents.subagents.boundary_advisor.tools import VarMeta

__all__ = [
    "VarMeta",
    "BoundaryRecommendation",
    "BoundaryReport",
    "recommend_boundaries",
    "recommend_boundaries_agent",
    "format_boundary_report",
]
