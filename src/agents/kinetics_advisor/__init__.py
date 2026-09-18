"""
kinetics_advisor — 反应动力学参数顾问 agent。

两条独立入口：
  recommend_kinetics_bounds / recommend_kinetics_bounds_agent
      冷启动边界推荐（无实验数据），风格对齐 boundary_advisor。
  fit_kinetics_params
      从用户实验数据（温度-速率常数点）拟合 Arrhenius 参数（Ea/k0），
      数值计算全部由确定性代码（fitting.py）完成，LLM 仅可选地对结果做
      自然语言解读，绝不参与任何计算。

硬约束：本 agent 产出的推荐结果不会自动写入 ConfigDraft，写入前仍需经过
write_feasibility_node 的 COM 试写验证和 human_confirm_node 的用户确认。

公开接口：
  recommend_kinetics_bounds        纯规则兜底(无 LLM 依赖)
  recommend_kinetics_bounds_agent  LLM 层,失败降级规则兜底
  fit_kinetics_params               数据拟合(确定性)+可选 LLM 解读
  KineticsBoundsReport / KineticsFitOutcome  结果容器
  KineticsVarMeta / KineticsDataPoint        输入数据结构（从 models.kinetics 转发）
"""
from src.agents.kinetics_advisor.agent import (
    KineticsBoundsReport,
    KineticsFitOutcome,
    fit_kinetics_params,
    recommend_kinetics_bounds,
    recommend_kinetics_bounds_agent,
)
from src.models.kinetics import KineticsDataPoint, KineticsVarMeta

__all__ = [
    "KineticsVarMeta",
    "KineticsDataPoint",
    "KineticsBoundsReport",
    "KineticsFitOutcome",
    "recommend_kinetics_bounds",
    "recommend_kinetics_bounds_agent",
    "fit_kinetics_params",
]
