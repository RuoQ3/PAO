"""
src/agents/process_advisor/__init__.py

re-export 所有对外接口，调用方无需感知内部文件结构。

向后兼容：
  from src.agents.process_advisor import run_process_advisor
  from src.agents.process_advisor import run_process_advisor_agent
  from src.agents.process_advisor import ProcessAdvisorToolRunner
  from src.agents.process_advisor import ReadOnlyToolRunner
"""
from src.agents.process_advisor.agent import (
    run_process_advisor,
    run_process_advisor_agent,
)
from src.agents.process_advisor.tools import (
    ProcessAdvisorToolRunner,
    ReadOnlyToolRunner,
)
from src.agents.process_advisor.prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)

__all__ = [
    "run_process_advisor",
    "run_process_advisor_agent",
    "ProcessAdvisorToolRunner",
    "ReadOnlyToolRunner",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
]
