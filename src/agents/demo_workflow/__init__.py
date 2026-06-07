"""
src/agents/demo_workflow/__init__.py

re-export 对外接口，调用方无需感知内部文件结构。

对外接口：
  from src.agents.demo_workflow import run_demo_case_workflow
  from src.agents.demo_workflow import prepare_demo_workflow_state
  from src.agents.demo_workflow import build_demo_workflow_report
  from src.agents.demo_workflow import DemoWorkflowState, WorkflowStep
"""
from src.agents.demo_workflow.workflow import run_demo_case_workflow
from src.agents.demo_workflow.helpers import prepare_demo_workflow_state
from src.agents.demo_workflow.report import build_demo_workflow_report
from src.agents.demo_workflow.state import DemoWorkflowState, WorkflowStep

__all__ = [
    "run_demo_case_workflow",
    "prepare_demo_workflow_state",
    "build_demo_workflow_report",
    "DemoWorkflowState",
    "WorkflowStep",
]
