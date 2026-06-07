"""
src/agents/onboarding_agent/__init__.py

接入向导 Agent（B1）对外接口。

用法：
    from src.agents.onboarding_agent import OnboardingResult, run_onboarding, apply_user_feedback
"""
from src.agents.onboarding_agent.agent import (
    OnboardingResult,
    apply_user_feedback,
    run_onboarding,
)

__all__ = ["OnboardingResult", "run_onboarding", "apply_user_feedback"]
