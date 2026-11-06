from __future__ import annotations

from typing import Any

from src.agents.orchestration.models import ActionProposal, Answer, ApprovalRequest, Evidence, ExecutionMode, RuntimePolicy, SessionState, TaskSpec
from src.agents.orchestration.runtime import OrchestrationRuntime
from src.agents.subagents.registry import SubagentRegistry, default_registry


_PAO_TERMS = ("aspen", "仿真", "优化", "pareto", "边界", "动力学", "物性", "流程", "工艺")


class LeadAgent:
    """Single user-facing decision maker; execution remains in Runtime/Gateway."""

    def __init__(self, *, registry: SubagentRegistry | None = None,
                 policy: RuntimePolicy | None = None, context_builder: Any = None,
                 action_gateway: Any = None):
        self.registry = registry or default_registry()
        self.policy = policy or RuntimePolicy()
        self.context_builder = context_builder
        self.action_gateway = action_gateway

    def answer(self, question: str, session: SessionState | None = None) -> Answer:
        session = session or SessionState(mode=self.policy.mode)
        if self._is_general(question):
            return Answer(text=question, facts=[{"kind": "direct_answer", "text": question}],
                          evidence=[Evidence("user", "prompt", "User request", question, confidence=1.0)])
        if self._is_action_request(question):
            proposal = ActionProposal(action_type="inspect", target="project",
                                      reason=question, requires_confirmation=True)
            approval = ApprovalRequest(proposal=proposal, message="Human confirmation is required before execution.")
            if self.action_gateway is not None:
                try:
                    approval.preview = self.action_gateway.preview_proposal(proposal)
                except Exception as exc:
                    approval.preview = {"status": "preview_failed", "error": str(exc)}
            session.pending_approval = approval
            return Answer(text="Action proposal created; human confirmation is required before execution.",
                          recommendations=[proposal.to_dict()], approval_request=approval)
        task_type, agent = self._route(question)
        task = TaskSpec(task_type=task_type, prompt=question, agent=agent,
                        max_steps=self.policy.max_steps_per_task, budget=self.policy.max_steps_per_task)
        runtime = OrchestrationRuntime(self.registry.as_mapping(), policy=self.policy, audit=session.audit_events)
        results = runtime.run([task], context=self._context(session))
        issues = runtime.validate_results(results)
        session.task_results.extend(results)
        result = results[0]
        text = result.summary if not issues else "无法形成有充分证据的结论：" + "；".join(issues)
        if result.status.value != "completed":
            text = "任务未完成：" + ("；".join(result.errors) or result.status.value)
        return Answer(text=text, facts=result.facts, evidence=result.evidence,
                      assumptions=result.assumptions, uncertainties=result.uncertainties,
                      recommendations=result.recommended_actions, task_results=results)

    @staticmethod
    def _is_general(question: str) -> bool:
        return not any(term in question.casefold() for term in _PAO_TERMS)

    @staticmethod
    def _route(question: str) -> tuple[str, str]:
        q = question.casefold()
        if "动力学" in q or "kinetic" in q:
            return "kinetics_analysis", "kinetics_advisor"
        if "边界" in q or "boundary" in q:
            return "boundary_review", "boundary_advisor"
        if "优化" in q or "pareto" in q:
            return "optimization_analysis", "optimization_advisor"
        if "风险" in q or "risk" in q:
            return "general_question", "risk_reviewer"
        return "project_research", "project_researcher"

    def _context(self, session: SessionState) -> Any:
        if self.context_builder is None:
            return None
        try:
            return self.context_builder.build(session=session)
        except TypeError:
            return self.context_builder.build(session=session, model_id=None)

    @staticmethod
    def _is_action_request(question: str) -> bool:
        q = question.casefold()
        return any(token in q for token in ("set ", "修改", "写入", "运行仿真", "执行优化", "改变边界"))
