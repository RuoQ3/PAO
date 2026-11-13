from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import json

from src.agents.orchestration.models import Answer, ExecutionMode, SessionState
from src.agents.errors import ensure_session_available, session_boundary


@dataclass
class ConversationSession:
    state: SessionState
    lead: Any

    @session_boundary
    def submit(self, text: str) -> Answer:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入非空问题。")
        ensure_session_available(self.state)
        from src.agents.subagents.onboarding_agent.project import ProjectOnboarding
        project = ProjectOnboarding(self)
        command, _, argument = text.strip().partition(" ")
        if command == "/project":
            return self._record_answer(project.command(argument))
        if command == "/capabilities":
            return self._record_answer(self.lead.capabilities(self.state))
        if project.draft is not None and project.draft.status in {"draft", "confirmed"}:
            self.state.history.append({"role": "user", "content": text})
            return self._record_answer(project.turn(text))
        self.state.history.append({"role": "user", "content": text})
        task = self.state.pending_task
        confirmation = text.strip().casefold() in {"确认", "确认执行", "confirm", "yes", "批准"}
        if (confirmation and self.state.pending_approval is None and task is not None
                and task.phase == "ready"
                and any(e.get("event") == "approval_invalidated_on_restore" for e in task.audit)):
            self.lead._revoke_restored_approvals(self.state)
            # Restoration never grants execution authority. Revalidate the task
            # and create a fresh Gateway preview; a later confirmation is needed.
            answer = self.lead._continue_task(task, self.state)
            answer.text = "恢复的审批已失效；以下为重新校验的计划，请重新审核后另行确认。\n" + answer.text
            return self._record_answer(answer)
        if confirmation and self.state.pending_approval is None and (task is None or not task.active):
            executions = [e for e in self.state.audit_events if e.get("event") == "action_executed"]
            suffix = (f"上次动作 {executions[-1]['result'].get('audit_id', '')} 已有执行记录，不能重复确认；如需新运行，请先生成并审核新计划。"
                      if executions else "请先提出任务并审核具体计划，再确认执行。")
            return self._record_answer(Answer(text="当前没有待审批的执行计划。" + suffix))
        if (text.strip().casefold() in {"确认", "确认执行", "confirm", "yes", "批准"}
                and self.state.pending_approval is not None
                and (task is None or task.phase == "awaiting_approval")):
            result = self.lead.confirm_action(self.state, approved=True)
            answer = self._action_answer(result)
            return self._record_answer(answer)
        if task is None or not task.active:
            self._update_focus_from_turn(text)
        answer = self.lead.answer(text, self.state)
        return self._record_answer(answer)

    @staticmethod
    def _action_answer(result: Any) -> Answer:
        """Adapt the gateway payload once, at the conversation boundary."""
        if isinstance(result, Answer):
            return result
        if not isinstance(result, dict):
            raise TypeError(f"Unsupported action result: {type(result).__name__}")
        report = result.get("answer")
        if isinstance(report, dict) and "text" in report:
            return Answer.from_dict(report)
        from src.agents.errors import unavailable
        raise unavailable("agent_output_invalid", stage="action_answer",
                          user_message="Agent 不可用：动作结果缺少已验证的分析。已执行结果保留在审计记录中。")

    def _record_answer(self, answer):
        from src.agents.reasoning import redact
        import logging
        payload = redact(answer.to_dict(), getattr(self.lead, "llm_config", None))
        self.state.history.append({"role": "assistant", "content": answer.text, "answer": payload})
        self.state.audit_events.append({"event": "answer_created", "answer": payload})
        logging.getLogger(__name__).info("Answer audit: %s", json.dumps(payload, ensure_ascii=False),
                                        extra={"pao_file_only": True})
        return answer

    @property
    def last_answer(self):
        for turn in reversed(self.state.history):
            if turn.get("answer"):
                return Answer.from_dict(turn["answer"])
        return None

    def _update_focus_from_turn(self, text: str, *, answer: Any = None) -> None:
        """Persist explicit conversation targets; never scan arbitrary history."""
        focus = self.state.runtime_state.setdefault("conversation_focus", {})
        ids = re.findall(r"(?<![A-Za-z0-9])([A-Za-z]{1,4}\d{3,4})(?![A-Za-z0-9])", str(text))
        streams = re.findall(r"(?<![A-Za-z0-9.])\d{4}(?![A-Za-z0-9.])", str(text))
        comps = re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9_-]{1,15})(?![A-Za-z0-9])", str(text))
        comps = [x for x in comps if x not in {"ASPEN", "GA", "PSO", "BO", "NSGA-II", "YAML"}
                 and not re.match(r"^[A-Z]{1,4}\d{3,4}$", x)]
        if ids:
            focus["equipment"] = ids[-1]
        if streams:
            focus["stream"] = streams[-1]
        if comps:
            focus["component"] = comps[0]
        low = str(text).casefold()
        if any(x in low for x in ("mole fraction", "molar fraction", "富集", "摩尔分数")): focus["metric"] = "mole_fraction"
        elif any(x in low for x in ("purity", "纯度")): focus["metric"] = "purity"
        elif any(x in low for x in ("recovery", "回收率")): focus["metric"] = "recovery"

    @session_boundary
    def propose_closed_loop(self, text: str, *, config: Any, **kwargs: Any):
        """Route an optimization request through the same LeadAgent session."""
        self.state.history.append({"role": "user", "content": text})
        answer = self.lead.propose_closed_loop(question=text, config=config, session=self.state, **kwargs)
        return self._record_answer(answer)

    @session_boundary
    def confirm_closed_loop(self, *, approved: bool = True):
        result = self.lead.confirm_closed_loop(self.state, approved=approved)
        return self._record_answer(self._action_answer(result))

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()

    def save_checkpoint(self, path) -> None:
        """Persist conversation state separately from numerical runtime caches."""
        from src.agents.orchestration.closed_loop.journal import SessionJournal
        with SessionJournal(path) as journal:
            journal.save({"kind": "conversation", "version": 1, "session": self.to_dict()})

    @classmethod
    def from_checkpoint(cls, path, lead):
        from src.agents.orchestration.closed_loop.journal import SessionJournal
        from src.agents.errors import internal_error
        with SessionJournal(path) as journal:
            payload = journal.load()
        if not payload or payload.get("kind") != "conversation" or payload.get("version") != 1:
            raise internal_error("conversation_checkpoint_invalid", stage="session_restore")
        session = cls(SessionState.from_dict(payload["session"]), lead)
        lead._revoke_restored_approvals(session.state)
        if session.state.runtime_state.get("project_draft"):
            from src.agents.subagents.onboarding_agent.project import ProjectOnboarding
            project = ProjectOnboarding(session)
            draft = project.draft
            draft.confirmed_digest, draft.status = None, "draft"
            for entry in draft.fields.values():
                entry["confirmed"] = False
            project._clear()
            project._store(draft)
            project._refresh(draft)
        return session
