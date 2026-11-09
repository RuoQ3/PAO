from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import json

from src.agents.orchestration.models import Answer, ExecutionMode, SessionState


@dataclass
class ConversationSession:
    state: SessionState
    lead: Any

    def submit(self, text: str) -> Answer:
        self.state.history.append({"role": "user", "content": text})
        if text.strip().casefold() in {"确认", "确认执行", "confirm", "yes", "批准"} and self.state.pending_approval is not None:
            result = self.lead.confirm_action(self.state, approved=True)
            answer = self._action_answer(result)
            self.state.history.append({"role": "assistant", "content": answer.text})
            return answer
        self._update_focus_from_turn(text)
        answer = self.lead.answer(text, self.state)
        self.state.history.append({"role": "assistant", "content": answer.text})
        return answer

    @staticmethod
    def _action_answer(result: Any) -> Answer:
        """Adapt the gateway payload once, at the conversation boundary."""
        if isinstance(result, Answer):
            return result
        if not isinstance(result, dict):
            raise TypeError(f"Unsupported action result: {type(result).__name__}")
        report = result.get("answer") or {}
        if isinstance(report, dict) and "backtest_conclusion" in report:
            facts = list(report.get("observed_facts") or [])
            limitations = list(report.get("limitations") or [])
            lines = ["参数影响回测结果："]
            for fact in facts:
                lines.append(f"已运行 {fact.get('case_count', 0)} 个工况，收敛 {fact.get('converged_case_count', 0)} 个，目标指标有效 {fact.get('valid_case_count', 0)} 个，未执行 {fact.get('skipped_case_count', 0)} 个。")
            lines.append(str(report["backtest_conclusion"]))
            # Full diagnostics stay in structured facts/uncertainties and the
            # file log. Terminal text contains only bounded user summaries.
            lines.extend("失败原因：" + str(x) for x in report.get("failure_summaries", []))
            brief_limits = report.get("user_limitations", limitations)
            lines.extend("限制：" + str(x).splitlines()[0][:160] for x in brief_limits[:3])
            if limitations or report.get("failure_summaries"):
                lines.append("详细诊断请查看日志文件。")
            return Answer(text="\n".join(lines),
                          facts=[*facts, {"kind": "action_result", "result": result}],
                          uncertainties=limitations)
        return Answer(text="动作执行结果：" + json.dumps(result, ensure_ascii=False, default=str),
                      facts=[{"kind": "action_result", "result": result}])

    def _update_focus_from_turn(self, text: str, *, answer: Any = None) -> None:
        """Persist explicit conversation targets; never scan arbitrary history."""
        focus = self.state.runtime_state.setdefault("conversation_focus", {})
        ids = re.findall(r"(?<![A-Za-z0-9])([A-Za-z]{1,4}\d{3,4})(?![A-Za-z0-9])", str(text))
        streams = re.findall(r"(?<![A-Za-z0-9])\d{4}(?![A-Za-z0-9])", str(text))
        comps = re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9_-]{1,15})(?![A-Za-z0-9])", str(text))
        comps = [x for x in comps if x != "ASPEN" and not re.match(r"^[A-Z]{1,4}\d{3,4}$", x)]
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
        # Resolve the explicitly requested bottom outlet using the loaded
        # process graph.  Multiple candidates remain unresolved.
        if "塔底" in str(text) and focus.get("equipment"):
            try:
                context = self.lead._context(self.state)
                candidates = [s for s in (getattr(context, "streams", []) or [])
                              if str(s.get("source") or s.get("source_node") or "").casefold() == str(focus["equipment"]).casefold()
                              and any(k in str(s).casefold() for k in ("bottom", "塔底", "bottoms"))]
                if len(candidates) == 1:
                    focus["stream"] = str(candidates[0].get("id") or candidates[0].get("name"))
            except Exception:
                pass

    def propose_closed_loop(self, text: str, *, config: Any, **kwargs: Any):
        """Route an optimization request through the same LeadAgent session."""
        self.state.history.append({"role": "user", "content": text})
        answer = self.lead.propose_closed_loop(question=text, config=config, session=self.state, **kwargs)
        self.state.history.append({"role": "assistant", "content": answer.text})
        return answer

    def confirm_closed_loop(self, *, approved: bool = True):
        return self.lead.confirm_closed_loop(self.state, approved=approved)

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()


