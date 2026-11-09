from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from src.agents.orchestration.models import ExecutionMode, SessionState


@dataclass
class ConversationSession:
    state: SessionState
    lead: Any

    def submit(self, text: str):
        self.state.history.append({"role": "user", "content": text})
        if text.strip().casefold() in {"确认", "确认执行", "confirm", "yes", "批准"} and self.state.pending_approval is not None:
            result = self.lead.confirm_action(self.state, approved=True)
            assistant_text = getattr(result, "text", None) or ("参数影响回测已执行。" if isinstance(result, dict) else str(result))
            self.state.history.append({"role": "assistant", "content": assistant_text})
            return result
        self._update_focus_from_turn(text)
        answer = self.lead.answer(text, self.state)
        self._update_focus_from_turn(getattr(answer, "text", ""), answer=answer)
        self.state.history.append({"role": "assistant", "content": answer.text})
        return answer

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


