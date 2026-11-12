from agent_fakes import fake_kwargs, CFG, FakeTechnicalAdvisor
from src.agents.errors import AgentUnavailableError
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.lead import ConversationSession, LeadAgent
from src.agents.orchestration import SessionState
from src.workflows.agent_entry import run_configured_agent


def test_conversation_session_routes_turns_to_lead_and_preserves_history():
    session = ConversationSession(SessionState(), LeadAgent(**fake_kwargs(kind="chat")))
    answer = session.submit("hello")
    assert answer.text
    assert session.state.history[0]["role"] == "user"
    assert session.state.history[-1]["role"] == "assistant"


def test_closed_loop_entry_uses_same_conversation_session():
    session = ConversationSession(SessionState(), LeadAgent(**fake_kwargs(kind="optimization_request", action="optimize")))
    answer = session.propose_closed_loop("run optimization", config=SimpleNamespace())
    assert answer.approval_request is not None
    assert session.state.history[0]["content"] == "run optimization"


def test_internal_workflow_adapter_rejects_public_bypass():
    with pytest.raises(PermissionError):
        run_configured_agent(None, SimpleNamespace())


def test_normal_optimization_module_entry_remains_separate():
    # The ordinary CLI function is still present and does not import LeadAgent
    # as part of its public module contract.
    source = Path("src/main.py").read_text(encoding="utf-8")
    assert "if args.agent" not in source
    assert "if args.conversation:" in source
    assert "build_optimization_plan" in source and "gateway.execute" in source


def test_removed_agent_flags_are_not_public_cli_options():
    source = Path("src/main.py").read_text(encoding="utf-8")
    assert '"--agent"' not in source
    assert '"--chat"' not in source
    assert '"--unified-chat"' not in source


def test_main_reconfigures_stdin_for_utf8_agent_prompts():
    source = Path("src/main.py").read_text(encoding="utf-8")
    assert 'sys.stdin.reconfigure(encoding="utf-8"' in source
