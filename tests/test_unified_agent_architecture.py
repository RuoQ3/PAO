"""Smoke tests for the public unified Agent architecture."""
from pathlib import Path

from src.agents import LeadAgent, OrchestrationRuntime, SessionState
from src.agents.context import ProcessContextBuilder


def test_public_entry_points_and_top_level_layout_are_new_architecture_only():
    root = Path("src/agents")
    assert {p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"} == {
        "actions", "context", "evidence", "lead", "orchestration", "subagents"
    }
    assert LeadAgent and OrchestrationRuntime and SessionState


def test_lead_general_answer_is_read_only_and_context_builder_is_lazy():
    calls = []

    class FakeBuilder:
        def build(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("general questions must not load process context")

    answer = LeadAgent(context_builder=FakeBuilder()).answer("Explain a mass balance")
    assert answer.facts and not calls
    assert answer.text != "Explain a mass balance"


def test_rule_mode_does_not_echo_general_chat():
    answer = LeadAgent().answer("你好呀")
    assert answer.text != "你好呀"
    assert any(f.get("kind") == "rule_fallback" for f in answer.facts)


def test_process_context_builder_accepts_in_memory_config_without_aspen():
    context = ProcessContextBuilder(config={"design_variables": []}, read_only=True).build()
    assert context.status in {"ok", "partial"}
    assert context.model_id == "unknown"


def test_project_equipment_question_uses_yaml_context_without_aspen():
    from src.agents.context import ProcessContextBuilder

    builder = ProcessContextBuilder(
        config={"design_variables": [
            {"name": "rr", "aspen_path": r"\Data\Blocks\T1\Input\RR"},
            {"name": "temp", "aspen_path": r"\Data\Blocks\T2\Input\TEMP"},
        ]}, read_only=True,
    )
    answer = LeadAgent(context_builder=builder).answer("这个项目里有什么设备？")
    assert "T1" in answer.text and "T2" in answer.text
    assert any(item.get("kind") == "equipment_inventory" for item in answer.facts)


def test_yaml_conversation_reads_equipment_and_streams_from_bkp_index():
    """A YAML conversation still sees the referenced Aspen model without COM."""
    builder = ProcessContextBuilder(
        yaml_path="cases/demo_case/pareto_config.yaml", read_only=True,
    )
    context = builder.build()
    assert {item["name"] for item in context.equipment} >= {
        "P0301", "T0301", "T0302", "T0303", "R0303-1",
    }
    assert len(context.equipment) >= 15
    assert len(context.streams) >= 20
    assert any(item.get("source") == "bkp_index" for item in context.equipment)


def test_conversation_first_general_turn_does_not_inherit_current_history_as_project_scope():
    from src.agents import ConversationSession, SessionState

    class FakeBuilder:
        def build(self, **kwargs):
            raise AssertionError("a first general turn must not load process context")

    session = ConversationSession(SessionState(), LeadAgent(context_builder=FakeBuilder(), use_llm=False))
    answer = session.submit("???")
    assert answer.facts
    assert session.state.context_summary == {}
    assert session.state.intent.project_scope == "none"
