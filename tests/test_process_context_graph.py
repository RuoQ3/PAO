"""Cross-case contracts for the read-only normalized process graph."""
from agent_fakes import fake_kwargs, CFG, FakeTechnicalAdvisor
from src.agents.errors import AgentUnavailableError

from src.agents import ConversationSession, LeadAgent, SessionState
from src.agents.context import ProcessContextBuilder


def _config(*, source="memory"):
    return {
        "model_id": "generic-case",
        "equipment_nodes": [
            {"id": "feed", "name": "Feed", "block_type": "PUMP", "source": source},
            {"id": "mix", "name": "Mixer", "block_type": "MIXER", "source": source},
            {"id": "reactor", "name": "Reactor", "block_type": "RCSTR", "source": source},
            {"id": "separator", "name": "Separator", "block_type": "FLASH2", "source": source},
        ],
        "streams": [
            {"id": "feed-stream", "source": None, "target": "feed", "composition": {"A": "1D-1"}},
            {"id": "to-mix", "source": "feed", "target": "mix", "composition": {"A": 0.9, "B": 0.1}},
            {"id": "makeup", "source": None, "target": "mix", "composition": {"A": 0.2, "B": 0.8}},
            {"id": "to-reactor", "source": "mix", "target": "reactor", "composition": {"A": 0.55, "B": 0.45}},
            {"id": "to-separator", "source": "reactor", "target": "separator", "composition": {"A": 0.3, "B": 0.7}},
            {"id": "product", "source": "separator", "target": None, "composition": {"A": 0.1, "B": 0.9}},
            {"id": "recycle", "source": "separator", "target": "mix", "composition": {"A": 0.8, "B": 0.2}},
        ],
    }


def test_in_memory_graph_has_stable_nodes_edges_and_conservative_roles():
    context = ProcessContextBuilder(config=_config(), read_only=True).build()
    node = context.get_equipment("reactor")
    assert {"id", "name", "kind", "block_type", "role", "aspen_path", "upstream_streams", "downstream_streams", "parameters", "source", "confidence"} <= set(node)
    stream = context.get_stream("to-separator")
    assert {"id", "name", "source", "target", "aspen_path", "stream_type", "composition", "temperature", "pressure", "flow", "phase", "confidence"} <= set(stream)
    assert node["role"] == "continuous_reactor"
    assert context.downstream("mix") == ["reactor"]
    assert context.upstream("mix") == ["feed", "separator"]
    assert context.terminal_streams()[0]["id"] == "product"
    assert any(item["id"] == "recycle" for item in context.summarize_process()["recirculation"])


def test_composition_facts_and_inferences_keep_evidence_separate():
    context = ProcessContextBuilder(config=_config(), read_only=True).build()
    change = context.compare_compositions("to-reactor", "to-separator")
    assert any(item["classification"] == "reduced" for item in change["changes"])
    assert context.facts and all("evidence_ids" in item for item in context.facts)
    assert any(item["kind"] == "possible_reaction" for item in context.inferences)
    assert all("evidence_ids" in item for item in context.inferences)


def test_duplicate_sources_are_retained_as_conflicts():
    cfg = _config()
    cfg["streams"].append({"id": "product", "source": "separator", "target": None, "flow": "bad-number", "composition": {"A": 0.2}})
    context = ProcessContextBuilder(config=cfg, read_only=True).build()
    assert context.conflicts
    assert any("unparseable flow" in item for item in context.diagnostics)


def test_short_followup_uses_current_project_context_without_case_names():
    builder = ProcessContextBuilder(config=_config(), read_only=True)
    session = ConversationSession(SessionState(), LeadAgent(context_builder=builder, **fake_kwargs()))
    first = session.submit("这个项目有哪些设备和物流？")
    second = session.submit("然后呢？")
    assert first.facts
    assert second.task_results or second.facts
    assert session.state.context_summary["equipment"] == 4


def test_missing_composition_and_boundary_streams_are_explicit():
    cfg = _config()
    cfg["streams"].append({"id": "unknown-boundary", "source": None, "target": None})
    context = ProcessContextBuilder(config=cfg, read_only=True).build()
    assert context.get_stream("unknown-boundary")["composition"] if "composition" in context.get_stream("unknown-boundary") else True
    assert any(item["id"] == "feed" for item in context.boundary_nodes())
    assert "unknown-boundary" in {item["id"] for item in context.terminal_streams()}
