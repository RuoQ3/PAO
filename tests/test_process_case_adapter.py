"""Contract tests for the configuration-driven read-only case boundary."""
from pathlib import Path

from src.agents.context import CaseInputConfig, ProcessCaseAdapter
from src.agents import ConversationSession, LeadAgent, SessionState


def test_memory_case_has_one_stable_context_contract():
    adapter = ProcessCaseAdapter.from_config({
        "case": {"id": "case-a"},
        "memory": {
            "equipment_nodes": [{"id": "u1", "name": "unit", "block_type": "PUMP"}],
            "streams": [{"id": "feed", "target": "u1", "composition": {"A": 1.0}}],
        },
    })
    context = adapter.load()
    assert context.model_id == "case-a"
    assert context.get_equipment("u1")["role"] == "pump"
    assert context.get_stream("feed")["stream_type"] == "feed"
    assert context.sources


def test_paths_are_case_local_and_absolute_paths_are_preserved():
    base = Path("cases/demo_case").resolve()
    cfg = CaseInputConfig.from_mapping({"case": {"aspen_file": "二级氢氰化工段.bkp"}}, base_dir=base)
    assert cfg.aspen_file == base / "二级氢氰化工段.bkp"
    absolute = (base / "二级氢氰化工段.bkp").resolve()
    cfg = CaseInputConfig.from_mapping({"case": {"aspen_file": str(absolute)}}, base_dir=Path("."))
    assert cfg.aspen_file == absolute


def test_missing_optional_sources_are_diagnostics_and_never_fabricated():
    adapter = ProcessCaseAdapter.from_config({
        "case": {"id": "missing", "node_db": "missing.db", "simulation_db": "history.db"},
        "memory": {"equipment_nodes": [{"id": "u", "block_type": "UNKNOWN"}]},
    })
    context = adapter.load()
    assert any("NodeDB" in item for item in context.diagnostics)
    assert any("SimulationDB" in item for item in context.diagnostics)
    assert context.get_equipment("u")["role"] == "unknown"
    assert not any("simulated" in item.casefold() for item in context.diagnostics)


def test_config_file_does_not_exist_is_reported():
    adapter = ProcessCaseAdapter.from_config(Path("does-not-exist-adapter.yaml"))
    assert adapter.validate()
    context = adapter.load()
    assert context.status == "partial"


def test_source_conflict_survives_adapter_boundary():
    adapter = ProcessCaseAdapter.from_config({
        "case": {"id": "conflict"},
        "memory": {
            "equipment_nodes": [
                {"id": "u", "block_type": "PUMP", "provenance_source": "bkp"},
                {"id": "u", "block_type": "HEATER", "provenance_source": "yaml"},
            ]
        },
    })
    context = adapter.load()
    assert context.conflicts
    assert any("conflicting values retained" in item for item in context.diagnostics)


def test_adapter_load_is_cached_for_followup_turns():
    adapter = ProcessCaseAdapter.from_config({"case": {"id": "shared"}, "memory": {}})
    assert adapter.load() is adapter.load()


def test_lead_resolves_and_reuses_case_adapter_from_session_ref():
    session = SessionState(context_refs={"case_config": "case-config-does-not-exist.yaml"})
    lead = LeadAgent(use_llm=False)
    lead.answer("项目有哪些设备？", session)
    first = lead._case_adapters[session.session_id]
    lead.answer("流程怎么走？", session)
    assert lead._case_adapters[session.session_id] is first
