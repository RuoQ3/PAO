from pathlib import Path
from types import SimpleNamespace

from src.agents.context import ContextLoader
from src.agents.evidence.models import ProcessContext
from src.agents.lead import LeadAgent
from src.agents.orchestration import SessionState


def _context():
    return ProcessContext(
        model_id="demo", equipment=[{"name": "T1"}], streams=[{"name": "S1"}],
        topology={"nodes": [{"id": "T1"}]}, variables={"rr": {"source": "yaml"}},
        objectives=[{"name": "purity"}], constraints=[{"name": "pressure"}],
        current_values={"/T1": 1}, equipment_parameters={"T1": [{"name": "TEMP"}]},
        history_evidence=[], sources=[], diagnostics=[],
    )


class Builder:
    def __init__(self):
        self.calls = 0

    def build(self, **kwargs):
        self.calls += 1
        return _context()


def test_none_does_not_load_project_context():
    builder = Builder()
    loaded = ContextLoader(builder).load(["none"])
    assert loaded.context is None
    assert builder.calls == 0


def test_process_context_is_loaded_and_summarized():
    builder = Builder()
    loaded = ContextLoader(builder).load(["process_context"])
    assert loaded.context.model_id == "demo"
    assert loaded.summary["equipment"] == 1
    assert builder.calls == 1


def test_projection_limits_unrequested_large_sources():
    loaded = ContextLoader(Builder()).load(["yaml"])
    assert loaded.context.variables
    assert loaded.context.equipment == []
    assert loaded.context.history_evidence == []


def test_selected_file_is_read_only_and_provenance_tracked():
    path = Path("README.md")
    loaded = ContextLoader(Builder()).load(["files"], files=[str(path)])
    assert loaded.context is not None
    assert any(item.get("source") == "file" for item in loaded.sources)


def test_lead_persists_context_summary_and_sources():
    builder = Builder()
    state = SessionState()
    LeadAgent(context_builder=builder, use_llm=False).answer("设备有哪些？", state)
    assert state.context_summary["equipment"] == 1
    assert isinstance(state.context_sources, list)
