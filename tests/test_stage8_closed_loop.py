from agent_fakes import fake_kwargs, CFG, FakeTechnicalAdvisor
from src.agents.errors import AgentUnavailableError
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.orchestration.closed_loop.service import ClosedLoopService
from src.agents.lead import LeadAgent
from src.agents.orchestration import SessionState


def test_closed_loop_service_requires_explicit_confirmation():
    calls = []
    service = ClosedLoopService(runner=lambda driver, config, **kwargs: calls.append((driver, config)) or {"status": "done"})
    with pytest.raises(PermissionError):
        service.run(driver=object(), config=object())
    assert calls == []


def test_confirmed_closed_loop_runs_fake_runner_and_writes_metadata(tmp_path: Path):
    calls = []
    service = ClosedLoopService(runner=lambda *a, **k: calls.append(True))
    checkpoint = tmp_path / "agent.db"
    with pytest.raises(PermissionError, match="disabled"):
        service.run(config=SimpleNamespace(), approved=True, checkpoint_path=checkpoint)
    assert calls == [] and not checkpoint.exists()



def test_lead_closed_loop_does_not_start_before_confirmation():
    calls = []
    service = ClosedLoopService(runner=lambda *args, **kwargs: calls.append(True) or {"status": "done"})
    lead = LeadAgent(action_gateway=None, closed_loop_service=service, **fake_kwargs(kind="optimization_request", action="optimize"))
    # No gateway means proposal remains pending and no service call occurs.
    session = SessionState()
    answer = lead.propose_closed_loop(question="run optimization", config=SimpleNamespace(), session=session)
    assert answer.approval_request is not None
    assert calls == []


def test_service_checkpoint_does_not_contain_driver_object(tmp_path: Path):
    service = ClosedLoopService()
    checkpoint = tmp_path / "checkpoint.db"
    service.save_checkpoint(checkpoint, {"status": "read_only", "driver": "excluded"})
    assert service.load_checkpoint(checkpoint)["driver"] == "excluded"
    assert b"object at" not in checkpoint.read_bytes()
