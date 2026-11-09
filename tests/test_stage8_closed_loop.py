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
    service = ClosedLoopService(runner=lambda driver, config, **kwargs: calls.append(kwargs) or {"status": "done", "used": 1})
    checkpoint = tmp_path / "agent.db"
    result = service.run(driver=None, config=SimpleNamespace(agent_loop={"max_evaluations": 3}), approved=True,
                         checkpoint_path=checkpoint, checkpoint_metadata={"intent": {"intent": "optimization_request"}})
    assert result["status"] == "done"
    assert calls
    saved = service.load_checkpoint(checkpoint)
    assert saved["status"] == "completed"
    assert saved["metadata"]["intent"]["intent"] == "optimization_request"


def test_lead_closed_loop_does_not_start_before_confirmation():
    calls = []
    service = ClosedLoopService(runner=lambda *args, **kwargs: calls.append(True) or {"status": "done"})
    lead = LeadAgent(action_gateway=None, closed_loop_service=service, use_llm=False)
    # No gateway means proposal remains pending and no service call occurs.
    session = SessionState()
    answer = lead.propose_closed_loop(question="run optimization", config=SimpleNamespace(), session=session)
    assert answer.approval_request is not None
    assert calls == []


def test_service_checkpoint_does_not_contain_driver_object(tmp_path: Path):
    service = ClosedLoopService(runner=lambda driver, config, **kwargs: {"status": "done"})
    driver = object()
    checkpoint = tmp_path / "checkpoint.db"
    service.run(driver=driver, config=SimpleNamespace(agent_loop={}), approved=True,
                checkpoint_path=checkpoint, checkpoint_metadata={"driver": "excluded"})
    raw = checkpoint.read_bytes()
    assert b"object at" not in raw
    assert b"driver" in raw
