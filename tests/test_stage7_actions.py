from datetime import timedelta

import pytest

from src.agents.actions import ActionGateway, ActionRequest, ActionType, ModelIdentity, ModelSchema, MockExecutor, ParameterSpec
from src.agents.actions.gateway import ApprovalError
from src.agents.orchestration.models import ActionProposal


class FakeExecutor(MockExecutor):
    def __init__(self):
        super().__init__()
        self.direct_aspen_access = False


def _gateway(*, executor=None, ttl_seconds=3600):
    schema = ModelSchema(
        identity=ModelIdentity("model", "v1"), targets=("model",),
        parameters={"\\Data\\Blocks\\T1\\Input\\RR": ParameterSpec("\\Data\\Blocks\\T1\\Input\\RR", minimum=0, maximum=1, impact_targets=("T1",))},
        current_values={"\\Data\\Blocks\\T1\\Input\\RR": 0.5},
        permissions={"inspect", "set_parameter", "run_simulation", "run_optimization"},
    )
    return ActionGateway(schema, executor=executor or FakeExecutor(), ttl_seconds=ttl_seconds)


def test_proposal_preview_carries_payload_evidence_risks_and_requires_approval():
    gateway = _gateway()
    proposal = ActionProposal("set_parameter", "\\Data\\Blocks\\T1\\Input\\RR", reason="test", payload={"new_value": 0.7, "model_id": "model"}, evidence_ids=["yaml.variable.rr"], risks=["sensitivity"])
    preview = gateway.preview_proposal(proposal)
    assert preview["valid"] is True
    assert preview["payload"]["new_value"] == 0.7
    assert preview["evidence"] == ["yaml.variable.rr"]
    assert preview["audit_id"]


def test_confirmed_action_executes_and_audits():
    executor = FakeExecutor()
    gateway = _gateway(executor=executor)
    action = ActionRequest(ActionType.SET_PARAMETER, "\\Data\\Blocks\\T1\\Input\\RR", new_value=0.7, reason="change")
    result = gateway.submit(action)
    assert result.valid
    gateway.preview(action.audit_id)
    gateway.approve(action.audit_id)
    output = gateway.execute(action.audit_id)
    assert output["status"] == "simulated" and not output["real_execution"]
    assert executor.calls
    assert any(event.event_type.value == "execution" for event in gateway.audit.events)


def test_preview_and_validation_failures_are_not_executable():
    gateway = _gateway()
    bad = ActionProposal("set_parameter", "\\Data\\Blocks\\T1\\Input\\RR", reason="bad", payload={"new_value": 2.0})
    preview = gateway.preview_proposal(bad)
    assert preview["valid"] is False
    assert "audit_id" not in preview
    structural = ActionProposal("modify_structure", "model", reason="no")
    rejected = gateway.preview_proposal(structural)
    assert rejected["valid"] is False


def test_expired_action_cannot_be_approved():
    gateway = _gateway(ttl_seconds=0)
    action = ActionRequest(ActionType.SET_PARAMETER, r"\Data\Blocks\T1\Input\RR", new_value=.7, reason="simulate")
    assert gateway.submit(action).valid
    with pytest.raises(ApprovalError):
        gateway.approve(action.audit_id)


def test_read_action_maps_to_inspect_and_does_not_expose_executor_sdk():
    executor = FakeExecutor()
    gateway = _gateway(executor=executor)
    proposal = ActionProposal("read", "model", reason="inspect", payload={"model_id": "model"}, requires_confirmation=False)
    preview = gateway.preview_proposal(proposal)
    assert preview["valid"]
    assert preview["action_type"] == "inspect"
    assert executor.direct_aspen_access is False
