from __future__ import annotations
from datetime import datetime, timedelta, timezone
from .audit import AuditLog
from .executor import MockExecutor
from .models import ActionRequest, ApprovalStatus, AuditEventType, ModelSchema, ActionPreview
from .validation import validate_action, ValidationResult

class ApprovalError(RuntimeError): pass

class ActionGateway:
    """Stateful approval gate. Execution is impossible before approval."""
    def __init__(self, schema: ModelSchema, *, executor=None, audit=None, ttl_seconds: int = 3600, simulation_db=None, simulation_runner=None):
        self.schema = schema; self.executor = executor or MockExecutor(); self.audit = audit or AuditLog(); self.ttl_seconds = ttl_seconds
        self.simulation_db = simulation_db
        self.simulation_runner = simulation_runner
        self._actions: dict[str, tuple[ActionRequest, ApprovalStatus, datetime, ActionPreview | None]] = {}
        self._backups: dict[str, object] = {}
    def submit(self, action: ActionRequest, *, actor="user") -> ValidationResult:
        # Read current values at the boundary immediately before validation.
        # This prevents approving a stale preview after another process changed
        # the Aspen document.
        if hasattr(self.executor, "refresh_schema"):
            self.executor.refresh_schema(self.schema)
        result = validate_action(action, self.schema, actor=actor)
        status = ApprovalStatus.PENDING if result.valid else ApprovalStatus.REJECTED
        self._actions[action.audit_id] = (action, status, datetime.now(timezone.utc), result.preview)
        self.audit.record(action.audit_id, AuditEventType.USER_REQUEST, actor=actor, status=status.value, payload=action.to_dict())
        self.audit.record(action.audit_id, AuditEventType.VALIDATION, status=status.value, message="; ".join(result.errors), payload={"warnings": result.warnings})
        return result

    def preview_proposal(self, proposal):
        """Translate a Lead proposal to a validated preview without executing it."""
        action_type = {"read": "inspect", "simulate_once": "run_simulation", "optimize": "run_optimization"}.get(proposal.action_type, proposal.action_type)
        payload = dict(getattr(proposal, "payload", {}) or {})
        action = ActionRequest(
            action_type=action_type,
            target=proposal.target,
            old_value=payload.get("old_value"), new_value=payload.get("new_value"),
            unit=payload.get("unit"), model_id=payload.get("model_id"),
            model_version=payload.get("model_version"), run_simulation=payload.get("run_simulation"),
            reason=proposal.reason,
            evidence=list(proposal.evidence_ids),
            risk=list(proposal.risks),
            requires_confirmation=proposal.requires_confirmation,
            payload=payload,
        )
        result = self.submit(action, actor="lead_agent")
        if not result.valid or result.preview is None:
            return {"valid": False, "errors": list(result.errors), "warnings": list(result.warnings)}
        return {"valid": True, "audit_id": action.audit_id, "preview": result.preview.to_dict(),
                "payload": payload, "evidence": list(proposal.evidence_ids), "risks": list(proposal.risks),
                "target": proposal.target, "action_type": action_type}
    def preview(self, audit_id: str) -> ActionPreview:
        action, status, created, preview = self._get(audit_id); self._expire_if_needed(audit_id)
        _, status, _, preview = self._get(audit_id)
        if status in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}: raise ApprovalError(f"action is {status.value}")
        if preview is None: raise ApprovalError("no valid preview")
        self.audit.record(audit_id, AuditEventType.PREVIEW, status=status.value, payload=preview.to_dict()); return preview
    def approve(self, audit_id: str, *, actor="user"):
        action, status, created, preview = self._get(audit_id); self._expire_if_needed(audit_id); action, status, created, preview = self._get(audit_id)
        if status != ApprovalStatus.PENDING: raise ApprovalError(f"cannot approve action in {status.value} state")
        self._actions[audit_id] = (action, ApprovalStatus.APPROVED, created, preview); self.audit.record(audit_id, AuditEventType.CONFIRMATION, actor=actor, status="approved")
    def reject(self, audit_id: str, *, actor="user"):
        action, status, created, preview = self._get(audit_id)
        if status in {ApprovalStatus.EXECUTED, ApprovalStatus.FAILED}: raise ApprovalError(f"cannot reject action in {status.value} state")
        self._actions[audit_id] = (action, ApprovalStatus.REJECTED, created, preview); self.audit.record(audit_id, AuditEventType.CONFIRMATION, actor=actor, status="rejected")
    def execute(self, audit_id: str, *, actor="system"):
        action, status, created, preview = self._get(audit_id); self._expire_if_needed(audit_id); action, status, created, preview = self._get(audit_id)
        if status != ApprovalStatus.APPROVED: raise ApprovalError(f"execution requires approved action, got {status.value}")
        result = validate_action(action, self.schema, actor=actor)
        if not result.valid:
            self._actions[audit_id] = (action, ApprovalStatus.FAILED, created, preview); self.audit.record(audit_id, AuditEventType.ERROR, status="failed", message="; ".join(result.errors)); raise ApprovalError("action invalid at execution: " + "; ".join(result.errors))
        try:
            if hasattr(self.executor, "backup"): self._backups[audit_id] = self.executor.backup(action)
            # A controlled parameter-impact plan is still dispatched through
            # this gateway.  The runner is optional and injected by the host;
            # without it the existing executor behavior is preserved.
            if action.action_type.value == "run_simulation" and isinstance(action.payload.get("simulation_plan"), dict):
                if self.simulation_runner is None:
                    raise RuntimeError("已创建回测计划，但当前运行环境没有配置 Aspen simulation_runner。")
                from src.agents.orchestration.parameter_impact import SimulationPlan
                sim_plan = SimulationPlan.from_dict(action.payload["simulation_plan"])
                sim_plan.approved = True
                cases = self.simulation_runner.run(sim_plan)
                if self.simulation_db is not None and hasattr(self.simulation_db, "save_cases"):
                    self.simulation_db.save_cases([c.to_dict() for c in cases])
                impact = action.payload.get("parameter_impact_plan") or {}
                try:
                    from src.agents.orchestration.parameter_impact import SensitivityAnalyzer
                    parameters = [str(x.get("aspen_path") or x.get("name")) for x in impact.get("parameters", []) if isinstance(x, dict)]
                    sensitivity = SensitivityAnalyzer().analyze(cases, metric=str(impact.get("metric") or "metric"), parameters=parameters)
                except Exception as exc:
                    sensitivity = {"error": f"sensitivity analysis failed: {type(exc).__name__}: {exc}"}
                output = {"audit_id": action.audit_id, "status": "executed", "action_type": action.action_type.value,
                          "simulation_cases": [c.to_dict() for c in cases], "sensitivity": sensitivity,
                          "planned_case_count": len(sim_plan.parameter_changes),
                          "skipped_case_count": max(0, len(sim_plan.parameter_changes) - len(cases))}
            else:
                output = self.executor.execute(action, preview=result.preview)
            if self.simulation_db is not None and hasattr(self.simulation_db, "save_action_audit"):
                self.simulation_db.save_action_audit(action.audit_id, {
                    "action": action.to_dict(), "preview": result.preview.to_dict() if result.preview else None,
                    "result": output,
                })
        except Exception as exc:
            self._actions[audit_id] = (action, ApprovalStatus.FAILED, created, preview); self.audit.record(audit_id, AuditEventType.ERROR, status="failed", message=str(exc)); raise
        self._actions[audit_id] = (action, ApprovalStatus.EXECUTED, created, preview); self.audit.record(audit_id, AuditEventType.EXECUTION, actor=actor, status="executed", payload=output); return output
    def rollback(self, audit_id: str):
        if audit_id not in self._backups: raise ApprovalError("no backup for action")
        if not hasattr(self.executor, "rollback"): raise ApprovalError("executor does not support rollback")
        self.executor.rollback(self._backups[audit_id]); self.audit.record(audit_id, AuditEventType.EXECUTION, status="rolled_back")
    def status(self, audit_id): self._expire_if_needed(audit_id); return self._get(audit_id)[1]
    def _get(self, audit_id):
        if audit_id not in self._actions: raise ApprovalError("unknown audit_id")
        return self._actions[audit_id]
    def _expire_if_needed(self, audit_id):
        action, status, created, preview = self._actions[audit_id]
        if status == ApprovalStatus.PENDING and datetime.now(timezone.utc) - created > timedelta(seconds=self.ttl_seconds): self._actions[audit_id] = (action, ApprovalStatus.EXPIRED, created, preview)
