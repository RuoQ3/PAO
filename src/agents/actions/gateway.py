from __future__ import annotations
import logging
import threading
from functools import wraps
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from .audit import AuditLog
from .executor import MockExecutor
from .models import ActionRequest, ApprovalStatus, AuditEventType, ModelSchema, ActionPreview
from .validation import validate_action, ValidationResult
from .authorization import _Grant, _activate, digest, schema_digest, file_digest, ExecutionBoundaryError
from .plans import simulation_scope, validate_optimization_plan

class ApprovalError(ExecutionBoundaryError): pass

def _single_execution(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not self._execution_lock.acquire(blocking=False):
            raise ApprovalError("Gateway execution already in progress")
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._execution_lock.release()
    return wrapped

class ActionGateway:
    """Stateful approval gate. Execution is impossible before approval."""
    def __init__(self, schema: ModelSchema, *, executor=None, audit=None, ttl_seconds: int = 3600, simulation_db=None, simulation_runner=None):
        self.schema = schema; self.executor = executor; self.audit = audit or AuditLog(); self.ttl_seconds = ttl_seconds
        self.simulation_db = simulation_db
        self.simulation_runner = simulation_runner
        self._actions: dict[str, tuple[ActionRequest, ApprovalStatus, datetime, ActionPreview | None]] = {}
        self._backups: dict[str, object] = {}
        self._bindings = {}
        self._previewed = set()
        self._execution_lock = threading.Lock()
    def submit(self, action: ActionRequest, *, actor="user", refresh: bool = True) -> ValidationResult:
        # Static validation only. Preview must never refresh a live Aspen tree.
        if action.audit_id in self._actions:
            raise ApprovalError("audit_id already submitted; create a new preview")
        if self.simulation_db is not None and self.simulation_db.get_action_audit(action.audit_id):
            raise ApprovalError("action already recorded; automatic replay is disabled")
        action = deepcopy(action)  # Approval owns an immutable payload snapshot.
        result = validate_action(action, self.schema, actor=actor)
        if result.valid:
            try:
                self._bindings[action.audit_id] = self._binding(action)
            except (ValueError, OSError, TypeError) as exc:
                result = ValidationResult(False, [str(exc)])
        status = ApprovalStatus.PENDING if result.valid else ApprovalStatus.REJECTED
        self._actions[action.audit_id] = (action, status, datetime.now(timezone.utc), deepcopy(result.preview))
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
            session_id=str(payload.get("session_id") or ""),
        )
        # Conversational planning/clarification must never touch Aspen COM.
        result = self.submit(action, actor="lead_agent", refresh=False)
        if not result.valid or result.preview is None:
            return {"valid": False, "errors": list(result.errors), "warnings": list(result.warnings)}
        return {"valid": True, "audit_id": action.audit_id, "preview": self.preview(action.audit_id).to_dict(),
                "payload": payload, "evidence": list(proposal.evidence_ids), "risks": list(proposal.risks),
                "target": proposal.target, "action_type": action_type}
    def preview(self, audit_id: str) -> ActionPreview:
        action, status, created, preview = self._get(audit_id); self._expire_if_needed(audit_id)
        _, status, _, preview = self._get(audit_id)
        if status in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}: raise ApprovalError(f"action is {status.value}")
        if preview is None: raise ApprovalError("no valid preview")
        self._previewed.add(audit_id)
        self.audit.record(audit_id, AuditEventType.PREVIEW, status=status.value, payload=preview.to_dict()); return deepcopy(preview)
    def approve(self, audit_id: str, *, actor="user"):
        action, status, created, preview = self._get(audit_id); self._expire_if_needed(audit_id); action, status, created, preview = self._get(audit_id)
        if status != ApprovalStatus.PENDING: raise ApprovalError(f"cannot approve action in {status.value} state")
        if audit_id not in self._previewed:
            raise ApprovalError("display a concrete preview before approval")
        self._check_binding(action)
        self._actions[audit_id] = (action, ApprovalStatus.APPROVED, created, preview); self.audit.record(audit_id, AuditEventType.CONFIRMATION, actor=actor, status="approved")
    def reject(self, audit_id: str, *, actor="user"):
        action, status, created, preview = self._get(audit_id)
        if status in {ApprovalStatus.EXECUTED, ApprovalStatus.FAILED}: raise ApprovalError(f"cannot reject action in {status.value} state")
        self._actions[audit_id] = (action, ApprovalStatus.REJECTED, created, preview); self.audit.record(audit_id, AuditEventType.CONFIRMATION, actor=actor, status="rejected")
    @_single_execution
    def execute(self, audit_id: str, *, actor="system"):
        action, status, created, preview = self._get(audit_id); self._expire_if_needed(audit_id); action, status, created, preview = self._get(audit_id)
        if status != ApprovalStatus.APPROVED: raise ApprovalError(f"execution requires approved action, got {status.value}")
        self._check_binding(action)
        result = validate_action(action, self.schema, actor=actor)
        if not result.valid:
            self._actions[audit_id] = (action, ApprovalStatus.FAILED, created, preview); self.audit.record(audit_id, AuditEventType.ERROR, status="failed", message="; ".join(result.errors)); raise ApprovalError("action invalid at execution: " + "; ".join(result.errors))
        claimed = False
        try:
            if self.simulation_db is not None:
                self.simulation_db.claim_execution(audit_id, {"action": action.to_dict(), "status": "started"})
                claimed = True
            # Consume the action before entering user/injected execution code.
            self._actions[audit_id] = (action, ApprovalStatus.RUNNING, created, preview)
            # Only complete plans receive a scoped execution grant.
            if action.action_type.value == "run_simulation" and isinstance(action.payload.get("simulation_plan"), dict):
                if self.simulation_runner is None:
                    raise RuntimeError("已创建回测计划，但当前运行环境没有配置 Aspen simulation_runner。")
                from src.agents.orchestration.parameter_impact import SimulationPlan
                sim_plan = SimulationPlan.from_dict(action.payload["simulation_plan"])
                sim_plan.approved = True
                from .authorization import file_digest
                sim_plan._approved_model_hash = file_digest(sim_plan.base_model)
                grant = self._grant(action, simulation_scope(sim_plan, self.schema),
                                    sim_plan.max_evaluations,
                                    sim_plan.adaptive_policy.get("max_seconds", sim_plan.timeout_seconds * sim_plan.max_evaluations),
                                    sim_plan.adaptive_policy.get("max_failures", sim_plan.max_evaluations))
                if not sim_plan.adaptive_policy.get("parameters"):
                    grant.allowed_points = [deepcopy(c["inputs"]) for c in sim_plan.parameter_changes]
                previous_callback = getattr(self.simulation_runner, "case_callback", None)
                if self.simulation_db is not None:
                    def persist_case(case):
                        self.simulation_db.save_cases([case.to_dict()])
                        self.simulation_db.finish_execution_evaluation(audit_id, grant.used, case)
                        if previous_callback is not None:
                            previous_callback(case)
                    self.simulation_runner.case_callback = persist_case
                try:
                    with _activate(grant):
                        cases = self.simulation_runner.run(sim_plan)
                finally:
                    self.simulation_runner.case_callback = previous_callback
                if self.simulation_db is not None and hasattr(self.simulation_db, "save_cases"):
                    self.simulation_db.save_cases([c.to_dict() for c in cases])
                for case in cases:
                    if case.errors or case.warnings:
                        logging.getLogger(__name__).warning(
                            "回测工况诊断 case_id=%s errors=%s warnings=%s",
                            case.case_id, case.errors, case.warnings)
                impact = action.payload.get("parameter_impact_plan") or {}
                try:
                    from src.agents.orchestration.parameter_impact import SensitivityAnalyzer
                    parameters = [str(x.get("aspen_path") or x.get("name")) for x in impact.get("parameters", []) if isinstance(x, dict)]
                    sensitivity = SensitivityAnalyzer().analyze(cases, metric=str(impact.get("metric") or "metric"), parameters=parameters,
                                                               parameter_specs=sim_plan.adaptive_policy.get("parameters"))
                    sensitivity["unit"] = (sim_plan.output_spec or {}).get("unit")
                except Exception as exc:
                    sensitivity = {"error": f"sensitivity analysis failed: {type(exc).__name__}: {exc}"}
                simulated = bool(getattr(self.simulation_runner, "simulated", False)) or any(c.result_source in {"fake_aspen", "mock"} for c in cases)
                output = {"audit_id": action.audit_id, "status": "simulated" if simulated else "execution_completed" if grant.started_runs else "not_executed",
                          "simulated": simulated, "real_execution": bool(grant.started_runs and not simulated),
                          "model_runs_started": grant.started_runs,
                          "evaluations_used": grant.used, "action_type": action.action_type.value,
                          "simulation_cases": [c.to_dict() for c in cases], "sensitivity": sensitivity,
                          "planned_case_count": len(sim_plan.parameter_changes),
                          "skipped_case_count": max(0, len(sim_plan.parameter_changes) - len(cases))}
                output.update({k: v for k, v in getattr(self.simulation_runner, "last_report", {}).items()
                               if k not in {"status", "simulated", "real_execution", "evaluations_used"}})
                from src.agents.orchestration.impact_evidence import execution_evidence
                output["execution_evidence"] = execution_evidence(output, sim_plan)
                if output.get("stop_reason") == "execution_unknown":
                    output["status"] = "outcome_unknown"
                if output.get("reapproval_plan"):
                    output["requires_reapproval"] = True
            elif action.action_type.value == "run_optimization":
                from .plans import execute_optimization
                plan = action.payload["optimization_plan"]
                scope = validate_optimization_plan(plan, self.schema)
                limits = plan["limits"]
                grant = self._grant(action, scope, limits["max_evaluations"], limits["max_seconds"], limits["max_failures"])
                with _activate(grant):
                    output = execute_optimization(plan)
                simulated = bool(output.get("simulated"))
                output.update(audit_id=audit_id, evaluations_used=grant.used,
                              real_execution=bool(grant.started_runs and not simulated), model_runs_started=grant.started_runs,
                              simulated=simulated)
                if not grant.started_runs and output.get("status") != "outcome_unknown":
                    output["status"] = "not_executed"
                evaluations = self.simulation_db.get_execution_evaluations(audit_id)
                output["evaluations"] = evaluations
                output["simulation_cases"] = [case for e in evaluations if e.get("case_id")
                    and (case := self.simulation_db.get_case(e["case_id"])) is not None]
                output["planned_case_count"] = plan["limits"]["max_evaluations"]
                output["actual_case_count"] = grant.used
                output["unknown_case_count"] = sum(e.get("status") in {"reserved_outcome_unknown", "outcome_unknown"} for e in evaluations)
                output["skipped_case_count"] = max(0, output["planned_case_count"] - grant.used)
                from types import SimpleNamespace
                from src.agents.orchestration.impact_evidence import execution_evidence
                output["execution_evidence"] = execution_evidence(output,
                    SimpleNamespace(plan_id=audit_id, output_spec={"kind": "optimization", "algorithm": plan["algorithm"]}))
            else:
                if self.executor is None:
                    raise RuntimeError("execution capability unavailable: no executor configured")
                if not isinstance(self.executor, MockExecutor):
                    raise RuntimeError("standalone write executors are disabled; submit an explicit simulation plan")
                output = self.executor.execute(action, preview=result.preview)
            if self.simulation_db is not None and hasattr(self.simulation_db, "save_action_audit"):
                self.simulation_db.save_action_audit(action.audit_id, {
                    "action": action.to_dict(), "preview": result.preview.to_dict() if result.preview else None,
                    "result": output,
                })
        except BaseException as exc:
            if claimed:
                self.simulation_db.save_action_audit(audit_id, {"action": action.to_dict(), "status": "failed" if isinstance(exc, Exception) else "outcome_unknown",
                    "error": str(exc), "evaluations_used": getattr(locals().get("grant"), "used", 0),
                    "resources": getattr(exc, "resources", None) or getattr(self.simulation_runner, "last_report", {}).get("resources")})
            self._actions[audit_id] = (action, ApprovalStatus.FAILED, created, preview); self.audit.record(audit_id, AuditEventType.ERROR, status="failed", message=str(exc)); raise
        final_status = (ApprovalStatus.FAILED if output.get("status") in {"outcome_unknown", "failed"} else
                        ApprovalStatus.SIMULATED if output.get("simulated") else
                        ApprovalStatus.EXECUTED if output.get("real_execution") else ApprovalStatus.FAILED)
        self._actions[audit_id] = (action, final_status, created, preview)
        self.audit.record(audit_id, AuditEventType.EXECUTION, actor=actor, status=output.get("status"), payload=output)
        return output
    def rollback(self, audit_id: str):
        raise ApprovalError("rollback requires a newly previewed and approved plan")
    def status(self, audit_id): self._expire_if_needed(audit_id); return self._get(audit_id)[1]
    def _get(self, audit_id):
        if audit_id not in self._actions: raise ApprovalError("unknown audit_id")
        return self._actions[audit_id]
    def _expire_if_needed(self, audit_id):
        action, status, created, preview = self._actions[audit_id]
        if status in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED, ApprovalStatus.RUNNING} and datetime.now(timezone.utc) - created > timedelta(seconds=self.ttl_seconds): self._actions[audit_id] = (action, ApprovalStatus.EXPIRED, created, preview)

    def _binding(self, action):
        files = {}
        if self.schema.config_path:
            files[self.schema.config_path] = file_digest(self.schema.config_path)
        sim = action.payload.get("simulation_plan")
        if sim:
            files[str(sim["base_model"])] = file_digest(sim["base_model"])
        opt = action.payload.get("optimization_plan")
        if opt:
            files[opt["config_path"]] = file_digest(opt["config_path"])
            files[opt["model_path"]] = file_digest(opt["model_path"])
            for path in opt.get("objective_module_hashes", {}):
                files[path] = file_digest(path)
        return digest({"action": action.to_dict(), "schema": schema_digest(self.schema), "files": files})

    def _check_binding(self, action):
        self._expire_if_needed(action.audit_id)
        if self.status(action.audit_id) not in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED, ApprovalStatus.RUNNING}:
            raise ApprovalError("authorization expired or revoked")
        if self._binding(action) != self._bindings.get(action.audit_id):
            raise ApprovalError("approved model/configuration/algorithm/plan changed; preview and approve again")

    def _grant(self, action, scope, count, seconds, failures):
        if self.simulation_db is None and not getattr(self.simulation_runner, "simulated", False):
            raise RuntimeError("durable SimulationDB execution ledger is required for real execution")
        grant = _Grant(action_id=action.audit_id, schema=self.schema, scope=scope,
                      max_evaluations=count, max_seconds=seconds, max_failures=failures,
                      check=lambda: self._check_binding(action), audit=self.audit, ledger=self.simulation_db)
        grant.session_id = action.session_id
        return grant
