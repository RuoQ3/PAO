from __future__ import annotations

from typing import Any
from pathlib import Path
import re
import json
import logging

from src.agents.orchestration.models import ActionProposal, Answer, ApprovalRequest, Classification, Evidence, ExecutionMode, RuntimePolicy, SessionState, SubagentRequest, TaskSpec
from src.agents.orchestration.closed_loop.service import ClosedLoopService
from src.agents.orchestration.runtime import OrchestrationRuntime, TaskPlanner
from src.agents.lead.intent import IntentClassifier
from src.agents.context import ContextLoader
from src.agents.context import ProcessCaseAdapter
from src.agents.context.resources import resolve_case_resource
from src.agents.subagents.registry import SubagentRegistry, default_registry
from src.agents.orchestration.parameter_impact import (
    build_parameter_impact_plan, make_simulation_plan, validate_parameter_impact_plan,
    select_impact_parameters, parameter_issues,
)
from src.agents.errors import AgentUnavailableError, unavailable, ensure_session_available, session_boundary, internal_error
from src.agents.reasoning import reason, conversation_context
from src.agents.answer_content import AnswerBlock
from src.agents.lead.continuation import (new_task, validate_task, clarification, interpret_turn, merge_turn, task_prompt, provider_json)


class LeadAgent:
    """Single user-facing decision maker; execution remains in Runtime/Gateway."""

    def __init__(self, *, registry: SubagentRegistry | None = None,
                 policy: RuntimePolicy | None = None, context_builder: Any = None,
                 action_gateway: Any = None, closed_loop_service: ClosedLoopService | None = None,
                 use_llm: bool | None = None, llm_config: Any = None,
                 intent_classifier: IntentClassifier | None = None, llm_call: Any = None):
        self.registry = registry or default_registry(llm_config=llm_config, llm_call=llm_call)
        self.llm_call = llm_call
        self.policy = policy or RuntimePolicy()
        self.context_builder = context_builder
        self.action_gateway = action_gateway
        self.closed_loop_service = closed_loop_service
        self.use_llm = True if use_llm is None else bool(use_llm)
        self.llm_config = llm_config
        self.intent_classifier = intent_classifier or IntentClassifier(
            llm_config=llm_config, enabled=self.use_llm, llm_call=llm_call,
        )
        self._pending_closed_loops: dict[str, dict[str, Any]] = {}
        self._case_adapters: dict[str, ProcessCaseAdapter] = {}

    @session_boundary
    def answer(self, question: str, session: SessionState | None = None) -> Answer:
        session = session or SessionState(mode=self.policy.mode)
        ensure_session_available(session)
        return self._answer(question, session)

    def _answer(self, question: str, session: SessionState) -> Answer:
        session = session or SessionState(mode=self.policy.mode)
        self._revoke_restored_approvals(session)
        task = session.pending_task
        if task is not None and task.active:
            # Refresh/migrate the resource repair contract before interpreting
            # a reply, including tasks restored from older checkpoints.
            self._check_case_resource(task, session)
            decision = interpret_turn(self, task, question)
            task.audit.append({"event": "turn_routed", "kind": decision["kind"]})
            if decision["kind"] in {"cancel", "new_task"}:
                self.invalidate_approval(session)
                task.phase = "cancelled" if decision["kind"] == "cancel" else "superseded"
                session.audit_events.append({"event": "task_ended", "task": task.to_dict()})
                if decision["kind"] == "cancel":
                    return Answer(text="当前任务已取消。")
                session.runtime_state.pop("conversation_focus", None)
            else:
                self.invalidate_approval(session)
                context = self._task_context(task.verified_intent, session).context
                previous_evidence = task.evidence_digest
                had_blockers = bool(task.planning_blockers)
                validate_task(task, context)
                merged = merge_turn(task, decision, question)
                resource_answer = self._check_case_resource(task, session)
                if resource_answer is not None:
                    return resource_answer
                context = self._task_context(task.verified_intent, session).context
                valid = validate_task(task, context)
                if had_blockers and valid and (previous_evidence != task.evidence_digest or decision["kind"] == "answer"):
                    return self._continue_task(task, session)
                if not merged or not valid:
                    task.phase = "clarifying"
                    if had_blockers and not merged and valid:
                        return Answer(text="需要补充 case 证据：" + "；".join(b["message"] for b in task.planning_blockers))
                    return clarification(task, prefix="请明确要补充或修改的字段。\n" if not merged else "")
                return self._continue_task(task, session)
        self.invalidate_approval(session)
        classification = self.intent_classifier.classify(question, session=session)
        if not isinstance(classification, Classification) or classification.classification_source not in {"llm", "llm_repaired"}:
            raise internal_error("classification_protocol_invalid", stage="classification")
        if classification.intent.confidence < .55 and not classification.intent.missing_information:
            raise unavailable("llm_low_confidence", stage="classification")
        session.pending_approval = None
        intent = classification.intent
        session.intent = intent
        session.classification_source = classification.classification_source
        session.classification_retry_count = classification.retry_count
        session.audit_events.append({
            "event": "intent_classified", "source": classification.classification_source,
            "retry_count": classification.retry_count, "intent": intent.to_dict(),
        })
        loaded = self._task_context(intent, session)
        task = new_task(self, question, intent, loaded.context)
        session.pending_task = task
        resource_answer = self._check_case_resource(task, session)
        if resource_answer is not None:
            return resource_answer
        if not validate_task(task, loaded.context):
            return clarification(task)
        return self._continue_task(task, session)

    def _check_case_resource(self, task, session):
        if task.verified_intent.intent != "parameter_impact":
            return None
        builder = self._builder_for_session(session)
        refs = dict(session.context_refs)
        proposed = task.parameters.get("model_file")
        if proposed:
            refs["model_file"] = proposed
        resource = resolve_case_resource(builder, session={"context_refs": refs})
        if resource.error:
            # Resource repair is an explicitly declared, validated protocol
            # field. Other planning blockers never become user input slots.
            task.field_specs["model_file"] = {"type": "resource_path", "multiple": False,
                                               "resource": "model_file", "extensions": [".bkp", ".apw"]}
            task.phase = "clarifying"
            task.missing_fields = list(dict.fromkeys([*task.missing_fields, "model_file"]))
            task.waiting_fields = ["model_file"]
            task.planning_blockers = [{"field": "model_file", "message": resource.error}]
            task.audit.append({"event": "case_resource_blocked", "resource": resource.to_dict()})
            return Answer(text=f"case 资源不可用：{resource.error}。请补充可读的 BKP/APW 路径（model_file），然后重新构建计划。未执行 Aspen。",
                          facts=[{"kind": "case_resource", **resource.to_dict()}])
        if proposed and refs != session.context_refs:
            session.context_refs["model_file"] = str(resource.path)
            task.parameters["model_file"] = str(resource.path)
            task.parameter_sources["model_file"].update(valid=True, evidence=[resource.to_dict()])
            task.audit.append({"event": "case_resource_updated", "resource": resource.to_dict()})
        task.planning_blockers = [b for b in task.planning_blockers if b["field"] != "model_file"]
        return None

    def _continue_task(self, task, session):
        # Re-read case evidence before every continuation. Persisted candidates
        # are never treated as fresh execution authorization.
        intent = type(task.verified_intent).from_dict(task.verified_intent.to_dict())
        loaded = self._task_context(intent, session)
        if not validate_task(task, loaded.context):
            return clarification(task)
        intent.missing_information = []
        question = task_prompt(task) if task.parameters else task.original_goal
        if intent.intent == "parameter_impact":
            intent.analysis = dict(task.parameters)
        session.intent = intent
        classification = Classification(intent=intent, classification_source=session.classification_source or "llm")
        answer = self._execute_intent(question, session, classification, loaded)
        if task.phase != "clarifying":
            task.phase = "awaiting_approval" if session.pending_approval is not None else "completed"
        task.audit.append({"event": "task_phase", "phase": task.phase})
        return answer

    def _execute_intent(self, question, session, classification, loaded):
        intent = classification.intent
        if intent.intent == "parameter_impact":
            slots = dict(intent.analysis)
            context = loaded.context
            # The deterministic perturbation builder is a numerical tool. A
            # validated advisor response must precede plan/approval creation.
            from src.agents.reasoning import snapshot
            reasoning, reasoning_evidence = reason(question, context={**snapshot(context), "conversation": conversation_context(session)}, role="parameter_impact",
                                  llm_config=self.llm_config or getattr(self.intent_classifier, "llm_config", None),
                                  llm_call=self.llm_call,
                                  role_contract="Assess the requested local sensitivity experiment using the supplied paths, bounds and target evidence. Do not invent recommendations or simulation results.")
            impact = build_parameter_impact_plan(
                context, equipment_id=str(slots["equipment"]), target_stream=slots.get("target_stream"),
                target_component=slots.get("target_component"), metric=str(slots["metric"]),
                candidate_parameters=slots.get("candidate_parameters") or None,
                max_cases=max(1, min(int(slots.get("max_cases") or 20), self.policy.total_budget)),
            )
            eligible = [p for p in impact.parameters if not parameter_issues(p)]
            decisions = []
            if eligible:
                selection = provider_json(self, "select_impact_parameters", {
                    "goal": question, "metric": slots["metric"], "target_stream": slots["target_stream"],
                    "target_component": slots["target_component"], "parameters": eligible,
                    "max_cases": impact.max_cases,
                }, 'Rank the authorized equipment Input parameters for this metric using the supplied case evidence. '
                   'Return exactly {"decisions":[{"path":string,"include":boolean,"reason":nonempty string}]}. '
                   'Include every supplied path exactly once, highest metric relevance first. Explain the evidence '
                   'and mechanistic hypothesis for each inclusion/exclusion; distinguish hypotheses from results. '
                   'Consider baseline, units, bounds, continuous/integer type and finite experiment budget. '
                   'Never invent paths, authorization or simulation conclusions.')
                decisions = selection.get("decisions")
                if (set(selection) != {"decisions"} or not isinstance(decisions, list)
                        or any(not isinstance(d, dict) or set(d) != {"path", "include", "reason"}
                               or not isinstance(d["path"], str) or type(d["include"]) is not bool
                               or not isinstance(d["reason"], str) or not d["reason"].strip() for d in decisions)
                        or len(decisions) != len(eligible)
                        or {d["path"] for d in decisions} != {p["aspen_path"] for p in eligible}):
                    raise internal_error("parameter_selection_protocol_invalid", stage="parameter_selection")
            decisions.extend({"path": p["aspen_path"], "include": False, "reason": "安全前提未满足"}
                             for p in impact.parameters if parameter_issues(p))
            if not impact.parameters:
                impact.blockers.append({"field": "parameter_authorization", "message": "设备没有已配置或明确授权的 Input 参数证据"})
            select_impact_parameters(impact, decisions=decisions, restricted=bool(slots.get("candidate_parameters")))
            errors = validate_parameter_impact_plan(impact, budget=self.policy.total_budget)
            if errors and not impact.blockers:
                impact.diagnostics.extend(errors)
                impact.blockers.extend({"field": "plan_validation", "message": e} for e in errors)
            if not impact.expected_outputs or any(not str(p).startswith("\\Data\\") or "\\Output\\" not in str(p) for p in impact.expected_outputs):
                impact.blockers.append({"field": "output_path", "message": "缺少可用 Aspen Output 路径"})
            resource = resolve_case_resource(self._builder_for_session(session), session=session)
            base_model = str(resource.path or "")
            if resource.error:
                return self._check_case_resource(session.pending_task, session)
            simulation = None
            try:
                if not impact.blockers:
                    simulation = make_simulation_plan(impact, base_model=base_model,
                                                  evidence_ids=[x.id for x in getattr(context, "sources", [])])
            except ValueError as exc:
                impact.blockers.append({"field": "plan_validation", "message": str(exc)})
            details = {"parameter_impact_plan": impact.to_dict(), "case_resource": resource.to_dict()}
            logging.getLogger(__name__).info("Parameter impact planning audit: %s",
                json.dumps(details, ensure_ascii=False, default=str), extra={"pao_file_only": True})
            if simulation is None or not simulation.parameter_changes or not simulation.executable:
                task = session.pending_task
                task.phase = "clarifying"
                task.planning_blockers = impact.blockers
                task.audit.append({"event": "plan_blocked", "blockers": impact.blockers,
                                   "parameter_impact_plan": impact.to_dict(), "case_resource": resource.to_dict()})
                return Answer(
                    text="暂时无法自动设计参数影响回测：" + "；".join(b["message"] for b in impact.blockers),
                    facts=self._classification_facts(classification),
                    uncertainties=["未执行 Aspen；请补充上述 case 证据后继续。"],
                    recommendations=[{"next_step": "更新 case 中缺失的参数或输出证据后回复继续。"}],
                    details={"parameter_impact_plan": impact.to_dict(), "case_resource": resource.to_dict()},
                )
            session.pending_task.planning_blockers = []
            session.pending_task.audit.append({"event": "parameters_selected", "selection": impact.selection,
                "case_count": len(simulation.parameter_changes), "max_cases": impact.max_cases})
            impact.diagnostics.append("参数相关性属于待验证假设；单因素局部扰动不能识别交互作用或全局因果关系。")
            if self.action_gateway is None:
                raise unavailable("action_gateway_unavailable", stage="plan_preview")
            payload = {"question": question, "intent": intent.to_dict(),
                       "case_resource": resource.to_dict(),
                       "parameter_impact_plan": impact.to_dict(),
                       "simulation_plan": simulation.to_dict() if simulation else None}
            proposal = ActionProposal(action_type="run_simulation", target="model", reason=question,
                                      payload=payload, evidence_ids=[f"intent:{session.session_id}"],
                                      risks=["Only a temporary Aspen model copy may be used.", "Results are limited to the local perturbation range."],
                                      requires_confirmation=True)
            approval = ApprovalRequest(proposal=proposal,
                                       message="参数影响回测将消耗仿真预算；请确认后执行。")
            if self.action_gateway is not None:
                try:
                    proposal.payload["session_id"] = session.session_id
                    approval.preview = self.action_gateway.preview_proposal(proposal)
                except AgentUnavailableError:
                    raise
                except Exception as exc:
                    raise unavailable("action_gateway_unavailable", stage="plan_preview", cause=exc) from None
            session.pending_approval = approval
            session.context_summary = loaded.summary
            session.context_sources = loaded.sources
            names = [str(p.get("name") or p["aspen_path"]) for p in impact.parameters]
            selected = "、".join(names[:6]) + ("等" if len(names) > 6 else "")
            excluded = sum(not s["selected"] for s in impact.selection)
            cases = len(simulation.parameter_changes)
            return Answer(text=(reasoning["text"] + "\n" + f"已生成 {slots['equipment']} 的参数影响计划：物流 {slots['target_stream']}，"
                                f"组分 {slots['target_component']}，指标 {slots['metric']}。\n"
                                f"选择 {len(names)} 个参数（{selected}），排除 {excluded} 项；"
                                f"共 {cases} 个工况，包含基准和局部扰动。\n"
                                "结论仅适用于本次扰动范围；完整参数证据和选择理由保留在结构化计划及文件审计中。\n"
                                "尚未执行 Aspen。请审批后执行。"),
                          facts=self._classification_facts(classification), approval_request=approval,
                          uncertainties=reasoning["uncertainties"], details={**payload, "reasoning_evidence": reasoning_evidence},
                          blocks=reasoning.get("blocks", []))
        if intent.action_type in {"modify_file", "set_parameter", "simulate_once", "optimize", "modify_structure"}:
            from src.agents.reasoning import snapshot
            analysis, analysis_evidence = reason(question, context={**snapshot(loaded.context), "conversation": conversation_context(session)},
                                 role="action_planning", llm_config=self.llm_config, llm_call=self.llm_call)
            payload = {"question": question, "intent": intent.to_dict()}
            if intent.action_type == "simulate_once":
                from src.agents.actions.plans import single_simulation_plan
                try:
                    resource = self.context_builder.resolve_model_resource(session=session)
                    payload["simulation_plan"] = single_simulation_plan(loaded.context, resource.path).to_dict()
                except (ValueError, AttributeError) as exc:
                    return Answer(text=f"无法生成单次仿真计划：{exc}。未执行 Aspen。")
            proposal = ActionProposal(action_type=self._proposal_action_type(intent.action_type), target="model",
                                      reason=question, payload=payload,
                                      evidence_ids=[f"intent:{session.session_id}"],
                                      risks=["ActionGateway will revalidate the current model before execution."],
                                      requires_confirmation=True)
            approval = ApprovalRequest(proposal=proposal, message="Human confirmation is required before execution.")
            if self.action_gateway is not None:
                try:
                    proposal.payload["session_id"] = session.session_id
                    approval.preview = self.action_gateway.preview_proposal(proposal)
                except AgentUnavailableError:
                    raise
                except Exception as exc:
                    raise unavailable("action_gateway_unavailable", stage="plan_preview", cause=exc) from None
            session.pending_approval = approval
            return Answer(text=analysis["text"], blocks=analysis.get("blocks", []), uncertainties=analysis["uncertainties"],
                          details={"reasoning_evidence": analysis_evidence},
                          facts=self._classification_facts(classification), recommendations=[proposal.to_dict()], approval_request=approval)
        if intent.intent in {"chat", "general_question"} and not intent.subagents and intent.project_scope in {"none", "external"}:
            answer = self._answer_general(question, session)
            answer.facts.extend(self._classification_facts(classification))
            return answer
        # Carry the last explicitly named equipment/stream into short
        # follow-ups such as “所以呢？”.  This keeps the conversation scoped
        # without making the context loader or subagents guess from prose.
        task_question = question
        if intent.intent in {"analysis", "project_query"}:
            ids = re.findall(r"\b[A-Za-z]{1,4}\d{3,4}(?:-[A-Za-z0-9]+)?\b|(?<!\d)\d{4}(?!\d)", question)
            if not ids:
                history = getattr(session, "history", []) or []
                prior = " ".join(str(x.get("content", "")) for x in history[:-1]
                                   if isinstance(x, dict) and x.get("role") == "user")
                prior_ids = re.findall(r"\b[A-Za-z]{1,4}\d{3,4}(?:-[A-Za-z0-9]+)?\b|(?<!\d)\d{4}(?!\d)", prior)
                if prior_ids:
                    task_question = f"{question} [conversation object: {prior_ids[-1]}]"
        task_type = intent.task_type
        requested = intent.subagents[0] if intent.subagents else None
        if not intent.subagents or any(r.name not in self.registry.names() for r in intent.subagents):
            raise internal_error("subagent_route_missing", stage="subagent_routing")
        agent = requested.name
        # A project overview needs the flowsheet snapshot even when an LLM
        # happens to classify it as file/YAML research only.  Without this
        # augmentation ContextLoader projects away equipment and streams,
        # leaving the compatibility adapter with no facts to summarize.
        if agent in {"project_researcher", "process_advisor"} and "process_context" not in intent.context_required:
            intent = type(intent)(
                intent=intent.intent, task_type=task_type, project_scope=intent.project_scope,
                context_required=["process_context", *list(intent.context_required)],
                subagents=list(intent.subagents), action_type=intent.action_type,
                requires_confirmation=intent.requires_confirmation, confidence=intent.confidence,
                missing_information=list(intent.missing_information), reason=intent.reason, analysis=dict(intent.analysis),
            )
        if requested is None or agent != requested.name:
            intent = type(intent)(
                intent=intent.intent, task_type=task_type, project_scope=intent.project_scope,
                context_required=list(intent.context_required),
                subagents=[SubagentRequest(name=agent, task=question)],
                action_type=intent.action_type, requires_confirmation=intent.requires_confirmation,
                confidence=intent.confidence, missing_information=list(intent.missing_information), reason=intent.reason, analysis=dict(intent.analysis),
            )
        task_specs = TaskPlanner(policy=self.policy).plan(intent, task_question)
        for spec in task_specs:
            spec.metadata["conversation"] = conversation_context(session)
            spec.metadata["session_id"] = session.session_id
        runtime = OrchestrationRuntime(
            self.registry.as_mapping(), policy=self.policy, audit=session.audit_events,
            role_definitions=self.registry.definitions(),
            checkpoint_path=session.checkpoint_ref, resume=False,
            session_id=session.session_id, resident=session.runtime_state,
        )
        loaded = self._load_context(intent.context_required, session)
        context = loaded.context
        session.context_summary = loaded.summary
        session.context_sources = loaded.sources
        results = runtime.rank_results(runtime.run(task_specs, context=context))
        session.runtime_state = runtime.resident_state()
        issues = runtime.validate_results(results)
        session.task_results.extend(results)
        if not results or any(r.status.value != "completed" for r in results):
            raise unavailable("subagent_unavailable", stage="subagent_result")
        if issues:
            raise unavailable("agent_output_invalid", stage="evidence_validation")
        result = results[0]
        if len(results) > 1:
            session.audit_events.append({
                "event": "results_synthesized", "task_ids": [item.task_id for item in results],
                "issues": issues,
            })
        text = result.summary if not issues else "无法形成有充分证据的结论：" + "；".join(issues)
        if result.status.value != "completed":
            text = "任务未完成：" + ("；".join(result.errors) or result.status.value)
        blocks = result.blocks
        summary = result.summary
        reasoning_evidence = result.evidence[0].value if result.evidence else {}
        approval = None
        proposals = [p for r in results for p in r.recommended_actions if isinstance(p, dict) and p.get("action_type")]
        if proposals:
            if len(proposals) != 1 or self.action_gateway is None:
                raise unavailable("action_gateway_unavailable", stage="tool_plan_preview")
            proposal = ActionProposal(**proposals[0])
            proposal.payload["session_id"] = session.session_id
            preview = self.action_gateway.preview_proposal(proposal)
            if not preview.get("valid"):
                raise unavailable("tool_plan_invalid", stage="tool_plan_preview")
            approval = ApprovalRequest(proposal, preview=preview, message="请审核具体计划；尚未执行 Aspen。")
            session.pending_approval = approval
        if len(results) > 1:
            synthesis, reasoning_evidence = reason(question, context={"results": [r.to_dict() for r in results],
                                  "conversation": conversation_context(session)}, role="lead_synthesis",
                                  llm_config=self.llm_config or getattr(self.intent_classifier, "llm_config", None),
                                  llm_call=self.llm_call)
            summary, blocks = synthesis["text"], synthesis.get("blocks", [])
        return Answer(text=summary, blocks=blocks, approval_request=approval,
                      details={"reasoning_evidence": reasoning_evidence},
                      facts=[x for r in results for x in r.facts],
                      evidence=[x for r in results for x in r.evidence],
                      assumptions=[x for r in results for x in r.assumptions],
                      uncertainties=[x for r in results for x in r.uncertainties],
                      recommendations=[x for r in results for x in r.recommended_actions], task_results=results)

    def invalidate_approval(self, session):
        self._revoke_restored_approvals(session)
        approval = session.pending_approval
        session.pending_approval = None
        session.runtime_state.pop("active_plan", None)
        self._pending_closed_loops.pop(session.session_id, None)
        if approval is not None:
            approval.status = "revoked"
            audit_id = approval.preview.get("audit_id")
            if audit_id and self.action_gateway is not None:
                self._revoke_gateway_id(audit_id)
            session.audit_events.append({"event": "approval_revoked", "audit_id": audit_id})

    def _revoke_gateway_id(self, audit_id):
        from src.agents.actions.gateway import ApprovalError
        try:
            self.action_gateway.reject(audit_id, actor="lead_agent")
        except ApprovalError:
            # Absent, expired, executed or failed capabilities cannot be reused.
            pass

    def _revoke_restored_approvals(self, session):
        if self.action_gateway is not None:
            for audit_id in session.runtime_state.pop("revoked_approval_ids", []):
                self._revoke_gateway_id(audit_id)

    @staticmethod
    def _classification_facts(classification: Classification) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = [{
            "kind": "intent_classification",
            "classification_source": classification.classification_source,
            "retry_count": classification.retry_count,
            "intent": classification.intent.intent,
            "project_scope": classification.intent.project_scope,
            "confidence": classification.intent.confidence,
        }]
        return facts

    @staticmethod
    def _proposal_action_type(action_type: str) -> str:
        # IntentSchema uses the public conversational vocabulary; the gateway
        # retains PAO's existing deterministic action vocabulary.
        return {
            "simulate_once": "run_simulation",
            "optimize": "run_optimization",
        }.get(action_type, action_type)

    @staticmethod
    def _context_summary(context: Any) -> dict[str, Any]:
        return {
            "model_id": getattr(context, "model_id", None),
            "equipment": len(getattr(context, "equipment", []) or []),
            "streams": len(getattr(context, "streams", []) or []),
            "variables": len(getattr(context, "variables", {}) or {}),
            "history": len(getattr(context, "history_evidence", []) or []),
        }

    def _answer_general(self, question: str, session=None) -> Answer:
        data, evidence = reason(question, context={"conversation": conversation_context(session)},
                         llm_config=self.llm_config, llm_call=self.llm_call)
        return Answer(text=data["text"], blocks=data.get("blocks", []), uncertainties=data["uncertainties"],
                      details={"reasoning_evidence": evidence} if any(b.get("evidence_refs") for b in data.get("blocks", [])) else {})

    @session_boundary
    def propose_closed_loop(self, *, question: str, config: Any, driver: Any = None,
                            session: SessionState | None = None, **kwargs: Any) -> Answer:
        """Create an approval request for an internal optimization loop.

        No driver is opened and no workflow is started here.  The returned
        proposal is the only hand-off from the public Agent boundary to the
        closed-loop service.
        """
        session = session or SessionState(mode=self.policy.mode)
        ensure_session_available(session)
        self.invalidate_approval(session)
        if session.pending_task is not None and session.pending_task.active:
            session.pending_task.phase = "superseded"
            session.audit_events.append({"event": "task_ended", "task": session.pending_task.to_dict()})
        session.pending_task = None
        classification = self.intent_classifier.classify(question, session=session)
        if (not isinstance(classification, Classification)
                or classification.classification_source not in {"llm", "llm_repaired"}
                or classification.intent.action_type != "optimize"
                or not classification.intent.requires_confirmation
                or classification.intent.confidence < .55):
            raise unavailable("agent_output_invalid", stage="closed_loop_intent")
        session.intent = classification.intent
        session.classification_source = classification.classification_source
        session.classification_retry_count = classification.retry_count
        session.audit_events.append({
            "event": "intent_classified", "source": classification.classification_source,
            "retry_count": classification.retry_count, "intent": classification.intent.to_dict(),
        })
        data, evidence = reason(question, context={"configuration": getattr(config, "agent_context", {}),
                         "conversation": conversation_context(session)}, role="closed_loop_planning",
                         llm_config=self.llm_config, llm_call=self.llm_call)
        proposal = ActionProposal(
            action_type="run_optimization", target="model", reason=question,
            payload={"service": "closed_loop", "config_type": type(config).__name__},
            risks=["Aspen execution and optimization will consume the configured budget"],
            requires_confirmation=True,
        )
        approval = ApprovalRequest(proposal=proposal,
                                   message="Human confirmation is required before the closed loop starts.")
        if self.action_gateway is not None:
            try:
                approval.preview = self.action_gateway.preview_proposal(proposal)
            except AgentUnavailableError:
                raise
            except Exception as exc:
                raise unavailable("action_gateway_unavailable", stage="plan_preview", cause=exc) from None
        session.pending_approval = approval
        self._pending_closed_loops[session.session_id] = {
            "driver": driver, "config": config, "kwargs": kwargs,
            "approval": approval,
        }
        return Answer(text=data["text"], blocks=data.get("blocks", []), uncertainties=data["uncertainties"],
                      details={"reasoning_evidence": evidence},
                      facts=self._classification_facts(classification),
                      recommendations=[proposal.to_dict()], approval_request=approval)

    @session_boundary
    def confirm_closed_loop(self, session: SessionState, *, approved: bool = True) -> Any:
        """Confirm/reject a pending loop and execute only after approval."""
        ensure_session_available(session)
        pending = self._pending_closed_loops.get(session.session_id)
        if pending is None or session.pending_approval is None:
            raise ValueError("no pending closed-loop approval")
        approval = session.pending_approval
        if not approved:
            audit_id = approval.preview.get("audit_id")
            if audit_id and self.action_gateway is not None:
                self._revoke_gateway_id(audit_id)
            approval.status = "rejected"
            session.pending_approval = None
            self._pending_closed_loops.pop(session.session_id, None)
            return Answer(text="Closed-loop request rejected.", approval_request=approval)
        if self.action_gateway is None:
            raise RuntimeError("closed-loop execution requires an ActionGateway")
        audit_id = approval.preview.get("audit_id") if isinstance(approval.preview, dict) else None
        if not audit_id:
            raise RuntimeError("closed-loop proposal has no validated ActionGateway audit")
        self.action_gateway.approve(audit_id, actor="user")
        gateway_result = self.action_gateway.execute(audit_id, actor="lead_agent")
        # A closed-loop executor may perform the service call inside
        # ActionGateway.execute(). This keeps Aspen/optimization actions behind
        # the same approval boundary while allowing fake executors in tests.
        if isinstance(gateway_result, dict) and "closed_loop_result" in gateway_result:
            result = gateway_result["closed_loop_result"]
        else:
            raise PermissionError("Gateway did not execute the closed loop; legacy post-approval dispatch is disabled")
        approval.status = result.get("status", "failed") if isinstance(result, dict) else "failed"
        session.pending_approval = None
        self._pending_closed_loops.pop(session.session_id, None)
        return self._analyze_execution(result, approval, session)

    @session_boundary
    def confirm_action(self, session: SessionState, *, approved: bool = True) -> Any:
        """Confirm and execute a pending proposal through ActionGateway."""
        ensure_session_available(session)
        task = session.pending_task
        if task is not None and task.phase != "awaiting_approval":
            self.invalidate_approval(session)
            raise internal_error("task_not_awaiting_approval", stage="confirm_action")
        approval = session.pending_approval
        if approval is None or self.action_gateway is None:
            raise ValueError("no pending approval or ActionGateway")
        audit_id = approval.preview.get("audit_id") if isinstance(approval.preview, dict) else None
        if not audit_id:
            raise ValueError("pending proposal has no validated audit")
        if not approved:
            self.action_gateway.reject(audit_id, actor="user")
            approval.status = "rejected"
            session.pending_approval = None
            if task is not None:
                task.phase = "cancelled"
            return Answer(text="Action request rejected.", approval_request=approval)
        if task is not None:
            from copy import deepcopy
            checked = deepcopy(task)
            context = self._task_context(task.verified_intent, session).context
            if not validate_task(checked, context) or checked.evidence_digest != task.evidence_digest:
                self.invalidate_approval(session)
                task.phase = "validating"
                return self._continue_task(task, session)
        try:
            self.action_gateway.approve(audit_id, actor="user")
            result = self.action_gateway.execute(audit_id, actor="lead_agent")
        except AgentUnavailableError:
            raise
        except Exception as exc:
            self.invalidate_approval(session)
            approval.status = "failed"
            if task is not None:
                task.phase = "failed"
            return Answer(text=f"动作未完成：{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else '执行器未提供原因'}",
                          uncertainties=["执行失败；请诊断原因后重新创建计划。"],
                          details={"execution_error": str(exc)},
                          approval_request=approval)
        # Execution is final before interpretation begins. A failed LLM call
        # must neither replay Aspen nor erase already persisted evidence.
        approval.status = result.get("status", "failed") if isinstance(result, dict) else "failed"
        session.pending_approval = None
        if task is not None:
            task.phase = "completed"
        return self._analyze_execution(result, approval, session)

    def _analyze_execution(self, result, approval, session):
        """One semantic boundary for all completed actions, never a renderer."""
        audit_id = approval.preview.get("audit_id")
        from src.agents.reasoning import redact, snapshot
        result_data = snapshot(result) if not isinstance(result, dict) else result
        execution = redact({"result": result_data, "proposal": approval.proposal.to_dict()}, self.llm_config)
        session.audit_events.append({"event": "action_executed", **execution})
        logging.getLogger(__name__).info("Executed action audit: %s", json.dumps(execution, ensure_ascii=False, default=str),
                                        extra={"pao_file_only": True})
        try:
            data, evidence = reason(approval.proposal.reason or "解释本次动作的实际结果", context={
                **execution, "conversation": conversation_context(session)}, role="action_result_analysis",
                llm_config=self.llm_config or getattr(self.intent_classifier, "llm_config", None),
                llm_call=self.llm_call,
                role_contract="Explain executed evidence for the user's goal, including failures and what remains unknown. Never infer engineering significance from normalized ranking alone.")
        except AgentUnavailableError as exc:
            self._save_analysis(audit_id, result_data, {"status": "unavailable", "reason": exc.code if hasattr(exc, "code") else str(exc),
                "version": 1, "evidence_digest": (result_data.get("execution_evidence") or {}).get("evidence_digest")})
            exc.user_message = ("Agent 不可用：执行记录已保存，但结果分析失败；底层停止和清理状态请查看执行证据。"
                                f"审计编号 {audit_id}。不会自动重复执行。")
            exc.args = (exc.user_message,)
            raise
        answer = Answer(text=data["text"], blocks=data.get("blocks", []),
                        uncertainties=data["uncertainties"], details={"reasoning_evidence": evidence,
                            "analysis_version": 1, "confidence_semantics": "uncalibrated model self-report"})
        version = self._save_analysis(audit_id, result_data, {"status": "available", "version": 1,
            "evidence_digest": (result_data.get("execution_evidence") or {}).get("evidence_digest"),
            "answer": answer.to_dict()})
        answer.details["analysis_version"] = version
        return {**result_data, "answer": answer.to_dict()}

    def _save_analysis(self, audit_id, result, analysis):
        db = getattr(self.action_gateway, "simulation_db", None)
        if db is not None:
            record = db.get_action_audit(audit_id) or {}
            versions = list(record.get("analyses", []))
            analysis["version"] = len(versions) + 1
            if analysis.get("answer"):
                analysis["answer"]["details"]["analysis_version"] = analysis["version"]
            versions.append(analysis)
            db.save_action_audit(audit_id, {**record, "result": result, "analyses": versions})
            return analysis["version"]
        return 1

    def reanalyze_execution(self, audit_id, session):
        """Explicit read-only reanalysis; never grants or invokes execution."""
        from src.agents.actions.authorization import digest
        db = getattr(self.action_gateway, "simulation_db", None)
        record = db.get_action_audit(audit_id) if db else None
        if not record or record.get("action", {}).get("session_id") != session.session_id:
            raise ValueError("execution audit unavailable for this exact session")
        result = record["result"]
        packet = dict(result["execution_evidence"])
        expected = packet.pop("evidence_digest")
        if digest(packet) != expected:
            raise ValueError("execution evidence digest mismatch")
        proposal = ActionProposal("run_simulation", "model", reason="重新分析同一份已保存执行证据")
        approval = ApprovalRequest(proposal, preview={"audit_id": audit_id})
        return self._analyze_execution(result, approval, session)

    def _context(self, session: SessionState) -> Any:
        builder = self._builder_for_session(session)
        if builder is None:
            return None
        try:
            return builder.build(session=session)
        except TypeError:
            return builder.build(session=session, model_id=None)

    def _builder_for_session(self, session: SessionState) -> Any:
        """Resolve one case adapter for a session and reuse it across turns."""
        if self.context_builder is not None:
            return self.context_builder
        refs = getattr(session, "context_refs", {}) or {}
        config_ref = refs.get("case_config") or refs.get("config") or refs.get("case_config_path")
        if not config_ref:
            return None
        key = str(getattr(session, "session_id", "default"))
        cached = self._case_adapters.get(key)
        if cached is None or str(cached.config.config_path) != str(Path(config_ref).resolve()):
            self._case_adapters[key] = ProcessCaseAdapter.from_config(config_ref)
        return self._case_adapters[key]

    def _load_context(self, required: list[str], session: SessionState) -> Any:
        builder = self._builder_for_session(session)
        loader = builder if isinstance(builder, ContextLoader) else ContextLoader(builder)
        files = session.context_refs.get("files") if isinstance(session.context_refs, dict) else None
        try:
            return loader.load(required, session=session, files=[files] if isinstance(files, str) else files)
        except AgentUnavailableError:
            raise
        except Exception as exc:
            raise unavailable("context_component_unavailable", stage="context_load", cause=exc) from None

    def _task_context(self, intent, session):
        required = list(intent.context_required)
        if intent.intent == "parameter_impact":
            required = list(dict.fromkeys([*required, "process_context", "yaml", "node_db"]))
        return self._load_context(required, session)

    @staticmethod
    def _is_action_request(question: str) -> bool:
        q = question.casefold()
        return any(token in q for token in (
            "set ", "modify", "write", "run simulation", "run optimization",
            "修改", "写入", "运行仿真", "执行优化", "改变边界",
        ))
