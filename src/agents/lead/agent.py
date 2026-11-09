from __future__ import annotations

from typing import Any
import re
from pathlib import Path

from src.agents.orchestration.models import ActionProposal, Answer, ApprovalRequest, Classification, Evidence, ExecutionMode, RuntimePolicy, SessionState, SubagentRequest, TaskSpec
from src.agents.orchestration.closed_loop.service import ClosedLoopService
from src.agents.orchestration.runtime import OrchestrationRuntime, TaskPlanner
from src.agents.lead.intent import IntentClassifier
from src.agents.context import ContextLoader
from src.agents.context import ProcessCaseAdapter
from src.agents.subagents.registry import SubagentRegistry, default_registry
from src.agents.orchestration.parameter_impact import (
    build_parameter_impact_plan, make_simulation_plan, validate_parameter_impact_plan,
)


_PAO_TERMS = (
    "aspen", "仿真", "优化", "pareto", "边界", "动力学", "物性", "流程", "工艺",
    "项目", "设备", "装置", "物流", "project", "equipment", "unit", "process", "stream",
)


class LeadAgent:
    """Single user-facing decision maker; execution remains in Runtime/Gateway."""

    def __init__(self, *, registry: SubagentRegistry | None = None,
                 policy: RuntimePolicy | None = None, context_builder: Any = None,
                 action_gateway: Any = None, closed_loop_service: ClosedLoopService | None = None,
                 use_llm: bool | None = None, llm_config: Any = None,
                 intent_classifier: IntentClassifier | None = None):
        self.registry = registry or default_registry()
        self.policy = policy or RuntimePolicy()
        self.context_builder = context_builder
        self.action_gateway = action_gateway
        self.closed_loop_service = closed_loop_service
        # LLM classification is the normal path.  ``use_llm=False`` remains
        # an explicit test/configuration switch and is the only way to emit
        # the ``llm_disabled`` fallback reason.
        # Keep deterministic rule routing as the default for local/offline
        # use.  Callers can explicitly opt into an LLM provider with
        # ``use_llm=True``; this prevents an ambient developer .env from
        # changing tests or process-context answers.
        self.use_llm = False if use_llm is None else bool(use_llm)
        self.llm_config = llm_config
        self.intent_classifier = intent_classifier or IntentClassifier(
            llm_config=llm_config, enabled=self.use_llm,
        )
        self._pending_closed_loops: dict[str, dict[str, Any]] = {}
        self._case_adapters: dict[str, ProcessCaseAdapter] = {}

    def answer(self, question: str, session: SessionState | None = None) -> Answer:
        session = session or SessionState(mode=self.policy.mode)
        classification = self.intent_classifier.classify(question, session=session)
        intent = classification.intent
        session.intent = intent
        session.classification_source = classification.classification_source
        session.classification_retry_count = classification.retry_count
        session.fallback_reason = classification.fallback_reason
        session.audit_events.append({
            "event": "intent_classified", "source": classification.classification_source,
            "fallback_reason": classification.fallback_reason, "retry_count": classification.retry_count, "intent": intent.to_dict(),
        })
        if intent.intent == "parameter_impact":
            slots = dict(getattr(intent, "analysis", {}) or {})
            # Load the read-only snapshot even for an incomplete follow-up so
            # callers can retain context/evidence while we ask for fields.
            loaded = self._load_context(["process_context", "yaml", "node_db"], session)
            # Refresh slots after loading the process graph so implicit
            # equipment/stream focus can be resolved by the Lead boundary.
            if hasattr(context := loaded.context, "get_equipment"):
                if not slots.get("target_stream") and slots.get("equipment"):
                    candidates = [s for s in (getattr(context, "streams", []) or [])
                                  if str(s.get("source") or s.get("source_node") or "").casefold() == str(slots["equipment"]).casefold()]
                    if "塔底" in question:
                        bottom = [s for s in candidates if any(k in str(s).casefold() for k in ("bottom", "bottoms", "塔底"))]
                        if len(bottom) == 1:
                            candidates = bottom
                    if len(candidates) == 1:
                        slots["target_stream"] = str(candidates[0].get("id") or candidates[0].get("name"))
                    elif len(candidates) > 1:
                        return Answer(text=f"请明确 {slots['equipment']} 的目标物流（候选：{', '.join(str(s.get('id') or s.get('name')) for s in candidates)}）。",
                                      facts=self._classification_facts(classification), uncertainties=["设备存在多个出口，无法安全推断目标物流。"])
            # Recompute after conservative graph-based stream resolution;
            # classifier diagnostics may have been produced before context was
            # available.
            missing = [f for f, value in (("equipment", slots.get("equipment")), ("target_stream", slots.get("target_stream")),
                                           ("target_component", slots.get("target_component")), ("metric", slots.get("metric"))) if not value]
            if missing:
                return Answer(
                    text=(self._fallback_notice(classification) if classification.classification_source == "rule_fallback" else "") + "需要补充参数影响分析所需信息：" + "；".join({"metric": "评价指标（摩尔分数、纯度或回收率）", "equipment": "设备", "target_stream": "目标物流", "target_component": "目标组分"}.get(x, x) for x in missing),
                    facts=self._classification_facts(classification),
                    uncertainties=["未提供设备、目标物流、目标组分或评价指标，未进行任何 Aspen 操作。"],
                )
            context = loaded.context
            # Resolve a missing stream conservatively from the selected
            # equipment's outgoing streams.  Ambiguous topology requires a
            # clarification instead of silently choosing a product.
            if not slots.get("target_stream") and slots.get("equipment"):
                node = context.get_equipment(str(slots["equipment"])) if hasattr(context, "get_equipment") else None
                candidates = [s for s in (getattr(context, "streams", []) or [])
                              if str(s.get("source") or s.get("source_node") or "").casefold() == str(slots["equipment"]).casefold()]
                if len(candidates) == 1:
                    slots["target_stream"] = str(candidates[0].get("id") or candidates[0].get("name"))
                elif len(candidates) > 1:
                    return Answer(text=f"请明确 {slots['equipment']} 的目标物流（候选：{', '.join(str(s.get('id') or s.get('name')) for s in candidates)}）。",
                                  facts=self._classification_facts(classification), uncertainties=["设备存在多个出口，无法安全推断目标物流。"])
            if slots.get("equipment") and slots.get("target_component") and slots.get("metric"):
                session.runtime_state.setdefault("conversation_focus", {}).update({
                    "equipment": slots.get("equipment"), "stream": slots.get("target_stream"),
                    "component": slots.get("target_component"), "metric": slots.get("metric")})
            impact = build_parameter_impact_plan(
                context, equipment_id=str(slots["equipment"]), target_stream=slots.get("target_stream"),
                target_component=slots.get("target_component"), metric=str(slots["metric"]),
                candidate_parameters=slots.get("candidate_parameters") or None,
                max_cases=int((slots.get("max_cases") or 20)),
            )
            errors = validate_parameter_impact_plan(impact, budget=self.policy.total_budget)
            if errors:
                impact.diagnostics.extend(errors)
            base_model = ""
            builder = self._builder_for_session(session)
            if builder is not None:
                # ProcessContextBuilder keeps the model path on its explicit
                # ``aspen_file`` attribute.  ``config`` is a plain mapping;
                # adapters expose the same path under ``adapter.config``.
                direct = getattr(builder, "aspen_file", None)
                if direct is None:
                    direct = getattr(getattr(builder, "config", None), "aspen_file", None)
                base_model = str(direct or "")
                if not base_model and isinstance(getattr(builder, "config", None), dict):
                    base_model = str(builder.config.get("aspen_file") or builder.config.get("aspen_path") or "")
            if not base_model:
                impact.diagnostics.append("缺少 Aspen base_model：case 未提供 BKP/APW 路径")
            elif not Path(base_model).is_file():
                impact.diagnostics.append(f"Aspen base_model 不存在: {base_model}")
            try:
                simulation = make_simulation_plan(impact, base_model=base_model,
                                                  evidence_ids=[x.id for x in getattr(context, "sources", [])])
            except Exception as exc:
                simulation = None
                impact.diagnostics.append(str(exc))
            if simulation is None or not simulation.parameter_changes or not simulation.executable:
                return Answer(
                    text="暂时无法自动设计参数影响回测：" + "；".join(impact.diagnostics or ["没有可调参数的基准值、边界或单位"]),
                    facts=self._classification_facts(classification),
                    uncertainties=["未执行 Aspen，也未消耗仿真预算；请补充可授权的 Input 参数及其边界。"],
                )
            if self.action_gateway is None:
                return Answer(text="已创建回测计划，但当前运行环境没有配置 Aspen simulation_runner。",
                              facts=self._classification_facts(classification),
                              uncertainties=["未配置 ActionGateway/simulation_runner，计划未执行。"],
                              recommendations=[impact.to_dict(), simulation.to_dict()])
            payload = {"question": question, "intent": intent.to_dict(),
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
                    approval.preview = self.action_gateway.preview_proposal(proposal)
                except Exception as exc:
                    approval.preview = {"status": "preview_failed", "error": str(exc)}
            session.pending_approval = approval
            session.context_summary = loaded.summary
            session.context_sources = loaded.sources
            param_lines = "\n".join(f"- {p.get('name')} ({p.get('aspen_path')})：基准 {p.get('baseline')}，范围 [{p.get('lower')}, {p.get('upper')}]" for p in impact.parameters) or "- 未发现具有完整基准值和边界的可调参数"
            cases = len((simulation.parameter_changes if simulation else []))
            return Answer(text=((self._fallback_notice(classification) if classification.classification_source == "rule_fallback" else "") + f"我可以对 {slots['equipment']} 做局部参数回测。\n\n目标：\n- 目标物流：{slots['target_stream']}\n- 目标组分：{slots['target_component']}\n- 指标：{slots['metric']}\n\n候选参数：\n{param_lines}\n\n计划：\n- 共 {cases} 个有限工况，使用基准附近局部扰动\n- 输出：{', '.join(impact.expected_outputs)}\n\n限制：\n- " + "\n- ".join(impact.diagnostics or ["结论仅适用于本次扰动范围"]) + "\n\n请确认后执行。"),
                          facts=self._classification_facts(classification), recommendations=[proposal.to_dict()], approval_request=approval)
        if intent.action_type in {"modify_file", "set_parameter", "simulate_once", "optimize", "modify_structure"}:
            payload = {"question": question, "intent": intent.to_dict()}
            proposal = ActionProposal(action_type=self._proposal_action_type(intent.action_type), target="model",
                                      reason=question, payload=payload,
                                      evidence_ids=[f"intent:{session.session_id}"],
                                      risks=["ActionGateway will revalidate the current model before execution."],
                                      requires_confirmation=True)
            approval = ApprovalRequest(proposal=proposal, message="Human confirmation is required before execution.")
            if self.action_gateway is not None:
                try:
                    approval.preview = self.action_gateway.preview_proposal(proposal)
                except Exception as exc:
                    approval.preview = {"status": "preview_failed", "error": str(exc)}
            session.pending_approval = approval
            return Answer(text="Action proposal created; human confirmation is required before execution.",
                          facts=self._classification_facts(classification), recommendations=[proposal.to_dict()], approval_request=approval)
        if intent.intent in {"chat", "general_question"} and not intent.subagents:
            answer = self._answer_general(question)
            answer.facts.extend(self._classification_facts(classification))
            if classification.classification_source == "rule_fallback" and not any(
                isinstance(item, dict) and item.get("kind") == "rule_fallback"
                for item in answer.facts
            ):
                answer.facts.insert(0, {"kind": "rule_fallback", "reason": classification.fallback_reason or "unknown"})
            if classification.classification_source == "rule_fallback":
                answer.text = self._fallback_notice(classification) + answer.text
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
        agent = requested.name if requested is not None and requested.name in self.registry.names() else self._route(question)[1]
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
        runtime = OrchestrationRuntime(
            self.registry.as_mapping(), policy=self.policy, audit=session.audit_events,
            role_definitions=self.registry.definitions(),
            checkpoint_path=session.checkpoint_ref, resume=bool(session.checkpoint_ref),
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
        result = results[0]
        if len(results) > 1:
            session.audit_events.append({
                "event": "results_synthesized", "task_ids": [item.task_id for item in results],
                "issues": issues,
            })
        text = result.summary if not issues else "无法形成有充分证据的结论：" + "；".join(issues)
        if classification.classification_source == "rule_fallback":
            text = self._fallback_notice(classification) + text
        if result.status.value != "completed":
            text = "任务未完成：" + ("；".join(result.errors) or result.status.value)
        return Answer(text=text, facts=result.facts, evidence=result.evidence,
                      assumptions=result.assumptions, uncertainties=result.uncertainties,
                      recommendations=result.recommended_actions, task_results=results)

    @staticmethod
    def _classification_facts(classification: Classification) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = [{
            "kind": "intent_classification",
            "classification_source": classification.classification_source,
            "retry_count": classification.retry_count,
            "fallback_reason": classification.fallback_reason,
            "intent": classification.intent.intent,
            "project_scope": classification.intent.project_scope,
            "confidence": classification.intent.confidence,
        }]
        # Keep the fallback visible even when a separately configured answer
        # provider happens to succeed after intent classification failed.
        # Callers and the UI can therefore rely on one explicit fallback fact
        # whenever the structured classifier used rules.
        if classification.classification_source == "rule_fallback":
            facts.insert(0, {
                "kind": "rule_fallback",
                "reason": classification.fallback_reason or "unknown",
            })
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
    def _fallback_notice(classification: Classification) -> str:
        reasons = {"llm_disabled": "未启用模型", "llm_key_missing": "未配置 API key",
                   "request_timeout": "模型请求超时", "llm_api_failure": "模型服务不可用",
                   "low_confidence": "意图不够明确", "invalid_json": "模型响应格式异常",
                   "schema_validation_failed": "模型响应校验失败", "empty_model_result": "模型返回为空"}
        return f"已切换规则模式（{reasons.get(classification.fallback_reason, '意图识别暂不可用')}）。"

    @staticmethod
    def _context_summary(context: Any) -> dict[str, Any]:
        return {
            "model_id": getattr(context, "model_id", None),
            "equipment": len(getattr(context, "equipment", []) or []),
            "streams": len(getattr(context, "streams", []) or []),
            "variables": len(getattr(context, "variables", {}) or {}),
            "history": len(getattr(context, "history_evidence", []) or []),
        }

    def _answer_general(self, question: str) -> Answer:
        """Answer a general turn without echoing it back to the user.

        The CLI can opt into an LLM provider.  Missing keys or provider
        failures produce an explicit deterministic response instead of a
        fabricated answer or a confusing verbatim echo.
        """
        if self.use_llm:
            try:
                from src.agents.lead import llm_client
                cfg = self.llm_config or llm_client.load_llm_config()
                if llm_client.is_configured(cfg):
                    text = llm_client.chat(
                        cfg,
                        "You are PAO LeadAgent. Answer clearly and concisely. "
                        "Do not claim Aspen simulation results unless evidence is supplied.",
                        question,
                    ).strip()
                    if text:
                        return Answer(
                            text=text,
                            facts=[{"kind": "llm_answer", "provider": cfg.provider, "model": cfg.model}],
                            evidence=[Evidence("llm", cfg.model, "Provider response", text, confidence=None)],
                        )
            except Exception as exc:  # noqa: BLE001
                fallback_reason = f"LLM unavailable ({type(exc).__name__})"
            else:
                fallback_reason = "LLM key is not configured"
        else:
            fallback_reason = "LLM mode is disabled"
        return Answer(
            text=("我是 PAO LeadAgent。当前使用规则模式，暂时无法生成开放式问答；"
                  "你可以询问工艺、边界、优化目标，或配置 LLM key 后继续。"),
            facts=[{"kind": "rule_fallback", "reason": fallback_reason}],
            evidence=[Evidence("system", "LeadAgent", "Deterministic fallback", question, confidence=1.0)],
            uncertainties=[fallback_reason],
        )

    def propose_closed_loop(self, *, question: str, config: Any, driver: Any = None,
                            session: SessionState | None = None, **kwargs: Any) -> Answer:
        """Create an approval request for an internal optimization loop.

        No driver is opened and no workflow is started here.  The returned
        proposal is the only hand-off from the public Agent boundary to the
        closed-loop service.
        """
        session = session or SessionState(mode=self.policy.mode)
        classification = self.intent_classifier.classify(question, session=session)
        session.intent = classification.intent
        session.classification_source = classification.classification_source
        session.classification_retry_count = classification.retry_count
        session.fallback_reason = classification.fallback_reason
        session.audit_events.append({
            "event": "intent_classified", "source": classification.classification_source,
            "fallback_reason": classification.fallback_reason, "retry_count": classification.retry_count, "intent": classification.intent.to_dict(),
        })
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
            except Exception as exc:
                approval.preview = {"status": "preview_failed", "error": str(exc)}
        session.pending_approval = approval
        self._pending_closed_loops[session.session_id] = {
            "driver": driver, "config": config, "kwargs": kwargs,
            "approval": approval,
        }
        notice = self._fallback_notice(classification) if classification.classification_source == "rule_fallback" else ""
        return Answer(text=notice + "Closed-loop proposal created; waiting for human confirmation.",
                      facts=self._classification_facts(classification),
                      recommendations=[proposal.to_dict()], approval_request=approval)

    def confirm_closed_loop(self, session: SessionState, *, approved: bool = True) -> Any:
        """Confirm/reject a pending loop and execute only after approval."""
        pending = self._pending_closed_loops.get(session.session_id)
        if pending is None or session.pending_approval is None:
            raise ValueError("no pending closed-loop approval")
        approval = session.pending_approval
        if not approved:
            approval.status = "rejected"
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
            if self.closed_loop_service is None:
                self.closed_loop_service = ClosedLoopService()
            metadata = {
                "task": pending["config"].__class__.__name__,
                "intent": session.intent.to_dict() if session.intent else None,
                "context_summary": session.context_summary,
                "context_sources": session.context_sources,
                "evidence": list(approval.proposal.evidence_ids),
                "action_proposal": approval.proposal.to_dict(),
                "approval_request": approval.to_dict(),
                "audit_events": list(session.audit_events),
            }
            result = self.closed_loop_service.run(
                driver=pending["driver"], config=pending["config"], approved=True,
                checkpoint_path=getattr(pending["config"], "agent_checkpoint_path", None),
                checkpoint_metadata=metadata, **pending["kwargs"])
        approval.status = "executed"
        session.pending_approval = None
        self._pending_closed_loops.pop(session.session_id, None)
        return result

    def confirm_action(self, session: SessionState, *, approved: bool = True) -> Any:
        """Confirm and execute a pending proposal through ActionGateway."""
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
            return Answer(text="Action request rejected.", approval_request=approval)
        try:
            self.action_gateway.approve(audit_id, actor="user")
            result = self.action_gateway.execute(audit_id, actor="lead_agent")
        except Exception as exc:
            approval.status = "failed"
            session.pending_approval = None
            return Answer(text=f"动作未完成：{type(exc).__name__}: {exc}",
                          uncertainties=["执行失败；请诊断原因后重新创建计划。"],
                          approval_request=approval)
        if isinstance(result, dict) and approval.proposal.payload.get("parameter_impact_plan"):
            result = dict(result)
            result["answer"] = self._render_parameter_impact_result(
                approval.proposal.payload.get("parameter_impact_plan") or {}, result)
        approval.status = "executed"
        session.pending_approval = None
        return result

    @staticmethod
    def _render_parameter_impact_result(plan: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Render executed evidence in facts/mechanism/statistical layers."""
        cases = list(result.get("simulation_cases") or [])
        sensitivity = dict(result.get("sensitivity") or {})
        valid = [c for c in cases if c.get("converged") is True]
        slots = {
            "equipment": plan.get("equipment_id"), "target_stream": plan.get("target_stream"),
            "target_component": plan.get("target_component"), "metric": plan.get("metric"),
        }
        facts = [{"layer": "observed_fact", "case_count": len(cases),
                  "converged_case_count": len(valid),
                  "planned_case_count": result.get("planned_case_count", len(cases)),
                  "skipped_case_count": result.get("skipped_case_count", 0),
                  "valid_case_count": sensitivity.get("effective_case_count", 0), **slots}]
        mechanism = ["回流比、进料板位置和再沸器负荷等机理候选需结合具体参数和流程证据核验。"]
        ranking = sensitivity.get("parameter_rank") or []
        if sensitivity.get("error"):
            conclusion = "敏感性分析失败：" + str(sensitivity["error"])
        elif sensitivity.get("effective_case_count", 0) == 0:
            conclusion = "没有可用于分析的有效目标指标，无法判断参数影响。"
        elif sensitivity.get("conclusion_type") == "insufficient_data":
            conclusion = "有效独立工况不足，无法判断参数影响。"
        elif ranking:
            conclusion = "；".join(f"{x.get('parameter')}：{x.get('direction')}，局部敏感性 {x.get('sensitivity', 0):.4g}" for x in ranking)
        else:
            conclusion = "有效工况不足以形成参数排序。"
        limitations = list(sensitivity.get("limitations") or [])
        user_limitations = list(limitations)
        failure_summaries = []
        for case in cases:
            errors = case.get("errors") or []
            if errors:
                summary = (str(errors[0]).splitlines() or ["未知错误"])[0][:160]
                if summary not in failure_summaries:
                    failure_summaries.append(summary)
            for message in [*(case.get("errors") or []), *(case.get("warnings") or [])]:
                if str(message) not in limitations:
                    limitations.append(str(message))
        limitations.append("结论只适用于本次扰动范围；不代表全局因果关系。")
        return {"observed_facts": facts, "mechanism_candidates": mechanism,
                "backtest_conclusion": conclusion, "limitations": limitations,
                "sensitivity": sensitivity, "user_limitations": user_limitations,
                "failure_summaries": failure_summaries[:3]}

    @staticmethod
    def _is_general(question: str) -> bool:
        return not any(term in question.casefold() for term in _PAO_TERMS)

    @staticmethod
    def _route(question: str) -> tuple[str, str]:
        q = question.casefold()
        if "动力学" in q or "kinetic" in q:
            return "kinetics_analysis", "kinetics_advisor"
        if "边界" in q or "boundary" in q:
            return "boundary_review", "boundary_advisor"
        if "优化" in q or "pareto" in q:
            return "optimization_analysis", "optimization_advisor"
        if "风险" in q or "risk" in q:
            return "general_question", "risk_reviewer"
        return "project_research", "project_researcher"

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
        if key not in self._case_adapters:
            self._case_adapters[key] = ProcessCaseAdapter.from_config(config_ref)
        return self._case_adapters[key]

    def _load_context(self, required: list[str], session: SessionState) -> Any:
        builder = self._builder_for_session(session)
        loader = builder if isinstance(builder, ContextLoader) else ContextLoader(builder)
        files = session.context_refs.get("files") if isinstance(session.context_refs, dict) else None
        return loader.load(required, session=session, files=[files] if isinstance(files, str) else files)

    @staticmethod
    def _is_action_request(question: str) -> bool:
        q = question.casefold()
        return any(token in q for token in (
            "set ", "modify", "write", "run simulation", "run optimization",
            "修改", "写入", "运行仿真", "执行优化", "改变边界",
        ))
