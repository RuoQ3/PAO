"""First usable, read-only engineering conversation session.

This adapter intentionally never opens Aspen COM, writes YAML/Aspen, runs a
simulation, or starts optimization.  It composes the existing ProcessContext
builder and TechnicalAdvisor behind the CLI conversation commands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.contracts.models import EvidenceBundle, EvidenceSource
from src.agents.conversation.session import ConversationSession
from src.agents.conversation.state import ConversationState
from src.agents.process_context import ProcessContextBuilder, ProcessContextBuildError
from src.agents.coordinator.orchestrator import orchestrate_turn
from src.agents.conversation.dynamic import plan_conversation_turn, scope_plan_tasks, DynamicPlan
from src.agents.conversation.agents import LocalCaseAgents
from src.agents.conversation.synthesis import synthesize_answer
from src.agents.llm_client import load_llm_config
from src.integrations.ima import ImaServiceClient, MockImaClient


class UnifiedConversation:
    """Stateful read-only session used by ``python -m src.main --chat``."""

    def __init__(
        self,
        *,
        aspen_file: str | Path | None = None,
        yaml_path: str | Path | None = None,
        simulation_db_path: str | Path | None = None,
        node_db_path: str | Path | None = None,
        ima_client: Any = None,
        llm_config: Any = None,
        planner_chat_fn: Any = None,
        synthesizer_chat_fn: Any = None,
    ) -> None:
        self.state = ConversationState(
            aspen_file=str(Path(aspen_file).resolve()) if aspen_file else None,
            yaml_path=str(Path(yaml_path).resolve()) if yaml_path else None,
            simulation_db_path=str(Path(simulation_db_path).resolve()) if simulation_db_path else None,
            node_db_path=str(Path(node_db_path).resolve()) if node_db_path else None,
        )
        base = Path(aspen_file or yaml_path).resolve().parent if (aspen_file or yaml_path) else Path.cwd()
        if self.state.simulation_db_path is None:
            for candidate in (base / "output" / "simulation.db", base / "simulation.db"):
                if candidate.exists():
                    self.state.simulation_db_path = str(candidate)
                    break
        if self.state.node_db_path is None:
            for candidate in (base / "output" / "node.db", base / "node.db", base / "output" / "nodes.db"):
                if candidate.exists():
                    self.state.node_db_path = str(candidate)
                    break
        self._ima = ima_client
        self._llm_config = llm_config
        self._planner_chat_fn = planner_chat_fn
        self._synthesizer_chat_fn = synthesizer_chat_fn
        self._builder: ProcessContextBuilder | None = None
        self._context = None
        self.last_bundle = EvidenceBundle(complete=False)
        self.last_plan = None
        self._initialize()
        self.session = ConversationSession(self.state, self._answer)

    @property
    def context(self):
        return self._context

    def _initialize(self) -> None:
        if self._builder is not None:
            self._builder.close()
            self._builder = None
        diagnostics: list[str] = []
        if self.state.aspen_file and not Path(self.state.aspen_file).exists():
            diagnostics.append(f"Aspen 文件不存在：{self.state.aspen_file}")
        if self.state.node_db_path and not Path(self.state.node_db_path).exists():
            diagnostics.append(f"NodeDB 不存在：{self.state.node_db_path}")
        if self.state.simulation_db_path and not Path(self.state.simulation_db_path).exists():
            diagnostics.append(f"SimulationDB 不存在：{self.state.simulation_db_path}")
        if self.state.yaml_path and not Path(self.state.yaml_path).exists():
            diagnostics.append(f"YAML 配置不存在：{self.state.yaml_path}")
        try:
            self._builder = ProcessContextBuilder(
                yaml_path=self.state.yaml_path,
                node_db=self.state.node_db_path if self.state.node_db_path and Path(self.state.node_db_path).exists() else None,
                aspen_file=self.state.aspen_file,
                # Do not create a database during a read-only conversation.
                simulation_db=self.state.simulation_db_path if self.state.simulation_db_path and Path(self.state.simulation_db_path).exists() else None,
                read_only=True,
            )
            # ``audit_id`` identifies this chat session; it is not the
            # optimizer's SimulationDB session_id.  Filtering the database
            # with it makes an otherwise populated history look empty.  A
            # read-only conversation should inspect the available history
            # unless an explicit simulation session selector is introduced.
            self._context = self._builder.build(session=None)
            self._context.diagnostics[:0] = diagnostics
        except (ProcessContextBuildError, OSError, ValueError) as exc:
            diagnostics.append(str(exc))
            self._context = None
        if not self.state.node_db_path:
            diagnostics.append("NodeDB 未提供；无法读取 Aspen 节点目录和当前值。")
        if not self.state.simulation_db_path:
            diagnostics.append("SimulationDB 未提供；无法读取历史工况和收敛记录。")
        if self.state.simulation_db_path and self._context is not None and not self._context.history_evidence:
            diagnostics.append("SimulationDB 历史数据为空；无法支持历史比较或收敛趋势判断。")
        self.state.context_summary = self._summary(diagnostics)

    @staticmethod
    def _summary(extra: list[str]) -> str:
        return "\n".join(["只读 ProcessContext 已初始化。"] + [f"- {x}" for x in extra])

    def _ima_client(self):
        if self._ima is not None:
            return self._ima
        try:
            self._ima = ImaServiceClient.from_env()
        except Exception:
            self._ima = None
        return self._ima

    def _answer(self, question: str, state: ConversationState) -> str:
        if self._context is None:
            self.last_bundle = EvidenceBundle(missing=["ProcessContext 不可用"], complete=False)
            state.last_analysis = {"execution_path": {
                "planner": "local_rules", "executor": "local_agent",
                "synthesizer": "local_rules", "fallback": "ProcessContext 不可用",
            }}
            return self._structured(
                "无法构建 ProcessContext。",
                [], [], ["请先用 /open 打开存在的 Aspen 文件，并提供可读 YAML/NodeDB。"],
                ["Aspen/YAML/NodeDB 路径或内容不可用"],
            )
        cfg = self._llm_config if self._llm_config is not None else load_llm_config()
        plan = plan_conversation_turn(
            question, self._context, current_equipment=state.current_equipment,
            history=state.history[-8:], llm_config=cfg, chat_fn=self._planner_chat_fn,
        )
        direct_equipment = [str(item.get("name")) for item in self._context.equipment
                            if str(item.get("name", "")).casefold() in question.casefold()]
        if not direct_equipment and _pronoun_question(question) and len(state.equipment_focus) > 1:
            plan = DynamicPlan(
                "clarification", needs_clarification=True,
                clarification=f"你指的是 {' 还是 '.join(state.equipment_focus)}？",
                planner_mode=plan.planner_mode, used_llm=plan.used_llm,
                fallback_reason=plan.fallback_reason, confidence=1.0,
            )
        plan = scope_plan_tasks(plan)
        self.last_plan = plan
        if not plan.needs_clarification and plan.entities.get("equipment"):
            state.equipment_focus = list(plan.entities["equipment"])
            state.current_equipment = state.equipment_focus[0] if len(state.equipment_focus) == 1 else None
        if plan.entities.get("objectives"):
            state.current_objective = plan.entities["objectives"][0]

        agents = LocalCaseAgents(self._ima_client())
        orchestration = orchestrate_turn(
            question, process_context=self._context,
            consultants=agents.registry(), session=state, dynamic_plan=plan,
        )
        execution_path = {
            "planner": plan.planner_mode,
            "executor": "local_agent",
            "synthesizer": "local_rules",
            "fallback": plan.fallback_reason or ("none" if plan.planner_mode == "llm" else "planner=local_rules"),
        }
        response = synthesize_answer(
            question, plan, orchestration, llm_config=cfg,
            chat_fn=self._synthesizer_chat_fn, path=execution_path,
        )
        execution_path["synthesizer"] = response.mode
        execution_path["fallback"] = response.fallback_reason or (
            "none" if response.mode == "llm" and plan.planner_mode == "llm"
            else execution_path.get("fallback", "none")
        )
        self.last_bundle = orchestration.evidence
        self.last_plan = plan
        state.last_analysis = {
            **plan.to_dict(),
            "call_trace": list(orchestration.call_trace),
            "evidence_ids": list(response.used_evidence_ids or []),
            "missing": list(orchestration.evidence.missing),
            "execution_path": execution_path,
            "synthesis_diagnostics": response.diagnostics,
            "resolved_context": {"equipment": list(state.equipment_focus),
                                 "objective": state.current_objective,
                                 "experiment": state.current_experiment},
        }
        return response.answer
    @staticmethod
    def _structured(
        conclusion: str,
        sources: list[str],
        assumptions: list[str],
        next_steps: list[str],
        diagnostic_notes: list[str],
        *, mechanism: str | None = None,
        evidence: list[Any] | None = None,
        uncertainty_level: str | None = None,
    ) -> str:
        # Keep the output stable and readable for both humans and integration tests.
        level = uncertainty_level or "unknown"
        lines = [f"结论：{conclusion}"]
        if mechanism:
            lines.append(f"机制/解释：{mechanism}")
        lines.append("来源：" + (", ".join(sources) if sources else "暂无可用来源"))
        lines.append("假设：" + ("；".join(dict.fromkeys(assumptions)) if assumptions else "无"))
        lines.append(f"不确定性：{level}")
        lines.append("下一步：" + ("；".join(dict.fromkeys(next_steps)) if next_steps else "无"))
        if diagnostic_notes:
            lines.append("诊断提示：" + "；".join(dict.fromkeys(diagnostic_notes)))
        return "\n".join(lines)

    def submit(self, text: str):
        return self.session.submit(text)

    def close(self) -> None:
        if self._builder is not None:
            self._builder.close()
        if self._ima is not None and hasattr(self._ima, "close"):
            self._ima.close()


def build_unified_session(**kwargs: Any) -> UnifiedConversation:
    return UnifiedConversation(**kwargs)


def _pronoun_question(text: str) -> bool:
    return any(token in text for token in ("它", "这个塔", "这个设备", "该设备", "前面的塔"))
