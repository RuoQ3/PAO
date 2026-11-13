"""Serial live acceptance driver. Uses real provider, production sessions and gateway.

Run with the repository Python; artifacts contain redacted requests/responses.
No fake providers, executors, classification or engineering evidence are installed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.lead import ConversationSession, LeadAgent
from src.agents.lead.llm_client import load_llm_config, chat_json
from src.agents.context import ProcessContextBuilder
from src.agents.actions import ActionGateway
from src.agents.actions.plans import build_optimization_plan
from src.agents.actions.authorization import file_digest
from src.agents.orchestration.models import SessionState, RuntimePolicy
from src.agents.orchestration.parameter_impact import AspenSandboxRunner
from src.agents.reasoning import redact
from src.database.simulation_db import SimulationDB
from src.reporting.answer_terminal import TerminalRenderer
from src.reporting.answer_export import markdown_answer
import yaml


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class LiveRun:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.cfg = load_llm_config()
        self.current = {}
        self.sessions = {}
        self.db = SimulationDB(self.directory / "simulation.db")
        self.llm_calls = 0
        self.pending_provider_calls = 0

    def provider(self, cfg, *, system, user):
        started = time.monotonic()
        self.llm_calls += 1
        self.pending_provider_calls += 1
        event = {**self.current, "call_id": uuid.uuid4().hex, "process_call_index": self.llm_calls, "provider": cfg.provider,
                 "model": cfg.model, "system": system, "user": user}
        try:
            result = chat_json(cfg, system, user)
            event.update(response=result, success=True)
            return result
        except Exception as exc:
            event.update(success=False, error=exc.to_dict() if hasattr(exc, "to_dict") else type(exc).__name__)
            raise
        finally:
            event["elapsed_seconds"] = time.monotonic() - started
            with (self.directory / "llm_calls.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(redact(event, cfg), ensure_ascii=False, default=str) + "\n")
            self.pending_provider_calls -= 1

    def session(self, name, config="pareto"):
        if name in self.sessions:
            return self.sessions[name]
        path = self.directory / f"{config}.yaml"
        builder = ProcessContextBuilder(yaml_path=path, node_db=ROOT / "cases/demo_case/output/node.db",
                                        simulation_db=self.directory / "simulation.db", read_only=True)
        owner = self
        class Gateway:
            instance = None
            instances = []
            def preview_proposal(self, proposal):
                plan = proposal.payload.get("optimization_plan")
                if plan is not None or self.instance is None:
                    _, schema = build_optimization_plan(owner.directory / "ga.yaml" if plan is None else plan["config_path"],
                        overrides=plan.get("configuration_overrides") if plan else None)
                    runner = AspenSandboxRunner(driver_factory=lambda _path: __import__("src.aspen_driver.driver", fromlist=["AspenDriver"]).AspenDriver(),
                                                allowed_paths=set(schema.parameters), require_path_whitelist=True)
                    self.instance = ActionGateway(schema, simulation_runner=runner, simulation_db=owner.db)
                    self.instances.append(self.instance)
                return self.instance.preview_proposal(proposal)
            def reset_project(self):
                self.instance = None
            def __getattr__(self, key):
                if self.instance is None:
                    raise AttributeError(key)
                return getattr(self.instance, key)
        gateway = Gateway()
        session = ConversationSession(SessionState(), LeadAgent(llm_config=self.cfg, llm_call=self.provider,
            context_builder=builder, action_gateway=gateway, policy=RuntimePolicy(max_concurrency=1)))
        self.sessions[name] = (session, gateway, TerminalRenderer())
        return self.sessions[name]

    def turn(self, spec):
        if self.pending_provider_calls:
            raise RuntimeError("A timed-out provider request is still pending; stop this batch before dispatching another request")
        case = spec["id"]
        session, gateway, renderer = self.session(spec.get("session", case), spec.get("config", "pareto"))
        if spec.get("resume_from"):
            saved = json.loads((self.directory / "sessions" / spec["resume_from"]).read_text(encoding="utf-8"))
            session = ConversationSession(SessionState.from_dict(saved["postconditions"]), session.lead)
            self.sessions[spec.get("session", case)] = session, gateway, renderer
        before = deepcopy(session.to_dict())
        started = time.monotonic()
        first_call = self.llm_calls
        self.current = {"case_id": case, "session_id": session.state.session_id}
        if spec.get("files"):
            session.state.context_refs["files"] = spec["files"]
        result = {**spec, "session_id": session.state.session_id, "label": "real",
                  "preconditions": before, "expected": spec.get("expected", "参见原始验收任务中同 ID 的产品契约"),
                  "evidence": str(self.directory / "sessions" / f"{case}.json")}
        try:
            if spec.get("restore"):
                checkpoint = self.directory / "sessions" / f"{case}.checkpoint.db"
                session.save_checkpoint(checkpoint)
                session = ConversationSession.from_checkpoint(checkpoint, session.lead)
                self.sessions[spec.get("session", case)] = session, gateway, renderer
            text = spec["input"]
            command, _, argument = text.partition(" ")
            if command in {"/details", "/value", "/export"}:
                answer = session.last_answer
                if answer is None:
                    result["response"] = "当前没有可查看的回答。"
                elif command == "/details":
                    result["response"] = TerminalRenderer().render(answer, details=True)
                elif command == "/value":
                    try:
                        result["response"] = renderer.render_value(argument)
                    except KeyError:
                        result["response"] = "路径编号不存在；输入 /answer 重新显示编号。"
                else:
                    destination = Path(argument)
                    with destination.open("x", encoding="utf-8") as stream:
                        stream.write(answer.to_json(indent=2) if destination.suffix == ".json" else markdown_answer(answer))
                    result["response"] = f"已导出：{destination}"
            else:
                # Only allowlisted real action cases can submit an approval.
                if text.strip() in {"确认", "确认执行", "yes", "批准"} and session.state.pending_approval:
                    if not spec.get("execute"):
                        if not spec.get("restore"):
                            raise RuntimeError("Safety stop: approval not allowlisted for execution")
                    else:
                        self.check_execution(session)
                answer = session.submit(text)
                result["answer"] = answer.to_dict()
                result["response"] = renderer.render(answer)
            result["status"] = "UNREVIEWED"
        except Exception as exc:
            result.update(status="BLOCKED", response=str(exc), error=exc.to_dict() if hasattr(exc, "to_dict") else {"type": type(exc).__name__, "message": str(exc)})
        result.update(elapsed_seconds=time.monotonic() - started, llm_calls=self.llm_calls - first_call,
                      postconditions=session.to_dict(), gateway_audits=[g.audit.to_dict() for g in gateway.instances])
        for event in session.state.audit_events:
            if event.get("event") == "action_executed":
                execution = event["result"]
                target = self.directory / "executions" / f"{execution['audit_id']}.json"
                if not target.exists():
                    write(target, redact(event, self.cfg))
        result = redact(result, self.cfg)
        write(self.directory / "sessions" / f"{case}.json", result)
        session.save_checkpoint(self.directory / "sessions" / f"session-{session.state.session_id}.db")
        with (self.directory / "cases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        print(json.dumps({k: (result[k][:550] if k == "response" else result[k]) for k in ("id", "status", "llm_calls", "elapsed_seconds", "response")}, ensure_ascii=False), flush=True)
        write(self.directory / "progress.json", {"last_case": case, "llm_calls_this_process": self.llm_calls,
              "budget": self.budget(), "updated": datetime.now().isoformat()})
        return result

    def budget(self):
        import sqlite3
        conn = sqlite3.connect(self.directory / "simulation.db")
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
        evaluations = [dict(r) for r in conn.execute("select * from execution_evaluations")] if "execution_evaluations" in tables else []
        budgets = [dict(r) for r in conn.execute("select * from execution_budget")] if "execution_budget" in tables else []
        conn.close()
        action_seconds = 0.0
        known_actions = {}
        for path in (self.directory / "executions").glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("result"):
                known_actions[data["result"]["audit_id"]] = data.get("action_wall_seconds", 600)
        action_seconds = sum(known_actions.values())
        return {"evaluations_used": len(evaluations), "evaluations_remaining": 64 - len(evaluations),
                "conservative_action_wall_seconds": action_seconds,
                "aspen_seconds_remaining": 3600 - action_seconds, "ledgers": budgets}

    def check_execution(self, session):
        approval = session.state.pending_approval
        preview = approval.preview or {}
        if not preview.get("valid"):
            raise RuntimeError("No validated concrete gateway preview")
        payload = approval.proposal.payload
        plan = payload.get("optimization_plan") or payload.get("simulation_plan")
        if not plan:
            raise RuntimeError("No concrete plan")
        limits = plan.get("limits") or {"max_evaluations": plan["max_evaluations"], **plan.get("adaptive_policy", {})}
        used = self.budget()["evaluations_used"]
        if self.budget()["aspen_seconds_remaining"] < limits.get("max_seconds", 600):
            raise RuntimeError("Plan exceeds cumulative Aspen time allowance")
        if limits["max_evaluations"] > 8 or limits.get("max_seconds", 9999) > 600 or limits.get("max_failures", 9999) > 4 or used + limits["max_evaluations"] > 64:
            raise RuntimeError("Plan exceeds acceptance budget")
        if payload.get("optimization_plan"):
            cfg = plan["configuration"]
            original = yaml.safe_load((self.directory / "ga.yaml").read_text(encoding="utf-8"))
            constraints = lambda c: [(r["aspen_path"], r["operator"], r["threshold"]) for r in c["constraints"]]
            if cfg["design_variables"] != original["design_variables"] or constraints(cfg) != constraints(original):
                raise RuntimeError("Engineering bounds or purity constraint changed")
        write(self.directory / "executions" / f"{self.current['case_id']}-preview.json", approval.to_dict())


def initialize(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for name, source in (("ga", "ga_purity_config.yaml"), ("pareto", "pareto_config.yaml")):
        # byte-identical source copy; model path is project-root relative.
        (directory / f"{name}.yaml").write_bytes((ROOT / "cases/demo_case" / source).read_bytes())
    write(directory / "baseline.json", {"source_hashes": {str(p): file_digest(p) for p in (
        ROOT / "cases/demo_case/ga_purity_config.yaml", ROOT / "cases/demo_case/pareto_config.yaml",
        ROOT / "cases/demo_case/二级氢氰化工段.bkp")}, "limits": {"evaluations": 64, "aspen_seconds": 3600},
        "provider": load_llm_config().provider, "model": load_llm_config().model})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    if args.init:
        initialize(args.directory)
    logging.basicConfig(filename=args.directory / "run.log", level=logging.DEBUG, encoding="utf-8")
    run = LiveRun(args.directory)
    if args.spec:
        for spec in json.loads(args.spec.read_text(encoding="utf-8")):
            result = run.turn(spec)
            if result.get("error", {}).get("code") in {"llm_payment_required", "llm_request_failed", "llm_timeout"}:
                print("Provider request failed; batch stopped without further dispatch.", flush=True)
                break
