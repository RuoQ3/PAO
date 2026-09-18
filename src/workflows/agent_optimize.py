"""Persistent observe/decide/ask/evaluate/tell loop with one Aspen writer."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import uuid
from dataclasses import asdict, replace

from src.agents.closed_loop.advisor import ProcessDecisionAgent, fallback_plan
from src.agents.closed_loop.contracts import ActionPlan, LoopConfig, finite, validate_plan
from src.agents.closed_loop.journal import SessionJournal, decode_case, encode_case
from src.models.process_case import CaseStatus, ProcessCase
from src.optimization.search_session import ParetoSearchSession
from src.optimization.feasibility import FeasibilityClassifier
from src.optimization.convergence_kb import match_failure_patterns
from src.workflows.common import (
    apply_derived_vars,
    feasibility_feature_names,
    fingerprint_design_vars,
    repair_design_vars,
)
from src.workflows.optimize_pareto_case import (
    _compute_hv_fixed, _extract_all_objectives, _maybe_recover_driver, _preflight_blocked,
    _build_feasibility_rows,
)
from src.workflows.run_case import run_case

_log = logging.getLogger(__name__)


def _valid(case, config):
    return case.success and _extract_all_objectives(case, config) is not None


def _metrics(state, config):
    cases = [decode_case(row) for row in state["observations"]]
    ref, hv = _compute_hv_fixed(cases, config, state.get("hv_reference"))
    state["hv_reference"] = ref
    recent = cases[-state["batch_size"]:]
    return {
        "hypervolume": hv,
        "feasible_rate": sum(_valid(c, config) for c in recent) / max(1, len(recent)),
        "convergence_rate": sum(c.simulation_valid for c in recent) / max(1, len(recent)),
        "feasible_count": sum(_valid(c, config) for c in cases),
    }


def _anchor(state):
    # Prefer the latest valid feasible point, then a converged point, then the baseline.
    for flag in ("success", "simulation_valid"):
        for row in reversed(state["observations"]):
            if row[flag]:
                return dict(row["optimizer_inputs"])
    return dict(state["initial_point"])


def build_snapshot(state, config, settings, context, knowledge):
    metrics = _metrics(state, config)
    recent = state["observations"][-max(12, settings.batch_size):]
    errors = [(r.get("sim_result") or {}).get("error", "") or r.get("notes", "")
              for r in recent if r.get("sim_result") or r.get("notes")]
    diagnoses = [asdict(d) for d in match_failure_patterns(errors)]
    # Keep raw checkpoints complete; bound the LLM context size.
    compact = []
    for row in recent:
        compact.append({k: row[k] for k in (
            "case_id", "status", "success", "simulation_valid", "optimizer_inputs",
            "actual_inputs", "objectives", "constraints", "initialization", "run_time",
        )})
        compact[-1]["error"] = ((row.get("sim_result") or {}).get("error") or row.get("notes", ""))[:2000]
        compact[-1]["block_statuses"] = row.get("block_statuses", [])[:20]
    from src.optimization.pareto import compute_pareto
    front = compute_pareto([decode_case(r) for r in state["observations"]], config.objective_names).first_front
    front_ids = {c.case_id for c in front.cases} if front else set()
    front_evidence = [{"case_id": r["case_id"], "optimizer_inputs": r["optimizer_inputs"],
                       "objectives": r["objectives"], "constraints": r["constraints"]}
                      for r in state["observations"] if r["case_id"] in front_ids][:12]
    return {
        "process_context": context, "knowledge": knowledge,
        "pareto_evidence": front_evidence,
        "hard_bounds": state["hard_bounds"], "search_region": state["search_region"],
        "integer_paths": sorted(config.integer_var_paths), "anchor": _anchor(state),
        "remaining_evaluations": settings.max_evaluations - state["used"],
        "batch_size_limit": settings.batch_size,
        "max_step_fraction": settings.max_step_fraction,
        "objective_targets": settings.objective_targets,
        "metrics": metrics, "recent_convergence_rate": metrics["convergence_rate"],
        "stagnation_count": state["stagnation_count"], "recent_observations": compact,
        "diagnostic_hypotheses": diagnoses, "previous_actions": state["decisions"][-4:],
    }


def _finish_action(state, config):
    if not state["decisions"] or "outcome" in state["decisions"][-1]:
        return
    decision = state["decisions"][-1]
    before, after = decision["before"], _metrics(state, config)
    objective_gain = ((after["hypervolume"] or 0) > (before["hypervolume"] or 0) + 1e-10)
    gain = objective_gain or after["feasible_count"] > before["feasible_count"]
    state["stagnation_count"] = 0 if gain else state["stagnation_count"] + 1
    effect = decision["plan"]["expected_effect"]
    improved = {
        "objective": objective_gain,
        "feasibility": after["feasible_rate"] > before["feasible_rate"],
        "convergence": after["convergence_rate"] > before["convergence_rate"],
        "information": None,
    }[effect]
    decision["outcome"] = {
        "after": after, "expected_metric_improved": improved,
        "case_ids": [r["case_id"] for r in state["observations"][decision["start_index"]:]],
        "interpretation": "Observed association only; not proof of causal attribution",
    }


def _choose_plan(state, config, settings, agent, context, knowledge):
    snapshot = build_snapshot(state, config, settings, context, knowledge)
    rejected = None
    try:
        plan, source = agent.propose(snapshot)
        validate_plan(plan, state, settings, config.integer_var_paths, {r["id"] for r in knowledge})
    except Exception as exc:
        rejected = f"{type(exc).__name__}: {exc}"[:1000]
        _log.warning("Agent proposal rejected; using deterministic fallback: %s", rejected)
        plan, source = fallback_plan(snapshot, settings.batch_size), "rules:rejected_proposal"
        validate_plan(plan, state, settings, config.integer_var_paths, {r["id"] for r in knowledge})
    decision = {"id": str(uuid.uuid4()), "plan": plan.to_dict(), "source": source,
                "rejection": rejected, "before": snapshot["metrics"],
                "start_index": len(state["observations"])}
    state["decisions"].append(decision)
    state["search_region"].update(plan.search_region)
    state["active_plan"] = plan.to_dict()
    state["batch_remaining"] = plan.batch_size
    return plan


def _prepare_candidate(x, config, hard):
    if set(x) != set(hard):
        raise ValueError("Candidate variable set differs from configured variables")
    repaired, _ = repair_design_vars(
        {**config.fixed_vars, **x}, config.integer_var_paths, hard, config.var_dependencies,
    )
    point = {p: finite(repaired[p]) for p in hard}
    # Repair may violate a bound when a dependency is incompatible. Reject it explicitly.
    for key, value in point.items():
        lo, hi = hard[key]
        if not lo <= value <= hi or (key in config.integer_var_paths and not value.is_integer()):
            raise ValueError("Repaired candidate violates bounds/type")
    for key, rules in config.var_dependencies.items():
        for op, other in rules.items():
            if key in repaired and other in repaired:
                if op not in {"lt", "le"}:
                    raise ValueError("Unknown dependency operator")
                if not (repaired[key] < repaired[other] if op == "lt" else repaired[key] <= repaired[other]):
                    raise ValueError("Unsatisfied variable dependency")
    actual, _ = apply_derived_vars(repaired, config.derived_var_specs)
    return point, actual


def _local_region(state, config, settings):
    center = _anchor(state)
    local = {}
    for key, (lo, hi) in state["search_region"].items():
        width = state["hard_bounds"][key][1] - state["hard_bounds"][key][0]
        step = settings.max_step_fraction * width
        if key in config.integer_var_paths:
            step = max(1, math.floor(step))
        left, right = max(lo, center[key] - step), min(hi, center[key] + step)
        if left > right:
            raise ValueError("Requested search region is not reachable within one allowed step")
        local[key] = [left, right]
    return local


def _pick_candidate(state, config, settings, session):
    plan = ActionPlan.parse(state["active_plan"])
    anchor = _anchor(state)
    local = _local_region(state, config, settings)
    seen = {fingerprint_design_vars(r["design_vars"]) for r in state["observations"]}
    options = []
    if not state["observations"]:
        options.append(state["initial_point"])
    if plan.candidate:
        # A long probe is approached through bounded intermediate steps.
        point = {}
        for key, target in plan.candidate.items():
            width = state["hard_bounds"][key][1] - state["hard_bounds"][key][0]
            step = settings.max_step_fraction * width
            if key in config.integer_var_paths:
                step = max(1, math.floor(step))
            point[key] = max(anchor[key] - step, min(anchor[key] + step, target))
        options.append(point)
    # Advisor-specified probes/baselines have priority; invalid probes are not silently replaced.
    if options:
        candidates = options[:1]
    else:
        # A collapsed integer/local dimension is supplied deterministically; optimizers need lo<hi.
        ask_region = {p: (pair if pair[0] < pair[1] else state["hard_bounds"][p]) for p, pair in local.items()}
        candidates = [session.ask(ask_region)]
        rng = random.Random((config.random_seed or 0) + state["used"] * 1009)
        candidates += [{p: rng.uniform(*pair) for p, pair in local.items()}
                       for _ in range(settings.candidate_pool_size - 1)]
    prepared = []
    for candidate in candidates:
        try:
            point, actual = _prepare_candidate(candidate, config, state["hard_bounds"])
            if not plan.candidate and state["observations"]:
                if any(not local[p][0] <= v <= local[p][1] for p, v in point.items()):
                    continue
            if fingerprint_design_vars(actual) in seen and not (plan.repeat and plan.candidate):
                continue
            prepared.append((point, actual))
        except ValueError:
            continue
    if not prepared:
        return None
    if config.feasibility_filter.enabled and not options:
        clf = FeasibilityClassifier(config.feasibility_filter)
        rows = _build_feasibility_rows([decode_case(r) for r in state["observations"]])
        features = feasibility_feature_names(list(config.param_bounds), config.derived_var_specs)
        clf.fit(rows, features)
        screened = clf.screen([{**values, "__index": i} for i, (_, values) in enumerate(prepared)], fallback_top_k=1)
        if screened:
            return prepared[int(screened[0]["__index"])]
    return prepared[0]


def _target_met(case, config, settings):
    if not settings.objective_targets or not _valid(case, config):
        return False
    for name, threshold in settings.objective_targets.items():
        obj = case.get_objective(name)
        if obj is None or not obj.available:
            return False
        if not (obj.value <= threshold if obj.minimize else obj.value >= threshold):
            return False
    return True


def _verification_candidate(state, config, settings):
    valid = [(r, decode_case(r)) for r in state["observations"]
             if "verification" not in r.get("tags", [])]
    valid = [(r, c) for r, c in valid if _valid(c, config)]
    if not valid:
        return None
    for row, case in valid:
        if _target_met(case, config, settings):
            return row
    # Representative compromise point; this does not claim verification of the whole front.
    vectors = [_extract_all_objectives(c, config) for _, c in valid]
    minima = [min(v[i] for v in vectors) for i in range(len(config.objective_names))]
    spans = [max(v[i] for v in vectors) - minima[i] for i in range(len(minima))]
    scores = [sum((v[i] - minima[i]) / max(spans[i], 1e-12) for i in range(len(v))) for v in vectors]
    return valid[min(range(len(scores)), key=scores.__getitem__)][0]


def optimize_agent_case(driver, config, *, settings: LoopConfig, checkpoint_path,
                        context: dict, knowledge: list[dict], resume=False, agent=None,
                        evaluator=None, session_factory=ParetoSearchSession) -> dict:
    """Returns a JSON-ready audit report. All retries and verification consume one total budget.

    evaluator(driver, actual_inputs, run_config, iteration, tags) can be injected for tests.
    A resumed run reconstructs the surrogate from observations, not a pickled LLM/COM object.
    """
    settings = LoopConfig.from_dict(asdict(settings))
    agent = agent or ProcessDecisionAgent()
    if evaluator is None:
        def evaluator(driver, actual_inputs, run_config, iteration, tags):
            return run_case(driver, actual_inputs, run_config, iteration=iteration, tags=tags,
                            run_id=str(uuid.uuid4()))
    if set(settings.objective_targets) - set(config.objective_names):
        raise ValueError("Unknown objective target")
    if not config.run_config.reinit:
        raise ValueError("Closed loop requires simulator.reinit=true as the reset baseline")
    hard = {p: list(pair) for p, pair in config.param_bounds.items()}
    region = getattr(config, "search_region", None) or hard
    for key, (lo, hi) in hard.items():
        if not finite(lo) < finite(hi):
            raise ValueError("Invalid engineering bounds")
    identity = {"context": context, "hard_bounds": hard, "search_region": region,
                "settings": asdict(settings), "knowledge": knowledge,
                "objectives": config.objective_names, "fixed": config.fixed_vars,
                "derived": config.derived_var_specs, "dependencies": config.var_dependencies,
                "integer_paths": sorted(config.integer_var_paths), "seed": config.random_seed}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with SessionJournal(checkpoint_path) as journal:
        state = journal.load()
        if state is not None and not resume:
            raise ValueError("Checkpoint exists; use --resume or a new checkpoint path")
        if resume and state is None:
            raise ValueError("No checkpoint exists to resume")
        if state is not None and (state.get("version") != 1 or state["fingerprint"] != fingerprint):
            raise ValueError("Configuration/model/knowledge changed; start a new checkpoint")
        if state is None:
            initial = {p: config.reference_values.get(p, (lo + hi) / 2) for p, (lo, hi) in hard.items()}
            for key, value in initial.items():
                if not hard[key][0] <= finite(value) <= hard[key][1]:
                    raise ValueError(f"Initial point outside engineering bounds: {key}")
            initial, _ = _prepare_candidate(initial, config, hard)
            state = {"version": 1, "fingerprint": fingerprint, "session_id": config.session_id,
                     "problem": identity, "hard_bounds": hard, "search_region": dict(region), "initial_point": initial,
                     "observations": [], "decisions": [], "used": 0, "batch_size": settings.batch_size,
                     "batch_remaining": 0, "stagnation_count": 0, "pending": None,
                     "status": "running", "phase": "search", "verification_ids": [],
                     "termination_reason": None, "hv_history": [], "rng": None}
            journal.save(state)
        config.session_id = state["session_id"]
        # Unknown outcome after crash is never treated as success or automatically replayed.
        if state["pending"]:
            pending = state["pending"]
            case = ProcessCase(status=CaseStatus.SIM_FAILED, iteration=pending["iteration"],
                               design_vars=pending["actual"], tags=pending["tags"],
                               notes="Interrupted evaluation: outcome unknown; budget consumed")
            state["observations"].append(encode_case(case, pending["x"], pending["initialization"]))
            if "verification" in pending["tags"]:
                state["verification_ids"].append(case.case_id)
            state["pending"] = None
            state["batch_remaining"] = 0
            journal.save(state)
        # Rebuild reporting projection after a crash between checkpoint commit and DB write.
        if config.db_path:
            from src.database.simulation_db import SimulationDB
            db = SimulationDB(config.db_path)
            try:
                db.save_cases([{**r, "session_id": state["session_id"]} for r in state["observations"]])
            finally:
                db.close()
        if state["status"] == "done":
            return state
        session = session_factory(config)
        for row in state["observations"]:
            session.tell(decode_case(row), row["optimizer_inputs"])
        session.restore_rng(state["rng"])
        live_success = False  # Aspen numerical state is never assumed restored from disk.

        while state["used"] < settings.max_evaluations:
            if state["phase"] == "search":
                last = decode_case(state["observations"][-1]) if state["observations"] else None
                reason = None
                if last and _target_met(last, config, settings):
                    reason = "target_observed"
                elif state["used"] >= settings.max_evaluations - settings.verification_repeats:
                    reason = "evaluation_budget"
                if not reason and state["batch_remaining"] <= 0:
                    _finish_action(state, config)
                    if state["stagnation_count"] >= settings.stagnation_batches:
                        reason = "stagnation"
                    elif len(state["decisions"]) >= settings.max_decisions:
                        reason = "decision_budget"
                    else:
                        plan = _choose_plan(state, config, settings, agent, context, knowledge)
                        journal.save(state)
                        _log.info("Agent decision %d [%s]: %s", len(state["decisions"]),
                                  state["decisions"][-1]["source"], plan.hypothesis)
                        if plan.action == "stop":
                            reason = "agent_stop"
                if not reason:
                    try:
                        picked = _pick_candidate(state, config, settings, session)
                    except ValueError as exc:
                        picked = None
                        state["decisions"][-1]["execution_rejection"] = str(exc)
                    if picked is None:
                        # Ask for a different region instead of repeatedly running duplicates.
                        state["decisions"][-1]["execution_rejection"] = "No valid unseen candidate within allowed step"
                        state["batch_remaining"] = 0
                        journal.save(state)
                        continue
                    point, actual = picked
                    tags = ["agent_closed_loop", state["decisions"][-1]["id"]]
                    initialization = state["active_plan"]["initialization"]
                    if initialization == "previous" and not live_success:
                        initialization = "reset"
                else:
                    _finish_action(state, config)
                    state["termination_reason"] = reason
                    chosen = _verification_candidate(state, config, settings)
                    if chosen is None:
                        state["status"] = "done"
                        state["result"] = "no_feasible_solution"
                        journal.save(state)
                        break
                    state["phase"] = "verification"
                    state["verification_reference"] = chosen["case_id"]
                    journal.save(state)
                    continue
            else:
                if len(state["verification_ids"]) >= settings.verification_repeats:
                    break
                reference = next(r for r in state["observations"] if r["case_id"] == state["verification_reference"])
                point, actual = _prepare_candidate(reference["optimizer_inputs"], config, hard)
                tags, initialization = ["agent_closed_loop", "verification"], "reset"

            # Reserve the evaluation durably BEFORE touching Aspen.
            pending = {"x": point, "actual": actual, "initialization": initialization,
                       "iteration": state["used"], "tags": tags}
            state["pending"] = pending
            state["used"] += 1
            state["rng"] = session.rng_state()
            journal.save(state)
            warmstart = config.run_config.recycle_warmstart
            if initialization == "reset" and warmstart is not None:
                # Reset means fixed configured estimates, never inherited estimates from a prior trial.
                warmstart = replace(warmstart, mode="fixed", _last_values={})
            run_config = replace(config.run_config, reinit=initialization == "reset", recycle_warmstart=warmstart)
            blocked = _preflight_blocked(actual, config)
            try:
                if blocked:
                    case = ProcessCase(status=CaseStatus.SIM_FAILED, iteration=pending["iteration"],
                                       design_vars=actual, tags=tags, notes=f"Preflight rejected: {blocked}")
                else:
                    case = evaluator(driver, actual, run_config, pending["iteration"], tags)
            except Exception as exc:
                case = ProcessCase(status=CaseStatus.SIM_FAILED, iteration=pending["iteration"],
                                   design_vars=actual, tags=tags, notes=f"Evaluation error: {type(exc).__name__}: {exc}")
            row = encode_case(case, point, initialization)
            state["observations"].append(row)
            state["pending"] = None
            state["batch_remaining"] -= 1
            if state["phase"] == "verification":
                state["verification_ids"].append(case.case_id)
            else:
                tail = state["observations"][-settings.failure_trigger:]
                if len(tail) >= settings.failure_trigger and all(not r["success"] for r in tail):
                    state["batch_remaining"] = 0
            live_success = case.simulation_valid
            session.tell(case, point)
            state["hv_history"].append(_metrics(state, config)["hypervolume"])
            journal.save(state)
            # Existing reporting DB is a projection; checkpoint remains authoritative on resume.
            if config.db_path:
                from src.database.simulation_db import SimulationDB
                db = SimulationDB(config.db_path)
                try:
                    db.save_case({**case.to_dict(), "session_id": state["session_id"]})
                finally:
                    db.close()
            if config.on_case_complete:
                try:
                    config.on_case_complete(case, state["used"] - 1, settings.max_evaluations)
                except Exception:
                    _log.exception("Progress callback failed")
            needed_recovery = driver is not None and getattr(driver, "needs_recovery", False)
            try:
                driver_available = driver is None or _maybe_recover_driver(driver, "agent_closed_loop")
            except Exception:
                _log.exception("Aspen recovery failed")
                driver_available = False
            if not driver_available:
                state["termination_reason"] = "driver_unavailable"
                state["result"] = "unverified"
                break
            if needed_recovery:
                live_success = False

        if state.get("verification_reference"):
            reference = decode_case(next(r for r in state["observations"] if r["case_id"] == state["verification_reference"]))
            checks = [decode_case(r) for r in state["observations"] if r["case_id"] in state["verification_ids"]]
            verified = len(checks) == settings.verification_repeats and all(
                _valid(c, config) and all(math.isclose(
                    c.get_objective(name).value, reference.get_objective(name).value,
                    rel_tol=settings.verification_rtol, abs_tol=settings.verification_atol,
                ) for name in config.objective_names) for c in checks)
            target_verified = verified and all(_target_met(c, config, settings) for c in checks)
            state["result"] = "target_verified" if target_verified else "completed_verified" if verified else "verification_failed"
        state.setdefault("result", "unverified")
        state["status"] = "done"
        _finish_action(state, config)
        journal.save(state)
        return state
