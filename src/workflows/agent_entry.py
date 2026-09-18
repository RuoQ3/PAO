"""Common configuration and result adapter for CLI, tools, and LangGraph."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from src.agents.closed_loop.advisor import load_knowledge
from src.agents.closed_loop.contracts import LoopConfig, finite


def _resolve_file(raw, base):
    path = Path(raw)
    if path.is_absolute():
        return path
    if (base / path).exists():
        return (base / path).resolve()
    return path.resolve()


def configure_agent(config, raw: dict, yaml_path: Path, sim_filepath: Path):
    config.agent_loop = dict(raw.get("agent_loop") or {})
    config.agent_resume = bool(config.agent_loop.get("resume", False))
    if config.agent_loop.get("checkpoint_path"):
        config.agent_checkpoint_path = str((yaml_path.parent / Path(config.agent_loop["checkpoint_path"])).resolve())
    names = {dv.get("name", dv.get("aspen_path")): (dv["name"] if dv.get("type") == "derived" else dv.get("aspen_path"))
             for dv in raw.get("design_variables", [])}
    config.search_region = dict(config.param_bounds)
    for name, pair in (raw.get("search_region") or {}).items():
        path = names.get(name, name)
        if path not in config.param_bounds or len(pair) != 2:
            raise ValueError(f"Unknown search-region variable: {name}")
        config.search_region[path] = tuple(map(finite, pair))
    # Only domain/algorithm configuration is put in the agent context. Never LLM credentials.
    keys = ("simulator", "design_variables", "objectives", "constraints", "optimizer",
            "extraction", "preflight", "feasibility_filter", "trust_region",
            "sensitivity_probe", "boundary_refine", "early_stopping")
    config.agent_context = {key: raw[key] for key in keys if key in raw}
    # Session IDs are routing metadata, not a change to the optimization problem.
    if "optimizer" in config.agent_context:
        config.agent_context["optimizer"] = dict(config.agent_context["optimizer"])
        config.agent_context["optimizer"].pop("session_id", None)
    config.agent_context["process_facts"] = config.agent_loop.get("process_facts", {})
    config.agent_context["model_filepath"] = str(sim_filepath)
    # Model fingerprint is checked before execution, including on resume.
    config.agent_context["model_sha256"] = hashlib.sha256(sim_filepath.read_bytes()).hexdigest() if sim_filepath.exists() else None
    modules = {}
    for obj in raw.get("objectives", []):
        for key in ("module_path", "module"):
            if obj.get(key):
                candidate = _resolve_file(obj[key], yaml_path.parent)
                if not candidate.is_file() and key == "module":
                    candidate = _resolve_file(str(obj[key]).replace(".", "/") + ".py", yaml_path.parent)
                if candidate.is_file():
                    modules[str(candidate)] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    config.agent_context["objective_module_hashes"] = modules
    default = Path(__file__).resolve().parents[2] / "configs/process_knowledge/distillation.yaml"
    files = config.agent_loop.get("knowledge_files", [str(default)])
    config.agent_knowledge = load_knowledge([str(_resolve_file(f, yaml_path.parent)) for f in files])


def validate_agent_config(config):
    from src.workflows.optimize_pareto_case import _validate_config
    _validate_config(config)
    settings = LoopConfig.from_dict({"max_evaluations": config.n_iterations, **config.agent_loop})
    if not config.run_config.reinit:
        raise ValueError("Agent loop requires simulator.reinit=true")
    if set(settings.objective_targets) - set(config.objective_names):
        raise ValueError("Unknown objective_targets name")
    for path, (lo, hi) in config.param_bounds.items():
        if not finite(lo) < finite(hi):
            raise ValueError("Non-finite or empty engineering bounds")
        left, right = config.search_region.get(path, (lo, hi))
        if not lo <= finite(left) <= finite(right) <= hi:
            raise ValueError(f"search_region must stay inside hard bounds: {path}")
        if path in config.integer_var_paths and math.ceil(left) > math.floor(right):
            raise ValueError("Search region has no integer candidate")
    return settings


def run_configured_agent(driver, config):
    from src.agents.closed_loop.journal import decode_case
    from src.agents.closed_loop.advisor import ProcessDecisionAgent
    from src.optimization.pareto import compute_pareto, _restore_reference_point
    from src.workflows.agent_optimize import optimize_agent_case
    from src.workflows.optimize_pareto_case import ParetoOptimizeResult
    from src.models.process_case import CaseStatus

    settings = validate_agent_config(config)
    checkpoint = config.agent_checkpoint_path or str(
        Path(config.db_path or "output/simulation.db").with_name("agent_checkpoint.db"))
    start = time.monotonic()
    state = optimize_agent_case(driver, config, settings=settings, checkpoint_path=checkpoint,
                                context=config.agent_context, knowledge=config.agent_knowledge,
                                resume=config.agent_resume,
                                agent=ProcessDecisionAgent(config.agent_llm_config))
    cases = [decode_case(row) for row in state["observations"]]
    reference = state.get("hv_reference")
    sample = next((c for c in cases if c.success), None)
    reference_raw = _restore_reference_point(reference, sample, config.objective_names) if reference and sample else None
    pareto = compute_pareto(cases, config.objective_names, reference_point=reference_raw, compute_hv=True)
    # Human-readable report and machine-readable audit share the same checkpoint identity.
    report_path = Path(checkpoint).with_suffix(".report.json")
    report_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# PAO 自主实验闭环报告", "", f"会话：{state['session_id']}",
             f"结果：{state['result']}", f"停止原因：{state['termination_reason']}",
             f"预算使用：{state['used']}/{settings.max_evaluations}",
             "", "复验只针对选定代表工况，不代表整个 Pareto 前沿或全局最优。", ""]
    for i, decision in enumerate(state["decisions"], 1):
        lines += [f"## 决策 {i}（{decision['source']}）", decision["plan"]["hypothesis"],
                  "", "```json", json.dumps(decision, ensure_ascii=False, indent=2), "```", ""]
    Path(checkpoint).with_suffix(".report.md").write_text("\n".join(lines), encoding="utf-8")
    return ParetoOptimizeResult(
        cases=cases, pareto_result=pareto, param_bounds=config.param_bounds,
        fixed_vars=config.fixed_vars, objective_names=config.objective_names,
        n_total=state["used"], n_success=sum(c.success for c in cases),
        n_sim_failed=sum(c.status == CaseStatus.SIM_FAILED for c in cases),
        n_objective_error=sum(c.status == CaseStatus.OBJECTIVE_ERROR for c in cases),
        n_initial=1, elapsed=time.monotonic() - start, hv_history=state["hv_history"],
        hv_reference_point=reference, session_id=state["session_id"],
        early_stopped=True, early_stop_reason=f"{state['termination_reason']}:{state['result']}",
        completed_iterations=state["used"], no_improvement_count=state["stagnation_count"],
    )
