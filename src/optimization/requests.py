"""Translate explicit conversation choices into reviewable configuration overrides."""
from __future__ import annotations

from copy import deepcopy
import math

from src.agents.reasoning import snapshot


class OptimizationRequestError(ValueError):
    """A request needs a user decision or additional project evidence."""

    def __init__(self, message, *, field=None):
        super().__init__(message)
        self.field = field


def canonical_algorithm(value):
    aliases = {"ga": "GA", "genetic algorithm": "GA", "遗传算法": "GA",
               "nsga2": "NSGA-II", "nsga-ii": "NSGA-II", "nsga ii": "NSGA-II",
               "pso": "PSO", "粒子群": "PSO", "粒子群算法": "PSO", "bo": "BO",
               "bayesian": "BO", "贝叶斯优化": "BO", "parego": "ParEGO",
               "qehvi": "qEHVI", "nehvi": "NEHVI", "random": "random"}
    return aliases.get(str(value).strip().casefold(), str(value).strip())


def configured_algorithm(raw):
    options = raw.get("optimizer") or {}
    kind = options.get("type", "bayesian")
    if kind in {"pso", "nsga2", "ga"}:
        return {"pso": "PSO", "nsga2": "NSGA-II", "ga": "GA"}[kind]
    surrogate = options.get("surrogate_model", "GP")
    return surrogate if surrogate in {"qEHVI", "NEHVI", "random"} else "ParEGO" if kind == "pareto_bayesian" else "BO"


def prepare_request(analysis, context):
    """Use case facts for paths; never silently substitute an unavailable algorithm."""
    from src.agents.orchestration.models import validate_optimization_analysis
    analysis = validate_optimization_analysis(analysis)
    document = snapshot(context)
    raw = document.get("configuration") or {}
    selected = analysis.get("algorithm") or analysis.get("method")
    algorithm = canonical_algorithm(selected) if selected else configured_algorithm(raw)
    from .registry import capabilities
    cap = capabilities(algorithm)
    if cap["status"] == "not_connected":
        raise OptimizationRequestError(f"你选择的算法 {algorithm} 尚未接入，无法生成执行计划。请明确选择其他已接入算法，或先接入此算法；未执行 Aspen。")
    if not raw:
        raise OptimizationRequestError("缺少可用项目配置，请先提供 YAML 或通过 /project new 建立项目。")
    equipment = analysis.get("equipment")
    if equipment:
        active = [v for v in raw.get("design_variables", []) if v.get("type") != "fixed"]
        if not active or any(f"\\Blocks\\{equipment}\\".casefold() not in v.get("aspen_path", "").casefold() for v in active):
            raise OptimizationRequestError(f"当前可调变量范围与指定设备 {equipment} 不一致，请明确优化范围或调整项目变量；未执行 Aspen。")
    overrides, changes = {}, []
    original = configured_algorithm(raw)
    options = deepcopy(raw.get("optimizer") or {})
    if algorithm != original:
        seed = analysis.get("random_seed", options.get("random_seed"))
        if algorithm in {"PSO", "NSGA-II", "GA"}:
            options = {"type": {"PSO": "pso", "NSGA-II": "nsga2", "GA": "ga"}[algorithm], "backend": "pymoo",
                       "parameters": {}, "random_seed": seed}
            # BO policies cannot be applied to a population algorithm. Record
            # the disabled policies in the preview instead of inheriting them.
            for key in ("trust_region", "boundary_refine", "feasibility_filter", "feasibility_search",
                        "sensitivity_probe", "search_region", "agent_loop", "early_stopping", "preflight"):
                if raw.get(key):
                    overrides[key] = {"enabled": False}
                    changes.append(f"{key}: disabled for {algorithm}")
        else:
            multi = algorithm in {"ParEGO", "qEHVI", "NEHVI"} or algorithm == "random" and len(raw.get("objectives", [])) > 1
            previous = options if original not in {"PSO", "NSGA-II", "GA"} else {}
            options = {**previous, "type": "pareto_bayesian" if multi else "bayesian",
                       "surrogate_model": algorithm if algorithm in {"qEHVI", "NEHVI", "random"} else "GP",
                       "random_seed": seed}
        changes.insert(0, f"algorithm: {original} -> {algorithm} (user selection)")
    if "optimizer_parameters" in analysis:
        if algorithm in {"PSO", "NSGA-II", "GA"}:
            options["parameters"] = {**options.get("parameters", {}), **analysis["optimizer_parameters"]}
        else:
            allowed = {"n_initial_points", "n_iterations", "n_initial_min", "acquisition_function", "xi", "kappa", "hv_margin", "scalarization"}
            if set(analysis["optimizer_parameters"]) - allowed:
                raise OptimizationRequestError("所选算法含未知优化参数，请核对参数名称。")
            options.update(analysis["optimizer_parameters"])
    if "random_seed" in analysis:
        options["random_seed"] = analysis["random_seed"]
    overrides["optimizer"] = options
    overrides["execution"] = {**(raw.get("execution") or {}), **analysis.get("execution", {})}
    for name in ("max_evaluations", "max_failures", "max_seconds"):
        value = overrides["execution"].get(name)
        if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                or name != "max_seconds" and (type(value) is not int)):
            raise OptimizationRequestError(f"请补充有效的执行预算 {name}（次数为正整数，时限为正数）。", field=name)
    if algorithm in {"PSO", "NSGA-II", "GA"}:
        from .population import resolve_parameters
        try:
            params = resolve_parameters(algorithm, options.get("parameters", {}))
        except ValueError as exc:
            raise OptimizationRequestError(f"优化参数无效：{exc}") from None
        if params["pop_size"] > overrides["execution"]["max_evaluations"]:
            raise OptimizationRequestError(f"总评估预算不足一代种群（pop_size={params['pop_size']}），请补充预算或明确调小种群。")
        options["parameters"] = params
    target = analysis.get("target")
    if target is not None:
        metric = analysis.get("metric")
        if metric not in {"purity", "mass_fraction", "mole_fraction", "纯度"}:
            raise OptimizationRequestError("目标指标尚未映射到配置，请明确纯度指标及质量/摩尔基准。")
        try:
            from .targets import parse_purity_target
            threshold = parse_purity_target(target)
        except ValueError as exc:
            raise OptimizationRequestError(str(exc), field="target") from None
        component = analysis.get("component") or analysis.get("target_component")
        stream = analysis.get("target_stream")
        matches = []
        constraints = deepcopy(raw.get("constraints") or [])
        for index, con in enumerate(constraints):
            parts = con.get("aspen_path", "").split("\\")
            upper = [part.upper() for part in parts]
            if not any(p in upper for p in ("MASSFRAC", "MOLEFRAC")) or con.get("operator") != ">=":
                continue
            if component and parts[-1] != component:
                continue
            if stream and ("Streams" not in parts or parts[parts.index("Streams") + 1] != stream):
                continue
            if metric == "mass_fraction" and "MASSFRAC" not in upper or metric == "mole_fraction" and "MOLEFRAC" not in upper:
                continue
            matches.append(index)
        if len(matches) != 1:
            raise OptimizationRequestError("无法唯一匹配产品纯度下限，请明确产品物流、组分及质量/摩尔基准，或补充对应约束。")
        con = constraints[matches[0]]
        changes.append(f"{con['name']}: {con.get('threshold')} -> {threshold}")
        con["threshold"] = threshold
        overrides["constraints"] = constraints
        if algorithm == "GA":
            # The explicit user goal is the single GA objective. Other YAML
            # objectives remain in the source and their replacement is visible.
            overrides["objectives"] = [{"name": "requested_purity", "type": "aspen_path",
                "aspen_path": con["aspen_path"], "minimize": False, "unit": "1"}]
            changes.append("objectives: maximize requested product purity (GA user goal)")
        if algorithm in {"GA", "NSGA-II"}:
            for constraint in constraints:
                parts = constraint.get("aspen_path", "").upper().split("\\")
                if any(p in parts for p in ("MASSFRAC", "MOLEFRAC")):
                    constraint.setdefault("scale", 1.0)  # raw fraction residual, no rescaling
    count = len(overrides.get("objectives", raw.get("objectives")) or [])
    if cap["objectives"] == "single" and count != 1 or cap["objectives"] == "multi" and count < 2:
        raise OptimizationRequestError(f"所选 {algorithm} 与当前 {count} 个目标不兼容，请明确目标配置；算法选择已保留。")
    unsupported = [v.get("name", v.get("aspen_path")) for v in raw.get("design_variables", [])
                   if v.get("type", "continuous") != "fixed" and v.get("type", "continuous") not in cap["variable_types"]]
    if unsupported:
        raise OptimizationRequestError(f"所选 {algorithm} 当前不支持这些变量的类型：{', '.join(unsupported)}。请调整变量配置或明确选择其他算法。")
    request = {"algorithm": algorithm, "source": "conversation" if selected else "configuration",
               "overrides": overrides, "changes": changes,
               "evidence_policy": {"feasibility": "unknown until actual simulations",
                   "purity_basis": "use requested output path; do not compare mass and mole fractions",
                   "source_descriptions": "source comments may be outdated; effective numeric overrides apply",
                   "objective_constraint_overlap": "allowed: maximize purity subject to a minimum; GA uses no gradient"}}
    if algorithm in {"PSO", "NSGA-II", "GA"}:
        request["evaluation_schedule"] = {"pop_size": params["pop_size"], "generations": params["generations"],
            "generations_include_initial_population": True,
            "planned_evaluations": params["pop_size"] * params["generations"],
            "max_evaluations": overrides["execution"]["max_evaluations"],
            "partial_population": False, "seconds_per_evaluation": "unknown; enforced by shared deadline"}
    return request
