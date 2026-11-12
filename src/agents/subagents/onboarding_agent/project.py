"""Project onboarding in ConversationSession; static data only, no execution grant."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import difflib
import json
import math
from pathlib import Path
import tempfile
import uuid

from src.agents.answer_content import AnswerBlock
from src.agents.context import ProcessContextBuilder
from src.agents.context.versions import file_version
from src.agents.lead.continuation import provider_json, pointer
from src.agents.orchestration.models import Answer
from src.agents.reasoning import redact, snapshot
from src.models.tunable import ConfigDraft


FIELDS = {"product", "objectives", "constraints", "design_variables", "optimizer", "execution", "requirements"}
REQUIRED = FIELDS - {"requirements"}
ORIGINS = {"user", "model_fact", "history", "agent_inference", "system_default"}


def algorithm_catalogue():
    """Report actual existing backend dependencies without fitting or executing."""
    from src.optimization.surrogate import _HAS_SKOPT
    import importlib.util
    botorch = all(importlib.util.find_spec(n) is not None for n in ("torch", "botorch", "gpytorch"))
    return [{"id": n, "available": available, "dependency": dep,
             "status": "connected" if available else "unavailable"}
            for n, available, dep in [(n, _HAS_SKOPT, "scikit-optimize") for n in ("GP", "RF", "ET", "GBRT")]
            + [("random", True, "standard library"), ("qEHVI", botorch, "BoTorch/torch/gpytorch"),
               ("NEHVI", botorch, "BoTorch/torch/gpytorch"), ("PSO", False, "not implemented"), ("GA", False, "not implemented")]]


@dataclass
class ProjectDraft:
    project_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    version: int = 1
    revision: int = 0
    status: str = "draft"
    model: str | None = None
    config_source: str | None = None
    node_db: str | None = None
    simulation_db: str | None = None
    fields: dict = field(default_factory=dict)
    conflicts: list = field(default_factory=list)
    history: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    confirmed_digest: str | None = None
    saved_config: str | None = None
    missing: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def _secret(path):
    from src.agents.context.loader import ContextLoader
    # Match the attachment loader's scope; never read a secret as config/model/DB.
    p = Path(path)
    return (p.name.lower().startswith(".env") or p.name.lower() in {"credentials", "id_rsa", "id_ed25519"}
            or p.suffix.lower() in {".key", ".pem", ".p12"}
            or any(x.lower() in {".aws", ".ssh", ".codex"} for x in p.parts))


def _selected(path, suffixes):
    p = Path(path).expanduser().resolve()
    if _secret(p) or p.suffix.lower() not in suffixes or not p.is_file():
        raise ValueError("请选择存在且受支持的项目文件；密钥文件不可用。")
    return str(p)


class ProjectOnboarding:
    def __init__(self, conversation):
        self.conversation = conversation
        self.lead, self.state = conversation.lead, conversation.state

    @property
    def draft(self):
        data = self.state.runtime_state.get("project_draft")
        return ProjectDraft(**deepcopy(data)) if data else None

    def _store(self, draft):
        self.state.runtime_state["project_draft"] = draft.to_dict()

    def _clear(self):
        self.lead.invalidate_approval(self.state)
        self.state.context_refs.clear()
        self.state.context_summary.clear()
        self.state.context_sources.clear()
        self.state.task_results.clear()
        self.state.history.clear()
        self.state.pending_task = None
        self.state.runtime_state.clear()
        adapter = self.lead._case_adapters.pop(self.state.session_id, None)
        # Project onboarding always constructs its own static, selected builder.

    def start(self, model=None, *, config=None, node_db=None, simulation_db=None):
        selected_model = _selected(model, {".bkp", ".apw"}) if model else None
        selected_config = _selected(config, {".yaml", ".yml"}) if config else None
        if selected_config:
            from src.agents.context.adapter import ProcessCaseAdapter
            adapter = ProcessCaseAdapter.from_config(selected_config)
            resolved = adapter.resolve_model_resource()
            if not selected_model and resolved.path:
                selected_model = _selected(resolved.path, {".bkp", ".apw"})
        draft = ProjectDraft(model=selected_model, config_source=selected_config,
            node_db=_selected(node_db, {".db", ".sqlite", ".sqlite3"}) if node_db else None,
            simulation_db=_selected(simulation_db, {".db", ".sqlite", ".sqlite3"}) if simulation_db else None)
        self._clear()
        self._store(draft)
        evidence = self._refresh(draft)
        self._store(draft)
        # No rule based interview: a required LLM formulates the first relevant question.
        return self.turn("开始建立项目，请根据已选静态资料确认接下来需要澄清什么。", evidence=evidence)

    def _refresh(self, draft):
        for p in (draft.model, draft.config_source, draft.node_db, draft.simulation_db):
            if p and _secret(p):
                raise ValueError("project source cannot be a secret file")
        versions = {key: file_version(value) for key, value in {
            "model": draft.model, "config": draft.config_source, "node_db": draft.node_db,
            "simulation_db": draft.simulation_db}.items()}
        changed = bool(draft.sources and draft.sources != versions)
        if changed:
            self.lead.invalidate_approval(self.state)
            draft.confirmed_digest = None
            draft.status = "draft"
            for item in draft.fields.values():
                item["confirmed"] = False
            draft.history.append({"event": "sources_changed", "old": draft.sources, "new": versions})
            if not any(c["id"] == "source-change" for c in draft.conflicts):
                draft.conflicts.append({"id": "source-change", "message": "来源已变化，请核对当前草稿后明确解决此冲突。"})
        draft.sources = versions
        # Configuration is optional. Read-only builder never imports custom modules.
        with ProcessContextBuilder(yaml_path=draft.config_source, aspen_file=draft.model,
                node_db=draft.node_db if draft.node_db and Path(draft.node_db).exists() else None,
                simulation_db=draft.simulation_db if draft.simulation_db and Path(draft.simulation_db).exists() else None,
                read_only=True) as builder:
            context = snapshot(builder.build(session=self.state))
        draft.missing = sorted(REQUIRED - draft.fields.keys())
        self._store(draft)
        return context

    def turn(self, message, *, evidence=None):
        draft = self.draft
        if draft is None or draft.status == "cancelled":
            raise ValueError("当前没有项目草稿；使用 /project new。")
        context = evidence if evidence is not None else self._refresh(draft)
        contract = '''Interpret project requirements and corrections using the current message and selected static evidence.
Return exactly {"updates":{field:{"value":any,"origin":"user|model_fact|history|agent_inference|system_default","quote":string,"evidence_ref":string}},
"conflicts":[{"id":string,"message":string}],"resolved_conflicts":[string],"question":string}.
Only update fields explicitly changed by the user or supported by evidence; preserve unrelated fields.
Fields: product {name,stream,component}; objectives [{name,type:"aspen_path",aspen_path,minimize:boolean,metric,basis,unit,stream,component}];
constraints [{name,aspen_path,operator,threshold,metric,basis,unit,stream,component}];
design_variables [{name,aspen_path,type:"continuous|integer",lower_bound,upper_bound,unit,initial_value?}];
optimizer {type:"bayesian|pareto_bayesian",surrogate_model,n_initial_points,n_iterations,acquisition_function};
execution {max_evaluations,max_seconds,max_failures,timeout}; requirements [string].
An update replaces one complete field, never patches a list element. Carry forward unchanged list entries on correction.
User origin requires an exact substring quote from THIS message; model_fact/history require a JSON pointer into evidence.
Preserve description/threshold conflicts and mass/mole ambiguity; do not silently decide them.
Never invent Aspen paths, safe bounds, budget, algorithm selection, units, cost semantics or business requirements.
Use human-readable node names with their evidence-backed internal paths. Never infer that historical ranges are authorized bounds.
Model facts, historical data, defaults and inferences are not confirmed user requirements.
No fake Agent questionnaire if required evidence is missing: ask the highest priority blocking question.
BKP static index is partial; APW has no static body parser. No Aspen connection is available here.
Resolve an existing conflict only when the current user message explicitly addresses it.
question: concise current guidance or next clarification, using no raw schema/JSON/YAML.'''
        data = provider_json(self.lead, "project_onboarding", {"message": message, "draft": draft.to_dict(),
            "evidence": context, "algorithms": algorithm_catalogue()}, contract)
        if (set(data) != {"updates", "conflicts", "resolved_conflicts", "question"}
                or not isinstance(data["updates"], dict) or set(data["updates"]) - FIELDS
                or not isinstance(data["question"], str) or not data["question"].strip()
                or not isinstance(data["conflicts"], list) or not isinstance(data["resolved_conflicts"], list)):
            raise ValueError("invalid project Agent protocol")
        working = deepcopy(draft)
        for key, update in data["updates"].items():
            if not isinstance(update, dict) or set(update) != {"value", "origin", "quote", "evidence_ref"} or update["origin"] not in ORIGINS:
                raise ValueError("invalid requirement provenance")
            if update["origin"] == "user" and (not update["quote"] or update["quote"] not in message):
                raise ValueError("requirement is not grounded in current user statement")
            if update["origin"] in {"model_fact", "history"}:
                pointer(context, update["evidence_ref"])
            # Aspen paths must actually occur in selected evidence or in the user's statement.
            def paths(v):
                if isinstance(v, dict):
                    for k, child in v.items():
                        if k == "aspen_path":
                            yield child
                        else:
                            yield from paths(child)
                elif isinstance(v, list):
                    for child in v:
                        yield from paths(child)
            def strings(v):
                if isinstance(v, str):
                    yield v
                elif isinstance(v, dict):
                    for child in v.values():
                        yield from strings(child)
                elif isinstance(v, list):
                    for child in v:
                        yield from strings(child)
            known = set(strings(context))
            # Retain previously grounded paths when correcting another field.
            known.update(strings(draft.fields))
            if any(not isinstance(p, str) or p not in known and p not in message for p in paths(update["value"])):
                raise ValueError("unverified Aspen path; a selected static node source is required")
            working.history.append({"event": "field_changed", "field": key, "previous": working.fields.get(key),
                                    "statement": message, "revision": working.revision + 1})
            working.fields[key] = {**deepcopy(update), "confirmed": False, "revision": working.revision + 1}
        if not all(isinstance(c, dict) and set(c) == {"id", "message"} and all(isinstance(v, str) and v for v in c.values()) for c in data["conflicts"]):
            raise ValueError("invalid conflicts")
        if not all(isinstance(x, str) for x in data["resolved_conflicts"]):
            raise ValueError("invalid conflict resolution")
        conflicts = {c["id"]: c for c in working.conflicts}
        for cid in data["resolved_conflicts"]:
            if cid not in conflicts:
                raise ValueError("unknown conflict resolution")
            working.history.append({"event": "conflict_resolved", "conflict": conflicts.pop(cid), "statement": message})
        conflicts.update({c["id"]: c for c in data["conflicts"]})
        working.conflicts = list(conflicts.values())
        if data["updates"] or data["conflicts"] or data["resolved_conflicts"]:
            self.lead.invalidate_approval(self.state)
            working.revision += 1
            working.status, working.confirmed_digest = "draft", None
        working.missing = self.issues(working)
        self._store(working)
        self.state.audit_events.append({"event": "project_draft_updated", "project_id": working.project_id,
            "revision": working.revision, "fields": list(data["updates"]), "conflicts": working.conflicts})
        return self.summary(data["question"])

    def issues(self, draft):
        issues = ["缺少：" + key for key in sorted(REQUIRED - draft.fields.keys())]
        if not draft.model or not Path(draft.model).is_file():
            issues.append("请选择存在的模型文件")
        if draft.conflicts:
            issues += ["冲突：" + c["message"] for c in draft.conflicts]
        if not issues:
            try:
                self.configuration(draft)
            except (ValueError, KeyError, TypeError) as exc:
                issues.append(str(exc))
        return issues

    def configuration(self, draft):
        """No semantic defaults for execution-critical fields; no custom code imports."""
        values = {k: v["value"] for k, v in draft.fields.items()}
        if REQUIRED - values.keys():
            raise ValueError("关键配置仍有缺失")
        product = values["product"]
        if not isinstance(product, dict) or any(not isinstance(product.get(k), str) or not product[k].strip() for k in ("name", "stream", "component")):
            raise ValueError("请明确产品、物流与组分")
        variables, objectives, constraints = (values[k] for k in ("design_variables", "objectives", "constraints"))
        if not isinstance(variables, list) or not variables or not isinstance(objectives, list) or not objectives or not isinstance(constraints, list):
            raise ValueError("需要可调参数、目标和明确约束列表（无约束请明确说明）")
        from src.agents.actions.validation import _SAFE_PATH
        for var in variables:
            for key in ("name", "aspen_path", "type", "lower_bound", "upper_bound", "unit"):
                if key not in var or var[key] is None:
                    raise ValueError("参数缺少名称、路径、类型、单位或硬边界")
            lo, hi = var["lower_bound"], var["upper_bound"]
            if var["type"] not in {"continuous", "integer"}:
                raise ValueError("项目建立仅支持 continuous/integer；离散和派生变量尚未接入")
            if not _finite(lo) or not _finite(hi) or lo >= hi:
                raise ValueError("参数硬边界必须为有限递增数值；不会编造安全范围")
            if var["type"] == "integer" and any(int(v) != v for v in (lo, hi)):
                raise ValueError("整数边界不得自动取整")
            if "initial_value" in var and (not _finite(var["initial_value"]) or not lo <= var["initial_value"] <= hi
                    or var["type"] == "integer" and int(var["initial_value"]) != var["initial_value"]):
                raise ValueError("初值必须在硬边界内并满足整数类型")
            if not _SAFE_PATH.match(var["aspen_path"]) or "\\input\\" not in var["aspen_path"].lower():
                raise ValueError("参数需要稳定 Aspen Input 路径")
            if not isinstance(var["unit"], str) or not var["unit"]:
                raise ValueError("参数单位必须明确，无量纲可使用 1")
        for group in (variables, objectives, constraints):
            names = [x.get("name") for x in group]
            if any(not isinstance(n, str) or not n for n in names) or len(names) != len(set(names)):
                raise ValueError("同类参数/指标名称必须非空且唯一")
        if len({v["aspen_path"] for v in variables}) != len(variables):
            raise ValueError("参数路径重复")
        for item in objectives + constraints:
            for key in ("aspen_path", "metric", "basis", "unit", "stream", "component"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError("指标需要路径、计算含义、基准、单位、物流和组分；不适用请明确写明")
            if not _SAFE_PATH.match(item["aspen_path"]) or "\\output\\" not in item["aspen_path"].lower():
                raise ValueError("指标需要已知 Aspen Output 路径")
            if item["metric"] in {"purity", "mass_fraction", "mole_fraction"}:
                if item["basis"] not in {"mass", "mole"} or item["unit"] not in {"1", "fraction"}:
                    raise ValueError("纯度必须明确 mass/mole 基准及分数单位；不做隐式换算")
                upper_path = item["aspen_path"].upper()
                if ("MASSFRAC" in upper_path and item["basis"] != "mass"
                        or "MOLEFRAC" in upper_path and item["basis"] != "mole"
                        or item["metric"] == "mass_fraction" and item["basis"] != "mass"
                        or item["metric"] == "mole_fraction" and item["basis"] != "mole"):
                    raise ValueError("指标路径与质量/摩尔基准冲突")
        for obj in objectives:
            if obj.get("type") != "aspen_path" or type(obj.get("minimize")) is not bool:
                raise ValueError("本入口仅支持语义明确的 aspen_path 目标；成本计算模型尚未接入")
        for con in constraints:
            if con.get("operator") not in {"<=", "<", ">=", ">", "=="} or not _finite(con.get("threshold")):
                raise ValueError("约束运算符或阈值缺失/无效")
            if con["metric"] in {"purity", "mass_fraction", "mole_fraction"} and not 0 <= con["threshold"] <= 1:
                raise ValueError("分数阈值必须在 0 到 1 之间")
        opt, limits = values["optimizer"], deepcopy(values["execution"])
        if set(opt) != {"type", "surrogate_model", "n_initial_points", "n_iterations", "acquisition_function"}:
            raise ValueError("算法、初始点、迭代数及采集函数必须明确选择")
        algorithms = {x["id"] for x in algorithm_catalogue() if x["available"]}
        if opt["surrogate_model"] not in algorithms or opt["type"] not in {"bayesian", "pareto_bayesian"}:
            raise ValueError("所选算法未实现或依赖不可用；不会替换算法")
        if (opt["type"] == "bayesian" and len(objectives) != 1 or opt["type"] == "pareto_bayesian" and len(objectives) < 2
                or opt["surrogate_model"] in {"qEHVI", "NEHVI"} and opt["type"] != "pareto_bayesian"):
            raise ValueError("算法与目标数量不兼容")
        if (type(opt["n_initial_points"]) is not int or opt["n_initial_points"] < 1
                or type(opt["n_iterations"]) is not int or opt["n_iterations"] < 0
                or opt["acquisition_function"] not in {"EI", "UCB", "PI"}):
            raise ValueError("算法采样数或采集函数无效")
        if set(limits) != {"max_evaluations", "max_seconds", "max_failures", "timeout"}:
            raise ValueError("需要总评估次数、总时限、最大失败数和单次超时")
        for key in ("max_evaluations", "max_failures"):
            if type(limits[key]) is not int or limits[key] < 1:
                raise ValueError("总评估预算及失败限额必须是正整数")
        if any(not _finite(limits[k]) or limits[k] <= 0 for k in ("max_seconds", "timeout")):
            raise ValueError("执行时限必须是有限正数")
        if opt["n_initial_points"] + opt["n_iterations"] > limits["max_evaluations"]:
            raise ValueError("DOE 加优化采样超过总预算；总预算还包括探测、重试与验证")
        cfg = ConfigDraft(aspen_file=draft.model, design_variables=deepcopy(variables),
            objectives=deepcopy(objectives), constraints=deepcopy(constraints), optimizer=deepcopy(opt)).to_yaml_dict()
        cfg["simulator"]["timeout"] = limits.pop("timeout")
        cfg["execution"] = limits
        cfg["project"] = {"id": draft.project_id, "revision": draft.revision, "product": deepcopy(product),
                          "requirements": deepcopy(values.get("requirements", []))}
        return cfg

    def validate(self, draft):
        if self.issues(draft):
            raise ValueError("；".join(self.issues(draft)))
        import yaml
        cfg = self.configuration(draft)
        # Actual parser and first-batch plan semantic checks, without preflight/COM.
        from src.agents.actions.plans import build_optimization_plan, preflight_optimizer
        from src.utils.file_io import load_optimize_config
        with tempfile.TemporaryDirectory(prefix="pao-project-validate-") as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            parsed, _, _ = load_optimize_config(path)
            build_optimization_plan(path)
            preflight_optimizer(parsed)
        return cfg

    def summary(self, text="当前项目草稿。"):
        draft = self.draft
        if draft is None:
            return Answer("当前没有项目草稿。")
        issues = self.issues(draft)
        rows = []
        labels = {"product": "产品", "objectives": "目标", "constraints": "约束", "design_variables": "可调参数",
                  "optimizer": "算法", "execution": "总预算与时限", "requirements": "业务要求"}
        for name, entry in draft.fields.items():
            value = entry["value"]
            if name == "product" and isinstance(value, dict):
                human = " / ".join(str(value.get(k, "待明确")) for k in ("name", "stream", "component"))
            elif name in {"objectives", "constraints", "design_variables"} and isinstance(value, list):
                lines = []
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    if name == "design_variables":
                        lines.append(f"{item.get('name', '?')}: {item.get('lower_bound', '?')}～{item.get('upper_bound', '?')} {item.get('unit', '?')} ({item.get('type', '?')})")
                    else:
                        goal = ("最小化" if item.get("minimize") else "最大化") if name == "objectives" else f"{item.get('operator', '?')} {item.get('threshold', '?')}"
                        lines.append(f"{item.get('name', '?')}: {goal}；{item.get('basis', '?')}；{item.get('unit', '?')}；物流 {item.get('stream', '?')} / {item.get('component', '?')}")
                human = "；".join(lines) or "明确无约束"
            elif name == "optimizer" and isinstance(value, dict):
                human = f"{value.get('type', '待选择')} / {value.get('surrogate_model', '待选择')}；DOE {value.get('n_initial_points', '?')}，优化 {value.get('n_iterations', '?')}"
            elif name == "execution" and isinstance(value, dict):
                human = f"总评估 {value.get('max_evaluations', '?')} 次；总时限 {value.get('max_seconds', '?')} s；失败上限 {value.get('max_failures', '?')}；单次 {value.get('timeout', '?')} s"
            else:
                human = "；".join(map(str, value)) if isinstance(value, list) else str(value)
            rows.append([labels.get(name, name), human, entry["origin"], "已确认" if entry.get("confirmed") else "待确认"])
        blocks = [AnswerBlock("paragraph", text=f"模型：{Path(draft.model).name if draft.model else '尚未选择'}；草稿版本 {draft.revision}；状态 {draft.status}。")]
        if rows:
            blocks.append(AnswerBlock("table", columns=["项目", "内容", "来源", "确认"], rows=rows))
        if issues:
            blocks.append(AnswerBlock("bullets", title="待解决", items=issues))
        blocks.append(AnswerBlock("next_steps", items=["直接描述修正；/project confirm 确认业务配置；/project save 路径.yaml 保存配置；/project checkpoint 路径.json 保存草稿。",
            "配置确认和保存不会授权或运行 Aspen。缺少静态节点资料时需另行建立受审核读取计划，本入口不会连接 COM。 "]))
        return Answer(text, blocks=blocks, details={"project_draft": draft.to_dict()})

    def confirm(self):
        from src.agents.actions.authorization import digest
        draft = self.draft
        self._refresh(draft)
        self.validate(draft)
        for entry in draft.fields.values():
            entry["confirmed"] = True
        draft.confirmed_digest = digest(self.configuration(draft))
        draft.status = "confirmed"
        draft.history.append({"event": "business_configuration_confirmed", "revision": draft.revision})
        self._store(draft)
        return self.summary("业务配置已确认，可以保存；执行仍需具体计划及 Gateway 审批。")

    def save_config(self, destination):
        from src.agents.actions.authorization import digest
        import yaml
        draft = self.draft
        self._refresh(draft)
        cfg = self.validate(draft)
        if draft.status != "confirmed" or digest(cfg) != draft.confirmed_digest:
            raise ValueError("请先查看并确认当前业务配置")
        path = Path(destination).expanduser().resolve()
        if path.suffix.lower() not in {".yaml", ".yml"} or _secret(path):
            raise ValueError("保存配置需要普通 YAML 路径")
        payload = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        if path.exists():
            difference = "\n".join(difflib.unified_diff(path.read_text(encoding="utf-8").splitlines(), payload.splitlines(),
                fromfile=str(path), tofile="新草稿"))
            return Answer("目标已存在，原文件已保护。请改用新文件名保存。",
                blocks=[AnswerBlock("paragraph", text=difference or "配置内容相同。", detail=True)])
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects original content, including a competing writer.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        audit_path = path.with_name(path.name + ".project.json")
        draft.saved_config = str(path)
        self._store(draft)
        self.checkpoint(audit_path)
        return self.summary(f"配置已保存：{path}\n草稿及来源审计：{audit_path}；未运行 Aspen。")

    def checkpoint(self, destination):
        path = Path(destination).expanduser().resolve()
        if _secret(path) or path.suffix.lower() != ".json":
            raise ValueError("草稿保存需要普通 JSON 路径")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": "pao_project_draft", "version": 1, "draft": self.draft.to_dict()}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(redact(payload, self.lead.llm_config), stream, ensure_ascii=False, indent=2, allow_nan=False)
        return Answer(f"项目草稿已保存：{path}。恢复不会继承执行授权。")

    def restore(self, source):
        path = _selected(source, {".json"})
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(data) != {"kind", "version", "draft"} or data["kind"] != "pao_project_draft" or data["version"] != 1:
            raise ValueError("不支持的项目草稿格式")
        draft = ProjectDraft(**data["draft"])
        if draft.version != 1 or set(draft.fields) - FIELDS:
            raise ValueError("不支持的草稿版本或字段")
        self._clear()
        draft.confirmed_digest, draft.status = None, "draft"
        for entry in draft.fields.values():
            entry["confirmed"] = False
        self._store(draft)
        self._refresh(draft)
        self._store(draft)
        return self.summary("已恢复草稿；请核对来源和待解决项，再确认业务配置。未恢复任何执行授权。")

    def command(self, text):
        command, _, arg = text.strip().partition(" ")
        if command in {"show", "preview", "back"}:
            draft = self.draft
            if draft:
                self._refresh(draft)
            return self.summary()
        if command == "new":
            return self.start(arg.strip().strip('"') or None)
        if command == "model":
            # Switching a model starts a fresh scope, without old facts or authorization.
            return self.start(arg.strip().strip('"'))
        if command == "cancel":
            draft = self.draft
            self._clear()
            if draft:
                draft.status, draft.confirmed_digest = "cancelled", None
                self._store(draft)
            return Answer("项目建立已取消；相关执行计划与会话证据已清除。")
        if command == "confirm":
            return self.confirm()
        if command == "save":
            return self.save_config(arg.strip().strip('"'))
        if command == "checkpoint":
            return self.checkpoint(arg.strip().strip('"'))
        if command == "resume":
            return self.restore(arg.strip().strip('"'))
        if command == "algorithms":
            rows = [[x["id"], "可选择" if x["available"] else "不可用", x["dependency"]] for x in algorithm_catalogue()]
            return Answer("当前算法接线与依赖状态。", blocks=[AnswerBlock("table", columns=["算法", "状态", "依赖"], rows=rows)])
        raise ValueError("项目命令：new、model、show、preview、back、confirm、save、checkpoint、resume、cancel、algorithms")
