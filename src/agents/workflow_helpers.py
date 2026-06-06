"""
workflow_helpers.py — run_demo_case_workflow 的配置解析辅助层。

只做纯 Python 的配置读取与路径解析，把 case_config_path 的关键信息
填入 DemoWorkflowState。

禁止导入：
  AspenDriver、SimulationRunner、SimulationDB、NodeDB、
  src.aspen_driver、src.database、src.workflows、LangGraph
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.state import DemoWorkflowState

# 项目根目录（src/agents/workflow_helpers.py → 上三层）
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# 1. 路径解析
# ---------------------------------------------------------------------------

def _resolve_config_path(case_config_path: str) -> str:
    """将配置路径解析为绝对路径字符串。

    解析优先级：
    1. 绝对路径直接使用
    2. 相对于当前工作目录
    3. 相对于项目根目录（src/agents/workflow_helpers.py 往上三层）

    Returns:
        已验证存在的绝对路径字符串。

    Raises:
        FileNotFoundError: 所有候选路径均不存在时抛出，附带已尝试的路径列表。
    """
    p = Path(case_config_path)

    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在：{p}")
        return str(p.resolve())

    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return str(from_cwd.resolve())

    from_root = _PROJECT_ROOT / p
    if from_root.exists():
        return str(from_root.resolve())

    raise FileNotFoundError(
        f"配置文件不存在：{case_config_path!r}\n"
        f"  已尝试：\n"
        f"    {from_cwd}\n"
        f"    {from_root}"
    )


# ---------------------------------------------------------------------------
# 2. YAML 读取
# ---------------------------------------------------------------------------

def _load_case_config(resolved_config_path: str) -> dict[str, Any]:
    """读取 YAML 文件，返回顶层字典。

    空文件返回空 dict（`yaml.safe_load` 对空文件返回 None，此处规范化为 {}）。
    YAML 语法错误直接向上抛出，不捕获不伪装。

    Args:
        resolved_config_path: 已经过 _resolve_config_path 验证的绝对路径字符串。

    Returns:
        YAML 顶层字典，文件为空时返回 {}。

    Raises:
        yaml.YAMLError: YAML 语法错误时抛出。
        ImportError: PyYAML 未安装时抛出。
    """
    import yaml  # 项目已有依赖，requirements.txt 中已列出

    with open(resolved_config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML 根节点应为字典，实际类型为 {type(data).__name__}。"
            f"路径：{resolved_config_path}"
        )
    return data


# ---------------------------------------------------------------------------
# 3. 提取 workflow 所需控制字段
# ---------------------------------------------------------------------------

def _extract_workflow_config(
    config: dict[str, Any],
    resolved_config_path: str,
) -> dict[str, Any]:
    """从 YAML 字典中提取 workflow 控制层所需的字段。

    Args:
        config:               _load_case_config 返回的顶层 dict。
        resolved_config_path: 配置文件的绝对路径字符串，用于推断目录相关路径。

    Returns:
        包含以下键的字典：
          optimizer_type  (str)       — 来自 config["optimizer"]["type"]，缺失为 ""
          objective_names (list[str]) — 来自顶层 config["objectives"][*]["name"]，
                                        缺失或格式不对为 []
          db_path         (str|None)  — pareto_bayesian 时推断为
                                        {config_dir}/output/simulation.db，否则 None
          node_db_path    (str)       — 优先 extraction.catalog_db（相对路径按
                                        config_dir 解析），缺失时默认
                                        {config_dir}/output/node.db

    不检查 db_path / node_db_path 是否实际存在。
    """
    config_dir = Path(resolved_config_path).parent

    # ── optimizer_type ──────────────────────────────────────────────────────
    optimizer_type: str = ""
    try:
        optimizer_type = str(config["optimizer"]["type"])
    except (KeyError, TypeError):
        pass

    # ── objective_names — 顶层 objectives 列表 ─────────────────────────────
    objective_names: list[str] = []
    try:
        objs = config["objectives"]
        if isinstance(objs, list):
            for item in objs:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    name = item["name"].strip()
                    if name:
                        objective_names.append(name)
    except (KeyError, TypeError):
        pass

    # ── db_path ─────────────────────────────────────────────────────────────
    db_path: str | None = None
    if optimizer_type == "pareto_bayesian":
        db_path = str(config_dir / "output" / "simulation.db")

    # ── node_db_path ─────────────────────────────────────────────────────────
    # 优先 extraction.catalog_db。相对路径解析规则：
    #   1. 绝对路径直接使用；
    #   2. 相对路径依次尝试 config_dir / p 和 PROJECT_ROOT / p；
    #      有一个存在就用存在的（PROJECT_ROOT 版匹配时优先，因为 catalog_db
    #      通常以项目根为基准写入，如 cases/demo_case/output/node.db）；
    #      两个都不存在则回退到 config_dir / p（保持"配置目录相对"默认语义）。
    # 缺失时默认 config_dir/output/node.db。
    node_db_path: str = str(config_dir / "output" / "node.db")
    try:
        catalog_db_raw = config["extraction"]["catalog_db"]
        if isinstance(catalog_db_raw, str) and catalog_db_raw.strip():
            p = Path(catalog_db_raw.strip())
            if p.is_absolute():
                node_db_path = str(p)
            else:
                from_config = config_dir / p
                from_root = _PROJECT_ROOT / p
                if from_root.exists():
                    node_db_path = str(from_root)
                elif from_config.exists():
                    node_db_path = str(from_config)
                else:
                    node_db_path = str(from_config)
    except (KeyError, TypeError):
        pass

    return {
        "optimizer_type": optimizer_type,
        "objective_names": objective_names,
        "db_path": db_path,
        "node_db_path": node_db_path,
    }


# ---------------------------------------------------------------------------
# 4. 状态初始化
# ---------------------------------------------------------------------------

def prepare_demo_workflow_state(case_config_path: str) -> DemoWorkflowState:
    """解析配置路径并创建已填充关键字段的 DemoWorkflowState。

    只负责"准备 state"，不调用任何 tool，不 add_step，不生成报告。

    Args:
        case_config_path: 用户传入的配置路径（相对或绝对均可）。

    Returns:
        已填充以下字段的 DemoWorkflowState：
          case_config_path、resolved_config_path、optimizer_type、
          objective_names、db_path、node_db_path。

    Raises:
        FileNotFoundError: 配置文件不存在时。
        yaml.YAMLError:    YAML 语法错误时。
        ValueError:        YAML 根节点非字典时。
    """
    resolved = _resolve_config_path(case_config_path)
    config = _load_case_config(resolved)
    extracted = _extract_workflow_config(config, resolved)

    state = DemoWorkflowState(case_config_path=case_config_path)
    state.resolved_config_path = resolved
    state.optimizer_type = extracted["optimizer_type"]
    state.objective_names = extracted["objective_names"]
    state.db_path = extracted["db_path"]
    state.node_db_path = extracted["node_db_path"]
    return state
