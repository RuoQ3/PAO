"""
_common.py — tools 子包共享工具。

提供所有工具都需要的：
  - YAML 文件读取（_load_yaml_raw）
  - 配置路径解析（_resolve_config_path）
  - 可替换的运行时依赖引用（_load_optimize_config、_AspenDriver 等）
  - 运行时依赖按需导入函数（_import_run_time_deps、_import_pareto_deps）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML 工具
# ---------------------------------------------------------------------------

def _load_yaml_raw(yaml_path: Path) -> dict[str, Any]:
    """读取 YAML 文件，返回原始字典。"""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML 未安装，请执行：pip install pyyaml\n"
            f"原始错误：{exc}"
        ) from exc

    with yaml_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(config_path: str) -> Path:
    """
    解析配置路径，优先级：
    1. 绝对路径直接使用
    2. 相对于当前工作目录
    3. 相对于项目根目录（src/agents/tools/ 往上三层）
    """
    p = Path(config_path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在：{p}")
        return p

    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd.resolve()

    # src/agents/tools/_common.py → 项目根
    project_root = Path(__file__).parent.parent.parent.parent
    from_root = project_root / p
    if from_root.exists():
        return from_root.resolve()

    raise FileNotFoundError(
        f"配置文件不存在：{config_path!r}\n"
        f"  已尝试：\n"
        f"    {from_cwd}\n"
        f"    {from_root}"
    )


# ---------------------------------------------------------------------------
# Aspen 工具的可替换依赖引用（方便测试时 monkeypatch）
# 实际运行时由各 _import_*_deps() 在首次调用时按需 import 并赋值。
# 测试中通过 patch("src.agents.tools._common._xxx", ...) 等路径打桩。
# ---------------------------------------------------------------------------
_load_optimize_config: Any = None    # src.utils.file_io.load_optimize_config
_AspenDriver: Any = None             # src.aspen_driver.driver.AspenDriver
_run_case_fn: Any = None             # src.workflows.run_case.run_case
_optimize_pareto_fn: Any = None      # src.workflows.optimize_pareto_case.optimize_pareto_case


def _import_run_time_deps() -> str | None:
    """
    按需导入 run_case_tool 运行时依赖。
    成功时更新模块级引用并返回 None；失败时返回错误字符串。
    """
    global _load_optimize_config, _AspenDriver, _run_case_fn
    try:
        from src.utils.file_io import load_optimize_config
        _load_optimize_config = load_optimize_config
    except ImportError as exc:
        return f"错误：无法导入 load_optimize_config — {exc}"
    try:
        from src.aspen_driver.driver import AspenDriver
        _AspenDriver = AspenDriver
    except ImportError as exc:
        return f"错误：无法导入 AspenDriver（请确认在 Windows + pywin32 环境中运行）— {exc}"
    try:
        from src.workflows.run_case import run_case
        _run_case_fn = run_case
    except ImportError as exc:
        return f"错误：无法导入 run_case — {exc}"
    return None


def _import_pareto_deps() -> str | None:
    """
    按需导入 optimize_pareto_tool 运行时依赖。
    成功时更新模块级引用并返回 None；失败时返回错误字符串。
    """
    global _load_optimize_config, _AspenDriver, _optimize_pareto_fn
    try:
        from src.utils.file_io import load_optimize_config
        _load_optimize_config = load_optimize_config
    except ImportError as exc:
        return f"错误：无法导入 load_optimize_config — {exc}"
    try:
        from src.aspen_driver.driver import AspenDriver
        _AspenDriver = AspenDriver
    except ImportError as exc:
        return f"错误：无法导入 AspenDriver（请确认在 Windows + pywin32 环境中运行）— {exc}"
    try:
        from src.workflows.optimize_pareto_case import optimize_pareto_case
        _optimize_pareto_fn = optimize_pareto_case
    except ImportError as exc:
        return f"错误：无法导入 optimize_pareto_case — {exc}"
    return None
