"""
smoke_case_prepare.py — 隔离 full smoke case 准备器（纯 Python）。

从现有 case_config 生成隔离运行目录，复制必要文件并改写 YAML，
使下一步可安全运行小规模 full --allow-aspen。

严禁：
  - 启动 Aspen
  - 调用 run_case / optimize_pareto / run_demo_case_workflow
  - 写 SimulationDB / NodeDB
  - 导入 AspenDriver / SimulationRunner / SimulationDB / NodeDB
  - 导入 src.aspen_driver / src.database / src.workflows
"""
from __future__ import annotations

import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 项目根目录（src/agents/smoke_case_prepare.py → 上三层）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class PrepareResult:
    """prepare_isolated_smoke_case 的返回结果。"""
    success: bool

    # 原始路径
    orig_config_path: str
    orig_bkp_path: str | None

    # 新目录
    out_dir: str

    # 新路径
    new_config_path: str
    new_bkp_path: str | None
    new_simulation_db_path: str
    new_node_db_path: str

    # 改写内容
    n_initial_written: int
    n_iterations_written: int

    # 状态
    output_copied: bool   # 应始终为 False
    error: str | None = None

    def next_steps(self) -> list[str]:
        """返回下一步建议命令列表。"""
        return [
            (
                f"python scripts/preflight_full_aspen.py "
                f"--config {self.new_config_path} --suggest-copy"
            ),
            (
                f"python scripts/smoke_agent_workflow.py "
                f"--config {self.new_config_path} --mode full --allow-aspen"
            ),
        ]


# ---------------------------------------------------------------------------
# 路径解析辅助
# ---------------------------------------------------------------------------

def _resolve_bkp_path(raw: str, config_dir: Path) -> Path | None:
    """解析 simulator.filepath，返回找到的绝对路径，找不到返回 None。"""
    p = Path(raw)
    candidates: list[Path]
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [
            Path.cwd() / p,
            config_dir / p,
            _PROJECT_ROOT / p,
        ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(data: dict[str, Any], path: Path) -> None:
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def prepare_isolated_smoke_case(
    config_path: str,
    out_root: str = "runs",
    n_initial: int = 2,
    n_iterations: int = 1,
    force: bool = False,
) -> PrepareResult:
    """创建隔离运行目录并准备小规模 smoke 配置。

    流程：
    1. 解析 config_path，读取 YAML。
    2. 生成目标目录名 {out_root}/{case_name}_YYYYMMDD_HHMMSS/。
    3. 如果目录已存在且 force=False，返回失败。
    4. 复制 .bkp 文件到新目录。
    5. 改写 YAML：
       - simulator.filepath → 新 .bkp 相对路径
       - extraction.catalog_db → output/node.db
       - optimizer.n_initial_points / n_iterations → 传入参数
    6. 写新 YAML 到新目录。
    7. 不复制 output/ 目录。

    Args:
        config_path: 原始 YAML 配置路径。
        out_root:    运行目录父路径（默认 "runs"）。
        n_initial:   新 YAML 的 optimizer.n_initial_points。
        n_iterations: 新 YAML 的 optimizer.n_iterations。
        force:       True 时允许覆盖已有目录。

    Returns:
        PrepareResult 实例，success=False 时 error 字段有说明。
    """
    # ── 解析原始配置 ────────────────────────────────────────────────────────
    orig_p = Path(config_path)
    if not orig_p.is_absolute():
        # 依次尝试 cwd 和项目根
        for base in (Path.cwd(), _PROJECT_ROOT):
            candidate = base / orig_p
            if candidate.exists():
                orig_p = candidate.resolve()
                break
        else:
            return PrepareResult(
                success=False,
                orig_config_path=config_path,
                orig_bkp_path=None,
                out_dir="",
                new_config_path="",
                new_bkp_path=None,
                new_simulation_db_path="",
                new_node_db_path="",
                n_initial_written=n_initial,
                n_iterations_written=n_iterations,
                output_copied=False,
                error=f"配置文件不存在：{config_path}",
            )
    else:
        if not orig_p.exists():
            return PrepareResult(
                success=False,
                orig_config_path=config_path,
                orig_bkp_path=None,
                out_dir="",
                new_config_path="",
                new_bkp_path=None,
                new_simulation_db_path="",
                new_node_db_path="",
                n_initial_written=n_initial,
                n_iterations_written=n_iterations,
                output_copied=False,
                error=f"配置文件不存在：{config_path}",
            )
        orig_p = orig_p.resolve()

    config_dir = orig_p.parent

    try:
        cfg = _load_yaml(orig_p)
    except Exception as exc:
        return PrepareResult(
            success=False,
            orig_config_path=config_path,
            orig_bkp_path=None,
            out_dir="",
            new_config_path="",
            new_bkp_path=None,
            new_simulation_db_path="",
            new_node_db_path="",
            n_initial_written=n_initial,
            n_iterations_written=n_iterations,
            output_copied=False,
            error=f"YAML 解析失败：{exc}",
        )

    # ── 解析 .bkp 路径 ───────────────────────────────────────────────────────
    bkp_raw: str = cfg.get("simulator", {}).get("filepath", "")
    orig_bkp: Path | None = None
    if bkp_raw:
        orig_bkp = _resolve_bkp_path(bkp_raw, config_dir)

    # ── 生成目标目录 ─────────────────────────────────────────────────────────
    case_name = config_dir.name
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(out_root) / f"{case_name}_{ts}").resolve()

    if out_dir.exists() and not force:
        return PrepareResult(
            success=False,
            orig_config_path=str(orig_p),
            orig_bkp_path=str(orig_bkp) if orig_bkp else None,
            out_dir=str(out_dir),
            new_config_path="",
            new_bkp_path=None,
            new_simulation_db_path="",
            new_node_db_path="",
            n_initial_written=n_initial,
            n_iterations_written=n_iterations,
            output_copied=False,
            error=(
                f"目标目录已存在：{out_dir}。"
                "请传入 --force 覆盖，或等待下一秒重新运行（时间戳不同）。"
            ),
        )

    out_dir.mkdir(parents=True, exist_ok=force)

    # ── 复制 .bkp ────────────────────────────────────────────────────────────
    new_bkp: Path | None = None
    if orig_bkp is not None:
        new_bkp = out_dir / orig_bkp.name
        shutil.copy2(orig_bkp, new_bkp)

    # ── 改写 YAML ────────────────────────────────────────────────────────────
    # simulator.filepath：指向新目录下的 .bkp（相对于新 YAML 所在目录）
    if new_bkp is not None:
        cfg.setdefault("simulator", {})["filepath"] = new_bkp.name  # 同目录，直接文件名

    # extraction.catalog_db → output/node.db（相对于新 YAML 目录）
    cfg.setdefault("extraction", {})["catalog_db"] = "output/node.db"

    # optimizer 参数
    cfg.setdefault("optimizer", {})["n_initial_points"] = n_initial
    cfg.setdefault("optimizer", {})["n_iterations"] = n_iterations

    # ── 写新 YAML ────────────────────────────────────────────────────────────
    new_config = out_dir / orig_p.name
    _dump_yaml(cfg, new_config)

    # ── 推断路径（不创建 output/）────────────────────────────────────────────
    new_sim_db = out_dir / "output" / "simulation.db"
    new_node_db = out_dir / "output" / "node.db"

    return PrepareResult(
        success=True,
        orig_config_path=str(orig_p),
        orig_bkp_path=str(orig_bkp) if orig_bkp else None,
        out_dir=str(out_dir),
        new_config_path=str(new_config),
        new_bkp_path=str(new_bkp) if new_bkp else None,
        new_simulation_db_path=str(new_sim_db),
        new_node_db_path=str(new_node_db),
        n_initial_written=n_initial,
        n_iterations_written=n_iterations,
        output_copied=False,
    )


# ---------------------------------------------------------------------------
# 报告格式化
# ---------------------------------------------------------------------------

def format_prepare_report(result: PrepareResult) -> str:
    """将 PrepareResult 格式化为可读文本报告。"""
    lines = ["=== prepare_isolated_full_smoke 准备报告 ===", ""]

    if not result.success:
        lines.append(f"[失败]  {result.error}")
        return "\n".join(lines)

    lines.extend([
        "【路径对比】",
        f"  原 config  : {result.orig_config_path}",
        f"  新 config  : {result.new_config_path}",
        f"  原 bkp     : {result.orig_bkp_path or '（未找到）'}",
        f"  新 bkp     : {result.new_bkp_path or '（未复制）'}",
        f"  新 sim.db  : {result.new_simulation_db_path}（尚不存在，运行后生成）",
        f"  新 node.db : {result.new_node_db_path}（尚不存在，运行后生成）",
        "",
        "【改写内容】",
        f"  simulator.filepath       → {Path(result.new_bkp_path).name if result.new_bkp_path else '（未改写）'}",
        "  extraction.catalog_db    → output/node.db",
        f"  optimizer.n_initial_points → {result.n_initial_written}",
        f"  optimizer.n_iterations     → {result.n_iterations_written}",
        "",
        "【安全确认】",
        f"  output/ 目录已复制：{'是（警告）' if result.output_copied else '否'}",
        "",
        "【下一步建议】",
    ])
    for cmd in result.next_steps():
        lines.append(f"  {cmd}")

    return "\n".join(lines)
