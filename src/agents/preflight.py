"""
preflight.py — full --allow-aspen 前置检查的纯 Python 逻辑层。

不导入：AspenDriver、SimulationRunner、SimulationDB、NodeDB、
        src.workflows、src.aspen_driver、src.database、LangGraph。
不调用：run_case_tool、optimize_pareto_tool、run_demo_case_workflow。
不写数据库、不启动 Aspen（--check-com 路径除外，且必须显式开启）。
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 评估次数风险阈值
_RISK_LOW = 5        # <= 5：低风险
_RISK_MID = 20       # 6-20：中风险（WARN）
# > 20：高风险（WARN）

# 项目根目录（src/agents/preflight.py → 上三层）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """单项检查结果。"""
    name: str
    passed: bool          # True=通过，False=失败/警告
    is_warn: bool = False # True=WARN（不阻断但需关注），False=FAIL（阻断）
    message: str = ""
    detail: str = ""      # 可选详情行


@dataclass
class PreflightReport:
    """完整前置检查结果。"""
    config_path: str
    resolved_config_path: str | None
    optimizer_type: str
    objective_names: list[str]
    db_path: str | None
    node_db_path: str | None

    checks: list[CheckResult] = field(default_factory=list)
    suggest_copy_dir: str | None = None

    @property
    def overall(self) -> str:
        """综合结论：PASS / WARN / FAIL。"""
        has_fail = any(not c.passed and not c.is_warn for c in self.checks)
        has_warn = any(not c.passed and c.is_warn for c in self.checks)
        if has_fail:
            return "FAIL"
        if has_warn:
            return "WARN"
        return "PASS"

    @property
    def exit_code(self) -> int:
        """退出码：0=PASS，1=FAIL，2=WARN。"""
        return {"PASS": 0, "WARN": 2, "FAIL": 1}[self.overall]


# ---------------------------------------------------------------------------
# 1. 配置解析
# ---------------------------------------------------------------------------

def check_config(config_path: str) -> tuple[PreflightReport | None, CheckResult]:
    """解析配置，返回 (report_with_fields, check_result)。

    若解析失败返回 (None, CheckResult(passed=False))。
    """
    from src.agents.workflow_helpers import prepare_demo_workflow_state

    try:
        state = prepare_demo_workflow_state(config_path)
    except FileNotFoundError as exc:
        return None, CheckResult(
            name="配置路径解析",
            passed=False,
            is_warn=False,
            message=f"配置文件不存在：{exc}",
        )
    except Exception as exc:
        return None, CheckResult(
            name="配置路径解析",
            passed=False,
            is_warn=False,
            message=f"配置解析失败 [{type(exc).__name__}]：{exc}",
        )

    report = PreflightReport(
        config_path=config_path,
        resolved_config_path=state.resolved_config_path,
        optimizer_type=state.optimizer_type,
        objective_names=state.objective_names,
        db_path=state.db_path,
        node_db_path=state.node_db_path,
    )
    check = CheckResult(
        name="配置路径解析",
        passed=True,
        message="YAML 解析成功，optimizer_type / objective_names / db_path 已填充",
    )
    return report, check


# ---------------------------------------------------------------------------
# 2. Aspen 文件存在性
# ---------------------------------------------------------------------------

def check_aspen_file(yaml_path: str) -> CheckResult:
    """检查 simulator.filepath 指向的 Aspen .bkp 是否存在。"""
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as exc:
        return CheckResult(
            name="Aspen 文件存在性",
            passed=False,
            message=f"无法读取 YAML：{exc}",
        )

    raw = cfg.get("simulator", {}).get("filepath", "")
    if not raw:
        return CheckResult(
            name="Aspen 文件存在性",
            passed=False,
            message="simulator.filepath 未配置",
        )

    p = Path(raw)
    config_dir = Path(yaml_path).parent
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / p,
            config_dir / p,
            _PROJECT_ROOT / p,
        ])

    for candidate in candidates:
        if candidate.exists():
            return CheckResult(
                name="Aspen 文件存在性",
                passed=True,
                message=f"文件存在",
                detail=f"原始路径: {raw}\n  解析路径: {candidate.resolve()}",
            )

    tried = "\n    ".join(str(c) for c in candidates)
    return CheckResult(
        name="Aspen 文件存在性",
        passed=False,
        message=f"Aspen 文件不存在（原始路径: {raw!r}）",
        detail=f"已检查：\n    {tried}",
    )


# ---------------------------------------------------------------------------
# 3. 优化规模
# ---------------------------------------------------------------------------

def check_optimization_scale(yaml_path: str) -> CheckResult:
    """检查预计 Aspen 评估次数，给出风险等级。"""
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as exc:
        return CheckResult(
            name="优化规模",
            passed=False,
            message=f"无法读取 YAML：{exc}",
        )

    opt = cfg.get("optimizer", {}) or {}
    try:
        n_init = int(opt.get("n_initial_points", 10))
    except (TypeError, ValueError) as exc:
        return CheckResult(
            name="优化规模",
            passed=False,
            is_warn=False,
            message=f"n_initial_points 值非法，无法解析为整数：{opt.get('n_initial_points')!r}（{exc}）",
        )
    try:
        n_iter = int(opt.get("n_iterations", 30))
    except (TypeError, ValueError) as exc:
        return CheckResult(
            name="优化规模",
            passed=False,
            is_warn=False,
            message=f"n_iterations 值非法，无法解析为整数：{opt.get('n_iterations')!r}（{exc}）",
        )
    total = n_init + n_iter

    # Phase 0 / feasibility_search
    phase0_n = 0
    fs = cfg.get("feasibility_search") or {}
    if fs.get("enabled"):
        try:
            phase0_n = int(fs.get("n_trials", 0))
        except (TypeError, ValueError) as exc:
            return CheckResult(
                name="优化规模",
                passed=False,
                is_warn=False,
                message=(
                    f"feasibility_search.n_trials 值非法，无法解析为整数："
                    f"{fs.get('n_trials')!r}（{exc}）"
                ),
            )
        total += phase0_n

    detail_lines = [
        f"n_initial_points = {n_init}",
        f"n_iterations      = {n_iter}",
    ]
    if phase0_n:
        detail_lines.append(f"phase0.n_trials   = {phase0_n}")
    detail_lines.append(f"预计总评估次数    = {total}")

    if total <= _RISK_LOW:
        return CheckResult(
            name="优化规模",
            passed=True,
            message=f"低风险：预计 {total} 次评估，适合 smoke 验证",
            detail="\n  ".join(detail_lines),
        )
    elif total <= _RISK_MID:
        return CheckResult(
            name="优化规模",
            passed=False,
            is_warn=True,
            message=f"中风险：预计 {total} 次评估，建议确认后再运行",
            detail="\n  ".join(detail_lines),
        )
    else:
        return CheckResult(
            name="优化规模",
            passed=False,
            is_warn=True,
            message=(
                f"高风险：预计 {total} 次 Aspen 评估，不建议作为第一次 full smoke 直接运行。"
                " 建议先复制配置并把 n_initial_points/n_iterations 降到 1/1 或 2/1。"
            ),
            detail="\n  ".join(detail_lines),
        )


# ---------------------------------------------------------------------------
# 4. 输出目录风险
# ---------------------------------------------------------------------------

def check_output_risk(db_path: str | None, node_db_path: str | None) -> CheckResult:
    """检查 output/simulation.db 和 node.db 是否已存在。"""
    found: list[str] = []

    for label, path_str in [("simulation.db", db_path), ("node.db", node_db_path)]:
        if path_str is None:
            continue
        p = Path(path_str)
        if p.exists():
            found.append(f"{label} ({p})")
        # WAL / SHM 文件：SQLite 在数据库文件名后直接追加 -wal / -shm
        # 例如 simulation.db-wal，而非 simulation.db.db-wal
        for suffix in ("-wal", "-shm"):
            wal = Path(str(p) + suffix)
            if wal.exists():
                found.append(f"{p.name}{suffix}（可能有未提交事务）")

    if not found:
        return CheckResult(
            name="输出目录风险",
            passed=True,
            message="output 目录无已有数据库文件，可直接运行",
        )

    files_str = "\n    ".join(found)
    return CheckResult(
        name="输出目录风险",
        passed=False,
        is_warn=True,
        message="检测到已有历史输出，full 运行可能追加或修改数据库",
        detail=(
            f"已存在文件：\n    {files_str}\n"
            "  建议：复制 case 到 runs/... 临时目录后再运行 full --allow-aspen"
        ),
    )


# ---------------------------------------------------------------------------
# 5. 临时运行目录建议
# ---------------------------------------------------------------------------

def suggest_copy_dir(config_path: str) -> str:
    """生成建议的临时运行目录名（不创建目录，不复制文件）。"""
    case_name = Path(config_path).parent.name
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"runs/{case_name}_{ts}"


def build_copy_suggestion(
    config_path: str,
    yaml_path: str | None,
    copy_dir: str | None = None,
) -> str:
    """输出详细的临时目录复制建议（纯文本，不执行）。

    Args:
        copy_dir: 若已由调用方生成（如 rpt.suggest_copy_dir），直接复用，
                  避免重新生成时间戳导致与报告字段不一致。
    """
    if copy_dir is None:
        copy_dir = suggest_copy_dir(config_path)
    lines = [
        f"建议临时运行目录：{copy_dir}",
        "需要复制的内容：",
        f"  1. YAML 配置：{config_path}",
    ]
    bkp_path = ""
    catalog_db_path = ""
    if yaml_path:
        try:
            import yaml
            with open(yaml_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            bkp_path = cfg.get("simulator", {}).get("filepath", "")
            if bkp_path:
                lines.append(f"  2. Aspen .bkp 文件：{bkp_path}")
            catalog_db_path = (cfg.get("extraction") or {}).get("catalog_db", "")
        except Exception:
            pass
    lines.extend([
        "  3. 相关语义规则配置（configs/aspen_semantics/）可只读引用",
        "  4. 不要复制 output/ 目录（新运行应从空 output 开始）",
        "  5. 复制后将 n_initial_points 和 n_iterations 降到 2/1 用于首次 smoke",
    ])
    # 路径改写提醒
    lines.append("")
    lines.append("复制后必须在新 YAML 中改写以下路径（否则仍指向原始目录）：")
    if bkp_path:
        lines.append(f"  - simulator.filepath: 改为新目录下的 .bkp 路径（原: {bkp_path}）")
    else:
        lines.append("  - simulator.filepath: 改为新目录下的 .bkp 路径")
    if catalog_db_path:
        lines.append(
            f"  - extraction.catalog_db: 改为新 output/ 下的 node.db 路径"
            f"（原: {catalog_db_path}）"
        )
    else:
        lines.append("  - extraction.catalog_db（若有）: 改为新 output/ 下的 node.db 路径")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Aspen COM 轻量检查（可选，仅 --check-com 时执行）
# ---------------------------------------------------------------------------

def check_aspen_com() -> CheckResult:
    """尝试创建 Apwn.Document COM 对象，不打开任何文件，不运行 Engine。

    只应在用户显式传 --check-com 时调用。
    """
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return CheckResult(
            name="Aspen COM 检查",
            passed=False,
            message="pywin32 未安装，无法检查 Aspen COM 环境",
        )

    app = None
    try:
        app = win32com.client.Dispatch("Apwn.Document")
        # 静默模式
        if hasattr(app, "SuppressDialogs"):
            app.SuppressDialogs = True
        return CheckResult(
            name="Aspen COM 检查",
            passed=True,
            message="Aspen COM 对象创建成功（未打开文件）",
        )
    except Exception as exc:
        return CheckResult(
            name="Aspen COM 检查",
            passed=False,
            message=f"Aspen COM 不可用 [{type(exc).__name__}]：{exc}",
        )
    finally:
        if app is not None:
            try:
                if hasattr(app, "Quit"):
                    app.Quit()
                elif hasattr(app, "Close"):
                    app.Close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 报告格式化
# ---------------------------------------------------------------------------

_ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
_CHECK_ICON = {True: "[OK]  ", False: "[!!]  "}


def format_report(rpt: PreflightReport, suggest_copy: bool) -> str:
    """将 PreflightReport 格式化为可读文本报告。"""
    lines: list[str] = [
        "=== preflight_full_aspen 前置检查报告 ===",
        "",
        "【配置摘要】",
        f"  case_config_path  : {rpt.config_path}",
        f"  resolved_path     : {rpt.resolved_config_path or '（解析失败）'}",
        f"  optimizer_type    : {rpt.optimizer_type or '（未知）'}",
        f"  objective_names   : {', '.join(rpt.objective_names) if rpt.objective_names else '（无）'}",
        f"  db_path           : {rpt.db_path or '（无）'}",
        f"  node_db_path      : {rpt.node_db_path or '（无）'}",
        "",
        "【检查项目】",
    ]

    for c in rpt.checks:
        status = _CHECK_ICON[c.passed]
        if not c.passed and c.is_warn:
            status = "[WW]  "
        lines.append(f"  {status}{c.name}：{c.message}")
        if c.detail:
            for dline in c.detail.splitlines():
                lines.append(f"         {dline}")

    lines.append("")

    if suggest_copy and rpt.resolved_config_path:
        lines.append("【临时运行目录建议】")
        suggestion = build_copy_suggestion(
            rpt.config_path,
            rpt.resolved_config_path,
            copy_dir=rpt.suggest_copy_dir,  # 复用已生成的时间戳，保持一致
        )
        for l in suggestion.splitlines():
            lines.append(f"  {l}")
        lines.append("")

    overall = rpt.overall
    lines.append("【综合结论】")
    lines.append(f"  {_ICON[overall]}  {overall}")
    if overall == "PASS":
        lines.append("  可以进入 full --allow-aspen 运行。")
    elif overall == "WARN":
        warns = [c for c in rpt.checks if not c.passed and c.is_warn]
        lines.append("  不建议直接运行 full --allow-aspen，存在以下警告：")
        for w in warns:
            lines.append(f"    - {w.name}：{w.message}")
    else:
        fails = [c for c in rpt.checks if not c.passed and not c.is_warn]
        lines.append("  必须修复以下问题后才能进入 full：")
        for f in fails:
            lines.append(f"    - {f.name}：{f.message}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口（供 CLI 调用）
# ---------------------------------------------------------------------------

def run_preflight(
    config_path: str,
    check_com: bool = False,
    suggest_copy: bool = False,
) -> tuple[PreflightReport | None, str]:
    """执行全部前置检查，返回 (report, formatted_text)。

    report 为 None 时表示配置解析失败，无法继续后续检查。
    """
    # 1. 配置解析
    rpt, config_check = check_config(config_path)
    if rpt is None:
        # 无法继续
        rpt = PreflightReport(
            config_path=config_path,
            resolved_config_path=None,
            optimizer_type="",
            objective_names=[],
            db_path=None,
            node_db_path=None,
        )
        rpt.checks.append(config_check)
        return rpt, format_report(rpt, suggest_copy)

    rpt.checks.append(config_check)

    # 2. Aspen 文件
    rpt.checks.append(check_aspen_file(rpt.resolved_config_path))

    # 3. 优化规模
    rpt.checks.append(check_optimization_scale(rpt.resolved_config_path))

    # 4. 输出目录风险
    rpt.checks.append(check_output_risk(rpt.db_path, rpt.node_db_path))

    # 5. COM 检查（可选）
    if check_com:
        rpt.checks.append(check_aspen_com())

    # 生成建议目录
    if suggest_copy:
        rpt.suggest_copy_dir = suggest_copy_dir(config_path)

    return rpt, format_report(rpt, suggest_copy)
