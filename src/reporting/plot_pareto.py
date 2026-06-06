"""
src/reporting/plot_pareto.py — 从 simulation.db 生成 Pareto 可视化图表。

可作为框架内部模块调用（main.py 优化结束后自动触发），
也保留与 scripts/plot_pareto.py 相同的 CLI 接口向后兼容。

主要入口
--------
    generate_pareto_report(db_path, out_dir)   ← 框架调用
    main()                                      ← CLI 调用（供 scripts/ 复用）
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 字体配置
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> bool:
    """尝试配置中文字体，返回是否成功。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Source Han Sans CN",
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return True
    return False


_HAS_CJK = _setup_chinese_font()
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def _detect_objectives(cur: sqlite3.Cursor, session_id: str | None = None) -> tuple[str, str]:
    """
    从 objectives 表自动检测两个目标名称。

    优先级：
    1. 若存在 TAC 和 EMISSIONS，保持向后兼容。
    2. 否则取 available=1 中出现次数最多的前两个名称。

    指定 session_id 时仅在该优化 session 的工况内检测，避免历史 session
    使用不同目标组合（如旧库为 TAC/EMISSIONS、本轮为 CAPEX/OPEX）时
    误选到全库目标。

    返回 (obj_x_name, obj_y_name)。
    """
    if session_id is not None:
        cur.execute("""
            SELECT o.name, COUNT(*) AS cnt
            FROM objectives o
            JOIN cases c ON c.case_id = o.case_id
            WHERE o.available = 1 AND c.session_id = ?
            GROUP BY o.name
            ORDER BY cnt DESC, o.name ASC
        """, (session_id,))
    else:
        cur.execute("""
            SELECT name, COUNT(*) AS cnt
            FROM objectives
            WHERE available = 1
            GROUP BY name
            ORDER BY cnt DESC, name ASC
        """)
    names = [r[0] for r in cur.fetchall()]

    if not names:
        raise ValueError("objectives 表中没有 available=1 的数据，无法确定目标名称。")

    if "TAC" in names and "EMISSIONS" in names:
        return "TAC", "EMISSIONS"

    if len(names) < 2:
        raise ValueError(
            f"只检测到 1 个目标名称 ({names[0]})，至少需要 2 个才能绘制 Pareto 图。"
        )

    return names[0], names[1]


def load_data(db_path: Path, session_id: str | None = None) -> dict:
    """从 simulation.db 读取所有绘图所需数据，返回统一结构。

    Parameters
    ----------
    db_path : 数据库文件路径
    session_id : 若指定，则仅读取该优化 session 的工况；为 None 时读取全库历史数据。
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    obj_x, obj_y = _detect_objectives(cur, session_id=session_id)
    log.info("检测到目标：X=%s, Y=%s", obj_x, obj_y)

    session_filter = "AND c.session_id = ?" if session_id is not None else ""
    cur.execute(f"""
        SELECT c.case_id, c.iteration, c.status, c.feasible,
               MAX(CASE WHEN o.name=? THEN o.value   END) AS obj_x,
               MAX(CASE WHEN o.name=? THEN o.value   END) AS obj_y,
               MAX(CASE WHEN o.name=? THEN o.minimize END) AS obj_x_minimize,
               MAX(CASE WHEN o.name=? THEN o.minimize END) AS obj_y_minimize,
               MAX(CASE WHEN o.name=? THEN o.unit    END) AS obj_x_unit,
               MAX(CASE WHEN o.name=? THEN o.unit    END) AS obj_y_unit
        FROM cases c
        JOIN objectives o ON c.case_id = o.case_id AND o.available = 1
        WHERE 1=1 {session_filter}
        GROUP BY c.case_id
        HAVING obj_x IS NOT NULL AND obj_y IS NOT NULL
        ORDER BY c.iteration
    """, (obj_x, obj_y, obj_x, obj_y, obj_x, obj_y)
         + ((session_id,) if session_id is not None else ()))
    rows = cur.fetchall()

    dv_filter = "AND session_id = ?" if session_id is not None else ""
    cur.execute(f"""
        SELECT design_vars FROM cases
        WHERE status = 'success' AND feasible = 1
          AND design_vars IS NOT NULL
          {dv_filter}
    """, (session_id,) if session_id is not None else ())
    dv_rows = cur.fetchall()

    conn.close()

    x_minimize = bool(rows[0][6]) if rows and rows[0][6] is not None else True
    y_minimize = bool(rows[0][7]) if rows and rows[0][7] is not None else True
    x_unit     = rows[0][8] if rows else ""
    y_unit     = rows[0][9] if rows else ""

    cases = [
        {
            "case_id":   r[0],
            "iteration": r[1],
            "status":    r[2],
            "feasible":  bool(r[3]),
            "obj_x":     r[4],
            "obj_y":     r[5],
        }
        for r in rows
    ]

    design_vars: list[dict] = []
    for (dv_json,) in dv_rows:
        try:
            design_vars.append(json.loads(dv_json))
        except Exception:
            pass

    return {
        "cases":       cases,
        "design_vars": design_vars,
        "obj_x":       obj_x,
        "obj_y":       obj_y,
        "x_minimize":  x_minimize,
        "y_minimize":  y_minimize,
        "x_unit":      x_unit or "",
        "y_unit":      y_unit or "",
    }


# ---------------------------------------------------------------------------
# Pareto 前沿计算
# ---------------------------------------------------------------------------

def _pareto_front(
    cases: list[dict],
    x_minimize: bool = True,
    y_minimize: bool = True,
) -> list[dict]:
    """从可行工况中提取非支配解（支持最大化/最小化方向）。"""
    feasible = [c for c in cases if c["feasible"]]
    front = []
    for c in feasible:
        cx, cy = c["obj_x"], c["obj_y"]
        dominated = False
        for o in feasible:
            ox, oy = o["obj_x"], o["obj_y"]
            x_better_eq = (ox <= cx) if x_minimize else (ox >= cx)
            y_better_eq = (oy <= cy) if y_minimize else (oy >= cy)
            x_strict    = (ox < cx)  if x_minimize else (ox > cx)
            y_strict    = (oy < cy)  if y_minimize else (oy > cy)
            if x_better_eq and y_better_eq and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda p: p["obj_x"])


# ---------------------------------------------------------------------------
# 图 1：Pareto 散点图
# ---------------------------------------------------------------------------

def plot_pareto_scatter(
    cases: list[dict],
    out_dir: Path,
    *,
    obj_x: str = "obj_x",
    obj_y: str = "obj_y",
    x_minimize: bool = True,
    y_minimize: bool = True,
    x_unit: str = "",
    y_unit: str = "",
) -> Path:
    feasible   = [c for c in cases if c["feasible"]]
    infeasible = [c for c in cases if not c["feasible"] and c["status"] != "sim_failed"]
    front      = _pareto_front(cases, x_minimize=x_minimize, y_minimize=y_minimize)

    # X 轴自动缩放：量级 > 1e5 时转为 M 前缀
    x_vals_all = [c["obj_x"] for c in feasible + infeasible if c["obj_x"] is not None]
    if x_vals_all and max(abs(v) for v in x_vals_all) > 1e5:
        x_scale, x_prefix = 1e6, "M"
    else:
        x_scale, x_prefix = 1.0, ""

    x_unit_str = f"{x_prefix}{x_unit}" if x_unit else x_prefix
    x_dir = "↓min" if x_minimize else "↑max"
    y_dir = "↓min" if y_minimize else "↑max"
    x_label = f"{obj_x} ({x_unit_str}) {x_dir}" if x_unit_str else f"{obj_x} {x_dir}"
    y_label = f"{obj_y} ({y_unit}) {y_dir}"   if y_unit      else f"{obj_y} {y_dir}"

    fig, ax = plt.subplots(figsize=(8, 6))

    if infeasible:
        ax.scatter(
            [c["obj_x"] / x_scale for c in infeasible],
            [c["obj_y"]           for c in infeasible],
            c="#cccccc", s=30, alpha=0.6,
            label=f"Infeasible ({len(infeasible)})", zorder=2,
        )

    if feasible:
        iters = [c["iteration"] for c in feasible]
        sc = ax.scatter(
            [c["obj_x"] / x_scale for c in feasible],
            [c["obj_y"]           for c in feasible],
            c=iters, cmap="viridis_r", s=50, alpha=0.85,
            label=f"Feasible ({len(feasible)})", zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Iteration", fontsize=10)

    if front:
        ax.plot(
            [c["obj_x"] / x_scale for c in front],
            [c["obj_y"]           for c in front],
            "r-o", ms=8, lw=1.5, zorder=4,
            label=f"Pareto Front ({len(front)})",
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"Pareto Front — {obj_x} vs {obj_y}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    out = out_dir / "pareto_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("已保存：%s", out)
    return out


# ---------------------------------------------------------------------------
# 图 2：超体积收敛曲线
# ---------------------------------------------------------------------------

def _hv2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """2D 超体积（最小化空间，扫描线法）。"""
    pts = sorted(
        [(x, y) for x, y in points if x < ref[0] and y < ref[1]],
        key=lambda p: p[0],
    )
    if not pts:
        return 0.0
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv


def _compute_hv_history(
    cases: list[dict],
    x_minimize: bool = True,
    y_minimize: bool = True,
) -> list[tuple[int, float]]:
    """按迭代顺序逐步计算超体积（统一转换到最小化空间）。"""
    feasible = sorted(
        [c for c in cases if c["feasible"]],
        key=lambda c: c["iteration"],
    )
    if not feasible:
        return []

    def to_min(c: dict) -> tuple[float, float]:
        return (
            c["obj_x"] if x_minimize else -c["obj_x"],
            c["obj_y"] if y_minimize else -c["obj_y"],
        )

    pts_min = [to_min(c) for c in feasible]
    ref = (max(p[0] for p in pts_min) * 1.1,
           max(p[1] for p in pts_min) * 1.1)

    history: list[tuple[int, float]] = []
    seen: list[dict] = []
    for c in feasible:
        seen.append(c)
        front = _pareto_front(seen, x_minimize=x_minimize, y_minimize=y_minimize)
        hv = _hv2d([to_min(p) for p in front], ref)
        history.append((c["iteration"], hv))
    return history


def plot_hv_history(
    cases: list[dict],
    out_dir: Path,
    *,
    x_minimize: bool = True,
    y_minimize: bool = True,
) -> Path | None:
    history = _compute_hv_history(cases, x_minimize=x_minimize, y_minimize=y_minimize)
    if not history:
        log.info("无可行工况，跳过超体积曲线。")
        return None

    iters = [h[0] for h in history]
    hvs   = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, hvs, "b-", lw=1.5, alpha=0.8)
    ax.fill_between(iters, hvs, alpha=0.15, color="blue")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Hypervolume (HV)", fontsize=12)
    ax.set_title("Hypervolume Convergence", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    out = out_dir / "hv_history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("已保存：%s", out)
    return out


# ---------------------------------------------------------------------------
# 图 3：设计变量分布
# ---------------------------------------------------------------------------

def plot_design_vars(design_vars: list[dict], out_dir: Path) -> Path | None:
    if not design_vars:
        log.info("无设计变量数据，跳过分布图。")
        return None

    # 动态收集所有键（保持首次出现顺序）
    all_keys: list[str] = []
    seen_keys: set[str] = set()
    for dv in design_vars:
        for k in dv:
            if k not in seen_keys:
                seen_keys.add(k)
                all_keys.append(k)

    def _short_label(path: str) -> str:
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else path)

    data_by_key: dict[str, list[float]] = {k: [] for k in all_keys}
    for dv in design_vars:
        for k in all_keys:
            if k in dv and dv[k] is not None:
                try:
                    data_by_key[k].append(float(dv[k]))
                except (TypeError, ValueError):
                    pass

    active_keys = [k for k in all_keys if data_by_key[k]]
    if not active_keys:
        log.info("设计变量数据为空，跳过分布图。")
        return None

    n = len(active_keys)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    # 每个变量独立子图，各自使用自己的 Y 轴范围
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.8, nrows * 4.0),
        squeeze=False,
    )

    for idx, key in enumerate(active_keys):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        vals = data_by_key[key]
        color = palette[idx % len(palette)]

        bp = ax.boxplot(
            [vals],
            tick_labels=[_short_label(key)],
            patch_artist=True,
            notch=False,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.6)

        # 在箱线图旁叠加散点（jitter），让分布更直观
        import numpy as np
        rng = np.random.default_rng(seed=42)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            1 + jitter, vals,
            color=color, alpha=0.35, s=12, zorder=3,
        )

        # Y 轴：在数据范围基础上留 10% 边距
        lo, hi = min(vals), max(vals)
        margin = (hi - lo) * 0.10 if hi > lo else abs(lo) * 0.10 + 0.1
        ax.set_ylim(lo - margin, hi + margin)

        # 在图内标注统计数字
        median = float(np.median(vals))
        ax.set_title(
            f"median={median:.3g}\nn={len(vals)}",
            fontsize=8, pad=3,
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    # 隐藏多余子图格
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Design Variable Distribution — Feasible Cases (n={len(design_vars)})",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "design_vars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("已保存：%s", out)
    return out


# ---------------------------------------------------------------------------
# 统一入口（框架调用）
# ---------------------------------------------------------------------------

def generate_pareto_report(
    db_path: Path | str,
    out_dir: Path | str | None = None,
    session_id: str | None = None,
) -> None:
    """
    从 simulation.db 自动生成 Pareto 可视化图表，保存到 out_dir。

    Parameters
    ----------
    db_path : 数据库文件路径
    out_dir : 输出目录；默认与数据库文件同目录
    session_id : 若指定，则仅绘制该优化 session 的结果；为 None 时绘制全库历史数据
    """
    db_path = Path(db_path)
    out_dir = Path(out_dir) if out_dir else db_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if session_id is not None:
        log.info("生成 Pareto 报告：%s → %s，session_id=%s", db_path.name, out_dir, session_id)
    else:
        log.info("生成 Pareto 报告：%s → %s，全库历史数据", db_path.name, out_dir)

    try:
        data = load_data(db_path, session_id=session_id)
    except ValueError as exc:
        log.warning("跳过报告生成：%s", exc)
        return

    cases       = data["cases"]
    design_vars = data["design_vars"]
    n_feasible  = sum(1 for c in cases if c["feasible"])
    scope = f"session_id={session_id}" if session_id is not None else "全库历史"
    log.info(
        "读取工况：%s，共 %d 条，可行 %d，不可行 %d",
        scope, len(cases), n_feasible, len(cases) - n_feasible,
    )

    if not cases:
        log.warning("数据库中无有效工况，跳过绘图。")
        return

    plot_pareto_scatter(
        cases, out_dir,
        obj_x=data["obj_x"],      obj_y=data["obj_y"],
        x_minimize=data["x_minimize"], y_minimize=data["y_minimize"],
        x_unit=data["x_unit"],    y_unit=data["y_unit"],
    )
    plot_hv_history(
        cases, out_dir,
        x_minimize=data["x_minimize"], y_minimize=data["y_minimize"],
    )
    plot_design_vars(design_vars, out_dir)
    log.info("Pareto 报告生成完毕。")


# ---------------------------------------------------------------------------
# CLI 入口（供 scripts/plot_pareto.py 复用）
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, sys
    parser = argparse.ArgumentParser(description="PAO 优化结果可视化")
    parser.add_argument("db",    help="simulation.db 路径")
    parser.add_argument("--out", default=None, help="输出目录（默认与 db 同目录）")
    parser.add_argument(
        "--session-id", default=None,
        help="仅绘制指定优化 session（默认绘制全库历史数据）",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"错误：数据库文件不存在：{db_path}", file=sys.stderr)
        sys.exit(1)

    # 使用标准输出打印（CLI 模式不依赖 logging 配置）
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_pareto_report(db_path, args.out, session_id=args.session_id)
    print("完成。")


if __name__ == "__main__":
    main()
