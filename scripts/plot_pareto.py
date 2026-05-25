"""
plot_pareto.py — 从 simulation.db 读取多目标优化结果并绘图。

用法
----
    # 从 objectives 表读取（新配置运行后）
    python scripts/plot_pareto.py cases/demo_case/output/simulation.db --obj1 ADN_FLOW --obj2 REB_DUTY

    # 从 blocks/streams JSON 回填（旧数据库，目标未写入 objectives 表时）
    python scripts/plot_pareto.py cases/demo_case/output/simulation.db \\
        --obj1 ADN_FLOW --obj1-src stream:ADN:total_mass_flow --obj1-minimize false \\
        --obj2 REB_DUTY --obj2-src block:T0301:REB_DUTY

    # 保存图片
    python scripts/plot_pareto.py cases/demo_case/output/simulation.db --save pareto.png

--obj1-src / --obj2-src 格式
-----------------------------
    objectives              从 objectives 表读取（默认，新配置运行后使用）
    block:BLOCK:OUTPUT      从 blocks JSON 中指定 block 的 outputs[name] 读取
    stream:STREAM:FIELD     从 streams JSON 中读取，FIELD 可以是：
                              total_mass_flow, total_mole_flow, temp, pres, vfrac
                              或 comp:COMPNAME:mass_frac / comp:COMPNAME:mass_flow

--obj1-minimize / --obj2-minimize
    true（默认）：该目标为最小化方向（Pareto 前沿取左下角）
    false：该目标为最大化方向（内部取负值后计算 Pareto）

依赖
----
    pip install matplotlib
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 从 blocks / streams JSON 提取单个目标值
# ---------------------------------------------------------------------------

def _extract_from_block(blocks_json: str, block_name: str, output_name: str) -> float | None:
    try:
        blocks = json.loads(blocks_json or "{}")
        block = blocks.get(block_name)
        if block is None:
            return None
        for o in block.get("outputs", []):
            if o.get("name") == output_name:
                v = o.get("value")
                return float(v) if v is not None else None
    except Exception:
        pass
    return None


def _extract_from_stream(streams_json: str, stream_name: str, field: str) -> float | None:
    try:
        streams = json.loads(streams_json or "{}")
        stream = streams.get(stream_name)
        if stream is None:
            return None
        # comp:COMPNAME:mass_frac 等组分字段
        if field.startswith("comp:"):
            _, comp_name, comp_field = field.split(":", 2)
            for c in stream.get("components", []):
                if c.get("component") == comp_name:
                    v = c.get(comp_field)
                    return float(v) if v is not None else None
            return None
        v = stream.get(field)
        return float(v) if v is not None else None
    except Exception:
        pass
    return None


def _extract_value(row: sqlite3.Row, src: str) -> float | None:
    """根据 src 描述符从 row 中提取目标值。"""
    if src == "objectives":
        return None  # 由 JOIN 处理，不走此路径
    parts = src.split(":", 2)
    if parts[0] == "block" and len(parts) == 3:
        return _extract_from_block(row["blocks"] or "{}", parts[1], parts[2])
    if parts[0] == "stream" and len(parts) == 3:
        return _extract_from_stream(row["streams"] or "{}", parts[1], parts[2])
    return None


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def load_data(
    db_path: str,
    obj1: str,
    obj2: str,
    obj1_src: str = "objectives",
    obj2_src: str = "objectives",
    obj1_minimize: bool = True,
    obj2_minimize: bool = True,
) -> dict:
    """
    从 simulation.db 读取所有同时具有 obj1 和 obj2 的工况。

    obj1_src / obj2_src 决定数据来源：
      "objectives"          — 从 objectives 子表读取（新配置运行后）
      "block:NAME:OUTPUT"   — 从 blocks JSON 回填
      "stream:NAME:FIELD"   — 从 streams JSON 回填

    obj1_minimize / obj2_minimize 决定 Pareto 方向：
      True  — 最小化（Pareto 取左下角）
      False — 最大化（内部取负值后计算 Pareto，绘图时还原）
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    both_from_objectives = (obj1_src == "objectives" and obj2_src == "objectives")

    if both_from_objectives:
        rows = conn.execute(
            """
            SELECT c.iteration, c.status, c.feasible,
                   o1.value AS v1, o1.unit AS u1,
                   o2.value AS v2, o2.unit AS u2,
                   c.design_vars, c.tags_json
            FROM cases c
            JOIN objectives o1
              ON o1.case_id = c.case_id AND o1.name = ? AND o1.available = 1
            JOIN objectives o2
              ON o2.case_id = c.case_id AND o2.name = ? AND o2.available = 1
            ORDER BY c.iteration ASC
            """,
            (obj1, obj2),
        ).fetchall()
        conn.close()

        result = _empty_result()
        for r in rows:
            result["all_x"].append(float(r["v1"]))
            result["all_y"].append(float(r["v2"]))
            result["all_status"].append(r["status"])
            result["all_tags"].append(json.loads(r["tags_json"] or "[]"))
            result["all_dv"].append(json.loads(r["design_vars"] or "{}"))
            result["all_iter"].append(r["iteration"])
            result["obj1_unit"] = r["u1"] or ""
            result["obj2_unit"] = r["u2"] or ""
        result["obj1_minimize"] = obj1_minimize
        result["obj2_minimize"] = obj2_minimize
        return result

    # 至少一个目标来自 blocks/streams JSON，需要读完整行
    rows = conn.execute(
        """
        SELECT c.iteration, c.status, c.feasible,
               c.design_vars, c.tags_json, c.blocks, c.streams,
               c.objectives_json
        FROM cases c
        WHERE c.simulation_valid = 1
        ORDER BY c.iteration ASC
        """
    ).fetchall()
    conn.close()

    result = _empty_result()
    for r in rows:
        v1 = _resolve_value(r, obj1, obj1_src)
        v2 = _resolve_value(r, obj2, obj2_src)
        if v1 is None or v2 is None:
            continue
        result["all_x"].append(v1)
        result["all_y"].append(v2)
        result["all_status"].append(r["status"])
        result["all_tags"].append(json.loads(r["tags_json"] or "[]"))
        result["all_dv"].append(json.loads(r["design_vars"] or "{}"))
        result["all_iter"].append(r["iteration"])

    result["obj1_minimize"] = obj1_minimize
    result["obj2_minimize"] = obj2_minimize
    return result


def _empty_result() -> dict:
    return {
        "all_x": [], "all_y": [], "all_status": [],
        "all_tags": [], "all_dv": [], "all_iter": [],
        "obj1_unit": "", "obj2_unit": "",
        "obj1_minimize": True, "obj2_minimize": True,
    }


def _resolve_value(row: sqlite3.Row, obj_name: str, src: str) -> float | None:
    """从 row 中按 src 描述符解析目标值。"""
    if src == "objectives":
        # 从 objectives_json 列解析
        try:
            objs = json.loads(row["objectives_json"] or "[]")
            for o in objs:
                if o.get("name") == obj_name and o.get("available"):
                    v = o.get("value")
                    return float(v) if v is not None else None
        except Exception:
            pass
        return None
    return _extract_value(row, src)


# ---------------------------------------------------------------------------
# Pareto 前沿计算（双目标均最小化方向）
# ---------------------------------------------------------------------------

def compute_pareto_front(xs: list[float], ys: list[float]) -> list[int]:
    """返回非支配点索引（xs、ys 均为最小化方向）。"""
    n = len(xs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot(
    db_path: str,
    obj1: str = "ADN_FLOW",
    obj2: str = "REB_DUTY",
    obj1_src: str = "objectives",
    obj2_src: str = "objectives",
    obj1_minimize: bool = False,
    obj2_minimize: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("错误：需要安装 matplotlib。运行：pip install matplotlib")
        sys.exit(1)

    _CJK_CANDIDATES = [
        "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS",
        "WenQuanYi Micro Hei", "Noto Sans CJK SC", "PingFang SC",
    ]
    import matplotlib.font_manager as _fm
    _available = {f.name for f in _fm.fontManager.ttflist}
    _cjk_font = next((f for f in _CJK_CANDIDATES if f in _available), None)
    if _cjk_font:
        matplotlib.rcParams["font.family"] = _cjk_font
    matplotlib.rcParams["axes.unicode_minus"] = False

    data = load_data(db_path, obj1, obj2, obj1_src, obj2_src, obj1_minimize, obj2_minimize)
    if not data["all_x"]:
        print(f"未找到同时具有 {obj1}（{obj1_src}）和 {obj2}（{obj2_src}）的有效工况。")
        print("提示：若目标未写入 objectives 表，请用 --obj1-src / --obj2-src 指定数据来源。")
        sys.exit(1)

    xs_raw   = data["all_x"]
    ys_raw   = data["all_y"]
    statuses = data["all_status"]
    tags     = data["all_tags"]
    dvs      = data["all_dv"]
    iters    = data["all_iter"]
    u1       = data["obj1_unit"]
    u2       = data["obj2_unit"]
    min1     = data["obj1_minimize"]
    min2     = data["obj2_minimize"]

    # 转为最小化方向用于 Pareto 计算，绘图时还原
    xs_min = [x if min1 else -x for x in xs_raw]
    ys_min = [y if min2 else -y for y in ys_raw]

    # 分类索引
    doe_idx     = [i for i, t in enumerate(tags) if "initial_doe" in t]
    bo_idx      = [i for i, t in enumerate(tags) if "bayesian_opt" in t]
    infeas_idx  = [i for i, s in enumerate(statuses) if s == "infeasible"]
    success_idx = [i for i, s in enumerate(statuses) if s == "success"]

    # Pareto 前沿（仅 success 工况）
    sx = [xs_min[i] for i in success_idx]
    sy = [ys_min[i] for i in success_idx]
    if sx:
        front_local = compute_pareto_front(sx, sy)
        front_idx = [success_idx[j] for j in front_local]
        front_idx.sort(key=lambda i: xs_raw[i])
    else:
        front_idx = []

    # ------------------------------------------------------------------ #
    # 图1：目标空间 Pareto 散点图
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"多目标优化结果  |  {Path(db_path).name}\n"
        f"共 {len(xs_raw)} 个有效工况（success={len(success_idx)}, infeasible={len(infeas_idx)}）",
        fontsize=12,
    )

    ax = axes[0]

    if infeas_idx:
        ax.scatter(
            [xs_raw[i] for i in infeas_idx],
            [ys_raw[i] for i in infeas_idx],
            c="lightgray", marker="x", s=40, linewidths=1,
            label=f"infeasible ({len(infeas_idx)})", zorder=2,
        )

    doe_success = [i for i in doe_idx if statuses[i] == "success"]
    if doe_success:
        ax.scatter(
            [xs_raw[i] for i in doe_success],
            [ys_raw[i] for i in doe_success],
            c="#4C9BE8", marker="o", s=55, alpha=0.75,
            label=f"初始 DOE ({len(doe_success)})", zorder=3,
        )

    bo_success = [i for i in bo_idx if statuses[i] == "success"]
    if bo_success:
        ax.scatter(
            [xs_raw[i] for i in bo_success],
            [ys_raw[i] for i in bo_success],
            c="#F28C28", marker="^", s=55, alpha=0.75,
            label=f"贝叶斯优化 ({len(bo_success)})", zorder=3,
        )

    if front_idx:
        fx = [xs_raw[i] for i in front_idx]
        fy = [ys_raw[i] for i in front_idx]
        ax.plot(fx, fy, "r--", linewidth=1.2, alpha=0.6, zorder=4)
        ax.scatter(fx, fy, c="red", marker="*", s=160, zorder=5,
                   label=f"Pareto 前沿 ({len(front_idx)})")

    dir1 = "↓最小化" if min1 else "↑最大化"
    dir2 = "↓最小化" if min2 else "↑最大化"
    x_label = f"{obj1}  {dir1}" + (f"  [{u1}]" if u1 else "")
    y_label = f"{obj2}  {dir2}" + (f"  [{u2}]" if u2 else "")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title("目标空间 Pareto 图", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ------------------------------------------------------------------ #
    # 图2：收敛历史
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    cumulative_success = []
    front_size_history = []
    best_sum_history   = []

    for i in range(len(xs_raw)):
        if statuses[i] != "success":
            continue
        cumulative_success.append((iters[i], xs_min[i], ys_min[i]))
        cx = [t[1] for t in cumulative_success]
        cy = [t[2] for t in cumulative_success]
        fi = compute_pareto_front(cx, cy)
        front_size_history.append((iters[i], len(fi)))
        mx, my = max(abs(v) for v in cx) or 1, max(abs(v) for v in cy) or 1
        best_sum = min(cx[j] / mx + cy[j] / my for j in fi)
        best_sum_history.append((iters[i], best_sum))

    if front_size_history:
        it_fs, fs = zip(*front_size_history)
        it_bs, bs = zip(*best_sum_history)

        ax2_r = ax2.twinx()
        ax2.step(it_fs, fs, where="post", color="#4C9BE8", linewidth=1.8,
                 label="Pareto 前沿点数")
        ax2.set_ylabel("Pareto 前沿点数", color="#4C9BE8", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#4C9BE8")

        ax2_r.plot(it_bs, bs, color="#E84C4C", linewidth=1.5, linestyle="--",
                   label="归一化最优 (x̂+ŷ)")
        ax2_r.set_ylabel("归一化最优 (x̂+ŷ)", color="#E84C4C", fontsize=10)
        ax2_r.tick_params(axis="y", labelcolor="#E84C4C")

        ax2.set_xlabel("迭代编号", fontsize=11)
        ax2.set_title("收敛历史", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.4)
        lines1, l1 = ax2.get_legend_handles_labels()
        lines2, l2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, l1 + l2, fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"图表已保存：{save_path}")
    else:
        plt.show()

    # ------------------------------------------------------------------ #
    # 打印 Pareto 前沿摘要
    # ------------------------------------------------------------------ #
    if front_idx:
        print(f"\nPareto 前沿（{len(front_idx)} 个解，按 {obj1} 升序）：")
        print(f"  {'iter':>4}  {obj1:>18}  {obj2:>18}  {'B:F':>6}  {'RR':>6}")
        print("  " + "-" * 60)
        for i in front_idx:
            dv = dvs[i]
            bf = next((v for k, v in dv.items() if k.endswith("B:F")), None)
            rr = next((v for k, v in dv.items() if k.endswith("BASIS_RR")), None)
            bf_s = f"{bf:.4f}" if bf is not None else "N/A"
            rr_s = f"{rr:.4f}" if rr is not None else "N/A"
            print(f"  {iters[i]:>4}  {xs_raw[i]:>18.2f}  {ys_raw[i]:>18.2f}"
                  f"  {bf_s:>6}  {rr_s:>6}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从 simulation.db 绘制多目标 Pareto 图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("db", help="simulation.db 路径")
    p.add_argument("--obj1",          default="ADN_FLOW",
                   help="X 轴目标名称（默认 ADN_FLOW）")
    p.add_argument("--obj2",          default="REB_DUTY",
                   help="Y 轴目标名称（默认 REB_DUTY）")
    p.add_argument("--obj1-src",      default="objectives",
                   dest="obj1_src",
                   help="obj1 数据来源（默认 objectives；或 block:NAME:OUTPUT / stream:NAME:FIELD）")
    p.add_argument("--obj2-src",      default="objectives",
                   dest="obj2_src",
                   help="obj2 数据来源（默认 objectives；或 block:NAME:OUTPUT / stream:NAME:FIELD）")
    p.add_argument("--obj1-minimize", default="false",
                   dest="obj1_minimize",
                   help="obj1 是否最小化（true/false，默认 false 即最大化）")
    p.add_argument("--obj2-minimize", default="true",
                   dest="obj2_minimize",
                   help="obj2 是否最小化（true/false，默认 true）")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="保存图片路径（不指定则弹窗显示）")
    p.add_argument("--dpi",  default=150, type=int,
                   help="保存分辨率（默认 150）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot(
        db_path=args.db,
        obj1=args.obj1,
        obj2=args.obj2,
        obj1_src=args.obj1_src,
        obj2_src=args.obj2_src,
        obj1_minimize=args.obj1_minimize.lower() == "true",
        obj2_minimize=args.obj2_minimize.lower() == "true",
        save_path=args.save,
        dpi=args.dpi,
    )
