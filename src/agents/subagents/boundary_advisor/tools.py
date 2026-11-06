"""
tools.py — BoundaryAdvisor agent 的确定性工具函数。

职责（不含任何 LLM 调用，纯计算/解析/文本写回，便于单测）：
  - VarMeta：单个变量的输入元信息容器。
  - build_variables_block：把变量元信息渲染成给 LLM 的文本清单。
  - parse_llm_boundary_json：稳健解析 LLM 返回的 JSON（容忍代码围栏/前后缀）。
  - heuristic_k：无 LLM（缺 key 或调用失败）时的规则兜底，从单位+初始值推断 k。
  - bounds_from_k：把 k_lo/k_hi + 初始值 + 全局边界 换算成最终搜索边界。
  - plan_yaml_edits / apply_yaml_edits：文本级精确改写 YAML 边界,保留全部注释。

设计原则：
  - 所有函数对缺失字段、异常输入都返回安全默认值，绝不抛异常给上层。
  - heuristic_k 与 SYSTEM_PROMPT 的物理规则保持一致，作为 LLM 不可用时的退路，
    保证 boundary_advisor 在无网络/无 key 环境下仍能产出合理边界。
  - YAML 写回用文本级正则替换(非 yaml.dump),只改边界数值行,
    完整保留用户文件里的注释、缩进、MATLAB 对齐说明等。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 输入元信息
# ---------------------------------------------------------------------------

@dataclass
class VarMeta:
    """单个设计变量的输入元信息。

    Attributes
    ----------
    name:
        变量标识符(与 LLM 输出 name 对应,通常是 YAML name 或 aspen_path)。
    initial_value:
        初始收敛解值;None 表示未知(将走最保守处理)。
    unit:
        单位字符串,如 "atm" / "kmol/hr" / "-";未知填 ""。
    semantic_role:
        语义角色,如 "pressure" / "reflux_ratio" / "nstage";未知填 ""。
    var_type:
        "continuous" / "integer" / "derived";用于辅助判断。
    lower_global / upper_global:
        全局物理边界(YAML 原始 lower/upper),用于裁剪最终边界,None 表示无限制。
    """
    name: str
    initial_value: float | None
    unit: str = ""
    semantic_role: str = ""
    var_type: str = "continuous"
    lower_global: float | None = None
    upper_global: float | None = None


# ---------------------------------------------------------------------------
# 渲染给 LLM 的变量清单
# ---------------------------------------------------------------------------

def build_variables_block(variables: list[VarMeta]) -> str:
    """把变量元信息渲染成逐行文本清单,供 USER_TEMPLATE 注入。"""
    lines: list[str] = []
    for v in variables:
        iv = "未知" if v.initial_value is None else f"{v.initial_value:g}"
        unit = v.unit or "-"
        role = v.semantic_role or "未标注"
        rng = ""
        if v.lower_global is not None and v.upper_global is not None:
            rng = f", 全局范围=[{v.lower_global:g}, {v.upper_global:g}]"
        lines.append(
            f"- name={v.name}, initial_value={iv}, unit={unit}, "
            f"semantic_role={role}, type={v.var_type}{rng}"
        )
    return "\n".join(lines) if lines else "(无变量)"


# ---------------------------------------------------------------------------
# 解析 LLM JSON
# ---------------------------------------------------------------------------

def parse_llm_boundary_json(text: str) -> dict | None:
    """稳健解析 LLM 返回的边界 JSON。

    容忍以下情况：
      - 回复被 ```json ... ``` 代码围栏包裹
      - JSON 前后有少量散文
    解析失败返回 None(由上层降级到 heuristic_k)。
    """
    if not text or not text.strip():
        return None

    # 去掉 markdown 代码围栏
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        # 截取第一个 { 到最后一个 } 之间的内容
        lo = cleaned.find("{")
        hi = cleaned.rfind("}")
        if lo != -1 and hi != -1 and hi > lo:
            cleaned = cleaned[lo:hi + 1]

    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        _log.warning("boundary_advisor：LLM JSON 解析失败：%s", exc)
        return None

    if not isinstance(obj, dict) or "variables" not in obj:
        _log.warning("boundary_advisor：LLM JSON 缺少 'variables' 字段。")
        return None
    return obj


def extract_k_map(parsed: dict) -> dict[str, tuple[float, float]]:
    """从解析后的 JSON 提取 {name: (k_lo, k_hi)}，并裁剪到 [1.2, 5.0]。

    非法/缺失项跳过,由上层用 heuristic_k 补齐。
    """
    result: dict[str, tuple[float, float]] = {}
    for item in parsed.get("variables", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        try:
            k_lo = float(item.get("k_lo"))
            k_hi = float(item.get("k_hi"))
        except (TypeError, ValueError):
            continue
        k_lo = _clamp(k_lo, 1.2, 5.0)
        k_hi = _clamp(k_hi, 1.2, 5.0)
        result[str(name)] = (k_lo, k_hi)
    return result


# ---------------------------------------------------------------------------
# 规则兜底（无 LLM 时使用，与 SYSTEM_PROMPT 物理规则一致）
# ---------------------------------------------------------------------------

def heuristic_k(v: VarMeta) -> tuple[float, float, str]:
    """无 LLM 时,从单位+初始值规则推断 (k_lo, k_hi, reason)。

    与 SYSTEM_PROMPT 的物理规则保持一致,保证降级路径也产出合理边界。
    """
    unit = (v.unit or "").lower()
    role = (v.semantic_role or "").lower()
    iv = v.initial_value

    # 压力：从单位 + 初始值量级判断真空/常压/加压
    if "atm" in unit or "bar" in unit or "pa" in unit or "psi" in unit or "pressure" in role:
        if iv is not None and _to_atm_like(iv, unit) < 0.5:
            return 2.0, 2.5, "压力单位但初值远低于常压,判真空操作,高敏感,小 k 锁初值"
        return 2.5, 3.0, "压力变量,中等 k"

    # 回流比
    if "reflux" in role or "basis_rr" in v.name.lower() or "rr" == role:
        if iv is not None and iv < 1.0:
            return 2.0, 2.5, "低回流比系统,对分离/能耗高敏感,小 k"
        return 2.5, 3.0, "回流比变量,中等偏小 k"

    # 塔板数
    if "nstage" in v.name.lower() or "nstage" in role or "stage" in role and v.var_type == "integer":
        return 3.0, 3.0, "塔板数对收敛相对迟钝,较大 k 保留探索"

    # 进料板比例 / 位置
    if "feed" in role or "frac" in v.name.lower() or "feed_stage" in role:
        return 2.5, 2.5, "进料位置受塔板数约束,中等 k"

    # 流量（杠杆变量,保守）
    if "flow" in role or "kmol" in unit or "kg/hr" in unit or "mol/s" in unit or "sol" in v.name.lower():
        return 2.0, 2.5, "流量常为可行域杠杆变量,保守小 k"

    # 温度
    if "temperature" in role or unit in ("c", "k", "f", "°c", "°k", "°f"):
        return 2.5, 3.0, "温度变量,中等 k"

    # 默认：中等保守
    return 2.5, 2.5, "未识别类型,取中等保守 k"


# ---------------------------------------------------------------------------
# k → 最终边界
# ---------------------------------------------------------------------------

def bounds_from_k(
    v: VarMeta,
    k_lo: float,
    k_hi: float,
) -> tuple[float, float] | None:
    """把 (k_lo, k_hi) + 初始值 换算成搜索边界,并裁剪到全局物理边界内。

    两种锚定模式：
      A. 乘除式(默认,适合跨量级正量如压力/流量/塔板数)：
         [初始值 / k_lo, 初始值 × k_hi]。
      B. 对称偏移式(用于 derived/归一化 [0,1] 变量,或初始值<=0 无法乘除)：
         以初始值为中心 ± 半窗,半窗 = (k-1)/k 映射到全局宽度的一部分,
         结果裁剪到全局边界(对 frac 即 [0,1])。

    选择依据：
      - var_type == "derived",或全局边界落在 [0,1] 内(归一化变量)→ 用对称偏移。
      - 初始值 <= 0 或缺失 → 用对称偏移(以全局中点或初值为中心)。
      - 其他正量 → 乘除式。
    结果一律用 lower_global/upper_global 裁剪;非法时回退全局边界或 None。

    Returns
    -------
    (lo, hi) 或 None（无法确定时,调用方应保留该变量原全局边界）。
    """
    iv = v.initial_value
    glo, ghi = v.lower_global, v.upper_global
    is_normalized = (
        v.var_type == "derived"
        or (glo is not None and ghi is not None and glo >= 0.0 and ghi <= 1.0 + 1e-9)
    )

    # ── 模式 B：对称偏移(derived/归一化,或初值<=0)──────────────────────
    if is_normalized or iv is None or iv <= 0:
        if glo is None or ghi is None or glo >= ghi:
            return None
        width = ghi - glo
        center = iv if (iv is not None and glo <= iv <= ghi) else (glo + ghi) / 2.0
        # k 越大窗越宽：用 max(k_lo,k_hi) 映射到全局宽度的比例(k=1.2→~0.17,k=5→~0.8)
        k = max(k_lo, k_hi)
        frac = min(0.8, (k - 1.0) / k)
        half = width * frac / 2.0
        lo = max(glo, center - half)
        hi = min(ghi, center + half)
        if lo >= hi:
            return (glo, ghi)
        return (lo, hi)

    # ── 模式 A：乘除式(跨量级正量)────────────────────────────────────
    lo = iv / max(k_lo, 1e-9)
    hi = iv * max(k_hi, 1e-9)

    # 裁剪到全局物理边界
    if glo is not None:
        lo = max(lo, glo)
    if ghi is not None:
        hi = min(hi, ghi)

    if lo >= hi:
        if glo is not None and ghi is not None and glo < ghi:
            return (glo, ghi)
        return None
    return (lo, hi)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _to_atm_like(value: float, unit: str) -> float:
    """把压力粗略换算到 atm 量级,仅用于真空判断(不需要高精度)。"""
    u = unit.lower()
    if "kpa" in u:
        return value / 101.325
    if "mpa" in u:
        return value * 1000 / 101.325
    if "pa" in u and "kpa" not in u and "mpa" not in u:
        return value / 101325.0
    if "bar" in u:
        return value / 1.01325
    if "psi" in u:
        return value / 14.696
    # atm 或未知:按原值
    return value


# ---------------------------------------------------------------------------
# YAML 文本级写回（保留注释）
# ---------------------------------------------------------------------------

@dataclass
class YamlEdit:
    """一次将要应用到 YAML 的边界改动(供确认展示)。"""
    name: str
    field_lo: str          # "lower_bound" 或 "lo_frac"
    field_hi: str          # "upper_bound" 或 "hi_frac"
    old_lo: str            # 原值文本("?" 表示未找到该行)
    old_hi: str
    new_lo: str
    new_hi: str
    is_integer: bool
    line_lo: int           # 1-based 行号,0 表示未找到
    line_hi: int


def _round_for_field(value: float, is_integer: bool) -> str:
    """把边界值格式化为写回文本：整数变量取整,其余保留 6 位有效数字。"""
    if is_integer:
        return str(int(round(value)))
    return f"{value:.6g}"


def plan_yaml_edits(
    yaml_text: str,
    bounds_by_name: dict[str, tuple[float, float]],
    var_types: dict[str, str],
    integer_names: set[str],
) -> list[YamlEdit]:
    """规划文本级 YAML 边界改动,不修改文本,只返回改动清单。

    对每个变量,在其 `- name: <name>` 块内定位 lower_bound/upper_bound
    (continuous/integer) 或 lo_frac/hi_frac (derived) 行,生成 YamlEdit。

    Parameters
    ----------
    yaml_text:        原始 YAML 全文。
    bounds_by_name:   {变量名: (新下界, 新上界)}。
    var_types:        {变量名: "continuous"|"integer"|"derived"}。
    integer_names:    需要取整写回的变量名集合(整数变量)。

    Returns
    -------
    list[YamlEdit]   每个变量一条(找不到的行 line=0,old="?")。
    """
    lines = yaml_text.splitlines()
    edits: list[YamlEdit] = []

    for name, (lo, hi) in bounds_by_name.items():
        vtype = var_types.get(name, "continuous")
        is_int = name in integer_names
        field_lo, field_hi = (
            ("lo_frac", "hi_frac") if vtype == "derived"
            else ("lower_bound", "upper_bound")
        )
        # derived 的 frac 不取整;其余按 integer_names 决定
        is_int_eff = is_int and vtype != "derived"

        blk_start, blk_end = _find_var_block(lines, name)
        old_lo = old_hi = "?"
        line_lo = line_hi = 0
        if blk_start >= 0:
            line_lo, old_lo = _find_field(lines, blk_start, blk_end, field_lo)
            line_hi, old_hi = _find_field(lines, blk_start, blk_end, field_hi)

        edits.append(YamlEdit(
            name=name, field_lo=field_lo, field_hi=field_hi,
            old_lo=old_lo, old_hi=old_hi,
            new_lo=_round_for_field(lo, is_int_eff),
            new_hi=_round_for_field(hi, is_int_eff),
            is_integer=is_int_eff, line_lo=line_lo, line_hi=line_hi,
        ))
    return edits


def apply_yaml_edits(yaml_text: str, edits: list[YamlEdit]) -> tuple[str, list[str]]:
    """把 plan_yaml_edits 的改动应用到文本,返回 (新文本, 跳过说明列表)。

    只替换数值,保留该行原有的行内注释和缩进。找不到的行跳过并记录。
    """
    lines = yaml_text.splitlines(keepends=True)
    skipped: list[str] = []

    # 收集 (行号, 字段, 新值)；按行号倒序应用,避免行号漂移(本实现按行号定位,顺序无关)
    for e in edits:
        for ln, field, new_val, old_val in (
            (e.line_lo, e.field_lo, e.new_lo, e.old_lo),
            (e.line_hi, e.field_hi, e.new_hi, e.old_hi),
        ):
            if ln <= 0:
                skipped.append(f"{e.name}.{field}: 未在 YAML 中找到该行,已跳过")
                continue
            idx = ln - 1
            if idx >= len(lines):
                skipped.append(f"{e.name}.{field}: 行号越界,已跳过")
                continue
            lines[idx] = _replace_value_in_line(lines[idx], field, new_val)

    return "".join(lines), skipped


# ---------------------------------------------------------------------------
# YAML 文本定位内部函数
# ---------------------------------------------------------------------------

def _find_var_block(lines: list[str], name: str) -> tuple[int, int]:
    """定位 `- name: <name>` 所在块的 [start, end) 行索引(0-based,end 不含)。

    块结束 = 下一个 `- name:`(同级列表项)或缩进回退到列表项级别之前。
    找不到返回 (-1, -1)。
    """
    name_re = re.compile(r"^\s*-\s*name:\s*([^\s#]+)")
    start = -1
    for i, line in enumerate(lines):
        m = name_re.match(line)
        if m and m.group(1).strip().strip('"').strip("'") == name:
            start = i
            break
    if start < 0:
        return -1, -1
    # 找块结束：下一个 `- name:` 或下一个顶层 key(无缩进且含冒号)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if name_re.match(line):
            end = j
            break
        # 顶层 key(行首非空白且形如 key:)视为块结束
        if line and not line[0].isspace() and re.match(r"^[A-Za-z_]\w*\s*:", line):
            end = j
            break
    return start, end


def _find_field(
    lines: list[str], start: int, end: int, field: str
) -> tuple[int, str]:
    """在 [start, end) 内找 `<field>: <value>` 行,返回 (1-based行号, 原值文本)。

    找不到返回 (0, "?")。
    """
    field_re = re.compile(rf"^(\s*){re.escape(field)}\s*:\s*([^\s#]+)")
    for i in range(start, end):
        m = field_re.match(lines[i])
        if m:
            return i + 1, m.group(2).strip()
    return 0, "?"


def _replace_value_in_line(line: str, field: str, new_val: str) -> str:
    """替换某行 `<field>: <old>` 的值为 new_val,保留缩进与行内注释。"""
    # 拆出行内注释(# 之后),只在值部分替换
    nl = ""
    if line.endswith("\n"):
        nl = "\n"
        body = line[:-1]
    else:
        body = line
    comment = ""
    # 简单处理:# 前面有空格才视为行内注释,避免误伤值里的 #(边界值不会含 #)
    hash_idx = body.find("#")
    if hash_idx != -1:
        comment = body[hash_idx:]
        body = body[:hash_idx]
    m = re.match(rf"^(\s*{re.escape(field)}\s*:\s*)(\S+)(\s*)$", body)
    if not m:
        return line  # 形态不符,保守不改
    prefix, _old, trailing = m.group(1), m.group(2), m.group(3)
    if comment:
        # 保留原注释前的对齐空格(body 已含到注释前)
        return f"{prefix}{new_val}{trailing}{comment}{nl}"
    return f"{prefix}{new_val}{trailing}{nl}"
