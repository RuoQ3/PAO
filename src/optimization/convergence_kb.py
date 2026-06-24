"""
convergence_kb.py — Aspen Plus 收敛失败模式知识库。

职责
----
提供 7 种常见 Aspen Plus 收敛失败模式的结构化知识，以及：

  match_failure_patterns(error_texts)
      对 runner._read_history_diagnostics() 返回的原始 .his 错误文本做
      关键词匹配，返回结构化 ConvergenceDiagnosis 列表（中文描述 + 修复建议）。

  format_kb_summary()
      返回 KB 的简短摘要文本，用于注入 boundary_advisor 的 SYSTEM_PROMPT，
      让 LLM 在推荐搜索边界时感知高风险变量类型。

设计原则
--------
- 纯 Python，无 I/O，无网络，无 COM 依赖，任何环境都能导入。
- 关键词匹配大小写不敏感（统一转大写后比较）。
- 匹配失败（空输入、无命中、任何异常）静默返回空列表/空字符串，
  绝不向调用方抛出异常——诊断功能不能破坏主流程。

参考来源
--------
aspen-mcp/src/aspen_mcp/knowledge/convergence.py（FailureMode 结构 + 症状文本）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailurePattern:
    """
    一种已知的 Aspen Plus 收敛失败模式。

    Attributes
    ----------
    pattern_id:
        唯一标识，如 "tear_stream"、"column_design"。
    title:
        英文标题（来自 aspen-mcp 原始定义，便于对照）。
    description:
        中文一句话描述，用于日志和用户提示。
    symptoms_keywords:
        触发关键词元组（全大写）。对 .his 原始文本做 upper() 后做子串匹配。
        关键词应简短且有区分度，避免误触发。
    fixes:
        中文修复建议列表，首条为最优先建议。
    severity:
        严重程度 "high" | "medium" | "low"，供排序或过滤使用。
    """
    pattern_id: str
    title: str
    description: str
    symptoms_keywords: tuple[str, ...]
    fixes: tuple[str, ...]
    severity: str = "medium"


@dataclass
class ConvergenceDiagnosis:
    """
    单次匹配的诊断结果。

    Attributes
    ----------
    pattern_id:
        命中的模式 ID。
    description:
        中文描述。
    matched_keywords:
        本次在 error_texts 中命中的关键词列表（用于调试说明）。
    fixes:
        修复建议列表（来自 FailurePattern.fixes）。
    severity:
        严重程度。
    """
    pattern_id: str
    description: str
    matched_keywords: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    severity: str = "medium"


# ---------------------------------------------------------------------------
# 7 种失败模式定义
# ---------------------------------------------------------------------------

FAILURE_PATTERNS: tuple[FailurePattern, ...] = (
    FailurePattern(
        pattern_id="tear_stream",
        title="Tear Stream Not Converging",
        description="Tear stream（循环流股）迭代不收敛",
        symptoms_keywords=(
            "NOT CONVERGED TO TOLERANCE",
            "MAXIMUM NUMBER OF ITERATIONS",
            "TEAR STREAM",
            "CONVERGENCE ITERATIONS COMPLETED",
            "WEGSTEIN",
            "RECYCLE",
        ),
        fixes=(
            "增大收敛迭代次数上限（默认 30 → 尝试 200~500）",
            "为 Tear stream 提供合理的初始估值",
            "改用 Broyden 或 Direct 求解器代替 Wegstein",
            "启用 Wegstein 阻尼（damping factor 0.3~0.7）",
        ),
        severity="high",
    ),
    FailurePattern(
        pattern_id="bad_initial_guess",
        title="Poor Initial Estimates",
        description="初始估值偏离可行区域导致无法收敛",
        symptoms_keywords=(
            "INITIAL ESTIMATE",
            "INITIAL GUESS",
            "DOES NOT CONVERGE",
            "STARTING POINT",
            "DIVERGE",
        ),
        fixes=(
            "移除用户指定的初始估值，让 Aspen 自动生成默认值",
            "检查温度/压力的初始估值是否在操作范围内",
            "先运行简化模型，再逐步增加复杂度",
        ),
        severity="medium",
    ),
    FailurePattern(
        pattern_id="chemistry_reaction",
        title="Chemistry / Reaction Issues",
        description="化学反应/电解质计算失败或结果不合理",
        symptoms_keywords=(
            "REACTION",
            "STOICHIOMETRY",
            "EQUILIBRIUM CONSTANT",
            "ELECTROLYTE",
            "ION PAIR",
            "SALT FORMATION",
        ),
        fixes=(
            "检查反应化学计量系数的摩尔守恒",
            "对电解质体系，验证离子对和盐沉淀设置",
            "确认所有涉及组分均已定义",
        ),
        severity="high",
    ),
    FailurePattern(
        pattern_id="property_method",
        title="Property Method Inappropriate",
        description="热力学物性方法不适用于当前体系",
        symptoms_keywords=(
            "PROPERTY METHOD",
            "K-VALUE",
            "ENTHALPY CALCULATION",
            "PHASE EQUILIBRIUM",
            "EQUATION OF STATE",
            "ACTIVITY COEFFICIENT",
            "INCONSISTENT PROPERTIES",
        ),
        fixes=(
            "切换为更适合该体系的物性方法（如极性体系用 NRTL/UNIQUAC）",
            "补充二元交互参数（Binary Interaction Parameters）",
            "检查操作温度/压力是否超出物性方法的有效范围",
        ),
        severity="high",
    ),
    FailurePattern(
        pattern_id="column_design",
        title="Column / Distillation Design Issues",
        description="精馏塔设计规格不可行或塔模型不收敛",
        symptoms_keywords=(
            "RADFRAC",
            "DESIGN SPEC",
            "ALGORITHM FAILED",
            "COLUMN DRIES UP",
            "COLUMN NOT IN MASS BALANCE",
            "REFLUX RATIO",
            "TRAY SIZING",
            "COLUMN ALGORITHMS",
        ),
        fixes=(
            "放宽或取消 Design Specification 约束条件",
            "增加理论板数，确认进料板位置合理",
            "先用 DSTWU 估算再切换到 RadFrac",
            "检查产品流量之和是否等于进料流量",
        ),
        severity="medium",
    ),
    FailurePattern(
        pattern_id="stream_flash",
        title="Stream Flash Calculation Failure",
        description="流股闪蒸计算失败（温度/压力超出范围或相态判断错误）",
        symptoms_keywords=(
            "FLASH",
            "TEMPERATURE OUT OF",
            "VAPOR FRACTION",
            "BUBBLE POINT",
            "DEW POINT",
            "THREE-PHASE",
            "TWO LIQUID",
        ),
        fixes=(
            "检查流股温度/压力是否在物理合理范围内",
            "对含水/烃类体系，将闪蒸类型切换为三相（VLL）",
            "为流股提供合理的温度/压力初始估值",
        ),
        severity="medium",
    ),
    FailurePattern(
        pattern_id="solver_options",
        title="Solver / Convergence Block Options",
        description="求解器配置不当导致收敛缓慢或震荡",
        symptoms_keywords=(
            "CONVERGENCE BLOCK",
            "OSCILLAT",
            "TOLERANCE",
            "BROYDEN",
            "DIRECT SUBSTITUTION",
            "CONVERGENCE OPTIONS",
        ),
        fixes=(
            "将不稳定循环的求解器从 Wegstein 切换为 Direct",
            "增大 Wegstein 阻尼系数",
            "对耦合循环使用单个收敛块统一求解",
        ),
        severity="low",
    ),
)

# 快速查找字典：pattern_id → FailurePattern
_PATTERN_MAP: dict[str, FailurePattern] = {p.pattern_id: p for p in FAILURE_PATTERNS}


# ---------------------------------------------------------------------------
# 公共函数
# ---------------------------------------------------------------------------

def match_failure_patterns(
    error_texts: list[str],
    max_patterns: int = 3,
) -> list[ConvergenceDiagnosis]:
    """
    对原始 Aspen .his 错误文本做收敛失败模式匹配。

    Parameters
    ----------
    error_texts:
        来自 runner._read_history_diagnostics() 的原始错误文本列表。
        每条通常形如 "sim.his: SEVERE ERROR IN THE DISTILLATION SECTION"。
    max_patterns:
        最多返回的诊断条数，按命中关键词数降序截取，默认 3。

    Returns
    -------
    list[ConvergenceDiagnosis]
        按命中关键词数降序排列的诊断结果。
        空输入或无命中时返回 []。
        任何异常静默处理，返回 []（诊断不能破坏主流程）。
    """
    if not error_texts:
        return []

    try:
        # 将所有错误文本拼接为一个大写字符串，一次性匹配
        combined = " ".join(error_texts).upper()

        results: list[tuple[int, ConvergenceDiagnosis]] = []

        for pattern in FAILURE_PATTERNS:
            matched = [
                kw for kw in pattern.symptoms_keywords
                if kw in combined
            ]
            if not matched:
                continue

            diag = ConvergenceDiagnosis(
                pattern_id=pattern.pattern_id,
                description=pattern.description,
                matched_keywords=matched,
                fixes=list(pattern.fixes),
                severity=pattern.severity,
            )
            results.append((len(matched), diag))

        # 按命中关键词数降序排列
        results.sort(key=lambda x: x[0], reverse=True)

        return [diag for _, diag in results[:max_patterns]]

    except Exception as exc:  # noqa: BLE001
        _log.debug("convergence_kb.match_failure_patterns: 匹配时出现异常（已忽略）：%s", exc)
        return []


def format_kb_summary() -> str:
    """
    返回知识库的简短摘要文本。

    用途：注入 boundary_advisor 的 SYSTEM_PROMPT，让 LLM 在推荐搜索
    边界时了解常见收敛失败模式，对高风险变量类型取更保守的 k 值。

    格式：每种模式一行，包含 pattern_id、中文描述和首条修复建议。

    Returns
    -------
    str
        多行纯文本，不含 Markdown 格式。
    """
    lines: list[str] = []
    for p in FAILURE_PATTERNS:
        first_fix = p.fixes[0] if p.fixes else "见文档"
        lines.append(f"  [{p.pattern_id}] {p.description} → {first_fix}")
    return "\n".join(lines)
