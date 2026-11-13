"""Plain PowerShell/terminal layout for structured answers. No domain analysis."""
from __future__ import annotations

import re
import shutil
import unicodedata

from src.agents.answer_content import AnswerBlock


def clean(text):
    text = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|$))", "", str(text))
    return "".join(c for c in text if c in "\n\t" or unicodedata.category(c) not in {"Cc", "Cs"}).replace("\t", "    ")


def cell_width(text):
    return sum(0 if unicodedata.combining(c) or unicodedata.category(c) == "Cf"
               else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in clean(text))


def wrap(text, width, prefix="", continuation=None):
    """Wrap by display cells, preserving CJK, combining accents and indentation."""
    continuation = " " * cell_width(prefix) if continuation is None else continuation
    output = []
    for logical in clean(text).split("\n"):
        tokens = re.findall(r"[A-Za-z0-9_./:+%\\\-]+|[^\x00-\x7f]|.", logical)
        # Keep combining accents with their base character.
        clusters = []
        for token in tokens:
            if unicodedata.combining(token[0]) and clusters:
                clusters[-1] += token
            else:
                clusters.append(token)
        limit = width - max(cell_width(prefix), cell_width(continuation))
        tokens = []
        for token in clusters:
            if cell_width(token) > limit:
                tokens.extend(token)
            else:
                tokens.append(token)
        indent, parts = prefix, []
        closing = set("，。；：？！、）》】」』,.!?;:%")
        opening = set("（《【「『(")
        for token in tokens:
            if cell_width(indent + "".join(parts) + token) > width and parts:
                carry = []
                if token[0] in closing or parts[-1][-1] in opening:
                    # Carry one preceding token when it fits, avoiding an
                    # isolated punctuation mark at the start of a line.
                    if cell_width(parts[-1] + token) <= width - cell_width(continuation):
                        carry = [parts.pop()]
                if parts:
                    output.append(indent + "".join(parts).rstrip())
                indent, parts = continuation, carry
            parts.append(token)
        output.append((indent + "".join(parts)).rstrip())
    return output


def value_text(value):
    if value is None:
        return "未提供"
    if type(value) is bool:
        return "是" if value else "否"
    if isinstance(value, float):
        # Preserve small changes; never round a scientific result to zero.
        return format(value, ".12g")
    return clean(value)


class TerminalRenderer:
    def __init__(self, width=None):
        self.width = width
        self.full_values = {}

    def _shorten(self, text):
        # Aspen paths are tokens, not Windows filesystem paths with spaces.
        pattern = r"\\Data\\[^\s，。；、）)\]`]+"
        def replace(match):
            path = match.group()
            if cell_width(path) <= min(36, max(12, self._width // 2)):
                return path
            key = next((k for k, v in self.full_values.items() if v == path), None)
            if key is None:
                key = f"p{len(self.full_values) + 1}"
                self.full_values[key] = path
            parts = path.split("\\")
            leaf = parts[-1]
            # A long node name must not defeat path abbreviation.
            while cell_width(leaf) > max(4, self._width // 3):
                leaf = leaf[1:]
            return f"…/{leaf} [{key}]"
        return re.sub(pattern, replace, clean(text))

    def render(self, answer, *, details=False):
        # Read terminal width on every render, including after resize.
        self._width = max(8, int(self.width or shutil.get_terminal_size((88, 24)).columns) - 1)
        self.full_values = {}
        lines = []
        for block in answer.content:
            if details or not block.detail:
                lines.extend(self._block(block, full=details))
        approval = answer.approval_request
        if approval:
            p = approval.proposal
            lines.extend(self._block(AnswerBlock("bullets", title="审批",
                items=[f"状态：{approval.status}", f"动作：{p.action_type}；对象：{p.target}",
                       *([p.reason] if p.reason else []), *([approval.message] if approval.message else []),
                       *p.risks, *(["输入“确认”执行；修改需求后需重新审批。"] if approval.status == "pending" else [])]), full=details))
        for title, values in (("假设", answer.assumptions), ("限制", answer.uncertainties)):
            # Detailed multiline diagnostics stay in the detail view.
            brief = [str(x) if details else str(x).splitlines()[0] for x in values]
            brief = list(dict.fromkeys(x for x in brief if x and x not in answer.text
                                       and not any(x == b.text for b in answer.blocks)))
            if brief:
                lines.extend(self._block(AnswerBlock("bullets", title=title, items=brief), full=details))
        if answer.recommendations and not approval:
            for recommendation in answer.recommendations:
                if isinstance(recommendation, str):
                    lines.extend(self._block(AnswerBlock("next_steps", title="建议：", items=[recommendation]), full=details))
                elif isinstance(recommendation, dict) and isinstance(recommendation.get("next_step"), str):
                    lines.extend(self._block(AnswerBlock("next_steps", title="建议：", items=[recommendation["next_step"]]), full=details))
        if details:
            for title, value in (("详细证据", [e.to_dict() for e in answer.evidence]),
                                 ("事实记录", answer.facts), ("详情", answer.details),
                                 ("建议记录", answer.recommendations)):
                if value:
                    lines.extend([title, *self._tree(value)])
            if approval:
                lines.extend(["审批完整预览", *self._tree(approval.to_dict())])
        else:
            if (answer.details or answer.facts or answer.evidence or answer.recommendations
                    or approval or any(b.detail for b in answer.blocks)
                    or any("\n" in x for x in answer.uncertainties)):
                lines.extend(wrap("详情：/details；审计导出：/export 文件.json", self._width))
            if self.full_values:
                lines.extend(wrap("完整路径：/value p1（编号见正文）", self._width))
        return "\n".join(lines).rstrip()

    def render_value(self, key):
        # Width wrapping remains lossless; source is also available in JSON export.
        width = max(8, int(self.width or shutil.get_terminal_size((88, 24)).columns) - 1)
        return "\n".join(wrap(self.full_values[key], width))

    def _block(self, block, *, full=False):
        fmt = clean if full else self._shorten
        lines = []
        labels = {"observed": "证据记录", "mechanism": "机理解释", "hypothesis": "推测", "insufficient": "数据不足"}
        title = " · ".join(x for x in (block.title, labels.get(block.epistemic, "")) if x)
        if title:
            lines.extend(wrap(fmt(title), self._width))
        if block.text:
            lines.extend(wrap(fmt(block.text), self._width))
        if block.kind in {"bullets", "next_steps"}:
            for item in block.items:
                lines.extend(wrap(fmt(item), self._width, prefix="  • "))
        if block.kind == "metrics":
            for metric in block.metrics:
                unit = metric["unit"]
                text = (f"{metric['label']}：{value_text(metric['value'])} {unit}；"
                        f"基准 {value_text(metric['baseline'])} {unit}；"
                        f"变化量 {value_text(metric['delta'])} {unit}；比较范围：{metric['scope']}")
                lines.extend(wrap(fmt(text), self._width, prefix="  "))
        if block.kind in {"comparison", "table"}:
            headers = [fmt(c) for c in block.columns]
            rows = [[fmt(value_text(v)) for v in row] for row in block.rows]
            widths = [max(cell_width(r[i]) for r in [headers, *rows]) for i in range(len(headers))]
            readable = (sum(widths) + 3 * (len(widths) - 1) <= self._width
                        and len(rows) <= 12 and all("\n" not in c for r in [headers, *rows] for c in r))
            if readable:
                def line(row):
                    return "   ".join(c + " " * (n - cell_width(c)) for c, n in zip(row, widths)).rstrip()
                lines.extend([line(headers), "   ".join("─" * n for n in widths), *map(line, rows)])
            else:
                for i, row in enumerate(rows, 1):
                    lines.extend(wrap(f"{i}.", self._width))
                    for header, cell in zip(headers, row):
                        lines.extend(wrap(f"{header}：{cell}", self._width, prefix="  "))
        if block.evidence_refs:
            lines.extend(wrap("证据：" + "；".join(block.evidence_refs), self._width, prefix="  "))
        return [*lines, ""]

    def _tree(self, value, depth=0):
        """Lossless, bounded-indent inspection of arbitrary audit payloads."""
        prefix = " " * min(depth * 2, self._width // 3)
        lines = []
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list, tuple)):
                    lines.extend(wrap(f"{key}：", self._width, prefix=prefix))
                    lines.extend(self._tree(item, depth + 1))
                else:
                    lines.extend(wrap(f"{key}：{value_text(item)}", self._width, prefix=prefix))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value, 1):
                lines.extend(wrap(f"{i}.", self._width, prefix=prefix))
                lines.extend(self._tree(item, depth + 1))
        else:
            lines.extend(wrap(value_text(value), self._width, prefix=prefix))
        return lines
