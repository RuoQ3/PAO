"""UI-independent answer content. This module validates data; it never reasons."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any


KINDS = {"paragraph", "heading", "bullets", "metrics", "comparison", "table",
         "limitation", "next_steps", "evidence"}
EPISTEMIC = {"", "observed", "mechanism", "hypothesis", "insufficient"}


def scalar(value):
    return value is None or type(value) in (str, bool, int, float) and (
        not isinstance(value, float) or math.isfinite(value))


@dataclass
class AnswerBlock:
    kind: str
    text: str = ""
    title: str = ""
    items: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    epistemic: str = ""
    detail: bool = False

    def __post_init__(self):
        if self.kind not in KINDS or self.epistemic not in EPISTEMIC or type(self.detail) is not bool:
            raise ValueError("invalid answer block kind, epistemic status or detail flag")
        if not isinstance(self.text, str) or not isinstance(self.title, str):
            raise ValueError("block text/title must be strings")
        for values in (self.items, self.columns, self.evidence_refs):
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError("block lists must contain nonempty strings")
        if not isinstance(self.rows, list) or any(not isinstance(r, list) or len(r) != len(self.columns)
                                                or not all(scalar(v) for v in r) for r in self.rows):
            raise ValueError("table rows must match columns and contain scalar values")
        if not isinstance(self.metrics, list):
            raise ValueError("metrics must be a list")
        required = {"label", "value", "unit", "baseline", "delta", "scope"}
        for metric in self.metrics:
            if not isinstance(metric, dict) or set(metric) != required:
                raise ValueError("metric requires label, value, unit, baseline, delta and scope")
            if not all(isinstance(metric[k], str) and metric[k].strip() for k in ("label", "unit", "scope")):
                raise ValueError("metric labels, units and comparison scope are required")
            if not all(scalar(metric[k]) for k in ("value", "baseline", "delta")):
                raise ValueError("metric values must be finite scalars or null")
        allowed = {"text", "title", "evidence_refs", "epistemic", "detail"}
        allowed |= {"items"} if self.kind in {"bullets", "next_steps"} else set()
        allowed |= {"columns", "rows"} if self.kind in {"table", "comparison"} else set()
        allowed |= {"metrics"} if self.kind == "metrics" else set()
        for key in ("items", "columns", "rows", "metrics"):
            if key not in allowed and getattr(self, key):
                raise ValueError(f"{key} is not valid for {self.kind}")
        if self.kind in {"table", "comparison"} and (not self.columns or not self.rows):
            raise ValueError("table requires columns and rows")
        if self.kind == "metrics" and not self.metrics:
            raise ValueError("metrics block is empty")
        if self.kind in {"bullets", "next_steps"} and not self.items:
            raise ValueError("list block is empty")
        if self.kind in {"paragraph", "heading", "limitation", "evidence"} and not self.text.strip():
            raise ValueError("text block is empty")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("answer block must be an object")
        return cls(**value)


def blocks_from(values):
    if not isinstance(values, list):
        raise ValueError("blocks must be an array")
    return [v if isinstance(v, AnswerBlock) else AnswerBlock.from_dict(v) for v in values]


def legacy_blocks(text):
    """Lossless structural migration of old prose; no summarization or inference.

    Markdown is accepted only here at the compatibility boundary, never used
    as the renderer's core representation.
    """
    lines = str(text).splitlines()
    blocks, paragraph = [], []
    def flush():
        if paragraph:
            blocks.append(AnswerBlock("paragraph", text="\n".join(paragraph)))
            paragraph.clear()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if i + 1 < len(lines) and "|" in line and re.fullmatch(r"[\s|:\-]+", lines[i + 1]) and "-" in lines[i + 1]:
            headers = [x.strip() for x in line.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                row = [x.strip() for x in lines[j].strip().strip("|").split("|")]
                if len(row) != len(headers):
                    break
                rows.append(row)
                j += 1
            if rows and all(headers):
                flush()
                blocks.append(AnswerBlock("table", columns=headers, rows=rows))
                i = j
                continue
        if not line:
            flush()
        elif re.match(r"^#{1,6}\s", line):
            flush()
            blocks.append(AnswerBlock("heading", text=re.sub(r"^#+\s+", "", line)))
        elif re.match(r"^[-*+]\s", line):
            flush()
            blocks.append(AnswerBlock("bullets", items=[line[2:]]))
        else:
            paragraph.append(line)
        i += 1
    flush()
    return blocks


CONTENT_CONTRACT = '''Return strict JSON: text (a nonempty standalone answer to the
current question, answer first), uncertainties (string array), confidence (0..1),
blocks (array, may be empty for a simple reply). No fixed sections or answer
templates. Choose only blocks needed for this question; do not repeat text.
Each block has kind: paragraph|heading|bullets|metrics|comparison|table|limitation|
next_steps|evidence. Optional fields: text, title, epistemic
(observed|mechanism|hypothesis|insufficient), detail (boolean), evidence_refs
(JSON pointers into the supplied evidence, e.g. /result/status).
bullets/next_steps use items (strings); table/comparison use columns (strings)
and rows (arrays of scalar cells); metrics uses metrics, an array of objects
with exactly label, value, unit, baseline, delta, scope. Values may be null when
unknown; explicitly say units or comparison scope are unknown when missing.
Never return raw dict/JSON or Markdown tables inside text. Use plain prose.
Put key evidence after the answer; long diagnostics and full paths belong in
detail blocks. The summary must explain what happened, failures and limits
without requiring a log. Distinguish observed measurements, mechanism,
hypotheses and insufficient data. Numerical comparisons must include actual
values, units, reference/baseline, deltas and tested scope; normalized
sensitivity is not an engineering effect or statistical significance.
Do not describe tiny normalized effects as substantial without actual changes
and an evidence-backed engineering threshold. Do not invent source pointers.
For execution analysis, every substantive block must cite JSON pointers.
Metrics blocks must copy the full label/value/unit/baseline/delta/scope tuple
from the referenced numerical evidence. Explain engineering mechanisms as
inference, never as mechanically verified fact. Confidence is an uncalibrated
model self-report, not a measured probability.
'''
