"""Optional Markdown export of the same content consumed by terminal/web UIs."""
from src.reporting.answer_terminal import TerminalRenderer, value_text


def markdown_answer(answer):
    lines = []
    def escape(value):
        return value_text(value).replace("|", "\\|").replace("\n", "<br>")
    for block in answer.content:
        if block.title:
            lines.extend(["## " + block.title, ""])
        if block.epistemic:
            lines.extend(["*" + {"observed": "证据记录", "mechanism": "机理解释", "hypothesis": "推测", "insufficient": "数据不足"}[block.epistemic] + "*", ""])
        if block.text:
            lines.extend([("## " if block.kind == "heading" else "") + block.text, ""])
        lines.extend("- " + item for item in block.items)
        for m in block.metrics:
            lines.append(f"- {m['label']}：{value_text(m['value'])} {m['unit']}；基准 {value_text(m['baseline'])} {m['unit']}；"
                         f"变化量 {value_text(m['delta'])} {m['unit']}；比较范围：{m['scope']}")
        if block.columns:
            lines.extend(["| " + " | ".join(map(escape, block.columns)) + " |",
                          "| " + " | ".join("---" for _ in block.columns) + " |"])
            lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in block.rows)
        if block.evidence_refs:
            lines.append("证据：" + "、".join(block.evidence_refs))
        lines.append("")
    # Full inspection remains readable, including old structured payloads.
    renderer = TerminalRenderer(width=100)
    rendered = renderer.render(answer, details=True)
    lines.extend(["<details><summary>完整详情与审计</summary>", "", "```text", rendered.replace("```", "~~~"), "```", "", "</details>"])
    return "\n".join(lines).rstrip() + "\n"
