"""Command parsing and state-only command handling for the chat CLI."""
from __future__ import annotations

from dataclasses import dataclass
import shlex
import os
from pathlib import Path

from .state import ConversationState, PendingAction


COMMANDS = ("open", "status", "sources", "history", "context", "approve", "reject", "rollback", "help", "exit")


@dataclass(frozen=True)
class ParsedInput:
    kind: str  # command, text, empty
    name: str | None = None
    args: tuple[str, ...] = ()
    raw: str = ""
    error: str | None = None


def parse_input(text: str | None) -> ParsedInput:
    if text is None:
        return ParsedInput("eof", raw="")
    if not text.strip():
        return ParsedInput("empty", raw=text)
    if not text.lstrip().startswith("/"):
        return ParsedInput("text", raw=text)
    # Paths commonly contain spaces; /open consumes the complete remainder as
    # one path unless the caller explicitly quotes it.
    stripped = text.strip()
    head, _, remainder = stripped.partition(" ")
    if head.lower() == "/open" and remainder.strip():
        arg = remainder.strip()
        if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
            arg = arg[1:-1]
        return ParsedInput("command", name="open", args=(arg,), raw=text)
    try:
        # POSIX shlex treats Windows path separators as escapes.  Preserve them
        # when the CLI runs on Windows, while still accepting quoted arguments.
        tokens = shlex.split(text.strip(), posix=os.name != "nt")
        if os.name == "nt":
            tokens = [token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'" else token for token in tokens]
    except ValueError as exc:
        return ParsedInput("command", raw=text, error=f"命令格式错误：{exc}")
    name = tokens[0][1:].lower() if tokens else ""
    if name not in COMMANDS:
        return ParsedInput("command", name=name, args=tuple(tokens[1:]), raw=text, error=f"未知命令：/{name}")
    return ParsedInput("command", name=name, args=tuple(tokens[1:]), raw=text)


@dataclass(frozen=True)
class CommandResult:
    output: str = ""
    should_exit: bool = False
    action: PendingAction | None = None
    forwarded: str | None = None
    error: str | None = None


HELP_TEXT = """命令：
/open <path>       设置或切换 Aspen 文件（仅记录路径）
/status             查看会话状态
/sources            查看当前模型、YAML、SimulationDB、NodeDB
/history            查看最近对话
/approve [id]       确认待执行动作并交给上层编排器
/reject [id]        拒绝待执行动作
/rollback           清除最近一次会话元数据变更
/help               显示帮助
/exit               退出会话
普通文本会原样交给上层编排器。"""


def handle_command(parsed: ParsedInput, state: ConversationState) -> CommandResult:
    if parsed.kind == "eof":
        return CommandResult(output="再见。", should_exit=True)
    if parsed.kind == "empty":
        return CommandResult(output="")
    if parsed.kind == "text":
        return CommandResult(forwarded=parsed.raw)
    if parsed.error:
        return CommandResult(error=parsed.error)
    name = parsed.name
    if name == "help":
        return CommandResult(output=HELP_TEXT)
    if name == "exit":
        return CommandResult(output="再见。", should_exit=True)
    if name == "open":
        if not parsed.args:
            return CommandResult(error="用法：/open <Aspen 文件路径>")
        path = Path(parsed.args[0]).expanduser()
        if not path.exists():
            return CommandResult(error=f"文件不存在：{path}")
        state.checkpoint()
        state.aspen_file = str(path)
        return CommandResult(output=f"已打开：{path}")
    if name == "status":
        return CommandResult(output="\n".join(f"{k}: {v}" for k, v in state.as_display_dict().items()))
    if name == "sources":
        return CommandResult(output="\n".join([
            f"Aspen: {state.aspen_file or '(未设置)'}",
            f"YAML: {state.yaml_path or '(未设置)'}",
            f"SimulationDB: {state.simulation_db_path or '(未设置)'}",
            f"NodeDB: {state.node_db_path or '(未设置)'}",
        ]))
    if name == "context":
        return CommandResult(output=state.context_summary or "ProcessContext 尚未构建。")
    if name == "history":
        if not state.history:
            return CommandResult(output="暂无历史记录。")
        lines = []
        for i, item in enumerate(state.history[-20:], 1):
            entry = f"{i}. 用户：{item.get('user', '')}\n   回答：{item.get('assistant', '')}"
            if item.get("execution_path"):
                entry += f"\n   执行路径：{item['execution_path']}"
            lines.append(entry)
        return CommandResult(output="\n".join(lines))
    if name in ("approve", "reject"):
        action_id = parsed.args[0] if parsed.args else None
        action = state.find_pending(action_id)
        if action is None:
            return CommandResult(error="没有匹配的待确认动作。")
        state.remove_pending(action.action_id)
        if name == "approve":
            return CommandResult(output=f"已确认动作 {action.action_id}，等待上层编排器执行。", action=action)
        return CommandResult(output=f"已拒绝动作 {action.action_id}。")
    if name == "rollback":
        if not state.rollback():
            return CommandResult(output="没有可回滚的会话变更。")
        return CommandResult(output="已回滚最近一次会话元数据变更。")
    return CommandResult(error=f"未知命令：/{name}")
