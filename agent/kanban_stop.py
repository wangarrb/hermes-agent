"""Turn-end guard for kanban workers.

Kanban workers must end with ``kanban_complete`` or ``kanban_block``. Models
(especially GLM / Qwen families) sometimes narrate the next step
("Let me write the report now") and stop with ``finish_reason=stop`` and no
tool calls. Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``.

This module is policy-only: when a kanban worker tries to finish without a
terminal board tool, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})

_DEFAULT_MAX_ATTEMPTS = 2
_MUSE_SHORT_STOP_MAX_CHARS = 120
_MUSE_SHORT_STOP_MAX_ATTEMPTS = 4
_MUSE_TEXT_TOOL_CALL_MARKERS = (
    "<atem:function_calls",
    "<atem:invoke",
    "<tool_call>",
    "<function_call>",
)
_WATCHER_TASK_RE = re.compile(r"\[任务\s+(t_[A-Za-z0-9_-]+)\b")
_MUSE_DEFAULT_TOOL_PROSE_RE = re.compile(r"\bdefault\.(?:\*|[a-z_][a-z0-9_-]*)")


def kanban_stop_nudge_enabled() -> bool:
    """Return whether the kanban stop-guard is active for this process.

    On when ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), unless
    ``HERMES_KANBAN_STOP_NUDGE`` explicitly disables it.
    """
    env = os.environ.get("HERMES_KANBAN_STOP_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    return bool(task)


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS:
                    return True
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
                return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return a synthetic follow-up when a kanban worker exits without a terminal tool.

    Returns ``None`` when the guard should not fire (not a kanban worker,
    already completed/blocked, or nudge budget exhausted).
    """
    if not kanban_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None
    if session_called_kanban_terminal(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


def _message_text(value: Any) -> str:
    """Return a bounded plain-text view of a user message payload."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(value or "")


def _latest_watcher_task(messages: Iterable[dict] | None) -> tuple[int, str] | None:
    """Return the latest watcher-injected task marker and its message index."""
    if not isinstance(messages, list):
        return None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message.get("content"))
        if "[by watcher]" not in text:
            continue
        match = _WATCHER_TASK_RE.search(text)
        if match:
            return index, match.group(1)
        env_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
        if env_task:
            return index, env_task
    return None


def _task_segment_called_terminal(
    messages: Iterable[dict] | None,
    start_index: int,
) -> bool:
    """Check terminal Kanban calls after the latest watcher task marker."""
    if not isinstance(messages, list):
        return False
    for message in messages[start_index + 1:]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            if any(
                _tool_call_name(call) in _TERMINAL_KANBAN_TOOLS
                for call in message.get("tool_calls") or []
            ):
                return True
        elif message.get("role") == "tool":
            if str(message.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
                return True
    return False


def build_muse_short_stop_nudge(
    *,
    model: str | None,
    provider: str | None,
    finish_reason: str | None,
    assistant_content: str | None,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _MUSE_SHORT_STOP_MAX_ATTEMPTS,
) -> Optional[str]:
    """Re-prompt Muse when it emits a degenerate non-terminal Kanban final.

    Muse Spark can return a tiny unrelated text fragment with
    ``finish_reason=stop`` in long agentic sessions, or leak an XML/tool-call
    wrapper such as ``<atem:invoke name="default.terminal">`` as prose instead
    of issuing a structured Hermes tool call. This guard is deliberately narrow:
    it only applies to OpenCode Go Muse models, only to a watcher-marked Kanban
    task, and only before that task has called a terminal board tool. The bounded
    retry prevents a model/provider failure from becoming an infinite loop,
    while avoiding acceptance of a random fragment or leaked tool syntax as the
    task result.
    """
    if str(provider or "").strip().lower() != "opencode-go":
        return None
    model_id = str(model or "").strip().lower().rsplit("/", 1)[-1]
    if not model_id.startswith("muse-spark-"):
        return None
    if str(finish_reason or "").strip().lower() != "stop":
        return None
    content = str(assistant_content or "").strip()
    if not content:
        return None
    lowered_content = content.lower()
    is_text_tool_call = any(marker in lowered_content for marker in _MUSE_TEXT_TOOL_CALL_MARKERS)
    if not is_text_tool_call and _MUSE_DEFAULT_TOOL_PROSE_RE.search(lowered_content):
        is_text_tool_call = any(
            term in lowered_content
            for term in ("tool", "工具", "namespace", "命名", "调用", "通道", "xml")
        )
    if not is_text_tool_call and len(content) > _MUSE_SHORT_STOP_MAX_CHARS:
        return None
    if attempts >= max_attempts:
        return None

    marker = _latest_watcher_task(messages)
    if marker is None:
        return None
    start_index, task_id = marker
    if _task_segment_called_terminal(messages, start_index):
        return None

    return (
        "[System: Muse Spark returned a non-terminal response while the "
        f"Hermes Kanban task `{task_id}` is still running. This is not a final "
        "answer. Continue the task now.\n\n"
        "Do not repeat status text, do not emit XML/Harmony tool-call prose, do "
        "not use a `default.` namespace, and do not describe what you would do. "
        "Use the exact bare tool names from the provided tool list (for example "
        "`read_file`) as a structured function call, execute the remaining work, "
        "verify it, and finish with `kanban_complete(...)` or "
        "`kanban_block(reason=...)` only when the task is actually terminal.\n\n"
        "This is a bounded recovery attempt; make real progress in the next "
        "response."
    )


__all__ = [
    "build_kanban_stop_nudge",
    "build_muse_short_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
]
