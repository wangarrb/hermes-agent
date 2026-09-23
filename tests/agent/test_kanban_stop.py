"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    build_muse_short_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_disabled_inside_delegated_child(clear_kanban_env):
    from agent.delegation_context import delegated_child_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_disabled_inside_non_dispatcher_context(clear_kanban_env):
    from agent.delegation_context import non_dispatcher_owned_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_muse_short_stop_nudge_targets_watcher_task():
    messages = [
        {
            "role": "user",
            "content": "请读取任务文件并执行。[任务 t_abc123: bounded work] [by watcher]",
        },
    ]
    nudge = build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="shit fuck damn",
        messages=messages,
    )

    assert nudge is not None
    assert "t_abc123" in nudge
    assert "read_file" in nudge
    assert "default." in nudge


def test_muse_short_stop_nudge_is_bounded_and_ignores_non_muse():
    messages = [
        {"role": "user", "content": "[任务 t_abc123: work] [by watcher]"},
    ]
    assert build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="short",
        messages=messages,
        attempts=4,
    ) is None
    assert build_muse_short_stop_nudge(
        model="glm-5.3-flash",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="short",
        messages=messages,
    ) is None
    assert build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="x" * 121,
        messages=messages,
    ) is None


def test_muse_textual_tool_call_leak_is_reprompted():
    messages = [
        {"role": "user", "content": "[任务 t_abc123: work] [by watcher]"},
    ]
    nudge = build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content=(
            '<atem:function_calls>\n'
            '<atem:invoke name="default.terminal">\n'
            'echo "still working"\n'
            '</atem:invoke>\n'
            '</atem:function_calls>'
        ),
        messages=messages,
    )

    assert nudge is not None
    assert "structured function call" in nudge


def test_muse_default_tool_namespace_prose_is_reprompted():
    messages = [
        {"role": "user", "content": "[任务 t_abc123: work] [by watcher]"},
    ]
    nudge = build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content=(
            "未完成：继续取证。有效工具命名仍为 default.*，按既有通道继续取证。"
        ),
        messages=messages,
        attempts=2,
    )

    assert nudge is not None
    assert "bare tool names" in nudge


def test_muse_short_stop_nudge_stops_after_terminal_board_tool():
    messages = [
        {"role": "user", "content": "[任务 t_abc123: work] [by watcher]"},
        {
            "role": "assistant",
            "content": "done",
            "tool_calls": [{
                "id": "1",
                "function": {"name": "kanban_complete", "arguments": "{}"},
            }],
        },
    ]
    assert build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="short",
        messages=messages,
    ) is None






def test_muse_nudge_survives_pre_strip_losing_the_xml():
    """The stripper must not blind this guard.

    ``conversation_loop`` strips text-channel tool-call XML from the visible
    final before calling the guard, so a turn whose entire final was a leaked
    ``<atem:function_calls>`` block arrives as an empty ``assistant_content``.
    Without ``raw_content`` the guard would go silent on exactly the leak it
    exists to catch, and the turn would fall through to the generic
    (session-scoped, one-shot) empty-response path instead.
    """
    messages = [
        {"role": "user", "content": "[任务 t_abc123: work] [by watcher]"},
    ]
    leaked = (
        '<atem:function_calls>\n'
        '<atem:invoke name="default.terminal">\n'
        'echo "still working"\n'
        '</atem:invoke>\n'
        '</atem:function_calls>'
    )

    # What the loop actually hands over after stripping: empty visible text,
    # with the pre-strip original alongside it.
    nudge = build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="",
        raw_content=leaked,
        messages=messages,
    )

    assert nudge is not None, "guard went silent on a leaked tool call"
    assert "structured function call" in nudge


def test_muse_nudge_still_ignores_an_empty_final_with_no_leak():
    """An empty final with no tool-call leak keeps the old behaviour.

    A genuinely empty response must stay with the generic empty-response path;
    this guard must not claim it.
    """
    messages = [
        {"role": "user", "content": "[任务 t_abc123: work] [by watcher]"},
    ]
    assert build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="",
        raw_content="",
        messages=messages,
    ) is None


def test_muse_nudge_unchanged_on_the_nonempty_path():
    """``raw_content`` must not widen detection beyond the empty case.

    When the visible final is non-empty the guard sees exactly what it saw
    before this change: a long answer with no tool-call syntax is left alone,
    while a short fragment is still treated as degenerate.
    """
    messages = [
        {"role": "user", "content": "[任务 t_abc123: work] [by watcher]"},
    ]
    long_answer = (
        "The task is complete. I refactored the replay adapter, added regression "
        "tests for the reasoning follower, and verified the full suite passes. "
        "See the diff above for the exact changes."
    )
    assert len(long_answer) > 120
    assert build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content=long_answer,
        raw_content=long_answer,
        messages=messages,
    ) is None

    # Degenerate short final still caught (pre-existing path).
    assert build_muse_short_stop_nudge(
        model="muse-spark-1.3-contributor",
        provider="opencode-go",
        finish_reason="stop",
        assistant_content="пар",
        raw_content="пар",
        messages=messages,
    ) is not None


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.






@pytest.mark.parametrize(
    "tool_name,who",
    [
        ("kanban_request_review", "build worker handing off for same-card review"),
        ("kanban_request_changes", "review agent sending the card back"),
    ],
)
def test_no_nudge_after_handoff_tool(clear_kanban_env, tool_name, who):
    """Handoff tools end the worker's turn just like complete/block.

    Both move the card out of ``running``, and the worker is told to call
    them — goals.py's continuation/finalize prompts name
    ``kanban_request_review``; the force-loaded sdlc-review skill names
    ``kanban_request_changes``. Nudging afterwards asks a worker that did
    the right thing to close a card it must not close.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_handoff")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is True, who
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_still_fires_for_non_terminal_kanban_tool(clear_kanban_env):
    """Widening the set must not swallow the case the guard exists for."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "Let me open the review next.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_comment", "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is False
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    # The nudge offers every worker exit, not just close-out; a card that must go
    # through review must never be steered to ``kanban_complete`` alone.
    assert "kanban_request_review" in nudge and "kanban_block" in nudge
