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
        attempts=2,
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






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




