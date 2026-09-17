"""Responses replay must not invent a blank assistant turn before a tool call.

Regression tests for the Muse Spark degenerate-final class (#103483) and for
strict Responses-compatible providers rejecting an empty ``content`` string
with HTTP 400 (#75202).

Background: the Responses API requires a following item after each reasoning
item (``missing_following_item``).  A ``function_call`` already satisfies that
requirement, so an invented ``{"role": "assistant", "content": ""}`` carrier
between the reasoning item and the call corrupts the replayed turn shape — the
model re-reads the malformed row once per tool round, all session.

The failing-turn fixture is the real shape captured in #103483 (3
reasoning+tool assistant rows, 5 tool calls/outputs, tool output bodies
elided), so this test locks in behaviour on real data rather than a synthetic
minimum.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.codex_responses_adapter import _chat_messages_to_responses_input

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "spark_failing_turn_shape.json"
)


def test_reasoning_without_tool_keeps_nonempty_following_item():
    """Control: a lone reasoning item still needs a follower.

    ``missing_following_item`` applies when nothing else follows the reasoning
    item, and the follower must be non-empty because strict Responses-compatible
    providers reject ``content=""`` with 400.
    """
    items = _chat_messages_to_responses_input([{
        "role": "assistant",
        "content": "",
        "codex_reasoning_items": [{
            "type": "reasoning",
            "id": "rs_test",
            "encrypted_content": "synthetic-encrypted-fixture",
            "summary": [],
        }],
    }])
    assert items[-1] == {"role": "assistant", "content": " "}


def test_failing_turn_fixture_emits_no_invented_carrier():
    """The real failing-turn shape must replay without a blank carrier.

    Every reasoning item must be followed directly by the ``function_call`` that
    already follows it in the stored history, and all call/output pairings must
    survive the replay.
    """
    messages = json.loads(FIXTURE.read_text(encoding="utf-8"))
    items = _chat_messages_to_responses_input(messages)

    assert {"role": "assistant", "content": ""} not in items

    for idx, item in enumerate(items):
        if item.get("type") == "reasoning":
            # Reasoning items may cluster; the next non-reasoning item after
            # each of them must be the function_call that follows it — never an
            # invented blank assistant message.
            nxt = next(
                (it for it in items[idx + 1:] if it.get("type") != "reasoning"),
                None,
            )
            assert nxt is not None, f"reasoning item at {idx} has no follower"
            assert nxt.get("type") == "function_call", (
                f"reasoning item at {idx} is followed by "
                f"{nxt.get('type') or nxt.get('role')}, not a function_call"
            )

    calls = [i["call_id"] for i in items if i.get("type") == "function_call"]
    outputs = [
        i["call_id"] for i in items if i.get("type") == "function_call_output"
    ]
    assert len(calls) == 5
    assert sorted(calls) == sorted(outputs)


def test_fixture_turn_with_visible_content_still_emits_it():
    """A reasoning+tool round that also has visible text keeps that text.

    Guards against the carrier fix swallowing real assistant content: text is
    still emitted, and it stays the item right after the reasoning block.
    """
    items = _chat_messages_to_responses_input([{
        "role": "assistant",
        "content": "Working on it.",
        "codex_reasoning_items": [{
            "type": "reasoning",
            "id": "rs_text",
            "encrypted_content": "synthetic-encrypted-fixture",
            "summary": [],
        }],
        "tool_calls": [{
            "id": "call_text_01",
            "call_id": "call_text_01",
            "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }],
    }])

    texts = [i for i in items if i.get("role") == "assistant"]
    assert texts == [{"role": "assistant", "content": "Working on it."}]
    assert {"role": "assistant", "content": ""} not in items
    assert [i["call_id"] for i in items if i.get("type") == "function_call"] == [
        "call_text_01"
    ]
