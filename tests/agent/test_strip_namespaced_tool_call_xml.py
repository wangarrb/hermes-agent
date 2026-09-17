"""Namespace-prefixed tool-call XML must not leak into visible text.

Regression tests for the shape where Muse Spark (opencode-go, Responses wire)
serializes its native tool call onto the *text* channel with a namespace prefix
(``<atem:function_calls>…</atem:function_calls>``) instead of emitting a
``function_call`` item.  On a turn that already did real tool work, that literal
XML was then delivered as the turn's final answer (``finish_reason=stop``), so
the user saw markup where the answer should be.

Both strip positions must agree, because the CLI keeps its own copy for display:
the agent-side scrubber (``agent.agent_runtime_helpers.strip_think_blocks``) and
the CLI display copy (``cli._strip_reasoning_tags``).  A fix in only one place
still shows the XML to the user.

The unprefixed cases are pinned alongside the prefixed ones so a future pattern
edit cannot regress the original openclaw/openclaw#67318 behaviour.
"""

from __future__ import annotations

import pytest

from agent.agent_runtime_helpers import strip_think_blocks

PREFIXED_BLOCK = (
    '<atem:function_calls><atem:invoke name="terminal">'
    '<atem:parameter name="command">ls</atem:parameter>'
    "</atem:invoke></atem:function_calls>"
)
UNPREFIXED_BLOCK = (
    '<function_calls><invoke name="terminal">'
    '<parameter name="command">ls</parameter>'
    "</invoke></function_calls>"
)


class _BareAgent:
    """``strip_think_blocks`` takes an agent only for logging/patch routing."""

    def __getattr__(self, _name):  # pragma: no cover - defensive
        return None


def _agent_strip(text: str) -> str:
    return strip_think_blocks(_BareAgent(), text)


def _cli_strip(text: str) -> str:
    import cli

    return cli._strip_reasoning_tags(text)


STRIPPERS = {
    "agent.strip_think_blocks": _agent_strip,
    "cli._strip_reasoning_tags": _cli_strip,
}


@pytest.mark.parametrize("name", sorted(STRIPPERS))
class TestToolCallXml:
    def test_unprefixed_block_still_stripped(self, name):
        """Pre-existing behaviour (openclaw/openclaw#67318) must not regress."""
        assert STRIPPERS[name](UNPREFIXED_BLOCK) == ""

    def test_namespace_prefixed_block_stripped(self, name):
        assert STRIPPERS[name](PREFIXED_BLOCK) == ""

    def test_prefixed_block_inside_prose_keeps_prose(self, name):
        out = STRIPPERS[name](f"Before {PREFIXED_BLOCK} after")
        assert "<" not in out
        assert "Before" in out and "after" in out

    def test_prefixed_stray_closer_stripped(self, name):
        out = STRIPPERS[name]("the answer is 42</atem:function_calls>")
        assert out == "the answer is 42"

    @pytest.mark.parametrize(
        "tag",
        ["tool_call", "tool_calls", "tool_result", "function_call", "function_calls"],
    )
    def test_every_tag_name_accepts_a_prefix(self, name, tag):
        block = f'<atem:{tag}><atem:invoke name="x"></atem:invoke></atem:{tag}>'
        assert STRIPPERS[name](block) == ""
