"""Tests for StreamingToolCallScrubber and its wiring into ``_fire_stream_delta``.

Muse Spark (opencode-go, Responses wire) occasionally serializes its next native
tool call onto the *text* channel as
``<atem:function_calls>…</atem:function_calls>`` while the parsed
``function_call`` item still executes normally, so the turn looks healthy but the
markup reaches the pane verbatim.

The two strippers (``strip_think_blocks`` and ``cli._strip_reasoning_tags``) only
run on *completed* text — final response and stored content — so they never saw a
streamed delta.  Upstream #115662 fixed exactly those two positions and said so:
the streaming-scrubber half was left as "a separate pre-existing class".  These
tests pin both the suppression semantics and the wiring that closes that gap.

Whitespace expectations mirror the owner stripper, measured rather than assumed:
``strip_think_blocks`` keeps whitespace *after* a suppressed pair (a trailing
newline survives) and swallows whitespace after a stray close tag.
"""

from __future__ import annotations

import threading

import pytest

import run_agent
from agent.think_scrubber import StreamingThinkScrubber, StreamingToolCallScrubber

LEAK = (
    '<atem:function_calls>\n'
    '<atem:invoke name="default.terminal">\n'
    '<atem:parameter name="command">ls -la /tmp</atem:parameter>\n'
    '</atem:invoke>\n'
    '</atem:function_calls>\n'
)

TAG_NAMES = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")


def _drive(scrubber: StreamingToolCallScrubber, deltas: list[str]) -> str:
    """Feed deltas, flush, return everything the consumer would have seen."""
    return "".join([scrubber.feed(d) for d in deltas] + [scrubber.flush()])


def _chunks(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def _bare_agent():
    """An AIAgent built without ``__init__`` — just enough state for the delta sink."""
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent._stream_needs_break = False
    agent._stream_think_scrubber = StreamingThinkScrubber()
    agent._stream_toolcall_scrubber = StreamingToolCallScrubber()
    agent._stream_context_scrubber = None
    agent._current_streamed_assistant_text = ""
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent._stream_writer_token = 0
    agent._stream_writer_tls = threading.local()
    return agent


class TestClosedPairs:
    """A closed tool-call block is suppressed anywhere, like the strippers do."""

    def test_namespaced_block_in_one_delta(self) -> None:
        out = _drive(StreamingToolCallScrubber(), [LEAK])
        assert "atem:" not in out
        assert out.strip() == ""

    def test_block_after_prose_keeps_the_prose(self) -> None:
        out = _drive(StreamingToolCallScrubber(), ["hello\n" + LEAK])
        assert "atem:" not in out
        assert out.strip() == "hello"

    @pytest.mark.parametrize("tag", TAG_NAMES)
    def test_every_tag_name_bare_and_prefixed(self, tag: str) -> None:
        assert _drive(StreamingToolCallScrubber(), [f"ok\n<{tag}>x</{tag}>after"]) == "ok\nafter"
        assert _drive(StreamingToolCallScrubber(), [f"ok\n<atem:{tag}>x</atem:{tag}>after"]) == "ok\nafter"


class TestSplitAcrossDeltas:
    """The block usually arrives over several deltas — the real failure shape."""

    def test_block_split_at_arbitrary_boundaries(self) -> None:
        out = _drive(StreamingToolCallScrubber(), _chunks(LEAK, 17))
        assert "atem:" not in out
        assert out.strip() == ""

    def test_prose_before_a_split_block_survives(self) -> None:
        out = _drive(StreamingToolCallScrubber(), ["Working on it.\n"] + _chunks(LEAK, 9))
        assert "atem:" not in out
        assert out.strip() == "Working on it."

    def test_nested_closes_do_not_end_the_outer_block(self) -> None:
        """``</atem:parameter>`` must not terminate the enclosing function_calls."""
        deltas = [
            "<atem:function_calls>\n",
            '<atem:invoke name="x">\n',
            '<atem:parameter name="c">v</atem:parameter>\n',
            "CONTENT BETWEEN NESTED CLOSES\n",
            "</atem:invoke>\n",
            "</atem:function_calls>\n",
            "done",
        ]
        out = _drive(StreamingToolCallScrubber(), deltas)
        assert out.strip() == "done"
        assert "CONTENT BETWEEN" not in out


class TestProseMentionPreserved:
    """Text that merely *mentions* the tag must not be eaten."""

    def test_mid_line_mention_survives(self) -> None:
        text = "we discuss the <atem:function_calls> marker here\n"
        assert _drive(StreamingToolCallScrubber(), [text]) == text

    @pytest.mark.parametrize(
        "text",
        ["plain answer\n", "SQLite\n", "report.csv\n", "a < b\n", "-1 < x\n", "no tags here"],
    )
    def test_ordinary_text_round_trips(self, text: str) -> None:
        """No text may be lost; a partial-tag tail may arrive one delta late."""
        for size in (1, 3, 7, len(text)):
            assert _drive(StreamingToolCallScrubber(), _chunks(text, size)) == text


class TestHangingBlock:
    def test_unterminated_open_discards_to_end_of_stream(self) -> None:
        assert _drive(StreamingToolCallScrubber(), ["<atem:function_calls>\nnever closes at all"]) == ""

    def test_stray_close_removed_with_trailing_whitespace(self) -> None:
        assert _drive(StreamingToolCallScrubber(), ["before </atem:function_calls> after"]) == "before after"


class TestReset:
    def test_reset_drops_block_state(self) -> None:
        scrubber = StreamingToolCallScrubber()
        scrubber.feed("<atem:function_calls>\n")
        scrubber.reset()
        assert _drive(scrubber, ["visible again"]) == "visible again"


class TestTagNameSync:
    def test_tag_names_match_the_owner_tuple(self) -> None:
        """The scrubber mirrors the strippers' tag list; drift would reopen the leak."""
        from agent.agent_runtime_helpers import _TOOL_CALL_TAG_NAMES as owner

        import agent.think_scrubber as think_scrubber

        assert think_scrubber._TOOL_CALL_TAG_NAMES == owner


class TestSuppressedContentForward:
    """The tool-call-suppressed forward bypasses ``_fire_stream_delta``.

    ``agent.chat_completion_helpers`` hands suppressed content straight to the
    stream callback (so the CLI can extract reasoning tags from it).  That path
    needs its own scrub, or the markup shows up while the call is accumulated.
    """

    def test_xml_dropped_but_reasoning_tags_kept(self) -> None:
        """The forward exists for tag extraction — it must not eat think tags."""
        agent = _bare_agent()
        seen: list[str] = []
        agent.stream_delta_callback = seen.append
        agent._forward_suppressed_stream_text("<think>hmm</think>calling a tool now")
        assert "".join(seen) == "<think>hmm</think>calling a tool now"

    def test_tool_call_xml_is_not_forwarded(self) -> None:
        agent = _bare_agent()
        seen: list[str] = []
        agent.stream_delta_callback = seen.append
        for chunk in _chunks(LEAK, 11):
            agent._forward_suppressed_stream_text(chunk)
        out = "".join(seen)
        assert "atem:" not in out
        assert out.strip() == ""

    def test_missing_callback_is_safe(self) -> None:
        agent = _bare_agent()
        agent.stream_delta_callback = None
        agent._forward_suppressed_stream_text(LEAK)


class TestOwnerStripperParity:
    """The display copy must not diverge from the strippers it mirrors.

    ``strip_think_blocks`` is the owner for the completed-text path; if the
    scrubber disagreed with it, the same reply would render differently while
    streaming than it does from history.  The nested same-name case is included
    on purpose: both use a non-greedy per-name match, so both stop at the first
    close and emit what sits between it and the outer close — that shared
    limitation is the parity, not an accident.
    """

    @pytest.mark.parametrize("text", [
        LEAK,
        "hello\n" + LEAK,
        "before </atem:function_calls> after",
        "we discuss the <atem:function_calls> marker here\n",
        "<function_calls><function_calls>x</function_calls>y</function_calls>tail",
        "ok\n<atem:tool_calls>body</atem:tool_calls>after",
        "plain answer\n",
        "SQLite\n",
    ])
    def test_streamed_output_matches_strip_think_blocks(self, text: str) -> None:
        from agent.agent_runtime_helpers import strip_think_blocks

        for size in (1, 5, 13, len(text)):
            streamed = _drive(StreamingToolCallScrubber(), _chunks(text, size))
            assert streamed == strip_think_blocks(None, text), f"split={size}"


class TestWiring:
    """The scrubber must be inline in ``_fire_stream_delta``, not merely importable."""

    @staticmethod
    def _drain(agent, deltas: list[str]) -> str:
        seen: list[str] = []
        agent.stream_delta_callback = seen.append
        for chunk in deltas:
            agent._fire_stream_delta(chunk)
        agent._reset_stream_delivery_tracking()
        return "".join(seen)

    def test_leak_never_reaches_the_delta_sink(self) -> None:
        out = self._drain(_bare_agent(), _chunks(LEAK, 13))
        assert "atem:" not in out
        assert out.strip() == ""

    def test_without_the_scrubber_the_same_deltas_leak(self) -> None:
        """Control: pins that the wiring — not some other filter — is what fixes it."""
        agent = _bare_agent()
        agent._stream_toolcall_scrubber = None
        assert "atem:" in self._drain(agent, _chunks(LEAK, 13))

    def test_normal_text_still_flows(self) -> None:
        assert self._drain(_bare_agent(), ["hello ", "world"]) == "hello world"
