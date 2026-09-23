"""Stateful scrubber for reasoning/thinking blocks in streamed assistant text.

The regex ``_strip_think_blocks`` is correct for a complete string but, run per-delta, erases an
opening ``<think>`` that arrives alone, so downstream state machines leak reasoning. This class
holds partial tags at delta boundaries until resolved; ``flush()`` releases held-back prose that
was not a tag; ``reset()`` at the top of each turn. An open tag only starts a block at a block
boundary (stream start / after a newline / whitespace-only line so far), so prose that *mentions*
``<think>`` is not suppressed; closed pairs are always suppressed (intentional).
"""

from __future__ import annotations

import re
from typing import Tuple

__all__ = ["StreamingThinkScrubber", "StreamingToolCallScrubber", "THINK_TAG_NAMES", "THINK_OPEN_TAGS", "THINK_CLOSE_TAGS"]

# The one list of model reasoning tag names. Every surface that hides reasoning (this scrubber,
# the CLI stream filter, the gateway stream filter, the final-response regex stripper) binds to
# these; a tag added here is covered everywhere. Consumers match case-insensitively, so the
# literal tags are lowercase.
THINK_TAG_NAMES: Tuple[str, ...] = ("think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD")
THINK_OPEN_TAGS: Tuple[str, ...] = tuple(f"<{name.lower()}>" for name in THINK_TAG_NAMES)
THINK_CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{name.lower()}>" for name in THINK_TAG_NAMES)


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks.

    State: ``_in_block`` (inside an open block; text discarded), ``_buf`` (held-back partial-tag
    tail), ``_last_emitted_ended_newline`` (True iff the last emission ended with ``\\n`` or nothing
    was emitted yet — decides whether an open tag at buffer position 0 sits at a block boundary).
    """

    # Literal tags so the hot path does string ops, not regex per feed().
    _OPEN_TAGS: Tuple[str, ...] = THINK_OPEN_TAGS
    _CLOSE_TAGS: Tuple[str, ...] = THINK_CLOSE_TAGS
    _ALL_TAGS: Tuple[str, ...] = _OPEN_TAGS + _CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _ALL_TAGS)
    # Orphan close tag plus trailing whitespace (matches _strip_think_blocks case 3).
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")[ \t\n\r]*", re.IGNORECASE
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* (orphan close tags stripped) and track the newline flag."""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion ("" when it is all reasoning or held back)."""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_block:
                close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
                if close_idx == -1:
                    # No close yet: hold back a possible partial close-tag prefix, drop the rest.
                    self._hold_partial(buf, self._CLOSE_TAGS)
                    break
                buf = buf[close_idx + close_len:]
                self._in_block = False
                continue

            # Priority 1: closed <tag>X</tag> pair anywhere (even inline pairs are almost
            # certainly leaked reasoning). Priority 2: unterminated open tag at a block
            # boundary (gated so prose mentioning '<think>' isn't over-stripped). Earliest wins.
            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            # No resolvable tag: hold back any partial-tag prefix at the tail
            # so a tag split across deltas isn't missed, then emit the rest.
            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        return "".join(out)

    def _hold_partial(self, buf: str, tags: Tuple[str, ...]) -> str:
        """Move a trailing partial-tag prefix of *buf* into ``_buf``; return the remainder."""
        held = self._max_partial_suffix(buf, tags)
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    def flush(self) -> str:
        """End-of-stream flush: inside an unterminated block the held-back content is discarded (leaking
        partial reasoning is worse than a truncated answer), otherwise the tail is emitted verbatim.
        Always resets the boundary flag — intra-turn retries flush then stream again without ``reset()``,
        and a stale False flag made the new stream's opening ``<think>`` look mid-line."""
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        return self._strip_orphan_close_tags(tail) if tail else ""

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_first_tag(buf: str, tags: Tuple[str, ...]) -> Tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags* (case-insensitive), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
        return min(hits) if hits else (-1, 0)

    def _find_earliest_closed_pair(self, buf: str):
        """(start_idx, end_idx) of the earliest ``<tag>...</tag>`` pair (non-greedy, case-insensitive), else None."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = buf_lower.find(open_tag)
            close_idx = buf_lower.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> Tuple[int, int]:
        """Return the earliest block-boundary open-tag (idx, len), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = []
        for tag in self._OPEN_TAGS:
            idx = buf_lower.find(tag)
            while idx != -1 and not self._is_block_boundary(buf, idx, already_emitted):
                idx = buf_lower.find(tag, idx + 1)
            if idx != -1:
                hits.append((idx, len(tag)))
        return min(hits) if hits else (-1, 0)

    def _is_block_boundary(self, buf: str, idx: int, already_emitted: list[str]) -> bool:
        """True iff *idx* is a block boundary: position 0 after a newline-terminated (or no) prior emission,
        or any position whose preceding text on the current line is whitespace-only (when no newline
        precedes it in *buf*, the prior emission must also have ended with a newline)."""
        prior_newline = already_emitted[-1].endswith("\n") if already_emitted else self._last_emitted_ended_newline
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        return (prior_newline if last_nl == -1 else True) and preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: Tuple[str, ...]) -> int:
        """Longest buf-suffix that is a strict prefix of any tag (full matches are real tags, handled elsewhere)."""
        buf_lower = buf.lower()
        for i in range(min(len(buf_lower), cls._MAX_TAG_LEN - 1), 0, -1):
            suffix = buf_lower[-i:]
            if any(len(tag) > i and tag.startswith(suffix) for tag in tags):
                return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags with no matching open (always noise) plus trailing whitespace."""
        return cls._ORPHAN_CLOSE_RE.sub("", text) if "</" in text else text

# ---------------------------------------------------------------------------
# Tool-call XML scrubber (text-channel tool calls leaking into the pane)
# ---------------------------------------------------------------------------

#: Tag names carrying a text-channel tool call. Mirrors the owner tuple
#: ``agent_runtime_helpers._TOOL_CALL_TAG_NAMES``; a test pins the two together
#: so this module can stay dependency-free and cheap to import.
_TOOL_CALL_TAG_NAMES: Tuple[str, ...] = (
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
)

#: Optional namespace prefix — ``<atem:function_calls>``.
_NS_OPT = r"(?:[\w.-]+:)?"
_TAGS_ALT = "|".join(_TOOL_CALL_TAG_NAMES)

#: Closed pair. The open and close tags must agree on being prefixed or bare,
#: mirroring ``agent_runtime_helpers._TOOL_CALL_BLOCK_PATTERNS`` so the stream
#: and storage layers agree on what counts as a tool-call block.
_PAIR_RES: Tuple[Tuple[str, re.Pattern[str]], ...] = tuple(
    (
        name,
        re.compile(
            rf"<{_NS_OPT}{name}\b[^>]*>.*?</{_NS_OPT}{name}\s*>",
            re.DOTALL | re.IGNORECASE,
        ),
    )
    for name in _TOOL_CALL_TAG_NAMES
)

_OPEN_NAMED_RE = re.compile(
    rf"<{_NS_OPT}(?P<name>{_TAGS_ALT})\b[^>]*>", re.IGNORECASE,
)
#: While inside a block, wait for the close of the tag that opened it — a nested
#: ``</atem:parameter>`` must not end the outer block and leak what follows.
_CLOSE_BY_NAME = {
    name: re.compile(rf"</{_NS_OPT}{name}\s*>", re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
}
_STRAY_CLOSE_RE = re.compile(
    rf"</{_NS_OPT}(?:{_TAGS_ALT}|function)\s*>\s*", re.IGNORECASE,
)
#: A buffer tail that could still become a tag once the next delta arrives.
_PARTIAL_RE = re.compile(r"</?(?:[\w.-]+:)?[A-Za-z_]{0,17}$")
_PARTIAL_MAX = 40


class StreamingToolCallScrubber:
    """Stateful scrubber for text-channel tool-call XML in streamed deltas.

    Muse Spark on the Responses wire — and, less often, other reasoning models —
    occasionally serializes its next native tool call onto the *text* channel as
    ``<atem:function_calls>…</atem:function_calls>``.  The call itself is parsed
    from the provider's ``function_call`` item and still executes, so the turn
    looks healthy, but the markup reaches the pane verbatim: ``_strip_think_blocks``
    only runs on *completed* text (final response, stored content), never per
    delta, and the reasoning scrubber has no concept of a tool-call tag.

    Semantics deliberately mirror the two strippers:

      - a closed pair is suppressed anywhere (no boundary gating), which is how
        ``strip_think_blocks`` treats it;
      - an unterminated open is only treated as a block at a line boundary, so
        prose that *mentions* ``<atem:function_calls>`` survives;
      - a block that never closes is discarded at end of stream, matching the
        reasoning scrubber's rule for a hung block (leaking half a tool call is
        worse than losing a malformed one);
      - stray close tags are removed together with their trailing whitespace,
        which is what ``_STRAY_TOOL_CALL_CLOSER_PATTERN`` does.

    Whitespace *after* a suppressed pair is kept: measured against the owner
    stripper, ``strip_think_blocks("<a:function_calls>…</a:function_calls>\\n")``
    returns ``"\\n"``, and the display copy must not diverge from it.
    """

    def __init__(self) -> None:
        self._in_block: bool = False
        self._block_name: str = ""
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True
        # Set when an emitted piece ended exactly at a stray close tag: the owner
        # stripper removes the whitespace that followed it, and at a delta
        # boundary that whitespace has not arrived yet, so the next feed drops it.
        self._swallow_leading_space: bool = False

    def reset(self) -> None:
        """Reset all state. Call at the top of every new turn."""
        self._in_block = False
        self._block_name = ""
        self._buf = ""
        self._last_emitted_ended_newline = True
        self._swallow_leading_space = False

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion."""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        if self._swallow_leading_space:
            self._swallow_leading_space = False
            buf = buf.lstrip()
        out: list[str] = []

        while buf:
            if self._in_block:
                close_re = _CLOSE_BY_NAME.get(
                    self._block_name, _CLOSE_BY_NAME["function_calls"],
                )
                match = close_re.search(buf)
                if match is None:
                    held = self._partial_length(buf)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                buf = buf[match.end():]
                self._in_block = False
                self._block_name = ""
                continue

            pair = self._earliest_pair(buf)
            opener = self._open_at_boundary(buf, out)

            if pair is not None and (opener is None or pair[0] <= opener.start()):
                out.extend(self._emit(buf[:pair[0]]))
                buf = buf[pair[1]:]
                continue

            if opener is not None:
                out.extend(self._emit(buf[:opener.start()]))
                self._in_block = True
                self._block_name = (opener.group("name") or "").lower()
                buf = buf[opener.end():]
                continue

            held = self._partial_length(buf)
            emit_text = buf[:-held] if held else buf
            self._buf = buf[-held:] if held else ""
            out.extend(self._emit(emit_text))
            return "".join(out)

        return "".join(out)

    def flush(self) -> str:
        """End-of-stream flush: never leak a half-emitted tool call."""
        if self._in_block:
            self._buf = ""
            self._in_block = False
            self._block_name = ""
            self._last_emitted_ended_newline = True
            return ""
        tail = self._buf
        self._buf = ""
        self._last_emitted_ended_newline = True
        if not tail:
            return ""
        return self._strip_stray_closes(tail)

    # ── internal helpers ───────────────────────────────────────────────

    def _emit(self, text: str) -> list[str]:
        """Clean *text* and record it, returning the pieces to append."""
        if not text:
            return []
        matches = list(_STRAY_CLOSE_RE.finditer(text)) if "</" in text else []
        if matches:
            # The owner stripper removes whitespace after a stray closer; if the
            # closer ends this piece, the whitespace is in the next delta.
            self._swallow_leading_space = matches[-1].end() == len(text)
            text = _STRAY_CLOSE_RE.sub("", text)
        if not text:
            return []
        self._last_emitted_ended_newline = text.endswith("\n")
        return [text]

    @staticmethod
    def _earliest_pair(buf: str) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        for _name, pattern in _PAIR_RES:
            match = pattern.search(buf)
            if match is not None and (best is None or match.start() < best[0]):
                best = (match.start(), match.end())
        return best

    def _open_at_boundary(
        self, buf: str, already_emitted: list[str],
    ) -> Optional[re.Match[str]]:
        """First opener that sits at a line boundary, or None."""
        for match in _OPEN_NAMED_RE.finditer(buf):
            if self._is_boundary(buf, match.start(), already_emitted):
                return match
        return None

    def _is_boundary(self, buf: str, idx: int, already_emitted: list[str]) -> bool:
        if idx == 0:
            if already_emitted:
                return already_emitted[-1].endswith("\n")
            return self._last_emitted_ended_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        if last_nl == -1:
            prior_newline = (
                already_emitted[-1].endswith("\n") if already_emitted
                else self._last_emitted_ended_newline
            )
            return prior_newline and preceding.strip() == ""
        return preceding[last_nl + 1:].strip() == ""

    @staticmethod
    def _partial_length(buf: str) -> int:
        """Length of a buffer tail that may still become a tag."""
        if not buf:
            return 0
        tail = buf[-_PARTIAL_MAX:] if len(buf) > _PARTIAL_MAX else buf
        match = _PARTIAL_RE.search(tail)
        return len(match.group(0)) if match else 0

    @staticmethod
    def _strip_stray_closes(text: str) -> str:
        if "</" not in text:
            return text
        return _STRAY_CLOSE_RE.sub("", text)
