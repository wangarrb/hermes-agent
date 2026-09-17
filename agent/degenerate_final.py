"""Degenerate-final guard: a text stop whose whole answer is not an answer.

A turn can execute all of its tool work correctly and then end on
``finish_reason="stop"`` with a visible answer that is not an answer: a stray
token, a wrong-script word, a truncated non-sentence, or a progress note
describing work the model never did. The loop accepts it as the answer, the turn
reports ``completed``, and an unattended job silently abandons the task.

Ported from upstream PR #111472 (the ``turn_final_response.py`` degenerate-final
guard), which this fork cannot take verbatim because that module does not exist
at v0.20.0 and the local text-stop path lives in ``agent/conversation_loop.py``.
Two deliberate deviations, both to keep the blast radius sane on this install:

1. ``auto`` scope is narrowed to the reported collapse family (``muse-spark``).
   Upstream widens ``auto`` to every ``codex_responses`` route; on this install
   that would include the daily driver (``openai-codex`` / ``gpt-5.6-luna``) and
   ``cch``. Set ``agent._degenerate_final_guard`` to ``True`` or a model-substring
   list to widen it deliberately.
2. The wrong-script arm exempts the conversation's own script (CJK). Upstream's
   rule is "any non-ASCII character in a <=24-char answer is a collapse", which
   flags *every* terse Chinese answer ("已完成", "测试通过") — measured, not
   assumed. The reported collapse class is a stray word in a script the model
   should not be emitting (``пар``, ``กระทบ``, ``σ``, ``더보기``), so only a
   foreign script is treated as the signal.

This module is policy-only: it answers "is this text a degenerate final?" and
"which nudge applies?", and the conversation loop owns the bounded retry.
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: Above this length a reply is treated as a real (if terse) answer, never a collapse.
DEGENERATE_FINAL_MAX_CHARS = 24
#: Whitespace-token ceiling for the "short, but not a sentence" arm.
DEGENERATE_FINAL_MAX_TOKENS = 3

#: Legitimate terse answers that must never trip the guard.
DEGENERATE_FINAL_ALLOWLIST = frozenset({
    "done", "ok", "okay", "yes", "no", "yep", "nope", "fixed", "finished", "complete",
    "completed", "ready", "correct", "agreed", "understood", "ack", "none", "got it",
    # Plausible one-word legitimate answers after tool work; the allowlist is the
    # tuning surface when a false positive shows up in the warning log.
    "approved", "confirmed", "verified", "noted", "working", "checking", "pass", "passed",
    "failed", "error", "cancelled", "canceled",
})

#: Sentence terminators, ASCII and fullwidth, for the "finished sentence" test.
DEGENERATE_FINAL_TERMINATORS = (".", "!", "?", ":", "。", "！", "？", "：", "；")

_ON_VALUES = {"true", "always", "yes", "on"}
_OFF_VALUES = {"false", "never", "no", "off"}

#: Model substrings carrying the reported collapse class, used by the ``auto`` scope.
DEGENERATE_FINAL_AUTO_MODELS = ("muse-spark",)

#: Second collapse shape: a short-to-medium NON-answer announcing work in flight
#: without issuing a tool call. Observed live: a 57-char final, "technical check in
#: progress, pulling the term definitions", ending a turn after seven tool results.
DEGENERATE_FINAL_STALL_MAX_CHARS = 300
DEGENERATE_FINAL_STALL_RE = re.compile(
    r"(?:\b(?:check|work|task|step|run|process|search|review|analysis|inspection|generation|"
    r"verification|lookup)\b[^.!?\n]{0,40}\bin progress\b"
    r"|\b(?:pulling|checking|running|re-?running|generating|verifying|cross-checking|loading|"
    r"reading|gathering|searching|inspecting|mapping|reviewing|analyzing|computing|collecting|"
    r"working)\b[^.!?\n]{0,90})$",
    re.IGNORECASE,
)

FRAGMENT_NUDGE = (
    "Your previous message ended the turn with a fragment that is not a usable answer while the "
    "task was still in progress. Resume the task and finish it, then give a complete user-facing "
    "answer. If you believe the task IS finished, say so explicitly and summarize what was done "
    "and verified."
)
MID_TASK_STALL_NUDGE = (
    "Your previous message ended the turn by describing work in progress without doing it, so the "
    "task stopped mid-flight. Issue the next tool call now and continue. If the task IS finished, "
    "give the complete user-facing answer instead of a status line."
)

#: Per-turn re-prompt budget. The two arms compose in practice — a fragment retry
#: can come back as a stall note — so the bound covers both.
DEGENERATE_FINAL_MAX_NUDGES = 2
#: Tool results required before the guard may fire: a terse answer from a chat-only
#: turn is a legitimate answer, not a collapse.
DEGENERATE_FINAL_MIN_TOOL_RESULTS = 2


def _in_operator_script(ch: str) -> bool:
    """Whether ``ch`` belongs to the conversation's own script (CJK).

    Deliberately excludes Hangul, because ``더보기`` is one of the reported
    collapse fragments while Chinese is this operator's actual language.
    """
    cp = ord(ch)
    return (
        0x3000 <= cp <= 0x303F        # CJK punctuation
        or 0x3040 <= cp <= 0x30FF     # kana
        or 0x3400 <= cp <= 0x4DBF     # CJK ext A
        or 0x4E00 <= cp <= 0x9FFF     # CJK unified ideographs
        or 0xF900 <= cp <= 0xFAFF     # compatibility ideographs
        or 0xFE30 <= cp <= 0xFE4F     # CJK compatibility forms
        or 0xFF00 <= cp <= 0xFFEF     # fullwidth forms
        or 0x20000 <= cp <= 0x2FA1F   # CJK ext B-F
    )


def looks_like_degenerate_final(text: Any) -> bool:
    """Whether a text stop reads as a collapsed answer rather than a real one.

    True only for a short fragment carrying a concrete degeneration signal: a
    single token, a wrong-script word, or a sub-sentence run with no terminal
    punctuation. Allowlisted terse answers never match; a legitimate short answer
    above the char ceiling never matches; text wholly in the conversation's own
    script (CJK) never matches.
    """
    t = str(text or "").strip()
    if not t or len(t) > DEGENERATE_FINAL_MAX_CHARS:
        return False
    if t.lower().strip(".!") in DEGENERATE_FINAL_ALLOWLIST:
        return False

    non_ascii = [ch for ch in t if ord(ch) > 0x7F]
    if non_ascii:
        # Only a *foreign* script is a collapse signal here. A terse answer in the
        # conversation's own script is a legitimate reply, not the reported class.
        return not all(_in_operator_script(ch) for ch in non_ascii)

    tokens = t.split()
    if len(tokens) <= 1:
        return True
    return len(tokens) <= DEGENERATE_FINAL_MAX_TOKENS and not t.endswith(
        DEGENERATE_FINAL_TERMINATORS
    )


def looks_like_mid_task_stall(text: Any) -> bool:
    """Whether a text stop is an in-flight progress note passed off as the answer.

    Complements :func:`looks_like_degenerate_final`: that one catches the tiny
    fragment, this one the longer non-answer. A terminating ``.``/``!``/``?`` is
    treated as a finished sentence and therefore a real (if terse) answer, which
    keeps ordinary closing lines out of the guard.
    """
    t = str(text or "").strip()
    if not t or len(t) > DEGENERATE_FINAL_STALL_MAX_CHARS:
        return False
    if t.endswith((".", "!", "?")):
        return False
    return bool(DEGENERATE_FINAL_STALL_RE.search(t[-200:]))


def degenerate_final_arm(text: Any) -> Optional[str]:
    """Classify a text stop, or ``None`` when it reads as a real answer."""
    if looks_like_degenerate_final(text):
        return "fragment"
    if looks_like_mid_task_stall(text):
        return "mid-task stall note"
    return None


def degenerate_final_guard_mode(agent: Any) -> str:
    """``"off"``, ``"all"`` or ``"auto"`` for the degenerate-final re-prompt.

    ``agent._degenerate_final_guard`` overrides: ``True``/true-ish -> ``all``,
    ``False``/false-ish -> ``off``, a list -> ``all`` when a substring matches the
    model. The default ``auto`` fires only on the reported collapse family — see
    the module docstring for why this fork does not widen it to every
    ``codex_responses`` route.
    """
    mode = getattr(agent, "_degenerate_final_guard", "auto")
    if mode is False or (isinstance(mode, str) and mode.lower() in _OFF_VALUES):
        return "off"
    if mode is True or (isinstance(mode, str) and mode.lower() in _ON_VALUES):
        return "all"
    if isinstance(mode, list):
        model_lower = (getattr(agent, "model", "") or "").lower()
        return "all" if any(str(p).lower() in model_lower for p in mode if p) else "off"
    model_lower = (getattr(agent, "model", "") or "").lower()
    return "all" if any(p in model_lower for p in DEGENERATE_FINAL_AUTO_MODELS) else "off"


def tool_results_since_last_user(messages: Any, ephemeral_flags: Any = ()) -> int:
    """Tool-result rows after the most recent REAL user row — mid-task evidence.

    The re-prompt nudge rides ``role: "user"``, so a plain boundary scan would
    reset the window on the pass after the first nudge and the guard could never
    fire twice in a turn (the bound would be silently one). Flagged scaffolding
    therefore does not end the scan.
    """
    count = 0
    for msg in reversed(messages or ()):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user" and not any(msg.get(flag) for flag in ephemeral_flags):
            break
        if role == "tool":
            count += 1
    return count


def build_degenerate_final_nudge(arm: str) -> str:
    """The re-prompt text for a classified arm."""
    return MID_TASK_STALL_NUDGE if arm == "mid-task stall note" else FRAGMENT_NUDGE
