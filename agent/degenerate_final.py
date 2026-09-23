"""Degenerate-final guard: a text stop whose whole answer is not an answer.

A turn can execute all of its tool work correctly and then end on
``finish_reason="stop"`` with a visible answer that is not an answer: a stray
token, a wrong-script word, a truncated non-sentence, or a progress note
describing work the model never did. The loop accepts it as the answer, the turn
reports ``completed``, and an unattended job silently abandons the task.

Ported from upstream PR #111472 (the ``turn_final_response.py`` degenerate-final
guard), which this fork cannot take verbatim because that module does not exist
at v0.20.0 and the local text-stop path lives in ``agent/conversation_loop.py``.
The predicate itself is upstream's, ported from the released
``agent/agent_runtime_helpers.py::looks_like_degenerate_final`` (v2026.9.21).
The earlier hand-written version was measurably wrong: it treated *any*
single-token answer as a fragment, so legitimate terse answers (``42``,
``SQLite``, ``report.csv``, ``:8080``, ``€12.50``) would have been re-prompted on
a turn that did tool work, and its wrong-script arm ignored the user's own
language. Upstream's three refinements carry the weight:

1. any ASCII alphanumeric character means a real (if terse) answer — this is what
   protects ``42``/``SQLite``/``report.csv``/``:8080``/``€12.50``;
2. a sentence terminator means a finished sentence, therefore an answer;
3. a mid-punctuation opener (``?warming up``) is a fragment, and a reply in a
   script the *user* also uses is not a wrong-script collapse.

One deviation is kept, deliberately and measured:

* The wrong-script arm exempts the conversation's own script (CJK).  Upstream
  judges "wrong script" only against the user's message, which fails on this
  install: kanban task prompts are frequently English while the operator works in
  Chinese, so a legitimate terse ``已完成`` would be re-prompted.  Exempting CJK
  while still flagging Hangul (``더보기`` is a reported fragment) is what prevents
  that.  Upstream's user-script comparison is kept as well, so a reply in the
  *user's* non-Latin script is an answer (``да`` to a Russian prompt).

Scope: ``auto`` fires for every route.  Upstream widens ``auto`` to every
``codex_responses`` route; the collapse was measured on this install across
``opencode-go``/muse, ``openai-codex``/gpt-5.6-* and ``cch``/deepseek-flash
alike, so the fence is the model's behaviour, not the transport.  Narrow it with
``agent._degenerate_final_guard`` (``False``/``off``, or a model-substring list)
if a false positive shows up in the warning log.

This module is policy-only: it answers "is this text a degenerate final?" and
"which nudge applies?", and the conversation loop owns the bounded retry.
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: Above this length a reply is treated as a real (if terse) answer, never a collapse.
DEGENERATE_FINAL_MAX_CHARS = 24

#: Legitimate terse answers that must never trip the guard.  Upstream's
#: ASCII-alphanumeric rule already protects every one of these; the set is kept
#: as an explicit, cheap first pass and as the tuning surface for a false
#: positive that shows up in the warning log.
DEGENERATE_FINAL_ALLOWLIST = frozenset({
    "done", "ok", "okay", "yes", "no", "yep", "nope", "fixed", "finished", "complete",
    "completed", "ready", "correct", "agreed", "understood", "ack", "none", "got it",
    # Plausible one-word legitimate answers after tool work; the allowlist is the
    # tuning surface when a false positive shows up in the warning log.
    "approved", "confirmed", "verified", "noted", "working", "checking", "pass", "passed",
    "failed", "error", "cancelled", "canceled",
})

#: Sentence terminators — upstream's set.  A terminal means a finished sentence,
#: therefore an answer, which keeps ordinary closing lines out of the guard.
DEGENERATE_FINAL_TERMINATORS = (".", "!", "?", "。", "！", "？")

#: Leading punctuation no answer begins with when a letter follows (``?warming``).
#: Mirrors upstream's ``_DEGENERATE_LEADING_PUNCT``: ``$5``, ``#123``, ``-1``,
#: ``/tmp``, ``.env``, ``(a)``, ``:8080`` and ``:)`` all stay answers.
DEGENERATE_FINAL_LEADING_PUNCT = "?!,;:)]}"

_ON_VALUES = {"true", "always", "yes", "on"}
_OFF_VALUES = {"false", "never", "no", "off"}

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


def _message_text(content: Any) -> str:
    """Flatten a message body (a string, or a multi-part content list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return "" if content is None else str(content)


def _has_non_ascii_letter(content: Any) -> bool:
    """Whether the user's own text carries a non-ASCII letter (i.e. its script)."""
    return any(ch.isalpha() and not ch.isascii() for ch in _message_text(content))


def last_real_user_message(messages: Any, ephemeral_flags: Any = ()) -> str:
    """The most recent user row that is not loop scaffolding, as text."""
    for msg in reversed(messages or ()):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if any(msg.get(flag) for flag in ephemeral_flags):
            continue
        return _message_text(msg.get("content"))
    return ""


def looks_like_degenerate_final(text: Any, user_message: Any = None) -> bool:
    """Whether a text stop reads as a collapsed fragment rather than an answer.

    Upstream's predicate (v2026.9.21 ``agent_runtime_helpers``), ported.  A
    fragment is a short reply with no sentence terminator that carries a
    *foreign-script* word, or a token starting mid-punctuation (``?warming up``).
    Anything holding an ASCII alphanumeric character is a real answer — that is
    what protects ``42``, ``SQLite``, ``report.csv``, ``:8080`` and ``€12.50`` —
    and a letterless reply (``₽🔧``, ``✅``) is never a collapse.

    ``user_message`` is the turn's own user text: a reply in a script the user
    already uses (``да`` to a Russian prompt) is an answer, not a wrong-script
    collapse.  One local deviation on top: a reply wholly in the conversation's
    own script (CJK) is an answer even when the user wrote in another script —
    see the module docstring.
    """
    t = str(text or "").strip()
    if not t or len(t) > DEGENERATE_FINAL_MAX_CHARS:
        return False
    if t.endswith(DEGENERATE_FINAL_TERMINATORS):
        return False
    if t.lower().strip(".!") in DEGENERATE_FINAL_ALLOWLIST:
        return False
    # A letter straight after mid-punctuation is a truncated token, not an answer.
    if t[0] in DEGENERATE_FINAL_LEADING_PUNCT and len(t) > 1 and t[1].isalpha():
        return True
    # Any ASCII alphanumeric character means a real (if terse) answer.
    if any(ch.isascii() and ch.isalnum() for ch in t):
        return False
    # Letters are required for a wrong-script signal; "₽🔧"/"✅" are not collapses.
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    # Deviation: the operator's own script is never a wrong-script collapse.
    if all(_in_operator_script(ch) for ch in letters):
        return False
    # Upstream: a reply in a script the user already uses is an answer.
    return not _has_non_ascii_letter(user_message)


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


def degenerate_final_arm(text: Any, user_message: Any = None) -> Optional[str]:
    """Classify a text stop, or ``None`` when it reads as a real answer."""
    if looks_like_degenerate_final(text, user_message):
        return "fragment"
    if looks_like_mid_task_stall(text):
        return "mid-task stall note"
    return None


def degenerate_final_guard_mode(agent: Any) -> str:
    """``"off"`` or ``"all"`` for the degenerate-final re-prompt.

    ``agent._degenerate_final_guard`` overrides: ``True``/true-ish -> ``all``,
    ``False``/false-ish -> ``off``, a list -> ``all`` when a substring matches the
    model. The default ``auto`` is ``all`` — see the module docstring for why this
    install enables it on every route rather than only ``codex_responses``.
    """
    mode = getattr(agent, "_degenerate_final_guard", "auto")
    if mode is False or (isinstance(mode, str) and mode.lower() in _OFF_VALUES):
        return "off"
    if isinstance(mode, list):
        model_lower = (getattr(agent, "model", "") or "").lower()
        return "all" if any(str(p).lower() in model_lower for p in mode if p) else "off"
    return "all"


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
