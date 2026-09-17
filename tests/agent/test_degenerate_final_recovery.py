"""Degenerate-final guard: a text stop whose whole answer is not an answer.

Ported from upstream #111472, adapted for this fork (see
``agent/degenerate_final.py`` docstring for the two deliberate deviations):
the ``auto`` scope stays on the reported collapse family, and the wrong-script
arm exempts the conversation's own script so terse Chinese answers are not
mistaken for collapses.

The predicate tests are direct. The loop-integration behaviour is asserted
through the same policy functions the loop calls, with the loop's gate
conditions spelled out — this fork's loop is a single large function, so the
contract is pinned at the seam the loop actually uses.
"""

from __future__ import annotations

import pytest

from agent.degenerate_final import (
    DEGENERATE_FINAL_ALLOWLIST,
    DEGENERATE_FINAL_MAX_NUDGES,
    DEGENERATE_FINAL_MIN_TOOL_RESULTS,
    FRAGMENT_NUDGE,
    build_degenerate_final_nudge,
    degenerate_final_arm,
    degenerate_final_guard_mode,
    looks_like_degenerate_final,
    looks_like_mid_task_stall,
    tool_results_since_last_user,
)


class _Agent:
    def __init__(self, **kw):
        self.model = kw.pop("model", "muse-spark-1.3-contributor")
        self.provider = kw.pop("provider", "opencode-go")
        self.api_mode = kw.pop("api_mode", "codex_responses")
        for k, v in kw.items():
            setattr(self, k, v)


# ── Fragment arm ──────────────────────────────────────────────────────


@pytest.mark.parametrize("fragment", ["пар", "กระทบ", "σ", "더보기", "OUCH H ι", "더보기 보기"])
def test_reported_collapse_fragments_match(fragment):
    """The shapes from the upstream report must be caught."""
    assert looks_like_degenerate_final(fragment) is True


@pytest.mark.parametrize("answer", ["done", "ok", "fixed", "complete", "pass", "ack", "got it"])
def test_allowlisted_terse_answers_never_match(answer):
    assert answer in DEGENERATE_FINAL_ALLOWLIST
    assert looks_like_degenerate_final(answer) is False


@pytest.mark.parametrize("answer", [
    "已完成", "完成", "测试通过", "已修复", "改好了", "全部通过", "已提交", "没问题",
    "可以了", "已完成修改", "3 个文件已改", "已完成。", "测试通过。",
])
def test_terse_answers_in_the_conversation_script_do_not_match(answer):
    """This operator works in Chinese; a terse Chinese answer is an answer.

    Upstream's rule ("any non-ASCII character in a <=24-char answer is a
    collapse") flags every one of these. That is a measured false positive, not
    a hypothetical — see the module docstring.
    """
    assert looks_like_degenerate_final(answer) is False


@pytest.mark.parametrize("answer", [
    "Refactored the adapter and added regression tests for the reasoning follower.",
    "The task is complete; the diff is above.",
    "Done. See the diff above.",
])
def test_real_answers_above_the_char_ceiling_never_match(answer):
    assert looks_like_degenerate_final(answer) is False


def test_empty_and_whitespace_never_match():
    assert looks_like_degenerate_final("") is False
    assert looks_like_degenerate_final("   \n ") is False
    assert looks_like_degenerate_final(None) is False


def test_foreign_script_still_flagged_inside_punctuation():
    """Exempting CJK must not exempt the reported wrong-script class."""
    assert looks_like_degenerate_final("пар!") is True


# ── Mid-task stall arm ────────────────────────────────────────────────


@pytest.mark.parametrize("note", [
    "technical check in progress, pulling the term definitions",
    "analysis in progress",
    "still working",
    "verification in progress, verifying the sources",
])
def test_stall_notes_match(note):
    assert looks_like_mid_task_stall(note) is True


@pytest.mark.parametrize("answer", [
    "Done. I refactored the adapter.",
    "The task is complete",
    "分析已完成",
    "All tests pass.",
])
def test_stall_arm_leaves_real_answers_alone(answer):
    """A terminating period is a finished sentence, therefore an answer."""
    assert looks_like_mid_task_stall(answer) is False


def test_stall_arm_has_its_own_length_ceiling():
    long_note = "check in progress, " + ("pulling the term definitions " * 20)
    assert len(long_note) > 300
    assert looks_like_mid_task_stall(long_note) is False


def test_arm_classification_prefers_fragment():
    """Order matters: the fragment arm is evaluated first, so a short stall note
    inside its char ceiling classifies as a fragment. That matches upstream, and
    both arms produce a re-prompt, so only the nudge wording differs."""
    assert degenerate_final_arm("пар") == "fragment"
    # <= 24 chars → fragment arm wins.
    assert degenerate_final_arm("analysis in progress") == "fragment"
    # Above the fragment ceiling → the stall arm owns it.
    assert (
        degenerate_final_arm("technical check in progress, pulling the term definitions")
        == "mid-task stall note"
    )
    assert degenerate_final_arm("All done. Tests pass.") is None


# ── Tool-result window ────────────────────────────────────────────────


def test_tool_results_count_only_after_the_last_real_user_row():
    messages = [
        {"role": "user", "content": "earlier turn"},
        {"role": "tool", "content": "old"},
        {"role": "tool", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "this turn"},
        {"role": "tool", "content": "new"},
    ]
    assert tool_results_since_last_user(messages) == 1


def test_chat_only_turn_has_no_tool_evidence():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert tool_results_since_last_user(messages) == 0


def test_scaffolding_nudge_does_not_reset_the_window():
    """The nudge rides role=user, so a naive scan would cap the bound at one."""
    flags = ("_degenerate_final_nudge",)
    nat = [
        {"role": "user", "content": "this turn"},
        {"role": "tool", "content": "a"},
        {"role": "tool", "content": "b"},
        # First re-prompt already happened:
        {"role": "assistant", "content": "пар", "_degenerate_final_nudge": True},
        {"role": "user", "content": "resume", "_degenerate_final_nudge": True},
        {"role": "tool", "content": "c"},
    ]
    assert tool_results_since_last_user(nat, flags) == 3
    # Without the flag list the scan stops at the synthetic nudge and the guard
    # would never be able to fire a second time.
    assert tool_results_since_last_user(nat) == 1


def test_tool_evidence_threshold_is_met_by_two_results():
    assert DEGENERATE_FINAL_MIN_TOOL_RESULTS == 2
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "tool", "content": "a"},
        {"role": "tool", "content": "b"},
    ]
    assert (
        tool_results_since_last_user(msgs) >= DEGENERATE_FINAL_MIN_TOOL_RESULTS
    ) is True


# ── Scope ─────────────────────────────────────────────────────────────


def test_auto_is_on_for_the_reported_family():
    assert degenerate_final_guard_mode(
        _Agent(model="muse-spark-1.3-contributor", provider="opencode-go")
    ) == "all"


def test_auto_is_off_for_the_daily_driver():
    """Deviation 1: upstream would enable this for every codex_responses route.

    On this install that includes openai-codex/gpt-5.6-luna and cch, so the
    default scope is deliberately narrower than upstream.
    """
    assert degenerate_final_guard_mode(
        _Agent(model="gpt-5.6-luna", provider="openai-codex")
    ) == "off"
    assert degenerate_final_guard_mode(
        _Agent(model="deepseek-flash", provider="cch")
    ) == "off"


def test_explicit_override_wins_both_directions():
    assert degenerate_final_guard_mode(_Agent(_degenerate_final_guard=True)) == "all"
    assert degenerate_final_guard_mode(_Agent(_degenerate_final_guard=False)) == "off"
    assert degenerate_final_guard_mode(
        _Agent(model="deepseek-flash", _degenerate_final_guard="true")
    ) == "all"
    assert degenerate_final_guard_mode(
        _Agent(model="muse-spark-1.3-contributor", _degenerate_final_guard="off")
    ) == "off"


def test_model_list_override_scopes_by_substring():
    agent = _Agent(model="deepseek-flash", _degenerate_final_guard=["deepseek", "muse-spark"])
    assert degenerate_final_guard_mode(agent) == "all"
    assert degenerate_final_guard_mode(
        _Agent(model="gpt-5.6-luna", _degenerate_final_guard=["deepseek"])
    ) == "off"


# ── Nudges and bound ──────────────────────────────────────────────────


def test_each_arm_gets_its_own_nudge():
    fragment = build_degenerate_final_nudge("fragment")
    stall = build_degenerate_final_nudge("mid-task stall note")
    assert fragment != stall
    assert "fragment" in fragment
    assert "tool call" in stall


def test_bound_is_two_per_turn():
    assert DEGENERATE_FINAL_MAX_NUDGES == 2


# ── The loop's composed gate ──────────────────────────────────────────


def _loop_would_fire(agent, *, text, finish_reason="stop", tool_calls=None,
                    nudges=0, messages=None, stall_guards=True):
    """Reproduce the loop's gate exactly, so a drift is caught here."""
    messages = messages if messages is not None else [
        {"role": "user", "content": "go"},
        {"role": "tool", "content": "a"},
        {"role": "tool", "content": "b"},
    ]
    if not (
        finish_reason == "stop"
        and bool(getattr(agent, "_stall_guards", stall_guards))
        and degenerate_final_guard_mode(agent) != "off"
        and not tool_calls
        and nudges < DEGENERATE_FINAL_MAX_NUDGES
        and tool_results_since_last_user(messages, ("_degenerate_final_nudge",))
        >= DEGENERATE_FINAL_MIN_TOOL_RESULTS
    ):
        return False
    return bool(degenerate_final_arm(text))


def test_fragment_after_tool_work_fires():
    assert _loop_would_fire(_Agent(), text="пар") is True


def test_chat_only_turn_is_untouched():
    """A terse answer with no tool work behind it is a real answer."""
    assert _loop_would_fire(
        _Agent(), text="пар", messages=[{"role": "user", "content": "hi"}]
    ) is False


def test_out_of_scope_model_is_untouched():
    assert _loop_would_fire(_Agent(model="gpt-5.6-luna"), text="пар") is False


def test_tool_calls_present_is_untouched():
    assert _loop_would_fire(_Agent(), text="пар", tool_calls=[{"id": "x"}]) is False


def test_budget_exhausted_stops_re_prompting():
    assert _loop_would_fire(_Agent(), text="пар", nudges=2) is False
    assert _loop_would_fire(_Agent(), text="пар", nudges=1) is True


def test_non_stop_finish_reason_is_untouched():
    assert _loop_would_fire(_Agent(), text="пар", finish_reason="tool_calls") is False


def test_kill_switch_disables_the_guard():
    assert _loop_would_fire(_Agent(_stall_guards=False), text="пар") is False


def test_real_answer_after_tool_work_is_left_alone():
    assert _loop_would_fire(_Agent(), text="All tests pass. Diff is above.") is False
    assert _loop_would_fire(_Agent(), text="已完成，测试全部通过。") is False


# ── End-to-end through AIAgent.run_conversation ───────────────────────
#
# The policy tests above pin the predicate contract. These drive the real loop
# against an in-process mock provider (same harness pattern as
# test_empty_tool_name_loop_dampening.py) so the guard is proven to be wired
# into conversation_loop, not merely correct in isolation: scripted tool work,
# then a degenerate stop, then a real answer.

import json as _json
import os as _os
import shutil as _shutil
import sys as _sys
import tempfile as _tempfile
import threading as _threading
from http.server import BaseHTTPRequestHandler as _BaseHTTPRequestHandler
from http.server import HTTPServer as _HTTPServer

_REPO_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)


class _MockHandler(_BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = _json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        resp = (
            type(self).response_queue.pop(0)
            if type(self).response_queue
            else _text_resp("DONE")
        )
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls")
        chunks = [{"id": "m", "choices": [{"index": 0, "delta": {
            "role": "assistant", "content": ""}, "finish_reason": None}]}]
        if content:
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {
                "content": content}, "finish_reason": None}]})
        for ti, tc in enumerate(tcs or []):
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": ti, "id": tc["id"], "type": "function",
                "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]}}]},
                "finish_reason": None}]})
        chunks.append({"id": "m", "choices": [{"index": 0, "delta": {},
                                               "finish_reason": "tool_calls" if tcs else "stop"}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"data: {_json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a, **kw):
        pass


def _tc_resp(calls):
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": name, "arguments": args}}
                for i, (name, args) in enumerate(calls)
            ]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text):
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _make_agent(model):
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = _HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    _threading.Thread(target=srv.serve_forever, daemon=True).start()

    test_home = _tempfile.mkdtemp(prefix="hermes_e2e_degen_")
    _os.makedirs(_os.path.join(test_home, ".hermes"))
    prev_home = _os.environ.get("HERMES_HOME")
    _os.environ["HERMES_HOME"] = _os.path.join(test_home, ".hermes")

    # Import fresh so the patched conversation_loop is exercised even when the
    # module was imported earlier in the same worker.
    for mod in list(_sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools."):
            del _sys.modules[mod]
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model=model,
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )
    agent.valid_tool_names = {"terminal", "read_file"}
    return agent, srv, test_home, prev_home


def _teardown(srv, test_home, prev_home):
    srv.shutdown()
    _shutil.rmtree(test_home, ignore_errors=True)
    if prev_home is None:
        _os.environ.pop("HERMES_HOME", None)
    else:
        _os.environ["HERMES_HOME"] = prev_home


def _user_nudges(handler):
    """Every user-authored payload the model was sent, in order."""
    out = []
    for req in handler.captured_requests:
        for m in req.get("messages", []):
            if m.get("role") == "user":
                out.append(str(m.get("content") or ""))
    return out


TWO_TOOLS_THEN_FRAGMENT = [
    _tc_resp([("terminal", '{"command": "echo a"}'), ("terminal", '{"command": "echo b"}')]),
    _text_resp("пар"),
    _text_resp("All done. Both commands ran."),
]


def test_fragment_after_tool_work_is_re_prompted_in_the_real_loop():
    agent, srv, home, prev = _make_agent("muse-spark-1.3-contributor")
    try:
        _MockHandler.response_queue.extend(TWO_TOOLS_THEN_FRAGMENT)
        result = agent.run_conversation(
            "run the two commands and report", conversation_history=[], task_id="t"
        )
        nudges = _user_nudges(_MockHandler)
        assert any("ended the turn with a fragment" in n for n in nudges), (
            f"guard did not fire; user payloads were: {nudges!r}"
        )
        # The turn must end on the real answer, not the fragment.
        text = str(result) if not isinstance(result, tuple) else str(result[0])
        assert "All done" in text
    finally:
        _teardown(srv, home, prev)


def test_out_of_scope_model_is_not_re_prompted():
    """Control: the same script under a non-collapsing model is untouched."""
    agent, srv, home, prev = _make_agent("gpt-5.6-luna")
    try:
        _MockHandler.response_queue.extend(TWO_TOOLS_THEN_FRAGMENT)
        agent.run_conversation(
            "run the two commands and report", conversation_history=[], task_id="t"
        )
        nudges = _user_nudges(_MockHandler)
        assert not any("ended the turn with a fragment" in n for n in nudges)
    finally:
        _teardown(srv, home, prev)


def test_chat_only_fragment_is_not_re_prompted_in_the_real_loop():
    """A terse reply with no tool work behind it is an answer, not a collapse."""
    agent, srv, home, prev = _make_agent("muse-spark-1.3-contributor")
    try:
        _MockHandler.response_queue.extend([_text_resp("пар")])
        agent.run_conversation("say hi", conversation_history=[], task_id="t")
        nudges = _user_nudges(_MockHandler)
        assert not any("ended the turn with a fragment" in n for n in nudges)
    finally:
        _teardown(srv, home, prev)


def _nudges_in_one_request(handler):
    """Max number of degenerate-final nudges present in any single request.

    Each request re-sends the whole history, so counting occurrences across all
    captured requests would count the same nudge once per subsequent call. The
    final request carries every nudge that was appended, so the max over
    requests is the number of re-prompts that actually happened.
    """
    worst = 0
    for req in handler.captured_requests:
        n = sum(
            1 for m in req.get("messages", [])
            if m.get("role") == "user"
            and "ended the turn with a fragment" in str(m.get("content") or "")
        )
        worst = max(worst, n)
    return worst


def test_repeated_fragments_are_bounded_in_the_real_loop():
    """At most two re-prompts, then the fragment is accepted (today's behaviour)."""
    agent, srv, home, prev = _make_agent("muse-spark-1.3-contributor")
    try:
        _MockHandler.response_queue.extend([
            _tc_resp([("terminal", '{"command": "echo a"}'), ("terminal", '{"command": "echo b"}')]),
            _text_resp("пар"),
            _text_resp("пар"),
            _text_resp("пар"),
            _text_resp("пар"),
        ])
        agent.run_conversation("run the two commands", conversation_history=[], task_id="t")
        assert _nudges_in_one_request(_MockHandler) == DEGENERATE_FINAL_MAX_NUDGES
    finally:
        _teardown(srv, home, prev)


def test_re_prompt_pair_is_ephemeral_scaffolding():
    """The pair must never become durable transcript.

    Both rows ride the re-prompt (a collapsed assistant turn and a
    ``role: "user"`` nudge). If they were persisted, a resumed session would
    replay the internal retry instruction as user-authored context — the same
    failure mode the empty-response and dropped-tool-call nudges are flagged for.
    """
    import run_agent

    assert "_degenerate_final_nudge" in run_agent._EPHEMERAL_SCAFFOLDING_FLAGS

    pair = (
        {"role": "assistant", "content": "пар", "_degenerate_final_nudge": True},
        {"role": "user", "content": FRAGMENT_NUDGE, "_degenerate_final_nudge": True},
    )
    for msg in pair:
        assert run_agent._is_ephemeral_scaffolding(msg) is True
    assert run_agent._is_ephemeral_scaffolding({"role": "user", "content": "hi"}) is False
