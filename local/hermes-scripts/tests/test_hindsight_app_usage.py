"""Explicit telemetry preserves retries, identity and UTC+8 day attribution."""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "hindsight_app_usage", Path(__file__).resolve().parents[1] / "hindsight_app_usage.py"
)
usage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usage)


def _event(kind, timestamp, attempt="a1", raw="event"):
    return {
        "event_type": f"llm.{kind}", "timestamp": timestamp,
        "source": "chatgpt_app", "session_id": "session",
        "request_id": "request", "attempt_id": attempt,
        "raw_event_id": raw, "raw_ref": f"fixture.json#{raw}",
    }


def test_failed_attempt_retry_and_mirrored_events_keep_cross_day_identity():
    before = "2026-10-14T15:59:59Z"
    after = "2026-10-14T16:00:00Z"
    events = [
        _event("request", before, raw="request"),
        _event("attempt", before, raw="attempt1"),
        _event("failure", before, raw="failure1"),
        _event("retry", after, attempt="a2", raw="retry2"),
        _event("attempt", after, attempt="a2", raw="attempt2"),
        _event("success", after, attempt="a2", raw="success2"),
    ]
    # A mirrored copy with a new raw ID must not create another logical event;
    # a copy timestamped after midnight must not move the original request.
    mirrored = [dict(e, raw_event_id=f"copy-{e['raw_event_id']}") for e in events]
    mirrored[0]["timestamp"] = after
    all_events = list(reversed(events + events + mirrored))
    first = usage.aggregate_llm_events(all_events, "2026-10-14")
    second = usage.aggregate_llm_events(all_events, "2026-10-15")
    assert first["counts"] == dict(requests=1, attempts=1, retries=None, successes=None, failures=1)
    assert second["counts"] == dict(requests=None, attempts=1, retries=1, successes=1, failures=None)
    assert sum(r["counts"]["attempts"] for r in [first, second]) == 2
    assert len(second["evidence"]) == 3
    assert all(e["raw_ref"].startswith("fixture.json#") for e in second["evidence"])
    assert second["coverage"]["attempts"] == "observed_only"
    assert usage.aggregate_llm_events(all_events, "2026-10-15") == second
    assert first["duplicate_events"] == 12
    assert first["coverage"]["requests"] == "conflicted"
    assert len(first["conflicts"]) == 1
    assert first["conflicts"][0]["original"]["timestamp"] != first["conflicts"][0]["duplicate"]["timestamp"]


def test_chat_exports_do_not_establish_calls_or_zero_usage_under_utc_runner():
    export = {"turns": [{"items": [{"role": "assistant"}, {"type": "tool_call"}]}]}
    unknown = usage.aggregate_llm_events(export, "2026-10-15")
    assert unknown["status"] == "unknown"
    assert all(n is None for n in unknown["counts"].values())
    assert set(unknown["coverage"].values()) == {"unknown"}
    items = [{"type": "token_count"}, {"role": "assistant"}, {"type": "tool_call"}]
    assert usage.aggregate_llm_events(items, "2026-10-15")["counts"] == unknown["counts"]
    start, end = usage.day_window(datetime(2026, 10, 14, 16, tzinfo=timezone.utc))
    assert start.isoformat() == "2026-10-15T00:00:00+08:00"
    assert (end - start).total_seconds() == 86400
    observed = [_event("success", "2026-10-14T16:00:00Z", raw="completed")]
    output = usage.aggregate_llm_events({"llm_events": observed}, "2026-10-15", complete_kinds=["llm.failure"])
    assert output["counts"]["successes"] == 1
    assert output["counts"]["failures"] == 0
    assert output["coverage"]["failures"] == "complete"
    assert output["counts"]["attempts"] is None
    # Request IDs are scoped to their original session, including raw-ID fallback.
    other_session = dict(observed[0], session_id="another-session")
    assert usage.aggregate_llm_events(observed + [other_session], "2026-10-15")["counts"]["successes"] == 2
    raw_only = dict(observed[0], request_id=None, attempt_id=None)
    assert usage.aggregate_llm_events([raw_only, raw_only], "2026-10-15")["counts"]["successes"] == 1
    contradictory = dict(observed[0], event_type="llm.failure")
    conflict = usage.aggregate_llm_events(observed + [contradictory], "2026-10-15")
    assert len(conflict["conflicts"]) == 1
    assert conflict["coverage"]["failures"] == conflict["coverage"]["successes"] == "conflicted"
    snapshots = [{
        "source": "codex-app", "source_session_id": "session", "session_id": "codex-app-session",
        "messages": [
            {"role": "user", "timestamp": "2026-10-14T15:59:59Z", "origin_message_id": "prior-day", "content": "Earlier input"},
            {"role": "assistant", "timestamp": "2026-10-14T16:00:00Z", "origin_message_id": "app:item-1", "content": "Visible output"},
        ],
        "llm_events": observed,  # Snapshot summaries never imply telemetry coverage.
    }]
    markdown = usage.render_app_daily(snapshots + snapshots, "2026-10-15")
    assert "| codex-app | `session` | 0 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 |" in markdown
    assert "app:item-1" in markdown and "prior-day" not in markdown
    assert "非精确消息事件时刻" in markdown and "用户正文覆盖不完整" in markdown


def test_shared_app_items_and_cli_aliases_are_nonadditive_and_conflicts_visible():
    item = {'role': 'assistant', 'timestamp': '2026-10-14T17:00:00Z',
            'source_turn_id': 'shared-turn', 'source_item_id': 'shared-item',
            'origin_message_id': 'source-a', 'content': '[source:codex-app session=a]\nSame public body.'}
    first = {'source': 'codex-app', 'source_session_id': 'a', 'legacy_cli_session_overlap': True,
             'messages': [item]}
    second = {'source': 'codex-app', 'source_session_id': 'b',
              'messages': [dict(item, origin_message_id='source-b', content='[source:codex-app session=b]\nSame public body.')]}
    rendered = usage.render_app_daily([first, second], '2026-10-15')
    assert '来源条目：2；其中 CLI 同源或共享历史视图 2' in rendered
    assert '身份去重：1 条，跨 session 额外出现 1 次' in rendered
    assert '含 CLI 共享历史' in rendered
    second['messages'][0]['content'] = 'Conflicting visible body'
    conflicted = usage.render_app_daily([first, second], '2026-10-15')
    assert '唯一条目数量未知' in conflicted
