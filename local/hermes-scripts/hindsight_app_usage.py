"""Count explicit LLM telemetry without treating chat items as API calls.

The currently available App thread/history export exposes turns/items, not
request attempts. Such exports therefore yield unknown counts. An authorized
telemetry adapter may supply ``llm_events`` using this normalized schema::

    {"event_type": "llm.attempt", "timestamp": "2026-10-14T16:00:00Z",
     "source": "chatgpt_app", "session_id": "session-1",
     "request_id": "request-1", "attempt_id": "attempt-1",
     "raw_event_id": "event-1", "raw_ref": "export.json#event-1"}

Kinds are request/attempt/retry/success/failure. A retry event describes a
retry attempt; success/failure describe an attempt outcome. Adapters must
preserve these meanings and original identities. No event is synthesized.
Counts are observations within a day, not cohort outcomes for requests started
that day. Partial histories cannot prove daily totals or zero usage.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta, timezone
from typing import Any


UTC8 = timezone(timedelta(hours=8), "UTC+8")
KINDS = {
    "llm.request": "requests",
    "llm.attempt": "attempts",
    "llm.retry": "retries",
    "llm.success": "successes",
    "llm.failure": "failures",
}


def day_window(day: str | date | datetime) -> tuple[datetime, datetime]:
    """Return [00:00, next 00:00) for the given UTC+8 calendar day."""
    if isinstance(day, datetime):
        if day.tzinfo is None or day.utcoffset() is None:
            raise ValueError("datetime day must include a timezone")
        day = day.astimezone(UTC8).date()
    elif isinstance(day, str):
        day = date.fromisoformat(day)
    start = datetime.combine(day, time.min, tzinfo=UTC8)
    return start, start + timedelta(days=1)


def _identity(event: Mapping[str, Any]) -> tuple | None:
    kind = event["event_type"]
    request = event.get("request_id")
    attempt = event.get("attempt_id")
    if request and (kind == "llm.request" or attempt):
        return ("request_attempt", str(request), None if kind == "llm.request" else str(attempt))
    if event.get("raw_event_id"):
        return ("raw_event", str(event["raw_event_id"]))
    return None


def _reference(event: Mapping[str, Any], timestamp: datetime) -> dict[str, Any]:
    return {
        "event_type": event["event_type"], "timestamp": timestamp.isoformat(),
        "source": event["source"], "session_id": event["session_id"],
        **{field: event.get(field) for field in ("request_id", "attempt_id", "raw_event_id", "raw_ref")},
    }


def aggregate_llm_events(
    events: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    day: str | date | datetime,
    *,
    complete_kinds: Iterable[str] = (),
) -> dict[str, Any]:
    """Return explicit event counts, per-kind coverage and original references.

    ``complete_kinds`` is an adapter's assertion that its event stream covers
    the entire UTC+8 day for those kinds. Omit it for App history or partial
    logs. Missing kinds stay None unless that completeness is established.
    Deduplication occurs before time filtering so a later export copy cannot
    move an event across midnight. There is no state, network or filesystem IO.
    """
    start, end = day_window(day)
    complete = set(complete_kinds)
    if complete - KINDS.keys():
        raise ValueError("complete_kinds must contain normalized llm.* event kinds")
    if isinstance(events, Mapping):
        events = events.get("llm_events") or []
    candidates = []
    ignored = 0
    for event in events:
        if not isinstance(event, Mapping) or event.get("event_type") not in KINDS:
            ignored += 1
            continue
        if not event.get("source") or not event.get("session_id"):
            ignored += 1
            continue
        try:
            timestamp = datetime.fromisoformat(str(event.get("timestamp", "")).replace("Z", "+00:00"))
        except ValueError:
            ignored += 1
            continue
        identity = _identity(event)
        if timestamp.utcoffset() is None or identity is None:
            ignored += 1
            continue
        candidates.append((timestamp, event, identity))

    counts = {column: 0 if kind in complete else None for kind, column in KINDS.items()}
    evidence = []
    seen = {}
    seen_raw = {}
    conflicts = []
    conflicted_kinds = set()
    duplicate_events = 0
    for timestamp, event, identity in sorted(candidates, key=lambda row: row[0]):
        kind = event["event_type"]
        scope = (str(event["source"]), str(event["session_id"]), kind)
        key = (*scope, *identity)
        raw_key = (*scope[:2], str(event["raw_event_id"])) if event.get("raw_event_id") else None
        reference = _reference(event, timestamp)
        original = seen.get(key) or seen_raw.get(raw_key)
        seen.setdefault(key, original or reference)
        if raw_key is not None:
            seen_raw.setdefault(raw_key, original or reference)
        if original is not None:
            duplicate_events += 1
            # Export locations/IDs may differ for a mirror, but conflicting
            # telemetry identities or timestamps must remain visible.
            fields = ("event_type", "timestamp", "request_id", "attempt_id")
            in_window = start <= timestamp < end or start <= datetime.fromisoformat(original["timestamp"]) < end
            if in_window and any(original[field] != reference[field] for field in fields):
                conflicts.append({"original": original, "duplicate": reference})
                conflicted_kinds.update([original["event_type"], kind])
            continue
        if not start <= timestamp < end:
            continue
        column = KINDS[kind]
        counts[column] = (counts[column] or 0) + 1
        evidence.append(reference)
    coverage = {
        column: "conflicted" if kind in conflicted_kinds else
        "complete" if kind in complete else "observed_only" if counts[column] is not None else "unknown"
        for kind, column in KINDS.items()
    }
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat(), "timezone": "UTC+8"},
        "counts": counts,
        "coverage": coverage,
        "status": "complete" if complete == KINDS.keys() and not conflicts else "partial" if evidence or conflicts else "unknown",
        "note": "Only explicit LLM telemetry is counted; App turns/items and token_count do not establish request totals.",
        "evidence": evidence,
        "duplicate_events": duplicate_events,
        "conflicts": conflicts,
        "ignored_events": ignored,
    }


def render_app_daily(sessions: Iterable[Mapping[str, Any]], day: str | date | datetime) -> str:
    """Render a local sidecar from scoped native snapshots, without request inference.

    Native App snapshots carry turn start/completion timestamps, not precise
    item event times. Coverage is only the user/assistant text actually returned
    by the supported API. This function neither writes a report nor contacts a
    model; the reviewed daily entry point owns eventual concatenation.
    """
    start, end = day_window(day)
    grouped: dict[str, dict] = {}
    labels = {}
    legacy_overlap = set()
    for session in sessions:
        if session.get("source") not in {"codex-app", "chatgpt-app"}:
            continue
        sid = str(session.get("source_session_id") or session["session_id"])
        if sid in labels and labels[sid] != session['source']:
            raise ValueError('source kind changed for stable session identity')
        labels[sid] = session['source']
        if session.get('legacy_cli_session_overlap'):
            legacy_overlap.add(sid)
        events = grouped.setdefault(sid, {})
        for message in session.get("messages", []):
            if message.get("role") not in {"user", "assistant"}:
                continue
            origin = message.get("origin_message_id")
            if not origin:
                raise ValueError("App daily evidence requires original item identity")
            if origin in events and events[origin] != message:
                raise ValueError("App daily source item changed")
            events[origin] = message

    def item_key(sid, message):
        if message.get('source_turn_id') and message.get('source_item_id'):
            return (labels[sid], message['source_turn_id'], message['source_item_id'])
        return (labels[sid], sid, message['origin_message_id'])

    cli_keys = {item_key(sid, m) for sid in legacy_overlap for m in grouped[sid].values()}

    def cell(value):
        return str(value).replace("|", "\\|").replace("`", "'").replace("\n", " ")

    lines = [
        f"### ChatGPT App 会话补充：{start.date().isoformat()}（UTC+8）",
        f"窗口：{start.isoformat()} ≤ 时间 < {end.isoformat()}。",
        "仅统计受支持 API 实际返回的可见 user/assistant 条目，用户正文覆盖不完整。",
        "日期归属使用 turn start/completion 边界时间，非精确消息事件时刻。",
        "未暴露显式 LLM 请求遥测；请求、尝试、重试、成功、失败均未知。",
        "",
        "| 来源 | session | 可见 user | 可见 assistant | 请求 | 尝试 | 重试 | 成功 | 失败 |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    references = []
    missing_time = 0
    additive_messages = overlap_messages = 0
    unique_items = {}
    conflicting_items = set()
    for sid, events in sorted(grouped.items()):
        visible = []
        for origin, message in events.items():
            try:
                timestamp = datetime.fromisoformat(str(message.get("timestamp", "")).replace("Z", "+00:00"))
            except ValueError:
                missing_time += 1
                continue
            if timestamp.utcoffset() is None:
                missing_time += 1
                continue
            if start <= timestamp < end:
                visible.append(message)
                references.append(f"- `{cell(sid)}` / `{cell(origin)}` / {timestamp.astimezone(UTC8).isoformat()}")
                key = item_key(sid, message)
                body = str(message.get('content', ''))
                if body.startswith('[source:') and '\n' in body:
                    body = body.split('\n', 1)[1]
                signature = (message['role'], timestamp.isoformat(), body)
                if key in unique_items and unique_items[key] != signature:
                    conflicting_items.add(key)
                unique_items.setdefault(key, signature)
        users = sum(message["role"] == "user" for message in visible)
        assistants = sum(message["role"] == "assistant" for message in visible)
        if visible:
            shares_cli = any(item_key(sid, m) in cli_keys for m in visible)
            label = labels[sid] + ('（CLI 同源）' if sid in legacy_overlap else '（含 CLI 共享历史）' if shares_cli else '')
            lines.append(f"| {label} | `{cell(sid)}` | {users} | {assistants} | 未知 | 未知 | 未知 | 未知 | 未知 |")
            overlap = sum(item_key(sid, m) in cli_keys for m in visible)
            overlap_messages += overlap
            additive_messages += len(visible) - overlap
    total = additive_messages + overlap_messages
    lines.append(f"\n本日 App 来源条目：{total}；其中 CLI 同源或共享历史视图 {overlap_messages}，不得与旧 CLI 统计相加。")
    if conflicting_items:
        lines.append(f"App 唯一条目数量未知：{len(conflicting_items)} 个共享身份的正文或时间冲突，需核对来源。")
    else:
        lines.append(f"按返回的稳定 turn/item 身份去重：{len(unique_items)} 条，跨 session 额外出现 {total - len(unique_items)} 次。")
    lines.append(f"无已知 CLI 同源身份重叠的来源条目：{additive_messages}；旧导出缺 item 身份，跨 CLI 的精确新增差额仍未知。以上均不是 LLM 调用数。")
    if missing_time:
        lines.append(f"\n另有 {missing_time} 个可见条目缺少带时区时间，未分配至本日。")
    lines.extend(["", "原始条目核对（turn 边界时间）：", *sorted(references)])
    return "\n".join(lines) + "\n"
