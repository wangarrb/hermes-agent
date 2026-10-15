#!/usr/bin/env python3
"""Import explicitly scoped, supported read_thread exports into the existing feed.

No network, credentials, API emulation, or automatic Hindsight writes. Page
exports may overlap; item IDs, not dates/mtime/assistant counts, own identity.
Only completed turns are published. In-progress turns must be fetched again.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

from hindsight_session_manifest import SECRET_MATERIAL_RE, clean_text, is_noise_message

UUID = re.compile(r'^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$')
SCHEMA = 'codex-app-read-thread/1'


def discover_threads(listing, scope):
    """A listing is a bounded view, not proof all account history was enumerated."""
    rows = listing.get('pinnedThreads', []) + listing.get('threads', [])
    return {r['id']: r for r in rows if r.get('kind') == 'codex' and r.get('id') in scope}


def project_page(page):
    """Keep public conversation text only, excluding commands/tools/reasoning."""
    turns = []
    for turn in page.get('turns', []):
        items = [i for i in turn.get('items', []) if i.get('type') in {'userMessage', 'agentMessage'}]
        turns.append({**{k: turn.get(k) for k in ('id', 'status', 'startedAt', 'completedAt')},
                      'items': [{k: i[k] for k in ('id', 'type', 'text', 'content', 'phase', 'truncated') if k in i}
                                for i in items]})
    return {'schema': SCHEMA, 'schemaVersion': page.get('schemaVersion'),
            'thread': {k: page['thread'][k] for k in ('id', 'kind', 'hostId', 'createdAt', 'title')
                       if k in page['thread']}, 'page': page['page'], 'turns': turns,
            'coverage': {'user_text': 'only_if_returned', 'llm_events': 'unavailable',
                         'item_timestamps': 'unavailable_turn_time_fallback'}}


def timestamp(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('timezone required')
    return dt.astimezone(timezone.utc).isoformat()


def _public_text(item):
    if item.get('truncated'):
        raise ValueError('truncated item: refetch before publishing')
    if 'text' in item:
        return clean_text(item['text'])
    parts = item.get('content', [])
    if not isinstance(parts, list):
        raise ValueError('unsupported user content schema')
    return clean_text('\n'.join(p.get('text', '') for p in parts
                                if isinstance(p, dict) and p.get('type') in {'text', 'inputText', 'input_text'}))


def _message_order(message):
    return (message['timestamp'], message['source_turn_id'], message['source_item_order'])


def convert_pages(pages, scope):
    sessions, evidence = {}, {}
    for page in pages:
        thread = page['thread']; sid = thread['id']
        if sid not in scope or not UUID.fullmatch(sid) or thread.get('kind') != 'codex':
            raise ValueError('thread outside explicit authorized scope')
        if page.get('schema') not in (None, SCHEMA) or page.get('schemaVersion') != 1:
            raise ValueError('unsupported read_thread schema')
        entry = sessions.setdefault(sid, {'session_id': f'codex-app-{sid}',
            'platform': 'backfill-codex-app', 'model': 'unknown',
            'session_start': timestamp(thread['createdAt']), 'messages': [],
            'source': 'codex-app', 'source_session_id': sid,
            # App delegation is independent work, never Hermes compression lineage.
            'app_parent_session_id': scope[sid].get('parent_session_id'),
            'parent_relation_basis': scope[sid].get('relation_basis', 'unknown'),
            'tags': ['source:codex-app', f'session:{sid}']})
        ev = evidence.setdefault(sid, {'events': {}, 'deferred_turns': set(), 'completed_turns': set(), 'oldest_page_seen': False,
                                      'user_messages': 0, 'assistant_messages': 0, 'filtered_messages': 0})
        ev['oldest_page_seen'] |= page['page'].get('hasMore') is False
        for turn in page['turns']:
            if turn['status'] != 'completed':
                ev['deferred_turns'].add(turn['id']); continue
            ev['completed_turns'].add(turn['id'])
            ev['deferred_turns'].discard(turn['id'])
            public_items = [i for i in turn.get('items', []) if i.get('type') in {'userMessage', 'agentMessage'}]
            for item_order, item in enumerate(public_items):
                if not isinstance(item.get('id'), str) or not item['id']:
                    raise ValueError('stable item id required')
                text = _public_text(item)
                if not text or is_noise_message(text) or SECRET_MATERIAL_RE.search(text):
                    ev['filtered_messages'] += 1; continue
                role = 'user' if item['type'] == 'userMessage' else 'assistant'
                ts = timestamp(turn['startedAt'] if role == 'user' else turn['completedAt'])
                origin = f"codex-app:{sid}:{turn['id']}:{item['id']}"
                parent = entry['app_parent_session_id'] or 'unknown'
                header = (f"[source:codex-app session={sid} turn={turn['id']} item={item['id']} "
                          f"parent={parent} parent_basis={entry['parent_relation_basis']} "
                          f"time={ts} time_basis=turn_{'start' if role == 'user' else 'completion'}]\n")
                msg = {'role': role, 'content': header + text, 'timestamp': ts,
                       'origin_message_id': origin, 'source_item_id': item['id'],
                       'source_turn_id': turn['id'], 'source_item_order': item_order}
                old = ev['events'].get(origin)
                if old is not None and old != msg:
                    raise ValueError('source item changed: refetch/review required')
                ev['events'][origin] = msg
    for sid, entry in sessions.items():
        ev = evidence[sid]
        entry['messages'] = sorted(ev['events'].values(), key=_message_order)
        entry['message_count'] = len(entry['messages'])
        entry['coverage'] = {'oldest_page_seen': ev['oldest_page_seen'],
            'user_messages': sum(m['role'] == 'user' for m in entry['messages']),
            'assistant_messages': sum(m['role'] == 'assistant' for m in entry['messages']),
            'deferred_turns': sorted(ev['deferred_turns'] - ev['completed_turns']), 'filtered_messages': ev['filtered_messages'],
            'llm_requests': None, 'llm_attempts': None, 'llm_retries': None, 'llm_successes': None,
            'user_text_coverage': 'only_returned_items_not_full_transcript',
            'history_view': 'supported_read_thread_summary',
            'time_basis': 'turn_boundary_not_message_event',
            'pagination': 'oldest_page_seen_is_not_proof_of_unbroken_cursor_chain'}
    return list(sessions.values())


def _write_private(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True); stream.write('\n')
    os.replace(tmp, path)


def sync_sessions(sessions, output, *, write=False):
    """Union previously published stable events: partial pages never delete history."""
    output = Path(output); reports = []
    for incoming in sessions:
        sid = incoming['source_session_id']
        # Legacy IDs embed a rollout timestamp. Refuse overlapping source identity
        # rather than deduplicating differing event clocks/content heuristically.
        overlap = [p for p in output.glob(f'session_codex-*{sid}.json')
                   if p.name != f"session_{incoming['session_id']}.json"]
        if overlap:
            raise ValueError(f'CLI/app overlap requires identity reconciliation: {sid}')
        path = output / f"session_{incoming['session_id']}.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        events = {m['origin_message_id']: m for m in old.get('messages', [])}
        new = 0
        for msg in incoming['messages']:
            key = msg['origin_message_id']
            if key in events and events[key] != msg:
                raise ValueError('previously published event changed')
            new += key not in events; events[key] = msg
        merged = {**incoming, 'messages': sorted(events.values(), key=_message_order)}
        merged['message_count'] = len(events)
        merged['coverage'] = {**incoming['coverage'],
            'user_messages': sum(m['role'] == 'user' for m in events.values()),
            'assistant_messages': sum(m['role'] == 'assistant' for m in events.values()),
            'oldest_page_seen': incoming['coverage']['oldest_page_seen'] or old.get('coverage', {}).get('oldest_page_seen', False)}
        changed = merged != old
        if write and changed:
            _write_private(path, merged)
        reports.append({'session_id': sid, 'target': str(path), 'new_messages': new,
                        'total_messages': len(events), 'changed': changed,
                        'coverage': merged['coverage']})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True, help='Explicit authorized thread IDs -> parent metadata JSON')
    parser.add_argument('--output-sessions', type=Path, required=True)
    parser.add_argument('--write', action='store_true', help='Only write native snapshots; never invoke a model')
    args = parser.parse_args(argv)
    scope = json.loads(args.scope.read_text())
    pages = [json.loads(p.read_text()) for p in sorted(args.export_dir.glob('page-*.json'))]
    sessions = convert_pages(pages, scope)
    print(json.dumps(sync_sessions(sessions, args.output_sessions, write=args.write), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
