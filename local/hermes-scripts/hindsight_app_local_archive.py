#!/usr/bin/env python3
"""Persist scoped app snapshots and daily evidence without any model/API calls.

The archive root must stay outside production sessions/profiles and the live
incremental ledger. Captured local documents are not retained Hindsight facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import date, datetime, timedelta

from hindsight_app_sessions import sync_sessions
from hindsight_app_usage import render_app_daily, UTC8
from hindsight_incremental_ledger import IncrementalLedger
from hindsight_incremental_sources import checked_path, scan_source_snapshots


def private_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and path.read_text() == content:
        return False
    temp = path.with_suffix(path.suffix + '.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return True


def archive_local(source, archive, hermes_home, scope):
    source, archive, hermes_home = map(lambda p: checked_path(Path(p)), (source, archive, hermes_home))
    blocked = [hermes_home / 'sessions', hermes_home / 'profiles', hermes_home / 'hindsight/incremental']
    if archive == hermes_home or any(archive.is_relative_to(p) or p.is_relative_to(archive) for p in blocked):
        raise ValueError('archive must not overlap an automatically processed production feed')
    incoming = [json.loads(p.read_text()) for p in sorted(source.glob('session_*.json'))]
    if not incoming or any(s.get('source_session_id') not in scope or
                           s.get('source') not in {'codex-app', 'chatgpt-app'} for s in incoming):
        raise ValueError('archive inputs must match explicit authorized scope')
    reports = sync_sessions(incoming, archive / 'sessions', write=True)
    ledger_root = archive / 'local-capture'
    exists = (ledger_root / 'app-archive/ledger.sqlite3').exists()
    ledger = IncrementalLedger(ledger_root, 'app-archive', create=not exists)
    if not exists:
        ledger.initialize_legacy([], {'purpose': 'local_archive_only'}, [],
                                 confirm='initialize-hindsight-incremental')
    snapshots = scan_source_snapshots(archive, profile_mode='none')
    # These are local preservation documents, never model input. Keep a long
    # public message whole; production retain budgets remain unchanged.
    budgets = {s['source_key']: max([60000] + [len(m['content']) for m in s['messages']]) for s in snapshots}
    batches = [b for snapshot in snapshots for b in ledger.capture_snapshot(
        snapshot, max_input_chars=budgets[snapshot['source_key']])]
    all_batches = ledger.list_batches()
    sessions = [json.loads(p.read_text()) for p in sorted((archive / 'sessions').glob('session_*.json'))]
    if any(s.get('source_session_id') not in scope for s in sessions):
        raise ValueError('archive scope cannot silently shrink')
    manifest = {
        'schema': 'app-local-archive/1', 'scope': sorted(scope),
        'sessions': reports, 'messages': sum(len(s['messages']) for s in sessions),
        'local_documents': len(all_batches), 'local_document_state': 'captured_not_retained',
        'new_messages': sum(r['new_messages'] for r in reports), 'new_local_documents': len(batches),
        'local_preservation_document_char_budgets': budgets,
        'production_input_budget_changed': False,
        'production_feed_written': False, 'hindsight_api_calls': 0, 'embedding_calls': 0,
        'summary_llm_calls': 0, 'requires_destination_approval_before_model_processing': True,
        'documents': [{k: b[k] for k in ('document_id', 'source_key', 'source_event_ids', 'payload_path', 'payload_hash', 'phase')}
                      for b in all_batches],
    }
    private_write(archive / 'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest, sessions


def daily_local(sessions, evidence_dir, canonical_dir, *, today=None):
    """Refresh observed UTC+8 days and append evidence to existing next-day reports.

    Current/future dates stay partial. A sidecar is evidence coverage, never a
    claim that the account or that calendar day has been fully enumerated.
    """
    today = date.fromisoformat(today) if isinstance(today, str) else today or datetime.now(UTC8).date()
    outputs = []
    days = sorted({datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00')).astimezone(UTC8).date()
                   for s in sessions for m in s.get('messages', [])})
    for day in days:
        block = render_app_daily(sessions, day)
        partial = day >= today
        name = f"{day.isoformat()}{'.partial' if partial else ''}.md"
        content = '本地封存可访问会话的部分覆盖；不是整日全账号统计。\n\n' + block
        path = canonical_dir / 'app-sessions' / name
        changed = private_write(path, content)
        outputs.append({'path': str(path), 'changed': changed})
        if partial:
            continue
        canonical = canonical_dir / f'{(day + timedelta(days=1)).isoformat()}.md'
        if not canonical.exists():
            continue
        original = canonical.read_text()
        start, end = '<!-- app-session-local-evidence:start -->', '<!-- app-session-local-evidence:end -->'
        replacement = start + '\n' + block + end
        if start in original:
            if original.count(start) != 1 or original.count(end) != 1:
                raise ValueError('ambiguous app daily evidence block')
            a, b = original.index(start), original.index(end) + len(end)
            updated = original[:a] + replacement + original[b:]
        else:
            updated = original.rstrip() + '\n\n' + replacement + '\n'
        if updated != original:
            sha = hashlib.sha256(original.encode()).hexdigest()
            private_write(evidence_dir / f'daily-{canonical.stem}-before-{sha}.md', original)
        outputs.append({'path': str(canonical), 'changed': private_write(canonical, updated),
                        'base_report_regenerated': False, 'only_local_app_evidence_appended': True})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--hermes-home', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--daily-dir', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest, sessions = archive_local(args.source, args.archive, args.hermes_home, json.loads(args.scope.read_text()))
    result = {k: manifest[k] for k in ('messages', 'local_documents', 'new_messages', 'new_local_documents',
                                      'production_feed_written', 'hindsight_api_calls', 'summary_llm_calls', 'embedding_calls')}
    result['archive'] = str(args.archive)
    result['daily_outputs'] = daily_local(sessions, args.evidence_dir, args.daily_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
