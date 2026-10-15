#!/usr/bin/env python3
"""Run the existing production chain for exactly the approved frozen three.

Uses the existing ledger, native isolation, receipts and quality gates. It never
captures another source or selects unrelated pending work. Newly discovered
account history remains local-only until separately approved.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

from hindsight_app_sessions import sync_sessions
from hindsight_incremental_ledger import IncrementalLedger
from hindsight_incremental_pipeline import run_incremental
from hindsight_incremental_runtime import ArrivalRuntime
from hindsight_incremental_sources import scan_source_snapshots, _message, digest
from hindsight_pipeline_common import load_config

APPROVED = frozenset({
    '01a1381d-d6a5-7103-9aba-8d12ea797006',
    '01a137c5-3f40-76b8-a871-b2567a8f6025',
    '01a13ed4-a7da-71fa-ba33-e9879482c024',
})
TASK = Path('/home/wyr/Documents/Codex/2026-10-15/task')
SOURCE = TASK / 'evidence/staged-home-v2/sessions'
REPORT = TASK / 'evidence/approved-three-production.json'


def _proof(messages):
    return {m['origin']: digest({k: m.get(k) for k in ('role', 'timestamp', 'origin', 'content')})
            for m in messages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transition-certificate', type=Path)
    parser.add_argument('--approved-transition-sha256', help='Canonical SHA pinned independently after parent review')
    args = parser.parse_args(argv)
    if bool(args.transition_certificate) != bool(args.approved_transition_sha256):
        parser.error('a reviewed certificate and its independent approval pin are both required')
    transition = json.loads(args.transition_certificate.read_text()) if args.transition_certificate else None
    cfg = load_config()
    if cfg['bank'] != 'hermes' or cfg['api_url'] != 'http://127.0.0.1:8888':
        raise ValueError('approved production target changed')
    cfg['incremental'] = dict(cfg.get('incremental') or {})
    cfg['incremental']['mode'] = 'daily'
    cfg['incremental']['include_wiki'] = False
    sessions = [json.loads((SOURCE / f'session_codex-app-{sid}.json').read_text()) for sid in sorted(APPROVED)]
    if {s['source_session_id'] for s in sessions} != APPROVED or sum(len(s['messages']) for s in sessions) != 447:
        raise ValueError('approved frozen source changed')
    expected = {f"default:hermes:{s['session_id']}": _proof([_message(m, codex=True) for m in s['messages']])
                for s in sessions}
    runtime = ArrivalRuntime(cfg)
    ledger = IncrementalLedger(Path(cfg['incremental']['root']), cfg['bank'])
    runtime.ledger = ledger
    profile = runtime._summary_profile()
    if (profile.get('base_url'), profile.get('model')) != ('http://127.0.0.1:3000/v1', 'auto_cheep_model'):
        raise ValueError('approved summary destination changed')
    # Read-only attestation uses the existing native guard; never bypass it.
    runtime.preflight(require_llm=True)

    def scoped_reader(home, **options):
        if transition is not None:
            options.update(source_transition_certificate=transition,
                           approved_transition_sha256=args.approved_transition_sha256)
        selected = [s for s in scan_source_snapshots(home, **options) if s['source_key'] in expected]
        if {s['source_key'] for s in selected} != set(expected):
            raise ValueError('approved source absent from native physical reader')
        if any(_proof(s['messages']) != expected[s['source_key']] for s in selected):
            raise ValueError('approved source input changed')
        return selected

    with ledger.locked():
        batches = ledger.list_batches()
        own = {b['batch_id'] for b in batches if b.get('source_key') in expected}
        changed = True
        while changed:
            before = len(own)
            own |= {b['batch_id'] for b in batches if b.get('parent_batch_ids')
                    and set(b['parent_batch_ids']) <= own}
            changed = len(own) != before
        if any(b['phase'] != 'published' and b['batch_id'] not in own for b in batches):
            raise ValueError('unrelated pending work exists; do not expand approved processing')
        sync = sync_sessions(sessions, Path(cfg['hermes_home']) / 'sessions', write=True)
        print(json.dumps({'stage': 'approved_source_sync', 'sessions': 3,
                          'new_messages': sum(x['new_messages'] for x in sync)}, ensure_ascii=False), flush=True)
        report = run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 execute=True, source_reader=scoped_reader)
        approved_batches = [b for b in ledger.list_batches() if b.get('source_key') in expected
                            or any(p in own for p in b.get('parent_batch_ids', []))]
        # Include freshly generated children as well as source parents.
        source_ids = {b['batch_id'] for b in approved_batches if b['kind'] == 'source'}
        approved_batches = [b for b in ledger.list_batches() if b['batch_id'] in source_ids
                            or set(b.get('parent_batch_ids') or []) & source_ids]
        result = {'scope': sorted(APPROVED), 'approved_messages': 447, 'runtime_report': report,
                  'approved_transition_sha256': args.approved_transition_sha256,
                  'batches': [{k: b.get(k) for k in ('batch_id', 'document_id', 'operation_id', 'source_key',
                     'kind', 'phase', 'last_successful_phase', 'attempts', 'last_error_class', 'parent_batch_ids')}
                              for b in approved_batches],
                  'native_llm_attempts': None, 'native_llm_attempts_coverage': 'server-side telemetry not yet correlated'}
        tmp = REPORT.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n'); tmp.chmod(0o600); tmp.replace(REPORT)
        print(json.dumps({'stage': 'approved_run_result', 'status': report.get('status'),
                          'batches': len(approved_batches), 'report': str(REPORT)}, ensure_ascii=False), flush=True)
    return 0 if report.get('status') in {'completed', 'no_op'} else 2


if __name__ == '__main__':
    sys.exit(main())
