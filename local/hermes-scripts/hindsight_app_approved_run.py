#!/usr/bin/env python3
"""Run the existing production chain for exactly the approved frozen three.

Uses the existing ledger, native isolation, receipts and quality gates. It never
captures another source or selects unrelated pending work. Newly discovered
account history remains local-only until separately approved.
"""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
import sys

from hindsight_app_sessions import sync_sessions
from hindsight_incremental_ledger import IncrementalLedger
from hindsight_incremental_pipeline import run_incremental
from hindsight_incremental_runtime import ArrivalRuntime
from hindsight_incremental_sources import scan_source_snapshots, _message, digest, message_hash
from hindsight_pipeline_common import load_config

# Pinned to the original 447-message freeze, independently checked against the
# prior archive manifest/payload hashes. Mutable source proofs never grant approval.
APPROVED_SOURCE_SHA256 = {
    '01a1381d-d6a5-7103-9aba-8d12ea797006': '30f42c75d75c36866096906d1e613c9891816f6eebaee2bbbbac4d6b98c77095',
    '01a137c5-3f40-76b8-a871-b2567a8f6025': 'd38e987e80add1503b71739cd73bcb668c2ba9bc342d9b5189532a370c568f2b',
    '01a13ed4-a7da-71fa-ba33-e9879482c024': '33f743c8126c84ed41254f5a1889dd8171d51e051a05dec1ab797da8aa4468a1',
}
APPROVED = frozenset(APPROVED_SOURCE_SHA256)
TASK = Path('/home/wyr/Documents/Codex/2026-10-15/task')
SOURCE = TASK / 'evidence/staged-home-v2/sessions'
REPORT = TASK / 'evidence/approved-three-production.json'
# This task's existing approval target, never an alternate bank/root CLI option.
APP_LEDGER_ROOT = Path('/home/wyr/.hermes/hindsight/incremental')
APP_HERMES_HOME = Path('/home/wyr/.hermes')


def _proof(messages):
    return {m['origin']: digest({k: m.get(k) for k in ('role', 'timestamp', 'origin', 'content')})
            for m in messages}


def _load_approved_sessions():
    sessions = []
    for sid in sorted(APPROVED):
        raw = (SOURCE / f'session_codex-app-{sid}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != APPROVED_SOURCE_SHA256[sid]:
            raise ValueError('approved frozen source bytes changed')
        sessions.append(json.loads(raw))
    if {s['source_session_id'] for s in sessions} != APPROVED or sum(len(s['messages']) for s in sessions) != 447:
        raise ValueError('approved frozen source changed')
    return sessions


def _app_needs_summary(input_plan, batches):
    if input_plan['new_messages']:
        return True
    parents_with_children = {p for b in batches if b['kind'] == 'daily_summary'
                             for p in b['parent_batch_ids']}
    related = {b['batch_id'] for b in input_plan['related_batches']}
    return any(b['batch_id'] in related and (
        (b['kind'] == 'daily_summary' and b['phase'] == 'captured')
        or (b['kind'] == 'source' and b['phase'] in {'captured', 'generated', 'submitted', 'unknown', 'retained'}
            and b['batch_id'] not in parents_with_children)) for b in batches)


def _execute_app(sessions, ledger, input_plan):
    """Called only after frozen input admission, while holding this bank lock."""
    from hindsight_app_source_adapter import build_app_snapshots

    cfg = dict(load_config())
    if (cfg['bank'] != 'hermes' or cfg['api_url'] != 'http://127.0.0.1:8888'
            or Path(cfg['hermes_home']).resolve() != APP_HERMES_HOME.resolve()
            or Path((cfg.get('incremental') or {}).get('root', '')).resolve() != APP_LEDGER_ROOT.resolve()):
        raise ValueError('approved App production target changed')
    cfg['incremental'] = dict(cfg.get('incremental') or {})
    cfg['incremental'].update(mode='daily', include_wiki=False, resume_only=False)
    expected = {f"default:hermes:{s['session_id']}": {
                    message['event_id']: message_hash(message)
                    for message in (_message(raw, codex=True) for raw in s['messages'])}
                for s in sessions}
    admitted = {s['source_key']: s for s in input_plan['sources']}

    def app_reader(home, **_reader_options):
        if Path(home).resolve() != APP_HERMES_HOME.resolve():
            raise ValueError('approved App reader home changed')
        snapshots = build_app_snapshots(sessions, ledger=ledger, hermes_home=APP_HERMES_HOME,
            approved_session_ids=APPROVED, approved_source_sha256=APPROVED_SOURCE_SHA256)
        if {s['source_key'] for s in snapshots} != set(expected):
            raise ValueError('approved App source scope changed')
        for snapshot in snapshots:
            key = snapshot['source_key']
            if (snapshot['source_epoch'] != admitted[key]['source_epoch']
                    or snapshot['snapshot_hash'] != admitted[key]['snapshot_hash']
                    or snapshot.get('epoch_scope', key) != key
                    or snapshot.get('scope_source_keys', [key]) != [key]
                    or {m['event_id']: message_hash(m) for m in snapshot['messages']} != expected[key]):
                raise ValueError('approved App source input changed')
        return snapshots

    runtime = ArrivalRuntime(cfg)
    runtime.ledger = ledger
    if _app_needs_summary(input_plan, ledger.list_batches()):
        profile = runtime._summary_profile()
        if (profile.get('base_url'), profile.get('model')) != ('http://127.0.0.1:3000/v1', 'auto_cheep_model'):
            raise ValueError('approved summary destination changed')
    report = run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                             execute=True, source_reader=app_reader)
    batches = ledger.list_batches()
    own = {b['batch_id'] for b in batches if b['source_key'] in expected}
    while True:
        expanded = own | {b['batch_id'] for b in batches if b['parent_batch_ids']
                          and set(b['parent_batch_ids']) <= own}
        if expanded == own:
            break
        own = expanded
    approved_batches = [b for b in batches if b['batch_id'] in own]
    result = {'source_mode': 'app', 'scope': sorted(APPROVED), 'approved_messages': 447,
              'input_plan': input_plan, 'runtime_report': report,
              'batches': [{k: b.get(k) for k in ('batch_id', 'document_id', 'operation_id', 'source_key',
                  'kind', 'phase', 'last_successful_phase', 'attempts', 'last_error_class', 'parent_batch_ids')}
                  for b in approved_batches], 'native_llm_attempts': None,
              'native_llm_attempts_coverage': 'server-side telemetry not yet correlated'}
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    tmp.chmod(0o600)
    tmp.replace(REPORT)
    print(json.dumps({'stage': 'approved_run_result', 'source_mode': 'app', 'status': report.get('status'),
                      'batches': len(approved_batches), 'report': str(REPORT)}, ensure_ascii=False), flush=True)
    return 0 if report.get('status') in {'completed', 'no_op'} else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-mode', choices=('physical', 'app'), default='physical')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan-only', action='store_true', help='Inspect frozen App inputs without capture')
    action.add_argument('--execute', action='store_true', help='Execute the explicitly selected approved App inputs')
    parser.add_argument('--transition-certificate', type=Path)
    parser.add_argument('--approved-transition-sha256', help='Canonical SHA pinned independently after parent review')
    args = parser.parse_args(argv)
    if args.source_mode == 'app':
        if not (args.plan_only or args.execute):
            parser.error('App source mode requires --plan-only or explicit --execute')
        if args.transition_certificate or args.approved_transition_sha256:
            parser.error('physical transition options cannot authorize the App candidate')
        from hindsight_app_source_adapter import plan_app_input

        sessions = _load_approved_sessions()
        ledger = IncrementalLedger(APP_LEDGER_ROOT, 'hermes')
        with ledger.locked():
            report = plan_app_input(sessions, ledger=ledger, hermes_home=APP_HERMES_HOME,
                approved_session_ids=APPROVED, approved_source_sha256=APPROVED_SOURCE_SHA256)
            if args.plan_only or report['status'] == 'blocked':
                print(json.dumps(report, ensure_ascii=False), flush=True)
                return 2 if args.execute else 0
            return _execute_app(sessions, ledger, report)
    if args.plan_only or args.execute:
        parser.error('App candidate options require explicit --source-mode app')
    if bool(args.transition_certificate) != bool(args.approved_transition_sha256):
        parser.error('a reviewed certificate and its independent approval pin are both required')
    transition = json.loads(args.transition_certificate.read_text()) if args.transition_certificate else None
    sessions = _load_approved_sessions()
    cfg = load_config()
    if cfg['bank'] != 'hermes' or cfg['api_url'] != 'http://127.0.0.1:8888':
        raise ValueError('approved production target changed')
    if Path((cfg.get('incremental') or {}).get('root', '')).resolve() != APP_LEDGER_ROOT:
        raise ValueError('approved ledger root changed')
    cfg['incremental'] = dict(cfg.get('incremental') or {})
    cfg['incremental']['mode'] = 'daily'
    cfg['incremental']['include_wiki'] = False
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
            options.update(source_transition_certificates=[transition],
                           approved_transition_sha256s=[args.approved_transition_sha256])
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
