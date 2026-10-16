#!/usr/bin/env python3
"""Inspect or adjust incremental scan coverage (the fingerprint store).

The scan store only accelerates change detection. When it is lost or wrong,
this tool lets an operator delimit the incremental scope quickly instead of
falling back to a full scan:

  show          per-source coverage: stored fingerprint vs the current db
                state and the ledger's captured event count.
  mark-covered  set the store to the CURRENT state for matched sources, so
                the next scan treats them as clean. Refuses without
                --confirm mark-covered when the ledger looks behind.
  mark-dirty    drop the store entries for matched sources so the next scan
                re-reads them strictly.

Matching: --profile NAME, --sid GLOB (fnmatch on the session id), --since
DATE (ISO date/datetime; matches sessions whose started_at is on/after).
Writes are atomic via the ledger's scan-fingerprint store.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hindsight_incremental_ledger import IncrementalLedger
from hindsight_incremental_sources import compute_scan_fingerprints
from hindsight_pipeline_common import load_config

CONFIRM_COVERED = 'mark-covered'
CONFIRM_DIRTY = 'mark-dirty'


def parse_since(value: str | None):
    if value is None:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    raise SystemExit(f'invalid --since value: {value}')


def load_context(config_path: Path | None):
    config = load_config(config_path or Path.home() / '.hermes' / 'hindsight' / 'pipeline_config.json',
                         require_exists=True)
    ledger = IncrementalLedger(Path(config['incremental']['root']), config['bank'])
    baseline_reference = ledger.physical_baseline_reference()
    if baseline_reference is None:
        raise SystemExit('no physical baseline; this tool needs the initialized ledger')
    raw = Path(baseline_reference['artifact_path']).read_bytes()
    baseline = json.loads(raw)
    return config, ledger, baseline


def ledger_event_counts(ledger) -> dict:
    counts = {}
    with ledger._connection() as con:
        for key, in con.execute('SELECT source_key FROM sources'):
            state = ledger._read(con, 'sources', key)
            if isinstance(state, dict) and isinstance(state.get('events'), dict):
                counts[key] = len(state['events'])
    return counts


def matches(profile_name, sid, started_at, args):
    if args.profile and profile_name != args.profile:
        return False
    if args.sid and not fnmatch.fnmatch(sid, args.sid):
        return False
    if args.since is not None and (started_at is None or started_at < args.since):
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('show', 'mark-covered', 'mark-dirty'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--profile')
    parser.add_argument('--sid', help='fnmatch glob on the session id')
    parser.add_argument('--since', help='ISO date/datetime lower bound on session start')
    parser.add_argument('--confirm', help='required token for mark-* actions')
    args = parser.parse_args(argv)
    args.since = parse_since(args.since)

    config, ledger, baseline = load_context(args.config)
    home = Path(config['hermes_home'])
    current = compute_scan_fingerprints(home, physical_baseline=baseline)
    store = ledger.load_scan_fingerprints() or {}
    captured = ledger_event_counts(ledger)

    report = []
    for profile in baseline['profiles']:
        name = profile['profile']
        profile_current = current.get(name) or {}
        sessions = profile_current.get('sessions') or {}
        entry = store.setdefault(name, {'db': None, 'sessions': {}, 'artifacts': {},
                                        'scope_sids': []})
        try:
            from hindsight_incremental_sources import _read_physical_sessions
            started = {sid: row.get('started_at')
                       for sid, row in _read_physical_sessions(Path(profile['state_db'])).items()}
        except Exception:
            started = {}
        for sid, fingerprint in sorted(sessions.items()):
            if not matches(name, sid, started.get(sid), args):
                continue
            key = f"{name}:hermes:{sid}"
            if key not in captured:
                key = f"{name}:cutover-session:{sid}"
            stored = entry['sessions'].get(sid)
            item = {'profile': name, 'sid': sid, 'stored': stored, 'current': fingerprint,
                    'ledger_events': captured.get(key)}
            if args.action == 'show':
                item['state'] = ('clean' if stored == fingerprint else
                                 'dirty' if stored is not None else 'unseeded')
                report.append(item)
            elif args.action == 'mark-dirty':
                if args.confirm != CONFIRM_DIRTY:
                    raise SystemExit(f'mark-dirty requires --confirm {CONFIRM_DIRTY}')
                entry['sessions'].pop(sid, None)
                report.append(item)
            else:  # mark-covered
                if stored == fingerprint:
                    continue
                if args.confirm != CONFIRM_COVERED:
                    raise SystemExit(
                        f'mark-covered requires --confirm {CONFIRM_COVERED}; '
                        f'{len(sessions)} sessions match the range')
                entry['sessions'][sid] = fingerprint
                report.append(item)
        if args.action != 'show':
            scope = set(entry.get('scope_sids') or ())
            scope.update(entry['sessions'])
            entry['scope_sids'] = sorted(scope)
    if args.action == 'show':
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0
    ledger.save_scan_fingerprints(store)
    print(json.dumps({'action': args.action, 'changed': len(report),
                      'profiles': sorted({item['profile'] for item in report})}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
