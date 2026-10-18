#!/usr/bin/env python3
"""One read-only monitoring snapshot; safe for no-agent periodic cron.

No source text, credentials, lease tokens, POST/PATCH, LLM or repair action.
A backend being healthy never means the full pipeline succeeded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import time
import urllib.request

from hindsight_pipeline_common import load_config

PHASES = frozenset({'captured', 'generated', 'submitted', 'unknown', 'retained',
                   'consolidated', 'locally_merged', 'gated', 'published', 'readback_verified'})
KINDS = frozenset({'source', 'daily_summary'})
RETRY_CLASSES = frozenset({'transport', 'rate_limit', 'operation_pending', 'quality_gate',
    'call_cap', 'llm_call_cap', 'deadline', 'operation_unknown', 'operation_failed',
    'invalid_output', 'bank_mismatch', 'internal_error', 'cleanup'})


def _retry_class(value):
    if isinstance(value, str) and (value in RETRY_CLASSES or re.fullmatch(r'http_[1-5][0-9]{2}', value)):
        return value
    return 'unrecognized'


def _processes():
    targets = {'hindsight_incremental_pipeline.py', 'hindsight_memory_pipeline.py',
               'hindsight_daily_noagent.py', 'hindsight_weekly_wrapper.py'}
    out = subprocess.run(['ps', '-eo', 'pid,etime,args'], text=True, capture_output=True, timeout=5)
    rows = []
    for line in out.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            args = shlex.split(parts[2])
        except ValueError:
            continue
        names = [Path(x).name for x in args if Path(x).name in targets]
        if names:
            rows.append({'pid': int(parts[0]), 'elapsed': parts[1], 'script': names[0]})
    return rows


def collect(config, *, request=None, process_reader=_processes, now=time.time):
    bank = config['bank']
    if not isinstance(bank, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', bank) or bank in {'.', '..'}:
        raise ValueError('invalid_bank')
    if request is None:
        base = str(config.get('api_url') or 'http://127.0.0.1:8888').rstrip('/')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def fetch(path):
            with opener.open(urllib.request.Request(base + path, method='GET'), timeout=8) as response:
                return json.loads(response.read())
        request = fetch
    backend = {'healthy': False, 'pending_operations': None, 'pending_consolidation': None,
               'enable_observations': None, 'enable_auto_consolidation': None, 'errors': []}
    values = {}
    for key, path in [('health', '/health'), ('stats', f'/v1/default/banks/{bank}/stats'),
                      ('config', f'/v1/default/banks/{bank}/config')]:
        try:
            value = request(path)
            if not isinstance(value, dict):
                raise ValueError('invalid_response')
            values[key] = value
        except Exception as error:
            backend['errors'].append({'endpoint': key, 'type': type(error).__name__})
    backend['healthy'] = values.get('health', {}).get('status') == 'healthy'
    for key in ('pending_operations', 'pending_consolidation'):
        number = values.get('stats', {}).get(key)
        backend[key] = number if type(number) is int and number >= 0 else None
    for key in ('enable_observations', 'enable_auto_consolidation'):
        value = (values.get('config', {}).get('config') or {}).get(key)
        backend[key] = value if type(value) is bool else None
    root = Path((config.get('incremental') or {}).get('root') or
                Path(config['hermes_home']) / 'hindsight' / 'incremental') / bank
    state = {'initialized': False, 'phase_counts': {}, 'kind_counts': {}, 'due_batches': 0,
             'retry_classes': {}, 'next_retry_at': None,
             'window_status': None, 'producer_pid': None, 'error': None}
    db = root / 'ledger.sqlite3'
    if db.exists():
        try:
            if db.is_symlink() or db.stat().st_nlink != 1:
                raise ValueError('unsafe_state_path')
            con = sqlite3.connect(db.absolute().as_uri() + '?mode=ro', uri=True, timeout=1)
            try:
                con.execute('PRAGMA query_only=ON')
                phases, kinds, errors, retry_times, due = Counter(), Counter(), Counter(), [], 0
                for (text,) in con.execute('SELECT state FROM batches'):
                    row = json.loads(text)
                    phase, kind = row.get('phase'), row.get('kind')
                    phases[phase if isinstance(phase, str) and phase in PHASES else 'unrecognized'] += 1
                    kinds[kind if isinstance(kind, str) and kind in KINDS else 'unrecognized'] += 1
                    if phase not in {'published', 'readback_verified'}:
                        retry = row.get('next_retry_at', 0)
                        if type(retry) not in (int, float) or not math.isfinite(retry) or retry < 0:
                            raise ValueError('invalid_retry_time')
                        if retry <= now():
                            due += 1
                        if row.get('last_error_class'):
                            errors[_retry_class(row['last_error_class'])] += 1
                            retry_times.append(retry)
                initialized = False
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'meta' in tables:
                    row = con.execute("SELECT state FROM meta WHERE key='legacy'").fetchone()
                    if row:
                        legacy = json.loads(row[0])
                        initialized = legacy.get('bank') == bank and legacy.get('policy') == 'preserve-no-replay'
                state.update(initialized=initialized, phase_counts=dict(phases), kind_counts=dict(kinds), due_batches=due,
                    retry_classes=dict(errors), next_retry_at=min(retry_times, default=None))
            finally:
                con.close()
        except Exception as error:
            state['error'] = type(error).__name__
    lease = root / 'runtime-window.json'
    if lease.exists() and not lease.is_symlink():
        try:
            value = json.loads(lease.read_text())
            state['window_status'] = value.get('status')
            pid = (value.get('owner') or {}).get('pid')
            state['producer_pid'] = pid if type(pid) is int and pid > 0 else None
        except Exception as error:
            state['error'] = type(error).__name__
    alerts = []
    if backend['healthy'] is not True or backend['errors']:
        alerts.append('backend_unavailable')
    if backend['pending_consolidation'] and not state['initialized']:
        alerts.append('legacy_backlog')
    if state['window_status'] in {'cleanup_pending', 'restoring', 'opening'}:
        alerts.append('cleanup_pending')
    if state['error']:
        alerts.append('state_read_error')
    return {'schema_version': 'hindsight-readonly-monitor/1', 'timestamp': now(), 'bank': bank,
            'read_only': True, 'backend': backend, 'ledger': state,
            'processes': process_reader(), 'alerts': alerts,
            'completion_claim': 'requires_exact_run_gate_and_publish_readback'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    report = collect(load_config(args.config, require_exists=True))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
