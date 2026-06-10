#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest

DEFAULT_API = 'http://127.0.0.1:8888'
DEFAULT_BANK = 'hermes'
DEFAULT_PSQL = '/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql'
REPORT_DIR = Path.home() / '.hermes' / 'hindsight' / 'reports'


def post_json(url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload or {}).encode('utf-8')
    req = urlrequest.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode('utf-8', errors='replace')
    return json.loads(raw) if raw.strip() else {}


def psql(sql: str, args) -> str:
    cmd = [args.psql, '-h', args.host, '-p', str(args.port), '-U', args.user, '-d', args.db, '-q', '-t', '-A', '-F', '\t', '-c', sql]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def snapshot(args) -> dict:
    sql = f"""
select jsonb_build_object(
 'ts', now()::text,
 'docs', (select count(*) from documents where bank_id='{args.bank}'),
 'memory_units', (select count(*) from memory_units where bank_id='{args.bank}'),
 'observations', (select count(*) from memory_units where bank_id='{args.bank}' and fact_type='observation'),
 'unconsolidated_base', (select count(*) from memory_units where bank_id='{args.bank}' and fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null),
 'failed_base', (select count(*) from memory_units where bank_id='{args.bank}' and fact_type in ('world','experience') and consolidation_failed_at is not null),
 'operations', (select coalesce(jsonb_object_agg(status || ':' || operation_type, c), '{{}}'::jsonb) from (select status, operation_type, count(*) c from async_operations where bank_id='{args.bank}' group by status,operation_type) s),
 'offline_docs', (with od as (select d.id, coalesce(count(m.id),0) units from documents d left join memory_units m on m.bank_id=d.bank_id and m.document_id=d.id where d.bank_id='{args.bank}' and d.id like 'hermes-offline-consolidation::%' group by d.id) select jsonb_build_object('total',count(*),'zero',count(*) filter(where units=0),'with_units',count(*) filter(where units>0),'units',coalesce(sum(units),0)) from od)
)::text;
"""
    raw = psql(sql, args)
    return json.loads(raw) if raw else {}


def wait(args) -> list[dict]:
    rows = []
    start = time.time()
    while True:
        snap = snapshot(args)
        rows.append(snap)
        ops = snap.get('operations') or {}
        pending = sum(v for k, v in ops.items() if k.startswith('pending:') or k.startswith('processing:'))
        print(json.dumps(snap, ensure_ascii=False), flush=True)
        if pending == 0:
            return rows
        if time.time() - start > args.timeout:
            return rows
        time.sleep(args.poll)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default=DEFAULT_API)
    ap.add_argument('--bank', default=DEFAULT_BANK)
    ap.add_argument('--psql', default=DEFAULT_PSQL)
    ap.add_argument('--host', default='/tmp')
    ap.add_argument('--port', type=int, default=5432)
    ap.add_argument('--db', default='hindsight')
    ap.add_argument('--user', default='hindsight')
    ap.add_argument('--mode', choices=['status','trigger','wait'], default='status')
    ap.add_argument('--poll', type=int, default=30)
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--report-stem', default=None)
    args = ap.parse_args(argv)

    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'mode': args.mode, 'before': snapshot(args)}
    if args.mode == 'trigger':
        report['trigger_response'] = post_json(f"{args.api.rstrip('/')}/v1/default/banks/{args.bank}/consolidate")
        report['after_trigger'] = snapshot(args)
    if args.mode in ('trigger','wait'):
        report['timeline'] = wait(args)
        report['after_wait'] = snapshot(args)
    else:
        print(json.dumps(report['before'], ensure_ascii=False, indent=2))

    stem = args.report_stem or f"hindsight-observations-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f'{stem}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    print(f'report={path}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
