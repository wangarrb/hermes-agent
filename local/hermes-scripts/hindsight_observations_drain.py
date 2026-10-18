#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest

API = 'http://127.0.0.1:8888'
BANK = 'hermes'
PSQL = '/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql'
REPORT_DIR = Path.home() / '.hermes' / 'hindsight' / 'reports'
LOG_DIR = Path.home() / '.hermes' / 'logs' / 'hindsight-observations'
MAX_RUNTIME_SECONDS = 48 * 3600


def psql(sql: str) -> str:
    last_err = ''
    for attempt in range(12):
        proc = subprocess.run(
            [PSQL, '-h', '/tmp', '-p', '5432', '-U', 'hindsight', '-d', 'hindsight', '-q', '-t', '-A', '-c', sql],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        last_err = proc.stderr.strip() or proc.stdout.strip()
        if 'too many clients already' in last_err or 'connection to server' in last_err:
            time.sleep(min(5 + attempt * 5, 60))
            continue
        raise RuntimeError(last_err)
    raise RuntimeError(last_err)


def snap() -> dict:
    raw = psql("""
select jsonb_build_object(
 'ts', now()::text,
 'observations', (select count(*) from memory_units where bank_id='hermes' and fact_type='observation'),
 'unconsolidated_base', (select count(*) from memory_units where bank_id='hermes' and fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null),
 'failed_base', (select count(*) from memory_units where bank_id='hermes' and fact_type in ('world','experience') and consolidation_failed_at is not null),
 'ops', (select coalesce(jsonb_object_agg(status||':'||operation_type,c),'{}'::jsonb) from (select status,operation_type,count(*) c from async_operations where bank_id='hermes' group by status,operation_type) s),
 'offline_docs', (with od as (select d.id, coalesce(count(m.id),0) units from documents d left join memory_units m on m.bank_id=d.bank_id and m.document_id=d.id where d.bank_id='hermes' and d.id like 'hermes-offline-consolidation::%' group by d.id) select jsonb_build_object('total',count(*),'zero',count(*) filter(where units=0),'with_units',count(*) filter(where units>0),'units',coalesce(sum(units),0)) from od)
)::text;
""")
    return json.loads(raw) if raw else {}


def post(path: str) -> dict:
    req = urlrequest.Request(
        f'{API}/v1/default/banks/{BANK}/{path}',
        data=b'{}',
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlrequest.urlopen(req, timeout=30) as r:
        raw = r.read().decode('utf-8')
    return json.loads(raw) if raw else {}


def active_ops(s: dict) -> int:
    ops = s.get('ops') or {}
    return sum(v for k, v in ops.items() if k.startswith('pending:') or k.startswith('processing:'))


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = LOG_DIR / f'{stamp}-drain.jsonl'
    final_path = REPORT_DIR / f'{stamp}-observations-drain-final.json'
    start = time.time()
    triggered = []
    last = None
    with log_path.open('w', encoding='utf-8') as log:
        while True:
            try:
                s = snap()
            except Exception as e:
                s = {
                    'ts': datetime.now(timezone.utc).isoformat(),
                    'error': 'snap_failed',
                    'message': str(e),
                }
                log.write(json.dumps(s, ensure_ascii=False) + '\n')
                log.flush()
                print(json.dumps(s, ensure_ascii=False), flush=True)
                if time.time() - start > MAX_RUNTIME_SECONDS:
                    print('timeout_48h', flush=True)
                    break
                time.sleep(30)
                continue
            last = s
            log.write(json.dumps(s, ensure_ascii=False) + '\n')
            log.flush()
            print(json.dumps(s, ensure_ascii=False), flush=True)
            if active_ops(s) == 0:
                failed = s.get('failed_base') or 0
                uncon = s.get('unconsolidated_base') or 0
                if failed:
                    rec = post('consolidation/recover')
                    triggered.append({'ts': datetime.now(timezone.utc).isoformat(), 'recover': rec})
                    print('recover', json.dumps(rec, ensure_ascii=False), flush=True)
                elif uncon:
                    resp = post('consolidate')
                    triggered.append({'ts': datetime.now(timezone.utc).isoformat(), 'consolidate': resp})
                    print('trigger', json.dumps(resp, ensure_ascii=False), flush=True)
                else:
                    break
            if time.time() - start > MAX_RUNTIME_SECONDS:
                print('timeout_48h', flush=True)
                break
            time.sleep(30)
    final = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'last': last,
        'triggered': triggered,
        'log_path': str(log_path),
    }
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    print('final_report=' + str(final_path), flush=True)


if __name__ == '__main__':
    main()
