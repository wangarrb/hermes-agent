#!/usr/bin/env python3
"""Monitor production Hindsight retain queue without printing document content."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PSQL = Path('/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql')
SQL = r"""
WITH op AS (
  SELECT
    count(*) FILTER (WHERE operation_type='retain' AND status='completed') AS retain_completed,
    count(*) FILTER (WHERE operation_type='retain' AND status='pending') AS retain_pending,
    count(*) FILTER (WHERE operation_type='retain' AND status='processing') AS retain_processing,
    count(*) FILTER (WHERE operation_type='batch_retain' AND status='pending' AND task_payload IS NULL) AS parent_pending_null_payload,
    count(*) FILTER (WHERE status='failed') AS failed,
    max(updated_at) AS max_op_updated_at
  FROM async_operations
), docs AS (
  SELECT
    (SELECT count(*) FROM documents WHERE bank_id='hermes') AS hermes_docs,
    (SELECT count(*) FROM memory_units WHERE bank_id='hermes') AS hermes_units
)
SELECT json_build_object(
  'ts', now(),
  'retain_completed', op.retain_completed,
  'retain_pending', op.retain_pending,
  'retain_processing', op.retain_processing,
  'parent_pending_null_payload', op.parent_pending_null_payload,
  'failed', op.failed,
  'hermes_docs', docs.hermes_docs,
  'hermes_units', docs.hermes_units,
  'max_op_updated_at', op.max_op_updated_at
)::text
FROM op, docs;
"""


def query_once() -> dict:
    cp = subprocess.run(
        [str(PSQL), '-h', '127.0.0.1', '-p', '5432', '-U', 'hindsight', '-d', 'hindsight', '-At', '-v', 'ON_ERROR_STOP=1'],
        input=SQL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if cp.returncode != 0:
        return {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'query_error': cp.stderr.strip()[:500], 'returncode': cp.returncode}
    line = cp.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--poll', type=int, default=120)
    ap.add_argument('--timeout-hours', type=float, default=8.0)
    args = ap.parse_args()
    deadline = time.time() + args.timeout_hours * 3600
    while True:
        row = query_once()
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        if row.get('failed', 0):
            print(json.dumps({'event': 'failed_operations_detected', 'failed': row.get('failed')}, ensure_ascii=False), flush=True)
            return 2
        if row.get('retain_pending') == 0 and row.get('retain_processing') == 0:
            print(json.dumps({'event': 'retain_queue_drained'}, ensure_ascii=False), flush=True)
            return 0
        if time.time() >= deadline:
            print(json.dumps({'event': 'monitor_timeout'}, ensure_ascii=False), flush=True)
            return 124
        time.sleep(args.poll)


if __name__ == '__main__':
    raise SystemExit(main())
