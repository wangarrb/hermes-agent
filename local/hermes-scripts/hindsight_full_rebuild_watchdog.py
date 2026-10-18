#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import subprocess
import time
import urllib.request

STATS_URL = 'http://127.0.0.1:8888/v1/default/banks/hermes/stats'
HEALTH_URL = 'http://127.0.0.1:8888/health'
LOG_PATH = '/tmp/hindsight-full-rebuild-watchdog.log'


def now():
    return dt.datetime.now().isoformat(timespec='seconds')


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run(cmd, timeout=600):
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    return {
        'cmd': cmd,
        'returncode': p.returncode,
        'stdout_tail': (p.stdout or '')[-2000:],
        'stderr_tail': (p.stderr or '')[-2000:],
    }


def emit(record):
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wrapper-pid', type=int, required=True)
    ap.add_argument('--interval-s', type=int, default=300)
    ap.add_argument('--settle-s', type=int, default=180)
    args = ap.parse_args()

    health_errors = 0
    emit({'time': now(), 'event': 'watchdog_start', 'wrapper_pid': args.wrapper_pid, 'log_path': LOG_PATH})

    while True:
        try:
            health = fetch_json(HEALTH_URL)
            stats = fetch_json(STATS_URL)
            health_errors = 0
        except Exception as e:
            health_errors += 1
            emit({'time': now(), 'event': 'health_or_stats_error', 'error': repr(e), 'consecutive_errors': health_errors})
            if health_errors >= 6:
                return 3
            time.sleep(args.interval_s)
            continue

        record = {
            'time': now(),
            'event': 'poll',
            'health': health,
            'documents': stats.get('total_documents'),
            'nodes': stats.get('total_nodes'),
            'observations': stats.get('total_observations'),
            'pending': stats.get('pending_operations'),
            'failed': stats.get('failed_operations'),
            'operations_by_status': stats.get('operations_by_status'),
            'wrapper_alive': pid_alive(args.wrapper_pid),
        }
        emit(record)

        failed = int(stats.get('failed_operations') or 0)
        pending = int(stats.get('pending_operations') or 0)
        processing = int((stats.get('operations_by_status') or {}).get('processing') or 0)

        if failed > 0:
            emit({'time': now(), 'event': 'failed_operations_detected', 'last': record})
            return 2

        if pending == 0 and processing == 0:
            emit({'time': now(), 'event': 'queue_drained_waiting_for_wrapper', 'settle_s': args.settle_s})
            deadline = time.time() + args.settle_s
            while time.time() < deadline and pid_alive(args.wrapper_pid):
                time.sleep(10)
            if pid_alive(args.wrapper_pid):
                emit({'time': now(), 'event': 'wrapper_still_alive_after_settle_restore_normal_local'})
                restore = run(['python3', '/home/wyr/.hermes/scripts/hindsight_minimax_import.py', 'normal-local'], timeout=900)
                emit({'time': now(), 'event': 'normal_local_restore_result', 'result': restore})
            else:
                emit({'time': now(), 'event': 'wrapper_exited_before_forced_restore'})

            try:
                final_health = fetch_json(HEALTH_URL)
                final_stats = fetch_json(STATS_URL)
            except Exception as e:
                emit({'time': now(), 'event': 'final_fetch_error', 'error': repr(e)})
                return 4
            emit({
                'time': now(),
                'event': 'done',
                'health': final_health,
                'documents': final_stats.get('total_documents'),
                'nodes': final_stats.get('total_nodes'),
                'observations': final_stats.get('total_observations'),
                'pending': final_stats.get('pending_operations'),
                'failed': final_stats.get('failed_operations'),
                'operations_by_status': final_stats.get('operations_by_status'),
            })
            return 0

        time.sleep(args.interval_s)


if __name__ == '__main__':
    raise SystemExit(main())
