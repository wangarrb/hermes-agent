#!/usr/bin/env python3
"""Pre-consolidation cleanup: fix orphaned memory_units after container restart.

Hindsight v0.6.1 bug: after a container force-restart, memory_units created by
successful retain operations may have ``consolidated_at=NULL`` with no matching
active ``async_operations``. The consolidation worker won't pick them up, causing
``pending_consolidation`` to be stuck at a non-zero value and blocking
``wait_native_consolidation``.

This script checks for *truly* orphaned units (no active/pending consolidation op)
and either:
1. Re-queues them by creating new async_operations (preferred), or
2. Marks them as consolidated to unblock the pipeline (fallback).

It is safe to run while consolidation is active: it only touches units when there
are no active/pending consolidation operations for the bank.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # Will fall back to psql CLI

logger = logging.getLogger("consolidation_cleanup")


def has_active_consolidation(cur, bank: str) -> bool:
    """Check if there are active/pending consolidation operations for this bank."""
    cur.execute(
        "SELECT COUNT(*) FROM async_operations "
        "WHERE bank_id=%s AND operation_type='consolidation' "
        "AND status IN ('processing','pending')",
        (bank,),
    )
    return cur.fetchone()[0] > 0


def _find_psql() -> str | None:
    """Find psql binary — system PATH, Hindsight docker installation, or docker exec."""
    import shutil
    psql = shutil.which("psql")
    if psql:
        return psql
    candidate = __import__("pathlib").Path("/home/wyr/.hindsight-docker/installation") / "18.1.0" / "bin" / "psql"
    if candidate.exists():
        return str(candidate)
    return None


def _run_psql(psql_bin: str, db_url: str, sql: str, *, timeout: int = 15) -> str:
    """Run a SQL statement via psql CLI and return stdout."""
    import subprocess as sp
    r = sp.run([psql_bin, db_url, "-t", "-c", sql], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix orphaned pending_consolidation units")
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--db-url", default="postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bypass", action="store_true", help="Mark as consolidated (bypass actual processing)")
    ap.add_argument("--force", action="store_true", help="Run even if active consolidation operations exist (dangerous)")
    args = ap.parse_args()

    # --- Path 1: psycopg2 ---
    if psycopg2 is not None:
        try:
            return _main_psycopg2(args)
        except Exception as exc:
            logger.error(f"psycopg2 failed: {exc}, trying psql CLI fallback")

    # --- Path 2: psql CLI fallback ---
    psql_bin = _find_psql()
    if not psql_bin:
        logger.error("Neither psycopg2 nor psql CLI available; cannot fix orphans")
        return 2

    db_url = args.db_url
    bank = args.bank

    # Safety check
    if not args.force:
        active = _run_psql(psql_bin, db_url,
            f"SELECT COUNT(*) FROM async_operations "
            f"WHERE bank_id='{bank}' AND operation_type='consolidation' "
            f"AND status IN ('processing','pending')")
        if int(active or 0) > 0:
            print(f"Skipping: active/pending consolidation operations exist for bank {bank}. "
                  "Use --force to override.", file=sys.stderr)
            return 0

    # Count orphaned
    orphaned = _run_psql(psql_bin, db_url,
        f"SELECT COUNT(*) FROM memory_units "
        f"WHERE bank_id='{bank}' AND consolidated_at IS NULL "
        f"AND fact_type IN ('experience','world')")
    orphaned = int(orphaned or 0)

    if orphaned == 0:
        print(f"No orphaned consolidation units for {bank}")
        return 0

    print(f"Found {orphaned} orphaned units for bank {bank}")

    if args.dry_run:
        print(f"Dry run: would {'bypass' if args.bypass else 're-queue'} {orphaned} units")
        return 0

    if args.bypass:
        # Also reset failed consolidation flags
        _run_psql(psql_bin, db_url,
            f"UPDATE memory_units SET consolidation_failed_at=NULL "
            f"WHERE bank_id='{bank}' AND consolidation_failed_at IS NOT NULL")
        _run_psql(psql_bin, db_url,
            f"UPDATE memory_units SET consolidated_at=NOW() "
            f"WHERE bank_id='{bank}' AND consolidated_at IS NULL "
            f"AND fact_type IN ('experience','world')")
        print(f"Bypassed {orphaned} orphaned units (marked as consolidated) via psql CLI")
    else:
        # Re-queue: create async_operation
        import uuid, json
        op_id = str(uuid.uuid4())
        payload = json.dumps({
            "type": "consolidation",
            "bank_id": bank,
            "max_memories_per_round": 64,
            "llm_batch_size": 8,
        })
        _run_psql(psql_bin, db_url,
            f"INSERT INTO async_operations "
            f"(operation_id, bank_id, operation_type, status, created_at, updated_at, task_payload, retry_count) "
            f"VALUES ('{op_id}', '{bank}', 'consolidation', 'pending', NOW(), NOW(), "
            f"'{payload.replace(chr(39), chr(39)+chr(39))}', 0)")
        print(f"Created operation {op_id} for {orphaned} orphaned units via psql CLI")

    return 0


def _main_psycopg2(args) -> int:
    """Original psycopg2 implementation."""
    conn = psycopg2.connect(args.db_url)
    cur = conn.cursor()

    # Safety check: don't touch units while consolidation is actively processing
    if not args.force and has_active_consolidation(cur, args.bank):
        print(f"Skipping: active/pending consolidation operations exist for bank {args.bank}. "
              "Use --force to override (may corrupt in-flight work).", file=sys.stderr)
        conn.close()
        return 0

    # Find orphaned units (unconsolidated source facts)
    cur.execute(
        "SELECT COUNT(*) FROM memory_units "
        "WHERE bank_id=%s AND consolidated_at IS NULL AND fact_type IN ('experience','world')",
        (args.bank,),
    )
    orphaned = cur.fetchone()[0]

    if orphaned == 0:
        print(f"No orphaned consolidation units for {args.bank}")
        conn.close()
        return 0

    print(f"Found {orphaned} orphaned units for bank {args.bank}")

    if args.dry_run:
        print(f"Dry run: would {'bypass' if args.bypass else 're-queue'} {orphaned} units")
        conn.close()
        return 0

    if args.bypass:
        # Fallback: mark as consolidated to unblock pipeline
        # Also reset failed consolidation flags
        cur.execute(
            "UPDATE memory_units SET consolidation_failed_at=NULL "
            "WHERE bank_id=%s AND consolidation_failed_at IS NOT NULL",
            (args.bank,),
        )
        now = datetime.now(timezone.utc)
        cur.execute(
            "UPDATE memory_units SET consolidated_at=%s "
            "WHERE bank_id=%s AND consolidated_at IS NULL AND fact_type IN ('experience','world')",
            (now, args.bank),
        )
        updated = cur.rowcount
        conn.commit()
        print(f"Bypassed {updated} orphaned units (marked as consolidated)")
    else:
        # Preferred: create async_operations for the worker to pick up
        now = datetime.now(timezone.utc)
        import uuid, json
        payload = json.dumps({
            "type": "consolidation",
            "bank_id": args.bank,
            "max_memories_per_round": 64,
            "llm_batch_size": 8,
        })
        op_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO async_operations "
            "(operation_id, bank_id, operation_type, status, created_at, updated_at, task_payload, retry_count) "
            "VALUES (%s, %s, 'consolidation', 'pending', %s, %s, %s, 0)",
            (op_id, args.bank, now, now, payload),
        )
        conn.commit()
        print(f"Created operation {op_id} for {orphaned} orphaned units")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
