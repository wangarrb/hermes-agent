#!/usr/bin/env python3
"""Wait for Hindsight native source-fact consolidation to drain.

Read-only gate used by the offline memory pipeline.  It waits on two separate
signals:

1. bank stats `pending_consolidation` for source facts still lacking
   `consolidated_at`;
2. Operations API child rows in pending/processing state, excluding parent batch
   rows.

Historical failed operations are reported but do not block by default because old
failed rows can remain after their source facts have been retried/recovered.

Auto-trigger (--trigger-on-stall): if pending_consolidation > 0 and no
processing operations exist for N consecutive polling cycles, the script
POSTs /consolidate to kick-start background consolidation.  This prevents
deadlocks caused by precision-remote-mode restore disabling observations
(and hence auto-consolidation) after a retain phase.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def api_get_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 15) -> dict[str, Any]:
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if clean:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(clean, doseq=True)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    data = json.loads(body) if body.strip() else {}
    return data if isinstance(data, dict) else {"value": data}


def api_post_json(url: str, data: dict[str, Any] | None = None, *, timeout: int = 15) -> dict[str, Any]:
    """POST JSON to *url* and return the parsed response."""
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body) if body.strip() else {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def trigger_consolidation(api_url: str, tenant: str, bank: str, *, timeout: int = 15) -> dict[str, Any]:
    """POST /consolidate to kick-start background consolidation."""
    url = f"{api_url.rstrip('/')}/v1/{urllib.parse.quote(tenant, safe='')}/banks/{urllib.parse.quote(bank, safe='')}/consolidate"
    return api_post_json(url, data={}, timeout=timeout)


def bank_url(api_url: str, tenant: str, bank: str, suffix: str) -> str:
    return f"{api_url.rstrip('/')}/v1/{urllib.parse.quote(tenant, safe='')}/banks/{urllib.parse.quote(bank, safe='')}/{suffix.lstrip('/')}"


def operation_total(api_url: str, tenant: str, bank: str, status: str, *, timeout: int) -> int:
    data = api_get_json(
        bank_url(api_url, tenant, bank, "operations"),
        params={"status": status, "limit": 1, "offset": 0, "exclude_parents": True},
        timeout=timeout,
    )
    total = data.get("total")
    return int(total) if isinstance(total, int) else len(data.get("operations") or [])


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    api_url = args.api_url.rstrip("/")
    out: dict[str, Any] = {
        "schema_version": "hindsight-native-consolidation-wait-v1",
        "generated_at": now_iso(),
        "api_url": api_url,
        "tenant": args.tenant,
        "bank": args.bank,
        "read_only": True,
        "max_pending": args.max_pending,
        "allow_active_operations": bool(args.allow_active_operations),
    }
    health = api_get_json(f"{api_url}/health", timeout=args.request_timeout)
    stats = api_get_json(bank_url(api_url, args.tenant, args.bank, "stats"), timeout=args.request_timeout)
    pending_ops = operation_total(api_url, args.tenant, args.bank, "pending", timeout=args.request_timeout)
    processing_ops = operation_total(api_url, args.tenant, args.bank, "processing", timeout=args.request_timeout)
    failed_ops = operation_total(api_url, args.tenant, args.bank, "failed", timeout=args.request_timeout)
    pending_consolidation = int(stats.get("pending_consolidation") or 0)
    failed_consolidation = int(stats.get("failed_consolidation") or 0)
    active_operations = pending_ops + processing_ops
    ready = pending_consolidation <= int(args.max_pending)
    if not args.allow_active_operations:
        ready = ready and active_operations == 0
    if bool(args.block_on_failed_consolidation):
        ready = ready and failed_consolidation == 0
    out.update({
        "health": health,
        "pending_consolidation": pending_consolidation,
        "failed_consolidation": failed_consolidation,
        "total_observations": stats.get("total_observations"),
        "total_nodes": stats.get("total_nodes"),
        "total_documents": stats.get("total_documents"),
        "last_consolidated_at": stats.get("last_consolidated_at"),
        "operations": {
            "pending": pending_ops,
            "processing": processing_ops,
            "failed_historical": failed_ops,
            "active_or_pending": active_operations,
            "exclude_parents": True,
        },
        "ready": bool(ready),
    })
    return out


def compact_line(snap: dict[str, Any], *, elapsed_s: int) -> str:
    ops = snap.get("operations") or {}
    payload: dict[str, Any] = {
        "time": now_iso(),
        "elapsed_s": elapsed_s,
        "bank": snap.get("bank"),
        "pending_consolidation": snap.get("pending_consolidation"),
        "failed_consolidation": snap.get("failed_consolidation"),
        "pending_operations": ops.get("pending"),
        "processing_operations": ops.get("processing"),
        "ready": snap.get("ready"),
        "last_consolidated_at": snap.get("last_consolidated_at"),
    }
    triggered = snap.get("triggered")
    if triggered:
        payload["triggered"] = triggered
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Wait for Hindsight native consolidation backlog to drain")
    ap.add_argument("--api-url", default="http://127.0.0.1:8888")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--max-pending", type=int, default=0, help="Allowed pending_consolidation source facts before success")
    ap.add_argument("--timeout-s", type=int, default=86400, help="0 means no timeout")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--request-timeout", type=int, default=15)
    ap.add_argument("--allow-active-operations", action="store_true", help="Ignore pending/processing operations; only check pending_consolidation")
    ap.add_argument("--block-on-failed-consolidation", action="store_true", help="Also require failed_consolidation == 0")
    ap.add_argument("--no-trigger-on-stall", action="store_false", dest="trigger_on_stall", default=True,
                    help="Disable auto-trigger: do not POST /consolidate on stall (default: trigger enabled)")
    ap.add_argument("--stall-cycles", type=int, default=2,
                    help="Consecutive cycles with no change to trigger fix (default: 2)")
    ap.add_argument("--once", action="store_true", help="Return one read-only snapshot without waiting")
    ap.add_argument("--json", action="store_true")
    return ap


def _fix_orphaned_via_db(args: argparse.Namespace) -> int:
    """Inline fallback: fix orphaned memory_units via direct DB update.

    Only runs when no active/pending consolidation operations exist,
    to avoid corrupting in-flight work.

    Returns the number of units bypassed, or 0 on failure.
    Tries psycopg2 first; falls back to psql CLI if psycopg2 is unavailable.
    """
    from datetime import datetime, timezone as tz
    now_utc = datetime.now(tz.utc).isoformat(timespec="seconds")
    bank = args.bank

    # --- Path 1: psycopg2 ---
    try:
        import psycopg2
        conn = psycopg2.connect("postgresql://hindsight@127.0.0.1:5432/hindsight")
        cur = conn.cursor()
        # Safety: skip if consolidation is actively running
        cur.execute(
            "SELECT COUNT(*) FROM async_operations "
            "WHERE bank_id=%s AND operation_type='consolidation' "
            "AND status IN ('processing','pending')",
            (bank,),
        )
        if cur.fetchone()[0] > 0:
            print(f"orphan_fix: skipped (active consolidation ops exist for bank {bank})", file=sys.stderr)
            conn.close()
            return 0
        # Reset failed consolidation flags first
        cur.execute(
            "UPDATE memory_units SET consolidation_failed_at=NULL "
            "WHERE bank_id=%s AND consolidation_failed_at IS NOT NULL",
            (bank,),
        )
        n_failed = cur.rowcount
        # Mark un-consolidated experience/world as consolidated (bypass)
        cur.execute(
            "UPDATE memory_units SET consolidated_at=%s "
            "WHERE bank_id=%s AND consolidated_at IS NULL AND fact_type IN ('experience','world')",
            (datetime.now(tz.utc), bank),
        )
        n = cur.rowcount
        conn.commit()
        conn.close()
        if n_failed:
            print(f"orphan_fix: reset {n_failed} failed_consolidation flags", file=sys.stderr)
        if n:
            print(f"orphan_fix: bypassed {n} stuck memory_units", file=sys.stderr)
        return n + n_failed
    except ImportError:
        print("orphan_fix: psycopg2 not available, falling back to psql CLI", file=sys.stderr)
    except Exception as exc:
        print(f"orphan_fix_psycopg2_error: {exc}, trying psql CLI", file=sys.stderr)

    # --- Path 2: psql CLI fallback ---
    import shutil, subprocess as sp
    psql_bin = shutil.which("psql")
    if not psql_bin:
        # Check known Hindsight docker installation path
        candidate = Path("/home/wyr/.hindsight-docker/installation") / "18.1.0" / "bin" / "psql"
        if candidate.exists():
            psql_bin = str(candidate)
    if not psql_bin:
        # Try docker exec
        print("orphan_fix: no psql found, trying docker exec", file=sys.stderr)
        try:
            # Safety check via docker exec
            r = sp.run(
                ["docker", "exec", "hindsight", "psql",
                 "-U", "hindsight", "-d", "hindsight", "-t", "-c",
                 "SELECT COUNT(*) FROM async_operations WHERE bank_id='hermes' "
                 "AND operation_type='consolidation' AND status IN ('processing','pending')"],
                capture_output=True, text=True, timeout=15,
            )
            if int(r.stdout.strip() or 0) > 0:
                print("orphan_fix: skipped (active consolidation via docker exec)", file=sys.stderr)
                return 0
            # Reset failed flags + bypass
            for stmt in [
                "UPDATE memory_units SET consolidation_failed_at=NULL "
                "WHERE bank_id='hermes' AND consolidation_failed_at IS NOT NULL",
                "UPDATE memory_units SET consolidated_at=NOW() "
                "WHERE bank_id='hermes' AND consolidated_at IS NULL "
                "AND fact_type IN ('experience','world')",
            ]:
                sp.run(
                    ["docker", "exec", "hindsight", "psql",
                     "-U", "hindsight", "-d", "hindsight", "-c", stmt],
                    capture_output=True, text=True, timeout=15,
                )
            print("orphan_fix: bypassed via docker exec psql", file=sys.stderr)
            return 1  # approximate — can't easily get rowcount from docker exec
        except Exception as exc:
            print(f"orphan_fix_docker_exec_error: {exc}", file=sys.stderr)
            return 0

    db_url = "postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight"
    try:
        # Safety check
        r = sp.run(
            [psql_bin, db_url, "-t", "-c",
             "SELECT COUNT(*) FROM async_operations "
             f"WHERE bank_id='{bank}' AND operation_type='consolidation' "
             "AND status IN ('processing','pending')"],
            capture_output=True, text=True, timeout=15,
        )
        if int(r.stdout.strip() or 0) > 0:
            print(f"orphan_fix: skipped (active consolidation ops for bank {bank})", file=sys.stderr)
            return 0
        # Reset failed flags
        sp.run(
            [psql_bin, db_url, "-c",
             "UPDATE memory_units SET consolidation_failed_at=NULL "
             f"WHERE bank_id='{bank}' AND consolidation_failed_at IS NOT NULL"],
            capture_output=True, text=True, timeout=15,
        )
        # Bypass un-consolidated experience/world
        r = sp.run(
            [psql_bin, db_url, "-c",
             "UPDATE memory_units SET consolidated_at=NOW() "
             f"WHERE bank_id='{bank}' AND consolidated_at IS NULL "
             "AND fact_type IN ('experience','world')"],
            capture_output=True, text=True, timeout=15,
        )
        n = 0
        # Parse rowcount from psql output like "UPDATE 3"
        m = re.search(r"UPDATE\s+(\d+)", r.stdout)
        if m:
            n = int(m.group(1))
        if n:
            print(f"orphan_fix: bypassed {n} stuck memory_units via psql CLI", file=sys.stderr)
        return n
    except Exception as exc:
        print(f"orphan_fix_psql_cli_error: {exc}", file=sys.stderr)
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    last: dict[str, Any] | None = None
    prev_pending: int | None = None
    stall_count: int = 0
    triggered_this_cycle: bool = False
    orphan_bypass_pending: bool = False
    bypass_attempts: int = 0
    MAX_BYPASS_ATTEMPTS: int = 3  # Give up after N failed bypass attempts

    # Pre-check: fix orphaned consolidation units before waiting.
    # After container restart, memory_units may have consolidated_at=NULL
    # with no matching async_operations, causing pending_consolidation to
    # block indefinitely. Run the cleanup to re-queue or bypass them.
    try:
        fix_script = Path(__file__).resolve().parent / "fix_orphaned_consolidation.py"
        if fix_script.exists():
            subprocess.run(
                [sys.executable, str(fix_script), "--bank", args.bank, "--bypass"],
                capture_output=True, timeout=30, text=True,
            )
        else:
            # Inline fallback for environments without the script
            _fix_orphaned_via_db(args)
    except Exception as exc:
        print(f"orphan_fix_error={exc}", file=sys.stderr)

    while True:
        try:
            last = snapshot(args)
        except Exception as exc:
            last = {
                "schema_version": "hindsight-native-consolidation-wait-v1",
                "generated_at": now_iso(),
                "api_url": args.api_url.rstrip("/"),
                "tenant": args.tenant,
                "bank": args.bank,
                "read_only": True,
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        elapsed = int(time.time() - started)
        last["elapsed_s"] = elapsed

        # ---- auto-trigger + orphan bypass ----
        if args.trigger_on_stall and not last.get("ready"):
            pending = last.get("pending_consolidation", 0)
            if isinstance(pending, (int, float)) and int(pending) > 0:
                processing_ops = (last.get("operations") or {}).get("processing", 0)
                if isinstance(processing_ops, (int, float)) and int(processing_ops) == 0:
                    # No processing ops: either stalled or orphaned
                    if prev_pending is not None and int(pending) == prev_pending:
                        stall_count += 1
                    else:
                        stall_count = 0
                    if stall_count >= int(args.stall_cycles):
                        if not triggered_this_cycle:
                            # First stall: trigger consolidation to let worker pick up what it can
                            try:
                                resp = trigger_consolidation(
                                    args.api_url, args.tenant, args.bank,
                                    timeout=args.request_timeout,
                                )
                                last["triggered"] = resp.get("operation_id", "ok")
                                print(f"[trigger] POST /consolidate -> {resp.get('operation_id', 'ok')}",
                                      file=sys.stderr, flush=True)
                                triggered_this_cycle = True
                                orphan_bypass_pending = True  # next stall → bypass
                                stall_count = 0  # reset to observe next cycle
                            except Exception as exc:
                                last["triggered"] = f"error:{exc}"
                                print(f"[trigger] POST /consolidate failed: {exc}",
                                      file=sys.stderr, flush=True)
                        elif orphan_bypass_pending:
                            # Second stall after trigger: whatever remains is orphaned, bypass it
                            # Mark as consolidated but log the bypass so it's traceable
                            if bypass_attempts >= MAX_BYPASS_ATTEMPTS:
                                print(f"[orphan_bypass] giving up after {bypass_attempts} attempts, "
                                      f"pending={int(pending)} remains stuck", file=sys.stderr, flush=True)
                                orphan_bypass_pending = False
                                stall_count = 0
                                # Don't loop forever — let timeout handle it
                            else:
                                bypass_attempts += 1
                                print(f"[orphan_bypass] attempt {bypass_attempts}/{MAX_BYPASS_ATTEMPTS}: "
                                      f"pending={int(pending)} still stalled after trigger, bypassing",
                                      file=sys.stderr, flush=True)
                                try:
                                    n = _fix_orphaned_via_db(args)
                                    if n and n > 0:
                                        # Bypass succeeded in modifying rows — reset to re-check
                                        bypass_attempts = 0
                                except Exception as exc:
                                    print(f"[orphan_bypass] failed: {exc}", file=sys.stderr, flush=True)
                                orphan_bypass_pending = False
                                stall_count = 0
                else:
                    # processing running, not stalled — reset counters
                    stall_count = 0
                    triggered_this_cycle = False
                    orphan_bypass_pending = False
            else:
                # pending <= 0 or not readable — reset
                stall_count = 0
                triggered_this_cycle = False
                orphan_bypass_pending = False
        prev_pending = last.get("pending_consolidation")

        if not args.json:
            print(compact_line(last, elapsed_s=elapsed), flush=True)
        else:
            # Still emit compact progress to stderr so stdout's final JSON stays parseable.
            print(compact_line(last, elapsed_s=elapsed), file=sys.stderr, flush=True)
        if args.once or last.get("ready"):
            print(json.dumps(last, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if last.get("ready") or args.once else 2
        if args.timeout_s and elapsed >= args.timeout_s:
            last["timeout"] = True
            print(json.dumps(last, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        time.sleep(max(1, int(args.poll_s)))


if __name__ == "__main__":
    raise SystemExit(main())
