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
import sys
import time
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    last: dict[str, Any] | None = None
    prev_pending: int | None = None
    stall_count: int = 0
    triggered_this_cycle: bool = False
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

        # ---- auto-trigger: detect stalled consolidation ----
        if args.trigger_on_stall and not last.get("ready"):
            pending = last.get("pending_consolidation", 0)
            if isinstance(pending, (int, float)) and int(pending) > 0:
                processing_ops = (last.get("operations") or {}).get("processing", 0)
                if isinstance(processing_ops, (int, float)) and int(processing_ops) == 0:
                    if prev_pending is not None and int(pending) == prev_pending:
                        stall_count += 1
                    else:
                        stall_count = 0
                    if stall_count >= int(args.stall_cycles) and not triggered_this_cycle:
                        try:
                            resp = trigger_consolidation(
                                args.api_url, args.tenant, args.bank,
                                timeout=args.request_timeout,
                            )
                            last["triggered"] = resp.get("operation_id", "ok")
                            print(f"[trigger] POST /consolidate -> {resp.get('operation_id', 'ok')}",
                                  file=sys.stderr, flush=True)
                            triggered_this_cycle = True
                            stall_count = 0  # reset so we don't re-trigger right away
                        except Exception as exc:
                            last["triggered"] = f"error:{exc}"
                            print(f"[trigger] POST /consolidate failed: {exc}",
                                  file=sys.stderr, flush=True)
                else:
                    # processing running, not stalled — reset counters
                    stall_count = 0
                    triggered_this_cycle = False
            else:
                # pending <= 0 or not readable — reset
                stall_count = 0
                triggered_this_cycle = False
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
