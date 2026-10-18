#!/usr/bin/env python3
"""Cancel pending Hindsight operations for a specific bank.

用于清理旧 bank 的 pending 队列，避免继续消耗 LLM 配额。
只调用公开 DELETE /operations/{id}；不会删除已完成写入的 documents/nodes。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import DELETE_OPERATION_CONFIRM, HindsightNativeClient

API = "http://localhost:8888"
LIMIT = 100
SLEEP_SECONDS = 0.03


def get_pending(bank: str, *, client: HindsightNativeClient | None = None):
    client = client or HindsightNativeClient(api=API, bank=bank)
    data = client.list_operations(status="pending", limit=LIMIT, offset=0)
    return data.get("total", 0), data.get("operations", data.get("items", []))


def cancel_one(bank: str, op_id: str, *, client: HindsightNativeClient | None = None, dry_run: bool = True, confirm: str | None = None):
    client = client or HindsightNativeClient(api=API, bank=bank)
    if dry_run:
        return True, 0, f"dry-run: would DELETE /operations/{op_id} via official Hindsight API"
    data = client.delete_operation(op_id, dry_run=False, confirm=confirm)
    return True, 200, json.dumps(data, ensure_ascii=False)[:300]


def main():
    ap = argparse.ArgumentParser(description="Cancel pending Hindsight operations through the official operations API")
    ap.add_argument("bank_id")
    ap.add_argument("max_cancel", nargs="?", type=int)
    ap.add_argument("--api", default=API)
    ap.add_argument("--dry-run", action="store_true", default=True, help="Preview pending operation cancellations without deleting (default)")
    ap.add_argument("--execute", action="store_true", help="Actually cancel pending operations; requires --confirm-cancel")
    ap.add_argument("--confirm-cancel", help=f"Required for --execute: {DELETE_OPERATION_CONFIRM}")
    args = ap.parse_args()

    bank = args.bank_id
    max_cancel = args.max_cancel
    dry_run = not bool(args.execute)
    if args.execute and args.confirm_cancel != DELETE_OPERATION_CONFIRM:
        print(f"Refusing to cancel operations without --confirm-cancel {DELETE_OPERATION_CONFIRM}", file=sys.stderr)
        raise SystemExit(2)
    started = datetime.now().isoformat(timespec="seconds")
    out_path = Path.home() / ".hermes" / "hindsight" / f"cancel_pending_{bank}_{started.replace(':','')}.json"
    client = HindsightNativeClient(api=args.api, bank=bank)

    result = {
        "bank": bank,
        "api": args.api,
        "started_at": started,
        "finished_at": None,
        "dry_run": dry_run,
        "official_api": True,
        "required_confirm": DELETE_OPERATION_CONFIRM,
        "cancelled": [],
        "would_cancel": [],
        "errors": [],
        "snapshots": [],
    }

    total_cancelled = 0
    while True:
        total, ops = get_pending(bank, client=client)
        result["snapshots"].append({"time": datetime.now().isoformat(timespec="seconds"), "pending_total": total, "batch": len(ops)})
        print(f"pending_total={total} batch={len(ops)} {'would_cancel' if dry_run else 'cancelled'}={total_cancelled}", flush=True)
        if not ops:
            break

        for op in ops:
            if max_cancel is not None and total_cancelled >= max_cancel:
                result["finished_at"] = datetime.now().isoformat(timespec="seconds")
                out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Reached max_cancel={max_cancel}; log={out_path}")
                return
            op_id = op.get("id")
            if not op_id:
                continue
            ok, status, body = cancel_one(bank, op_id, client=client, dry_run=dry_run, confirm=args.confirm_cancel)
            rec = {"id": op_id, "task_type": op.get("task_type"), "created_at": op.get("created_at"), "status_code": status}
            if ok:
                if dry_run:
                    result["would_cancel"].append(rec)
                else:
                    result["cancelled"].append(rec)
                total_cancelled += 1
            else:
                rec["body"] = body
                result["errors"].append(rec)
            if total_cancelled % 100 == 0:
                out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"checkpoint {'would_cancel' if dry_run else 'cancelled'}={total_cancelled} errors={len(result['errors'])}", flush=True)
            time.sleep(SLEEP_SECONDS)
        if dry_run:
            break

    result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE bank={bank} dry_run={dry_run} would_cancel={len(result['would_cancel'])} cancelled={len(result['cancelled'])} errors={len(result['errors'])} log={out_path}")


if __name__ == "__main__":
    main()
