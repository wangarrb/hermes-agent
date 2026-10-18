#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/wyr/.hermes/scripts")

from hindsight_native_client import HindsightNativeClient
import time

client = HindsightNativeClient(api="http://127.0.0.1:8888", bank="hermes", timeout=30)
operation_ids = [
    "e38d8eb7-b0af-4483-a813-28533cad0fda",
    "df266012-e48a-4874-8727-da62331c1524",
    "64f84f57-4116-46f1-91b4-5e50735ec2fd",
    "3aafe5c0-5711-4b0e-b17a-c5f4804045a3",
    "0d50b03b-9443-4a78-8411-2992aff962cb",
]

wanted = set(operation_ids)
deadline = time.time() + 1800  # 30 min timeout
poll_s = 5

while True:
    seen = {}
    for op in client.iter_operations(max_items=1000):
        op_id = op.get("operation_id") or op.get("id")
        if op_id in wanted:
            seen[op_id] = op

    statuses = {op_id: (seen.get(op_id) or {}).get("status") for op_id in wanted}
    failed = {op_id: status for op_id, status in statuses.items() if status in {"failed", "cancelled", "error"}}
    completed = {op_id: status for op_id, status in statuses.items() if status == "completed"}

    print(f"[{time.strftime('%H:%M:%S')}] completed={len(completed)}/{len(wanted)} failed={len(failed)} pending={len(wanted)-len(completed)-len(failed)}")

    if failed:
        print("FAILED operations:")
        for op_id, status in failed.items():
            print(f"  {op_id}: {status}")
        sys.exit(1)

    if wanted.issubset(seen) and all(status == "completed" for status in statuses.values()):
        print("All operations completed successfully!")
        for op_id, op in seen.items():
            print(f"  {op_id}: {op.get('status')} items={op.get('items_count')} type={op.get('operation_type')}")
        break

    if time.time() >= deadline:
        print("TIMEOUT")
        for op_id, status in statuses.items():
            print(f"  {op_id}: {status}")
        sys.exit(1)

    time.sleep(poll_s)
