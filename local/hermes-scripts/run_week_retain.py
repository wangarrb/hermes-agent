#!/usr/bin/env python3
"""Retain week manifest to Hindsight production bank."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "/home/wyr/.hermes/scripts")
from hindsight_native_client import HindsightNativeClient

MANIFEST_PATH = "/home/wyr/.hermes/hindsight/runs/week-04-09-04-16-production-20260509-1549.jsonl"
API_URL = "http://127.0.0.1:8888"
BANK = "hermes"
BATCH_SIZE = 5

def load_manifest(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def record_to_item(record):
    item = {
        "content": record.get("content", ""),
        "document_id": record.get("document_id"),
        "context": "hermes_session",
        "event_date": record.get("event_date") or (record.get("metadata") or {}).get("started_at"),
        "metadata": {
            "source_json_path": (record.get("metadata") or {}).get("json_path", ""),
            "action": record.get("action", ""),
            "reason": record.get("reason", ""),
            "tags": json.dumps(record.get("tags", []), ensure_ascii=False),
        },
        "tags": record.get("tags", []) or [],
        "update_mode": "replace",
    }
    return {k: v for k, v in item.items() if k in {"content", "document_id"} or v}

records = load_manifest(MANIFEST_PATH)
print(f"Loaded {len(records)} records from manifest")

# Prepare items
items = [record_to_item(r) for r in records]
valid_items = [i for i in items if i.get("content") and i.get("document_id")]
print(f"Valid items: {len(valid_items)}")

# Submit
client = HindsightNativeClient(api=API_URL, bank=BANK, timeout=120)
operation_ids = []
submitted = 0

for i in range(0, len(valid_items), BATCH_SIZE):
    batch = valid_items[i:i+BATCH_SIZE]
    print(f"Submitting batch {i//BATCH_SIZE + 1}/{(len(valid_items)+BATCH_SIZE-1)//BATCH_SIZE} ({len(batch)} items)")
    resp = client.retain_items(batch, async_mode=True)
    op_id = resp.get("operation_id")
    if op_id:
        operation_ids.append(op_id)
        submitted += len(batch)
    print(f"  operation_id: {op_id}")

print(f"\nSubmitted {submitted} items, {len(operation_ids)} operations")

# Wait for completion
print("Waiting for operations to complete...")
deadline = time.time() + 1800  # 30 min
while operation_ids and time.time() < deadline:
    pending = []
    for op_id in operation_ids:
        ops = list(client.iter_operations(max_items=100))
        for op in ops:
            if op.get("operation_id") == op_id or op.get("id") == op_id:
                status = op.get("status")
                if status == "completed":
                    print(f"  {op_id}: completed")
                elif status in ("failed", "error", "cancelled"):
                    print(f"  {op_id}: FAILED - {op.get('error_message', '')[:100]}")
                else:
                    pending.append(op_id)
                break
        else:
            pending.append(op_id)  # not found yet

    operation_ids = pending
    if operation_ids:
        print(f"  Pending: {len(operation_ids)}, waiting 30s...")
        time.sleep(30)

# Final stats
stats = client.request("GET", "/v1/default/banks/hermes/stats")
print(f"\nFinal stats:")
print(f"  Documents: {stats.get('total_documents', 0)}")
print(f"  Nodes: {stats.get('total_nodes', 0)}")
print(f"  Pending: {stats.get('pending_operations', 0)}")
print(f"  Failed: {stats.get('failed_operations', 0)}")