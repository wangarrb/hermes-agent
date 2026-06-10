#!/usr/bin/env python3
"""Retain A-group candidates to production Hindsight bank after reset.

Reads scorer A-group manifest, rehydrates content from source session JSONs,
and submits to Hindsight production bank via native API.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import HindsightNativeClient, DEFAULT_API

RETAIN_CONFIRM = "retain-hindsight-session-manifest"


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def session_content_from_json(path: str) -> str:
    """Extract conversation text from Hermes session JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parts = []
    for msg in data.get("messages", []):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            parts.append(f"[{role}]\n{content}")
    return "\n\n---\n\n".join(parts)


def scorer_record_to_retain_item(record: dict[str, Any]) -> dict[str, Any] | None:
    source_path = record.get("source_json_path")
    if not source_path or not Path(source_path).exists():
        return None

    content = session_content_from_json(source_path)
    if not content:
        return None

    doc_id = record.get("document_id")
    if not doc_id:
        return None

    meta = {
        "scorer_route": record.get("scorer_route", ""),
        "scorer_topic": record.get("scorer_topic", ""),
        "score_mean_0_1": str(record.get("score_mean_0_1", "")),
        "score_total_0_20": str(record.get("score_total_0_20", "")),
        "value_classes": json.dumps(record.get("value_classes", []), ensure_ascii=False),
        "topic_group": record.get("topic_group", ""),
        "retainability_risk": record.get("retainability_risk", ""),
        "content_sha256": record.get("content_sha256", ""),
        "source_json_path": source_path,
        "candidate_id": record.get("candidate_id", ""),
    }

    item = {
        "content": content,
        "document_id": doc_id,
        "context": "hermes_session",
        "event_date": record.get("event_date"),
        "metadata": {k: str(v) for k, v in meta.items() if v},
        "tags": ["scorer:a_group", "route:next_repair"] + record.get("value_classes", []),
        "update_mode": "replace",
    }
    # Drop empty optionals
    return {k: v for k, v in item.items() if v not in (None, "", [], {})}


def run_retain(manifest_path: str, *, bank: str = "hermes", api: str = DEFAULT_API,
               dry_run: bool = True, confirm: str | None = None,
               batch_size: int = 3) -> dict[str, Any]:
    records = load_manifest(manifest_path)
    items = []
    errors = []
    for rec in records:
        item = scorer_record_to_retain_item(rec)
        if item:
            items.append(item)
        else:
            errors.append({"candidate_id": rec.get("candidate_id"), "document_id": rec.get("document_id")})

    result = {
        "manifest": str(manifest_path),
        "bank": bank,
        "dry_run": dry_run,
        "total_records": len(records),
        "valid_items": len(items),
        "errors": errors,
        "batch_size": batch_size,
    }

    if dry_run:
        result["would_submit_batches"] = (len(items) + batch_size - 1) // batch_size
        result["first_item_preview"] = items[0]["content"][:500] if items else None
        return result

    if confirm != RETAIN_CONFIRM:
        raise RuntimeError(f"Real retain requires confirm={RETAIN_CONFIRM}")

    client = HindsightNativeClient(api=api, bank=bank, timeout=120)
    operation_ids = []
    responses = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        resp = client.retain_items(batch, async_mode=True)
        responses.append(resp)
        op_ids = resp.get("operation_ids") or []
        if isinstance(op_ids, list):
            operation_ids.extend(op_ids)
        single_op = resp.get("operation_id")
        if isinstance(single_op, str):
            operation_ids.append(single_op)

    result.update({
        "submitted_items": len(items),
        "responses": responses,
        "operation_ids": list(dict.fromkeys(operation_ids)),
    })
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--confirm")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = run_retain(
        args.manifest,
        bank=args.bank,
        api=args.api,
        dry_run=not args.execute,
        confirm=args.confirm,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
