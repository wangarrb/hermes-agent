#!/usr/bin/env python3
"""Submit reviewed session manifest production records to Hindsight.

Default mode is dry-run. Real retain requires an explicit confirm token and should
be run only after provider/queue preflight. This runner rebuilds content from the
source JSON when the manifest is lean (content omitted), so the manifest can stay
small and non-sensitive.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import DEFAULT_API, HindsightNativeClient  # noqa: E402
import hindsight_session_manifest as session_manifest  # noqa: E402

RETAIN_CONFIRM = "retain-hindsight-session-manifest"


class UnsafeRetainOperation(RuntimeError):
    pass


class RetainOperationFailed(RuntimeError):
    pass


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_submit_state(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"schema_version": "session-submit-state-v1", "documents": {}}
    p = Path(path)
    if not p.exists():
        return {"schema_version": "session-submit-state-v1", "documents": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"schema_version": "session-submit-state-v1", "documents": {}}
    data.setdefault("schema_version", "session-submit-state-v1")
    data.setdefault("documents", {})
    return data


def save_submit_state(path: str | Path, state: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def record_incremental_key(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata") or {}
    return {
        "content_sha256": meta.get("content_sha256"),
        "full_content_sha256": meta.get("full_content_sha256"),
        "source_mtime_ns": meta.get("source_mtime_ns"),
        "source_size_bytes": meta.get("source_size_bytes"),
        "source_file_sha256": meta.get("source_file_sha256"),
        "session_last_updated": meta.get("session_last_updated") or meta.get("last_updated"),
        "event_date": record.get("event_date") or meta.get("event_date") or meta.get("started_at"),
        "schema_version": meta.get("schema_version"),
        "cleaning_version": meta.get("cleaning_version"),
        "candidate_filter_version": meta.get("candidate_filter_version"),
    }


def submit_state_document_key(document_id: str | None, bank: str | None = None) -> str:
    doc_id = str(document_id or "")
    if bank:
        return f"{bank}::{doc_id}"
    return doc_id


def submit_state_entry(record: dict[str, Any], submit_state: dict[str, Any] | None, *, bank: str | None = None) -> dict[str, Any] | None:
    if not submit_state:
        return None
    doc_id = record.get("document_id")
    docs = submit_state.get("documents") or {}
    if not isinstance(docs, dict):
        return None
    # New state is bank-scoped so a smoke bank cannot mask production/candidate
    # re-retain, and vice versa. Legacy unscoped keys are still read for
    # backward compatibility.
    for key in [submit_state_document_key(doc_id, bank), submit_state_document_key(doc_id, None)]:
        prev = docs.get(key)
        if isinstance(prev, dict):
            return prev
    return None


def is_record_unchanged(record: dict[str, Any], submit_state: dict[str, Any] | None, *, bank: str | None = None) -> bool:
    prev = submit_state_entry(record, submit_state, bank=bank)
    if not isinstance(prev, dict):
        return False
    prev_bank = prev.get("bank")
    if bank and prev_bank and prev_bank != bank:
        return False
    key = record_incremental_key(record)
    # content hash is authoritative; mtime/size are retained for audit and as a
    # fast human signal that the source changed.
    if key.get("content_sha256") and prev.get("content_sha256") == key.get("content_sha256"):
        return True
    return False


def update_submit_state_for_items(state: dict[str, Any], records: list[dict[str, Any]], *, manifest_path: str | Path, bank: str) -> dict[str, Any]:
    state.setdefault("schema_version", "session-submit-state-v1")
    docs = state.setdefault("documents", {})
    now = datetime.now(timezone.utc).isoformat()
    for record in records:
        doc_id = record.get("document_id")
        if not doc_id:
            continue
        entry = record_incremental_key(record)
        entry.update({
            "document_id": doc_id,
            "bank": bank,
            "last_submitted_at": now,
            "last_submit_manifest": str(manifest_path),
            "action": record.get("action"),
            "reason": record.get("reason"),
        })
        docs[submit_state_document_key(doc_id, bank)] = entry
    state["updated_at"] = now
    state["document_count"] = len(docs)
    return state


def rehydrate_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("content"):
        return record
    meta = record.get("metadata") or {}
    source_kind = str(meta.get("source_kind") or "")
    bank_target = record.get("bank_target") or meta.get("bank_target") or session_manifest.DEFAULT_BANK_TARGET
    doc_id = record.get("document_id")

    def merge_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        merged = dict(candidate)
        # Keep reviewed manifest decisions/tags/scopes rather than recomputing
        # blindly, but use freshly rebuilt content from source.
        for key in ["action", "reason", "tags", "observation_scopes", "metadata", "context", "update_mode", "bank_target", "event_date"]:
            if key in record:
                merged[key] = record[key]
        return merged

    if source_kind == "codex_rollout_jsonl":
        source_path = meta.get("jsonl_path")
        if not source_path:
            raise ValueError(f"record {doc_id} omitted content but has no metadata.jsonl_path")
        candidates = session_manifest.records_from_codex_rollout_file(
            source_path,
            bank_target=bank_target,
        )
        for candidate in candidates:
            if candidate.get("document_id") == doc_id:
                return merge_candidate(candidate)
        raise ValueError(f"could not rehydrate document_id={doc_id} from {source_path}")

    if source_kind == "codex_markdown_artifact":
        source_path = meta.get("source_path")
        if not source_path:
            raise ValueError(f"record {doc_id} omitted content but has no metadata.source_path")
        candidates = session_manifest.record_from_codex_markdown_artifact(
            source_path,
            bank_target=bank_target,
            producer=str(meta.get("producer") or "codex"),
            source_rollout_path=meta.get("source_rollout_path"),
            session_id=meta.get("session_id"),
        )
        for candidate in candidates:
            if candidate.get("document_id") == doc_id:
                return merge_candidate(candidate)
        raise ValueError(f"could not rehydrate document_id={doc_id} from {source_path}")

    source_path = meta.get("json_path")
    if not source_path:
        raise ValueError(f"record {record.get('document_id')} omitted content but has no metadata.json_path")
    candidates = session_manifest.records_from_json_file(
        source_path,
        bank_target=bank_target,
        source_profile=str(meta.get("source_profile") or "default"),
    )
    for candidate in candidates:
        if candidate.get("document_id") == doc_id:
            return merge_candidate(candidate)
    raise ValueError(f"could not rehydrate document_id={doc_id} from {source_path}")


def stringify_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_metadata_for_hindsight(metadata: dict[str, Any]) -> dict[str, str]:
    return {str(k): stringify_metadata_value(v) for k, v in (metadata or {}).items() if v is not None}


def record_to_memory_item(record: dict[str, Any]) -> dict[str, Any]:
    rec = rehydrate_record(record)
    item = {
        "content": rec.get("content") or "",
        "document_id": rec.get("document_id"),
        "context": rec.get("context") or "hermes_session",
        "event_date": rec.get("event_date") or (rec.get("metadata") or {}).get("event_date") or (rec.get("metadata") or {}).get("started_at"),
        "metadata": normalize_metadata_for_hindsight(rec.get("metadata") or {}),
        "tags": rec.get("tags") or [],
        "observation_scopes": rec.get("observation_scopes") or [],
        "update_mode": rec.get("update_mode") or "replace",
    }
    # Drop empty optional fields conservatively except content/document_id.
    return {k: v for k, v in item.items() if k in {"content", "document_id"} or v not in (None, [], {})}


def prepare_retain_records(records: list[dict[str, Any]], *, action: str = "production", submit_state: dict[str, Any] | None = None, limit: int | None = None, bank: str | None = None) -> tuple[list[dict[str, Any]], Counter]:
    selected: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for record in records:
        rec_action = record.get("action") or "unknown"
        if rec_action != action:
            skipped[rec_action] += 1
            continue
        if is_record_unchanged(record, submit_state, bank=bank):
            skipped["unchanged"] += 1
            continue
        # Validate rehydration now so invalid records are counted before submit.
        item = record_to_memory_item(record)
        if not item.get("content") or not item.get("document_id"):
            skipped["invalid_item"] += 1
            continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected, skipped


def prepare_retain_items(records: list[dict[str, Any]], *, action: str = "production", submit_state: dict[str, Any] | None = None, limit: int | None = None, bank: str | None = None) -> tuple[list[dict[str, Any]], Counter]:
    selected, skipped = prepare_retain_records(records, action=action, submit_state=submit_state, limit=limit, bank=bank)
    return [record_to_memory_item(record) for record in selected], skipped


def extract_operation_ids(response: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    op_id = response.get("operation_id")
    if isinstance(op_id, str) and op_id:
        ids.append(op_id)
    op_ids = response.get("operation_ids")
    if isinstance(op_ids, list):
        ids.extend([x for x in op_ids if isinstance(x, str) and x])
    return list(dict.fromkeys(ids))


def operation_id_from_record(op: dict[str, Any]) -> str | None:
    for key in ["operation_id", "id"]:
        val = op.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def wait_for_operation_ids(client: Any, operation_ids: list[str], *, timeout_s: int = 300, poll_s: float = 5.0) -> dict[str, dict[str, Any]]:
    if not operation_ids:
        return {}
    wanted = set(operation_ids)
    deadline = time.time() + max(1, timeout_s)
    seen: dict[str, dict[str, Any]] = {}
    while True:
        for op in client.iter_operations(max_items=1000):
            op_id = operation_id_from_record(op)
            if op_id in wanted:
                seen[op_id] = op
        statuses = {op_id: (seen.get(op_id) or {}).get("status") for op_id in wanted}
        failed = {op_id: status for op_id, status in statuses.items() if status in {"failed", "cancelled", "error"}}
        if failed:
            raise RetainOperationFailed(f"retain async operation failed: {failed}")
        if wanted.issubset(seen) and all(status == "completed" for status in statuses.values()):
            return seen
        if time.time() >= deadline:
            raise RetainOperationFailed(f"retain async operation timeout: {statuses}")
        time.sleep(poll_s)


def batch_items(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def run_manifest(path: str | Path, *, client: Any | None = None, bank: str = "hermes_v3", api: str = DEFAULT_API, action: str = "production", dry_run: bool = True, confirm: str | None = None, batch_size: int = 5, limit: int | None = None, submit_state_path: str | Path | None = None, wait: bool = True, wait_timeout_s: int = 300, poll_s: float = 5.0) -> dict[str, Any]:
    records = load_manifest(path)
    submit_state = load_submit_state(submit_state_path)
    selected_records, skipped = prepare_retain_records(records, action=action, submit_state=submit_state, limit=limit, bank=bank)
    items = [record_to_memory_item(record) for record in selected_records]
    result: dict[str, Any] = {
        "manifest": str(path),
        "bank": bank,
        "action_filter": action,
        "dry_run": dry_run,
        "total_records": len(records),
        "would_submit_items": len(items),
        "submitted_items": 0,
        "skipped": dict(skipped),
        "batch_size": batch_size,
        "submit_state_path": str(submit_state_path) if submit_state_path else None,
        "responses": [],
        "operation_ids": [],
        "waited_for_operations": False,
    }
    if dry_run:
        return result
    if confirm != RETAIN_CONFIRM:
        raise UnsafeRetainOperation(f"submit retain requires confirm={RETAIN_CONFIRM}")
    client = client or HindsightNativeClient(api=api, bank=bank, timeout=120)
    operation_ids: list[str] = []
    for batch in batch_items(items, max(1, batch_size)):
        response = client.retain_items(batch, async_mode=True)
        result["responses"].append(response)
        operation_ids.extend(extract_operation_ids(response))
        result["submitted_items"] += len(batch)
    result["operation_ids"] = list(dict.fromkeys(operation_ids))
    if wait and operation_ids:
        result["operations"] = wait_for_operation_ids(client, result["operation_ids"], timeout_s=wait_timeout_s, poll_s=poll_s)
        result["waited_for_operations"] = True
    elif operation_ids:
        result["submit_state_pending_reason"] = "async_operations_not_waited"
    if submit_state_path and selected_records:
        if operation_ids and not result.get("waited_for_operations"):
            result["submit_state_updated"] = False
        else:
            update_submit_state_for_items(submit_state, selected_records, manifest_path=path, bank=bank)
            save_submit_state(submit_state_path, submit_state)
            result["submit_state_updated"] = True
    else:
        result["submit_state_updated"] = False
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submit production records from a session manifest to Hindsight; dry-run by default.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--bank", default="hermes_v3")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--action", default="production")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--submit-state", type=Path, default=Path.home() / ".hermes" / "hindsight" / "session_ingest" / "submit_state.json", help="Persistent successful-submit state used for incremental replace/re-retain. Dry-run never writes it.")
    ap.add_argument("--ignore-submit-state", action="store_true", help="Do not skip unchanged records based on submit state.")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--no-wait", action="store_true", help="Do not wait for async retain operations. When used, submit-state will not be updated because success is not verified.")
    ap.add_argument("--wait-timeout-s", type=int, default=1800, help="Seconds to wait for async retain operations; MiniMax + observations can exceed 600s")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--confirm")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    result = run_manifest(
        args.manifest,
        bank=args.bank,
        api=args.api,
        action=args.action,
        dry_run=not args.execute,
        confirm=args.confirm,
        batch_size=args.batch_size,
        limit=args.limit,
        submit_state_path=None if args.ignore_submit_state else args.submit_state,
        wait=not args.no_wait,
        wait_timeout_s=args.wait_timeout_s,
        poll_s=args.poll_s,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
