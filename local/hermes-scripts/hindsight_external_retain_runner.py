#!/usr/bin/env python3
"""Submit reviewed manual external-import manifests to Hindsight.

Default mode is dry-run. Real retain requires explicit confirm token and is kept
separate from the daily/weekly Hindsight pipeline.
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
import hindsight_external_manifest as external_manifest  # noqa: E402

RETAIN_CONFIRM = "retain-hindsight-external-manifest"
DEFAULT_TARGET_BANK = "hermes"
DEFAULT_SUBMIT_STATE = Path.home() / ".hermes" / "hindsight" / "external_import" / "submit_state.json"
DEFAULT_EXTERNAL_BANK_CONFIG = {
    "retain_chunk_size": 8000,
    "retain_extraction_mode": "custom",
    "retain_custom_instructions": (
        "Extract durable user/project facts, decisions, results, preferences, stable environment facts. "
        "Skip tool logs, file listings, raw command output, process chatter, greetings. "
        "For external_conversation records, keep only high-signal durable facts (usually 3-5 per chunk). "
        "For external_markdown_artifact records, preserve the Markdown structure provided in the content header: "
        "report_date/title/artifact_type/section_path/item_index. Do not merge different numbered items or sections. "
        "Keep concrete metrics, project names, versions, chips, platforms, paths, dates, and acceptance/failure details."
    ),
    "enable_observations": True,
    "recall_max_tokens": 4096,
    "recall_chunks_max_tokens": 4096,
    "consolidation_llm_batch_size": 8,
    "consolidation_max_memories_per_round": 64,
    "consolidation_source_facts_max_tokens": 4096,
    "consolidation_source_facts_max_tokens_per_observation": 256,
}


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
        return {"schema_version": "external-submit-state-v1", "documents": {}}
    p = Path(path).expanduser()
    if not p.exists():
        return {"schema_version": "external-submit-state-v1", "documents": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"schema_version": "external-submit-state-v1", "documents": {}}
    data.setdefault("schema_version", "external-submit-state-v1")
    data.setdefault("documents", {})
    return data


def save_submit_state(path: str | Path, state: dict[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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


def record_incremental_key(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata") or {}
    return {
        "content_sha256": meta.get("content_sha256"),
        "full_content_sha256": meta.get("full_content_sha256"),
        "source_kind": meta.get("source_kind"),
        "source_path": meta.get("source_path"),
        "source_file_sha256": meta.get("source_file_sha256"),
        "source_mtime_ns": meta.get("source_mtime_ns"),
        "source_size_bytes": meta.get("source_size_bytes"),
        "external_updated_at": meta.get("conversation_updated_at") or meta.get("created_at"),
        "message_count": meta.get("message_count"),
        "adapter_version": meta.get("adapter_version"),
        "cleaning_version": meta.get("cleaning_version"),
        "tag_rule_version": meta.get("tag_rule_version"),
        "schema_version": meta.get("schema_version"),
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
    return bool(key.get("content_sha256") and prev.get("content_sha256") == key.get("content_sha256"))


def update_submit_state_for_items(state: dict[str, Any], records: list[dict[str, Any]], *, manifest_path: str | Path, bank: str) -> dict[str, Any]:
    state.setdefault("schema_version", "external-submit-state-v1")
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
    source_path = meta.get("source_path")
    if not source_path:
        raise ValueError(f"record {record.get('document_id')} omitted content but has no metadata.source_path")
    bank = record.get("bank_target") or meta.get("bank_target") or "external_import_smoke"
    if source_kind == "chat_memo_txt":
        candidates = external_manifest.records_from_chat_memo_file(source_path, bank_target=str(bank))
    elif source_kind == "openclaw_lcm":
        candidates, _ = external_manifest.records_from_openclaw_lcm(source_path, bank_target=str(bank), min_age_seconds=0)
    elif source_kind == "markdown_artifact_md":
        candidates = external_manifest.records_from_markdown_file(source_path, bank_target=str(bank), record_granularity="all")
    else:
        raise ValueError(f"unsupported external source_kind={source_kind!r}")
    doc_id = record.get("document_id")
    for candidate in candidates:
        if candidate.get("document_id") == doc_id:
            merged = dict(candidate)
            for key in ["action", "reason", "tags", "metadata", "context", "update_mode", "bank_target", "event_date"]:
                if key in record:
                    merged[key] = record[key]
            return merged
    raise ValueError(f"could not rehydrate document_id={doc_id} from {source_path}")


def record_to_memory_item(record: dict[str, Any]) -> dict[str, Any]:
    rec = rehydrate_record(record)
    item = {
        "content": rec.get("content") or "",
        "document_id": rec.get("document_id"),
        "context": rec.get("context") or "external_conversation",
        "event_date": rec.get("event_date") or (rec.get("metadata") or {}).get("segment_started_at") or (rec.get("metadata") or {}).get("created_at"),
        "metadata": normalize_metadata_for_hindsight(rec.get("metadata") or {}),
        "tags": rec.get("tags") or [],
        "update_mode": rec.get("update_mode") or "replace",
    }
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
        item = record_to_memory_item(record)
        if not item.get("content") or not item.get("document_id"):
            skipped["invalid_item"] += 1
            continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected, skipped


def ensure_bank_exists(client: Any, bank: str) -> dict[str, Any]:
    """Create the target bank if missing; no-op when it already exists."""
    banks = client.request("GET", "/v1/default/banks")
    for item in banks.get("banks") or []:
        if isinstance(item, dict) and str(item.get("bank_id") or "") == bank:
            return {"bank": bank, "created": False}
    client.request("PUT", f"/v1/default/banks/{bank}", payload={"name": bank})
    return {"bank": bank, "created": True}


def patch_external_bank_config(client: Any, *, enable_observations: bool = True) -> dict[str, Any]:
    """Set external-import defaults: concise retain + native observations enabled."""
    updates = dict(DEFAULT_EXTERNAL_BANK_CONFIG)
    updates["enable_observations"] = bool(enable_observations)
    current = client.get_config()
    supported_keys = set((current.get("config") or {}).keys())
    if supported_keys:
        updates = {k: v for k, v in updates.items() if k in supported_keys}
    response = client.patch_config(updates)
    cfg = response.get("config") or {}
    return {
        "bank": getattr(client, "bank", None),
        "updates": updates,
        "effective": {
            "enable_observations": cfg.get("enable_observations"),
            "retain_extraction_mode": cfg.get("retain_extraction_mode"),
            "retain_chunk_size": cfg.get("retain_chunk_size"),
            "consolidation_max_memories_per_round": cfg.get("consolidation_max_memories_per_round"),
            "consolidation_llm_batch_size": cfg.get("consolidation_llm_batch_size"),
            "consolidation_source_facts_max_tokens": cfg.get("consolidation_source_facts_max_tokens"),
            "consolidation_source_facts_max_tokens_per_observation": cfg.get("consolidation_source_facts_max_tokens_per_observation"),
        },
    }


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


def _operation_total(client: Any, status: str) -> int:
    page = client.list_operations(status=status, limit=1, offset=0, exclude_parents=True)
    total = page.get("total")
    if isinstance(total, int):
        return total
    items = page.get("operations") or page.get("items") or []
    return len(items) if isinstance(items, list) else 0


def consolidation_snapshot(client: Any, *, max_pending: int = 0, allow_active_operations: bool = False, block_on_failed_consolidation: bool = True) -> dict[str, Any]:
    stats = client.stats()
    pending_ops = _operation_total(client, "pending")
    processing_ops = _operation_total(client, "processing")
    active_ops = pending_ops + processing_ops
    pending_consolidation = int(stats.get("pending_consolidation") or 0)
    failed_consolidation = int(stats.get("failed_consolidation") or 0)
    ready = pending_consolidation <= int(max_pending)
    if not allow_active_operations:
        ready = ready and active_ops == 0
    if block_on_failed_consolidation:
        ready = ready and failed_consolidation == 0
    return {
        "bank": getattr(client, "bank", None),
        "pending_consolidation": pending_consolidation,
        "failed_consolidation": failed_consolidation,
        "total_observations": stats.get("total_observations"),
        "total_nodes": stats.get("total_nodes"),
        "total_documents": stats.get("total_documents"),
        "last_consolidated_at": stats.get("last_consolidated_at"),
        "operations": {
            "pending": pending_ops,
            "processing": processing_ops,
            "active_or_pending": active_ops,
            "exclude_parents": True,
        },
        "ready": bool(ready),
    }


def wait_for_consolidation(
    client: Any,
    *,
    timeout_s: int = 86400,
    poll_s: float = 60.0,
    max_pending: int = 0,
    trigger_on_stall: bool = True,
    stall_cycles: int = 2,
    block_on_failed_consolidation: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    """Wait for native source-fact consolidation/observation generation to drain.

    External import must not stop at retain completion.  This watches both
    pending_consolidation and child async operations, and can POST /consolidate
    when the queue is stalled with no processing operation.
    """
    started = time.time()
    deadline = None if not timeout_s else started + max(1, timeout_s)
    prev_pending: int | None = None
    stall_count = 0
    triggered_this_cycle = False
    while True:
        last = consolidation_snapshot(
            client,
            max_pending=max_pending,
            block_on_failed_consolidation=block_on_failed_consolidation,
        )
        last["elapsed_s"] = int(time.time() - started)
        if progress:
            compact = {
                "elapsed_s": last["elapsed_s"],
                "bank": last.get("bank"),
                "pending_consolidation": last.get("pending_consolidation"),
                "failed_consolidation": last.get("failed_consolidation"),
                "pending_operations": (last.get("operations") or {}).get("pending"),
                "processing_operations": (last.get("operations") or {}).get("processing"),
                "ready": last.get("ready"),
                "last_consolidated_at": last.get("last_consolidated_at"),
            }
            print(json.dumps(compact, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        if last.get("ready"):
            return last

        pending = int(last.get("pending_consolidation") or 0)
        processing_ops = int((last.get("operations") or {}).get("processing") or 0)
        if trigger_on_stall and pending > int(max_pending) and processing_ops == 0:
            if prev_pending is not None and pending == prev_pending:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= int(stall_cycles) and not triggered_this_cycle:
                try:
                    triggered = client.trigger_consolidation()
                    last["triggered"] = triggered
                    if progress:
                        print(f"[trigger] POST /consolidate -> {triggered.get('operation_id', 'ok')}", file=sys.stderr, flush=True)
                    triggered_this_cycle = True
                    stall_count = 0
                except Exception as exc:
                    last["trigger_error"] = f"{type(exc).__name__}: {exc}"
                    if progress:
                        print(f"[trigger] POST /consolidate failed: {exc}", file=sys.stderr, flush=True)
        else:
            stall_count = 0
            triggered_this_cycle = False
        prev_pending = pending

        if deadline is not None and time.time() >= deadline:
            raise RetainOperationFailed(f"consolidation timeout: {last}")
        time.sleep(max(1.0, float(poll_s)))


def batch_items(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = external_manifest.summarize_records(records)
    by_source = summary.get("by_source") or {}
    by_record_kind: Counter = Counter()
    for record in records:
        meta = record.get("metadata") or {}
        by_record_kind[str(meta.get("record_kind") or "unknown")] += 1
    summary["by_record_kind"] = dict(sorted(by_record_kind.items()))
    summary["by_source"] = dict(sorted(by_source.items()))
    summary["retain_policy"] = {
        "external_markdown_artifact_instruction": True,
        "external_conversation_instruction": True,
        "manual_only": True,
        "daily_pipeline_integrated": False,
    }
    return summary


def run_manifest(path: str | Path, *, client: Any | None = None, bank: str = DEFAULT_TARGET_BANK, api: str = DEFAULT_API, action: str = "production", dry_run: bool = True, confirm: str | None = None, batch_size: int = 5, limit: int | None = None, submit_state_path: str | Path | None = None, wait: bool = True, wait_timeout_s: int = 300, poll_s: float = 5.0, enable_observations: bool = True, wait_consolidation: bool = True, consolidation_timeout_s: int = 86400, consolidation_poll_s: float = 60.0, consolidation_max_pending: int = 0, trigger_consolidation_on_stall: bool = True) -> dict[str, Any]:
    records = load_manifest(path)
    submit_state = load_submit_state(submit_state_path)
    selected_records, skipped = prepare_retain_records(records, action=action, submit_state=submit_state, limit=limit, bank=bank)
    items = [record_to_memory_item(record) for record in selected_records]
    result: dict[str, Any] = {
        "manifest": str(path),
        "bank": bank,
        "action_filter": action,
        "dry_run": dry_run,
        "manual_only": True,
        "daily_pipeline_integrated": False,
        "enable_observations": bool(enable_observations),
        "post_retain_consolidation_integrated": bool(wait_consolidation),
        "wait_consolidation": bool(wait_consolidation),
        "total_records": len(records),
        "would_submit_items": len(items),
        "submitted_items": 0,
        "skipped": dict(skipped),
        "batch_size": batch_size,
        "submit_state_path": str(submit_state_path) if submit_state_path else None,
        "responses": [],
        "operation_ids": [],
        "waited_for_operations": False,
        "waited_for_consolidation": False,
    }
    if dry_run:
        return result
    if confirm != RETAIN_CONFIRM:
        raise UnsafeRetainOperation(f"submit retain requires confirm={RETAIN_CONFIRM}")
    client = client or HindsightNativeClient(api=api, bank=bank, timeout=120)
    result["bank_create"] = ensure_bank_exists(client, bank)
    result["bank_config"] = patch_external_bank_config(client, enable_observations=enable_observations)
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
        # Always update submit_state after successful submission.
        # Even in --no-wait mode, the records are queued in Hindsight and
        # update_mode=replace makes them idempotent. Not updating causes
        # full re-submission on next run, wasting LLM budget.
        update_submit_state_for_items(submit_state, selected_records, manifest_path=path, bank=bank)
        save_submit_state(submit_state_path, submit_state)
        result["submit_state_updated"] = True
    else:
        result["submit_state_updated"] = False
    if wait_consolidation and enable_observations and wait and result["submitted_items"] > 0:
        result["consolidation_trigger"] = client.trigger_consolidation()
        result["consolidation"] = wait_for_consolidation(
            client,
            timeout_s=consolidation_timeout_s,
            poll_s=consolidation_poll_s,
            max_pending=consolidation_max_pending,
            trigger_on_stall=trigger_consolidation_on_stall,
            progress=True,
        )
        result["waited_for_consolidation"] = True
    elif wait_consolidation and not wait:
        result["consolidation_wait_skipped_reason"] = "retain_async_operations_not_waited"
    elif wait_consolidation and not enable_observations:
        result["consolidation_wait_skipped_reason"] = "observations_disabled"
    elif wait_consolidation and result["submitted_items"] <= 0:
        result["consolidation_wait_skipped_reason"] = "no_submitted_items"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submit production records from a manual external manifest to Hindsight; dry-run by default.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--bank", default=DEFAULT_TARGET_BANK)
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--action", default="production")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--submit-state", type=Path, default=DEFAULT_SUBMIT_STATE, help="Persistent successful-submit state used for incremental replace/re-retain. Dry-run never writes it.")
    ap.add_argument("--ignore-submit-state", action="store_true", help="Do not skip unchanged records based on submit state.")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--no-wait", action="store_true", help="Do not wait for async retain operations. When used, submit-state will not be updated because success is not verified.")
    ap.add_argument("--enable-observations", dest="enable_observations", action="store_true", default=True, help="Patch target bank to enable native observations before retain (default: enabled).")
    ap.add_argument("--no-enable-observations", dest="enable_observations", action="store_false", help="Patch target bank with enable_observations=false for low-cost/debug imports.")
    ap.add_argument("--wait-timeout-s", type=int, default=1800)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--wait-consolidation", dest="wait_consolidation", action="store_true", default=True, help="After retain operations complete, trigger/wait for native consolidation so observations are generated (default).")
    ap.add_argument("--no-wait-consolidation", dest="wait_consolidation", action="store_false", help="Stop after retain operation completion; leaves observation drain to another process.")
    ap.add_argument("--consolidation-timeout-s", type=int, default=86400, help="Timeout for post-retain native consolidation drain; 0 means no timeout.")
    ap.add_argument("--consolidation-poll-s", type=float, default=60.0)
    ap.add_argument("--consolidation-max-pending", type=int, default=0)
    ap.add_argument("--no-trigger-consolidation-on-stall", dest="trigger_consolidation_on_stall", action="store_false", default=True)
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
        enable_observations=args.enable_observations,
        wait_consolidation=args.wait_consolidation,
        consolidation_timeout_s=args.consolidation_timeout_s,
        consolidation_poll_s=args.consolidation_poll_s,
        consolidation_max_pending=args.consolidation_max_pending,
        trigger_consolidation_on_stall=args.trigger_consolidation_on_stall,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
